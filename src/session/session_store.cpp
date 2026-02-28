/// @file session_store.cpp
/// @brief Implementation of SessionStore — persistent session summary storage.

#include "session_store.hpp"

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>

#include <spdlog/spdlog.h>

namespace engram {

// ---------------------------------------------------------------------------
// JSON serialization
// ---------------------------------------------------------------------------

void to_json(nlohmann::json& j, const SessionSummary& s) {
    j = nlohmann::json{
        {"id",            s.id},
        {"timestamp",     s.timestamp},
        {"summary",       s.summary},
        {"key_files",     s.key_files},
        {"key_decisions", s.key_decisions}
    };
}

void from_json(const nlohmann::json& j, SessionSummary& s) {
    j.at("id").get_to(s.id);
    j.at("timestamp").get_to(s.timestamp);
    j.at("summary").get_to(s.summary);
    j.at("key_files").get_to(s.key_files);
    j.at("key_decisions").get_to(s.key_decisions);
}

// ---------------------------------------------------------------------------
// SessionStore
// ---------------------------------------------------------------------------

SessionStore::SessionStore(const std::filesystem::path& storage_dir)
    : storage_dir_(storage_dir)
{
    std::error_code ec;
    std::filesystem::create_directories(storage_dir_, ec);
    if (ec) {
        spdlog::error("SessionStore: failed to create directory '{}': {}",
                      storage_dir_.generic_string(), ec.message());
    } else {
        spdlog::debug("SessionStore: using directory '{}'",
                      storage_dir_.generic_string());
    }
}

bool SessionStore::save(SessionSummary& session) {
    // Generate an ID if the caller didn't provide one.
    if (session.id.empty()) {
        session.id = generate_id();
    }

    // Fill in the timestamp if empty.
    if (session.timestamp.empty()) {
        // Produce an ISO 8601 local-time timestamp.
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        std::tm tm_buf{};
#ifdef _MSC_VER
        localtime_s(&tm_buf, &time_t_now);
#else
        localtime_r(&time_t_now, &tm_buf);
#endif
        std::ostringstream oss;
        oss << std::put_time(&tm_buf, "%Y-%m-%dT%H:%M:%S");
        session.timestamp = oss.str();
    }

    const auto path = file_path_for(session.id);
    nlohmann::json j = session;

    std::ofstream ofs(path, std::ios::out | std::ios::trunc);
    if (!ofs.is_open()) {
        spdlog::error("SessionStore::save: cannot open '{}' for writing",
                      path.generic_string());
        return false;
    }

    ofs << j.dump(2);
    if (!ofs.good()) {
        spdlog::error("SessionStore::save: write error for '{}'",
                      path.generic_string());
        return false;
    }

    spdlog::info("SessionStore: saved session '{}' to '{}'",
                 session.id, path.generic_string());
    return true;
}

std::vector<SessionSummary> SessionStore::load_all() const {
    std::vector<SessionSummary> results;

    std::error_code ec;
    if (!std::filesystem::exists(storage_dir_, ec)) {
        spdlog::warn("SessionStore::load_all: directory '{}' does not exist",
                     storage_dir_.generic_string());
        return results;
    }

    for (const auto& entry : std::filesystem::directory_iterator(storage_dir_, ec)) {
        if (ec) {
            spdlog::warn("SessionStore::load_all: error iterating directory: {}",
                         ec.message());
            break;
        }

        if (!entry.is_regular_file()) continue;

        const auto& p = entry.path();
        // Only look at session_*.json files.
        const auto filename = p.filename().string();
        if (filename.size() < 13 ||
            filename.substr(0, 8) != "session_" ||
            p.extension() != ".json") {
            continue;
        }

        std::ifstream ifs(p);
        if (!ifs.is_open()) {
            spdlog::warn("SessionStore::load_all: cannot open '{}'",
                         p.generic_string());
            continue;
        }

        try {
            nlohmann::json j = nlohmann::json::parse(ifs);
            results.push_back(j.get<SessionSummary>());
        } catch (const nlohmann::json::exception& ex) {
            spdlog::warn("SessionStore::load_all: failed to parse '{}': {}",
                         p.generic_string(), ex.what());
        }
    }

    spdlog::debug("SessionStore::load_all: loaded {} sessions", results.size());
    return results;
}

std::optional<SessionSummary> SessionStore::load(const std::string& id) const {
    const auto path = file_path_for(id);

    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        spdlog::debug("SessionStore::load: '{}' does not exist",
                      path.generic_string());
        return std::nullopt;
    }

    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        spdlog::warn("SessionStore::load: cannot open '{}'",
                      path.generic_string());
        return std::nullopt;
    }

    try {
        nlohmann::json j = nlohmann::json::parse(ifs);
        return j.get<SessionSummary>();
    } catch (const nlohmann::json::exception& ex) {
        spdlog::warn("SessionStore::load: failed to parse '{}': {}",
                     path.generic_string(), ex.what());
        return std::nullopt;
    }
}

bool SessionStore::remove(const std::string& id) {
    const auto path = file_path_for(id);

    std::error_code ec;
    if (!std::filesystem::remove(path, ec)) {
        if (ec) {
            spdlog::warn("SessionStore::remove: error removing '{}': {}",
                         path.generic_string(), ec.message());
        } else {
            spdlog::debug("SessionStore::remove: '{}' does not exist",
                          path.generic_string());
        }
        return false;
    }

    spdlog::info("SessionStore: removed session '{}'", id);
    return true;
}

std::string SessionStore::generate_id() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf{};
#ifdef _MSC_VER
    localtime_s(&tm_buf, &time_t_now);
#else
    localtime_r(&time_t_now, &tm_buf);
#endif

    char buf[16];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm_buf);
    return std::string(buf);
}

std::filesystem::path SessionStore::file_path_for(const std::string& id) const {
    return storage_dir_ / ("session_" + id + ".json");
}

} // namespace engram
