/// @file chunk_store.cpp
/// @brief Implementation of chunk metadata persistence (save/load as JSON).

#include "chunk_store.hpp"

#include <fstream>

#include <spdlog/spdlog.h>

namespace engram {

// -------------------------------------------------------------------------
// JSON serialization for Chunk
// -------------------------------------------------------------------------

void to_json(nlohmann::json& j, const Chunk& c) {
    j = nlohmann::json{
        {"chunk_id",    c.chunk_id},
        {"file_path",   c.file_path.generic_string()},
        {"start_line",  c.start_line},
        {"end_line",    c.end_line},
        {"language",    c.language},
        {"symbol_name", c.symbol_name},
        {"source_text", c.source_text}
    };
}

void from_json(const nlohmann::json& j, Chunk& c) {
    j.at("chunk_id").get_to(c.chunk_id);

    std::string path_str;
    j.at("file_path").get_to(path_str);
    c.file_path = std::filesystem::path(path_str);

    j.at("start_line").get_to(c.start_line);
    j.at("end_line").get_to(c.end_line);
    j.at("language").get_to(c.language);
    j.at("symbol_name").get_to(c.symbol_name);
    j.at("source_text").get_to(c.source_text);
}

// -------------------------------------------------------------------------
// Chunk store persistence
// -------------------------------------------------------------------------

bool save_chunks(const std::filesystem::path& path,
                 const std::unordered_map<std::string, Chunk>& chunks)
{
    // Serialize the map as a JSON object: { "chunk_id": { ... }, ... }
    nlohmann::json j = nlohmann::json::object();
    for (const auto& [id, chunk] : chunks) {
        j[id] = chunk;
    }

    // Write to a temporary file first, then rename for atomicity.
    auto tmp_path = path;
    tmp_path += ".tmp";

    std::ofstream ofs(tmp_path, std::ios::out | std::ios::trunc);
    if (!ofs.is_open()) {
        spdlog::error("save_chunks: cannot open '{}' for writing",
                      tmp_path.generic_string());
        return false;
    }

    ofs << j.dump(2);
    if (!ofs.good()) {
        spdlog::error("save_chunks: write error for '{}'",
                      tmp_path.generic_string());
        ofs.close();
        // Clean up the partial file.
        std::error_code ec;
        std::filesystem::remove(tmp_path, ec);
        return false;
    }
    ofs.close();

    // Rename tmp -> final.  On Windows std::filesystem::rename will fail if
    // the destination exists, so remove it first.
    std::error_code ec;
    std::filesystem::remove(path, ec);  // OK if it doesn't exist.
    std::filesystem::rename(tmp_path, path, ec);
    if (ec) {
        spdlog::error("save_chunks: failed to rename '{}' -> '{}': {}",
                      tmp_path.generic_string(), path.generic_string(),
                      ec.message());
        // Try to clean up the temp file.
        std::filesystem::remove(tmp_path, ec);
        return false;
    }

    spdlog::info("save_chunks: saved {} chunks to '{}'",
                 chunks.size(), path.generic_string());
    return true;
}

bool load_chunks(const std::filesystem::path& path,
                 std::unordered_map<std::string, Chunk>& chunks)
{
    chunks.clear();

    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        spdlog::debug("load_chunks: '{}' does not exist", path.generic_string());
        return false;
    }

    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        spdlog::error("load_chunks: cannot open '{}'", path.generic_string());
        return false;
    }

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(ifs);
    } catch (const nlohmann::json::exception& ex) {
        spdlog::error("load_chunks: JSON parse error in '{}': {}",
                      path.generic_string(), ex.what());
        return false;
    }

    if (!j.is_object()) {
        spdlog::error("load_chunks: expected JSON object in '{}'",
                      path.generic_string());
        return false;
    }

    size_t loaded = 0;
    size_t skipped = 0;
    for (auto it = j.begin(); it != j.end(); ++it) {
        try {
            Chunk chunk = it.value().get<Chunk>();
            chunks[it.key()] = std::move(chunk);
            loaded++;
        } catch (const nlohmann::json::exception& ex) {
            spdlog::warn("load_chunks: skipping malformed chunk '{}': {}",
                         it.key(), ex.what());
            skipped++;
        }
    }

    spdlog::info("load_chunks: loaded {} chunks from '{}' ({} skipped)",
                 loaded, path.generic_string(), skipped);
    return loaded > 0 || j.empty();
}

} // namespace engram
