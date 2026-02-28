#pragma once

/// @file session_store.hpp
/// @brief Session memory storage for Engram.
///
/// Provides persistent storage for session summaries.  Each session is
/// serialized as a JSON file in a configurable directory.  The SessionStore
/// class handles creation, loading, and deletion of session records.

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace engram {

/// A summary of a single Claude Code session.
///
/// Created at the end of a session and persisted to disk so that future
/// sessions can retrieve relevant past context via semantic search.
struct SessionSummary {
    /// Unique identifier for the session (format: YYYYMMDD_HHMMSS).
    std::string id;

    /// ISO 8601 timestamp of when the session ended.
    std::string timestamp;

    /// Free-text summary provided by Claude at session end.
    std::string summary;

    /// File paths that were important during the session.
    std::vector<std::string> key_files;

    /// Key architectural or design decisions made during the session.
    std::vector<std::string> key_decisions;
};

/// nlohmann/json serialization for SessionSummary.
void to_json(nlohmann::json& j, const SessionSummary& s);

/// nlohmann/json deserialization for SessionSummary.
void from_json(const nlohmann::json& j, SessionSummary& s);

/// @class SessionStore
/// @brief Manages persistent storage of session summaries on disk.
///
/// Each session is stored as a separate JSON file named `session_{id}.json`
/// inside the configured storage directory.  The directory is created
/// automatically if it does not exist.
class SessionStore {
public:
    /// @brief Construct a SessionStore that persists data to @p storage_dir.
    ///
    /// The directory is created (including parents) if it does not exist.
    ///
    /// @param storage_dir  Path to the directory for session JSON files.
    explicit SessionStore(const std::filesystem::path& storage_dir);

    /// @brief Save a session summary to disk.
    ///
    /// If the summary's `id` field is empty, a new ID is generated from the
    /// current timestamp.  Overwrites any existing file with the same ID.
    ///
    /// @param session  The session summary to persist.
    /// @return true on success, false if the file could not be written.
    bool save(SessionSummary& session);

    /// @brief Load all session summaries from the storage directory.
    ///
    /// Files that cannot be parsed are silently skipped (logged at warn level).
    ///
    /// @return A vector of all successfully loaded session summaries.
    std::vector<SessionSummary> load_all() const;

    /// @brief Load a specific session by its ID.
    ///
    /// @param id  The session identifier (e.g. "20260228_143022").
    /// @return The session summary, or std::nullopt if not found or unreadable.
    std::optional<SessionSummary> load(const std::string& id) const;

    /// @brief Delete a session file from disk.
    ///
    /// @param id  The session identifier to remove.
    /// @return true if the file was deleted, false if it did not exist or
    ///         could not be removed.
    bool remove(const std::string& id);

    /// @brief Generate a new session ID from the current system time.
    ///
    /// Format: YYYYMMDD_HHMMSS (e.g. "20260228_143022").
    ///
    /// @return The generated ID string.
    static std::string generate_id();

    /// @brief Returns the storage directory path.
    const std::filesystem::path& storage_directory() const { return storage_dir_; }

private:
    std::filesystem::path storage_dir_;

    /// Build the full file path for a given session ID.
    std::filesystem::path file_path_for(const std::string& id) const;
};

} // namespace engram
