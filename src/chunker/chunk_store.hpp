#pragma once

/// @file chunk_store.hpp
/// @brief Disk persistence for the chunk metadata store.
///
/// Provides functions to serialize and deserialize the chunk metadata map
/// (chunk_id -> Chunk) as a flat JSON file.  This keeps the chunk store in
/// sync with the HNSW vector index across restarts, avoiding orphaned
/// vectors caused by chunk ID mismatches after re-chunking.

#include "chunker/chunker.hpp"

#include <filesystem>
#include <string>
#include <unordered_map>

#include <nlohmann/json.hpp>

namespace engram {

// -------------------------------------------------------------------------
// JSON serialization for the Chunk struct
// -------------------------------------------------------------------------

/// Serialize a Chunk to nlohmann::json.
void to_json(nlohmann::json& j, const Chunk& c);

/// Deserialize a Chunk from nlohmann::json.
void from_json(const nlohmann::json& j, Chunk& c);

// -------------------------------------------------------------------------
// Chunk store persistence
// -------------------------------------------------------------------------

/// @brief Save the chunk metadata map to a JSON file.
///
/// Writes the full map as a JSON object keyed by chunk_id.  The file is
/// written atomically (to a temporary file first, then renamed) to avoid
/// corruption if the process is interrupted mid-write.
///
/// @param path    Destination file path (e.g. data_dir / "chunks.json").
/// @param chunks  The chunk metadata map to serialize.
/// @return true on success, false on I/O error.
bool save_chunks(const std::filesystem::path& path,
                 const std::unordered_map<std::string, Chunk>& chunks);

/// @brief Load the chunk metadata map from a JSON file.
///
/// Replaces the contents of @p chunks with the data from the file.
/// On failure (file missing, parse error, etc.) @p chunks is left empty.
///
/// @param path    Source file path (e.g. data_dir / "chunks.json").
/// @param chunks  Output map to populate.
/// @return true on success, false if the file could not be loaded or parsed.
bool load_chunks(const std::filesystem::path& path,
                 std::unordered_map<std::string, Chunk>& chunks);

} // namespace engram
