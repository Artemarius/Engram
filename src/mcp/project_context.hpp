#pragma once

/// @file project_context.hpp
/// @brief Per-project state bundle for multi-project Engram.
///
/// Each ProjectContext owns the vector index, chunk metadata, session store,
/// and file watcher for a single project directory.  Non-copyable and
/// non-movable because it owns a mutex.  Always stored behind unique_ptr.

#include "chunker/chunker.hpp"
#include "index/hnsw_index.hpp"
#include "session/session_store.hpp"
#include "watcher/win_watcher.hpp"

#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace engram {

/// Bundles all per-project state into one heap-allocated object.
///
/// A single engram-mcp process may manage multiple ProjectContext instances,
/// one for each indexed codebase.  The shared Embedder lives outside and is
/// referenced through ToolContext.
struct ProjectContext {
    /// Display name (typically the last path component of project_root).
    std::string name;

    /// Absolute path to the project root directory.
    std::filesystem::path project_root;

    /// Absolute path to the data directory (index, chunks.json, sessions).
    std::filesystem::path data_dir;

    /// Mapping from chunk_id to Chunk metadata.
    std::unordered_map<std::string, Chunk> chunk_map;

    /// HNSW vector index for this project's embeddings.
    HnswIndex vector_index;

    /// Guards chunk_map and vector_index against concurrent watcher writes.
    std::mutex index_mutex;

    /// On-disk paths for persistence.
    std::filesystem::path index_path;
    std::filesystem::path chunks_path;

    /// Persistent session memory store for this project.
    std::unique_ptr<SessionStore> session_store;

    /// Filesystem watcher for incremental re-indexing.
    WinFileWatcher watcher;

    /// Construct with the given embedding dimension for the vector index.
    explicit ProjectContext(size_t embedding_dim)
        : vector_index(embedding_dim) {}

    // Non-copyable, non-movable (owns mutex).
    ProjectContext(const ProjectContext&) = delete;
    ProjectContext& operator=(const ProjectContext&) = delete;
    ProjectContext(ProjectContext&&) = delete;
    ProjectContext& operator=(ProjectContext&&) = delete;
};

} // namespace engram
