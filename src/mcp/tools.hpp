#pragma once
/// @file tools.hpp
/// @brief Tool definitions for the Engram MCP server.
///
/// Declares the ToolContext struct that bundles all backend dependencies
/// and the registration function that wires tool handlers into an McpServer
/// instance.  Handlers use the context to perform real vector search,
/// symbol lookup, context retrieval, and session memory operations.

#include "mcp/mcp_server.hpp"

#include "chunker/chunker.hpp"
#include "embedder/embedder.hpp"
#include "index/vector_index.hpp"
#include "session/session_store.hpp"

#include <mutex>
#include <string>
#include <unordered_map>

namespace engram::mcp {

/// Bundles all backend dependencies that tool handlers need at runtime.
///
/// Pointer members may be nullptr when the corresponding subsystem is not
/// available (e.g. embedder is null when no ONNX model is loaded).  Handlers
/// must check for nullptr before use and return a graceful error message
/// rather than crashing.
struct ToolContext {
    /// Embedding model for converting query text into vectors.
    /// May be nullptr if no model is configured.
    Embedder* embedder = nullptr;

    /// Vector similarity index for nearest-neighbor search.
    /// May be nullptr if the index has not been built.
    VectorIndex* index = nullptr;

    /// Persistent session memory store.
    /// May be nullptr if session storage is disabled.
    SessionStore* session_store = nullptr;

    /// Mapping from chunk_id to Chunk metadata.
    /// Must not be nullptr when search/context tools are used, but handlers
    /// gracefully handle the nullptr case.
    std::unordered_map<std::string, Chunk>* chunk_store = nullptr;

    /// Absolute path to the project root, used for resolving relative file
    /// paths in tool results.
    std::string project_root;

    /// Optional mutex protecting shared mutable state (chunk_store, index).
    /// When non-null, tool handlers lock this before accessing chunk_store.
    /// The watcher callback also locks this when modifying chunk_store.
    std::mutex* shared_mutex = nullptr;
};

/// Register all Engram tools with the given server.
///
/// The @p context is captured by reference; it must outlive the server.
///
/// Tools registered:
///   - search_code         -- semantic code search (requires embedder + index)
///   - search_symbol       -- symbol lookup by name/kind (chunk_store only)
///   - get_context         -- retrieve surrounding context for a file+line
///   - get_session_memory  -- recall previous session summaries
///   - save_session_summary -- persist a session summary
void register_all_tools(McpServer& server, ToolContext& context);

} // namespace engram::mcp
