#pragma once
/// @file tools.hpp
/// @brief Tool definitions for the Engram MCP server.
///
/// Declares the ToolContext struct that bundles all backend dependencies
/// and the registration function that wires tool handlers into an McpServer
/// instance.  Handlers use the context to perform real vector search,
/// symbol lookup, context retrieval, and session memory operations.

#include "mcp/mcp_server.hpp"
#include "mcp/project_context.hpp"

#include "chunker/chunker.hpp"
#include "embedder/embedder.hpp"
#include "index/vector_index.hpp"
#include "session/session_store.hpp"

#include <memory>
#include <string>
#include <vector>

namespace engram::mcp {

/// Bundles all backend dependencies that tool handlers need at runtime.
///
/// With multi-project support, per-project state (index, chunks, session
/// store, mutex) lives inside ProjectContext.  ToolContext provides access
/// to the shared embedder and the vector of project contexts.
///
/// Pointer members may be nullptr when the corresponding subsystem is not
/// available (e.g. embedder is null when no ONNX model is loaded).  Handlers
/// must check for nullptr before use and return a graceful error message
/// rather than crashing.
struct ToolContext {
    /// Embedding model for converting query text into vectors.
    /// Shared across all projects.  May be nullptr if no model is configured.
    Embedder* embedder = nullptr;

    /// All loaded project contexts.  Each project has its own index,
    /// chunk store, session store, and mutex.
    /// May be nullptr if no projects are configured.
    std::vector<std::unique_ptr<engram::ProjectContext>>* projects = nullptr;

    /// Primary project's session store (first project in the list).
    /// Used by save_session_summary.  May be nullptr.
    SessionStore* session_store = nullptr;
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
