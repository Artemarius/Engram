#pragma once
/// @file tools.hpp
/// @brief Tool definitions for the Engram MCP server.
///
/// Declares the stub tool handlers and the registration function that wires
/// them into an McpServer instance.  Each handler currently returns a
/// placeholder response; real implementations will be connected once the
/// chunker, embedder, and vector index modules are ready.

#include "mcp/mcp_server.hpp"

namespace engram::mcp {

/// Register all Engram tools with the given server.
///
/// Tools registered:
///   - search_code       — semantic code search
///   - search_symbol     — symbol lookup by name/kind
///   - get_context       — retrieve surrounding context for a file+line
///   - get_session_memory  — recall previous session summaries
///   - save_session_summary — persist a session summary
void register_all_tools(McpServer& server);

} // namespace engram::mcp
