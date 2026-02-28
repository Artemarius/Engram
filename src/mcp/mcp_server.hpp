#pragma once
/// @file mcp_server.hpp
/// @brief MCP server — JSON-RPC 2.0 over stdio.
///
/// The server reads JSON-RPC messages from stdin, dispatches them to
/// registered handlers, and writes responses to stdout.  All diagnostic
/// output goes to stderr via spdlog.

#include <atomic>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "mcp/protocol.hpp"

namespace engram::mcp {

/// Signature for a tool handler callback.
///
/// Receives the "arguments" object from a `tools/call` request and returns
/// a JSON value that will be wrapped in the MCP content array.
using ToolHandler = std::function<nlohmann::json(const nlohmann::json& arguments)>;

/// A registered tool: definition metadata + runtime handler.
struct RegisteredTool {
    ToolDefinition definition;
    ToolHandler    handler;
};

/// MCP server that communicates over stdin / stdout using JSON-RPC 2.0.
///
/// Typical usage:
/// @code
///   engram::mcp::McpServer server;
///   register_all_tools(server);   // from tools.hpp
///   server.run();                 // blocks until stdin EOF or stop()
/// @endcode
class McpServer {
public:
    McpServer();
    ~McpServer() = default;

    // Non-copyable, non-movable (owns the stdio streams conceptually).
    McpServer(const McpServer&)            = delete;
    McpServer& operator=(const McpServer&) = delete;
    McpServer(McpServer&&)                 = delete;
    McpServer& operator=(McpServer&&)      = delete;

    // -----------------------------------------------------------------
    // Tool registration
    // -----------------------------------------------------------------

    /// Register a tool that the MCP client can call.
    ///
    /// @param name         Machine-readable tool name (e.g. "search_code").
    /// @param description  Human-readable explanation shown to the LLM.
    /// @param input_schema JSON Schema describing the tool's input.
    /// @param handler      Callback invoked when the tool is called.
    void register_tool(const std::string& name,
                       const std::string& description,
                       const nlohmann::json& input_schema,
                       ToolHandler handler);

    /// Return the list of all registered tool definitions.
    const std::vector<RegisteredTool>& tools() const noexcept;

    // -----------------------------------------------------------------
    // Server lifecycle
    // -----------------------------------------------------------------

    /// Enter the main read-dispatch-write loop.  Blocks until stdin reaches
    /// EOF or stop() is called from another thread.
    void run();

    /// Ask the server to exit its run loop at the next opportunity.
    void stop() noexcept;

    /// Return true while the server is running.
    bool running() const noexcept;

    // -----------------------------------------------------------------
    // Message handling (public for testability)
    // -----------------------------------------------------------------

    /// Process a single parsed JSON-RPC message and return the response
    /// (or std::nullopt for notifications that need no reply).
    std::optional<nlohmann::json> handle_message(const nlohmann::json& msg);

private:
    /// Dispatch table entry: method name -> handler returning optional JSON.
    using MethodHandler = std::function<std::optional<nlohmann::json>(
                              const nlohmann::json& id,
                              const nlohmann::json& params)>;

    /// Built-in method handlers.
    std::optional<nlohmann::json> handle_initialize(const nlohmann::json& id,
                                                    const nlohmann::json& params);
    std::optional<nlohmann::json> handle_initialized(const nlohmann::json& id,
                                                     const nlohmann::json& params);
    std::optional<nlohmann::json> handle_tools_list(const nlohmann::json& id,
                                                    const nlohmann::json& params);
    std::optional<nlohmann::json> handle_tools_call(const nlohmann::json& id,
                                                    const nlohmann::json& params);

    /// Write a JSON-RPC response to stdout with proper framing.
    void send_response(const nlohmann::json& response);

    /// Read a single complete message from stdin.
    /// Returns std::nullopt on EOF.
    std::optional<std::string> read_message();

    std::vector<RegisteredTool>                          tools_;
    std::unordered_map<std::string, std::size_t>         tool_lookup_;  ///< name -> index into tools_
    std::unordered_map<std::string, MethodHandler>       dispatch_;
    std::atomic<bool>                                    running_{false};
};

} // namespace engram::mcp
