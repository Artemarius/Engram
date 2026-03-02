/// @file mcp_server.cpp
/// @brief MCP server implementation — JSON-RPC 2.0 over stdio.

#include "mcp/mcp_server.hpp"

#include <cstdio>
#include <iostream>
#include <string>

#include <spdlog/spdlog.h>

#ifdef _WIN32
#   include <fcntl.h>
#   include <io.h>
#   define WIN32_LEAN_AND_MEAN
#   include <windows.h>
#endif

namespace engram::mcp {

// =========================================================================
// Construction
// =========================================================================

McpServer::McpServer() {
#ifdef _WIN32
    // Set stdin to binary mode to prevent MSVC runtime from translating \r\n.
    // We also accept Content-Length framed messages on read, which need
    // byte-exact reads.
    _setmode(_fileno(stdin),  _O_BINARY);
#endif

    // Register built-in protocol methods.
    dispatch_["initialize"] = [this](const nlohmann::json& id,
                                     const nlohmann::json& params) {
        return handle_initialize(id, params);
    };
    dispatch_["notifications/initialized"] = [this](const nlohmann::json& id,
                                                    const nlohmann::json& params) {
        return handle_initialized(id, params);
    };
    dispatch_["tools/list"] = [this](const nlohmann::json& id,
                                     const nlohmann::json& params) {
        return handle_tools_list(id, params);
    };
    dispatch_["tools/call"] = [this](const nlohmann::json& id,
                                     const nlohmann::json& params) {
        return handle_tools_call(id, params);
    };
}

// =========================================================================
// Tool registration
// =========================================================================

void McpServer::register_tool(const std::string& name,
                              const std::string& description,
                              const nlohmann::json& input_schema,
                              ToolHandler handler) {
    tool_lookup_[name] = tools_.size();
    tools_.push_back(RegisteredTool{
        ToolDefinition{name, description, input_schema},
        std::move(handler)
    });
    spdlog::debug("registered tool: {}", name);
}

const std::vector<RegisteredTool>& McpServer::tools() const noexcept {
    return tools_;
}

// =========================================================================
// Server lifecycle
// =========================================================================

void McpServer::run() {
    running_ = true;
    spdlog::info("MCP server entering main loop");

    while (running_) {
        auto raw = read_message();
        if (!raw) {
            spdlog::info("stdin EOF — shutting down");
            break;
        }

        spdlog::debug("recv: {}", *raw);

        auto msg = parse_message(*raw);
        if (msg.is_null()) {
            auto err = make_error(nullptr, ErrorCode::PARSE_ERROR,
                                  "Failed to parse JSON");
            send_response(err);
            continue;
        }

        auto response = handle_message(msg);
        if (response) {
            send_response(*response);
        }
    }

    running_ = false;
    spdlog::info("MCP server stopped");
}

void McpServer::stop() noexcept {
    running_ = false;
}

bool McpServer::running() const noexcept {
    return running_.load();
}

// =========================================================================
// Message handling
// =========================================================================

std::optional<nlohmann::json> McpServer::handle_message(const nlohmann::json& msg) {
    auto kind = classify_message(msg);

    if (kind == MessageKind::INVALID) {
        return make_error(msg.value("id", nlohmann::json(nullptr)),
                          ErrorCode::INVALID_REQUEST,
                          "Invalid JSON-RPC message");
    }

    const auto method = msg.at("method").get<std::string>();
    const auto params = msg.value("params", nlohmann::json::object());

    // For requests we need the id; for notifications it is absent.
    nlohmann::json id = nullptr;
    if (kind == MessageKind::REQUEST) {
        id = msg.at("id");
    }

    auto it = dispatch_.find(method);
    if (it != dispatch_.end()) {
        return it->second(id, params);
    }

    // Unknown method.
    if (kind == MessageKind::NOTIFICATION) {
        // Notifications with unknown methods are silently ignored per spec.
        spdlog::debug("ignoring unknown notification: {}", method);
        return std::nullopt;
    }

    return make_error(id, ErrorCode::METHOD_NOT_FOUND,
                      "Method not found: " + method);
}

// =========================================================================
// Built-in handlers
// =========================================================================

std::optional<nlohmann::json> McpServer::handle_initialize(
        const nlohmann::json& id,
        const nlohmann::json& /*params*/) {
    spdlog::info("client connected — initialize");

    nlohmann::json result = {
        {"protocolVersion", "2024-11-05"},
        {"capabilities", {
            {"tools", nlohmann::json::object()}
        }},
        {"serverInfo", {
            {"name",    "engram"},
            {"version", "0.1.0"}
        }}
    };

    return make_response(id, result);
}

std::optional<nlohmann::json> McpServer::handle_initialized(
        const nlohmann::json& /*id*/,
        const nlohmann::json& /*params*/) {
    // This is a notification — no response.
    spdlog::info("client initialization complete");
    return std::nullopt;
}

std::optional<nlohmann::json> McpServer::handle_tools_list(
        const nlohmann::json& id,
        const nlohmann::json& /*params*/) {
    nlohmann::json tools_json = nlohmann::json::array();
    for (const auto& rt : tools_) {
        tools_json.push_back(rt.definition);
    }

    nlohmann::json result = {
        {"tools", tools_json}
    };

    return make_response(id, result);
}

std::optional<nlohmann::json> McpServer::handle_tools_call(
        const nlohmann::json& id,
        const nlohmann::json& params) {
    // Extract tool name.
    if (!params.contains("name") || !params["name"].is_string()) {
        return make_error(id, ErrorCode::INVALID_PARAMS,
                          "tools/call requires a 'name' string parameter");
    }

    const auto tool_name = params["name"].get<std::string>();
    const auto arguments = params.value("arguments", nlohmann::json::object());

    auto it = tool_lookup_.find(tool_name);
    if (it == tool_lookup_.end()) {
        return make_error(id, ErrorCode::INVALID_PARAMS,
                          "Unknown tool: " + tool_name);
    }

    spdlog::debug("calling tool: {}", tool_name);

    try {
        auto tool_result = tools_[it->second].handler(arguments);

        // Wrap the tool result in the MCP content array format.
        nlohmann::json result = {
            {"content", nlohmann::json::array({
                {
                    {"type", "text"},
                    {"text", tool_result.dump()}
                }
            })}
        };

        return make_response(id, result);
    } catch (const std::exception& ex) {
        spdlog::error("tool '{}' threw: {}", tool_name, ex.what());
        return make_error(id, ErrorCode::INTERNAL_ERROR,
                          std::string("Tool execution failed: ") + ex.what());
    }
}

// =========================================================================
// I/O helpers
// =========================================================================

void McpServer::send_response(const nlohmann::json& response) {
    const std::string body = response.dump();
    spdlog::debug("send_response: {} bytes, id={}",
                  body.size(),
                  response.value("id", nlohmann::json(nullptr)).dump());

    // MCP stdio transport uses newline-delimited JSON: one JSON object per
    // line, terminated by '\n'.  Messages MUST NOT contain embedded newlines.
    std::string msg = body + "\n";

#ifdef _WIN32
    // Use Win32 WriteFile to bypass C/C++ runtime buffering on pipes.
    HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    if (h == INVALID_HANDLE_VALUE || h == nullptr) {
        spdlog::error("stdout handle is invalid!");
        return;
    }

    DWORD written = 0;
    BOOL ok = WriteFile(h, msg.data(), static_cast<DWORD>(msg.size()), &written, nullptr);
    if (!ok) {
        spdlog::error("WriteFile failed: error={}", GetLastError());
    }
    FlushFileBuffers(h);
#else
    std::cout.write(msg.data(), static_cast<std::streamsize>(msg.size()));
    std::cout.flush();
#endif
}

std::optional<std::string> McpServer::read_message() {
    // MCP stdio transport uses newline-delimited JSON (one JSON object per
    // line).  We also accept Content-Length header framing as a fallback for
    // compatibility with LSP-style clients.

    std::string line;
    while (true) {
        if (!std::getline(std::cin, line)) {
            return std::nullopt;  // EOF
        }

        // Strip trailing \r if present (getline removes \n but not \r).
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        // Skip empty lines (blank lines between messages).
        if (line.empty()) {
            continue;
        }

        break;
    }

    // Check for Content-Length header framing.
    const std::string prefix = "Content-Length:";
    if (line.rfind(prefix, 0) == 0) {
        // Parse the content length.
        std::string length_str = line.substr(prefix.size());
        // Trim leading whitespace.
        auto pos = length_str.find_first_not_of(' ');
        if (pos != std::string::npos) {
            length_str = length_str.substr(pos);
        }

        int content_length = 0;
        try {
            content_length = std::stoi(length_str);
        } catch (...) {
            spdlog::warn("invalid Content-Length value: '{}'", length_str);
            return std::nullopt;
        }

        // Consume the blank line after the header.
        std::string separator;
        if (!std::getline(std::cin, separator)) {
            return std::nullopt;
        }

        // Read exactly content_length bytes.
        std::string body(static_cast<size_t>(content_length), '\0');
        if (!std::cin.read(body.data(), content_length)) {
            return std::nullopt;
        }

        return body;
    }

    // Otherwise treat the line itself as a newline-delimited JSON message.
    return line;
}

} // namespace engram::mcp
