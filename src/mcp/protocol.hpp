#pragma once
/// @file protocol.hpp
/// @brief MCP (Model Context Protocol) message types built on JSON-RPC 2.0.
///
/// Provides strongly-typed helpers for constructing and parsing the JSON-RPC
/// messages that flow over stdio between the MCP client (Claude Code) and
/// this server.  All JSON work is done via nlohmann/json.

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace engram::mcp {

// -------------------------------------------------------------------------
// Standard JSON-RPC 2.0 error codes
// -------------------------------------------------------------------------
enum class ErrorCode : int {
    PARSE_ERROR      = -32700,
    INVALID_REQUEST  = -32600,
    METHOD_NOT_FOUND = -32601,
    INVALID_PARAMS   = -32602,
    INTERNAL_ERROR   = -32603
};

// -------------------------------------------------------------------------
// MCP Tool definition
// -------------------------------------------------------------------------

/// Describes a single tool exposed by the server to the MCP client.
struct ToolDefinition {
    std::string       name;         ///< Machine-readable tool name.
    std::string       description;  ///< Human-readable explanation.
    nlohmann::json    input_schema; ///< JSON Schema for the tool's input.
};

/// Serialize a ToolDefinition to the JSON shape expected by `tools/list`.
inline void to_json(nlohmann::json& j, const ToolDefinition& td) {
    j = nlohmann::json{
        {"name",        td.name},
        {"description", td.description},
        {"inputSchema", td.input_schema}
    };
}

/// Deserialize a ToolDefinition from JSON (useful in tests).
inline void from_json(const nlohmann::json& j, ToolDefinition& td) {
    j.at("name").get_to(td.name);
    j.at("description").get_to(td.description);
    td.input_schema = j.at("inputSchema");
}

// -------------------------------------------------------------------------
// Message parsing
// -------------------------------------------------------------------------

/// Possible message kinds returned by classify_message.
enum class MessageKind {
    REQUEST,       ///< Has "id" and "method".
    NOTIFICATION,  ///< Has "method" but no "id".
    RESPONSE,      ///< Has "id" but no "method" (not used server-side, included for completeness).
    INVALID        ///< Could not classify.
};

/// Parse a raw string into a JSON value.
///
/// @param raw  A single JSON-RPC message (UTF-8 string).
/// @return The parsed JSON, or a JSON null on failure.
///
/// On parse failure the returned value is `json(nullptr)` — callers should
/// check `j.is_null()` before using the result.
inline nlohmann::json parse_message(const std::string& raw) {
    try {
        return nlohmann::json::parse(raw);
    } catch (const nlohmann::json::parse_error&) {
        return nullptr;
    }
}

/// Classify a parsed JSON-RPC message.
inline MessageKind classify_message(const nlohmann::json& msg) {
    if (!msg.is_object()) return MessageKind::INVALID;
    const bool has_id     = msg.contains("id");
    const bool has_method = msg.contains("method");
    if (has_method && has_id)  return MessageKind::REQUEST;
    if (has_method && !has_id) return MessageKind::NOTIFICATION;
    if (!has_method && has_id) return MessageKind::RESPONSE;
    return MessageKind::INVALID;
}

// -------------------------------------------------------------------------
// Response / error construction helpers
// -------------------------------------------------------------------------

/// Build a successful JSON-RPC response.
///
/// @param id      The request id (number or string — forwarded as-is).
/// @param result  The result payload.
/// @return A complete JSON-RPC 2.0 response object.
inline nlohmann::json make_response(const nlohmann::json& id,
                                    const nlohmann::json& result) {
    return {
        {"jsonrpc", "2.0"},
        {"id",      id},
        {"result",  result}
    };
}

/// Build a JSON-RPC error response.
///
/// @param id       The request id (may be null for parse errors).
/// @param code     A standard or application-defined error code.
/// @param message  A short human-readable description.
/// @return A complete JSON-RPC 2.0 error response object.
inline nlohmann::json make_error(const nlohmann::json& id,
                                 ErrorCode code,
                                 const std::string& message) {
    return {
        {"jsonrpc", "2.0"},
        {"id",      id},
        {"error", {
            {"code",    static_cast<int>(code)},
            {"message", message}
        }}
    };
}

/// Overload that accepts a raw integer error code.
inline nlohmann::json make_error(const nlohmann::json& id,
                                 int code,
                                 const std::string& message) {
    return {
        {"jsonrpc", "2.0"},
        {"id",      id},
        {"error", {
            {"code",    code},
            {"message", message}
        }}
    };
}

} // namespace engram::mcp
