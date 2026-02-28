/// @file tools.cpp
/// @brief Tool handler stubs for the Engram MCP server.
///
/// Each handler validates its input, then returns a placeholder response.
/// The structure is designed so that wiring in real implementations
/// (vector index, chunker, session store) requires minimal changes.

#include "mcp/tools.hpp"

#include <spdlog/spdlog.h>

namespace engram::mcp {

// =========================================================================
// JSON Schema helpers (DRY builders for common schema fragments)
// =========================================================================

static nlohmann::json string_prop(const std::string& description) {
    return {{"type", "string"}, {"description", description}};
}

static nlohmann::json integer_prop(const std::string& description) {
    return {{"type", "integer"}, {"description", description}};
}

// =========================================================================
// search_code
// =========================================================================

static nlohmann::json handle_search_code(const nlohmann::json& args) {
    const auto query = args.value("query", std::string{});
    const auto limit = args.value("limit", 10);

    spdlog::debug("search_code: query='{}' limit={}", query, limit);

    return {
        {"status", "not_yet_implemented"},
        {"message", "search_code is a stub. The vector index is not wired yet."},
        {"echo", {
            {"query", query},
            {"limit", limit}
        }},
        {"results", nlohmann::json::array()}
    };
}

// =========================================================================
// search_symbol
// =========================================================================

static nlohmann::json handle_search_symbol(const nlohmann::json& args) {
    const auto name = args.value("name", std::string{});
    const auto kind = args.value("kind", std::string{"any"});

    spdlog::debug("search_symbol: name='{}' kind='{}'", name, kind);

    return {
        {"status", "not_yet_implemented"},
        {"message", "search_symbol is a stub. The symbol index is not wired yet."},
        {"echo", {
            {"name", name},
            {"kind", kind}
        }},
        {"results", nlohmann::json::array()}
    };
}

// =========================================================================
// get_context
// =========================================================================

static nlohmann::json handle_get_context(const nlohmann::json& args) {
    const auto file   = args.value("file", std::string{});
    const auto line   = args.value("line", 0);
    const auto radius = args.value("radius", 50);

    spdlog::debug("get_context: file='{}' line={} radius={}", file, line, radius);

    return {
        {"status", "not_yet_implemented"},
        {"message", "get_context is a stub. The chunker is not wired yet."},
        {"echo", {
            {"file",   file},
            {"line",   line},
            {"radius", radius}
        }},
        {"results", nlohmann::json::array()}
    };
}

// =========================================================================
// get_session_memory
// =========================================================================

static nlohmann::json handle_get_session_memory(const nlohmann::json& args) {
    const auto query = args.value("query", std::string{});

    spdlog::debug("get_session_memory: query='{}'", query);

    return {
        {"status", "not_yet_implemented"},
        {"message", "get_session_memory is a stub. The session store is not wired yet."},
        {"echo", {
            {"query", query}
        }},
        {"sessions", nlohmann::json::array()}
    };
}

// =========================================================================
// save_session_summary
// =========================================================================

static nlohmann::json handle_save_session_summary(const nlohmann::json& args) {
    const auto summary       = args.value("summary", std::string{});
    const auto key_files     = args.value("key_files", std::vector<std::string>{});
    const auto key_decisions = args.value("key_decisions", std::vector<std::string>{});

    spdlog::debug("save_session_summary: summary length={}", summary.size());

    return {
        {"status", "not_yet_implemented"},
        {"message", "save_session_summary is a stub. The session store is not wired yet."},
        {"echo", {
            {"summary",       summary},
            {"key_files",     key_files},
            {"key_decisions", key_decisions}
        }}
    };
}

// =========================================================================
// Registration
// =========================================================================

void register_all_tools(McpServer& server) {
    // -- search_code -------------------------------------------------------
    server.register_tool(
        "search_code",
        "Semantic code search. Returns code chunks ranked by relevance "
        "to the natural-language query.",
        {
            {"type", "object"},
            {"properties", {
                {"query", string_prop("Natural-language search query")},
                {"limit", integer_prop("Maximum number of results (default 10)")}
            }},
            {"required", nlohmann::json::array({"query"})}
        },
        handle_search_code
    );

    // -- search_symbol -----------------------------------------------------
    server.register_tool(
        "search_symbol",
        "Look up a code symbol (function, class, variable) by name. "
        "Optionally filter by symbol kind.",
        {
            {"type", "object"},
            {"properties", {
                {"name", string_prop("Symbol name or pattern to search for")},
                {"kind", {
                    {"type", "string"},
                    {"description", "Symbol kind filter"},
                    {"enum", nlohmann::json::array({"function", "class", "any"})}
                }}
            }},
            {"required", nlohmann::json::array({"name"})}
        },
        handle_search_symbol
    );

    // -- get_context -------------------------------------------------------
    server.register_tool(
        "get_context",
        "Retrieve code context around a specific file and line number. "
        "Returns related chunks within the given radius.",
        {
            {"type", "object"},
            {"properties", {
                {"file",   string_prop("File path (relative to project root)")},
                {"line",   integer_prop("Line number (1-based)")},
                {"radius", integer_prop("Number of surrounding lines to include (default 50)")}
            }},
            {"required", nlohmann::json::array({"file", "line"})}
        },
        handle_get_context
    );

    // -- get_session_memory ------------------------------------------------
    server.register_tool(
        "get_session_memory",
        "Recall summaries from previous coding sessions. Optionally filter "
        "by a relevance query.",
        {
            {"type", "object"},
            {"properties", {
                {"query", string_prop("Optional relevance query to filter sessions")}
            }}
        },
        handle_get_session_memory
    );

    // -- save_session_summary ---------------------------------------------
    server.register_tool(
        "save_session_summary",
        "Persist a summary of the current coding session for future recall.",
        {
            {"type", "object"},
            {"properties", {
                {"summary",       string_prop("Free-text summary of what was accomplished")},
                {"key_files",     {
                    {"type", "array"},
                    {"items", {{"type", "string"}}},
                    {"description", "Important files touched in this session"}
                }},
                {"key_decisions", {
                    {"type", "array"},
                    {"items", {{"type", "string"}}},
                    {"description", "Key design or implementation decisions made"}
                }}
            }},
            {"required", nlohmann::json::array({"summary"})}
        },
        handle_save_session_summary
    );

    spdlog::info("registered {} tools", server.tools().size());
}

} // namespace engram::mcp
