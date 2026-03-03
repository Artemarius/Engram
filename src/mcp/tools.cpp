/// @file tools.cpp
/// @brief Tool handler implementations for the Engram MCP server.
///
/// Each handler validates its input, then uses the ToolContext to perform
/// real operations against the vector index, chunk store, and session store.
/// With multi-project support, handlers iterate over all ProjectContexts and
/// merge results.  When a required backend component is nullptr, the handler
/// returns a descriptive error rather than crashing.

#include "mcp/tools.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>

namespace {

/// RAII lock guard that is a no-op when the mutex pointer is null.
/// Allows tool handlers to protect chunk_store access only when a
/// shared mutex has been configured (i.e. when the watcher is active).
class OptionalLock {
public:
    explicit OptionalLock(std::mutex* m) : mutex_(m) {
        if (mutex_) mutex_->lock();
    }
    ~OptionalLock() {
        if (mutex_) mutex_->unlock();
    }
    OptionalLock(const OptionalLock&) = delete;
    OptionalLock& operator=(const OptionalLock&) = delete;
private:
    std::mutex* mutex_;
};

} // anonymous namespace

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
// String utilities
// =========================================================================

/// Convert a string to lowercase (ASCII only, sufficient for symbol matching).
static std::string to_lower(const std::string& s) {
    std::string result = s;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return result;
}

/// Case-insensitive substring search.
static bool contains_icase(const std::string& haystack, const std::string& needle) {
    return to_lower(haystack).find(to_lower(needle)) != std::string::npos;
}

/// Check if ALL whitespace-separated words in @p query appear (case-insensitive)
/// somewhere in @p text.  Useful for fuzzy keyword matching where "MCP protocol"
/// should match "MCP stdio protocol".
static bool matches_all_words(const std::string& text, const std::string& query) {
    std::string lower_text = to_lower(text);
    std::string lower_query = to_lower(query);

    // Split query on whitespace and check each word.
    size_t pos = 0;
    while (pos < lower_query.size()) {
        // Skip whitespace.
        while (pos < lower_query.size() && std::isspace(static_cast<unsigned char>(lower_query[pos]))) {
            ++pos;
        }
        if (pos >= lower_query.size()) break;

        // Find end of word.
        size_t end = pos;
        while (end < lower_query.size() && !std::isspace(static_cast<unsigned char>(lower_query[end]))) {
            ++end;
        }

        std::string word = lower_query.substr(pos, end - pos);
        if (lower_text.find(word) == std::string::npos) {
            return false;  // Word not found — no match.
        }
        pos = end;
    }
    return true;
}

/// Make a file path relative to the project root for display.
static std::string make_relative(const std::filesystem::path& file_path,
                                 const std::filesystem::path& project_root) {
    if (project_root.empty()) {
        return file_path.generic_string();
    }

    std::error_code ec;
    auto rel = std::filesystem::relative(file_path, project_root, ec);
    if (ec || rel.empty()) {
        return file_path.generic_string();
    }
    return rel.generic_string();
}

/// Serialize a Chunk to a JSON object suitable for tool results.
static nlohmann::json chunk_to_json(const Chunk& chunk,
                                    const std::filesystem::path& project_root,
                                    float score = -1.0f,
                                    const std::string& project_name = {}) {
    nlohmann::json j = {
        {"file_path",   make_relative(chunk.file_path, project_root)},
        {"start_line",  chunk.start_line},
        {"end_line",    chunk.end_line},
        {"language",    chunk.language},
        {"source_text", chunk.source_text},
        {"chunk_id",    chunk.chunk_id}
    };

    if (!chunk.symbol_name.empty()) {
        j["symbol_name"] = chunk.symbol_name;
    }

    if (score >= 0.0f) {
        j["score"] = score;
    }

    if (!project_name.empty()) {
        j["project"] = project_name;
    }

    return j;
}

/// Return true if there are multiple projects loaded (for adding "project" field).
static bool is_multi_project(const ToolContext& ctx) {
    return ctx.projects && ctx.projects->size() > 1;
}

// =========================================================================
// search_code
// =========================================================================

static nlohmann::json handle_search_code(ToolContext& ctx,
                                         const nlohmann::json& args) {
    const auto query = args.value("query", std::string{});
    const auto limit = args.value("limit", 10);

    spdlog::debug("search_code: query='{}' limit={}", query, limit);

    if (query.empty()) {
        return {
            {"error", "missing required parameter 'query'"},
            {"results", nlohmann::json::array()}
        };
    }

    // Check that embedder is available.
    if (!ctx.embedder) {
        return {
            {"error", "embedder not configured -- semantic search is unavailable"},
            {"results", nlohmann::json::array()}
        };
    }

    // Check that we have projects.
    if (!ctx.projects || ctx.projects->empty()) {
        return {
            {"error", "no projects configured"},
            {"results", nlohmann::json::array()}
        };
    }

    // Embed the query string once (shared across all projects).
    auto embedding = ctx.embedder->embed(query);
    if (embedding.empty()) {
        spdlog::warn("search_code: embedder returned empty vector for query '{}'", query);
        return {
            {"error", "failed to embed query"},
            {"results", nlohmann::json::array()}
        };
    }

    // Collect scored results from all projects.
    struct ScoredResult {
        nlohmann::json json;
        float score;
    };
    std::vector<ScoredResult> all_results;
    bool multi = is_multi_project(ctx);

    for (const auto& proj : *ctx.projects) {
        // Search this project's vector index.
        auto hits = proj->vector_index.search(embedding.data(),
                                              embedding.size(),
                                              static_cast<size_t>(limit));

        OptionalLock lock(&proj->index_mutex);
        for (const auto& hit : hits) {
            auto it = proj->chunk_map.find(hit.chunk_id);
            if (it != proj->chunk_map.end()) {
                all_results.push_back({
                    chunk_to_json(it->second, proj->project_root, hit.score,
                                  multi ? proj->name : std::string{}),
                    hit.score
                });
            } else {
                nlohmann::json j = {
                    {"chunk_id", hit.chunk_id},
                    {"score",    hit.score},
                    {"warning",  "chunk metadata not found in store"}
                };
                if (multi) j["project"] = proj->name;
                all_results.push_back({std::move(j), hit.score});
            }
        }
    }

    // Sort by score descending and take top 'limit'.
    std::sort(all_results.begin(), all_results.end(),
              [](const ScoredResult& a, const ScoredResult& b) {
                  return a.score > b.score;
              });

    nlohmann::json results = nlohmann::json::array();
    size_t count = (std::min)(all_results.size(), static_cast<size_t>(limit));
    for (size_t i = 0; i < count; ++i) {
        results.push_back(std::move(all_results[i].json));
    }

    return {
        {"query",   query},
        {"count",   results.size()},
        {"results", results}
    };
}

// =========================================================================
// search_symbol
// =========================================================================

static nlohmann::json handle_search_symbol(ToolContext& ctx,
                                           const nlohmann::json& args) {
    const auto name = args.value("name", std::string{});
    const auto kind = args.value("kind", std::string{"any"});

    spdlog::debug("search_symbol: name='{}' kind='{}'", name, kind);

    if (name.empty()) {
        return {
            {"error", "missing required parameter 'name'"},
            {"results", nlohmann::json::array()}
        };
    }

    if (!ctx.projects || ctx.projects->empty()) {
        return {
            {"error", "chunk store not available"},
            {"results", nlohmann::json::array()}
        };
    }

    nlohmann::json results = nlohmann::json::array();
    bool multi = is_multi_project(ctx);

    for (const auto& proj : *ctx.projects) {
        OptionalLock lock(&proj->index_mutex);
        for (const auto& [chunk_id, chunk] : proj->chunk_map) {
            // Skip chunks with no symbol name.
            if (chunk.symbol_name.empty()) {
                continue;
            }

            // Case-insensitive substring match on symbol name.
            if (!contains_icase(chunk.symbol_name, name)) {
                continue;
            }

            // Filter by kind if specified and not "any".
            if (kind != "any" && !kind.empty()) {
                if (kind == "function") {
                    bool is_function = contains_icase(chunk.source_text, "def ") ||
                                       contains_icase(chunk.source_text, "function ") ||
                                       contains_icase(chunk.source_text, "fn ") ||
                                       (chunk.source_text.find('(') != std::string::npos &&
                                        !contains_icase(chunk.source_text, "class ") &&
                                        !contains_icase(chunk.source_text, "struct "));
                    if (!is_function) {
                        continue;
                    }
                } else if (kind == "class") {
                    bool is_class = contains_icase(chunk.source_text, "class ") ||
                                    contains_icase(chunk.source_text, "struct ");
                    if (!is_class) {
                        continue;
                    }
                }
            }

            results.push_back(chunk_to_json(chunk, proj->project_root, -1.0f,
                                            multi ? proj->name : std::string{}));
        }
    }

    return {
        {"name",    name},
        {"kind",    kind},
        {"count",   results.size()},
        {"results", results}
    };
}

// =========================================================================
// get_context
// =========================================================================

static nlohmann::json handle_get_context(ToolContext& ctx,
                                         const nlohmann::json& args) {
    const auto file   = args.value("file", std::string{});
    const auto line   = args.value("line", 0);
    const auto radius = args.value("radius", 50);

    spdlog::debug("get_context: file='{}' line={} radius={}", file, line, radius);

    if (file.empty()) {
        return {
            {"error", "missing required parameter 'file'"},
            {"results", nlohmann::json::array()}
        };
    }

    if (!ctx.projects || ctx.projects->empty()) {
        return {
            {"error", "chunk store not available"},
            {"results", nlohmann::json::array()}
        };
    }

    // Compute the line range window.
    int window_start = (std::max)(1, line - radius);
    int window_end   = line + radius;

    nlohmann::json local_results = nlohmann::json::array();
    nlohmann::json related_results = nlohmann::json::array();
    bool multi = is_multi_project(ctx);

    for (const auto& proj : *ctx.projects) {
        // Build the full file path for comparison against this project.
        std::filesystem::path target_path;
        if (std::filesystem::path(file).is_absolute()) {
            target_path = file;
        } else {
            target_path = proj->project_root / file;
        }

        std::string target_generic = target_path.generic_string();
        std::string target_lower = to_lower(target_generic);
        std::string proj_root_str = proj->project_root.generic_string();

        OptionalLock lock(&proj->index_mutex);

        for (const auto& [chunk_id, chunk] : proj->chunk_map) {
            // Match file path: check the generic string and relative form.
            std::string chunk_generic = chunk.file_path.generic_string();
            std::string chunk_lower = to_lower(chunk_generic);

            bool file_matches = (chunk_lower == target_lower);

            if (!file_matches) {
                std::string rel = make_relative(chunk.file_path, proj->project_root);
                file_matches = (to_lower(rel) == to_lower(file));
            }

            if (!file_matches) {
                file_matches = (to_lower(chunk.file_path.filename().generic_string()) ==
                               to_lower(std::filesystem::path(file).filename().generic_string()));
                if (file_matches) {
                    std::string rel = make_relative(chunk.file_path, proj->project_root);
                    std::string file_lower_str = to_lower(file);
                    std::string rel_lower = to_lower(rel);
                    file_matches = (rel_lower.size() >= file_lower_str.size() &&
                                   rel_lower.substr(rel_lower.size() - file_lower_str.size()) == file_lower_str);
                }
            }

            if (!file_matches) {
                continue;
            }

            int chunk_start = static_cast<int>(chunk.start_line);
            int chunk_end   = static_cast<int>(chunk.end_line);

            if (chunk_end < window_start || chunk_start > window_end) {
                continue;
            }

            local_results.push_back(chunk_to_json(chunk, proj->project_root, -1.0f,
                                                  multi ? proj->name : std::string{}));
        }

        // Semantic related chunks via the first local result.
        if (ctx.embedder && !local_results.empty()) {
            std::string seed_text;
            if (local_results[0].contains("source_text")) {
                seed_text = local_results[0]["source_text"].get<std::string>();
            }

            if (!seed_text.empty()) {
                auto embedding = ctx.embedder->embed(seed_text);
                if (!embedding.empty()) {
                    auto hits = proj->vector_index.search(embedding.data(), embedding.size(), 5);
                    for (const auto& hit : hits) {
                        bool is_local = false;
                        for (const auto& lr : local_results) {
                            if (lr.contains("chunk_id") &&
                                lr["chunk_id"].get<std::string>() == hit.chunk_id) {
                                is_local = true;
                                break;
                            }
                        }
                        if (is_local) continue;

                        auto it = proj->chunk_map.find(hit.chunk_id);
                        if (it != proj->chunk_map.end()) {
                            related_results.push_back(
                                chunk_to_json(it->second, proj->project_root, hit.score,
                                              multi ? proj->name : std::string{}));
                        }
                    }
                }
            }
        }
    }

    nlohmann::json result = {
        {"file",    file},
        {"line",    line},
        {"radius",  radius},
        {"count",   local_results.size()},
        {"results", local_results}
    };

    if (!related_results.empty()) {
        result["related"]       = related_results;
        result["related_count"] = related_results.size();
    }

    return result;
}

// =========================================================================
// get_session_memory
// =========================================================================

static nlohmann::json handle_get_session_memory(ToolContext& ctx,
                                                const nlohmann::json& args) {
    const auto query = args.value("query", std::string{});

    spdlog::debug("get_session_memory: query='{}'", query);

    // Collect session stores from all projects + the primary store.
    std::vector<SessionStore*> stores;
    if (ctx.session_store) {
        stores.push_back(ctx.session_store);
    }
    if (ctx.projects) {
        for (const auto& proj : *ctx.projects) {
            if (proj->session_store) {
                // Avoid duplicates (primary store is already in the list).
                if (proj->session_store.get() != ctx.session_store) {
                    stores.push_back(proj->session_store.get());
                }
            }
        }
    }

    if (stores.empty()) {
        return {
            {"error", "session store not configured"},
            {"sessions", nlohmann::json::array()}
        };
    }

    // Load all sessions from all stores.
    std::vector<SessionSummary> all_sessions;
    for (auto* store : stores) {
        auto sessions = store->load_all();
        all_sessions.insert(all_sessions.end(),
                           std::make_move_iterator(sessions.begin()),
                           std::make_move_iterator(sessions.end()));
    }

    if (all_sessions.empty()) {
        return {
            {"count", 0},
            {"sessions", nlohmann::json::array()}
        };
    }

    // Sort by timestamp descending (most recent first).
    std::sort(all_sessions.begin(), all_sessions.end(),
              [](const SessionSummary& a, const SessionSummary& b) {
                  return a.id > b.id;
              });

    // Deduplicate by ID (same session may appear in multiple stores).
    {
        auto it = std::unique(all_sessions.begin(), all_sessions.end(),
                              [](const SessionSummary& a, const SessionSummary& b) {
                                  return a.id == b.id;
                              });
        all_sessions.erase(it, all_sessions.end());
    }

    // Filter by query keywords.
    std::vector<SessionSummary> matched;
    if (query.empty()) {
        matched = std::move(all_sessions);
    } else {
        for (const auto& session : all_sessions) {
            std::string combined = session.summary;
            for (const auto& f : session.key_files) {
                combined += " " + f;
            }
            for (const auto& d : session.key_decisions) {
                combined += " " + d;
            }

            if (matches_all_words(combined, query)) {
                matched.push_back(session);
            }
        }
    }

    // Build the result array.
    nlohmann::json sessions_json = nlohmann::json::array();
    for (const auto& session : matched) {
        nlohmann::json sj;
        to_json(sj, session);
        sessions_json.push_back(sj);
    }

    return {
        {"query",    query},
        {"count",    sessions_json.size()},
        {"sessions", sessions_json}
    };
}

// =========================================================================
// save_session_summary
// =========================================================================

static nlohmann::json handle_save_session_summary(ToolContext& ctx,
                                                  const nlohmann::json& args) {
    const auto summary       = args.value("summary", std::string{});
    const auto key_files     = args.value("key_files", std::vector<std::string>{});
    const auto key_decisions = args.value("key_decisions", std::vector<std::string>{});

    spdlog::debug("save_session_summary: summary length={}", summary.size());

    if (summary.empty()) {
        return {
            {"error", "missing required parameter 'summary'"}
        };
    }

    if (!ctx.session_store) {
        return {
            {"error", "session store not configured"}
        };
    }

    SessionSummary session;
    session.summary       = summary;
    session.key_files     = key_files;
    session.key_decisions = key_decisions;

    if (!ctx.session_store->save(session)) {
        return {
            {"error", "failed to save session summary to disk"}
        };
    }

    return {
        {"status",     "saved"},
        {"session_id", session.id},
        {"timestamp",  session.timestamp}
    };
}

// =========================================================================
// Registration
// =========================================================================

void register_all_tools(McpServer& server, ToolContext& context) {
    auto& ctx = context;

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
        [&ctx](const nlohmann::json& args) {
            return handle_search_code(ctx, args);
        }
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
        [&ctx](const nlohmann::json& args) {
            return handle_search_symbol(ctx, args);
        }
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
        [&ctx](const nlohmann::json& args) {
            return handle_get_context(ctx, args);
        }
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
        [&ctx](const nlohmann::json& args) {
            return handle_get_session_memory(ctx, args);
        }
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
        [&ctx](const nlohmann::json& args) {
            return handle_save_session_summary(ctx, args);
        }
    );

    spdlog::info("registered {} tools", server.tools().size());
}

} // namespace engram::mcp
