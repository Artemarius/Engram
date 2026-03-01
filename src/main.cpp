/// @file main.cpp
/// @brief Entry point for the Engram MCP server.
///
/// Parses command-line arguments, initializes all subsystems (chunker, vector
/// index, session store, optional embedder), performs initial project indexing,
/// and enters the MCP server loop that communicates with Claude Code over stdio.
///
/// All diagnostic output goes to stderr via spdlog.  stdout is reserved
/// exclusively for the MCP JSON-RPC 2.0 protocol.

#include <chrono>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <filesystem>
#include <memory>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <nlohmann/json.hpp>

#include "chunker/chunker.hpp"
#include "chunker/chunk_store.hpp"
#include "chunker/regex_chunker.hpp"
#include "embedder/embedder.hpp"
#include "index/hnsw_index.hpp"
#include "session/session_store.hpp"
#include "mcp/mcp_server.hpp"
#include "mcp/tools.hpp"
#include "watcher/win_watcher.hpp"

#ifdef ENGRAM_HAS_ONNX
#include "embedder/ort_embedder.hpp"
#endif

namespace fs = std::filesystem;

// =========================================================================
// CLI argument parsing
// =========================================================================

/// Parse a simple --key value pair from the argument list.
/// Returns the value if found, or @p fallback otherwise.
static std::string parse_arg(const std::vector<std::string>& args,
                             const std::string& key,
                             const std::string& fallback = {})
{
    for (size_t i = 0; i + 1 < args.size(); ++i) {
        if (args[i] == key) {
            return args[i + 1];
        }
    }
    return fallback;
}

/// Check if a flag (no value) is present.
static bool has_flag(const std::vector<std::string>& args,
                     const std::string& flag)
{
    for (const auto& a : args) {
        if (a == flag) return true;
    }
    return false;
}

/// Print usage information to stderr.
static void print_help() {
    spdlog::info("Usage: engram-mcp [options]");
    spdlog::info("  --project  <path>   Root of the codebase to index");
    spdlog::info("  --model    <path>   Path to the ONNX embedding model");
    spdlog::info("  --data-dir <path>   Directory for persistent data (default: <project>/.engram/)");
    spdlog::info("  --dim      <int>    Embedding dimension (default: 384)");
    spdlog::info("  --reindex           Force a full re-index of the project");
    spdlog::info("  --verbose           Enable debug-level logging");
    spdlog::info("  --help, -h          Show this help message");
}

// =========================================================================
// File extension filter
// =========================================================================

/// Set of file extensions (lowercase, including leading dot) that are supported
/// for code chunking.
static const std::unordered_set<std::string>& supported_extensions() {
    static const std::unordered_set<std::string> exts = {
        ".cpp", ".hpp", ".h", ".c", ".cc", ".cxx", ".hxx",
        ".py",
        ".js", ".ts", ".jsx", ".tsx",
        ".java",
        ".rs",
        ".go",
        ".rb",
        ".cs"
    };
    return exts;
}

/// Directories to skip during recursive file walking.
static const std::unordered_set<std::string>& skip_directories() {
    static const std::unordered_set<std::string> dirs = {
        ".git", "build", "node_modules", "__pycache__",
        ".vs", ".vscode", ".engram", ".claude",
        "target",    // Rust
        "dist",      // JS/TS
        "out",       // generic build output
        ".idea"      // JetBrains
    };
    return dirs;
}

// =========================================================================
// Project file walking
// =========================================================================

/// Recursively walk a project directory and collect source files with
/// supported extensions, skipping well-known non-source directories.
///
/// @param project_root  The root directory to walk.
/// @return A sorted vector of absolute file paths.
static std::vector<fs::path> walk_project_files(const fs::path& project_root) {
    std::vector<fs::path> files;

    std::error_code ec;
    auto it = fs::recursive_directory_iterator(
        project_root,
        fs::directory_options::skip_permission_denied,
        ec
    );

    if (ec) {
        spdlog::error("failed to open project directory '{}': {}",
                       project_root.generic_string(), ec.message());
        return files;
    }

    const auto& skip_dirs = skip_directories();
    const auto& exts      = supported_extensions();

    for (; it != fs::recursive_directory_iterator(); ++it) {
        const auto& entry = *it;

        // Skip directories that should be ignored.
        if (entry.is_directory()) {
            const auto dirname = entry.path().filename().string();
            if (skip_dirs.count(dirname)) {
                it.disable_recursion_pending();
                continue;
            }
            continue;
        }

        // Only process regular files with supported extensions.
        if (!entry.is_regular_file()) continue;

        auto ext = entry.path().extension().string();
        // Lowercase the extension for case-insensitive matching.
        for (auto& ch : ext) {
            ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        }

        if (exts.count(ext)) {
            files.push_back(entry.path());
        }
    }

    std::sort(files.begin(), files.end());
    return files;
}

// =========================================================================
// Initial project indexing
// =========================================================================

/// Chunk all source files in the project and optionally embed them into the
/// vector index.
///
/// @param project_root  The root directory of the project.
/// @param chunker       The chunker to use.
/// @param chunk_map     Output map: chunk_id -> Chunk metadata.
/// @param index         The vector index (receives embeddings if embedder is available).
/// @param embedder_ptr  Optional embedder. Pass nullptr to skip embedding.
///
/// @return The total number of chunks created.
static size_t index_project(
    const fs::path& project_root,
    engram::RegexChunker& chunker,
    std::unordered_map<std::string, engram::Chunk>& chunk_map,
    engram::HnswIndex& index,
    engram::Embedder* embedder_ptr)
{
    auto start_time = std::chrono::steady_clock::now();

    auto files = walk_project_files(project_root);
    spdlog::info("found {} source files to index", files.size());

    size_t total_chunks = 0;
    size_t files_processed = 0;
    size_t chunks_embedded = 0;

    for (const auto& file_path : files) {
        auto chunks = chunker.chunk_file(file_path);

        if (chunks.empty()) {
            spdlog::debug("no chunks from '{}'", file_path.generic_string());
            continue;
        }

        files_processed++;

        for (auto& chunk : chunks) {
            const auto& id = chunk.chunk_id;

            // Store metadata.
            chunk_map[id] = chunk;
            total_chunks++;

            // Embed and add to index if embedder is available.
            if (embedder_ptr) {
                auto embedding = embedder_ptr->embed(chunk.source_text);
                if (!embedding.empty()) {
                    if (index.add(id, embedding.data(), embedding.size())) {
                        chunks_embedded++;
                    } else {
                        spdlog::warn("failed to add chunk '{}' to index", id);
                    }
                } else {
                    spdlog::warn("embedding failed for chunk '{}' in '{}'",
                                 id, file_path.generic_string());
                }
            }
        }

        // Log progress every 50 files.
        if (files_processed % 50 == 0) {
            spdlog::info("  progress: {}/{} files, {} chunks so far",
                         files_processed, files.size(), total_chunks);
        }
    }

    auto elapsed = std::chrono::steady_clock::now() - start_time;
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

    spdlog::info("indexing complete:");
    spdlog::info("  files processed:  {}", files_processed);
    spdlog::info("  chunks created:   {}", total_chunks);
    if (chunks_embedded > 0) {
        spdlog::info("  chunks embedded:  {}", chunks_embedded);
    }
    spdlog::info("  elapsed:          {} ms", elapsed_ms);

    return total_chunks;
}

// =========================================================================
// main
// =========================================================================

int main(int argc, char* argv[])
{
    // -----------------------------------------------------------------------
    // Logger -- everything goes to stderr; stdout is reserved for MCP protocol.
    // -----------------------------------------------------------------------
    auto stderr_sink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
    auto logger = std::make_shared<spdlog::logger>("engram", stderr_sink);
    logger->set_level(spdlog::level::info);
    spdlog::set_default_logger(logger);

    // -----------------------------------------------------------------------
    // Command-line arguments
    // -----------------------------------------------------------------------
    std::vector<std::string> args(argv, argv + argc);

    if (has_flag(args, "--help") || has_flag(args, "-h")) {
        print_help();
        return 0;
    }

    if (has_flag(args, "--verbose")) {
        logger->set_level(spdlog::level::debug);
    }

    const std::string project_path = parse_arg(args, "--project");
    const std::string model_path   = parse_arg(args, "--model");
    const std::string dim_str      = parse_arg(args, "--dim", "384");
    const bool        force_reindex = has_flag(args, "--reindex");

    // Parse embedding dimension.
    size_t embedding_dim = 384;
    try {
        embedding_dim = static_cast<size_t>(std::stoul(dim_str));
    } catch (...) {
        spdlog::error("invalid --dim value '{}', using default 384", dim_str);
    }

    // Resolve data directory.
    // Priority: --data-dir flag > {project}/.engram/ > ./data/
    std::string data_dir_str = parse_arg(args, "--data-dir");
    fs::path data_dir;

    if (!data_dir_str.empty()) {
        data_dir = fs::path(data_dir_str);
    } else if (!project_path.empty()) {
        data_dir = fs::path(project_path) / ".engram";
    } else {
        data_dir = fs::path("./data");
    }

    // -----------------------------------------------------------------------
    // Startup banner (stderr only)
    // -----------------------------------------------------------------------
    spdlog::info("engram-mcp starting up");
    spdlog::info("  build:     {} {}", __DATE__, __TIME__);
    spdlog::info("  dim:       {}", embedding_dim);
    spdlog::info("  data-dir:  {}", data_dir.generic_string());

    if (!project_path.empty()) {
        spdlog::info("  project:   {}", project_path);
    } else {
        spdlog::warn("  no --project specified; indexing disabled");
    }

    if (!model_path.empty()) {
        spdlog::info("  model:     {}", model_path);
    } else {
        spdlog::warn("  no --model specified; embedding disabled");
    }

    if (force_reindex) {
        spdlog::info("  reindex:   forced");
    }

    // -----------------------------------------------------------------------
    // Create data directories
    // -----------------------------------------------------------------------
    {
        std::error_code ec;
        fs::create_directories(data_dir, ec);
        if (ec) {
            spdlog::error("failed to create data directory '{}': {}",
                           data_dir.generic_string(), ec.message());
            return 1;
        }

        fs::create_directories(data_dir / "sessions", ec);
        if (ec) {
            spdlog::error("failed to create sessions directory: {}", ec.message());
            return 1;
        }
    }

    // -----------------------------------------------------------------------
    // Initialize components
    // -----------------------------------------------------------------------

    // Session store -- persists session summaries as JSON files.
    engram::SessionStore session_store(data_dir / "sessions");
    spdlog::info("session store initialized at '{}'",
                 session_store.storage_directory().generic_string());

    // Vector index -- HNSW for nearest-neighbor search.
    engram::HnswIndex vector_index(embedding_dim);

    // Chunk metadata store -- maps chunk_id to full Chunk struct.
    std::unordered_map<std::string, engram::Chunk> chunk_map;

    // Try to load existing index and chunk store from disk (unless --reindex).
    fs::path index_path  = data_dir / "index";
    fs::path chunks_path = data_dir / "chunks.json";
    bool loaded_from_disk = false;

    if (!force_reindex) {
        bool index_ok  = false;
        bool chunks_ok = false;

        std::error_code ec;
        if (fs::exists(index_path, ec)) {
            spdlog::info("loading existing index from '{}'", index_path.generic_string());
            if (vector_index.load(index_path)) {
                spdlog::info("loaded index with {} vectors", vector_index.size());
                index_ok = true;
            } else {
                spdlog::warn("failed to load index from '{}'; starting fresh",
                             index_path.generic_string());
            }
        } else {
            spdlog::debug("no existing index found at '{}'", index_path.generic_string());
        }

        if (index_ok && fs::exists(chunks_path, ec)) {
            spdlog::info("loading chunk store from '{}'", chunks_path.generic_string());
            if (engram::load_chunks(chunks_path, chunk_map)) {
                spdlog::info("loaded {} chunks from disk", chunk_map.size());
                chunks_ok = true;
            } else {
                spdlog::warn("failed to load chunk store; will re-index");
                chunk_map.clear();
            }
        }

        // Both must succeed to skip re-indexing.
        loaded_from_disk = index_ok && chunks_ok;
        if (loaded_from_disk) {
            spdlog::info("index and chunk store loaded successfully; skipping re-index");
        }
    } else {
        spdlog::info("--reindex set; skipping persistence load");
    }

    // Embedder -- optional, only available when built with ONNX Runtime.
    engram::Embedder* embedder_ptr = nullptr;

#ifdef ENGRAM_HAS_ONNX
    std::unique_ptr<engram::OrtEmbedder> embedder;
    if (!model_path.empty()) {
        fs::path model_file(model_path);
        // Derive tokenizer path: same directory, tokenizer.json
        fs::path tokenizer_file = model_file.parent_path() / "tokenizer.json";

        std::error_code ec;
        if (fs::exists(model_file, ec)) {
            spdlog::info("loading embedder from '{}'", model_path);
            embedder = std::make_unique<engram::OrtEmbedder>(
                model_file.string(),
                tokenizer_file.string(),
                engram::DevicePreference::CUDA
            );
            if (embedder->is_valid()) {
                embedder_ptr = embedder.get();
                spdlog::info("embedder loaded: {} (dim={}, provider={})",
                             embedder->model_name(),
                             embedder->dimension(),
                             embedder->active_provider());
                // Update index dimension to match the model if needed.
                if (embedder->dimension() != embedding_dim) {
                    spdlog::info("adjusting index dimension from {} to {} to match model",
                                 embedding_dim, embedder->dimension());
                    embedding_dim = embedder->dimension();
                    vector_index = engram::HnswIndex(embedding_dim);
                }
            } else {
                spdlog::error("embedder failed to initialize");
                spdlog::warn("continuing without embedding support");
            }
        } else {
            spdlog::error("model file '{}' does not exist", model_path);
            spdlog::warn("continuing without embedding support");
        }
    }
#else
    spdlog::info("built without ONNX Runtime; semantic search is disabled");
    if (!model_path.empty()) {
        spdlog::warn("--model was specified but ENGRAM_HAS_ONNX is not defined; "
                     "rebuild with -DENGRAM_USE_ONNX=ON to enable embedding");
    }
#endif

    // -----------------------------------------------------------------------
    // Initial project indexing
    // -----------------------------------------------------------------------
    if (!project_path.empty() && !loaded_from_disk) {
        fs::path project_root(project_path);
        std::error_code ec;

        if (!fs::is_directory(project_root, ec)) {
            spdlog::error("project path '{}' is not a directory", project_path);
        } else {
            spdlog::info("starting initial project indexing...");

            engram::RegexChunker chunker;
            size_t num_chunks = index_project(
                project_root, chunker, chunk_map, vector_index, embedder_ptr
            );

            spdlog::info("chunk metadata store contains {} entries", chunk_map.size());

            // Save the index to disk after initial indexing.
            if (num_chunks > 0 && embedder_ptr != nullptr) {
                std::error_code ec2;
                fs::create_directories(index_path, ec2);
                if (vector_index.save(index_path)) {
                    spdlog::info("index saved to '{}'", index_path.generic_string());
                } else {
                    spdlog::error("failed to save index to '{}'", index_path.generic_string());
                }
            } else if (num_chunks > 0) {
                spdlog::info("chunks created but no embedder available; "
                             "index not saved (no embeddings to persist)");
            }

            // Save chunk metadata to disk after initial indexing.
            if (num_chunks > 0) {
                if (engram::save_chunks(chunks_path, chunk_map)) {
                    spdlog::info("chunk store saved to '{}'", chunks_path.generic_string());
                } else {
                    spdlog::error("failed to save chunk store to '{}'",
                                  chunks_path.generic_string());
                }
            }
        }
    } else if (!project_path.empty() && loaded_from_disk) {
        spdlog::info("project indexing skipped (loaded from disk)");
    }

    // -----------------------------------------------------------------------
    // Shared mutex for thread-safe access to chunk_map from the watcher
    // callback (background thread) and MCP tool handlers (main thread).
    // -----------------------------------------------------------------------
    std::mutex index_mutex;

    // -----------------------------------------------------------------------
    // File watcher -- incremental re-indexing on file changes
    // -----------------------------------------------------------------------
    engram::WinFileWatcher watcher;

    if (!project_path.empty()) {
        fs::path project_root(project_path);
        std::error_code ec;

        if (fs::is_directory(project_root, ec)) {
            auto& exts      = supported_extensions();
            auto& skip_dirs = skip_directories();

            bool started = watcher.start(project_root,
                [&index_mutex, &chunk_map, &vector_index, embedder_ptr,
                 &exts, &skip_dirs, project_root]
                (const std::vector<engram::FileChange>& changes)
            {
                engram::RegexChunker chunker;

                for (const auto& change : changes) {
                    const auto& file_path = change.path;

                    // --- Filter: only supported extensions ---
                    auto ext = file_path.extension().string();
                    for (auto& ch : ext) {
                        ch = static_cast<char>(
                            std::tolower(static_cast<unsigned char>(ch)));
                    }
                    if (exts.count(ext) == 0) {
                        continue;
                    }

                    // --- Filter: skip ignored directories ---
                    {
                        bool skip = false;
                        auto rel = fs::relative(file_path, project_root);
                        for (const auto& component : rel) {
                            if (skip_dirs.count(component.string())) {
                                skip = true;
                                break;
                            }
                        }
                        if (skip) {
                            continue;
                        }
                    }

                    // --- Handle the event ---
                    switch (change.event) {
                    case engram::FileEvent::Created:
                    case engram::FileEvent::Modified:
                    case engram::FileEvent::Renamed: {
                        auto new_chunks = chunker.chunk_file(file_path);
                        if (new_chunks.empty()) {
                            spdlog::debug("watcher: no chunks from '{}'",
                                          file_path.generic_string());
                            break;
                        }

                        std::lock_guard<std::mutex> lock(index_mutex);

                        // Remove old chunks for this file first.
                        std::vector<std::string> old_ids;
                        for (const auto& [id, chunk] : chunk_map) {
                            if (chunk.file_path == file_path) {
                                old_ids.push_back(id);
                            }
                        }
                        for (const auto& id : old_ids) {
                            chunk_map.erase(id);
                            vector_index.remove(id);
                        }

                        // Insert new chunks.
                        size_t embedded = 0;
                        for (auto& chunk : new_chunks) {
                            const auto& id = chunk.chunk_id;
                            chunk_map[id] = chunk;

                            if (embedder_ptr) {
                                auto embedding = embedder_ptr->embed(
                                    chunk.source_text);
                                if (!embedding.empty()) {
                                    vector_index.add(id, embedding.data(),
                                                     embedding.size());
                                    embedded++;
                                }
                            }
                        }

                        spdlog::info("watcher: re-indexed '{}' ({} chunks, {} embedded)",
                                     file_path.generic_string(),
                                     new_chunks.size(), embedded);
                        break;
                    }

                    case engram::FileEvent::Deleted: {
                        std::lock_guard<std::mutex> lock(index_mutex);

                        std::vector<std::string> old_ids;
                        for (const auto& [id, chunk] : chunk_map) {
                            if (chunk.file_path == file_path) {
                                old_ids.push_back(id);
                            }
                        }
                        for (const auto& id : old_ids) {
                            chunk_map.erase(id);
                            vector_index.remove(id);
                        }

                        if (!old_ids.empty()) {
                            spdlog::info("watcher: removed {} chunks for deleted '{}'",
                                         old_ids.size(),
                                         file_path.generic_string());
                        }
                        break;
                    }
                    } // switch
                } // for each change
            }); // watcher callback

            if (started) {
                spdlog::info("file watcher started on '{}'",
                             project_root.generic_string());
            } else {
                spdlog::warn("failed to start file watcher on '{}'",
                             project_root.generic_string());
            }
        }
    }

    // -----------------------------------------------------------------------
    // Start MCP server
    // -----------------------------------------------------------------------
    spdlog::info("starting MCP server (stdio mode)...");

    engram::mcp::McpServer server;

    // Build ToolContext with references to all backend components.
    engram::mcp::ToolContext tool_context;
    tool_context.embedder      = embedder_ptr;
    tool_context.index         = &vector_index;
    tool_context.session_store = &session_store;
    tool_context.chunk_store   = &chunk_map;
    tool_context.project_root  = project_path;
    tool_context.shared_mutex  = &index_mutex;

    engram::mcp::register_all_tools(server, tool_context);

    // This call blocks, reading JSON-RPC messages from stdin and writing
    // responses to stdout, until stdin EOF or server.stop() is called.
    server.run();

    // -----------------------------------------------------------------------
    // Clean shutdown
    // -----------------------------------------------------------------------
    spdlog::info("MCP server stopped; performing clean shutdown...");

    // Stop the file watcher before touching shared state.
    if (watcher.is_watching()) {
        watcher.stop();
        spdlog::info("file watcher stopped");
    }

    // Save the index to disk on exit.
    if (vector_index.size() > 0) {
        std::error_code ec;
        fs::create_directories(index_path, ec);
        if (vector_index.save(index_path)) {
            spdlog::info("index saved to '{}' ({} vectors)",
                         index_path.generic_string(), vector_index.size());
        } else {
            spdlog::error("failed to save index on shutdown");
        }
    }

    // Save chunk metadata to disk on exit.
    if (!chunk_map.empty()) {
        if (engram::save_chunks(chunks_path, chunk_map)) {
            spdlog::info("chunk store saved to '{}' ({} chunks)",
                         chunks_path.generic_string(), chunk_map.size());
        } else {
            spdlog::error("failed to save chunk store on shutdown");
        }
    }

    spdlog::info("engram-mcp shut down cleanly");
    return 0;
}
