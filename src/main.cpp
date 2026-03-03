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
#include <fstream>
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

#ifdef ENGRAM_HAS_TREESITTER
#include "chunker/treesitter_chunker.hpp"
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
    spdlog::info("  --batch-size <int>  Batch size for GPU embedding (default: 32)");
    spdlog::info("  --reindex           Force a full re-index of the project");
    spdlog::info("  --treesitter        Use tree-sitter chunker (requires ENGRAM_USE_TREESITTER build)");
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
        ".idea",     // JetBrains
        ".venv", "venv", "env"  // Python virtual environments
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
// Content hashing for incremental re-indexing
// =========================================================================

/// Compute an FNV-1a 64-bit hash of a file's contents and return it as a
/// 16-character lowercase hex string.  Returns an empty string on read error.
static std::string hash_file_content(const fs::path& file_path) {
    std::ifstream ifs(file_path, std::ios::binary);
    if (!ifs.is_open()) return {};

    constexpr uint64_t fnv_offset = 14695981039346656037ULL;
    constexpr uint64_t fnv_prime  = 1099511628211ULL;
    uint64_t hash = fnv_offset;

    char buf[8192];
    while (ifs.read(buf, sizeof(buf)) || ifs.gcount() > 0) {
        auto n = ifs.gcount();
        for (std::streamsize i = 0; i < n; ++i) {
            hash ^= static_cast<uint64_t>(static_cast<unsigned char>(buf[i]));
            hash *= fnv_prime;
        }
    }

    char hex[17];
    snprintf(hex, sizeof(hex), "%016llx", static_cast<unsigned long long>(hash));
    return std::string(hex, 16);
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
    engram::Chunker& chunker,
    std::unordered_map<std::string, engram::Chunk>& chunk_map,
    engram::HnswIndex& index,
    engram::Embedder* embedder_ptr,
    size_t batch_size = 32)
{
    auto start_time = std::chrono::steady_clock::now();

    auto files = walk_project_files(project_root);
    spdlog::info("found {} source files to index", files.size());

    size_t total_chunks = 0;
    size_t files_processed = 0;
    size_t chunks_embedded = 0;

    // Batch buffer: accumulate (chunk_id, source_text) pairs for batched GPU embedding.
    std::vector<std::string> batch_ids;
    std::vector<std::string> batch_texts;

    // Flush the current batch through embed_batch() and insert into the index.
    auto flush_batch = [&]() {
        if (batch_ids.empty() || !embedder_ptr) return;

        auto embeddings = embedder_ptr->embed_batch(batch_texts);

        for (size_t i = 0; i < embeddings.size(); ++i) {
            if (!embeddings[i].empty()) {
                if (index.add(batch_ids[i], embeddings[i].data(), embeddings[i].size())) {
                    chunks_embedded++;
                } else {
                    spdlog::warn("failed to add chunk '{}' to index", batch_ids[i]);
                }
            } else {
                spdlog::warn("embedding failed for chunk '{}'", batch_ids[i]);
            }
        }

        batch_ids.clear();
        batch_texts.clear();
    };

    for (const auto& file_path : files) {
        auto chunks = chunker.chunk_file(file_path);

        if (chunks.empty()) {
            spdlog::debug("no chunks from '{}'", file_path.generic_string());
            continue;
        }

        files_processed++;

        // Compute content hash for this file (used by incremental re-indexing).
        auto file_hash = hash_file_content(file_path);

        for (auto& chunk : chunks) {
            chunk.file_content_hash = file_hash;
            const auto& id = chunk.chunk_id;

            // Store metadata.
            chunk_map[id] = chunk;
            total_chunks++;

            // Queue for batch embedding if embedder is available.
            if (embedder_ptr) {
                batch_ids.push_back(id);
                batch_texts.push_back(chunk.source_text);

                if (batch_ids.size() >= batch_size) {
                    flush_batch();
                }
            }
        }

        // Log progress every 50 files.
        if (files_processed % 50 == 0) {
            spdlog::info("  progress: {}/{} files, {} chunks so far",
                         files_processed, files.size(), total_chunks);
        }
    }

    // Flush any remaining chunks in the final partial batch.
    flush_batch();

    auto elapsed = std::chrono::steady_clock::now() - start_time;
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

    spdlog::info("indexing complete:");
    spdlog::info("  files processed:  {}", files_processed);
    spdlog::info("  chunks created:   {}", total_chunks);
    if (chunks_embedded > 0) {
        spdlog::info("  chunks embedded:  {}", chunks_embedded);
    }
    if (embedder_ptr) {
        spdlog::info("  batch size:       {}", batch_size);
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
    // Logger -- everything goes to stderr; stdout is reserved for MCP.
    // -----------------------------------------------------------------------
    auto stderr_sink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
    auto logger = std::make_shared<spdlog::logger>("engram", stderr_sink);
    logger->set_level(spdlog::level::info);
    logger->flush_on(spdlog::level::info);
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

    const std::string project_path   = parse_arg(args, "--project");
    const std::string model_path     = parse_arg(args, "--model");
    const std::string dim_str        = parse_arg(args, "--dim", "384");
    const std::string batch_size_str = parse_arg(args, "--batch-size", "32");
    const bool        force_reindex  = has_flag(args, "--reindex");
    const bool        use_treesitter = has_flag(args, "--treesitter");

    // Parse embedding dimension.
    size_t embedding_dim = 384;
    try {
        embedding_dim = static_cast<size_t>(std::stoul(dim_str));
    } catch (...) {
        spdlog::error("invalid --dim value '{}', using default 384", dim_str);
    }

    // Parse batch size.
    size_t batch_size = 32;
    try {
        batch_size = static_cast<size_t>(std::stoul(batch_size_str));
        if (batch_size == 0) batch_size = 1;
    } catch (...) {
        spdlog::error("invalid --batch-size value '{}', using default 32", batch_size_str);
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
    // Create chunker (polymorphic: tree-sitter or regex fallback)
    // -----------------------------------------------------------------------
    std::unique_ptr<engram::Chunker> chunker;
#ifdef ENGRAM_HAS_TREESITTER
    if (use_treesitter) {
        spdlog::info("using tree-sitter chunker (AST-aware)");
        chunker = std::make_unique<engram::TreeSitterChunker>();
    } else
#endif
    {
        if (use_treesitter) {
            spdlog::warn("--treesitter requested but ENGRAM_HAS_TREESITTER is not defined; "
                         "falling back to regex chunker");
        }
        spdlog::info("using regex chunker");
        chunker = std::make_unique<engram::RegexChunker>();
    }

    // -----------------------------------------------------------------------
    // Initial project indexing (cold start or incremental warm restart)
    // -----------------------------------------------------------------------
    if (!project_path.empty()) {
        fs::path project_root(project_path);
        std::error_code ec;

        if (!fs::is_directory(project_root, ec)) {
            spdlog::error("project path '{}' is not a directory", project_path);
        } else if (!loaded_from_disk) {
            // Cold start: full index from scratch.
            spdlog::info("starting initial project indexing...");

            size_t num_chunks = index_project(
                project_root, *chunker, chunk_map, vector_index,
                embedder_ptr, batch_size
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
        } else {
            // Warm restart: incremental re-indexing using content hashes.
            spdlog::info("performing incremental re-index (checking content hashes)...");
            auto incr_start = std::chrono::steady_clock::now();

            auto files = walk_project_files(project_root);

            // Build a map of file_path -> stored content hash from existing chunks.
            // All chunks from the same file share the same hash.
            std::unordered_map<std::string, std::string> stored_hashes;
            for (const auto& [id, chunk] : chunk_map) {
                auto key = chunk.file_path.generic_string();
                if (!chunk.file_content_hash.empty() && stored_hashes.find(key) == stored_hashes.end()) {
                    stored_hashes[key] = chunk.file_content_hash;
                }
            }

            // Track which files currently exist on disk.
            std::unordered_set<std::string> files_on_disk;

            size_t files_unchanged = 0;
            size_t files_reindexed = 0;
            size_t files_removed   = 0;

            // Batch buffer for re-indexed chunks.
            std::vector<std::string> batch_ids;
            std::vector<std::string> batch_texts;
            size_t chunks_embedded = 0;

            auto flush_batch = [&]() {
                if (batch_ids.empty() || !embedder_ptr) return;
                auto embeddings = embedder_ptr->embed_batch(batch_texts);
                for (size_t i = 0; i < embeddings.size(); ++i) {
                    if (!embeddings[i].empty()) {
                        if (vector_index.add(batch_ids[i], embeddings[i].data(), embeddings[i].size())) {
                            chunks_embedded++;
                        }
                    }
                }
                batch_ids.clear();
                batch_texts.clear();
            };

            for (const auto& file_path : files) {
                auto file_key = file_path.generic_string();
                files_on_disk.insert(file_key);

                auto current_hash = hash_file_content(file_path);
                auto it = stored_hashes.find(file_key);

                // If hash matches and we have a stored hash, skip this file.
                if (it != stored_hashes.end() && !it->second.empty() && it->second == current_hash) {
                    files_unchanged++;
                    continue;
                }

                // File is new or changed — re-chunk and re-embed.
                files_reindexed++;

                // Remove old chunks for this file.
                std::vector<std::string> old_ids;
                for (const auto& [id, chunk] : chunk_map) {
                    if (chunk.file_path.generic_string() == file_key) {
                        old_ids.push_back(id);
                    }
                }
                for (const auto& id : old_ids) {
                    chunk_map.erase(id);
                    vector_index.remove(id);
                }

                // Re-chunk.
                auto new_chunks = chunker->chunk_file(file_path);
                for (auto& chunk : new_chunks) {
                    chunk.file_content_hash = current_hash;
                    chunk_map[chunk.chunk_id] = chunk;

                    if (embedder_ptr) {
                        batch_ids.push_back(chunk.chunk_id);
                        batch_texts.push_back(chunk.source_text);
                        if (batch_ids.size() >= batch_size) {
                            flush_batch();
                        }
                    }
                }
            }

            // Flush remaining batch.
            flush_batch();

            // Remove chunks for files that no longer exist.
            std::vector<std::string> orphan_ids;
            for (const auto& [id, chunk] : chunk_map) {
                if (files_on_disk.find(chunk.file_path.generic_string()) == files_on_disk.end()) {
                    orphan_ids.push_back(id);
                }
            }
            if (!orphan_ids.empty()) {
                // Count distinct files being removed.
                std::unordered_set<std::string> removed_files;
                for (const auto& id : orphan_ids) {
                    removed_files.insert(chunk_map[id].file_path.generic_string());
                    chunk_map.erase(id);
                    vector_index.remove(id);
                }
                files_removed = removed_files.size();
            }

            auto incr_elapsed = std::chrono::steady_clock::now() - incr_start;
            auto incr_ms = std::chrono::duration_cast<std::chrono::milliseconds>(incr_elapsed).count();

            spdlog::info("incremental re-index complete:");
            spdlog::info("  {} files unchanged, {} files re-indexed, {} files removed",
                         files_unchanged, files_reindexed, files_removed);
            if (chunks_embedded > 0) {
                spdlog::info("  {} chunks embedded", chunks_embedded);
            }
            spdlog::info("  elapsed: {} ms", incr_ms);

            // Persist if anything changed.
            if (files_reindexed > 0 || files_removed > 0) {
                if (embedder_ptr && vector_index.size() > 0) {
                    std::error_code ec2;
                    fs::create_directories(index_path, ec2);
                    if (vector_index.save(index_path)) {
                        spdlog::info("index saved to '{}'", index_path.generic_string());
                    }
                }
                if (!chunk_map.empty()) {
                    engram::save_chunks(chunks_path, chunk_map);
                }
            }
        }
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

            engram::Chunker* watcher_chunker = chunker.get();
            bool started = watcher.start(project_root,
                [&index_mutex, &chunk_map, &vector_index, embedder_ptr,
                 &exts, &skip_dirs, project_root, watcher_chunker]
                (const std::vector<engram::FileChange>& changes)
            {

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
                        // Chunk and embed outside the lock to avoid blocking
                        // MCP tool handlers during potentially slow GPU work.
                        auto new_chunks = watcher_chunker->chunk_file(file_path);
                        if (new_chunks.empty()) {
                            spdlog::debug("watcher: no chunks from '{}'",
                                          file_path.generic_string());
                            break;
                        }

                        // Stamp content hash on new chunks.
                        auto file_hash = hash_file_content(file_path);
                        for (auto& chunk : new_chunks) {
                            chunk.file_content_hash = file_hash;
                        }

                        // Batch-embed all new chunks for this file at once.
                        std::vector<std::vector<float>> embeddings;
                        if (embedder_ptr) {
                            std::vector<std::string> texts;
                            texts.reserve(new_chunks.size());
                            for (const auto& chunk : new_chunks) {
                                texts.push_back(chunk.source_text);
                            }
                            embeddings = embedder_ptr->embed_batch(texts);
                        }

                        // Now acquire the lock and update shared state.
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

                        // Insert new chunks and their embeddings.
                        size_t embedded = 0;
                        for (size_t i = 0; i < new_chunks.size(); ++i) {
                            const auto& id = new_chunks[i].chunk_id;
                            chunk_map[id] = new_chunks[i];

                            if (i < embeddings.size() && !embeddings[i].empty()) {
                                vector_index.add(id, embeddings[i].data(),
                                                 embeddings[i].size());
                                embedded++;
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
