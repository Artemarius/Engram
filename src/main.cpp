/// @file main.cpp
/// @brief Entry point for the Engram MCP server.
///
/// Parses command-line arguments, initializes all subsystems (chunker, vector
/// index, session store, optional embedder), performs initial project indexing,
/// and enters the MCP server loop that communicates with Claude Code over stdio.
///
/// Supports multiple projects via repeated --project flags or a .engram.toml
/// config file.  Each project gets its own data directory, index, chunk store,
/// session store, and file watcher.
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
#include "mcp/project_context.hpp"
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

/// Collect ALL values for a repeated --key from the argument list.
static std::vector<std::string> parse_all_values(const std::vector<std::string>& args,
                                                  const std::string& key)
{
    std::vector<std::string> values;
    for (size_t i = 0; i + 1 < args.size(); ++i) {
        if (args[i] == key) {
            values.push_back(args[i + 1]);
        }
    }
    return values;
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
    spdlog::info("  --project  <path>   Root of the codebase to index (repeatable)");
    spdlog::info("  --model    <path>   Path to the ONNX embedding model");
    spdlog::info("  --data-dir <path>   Directory for persistent data (single-project only)");
    spdlog::info("  --config   <path>   Path to .engram.toml config file");
    spdlog::info("  --dim      <int>    Embedding dimension (default: 384)");
    spdlog::info("  --batch-size <int>  Batch size for GPU embedding (default: 32)");
    spdlog::info("  --reindex           Force a full re-index of all projects");
    spdlog::info("  --treesitter        Use tree-sitter chunker (requires ENGRAM_USE_TREESITTER build)");
    spdlog::info("  --verbose           Enable debug-level logging");
    spdlog::info("  --help, -h          Show this help message");
}

// =========================================================================
// .engram.toml minimal parser
// =========================================================================

/// A project specification from CLI or TOML.
struct ProjectSpec {
    fs::path path;
    fs::path data_dir;  // empty = auto (path/.engram/)
};

/// Parse a minimal .engram.toml file.
///
/// Only handles:
///   [[project]]
///   path = "..."
///   data_dir = "..."   # optional
///   # comments
///
/// Returns parsed project specs.  On error, logs a warning and returns
/// whatever was parsed so far.
static std::vector<ProjectSpec> load_engram_toml(const fs::path& toml_path) {
    std::vector<ProjectSpec> specs;

    std::ifstream ifs(toml_path);
    if (!ifs.is_open()) {
        spdlog::debug("no .engram.toml at '{}'", toml_path.generic_string());
        return specs;
    }

    spdlog::info("loading config from '{}'", toml_path.generic_string());

    ProjectSpec* current = nullptr;
    std::string line;
    int line_num = 0;

    while (std::getline(ifs, line)) {
        ++line_num;

        // Trim leading/trailing whitespace.
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        size_t end = line.find_last_not_of(" \t\r\n");
        line = line.substr(start, end - start + 1);

        // Skip comments.
        if (line.empty() || line[0] == '#') continue;

        // [[project]] section header.
        if (line == "[[project]]") {
            specs.emplace_back();
            current = &specs.back();
            continue;
        }

        // key = "value" pairs.
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;

        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);

        // Trim whitespace.
        auto trim = [](std::string& s) {
            size_t a = s.find_first_not_of(" \t\"");
            size_t b = s.find_last_not_of(" \t\"");
            if (a == std::string::npos) { s.clear(); return; }
            s = s.substr(a, b - a + 1);
        };
        trim(key);
        trim(val);

        if (!current) {
            spdlog::warn(".engram.toml:{}: key '{}' outside [[project]] section", line_num, key);
            continue;
        }

        if (key == "path") {
            current->path = fs::path(val);
        } else if (key == "data_dir") {
            current->data_dir = fs::path(val);
        }
    }

    spdlog::info("loaded {} project(s) from .engram.toml", specs.size());
    return specs;
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

        if (entry.is_directory()) {
            const auto dirname = entry.path().filename().string();
            if (skip_dirs.count(dirname)) {
                it.disable_recursion_pending();
                continue;
            }
            continue;
        }

        if (!entry.is_regular_file()) continue;

        auto ext = entry.path().extension().string();
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

    std::vector<std::string> batch_ids;
    std::vector<std::string> batch_texts;

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

        auto file_hash = hash_file_content(file_path);

        for (auto& chunk : chunks) {
            chunk.file_content_hash = file_hash;
            const auto& id = chunk.chunk_id;

            chunk_map[id] = chunk;
            total_chunks++;

            if (embedder_ptr) {
                batch_ids.push_back(id);
                batch_texts.push_back(chunk.source_text);

                if (batch_ids.size() >= batch_size) {
                    flush_batch();
                }
            }
        }

        if (files_processed % 50 == 0) {
            spdlog::info("  progress: {}/{} files, {} chunks so far",
                         files_processed, files.size(), total_chunks);
        }
    }

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
// Incremental re-indexing (warm restart)
// =========================================================================

/// Perform incremental re-indexing on a project using content hashes.
/// Updates chunk_map and vector_index in place.
static void incremental_reindex(
    engram::ProjectContext& proj,
    engram::Chunker& chunker,
    engram::Embedder* embedder_ptr,
    size_t batch_size)
{
    spdlog::info("[{}] performing incremental re-index (checking content hashes)...",
                 proj.name);
    auto incr_start = std::chrono::steady_clock::now();

    auto files = walk_project_files(proj.project_root);

    // Build a map of file_path -> stored content hash from existing chunks.
    std::unordered_map<std::string, std::string> stored_hashes;
    for (const auto& [id, chunk] : proj.chunk_map) {
        auto key = chunk.file_path.generic_string();
        if (!chunk.file_content_hash.empty() && stored_hashes.find(key) == stored_hashes.end()) {
            stored_hashes[key] = chunk.file_content_hash;
        }
    }

    std::unordered_set<std::string> files_on_disk;

    size_t files_unchanged = 0;
    size_t files_reindexed = 0;
    size_t files_removed   = 0;

    std::vector<std::string> batch_ids;
    std::vector<std::string> batch_texts;
    size_t chunks_embedded = 0;

    auto flush_batch = [&]() {
        if (batch_ids.empty() || !embedder_ptr) return;
        auto embeddings = embedder_ptr->embed_batch(batch_texts);
        for (size_t i = 0; i < embeddings.size(); ++i) {
            if (!embeddings[i].empty()) {
                if (proj.vector_index.add(batch_ids[i], embeddings[i].data(), embeddings[i].size())) {
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

        if (it != stored_hashes.end() && !it->second.empty() && it->second == current_hash) {
            files_unchanged++;
            continue;
        }

        files_reindexed++;

        // Remove old chunks for this file.
        std::vector<std::string> old_ids;
        for (const auto& [id, chunk] : proj.chunk_map) {
            if (chunk.file_path.generic_string() == file_key) {
                old_ids.push_back(id);
            }
        }
        for (const auto& id : old_ids) {
            proj.chunk_map.erase(id);
            proj.vector_index.remove(id);
        }

        // Re-chunk.
        auto new_chunks = chunker.chunk_file(file_path);
        for (auto& chunk : new_chunks) {
            chunk.file_content_hash = current_hash;
            proj.chunk_map[chunk.chunk_id] = chunk;

            if (embedder_ptr) {
                batch_ids.push_back(chunk.chunk_id);
                batch_texts.push_back(chunk.source_text);
                if (batch_ids.size() >= batch_size) {
                    flush_batch();
                }
            }
        }
    }

    flush_batch();

    // Remove chunks for files that no longer exist.
    std::vector<std::string> orphan_ids;
    for (const auto& [id, chunk] : proj.chunk_map) {
        if (files_on_disk.find(chunk.file_path.generic_string()) == files_on_disk.end()) {
            orphan_ids.push_back(id);
        }
    }
    if (!orphan_ids.empty()) {
        std::unordered_set<std::string> removed_files;
        for (const auto& id : orphan_ids) {
            removed_files.insert(proj.chunk_map[id].file_path.generic_string());
            proj.chunk_map.erase(id);
            proj.vector_index.remove(id);
        }
        files_removed = removed_files.size();
    }

    auto incr_elapsed = std::chrono::steady_clock::now() - incr_start;
    auto incr_ms = std::chrono::duration_cast<std::chrono::milliseconds>(incr_elapsed).count();

    spdlog::info("[{}] incremental re-index complete:", proj.name);
    spdlog::info("  {} files unchanged, {} files re-indexed, {} files removed",
                 files_unchanged, files_reindexed, files_removed);
    if (chunks_embedded > 0) {
        spdlog::info("  {} chunks embedded", chunks_embedded);
    }
    spdlog::info("  elapsed: {} ms", incr_ms);

    // Persist if anything changed.
    if (files_reindexed > 0 || files_removed > 0) {
        if (embedder_ptr && proj.vector_index.size() > 0) {
            std::error_code ec;
            fs::create_directories(proj.index_path, ec);
            if (proj.vector_index.save(proj.index_path)) {
                spdlog::info("[{}] index saved to '{}'", proj.name, proj.index_path.generic_string());
            }
        }
        if (!proj.chunk_map.empty()) {
            engram::save_chunks(proj.chunks_path, proj.chunk_map);
        }
    }
}

// =========================================================================
// File watcher setup
// =========================================================================

/// Start the file watcher for a project.
static void start_project_watcher(
    engram::ProjectContext& proj,
    engram::Chunker* chunker_ptr,
    engram::Embedder* embedder_ptr)
{
    std::error_code ec;
    if (!fs::is_directory(proj.project_root, ec)) return;

    auto& exts      = supported_extensions();
    auto& skip_dirs = skip_directories();

    bool started = proj.watcher.start(proj.project_root,
        [&proj, embedder_ptr, &exts, &skip_dirs, chunker_ptr]
        (const std::vector<engram::FileChange>& changes)
    {
        for (const auto& change : changes) {
            const auto& file_path = change.path;

            auto ext = file_path.extension().string();
            for (auto& ch : ext) {
                ch = static_cast<char>(
                    std::tolower(static_cast<unsigned char>(ch)));
            }
            if (exts.count(ext) == 0) continue;

            {
                bool skip = false;
                auto rel = fs::relative(file_path, proj.project_root);
                for (const auto& component : rel) {
                    if (skip_dirs.count(component.string())) {
                        skip = true;
                        break;
                    }
                }
                if (skip) continue;
            }

            switch (change.event) {
            case engram::FileEvent::Created:
            case engram::FileEvent::Modified:
            case engram::FileEvent::Renamed: {
                auto new_chunks = chunker_ptr->chunk_file(file_path);
                if (new_chunks.empty()) {
                    spdlog::debug("[{}] watcher: no chunks from '{}'",
                                  proj.name, file_path.generic_string());
                    break;
                }

                auto file_hash = hash_file_content(file_path);
                for (auto& chunk : new_chunks) {
                    chunk.file_content_hash = file_hash;
                }

                std::vector<std::vector<float>> embeddings;
                if (embedder_ptr) {
                    std::vector<std::string> texts;
                    texts.reserve(new_chunks.size());
                    for (const auto& chunk : new_chunks) {
                        texts.push_back(chunk.source_text);
                    }
                    embeddings = embedder_ptr->embed_batch(texts);
                }

                std::lock_guard<std::mutex> lock(proj.index_mutex);

                std::vector<std::string> old_ids;
                for (const auto& [id, chunk] : proj.chunk_map) {
                    if (chunk.file_path == file_path) {
                        old_ids.push_back(id);
                    }
                }
                for (const auto& id : old_ids) {
                    proj.chunk_map.erase(id);
                    proj.vector_index.remove(id);
                }

                size_t embedded = 0;
                for (size_t i = 0; i < new_chunks.size(); ++i) {
                    const auto& id = new_chunks[i].chunk_id;
                    proj.chunk_map[id] = new_chunks[i];

                    if (i < embeddings.size() && !embeddings[i].empty()) {
                        proj.vector_index.add(id, embeddings[i].data(),
                                             embeddings[i].size());
                        embedded++;
                    }
                }

                spdlog::info("[{}] watcher: re-indexed '{}' ({} chunks, {} embedded)",
                             proj.name, file_path.generic_string(),
                             new_chunks.size(), embedded);
                break;
            }

            case engram::FileEvent::Deleted: {
                std::lock_guard<std::mutex> lock(proj.index_mutex);

                std::vector<std::string> old_ids;
                for (const auto& [id, chunk] : proj.chunk_map) {
                    if (chunk.file_path == file_path) {
                        old_ids.push_back(id);
                    }
                }
                for (const auto& id : old_ids) {
                    proj.chunk_map.erase(id);
                    proj.vector_index.remove(id);
                }

                if (!old_ids.empty()) {
                    spdlog::info("[{}] watcher: removed {} chunks for deleted '{}'",
                                 proj.name, old_ids.size(),
                                 file_path.generic_string());
                }
                break;
            }
            } // switch
        } // for each change
    }); // watcher callback

    if (started) {
        spdlog::info("[{}] file watcher started on '{}'",
                     proj.name, proj.project_root.generic_string());
    } else {
        spdlog::warn("[{}] failed to start file watcher on '{}'",
                     proj.name, proj.project_root.generic_string());
    }
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

    const auto project_paths     = parse_all_values(args, "--project");
    const std::string model_path     = parse_arg(args, "--model");
    const std::string dim_str        = parse_arg(args, "--dim", "384");
    const std::string batch_size_str = parse_arg(args, "--batch-size", "32");
    const std::string config_path    = parse_arg(args, "--config");
    const std::string data_dir_str   = parse_arg(args, "--data-dir");
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

    // -----------------------------------------------------------------------
    // Phase 1: Collect project specs from CLI + TOML
    // -----------------------------------------------------------------------
    std::vector<ProjectSpec> specs;

    // CLI projects.
    for (const auto& p : project_paths) {
        ProjectSpec spec;
        spec.path = fs::path(p);
        specs.push_back(std::move(spec));
    }

    // --data-dir applies only to single-project mode.
    if (!data_dir_str.empty()) {
        if (specs.size() > 1) {
            spdlog::warn("--data-dir ignored with multiple --project flags");
        } else if (specs.size() == 1) {
            specs[0].data_dir = fs::path(data_dir_str);
        }
    }

    // TOML config.
    {
        fs::path toml_file;
        if (!config_path.empty()) {
            toml_file = fs::path(config_path);
        } else {
            // Default: .engram.toml in cwd.
            toml_file = fs::current_path() / ".engram.toml";
        }

        std::error_code ec;
        if (fs::exists(toml_file, ec)) {
            auto toml_specs = load_engram_toml(toml_file);
            for (auto& ts : toml_specs) {
                specs.push_back(std::move(ts));
            }
        } else if (!config_path.empty()) {
            spdlog::warn("config file '{}' not found", config_path);
        }
    }

    // Deduplicate by canonical path (CLI wins on conflict).
    {
        std::unordered_set<std::string> seen;
        std::vector<ProjectSpec> deduped;
        for (auto& spec : specs) {
            std::error_code ec;
            auto canonical = fs::canonical(spec.path, ec);
            std::string key = ec ? spec.path.generic_string() : canonical.generic_string();
            if (seen.count(key) == 0) {
                seen.insert(key);
                deduped.push_back(std::move(spec));
            } else {
                spdlog::debug("skipping duplicate project '{}'", key);
            }
        }
        specs = std::move(deduped);
    }

    // -----------------------------------------------------------------------
    // Startup banner (stderr only)
    // -----------------------------------------------------------------------
    spdlog::info("engram-mcp starting up");
    spdlog::info("  build:     {} {}", __DATE__, __TIME__);
    spdlog::info("  dim:       {}", embedding_dim);

    if (specs.empty()) {
        spdlog::warn("  no projects specified; indexing disabled");
    } else {
        spdlog::info("  projects:  {}", specs.size());
        for (const auto& spec : specs) {
            spdlog::info("    - {}", spec.path.generic_string());
        }
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
    // Embedder -- optional, only available when built with ONNX Runtime.
    // Initialized before projects so we can adjust embedding_dim.
    // -----------------------------------------------------------------------
    engram::Embedder* embedder_ptr = nullptr;

#ifdef ENGRAM_HAS_ONNX
    std::unique_ptr<engram::OrtEmbedder> embedder;
    if (!model_path.empty()) {
        fs::path model_file(model_path);
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
                if (embedder->dimension() != embedding_dim) {
                    spdlog::info("adjusting index dimension from {} to {} to match model",
                                 embedding_dim, embedder->dimension());
                    embedding_dim = embedder->dimension();
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
    // Phase 2: Initialize ProjectContext for each project
    // -----------------------------------------------------------------------
    std::vector<std::unique_ptr<engram::ProjectContext>> projects;

    for (const auto& spec : specs) {
        auto proj = std::make_unique<engram::ProjectContext>(embedding_dim);

        // Resolve project root.
        std::error_code ec;
        proj->project_root = fs::canonical(spec.path, ec);
        if (ec) {
            spdlog::error("cannot resolve project path '{}': {}",
                          spec.path.generic_string(), ec.message());
            continue;
        }

        if (!fs::is_directory(proj->project_root, ec)) {
            spdlog::error("project path '{}' is not a directory",
                          spec.path.generic_string());
            continue;
        }

        // Display name = last path component.
        proj->name = proj->project_root.filename().string();

        // Resolve data directory.
        if (!spec.data_dir.empty()) {
            proj->data_dir = spec.data_dir;
        } else {
            proj->data_dir = proj->project_root / ".engram";
        }

        proj->index_path  = proj->data_dir / "index";
        proj->chunks_path = proj->data_dir / "chunks.json";

        // Create data directories.
        fs::create_directories(proj->data_dir, ec);
        if (ec) {
            spdlog::error("[{}] failed to create data directory '{}': {}",
                          proj->name, proj->data_dir.generic_string(), ec.message());
            continue;
        }

        fs::create_directories(proj->data_dir / "sessions", ec);
        if (ec) {
            spdlog::error("[{}] failed to create sessions directory: {}",
                          proj->name, ec.message());
            continue;
        }

        // Session store.
        proj->session_store = std::make_unique<engram::SessionStore>(
            proj->data_dir / "sessions");
        spdlog::info("[{}] session store initialized at '{}'",
                     proj->name,
                     proj->session_store->storage_directory().generic_string());

        spdlog::info("[{}] data-dir: {}", proj->name, proj->data_dir.generic_string());

        projects.push_back(std::move(proj));
    }

    // -----------------------------------------------------------------------
    // Phase 3: Load from disk / Index each project
    // -----------------------------------------------------------------------
    for (auto& proj : projects) {
        bool loaded_from_disk = false;

        if (!force_reindex) {
            bool index_ok  = false;
            bool chunks_ok = false;

            std::error_code ec;
            if (fs::exists(proj->index_path, ec)) {
                spdlog::info("[{}] loading existing index from '{}'",
                             proj->name, proj->index_path.generic_string());
                if (proj->vector_index.load(proj->index_path)) {
                    spdlog::info("[{}] loaded index with {} vectors",
                                 proj->name, proj->vector_index.size());
                    index_ok = true;
                } else {
                    spdlog::warn("[{}] failed to load index; starting fresh", proj->name);
                }
            }

            if (index_ok && fs::exists(proj->chunks_path, ec)) {
                spdlog::info("[{}] loading chunk store from '{}'",
                             proj->name, proj->chunks_path.generic_string());
                if (engram::load_chunks(proj->chunks_path, proj->chunk_map)) {
                    spdlog::info("[{}] loaded {} chunks from disk",
                                 proj->name, proj->chunk_map.size());
                    chunks_ok = true;
                } else {
                    spdlog::warn("[{}] failed to load chunk store; will re-index", proj->name);
                    proj->chunk_map.clear();
                }
            }

            loaded_from_disk = index_ok && chunks_ok;
        } else {
            spdlog::info("[{}] --reindex set; skipping persistence load", proj->name);
        }

        if (!loaded_from_disk) {
            // Cold start: full index from scratch.
            spdlog::info("[{}] starting initial project indexing...", proj->name);

            size_t num_chunks = index_project(
                proj->project_root, *chunker, proj->chunk_map, proj->vector_index,
                embedder_ptr, batch_size
            );

            spdlog::info("[{}] chunk metadata store contains {} entries",
                         proj->name, proj->chunk_map.size());

            // Save index to disk.
            if (num_chunks > 0 && embedder_ptr != nullptr) {
                std::error_code ec;
                fs::create_directories(proj->index_path, ec);
                if (proj->vector_index.save(proj->index_path)) {
                    spdlog::info("[{}] index saved to '{}'",
                                 proj->name, proj->index_path.generic_string());
                } else {
                    spdlog::error("[{}] failed to save index", proj->name);
                }
            }

            // Save chunk metadata to disk.
            if (num_chunks > 0) {
                if (engram::save_chunks(proj->chunks_path, proj->chunk_map)) {
                    spdlog::info("[{}] chunk store saved to '{}'",
                                 proj->name, proj->chunks_path.generic_string());
                } else {
                    spdlog::error("[{}] failed to save chunk store", proj->name);
                }
            }
        } else {
            // Warm restart: incremental re-indexing using content hashes.
            incremental_reindex(*proj, *chunker, embedder_ptr, batch_size);
        }
    }

    // -----------------------------------------------------------------------
    // Phase 4: Start file watchers for each project
    // -----------------------------------------------------------------------
    for (auto& proj : projects) {
        start_project_watcher(*proj, chunker.get(), embedder_ptr);
    }

    // -----------------------------------------------------------------------
    // Start MCP server
    // -----------------------------------------------------------------------
    spdlog::info("starting MCP server (stdio mode)...");

    engram::mcp::McpServer server;

    // Build ToolContext with references to all backend components.
    engram::mcp::ToolContext tool_context;
    tool_context.embedder = embedder_ptr;
    tool_context.projects = &projects;

    // Primary session store = first project's store.
    if (!projects.empty() && projects[0]->session_store) {
        tool_context.session_store = projects[0]->session_store.get();
    }

    engram::mcp::register_all_tools(server, tool_context);

    // This call blocks, reading JSON-RPC messages from stdin and writing
    // responses to stdout, until stdin EOF or server.stop() is called.
    server.run();

    // -----------------------------------------------------------------------
    // Phase 5: Clean shutdown
    // -----------------------------------------------------------------------
    spdlog::info("MCP server stopped; performing clean shutdown...");

    // Stop all file watchers.
    for (auto& proj : projects) {
        if (proj->watcher.is_watching()) {
            proj->watcher.stop();
            spdlog::info("[{}] file watcher stopped", proj->name);
        }
    }

    // Save all indices and chunk stores.
    for (auto& proj : projects) {
        if (proj->vector_index.size() > 0) {
            std::error_code ec;
            fs::create_directories(proj->index_path, ec);
            if (proj->vector_index.save(proj->index_path)) {
                spdlog::info("[{}] index saved ({} vectors)",
                             proj->name, proj->vector_index.size());
            } else {
                spdlog::error("[{}] failed to save index on shutdown", proj->name);
            }
        }

        if (!proj->chunk_map.empty()) {
            if (engram::save_chunks(proj->chunks_path, proj->chunk_map)) {
                spdlog::info("[{}] chunk store saved ({} chunks)",
                             proj->name, proj->chunk_map.size());
            } else {
                spdlog::error("[{}] failed to save chunk store on shutdown", proj->name);
            }
        }
    }

    spdlog::info("engram-mcp shut down cleanly");
    return 0;
}
