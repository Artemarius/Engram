/// @file bench_chunker.cpp
/// @brief Standalone chunker and embedding performance benchmark for Engram.
///
/// Walks a project directory, chunks all source files with each available
/// chunker implementation, and reports timing statistics.  When --model is
/// provided, also benchmarks GPU embedding throughput and query latency.
///
/// Usage:
///   engram_benchmarks --project <path> [--iterations N] [--model <path>]
///                     [--batch-size N] [--warmup N] [--queries N]
///
/// Output goes to stdout (this is a standalone tool, NOT an MCP server).

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

#include "chunker/chunker.hpp"
#include "chunker/regex_chunker.hpp"

#ifdef ENGRAM_HAS_TREESITTER
#include "chunker/treesitter_chunker.hpp"
#endif

#ifdef ENGRAM_HAS_ONNX
#include "embedder/ort_embedder.hpp"
#include "index/hnsw_index.hpp"
#endif

namespace fs = std::filesystem;

// =========================================================================
// CLI argument parsing
// =========================================================================

/// Parse a simple --key value pair from the argument list.
static std::string parse_arg(const std::vector<std::string>& args,
                             const std::string& key,
                             const std::string& fallback = {})
{
    for (size_t i = 0; i + 1 < args.size(); ++i) {
        if (args[i] == key) return args[i + 1];
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

// =========================================================================
// File extension and directory filters
// =========================================================================

/// Set of file extensions (lowercase, including leading dot) supported for
/// code chunking.  Mirrors the set in main.cpp.
static const std::vector<std::string>& supported_extensions() {
    static const std::vector<std::string> exts = {
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
/// Mirrors the set in main.cpp.
static const std::vector<std::string>& skip_directories() {
    static const std::vector<std::string> dirs = {
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

/// Check if an extension is in the supported set (case-insensitive).
static bool is_supported_extension(const std::string& ext) {
    std::string lower_ext = ext;
    for (auto& ch : lower_ext) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    const auto& exts = supported_extensions();
    return std::find(exts.begin(), exts.end(), lower_ext) != exts.end();
}

/// Check if a directory name should be skipped.
static bool is_skip_directory(const std::string& dirname) {
    const auto& dirs = skip_directories();
    return std::find(dirs.begin(), dirs.end(), dirname) != dirs.end();
}

// =========================================================================
// Project file walking
// =========================================================================

/// Information about a discovered source file.
struct SourceFile {
    fs::path path;
    uintmax_t size_bytes = 0;
};

/// Recursively walk a project directory and collect source files.
static std::vector<SourceFile> walk_project_files(const fs::path& project_root) {
    std::vector<SourceFile> files;

    std::error_code ec;
    auto it = fs::recursive_directory_iterator(
        project_root,
        fs::directory_options::skip_permission_denied,
        ec
    );

    if (ec) {
        std::cerr << "Error: failed to open project directory '"
                  << project_root.generic_string() << "': " << ec.message() << "\n";
        return files;
    }

    for (; it != fs::recursive_directory_iterator(); ++it) {
        const auto& entry = *it;

        if (entry.is_directory()) {
            const auto dirname = entry.path().filename().string();
            if (is_skip_directory(dirname)) {
                it.disable_recursion_pending();
                continue;
            }
            continue;
        }

        if (!entry.is_regular_file()) continue;

        auto ext = entry.path().extension().string();
        if (is_supported_extension(ext)) {
            SourceFile sf;
            sf.path = entry.path();
            sf.size_bytes = entry.file_size(ec);
            if (ec) sf.size_bytes = 0;
            files.push_back(std::move(sf));
        }
    }

    std::sort(files.begin(), files.end(),
              [](const SourceFile& a, const SourceFile& b) {
                  return a.path < b.path;
              });
    return files;
}

// =========================================================================
// Size category classification
// =========================================================================

/// Return a human-readable size category for a file.
static const char* size_category(uintmax_t bytes) {
    if (bytes < 1024)        return "small  (<1KB)";
    if (bytes < 10 * 1024)   return "medium (1-10KB)";
    return                          "large  (>10KB)";
}

// =========================================================================
// Timing helpers
// =========================================================================

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>;  // milliseconds

// =========================================================================
// Benchmark result types
// =========================================================================

/// Per-file timing result.
struct FileResult {
    fs::path path;
    uintmax_t size_bytes = 0;
    size_t chunk_count = 0;
    double time_ms = 0.0;
};

/// Aggregate result for one chunker across all files and iterations.
struct ChunkerResult {
    std::string name;
    double avg_total_ms = 0.0;
    size_t total_chunks = 0;
    size_t files_processed = 0;
    std::vector<FileResult> per_file;  // from the last iteration
};

// =========================================================================
// Run benchmark for a single chunker
// =========================================================================

/// Benchmark a chunker over all files for the given number of iterations.
/// Returns aggregate results with per-file breakdown from the last iteration.
static ChunkerResult benchmark_chunker(const std::string& name,
                                       engram::Chunker& chunker,
                                       const std::vector<SourceFile>& files,
                                       int iterations)
{
    ChunkerResult result;
    result.name = name;

    std::vector<double> iteration_times;
    iteration_times.reserve(static_cast<size_t>(iterations));

    for (int iter = 0; iter < iterations; ++iter) {
        result.per_file.clear();
        result.total_chunks = 0;
        result.files_processed = 0;

        auto iter_start = Clock::now();

        for (const auto& sf : files) {
            auto file_start = Clock::now();
            auto chunks = chunker.chunk_file(sf.path);
            auto file_end = Clock::now();

            double file_ms = Duration(file_end - file_start).count();

            FileResult fr;
            fr.path = sf.path;
            fr.size_bytes = sf.size_bytes;
            fr.chunk_count = chunks.size();
            fr.time_ms = file_ms;
            result.per_file.push_back(std::move(fr));

            if (!chunks.empty()) {
                result.files_processed++;
            }
            result.total_chunks += chunks.size();
        }

        auto iter_end = Clock::now();
        double iter_ms = Duration(iter_end - iter_start).count();
        iteration_times.push_back(iter_ms);
    }

    // Compute average total time across iterations.
    result.avg_total_ms = std::accumulate(iteration_times.begin(),
                                          iteration_times.end(), 0.0)
                          / static_cast<double>(iterations);

    return result;
}

// =========================================================================
// Collect all chunk texts (non-timed, for embedding benchmark input)
// =========================================================================

/// Run a chunker once over all files and collect the source text of every chunk.
/// Returns a vector of strings ready for embed_batch().
static std::vector<std::string> collect_chunk_texts(
    engram::Chunker& chunker,
    const std::vector<SourceFile>& files)
{
    std::vector<std::string> texts;
    for (const auto& sf : files) {
        auto chunks = chunker.chunk_file(sf.path);
        for (auto& c : chunks) {
            texts.push_back(std::move(c.source_text));
        }
    }
    return texts;
}

// =========================================================================
// Output formatting
// =========================================================================

/// Print the summary block for a single chunker.
static void print_chunker_summary(const ChunkerResult& r, int iterations) {
    double throughput = (r.avg_total_ms > 0.0)
        ? (static_cast<double>(r.total_chunks) / (r.avg_total_ms / 1000.0))
        : 0.0;

    std::cout << "--- " << r.name << " ---\n";
    std::cout << "  Total time:   " << std::fixed << std::setprecision(0)
              << r.avg_total_ms << " ms (avg over "
              << iterations << " iterations)\n";
    std::cout << "  Files:        " << r.files_processed << "\n";
    std::cout << "  Chunks:       " << r.total_chunks << "\n";
    std::cout << "  Throughput:   " << std::fixed << std::setprecision(0)
              << throughput << " chunks/sec\n";
    std::cout << "\n";
}

/// Print the comparison table (only when two chunkers are available).
static void print_comparison(const ChunkerResult& regex_result,
                             const ChunkerResult& ts_result) {
    double speedup = (ts_result.avg_total_ms > 0.0)
        ? (regex_result.avg_total_ms / ts_result.avg_total_ms)
        : 0.0;

    std::cout << "--- Comparison ---\n";
    std::cout << "  Regex:        " << std::fixed << std::setprecision(0)
              << regex_result.avg_total_ms << " ms\n";
    std::cout << "  Tree-sitter:  " << std::fixed << std::setprecision(0)
              << ts_result.avg_total_ms << " ms\n";
    std::cout << "  Speedup:      " << std::fixed << std::setprecision(2)
              << speedup << "x\n";
    std::cout << "\n";
}

/// Print per-file breakdown table.
/// If two results are given (tree-sitter available), shows both columns.
static void print_per_file_breakdown(const ChunkerResult& regex_result,
                                     const ChunkerResult* ts_result,
                                     const fs::path& project_root) {
    std::cout << "--- Per-File Breakdown ---\n";

    // Header
    std::cout << std::left << std::setw(50) << "File"
              << std::setw(18) << "Size"
              << std::right << std::setw(10) << "Regex ms"
              << std::setw(10) << "Chunks";
    if (ts_result) {
        std::cout << std::setw(12) << "TS ms"
                  << std::setw(10) << "Chunks";
    }
    std::cout << "\n";

    // Separator
    size_t sep_len = 88 + (ts_result ? 22 : 0);
    std::cout << std::string(sep_len, '-') << "\n";

    for (size_t i = 0; i < regex_result.per_file.size(); ++i) {
        const auto& rf = regex_result.per_file[i];

        // Show path relative to project root for readability.
        std::error_code ec;
        auto rel = fs::relative(rf.path, project_root, ec);
        std::string display_path = ec ? rf.path.generic_string()
                                      : rel.generic_string();

        // Truncate long paths.
        if (display_path.size() > 48) {
            display_path = "..." + display_path.substr(display_path.size() - 45);
        }

        std::cout << std::left << std::setw(50) << display_path
                  << std::setw(18) << size_category(rf.size_bytes)
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(10) << rf.time_ms
                  << std::setw(10) << rf.chunk_count;

        if (ts_result && i < ts_result->per_file.size()) {
            const auto& tf = ts_result->per_file[i];
            std::cout << std::setw(12) << tf.time_ms
                      << std::setw(10) << tf.chunk_count;
        }

        std::cout << "\n";
    }

    std::cout << "\n";
}

// =========================================================================
// Embedding benchmark (requires ONNX)
// =========================================================================

#ifdef ENGRAM_HAS_ONNX

/// Benchmark batch embedding throughput.
///
/// @param embedder     The embedding model to benchmark.
/// @param texts        All chunk texts to embed.
/// @param batch_size   Batch size for embed_batch() calls.
/// @param warmup       Number of warmup iterations (first inference JIT-compiles CUDA kernels).
static void benchmark_embedding(engram::OrtEmbedder& embedder,
                                const std::vector<std::string>& texts,
                                size_t batch_size,
                                int warmup)
{
    std::cout << "--- Embedding Benchmark ---\n";
    std::cout << "  Provider:     " << embedder.active_provider() << "\n";
    std::cout << "  Model:        " << embedder.model_name() << "\n";
    std::cout << "  Dimension:    " << embedder.dimension() << "\n";
    std::cout << "  Chunks:       " << texts.size() << "\n";
    std::cout << "  Batch size:   " << batch_size << "\n";
    std::cout << "  Warmup iters: " << warmup << "\n";

    // Warmup: embed a small batch to trigger JIT compilation.
    {
        std::vector<std::string> warmup_texts;
        for (int i = 0; i < warmup && static_cast<size_t>(i) < texts.size(); ++i) {
            warmup_texts.push_back(texts[static_cast<size_t>(i)]);
        }
        for (int i = 0; i < warmup; ++i) {
            embedder.embed_batch(warmup_texts);
        }
    }

    // Timed run: embed all chunks in batches.
    auto start = Clock::now();

    size_t embedded = 0;
    for (size_t offset = 0; offset < texts.size(); offset += batch_size) {
        size_t end = std::min(offset + batch_size, texts.size());
        std::vector<std::string> batch(texts.begin() + static_cast<ptrdiff_t>(offset),
                                        texts.begin() + static_cast<ptrdiff_t>(end));
        auto results = embedder.embed_batch(batch);
        for (const auto& r : results) {
            if (!r.empty()) embedded++;
        }
    }

    auto elapsed = Clock::now() - start;
    double elapsed_ms = Duration(elapsed).count();
    double throughput = (elapsed_ms > 0.0)
        ? (static_cast<double>(embedded) / (elapsed_ms / 1000.0))
        : 0.0;

    std::cout << "  Total time:   " << std::fixed << std::setprecision(0)
              << elapsed_ms << " ms\n";
    std::cout << "  Embedded:     " << embedded << " / " << texts.size() << "\n";
    std::cout << "  Throughput:   " << std::fixed << std::setprecision(0)
              << throughput << " chunks/sec\n";
    std::cout << "\n";
}

/// Benchmark query latency: embed a query + HNSW search.
///
/// @param embedder     The embedding model.
/// @param index        A populated HNSW index.
/// @param warmup       Number of warmup queries.
/// @param num_queries  Number of measured queries.
static void benchmark_query_latency(engram::OrtEmbedder& embedder,
                                    engram::HnswIndex& index,
                                    int warmup,
                                    int num_queries)
{
    // Sample queries representative of real usage.
    static const std::vector<std::string> sample_queries = {
        "how does the file watcher work",
        "HNSW vector index implementation",
        "MCP protocol message handling",
        "session memory storage and retrieval",
        "tree-sitter chunker AST parsing",
        "ONNX Runtime CUDA embedding inference",
        "batch embedding GPU throughput",
        "incremental re-indexing content hash",
        "regex chunker function boundary detection",
        "JSON-RPC tool handler registration",
        "cosine similarity nearest neighbor search",
        "chunk metadata persistence",
        "Windows ReadDirectoryChangesW monitoring",
        "symbol lookup by name",
        "code context retrieval by line range",
        "project file walking with extension filter",
        "FNV-1a content hash comparison",
        "embedding dimension auto-detection",
        "debounced file change notification",
        "session summary keyword matching"
    };

    size_t query_count = std::min(static_cast<size_t>(num_queries), sample_queries.size());

    std::cout << "--- Query Latency Benchmark ---\n";
    std::cout << "  Index size:   " << index.size() << " vectors\n";
    std::cout << "  Warmup:       " << warmup << "\n";
    std::cout << "  Queries:      " << query_count << "\n";

    // Warmup.
    for (int i = 0; i < warmup; ++i) {
        auto emb = embedder.embed(sample_queries[0]);
        if (!emb.empty()) {
            index.search(emb.data(), emb.size(), 10);
        }
    }

    // Timed queries.
    std::vector<double> embed_times;
    std::vector<double> search_times;
    std::vector<double> total_times;

    for (size_t i = 0; i < query_count; ++i) {
        auto t0 = Clock::now();
        auto emb = embedder.embed(sample_queries[i]);
        auto t1 = Clock::now();

        if (emb.empty()) {
            std::cerr << "  Warning: embedding failed for query " << i << "\n";
            continue;
        }

        auto hits = index.search(emb.data(), emb.size(), 10);
        auto t2 = Clock::now();

        double embed_ms  = Duration(t1 - t0).count();
        double search_ms = Duration(t2 - t1).count();
        double total_ms  = Duration(t2 - t0).count();

        embed_times.push_back(embed_ms);
        search_times.push_back(search_ms);
        total_times.push_back(total_ms);
    }

    // Compute statistics.
    auto stats = [](const std::vector<double>& v) -> std::tuple<double, double, double> {
        if (v.empty()) return {0.0, 0.0, 0.0};
        double mean = std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
        double min_val = *std::min_element(v.begin(), v.end());
        double max_val = *std::max_element(v.begin(), v.end());
        return {mean, min_val, max_val};
    };

    auto [embed_mean, embed_min, embed_max] = stats(embed_times);
    auto [search_mean, search_min, search_max] = stats(search_times);
    auto [total_mean, total_min, total_max] = stats(total_times);

    std::cout << "\n";
    std::cout << "  " << std::left << std::setw(16) << "Phase"
              << std::right << std::setw(10) << "Mean"
              << std::setw(10) << "Min"
              << std::setw(10) << "Max" << "\n";
    std::cout << "  " << std::string(46, '-') << "\n";

    auto print_row = [](const char* label, double mean, double min_val, double max_val) {
        std::cout << "  " << std::left << std::setw(16) << label
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(8) << mean << " ms"
                  << std::setw(8) << min_val << " ms"
                  << std::setw(8) << max_val << " ms\n";
    };

    print_row("Embed query", embed_mean, embed_min, embed_max);
    print_row("HNSW search", search_mean, search_min, search_max);
    print_row("Total", total_mean, total_min, total_max);
    std::cout << "\n";
}

#endif // ENGRAM_HAS_ONNX

// =========================================================================
// Usage
// =========================================================================

static void print_usage(const char* program_name) {
    std::cerr << "Usage: " << program_name << " --project <path> [options]\n"
              << "\n"
              << "Options:\n"
              << "  --project <path>      Root of the codebase to benchmark (required)\n"
              << "  --iterations N        Number of timing iterations (default: 3)\n"
              << "  --model <path>        Path to ONNX model for embedding benchmarks\n"
              << "  --batch-size N        Batch size for embedding (default: 32)\n"
              << "  --warmup N            Warmup iterations for GPU benchmarks (default: 3)\n"
              << "  --queries N           Number of query latency samples (default: 20)\n"
              << "  --help, -h            Show this help message\n";
}

// =========================================================================
// main
// =========================================================================

int main(int argc, char* argv[])
{
    // -----------------------------------------------------------------------
    // Parse arguments
    // -----------------------------------------------------------------------
    std::vector<std::string> args(argv, argv + argc);

    if (has_flag(args, "--help") || has_flag(args, "-h")) {
        print_usage(argv[0]);
        return 0;
    }

    const std::string project_path    = parse_arg(args, "--project");
    const std::string iter_str        = parse_arg(args, "--iterations", "3");
    const std::string model_path      = parse_arg(args, "--model");
    const std::string batch_size_str  = parse_arg(args, "--batch-size", "32");
    const std::string warmup_str      = parse_arg(args, "--warmup", "3");
    const std::string queries_str     = parse_arg(args, "--queries", "20");

    if (project_path.empty()) {
        std::cerr << "Error: --project <path> is required.\n\n";
        print_usage(argv[0]);
        return 1;
    }

    int iterations = 3;
    try {
        iterations = std::stoi(iter_str);
        if (iterations < 1) iterations = 1;
    } catch (...) {
        std::cerr << "Warning: invalid --iterations value '" << iter_str
                  << "', using default 3.\n";
        iterations = 3;
    }

    size_t batch_size = 32;
    try {
        batch_size = static_cast<size_t>(std::stoul(batch_size_str));
        if (batch_size == 0) batch_size = 1;
    } catch (...) {
        batch_size = 32;
    }

    int warmup_iters = 3;
    try {
        warmup_iters = std::stoi(warmup_str);
        if (warmup_iters < 0) warmup_iters = 0;
    } catch (...) {
        warmup_iters = 3;
    }

    int num_queries = 20;
    try {
        num_queries = std::stoi(queries_str);
        if (num_queries < 1) num_queries = 1;
    } catch (...) {
        num_queries = 20;
    }

    fs::path project_root(project_path);
    {
        std::error_code ec;
        if (!fs::is_directory(project_root, ec)) {
            std::cerr << "Error: '" << project_path << "' is not a directory.\n";
            return 1;
        }
        // Canonicalize for display.
        auto canonical = fs::canonical(project_root, ec);
        if (!ec) {
            project_root = canonical;
        }
    }

    // -----------------------------------------------------------------------
    // Discover files
    // -----------------------------------------------------------------------
    auto files = walk_project_files(project_root);

    if (files.empty()) {
        std::cerr << "Error: no supported source files found in '"
                  << project_root.generic_string() << "'.\n";
        return 1;
    }

    // Compute total size.
    uintmax_t total_bytes = 0;
    for (const auto& sf : files) {
        total_bytes += sf.size_bytes;
    }

    // -----------------------------------------------------------------------
    // Print banner
    // -----------------------------------------------------------------------
    std::cout << "=== Engram Benchmark Suite ===\n";
    std::cout << "Project: " << project_root.generic_string() << "\n";
    std::cout << "Files found: " << files.size() << "\n";
    std::cout << "Total size: " << std::fixed << std::setprecision(1)
              << (static_cast<double>(total_bytes) / 1024.0) << " KB\n";
    std::cout << "Iterations: " << iterations << "\n";
    std::cout << "\n";

    // -----------------------------------------------------------------------
    // Benchmark: Regex Chunker
    // -----------------------------------------------------------------------
    engram::RegexChunker regex_chunker;
    auto regex_result = benchmark_chunker("Regex Chunker", regex_chunker,
                                          files, iterations);
    print_chunker_summary(regex_result, iterations);

    // -----------------------------------------------------------------------
    // Benchmark: Tree-sitter Chunker (conditional)
    // -----------------------------------------------------------------------
    ChunkerResult ts_result;
    bool has_treesitter = false;

#ifdef ENGRAM_HAS_TREESITTER
    {
        engram::TreeSitterChunker ts_chunker;
        ts_result = benchmark_chunker("Tree-sitter Chunker", ts_chunker,
                                      files, iterations);
        has_treesitter = true;
        print_chunker_summary(ts_result, iterations);
    }
#endif

    // -----------------------------------------------------------------------
    // Comparison (if both available)
    // -----------------------------------------------------------------------
    if (has_treesitter) {
        print_comparison(regex_result, ts_result);
    }

    // -----------------------------------------------------------------------
    // Per-file breakdown
    // -----------------------------------------------------------------------
    print_per_file_breakdown(regex_result,
                             has_treesitter ? &ts_result : nullptr,
                             project_root);

    // -----------------------------------------------------------------------
    // Embedding + query latency benchmarks (requires ONNX)
    // -----------------------------------------------------------------------
#ifdef ENGRAM_HAS_ONNX
    if (!model_path.empty()) {
        fs::path model_file(model_path);
        fs::path tokenizer_file = model_file.parent_path() / "tokenizer.json";

        std::error_code ec;
        if (!fs::exists(model_file, ec)) {
            std::cerr << "Error: model file '" << model_path << "' not found.\n";
            return 1;
        }

        std::cout << "--- Loading Embedder ---\n";
        engram::OrtEmbedder embedder(
            model_file.string(),
            tokenizer_file.string(),
            engram::DevicePreference::CUDA
        );

        if (!embedder.is_valid()) {
            std::cerr << "Error: embedder failed to initialize.\n";
            return 1;
        }

        std::cout << "  Model:    " << embedder.model_name() << "\n";
        std::cout << "  Provider: " << embedder.active_provider() << "\n";
        std::cout << "  Dim:      " << embedder.dimension() << "\n";
        std::cout << "\n";

        // Collect all chunk texts for embedding benchmark.
        // Use the best available chunker.
        engram::Chunker* bench_chunker = &regex_chunker;
#ifdef ENGRAM_HAS_TREESITTER
        engram::TreeSitterChunker ts_for_embed;
        bench_chunker = &ts_for_embed;
#endif
        auto chunk_texts = collect_chunk_texts(*bench_chunker, files);

        // --- Embedding throughput ---
        benchmark_embedding(embedder, chunk_texts, batch_size, warmup_iters);

        // --- Query latency (requires a populated index) ---
        std::cout << "--- Building Index for Query Benchmark ---\n";
        engram::HnswIndex index(embedder.dimension());

        size_t indexed = 0;
        for (size_t offset = 0; offset < chunk_texts.size(); offset += batch_size) {
            size_t end = std::min(offset + batch_size, chunk_texts.size());
            std::vector<std::string> batch(
                chunk_texts.begin() + static_cast<ptrdiff_t>(offset),
                chunk_texts.begin() + static_cast<ptrdiff_t>(end));
            auto embeddings = embedder.embed_batch(batch);
            for (size_t i = 0; i < embeddings.size(); ++i) {
                if (!embeddings[i].empty()) {
                    std::string id = "bench_" + std::to_string(offset + i);
                    index.add(id, embeddings[i].data(), embeddings[i].size());
                    indexed++;
                }
            }
        }
        std::cout << "  Indexed " << indexed << " chunks into HNSW\n\n";

        benchmark_query_latency(embedder, index, warmup_iters, num_queries);
    }
#else
    if (!model_path.empty()) {
        std::cout << "--- Embedding Benchmark ---\n";
        std::cout << "  (Skipped: built without ONNX Runtime. Rebuild with -DENGRAM_USE_ONNX=ON)\n\n";
    }
#endif

    return 0;
}
