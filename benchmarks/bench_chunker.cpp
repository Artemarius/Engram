/// @file bench_chunker.cpp
/// @brief Standalone chunker performance benchmark for Engram.
///
/// Walks a project directory, chunks all source files with each available
/// chunker implementation, and reports timing statistics.  Uses std::chrono
/// for measurement — no external benchmark framework required.
///
/// Usage:
///   bench_chunker --project <path> [--iterations N] [--model <path>]
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
// Usage
// =========================================================================

static void print_usage(const char* program_name) {
    std::cerr << "Usage: " << program_name << " --project <path> [options]\n"
              << "\n"
              << "Options:\n"
              << "  --project <path>      Root of the codebase to benchmark (required)\n"
              << "  --iterations N        Number of timing iterations (default: 3)\n"
              << "  --model <path>        Path to ONNX model (reserved for future use)\n"
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

    const std::string project_path = parse_arg(args, "--project");
    const std::string iter_str     = parse_arg(args, "--iterations", "3");
    const std::string model_path   = parse_arg(args, "--model");

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
    std::cout << "=== Engram Chunker Benchmark ===\n";
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
    // Query latency benchmark (TODO: requires embedder)
    // -----------------------------------------------------------------------
    if (!model_path.empty()) {
        std::cout << "--- Query Latency ---\n";
        std::cout << "  (TODO: --model was provided but query latency "
                  << "benchmarking is not yet implemented.)\n";
        std::cout << "  Model: " << model_path << "\n";
        std::cout << "\n";
    }

    return 0;
}
