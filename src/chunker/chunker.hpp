#pragma once

/// @file chunker.hpp
/// @brief Abstract chunker interface and Chunk data structure for Engram.
///
/// Defines the common types and interface that all chunker implementations
/// (tree-sitter, regex fallback, etc.) must satisfy. A Chunk represents a
/// semantically meaningful unit of source code together with its metadata.

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace engram {

/// A single chunk of source code with associated metadata.
///
/// Chunks are the fundamental indexing unit. Each chunk maps to a contiguous
/// range of lines in a source file and carries enough metadata to reconstruct
/// context (file path, line range, language, symbol name) and to be uniquely
/// identified across index rebuilds (chunk_id).
struct Chunk {
    /// The raw source text of the chunk.
    std::string source_text;

    /// Absolute path to the file this chunk originates from.
    std::filesystem::path file_path;

    /// 1-based inclusive start line within the file.
    uint32_t start_line = 0;

    /// 1-based inclusive end line within the file.
    uint32_t end_line = 0;

    /// Language identifier: "cpp", "python", "javascript", "typescript", etc.
    std::string language;

    /// Name of the enclosing symbol (function, class, method) if known.
    /// Empty string when no symbol can be determined.
    std::string symbol_name;

    /// Deterministic unique ID derived from file_path and line range.
    /// Format: hex digest of hash(canonical_path + ":" + start_line + ":" + end_line).
    std::string chunk_id;

    /// FNV-1a 64-bit hash of the source file contents at indexing time.
    /// Used for incremental re-indexing: if the hash matches, the file is
    /// unchanged and its chunks can be skipped.  Empty string for legacy
    /// chunks that were created before this field was added.
    std::string file_content_hash;
};

/// Generate a deterministic chunk ID from a file path and line range.
///
/// The ID is a 16-hex-character string derived from hashing the canonical
/// path concatenated with the line range. The same inputs always produce
/// the same output, and the result is stable across process restarts.
///
/// @param file_path  Path to the source file (will be canonicalized if possible).
/// @param start_line 1-based start line.
/// @param end_line   1-based end line.
/// @return A 16-character lowercase hex string.
inline std::string generate_chunk_id(const std::filesystem::path& file_path,
                                     uint32_t start_line,
                                     uint32_t end_line) {
    // Build a canonical key string.  If the path cannot be canonicalized
    // (e.g. file does not exist on disk yet during tests) fall back to the
    // generic string representation.
    std::string key;
    std::error_code ec;
    auto canonical = std::filesystem::canonical(file_path, ec);
    if (!ec) {
        key = canonical.generic_string();
    } else {
        key = file_path.generic_string();
    }
    key += ':';
    key += std::to_string(start_line);
    key += ':';
    key += std::to_string(end_line);

    // Use std::hash and format as 16-char hex.  std::hash<std::string> is
    // deterministic within a single program execution on MSVC (and on most
    // implementations across runs for the same binary), which is sufficient
    // for our index-lifetime needs.  For cross-build stability we use a
    // simple FNV-1a 64-bit hash instead.
    constexpr uint64_t fnv_offset = 14695981039346656037ULL;
    constexpr uint64_t fnv_prime  = 1099511628211ULL;
    uint64_t hash = fnv_offset;
    for (char c : key) {
        hash ^= static_cast<uint64_t>(static_cast<unsigned char>(c));
        hash *= fnv_prime;
    }

    // Format as 16 lowercase hex characters.
    char buf[17];
    snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(hash));
    return std::string(buf, 16);
}

/// Abstract base class for code chunkers.
///
/// Implementations split source files into semantically meaningful chunks
/// suitable for embedding and indexing.  The chunker is language-aware and
/// produces metadata-rich Chunk objects.
class Chunker {
public:
    virtual ~Chunker() = default;

    /// Split a source file into chunks.
    ///
    /// @param file_path Path to the file to chunk. Must exist and be readable.
    /// @return A vector of chunks covering the entire file. Returns an empty
    ///         vector if the file cannot be read or is empty.
    virtual std::vector<Chunk> chunk_file(const std::filesystem::path& file_path) = 0;
};

} // namespace engram
