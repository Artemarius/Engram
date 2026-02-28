#pragma once

/// @file regex_chunker.hpp
/// @brief Regex-based fallback chunker for Engram.
///
/// Splits source files into chunks by matching function/class boundaries with
/// regular expressions.  This is the fallback chunker used when tree-sitter
/// grammars are not available for a given language.

#include "chunker.hpp"

#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

namespace engram {

/// Configuration knobs for the regex chunker.
struct RegexChunkerConfig {
    /// Approximate minimum number of tokens per chunk (tokens ~ chars / 4).
    uint32_t min_tokens = 50;

    /// Approximate maximum number of tokens per chunk.
    uint32_t max_tokens = 500;

    /// Characters-per-token estimate used to convert between char counts and
    /// token counts.  A value of 4 is a reasonable average for code.
    uint32_t chars_per_token = 4;
};

/// Regex-based chunker implementation.
///
/// Detects the source language from the file extension, then applies
/// language-specific regex patterns to locate function and class boundaries.
/// Falls back to splitting on blank-line-separated blocks for unknown
/// languages.  Over-sized chunks are split at blank lines; tiny chunks are
/// merged with their predecessor.
class RegexChunker final : public Chunker {
public:
    /// Construct a RegexChunker with default configuration.
    RegexChunker();

    /// Construct a RegexChunker with custom configuration.
    ///
    /// @param config Chunking parameters (min/max tokens, chars-per-token).
    explicit RegexChunker(RegexChunkerConfig config);

    /// Split a source file into chunks.
    ///
    /// @param file_path Path to the file to chunk.
    /// @return Vector of chunks.  Empty if the file cannot be read.
    std::vector<Chunk> chunk_file(const std::filesystem::path& file_path) override;

    /// Split an in-memory string as if it came from the given file path.
    ///
    /// This is useful for testing and for callers that already have the file
    /// content loaded.
    ///
    /// @param source   Full source text.
    /// @param file_path Path used for metadata (language detection, chunk IDs).
    /// @return Vector of chunks.
    std::vector<Chunk> chunk_string(const std::string& source,
                                    const std::filesystem::path& file_path);

private:
    /// A raw boundary detected by regex matching.
    struct RawBoundary {
        uint32_t line;           // 1-based line number where the boundary starts
        std::string symbol_name; // extracted symbol name, may be empty
    };

    RegexChunkerConfig config_;

    /// Detect language from file extension.
    static std::string detect_language(const std::filesystem::path& file_path);

    /// Return a list of compiled regex patterns for the given language.
    /// Each regex is expected to have a named group "name" that captures
    /// the symbol name, or the implementation extracts from group(1).
    static const std::vector<std::regex>& patterns_for_language(const std::string& lang);

    /// Find all boundary lines in the source for the given language.
    std::vector<RawBoundary> find_boundaries(const std::vector<std::string>& lines,
                                             const std::string& language) const;

    /// Split lines into raw text blocks at the given boundaries.
    /// Returns pairs of (block_text, symbol_name) with start/end lines.
    struct RawBlock {
        std::string text;
        std::string symbol_name;
        uint32_t start_line; // 1-based
        uint32_t end_line;   // 1-based inclusive
    };
    std::vector<RawBlock> split_at_boundaries(
        const std::vector<std::string>& lines,
        const std::vector<RawBoundary>& boundaries) const;

    /// Split an oversized block at blank-line boundaries.
    std::vector<RawBlock> split_oversized(const RawBlock& block) const;

    /// Merge undersized blocks with their predecessor.
    std::vector<RawBlock> merge_tiny(std::vector<RawBlock> blocks) const;

    /// Convert a RawBlock into a Chunk with all metadata filled in.
    Chunk make_chunk(const RawBlock& block,
                     const std::filesystem::path& file_path,
                     const std::string& language) const;

    /// Estimate token count from character count.
    uint32_t estimate_tokens(const std::string& text) const;
};

} // namespace engram
