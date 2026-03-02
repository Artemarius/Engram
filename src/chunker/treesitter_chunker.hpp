#pragma once

/// @file treesitter_chunker.hpp
/// @brief AST-aware code chunker using tree-sitter grammars.
///
/// Uses tree-sitter's incremental parsing engine and S-expression queries to
/// locate function, class, and other named constructs with precise boundaries.
/// Falls back to blank-line splitting for unsupported languages.

#include "chunker.hpp"
#include "regex_chunker.hpp"   // RegexChunkerConfig + fallback

#include <string>
#include <unordered_map>
#include <vector>

// Forward-declare tree-sitter C types (avoids including api.h in the header).
struct TSLanguage;
struct TSQuery;

namespace engram {

/// AST-aware chunker backed by tree-sitter grammars.
///
/// Pre-compiles one S-expression query per supported language at construction
/// time.  Each `chunk_file` / `chunk_string` call creates its own TSParser and
/// TSTree — so the chunker is **thread-safe** for concurrent chunk_file calls
/// (the pre-compiled TSQuery objects are immutable and shared).
///
/// Supported languages (9): C++, Python, JavaScript, TypeScript/TSX, Java,
/// Rust, Go, Ruby, C#.
class TreeSitterChunker final : public Chunker {
public:
    /// Construct with default chunking config (50-500 tokens).
    TreeSitterChunker();

    /// Construct with custom min/max token settings.
    explicit TreeSitterChunker(RegexChunkerConfig config);

    /// Destructor frees pre-compiled TSQuery objects.
    ~TreeSitterChunker() override;

    // Non-copyable (owns TSQuery pointers).
    TreeSitterChunker(const TreeSitterChunker&) = delete;
    TreeSitterChunker& operator=(const TreeSitterChunker&) = delete;

    /// Split a source file into AST-aware chunks.
    std::vector<Chunk> chunk_file(const std::filesystem::path& file_path) override;

    /// Split an in-memory string as if it came from the given file path.
    /// Useful for testing.
    std::vector<Chunk> chunk_string(const std::string& source,
                                    const std::filesystem::path& file_path);

private:
    /// Pre-compiled language + query pair.
    struct LanguageInfo {
        const TSLanguage* language = nullptr;
        TSQuery*          query    = nullptr;
        std::string       chunk_language;  ///< language tag written into Chunk.language
    };

    RegexChunkerConfig config_;
    std::unordered_map<std::string, LanguageInfo> languages_;

    /// Detect language from file extension. Returns a language tag used to
    /// look up the LanguageInfo (e.g. "cpp", "python", "tsx").
    static std::string detect_language(const std::filesystem::path& file_path);

    /// Pre-compile queries for all supported languages.
    void init_languages();

    /// Approximate token count from character count.
    uint32_t estimate_tokens(const std::string& text) const;

    // ----- Block types shared with merge/split logic -----

    struct RawBlock {
        std::string  text;
        std::string  symbol_name;
        uint32_t     start_line;   // 1-based
        uint32_t     end_line;     // 1-based inclusive
    };

    std::vector<RawBlock> split_oversized(const RawBlock& block) const;
    std::vector<RawBlock> merge_tiny(std::vector<RawBlock> blocks) const;

    /// Fallback for unsupported languages: delegates to RegexChunker.
    std::vector<Chunk> fallback_chunk(const std::string& source,
                                      const std::filesystem::path& file_path);
};

} // namespace engram
