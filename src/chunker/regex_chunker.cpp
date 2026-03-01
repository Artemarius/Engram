/// @file regex_chunker.cpp
/// @brief Regex-based fallback chunker implementation.

#include "regex_chunker.hpp"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <sstream>
#include <unordered_map>

namespace engram {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

RegexChunker::RegexChunker()
    : config_{} {}

RegexChunker::RegexChunker(RegexChunkerConfig config)
    : config_(config) {}

// ---------------------------------------------------------------------------
// Language detection
// ---------------------------------------------------------------------------

std::string RegexChunker::detect_language(const std::filesystem::path& file_path) {
    // Map of extension (with dot) -> language tag.
    static const std::unordered_map<std::string, std::string> ext_map = {
        {".cpp",  "cpp"},
        {".cxx",  "cpp"},
        {".cc",   "cpp"},
        {".c",    "cpp"},       // treat C as cpp for chunking patterns
        {".hpp",  "cpp"},
        {".hxx",  "cpp"},
        {".h",    "cpp"},
        {".py",   "python"},
        {".pyw",  "python"},
        {".js",   "javascript"},
        {".jsx",  "javascript"},
        {".mjs",  "javascript"},
        {".ts",   "typescript"},
        {".tsx",  "typescript"},
        {".java", "java"},
        {".rs",   "rust"},
        {".go",   "go"},
        {".rb",   "ruby"},
        {".cs",   "csharp"},
    };

    std::string ext = file_path.extension().string();
    // Lowercase the extension for case-insensitive matching.
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    auto it = ext_map.find(ext);
    if (it != ext_map.end()) {
        return it->second;
    }
    return "unknown";
}

// ---------------------------------------------------------------------------
// Regex patterns per language
// ---------------------------------------------------------------------------

const std::vector<std::regex>& RegexChunker::patterns_for_language(const std::string& lang) {
    // We build the pattern vectors once (static) and return a const reference.
    // Each regex should match a line that starts a new semantic block.
    // Group 1 (or named group "name") captures the symbol name where possible.

    static const std::vector<std::regex> cpp_patterns = {
        // Free function / method definition: return_type name(
        // Handles templates, pointers, references, namespaces in the return type.
        std::regex(
            R"(^[ \t]*(?:(?:static|inline|virtual|explicit|constexpr|const|volatile|unsigned|signed|extern|friend|template\s*<[^>]*>)\s+)*)"
            R"((?:[\w:*&<>,\s]+?)\s+(\w[\w:]*)\s*\([^;]*$)",
            std::regex_constants::ECMAScript),
        // Class / struct declaration
        std::regex(
            R"(^[ \t]*(?:template\s*<[^>]*>\s*)?(?:class|struct)\s+(\w+))",
            std::regex_constants::ECMAScript),
        // Namespace
        std::regex(
            R"(^[ \t]*namespace\s+(\w+))",
            std::regex_constants::ECMAScript),
    };

    static const std::vector<std::regex> python_patterns = {
        // def function_name(
        std::regex(R"(^[ \t]*(?:async\s+)?def\s+(\w+)\s*\()",
                   std::regex_constants::ECMAScript),
        // class ClassName
        std::regex(R"(^[ \t]*class\s+(\w+))",
                   std::regex_constants::ECMAScript),
    };

    static const std::vector<std::regex> js_ts_patterns = {
        // function name(
        std::regex(R"(^[ \t]*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\()",
                   std::regex_constants::ECMAScript),
        // class Name
        std::regex(R"(^[ \t]*(?:export\s+)?class\s+(\w+))",
                   std::regex_constants::ECMAScript),
        // const name = (...) => or const name = function
        std::regex(R"(^[ \t]*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:\(|function))",
                   std::regex_constants::ECMAScript),
    };

    static const std::vector<std::regex> java_patterns = {
        // method or constructor
        std::regex(
            R"(^[ \t]*(?:(?:public|private|protected|static|final|abstract|synchronized|native)\s+)*)"
            R"((?:[\w<>\[\],\s]+?\s+)?(\w+)\s*\([^;]*$)",
            std::regex_constants::ECMAScript),
        // class / interface / enum
        std::regex(
            R"(^[ \t]*(?:(?:public|private|protected|static|final|abstract)\s+)*(?:class|interface|enum)\s+(\w+))",
            std::regex_constants::ECMAScript),
    };

    static const std::vector<std::regex> rust_patterns = {
        std::regex(R"(^[ \t]*(?:pub\s+)?(?:async\s+)?fn\s+(\w+))",
                   std::regex_constants::ECMAScript),
        std::regex(R"(^[ \t]*(?:pub\s+)?(?:struct|enum|trait|impl)\s+(\w+))",
                   std::regex_constants::ECMAScript),
    };

    static const std::vector<std::regex> go_patterns = {
        std::regex(R"(^func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\()",
                   std::regex_constants::ECMAScript),
        std::regex(R"(^type\s+(\w+)\s+(?:struct|interface))",
                   std::regex_constants::ECMAScript),
    };

    static const std::vector<std::regex> ruby_patterns = {
        std::regex(R"(^[ \t]*def\s+(\w+))",
                   std::regex_constants::ECMAScript),
        std::regex(R"(^[ \t]*class\s+(\w+))",
                   std::regex_constants::ECMAScript),
        std::regex(R"(^[ \t]*module\s+(\w+))",
                   std::regex_constants::ECMAScript),
    };

    static const std::vector<std::regex> csharp_patterns = {
        std::regex(
            R"(^[ \t]*(?:(?:public|private|protected|internal|static|virtual|override|abstract|async|sealed)\s+)*)"
            R"((?:[\w<>\[\]?,\s]+?\s+)?(\w+)\s*\([^;]*$)",
            std::regex_constants::ECMAScript),
        std::regex(
            R"(^[ \t]*(?:(?:public|private|protected|internal|static|abstract|sealed|partial)\s+)*(?:class|struct|interface|enum)\s+(\w+))",
            std::regex_constants::ECMAScript),
    };

    static const std::vector<std::regex> empty_patterns;

    static const std::unordered_map<std::string, const std::vector<std::regex>*> table = {
        {"cpp",        &cpp_patterns},
        {"python",     &python_patterns},
        {"javascript", &js_ts_patterns},
        {"typescript", &js_ts_patterns},
        {"java",       &java_patterns},
        {"rust",       &rust_patterns},
        {"go",         &go_patterns},
        {"ruby",       &ruby_patterns},
        {"csharp",     &csharp_patterns},
    };

    auto it = table.find(lang);
    if (it != table.end()) {
        return *(it->second);
    }
    return empty_patterns;
}

// ---------------------------------------------------------------------------
// Boundary detection
// ---------------------------------------------------------------------------

std::vector<RegexChunker::RawBoundary>
RegexChunker::find_boundaries(const std::vector<std::string>& lines,
                              const std::string& language) const {
    std::vector<RawBoundary> boundaries;
    const auto& patterns = patterns_for_language(language);

    if (patterns.empty()) {
        // No language-specific patterns: treat every blank-line gap as a boundary.
        bool prev_blank = true;
        for (uint32_t i = 0; i < static_cast<uint32_t>(lines.size()); ++i) {
            bool cur_blank = lines[i].find_first_not_of(" \t\r\n") == std::string::npos;
            if (prev_blank && !cur_blank) {
                boundaries.push_back({i + 1, ""});  // 1-based
            }
            prev_blank = cur_blank;
        }
        return boundaries;
    }

    for (uint32_t i = 0; i < static_cast<uint32_t>(lines.size()); ++i) {
        for (const auto& pat : patterns) {
            std::smatch match;
            if (std::regex_search(lines[i], match, pat)) {
                std::string name;
                if (match.size() > 1 && match[1].matched) {
                    name = match[1].str();
                }
                boundaries.push_back({i + 1, name});  // 1-based
                break;  // one match per line is enough
            }
        }
    }

    return boundaries;
}

// ---------------------------------------------------------------------------
// Splitting at boundaries
// ---------------------------------------------------------------------------

std::vector<RegexChunker::RawBlock>
RegexChunker::split_at_boundaries(const std::vector<std::string>& lines,
                                  const std::vector<RawBoundary>& boundaries) const {
    std::vector<RawBlock> blocks;
    if (lines.empty()) {
        return blocks;
    }

    // If there are no boundaries, treat the entire file as one block.
    if (boundaries.empty()) {
        std::string text;
        for (const auto& line : lines) {
            text += line;
            text += '\n';
        }
        blocks.push_back({std::move(text), "", 1, static_cast<uint32_t>(lines.size())});
        return blocks;
    }

    // Build blocks between consecutive boundaries.
    // A boundary at line B means "a new block starts at line B".
    // Everything before the first boundary is a preamble block.
    auto make_text = [&](uint32_t from_0, uint32_t to_0_exclusive) -> std::string {
        std::string text;
        for (uint32_t j = from_0; j < to_0_exclusive; ++j) {
            text += lines[j];
            text += '\n';
        }
        return text;
    };

    uint32_t first_boundary_0 = boundaries[0].line - 1;  // to 0-based

    // Preamble: lines before the first boundary.
    if (first_boundary_0 > 0) {
        blocks.push_back({
            make_text(0, first_boundary_0),
            "",  // no symbol for preamble
            1,
            first_boundary_0  // 1-based inclusive
        });
    }

    for (size_t b = 0; b < boundaries.size(); ++b) {
        uint32_t start_0 = boundaries[b].line - 1;
        uint32_t end_0_exclusive = (b + 1 < boundaries.size())
                                       ? (boundaries[b + 1].line - 1)
                                       : static_cast<uint32_t>(lines.size());
        blocks.push_back({
            make_text(start_0, end_0_exclusive),
            boundaries[b].symbol_name,
            start_0 + 1,                          // 1-based
            static_cast<uint32_t>(end_0_exclusive) // 1-based inclusive
        });
    }

    return blocks;
}

// ---------------------------------------------------------------------------
// Splitting oversized blocks
// ---------------------------------------------------------------------------

std::vector<RegexChunker::RawBlock>
RegexChunker::split_oversized(const RawBlock& block) const {
    const uint32_t max_chars = config_.max_tokens * config_.chars_per_token;

    if (block.text.size() <= max_chars) {
        return {block};
    }

    // Split the block text into lines, then group at blank-line boundaries
    // such that each sub-block stays under max_chars.
    std::vector<std::string> lines;
    {
        std::istringstream iss(block.text);
        std::string line;
        while (std::getline(iss, line)) {
            lines.push_back(std::move(line));
        }
    }

    std::vector<RawBlock> result;
    std::string current_text;
    uint32_t sub_start_line = block.start_line;  // 1-based
    uint32_t line_cursor = 0;

    auto flush = [&]() {
        if (current_text.empty()) return;
        uint32_t sub_end_line = sub_start_line + line_cursor - 1;
        // Clamp end line to the block's end line.
        if (sub_end_line > block.end_line) {
            sub_end_line = block.end_line;
        }
        result.push_back({
            std::move(current_text),
            result.empty() ? block.symbol_name : "",  // symbol only on first sub-block
            sub_start_line,
            sub_end_line
        });
        current_text.clear();
        sub_start_line = sub_end_line + 1;
        line_cursor = 0;
    };

    for (size_t i = 0; i < lines.size(); ++i) {
        bool is_blank = lines[i].find_first_not_of(" \t\r") == std::string::npos;

        // If adding this line would exceed the limit AND we are at a blank line,
        // flush.
        if (is_blank && current_text.size() + lines[i].size() + 1 > max_chars
            && !current_text.empty()) {
            flush();
        }

        current_text += lines[i];
        current_text += '\n';
        ++line_cursor;

        // If we've exceeded the limit and haven't found a blank line, do a hard
        // split at the next opportunity.  We still try to avoid splitting
        // mid-line; we flush at the end of any line that pushes us past the max.
        if (current_text.size() > max_chars * 2 && !current_text.empty()) {
            flush();
        }
    }
    flush();

    return result;
}

// ---------------------------------------------------------------------------
// Merging tiny blocks
// ---------------------------------------------------------------------------

std::vector<RegexChunker::RawBlock>
RegexChunker::merge_tiny(std::vector<RawBlock> blocks) const {
    const uint32_t min_chars = config_.min_tokens * config_.chars_per_token;

    if (blocks.size() <= 1) {
        return blocks;
    }

    std::vector<RawBlock> merged;
    merged.reserve(blocks.size());

    for (auto& block : blocks) {
        bool is_tiny = block.text.size() < min_chars;
        bool both_named = !merged.empty()
                          && !merged.back().symbol_name.empty()
                          && !block.symbol_name.empty();

        if (!merged.empty() && is_tiny && !both_named) {
            // Merge into previous block.
            auto& prev = merged.back();
            prev.text += block.text;
            prev.end_line = block.end_line;
            // Keep the previous block's symbol_name if it had one; otherwise
            // adopt the merged block's name.
            if (prev.symbol_name.empty()) {
                prev.symbol_name = std::move(block.symbol_name);
            }
        } else {
            merged.push_back(std::move(block));
        }
    }

    // Final pass: if the last block ended up tiny, merge it backwards —
    // but only if both blocks aren't named (to preserve symbol identity).
    if (merged.size() > 1 && merged.back().text.size() < min_chars) {
        bool both_named = !merged[merged.size() - 2].symbol_name.empty()
                          && !merged.back().symbol_name.empty();
        if (!both_named) {
            auto last = std::move(merged.back());
            merged.pop_back();
            auto& prev = merged.back();
            prev.text += last.text;
            prev.end_line = last.end_line;
        }
    }

    return merged;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

Chunk RegexChunker::make_chunk(const RawBlock& block,
                               const std::filesystem::path& file_path,
                               const std::string& language) const {
    Chunk chunk;
    chunk.source_text  = block.text;
    chunk.file_path    = file_path;
    chunk.start_line   = block.start_line;
    chunk.end_line     = block.end_line;
    chunk.language     = language;
    chunk.symbol_name  = block.symbol_name;
    chunk.chunk_id     = generate_chunk_id(file_path, block.start_line, block.end_line);
    return chunk;
}

uint32_t RegexChunker::estimate_tokens(const std::string& text) const {
    if (config_.chars_per_token == 0) return 0;
    return static_cast<uint32_t>(text.size() / config_.chars_per_token);
}

// ---------------------------------------------------------------------------
// Public API: chunk_file
// ---------------------------------------------------------------------------

std::vector<Chunk> RegexChunker::chunk_file(const std::filesystem::path& file_path) {
    // Read the entire file into memory.
    std::ifstream ifs(file_path, std::ios::binary);
    if (!ifs.is_open()) {
        return {};
    }

    std::string content((std::istreambuf_iterator<char>(ifs)),
                         std::istreambuf_iterator<char>());
    ifs.close();

    if (content.empty()) {
        return {};
    }

    return chunk_string(content, file_path);
}

// ---------------------------------------------------------------------------
// Public API: chunk_string
// ---------------------------------------------------------------------------

std::vector<Chunk> RegexChunker::chunk_string(const std::string& source,
                                              const std::filesystem::path& file_path) {
    if (source.empty()) {
        return {};
    }

    // Split source into lines.
    std::vector<std::string> lines;
    {
        std::istringstream iss(source);
        std::string line;
        while (std::getline(iss, line)) {
            // Remove trailing \r if present (Windows line endings).
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            lines.push_back(std::move(line));
        }
    }

    if (lines.empty()) {
        return {};
    }

    std::string language = detect_language(file_path);

    // 1. Find boundaries.
    auto boundaries = find_boundaries(lines, language);

    // 2. Split at boundaries to get raw blocks.
    auto blocks = split_at_boundaries(lines, boundaries);

    // 3. Split any oversized blocks.
    {
        std::vector<RawBlock> split_blocks;
        split_blocks.reserve(blocks.size());
        for (auto& block : blocks) {
            auto sub = split_oversized(block);
            for (auto& s : sub) {
                split_blocks.push_back(std::move(s));
            }
        }
        blocks = std::move(split_blocks);
    }

    // 4. Merge tiny blocks.
    blocks = merge_tiny(std::move(blocks));

    // 5. Convert to Chunk objects.
    std::vector<Chunk> chunks;
    chunks.reserve(blocks.size());
    for (const auto& block : blocks) {
        // Skip blocks that are entirely whitespace.
        if (block.text.find_first_not_of(" \t\r\n") == std::string::npos) {
            continue;
        }
        chunks.push_back(make_chunk(block, file_path, language));
    }

    return chunks;
}

} // namespace engram
