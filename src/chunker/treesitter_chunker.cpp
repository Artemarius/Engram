/// @file treesitter_chunker.cpp
/// @brief AST-aware code chunker implementation using tree-sitter.

#include "treesitter_chunker.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>

#include <spdlog/spdlog.h>

extern "C" {
#include <tree_sitter/api.h>
}

// ---------------------------------------------------------------------------
// External language function declarations (provided by grammar static libs)
// ---------------------------------------------------------------------------
extern "C" {
const TSLanguage* tree_sitter_cpp(void);
const TSLanguage* tree_sitter_python(void);
const TSLanguage* tree_sitter_javascript(void);
const TSLanguage* tree_sitter_typescript(void);
const TSLanguage* tree_sitter_tsx(void);
const TSLanguage* tree_sitter_java(void);
const TSLanguage* tree_sitter_rust(void);
const TSLanguage* tree_sitter_go(void);
const TSLanguage* tree_sitter_ruby(void);
const TSLanguage* tree_sitter_c_sharp(void);
}

namespace engram {

// =========================================================================
// RAII deleters for tree-sitter objects
// =========================================================================

struct ParserDeleter {
    void operator()(TSParser* p) const { if (p) ts_parser_delete(p); }
};
struct TreeDeleter {
    void operator()(TSTree* t) const { if (t) ts_tree_delete(t); }
};
struct QueryCursorDeleter {
    void operator()(TSQueryCursor* c) const { if (c) ts_query_cursor_delete(c); }
};

using ParserPtr      = std::unique_ptr<TSParser, ParserDeleter>;
using TreePtr        = std::unique_ptr<TSTree, TreeDeleter>;
using QueryCursorPtr = std::unique_ptr<TSQueryCursor, QueryCursorDeleter>;

// =========================================================================
// Query pattern strings per language
// =========================================================================
//
// Each pattern matches a named construct.  We only use `@definition` as the
// capture name — name extraction is done programmatically from the AST node.

static const char* query_string_for(const std::string& lang) {
    // C++
    static const char* cpp_query = R"(
        (function_definition) @definition
        (class_specifier) @definition
        (struct_specifier) @definition
        (namespace_definition) @definition
    )";

    // Python
    static const char* python_query = R"(
        (function_definition) @definition
        (class_definition) @definition
    )";

    // JavaScript
    static const char* js_query = R"(
        (function_declaration) @definition
        (class_declaration) @definition
        (method_definition) @definition
        (export_statement (function_declaration) @definition)
        (export_statement (class_declaration) @definition)
    )";

    // TypeScript (also used for TSX)
    static const char* ts_query = R"(
        (function_declaration) @definition
        (class_declaration) @definition
        (method_definition) @definition
        (interface_declaration) @definition
        (type_alias_declaration) @definition
        (enum_declaration) @definition
        (export_statement (function_declaration) @definition)
        (export_statement (class_declaration) @definition)
    )";

    // Java
    static const char* java_query = R"(
        (class_declaration) @definition
        (method_declaration) @definition
        (interface_declaration) @definition
        (constructor_declaration) @definition
    )";

    // Rust
    static const char* rust_query = R"(
        (function_item) @definition
        (struct_item) @definition
        (enum_item) @definition
        (trait_item) @definition
        (impl_item) @definition
    )";

    // Go
    static const char* go_query = R"(
        (function_declaration) @definition
        (method_declaration) @definition
        (type_declaration) @definition
    )";

    // Ruby
    static const char* ruby_query = R"(
        (method) @definition
        (class) @definition
        (module) @definition
    )";

    // C#
    static const char* csharp_query = R"(
        (class_declaration) @definition
        (method_declaration) @definition
        (struct_declaration) @definition
        (interface_declaration) @definition
        (namespace_declaration) @definition
    )";

    static const std::unordered_map<std::string, const char*> table = {
        {"cpp",        cpp_query},
        {"python",     python_query},
        {"javascript", js_query},
        {"typescript", ts_query},
        {"tsx",        ts_query},    // TSX reuses TypeScript patterns
        {"java",       java_query},
        {"rust",       rust_query},
        {"go",         go_query},
        {"ruby",       ruby_query},
        {"csharp",     csharp_query},
    };

    auto it = table.find(lang);
    return (it != table.end()) ? it->second : nullptr;
}

// =========================================================================
// Language factory table
// =========================================================================

struct LangFactory {
    const TSLanguage* (*create)();
    const char* chunk_language;   // tag written into Chunk.language
};

static const std::unordered_map<std::string, LangFactory>& lang_factories() {
    static const std::unordered_map<std::string, LangFactory> table = {
        {"cpp",        {tree_sitter_cpp,        "cpp"}},
        {"python",     {tree_sitter_python,     "python"}},
        {"javascript", {tree_sitter_javascript, "javascript"}},
        {"typescript", {tree_sitter_typescript, "typescript"}},
        {"tsx",        {tree_sitter_tsx,        "typescript"}},  // TSX → "typescript" in metadata
        {"java",       {tree_sitter_java,       "java"}},
        {"rust",       {tree_sitter_rust,       "rust"}},
        {"go",         {tree_sitter_go,         "go"}},
        {"ruby",       {tree_sitter_ruby,       "ruby"}},
        {"csharp",     {tree_sitter_c_sharp,    "csharp"}},
    };
    return table;
}

// =========================================================================
// Name extraction from AST nodes
// =========================================================================

/// Extract the raw text of an AST node.
static std::string node_text(TSNode node, const std::string& source) {
    uint32_t start = ts_node_start_byte(node);
    uint32_t end   = ts_node_end_byte(node);
    if (start >= source.size() || end > source.size() || start >= end) return "";
    return source.substr(start, end - start);
}

/// Extract the symbol name from a matched AST node.
///
/// Uses multiple strategies depending on node type:
///   1. "name" field (works for most constructs)
///   2. Declarator chain traversal (C++ function_definition)
///   3. type_spec navigation (Go type_declaration)
///   4. "type" field (Rust impl_item)
///   5. variable_declarator navigation (JS/TS lexical_declaration)
static std::string extract_name(TSNode node, const std::string& source) {
    const char* type = ts_node_type(node);

    // --- Strategy 1: direct "name" field ---
    TSNode name_node = ts_node_child_by_field_name(node, "name", 4);
    if (!ts_node_is_null(name_node)) {
        std::string name = node_text(name_node, source);
        // Strip qualifications: "Foo::bar" → "bar"
        auto pos = name.rfind("::");
        if (pos != std::string::npos) name = name.substr(pos + 2);
        pos = name.rfind('.');
        if (pos != std::string::npos) name = name.substr(pos + 1);
        return name;
    }

    // --- Strategy 2: C++ declarator chain ---
    // function_definition → [pointer_declarator →]* function_declarator → declarator
    TSNode decl = ts_node_child_by_field_name(node, "declarator", 10);
    if (!ts_node_is_null(decl)) {
        // Unwrap pointer_declarator / reference_declarator wrappers.
        while (true) {
            const char* dt = ts_node_type(decl);
            if (std::strcmp(dt, "pointer_declarator") == 0 ||
                std::strcmp(dt, "reference_declarator") == 0) {
                TSNode inner = ts_node_child_by_field_name(decl, "declarator", 10);
                if (ts_node_is_null(inner)) break;
                decl = inner;
            } else {
                break;
            }
        }

        if (std::strcmp(ts_node_type(decl), "function_declarator") == 0) {
            TSNode inner_decl = ts_node_child_by_field_name(decl, "declarator", 10);
            if (!ts_node_is_null(inner_decl)) {
                std::string name = node_text(inner_decl, source);
                auto pos = name.rfind("::");
                if (pos != std::string::npos) name = name.substr(pos + 2);
                return name;
            }
        }

        // Fallback: name field on the declarator itself.
        name_node = ts_node_child_by_field_name(decl, "name", 4);
        if (!ts_node_is_null(name_node)) {
            return node_text(name_node, source);
        }
    }

    // --- Strategy 3: Go type_declaration → type_spec → name ---
    if (std::strcmp(type, "type_declaration") == 0) {
        uint32_t n = ts_node_child_count(node);
        for (uint32_t i = 0; i < n; i++) {
            TSNode child = ts_node_child(node, i);
            if (std::strcmp(ts_node_type(child), "type_spec") == 0) {
                name_node = ts_node_child_by_field_name(child, "name", 4);
                if (!ts_node_is_null(name_node)) {
                    return node_text(name_node, source);
                }
            }
        }
    }

    // --- Strategy 4: Rust impl_item → "type" field ---
    if (std::strcmp(type, "impl_item") == 0) {
        TSNode type_node = ts_node_child_by_field_name(node, "type", 4);
        if (!ts_node_is_null(type_node)) {
            return node_text(type_node, source);
        }
    }

    // --- Strategy 5: JS/TS lexical/variable declaration ---
    if (std::strcmp(type, "lexical_declaration") == 0 ||
        std::strcmp(type, "variable_declaration") == 0) {
        uint32_t n = ts_node_child_count(node);
        for (uint32_t i = 0; i < n; i++) {
            TSNode child = ts_node_child(node, i);
            if (std::strcmp(ts_node_type(child), "variable_declarator") == 0) {
                name_node = ts_node_child_by_field_name(child, "name", 4);
                if (!ts_node_is_null(name_node)) {
                    return node_text(name_node, source);
                }
            }
        }
    }

    return "";
}

// =========================================================================
// Language detection (file extension → language tag)
// =========================================================================

std::string TreeSitterChunker::detect_language(const std::filesystem::path& file_path) {
    static const std::unordered_map<std::string, std::string> ext_map = {
        {".cpp",  "cpp"},  {".cxx",  "cpp"},  {".cc",  "cpp"},
        {".c",    "cpp"},  {".hpp",  "cpp"},  {".hxx", "cpp"},
        {".h",    "cpp"},
        {".py",   "python"}, {".pyw", "python"},
        {".js",   "javascript"}, {".jsx", "javascript"}, {".mjs", "javascript"},
        {".ts",   "typescript"},
        {".tsx",  "tsx"},
        {".java", "java"},
        {".rs",   "rust"},
        {".go",   "go"},
        {".rb",   "ruby"},
        {".cs",   "csharp"},
    };

    std::string ext = file_path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    auto it = ext_map.find(ext);
    return (it != ext_map.end()) ? it->second : "unknown";
}

// =========================================================================
// Construction / destruction
// =========================================================================

TreeSitterChunker::TreeSitterChunker()
    : config_{} {
    init_languages();
}

TreeSitterChunker::TreeSitterChunker(RegexChunkerConfig config)
    : config_(config) {
    init_languages();
}

TreeSitterChunker::~TreeSitterChunker() {
    for (auto& [lang, info] : languages_) {
        if (info.query) {
            ts_query_delete(info.query);
            info.query = nullptr;
        }
    }
}

void TreeSitterChunker::init_languages() {
    const auto& factories = lang_factories();

    for (const auto& [lang, factory] : factories) {
        const char* query_src = query_string_for(lang);
        if (!query_src) continue;

        const TSLanguage* ts_lang = factory.create();

        uint32_t     error_offset = 0;
        TSQueryError error_type   = TSQueryErrorNone;
        TSQuery*     query = ts_query_new(
            ts_lang, query_src,
            static_cast<uint32_t>(std::strlen(query_src)),
            &error_offset, &error_type);

        if (!query) {
            spdlog::warn("tree-sitter: failed to compile query for '{}' "
                         "(offset {}, error type {})",
                         lang, error_offset, static_cast<int>(error_type));
            continue;
        }

        languages_[lang] = {ts_lang, query, factory.chunk_language};
    }

    spdlog::debug("tree-sitter: initialised {} language parsers", languages_.size());
}

// =========================================================================
// Token estimation
// =========================================================================

uint32_t TreeSitterChunker::estimate_tokens(const std::string& text) const {
    if (config_.chars_per_token == 0) return 0;
    return static_cast<uint32_t>(text.size() / config_.chars_per_token);
}

// =========================================================================
// split_oversized  (same logic as RegexChunker)
// =========================================================================

std::vector<TreeSitterChunker::RawBlock>
TreeSitterChunker::split_oversized(const RawBlock& block) const {
    const uint32_t max_chars = config_.max_tokens * config_.chars_per_token;

    if (block.text.size() <= max_chars) {
        return {block};
    }

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
    uint32_t sub_start_line = block.start_line;
    uint32_t line_cursor    = 0;

    auto flush = [&]() {
        if (current_text.empty()) return;
        uint32_t sub_end_line = sub_start_line + line_cursor - 1;
        if (sub_end_line > block.end_line) sub_end_line = block.end_line;
        result.push_back({
            std::move(current_text),
            result.empty() ? block.symbol_name : "",
            sub_start_line,
            sub_end_line
        });
        current_text.clear();
        sub_start_line = sub_end_line + 1;
        line_cursor    = 0;
    };

    for (size_t i = 0; i < lines.size(); ++i) {
        bool is_blank = lines[i].find_first_not_of(" \t\r") == std::string::npos;

        if (is_blank && current_text.size() + lines[i].size() + 1 > max_chars
            && !current_text.empty()) {
            flush();
        }

        current_text += lines[i];
        current_text += '\n';
        ++line_cursor;

        if (current_text.size() > max_chars * 2 && !current_text.empty()) {
            flush();
        }
    }
    flush();

    return result;
}

// =========================================================================
// merge_tiny  (same logic as RegexChunker)
// =========================================================================

std::vector<TreeSitterChunker::RawBlock>
TreeSitterChunker::merge_tiny(std::vector<RawBlock> blocks) const {
    const uint32_t min_chars = config_.min_tokens * config_.chars_per_token;

    if (blocks.size() <= 1) return blocks;

    std::vector<RawBlock> merged;
    merged.reserve(blocks.size());

    for (auto& block : blocks) {
        bool is_tiny    = block.text.size() < min_chars;
        bool both_named = !merged.empty()
                          && !merged.back().symbol_name.empty()
                          && !block.symbol_name.empty();

        if (!merged.empty() && is_tiny && !both_named) {
            auto& prev   = merged.back();
            prev.text   += block.text;
            prev.end_line = block.end_line;
            if (prev.symbol_name.empty()) {
                prev.symbol_name = std::move(block.symbol_name);
            }
        } else {
            merged.push_back(std::move(block));
        }
    }

    // Final pass: if last block is tiny, merge backwards.
    if (merged.size() > 1 && merged.back().text.size() < min_chars) {
        bool both_named = !merged[merged.size() - 2].symbol_name.empty()
                          && !merged.back().symbol_name.empty();
        if (!both_named) {
            auto last = std::move(merged.back());
            merged.pop_back();
            auto& prev   = merged.back();
            prev.text   += last.text;
            prev.end_line = last.end_line;
        }
    }

    return merged;
}

// =========================================================================
// Fallback chunker (delegates to RegexChunker)
// =========================================================================

std::vector<Chunk> TreeSitterChunker::fallback_chunk(
    const std::string& source,
    const std::filesystem::path& file_path)
{
    RegexChunker fallback(config_);
    return fallback.chunk_string(source, file_path);
}

// =========================================================================
// chunk_file
// =========================================================================

std::vector<Chunk> TreeSitterChunker::chunk_file(
    const std::filesystem::path& file_path)
{
    std::ifstream ifs(file_path, std::ios::binary);
    if (!ifs.is_open()) return {};

    std::string content((std::istreambuf_iterator<char>(ifs)),
                         std::istreambuf_iterator<char>());
    ifs.close();

    if (content.empty()) return {};

    return chunk_string(content, file_path);
}

// =========================================================================
// chunk_string  — core tree-sitter chunking logic
// =========================================================================

std::vector<Chunk> TreeSitterChunker::chunk_string(
    const std::string& source,
    const std::filesystem::path& file_path)
{
    if (source.empty()) return {};

    const std::string lang = detect_language(file_path);

    // ---- Look up pre-compiled language info ----
    auto lang_it = languages_.find(lang);
    if (lang_it == languages_.end()) {
        return fallback_chunk(source, file_path);
    }

    const auto& info = lang_it->second;

    // ---- Parse ----
    ParserPtr parser(ts_parser_new());
    ts_parser_set_language(parser.get(), info.language);

    TreePtr tree(ts_parser_parse_string(
        parser.get(), nullptr,
        source.c_str(), static_cast<uint32_t>(source.size())));

    if (!tree) {
        spdlog::warn("tree-sitter: parse failed for '{}'",
                     file_path.generic_string());
        return fallback_chunk(source, file_path);
    }

    TSNode root = ts_tree_root_node(tree.get());

    // ---- Run query ----
    QueryCursorPtr cursor(ts_query_cursor_new());
    ts_query_cursor_exec(cursor.get(), info.query, root);

    struct Match {
        uint32_t    start_byte;
        uint32_t    end_byte;
        uint32_t    start_line;   // 0-based (TSPoint.row)
        uint32_t    end_line;     // 0-based
        std::string name;
    };

    std::vector<Match> matches;
    {
        TSQueryMatch qm;
        while (ts_query_cursor_next_match(cursor.get(), &qm)) {
            if (qm.capture_count == 0) continue;
            TSNode node = qm.captures[0].node;

            TSPoint sp = ts_node_start_point(node);
            TSPoint ep = ts_node_end_point(node);

            matches.push_back({
                ts_node_start_byte(node),
                ts_node_end_byte(node),
                sp.row,
                ep.row,
                extract_name(node, source)
            });
        }
    }

    if (matches.empty()) {
        return fallback_chunk(source, file_path);
    }

    // ---- Sort by start position ----
    std::sort(matches.begin(), matches.end(),
              [](const Match& a, const Match& b) {
                  return a.start_byte < b.start_byte;
              });

    // ---- Remove duplicate / overlapping matches ----
    // Deduplicate by (start_byte, end_byte).
    {
        auto last = std::unique(matches.begin(), matches.end(),
            [](const Match& a, const Match& b) {
                return a.start_byte == b.start_byte && a.end_byte == b.end_byte;
            });
        matches.erase(last, matches.end());
    }

    // ---- Remove container matches (keep innermost / leaf) ----
    // A match is a "container" if it fully encloses another match.
    std::vector<bool> is_container(matches.size(), false);
    for (size_t i = 0; i < matches.size(); i++) {
        for (size_t j = 0; j < matches.size(); j++) {
            if (i != j &&
                matches[j].start_byte >= matches[i].start_byte &&
                matches[j].end_byte   <= matches[i].end_byte   &&
                !(matches[j].start_byte == matches[i].start_byte &&
                  matches[j].end_byte   == matches[i].end_byte)) {
                is_container[i] = true;
                break;
            }
        }
    }

    std::vector<Match> leaves;
    leaves.reserve(matches.size());
    for (size_t i = 0; i < matches.size(); i++) {
        if (!is_container[i]) {
            leaves.push_back(std::move(matches[i]));
        }
    }

    // ---- Build blocks (named + gap) ----
    // Count total lines.
    uint32_t total_lines = 0;
    for (char c : source) {
        if (c == '\n') total_lines++;
    }
    if (!source.empty() && source.back() != '\n') total_lines++;

    std::vector<RawBlock> blocks;
    uint32_t current_byte = 0;
    uint32_t current_line = 0;   // 0-based

    for (const auto& leaf : leaves) {
        // Gap before this match.
        if (leaf.start_byte > current_byte) {
            std::string gap_text = source.substr(current_byte,
                                                 leaf.start_byte - current_byte);
            if (gap_text.find_first_not_of(" \t\r\n") != std::string::npos) {
                blocks.push_back({
                    std::move(gap_text),
                    "",                    // no symbol
                    current_line + 1,      // 1-based
                    leaf.start_line        // 1-based (prev line before match)
                });
            }
        }

        // Named block.
        std::string block_text = source.substr(leaf.start_byte,
                                               leaf.end_byte - leaf.start_byte);
        if (!block_text.empty() && block_text.back() != '\n') {
            block_text += '\n';
        }

        blocks.push_back({
            std::move(block_text),
            leaf.name,
            leaf.start_line + 1,   // 1-based
            leaf.end_line   + 1    // 1-based
        });

        current_byte = leaf.end_byte;
        current_line = leaf.end_line + 1;
    }

    // Gap after last match.
    if (current_byte < source.size()) {
        std::string gap_text = source.substr(current_byte);
        if (gap_text.find_first_not_of(" \t\r\n") != std::string::npos) {
            blocks.push_back({
                std::move(gap_text),
                "",
                current_line + 1,
                total_lines
            });
        }
    }

    // ---- Split oversized blocks ----
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

    // ---- Merge tiny blocks ----
    blocks = merge_tiny(std::move(blocks));

    // ---- Convert to Chunk objects ----
    std::vector<Chunk> chunks;
    chunks.reserve(blocks.size());

    for (const auto& block : blocks) {
        if (block.text.find_first_not_of(" \t\r\n") == std::string::npos) {
            continue;
        }

        Chunk chunk;
        chunk.source_text = block.text;
        chunk.file_path   = file_path;
        chunk.start_line  = block.start_line;
        chunk.end_line    = block.end_line;
        chunk.language    = info.chunk_language;
        chunk.symbol_name = block.symbol_name;
        chunk.chunk_id    = generate_chunk_id(file_path, block.start_line, block.end_line);
        chunks.push_back(std::move(chunk));
    }

    return chunks;
}

} // namespace engram
