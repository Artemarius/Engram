/// @file test_chunker.cpp
/// @brief Google Test suite for the chunker module.

#include <gtest/gtest.h>

#include "../src/chunker/chunker.hpp"
#include "../src/chunker/regex_chunker.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// RAII wrapper that writes content to a temporary file and removes it on
/// destruction.
class TempFile {
public:
    TempFile(const std::string& content, const std::string& extension) {
        // Use the system temp directory to avoid polluting the source tree.
        auto tmp_dir = fs::temp_directory_path();
        // Build a reasonably unique file name.
        std::string name = "engram_test_" + std::to_string(
            std::hash<std::string>{}(content + extension)) + extension;
        path_ = tmp_dir / name;
        std::ofstream ofs(path_, std::ios::binary);
        ofs.write(content.data(), static_cast<std::streamsize>(content.size()));
        ofs.close();
    }

    ~TempFile() {
        std::error_code ec;
        fs::remove(path_, ec);
    }

    const fs::path& path() const { return path_; }

    // Non-copyable.
    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;

private:
    fs::path path_;
};

// ---------------------------------------------------------------------------
// generate_chunk_id tests
// ---------------------------------------------------------------------------

TEST(ChunkIdTest, DeterministicForSameInputs) {
    fs::path p = "src/foo.cpp";
    auto id1 = engram::generate_chunk_id(p, 10, 20);
    auto id2 = engram::generate_chunk_id(p, 10, 20);
    EXPECT_EQ(id1, id2);
    EXPECT_EQ(id1.size(), 16u);
}

TEST(ChunkIdTest, DifferentForDifferentLines) {
    fs::path p = "src/foo.cpp";
    auto id1 = engram::generate_chunk_id(p, 10, 20);
    auto id2 = engram::generate_chunk_id(p, 10, 30);
    EXPECT_NE(id1, id2);
}

TEST(ChunkIdTest, DifferentForDifferentPaths) {
    auto id1 = engram::generate_chunk_id("src/foo.cpp", 1, 10);
    auto id2 = engram::generate_chunk_id("src/bar.cpp", 1, 10);
    EXPECT_NE(id1, id2);
}

TEST(ChunkIdTest, HexFormat) {
    auto id = engram::generate_chunk_id("x.py", 1, 2);
    EXPECT_EQ(id.size(), 16u);
    for (char c : id) {
        EXPECT_TRUE((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))
            << "Non-hex character in chunk_id: " << c;
    }
}

// ---------------------------------------------------------------------------
// C++ chunking tests
// ---------------------------------------------------------------------------

static const char* kSimpleCpp = R"(
#include <iostream>
#include <string>

namespace myns {

class Greeter {
public:
    void greet(const std::string& name) {
        std::cout << "Hello, " << name << "\n";
    }
};

int add(int a, int b) {
    return a + b;
}

double multiply(double x, double y) {
    return x * y;
}

} // namespace myns
)";

TEST(RegexChunkerCpp, BasicChunking) {
    TempFile tmp(kSimpleCpp, ".cpp");
    engram::RegexChunker chunker;
    auto chunks = chunker.chunk_file(tmp.path());

    // Should produce at least one chunk.
    ASSERT_FALSE(chunks.empty());

    // Verify all chunks have the expected language.
    for (const auto& c : chunks) {
        EXPECT_EQ(c.language, "cpp");
    }

    // Verify chunk fields are populated.
    for (const auto& c : chunks) {
        EXPECT_FALSE(c.source_text.empty());
        EXPECT_EQ(c.file_path, tmp.path());
        EXPECT_GT(c.start_line, 0u);
        EXPECT_GE(c.end_line, c.start_line);
        EXPECT_FALSE(c.chunk_id.empty());
        EXPECT_EQ(c.chunk_id.size(), 16u);
    }
}

TEST(RegexChunkerCpp, FindsSymbols) {
    engram::RegexChunker chunker;
    auto chunks = chunker.chunk_string(kSimpleCpp, "test.cpp");

    // Collect all symbol names.
    std::vector<std::string> symbols;
    for (const auto& c : chunks) {
        if (!c.symbol_name.empty()) {
            symbols.push_back(c.symbol_name);
        }
    }

    // We expect to find at least some of: myns, Greeter, greet, add, multiply.
    // The regex may not catch every one perfectly, but "add" and "multiply"
    // should definitely be identified since they are straightforward free
    // functions.
    auto has = [&](const std::string& name) {
        return std::find(symbols.begin(), symbols.end(), name) != symbols.end();
    };
    EXPECT_TRUE(has("add")) << "Expected to find symbol 'add'";
    EXPECT_TRUE(has("multiply")) << "Expected to find symbol 'multiply'";
}

TEST(RegexChunkerCpp, ChunkIdsDeterministic) {
    engram::RegexChunker chunker;
    auto chunks1 = chunker.chunk_string(kSimpleCpp, "test.cpp");
    auto chunks2 = chunker.chunk_string(kSimpleCpp, "test.cpp");

    ASSERT_EQ(chunks1.size(), chunks2.size());
    for (size_t i = 0; i < chunks1.size(); ++i) {
        EXPECT_EQ(chunks1[i].chunk_id, chunks2[i].chunk_id);
        EXPECT_EQ(chunks1[i].start_line, chunks2[i].start_line);
        EXPECT_EQ(chunks1[i].end_line, chunks2[i].end_line);
    }
}

TEST(RegexChunkerCpp, CoversEntireFile) {
    engram::RegexChunker chunker;
    auto chunks = chunker.chunk_string(kSimpleCpp, "test.cpp");

    ASSERT_FALSE(chunks.empty());

    // First chunk should start at line 1.
    EXPECT_EQ(chunks.front().start_line, 1u);

    // Chunks should be contiguous (no gaps, no overlaps).
    for (size_t i = 1; i < chunks.size(); ++i) {
        EXPECT_EQ(chunks[i].start_line, chunks[i - 1].end_line + 1)
            << "Gap or overlap between chunk " << (i - 1) << " and " << i;
    }
}

// ---------------------------------------------------------------------------
// Python chunking tests
// ---------------------------------------------------------------------------

static const char* kSimplePython = R"(import os
import sys

class Calculator:
    """A simple calculator."""

    def __init__(self):
        self.history = []

    def add(self, a, b):
        result = a + b
        self.history.append(result)
        return result

    def subtract(self, a, b):
        result = a - b
        self.history.append(result)
        return result


def main():
    calc = Calculator()
    print(calc.add(1, 2))
    print(calc.subtract(5, 3))


if __name__ == "__main__":
    main()
)";

TEST(RegexChunkerPython, DetectsLanguage) {
    engram::RegexChunker chunker;
    auto chunks = chunker.chunk_string(kSimplePython, "calculator.py");

    ASSERT_FALSE(chunks.empty());
    for (const auto& c : chunks) {
        EXPECT_EQ(c.language, "python");
    }
}

TEST(RegexChunkerPython, FindsPythonSymbols) {
    engram::RegexChunker chunker;
    auto chunks = chunker.chunk_string(kSimplePython, "calculator.py");

    std::vector<std::string> symbols;
    for (const auto& c : chunks) {
        if (!c.symbol_name.empty()) {
            symbols.push_back(c.symbol_name);
        }
    }

    auto has = [&](const std::string& name) {
        return std::find(symbols.begin(), symbols.end(), name) != symbols.end();
    };

    // Should find at least the class and some methods.
    EXPECT_TRUE(has("Calculator")) << "Expected to find symbol 'Calculator'";
    EXPECT_TRUE(has("main")) << "Expected to find symbol 'main'";
}

TEST(RegexChunkerPython, FieldsPopulated) {
    TempFile tmp(kSimplePython, ".py");
    engram::RegexChunker chunker;
    auto chunks = chunker.chunk_file(tmp.path());

    for (const auto& c : chunks) {
        EXPECT_FALSE(c.source_text.empty());
        EXPECT_EQ(c.file_path, tmp.path());
        EXPECT_GT(c.start_line, 0u);
        EXPECT_GE(c.end_line, c.start_line);
        EXPECT_EQ(c.chunk_id.size(), 16u);
    }
}

// ---------------------------------------------------------------------------
// Oversized function splitting test
// ---------------------------------------------------------------------------

TEST(RegexChunkerSplit, LargeFunctionGetsSplit) {
    // Create a Python function that is far too large for a single chunk.
    // With default config: max_tokens = 500, chars_per_token = 4 => max ~2000 chars.
    std::string large_func = "def huge_function():\n";
    for (int i = 0; i < 200; ++i) {
        large_func += "    x_" + std::to_string(i) + " = " + std::to_string(i * 42) + "\n";
        // Insert occasional blank lines so the splitter has split points.
        if (i % 20 == 19) {
            large_func += "\n";
        }
    }
    large_func += "    return x_0\n";

    engram::RegexChunker chunker;
    auto chunks = chunker.chunk_string(large_func, "large.py");

    // With ~200 lines of ~30 chars each = ~6000 chars => ~1500 tokens.
    // Should be split into multiple chunks.
    EXPECT_GT(chunks.size(), 1u) << "Large function should be split into multiple chunks";

    // Each chunk should be within the max size (with some tolerance for
    // hard-split edge cases).
    for (const auto& c : chunks) {
        // Allow up to 2x max_tokens as hard-split tolerance.
        uint32_t approx_tokens = static_cast<uint32_t>(c.source_text.size()) / 4;
        EXPECT_LE(approx_tokens, 1000u)
            << "Chunk has " << approx_tokens << " estimated tokens, expected <= 1000";
    }
}

// ---------------------------------------------------------------------------
// Tiny fragment merging test
// ---------------------------------------------------------------------------

TEST(RegexChunkerMerge, TinyUnnamedFragmentsGetMerged) {
    // Create content with many very small unnamed blocks.  Using an unknown
    // file extension triggers blank-line splitting which produces blocks
    // without symbol_name — these should be merged together.
    std::string source;
    for (int i = 0; i < 10; ++i) {
        source += "block " + std::to_string(i) + " line 1\n";
        source += "block " + std::to_string(i) + " line 2\n\n";
    }

    // With default min_tokens=50, chars_per_token=4 => min 200 chars.
    // Each block above is ~30 chars, so individually they are all tiny.
    engram::RegexChunker chunker;
    auto chunks = chunker.chunk_string(source, "data.txt");

    // We should have fewer chunks than there are blocks, because the tiny
    // unnamed ones get merged.
    EXPECT_LT(chunks.size(), 10u) << "Tiny fragments should have been merged";

    // But we should still have at least one chunk.
    EXPECT_GE(chunks.size(), 1u);
}

TEST(RegexChunkerMerge, NamedBlocksPreserveSymbols) {
    // Named blocks (functions) should not be merged into a predecessor that
    // already has a symbol_name, even if both are tiny.
    std::string source;
    for (int i = 0; i < 5; ++i) {
        source += "def func_" + std::to_string(i) + "():\n";
        source += "    pass\n\n";
    }

    engram::RegexChunker chunker;
    auto chunks = chunker.chunk_string(source, "named.py");

    // Each function is a named boundary — symbols should be preserved.
    std::vector<std::string> symbols;
    for (const auto& c : chunks) {
        if (!c.symbol_name.empty()) {
            symbols.push_back(c.symbol_name);
        }
    }
    EXPECT_GE(symbols.size(), 4u)
        << "Most named function symbols should be preserved";
}

// ---------------------------------------------------------------------------
// Edge case: empty file
// ---------------------------------------------------------------------------

TEST(RegexChunkerEdge, EmptyFileReturnsEmpty) {
    engram::RegexChunker chunker;
    auto chunks = chunker.chunk_string("", "empty.cpp");
    EXPECT_TRUE(chunks.empty());
}

TEST(RegexChunkerEdge, WhitespaceOnlyReturnsEmpty) {
    engram::RegexChunker chunker;
    auto chunks = chunker.chunk_string("   \n\n  \t  \n", "blank.py");
    EXPECT_TRUE(chunks.empty());
}

// ---------------------------------------------------------------------------
// Edge case: nonexistent file
// ---------------------------------------------------------------------------

TEST(RegexChunkerEdge, NonexistentFileReturnsEmpty) {
    engram::RegexChunker chunker;
    auto chunks = chunker.chunk_file(fs::path("this/does/not/exist.cpp"));
    EXPECT_TRUE(chunks.empty());
}

// ---------------------------------------------------------------------------
// Unknown language falls back to blank-line splitting
// ---------------------------------------------------------------------------

TEST(RegexChunkerFallback, UnknownExtensionUsesBlankLineSplit) {
    std::string source =
        "first block line 1\n"
        "first block line 2\n"
        "\n"
        "second block line 1\n"
        "second block line 2\n"
        "second block line 3\n"
        "\n"
        "third block line 1\n";

    // Use a custom config with very low min_tokens so the blocks don't merge.
    engram::RegexChunkerConfig cfg;
    cfg.min_tokens = 1;
    cfg.max_tokens = 5000;
    engram::RegexChunker chunker(cfg);
    auto chunks = chunker.chunk_string(source, "data.xyz");

    // Should detect as "unknown" language.
    for (const auto& c : chunks) {
        EXPECT_EQ(c.language, "unknown");
    }

    // With blank-line splitting and min_tokens=1, we expect 3 blocks.
    EXPECT_EQ(chunks.size(), 3u);
}

// ---------------------------------------------------------------------------
// JavaScript / TypeScript detection
// ---------------------------------------------------------------------------

TEST(RegexChunkerLanguage, DetectsJavaScript) {
    const char* js_source =
        "function greet(name) {\n"
        "    console.log('Hello ' + name);\n"
        "}\n"
        "\n"
        "class Animal {\n"
        "    constructor(name) {\n"
        "        this.name = name;\n"
        "    }\n"
        "}\n";

    engram::RegexChunker chunker;
    auto chunks = chunker.chunk_string(js_source, "app.js");

    ASSERT_FALSE(chunks.empty());
    EXPECT_EQ(chunks[0].language, "javascript");

    // Should find 'greet' symbol.
    bool found_greet = false;
    for (const auto& c : chunks) {
        if (c.symbol_name == "greet") found_greet = true;
    }
    EXPECT_TRUE(found_greet);
}

TEST(RegexChunkerLanguage, DetectsTypeScript) {
    engram::RegexChunker chunker;
    auto chunks = chunker.chunk_string("export function foo() {}\n", "mod.ts");
    ASSERT_FALSE(chunks.empty());
    EXPECT_EQ(chunks[0].language, "typescript");
}

// ---------------------------------------------------------------------------
// Custom configuration test
// ---------------------------------------------------------------------------

TEST(RegexChunkerConfig, RespectsCustomMinMax) {
    // Build content with functions large enough that a small max_tokens
    // causes split_oversized to split them, while a large max_tokens does not.
    std::string source;
    for (int i = 0; i < 5; ++i) {
        source += "def func_" + std::to_string(i) + "(x):\n";
        for (int j = 0; j < 10; ++j) {
            source += "    variable_" + std::to_string(j)
                    + " = x + " + std::to_string(i * 10 + j) + "\n";
        }
        source += "    return variable_0\n\n";
    }

    // Big max: each ~340-char function fits easily; no splitting.
    engram::RegexChunkerConfig big_cfg;
    big_cfg.min_tokens = 1;
    big_cfg.max_tokens = 100000;  // effectively unlimited

    engram::RegexChunker big_chunker(big_cfg);
    auto big_chunks = big_chunker.chunk_string(source, "funcs.py");

    // Small max: max_chars = 30 * 4 = 120.  Each function is ~340 chars,
    // which exceeds 120, so split_oversized kicks in.
    engram::RegexChunkerConfig small_cfg;
    small_cfg.min_tokens = 1;
    small_cfg.max_tokens = 30;  // ~120 chars per chunk
    small_cfg.chars_per_token = 4;

    engram::RegexChunker small_chunker(small_cfg);
    auto small_chunks = small_chunker.chunk_string(source, "funcs.py");

    // The small-max chunker should produce more chunks than the big-max one.
    EXPECT_GT(small_chunks.size(), big_chunks.size());
}

// ---------------------------------------------------------------------------
// chunk_string vs chunk_file consistency
// ---------------------------------------------------------------------------

TEST(RegexChunkerConsistency, StringAndFileProduceSameContent) {
    TempFile tmp(kSimplePython, ".py");
    engram::RegexChunker chunker;

    auto from_file = chunker.chunk_file(tmp.path());
    auto from_string = chunker.chunk_string(kSimplePython, tmp.path());

    // The number of chunks and their text content should match.
    ASSERT_EQ(from_file.size(), from_string.size());
    for (size_t i = 0; i < from_file.size(); ++i) {
        EXPECT_EQ(from_file[i].source_text, from_string[i].source_text);
        EXPECT_EQ(from_file[i].start_line, from_string[i].start_line);
        EXPECT_EQ(from_file[i].end_line, from_string[i].end_line);
        EXPECT_EQ(from_file[i].symbol_name, from_string[i].symbol_name);
        EXPECT_EQ(from_file[i].language, from_string[i].language);
    }
}

// ---------------------------------------------------------------------------
// Polymorphism through base class pointer
// ---------------------------------------------------------------------------

TEST(ChunkerInterface, WorksThroughBasePointer) {
    std::unique_ptr<engram::Chunker> chunker = std::make_unique<engram::RegexChunker>();
    TempFile tmp(kSimpleCpp, ".cpp");
    auto chunks = chunker->chunk_file(tmp.path());
    EXPECT_FALSE(chunks.empty());
}

// ===========================================================================
// Chunk store serialization tests (file_content_hash round-trip)
// ===========================================================================

#include "../src/chunker/chunk_store.hpp"
#include <unordered_map>

TEST(ChunkStore, RoundTripWithFileContentHash) {
    // Build a chunk map with the new file_content_hash field populated.
    std::unordered_map<std::string, engram::Chunk> original;
    engram::Chunk c;
    c.chunk_id          = "abcdef0123456789";
    c.file_path         = "src/foo.cpp";
    c.start_line        = 10;
    c.end_line          = 25;
    c.language          = "cpp";
    c.symbol_name       = "do_thing";
    c.source_text       = "void do_thing() { return; }";
    c.file_content_hash = "fedcba9876543210";
    original[c.chunk_id] = c;

    // Save to a temp file.
    auto tmp_dir = fs::temp_directory_path();
    auto path = tmp_dir / "engram_test_chunk_store_hash.json";

    ASSERT_TRUE(engram::save_chunks(path, original));

    // Load back.
    std::unordered_map<std::string, engram::Chunk> loaded;
    ASSERT_TRUE(engram::load_chunks(path, loaded));

    ASSERT_EQ(loaded.size(), 1u);
    auto it = loaded.find("abcdef0123456789");
    ASSERT_NE(it, loaded.end());
    EXPECT_EQ(it->second.file_content_hash, "fedcba9876543210");
    EXPECT_EQ(it->second.symbol_name, "do_thing");
    EXPECT_EQ(it->second.source_text, "void do_thing() { return; }");

    // Cleanup.
    std::error_code ec;
    fs::remove(path, ec);
}

TEST(ChunkStore, BackwardsCompatMissingHash) {
    // Simulate a legacy chunks.json that lacks the file_content_hash field.
    auto tmp_dir = fs::temp_directory_path();
    auto path = tmp_dir / "engram_test_chunk_store_legacy.json";

    // Write JSON without file_content_hash.
    {
        nlohmann::json j = nlohmann::json::object();
        j["id_0000000000000001"] = {
            {"chunk_id",    "id_0000000000000001"},
            {"file_path",   "src/old.cpp"},
            {"start_line",  1},
            {"end_line",    10},
            {"language",    "cpp"},
            {"symbol_name", "old_func"},
            {"source_text", "int old_func() { return 42; }"}
        };
        std::ofstream ofs(path);
        ofs << j.dump(2);
    }

    // Load should succeed and file_content_hash should be empty.
    std::unordered_map<std::string, engram::Chunk> loaded;
    ASSERT_TRUE(engram::load_chunks(path, loaded));
    ASSERT_EQ(loaded.size(), 1u);
    EXPECT_TRUE(loaded.begin()->second.file_content_hash.empty());

    std::error_code ec;
    fs::remove(path, ec);
}

TEST(ChunkStore, RoundTripEmptyHash) {
    // Chunks with empty file_content_hash should round-trip correctly.
    std::unordered_map<std::string, engram::Chunk> original;
    engram::Chunk c;
    c.chunk_id          = "0000000000000001";
    c.file_path         = "src/bar.py";
    c.start_line        = 1;
    c.end_line          = 5;
    c.language          = "python";
    c.symbol_name       = "bar";
    c.source_text       = "def bar(): pass";
    c.file_content_hash = "";  // explicitly empty
    original[c.chunk_id] = c;

    auto tmp_dir = fs::temp_directory_path();
    auto path = tmp_dir / "engram_test_chunk_store_empty_hash.json";

    ASSERT_TRUE(engram::save_chunks(path, original));

    std::unordered_map<std::string, engram::Chunk> loaded;
    ASSERT_TRUE(engram::load_chunks(path, loaded));
    ASSERT_EQ(loaded.size(), 1u);
    EXPECT_TRUE(loaded.begin()->second.file_content_hash.empty());

    std::error_code ec;
    fs::remove(path, ec);
}
