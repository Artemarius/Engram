/// @file test_treesitter_chunker.cpp
/// @brief Google Test suite for the TreeSitterChunker.
///
/// The entire file is guarded by ENGRAM_HAS_TREESITTER so it compiles
/// to nothing when tree-sitter support is disabled.

#ifdef ENGRAM_HAS_TREESITTER

#include <gtest/gtest.h>

#include "../src/chunker/chunker.hpp"
#include "../src/chunker/treesitter_chunker.hpp"
#include "../src/chunker/regex_chunker.hpp"

#include <algorithm>
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
        auto tmp_dir = fs::temp_directory_path();
        std::string name = "engram_ts_test_" + std::to_string(
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

/// Check whether any chunk in the vector has the given symbol name.
static bool has_symbol(const std::vector<engram::Chunk>& chunks,
                       const std::string& name) {
    return std::any_of(chunks.begin(), chunks.end(),
                       [&](const engram::Chunk& c) {
                           return c.symbol_name == name;
                       });
}

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

class TreeSitterChunkerTest : public ::testing::Test {
protected:
    engram::TreeSitterChunker chunker;
};

// ---------------------------------------------------------------------------
// 1. CppFunction
// ---------------------------------------------------------------------------

TEST_F(TreeSitterChunkerTest, CppFunction) {
    const char* source =
        "int add(int a, int b) {\n"
        "    return a + b;\n"
        "}\n";

    auto chunks = chunker.chunk_string(source, "add.cpp");

    ASSERT_FALSE(chunks.empty());
    EXPECT_TRUE(has_symbol(chunks, "add"))
        << "Expected to find symbol 'add'";

    for (const auto& c : chunks) {
        EXPECT_EQ(c.language, "cpp");
    }
}

// ---------------------------------------------------------------------------
// 2. CppMultipleFunctions
// ---------------------------------------------------------------------------

TEST_F(TreeSitterChunkerTest, CppMultipleFunctions) {
    const char* source =
        "int add(int a, int b) {\n"
        "    return a + b;\n"
        "}\n"
        "\n"
        "int subtract(int a, int b) {\n"
        "    return a - b;\n"
        "}\n";

    auto chunks = chunker.chunk_string(source, "math.cpp");

    // Collect named chunks.
    std::vector<std::string> symbols;
    for (const auto& c : chunks) {
        if (!c.symbol_name.empty()) {
            symbols.push_back(c.symbol_name);
        }
    }
    EXPECT_GE(symbols.size(), 2u)
        << "Expected at least two named chunks for two functions";
    EXPECT_TRUE(has_symbol(chunks, "add"));
    EXPECT_TRUE(has_symbol(chunks, "subtract"));
}

// ---------------------------------------------------------------------------
// 3. CppClassWithMethods
// ---------------------------------------------------------------------------

TEST_F(TreeSitterChunkerTest, CppClassWithMethods) {
    const char* source =
        "class Calculator {\n"
        "public:\n"
        "    int add(int a, int b) {\n"
        "        return a + b;\n"
        "    }\n"
        "\n"
        "    int multiply(int a, int b) {\n"
        "        return a * b;\n"
        "    }\n"
        "};\n";

    auto chunks = chunker.chunk_string(source, "calc.cpp");

    ASSERT_FALSE(chunks.empty());

    // Methods should be extracted as separate chunks with their own symbol names.
    EXPECT_TRUE(has_symbol(chunks, "add"))
        << "Expected to find method symbol 'add'";
    EXPECT_TRUE(has_symbol(chunks, "multiply"))
        << "Expected to find method symbol 'multiply'";
}

// ---------------------------------------------------------------------------
// 4. PythonFunction
// ---------------------------------------------------------------------------

TEST_F(TreeSitterChunkerTest, PythonFunction) {
    const char* source =
        "def greet(name):\n"
        "    return f\"Hello {name}\"\n";

    auto chunks = chunker.chunk_string(source, "greet.py");

    ASSERT_FALSE(chunks.empty());
    EXPECT_TRUE(has_symbol(chunks, "greet"))
        << "Expected to find symbol 'greet'";

    for (const auto& c : chunks) {
        EXPECT_EQ(c.language, "python");
    }
}

// ---------------------------------------------------------------------------
// 5. PythonClassAndMethod
// ---------------------------------------------------------------------------

TEST_F(TreeSitterChunkerTest, PythonClassAndMethod) {
    const char* source =
        "class Greeter:\n"
        "    def __init__(self, name):\n"
        "        self.name = name\n"
        "\n"
        "    def greet(self):\n"
        "        return f\"Hello {self.name}\"\n";

    auto chunks = chunker.chunk_string(source, "greeter.py");

    ASSERT_FALSE(chunks.empty());

    // Should find the class and/or its methods.
    std::vector<std::string> symbols;
    for (const auto& c : chunks) {
        if (!c.symbol_name.empty()) {
            symbols.push_back(c.symbol_name);
        }
    }
    EXPECT_FALSE(symbols.empty())
        << "Expected at least one named symbol from a Python class with methods";
}

// ---------------------------------------------------------------------------
// 6. JavaScriptFunction
// ---------------------------------------------------------------------------

TEST_F(TreeSitterChunkerTest, JavaScriptFunction) {
    const char* source =
        "function calculateSum(a, b) {\n"
        "    return a + b;\n"
        "}\n";

    auto chunks = chunker.chunk_string(source, "math.js");

    ASSERT_FALSE(chunks.empty());
    EXPECT_TRUE(has_symbol(chunks, "calculateSum"))
        << "Expected to find symbol 'calculateSum'";

    for (const auto& c : chunks) {
        EXPECT_EQ(c.language, "javascript");
    }
}

// ---------------------------------------------------------------------------
// 7. JavaScriptClass
// ---------------------------------------------------------------------------

TEST_F(TreeSitterChunkerTest, JavaScriptClass) {
    const char* source =
        "class Animal {\n"
        "    constructor(name) {\n"
        "        this.name = name;\n"
        "    }\n"
        "\n"
        "    speak() {\n"
        "        return this.name + ' makes a noise.';\n"
        "    }\n"
        "}\n";

    auto chunks = chunker.chunk_string(source, "animal.js");

    ASSERT_FALSE(chunks.empty());

    // Should find the class or its methods.
    std::vector<std::string> symbols;
    for (const auto& c : chunks) {
        if (!c.symbol_name.empty()) {
            symbols.push_back(c.symbol_name);
        }
    }
    EXPECT_FALSE(symbols.empty())
        << "Expected at least one named symbol from a JavaScript class";
}

// ---------------------------------------------------------------------------
// 8. TypeScriptInterface
// ---------------------------------------------------------------------------

TEST_F(TreeSitterChunkerTest, TypeScriptInterface) {
    const char* source =
        "interface Config {\n"
        "    host: string;\n"
        "    port: number;\n"
        "    debug: boolean;\n"
        "}\n";

    auto chunks = chunker.chunk_string(source, "config.ts");

    ASSERT_FALSE(chunks.empty());
    EXPECT_TRUE(has_symbol(chunks, "Config"))
        << "Expected to find symbol 'Config' for the interface";

    for (const auto& c : chunks) {
        EXPECT_EQ(c.language, "typescript");
    }
}

// ---------------------------------------------------------------------------
// 9. JavaClassAndMethod
// ---------------------------------------------------------------------------

TEST_F(TreeSitterChunkerTest, JavaClassAndMethod) {
    const char* source =
        "public class Calculator {\n"
        "    public int add(int a, int b) {\n"
        "        return a + b;\n"
        "    }\n"
        "\n"
        "    public int subtract(int a, int b) {\n"
        "        return a - b;\n"
        "    }\n"
        "}\n";

    auto chunks = chunker.chunk_string(source, "Calculator.java");

    ASSERT_FALSE(chunks.empty());

    // Should find both the class and its methods.
    std::vector<std::string> symbols;
    for (const auto& c : chunks) {
        if (!c.symbol_name.empty()) {
            symbols.push_back(c.symbol_name);
        }
    }
    EXPECT_GE(symbols.size(), 1u)
        << "Expected at least one named symbol from a Java class";
}

// ---------------------------------------------------------------------------
// 10. RustFunction
// ---------------------------------------------------------------------------

TEST_F(TreeSitterChunkerTest, RustFunction) {
    const char* source =
        "fn compute(x: i32) -> i32 {\n"
        "    x * 2\n"
        "}\n";

    auto chunks = chunker.chunk_string(source, "lib.rs");

    ASSERT_FALSE(chunks.empty());
    EXPECT_TRUE(has_symbol(chunks, "compute"))
        << "Expected to find symbol 'compute'";

    for (const auto& c : chunks) {
        EXPECT_EQ(c.language, "rust");
    }
}

// ---------------------------------------------------------------------------
// 11. RustStructAndImpl
// ---------------------------------------------------------------------------

TEST_F(TreeSitterChunkerTest, RustStructAndImpl) {
    const char* source =
        "struct Point {\n"
        "    x: f64,\n"
        "    y: f64,\n"
        "}\n"
        "\n"
        "impl Point {\n"
        "    fn new(x: f64, y: f64) -> Self {\n"
        "        Point { x, y }\n"
        "    }\n"
        "\n"
        "    fn distance(&self, other: &Point) -> f64 {\n"
        "        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()\n"
        "    }\n"
        "}\n";

    auto chunks = chunker.chunk_string(source, "point.rs");

    ASSERT_FALSE(chunks.empty());

    // Should find the struct and/or the impl methods.
    std::vector<std::string> symbols;
    for (const auto& c : chunks) {
        if (!c.symbol_name.empty()) {
            symbols.push_back(c.symbol_name);
        }
    }
    EXPECT_GE(symbols.size(), 1u)
        << "Expected at least one named symbol from Rust struct + impl";
}

// ---------------------------------------------------------------------------
// 12. GoFunction
// ---------------------------------------------------------------------------

TEST_F(TreeSitterChunkerTest, GoFunction) {
    const char* source =
        "package math\n"
        "\n"
        "func Add(a, b int) int {\n"
        "    return a + b\n"
        "}\n";

    auto chunks = chunker.chunk_string(source, "math.go");

    ASSERT_FALSE(chunks.empty());
    EXPECT_TRUE(has_symbol(chunks, "Add"))
        << "Expected to find symbol 'Add'";

    for (const auto& c : chunks) {
        EXPECT_EQ(c.language, "go");
    }
}

// ---------------------------------------------------------------------------
// 13. GoMethodWithReceiver
// ---------------------------------------------------------------------------

TEST_F(TreeSitterChunkerTest, GoMethodWithReceiver) {
    const char* source =
        "package server\n"
        "\n"
        "type Server struct {\n"
        "    port int\n"
        "}\n"
        "\n"
        "func (s *Server) Start() error {\n"
        "    return nil\n"
        "}\n";

    auto chunks = chunker.chunk_string(source, "server.go");

    ASSERT_FALSE(chunks.empty());
    EXPECT_TRUE(has_symbol(chunks, "Start"))
        << "Expected to find symbol 'Start' for Go method with receiver";
}

// ---------------------------------------------------------------------------
// 14. RubyMethod
// ---------------------------------------------------------------------------

TEST_F(TreeSitterChunkerTest, RubyMethod) {
    const char* source =
        "def hello\n"
        "  puts \"hi\"\n"
        "end\n";

    auto chunks = chunker.chunk_string(source, "greet.rb");

    ASSERT_FALSE(chunks.empty());
    EXPECT_TRUE(has_symbol(chunks, "hello"))
        << "Expected to find symbol 'hello'";

    for (const auto& c : chunks) {
        EXPECT_EQ(c.language, "ruby");
    }
}

// ---------------------------------------------------------------------------
// 15. RubyClassAndModule
// ---------------------------------------------------------------------------

TEST_F(TreeSitterChunkerTest, RubyClassAndModule) {
    const char* source =
        "module Greetings\n"
        "  class Greeter\n"
        "    def initialize(name)\n"
        "      @name = name\n"
        "    end\n"
        "\n"
        "    def greet\n"
        "      puts \"Hello, #{@name}\"\n"
        "    end\n"
        "  end\n"
        "end\n";

    auto chunks = chunker.chunk_string(source, "greetings.rb");

    ASSERT_FALSE(chunks.empty());

    // Should find symbols from the module/class/methods.
    std::vector<std::string> symbols;
    for (const auto& c : chunks) {
        if (!c.symbol_name.empty()) {
            symbols.push_back(c.symbol_name);
        }
    }
    EXPECT_GE(symbols.size(), 1u)
        << "Expected at least one named symbol from Ruby module/class";
}

// ---------------------------------------------------------------------------
// 16. CSharpClassAndMethod
// ---------------------------------------------------------------------------

TEST_F(TreeSitterChunkerTest, CSharpClassAndMethod) {
    const char* source =
        "using System;\n"
        "\n"
        "namespace App {\n"
        "    public class Calculator {\n"
        "        public int Add(int a, int b) {\n"
        "            return a + b;\n"
        "        }\n"
        "\n"
        "        public int Subtract(int a, int b) {\n"
        "            return a - b;\n"
        "        }\n"
        "    }\n"
        "}\n";

    auto chunks = chunker.chunk_string(source, "Calculator.cs");

    ASSERT_FALSE(chunks.empty());

    // Should find at least one symbol from the C# class.
    std::vector<std::string> symbols;
    for (const auto& c : chunks) {
        if (!c.symbol_name.empty()) {
            symbols.push_back(c.symbol_name);
        }
    }
    EXPECT_GE(symbols.size(), 1u)
        << "Expected at least one named symbol from a C# class";

    for (const auto& c : chunks) {
        EXPECT_EQ(c.language, "csharp");
    }
}

// ---------------------------------------------------------------------------
// 17. SymbolNameExtraction
// ---------------------------------------------------------------------------

TEST_F(TreeSitterChunkerTest, SymbolNameExtraction) {
    const char* source =
        "void initialize() {\n"
        "    // setup\n"
        "}\n"
        "\n"
        "int process(int data) {\n"
        "    return data * 2;\n"
        "}\n"
        "\n"
        "void cleanup() {\n"
        "    // teardown\n"
        "}\n";

    auto chunks = chunker.chunk_string(source, "pipeline.cpp");

    EXPECT_TRUE(has_symbol(chunks, "initialize"))
        << "Expected to find symbol 'initialize'";
    EXPECT_TRUE(has_symbol(chunks, "process"))
        << "Expected to find symbol 'process'";
    EXPECT_TRUE(has_symbol(chunks, "cleanup"))
        << "Expected to find symbol 'cleanup'";
}

// ---------------------------------------------------------------------------
// 18. LineNumbers
// ---------------------------------------------------------------------------

TEST_F(TreeSitterChunkerTest, LineNumbers) {
    // Lines are 1-based. The source has a function spanning lines 1-3
    // and another spanning lines 5-7.
    const char* source =
        "int foo() {\n"       // line 1
        "    return 1;\n"     // line 2
        "}\n"                 // line 3
        "\n"                  // line 4
        "int bar() {\n"       // line 5
        "    return 2;\n"     // line 6
        "}\n";                // line 7

    auto chunks = chunker.chunk_string(source, "lines.cpp");

    ASSERT_FALSE(chunks.empty());

    for (const auto& c : chunks) {
        EXPECT_GT(c.start_line, 0u)
            << "start_line should be 1-based (> 0)";
        EXPECT_GE(c.end_line, c.start_line)
            << "end_line should be >= start_line";
    }

    // Find the chunk containing "foo" and verify its line range.
    for (const auto& c : chunks) {
        if (c.symbol_name == "foo") {
            EXPECT_EQ(c.start_line, 1u);
            EXPECT_LE(c.end_line, 4u)
                << "foo chunk should end no later than line 4";
        }
        if (c.symbol_name == "bar") {
            EXPECT_GE(c.start_line, 4u)
                << "bar chunk should start no earlier than line 4";
            EXPECT_LE(c.end_line, 7u);
        }
    }
}

// ---------------------------------------------------------------------------
// 19. ChunkIdDeterministic
// ---------------------------------------------------------------------------

TEST_F(TreeSitterChunkerTest, ChunkIdDeterministic) {
    const char* source =
        "fn hello() {\n"
        "    println!(\"Hello\");\n"
        "}\n";

    auto chunks1 = chunker.chunk_string(source, "hello.rs");
    auto chunks2 = chunker.chunk_string(source, "hello.rs");

    ASSERT_EQ(chunks1.size(), chunks2.size());
    for (size_t i = 0; i < chunks1.size(); ++i) {
        EXPECT_EQ(chunks1[i].chunk_id, chunks2[i].chunk_id)
            << "chunk_id differs on repeated call for chunk " << i;
        EXPECT_EQ(chunks1[i].start_line, chunks2[i].start_line);
        EXPECT_EQ(chunks1[i].end_line, chunks2[i].end_line);
    }
}

// ---------------------------------------------------------------------------
// 20. ChunkIdFormat
// ---------------------------------------------------------------------------

TEST_F(TreeSitterChunkerTest, ChunkIdFormat) {
    const char* source =
        "def check():\n"
        "    pass\n";

    auto chunks = chunker.chunk_string(source, "check.py");

    ASSERT_FALSE(chunks.empty());
    for (const auto& c : chunks) {
        EXPECT_EQ(c.chunk_id.size(), 16u)
            << "chunk_id should be exactly 16 characters";
        for (char ch : c.chunk_id) {
            EXPECT_TRUE((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f'))
                << "Non-hex character in chunk_id: " << ch;
        }
    }
}

// ---------------------------------------------------------------------------
// 21. EmptyFile
// ---------------------------------------------------------------------------

TEST_F(TreeSitterChunkerTest, EmptyFile) {
    auto chunks = chunker.chunk_string("", "empty.cpp");
    EXPECT_TRUE(chunks.empty());
}

// ---------------------------------------------------------------------------
// 22. NonexistentFile
// ---------------------------------------------------------------------------

TEST_F(TreeSitterChunkerTest, NonexistentFile) {
    auto chunks = chunker.chunk_file(fs::path("this/does/not/exist.cpp"));
    EXPECT_TRUE(chunks.empty());
}

// ---------------------------------------------------------------------------
// 23. Polymorphism
// ---------------------------------------------------------------------------

TEST_F(TreeSitterChunkerTest, Polymorphism) {
    const char* source =
        "int add(int a, int b) {\n"
        "    return a + b;\n"
        "}\n";

    TempFile tmp(source, ".cpp");

    // Call through the base-class pointer to verify virtual dispatch works.
    engram::Chunker* ptr = new engram::TreeSitterChunker();
    auto chunks = ptr->chunk_file(tmp.path());
    delete ptr;

    ASSERT_FALSE(chunks.empty());
    EXPECT_TRUE(has_symbol(chunks, "add"))
        << "Expected to find symbol 'add' via base-class pointer";
}

// ---------------------------------------------------------------------------
// 24. FallbackForUnknownLanguage
// ---------------------------------------------------------------------------

TEST_F(TreeSitterChunkerTest, FallbackForUnknownLanguage) {
    const char* source =
        "first block line 1\n"
        "first block line 2\n"
        "\n"
        "second block line 1\n"
        "second block line 2\n"
        "second block line 3\n"
        "\n"
        "third block line 1\n";

    // Use chunk_string with a .txt extension — no tree-sitter grammar exists
    // for plain text, so the chunker should fall back to regex/blank-line
    // splitting and still produce chunks.
    auto chunks = chunker.chunk_string(source, "notes.txt");

    EXPECT_FALSE(chunks.empty())
        << "Unknown language should still produce chunks via fallback";

    // The chunks should cover the content.
    for (const auto& c : chunks) {
        EXPECT_FALSE(c.source_text.empty());
        EXPECT_GT(c.start_line, 0u);
        EXPECT_GE(c.end_line, c.start_line);
    }
}

#endif // ENGRAM_HAS_TREESITTER
