/// @file test_embedder.cpp
/// @brief Unit tests for the OrtTokenizer and OrtEmbedder.
///
/// Tests are split into two groups:
///   1. Tokenizer tests — use a synthetic tokenizer.json file to verify
///      tokenization logic without needing a real model.
///   2. Embedder tests (guarded by ENGRAM_HAS_ONNX) — require an actual
///      ONNX model and are skipped in builds without ONNX Runtime.
///
/// The tokenizer tests exercise the core WordPiece logic with a small
/// hand-crafted vocabulary.  This makes the test suite fast, deterministic,
/// and independent of external model downloads.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#ifdef ENGRAM_HAS_ONNX
#include "embedder/ort_tokenizer.hpp"
#include "embedder/ort_embedder.hpp"
#endif

using json = nlohmann::json;

// ===========================================================================
// Tokenizer tests (only compiled when ONNX is enabled)
// ===========================================================================

#ifdef ENGRAM_HAS_ONNX

// ===========================================================================
// Test fixtures and helpers
// ===========================================================================

namespace {

/// Helper RAII class that writes a temporary tokenizer.json file and
/// cleans it up on destruction.
class TempTokenizerFile {
public:
    /// Create a temporary tokenizer.json with the given vocabulary.
    ///
    /// The vocabulary maps token strings to integer IDs.  Special tokens
    /// [PAD], [UNK], [CLS], [SEP] should be included explicitly.
    explicit TempTokenizerFile(const std::unordered_map<std::string, int>& vocab)
    {
        namespace fs = std::filesystem;
        dir_ = fs::temp_directory_path() / "engram_test_tokenizer";
        std::error_code ec;
        fs::create_directories(dir_, ec);
        path_ = dir_ / "tokenizer.json";

        // Build a minimal HuggingFace tokenizer.json.
        json vocab_json = json::object();
        for (const auto& [token, id] : vocab) {
            vocab_json[token] = id;
        }

        json root = {
            {"model", {
                {"type", "WordPiece"},
                {"vocab", vocab_json},
                {"unk_token", "[UNK]"},
                {"continuing_subword_prefix", "##"},
                {"max_input_chars_per_word", 100}
            }},
            {"added_tokens", json::array({
                {{"id", vocab.at("[PAD]")}, {"content", "[PAD]"}, {"special", true}},
                {{"id", vocab.at("[UNK]")}, {"content", "[UNK]"}, {"special", true}},
                {{"id", vocab.at("[CLS]")}, {"content", "[CLS]"}, {"special", true}},
                {{"id", vocab.at("[SEP]")}, {"content", "[SEP]"}, {"special", true}}
            })},
            {"normalizer", {
                {"type", "BertNormalizer"},
                {"lowercase", true}
            }}
        };

        std::ofstream ofs(path_);
        ofs << root.dump(2);
    }

    ~TempTokenizerFile()
    {
        std::error_code ec;
        std::filesystem::remove_all(dir_, ec);
    }

    std::string path() const { return path_.string(); }

private:
    std::filesystem::path dir_;
    std::filesystem::path path_;
};

/// A small test vocabulary with WordPiece sub-words.
///
/// This covers:
///   - Special tokens: [PAD]=0, [UNK]=1, [CLS]=2, [SEP]=3
///   - Full words: "hello"=4, "world"=5, "int"=6, "main"=7, "return"=8
///   - Sub-words: "##lo"=9, "hel"=10, "un"=11, "##known"=12
///   - Punctuation: "("=13, ")"=14, "{"=15, "}"=16, ";"=17
///   - Other tokens: "0"=18, "code"=19, "test"=20
std::unordered_map<std::string, int> make_test_vocab()
{
    return {
        {"[PAD]", 0},
        {"[UNK]", 1},
        {"[CLS]", 2},
        {"[SEP]", 3},
        {"hello", 4},
        {"world", 5},
        {"int",   6},
        {"main",  7},
        {"return", 8},
        {"##lo",  9},
        {"hel",   10},
        {"un",    11},
        {"##known", 12},
        {"(",     13},
        {")",     14},
        {"{",     15},
        {"}",     16},
        {";",     17},
        {"0",     18},
        {"code",  19},
        {"test",  20},
        {"a",     21},
        {"b",     22},
        {"c",     23},
        {"##a",   24},
        {"##b",   25},
        {"##c",   26},
        {"##s",   27},
        {"##t",   28},
    };
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Loading and validity
// ---------------------------------------------------------------------------

TEST(OrtTokenizer, LoadsValidFile)
{
    TempTokenizerFile tmp(make_test_vocab());
    engram::OrtTokenizer tokenizer(tmp.path(), 128);

    EXPECT_TRUE(tokenizer.is_valid());
    EXPECT_EQ(tokenizer.vocab_size(), make_test_vocab().size());
}

TEST(OrtTokenizer, LoadNonExistentFileFails)
{
    engram::OrtTokenizer tokenizer("/no/such/path/tokenizer.json", 128);

    EXPECT_FALSE(tokenizer.is_valid());
    EXPECT_EQ(tokenizer.vocab_size(), 0u);
}

TEST(OrtTokenizer, LoadInvalidJsonFails)
{
    namespace fs = std::filesystem;
    auto dir = fs::temp_directory_path() / "engram_test_bad_json";
    fs::create_directories(dir);
    auto path = dir / "tokenizer.json";

    {
        std::ofstream ofs(path);
        ofs << "{not valid json}";
    }

    engram::OrtTokenizer tokenizer(path.string(), 128);
    EXPECT_FALSE(tokenizer.is_valid());

    std::error_code ec;
    fs::remove_all(dir, ec);
}

// ---------------------------------------------------------------------------
// Basic encoding
// ---------------------------------------------------------------------------

TEST(OrtTokenizer, EncodeSimpleWord)
{
    TempTokenizerFile tmp(make_test_vocab());
    engram::OrtTokenizer tokenizer(tmp.path(), 128);

    auto ids = tokenizer.encode("hello");
    ASSERT_FALSE(ids.empty());

    // Should be [CLS] hello [SEP] -> [2, 4, 3]
    ASSERT_EQ(ids.size(), 3u);
    EXPECT_EQ(ids[0], 2);  // [CLS]
    EXPECT_EQ(ids[1], 4);  // hello
    EXPECT_EQ(ids[2], 3);  // [SEP]
}

TEST(OrtTokenizer, EncodeMultipleWords)
{
    TempTokenizerFile tmp(make_test_vocab());
    engram::OrtTokenizer tokenizer(tmp.path(), 128);

    auto ids = tokenizer.encode("hello world");
    ASSERT_FALSE(ids.empty());

    // [CLS] hello world [SEP]
    ASSERT_EQ(ids.size(), 4u);
    EXPECT_EQ(ids[0], 2);  // [CLS]
    EXPECT_EQ(ids[1], 4);  // hello
    EXPECT_EQ(ids[2], 5);  // world
    EXPECT_EQ(ids[3], 3);  // [SEP]
}

TEST(OrtTokenizer, EncodeWithPunctuation)
{
    TempTokenizerFile tmp(make_test_vocab());
    engram::OrtTokenizer tokenizer(tmp.path(), 128);

    auto ids = tokenizer.encode("main()");
    ASSERT_FALSE(ids.empty());

    // [CLS] main ( ) [SEP]
    ASSERT_EQ(ids.size(), 5u);
    EXPECT_EQ(ids[0], 2);   // [CLS]
    EXPECT_EQ(ids[1], 7);   // main
    EXPECT_EQ(ids[2], 13);  // (
    EXPECT_EQ(ids[3], 14);  // )
    EXPECT_EQ(ids[4], 3);   // [SEP]
}

// ---------------------------------------------------------------------------
// WordPiece sub-word splitting
// ---------------------------------------------------------------------------

TEST(OrtTokenizer, WordPieceSubWordSplit)
{
    TempTokenizerFile tmp(make_test_vocab());
    engram::OrtTokenizer tokenizer(tmp.path(), 128);

    // "unknown" should split to "un" + "##known"
    auto ids = tokenizer.encode("unknown");
    ASSERT_FALSE(ids.empty());

    // [CLS] un ##known [SEP]
    ASSERT_EQ(ids.size(), 4u);
    EXPECT_EQ(ids[0], 2);   // [CLS]
    EXPECT_EQ(ids[1], 11);  // un
    EXPECT_EQ(ids[2], 12);  // ##known
    EXPECT_EQ(ids[3], 3);   // [SEP]
}

TEST(OrtTokenizer, UnknownWordMapsToUnk)
{
    TempTokenizerFile tmp(make_test_vocab());
    engram::OrtTokenizer tokenizer(tmp.path(), 128);

    // "xyz" is not in the vocab and can't be split into sub-words.
    auto ids = tokenizer.encode("xyz");
    ASSERT_FALSE(ids.empty());

    // [CLS] [UNK] [SEP]
    ASSERT_EQ(ids.size(), 3u);
    EXPECT_EQ(ids[0], 2);  // [CLS]
    EXPECT_EQ(ids[1], 1);  // [UNK]
    EXPECT_EQ(ids[2], 3);  // [SEP]
}

// ---------------------------------------------------------------------------
// Case normalization
// ---------------------------------------------------------------------------

TEST(OrtTokenizer, LowercaseNormalization)
{
    TempTokenizerFile tmp(make_test_vocab());
    engram::OrtTokenizer tokenizer(tmp.path(), 128);

    // "HELLO" should be lowercased to "hello" before lookup.
    auto ids = tokenizer.encode("HELLO");
    ASSERT_FALSE(ids.empty());

    // [CLS] hello [SEP]
    ASSERT_EQ(ids.size(), 3u);
    EXPECT_EQ(ids[1], 4);  // hello (lowercased from HELLO)
}

// ---------------------------------------------------------------------------
// Truncation
// ---------------------------------------------------------------------------

TEST(OrtTokenizer, TruncationAtMaxLength)
{
    TempTokenizerFile tmp(make_test_vocab());
    // Very short max_length to force truncation.
    engram::OrtTokenizer tokenizer(tmp.path(), 4);

    // "hello world" would normally produce 4 tokens [CLS, hello, world, SEP].
    // With max_length=4, all should fit.
    auto ids = tokenizer.encode("hello world");
    ASSERT_EQ(ids.size(), 4u);

    // Now try something that would be truncated.
    // "int main return" would be 5 tokens.
    // With max_length=4: [CLS] int main [SEP] (truncated 'return').
    auto ids2 = tokenizer.encode("int main return");
    ASSERT_EQ(ids2.size(), 4u);
    EXPECT_EQ(ids2[0], 2);  // [CLS]
    EXPECT_EQ(ids2[1], 6);  // int
    EXPECT_EQ(ids2[2], 7);  // main
    EXPECT_EQ(ids2[3], 3);  // [SEP] (return was truncated)
}

// ---------------------------------------------------------------------------
// Attention mask
// ---------------------------------------------------------------------------

TEST(OrtTokenizer, AttentionMaskAllOnes)
{
    TempTokenizerFile tmp(make_test_vocab());
    engram::OrtTokenizer tokenizer(tmp.path(), 128);

    auto output = tokenizer.encode_with_mask("hello world");
    ASSERT_FALSE(output.input_ids.empty());

    // No padding in encode_with_mask, so all mask values should be 1.
    EXPECT_EQ(output.input_ids.size(), output.attention_mask.size());
    for (auto m : output.attention_mask) {
        EXPECT_EQ(m, 1);
    }
}

// ---------------------------------------------------------------------------
// Batch encoding
// ---------------------------------------------------------------------------

TEST(OrtTokenizer, BatchEncodingPadding)
{
    TempTokenizerFile tmp(make_test_vocab());
    engram::OrtTokenizer tokenizer(tmp.path(), 128);

    std::vector<std::string> texts = {"hello", "hello world"};
    auto batch = tokenizer.encode_batch(texts);

    EXPECT_EQ(batch.batch_size, 2u);
    // "hello" -> [CLS, hello, SEP] = 3 tokens
    // "hello world" -> [CLS, hello, world, SEP] = 4 tokens
    // Max length in batch = 4
    EXPECT_EQ(batch.seq_length, 4u);

    // Total elements = 2 * 4 = 8
    EXPECT_EQ(batch.input_ids.size(), 8u);
    EXPECT_EQ(batch.attention_mask.size(), 8u);

    // First sequence: [CLS, hello, SEP, PAD]
    EXPECT_EQ(batch.input_ids[0], 2);  // [CLS]
    EXPECT_EQ(batch.input_ids[1], 4);  // hello
    EXPECT_EQ(batch.input_ids[2], 3);  // [SEP]
    EXPECT_EQ(batch.input_ids[3], 0);  // [PAD]

    // Attention mask for first sequence: [1, 1, 1, 0]
    EXPECT_EQ(batch.attention_mask[0], 1);
    EXPECT_EQ(batch.attention_mask[1], 1);
    EXPECT_EQ(batch.attention_mask[2], 1);
    EXPECT_EQ(batch.attention_mask[3], 0);

    // Second sequence: [CLS, hello, world, SEP]
    EXPECT_EQ(batch.input_ids[4], 2);  // [CLS]
    EXPECT_EQ(batch.input_ids[5], 4);  // hello
    EXPECT_EQ(batch.input_ids[6], 5);  // world
    EXPECT_EQ(batch.input_ids[7], 3);  // [SEP]

    // Attention mask for second sequence: all 1s
    EXPECT_EQ(batch.attention_mask[4], 1);
    EXPECT_EQ(batch.attention_mask[5], 1);
    EXPECT_EQ(batch.attention_mask[6], 1);
    EXPECT_EQ(batch.attention_mask[7], 1);
}

TEST(OrtTokenizer, BatchEncodingEmpty)
{
    TempTokenizerFile tmp(make_test_vocab());
    engram::OrtTokenizer tokenizer(tmp.path(), 128);

    std::vector<std::string> texts;
    auto batch = tokenizer.encode_batch(texts);

    EXPECT_EQ(batch.batch_size, 0u);
    EXPECT_TRUE(batch.input_ids.empty());
}

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

TEST(OrtTokenizer, DecodeSimple)
{
    TempTokenizerFile tmp(make_test_vocab());
    engram::OrtTokenizer tokenizer(tmp.path(), 128);

    // [CLS] hello world [SEP]
    std::vector<int64_t> ids = {2, 4, 5, 3};
    auto text = tokenizer.decode(ids);

    // Should strip [CLS] and [SEP], join with spaces.
    EXPECT_EQ(text, "hello world");
}

TEST(OrtTokenizer, DecodeWithSubWords)
{
    TempTokenizerFile tmp(make_test_vocab());
    engram::OrtTokenizer tokenizer(tmp.path(), 128);

    // [CLS] un ##known [SEP]
    std::vector<int64_t> ids = {2, 11, 12, 3};
    auto text = tokenizer.decode(ids);

    // "un" + "known" (## stripped, no space) = "unknown"
    EXPECT_EQ(text, "unknown");
}

TEST(OrtTokenizer, DecodeStripsSpecialTokens)
{
    TempTokenizerFile tmp(make_test_vocab());
    engram::OrtTokenizer tokenizer(tmp.path(), 128);

    // [CLS] hello [PAD] [PAD] [SEP]
    std::vector<int64_t> ids = {2, 4, 0, 0, 3};
    auto text = tokenizer.decode(ids);

    EXPECT_EQ(text, "hello");
}

// ---------------------------------------------------------------------------
// Empty / edge cases
// ---------------------------------------------------------------------------

TEST(OrtTokenizer, EncodeEmptyString)
{
    TempTokenizerFile tmp(make_test_vocab());
    engram::OrtTokenizer tokenizer(tmp.path(), 128);

    auto ids = tokenizer.encode("");
    // Even empty text should produce [CLS, SEP].
    ASSERT_EQ(ids.size(), 2u);
    EXPECT_EQ(ids[0], 2);  // [CLS]
    EXPECT_EQ(ids[1], 3);  // [SEP]
}

TEST(OrtTokenizer, EncodeWhitespaceOnly)
{
    TempTokenizerFile tmp(make_test_vocab());
    engram::OrtTokenizer tokenizer(tmp.path(), 128);

    auto ids = tokenizer.encode("   \t\n  ");
    // Whitespace-only should produce [CLS, SEP].
    ASSERT_EQ(ids.size(), 2u);
    EXPECT_EQ(ids[0], 2);  // [CLS]
    EXPECT_EQ(ids[1], 3);  // [SEP]
}

TEST(OrtTokenizer, InvalidTokenizerReturnsEmpty)
{
    engram::OrtTokenizer tokenizer("/nonexistent", 128);
    EXPECT_FALSE(tokenizer.is_valid());

    auto ids = tokenizer.encode("hello");
    EXPECT_TRUE(ids.empty());

    auto text = tokenizer.decode({2, 4, 3});
    EXPECT_TRUE(text.empty());
}

// ---------------------------------------------------------------------------
// MaxLength accessor
// ---------------------------------------------------------------------------

TEST(OrtTokenizer, MaxLengthAccessor)
{
    TempTokenizerFile tmp(make_test_vocab());
    engram::OrtTokenizer tokenizer(tmp.path(), 256);

    EXPECT_EQ(tokenizer.max_length(), 256u);
}

// ===========================================================================
// OrtEmbedder tests
//
// These tests require an actual ONNX model to be present.  They are designed
// to be skipped gracefully (via GTEST_SKIP) when the model file does not
// exist.  This lets CI builds without GPU / ONNX Runtime still pass.
// ===========================================================================

class OrtEmbedderTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        // Look for a test model in several common locations.
        const std::vector<std::string> search_paths = {
            "models/all-MiniLM-L6-v2.onnx",
            "../models/all-MiniLM-L6-v2.onnx",
            "../../models/all-MiniLM-L6-v2.onnx",
            "models/nomic-embed-code.onnx",
            "../models/nomic-embed-code.onnx",
            "../../models/nomic-embed-code.onnx",
        };

        for (const auto& p : search_paths) {
            if (std::filesystem::exists(p)) {
                model_path_ = p;
                break;
            }
        }

        // Look for tokenizer.json relative to the model.
        if (!model_path_.empty()) {
            auto model_dir = std::filesystem::path(model_path_).parent_path();
            auto tok_path = model_dir / "tokenizer.json";
            if (std::filesystem::exists(tok_path)) {
                tokenizer_path_ = tok_path.string();
            }
        }
    }

    std::string model_path_;
    std::string tokenizer_path_;
};

TEST_F(OrtEmbedderTest, EmbedSingleText)
{
    if (model_path_.empty() || tokenizer_path_.empty()) {
        GTEST_SKIP() << "ONNX model or tokenizer not found; skipping embedder test";
    }

    engram::OrtEmbedder embedder(model_path_, tokenizer_path_,
                                  engram::DevicePreference::CPU);

    if (!embedder.is_valid()) {
        GTEST_SKIP() << "Embedder failed to initialize; skipping";
    }

    auto vec = embedder.embed("int main() { return 0; }");
    ASSERT_FALSE(vec.empty());
    EXPECT_EQ(vec.size(), embedder.dimension());

    // Check that the output is L2-normalized (norm ~ 1.0).
    float norm_sq = 0.0f;
    for (float v : vec) {
        norm_sq += v * v;
    }
    EXPECT_NEAR(std::sqrt(norm_sq), 1.0f, 0.01f);
}

TEST_F(OrtEmbedderTest, EmbedBatch)
{
    if (model_path_.empty() || tokenizer_path_.empty()) {
        GTEST_SKIP() << "ONNX model or tokenizer not found; skipping embedder test";
    }

    engram::OrtEmbedder embedder(model_path_, tokenizer_path_,
                                  engram::DevicePreference::CPU);

    if (!embedder.is_valid()) {
        GTEST_SKIP() << "Embedder failed to initialize; skipping";
    }

    std::vector<std::string> texts = {
        "int main() { return 0; }",
        "def hello_world(): print('hello')",
        "class Foo { public: void bar(); };"
    };

    auto results = embedder.embed_batch(texts);
    ASSERT_EQ(results.size(), texts.size());

    for (size_t i = 0; i < results.size(); ++i) {
        EXPECT_EQ(results[i].size(), embedder.dimension())
            << "Batch item " << i << " has wrong dimension";

        // Each should be L2-normalized.
        float norm_sq = 0.0f;
        for (float v : results[i]) {
            norm_sq += v * v;
        }
        EXPECT_NEAR(std::sqrt(norm_sq), 1.0f, 0.01f)
            << "Batch item " << i << " is not L2-normalized";
    }
}

TEST_F(OrtEmbedderTest, SimilarTextsHaveHighSimilarity)
{
    if (model_path_.empty() || tokenizer_path_.empty()) {
        GTEST_SKIP() << "ONNX model or tokenizer not found; skipping embedder test";
    }

    engram::OrtEmbedder embedder(model_path_, tokenizer_path_,
                                  engram::DevicePreference::CPU);

    if (!embedder.is_valid()) {
        GTEST_SKIP() << "Embedder failed to initialize; skipping";
    }

    auto vec_a = embedder.embed("int main() { return 0; }");
    auto vec_b = embedder.embed("int main() { return 1; }");
    auto vec_c = embedder.embed("def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)");

    ASSERT_FALSE(vec_a.empty());
    ASSERT_FALSE(vec_b.empty());
    ASSERT_FALSE(vec_c.empty());

    // Cosine similarity (vectors are L2-normalized, so dot product = cosine sim).
    auto cosine = [](const std::vector<float>& a, const std::vector<float>& b) -> float {
        float dot = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            dot += a[i] * b[i];
        }
        return dot;
    };

    float sim_ab = cosine(vec_a, vec_b);
    float sim_ac = cosine(vec_a, vec_c);

    // Nearly identical C code should be more similar than C vs Python.
    EXPECT_GT(sim_ab, sim_ac)
        << "Similar C code (" << sim_ab
        << ") should have higher similarity than C vs Python (" << sim_ac << ")";
}

TEST_F(OrtEmbedderTest, DimensionAndModelName)
{
    if (model_path_.empty() || tokenizer_path_.empty()) {
        GTEST_SKIP() << "ONNX model or tokenizer not found; skipping embedder test";
    }

    engram::OrtEmbedder embedder(model_path_, tokenizer_path_,
                                  engram::DevicePreference::CPU);

    if (!embedder.is_valid()) {
        GTEST_SKIP() << "Embedder failed to initialize; skipping";
    }

    // Dimension should be 384 (MiniLM) or 768 (Nomic).
    size_t dim = embedder.dimension();
    EXPECT_TRUE(dim == 384 || dim == 768)
        << "Unexpected dimension: " << dim;

    // Model name should be non-empty.
    EXPECT_FALSE(embedder.model_name().empty());
}

TEST_F(OrtEmbedderTest, InvalidModelPathFails)
{
    engram::OrtEmbedder embedder("/nonexistent/model.onnx",
                                  "/nonexistent/tokenizer.json",
                                  engram::DevicePreference::CPU);

    EXPECT_FALSE(embedder.is_valid());
    EXPECT_EQ(embedder.dimension(), 0u);

    auto vec = embedder.embed("hello");
    EXPECT_TRUE(vec.empty());
}

#endif // ENGRAM_HAS_ONNX

// ===========================================================================
// Non-ONNX placeholder test
//
// When ENGRAM_HAS_ONNX is not defined, we still need at least one test
// case so that the test binary links and the test runner doesn't complain
// about an empty test suite.
// ===========================================================================

#ifndef ENGRAM_HAS_ONNX

TEST(EmbedderPlaceholder, OnnxNotEnabled)
{
    // This test simply confirms that the embedder module compiles and
    // links even without ONNX Runtime.  All ORT-specific code is
    // guarded by #ifdef ENGRAM_HAS_ONNX.
    SUCCEED() << "ENGRAM_HAS_ONNX is not defined; embedder tests skipped";
}

#endif // !ENGRAM_HAS_ONNX
