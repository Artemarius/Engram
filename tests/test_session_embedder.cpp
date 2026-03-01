/// @file test_session_embedder.cpp
/// @brief Unit tests for SessionEmbedderImpl — semantic search over sessions.

#include <gtest/gtest.h>

#include "session/session_embedder_impl.hpp"
#include "session/session_store.hpp"
#include "embedder/embedder.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

/// Dimensionality used across all session embedder tests.
constexpr size_t kDim = 64;

// =========================================================================
// Mock Embedder
// =========================================================================

/// A deterministic mock embedder that produces unique vectors for different
/// inputs.  Uses a simple hash-based approach: hash the input string and
/// use it to seed a deterministic vector.  This lets us verify that
/// different texts produce different embeddings and that searching for a
/// text returns the session that was indexed with that text.
class MockEmbedder : public engram::Embedder {
public:
    explicit MockEmbedder(size_t dim = kDim) : dim_(dim) {}

    std::vector<float> embed(const std::string& text) override {
        if (text.empty()) {
            return {};
        }

        // Track calls for test inspection.
        embed_calls_.push_back(text);

        // Produce a deterministic vector based on the text hash.
        std::hash<std::string> hasher;
        size_t h = hasher(text);

        std::vector<float> vec(dim_);
        for (size_t i = 0; i < dim_; ++i) {
            // Use a simple LCG-style sequence seeded by the hash.
            h = h * 6364136223846793005ULL + 1442695040888963407ULL;
            vec[i] = static_cast<float>(static_cast<int64_t>(h)) / static_cast<float>(INT64_MAX);
        }

        // L2-normalize so inner product == cosine similarity.
        float norm_sq = 0.0f;
        for (float x : vec) norm_sq += x * x;
        if (norm_sq > 1e-12f) {
            float inv_norm = 1.0f / std::sqrt(norm_sq);
            for (float& x : vec) x *= inv_norm;
        }

        return vec;
    }

    size_t dimension() const override { return dim_; }
    std::string model_name() const override { return "mock-embedder"; }

    /// Return the texts that were passed to embed(), for test assertions.
    const std::vector<std::string>& embed_calls() const { return embed_calls_; }

    /// Clear the call history.
    void clear_calls() { embed_calls_.clear(); }

private:
    size_t dim_;
    std::vector<std::string> embed_calls_;
};

/// Helper RAII class for a temporary directory.
class TempDir {
public:
    TempDir(const std::string& suffix = "engram_test_session_embedder") {
        path_ = std::filesystem::temp_directory_path() / suffix;
        std::error_code ec;
        std::filesystem::create_directories(path_, ec);
    }

    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }

    const std::filesystem::path& path() const { return path_; }

    /// Returns a file prefix for the session index.
    std::filesystem::path index_base() const { return path_ / "session_idx"; }

    /// Returns a subdirectory for the session store.
    std::filesystem::path store_dir() const { return path_ / "sessions"; }

private:
    std::filesystem::path path_;
};

/// Helper to create a sample SessionSummary.
engram::SessionSummary make_session(
    const std::string& id,
    const std::string& summary,
    const std::vector<std::string>& key_files = {},
    const std::vector<std::string>& key_decisions = {})
{
    engram::SessionSummary s;
    s.id = id;
    s.timestamp = "2026-03-01T10:00:00";
    s.summary = summary;
    s.key_files = key_files;
    s.key_decisions = key_decisions;
    return s;
}

} // anonymous namespace

// ===========================================================================
// Construction
// ===========================================================================

TEST(SessionEmbedderImpl, ConstructWithEmbedder)
{
    TempDir tmp;
    MockEmbedder embedder;
    engram::SessionStore store(tmp.store_dir());

    engram::SessionEmbedderImpl se(&embedder, tmp.index_base(), &store);
    EXPECT_EQ(se.size(), 0u);
}

TEST(SessionEmbedderImpl, ConstructWithNullEmbedder)
{
    TempDir tmp;
    engram::SessionStore store(tmp.store_dir());

    // Should not crash — creates the index with a default dimension.
    engram::SessionEmbedderImpl se(nullptr, tmp.index_base(), &store);
    EXPECT_EQ(se.size(), 0u);
}

TEST(SessionEmbedderImpl, ConstructWithNullStore)
{
    TempDir tmp;
    MockEmbedder embedder;

    // Should not crash.
    engram::SessionEmbedderImpl se(&embedder, tmp.index_base(), nullptr);
    EXPECT_EQ(se.size(), 0u);
}

TEST(SessionEmbedderImpl, ConstructWithAllNull)
{
    TempDir tmp;

    // Should not crash even with both null.
    engram::SessionEmbedderImpl se(nullptr, tmp.index_base(), nullptr);
    EXPECT_EQ(se.size(), 0u);
}

// ===========================================================================
// index_session
// ===========================================================================

TEST(SessionEmbedderImpl, IndexSessionWithNullEmbedderReturnsFalse)
{
    TempDir tmp;
    engram::SessionStore store(tmp.store_dir());

    engram::SessionEmbedderImpl se(nullptr, tmp.index_base(), &store);

    auto session = make_session("s1", "Implemented vector search");
    EXPECT_FALSE(se.index_session(session));
    EXPECT_EQ(se.size(), 0u);
}

TEST(SessionEmbedderImpl, IndexSessionWithEmptyIdReturnsFalse)
{
    TempDir tmp;
    MockEmbedder embedder;
    engram::SessionStore store(tmp.store_dir());

    engram::SessionEmbedderImpl se(&embedder, tmp.index_base(), &store);

    engram::SessionSummary session;
    // id is empty
    session.summary = "Some summary";

    EXPECT_FALSE(se.index_session(session));
    EXPECT_EQ(se.size(), 0u);
}

TEST(SessionEmbedderImpl, IndexSessionSuccess)
{
    TempDir tmp;
    MockEmbedder embedder;
    engram::SessionStore store(tmp.store_dir());

    engram::SessionEmbedderImpl se(&embedder, tmp.index_base(), &store);

    auto session = make_session("s1", "Implemented vector search",
                                {"src/index/hnsw_index.cpp"},
                                {"Use cosine similarity"});

    EXPECT_TRUE(se.index_session(session));
    EXPECT_EQ(se.size(), 1u);
}

TEST(SessionEmbedderImpl, IndexMultipleSessions)
{
    TempDir tmp;
    MockEmbedder embedder;
    engram::SessionStore store(tmp.store_dir());

    engram::SessionEmbedderImpl se(&embedder, tmp.index_base(), &store);

    auto s1 = make_session("s1", "Implemented vector search");
    auto s2 = make_session("s2", "Added MCP protocol support");
    auto s3 = make_session("s3", "Fixed chunker symbol preservation");

    EXPECT_TRUE(se.index_session(s1));
    EXPECT_TRUE(se.index_session(s2));
    EXPECT_TRUE(se.index_session(s3));
    EXPECT_EQ(se.size(), 3u);
}

TEST(SessionEmbedderImpl, IndexDuplicateIdReturnsFalse)
{
    TempDir tmp;
    MockEmbedder embedder;
    engram::SessionStore store(tmp.store_dir());

    engram::SessionEmbedderImpl se(&embedder, tmp.index_base(), &store);

    auto s1 = make_session("s1", "First summary");
    EXPECT_TRUE(se.index_session(s1));

    // Re-indexing the same session ID should fail (HnswIndex rejects duplicates).
    auto s1_dup = make_session("s1", "Different summary");
    EXPECT_FALSE(se.index_session(s1_dup));
    EXPECT_EQ(se.size(), 1u);
}

// ===========================================================================
// search_sessions
// ===========================================================================

TEST(SessionEmbedderImpl, SearchWithNullEmbedderReturnsEmpty)
{
    TempDir tmp;
    engram::SessionStore store(tmp.store_dir());

    engram::SessionEmbedderImpl se(nullptr, tmp.index_base(), &store);

    auto results = se.search_sessions("anything");
    EXPECT_TRUE(results.empty());
}

TEST(SessionEmbedderImpl, SearchEmptyIndexReturnsEmpty)
{
    TempDir tmp;
    MockEmbedder embedder;
    engram::SessionStore store(tmp.store_dir());

    engram::SessionEmbedderImpl se(&embedder, tmp.index_base(), &store);

    auto results = se.search_sessions("anything");
    EXPECT_TRUE(results.empty());
}

TEST(SessionEmbedderImpl, SearchFindsIndexedSession)
{
    TempDir tmp;
    MockEmbedder embedder;
    engram::SessionStore store(tmp.store_dir());

    engram::SessionEmbedderImpl se(&embedder, tmp.index_base(), &store);

    // Index a session.
    auto session = make_session("s1", "Implemented vector search",
                                {"src/index/hnsw_index.cpp"},
                                {"Use cosine similarity"});

    // Also save to the store so search can look it up.
    store.save(session);
    ASSERT_TRUE(se.index_session(session));

    // Search should find it.
    auto results = se.search_sessions("vector search", 5);
    ASSERT_FALSE(results.empty());
    EXPECT_EQ(results[0].id, "s1");
    EXPECT_EQ(results[0].summary, "Implemented vector search");
}

TEST(SessionEmbedderImpl, SearchReturnsUpToKResults)
{
    TempDir tmp;
    MockEmbedder embedder;
    engram::SessionStore store(tmp.store_dir());

    engram::SessionEmbedderImpl se(&embedder, tmp.index_base(), &store);

    // Index 5 sessions and save them to the store.
    for (int i = 0; i < 5; ++i) {
        auto s = make_session("s" + std::to_string(i),
                              "Session summary number " + std::to_string(i));
        store.save(s);
        ASSERT_TRUE(se.index_session(s));
    }

    // Request k=3 — should return at most 3.
    auto results = se.search_sessions("session summary", 3);
    EXPECT_LE(results.size(), 3u);
    EXPECT_GE(results.size(), 1u);
}

TEST(SessionEmbedderImpl, SearchWithNullStoreReturnsStubs)
{
    TempDir tmp;
    MockEmbedder embedder;

    // No store — search should still return stub SessionSummary objects.
    engram::SessionEmbedderImpl se(&embedder, tmp.index_base(), nullptr);

    auto session = make_session("s1", "Implemented vector search");
    ASSERT_TRUE(se.index_session(session));

    auto results = se.search_sessions("vector search", 5);
    ASSERT_FALSE(results.empty());
    // The stub should have the session ID but an empty summary.
    EXPECT_EQ(results[0].id, "s1");
    EXPECT_TRUE(results[0].summary.empty());
}

TEST(SessionEmbedderImpl, SearchMissingSessionInStoreSkips)
{
    TempDir tmp;
    MockEmbedder embedder;
    engram::SessionStore store(tmp.store_dir());

    engram::SessionEmbedderImpl se(&embedder, tmp.index_base(), &store);

    // Index a session but do NOT save it to the store.
    auto session = make_session("s_orphan", "Orphaned session");
    ASSERT_TRUE(se.index_session(session));

    // Search should find the ID in the index but not in the store.
    // The result should be empty since the store can't resolve it.
    auto results = se.search_sessions("orphaned session", 5);
    EXPECT_TRUE(results.empty());
}

// ===========================================================================
// compose_text
// ===========================================================================

TEST(SessionEmbedderImpl, ComposeTextSummaryOnly)
{
    auto session = make_session("s1", "Implemented vector search");

    auto text = engram::SessionEmbedderImpl::compose_text(session);
    EXPECT_EQ(text, "Implemented vector search");
}

TEST(SessionEmbedderImpl, ComposeTextWithKeyFiles)
{
    auto session = make_session("s1", "Summary here",
                                {"file_a.cpp", "file_b.hpp"});

    auto text = engram::SessionEmbedderImpl::compose_text(session);
    EXPECT_NE(text.find("Summary here"), std::string::npos);
    EXPECT_NE(text.find("Key files:"), std::string::npos);
    EXPECT_NE(text.find("file_a.cpp"), std::string::npos);
    EXPECT_NE(text.find("file_b.hpp"), std::string::npos);
}

TEST(SessionEmbedderImpl, ComposeTextWithKeyDecisions)
{
    auto session = make_session("s1", "Summary here",
                                {},
                                {"Use cosine similarity", "Prefer hnswlib"});

    auto text = engram::SessionEmbedderImpl::compose_text(session);
    EXPECT_NE(text.find("Summary here"), std::string::npos);
    EXPECT_NE(text.find("Key decisions:"), std::string::npos);
    EXPECT_NE(text.find("Use cosine similarity"), std::string::npos);
    EXPECT_NE(text.find("Prefer hnswlib"), std::string::npos);
}

TEST(SessionEmbedderImpl, ComposeTextWithAllFields)
{
    auto session = make_session("s1", "Refactored the index module",
                                {"src/index/hnsw_index.cpp", "src/index/vector_index.hpp"},
                                {"Use M=16 for HNSW", "Persist to disk"});

    auto text = engram::SessionEmbedderImpl::compose_text(session);

    // Verify all sections are present.
    EXPECT_NE(text.find("Refactored the index module"), std::string::npos);
    EXPECT_NE(text.find("Key files:"), std::string::npos);
    EXPECT_NE(text.find("hnsw_index.cpp"), std::string::npos);
    EXPECT_NE(text.find("vector_index.hpp"), std::string::npos);
    EXPECT_NE(text.find("Key decisions:"), std::string::npos);
    EXPECT_NE(text.find("Use M=16 for HNSW"), std::string::npos);
    EXPECT_NE(text.find("Persist to disk"), std::string::npos);
}

TEST(SessionEmbedderImpl, ComposeTextEmptySession)
{
    engram::SessionSummary session;
    session.id = "empty";

    auto text = engram::SessionEmbedderImpl::compose_text(session);
    EXPECT_TRUE(text.empty());
}

TEST(SessionEmbedderImpl, ComposeTextFilesJoinedWithCommas)
{
    auto session = make_session("s1", "Summary",
                                {"a.cpp", "b.cpp", "c.cpp"});

    auto text = engram::SessionEmbedderImpl::compose_text(session);
    // Files should be comma-separated.
    EXPECT_NE(text.find("a.cpp, b.cpp, c.cpp"), std::string::npos);
}

// ===========================================================================
// Persistence (save / load)
// ===========================================================================

TEST(SessionEmbedderImpl, SaveAndLoadRoundTrip)
{
    TempDir tmp;
    MockEmbedder embedder;
    engram::SessionStore store(tmp.store_dir());

    // Index a few sessions and persist to disk.
    {
        engram::SessionEmbedderImpl se(&embedder, tmp.index_base(), &store);

        auto s1 = make_session("s1", "Vector search implementation");
        auto s2 = make_session("s2", "MCP protocol support");
        store.save(s1);
        store.save(s2);
        ASSERT_TRUE(se.index_session(s1));
        ASSERT_TRUE(se.index_session(s2));
        EXPECT_EQ(se.size(), 2u);
        ASSERT_TRUE(se.save());
    }

    // Load into a new instance and verify.
    {
        engram::SessionEmbedderImpl se(&embedder, tmp.index_base(), &store);
        ASSERT_TRUE(se.load());
        EXPECT_EQ(se.size(), 2u);

        // Search should still work.
        auto results = se.search_sessions("vector search", 5);
        ASSERT_FALSE(results.empty());
        // One of the results should be s1.
        bool found_s1 = false;
        for (const auto& r : results) {
            if (r.id == "s1") {
                found_s1 = true;
                break;
            }
        }
        EXPECT_TRUE(found_s1);
    }
}

TEST(SessionEmbedderImpl, LoadNonexistentReturnsFalse)
{
    TempDir tmp;
    MockEmbedder embedder;

    engram::SessionEmbedderImpl se(&embedder, tmp.index_base(), nullptr);

    // No files saved — load should fail gracefully.
    EXPECT_FALSE(se.load());
    EXPECT_EQ(se.size(), 0u);
}

// ===========================================================================
// Embedder interaction
// ===========================================================================

TEST(SessionEmbedderImpl, IndexSessionCallsEmbedWithComposedText)
{
    TempDir tmp;
    MockEmbedder embedder;
    engram::SessionStore store(tmp.store_dir());

    engram::SessionEmbedderImpl se(&embedder, tmp.index_base(), &store);

    auto session = make_session("s1", "Refactored chunker",
                                {"chunker.cpp"},
                                {"Preserve symbol names"});

    embedder.clear_calls();
    ASSERT_TRUE(se.index_session(session));

    // The embedder should have been called exactly once.
    ASSERT_EQ(embedder.embed_calls().size(), 1u);

    // The text passed to embed should be the composed text.
    const auto& text = embedder.embed_calls()[0];
    EXPECT_NE(text.find("Refactored chunker"), std::string::npos);
    EXPECT_NE(text.find("chunker.cpp"), std::string::npos);
    EXPECT_NE(text.find("Preserve symbol names"), std::string::npos);
}
