/// @file test_index.cpp
/// @brief Unit tests for the HnswIndex vector index implementation.

#include <gtest/gtest.h>

#include "index/hnsw_index.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

/// Dimensionality used across all tests.
constexpr size_t kDim = 384;

/// Generate a random float vector and L2-normalize it so that
/// inner product == cosine similarity.
std::vector<float> random_unit_vector(size_t dim, std::mt19937& rng)
{
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> v(dim);
    float norm_sq = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        v[i] = dist(rng);
        norm_sq += v[i] * v[i];
    }
    const float inv_norm = 1.0f / std::sqrt(norm_sq);
    for (auto& x : v) x *= inv_norm;
    return v;
}

/// Helper to create a temporary directory path for index persistence tests.
/// The directory is cleaned up by the caller.
class TempDir {
public:
    TempDir()
    {
        namespace fs = std::filesystem;
        path_ = fs::temp_directory_path() / "engram_test_index";
        std::error_code ec;
        fs::create_directories(path_, ec);
    }

    ~TempDir()
    {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }

    /// Returns a base path (not a directory) for save/load.
    std::filesystem::path index_base() const { return path_ / "test_idx"; }

private:
    std::filesystem::path path_;
};

} // anonymous namespace

// ---------------------------------------------------------------------------
// Basic add / search
// ---------------------------------------------------------------------------

TEST(HnswIndex, AddAndSearchSelf)
{
    engram::HnswIndex index(kDim, /*max_elements=*/100);

    std::mt19937 rng(42);
    constexpr size_t kCount = 10;

    std::vector<std::vector<float>> vecs;
    std::vector<std::string> ids;

    for (size_t i = 0; i < kCount; ++i) {
        ids.push_back("chunk_" + std::to_string(i));
        vecs.push_back(random_unit_vector(kDim, rng));
        ASSERT_TRUE(index.add(ids.back(), vecs.back().data(), kDim));
    }

    EXPECT_EQ(index.size(), kCount);

    // Querying each vector for itself should return that vector as the top hit
    // with a score very close to 1.0 (cosine similarity of identical vectors).
    for (size_t i = 0; i < kCount; ++i) {
        auto results = index.search(vecs[i].data(), kDim, 1);
        ASSERT_FALSE(results.empty());
        EXPECT_EQ(results[0].chunk_id, ids[i]);
        EXPECT_GT(results[0].score, 0.99f)
            << "Self-similarity for " << ids[i] << " should be ~1.0";
    }
}

// ---------------------------------------------------------------------------
// Search returns results sorted by score descending
// ---------------------------------------------------------------------------

TEST(HnswIndex, SearchResultOrder)
{
    engram::HnswIndex index(kDim, /*max_elements=*/200);

    std::mt19937 rng(123);
    constexpr size_t kCount = 50;

    for (size_t i = 0; i < kCount; ++i) {
        auto v = random_unit_vector(kDim, rng);
        ASSERT_TRUE(index.add("vec_" + std::to_string(i), v.data(), kDim));
    }

    auto query = random_unit_vector(kDim, rng);
    auto results = index.search(query.data(), kDim, 10);

    ASSERT_GE(results.size(), 2u);

    for (size_t i = 1; i < results.size(); ++i) {
        EXPECT_GE(results[i - 1].score, results[i].score)
            << "Results must be sorted descending by score";
    }
}

// ---------------------------------------------------------------------------
// Duplicate add is rejected
// ---------------------------------------------------------------------------

TEST(HnswIndex, RejectDuplicateAdd)
{
    engram::HnswIndex index(kDim, /*max_elements=*/10);

    std::mt19937 rng(7);
    auto v = random_unit_vector(kDim, rng);

    ASSERT_TRUE(index.add("dup_test", v.data(), kDim));
    EXPECT_FALSE(index.add("dup_test", v.data(), kDim));
    EXPECT_EQ(index.size(), 1u);
}

// ---------------------------------------------------------------------------
// Dimension mismatch
// ---------------------------------------------------------------------------

TEST(HnswIndex, DimensionMismatch)
{
    engram::HnswIndex index(kDim, /*max_elements=*/10);

    std::vector<float> wrong(kDim + 1, 1.0f);
    EXPECT_FALSE(index.add("bad", wrong.data(), kDim + 1));

    // Search with wrong dimension returns empty.
    auto results = index.search(wrong.data(), kDim + 1, 5);
    EXPECT_TRUE(results.empty());
}

// ---------------------------------------------------------------------------
// Remove
// ---------------------------------------------------------------------------

TEST(HnswIndex, RemoveVector)
{
    engram::HnswIndex index(kDim, /*max_elements=*/100);

    std::mt19937 rng(99);
    constexpr size_t kCount = 5;

    std::vector<std::vector<float>> vecs;
    std::vector<std::string> ids;

    for (size_t i = 0; i < kCount; ++i) {
        ids.push_back("rm_" + std::to_string(i));
        vecs.push_back(random_unit_vector(kDim, rng));
        ASSERT_TRUE(index.add(ids.back(), vecs.back().data(), kDim));
    }

    ASSERT_EQ(index.size(), kCount);

    // Remove the middle element.
    const std::string removed_id = ids[2];
    ASSERT_TRUE(index.remove(removed_id));
    EXPECT_EQ(index.size(), kCount - 1);

    // Removing again should fail.
    EXPECT_FALSE(index.remove(removed_id));

    // Searching with the removed vector's embedding should NOT return it.
    auto results = index.search(vecs[2].data(), kDim, kCount);
    for (const auto& r : results) {
        EXPECT_NE(r.chunk_id, removed_id)
            << "Removed chunk_id '" << removed_id << "' must not appear in results";
    }
}

// ---------------------------------------------------------------------------
// Size tracking
// ---------------------------------------------------------------------------

TEST(HnswIndex, SizeTracking)
{
    engram::HnswIndex index(kDim, /*max_elements=*/50);

    EXPECT_EQ(index.size(), 0u);

    std::mt19937 rng(0);
    for (size_t i = 0; i < 5; ++i) {
        auto v = random_unit_vector(kDim, rng);
        ASSERT_TRUE(index.add("s_" + std::to_string(i), v.data(), kDim));
    }
    EXPECT_EQ(index.size(), 5u);

    ASSERT_TRUE(index.remove("s_2"));
    EXPECT_EQ(index.size(), 4u);

    ASSERT_TRUE(index.remove("s_0"));
    EXPECT_EQ(index.size(), 3u);
}

// ---------------------------------------------------------------------------
// Persistence: save then load into a fresh index
// ---------------------------------------------------------------------------

TEST(HnswIndex, SaveAndLoad)
{
    TempDir tmp;

    std::mt19937 rng(2024);
    constexpr size_t kCount = 20;

    std::vector<std::vector<float>> vecs;
    std::vector<std::string> ids;

    // Build and save.
    {
        engram::HnswIndex index(kDim, /*max_elements=*/100);

        for (size_t i = 0; i < kCount; ++i) {
            ids.push_back("persist_" + std::to_string(i));
            vecs.push_back(random_unit_vector(kDim, rng));
            ASSERT_TRUE(index.add(ids.back(), vecs.back().data(), kDim));
        }

        ASSERT_TRUE(index.save(tmp.index_base()));
    }

    // Load into a brand-new index and verify.
    {
        engram::HnswIndex index(kDim, /*max_elements=*/100);
        ASSERT_TRUE(index.load(tmp.index_base()));
        EXPECT_EQ(index.size(), kCount);

        // Every vector should still find itself as the top result.
        for (size_t i = 0; i < kCount; ++i) {
            auto results = index.search(vecs[i].data(), kDim, 1);
            ASSERT_FALSE(results.empty())
                << "Expected at least one result for " << ids[i];
            EXPECT_EQ(results[0].chunk_id, ids[i]);
            EXPECT_GT(results[0].score, 0.99f);
        }
    }
}

// ---------------------------------------------------------------------------
// Persistence: load from non-existent path returns false
// ---------------------------------------------------------------------------

TEST(HnswIndex, LoadNonExistent)
{
    engram::HnswIndex index(kDim, /*max_elements=*/10);
    EXPECT_FALSE(index.load(std::filesystem::temp_directory_path() / "no_such_index_12345"));
}

// ---------------------------------------------------------------------------
// Search on an empty index returns empty results
// ---------------------------------------------------------------------------

TEST(HnswIndex, SearchEmptyIndex)
{
    engram::HnswIndex index(kDim, /*max_elements=*/10);

    std::mt19937 rng(0);
    auto q = random_unit_vector(kDim, rng);

    auto results = index.search(q.data(), kDim, 5);
    EXPECT_TRUE(results.empty());
}

// ---------------------------------------------------------------------------
// Auto-resize: add more elements than initial max_elements
// ---------------------------------------------------------------------------

TEST(HnswIndex, AutoResize)
{
    constexpr size_t kInitCap = 4;
    constexpr size_t kInsert  = 20;

    engram::HnswIndex index(kDim, /*max_elements=*/kInitCap);

    std::mt19937 rng(55);
    for (size_t i = 0; i < kInsert; ++i) {
        auto v = random_unit_vector(kDim, rng);
        ASSERT_TRUE(index.add("grow_" + std::to_string(i), v.data(), kDim))
            << "Failed to add element " << i;
    }

    EXPECT_EQ(index.size(), kInsert);
}

// ---------------------------------------------------------------------------
// Search with k larger than index size
// ---------------------------------------------------------------------------

TEST(HnswIndex, SearchKLargerThanSize)
{
    engram::HnswIndex index(kDim, /*max_elements=*/10);

    std::mt19937 rng(77);
    constexpr size_t kCount = 3;

    for (size_t i = 0; i < kCount; ++i) {
        auto v = random_unit_vector(kDim, rng);
        ASSERT_TRUE(index.add("few_" + std::to_string(i), v.data(), kDim));
    }

    auto q = random_unit_vector(kDim, rng);
    auto results = index.search(q.data(), kDim, 100);

    // Should get at most kCount results, not crash.
    EXPECT_LE(results.size(), kCount);
    EXPECT_GE(results.size(), 1u);
}

// ---------------------------------------------------------------------------
// Persistence round-trip preserves remove state
// ---------------------------------------------------------------------------

TEST(HnswIndex, SaveLoadWithRemovals)
{
    TempDir tmp;
    std::mt19937 rng(333);

    std::vector<std::vector<float>> vecs;
    std::vector<std::string> ids;
    constexpr size_t kCount = 10;

    // Build, remove some, save.
    {
        engram::HnswIndex index(kDim, /*max_elements=*/50);

        for (size_t i = 0; i < kCount; ++i) {
            ids.push_back("sr_" + std::to_string(i));
            vecs.push_back(random_unit_vector(kDim, rng));
            ASSERT_TRUE(index.add(ids.back(), vecs.back().data(), kDim));
        }

        // Remove elements 3 and 7.
        ASSERT_TRUE(index.remove("sr_3"));
        ASSERT_TRUE(index.remove("sr_7"));
        ASSERT_EQ(index.size(), kCount - 2);

        ASSERT_TRUE(index.save(tmp.index_base()));
    }

    // Load and verify removed elements are still gone.
    {
        engram::HnswIndex index(kDim, /*max_elements=*/50);
        ASSERT_TRUE(index.load(tmp.index_base()));
        EXPECT_EQ(index.size(), kCount - 2);

        // Search using removed vector's embedding -- it should not appear.
        auto results = index.search(vecs[3].data(), kDim, kCount);
        for (const auto& r : results) {
            EXPECT_NE(r.chunk_id, "sr_3");
            EXPECT_NE(r.chunk_id, "sr_7");
        }
    }
}
