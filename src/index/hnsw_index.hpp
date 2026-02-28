#pragma once

/// @file hnsw_index.hpp
/// @brief HNSW-based vector index implementation wrapping hnswlib.
///
/// Uses hnswlib::InnerProductSpace with L2-normalized vectors so that the
/// inner product equals cosine similarity. Maintains bidirectional mappings
/// between string chunk IDs and hnswlib's integer labels.

#include "vector_index.hpp"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

// Forward-declare hnswlib types so the header doesn't pull in the full
// implementation.  The .cpp includes hnswlib/hnswlib.h.
namespace hnswlib {
template <typename dist_t> class HierarchicalNSW;
template <typename dist_t> class SpaceInterface;
} // namespace hnswlib

namespace engram {

/// @class HnswIndex
/// @brief Concrete VectorIndex backed by hnswlib (HNSW algorithm).
///
/// Thread-safety: concurrent search() calls are safe. All mutating
/// operations (add / remove / save / load) must be externally synchronized
/// or are serialized internally via a mutex.
class HnswIndex final : public VectorIndex {
public:
    /// @brief Construct a new (empty) HNSW index.
    /// @param dim              Dimensionality of each vector.
    /// @param max_elements     Initial capacity. The index auto-resizes when
    ///                         this limit is reached.
    /// @param M                HNSW graph connectivity parameter (default 16).
    /// @param ef_construction  Search depth during insertion (default 200).
    /// @param ef_search        Search depth during query (default 50).
    explicit HnswIndex(size_t dim,
                       size_t max_elements = 10000,
                       size_t M = 16,
                       size_t ef_construction = 200,
                       size_t ef_search = 50);

    ~HnswIndex() override;

    // Non-copyable, movable.
    HnswIndex(const HnswIndex&) = delete;
    HnswIndex& operator=(const HnswIndex&) = delete;
    HnswIndex(HnswIndex&&) noexcept;
    HnswIndex& operator=(HnswIndex&&) noexcept;

    // --- VectorIndex interface ------------------------------------------------

    bool add(const std::string& chunk_id,
             const float* embedding,
             size_t dim) override;

    bool remove(const std::string& chunk_id) override;

    std::vector<SearchResult> search(const float* query,
                                     size_t dim,
                                     size_t k) const override;

    bool save(const std::filesystem::path& path) const override;

    bool load(const std::filesystem::path& path) override;

    size_t size() const override;

    // --- Accessors -----------------------------------------------------------

    /// @brief Return the vector dimensionality the index was created with.
    size_t dimension() const noexcept { return dim_; }

private:
    /// Ensure the underlying hnswlib index can accommodate at least one more
    /// element, resizing if necessary.
    void ensure_capacity();

    /// Normalize a vector in-place to unit length.  Returns false if the
    /// vector has zero magnitude (and therefore cannot be normalized).
    static bool normalize(float* vec, size_t dim);

    /// Path helpers for the two on-disk artifacts.
    static std::filesystem::path hnsw_file(const std::filesystem::path& base);
    static std::filesystem::path meta_file(const std::filesystem::path& base);

    size_t dim_;               ///< Vector dimensionality.
    size_t max_elements_;      ///< Current capacity of the hnswlib index.
    size_t M_;                 ///< HNSW M parameter.
    size_t ef_construction_;   ///< HNSW efConstruction parameter.
    size_t ef_search_;         ///< HNSW efSearch parameter.

    /// hnswlib inner-product space (owned).
    std::unique_ptr<hnswlib::SpaceInterface<float>> space_;

    /// The core hnswlib index (owned).
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> index_;

    // Bidirectional mapping: chunk_id <-> hnswlib label.
    std::unordered_map<std::string, size_t> id_to_label_;
    std::unordered_map<size_t, std::string> label_to_id_;

    /// Monotonically increasing label counter.
    size_t next_label_ = 0;

    /// Guards all operations. Writers take exclusive lock; readers take shared.
    mutable std::shared_mutex mutex_;
};

} // namespace engram
