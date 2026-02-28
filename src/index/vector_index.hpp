#pragma once

/// @file vector_index.hpp
/// @brief Abstract interface for a vector similarity index.
///
/// All concrete index implementations (HNSW, flat, etc.) derive from
/// VectorIndex. The interface is intentionally minimal: add, remove, search,
/// and persist/restore.

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace engram {

/// A single search hit returned by VectorIndex::search().
struct SearchResult {
    std::string chunk_id;   ///< Identifier of the stored chunk.
    float       score;      ///< Similarity score (higher = more similar).
};

/// @class VectorIndex
/// @brief Pure-virtual interface for nearest-neighbor vector indices.
///
/// Implementations must be thread-safe for concurrent readers OR document
/// their threading guarantees.  The hot path (search) must avoid heap
/// allocation wherever possible.
class VectorIndex {
public:
    virtual ~VectorIndex() = default;

    /// @brief Insert a vector associated with a chunk identifier.
    /// @param chunk_id  Unique string identifying the chunk.
    /// @param embedding Pointer to the raw float vector.
    /// @param dim       Dimensionality of the vector.
    /// @return true on success, false on failure (duplicate id, bad dim, etc.).
    virtual bool add(const std::string& chunk_id,
                     const float* embedding,
                     size_t dim) = 0;

    /// @brief Remove a previously inserted vector by its chunk identifier.
    /// @param chunk_id  The identifier to remove.
    /// @return true if the entry was found and removed, false otherwise.
    virtual bool remove(const std::string& chunk_id) = 0;

    /// @brief Find the k nearest neighbors to a query vector.
    /// @param query Pointer to the raw float query vector.
    /// @param dim   Dimensionality (must match the index dimension).
    /// @param k     Number of neighbors to return.
    /// @return Vector of SearchResult sorted by score descending (best first).
    virtual std::vector<SearchResult> search(const float* query,
                                             size_t dim,
                                             size_t k) const = 0;

    /// @brief Persist the index and all metadata to disk.
    /// @param path  Directory or file prefix for the serialized data.
    /// @return true on success.
    virtual bool save(const std::filesystem::path& path) const = 0;

    /// @brief Restore the index and metadata from a previous save().
    /// @param path  Same path that was passed to save().
    /// @return true on success, false if the files are missing or corrupt.
    virtual bool load(const std::filesystem::path& path) = 0;

    /// @brief Return the number of active (non-deleted) vectors in the index.
    virtual size_t size() const = 0;
};

} // namespace engram
