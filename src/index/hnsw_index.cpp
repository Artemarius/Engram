/// @file hnsw_index.cpp
/// @brief Implementation of the HNSW vector index wrapper.

#include "hnsw_index.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <shared_mutex>
#include <stdexcept>

#include <hnswlib/hnswlib.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace engram {

// ---------------------------------------------------------------------------
// File-path helpers
// ---------------------------------------------------------------------------

std::filesystem::path HnswIndex::hnsw_file(const std::filesystem::path& base)
{
    auto p = base;
    p += ".hnsw";
    return p;
}

std::filesystem::path HnswIndex::meta_file(const std::filesystem::path& base)
{
    auto p = base;
    p += ".meta.json";
    return p;
}

// ---------------------------------------------------------------------------
// Construction / destruction / move
// ---------------------------------------------------------------------------

HnswIndex::HnswIndex(size_t dim,
                      size_t max_elements,
                      size_t M,
                      size_t ef_construction,
                      size_t ef_search)
    : dim_(dim)
    , max_elements_(max_elements)
    , M_(M)
    , ef_construction_(ef_construction)
    , ef_search_(ef_search)
{
    // InnerProductSpace: the distance returned is  1 - dot(a,b).
    // For L2-normalised vectors this equals 1 - cos_sim, so a perfect
    // match yields distance 0.
    space_ = std::make_unique<hnswlib::InnerProductSpace>(dim_);
    index_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(
        space_.get(),
        max_elements_,
        M_,
        ef_construction_);
    index_->setEf(ef_search_);

    spdlog::debug("HnswIndex created: dim={} max_elements={} M={} efC={} efS={}",
                  dim_, max_elements_, M_, ef_construction_, ef_search_);
}

HnswIndex::~HnswIndex() = default;

HnswIndex::HnswIndex(HnswIndex&& other) noexcept
    : dim_(other.dim_)
    , max_elements_(other.max_elements_)
    , M_(other.M_)
    , ef_construction_(other.ef_construction_)
    , ef_search_(other.ef_search_)
    , space_(std::move(other.space_))
    , index_(std::move(other.index_))
    , id_to_label_(std::move(other.id_to_label_))
    , label_to_id_(std::move(other.label_to_id_))
    , next_label_(other.next_label_)
{
}

HnswIndex& HnswIndex::operator=(HnswIndex&& other) noexcept
{
    if (this != &other) {
        dim_              = other.dim_;
        max_elements_     = other.max_elements_;
        M_                = other.M_;
        ef_construction_  = other.ef_construction_;
        ef_search_        = other.ef_search_;
        space_            = std::move(other.space_);
        index_            = std::move(other.index_);
        id_to_label_      = std::move(other.id_to_label_);
        label_to_id_      = std::move(other.label_to_id_);
        next_label_       = other.next_label_;
    }
    return *this;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

bool HnswIndex::normalize(float* vec, size_t dim)
{
    float norm_sq = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        norm_sq += vec[i] * vec[i];
    }
    if (norm_sq < 1e-12f) {
        return false; // zero vector
    }
    const float inv_norm = 1.0f / std::sqrt(norm_sq);
    for (size_t i = 0; i < dim; ++i) {
        vec[i] *= inv_norm;
    }
    return true;
}

void HnswIndex::ensure_capacity()
{
    // hnswlib exposes cur_element_count and max_elements_ as public members.
    if (index_->cur_element_count >= index_->max_elements_) {
        const size_t new_cap = max_elements_ * 2;
        spdlog::info("HnswIndex: resizing {} -> {}", max_elements_, new_cap);
        index_->resizeIndex(new_cap);
        max_elements_ = new_cap;
    }
}

// ---------------------------------------------------------------------------
// VectorIndex interface
// ---------------------------------------------------------------------------

bool HnswIndex::add(const std::string& chunk_id,
                     const float* embedding,
                     size_t dim)
{
    if (dim != dim_) {
        spdlog::error("HnswIndex::add: dimension mismatch (expected {}, got {})",
                      dim_, dim);
        return false;
    }
    if (!embedding) {
        spdlog::error("HnswIndex::add: null embedding pointer");
        return false;
    }

    std::unique_lock<std::shared_mutex> lock(mutex_);

    // Reject duplicates.
    if (id_to_label_.count(chunk_id)) {
        spdlog::warn("HnswIndex::add: chunk_id '{}' already exists", chunk_id);
        return false;
    }

    // Copy and normalize.
    std::vector<float> normed(embedding, embedding + dim);
    if (!normalize(normed.data(), dim)) {
        spdlog::error("HnswIndex::add: zero-magnitude vector for '{}'", chunk_id);
        return false;
    }

    ensure_capacity();

    const size_t label = next_label_++;

    try {
        index_->addPoint(normed.data(), label);
    } catch (const std::exception& e) {
        spdlog::error("HnswIndex::add: hnswlib error: {}", e.what());
        return false;
    }

    id_to_label_[chunk_id] = label;
    label_to_id_[label]    = chunk_id;

    spdlog::debug("HnswIndex::add: '{}' -> label {} (size={})",
                  chunk_id, label, id_to_label_.size());
    return true;
}

bool HnswIndex::remove(const std::string& chunk_id)
{
    std::unique_lock<std::shared_mutex> lock(mutex_);

    auto it = id_to_label_.find(chunk_id);
    if (it == id_to_label_.end()) {
        spdlog::debug("HnswIndex::remove: '{}' not found", chunk_id);
        return false;
    }

    const size_t label = it->second;

    try {
        index_->markDelete(label);
    } catch (const std::exception& e) {
        spdlog::error("HnswIndex::remove: hnswlib error: {}", e.what());
        return false;
    }

    label_to_id_.erase(label);
    id_to_label_.erase(it);

    spdlog::debug("HnswIndex::remove: '{}' (label {}) removed (size={})",
                  chunk_id, label, id_to_label_.size());
    return true;
}

std::vector<SearchResult> HnswIndex::search(const float* query,
                                             size_t dim,
                                             size_t k) const
{
    if (dim != dim_ || !query) {
        return {};
    }

    // Normalize query into a stack-local buffer when dim is small enough;
    // otherwise fall back to a heap allocation (still outside the hot inner
    // loop of hnswlib).
    constexpr size_t kStackLimit = 1024;
    float stack_buf[kStackLimit];
    std::vector<float> heap_buf;

    float* normed = nullptr;
    if (dim_ <= kStackLimit) {
        std::memcpy(stack_buf, query, dim_ * sizeof(float));
        normed = stack_buf;
    } else {
        heap_buf.assign(query, query + dim_);
        normed = heap_buf.data();
    }

    {
        float norm_sq = 0.0f;
        for (size_t i = 0; i < dim_; ++i) {
            norm_sq += normed[i] * normed[i];
        }
        if (norm_sq < 1e-12f) {
            return {};
        }
        const float inv_norm = 1.0f / std::sqrt(norm_sq);
        for (size_t i = 0; i < dim_; ++i) {
            normed[i] *= inv_norm;
        }
    }

    // Take a shared (reader) lock for the remainder — allows concurrent
    // searches while blocking mutating operations.
    std::shared_lock<std::shared_mutex> lock(mutex_);

    // Clamp k to the number of active elements.
    const size_t active = id_to_label_.size();
    if (active == 0) {
        return {};
    }
    const size_t effective_k = std::min(k, active);

    // hnswlib::searchKnn returns a max-heap of (distance, label).
    // For InnerProductSpace the distance is  1 - dot(a,b).
    std::priority_queue<std::pair<float, hnswlib::labeltype>> raw;
    try {
        raw = index_->searchKnn(normed, effective_k);
    } catch (const std::exception& e) {
        spdlog::error("HnswIndex::search: hnswlib error: {}", e.what());
        return {};
    }

    // Convert to SearchResult.  Score = 1 - distance = dot(a,b) = cosine_sim.
    std::vector<SearchResult> results;
    results.reserve(raw.size());

    while (!raw.empty()) {
        const auto& [dist, label] = raw.top();
        auto it = label_to_id_.find(static_cast<size_t>(label));
        if (it != label_to_id_.end()) {
            results.push_back({it->second, 1.0f - dist});
        }
        raw.pop();
    }

    // Release the lock before sorting (purely local operation).
    lock.unlock();

    // Sort descending by score (best first).
    std::sort(results.begin(), results.end(),
              [](const SearchResult& a, const SearchResult& b) {
                  return a.score > b.score;
              });

    return results;
}

bool HnswIndex::save(const std::filesystem::path& path) const
{
    std::unique_lock<std::shared_mutex> lock(mutex_);

    const auto hnsw_path = hnsw_file(path);
    const auto meta_path = meta_file(path);

    // Ensure parent directory exists.
    {
        std::error_code ec;
        const auto parent = hnsw_path.parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent, ec);
            if (ec) {
                spdlog::error("HnswIndex::save: failed to create directory '{}': {}",
                              parent.string(), ec.message());
                return false;
            }
        }
    }

    // Save the hnswlib binary index.
    try {
        index_->saveIndex(hnsw_path.string());
    } catch (const std::exception& e) {
        spdlog::error("HnswIndex::save: hnswlib error saving '{}': {}",
                      hnsw_path.string(), e.what());
        return false;
    }

    // Save chunk-id mappings and parameters as JSON.
    nlohmann::json meta;
    meta["dim"]              = dim_;
    meta["max_elements"]     = max_elements_;
    meta["M"]                = M_;
    meta["ef_construction"]  = ef_construction_;
    meta["ef_search"]        = ef_search_;
    meta["next_label"]       = next_label_;

    // Serialize the id<->label mapping as an array of [chunk_id, label] pairs.
    nlohmann::json mapping = nlohmann::json::array();
    for (const auto& [id, label] : id_to_label_) {
        mapping.push_back({id, label});
    }
    meta["mapping"] = std::move(mapping);

    {
        std::ofstream ofs(meta_path);
        if (!ofs) {
            spdlog::error("HnswIndex::save: cannot open '{}' for writing",
                          meta_path.string());
            return false;
        }
        ofs << meta.dump(2);
        if (!ofs) {
            spdlog::error("HnswIndex::save: write error on '{}'",
                          meta_path.string());
            return false;
        }
    }

    spdlog::info("HnswIndex::save: {} vectors written to '{}'",
                 id_to_label_.size(), path.string());
    return true;
}

bool HnswIndex::load(const std::filesystem::path& path)
{
    std::unique_lock<std::shared_mutex> lock(mutex_);

    const auto hnsw_path = hnsw_file(path);
    const auto meta_path = meta_file(path);

    // Check that both files exist.
    {
        std::error_code ec;
        if (!std::filesystem::exists(hnsw_path, ec) ||
            !std::filesystem::exists(meta_path, ec)) {
            spdlog::warn("HnswIndex::load: index files not found at '{}'",
                         path.string());
            return false;
        }
    }

    // Load the JSON metadata first so we know the parameters.
    nlohmann::json meta;
    {
        std::ifstream ifs(meta_path);
        if (!ifs) {
            spdlog::error("HnswIndex::load: cannot open '{}'",
                          meta_path.string());
            return false;
        }
        try {
            ifs >> meta;
        } catch (const nlohmann::json::exception& e) {
            spdlog::error("HnswIndex::load: JSON parse error in '{}': {}",
                          meta_path.string(), e.what());
            return false;
        }
    }

    // Validate and extract parameters.
    const size_t loaded_dim = meta.value("dim", size_t(0));
    if (loaded_dim == 0) {
        spdlog::error("HnswIndex::load: invalid dimension in metadata");
        return false;
    }
    if (loaded_dim != dim_) {
        spdlog::error("HnswIndex::load: dimension mismatch (index={}, file={})",
                      dim_, loaded_dim);
        return false;
    }

    const size_t loaded_max_elements    = meta.value("max_elements", max_elements_);
    const size_t loaded_M               = meta.value("M", M_);
    const size_t loaded_ef_construction = meta.value("ef_construction", ef_construction_);
    const size_t loaded_ef_search       = meta.value("ef_search", ef_search_);
    const size_t loaded_next_label      = meta.value("next_label", size_t(0));

    // Rebuild the space (dimension may have changed between constructor and load).
    auto new_space = std::make_unique<hnswlib::InnerProductSpace>(dim_);

    // Load hnswlib index from disk.
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> new_index;
    try {
        new_index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
            new_space.get(), hnsw_path.string(),
            false,  // nmslib format = false
            loaded_max_elements);
        new_index->setEf(loaded_ef_search);
    } catch (const std::exception& e) {
        spdlog::error("HnswIndex::load: hnswlib error loading '{}': {}",
                      hnsw_path.string(), e.what());
        return false;
    }

    // Rebuild the bidirectional mapping.
    std::unordered_map<std::string, size_t> new_id_to_label;
    std::unordered_map<size_t, std::string> new_label_to_id;

    if (meta.contains("mapping") && meta["mapping"].is_array()) {
        for (const auto& entry : meta["mapping"]) {
            if (!entry.is_array() || entry.size() != 2) continue;
            const std::string id = entry[0].get<std::string>();
            const size_t label   = entry[1].get<size_t>();
            new_id_to_label[id]  = label;
            new_label_to_id[label] = id;
        }
    }

    // Commit state.
    space_           = std::move(new_space);
    index_           = std::move(new_index);
    id_to_label_     = std::move(new_id_to_label);
    label_to_id_     = std::move(new_label_to_id);
    max_elements_    = loaded_max_elements;
    M_               = loaded_M;
    ef_construction_ = loaded_ef_construction;
    ef_search_       = loaded_ef_search;
    next_label_      = loaded_next_label;

    spdlog::info("HnswIndex::load: {} vectors loaded from '{}'",
                 id_to_label_.size(), path.string());
    return true;
}

size_t HnswIndex::size() const
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    // id_to_label_ accurately tracks active (non-deleted) entries.
    return id_to_label_.size();
}

} // namespace engram
