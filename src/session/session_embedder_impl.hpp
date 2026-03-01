#pragma once

/// @file session_embedder_impl.hpp
/// @brief Concrete SessionEmbedder that combines an Embedder and a dedicated
///        HNSW index to provide semantic search over past session summaries.

#include "session_embedder.hpp"

#include "embedder/embedder.hpp"
#include "index/hnsw_index.hpp"
#include "session/session_store.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engram {

/// @class SessionEmbedderImpl
/// @brief Embeds and searches session summaries using a dedicated HNSW index.
///
/// Owns a separate HnswIndex instance (distinct from the code chunk index)
/// sized for a small collection (hundreds of sessions at most).  Session
/// text is composed from the summary, key_files, and key_decisions fields,
/// embedded via the injected Embedder, and stored in the index keyed by
/// session ID.
///
/// When the embedder pointer is null (e.g. ONNX not available), all
/// operations degrade gracefully: index_session() returns false and
/// search_sessions() returns an empty vector.
class SessionEmbedderImpl final : public SessionEmbedder {
public:
    /// @brief Construct a SessionEmbedderImpl.
    ///
    /// @param embedder    Embedding model to use.  May be nullptr, in which
    ///                    case all operations return failure / empty results.
    /// @param index_path  File path prefix for persisting the session HNSW
    ///                    index (e.g. "data/session_index").  The index will
    ///                    write files at "<index_path>.hnsw" and
    ///                    "<index_path>.meta.json".
    /// @param store       Session store for looking up full SessionSummary
    ///                    objects by ID.  May be nullptr if lookup is not
    ///                    needed (search will still return sessions whose
    ///                    data was cached at index time).
    SessionEmbedderImpl(Embedder* embedder,
                        const std::filesystem::path& index_path,
                        SessionStore* store);

    ~SessionEmbedderImpl() override = default;

    // Non-copyable, non-movable (owns the HnswIndex).
    SessionEmbedderImpl(const SessionEmbedderImpl&) = delete;
    SessionEmbedderImpl& operator=(const SessionEmbedderImpl&) = delete;

    // --- SessionEmbedder interface -------------------------------------------

    bool index_session(const SessionSummary& session) override;

    std::vector<SessionSummary> search_sessions(
        const std::string& query, size_t k = 5) override;

    // --- Persistence ---------------------------------------------------------

    /// @brief Persist the session HNSW index to disk.
    /// @return true on success.
    bool save() const;

    /// @brief Restore the session HNSW index from disk.
    /// @return true on success, false if the files are missing or corrupt.
    bool load();

    // --- Accessors -----------------------------------------------------------

    /// @brief Return the number of indexed sessions.
    size_t size() const;

    /// @brief Compose a single text string from a SessionSummary's fields.
    ///
    /// Exposed as a public static method so that tests can verify the
    /// composition logic without needing a full embedder.
    ///
    /// @param session  The session summary to compose text from.
    /// @return A concatenated string suitable for embedding.
    static std::string compose_text(const SessionSummary& session);

private:
    Embedder* embedder_;                    ///< Borrowed; may be nullptr.
    std::filesystem::path index_path_;      ///< File prefix for save/load.
    SessionStore* store_;                   ///< Borrowed; may be nullptr.
    std::unique_ptr<HnswIndex> index_;      ///< Dedicated session HNSW index.
};

} // namespace engram
