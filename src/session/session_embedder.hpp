#pragma once

/// @file session_embedder.hpp
/// @brief Interface for embedding and searching session summaries.
///
/// Combines the Embedder and VectorIndex to provide semantic search over
/// past session summaries.  The concrete implementation will be provided
/// once the embedder and index modules are wired up.

#include "session_store.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace engram {

/// @class SessionEmbedder
/// @brief Pure-virtual interface for session-level semantic search.
///
/// Implementations embed session summaries into a dedicated vector index
/// (separate from the code chunk index) and support nearest-neighbor
/// retrieval to find past sessions relevant to a new query.
class SessionEmbedder {
public:
    virtual ~SessionEmbedder() = default;

    /// @brief Embed a session summary and insert it into the session index.
    ///
    /// The session's summary text (and optionally key_files / key_decisions)
    /// is embedded and stored.  The session's ID is used as the vector key.
    ///
    /// @param session  The session summary to index.
    /// @return true on success, false if embedding or indexing failed.
    virtual bool index_session(const SessionSummary& session) = 0;

    /// @brief Search for past sessions relevant to a free-text query.
    ///
    /// Embeds the query and performs nearest-neighbor search in the session
    /// index to find the most relevant past sessions.
    ///
    /// @param query  The search query (e.g. a description of the current task).
    /// @param k      Maximum number of results to return (default: 5).
    /// @return A vector of matching SessionSummary objects, ordered by
    ///         relevance (best match first).  May return fewer than @p k
    ///         results if fewer sessions are indexed.
    virtual std::vector<SessionSummary> search_sessions(
        const std::string& query, size_t k = 5) = 0;
};

} // namespace engram
