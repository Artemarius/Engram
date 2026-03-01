/// @file session_embedder_impl.cpp
/// @brief Implementation of SessionEmbedderImpl — semantic search over session
///        summaries backed by a dedicated HNSW index.

#include "session_embedder_impl.hpp"

#include <spdlog/spdlog.h>

namespace engram {

// ---------------------------------------------------------------------------
// Text composition
// ---------------------------------------------------------------------------

std::string SessionEmbedderImpl::compose_text(const SessionSummary& session)
{
    // Combine summary, key_files, and key_decisions into a single text
    // suitable for embedding.  Each section is separated by a newline for
    // clarity, and list items are joined with ", " for brevity.
    std::string text;

    // Reserve a rough estimate to avoid repeated reallocations.
    text.reserve(session.summary.size() + 256);

    // Summary is always the primary content.
    text += session.summary;

    // Append key files if present.
    if (!session.key_files.empty()) {
        text += "\nKey files: ";
        for (size_t i = 0; i < session.key_files.size(); ++i) {
            if (i > 0) text += ", ";
            text += session.key_files[i];
        }
    }

    // Append key decisions if present.
    if (!session.key_decisions.empty()) {
        text += "\nKey decisions: ";
        for (size_t i = 0; i < session.key_decisions.size(); ++i) {
            if (i > 0) text += ", ";
            text += session.key_decisions[i];
        }
    }

    return text;
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

SessionEmbedderImpl::SessionEmbedderImpl(Embedder* embedder,
                                         const std::filesystem::path& index_path,
                                         SessionStore* store)
    : embedder_(embedder)
    , index_path_(index_path)
    , store_(store)
{
    // Determine the embedding dimension.  If the embedder is available, use
    // its reported dimension; otherwise default to 384 (MiniLM) — the index
    // won't be usable without an embedder anyway, but this avoids crashing
    // on construction.
    const size_t dim = embedder_ ? embedder_->dimension() : 384;

    // Session count is expected to be small (hundreds), so use modest
    // HNSW parameters: small capacity, lower M for less memory overhead.
    index_ = std::make_unique<HnswIndex>(
        dim,
        /*max_elements=*/512,
        /*M=*/12,
        /*ef_construction=*/100,
        /*ef_search=*/30);

    spdlog::debug("SessionEmbedderImpl: created (dim={}, embedder={}, store={})",
                  dim,
                  embedder_ ? "yes" : "null",
                  store_    ? "yes" : "null");
}

// ---------------------------------------------------------------------------
// SessionEmbedder interface
// ---------------------------------------------------------------------------

bool SessionEmbedderImpl::index_session(const SessionSummary& session)
{
    if (!embedder_) {
        spdlog::warn("SessionEmbedderImpl::index_session: embedder is null");
        return false;
    }

    if (session.id.empty()) {
        spdlog::warn("SessionEmbedderImpl::index_session: session has empty id");
        return false;
    }

    // Compose the text to embed.
    const std::string text = compose_text(session);
    if (text.empty()) {
        spdlog::warn("SessionEmbedderImpl::index_session: composed text is empty "
                      "for session '{}'", session.id);
        return false;
    }

    // Embed the composed text.
    auto embedding = embedder_->embed(text);
    if (embedding.empty()) {
        spdlog::error("SessionEmbedderImpl::index_session: embedding failed for "
                      "session '{}'", session.id);
        return false;
    }

    // Add to the HNSW index using the session ID as the chunk_id.
    if (!index_->add(session.id, embedding.data(), embedding.size())) {
        spdlog::error("SessionEmbedderImpl::index_session: failed to add session "
                      "'{}' to index", session.id);
        return false;
    }

    spdlog::info("SessionEmbedderImpl: indexed session '{}' (index size={})",
                 session.id, index_->size());
    return true;
}

std::vector<SessionSummary> SessionEmbedderImpl::search_sessions(
    const std::string& query, size_t k)
{
    if (!embedder_) {
        spdlog::debug("SessionEmbedderImpl::search_sessions: embedder is null");
        return {};
    }

    if (index_->size() == 0) {
        spdlog::debug("SessionEmbedderImpl::search_sessions: index is empty");
        return {};
    }

    // Embed the query.
    auto query_embedding = embedder_->embed(query);
    if (query_embedding.empty()) {
        spdlog::warn("SessionEmbedderImpl::search_sessions: failed to embed query");
        return {};
    }

    // Search the session index.
    auto hits = index_->search(query_embedding.data(),
                               query_embedding.size(),
                               k);

    // Look up full SessionSummary objects from the SessionStore.
    std::vector<SessionSummary> results;
    results.reserve(hits.size());

    for (const auto& hit : hits) {
        // The chunk_id in our session index is the session ID.
        if (store_) {
            auto session_opt = store_->load(hit.chunk_id);
            if (session_opt.has_value()) {
                results.push_back(std::move(*session_opt));
            } else {
                spdlog::debug("SessionEmbedderImpl::search_sessions: session '{}' "
                              "not found in store (score={:.3f})",
                              hit.chunk_id, hit.score);
            }
        } else {
            // No store available — return a minimal SessionSummary with just
            // the ID so the caller at least knows which sessions matched.
            SessionSummary stub;
            stub.id = hit.chunk_id;
            results.push_back(std::move(stub));
        }
    }

    spdlog::debug("SessionEmbedderImpl::search_sessions: query='{}' k={} "
                  "hits={} results={}", query, k, hits.size(), results.size());
    return results;
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

bool SessionEmbedderImpl::save() const
{
    if (!index_) {
        return false;
    }
    return index_->save(index_path_);
}

bool SessionEmbedderImpl::load()
{
    if (!index_) {
        return false;
    }
    return index_->load(index_path_);
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

size_t SessionEmbedderImpl::size() const
{
    return index_ ? index_->size() : 0;
}

} // namespace engram
