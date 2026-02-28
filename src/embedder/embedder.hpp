#pragma once

/// @file embedder.hpp
/// @brief Abstract embedding model interface for Engram.
///
/// Defines the base class that all embedding implementations (ONNX Runtime,
/// mock, etc.) must satisfy.  An Embedder converts text strings into
/// fixed-dimension float vectors suitable for similarity search.

#include <string>
#include <vector>

namespace engram {

/// @class Embedder
/// @brief Pure-virtual interface for text embedding models.
///
/// Implementations wrap a specific inference backend (ONNX Runtime with
/// CUDA EP, a mock for testing, etc.) and produce dense float vectors
/// from arbitrary text input.  The hot path (embed / embed_batch) should
/// avoid heap allocation beyond the output vectors themselves.
class Embedder {
public:
    virtual ~Embedder() = default;

    /// @brief Embed a single text string into a dense vector.
    /// @param text  The input text to embed.
    /// @return A vector of floats with size == dimension().
    ///         Returns an empty vector on failure.
    virtual std::vector<float> embed(const std::string& text) = 0;

    /// @brief Embed a batch of texts in one call.
    ///
    /// The default implementation simply calls embed() in a loop.
    /// Concrete implementations should override this to exploit batched
    /// inference on the GPU when available.
    ///
    /// @param texts  The input texts to embed.
    /// @return A vector of embedding vectors, one per input text.
    ///         Individual entries may be empty if that particular input failed.
    virtual std::vector<std::vector<float>> embed_batch(
        const std::vector<std::string>& texts)
    {
        std::vector<std::vector<float>> results;
        results.reserve(texts.size());
        for (const auto& text : texts) {
            results.push_back(embed(text));
        }
        return results;
    }

    /// @brief Returns the dimensionality of the embedding vectors.
    ///
    /// All vectors produced by embed() and embed_batch() will have exactly
    /// this many elements (e.g. 384 for MiniLM, 768 for Nomic Embed).
    virtual size_t dimension() const = 0;

    /// @brief Returns a human-readable model name for logging and diagnostics.
    virtual std::string model_name() const = 0;
};

} // namespace engram
