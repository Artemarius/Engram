#pragma once

/// @file ort_embedder.hpp
/// @brief ONNX Runtime embedder with CUDA EP support for Engram.
///
/// Wraps an ONNX Runtime inference session that runs an exported embedding
/// model (e.g. Nomic Embed Code, all-MiniLM-L6-v2).  The model is expected
/// to accept input_ids and attention_mask (int64 tensors) and produce a
/// single float output tensor containing mean-pooled embeddings.
///
/// On construction, the embedder attempts to use the CUDA Execution Provider
/// for GPU-accelerated inference.  If CUDA EP is unavailable, it falls back
/// to the CPU Execution Provider and logs a warning.

#ifdef ENGRAM_HAS_ONNX

#include "embedder.hpp"
#include "ort_tokenizer.hpp"

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// Forward-declare ONNX Runtime types to keep the header lightweight.
// The .cpp includes the full onnxruntime_cxx_api.h.
namespace Ort {
struct Env;
struct Session;
struct SessionOptions;
struct MemoryInfo;
} // namespace Ort

namespace engram {

/// @brief Device preference for ONNX Runtime execution.
enum class DevicePreference {
    /// Try CUDA first, fall back to CPU if unavailable.
    CUDA,
    /// Force CPU execution only.
    CPU
};

/// @class OrtEmbedder
/// @brief Concrete Embedder backed by ONNX Runtime.
///
/// Thread-safety: concurrent calls to embed() and embed_batch() are safe.
/// The ONNX Runtime session itself is thread-safe for inference once
/// constructed.  A mutex guards construction/destruction of internal state.
///
/// Typical usage:
/// @code
///   engram::OrtEmbedder embedder("models/all-MiniLM-L6-v2.onnx",
///                                "models/tokenizer.json",
///                                engram::DevicePreference::CUDA);
///   auto vec = embedder.embed("int main() { return 0; }");
/// @endcode
class OrtEmbedder final : public Embedder {
public:
    /// @brief Construct an ONNX Runtime embedder.
    ///
    /// Loads the ONNX model and tokenizer, configures the execution provider,
    /// and probes the model to determine the embedding dimension.
    ///
    /// @param model_path       Path to the .onnx model file.
    /// @param tokenizer_path   Path to the HuggingFace tokenizer.json file.
    /// @param device           Device preference: CUDA (with CPU fallback) or
    ///                         CPU only.  Default: CUDA.
    /// @param cuda_device_id   CUDA device ordinal (0-based).  Ignored when
    ///                         device == CPU.  Default: 0.
    /// @param max_seq_length   Maximum token sequence length passed to the
    ///                         tokenizer.  Default: 512.
    explicit OrtEmbedder(const std::string& model_path,
                         const std::string& tokenizer_path,
                         DevicePreference device = DevicePreference::CUDA,
                         int cuda_device_id = 0,
                         size_t max_seq_length = 512);

    ~OrtEmbedder() override;

    // Non-copyable, non-movable (owns ORT session and env).
    OrtEmbedder(const OrtEmbedder&) = delete;
    OrtEmbedder& operator=(const OrtEmbedder&) = delete;
    OrtEmbedder(OrtEmbedder&&) = delete;
    OrtEmbedder& operator=(OrtEmbedder&&) = delete;

    // --- Embedder interface --------------------------------------------------

    /// @brief Embed a single text string.
    ///
    /// Tokenizes the input, runs ONNX Runtime inference, and L2-normalizes
    /// the output.  Returns an empty vector on failure.
    ///
    /// @param text  Input text to embed.
    /// @return L2-normalized embedding vector with dimension() elements.
    std::vector<float> embed(const std::string& text) override;

    /// @brief Embed a batch of texts in one inference call.
    ///
    /// Tokenizes all inputs with padding, runs batched inference, and
    /// L2-normalizes each output vector independently.
    ///
    /// @param texts  Input texts to embed.
    /// @return Vector of L2-normalized embeddings, one per input text.
    std::vector<std::vector<float>> embed_batch(
        const std::vector<std::string>& texts) override;

    /// @brief Returns the embedding dimensionality.
    size_t dimension() const override;

    /// @brief Returns the model name for logging.
    std::string model_name() const override;

    // --- Accessors -----------------------------------------------------------

    /// @brief Returns true if the embedder initialized successfully.
    ///
    /// When false, embed() and embed_batch() will return empty results.
    bool is_valid() const noexcept { return valid_; }

    /// @brief Returns the execution provider that is actively being used.
    ///
    /// E.g. "CUDAExecutionProvider" or "CPUExecutionProvider".
    const std::string& active_provider() const noexcept { return active_provider_; }

private:
    /// Run inference on pre-tokenized batch data.
    ///
    /// @param batch  Tokenizer output with flat input_ids and attention_mask.
    /// @return Raw model output as a flat float vector (batch_size * dim).
    ///         Empty on failure.
    std::vector<float> run_inference(const BatchTokenizerOutput& batch);

    /// L2-normalize a vector in-place.  Returns false if the vector has zero
    /// magnitude (degenerate input).
    static bool l2_normalize(float* data, size_t dim);

    // --- ONNX Runtime state --------------------------------------------------
    // These are stored as unique_ptr to opaque types so the header doesn't
    // need to include the full ORT API headers.

    struct OrtState;
    std::unique_ptr<OrtState> ort_;

    // --- Tokenizer -----------------------------------------------------------
    std::unique_ptr<OrtTokenizer> tokenizer_;

    // --- Metadata ------------------------------------------------------------
    std::string model_path_;
    std::string model_name_;
    std::string active_provider_;
    size_t dim_ = 0;
    bool valid_ = false;

    /// Protects ORT session during inference (ORT session is thread-safe
    /// but we guard state transitions).
    mutable std::mutex mutex_;
};

} // namespace engram

#endif // ENGRAM_HAS_ONNX
