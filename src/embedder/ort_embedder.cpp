/// @file ort_embedder.cpp
/// @brief ONNX Runtime embedder implementation with CUDA EP support.

#ifdef ENGRAM_HAS_ONNX

#include "ort_embedder.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <numeric>

#include <onnxruntime_cxx_api.h>
#include <spdlog/spdlog.h>

namespace engram {

// ---------------------------------------------------------------------------
// Internal ORT state — hidden from the header via the pimpl idiom so that
// downstream code does not need to include onnxruntime headers.
// ---------------------------------------------------------------------------

struct OrtEmbedder::OrtState {
    Ort::Env env;
    Ort::Session session{nullptr};
    Ort::MemoryInfo memory_info{nullptr};

    OrtState()
        : env(ORT_LOGGING_LEVEL_WARNING, "engram")
        , memory_info(Ort::MemoryInfo::CreateCpu(
              OrtArenaAllocator, OrtMemTypeDefault))
    {}
};

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

OrtEmbedder::OrtEmbedder(const std::string& model_path,
                         const std::string& tokenizer_path,
                         DevicePreference device,
                         int cuda_device_id,
                         size_t max_seq_length)
    : model_path_(model_path)
{
    spdlog::info("OrtEmbedder: initializing");
    spdlog::info("  model:     {}", model_path);
    spdlog::info("  tokenizer: {}", tokenizer_path);
    spdlog::info("  device:    {}",
                 device == DevicePreference::CUDA ? "CUDA" : "CPU");

    // -----------------------------------------------------------------
    // 1. Load tokenizer.
    // -----------------------------------------------------------------
    tokenizer_ = std::make_unique<OrtTokenizer>(tokenizer_path, max_seq_length);
    if (!tokenizer_->is_valid()) {
        spdlog::error("OrtEmbedder: tokenizer initialization failed");
        return;
    }

    // -----------------------------------------------------------------
    // 2. Create ORT environment and session.
    // -----------------------------------------------------------------
    try {
        ort_ = std::make_unique<OrtState>();

        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Try CUDA EP first if requested.
        bool cuda_success = false;
        if (device == DevicePreference::CUDA) {
            try {
                OrtCUDAProviderOptions cuda_options{};
                cuda_options.device_id = cuda_device_id;
                // Use default GPU memory settings; the RTX 3060 6GB has
                // plenty for a ~100MB embedding model.
                session_options.AppendExecutionProvider_CUDA(cuda_options);
                cuda_success = true;
                spdlog::info("  CUDA EP appended (device {})", cuda_device_id);
            } catch (const Ort::Exception& e) {
                spdlog::warn("  CUDA EP not available: {}", e.what());
                spdlog::warn("  Falling back to CPU");
            }
        }

        // Convert model path to wchar_t for the Windows ORT API.
        // On non-Windows platforms, ORT accepts char* paths directly.
#ifdef _WIN32
        std::wstring wmodel_path(model_path.begin(), model_path.end());
        ort_->session = Ort::Session(ort_->env, wmodel_path.c_str(),
                                     session_options);
#else
        ort_->session = Ort::Session(ort_->env, model_path.c_str(),
                                     session_options);
#endif

        // Determine which EP is actually active.
        if (cuda_success) {
            active_provider_ = "CUDAExecutionProvider";
        } else {
            active_provider_ = "CPUExecutionProvider";
        }
        spdlog::info("  Active EP: {}", active_provider_);

    } catch (const Ort::Exception& e) {
        spdlog::error("OrtEmbedder: failed to create ORT session: {}",
                      e.what());
        ort_.reset();
        return;
    }

    // -----------------------------------------------------------------
    // 3. Probe the model to determine embedding dimension.
    //    Feed a minimal input (single token) and check the output shape.
    // -----------------------------------------------------------------
    try {
        // Minimal input: [CLS][SEP] -> 2 tokens.
        std::vector<int64_t> probe_ids = {0, 0};
        std::vector<int64_t> probe_mask = {1, 1};

        std::array<int64_t, 2> input_shape = {1, 2};

        auto ids_tensor = Ort::Value::CreateTensor<int64_t>(
            ort_->memory_info, probe_ids.data(), probe_ids.size(),
            input_shape.data(), input_shape.size());

        auto mask_tensor = Ort::Value::CreateTensor<int64_t>(
            ort_->memory_info, probe_mask.data(), probe_mask.size(),
            input_shape.data(), input_shape.size());

        const char* input_names[] = {"input_ids", "attention_mask"};
        const char* output_names[] = {"embeddings"};

        std::array<Ort::Value, 2> input_tensors{
            std::move(ids_tensor), std::move(mask_tensor)};

        auto outputs = ort_->session.Run(
            Ort::RunOptions{nullptr},
            input_names, input_tensors.data(), 2,
            output_names, 1);

        auto output_info = outputs[0].GetTensorTypeAndShapeInfo();
        auto output_shape = output_info.GetShape();

        if (output_shape.size() >= 2) {
            dim_ = static_cast<size_t>(output_shape[1]);
        } else if (output_shape.size() == 1) {
            dim_ = static_cast<size_t>(output_shape[0]);
        }

        spdlog::info("  Embedding dimension: {}", dim_);

    } catch (const Ort::Exception& e) {
        spdlog::error("OrtEmbedder: model probe failed: {}", e.what());
        ort_.reset();
        return;
    }

    if (dim_ == 0) {
        spdlog::error("OrtEmbedder: could not determine embedding dimension");
        ort_.reset();
        return;
    }

    // -----------------------------------------------------------------
    // 4. Derive model name from the file path.
    // -----------------------------------------------------------------
    {
        auto pos = model_path.find_last_of("/\\");
        model_name_ = (pos != std::string::npos)
                          ? model_path.substr(pos + 1)
                          : model_path;
        // Strip .onnx extension.
        auto dot = model_name_.rfind(".onnx");
        if (dot != std::string::npos) {
            model_name_ = model_name_.substr(0, dot);
        }
    }

    valid_ = true;
    spdlog::info("OrtEmbedder: ready (model={}, dim={}, ep={})",
                 model_name_, dim_, active_provider_);
}

OrtEmbedder::~OrtEmbedder()
{
    // OrtState destructor handles cleanup.
    // Explicit destructor needed because OrtState is an incomplete type
    // in the header.
}

// ---------------------------------------------------------------------------
// Embedder interface
// ---------------------------------------------------------------------------

std::vector<float> OrtEmbedder::embed(const std::string& text)
{
    if (!valid_ || !ort_ || !tokenizer_) return {};

    // Tokenize.
    auto tok_output = tokenizer_->encode_with_mask(text);
    if (tok_output.input_ids.empty()) return {};

    // Wrap in a batch of 1.
    BatchTokenizerOutput batch;
    batch.batch_size = 1;
    batch.seq_length = tok_output.input_ids.size();
    batch.input_ids = std::move(tok_output.input_ids);
    batch.attention_mask = std::move(tok_output.attention_mask);

    // Run inference.
    auto raw = run_inference(batch);
    if (raw.size() != dim_) return {};

    // L2-normalize.
    if (!l2_normalize(raw.data(), dim_)) {
        spdlog::warn("OrtEmbedder: zero-magnitude embedding for input text");
    }

    return raw;
}

std::vector<std::vector<float>> OrtEmbedder::embed_batch(
    const std::vector<std::string>& texts)
{
    std::vector<std::vector<float>> results;

    if (!valid_ || !ort_ || !tokenizer_ || texts.empty()) return results;

    // Tokenize all inputs with padding.
    auto batch = tokenizer_->encode_batch(texts);
    if (batch.input_ids.empty()) return results;

    // Run batched inference.
    auto raw = run_inference(batch);
    if (raw.size() != batch.batch_size * dim_) {
        spdlog::error("OrtEmbedder: unexpected output size: {} (expected {})",
                      raw.size(), batch.batch_size * dim_);
        return results;
    }

    // Split flat output into individual vectors and normalize each.
    results.reserve(batch.batch_size);
    for (size_t i = 0; i < batch.batch_size; ++i) {
        std::vector<float> vec(raw.begin() + static_cast<ptrdiff_t>(i * dim_),
                               raw.begin() + static_cast<ptrdiff_t>((i + 1) * dim_));
        if (!l2_normalize(vec.data(), dim_)) {
            spdlog::warn("OrtEmbedder: zero-magnitude embedding for batch item {}",
                         i);
        }
        results.push_back(std::move(vec));
    }

    return results;
}

size_t OrtEmbedder::dimension() const
{
    return dim_;
}

std::string OrtEmbedder::model_name() const
{
    return model_name_;
}

// ---------------------------------------------------------------------------
// Internal inference
// ---------------------------------------------------------------------------

std::vector<float> OrtEmbedder::run_inference(const BatchTokenizerOutput& batch)
{
    if (!ort_) return {};

    try {
        std::array<int64_t, 2> input_shape = {
            static_cast<int64_t>(batch.batch_size),
            static_cast<int64_t>(batch.seq_length)};

        // Create input tensors from the batch data.
        // Note: Ort::Value::CreateTensor takes non-const pointers but does not
        // modify the data.  We const_cast here because the batch data is
        // logically immutable during inference.
        auto ids_tensor = Ort::Value::CreateTensor<int64_t>(
            ort_->memory_info,
            const_cast<int64_t*>(batch.input_ids.data()),
            batch.input_ids.size(),
            input_shape.data(), input_shape.size());

        auto mask_tensor = Ort::Value::CreateTensor<int64_t>(
            ort_->memory_info,
            const_cast<int64_t*>(batch.attention_mask.data()),
            batch.attention_mask.size(),
            input_shape.data(), input_shape.size());

        const char* input_names[] = {"input_ids", "attention_mask"};
        const char* output_names[] = {"embeddings"};

        std::array<Ort::Value, 2> input_tensors{
            std::move(ids_tensor), std::move(mask_tensor)};

        // Run inference.
        std::vector<Ort::Value> outputs;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            outputs = ort_->session.Run(
                Ort::RunOptions{nullptr},
                input_names, input_tensors.data(), 2,
                output_names, 1);
        }

        // Extract output data.
        auto output_info = outputs[0].GetTensorTypeAndShapeInfo();
        auto output_shape = output_info.GetShape();
        size_t total_elements = static_cast<size_t>(
            std::accumulate(output_shape.begin(), output_shape.end(),
                            int64_t{1}, std::multiplies<int64_t>{}));

        const float* output_data = outputs[0].GetTensorData<float>();
        return std::vector<float>(output_data, output_data + total_elements);

    } catch (const Ort::Exception& e) {
        spdlog::error("OrtEmbedder: inference failed: {}", e.what());
        return {};
    }
}

// ---------------------------------------------------------------------------
// L2 normalization
// ---------------------------------------------------------------------------

bool OrtEmbedder::l2_normalize(float* data, size_t dim)
{
    float sum_sq = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        sum_sq += data[i] * data[i];
    }

    if (sum_sq < 1e-12f) {
        return false;
    }

    const float inv_norm = 1.0f / std::sqrt(sum_sq);
    for (size_t i = 0; i < dim; ++i) {
        data[i] *= inv_norm;
    }

    return true;
}

} // namespace engram

#endif // ENGRAM_HAS_ONNX
