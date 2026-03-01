/// @file ort_tokenizer.cpp
/// @brief WordPiece tokenizer implementation — loads HuggingFace tokenizer.json.

#ifdef ENGRAM_HAS_ONNX

#include "ort_tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace engram {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

OrtTokenizer::OrtTokenizer(const std::string& tokenizer_json_path,
                           size_t max_length)
    : max_length_(max_length)
{
    valid_ = load_tokenizer(tokenizer_json_path);
    if (valid_) {
        spdlog::info("OrtTokenizer: loaded {} tokens from {}",
                     vocab_.size(), tokenizer_json_path);
    } else {
        spdlog::error("OrtTokenizer: failed to load tokenizer from {}",
                      tokenizer_json_path);
    }
}

// ---------------------------------------------------------------------------
// Tokenizer interface
// ---------------------------------------------------------------------------

std::vector<int64_t> OrtTokenizer::encode(const std::string& text)
{
    if (!valid_) return {};

    auto output = encode_with_mask(text);
    return output.input_ids;
}

std::string OrtTokenizer::decode(const std::vector<int64_t>& ids)
{
    if (!valid_) return {};

    std::string result;
    bool first = true;

    for (auto id : ids) {
        // Skip special tokens.
        if (id == cls_id_ || id == sep_id_ || id == pad_id_) {
            continue;
        }

        std::string token = id_to_token(id);

        // Handle WordPiece continuation tokens: strip "##" prefix and
        // concatenate without a space.
        if (token.size() > continuation_prefix_.size() &&
            token.substr(0, continuation_prefix_.size()) == continuation_prefix_)
        {
            result += token.substr(continuation_prefix_.size());
        } else {
            if (!first) {
                result += ' ';
            }
            result += token;
            first = false;
        }
    }

    return result;
}

size_t OrtTokenizer::vocab_size() const
{
    return vocab_.size();
}

// ---------------------------------------------------------------------------
// Extended API
// ---------------------------------------------------------------------------

TokenizerOutput OrtTokenizer::encode_with_mask(const std::string& text)
{
    TokenizerOutput output;

    if (!valid_) return output;

    // 1. Basic tokenization: whitespace + punctuation splitting.
    auto words = basic_tokenize(text);

    // 2. WordPiece sub-word tokenization.
    std::vector<int64_t> token_ids;
    token_ids.reserve(max_length_);

    // Reserve space for [CLS] at the front.
    token_ids.push_back(cls_id_);

    for (const auto& word : words) {
        auto sub_tokens = wordpiece_tokenize(word);
        for (const auto& sub : sub_tokens) {
            if (token_ids.size() >= max_length_ - 1) {
                // Leave room for [SEP].
                break;
            }
            token_ids.push_back(token_to_id(sub));
        }
        if (token_ids.size() >= max_length_ - 1) {
            break;
        }
    }

    // 3. Append [SEP].
    token_ids.push_back(sep_id_);

    // 4. Build attention mask (all 1s, no padding at this stage).
    output.input_ids = std::move(token_ids);
    output.attention_mask.assign(output.input_ids.size(), 1);

    return output;
}

BatchTokenizerOutput OrtTokenizer::encode_batch(const std::vector<std::string>& texts)
{
    BatchTokenizerOutput batch;
    batch.batch_size = texts.size();

    if (!valid_ || texts.empty()) return batch;

    // Tokenize each input individually.
    std::vector<TokenizerOutput> individual;
    individual.reserve(texts.size());

    size_t max_len = 0;
    for (const auto& text : texts) {
        individual.push_back(encode_with_mask(text));
        max_len = std::max(max_len, individual.back().input_ids.size());
    }

    batch.seq_length = max_len;

    // Allocate flat arrays and pad shorter sequences.
    const size_t total = batch.batch_size * batch.seq_length;
    batch.input_ids.resize(total, pad_id_);
    batch.attention_mask.resize(total, 0);

    for (size_t i = 0; i < batch.batch_size; ++i) {
        const auto& ids = individual[i].input_ids;
        const auto& mask = individual[i].attention_mask;
        const size_t offset = i * batch.seq_length;

        for (size_t j = 0; j < ids.size(); ++j) {
            batch.input_ids[offset + j] = ids[j];
            batch.attention_mask[offset + j] = mask[j];
        }
        // Remaining positions are already pad_id_ / 0.
    }

    return batch;
}

// ---------------------------------------------------------------------------
// Loading tokenizer.json
// ---------------------------------------------------------------------------

bool OrtTokenizer::load_tokenizer(const std::string& path)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        spdlog::error("OrtTokenizer: cannot open file: {}", path);
        return false;
    }

    nlohmann::json root;
    try {
        root = nlohmann::json::parse(ifs);
    } catch (const nlohmann::json::parse_error& e) {
        spdlog::error("OrtTokenizer: JSON parse error: {}", e.what());
        return false;
    }

    // -----------------------------------------------------------------
    // Extract vocabulary from the "model" section.
    //
    // HuggingFace tokenizer.json has this structure:
    //   { "model": { "type": "WordPiece", "vocab": { "[PAD]": 0, ... } } }
    //
    // Some tokenizers may use "model.vocab" as a dict or as an array
    // of [token, id] pairs.  We handle the dict case.
    // -----------------------------------------------------------------
    if (!root.contains("model")) {
        spdlog::error("OrtTokenizer: tokenizer.json missing 'model' section");
        return false;
    }

    const auto& model = root["model"];

    // Check model type (informational).
    if (model.contains("type")) {
        std::string model_type = model["type"].get<std::string>();
        spdlog::debug("OrtTokenizer: model type = {}", model_type);
    }

    // Load continuation prefix if specified.
    if (model.contains("continuing_subword_prefix")) {
        continuation_prefix_ = model["continuing_subword_prefix"].get<std::string>();
    }

    // Load max word length for WordPiece.
    if (model.contains("max_input_chars_per_word")) {
        max_word_length_ = model["max_input_chars_per_word"].get<size_t>();
    }

    // Load vocabulary.
    if (!model.contains("vocab")) {
        spdlog::error("OrtTokenizer: tokenizer.json missing 'model.vocab'");
        return false;
    }

    const auto& vocab_json = model["vocab"];
    if (!vocab_json.is_object()) {
        spdlog::error("OrtTokenizer: 'model.vocab' is not a JSON object");
        return false;
    }

    for (auto it = vocab_json.begin(); it != vocab_json.end(); ++it) {
        int64_t id = it.value().get<int64_t>();
        vocab_[it.key()] = id;
        id_to_token_map_[id] = it.key();
    }

    if (vocab_.empty()) {
        spdlog::error("OrtTokenizer: vocabulary is empty");
        return false;
    }

    // -----------------------------------------------------------------
    // Process added_tokens for special token IDs.
    // -----------------------------------------------------------------
    if (root.contains("added_tokens") && root["added_tokens"].is_array()) {
        for (const auto& tok : root["added_tokens"]) {
            if (!tok.contains("content") || !tok.contains("id")) continue;
            std::string content = tok["content"].get<std::string>();
            int64_t id = tok["id"].get<int64_t>();
            // Also add to vocab if not already present.
            if (vocab_.find(content) == vocab_.end()) {
                vocab_[content] = id;
                id_to_token_map_[id] = content;
            }
        }
    }

    // -----------------------------------------------------------------
    // Resolve special token IDs.
    // -----------------------------------------------------------------
    auto resolve_special = [&](const std::string& name, int64_t& target) {
        auto it = vocab_.find(name);
        if (it != vocab_.end()) {
            target = it->second;
            return true;
        }
        return false;
    };

    if (!resolve_special("[CLS]", cls_id_)) {
        // Some models use <s> instead of [CLS].
        if (!resolve_special("<s>", cls_id_)) {
            spdlog::warn("OrtTokenizer: [CLS]/<s> token not found in vocab, using ID 0");
            cls_id_ = 0;
        }
    }

    if (!resolve_special("[SEP]", sep_id_)) {
        // Some models use </s> instead of [SEP].
        if (!resolve_special("</s>", sep_id_)) {
            spdlog::warn("OrtTokenizer: [SEP]/</s> token not found in vocab, using ID 0");
            sep_id_ = 0;
        }
    }

    if (!resolve_special("[PAD]", pad_id_)) {
        if (!resolve_special("<pad>", pad_id_)) {
            spdlog::warn("OrtTokenizer: [PAD]/<pad> token not found in vocab, using ID 0");
            pad_id_ = 0;
        }
    }

    if (!resolve_special("[UNK]", unk_id_)) {
        if (!resolve_special("<unk>", unk_id_)) {
            spdlog::warn("OrtTokenizer: [UNK]/<unk> token not found in vocab, using ID 0");
            unk_id_ = 0;
        }
    }

    spdlog::debug("OrtTokenizer: special tokens — CLS={}, SEP={}, PAD={}, UNK={}",
                  cls_id_, sep_id_, pad_id_, unk_id_);

    // -----------------------------------------------------------------
    // Detect lowercasing preference from normalizer config.
    // -----------------------------------------------------------------
    if (root.contains("normalizer") && root["normalizer"].is_object()) {
        const auto& norm = root["normalizer"];
        if (norm.contains("lowercase")) {
            do_lower_case_ = norm["lowercase"].get<bool>();
        }
        // BertNormalizer style
        if (norm.contains("type") && norm["type"] == "BertNormalizer") {
            if (norm.contains("lowercase")) {
                do_lower_case_ = norm["lowercase"].get<bool>();
            }
        }
        // Sequence of normalizers
        if (norm.contains("type") && norm["type"] == "Sequence" &&
            norm.contains("normalizers") && norm["normalizers"].is_array())
        {
            for (const auto& n : norm["normalizers"]) {
                if (n.contains("type") && n["type"] == "Lowercase") {
                    do_lower_case_ = true;
                }
            }
        }
    }

    spdlog::debug("OrtTokenizer: do_lower_case = {}", do_lower_case_);

    return true;
}

// ---------------------------------------------------------------------------
// Basic pre-tokenization
// ---------------------------------------------------------------------------

std::vector<std::string> OrtTokenizer::basic_tokenize(const std::string& text) const
{
    std::vector<std::string> tokens;
    std::string current;

    // Optionally lowercase the text.
    std::string normalized = text;
    if (do_lower_case_) {
        std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                       [](unsigned char c) -> char {
                           return static_cast<char>(std::tolower(c));
                       });
    }

    // Strip accents is not implemented — for ASCII-dominated code content
    // this is rarely needed.  A production tokenizer would add NFD
    // normalization and accent stripping here.

    // Split on whitespace and punctuation, keeping punctuation as separate
    // tokens.  This matches BERT's BasicTokenizer behavior for ASCII input.
    for (size_t i = 0; i < normalized.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(normalized[i]);

        if (std::isspace(c)) {
            // Whitespace: flush current token.
            if (!current.empty()) {
                tokens.push_back(std::move(current));
                current.clear();
            }
        } else if (std::ispunct(c)) {
            // Punctuation: flush current, then emit punctuation as its own token.
            if (!current.empty()) {
                tokens.push_back(std::move(current));
                current.clear();
            }
            tokens.push_back(std::string(1, static_cast<char>(c)));
        } else if (c <= 0x1F || c == 0x7F) {
            // Control characters: skip.
        } else {
            current += static_cast<char>(c);
        }
    }

    if (!current.empty()) {
        tokens.push_back(std::move(current));
    }

    return tokens;
}

// ---------------------------------------------------------------------------
// WordPiece sub-word tokenization
// ---------------------------------------------------------------------------

std::vector<std::string> OrtTokenizer::wordpiece_tokenize(const std::string& word) const
{
    if (word.size() > max_word_length_) {
        return {"[UNK]"};
    }

    std::vector<std::string> sub_tokens;
    size_t start = 0;
    bool is_bad = false;

    while (start < word.size()) {
        size_t end = word.size();
        std::string best_match;

        while (start < end) {
            std::string substr = word.substr(start, end - start);
            if (start > 0) {
                substr = continuation_prefix_ + substr;
            }

            auto it = vocab_.find(substr);
            if (it != vocab_.end()) {
                best_match = substr;
                break;
            }
            --end;
        }

        if (best_match.empty()) {
            // No match found for any substring starting at `start`.
            is_bad = true;
            break;
        }

        sub_tokens.push_back(std::move(best_match));
        start = end;
    }

    if (is_bad) {
        return {"[UNK]"};
    }

    return sub_tokens;
}

// ---------------------------------------------------------------------------
// Token <-> ID helpers
// ---------------------------------------------------------------------------

int64_t OrtTokenizer::token_to_id(const std::string& token) const
{
    auto it = vocab_.find(token);
    if (it != vocab_.end()) {
        return it->second;
    }
    return unk_id_;
}

std::string OrtTokenizer::id_to_token(int64_t id) const
{
    auto it = id_to_token_map_.find(id);
    if (it != id_to_token_map_.end()) {
        return it->second;
    }
    return "[UNK]";
}

} // namespace engram

#endif // ENGRAM_HAS_ONNX
