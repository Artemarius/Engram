#pragma once

/// @file ort_tokenizer.hpp
/// @brief WordPiece tokenizer that loads from HuggingFace tokenizer.json.
///
/// Parses the vocabulary and merge/WordPiece rules from a standard
/// HuggingFace tokenizer.json file and produces token ID sequences
/// compatible with BERT-style embedding models (with [CLS] and [SEP]
/// special tokens).

#ifdef ENGRAM_HAS_ONNX

#include "tokenizer.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace engram {

/// @brief Result of tokenizing a single text string.
///
/// Contains the token ID sequence and the corresponding attention mask,
/// ready to be fed into an ONNX Runtime session as int64 tensors.
struct TokenizerOutput {
    /// Token IDs including [CLS] at position 0 and [SEP] at the end.
    std::vector<int64_t> input_ids;

    /// Attention mask: 1 for real tokens, 0 for padding positions.
    std::vector<int64_t> attention_mask;
};

/// @brief Result of tokenizing a batch of texts with padding.
///
/// All sequences are padded to the same length (the longest in the batch
/// or max_length, whichever is smaller).  Data is stored as flat arrays
/// in row-major order for direct use as ONNX Runtime tensor inputs.
struct BatchTokenizerOutput {
    /// Flat array of token IDs, shape [batch_size, seq_length].
    std::vector<int64_t> input_ids;

    /// Flat array of attention masks, shape [batch_size, seq_length].
    std::vector<int64_t> attention_mask;

    /// Number of sequences in the batch.
    size_t batch_size = 0;

    /// Padded sequence length (columns).
    size_t seq_length = 0;
};

/// @class OrtTokenizer
/// @brief Concrete Tokenizer that loads a HuggingFace tokenizer.json file.
///
/// Implements WordPiece tokenization with the following pipeline:
///   1. Unicode-aware lowercasing (when do_lower_case is enabled)
///   2. Whitespace + punctuation splitting (basic tokenization)
///   3. WordPiece sub-word splitting using the loaded vocabulary
///   4. Prepending [CLS] and appending [SEP] special tokens
///   5. Truncation to max_length
///
/// The tokenizer is designed for BERT-family models (MiniLM, Nomic Embed)
/// that expect WordPiece-tokenized input with special tokens.
class OrtTokenizer final : public Tokenizer {
public:
    /// @brief Construct a tokenizer by loading a HuggingFace tokenizer.json.
    ///
    /// The file is expected to contain at minimum:
    ///   - "model.vocab" — a mapping from token string to integer ID
    ///   - Optionally "model.type" == "WordPiece"
    ///   - Optionally "added_tokens" for special tokens
    ///
    /// @param tokenizer_json_path  Path to the tokenizer.json file.
    /// @param max_length           Maximum sequence length including special
    ///                             tokens. Sequences longer than this are
    ///                             truncated. Default: 512.
    explicit OrtTokenizer(const std::string& tokenizer_json_path,
                          size_t max_length = 512);

    ~OrtTokenizer() override = default;

    // Non-copyable but movable.
    OrtTokenizer(const OrtTokenizer&) = delete;
    OrtTokenizer& operator=(const OrtTokenizer&) = delete;
    OrtTokenizer(OrtTokenizer&&) noexcept = default;
    OrtTokenizer& operator=(OrtTokenizer&&) noexcept = default;

    // --- Tokenizer interface -------------------------------------------------

    /// @brief Tokenize text into a sequence of token IDs.
    ///
    /// Includes [CLS] at the front and [SEP] at the end.  Truncates to
    /// max_length if necessary.
    ///
    /// @param text  Raw input text.
    /// @return Token ID sequence.  Empty on failure.
    std::vector<int64_t> encode(const std::string& text) override;

    /// @brief Decode token IDs back to text.
    ///
    /// Strips [CLS], [SEP], [PAD], and [UNK] tokens.  Joins sub-word
    /// tokens by removing the "##" prefix.
    ///
    /// @param ids  Token ID sequence.
    /// @return Reconstructed text.  Empty on failure.
    std::string decode(const std::vector<int64_t>& ids) override;

    /// @brief Returns the vocabulary size.
    size_t vocab_size() const override;

    // --- Extended API for the embedder ---------------------------------------

    /// @brief Tokenize a single text and produce input_ids + attention_mask.
    ///
    /// The output is NOT padded (variable length up to max_length).
    ///
    /// @param text  Raw input text.
    /// @return TokenizerOutput with input_ids and attention_mask.
    TokenizerOutput encode_with_mask(const std::string& text);

    /// @brief Tokenize a batch of texts with padding to the longest sequence.
    ///
    /// All outputs are padded to the same length for batched inference.
    /// The padding token ID is taken from the vocabulary ([PAD]).
    ///
    /// @param texts  Vector of input texts.
    /// @return BatchTokenizerOutput with flat arrays suitable for ONNX tensors.
    BatchTokenizerOutput encode_batch(const std::vector<std::string>& texts);

    /// @brief Returns the maximum sequence length (including special tokens).
    size_t max_length() const noexcept { return max_length_; }

    /// @brief Returns true if the tokenizer loaded successfully.
    bool is_valid() const noexcept { return valid_; }

private:
    /// Load and parse the tokenizer.json file.
    bool load_tokenizer(const std::string& path);

    /// Basic pre-tokenization: split on whitespace and punctuation.
    std::vector<std::string> basic_tokenize(const std::string& text) const;

    /// WordPiece sub-word tokenization of a single pre-tokenized word.
    std::vector<std::string> wordpiece_tokenize(const std::string& word) const;

    /// Look up the ID for a token string, returning unk_id_ if not found.
    int64_t token_to_id(const std::string& token) const;

    /// Look up the string for a token ID.
    std::string id_to_token(int64_t id) const;

    /// Token string -> ID mapping.
    std::unordered_map<std::string, int64_t> vocab_;

    /// ID -> token string reverse mapping (for decode).
    std::unordered_map<int64_t, std::string> id_to_token_map_;

    /// Special token IDs.
    int64_t cls_id_ = 0;
    int64_t sep_id_ = 0;
    int64_t pad_id_ = 0;
    int64_t unk_id_ = 0;

    /// Maximum sequence length including [CLS] and [SEP].
    size_t max_length_ = 512;

    /// Whether to lowercase input text before tokenizing.
    bool do_lower_case_ = true;

    /// WordPiece continuation prefix (usually "##").
    std::string continuation_prefix_ = "##";

    /// Maximum characters in a single word for WordPiece; longer words
    /// are replaced with [UNK].
    size_t max_word_length_ = 200;

    /// Whether the tokenizer loaded successfully.
    bool valid_ = false;
};

} // namespace engram

#endif // ENGRAM_HAS_ONNX
