#pragma once

/// @file tokenizer.hpp
/// @brief Abstract tokenizer interface for Engram.
///
/// Defines the base class for text tokenization.  Concrete implementations
/// may wrap HuggingFace tokenizers-cpp, SentencePiece, or a custom BPE
/// implementation.  The tokenizer converts raw text to token IDs expected
/// by the embedding model and can decode IDs back to text.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace engram {

/// @class Tokenizer
/// @brief Pure-virtual interface for text tokenizers.
///
/// A Tokenizer is tightly coupled to a specific model's vocabulary.
/// Implementations must produce token ID sequences that are compatible
/// with the model they are paired with.
class Tokenizer {
public:
    virtual ~Tokenizer() = default;

    /// @brief Tokenize text into a sequence of token IDs.
    ///
    /// The output includes any special tokens the model expects (e.g.
    /// [CLS], [SEP] for BERT-style models) and respects the model's
    /// maximum sequence length by truncating if necessary.
    ///
    /// @param text  The raw input text to tokenize.
    /// @return A vector of token IDs.  Empty on failure.
    virtual std::vector<int64_t> encode(const std::string& text) = 0;

    /// @brief Decode a sequence of token IDs back to text.
    ///
    /// Special tokens are stripped from the output.  The result is a
    /// best-effort reconstruction; round-tripping (encode then decode)
    /// may not preserve whitespace exactly.
    ///
    /// @param ids  Token ID sequence to decode.
    /// @return The decoded text string.  Empty on failure.
    virtual std::string decode(const std::vector<int64_t>& ids) = 0;

    /// @brief Returns the vocabulary size of the loaded tokenizer model.
    virtual size_t vocab_size() const = 0;
};

} // namespace engram
