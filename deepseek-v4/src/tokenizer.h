#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_map>

// Handles text encoding (string to token IDs) and decoding (token IDs to string)
class Tokenizer {
public:
    Tokenizer() = default;
    ~Tokenizer() = default;

    // Loads vocabulary mapping from a simple text file
    // File format: each line contains "<token_id> <token_bytes_in_hex_or_text>"
    bool load(const std::string& vocab_path);

    // Encodes a text prompt into a sequence of token IDs
    std::vector<int> encode(const std::string& text) const;

    // Decodes a single token ID back to its string representation
    std::string decode(int token_id) const;

    // Decodes a sequence of token IDs back to a full string
    std::string decode(const std::vector<int>& tokens) const;

    size_t vocab_size() const { return id_to_token.size(); }

private:
    // Lookup tables for fast encoding and decoding
    std::unordered_map<std::string, int> token_to_id;
    std::unordered_map<int, std::string> id_to_token;

    // Helper to replace escape sequences (like \n, \t, spaces)
    std::string clean_token_text(const std::string& raw) const;
};

#endif // TOKENIZER_H
