#include "tokenizer.h"
#include <fstream>
#include <sstream>
#include <iostream>

bool Tokenizer::load(const std::string& vocab_path) {
    std::ifstream file(vocab_path);
    if (!file.is_open()) {
        std::cerr << "[Error] Failed to open vocabulary file: " << vocab_path << std::endl;
        return false;
    }

    token_to_id.clear();
    id_to_token.clear();

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        size_t space_idx = line.find(' ');
        if (space_idx == std::string::npos) continue;
        
        try {
            int id = std::stoi(line.substr(0, space_idx));
            std::string token = line.substr(space_idx + 1);
            
            token = clean_token_text(token);
            token_to_id[token] = id;
            id_to_token[id] = token;
        } catch (const std::exception& e) {
            // Skip malformed lines
            continue;
        }
    }
    
    std::cout << "[Tokenizer] Loaded " << id_to_token.size() << " tokens from vocabulary file." << std::endl;
    return true;
}

std::string Tokenizer::clean_token_text(const std::string& raw) const {
    std::string clean;
    clean.reserve(raw.size());
    
    // Parse escaped sequences (like \n, \t, or spaces)
    for (size_t i = 0; i < raw.size(); ++i) {
        if (raw[i] == '\\' && i + 1 < raw.size()) {
            char next = raw[i + 1];
            if (next == 'n') clean.push_back('\n');
            else if (next == 't') clean.push_back('\t');
            else if (next == '\\') clean.push_back('\\');
            else if (next == 's') clean.push_back(' '); // custom escape for space
            else clean.push_back(next);
            i++; // skip next char
        } else {
            clean.push_back(raw[i]);
        }
    }
    return clean;
}

// Greedy longest-match token encoding
std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> tokens;
    size_t pos = 0;
    size_t len = text.size();
    
    while (pos < len) {
        size_t match_len = 0;
        int match_id = -1;
        
        // Find the longest matching substring starting at current position
        // We limit search length to prevent infinite lookup loops
        size_t max_search = std::min(static_cast<size_t>(32), len - pos);
        for (size_t l = 1; l <= max_search; ++l) {
            std::string substr = text.substr(pos, l);
            auto it = token_to_id.find(substr);
            if (it != token_to_id.end()) {
                match_len = l;
                match_id = it->second;
            }
        }
        
        if (match_len > 0) {
            tokens.push_back(match_id);
            pos += match_len;
        } else {
            // Byte-fallback: if character not found in vocabulary,
            // cast the raw byte value directly to a token ID (first 256 are byte values)
            unsigned char raw_byte = static_cast<unsigned char>(text[pos]);
            tokens.push_back(static_cast<int>(raw_byte));
            pos += 1;
        }
    }
    
    return tokens;
}

std::string Tokenizer::decode(int token_id) const {
    auto it = id_to_token.find(token_id);
    if (it != id_to_token.end()) {
        return it->second;
    }
    
    // Byte-fallback decode
    if (token_id >= 0 && token_id < 256) {
        return std::string(1, static_cast<char>(token_id));
    }
    
    return ""; // Unknown token ID fallback
}

std::string Tokenizer::decode(const std::vector<int>& tokens) const {
    std::string text;
    for (int id : tokens) {
        text += decode(id);
    }
    return text;
}
