import os
import json

def extract_vocab():
    vocab_in = "D:/qwen_base_raw/vocab.json"
    vocab_out = "D:/qwen_sharded/vocab.txt"
    
    print("==================================================")
    # Ensure output directory exists
    os.makedirs(os.path.dirname(vocab_out), exist_ok=True)
    
    print(f"Loading vocabulary from: {vocab_in}")
    with open(vocab_in, "r", encoding="utf-8") as f:
        vocab = json.load(f)
        
    print(f"Mapping and sorting {len(vocab)} tokens...")
    
    # Sort tokens by ID
    # vocab is dict: token_str -> id
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    
    print(f"Writing clean vocabulary to: {vocab_out}")
    with open(vocab_out, "w", encoding="utf-8") as f:
        for token, token_id in sorted_vocab:
            # Escape according to C++ clean_token_text rules:
            # \ -> \\
            # newline -> \n
            # tab -> \t
            # space -> \s (custom escape)
            escaped = ""
            for char in token:
                if char == '\\':
                    escaped += '\\\\'
                elif char == '\n':
                    escaped += '\\n'
                elif char == '\t':
                    escaped += '\\t'
                elif char == ' ':
                    escaped += '\\s'
                else:
                    escaped += char
            
            f.write(f"{token_id} {escaped}\n")
            
    print("[Success] Vocabulary extracted and formatted successfully!")

if __name__ == "__main__":
    extract_vocab()
