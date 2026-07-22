import json
import struct
import os

SHARD_DIR = "D:/deepseek_sharded"

def read_safetensors_keys(file_path):
    try:
        with open(file_path, "rb") as f:
            header_len_bytes = f.read(8)
            if len(header_len_bytes) < 8:
                return None
            header_len = struct.unpack("<Q", header_len_bytes)[0]
            header_bytes = f.read(header_len)
            header = json.loads(header_bytes.decode('utf-8'))
            return [k for k in header.keys() if k != "__metadata__"]
    except Exception as e:
        return f"Error: {e}"

def main():
    print("==================================================")
    print(" HydraEngine DeepSeek-V4 Shard Verification Tool")
    print("==================================================")
    
    # 1. Check metadata and layer files
    meta_path = os.path.join(SHARD_DIR, "base_meta.safetensors")
    if os.path.exists(meta_path):
        keys = read_safetensors_keys(meta_path)
        print(f"[OK] base_meta.safetensors (Tensors: {keys})")
    else:
        print("[FAIL] base_meta.safetensors is missing!")
        
    layer0_path = os.path.join(SHARD_DIR, "base_layer_0.safetensors")
    if os.path.exists(layer0_path):
        keys = read_safetensors_keys(layer0_path)
        print(f"[OK] base_layer_0.safetensors (Total Tensors: {len(keys)})")
        print(f"     Tensors: {keys[:5]}...")
    else:
        print("[FAIL] base_layer_0.safetensors is missing!")

    # 2. Check Vocabulary
    vocab_path = os.path.join(SHARD_DIR, "vocab.txt")
    if os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        print(f"[OK] vocab.txt (Total tokens loaded: {len(lines)})")
    else:
        print("[FAIL] vocab.txt is missing!")
        
    # 3. Check a few random expert shards
    experts_dir = os.path.join(SHARD_DIR, "experts")
    test_shards = [
        os.path.join(experts_dir, "expert_0_0.safetensors"),
        os.path.join(experts_dir, "expert_12_50.safetensors"),
        os.path.join(experts_dir, "expert_24_100.safetensors"),
        os.path.join(experts_dir, "expert_42_255.safetensors")
    ]
    
    print("\nChecking Expert Shards:")
    for path in test_shards:
        filename = os.path.basename(path)
        if os.path.exists(path):
            keys = read_safetensors_keys(path)
            print(f"- {filename}: OK (Tensors: {keys})")
        else:
            print(f"- {filename}: NOT YET GENERATED (pipeline is running)")
            
    print("\nVerification complete.")

if __name__ == "__main__":
    main()
