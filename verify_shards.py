import os
import json
import struct

def verify_shards():
    sharded_dir = "D:/qwen_sharded"
    experts_dir = os.path.join(sharded_dir, "experts")
    
    print("==================================================")
    print(" Qwen3.6-35B-A3B Shard Verification Utility")
    print("==================================================")
    
    # 1. Verify Base Layers
    base_file = os.path.join(sharded_dir, "base.safetensors")
    if not os.path.exists(base_file):
        print(f"[FAIL] Base layers file does not exist: {base_file}")
        return False
        
    print("Verifying base.safetensors...")
    try:
        with open(base_file, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header_json = f.read(header_size).decode('utf-8')
            header = json.loads(header_json)
            
            # Check key components
            required_base = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
            for req in required_base:
                if req not in header:
                    print(f"  [FAIL] Missing required base tensor: {req}")
                    return False
            print(f"  [OK] base.safetensors header is clean. Contains {len(header)} tensors.")
    except Exception as e:
        print(f"  [FAIL] base.safetensors is corrupted: {e}")
        return False

    # 2. Verify all 10,240 Experts
    print("\nScanning 10,240 expert files in experts/ directory...")
    if not os.path.exists(experts_dir):
        print(f"[FAIL] Experts directory does not exist: {experts_dir}")
        return False
        
    total_expected = 10240
    expert_files = [f for f in os.listdir(experts_dir) if f.endswith(".safetensors")]
    
    if len(expert_files) != total_expected:
        print(f"  [FAIL] Expected {total_expected} expert files, but found {len(expert_files)}.")
        return False

    # Check a sample of 200 experts spread across different layers to verify integrity
    # (Checking all 10,240 in detail is slow, but we can do a quick check of all,
    # and a deep check of a sample)
    print("Performing deep mathematical check on expert file headers...")
    
    expected_shapes = {
        "gate_proj.weight": [512, 2048],
        "up_proj.weight": [512, 2048],
        "down_proj.weight": [2048, 512]
    }
    
    # Block size math: 1,048,576 elements / 32 = 32,768 blocks. 
    # Each block is 4 bytes float + 16 bytes Q4 weights = 20 bytes.
    # Total binary bytes = 655,360 bytes per tensor.
    expected_bytes = 655360 
    
    corrupted_count = 0
    
    for filename in sorted(expert_files):
        path = os.path.join(experts_dir, filename)
        
        # Verify file is not empty
        if os.path.getsize(path) == 0:
            print(f"  [FAIL] Empty expert file found: {filename}")
            corrupted_count += 1
            continue
            
        try:
            with open(path, "rb") as f:
                header_size = struct.unpack("<Q", f.read(8))[0]
                header_json = f.read(header_size).decode('utf-8')
                header = json.loads(header_json)
                
                # Check for all three weights
                for proj in ["gate_proj.weight", "up_proj.weight", "down_proj.weight"]:
                    if proj not in header:
                        print(f"  [FAIL] {filename} is missing projection: {proj}")
                        corrupted_count += 1
                        break
                        
                    meta = header[proj]
                    # Verify shape
                    if meta["shape"] != expected_shapes[proj]:
                        print(f"  [FAIL] {filename} {proj} has wrong shape: {meta['shape']}")
                        corrupted_count += 1
                        break
                        
                    # Verify dtype
                    if meta["dtype"] != "Q4_0":
                        print(f"  [FAIL] {filename} {proj} has wrong dtype: {meta['dtype']}")
                        corrupted_count += 1
                        break
                        
                    # Verify binary size
                    data_size = meta["data_offsets"][1] - meta["data_offsets"][0]
                    if data_size != expected_bytes:
                        print(f"  [FAIL] {filename} {proj} data size mismatch: {data_size} bytes (expected {expected_bytes})")
                        corrupted_count += 1
                        break
        except Exception as e:
            print(f"  [FAIL] Failed to parse {filename}: {e}")
            corrupted_count += 1

    if corrupted_count > 0:
        print(f"\n[FAIL] Verification complete. Found {corrupted_count} corrupted or incomplete expert files.")
        return False
        
    print(f"\n[SUCCESS] All {total_expected} experts have been verified successfully!")
    print("- All files are non-empty.")
    print("- All headers contain the required Q4_0 projection tensors.")
    print("- All tensor dimensions and sizes match the C++ engine layout.")
    return True

if __name__ == "__main__":
    verify_shards()
