import os
import json
import struct

raw_dir = "D:/qwen_base_raw"
files = sorted([f for f in os.listdir(raw_dir) if f.endswith(".safetensors")])

found_tensors = {}

for filename in files:
    file_path = os.path.join(raw_dir, filename)
    with open(file_path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_json = f.read(header_size).decode('utf-8')
        header = json.loads(header_json)
        
        for k, v in header.items():
            if "linear_attn" in k:
                found_tensors[k] = v.get("shape")

# Print unique configurations for Layer 0
print("Layer 0 Gated DeltaNet Tensors:")
for k, shape in sorted(found_tensors.items()):
    if "layers.0." in k:
        print(f"  {k}: {shape}")
