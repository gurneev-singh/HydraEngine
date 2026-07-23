import os
import json
import struct
import numpy as np
import torch
import mmap

D_DIR = "D:/deepseek_raw"
OUT_DIR = "D:/deepseek_sharded"

print("==================================================")
print(" DeepSeek-V4-Flash Sharder & Quantizer (Q8/Q4)")
print("==================================================")
print(f"Source: {D_DIR}")
print(f"Output: {OUT_DIR}")

if not os.path.exists(D_DIR):
    print(f"[Error] Source directory {D_DIR} does not exist.")
    exit(1)

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "experts"), exist_ok=True)

# Helper function to convert BF16 bytes to float32 NumPy array
def convert_bf16_to_f32(data_bytes):
    bf16_arr = np.frombuffer(data_bytes, dtype=np.uint8).view(np.uint16)
    f32_arr = np.zeros(len(bf16_arr), dtype=np.float32)
    f32_view = f32_arr.view(np.uint32)
    f32_view[:] = bf16_arr.astype(np.uint32) << 16
    return f32_arr

# Helper to quantize Float32 to Q8_0 block format
def quantize_q8_0(f32_arr):
    size = f32_arr.size
    remainder = size % 32
    if remainder != 0:
        padding = 32 - remainder
        f32_arr = np.concatenate([f32_arr, np.zeros(padding, dtype=np.float32)])
        
    blocks = f32_arr.reshape(-1, 32)
    max_vals = np.max(np.abs(blocks), axis=1)
    scales = max_vals / 127.0
    scales = np.where(scales == 0.0, 1e-5, scales)
    
    q_vals = np.round(blocks / scales[:, np.newaxis])
    q_vals = np.clip(q_vals, -128, 127).astype(np.int8)
    
    block_type = np.dtype([('d', '<f4'), ('qs', 'i1', (32,))])
    packed = np.zeros(blocks.shape[0], dtype=block_type)
    packed['d'] = scales
    packed['qs'] = q_vals
    return packed.tobytes()

# Helper to quantize Float32 to Q4_0 block format
def quantize_q4_0(f32_arr):
    size = f32_arr.size
    remainder = size % 32
    if remainder != 0:
        padding = 32 - remainder
        f32_arr = np.concatenate([f32_arr, np.zeros(padding, dtype=np.float32)])
        
    blocks = f32_arr.reshape(-1, 32)
    max_vals = np.max(np.abs(blocks), axis=1)
    scales = max_vals / 8.0
    scales = np.where(scales == 0.0, 1e-5, scales)
    
    q_vals = np.round(blocks / scales[:, np.newaxis]) + 8
    q_vals = np.clip(q_vals, 0, 15).astype(np.uint8)
    
    q_pairs = q_vals.reshape(-1, 16, 2)
    packed_qs = q_pairs[:, :, 0] | (q_pairs[:, :, 1] << 4)
    
    block_type = np.dtype([('d', '<f4'), ('qs', 'u1', (16,))])
    packed = np.zeros(blocks.shape[0], dtype=block_type)
    packed['d'] = scales
    packed['qs'] = packed_qs
    return packed.tobytes()

# Helper to quantize Float32 to Q2_K block format (2-bit weights, block size 32)
def quantize_q2_k(f32_arr):
    size = f32_arr.size
    remainder = size % 32
    if remainder != 0:
        padding = 32 - remainder
        f32_arr = np.concatenate([f32_arr, np.zeros(padding, dtype=np.float32)])
        
    blocks = f32_arr.reshape(-1, 32)
    min_vals = np.min(blocks, axis=1)
    max_vals = np.max(blocks, axis=1)
    ranges = max_vals - min_vals
    scales = ranges / 3.0
    scales = np.where(scales == 0.0, 1e-5, scales)
    
    q_vals = np.round((blocks - min_vals[:, np.newaxis]) / scales[:, np.newaxis])
    q_vals = np.clip(q_vals, 0, 3).astype(np.uint8)
    
    q_grouped = q_vals.reshape(-1, 8, 4)
    packed_qs = np.zeros((q_grouped.shape[0], 8), dtype=np.uint8)
    for i in range(4):
        packed_qs |= (q_grouped[:, :, i] << (2 * i))
        
    block_type = np.dtype([('d', '<f4'), ('min', '<f4'), ('qs', 'u1', (8,))])
    packed = np.zeros(blocks.shape[0], dtype=block_type)
    packed['d'] = scales
    packed['min'] = min_vals
    packed['qs'] = packed_qs
    return packed.tobytes()

# Write safetensors file
# Temporary directory for disk-based tensor staging to prevent RAM exhaustion
TEMP_DIR = "D:/deepseek_sharded/temp_base"
os.makedirs(TEMP_DIR, exist_ok=True)

def stage_tensor(name, data_bytes):
    safe_name = name.replace(".", "_").replace("/", "_")
    path = os.path.join(TEMP_DIR, f"{safe_name}.bin")
    with open(path, "wb") as f:
        f.write(data_bytes)
    return path

# Write safetensors file
def write_safetensors(file_path, tensors):
    header = {}
    current_offset = 0
    
    for name, (shape, dtype_str, path_or_bytes) in tensors.items():
        if isinstance(path_or_bytes, str):
            data_len = os.path.getsize(path_or_bytes)
        else:
            data_len = len(path_or_bytes)
            
        start = current_offset
        end = start + data_len
        header[name] = {
            "dtype": dtype_str,
            "shape": shape,
            "data_offsets": [start, end]
        }
        current_offset = end
        
    header["__metadata__"] = {"format": "pt"}
    header_json = json.dumps(header, separators=(',', ':')).encode('utf-8')
    
    header_len = len(header_json)
    padding = (8 - ((8 + header_len) % 8)) % 8
    header_json += b' ' * padding
    
    with open(file_path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        
        # Stream data chunk by chunk to prevent loading everything into RAM
        for name, (shape, dtype_str, path_or_bytes) in tensors.items():
            if isinstance(path_or_bytes, str):
                with open(path_or_bytes, "rb") as temp_f:
                    while chunk := temp_f.read(16 * 1024 * 1024): # 16 MB chunk size
                        f.write(chunk)
            else:
                f.write(path_or_bytes)

# Find raw partition files
raw_files = sorted([f for f in os.listdir(D_DIR) if f.endswith(".safetensors")])
print(f"Found {len(raw_files)} partition files.")

# To prevent memory issues, we write layer files on-the-fly.
# We'll use a dictionary to hold layer-specific base tensors temporarily.
layer_buffers = {} # layer_id -> dict
meta_tensors = {}  # embed, head, final norm

# Dequantization helper for standard block-quantized weight + E8M0 scale
def dequantize_weight_and_scale(weight_bytes, scale_bytes, w_shape, s_shape):
    scale_uint8 = np.frombuffer(scale_bytes, dtype=np.uint8)
    scale_float = (2.0 ** (scale_uint8.astype(np.float32) - 127.0)).reshape(s_shape)
    
    weight_int8 = np.frombuffer(weight_bytes, dtype=np.int8).reshape(w_shape)
    
    rows, cols = w_shape
    block_size = cols // s_shape[1]
    
    weight_blocks = weight_int8.reshape(rows, cols // block_size, block_size).astype(np.float32)
    weight_f32_blocks = weight_blocks * scale_float[:, :, np.newaxis]
    return weight_f32_blocks.reshape(rows, cols)

# Dequantization helper for FP8 (float8_e4m3fn) weight + E8M0 scale
def dequantize_fp8_weight_and_scale(weight_bytes, scale_bytes, w_shape, s_shape):
    t_f8 = torch.frombuffer(weight_bytes, dtype=torch.float8_e4m3fn)
    arr_f32 = t_f8.to(torch.float32).numpy().reshape(w_shape)
    
    scale_uint8 = np.frombuffer(scale_bytes, dtype=np.uint8)
    scale_float = (2.0 ** (scale_uint8.astype(np.float32) - 127.0)).reshape(s_shape)
    
    rows, cols = w_shape
    r_blocks, c_blocks = s_shape
    r_block_size = rows // r_blocks
    c_block_size = cols // c_blocks
    
    weight_tiles = arr_f32.reshape(r_blocks, r_block_size, c_blocks, c_block_size)
    weight_f32_tiles = weight_tiles * scale_float[:, np.newaxis, :, np.newaxis]
    return weight_f32_tiles.reshape(rows, cols)

# Process partition by partition
for idx, filename in enumerate(raw_files):
    file_path = os.path.join(D_DIR, filename)
    print(f"\nProcessing partition [{idx+1}/{len(raw_files)}]: {filename}...")
    
    with open(file_path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_json = f.read(header_size).decode('utf-8')
        header = json.loads(header_json)
        
        # Mmap file to read byte ranges efficiently
        mmapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        header_offset = 8 + header_size
        
        # We need to scan weight tensors and match them with their corresponding scales.
        # Since safetensors lists them in any order, we do a two-pass lookup.
        # We find all keys and group them.
        keys = list(header.keys())
        
        # We will build a set of processed keys to prevent double-processing
        processed_keys = set()
        
        for name in keys:
            if name == "__metadata__" or name in processed_keys:
                continue
                
            # A. Process routed experts: layers.{L}.ffn.experts.{E}.{w_id}.weight
            if "ffn.experts" in name and name.endswith(".weight"):
                # Parse layer and expert IDs
                parts = name.split(".")
                layer_id = int(parts[1])
                expert_id = int(parts[4])
                w_id = parts[5] # "w1", "w2", or "w3"
                
                scale_name = name.replace(".weight", ".scale")
                if scale_name in header:
                    meta_w = header[name]
                    meta_s = header[scale_name]
                    
                    # Read weight bytes
                    start_w = header_offset + meta_w["data_offsets"][0]
                    end_w = header_offset + meta_w["data_offsets"][1]
                    w_bytes = mmapped_file[start_w:end_w]
                    
                    # Read scale bytes
                    start_s = header_offset + meta_s["data_offsets"][0]
                    end_s = header_offset + meta_s["data_offsets"][1]
                    s_bytes = mmapped_file[start_s:end_s]
                    
                    # Dequantize
                    f32_weight = dequantize_weight_and_scale(w_bytes, s_bytes, meta_w["shape"], meta_s["shape"])
                    
                    # Quantize to Q2_K
                    q2_bytes = quantize_q2_k(f32_weight)
                    
                    # Write to expert file
                    exp_file_name = f"expert_{layer_id}_{expert_id}.safetensors"
                    exp_file_path = os.path.join(OUT_DIR, "experts", exp_file_name)
                    
                    # Load existing keys if file exists to merge w1, w2, w3
                    existing = {}
                    if os.path.exists(exp_file_path):
                        with open(exp_file_path, "rb") as exp_f:
                            e_h_size = struct.unpack("<Q", exp_f.read(8))[0]
                            e_h_json = exp_f.read(e_h_size).decode('utf-8')
                            e_h = json.loads(e_h_json)
                            e_offset = 8 + e_h_size
                            for e_name, e_meta in e_h.items():
                                if e_name == "__metadata__": continue
                                exp_f.seek(e_offset + e_meta["data_offsets"][0])
                                existing[e_name] = (e_meta["shape"], e_meta["dtype"], exp_f.read(e_meta["data_offsets"][1] - e_meta["data_offsets"][0]))
                    
                    # Map to standard projection names
                    proj_map = {"w1": "gate_proj.weight", "w2": "down_proj.weight", "w3": "up_proj.weight"}
                    existing[proj_map[w_id]] = (meta_w["shape"], "Q2_K", q2_bytes)
                    
                    write_safetensors(exp_file_path, existing)
                    
                    processed_keys.add(name)
                    processed_keys.add(scale_name)
                    
            # B. Process shared experts: layers.{L}.ffn.shared_experts.{w_id}.weight
            elif "shared_experts" in name and name.endswith(".weight"):
                parts = name.split(".")
                layer_id = int(parts[1])
                w_id = parts[4]
                
                scale_name = name.replace(".weight", ".scale")
                if scale_name in header:
                    meta_w = header[name]
                    meta_s = header[scale_name]
                    
                    w_bytes = mmapped_file[header_offset + meta_w["data_offsets"][0] : header_offset + meta_w["data_offsets"][1]]
                    s_bytes = mmapped_file[header_offset + meta_s["data_offsets"][0] : header_offset + meta_s["data_offsets"][1]]
                    
                    f32_weight = dequantize_fp8_weight_and_scale(w_bytes, s_bytes, meta_w["shape"], meta_s["shape"])
                    q8_bytes = quantize_q8_0(f32_weight)
                    
                    if layer_id not in layer_buffers:
                        layer_buffers[layer_id] = {}
                    
                    proj_map = {"w1": "ffn.shared_experts.gate_proj.weight", "w2": "ffn.shared_experts.down_proj.weight", "w3": "ffn.shared_experts.up_proj.weight"}
                    layer_buffers[layer_id][proj_map[w_id]] = (meta_w["shape"], "Q8_0", stage_tensor(f"layer_{layer_id}_{w_id}_shared", q8_bytes))
                    
                    processed_keys.add(name)
                    processed_keys.add(scale_name)

            # C. Process FP8 attention projection weights: layers.{L}.attn.{proj}.weight
            elif "attn" in name and name.endswith(".weight") and header[name]["dtype"] == "F8_E4M3":
                parts = name.split(".")
                layer_id = int(parts[1])
                proj_name = parts[3] # "wq_a", "wq_b", "wkv", "wo_a", "wo_b"
                
                scale_name = name.replace(".weight", ".scale")
                if scale_name in header:
                    meta_w = header[name]
                    meta_s = header[scale_name]
                    
                    w_bytes = mmapped_file[header_offset + meta_w["data_offsets"][0] : header_offset + meta_w["data_offsets"][1]]
                    s_bytes = mmapped_file[header_offset + meta_s["data_offsets"][0] : header_offset + meta_s["data_offsets"][1]]
                    
                    f32_weight = dequantize_fp8_weight_and_scale(w_bytes, s_bytes, meta_w["shape"], meta_s["shape"])
                    q8_bytes = quantize_q8_0(f32_weight)
                    
                    if layer_id not in layer_buffers:
                        layer_buffers[layer_id] = {}
                        
                    cpp_name = f"self_attn.{proj_name}.weight"
                    layer_buffers[layer_id][cpp_name] = (meta_w["shape"], "Q8_0", stage_tensor(f"layer_{layer_id}_{proj_name}", q8_bytes))
                    
                    processed_keys.add(name)
                    processed_keys.add(scale_name)
            
            # D. Handle all other 1D norms, routers, biases, and metadata
            elif name not in processed_keys:
                meta = header[name]
                data_bytes = mmapped_file[header_offset + meta["data_offsets"][0] : header_offset + meta["data_offsets"][1]]
                
                # Check if it belongs to a specific layer
                if name.startswith("layers."):
                    parts = name.split(".")
                    layer_id = int(parts[1])
                    sub_name = ".".join(parts[2:])
                    
                    if layer_id not in layer_buffers:
                        layer_buffers[layer_id] = {}
                    
                    # Convert BF16 1D weights to Float32
                    if meta["dtype"] == "BF16":
                        f32_arr = convert_bf16_to_f32(data_bytes)
                        layer_buffers[layer_id][sub_name] = (meta["shape"], "F32", stage_tensor(f"layer_{layer_id}_{sub_name}", f32_arr.tobytes()))
                    else:
                        layer_buffers[layer_id][sub_name] = (meta["shape"], meta["dtype"], stage_tensor(f"layer_{layer_id}_{sub_name}", data_bytes))
                
                else:
                    # Global metadata (embed.weight, lm_head.weight, norm.weight)
                    if meta["dtype"] == "BF16":
                        f32_arr = convert_bf16_to_f32(data_bytes)
                        meta_tensors[name] = (meta["shape"], "F32", stage_tensor(f"meta_{name}", f32_arr.tobytes()))
                    else:
                        meta_tensors[name] = (meta["shape"], meta["dtype"], stage_tensor(f"meta_{name}", data_bytes))
                
                processed_keys.add(name)
        
        # Flush completed layer base buffers to individual files to free RAM
        # If we have completed layers in layer_buffers, we check if they are done.
        # Since files are sequential, if a layer index is fully parsed (or we can just write them layer by layer at the very end),
        # let's save completed layers.
        # To be safe, we can write out the layer files at the end of the script, but if we want to save RAM,
        # we can check if the current file does not contain a layer, we write it out.
        # Let's just write them out at the end, as 43 layers of 200MB each is only 8.6 GB, which fits easily in 32GB RAM.
        mmapped_file.close()

# Write out the base meta file
print("\nWriting base_meta.safetensors...")
write_safetensors(os.path.join(OUT_DIR, "base_meta.safetensors"), meta_tensors)
print("[Success] base_meta.safetensors written.")

# Write out layer-specific base files
print("\nWriting layer-specific base files...")
for layer_id, tensors in layer_buffers.items():
    layer_file = f"base_layer_{layer_id}.safetensors"
    print(f"  Writing {layer_file}...")
    write_safetensors(os.path.join(OUT_DIR, layer_file), tensors)

# Cleanup temporary staging directory
import shutil
print("\nCleaning up temporary disk-staging directory...")
if os.path.exists(TEMP_DIR):
    shutil.rmtree(TEMP_DIR)
print("Cleanup complete.")

print("\n==================================================")
print(" Sharding and Quantization Completed Successfully!")
print("==================================================")
