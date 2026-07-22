import os
import json
import struct
import numpy as np

DTYPE_MAP = {
    "F32": np.float32,
    "F16": np.float16,
    "BF16": np.uint16,
}

def convert_bf16_to_f32(data_bytes):
    bf16_arr = np.frombuffer(data_bytes, dtype=np.uint16)
    f32_arr = np.zeros(len(bf16_arr), dtype=np.float32)
    f32_view = f32_arr.view(np.uint32)
    f32_view[:] = bf16_arr.astype(np.uint32) << 16
    return f32_arr

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

def write_safetensors(file_path, tensors):
    header = {}
    current_offset = 0
    raw_data = bytearray()
    
    for name, (shape, dtype_str, data_bytes) in tensors.items():
        start = current_offset
        end = start + len(data_bytes)
        header[name] = {
            "dtype": dtype_str,
            "shape": shape,
            "data_offsets": [start, end]
        }
        raw_data.extend(data_bytes)
        current_offset = end
        
    header["__metadata__"] = {"format": "pt"}
    header_json = json.dumps(header, separators=(',', ':')).encode('utf-8')
    
    header_len = len(header_json)
    padding = (8 - ((8 + header_len) % 8)) % 8
    header_json += b' ' * padding
    
    with open(file_path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        f.write(raw_data)

def shard_moe():
    raw_dir = "D:/qwen_base_raw"
    out_dir = "D:/qwen_sharded"
    
    print("==================================================")
    print(" Qwen3.6-35B-A3B MoE Sharder & Quantizer (Q8/Q4)")
    print("==================================================")
    print(f"Source Directory: {raw_dir}")
    print(f"Output Directory: {out_dir}")
    
    if not os.path.exists(raw_dir):
        print(f"[Error] Source directory {raw_dir} does not exist.")
        return
        
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "experts"), exist_ok=True)
    
    raw_files = sorted([f for f in os.listdir(raw_dir) if f.endswith(".safetensors")])
    if not raw_files:
        print("[Error] No safetensors files found.")
        return
        
    print(f"Found {len(raw_files)} partition files.")
    
    base_tensors = {}
    
    for idx, filename in enumerate(raw_files):
        file_path = os.path.join(raw_dir, filename)
        print(f"\nProcessing partition [{idx+1}/{len(raw_files)}]: {filename}...")
        
        with open(file_path, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header_json = f.read(header_size).decode('utf-8')
            header = json.loads(header_json)
            
            import mmap
            mmapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            header_offset = 8 + header_size
            
            for t_name, t_meta in header.items():
                if t_name == "__metadata__": continue
                
                # 1. Skip visual tower / image layers completely to save disk space
                if "model.visual." in t_name:
                    continue
                    
                # 2. Skip MTP auxiliary weight layers (if they are separate in the base files)
                # Keep them if needed, but for baseline text generation we focus on main text weights
                if t_name.startswith("mtp."):
                    continue
                
                start_offset = header_offset + t_meta["data_offsets"][0]
                end_offset = header_offset + t_meta["data_offsets"][1]
                t_bytes = mmapped_file[start_offset:end_offset]
                
                shape = t_meta["shape"]
                dtype_str = t_meta["dtype"]
                
                # Convert to Float32 NumPy array
                if dtype_str == "BF16":
                    f32_arr = convert_bf16_to_f32(t_bytes)
                else:
                    f32_arr = np.frombuffer(t_bytes, dtype=DTYPE_MAP.get(dtype_str, np.float32)).astype(np.float32)
                
                # Translate names to map standard C++ expectations
                clean_name = t_name.replace("model.language_model.", "model.")
                
                # 3. Handle stacked routed experts (3D Tensors: [256, out, in])
                if "mlp.experts" in clean_name:
                    parts = clean_name.split(".")
                    # clean_name: "model.layers.L.mlp.experts.proj"
                    layer_id = int(parts[2])
                    proj_type = parts[5] # "down_proj" or "gate_up_proj"
                    
                    # Reshape 1D flat array back to its original 3D shape
                    f32_arr = f32_arr.reshape(shape)
                    
                    # Iterate over all 256 experts
                    for exp_id in range(shape[0]):
                        expert_slice = f32_arr[exp_id] # 2D slice
                        
                        exp_out_dir = os.path.join(out_dir, f"experts")
                        exp_file_name = f"expert_{layer_id}_{exp_id}.safetensors"
                        exp_file_path = os.path.join(exp_out_dir, exp_file_name)
                        
                        # Load existing expert tensors or initialize a dictionary
                        # Since gate_up_proj and down_proj are in different partition files,
                        # we read/update the existing file on-the-fly to merge them.
                        existing_tensors = {}
                        if os.path.exists(exp_file_path):
                            with open(exp_file_path, "rb") as exp_f:
                                e_h_size = struct.unpack("<Q", exp_f.read(8))[0]
                                e_h_json = exp_f.read(e_h_size).decode('utf-8')
                                e_h = json.loads(e_h_json)
                                e_h_offset = 8 + e_h_size
                                
                                # Read current contents of this expert file
                                for name, meta in e_h.items():
                                    if name == "__metadata__": continue
                                    exp_f.seek(e_h_offset + meta["data_offsets"][0])
                                    existing_tensors[name] = (meta["shape"], meta["dtype"], exp_f.read(meta["data_offsets"][1] - meta["data_offsets"][0]))
                        
                        if proj_type == "down_proj":
                            # Quantize down_proj matrix
                            quantized_bytes = quantize_q4_0(expert_slice)
                            # Shape: [2048, 512] -> transposed/written as is
                            existing_tensors["down_proj.weight"] = ([shape[1], shape[2]], "Q4_0", quantized_bytes)
                            
                        elif proj_type == "gate_up_proj":
                            # gate_up_proj: [1024, 2048]. Slice into gate_proj and up_proj
                            # gate_proj: first 512 rows; up_proj: second 512 rows
                            split_dim = shape[1] // 2 # 512
                            gate_slice = expert_slice[:split_dim, :]
                            up_slice = expert_slice[split_dim:, :]
                            
                            # Quantize both slices
                            gate_bytes = quantize_q4_0(gate_slice)
                            up_bytes = quantize_q4_0(up_slice)
                            
                            existing_tensors["gate_proj.weight"] = ([split_dim, shape[2]], "Q4_0", gate_bytes)
                            existing_tensors["up_proj.weight"] = ([split_dim, shape[2]], "Q4_0", up_bytes)
                        
                        # Re-write the updated expert file
                        write_safetensors(exp_file_path, existing_tensors)
                        
                else:
                    # 4. Non-expert tensors (base layers, embeddings, norms, shared expert)
                    # Keep shared experts in the base model (always on)
                    tensor_shape = shape
                    
                    if len(tensor_shape) > 1:
                        # 2D base weight matrices -> Q8_0
                        print(f"  [Quantizing Base] {clean_name} (FP32 -> Q8_0)...")
                        quantized_bytes = quantize_q8_0(f32_arr)
                        base_tensors[clean_name] = (tensor_shape, "Q8_0", quantized_bytes)
                    else:
                        # 1D norms, scales, biases -> FP32
                        base_tensors[clean_name] = (tensor_shape, "F32", f32_arr.tobytes())
            
            mmapped_file.close()

    # Write the base layers to base.safetensors
    print("\nWriting base.safetensors...")
    base_out = os.path.join(out_dir, "base.safetensors")
    write_safetensors(base_out, base_tensors)
    print(f"[Done] Base layers written successfully to: {base_out}")
    print(f"[Success] Sharding and Quantization complete!")

if __name__ == "__main__":
    shard_moe()
