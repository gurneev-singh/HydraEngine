import os
import json
import struct
import numpy as np
import torch

D_DIR = "D:/deepseek_raw"
file_path = os.path.join(D_DIR, "model-00002-of-00046.safetensors")

print("==================================================")
print(" DeepSeek V4 Dequantization Diagnostic")
print("==================================================")

with open(file_path, "rb") as f:
    header_size = struct.unpack("<Q", f.read(8))[0]
    header_json = f.read(header_size).decode('utf-8')
    header = json.loads(header_json)
    
    # 1. Inspect Expert 0, W1
    t_name_w = "layers.0.ffn.experts.0.w1.weight"
    t_name_s = "layers.0.ffn.experts.0.w1.scale"
    
    offset = 8 + header_size
    
    if t_name_w in header and t_name_s in header:
        meta_w = header[t_name_w]
        meta_s = header[t_name_s]
        
        f.seek(offset + meta_s["data_offsets"][0])
        scale_bytes = f.read(meta_s["data_offsets"][1] - meta_s["data_offsets"][0])
        
        f.seek(offset + meta_w["data_offsets"][0])
        weight_bytes = f.read(meta_w["data_offsets"][1] - meta_w["data_offsets"][0])
        
        print("Successfully read expert weight and scale bytes.")
        
        scale_uint8 = np.frombuffer(scale_bytes, dtype=np.uint8)
        scale_float = (2.0 ** (scale_uint8.astype(np.float32) - 127.0)).reshape(meta_s["shape"])
        
        weight_int8 = np.frombuffer(weight_bytes, dtype=np.int8).reshape(meta_w["shape"])
        
        rows, cols = meta_w["shape"]
        weight_blocks = weight_int8.reshape(rows, cols // 16, 16).astype(np.float32)
        weight_f32_blocks = weight_blocks * scale_float[:, :, np.newaxis]
        weight_f32 = weight_f32_blocks.reshape(rows, cols)
        
        print(f"Expert Weight Dequantized Shape: {weight_f32.shape}")
        print(f"First few values:\n{weight_f32[:3, :5]}")
        print(f"Expert range: min={np.min(weight_f32)}, max={np.max(weight_f32)}, mean={np.mean(weight_f32)}\n")
    
    # 2. Inspect Attention WQ_A (FP8)
    t_attn_w = "layers.0.attn.wq_a.weight"
    t_attn_s = "layers.0.attn.wq_a.scale"
    
    if t_attn_w in header and t_attn_s in header:
        meta_w = header[t_attn_w]
        meta_s = header[t_attn_s]
        
        f.seek(offset + meta_s["data_offsets"][0])
        scale_bytes = f.read(meta_s["data_offsets"][1] - meta_s["data_offsets"][0])
        
        f.seek(offset + meta_w["data_offsets"][0])
        weight_bytes = f.read(meta_w["data_offsets"][1] - meta_w["data_offsets"][0])
        
        print("Successfully read attention weight and scale bytes.")
        
        # Parse FP8 weight using torch
        # float8_e4m3fn in PyTorch:
        # Since weight_bytes contains 1024 * 4096 = 4,194,304 bytes, we load it as float8_e4m3fn
        t_f8 = torch.frombuffer(weight_bytes, dtype=torch.float8_e4m3fn)
        arr_f32 = t_f8.to(torch.float32).numpy().reshape(meta_w["shape"])
        
        # Parse E8M0 scale
        scale_uint8 = np.frombuffer(scale_bytes, dtype=np.uint8)
        scale_float = (2.0 ** (scale_uint8.astype(np.float32) - 127.0)).reshape(meta_s["shape"])
        
        # Apply block scale of size 128x128
        # weight: [1024, 4096]
        # scale: [8, 32]
        # 1024 / 8 = 128 rows per block, 4096 / 32 = 128 cols per block
        rows, cols = meta_w["shape"]
        r_blocks, c_blocks = meta_s["shape"]
        
        weight_tiles = arr_f32.reshape(r_blocks, 128, c_blocks, 128)
        weight_f32_tiles = weight_tiles * scale_float[:, np.newaxis, :, np.newaxis]
        attn_weight_f32 = weight_f32_tiles.reshape(rows, cols)
        
        print(f"Attention Weight Dequantized Shape: {attn_weight_f32.shape}")
        print(f"First few values:\n{attn_weight_f32[:3, :5]}")
        print(f"Attention range: min={np.min(attn_weight_f32)}, max={np.max(attn_weight_f32)}, mean={np.mean(attn_weight_f32)}")
    else:
        print("Attention tensors not found.")
