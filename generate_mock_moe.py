import os
import json
import struct

def generate_mock_moe(output_dir):
    print("==================================================")
    print(" Generating Full Mock MoE Model (safetensors format)")
    print("==================================================")
    
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "model-00001-of-00001.safetensors")
    
    vocab_size = 1000
    hidden_dim = 64
    ffn_hidden_dim = 128
    num_experts = 2
    
    # Define all the tensors the C++ transformer forward pass expects
    tensors_to_create = {
        # Permanent base layers
        "model.embed_tokens.weight": [vocab_size, hidden_dim],
        "model.norm.weight": [hidden_dim],
        "lm_head.weight": [vocab_size, hidden_dim],
        
        # Layer 0 Attention & Router
        "model.layers.0.input_layernorm.weight": [hidden_dim],
        "model.layers.0.post_attention_layernorm.weight": [hidden_dim],
        "model.layers.0.self_attn.q_proj.weight": [hidden_dim, hidden_dim],
        "model.layers.0.self_attn.k_proj.weight": [hidden_dim, hidden_dim],
        "model.layers.0.self_attn.v_proj.weight": [hidden_dim, hidden_dim],
        "model.layers.0.self_attn.o_proj.weight": [hidden_dim, hidden_dim],
        "model.layers.0.mlp.gate.weight": [num_experts, hidden_dim],
        
        # Layer 1 Attention & Router
        "model.layers.1.input_layernorm.weight": [hidden_dim],
        "model.layers.1.post_attention_layernorm.weight": [hidden_dim],
        "model.layers.1.self_attn.q_proj.weight": [hidden_dim, hidden_dim],
        "model.layers.1.self_attn.k_proj.weight": [hidden_dim, hidden_dim],
        "model.layers.1.self_attn.v_proj.weight": [hidden_dim, hidden_dim],
        "model.layers.1.self_attn.o_proj.weight": [hidden_dim, hidden_dim],
        "model.layers.1.mlp.gate.weight": [num_experts, hidden_dim],
        
        # Expert layers (Layer 0, Experts 0 and 1)
        "model.layers.0.mlp.experts.0.up_proj.weight": [ffn_hidden_dim, hidden_dim],
        "model.layers.0.mlp.experts.0.down_proj.weight": [hidden_dim, ffn_hidden_dim],
        "model.layers.0.mlp.experts.0.gate_proj.weight": [ffn_hidden_dim, hidden_dim],
        "model.layers.0.mlp.experts.1.up_proj.weight": [ffn_hidden_dim, hidden_dim],
        "model.layers.0.mlp.experts.1.down_proj.weight": [hidden_dim, ffn_hidden_dim],
        "model.layers.0.mlp.experts.1.gate_proj.weight": [ffn_hidden_dim, hidden_dim],
        
        # Expert layers (Layer 1, Experts 0 and 1)
        "model.layers.1.mlp.experts.0.up_proj.weight": [ffn_hidden_dim, hidden_dim],
        "model.layers.1.mlp.experts.0.down_proj.weight": [hidden_dim, ffn_hidden_dim],
        "model.layers.1.mlp.experts.0.gate_proj.weight": [ffn_hidden_dim, hidden_dim],
        "model.layers.1.mlp.experts.1.up_proj.weight": [ffn_hidden_dim, hidden_dim],
        "model.layers.1.mlp.experts.1.down_proj.weight": [hidden_dim, ffn_hidden_dim],
        "model.layers.1.mlp.experts.1.gate_proj.weight": [ffn_hidden_dim, hidden_dim]
    }
    
    # Calculate bytes and construct offsets
    raw_data = bytearray()
    header = {}
    current_offset = 0
    
    for name, shape in tensors_to_create.items():
        # Create dummy float32 data (all 1.0f)
        num_elements = shape[0]
        if len(shape) > 1:
            num_elements *= shape[1]
            
        tensor_bytes = struct.pack(f"<{num_elements}f", *[0.1] * num_elements)
        
        start = current_offset
        end = start + len(tensor_bytes)
        
        header[name] = {
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [start, end]
        }
        
        raw_data.extend(tensor_bytes)
        current_offset = end
        
    # Add metadata
    header["__metadata__"] = {"format": "pt"}
    
    # Serialize header to JSON
    header_json = json.dumps(header, separators=(',', ':')).encode('utf-8')
    
    # Calculate padding to ensure binary data starts at a multiple of 8 bytes
    header_len = len(header_json)
    padding = (8 - ((8 + header_len) % 8)) % 8
    header_json += b' ' * padding
    
    # Write the safetensors file
    with open(file_path, "wb") as f:
        # Write 8-byte header size
        f.write(struct.pack("<Q", len(header_json)))
        # Write header
        f.write(header_json)
        # Write raw tensor bytes
        f.write(raw_data)
        
    print(f"[Success] Mock MoE model generated at: {file_path}")
    print(f"- Total tensors: {len(tensors_to_create)}")
    print(f"- File size: {os.path.getsize(file_path) / (1024):.2f} KB")

if __name__ == "__main__":
    # Output to raw_model directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "raw_model")
    generate_mock_moe(output_dir)
