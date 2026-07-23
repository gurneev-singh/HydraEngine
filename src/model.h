#ifndef MODEL_H
#define MODEL_H

#include "tensor.h"
#include "cache.h"
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

// Model hyperparameters
struct ModelConfig {
    int num_layers = 43;
    int num_heads = 64;
    int head_dim = 512;       // Query projection head dimension
    int hidden_dim = 4096;    // Base hidden dimension (d_model)
    int ffn_hidden_dim = 2048; // Expert intermediate dimension
    int vocab_size = 129280;  // Vocabulary size
    int num_experts = 256;    // 256 total routed experts
    int num_active_experts = 6; // 6 active experts per token
    int q_lora_rank = 1024;   // Latent query dimension
    int kv_lora_rank = 512;   // Latent KV dimension
    int rope_head_dim = 64;   // RoPE head dimension
    float norm_epsilon = 1e-6f;
    float rope_theta = 10000.0f;
    int max_seq_len = 4096;
};

// Scratchpad context buffers to prevent dynamic memory allocation during token inference loops
struct InferenceContext {
    std::vector<float> hidden_state;        // Current hidden state [hidden_dim]
    std::vector<float> norm_buffer;          // Buffer for RMSNorm output [hidden_dim]
    
    // Self-Attention buffers
    std::vector<float> q_latent;            // Latent query vector [q_lora_rank]
    std::vector<float> kv_latent;           // Latent KV vector [kv_lora_rank]
    std::vector<float> q;                   // Query vector [num_heads * head_dim]
    std::vector<float> k;                   // Key vector [num_heads * head_dim]
    std::vector<float> v;                   // Value vector [num_heads * head_dim]
    std::vector<float> attn_scores;          // Attention weights buffer [num_heads * max_seq_len]
    std::vector<float> attn_out;             // Weighted attention output [hidden_dim]
    
    // Router / Expert gate buffers
    std::vector<float> router_logits;        // Logits for expert selection [num_experts]
    std::vector<float> expert_in;            // Input to active experts [hidden_dim]
    std::vector<float> expert_gate_out;      // Intermediate gate output [ffn_hidden_dim]
    std::vector<float> expert_up_out;        // Intermediate up-projection output [ffn_hidden_dim]
    std::vector<float> expert_down_out;      // Expert final output [hidden_dim]
    std::vector<float> layer_mlp_out;        // Summed FFN outputs [hidden_dim]
    
    // Output logits
    std::vector<float> logits;               // Final vocabulary logits [vocab_size]
    
    // Key-Value Cache buffers (stores K and V states for past positions to save computation)
    // In MLA, we cache the 512-dimensional latent vector and the 64-dimensional RoPE key.
    // So the cache size per layer is max_seq_len * (kv_lora_rank + rope_head_dim) = max_seq_len * 576.
    std::vector<std::vector<float>> kv_cache;
    
    void resize(const ModelConfig& cfg);
};

// Base class representing the Mixture of Experts Model interface
class MoEModel {
public:
    virtual ~MoEModel() = default;
    
    // Loads the base model weights (attention, routers, output embeddings)
    virtual bool load_base_model() = 0;
    
    // Runs a single forward pass of the model for a token at a specific position
    virtual void forward(int token_id, int pos, InferenceContext& ctx) = 0;
};

#endif // MODEL_H
