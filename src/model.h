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

// Represents the complete Mixture of Experts Transformer model
class MoEModel {
public:
    ModelConfig config;
    
    MoEModel(const ModelConfig& cfg, const std::string& model_dir);
    ~MoEModel() = default;
    
    // Loads the base model weights (attention, routers, output embeddings)
    bool load_base_model();
    
    // Runs a single forward pass of the model for a token at a specific position
    void forward(int token_id, int pos, InferenceContext& ctx);

private:
    std::string model_directory;
    
    // Base weight file handler for global metadata
    std::shared_ptr<SafetensorsFile> meta_file;
    
    // Expert cache pager
    std::shared_ptr<ExpertCache> expert_cache;
    
    // ----------------------------------------------------
    // Permanent Base Tensors (loaded in memory)
    // ----------------------------------------------------
    std::shared_ptr<Tensor> token_embeddings; // [vocab_size, hidden_dim]
    std::shared_ptr<Tensor> output_norm;      // [hidden_dim]
    std::shared_ptr<Tensor> lm_head;          // [vocab_size, hidden_dim]
    
    // Per-layer base weights
    struct LayerBaseWeights {
        // Layer-specific file handler
        std::shared_ptr<SafetensorsFile> layer_file;

        std::shared_ptr<Tensor> input_norm;             // [hidden_dim]
        std::shared_ptr<Tensor> post_attention_norm;    // [hidden_dim]
        std::shared_ptr<Tensor> q_norm;                 // [q_lora_rank]
        std::shared_ptr<Tensor> kv_norm;                // [kv_lora_rank]
        
        // Attention weights
        std::shared_ptr<Tensor> wq_a;                   // [q_lora_rank, hidden_dim]
        std::shared_ptr<Tensor> wq_b;                   // [num_heads * head_dim, q_lora_rank]
        std::shared_ptr<Tensor> wkv;                    // [kv_lora_rank, hidden_dim]
        std::shared_ptr<Tensor> wo_a;                   // [o_group_dim, o_lora_rank] or similar
        std::shared_ptr<Tensor> wo_b;                   // [hidden_dim, o_group_dim]
        
        // Shared Expert weights
        std::shared_ptr<Tensor> shared_gate;            // [ffn_hidden_dim, hidden_dim]
        std::shared_ptr<Tensor> shared_up;              // [ffn_hidden_dim, hidden_dim]
        std::shared_ptr<Tensor> shared_down;            // [hidden_dim, ffn_hidden_dim]

        // Expert Router gate weight
        std::shared_ptr<Tensor> router_gate;            // [num_experts, hidden_dim]
    };
    
    std::vector<LayerBaseWeights> layers;
};

#endif // MODEL_H
