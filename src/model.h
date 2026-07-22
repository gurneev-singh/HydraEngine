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
    int num_layers = 40;
    int num_heads = 16;
    int head_dim = 256;       // Gated attention head dimension
    int hidden_dim = 2048;    // Base hidden dimension
    int ffn_hidden_dim = 512; // Expert intermediate dimension
    int vocab_size = 248320;  // Vocabulary size
    int num_experts = 256;    // 256 total experts
    int num_active_experts = 8; // 8 active experts per token
    float norm_epsilon = 1e-6f;
    float rope_theta = 1000000.0f;
    int max_seq_len = 2048;
};

// Scratchpad context buffers to prevent dynamic memory allocation during token inference loops
struct InferenceContext {
    std::vector<float> hidden_state;        // Current hidden state [hidden_dim]
    std::vector<float> norm_buffer;          // Buffer for RMSNorm output [hidden_dim]
    
    // Self-Attention buffers
    std::vector<float> q;                   // Query vector [hidden_dim]
    std::vector<float> k;                   // Key vector [hidden_dim]
    std::vector<float> v;                   // Value vector [hidden_dim]
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
    // Layout: [num_layers][max_seq_len * hidden_dim]
    std::vector<std::vector<float>> k_cache;
    std::vector<std::vector<float>> v_cache;
    
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
    
    // Base weight file handler
    std::shared_ptr<SafetensorsFile> base_file;
    
    // Expert cache pager
    std::shared_ptr<ExpertCache> expert_cache;
    
    // ----------------------------------------------------
    // Permanent Base Tensors (loaded in memory)
    // ----------------------------------------------------
    std::shared_ptr<Tensor> token_embeddings; // [vocab_size, hidden_dim]
    std::shared_ptr<Tensor> output_norm;      // [hidden_dim]
    std::shared_ptr<Tensor> lm_head;          // [vocab_size, hidden_dim]
    
    // Per-layer base tensors
    struct LayerBaseWeights {
        std::shared_ptr<Tensor> input_norm;             // [hidden_dim]
        std::shared_ptr<Tensor> post_attention_norm;    // [hidden_dim]
        
        // Attention weights
        std::shared_ptr<Tensor> q_proj;                 // [hidden_dim, hidden_dim]
        std::shared_ptr<Tensor> k_proj;                 // [hidden_dim, hidden_dim]
        std::shared_ptr<Tensor> v_proj;                 // [hidden_dim, hidden_dim]
        std::shared_ptr<Tensor> o_proj;                 // [hidden_dim, hidden_dim]
        
        // Expert Router gate weight
        std::shared_ptr<Tensor> router_gate;            // [num_experts, hidden_dim]
    };
    
    std::vector<LayerBaseWeights> layers;
};

#endif // MODEL_H
