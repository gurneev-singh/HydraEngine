#include "model.h"
#include "ops.h"
#include <iostream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cstring>

// ----------------------------------------------------
// InferenceContext Implementation
// ----------------------------------------------------
void InferenceContext::resize(const ModelConfig& cfg) {
    hidden_state.resize(cfg.num_heads * cfg.head_dim);
    norm_buffer.resize(cfg.num_heads * cfg.head_dim);
    
    q.resize(cfg.num_heads * cfg.head_dim);
    k.resize(cfg.num_heads * cfg.head_dim);
    v.resize(cfg.num_heads * cfg.head_dim);
    
    attn_scores.resize(cfg.num_heads * cfg.max_seq_len);
    attn_out.resize(cfg.num_heads * cfg.head_dim);
    
    router_logits.resize(cfg.num_experts);
    expert_in.resize(cfg.num_heads * cfg.head_dim);
    
    expert_gate_out.resize(cfg.ffn_hidden_dim);
    expert_up_out.resize(cfg.ffn_hidden_dim);
    expert_down_out.resize(cfg.num_heads * cfg.head_dim);
    layer_mlp_out.resize(cfg.num_heads * cfg.head_dim);
    
    logits.resize(cfg.vocab_size);
    
    // Allocate space for KV cache
    k_cache.resize(cfg.num_layers, std::vector<float>(cfg.max_seq_len * cfg.num_heads * cfg.head_dim, 0.0f));
    v_cache.resize(cfg.num_layers, std::vector<float>(cfg.max_seq_len * cfg.num_heads * cfg.head_dim, 0.0f));
}

// ----------------------------------------------------
// MoEModel Implementation
// ----------------------------------------------------
MoEModel::MoEModel(const ModelConfig& cfg, const std::string& model_dir) 
    : config(cfg), model_directory(model_dir) {
    
    // Instantiate our custom paging expert cache manager
    std::string experts_path = model_directory + "/experts";
    expert_cache = std::make_shared<ExpertCache>(experts_path, 128); // Increased capacity to 128 for real model
}

bool MoEModel::load_base_model() {
    std::string base_path = model_directory + "/base.safetensors";
    base_file = std::make_shared<SafetensorsFile>(base_path);
    
    if (!base_file->parse_header()) {
        std::cerr << "[Error] Failed to load base weights from: " << base_path << std::endl;
        return false;
    }
    
    std::cout << "[Model Loading] Mapping permanent base tensors..." << std::endl;
    
    // Map permanent vocabulary embeddings and output weights
    token_embeddings = base_file->map_tensor("model.embed_tokens.weight");
    output_norm = base_file->map_tensor("model.norm.weight");
    lm_head = base_file->map_tensor("lm_head.weight");
    
    // Pre-allocate and map per-layer base weights
    layers.resize(config.num_layers);
    for (int l = 0; l < config.num_layers; ++l) {
        std::stringstream ss_in, ss_post, ss_q, ss_k, ss_v, ss_o, ss_gate;
        
        ss_in << "model.layers." << l << ".input_layernorm.weight";
        ss_post << "model.layers." << l << ".post_attention_layernorm.weight";
        ss_q << "model.layers." << l << ".self_attn.q_proj.weight";
        ss_k << "model.layers." << l << ".self_attn.k_proj.weight";
        ss_v << "model.layers." << l << ".self_attn.v_proj.weight";
        ss_o << "model.layers." << l << ".self_attn.o_proj.weight";
        ss_gate << "model.layers." << l << ".mlp.gate.weight"; // Router gate
        
        layers[l].input_norm = base_file->map_tensor(ss_in.str());
        layers[l].post_attention_norm = base_file->map_tensor(ss_post.str());
        
        layers[l].q_proj = base_file->map_tensor(ss_q.str());
        layers[l].k_proj = base_file->map_tensor(ss_k.str());
        layers[l].v_proj = base_file->map_tensor(ss_v.str());
        layers[l].o_proj = base_file->map_tensor(ss_o.str());
        
        layers[l].router_gate = base_file->map_tensor(ss_gate.str());
    }
    
    std::cout << "[Model Loading] Base model tensors mapped successfully." << std::endl;
    return true;
}

void MoEModel::forward(int token_id, int pos, InferenceContext& ctx) {
    int hidden_dim = config.num_heads * config.head_dim;
    
    // 1. Embedding lookup
    // copy row from embedding table: token_embeddings[token_id, :] -> ctx.hidden_state
    const float* embed_data = static_cast<const float*>(token_embeddings->data);
    const float* token_vector = embed_data + (token_id * hidden_dim);
    std::memcpy(ctx.hidden_state.data(), token_vector, hidden_dim * sizeof(float));
    
    // 2. Loop over layers
    for (int l = 0; l < config.num_layers; ++l) {
        const auto& layer_w = layers[l];
        
        // ----------------------------------------------------
        // A. Self-Attention Block (Residual Connection)
        // ----------------------------------------------------
        // Normalize the state
        ops::rms_norm(ctx.norm_buffer.data(), ctx.hidden_state.data(), 
                      static_cast<const float*>(layer_w.input_norm->data), 
                      hidden_dim, config.norm_epsilon);
                      
        // Project Query, Key, and Value vectors
        ops::matmul(ctx.q.data(), ctx.norm_buffer.data(), *layer_w.q_proj);
        ops::matmul(ctx.k.data(), ctx.norm_buffer.data(), *layer_w.k_proj);
        ops::matmul(ctx.v.data(), ctx.norm_buffer.data(), *layer_w.v_proj);
        
        // Apply Rotary Position Embeddings (RoPE)
        ops::rope(ctx.q.data(), config.num_heads, config.head_dim, pos, config.rope_theta);
        ops::rope(ctx.k.data(), config.num_heads, config.head_dim, pos, config.rope_theta);
        
        // Store Key-Value states in Cache for future token lookups
        float* layer_k_cache = ctx.k_cache[l].data() + (pos * hidden_dim);
        float* layer_v_cache = ctx.v_cache[l].data() + (pos * hidden_dim);
        std::memcpy(layer_k_cache, ctx.k.data(), hidden_dim * sizeof(float));
        std::memcpy(layer_v_cache, ctx.v.data(), hidden_dim * sizeof(float));
        
        // Clear attention output buffer
        std::fill(ctx.attn_out.begin(), ctx.attn_out.end(), 0.0f);
        
        // Compute scaled attention scores and weights
        for (int h = 0; h < config.num_heads; ++h) {
            float* head_scores = ctx.attn_scores.data() + (h * config.max_seq_len);
            float scale_factor = 1.0f / std::sqrt(static_cast<float>(config.head_dim));
            const float* head_q = ctx.q.data() + (h * config.head_dim);
            
            // Dot product between Q and past Keys
            for (int p = 0; p <= pos; ++p) {
                const float* cached_k = ctx.k_cache[l].data() + (p * hidden_dim) + (h * config.head_dim);
                float score = 0.0f;
                for (int d = 0; d < config.head_dim; ++d) {
                    score += head_q[d] * cached_k[d];
                }
                head_scores[p] = score * scale_factor;
            }
            
            // Softmax over valid past token positions
            ops::softmax(head_scores, pos + 1);
            
            // Multiply attention weights by Value vectors
            float* head_attn_out = ctx.attn_out.data() + (h * config.head_dim);
            for (int p = 0; p <= pos; ++p) {
                float weight = head_scores[p];
                const float* cached_v = ctx.v_cache[l].data() + (p * hidden_dim) + (h * config.head_dim);
                for (int d = 0; d < config.head_dim; ++d) {
                    head_attn_out[d] += weight * cached_v[d];
                }
            }
        }
        
        // Project Attention output back to hidden dimension and add residual
        // hidden_state = hidden_state + (attn_out * o_proj)
        ops::matmul(ctx.norm_buffer.data(), ctx.attn_out.data(), *layer_w.o_proj);
        ops::vec_add(ctx.hidden_state.data(), ctx.norm_buffer.data(), hidden_dim);
        
        // ----------------------------------------------------
        // B. MoE MLP Block (Residual Connection)
        // ----------------------------------------------------
        // Normalize state before MLP FFN
        ops::rms_norm(ctx.norm_buffer.data(), ctx.hidden_state.data(), 
                      static_cast<const float*>(layer_w.post_attention_norm->data), 
                      hidden_dim, config.norm_epsilon);
                      
        // Get logits from Router Gate: router_logits = norm_buffer * router_gate
        ops::matmul(ctx.router_logits.data(), ctx.norm_buffer.data(), *layer_w.router_gate);
        
        // Convert router logits to probabilities
        ops::softmax(ctx.router_logits.data(), config.num_experts);
        
        // Select Top-K (config.num_active_experts) experts
        std::vector<std::pair<float, int>> expert_scores;
        for (int e = 0; e < config.num_experts; ++e) {
            expert_scores.push_back({ctx.router_logits[e], e});
        }
        std::sort(expert_scores.rbegin(), expert_scores.rend()); // Sort descending
        
        // Re-normalize active expert probabilities to sum to 1.0
        float sum_probs = 0.0f;
        for (int k = 0; k < config.num_active_experts; ++k) {
            sum_probs += expert_scores[k].first;
        }
        
        // Clear layer FFN output accumulator
        std::fill(ctx.layer_mlp_out.begin(), ctx.layer_mlp_out.end(), 0.0f);
        
        // Run active experts
        for (int k = 0; k < config.num_active_experts; ++k) {
            float route_prob = expert_scores[k].first / sum_probs;
            int expert_id = expert_scores[k].second;
            
            // Retrieve/page expert via LRU cache
            auto expert = expert_cache->get_expert(l, expert_id);
            if (!expert) continue;
            
            // 1. Gate projection: gate_out = norm_buffer * gate_proj
            if (expert->gate_proj) {
                ops::matmul(ctx.expert_gate_out.data(), ctx.norm_buffer.data(), *expert->gate_proj);
                ops::silu(ctx.expert_gate_out.data(), ctx.expert_gate_out.data(), config.ffn_hidden_dim);
            } else {
                // If model has no gate layer, default to zeros
                std::fill(ctx.expert_gate_out.begin(), ctx.expert_gate_out.end(), 1.0f);
            }
            
            // 2. Up projection: up_out = norm_buffer * up_proj
            ops::matmul(ctx.expert_up_out.data(), ctx.norm_buffer.data(), *expert->up_proj);
            
            // 3. Activation gating: gate_out = gate_out * up_out
            ops::vec_mul(ctx.expert_gate_out.data(), ctx.expert_up_out.data(), config.ffn_hidden_dim);
            
            // 4. Down projection: down_out = gate_out * down_proj
            ops::matmul(ctx.expert_down_out.data(), ctx.expert_gate_out.data(), *expert->down_proj);
            
            // 5. Accumulate scaled expert output: layer_mlp_out += down_out * route_prob
            for (int d = 0; d < hidden_dim; ++d) {
                ctx.layer_mlp_out[d] += ctx.expert_down_out[d] * route_prob;
            }
        }
        
        // Add MLP output back to hidden state (residual connection)
        ops::vec_add(ctx.hidden_state.data(), ctx.layer_mlp_out.data(), hidden_dim);
    }
    
    // 3. Final Output Projections
    // Normalize final state
    ops::rms_norm(ctx.norm_buffer.data(), ctx.hidden_state.data(), 
                  static_cast<const float*>(output_norm->data), 
                  hidden_dim, config.norm_epsilon);
                  
    // Calculate vocabulary logits: logits = norm_buffer * lm_head
    ops::matmul(ctx.logits.data(), ctx.norm_buffer.data(), *lm_head);
}
