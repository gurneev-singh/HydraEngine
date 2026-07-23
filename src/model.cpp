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
    hidden_state.resize(cfg.hidden_dim);
    norm_buffer.resize(cfg.hidden_dim);
    
    q_latent.resize(cfg.q_lora_rank);
    kv_latent.resize(cfg.kv_lora_rank);
    q.resize(cfg.num_heads * cfg.head_dim);
    k.resize(cfg.num_heads * cfg.head_dim);
    v.resize(cfg.num_heads * cfg.head_dim);
    
    attn_scores.resize(cfg.num_heads * cfg.max_seq_len);
    attn_out.resize(cfg.num_heads * cfg.head_dim);
    
    router_logits.resize(cfg.num_experts);
    expert_in.resize(cfg.hidden_dim);
    
    expert_gate_out.resize(cfg.ffn_hidden_dim);
    expert_up_out.resize(cfg.ffn_hidden_dim);
    expert_down_out.resize(cfg.hidden_dim);
    layer_mlp_out.resize(cfg.hidden_dim);
    
    logits.resize(cfg.vocab_size);
    
    // Allocate space for MLA KV cache: 512 latent dimensions + 64 RoPE dimensions = 576 per token.
    kv_cache.resize(cfg.num_layers, std::vector<float>(cfg.max_seq_len * (cfg.kv_lora_rank + cfg.rope_head_dim), 0.0f));
}

// InferenceContext resizing is complete. subclass models will implement MoEModel.
