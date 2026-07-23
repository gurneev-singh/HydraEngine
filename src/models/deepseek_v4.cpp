#include "deepseek_v4.h"
#include "../ops.h"
#include <iostream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cstring>

DeepSeekV4Model::DeepSeekV4Model(const ModelConfig& cfg, const std::string& model_dir) 
    : config(cfg), model_directory(model_dir) {
    
    std::string experts_path = model_directory + "/experts";
    expert_cache = std::make_shared<ExpertCache>(experts_path, 128);
}

bool DeepSeekV4Model::load_base_model() {
    std::string meta_path = model_directory + "/base_meta.safetensors";
    meta_file = std::make_shared<SafetensorsFile>(meta_path);
    
    if (!meta_file->parse_header()) {
        std::cerr << "[Error] Failed to load metadata weights from: " << meta_path << std::endl;
        return false;
    }
    
    std::cout << "[Model Loading] Mapping permanent base metadata tensors..." << std::endl;
    
    token_embeddings = meta_file->map_tensor("embed.weight");
    output_norm = meta_file->map_tensor("norm.weight");
    lm_head = meta_file->map_tensor("lm_head.weight");
    
    layers.resize(config.num_layers);
    for (int l = 0; l < config.num_layers; ++l) {
        std::stringstream ss_path;
        ss_path << model_directory << "/base_layer_" << l << ".safetensors";
        
        layers[l].layer_file = std::make_shared<SafetensorsFile>(ss_path.str());
        if (!layers[l].layer_file->parse_header()) {
            std::cerr << "[Error] Failed to parse layer " << l << " weights from: " << ss_path.str() << std::endl;
            return false;
        }
        
        layers[l].input_norm = layers[l].layer_file->map_tensor("input_layernorm.weight");
        layers[l].post_attention_norm = layers[l].layer_file->map_tensor("post_attention_layernorm.weight");
        layers[l].q_norm = layers[l].layer_file->map_tensor("self_attn.q_norm.weight");
        layers[l].kv_norm = layers[l].layer_file->map_tensor("self_attn.kv_norm.weight");
        
        layers[l].wq_a = layers[l].layer_file->map_tensor("self_attn.wq_a.weight");
        layers[l].wq_b = layers[l].layer_file->map_tensor("self_attn.wq_b.weight");
        layers[l].wkv = layers[l].layer_file->map_tensor("self_attn.wkv.weight");
        layers[l].wo_a = layers[l].layer_file->map_tensor("self_attn.wo_a.weight");
        layers[l].wo_b = layers[l].layer_file->map_tensor("self_attn.wo_b.weight");
        
        layers[l].shared_gate = layers[l].layer_file->map_tensor("ffn.shared_experts.gate_proj.weight");
        layers[l].shared_up = layers[l].layer_file->map_tensor("ffn.shared_experts.up_proj.weight");
        layers[l].shared_down = layers[l].layer_file->map_tensor("ffn.shared_experts.down_proj.weight");
        
        layers[l].router_gate = layers[l].layer_file->map_tensor("ffn.gate.weight");
    }
    
    std::cout << "[Model Loading] Base model tensors mapped successfully." << std::endl;
    return true;
}

void DeepSeekV4Model::forward(int token_id, int pos, InferenceContext& ctx) {
    int hidden_dim = config.hidden_dim;
    
    const float* embed_data = static_cast<const float*>(token_embeddings->data);
    const float* token_vector = embed_data + (token_id * hidden_dim);
    std::memcpy(ctx.hidden_state.data(), token_vector, hidden_dim * sizeof(float));
    
    for (int l = 0; l < config.num_layers; ++l) {
        const auto& layer_w = layers[l];
        
        ops::rms_norm(ctx.norm_buffer.data(), ctx.hidden_state.data(), 
                      static_cast<const float*>(layer_w.input_norm->data), 
                      config.hidden_dim, config.norm_epsilon);
                      
        ops::matmul(ctx.q_latent.data(), ctx.norm_buffer.data(), *layer_w.wq_a);
        ops::rms_norm(ctx.q_latent.data(), ctx.q_latent.data(), 
                      static_cast<const float*>(layer_w.q_norm->data), 
                      config.q_lora_rank, config.norm_epsilon);
        ops::matmul(ctx.q.data(), ctx.q_latent.data(), *layer_w.wq_b);
        
        ops::matmul(ctx.kv_latent.data(), ctx.norm_buffer.data(), *layer_w.wkv);
        ops::rms_norm(ctx.kv_latent.data(), ctx.kv_latent.data(), 
                      static_cast<const float*>(layer_w.kv_norm->data), 
                      config.kv_lora_rank, config.norm_epsilon);
        
        for (int h = 0; h < config.num_heads; ++h) {
            float* head_q_rope = ctx.q.data() + (h * config.head_dim) + (config.head_dim - config.rope_head_dim);
            ops::rope(head_q_rope, 1, config.rope_head_dim, pos, config.rope_theta);
        }
        
        float* kv_rope_ptr = ctx.kv_latent.data() + (config.head_dim - config.rope_head_dim);
        ops::rope(kv_rope_ptr, 1, config.rope_head_dim, pos, config.rope_theta);
        
        float* layer_cache_ptr = ctx.kv_cache[l].data() + (pos * config.head_dim);
        std::memcpy(layer_cache_ptr, ctx.kv_latent.data(), config.head_dim * sizeof(float));
        
        std::fill(ctx.attn_out.begin(), ctx.attn_out.end(), 0.0f);
        
        float scale_factor = 1.0f / std::sqrt(static_cast<float>(config.head_dim));
        for (int h = 0; h < config.num_heads; ++h) {
            float* head_scores = ctx.attn_scores.data() + (h * config.max_seq_len);
            const float* head_q = ctx.q.data() + (h * config.head_dim);
            
            for (int p = 0; p <= pos; ++p) {
                const float* cached_kv = ctx.kv_cache[l].data() + (p * config.head_dim);
                float score = 0.0f;
                for (int d = 0; d < config.head_dim; ++d) {
                    score += head_q[d] * cached_kv[d];
                }
                head_scores[p] = score * scale_factor;
            }
            
            ops::softmax(head_scores, pos + 1);
            
            float* head_attn_out = ctx.attn_out.data() + (h * config.head_dim);
            for (int p = 0; p <= pos; ++p) {
                float weight = head_scores[p];
                const float* cached_kv = ctx.kv_cache[l].data() + (p * config.head_dim);
                for (int d = 0; d < config.head_dim; ++d) {
                    head_attn_out[d] += weight * cached_kv[d];
                }
            }
        }
        
        std::vector<float> o_latent(8 * 1024, 0.0f);
        for (int g = 0; g < 8; ++g) {
            float* group_in = ctx.attn_out.data() + (g * 8 * config.head_dim);
            float* group_out = o_latent.data() + (g * 1024);
            
            Tensor wo_a_slice = *layer_w.wo_a;
            wo_a_slice.shape = {1024, 8 * config.head_dim};
            wo_a_slice.data = static_cast<char*>(layer_w.wo_a->data) + (g * 1024 * (8 * config.head_dim) * sizeof(float));
            
            ops::matmul(group_out, group_in, wo_a_slice);
        }
        
        ops::matmul(ctx.norm_buffer.data(), o_latent.data(), *layer_w.wo_b);
        ops::vec_add(ctx.hidden_state.data(), ctx.norm_buffer.data(), config.hidden_dim);
        
        ops::rms_norm(ctx.norm_buffer.data(), ctx.hidden_state.data(), 
                      static_cast<const float*>(layer_w.post_attention_norm->data), 
                      config.hidden_dim, config.norm_epsilon);
                      
        ops::matmul(ctx.router_logits.data(), ctx.norm_buffer.data(), *layer_w.router_gate);
        ops::softmax(ctx.router_logits.data(), config.num_experts);
        
        std::vector<std::pair<float, int>> expert_scores;
        for (int e = 0; e < config.num_experts; ++e) {
            expert_scores.push_back({ctx.router_logits[e], e});
        }
        std::sort(expert_scores.rbegin(), expert_scores.rend());
        
        float sum_probs = 0.0f;
        for (int k = 0; k < config.num_active_experts; ++k) {
            sum_probs += expert_scores[k].first;
        }
        
        std::fill(ctx.layer_mlp_out.begin(), ctx.layer_mlp_out.end(), 0.0f);
        
        for (int k = 0; k < config.num_active_experts; ++k) {
            float route_prob = expert_scores[k].first / sum_probs;
            int expert_id = expert_scores[k].second;
            
            auto expert = expert_cache->get_expert(l, expert_id);
            if (!expert) continue;
            
            ops::matmul(ctx.expert_gate_out.data(), ctx.norm_buffer.data(), *expert->gate_proj);
            ops::silu(ctx.expert_gate_out.data(), ctx.expert_gate_out.data(), config.ffn_hidden_dim);
            
            ops::matmul(ctx.expert_up_out.data(), ctx.norm_buffer.data(), *expert->up_proj);
            ops::vec_mul(ctx.expert_gate_out.data(), ctx.expert_up_out.data(), config.ffn_hidden_dim);
            ops::matmul(ctx.expert_down_out.data(), ctx.expert_gate_out.data(), *expert->down_proj);
            
            for (int d = 0; d < config.hidden_dim; ++d) {
                ctx.layer_mlp_out[d] += ctx.expert_down_out[d] * route_prob;
            }
        }
        
        ops::matmul(ctx.expert_gate_out.data(), ctx.norm_buffer.data(), *layer_w.shared_gate);
        ops::silu(ctx.expert_gate_out.data(), ctx.expert_gate_out.data(), config.ffn_hidden_dim);
        
        ops::matmul(ctx.expert_up_out.data(), ctx.norm_buffer.data(), *layer_w.shared_up);
        ops::vec_mul(ctx.expert_gate_out.data(), ctx.expert_up_out.data(), config.ffn_hidden_dim);
        ops::matmul(ctx.expert_down_out.data(), ctx.expert_gate_out.data(), *layer_w.shared_down);
        
        for (int d = 0; d < config.hidden_dim; ++d) {
            ctx.layer_mlp_out[d] += ctx.expert_down_out[d];
        }
        
        ops::vec_add(ctx.hidden_state.data(), ctx.layer_mlp_out.data(), config.hidden_dim);
    }
    
    ops::rms_norm(ctx.norm_buffer.data(), ctx.hidden_state.data(), 
                  static_cast<const float*>(output_norm->data), 
                  hidden_dim, config.norm_epsilon);
                  
    ops::matmul(ctx.logits.data(), ctx.norm_buffer.data(), *lm_head);
}
