#ifndef MODEL_H
#define MODEL_H

#include "tensor.h"
#include "cache.h"
#include <vector>
#include <string>
#include <memory>

struct ModelConfig {
    // GLM-5.2 model parameters placeholder
    int num_layers = 0;
    int num_heads = 0;
    int head_dim = 0;
    int hidden_dim = 0;
    int ffn_hidden_dim = 0;
    int vocab_size = 0;
    int num_experts = 0;
    int num_active_experts = 0;
    float norm_epsilon = 1e-5f;
    float rope_theta = 10000.0f;
    int max_seq_len = 2048;
};

struct InferenceContext {
    std::vector<float> hidden_state;
    std::vector<float> norm_buffer;
    std::vector<float> logits;
    
    void resize(const ModelConfig& cfg);
};

class MoEModel {
public:
    ModelConfig config;

    MoEModel(const ModelConfig& cfg, const std::string& model_dir);
    ~MoEModel() = default;

    bool load_base_model();
    void forward(int token_id, int pos, InferenceContext& ctx);

private:
    std::string model_directory;
};

#endif // MODEL_H
