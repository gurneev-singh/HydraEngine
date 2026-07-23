#ifndef DEEPSEEK_V4_H
#define DEEPSEEK_V4_H

#include "../model.h"
#include "../tensor.h"
#include "../cache.h"
#include <string>
#include <vector>
#include <memory>

class DeepSeekV4Model : public MoEModel {
public:
    ModelConfig config;
    
    DeepSeekV4Model(const ModelConfig& cfg, const std::string& model_dir);
    ~DeepSeekV4Model() override = default;
    
    bool load_base_model() override;
    void forward(int token_id, int pos, InferenceContext& ctx) override;

private:
    std::string model_directory;
    
    std::shared_ptr<SafetensorsFile> meta_file;
    std::shared_ptr<ExpertCache> expert_cache;
    
    std::shared_ptr<Tensor> token_embeddings;
    std::shared_ptr<Tensor> output_norm;
    std::shared_ptr<Tensor> lm_head;
    
    struct LayerBaseWeights {
        std::shared_ptr<SafetensorsFile> layer_file;

        std::shared_ptr<Tensor> input_norm;
        std::shared_ptr<Tensor> post_attention_norm;
        std::shared_ptr<Tensor> q_norm;
        std::shared_ptr<Tensor> kv_norm;
        
        std::shared_ptr<Tensor> wq_a;
        std::shared_ptr<Tensor> wq_b;
        std::shared_ptr<Tensor> wkv;
        std::shared_ptr<Tensor> wo_a;
        std::shared_ptr<Tensor> wo_b;
        
        std::shared_ptr<Tensor> shared_gate;
        std::shared_ptr<Tensor> shared_up;
        std::shared_ptr<Tensor> shared_down;

        std::shared_ptr<Tensor> router_gate;
    };
    
    std::vector<LayerBaseWeights> layers;
};

#endif // DEEPSEEK_V4_H
