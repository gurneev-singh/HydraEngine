#ifndef KIMI_K3_H
#define KIMI_K3_H

#include "../model.h"
#include <string>

class KimiK3Model : public MoEModel {
public:
    ModelConfig config;

    KimiK3Model(const ModelConfig& cfg, const std::string& model_dir);
    ~KimiK3Model() override = default;

    bool load_base_model() override;
    void forward(int token_id, int pos, InferenceContext& ctx) override;

private:
    std::string model_directory;
};

#endif // KIMI_K3_H
