#ifndef GLM_5_2_H
#define GLM_5_2_H

#include "../model.h"
#include <string>

class GLM52Model : public MoEModel {
public:
    ModelConfig config;

    GLM52Model(const ModelConfig& cfg, const std::string& model_dir);
    ~GLM52Model() override = default;

    bool load_base_model() override;
    void forward(int token_id, int pos, InferenceContext& ctx) override;

private:
    std::string model_directory;
};

#endif // GLM_5_2_H
