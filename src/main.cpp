#include "model.h"
#include "tokenizer.h"
#include "models/deepseek_v4.h"
#include "models/glm_5_2.h"
#include "models/kimi_k3.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <memory>

int main(int argc, char* argv[]) {
    std::cout << "==================================================" << std::endl;
    std::cout << "          HydraEngine C++ Inference Diagnostics" << std::endl;
    std::cout << "==================================================" << std::endl;

    std::string model_dir = "D:/deepseek_sharded";
    if (argc > 1) {
        model_dir = argv[1];
    }

    std::string prompt = "Hello";
    if (argc > 2) {
        prompt = argv[2];
    }

    std::string model_type = "deepseek";
    if (argc > 3) {
        model_type = argv[3];
    }

    // 1. Configure ModelConfig to match DeepSeek-V4-Flash exactly
    ModelConfig cfg;
    cfg.num_layers = 43;
    cfg.num_heads = 64;
    cfg.head_dim = 512;
    cfg.hidden_dim = 4096;
    cfg.ffn_hidden_dim = 2048;
    cfg.vocab_size = 129280;
    cfg.num_experts = 256;
    cfg.num_active_experts = 6;
    cfg.norm_epsilon = 1e-6f;
    cfg.rope_theta = 10000.0f;
    cfg.max_seq_len = 4096;

    std::cout << "Initializing HydraEngine MoE Model..." << std::endl;
    std::cout << "- Model Directory: " << model_dir << std::endl;
    std::cout << "- Model Type: " << model_type << std::endl;
    std::cout << "- Layers: " << cfg.num_layers << std::endl;
    std::cout << "- Heads: " << cfg.num_heads << " (dim " << cfg.head_dim << ")" << std::endl;
    std::cout << "- Routed Experts per layer: " << cfg.num_experts << " (active " << cfg.num_active_experts << ")" << std::endl;
    std::cout << "- Shared Experts per layer: 1" << std::endl;
    std::cout << "- Vocab Size: " << cfg.vocab_size << std::endl << std::endl;

    std::unique_ptr<MoEModel> model;
    if (model_type == "deepseek") {
        model = std::make_unique<DeepSeekV4Model>(cfg, model_dir);
    } else if (model_type == "glm") {
        model = std::make_unique<GLM52Model>(cfg, model_dir);
    } else if (model_type == "kimi") {
        model = std::make_unique<KimiK3Model>(cfg, model_dir);
    } else {
        std::cerr << "[Error] Unknown model type: " << model_type << std::endl;
        return 1;
    }

    // 2. Load the base safetensors
    if (!model->load_base_model()) {
        std::cerr << "[Error] Failed to load base weights" << std::endl;
        return 1;
    }

    // 3. Load the Tokenizer
    std::cout << "\nLoading Tokenizer..." << std::endl;
    Tokenizer tokenizer;
    std::string vocab_path = model_dir + "/vocab.txt";
    if (!tokenizer.load(vocab_path)) {
        std::cerr << "[Error] Failed to load vocabulary file: " << vocab_path << std::endl;
        return 1;
    }

    // 4. Prepare inference context
    std::cout << "\nAllocating Inference Context..." << std::endl;
    InferenceContext ctx;
    ctx.resize(cfg);

    std::cout << "\nEncoding Prompt: \"" << prompt << "\"" << std::endl;
    std::vector<int> tokens = tokenizer.encode(prompt);
    std::cout << "Tokens: ";
    for (int t : tokens) {
        std::cout << t << " ";
    }
    std::cout << std::endl;

    // 6. Run generation loop
    std::cout << "\nRunning Forward Pass..." << std::endl;
    std::cout << "==================================================" << std::endl;

    int pos = 0;
    // Process prompt tokens
    for (; pos < static_cast<int>(tokens.size()); ++pos) {
        int token_id = tokens[pos];
        std::cout << "Processing token pos " << pos << " (ID: " << token_id << ")" << std::endl;
        model->forward(token_id, pos, ctx);
    }

    // Generate the next token
    int next_token_id = 0;
    float max_logit = ctx.logits[0];
    for (int i = 1; i < cfg.vocab_size; ++i) {
        if (ctx.logits[i] > max_logit) {
            max_logit = ctx.logits[i];
            next_token_id = i;
        }
    }

    std::cout << "\nPredicted Next Token ID: " << next_token_id << std::endl;
    std::string next_word = tokenizer.decode(next_token_id);
    std::cout << "Decoded Output Word: \"" << next_word << "\"" << std::endl;
    std::cout << "==================================================" << std::endl;

    return 0;
}
