#include "model.h"
#include "tokenizer.h"
#include <iostream>
#include <vector>
#include <algorithm>

int main(int argc, char* argv[]) {
    std::cout << "==================================================" << std::endl;
    std::cout << "          HydraEngine C++ Inference Diagnostics" << std::endl;
    std::cout << "==================================================" << std::endl;

    std::string model_dir = "D:/deepseek_sharded";
    if (argc > 1) {
        model_dir = argv[1];
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
    std::cout << "- Layers: " << cfg.num_layers << std::endl;
    std::cout << "- Heads: " << cfg.num_heads << " (dim " << cfg.head_dim << ")" << std::endl;
    std::cout << "- Routed Experts per layer: " << cfg.num_experts << " (active " << cfg.num_active_experts << ")" << std::endl;
    std::cout << "- Shared Experts per layer: 1" << std::endl;
    std::cout << "- Vocab Size: " << cfg.vocab_size << std::endl << std::endl;

    MoEModel model(cfg, model_dir);

    // 2. Load the base safetensors
    if (!model.load_base_model()) {
        std::cerr << "[Error] Failed to load base weights from base.safetensors" << std::endl;
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

    // 5. User prompt and encoding
    std::string prompt = "Hello";
    if (argc > 2) {
        prompt = argv[2];
    }
    
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
        model.forward(token_id, pos, ctx);
    }

    // Generate the next token
    // Find the argmax (index of highest logit value)
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
