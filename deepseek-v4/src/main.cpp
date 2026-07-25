#include "model.h"
#include "tokenizer.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>

void apply_chat_template_fixes(std::string& prompt) {
    // Replace "<|User|>" with "<｜User｜>"
    size_t pos = 0;
    while ((pos = prompt.find("<|User|>")) != std::string::npos) {
        prompt.replace(pos, 8, "<｜User｜>");
    }
    // Replace "<|Assistant|>" with "<｜Assistant｜>"
    pos = 0;
    while ((pos = prompt.find("<|Assistant|>")) != std::string::npos) {
        prompt.replace(pos, 13, "<｜Assistant｜>");
    }
}

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
    MoEModel model(cfg, model_dir);

    // 2. Load the base safetensors
    if (!model.load_base_model()) {
        std::cerr << "[Error] Failed to load base weights" << std::endl;
        return 1;
    }

    // 3. Load the Tokenizer
    Tokenizer tokenizer;
    std::string vocab_path = model_dir + "/vocab.txt";
    if (!tokenizer.load(vocab_path)) {
        std::cerr << "[Error] Failed to load vocabulary file" << std::endl;
        return 1;
    }

    // 4. Prepare inference context
    InferenceContext ctx;
    ctx.resize(cfg);

    // 5. User prompt and encoding
    std::string prompt = "Hello";
    if (argc > 2) {
        prompt = argv[2];
    }
    
    int max_tokens = 5;
    if (argc > 3) {
        max_tokens = std::stoi(argv[3]);
    }
    
    apply_chat_template_fixes(prompt);
    std::vector<int> tokens = tokenizer.encode(prompt);
    std::cout << "Encoded prompt tokens: ";
    for (int t : tokens) {
        std::cout << t << " (" << tokenizer.decode(t) << ") ";
    }
    std::cout << std::endl;

    // 6. Run generation loop
    std::cout << "\nProcessing Prompt & Generating Output..." << std::endl;
    std::cout << "==================================================" << std::endl;

    auto t_start = std::chrono::high_resolution_clock::now();

    int pos = 0;
    std::vector<int> generated_tokens;

    // Process prompt tokens
    for (; pos < static_cast<int>(tokens.size()); ++pos) {
        int token_id = tokens[pos];
        model.forward(token_id, pos, ctx);
        generated_tokens.push_back(token_id);
    }

    // Print debug logits to check for NaN or uninitialized values
    std::cout << "\n--- Logit Debug ---" << std::endl;
    int nan_count = 0;
    float min_l = ctx.logits[0];
    float max_l = ctx.logits[0];
    for (int i = 0; i < cfg.vocab_size; ++i) {
        if (std::isnan(ctx.logits[i])) {
            nan_count++;
        }
        if (ctx.logits[i] < min_l) min_l = ctx.logits[i];
        if (ctx.logits[i] > max_l) max_l = ctx.logits[i];
    }
    std::cout << "- First 5 logits: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << ctx.logits[i] << " ";
    }
    std::cout << "\n- Min logit: " << min_l << ", Max logit: " << max_l << ", NaN count: " << nan_count << std::endl;
    std::cout << "-------------------\n" << std::endl;

    // Argmax for the very first token (with repetition penalty & BOS suppression)
    int next_token_id = 0;
    float max_logit = -1e9f;
    for (int i = 0; i < cfg.vocab_size; ++i) {
        float logit = ctx.logits[i];
        
        // Suppress BOS token (0) during generation to avoid loop stutters
        if (i == 0) {
            logit = -1e9f;
        }
        
        // Repetition penalty
        for (int prev_id : generated_tokens) {
            if (prev_id == i) {
                logit = (logit > 0) ? (logit / 1.15f) : (logit * 1.15f);
            }
        }
        
        if (logit > max_logit) {
            max_logit = logit;
            next_token_id = i;
        }
    }

    auto t_first_token = std::chrono::high_resolution_clock::now();
    double ttft_ms = std::chrono::duration<double, std::milli>(t_first_token - t_start).count();
    std::cout << "\n[TTFT] Time to First Token: " << ttft_ms << " ms" << std::endl;

    std::cout << "\nOutput: " << std::flush;
    std::string word = tokenizer.decode(next_token_id);
    std::cout << word << std::flush;
    generated_tokens.push_back(next_token_id);

    // Generate remaining tokens
    int tokens_generated = 1;
    auto t_gen_start = std::chrono::high_resolution_clock::now();

    while (tokens_generated < max_tokens) {
        // Forward pass with the newly predicted token
        model.forward(next_token_id, pos, ctx);
        pos++;

        // Argmax with repetition penalty & BOS suppression
        next_token_id = 0;
        max_logit = -1e9f;
        for (int i = 0; i < cfg.vocab_size; ++i) {
            float logit = ctx.logits[i];
            
            // Suppress BOS token
            if (i == 3) {
                logit = -1e9f;
            }
            
            // Repetition penalty
            for (int prev_id : generated_tokens) {
                if (prev_id == i) {
                    logit = (logit > 0) ? (logit / 1.15f) : (logit * 1.15f);
                }
            }
            
            if (logit > max_logit) {
                max_logit = logit;
                next_token_id = i;
            }
        }

        // Check for EOS tokens (typically 1 or 2)
        if (next_token_id == 1 || next_token_id == 2) {
            std::cout << " [EOS]" << std::endl;
            break;
        }

        std::string next_word = tokenizer.decode(next_token_id);
        std::cout << next_word << std::flush;
        generated_tokens.push_back(next_token_id);
        tokens_generated++;
    }
    std::cout << std::endl;

    auto t_gen_end = std::chrono::high_resolution_clock::now();
    double gen_time_sec = std::chrono::duration<double>(t_gen_end - t_gen_start).count();
    double tokens_per_sec = (tokens_generated - 1) / gen_time_sec;

    std::cout << "==================================================" << std::endl;
    std::cout << "[Benchmark Summary]" << std::endl;
    std::cout << "- TTFT (Prompt Processing): " << ttft_ms / 1000.0 << " sec" << std::endl;
    std::cout << "- Generation Speed: " << tokens_per_sec << " tokens/sec" << std::endl;
    std::cout << "- Generated Tokens: " << tokens_generated << std::endl;
    std::cout << "==================================================" << std::endl;

    return 0;
}
