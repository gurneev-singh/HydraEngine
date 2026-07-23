#ifndef OPS_H
#define OPS_H

#include "tensor.h"
#include <cstdint>

// Custom namespace for low-level mathematical operations
namespace ops {

    // 1. Root Mean Square Normalization (RMSNorm) - Used before self-attention and MLP layers
    void rms_norm(float* out, const float* in, const float* weight, int size, float epsilon);

    // 2. Softmax activation - Used to normalize attention scores and router gate probabilities
    void softmax(float* x, int size);

    // 3. Sigmoid Linear Unit (SiLU) - The activation function used inside MoE expert FFN layers
    void silu(float* out, const float* in, int size);

    // 4. Element-wise multiplication and accumulation (e.g., residual connections)
    void vec_add(float* out, const float* in, int size);
    void vec_mul(float* out, const float* in, int size);

    // 5. Rotary Position Embeddings (RoPE) - Applies positional encoding to query (Q) and key (K) vectors
    void rope(float* vec, int num_heads, int head_dim, int position, float theta = 10000.0f);

    // 6. Matrix Multiplication (GEMM) - The core compute bottleneck of the model
    // Multiplies vector x (1 x cols) by matrix w (rows x cols) to output vector (1 x rows)
    // Supports on-the-fly dequantization if weights are stored in quantized format
    void matmul(float* out, const float* x, const Tensor& w);

} // namespace ops

#endif // OPS_H
