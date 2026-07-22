#include "ops.h"
#include <cmath>
#include <iostream>
#include <cstring>

// Custom block structures for our quantization formats
struct BlockQ8_0 {
    float d;          // scale factor (delta)
    int8_t qs[32];    // 32 quantized 8-bit integers
};

struct BlockQ4_0 {
    float d;          // scale factor (delta)
    uint8_t qs[16];   // 32 packed 4-bit integers (2 values per byte)
};

struct BlockQ2 {
    float d;          // scale factor (delta)
    float min;        // minimum bias value
    uint8_t qs[8];    // 32 packed 2-bit weights (4 values per byte, 4 * 8 = 32)
};

namespace ops {

    // 1. RMSNorm
    void rms_norm(float* out, const float* in, const float* weight, int size, float epsilon) {
        // Calculate sum of squares
        float sum = 0.0f;
        for (int i = 0; i < size; ++i) {
            sum += in[i] * in[i];
        }
        
        // Calculate root mean square scale factor
        float mean = sum / size;
        float scale = 1.0f / std::sqrt(mean + epsilon);
        
        // Normalize and scale by learnable weight parameter
        for (int i = 0; i < size; ++i) {
            out[i] = in[i] * scale * weight[i];
        }
    }

    // 2. Softmax (numerically stable version)
    void softmax(float* x, int size) {
        // Find maximum value to prevent exponent overflow
        float max_val = x[0];
        for (int i = 1; i < size; ++i) {
            if (x[i] > max_val) {
                max_val = x[i];
            }
        }
        
        // Sum up exponents
        float sum = 0.0f;
        for (int i = 0; i < size; ++i) {
            x[i] = std::exp(x[i] - max_val);
            sum += x[i];
        }
        
        // Divide by sum to get probability distribution
        for (int i = 0; i < size; ++i) {
            x[i] /= sum;
        }
    }

    // 3. SiLU (Sigmoid Linear Unit) activation: x * sigmoid(x)
    void silu(float* out, const float* in, int size) {
        for (int i = 0; i < size; ++i) {
            float x = in[i];
            float sigmoid = 1.0f / (1.0f + std::exp(-x));
            out[i] = x * sigmoid;
        }
    }

    // 4. Vector additions and multiplications
    void vec_add(float* out, const float* in, int size) {
        for (int i = 0; i < size; ++i) {
            out[i] += in[i];
        }
    }

    void vec_mul(float* out, const float* in, int size) {
        for (int i = 0; i < size; ++i) {
            out[i] *= in[i];
        }
    }

    // 5. Rotary Position Embeddings (RoPE)
    void rope(float* vec, int num_heads, int head_dim, int position, float theta) {
        for (int h = 0; h < num_heads; ++h) {
            float* head_vec = vec + (h * head_dim);
            
            // Apply rotation to pairs of coordinates
            for (int i = 0; i < head_dim / 2; ++i) {
                float x0 = head_vec[i];
                float x1 = head_vec[i + (head_dim / 2)];
                
                // Calculate position angle
                float freq = 1.0f / std::pow(theta, static_cast<float>(2 * i) / head_dim);
                float angle = position * freq;
                
                float cos_val = std::cos(angle);
                float sin_val = std::sin(angle);
                
                // Rotate vector
                head_vec[i] = x0 * cos_val - x1 * sin_val;
                head_vec[i + (head_dim / 2)] = x0 * sin_val + x1 * cos_val;
            }
        }
    }

    // 6. Mixed-Precision Matrix Multiplication (GEMM)
    void matmul(float* out, const float* x, const Tensor& w) {
        int rows = w.shape[0];
        int cols = w.shape[1];
        
        // Case A: Float32 (Standard Uncompressed weights)
        if (w.type == DataType::F32) {
            const float* w_ptr = static_cast<const float*>(w.data);
            
            // Loop over rows of matrix
            for (int r = 0; r < rows; ++r) {
                float sum = 0.0f;
                const float* row_ptr = w_ptr + (r * cols);
                
                // Dot product
                for (int c = 0; c < cols; ++c) {
                    sum += x[c] * row_ptr[c];
                }
                out[r] = sum;
            }
        }
        
        // Case B: 8-Bit Quantized weights (Q8_0)
        else if (w.type == DataType::Q8_0) {
            const BlockQ8_0* w_blocks = static_cast<const BlockQ8_0*>(w.data);
            int blocks_per_row = cols / 32;
            
            for (int r = 0; r < rows; ++r) {
                float sum = 0.0f;
                const BlockQ8_0* row_blocks = w_blocks + (r * blocks_per_row);
                
                for (int b = 0; b < blocks_per_row; ++b) {
                    const BlockQ8_0& block = row_blocks[b];
                    float block_sum = 0.0f;
                    int x_offset = b * 32;
                    
                    // Unroll 32-value block dot product
                    for (int i = 0; i < 32; ++i) {
                        block_sum += x[x_offset + i] * block.qs[i];
                    }
                    // Apply block scale factor
                    sum += block_sum * block.d;
                }
                out[r] = sum;
            }
        }
        
        // Case C: 2-Bit Quantized weights (Q2_K) - Used for MoE experts
        else if (w.type == DataType::Q2_K) {
            const BlockQ2* w_blocks = static_cast<const BlockQ2*>(w.data);
            int blocks_per_row = cols / 32;
            
            for (int r = 0; r < rows; ++r) {
                float sum = 0.0f;
                const BlockQ2* row_blocks = w_blocks + (r * blocks_per_row);
                
                for (int b = 0; b < blocks_per_row; ++b) {
                    const BlockQ2& block = row_blocks[b];
                    float block_sum = 0.0f;
                    int x_offset = b * 32;
                    
                    // Decode 32 packed 2-bit values (4 values per byte)
                    for (int i = 0; i < 32; ++i) {
                        int byte_idx = i / 4;
                        int bit_shift = (i % 4) * 2;
                        
                        // Extract 2-bit value (0, 1, 2, or 3)
                        uint8_t q = (block.qs[byte_idx] >> bit_shift) & 0x03;
                        
                        // Dequantize: value = (quantized * scale) + bias
                        float dequantized = (q * block.d) + block.min;
                        
                        block_sum += x[x_offset + i] * dequantized;
                    }
                    sum += block_sum;
                }
                out[r] = sum;
            }
        }
        
        // Case D: 4-Bit Quantized weights (Q4_0) - Used for MoE experts
        else if (w.type == DataType::Q4_0) {
            const BlockQ4_0* w_blocks = static_cast<const BlockQ4_0*>(w.data);
            int blocks_per_row = cols / 32;
            
            for (int r = 0; r < rows; ++r) {
                float sum = 0.0f;
                const BlockQ4_0* row_blocks = w_blocks + (r * blocks_per_row);
                
                for (int b = 0; b < blocks_per_row; ++b) {
                    const BlockQ4_0& block = row_blocks[b];
                    float block_sum = 0.0f;
                    int x_offset = b * 32;
                    
                    // Unroll 32-value block dot product using 4-bit values (2 values per byte)
                    for (int i = 0; i < 32; i += 2) {
                        uint8_t byte_val = block.qs[i / 2];
                        
                        // Extract lower 4 bits (first weight)
                        int q0 = byte_val & 0x0F;
                        // Extract upper 4 bits (second weight)
                        int q1 = byte_val >> 4;
                        
                        // Scale and center around zero
                        float w0 = block.d * (q0 - 8);
                        float w1 = block.d * (q1 - 8);
                        
                        block_sum += x[x_offset + i] * w0;
                        block_sum += x[x_offset + i + 1] * w1;
                    }
                    sum += block_sum;
                }
                out[r] = sum;
            }
        }
    }

} // namespace ops
