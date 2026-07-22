# HydraEngine

**HydraEngine** is a high-performance, zero-dependency C++ inference engine designed to run ultra-large Mixture-of-Experts (MoE) models (such as the 284B Parameter DeepSeek-V4-Flash) on consumer-grade hardware. 

By leveraging **Active-Expert SSD Streaming** and **Zipfian RAM Caching**, HydraEngine bypasses traditional GPU VRAM bottlenecks, allowing models that typically require datacenter clusters to run locally on standard PCs and laptops.

---

## Key Features

- 🚀 **Zero-Dependency C++:** Pure C++ implementation with no heavy dependencies (no PyTorch runtime, no CUDA toolkit overhead, no heavy Python packaging).
- 💾 **Active-Expert SSD Streaming:** Keeps the base attention layers permanently pinned in VRAM, while dynamically paging active expert weights from SSD/RAM to GPU on-the-fly during inference.
- 🧠 **Dynamic RAM Cache:** Caches the most frequently selected routed experts in system RAM, utilizing power-law routing patterns (Zipf's Law) to achieve a high cache-hit rate and minimize SSD read latency.
- 🔢 **Multi-Format Dequantization:** Built-in support for mixed-precision computation:
  - Base layers and shared experts are dequantized from native FP8/Int8 to Float32 on the CPU and processed via highly optimized block-quantized **Q8_0** kernels.
  - Routed experts are block-quantized to **Q4_0** (4-bit nibbles), bringing individual expert sizes down to just **7.08 MB** each.
- ⚡ **Low-Rank Attention Support:** Natively supports low-rank Multi-head Latent Attention (MLA) projections to maintain high accuracy and low key-value cache size.

---

## Architecture Overview

Mixture-of-Experts (MoE) models activate only a fraction of their total parameters per token. HydraEngine exploits this sparsity by decoupling the memory layout:

```
                  +-----------------------------------+
                  |           User Prompt             |
                  +-----------------------------------+
                                    |
                                    v
+------------------+    +-----------------------+    +-----------------------+
|  Base Layers &   |    |    System RAM Cache   |    |   NVMe SSD Storage    |
|   Embeddings     |    |   (Hot Experts, Q4)   |    |  (Cold Experts, Q4)   |
| (Pinned in VRAM) |    +-----------------------+    +-----------------------+
+------------------+                |                            |
         |                          | (Cache Hit)                | (Cache Miss)
         |                          +-------------\    /---------+
         |                                         \  /
         v                                          v
+------------------------------------------------------------+
|                  Active Matrix Multiplication              |
+------------------------------------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |         Generated Token           |
                  +-----------------------------------+
```

---

## File Structure

- `/src`: Core C++ source files:
  - `tensor.h` / `tensor.cpp`: Memory-mapped tensor abstractions and Zero-dependency Safetensors parser.
  - `ops.h` / `ops.cpp`: High-performance mixed-precision GEMM math kernels (F32, Q8_0, Q4_0).
  - `cache.h` / `cache.cpp`: Dynamic expert cache manager.
  - `model.h` / `model.cpp`: Model loader, layer execution graph, and state management.
  - `main.cpp`: CLI interface and token generation loop.
- `shard_deepseek.py`: Python script that reads raw 8-bit model partitions, dequantizes them, and outputs sharded base layers and Q4_0 expert files.

---

## Setup & Compilation

### Prerequisites
- A modern C++ compiler supporting C++17 (e.g., `g++` or `clang++`).
- Python 3.10+ with `numpy` and `torch` (used only for the initial one-time sharding step).

### Step 1: Shard the Model
Run the Python sharding script to convert raw model weights into the memory-efficient HydraEngine layout:
```bash
python shard_deepseek.py
```
This will output:
- `base_meta.safetensors` (embedding and output projection tables).
- `base_layer_{0..42}.safetensors` (pinned attention and shared expert weights).
- `/experts/expert_{0..42}_{0..255}.safetensors` (the 11,008 individual 4-bit expert shards).

### Step 2: Compile the Engine
Compile the C++ binary:
```bash
g++ -O3 -std=c++17 src/*.cpp -o bin/hydraengine
```

### Step 3: Run Inference
Generate responses locally:
```bash
./bin/hydraengine --prompt "Aether is the real and it will work no matter what." --max-tokens 128
```

---

## License

This project is licensed under a proprietary **Non-Commercial License**. 

Personal, educational, and academic research use is fully permitted. Any commercial reproduction, distribution, hosting as a service, or integration into commercial products is strictly prohibited without explicit written permission from Gurneev Singh (singh.gurneev140@gmail.com).
