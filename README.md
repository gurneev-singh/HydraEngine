# 🌌 HydraEngine

<div align="center">

<pre>
    __  __           __              ______            _     
   / / / /_  _______/ /________ _   / ____/___  ____ _(_)___  ___ 
  / /_/ / / / / __  / / ___/ __ `/ / __/ / __ \/ __ `/ / __ \/ _ \
 / __  / /_/ / /_/ / / /  / /_/ / / /___/ / / / /_/ / / / / /  __/
/_/ /_/\__, /\__,_/_/_/   \__,_/ /_____/_/ /_/\__, /_/_/ /_/\___/ 
      /____/                                /____/                
</pre>

**Next-Generation Zero-Dependency C++ Virtual Memory Engine for Ultra-Scale Mixture-of-Experts (MoE)**

[![License](https://img.shields.io/badge/License-Proprietary%20Non--Commercial-red.svg)](#license)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/compiler_support/17)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-green.svg)](#platform-support)
[![Engine Mode](https://img.shields.io/badge/Engine%20Mode-Active--Expert%20Streaming-orange.svg)](#virtual-memory-caching)

</div>

---

## 🚀 Overview

**HydraEngine** is a high-performance, zero-dependency C++ inference engine designed to execute ultra-large Mixture-of-Experts (MoE) models on consumer-grade hardware. 

By implementing **Active-Expert SSD Streaming** and **Zipfian RAM Caching**, HydraEngine bypasses traditional GPU VRAM bottlenecks, allowing models that typically require server-class datacenter clusters (like the 284 Billion Parameter **DeepSeek-V4-Flash**) to run locally on standard PCs and laptops with as little as 8GB of RAM.

---

## 📊 Supported Model Roadmap

HydraEngine is built with a highly modular loading graph. While currently optimized for **DeepSeek-V4-Flash**, its virtual paging system is designed to natively support upcoming state-of-the-art MoE architectures.

| Model | Total Params | Active Params | Experts Structure | Status |
| :--- | :--- | :--- | :--- | :--- |
| **DeepSeek-V4-Flash** | **284B** | **13B** | 256 Routed Experts + 1 Shared Expert | **Fully Operational** (Running) |
| **GLM-5.2** | *TBD* | *TBD* | Multi-Query Routing & Gated FFN | *Planned Support* |
| **Kimi-K3** | *TBD* | *TBD* | Massive Context MoE Pager | *Planned Support* |

---

## 🧠 Core Architecture Highlights

```mermaid
graph TD
    UserPrompt["User Input (Prompt)"] --> Embedding["Embedding Lookup & Layer Norm"]
    
    subgraph AttentionBlock ["Multi-head Latent Attention (MLA) Layer"]
        Embedding --> Q_Proj["Query Low-Rank Projection (wq_a -> wq_b)"]
        Embedding --> KV_Proj["KV Compression (wkv)"]
        Q_Proj --> LatentAttention["Latent Dot Product (Q_h * KV_p)"]
        KV_Proj --> LatentAttention
        LatentAttention --> O_Proj["Grouped Low-Rank Out-Proj (wo_a -> wo_b)"]
    end
    
    O_Proj --> RoutingGate["MoE Router Gate (router_gate)"]
    
    subgraph PagingEngine ["Virtual Memory Paging Engine"]
        RoutingGate --> Softmax["Softmax & Top-6 Selection"]
        Softmax --> LRUCache{"LRU RAM Cache (Capacity: 128)"}
        
        LRUCache -- "Cache Hit (Fast)" --> RunExpert["Execute SwiGLU Expert (Q4_0)"]
        LRUCache -- "Cache Miss (Evicts oldest)" --> SSD_Read["Memory-Map from SSD"]
        SSD_Read --> RunExpert
    end

    RunExpert --> Accumulate["Accumulate (Routed + Shared Expert)"]
    Accumulate --> Output["RMSNorm & LM Head Logits"]
```

---

## ⚡ Technical Deep Dives

### 1. Multi-head Latent Attention (MLA)
To run massive contexts (up to 4096+ tokens) on low VRAM, HydraEngine implements DeepSeek's **Multi-head Latent Attention**. 
Instead of caching massive, full-rank Key and Value states ($64 \text{ heads} \times 512 \text{ dim} = 32,768$ float elements per token), we project Key-Value states into a compressed **512-dimensional latent vector** plus a **64-dimensional RoPE key**. 

The attention weights are computed directly in this compressed space:
$$S_{h, p} = Q_{h} \cdot KV_{p}^T = \sum_{d=0}^{511} Q_{h}[d] \times KV_{p}[d]$$
This achieves a **57x reduction** in KV Cache size without losing positional or representational accuracy.

### 2. Zipfian RAM Caching
Expert routing in large MoE models follows a power-law distribution (Zipf's Law): a small subset of "hot" experts are selected for almost 80% of all tokens in a natural conversation. 
HydraEngine utilizes an **LRU (Least Recently Used) Paging Cache** for the 256 routed experts:
* Base layers and embeddings are permanently loaded in memory.
* The 11,008 routed experts (quantized to **Q4_0** at just **7.08 MB** each) are loaded dynamically.
* If an expert is in the LRU Cache, it is reused instantly (**Cache Hit**).
* If a Cache Miss occurs, the oldest inactive expert is unmapped from RAM, and the new expert is memory-mapped from NVMe SSD on-the-fly.

---

## 📂 File Layout

```
HydraEngine/
├── src/
│   ├── tensor.h / tensor.cpp        # Zero-dependency Safetensors parser & memory mapper
│   ├── cache.h / cache.cpp          # Virtual LRU paging expert cache
│   ├── ops.h / ops.cpp              # Highly optimized mixed-precision GEMM math kernels
│   ├── model.h / model.cpp          # MLA execution graph and layer forward passes
│   ├── tokenizer.h / tokenizer.cpp  # hex/text token encoder & decoder
│   └── main.cpp                     # HydraEngine CLI execution entry point
├── shard_deepseek.py                # Python weight sharding and Q4_0 quantizer
├── verify_shards.py                 # Diagnostic script to check sharded weights integrity
├── build.ps1                        # Windows compiler automation script
└── LICENSE                          # Proprietary non-commercial terms
```

---

## 🛠️ Step-by-Step Setup

### Step 1: Weight Sharding & Quantization
Download the model weights from Hugging Face (`deepseek-ai/DeepSeek-V4-Flash`) into `D:/deepseek_raw` and execute the pipeline:
```bash
python shard_deepseek.py
```
This script performs a single-pass extraction:
1. Dequantizes base attention layer weights and saves them layer-by-layer as Q8_0 binaries (`base_layer_{0..42}.safetensors`).
2. Quantizes all 11,008 routed experts to Q4_0 format, saving them as individual files (`expert_{layer}_{expert}.safetensors` at ~7.08 MB each).
3. Generates the vocabulary lookup files (`vocab.txt`).

### Step 2: Verification
Check the integrity of the generated shards:
```bash
python verify_shards.py
```

### Step 3: Compilation
Compile the zero-dependency C++ executable:
* **Windows (via build.ps1):**
  ```powershell
  powershell -ExecutionPolicy Bypass -File build.ps1
  ```
* **Linux (GCC):**
  ```bash
  g++ -O3 -std=c++17 src/*.cpp -o bin/hydraengine
  ```

### Step 4: Run Inference
Provide your prompt directly to the engine:
```bash
./bin/hydraengine "D:/deepseek_sharded" "Hello! How can I help you today?"
```

---

## 🔒 License

This project is licensed under a proprietary **Non-Commercial License**.

Personal, educational, and academic research use is fully permitted. Any commercial reproduction, distribution, hosting as a service, or integration into commercial products is strictly prohibited without explicit written permission from Gurneev Singh (singh.gurneev140@gmail.com).

*For commercial licensing inquiries, please contact the author directly.*
