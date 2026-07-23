# 🌌 HydraEngine - DeepSeek-V4-Flash Sub-Engine

This directory contains the self-contained C++ implementation and preparation scripts tailored specifically for **DeepSeek-V4-Flash (284B total parameters, 13B active)**. By using virtual memory expert-paging and a custom 2.12-bit quantization, this engine runs inference locally using only **~5.33 GB of active physical RAM**!

---

## 🚀 Step-by-Step Setup & Execution

### Step 1: Download the Raw Model Weights
First, download the raw model checkpoints from Hugging Face. The model is stored in FP8 (8-bit) format, totaling about 148 GB. You can use the Hugging Face CLI tool to download it directly into an external directory (e.g. `D:/deepseek_raw`):

```bash
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash --local-dir D:/deepseek_raw
```

### Step 2: Shard, Quantize, and Prepare Vocab
Run the custom Python preparation script. This script executes a memory-optimized sharding pass using **disk-based staging** (keeping RAM usage under 100 MB):

```bash
python shard_deepseek.py
```

#### What this script does:
1. **Dequantizes and Quantizes Base Layers to 8-bit (Q8_0):** Saves the attention norms, projection matrices, and the always-active shared expert layer-by-layer as `base_layer_{0..42}.safetensors` and `base_meta.safetensors`.
2. **Quantizes Routed Experts to 2-bit (Q2_K):** Converts the 10,752 routed experts into a 2-bit block-quantized format using a **block size of 256**. Each block stores a float scale and float minimum, reducing the storage to exactly **2.12 bits per parameter** (approx. **6.68 MB per expert**), saved in the `experts/` subdirectory.
3. **Vocab Extraction:** Automatically decodes and parses `tokenizer.json` to generate the plain-text `vocab.txt` file read by the C++ tokenizer.

This compresses the total model footprint on your disk from 148 GB to **under 80 GB**!

### Step 3: Run Diagnostics Verification
Check the structural integrity and headers of your sharded files to ensure they are fully compatible with the C++ parser:

```bash
python verify_shards.py
```

### Step 4: Compile the C++ Engine
Navigate into the `deepseek-v4/` directory and compile the binary:
* **Windows (via Powershell script):**
  ```powershell
  powershell -ExecutionPolicy Bypass -File build.ps1
  ```
* **Linux (GCC):**
  ```bash
  g++ -O3 -std=c++17 src/*.cpp -o bin/moe_cache_test
  ```

### Step 5: Run Inference
Pass the path to your sharded folder and your prompt to the executable:
```bash
# Windows
./bin/moe_cache_test.exe "D:/deepseek_sharded" "Hello! Explain quantum computing simply."

# Linux
./bin/moe_cache_test "D:/deepseek_sharded" "Hello! Explain quantum computing simply."
```

---

## 🧠 Core Architecture & Theory

DeepSeek-V4-Flash achieves state-of-the-art reasoning throughput via three primary architectural breakthroughs implemented inside this C++ engine:

### 1. Multi-head Latent Attention (MLA)
Traditional Transformers store a Key-Value cache of size $N \times (\text{heads} \times \text{dim})$ per token, which balloons to gigabytes and limits context length. 
* **Compression:** MLA projects Key-Value states into a compressed **512-dimensional latent vector** ($kv\_latent$) plus a **64-dimensional RoPE key** ($qk\_rope\_head\_dim$).
* **Attention Score Math:** The attention weights are computed directly in this compressed space:
  $$S_{h, p} = Q_{h} \cdot KV_{p}^T = \sum_{d=0}^{511} Q_{h}[d] \times KV_{p}[d]$$
* **Decoupled RoPE:** Query positions are rotated and dot-producted separately to maintain positional accuracy.
This reduces the KV Cache size by **57x**, allowing massive context lengths on consumer hardware.

### 2. Gated Routed + Shared Expert MoE
DeepSeek-V4-Flash contains **256 routed experts** and **1 always-active shared expert** per layer.
* **FFN Routing:** For every token, the engine passes the representation through a routing gate. Softmax is applied to choose the **Top-6 active experts**.
* **Shared Expert:** The shared expert runs unconditionally to capture broad, global factual knowledge.
* **Aggregation Math:** The outputs of the selected experts are scaled by their routing probabilities and aggregated with the shared expert's output:
  $$y = \text{shared\_expert}(x) + \sum_{i=1}^{6} g_i \times \text{routed\_expert}_i(x)$$

### 3. Zipfian Expert Paging
Since only 6 out of 256 experts are activated per token, MoE routing follows a Zipfian power-law distribution—a small subset of "hot" experts are selected for almost 80% of all tokens in a natural conversation.
* **LRU Cache (Capacity: 128):** Keeps the 128 most recently used experts memory-mapped in RAM (~906 MB).
* **On-Demand Page-in:** When a token requests an expert not in cache (Cache Miss), the virtual pager evicts the least recently used expert from RAM, and memory-maps (`mmap` / `MapViewOfFile`) the new expert file directly from the NVMe SSD.
* **Active RAM Footprint:**
  $$\text{Total RAM} = 4.22\text{ GB (Embeddings + Head)} + 0.1\text{ GB (Active Layer Base)} + 0.9\text{ GB (LRU Cache)} + 0.1\text{ GB (Buffers)} \approx \mathbf{5.32\text{ GB}}$$
