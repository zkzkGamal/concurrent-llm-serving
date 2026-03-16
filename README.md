# 🚀 Concurrent LLM Serving: vLLM vs SGLang vs Ollama

> A hands-on benchmark and deep-dive guide to the three dominant open-source LLM serving engines.  
> **Model tested:** `Qwen/Qwen3.5-0.8B` | **Hardware:** Single GPU | **Concurrency:** 16 simultaneous requests

---

## 📋 Table of Contents

1. [What is Concurrent LLM Serving?](#-what-is-concurrent-llm-serving)
2. [The Three Engines](#-the-three-engines)
   - [SGLang](#-sglang--structured-generation-language)
   - [vLLM](#-vllm--virtual-large-language-model)
   - [Ollama](#-ollama)
3. [Core Algorithms Deep Dive](#-core-algorithms-deep-dive)
   - [KV-Cache & Paged Attention](#kv-cache--paged-attention)
   - [Continuous Batching](#continuous-batching)
   - [Prefix Caching](#prefix-caching)
   - [Tensor Parallelism](#tensor-parallelism)
4. [Benchmark Results](#-benchmark-results)
5. [When to Use Each One](#-when-to-use-each-one)
6. [How to Run the Tests](#-how-to-run-the-tests)
7. [API Compatibility](#-api-compatibility)
8. [Project Structure](#-project-structure)

---

## 🧠 What is Concurrent LLM Serving?

When you run a large language model in production, you almost never serve one user at a time. Hundreds or thousands of requests arrive simultaneously. A naive serving engine processes them **one by one** (sequential), wasting 90%+ of GPU capacity.

**Concurrent serving engines** solve this by:

1. **Batching requests together** — feeding multiple prompts to the GPU in one forward pass
2. **Smart memory management** — reusing GPU memory across requests without fragmentation
3. **Continuous scheduling** — not waiting for all requests to finish before starting new ones

The key insight: **a GPU runs at the same cost whether it processes 1 token or 64 tokens per step**. Efficient engines saturate the GPU with as many tokens as possible on every single forward pass.

---

## 🏗️ The Three Engines

### 🟣 SGLang — Structured Generation Language

**Repository:** [github.com/sgl-project/sglang](https://github.com/sgl-project/sglang)  
**Origin:** Stanford / UC Berkeley research  
**Philosophy:** Maximum throughput through aggressive optimization at every layer

SGLang is built ground-up for **high-throughput production serving**. It provides an OpenAI-compatible API but replaces the entire inference stack with custom CUDA kernels, a native scheduler, and a novel caching system.

#### How SGLang Works

```
Client Requests (16 concurrent)
         │
         ▼
┌─────────────────────────┐
│   HTTP Server (FastAPI) │
│   Port 8000             │
└────────────┬────────────┘
             │ batches requests
             ▼
┌─────────────────────────┐
│  RadixAttention Cache   │  ← Key innovation: prefix-aware KV cache
│  (Trie-based prefix     │
│   sharing)              │
└────────────┬────────────┘
             │ cache hits skip re-computation
             ▼
┌─────────────────────────┐
│  Continuous Batching    │  ← Requests enter/exit without waiting
│  Scheduler              │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Custom Triton Kernels  │  ← Replaces standard CUDA/PyTorch ops
│  (FlashAttention v3)    │
└─────────────────────────┘
```

#### SGLang Key Features

| Feature | Description |
|---|---|
| **RadixAttention** | A trie (prefix tree) that stores KV cache blocks. If two requests share the same system prompt, SGLang computes attention for it **once** and shares the result. |
| **Chunked Prefill** | Long prompts are broken into chunks and interleaved with decode steps, preventing stalls |
| **Triton Kernels** | Custom GPU kernels bypass PyTorch overhead for maximum throughput |
| **Zero warm-up** | Server is ready to serve instantly on startup |
| **Native JSON/Regex output** | Forces structured output formats without post-processing |

#### SGLang Startup
```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-0.8B \
    --port 8000 \
    --tp-size 1 \
    --mem-fraction-static 0.8 \
    --context-length 32768 \
    --attention-backend triton
```

> **Notice:** SGLang outputs `The server is fired up and ready to roll!` — no warm-up needed.

---

### 🔵 vLLM — Virtual Large Language Model

**Repository:** [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)  
**Origin:** UC Berkeley Sky Computing Lab  
**Philosophy:** The industry standard — rock-solid PagedAttention for memory efficiency

vLLM introduced **PagedAttention** in 2023 and became the de-facto standard for LLM serving in production. It prioritizes stability, broad model support, and correct memory management.

#### How vLLM Works

```
Client Requests (16 concurrent)
         │
         ▼
┌─────────────────────────┐
│  AsyncLLMEngine         │
│  (async request queue)  │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  PagedAttention         │  ← Key innovation: virtual KV cache pages
│  Block Manager          │
│  ┌──────┬──────┬──────┐ │
│  │Block │Block │Block │ │  KV cache split into fixed-size blocks
│  │  0   │  1   │  2   │ │  (like OS virtual memory pages)
│  └──────┴──────┴──────┘ │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Continuous Batching    │
│  Scheduler              │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  CUDA / Flash Attention │
│  (standard kernels)     │
└─────────────────────────┘
```

#### PagedAttention — The Core Innovation

Traditional LLM serving pre-allocates a contiguous chunk of GPU memory for the KV cache of each request. If you estimate max sequence length as 4096 tokens, you reserve 4096 × `hidden_size` × 2 (K+V) bytes **per request** — even if the actual response is 50 tokens. This causes massive **memory fragmentation** and limits concurrency.

PagedAttention solves this like an OS manages RAM:
- KV cache is split into **fixed-size blocks** (e.g., 16 tokens per block)
- Blocks are **allocated on demand** as tokens are generated
- Non-contiguous blocks are linked via a virtual address table
- When a request finishes, its blocks are **immediately freed** and reused

This allows vLLM to serve **~2-4× more concurrent requests** compared to pre-allocation servers.

#### vLLM Key Features

| Feature | Description |
|---|---|
| **PagedAttention** | Virtual memory paging for KV cache — eliminates fragmentation |
| **Continuous Batching** | New requests join in-flight batches without waiting |
| **Prefix Caching** | Caches shared prefixes (system prompts) across requests |
| **Tensor Parallelism** | Split model across multiple GPUs with `--tensor-parallel-size` |
| **Broad Model Support** | Supports 50+ model architectures including all Qwen, Llama, Mistral variants |
| **Warm-up required** | First request triggers CUDA graph compilation — send a warm-up message |

#### vLLM Startup
```bash
vllm serve Qwen/Qwen3.5-0.8B \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.75 \
    --reasoning-parser qwen3
```

> **Important:** vLLM needs a **warm-up message** after startup before serving real requests. The first cold request may be slow due to CUDA graph compilation.

---

### 🟠 Ollama

**Repository:** [github.com/ollama/ollama](https://github.com/ollama/ollama)  
**Origin:** Community project (open-source)  
**Philosophy:** Developer experience and ease of use — batteries included

Ollama is **not designed for concurrent production serving**. It is designed for **individual developers** to run models locally with zero friction. It handles model downloads, quantization, and management automatically.

#### How Ollama Works

```
Client Requests (4 sequential-ish)
         │
         ▼
┌─────────────────────────┐
│   Ollama HTTP Server    │
│   Port 11434            │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   llama.cpp backend     │  ← CPU/GPU inference via llama.cpp
│   (GGUF format models)  │
└────────────┬────────────┘
             │  processes requests
             ▼  one at a time (default)
┌─────────────────────────┐
│   Metal / CUDA / CPU    │
│   (best available)      │
└─────────────────────────┘
```

#### Ollama Key Features

| Feature | Description |
|---|---|
| **GGUF / quantization** | Models stored in compressed GGUF format (4-bit, 8-bit, etc.) — fits in less VRAM |
| **Zero-config** | `ollama run qwen3.5:0.8b` — just works |
| **Memory mapping** | `use_mmap=True` maps model weights from disk, reduces RAM usage |
| **keep_alive** | Keeps model loaded in memory between requests |
| **Sequential by default** | Processes one request at a time by default (concurrency via `OLLAMA_NUM_PARALLEL`) |
| **Auto-GPU detection** | Automatically uses GPU if available, falls back to CPU |

#### Concurrency Note
Ollama does support parallel requests via the `OLLAMA_NUM_PARALLEL` environment variable, but it uses **time-slicing** between requests, not true batching. Each request competes for the same memory, which is why our test showed dramatically increasing response times (26s → 50s → 64s → 134s) for just 4 concurrent requests.

---

## 🔬 Core Algorithms Deep Dive

### KV-Cache & Paged Attention

Every transformer layer computes **Key** and **Value** matrices for each token in the sequence. During auto-regressive generation (token by token), these matrices must be recomputed or cached for all previous tokens.

```
Prompt: "What is AI?"  (4 tokens)
                         ┌──────────────────────┐
Token 1: "What"    ───►  │ K1, V1               │
Token 2: "is"      ───►  │ K1, V1, K2, V2       │  KV grows each step
Token 3: "AI"      ───►  │ K1, V1, K2, V2, K3,  │
Token 4: "?"       ───►  │ V3, K4, V4            │
                         └──────────────────────┘
Generated: "AI is..."
                         ┌──────────────────────────────┐
New token: "AI"    ───►  │ K1..K4, V1..V4, K5, V5      │  must keep ALL past KV
                         └──────────────────────────────┘
```

**Without KV cache:** Every new token re-computes attention over the entire prompt → O(n²) cost.  
**With KV cache:** Store K and V once, only compute for the new token → O(n) per step.

The challenge is that this cache grows as sequences get longer, and managing it across many concurrent requests is the core problem all three engines solve differently.

---

### Continuous Batching

**Naive batching:** Wait for N requests to arrive → process all together → wait again.  
**Problem:** Requests finish at different times. Short responses leave GPU slots empty while waiting.

```
Naive Batching (Bad):
Time ──────────────────────────────────────►
Slot A: [Request 1 ████████████] [wait] [Request 4 ██████████]
Slot B: [Request 2 ████] [──wait──────] [Request 4 ██████████]
                               ↑ GPU IDLE

Continuous Batching (Good):
Time ──────────────────────────────────────►
Slot A: [Request 1 ████████████][Request 3 ████████][Request 5 ██]
Slot B: [Request 2 ████][Request 4 ██████████][Request 6 ██████]
              ↑ new request joins IMMEDIATELY when slot opens
```

SGLang and vLLM both implement continuous batching. Ollama does not — it's sequential by default.

---

### Prefix Caching

If multiple users send requests with the same system prompt:

```
Request 1: [SYSTEM: You are a helpful assistant.] + "What is Python?"
Request 2: [SYSTEM: You are a helpful assistant.] + "Explain Docker."
Request 3: [SYSTEM: You are a helpful assistant.] + "Write a haiku."
```

The KV cache for `[SYSTEM: You are a helpful assistant.]` is computed **once** and reused across all three.

**SGLang's RadixAttention** extends this with a **trie (prefix tree)** data structure:

```
                 Root
                  │
      [SYSTEM: You are a helpful assistant.]
                  │
        ┌─────────┼──────────┐
   "What is   "Explain    "Write a
    Python?"   Docker."    haiku."
```

This means any common prefix — system prompt, few-shot examples, long context — is cached and shared. This is why SGLang is dramatically faster for repeated-prefix workloads.

---

### Tensor Parallelism

For large models (70B+), a single GPU can't hold all weights. Tensor parallelism splits the model's weight matrices across multiple GPUs:

```
Model Layer (e.g., Attention)
┌─────────────────────────────────┐
│  Weight Matrix W (4096 × 4096)  │
└─────────────────────────────────┘
            Splits into:
┌──────────────────┐  ┌──────────────────┐
│  GPU 0: W[:2048] │  │  GPU 1: W[2048:] │
└──────────────────┘  └──────────────────┘
        ↓ partial result              ↓ partial result
              all-reduce (combine results)
                     ↓
              final output
```

Both vLLM (`--tensor-parallel-size`) and SGLang (`--tp-size`) support this. Ollama does not (uses llama.cpp which handles multi-GPU differently).

---

## 📊 Benchmark Results

### Setup
- **Model:** `Qwen/Qwen3.5-0.8B`
- **Requests:** 16 concurrent (4 for Ollama due to its sequential nature)
- **Task:** Each request: diverse AI/programming questions, `max_tokens=150`
- **Date:** 2026-03-15

### Summary Table

| Engine | Requests | Total Time | Avg per Request | Concurrency Model |
|--------|----------|------------|-----------------|-------------------|
| 🥇 **SGLang** | 16 | **2.47s** | ~0.68–2.46s | True parallel batching |
| 🥈 **vLLM** | 16 | **11.26s** | ~10.25–11.26s | PagedAttention batching |
| 🥉 **Ollama** | 4 | **134.72s** | 26–134s | Sequential (time-sliced) |

### Speed Comparison (Visual)

```
SGLang  ██  2.47s   ← 4.6× faster than vLLM
vLLM    ██████████████████████████████████████████████  11.26s
Ollama  (off the chart — 134.72s for only 4 requests)
```

### SGLang — Per-Request Breakdown (16 concurrent)

| # | Duration | Note |
|---|----------|------|
| 3, 9 | 0.677–0.678s | Cache hit / short response |
| 2, 6 | 0.883–1.072s | Cache hit |
| 15 | 2.219s | Longer generation |
| 1, 4–8, 10–14, 16 | ~2.461–2.464s | Full generation |
| **Total** | **2.47s** | All 16 done |

> **Why are some SGLang requests under 1 second?** RadixAttention prefix caching. Short prompts that share token prefixes with already-cached sequences skip re-computation entirely.

### vLLM — Per-Request Breakdown (16 concurrent)

| # | Duration | Note |
|---|----------|------|
| 3, 9 | 10.25–10.29s | Shortest response |
| 6 | 10.516s | |
| 2 | 10.686s | |
| 15 | 11.115s | |
| 1, 4–5, 7–8, 10–14, 16 | ~11.26s | Typical |
| **Total** | **11.26s** | All 16 done |

> **Why all requests ~10–11s?** vLLM processes the batch together — all 16 start simultaneously and finish around the same time. The spread reflects token count differences.

### Ollama — Per-Request Breakdown (4 sequential)

| # | Duration | Note |
|---|----------|------|
| 1 | 26.08s | First in queue |
| 2 | 50.52s | Waited for #1 |
| 4 | 64.65s | |
| 3 | **134.55s** | Last in queue — waited for all |
| **Total** | **134.72s** | Only 4 requests! |

> **Why does Ollama take so long?** Requests queue up sequentially. Request 3 had to wait for requests 1, 2, and 4 to complete first. Each new request must wait for all previous ones to finish before getting GPU time.

---

## 🎯 When to Use Each One

### Use SGLang when...

✅ You need **maximum throughput** in production  
✅ You have a **fixed system prompt** shared across many users (RAG chatbots, agents)  
✅ You need **structured output** (JSON mode, regex constraints)  
✅ Running a **high-traffic API** with 100+ concurrent users  
✅ You want **the fastest inference** with the latest optimization research  
✅ Working with **multi-modal models** (vision + text)  
✅ Need **low latency** at scale  

**Ideal use cases:**
- Production AI chatbots
- LLM-powered data extraction pipelines
- Multi-agent frameworks (many parallel agents)
- Document processing at scale

```python
# SGLang — best for: shared prefix + high concurrency
llm = ChatOpenAI(
    model_name="Qwen/Qwen3.5-0.8B",
    base_url="http://127.0.0.1:8000/v1",
    api_key="dummy"
)
# 16 requests → 2.47 seconds total ✨
```

---

### Use vLLM when...

✅ You need **production-grade stability** and broad model support  
✅ Running **diverse model architectures** (Llama, Qwen, Mistral, Falcon, etc.)  
✅ You need **multi-GPU tensor parallelism** for large models (13B, 70B+)  
✅ Need **OpenAI-compatible drop-in replacement**  
✅ Your team is familiar with it and you value **ecosystem maturity**  
✅ Need **quantization support** (AWQ, GPTQ, FP8)  
✅ Enterprise deployments requiring **long-term support**  

**Ideal use cases:**
- Self-hosted OpenAI API replacement
- Large model serving (70B+) across multiple GPUs
- Teams migrating from OpenAI to open-source
- When SGLang doesn't yet support your model

```python
# vLLM — best for: stability, model variety, multi-GPU
llm = ChatOpenAI(
    model_name="Qwen/Qwen3.5-0.8B",
    base_url="http://127.0.0.1:8000/v1",
    api_key="dummy"
)
# 16 requests → 11.26 seconds total (solid & reliable)
```

> **Note:** vLLM requires a warm-up request after startup due to CUDA graph compilation.

---

### Use Ollama when...

✅ You are a **developer experimenting locally**  
✅ You want **zero-config model management** (`ollama pull`, `ollama run`)  
✅ Building a **personal tool or prototype** with 1–2 users  
✅ Your GPU/CPU is limited and you need **quantized models** (4-bit GGUF)  
✅ You want to try a model **in 30 seconds** without setting up a server  
✅ No Python environment required — just install and run  

**Ideal use cases:**
- Local development and testing
- Personal chatbots for 1 user
- Quick prototypes before moving to production
- Learning/education environments
- Edge devices or low-resource hardware

```python
# Ollama — best for: local dev, easy setup, single user
from langchain_ollama import ChatOllama
llm = ChatOllama(
    model="qwen3.5:0.8b",
    base_url="http://127.0.0.1:11434"
)
# 4 requests → 134.72 seconds (sequential — not for production)
```

> **Warning:** Do not use Ollama for concurrent production workloads. Even 4 concurrent users will experience 2+ minute waits.

---

### Decision Flowchart

```
Is this for production with many users?
├── YES → Do you need max throughput or latest optimizations?
│          ├── YES → Use SGLang 🟣
│          └── NO  → Do you need broad model support or multi-GPU?
│                    ├── YES → Use vLLM 🔵
│                    └── NO  → Use SGLang (still faster) 🟣
└── NO (local dev / prototype / single user)
           └── Use Ollama 🟠
```

---

## 🛠️ How to Run the Tests

### Prerequisites

- Python 3.12
- NVIDIA GPU with CUDA (or Apple Silicon for Ollama)
- `uv` package manager (recommended for speed)

### Quick Install

```bash
# Install using the provided script
chmod +x install.sh
./install.sh
```

Choose `1` for vLLM, `2` for SGLang, or `3` for Ollama.

### Manual Install

**vLLM:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
python3.12 -m venv ~/sglang-env
source ~/sglang-env/bin/activate
uv pip install vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly

# Start server
vllm serve Qwen/Qwen3.5-0.8B \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.75 \
    --reasoning-parser qwen3
```

**SGLang:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
python3.12 -m venv ~/sglang-env
source ~/sglang-env/bin/activate
uv pip install 'git+https://github.com/sgl-project/sglang.git#subdirectory=python&egg=sglang[all]'

# Start server
python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-0.8B \
    --port 8000 \
    --tp-size 1 \
    --mem-fraction-static 0.8 \
    --context-length 32768 \
    --attention-backend triton
```

**Ollama:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull qwen3.5:0.8b
```

### Run Concurrent Tests

```bash
pip install langchain-openai langchain-ollama

# After starting the server of your choice:
python vllm_concurrent_test.py    # 16 concurrent requests to vLLM
python sglang_concurrent_test.py  # 16 concurrent requests to SGLang
python ollama_concurrent_test.py  # 4 concurrent requests to Ollama
```

Results are saved to `*_results.md` and logs to `*_concurrent_test.log`.

---

## 🔌 API Compatibility

All three engines expose an HTTP API, but with different ports and compatibility levels:

| Engine | Port | Path | OpenAI Compatible |
|--------|------|------|-------------------|
| vLLM | 8000 | `/v1/chat/completions` | ✅ Full |
| SGLang | 8000 | `/v1/chat/completions` | ✅ Full |
| Ollama | 11434 | `/api/chat` | ⚠️ Partial (own format + OpenAI compat) |

Since vLLM and SGLang are fully OpenAI-compatible, you can use `langchain_openai.ChatOpenAI` with both — just point `base_url` to `http://127.0.0.1:8000/v1`.

Ollama requires `langchain_ollama.ChatOllama` or using Ollama's OpenAI-compatible endpoint at `http://127.0.0.1:11434/v1`.

---

## 📚 Project Structure

```
test_concurrent_llm/
├── README.md                      # This file
├── install.sh                     # Interactive installer for all three engines
│
├── # Test Scripts
├── vllm_concurrent_test.py        # 16 concurrent requests → vLLM
├── sglang_concurrent_test.py      # 16 concurrent requests → SGLang
├── ollama_concurrent_test.py      # 4 concurrent requests → Ollama
│
├── # Setup Instructions
├── setup_vllm.txt                 # vLLM install + server startup commands
├── setup_sglang.txt               # SGLang install + server startup commands
│
├── # Benchmark Logs (raw)
├── vllm_concurrent_test.log       # Raw timing log from vLLM run
├── sglang_concurrent_test.log     # Raw timing log from SGLang run
├── ollama_concurrent_test.log     # Raw timing log from Ollama run
│
└── # Benchmark Results (markdown)
    ├── vllm_results.md            # vLLM: 16 requests in 11.26s
    ├── sglang_results.md          # SGLang: 16 requests in 2.47s
    └── ollama_results.md          # Ollama: 4 requests in 134.72s
```

---

## 🔑 Key Takeaways

| | SGLang | vLLM | Ollama |
|---|---|---|---|
| **Speed (16 concurrent)** | 🔥 2.47s | ⚡ 11.26s | 🐢 134.72s (4 req) |
| **Algorithm** | RadixAttention + Triton | PagedAttention | llama.cpp sequential |
| **KV Cache** | Trie prefix sharing | Virtual paging | Standard |
| **Concurrency** | True parallel batching | True parallel batching | Sequential (default) |
| **Best for** | Production, high traffic | Stability, large models | Local dev |
| **Setup complexity** | Medium | Medium | ⭐ Very easy |
| **Model support** | Growing fast | 50+ architectures | 100+ via GGUF |
| **Warm-up needed** | ❌ No | ✅ Yes | ❌ No |

> **Bottom line:**  
> 🟣 **SGLang** if you need speed in production.  
> 🔵 **vLLM** if you need stability and model variety.  
> 🟠 **Ollama** if you're building locally and want it simple.

---

*Tested on 2026-03-15 | Model: Qwen/Qwen3.5-0.8B | Single GPU*
