# 🧪 Heavy Load Test Report: VLLM

| Config | Value |
|---|---|
| Server Type | `VLLM` |
| Concurrency | `3` sessions |
| Total Time | `229.79s` |
| Success Rate | `3/3` |
| Total Turns | `15` |
| Max Context (tokens) | `~17,645` |

## 📊 Performance by Node Type

| Node | Sessions | Success % | Avg Turn Latency | Avg Context Tokens |
|---|---|---|---|---|
| CONVERSATION | 1 | 100.0% | `14.007s` | `~3,232` |
| ACT | 1 | 100.0% | `18.431s` | `~1,253` |
| SUMMARIZE | 1 | 100.0% | `45.957s` | `~9,216` |

## 📋 Session Details

### Session 1 [CONVERSATION] — vLLM vs SGLang vs Ollama Concurrency Analysis ✅
- Total duration: `70.03s` | Max context: `~6,259 tokens`

| Turn | Input (truncated) | Latency | Tokens |
|---|---|---|---|
| 1 | Concurrency is where most local setups break. Walk me through what happens insid... | `19.43s` | `~641` |
| 2 | Now explain SGLang's RadixAttention. How does prefix caching work as a trie stru... | `7.77s` | `~1,842` |
| 3 | We tested mixed request sizes — short 128-token prompts mixed with 8k-token long... | `14.57s` | `~2,957` |
| 4 | What does Ollama do differently from vLLM and SGLang? When is Ollama actually th... | `12.48s` | `~4,461` |
| 5 | Design a production load test for an LLM API endpoint. Specify: ramp-up strategy... | `15.79s` | `~6,259` |

### Session 2 [ACT] — Deep Technical Research ✅
- Total duration: `92.16s` | Max context: `~3,057 tokens`

| Turn | Input (truncated) | Latency | Tokens |
|---|---|---|---|
| 1 | Search for 'paged attention vLLM memory fragmentation solution' and provide a de... | `17.01s` | `~45` |
| 2 | Now search for 'continuous batching LLM serving scheduler design' and explain th... | `52.54s` | `~102` |
| 3 | Calculate total memory footprint for: Llama-3-70B in BF16 with 4-way tensor para... | `7.47s` | `~1,225` |
| 4 | Search for 'LLM inference quantization accuracy vs speed 2025' and build a compa... | `4.11s` | `~1,838` |
| 5 | Given a cluster of 8 H100 80GB GPUs: design the optimal serving configuration fo... | `11.02s` | `~3,057` |

### Session 3 [SUMMARIZE] — Research Paper Analysis and Action Plan ✅
- Total duration: `229.79s` | Max context: `~17,645 tokens`

| Turn | Input (truncated) | Latency | Tokens |
|---|---|---|---|
| 1 | Here is a detailed research paper summary on FlashAttention-3 and its production... | `8.59s` | `~2,146` |
| 2 | Write a comprehensive technical review of FA3 covering: (1) the three core innov... | `15.21s` | `~3,747` |
| 3 | Create a migration guide for a team currently running vLLM on A100s who wants to... | `21.44s` | `~5,395` |
| 4 | Based on the benchmarks in the paper, build a cost-benefit analysis for upgradin... | `174.91s` | `~17,150` |
| 5 | Write GitHub-style issue descriptions for the top 3 engineering tasks required t... | `9.63s` | `~17,645` |


> [!NOTE]
> This test uses real multi-turn prompts with 10k+ token contexts and live web search via DDGS.
> Results are labeled `vllm` for direct comparison with other serve types.
