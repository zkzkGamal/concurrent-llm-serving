# 🧪 Heavy Load Test Report: SGLANG

| Config | Value |
|---|---|
| Server Type | `SGLANG` |
| Concurrency | `3` sessions |
| Total Time | `255.83s` |
| Success Rate | `3/3` |
| Total Turns | `15` |
| Max Context (tokens) | `~24,631` |

## 📊 Performance by Node Type

| Node | Sessions | Success % | Avg Turn Latency | Avg Context Tokens |
|---|---|---|---|---|
| CONVERSATION | 1 | 100.0% | `10.022s` | `~3,060` |
| ACT | 1 | 100.0% | `5.310s` | `~1,186` |
| SUMMARIZE | 1 | 100.0% | `51.165s` | `~15,928` |

## 📋 Session Details

### Session 1 [CONVERSATION] — vLLM vs SGLang vs Ollama Concurrency Analysis ✅
- Total duration: `50.11s` | Max context: `~6,045 tokens`

| Turn | Input (truncated) | Latency | Tokens |
|---|---|---|---|
| 1 | Concurrency is where most local setups break. Walk me through what happens insid... | `5.39s` | `~561` |
| 2 | Now explain SGLang's RadixAttention. How does prefix caching work as a trie stru... | `9.29s` | `~1,774` |
| 3 | We tested mixed request sizes — short 128-token prompts mixed with 8k-token long... | `9.08s` | `~2,751` |
| 4 | What does Ollama do differently from vLLM and SGLang? When is Ollama actually th... | `12.27s` | `~4,173` |
| 5 | Design a production load test for an LLM API endpoint. Specify: ramp-up strategy... | `14.08s` | `~6,045` |

### Session 2 [ACT] — Time + Search + Calculation Chain ✅
- Total duration: `26.55s` | Max context: `~2,258 tokens`

| Turn | Input (truncated) | Latency | Tokens |
|---|---|---|---|
| 1 | What is the exact current date and time? Express it in ISO 8601, Unix timestamp,... | `7.23s` | `~41` |
| 2 | Search the web for 'Qwen3 model release 2025 performance' and summarize the key ... | `3.84s` | `~766` |
| 3 | Calculate the following: sqrt(1099511627776) + log base 10 of 10000000000 + (2^3... | `5.67s` | `~1,017` |
| 4 | Search for 'LangGraph production use cases 2025' and summarize what teams are ac... | `5.30s` | `~1,850` |
| 5 | Fetch the document 'performance_analysis_q1_2025'. Then compute: if average TTFT... | `4.50s` | `~2,258` |

### Session 3 [SUMMARIZE] — Architecture Document Deep Analysis ✅
- Total duration: `255.83s` | Max context: `~24,631 tokens`

| Turn | Input (truncated) | Latency | Tokens |
|---|---|---|---|
| 1 | Here is a comprehensive architecture document for our LLM serving system. Read i... | `11.49s` | `~2,550` |
| 2 | Provide a structured executive summary of the architecture document. Cover: (1) ... | `7.62s` | `~3,470` |
| 3 | Extract every specific numerical metric from the document. Present them in a str... | `236.41s` | `~24,448` |
| 4 | Analyze the failure modes section. For each failure mode: (1) assess its severit... | `0.16s` | `~24,541` |
| 5 | Based on the capacity planning section, a product manager wants to cut costs by ... | `0.15s` | `~24,631` |


> [!NOTE]
> This test uses real multi-turn prompts with 10k+ token contexts and live web search via DDGS.
> Results are labeled `sglang` for direct comparison with other serve types.
