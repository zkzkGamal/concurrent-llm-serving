import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import argparse
import time
import json
import logging
import random
from datetime import datetime
from typing import List, Dict, Any

from simpleagent.agent.graph import create_agent_graph
from langchain_core.messages import HumanMessage, BaseMessage

# ---------------------------
# 1. Setup Logging
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

# ---------------------------
# 2. Long Context Documents (for summarize node)
# ---------------------------

_ARCH_DOC = """
## Distributed LLM Serving Architecture — Internal Design Doc v3.1

### 1. Overview
This document describes the production architecture used to serve large language models at scale.
The system supports both synchronous and streaming inference, handles multi-tenant workloads,
and is designed for horizontal scalability across GPU clusters.

### 2. Core Components

#### 2.1 Load Balancer
We use a layer-7 load balancer (Nginx + custom Lua middleware) to distribute requests across
inference nodes. Requests are routed based on model ID and token budget. The balancer is
stateless and supports health-check-based failover with automatic retry logic.

#### 2.2 Inference Node
Each inference node runs a single model replica. Nodes are deployed as Kubernetes pods with
GPU affinity rules. Nodes expose an OpenAI-compatible REST API (/v1/chat/completions).
Key parameters: tensor-parallel-size, max-model-len, gpu-memory-utilization.
Each node also runs a local monitoring sidecar that emits Prometheus metrics every 5 seconds.

#### 2.3 KV Cache Manager
We implement a two-tier KV cache:
- L1: On-device HBM (High Bandwidth Memory) — fast but limited to ~60GB on A100.
- L2: CPU RAM overflow using pin-memory — slower (~10x), used only for very long contexts.
Paged attention (as in vLLM) is used to fragment the KV cache into fixed-size blocks (16 tokens),
allowing efficient memory reuse across concurrent requests without fragmentation.

#### 2.4 Request Scheduler
Requests are batched using a continuous batching strategy. The scheduler prioritizes:
1. Short requests (< 512 tokens) for low latency SLO compliance.
2. Long requests (> 4096 tokens) for throughput maximization.
Preemption is used when new high-priority requests arrive and the batch is full.
A fairness queue ensures no request waits more than 30 seconds.

#### 2.5 Token Streaming
Streaming is implemented via Server-Sent Events (SSE). The first token latency (TTFT) is
tracked per-request via distributed tracing. P50, P90, P99 TTFT are monitored in Grafana.
Streaming is enabled by default and can be disabled via the 'stream: false' parameter.

### 3. Performance Characteristics
- Throughput: 1,200 tokens/sec per A100 (80GB) at batch size 16 with FP16.
- TTFT P50: 120ms at batch size 1, 450ms at batch size 16.
- TTFT P99: 800ms at batch size 16 during peak load.
- Context length: Up to 32,768 tokens supported via paged attention.
- GPU memory utilization: 75% reserved for KV cache, 25% for model weights.
- Sustained throughput degrades by ~8% after 4 hours due to KV cache fragmentation.

### 4. Failure Modes and Mitigations
| Failure | Root Cause | Mitigation |
|---|---|---|
| OOM errors | High batch size + long contexts exceed HBM | Scheduling caps per context bucket |
| Latency spikes | KV cache eviction during GC | Pre-emptive preemption, cache warmup |
| Node failures | Pod crashes, GPU faults | K8s health checks, auto-restart |
| Tool call timeouts | Unexpectedly long model output | Per-request token budget enforcement |
| Throughput degradation | KV cache fragmentation | Periodic cache defragmentation job |

### 5. Comparison: vLLM vs SGLang

| Feature | vLLM | SGLang |
|---|---|---|
| Batching Strategy | Continuous Batching | Continuous Batching + RadixAttention |
| KV Cache | Paged Attention | Shared KV Prefix Cache (RadixAttention) |
| Tool Calling | Requires --enable-auto-tool-choice | Native OpenAI format support |
| Streaming | Yes (SSE) | Yes (SSE) |
| Max Throughput | ~1,200 tok/s (A100 80GB) | ~1,450 tok/s (A100 80GB) |
| Long-Context Handling | Excellent with paged attn | Better for shared prefix workloads |
| Multi-turn Efficiency | Standard | +40% via prefix caching |
| Cold Start | ~90s for 70B model | ~85s for 70B model |
| Community | Large, mature, widely deployed | Newer, rapidly growing, academic origin |
| FP8 Support | Yes (via quantization) | Yes (native FA3 integration roadmap) |

### 6. Deployment Checklist
- [ ] Set --tensor-parallel-size to number of GPUs per node.
- [ ] Set --max-model-len based on maximum expected context (recommend 32768).
- [ ] Set --gpu-memory-utilization to 0.75 to avoid OOM.
- [ ] Enable --enable-auto-tool-choice for function calling.
- [ ] Set --tool-call-parser to 'qwen' for Qwen-family models.
- [ ] Configure health check endpoints in Kubernetes (/health, /metrics).
- [ ] Set up Grafana dashboards for TTFT, throughput, GPU utilization, and queue depth.
- [ ] Test preemption behavior under load before production launch.
- [ ] Validate streaming with 50+ concurrent clients using wrk2 or locust.
- [ ] Configure log rotation and retention policy (recommend 7 days for debug logs).

### 7. Open Engineering Issues
- Issue #1142: KV cache fragmentation under mixed workloads degrades throughput by ~15%.
  ETA: Q2 2025 — implementing block compaction algorithm.
- Issue #1187: TTFT increases sharply when batch size exceeds 24 on H100 nodes.
  Root cause: WGMMA scheduling conflicts. No mitigation yet.
- Issue #1203: Tool calling occasionally times out when model output is unexpectedly long.
  Mitigation: Added per-request output token cap enforcement (max_tokens parameter).
- Issue #1221: Memory leak in request trace buffer after 48h uptime.
  Workaround: Scheduled pod restart every 24h. Permanent fix in progress.

### 8. Capacity Planning
For 500 concurrent users targeting p95 < 2s with average 800-token output:
- Minimum: 4x A100 80GB nodes behind the load balancer.
- Recommended: 6x A100 80GB nodes for headroom and rolling deployment.
- Estimated monthly cost at $2.50/hr per GPU: ~$10,800/month for 4 nodes.

End of Architecture Document.
""".strip()

_RESEARCH_DOC = """
## Paper Summary: FlashAttention-3 and Its Impact on Production LLM Serving

### Abstract
FlashAttention-3 (FA3) achieves 1.5–2.0x speedup over FlashAttention-2 on H100 GPUs by exploiting:
(1) warp-specialized asynchronous pipelining, (2) interleaving block-wise matmul and softmax,
and (3) incoherent processing with FP8 low-precision rounding. FA3 reaches 740 TFLOPs/s on H100
for FP16, representing 75% of model FLOP utilization — a record for attention kernels.

### Key Techniques

#### Technique 1: Warp-Specialized Asynchronous Pipelining
H100 GPUs introduce Tensor Memory Accelerator (TMA) units that can initiate async memory transfers
independently from the CUDA cores. FA3 assigns dedicated warpgroups as "producers" handling TMA
memory operations, while other warpgroups act as "consumers" performing WGMMA (Warpgroup Matrix
Multiply Accumulate) operations. This allows memory transfers and computation to overlap, hiding
the HBM latency that was a bottleneck in FA2.

#### Technique 2: Block-wise MMA and Softmax Interleaving
In standard attention, softmax must be computed after all QK^T products are computed. FA3 introduces
a 2-stage pipeline where two warpgroups alternate: while one warpgroup computes WGMMA for the current
block, the other computes softmax for the previous block. This "ping-pong" scheduling effectively
eliminates the softmax serialization bottleneck.

#### Technique 3: FP8 with Incoherent Processing
FP8 has lower dynamic range than FP16, leading to accuracy loss from outlier values in attention weights.
FA3 uses "incoherent processing" — multiplying Q and K by a random orthogonal matrix before the
attention computation. This spreads outlier values across many dimensions without changing the attention
output. The random matrix is the same for Q and K, so the transformation cancels out.

### Benchmark Results
| Configuration | Throughput | vs FA2 |
|---|---|---|
| H100 FP16 forward | 740 TFLOP/s | +1.5-2.0x |
| H100 FP16 backward | 570 TFLOP/s | +1.5x |
| H100 FP8 forward | ~1,200 TFLOP/s | +1.2x (vs FA2 FP16) |
| A100 FP16 forward | ~310 TFLOP/s | +1.1x (limited gain) |

### Implications for vLLM and SGLang

**TTFT Impact:**
For long-context prompts (8k+ tokens), attention computation is the dominant cost.
FA3's 1.5-2x speedup translates directly to 30-50% TTFT reduction for these workloads.
For short prompts (< 512 tokens), the benefit is smaller (~10-15%) because compute time
is dominated by linear layers, not attention.

**Throughput Impact:**
At batch size 16 with 2048-token sequences on H100:
- FA2 baseline: ~1,800 tokens/sec
- FA3 estimated: ~2,700-3,600 tokens/sec
- FP8 FA3 estimated: ~3,500-4,500 tokens/sec (with minimal accuracy loss)

**Integration Status:**
- FA3 is available in the official FlashAttention repository.
- vLLM: Experimental support added in v0.4.1 for H100. Not default yet.
- SGLang: Integration roadmap announced for Q2 2025.
- TensorRT-LLM: Fully integrated as of v0.8.0.

### Limitations and Risks
1. H100-only optimization: A100 and older GPUs see minimal improvement from FA3-specific techniques.
2. FP8 accuracy: Requires careful calibration. Perplexity increase of ~0.3-0.8 observed on LLaMA-3.
3. Debugging complexity: Async pipelining makes profiling and debugging significantly harder.
4. Memory overhead: The incoherent processing random matrices add minor memory overhead (~2%).

### Conclusion
FA3 is a significant step forward for H100 inference. Production teams with H100 clusters should
prioritize evaluating FA3 integration, especially for long-context workloads where the 1.5-2x
speedup directly reduces infrastructure cost at scale.
""".strip()


# ---------------------------
# 3. Test Scenarios — Real, Heavy, Diverse
# ---------------------------
SCENARIOS = {
    "conversation": [
        {
            "name": "LLM Serving Architecture Deep Dive",
            "inputs": [
                "What are the fundamental differences between decoder-only and encoder-decoder transformer architectures? Why did decoder-only win out for large language models? Give me a detailed technical answer.",
                "Excellent. Now explain KV caching in depth — how does it work mechanically in a decoder-only model, what data is cached, how does it grow with context length, and why does it matter so much for production serving latency?",
                "Walk me through the tradeoffs between greedy decoding, beam search, top-k sampling, and nucleus (top-p) sampling. When would you choose each in a production system with strict P99 latency SLOs?",
                "Let's talk alignment. Walk me through RLHF step-by-step — reward modeling, proximal policy optimization, and the KL divergence penalty. Then compare it to DPO: what problem does DPO solve and what are its failure modes?",
                "We've covered architecture, caching, decoding, and alignment. Now synthesize everything: what are the top 5 engineering decisions that most impact production LLM serving performance, and what should an engineer focus on first?",
            ]
        },
        {
            "name": "GPU Infrastructure and Scaling",
            "inputs": [
                "Explain tensor parallelism, pipeline parallelism, and data parallelism. For a 70B parameter model running inference on 8 A100s, which combination would you recommend and why? Walk me through the GPU communication patterns for each.",
                "I'm deploying Llama-3-70B in BF16. Give me the exact formula for KV cache memory consumption as a function of: batch size, sequence length, number of layers, number of attention heads, and head dimension. Then plug in realistic numbers.",
                "Compare GPTQ, AWQ, SmoothQuant, and FP8 quantization methods in detail. For each: explain the core idea, accuracy impact, inference speedup on A100, inference speedup on H100, and which models it works best with.",
                "Design an auto-scaling policy for an LLM serving cluster. What metrics trigger scale-out? What's your cooldown period? How do you handle model loading latency when a new pod starts? What's your strategy for graceful preemption of in-flight requests?",
                "I have a 32k context window model serving 200 concurrent users. Each user has an average of 4k tokens of context and 500 tokens of output. Calculate the sustained throughput requirement, GPU memory requirement, and minimum cluster size for p95 < 3s latency.",
            ]
        },
        {
            "name": "vLLM vs SGLang vs Ollama Concurrency Analysis",
            "inputs": [
                "Concurrency is where most local setups break. Walk me through what happens inside vLLM when 50 simultaneous requests arrive. How does the continuous batching scheduler decide which requests to combine into a batch? What limits the batch size?",
                "Now explain SGLang's RadixAttention. How does prefix caching work as a trie structure? In a multi-turn conversation with 10 turns, how much KV cache reuse can we expect? What's the cache eviction policy?",
                "We tested mixed request sizes — short 128-token prompts mixed with 8k-token long-context prompts at a ratio of 70:30. Predict the latency distribution differences between vLLM and SGLang for this workload and explain your reasoning.",
                "What does Ollama do differently from vLLM and SGLang? When is Ollama actually the right tool, and when does it break down? Be specific about batch size limits, quantization quality, and streaming support.",
                "Design a production load test for an LLM API endpoint. Specify: ramp-up strategy, request mix, metrics to capture, success criteria, and how you'd identify whether the bottleneck is compute, memory bandwidth, or network I/O.",
            ]
        },
    ],
    "act": [
        {
            "name": "Research + Heavy Math",
            "inputs": [
                "Search the web for the latest vLLM vs SGLang throughput benchmarks comparing tokens per second in 2025. Summarize the top 3 results.",
                "Based on those results: calculate the percentage difference in throughput, and if an A100 costs $2.50/hr, compute the annual cost difference to process 1 trillion tokens with vLLM vs SGLang.",
                "Search for 'FlashAttention-3 H100 benchmark results' and extract all specific performance numbers you can find.",
                "Now calculate: if FA3 gives 1.8x speedup on H100 over FA2, and FA2 achieves 400 TFLOP/s on H100, what is FA3's absolute throughput? Convert to tokens/sec assuming 70B model with 80 attention layers, 64 heads, and 128 head_dim at batch size 8, sequence length 2048.",
                "Search for 'speculative decoding LLM inference speedup 2025' and tell me what the current state of the art is. Then calculate: if speculative decoding gives 2.5x speedup and base throughput is 1,200 tok/s, what is the new throughput and by how much does it reduce hourly GPU cost?",
            ]
        },
        {
            "name": "Time + Search + Calculation Chain",
            "inputs": [
                "What is the exact current date and time? Express it in ISO 8601, Unix timestamp, and human-readable format in both UTC and UTC+2.",
                "Search the web for 'Qwen3 model release 2025 performance' and summarize the key announced capabilities and benchmark results.",
                "Calculate the following: sqrt(1099511627776) + log base 10 of 10000000000 + (2^32 / 1024^3). Show each step.",
                "Search for 'LangGraph production use cases 2025' and summarize what teams are actually building with it in production.",
                "Fetch the document 'performance_analysis_q1_2025'. Then compute: if average TTFT is 340ms and we serve 1,000 requests/minute with average output of 600 tokens, what is the total tokens-per-minute and what is our GPU utilization if peak capacity is 1,200 tok/s?",
            ]
        },
        {
            "name": "Deep Technical Research",
            "inputs": [
                "Search for 'paged attention vLLM memory fragmentation solution' and provide a deep technical explanation of how the block manager works internally.",
                "Now search for 'continuous batching LLM serving scheduler design' and explain the key algorithmic differences between FCFS, shortest-job-first, and priority-based scheduling for LLM inference.",
                "Calculate total memory footprint for: Llama-3-70B in BF16 with 4-way tensor parallelism, batch size 16, sequence length 4096, 80 layers, 64 heads, head_dim=128. Include both model weights and KV cache.",
                "Search for 'LLM inference quantization accuracy vs speed 2025' and build a comparison table of the top 4 quantization methods.",
                "Given a cluster of 8 H100 80GB GPUs: design the optimal serving configuration for Llama-3-70B targeting 500 concurrent users. Include tensor-parallel-size, max batch size, recommended quantization, and expected throughput. Show your calculations.",
            ]
        },
    ],
    "summarize": [
        {
            "name": "Architecture Document Deep Analysis",
            "inputs": [
                f"Here is a comprehensive architecture document for our LLM serving system. Read it carefully and confirm once you have understood all sections:\n\n{_ARCH_DOC}",
                "Provide a structured executive summary of the architecture document. Cover: (1) system purpose, (2) key components and their roles, (3) performance characteristics, (4) comparison between vLLM and SGLang, and (5) the most critical operational risks.",
                "Extract every specific numerical metric from the document. Present them in a structured table with columns: Metric Name, Value, Unit, Condition/Context, and Which Component It Applies To.",
                "Analyze the failure modes section. For each failure mode: (1) assess its severity on a 1-10 scale with justification, (2) evaluate the effectiveness of the proposed mitigation, and (3) propose one additional mitigation not mentioned in the document.",
                "Based on the capacity planning section, a product manager wants to cut costs by 30% without violating the p95 < 2s SLA. Write a proposal with at least 3 specific engineering changes, their expected cost reduction, and their risk level.",
            ]
        },
        {
            "name": "Research Paper Analysis and Action Plan",
            "inputs": [
                f"Here is a detailed research paper summary on FlashAttention-3 and its production implications. Process it thoroughly:\n\n{_RESEARCH_DOC}",
                "Write a comprehensive technical review of FA3 covering: (1) the three core innovations and how they exploit H100 hardware, (2) the quantitative performance improvements with specific numbers, (3) accuracy tradeoffs with FP8, and (4) practical deployment considerations.",
                "Create a migration guide for a team currently running vLLM on A100s who wants to adopt FA3 on H100s. Include: prerequisites, step-by-step migration steps, validation procedure, rollback plan, and expected performance gains with realistic estimates.",
                "Based on the benchmarks in the paper, build a cost-benefit analysis for upgrading from A100 to H100 for an LLM serving workload that processes 500M tokens/day. Include hardware cost difference, throughput gain, and break-even timeline.",
                "Write GitHub-style issue descriptions for the top 3 engineering tasks required to fully integrate FA3 into vLLM. Each issue should include: title, problem statement, proposed solution, acceptance criteria, estimated complexity (S/M/L/XL), and dependencies.",
            ]
        },
    ]
}

# ---------------------------
# 4. Concurrent Session Runner
# ---------------------------
async def run_agent_session(session_id: int, scenario_type: str, args: argparse.Namespace) -> Dict[str, Any]:
    """Run a single multi-turn agent session using the LangGraph graph."""
    app = create_agent_graph(model_name=args.model, base_url=args.url)
    scenario = random.choice(SCENARIOS[scenario_type])

    session_start = time.perf_counter()
    turn_results = []
    messages: List[BaseMessage] = []

    logging.info(f"Session {session_id} [{scenario_type.upper()}] → '{scenario['name']}'")

    for i, user_input in enumerate(scenario["inputs"]):
        logging.info(f"Session {session_id} | Turn {i+1}/{len(scenario['inputs'])} | {user_input[:60]}...")

        messages.append(HumanMessage(content=user_input))
        turn_start = time.perf_counter()

        try:
            result_state = await app.ainvoke({"messages": messages})
            final_messages = result_state.get("messages", [])
            output_msg = final_messages[-1].content if final_messages else ""
            messages = final_messages  # carry forward full history
            success = True
            error = None
        except Exception as e:
            output_msg = ""
            success = False
            error = str(e)
            logging.error(f"Session {session_id} | Turn {i+1} | ERROR: {e}")

        duration = time.perf_counter() - turn_start
        context_chars = sum(len(m.content) for m in messages)

        turn_results.append({
            "turn": i + 1,
            "input": user_input,
            "output": output_msg,
            "duration": duration,
            "success": success,
            "error": error,
            "context_chars": context_chars,  # ~4 chars per token
            "approx_tokens": context_chars // 4,
        })

        if not success:
            break

    total_duration = time.perf_counter() - session_start

    return {
        "session_id": session_id,
        "type": scenario_type,
        "scenario": scenario["name"],
        "total_duration": total_duration,
        "turns": turn_results,
        "success": all(t["success"] for t in turn_results),
        "total_turns": len(turn_results),
        "max_context_tokens": max((t["approx_tokens"] for t in turn_results), default=0),
    }


# ---------------------------
# 5. Main Entry Point
# ---------------------------
async def main():
    parser = argparse.ArgumentParser(description="LangGraph Heavy Load Tester")
    parser.add_argument("--concurrency", type=int, default=3, help="Total concurrent sessions")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000/v1", help="LLM server base URL")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-0.8B", help="Model name")
    parser.add_argument("--serve-type", type=str, required=True, choices=["vllm", "sglang"],
                        help="Server type — used to prefix all output files")
    args = parser.parse_args()

    # Output files are prefixed with serve-type
    output_json = f"{args.serve_type}_agent_results.json"
    report_file = f"{args.serve_type}_agent_performance_report.md"
    log_file    = f"{args.serve_type}_agent_load_test.log"

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(fh)

    logging.info(f"🚀 Starting {args.concurrency} sessions → {args.serve_type.upper()} @ {args.url}")

    total_start = time.perf_counter()
    scenario_types = list(SCENARIOS.keys())

    tasks = [
        run_agent_session(i + 1, scenario_types[i % len(scenario_types)], args)
        for i in range(args.concurrency)
    ]
    results = list(await asyncio.gather(*tasks))
    total_elapsed = time.perf_counter() - total_start

    # ---- Summary stats ----
    successful = sum(1 for r in results if r["success"])
    total_turns = sum(r["total_turns"] for r in results)
    max_ctx     = max((r["max_context_tokens"] for r in results), default=0)

    stats_by_type: Dict[str, Any] = {}
    for stype in scenario_types:
        type_res = [r for r in results if r["type"] == stype]
        if not type_res:
            continue
        type_turns = [t for r in type_res for t in r["turns"]]
        stats_by_type[stype] = {
            "sessions": len(type_res),
            "success_rate": sum(1 for r in type_res if r["success"]) / len(type_res),
            "avg_turn_latency": sum(t["duration"] for t in type_turns) / len(type_turns) if type_turns else 0,
            "avg_context_tokens": sum(t["approx_tokens"] for t in type_turns) / len(type_turns) if type_turns else 0,
        }

    # ---- Save JSON ----
    summary = {
        "serve_type": args.serve_type,
        "test_time": datetime.now().isoformat(),
        "concurrency": args.concurrency,
        "total_duration_sec": total_elapsed,
        "successful_sessions": successful,
        "total_turns": total_turns,
        "max_context_tokens_observed": max_ctx,
        "stats_by_type": stats_by_type,
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "sessions": results}, f, indent=2)

    # ---- Markdown Report ----
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"# 🧪 Heavy Load Test Report: {args.serve_type.upper()}\n\n")
        f.write(f"| Config | Value |\n|---|---|\n")
        f.write(f"| Server Type | `{args.serve_type.upper()}` |\n")
        f.write(f"| Concurrency | `{args.concurrency}` sessions |\n")
        f.write(f"| Total Time | `{total_elapsed:.2f}s` |\n")
        f.write(f"| Success Rate | `{successful}/{args.concurrency}` |\n")
        f.write(f"| Total Turns | `{total_turns}` |\n")
        f.write(f"| Max Context (tokens) | `~{max_ctx:,}` |\n\n")

        f.write("## 📊 Performance by Node Type\n\n")
        f.write("| Node | Sessions | Success % | Avg Turn Latency | Avg Context Tokens |\n")
        f.write("|---|---|---|---|---|\n")
        for stype, stats in stats_by_type.items():
            f.write(f"| {stype.upper()} | {stats['sessions']} | "
                    f"{stats['success_rate']*100:.1f}% | "
                    f"`{stats['avg_turn_latency']:.3f}s` | "
                    f"`~{int(stats['avg_context_tokens']):,}` |\n")

        f.write("\n## 📋 Session Details\n\n")
        for r in results:
            status = "✅" if r["success"] else "❌"
            f.write(f"### Session {r['session_id']} [{r['type'].upper()}] — {r['scenario']} {status}\n")
            f.write(f"- Total duration: `{r['total_duration']:.2f}s` | Max context: `~{r['max_context_tokens']:,} tokens`\n\n")
            f.write("| Turn | Input (truncated) | Latency | Tokens |\n|---|---|---|---|\n")
            for t in r["turns"]:
                trunc = t["input"][:80].replace("|", "\\|") + "..."
                f.write(f"| {t['turn']} | {trunc} | `{t['duration']:.2f}s` | `~{t['approx_tokens']:,}` |\n")
            f.write("\n")

        f.write("\n> [!NOTE]\n")
        f.write("> This test uses real multi-turn prompts with 10k+ token contexts and live web search via DDGS.\n")
        f.write(f"> Results are labeled `{args.serve_type}` for direct comparison with other serve types.\n")

    logging.info(f"📁 Report: {report_file} | JSON: {output_json} | Log: {log_file}")
    print(f"\n🎉 Done! {successful}/{args.concurrency} sessions succeeded in {total_elapsed:.1f}s")
    print(f"📄 Report → {report_file}")


if __name__ == "__main__":
    asyncio.run(main())

