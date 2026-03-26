import asyncio
import argparse
import time
import json
import logging
from datetime import datetime
from typing import List, Dict, Any
from agent_lib import LLMAgent

# ---------------------------
# 1️⃣ Setup Logging
# ---------------------------
log_file = "agent_load_test.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# ---------------------------
# 2️⃣ Test Scenarios
# ---------------------------

SCENARIOS = {
    "short": [
        {
            "name": "Quick Fact",
            "inputs": ["What is the current time?", "What is 15 * 15?"]
        },
        {
            "name": "Simple Greeting",
            "inputs": ["Hello! How are you?", "Tell me a short joke."]
        }
    ],
    "long": [
        {
            "name": "Deep Research",
            "inputs": [
                "Search for 'detailed research on AI latency'.",
                "Summarize the main points of that research.",
                "Compare these findings with current vLLM benchmarks."
            ]
        },
        {
            "name": "Document Analysis",
            "inputs": [
                "Get the context for 'technical_specs_v1'.",
                "Extract all the performance metrics from the document.",
                "Write a 3-paragraph summary of the document's conclusions."
            ]
        }
    ]
}

# ---------------------------
# 3️⃣ Concurrent Runner
# ---------------------------

async def run_agent_session(session_id: int, scenario_type: str, args: argparse.Namespace) -> Dict[str, Any]:
    agent = LLMAgent(model_name=args.model, base_url=args.url)
    
    # Pick a random scenario from the selected type
    import random
    scenario = random.choice(SCENARIOS[scenario_type])
    
    session_start = time.perf_counter()
    turn_results = []
    
    logging.info(f"Session {session_id} [{scenario_type.upper()}] started (Scenario: {scenario['name']})")
    
    for i, user_input in enumerate(scenario["inputs"]):
        logging.info(f"Session {session_id} | Turn {i+1} | Input: {user_input[:40]}...")
        res = await agent.run(user_input)
        
        turn_data = {
            "turn": i + 1,
            "input": user_input,
            "output": res.get("output", ""),
            "duration": res.get("duration", 0),
            "success": res.get("success", False),
            "error": res.get("error", None),
            "context_len": len(user_input) + len(res.get("output", "")) # Simplified context tracking
        }
        turn_results.append(turn_data)
        
        if not res.get("success"):
            logging.error(f"Session {session_id} | Turn {i+1} | ERROR: {res.get('error')}")
            break
            
    session_end = time.perf_counter()
    total_duration = session_end - session_start
    
    return {
        "session_id": session_id,
        "type": scenario_type,
        "scenario": scenario["name"],
        "total_duration": total_duration,
        "turns": turn_results,
        "success": all(t["success"] for t in turn_results)
    }

async def main():
    parser = argparse.ArgumentParser(description="Agent Load Tester for vLLM/SGLang")
    parser.add_argument("--concurrency", type=int, default=4, help="Total concurrent sessions")
    parser.add_argument("--mixed", action="store_true", help="Run 50/50 short and long tasks")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000/v1", help="Base URL")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-0.8B", help="Model name")
    parser.add_argument("--output", type=str, default="agent_test_results.json", help="Output file")
    
    args = parser.parse_args()
    
    logging.info(f"🚀 Starting {args.concurrency} agent sessions (Mixed: {args.mixed})...")
    
    total_start_time = time.perf_counter()
    
    tasks = []
    for i in range(args.concurrency):
        # Assign scenario types
        if args.mixed:
            stype = "short" if i % 2 == 0 else "long"
        else:
            stype = "short" # Default to short
            
        tasks.append(run_agent_session(i+1, stype, args))
        
    results = await asyncio.gather(*tasks)
    
    total_end_time = time.perf_counter()
    total_elapsed = total_end_time - total_start_time
    
    # ---------------------------
    # 4️⃣ Generate Report & Summary
    # ---------------------------
    
    successful_sessions = sum(1 for r in results if r["success"])
    total_turns = sum(len(r["turns"]) for r in results)
    
    # Metrics by Type
    stats_by_type = {}
    for stype in ["short", "long"]:
        type_res = [r for r in results if r["type"] == stype]
        if not type_res: continue
        
        type_turns = [t for r in type_res for t in r["turns"]]
        avg_lat = sum(t["duration"] for t in type_turns) / len(type_turns) if type_turns else 0
        stats_by_type[stype] = {
            "avg_latency": avg_lat,
            "count": len(type_res),
            "success_rate": sum(1 for r in type_res if r["success"]) / len(type_res)
        }

    summary = {
        "test_time": datetime.now().isoformat(),
        "concurrency": args.concurrency,
        "total_duration": total_elapsed,
        "stats_by_type": stats_by_type
    }
    
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "sessions": results}, f, indent=2)
        
    # Generate Markdown Report
    report_file = "agent_performance_report.md"
    with open(report_file, "w") as f:
        f.write("# 🧪 Mixed Load Test Report\n\n")
        f.write(f"- **Concurrency:** `{args.concurrency}` sessions (Mixed: `{args.mixed}`)\n")
        f.write(f"- **Total Elapsed:** `{total_elapsed:.2f}s`\n\n")
        
        f.write("## 📊 Performance by Task Type\n\n")
        f.write("| Type | Count | Avg Latency | Success Rate |\n")
        f.write("|---|---|---|---|\n")
        for stype, stats in stats_by_type.items():
            f.write(f"| {stype.upper()} | {stats['count']} | `{stats['avg_latency']:.3f}s` | {stats['success_rate']*100:.1f}% |\n")
            
        f.write("\n## 🔍 Insights\n")
        f.write("> [!NOTE]\n")
        f.write("> Concurrency is where vLLM scales better with mixed batch sizes. Long-context tasks typically see more stable latency in vLLM compared to simpler setups under high load.\n\n")

    print(f"\n🎉 Test complete! Report saved to {report_file}")

if __name__ == "__main__":
    asyncio.run(main())
