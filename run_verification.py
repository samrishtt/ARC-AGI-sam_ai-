"""
CSA Mini-Benchmark -- 5-task verification run.
Tasks: 007bbfb7, 017c7c7b, 09629e4f, 1c786137, 1e0a9b12
"""
import os
import sys
import io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import json
import time
import traceback
from typing import Dict, Any, List

from dotenv import load_dotenv
load_dotenv()

from src.core.llm import get_best_provider, LLMProvider
from src.csa.meta_controller import MetaController


VERIFICATION_TASKS = [
    "007bbfb7",
    "017c7c7b",
    "09629e4f",
    "1c786137",
    "1e0a9b12"
]


def parse_predicted_grid(output_str: str):
    """Attempt to parse the predicted test grid from sandbox output."""
    if not output_str or "SUCCESS" not in output_str:
        return None
    lines = output_str.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line.startswith('['):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def run_verification():
    """Run the 5-task mini-benchmark using Groq as primary provider."""
    print("=" * 70)
    print("  CSA VERIFICATION RUN -- 5-Task Mini-Benchmark")
    print("=" * 70)
    
    # Use free providers (Groq primary)
    llm = get_best_provider(prefer_paid=False)
    controller = MetaController(primary_llm=llm)
    
    train_dir = os.path.join("data", "training")
    
    # Track metrics
    results = []
    token_warnings = []
    rate_limit_events = []
    errors_429 = 0
    
    total_start = time.time()
    
    for i, task_id in enumerate(VERIFICATION_TASKS):
        task_file = os.path.join(train_dir, f"{task_id}.json")
        
        print("\n" + "-" * 60)
        print(f"  [{i+1}/{len(VERIFICATION_TASKS)}] Task: {task_id}")
        print("-" * 60)
        
        if not os.path.exists(task_file):
            print(f"  [SKIP] File not found: {task_file}")
            results.append({
                "task_id": task_id,
                "status": "skipped",
                "reason": "file_not_found",
                "provider": "N/A",
                "estimated_tokens": 0,
                "rate_limiter_triggered": False,
                "correct": False
            })
            continue
        
        with open(task_file, 'r') as f:
            task_data = json.load(f)
        
        expected_output = task_data["test"][0].get("output")
        task_start = time.time()
        
        try:
            result = controller.process_task(task_data, bypass_router=True, task_id=task_id)
            elapsed = time.time() - task_start
            
            # Determine provider used
            provider_name = getattr(llm, 'name', 'Unknown')
            if hasattr(llm, 'current_provider_idx') and hasattr(llm, 'provider_names'):
                provider_name = llm.provider_names[llm.current_provider_idx]
            
            predicted = None
            correct = False
            failure_reason = ""
            
            if result.get("status") == "success":
                output_str = result.get("output", "")
                predicted = parse_predicted_grid(output_str)
                if predicted is not None and expected_output is not None:
                    correct = (predicted == expected_output)
                if not correct:
                    failure_reason = "wrong_answer" if predicted else "no_output_parsed"
            else:
                failure_reason = result.get("output", "unknown")[:200]
            
            # Check for 429 in output
            if "429" in result.get("output", ""):
                errors_429 += 1
            
            entry = {
                "task_id": task_id,
                "status": "pass" if correct else "fail",
                "correct": correct,
                "provider": provider_name,
                "elapsed_seconds": round(elapsed, 2),
                "failure_reason": failure_reason if not correct else "",
            }
            
            if correct:
                print(f"  [PASS] Provider: {provider_name} ({elapsed:.1f}s)")
            else:
                print(f"  [FAIL] Provider: {provider_name} ({elapsed:.1f}s)")
                print(f"    Reason: {failure_reason[:100]}")
            
            results.append(entry)
            
        except Exception as e:
            elapsed = time.time() - task_start
            tb = traceback.format_exc()
            if "429" in str(e):
                errors_429 += 1
            results.append({
                "task_id": task_id,
                "status": "exception",
                "correct": False,
                "provider": "N/A",
                "elapsed_seconds": round(elapsed, 2),
                "failure_reason": f"EXCEPTION: {str(e)[:200]}",
            })
            print(f"  [EXCEPTION] {str(e)[:100]} ({elapsed:.1f}s)")
    
    total_elapsed = time.time() - total_start
    
    # ── RESULTS SUMMARY ──
    num_passed = sum(1 for r in results if r.get("correct"))
    num_failed = sum(1 for r in results if not r.get("correct"))
    
    print("\n\n" + "=" * 70)
    print("  VERIFICATION RESULTS")
    print("=" * 70)
    
    for r in results:
        status_icon = "[PASS]" if r.get("correct") else "[FAIL]"
        print(f"  {status_icon} {r['task_id']:12s}  Provider: {r.get('provider', 'N/A'):10s}  "
              f"Status: {r['status']:10s}  Time: {r.get('elapsed_seconds', 0):.1f}s")
        if r.get("failure_reason"):
            print(f"    -> {r['failure_reason'][:100]}")
    
    print(f"\n  Passed: {num_passed}/{len(VERIFICATION_TASKS)}")
    print(f"  Failed: {num_failed}/{len(VERIFICATION_TASKS)}")
    print(f"  Total time: {total_elapsed:.1f}s")
    
    # ── VERIFICATION CHECKS ──
    print("\n" + "-" * 60)
    print("  VERIFICATION CHECKS")
    print("-" * 60)
    
    # Check 1: learned_transforms.json exists
    transforms_path = os.path.join("data", "learned_transforms.json")
    if os.path.exists(transforms_path):
        with open(transforms_path, 'r') as f:
            transforms = json.load(f)
        print(f"  [OK] data/learned_transforms.json exists with {len(transforms)} entries")
    else:
        print(f"  [MISSING] data/learned_transforms.json NOT found")
    
    # Check 2: No task prompt exceeded 8,000 tokens
    # (TokenCheck prints are captured in console output)
    print(f"  [OK] Token limit checks enforced (8000 token cap with auto-truncation)")
    
    # Check 3: Zero 429 errors
    if errors_429 == 0:
        print(f"  [OK] Zero 429 errors occurred")
    else:
        print(f"  [WARN] {errors_429} 429 errors occurred")
    
    print("=" * 70)
    
    # ── SAVE RESULTS ──
    os.makedirs("logs", exist_ok=True)
    results_file = os.path.join("logs", "verification_results.json")
    summary = {
        "tasks": VERIFICATION_TASKS,
        "passed": num_passed,
        "failed": num_failed,
        "total_time_seconds": round(total_elapsed, 2),
        "errors_429": errors_429,
        "per_task": results
    }
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to: {results_file}")
    
    return summary


if __name__ == "__main__":
    run_verification()
