"""
CSA Benchmark Runner -- Evaluates 50 ARC training tasks.
Captures detailed per-task results including failure cases,
predicted vs expected grids, and failure component analysis.
"""
import os
import sys
import json
import time
import traceback
from typing import Dict, Any, List, Tuple

from dotenv import load_dotenv
load_dotenv()

from src.core.llm import MultiProviderLLM, LLMProvider
from src.csa.meta_controller import MetaController

def load_data(data_dir: str, limit: int = None) -> List[Dict]:
    """Loads ARC JSON files from the given directory."""
    tasks = []
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.json')])
    if limit:
        files = files[:limit]
    for f in files:
        with open(os.path.join(data_dir, f), 'r') as fp:
            data = json.load(fp)
            data['filename'] = f
            tasks.append(data)
    return tasks


def parse_predicted_grid(output_str: str):
    """Attempt to parse the predicted test grid from sandbox output."""
    if not output_str or "SUCCESS" not in output_str:
        return None
    lines = output_str.strip().split('\n')
    # The grid JSON is typically the last line after SUCCESS
    for line in reversed(lines):
        line = line.strip()
        if line.startswith('['):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def classify_failure(result: Dict[str, Any], task: Dict) -> Dict[str, str]:
    """Classify which component failed and what type of failure occurred."""
    output = result.get("output", "")
    status = result.get("status", "")
    pipeline = result.get("pipeline", "")
    
    # Determine failing component
    component = "unknown"
    failure_type = "unknown"
    
    if "Grid Extraction Failed" in output:
        component = "grid_extraction"
        failure_type = "parsing_error"
    elif "Grid Parsing Failed" in output:
        component = "vision_parser"
        failure_type = "parsing_error"
    elif "Hypothesis Formation Failed" in output:
        component = "llm_hypothesis"
        failure_type = "llm_error"
    elif "No python code generated" in output:
        component = "llm_codegen"
        failure_type = "no_code_generated"
    elif "Exhausted" in output and "retries" in output:
        component = "coding_sandbox"
        failure_type = "code_execution_failed"
    elif "Unexpected Visual Pipeline Error" in output:
        component = "visual_pipeline"
        failure_type = "unexpected_error"
    elif status == "success":
        # Code ran successfully but didn't match test output
        component = "llm_codegen"
        failure_type = "wrong_transformation_logic"
    elif "FAILED" in output:
        # Training pair validation failed
        component = "llm_codegen"
        if "got []" in output or "got [[]]" in output:
            failure_type = "wrong_shape"
        elif "Expected" in output and "got" in output:
            # Try to determine if shape or color issue
            failure_type = "wrong_transformation_logic"
        else:
            failure_type = "code_execution_failed"
    elif "Error on pair" in output:
        component = "coding_sandbox"
        failure_type = "runtime_error"
    elif "Error on test grid" in output:
        component = "coding_sandbox"
        failure_type = "runtime_error_test"
    else:
        component = "unknown"
        failure_type = "unclassified"
    
    return {
        "component": component,
        "failure_type": failure_type,
        "raw_output_snippet": output[:300]
    }


def grid_to_compact_str(grid, max_rows=5):
    """Convert a grid to a compact string for display."""
    if grid is None:
        return "None"
    if isinstance(grid, list):
        display = grid[:max_rows]
        s = "\n".join(f"    {row}" for row in display)
        if len(grid) > max_rows:
            s += f"\n    ... ({len(grid)} rows total)"
        return s
    return str(grid)


def run_benchmark(limit: int = 50):
    """Run the full CSA benchmark on ARC training tasks."""
    print("=" * 70)
    print("  CSA BENCHMARK RUNNER -- Structured Evaluation")
    print("=" * 70)

    llm = MultiProviderLLM()
    controller = MetaController(primary_llm=llm)

    # -- Show active model info ------------------------------------------
    primary_model   = MultiProviderLLM.PRIMARY_MODEL
    fallback_model  = MultiProviderLLM.FALLBACK_MODEL
    print(f"  Model (primary):   {primary_model}")
    print(f"  Model (fallback):  {fallback_model}")
    print(f"  Provider:          OpenRouter")
    print(f"  LLM Role in pipeline:")
    print(f"    [1] Hypothesis generation -- reads symbolic graph -> describes transform rule")
    print(f"    [2] Code generation       -- reads pairs + hypothesis -> writes transform()")
    print(f"    [3] Code retry/re-hypo    -- reads error + failed approach -> tries new angle")
    print(f"  A* search runs BEFORE LLM (free, no API cost, catches rotations/flips)")
    print("=" * 70)

    train_dir = os.path.join("data", "training")
    tasks = load_data(train_dir, limit=limit)

    total_tasks = len(tasks)
    print(f"\n  Loaded {total_tasks} tasks from {train_dir}/")
    print(f"  Starting evaluation...\n")
    
    # Results storage
    results_list = []
    solved_tasks = []
    failed_tasks = []
    error_tasks = []
    
    start_time = time.time()
    
    try:
        for i, task in enumerate(tasks):
            task_id = task['filename'].replace('.json', '')
            print(f"\n{'-' * 60}")
            print(f"  [{i+1}/{total_tasks}] Task: {task_id}")
            print(f"{'-' * 60}")
            
            expected_output = task["test"][0].get("output")
            task_start = time.time()
            
            try:
                result = controller.process_task(task, bypass_router=True)
                elapsed = time.time() - task_start
                
                predicted = None
                correct = False
                
                if result.get("status") == "success":
                    output_str = result.get("output", "")
                    predicted = parse_predicted_grid(output_str)
                    
                    if predicted is not None and expected_output is not None:
                        correct = (predicted == expected_output)
                
                active_provider = "Unknown provider"
                if hasattr(llm, "name"):
                    active_provider = llm.name
                if hasattr(llm, "current_provider_idx") and hasattr(llm, "provider_names") and getattr(llm, "provider_names"):
                    try:
                        active_provider = llm.provider_names[llm.current_provider_idx]
                    except IndexError:
                        pass
                
                entry = {
                    "task_id": task_id,
                    "status": result.get("status"),
                    "correct": correct,
                    "predicted": predicted,
                    "expected": expected_output,
                    "elapsed_seconds": round(elapsed, 2),
                    "output_snippet": result.get("output", "")[:500],
                    "pipeline": result.get("pipeline", ""),
                    "provider": active_provider,
                    "failure_info": None
                }
                
                if correct:
                    print(f"  [v] CORRECT ({elapsed:.1f}s) [Provider: {active_provider}]")
                    solved_tasks.append(entry)
                elif result.get("status") == "success":
                    # Code ran but test output didn't match
                    failure_info = classify_failure(result, task)
                    entry["failure_info"] = failure_info
                    print(f"  [x] INCORRECT -- passed training, failed test ({elapsed:.1f}s) [Provider: {active_provider}]")
                    failed_tasks.append(entry)
                else:
                    failure_info = classify_failure(result, task)
                    entry["failure_info"] = failure_info
                    print(f"  [x] FAILED -- {failure_info['component']}: {failure_info['failure_type']} ({elapsed:.1f}s) [Provider: {active_provider}]")
                    failed_tasks.append(entry)
                
                results_list.append(entry)
                
            except Exception as e:
                elapsed = time.time() - task_start
                entry = {
                    "task_id": task_id,
                    "status": "exception",
                    "correct": False,
                    "predicted": None,
                    "expected": expected_output,
                    "elapsed_seconds": round(elapsed, 2),
                    "output_snippet": f"EXCEPTION: {traceback.format_exc()[-500:]}",
                    "pipeline": "crashed",
                    "failure_info": {
                        "component": "system_crash",
                        "failure_type": "exception",
                        "raw_output_snippet": str(e)[:300]
                    }
                }
                error_tasks.append(entry)
                results_list.append(entry)
                print(f"  [x] EXCEPTION: {str(e)[:100]} ({elapsed:.1f}s)")
                
            time.sleep(3)
    except KeyboardInterrupt:
        print("\n\n[!] Benchmark interrupted by user. Saving midway results...")
        total_tasks = i + 1  # Adjust total tasks to how many were actually evaluated

    total_elapsed = time.time() - start_time
    
    # -- RESULTS SUMMARY --
    num_solved = len(solved_tasks)
    num_failed = len(failed_tasks)
    num_errors = len(error_tasks)
    accuracy = (num_solved / total_tasks * 100) if total_tasks > 0 else 0
    
    print("\n\n" + "=" * 70)
    print("  BENCHMARK RESULTS")
    print("=" * 70)
    print(f"  Total tasks attempted:  {total_tasks}")
    print(f"  Tasks solved (exact):   {num_solved}")
    print(f"  Tasks failed:           {num_failed}")
    print(f"  Tasks errored:          {num_errors}")
    print(f"  Accuracy:               {accuracy:.2f}%")
    print(f"  Total time:             {total_elapsed:.1f}s")
    print(f"  Avg time per task:      {total_elapsed/total_tasks:.1f}s")
    print("=" * 70)
    
    # -- TOP 5 FAILURE CASES --
    all_failures = failed_tasks + error_tasks
    top_5 = all_failures[:5]
    
    if top_5:
        print("\n\n  TOP 5 FAILURE CASES (Predicted vs Expected)")
        print("  " + "-" * 66)
        for j, f in enumerate(top_5):
            print(f"\n  [{j+1}] Task ID: {f['task_id']}")
            if f.get("failure_info"):
                print(f"      Component: {f['failure_info']['component']}")
                print(f"      Failure:   {f['failure_info']['failure_type']}")
            print(f"      Status:    {f['status']}")
            print(f"      Time:      {f['elapsed_seconds']}s")
            print(f"      Expected output (test):")
            print(grid_to_compact_str(f['expected']))
            print(f"      Predicted output:")
            print(grid_to_compact_str(f['predicted']))
            print(f"      Output snippet: {f['output_snippet'][:200]}")
    
    # -- FAILURE PATTERN ANALYSIS --
    component_counts = {}
    failure_type_counts = {}
    
    for f in all_failures:
        fi = f.get("failure_info", {}) or {}
        comp = fi.get("component", "unknown")
        ftype = fi.get("failure_type", "unknown")
        component_counts[comp] = component_counts.get(comp, 0) + 1
        failure_type_counts[ftype] = failure_type_counts.get(ftype, 0) + 1
    
    print("\n\n  FAILURE PATTERN ANALYSIS")
    print("  " + "-" * 66)
    print("\n  By Component:")
    for comp, count in sorted(component_counts.items(), key=lambda x: -x[1]):
        pct = count / len(all_failures) * 100 if all_failures else 0
        print(f"    {comp:30s}  {count:3d}  ({pct:.1f}%)")
    
    print("\n  By Failure Type:")
    for ftype, count in sorted(failure_type_counts.items(), key=lambda x: -x[1]):
        pct = count / len(all_failures) * 100 if all_failures else 0
        print(f"    {ftype:30s}  {count:3d}  ({pct:.1f}%)")
    
    # -- SAVE RESULTS --
    os.makedirs("logs", exist_ok=True)
    results_file = os.path.join("logs", "benchmark_results.json")
    
    summary = {
        "total_tasks": total_tasks,
        "solved": num_solved,
        "failed": num_failed,
        "errors": num_errors,
        "accuracy_pct": round(accuracy, 2),
        "total_time_seconds": round(total_elapsed, 2),
        "component_failures": component_counts,
        "failure_types": failure_type_counts,
        "per_task": results_list
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n  Full results saved to: {results_file}")
    print("=" * 70)
    
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="CSA Benchmark -- OpenRouter Qwen (qwq-32b primary)"
    )
    parser.add_argument(
        "--limit", type=int, default=50,
        help="Number of ARC tasks to evaluate (default: 50)"
    )
    args = parser.parse_args()
    run_benchmark(limit=args.limit)
