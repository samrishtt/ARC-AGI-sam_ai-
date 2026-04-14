"""
Batch Evaluator for ARC-AGI using Anthropic Message Batches API (50% discount).

Architecture:
  Phase 1 (Batch): Submit all 50 hypothesis requests in one batch → wait for results
  Phase 2 (Batch): Using the hypotheses, submit all 50 code-gen requests → wait for results
  Phase 3 (Batch): For any failed tasks, submit retry requests with error context → wait
  Phase 4 (Local): Validate generated code in the local Python sandbox

Cost savings: 50% off both input and output tokens vs real-time API.
"""

import os
import sys
import json
import time
import logging
import datetime
from typing import Dict, Any, List, Tuple, Optional

from dotenv import load_dotenv
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import subprocess
import tempfile
import anthropic

# Direct import of vision module only (avoids heavy __init__.py chain)
import importlib
_vision_mod = importlib.import_module("src.csa.vision")
SymbolicGridParser = _vision_mod.SymbolicGridParser

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ─── Prompt Templates (same as meta_controller.py + coding.py) ──────────────

HYPOTHESIS_SYSTEM_PROMPT = (
    "You are an ARC-AGI pattern analyst. You are given raw integer grids AND "
    "symbolic geometric graphs of input→output training pairs. Your job is to describe "
    "the single transformation rule that converts every input into its "
    "corresponding output. Be precise and concise. Focus on what changes "
    "and what stays the same."
)

CODEGEN_SYSTEM_PROMPT = (
    "You are an expert Python coder for ARC-AGI puzzles.\n"
    "Write a Python function `def transform(grid):` that takes a 2D list of ints "
    "and returns a transformed 2D list of ints.\n"
    "You may optionally import and use the following helper functions from `src.dsl.primitives`:\n"
    "- rotate_cw(grid)\n"
    "- rotate_ccw(grid)\n"
    "- flip_horizontal(grid)\n"
    "- flip_vertical(grid)\n"
    "- fill_color(grid, target, new)\n"
    "- extract_color(grid, target)\n"
    "- crop_to_content(grid)\n"
    "- tile_grid(grid, v, h)\n"
    "- shift_grid(grid, row_shift, col_shift, bg_color=0)\n"
    "- draw_line(grid, r1, c1, r2, c2, color)\n"
    "- flood_fill(grid, r, c, replacement_color)\n"
    "- get_bounding_boxes(grid, bg_color=0) -> returns list of dicts with keys: top, bottom, left, right, color\n\n"
    "CRITICAL RULES:\n"
    "1. ONLY output the function implementation wrapped in ```python blocks.\n"
    "2. Do NOT include ANY testing code, test arrays, or prints.\n"
    "3. Ensure your logic perfectly matches the described hypothesis."
)


# ─── Data Loading ───────────────────────────────────────────────────────────

def load_arc_tasks(data_dir: str, limit: int = None) -> List[Dict]:
    """Loads ARC JSON files from given directory."""
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


# ─── Symbolic Parsing (local, no API cost) ──────────────────────────────────

def parse_task_locally(task: Dict) -> Dict[str, Any]:
    """
    Runs the SymbolicGridParser on all training pairs to produce
    the symbolic graph + observations, same as MetaController does.
    Returns everything needed to build the hypothesis prompt.
    """
    parser = SymbolicGridParser()
    training_pairs = [(ex["input"], ex["output"]) for ex in task["train"]]
    test_grid = task["test"][0]["input"]
    
    parsed_data = parser.parse_pairs(training_pairs)
    
    # Build observations string (same as meta_controller)
    observations = {}
    for pair_info in parsed_data["pairs"]:
        idx = pair_info["pair_index"]
        changes = pair_info["observed_changes"]
        observations[idx] = [
            f"Object delta: {changes['object_count_delta']}",
            f"Dim changed: {changes['dimension_changed']}",
            f"In dims: {changes['input_dimensions']}, Out dims: {changes['output_dimensions']}"
        ]
    
    return {
        "training_pairs": training_pairs,
        "test_grid": test_grid,
        "parsed_data": parsed_data,
        "observations": observations
    }


def build_hypothesis_prompt(parsed_info: Dict) -> str:
    """Builds the user prompt for hypothesis generation."""
    raw_pairs = [{"input": inp, "output": out} for inp, out in parsed_info["training_pairs"]]
    compressed_data = {"pairs": parsed_info["parsed_data"].get("pairs", [])}
    
    return (
        f"Raw training grids (integer matrices):\n"
        f"{json.dumps(raw_pairs, separators=(',', ':'))}\n\n"
        f"Parsed symbolic graphs for ALL training pairs:\n\n"
        f"{json.dumps(compressed_data, separators=(',', ':'))}\n\n"
        f"Observations from memory:\n{json.dumps(parsed_info['observations'])}\n\n"
        f"Describe the transformation rule in one paragraph."
    )


def build_codegen_prompt(parsed_info: Dict, hypothesis: str) -> str:
    """Builds the user prompt for code generation."""
    pairs_for_prompt = [
        {"input": inp, "output": out}
        for inp, out in parsed_info["training_pairs"]
    ]
    return (
        f"Here are ALL the ARC training pairs showing the transformation:\n\n"
        f"{json.dumps(pairs_for_prompt, separators=(',', ':'))}\n\n"
        f"Observed hypothesis from pattern analysis:\n{hypothesis}\n\n"
        f"Write the `transform(grid)` function that correctly maps every input to its output."
    )


def build_retry_prompt(parsed_info: Dict, hypothesis: str, prev_code: str, error: str) -> str:
    """Builds a retry prompt with error context for self-correction."""
    pairs_for_prompt = [
        {"input": inp, "output": out}
        for inp, out in parsed_info["training_pairs"]
    ]
    return (
        f"Here are ALL the ARC training pairs showing the transformation:\n\n"
        f"{json.dumps(pairs_for_prompt, separators=(',', ':'))}\n\n"
        f"Observed hypothesis from pattern analysis:\n{hypothesis}\n\n"
        f"Your PREVIOUS `transform` function was:\n```python\n{prev_code}\n```\n\n"
        f"It failed during validation with this error:\n{error}\n\n"
        f"Carefully analyze the error relative to the training pairs above. "
        f"Rewrite the `transform` function to fix these issues."
    )


# ─── Code Extraction & Sandbox Validation ───────────────────────────────────

def extract_python_code(raw_text: str) -> str:
    """Parses response for ```python blocks."""
    if "```python" in raw_text:
        try:
            blocks = raw_text.split("```python")[1:]
            first_block = blocks[0].split("```")[0]
            return first_block.strip()
        except IndexError:
            pass
    return ""


def validate_code_in_sandbox(code_str: str, training_pairs: List[Tuple], 
                              test_grid: Any) -> Tuple[bool, str]:
    """
    Builds & runs a validation script that tests the transform() function 
    against all training pairs, then outputs the test grid result.
    """
    test_script = "import sys, json, os\n"
    test_script += "sys.path.append(os.getcwd())\n\n"
    test_script += code_str + "\n\n"
    test_script += f"training_pairs = {repr(training_pairs)}\n"
    test_script += f"test_grid = {repr(test_grid)}\n"
    test_script += "all_correct = True\n"
    test_script += "for idx, (in_grid, out_grid) in enumerate(training_pairs):\n"
    test_script += "    try:\n"
    test_script += "        result = transform(in_grid)\n"
    test_script += "        if result != out_grid:\n"
    test_script += "            print(f'Pair {idx} FAILED. Expected {out_grid}, got {result}')\n"
    test_script += "            all_correct = False\n"
    test_script += "            break\n"
    test_script += "    except Exception as e:\n"
    test_script += "        print(f'Error on pair {idx}: {str(e)}')\n"
    test_script += "        all_correct = False\n"
    test_script += "        break\n"
    test_script += "if all_correct:\n"
    test_script += "    try:\n"
    test_script += "        print('SUCCESS')\n"
    test_script += "        print(json.dumps(transform(test_grid)))\n"
    test_script += "    except Exception as e:\n"
    test_script += "        print(f'Error on test grid: {str(e)}')\n"
    
    return _run_sandbox(test_script)


def _run_sandbox(code_string: str) -> Tuple[bool, str]:
    """Lightweight inline sandbox — runs Python code in a subprocess."""
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir='.') as f:
            f.write(code_string)
            file_path = f.name
        result = subprocess.run(
            ['python', file_path],
            capture_output=True, text=True, timeout=15
        )
        os.remove(file_path)
        if result.returncode == 0:
            return True, result.stdout.strip()
        else:
            return False, result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "Runtime Error: Execution exceeded time limit."
    except Exception as e:
        return False, f"Runtime Error: {str(e)}"


# ─── Anthropic Batch API Helpers ────────────────────────────────────────────

def create_batch(client: anthropic.Anthropic, requests: List[Dict],
                 description: str = "") -> str:
    """Submits a batch and returns the batch ID."""
    batch = client.messages.batches.create(requests=requests)
    print(f"  [Batch] Created: {batch.id} | {len(requests)} requests | Status: {batch.processing_status}")
    return batch.id


def wait_for_batch(client: anthropic.Anthropic, batch_id: str, 
                    poll_interval: int = 15) -> Dict[str, Any]:
    """
    Polls the batch until it reaches 'ended' status.
    Returns a dict mapping custom_id -> result content.
    """
    print(f"  [Batch] Waiting for {batch_id}...")
    
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        
        total = counts.processing + counts.succeeded + counts.errored + counts.canceled + counts.expired
        done = counts.succeeded + counts.errored + counts.canceled + counts.expired
        
        print(f"  [Batch] Progress: {done}/{total} | "
              f"✓{counts.succeeded} ✗{counts.errored} ⏳{counts.processing} "
              f"⊘{counts.canceled} ⏰{counts.expired}")
        
        if batch.processing_status == "ended":
            break
        
        time.sleep(poll_interval)
    
    # Retrieve results
    results = {}
    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id
        if result.result.type == "succeeded":
            message = result.result.message
            content = message.content[0].text if message.content else ""
            usage = {
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens
            }
            results[custom_id] = {"content": content, "usage": usage, "status": "succeeded"}
        else:
            results[custom_id] = {
                "content": "", 
                "usage": {},
                "status": result.result.type,
                "error": str(getattr(result.result, 'error', 'Unknown'))
            }
    
    return results


# ─── Main Batch Evaluation Pipeline ─────────────────────────────────────────

def batch_evaluate(limit: int = 50, model: str = "claude-sonnet-4-6",
                   max_retries: int = 2):
    """
    Multi-phase batch evaluation of ARC tasks using the Anthropic Batch API.
    
    Phase 1: Batch all hypothesis requests (50% off)
    Phase 2: Batch all code-gen requests (50% off)
    Phase 3: Batch retry requests for failed tasks (50% off)
    Phase 4: Local sandbox validation
    
    Args:
        limit: Number of ARC tasks to evaluate
        model: Anthropic model to use
        max_retries: Number of batch retry rounds for failed tasks
    """
    # Cost estimate at 50% batch pricing
    estimated_cost_full = limit * 7.73
    estimated_cost_batch = estimated_cost_full * 0.5
    
    print("=" * 70)
    print("  CSA Batch Evaluator — Anthropic Message Batches API (50% OFF)")
    print("=" * 70)
    print(f"  Tasks:          {limit}")
    print(f"  Model:          {model}")
    print(f"  Max retries:    {max_retries}")
    print(f"  Est. cost:      ~₹{estimated_cost_batch:.0f} (batch) vs ~₹{estimated_cost_full:.0f} (real-time)")
    print(f"  Savings:        ~₹{estimated_cost_full - estimated_cost_batch:.0f}")
    print("=" * 70)
    
    # Initialize Anthropic client
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key or len(api_key) < 10:
        print("[ERROR] No ANTHROPIC_API_KEY found in .env")
        return
    
    client = anthropic.Anthropic(api_key=api_key)
    
    # Load ARC tasks
    train_dir = os.path.join("data", "training")
    tasks = load_arc_tasks(train_dir, limit=limit)
    print(f"\n[1/5] Loaded {len(tasks)} ARC tasks from {train_dir}/")
    
    # ─── LOCAL PREPROCESSING (free) ─────────────────────────────────────
    print(f"\n[2/5] Parsing grids locally (symbolic analysis — no API cost)...")
    
    task_data = {}  # filename -> parsed info
    for i, task in enumerate(tasks):
        fname = task['filename']
        try:
            parsed_info = parse_task_locally(task)
            task_data[fname] = {
                "task": task,
                "parsed_info": parsed_info,
                "hypothesis": None,
                "code": None,
                "sandbox_output": None,
                "solved": False
            }
        except Exception as e:
            print(f"  [SKIP] {fname}: local parsing error: {e}")
    
    print(f"  → Successfully parsed {len(task_data)}/{len(tasks)} tasks")

    # ─── PHASE 0: A* DSL PRE-SOLVE (free, local) ────────────────────────
    print(f"\n[2.5/5] Phase 0: A* DSL pre-solve (free, no API cost)...")
    from src.core.searcher import ProgramSearch
    import numpy as np

    astar_solved = 0
    for fname, data in list(task_data.items()):
        try:
            training_pairs = data["parsed_info"]["training_pairs"]
            test_grid = data["parsed_info"]["test_grid"]
            np_pairs = [(np.array(inp), np.array(out)) for inp, out in training_pairs]

            searcher = ProgramSearch(max_depth=3)
            dsl_solution = searcher.solve(np_pairs, max_iterations=5000)

            if dsl_solution:
                dsl_code = (
                    "import sys, os\n"
                    "sys.path.append(os.getcwd())\n"
                    "from src.dsl.primitives import *\n\n"
                    f"def transform(grid):\n"
                    f"    return {dsl_solution.replace('x', 'grid')}\n"
                )
                success, output = validate_code_in_sandbox(dsl_code, training_pairs, test_grid)
                if success and "SUCCESS" in output:
                    data["hypothesis"] = f"A* DSL solution: {dsl_solution}"
                    data["code"] = dsl_code
                    data["sandbox_output"] = output
                    data["solved"] = True
                    lines = output.split('\n')
                    try:
                        llm_answer = json.loads(lines[-1])
                        data["test_correct"] = (llm_answer == data["task"]["test"][0]["output"])
                    except (json.JSONDecodeError, IndexError):
                        data["test_correct"] = False
                    astar_solved += 1
        except Exception:
            pass

    print(f"  → A* pre-solved: {astar_solved} tasks "
          f"(saved {astar_solved} hypothesis + codegen API calls)")

    # Filter out already-solved tasks from LLM batch phases
    unsolved_task_data = {f: d for f, d in task_data.items() if not d.get("solved")}

    # ─── PHASE 1: BATCH HYPOTHESIS REQUESTS ─────────────────────────────
    print(f"\n[3/5] Phase 1: Submitting {len(unsolved_task_data)} hypothesis requests (batch, 50% off)...")

    hypothesis_requests = []
    for fname, data in unsolved_task_data.items():
        user_prompt = build_hypothesis_prompt(data["parsed_info"])
        hypothesis_requests.append({
            "custom_id": f"hyp_{fname}",
            "params": {
                "model": model,
                "max_tokens": 1024,
                "temperature": 0.3,
                "system": HYPOTHESIS_SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user_prompt}]
            }
        })
    
    hyp_batch_id = create_batch(client, hypothesis_requests, "Phase 1: Hypotheses")
    hyp_results = wait_for_batch(client, hyp_batch_id)
    
    # Store hypotheses
    hyp_success = 0
    for fname, data in unsolved_task_data.items():
        custom_id = f"hyp_{fname}"
        result = hyp_results.get(custom_id, {})
        if result.get("status") == "succeeded" and result["content"]:
            data["hypothesis"] = result["content"]
            hyp_success += 1
        else:
            print(f"  [WARN] Hypothesis failed for {fname}: {result.get('error', 'empty')}")

    print(f"  → Hypotheses received: {hyp_success}/{len(unsolved_task_data)}")

    # ─── PHASE 2: BATCH CODE GENERATION REQUESTS ────────────────────────
    codegen_candidates = {f: d for f, d in unsolved_task_data.items() if d["hypothesis"]}
    print(f"\n[4/5] Phase 2: Submitting {len(codegen_candidates)} code-gen requests (batch, 50% off)...")
    
    codegen_requests = []
    for fname, data in codegen_candidates.items():
        user_prompt = build_codegen_prompt(data["parsed_info"], data["hypothesis"])
        codegen_requests.append({
            "custom_id": f"code_{fname}",
            "params": {
                "model": model,
                "max_tokens": 4096,
                "temperature": 0.0,
                "system": CODEGEN_SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user_prompt}]
            }
        })
    
    code_batch_id = create_batch(client, codegen_requests, "Phase 2: Code Generation")
    code_results = wait_for_batch(client, code_batch_id)
    
    # ─── PHASE 2b: LOCAL SANDBOX VALIDATION ─────────────────────────────
    print(f"\n  Validating generated code in local sandbox...")
    
    needs_retry = {}  # fname -> error message
    
    for fname, data in codegen_candidates.items():
        custom_id = f"code_{fname}"
        result = code_results.get(custom_id, {})
        
        if result.get("status") != "succeeded":
            needs_retry[fname] = f"API error: {result.get('error', 'unknown')}"
            continue
        
        code_str = extract_python_code(result["content"])
        if not code_str:
            needs_retry[fname] = "No Python code block found in LLM response"
            continue
        
        data["code"] = code_str
        
        # Run in sandbox
        success, output = validate_code_in_sandbox(
            code_str,
            data["parsed_info"]["training_pairs"],
            data["parsed_info"]["test_grid"]
        )
        
        data["sandbox_output"] = output
        
        if success and "SUCCESS" in output:
            data["solved"] = True
            # Check if test output matches
            lines = output.split('\n')
            try:
                llm_answer = json.loads(lines[-1])
                actual_answer = data["task"]["test"][0]["output"]
                if llm_answer == actual_answer:
                    data["test_correct"] = True
                else:
                    data["test_correct"] = False
            except (json.JSONDecodeError, IndexError):
                data["test_correct"] = False
        else:
            needs_retry[fname] = output or "Sandbox execution failed"
    
    solved_count = sum(1 for d in task_data.values() if d.get("solved"))
    print(f"  → Passed training validation: {solved_count}/{len(codegen_candidates)}")
    print(f"  → Need retry: {len(needs_retry)}")
    
    # ─── PHASE 3: BATCH RETRY ROUNDS ────────────────────────────────────
    for retry_round in range(max_retries):
        if not needs_retry:
            break
        
        print(f"\n[5/5] Phase 3 (Retry {retry_round + 1}/{max_retries}): "
              f"Submitting {len(needs_retry)} retry requests (batch, 50% off)...")
        
        retry_requests = []
        for fname, error_msg in needs_retry.items():
            data = task_data[fname]
            prev_code = data.get("code", "# No previous code generated")
            
            user_prompt = build_retry_prompt(
                data["parsed_info"],
                data["hypothesis"] or "No hypothesis available",
                prev_code,
                error_msg
            )
            
            retry_requests.append({
                "custom_id": f"retry{retry_round}_{fname}",
                "params": {
                    "model": model,
                    "max_tokens": 4096,
                    "temperature": 0.0,
                    "system": CODEGEN_SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": user_prompt}]
                }
            })
        
        retry_batch_id = create_batch(client, retry_requests, f"Phase 3: Retry {retry_round+1}")
        retry_results = wait_for_batch(client, retry_batch_id)
        
        # Validate retry results
        new_needs_retry = {}
        for fname in list(needs_retry.keys()):
            data = task_data[fname]
            custom_id = f"retry{retry_round}_{fname}"
            result = retry_results.get(custom_id, {})
            
            if result.get("status") != "succeeded":
                new_needs_retry[fname] = f"API error: {result.get('error', 'unknown')}"
                continue
            
            code_str = extract_python_code(result["content"])
            if not code_str:
                new_needs_retry[fname] = "No Python code block in retry response"
                continue
            
            data["code"] = code_str
            
            success, output = validate_code_in_sandbox(
                code_str,
                data["parsed_info"]["training_pairs"],
                data["parsed_info"]["test_grid"]
            )
            
            data["sandbox_output"] = output
            
            if success and "SUCCESS" in output:
                data["solved"] = True
                lines = output.split('\n')
                try:
                    llm_answer = json.loads(lines[-1])
                    actual_answer = data["task"]["test"][0]["output"]
                    data["test_correct"] = (llm_answer == actual_answer)
                except (json.JSONDecodeError, IndexError):
                    data["test_correct"] = False
            else:
                new_needs_retry[fname] = output or "Sandbox failed"
        
        needs_retry = new_needs_retry
        solved_count = sum(1 for d in task_data.values() if d.get("solved"))
        print(f"  → After retry {retry_round + 1}: {solved_count} solved, {len(needs_retry)} still failing")
    
    # ─── FINAL RESULTS ────────────────────────────────────────────────────
    total = len(task_data)
    solved = sum(1 for d in task_data.values() if d.get("solved"))
    test_correct = sum(1 for d in task_data.values() if d.get("test_correct"))
    
    print("\n" + "=" * 70)
    print("  BATCH EVALUATION COMPLETE")
    print("=" * 70)
    print(f"  Total tasks:                 {total}")
    print(f"  Training-validated (solved):  {solved}  ({solved/total*100:.1f}%)")
    print(f"  Test-correct (exact match):   {test_correct}  ({test_correct/total*100:.1f}%)")
    print(f"  Failed:                      {total - solved}")
    print("=" * 70)
    
    # Save detailed results
    results_dir = "logs"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"batch_eval_{timestamp}.json")
    
    summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model": model,
        "total_tasks": total,
        "training_validated": solved,
        "test_correct": test_correct,
        "accuracy_training": f"{solved/total*100:.2f}%",
        "accuracy_test": f"{test_correct/total*100:.2f}%",
        "per_task": {}
    }
    
    for fname, data in task_data.items():
        summary["per_task"][fname] = {
            "solved": data.get("solved", False),
            "test_correct": data.get("test_correct", False),
            "hypothesis": (data.get("hypothesis") or "")[:200],
            "code_length": len(data.get("code") or ""),
            "sandbox_output": (data.get("sandbox_output") or "")[:300]
        }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n  Detailed results saved to: {results_file}")
    print(f"  You saved ~₹{estimated_cost_full - estimated_cost_batch:.0f} with the Batch API! 🎉")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CSA Batch Evaluator (50% off)")
    parser.add_argument("--limit", type=int, default=50, help="Number of tasks to evaluate")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-6", help="Anthropic model")
    parser.add_argument("--retries", type=int, default=2, help="Max retry rounds for failed tasks")
    args = parser.parse_args()
    
    batch_evaluate(limit=args.limit, model=args.model, max_retries=args.retries)
