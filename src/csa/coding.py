"""
CodingHandler — Orchestrates LLM code generation with Python sandbox execution.

TASK 4 FIXES:
  - Full traceback capture on sandbox execution failures
  - Error details passed into retry prompts (attempt 2 and 3)
  - Output shape validation after execution
  - Pre-execution static checks (return keyword, hardcoded indices)
TASK 6 INTEGRATION:
  - TransformationLibrary used for recording successes and prepending candidates
"""

import subprocess
import os
import re
import tempfile
import json
import traceback
from typing import Tuple, Dict, Any, List

import numpy as np

from src.core.llm import LLMProvider
from src.core.searcher import ProgramSearch
from src.csa.models import RouteDecision
from src.csa.memory import WorkingMemory


class PythonSandbox:
    """
    Executes Python scripts safely.
    It isolates deterministic logic from LLM hallucinations.
    """
    @staticmethod
    def run_code(code_string: str) -> Tuple[bool, str]:
        """
        Runs the python code and returns (Success_Flag, Stderr_Or_Stdout).
        TASK 4 FIX: Captures full traceback on failure.
        """
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as temp_file:
                temp_file.write(code_string)
                file_path = temp_file.name

            result = subprocess.run(
                ['python', file_path],
                capture_output=True,
                text=True,
                timeout=15  # Slightly higher timeout for complex tasks
            )

            try:
                os.remove(file_path)
            except OSError:
                pass

            if result.returncode == 0:
                print(f"[Sandbox] Execution Successful")
                return True, result.stdout.strip()
            else:
                error_detail = result.stderr.strip()
                print(f"[Sandbox] Execution Failed: {error_detail[:200]}")
                return False, error_detail
                
        except subprocess.TimeoutExpired:
            try:
                os.remove(file_path)
            except (OSError, NameError):
                pass
            return False, "Runtime Error: Execution exceeded time limit (15s)."
        except Exception as e:
            error_detail = traceback.format_exc()
            return False, f"Runtime Error: {error_detail}"


class CodingHandler:
    """
    Orchestrates the LLM with the PythonSandbox.
    If the code fails, it asks the LLM to reflect and try fixing it.
    
    TASK 4 FIXES: Full error capture, shape validation, static checks.
    """
    def __init__(self, llm: LLMProvider):
        self.llm = llm

    # ─────────────────────────────────────────────────
    # TASK 4: Pre-execution static checks
    # ─────────────────────────────────────────────────
    def _static_checks(self, code_str: str) -> List[str]:
        """
        Run pre-execution static checks on code before sandbox execution.
        Returns list of warnings (non-blocking).
        """
        warnings = []
        
        # Check 1: Does the function return a value?
        if 'def transform' in code_str and 'return' not in code_str:
            warnings.append("WARNING: transform() function has no 'return' statement — will return None")
        
        # Check 2: Hardcoded grid indices like [3][4]
        hardcoded_pattern = re.findall(r'\[\d+\]\[\d+\]', code_str)
        if hardcoded_pattern:
            warnings.append(f"WARNING: Hardcoded grid indices detected: {hardcoded_pattern[:3]} — may not generalize")
        
        # Check 3: Is output assigned to a standard variable name?
        if 'def transform' in code_str:
            # Check that the function body has a return statement
            lines = code_str.split('\n')
            in_func = False
            has_return = False
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('def transform'):
                    in_func = True
                elif in_func and stripped.startswith('return'):
                    has_return = True
                    break
            if not has_return:
                warnings.append("WARNING: transform() may not return a grid — check control flow")
        
        for w in warnings:
            print(f"[StaticCheck] {w}")
        
        return warnings

    # ─────────────────────────────────────────────────
    # TASK 4: Output shape validation
    # ─────────────────────────────────────────────────
    @staticmethod
    def _validate_output_shape(output_grid, expected_shape: Tuple[int, int] = None) -> Dict[str, Any]:
        """
        Validates the output grid shape and type after execution.
        Returns {"valid": True/False, "reason": str}
        """
        if output_grid is None:
            return {"valid": False, "reason": "null_output"}
        
        if not isinstance(output_grid, list):
            return {"valid": False, "reason": f"wrong_type: got {type(output_grid).__name__}, expected list"}
        
        if len(output_grid) == 0:
            return {"valid": False, "reason": "empty_grid: output is empty list"}
        
        if not isinstance(output_grid[0], list):
            return {"valid": False, "reason": f"wrong_type: rows are {type(output_grid[0]).__name__}, expected list"}
        
        actual_shape = (len(output_grid), len(output_grid[0]) if output_grid[0] else 0)
        
        if expected_shape and actual_shape != expected_shape:
            return {
                "valid": False, 
                "reason": f"shape_mismatch: got {actual_shape} expected {expected_shape}"
            }
        
        return {"valid": True, "reason": "ok"}

    def execute_logic_task(self, task_description: str, decision: RouteDecision, 
                          memory: WorkingMemory) -> Dict[str, Any]:
        """Generates code, runs it, and repeats until it works (or hits max retries)."""
        system_prompt = (
            "You are an expert Python coder for a Cognitive Synthesis Architecture. "
            "Write ONLY pure python code wrapped in ```python blocks. "
            "Print the final answer explicitly so it can be captured by standard output. "
            "You have access to standard libraries (math, collections, itertools)."
        )

        memory.set_context(task_description)
        max_attempts = 3
        current_prompt = f"Task: {task_description}\nPlease provide a python script that prints the final result out."
        last_error = ""

        for attempt in range(max_attempts):
            print(f"\n[CodingHandler] Attempt {attempt+1}/{max_attempts}...")
            
            response = self.llm.generate(
                system_prompt=system_prompt,
                user_prompt=current_prompt
            )

            code_str = self._extract_python_code(response.content)

            if not code_str:
                memory.add_step("Extracting Code", "Failed to parse Python from response. Halting.")
                return {
                    "status": "failed",
                    "pipeline": "coding",
                    "output": "No python code generated.",
                    "decision": decision.model_dump()
                }

            memory.add_step("Writing Script", f"Generated script of {len(code_str)} chars.")
            
            # TASK 4: Static checks
            self._static_checks(code_str)

            success, output = PythonSandbox.run_code(code_str)

            if success:
                memory.add_step("Running Sandbox", f"Success. Output: {output}")
                return {
                    "status": "success",
                    "pipeline": "coding",
                    "output": f"Logic executed successfully via Sandbox. Final Result: \n\n{output}",
                    "decision": decision.model_dump()
                }
            else:
                # TASK 4 FIX: Full error detail passed into retry prompt
                last_error = output
                memory.add_step("Running Sandbox", f"Error encountered: {output}")
                current_prompt = (
                    f"Your previous attempt failed with this error:\n{output}\n\n"
                    f"Fix the exact bug above and return corrected Python code.\n\n"
                    f"Original task: {task_description}"
                )

        return {
            "status": "failed",
            "pipeline": "coding",
            "output": f"Exhausted {max_attempts} retries. Logic execution failed.\nLast error: {last_error}",
            "decision": decision.model_dump()
        }

    def _regenerate_hypothesis(self, training_pairs: list, failed_approach: str) -> str:
        """Ask LLM for a fresh hypothesis, explicitly telling it what didn't work."""
        pairs_for_prompt = [{"input": inp, "output": out} for inp, out in training_pairs]
        system_prompt = (
            "You are an ARC-AGI expert. A previous attempt to solve this puzzle used the "
            "following approach and FAILED. You must describe a COMPLETELY DIFFERENT "
            "transformation rule. Do not repeat the failed approach."
        )
        user_prompt = (
            f"Training pairs:\n{json.dumps(pairs_for_prompt, separators=(',', ':'))}\n\n"
            f"FAILED approach (do NOT use this):\n{failed_approach}\n\n"
            f"Describe an alternative transformation rule in one paragraph."
        )
        response = self.llm.generate(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.7)
        return response.content

    def execute_visual_task(self, training_pairs: List[Tuple], test_grid: Any,
                            decision: RouteDecision, memory: WorkingMemory,
                            task_id: str = "") -> Dict[str, Any]:
        """
        Asks the LLM to write a Python function transform(grid) -> grid,
        then uses the sandbox to validate it against every training pair
        before submitting the answer for the test grid.

        UPGRADES:
        - [1] A* DSL search runs FIRST (free, no API cost)
        - [2] Re-hypothesize on last retry with anti-hint
        - [3] Structured memory features injected into prompt
        - [4] Full traceback capture in sandbox
        - [5] Error details in retry prompts + output shape validation
        - [6] TransformationLibrary integration
        """
        # Import TransformationLibrary here to avoid circular imports
        from src.core.searcher import TransformationLibrary
        transform_lib = TransformationLibrary()

        # ── UPGRADE 2: A* DSL PRE-SOLVE (fast, free) ──────────────────────
        print(f"[CodingHandler] Attempting A* DSL search first (fast, free)...")
        try:
            searcher = ProgramSearch(max_depth=3)
            np_pairs = [(np.array(inp), np.array(out)) for inp, out in training_pairs]
            dsl_solution = searcher.solve(np_pairs, max_iterations=5000)
            if dsl_solution:
                print(f"[CodingHandler] A* found DSL solution: {dsl_solution}")
                dsl_code = (
                    "import sys, os\n"
                    "sys.path.append(os.getcwd())\n"
                    "from src.dsl.primitives import *\n\n"
                    f"def transform(grid):\n"
                    f"    return {dsl_solution.replace('x', 'grid')}\n"
                )
                astar_script = (
                    dsl_code +
                    f"\n\ntraining_pairs = {repr(training_pairs)}\n"
                    "import json\nall_ok = True\n"
                    "for inp, out in training_pairs:\n"
                    "    if transform(inp) != out:\n"
                    "        all_ok = False\n"
                    "        break\n"
                    "if all_ok:\n"
                    "    print('SUCCESS')\n"
                    "    import json\n"
                    f"    print(json.dumps(transform({repr(test_grid)})))\n"
                )
                success, output = PythonSandbox.run_code(astar_script)
                if success and "SUCCESS" in output:
                    memory.add_step("A* Search", f"Found DSL solution: {dsl_solution}")
                    return {
                        "status": "success",
                        "pipeline": "visual_spatial_astar",
                        "output": f"A* DSL solution found!\n{output}",
                        "decision": decision.model_dump()
                    }
        except Exception as e:
            print(f"[CodingHandler] A* search failed/timed out: {e}. Falling back to LLM.")
        # ── END A* PRE-SOLVE ───────────────────────────────────────────────

        system_prompt = (
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
            "3. Ensure your logic perfectly matches the described hypothesis.\n"
            "4. Your function MUST have a 'return' statement that returns a 2D list."
        )

        memory.set_context("Visual-Spatial Transform Code Generation")

        max_attempts = 3
        pairs_for_prompt = [
            {"input": inp, "output": out}
            for inp, out in training_pairs
        ]
        hypothesis_summary = memory.get_summary()

        # UPGRADE 5: Structured memory features
        structured_features = memory.get_structured_features()
        features_str = json.dumps(structured_features, indent=2)

        # TASK 6: Get past successful solutions as examples
        library_section = ""
        category = self._infer_category(hypothesis_summary)
        candidates = transform_lib.get_candidates(category, top_k=3)
        if candidates:
            library_section = (
                "\n\nHere are similar past solutions that worked for similar tasks:\n"
            )
            for c in candidates:
                library_section += f"--- Task {c['task_id']} ({c['category']}) ---\n{c['code']}\n\n"

        base_context = (
            f"Here are ALL the ARC training pairs showing the transformation:\n\n"
            f"{json.dumps(pairs_for_prompt, separators=(',', ':'))}\n\n"
            f"Observed hypothesis from pattern analysis:\n{hypothesis_summary}\n\n"
            f"Structured grid features extracted from training pairs:\n{features_str}\n\n"
            f"{library_section}"
        )

        # TASK 3 FIX: TokenCheck for code generation prompt
        estimated = len(system_prompt + base_context) // 4
        print(f"[TokenCheck] Task {task_id} code-gen prompt: ~{estimated} tokens")
        if estimated > 8000:
            print(f"[TokenCheck] WARNING: Truncating code-gen prompt from {estimated} tokens")
            base_context = (
                f"Training pairs:\n{json.dumps(pairs_for_prompt, separators=(',', ':'))}\n\n"
                f"Hypothesis:\n{hypothesis_summary}\n\n"
                f"Structured grid features:\n{features_str}\n\n"
            )

        current_prompt = (
            base_context +
            f"Write the `transform(grid)` function that correctly maps every input to its output."
        )

        last_code_str = ""
        last_error = ""

        for attempt in range(max_attempts):
            print(f"\n[CodingHandler] Visual Task Attempt {attempt+1}/{max_attempts}...")

            response = self.llm.generate(
                system_prompt=system_prompt,
                user_prompt=current_prompt
            )

            code_str = self._extract_python_code(response.content)

            if not code_str:
                return {
                    "status": "failed",
                    "pipeline": "visual_spatial",
                    "output": "No python code generated by LLM.",
                    "decision": decision.model_dump()
                }

            last_code_str = code_str

            # TASK 4: Pre-execution static checks
            warnings = self._static_checks(code_str)

            # Build a validation script that tests against ALL training pairs
            test_script = "import sys, json, os, traceback\n"
            test_script += "import numpy as np\n"
            test_script += "import math, collections, itertools\n"
            test_script += "from copy import deepcopy\n"
            test_script += "sys.path.append(os.getcwd())\n\n"
            test_script += code_str + "\n\n"
            test_script += f"training_pairs = {repr(training_pairs)}\n"
            test_script += f"test_grid = {repr(test_grid)}\n"
            test_script += "all_correct = True\n"
            test_script += "for idx, (in_grid, out_grid) in enumerate(training_pairs):\n"
            test_script += "    try:\n"
            test_script += "        result = transform(in_grid)\n"
            # TASK 4: Validate output shape
            test_script += "        if result is None:\n"
            test_script += "            print(f'Pair {idx} FAILED: null_output — transform() returned None')\n"
            test_script += "            all_correct = False\n"
            test_script += "            break\n"
            test_script += "        if not isinstance(result, list) or (len(result) > 0 and not isinstance(result[0], list)):\n"
            test_script += "            print(f'Pair {idx} FAILED: wrong_type — got {type(result).__name__}')\n"
            test_script += "            all_correct = False\n"
            test_script += "            break\n"
            test_script += "        expected_shape = (len(out_grid), len(out_grid[0]) if out_grid else 0)\n"
            test_script += "        actual_shape = (len(result), len(result[0]) if result else 0)\n"
            test_script += "        if actual_shape != expected_shape:\n"
            test_script += "            print(f'Pair {idx} FAILED: shape_mismatch: got {actual_shape} expected {expected_shape}')\n"
            test_script += "            all_correct = False\n"
            test_script += "            break\n"
            test_script += "        if result != out_grid:\n"
            test_script += "            print(f'Pair {idx} FAILED. Expected {out_grid}, got {result}')\n"
            test_script += "            all_correct = False\n"
            test_script += "            break\n"
            test_script += "    except Exception as e:\n"
            test_script += "        error_detail = traceback.format_exc()\n"
            test_script += "        print(f'Error on pair {idx}: {error_detail}')\n"
            test_script += "        all_correct = False\n"
            test_script += "        break\n"
            test_script += "if all_correct:\n"
            test_script += "    try:\n"
            test_script += "        test_result = transform(test_grid)\n"
            test_script += "        if test_result is None:\n"
            test_script += "            print('Error on test grid: transform() returned None')\n"
            test_script += "        else:\n"
            test_script += "            print('SUCCESS')\n"
            test_script += "            print(json.dumps(test_result))\n"
            test_script += "    except Exception as e:\n"
            test_script += "        error_detail = traceback.format_exc()\n"
            test_script += "        print(f'Error on test grid: {error_detail}')\n"

            success, output = PythonSandbox.run_code(test_script)

            if success and "SUCCESS" in output:
                memory.add_step("Visual Sandbox", f"Code validated on all pairs. Output: {output}")
                
                # TASK 6: Record successful transformation
                try:
                    transform_lib.record_success(
                        task_id=task_id,
                        code=last_code_str,
                        category=category
                    )
                    print(f"[Library] Recorded successful solution for task {task_id} (category: {category})")
                except Exception as e:
                    print(f"[Library] Failed to record: {e}")
                
                return {
                    "status": "success",
                    "pipeline": "visual_spatial",
                    "output": f"Code validated against training pairs!\n{output}",
                    "decision": decision.model_dump()
                }
            else:
                # TASK 4 FIX: Detailed error in retry prompt
                last_error = output
                error_detail = output if output else "Unknown error (no output captured)"
                print(f"\n[Sandbox Execution Error Verbatim]:\n{error_detail}\n")
                memory.add_step("Visual Sandbox", f"Attempt {attempt+1} failed: {error_detail}")

                # UPGRADE 3: Re-hypothesize on last retry (break single-hypothesis trap)
                if attempt == max_attempts - 1 and last_error:
                    print(f"[CodingHandler] Last attempt — regenerating hypothesis...")
                    fresh_hypothesis = self._regenerate_hypothesis(training_pairs, hypothesis_summary)
                    current_prompt = (
                        base_context.replace(hypothesis_summary, fresh_hypothesis) +
                        f"Previous approach failed with: {last_error}\n\n"
                        f"Write a NEW `transform(grid)` function based on this alternative hypothesis."
                    )
                else:
                    current_prompt = (
                        base_context +
                        f"Your PREVIOUS `transform` function was:\n```python\n{last_code_str}\n```\n\n"
                        f"Your previous attempt failed with this error:\n{error_detail}\n\n"
                        f"Fix the exact bug above and return corrected Python code.\n"
                        f"Make sure your function returns a 2D list with the correct shape."
                    )

        return {
            "status": "failed",
            "pipeline": "visual_spatial",
            "output": f"Exhausted {max_attempts} retries.\nLast error: {last_error}",
            "decision": decision.model_dump()
        }

    def _extract_python_code(self, raw_text: str) -> str:
        """Parses the generated response for code blocks."""
        if "```python" in raw_text:
            try:
                blocks = raw_text.split("```python")[1:]
                first_block = blocks[0].split("```")[0]
                return first_block.strip()
            except IndexError:
                pass
        return ""

    @staticmethod
    def _infer_category(hypothesis: str) -> str:
        """
        Score hypothesis against category keyword sets.
        Returns the highest-scoring category instead of first-match.
        """
        h = hypothesis.lower()

        categories = {
            "rotation":    ["rotat", "90", "clockwise", "counter-clock", "turn", "orient"],
            "reflection":  ["flip", "mirror", "reflect", "symmetric", "symmetry", "invert axis"],
            "recoloring":  ["color", "recolor", "fill", "replace", "hue", "shade", "map", "remap",
                            "palette", "value", "integer", "cell", "frequency", "count"],
            "scaling":     ["scale", "resize", "enlarge", "shrink", "tile", "repeat", "zoom",
                            "double", "halve", "magnif"],
            "translation": ["move", "shift", "translate", "slide", "offset", "displace"],
            "cropping":    ["crop", "extract", "cut", "trim", "bounding", "content"],
            "pattern":     ["pattern", "repeat", "tiling", "periodic", "motif", "stamp"],
            "filling":     ["flood", "fill interior", "enclose", "inside", "hole", "enclosed"],
        }

        scores = {}
        for category, keywords in categories.items():
            scores[category] = sum(1 for kw in keywords if kw in h)

        best = max(scores, key=scores.get)
        if scores[best] == 0:
            return "general"
        return best
