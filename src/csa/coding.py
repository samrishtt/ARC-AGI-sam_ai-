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

from src.core.llm import LLMProvider
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

    def execute_visual_task(self, training_pairs: List[Tuple], test_grid: Any,
                            decision: RouteDecision, memory: WorkingMemory,
                            task_id: str = "") -> Dict[str, Any]:
        """
        Asks the LLM to write a Python function transform(grid) -> grid,
        then uses the sandbox to validate it against every training pair
        before submitting the answer for the test grid.
        
        TASK 4 FIXES:
        - Full traceback capture in sandbox
        - Error details in retry prompts
        - Output shape validation
        - Static checks before execution
        TASK 6: TransformationLibrary integration
        """
        # Import TransformationLibrary here to avoid circular imports
        from src.core.searcher import TransformationLibrary
        transform_lib = TransformationLibrary()

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
        
        # TASK 6: Get past successful solutions as examples
        library_section = ""
        # Try to determine a category from the hypothesis
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
            f"{library_section}"
        )

        # TASK 3 FIX: TokenCheck for code generation prompt
        estimated = len(system_prompt + base_context) // 4
        print(f"[TokenCheck] Task {task_id} code-gen prompt: ~{estimated} tokens")
        if estimated > 8000:
            print(f"[TokenCheck] WARNING: Truncating code-gen prompt from {estimated} tokens")
            # Reduce to just raw pairs + hypothesis
            base_context = (
                f"Training pairs:\n{json.dumps(pairs_for_prompt, separators=(',', ':'))}\n\n"
                f"Hypothesis:\n{hypothesis_summary}\n\n"
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
                memory.add_step("Visual Sandbox", f"Attempt {attempt+1} failed: {error_detail}")
                
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
        """Infer a transformation category from the hypothesis text."""
        hypothesis_lower = hypothesis.lower()
        if any(w in hypothesis_lower for w in ["rotate", "rotation", "90", "clockwise"]):
            return "rotation"
        elif any(w in hypothesis_lower for w in ["flip", "mirror", "reflect"]):
            return "reflection"
        elif any(w in hypothesis_lower for w in ["color", "recolor", "fill", "replace"]):
            return "recoloring"
        elif any(w in hypothesis_lower for w in ["scale", "resize", "enlarge", "shrink", "tile"]):
            return "scaling"
        elif any(w in hypothesis_lower for w in ["move", "shift", "translate", "slide"]):
            return "translation"
        elif any(w in hypothesis_lower for w in ["crop", "extract", "cut"]):
            return "cropping"
        elif any(w in hypothesis_lower for w in ["pattern", "repeat", "symmetr"]):
            return "pattern"
        else:
            return "general"
