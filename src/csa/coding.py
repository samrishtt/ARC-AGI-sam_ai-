import subprocess
import os
import tempfile
import json
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
        In a real application, you'd want Docker or WASM here.
        """
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                temp_file.write(code_string)
                file_path = temp_file.name

            # Running without shell to avoid injection
            result = subprocess.run(
                ['python', file_path],
                capture_output=True,
                text=True,
                timeout=10 # Hard cap so infinite loops don't hang AGI
            )

            os.remove(file_path)

            if result.returncode == 0:
                print(f"[Sandbox] Execution Successful")
                return True, result.stdout.strip()
            else:
                print(f"[Sandbox] Execution Failed: {result.stderr}")
                return False, result.stderr.strip()
                
        except subprocess.TimeoutExpired:
             return False, "Runtime Error: Execution exceeded time limit."
        except Exception as e:
            return False, f"Runtime Error: {str(e)}"


class CodingHandler:
    """
    Orchestrates the LLM with the PythonSandbox.
    If the code fails, it asks the LLM to reflect and try fixing it.
    """
    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def execute_logic_task(self, task_description: str, decision: RouteDecision, memory: WorkingMemory) -> Dict[str, Any]:
        """
        Generates code, runs it, and repeats until it works (or hits max retries).
        """
        # Load memory into the prompt context
        system_prompt = (
            "You are an expert Python coder for a Cognitive Synthesis Architecture. "
            "Write ONLY pure python code wrapped in ```python blocks. "
            "Print the final answer explicitly so it can be captured by standard output. "
            "You have access to standard libraries (math, collections, itertools)."
        )

        memory.set_context(task_description)

        # Loop for self-correction (Reflection Loop)
        max_attempts = 3
        
        current_prompt = f"Task: {task_description}\nPlease provide a python script that prints the final result out."

        for attempt in range(max_attempts):
             print(f"\n[CodingHandler] Attempt {attempt+1}/{max_attempts}...")
             
             response = self.llm.generate(
                 system_prompt=system_prompt,
                 user_prompt=current_prompt
             )

             # Extract code
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

             # Execute code
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
                 # Reflection! Tell the LLM what went wrong and let it try again.
                 memory.add_step("Running Sandbox", f"Error encountered: {output}")
                 current_prompt = (
                     f"Your previous code threw this error:\n{output}\n"
                     f"Please write a corrected python script to solve the original task: {task_description}"
                 )

        # If we reach here, we exhausted retries
        return {
             "status": "failed",
             "pipeline": "coding",
             "output": f"Exhausted {max_attempts} retries. Logic execution failed.\nLast error: {output}",
             "decision": decision.model_dump()
        }

    def execute_visual_task(self, training_pairs: List[Tuple], test_grid: Any,
                            decision: RouteDecision, memory: WorkingMemory) -> Dict[str, Any]:
        """
        Asks the LLM to write a Python function transform(grid) -> grid,
        then uses the sandbox to validate it against every training pair
        before submitting the answer for the test grid.
        """
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
            "3. Ensure your logic perfectly matches the described hypothesis."
        )

        memory.set_context("Visual-Spatial Transform Code Generation")

        max_attempts = 3
        # Show the LLM the raw training pairs so it can observe the pattern directly
        pairs_for_prompt = [
            {"input": inp, "output": out}
            for inp, out in training_pairs  # Provide ALL pairs to Claude for maximum accuracy
        ]
        hypothesis_summary = memory.get_summary()
        base_context = (
            f"Here are ALL the ARC training pairs showing the transformation:\n\n"
            f"{json.dumps(pairs_for_prompt, separators=(',', ':'))}\n\n"
            f"Observed hypothesis from pattern analysis:\n{hypothesis_summary}\n\n"
        )
        current_prompt = (
            base_context +
            f"Write the `transform(grid)` function that correctly maps every input to its output."
        )

        last_code_str = ""

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

            # Build a validation script that tests against ALL training pairs
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

            success, output = PythonSandbox.run_code(test_script)

            if success and "SUCCESS" in output:
                memory.add_step("Visual Sandbox", f"Code validated on all pairs. Output: {output}")
                return {
                    "status": "success",
                    "pipeline": "visual_spatial",
                    "output": f"Code validated against training pairs!\n{output}",
                    "decision": decision.model_dump()
                }
            else:
                memory.add_step("Visual Sandbox", f"Attempt {attempt+1} failed: {output}")
                # FIX: Retry prompt includes previous code + training pairs + hypothesis
                # so Claude can see exactly what it wrote and self-correct intelligently
                current_prompt = (
                    base_context +
                    f"Your PREVIOUS `transform` function was:\n```python\n{last_code_str}\n```\n\n"
                    f"It failed during validation with this error:\n{output}\n\n"
                    f"Carefully analyze the error relative to the training pairs above. "
                    f"Rewrite the `transform` function to fix these issues."
                )

        return {
            "status": "failed",
            "pipeline": "visual_spatial",
            "output": f"Exhausted {max_attempts} retries.\nLast error: {output}",
            "decision": decision.model_dump()
        }

    def _extract_python_code(self, raw_text: str) -> str:
        """Parses the generated response for code blocks."""
        if "```python" in raw_text:
            try:
                # Split at ```python and take everything until next ``` 
                blocks = raw_text.split("```python")[1:]
                first_block = blocks[0].split("```")[0]
                return first_block.strip()
            except IndexError:
                pass
        return ""
