import subprocess
import os
import tempfile
from typing import Tuple, Dict, Any

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
