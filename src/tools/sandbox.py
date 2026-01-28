import subprocess
import sys
import tempfile
import os
from .base import BaseTool, ToolResult

class PythonSandbox(BaseTool):
    def __init__(self):
        super().__init__(
            name="python_sandbox",
            description="Executes Python code in a separate process to ensure safety and isolation."
        )

    def execute(self, code: str) -> ToolResult:
        """
        Runs the provided python code and returns stdout/stderr.
        """
        # Create a temporary file to run
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
            tmp.write(code)
            tmp_path = tmp.name
            
        try:
            # Set up environment with current directory in PYTHONPATH
            env = os.environ.copy()
            cwd = os.getcwd()
            env["PYTHONPATH"] = f"{cwd}{os.pathsep}{env.get('PYTHONPATH', '')}"
            
            # Run the code
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=10,  # 10 second timeout for safety
                env=env
            )
            
            success = result.returncode == 0
            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"
                
            return ToolResult(success=success, output=output)
            
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output="Execution timed out (10s limit).")
        except Exception as e:
            return ToolResult(success=False, output=f"System Error: {str(e)}")
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
