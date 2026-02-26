import json
from typing import List, Dict, Any

class MemoryStep:
    def __init__(self, action: str, result: str):
        self.action = action
        self.result = result

    def to_dict(self) -> Dict[str, str]:
        return {
            "action": self.action,
            "result": self.result
        }

class WorkingMemory:
    """
    Simulates short-term reasoning memory (the 'scratchpad').
    Instead of relying purely on the LLM's transformer context window,
    we explicitly track steps taken to solve a problem.
    """
    def __init__(self):
        self.context: str = ""
        self.steps: List[MemoryStep] = []
        self.variables: Dict[str, Any] = {}

    def set_context(self, task_description: str):
        self.context = task_description

    def add_step(self, action: str, result: str):
        self.steps.append(MemoryStep(action, result))

    def set_variable(self, key: str, value: Any):
        self.variables[key] = value

    def get_summary(self) -> str:
        summary = f"Task Context: {self.context}\n"
        summary += "Reasoning Steps Taken So Far:\n"
        for i, step in enumerate(self.steps):
            summary += f"[{i+1}] {step.action} -> {step.result}\n"
        return summary

    def dump(self) -> str:
        data = {
            "context": self.context,
            "variables": self.variables,
            "steps": [s.to_dict() for s in self.steps]
        }
        return json.dumps(data, indent=2)
