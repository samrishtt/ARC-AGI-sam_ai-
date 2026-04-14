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
        self.observations: Dict[int, List[str]] = {}

    def set_context(self, task_description: str):
        self.context = task_description

    def add_step(self, action: str, result: str):
        self.steps.append(MemoryStep(action, result))

    def set_variable(self, key: str, value: Any):
        self.variables[key] = value

    def add_observation(self, pair_index: int, note: str):
        if pair_index not in self.observations:
            self.observations[pair_index] = []
        self.observations[pair_index].append(note)

    def get_all_observations(self) -> Dict[int, List[str]]:
        return self.observations

    def clear(self):
        self.context = ""
        self.steps = []
        self.variables = {}
        self.observations = {}

    def get_summary(self) -> str:
        summary = f"Task Context: {self.context}\n"
        summary += "Reasoning Steps Taken So Far:\n"
        for i, step in enumerate(self.steps):
            summary += f"[{i+1}] {step.action} -> {step.result}\n"
        return summary

    def get_structured_features(self) -> dict:
        """
        Returns a structured dict of features derived from observations.
        This is what we actually feed into code-gen prompts — not the raw log.
        """
        features = {
            "dimension_changes": [],
            "object_count_changes": [],
            "color_observations": []
        }
        for pair_idx, obs_list in self.observations.items():
            for obs in obs_list:
                if "Dim changed: True" in obs:
                    features["dimension_changes"].append(f"Pair {pair_idx}: dimensions change")
                if "Object delta:" in obs:
                    try:
                        delta = int(obs.split("Object delta:")[1].strip())
                        features["object_count_changes"].append(f"Pair {pair_idx}: object delta={delta}")
                    except ValueError:
                        pass
        for key, val in self.variables.items():
            if "color" in key.lower():
                features["color_observations"].append(f"{key}: {val}")
        return features

    def dump(self) -> str:
        data = {
            "context": self.context,
            "variables": self.variables,
            "steps": [s.to_dict() for s in self.steps]
        }
        return json.dumps(data, indent=2)
