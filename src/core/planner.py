from typing import List
from pydantic import BaseModel
from .llm import LLMProvider

class plan_step(BaseModel):
    action: str
    description: str

class Plan(BaseModel):
    steps: List[plan_step]

class Planner:
    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def create_plan(self, goal: str, context: str = "") -> Plan:
        system_prompt = """You are a strategic planner for an ARC-AGI system.
        decompose the user's goal into clear, executable steps.
        Available actions:
        - ANALYZE: Patterns in input.
        - TRANSFORM: Apply a transformation (rotate, flip, color, etc).
        - VERIFY: Check if the result matches expectations.
        
        Output valid JSON conforming to the Plan model (list of steps).
        """
        
        prompt = f"Goal: {goal}\nContext: {context}"
        response = self.llm.generate(system_prompt, prompt)
        
        # Try to parse real JSON (Mock LLM now returns JSON string)
        import json
        try:
            data = json.loads(response.content)
            return Plan(**data)
        except:
             pass
        
        # Mocking a parsed plan for robustness in this iteration
        # This fallback is only reached if LLM output isn't valid JSON
        return Plan(steps=[
            plan_step(action="ANALYZE", description="Identify grid dimensions and colors"),
            plan_step(action="TRANSFORM", description="Attempt to find a transformation based on context"),
            plan_step(action="VERIFY", description="Check against training example")
        ])
