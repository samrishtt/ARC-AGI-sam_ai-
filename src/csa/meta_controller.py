from typing import Dict, Any
from src.core.llm import LLMProvider, LLMResponse
from src.csa.models import RouteDecision, TaskDomain, TaskComplexity
from src.csa.router import IntentRouter
from src.csa.memory import WorkingMemory
from src.csa.coding import CodingHandler

class MetaController:
    """
    The orchestrator of the Cognitive Synthesis Architecture.
    It receives tasks, consults the IntentRouter, and dispatches
    the task to the correct specialized subsystem.
    """
    def __init__(self, primary_llm: LLMProvider):
        self.llm = primary_llm
        self.router = IntentRouter(llm=self.llm)
        self.coding_handler = CodingHandler(llm=self.llm)
        
    def process_task(self, user_input: str) -> Dict[str, Any]:
        """
        The main entry point for a general task.
        """
        print(f"\n[Meta-Controller] Received Task: '{user_input}'")
        
        # Step 1: Route the Intent
        decision: RouteDecision = self.router.route(user_input)
        
        print(f"[Meta-Controller] Decision: Domain={decision.domain.value}, "
              f"Complexity={decision.complexity.value}")
        print(f"[Meta-Controller] Reasoning: {decision.reasoning}")
        
        # Step 2: Dispatch based on routing decision
        if decision.domain == TaskDomain.CONVERSATIONAL:
            return self._handle_conversational(user_input, decision)
            
        elif decision.domain == TaskDomain.MATH_LOGIC:
            return self._handle_math_logic(user_input, decision)
            
        elif decision.domain == TaskDomain.CODING:
            return self._handle_coding(user_input, decision)
            
        elif decision.domain == TaskDomain.VISUAL_SPATIAL:
            return self._handle_visual_spatial(user_input, decision)
            
        else:
            raise ValueError(f"Unknown domain: {decision.domain}")

    def _handle_conversational(self, task: str, decision: RouteDecision) -> Dict[str, Any]:
        """Simple pass-through to the LLM for chat/facts."""
        system_prompt = "You are a helpful AGI assistant engaging in standard conversation."
        
        if decision.complexity == TaskComplexity.HIGH:
            system_prompt += " The user has asked a complex question. Take your time to think it through deeply."
            
        response: LLMResponse = self.llm.generate(
            system_prompt=system_prompt,
            user_prompt=task
        )
        return {
            "status": "success",
            "pipeline": "conversational",
            "output": response.content,
            "decision": decision.model_dump()
        }

    def _handle_math_logic(self, task: str, decision: RouteDecision) -> Dict[str, Any]:
        """Uses the Python Sandbox to mathematically solve logic puzzles instead of guessing."""
        memory = WorkingMemory()
        
        if decision.requires_tools:
            return self.coding_handler.execute_logic_task(task, decision, memory)
        else:
            # Fallback for simple logic without code
            system_prompt = "You are a logical math genius. Use step-by-step reasoning to solve this puzzle."
            response: LLMResponse = self.llm.generate(
                system_prompt=system_prompt,
                user_prompt=task
            )
            return {
                "status": "success",
                "pipeline": "math_logic",
                "output": response.content,
                "decision": decision.model_dump()
            }

    def _handle_coding(self, task: str, decision: RouteDecision) -> Dict[str, Any]:
        """Uses Python execution sandbox explicitly for generating and validating code."""
        memory = WorkingMemory()
        return self.coding_handler.execute_logic_task(task, decision, memory)

    def _handle_visual_spatial(self, task: str, decision: RouteDecision) -> Dict[str, Any]:
        """Stub for handling visual grids, such as ARC-AGI."""
        return {
            "status": "pending_implementation",
            "pipeline": "visual_spatial",
            "output": "[Visual Grid Parser Stub] - Future integration point for ARC grids.",
            "decision": decision.model_dump()
        }
