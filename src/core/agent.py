# NOTE: This file is legacy/unused. The active pipeline uses MetaController in src/csa/meta_controller.py
# Do not import this module — dependencies (tools.base, tools.sandbox, memory.store) are deprecated.
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import time
import os
from rich.console import Console
from ..tools.base import BaseTool
from ..tools.sandbox import PythonSandbox
from .llm import LLMProvider, OpenAIProvider, MockLLMProvider
from .planner import Planner
from .searcher import ProgramSearch
from .reflector import Reflector
from ..memory.store import MemoryModule
import numpy as np
import json

console = Console()

class Message(BaseModel):
    role: str
    content: str
    timestamp: float = Field(default_factory=time.time)

class AgentContext(BaseModel):
    history: List[Message] = []
    variables: dict = {}

class Agent:
    """
    The main reasoning agent.
    """
    tools: Dict[str, BaseTool] = {}
    llm: LLMProvider
    memory: MemoryModule
    planner: Planner
    searcher: ProgramSearch

    def __init__(self, name: str = "Sam"):
        self.name = name
        self.context = AgentContext()
        self._register_tools()
        
        # Initialize Components
        self.searcher = ProgramSearch()
        
        # Initialize LLM (Fall back to Mock if no key)
        if os.getenv("OPENAI_API_KEY"):
            self.llm = OpenAIProvider()
            console.print("[green]Connected to OpenAI Brain[/green]")
        else:
            self.llm = MockLLMProvider()
            console.print("[yellow]Using Mock Brain (No API Key found)[/yellow]")
            
        self.planner = Planner(self.llm)
        self.reflector = Reflector(self.llm)
        
        # Initialize Memory
        try:
            self.memory = MemoryModule()
            console.print("[green]Memory Module Online[/green]")
        except Exception as e:
            console.print(f"[red]Memory Init Failed: {e}[/red]")
            # Fallback mock memory could go here

        
    def _register_tools(self):
        sandbox = PythonSandbox()
        self.tools[sandbox.name] = sandbox


    def solve_task(self, input_grid: np.ndarray, expected_output: np.ndarray) -> str:
        """
        Main solver loop: Memory -> Search -> Learn
        """
        # 1. Memory Lookup & Macro Loading
        # Flatten grid or use simple query
        past_solutions = self.memory.retrieve("Solution:")
        macros = []
        for sol in past_solutions:
             if "Solution:" in sol:
                  macros.append(sol.split("Solution:")[1].strip())
        
        if macros:
             console.print(f"[blue]Loading {len(macros)} macros from Brain...[/blue]")
             self.searcher.load_macros(macros)
        
        # 2. Independent Search
        console.print("[yellow]Searching for solution...[/yellow]")
        solution_primitive = self.searcher.solve([(input_grid, expected_output)])
        
        # Reset searcher for next time (clean slate)
        self.searcher.primitives = dict(self.searcher.base_primitives)
        
        if solution_primitive:
            console.print(f"[green]Solution Found:[/green] {solution_primitive}")
            
            # 3. Learning (Reinforcement)
            self.memory.store(
                content=f"Solution: {solution_primitive}",
                metadata={"trigger_pattern": "exact_match"}  # simplified metadata
            )
            return solution_primitive
        
        # 3. Reflection on failure
        console.print("[red]No solution found. Reflecting...[/red]")
        hint = self.reflector.reflect([input_grid], [expected_output], list(self.searcher.base_primitives.keys()))
        console.print(f"[magenta]Brain Hint:[/magenta] {hint}")
        self.memory.store(f"Failure Analysis for task: {hint}")
            
        return "No solution found in basic primitives."

    def process(self, input_text: str) -> str:
        """
        Main entry point for processing user input.
        """
        self.context.history.append(Message(role="user", content=input_text))
        
        console.print(f"[bold blue]{self.name} is thinking...[/bold blue]")
        
        # Check for "SOLVE:" command to trigger independent solver mode
        if input_text.startswith("SOLVE:"):
             # Mock parsing of grid from text for demo
             # Real implementation would parse JSON grids
             return "Please use test harness for Grid Solving mode."

        plan = self._plan_response(input_text)
        response_content = self._execute_plan(plan)
        
        self.context.history.append(Message(role="assistant", content=response_content))
        return response_content

    def _plan_response(self, input_text: str) -> str:
        # Retrieve relevant memories
        context_docs = self.memory.retrieve(input_text)
        context_str = "\n".join(context_docs)
        
        # Use Planner (System 2)
        try:
            plan = self.planner.create_plan(input_text, context=context_str)
            console.print(f"[blue]Plan Generated:[/blue] {[s.action for s in plan.steps]}")
            
            # Simple heuristic mapping for now
            # In real AGI, we would execute each step. 
            # Here we check if any step requires code.
            if any(s.action == "TRANSFORM" or s.action == "VERIFY" for s in plan.steps):
                return "execute_code"
        except Exception as e:
            console.print(f"[red]Planning failed: {e}[/red]")
        
        # Fallback to direct routing if planning fails or decides no code needed
        if input_text.strip().lower().startswith("run:"):
             return "execute_code"

        return "chat"

    def _execute_plan(self, plan: str) -> str:
        last_msg = self.context.history[-1]
        
        if plan == "execute_code":
            # Check if this is a direct run command
            if last_msg.content.strip().lower().startswith("run:"):
                 code = last_msg.content.split(":", 1)[1].strip()
                 console.print(f"[yellow]Direct Execution Detected...[/yellow]")
            else:
                # Let the LLM generate the code
                prompt = f"Write python code to solve: {last_msg.content}. Output only the code block."
                code_response = self.llm.generate("You are a python coder. Output ONLY valid python code.", last_msg.content)
                # Clean up code fences if present
                code = code_response.content.replace("```python", "").replace("```", "").strip()
            
            console.print(f"[yellow]Executing code...[/yellow]")
            result = self.tools["python_sandbox"].execute(code)
            
            # Store success in memory
            if result.success:
                self.memory.store(f"Success for '{last_msg.content}':\n{code}")
                return f"Execution Successful:\n{result.output}"
            else:
                return f"Execution Failed:\n{result.output}"
                
        # Normal chat
        response = self.llm.generate("You are a helpful assistant.", last_msg.content)
        return response.content
