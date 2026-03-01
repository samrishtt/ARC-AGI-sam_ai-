import os
import json
import datetime
from typing import Dict, Any, List, Tuple

from src.core.llm import LLMProvider, LLMResponse
from src.csa.models import RouteDecision, TaskDomain, TaskComplexity
from src.csa.router import IntentRouter
from src.csa.memory import WorkingMemory
from src.csa.coding import CodingHandler
from src.csa.vision import SymbolicGridParser


# --- Custom Exceptions for the Visual Pipeline ---
class GridExtractionError(Exception):
    """Raised when no grid can be extracted from the task string."""
    pass

class GridParsingError(Exception):
    """Raised when the SymbolicGridParser fails to parse a grid."""
    pass

class HypothesisFormationError(Exception):
    """Raised when the LLM fails to form a hypothesis."""
    pass


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
        # FIX #2: Memory persists across calls instead of being recreated per task
        self.memory = WorkingMemory()
        
    def process_task(self, user_input: str) -> Dict[str, Any]:
        """
        The main entry point for a general task.
        """
        print(f"\n[Meta-Controller] Received Task: '{user_input[:100]}...'")
        
        # Step 1: Route the Intent
        # Truncate input severely for the router to save tokens on huge ARC grids
        router_input = str(user_input)[:1000] if len(str(user_input)) > 1000 else str(user_input)
        if "train" in router_input and "test" in router_input:
            router_input = "Determine domain for this ARC JSON grid matrix puzzle."

        decision: RouteDecision = self.router.route(router_input)
        
        print(f"[Meta-Controller] Decision: Domain={decision.domain.value}, "
              f"Complexity={decision.complexity.value}")
        print(f"[Meta-Controller] Reasoning: {decision.reasoning}")
        
        # Step 2: Dispatch based on routing decision
        if decision.domain == TaskDomain.CONVERSATIONAL:
            result = self._handle_conversational(user_input, decision)
            
        elif decision.domain == TaskDomain.MATH_LOGIC:
            result = self._handle_math_logic(user_input, decision)
            
        elif decision.domain == TaskDomain.CODING:
            result = self._handle_coding(user_input, decision)
            
        elif decision.domain == TaskDomain.VISUAL_SPATIAL:
            result = self._handle_visual_spatial(user_input, decision)
            
        else:
            raise ValueError(f"Unknown domain: {decision.domain}")

        # FIX #10: Append result log after every process_task call
        self._log_result(user_input, decision, result)
        return result

    # -------------------------------------------------------------------------
    # Pipeline Handlers
    # -------------------------------------------------------------------------

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
        if decision.requires_tools:
            return self.coding_handler.execute_logic_task(task, decision, self.memory)
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
        return self.coding_handler.execute_logic_task(task, decision, self.memory)

    def _handle_visual_spatial(self, task: str, decision: RouteDecision) -> Dict[str, Any]:
        """
        Full ARC-AGI visual-spatial pipeline:
        1. Extract grids from the task (or structured data).
        2. Parse all training pairs into symbolic graphs.
        3. Accumulate observations across pairs.
        4. Ask the LLM for a hypothesis describing the transformation rule.
        5. Use the CodingHandler sandbox to generate & verify a transform() function.
        """
        self.memory.clear()
        self.memory.set_context("ARC Visual-Spatial Reasoning")

        try:
            # Step 1: Extract training pairs and test grid
            training_pairs, test_grid = self._extract_grids(task)
            self.memory.add_step("Grid Extraction", f"Extracted {len(training_pairs)} training pairs")

            # Step 2: Parse all pairs into symbolic graphs
            parser = SymbolicGridParser()
            parsed_data = self._parse_grids(parser, training_pairs)
            self.memory.add_step("Grid Parsing", f"Parsed {len(parsed_data['pairs'])} symbolic pair graphs")

            # Step 3: Accumulate observations into memory
            for pair_info in parsed_data["pairs"]:
                idx = pair_info["pair_index"]
                changes = pair_info["observed_changes"]
                self.memory.add_observation(idx, f"Object delta: {changes['object_count_delta']}")
                self.memory.add_observation(idx, f"Dim changed: {changes['dimension_changed']}")
                self.memory.add_observation(idx, f"In dims: {changes['input_dimensions']}, Out dims: {changes['output_dimensions']}")

            # Step 4: Hypothesis formation — ask the LLM to describe the rule
            hypothesis = self._form_hypothesis(parsed_data)
            self.memory.add_step("Hypothesis", hypothesis)

            # Step 5: Program-first solving — generate & validate transform()
            print(f"[Meta-Controller] Hypothesis formed. Dispatching to CodingHandler sandbox...")
            result = self.coding_handler.execute_visual_task(
                training_pairs=training_pairs,
                test_grid=test_grid,
                decision=decision,
                memory=self.memory
            )
            return result

        except GridExtractionError as e:
            return {
                "status": "error",
                "pipeline": "visual_spatial",
                "output": f"Grid Extraction Failed: {str(e)}",
                "decision": decision.model_dump()
            }
        except GridParsingError as e:
            return {
                "status": "error",
                "pipeline": "visual_spatial",
                "output": f"Grid Parsing Failed: {str(e)}",
                "decision": decision.model_dump()
            }
        except HypothesisFormationError as e:
            return {
                "status": "error",
                "pipeline": "visual_spatial",
                "output": f"Hypothesis Formation Failed: {str(e)}",
                "decision": decision.model_dump()
            }
        except Exception as e:
            return {
                "status": "error",
                "pipeline": "visual_spatial",
                "output": f"Unexpected Visual Pipeline Error: {str(e)}",
                "decision": decision.model_dump()
            }

    # -------------------------------------------------------------------------
    # Visual Pipeline Sub-Steps (Refactored from monolithic handler)
    # -------------------------------------------------------------------------

    def _extract_grids(self, task: str) -> Tuple[List[Tuple], Any]:
        """
        Extracts training pairs and test grid from a task.
        Accepts either:
          - A raw task string containing an ARC JSON structure
          - A dict already parsed from an ARC JSON file
        Raises GridExtractionError if no grid data is found.
        """
        # If the task is a dict (loaded from ARC JSON), extract directly
        if isinstance(task, dict):
            try:
                training_pairs = [
                    (ex["input"], ex["output"]) for ex in task["train"]
                ]
                test_grid = task["test"][0]["input"]
                return training_pairs, test_grid
            except (KeyError, IndexError) as e:
                raise GridExtractionError(f"Malformed ARC JSON structure: {e}")

        # Otherwise attempt to find a JSON blob in the string
        start_idx = task.find("{")
        end_idx = task.rfind("}") + 1

        if start_idx != -1 and end_idx > start_idx:
            try:
                arc_data = json.loads(task[start_idx:end_idx])
                training_pairs = [
                    (ex["input"], ex["output"]) for ex in arc_data["train"]
                ]
                test_grid = arc_data["test"][0]["input"]
                return training_pairs, test_grid
            except (json.JSONDecodeError, KeyError):
                pass

        # FIX #3: No silent dummy grid — raise an explicit error
        raise GridExtractionError(
            "No ARC grid data found in the task string. "
            "Expected a JSON object with 'train' and 'test' keys."
        )

    def _parse_grids(self, parser: SymbolicGridParser,
                     training_pairs: List[Tuple]) -> Dict[str, Any]:
        """Parses all training pairs into a combined symbolic graph."""
        try:
            return parser.parse_pairs(training_pairs)
        except Exception as e:
            raise GridParsingError(f"SymbolicGridParser failed: {str(e)}")

    def _form_hypothesis(self, parsed_data: Dict[str, Any]) -> str:
        """
        Asks the LLM to describe the transformation rule observed
        across UP TO 2 training pairs before attempting code generation.
        """
        # Compress data to avoid TPM limits limits (max 2 pairs, compressed JSON)
        compressed_data = {
            "pairs": parsed_data.get("pairs", [])[:2]
        }
        
        system_prompt = (
            "You are an ARC-AGI pattern analyst. You are given symbolic graphs "
            "of input→output training pairs. Your job is to describe "
            "the single transformation rule that converts every input into its "
            "corresponding output. Be precise and concise. Focus on what changes "
            "and what stays the same."
        )
        user_prompt = (
            f"Here are the parsed symbolic graphs for up to 2 training pairs:\n\n"
            f"{json.dumps(compressed_data, separators=(',', ':'))}\n\n"
            f"Observations from memory:\n{json.dumps(self.memory.get_all_observations())}\n\n"
            f"Describe the transformation rule in one paragraph."
        )

        response = self.llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )

        if not response.content or response.content.startswith("Error"):
            raise HypothesisFormationError(
                f"LLM failed to generate a hypothesis: {response.content}"
            )

        print(f"[Meta-Controller] Hypothesis: {response.content[:120]}...")
        return response.content

    # -------------------------------------------------------------------------
    # Result Logging
    # -------------------------------------------------------------------------

    def _log_result(self, user_input: Any, decision: RouteDecision, result: Dict[str, Any]):
        """
        Appends the RouteDecision + status + pipeline to logs/results.jsonl
        to build an evaluation dataset for measuring routing accuracy over time.
        """
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "results.jsonl")

        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "input": str(user_input)[:200],  # Truncate to avoid huge logs
            "decision": decision.model_dump(),
            "status": result.get("status", "unknown"),
            "pipeline": result.get("pipeline", "unknown"),
        }

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
