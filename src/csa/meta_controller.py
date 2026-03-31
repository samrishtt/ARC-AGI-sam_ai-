"""
MetaController — The orchestrator of the Cognitive Synthesis Architecture.

TASK 3 FIX: Every task starts with completely fresh message history.
            TokenCheck logging + truncation if estimated > 8000 tokens.
TASK 5 INTEGRATION: Object correspondence data fed into hypothesis formation.
TASK 6 INTEGRATION: TransformationLibrary candidates prepended to prompts.
"""

import os
import json
import datetime
from typing import Dict, Any, List, Tuple

from src.core.llm import LLMProvider, LLMResponse
from src.csa.models import RouteDecision, TaskDomain, TaskComplexity
from src.csa.router import IntentRouter
from src.csa.memory import WorkingMemory
from src.csa.coding import CodingHandler
from src.csa.vision import SymbolicGridParser, find_object_correspondence


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

    TASK 3 FIX: memory is recreated per task via _reset_for_task().
    No state carries over between tasks.
    """
    def __init__(self, primary_llm: LLMProvider):
        self.llm = primary_llm
        self.router = IntentRouter(llm=self.llm)
        self.coding_handler = CodingHandler(llm=self.llm)
        self.memory = WorkingMemory()
        
    def _reset_for_task(self, task_id: str = ""):
        """TASK 3 FIX: Completely reset all state before each task."""
        self.memory = WorkingMemory()  # Fresh memory, no carryover
        self.coding_handler = CodingHandler(llm=self.llm)  # Fresh coding handler
        if task_id:
            print(f"\n[Reset] Task {task_id}: all state cleared for fresh start.")

    def process_task(self, user_input: str, bypass_router: bool = False,
                     task_id: str = "") -> Dict[str, Any]:
        """
        The main entry point for a general task.
        
        Args:
            user_input: The task string or dict.
            bypass_router: If True, skip the LLM-based router and go straight
                           to visual_spatial pipeline. Saves one API call per task.
            task_id: Optional task ID for logging.
        """
        # TASK 3 FIX: Reset all state before each task
        self._reset_for_task(task_id)

        print(f"\n[Meta-Controller] Received Task: '{str(user_input)[:100]}...'")
        
        if bypass_router:
            decision = RouteDecision(
                domain=TaskDomain.VISUAL_SPATIAL,
                complexity=TaskComplexity.HIGH,
                reasoning="Router bypassed — task is known ARC-AGI data.",
                requires_tools=True
            )
            print(f"[Meta-Controller] Router BYPASSED — direct to visual_spatial pipeline.")
        else:
            router_input = str(user_input)[:1000] if len(str(user_input)) > 1000 else str(user_input)
            if "train" in router_input and "test" in router_input:
                router_input = "Determine domain for this ARC JSON grid matrix puzzle."
            decision: RouteDecision = self.router.route(router_input)
        
        print(f"[Meta-Controller] Decision: Domain={decision.domain.value}, "
              f"Complexity={decision.complexity.value}")
        print(f"[Meta-Controller] Reasoning: {decision.reasoning}")
        
        if decision.domain == TaskDomain.CONVERSATIONAL:
            result = self._handle_conversational(user_input, decision)
        elif decision.domain == TaskDomain.MATH_LOGIC:
            result = self._handle_math_logic(user_input, decision)
        elif decision.domain == TaskDomain.CODING:
            result = self._handle_coding(user_input, decision)
        elif decision.domain == TaskDomain.VISUAL_SPATIAL:
            result = self._handle_visual_spatial(user_input, decision, task_id=task_id)
        else:
            raise ValueError(f"Unknown domain: {decision.domain}")

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
        if decision.requires_tools:
            return self.coding_handler.execute_logic_task(task, decision, self.memory)
        else:
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
        return self.coding_handler.execute_logic_task(task, decision, self.memory)

    def _handle_visual_spatial(self, task: str, decision: RouteDecision,
                               task_id: str = "") -> Dict[str, Any]:
        """
        Full ARC-AGI visual-spatial pipeline:
        1. Extract grids from the task (or structured data).
        2. Parse all training pairs into symbolic graphs.
        3. Compute object correspondences (TASK 5).
        4. Accumulate observations across pairs.
        5. Ask the LLM for a hypothesis describing the transformation rule.
        6. Use the CodingHandler sandbox to generate & verify a transform() function.
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

            # Step 3 (TASK 5): Compute object correspondences
            all_correspondences = []
            for inp, out in training_pairs:
                try:
                    corr = find_object_correspondence(inp, out)
                    all_correspondences.append(corr)
                except Exception as e:
                    all_correspondences.append([])
                    print(f"[Vision] Object correspondence failed: {e}")

            # Step 4: Accumulate observations into memory
            for pair_info in parsed_data["pairs"]:
                idx = pair_info["pair_index"]
                changes = pair_info["observed_changes"]
                self.memory.add_observation(idx, f"Object delta: {changes['object_count_delta']}")
                self.memory.add_observation(idx, f"Dim changed: {changes['dimension_changed']}")
                self.memory.add_observation(idx, f"In dims: {changes['input_dimensions']}, Out dims: {changes['output_dimensions']}")

            # Step 5: Hypothesis formation with object correspondence data
            hypothesis = self._form_hypothesis(
                parsed_data,
                training_pairs=training_pairs,
                correspondences=all_correspondences,
                task_id=task_id
            )
            self.memory.add_step("Hypothesis", hypothesis)

            # Step 6: Program-first solving
            print(f"[Meta-Controller] Hypothesis formed. Dispatching to CodingHandler sandbox...")
            result = self.coding_handler.execute_visual_task(
                training_pairs=training_pairs,
                test_grid=test_grid,
                decision=decision,
                memory=self.memory,
                task_id=task_id
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
    # Visual Pipeline Sub-Steps
    # -------------------------------------------------------------------------

    def _extract_grids(self, task: str) -> Tuple[List[Tuple], Any]:
        """
        Extracts training pairs and test grid from a task.
        Raises GridExtractionError if no grid data is found.
        """
        if isinstance(task, dict):
            try:
                training_pairs = [
                    (ex["input"], ex["output"]) for ex in task["train"]
                ]
                test_grid = task["test"][0]["input"]
                return training_pairs, test_grid
            except (KeyError, IndexError) as e:
                raise GridExtractionError(f"Malformed ARC JSON structure: {e}")

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

    def _form_hypothesis(self, parsed_data: Dict[str, Any],
                         training_pairs: List[Tuple] = None,
                         correspondences: List = None,
                         task_id: str = "") -> str:
        """
        Asks the LLM to describe the transformation rule observed
        across ALL training pairs before attempting code generation.
        
        TASK 3 FIX: TokenCheck — estimates tokens and truncates if > 8000.
        TASK 5: Includes object correspondence data in prompt.
        """
        compressed_data = {
            "pairs": parsed_data.get("pairs", [])
        }
        
        system_prompt = (
            "You are an ARC-AGI pattern analyst. You are given raw integer grids AND "
            "symbolic geometric graphs of input→output training pairs. Your job is to describe "
            "the single transformation rule that converts every input into its "
            "corresponding output. Be precise and concise. Focus on what changes "
            "and what stays the same."
        )
        
        # Include raw grids alongside symbolic analysis
        raw_grids_section = ""
        if training_pairs:
            raw_pairs = [{"input": inp, "output": out} for inp, out in training_pairs]
            raw_grids_section = (
                f"Raw training grids (integer matrices):\n"
                f"{json.dumps(raw_pairs, separators=(',', ':'))}\n\n"
            )
        
        # TASK 5: Include object correspondence data
        correspondence_section = ""
        if correspondences:
            correspondence_section = (
                f"Object correspondences: {json.dumps(correspondences, separators=(',', ':'))}\n\n"
            )
        
        user_prompt = (
            f"{raw_grids_section}"
            f"{correspondence_section}"
            f"Parsed symbolic graphs for ALL training pairs:\n\n"
            f"{json.dumps(compressed_data, separators=(',', ':'))}\n\n"
            f"Observations from memory:\n{json.dumps(self.memory.get_all_observations())}\n\n"
            f"Describe the transformation rule in one paragraph."
        )

        # TASK 3 FIX: TokenCheck with truncation
        messages_str = system_prompt + user_prompt
        estimated = len(messages_str) // 4
        print(f"[TokenCheck] Task {task_id} prompt: ~{estimated} tokens")
        
        if estimated > 8000:
            print(f"[TokenCheck] WARNING: {estimated} tokens exceeds 8000 limit. Truncating to training examples only.")
            # Truncate: only keep raw grids (no symbolic data or correspondences)
            if training_pairs:
                raw_pairs = [{"input": inp, "output": out} for inp, out in training_pairs]
                user_prompt = (
                    f"Raw training grids (integer matrices):\n"
                    f"{json.dumps(raw_pairs, separators=(',', ':'))}\n\n"
                    f"Describe the transformation rule in one paragraph."
                )
                # Re-estimate
                estimated = len(system_prompt + user_prompt) // 4
                print(f"[TokenCheck] After truncation: ~{estimated} tokens")
                # If still too large, truncate grids themselves
                if estimated > 8000:
                    print(f"[TokenCheck] Still too large. Using minimal prompt.")
                    # Take only first 2 training pairs
                    minimal_pairs = [{"input": inp, "output": out} for inp, out in training_pairs[:2]]
                    user_prompt = (
                        f"Training grids:\n{json.dumps(minimal_pairs, separators=(',', ':'))}\n\n"
                        f"Describe the transformation rule in one paragraph."
                    )

        response = self.llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.3
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
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "results.jsonl")

        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "input": str(user_input)[:200],
            "decision": decision.model_dump(),
            "status": result.get("status", "unknown"),
            "pipeline": result.get("pipeline", "unknown"),
        }

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
