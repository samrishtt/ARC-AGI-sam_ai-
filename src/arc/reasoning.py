"""
Advanced Reasoning Engine for ARC-AGI

Implements multiple reasoning strategies:
1. Abstraction - Extract high-level concepts
2. Analogy - Use past solutions for similar problems
3. Decomposition - Break complex tasks into simpler ones
4. Synthesis - Combine primitive operations
5. Iterative Refinement - Progressive hypothesis testing
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from .pattern_engine import PatternEngine, TransformationHypothesis, PatternType, GridFeatures
from rich.console import Console
from rich.table import Table
import json
import hashlib

console = Console()


@dataclass
class ReasoningTrace:
    """Records the reasoning process for debugging and learning."""
    task_id: str
    strategy_used: str
    hypotheses_considered: List[str]
    solution_found: bool
    solution_description: Optional[str]
    execution_time_ms: float
    error: Optional[str] = None


@dataclass 
class SolverResult:
    """Result of attempting to solve an ARC task."""
    success: bool
    prediction: Optional[np.ndarray]
    reasoning_trace: ReasoningTrace
    confidence: float
    transformation_description: str


class ReasoningStrategy(Enum):
    """Available reasoning strategies."""
    PATTERN_MATCH = "pattern_match"    # Direct pattern recognition
    ANALOGY = "analogy"                # Use similar past solutions
    DECOMPOSE = "decompose"            # Break into sub-problems
    SYNTHESIZE = "synthesize"          # Combine primitives via search
    NEURAL = "neural"                  # Neural network (placeholder)


class ReasoningEngine:
    """
    Core reasoning engine that orchestrates solving ARC tasks.
    
    Uses a hierarchy of strategies:
    1. Try pattern matching first (fast, high confidence)
    2. Fall back to analogical reasoning
    3. Use decomposition for complex patterns
    4. Finally, try synthesis search
    """
    
    def __init__(self, dsl_primitives: Optional[Dict[str, Callable]] = None):
        self.pattern_engine = PatternEngine()
        self.solution_memory: Dict[str, TransformationHypothesis] = {}
        self.failure_memory: Dict[str, List[str]] = {}
        
        # Import DSL primitives
        if dsl_primitives is None:
            from ..dsl.primitives import dsl_registry
            self.dsl_primitives = dsl_registry
        else:
            self.dsl_primitives = dsl_primitives
        
        # Import advanced detectors
        try:
            from .advanced_detectors import AdvancedPatternDetectors
            self.advanced_detectors = AdvancedPatternDetectors.get_all_detectors()
        except ImportError:
            self.advanced_detectors = []
        
        # Strategy order (priority)
        self.strategies = [
            ReasoningStrategy.PATTERN_MATCH,
            "ADVANCED_DETECT",  # New strategy
            ReasoningStrategy.ANALOGY,
            ReasoningStrategy.DECOMPOSE,
            ReasoningStrategy.SYNTHESIZE
        ]
        
        # Composite transformations (learned combinations)
        self.composite_transforms: Dict[str, Callable] = {}
        
    def solve(
        self, 
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        task_id: str = "unknown"
    ) -> SolverResult:
        """
        Solve an ARC task given training examples and a test input.
        
        Args:
            train_examples: List of (input, output) training pairs
            test_input: The test input grid to produce output for
            task_id: Identifier for the task
            
        Returns:
            SolverResult with prediction and reasoning trace
        """
        import time
        start_time = time.time()
        
        hypotheses_tried = []
        
        # Try each strategy in order
        for strategy in self.strategies:
            result = self._try_strategy(
                strategy, train_examples, test_input, task_id
            )
            
            hypotheses_tried.append(f"{strategy.value}:{result.transformation_description}")
            
            if result.success:
                # Record solution for future use
                if result.prediction is not None:
                    self._record_solution(task_id, train_examples, result)
                
                result.reasoning_trace.hypotheses_considered = hypotheses_tried
                result.reasoning_trace.execution_time_ms = (time.time() - start_time) * 1000
                return result
        
        # All strategies failed
        elapsed = (time.time() - start_time) * 1000
        trace = ReasoningTrace(
            task_id=task_id,
            strategy_used="none",
            hypotheses_considered=hypotheses_tried,
            solution_found=False,
            solution_description=None,
            execution_time_ms=elapsed,
            error="All strategies exhausted"
        )
        
        # Record failure for learning
        self._record_failure(task_id, train_examples, hypotheses_tried)
        
        return SolverResult(
            success=False,
            prediction=None,
            reasoning_trace=trace,
            confidence=0.0,
            transformation_description="No solution found"
        )
    
    def _try_strategy(
        self,
        strategy,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        task_id: str
    ) -> SolverResult:
        """Try a specific reasoning strategy."""
        
        if strategy == ReasoningStrategy.PATTERN_MATCH:
            return self._pattern_match(train_examples, test_input, task_id)
        elif strategy == "ADVANCED_DETECT":
            return self._advanced_detect(train_examples, test_input, task_id)
        elif strategy == ReasoningStrategy.ANALOGY:
            return self._analogical_reasoning(train_examples, test_input, task_id)
        elif strategy == ReasoningStrategy.DECOMPOSE:
            return self._decomposition(train_examples, test_input, task_id)
        elif strategy == ReasoningStrategy.SYNTHESIZE:
            return self._synthesis(train_examples, test_input, task_id)
        else:
            return self._empty_result(task_id, str(strategy))
    
    def _pattern_match(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        task_id: str
    ) -> SolverResult:
        """
        Try direct pattern matching.
        Find a single transformation that explains all examples.
        """
        hypothesis = self.pattern_engine.find_consistent_transformation(train_examples)
        
        if hypothesis and hypothesis.transform_fn:
            try:
                prediction = hypothesis.transform_fn(test_input)
                
                trace = ReasoningTrace(
                    task_id=task_id,
                    strategy_used="pattern_match",
                    hypotheses_considered=[hypothesis.description],
                    solution_found=True,
                    solution_description=hypothesis.description,
                    execution_time_ms=0
                )
                
                return SolverResult(
                    success=True,
                    prediction=prediction,
                    reasoning_trace=trace,
                    confidence=hypothesis.confidence,
                    transformation_description=hypothesis.description
                )
            except Exception as e:
                pass
        
        return self._empty_result(task_id, "pattern_match")
    
    def _analogical_reasoning(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        task_id: str
    ) -> SolverResult:
        """
        Use similar past solutions.
        Compare feature signatures to find analogous tasks.
        """
        # Extract features from current task
        input_features = [self.pattern_engine.extract_features(inp) for inp, _ in train_examples]
        output_features = [self.pattern_engine.extract_features(out) for _, out in train_examples]
        
        # Look for similar patterns in solution memory
        for stored_id, stored_hypothesis in self.solution_memory.items():
            if stored_hypothesis.transform_fn is None:
                continue
                
            # Try the stored transformation
            try:
                works_for_all = True
                for inp, out in train_examples:
                    result = stored_hypothesis.transform_fn(inp)
                    if not np.array_equal(result, out):
                        works_for_all = False
                        break
                
                if works_for_all:
                    prediction = stored_hypothesis.transform_fn(test_input)
                    
                    trace = ReasoningTrace(
                        task_id=task_id,
                        strategy_used="analogy",
                        hypotheses_considered=[f"analogy_from:{stored_id}"],
                        solution_found=True,
                        solution_description=f"Analogical: {stored_hypothesis.description}",
                        execution_time_ms=0
                    )
                    
                    return SolverResult(
                        success=True,
                        prediction=prediction,
                        reasoning_trace=trace,
                        confidence=stored_hypothesis.confidence * 0.9,  # Slight penalty for analogy
                        transformation_description=f"Analogy from {stored_id}: {stored_hypothesis.description}"
                    )
            except Exception:
                continue
        
        return self._empty_result(task_id, "analogy")
    
    def _decomposition(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        task_id: str
    ) -> SolverResult:
        """
        Try to decompose the task into sub-tasks.
        Look for sequential transformations.
        """
        # Try common 2-step compositions
        compositions = [
            ("rotate_then_reflect", lambda x: np.fliplr(np.rot90(x, k=-1))),
            ("reflect_then_rotate", lambda x: np.rot90(np.fliplr(x), k=-1)),
            ("crop_then_scale", lambda x: np.repeat(np.repeat(self._crop(x), 2, axis=0), 2, axis=1)),
            ("color_then_crop", lambda x: self._crop(self._replace_colors(x, {1: 2}))),
        ]
        
        for name, transform in compositions:
            try:
                works_for_all = True
                for inp, out in train_examples:
                    result = transform(inp)
                    if not np.array_equal(result, out):
                        works_for_all = False
                        break
                
                if works_for_all:
                    prediction = transform(test_input)
                    
                    trace = ReasoningTrace(
                        task_id=task_id,
                        strategy_used="decompose",
                        hypotheses_considered=[name],
                        solution_found=True,
                        solution_description=name,
                        execution_time_ms=0
                    )
                    
                    return SolverResult(
                        success=True,
                        prediction=prediction,
                        reasoning_trace=trace,
                        confidence=0.85,
                        transformation_description=name
                    )
            except Exception:
                continue
        
        # Try building compositions dynamically
        result = self._build_composition(train_examples, test_input, task_id)
        if result.success:
            return result
        
        return self._empty_result(task_id, "decompose")
    
    def _build_composition(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        task_id: str,
        max_depth: int = 2
    ) -> SolverResult:
        """Build a composition of DSL primitives."""
        
        primitives = list(self.dsl_primitives.items())
        
        # Try all pairs of primitives
        for name1, func1 in primitives:
            for name2, func2 in primitives:
                try:
                    # Compose: func2(func1(x))
                    def composed(x, f1=func1, f2=func2):
                        return f2(f1(x))
                    
                    works_for_all = True
                    for inp, out in train_examples:
                        result = composed(inp)
                        if not np.array_equal(result, out):
                            works_for_all = False
                            break
                    
                    if works_for_all:
                        prediction = composed(test_input)
                        description = f"{name2}({name1}(x))"
                        
                        trace = ReasoningTrace(
                            task_id=task_id,
                            strategy_used="decompose/compose",
                            hypotheses_considered=[description],
                            solution_found=True,
                            solution_description=description,
                            execution_time_ms=0
                        )
                        
                        return SolverResult(
                            success=True,
                            prediction=prediction,
                            reasoning_trace=trace,
                            confidence=0.8,
                            transformation_description=description
                        )
                except Exception:
                    continue
        
        return self._empty_result(task_id, "compose")
    
    def _synthesis(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        task_id: str
    ) -> SolverResult:
        """
        Use A* search over DSL primitives.
        This is the fallback brute-force approach.
        Limited to prevent long runtimes.
        """
        from ..core.searcher import ProgramSearch
        
        # Use reduced depth and limited primitives for speed
        fast_primitives = {k: v for k, v in list(self.dsl_primitives.items())[:20]}
        
        searcher = ProgramSearch(
            primitives=fast_primitives,
            max_depth=2,  # Reduced from 3 for speed
            heuristic_weight=1.5  # More aggressive pruning
        )
        
        # Add iteration limit to searcher
        solution = searcher.solve(train_examples, max_iterations=500)
        
        if solution:
            # Build the transform function from the solution string
            try:
                ctx = {**self.dsl_primitives, "np": np}
                transform_fn = eval(f"lambda x: {solution}", ctx)
                prediction = transform_fn(test_input)
                
                trace = ReasoningTrace(
                    task_id=task_id,
                    strategy_used="synthesis",
                    hypotheses_considered=[solution],
                    solution_found=True,
                    solution_description=solution,
                    execution_time_ms=0
                )
                
                return SolverResult(
                    success=True,
                    prediction=prediction,
                    reasoning_trace=trace,
                    confidence=0.7,
                    transformation_description=solution
                )
            except Exception:
                pass
        
        return self._empty_result(task_id, "synthesis")
    
    def _crop(self, grid: np.ndarray) -> np.ndarray:
        """Crop to non-zero bounding box."""
        coords = np.argwhere(grid > 0)
        if len(coords) == 0:
            return grid
        min_r, min_c = coords.min(axis=0)
        max_r, max_c = coords.max(axis=0)
        return grid[min_r:max_r+1, min_c:max_c+1]
    
    def _replace_colors(self, grid: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
        """Replace colors according to mapping."""
        result = grid.copy()
        for src, dst in mapping.items():
            result = np.where(grid == src, dst, result)
        return result
    
    def _empty_result(self, task_id: str, strategy: str) -> SolverResult:
        """Create an empty/failed result."""
        trace = ReasoningTrace(
            task_id=task_id,
            strategy_used=strategy,
            hypotheses_considered=[],
            solution_found=False,
            solution_description=None,
            execution_time_ms=0
        )
        return SolverResult(
            success=False,
            prediction=None,
            reasoning_trace=trace,
            confidence=0.0,
            transformation_description=f"{strategy} failed"
        )
    
    def _record_solution(
        self, 
        task_id: str, 
        examples: List[Tuple[np.ndarray, np.ndarray]],
        result: SolverResult
    ):
        """Record successful solution for future analogical use."""
        # Create a hypothesis object for storage
        hypothesis = TransformationHypothesis(
            pattern_type=PatternType.COMPOSITE,
            description=result.transformation_description,
            confidence=result.confidence
        )
        
        # Try to extract transform function
        if "lambda" not in result.transformation_description:
            try:
                ctx = {**self.dsl_primitives, "np": np}
                hypothesis.transform_fn = eval(
                    f"lambda x: {result.transformation_description.split(':')[-1].strip()}", 
                    ctx
                )
            except:
                hypothesis.transform_fn = None
        
        self.solution_memory[task_id] = hypothesis
    
    def _record_failure(
        self, 
        task_id: str,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        hypotheses_tried: List[str]
    ):
        """Record failure for future learning."""
        self.failure_memory[task_id] = hypotheses_tried
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the reasoning engine."""
        return {
            "solutions_cached": len(self.solution_memory),
            "failures_recorded": len(self.failure_memory),
            "dsl_primitives": len(self.dsl_primitives),
            "composite_transforms": len(self.composite_transforms)
        }
