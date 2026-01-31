"""
Super-Enhanced Reasoning Engine for ARC-AGI

This is the main orchestrator that combines all strategies:
1. Direct Pattern Matching (fastest)
2. Parametric Color Operations (new)
3. Grid Subdivision Patterns (new)
4. Advanced Pattern Detectors (template, counting, etc.)
5. Object-Centric Reasoning
6. Deep Composition Search
7. Analogical Reasoning
8. LLM-Guided Hypothesis (when available)

Goal: Achieve 85%+ accuracy (human-level)
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import time
import itertools
import hashlib
import json
from collections import defaultdict

# Import our modules
from .pattern_engine import PatternEngine
from .enhanced_dsl import enhanced_dsl_registry
from .object_detector import ObjectDetector, ArcObject, GridAnalysis, analyze_grid
from .advanced_patterns import AdvancedPatternDetector, detect_patterns, DetectedPattern
from .color_ops import COLOR_OPS, infer_color_mapping, create_transform_from_mapping
from .grid_ops import SUBDIVISION_OPS, detect_subdivision_pattern

from rich.console import Console
console = Console()


class Strategy(Enum):
    """Solving strategies in order of preference."""
    EXACT_MATCH = "exact_match"           # Check if input == output
    PATTERN_MATCH = "pattern_match"       # Single DSL primitive
    PARAMETRIC_COLOR = "parametric_color" # Inferred color maps
    GRID_SUBDIVISION = "grid_subdivision" # Split/merge/overlay
    ADVANCED_PATTERN = "advanced_pattern" # Template, counting, etc.
    OBJECT_REASONING = "object_reasoning" # Object-centric
    COMPOSITION_2 = "composition_2"       # 2-step composition
    COMPOSITION_3 = "composition_3"       # 3-step composition
    ANALOGICAL = "analogical"             # Use past solutions
    DEEP_SEARCH = "deep_search"           # A* search up to 5 steps


@dataclass
class SolveResult:
    """Result of solving an ARC task."""
    success: bool
    prediction: Optional[np.ndarray]
    strategy: Strategy
    description: str
    confidence: float
    time_ms: float
    transform_chain: List[str] = field(default_factory=list)


@dataclass
class TaskSignature:
    """Signature of a task for analogical reasoning."""
    input_shape: Tuple[int, int]
    output_shape: Tuple[int, int]
    shape_ratio: Tuple[float, float]
    num_colors_in: int
    num_colors_out: int
    num_objects_in: int
    num_objects_out: int
    has_symmetry: bool
    
    def to_key(self) -> str:
        """Convert to hashable key."""
        return f"{self.input_shape}-{self.output_shape}-{self.num_colors_in}-{self.num_colors_out}"


class SuperReasoningEngine:
    """
    The ultimate ARC-AGI reasoning engine.
    
    Combines multiple strategies to achieve maximum accuracy.
    """
    
    def __init__(self):
        self.pattern_engine = PatternEngine()
        self.object_detector = ObjectDetector()
        self.advanced_detector = AdvancedPatternDetector()
        
        # Combine all DSLs
        self.dsl = {**enhanced_dsl_registry, **COLOR_OPS, **SUBDIVISION_OPS}
        
        # Memory for analogical reasoning
        self.solution_memory: Dict[str, Dict] = {}
        self.failure_memory: Set[str] = set()
        
        # Statistics
        self.stats = defaultdict(int)
        
    def solve(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        task_id: str = "unknown"
    ) -> SolveResult:
        """
        Solve an ARC task using hierarchical strategies.
        
        Tries strategies in order of speed and reliability.
        """
        start_time = time.time()
        
        # Try each strategy in order
        strategies = [
            (Strategy.EXACT_MATCH, self._try_exact_match),
            (Strategy.PATTERN_MATCH, self._try_pattern_match),
            (Strategy.PARAMETRIC_COLOR, self._try_parametric_color),
            (Strategy.GRID_SUBDIVISION, self._try_grid_subdivision),
            (Strategy.ADVANCED_PATTERN, self._try_advanced_patterns),
            (Strategy.OBJECT_REASONING, self._try_object_reasoning),
            (Strategy.COMPOSITION_2, self._try_composition_2),
            (Strategy.COMPOSITION_3, self._try_composition_3),
            (Strategy.ANALOGICAL, self._try_analogical),
            (Strategy.DEEP_SEARCH, self._try_deep_search),
        ]
        
        for strategy, method in strategies:
            try:
                result = method(train_examples, test_input, task_id)
                if result and result.success:
                    result.time_ms = (time.time() - start_time) * 1000
                    self.stats[strategy.value] += 1
                    
                    # Record solution for future analogical use
                    self._record_solution(task_id, train_examples, result)
                    
                    return result
            except Exception as e:
                # Log but continue to next strategy
                continue
        
        # All strategies failed
        elapsed = (time.time() - start_time) * 1000
        return SolveResult(
            success=False,
            prediction=None,
            strategy=Strategy.DEEP_SEARCH,
            description="No solution found",
            confidence=0.0,
            time_ms=elapsed
        )
    
    def _try_exact_match(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        task_id: str
    ) -> Optional[SolveResult]:
        """Check if input equals output (identity)."""
        all_identity = all(np.array_equal(inp, out) for inp, out in examples)
        
        if all_identity:
            return SolveResult(
                success=True,
                prediction=test_input.copy(),
                strategy=Strategy.EXACT_MATCH,
                description="Identity transformation",
                confidence=1.0,
                time_ms=0,
                transform_chain=["identity"]
            )
        
        return None
    
    def _try_pattern_match(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        task_id: str
    ) -> Optional[SolveResult]:
        """Try single DSL primitives."""
        # Prioritize geometric and simple ops first for speed
        geometric_ops = [k for k in self.dsl.keys() if "rotate" in k or "reflect" in k or "flip" in k]
        other_ops = [k for k in self.dsl.keys() if k not in geometric_ops]
        
        all_ops = geometric_ops + other_ops
        
        for name in all_ops:
            func = self.dsl[name]
            try:
                # Check if this primitive works for all examples
                all_match = True
                for inp, out in examples:
                    result = func(inp)
                    if result is None or not np.array_equal(result, out):
                        all_match = False
                        break
                
                if all_match:
                    prediction = func(test_input)
                    return SolveResult(
                        success=True,
                        prediction=prediction,
                        strategy=Strategy.PATTERN_MATCH,
                        description=f"Single primitive: {name}",
                        confidence=0.95,
                        time_ms=0,
                        transform_chain=[name]
                    )
            except Exception:
                continue
        
        return None

    def _try_parametric_color(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        task_id: str
    ) -> Optional[SolveResult]:
        """Try to infer a color mapping."""
        mapping = infer_color_mapping(examples)
        
        if mapping:
            transform = create_transform_from_mapping(mapping)
            
            # Double check (infer_color_mapping already checks consistency but good to be safe)
            if self._verify_transform(examples, transform):
                prediction = transform(test_input)
                desc = f"Color mapping: {mapping}"
                return SolveResult(
                    success=True,
                    prediction=prediction,
                    strategy=Strategy.PARAMETRIC_COLOR,
                    description=desc,
                    confidence=0.98,  # Very high confidence for explicit mappings
                    time_ms=0,
                    transform_chain=[f"color_map_{len(mapping)}"]
                )
        return None

    def _try_grid_subdivision(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        task_id: str
    ) -> Optional[SolveResult]:
        """Try grid subdivision patterns."""
        result = detect_subdivision_pattern(examples)
        
        if result:
            name, func = result
            prediction = func(test_input)
            return SolveResult(
                success=True,
                prediction=prediction,
                strategy=Strategy.GRID_SUBDIVISION,
                description=f"Grid pattern: {name}",
                confidence=0.9,
                time_ms=0,
                transform_chain=[name]
            )
        return None
    
    def _try_advanced_patterns(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        task_id: str
    ) -> Optional[SolveResult]:
        """Try advanced pattern detectors."""
        patterns = self.advanced_detector.detect_all_patterns(examples)
        
        for pattern in patterns:
            if pattern.transform_fn is not None and pattern.confidence > 0.7:
                try:
                    # Verify on all examples
                    if self._verify_transform(examples, pattern.transform_fn):
                        prediction = pattern.transform_fn(test_input)
                        return SolveResult(
                            success=True,
                            prediction=prediction,
                            strategy=Strategy.ADVANCED_PATTERN,
                            description=pattern.description,
                            confidence=pattern.confidence,
                            time_ms=0,
                            transform_chain=[pattern.description]
                        )
                except Exception:
                    continue
        
        return None
    
    def _try_object_reasoning(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        task_id: str
    ) -> Optional[SolveResult]:
        """Try object-centric reasoning strategies."""
        
        if not examples:
            return None
        
        inp, out = examples[0]
        in_analysis = analyze_grid(inp)
        out_analysis = analyze_grid(out)
        
        strategies = [
            self._object_filter_by_color,
            # Add more specific object strategies here
        ]
        
        for strategy_fn in strategies:
            try:
                result = strategy_fn(examples, test_input, in_analysis, out_analysis)
                if result:
                    return result
            except Exception:
                continue
        
        return None
    
    def _object_filter_by_color(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        in_analysis: GridAnalysis,
        out_analysis: GridAnalysis
    ) -> Optional[SolveResult]:
        """Filter objects by color criteria."""
        
        # Check if output contains only objects of a specific color
        if out_analysis.unique_colors and len(out_analysis.unique_colors) == 1:
            target_color = list(out_analysis.unique_colors)[0]
            if target_color == 0: return None 

            def keep_color(grid, color=target_color):
                return np.where(grid == color, color, 0)
            
            if self._verify_transform(examples, keep_color):
                prediction = keep_color(test_input)
                return SolveResult(
                    success=True,
                    prediction=prediction,
                    strategy=Strategy.OBJECT_REASONING,
                    description=f"Keep only color {target_color}",
                    confidence=0.85,
                    time_ms=0,
                    transform_chain=[f"keep_color_{target_color}"]
                )
        
        return None
    
    def _try_composition_2(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        task_id: str
    ) -> Optional[SolveResult]:
        """Try 2-step compositions of DSL primitives."""
        
        # Limit primitives for composition to core ones to avoid combinatorial explosion
        # We exclude the specialized color ops created dynamically
        core_primitives = [k for k in self.dsl.keys() if not k.startswith("replace_") and not k.startswith("swap_")]
        
        # Try all pairs
        for name1 in core_primitives:
            func1 = self.dsl[name1]
            for name2 in core_primitives:
                func2 = self.dsl[name2]
                try:
                    def composed(grid, f1=func1, f2=func2):
                        return f2(f1(grid))
                    
                    if self._verify_transform(examples, composed):
                        prediction = composed(test_input)
                        return SolveResult(
                            success=True,
                            prediction=prediction,
                            strategy=Strategy.COMPOSITION_2,
                            description=f"{name1} → {name2}",
                            confidence=0.85,
                            time_ms=0,
                            transform_chain=[name1, name2]
                        )
                except Exception:
                    continue
        
        return None
    
    def _try_composition_3(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        task_id: str
    ) -> Optional[SolveResult]:
        """Try 3-step compositions of DSL primitives."""
        
        # Strictly limit to most common primitives
        priority_primitives = [
            "rotate_cw", "rotate_ccw", "rotate_180",
            "reflect_h", "reflect_v", "transpose",
            "crop", "scale_2x", "tile_2x2",
            "invert", "gravity_down", "keep_largest",
            "fill_column", "fill_row", "outline",
            "remove_border", "pad_square", "deduplicate"
        ]
        
        subset = {k: self.dsl[k] for k in priority_primitives if k in self.dsl}
        
        subset_keys = list(subset.keys())
        for name1 in subset_keys:
            for name2 in subset_keys:
                for name3 in subset_keys:
                    try:
                        func1, func2, func3 = subset[name1], subset[name2], subset[name3]
                        def composed(grid, f1=func1, f2=func2, f3=func3):
                            return f3(f2(f1(grid)))
                        
                        if self._verify_transform(examples, composed):
                            prediction = composed(test_input)
                            return SolveResult(
                                success=True,
                                prediction=prediction,
                                strategy=Strategy.COMPOSITION_3,
                                description=f"{name1} → {name2} → {name3}",
                                confidence=0.8,
                                time_ms=0,
                                transform_chain=[name1, name2, name3]
                            )
                    except Exception:
                        continue
        
        return None
    
    def _try_analogical(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        task_id: str
    ) -> Optional[SolveResult]:
        """Use solutions from similar past tasks."""
        
        # Compute task signature
        sig = self._compute_signature(examples)
        sig_key = sig.to_key()
        
        # Look for similar solved tasks
        for stored_key, stored_data in self.solution_memory.items():
            if self._signatures_similar(sig_key, stored_key):
                # Try stored transform
                transform_chain = stored_data.get('transform_chain', [])
                
                if transform_chain:
                    try:
                        transform = self._chain_to_function(transform_chain)
                        if transform and self._verify_transform(examples, transform):
                            prediction = transform(test_input)
                            return SolveResult(
                                success=True,
                                prediction=prediction,
                                strategy=Strategy.ANALOGICAL,
                                description=f"Analogical: {' → '.join(transform_chain)}",
                                confidence=0.75,
                                time_ms=0,
                                transform_chain=transform_chain
                            )
                    except Exception:
                        continue
        
        return None
    
    def _try_deep_search(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        task_id: str
    ) -> Optional[SolveResult]:
        """A* search for longer compositions (up to 5 steps)."""
        
        # Only use core primitives to keep average branching factor reasonable
        core_primitives = [k for k in self.dsl.keys() if not k.startswith("replace_") and not k.startswith("swap_")]
        
        # Heuristic setup
        inp0, out0 = examples[0]
        
        visited = set()
        queue = [(0, [], test_input)]  # (depth, chain, current_grid)
        
        max_depth = 4
        max_iterations = 2000 # Increased iterations
        iterations = 0
        
        while queue and iterations < max_iterations:
            iterations += 1
            depth, chain, current = queue.pop(0)
            
            if depth > max_depth:
                continue
            
            # Hash current state to avoid cycles
            try:
                state_hash = hashlib.md5(current.tobytes()).hexdigest()
            except:
                continue
                
            if state_hash in visited:
                continue
            visited.add(state_hash)
            
            # Try each primitive
            for name in core_primitives:
                func = self.dsl[name]
                try:
                    new_grid = func(current)
                    new_chain = chain + [name]
                    
                    # Build composed function
                    def composed(grid, ch=new_chain):
                        result = grid
                        for n in ch:
                            result = self.dsl[n](result)
                        return result
                    
                    # Check if this solves the task
                    if self._verify_transform(examples, composed):
                        prediction = composed(test_input)
                        return SolveResult(
                            success=True,
                            prediction=prediction,
                            strategy=Strategy.DEEP_SEARCH,
                            description=" → ".join(new_chain),
                            confidence=0.7,
                            time_ms=0,
                            transform_chain=new_chain
                        )
                    
                    # Add to queue for further exploration (BF Search)
                    # For A*, we would need a proper heuristic function
                    if depth + 1 <= max_depth:
                         # Simple optimization: don't add if grid didn't change (idempotent step) in most cases
                         # unless it's a specific logical op. 
                         if not np.array_equal(new_grid, current):
                            queue.append((depth + 1, new_chain, new_grid))
                        
                except Exception:
                    continue
        
        return None
    
    def _verify_transform(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        transform: Callable
    ) -> bool:
        """Verify transform works on all examples."""
        try:
            for inp, out in examples:
                result = transform(inp)
                if result is None:
                    return False
                if not np.array_equal(result, out):
                    return False
            return True
        except Exception:
            return False
    
    def _compute_signature(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> TaskSignature:
        """Compute task signature for analogical matching."""
        inp0, out0 = examples[0]
        
        in_colors = len(np.unique(inp0)) - (1 if 0 in inp0 else 0)
        out_colors = len(np.unique(out0)) - (1 if 0 in out0 else 0)
        
        try:
            in_objs = len(self.object_detector.detect_objects(inp0))
            out_objs = len(self.object_detector.detect_objects(out0))
        except:
             in_objs = 0
             out_objs = 0
        
        has_sym = (np.array_equal(inp0, np.fliplr(inp0)) or 
                   np.array_equal(inp0, np.flipud(inp0)))
        
        return TaskSignature(
            input_shape=inp0.shape,
            output_shape=out0.shape,
            shape_ratio=(out0.shape[0] / inp0.shape[0], out0.shape[1] / inp0.shape[1]) if inp0.shape[0]>0 and inp0.shape[1]>0 else (1.0, 1.0),
            num_colors_in=in_colors,
            num_colors_out=out_colors,
            num_objects_in=in_objs,
            num_objects_out=out_objs,
            has_symmetry=has_sym
        )
    
    def _signatures_similar(self, sig1: str, sig2: str) -> bool:
        """Check if two task signatures are similar."""
        return sig1 == sig2
    
    def _chain_to_function(self, chain: List[str]) -> Optional[Callable]:
        """Convert a chain of primitive names to a function."""
        if not chain:
            return None
        
        def composed(grid):
            result = grid
            for name in chain:
                if name in self.dsl:
                    result = self.dsl[name](result)
                else:
                    return None
            return result
        
        return composed
    
    def _record_solution(
        self,
        task_id: str,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        result: SolveResult
    ) -> None:
        """Record successful solution for future use."""
        sig = self._compute_signature(examples)
        self.solution_memory[sig.to_key()] = {
            'task_id': task_id,
            'strategy': result.strategy.value,
            'description': result.description,
            'transform_chain': result.transform_chain,
            'confidence': result.confidence
        }
    
    def get_statistics(self) -> Dict[str, int]:
        """Get solving statistics."""
        return dict(self.stats)


# Convenience function
def solve_task(
    train_examples: List[Tuple[np.ndarray, np.ndarray]],
    test_input: np.ndarray,
    task_id: str = "unknown"
) -> SolveResult:
    """Solve an ARC task."""
    engine = SuperReasoningEngine()
    return engine.solve(train_examples, test_input, task_id)
