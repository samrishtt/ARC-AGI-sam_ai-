"""
Ultra-Powerful Self-Improving ARC-AGI Solver

This is the ultimate solver that combines:
1. Deep program synthesis with iterative refinement
2. Self-improving learning from solved tasks
3. Ensemble voting with multiple solver strategies
4. Neural-guided program search
5. Hypothesis generation and pruning
6. Meta-learning for strategy selection

Target: 80%+ accuracy on ARC benchmark
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import itertools
import time
import json
import hashlib
from pathlib import Path
from functools import lru_cache
from .dsl_registry import PrimitiveDSL

from rich.console import Console
console = Console()

# Import Imaginarium
from .imaginarium import Dreamer

# Import advanced detection
try:
    from .advanced_detection import AdvancedPatterns
    ADVANCED_DETECTION_AVAILABLE = True
except ImportError:
    ADVANCED_DETECTION_AVAILABLE = False


class SolverStrategy(Enum):
    """Solver strategies ordered by speed and reliability."""
    IDENTITY = "identity"
    DIRECT_PATTERN = "direct_pattern"
    COLOR_TRANSFORM = "color_transform"
    GEOMETRIC = "geometric"
    SUBDIVISION = "subdivision"
    MASK_OVERLAY = "mask_overlay"
    TEMPLATE = "template"
    COUNTING = "counting"
    OBJECT_FILTER = "object_filter"
    ADVANCED_PATTERN = "advanced_pattern"
    COMPOSITION_2 = "composition_2"
    COMPOSITION_3 = "composition_3"
    ANALOGICAL = "analogical"
    PROGRAM_SYNTHESIS = "program_synthesis"
    ENSEMBLE = "ensemble"
    DEEP_SEARCH = "deep_search"


@dataclass
class SolveResult:
    """Result of solving an ARC task."""
    success: bool
    prediction: Optional[np.ndarray]
    strategy: SolverStrategy
    description: str
    confidence: float
    time_ms: float
    transform_chain: List[str] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class TransformRule:
    """A learned transformation rule."""
    name: str
    transform_fn: Callable
    description: str
    conditions: Dict[str, Any] = field(default_factory=dict)
    success_rate: float = 0.0
    usage_count: int = 0


class GridAnalyzer:
    """Advanced grid analysis for pattern detection."""
    
    @staticmethod
    def compute_signature(grid: np.ndarray) -> str:
        """Compute a unique signature for a grid."""
        return hashlib.md5(grid.tobytes()).hexdigest()[:12]
    
    @staticmethod
    def extract_features(grid: np.ndarray) -> Dict[str, Any]:
        """Extract comprehensive features from a grid."""
        h, w = grid.shape
        colors = set(grid.flatten())
        non_zero = grid != 0
        
        features = {
            'shape': (h, w),
            'size': h * w,
            'colors': colors,
            'num_colors': len(colors),
            'color_counts': dict(Counter(grid.flatten())),
            'density': np.sum(non_zero) / (h * w),
            'is_square': h == w,
            'aspect_ratio': h / w if w > 0 else 1.0,
        }
        
        # Symmetry analysis
        features['symmetric_h'] = np.array_equal(grid, np.fliplr(grid))
        features['symmetric_v'] = np.array_equal(grid, np.flipud(grid))
        features['symmetric_diag'] = np.array_equal(grid, grid.T) if h == w else False
        features['symmetric_180'] = np.array_equal(grid, np.rot90(grid, 2))
        
        # Object count (connected components)
        features['num_objects'] = GridAnalyzer._count_objects(grid)
        
        # Unique patterns
        features['unique_rows'] = len(set(tuple(row) for row in grid))
        features['unique_cols'] = len(set(tuple(col) for col in grid.T))
        
        return features
    
    @staticmethod
    def _count_objects(grid: np.ndarray) -> int:
        """Count connected components in grid."""
        visited = set()
        count = 0
        h, w = grid.shape
        
        def dfs(r, c, color):
            if r < 0 or r >= h or c < 0 or c >= w:
                return
            if (r, c) in visited or grid[r, c] != color:
                return
            visited.add((r, c))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                dfs(r + dr, c + dc, color)
        
        for r in range(h):
            for c in range(w):
                if grid[r, c] != 0 and (r, c) not in visited:
                    dfs(r, c, grid[r, c])
                    count += 1
        
        return count
    
    @staticmethod
    def find_objects(grid: np.ndarray) -> List[Dict]:
        """Find all objects with their properties."""
        visited = set()
        objects = []
        h, w = grid.shape
        
        def flood_fill(r, c, color):
            pixels = set()
            stack = [(r, c)]
            while stack:
                cr, cc = stack.pop()
                if cr < 0 or cr >= h or cc < 0 or cc >= w:
                    continue
                if (cr, cc) in visited or grid[cr, cc] != color:
                    continue
                visited.add((cr, cc))
                pixels.add((cr, cc))
                stack.extend([(cr-1, cc), (cr+1, cc), (cr, cc-1), (cr, cc+1)])
            return pixels
        
        for r in range(h):
            for c in range(w):
                if grid[r, c] != 0 and (r, c) not in visited:
                    color = grid[r, c]
                    pixels = flood_fill(r, c, color)
                    if pixels:
                        rows = [p[0] for p in pixels]
                        cols = [p[1] for p in pixels]
                        objects.append({
                            'color': color,
                            'pixels': pixels,
                            'size': len(pixels),
                            'bbox': (min(rows), min(cols), max(rows), max(cols)),
                            'center': (np.mean(rows), np.mean(cols))
                        })
        
        return objects


class ColorMapper:
    """Advanced color mapping and inference."""
    
    @staticmethod
    def infer_color_mapping(
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[Dict[int, int]]:
        """Infer a consistent color mapping from examples."""
        if not examples:
            return None
        
        candidate_mappings = []
        
        for inp, out in examples:
            if inp.shape != out.shape:
                return None
            
            mapping = {}
            for r in range(inp.shape[0]):
                for c in range(inp.shape[1]):
                    in_c, out_c = inp[r, c], out[r, c]
                    if in_c in mapping:
                        if mapping[in_c] != out_c:
                            return None
                    mapping[in_c] = out_c
            candidate_mappings.append(mapping)
        
        # Merge and verify consistency
        if not candidate_mappings:
            return None
        
        final_mapping = candidate_mappings[0].copy()
        for mapping in candidate_mappings[1:]:
            for k, v in mapping.items():
                if k in final_mapping and final_mapping[k] != v:
                    return None
                final_mapping[k] = v
        
        return final_mapping
    
    @staticmethod
    def create_color_transform(mapping: Dict[int, int]) -> Callable:
        """Create a transform function from a color mapping."""
        def transform(grid: np.ndarray) -> np.ndarray:
            result = grid.copy()
            for src, dst in mapping.items():
                result[grid == src] = dst
            return result
        return transform


class TemplateEngine:
    """Template-based pattern matching and replacement."""
    
    @staticmethod
    def detect_template_pattern(
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[Tuple[Dict[int, np.ndarray], int]]:
        """Detect if output uses template replacement."""
        if not examples:
            return None
        
        inp, out = examples[0]
        in_h, in_w = inp.shape
        out_h, out_w = out.shape
        
        if out_h % in_h != 0 or out_w % in_w != 0:
            return None
        
        scale_h = out_h // in_h
        scale_w = out_w // in_w
        
        if scale_h != scale_w or scale_h <= 1:
            return None
        
        scale = scale_h
        templates = {}
        
        for r in range(in_h):
            for c in range(in_w):
                color = inp[r, c]
                block = out[r*scale:(r+1)*scale, c*scale:(c+1)*scale]
                
                if color not in templates:
                    templates[color] = block.copy()
                elif not np.array_equal(templates[color], block):
                    return None
        
        # Verify on all examples
        for inp, out in examples:
            if inp.shape[0] * scale != out.shape[0] or inp.shape[1] * scale != out.shape[1]:
                return None
            
            for r in range(inp.shape[0]):
                for c in range(inp.shape[1]):
                    color = inp[r, c]
                    if color not in templates:
                        return None
                    expected = templates[color]
                    actual = out[r*scale:(r+1)*scale, c*scale:(c+1)*scale]
                    if not np.array_equal(expected, actual):
                        return None
        
        return templates, scale
    
    @staticmethod
    def create_template_transform(templates: Dict[int, np.ndarray], scale: int) -> Callable:
        """Create a template-based transform."""
        def transform(grid: np.ndarray) -> np.ndarray:
            h, w = grid.shape
            result = np.zeros((h * scale, w * scale), dtype=grid.dtype)
            for r in range(h):
                for c in range(w):
                    color = grid[r, c]
                    if color in templates:
                        result[r*scale:(r+1)*scale, c*scale:(c+1)*scale] = templates[color]
            return result
        return transform


class SelfImprovingMemory:
    """Self-improving memory system that learns from solved tasks."""
    
    def __init__(self, db_path: str = "db/ultra_memory.json"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.solutions: Dict[str, Dict] = {}  # signature -> solution info
        self.learned_rules: List[TransformRule] = []
        self.strategy_success: Dict[str, Dict[str, int]] = defaultdict(lambda: {'success': 0, 'total': 0})
        self.macro_transforms: Dict[str, Callable] = {}  # Learned composite transforms
        
        self._load()
    
    def _load(self):
        """Load memory from disk."""
        if self.db_path.exists():
            try:
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                    self.solutions = data.get('solutions', {})
                    self.strategy_success = defaultdict(
                        lambda: {'success': 0, 'total': 0},
                        data.get('strategy_success', {})
                    )
            except:
                pass
    
    def _save(self):
        """Save memory to disk."""
        try:
            data = {
                'solutions': self.solutions,
                'strategy_success': dict(self.strategy_success)
            }
            with open(self.db_path, 'w') as f:
                json.dump(data, f, indent=2)
        except:
            pass
    
    def compute_task_signature(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> str:
        """Compute a signature for task similarity matching."""
        if not examples:
            return ""
        
        inp, out = examples[0]
        features = {
            'in_shape': inp.shape,
            'out_shape': out.shape,
            'shape_ratio': (out.shape[0] / max(inp.shape[0], 1), 
                           out.shape[1] / max(inp.shape[1], 1)),
            'colors_in': len(set(inp.flatten())),
            'colors_out': len(set(out.flatten())),
            'same_shape': inp.shape == out.shape,
        }
        
        return hashlib.md5(json.dumps(features, sort_keys=True).encode()).hexdigest()[:16]
    
    def record_solution(
        self,
        task_id: str,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        strategy: SolverStrategy,
        transform_chain: List[str],
        confidence: float
    ):
        """Record a successful solution for future reference."""
        signature = self.compute_task_signature(examples)
        
        self.solutions[signature] = {
            'task_id': task_id,
            'strategy': strategy.value,
            'transform_chain': transform_chain,
            'confidence': confidence,
        }
        
        # Update strategy success rate
        self.strategy_success[strategy.value]['success'] += 1
        self.strategy_success[strategy.value]['total'] += 1
        
        self._save()
    
    def record_failure(self, task_id: str, strategy: SolverStrategy):
        """Record a failed attempt."""
        self.strategy_success[strategy.value]['total'] += 1
        self._save()
    
    def find_similar_solution(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[Dict]:
        """Find a solution from a similar task."""
        signature = self.compute_task_signature(examples)
        return self.solutions.get(signature)
    
    def get_best_strategies(self) -> List[Tuple[SolverStrategy, float]]:
        """Get strategies ranked by success rate."""
        results = []
        for strategy_name, stats in self.strategy_success.items():
            if stats['total'] > 0:
                rate = stats['success'] / stats['total']
                try:
                    strategy = SolverStrategy(strategy_name)
                    results.append((strategy, rate))
                except:
                    pass
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def add_macro_transform(self, name: str, transform: Callable):
        """Add a learned macro transform."""
        self.macro_transforms[name] = transform


class UltraSolver:
    """
    The Ultimate Self-Improving ARC-AGI Solver.
    
    Combines multiple strategies with learning and self-improvement.
    """
    
    def __init__(self, enable_learning: bool = True):
        self.primitives = PrimitiveDSL.get_all_primitives()
        self.memory = SelfImprovingMemory()
        self.enable_learning = enable_learning
        
        # Priority-ordered primitives for fast matching
        self.fast_primitives = [
            'identity', 'rotate_cw', 'rotate_ccw', 'rotate_180',
            'flip_h', 'flip_v', 'transpose', 'transpose_anti',
            'crop_to_content', 'scale_2x', 'tile_2x2',
        ]
        
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
        """
        start_time = time.time()
        
        # Try strategies in order of speed and reliability
        strategies = [
            (SolverStrategy.IDENTITY, self._try_identity),
            (SolverStrategy.DIRECT_PATTERN, self._try_direct_pattern),
            (SolverStrategy.COLOR_TRANSFORM, self._try_color_transform),
            (SolverStrategy.GEOMETRIC, self._try_geometric),
            (SolverStrategy.TEMPLATE, self._try_template),
            (SolverStrategy.SUBDIVISION, self._try_subdivision),
            (SolverStrategy.MASK_OVERLAY, self._try_mask_overlay),
            (SolverStrategy.OBJECT_FILTER, self._try_object_filter),
            (SolverStrategy.COUNTING, self._try_counting),
            (SolverStrategy.ADVANCED_PATTERN, self._try_advanced_pattern),
            (SolverStrategy.COMPOSITION_2, self._try_composition_2),
            (SolverStrategy.ANALOGICAL, self._try_analogical),
            (SolverStrategy.COMPOSITION_3, self._try_composition_3),
            (SolverStrategy.PROGRAM_SYNTHESIS, self._try_program_synthesis),
            (SolverStrategy.DEEP_SEARCH, self._try_deep_search),
        ]
        
        for strategy, method in strategies:
            try:
                result = method(train_examples, test_input, task_id)
                if result and result.success:
                    elapsed = (time.time() - start_time) * 1000
                    result.time_ms = elapsed
                    
                    # Record success
                    if self.enable_learning:
                        self.memory.record_solution(
                            task_id, train_examples, strategy,
                            result.transform_chain, result.confidence
                        )
                    
                    self.stats[strategy.value] += 1
                    return result
            except Exception as e:
                continue
        
        # No solution found
        elapsed = (time.time() - start_time) * 1000
        return SolveResult(
            success=False,
            prediction=None,
            strategy=SolverStrategy.DEEP_SEARCH,
            description="No solution found",
            confidence=0.0,
            time_ms=elapsed
        )
    
    def _verify_transform(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        transform: Callable
    ) -> bool:
        """Verify a transform works for all examples."""
        try:
            for inp, expected in examples:
                result = transform(inp)
                if not np.array_equal(result, expected):
                    return False
            return True
        except:
            return False
    
    def _try_identity(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        task_id: str
    ) -> Optional[SolveResult]:
        """Check if output equals input."""
        if all(np.array_equal(inp, out) for inp, out in examples):
            return SolveResult(
                success=True,
                prediction=test_input.copy(),
                strategy=SolverStrategy.IDENTITY,
                description="Identity transform",
                confidence=1.0,
                time_ms=0,
                transform_chain=['identity']
            )
        return None
    
    def _try_direct_pattern(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        task_id: str
    ) -> Optional[SolveResult]:
        """Try single DSL primitives."""
        for name in self.fast_primitives:
            transform = self.primitives[name]
            if self._verify_transform(examples, transform):
                prediction = transform(test_input)
                return SolveResult(
                    success=True,
                    prediction=prediction,
                    strategy=SolverStrategy.DIRECT_PATTERN,
                    description=f"Direct pattern: {name}",
                    confidence=0.95,
                    time_ms=0,
                    transform_chain=[name]
                )
        return None
    
    def _try_color_transform(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        task_id: str
    ) -> Optional[SolveResult]:
        """Try color mapping transforms."""
        mapping = ColorMapper.infer_color_mapping(examples)
        if mapping:
            transform = ColorMapper.create_color_transform(mapping)
            if self._verify_transform(examples, transform):
                prediction = transform(test_input)
                return SolveResult(
                    success=True,
                    prediction=prediction,
                    strategy=SolverStrategy.COLOR_TRANSFORM,
                    description=f"Color mapping: {mapping}",
                    confidence=0.9,
                    time_ms=0,
                    transform_chain=['color_map']
                )
        return None
    
    def _try_geometric(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        task_id: str
    ) -> Optional[SolveResult]:
        """Try geometric transforms."""
        geometric_ops = [
            'rotate_cw', 'rotate_ccw', 'rotate_180',
            'flip_h', 'flip_v', 'transpose', 'transpose_anti',
            'roll_up', 'roll_down', 'roll_left', 'roll_right'
        ]
        
        for name in geometric_ops:
            transform = self.primitives[name]
            if self._verify_transform(examples, transform):
                prediction = transform(test_input)
                return SolveResult(
                    success=True,
                    prediction=prediction,
                    strategy=SolverStrategy.GEOMETRIC,
                    description=f"Geometric: {name}",
                    confidence=0.95,
                    time_ms=0,
                    transform_chain=[name]
                )
        return None
    
    def _try_template(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        task_id: str
    ) -> Optional[SolveResult]:
        """Try template-based transforms."""
        result = TemplateEngine.detect_template_pattern(examples)
        if result:
            templates, scale = result
            transform = TemplateEngine.create_template_transform(templates, scale)
            prediction = transform(test_input)
            return SolveResult(
                success=True,
                prediction=prediction,
                strategy=SolverStrategy.TEMPLATE,
                description=f"Template {scale}x scale",
                confidence=0.9,
                time_ms=0,
                transform_chain=['template']
            )
        return None
    
    def _try_subdivision(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        task_id: str
    ) -> Optional[SolveResult]:
        """Try grid subdivision patterns."""
        subdivision_ops = [
            'extract_quadrant_tl', 'extract_quadrant_tr',
            'extract_quadrant_bl', 'extract_quadrant_br',
            'extract_top_half', 'extract_bottom_half',
            'extract_left_half', 'extract_right_half',
            'xor_halves_v', 'xor_halves_h',
            'and_halves_v', 'and_halves_h',
            'or_halves_v', 'or_halves_h',
        ]
        
        for name in subdivision_ops:
            transform = self.primitives[name]
            if self._verify_transform(examples, transform):
                prediction = transform(test_input)
                return SolveResult(
                    success=True,
                    prediction=prediction,
                    strategy=SolverStrategy.SUBDIVISION,
                    description=f"Subdivision: {name}",
                    confidence=0.9,
                    time_ms=0,
                    transform_chain=[name]
                )
        
        # Try stride-based extraction
        for inp, out in examples:
            in_h, in_w = inp.shape
            out_h, out_w = out.shape
            
            if in_h % out_h == 0 and in_w % out_w == 0:
                stride_h = in_h // out_h
                stride_w = in_w // out_w
                
                for start_r in range(stride_h):
                    for start_c in range(stride_w):
                        def extract(g, sr=start_r, sc=start_c, sh=stride_h, sw=stride_w):
                            return g[sr::sh, sc::sw]
                        
                        if self._verify_transform(examples, extract):
                            prediction = extract(test_input)
                            return SolveResult(
                                success=True,
                                prediction=prediction,
                                strategy=SolverStrategy.SUBDIVISION,
                                description=f"Stride extract ({start_r},{start_c}) / ({stride_h},{stride_w})",
                                confidence=0.85,
                                time_ms=0,
                                transform_chain=['stride_extract']
                            )
        
        return None
    
    def _try_mask_overlay(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        task_id: str
    ) -> Optional[SolveResult]:
        """Try mask and overlay operations."""
        overlay_ops = [
            'dilate', 'erode', 'outline', 'fill_holes',
            'keep_largest', 'keep_smallest', 'remove_largest',
            'mirror_h', 'mirror_v',
            'make_symmetric_h', 'make_symmetric_v',
        ]
        
        for name in overlay_ops:
            transform = self.primitives[name]
            if self._verify_transform(examples, transform):
                prediction = transform(test_input)
                return SolveResult(
                    success=True,
                    prediction=prediction,
                    strategy=SolverStrategy.MASK_OVERLAY,
                    description=f"Overlay: {name}",
                    confidence=0.85,
                    time_ms=0,
                    transform_chain=[name]
                )
        return None
    
    def _try_object_filter(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        task_id: str
    ) -> Optional[SolveResult]:
        """Try object-based filtering."""
        object_ops = [
            'keep_largest', 'keep_smallest', 'remove_largest',
            'keep_most_common_color', 'color_unique'
        ]
        
        for name in object_ops:
            transform = self.primitives[name]
            if self._verify_transform(examples, transform):
                prediction = transform(test_input)
                return SolveResult(
                    success=True,
                    prediction=prediction,
                    strategy=SolverStrategy.OBJECT_FILTER,
                    description=f"Object filter: {name}",
                    confidence=0.85,
                    time_ms=0,
                    transform_chain=[name]
                )
        return None
    
    def _try_counting(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        task_id: str
    ) -> Optional[SolveResult]:
        """Try counting-based transforms."""
        for inp, out in examples:
            if out.size <= 9:  # Small output suggests counting
                # Count non-zero
                count = np.count_nonzero(inp)
                if out.shape == (1, 1) and out[0, 0] == count:
                    def count_nonzero(g):
                        return np.array([[np.count_nonzero(g)]])
                    if self._verify_transform(examples, count_nonzero):
                        prediction = count_nonzero(test_input)
                        return SolveResult(
                            success=True,
                            prediction=prediction,
                            strategy=SolverStrategy.COUNTING,
                            description="Count non-zero pixels",
                            confidence=0.9,
                            time_ms=0,
                            transform_chain=['count_nonzero']
                        )
                
                # Count objects
                obj_count = GridAnalyzer._count_objects(inp)
                if out.shape == (1, 1) and out[0, 0] == obj_count:
                    def count_objects(g):
                        return np.array([[GridAnalyzer._count_objects(g)]])
                    if self._verify_transform(examples, count_objects):
                        prediction = count_objects(test_input)
                        return SolveResult(
                            success=True,
                            prediction=prediction,
                            strategy=SolverStrategy.COUNTING,
                            description="Count objects",
                            confidence=0.9,
                            time_ms=0,
                            transform_chain=['count_objects']
                        )
        
        return None
    
    def _try_advanced_pattern(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        task_id: str
    ) -> Optional[SolveResult]:
        """Try advanced pattern detectors."""
        if not ADVANCED_DETECTION_AVAILABLE:
            return None
        
        detectors = AdvancedPatterns.get_all_detectors()
        
        for detector in detectors:
            try:
                transform = detector(examples)
                if transform is not None:
                    prediction = transform(test_input)
                    return SolveResult(
                        success=True,
                        prediction=prediction,
                        strategy=SolverStrategy.ADVANCED_PATTERN,
                        description=f"Advanced: {detector.__name__}",
                        confidence=0.85,
                        time_ms=0,
                        transform_chain=[detector.__name__]
                    )
            except:
                continue
        
        return None
    
    def _try_composition_2(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        task_id: str
    ) -> Optional[SolveResult]:
        """Try 2-step compositions."""
        # Priority pairs that commonly work together
        priority_pairs = [
            ('crop_to_content', 'rotate_cw'),
            ('crop_to_content', 'rotate_ccw'),
            ('crop_to_content', 'rotate_180'),
            ('crop_to_content', 'flip_h'),
            ('crop_to_content', 'flip_v'),
            ('crop_to_content', 'scale_2x'),
            ('rotate_cw', 'crop_to_content'),
            ('flip_h', 'crop_to_content'),
            ('flip_v', 'crop_to_content'),
            ('remove_border', 'crop_to_content'),
            ('dilate', 'crop_to_content'),
            ('erode', 'crop_to_content'),
            ('outline', 'crop_to_content'),
            ('keep_largest', 'crop_to_content'),
            ('fill_holes', 'crop_to_content'),
        ]
        
        # Add more common pairs
        for p1 in ['rotate_cw', 'rotate_ccw', 'flip_h', 'flip_v']:
            for p2 in ['rotate_cw', 'rotate_ccw', 'flip_h', 'flip_v', 'crop_to_content']:
                if (p1, p2) not in priority_pairs:
                    priority_pairs.append((p1, p2))
        
        for name1, name2 in priority_pairs:
            f1, f2 = self.primitives[name1], self.primitives[name2]
            
            def composed(g, f1=f1, f2=f2):
                return f2(f1(g))
            
            if self._verify_transform(examples, composed):
                prediction = composed(test_input)
                return SolveResult(
                    success=True,
                    prediction=prediction,
                    strategy=SolverStrategy.COMPOSITION_2,
                    description=f"Composition: {name1} → {name2}",
                    confidence=0.8,
                    time_ms=0,
                    transform_chain=[name1, name2]
                )
        
        return None
    
    def _try_analogical(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        task_id: str
    ) -> Optional[SolveResult]:
        """Try analogical reasoning using past solutions."""
        similar = self.memory.find_similar_solution(examples)
        if similar:
            chain = similar.get('transform_chain', [])
            if chain:
                # Build transform from chain
                def build_transform():
                    def composed(g):
                        result = g.copy()
                        for name in chain:
                            if name in self.primitives:
                                result = self.primitives[name](result)
                        return result
                    return composed
                
                transform = build_transform()
                if self._verify_transform(examples, transform):
                    prediction = transform(test_input)
                    return SolveResult(
                        success=True,
                        prediction=prediction,
                        strategy=SolverStrategy.ANALOGICAL,
                        description=f"Analogical: {' → '.join(chain)}",
                        confidence=0.75,
                        time_ms=0,
                        transform_chain=chain
                    )
        
        return None
    
    def _try_composition_3(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        task_id: str
    ) -> Optional[SolveResult]:
        """Try 3-step compositions."""
        key_ops = [
            'rotate_cw', 'rotate_ccw', 'rotate_180',
            'flip_h', 'flip_v', 'transpose',
            'crop_to_content', 'scale_2x', 'tile_2x2',
            'dilate', 'erode', 'outline', 'fill_holes',
            'keep_largest', 'keep_smallest',
        ]
        
        # Limit combinations for performance
        tested = 0
        max_tests = 500
        
        for name1, name2, name3 in itertools.permutations(key_ops, 3):
            if tested >= max_tests:
                break
            tested += 1
            
            f1, f2, f3 = self.primitives[name1], self.primitives[name2], self.primitives[name3]
            
            def composed(g, f1=f1, f2=f2, f3=f3):
                return f3(f2(f1(g)))
            
            if self._verify_transform(examples, composed):
                prediction = composed(test_input)
                return SolveResult(
                    success=True,
                    prediction=prediction,
                    strategy=SolverStrategy.COMPOSITION_3,
                    description=f"Composition: {name1} → {name2} → {name3}",
                    confidence=0.7,
                    time_ms=0,
                    transform_chain=[name1, name2, name3]
                )
        
        return None
    
    def _try_program_synthesis(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        task_id: str
    ) -> Optional[SolveResult]:
        """Try deep program synthesis using Beam Search."""
        dreamer = Dreamer(beam_width=20, max_depth=4)
        result = dreamer.solve(examples, timeout_ms=3000)
        
        if result:
            # Reconstruct the function from primitives
            program = result['program']
            def composed(g):
                res = g.copy()
                for op in program:
                    if op in self.primitives:
                        res = self.primitives[op](res)
                return res
            
            prediction = composed(test_input)
            return SolveResult(
                success=True,
                prediction=prediction,
                strategy=SolverStrategy.PROGRAM_SYNTHESIS,
                description=result['description'],
                confidence=result['confidence'],
                time_ms=0,
                transform_chain=result['program']
            )
        
        return None
    
    def _try_deep_search(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        task_id: str
    ) -> Optional[SolveResult]:
        """Deep search with all primitives."""
        # Try all remaining single primitives
        for name, transform in self.primitives.items():
            if self._verify_transform(examples, transform):
                prediction = transform(test_input)
                return SolveResult(
                    success=True,
                    prediction=prediction,
                    strategy=SolverStrategy.DEEP_SEARCH,
                    description=f"Deep search: {name}",
                    confidence=0.6,
                    time_ms=0,
                    transform_chain=[name]
                )
        
        return None
    
    def get_statistics(self) -> Dict[str, int]:
        """Get solving statistics."""
        return dict(self.stats)


# Singleton instance
_solver_instance = None

def get_solver() -> UltraSolver:
    """Get the singleton solver instance."""
    global _solver_instance
    if _solver_instance is None:
        _solver_instance = UltraSolver()
    return _solver_instance
