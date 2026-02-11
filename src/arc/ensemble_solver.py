"""
Ensemble Solver - Voting System for Maximum Accuracy

Combines multiple solving approaches and uses voting to select the best answer.
Key strategies:
1. Multiple solver variants with different configurations
2. Confidence-weighted voting
3. Consistency checking across solvers
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from collections import Counter
import time
from concurrent.futures import ThreadPoolExecutor
import hashlib

from .ultra_solver import UltraSolver, SolverStrategy, SolveResult, PrimitiveDSL, GridAnalyzer
from .advanced_detection import AdvancedPatterns


@dataclass
class VotingResult:
    """Result from ensemble voting."""
    success: bool
    prediction: Optional[np.ndarray]
    confidence: float
    num_votes: int
    strategies_used: List[str]
    time_ms: float
    reasoning: str


class EnsembleSolver:
    """
    Ensemble solver that combines multiple approaches.
    
    Uses voting to achieve higher accuracy by leveraging
    multiple solving strategies and configurations.
    """
    
    def __init__(self, num_variants: int = 3):
        self.num_variants = num_variants
        self.base_solver = UltraSolver(enable_learning=True)
        self.primitives = PrimitiveDSL.get_all_primitives()
        
        # Additional specialized solvers
        self.specialized_solvers = {
            'geometric': self._solve_geometric,
            'color': self._solve_color,
            'object': self._solve_object,
            'template': self._solve_template,
            'subdivision': self._solve_subdivision,
            'composition': self._solve_composition,
            'advanced': self._solve_advanced,
        }
    
    def solve(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        task_id: str = "unknown"
    ) -> VotingResult:
        """Solve using ensemble voting."""
        start_time = time.time()
        
        # Collect predictions from all approaches
        predictions = []
        
        # 1. Main solver
        main_result = self.base_solver.solve(train_examples, test_input, task_id)
        if main_result.success:
            predictions.append({
                'prediction': main_result.prediction,
                'confidence': main_result.confidence,
                'strategy': main_result.strategy.value,
                'description': main_result.description
            })
        
        # 2. Specialized solvers
        for name, solver_fn in self.specialized_solvers.items():
            try:
                result = solver_fn(train_examples, test_input)
                if result is not None:
                    predictions.append({
                        'prediction': result['prediction'],
                        'confidence': result['confidence'],
                        'strategy': name,
                        'description': result.get('description', name)
                    })
            except:
                continue
        
        # 3. Vote on predictions
        if not predictions:
            elapsed = (time.time() - start_time) * 1000
            return VotingResult(
                success=False,
                prediction=None,
                confidence=0.0,
                num_votes=0,
                strategies_used=[],
                time_ms=elapsed,
                reasoning="No predictions generated"
            )
        
        # Group predictions by hash
        prediction_groups = {}
        for pred in predictions:
            pred_hash = hashlib.md5(pred['prediction'].tobytes()).hexdigest()
            if pred_hash not in prediction_groups:
                prediction_groups[pred_hash] = {
                    'prediction': pred['prediction'],
                    'votes': 0,
                    'total_confidence': 0.0,
                    'strategies': []
                }
            prediction_groups[pred_hash]['votes'] += 1
            prediction_groups[pred_hash]['total_confidence'] += pred['confidence']
            prediction_groups[pred_hash]['strategies'].append(pred['strategy'])
        
        # Find best prediction by weighted vote
        best_group = max(
            prediction_groups.values(),
            key=lambda g: g['votes'] * g['total_confidence']
        )
        
        elapsed = (time.time() - start_time) * 1000
        
        avg_confidence = best_group['total_confidence'] / best_group['votes']
        
        return VotingResult(
            success=True,
            prediction=best_group['prediction'],
            confidence=avg_confidence,
            num_votes=best_group['votes'],
            strategies_used=best_group['strategies'],
            time_ms=elapsed,
            reasoning=f"Selected by {best_group['votes']} votes from {len(predictions)} predictions"
        )
    
    def _verify_transform(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        transform
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
            
    def _solve_advanced(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray
    ) -> Optional[Dict]:
        """Specialized advanced pattern solver."""
        detectors = AdvancedPatterns.get_all_detectors()
        
        for detector in detectors:
            try:
                transform = detector(examples)
                if transform is not None:
                    return {
                        'prediction': transform(test_input),
                        'confidence': 0.9,
                        'description': f"Advanced: {detector.__name__}"
                    }
            except:
                continue
        return None

    
    def _solve_geometric(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray
    ) -> Optional[Dict]:
        """Specialized geometric solver."""
        ops = [
            ('identity', lambda g: g.copy()),
            ('rotate_cw', lambda g: np.rot90(g, -1)),
            ('rotate_ccw', lambda g: np.rot90(g, 1)),
            ('rotate_180', lambda g: np.rot90(g, 2)),
            ('flip_h', lambda g: np.fliplr(g)),
            ('flip_v', lambda g: np.flipud(g)),
            ('transpose', lambda g: g.T),
            ('transpose_anti', lambda g: np.rot90(np.flipud(g))),
        ]
        
        for name, transform in ops:
            if self._verify_transform(examples, transform):
                return {
                    'prediction': transform(test_input),
                    'confidence': 0.95,
                    'description': f"Geometric: {name}"
                }
        
        return None
    
    def _solve_color(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray
    ) -> Optional[Dict]:
        """Specialized color mapping solver."""
        if not examples:
            return None
        
        # Try to infer color mapping
        mapping = {}
        consistent = True
        
        for inp, out in examples:
            if inp.shape != out.shape:
                consistent = False
                break
            
            for r in range(inp.shape[0]):
                for c in range(inp.shape[1]):
                    in_c, out_c = inp[r, c], out[r, c]
                    if in_c in mapping:
                        if mapping[in_c] != out_c:
                            consistent = False
                            break
                    mapping[in_c] = out_c
                if not consistent:
                    break
            if not consistent:
                break
        
        if consistent and mapping:
            def apply_mapping(g):
                result = g.copy()
                for src, dst in mapping.items():
                    result[g == src] = dst
                return result
            
            if self._verify_transform(examples, apply_mapping):
                return {
                    'prediction': apply_mapping(test_input),
                    'confidence': 0.9,
                    'description': f"Color mapping: {mapping}"
                }
        
        return None
    
    def _solve_object(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray
    ) -> Optional[Dict]:
        """Specialized object-based solver."""
        ops = [
            ('keep_largest', self.primitives.get('keep_largest', lambda g: g)),
            ('keep_smallest', self.primitives.get('keep_smallest', lambda g: g)),
            ('remove_largest', self.primitives.get('remove_largest', lambda g: g)),
            ('crop_to_content', self.primitives.get('crop_to_content', lambda g: g)),
            ('outline', self.primitives.get('outline', lambda g: g)),
        ]
        
        for name, transform in ops:
            if self._verify_transform(examples, transform):
                return {
                    'prediction': transform(test_input),
                    'confidence': 0.85,
                    'description': f"Object: {name}"
                }
        
        return None
    
    def _solve_template(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray
    ) -> Optional[Dict]:
        """Specialized template matching solver."""
        if not examples:
            return None
        
        inp, out = examples[0]
        in_h, in_w = inp.shape
        out_h, out_w = out.shape
        
        # Check for scaling pattern
        if out_h % in_h == 0 and out_w % in_w == 0:
            scale_h = out_h // in_h
            scale_w = out_w // in_w
            
            if scale_h == scale_w and scale_h > 1:
                scale = scale_h
                templates = {}
                consistent = True
                
                for r in range(in_h):
                    for c in range(in_w):
                        color = inp[r, c]
                        block = out[r*scale:(r+1)*scale, c*scale:(c+1)*scale]
                        
                        if color not in templates:
                            templates[color] = block.copy()
                        elif not np.array_equal(templates[color], block):
                            consistent = False
                            break
                    if not consistent:
                        break
                
                if consistent and templates:
                    def apply_template(g):
                        h, w = g.shape
                        result = np.zeros((h * scale, w * scale), dtype=g.dtype)
                        for r in range(h):
                            for c in range(w):
                                color = g[r, c]
                                if color in templates:
                                    result[r*scale:(r+1)*scale, c*scale:(c+1)*scale] = templates[color]
                        return result
                    
                    if self._verify_transform(examples, apply_template):
                        return {
                            'prediction': apply_template(test_input),
                            'confidence': 0.9,
                            'description': f"Template {scale}x"
                        }
        
        return None
    
    def _solve_subdivision(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray
    ) -> Optional[Dict]:
        """Specialized grid subdivision solver."""
        ops = [
            ('quadrant_tl', lambda g: g[:g.shape[0]//2, :g.shape[1]//2]),
            ('quadrant_tr', lambda g: g[:g.shape[0]//2, g.shape[1]//2:]),
            ('quadrant_bl', lambda g: g[g.shape[0]//2:, :g.shape[1]//2]),
            ('quadrant_br', lambda g: g[g.shape[0]//2:, g.shape[1]//2:]),
            ('top_half', lambda g: g[:g.shape[0]//2, :]),
            ('bottom_half', lambda g: g[g.shape[0]//2:, :]),
            ('left_half', lambda g: g[:, :g.shape[1]//2]),
            ('right_half', lambda g: g[:, g.shape[1]//2:]),
        ]
        
        for name, transform in ops:
            if self._verify_transform(examples, transform):
                return {
                    'prediction': transform(test_input),
                    'confidence': 0.9,
                    'description': f"Subdivision: {name}"
                }
        
        # XOR/AND/OR operations on halves
        for inp, out in examples:
            h, w = inp.shape
            
            # Vertical split
            if w % 2 == 0:
                left = inp[:, :w//2]
                right = inp[:, w//2:]
                
                if left.shape == out.shape:
                    # XOR
                    xor_result = ((left != 0) ^ (right != 0)).astype(inp.dtype)
                    if np.array_equal(xor_result, out):
                        def xor_v(g):
                            w = g.shape[1]
                            return ((g[:, :w//2] != 0) ^ (g[:, w//2:] != 0)).astype(g.dtype)
                        if self._verify_transform(examples, xor_v):
                            return {
                                'prediction': xor_v(test_input),
                                'confidence': 0.85,
                                'description': "XOR vertical halves"
                            }
                    
                    # AND
                    and_result = np.where((left != 0) & (right != 0), left, 0)
                    if np.array_equal(and_result, out):
                        def and_v(g):
                            w = g.shape[1]
                            l, r = g[:, :w//2], g[:, w//2:]
                            return np.where((l != 0) & (r != 0), l, 0)
                        if self._verify_transform(examples, and_v):
                            return {
                                'prediction': and_v(test_input),
                                'confidence': 0.85,
                                'description': "AND vertical halves"
                            }
            
            # Horizontal split
            if h % 2 == 0:
                top = inp[:h//2, :]
                bottom = inp[h//2:, :]
                
                if top.shape == out.shape:
                    # XOR
                    xor_result = ((top != 0) ^ (bottom != 0)).astype(inp.dtype)
                    if np.array_equal(xor_result, out):
                        def xor_h(g):
                            h = g.shape[0]
                            return ((g[:h//2, :] != 0) ^ (g[h//2:, :] != 0)).astype(g.dtype)
                        if self._verify_transform(examples, xor_h):
                            return {
                                'prediction': xor_h(test_input),
                                'confidence': 0.85,
                                'description': "XOR horizontal halves"
                            }
        
        return None
    
    def _solve_composition(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray
    ) -> Optional[Dict]:
        """Specialized composition solver."""
        # Try common 2-step compositions
        priority_pairs = [
            ('crop_to_content', 'rotate_cw'),
            ('crop_to_content', 'rotate_ccw'),
            ('crop_to_content', 'flip_h'),
            ('crop_to_content', 'flip_v'),
            ('rotate_cw', 'crop_to_content'),
            ('rotate_ccw', 'crop_to_content'),
            ('flip_h', 'crop_to_content'),
            ('flip_v', 'crop_to_content'),
            ('remove_border', 'crop_to_content'),
            ('outline', 'crop_to_content'),
            ('keep_largest', 'crop_to_content'),
            ('dilate', 'crop_to_content'),
            ('erode', 'crop_to_content'),
        ]
        
        for name1, name2 in priority_pairs:
            if name1 in self.primitives and name2 in self.primitives:
                f1, f2 = self.primitives[name1], self.primitives[name2]
                
                def composed(g, f1=f1, f2=f2):
                    return f2(f1(g))
                
                if self._verify_transform(examples, composed):
                    return {
                        'prediction': composed(test_input),
                        'confidence': 0.8,
                        'description': f"Composition: {name1} → {name2}"
                    }
        
        return None


class IterativeRefinementSolver:
    """
    Solver that iteratively refines predictions.
    
    Uses hypothesis testing and refinement to improve accuracy.
    """
    
    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.base_solver = UltraSolver(enable_learning=True)
        self.primitives = PrimitiveDSL.get_all_primitives()
    
    def solve(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        task_id: str = "unknown"
    ) -> SolveResult:
        """Solve with iterative refinement."""
        start_time = time.time()
        
        # First, try base solver
        result = self.base_solver.solve(train_examples, test_input, task_id)
        if result.success:
            result.time_ms = (time.time() - start_time) * 1000
            return result
        
        # Iterative refinement
        hypotheses = self._generate_hypotheses(train_examples)
        
        for iteration in range(self.max_iterations):
            for hypothesis in hypotheses:
                try:
                    if self._verify_hypothesis(hypothesis, train_examples):
                        prediction = hypothesis(test_input)
                        elapsed = (time.time() - start_time) * 1000
                        return SolveResult(
                            success=True,
                            prediction=prediction,
                            strategy=SolverStrategy.PROGRAM_SYNTHESIS,
                            description=f"Refined hypothesis (iteration {iteration + 1})",
                            confidence=0.7,
                            time_ms=elapsed
                        )
                except:
                    continue
            
            # Generate more hypotheses for next iteration
            hypotheses = self._refine_hypotheses(hypotheses, train_examples)
        
        elapsed = (time.time() - start_time) * 1000
        return SolveResult(
            success=False,
            prediction=None,
            strategy=SolverStrategy.DEEP_SEARCH,
            description="No solution found after refinement",
            confidence=0.0,
            time_ms=elapsed
        )
    
    def _generate_hypotheses(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[callable]:
        """Generate initial hypotheses."""
        hypotheses = []
        
        # All single primitives
        for name, fn in self.primitives.items():
            hypotheses.append(fn)
        
        # Common compositions
        for name1, fn1 in list(self.primitives.items())[:20]:
            for name2, fn2 in list(self.primitives.items())[:20]:
                def composed(g, f1=fn1, f2=fn2):
                    return f2(f1(g))
                hypotheses.append(composed)
        
        return hypotheses
    
    def _verify_hypothesis(
        self,
        hypothesis: callable,
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> bool:
        """Verify a hypothesis works for all examples."""
        try:
            for inp, expected in examples:
                result = hypothesis(inp)
                if not np.array_equal(result, expected):
                    return False
            return True
        except:
            return False
    
    def _refine_hypotheses(
        self,
        current: List[callable],
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[callable]:
        """Refine hypotheses based on partial matches."""
        # For now, just add more compositions
        refined = []
        
        for fn in current[:50]:  # Take best subset
            for name, prim in list(self.primitives.items())[:10]:
                def composed1(g, f1=fn, f2=prim):
                    return f2(f1(g))
                def composed2(g, f1=prim, f2=fn):
                    return f2(f1(g))
                refined.extend([composed1, composed2])
        
        return refined[:500]  # Limit for performance
