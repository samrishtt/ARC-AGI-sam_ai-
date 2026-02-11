"""
The Imaginarium - Deep Program Synthesis for ARC-AGI

This module implements advanced program synthesis using Beam Search and 
heuristic guidance to construct complex transformation programs.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable, Set, Any, NamedTuple
from dataclasses import dataclass, field
import heapq
import time
from .dsl_registry import PrimitiveDSL


@dataclass
class ProgramNode:
    """A node in the program search graph."""
    program: List[str]                  # List of primitive names
    transform: Callable                 # The executable function
    score: float                        # Heuristic score
    cost: int                           # Length/complexity of program
    parent: Optional['ProgramNode'] = None
    
    def __lt__(self, other):
        # Higher score is better, but heapq is min-heap, so invert
        return self.score > other.score

class Dreamer:
    """
    The Dreamer synthesizes programs by 'imagining' combinations of primitives.
    Uses Beam Search to explore the space of programs efficiently.
    """
    
    def __init__(self, beam_width: int = 50, max_depth: int = 5):
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.primitives = PrimitiveDSL.get_all_primitives()
        
        # Categorize primitives for smarter search
        self.categories = {
            'geometric': ['rotate_cw', 'rotate_ccw', 'rotate_180', 'flip_h', 'flip_v', 'transpose'],
            'extraction': ['crop_to_content', 'keep_largest', 'keep_smallest', 'remove_border'],
            'color': ['color_unique', 'keep_most_common_color', 'inverse_color', 'invert_binary', 'replace_color', 'swap_colors'],
            'morph': ['dilate', 'erode', 'calm_down', 'outline', 'fill_holes'],
            'grid': ['tile_2x2', 'repeat_3x3', 'grid_split_2x2', 'tile_3x3'],
            'gravity': ['gravity_down', 'gravity_up', 'gravity_left', 'gravity_right'],
            'object': ['keep_largest_object', 'keep_smallest_object', 'remove_largest_object', 'extract_object', 'color_each_object'],
            'fill': ['fill_row', 'fill_column', 'fill_down_from_top', 'flood_fill_exterior', 'diagonal_fill'],
            'region': ['top_row', 'bottom_row', 'left_col', 'right_col', 'main_diagonal'],
            'logic': ['logical_or', 'logical_xor', 'logical_and', 'difference']
        }

    def _evaluate(self, node: ProgramNode, examples: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """
        Evaluate a program node against examples.
        Returns a score between 0.0 and 1.0 (1.0 = perfect solution).
        """
        total_score = 0.0
        
        try:
            for inp, out in examples:
                pred = node.transform(inp)
                
                # Exact match check
                if pred.shape == out.shape:
                    match = np.sum(pred == out)
                    total = out.size
                    score = match / total
                else:
                    # Penalty for wrong shape
                    score = 0.0
                
                total_score += score
                
            return total_score / len(examples)
            
        except Exception:
            return 0.0

    def _compose(self, f1: Callable, f2: Callable) -> Callable:
        """Compose two functions: f2(f1(x))"""
        return lambda x: f2(f1(x))

    def solve(self, examples: List[Tuple[np.ndarray, np.ndarray]], timeout_ms: int = 2000) -> Optional[Dict]:
        """
        Run beam search to find a program that solves the examples.
        """
        start_time = time.time()
        
        # Initialize beam with identity and single primitives
        beam = []
        
        # 0. Add Identity
        identity = lambda x: x
        node = ProgramNode(['identity'], identity, 0.0, 0)
        node.score = self._evaluate(node, examples)
        if node.score == 1.0:
            return {'program': ['identity'], 'description': 'Identity', 'confidence': 1.0}
        heapq.heappush(beam, node)
        
        # 1. Add Single Primitives
        for name, func in self.primitives.items():
            node = ProgramNode([name], func, 0.0, 1)
            node.score = self._evaluate(node, examples)
            if node.score == 1.0:
                return {'program': [name], 'description': f"Direct: {name}", 'confidence': 1.0}
            heapq.heappush(beam, node)
            
        # Keep top K
        current_beam = heapq.nlargest(self.beam_width, beam)
        
        # 2. Deep Search Loop
        for depth in range(2, self.max_depth + 1):
            if (time.time() - start_time) * 1000 > timeout_ms:
                break
                
            next_beam = []
            
            for node in current_beam:
                # Try extending this node with every primitive
                for prim_name, prim_func in self.primitives.items():
                    # Compose: new = prim_func(node.transform(x))
                    new_prog = node.program + [prim_name]
                    new_transform = self._compose(node.transform, prim_func)
                    
                    new_node = ProgramNode(new_prog, new_transform, 0.0, depth, parent=node)
                    
                    # Optimization: Don't fully evaluate everything, use heuristics or quick checks?
                    # For now, evaluate fully but catch exceptions
                    new_node.score = self._evaluate(new_node, examples)
                    
                    if new_node.score == 1.0:
                        desc = " -> ".join(new_prog)
                        return {
                            'program': new_prog,
                            'description': f"Synthesized: {desc}",
                            'confidence': 1.0
                        }
                    
                    # Only add reasonably good candidates to next beam to save memory
                    if new_node.score > 0.0:
                        heapq.heappush(next_beam, new_node)
            
            # Prune to beam width
            current_beam = heapq.nlargest(self.beam_width, next_beam)
            if not current_beam:
                break
                
        return None

    def synthesize_macro(self, 
                         examples: List[Tuple[np.ndarray, np.ndarray]], 
                         hint_primitives: List[str] = None) -> Optional[Callable]:
        """
        Attempt to synthesize a solution focusing on specific primitives if provided.
        """
        # This can be used by the UltraSolver to perform targeted synthesis
        # if the PatternEngine suggests certain types of operations.
        pass
