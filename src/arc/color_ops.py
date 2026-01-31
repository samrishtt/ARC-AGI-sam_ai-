"""
Color Mapping DSL - Parametric Color Operations

Many ARC tasks involve specific color mappings like:
- Replace color A with color B
- Swap colors A and B
- Map multiple colors

This module adds all possible color operations.
"""

import numpy as np
from typing import Dict, Callable, List, Tuple
from itertools import permutations, product


def create_replace_color(src: int, dst: int) -> Callable:
    """Create a function that replaces src color with dst."""
    def replace(grid: np.ndarray) -> np.ndarray:
        return np.where(grid == src, dst, grid)
    replace.__name__ = f"replace_{src}_to_{dst}"
    return replace


def create_swap_colors(c1: int, c2: int) -> Callable:
    """Create a function that swaps two colors."""
    def swap(grid: np.ndarray) -> np.ndarray:
        result = grid.copy()
        result[grid == c1] = c2
        result[grid == c2] = c1
        return result
    swap.__name__ = f"swap_{c1}_{c2}"
    return swap


def create_keep_only_color(color: int) -> Callable:
    """Create a function that keeps only one color."""
    def keep(grid: np.ndarray) -> np.ndarray:
        return np.where(grid == color, color, 0)
    keep.__name__ = f"keep_only_{color}"
    return keep


def create_remove_color(color: int) -> Callable:
    """Create a function that removes a color."""
    def remove(grid: np.ndarray) -> np.ndarray:
        return np.where(grid == color, 0, grid)
    remove.__name__ = f"remove_{color}"
    return remove


def create_set_nonzero_to(color: int) -> Callable:
    """Set all non-zero to a specific color."""
    def set_color(grid: np.ndarray) -> np.ndarray:
        return np.where(grid > 0, color, 0)
    set_color.__name__ = f"set_all_to_{color}"
    return set_color


def create_color_map(mapping: Dict[int, int]) -> Callable:
    """Create a function that applies a color mapping."""
    def apply_map(grid: np.ndarray) -> np.ndarray:
        result = grid.copy()
        for src, dst in mapping.items():
            result[grid == src] = dst
        return result
    apply_map.__name__ = f"color_map_{len(mapping)}"
    return apply_map


# Pre-generate common color operations
def get_color_operations() -> Dict[str, Callable]:
    """Get all color operations for colors 0-9."""
    ops = {}
    
    # Replace operations (excluding 0->0 etc)
    for src in range(1, 10):
        for dst in range(10):
            if src != dst:
                name = f"replace_{src}_to_{dst}"
                ops[name] = create_replace_color(src, dst)
    
    # Swap operations
    for c1 in range(1, 10):
        for c2 in range(c1 + 1, 10):
            name = f"swap_{c1}_{c2}"
            ops[name] = create_swap_colors(c1, c2)
    
    # Keep only color
    for c in range(1, 10):
        name = f"keep_only_{c}"
        ops[name] = create_keep_only_color(c)
    
    # Remove color
    for c in range(1, 10):
        name = f"remove_{c}"
        ops[name] = create_remove_color(c)
    
    # Set all to color
    for c in range(1, 10):
        name = f"set_all_to_{c}"
        ops[name] = create_set_nonzero_to(c)
    
    return ops


def infer_color_mapping(
    examples: List[Tuple[np.ndarray, np.ndarray]]
) -> Dict[int, int]:
    """
    Infer color mapping from input/output examples.
    
    Returns mapping if consistent across examples, else empty dict.
    """
    if not examples:
        return {}
    
    # Collect mappings from first example
    inp, out = examples[0]
    
    if inp.shape != out.shape:
        return {}
    
    mapping = {}
    for i in range(inp.shape[0]):
        for j in range(inp.shape[1]):
            src_color = inp[i, j]
            dst_color = out[i, j]
            
            if src_color in mapping:
                if mapping[src_color] != dst_color:
                    return {}  # Inconsistent
            else:
                mapping[src_color] = dst_color
    
    # Verify on other examples
    for inp, out in examples[1:]:
        if inp.shape != out.shape:
            return {}
        
        for i in range(inp.shape[0]):
            for j in range(inp.shape[1]):
                src_color = inp[i, j]
                dst_color = out[i, j]
                
                if src_color in mapping:
                    if mapping[src_color] != dst_color:
                        return {}
    
    # Only return if there's an actual change
    if all(k == v for k, v in mapping.items()):
        return {}
    
    return mapping


def create_transform_from_mapping(
    mapping: Dict[int, int]
) -> Callable:
    """Create a transform function from a color mapping."""
    def transform(grid: np.ndarray) -> np.ndarray:
        result = np.zeros_like(grid)
        for src, dst in mapping.items():
            result[grid == src] = dst
        return result
    return transform


# Generate all color ops at module load
COLOR_OPS = get_color_operations()
