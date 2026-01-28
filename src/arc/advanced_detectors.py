"""
Advanced Pattern Detectors for ARC-AGI

These specialized detectors handle common ARC transformation patterns
that require deeper analysis than simple geometric transforms.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Callable, Any
from dataclasses import dataclass


@dataclass
class DetectionResult:
    """Result of a pattern detection attempt."""
    detected: bool
    transform_fn: Optional[Callable] = None
    description: str = ""
    confidence: float = 0.0
    parameters: Dict[str, Any] = None


class AdvancedPatternDetectors:
    """
    Collection of specialized pattern detectors for ARC tasks.
    """
    
    @staticmethod
    def detect_color_mapping(examples: List[Tuple[np.ndarray, np.ndarray]]) -> DetectionResult:
        """
        Detect if there's a consistent color mapping between input and output.
        Works even when shapes differ (maps pixel-by-pixel where possible).
        """
        if not examples:
            return DetectionResult(False)
        
        # Collect color mappings from all examples
        color_map = {}
        
        for input_grid, output_grid in examples:
            if input_grid.shape != output_grid.shape:
                continue
                
            for r in range(input_grid.shape[0]):
                for c in range(input_grid.shape[1]):
                    in_c = int(input_grid[r, c])
                    out_c = int(output_grid[r, c])
                    
                    if in_c in color_map:
                        if color_map[in_c] != out_c:
                            return DetectionResult(False)
                    else:
                        color_map[in_c] = out_c
        
        if not color_map:
            return DetectionResult(False)
        
        # Check if mapping is non-trivial
        non_trivial = {k: v for k, v in color_map.items() if k != v}
        if not non_trivial:
            return DetectionResult(False)
        
        def transform(grid, mapping=color_map.copy()):
            result = grid.copy()
            for src, dst in mapping.items():
                result = np.where(grid == src, dst, result)
            return result
        
        return DetectionResult(
            detected=True,
            transform_fn=transform,
            description=f"color_map:{non_trivial}",
            confidence=0.95,
            parameters={"mapping": color_map}
        )
    
    @staticmethod
    def detect_border_extraction(examples: List[Tuple[np.ndarray, np.ndarray]]) -> DetectionResult:
        """
        Detect if the output is the input with its border removed.
        """
        for input_grid, output_grid in examples:
            if input_grid.shape[0] <= 2 or input_grid.shape[1] <= 2:
                return DetectionResult(False)
            
            expected = input_grid[1:-1, 1:-1]
            if not np.array_equal(expected, output_grid):
                return DetectionResult(False)
        
        def transform(grid):
            return grid[1:-1, 1:-1]
        
        return DetectionResult(
            detected=True,
            transform_fn=transform,
            description="remove_border",
            confidence=0.95
        )
    
    @staticmethod
    def detect_grid_splitting(examples: List[Tuple[np.ndarray, np.ndarray]]) -> DetectionResult:
        """
        Detect if the output is a specific quadrant/section of the input.
        """
        for input_grid, output_grid in examples:
            rows, cols = input_grid.shape
            out_rows, out_cols = output_grid.shape
            
            # Check various splits
            splits = [
                ("top_half", input_grid[:rows//2, :]),
                ("bottom_half", input_grid[rows//2:, :]),
                ("left_half", input_grid[:, :cols//2]),
                ("right_half", input_grid[:, cols//2:]),
                ("top_left", input_grid[:rows//2, :cols//2]),
                ("top_right", input_grid[:rows//2, cols//2:]),
                ("bottom_left", input_grid[rows//2:, :cols//2]),
                ("bottom_right", input_grid[rows//2:, cols//2:]),
            ]
            
            for name, section in splits:
                if np.array_equal(section, output_grid):
                    # Verify with other examples
                    all_match = True
                    for inp, out in examples:
                        r, c = inp.shape
                        if name == "top_half":
                            expected = inp[:r//2, :]
                        elif name == "bottom_half":
                            expected = inp[r//2:, :]
                        elif name == "left_half":
                            expected = inp[:, :c//2]
                        elif name == "right_half":
                            expected = inp[:, c//2:]
                        elif name == "top_left":
                            expected = inp[:r//2, :c//2]
                        elif name == "top_right":
                            expected = inp[:r//2, c//2:]
                        elif name == "bottom_left":
                            expected = inp[r//2:, :c//2]
                        elif name == "bottom_right":
                            expected = inp[r//2:, c//2:]
                        
                        if not np.array_equal(expected, out):
                            all_match = False
                            break
                    
                    if all_match:
                        def make_transform(split_name):
                            def transform(grid):
                                r, c = grid.shape
                                if split_name == "top_half":
                                    return grid[:r//2, :]
                                elif split_name == "bottom_half":
                                    return grid[r//2:, :]
                                elif split_name == "left_half":
                                    return grid[:, :c//2]
                                elif split_name == "right_half":
                                    return grid[:, c//2:]
                                elif split_name == "top_left":
                                    return grid[:r//2, :c//2]
                                elif split_name == "top_right":
                                    return grid[:r//2, c//2:]
                                elif split_name == "bottom_left":
                                    return grid[r//2:, :c//2]
                                elif split_name == "bottom_right":
                                    return grid[r//2:, c//2:]
                                return grid
                            return transform
                        
                        return DetectionResult(
                            detected=True,
                            transform_fn=make_transform(name),
                            description=f"extract_{name}",
                            confidence=0.9
                        )
        
        return DetectionResult(False)
    
    @staticmethod
    def detect_replication(examples: List[Tuple[np.ndarray, np.ndarray]]) -> DetectionResult:
        """
        Detect if the output is the input replicated/tiled.
        """
        for input_grid, output_grid in examples:
            in_rows, in_cols = input_grid.shape
            out_rows, out_cols = output_grid.shape
            
            if out_rows % in_rows == 0 and out_cols % in_cols == 0:
                tile_r = out_rows // in_rows
                tile_c = out_cols // in_cols
                
                tiled = np.tile(input_grid, (tile_r, tile_c))
                if np.array_equal(tiled, output_grid):
                    # Verify with other examples
                    all_match = True
                    for inp, out in examples:
                        ir, ic = inp.shape
                        orr, oc = out.shape
                        if orr % ir != 0 or oc % ic != 0:
                            all_match = False
                            break
                        tr = orr // ir
                        tc = oc // ic
                        if tr != tile_r or tc != tile_c:
                            all_match = False
                            break
                        if not np.array_equal(np.tile(inp, (tr, tc)), out):
                            all_match = False
                            break
                    
                    if all_match:
                        def make_transform(tr, tc):
                            def transform(grid):
                                return np.tile(grid, (tr, tc))
                            return transform
                        
                        return DetectionResult(
                            detected=True,
                            transform_fn=make_transform(tile_r, tile_c),
                            description=f"tile_{tile_r}x{tile_c}",
                            confidence=0.95
                        )
        
        return DetectionResult(False)
    
    @staticmethod
    def detect_mask_application(examples: List[Tuple[np.ndarray, np.ndarray]]) -> DetectionResult:
        """
        Detect if output is input with a mask/pattern applied.
        E.g., keeping only certain positions based on color.
        """
        for input_grid, output_grid in examples:
            if input_grid.shape != output_grid.shape:
                continue
            
            # Check if output is a subset of input (some pixels zeroed)
            mask = output_grid != 0
            if np.all(input_grid[mask] == output_grid[mask]):
                # Output keeps some input pixels and zeros others
                # Find the rule (which color is kept)
                kept_colors = set(output_grid[output_grid != 0])
                if len(kept_colors) == 1:
                    kept_color = kept_colors.pop()
                    
                    # Verify
                    all_match = True
                    for inp, out in examples:
                        expected = np.where(inp == kept_color, inp, 0)
                        if not np.array_equal(expected, out):
                            all_match = False
                            break
                    
                    if all_match:
                        def make_transform(color):
                            def transform(grid):
                                return np.where(grid == color, grid, 0)
                            return transform
                        
                        return DetectionResult(
                            detected=True,
                            transform_fn=make_transform(kept_color),
                            description=f"keep_color_{kept_color}",
                            confidence=0.9
                        )
        
        return DetectionResult(False)
    
    @staticmethod
    def detect_sort_colors(examples: List[Tuple[np.ndarray, np.ndarray]]) -> DetectionResult:
        """
        Detect if output has colors sorted in some way.
        E.g., colors pushed to one side.
        """
        # Check for gravity-like behavior
        for direction in ['down', 'up', 'left', 'right']:
            all_match = True
            
            for input_grid, output_grid in examples:
                if input_grid.shape != output_grid.shape:
                    all_match = False
                    break
                
                expected = AdvancedPatternDetectors._apply_gravity(input_grid, direction)
                if not np.array_equal(expected, output_grid):
                    all_match = False
                    break
            
            if all_match:
                def make_transform(dir):
                    def transform(grid):
                        return AdvancedPatternDetectors._apply_gravity(grid, dir)
                    return transform
                
                return DetectionResult(
                    detected=True,
                    transform_fn=make_transform(direction),
                    description=f"gravity_{direction}",
                    confidence=0.95
                )
        
        return DetectionResult(False)
    
    @staticmethod
    def _apply_gravity(grid: np.ndarray, direction: str) -> np.ndarray:
        """Apply gravity in a direction."""
        result = np.zeros_like(grid)
        rows, cols = grid.shape
        
        if direction == 'down':
            for c in range(cols):
                col = grid[:, c]
                non_zero = col[col > 0]
                if len(non_zero) > 0:
                    result[rows-len(non_zero):rows, c] = non_zero
        elif direction == 'up':
            for c in range(cols):
                col = grid[:, c]
                non_zero = col[col > 0]
                if len(non_zero) > 0:
                    result[:len(non_zero), c] = non_zero
        elif direction == 'left':
            for r in range(rows):
                row = grid[r, :]
                non_zero = row[row > 0]
                if len(non_zero) > 0:
                    result[r, :len(non_zero)] = non_zero
        elif direction == 'right':
            for r in range(rows):
                row = grid[r, :]
                non_zero = row[row > 0]
                if len(non_zero) > 0:
                    result[r, cols-len(non_zero):cols] = non_zero
        
        return result
    
    @staticmethod
    def detect_unique_row_col(examples: List[Tuple[np.ndarray, np.ndarray]]) -> DetectionResult:
        """
        Detect if output extracts unique rows or columns.
        """
        for mode in ['unique_rows', 'unique_cols']:
            all_match = True
            
            for input_grid, output_grid in examples:
                if mode == 'unique_rows':
                    seen = set()
                    unique = []
                    for row in input_grid:
                        row_tuple = tuple(row)
                        if row_tuple not in seen:
                            seen.add(row_tuple)
                            unique.append(row)
                    if unique:
                        expected = np.array(unique)
                    else:
                        expected = input_grid
                else:
                    seen = set()
                    unique = []
                    for c in range(input_grid.shape[1]):
                        col_tuple = tuple(input_grid[:, c])
                        if col_tuple not in seen:
                            seen.add(col_tuple)
                            unique.append(input_grid[:, c])
                    if unique:
                        expected = np.column_stack(unique)
                    else:
                        expected = input_grid
                
                if not np.array_equal(expected, output_grid):
                    all_match = False
                    break
            
            if all_match:
                def make_transform(m):
                    def transform(grid):
                        if m == 'unique_rows':
                            seen = set()
                            unique = []
                            for row in grid:
                                row_tuple = tuple(row)
                                if row_tuple not in seen:
                                    seen.add(row_tuple)
                                    unique.append(row)
                            return np.array(unique) if unique else grid
                        else:
                            seen = set()
                            unique = []
                            for c in range(grid.shape[1]):
                                col_tuple = tuple(grid[:, c])
                                if col_tuple not in seen:
                                    seen.add(col_tuple)
                                    unique.append(grid[:, c])
                            return np.column_stack(unique) if unique else grid
                    return transform
                
                return DetectionResult(
                    detected=True,
                    transform_fn=make_transform(mode),
                    description=mode,
                    confidence=0.85
                )
        
        return DetectionResult(False)
    
    @staticmethod
    def detect_mirror_completion(examples: List[Tuple[np.ndarray, np.ndarray]]) -> DetectionResult:
        """
        Detect if output completes the grid to be symmetric.
        """
        for axis in ['horizontal', 'vertical', 'both']:
            all_match = True
            
            for input_grid, output_grid in examples:
                if input_grid.shape != output_grid.shape:
                    all_match = False
                    break
                
                if axis == 'horizontal':
                    expected = input_grid.copy()
                    rows = expected.shape[0]
                    for r in range(rows // 2):
                        for c in range(expected.shape[1]):
                            if expected[r, c] == 0 and expected[rows-1-r, c] != 0:
                                expected[r, c] = expected[rows-1-r, c]
                            elif expected[rows-1-r, c] == 0 and expected[r, c] != 0:
                                expected[rows-1-r, c] = expected[r, c]
                elif axis == 'vertical':
                    expected = input_grid.copy()
                    cols = expected.shape[1]
                    for r in range(expected.shape[0]):
                        for c in range(cols // 2):
                            if expected[r, c] == 0 and expected[r, cols-1-c] != 0:
                                expected[r, c] = expected[r, cols-1-c]
                            elif expected[r, cols-1-c] == 0 and expected[r, c] != 0:
                                expected[r, cols-1-c] = expected[r, c]
                else:
                    expected = input_grid.copy()
                    rows, cols = expected.shape
                    # Vertical first
                    for r in range(rows):
                        for c in range(cols // 2):
                            if expected[r, c] == 0 and expected[r, cols-1-c] != 0:
                                expected[r, c] = expected[r, cols-1-c]
                            elif expected[r, cols-1-c] == 0 and expected[r, c] != 0:
                                expected[r, cols-1-c] = expected[r, c]
                    # Then horizontal
                    for r in range(rows // 2):
                        for c in range(cols):
                            if expected[r, c] == 0 and expected[rows-1-r, c] != 0:
                                expected[r, c] = expected[rows-1-r, c]
                            elif expected[rows-1-r, c] == 0 and expected[r, c] != 0:
                                expected[rows-1-r, c] = expected[r, c]
                
                if not np.array_equal(expected, output_grid):
                    all_match = False
                    break
            
            if all_match:
                def make_transform(ax):
                    def transform(grid):
                        result = grid.copy()
                        rows, cols = result.shape
                        if ax in ['horizontal', 'both']:
                            for r in range(rows // 2):
                                for c in range(cols):
                                    if result[r, c] == 0 and result[rows-1-r, c] != 0:
                                        result[r, c] = result[rows-1-r, c]
                                    elif result[rows-1-r, c] == 0 and result[r, c] != 0:
                                        result[rows-1-r, c] = result[r, c]
                        if ax in ['vertical', 'both']:
                            for r in range(rows):
                                for c in range(cols // 2):
                                    if result[r, c] == 0 and result[r, cols-1-c] != 0:
                                        result[r, c] = result[r, cols-1-c]
                                    elif result[r, cols-1-c] == 0 and result[r, c] != 0:
                                        result[r, cols-1-c] = result[r, c]
                        return result
                    return transform
                
                return DetectionResult(
                    detected=True,
                    transform_fn=make_transform(axis),
                    description=f"mirror_complete_{axis}",
                    confidence=0.85
                )
        
        return DetectionResult(False)
    
    @staticmethod
    def get_all_detectors() -> List[Callable]:
        """Get all detector functions."""
        return [
            AdvancedPatternDetectors.detect_color_mapping,
            AdvancedPatternDetectors.detect_border_extraction,
            AdvancedPatternDetectors.detect_grid_splitting,
            AdvancedPatternDetectors.detect_replication,
            AdvancedPatternDetectors.detect_mask_application,
            AdvancedPatternDetectors.detect_sort_colors,
            AdvancedPatternDetectors.detect_unique_row_col,
            AdvancedPatternDetectors.detect_mirror_completion,
        ]
