"""
Pattern Engine for ARC-AGI

Intelligent pattern recognition and feature extraction for ARC grids.
Uses abstraction and analogy instead of brute-force search.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from rich.console import Console

console = Console()


class PatternType(Enum):
    """Categories of patterns found in ARC tasks."""
    GEOMETRIC = "geometric"      # Rotations, reflections, translations
    COLOR = "color"              # Color changes, mappings
    OBJECT = "object"            # Object-level operations
    STRUCTURAL = "structural"    # Size changes, cropping, scaling
    LOGICAL = "logical"          # AND, OR, XOR operations
    COUNTING = "counting"        # Counting and arithmetic
    COMPOSITE = "composite"      # Multiple pattern types


@dataclass
class GridFeatures:
    """Extracted features from an ARC grid."""
    shape: Tuple[int, int]
    colors: Set[int]
    num_colors: int
    color_counts: Dict[int, int]
    num_objects: int
    bounding_box: Tuple[int, int, int, int]  # min_r, min_c, max_r, max_c
    has_symmetry_h: bool
    has_symmetry_v: bool
    has_symmetry_diag: bool
    density: float  # non-zero pixels / total
    aspect_ratio: float
    is_square: bool
    unique_rows: int
    unique_cols: int
    objects: List[Tuple[int, np.ndarray]] = field(default_factory=list)  # (color, mask)


@dataclass
class TransformationHypothesis:
    """A hypothesis about the transformation between input and output."""
    pattern_type: PatternType
    description: str
    confidence: float
    transform_fn: Optional[Callable] = None
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return f"Hypothesis({self.pattern_type.value}: {self.description}, conf={self.confidence:.2f})"


class PatternEngine:
    """
    Intelligent pattern recognition engine for ARC-AGI.
    
    Instead of brute-force search, this engine:
    1. Extracts high-level features from grids
    2. Compares input/output features to hypothesize transformations
    3. Ranks hypotheses by likelihood
    4. Applies transformations intelligently
    """
    
    def __init__(self):
        self.feature_cache: Dict[str, GridFeatures] = {}
        
    def extract_features(self, grid: np.ndarray) -> GridFeatures:
        """Extract comprehensive features from a grid."""
        rows, cols = grid.shape
        
        # Color analysis
        unique_colors = set(grid.flatten())
        color_counts = {}
        for c in unique_colors:
            color_counts[c] = int(np.sum(grid == c))
        
        # Bounding box of non-zero content
        non_zero = np.argwhere(grid > 0)
        if len(non_zero) > 0:
            min_r, min_c = non_zero.min(axis=0)
            max_r, max_c = non_zero.max(axis=0)
            bbox = (int(min_r), int(min_c), int(max_r), int(max_c))
        else:
            bbox = (0, 0, 0, 0)
        
        # Symmetry checks
        sym_h = np.array_equal(grid, np.flipud(grid))
        sym_v = np.array_equal(grid, np.fliplr(grid))
        sym_diag = rows == cols and np.array_equal(grid, grid.T)
        
        # Density
        non_zero_count = np.count_nonzero(grid)
        density = non_zero_count / (rows * cols) if rows * cols > 0 else 0
        
        # Aspect ratio
        aspect = rows / cols if cols > 0 else 1
        
        # Unique rows/cols (for detecting patterns)
        unique_rows = len(set(tuple(row) for row in grid))
        unique_cols = len(set(tuple(col) for col in grid.T))
        
        # Object detection
        objects = self._find_objects(grid)
        
        return GridFeatures(
            shape=(rows, cols),
            colors=unique_colors,
            num_colors=len(unique_colors),
            color_counts=color_counts,
            num_objects=len(objects),
            bounding_box=bbox,
            has_symmetry_h=sym_h,
            has_symmetry_v=sym_v,
            has_symmetry_diag=sym_diag,
            density=density,
            aspect_ratio=aspect,
            is_square=(rows == cols),
            unique_rows=unique_rows,
            unique_cols=unique_cols,
            objects=objects
        )
    
    def _find_objects(self, grid: np.ndarray, connectivity: int = 4) -> List[Tuple[int, np.ndarray]]:
        """Find connected components (objects) in the grid."""
        objects = []
        visited = np.zeros_like(grid, dtype=bool)
        rows, cols = grid.shape
        
        def get_neighbors(r, c):
            deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if connectivity == 8:
                deltas += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            for dr, dc in deltas:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    yield nr, nc
        
        for r in range(rows):
            for c in range(cols):
                if not visited[r, c] and grid[r, c] != 0:
                    color = grid[r, c]
                    mask = np.zeros_like(grid, dtype=bool)
                    stack = [(r, c)]
                    visited[r, c] = True
                    mask[r, c] = True
                    
                    while stack:
                        curr_r, curr_c = stack.pop()
                        for nr, nc in get_neighbors(curr_r, curr_c):
                            if not visited[nr, nc] and grid[nr, nc] == color:
                                visited[nr, nc] = True
                                mask[nr, nc] = True
                                stack.append((nr, nc))
                    
                    objects.append((int(color), mask))
        
        return objects
    
    def compare_features(self, input_feat: GridFeatures, output_feat: GridFeatures) -> List[str]:
        """Compare features between input and output to identify what changed."""
        changes = []
        
        if input_feat.shape != output_feat.shape:
            changes.append(f"shape_change:{input_feat.shape}->{output_feat.shape}")
            
            # Check for scaling
            if (output_feat.shape[0] % input_feat.shape[0] == 0 and 
                output_feat.shape[1] % input_feat.shape[1] == 0):
                scale_r = output_feat.shape[0] // input_feat.shape[0]
                scale_c = output_feat.shape[1] // input_feat.shape[1]
                if scale_r == scale_c:
                    changes.append(f"scale_up:{scale_r}x")
            elif (input_feat.shape[0] % output_feat.shape[0] == 0 and 
                  input_feat.shape[1] % output_feat.shape[1] == 0):
                changes.append("crop_or_scale_down")
        
        if input_feat.colors != output_feat.colors:
            added = output_feat.colors - input_feat.colors
            removed = input_feat.colors - output_feat.colors
            if added:
                changes.append(f"colors_added:{added}")
            if removed:
                changes.append(f"colors_removed:{removed}")
        
        if input_feat.num_objects != output_feat.num_objects:
            changes.append(f"object_count:{input_feat.num_objects}->{output_feat.num_objects}")
        
        if input_feat.has_symmetry_h != output_feat.has_symmetry_h:
            changes.append("symmetry_h_changed")
        if input_feat.has_symmetry_v != output_feat.has_symmetry_v:
            changes.append("symmetry_v_changed")
            
        if abs(input_feat.density - output_feat.density) > 0.1:
            changes.append(f"density_change:{input_feat.density:.2f}->{output_feat.density:.2f}")
        
        return changes
    
    def hypothesize_transformation(
        self, 
        input_grid: np.ndarray, 
        output_grid: np.ndarray
    ) -> List[TransformationHypothesis]:
        """
        Generate hypotheses about what transformation was applied.
        Ranked by confidence.
        """
        hypotheses = []
        
        input_feat = self.extract_features(input_grid)
        output_feat = self.extract_features(output_grid)
        
        # Check identity (no change)
        if np.array_equal(input_grid, output_grid):
            hypotheses.append(TransformationHypothesis(
                PatternType.GEOMETRIC, "identity", 1.0,
                lambda x: x
            ))
            return hypotheses
        
        # Check geometric transformations
        hypotheses.extend(self._check_geometric(input_grid, output_grid))
        
        # Check color transformations
        hypotheses.extend(self._check_color(input_grid, output_grid, input_feat, output_feat))
        
        # Check structural transformations
        hypotheses.extend(self._check_structural(input_grid, output_grid, input_feat, output_feat))
        
        # Check object-based transformations  
        hypotheses.extend(self._check_object(input_grid, output_grid, input_feat, output_feat))
        
        # Check logical/counting transformations
        hypotheses.extend(self._check_logical(input_grid, output_grid, input_feat, output_feat))
        
        # Sort by confidence
        hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        
        return hypotheses
    
    def _check_geometric(self, input_grid: np.ndarray, output_grid: np.ndarray) -> List[TransformationHypothesis]:
        """Check for geometric transformations."""
        hypotheses = []
        
        # Rotation 90 CW
        rotated_cw = np.rot90(input_grid, k=-1)
        if np.array_equal(rotated_cw, output_grid):
            hypotheses.append(TransformationHypothesis(
                PatternType.GEOMETRIC, "rotate_90_cw", 1.0,
                lambda x: np.rot90(x, k=-1)
            ))
        
        # Rotation 90 CCW  
        rotated_ccw = np.rot90(input_grid, k=1)
        if np.array_equal(rotated_ccw, output_grid):
            hypotheses.append(TransformationHypothesis(
                PatternType.GEOMETRIC, "rotate_90_ccw", 1.0,
                lambda x: np.rot90(x, k=1)
            ))
        
        # Rotation 180
        rotated_180 = np.rot90(input_grid, k=2)
        if np.array_equal(rotated_180, output_grid):
            hypotheses.append(TransformationHypothesis(
                PatternType.GEOMETRIC, "rotate_180", 1.0,
                lambda x: np.rot90(x, k=2)
            ))
        
        # Horizontal flip
        flipped_h = np.fliplr(input_grid)
        if np.array_equal(flipped_h, output_grid):
            hypotheses.append(TransformationHypothesis(
                PatternType.GEOMETRIC, "reflect_horizontal", 1.0,
                lambda x: np.fliplr(x)
            ))
        
        # Vertical flip
        flipped_v = np.flipud(input_grid)
        if np.array_equal(flipped_v, output_grid):
            hypotheses.append(TransformationHypothesis(
                PatternType.GEOMETRIC, "reflect_vertical", 1.0,
                lambda x: np.flipud(x)
            ))
        
        # Transpose
        if input_grid.shape[0] == input_grid.shape[1]:
            transposed = np.transpose(input_grid)
            if np.array_equal(transposed, output_grid):
                hypotheses.append(TransformationHypothesis(
                    PatternType.GEOMETRIC, "transpose", 1.0,
                    lambda x: np.transpose(x)
                ))
        
        return hypotheses
    
    def _check_color(
        self, 
        input_grid: np.ndarray, 
        output_grid: np.ndarray,
        input_feat: GridFeatures,
        output_feat: GridFeatures
    ) -> List[TransformationHypothesis]:
        """Check for color-based transformations."""
        hypotheses = []
        
        # Same shape required for direct color mapping
        if input_feat.shape != output_feat.shape:
            return hypotheses
        
        # Build color mapping
        color_map = {}
        valid_mapping = True
        
        for r in range(input_feat.shape[0]):
            for c in range(input_feat.shape[1]):
                in_color = input_grid[r, c]
                out_color = output_grid[r, c]
                
                if in_color in color_map:
                    if color_map[in_color] != out_color:
                        valid_mapping = False
                        break
                else:
                    color_map[in_color] = out_color
            if not valid_mapping:
                break
        
        if valid_mapping and len(color_map) > 0:
            # Check if it's a simple replacement
            non_trivial = {k: v for k, v in color_map.items() if k != v}
            
            if len(non_trivial) == 1:
                # Single color replacement
                src, dst = list(non_trivial.items())[0]
                hypotheses.append(TransformationHypothesis(
                    PatternType.COLOR, f"replace_color_{src}_with_{dst}", 0.95,
                    lambda x, s=src, d=dst: np.where(x == s, d, x),
                    {"source": src, "target": dst}
                ))
            elif len(non_trivial) > 1:
                hypotheses.append(TransformationHypothesis(
                    PatternType.COLOR, f"color_map:{non_trivial}", 0.85,
                    lambda x, m=color_map.copy(): self._apply_color_map(x, m),
                    {"mapping": color_map.copy()}
                ))
        
        # Check for inversion (swap 0 and non-zero)
        if len(input_feat.colors) == 2 and 0 in input_feat.colors:
            other_color = (input_feat.colors - {0}).pop()
            inverted = np.where(input_grid == 0, other_color, 0)
            if np.array_equal(inverted, output_grid):
                hypotheses.append(TransformationHypothesis(
                    PatternType.COLOR, "invert_binary", 0.95,
                    lambda x, c=other_color: np.where(x == 0, c, 0)
                ))
        
        return hypotheses
    
    def _apply_color_map(self, grid: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
        """Apply a color mapping to a grid."""
        result = grid.copy()
        for src, dst in mapping.items():
            result = np.where(grid == src, dst, result)
        return result
    
    def _check_structural(
        self, 
        input_grid: np.ndarray, 
        output_grid: np.ndarray,
        input_feat: GridFeatures,
        output_feat: GridFeatures
    ) -> List[TransformationHypothesis]:
        """Check for structural transformations (size changes)."""
        hypotheses = []
        
        # Crop to non-zero content
        coords = np.argwhere(input_grid > 0)
        if len(coords) > 0:
            min_r, min_c = coords.min(axis=0)
            max_r, max_c = coords.max(axis=0)
            cropped = input_grid[min_r:max_r+1, min_c:max_c+1]
            if np.array_equal(cropped, output_grid):
                hypotheses.append(TransformationHypothesis(
                    PatternType.STRUCTURAL, "crop_to_content", 1.0,
                    self._crop_to_content
                ))
        
        # Scale up
        if (output_feat.shape[0] % input_feat.shape[0] == 0 and 
            output_feat.shape[1] % input_feat.shape[1] == 0):
            scale_r = output_feat.shape[0] // input_feat.shape[0]
            scale_c = output_feat.shape[1] // input_feat.shape[1]
            if scale_r == scale_c and scale_r > 1:
                scaled = np.repeat(np.repeat(input_grid, scale_r, axis=0), scale_c, axis=1)
                if np.array_equal(scaled, output_grid):
                    hypotheses.append(TransformationHypothesis(
                        PatternType.STRUCTURAL, f"scale_up_{scale_r}x", 1.0,
                        lambda x, s=scale_r: np.repeat(np.repeat(x, s, axis=0), s, axis=1),
                        {"scale": scale_r}
                    ))
        
        # Tile/Repeat
        if (output_feat.shape[0] % input_feat.shape[0] == 0 and 
            output_feat.shape[1] % input_feat.shape[1] == 0):
            tile_r = output_feat.shape[0] // input_feat.shape[0]
            tile_c = output_feat.shape[1] // input_feat.shape[1]
            tiled = np.tile(input_grid, (tile_r, tile_c))
            if np.array_equal(tiled, output_grid):
                hypotheses.append(TransformationHypothesis(
                    PatternType.STRUCTURAL, f"tile_{tile_r}x{tile_c}", 1.0,
                    lambda x, tr=tile_r, tc=tile_c: np.tile(x, (tr, tc)),
                    {"tile_rows": tile_r, "tile_cols": tile_c}
                ))
        
        return hypotheses
    
    def _crop_to_content(self, grid: np.ndarray) -> np.ndarray:
        """Crop grid to bounding box of non-zero content."""
        coords = np.argwhere(grid > 0)
        if len(coords) == 0:
            return grid
        min_r, min_c = coords.min(axis=0)
        max_r, max_c = coords.max(axis=0)
        return grid[min_r:max_r+1, min_c:max_c+1]
    
    def _check_object(
        self, 
        input_grid: np.ndarray, 
        output_grid: np.ndarray,
        input_feat: GridFeatures,
        output_feat: GridFeatures
    ) -> List[TransformationHypothesis]:
        """Check for object-based transformations."""
        hypotheses = []
        
        # Keep largest object
        if input_feat.num_objects > 1 and output_feat.num_objects == 1:
            # Find largest object in input
            if input_feat.objects:
                largest_obj = max(input_feat.objects, key=lambda x: np.sum(x[1]))
                result = np.zeros_like(input_grid)
                result[largest_obj[1]] = largest_obj[0]
                if np.array_equal(result, output_grid):
                    hypotheses.append(TransformationHypothesis(
                        PatternType.OBJECT, "keep_largest_object", 0.95,
                        self._keep_largest_object
                    ))
        
        # Check for gravity (dropping objects down)
        if input_feat.shape == output_feat.shape:
            dropped = self._apply_gravity(input_grid)
            if np.array_equal(dropped, output_grid):
                hypotheses.append(TransformationHypothesis(
                    PatternType.OBJECT, "apply_gravity", 0.95,
                    self._apply_gravity
                ))
        
        # Check for filling patterns (extend object in direction)
        if input_feat.shape == output_feat.shape:
            for direction in ['down', 'up', 'left', 'right']:
                filled = self._fill_direction(input_grid, direction)
                if np.array_equal(filled, output_grid):
                    hypotheses.append(TransformationHypothesis(
                        PatternType.OBJECT, f"fill_{direction}", 0.9,
                        lambda x, d=direction: self._fill_direction(x, d)
                    ))
        
        return hypotheses
    
    def _keep_largest_object(self, grid: np.ndarray) -> np.ndarray:
        """Keep only the largest connected component."""
        objects = self._find_objects(grid)
        if not objects:
            return grid
        largest = max(objects, key=lambda x: np.sum(x[1]))
        result = np.zeros_like(grid)
        result[largest[1]] = largest[0]
        return result
    
    def _apply_gravity(self, grid: np.ndarray) -> np.ndarray:
        """Drop all pixels down (gravity)."""
        result = np.zeros_like(grid)
        rows, cols = grid.shape
        
        for c in range(cols):
            column = grid[:, c]
            non_zero = column[column > 0]
            if len(non_zero) > 0:
                result[rows-len(non_zero):rows, c] = non_zero
        
        return result
    
    def _fill_direction(self, grid: np.ndarray, direction: str) -> np.ndarray:
        """Fill/extend non-zero pixels in a given direction."""
        result = grid.copy()
        rows, cols = grid.shape
        
        if direction == 'down':
            for c in range(cols):
                for r in range(rows):
                    if grid[r, c] > 0:
                        result[r:, c] = grid[r, c]
        elif direction == 'up':
            for c in range(cols):
                for r in range(rows-1, -1, -1):
                    if grid[r, c] > 0:
                        result[:r+1, c] = grid[r, c]
        elif direction == 'right':
            for r in range(rows):
                for c in range(cols):
                    if grid[r, c] > 0:
                        result[r, c:] = grid[r, c]
        elif direction == 'left':
            for r in range(rows):
                for c in range(cols-1, -1, -1):
                    if grid[r, c] > 0:
                        result[r, :c+1] = grid[r, c]
        
        return result
    
    def _check_logical(
        self, 
        input_grid: np.ndarray, 
        output_grid: np.ndarray,
        input_feat: GridFeatures,
        output_feat: GridFeatures
    ) -> List[TransformationHypothesis]:
        """Check for logical and counting transformations."""
        hypotheses = []
        
        # Counting: output is a single number representing count
        if output_feat.shape == (1, 1):
            count = output_grid[0, 0]
            if count == input_feat.num_objects:
                hypotheses.append(TransformationHypothesis(
                    PatternType.COUNTING, "count_objects", 0.9,
                    lambda x: np.array([[len(self._find_objects(x))]])
                ))
            non_zero_count = np.count_nonzero(input_grid)
            if count == non_zero_count:
                hypotheses.append(TransformationHypothesis(
                    PatternType.COUNTING, "count_nonzero", 0.9,
                    lambda x: np.array([[np.count_nonzero(x)]])
                ))
        
        return hypotheses
    
    def find_consistent_transformation(
        self, 
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[TransformationHypothesis]:
        """
        Find a transformation that works for ALL examples.
        This is the key to ARC: induction from examples.
        """
        if not examples:
            return None
        
        # Get hypotheses for first example
        first_input, first_output = examples[0]
        candidates = self.hypothesize_transformation(first_input, first_output)
        
        if not candidates:
            return None
        
        # Filter to those that work for all examples
        valid_candidates = []
        
        for hypothesis in candidates:
            if hypothesis.transform_fn is None:
                continue
            
            all_valid = True
            for input_grid, output_grid in examples:
                try:
                    result = hypothesis.transform_fn(input_grid)
                    if not np.array_equal(result, output_grid):
                        all_valid = False
                        break
                except Exception:
                    all_valid = False
                    break
            
            if all_valid:
                valid_candidates.append(hypothesis)
        
        if valid_candidates:
            # Return highest confidence
            return max(valid_candidates, key=lambda h: h.confidence)
        
        return None
    
    def grid_to_str(self, grid: np.ndarray) -> str:
        """Convert grid to colored string representation."""
        color_chars = "0123456789"
        lines = []
        for row in grid:
            lines.append(''.join(color_chars[min(c, 9)] for c in row))
        return '\n'.join(lines)
