"""
Neural Pattern Matcher for ARC-AGI

Uses learned embeddings and similarity matching to identify
transformation patterns. Works alongside symbolic reasoning.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import hashlib
import json


@dataclass
class PatternSignature:
    """Compact signature of a grid pattern for similarity matching."""
    shape_ratio: float              # height/width ratio
    density: float                  # non-zero / total
    color_histogram: Tuple[int, ...]  # counts for colors 0-9
    edge_density: float             # edges / total
    symmetry_score: float           # 0-1 symmetry measure
    object_count: int               # number of connected components
    unique_colors: int              # number of distinct colors
    
    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for similarity computation."""
        return np.array([
            self.shape_ratio,
            self.density,
            *self.color_histogram,
            self.edge_density,
            self.symmetry_score,
            self.object_count / 10,  # Normalize
            self.unique_colors / 10
        ], dtype=np.float32)
    
    def similarity(self, other: 'PatternSignature') -> float:
        """Compute cosine similarity with another signature."""
        v1 = self.to_vector()
        v2 = other.to_vector()
        
        # Normalize
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        
        if n1 < 1e-8 or n2 < 1e-8:
            return 0.0
        
        return float(np.dot(v1, v2) / (n1 * n2))


@dataclass
class TransformSignature:
    """Signature capturing how input transforms to output."""
    shape_change: Tuple[int, int]  # (delta_rows, delta_cols)
    density_change: float
    color_mapping: Dict[int, int]
    structure_preserved: bool
    symmetry_introduced: bool
    object_count_change: int


class NeuralPatternMatcher:
    """
    Pattern matching using learned representations.
    
    This is a lightweight version that doesn't require PyTorch/TensorFlow.
    Uses handcrafted features + nearest neighbor matching.
    """
    
    def __init__(self):
        # Pattern database: maps signature hash to (transform_name, transform_fn)
        self.pattern_db: Dict[str, Tuple[str, Any]] = {}
        
        # Transform templates learned from examples
        self.transform_templates: List[Tuple[TransformSignature, str]] = []
    
    def compute_signature(self, grid: np.ndarray) -> PatternSignature:
        """Compute the signature of a grid."""
        rows, cols = grid.shape
        
        # Shape ratio
        shape_ratio = rows / cols if cols > 0 else 1.0
        
        # Density
        non_zero = np.count_nonzero(grid)
        density = non_zero / (rows * cols) if rows * cols > 0 else 0
        
        # Color histogram (0-9)
        histogram = [0] * 10
        for c in range(10):
            histogram[c] = int(np.sum(grid == c))
        
        # Edge density (Sobel-like)
        edge_count = 0
        for r in range(rows):
            for c in range(cols):
                if r > 0 and grid[r, c] != grid[r-1, c]:
                    edge_count += 1
                if c > 0 and grid[r, c] != grid[r, c-1]:
                    edge_count += 1
        edge_density = edge_count / (2 * rows * cols) if rows * cols > 0 else 0
        
        # Symmetry score
        sym_h = np.array_equal(grid, np.flipud(grid))
        sym_v = np.array_equal(grid, np.fliplr(grid))
        symmetry_score = (int(sym_h) + int(sym_v)) / 2
        
        # Object count
        obj_count = self._count_objects(grid)
        
        # Unique colors
        unique_colors = len(set(grid.flatten()))
        
        return PatternSignature(
            shape_ratio=shape_ratio,
            density=density,
            color_histogram=tuple(histogram),
            edge_density=edge_density,
            symmetry_score=symmetry_score,
            object_count=obj_count,
            unique_colors=unique_colors
        )
    
    def _count_objects(self, grid: np.ndarray) -> int:
        """Count connected components."""
        visited = np.zeros_like(grid, dtype=bool)
        rows, cols = grid.shape
        count = 0
        
        def bfs(start_r, start_c):
            queue = [(start_r, start_c)]
            visited[start_r, start_c] = True
            color = grid[start_r, start_c]
            while queue:
                r, c = queue.pop(0)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if not visited[nr, nc] and grid[nr, nc] == color:
                            visited[nr, nc] = True
                            queue.append((nr, nc))
        
        for r in range(rows):
            for c in range(cols):
                if not visited[r, c] and grid[r, c] != 0:
                    count += 1
                    bfs(r, c)
        
        return count
    
    def compute_transform_signature(
        self, 
        input_grid: np.ndarray, 
        output_grid: np.ndarray
    ) -> TransformSignature:
        """Compute signature of the transformation between input and output."""
        in_rows, in_cols = input_grid.shape
        out_rows, out_cols = output_grid.shape
        
        shape_change = (out_rows - in_rows, out_cols - in_cols)
        
        in_density = np.count_nonzero(input_grid) / (in_rows * in_cols)
        out_density = np.count_nonzero(output_grid) / (out_rows * out_cols)
        density_change = out_density - in_density
        
        # Color mapping (for same-shape grids)
        color_mapping = {}
        if input_grid.shape == output_grid.shape:
            for r in range(in_rows):
                for c in range(in_cols):
                    in_c = input_grid[r, c]
                    out_c = output_grid[r, c]
                    if in_c not in color_mapping:
                        color_mapping[int(in_c)] = int(out_c)
        
        # Structure preserved
        structure_preserved = input_grid.shape == output_grid.shape
        
        # Symmetry introduced
        out_sym_h = np.array_equal(output_grid, np.flipud(output_grid))
        out_sym_v = np.array_equal(output_grid, np.fliplr(output_grid))
        in_sym_h = np.array_equal(input_grid, np.flipud(input_grid))
        in_sym_v = np.array_equal(input_grid, np.fliplr(input_grid))
        symmetry_introduced = (out_sym_h or out_sym_v) and not (in_sym_h or in_sym_v)
        
        # Object count change
        in_objs = self._count_objects(input_grid)
        out_objs = self._count_objects(output_grid)
        
        return TransformSignature(
            shape_change=shape_change,
            density_change=density_change,
            color_mapping=color_mapping,
            structure_preserved=structure_preserved,
            symmetry_introduced=symmetry_introduced,
            object_count_change=out_objs - in_objs
        )
    
    def learn_from_example(
        self, 
        input_grid: np.ndarray, 
        output_grid: np.ndarray,
        transform_name: str,
        transform_fn: Any = None
    ):
        """Learn a pattern from an input/output example."""
        in_sig = self.compute_signature(input_grid)
        out_sig = self.compute_signature(output_grid)
        trans_sig = self.compute_transform_signature(input_grid, output_grid)
        
        # Create a combined hash for lookup
        sig_hash = self._hash_signatures(in_sig, out_sig)
        
        self.pattern_db[sig_hash] = (transform_name, transform_fn)
        self.transform_templates.append((trans_sig, transform_name))
    
    def _hash_signatures(self, in_sig: PatternSignature, out_sig: PatternSignature) -> str:
        """Create a hash from input/output signatures."""
        combined = {
            "in_ratio": round(in_sig.shape_ratio, 2),
            "out_ratio": round(out_sig.shape_ratio, 2),
            "in_density": round(in_sig.density, 2),
            "out_density": round(out_sig.density, 2),
            "in_sym": in_sig.symmetry_score,
            "out_sym": out_sig.symmetry_score
        }
        return hashlib.md5(json.dumps(combined, sort_keys=True).encode()).hexdigest()[:16]
    
    def find_similar_transform(
        self, 
        input_grid: np.ndarray,
        target_properties: Optional[Dict] = None
    ) -> Optional[Tuple[str, Any]]:
        """
        Find a transform that might work based on input properties
        and optional target properties.
        """
        in_sig = self.compute_signature(input_grid)
        
        best_match = None
        best_similarity = 0.0
        
        for sig_hash, (name, fn) in self.pattern_db.items():
            # This is a simplified matching - in production would use
            # vector similarity on learned embeddings
            similarity = 0.5  # Placeholder
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = (name, fn)
        
        return best_match if best_similarity > 0.3 else None
    
    def suggest_transforms(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[str]:
        """
        Analyze examples and suggest likely transform types.
        Returns list of suggested transform names.
        """
        suggestions = []
        
        if not examples:
            return suggestions
        
        # Analyze first example
        input_grid, output_grid = examples[0]
        trans_sig = self.compute_transform_signature(input_grid, output_grid)
        
        # Shape-based suggestions
        if trans_sig.shape_change == (0, 0):
            if trans_sig.color_mapping:
                non_trivial = {k: v for k, v in trans_sig.color_mapping.items() if k != v}
                if non_trivial:
                    suggestions.append("color_replacement")
            suggestions.append("geometric_transform")
        
        elif trans_sig.shape_change[0] > 0 or trans_sig.shape_change[1] > 0:
            # Output larger
            in_h, in_w = input_grid.shape
            out_h, out_w = output_grid.shape
            
            if out_h == 2 * in_h and out_w == 2 * in_w:
                suggestions.append("scale_2x")
            elif out_h == in_h and out_w == 2 * in_w:
                suggestions.append("horizontal_concat")
            elif out_h == 2 * in_h and out_w == in_w:
                suggestions.append("vertical_concat")
            suggestions.append("tiling")
            
        else:
            # Output smaller
            suggestions.append("crop")
            suggestions.append("extract_object")
        
        # Symmetry-based
        if trans_sig.symmetry_introduced:
            suggestions.append("make_symmetric")
        
        # Object-based
        if trans_sig.object_count_change < 0:
            suggestions.append("keep_object")
            suggestions.append("filter_objects")
        elif trans_sig.object_count_change > 0:
            suggestions.append("split_objects")
        
        return suggestions


class AbstractionEngine:
    """
    Higher-level abstraction for ARC patterns.
    
    Goes beyond pixel-level to understand:
    - Objects and their relationships
    - Spatial patterns
    - Transformational rules
    """
    
    def __init__(self):
        self.learned_abstractions: Dict[str, Any] = {}
    
    def analyze_grid_structure(self, grid: np.ndarray) -> Dict[str, Any]:
        """Analyze the structural properties of a grid."""
        rows, cols = grid.shape
        
        # Check for grid subdivisions
        subdivisions = self._detect_subdivisions(grid)
        
        # Check for repeating patterns
        repeating = self._detect_repetition(grid)
        
        # Check for borders/frames
        has_border = self._detect_border(grid)
        
        # Check for diagonal patterns
        diagonal_pattern = self._detect_diagonal(grid)
        
        return {
            "subdivisions": subdivisions,
            "repeating_pattern": repeating,
            "has_border": has_border,
            "diagonal_pattern": diagonal_pattern,
            "shape": (rows, cols),
            "is_square": rows == cols
        }
    
    def _detect_subdivisions(self, grid: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect if grid can be subdivided into equal parts."""
        rows, cols = grid.shape
        
        for div_r in range(2, min(rows + 1, 6)):
            for div_c in range(2, min(cols + 1, 6)):
                if rows % div_r == 0 and cols % div_c == 0:
                    cell_r = rows // div_r
                    cell_c = cols // div_c
                    
                    # Check if all cells are identical or follow a pattern
                    cells = []
                    for i in range(div_r):
                        for j in range(div_c):
                            cell = grid[i*cell_r:(i+1)*cell_r, j*cell_c:(j+1)*cell_c]
                            cells.append(cell)
                    
                    # Check for repetition
                    if all(np.array_equal(cells[0], c) for c in cells):
                        return (div_r, div_c)
        
        return None
    
    def _detect_repetition(self, grid: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detect repeating patterns."""
        rows, cols = grid.shape
        
        # Check horizontal repetition
        for period in range(1, cols // 2 + 1):
            if cols % period == 0:
                pattern = grid[:, :period]
                if all(np.array_equal(pattern, grid[:, i:i+period]) 
                       for i in range(0, cols, period)):
                    return {"direction": "horizontal", "period": period}
        
        # Check vertical repetition
        for period in range(1, rows // 2 + 1):
            if rows % period == 0:
                pattern = grid[:period, :]
                if all(np.array_equal(pattern, grid[i:i+period, :]) 
                       for i in range(0, rows, period)):
                    return {"direction": "vertical", "period": period}
        
        return None
    
    def _detect_border(self, grid: np.ndarray) -> bool:
        """Detect if grid has a consistent border."""
        rows, cols = grid.shape
        if rows < 3 or cols < 3:
            return False
        
        # Check if border pixels are all the same
        border_pixels = []
        border_pixels.extend(grid[0, :].tolist())
        border_pixels.extend(grid[-1, :].tolist())
        border_pixels.extend(grid[1:-1, 0].tolist())
        border_pixels.extend(grid[1:-1, -1].tolist())
        
        return len(set(border_pixels)) == 1
    
    def _detect_diagonal(self, grid: np.ndarray) -> Optional[str]:
        """Detect diagonal patterns."""
        rows, cols = grid.shape
        if rows != cols:
            return None
        
        # Check main diagonal
        main_diag = [grid[i, i] for i in range(min(rows, cols))]
        if len(set(main_diag)) == 1 and main_diag[0] != 0:
            return "main_diagonal"
        
        # Check anti-diagonal
        anti_diag = [grid[i, cols-1-i] for i in range(min(rows, cols))]
        if len(set(anti_diag)) == 1 and anti_diag[0] != 0:
            return "anti_diagonal"
        
        return None
    
    def infer_rule(
        self, 
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[str]:
        """
        Try to infer a high-level rule from examples.
        Returns a description of the inferred rule.
        """
        if not examples:
            return None
        
        # Analyze each example
        input_structures = [self.analyze_grid_structure(inp) for inp, _ in examples]
        output_structures = [self.analyze_grid_structure(out) for _, out in examples]
        
        # Look for consistent patterns
        rules = []
        
        # Check for consistent shape changes
        shape_changes = [(out["shape"][0] - inp["shape"][0], 
                         out["shape"][1] - inp["shape"][1])
                        for inp, out in zip(input_structures, output_structures)]
        
        if all(sc == shape_changes[0] for sc in shape_changes):
            if shape_changes[0] == (0, 0):
                rules.append("preserves_shape")
            else:
                rules.append(f"changes_shape_by_{shape_changes[0]}")
        
        # Check for border changes
        if all(inp["has_border"] for inp in input_structures):
            rules.append("input_has_border")
        if all(out["has_border"] for out in output_structures):
            rules.append("output_has_border")
        
        # Check for subdivision patterns
        subdivs = [inp["subdivisions"] for inp in input_structures]
        if all(s is not None for s in subdivs) and all(s == subdivs[0] for s in subdivs):
            rules.append(f"subdivided_{subdivs[0]}")
        
        return " | ".join(rules) if rules else None
