"""
Advanced Pattern Detectors for ARC-AGI

This module contains sophisticated pattern detectors for complex ARC tasks:
- Template matching and sprite detection
- Recursive patterns and fractals
- Counting and arithmetic patterns  
- Conditional rules
- Grid subdivision patterns
- Path and line completion
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable, Any, Set
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict, Counter
import itertools

from .object_detector import ObjectDetector, ArcObject, GridAnalysis, ShapeType


class PatternCategory(Enum):
    """Categories of advanced patterns."""
    TEMPLATE = "template"
    COUNTING = "counting"
    CONDITIONAL = "conditional"
    SUBDIVISION = "subdivision"
    COMPLETION = "completion"
    MASKING = "masking"
    OVERLAY = "overlay"
    RECURSIVE = "recursive"
    SORTING = "sorting"
    SELECTION = "selection"


@dataclass
class DetectedPattern:
    """A detected pattern with associated transformation."""
    category: PatternCategory
    description: str
    confidence: float
    transform_fn: Optional[Callable] = None
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class AdvancedPatternDetector:
    """
    Detects complex patterns in ARC input→output pairs.
    
    Goes beyond simple geometric transforms to find:
    - Template/sprite based rules
    - Counting patterns
    - Conditional logic
    - And more...
    """
    
    def __init__(self):
        self.object_detector = ObjectDetector()
        self.sprite_library: Dict[str, np.ndarray] = {}
        
    def detect_all_patterns(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[DetectedPattern]:
        """
        Try all pattern detectors on the examples.
        Returns list of detected patterns ranked by confidence.
        """
        patterns = []
        
        # Try each detector
        detectors = [
            self._detect_template_replacement,
            self._detect_counting_pattern,
            self._detect_conditional_coloring,
            self._detect_grid_subdivision,
            self._detect_line_completion,
            self._detect_overlay_pattern,
            self._detect_object_selection,
            self._detect_sorting_pattern,
            self._detect_masking_pattern,
            self._detect_fill_pattern,
            self._detect_boundary_pattern,
            self._detect_replication_pattern,
            self._detect_extraction_pattern,
            self._detect_connection_pattern,
        ]
        
        for detector in detectors:
            try:
                result = detector(examples)
                if result and result.confidence > 0.5:
                    patterns.append(result)
            except Exception as e:
                continue
        
        # Sort by confidence
        patterns.sort(key=lambda x: x.confidence, reverse=True)
        return patterns
    
    def _detect_template_replacement(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[DetectedPattern]:
        """
        Detect if output replaces certain colors/patterns with templates.
        E.g., replace each blue pixel with a 3x3 cross.
        """
        if not examples:
            return None
        
        for example in examples:
            input_grid, output_grid = example
            in_h, in_w = input_grid.shape
            out_h, out_w = output_grid.shape
            
            # Check for scaling pattern
            if out_h % in_h == 0 and out_w % in_w == 0:
                scale_h = out_h // in_h
                scale_w = out_w // in_w
                
                if scale_h == scale_w and scale_h > 1:
                    # Check if each input pixel maps to a template in output
                    templates = {}
                    consistent = True
                    
                    for r in range(in_h):
                        for c in range(in_w):
                            color = input_grid[r, c]
                            out_block = output_grid[
                                r*scale_h:(r+1)*scale_h,
                                c*scale_w:(c+1)*scale_w
                            ]
                            
                            key = tuple(out_block.flatten())
                            if color not in templates:
                                templates[color] = out_block.copy()
                            else:
                                if not np.array_equal(templates[color], out_block):
                                    consistent = False
                                    break
                        if not consistent:
                            break
                    
                    if consistent and len(templates) > 0:
                        def apply_template(grid, templates=templates, scale=scale_h):
                            h, w = grid.shape
                            result = np.zeros((h * scale, w * scale), dtype=grid.dtype)
                            for r in range(h):
                                for c in range(w):
                                    color = grid[r, c]
                                    if color in templates:
                                        result[r*scale:(r+1)*scale, c*scale:(c+1)*scale] = templates[color]
                            return result
                        
                        # Verify on all examples
                        if self._verify_transform(examples, apply_template):
                            return DetectedPattern(
                                category=PatternCategory.TEMPLATE,
                                description=f"Template replacement with {scale_h}x{scale_w} blocks",
                                confidence=0.95,
                                transform_fn=apply_template,
                                parameters={'scale': scale_h, 'templates': templates}
                            )
        
        return None
    
    def _detect_counting_pattern(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[DetectedPattern]:
        """
        Detect if output is based on counting objects/colors in input.
        E.g., output is N×1 grid where N = number of blue pixels.
        """
        for example in examples:
            input_grid, output_grid = example
            
            # Check if output is small (likely a count)
            if output_grid.size <= 10:  # Small output
                # Try counting various things
                
                # Count non-zero pixels
                count = np.count_nonzero(input_grid)
                if self._output_represents_count(output_grid, count):
                    def count_nonzero(grid):
                        c = np.count_nonzero(grid)
                        return np.array([[c]])
                    
                    if self._verify_transform(examples, count_nonzero):
                        return DetectedPattern(
                            category=PatternCategory.COUNTING,
                            description="Count non-zero pixels",
                            confidence=0.9,
                            transform_fn=count_nonzero
                        )
                
                # Count objects
                objects = self.object_detector.detect_objects(input_grid)
                if self._output_represents_count(output_grid, len(objects)):
                    def count_objects(grid, detector=self.object_detector):
                        objs = detector.detect_objects(grid)
                        return np.array([[len(objs)]])
                    
                    if self._verify_transform(examples, count_objects):
                        return DetectedPattern(
                            category=PatternCategory.COUNTING,
                            description="Count objects",
                            confidence=0.9,
                            transform_fn=count_objects
                        )
                
                # Count by color
                for color in range(1, 10):
                    color_count = np.sum(input_grid == color)
                    if color_count > 0 and self._output_represents_count(output_grid, color_count):
                        def count_color(grid, c=color):
                            return np.array([[np.sum(grid == c)]])
                        
                        if self._verify_transform(examples, count_color):
                            return DetectedPattern(
                                category=PatternCategory.COUNTING,
                                description=f"Count color {color}",
                                confidence=0.85,
                                transform_fn=count_color
                            )
        
        return None
    
    def _output_represents_count(self, output: np.ndarray, count: int) -> bool:
        """Check if output represents a count value."""
        # Output is [[count]]
        if output.shape == (1, 1) and output[0, 0] == count:
            return True
        
        # Output is row of count pixels
        if output.shape[0] == 1 and output.shape[1] == count:
            return True
        
        # Output is column of count pixels
        if output.shape[1] == 1 and output.shape[0] == count:
            return True
        
        return False
    
    def _detect_conditional_coloring(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[DetectedPattern]:
        """
        Detect conditional coloring rules.
        E.g., if pixel neighbors contain blue, make it red.
        """
        for example in examples:
            input_grid, output_grid = example
            
            if input_grid.shape != output_grid.shape:
                continue
            
            h, w = input_grid.shape
            
            # Analyze what changed
            changed = input_grid != output_grid
            if not np.any(changed):
                continue
            
            # Find pattern in changes
            changed_positions = list(zip(*np.where(changed)))
            
            # Check if change is based on neighbors
            for r, c in changed_positions:
                old_color = input_grid[r, c]
                new_color = output_grid[r, c]
                
                # Get neighbors
                neighbors = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        neighbors.append(input_grid[nr, nc])
                
                # Check patterns
                # If has neighbor of specific color, change color
                neighbor_colors = set(neighbors)
                
        return None
    
    def _detect_grid_subdivision(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[DetectedPattern]:
        """
        Detect grid subdivision patterns.
        E.g., output is one cell from a divided input grid.
        """
        for example in examples:
            input_grid, output_grid = example
            in_h, in_w = input_grid.shape
            out_h, out_w = output_grid.shape
            
            # Check if output is a subdivision of input
            if in_h % out_h == 0 and in_w % out_w == 0:
                scale_h = in_h // out_h
                scale_w = in_w // out_w
                
                for start_r in range(scale_h):
                    for start_c in range(scale_w):
                        cell = input_grid[start_r::scale_h, start_c::scale_w]
                        if cell.shape == output_grid.shape and np.array_equal(cell, output_grid):
                            def extract_cell(grid, sr=start_r, sc=start_c, sh=scale_h, sw=scale_w):
                                return grid[sr::sh, sc::sw]
                            
                            if self._verify_transform(examples, extract_cell):
                                return DetectedPattern(
                                    category=PatternCategory.SUBDIVISION,
                                    description=f"Extract cell at ({start_r}, {start_c}) with stride ({scale_h}, {scale_w})",
                                    confidence=0.9,
                                    transform_fn=extract_cell
                                )
        
        return None
    
    def _detect_line_completion(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[DetectedPattern]:
        """
        Detect line completion patterns.
        E.g., extend lines to grid boundary.
        """
        for example in examples:
            input_grid, output_grid = example
            
            if input_grid.shape != output_grid.shape:
                continue
            
            h, w = input_grid.shape
            
            # Check for horizontal line completion
            def extend_horizontal(grid):
                result = grid.copy()
                h, w = grid.shape
                for r in range(h):
                    nonzero = np.where(grid[r] != 0)[0]
                    if len(nonzero) > 0:
                        color = grid[r, nonzero[0]]
                        result[r, :] = color
                return result
            
            if np.array_equal(extend_horizontal(input_grid), output_grid):
                if self._verify_transform(examples, extend_horizontal):
                    return DetectedPattern(
                        category=PatternCategory.COMPLETION,
                        description="Extend colored pixels horizontally",
                        confidence=0.85,
                        transform_fn=extend_horizontal
                    )
            
            # Check for vertical line completion
            def extend_vertical(grid):
                result = grid.copy()
                h, w = grid.shape
                for c in range(w):
                    nonzero = np.where(grid[:, c] != 0)[0]
                    if len(nonzero) > 0:
                        color = grid[nonzero[0], c]
                        result[:, c] = color
                return result
            
            if np.array_equal(extend_vertical(input_grid), output_grid):
                if self._verify_transform(examples, extend_vertical):
                    return DetectedPattern(
                        category=PatternCategory.COMPLETION,
                        description="Extend colored pixels vertically",
                        confidence=0.85,
                        transform_fn=extend_vertical
                    )
        
        return None
    
    def _detect_overlay_pattern(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[DetectedPattern]:
        """
        Detect overlay/composition patterns.
        E.g., overlay two objects with specific rule.
        """
        for example in examples:
            input_grid, output_grid = example
            
            if input_grid.shape != output_grid.shape:
                continue
            
            # Detect multiple objects
            objects = self.object_detector.detect_objects(input_grid)
            
            if len(objects) >= 2:
                # Try AND overlay (intersection)
                def overlay_and(grid, detector=self.object_detector):
                    objs = detector.detect_objects(grid)
                    if len(objs) < 2:
                        return grid.copy()
                    
                    result = np.zeros_like(grid)
                    all_positions = set()
                    for obj in objs:
                        if not all_positions:
                            all_positions = obj.pixels.copy()
                        else:
                            all_positions &= obj.pixels
                    
                    for r, c in all_positions:
                        result[r, c] = objs[0].color
                    
                    return result
                
                # Try OR overlay (union)
                def overlay_or(grid, detector=self.object_detector):
                    objs = detector.detect_objects(grid)
                    if len(objs) < 2:
                        return grid.copy()
                    
                    result = np.zeros_like(grid)
                    for obj in objs:
                        for r, c in obj.pixels:
                            result[r, c] = obj.color
                    
                    return result
                
                # Try XOR overlay
                def overlay_xor(grid, detector=self.object_detector):
                    objs = detector.detect_objects(grid)
                    if len(objs) < 2:
                        return grid.copy()
                    
                    result = np.zeros_like(grid)
                    position_counts = defaultdict(int)
                    position_colors = {}
                    
                    for obj in objs:
                        for r, c in obj.pixels:
                            position_counts[(r, c)] += 1
                            position_colors[(r, c)] = obj.color
                    
                    for pos, count in position_counts.items():
                        if count == 1:
                            result[pos[0], pos[1]] = position_colors[pos]
                    
                    return result
                
                for fn, desc in [
                    (overlay_and, "Objects AND (intersection)"),
                    (overlay_or, "Objects OR (union)"),
                    (overlay_xor, "Objects XOR (exclusive)"),
                ]:
                    try:
                        if np.array_equal(fn(input_grid), output_grid):
                            if self._verify_transform(examples, fn):
                                return DetectedPattern(
                                    category=PatternCategory.OVERLAY,
                                    description=desc,
                                    confidence=0.85,
                                    transform_fn=fn
                                )
                    except:
                        pass
        
        return None
    
    def _detect_object_selection(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[DetectedPattern]:
        """
        Detect object selection patterns.
        E.g., keep only the largest/smallest/most-colorful object.
        """
        for example in examples:
            input_grid, output_grid = example
            
            input_objects = self.object_detector.detect_objects(input_grid)
            output_objects = self.object_detector.detect_objects(output_grid)
            
            if len(input_objects) > 1 and len(output_objects) == 1:
                # Find which object was kept
                out_pattern = output_objects[0].get_relative_pattern()
                
                # Check if largest
                largest = max(input_objects, key=lambda o: o.size)
                if largest.size == output_objects[0].size:
                    def keep_largest(grid, detector=self.object_detector):
                        objs = detector.detect_objects(grid)
                        if not objs:
                            return grid.copy()
                        largest = max(objs, key=lambda o: o.size)
                        result = np.zeros_like(grid)
                        for r, c in largest.pixels:
                            result[r, c] = grid[r, c]
                        return result
                    
                    if self._verify_transform(examples, keep_largest):
                        return DetectedPattern(
                            category=PatternCategory.SELECTION,
                            description="Keep largest object",
                            confidence=0.9,
                            transform_fn=keep_largest
                        )
                
                # Check if smallest
                smallest = min(input_objects, key=lambda o: o.size)
                if smallest.size == output_objects[0].size:
                    def keep_smallest(grid, detector=self.object_detector):
                        objs = detector.detect_objects(grid)
                        if not objs:
                            return grid.copy()
                        smallest = min(objs, key=lambda o: o.size)
                        result = np.zeros_like(grid)
                        for r, c in smallest.pixels:
                            result[r, c] = grid[r, c]
                        return result
                    
                    if self._verify_transform(examples, keep_smallest):
                        return DetectedPattern(
                            category=PatternCategory.SELECTION,
                            description="Keep smallest object",
                            confidence=0.9,
                            transform_fn=keep_smallest
                        )
        
        return None
    
    def _detect_sorting_pattern(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[DetectedPattern]:
        """
        Detect sorting/ordering patterns.
        E.g., sort objects by size or color.
        """
        # Implementation for sorting patterns
        return None
    
    def _detect_masking_pattern(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[DetectedPattern]:
        """
        Detect masking patterns.
        E.g., use one object as mask for another.
        """
        for example in examples:
            input_grid, output_grid = example
            
            if input_grid.shape != output_grid.shape:
                continue
            
            # Check if output is masked input
            # Background where input was background, else some value
            input_objects = self.object_detector.detect_objects(input_grid)
            
            if len(input_objects) >= 2:
                # Try using first object as mask for second
                for mask_obj in input_objects:
                    for target_obj in input_objects:
                        if mask_obj.id == target_obj.id:
                            continue
                        
                        # Apply mask
                        result = np.zeros_like(input_grid)
                        for r, c in mask_obj.pixels & target_obj.pixels:
                            result[r, c] = target_obj.color
                        
                        if np.array_equal(result, output_grid):
                            return DetectedPattern(
                                category=PatternCategory.MASKING,
                                description="Apply object as mask",
                                confidence=0.85
                            )
        
        return None
    
    def _detect_fill_pattern(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[DetectedPattern]:
        """
        Detect fill patterns like flood fill.
        """
        for example in examples:
            input_grid, output_grid = example
            
            if input_grid.shape != output_grid.shape:
                continue
            
            # Check for fill holes pattern
            def fill_holes(grid, detector=self.object_detector):
                result = grid.copy()
                h, w = grid.shape
                
                # Find enclosed zeros
                visited = set()
                
                # Mark all zeros reachable from edges
                def bfs_from_edge():
                    edge_zeros = set()
                    for r in range(h):
                        if grid[r, 0] == 0:
                            edge_zeros.add((r, 0))
                        if grid[r, w-1] == 0:
                            edge_zeros.add((r, w-1))
                    for c in range(w):
                        if grid[0, c] == 0:
                            edge_zeros.add((0, c))
                        if grid[h-1, c] == 0:
                            edge_zeros.add((h-1, c))
                    
                    reachable = set()
                    queue = list(edge_zeros)
                    while queue:
                        r, c = queue.pop(0)
                        if (r, c) in reachable:
                            continue
                        if r < 0 or r >= h or c < 0 or c >= w:
                            continue
                        if grid[r, c] != 0:
                            continue
                        reachable.add((r, c))
                        queue.extend([(r-1, c), (r+1, c), (r, c-1), (r, c+1)])
                    
                    return reachable
                
                edge_reachable = bfs_from_edge()
                
                # Fill non-reachable zeros
                for r in range(h):
                    for c in range(w):
                        if grid[r, c] == 0 and (r, c) not in edge_reachable:
                            # Find surrounding color
                            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != 0:
                                    result[r, c] = grid[nr, nc]
                                    break
                
                return result
            
            if np.array_equal(fill_holes(input_grid), output_grid):
                if self._verify_transform(examples, fill_holes):
                    return DetectedPattern(
                        category=PatternCategory.COMPLETION,
                        description="Fill enclosed holes",
                        confidence=0.85,
                        transform_fn=fill_holes
                    )
        
        return None
    
    def _detect_boundary_pattern(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[DetectedPattern]:
        """
        Detect boundary/outline patterns.
        """
        for example in examples:
            input_grid, output_grid = example
            
            if input_grid.shape != output_grid.shape:
                continue
            
            # Check for outline pattern
            def make_outline(grid):
                result = grid.copy()
                h, w = grid.shape
                
                for r in range(h):
                    for c in range(w):
                        if grid[r, c] != 0:
                            # Check if interior (all 4 neighbors same color)
                            is_interior = True
                            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < h and 0 <= nc < w:
                                    if grid[nr, nc] != grid[r, c]:
                                        is_interior = False
                                        break
                                else:
                                    is_interior = False
                                    break
                            
                            if is_interior:
                                result[r, c] = 0
                
                return result
            
            if np.array_equal(make_outline(input_grid), output_grid):
                if self._verify_transform(examples, make_outline):
                    return DetectedPattern(
                        category=PatternCategory.MASKING,
                        description="Extract outline",
                        confidence=0.85,
                        transform_fn=make_outline
                    )
        
        return None
    
    def _detect_replication_pattern(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[DetectedPattern]:
        """
        Detect replication/tiling patterns.
        """
        for example in examples:
            input_grid, output_grid = example
            in_h, in_w = input_grid.shape
            out_h, out_w = output_grid.shape
            
            # Check for tiling
            if out_h % in_h == 0 and out_w % in_w == 0:
                rep_h = out_h // in_h
                rep_w = out_w // in_w
                
                if rep_h > 1 or rep_w > 1:
                    # Create tiled version
                    tiled = np.tile(input_grid, (rep_h, rep_w))
                    
                    if np.array_equal(tiled, output_grid):
                        def tile_grid(grid, rh=rep_h, rw=rep_w):
                            return np.tile(grid, (rh, rw))
                        
                        if self._verify_transform(examples, tile_grid):
                            return DetectedPattern(
                                category=PatternCategory.TEMPLATE,
                                description=f"Tile {rep_h}x{rep_w}",
                                confidence=0.9,
                                transform_fn=tile_grid
                            )
        
        return None
    
    def _detect_extraction_pattern(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[DetectedPattern]:
        """
        Detect pattern extraction from larger grid.
        """
        for example in examples:
            input_grid, output_grid = example
            in_h, in_w = input_grid.shape
            out_h, out_w = output_grid.shape
            
            if out_h < in_h or out_w < in_w:
                # Output is smaller - check for extraction
                
                # Check for crop to content
                nonzero = np.argwhere(input_grid != 0)
                if len(nonzero) > 0:
                    min_r, min_c = nonzero.min(axis=0)
                    max_r, max_c = nonzero.max(axis=0)
                    cropped = input_grid[min_r:max_r+1, min_c:max_c+1]
                    
                    if np.array_equal(cropped, output_grid):
                        def crop_to_content(grid):
                            nonzero = np.argwhere(grid != 0)
                            if len(nonzero) == 0:
                                return np.array([[0]])
                            min_r, min_c = nonzero.min(axis=0)
                            max_r, max_c = nonzero.max(axis=0)
                            return grid[min_r:max_r+1, min_c:max_c+1]
                        
                        if self._verify_transform(examples, crop_to_content):
                            return DetectedPattern(
                                category=PatternCategory.SELECTION,
                                description="Crop to content",
                                confidence=0.9,
                                transform_fn=crop_to_content
                            )
        
        return None
    
    def _detect_connection_pattern(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[DetectedPattern]:
        """
        Detect patterns that connect objects with lines.
        """
        for example in examples:
            input_grid, output_grid = example
            
            if input_grid.shape != output_grid.shape:
                continue
            
            input_objects = self.object_detector.detect_objects(input_grid)
            
            if len(input_objects) >= 2:
                # Check if same-color objects are connected with lines
                def connect_same_color(grid, detector=self.object_detector):
                    result = grid.copy()
                    h, w = grid.shape
                    objs = detector.detect_objects(grid)
                    
                    # Group by color
                    color_groups = defaultdict(list)
                    for obj in objs:
                        color_groups[obj.color].append(obj)
                    
                    # Connect objects of same color
                    for color, group in color_groups.items():
                        if len(group) >= 2:
                            # Sort by position for consistent connection
                            group.sort(key=lambda o: (o.bbox.center[0], o.bbox.center[1]))
                            
                            for i in range(len(group) - 1):
                                obj1, obj2 = group[i], group[i+1]
                                c1, c2 = obj1.bbox.center, obj2.bbox.center
                                
                                # Draw line between centers
                                r1, c1 = int(c1[0]), int(c1[1])
                                r2, c2 = int(c2[0]), int(c2[1])
                                
                                # Simple line drawing
                                if r1 == r2:
                                    for c in range(min(c1, c2), max(c1, c2) + 1):
                                        result[r1, c] = color
                                elif c1 == c2:
                                    for r in range(min(r1, r2), max(r1, r2) + 1):
                                        result[r, c1] = color
                    
                    return result
                
                if np.array_equal(connect_same_color(input_grid), output_grid):
                    if self._verify_transform(examples, connect_same_color):
                        return DetectedPattern(
                            category=PatternCategory.COMPLETION,
                            description="Connect same-color objects with lines",
                            confidence=0.85,
                            transform_fn=connect_same_color
                        )
        
        return None
    
    def _verify_transform(
        self, 
        examples: List[Tuple[np.ndarray, np.ndarray]],
        transform_fn: Callable
    ) -> bool:
        """Verify a transform works on all examples."""
        try:
            for input_grid, output_grid in examples:
                result = transform_fn(input_grid)
                if not np.array_equal(result, output_grid):
                    return False
            return True
        except Exception:
            return False


# Convenience function
def detect_patterns(
    examples: List[Tuple[np.ndarray, np.ndarray]]
) -> List[DetectedPattern]:
    """Detect patterns in ARC examples."""
    detector = AdvancedPatternDetector()
    return detector.detect_all_patterns(examples)
