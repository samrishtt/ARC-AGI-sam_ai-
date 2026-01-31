"""
Object-Centric Perception for ARC-AGI

This module provides sophisticated object detection and analysis,
treating grids as collections of objects rather than raw matrices.

Key Features:
- Connected component detection with 4/8 connectivity
- Object property extraction (size, color, shape, position)
- Spatial relationship analysis (above, below, inside, touching)
- Object grouping and clustering
- Shape classification (rectangle, line, L-shape, etc.)
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import itertools


class ShapeType(Enum):
    """Classification of object shapes."""
    POINT = "point"
    LINE_H = "line_horizontal"
    LINE_V = "line_vertical"
    LINE_DIAG = "line_diagonal"
    RECTANGLE = "rectangle"
    SQUARE = "square"
    L_SHAPE = "l_shape"
    T_SHAPE = "t_shape"
    PLUS = "plus"
    HOLLOW_RECT = "hollow_rectangle"
    IRREGULAR = "irregular"
    SPRITE = "sprite"


class SpatialRelation(Enum):
    """Spatial relationships between objects."""
    ABOVE = "above"
    BELOW = "below"
    LEFT = "left"
    RIGHT = "right"
    INSIDE = "inside"
    CONTAINS = "contains"
    TOUCHING = "touching"
    OVERLAPPING = "overlapping"
    ALIGNED_H = "aligned_horizontal"
    ALIGNED_V = "aligned_vertical"
    SAME_ROW = "same_row"
    SAME_COL = "same_column"


@dataclass
class BoundingBox:
    """Bounding box of an object."""
    min_row: int
    max_row: int
    min_col: int
    max_col: int
    
    @property
    def height(self) -> int:
        return self.max_row - self.min_row + 1
    
    @property
    def width(self) -> int:
        return self.max_col - self.min_col + 1
    
    @property
    def area(self) -> int:
        return self.height * self.width
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.min_row + self.max_row) / 2, 
                (self.min_col + self.max_col) / 2)
    
    @property
    def is_square(self) -> bool:
        return self.height == self.width


@dataclass
class ArcObject:
    """
    Represents a single object detected in an ARC grid.
    
    An object is a connected component of non-background pixels.
    """
    id: int
    pixels: Set[Tuple[int, int]]  # Set of (row, col) coordinates
    color: int  # Primary color (most frequent)
    colors: Set[int]  # All colors in object
    bbox: BoundingBox
    
    # Derived properties
    size: int = 0
    shape_type: ShapeType = ShapeType.IRREGULAR
    is_solid: bool = True
    has_hole: bool = False
    symmetry_h: bool = False
    symmetry_v: bool = False
    
    # The actual pixel pattern (cropped)
    pattern: Optional[np.ndarray] = None
    
    def __post_init__(self):
        self.size = len(self.pixels)
    
    def __hash__(self):
        return hash(self.id)
    
    def get_relative_pattern(self) -> np.ndarray:
        """Get the object pattern in its local coordinate system."""
        if self.pattern is not None:
            return self.pattern
        
        pattern = np.zeros((self.bbox.height, self.bbox.width), dtype=np.int32)
        for r, c in self.pixels:
            pattern[r - self.bbox.min_row, c - self.bbox.min_col] = self.color
        self.pattern = pattern
        return pattern


@dataclass
class ObjectRelation:
    """Relationship between two objects."""
    obj1_id: int
    obj2_id: int
    relation: SpatialRelation
    distance: float = 0.0


@dataclass 
class GridAnalysis:
    """Complete analysis of a grid's objects and structure."""
    objects: List[ArcObject]
    relations: List[ObjectRelation]
    background_color: int
    grid_shape: Tuple[int, int]
    num_objects: int
    unique_colors: Set[int]
    
    # Grid-level patterns
    has_grid_structure: bool = False
    grid_cell_size: Optional[Tuple[int, int]] = None
    is_symmetric_h: bool = False
    is_symmetric_v: bool = False
    
    # Object groupings
    color_groups: Dict[int, List[int]] = field(default_factory=dict)
    size_groups: Dict[int, List[int]] = field(default_factory=dict)
    shape_groups: Dict[ShapeType, List[int]] = field(default_factory=dict)


class ObjectDetector:
    """
    Advanced object detection for ARC grids.
    
    Provides comprehensive analysis of grid contents including:
    - Object segmentation
    - Property extraction
    - Relationship analysis
    - Pattern matching
    """
    
    def __init__(self, connectivity: int = 4):
        """
        Initialize detector.
        
        Args:
            connectivity: 4 or 8 for neighbor connectivity
        """
        self.connectivity = connectivity
        self._object_counter = 0
        
    def detect_objects(
        self, 
        grid: np.ndarray, 
        background: int = 0
    ) -> List[ArcObject]:
        """
        Detect all objects in a grid.
        
        Args:
            grid: Input grid
            background: Background color (default 0)
            
        Returns:
            List of detected objects
        """
        objects = []
        visited = set()
        rows, cols = grid.shape
        
        for r in range(rows):
            for c in range(cols):
                if (r, c) not in visited and grid[r, c] != background:
                    # BFS to find connected component
                    pixels, colors = self._flood_fill(grid, r, c, background, visited)
                    
                    if pixels:
                        obj = self._create_object(pixels, colors)
                        objects.append(obj)
        
        # Analyze each object
        for obj in objects:
            self._analyze_object(obj, grid)
        
        return objects
    
    def _flood_fill(
        self, 
        grid: np.ndarray, 
        start_r: int, 
        start_c: int,
        background: int,
        visited: Set[Tuple[int, int]]
    ) -> Tuple[Set[Tuple[int, int]], Set[int]]:
        """Flood fill to find connected component."""
        rows, cols = grid.shape
        pixels = set()
        colors = set()
        queue = [(start_r, start_c)]
        
        while queue:
            r, c = queue.pop(0)
            
            if (r, c) in visited:
                continue
            if r < 0 or r >= rows or c < 0 or c >= cols:
                continue
            if grid[r, c] == background:
                continue
                
            visited.add((r, c))
            pixels.add((r, c))
            colors.add(grid[r, c])
            
            # Add neighbors based on connectivity
            if self.connectivity == 4:
                neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
            else:  # 8-connectivity
                neighbors = [
                    (r-1, c), (r+1, c), (r, c-1), (r, c+1),
                    (r-1, c-1), (r-1, c+1), (r+1, c-1), (r+1, c+1)
                ]
            
            for nr, nc in neighbors:
                if (nr, nc) not in visited:
                    queue.append((nr, nc))
        
        return pixels, colors
    
    def _create_object(
        self, 
        pixels: Set[Tuple[int, int]], 
        colors: Set[int]
    ) -> ArcObject:
        """Create an ArcObject from pixel set."""
        self._object_counter += 1
        
        rows = [p[0] for p in pixels]
        cols = [p[1] for p in pixels]
        
        bbox = BoundingBox(
            min_row=min(rows),
            max_row=max(rows),
            min_col=min(cols),
            max_col=max(cols)
        )
        
        # Find primary color (most frequent)
        color_counts = defaultdict(int)
        for c in colors:
            color_counts[c] += 1
        primary_color = max(color_counts.keys(), key=lambda x: color_counts[x])
        
        return ArcObject(
            id=self._object_counter,
            pixels=pixels,
            color=primary_color,
            colors=colors,
            bbox=bbox
        )
    
    def _analyze_object(self, obj: ArcObject, grid: np.ndarray) -> None:
        """Analyze object properties."""
        pattern = obj.get_relative_pattern()
        h, w = pattern.shape
        
        # Check if solid (filled rectangle)
        obj.is_solid = np.count_nonzero(pattern) == h * w
        
        # Classify shape
        obj.shape_type = self._classify_shape(pattern, obj)
        
        # Check for holes
        obj.has_hole = self._has_hole(pattern)
        
        # Check symmetry
        obj.symmetry_h = np.array_equal(pattern, np.fliplr(pattern))
        obj.symmetry_v = np.array_equal(pattern, np.flipud(pattern))
    
    def _classify_shape(self, pattern: np.ndarray, obj: ArcObject) -> ShapeType:
        """Classify the shape of an object."""
        h, w = pattern.shape
        size = obj.size
        
        # Point
        if size == 1:
            return ShapeType.POINT
        
        # Lines
        if h == 1 and w > 1:
            return ShapeType.LINE_H
        if w == 1 and h > 1:
            return ShapeType.LINE_V
        
        # Check diagonal line
        if self._is_diagonal_line(pattern):
            return ShapeType.LINE_DIAG
        
        # Rectangle/Square
        if obj.is_solid:
            if h == w:
                return ShapeType.SQUARE
            return ShapeType.RECTANGLE
        
        # Hollow rectangle
        if self._is_hollow_rectangle(pattern):
            return ShapeType.HOLLOW_RECT
        
        # Plus shape
        if self._is_plus_shape(pattern):
            return ShapeType.PLUS
        
        # L-shape
        if self._is_l_shape(pattern):
            return ShapeType.L_SHAPE
        
        # T-shape
        if self._is_t_shape(pattern):
            return ShapeType.T_SHAPE
        
        return ShapeType.IRREGULAR
    
    def _is_diagonal_line(self, pattern: np.ndarray) -> bool:
        """Check if pattern is a diagonal line."""
        h, w = pattern.shape
        if h != w:
            return False
        
        nonzero = np.count_nonzero(pattern)
        if nonzero != h:
            return False
        
        # Check main diagonal
        main_diag = all(pattern[i, i] != 0 for i in range(h))
        if main_diag:
            return True
        
        # Check anti-diagonal
        anti_diag = all(pattern[i, w-1-i] != 0 for i in range(h))
        return anti_diag
    
    def _is_hollow_rectangle(self, pattern: np.ndarray) -> bool:
        """Check if pattern is a hollow rectangle."""
        h, w = pattern.shape
        if h < 3 or w < 3:
            return False
        
        # Check border is filled
        border_filled = (
            np.all(pattern[0, :] != 0) and
            np.all(pattern[-1, :] != 0) and
            np.all(pattern[:, 0] != 0) and
            np.all(pattern[:, -1] != 0)
        )
        
        # Check interior is empty (for at least part of it)
        interior = pattern[1:-1, 1:-1]
        has_empty_interior = np.any(interior == 0)
        
        return border_filled and has_empty_interior
    
    def _is_plus_shape(self, pattern: np.ndarray) -> bool:
        """Check if pattern is a plus/cross shape."""
        h, w = pattern.shape
        if h < 3 or w < 3 or h != w:
            return False
        
        center_r, center_c = h // 2, w // 2
        
        # Check cross pattern
        nonzero_count = np.count_nonzero(pattern)
        expected = h + w - 1  # Cross has h + w - 1 pixels
        
        if nonzero_count != expected:
            return False
        
        # Check center row and column are filled
        center_row_filled = np.all(pattern[center_r, :] != 0)
        center_col_filled = np.all(pattern[:, center_c] != 0)
        
        return center_row_filled and center_col_filled
    
    def _is_l_shape(self, pattern: np.ndarray) -> bool:
        """Check if pattern is L-shaped."""
        nonzero = pattern != 0
        
        # Count pixels in each row and column
        row_counts = np.sum(nonzero, axis=1)
        col_counts = np.sum(nonzero, axis=0)
        
        # L-shape has specific row/column pattern
        rows_with_pixels = np.sum(row_counts > 0)
        cols_with_pixels = np.sum(col_counts > 0)
        
        h, w = pattern.shape
        
        # Check L-shape pattern (2 arms)
        if rows_with_pixels == h and cols_with_pixels >= 2:
            # One row with many pixels, rest with 1
            if np.max(row_counts) >= 2 and np.sum(row_counts == 1) >= h - 1:
                return True
        
        return False
    
    def _is_t_shape(self, pattern: np.ndarray) -> bool:
        """Check if pattern is T-shaped."""
        nonzero = pattern != 0
        h, w = pattern.shape
        
        if h < 2 or w < 3:
            return False
        
        # T-shape: one full row/column, one perpendicular
        row_counts = np.sum(nonzero, axis=1)
        col_counts = np.sum(nonzero, axis=0)
        
        # Check for T pattern
        full_rows = np.where(row_counts == w)[0]
        full_cols = np.where(col_counts == h)[0]
        
        if len(full_rows) == 1 and len(full_cols) == 1:
            return True
        
        return False
    
    def _has_hole(self, pattern: np.ndarray) -> bool:
        """Check if object has enclosed holes."""
        h, w = pattern.shape
        if h < 3 or w < 3:
            return False
        
        # Simple check: interior zeros surrounded by non-zeros
        for r in range(1, h-1):
            for c in range(1, w-1):
                if pattern[r, c] == 0:
                    # Check if surrounded
                    neighbors = [
                        pattern[r-1, c], pattern[r+1, c],
                        pattern[r, c-1], pattern[r, c+1]
                    ]
                    if all(n != 0 for n in neighbors):
                        return True
        
        return False
    
    def find_relations(
        self, 
        objects: List[ArcObject]
    ) -> List[ObjectRelation]:
        """Find spatial relationships between all objects."""
        relations = []
        
        for obj1, obj2 in itertools.combinations(objects, 2):
            rels = self._compute_relations(obj1, obj2)
            relations.extend(rels)
        
        return relations
    
    def _compute_relations(
        self, 
        obj1: ArcObject, 
        obj2: ArcObject
    ) -> List[ObjectRelation]:
        """Compute relationships between two objects."""
        relations = []
        
        b1, b2 = obj1.bbox, obj2.bbox
        c1, c2 = b1.center, b2.center
        
        # Vertical relations
        if c1[0] < c2[0]:  # obj1 above obj2
            relations.append(ObjectRelation(obj1.id, obj2.id, SpatialRelation.ABOVE))
            relations.append(ObjectRelation(obj2.id, obj1.id, SpatialRelation.BELOW))
        elif c1[0] > c2[0]:
            relations.append(ObjectRelation(obj1.id, obj2.id, SpatialRelation.BELOW))
            relations.append(ObjectRelation(obj2.id, obj1.id, SpatialRelation.ABOVE))
        
        # Horizontal relations
        if c1[1] < c2[1]:  # obj1 left of obj2
            relations.append(ObjectRelation(obj1.id, obj2.id, SpatialRelation.LEFT))
            relations.append(ObjectRelation(obj2.id, obj1.id, SpatialRelation.RIGHT))
        elif c1[1] > c2[1]:
            relations.append(ObjectRelation(obj1.id, obj2.id, SpatialRelation.RIGHT))
            relations.append(ObjectRelation(obj2.id, obj1.id, SpatialRelation.LEFT))
        
        # Alignment
        if abs(c1[0] - c2[0]) < 0.5:
            relations.append(ObjectRelation(obj1.id, obj2.id, SpatialRelation.ALIGNED_H))
        if abs(c1[1] - c2[1]) < 0.5:
            relations.append(ObjectRelation(obj1.id, obj2.id, SpatialRelation.ALIGNED_V))
        
        # Same row/column
        if b1.min_row <= c2[0] <= b1.max_row or b2.min_row <= c1[0] <= b2.max_row:
            relations.append(ObjectRelation(obj1.id, obj2.id, SpatialRelation.SAME_ROW))
        if b1.min_col <= c2[1] <= b1.max_col or b2.min_col <= c1[1] <= b2.max_col:
            relations.append(ObjectRelation(obj1.id, obj2.id, SpatialRelation.SAME_COL))
        
        # Touching
        if self._are_touching(obj1, obj2):
            relations.append(ObjectRelation(obj1.id, obj2.id, SpatialRelation.TOUCHING))
        
        # Containment
        if self._contains(obj1, obj2):
            relations.append(ObjectRelation(obj1.id, obj2.id, SpatialRelation.CONTAINS))
            relations.append(ObjectRelation(obj2.id, obj1.id, SpatialRelation.INSIDE))
        elif self._contains(obj2, obj1):
            relations.append(ObjectRelation(obj2.id, obj1.id, SpatialRelation.CONTAINS))
            relations.append(ObjectRelation(obj1.id, obj2.id, SpatialRelation.INSIDE))
        
        # Distance
        dist = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
        for rel in relations:
            rel.distance = dist
        
        return relations
    
    def _are_touching(self, obj1: ArcObject, obj2: ArcObject) -> bool:
        """Check if two objects are touching (adjacent)."""
        for r1, c1 in obj1.pixels:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if (r1 + dr, c1 + dc) in obj2.pixels:
                    return True
        return False
    
    def _contains(self, outer: ArcObject, inner: ArcObject) -> bool:
        """Check if outer object's bbox contains inner object."""
        b1, b2 = outer.bbox, inner.bbox
        return (b1.min_row <= b2.min_row and b1.max_row >= b2.max_row and
                b1.min_col <= b2.min_col and b1.max_col >= b2.max_col)
    
    def analyze_grid(
        self, 
        grid: np.ndarray, 
        background: int = 0
    ) -> GridAnalysis:
        """
        Perform complete analysis of a grid.
        
        Detects objects, analyzes relationships, and identifies patterns.
        """
        objects = self.detect_objects(grid, background)
        relations = self.find_relations(objects)
        
        # Group objects
        color_groups = defaultdict(list)
        size_groups = defaultdict(list)
        shape_groups = defaultdict(list)
        
        for obj in objects:
            color_groups[obj.color].append(obj.id)
            size_groups[obj.size].append(obj.id)
            shape_groups[obj.shape_type].append(obj.id)
        
        # Check grid-level symmetry
        is_symmetric_h = np.array_equal(grid, np.fliplr(grid))
        is_symmetric_v = np.array_equal(grid, np.flipud(grid))
        
        # Detect grid structure
        has_grid_structure, grid_cell_size = self._detect_grid_structure(grid, background)
        
        return GridAnalysis(
            objects=objects,
            relations=relations,
            background_color=background,
            grid_shape=grid.shape,
            num_objects=len(objects),
            unique_colors=set(grid.flatten()) - {background},
            has_grid_structure=has_grid_structure,
            grid_cell_size=grid_cell_size,
            is_symmetric_h=is_symmetric_h,
            is_symmetric_v=is_symmetric_v,
            color_groups=dict(color_groups),
            size_groups=dict(size_groups),
            shape_groups=dict(shape_groups)
        )
    
    def _detect_grid_structure(
        self, 
        grid: np.ndarray, 
        background: int
    ) -> Tuple[bool, Optional[Tuple[int, int]]]:
        """Detect if grid has regular cell structure (like tic-tac-toe)."""
        h, w = grid.shape
        
        # Try common cell sizes
        for cell_h in range(2, h // 2 + 1):
            if h % cell_h != 0:
                continue
            for cell_w in range(2, w // 2 + 1):
                if w % cell_w != 0:
                    continue
                
                # Check if subdivisions create uniform cells
                cells = []
                for i in range(0, h, cell_h):
                    for j in range(0, w, cell_w):
                        cell = grid[i:i+cell_h, j:j+cell_w]
                        cells.append(cell)
                
                # Check for pattern repetition or structure
                if len(cells) >= 4:  # At least 2x2 grid
                    return True, (cell_h, cell_w)
        
        return False, None
    
    def compare_objects(
        self, 
        obj1: ArcObject, 
        obj2: ArcObject
    ) -> Dict[str, Any]:
        """Compare two objects and return differences."""
        comparison = {
            'same_color': obj1.color == obj2.color,
            'same_size': obj1.size == obj2.size,
            'same_shape': obj1.shape_type == obj2.shape_type,
            'same_pattern': False,
            'position_delta': (
                obj1.bbox.center[0] - obj2.bbox.center[0],
                obj1.bbox.center[1] - obj2.bbox.center[1]
            ),
            'size_ratio': obj1.size / obj2.size if obj2.size > 0 else float('inf'),
        }
        
        # Compare patterns
        if obj1.bbox.height == obj2.bbox.height and obj1.bbox.width == obj2.bbox.width:
            p1 = obj1.get_relative_pattern()
            p2 = obj2.get_relative_pattern()
            comparison['same_pattern'] = np.array_equal(p1, p2)
        
        return comparison


# Convenience functions
def detect_objects(grid: np.ndarray, background: int = 0) -> List[ArcObject]:
    """Detect all objects in a grid."""
    detector = ObjectDetector()
    return detector.detect_objects(grid, background)


def analyze_grid(grid: np.ndarray, background: int = 0) -> GridAnalysis:
    """Perform complete grid analysis."""
    detector = ObjectDetector()
    return detector.analyze_grid(grid, background)


def compare_grids(
    input_grid: np.ndarray, 
    output_grid: np.ndarray
) -> Dict[str, Any]:
    """Compare input and output grids for transformation analysis."""
    detector = ObjectDetector()
    
    input_analysis = detector.analyze_grid(input_grid)
    output_analysis = detector.analyze_grid(output_grid)
    
    return {
        'input': input_analysis,
        'output': output_analysis,
        'object_count_change': output_analysis.num_objects - input_analysis.num_objects,
        'shape_change': input_grid.shape != output_grid.shape,
        'color_change': input_analysis.unique_colors != output_analysis.unique_colors,
        'new_colors': output_analysis.unique_colors - input_analysis.unique_colors,
        'removed_colors': input_analysis.unique_colors - output_analysis.unique_colors,
    }
