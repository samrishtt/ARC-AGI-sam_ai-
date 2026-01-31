"""
Grid Subdivision Operations

Many ARC tasks involve:
- Splitting grids into regions
- Extracting specific cells from subdivided grids
- Overlaying subdivisions
- Comparing subdivisions

This module handles these operations.
"""

import numpy as np
from typing import List, Tuple, Optional, Callable


def subdivide_grid(grid: np.ndarray, rows: int, cols: int) -> List[List[np.ndarray]]:
    """
    Subdivide a grid into rows x cols sub-grids.
    
    Returns a 2D list of sub-grids.
    """
    h, w = grid.shape
    if h % rows != 0 or w % cols != 0:
        return []
    
    cell_h = h // rows
    cell_w = w // cols
    
    result = []
    for r in range(rows):
        row_cells = []
        for c in range(cols):
            cell = grid[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
            row_cells.append(cell)
        result.append(row_cells)
    
    return result


def get_subdivision_cell(grid: np.ndarray, rows: int, cols: int, cell_r: int, cell_c: int) -> np.ndarray:
    """Get a specific cell from a subdivided grid."""
    cells = subdivide_grid(grid, rows, cols)
    if not cells or cell_r >= len(cells) or cell_c >= len(cells[0]):
        return grid
    return cells[cell_r][cell_c]


def overlay_subdivisions(grid: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """
    Overlay all subdivisions using OR (maximum).
    
    Common pattern: grid has multiple cells, output is one cell with all patterns combined.
    """
    cells = subdivide_grid(grid, rows, cols)
    if not cells:
        return grid
    
    result = np.zeros_like(cells[0][0])
    for row in cells:
        for cell in row:
            result = np.maximum(result, cell)
    
    return result


def xor_subdivisions(grid: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """XOR all subdivisions (find unique elements)."""
    cells = subdivide_grid(grid, rows, cols)
    if not cells:
        return grid
    
    # Count occurrences at each position
    first_cell = cells[0][0]
    h, w = first_cell.shape
    counts = np.zeros((h, w), dtype=int)
    values = np.zeros((h, w), dtype=first_cell.dtype)
    
    for row in cells:
        for cell in row:
            mask = cell > 0
            counts += mask.astype(int)
            values = np.where(mask, cell, values)
    
    # Keep only positions that appear exactly once
    result = np.where(counts == 1, values, 0)
    return result


def and_subdivisions(grid: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """AND all subdivisions (find common elements)."""
    cells = subdivide_grid(grid, rows, cols)
    if not cells:
        return grid
    
    result = cells[0][0].copy()
    for row in cells:
        for cell in row:
            result = np.where((result > 0) & (cell > 0), result, 0)
    
    return result


def subtract_subdivisions(grid: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """
    Subtract second subdivision from first.
    
    Common pattern: difference between two halves.
    """
    cells = subdivide_grid(grid, rows, cols)
    if not cells or len(cells) < 1 or len(cells[0]) < 2:
        return grid
    
    first = cells[0][0]
    second = cells[0][1] if len(cells[0]) > 1 else cells[1][0] if len(cells) > 1 else first
    
    result = np.where((first > 0) & (second == 0), first, 0)
    return result


def find_grid_separators(grid: np.ndarray) -> Tuple[List[int], List[int]]:
    """
    Find rows/columns that act as separators (all same color).
    
    Returns (row_indices, col_indices) of separator lines.
    """
    h, w = grid.shape
    
    row_seps = []
    for r in range(h):
        row = grid[r, :]
        if len(np.unique(row)) == 1:
            row_seps.append(r)
    
    col_seps = []
    for c in range(w):
        col = grid[:, c]
        if len(np.unique(col)) == 1:
            col_seps.append(c)
    
    return row_seps, col_seps


def split_by_separators(
    grid: np.ndarray, 
    sep_color: Optional[int] = None
) -> List[np.ndarray]:
    """
    Split grid by separator lines of a specific color.
    
    Finds rows/cols that are entirely one color and splits there.
    """
    h, w = grid.shape
    
    # Find horizontal separators
    h_seps = [-1]  # Start boundary
    for r in range(h):
        row = grid[r, :]
        unique = np.unique(row)
        if len(unique) == 1:
            if sep_color is None or unique[0] == sep_color:
                h_seps.append(r)
    h_seps.append(h)  # End boundary
    
    # Find vertical separators
    v_seps = [-1]
    for c in range(w):
        col = grid[:, c]
        unique = np.unique(col)
        if len(unique) == 1:
            if sep_color is None or unique[0] == sep_color:
                v_seps.append(c)
    v_seps.append(w)
    
    # Extract cells
    cells = []
    for i in range(len(h_seps) - 1):
        for j in range(len(v_seps) - 1):
            r1, r2 = h_seps[i] + 1, h_seps[i + 1]
            c1, c2 = v_seps[j] + 1, v_seps[j + 1]
            if r1 < r2 and c1 < c2:
                cell = grid[r1:r2, c1:c2]
                if cell.size > 0:
                    cells.append(cell)
    
    return cells


def detect_subdivision_pattern(
    examples: List[Tuple[np.ndarray, np.ndarray]]
) -> Optional[Tuple[str, Callable]]:
    """
    Detect if a subdivision pattern applies.
    
    Returns (description, transform_fn) or None.
    """
    if not examples:
        return None
    
    inp, out = examples[0]
    in_h, in_w = inp.shape
    out_h, out_w = out.shape
    
    # Check for simple subdivision ratios
    if in_h == out_h * 2 and in_w == out_w:
        # Vertical halves
        for op_name, op_fn in [
            ("top_half", lambda g: g[:g.shape[0]//2, :]),
            ("bottom_half", lambda g: g[g.shape[0]//2:, :]),
            ("overlay_v", lambda g: overlay_subdivisions(g, 2, 1)),
            ("xor_v", lambda g: xor_subdivisions(g, 2, 1)),
            ("and_v", lambda g: and_subdivisions(g, 2, 1)),
        ]:
            if _verify(examples, op_fn):
                return (op_name, op_fn)
    
    if in_w == out_w * 2 and in_h == out_h:
        # Horizontal halves
        for op_name, op_fn in [
            ("left_half", lambda g: g[:, :g.shape[1]//2]),
            ("right_half", lambda g: g[:, g.shape[1]//2:]),
            ("overlay_h", lambda g: overlay_subdivisions(g, 1, 2)),
            ("xor_h", lambda g: xor_subdivisions(g, 1, 2)),
            ("and_h", lambda g: and_subdivisions(g, 1, 2)),
        ]:
            if _verify(examples, op_fn):
                return (op_name, op_fn)
    
    if in_h == out_h * 2 and in_w == out_w * 2:
        # Quadrants
        for op_name, op_fn in [
            ("quarter_tl", lambda g: g[:g.shape[0]//2, :g.shape[1]//2]),
            ("quarter_tr", lambda g: g[:g.shape[0]//2, g.shape[1]//2:]),
            ("quarter_bl", lambda g: g[g.shape[0]//2:, :g.shape[1]//2]),
            ("quarter_br", lambda g: g[g.shape[0]//2:, g.shape[1]//2:]),
            ("overlay_4", lambda g: overlay_subdivisions(g, 2, 2)),
            ("xor_4", lambda g: xor_subdivisions(g, 2, 2)),
            ("and_4", lambda g: and_subdivisions(g, 2, 2)),
        ]:
            if _verify(examples, op_fn):
                return (op_name, op_fn)
    
    if in_h == out_h * 3 and in_w == out_w * 3:
        # 3x3 grid
        for op_name, op_fn in [
            ("overlay_9", lambda g: overlay_subdivisions(g, 3, 3)),
            ("center_cell", lambda g: get_subdivision_cell(g, 3, 3, 1, 1)),
        ]:
            if _verify(examples, op_fn):
                return (op_name, op_fn)
    
    return None


def _verify(examples: List[Tuple[np.ndarray, np.ndarray]], fn: Callable) -> bool:
    """Verify a function works on all examples."""
    try:
        for inp, out in examples:
            result = fn(inp)
            if not np.array_equal(result, out):
                return False
        return True
    except:
        return False


# Pre-built operations for common subdivisions
SUBDIVISION_OPS = {
    # Halves
    "top_half": lambda g: g[:g.shape[0]//2, :],
    "bottom_half": lambda g: g[g.shape[0]//2:, :],
    "left_half": lambda g: g[:, :g.shape[1]//2],
    "right_half": lambda g: g[:, g.shape[1]//2:],
    
    # Quadrants
    "quarter_tl": lambda g: g[:g.shape[0]//2, :g.shape[1]//2],
    "quarter_tr": lambda g: g[:g.shape[0]//2, g.shape[1]//2:],
    "quarter_bl": lambda g: g[g.shape[0]//2:, :g.shape[1]//2],
    "quarter_br": lambda g: g[g.shape[0]//2:, g.shape[1]//2:],
    
    # Overlays
    "overlay_h2": lambda g: overlay_subdivisions(g, 1, 2),
    "overlay_v2": lambda g: overlay_subdivisions(g, 2, 1),
    "overlay_4": lambda g: overlay_subdivisions(g, 2, 2),
    "overlay_9": lambda g: overlay_subdivisions(g, 3, 3),
    
    # XOR
    "xor_h2": lambda g: xor_subdivisions(g, 1, 2),
    "xor_v2": lambda g: xor_subdivisions(g, 2, 1),
    "xor_4": lambda g: xor_subdivisions(g, 2, 2),
    
    # AND
    "and_h2": lambda g: and_subdivisions(g, 1, 2),
    "and_v2": lambda g: and_subdivisions(g, 2, 1),
    "and_4": lambda g: and_subdivisions(g, 2, 2),
}
