"""
Enhanced DSL Primitives for ARC-AGI

Extended set of primitives that cover more ARC transformation patterns.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from scipy import ndimage


# ==================== GEOMETRIC TRANSFORMATIONS ====================

def rotate_cw(grid: np.ndarray) -> np.ndarray:
    """Rotate grid 90 degrees clockwise."""
    return np.rot90(grid, k=-1)

def rotate_ccw(grid: np.ndarray) -> np.ndarray:
    """Rotate grid 90 degrees counter-clockwise."""
    return np.rot90(grid, k=1)

def rotate_180(grid: np.ndarray) -> np.ndarray:
    """Rotate grid 180 degrees."""
    return np.rot90(grid, k=2)

def reflect_horizontal(grid: np.ndarray) -> np.ndarray:
    """Flip grid horizontally (left-right)."""
    return np.fliplr(grid)

def reflect_vertical(grid: np.ndarray) -> np.ndarray:
    """Flip grid vertically (up-down)."""
    return np.flipud(grid)

def transpose(grid: np.ndarray) -> np.ndarray:
    """Transpose grid (mirror along diagonal)."""
    return np.transpose(grid)

def transpose_anti(grid: np.ndarray) -> np.ndarray:
    """Anti-transpose (mirror along anti-diagonal)."""
    return np.flipud(np.fliplr(np.transpose(grid)))


# ==================== ROLLING/SHIFTING ====================

def roll_up(grid: np.ndarray) -> np.ndarray:
    """Roll grid up by 1 row."""
    return np.roll(grid, -1, axis=0)

def roll_down(grid: np.ndarray) -> np.ndarray:
    """Roll grid down by 1 row."""
    return np.roll(grid, 1, axis=0)

def roll_left(grid: np.ndarray) -> np.ndarray:
    """Roll grid left by 1 column."""
    return np.roll(grid, -1, axis=1)

def roll_right(grid: np.ndarray) -> np.ndarray:
    """Roll grid right by 1 column."""
    return np.roll(grid, 1, axis=1)


# ==================== CROPPING/PADDING ====================

def crop_to_content(grid: np.ndarray) -> np.ndarray:
    """Crop grid to bounding box of non-zero content."""
    coords = np.argwhere(grid > 0)
    if len(coords) == 0:
        return grid
    min_r, min_c = coords.min(axis=0)
    max_r, max_c = coords.max(axis=0)
    return grid[min_r:max_r+1, min_c:max_c+1]

def crop_to_color(grid: np.ndarray, color: int = 1) -> np.ndarray:
    """Crop to bounding box of specific color."""
    coords = np.argwhere(grid == color)
    if len(coords) == 0:
        return grid
    min_r, min_c = coords.min(axis=0)
    max_r, max_c = coords.max(axis=0)
    return grid[min_r:max_r+1, min_c:max_c+1]

def pad_to_square(grid: np.ndarray) -> np.ndarray:
    """Pad grid with zeros to make it square."""
    rows, cols = grid.shape
    size = max(rows, cols)
    result = np.zeros((size, size), dtype=grid.dtype)
    result[:rows, :cols] = grid
    return result

def remove_border(grid: np.ndarray) -> np.ndarray:
    """Remove one pixel border from all sides."""
    if grid.shape[0] > 2 and grid.shape[1] > 2:
        return grid[1:-1, 1:-1]
    return grid


# ==================== SCALING ====================

def scale_2x(grid: np.ndarray) -> np.ndarray:
    """Scale grid up by 2x."""
    return np.repeat(np.repeat(grid, 2, axis=0), 2, axis=1)

def scale_3x(grid: np.ndarray) -> np.ndarray:
    """Scale grid up by 3x."""
    return np.repeat(np.repeat(grid, 3, axis=0), 3, axis=1)

def scale_down_2x(grid: np.ndarray) -> np.ndarray:
    """Scale grid down by 2x (taking every 2nd pixel)."""
    return grid[::2, ::2]

def tile_2x2(grid: np.ndarray) -> np.ndarray:
    """Tile the grid 2x2."""
    return np.tile(grid, (2, 2))

def tile_3x3(grid: np.ndarray) -> np.ndarray:
    """Tile the grid 3x3."""
    return np.tile(grid, (3, 3))


# ==================== COLOR OPERATIONS ====================

def invert_binary(grid: np.ndarray) -> np.ndarray:
    """Invert binary grid (swap 0 and non-zero)."""
    unique = np.unique(grid)
    if len(unique) == 2 and 0 in unique:
        other = unique[unique != 0][0]
        return np.where(grid == 0, other, 0)
    return grid

def replace_color(grid: np.ndarray, src: int = 1, dst: int = 2) -> np.ndarray:
    """Replace one color with another."""
    return np.where(grid == src, dst, grid)

def swap_colors(grid: np.ndarray, c1: int = 1, c2: int = 2) -> np.ndarray:
    """Swap two colors."""
    result = grid.copy()
    result[grid == c1] = c2
    result[grid == c2] = c1
    return result

def map_colors_to_indices(grid: np.ndarray) -> np.ndarray:
    """Map each unique color to sequential indices 0, 1, 2, ..."""
    unique = np.unique(grid)
    mapping = {c: i for i, c in enumerate(unique)}
    result = np.zeros_like(grid)
    for c, i in mapping.items():
        result[grid == c] = i
    return result

def set_all_nonzero_to(grid: np.ndarray, color: int = 1) -> np.ndarray:
    """Set all non-zero pixels to a specific color."""
    return np.where(grid > 0, color, 0)

def normalize_colors(grid: np.ndarray) -> np.ndarray:
    """Normalize colors to 1 for any non-background."""
    return (grid > 0).astype(grid.dtype)


# ==================== OBJECT OPERATIONS ====================

def find_objects(grid: np.ndarray, connectivity: int = 4) -> List[Tuple[int, np.ndarray]]:
    """Find all connected components."""
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

def keep_largest_object(grid: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component."""
    objects = find_objects(grid)
    if not objects:
        return grid
    largest = max(objects, key=lambda x: np.sum(x[1]))
    result = np.zeros_like(grid)
    result[largest[1]] = largest[0]
    return result

def keep_smallest_object(grid: np.ndarray) -> np.ndarray:
    """Keep only the smallest connected component."""
    objects = find_objects(grid)
    if not objects:
        return grid
    smallest = min(objects, key=lambda x: np.sum(x[1]))
    result = np.zeros_like(grid)
    result[smallest[1]] = smallest[0]
    return result

def remove_largest_object(grid: np.ndarray) -> np.ndarray:
    """Remove the largest connected component."""
    objects = find_objects(grid)
    if len(objects) <= 1:
        return np.zeros_like(grid)
    largest = max(objects, key=lambda x: np.sum(x[1]))
    result = grid.copy()
    result[largest[1]] = 0
    return result

def count_objects(grid: np.ndarray) -> int:
    """Count number of connected components."""
    return len(find_objects(grid))

def extract_object(grid: np.ndarray, obj_idx: int = 0) -> np.ndarray:
    """Extract a specific object (cropped to its bounding box)."""
    objects = find_objects(grid)
    if obj_idx >= len(objects):
        return grid
    color, mask = objects[obj_idx]
    result = np.zeros_like(grid)
    result[mask] = color
    return crop_to_content(result)

def color_each_object(grid: np.ndarray) -> np.ndarray:
    """Assign a unique color (1-9) to each object."""
    objects = find_objects(grid)
    result = np.zeros_like(grid)
    for i, (_, mask) in enumerate(objects):
        result[mask] = (i % 9) + 1
    return result


# ==================== GRAVITY/MOVEMENT ====================

def gravity_down(grid: np.ndarray) -> np.ndarray:
    """Apply gravity - drop all non-zero pixels down."""
    result = np.zeros_like(grid)
    rows, cols = grid.shape
    for c in range(cols):
        column = grid[:, c]
        non_zero = column[column > 0]
        if len(non_zero) > 0:
            result[rows-len(non_zero):rows, c] = non_zero
    return result

def gravity_up(grid: np.ndarray) -> np.ndarray:
    """Apply upward gravity."""
    result = np.zeros_like(grid)
    rows, cols = grid.shape
    for c in range(cols):
        column = grid[:, c]
        non_zero = column[column > 0]
        if len(non_zero) > 0:
            result[:len(non_zero), c] = non_zero
    return result

def gravity_left(grid: np.ndarray) -> np.ndarray:
    """Apply leftward gravity."""
    result = np.zeros_like(grid)
    rows, cols = grid.shape
    for r in range(rows):
        row = grid[r, :]
        non_zero = row[row > 0]
        if len(non_zero) > 0:
            result[r, :len(non_zero)] = non_zero
    return result

def gravity_right(grid: np.ndarray) -> np.ndarray:
    """Apply rightward gravity."""
    result = np.zeros_like(grid)
    rows, cols = grid.shape
    for r in range(rows):
        row = grid[r, :]
        non_zero = row[row > 0]
        if len(non_zero) > 0:
            result[r, cols-len(non_zero):cols] = non_zero
    return result


# ==================== FILL OPERATIONS ====================

def fill_row(grid: np.ndarray) -> np.ndarray:
    """Fill entire row where there's any non-zero pixel."""
    result = grid.copy()
    for r in range(grid.shape[0]):
        if np.any(grid[r, :] > 0):
            color = grid[r, grid[r, :] > 0][0]
            result[r, :] = color
    return result

def fill_column(grid: np.ndarray) -> np.ndarray:
    """Fill entire column where there's any non-zero pixel."""
    result = grid.copy()
    for c in range(grid.shape[1]):
        if np.any(grid[:, c] > 0):
            color = grid[grid[:, c] > 0, c][0]
            result[:, c] = color
    return result

def fill_down_from_top(grid: np.ndarray) -> np.ndarray:
    """Extend each non-zero pixel downward to fill the column below it."""
    result = grid.copy()
    rows, cols = grid.shape
    for c in range(cols):
        fill_color = 0
        for r in range(rows):
            if grid[r, c] > 0:
                fill_color = grid[r, c]
            if fill_color > 0:
                result[r, c] = fill_color
    return result

def fill_up_from_bottom(grid: np.ndarray) -> np.ndarray:
    """Extend each non-zero pixel upward."""
    result = grid.copy()
    rows, cols = grid.shape
    for c in range(cols):
        fill_color = 0
        for r in range(rows-1, -1, -1):
            if grid[r, c] > 0:
                fill_color = grid[r, c]
            if fill_color > 0:
                result[r, c] = fill_color
    return result

def flood_fill_exterior(grid: np.ndarray, fill_color: int = 1) -> np.ndarray:
    """Fill the exterior (connected to border) with a color."""
    rows, cols = grid.shape
    result = grid.copy()
    visited = np.zeros_like(grid, dtype=bool)
    
    # Start from all border cells that are 0
    queue = []
    for r in range(rows):
        for c in [0, cols-1]:
            if grid[r, c] == 0:
                queue.append((r, c))
                visited[r, c] = True
    for c in range(cols):
        for r in [0, rows-1]:
            if grid[r, c] == 0 and not visited[r, c]:
                queue.append((r, c))
                visited[r, c] = True
    
    while queue:
        r, c = queue.pop(0)
        result[r, c] = fill_color
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if not visited[nr, nc] and grid[nr, nc] == 0:
                    visited[nr, nc] = True
                    queue.append((nr, nc))
    
    return result


# ==================== PATTERN/SYMMETRY OPERATIONS ====================

def make_symmetric_h(grid: np.ndarray) -> np.ndarray:
    """Make grid horizontally symmetric by mirroring left half."""
    cols = grid.shape[1]
    mid = cols // 2
    result = grid.copy()
    result[:, mid:] = np.fliplr(result[:, :mid])[:, -(cols-mid):]
    return result

def make_symmetric_v(grid: np.ndarray) -> np.ndarray:
    """Make grid vertically symmetric by mirroring top half."""
    rows = grid.shape[0]
    mid = rows // 2
    result = grid.copy()
    result[mid:, :] = np.flipud(result[:mid, :])[:-(rows-mid) or None, :]
    return result

def detect_pattern_and_tile(grid: np.ndarray) -> np.ndarray:
    """Detect repeating pattern and extract it."""
    rows, cols = grid.shape
    for pr in range(1, rows//2 + 1):
        for pc in range(1, cols//2 + 1):
            if rows % pr == 0 and cols % pc == 0:
                pattern = grid[:pr, :pc]
                reconstructed = np.tile(pattern, (rows//pr, cols//pc))
                if np.array_equal(reconstructed, grid):
                    return pattern
    return grid


# ==================== LOGICAL OPERATIONS ====================

def logical_and(grid1: np.ndarray, grid2: np.ndarray) -> np.ndarray:
    """Logical AND of two grids (both non-zero)."""
    return np.where((grid1 > 0) & (grid2 > 0), grid1, 0)

def logical_or(grid1: np.ndarray, grid2: np.ndarray) -> np.ndarray:
    """Logical OR of two grids."""
    return np.where(grid1 > 0, grid1, grid2)

def logical_xor(grid1: np.ndarray, grid2: np.ndarray) -> np.ndarray:
    """Logical XOR of two grids."""
    return np.where((grid1 > 0) != (grid2 > 0), np.maximum(grid1, grid2), 0)

def difference(grid1: np.ndarray, grid2: np.ndarray) -> np.ndarray:
    """Pixels in grid1 that are not in grid2."""
    return np.where((grid1 > 0) & (grid2 == 0), grid1, 0)


# ==================== MORPHOLOGICAL OPERATIONS ====================

def dilate(grid: np.ndarray) -> np.ndarray:
    """Dilate the grid (expand non-zero regions)."""
    binary = (grid > 0).astype(np.int8)
    dilated = ndimage.binary_dilation(binary).astype(grid.dtype)
    # Keep original colors where they exist, fill new areas with surrounding color
    result = np.zeros_like(grid)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if dilated[r, c]:
                if grid[r, c] > 0:
                    result[r, c] = grid[r, c]
                else:
                    # Find nearest color
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                            if grid[nr, nc] > 0:
                                result[r, c] = grid[nr, nc]
                                break
    return result

def erode(grid: np.ndarray) -> np.ndarray:
    """Erode the grid (shrink non-zero regions)."""
    binary = (grid > 0).astype(np.int8)
    eroded = ndimage.binary_erosion(binary)
    return np.where(eroded, grid, 0)

def outline(grid: np.ndarray) -> np.ndarray:
    """Get the outline/border of non-zero regions."""
    binary = (grid > 0).astype(np.int8)
    eroded = ndimage.binary_erosion(binary)
    outline_mask = binary & ~eroded
    return np.where(outline_mask, grid, 0)

def fill_holes(grid: np.ndarray) -> np.ndarray:
    """Fill holes in the grid."""
    binary = (grid > 0).astype(np.int8)
    filled = ndimage.binary_fill_holes(binary)
    # Get the most common non-zero color
    colors = grid[grid > 0]
    if len(colors) == 0:
        return grid
    most_common = np.bincount(colors).argmax()
    return np.where(filled & (grid == 0), most_common, grid)


# ==================== DIAGONAL OPERATIONS ====================

def diagonal_fill(grid: np.ndarray) -> np.ndarray:
    """Fill diagonals from non-zero pixels."""
    result = grid.copy()
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if grid[r, c] > 0:
                color = grid[r, c]
                # Fill main diagonal
                for d in range(1, max(h, w)):
                    if r + d < h and c + d < w:
                        result[r + d, c + d] = color
                    if r - d >= 0 and c - d >= 0:
                        result[r - d, c - d] = color
    return result

def anti_diagonal_fill(grid: np.ndarray) -> np.ndarray:
    """Fill anti-diagonals from non-zero pixels."""
    result = grid.copy()
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if grid[r, c] > 0:
                color = grid[r, c]
                for d in range(1, max(h, w)):
                    if r + d < h and c - d >= 0:
                        result[r + d, c - d] = color
                    if r - d >= 0 and c + d < w:
                        result[r - d, c + d] = color
    return result

def extend_lines(grid: np.ndarray) -> np.ndarray:
    """Extend all lines (horizontal, vertical) to grid boundaries."""
    result = grid.copy()
    h, w = grid.shape
    
    # Horizontal extension
    for r in range(h):
        nonzero = np.where(grid[r] > 0)[0]
        if len(nonzero) > 0:
            color = grid[r, nonzero[0]]
            result[r, :] = color
    
    return result

def connect_same_color(grid: np.ndarray) -> np.ndarray:
    """Connect pixels of same color with lines."""
    from collections import defaultdict
    result = grid.copy()
    h, w = grid.shape
    
    # Group positions by color
    color_positions = defaultdict(list)
    for r in range(h):
        for c in range(w):
            if grid[r, c] > 0:
                color_positions[grid[r, c]].append((r, c))
    
    # Connect each color group
    for color, positions in color_positions.items():
        if len(positions) < 2:
            continue
        positions.sort()
        for i in range(len(positions) - 1):
            r1, c1 = positions[i]
            r2, c2 = positions[i + 1]
            # Draw line
            if r1 == r2:  # Horizontal
                for c in range(min(c1, c2), max(c1, c2) + 1):
                    result[r1, c] = color
            elif c1 == c2:  # Vertical
                for r in range(min(r1, r2), max(r1, r2) + 1):
                    result[r, c1] = color
    
    return result


# ==================== MIRRORING OPERATIONS ====================

def mirror_right(grid: np.ndarray) -> np.ndarray:
    """Mirror grid to the right (double width)."""
    return np.hstack([grid, np.fliplr(grid)])

def mirror_left(grid: np.ndarray) -> np.ndarray:
    """Mirror grid to the left (double width)."""
    return np.hstack([np.fliplr(grid), grid])

def mirror_down(grid: np.ndarray) -> np.ndarray:
    """Mirror grid downward (double height)."""
    return np.vstack([grid, np.flipud(grid)])

def mirror_up(grid: np.ndarray) -> np.ndarray:
    """Mirror grid upward (double height)."""
    return np.vstack([np.flipud(grid), grid])

def mirror_4way(grid: np.ndarray) -> np.ndarray:
    """Create 4-way mirror (quadruple size)."""
    top = np.hstack([grid, np.fliplr(grid)])
    return np.vstack([top, np.flipud(top)])


# ==================== GRID MANIPULATION ====================

def split_horizontal(grid: np.ndarray) -> np.ndarray:
    """Take left half of grid."""
    return grid[:, :grid.shape[1]//2]

def split_vertical(grid: np.ndarray) -> np.ndarray:
    """Take top half of grid."""
    return grid[:grid.shape[0]//2, :]

def quarter_top_left(grid: np.ndarray) -> np.ndarray:
    """Get top-left quarter."""
    h, w = grid.shape
    return grid[:h//2, :w//2]

def quarter_top_right(grid: np.ndarray) -> np.ndarray:
    """Get top-right quarter."""
    h, w = grid.shape
    return grid[:h//2, w//2:]

def quarter_bottom_left(grid: np.ndarray) -> np.ndarray:
    """Get bottom-left quarter."""
    h, w = grid.shape
    return grid[h//2:, :w//2]

def quarter_bottom_right(grid: np.ndarray) -> np.ndarray:
    """Get bottom-right quarter."""
    h, w = grid.shape
    return grid[h//2:, w//2:]

def extract_unique_pattern(grid: np.ndarray) -> np.ndarray:
    """Extract the minimal repeating pattern."""
    h, w = grid.shape
    for ph in range(1, h + 1):
        if h % ph != 0:
            continue
        for pw in range(1, w + 1):
            if w % pw != 0:
                continue
            pattern = grid[:ph, :pw]
            tiled = np.tile(pattern, (h // ph, w // pw))
            if np.array_equal(tiled, grid):
                return pattern
    return grid

def add_border(grid: np.ndarray, color: int = 1) -> np.ndarray:
    """Add a 1-pixel border around the grid."""
    h, w = grid.shape
    result = np.full((h + 2, w + 2), color, dtype=grid.dtype)
    result[1:-1, 1:-1] = grid
    return result

def frame_grid(grid: np.ndarray, color: int = 1) -> np.ndarray:
    """Add a frame (hollow border) around the grid."""
    h, w = grid.shape
    result = np.zeros((h + 2, w + 2), dtype=grid.dtype)
    result[0, :] = color
    result[-1, :] = color
    result[:, 0] = color
    result[:, -1] = color
    result[1:-1, 1:-1] = grid
    return result


# ==================== OVERLAY OPERATIONS ====================

def overlay_max(grid: np.ndarray) -> np.ndarray:
    """If grid has pattern, overlay parts taking max color."""
    return grid  # Base case

def compress_horizontal(grid: np.ndarray) -> np.ndarray:
    """Compress horizontally by OR-ing columns."""
    h, w = grid.shape
    if w < 2:
        return grid
    mid = w // 2
    left = grid[:, :mid]
    right = grid[:, mid:mid+left.shape[1]]
    return np.maximum(left, right)

def compress_vertical(grid: np.ndarray) -> np.ndarray:
    """Compress vertically by OR-ing rows."""
    h, w = grid.shape
    if h < 2:
        return grid
    mid = h // 2
    top = grid[:mid, :]
    bottom = grid[mid:mid+top.shape[0], :]
    return np.maximum(top, bottom)


# ==================== SORTING/ORDERING ====================

def sort_rows_by_sum(grid: np.ndarray) -> np.ndarray:
    """Sort rows by sum of pixel values."""
    row_sums = np.sum(grid, axis=1)
    sorted_indices = np.argsort(row_sums)[::-1]
    return grid[sorted_indices]

def sort_cols_by_sum(grid: np.ndarray) -> np.ndarray:
    """Sort columns by sum of pixel values."""
    col_sums = np.sum(grid, axis=0)
    sorted_indices = np.argsort(col_sums)[::-1]
    return grid[:, sorted_indices]

def reverse_rows(grid: np.ndarray) -> np.ndarray:
    """Reverse row order."""
    return grid[::-1, :]

def reverse_cols(grid: np.ndarray) -> np.ndarray:
    """Reverse column order."""
    return grid[:, ::-1]


# ==================== SPECIAL OPERATIONS ====================

def identity(grid: np.ndarray) -> np.ndarray:
    """Return grid unchanged."""
    return grid.copy()

def clear_grid(grid: np.ndarray) -> np.ndarray:
    """Return empty grid of same size."""
    return np.zeros_like(grid)

def sample_to_1x1(grid: np.ndarray) -> np.ndarray:
    """Sample grid to 1x1 (most common non-zero color)."""
    nonzero = grid[grid > 0]
    if len(nonzero) == 0:
        return np.array([[0]])
    most_common = np.bincount(nonzero).argmax()
    return np.array([[most_common]])

def count_nonzero_to_grid(grid: np.ndarray) -> np.ndarray:
    """Return 1x1 grid with count of non-zero pixels."""
    return np.array([[np.count_nonzero(grid)]])

def unique_colors_count(grid: np.ndarray) -> np.ndarray:
    """Return 1x1 grid with count of unique colors."""
    return np.array([[len(np.unique(grid[grid > 0]))]])

def most_common_color(grid: np.ndarray) -> np.ndarray:
    """Return 1x1 grid with most common non-zero color."""
    nonzero = grid[grid > 0]
    if len(nonzero) == 0:
        return np.array([[0]])
    return np.array([[np.bincount(nonzero).argmax()]])

def min_color(grid: np.ndarray) -> np.ndarray:
    """Set all non-zero to minimum color value."""
    if np.any(grid > 0):
        min_c = np.min(grid[grid > 0])
        return np.where(grid > 0, min_c, 0)
    return grid

def max_color(grid: np.ndarray) -> np.ndarray:
    """Set all non-zero to maximum color value."""
    if np.any(grid > 0):
        max_c = np.max(grid)
        return np.where(grid > 0, max_c, 0)
    return grid


# ==================== REGION OPERATIONS ====================

def top_row(grid: np.ndarray) -> np.ndarray:
    """Extract top row."""
    return grid[0:1, :]

def bottom_row(grid: np.ndarray) -> np.ndarray:
    """Extract bottom row."""
    return grid[-1:, :]

def left_col(grid: np.ndarray) -> np.ndarray:
    """Extract left column."""
    return grid[:, 0:1]

def right_col(grid: np.ndarray) -> np.ndarray:
    """Extract right column."""
    return grid[:, -1:]

def main_diagonal(grid: np.ndarray) -> np.ndarray:
    """Extract main diagonal as column."""
    diag = np.diag(grid)
    return diag.reshape(-1, 1)

def anti_diagonal(grid: np.ndarray) -> np.ndarray:
    """Extract anti-diagonal as column."""
    diag = np.diag(np.fliplr(grid))
    return diag.reshape(-1, 1)


# ==================== COLOR OPERATIONS EXTENDED ====================

def shift_colors_up(grid: np.ndarray) -> np.ndarray:
    """Shift all colors up by 1 (1->2, 2->3, etc)."""
    result = grid.copy()
    result[grid > 0] = grid[grid > 0] + 1
    result[result > 9] = 9
    return result

def shift_colors_down(grid: np.ndarray) -> np.ndarray:
    """Shift all colors down by 1."""
    result = grid.copy()
    result[grid > 1] = grid[grid > 1] - 1
    return result

def color_to_1(grid: np.ndarray) -> np.ndarray:
    """Map all non-zero to color 1."""
    return np.where(grid > 0, 1, 0)

def color_to_2(grid: np.ndarray) -> np.ndarray:
    """Map all non-zero to color 2."""
    return np.where(grid > 0, 2, 0)

def color_to_3(grid: np.ndarray) -> np.ndarray:
    """Map all non-zero to color 3."""
    return np.where(grid > 0, 3, 0)

def swap_background_foreground(grid: np.ndarray) -> np.ndarray:
    """Swap 0 with most common non-zero color."""
    if not np.any(grid > 0):
        return grid
    nonzero = grid[grid > 0]
    most_common = np.bincount(nonzero).argmax()
    result = grid.copy()
    result[grid == 0] = most_common
    result[grid == most_common] = 0
    return result


# ==================== ENHANCED DSL REGISTRY ====================

enhanced_dsl_registry = {
    # Geometric
    "rotate_cw": rotate_cw,
    "rotate_ccw": rotate_ccw,
    "rotate_180": rotate_180,
    "reflect_h": reflect_horizontal,
    "reflect_v": reflect_vertical,
    "transpose": transpose,
    "transpose_anti": transpose_anti,
    
    # Rolling
    "roll_up": roll_up,
    "roll_down": roll_down,
    "roll_left": roll_left,
    "roll_right": roll_right,
    
    # Cropping
    "crop": crop_to_content,
    "remove_border": remove_border,
    "pad_square": pad_to_square,
    
    # Scaling
    "scale_2x": scale_2x,
    "scale_3x": scale_3x,
    "scale_down_2x": scale_down_2x,
    "tile_2x2": tile_2x2,
    "tile_3x3": tile_3x3,
    
    # Color
    "invert": invert_binary,
    "normalize": normalize_colors,
    "color_each": color_each_object,
    "color_to_1": color_to_1,
    "color_to_2": color_to_2,
    "color_to_3": color_to_3,
    "shift_colors_up": shift_colors_up,
    "shift_colors_down": shift_colors_down,
    "min_color": min_color,
    "max_color": max_color,
    "swap_bg_fg": swap_background_foreground,
    
    # Object
    "keep_largest": keep_largest_object,
    "keep_smallest": keep_smallest_object,
    "remove_largest": remove_largest_object,
    
    # Gravity
    "gravity_down": gravity_down,
    "gravity_up": gravity_up,
    "gravity_left": gravity_left,
    "gravity_right": gravity_right,
    
    # Fill
    "fill_row": fill_row,
    "fill_column": fill_column,
    "fill_down": fill_down_from_top,
    "fill_up": fill_up_from_bottom,
    "diagonal_fill": diagonal_fill,
    "anti_diagonal_fill": anti_diagonal_fill,
    "extend_lines": extend_lines,
    "connect_same_color": connect_same_color,
    
    # Symmetry/Mirroring
    "sym_h": make_symmetric_h,
    "sym_v": make_symmetric_v,
    "mirror_right": mirror_right,
    "mirror_left": mirror_left,
    "mirror_down": mirror_down,
    "mirror_up": mirror_up,
    "mirror_4way": mirror_4way,
    
    # Morphological
    "dilate": dilate,
    "erode": erode,
    "outline": outline,
    "fill_holes": fill_holes,
    
    # Grid manipulation
    "split_h": split_horizontal,
    "split_v": split_vertical,
    "quarter_tl": quarter_top_left,
    "quarter_tr": quarter_top_right,
    "quarter_bl": quarter_bottom_left,
    "quarter_br": quarter_bottom_right,
    "extract_pattern": extract_unique_pattern,
    "add_border": add_border,
    "frame": frame_grid,
    "compress_h": compress_horizontal,
    "compress_v": compress_vertical,
    
    # Sorting
    "sort_rows": sort_rows_by_sum,
    "sort_cols": sort_cols_by_sum,
    "reverse_rows": reverse_rows,
    "reverse_cols": reverse_cols,
    
    # Region extraction
    "top_row": top_row,
    "bottom_row": bottom_row,
    "left_col": left_col,
    "right_col": right_col,
    "main_diag": main_diagonal,
    "anti_diag": anti_diagonal,
    
    # Special
    "identity": identity,
    "clear": clear_grid,
    "sample_1x1": sample_to_1x1,
    "count_to_grid": count_nonzero_to_grid,
    "most_common": most_common_color,
}
