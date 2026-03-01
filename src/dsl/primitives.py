import numpy as np
from typing import List, Any

# =============================================================================
# ARC Domain-Specific Language (DSL) Primitives
# These functions provide a robust toolkit for the LLM to use when 
# writing `transform(grid)` logic. Instead of writing complex matrix
# math from scratch, the LLM can call these pre-tested functions.
# =============================================================================

def rotate_cw(grid: List[List[int]]) -> List[List[int]]:
    """Rotates a 2D grid 90 degrees clockwise."""
    if not grid: return []
    np_grid = np.array(grid)
    rotated = np.rot90(np_grid, k=-1)
    return rotated.tolist()

def rotate_ccw(grid: List[List[int]]) -> List[List[int]]:
    """Rotates a 2D grid 90 degrees counter-clockwise."""
    if not grid: return []
    np_grid = np.array(grid)
    rotated = np.rot90(np_grid, k=1)
    return rotated.tolist()

def flip_horizontal(grid: List[List[int]]) -> List[List[int]]:
    """Flips a 2D grid horizontally (Left-Right)."""
    if not grid: return []
    np_grid = np.array(grid)
    flipped = np.fliplr(np_grid)
    return flipped.tolist()

def flip_vertical(grid: List[List[int]]) -> List[List[int]]:
    """Flips a 2D grid vertically (Top-Bottom)."""
    if not grid: return []
    np_grid = np.array(grid)
    flipped = np.flipud(np_grid)
    return flipped.tolist()

def fill_color(grid: List[List[int]], target_color: int, new_color: int) -> List[List[int]]:
    """Replaces all instances of target_color with new_color."""
    np_grid = np.array(grid)
    np_grid[np_grid == target_color] = new_color
    return np_grid.tolist()

def extract_color(grid: List[List[int]], target_color: int) -> List[List[int]]:
    """Returns a grid with ONLY the target_color. Everything else becomes 0 (black)."""
    np_grid = np.array(grid)
    mask = (np_grid == target_color)
    output = np.zeros_like(np_grid)
    output[mask] = target_color
    return output.tolist()

def crop_to_content(grid: List[List[int]], bg_color: int = 0) -> List[List[int]]:
    """Crops the grid to the bounding box of non-background colors."""
    np_grid = np.array(grid)
    if not np.any(np_grid != bg_color):
        return grid # Return as is if totally empty
        
    mask = np_grid != bg_color
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]
    
    cropped = np_grid[r_min:r_max+1, c_min:c_max+1]
    return cropped.tolist()

def tile_grid(grid: List[List[int]], vertical_repeats: int, horizontal_repeats: int) -> List[List[int]]:
    """Tiles the entire grid M times vertically and N times horizontally."""
    np_grid = np.array(grid)
    tiled = np.tile(np_grid, (vertical_repeats, horizontal_repeats))
    return tiled.tolist()

def shift_grid(grid: List[List[int]], row_shift: int, col_shift: int, bg_color: int = 0) -> List[List[int]]:
    """Shifts the entire grid by (row_shift, col_shift) pixels, filling with bg_color."""
    np_grid = np.array(grid)
    shifted = np.full_like(np_grid, fill_value=bg_color)
    
    # Calculate valid slice ranges
    src_r1 = max(0, -row_shift)
    src_r2 = np_grid.shape[0] - max(0, row_shift)
    src_c1 = max(0, -col_shift)
    src_c2 = np_grid.shape[1] - max(0, col_shift)
    
    dst_r1 = max(0, row_shift)
    dst_r2 = np_grid.shape[0] - max(0, -row_shift)
    dst_c1 = max(0, col_shift)
    dst_c2 = np_grid.shape[1] - max(0, -col_shift)
    
    if src_r1 < src_r2 and src_c1 < src_c2:
        shifted[dst_r1:dst_r2, dst_c1:dst_c2] = np_grid[src_r1:src_r2, src_c1:src_c2]
        
    return shifted.tolist()

def draw_line(grid: List[List[int]], r1: int, c1: int, r2: int, c2: int, color: int) -> List[List[int]]:
    """Draws a straight line from (r1, c1) to (r2, c2) using classic Bresenham's algorithm."""
    np_grid = np.array(grid)
    
    # Bresenham's Line Algorithm
    dr = abs(r2 - r1)
    dc = abs(c2 - c1)
    sr = 1 if r1 < r2 else -1
    sc = 1 if c1 < c2 else -1
    err = dr - dc
    
    while True:
        if 0 <= r1 < np_grid.shape[0] and 0 <= c1 < np_grid.shape[1]:
            np_grid[r1, c1] = color
            
        if r1 == r2 and c1 == c2:
            break
            
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r1 += sr
        if e2 < dr:
            err += dr
            c1 += sc
            
    return np_grid.tolist()

def flood_fill(grid: List[List[int]], r: int, c: int, replacement_color: int) -> List[List[int]]:
    """Fills a contiguous region of the same color starting at (r, c)."""
    np_grid = np.array(grid)
    if not (0 <= r < np_grid.shape[0] and 0 <= c < np_grid.shape[1]):
        return np_grid.tolist()
        
    target_color = np_grid[r, c]
    if target_color == replacement_color:
        return np_grid.tolist()
        
    queue = [(r, c)]
    while queue:
        curr_r, curr_c = queue.pop(0)
        if np_grid[curr_r, curr_c] == target_color:
            np_grid[curr_r, curr_c] = replacement_color
            for nr, nc in [(curr_r-1, curr_c), (curr_r+1, curr_c), (curr_r, curr_c-1), (curr_r, curr_c+1)]:
                if 0 <= nr < np_grid.shape[0] and 0 <= nc < np_grid.shape[1]:
                    queue.append((nr, nc))
                    
    return np_grid.tolist()

def get_bounding_boxes(grid: List[List[int]], bg_color: int = 0) -> List[dict]:
    """Returns a list of bounding boxes for each contiguous object."""
    from scipy.ndimage import label, find_objects
    np_grid = np.array(grid)
    color_mask = (np_grid != bg_color).astype(int)
    labeled_array, _ = label(color_mask, structure=np.ones((3, 3)))
    slices = find_objects(labeled_array)
    
    boxes = []
    for i, slc in enumerate(slices):
        if slc is not None:
            obj_mask = (labeled_array == (i + 1))
            top, bottom = slc[0].start, slc[0].stop - 1
            left, right = slc[1].start, slc[1].stop - 1
            boxes.append({"top": top, "bottom": bottom, "left": left, "right": right, "color": int(np.median(np_grid[obj_mask]))})
    return boxes
