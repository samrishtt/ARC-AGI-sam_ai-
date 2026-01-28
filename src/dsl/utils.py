
import numpy as np
from typing import Dict, List, Set, Tuple

def get_color_counts(grid: np.ndarray) -> Dict[int, int]:
    unique, counts = np.unique(grid, return_counts=True)
    return dict(zip(unique, counts))

def count_objects(grid: np.ndarray, connectivity: int = 4) -> int:
    """
    Count connected components of non-zero/non-background pixels.
    Assumes 0 is background.
    """
    visited = np.zeros_like(grid, dtype=bool)
    rows, cols = grid.shape
    count = 0
    
    def get_neighbors(r, c):
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if connectivity == 8:
            deltas += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dr, dc in deltas:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                yield nr, nc

    def bfs(start_r, start_c):
        queue = [(start_r, start_c)]
        visited[start_r, start_c] = True
        while queue:
            r, c = queue.pop(0)
            for nr, nc in get_neighbors(r, c):
                if not visited[nr, nc] and grid[nr, nc] == grid[start_r, start_c]: # Same color object
                    visited[nr, nc] = True
                    queue.append((nr, nc))

    for r in range(rows):
        for c in range(cols):
            if not visited[r, c] and grid[r, c] != 0:
                count += 1
                bfs(r, c)
                
    return count

def check_symmetry(grid: np.ndarray) -> Dict[str, bool]:
    sym_h = np.array_equal(grid, grid[::-1, :])
    sym_v = np.array_equal(grid, grid[:, ::-1])
    return {"horizontal": sym_h, "vertical": sym_v}

def get_largest_object_mask(grid: np.ndarray, connectivity: int = 4) -> np.ndarray:
    """
    Returns a boolean mask for the largest connected component (excluding background 0).
    """
    visited = np.zeros_like(grid, dtype=bool)
    rows, cols = grid.shape
    max_size = 0
    max_mask = np.zeros_like(grid, dtype=bool)
    
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
                # Start new component
                component_mask = np.zeros_like(grid, dtype=bool)
                stack = [(r, c)]
                visited[r, c] = True
                component_mask[r, c] = True
                size = 0
                color = grid[r, c]
                
                while stack:
                    curr_r, curr_c = stack.pop()
                    size += 1
                    for nr, nc in get_neighbors(curr_r, curr_c):
                        if not visited[nr, nc] and grid[nr, nc] == color:
                            visited[nr, nc] = True
                            component_mask[nr, nc] = True
                            stack.append((nr, nc))
                
                if size > max_size:
                    max_size = size
                    max_mask = component_mask
                    
    return max_mask
