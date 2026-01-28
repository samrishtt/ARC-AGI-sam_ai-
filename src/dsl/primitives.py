import numpy as np

# Basic DSL Primitives for ARC Grids
# Grids are represented as 2D numpy arrays

def rotate_cw(grid: np.ndarray) -> np.ndarray:
    """Rotate grid 90 degrees clockwise."""
    return np.rot90(grid, k=-1)

def rotate_ccw(grid: np.ndarray) -> np.ndarray:
    """Rotate grid 90 degrees counter-clockwise."""
    return np.rot90(grid, k=1)

def reflect_horizontal(grid: np.ndarray) -> np.ndarray:
    """Flip grid horizontally (left-right)."""
    return np.fliplr(grid)

def reflect_vertical(grid: np.ndarray) -> np.ndarray:
    """Flip grid vertically (up-down)."""
    return np.flipud(grid)

def crop_non_zero(grid: np.ndarray) -> np.ndarray:
    """Crop the grid to the smallest bounding box containing non-zero elements."""
    coords = np.argwhere(grid)
    if coords.size == 0:
        return grid
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    return grid[x_min:x_max+1, y_min:y_max+1]

def get_object_mask(grid: np.ndarray, color: int) -> np.ndarray:
    """Return a boolean mask for a specific color."""
    return grid == color

from .utils import get_largest_object_mask, count_objects

def keep_largest_object(grid: np.ndarray) -> np.ndarray:
    mask = get_largest_object_mask(grid)
    result = np.zeros_like(grid)
    result[mask] = grid[mask]
    return result

def color_objects(grid: np.ndarray) -> np.ndarray:
    """Assigns a unique color to each connected object. Fallback if scipy fails."""
    try:
        from scipy.ndimage import label
        labeled, n = label(grid > 0)
        # Map labels to some colors (1-9)
        result = np.zeros_like(grid)
        for i in range(1, n + 1):
            result[labeled == i] = (i % 9) + 1
        return result
    except Exception:
        # Fallback: just return the grid if scipy is broken
        return grid

dsl_registry = {
    "rotate_cw": rotate_cw,
    "rotate_ccw": rotate_ccw,
    "reflect_horizontal": reflect_horizontal,
    "reflect_vertical": reflect_vertical,
    "crop": crop_non_zero,
    "transpose": lambda g: np.transpose(g),
    "roll_up": lambda g: np.roll(g, -1, axis=0),
    "roll_down": lambda g: np.roll(g, 1, axis=0),
    "roll_left": lambda g: np.roll(g, -1, axis=1),
    "roll_right": lambda g: np.roll(g, 1, axis=1),
    "keep_largest": keep_largest_object,
    "color_objects": color_objects
}
