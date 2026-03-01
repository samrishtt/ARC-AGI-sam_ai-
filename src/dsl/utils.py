import numpy as np
from scipy.ndimage import label

def count_objects(grid: np.ndarray, bg_color: int = 0) -> int:
    """Counts the number of distinct contiguous non-background objects."""
    color_mask = (grid != bg_color).astype(int)
    _, num_features = label(color_mask, structure=np.ones((3, 3)))
    return num_features

def check_symmetry(grid: np.ndarray) -> str:
    """
    Returns a string describing the grid's symmetry.
    Returns 'both', 'horizontal', 'vertical', or 'none'.
    """
    h_sym = np.array_equal(grid, np.fliplr(grid))
    v_sym = np.array_equal(grid, np.flipud(grid))
    if h_sym and v_sym:
        return "both"
    elif h_sym:
        return "horizontal"
    elif v_sym:
        return "vertical"
    return "none"
