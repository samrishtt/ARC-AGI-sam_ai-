"""
DSL Registry for ARC-AGI

Central registry for all DSL primitives to avoid circular imports.
"""

from typing import Dict, Callable
import numpy as np
from .enhanced_dsl import *

class PrimitiveDSL:
    """Registry of all available DSL primitives."""
    
    @staticmethod
    def get_all_primitives() -> Dict[str, Callable]:
        """Get all available primitives as a dictionary."""
        return {
            # === FRACTAL ===
            'fractal_copy': fractal_copy,
            
            # === GEOMETRIC ===
            'identity': identity,
            'rotate_cw': rotate_cw,
            'rotate_ccw': rotate_ccw,
            'rotate_180': rotate_180,
            'flip_h': reflect_horizontal,
            'flip_v': reflect_vertical,
            'transpose': transpose,
            'transpose_anti': transpose_anti,
            
            # === ROLLING ===
            'roll_up': roll_up,
            'roll_down': roll_down,
            'roll_left': roll_left,
            'roll_right': roll_right,
            
            # === SCALING ===
            'scale_2x': scale_2x,
            'scale_3x': scale_3x,
            'scale_down_2x': scale_down_2x,
            'tile_2x2': tile_2x2,
            'tile_3x3': tile_3x3,
            
            # === CROPPING/PADDING ===
            'crop_to_content': crop_to_content,
            'remove_border': remove_border,
            'pad_to_square': pad_to_square,
            'add_border': add_border,
            
            # === COLOR ===
            'invert_binary': invert_binary,
            'normalize_colors': normalize_colors,
            'replace_color': replace_color,
            'swap_colors': swap_colors,
            'shift_colors_up': shift_colors_up,
            'shift_colors_down': shift_colors_down,
            'color_to_1': color_to_1,
            'color_to_2': color_to_2,
            
            # === MORPHOLOGICAL ===
            'dilate': dilate,
            'erode': erode,
            'outline': outline,
            'fill_holes': fill_holes,
            
            # === OBJECT ===
            'keep_largest': keep_largest_object,
            'keep_smallest': keep_smallest_object,
            'remove_largest': remove_largest_object,
            'keep_largest_object': keep_largest_object,
            'keep_smallest_object': keep_smallest_object,
            'remove_largest_object': remove_largest_object,
            'color_each_object': color_each_object,
            
            # === GRAVITY ===
            'gravity_down': gravity_down,
            'gravity_up': gravity_up,
            'gravity_left': gravity_left,
            'gravity_right': gravity_right,
            
            # === FILL ===
            'fill_row': fill_row,
            'fill_column': fill_column,
            'fill_down_from_top': fill_down_from_top,
            'flood_fill_exterior': flood_fill_exterior,
            'diagonal_fill': diagonal_fill,
            'anti_diagonal_fill': anti_diagonal_fill,
            'extend_lines': extend_lines,
            
            # === REGION ===
            'top_row': top_row,
            'bottom_row': bottom_row,
            'left_col': left_col,
            'right_col': right_col,
            'main_diagonal': main_diagonal,
            
            # === MIRROR ===
            'mirror_right': mirror_right,
            'mirror_down': mirror_down,
            'mirror_4way': mirror_4way,
            
            # === SPECIAL ===
            'sample_to_1x1': sample_to_1x1,
            'count_nonzero': count_nonzero_to_grid,
            'unique_colors': unique_colors_count,
            'most_common_color': most_common_color,
        }
