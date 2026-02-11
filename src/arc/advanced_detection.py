"""
Advanced Pattern Detection for ARC-AGI

Additional advanced patterns to boost accuracy:
- Line filling patterns
- Boolean overlay patterns
- Flood fill patterns
- Repeat/tile detection
- Block copying
- Marker-based patterns
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable, Set
from collections import Counter, defaultdict


class AdvancedPatterns:
    """Advanced pattern detection for higher accuracy."""
    
    @staticmethod
    def detect_line_fill(examples: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[Callable]:
        """Detect horizontal/vertical line filling patterns."""
        if not examples:
            return None
        
        # Check horizontal fill
        def fill_horizontal(grid: np.ndarray) -> np.ndarray:
            result = grid.copy()
            h, w = grid.shape
            for r in range(h):
                colors = grid[r][grid[r] != 0]
                if len(colors) > 0:
                    result[r, :] = colors[0]
            return result
        
        if AdvancedPatterns._verify(examples, fill_horizontal):
            return fill_horizontal
        
        # Check vertical fill
        def fill_vertical(grid: np.ndarray) -> np.ndarray:
            result = grid.copy()
            h, w = grid.shape
            for c in range(w):
                colors = grid[:, c][grid[:, c] != 0]
                if len(colors) > 0:
                    result[:, c] = colors[0]
            return result
        
        if AdvancedPatterns._verify(examples, fill_vertical):
            return fill_vertical
        
        # Check fill between markers
        def fill_between_horizontal(grid: np.ndarray) -> np.ndarray:
            result = grid.copy()
            h, w = grid.shape
            for r in range(h):
                nonzero = np.where(grid[r] != 0)[0]
                if len(nonzero) >= 2:
                    start, end = nonzero[0], nonzero[-1]
                    color = grid[r, start]
                    result[r, start:end+1] = color
            return result
        
        if AdvancedPatterns._verify(examples, fill_between_horizontal):
            return fill_between_horizontal
        
        # Check fill between markers vertically
        def fill_between_vertical(grid: np.ndarray) -> np.ndarray:
            result = grid.copy()
            h, w = grid.shape
            for c in range(w):
                nonzero = np.where(grid[:, c] != 0)[0]
                if len(nonzero) >= 2:
                    start, end = nonzero[0], nonzero[-1]
                    color = grid[start, c]
                    result[start:end+1, c] = color
            return result
        
        if AdvancedPatterns._verify(examples, fill_between_vertical):
            return fill_between_vertical
        
        return None
    
    @staticmethod
    def detect_boolean_ops(examples: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[Callable]:
        """Detect XOR, AND, OR operations between grid halves or objects."""
        if not examples:
            return None
        
        inp, out = examples[0]
        h, w = inp.shape
        
        # Vertical XOR
        if w % 2 == 0 and out.shape == (h, w // 2):
            def xor_v(g):
                w = g.shape[1]
                left = g[:, :w//2]
                right = g[:, w//2:]
                return ((left > 0) ^ (right > 0)).astype(g.dtype)
            if AdvancedPatterns._verify(examples, xor_v):
                return xor_v
            
            # Try keeping colors
            def xor_v_colored(g):
                w = g.shape[1]
                left = g[:, :w//2]
                right = g[:, w//2:]
                result = np.zeros_like(left)
                for r in range(left.shape[0]):
                    for c in range(left.shape[1]):
                        l, ri = left[r, c], right[r, c]
                        if (l > 0) != (ri > 0):
                            result[r, c] = l if l > 0 else ri
                return result
            if AdvancedPatterns._verify(examples, xor_v_colored):
                return xor_v_colored
        
        # Horizontal XOR
        if h % 2 == 0 and out.shape == (h // 2, w):
            def xor_h(g):
                h = g.shape[0]
                top = g[:h//2, :]
                bottom = g[h//2:, :]
                return ((top > 0) ^ (bottom > 0)).astype(g.dtype)
            if AdvancedPatterns._verify(examples, xor_h):
                return xor_h
        
        # Vertical AND
        if w % 2 == 0 and out.shape == (h, w // 2):
            def and_v(g):
                w = g.shape[1]
                left = g[:, :w//2]
                right = g[:, w//2:]
                result = np.zeros_like(left)
                for r in range(left.shape[0]):
                    for c in range(left.shape[1]):
                        if left[r, c] > 0 and right[r, c] > 0:
                            result[r, c] = left[r, c]
                return result
            if AdvancedPatterns._verify(examples, and_v):
                return and_v
        
        # Horizontal AND
        if h % 2 == 0 and out.shape == (h // 2, w):
            def and_h(g):
                h = g.shape[0]
                top = g[:h//2, :]
                bottom = g[h//2:, :]
                result = np.zeros_like(top)
                for r in range(top.shape[0]):
                    for c in range(top.shape[1]):
                        if top[r, c] > 0 and bottom[r, c] > 0:
                            result[r, c] = top[r, c]
                return result
            if AdvancedPatterns._verify(examples, and_h):
                return and_h
        
        return None
    
    @staticmethod
    def detect_flood_fill(examples: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[Callable]:
        """Detect flood fill patterns."""
        if not examples:
            return None
        
        # Simple flood fill enclosed regions
        def flood_fill_enclosed(grid: np.ndarray) -> np.ndarray:
            result = grid.copy()
            h, w = grid.shape
            
            # Find background pixels reachable from edge
            edge_reachable = set()
            queue = []
            
            for r in range(h):
                if grid[r, 0] == 0:
                    queue.append((r, 0))
                if grid[r, w-1] == 0:
                    queue.append((r, w-1))
            for c in range(w):
                if grid[0, c] == 0:
                    queue.append((0, c))
                if grid[h-1, c] == 0:
                    queue.append((h-1, c))
            
            while queue:
                r, c = queue.pop()
                if (r, c) in edge_reachable:
                    continue
                if not (0 <= r < h and 0 <= c < w):
                    continue
                if grid[r, c] != 0:
                    continue
                edge_reachable.add((r, c))
                queue.extend([(r-1, c), (r+1, c), (r, c-1), (r, c+1)])
            
            # Fill non-reachable zeros with surrounding color
            for r in range(h):
                for c in range(w):
                    if grid[r, c] == 0 and (r, c) not in edge_reachable:
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != 0:
                                result[r, c] = grid[nr, nc]
                                break
            
            return result
        
        if AdvancedPatterns._verify(examples, flood_fill_enclosed):
            return flood_fill_enclosed
        
        return None
    
    @staticmethod
    def detect_block_copy(examples: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[Callable]:
        """Detect block copying patterns."""
        if not examples:
            return None
        
        inp, out = examples[0]
        in_h, in_w = inp.shape
        out_h, out_w = out.shape
        
        # Check tiling
        if out_h % in_h == 0 and out_w % in_w == 0:
            rep_h = out_h // in_h
            rep_w = out_w // in_w
            
            def tile(g, rh=rep_h, rw=rep_w):
                return np.tile(g, (rh, rw))
            
            if AdvancedPatterns._verify(examples, tile):
                return tile
        
        # Check cropped content tiling
        def get_content_bbox(g):
            nonzero = np.argwhere(g != 0)
            if len(nonzero) == 0:
                return None
            r_min, c_min = nonzero.min(axis=0)
            r_max, c_max = nonzero.max(axis=0)
            return r_min, c_min, r_max, c_max
        
        bbox = get_content_bbox(inp)
        if bbox:
            r_min, c_min, r_max, c_max = bbox
            content = inp[r_min:r_max+1, c_min:c_max+1]
            
            if out_h % content.shape[0] == 0 and out_w % content.shape[1] == 0:
                rep_h = out_h // content.shape[0]
                rep_w = out_w // content.shape[1]
                
                def tile_content(g, rh=rep_h, rw=rep_w):
                    nonzero = np.argwhere(g != 0)
                    if len(nonzero) == 0:
                        return g
                    r_min, c_min = nonzero.min(axis=0)
                    r_max, c_max = nonzero.max(axis=0)
                    content = g[r_min:r_max+1, c_min:c_max+1]
                    return np.tile(content, (rh, rw))
                
                if AdvancedPatterns._verify(examples, tile_content):
                    return tile_content
        
        return None
    
    @staticmethod
    def detect_marker_based(examples: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[Callable]:
        """Detect marker-based transformation patterns."""
        if not examples:
            return None
        
        inp, out = examples[0]
        
        # Detect if a single unique-colored pixel is used as a marker
        colors = Counter(inp.flatten())
        single_colors = [c for c, cnt in colors.items() if cnt == 1 and c != 0]
        
        for marker_color in single_colors:
            # Find marker position
            marker_pos = tuple(np.argwhere(inp == marker_color)[0])
            
            # Try using marker as rotation point
            # Try outputting just the shape at marker
            pass
        
        return None
    
    @staticmethod
    def detect_upscale_pattern(examples: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[Callable]:
        """Detect upscaling patterns beyond simple scaling."""
        if not examples:
            return None
        
        inp, out = examples[0]
        in_h, in_w = inp.shape
        out_h, out_w = out.shape
        
        # Simple integer scaling
        for scale in [2, 3, 4, 5]:
            if out_h == in_h * scale and out_w == in_w * scale:
                def upscale(g, s=scale):
                    return np.repeat(np.repeat(g, s, axis=0), s, axis=1)
                if AdvancedPatterns._verify(examples, upscale):
                    return upscale
        
        return None
    
    @staticmethod
    def detect_color_count_transform(examples: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[Callable]:
        """Detect transforms based on color counts."""
        if not examples:
            return None
        
        inp, out = examples[0]
        
        # Keep only most frequent color
        def keep_most_frequent(grid: np.ndarray) -> np.ndarray:
            colors = grid[grid != 0]
            if len(colors) == 0:
                return np.zeros_like(grid)
            most_common = Counter(colors).most_common(1)[0][0]
            return np.where(grid == most_common, most_common, 0)
        
        if AdvancedPatterns._verify(examples, keep_most_frequent):
            return keep_most_frequent
        
        # Keep only least frequent color
        def keep_least_frequent(grid: np.ndarray) -> np.ndarray:
            colors = grid[grid != 0]
            if len(colors) == 0:
                return np.zeros_like(grid)
            least_common = Counter(colors).most_common()[-1][0]
            return np.where(grid == least_common, least_common, 0)
        
        if AdvancedPatterns._verify(examples, keep_least_frequent):
            return keep_least_frequent
        
        return None
    
    @staticmethod
    def detect_border_transform(examples: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[Callable]:
        """Detect border-related transforms."""
        if not examples:
            return None
        
        # Add border
        inp, out = examples[0]
        in_h, in_w = inp.shape
        out_h, out_w = out.shape
        
        if out_h == in_h + 2 and out_w == in_w + 2:
            for color in range(1, 10):
                def add_border(g, c=color):
                    return np.pad(g, 1, constant_values=c)
                if AdvancedPatterns._verify(examples, add_border):
                    return add_border
        
        # Remove border
        if out_h == in_h - 2 and out_w == in_w - 2 and in_h > 2 and in_w > 2:
            def remove_border(g):
                return g[1:-1, 1:-1]
            if AdvancedPatterns._verify(examples, remove_border):
                return remove_border
        
        return None
    
    @staticmethod
    def detect_reflection_completion(examples: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[Callable]:
        """Detect reflection/symmetry completion patterns."""
        if not examples:
            return None
        
        inp, out = examples[0]
        
        # Horizontal mirror to right
        if out.shape[1] == inp.shape[1] * 2:
            def mirror_right(g):
                return np.hstack([g, np.fliplr(g)])
            if AdvancedPatterns._verify(examples, mirror_right):
                return mirror_right
        
        # Vertical mirror down
        if out.shape[0] == inp.shape[0] * 2:
            def mirror_down(g):
                return np.vstack([g, np.flipud(g)])
            if AdvancedPatterns._verify(examples, mirror_down):
                return mirror_down
        
        # Four-way mirror
        if out.shape[0] == inp.shape[0] * 2 and out.shape[1] == inp.shape[1] * 2:
            def mirror_4way(g):
                top = np.hstack([g, np.fliplr(g)])
                bottom = np.flipud(top)
                return np.vstack([top, bottom])
            if AdvancedPatterns._verify(examples, mirror_4way):
                return mirror_4way
        
        return None
    
    @staticmethod
    def detect_gravity(examples: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[Callable]:
        """Detect gravity/pushing patterns."""
        if not examples:
            return None
        
        # Gravity down
        def gravity_down(grid: np.ndarray) -> np.ndarray:
            result = np.zeros_like(grid)
            h, w = grid.shape
            for c in range(w):
                col_pixels = [(grid[r, c], r) for r in range(h) if grid[r, c] != 0]
                for i, (color, _) in enumerate(reversed(col_pixels)):
                    result[h - 1 - i, c] = color
            return result
        
        if AdvancedPatterns._verify(examples, gravity_down):
            return gravity_down
        
        # Gravity up
        def gravity_up(grid: np.ndarray) -> np.ndarray:
            result = np.zeros_like(grid)
            h, w = grid.shape
            for c in range(w):
                col_pixels = [(grid[r, c], r) for r in range(h) if grid[r, c] != 0]
                for i, (color, _) in enumerate(col_pixels):
                    result[i, c] = color
            return result
        
        if AdvancedPatterns._verify(examples, gravity_up):
            return gravity_up
        
        # Gravity left
        def gravity_left(grid: np.ndarray) -> np.ndarray:
            result = np.zeros_like(grid)
            h, w = grid.shape
            for r in range(h):
                row_pixels = [(grid[r, c], c) for c in range(w) if grid[r, c] != 0]
                for i, (color, _) in enumerate(row_pixels):
                    result[r, i] = color
            return result
        
        if AdvancedPatterns._verify(examples, gravity_left):
            return gravity_left
        
        # Gravity right
        def gravity_right(grid: np.ndarray) -> np.ndarray:
            result = np.zeros_like(grid)
            h, w = grid.shape
            for r in range(h):
                row_pixels = [(grid[r, c], c) for c in range(w) if grid[r, c] != 0]
                for i, (color, _) in enumerate(reversed(row_pixels)):
                    result[r, w - 1 - i] = color
            return result
        
        if AdvancedPatterns._verify(examples, gravity_right):
            return gravity_right
        
        return None
    
    @staticmethod
    def detect_diagonal_patterns(examples: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[Callable]:
        """Detect diagonal-based patterns."""
        if not examples:
            return None
        
        # Diagonal fill
        def fill_diagonal(grid: np.ndarray) -> np.ndarray:
            result = grid.copy()
            h, w = grid.shape
            for r in range(h):
                for c in range(w):
                    if grid[r, c] != 0:
                        color = grid[r, c]
                        # Fill along main diagonal
                        for d in range(1, max(h, w)):
                            if r + d < h and c + d < w and result[r+d, c+d] == 0:
                                result[r+d, c+d] = color
                            if r - d >= 0 and c - d >= 0 and result[r-d, c-d] == 0:
                                result[r-d, c-d] = color
            return result
        
        if AdvancedPatterns._verify(examples, fill_diagonal):
            return fill_diagonal
        
        return None
    
    @staticmethod
    def _verify(examples: List[Tuple[np.ndarray, np.ndarray]], transform: Callable) -> bool:
        """Verify transform works on all examples."""
        try:
            for inp, expected in examples:
                result = transform(inp)
                if not np.array_equal(result, expected):
                    return False
            return True
        except:
            return False
    
    @classmethod
    def get_all_detectors(cls) -> List[Callable]:
        """Get all pattern detectors."""
        return [
            cls.detect_line_fill,
            cls.detect_boolean_ops,
            cls.detect_flood_fill,
            cls.detect_block_copy,
            cls.detect_upscale_pattern,
            cls.detect_color_count_transform,
            cls.detect_border_transform,
            cls.detect_reflection_completion,
            cls.detect_gravity,
            cls.detect_diagonal_patterns,
        ]
