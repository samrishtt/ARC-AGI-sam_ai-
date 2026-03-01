
import heapq
import numpy as np
from typing import List, Callable, Optional, Dict, Tuple, Set
from ..dsl.primitives import dsl_registry
from ..dsl.utils import count_objects, check_symmetry

class ProgramSearch:
    def __init__(self, primitives: Dict[str, Callable] = dsl_registry, max_depth: int = 4, heuristic_weight: float = 1.0):
        self.base_primitives = primitives
        self.max_depth = max_depth
        self.heuristic_weight = heuristic_weight
        self.primitives = dict(primitives) # Copy

    def load_macros(self, programs: List[str]):
        """
        Add learned programs from memory as new search primitives.
        Programs should be strings like "rotate_cw(reflect_h(x))"
        """
        for prog in programs:
            if not prog.startswith("x") and "(" in prog:
                # Wrap it in a lambda that can be executed
                # Caution: eval is used here against 'safe' programs from memory
                try:
                    name = prog
                    # Create a callable that takes 'x' and applies the prog
                    # We need a context for the eval
                    ctx = {**self.base_primitives, "np": np}
                    func = eval(f"lambda x: {prog}", ctx)
                    self.primitives[name] = func
                except:
                    continue

    def heuristic(self, current_grids: List[np.ndarray], target_grids: List[np.ndarray]) -> float:
        """
        Estimate distance to target across all examples.
        """
        total_cost = 0.0
        for current, target in zip(current_grids, target_grids):
            # 1. Check shape mismatch (High penalty)
            if current.shape != target.shape:
                 rows_diff = abs(current.shape[0] - target.shape[0])
                 cols_diff = abs(current.shape[1] - target.shape[1])
                 total_cost += 1000.0 + (rows_diff * 10) + (cols_diff * 10)
            else:
                 # 2. Pixel-wise difference (for same shape)
                 pixel_diff = np.count_nonzero(current != target)
                 total_cost += pixel_diff
                 
            # 3. Object-count mismatch (Medium penalty)
            # This helps guide towards solutions that preserve/match structure
            try:
                c_objs = count_objects(current)
                t_objs = count_objects(target)
                total_cost += abs(c_objs - t_objs) * 50
            except:
                pass # Fallback if count_objects fails
            
            # 4. Symmetry mismatch (Medium penalty)
            try:
                c_sym = check_symmetry(current)
                t_sym = check_symmetry(target)
                if c_sym != t_sym:
                    total_cost += 50
            except:
                pass
                
            # 5. Trivial solution check (e.g. all black) - high penalty if target is not all black
            if np.all(current == 0) and not np.all(target == 0):
                total_cost += 500
                
        return total_cost

    def solve(self, examples: List[Tuple[np.ndarray, np.ndarray]], max_iterations: int = 10000) -> Optional[str]:
        """
        A* Search for a sequence of DSL primitives that maps inputs -> outputs for ALL examples.
        
        Args:
            examples: List of (input, output) pairs
            max_iterations: Maximum iterations before giving up (default 10000)
        """
        inputs = [np.array(ex[0]) for ex in examples]
        targets = [np.array(ex[1]) for ex in examples]
        
        start_h = self.heuristic(inputs, targets)
        counter = 0
        iterations = 0
        
        # (f, count, g, path, current_grids)
        # We store list of grids. Not hashable, so we rely on counter for heap order.
        queue = []
        heapq.heappush(queue, (start_h, counter, 0, "x", inputs))
        
        # Visited: Tuple of bytes for all grids
        visited_hashes = set()
        visited_hashes.add(tuple(g.tobytes() for g in inputs))
        
        while queue and iterations < max_iterations:
            iterations += 1
            f, _, g, path, current_grids = heapq.heappop(queue)
            
            # Check goal: All grids match targets
            if all(np.array_equal(c, t) for c, t in zip(current_grids, targets)):
                return path

            # Pruning
            if g >= self.max_depth:
                continue
                
            # Expand
            for name, func in self.primitives.items():
                try:
                    new_grids = []
                    valid_op = True
                    for grid in current_grids:
                        res = np.array(func(grid.tolist()))
                        if res is None: # Should not happen with valid primitives
                            valid_op = False
                            break
                        new_grids.append(res)
                    
                    if not valid_op:
                        continue
                        
                    # Dedup
                    grids_hash = tuple(ng.tobytes() for ng in new_grids)
                    if grids_hash in visited_hashes:
                        continue
                    visited_hashes.add(grids_hash)
                    
                    # Calculate scores
                    h = self.heuristic(new_grids, targets)
                    new_g = g + 1
                    new_f = new_g + (self.heuristic_weight * h)
                    
                    new_path = f"{name}({path})"
                    
                    counter += 1
                    heapq.heappush(queue, (new_f, counter, new_g, new_path, new_grids))
                    
                except Exception:
                    continue
                    
        return None

