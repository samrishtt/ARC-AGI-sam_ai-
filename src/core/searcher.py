"""
Searcher module — A* Program Search over DSL primitives + Transformation Library.

TASK 6: Added TransformationLibrary class for library learning.
Records successful transformations and provides past solutions as candidates.
"""

import heapq
import os
import json
import time
import numpy as np
from typing import List, Callable, Optional, Dict, Tuple, Set
from ..dsl.primitives import dsl_registry
from ..dsl.utils import count_objects, check_symmetry


# ═══════════════════════════════════════════════════════
# TASK 6 — TRANSFORMATION LIBRARY (Library Learning)
# ═══════════════════════════════════════════════════════

class TransformationLibrary:
    """
    Records successful transformations and provides past solutions
    as candidates for similar new tasks.
    
    On every successful task: call record_success()
    At start of each new task: call get_candidates() to prepend
    past solutions as examples in the LLM prompt.
    """
    def __init__(self, path: str = "data/learned_transforms.json"):
        self.path = path
        self.library = self._load()

    def _load(self) -> List[Dict]:
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return []
        return []

    def record_success(self, task_id: str, code: str, category: str):
        """Record a successful transformation solution."""
        self.library.append({
            "task_id": task_id,
            "category": category,
            "code": code,
            "timestamp": time.time()
        })
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.path) if os.path.dirname(self.path) else "data", exist_ok=True)
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(self.library, f, indent=2)

    def get_candidates(self, category: str, top_k: int = 3) -> List[Dict]:
        """Get the most recent successful solutions for a given category."""
        matches = [e for e in self.library if e["category"] == category]
        return matches[-top_k:]


# ═══════════════════════════════════════════════════════
# ORIGINAL A* PROGRAM SEARCH
# ═══════════════════════════════════════════════════════

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
                try:
                    name = prog
                    ctx = {**self.base_primitives, "np": np}
                    func = eval(f"lambda x: {prog}", ctx)
                    self.primitives[name] = func
                except:
                    continue

    def heuristic(self, current_grids: List[np.ndarray], target_grids: List[np.ndarray]) -> float:
        """Estimate distance to target across all examples."""
        total_cost = 0.0
        for current, target in zip(current_grids, target_grids):
            if current.shape != target.shape:
                 rows_diff = abs(current.shape[0] - target.shape[0])
                 cols_diff = abs(current.shape[1] - target.shape[1])
                 total_cost += 1000.0 + (rows_diff * 10) + (cols_diff * 10)
            else:
                 pixel_diff = np.count_nonzero(current != target)
                 total_cost += pixel_diff
                 
            try:
                c_objs = count_objects(current)
                t_objs = count_objects(target)
                total_cost += abs(c_objs - t_objs) * 50
            except:
                pass
            
            try:
                c_sym = check_symmetry(current)
                t_sym = check_symmetry(target)
                if c_sym != t_sym:
                    total_cost += 50
            except:
                pass
                
            if np.all(current == 0) and not np.all(target == 0):
                total_cost += 500
                
        return total_cost

    def solve(self, examples: List[Tuple[np.ndarray, np.ndarray]], max_iterations: int = 10000) -> Optional[str]:
        """
        A* Search for a sequence of DSL primitives that maps inputs -> outputs for ALL examples.
        """
        inputs = [np.array(ex[0]) for ex in examples]
        targets = [np.array(ex[1]) for ex in examples]
        
        start_h = self.heuristic(inputs, targets)
        counter = 0
        iterations = 0
        
        queue = []
        heapq.heappush(queue, (start_h, counter, 0, "x", inputs))
        
        visited_hashes = set()
        visited_hashes.add(tuple(g.tobytes() for g in inputs))
        
        while queue and iterations < max_iterations:
            iterations += 1
            f, _, g, path, current_grids = heapq.heappop(queue)
            
            if all(np.array_equal(c, t) for c, t in zip(current_grids, targets)):
                return path

            if g >= self.max_depth:
                continue
                
            for name, func in self.primitives.items():
                try:
                    new_grids = []
                    valid_op = True
                    for grid in current_grids:
                        res = np.array(func(grid.tolist()))
                        if res is None:
                            valid_op = False
                            break
                        new_grids.append(res)
                    
                    if not valid_op:
                        continue
                        
                    grids_hash = tuple(ng.tobytes() for ng in new_grids)
                    if grids_hash in visited_hashes:
                        continue
                    visited_hashes.add(grids_hash)
                    
                    h = self.heuristic(new_grids, targets)
                    new_g = g + 1
                    new_f = new_g + (self.heuristic_weight * h)
                    
                    new_path = f"{name}({path})"
                    
                    counter += 1
                    heapq.heappush(queue, (new_f, counter, new_g, new_path, new_grids))
                    
                except Exception:
                    continue
                    
        return None
