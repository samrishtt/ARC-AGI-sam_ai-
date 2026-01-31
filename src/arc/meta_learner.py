"""
Meta-Learning Module for ARC-AGI

Learns from solved tasks to improve future performance:
1. Stores successful solution patterns
2. Analyzes failures to avoid repeated mistakes
3. Learns which strategies work for which task types
4. Enables transfer learning between similar tasks

This is key to achieving human-level accuracy through
continuous improvement.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import hashlib
from datetime import datetime


@dataclass
class TaskProfile:
    """Profile of an ARC task based on its characteristics."""
    # Shape features
    input_shape: Tuple[int, int]
    output_shape: Tuple[int, int]
    shape_changes: bool
    shape_ratio: Tuple[float, float]
    
    # Color features
    num_input_colors: int
    num_output_colors: int
    colors_change: bool
    
    # Object features
    num_input_objects: int
    num_output_objects: int
    objects_change: bool
    
    # Symmetry features
    input_symmetric_h: bool
    input_symmetric_v: bool
    output_symmetric: bool
    
    # Other features
    is_tiling: bool
    is_scaling: bool
    
    def to_key(self) -> str:
        """Convert to a hashable key."""
        return f"{self.shape_ratio[0]:.1f}_{self.shape_ratio[1]:.1f}_{self.num_input_colors}_{self.num_output_colors}"
    
    def similarity(self, other: 'TaskProfile') -> float:
        """Compute similarity to another profile (0-1)."""
        score = 0.0
        
        # Shape similarity
        if self.shape_ratio == other.shape_ratio:
            score += 0.3
        elif abs(self.shape_ratio[0] - other.shape_ratio[0]) < 0.1:
            score += 0.15
        
        # Color similarity
        if self.num_input_colors == other.num_input_colors:
            score += 0.15
        if self.num_output_colors == other.num_output_colors:
            score += 0.15
        
        # Object similarity
        if self.objects_change == other.objects_change:
            score += 0.1
        
        # Symmetry similarity
        if self.input_symmetric_h == other.input_symmetric_h:
            score += 0.1
        if self.input_symmetric_v == other.input_symmetric_v:
            score += 0.1
        
        # Pattern similarity
        if self.is_tiling == other.is_tiling:
            score += 0.05
        if self.is_scaling == other.is_scaling:
            score += 0.05
        
        return min(score, 1.0)


@dataclass
class SolutionRecord:
    """Record of a successful solution."""
    task_id: str
    profile: TaskProfile
    strategy: str
    transform_chain: List[str]
    confidence: float
    solve_time_ms: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class FailureRecord:
    """Record of a failed attempt."""
    task_id: str
    profile: TaskProfile
    strategies_tried: List[str]
    hypotheses_tried: List[str]
    error_message: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class MetaLearner:
    """
    Meta-learning system for ARC-AGI.
    
    Learns from experience to improve solving:
    - Records successful solutions for similar tasks
    - Tracks which strategies work for which task types
    - Learns from failures to avoid dead ends
    - Enables transfer learning
    """
    
    def __init__(self, db_path: str = "db/meta_learner.json"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Solution database
        self.solutions: Dict[str, SolutionRecord] = {}
        
        # Failure database
        self.failures: Dict[str, FailureRecord] = {}
        
        # Strategy success rates by task profile type
        self.strategy_stats: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: {"success": 0, "fail": 0})
        )
        
        # Transform chain patterns
        self.chain_patterns: Dict[str, int] = defaultdict(int)
        
        # Load existing data
        self._load()
    
    def _load(self) -> None:
        """Load database from disk."""
        if self.db_path.exists():
            try:
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                
                # Reconstruct solutions
                for task_id, sol_data in data.get('solutions', {}).items():
                    profile = TaskProfile(**sol_data['profile'])
                    self.solutions[task_id] = SolutionRecord(
                        task_id=task_id,
                        profile=profile,
                        strategy=sol_data['strategy'],
                        transform_chain=sol_data['transform_chain'],
                        confidence=sol_data['confidence'],
                        solve_time_ms=sol_data['solve_time_ms'],
                        timestamp=sol_data.get('timestamp', '')
                    )
                
                # Load stats
                self.strategy_stats = defaultdict(
                    lambda: defaultdict(lambda: {"success": 0, "fail": 0}),
                    data.get('strategy_stats', {})
                )
                
                self.chain_patterns = defaultdict(int, data.get('chain_patterns', {}))
                
            except Exception as e:
                print(f"Warning: Could not load meta-learner data: {e}")
    
    def _save(self) -> None:
        """Save database to disk."""
        try:
            data = {
                'solutions': {
                    task_id: {
                        'profile': asdict(sol.profile),
                        'strategy': sol.strategy,
                        'transform_chain': sol.transform_chain,
                        'confidence': sol.confidence,
                        'solve_time_ms': sol.solve_time_ms,
                        'timestamp': sol.timestamp
                    }
                    for task_id, sol in self.solutions.items()
                },
                'strategy_stats': dict(self.strategy_stats),
                'chain_patterns': dict(self.chain_patterns),
            }
            
            with open(self.db_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not save meta-learner data: {e}")
    
    def profile_task(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> TaskProfile:
        """Create a profile for an ARC task."""
        if not examples:
            return TaskProfile(
                input_shape=(0, 0),
                output_shape=(0, 0),
                shape_changes=False,
                shape_ratio=(1.0, 1.0),
                num_input_colors=0,
                num_output_colors=0,
                colors_change=False,
                num_input_objects=0,
                num_output_objects=0,
                objects_change=False,
                input_symmetric_h=False,
                input_symmetric_v=False,
                output_symmetric=False,
                is_tiling=False,
                is_scaling=False
            )
        
        inp, out = examples[0]
        
        # Shape analysis
        shape_changes = inp.shape != out.shape
        shape_ratio = (
            out.shape[0] / inp.shape[0] if inp.shape[0] > 0 else 1.0,
            out.shape[1] / inp.shape[1] if inp.shape[1] > 0 else 1.0
        )
        
        # Color analysis
        in_colors = set(inp.flatten()) - {0}
        out_colors = set(out.flatten()) - {0}
        
        # Object counting (simple connected components)
        def count_objects(grid):
            from scipy import ndimage
            labeled, num = ndimage.label(grid > 0)
            return num
        
        try:
            in_objects = count_objects(inp)
            out_objects = count_objects(out)
        except:
            in_objects = 0
            out_objects = 0
        
        # Symmetry check
        in_sym_h = np.array_equal(inp, np.fliplr(inp))
        in_sym_v = np.array_equal(inp, np.flipud(inp))
        out_sym = np.array_equal(out, np.fliplr(out)) or np.array_equal(out, np.flipud(out))
        
        # Pattern checks
        is_tiling = (shape_ratio[0] > 1 and shape_ratio[0] == int(shape_ratio[0]) and
                     shape_ratio[1] > 1 and shape_ratio[1] == int(shape_ratio[1]))
        is_scaling = (shape_ratio[0] == shape_ratio[1] and shape_ratio[0] > 1)
        
        return TaskProfile(
            input_shape=inp.shape,
            output_shape=out.shape,
            shape_changes=shape_changes,
            shape_ratio=shape_ratio,
            num_input_colors=len(in_colors),
            num_output_colors=len(out_colors),
            colors_change=in_colors != out_colors,
            num_input_objects=in_objects,
            num_output_objects=out_objects,
            objects_change=in_objects != out_objects,
            input_symmetric_h=in_sym_h,
            input_symmetric_v=in_sym_v,
            output_symmetric=out_sym,
            is_tiling=is_tiling,
            is_scaling=is_scaling
        )
    
    def record_solution(
        self,
        task_id: str,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        strategy: str,
        transform_chain: List[str],
        confidence: float,
        solve_time_ms: float
    ) -> None:
        """Record a successful solution."""
        profile = self.profile_task(examples)
        
        record = SolutionRecord(
            task_id=task_id,
            profile=profile,
            strategy=strategy,
            transform_chain=transform_chain,
            confidence=confidence,
            solve_time_ms=solve_time_ms
        )
        
        self.solutions[task_id] = record
        
        # Update strategy stats
        profile_key = profile.to_key()
        self.strategy_stats[profile_key][strategy]["success"] += 1
        
        # Record chain pattern
        chain_key = " → ".join(transform_chain)
        self.chain_patterns[chain_key] += 1
        
        self._save()
    
    def record_failure(
        self,
        task_id: str,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        strategies_tried: List[str],
        hypotheses_tried: List[str],
        error: Optional[str] = None
    ) -> None:
        """Record a failed attempt."""
        profile = self.profile_task(examples)
        
        record = FailureRecord(
            task_id=task_id,
            profile=profile,
            strategies_tried=strategies_tried,
            hypotheses_tried=hypotheses_tried,
            error_message=error
        )
        
        self.failures[task_id] = record
        
        # Update strategy stats
        profile_key = profile.to_key()
        for strategy in strategies_tried:
            self.strategy_stats[profile_key][strategy]["fail"] += 1
        
        self._save()
    
    def find_similar_solutions(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        min_similarity: float = 0.5
    ) -> List[SolutionRecord]:
        """Find solutions to similar tasks."""
        profile = self.profile_task(examples)
        
        similar = []
        for task_id, solution in self.solutions.items():
            sim = profile.similarity(solution.profile)
            if sim >= min_similarity:
                similar.append((sim, solution))
        
        # Sort by similarity
        similar.sort(key=lambda x: x[0], reverse=True)
        
        return [s[1] for s in similar]
    
    def suggest_strategy(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[Tuple[str, float]]:
        """
        Suggest strategies ranked by past success rate.
        
        Returns list of (strategy_name, confidence) tuples.
        """
        profile = self.profile_task(examples)
        profile_key = profile.to_key()
        
        strategies = []
        
        # Check stats for this profile type
        if profile_key in self.strategy_stats:
            for strategy, counts in self.strategy_stats[profile_key].items():
                total = counts["success"] + counts["fail"]
                if total > 0:
                    success_rate = counts["success"] / total
                    strategies.append((strategy, success_rate))
        
        # Sort by success rate
        strategies.sort(key=lambda x: x[1], reverse=True)
        
        # If no stats, return default order
        if not strategies:
            strategies = [
                ("pattern_match", 0.5),
                ("composition_2", 0.4),
                ("advanced_pattern", 0.3),
                ("object_reasoning", 0.2),
            ]
        
        return strategies
    
    def suggest_transforms(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[str]:
        """
        Suggest transform chains to try based on similar tasks.
        """
        similar = self.find_similar_solutions(examples)
        
        chains = []
        for sol in similar[:5]:  # Top 5 similar
            chains.append(sol.transform_chain)
        
        # Also add most common chains
        common_chains = sorted(
            self.chain_patterns.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        for chain_str, _ in common_chains:
            chain = chain_str.split(" → ")
            if chain not in chains:
                chains.append(chain)
        
        return chains
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get meta-learning statistics."""
        return {
            "total_solutions": len(self.solutions),
            "total_failures": len(self.failures),
            "unique_profiles": len(self.strategy_stats),
            "unique_chains": len(self.chain_patterns),
            "top_chains": dict(sorted(
                self.chain_patterns.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]),
            "strategy_breakdown": {
                profile: {
                    s: f"{c['success']}/{c['success']+c['fail']}"
                    for s, c in strats.items()
                }
                for profile, strats in list(self.strategy_stats.items())[:5]
            }
        }


class TransferLearner:
    """
    Transfer learning between ARC tasks.
    
    Uses solutions from similar tasks to solve new ones.
    """
    
    def __init__(self, meta_learner: MetaLearner):
        self.meta_learner = meta_learner
    
    def transfer_solution(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        dsl: Dict[str, callable]
    ) -> Optional[Tuple[np.ndarray, str]]:
        """
        Try to transfer a solution from a similar task.
        
        Returns (prediction, description) if successful.
        """
        similar = self.meta_learner.find_similar_solutions(examples, min_similarity=0.6)
        
        for solution in similar:
            chain = solution.transform_chain
            
            try:
                # Build and test the transform
                def build_transform(chain):
                    def composed(grid):
                        result = grid
                        for name in chain:
                            if name in dsl:
                                result = dsl[name](result)
                            else:
                                return None
                        return result
                    return composed
                
                transform = build_transform(chain)
                
                # Verify on examples
                all_match = True
                for inp, out in examples:
                    result = transform(inp)
                    if result is None or not np.array_equal(result, out):
                        all_match = False
                        break
                
                if all_match:
                    prediction = transform(test_input)
                    desc = f"Transfer from {solution.task_id}: {' → '.join(chain)}"
                    return prediction, desc
                    
            except Exception:
                continue
        
        return None


# Convenience function
def learn_from_solution(
    task_id: str,
    examples: List[Tuple[np.ndarray, np.ndarray]],
    strategy: str,
    transform_chain: List[str],
    db_path: str = "db/meta_learner.json"
) -> None:
    """Record a solution for meta-learning."""
    learner = MetaLearner(db_path)
    learner.record_solution(
        task_id=task_id,
        examples=examples,
        strategy=strategy,
        transform_chain=transform_chain,
        confidence=0.9,
        solve_time_ms=0
    )
