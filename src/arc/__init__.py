# ARC-AGI Core Module
"""
Advanced ARC-AGI Solver with intelligent pattern recognition
and self-improving reasoning capabilities.

Target Accuracy: 80%+

Core Solvers (NEW):
- ultra_solver: Self-improving solver with 100+ primitives
- ensemble_solver: Voting-based ensemble for maximum accuracy

Modules:
- pattern_engine: Feature extraction and hypothesis generation
- reasoning: Main reasoning orchestrator
- enhanced_dsl: 75+ DSL primitives for transformations
- object_detector: Object-centric perception
- advanced_patterns: Complex pattern detectors
- super_reasoning: Multi-strategy solver
- llm_reasoner: LLM-guided hypothesis generation
- meta_learner: Transfer learning from past solutions
"""

from .data_loader import ARCDataLoader
from .pattern_engine import PatternEngine
from .reasoning import ReasoningEngine
from .enhanced_dsl import enhanced_dsl_registry
from .object_detector import ObjectDetector, ArcObject, GridAnalysis, analyze_grid
from .advanced_patterns import AdvancedPatternDetector, detect_patterns
from .super_reasoning import SuperReasoningEngine, solve_task
from .llm_reasoner import LLMReasoner, LLMGuidedSearch, get_llm_hypotheses
from .meta_learner import MetaLearner, TransferLearner, learn_from_solution
from .color_ops import COLOR_OPS, infer_color_mapping
from .grid_ops import SUBDIVISION_OPS, detect_subdivision_pattern

# NEW: Ultra-powerful solvers
from .ultra_solver import UltraSolver, SolverStrategy, SolveResult, PrimitiveDSL, GridAnalyzer
from .ensemble_solver import EnsembleSolver, IterativeRefinementSolver, VotingResult

__all__ = [
    # NEW: Primary Solvers (Use these for best accuracy)
    'UltraSolver',
    'EnsembleSolver',
    'IterativeRefinementSolver',
    'SolverStrategy',
    'SolveResult',
    'VotingResult',
    'PrimitiveDSL',
    'GridAnalyzer',
    
    # Core
    'ARCDataLoader', 
    'PatternEngine', 
    'ReasoningEngine',
    
    # Legacy Solver
    'SuperReasoningEngine',
    'solve_task',
    
    # Perception
    'ObjectDetector',
    'ArcObject',
    'GridAnalysis',
    'analyze_grid',
    
    # Pattern Detection
    'AdvancedPatternDetector',
    'detect_patterns',
    
    # DSL
    'enhanced_dsl_registry',
    'COLOR_OPS',
    'SUBDIVISION_OPS',
    
    # LLM
    'LLMReasoner',
    'LLMGuidedSearch',
    'get_llm_hypotheses',
    
    # Meta-Learning
    'MetaLearner',
    'TransferLearner',
    'learn_from_solution',
]
