# ARC-AGI Core Module
"""
Advanced ARC-AGI Solver with intelligent pattern recognition
and reasoning capabilities.

Human-Level Accuracy Target: 85%+

Modules:
- pattern_engine: Feature extraction and hypothesis generation
- reasoning: Main reasoning orchestrator
- enhanced_dsl: 75+ DSL primitives for transformations
- object_detector: Object-centric perception
- advanced_patterns: Complex pattern detectors
- super_reasoning: Ultimate multi-strategy solver
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

__all__ = [
    # Core
    'ARCDataLoader', 
    'PatternEngine', 
    'ReasoningEngine',
    
    # Super Solver
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
