# 🧠 ARC-AGI: Human-Level Abstract Reasoning Solver

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Accuracy Target](https://img.shields.io/badge/Accuracy-85%25+-green.svg)](#)
[![MIT Application Project](https://img.shields.io/badge/MIT-Application_Project-red.svg)](#)

> **A state-of-the-art solver for the Abstraction and Reasoning Corpus (ARC-AGI) benchmark, 
> targeting human-level accuracy (85%+) through multi-strategy reasoning.**

## 🎯 Mission

Build the most advanced ARC-AGI solver that achieves **human-level accuracy** through:
1. **Object-Centric Perception**: Understanding grids as collections of objects with relationships
2. **Multi-Strategy Reasoning**: Hierarchical approach from simple to complex strategies
3. **Extended DSL**: 75+ transformation primitives covering diverse patterns
4. **LLM-Guided Search**: Using language models for hypothesis generation
5. **Meta-Learning**: Learning from past solutions to improve future performance

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ARC-AGI HUMAN-LEVEL SOLVER                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                     PERCEPTION LAYER                              │  │
│  │   ObjectDetector │ GridAnalyzer │ SymmetryDetector │ Patterns   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                     REASONING LAYER                               │  │
│  │   LLMReasoner │ HypothesisGenerator │ ProgramSynthesizer         │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                     EXECUTION LAYER                               │  │
│  │   DSL Engine (75+ primitives) │ SearchEngine │ Verification      │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                     LEARNING LAYER                                │  │
│  │   MetaLearner │ TransferLearning │ FailureAnalysis               │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 📊 Solving Strategies (Hierarchical)

| Strategy | Description | Speed | Accuracy |
|----------|-------------|-------|----------|
| `EXACT_MATCH` | Identity transform detection | ⚡ Fastest | Very High |
| `PATTERN_MATCH` | Single DSL primitive | ⚡ Fast | High |
| `ADVANCED_PATTERN` | Template, counting, conditional | 🔄 Medium | High |
| `OBJECT_REASONING` | Object-centric transforms | 🔄 Medium | Medium |
| `COMPOSITION_2` | 2-step DSL composition | 🔄 Medium | Medium |
| `COMPOSITION_3` | 3-step DSL composition | 🐢 Slow | Medium |
| `ANALOGICAL` | Transfer from similar tasks | 🔄 Medium | Variable |
| `DEEP_SEARCH` | A* search up to 5 steps | 🐢 Slowest | High |

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ARC-AGI.git
cd ARC-AGI

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

```python
from src.arc import SuperReasoningEngine, solve_task
import numpy as np

# Create example task
train = [
    (np.array([[1, 0], [0, 0]]), np.array([[0, 1], [0, 0]])),  # 90° rotation
    (np.array([[2, 0], [0, 0]]), np.array([[0, 2], [0, 0]])),
]
test_input = np.array([[3, 0], [0, 0]])

# Solve!
engine = SuperReasoningEngine()
result = engine.solve(train, test_input, "test_task")

print(f"Strategy: {result.strategy.value}")
print(f"Success: {result.success}")
print(f"Prediction:\n{result.prediction}")
```

## 📈 Running Benchmarks

```bash
# Quick test
python quick_test.py

# Full benchmark
python benchmark_ultimate.py --data data/training --verbose

# Run on evaluation set
python benchmark_ultimate.py --data data/evaluation --save

# Run complete benchmark (training + evaluation)
python benchmark_ultimate.py --full
```

## 🧩 Core Modules

| Module | Description | LOC |
|--------|-------------|-----|
| `enhanced_dsl.py` | 75+ transformation primitives | 800+ |
| `object_detector.py` | Object-centric perception | 560+ |
| `advanced_patterns.py` | Template, counting, conditional patterns | 580+ |
| `super_reasoning.py` | Multi-strategy orchestrator | 500+ |
| `llm_reasoner.py` | LLM-guided hypothesis generation | 400+ |
| `meta_learner.py` | Transfer learning & strategy optimization | 500+ |

## 🔬 DSL Primitives (75+)

### Geometric Transforms
- `rotate_cw`, `rotate_ccw`, `rotate_180`
- `reflect_h`, `reflect_v`, `transpose`

### Object Operations
- `keep_largest`, `keep_smallest`, `remove_largest`
- `color_each`, `extract_object`

### Grid Manipulation
- `crop`, `scale_2x`, `scale_3x`, `tile_2x2`
- `mirror_right`, `mirror_4way`, `split_h`, `split_v`

### Fill Operations
- `gravity_down`, `fill_row`, `fill_column`
- `diagonal_fill`, `connect_same_color`

### Color Operations
- `invert`, `swap_colors`, `normalize`
- `shift_colors_up`, `shift_colors_down`

### Morphological
- `dilate`, `erode`, `outline`, `fill_holes`

## 📋 Implementation Plan

See [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) for detailed roadmap.

**All 5 phases completed:**
- ✅ Phase 1: Enhanced Perception
- ✅ Phase 2: Extended DSL (75+ primitives)
- ✅ Phase 3: Intelligent Reasoning
- ✅ Phase 4: Advanced Patterns
- ✅ Phase 5: Meta-Learning

## 🎓 For MIT Application

This project demonstrates:
- **Deep Understanding of AGI Challenges**: ARC-AGI is a key benchmark for AI reasoning
- **Advanced Software Architecture**: Modular, extensible, well-documented
- **Machine Learning Innovation**: Multi-strategy approach with meta-learning
- **Research-Quality Code**: Type hints, comprehensive tests, clean structure

---

**Target: Human-Level Accuracy (85%+)**

*Built with 💡 by Samrisht for MIT Application*

