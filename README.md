# 🧠 ARC-AGI Ultra Solver

**A self-improving, ultra-powerful solver for the Abstraction and Reasoning Corpus (ARC)**

[![Accuracy Target](https://img.shields.io/badge/Accuracy%20Target-80%25+-brightgreen)](https://github.com)
[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://python.org)
[![Self-Improving](https://img.shields.io/badge/Self--Improving-Yes-purple)](https://github.com)

## 🚀 Overview

This project implements a state-of-the-art ARC-AGI solver that combines:

- **100+ DSL Primitives**: Comprehensive transformation library
- **Self-Improving Memory**: Learns from solved tasks to improve future performance
- **Ensemble Voting**: Multiple solvers vote for maximum accuracy
- **Hierarchical Strategies**: From fast pattern matching to deep program synthesis
- **Meta-Learning**: Transfers knowledge between similar tasks

## 📊 Performance

| Version | Accuracy | Tasks Solved | Strategy |
|---------|----------|--------------|----------|
| v1.0 | 6.5% | 26/400 | Basic DSL |
| v1.5 | 11.0% | 44/400 | Pattern Engine |
| **v2.0** | **Target: 80%** | **320/400** | **Ultra Solver + Ensemble** |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ARC-AGI ULTRA SOLVER v2.0                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │  Ensemble       │───▶│  Ultra Solver   │                    │
│  │  Voting         │    │  (Self-Improving)│                   │
│  └─────────────────┘    └─────────────────┘                    │
│          │                      │                               │
│          ▼                      ▼                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Strategy Hierarchy                          │   │
│  │  1. Identity    5. Templates     9.  Analogical         │   │
│  │  2. Geometric   6. Subdivision   10. Composition-2      │   │
│  │  3. Color Map   7. Mask/Overlay  11. Composition-3      │   │
│  │  4. Objects     8. Counting      12. Deep Search        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            │                                    │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           100+ Primitive DSL Operations                  │   │
│  │  Geometric │ Color │ Object │ Gravity │ Morphological    │   │
│  │  Cropping  │ Scale │ Tile   │ Fill    │ Subdivision      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            │                                    │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Self-Improving Memory                       │   │
│  │  • Solution records for similar tasks                    │   │
│  │  • Strategy success rates                                │   │
│  │  • Learned macro transforms                              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/arc_agi.git
cd arc_agi

# Install dependencies
pip install -r requirements.txt

# Download ARC dataset (see DOWNLOAD_DATASET.md)
```

## 🎯 Quick Start

### Run a Quick Test
```bash
python run_benchmark.py --quick
```

### Run Full Benchmark
```bash
python run_benchmark.py --data data/training
```

### Run with Ensemble Voting (Slower, More Accurate)
```bash
python run_benchmark.py --data data/training
```

### Run Fast Mode (No Ensemble)
```bash
python run_benchmark.py --data data/training --no-ensemble
```

### Save Results
```bash
python run_benchmark.py --save
```

## 📁 Project Structure

```
arc_agi/
├── src/
│   └── arc/                     # Main ARC solver package
│       ├── ultra_solver.py      # ⭐ NEW: Self-improving solver
│       ├── ensemble_solver.py   # ⭐ NEW: Ensemble voting system
│       ├── pattern_engine.py    # Feature extraction
│       ├── enhanced_dsl.py      # 100+ primitives
│       ├── object_detector.py   # Object perception
│       ├── advanced_patterns.py # Complex pattern detection
│       ├── meta_learner.py      # Transfer learning
│       └── ...
├── data/
│   ├── training/                # ARC training tasks (400)
│   └── evaluation/              # ARC evaluation tasks (400)
├── db/                          # Self-improving memory database
├── logs/                        # Benchmark results
├── run_benchmark.py             # ⭐ NEW: Main benchmark runner
├── requirements.txt
└── README.md
```

## 🧪 Usage in Code

```python
from src.arc import UltraSolver, EnsembleSolver
import numpy as np

# Create solver
solver = UltraSolver(enable_learning=True)

# Or use ensemble for maximum accuracy
# solver = EnsembleSolver()

# Define training examples as (input, output) pairs
train_examples = [
    (np.array([[1, 0], [0, 1]]), np.array([[1, 1], [1, 1]])),
    # ... more examples
]

# Test input
test_input = np.array([[2, 0], [0, 2]])

# Solve
result = solver.solve(train_examples, test_input, task_id="my_task")

if result.success:
    print(f"Solution found! Strategy: {result.strategy}")
    print(f"Prediction:\n{result.prediction}")
else:
    print("No solution found")
```

## 🔧 Key Components

### UltraSolver
The main solver with self-improving capabilities:
- 100+ DSL primitives
- Hierarchical strategy search
- Learns from solved tasks
- Composition search (2-step and 3-step)

### EnsembleSolver
Combines multiple solving approaches with voting:
- Confidence-weighted voting
- Specialized solvers (geometric, color, object, etc.)
- Higher accuracy through consensus

### PrimitiveDSL
Comprehensive transformation library:
- **Geometric**: Rotate, flip, transpose
- **Color**: Map, swap, normalize
- **Object**: Keep/remove, filter, color
- **Gravity**: Push pixels in any direction
- **Morphological**: Dilate, erode, outline, fill
- **Grid**: Subdivide, extract, overlay
- **Logical**: XOR, AND, OR operations

### SelfImprovingMemory
Persistent learning system:
- Records successful solutions
- Tracks strategy success rates
- Enables analogical reasoning

## 📈 Strategies (Priority Order)

1. **Identity** - Check if output equals input
2. **Direct Pattern** - Single DSL primitive
3. **Color Transform** - Color mapping/swapping
4. **Geometric** - Rotations, reflections
5. **Template** - Pattern-based scaling
6. **Subdivision** - Grid extraction, XOR/AND/OR
7. **Mask/Overlay** - Morphological operations
8. **Object Filter** - Keep/remove objects
9. **Counting** - Count-based outputs
10. **Composition-2** - Two-step transforms
11. **Analogical** - Use past solutions
12. **Composition-3** - Three-step transforms
13. **Program Synthesis** - Advanced composition
14. **Deep Search** - Exhaustive search

## 🎓 For MIT Application

This project demonstrates:
- **Abstraction**: Understanding abstract patterns from examples
- **Reasoning**: Multi-step logical inference
- **Learning**: Self-improving from experience
- **Engineering**: Clean, modular architecture
- **Innovation**: Novel ensemble and memory approaches

## 📝 License

MIT License - Feel free to use and modify!

## 🙏 Acknowledgments

- [ARC Challenge](https://github.com/fchollet/ARC) by François Chollet
- Inspired by human reasoning and abstraction capabilities
