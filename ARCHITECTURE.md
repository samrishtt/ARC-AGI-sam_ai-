# System Architecture - ARC-AGI God-Level Solver

## Overview

This is an advanced ARC-AGI solver that combines **intelligent pattern recognition**, **analogical reasoning**, **abstraction**, and **symbolic synthesis** to solve Abstract Reasoning Corpus tasks with high accuracy.

## Core Philosophy

Instead of brute-force search, we use a **hierarchy of reasoning strategies**:

1. **Pattern Matching** (Fast) - Direct recognition of known transformations
2. **Analogical Reasoning** - Use similar past solutions
3. **Decomposition** - Break complex tasks into simpler sub-problems
4. **Synthesis** - Compose primitives via guided search (last resort)

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     ARC-AGI SOLVER                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    │
│  │   Pattern   │───▶│  Reasoning   │───▶│   Prediction    │    │
│  │   Engine    │    │   Engine     │    │                 │    │
│  └─────────────┘    └──────────────┘    └─────────────────┘    │
│        │                   │                                    │
│        ▼                   ▼                                    │
│  ┌─────────────┐    ┌──────────────┐                           │
│  │  Feature    │    │   Solution   │                           │
│  │  Extractor  │    │   Memory     │                           │
│  └─────────────┘    └──────────────┘                           │
│        │                   │                                    │
│        ▼                   ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Enhanced DSL (50+ Primitives)              │   │
│  │  Geometric | Color | Object | Gravity | Morphological   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Pattern Engine (`src/arc/pattern_engine.py`)

The brain of pattern recognition:

- **Feature Extraction**: Extracts 15+ features from each grid
  - Shape, density, color histogram
  - Symmetry (horizontal, vertical, diagonal)
  - Object count, bounding box
  - Edge density, aspect ratio
  
- **Hypothesis Generation**: Creates ranked transformation hypotheses
  - Geometric (rotation, reflection, transpose)
  - Color (replacement, inversion, mapping)
  - Structural (crop, scale, tile)
  - Object (keep largest, filter, gravity)
  - Logical (AND, OR, XOR, counting)

- **Consistency Checking**: Validates hypotheses against ALL training examples

### 2. Reasoning Engine (`src/arc/reasoning.py`)

Orchestrates solving with multiple strategies:

- **Pattern Match**: Try direct pattern recognition first (fastest)
- **Analogy**: Check solution memory for similar problems
- **Decomposition**: Try common 2-step compositions
- **Synthesis**: Fall back to A* search over DSL

### 3. Enhanced DSL (`src/arc/enhanced_dsl.py`)

50+ primitives covering:

| Category | Primitives |
|----------|-----------|
| Geometric | rotate_cw, rotate_ccw, rotate_180, reflect_h, reflect_v, transpose, transpose_anti |
| Rolling | roll_up, roll_down, roll_left, roll_right |
| Cropping | crop, remove_border, pad_square |
| Scaling | scale_2x, scale_3x, scale_down_2x, tile_2x2, tile_3x3 |
| Color | invert, normalize, color_each |
| Object | keep_largest, keep_smallest, remove_largest |
| Gravity | gravity_down, gravity_up, gravity_left, gravity_right |
| Fill | fill_row, fill_column, fill_down, fill_up |
| Symmetry | sym_h, sym_v |
| Morphological | dilate, erode, outline, fill_holes |

### 4. Neural Pattern Matcher (`src/arc/neural_matcher.py`)

Lightweight pattern matching using handcrafted embeddings:

- **Pattern Signatures**: Compact vector representation of grids
- **Transform Signatures**: Capture input→output relationships
- **Similarity Search**: Find analogous patterns in memory
- **Rule Inference**: High-level understanding of transformations

### 5. Solver Runner (`src/arc/solver.py`)

Complete benchmark system with:

- **Progress Tracking**: Rich terminal UI
- **Logging**: JSON logs of all successes and failures
- **Visualization**: Color-coded grid display
- **Statistics**: Strategy breakdown, timing, accuracy

### 6. Data Loader (`src/arc/data_loader.py`)

Handles ARC data:

- Load from local JSON files
- Download from GitHub
- 10 built-in sample tasks
- Support for ARC Prize format

## Data Flow

```
Training Examples ──┐
                    │
                    ▼
        ┌──────────────────────┐
        │   Feature Extraction │
        │   (Pattern Engine)   │
        └──────────────────────┘
                    │
                    ▼
        ┌──────────────────────┐
        │  Hypothesis Generation│
        │  (Ranked by confidence)│
        └──────────────────────┘
                    │
                    ▼
        ┌──────────────────────┐
        │  Consistency Check   │
        │  (All examples must pass)│
        └──────────────────────┘
                    │
            ┌───────┴───────┐
            │               │
            ▼               ▼
      ┌──────────┐    ┌──────────┐
      │ Solution │    │ Fallback │
      │  Found   │    │ Strategies│
      └──────────┘    └──────────┘
            │               │
            ▼               ▼
        Prediction    Try Next Strategy
```

## Solving Strategy Hierarchy

1. **Pattern Match** (Confidence: 0.95-1.0)
   - Direct transformation detected
   - Works for all examples
   
2. **Analogical Reasoning** (Confidence: 0.8-0.9)
   - Similar task in memory
   - Apply stored transformation
   
3. **Decomposition** (Confidence: 0.75-0.85)
   - 2-step composition
   - Common patterns (rotate+reflect, crop+scale)
   
4. **Synthesis** (Confidence: 0.6-0.75)
   - A* search over DSL
   - Guided by heuristics

## Memory System

- **Solution Memory**: Stores successful transformations
- **Failure Memory**: Records what didn't work (for learning)
- **Macro Learning**: Successful solutions become new primitives

## Testing

### Run All Tests
```bash
python test_arc_comprehensive.py
```

### Test Specific Task
```bash
python test_arc_comprehensive.py --task rotate_90
```

### Quick Mode
```bash
python test_arc_comprehensive.py --quick
```

## Performance Metrics

- **Accuracy**: Percentage of tasks fully solved
- **Strategy Distribution**: Which strategies work best
- **Time per Task**: Average solving time
- **Failure Analysis**: Why tasks failed

## File Structure

```
arc_agi/
├── src/
│   ├── arc/                    # NEW: Main ARC solver
│   │   ├── __init__.py
│   │   ├── data_loader.py      # ARC data loading
│   │   ├── pattern_engine.py   # Pattern recognition
│   │   ├── reasoning.py        # Multi-strategy reasoning
│   │   ├── enhanced_dsl.py     # 50+ primitives
│   │   ├── neural_matcher.py   # Pattern matching
│   │   └── solver.py           # Main runner + logging
│   ├── core/                   # Core infrastructure
│   │   ├── agent.py
│   │   ├── searcher.py         # A* search
│   │   ├── reflector.py
│   │   └── llm.py
│   ├── dsl/                    # Original DSL
│   │   ├── primitives.py
│   │   └── utils.py
│   ├── memory/
│   │   └── store.py
│   └── tools/
│       ├── base.py
│       └── sandbox.py
├── tests/
├── logs/                       # Benchmark logs
├── data/                       # ARC datasets
└── test_arc_comprehensive.py   # Main test runner
```

## Future Enhancements

1. **Deep Learning Integration**: CNN for pattern embedding
2. **LLM Reasoning**: Use LLM for hypothesis generation
3. **Meta-Learning**: Learn which strategies work best
4. **Transfer Learning**: Apply solutions across similar tasks
5. **Active Learning**: Identify difficult patterns for improvement
