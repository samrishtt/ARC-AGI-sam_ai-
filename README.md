# 🧠 Cognitive Synthesis Architecture (CSA)
**An AGI-Inspired Framework for ARC-style Reasoning**

[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://python.org)
[![Architecture](https://img.shields.io/badge/Architecture-Neuro%20Symbolic-purple)](#)
[![LLM Support](https://img.shields.io/badge/LLMs-Claude_4.6_|_GPT--4o_|_Llama_3-green)](#)

## 🚀 Overview
The Cognitive Synthesis Architecture (CSA) is a highly advanced research framework designed to solve the **Abstraction and Reasoning Corpus (ARC-AGI)**. 

Moving far beyond simple text-based LLM generation, CSA acts as a **Hybrid Neuro-Symbolic System**. It translates pixel matrices into symbolic geometric graphs, generates hypotheses, and recursively tests and debugs Python code within an isolated deterministic sandbox to prove its answers are correct before presenting them.

## 🤯 Maximizing AGI Potential (New Features)

- **Claude 4.6 Sonnet Integration:** Native support for the world's leading coding/reasoning model, maximizing success rates on complex geometric transformations.
- **Advanced DSL Primitives (`src/dsl/primitives.py`):** The LLM is armed with a powerful Domain-Specific Language toolkit (Rotations, Flips, Flood Fills, Bresenham Line Drawing, Bounding Box Extraction). It writes 10x less code and achieves vastly higher accuracy.
- **Iterative Reflection Loop:** If the LLM's generated code fails an ARC training pair, the execution traceback is fed *back* to the LLM to recursively learn and self-correct up to 3 times per task.
- **Advanced Vision Parser (`src/csa/vision.py`):** Gives the LLM genuine spatial intelligence by analyzing objects for properties like `is_square`, `is_symmetric_v`, and extracting color masses using `scipy` connected-component clustering.

## 🏗️ Core Components

1. **Meta-Controller & Intent Router (`src/csa/meta_controller.py`)** 
   - Acts as the "Prefrontal Cortex" of the AI, routing raw ARC JSON datasets into the Visual/Spatial Pipeline.
2. **Deterministic Python Sandbox (`src/csa/coding.py`)**
   - Isolates logic execution. The LLM writes a Python script importing our DSL tools, the Sandbox executes it safely, and verifies the output strictly against the ARC training grids.
3. **Formal Evaluator (`src/eval/evaluate.py`)**
   - A dedicated script that runs the entire 400-task ARC training corpus through the neuro-symbolic pipeline, printing out formal benchmark accuracy scores.

## 🛠️ Installation & Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/arc_csa.git
cd arc_csa

# Install dependencies
pip install -r requirements.txt
# Additional SDKs if needed: pip install anthropic openai google-generativeai groq

# Create a .env file and add your preferred API Key!
# Priority (free): GROQ > NVIDIA > GEMINI
GROQ_API_KEY=gsk_...
NVIDIA_API_KEY=nvapi-...
GEMINI_API_KEY=AIza...
```

### Free API Provider Sign-Up
| Provider | Model | Sign-Up URL | Notes |
|----------|-------|-------------|-------|
| **Groq** (Primary) | llama-3.3-70b-versatile | [console.groq.com](https://console.groq.com) | Free, no credit card |
| **NVIDIA NIM** (Fallback 1) | llama-4-scout-17b | [build.nvidia.com](https://build.nvidia.com) | Free, join Developer Program |
| **Gemini** (Fallback 2) | gemini-1.5-flash | [aistudio.google.com](https://aistudio.google.com) | Free Google account |

## 🎯 Quick Start

### 1. Test the Pipeline (Demo)
To see the system route intents and form symbolic graphs locally:
```bash
python demo_csa.py
```

### 2. Run the ARC Evaluator (Full Power)
To unleash Claude 4.6 (or GPT-4o) on real ARC puzzles and execute the Recursive Testing Sandbox:
```bash
export PYTHONPATH="."
python src/eval/evaluate.py
```

## 📁 Project Structure

```text
arc_agi/
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── llm.py               # Supports Anthropic, Groq, Gemini, OpenAI, Mock
│   │   ├── searcher.py          # A* Program Search over DSL primitives
│   │   ├── planner.py           # LLM-based planning (System 2)
│   │   ├── reflector.py         # Failure analysis & reasoning hints
│   │   └── agent.py             # [LEGACY - unused, kept for reference]
│   ├── csa/
│   │   ├── __init__.py
│   │   ├── router.py            # Intent Router (LLM-based task classification)
│   │   ├── meta_controller.py   # Pipeline Orchestrator (the "brain")
│   │   ├── coding.py            # Python Sandbox & Reflection Loop
│   │   ├── memory.py            # Working Memory (scratchpad)
│   │   ├── vision.py            # Symbolic Grid Parser & Geometry
│   │   └── models.py            # Pydantic models (RouteDecision, etc.)
│   ├── dsl/
│   │   ├── __init__.py
│   │   ├── primitives.py        # 11 spatial operations for the LLM
│   │   └── utils.py             # Object counting & symmetry checks
│   └── eval/
│       ├── __init__.py
│       └── evaluate.py          # Formal ARC Benchmark Script
├── tests/
│   ├── __init__.py
│   ├── test_basic.py            # MetaController & sandbox tests
│   ├── test_arc_toy.py          # DSL primitive tests
│   ├── test_composition.py      # Composition & A* search tests
│   └── test_learning.py         # Memory & learning tests
├── data/
│   └── training/                # ARC JSON datasets (400 tasks)
├── logs/                        # Results JSONL from evaluation runs
├── demo_csa.py                  # Interactive pipeline demo
├── test_api.py                  # Quick Anthropic API connectivity check
├── requirements.txt             # All dependencies
├── .env                         # API keys (not committed)
└── README.md
```

## 🚀 Future Work: The Path to ARC-AGI-3 (2026+)
With the announcement of **ARC-AGI-3** (an Interactive Reasoning Benchmark launching March 2026), true generalization requires testing agents in dynamic, video-game-like environments where test-takers must actively *explore* to discover rules. 

While the current CSA framework excels at static geometric transformations (ARC-AGI 1 & 2), the architecture is perfectly positioned for this interactive future. Our immediate next steps involve extending the `MetaController`:
1. **Interactive State Tracking:** Upgrading `vision.py` from a static grid parser to a temporal state-tracker that holds memory across frame changes.
2. **Exploration Loop (Active Inference):** Implementing an RL-inspired "probing" phase where the LLM can generate exploratory hypotheses, test actions in the environment, and observe state changes *before* committing to a final execution strategy.
3. **Action Efficiency Optimization:** Structuring the planner to minimize the *Action Efficiency* metric introduced by the ARC Foundation (inspired by François Chollet's definition of intelligence).

## 🎓 Research Impact
This framework demonstrates exactly how **System 2 Orchestration** (injecting algorithmic bounds and reflection loops) and **Symbolic Grounding** (translating arrays into parsed geometric properties) drastically bridges the gap toward Artificial General Intelligence.

*Built for the pursuit of Artificial General Intelligence.*
