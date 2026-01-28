# Arc-AGI: Advanced Reasoning & Codegen Agent

## Mission
To build a robust, iteratively self-improving Artificial General Intelligence focused on:
1.  **Deep Reasoning**: Breaking down complex problems into solvable steps.
2.  **Reliable Execution**: Generating and executing code safely to verify assumptions.
3.  **Iterative Refinement**: Learning from mistakes and improving outputs over time.
4.  **Universal Interface**: Capable of interacting through CLI, Web, or API.

## The "Strong Approach" Philosophy
- **Type Safety**: strict type checking with MyPy/Pydantic from day one.
- **Test-Driven**: Comprehensive unit and integration tests.
- **Modularity**: decoupled components (Brain, Tools, Memory) for easy upgrades.
- **Observability**: Detailed logging of thought processes and actions.

## Architecture
See [ARCHITECTURE.md](./ARCHITECTURE.md) for detailed system design.

## Setup
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```
