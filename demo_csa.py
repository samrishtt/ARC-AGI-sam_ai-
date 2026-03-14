import os
from dotenv import load_dotenv

# Loading project environment variables
load_dotenv()

# We import the LLM Providers
from src.core.llm import get_best_provider
from src.csa.meta_controller import MetaController
import json

def run_demo():
    print("==================================================")
    print("Cognitive Synthesis Architecture (CSA) - Phase 4")
    print("Full Pipeline: Router -> Sandbox -> Vision -> Hypothesis")
    print("==================================================\n")

    # Smart fallback: Claude (paid) → Gemini (free) → Groq (free) → Mock (offline)
    llm = get_best_provider(prefer_paid=True)

    # Instantiate the Meta-Controller
    controller = MetaController(primary_llm=llm)

    # ---- Test Case 1: Conversational ----
    print("\n--- Test Case 1: Conversational ---")
    try:
        result = controller.process_task("Hey! How are you doing today?")
        _print_result(result)
    except Exception as e:
        print(f"Error: {e}")

    # ---- Test Case 2: Math/Logic with Sandbox ----
    print("\n--- Test Case 2: Math/Logic (Sandbox) ---")
    try:
        result = controller.process_task(
            "If Mary had 3 apples and gives 1 to John who then multiplies "
            "his apples by the square root of 16, how many do they have in total?"
        )
        _print_result(result)
    except Exception as e:
        print(f"Error: {e}")

    # ---- Test Case 3: Coding with Reflection ----
    print("\n--- Test Case 3: Coding (Reflection Loop) ---")
    try:
        result = controller.process_task(
            "Can you write a Python script that implements a simple BFS algorithm?"
        )
        _print_result(result)
    except Exception as e:
        print(f"Error: {e}")

    # ---- Test Case 4: Visual-Spatial (ARC-style with training pairs) ----
    print("\n--- Test Case 4: Visual-Spatial (ARC Pipeline) ---")
    arc_task = {
        "train": [
            {
                "input":  [[0, 0, 1], [0, 0, 0], [0, 0, 0]],
                "output": [[0, 0, 1], [0, 0, 1], [0, 0, 1]]
            },
            {
                "input":  [[0, 2, 0], [0, 0, 0], [0, 0, 0]],
                "output": [[0, 2, 0], [0, 2, 0], [0, 2, 0]]
            }
        ],
        "test": [
            {
                "input":  [[3, 0, 0], [0, 0, 0], [0, 0, 0]],
                "output": [[3, 0, 0], [3, 0, 0], [3, 0, 0]]
            }
        ]
    }
    try:
        result = controller.process_task(arc_task)
        _print_result(result)
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "="*50)
    print("Demo complete! Check logs/results.jsonl for routing history.")
    print("="*50)


def _print_result(result):
    print(f"> Status: {result['status']}")
    print(f"> Pipeline: {result['pipeline']}")
    print(f"> Domain: {result['decision']['domain']}")
    print(f"> Complexity: {result['decision']['complexity']}")
    print(f"> Tools Required: {result['decision']['requires_tools']}")
    output_snippet = result['output'][:150].replace('\n', ' ')
    print(f"> Output: {output_snippet}...")


if __name__ == "__main__":
    run_demo()
