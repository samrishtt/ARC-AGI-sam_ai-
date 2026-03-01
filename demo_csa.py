import os
from dotenv import load_dotenv

# Loading project environment variables
load_dotenv()

# We import the LLM Providers
from src.core.llm import OpenAIProvider, GeminiProvider, GroqProvider, AnthropicProvider, MockLLMProvider, LLMResponse
from src.csa.meta_controller import MetaController
import json

def run_demo():
    print("==================================================")
    print("Cognitive Synthesis Architecture (CSA) - Phase 4")
    print("Full Pipeline: Router -> Sandbox -> Vision -> Hypothesis")
    print("==================================================\n")

    # Priority: Anthropic (paid, best) > Groq (free, high quota) > Gemini (free, daily quota) > OpenAI (paid) > Mock (offline)
    if os.getenv("ANTHROPIC_API_KEY"):
        print("Using Anthropic Claude 3.7 Sonnet (PAID tier - BEST)\n")
        llm = AnthropicProvider()
    elif os.getenv("GROQ_API_KEY"):
        print("Using Groq Llama 3.1 8B (FREE tier)\n")
        llm = GroqProvider()
    elif os.getenv("GEMINI_API_KEY"):
        print("Using Google Gemini 2.0 Flash (FREE tier)\n")
        llm = GeminiProvider(model="gemini-2.0-flash")
    elif os.getenv("OPENAI_API_KEY"):
        print("Using OpenAI GPT-4o (paid tier)\n")
        llm = OpenAIProvider(model="gpt-4o")
    else:
        print("WARNING: No API keys found.")
        print("Set GEMINI_API_KEY in .env for FREE testing (https://aistudio.google.com/apikey)")
        print("Falling back to Demo Mock.\n")

        class DemoMockLLM(MockLLMProvider):
            def __init__(self):
                self.coding_attempts = 0

            def generate(self, system_prompt: str, user_prompt: str):
                if "routing engine" in system_prompt:
                    if "Python" in user_prompt:
                        return LLMResponse(content='{"domain": "coding", "complexity": "medium", "reasoning": "User asked for python code.", "requires_tools": true}')
                    elif "JSON matrix" in user_prompt or "ARC" in user_prompt or '"train"' in user_prompt:
                        return LLMResponse(content='{"domain": "visual_spatial", "complexity": "high", "reasoning": "ARC style spatial grid mapping.", "requires_tools": true}')
                    elif "apples" in user_prompt:
                        return LLMResponse(content='{"domain": "math_logic", "complexity": "medium", "reasoning": "Simple algebraic math word problem.", "requires_tools": true}')
                    else:
                        return LLMResponse(content='{"domain": "conversational", "complexity": "low", "reasoning": "Greeting or simple chat.", "requires_tools": false}')

                if "pattern analyst" in system_prompt:
                    return LLMResponse(content="The transformation rule appears to be: every non-background colored cell is duplicated symmetrically across the vertical axis of the grid.")

                if "Code" in system_prompt or "coder" in system_prompt:
                    self.coding_attempts += 1
                    if self.coding_attempts % 2 == 1:
                        return LLMResponse(content="```python\nprint(undefined_variable)\n```")
                    else:
                        return LLMResponse(content="```python\nprint('Total apples: 16 (Calculated by Mock Resolver)')\n```")

                return super().generate(system_prompt, user_prompt)

        llm = DemoMockLLM()

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
        result = controller.process_task(json.dumps(arc_task))
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
