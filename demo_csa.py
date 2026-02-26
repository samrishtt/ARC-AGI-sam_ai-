import os
from dotenv import load_dotenv

# Loading project environment variables
load_dotenv()

# We import the LLM Provider from the existing core we found!
from src.core.llm import OpenAIProvider, MockLLMProvider, LLMResponse
from src.csa.meta_controller import MetaController
import json

def run_demo():
    print("==================================================")
    print("Cognitive Synthesis Architecture (CSA) - Phase 1")
    print("Testing the Meta-Controller & Routing System")
    print("==================================================\n")

    # Ensure API Key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY is not set in your .env file.")
        print("Falling back to specialized Demo Mock for routing.\n")
        
        # We build a Quick Mock just for demonstrating the Router logic
        class DemoMockLLM(MockLLMProvider):
            def generate(self, system_prompt: str, user_prompt: str):
                if "routing engine" in system_prompt:
                    # Provide an appropriate JSON based on keywords in user prompt
                    if "Python" in user_prompt:
                        return LLMResponse(content='{"domain": "coding", "complexity": "medium", "reasoning": "User asked for python code.", "requires_tools": true}')
                    elif "JSON matrix" in user_prompt:
                        return LLMResponse(content='{"domain": "visual_spatial", "complexity": "high", "reasoning": "ARC style spatial grid mapping.", "requires_tools": false}')
                    elif "apples" in user_prompt:
                        return LLMResponse(content='{"domain": "math_logic", "complexity": "medium", "reasoning": "Simple algebraic math word problem.", "requires_tools": true}')
                    else:
                        return LLMResponse(content='{"domain": "conversational", "complexity": "low", "reasoning": "Greeting or simple chat.", "requires_tools": false}')
                
                # Default mock for other calls (like the actual handlers)
                return super().generate(system_prompt, user_prompt)
        
        llm = DemoMockLLM()
    else:
        llm = OpenAIProvider(model="gpt-4-turbo")
        
    # Instantiate our brand new Meta-Controller
    controller = MetaController(primary_llm=llm)

    test_queries = [
        "Hey! How are you doing today?",
        "If Mary had 3 apples and gives 1 to John who then multiplies his apples by the square root of 16, how many do they have in total?",
        "Can you write a Python script that implements a simple BFS algorithm?",
        "Look at this JSON matrix of colors. If I map blue to red, what is the output shape?"
    ]

    for i, query in enumerate(test_queries):
        print(f"\n--- Test Case {i+1} ---")
        try:
            result = controller.process_task(query)
            
            print(f"> Handled by: {result['pipeline']} pipeline")
            print(f"> Task Domain Classified as: {result['decision']['domain']}")
            print(f"> Task Complexity: {result['decision']['complexity']}")
            print(f"> Is it requiring tools?: {result['decision']['requires_tools']}")
            print(f"> Assistant Output Snippet: {result['output'][:100]}...")
            
        except Exception as e:
            print(f"Error processing test case: {getattr(e, 'message', str(e))}")

if __name__ == "__main__":
    run_demo()
