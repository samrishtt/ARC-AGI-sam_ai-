import os
import json
import logging
from typing import Dict, Any, List

from src.core.llm import OpenAIProvider, GroqProvider, GeminiProvider, AnthropicProvider
from src.csa.meta_controller import MetaController

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(data_dir: str, limit: int = None) -> List[Dict]:
    """Loads ARC JSON files from the given directory."""
    tasks = []
    files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    
    # Sort files to ensure deterministic order across runs
    files.sort()
    
    if limit:
        files = files[:limit]
        
    for f in files:
        with open(os.path.join(data_dir, f), 'r') as file:
            data = json.load(file)
            data['filename'] = f
            tasks.append(data)
    return tasks

def evaluate_csa(limit: int = 5):
    """
    Runs the MetaController on a subset of the ARC training dataset 
    and measures its accuracy in finding a solution that passes the test grid.
    """
    print("=" * 60)
    print("Cognitive Synthesis Architecture - Evaluator")
    print(f"Running evaluation on {limit} tasks.")
    print("=" * 60)

    # Boot up LLM (same priority as demo)
    if os.getenv("ANTHROPIC_API_KEY"):
        llm = AnthropicProvider()
        print("> Using Provider: Anthropic (Claude 4.6 Sonnet)")
    elif os.getenv("GROQ_API_KEY"):
        llm = GroqProvider()
        print("> Using Provider: Groq (Llama 3.1 8B)")
    elif os.getenv("GEMINI_API_KEY"):
        llm = GeminiProvider(model="gemini-2.0-flash")
        print("> Using Provider: Google Gemini")
    elif os.getenv("OPENAI_API_KEY"):
        llm = OpenAIProvider(model="gpt-4o")
        print("> Using Provider: OpenAI")
    else:
        print("ERROR: No API key found. Evaluator requires a real LLM.")
        return

    controller = MetaController(primary_llm=llm)

    # Load from the training subset
    train_dir = os.path.join("data", "training")
    tasks = load_data(train_dir, limit=limit)
    
    total_tasks = len(tasks)
    successful_tasks = 0

    for i, task in enumerate(tasks):
        print(f"\n[{i+1}/{total_tasks}] Currently processing task: {task['filename']}")
        
        try:
            # We pass the dict directly 
            result = controller.process_task(task)
            
            # The coding handler emits "Code validated against training pairs!" if successful,
            # and then also needs to print the test grid answer correctly.
            # However, `process_task` returning 'success' means the python script
            # executed without crashing and passed the training loop validation assertions.
            
            if result.get("status") == "success":
                # To truly verify if the result matches the hidden test grid output,
                # we need to parse the JSON string the code printed.
                output_str = result.get("output", "")
                print(f"  [SUCCESS] LLM found a transform function that satisfies training pairs!")
                
                # Check if it solved the test case too
                if "SUCCESS" in output_str:
                    lines = output_str.split('\n')
                    try:
                        # The last line should be the JSON dumped test array
                        llm_test_answer = json.loads(lines[-1])
                        actual_test_answer = task["test"][0]["output"]
                        
                        if llm_test_answer == actual_test_answer:
                            print(f"  [CORRECT] Output matches official test grid perfectly.")
                            successful_tasks += 1
                        else:
                            print(f"  [INCORRECT] Code passed training pairs but failed hidden test grid.")
                            
                    except json.JSONDecodeError:
                        print("  [ERROR] Could not parse test grid output from sandbox.")
            else:
                 print(f"  [FAILED] Reason: {result.get('output', 'Unknown error')}")
                 
        except Exception as e:
            print(f"  [EXCEPTION] Crash during task: {str(e)}")

    print("\n" + "=" * 60)
    print(f"EVALUATION COMPLETE")
    print(f"Total Tasks: {total_tasks}")
    print(f"Successful Solves: {successful_tasks}")
    print(f"Accuracy: {(successful_tasks / total_tasks * 100):.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    evaluate_csa(limit=2)
