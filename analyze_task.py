
import json
import numpy as np
from src.arc.imaginarium import Dreamer
from src.arc.dsl_registry import PrimitiveDSL
from rich.console import Console

console = Console()

def load_task(path):
    with open(path, 'r') as f:
        data = json.load(f)
    train = []
    for pair in data['train']:
        train.append((np.array(pair['input']), np.array(pair['output'])))
    test_in = np.array(data['test'][0]['input'])
    return train, test_in

def analyze(task_id):
    path = f'data/training/{task_id}.json'
    train, test_in = load_task(path)
    
    console.print(f"[bold]Analyzing Task {task_id}[/bold]")
    console.print(f"Train examples: {len(train)}")
    for i, (inp, out) in enumerate(train):
        console.print(f"Ex {i}: In {inp.shape} -> Out {out.shape}")
        
    # Try Dreamer
    dreamer = Dreamer(beam_width=100, max_depth=5)
    
    # Test Hypothesis: Fractal Copy
    console.print("\n[bold]Testing Hypothesis: Fractal Copy[/bold]")
    correct = True
    for inp, out in train:
        # H1: np.kron((inp>0), inp)
        mask = (inp > 0).astype(int)
        pred = np.kron(mask, inp)
        if not np.array_equal(pred, out):
            console.print(f"Failed on example. Pred shape {pred.shape}, Out shape {out.shape}")
            correct = False
            break
    
    if correct:
        console.print("[green]Hypothesis Correct! It is fractal copy.[/green]")
    else:
        console.print("[red]Hypothesis Incorrect.[/red]")

    console.print("\n[bold]Running Dreamer...[/bold]")
    result = dreamer.solve(train, timeout_ms=10000) # Give it 10s
    
    if result:
        console.print(f"[green]Solved![/green]")
        console.print(f"Program: {result['program']}")
    else:
        console.print("[red]Failed to solve.[/red]")

if __name__ == "__main__":
    analyze("007bbfb7")
