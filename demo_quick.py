"""
Quick Demo - Shows the ARC-AGI Solver in action
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

def print_grid(grid, title=""):
    """Print a grid with colors."""
    colors = ['black', 'blue', 'red', 'green', 'yellow', 'grey', 'magenta', 'orange1', 'cyan', 'bright_red']
    lines = []
    for row in grid:
        chars = []
        for c in row:
            color = colors[min(int(c), 9)]
            chars.append(f"[{color}]#[/{color}]" if c > 0 else "[dim].[/dim]")
        lines.append(' '.join(chars))
    console.print(Panel('\n'.join(lines), title=title, border_style="cyan"))

def main():
    console.print(Panel.fit(
        "[bold green]ARC-AGI GOD-LEVEL SOLVER DEMO[/bold green]\n"
        "[dim]Intelligent Pattern Recognition in Action[/dim]",
        border_style="green"
    ))
    
    # Import our solver
    from src.arc.reasoning import ReasoningEngine
    from src.arc.enhanced_dsl import enhanced_dsl_registry
    from src.arc.pattern_engine import PatternEngine
    
    engine = ReasoningEngine(dsl_primitives=enhanced_dsl_registry)
    pattern_engine = PatternEngine()
    
    # Demo tasks with expected outputs
    demo_tasks = [
        {
            "name": "Horizontal Reflection",
            "train": [
                (np.array([[1,2,0],[0,0,0],[0,0,0]]), np.array([[0,2,1],[0,0,0],[0,0,0]])),
            ],
            "test_input": np.array([[5,6,0],[0,0,0],[7,0,0]]),
            "test_expected": np.array([[0,6,5],[0,0,0],[0,0,7]])
        },
        {
            "name": "Crop to Content",
            "train": [
                (np.array([[0,0,0,0],[0,1,1,0],[0,1,1,0],[0,0,0,0]]), np.array([[1,1],[1,1]])),
            ],
            "test_input": np.array([[0,0,0],[0,5,0],[0,0,0]]),
            "test_expected": np.array([[5]])
        },
        {
            "name": "Color Replacement",
            "train": [
                (np.array([[1,1,0],[1,0,0],[0,0,0]]), np.array([[2,2,0],[2,0,0],[0,0,0]])),
            ],
            "test_input": np.array([[1,0,1],[0,0,0],[1,0,1]]),
            "test_expected": np.array([[2,0,2],[0,0,0],[2,0,2]])
        },
        {
            "name": "Scale 2x",
            "train": [
                (np.array([[1]]), np.array([[1,1],[1,1]])),
            ],
            "test_input": np.array([[3]]),
            "test_expected": np.array([[3,3],[3,3]])
        },
        {
            "name": "Rotate 90 CW",
            "train": [
                (np.array([[1,2],[3,4]]), np.array([[3,1],[4,2]])),
            ],
            "test_input": np.array([[5,6],[7,8]]),
            "test_expected": np.array([[7,5],[8,6]])
        },
        {
            "name": "Gravity Down",
            "train": [
                (np.array([[1,0,0],[0,0,0],[0,0,0]]), np.array([[0,0,0],[0,0,0],[1,0,0]])),
                (np.array([[0,2,0],[0,0,0],[0,0,0]]), np.array([[0,0,0],[0,0,0],[0,2,0]])),
            ],
            "test_input": np.array([[0,0,3],[0,0,0],[0,0,0]]),
            "test_expected": np.array([[0,0,0],[0,0,0],[0,0,3]])
        },
    ]
    
    results = []
    
    for i, task in enumerate(demo_tasks):
        console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
        console.print(f"[bold yellow]Task {i+1}: {task['name']}[/bold yellow]")
        
        # Show training example
        console.print("\n[dim]Training Example:[/dim]")
        inp, out = task['train'][0]
        print_grid(inp, "Input")
        console.print("  [bold white]-->[/bold white]")
        print_grid(out, "Output")
        
        # Extract features
        inp_feat = pattern_engine.extract_features(inp)
        out_feat = pattern_engine.extract_features(out)
        changes = pattern_engine.compare_features(inp_feat, out_feat)
        console.print(f"\n[dim]Feature Changes: {changes}[/dim]")
        
        # Solve
        console.print("\n[yellow]Solving...[/yellow]")
        result = engine.solve(task['train'], task['test_input'], f"demo_{i}")
        
        console.print(f"\n[dim]Test Input:[/dim]")
        print_grid(task['test_input'], "Test Input")
        
        if result.success and result.prediction is not None:
            console.print(f"\n[green]SOLVED![/green]")
            console.print(f"Strategy: [cyan]{result.reasoning_trace.strategy_used}[/cyan]")
            console.print(f"Transform: [cyan]{result.transformation_description}[/cyan]")
            console.print(f"Confidence: [cyan]{result.confidence:.2f}[/cyan]")
            
            print_grid(result.prediction, "Prediction")
            
            # Check if correct
            if np.array_equal(result.prediction, task['test_expected']):
                console.print("[bold green]CORRECT![/bold green]")
                results.append(True)
            else:
                console.print("[bold yellow]Predicted but doesn't match expected[/bold yellow]")
                print_grid(task['test_expected'], "Expected")
                results.append(False)
        else:
            console.print(f"\n[red]FAILED[/red]")
            console.print(f"Error: {result.reasoning_trace.error}")
            results.append(False)
    
    # Summary
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(Panel.fit(
        f"[bold green]DEMO COMPLETE[/bold green]\n\n"
        f"Tasks Solved: [bold]{sum(results)}/{len(results)}[/bold]\n"
        f"Accuracy: [bold]{sum(results)/len(results)*100:.1f}%[/bold]",
        border_style="green"
    ))
    
    # Strategy statistics
    console.print("\n[bold]Key Capabilities Demonstrated:[/bold]")
    console.print("  [green]>[/green] Intelligent Pattern Recognition (not brute force)")
    console.print("  [green]>[/green] Feature-based Hypothesis Generation")
    console.print("  [green]>[/green] Multi-strategy Reasoning (Pattern Match -> Analogy -> Synthesis)")
    console.print("  [green]>[/green] Generalization from Single Example")
    console.print("  [green]>[/green] 50+ DSL Primitives Available")

if __name__ == "__main__":
    main()
