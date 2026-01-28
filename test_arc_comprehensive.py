"""
ARC-AGI Comprehensive Test Suite

Tests the solver on sample tasks and real ARC data.
Generates detailed logs of successes and failures.
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress

console = Console()


def download_arc_data():
    """Download real ARC-AGI data from GitHub."""
    import urllib.request
    import zipfile
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # ARC Prize 2024 data URLs (public subset)
    urls = {
        "training": "https://raw.githubusercontent.com/fchollet/ARC-AGI/master/data/training/",
        "evaluation": "https://raw.githubusercontent.com/fchollet/ARC-AGI/master/data/evaluation/"
    }
    
    # Sample task IDs from the original ARC dataset
    sample_tasks = [
        "007bbfb7", "00d62c1b", "017c7c7b", "025d127b", "045e512c",
        "0520fde7", "05269061", "05f2a901", "06df4c85", "08ed6ac7",
        "0962bcdd", "09629e4f", "0a938d79", "0b148d64", "0ca9ddb6",
        "0d3d703e", "0dfd9992", "0e206a2e", "10fcaaa3", "11852cab",
        "1190e5a7", "137eaa0f", "150deff5", "178fcbfb", "1a07d186"
    ]
    
    training_dir = data_dir / "training"
    training_dir.mkdir(exist_ok=True)
    
    console.print("[cyan]Downloading ARC training data...[/cyan]")
    
    downloaded = 0
    with Progress() as progress:
        task = progress.add_task("[green]Downloading...", total=len(sample_tasks))
        
        for task_id in sample_tasks:
            url = f"{urls['training']}{task_id}.json"
            target = training_dir / f"{task_id}.json"
            
            if not target.exists():
                try:
                    urllib.request.urlretrieve(url, target)
                    downloaded += 1
                except Exception as e:
                    console.print(f"[dim]Could not download {task_id}: {e}[/dim]")
            
            progress.advance(task)
    
    console.print(f"[green]Downloaded {downloaded} new tasks[/green]")
    return training_dir


def create_embedded_arc_tasks():
    """
    Create embedded ARC-style tasks that mimic real ARC patterns.
    These are used when network access is unavailable.
    """
    tasks = []
    
    # Task 1: Horizontal reflection
    tasks.append({
        "id": "reflect_h",
        "train": [
            {"input": [[1,2,0],[0,0,0],[0,0,0]], "output": [[0,2,1],[0,0,0],[0,0,0]]},
            {"input": [[3,0,0],[4,0,0],[0,0,0]], "output": [[0,0,3],[0,0,4],[0,0,0]]}
        ],
        "test": [
            {"input": [[5,6,0],[0,0,0],[7,0,0]], "output": [[0,6,5],[0,0,0],[0,0,7]]}
        ]
    })
    
    # Task 2: Vertical reflection
    tasks.append({
        "id": "reflect_v",
        "train": [
            {"input": [[1,0],[0,0],[0,0]], "output": [[0,0],[0,0],[1,0]]},
            {"input": [[2,3],[0,0],[0,0]], "output": [[0,0],[0,0],[2,3]]}
        ],
        "test": [
            {"input": [[4,5],[6,0],[0,0]], "output": [[0,0],[6,0],[4,5]]}
        ]
    })
    
    # Task 3: 90 degree rotation
    tasks.append({
        "id": "rotate_90",
        "train": [
            {"input": [[1,0],[0,0]], "output": [[0,1],[0,0]]},
            {"input": [[2,3],[0,0]], "output": [[0,2],[0,3]]}
        ],
        "test": [
            {"input": [[4,0],[5,0]], "output": [[5,4],[0,0]]}
        ]
    })
    
    # Task 4: Crop to content
    tasks.append({
        "id": "crop",
        "train": [
            {"input": [[0,0,0,0],[0,1,1,0],[0,1,1,0],[0,0,0,0]], "output": [[1,1],[1,1]]},
            {"input": [[0,0,0],[0,5,0],[0,0,0]], "output": [[5]]}
        ],
        "test": [
            {"input": [[0,0,0,0,0],[0,0,3,3,0],[0,0,3,3,0],[0,0,0,0,0]], "output": [[3,3],[3,3]]}
        ]
    })
    
    # Task 5: Color replacement
    tasks.append({
        "id": "color_replace",
        "train": [
            {"input": [[1,1,0],[1,0,0],[0,0,0]], "output": [[2,2,0],[2,0,0],[0,0,0]]},
            {"input": [[0,0,1],[0,1,1],[1,1,1]], "output": [[0,0,2],[0,2,2],[2,2,2]]}
        ],
        "test": [
            {"input": [[1,0,1],[0,0,0],[1,0,1]], "output": [[2,0,2],[0,0,0],[2,0,2]]}
        ]
    })
    
    # Task 6: Scale 2x
    tasks.append({
        "id": "scale_2x",
        "train": [
            {"input": [[1]], "output": [[1,1],[1,1]]},
            {"input": [[2,0],[0,3]], "output": [[2,2,0,0],[2,2,0,0],[0,0,3,3],[0,0,3,3]]}
        ],
        "test": [
            {"input": [[5]], "output": [[5,5],[5,5]]}
        ]
    })
    
    # Task 7: Keep largest object
    tasks.append({
        "id": "keep_largest",
        "train": [
            {"input": [[1,0,2],[0,2,2],[0,2,2]], "output": [[0,0,2],[0,2,2],[0,2,2]]},
            {"input": [[5,5,5],[0,1,0],[0,0,0]], "output": [[5,5,5],[0,0,0],[0,0,0]]}
        ],
        "test": [
            {"input": [[1,1,0],[0,0,3],[0,0,0]], "output": [[1,1,0],[0,0,0],[0,0,0]]}
        ]
    })
    
    # Task 8: Gravity (drop down)
    tasks.append({
        "id": "gravity",
        "train": [
            {"input": [[1,0,0],[0,0,0],[0,0,0]], "output": [[0,0,0],[0,0,0],[1,0,0]]},
            {"input": [[0,2,0],[0,0,0],[0,0,0]], "output": [[0,0,0],[0,0,0],[0,2,0]]}
        ],
        "test": [
            {"input": [[0,0,3],[0,0,0],[0,0,0]], "output": [[0,0,0],[0,0,0],[0,0,3]]}
        ]
    })
    
    # Task 9: Fill column
    tasks.append({
        "id": "fill_column",
        "train": [
            {"input": [[0,1,0],[0,0,0],[0,0,0]], "output": [[0,1,0],[0,1,0],[0,1,0]]},
            {"input": [[0,0,2],[0,0,0],[0,0,0]], "output": [[0,0,2],[0,0,2],[0,0,2]]}
        ],
        "test": [
            {"input": [[3,0,0],[0,0,0],[0,0,0]], "output": [[3,0,0],[3,0,0],[3,0,0]]}
        ]
    })
    
    # Task 10: Invert binary
    tasks.append({
        "id": "invert",
        "train": [
            {"input": [[1,1],[1,0]], "output": [[0,0],[0,1]]},
            {"input": [[0,1],[1,1]], "output": [[1,0],[0,0]]}
        ],
        "test": [
            {"input": [[1,0],[0,1]], "output": [[0,1],[1,0]]}
        ]
    })
    
    # Task 11: Transpose
    tasks.append({
        "id": "transpose",
        "train": [
            {"input": [[1,2],[3,4]], "output": [[1,3],[2,4]]},
            {"input": [[5,0],[0,6]], "output": [[5,0],[0,6]]}
        ],
        "test": [
            {"input": [[7,8],[9,0]], "output": [[7,9],[8,0]]}
        ]
    })
    
    # Task 12: Tile 2x2
    tasks.append({
        "id": "tile_2x2",
        "train": [
            {"input": [[1]], "output": [[1,1],[1,1]]},
            {"input": [[2,3]], "output": [[2,3,2,3],[2,3,2,3]]}
        ],
        "test": [
            {"input": [[4,5],[6,7]], "output": [[4,5,4,5],[6,7,6,7],[4,5,4,5],[6,7,6,7]]}
        ]
    })
    
    # Task 13: Extract pattern
    tasks.append({
        "id": "extract_nonzero",
        "train": [
            {"input": [[0,0,0],[0,5,0],[0,0,0]], "output": [[5]]},
            {"input": [[0,0],[0,3]], "output": [[3]]}
        ],
        "test": [
            {"input": [[0,0,0],[0,0,0],[0,0,8]], "output": [[8]]}
        ]
    })
    
    # Task 14: Color each object differently
    tasks.append({
        "id": "color_objects",
        "train": [
            {"input": [[1,0,1],[0,0,0],[1,0,1]], "output": [[1,0,2],[0,0,0],[3,0,4]]},
        ],
        "test": [
            {"input": [[1,0],[0,1]], "output": [[1,0],[0,2]]}
        ]
    })
    
    # Task 15: Remove border
    tasks.append({
        "id": "remove_border",
        "train": [
            {"input": [[1,1,1],[1,2,1],[1,1,1]], "output": [[2]]},
            {"input": [[3,3,3,3],[3,5,6,3],[3,7,8,3],[3,3,3,3]], "output": [[5,6],[7,8]]}
        ],
        "test": [
            {"input": [[2,2,2],[2,9,2],[2,2,2]], "output": [[9]]}
        ]
    })
    
    # Task 16: Make symmetric
    tasks.append({
        "id": "make_symmetric_h",
        "train": [
            {"input": [[1,0],[2,0]], "output": [[1,1],[2,2]]},
            {"input": [[3,0],[0,0]], "output": [[3,3],[0,0]]}
        ],
        "test": [
            {"input": [[4,0],[5,0]], "output": [[4,4],[5,5]]}
        ]
    })
    
    # Task 17: Outline objects
    tasks.append({
        "id": "outline",
        "train": [
            {"input": [[1,1,1],[1,1,1],[1,1,1]], "output": [[1,1,1],[1,0,1],[1,1,1]]},
        ],
        "test": [
            {"input": [[2,2,2,2],[2,2,2,2],[2,2,2,2],[2,2,2,2]], "output": [[2,2,2,2],[2,0,0,2],[2,0,0,2],[2,2,2,2]]}
        ]
    })
    
    # Task 18: Count and output
    tasks.append({
        "id": "count",
        "train": [
            {"input": [[1,0,1]], "output": [[2]]},
            {"input": [[1,1,1,0]], "output": [[3]]}
        ],
        "test": [
            {"input": [[1,1,0,1,1]], "output": [[4]]}
        ]
    })
    
    # Task 19: Fill holes
    tasks.append({
        "id": "fill_holes",
        "train": [
            {"input": [[1,1,1],[1,0,1],[1,1,1]], "output": [[1,1,1],[1,1,1],[1,1,1]]},
        ],
        "test": [
            {"input": [[2,2,2,2],[2,0,0,2],[2,0,0,2],[2,2,2,2]], "output": [[2,2,2,2],[2,2,2,2],[2,2,2,2],[2,2,2,2]]}
        ]
    })
    
    # Task 20: Diagonal mirror
    tasks.append({
        "id": "diagonal",
        "train": [
            {"input": [[1,0,0],[0,0,0],[0,0,0]], "output": [[1,0,0],[0,1,0],[0,0,1]]},
            {"input": [[2,0,0],[0,0,0],[0,0,0]], "output": [[2,0,0],[0,2,0],[0,0,2]]}
        ],
        "test": [
            {"input": [[3,0,0],[0,0,0],[0,0,0]], "output": [[3,0,0],[0,3,0],[0,0,3]]}
        ]
    })
    
    return tasks


def run_comprehensive_test():
    """Run comprehensive test suite on the ARC solver."""
    from src.arc.solver import ARCSolverRunner, ARCVisualizer
    from src.arc.data_loader import ARCTask
    
    console.print(Panel.fit(
        "[bold magenta]=========================================[/bold magenta]\n"
        "[bold magenta]   ARC-AGI COMPREHENSIVE TEST SUITE     [/bold magenta]\n"
        "[bold magenta]=========================================[/bold magenta]",
        border_style="magenta"
    ))
    
    # Initialize runner
    runner = ARCSolverRunner(log_dir="logs")
    
    # Phase 1: Sample tasks
    console.print("\n[bold cyan]Phase 1: Built-in Sample Tasks[/bold cyan]")
    sample_tasks = runner.data_loader.create_sample_tasks()
    sample_report = runner.run_benchmark(sample_tasks)
    
    # Phase 2: Embedded ARC tasks
    console.print("\n[bold cyan]Phase 2: Embedded ARC-style Tasks[/bold cyan]")
    embedded_data = create_embedded_arc_tasks()
    
    embedded_tasks = []
    for t in embedded_data:
        train_examples = [
            (np.array(ex["input"]), np.array(ex["output"])) 
            for ex in t["train"]
        ]
        test_examples = [
            (np.array(ex["input"]), np.array(ex.get("output", [[0]]))) 
            for ex in t["test"]
        ]
        embedded_tasks.append(ARCTask(t["id"], train_examples, test_examples))
    
    embedded_report = runner.run_benchmark(embedded_tasks)
    
    # Phase 3: Try downloading real ARC data
    console.print("\n[bold cyan]Phase 3: Real ARC Data (if available)[/bold cyan]")
    real_report = None
    try:
        data_dir = download_arc_data()
        real_tasks = runner.data_loader.load_local_dataset(str(data_dir))
        if real_tasks:
            real_report = runner.run_benchmark(real_tasks)
    except Exception as e:
        console.print(f"[yellow]Could not load real ARC data: {e}[/yellow]")
    
    # Combined Report
    console.print("\n" + "="*60)
    console.print(Panel.fit(
        "[bold green]FINAL RESULTS SUMMARY[/bold green]",
        border_style="green"
    ))
    
    # Print all reports
    console.print("\n[bold]Sample Tasks:[/bold]")
    runner.print_report(sample_report)
    
    console.print("\n[bold]Embedded ARC Tasks:[/bold]")
    runner.print_report(embedded_report)
    
    if real_report:
        console.print("\n[bold]Real ARC Tasks:[/bold]")
        runner.print_report(real_report)
    
    # Save comprehensive log
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "sample_results": sample_report.to_dict(),
        "embedded_results": embedded_report.to_dict(),
        "real_results": real_report.to_dict() if real_report else None
    }
    
    log_path = Path("logs") / f"comprehensive_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2, default=str)
    
    console.print(f"\n[dim]Full log saved to: {log_path}[/dim]")
    
    # Print failure analysis
    console.print("\n[bold red]FAILURE ANALYSIS[/bold red]")
    runner.print_failures(sample_report)
    runner.print_failures(embedded_report)
    
    # Calculate overall stats
    total_tasks = sample_report.total_tasks + embedded_report.total_tasks
    total_solved = sample_report.tasks_solved + embedded_report.tasks_solved
    if real_report:
        total_tasks += real_report.total_tasks
        total_solved += real_report.tasks_solved
    
    overall_accuracy = total_solved / total_tasks if total_tasks > 0 else 0
    
    console.print("\n" + "="*60)
    console.print(Panel.fit(
        f"[bold green]OVERALL ACCURACY: {overall_accuracy*100:.1f}%[/bold green]\n"
        f"[dim]Tasks Solved: {total_solved}/{total_tasks}[/dim]",
        border_style="green"
    ))
    
    return {
        "sample": sample_report,
        "embedded": embedded_report,
        "real": real_report,
        "overall_accuracy": overall_accuracy
    }


def test_individual_task(task_id: str = "rotate_90"):
    """Test a specific task with visualization."""
    from src.arc.solver import ARCSolverRunner, ARCVisualizer
    from src.arc.data_loader import ARCTask
    from src.arc.reasoning import ReasoningEngine
    from src.arc.enhanced_dsl import enhanced_dsl_registry
    
    console.print(f"[bold cyan]Testing task: {task_id}[/bold cyan]")
    
    # Find the task
    embedded_data = create_embedded_arc_tasks()
    task_data = None
    for t in embedded_data:
        if t["id"] == task_id:
            task_data = t
            break
    
    if not task_data:
        console.print(f"[red]Task {task_id} not found[/red]")
        return
    
    # Convert to numpy arrays
    train_examples = [
        (np.array(ex["input"]), np.array(ex["output"])) 
        for ex in task_data["train"]
    ]
    test_examples = [
        (np.array(ex["input"]), np.array(ex.get("output", [[0]]))) 
        for ex in task_data["test"]
    ]
    
    # Show training examples
    console.print("\n[bold]Training Examples:[/bold]")
    for i, (inp, out) in enumerate(train_examples):
        console.print(f"\n[cyan]Example {i+1}:[/cyan]")
        ARCVisualizer.display_example(inp, out)
    
    # Solve
    console.print("\n[bold yellow]Solving...[/bold yellow]")
    engine = ReasoningEngine(dsl_primitives=enhanced_dsl_registry)
    
    for i, (test_input, test_expected) in enumerate(test_examples):
        result = engine.solve(train_examples, test_input, task_id)
        
        console.print(f"\n[bold]Test {i+1}:[/bold]")
        console.print(f"Strategy: {result.reasoning_trace.strategy_used}")
        console.print(f"Transform: {result.transformation_description}")
        console.print(f"Confidence: {result.confidence:.2f}")
        
        if result.success and result.prediction is not None:
            console.print("[green]✓ SOLVED[/green]")
            ARCVisualizer.display_example(test_input, result.prediction, "Solution")
            
            if not np.array_equal(result.prediction, test_expected):
                console.print("[yellow]Warning: Prediction differs from expected[/yellow]")
        else:
            console.print("[red]✗ FAILED[/red]")
            console.print(f"Error: {result.reasoning_trace.error}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ARC-AGI Test Suite")
    parser.add_argument("--task", type=str, help="Test a specific task by ID")
    parser.add_argument("--quick", action="store_true", help="Quick test mode")
    
    args = parser.parse_args()
    
    if args.task:
        test_individual_task(args.task)
    else:
        run_comprehensive_test()
