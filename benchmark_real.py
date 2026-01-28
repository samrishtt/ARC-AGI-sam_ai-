"""
Run benchmark on real ARC data - Fast version
"""
import sys
import os
import json
from datetime import datetime
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress

console = Console()

def main():
    console.print(Panel.fit(
        "[bold green]ARC-AGI REAL DATA BENCHMARK[/bold green]\n"
        "[dim]Testing on Downloaded ARC Tasks[/dim]",
        border_style="green"
    ))
    
    from src.arc.reasoning import ReasoningEngine
    from src.arc.enhanced_dsl import enhanced_dsl_registry
    from src.arc.data_loader import ARCDataLoader
    
    # Load real tasks
    loader = ARCDataLoader()
    data_path = Path("data/training")
    
    if not data_path.exists():
        console.print("[red]No data found. Run test_arc_comprehensive.py first to download.[/red]")
        return
    
    tasks = loader.load_local_dataset(str(data_path))
    console.print(f"[cyan]Loaded {len(tasks)} real ARC tasks[/cyan]")
    
    if not tasks:
        console.print("[red]No tasks loaded![/red]")
        return
    
    # Initialize engine
    engine = ReasoningEngine(dsl_primitives=enhanced_dsl_registry)
    
    results = []
    strategies_used = {}
    
    with Progress() as progress:
        task_bar = progress.add_task("[cyan]Solving tasks...", total=len(tasks))
        
        for task in tasks:
            train_examples = task.train_examples
            test_examples = task.test_examples
            
            task_correct = 0
            task_strategy = "none"
            
            for test_input, test_expected in test_examples:
                result = engine.solve(train_examples, test_input, task.task_id)
                
                if result.success and result.prediction is not None:
                    task_strategy = result.reasoning_trace.strategy_used
                    
                    if test_expected is not None and np.array_equal(result.prediction, test_expected):
                        task_correct += 1
            
            solved = task_correct == len(test_examples) and task_correct > 0
            results.append({
                "task_id": task.task_id,
                "solved": solved,
                "correct": task_correct,
                "total": len(test_examples),
                "strategy": task_strategy
            })
            
            if solved:
                strategies_used[task_strategy] = strategies_used.get(task_strategy, 0) + 1
            
            progress.advance(task_bar)
    
    # Summary
    total = len(results)
    solved = sum(1 for r in results if r["solved"])
    accuracy = solved / total * 100 if total > 0 else 0
    
    console.print("\n")
    console.print(Panel.fit(
        f"[bold green]BENCHMARK RESULTS[/bold green]\n\n"
        f"Total Tasks: [bold]{total}[/bold]\n"
        f"Tasks Solved: [bold green]{solved}[/bold green]\n"
        f"Accuracy: [bold cyan]{accuracy:.1f}%[/bold cyan]",
        border_style="green"
    ))
    
    # Strategy breakdown
    if strategies_used:
        console.print("\n[bold]Strategies Used (Successful):[/bold]")
        strat_table = Table(show_header=True)
        strat_table.add_column("Strategy", style="cyan")
        strat_table.add_column("Count", style="green")
        for strat, count in sorted(strategies_used.items(), key=lambda x: -x[1]):
            strat_table.add_row(strat, str(count))
        console.print(strat_table)
    
    # Show solved tasks
    console.print("\n[bold green]SOLVED TASKS:[/bold green]")
    for r in results:
        if r["solved"]:
            console.print(f"  [green]>[/green] {r['task_id']} - {r['strategy']}")
    
    # Show failed tasks
    console.print("\n[bold red]FAILED TASKS:[/bold red]")
    failed_count = 0
    for r in results:
        if not r["solved"]:
            console.print(f"  [red]x[/red] {r['task_id']}")
            failed_count += 1
            if failed_count >= 15:
                console.print(f"  ... and {total - solved - failed_count} more")
                break
    
    # Save log
    log_path = Path("logs") / f"real_arc_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_path.parent.mkdir(exist_ok=True)
    with open(log_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_tasks": total,
            "solved": solved,
            "accuracy": accuracy,
            "strategies": strategies_used,
            "results": results
        }, f, indent=2)
    
    console.print(f"\n[dim]Log saved to: {log_path}[/dim]")
    
    # Performance note
    console.print("\n[bold yellow]ANALYSIS:[/bold yellow]")
    if accuracy > 20:
        console.print(f"  [green]>[/green] Solver is performing above baseline ({accuracy:.1f}% vs ~5% random)")
    if accuracy > 50:
        console.print(f"  [green]>[/green] Strong performance - approaching competitive levels")
    if accuracy < 50:
        console.print(f"  [yellow]>[/yellow] Room for improvement - need more composite strategies")
        console.print(f"  [yellow]>[/yellow] Consider adding task-specific pattern detectors")

if __name__ == "__main__":
    main()
