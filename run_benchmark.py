"""
Ultra Benchmark Runner for ARC-AGI

Run the ultra-powerful self-improving solver against the ARC benchmark.
Target: 80%+ accuracy
"""

import sys
import os
import io

# Fix Windows encoding issues
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich import box

console = Console()


@dataclass
class TaskResult:
    """Result for a single task."""
    task_id: str
    success: bool
    strategy: str
    description: str
    confidence: float
    time_ms: float
    num_examples: int
    test_correct: int
    test_total: int


@dataclass
class BenchmarkResult:
    """Complete benchmark result."""
    timestamp: str
    solver_version: str
    total_tasks: int
    tasks_solved: int
    accuracy: float
    avg_time_ms: float
    strategy_breakdown: Dict[str, int]
    results: List[TaskResult]
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'solver_version': self.solver_version,
            'total_tasks': self.total_tasks,
            'tasks_solved': self.tasks_solved,
            'accuracy': self.accuracy,
            'avg_time_ms': self.avg_time_ms,
            'strategy_breakdown': self.strategy_breakdown,
            'results': [asdict(r) for r in self.results]
        }


def load_arc_task(json_path: Path) -> Dict:
    """Load a single ARC task from JSON."""
    with open(json_path, 'r') as f:
        return json.load(f)


def parse_arc_task(data: Dict) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[Tuple[np.ndarray, np.ndarray]]]:
    """Parse ARC task JSON into train/test examples."""
    train_examples = []
    for ex in data.get('train', []):
        inp = np.array(ex['input'], dtype=np.int32)
        out = np.array(ex['output'], dtype=np.int32)
        train_examples.append((inp, out))
    
    test_examples = []
    for ex in data.get('test', []):
        inp = np.array(ex['input'], dtype=np.int32)
        out = np.array(ex.get('output', [[0]]), dtype=np.int32)
        test_examples.append((inp, out))
    
    return train_examples, test_examples


def run_ultra_benchmark(
    data_dir: str,
    max_tasks: Optional[int] = None,
    show_progress: bool = True,
    verbose: bool = False,
    use_ensemble: bool = True
) -> BenchmarkResult:
    """
    Run the ultra benchmark on ARC tasks.
    
    Args:
        data_dir: Directory containing ARC JSON files
        max_tasks: Maximum number of tasks to run (None for all)
        show_progress: Show progress bar
        verbose: Print details for each task
        use_ensemble: Use ensemble voting (slower but more accurate)
        
    Returns:
        BenchmarkResult with all statistics
    """
    # Import solvers
    if use_ensemble:
        from src.arc.ensemble_solver import EnsembleSolver
        solver = EnsembleSolver()
        solver_version = "UltraSolver-Ensemble-v2.0"
    else:
        from src.arc.ultra_solver import UltraSolver
        solver = UltraSolver(enable_learning=True)
        solver_version = "UltraSolver-v2.0"
    
    # Find all task files
    data_path = Path(data_dir)
    task_files = sorted(data_path.glob("*.json"))
    
    if max_tasks:
        task_files = task_files[:max_tasks]
    
    console.print(Panel.fit(
        f"[bold magenta]>>> ARC-AGI ULTRA BENCHMARK <<<[/bold magenta]\n"
        f"[dim]Solver: {solver_version}[/dim]\n"
        f"[dim]Tasks: {len(task_files)} | Directory: {data_dir}[/dim]",
        border_style="magenta"
    ))
    
    results: List[TaskResult] = []
    strategy_counts: Dict[str, int] = defaultdict(int)
    total_time = 0
    solved = 0
    
    # Run benchmark
    if show_progress:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Solving tasks...", total=len(task_files))
            
            for task_file in task_files:
                result = _solve_task_file(solver, task_file, verbose, use_ensemble)
                results.append(result)
                
                if result.success:
                    solved += 1
                    strategy_counts[result.strategy] += 1
                
                total_time += result.time_ms
                
                accuracy_so_far = (solved / len(results)) * 100
                progress.update(
                    task, 
                    advance=1, 
                    description=f"[cyan]Solved: {solved}/{len(results)} ({accuracy_so_far:.1f}%)"
                )
    else:
        for i, task_file in enumerate(task_files):
            result = _solve_task_file(solver, task_file, verbose, use_ensemble)
            results.append(result)
            
            if result.success:
                solved += 1
                strategy_counts[result.strategy] += 1
            
            total_time += result.time_ms
            
            if verbose and (i + 1) % 50 == 0:
                accuracy_so_far = (solved / len(results)) * 100
                console.print(f"  Progress: {i + 1}/{len(task_files)} | Accuracy: {accuracy_so_far:.1f}%")
    
    # Build result
    accuracy = (solved / len(results)) * 100 if results else 0
    avg_time = total_time / len(results) if results else 0
    
    benchmark_result = BenchmarkResult(
        timestamp=datetime.now().isoformat(),
        solver_version=solver_version,
        total_tasks=len(results),
        tasks_solved=solved,
        accuracy=accuracy,
        avg_time_ms=avg_time,
        strategy_breakdown=dict(strategy_counts),
        results=results
    )
    
    # Print summary
    _print_summary(benchmark_result)
    
    return benchmark_result


def _solve_task_file(
    solver,
    task_file: Path,
    verbose: bool,
    use_ensemble: bool
) -> TaskResult:
    """Solve a single task file."""
    task_id = task_file.stem
    
    try:
        data = load_arc_task(task_file)
        train_examples, test_examples = parse_arc_task(data)
        
        correct = 0
        total = len(test_examples)
        final_strategy = "none"
        final_description = ""
        final_confidence = 0.0
        total_time = 0
        
        for test_input, test_expected in test_examples:
            result = solver.solve(train_examples, test_input, task_id)
            
            if use_ensemble:
                total_time += result.time_ms
                if result.success and result.prediction is not None:
                    if np.array_equal(result.prediction, test_expected):
                        correct += 1
                        final_strategy = ", ".join(result.strategies_used[:2])
                        final_description = result.reasoning
                        final_confidence = result.confidence
            else:
                total_time += result.time_ms
                if result.success and result.prediction is not None:
                    if np.array_equal(result.prediction, test_expected):
                        correct += 1
                        final_strategy = result.strategy.value
                        final_description = result.description
                        final_confidence = result.confidence
        
        success = correct == total
        
        if verbose:
            status = "[green]OK[/green]" if success else "[red]FAIL[/red]"
            console.print(f"  {status} {task_id}: {final_strategy} ({correct}/{total})")
        
        return TaskResult(
            task_id=task_id,
            success=success,
            strategy=final_strategy,
            description=final_description,
            confidence=final_confidence,
            time_ms=total_time,
            num_examples=len(train_examples),
            test_correct=correct,
            test_total=total
        )
        
    except Exception as e:
        if verbose:
            console.print(f"  [red]ERR[/red] {task_id}: Error - {e}")
        
        return TaskResult(
            task_id=task_id,
            success=False,
            strategy="error",
            description=str(e),
            confidence=0.0,
            time_ms=0,
            num_examples=0,
            test_correct=0,
            test_total=1
        )


def _print_summary(result: BenchmarkResult) -> None:
    """Print benchmark summary."""
    console.print()
    
    # Accuracy panel with colored tier
    if result.accuracy >= 80:
        tier = "*** EXCEPTIONAL ***"
        acc_color = "green"
    elif result.accuracy >= 60:
        tier = "** EXCELLENT **"
        acc_color = "green"
    elif result.accuracy >= 40:
        tier = "* GOOD *"
        acc_color = "yellow"
    elif result.accuracy >= 20:
        tier = "IMPROVING"
        acc_color = "yellow"
    else:
        tier = "DEVELOPING"
        acc_color = "red"
    
    console.print(Panel.fit(
        f"[bold {acc_color}]{tier}[/bold {acc_color}]\n"
        f"[bold {acc_color}]ACCURACY: {result.accuracy:.1f}%[/bold {acc_color}]\n"
        f"[dim]Solved {result.tasks_solved}/{result.total_tasks} tasks[/dim]\n"
        f"[dim]Average time: {result.avg_time_ms:.1f}ms per task[/dim]\n"
        f"[dim]Solver: {result.solver_version}[/dim]",
        title="=== Results ===",
        border_style=acc_color
    ))
    
    # Strategy breakdown
    if result.strategy_breakdown:
        table = Table(title="Strategy Breakdown", show_header=True, box=box.ROUNDED)
        table.add_column("Strategy", style="cyan")
        table.add_column("Count", justify="right", style="green")
        table.add_column("Percentage", justify="right")
        
        for strategy, count in sorted(result.strategy_breakdown.items(), 
                                      key=lambda x: x[1], reverse=True):
            pct = (count / result.tasks_solved * 100) if result.tasks_solved > 0 else 0
            table.add_row(strategy, str(count), f"{pct:.1f}%")
        
        console.print(table)
    
    # Success/failure distribution chart
    console.print(f"\n[bold]Performance Distribution:[/bold]")
    success_bar = int(result.accuracy / 5)  # Scale to 20 chars
    failure_bar = 20 - success_bar
    console.print(f"  [green]{'#' * success_bar}[/green][red]{'-' * failure_bar}[/red] {result.accuracy:.1f}%")
    
    # Failure analysis
    failures = [r for r in result.results if not r.success]
    if failures:
        console.print(f"\n[yellow]Failed tasks ({len(failures)}):[/yellow]")
        for f in failures[:10]:  # Show first 10
            console.print(f"  - {f.task_id}")
        if len(failures) > 10:
            console.print(f"  ... and {len(failures) - 10} more")


def save_benchmark_result(result: BenchmarkResult, output_dir: str = "logs") -> str:
    """Save benchmark result to JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = f"ultra_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = output_path / filename
    
    with open(filepath, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    
    console.print(f"\n[dim]Results saved to: {filepath}[/dim]")
    return str(filepath)


def run_quick_test(num_tasks: int = 20) -> BenchmarkResult:
    """Run a quick test on a subset of tasks."""
    console.print("[bold cyan]Running quick test...[/bold cyan]")
    return run_ultra_benchmark("data/training", max_tasks=num_tasks, use_ensemble=False)


def run_full_benchmark(use_ensemble: bool = True) -> BenchmarkResult:
    """Run full benchmark on all training data."""
    console.print("[bold cyan]Running full benchmark...[/bold cyan]")
    return run_ultra_benchmark("data/training", use_ensemble=use_ensemble)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ARC-AGI Ultra Benchmark")
    parser.add_argument("--data", type=str, default="data/training",
                        help="Directory containing ARC tasks")
    parser.add_argument("--max", type=int, default=None,
                        help="Maximum number of tasks to run")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    parser.add_argument("--save", action="store_true",
                        help="Save results to file")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick test (20 tasks)")
    parser.add_argument("--no-ensemble", action="store_true",
                        help="Disable ensemble voting (faster)")
    
    args = parser.parse_args()
    
    if args.quick:
        result = run_quick_test(20)
    else:
        result = run_ultra_benchmark(
            args.data, 
            max_tasks=args.max, 
            verbose=args.verbose,
            use_ensemble=not args.no_ensemble
        )
    
    if args.save:
        save_benchmark_result(result)
