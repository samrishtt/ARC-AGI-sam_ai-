"""
ARC-AGI Ultimate Benchmark Runner

Run benchmarks using the super-enhanced reasoning engine.
Goal: Demonstrate human-level accuracy (85%+)
"""

import sys
import os
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, asdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.live import Live

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
    total_tasks: int
    tasks_solved: int
    accuracy: float
    avg_time_ms: float
    strategy_breakdown: Dict[str, int]
    results: List[TaskResult]
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
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


def run_benchmark(
    data_dir: str,
    max_tasks: Optional[int] = None,
    show_progress: bool = True,
    verbose: bool = False
) -> BenchmarkResult:
    """
    Run benchmark on ARC tasks in a directory.
    
    Args:
        data_dir: Directory containing ARC JSON files
        max_tasks: Maximum number of tasks to run (None for all)
        show_progress: Show progress bar
        verbose: Print details for each task
        
    Returns:
        BenchmarkResult with all statistics
    """
    from src.arc.super_reasoning import SuperReasoningEngine
    
    # Find all task files
    data_path = Path(data_dir)
    task_files = sorted(data_path.glob("*.json"))
    
    if max_tasks:
        task_files = task_files[:max_tasks]
    
    console.print(Panel.fit(
        f"[bold magenta]🚀 ARC-AGI ULTIMATE BENCHMARK[/bold magenta]\n"
        f"[dim]Tasks: {len(task_files)} | Directory: {data_dir}[/dim]",
        border_style="magenta"
    ))
    
    # Initialize engine
    engine = SuperReasoningEngine()
    
    results: List[TaskResult] = []
    strategy_counts: Dict[str, int] = {}
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
                result = _solve_task_file(engine, task_file, verbose)
                results.append(result)
                
                if result.success:
                    solved += 1
                    strategy_counts[result.strategy] = strategy_counts.get(result.strategy, 0) + 1
                
                total_time += result.time_ms
                progress.update(task, advance=1, description=f"[cyan]Solved: {solved}/{len(results)}")
    else:
        for task_file in task_files:
            result = _solve_task_file(engine, task_file, verbose)
            results.append(result)
            
            if result.success:
                solved += 1
                strategy_counts[result.strategy] = strategy_counts.get(result.strategy, 0) + 1
            
            total_time += result.time_ms
    
    # Build result
    accuracy = (solved / len(results)) * 100 if results else 0
    avg_time = total_time / len(results) if results else 0
    
    benchmark_result = BenchmarkResult(
        timestamp=datetime.now().isoformat(),
        total_tasks=len(results),
        tasks_solved=solved,
        accuracy=accuracy,
        avg_time_ms=avg_time,
        strategy_breakdown=strategy_counts,
        results=results
    )
    
    # Print summary
    _print_summary(benchmark_result)
    
    return benchmark_result


def _solve_task_file(
    engine,
    task_file: Path,
    verbose: bool
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
            result = engine.solve(train_examples, test_input, task_id)
            total_time += result.time_ms
            
            if result.success and result.prediction is not None:
                if np.array_equal(result.prediction, test_expected):
                    correct += 1
                    final_strategy = result.strategy.value
                    final_description = result.description
                    final_confidence = result.confidence
        
        success = correct == total
        
        if verbose:
            status = "✓" if success else "✗"
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
            console.print(f"  ✗ {task_id}: Error - {e}")
        
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
    
    # Accuracy panel
    acc_color = "green" if result.accuracy >= 50 else "yellow" if result.accuracy >= 25 else "red"
    console.print(Panel.fit(
        f"[bold {acc_color}]ACCURACY: {result.accuracy:.1f}%[/bold {acc_color}]\n"
        f"[dim]Solved {result.tasks_solved}/{result.total_tasks} tasks[/dim]\n"
        f"[dim]Average time: {result.avg_time_ms:.1f}ms per task[/dim]",
        title="📊 Results",
        border_style=acc_color
    ))
    
    # Strategy breakdown
    if result.strategy_breakdown:
        table = Table(title="Strategy Breakdown", show_header=True)
        table.add_column("Strategy", style="cyan")
        table.add_column("Count", justify="right", style="green")
        table.add_column("Percentage", justify="right")
        
        for strategy, count in sorted(result.strategy_breakdown.items(), 
                                      key=lambda x: x[1], reverse=True):
            pct = (count / result.tasks_solved * 100) if result.tasks_solved > 0 else 0
            table.add_row(strategy, str(count), f"{pct:.1f}%")
        
        console.print(table)
    
    # Failure analysis
    failures = [r for r in result.results if not r.success]
    if failures:
        console.print(f"\n[yellow]Failed tasks ({len(failures)}):[/yellow]")
        for f in failures[:10]:  # Show first 10
            console.print(f"  • {f.task_id}")
        if len(failures) > 10:
            console.print(f"  ... and {len(failures) - 10} more")


def save_benchmark_result(result: BenchmarkResult, output_dir: str = "logs") -> str:
    """Save benchmark result to JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = output_path / filename
    
    with open(filepath, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    
    console.print(f"\n[dim]Results saved to: {filepath}[/dim]")
    return str(filepath)


def run_training_benchmark(max_tasks: Optional[int] = None) -> BenchmarkResult:
    """Run benchmark on training data."""
    return run_benchmark("data/training", max_tasks=max_tasks)


def run_evaluation_benchmark(max_tasks: Optional[int] = None) -> BenchmarkResult:
    """Run benchmark on evaluation data."""
    return run_benchmark("data/evaluation", max_tasks=max_tasks)


def run_full_benchmark() -> Tuple[BenchmarkResult, BenchmarkResult]:
    """Run benchmark on both training and evaluation."""
    console.print("[bold]Phase 1: Training Data[/bold]")
    train_result = run_training_benchmark()
    
    console.print("\n[bold]Phase 2: Evaluation Data[/bold]")
    eval_result = run_evaluation_benchmark()
    
    # Combined summary
    total = train_result.total_tasks + eval_result.total_tasks
    solved = train_result.tasks_solved + eval_result.tasks_solved
    accuracy = (solved / total * 100) if total > 0 else 0
    
    console.print(Panel.fit(
        f"[bold green]COMBINED ACCURACY: {accuracy:.1f}%[/bold green]\n"
        f"[dim]Training: {train_result.accuracy:.1f}% | Evaluation: {eval_result.accuracy:.1f}%[/dim]",
        title="🏆 Overall Results",
        border_style="green"
    ))
    
    return train_result, eval_result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ARC-AGI Ultimate Benchmark")
    parser.add_argument("--data", type=str, default="data/training",
                        help="Directory containing ARC tasks")
    parser.add_argument("--max", type=int, default=None,
                        help="Maximum number of tasks to run")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    parser.add_argument("--save", action="store_true",
                        help="Save results to file")
    parser.add_argument("--full", action="store_true",
                        help="Run on both training and evaluation")
    
    args = parser.parse_args()
    
    if args.full:
        train_result, eval_result = run_full_benchmark()
        if args.save:
            save_benchmark_result(train_result, "logs/training")
            save_benchmark_result(eval_result, "logs/evaluation")
    else:
        result = run_benchmark(args.data, max_tasks=args.max, verbose=args.verbose)
        if args.save:
            save_benchmark_result(result)
