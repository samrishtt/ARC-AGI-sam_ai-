"""
ARC-AGI Solver Runner with Comprehensive Logging

Main entry point for solving ARC tasks with detailed logging
of successes and failures for analysis.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
import numpy as np

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.live import Live
from rich.layout import Layout

console = Console()


@dataclass
class TaskResult:
    """Result of solving a single task."""
    task_id: str
    success: bool
    num_train_examples: int
    num_test_examples: int
    test_correct: int
    test_total: int
    time_ms: float
    strategy_used: str
    transformation: str
    error: Optional[str] = None
    predictions: List[Any] = field(default_factory=list)
    expected: List[Any] = field(default_factory=list)


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""
    timestamp: str
    total_tasks: int
    tasks_solved: int
    accuracy: float
    avg_time_ms: float
    strategy_breakdown: Dict[str, int]
    results: List[TaskResult]
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "total_tasks": self.total_tasks,
            "tasks_solved": self.tasks_solved,
            "accuracy": self.accuracy,
            "avg_time_ms": self.avg_time_ms,
            "strategy_breakdown": self.strategy_breakdown,
            "results": [asdict(r) for r in self.results]
        }


class ARCSolverRunner:
    """
    Main runner for ARC-AGI solving with comprehensive logging.
    """
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        from .data_loader import ARCDataLoader
        from .reasoning import ReasoningEngine
        from .enhanced_dsl import enhanced_dsl_registry
        
        self.data_loader = ARCDataLoader()
        self.reasoning_engine = ReasoningEngine(dsl_primitives=enhanced_dsl_registry)
        
        self.results: List[TaskResult] = []
        
    def solve_task(self, task) -> TaskResult:
        """Solve a single ARC task and record results."""
        start_time = time.time()
        
        task_id = task.task_id
        train_examples = task.train_examples
        test_examples = task.test_examples
        
        predictions = []
        correct_count = 0
        strategy_used = "unknown"
        transformation = "none"
        error = None
        
        try:
            for test_input, test_expected in test_examples:
                result = self.reasoning_engine.solve(
                    train_examples=train_examples,
                    test_input=test_input,
                    task_id=task_id
                )
                
                if result.success and result.prediction is not None:
                    predictions.append(result.prediction.tolist())
                    strategy_used = result.reasoning_trace.strategy_used
                    transformation = result.transformation_description
                    
                    # Check correctness
                    if test_expected is not None:
                        if np.array_equal(result.prediction, test_expected):
                            correct_count += 1
                else:
                    predictions.append(None)
                    if result.reasoning_trace.error:
                        error = result.reasoning_trace.error
                        
        except Exception as e:
            error = str(e)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Determine success
        success = correct_count == len(test_examples) and correct_count > 0
        
        return TaskResult(
            task_id=task_id,
            success=success,
            num_train_examples=len(train_examples),
            num_test_examples=len(test_examples),
            test_correct=correct_count,
            test_total=len(test_examples),
            time_ms=elapsed_ms,
            strategy_used=strategy_used,
            transformation=transformation,
            error=error,
            predictions=predictions,
            expected=[e[1].tolist() if e[1] is not None else None for e in test_examples]
        )
    
    def run_benchmark(self, tasks, show_progress: bool = True) -> BenchmarkReport:
        """Run benchmark on a list of tasks."""
        self.results = []
        
        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console
            ) as progress:
                task_progress = progress.add_task(
                    "[cyan]Solving ARC tasks...", 
                    total=len(tasks)
                )
                
                for task in tasks:
                    result = self.solve_task(task)
                    self.results.append(result)
                    
                    status = "[green]✓[/green]" if result.success else "[red]✗[/red]"
                    progress.update(
                        task_progress, 
                        advance=1,
                        description=f"[cyan]Solving: {task.task_id} {status}"
                    )
        else:
            for task in tasks:
                result = self.solve_task(task)
                self.results.append(result)
        
        return self._generate_report()
    
    def _generate_report(self) -> BenchmarkReport:
        """Generate a benchmark report from results."""
        total = len(self.results)
        solved = sum(1 for r in self.results if r.success)
        accuracy = solved / total if total > 0 else 0
        avg_time = sum(r.time_ms for r in self.results) / total if total > 0 else 0
        
        # Strategy breakdown
        strategy_counts = {}
        for r in self.results:
            if r.success:
                strategy = r.strategy_used
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        report = BenchmarkReport(
            timestamp=datetime.now().isoformat(),
            total_tasks=total,
            tasks_solved=solved,
            accuracy=accuracy,
            avg_time_ms=avg_time,
            strategy_breakdown=strategy_counts,
            results=self.results
        )
        
        return report
    
    def save_report(self, report: BenchmarkReport, filename: Optional[str] = None) -> str:
        """Save benchmark report to JSON file."""
        if filename is None:
            filename = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.log_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        return str(filepath)
    
    def print_report(self, report: BenchmarkReport):
        """Print a formatted report to console."""
        console.print()
        console.print(Panel.fit(
            f"[bold cyan]ARC-AGI Benchmark Report[/bold cyan]\n"
            f"[dim]{report.timestamp}[/dim]",
            border_style="cyan"
        ))
        
        # Summary table
        summary_table = Table(title="Summary", show_header=False, box=None)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("Total Tasks", str(report.total_tasks))
        summary_table.add_row("Tasks Solved", f"{report.tasks_solved} ({report.accuracy*100:.1f}%)")
        summary_table.add_row("Average Time", f"{report.avg_time_ms:.2f}ms")
        
        console.print(summary_table)
        console.print()
        
        # Strategy breakdown
        if report.strategy_breakdown:
            strat_table = Table(title="Strategies Used", show_header=True)
            strat_table.add_column("Strategy", style="cyan")
            strat_table.add_column("Count", style="green")
            strat_table.add_column("Percentage", style="yellow")
            
            for strategy, count in sorted(report.strategy_breakdown.items(), key=lambda x: -x[1]):
                pct = count / report.tasks_solved * 100 if report.tasks_solved > 0 else 0
                strat_table.add_row(strategy, str(count), f"{pct:.1f}%")
            
            console.print(strat_table)
            console.print()
        
        # Detailed results
        results_table = Table(title="Task Results", show_header=True)
        results_table.add_column("Task ID", style="cyan")
        results_table.add_column("Status", justify="center")
        results_table.add_column("Correct", justify="center")
        results_table.add_column("Time (ms)", justify="right")
        results_table.add_column("Strategy", style="dim")
        
        for r in sorted(report.results, key=lambda x: (-int(x.success), x.task_id)):
            status = "[green]PASS[/green]" if r.success else "[red]FAIL[/red]"
            results_table.add_row(
                r.task_id,
                status,
                f"{r.test_correct}/{r.test_total}",
                f"{r.time_ms:.1f}",
                r.strategy_used[:20]
            )
        
        console.print(results_table)
    
    def print_failures(self, report: BenchmarkReport, max_failures: int = 10):
        """Print detailed failure analysis."""
        failures = [r for r in report.results if not r.success][:max_failures]
        
        if not failures:
            console.print("[green]No failures to report![/green]")
            return
        
        console.print(Panel.fit(
            f"[bold red]Failure Analysis[/bold red]\n"
            f"[dim]Showing {len(failures)} of {len([r for r in report.results if not r.success])} failures[/dim]",
            border_style="red"
        ))
        
        for r in failures:
            console.print(f"\n[bold red]Task: {r.task_id}[/bold red]")
            console.print(f"  Strategy tried: {r.strategy_used}")
            console.print(f"  Transformation: {r.transformation}")
            if r.error:
                console.print(f"  [red]Error: {r.error}[/red]")
            console.print(f"  Test examples: {r.test_total}")


class ARCVisualizer:
    """Visualize ARC grids in the terminal."""
    
    # ARC color palette (0-9)
    COLORS = [
        'black',    # 0
        'blue',     # 1
        'red',      # 2
        'green',    # 3
        'yellow',   # 4
        'grey',     # 5
        'magenta',  # 6
        'orange1',  # 7
        'cyan',     # 8
        'bright_red'  # 9
    ]
    
    @staticmethod
    def grid_to_rich(grid: np.ndarray, title: str = "") -> Panel:
        """Convert a grid to Rich panel for display."""
        rows, cols = grid.shape
        lines = []
        
        for r in range(rows):
            row_chars = []
            for c in range(cols):
                val = int(grid[r, c])
                color = ARCVisualizer.COLORS[min(val, 9)]
                row_chars.append(f"[{color}]■[/{color}]")
            lines.append(' '.join(row_chars))
        
        content = '\n'.join(lines)
        return Panel(content, title=title, border_style="dim")
    
    @staticmethod
    def display_example(input_grid: np.ndarray, output_grid: np.ndarray, title: str = ""):
        """Display an input/output pair side by side."""
        layout = Layout()
        
        input_panel = ARCVisualizer.grid_to_rich(input_grid, "Input")
        output_panel = ARCVisualizer.grid_to_rich(output_grid, "Output")
        
        console.print(f"\n[bold]{title}[/bold]" if title else "")
        console.print(input_panel)
        console.print("→")
        console.print(output_panel)


def run_sample_benchmark():
    """Run a benchmark on sample tasks."""
    runner = ARCSolverRunner()
    
    console.print(Panel.fit(
        "[bold green]ARC-AGI Sample Benchmark[/bold green]\n"
        "Testing on built-in sample tasks",
        border_style="green"
    ))
    
    # Load sample tasks
    tasks = runner.data_loader.create_sample_tasks()
    
    # Run benchmark
    report = runner.run_benchmark(tasks)
    
    # Print results
    runner.print_report(report)
    
    # Save report
    filepath = runner.save_report(report)
    console.print(f"\n[dim]Report saved to: {filepath}[/dim]")
    
    # Print failure analysis
    runner.print_failures(report)
    
    return report


def run_arc_file_benchmark(data_path: str):
    """Run benchmark on ARC JSON files from a directory."""
    runner = ARCSolverRunner()
    
    console.print(Panel.fit(
        f"[bold green]ARC-AGI File Benchmark[/bold green]\n"
        f"Loading tasks from: {data_path}",
        border_style="green"
    ))
    
    # Load tasks from files
    tasks = runner.data_loader.load_local_dataset(data_path)
    
    if not tasks:
        console.print("[red]No tasks found![/red]")
        return None
    
    # Run benchmark
    report = runner.run_benchmark(tasks)
    
    # Print results
    runner.print_report(report)
    
    # Save report
    filepath = runner.save_report(report)
    console.print(f"\n[dim]Report saved to: {filepath}[/dim]")
    
    # Print failure analysis
    runner.print_failures(report)
    
    return report


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Run on provided data path
        run_arc_file_benchmark(sys.argv[1])
    else:
        # Run on sample tasks
        run_sample_benchmark()
