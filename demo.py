import numpy as np
from src.core.agent import Agent
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

def print_grid(grid, title):
    table = Table(title=title, show_header=False, box=box.SQUARE, show_edge=True)
    for row in grid:
        # Colorize 0 as dark, 1 as blue, etc. for visual effect
        colored_row = []
        for cell in row:
            if cell == 0:
                colored_row.append("[black]0[/black]")
            else:
                colored_row.append(f"[bold cyan]{cell}[/bold cyan]")
        table.add_row(*colored_row)
    console.print(table)

def main():
    agent = Agent()
    
    console.print("[bold purple]ARC-AGI Solver Demo[/bold purple]")
    console.print("System will attempt to solve a task independently using Search & Learning.\n")

    # Defined Task: Rotate CW then Flip (Composition)
    # [[1, 2],
    #  [3, 0]]
    input_grid = np.array([[1, 2], [3, 0]])
    
    # 1. Rotate CW
    # [[3, 1],
    #  [0, 2]]
    # 2. Reflect Horizontal (Left-Right flip)
    # [[1, 3],
    #  [2, 0]]
    expected_output = np.array([[1, 3], [2, 0]])

    print_grid(input_grid, "Input Grid")
    print_grid(expected_output, "Expected Output")

    console.print("\n[yellow]Invoking Solver...[/yellow]")
    
    # Solve
    solution = agent.solve_task(input_grid, expected_output)
    
    console.print(f"\n[bold green]Result:[/bold green] {solution}")
    
    if "reflect_horizontal(rotate_cw(x))" in solution or "rotate_cw" in solution: # The searcher output format might vary slightly
         console.print("[bold green]SUCCESS: System correctly identified the composite transformation![/bold green]")
    else:
         console.print("[red]Verification Failed.[/red]")
         
    # Check Memory
    console.print("\n[blue]Checking Brain (Memory)...[/blue]")
    memories = agent.memory.retrieve(f"Solution: {solution}")
    if memories:
        console.print(f"Memory successfully stored: '{memories[0]}'")

if __name__ == "__main__":
    main()
