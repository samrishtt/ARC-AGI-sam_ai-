import sys
import os

# Add src to python path so imports work
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.agent import Agent
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel

console = Console()

def main():
    console.print(Panel.fit("[bold green]Welcome to Arc-AGI (Sam-AI 2.0)[/bold green]\n[italic]Iterative. Robust. Powerful.[/italic]"))
    
    agent = Agent(name="Sam")
    
    while True:
        try:
            user_input = Prompt.ask("[bold yellow]You[/bold yellow]")
            if user_input.lower() in ["exit", "quit", "q"]:
                console.print("[red]Shutting down...[/red]")
                break
            
            response = agent.process(user_input)
            console.print(Panel(response, title=f"[bold blue]{agent.name}[/bold blue]", border_style="blue"))
            
        except KeyboardInterrupt:
            console.print("\n[red]Interrupted. Exiting.[/red]")
            break
        except Exception as e:
            console.print(f"[bold red]Error: {e}[/bold red]")

if __name__ == "__main__":
    main()
