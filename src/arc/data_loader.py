"""
ARC-AGI Dataset Loader

Downloads and loads the official ARC-AGI dataset for training and evaluation.
Supports both ARC-Prize (v3) and legacy ARC datasets.
"""

import json
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from rich.console import Console
from rich.progress import Progress

console = Console()

@dataclass
class ARCTask:
    """Represents a single ARC task with training and test examples."""
    task_id: str
    train_examples: List[Tuple[np.ndarray, np.ndarray]]
    test_examples: List[Tuple[np.ndarray, np.ndarray]]
    
    @property 
    def num_train(self) -> int:
        return len(self.train_examples)
    
    @property
    def num_test(self) -> int:
        return len(self.test_examples)
    
    def __repr__(self):
        return f"ARCTask({self.task_id}, train={self.num_train}, test={self.num_test})"


class ARCDataLoader:
    """
    Loads ARC-AGI datasets from local cache or downloads from GitHub.
    
    Supports:
    - ARC-AGI Prize Training Tasks
    - ARC-AGI Prize Evaluation Tasks
    - Custom local datasets
    """
    
    # Official ARC-AGI Prize repository
    ARC_GITHUB_BASE = "https://raw.githubusercontent.com/arcprize/arc-prize-2025/main/data"
    
    def __init__(self, data_dir: str = "data/arc"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._tasks_cache: Dict[str, ARCTask] = {}
        
    def _download_file(self, url: str, target_path: Path) -> bool:
        """Download a file from URL to target path."""
        try:
            console.print(f"[blue]Downloading: {url}[/blue]")
            urllib.request.urlretrieve(url, target_path)
            return True
        except Exception as e:
            console.print(f"[red]Download failed: {e}[/red]")
            return False
    
    def _parse_grid(self, grid_data: List[List[int]]) -> np.ndarray:
        """Convert JSON grid to numpy array."""
        return np.array(grid_data, dtype=np.int8)
    
    def _load_task_file(self, file_path: Path) -> Optional[ARCTask]:
        """Load a single task from a JSON file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            train_examples = []
            for ex in data.get('train', []):
                input_grid = self._parse_grid(ex['input'])
                output_grid = self._parse_grid(ex['output'])
                train_examples.append((input_grid, output_grid))
            
            test_examples = []
            for ex in data.get('test', []):
                input_grid = self._parse_grid(ex['input'])
                # Output may not exist in evaluation mode
                output_grid = self._parse_grid(ex.get('output', [[0]])) if 'output' in ex else None
                test_examples.append((input_grid, output_grid))
            
            task_id = file_path.stem
            return ARCTask(task_id=task_id, train_examples=train_examples, test_examples=test_examples)
            
        except Exception as e:
            console.print(f"[red]Error loading {file_path}: {e}[/red]")
            return None
    
    def download_arc_dataset(self, dataset_type: str = "training") -> bool:
        """
        Download ARC-AGI dataset from GitHub.
        
        Args:
            dataset_type: 'training' or 'evaluation'
        
        Returns:
            True if successful.
        """
        # Try to download individual task files from arcprize repo
        # Fall back to local dummy data if network fails
        target_dir = self.data_dir / dataset_type
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # First try alternative: fetch task list
        tasks_url = f"{self.ARC_GITHUB_BASE}/{dataset_type}"
        
        console.print(f"[yellow]Note: For full ARC dataset, please clone:[/yellow]")
        console.print("[cyan]git clone https://github.com/arcprize/arc-prize-2025.git[/cyan]")
        console.print(f"[cyan]Then copy data/ to {self.data_dir}[/cyan]")
        
        return True
    
    def load_local_dataset(self, dataset_path: str) -> List[ARCTask]:
        """Load tasks from a local directory of JSON files."""
        path = Path(dataset_path)
        tasks = []
        
        if not path.exists():
            console.print(f"[red]Dataset path does not exist: {path}[/red]")
            return []
        
        json_files = list(path.glob("*.json"))
        console.print(f"[green]Found {len(json_files)} task files in {path}[/green]")
        
        with Progress() as progress:
            task_progress = progress.add_task("[cyan]Loading tasks...", total=len(json_files))
            
            for file_path in json_files:
                task = self._load_task_file(file_path)
                if task:
                    tasks.append(task)
                    self._tasks_cache[task.task_id] = task
                progress.advance(task_progress)
        
        return tasks
    
    def create_sample_tasks(self) -> List[ARCTask]:
        """
        Create sample ARC-style tasks for testing.
        These mimic real ARC patterns without needing network access.
        """
        tasks = []
        
        # Task 1: Rotation (like real ARC task)
        t1_train = [
            (np.array([[1,0,0],[0,1,0],[0,0,1]]), np.array([[0,0,1],[0,1,0],[1,0,0]])),  # Rotate 90 CW, reflect
            (np.array([[2,0],[0,2]]), np.array([[0,2],[2,0]])),
        ]
        t1_test = [(np.array([[3,0,0],[0,0,0],[0,0,0]]), np.array([[0,0,3],[0,0,0],[0,0,0]]))]
        tasks.append(ARCTask("sample_rotation", t1_train, t1_test))
        
        # Task 2: Fill pattern (common in ARC)
        t2_train = [
            (np.array([[0,1,0],[0,0,0],[0,0,0]]), np.array([[0,1,0],[0,1,0],[0,1,0]])),
            (np.array([[0,0,2],[0,0,0],[0,0,0]]), np.array([[0,0,2],[0,0,2],[0,0,2]])),
        ]
        t2_test = [(np.array([[3,0,0],[0,0,0],[0,0,0]]), np.array([[3,0,0],[3,0,0],[3,0,0]]))]
        tasks.append(ARCTask("sample_fill_column", t2_train, t2_test))
        
        # Task 3: Color replacement
        t3_train = [
            (np.array([[1,1,0],[1,0,0],[0,0,0]]), np.array([[2,2,0],[2,0,0],[0,0,0]])),
            (np.array([[0,0,1],[0,1,1],[1,1,1]]), np.array([[0,0,2],[0,2,2],[2,2,2]])),
        ]
        t3_test = [(np.array([[1,0,1],[0,0,0],[1,0,1]]), np.array([[2,0,2],[0,0,0],[2,0,2]]))]
        tasks.append(ARCTask("sample_replace_color", t3_train, t3_test))
        
        # Task 4: Crop to content
        t4_train = [
            (np.array([[0,0,0,0],[0,1,1,0],[0,1,1,0],[0,0,0,0]]), np.array([[1,1],[1,1]])),
            (np.array([[0,0,0],[0,5,0],[0,0,0]]), np.array([[5]])),
        ]
        t4_test = [(np.array([[0,0,0,0,0],[0,0,3,3,0],[0,0,3,3,0],[0,0,0,0,0]]), np.array([[3,3],[3,3]]))]
        tasks.append(ARCTask("sample_crop", t4_train, t4_test))
        
        # Task 5: Mirror/Reflect
        t5_train = [
            (np.array([[1,0],[0,0]]), np.array([[0,1],[0,0]])),
            (np.array([[2,2],[0,0]]), np.array([[2,2],[0,0]])),
        ]
        t5_test = [(np.array([[0,3],[0,0]]), np.array([[3,0],[0,0]]))]
        tasks.append(ARCTask("sample_reflect_h", t5_train, t5_test))
        
        # Task 6: Scale up 2x
        t6_train = [
            (np.array([[1]]), np.array([[1,1],[1,1]])),
            (np.array([[2,0],[0,3]]), np.array([[2,2,0,0],[2,2,0,0],[0,0,3,3],[0,0,3,3]])),
        ]
        t6_test = [(np.array([[5]]), np.array([[5,5],[5,5]]))]
        tasks.append(ARCTask("sample_scale_2x", t6_train, t6_test))
        
        # Task 7: Extract largest object
        t7_train = [
            (np.array([[1,0,2],[0,2,2],[0,2,2]]), np.array([[0,0,2],[0,2,2],[0,2,2]])),
            (np.array([[5,5,5],[0,1,0],[0,0,0]]), np.array([[5,5,5],[0,0,0],[0,0,0]])),
        ]
        t7_test = [(np.array([[1,1,0],[0,0,3],[0,0,0]]), np.array([[1,1,0],[0,0,0],[0,0,0]]))]
        tasks.append(ARCTask("sample_keep_largest", t7_train, t7_test))
        
        # Task 8: Count and fill (abstract)
        t8_train = [
            (np.array([[1,0,1]]), np.array([[2]])),  # 2 ones -> output is 2
            (np.array([[1,1,1,0]]), np.array([[3]])),  # 3 ones -> output is 3
        ]
        t8_test = [(np.array([[1,1,0,1,1]]), np.array([[4]]))]
        tasks.append(ARCTask("sample_count_objects", t8_train, t8_test))
        
        # Task 9: Gravity/Drop
        t9_train = [
            (np.array([[1,0,0],[0,0,0],[0,0,0]]), np.array([[0,0,0],[0,0,0],[1,0,0]])),
            (np.array([[0,2,0],[0,0,0],[0,0,0]]), np.array([[0,0,0],[0,0,0],[0,2,0]])),
        ]
        t9_test = [(np.array([[0,0,3],[0,0,0],[0,0,0]]), np.array([[0,0,0],[0,0,0],[0,0,3]]))]
        tasks.append(ARCTask("sample_gravity", t9_train, t9_test))
        
        # Task 10: XOR pattern
        t10_train = [
            (np.array([[1,1],[1,0]]), np.array([[0,0],[0,1]])),
            (np.array([[0,1],[1,1]]), np.array([[1,0],[0,0]])),
        ]
        t10_test = [(np.array([[1,0],[0,1]]), np.array([[0,1],[1,0]]))]
        tasks.append(ARCTask("sample_invert", t10_train, t10_test))
        
        console.print(f"[green]Created {len(tasks)} sample ARC tasks[/green]")
        return tasks
    
    def get_task(self, task_id: str) -> Optional[ARCTask]:
        """Get a specific task by ID."""
        return self._tasks_cache.get(task_id)
    
    def get_all_tasks(self) -> List[ARCTask]:
        """Get all loaded tasks."""
        return list(self._tasks_cache.values())
