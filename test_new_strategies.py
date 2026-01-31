"""
Test New Strategies: Parametric Color & Grid Subdivision

Verifies that the new strategies can solve tasks that:
1. Involve color mapping (A->B, B->A, etc.)
2. Involve splitting/merging grids
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.arc.super_reasoning import SuperReasoningEngine, Strategy
from rich.console import Console

console = Console()

def print(text):
    console.print(text)
    
    # Task: Swap Red (2) and Blue (1)
    train = [
        (
            np.array([[1, 2, 0], [2, 1, 0]]), 
            np.array([[2, 1, 0], [1, 2, 0]])
        ),
        (
            np.array([[0, 0, 1], [2, 2, 2]]), 
            np.array([[0, 0, 2], [1, 1, 1]])
        )
    ]
    test_input = np.array([[1, 1, 1], [0, 2, 0]])
    expected = np.array([[2, 2, 2], [0, 1, 0]])
    
    engine = SuperReasoningEngine()
    result = engine.solve(train, test_input, "test_color_swap")
    
    if result.success and result.strategy == Strategy.PARAMETRIC_COLOR:
        print("  ✓ Color Swap: PASSED")
        print(f"    Description: {result.description}")
        return True
    else:
        print(f"  ✗ Color Swap: FAILED (Strategy: {result.strategy})")
        return False

def test_grid_subdivision():
    print("\nTesting Grid Subdivision Strategy...")
    
    # Task: Take top half of grid
    train = [
        (
            np.array([[1, 1], [1, 1], [2, 2], [2, 2]]), 
            np.array([[1, 1], [1, 1]])
        ),
        (
            np.array([[3, 3], [3, 3], [4, 4], [4, 4]]), 
            np.array([[3, 3], [3, 3]])
        )
    ]
    test_input = np.array([[5, 5], [5, 5], [6, 6], [6, 6]])
    expected = np.array([[5, 5], [5, 5]])
    
    engine = SuperReasoningEngine()
    result = engine.solve(train, test_input, "test_top_half")
    
    if result.success and result.strategy == Strategy.GRID_SUBDIVISION:
        print("  ✓ Top Half: PASSED")
        print(f"    Description: {result.description}")
        return True
    else:
        print(f"  ✗ Top Half: FAILED (Strategy: {result.strategy})")
        return False

def test_grid_overlay():
    print("\nTesting Grid Overlay Strategy...")
    
    # Task: Overlay left and right halves (Union)
    train = [
        (
            # Left has 1s, Right has 2s -> Result has both
            np.array([[1, 0, 0, 2], [0, 0, 0, 0]]), 
            np.array([[1, 2], [0, 0]])
        ),
        (
            np.array([[3, 0, 0, 0], [0, 0, 0, 4]]), 
            np.array([[3, 0], [0, 4]])
        )
    ]
    test_input = np.array([[5, 0, 0, 0], [0, 0, 0, 0]])
    expected = np.array([[5, 0], [0, 0]])
    
    engine = SuperReasoningEngine()
    result = engine.solve(train, test_input, "test_overlay")
    
    if result.success and result.strategy == Strategy.GRID_SUBDIVISION:
        print("  ✓ Overlay: PASSED")
        print(f"    Description: {result.description}")
        return True
    else:
        print(f"  ✗ Overlay: FAILED (Strategy: {result.strategy})")
        return False

if __name__ == "__main__":
    print("="*50)
    print("TESTING NEW STRATEGIES")
    print("="*50)
    
    p1 = test_parametric_color()
    p2 = test_grid_subdivision()
    p3 = test_grid_overlay()
    
    print("\n" + "="*50)
    if p1 and p2 and p3:
        print("✓ ALL NEW STRATEGIES WORKING")
    else:
        print("✗ SOME TESTS FAILED")
