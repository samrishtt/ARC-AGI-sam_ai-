import numpy as np
import pytest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.agent import Agent
from src.dsl.primitives import rotate_cw, reflect_horizontal

def test_composition_search():
    """
    Test finding a composite solution: Rotate CW then Flip Horizontal
    """
    input_A = np.array([[1, 2], [3, 4]])
    
    # Expected: Rotate CW -> [[3, 1], [4, 2]] -> Reflect H -> [[1, 3], [2, 4]]
    # Wait, fliplr on [[3, 1], [4, 2]]:
    # Row 0: 3, 1 -> 1, 3
    # Row 1: 4, 2 -> 2, 4
    # Result: [[1, 3], [2, 4]]
    
    intermediate = rotate_cw(input_A)
    expected_A = reflect_horizontal(intermediate)
    
    agent = Agent()
    solution = agent.solve_task(input_A, expected_A)
    
    assert "reflect_horizontal" in solution
    assert "rotate_cw" in solution
