import numpy as np
import pytest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.agent import Agent
from src.dsl.primitives import rotate_cw

def test_independent_learning_loop():
    """
    Test that the agent can:
    1. Search for a solution (Rotation)
    2. Learn it (Store in Memory)
    3. Recall it (Implicitly via retrieval check)
    """
    # Grid A -> Grid B (Rotation)
    input_A = np.array([[1, 0], [0, 0]])
    output_A = np.array([[0, 1], [0, 0]]) # Rotated 90 CW + appropriate translation if needed, 
                                         # actually rot90(k=-1) of [[1,0],[0,0]] is [[0,1],[0,0]]?
                                         # [[1,0],[0,0]] -> rot -> [[0,0],[1,0]]? Wait.
                                         # let's trust the searcher to find the right one.
    
    # Let's use a clear rotation
    # 1 0
    # 0 2
    # rot CW ->
    # 0 1
    # 2 0
    input_A = np.array([[1, 0], [0, 2]])
    output_A = rotate_cw(input_A)
    
    agent = Agent()
    
    # 1. Search & Learn
    # Should find 'rotate_cw' and store it
    solution = agent.solve_task(input_A, output_A)
    assert solution == "rotate_cw"
    
    # 2. Verify Memory
    # The agent should have stored "Solution: rotate_cw"
    # We can check by retrieving
    memories = agent.memory.retrieve("Solution: rotate_cw")
    assert len(memories) > 0
    assert "rotate_cw" in memories[0]
