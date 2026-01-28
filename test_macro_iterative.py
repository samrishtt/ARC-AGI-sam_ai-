
import numpy as np
from src.core.agent import Agent

def test_macro_learning():
    agent = Agent()
    
    # Task 1: Simple Rotate (to learn it)
    input1 = np.array([[1, 2], [3, 4]])
    output1 = np.rot90(input1, k=-1) # rotate_cw
    
    print("Solving Task 1...")
    agent.solve_task(input1, output1) # This should store "Solution: rotate_cw(x)"
    
    # Task 2: Double Rotate (requires macro or depth 2)
    # If we use macro, it might find it as a single step if we are lucky or at least use it.
    input2 = np.array([[5, 6], [7, 8]])
    output2 = np.rot90(input2, k=-2)
    
    print("\nSolving Task 2 (should use macro rotate_cw)...")
    solution = agent.solve_task(input2, output2)
    print(f"Final Solution for Task 2: {solution}")

if __name__ == "__main__":
    test_macro_learning()
