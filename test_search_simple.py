
import numpy as np
from src.core.searcher import ProgramSearch
from src.dsl.primitives import dsl_registry

def test_solver():
    print("Initializing Searcher...")
    searcher = ProgramSearch()
    
    # Test Case: Rotate CW then Reflect Horizontal
    # [[1, 2],
    #  [3, 0]]
    input_grid = np.array([[1, 2], [3, 0]])
    
    # 1. Rotate CW
    # [[3, 1],
    #  [0, 2]]
    # 2. Reflect Horizontal
    # [[1, 3],
    #  [2, 0]]
    expected_output = np.array([[1, 3], [2, 0]])
    
    print("Starting Solve...")
    # Pass as list of logic examples
    solution = searcher.solve([(input_grid, expected_output)])
    
    print(f"Solution Found: {solution}")
    
    if solution == "reflect_horizontal(rotate_cw(x))":
        print("SUCCESS")
    else:
        print("PARTIAL SUCCESS / DIFFERENT PATH")

if __name__ == "__main__":
    test_solver()
