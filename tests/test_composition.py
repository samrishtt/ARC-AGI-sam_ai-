"""
Test suite for DSL primitive composition.
Verifies that chaining primitives produces correct results,
and that the A* ProgramSearch can discover composite solutions.
"""
import numpy as np
import pytest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.dsl.primitives import rotate_cw, flip_horizontal, flip_vertical, rotate_ccw
from src.core.searcher import ProgramSearch


def test_rotate_then_flip():
    """
    Test composing: Rotate CW then Flip Horizontal.
    [[1, 2], [3, 4]]  →  rotate_cw  →  [[3, 1], [4, 2]]  →  flip_horizontal  →  [[1, 3], [2, 4]]
    """
    input_grid = np.array([[1, 2], [3, 4]])
    intermediate = rotate_cw(input_grid.tolist())
    expected = flip_horizontal(intermediate)

    # Verify the manual composition is correct
    assert expected == [[1, 3], [2, 4]]


def test_searcher_finds_single_primitive():
    """Test that the A* searcher can find a single-step rotation solution."""
    input_grid = np.array([[1, 0], [0, 2]])
    output_grid = np.array(rotate_cw(input_grid.tolist()))

    searcher = ProgramSearch(max_depth=2)
    solution = searcher.solve([(input_grid, output_grid)])

    assert solution is not None
    assert "rotate_cw" in solution


def test_searcher_finds_composite():
    """Test that the A* searcher can find a 2-step composite solution."""
    input_grid = np.array([[1, 2], [3, 4]])
    intermediate = rotate_cw(input_grid.tolist())
    target = np.array(flip_horizontal(intermediate))

    searcher = ProgramSearch(max_depth=3)
    solution = searcher.solve([(input_grid, target)])

    assert solution is not None
    assert "flip_horizontal" in solution
    assert "rotate_cw" in solution


def test_flip_roundtrip():
    """Test that flipping twice returns the original grid."""
    input_grid = [[5, 6, 7], [8, 9, 0]]
    result = flip_horizontal(flip_horizontal(input_grid))
    assert result == input_grid
