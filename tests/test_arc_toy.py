"""
Test suite for ARC-style grid transformations using DSL primitives
and the CSA visual-spatial pipeline.
"""
import numpy as np
import pytest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.dsl.primitives import rotate_cw, rotate_ccw, flip_horizontal, flip_vertical
from src.csa.coding import PythonSandbox


def test_rotate_cw_primitive():
    """Test that rotate_cw correctly rotates a 2x2 grid."""
    input_grid = [[1, 0], [0, 1]]
    expected = [[0, 1], [1, 0]]
    result = rotate_cw(input_grid)
    assert result == expected


def test_rotate_ccw_primitive():
    """Test that rotate_ccw correctly rotates a 2x2 grid."""
    input_grid = [[1, 0], [0, 1]]
    expected = [[0, 1], [1, 0]]
    result = rotate_ccw(input_grid)
    assert result == expected


def test_rotation_roundtrip():
    """Test that 4x rotate_cw returns the original grid."""
    input_grid = [[1, 2], [3, 4]]
    result = input_grid
    for _ in range(4):
        result = rotate_cw(result)
    assert result == input_grid


def test_sandbox_runs_dsl_code():
    """Test that the sandbox can import and run DSL primitives."""
    code = (
        "import sys, os\n"
        "sys.path.append(os.getcwd())\n"
        "from src.dsl.primitives import rotate_cw\n"
        "grid = [[1, 0], [0, 1]]\n"
        "print(rotate_cw(grid))\n"
    )
    success, output = PythonSandbox.run_code(code)
    assert success is True
    assert "[[0, 1]" in output or "[[0 1]" in output  # numpy vs list formatting


def test_flip_horizontal_primitive():
    """Test that flip_horizontal correctly flips a grid left-right."""
    input_grid = [[1, 2], [3, 4]]
    expected = [[2, 1], [4, 3]]
    result = flip_horizontal(input_grid)
    assert result == expected


def test_flip_vertical_primitive():
    """Test that flip_vertical correctly flips a grid top-bottom."""
    input_grid = [[1, 2], [3, 4]]
    expected = [[3, 4], [1, 2]]
    result = flip_vertical(input_grid)
    assert result == expected
