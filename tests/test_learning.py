"""
Test suite for the learning & memory subsystems.
Tests ProgramSearch solution discovery and WorkingMemory persistence.
"""
import numpy as np
import pytest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.dsl.primitives import rotate_cw
from src.core.searcher import ProgramSearch
from src.csa.memory import WorkingMemory


def test_searcher_finds_rotation():
    """
    Test that the ProgramSearch can:
    1. Find a rotation solution for a simple grid.
    """
    input_grid = np.array([[1, 0], [0, 2]])
    output_grid = np.array(rotate_cw(input_grid.tolist()))

    searcher = ProgramSearch(max_depth=2)
    solution = searcher.solve([(input_grid, output_grid)])
    assert solution == "rotate_cw(x)"


def test_working_memory_steps():
    """
    Test that WorkingMemory correctly tracks reasoning steps.
    """
    memory = WorkingMemory()
    memory.set_context("Test task")
    memory.add_step("Grid Extraction", "Extracted 2 training pairs")
    memory.add_step("Hypothesis", "Rotation detected")

    summary = memory.get_summary()
    assert "Test task" in summary
    assert "Grid Extraction" in summary
    assert "Hypothesis" in summary
    assert len(memory.steps) == 2


def test_working_memory_observations():
    """
    Test that WorkingMemory correctly tracks per-pair observations.
    """
    memory = WorkingMemory()
    memory.add_observation(0, "Object delta: +1")
    memory.add_observation(0, "Dim changed: True")
    memory.add_observation(1, "Object delta: 0")

    observations = memory.get_all_observations()
    assert len(observations[0]) == 2
    assert len(observations[1]) == 1
    assert "Object delta: +1" in observations[0]


def test_working_memory_clear():
    """
    Test that WorkingMemory clear() resets all state.
    """
    memory = WorkingMemory()
    memory.set_context("Old context")
    memory.add_step("Step", "Data")
    memory.add_observation(0, "Note")
    memory.set_variable("key", "value")

    memory.clear()
    assert memory.context == ""
    assert len(memory.steps) == 0
    assert len(memory.observations) == 0
    assert len(memory.variables) == 0


def test_searcher_macro_loading():
    """
    Test that ProgramSearch can load learned macros from memory.
    """
    searcher = ProgramSearch(max_depth=2)
    initial_count = len(searcher.primitives)

    # Simulate loading a previously learned macro
    searcher.load_macros(["rotate_cw(rotate_cw(x))"])

    # After loading, we should have more primitives
    assert len(searcher.primitives) > initial_count
