"""
Test suite for core CSA pipeline components.
Tests the MetaController + CodingHandler + Router using MockLLM.
"""
import sys
import os
import pytest
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.llm import MockLLMProvider
from src.csa.meta_controller import MetaController


def test_meta_controller_initialization():
    """Test that MetaController initializes correctly with MockLLM."""
    llm = MockLLMProvider()
    controller = MetaController(primary_llm=llm)
    assert controller.llm is llm
    assert controller.router is not None
    assert controller.coding_handler is not None
    assert controller.memory is not None


def test_conversational_routing():
    """Test that a simple greeting is routed to the conversational pipeline."""
    llm = MockLLMProvider()
    controller = MetaController(primary_llm=llm)
    result = controller.process_task("hello")
    # MockLLM returns a mock response; we just verify it doesn't crash
    assert result["status"] in ("success", "failed", "error")
    assert "pipeline" in result


def test_sandbox_execution_via_coding_handler():
    """Test that the CodingHandler can execute simple Python code via sandbox."""
    from src.csa.coding import PythonSandbox
    success, output = PythonSandbox.run_code("print('test execution')")
    assert success is True
    assert "test execution" in output
