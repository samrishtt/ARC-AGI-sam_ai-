import sys
import os
import pytest
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.agent import Agent

def test_agent_initialization():
    agent = Agent()
    assert agent.name == "Sam"
    assert "python_sandbox" in agent.tools

def test_sandbox_execution():
    agent = Agent()
    response = agent.process("run: print('test execution')")
    assert "Execution Successful" in response
    assert "test execution" in response

def test_chat_response():
    agent = Agent()
    response = agent.process("hello")
    assert "[MOCK]" in response
