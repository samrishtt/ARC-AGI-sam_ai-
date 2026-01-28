
import numpy as np
from typing import List, Tuple, Dict
from .llm import LLMProvider

class Reflector:
    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def reflect(self, inputs: List[np.ndarray], targets: List[np.ndarray], attempted_primitives: List[str]) -> str:
        """
        Analyze why the search failed and suggest a conceptual direction.
        """
        prompt = f"""
        An ARC-AGI solver failed to find a transformation.
        Input Grid Sample: {inputs[0].tolist()}
        Target Grid Sample: {targets[0].tolist()}
        Attempted primitives included: {attempted_primitives[:10]}...
        
        Analyze the difference between input and output.
        Is it a color change? A movement? A scaling?
        Provide a short 'Reasoning Hint' for the next iteration.
        """
        
        response = self.llm.generate("You are a reasoning reflector.", prompt)
        return response.content
