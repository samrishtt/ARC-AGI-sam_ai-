"""
LLM-Guided Reasoning for ARC-AGI

Uses Large Language Models to:
1. Generate hypotheses in natural language
2. Guide program synthesis
3. Explain transformations 
4. Learn from failures

This enables the solver to handle complex, novel patterns
that pure symbolic methods miss.

Supports multiple LLM backends:
- OpenAI API
- Local models via Ollama
- Mock mode for testing
"""

import os
import json
import re
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum


class LLMBackend(Enum):
    """Available LLM backends."""
    OPENAI = "openai"
    OLLAMA = "ollama"
    MOCK = "mock"


@dataclass
class Hypothesis:
    """A natural language hypothesis about a transformation."""
    description: str
    confidence: float
    suggested_operations: List[str]
    reasoning: str


@dataclass
class LLMResponse:
    """Response from LLM."""
    success: bool
    content: str
    hypotheses: List[Hypothesis]
    error: Optional[str] = None


class LLMReasoner:
    """
    Uses LLM to reason about ARC transformations.
    
    Can work with various backends or in mock mode.
    """
    
    def __init__(
        self, 
        backend: LLMBackend = LLMBackend.MOCK,
        api_key: Optional[str] = None,
        model: str = "gpt-4"
    ):
        self.backend = backend
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        
        # System prompt for ARC reasoning
        self.system_prompt = """You are an expert at solving ARC-AGI (Abstraction and Reasoning Corpus) puzzles.

Given input→output grid pairs, you analyze what transformation is being applied.

ARC grids use colors 0-9:
- 0: black (background)
- 1: blue, 2: red, 3: green, 4: yellow, 5: gray, 6: magenta, 7: orange, 8: cyan, 9: maroon

Common transformations include:
- Geometric: rotation, reflection, transpose, scaling
- Color: replacement, inversion, mapping
- Object: keep largest, filter by color, sort
- Structural: crop, tile, fill, pad
- Logical: AND, OR, XOR of patterns

Respond with a JSON object containing:
{
    "description": "Brief description of the transformation",
    "confidence": 0.0-1.0,
    "operations": ["list", "of", "primitive", "operations"],
    "reasoning": "Step by step reasoning"
}"""
    
    def analyze_task(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> LLMResponse:
        """
        Analyze an ARC task and generate hypotheses.
        
        Args:
            examples: List of (input, output) training pairs
            
        Returns:
            LLMResponse with hypotheses
        """
        if self.backend == LLMBackend.MOCK:
            return self._mock_analyze(examples)
        elif self.backend == LLMBackend.OPENAI:
            return self._openai_analyze(examples)
        elif self.backend == LLMBackend.OLLAMA:
            return self._ollama_analyze(examples)
        else:
            return LLMResponse(
                success=False,
                content="",
                hypotheses=[],
                error=f"Unknown backend: {self.backend}"
            )
    
    def _format_grid(self, grid: np.ndarray) -> str:
        """Format a grid for LLM prompt."""
        lines = []
        for row in grid:
            lines.append(" ".join(str(c) for c in row))
        return "\n".join(lines)
    
    def _format_examples(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> str:
        """Format examples for LLM prompt."""
        parts = []
        for i, (inp, out) in enumerate(examples):
            parts.append(f"Example {i+1}:")
            parts.append(f"Input ({inp.shape[0]}x{inp.shape[1]}):")
            parts.append(self._format_grid(inp))
            parts.append(f"Output ({out.shape[0]}x{out.shape[1]}):")
            parts.append(self._format_grid(out))
            parts.append("")
        return "\n".join(parts)
    
    def _mock_analyze(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> LLMResponse:
        """Mock analysis for testing without LLM."""
        # Analyze patterns ourselves
        hypotheses = []
        
        if not examples:
            return LLMResponse(success=True, content="", hypotheses=[])
        
        inp, out = examples[0]
        
        # Check for common patterns
        # Same size - likely pixel-wise operation
        if inp.shape == out.shape:
            # Check rotation
            if np.array_equal(np.rot90(inp, -1), out):
                hypotheses.append(Hypothesis(
                    description="90 degree clockwise rotation",
                    confidence=0.95,
                    suggested_operations=["rotate_cw"],
                    reasoning="Output is input rotated 90 degrees clockwise"
                ))
            
            # Check reflection
            if np.array_equal(np.fliplr(inp), out):
                hypotheses.append(Hypothesis(
                    description="Horizontal reflection",
                    confidence=0.95,
                    suggested_operations=["reflect_h"],
                    reasoning="Output is input flipped left-to-right"
                ))
            
            if np.array_equal(np.flipud(inp), out):
                hypotheses.append(Hypothesis(
                    description="Vertical reflection",
                    confidence=0.95,
                    suggested_operations=["reflect_v"],
                    reasoning="Output is input flipped top-to-bottom"
                ))
        
        # Size doubled - likely scaling
        if out.shape[0] == inp.shape[0] * 2 and out.shape[1] == inp.shape[1] * 2:
            hypotheses.append(Hypothesis(
                description="2x scaling",
                confidence=0.9,
                suggested_operations=["scale_2x"],
                reasoning="Output size is exactly 2x input in both dimensions"
            ))
        
        # Output smaller - likely crop
        if out.shape[0] < inp.shape[0] or out.shape[1] < inp.shape[1]:
            hypotheses.append(Hypothesis(
                description="Cropping operation",
                confidence=0.7,
                suggested_operations=["crop"],
                reasoning="Output is smaller than input, likely cropped"
            ))
        
        return LLMResponse(
            success=True,
            content="Mock analysis complete",
            hypotheses=hypotheses
        )
    
    def _openai_analyze(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> LLMResponse:
        """Analyze using OpenAI API."""
        try:
            import openai
            
            if not self.api_key:
                return LLMResponse(
                    success=False,
                    content="",
                    hypotheses=[],
                    error="OpenAI API key not set"
                )
            
            client = openai.OpenAI(api_key=self.api_key)
            
            user_prompt = f"""Analyze the following ARC-AGI task and determine the transformation rule:

{self._format_examples(examples)}

What transformation converts each input to its corresponding output?"""
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content
            
            # Parse JSON from response
            hypotheses = self._parse_response(content)
            
            return LLMResponse(
                success=True,
                content=content,
                hypotheses=hypotheses
            )
            
        except ImportError:
            return LLMResponse(
                success=False,
                content="",
                hypotheses=[],
                error="openai package not installed"
            )
        except Exception as e:
            return LLMResponse(
                success=False,
                content="",
                hypotheses=[],
                error=str(e)
            )
    
    def _ollama_analyze(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> LLMResponse:
        """Analyze using local Ollama model."""
        try:
            import requests
            
            user_prompt = f"""Analyze the following ARC-AGI task and determine the transformation rule:

{self._format_examples(examples)}

What transformation converts each input to its corresponding output?"""
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model,
                    "prompt": f"{self.system_prompt}\n\n{user_prompt}",
                    "stream": False
                },
                timeout=60
            )
            
            if response.status_code == 200:
                content = response.json().get("response", "")
                hypotheses = self._parse_response(content)
                
                return LLMResponse(
                    success=True,
                    content=content,
                    hypotheses=hypotheses
                )
            else:
                return LLMResponse(
                    success=False,
                    content="",
                    hypotheses=[],
                    error=f"Ollama error: {response.status_code}"
                )
                
        except Exception as e:
            return LLMResponse(
                success=False,
                content="",
                hypotheses=[],
                error=str(e)
            )
    
    def _parse_response(self, content: str) -> List[Hypothesis]:
        """Parse LLM response to extract hypotheses."""
        hypotheses = []
        
        # Try to find JSON in response
        json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                hypotheses.append(Hypothesis(
                    description=data.get("description", ""),
                    confidence=float(data.get("confidence", 0.5)),
                    suggested_operations=data.get("operations", []),
                    reasoning=data.get("reasoning", "")
                ))
            except json.JSONDecodeError:
                pass
        
        # If no JSON found, extract key info from text
        if not hypotheses:
            hypotheses.append(Hypothesis(
                description=content[:200] if content else "Unknown",
                confidence=0.5,
                suggested_operations=[],
                reasoning=content
            ))
        
        return hypotheses
    
    def suggest_operations(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[str]:
        """
        Get suggested DSL operations for a task.
        
        Returns list of primitive names to try.
        """
        response = self.analyze_task(examples)
        
        if response.success and response.hypotheses:
            # Collect all suggested operations
            all_ops = []
            for h in response.hypotheses:
                all_ops.extend(h.suggested_operations)
            return all_ops
        
        return []
    
    def explain_transformation(
        self,
        input_grid: np.ndarray,
        output_grid: np.ndarray
    ) -> str:
        """
        Generate human-readable explanation of a transformation.
        """
        response = self.analyze_task([(input_grid, output_grid)])
        
        if response.success and response.hypotheses:
            h = response.hypotheses[0]
            return f"{h.description}\n\nReasoning: {h.reasoning}"
        
        return "Unable to explain transformation"


class LLMGuidedSearch:
    """
    Uses LLM to guide the search for solutions.
    
    The LLM suggests which operations to try first,
    dramatically reducing search space.
    """
    
    def __init__(self, reasoner: Optional[LLMReasoner] = None):
        self.reasoner = reasoner or LLMReasoner(backend=LLMBackend.MOCK)
    
    def prioritize_operations(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        all_operations: List[str]
    ) -> List[str]:
        """
        Reorder operations based on LLM suggestions.
        
        Puts LLM-suggested operations first.
        """
        suggested = self.reasoner.suggest_operations(examples)
        
        # Put suggested operations first
        prioritized = []
        remaining = list(all_operations)
        
        for op in suggested:
            if op in remaining:
                prioritized.append(op)
                remaining.remove(op)
        
        # Add remaining operations
        prioritized.extend(remaining)
        
        return prioritized
    
    def guided_solve(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        operations: Dict[str, callable],
        max_depth: int = 3
    ) -> Optional[Tuple[np.ndarray, List[str]]]:
        """
        Solve using LLM-guided search.
        
        Returns (prediction, operation_chain) if successful.
        """
        # Get prioritized operations
        op_names = self.prioritize_operations(examples, list(operations.keys()))
        
        # Try single operations
        for name in op_names:
            try:
                func = operations[name]
                success = True
                
                for inp, out in examples:
                    result = func(inp)
                    if not np.array_equal(result, out):
                        success = False
                        break
                
                if success:
                    prediction = func(test_input)
                    return prediction, [name]
                    
            except Exception:
                continue
        
        # Try compositions if single ops failed
        if max_depth >= 2:
            for name1 in op_names[:20]:  # Limit for performance
                for name2 in op_names[:20]:
                    try:
                        def composed(g, f1=operations[name1], f2=operations[name2]):
                            return f2(f1(g))
                        
                        success = True
                        for inp, out in examples:
                            if not np.array_equal(composed(inp), out):
                                success = False
                                break
                        
                        if success:
                            return composed(test_input), [name1, name2]
                            
                    except Exception:
                        continue
        
        return None


# Convenience function
def get_llm_hypotheses(
    examples: List[Tuple[np.ndarray, np.ndarray]],
    backend: str = "mock"
) -> List[Hypothesis]:
    """Get LLM hypotheses for an ARC task."""
    backend_enum = LLMBackend(backend)
    reasoner = LLMReasoner(backend=backend_enum)
    response = reasoner.analyze_task(examples)
    return response.hypotheses
