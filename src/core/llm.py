from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import os
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class LLMResponse(BaseModel):
    content: str
    token_usage: Dict[str, int] = {}

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        pass

class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = "gpt-4-turbo"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            return LLMResponse(
                content=response.choices[0].message.content,
                token_usage=response.usage.model_dump()
            )
        except Exception as e:
            return LLMResponse(content=f"Error: {str(e)}")

class MockLLMProvider(LLMProvider):
    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        # Smart Mocking for Demo/Tests
        
        # If asking for Python code
        if "python code" in system_prompt.lower():
            if "rotate" in user_prompt.lower():
                 return LLMResponse(content="```python\nimport numpy as np\nfrom src.dsl.primitives import rotate_cw\n\n# Mock solution\nprint('Rotation Logic Here')\n```")
            return LLMResponse(content="```python\nprint('Hello from Mock Code')\n```")
            
        # If asking for a Plan (JSON)
        if "valid JSON" in system_prompt:
             # If it looks like a chat message
             if "hello" in user_prompt.lower():
                 return LLMResponse(content='{"steps": [{"action": "ANALYZE", "description": "User greeting"}]}')
             
             # Default to transformation plan
             return LLMResponse(content='{"steps": [{"action": "ANALYZE", "description": "Analyze grid"}, {"action": "TRANSFORM", "description": "Apply rotation"}, {"action": "VERIFY", "description": "Check result"}]}')

        return LLMResponse(content=f"[MOCK] Processed: {user_prompt[:50]}...")
