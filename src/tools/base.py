from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Any, Dict

class ToolResult(BaseModel):
    success: bool
    output: str
    metadata: Dict[str, Any] = {}

class BaseTool(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        pass
