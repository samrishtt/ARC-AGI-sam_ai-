from enum import Enum
from pydantic import BaseModel, Field

class TaskComplexity(str, Enum):
    LOW = "low"          # Simple questions, chitchat, direct facts
    MEDIUM = "medium"    # Requires some thought, standard math, summaries
    HIGH = "high"        # ARC-AGI puzzles, complex coding, multi-step logic

class TaskDomain(str, Enum):
    CONVERSATIONAL = "conversational"
    MATH_LOGIC = "math_logic"
    CODING = "coding"
    VISUAL_SPATIAL = "visual_spatial" # e.g. ARC grids

class RouteDecision(BaseModel):
    domain: TaskDomain = Field(..., description="The classified domain of the task.")
    complexity: TaskComplexity = Field(..., description="The estimated complexity of the task.")
    reasoning: str = Field(..., description="A short explanation of why this route was chosen. Important for transparency.")
    requires_tools: bool = Field(..., description="True if the task requires external tools (like a Python interpreter).")
