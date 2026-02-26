import json
from src.core.llm import LLMProvider
from src.csa.models import RouteDecision, TaskDomain, TaskComplexity

class IntentRouter:
    """
    The IntentRouter is responsible for taking a user's raw input
    and categorizing it according to our taxonomy. This is the first step
    in the Meta-Controller pipeline.
    """
    def __init__(self, llm: LLMProvider):
        self.llm = llm
        
        # We explicitly teach the LLM the JSON structure we want.
        self.system_prompt = """
You are the routing engine for an advanced Cognitive Synthesis Architecture (CSA).
Your job is to analyze the user's input and categorize it strictly into a JSON structure.
Do NOT solve the problem. Only classify it.

Here are the domains you can choose from:
- "conversational": General chitchat, facts, normal queries.
- "math_logic": Equations, word problems, formal logic puzzles.
- "coding": Writing or analyzing software code.
- "visual_spatial": Grids, matrices, spatial transformations (like ARC-AGI).

Here is the complexity scale:
- "low": Direct lookup, single step retrieval or basic greeting.
- "medium": Multi-step operations, basic algebra, short code snippets.
- "high": Complex unseen puzzle patterns, large algorithms, architecture design.

You MUST respond with RAW JSON matching this exact schema:
{
    "domain": "...", # Must be one of the domains listed above
    "complexity": "...", # Must be low, medium, or high
    "reasoning": "...", # Short sentence explaining why you chose this
    "requires_tools": bool # true if it needs python execution, math tools, etc., false otherwise
}
"""

    def route(self, user_input: str) -> RouteDecision:
        """
        Takes the user input, queries the LLM for classification, and returns a verified Pydantic model.
        """
        response = self.llm.generate(
            system_prompt=self.system_prompt,
            user_prompt=user_input
        )
        
        # Clean the response in case the LLM wrapped it in markdown code blocks
        raw_content = response.content.strip()
        if raw_content.startswith("```json"):
            raw_content = raw_content[7:]
        if raw_content.startswith("```"):
            raw_content = raw_content[3:]
        if raw_content.endswith("```"):
            raw_content = raw_content[:-3]
            
        raw_content = raw_content.strip()
        
        # Standard JSON parsing
        decision_dict = json.loads(raw_content)
        
        # Pydantic will automatically validate that the fields match the required enums!
        return RouteDecision(**decision_dict)
