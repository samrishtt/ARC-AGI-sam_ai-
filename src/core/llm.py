from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import os
import logging
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()
logger = logging.getLogger(__name__)

class LLMResponse(BaseModel):
    content: str
    token_usage: Dict[str, Any] = {}

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> LLMResponse:
        """Generate a response. temperature=0.0 for deterministic code, 0.5 for creative reasoning."""
        pass

class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> LLMResponse:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=temperature,
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


class GeminiProvider(LLMProvider):
    """Free-tier Google Gemini provider. No billing required."""
    def __init__(self, model: str = "gemini-2.0-flash"):
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel(model)

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> LLMResponse:
        try:
            combined_prompt = f"SYSTEM INSTRUCTIONS:\n{system_prompt}\n\nUSER:\n{user_prompt}"
            response = self.model.generate_content(
                combined_prompt,
                generation_config={"temperature": temperature}
            )
            return LLMResponse(content=response.text)
        except Exception as e:
            return LLMResponse(content=f"Error: {str(e)}")


class GroqProvider(LLMProvider):
    """Free-tier Groq provider. Runs Llama 3.1 8B with higher RPM/TPM limits."""
    def __init__(self, model: str = "llama-3.1-8b-instant"):
        from groq import Groq
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> LLMResponse:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            return LLMResponse(
                content=response.choices[0].message.content,
                token_usage=dict(response.usage) if response.usage else {}
            )
        except Exception as e:
            return LLMResponse(content=f"Error: {str(e)}")

class AnthropicProvider(LLMProvider):
    """Paid-tier Anthropic provider for Claude 4.6 Sonnet.
    
    Pricing (as of March 2026):
      - Input:  $3/MTok  (~₹276/MTok)
      - Output: $15/MTok (~₹1382/MTok)
    """
    def __init__(self, model: str = "claude-sonnet-4-6", max_tokens: int = 8192):
        import anthropic
        self._anthropic = anthropic  # Store module ref for retry exception types
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((Exception,)),  # Catches rate limits, connection, server errors
        reraise=True,
        before_sleep=lambda retry_state: logger.warning(
            f"Anthropic API error: {retry_state.outcome.exception()}. "
            f"Retrying in {retry_state.next_action.sleep:.1f}s... "
            f"(attempt {retry_state.attempt_number}/5)"
        )
    )
    def _call_api(self, system_prompt: str, user_prompt: str, temperature: float):
        """Internal method with tenacity retry for rate-limit resilience."""
        try:
            return self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
        except self._anthropic.RateLimitError:
            raise  # Let tenacity handle retry
        except self._anthropic.APIConnectionError:
            raise  # Let tenacity handle retry
        except self._anthropic.InternalServerError:
            raise  # Let tenacity handle retry
        except self._anthropic.AuthenticationError:
            raise  # Don't retry auth errors — re-raise immediately
        except Exception:
            raise  # Other errors also get retried

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> LLMResponse:
        try:
            response = self._call_api(system_prompt, user_prompt, temperature)
            content = response.content[0].text if response.content else ""
            return LLMResponse(
                content=content,
                token_usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }
            )
        except Exception as e:
            return LLMResponse(content=f"Error: {str(e)}")

class MockLLMProvider(LLMProvider):
    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> LLMResponse:
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


def get_best_provider(prefer_paid: bool = True) -> LLMProvider:
    """
    Smart provider factory with automatic fallback.

    Priority order:
      1. Claude Sonnet 4.6  (paid — best quality)
      2. Google Gemini Flash (free tier)
      3. Groq Llama 3.1     (free tier)
      4. MockLLM             (offline — always works)

    Args:
        prefer_paid: If True, try Claude first. If False, skip straight to free models.

    Returns:
        The best available LLMProvider instance.
    """
    # --- 1. Claude (primary, paid) ---
    if prefer_paid:
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if anthropic_key and len(anthropic_key) > 10:
            try:
                provider = AnthropicProvider()
                logger.info("LLM Provider: Anthropic Claude Sonnet 4.6 (paid)")
                print("[OK] Using Anthropic Claude Sonnet 4.6 (primary)")
                return provider
            except Exception as e:
                logger.warning(f"Anthropic init failed: {e}")

    # --- 2. Gemini (free fallback) ---
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if gemini_key and len(gemini_key) > 5:
        try:
            provider = GeminiProvider()
            logger.info("LLM Provider: Google Gemini Flash (free tier)")
            print("[FREE] Using Google Gemini Flash (free fallback)")
            return provider
        except Exception as e:
            logger.warning(f"Gemini init failed: {e}")

    # --- 3. Groq (free fallback) ---
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key and len(groq_key) > 5:
        try:
            provider = GroqProvider()
            logger.info("LLM Provider: Groq Llama 3.1 (free tier)")
            print("[FREE] Using Groq Llama 3.1 (free fallback)")
            return provider
        except Exception as e:
            logger.warning(f"Groq init failed: {e}")

    # --- 4. Mock (offline, always works) ---
    logger.warning("No API keys found -- using MockLLM (offline mode)")
    print("[WARN] No API keys found -- using MockLLM (offline mode)")
    return MockLLMProvider()
