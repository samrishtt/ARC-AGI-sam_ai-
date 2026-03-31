"""
Multi-provider LLM module with automatic failover and per-provider rate limiting.

Provider Priority:
  1. Groq (llama-3.3-70b-versatile)       — PRIMARY
  2. NVIDIA NIM (llama-4-scout-17b)        — FALLBACK 1
  3. Google Gemini 1.5 Flash               — FALLBACK 2
  4. MockLLM                               — OFFLINE

Each provider uses the OpenAI Python SDK with base_url override.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from collections import deque
import os
import time
import logging

from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
# TASK 2 — RATE LIMITER CLASS (per provider)
# ═══════════════════════════════════════════════════════

class RateLimiter:
    def __init__(self, max_tokens_per_min: int, min_delay_sec: float):
        self.max_tokens = max_tokens_per_min
        self.min_delay = min_delay_sec
        self.window = deque()
        self.last_call = 0.0

    def wait_if_needed(self, estimated_tokens: int):
        now = time.time()

        # Enforce minimum inter-call delay
        elapsed = now - self.last_call
        if elapsed < self.min_delay:
            sleep_time = self.min_delay - elapsed
            print(f"[RateLimiter] Inter-call delay. Sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)

        # Enforce token-per-minute window
        cutoff = time.time() - 60
        while self.window and self.window[0][0] < cutoff:
            self.window.popleft()
        used = sum(tk for _, tk in self.window)
        if used + estimated_tokens > self.max_tokens and self.window:
            wait_until = self.window[0][0] + 60
            sleep_time = wait_until - time.time() + 1
            if sleep_time > 0:
                print(f"[RateLimiter] Token limit approaching. Sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)

        self.window.append((time.time(), estimated_tokens))
        self.last_call = time.time()


# Module-level rate limiter instances (one per provider)
groq_limiter = RateLimiter(max_tokens_per_min=5000, min_delay_sec=3)
nvidia_limiter = RateLimiter(max_tokens_per_min=999999, min_delay_sec=2)
gemini_limiter = RateLimiter(max_tokens_per_min=900000, min_delay_sec=5)


# ═══════════════════════════════════════════════════════
# BASE CLASSES
# ═══════════════════════════════════════════════════════

class LLMResponse(BaseModel):
    content: str
    token_usage: Dict[str, Any] = {}


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> LLMResponse:
        """Generate a response. temperature=0.0 for deterministic code, 0.5 for creative reasoning."""
        pass


# ═══════════════════════════════════════════════════════
# PROVIDER 1 — Groq (PRIMARY)
# ═══════════════════════════════════════════════════════

class GroqProvider(LLMProvider):
    """Free-tier Groq provider using llama-3.3-70b-versatile via OpenAI SDK."""
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        self.model = model
        self.limiter = groq_limiter
        self.name = "Groq"

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> LLMResponse:
        estimated_tokens = (len(system_prompt) + len(user_prompt)) // 4
        self.limiter.wait_if_needed(estimated_tokens)

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return LLMResponse(
            content=response.choices[0].message.content or "",
            token_usage=response.usage.model_dump() if response.usage else {}
        )


# ═══════════════════════════════════════════════════════
# PROVIDER 2 — NVIDIA NIM (FALLBACK 1)
# ═══════════════════════════════════════════════════════

class NvidiaProvider(LLMProvider):
    """Free-tier NVIDIA NIM provider using meta/llama-4-scout via OpenAI SDK."""
    def __init__(self, model: str = "meta/llama-4-scout-17b-16e-instruct"):
        api_key = os.getenv("NVIDIA_API_KEY", "").strip()
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://integrate.api.nvidia.com/v1"
        )
        self.model = model
        self.limiter = nvidia_limiter
        self.name = "NVIDIA"

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> LLMResponse:
        estimated_tokens = (len(system_prompt) + len(user_prompt)) // 4
        self.limiter.wait_if_needed(estimated_tokens)

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return LLMResponse(
            content=response.choices[0].message.content or "",
            token_usage=response.usage.model_dump() if response.usage else {}
        )


# ═══════════════════════════════════════════════════════
# PROVIDER 3 — Google Gemini 1.5 Flash (FALLBACK 2)
# ═══════════════════════════════════════════════════════

class GeminiProvider(LLMProvider):
    """Free-tier Google Gemini 1.5 Flash via OpenAI-compatible endpoint."""
    def __init__(self, model: str = "gemini-1.5-flash"):
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        self.model = model
        self.limiter = gemini_limiter
        self.name = "Gemini"

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> LLMResponse:
        estimated_tokens = (len(system_prompt) + len(user_prompt)) // 4
        self.limiter.wait_if_needed(estimated_tokens)

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return LLMResponse(
            content=response.choices[0].message.content or "",
            token_usage=response.usage.model_dump() if response.usage else {}
        )


# ═══════════════════════════════════════════════════════
# LEGACY PROVIDERS (kept for backwards compatibility)
# ═══════════════════════════════════════════════════════

class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.name = "OpenAI"

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
                content=response.choices[0].message.content or "",
                token_usage=response.usage.model_dump() if response.usage else {}
            )
        except Exception as e:
            return LLMResponse(content=f"Error: {str(e)}")


class AnthropicProvider(LLMProvider):
    """Paid-tier Anthropic provider for Claude 4.6 Sonnet."""
    def __init__(self, model: str = "claude-sonnet-4-6", max_tokens: int = 8192):
        import anthropic
        self._anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens
        self.name = "Anthropic"

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> LLMResponse:
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
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
    def __init__(self):
        self.name = "MockLLM"

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> LLMResponse:
        if "python code" in system_prompt.lower():
            if "rotate" in user_prompt.lower():
                return LLMResponse(content="```python\nimport numpy as np\nfrom src.dsl.primitives import rotate_cw\n\nprint('Rotation Logic Here')\n```")
            return LLMResponse(content="```python\nprint('Hello from Mock Code')\n```")
        if "valid JSON" in system_prompt:
            if "hello" in user_prompt.lower():
                return LLMResponse(content='{"steps": [{"action": "ANALYZE", "description": "User greeting"}]}')
            return LLMResponse(content='{"steps": [{"action": "ANALYZE", "description": "Analyze grid"}, {"action": "TRANSFORM", "description": "Apply rotation"}, {"action": "VERIFY", "description": "Check result"}]}')
        return LLMResponse(content=f"[MOCK] Processed: {user_prompt[:50]}...")


# ═══════════════════════════════════════════════════════
# MULTI-PROVIDER FAILOVER ENGINE
# ═══════════════════════════════════════════════════════

class MultiProviderLLM(LLMProvider):
    """
    Wraps multiple providers with automatic failover logic.

    Failover rules:
    - Start with Provider 1 (Groq)
    - On 429 (rate limit): wait + retry same provider (up to 3 retries)
    - On 503 (NVIDIA overloaded): try next provider immediately
    - On credits exhausted or persistent 429: switch to next provider
    - Logs which provider was used for each call
    """
    def __init__(self):
        self.providers: List[LLMProvider] = []
        self.provider_names: List[str] = []
        self.current_provider_idx = 0
        self.name = "MultiProvider"

        # Initialize all available providers
        self._init_providers()

    def reset_provider(self):
        """Reset to primary provider (Groq). Called at start of each task."""
        self.current_provider_idx = 0

    def _init_providers(self):
        """Initialize providers in priority order: Groq → NVIDIA → Gemini → Mock"""
        # Provider 1: Groq (PRIMARY)
        groq_key = os.getenv("GROQ_API_KEY", "").strip()
        if groq_key and len(groq_key) > 10:
            try:
                p = GroqProvider()
                self.providers.append(p)
                self.provider_names.append("Groq")
                print("[OK] Provider 1: Groq (llama-3.3-70b-versatile) -- PRIMARY")
            except Exception as e:
                logger.warning(f"Groq init failed: {e}")

        # Provider 2: NVIDIA NIM (FALLBACK 1)
        nvidia_key = os.getenv("NVIDIA_API_KEY", "").strip()
        if nvidia_key and len(nvidia_key) > 10 and nvidia_key != "your_key_here":
            try:
                p = NvidiaProvider()
                self.providers.append(p)
                self.provider_names.append("NVIDIA")
                print("[OK] Provider 2: NVIDIA NIM (llama-4-scout) -- FALLBACK 1")
            except Exception as e:
                logger.warning(f"NVIDIA init failed: {e}")

        # Provider 3: Gemini (FALLBACK 2)
        gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        if gemini_key and len(gemini_key) > 10:
            try:
                p = GeminiProvider()
                self.providers.append(p)
                self.provider_names.append("Gemini")
                print("[OK] Provider 3: Gemini 1.5 Flash -- FALLBACK 2")
            except Exception as e:
                logger.warning(f"Gemini init failed: {e}")

        # Always have Mock as last resort
        self.providers.append(MockLLMProvider())
        self.provider_names.append("MockLLM")

    def generate(self, system_prompt: str, user_prompt: str,
                 temperature: float = 0.0, task_id: str = "") -> LLMResponse:
        """Generate with automatic failover across providers."""
        # Always start from current_provider_idx (reset per task)
        start_idx = self.current_provider_idx

        for provider_offset in range(len(self.providers)):
            idx = (start_idx + provider_offset) % len(self.providers)
            provider = self.providers[idx]
            provider_name = self.provider_names[idx]

            max_retries = 3
            for retry in range(max_retries):
                try:
                    if task_id:
                        print(f"[Provider] Task {task_id}: using {provider_name}")
                    else:
                        print(f"[Provider] Using {provider_name}")

                    response = provider.generate(system_prompt, user_prompt, temperature)

                    # Check for error responses that indicate we should failover
                    if response.content.startswith("Error:"):
                        error_lower = response.content.lower()
                        if "429" in error_lower or "rate" in error_lower:
                            if retry < max_retries - 1:
                                wait_time = (retry + 1) * 5
                                print(f"[Failover] 429 rate limit on {provider_name}. "
                                      f"Retry {retry+1}/{max_retries} in {wait_time}s...")
                                time.sleep(wait_time)
                                continue
                            else:
                                print(f"[Failover] Persistent 429 on {provider_name}. "
                                      f"Switching to next provider.")
                                break  # try next provider
                        elif "503" in error_lower or "overloaded" in error_lower:
                            print(f"[Failover] 503 overloaded on {provider_name}. "
                                  f"Trying next provider...")
                            break  # try next provider
                        elif "credit" in error_lower or "quota" in error_lower:
                            print(f"[Failover] Credits exhausted on {provider_name}. "
                                  f"Switching to next provider.")
                            break  # try next provider
                        else:
                            # Non-retryable error but still an error response
                            # Try to return it if it's the last provider
                            if provider_offset == len(self.providers) - 1:
                                return response
                            break  # try next provider

                    # Success!
                    self.current_provider_idx = idx  # Sticky: keep using what works
                    return response

                except Exception as e:
                    error_str = str(e).lower()
                    if "429" in error_str or "rate" in error_str:
                        if retry < max_retries - 1:
                            wait_time = (retry + 1) * 5
                            print(f"[Failover] 429 exception on {provider_name}. "
                                  f"Retry {retry+1}/{max_retries} in {wait_time}s...")
                            time.sleep(wait_time)
                            continue
                        else:
                            print(f"[Failover] Persistent 429 on {provider_name}. Switching.")
                            break
                    elif "503" in error_str or "overloaded" in error_str:
                        print(f"[Failover] 503 on {provider_name}. Trying next provider.")
                        break
                    else:
                        logger.warning(f"Provider {provider_name} error: {e}")
                        break  # try next provider

        # If absolutely everything failed
        return LLMResponse(content="Error: All providers exhausted. No response generated.")


# ═══════════════════════════════════════════════════════
# FACTORY FUNCTION
# ═══════════════════════════════════════════════════════

def get_best_provider(prefer_paid: bool = False) -> LLMProvider:
    """
    Smart provider factory with automatic failover.

    Priority order (free-first for ARC benchmarking):
      1. Groq llama-3.3-70b       (free — primary)
      2. NVIDIA NIM llama-4-scout  (free — fallback 1)
      3. Gemini 1.5 Flash          (free — fallback 2)
      4. MockLLM                   (offline — always works)

    Args:
        prefer_paid: If True, try Anthropic first (legacy behavior).

    Returns:
        A MultiProviderLLM instance with failover.
    """
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

    # Default: use multi-provider failover engine with free providers
    return MultiProviderLLM()
