"""
Multi-provider LLM module -- OpenRouter ONLY (Qwen-powered).

Model stack (via OpenRouter):
  PRIMARY  -> qwen/qwq-32b               (32B reasoning model -- best for ARC)
  FALLBACK -> qwen/qwen-2.5-72b-instruct (72B instruct -- large, reliable backup)

Why QwQ-32B first?
  - Dedicated chain-of-thought reasoning model
  - Outperforms much larger models on logical/spatial tasks
  - Perfect fit for ARC-AGI pattern discovery

API key: set OPENROUTER_API_KEY in .env
Get a free key: https://openrouter.ai (no credit card needed)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from collections import deque
import os
import time
import logging

from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ===================================================
# RATE LIMITER
# ===================================================

class RateLimiter:
    def __init__(self, max_tokens_per_min: int, min_delay_sec: float):
        self.max_tokens = max_tokens_per_min
        self.min_delay = min_delay_sec
        self.window = deque()
        self.last_call = 0.0

    def wait_if_needed(self, estimated_tokens: int):
        now = time.time()

        # Enforce minimum inter-call delay (FIXED: was `pass` -- now actually sleeps)
        elapsed = now - self.last_call
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)

        # Enforce token-per-minute window
        cutoff = time.time() - 60
        while self.window and self.window[0][0] < cutoff:
            self.window.popleft()
        used = sum(tk for _, tk in self.window)
        if used + estimated_tokens > self.max_tokens and self.window:
            wait_until = self.window[0][0] + 60
            sleep_time = wait_until - time.time() + 1
            if sleep_time > 30.0:
                print(f"[RateLimiter] Token budget exceeded. Sleeping {sleep_time:.1f}s...")
                time.sleep(sleep_time)
            elif sleep_time > 0:
                print(f"[RateLimiter] Token budget low. Sleeping {sleep_time:.1f}s...")
                time.sleep(sleep_time)

        self.window.append((time.time(), estimated_tokens))
        self.last_call = time.time()


# ===================================================
# BASE CLASS
# ===================================================

class LLMResponse(BaseModel):
    content: str
    token_usage: Dict[str, Any] = {}


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str,
                 temperature: float = 0.0) -> LLMResponse:
        pass


# ===================================================
# OPENROUTER PROVIDER
# ===================================================

# Separate rate limiter instances per slot so they don't share state
openrouter_limiter_primary  = RateLimiter(max_tokens_per_min=150000, min_delay_sec=1.5)
openrouter_limiter_fallback = RateLimiter(max_tokens_per_min=150000, min_delay_sec=1.5)


class OpenRouterProvider(LLMProvider):
    """
    Routes requests through OpenRouter to any Qwen model.
    Uses the OpenAI-compatible API -- drop-in replacement.

    Primary model  : qwen/qwq-32b  (chain-of-thought reasoning)
    Fallback model : qwen/qwen-2.5-72b-instruct (72B production instruct)
    """
    def __init__(self, model: str, limiter: RateLimiter):
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key or len(api_key) < 10:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set in .env\n"
                "Get a free key at https://openrouter.ai"
            )
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/samrishtt/ARC-AGI",
                "X-Title": "CSA ARC-AGI Solver"
            }
        )
        self.model = model
        self.limiter = limiter
        self.name = f"OpenRouter({model})"

    def generate(self, system_prompt: str, user_prompt: str,
                 temperature: float = 0.0) -> LLMResponse:
        estimated_tokens = (len(system_prompt) + len(user_prompt)) // 4
        self.limiter.wait_if_needed(estimated_tokens)

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            max_tokens=4096,   # cap for free-tier compatibility
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ]
        )
        return LLMResponse(
            content=response.choices[0].message.content or "",
            token_usage=response.usage.model_dump() if response.usage else {}
        )


# ===================================================
# MULTI-PROVIDER FAILOVER ENGINE (OpenRouter-only)
# ===================================================

class MultiProviderLLM(LLMProvider):
    """
    Two-tier OpenRouter failover:
      [1] qwen/qwq-32b               -- reasoning powerhouse (PRIMARY)
      [2] qwen/qwen-2.5-72b-instruct -- 72B instruct backup  (FALLBACK)

    Failover rules:
      - 429/rate-limit  -> sleep + retry same model (up to 3 times)
      - 503/overloaded  -> switch to fallback immediately
      - Exhausted quota -> switch to fallback immediately
    """

    # Model constants -- change these two lines to swap models globally
    PRIMARY_MODEL  = "qwen/qwen3-next-80b-a3b-instruct:free"   # 80B MoE -- FREE
    FALLBACK_MODEL = "qwen/qwen3-coder:free"                    # Coder model -- FREE

    def __init__(self):
        self.providers: List[LLMProvider] = []
        self.provider_names: List[str] = []
        self.current_provider_idx = 0
        self.name = "OpenRouter-Qwen"
        self._init_providers()

    def reset_provider(self):
        """Reset to primary (qwen/qwq-32b). Call at the start of each task."""
        self.current_provider_idx = 0

    def _init_providers(self):
        """Build the two-model OpenRouter stack."""
        sep = "=" * 55
        print(f"\n{sep}")
        print("  CSA -- OpenRouter Qwen Stack")
        print(sep)

        # PRIMARY: qwen3-next-80b-a3b-instruct:free
        try:
            p = OpenRouterProvider(
                model=self.PRIMARY_MODEL,
                limiter=openrouter_limiter_primary
            )
            self.providers.append(p)
            self.provider_names.append("Qwen3-80B")
            print(f"  [1] PRIMARY  : {self.PRIMARY_MODEL}")
            print(f"      Role     : 80B MoE reasoning | Best for ARC pattern logic (FREE)")
        except Exception as e:
            logger.error(f"PRIMARY provider init failed: {e}")

        # FALLBACK: qwen3-coder:free
        try:
            p = OpenRouterProvider(
                model=self.FALLBACK_MODEL,
                limiter=openrouter_limiter_fallback
            )
            self.providers.append(p)
            self.provider_names.append("Qwen3-Coder")
            print(f"  [2] FALLBACK : {self.FALLBACK_MODEL}")
            print(f"      Role     : Coder model | Perfect for transform() code gen (FREE)")
        except Exception as e:
            logger.error(f"FALLBACK provider init failed: {e}")

        print(f"{sep}\n")

        if len(self.providers) == 0:
            raise RuntimeError(
                "CRITICAL: OpenRouter provider failed to initialize.\n"
                "Check OPENROUTER_API_KEY in your .env file."
            )

    def generate(self, system_prompt: str, user_prompt: str,
                 temperature: float = 0.0, task_id: str = "") -> LLMResponse:
        """Generate with automatic failover between Qwen models."""
        start_idx = self.current_provider_idx

        for provider_offset in range(len(self.providers)):
            idx = (start_idx + provider_offset) % len(self.providers)
            provider = self.providers[idx]
            provider_name = self.provider_names[idx]

            max_retries = 3
            for retry in range(max_retries):
                try:
                    label = f"Task {task_id}" if task_id else "Call"
                    print(f"[LLM] {label} -> {provider_name} (attempt {retry+1})")

                    response = provider.generate(system_prompt, user_prompt, temperature)

                    # Check for error strings returned in content
                    if response.content.startswith("Error:"):
                        err = response.content.lower()
                        if "429" in err or "rate" in err:
                            if retry < max_retries - 1:
                                wait = (retry + 1) * 10
                                print(f"[LLM] 429 on {provider_name}. Retry in {wait}s...")
                                time.sleep(wait)
                                continue
                            else:
                                print(f"[LLM] Persistent 429 on {provider_name}. Switching...")
                                break
                        elif "503" in err or "overload" in err:
                            print(f"[LLM] 503 on {provider_name}. Switching to fallback...")
                            break
                        elif "quota" in err or "credit" in err or "billing" in err:
                            print(f"[LLM] Quota/credits exhausted on {provider_name}. Switching...")
                            break
                        else:
                            if provider_offset == len(self.providers) - 1:
                                return response
                            break

                    # Success -- stick with this provider
                    self.current_provider_idx = idx
                    return response

                except Exception as e:
                    err = str(e).lower()
                    if "429" in err or "rate" in err:
                        if retry < max_retries - 1:
                            wait = (retry + 1) * 10
                            print(f"[LLM] 429 exception on {provider_name}. Retry in {wait}s...")
                            time.sleep(wait)
                            continue
                        else:
                            print(f"[LLM] Persistent 429 on {provider_name}. Switching...")
                            break
                    elif "503" in err or "overload" in err:
                        print(f"[LLM] 503 on {provider_name}. Switching to fallback...")
                        break
                    else:
                        logger.warning(f"Provider {provider_name} error: {e}")
                        break

        raise RuntimeError(
            "CRITICAL: All OpenRouter Qwen providers failed.\n"
            "Check rate limits at https://openrouter.ai/activity"
        )
