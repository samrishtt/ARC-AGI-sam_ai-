"""
Multi-provider LLM module -- OpenRouter FREE stack (max power).

Provider stack (all FREE, all open-source, via OpenRouter):
  [1] nvidia/nemotron-3-super-120b-a12b:free  -- 120B MoE, 262K ctx  (PRIMARY)
  [2] nousresearch/hermes-3-llama-3.1-405b:free -- 405B reasoning   (FALLBACK 1)
  [3] qwen/qwen3-next-80b-a3b-instruct:free   -- 80B Qwen3 MoE      (FALLBACK 2)
  [4] meta-llama/llama-3.3-70b-instruct:free  -- 70B Llama3, fast   (FALLBACK 3)

All models: 100% free, 100% open-source weights.
Each is on a separate upstream provider so rate limits are independent.

API key: OPENROUTER_API_KEY in .env
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

        # Enforce minimum inter-call delay (fixed: was a no-op `pass`)
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
            if sleep_time > 0:
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
# OPENROUTER PROVIDER (generic -- any model)
# ===================================================

def _make_limiter():
    """Each provider slot gets its own rate limiter so they don't share state."""
    return RateLimiter(max_tokens_per_min=120000, min_delay_sec=1.0)


class OpenRouterProvider(LLMProvider):
    """
    Routes requests through OpenRouter to any open-source model.
    Uses the OpenAI-compatible API -- drop-in replacement.
    """
    def __init__(self, model: str, limiter: RateLimiter, display_name: str = ""):
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key or len(api_key) < 10:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set in .env\n"
                "Get a free key at https://openrouter.ai"
            )
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=300.0,  # 5 minutes for large models to reason
            default_headers={
                "HTTP-Referer": "https://github.com/samrishtt/ARC-AGI",
                "X-Title": "CSA ARC-AGI Solver"
            }
        )
        self.model = model
        self.limiter = limiter
        self.name = display_name or model

    def generate(self, system_prompt: str, user_prompt: str,
                 temperature: float = 0.0) -> LLMResponse:
        estimated_tokens = (len(system_prompt) + len(user_prompt)) // 4
        self.limiter.wait_if_needed(estimated_tokens)

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ]
        )

        # Some reasoning models (Nemotron, Qwen3-thinking) return content=None
        # when the answer is in the reasoning buffer -- extract it correctly
        content = ""
        if response.choices:
            msg = response.choices[0].message
            content = msg.content or ""
            if not content:
                # Try reasoning_content (Nemotron / OpenRouter thinking models)
                raw = getattr(msg, "reasoning_content", None)
                if raw:
                    content = raw
                else:
                    # Last resort: check if model_extra has the content
                    extra = getattr(msg, "model_extra", {}) or {}
                    content = extra.get("content", "") or extra.get("reasoning", "") or ""

        return LLMResponse(
            content=content,
            token_usage=response.usage.model_dump() if response.usage else {}
        )


# ===================================================
# MULTI-PROVIDER FAILOVER ENGINE
# ===================================================

class MultiProviderLLM(LLMProvider):
    """
    4-tier free open-source model stack via OpenRouter.

    Each model is on a DIFFERENT upstream provider, so their rate limits
    are independent -- if one pool is exhausted, the next has a fresh quota.

    # Provider order (all free, all open-source, FAST models for 120s timeout):
      [1] Llama3.3-70B (PRIMARY)    -- Meta via Together, fast reasoning
      [2] Qwen3-Coder  (FALLBACK 1) -- Venice.ai, excellent code generation
      [3] Gemma-3-27B  (FALLBACK 2) -- Google, strict instruct adherence
      [4] OpenRouter-Auto (FALLBACK 3) -- Dynamic routing mechanism

    Failover triggers:
      - 429 / upstream rate-limit -> wait 65s then retry (resets per minute)
      - 503 / overloaded          -> switch immediately
      - Credits exhausted         -> switch immediately
    """

    # -- Model IDs (all verified free, selected for high speed) -----------
    MODELS = [
        {
            "id":   "meta-llama/llama-3.3-70b-instruct:free",
            "name": "Llama3.3-70B",
            "desc": "70B | 65K ctx | Fast & reliable reasoning"
        },
        {
            "id":   "qwen/qwen3-coder:free",
            "name": "Qwen3-Coder",
            "desc": "Coder | 262K ctx | Excellent at writing transform()"
        },
        {
            "id":   "google/gemma-3-27b-it:free",
            "name": "Gemma-3-27B",
            "desc": "27B | 131K ctx | Strict instruction following"
        },
        {
            "id":   "openrouter/free",
            "name": "OpenRouter-Auto",
            "desc": "Dynamic free routing | Fastest available online"
        },
    ]

    # Convenience properties for run_benchmark.py display
    PRIMARY_MODEL  = MODELS[0]["id"]
    FALLBACK_MODEL = MODELS[1]["id"]

    def __init__(self):
        self.providers: List[LLMProvider] = []
        self.provider_names: List[str] = []
        self.current_provider_idx = 0
        self.name = "OpenRouter-FreeStack"
        self._init_providers()

    def reset_provider(self):
        """Reset to primary provider. Called at start of each task."""
        self.current_provider_idx = 0

    def _init_providers(self):
        """Build the free model stack."""
        sep = "=" * 60
        print(f"\n{sep}")
        print("  CSA -- OpenRouter Free Stack (4 models, all open-source)")
        print(sep)

        for i, m in enumerate(self.MODELS):
            try:
                p = OpenRouterProvider(
                    model=m["id"],
                    limiter=_make_limiter(),
                    display_name=m["name"]
                )
                self.providers.append(p)
                self.provider_names.append(m["name"])
                role = "PRIMARY " if i == 0 else f"FALLBACK {i}"
                print(f"  [{i+1}] {role:<12}: {m['name']}")
                print(f"        {m['desc']}")
            except Exception as e:
                logger.error(f"Model {m['id']} init failed: {e}")

        print(f"\n  Rate-limit strategy: 65s wait on 429 (upstream resets per minute)")
        print(f"  Each model is on a DIFFERENT upstream -- independent quotas")
        print(f"{sep}\n")

        if len(self.providers) == 0:
            raise RuntimeError(
                "CRITICAL: No providers initialized.\n"
                "Check OPENROUTER_API_KEY in your .env file."
            )

    def generate(self, system_prompt: str, user_prompt: str,
                 temperature: float = 0.0, task_id: str = "") -> LLMResponse:
        """Generate with automatic failover across the 4-model free stack."""
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

                    # Detect error strings returned in content body
                    if response.content.startswith("Error:"):
                        err = response.content.lower()
                        if "429" in err or "rate" in err or "temporarily" in err:
                            if retry < max_retries - 1:
                                wait = 65 if retry == 0 else 90
                                print(f"[LLM] 429/rate-limit on {provider_name}.")
                                print(f"[LLM] Waiting {wait}s for upstream reset...")
                                time.sleep(wait)
                                continue
                            else:
                                print(f"[LLM] Persistent 429 on {provider_name}. Switching...")
                                break
                        elif "503" in err or "overload" in err:
                            print(f"[LLM] 503 on {provider_name}. Switching...")
                            break
                        elif "afford" in err or "credit" in err or "quota" in err or "billing" in err:
                            print(f"[LLM] Credits exhausted on {provider_name}. Switching...")
                            break
                        else:
                            if provider_offset == len(self.providers) - 1:
                                return response
                            break

                    # Success
                    self.current_provider_idx = idx
                    return response

                except Exception as e:
                    err = str(e).lower()
                    if "429" in err or "rate" in err or "temporarily" in err:
                        if retry < max_retries - 1:
                            wait = 65 if retry == 0 else 90
                            print(f"[LLM] 429/rate-limit on {provider_name}.")
                            print(f"[LLM] Waiting {wait}s for upstream reset...")
                            time.sleep(wait)
                            continue
                        else:
                            print(f"[LLM] Persistent 429 on {provider_name}. Switching...")
                            break
                    elif "503" in err or "overload" in err:
                        print(f"[LLM] 503 on {provider_name}. Switching...")
                        break
                    elif "afford" in err or "credit" in err or "quota" in err or "billing" in err:
                        print(f"[LLM] Credits exhausted on {provider_name}. Switching...")
                        break
                    else:
                        print(f"[{provider_name}] Exception: {str(e)[:200]}")
                        break

        raise RuntimeError(
            "CRITICAL: All 4 free providers failed.\n"
            "Try again in 1-2 minutes (upstream pools reset per minute).\n"
            "Check activity: https://openrouter.ai/activity"
        )
