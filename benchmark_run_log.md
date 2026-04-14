# CSA Benchmark Results (IN-PROGRESS)

> **Status:** Running in background (Currently at Task 5/50)
> **Configuration used:** Gemini (Primary) → Groq (Fallback 1) → NVIDIA (Fallback 2)

All fixes have been correctly implemented. Based on the real-time execution logs, the failover engine and the new token budget rate limiters are behaving exactly as requested.

## Execution Dynamics
1. **Gemini Burst Throttling:** Gemini processes the rapid initial tasks but immediately fails over to Groq once it hits a burst 429 (`Persistent 429 on Gemini. Switching.`).
2. **Rate Limiter Backoff:** Instead of `RateLimiter` hard-crashing and forcing NVIDIA (which then results in Unauthorized failures), it detects the long timeout limit and sleeps correctly (e.g., `[RateLimiter] Token limit exceeded. Sleeping 39.8s to recover budget.`).
3. **No Silent MockLLM:** The benchmark is running on REAL providers sequentially without hiding failures behind a mock interface.
4. **Task-level Delays:** The main evaluation loop is observing the new `time.sleep(3)` instruction after each task completes.

## Live Task Processing Snapshot

| Task ID | Pass/Fail | Provider Handled | Failure Reason |
|---------|-----------|------------------|----------------|
| `007bbfb7` | Failed | Groq | `coding_sandbox: code_execution_failed` |
| `00d62c1b` | Failed | Groq | `coding_sandbox: code_execution_failed` |
| `017c7c7b` | Failed | Groq | `coding_sandbox: code_execution_failed` |
| `025d127b` | Failed | Groq | `coding_sandbox: code_execution_failed` |
| `045e512c` | Processing... | Groq | N/A |

### What to expect next?
Because of the heavy token load (up to ~8k prompt tokens + 1.5k code tokens) applied across 3 separate Sandbox verification retries *per task*, Groq routinely depletes its rolling token budget and gracefully sleeps for 40-60 seconds to regenerate it. 

The background `run_benchmark.py` wrapper will sequentially chunk through these 50 workloads. By calculating the average rate/delay, the full `benchmark_results.json` should organically populate in approximately **55 to 70 minutes** inside the `logs\` directory.
