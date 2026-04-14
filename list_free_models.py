"""Query OpenRouter for all currently free models, sorted by parameter count."""
import os, json, urllib.request
from dotenv import load_dotenv
load_dotenv()

key = os.getenv("OPENROUTER_API_KEY", "").strip()
req = urllib.request.Request(
    "https://openrouter.ai/api/v1/models",
    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
)
with urllib.request.urlopen(req, timeout=15) as r:
    data = json.loads(r.read())

models = data.get("data", [])
free_models = []
for m in models:
    pricing = m.get("pricing", {})
    prompt_cost = float(pricing.get("prompt", 999))
    completion_cost = float(pricing.get("completion", 999))
    if prompt_cost == 0 and completion_cost == 0:
        ctx = m.get("context_length", 0)
        free_models.append({
            "id": m.get("id"),
            "name": m.get("name", ""),
            "ctx": ctx,
        })

free_models.sort(key=lambda x: x["ctx"], reverse=True)

print(f"All FREE models on OpenRouter ({len(free_models)} total):\n")
print(f"  {'Model ID':<55} {'Context':>10}")
print(f"  {'-'*55} {'-'*10}")
for m in free_models:
    print(f"  {m['id']:<55} {m['ctx']:>10,}")
