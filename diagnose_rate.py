import os, json, urllib.request, urllib.error
from dotenv import load_dotenv
load_dotenv()

key = os.getenv("OPENROUTER_API_KEY", "").strip()
model = "qwen/qwen3-next-80b-a3b-instruct:free"

data = json.dumps({
    "model": model,
    "max_tokens": 10,
    "messages": [{"role": "user", "content": "Say yes"}]
}).encode()

req = urllib.request.Request(
    "https://openrouter.ai/api/v1/chat/completions",
    data=data,
    headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/samrishtt/ARC-AGI"
    }
)
try:
    with urllib.request.urlopen(req, timeout=30) as r:
        result = json.loads(r.read())
        print("SUCCESS:", json.dumps(result, indent=2)[:500])
except urllib.error.HTTPError as e:
    body = e.read().decode()
    print(f"HTTP {e.code}")
    print(f"Body: {body[:800]}")
    print("All headers:")
    for k, v in e.headers.items():
        print(f"  {k}: {v}")
