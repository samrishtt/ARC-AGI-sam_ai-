"""Diagnose what Nemotron-120B actually returns for content."""
import os, json
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

key = os.getenv("OPENROUTER_API_KEY", "").strip()
client = OpenAI(
    api_key=key,
    base_url="https://openrouter.ai/api/v1",
    default_headers={"HTTP-Referer": "https://github.com/samrishtt/ARC-AGI"}
)

response = client.chat.completions.create(
    model="nvidia/nemotron-3-super-120b-a12b:free",
    max_tokens=100,
    messages=[
        {"role": "system", "content": "Be very brief."},
        {"role": "user",   "content": "What is 2+2?"}
    ]
)

msg = response.choices[0].message
print(f"content          : {repr(msg.content)}")
print(f"reasoning_content: {repr(getattr(msg, 'reasoning_content', 'N/A'))}")
print(f"model_extra      : {getattr(msg, 'model_extra', {})}")
print(f"role             : {msg.role}")
print()
print(f"Full choices[0]  : {response.choices[0]}")
