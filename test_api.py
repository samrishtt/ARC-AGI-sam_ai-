import os
import anthropic
from dotenv import load_dotenv

load_dotenv()
try:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=64,
        messages=[{"role": "user", "content": "Hello"}]
    )
    print("SUCCESS")
    print(response.content[0].text)
except Exception as e:
    print("ERROR:", e)
