from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
try:
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{'role': 'user', 'content': 'Hello'}]
    )
    print("SUCCESS")
    print(response.choices[0].message.content)
except Exception as e:
    print('ERROR:', e)
