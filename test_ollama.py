import os
import requests
from dotenv import load_dotenv

load_dotenv()

response = requests.get(
    "https://ollama.com/api/tags",
    headers={
        "Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"
    }
)

print(response.status_code)
print(response.json())