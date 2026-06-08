import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3.2",
        "prompt": "Explain why sleep is important for students.",
        "stream": False
    }
)

print(response.json()["response"])