import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

url = "https://api.anthropic.com/v1/messages"
headers = {
    "Content-Type": "application/json",
    "x-api-key": os.environ["ANTHROPIC_API_KEY"],   # la clave va en una cabecera
    "anthropic-version": "2023-06-01",               # versión de la API
}
body = json.dumps({
    "model": os.environ.get("LLM_MODEL", "claude-haiku-5-5"),
    "max_tokens": 256,
    "messages": [{"role": "user", "content": "¿Qué es una red neuronal? Responde en una frase."}],
}).encode()

req = urllib.request.Request(url, data=body, headers=headers, method="POST")

try:
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
        print("=== RESPUESTA COMPLETA (JSON) ===")
        print(json.dumps(result, indent=2, ensure_ascii=False))
except urllib.error.HTTPError as e:
    print(f"=== ERROR HTTP {e.code} ===")
    print(e.read().decode())