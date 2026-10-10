import os
from pathlib import Path

from dotenv import load_dotenv
import anthropic

# Cargar el .env que está en la misma carpeta que este script
load_dotenv(Path(__file__).parent / ".env")

client = anthropic.Anthropic()  # lee ANTHROPIC_API_KEY automáticamente
MODEL = os.environ.get("LLM_MODEL", "claude-haiku-5-5")

response = client.messages.create(
    model=MODEL,
    max_tokens=256,
    messages=[{"role": "user", "content": "¿Qué es una red neuronal? Responde en una frase."}],
)

print(response.content[0].text)
print("---")
print(f"Tokens de entrada: {response.usage.input_tokens}")
print(f"Tokens de salida:  {response.usage.output_tokens}")