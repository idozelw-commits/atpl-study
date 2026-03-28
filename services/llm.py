"""Shared LLM client using Groq (free tier)."""
import os
from groq import Groq

_client = None

# Models in priority order — fall back if rate limited
MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "qwen/qwen3-32b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
]


def get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
    return _client


def generate(prompt: str, system: str = None) -> str:
    """Generate text using Groq. Falls back to smaller models if rate limited."""
    client = get_client()

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for model in MODELS:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=2048,
            )
            return response.choices[0].message.content
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                print(f"Rate limited on {model}, trying next...")
                continue
            raise

    raise Exception("All Groq models rate limited. Please try again later.")
