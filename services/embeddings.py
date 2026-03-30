"""OpenAI embeddings for semantic search."""
import os
from openai import OpenAI

_client = None


def get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    return _client


def get_embedding(text: str) -> list:
    """Get embedding for a single text."""
    client = get_client()
    # Truncate to ~8000 tokens (~32000 chars) to stay within limits
    text = text[:32000]
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


def get_embeddings_batch(texts: list) -> list:
    """Get embeddings for a batch of texts. Returns list of embedding vectors."""
    client = get_client()
    all_embeddings = []
    batch_size = 100  # OpenAI supports up to 2048 per request
    for i in range(0, len(texts), batch_size):
        batch = [t[:32000] for t in texts[i:i + batch_size]]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch,
        )
        # Sort by index to maintain order
        sorted_data = sorted(response.data, key=lambda x: x.index)
        all_embeddings.extend([d.embedding for d in sorted_data])
    return all_embeddings
