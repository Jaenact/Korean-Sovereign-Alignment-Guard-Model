# moderator_guard/llm/client.py

from typing import List
import numpy as np
from openai import OpenAI

from moderator_guard.config import EMBEDDING_MODEL

# It's recommended to set the API key in the environment variable OPENAI_API_KEY
# For example: export OPENAI_API_KEY='your-api-key'
client = OpenAI()

def get_embedding(text: str) -> np.ndarray:
    """Returns the OpenAI embedding vector for the given text."""
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    vec = np.array(resp.data[0].embedding, dtype=np.float32)
    return vec

def embed_texts(texts: List[str]) -> np.ndarray:
    """Receives a list of texts and converts it into an (N, d) embedding array."""
    embs = []
    total = len(texts)
    for i, t in enumerate(texts):
        if (i + 1) % 50 == 0 or i == total - 1:
            print(f"[Embedding] {i+1}/{total} processed")
        e = get_embedding(t)
        embs.append(e)
    embs_arr = np.vstack(embs)
    return embs_arr
