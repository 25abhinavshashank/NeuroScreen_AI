
from __future__ import annotations

from typing import List

from langchain_ollama import OllamaEmbeddings

from backend.app.core.config import get_settings

EMBEDDING_MODEL = "nomic-embed-text"


def get_embeddings() -> OllamaEmbeddings:

    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=get_settings().ollama_host,
    )


def embed_documents(
    texts: List[str],
) -> List[List[float]]:

    embeddings = get_embeddings()

    return embeddings.embed_documents(
        texts
    )


def embed_query(
    query: str,
) -> List[float]:

    embeddings = get_embeddings()

    return embeddings.embed_query(
        query
    )