from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from backend.app.rag.embeddings.embedder import embed_documents, embed_query


PROJECT_ROOT = Path(__file__).resolve().parents[4]

RAG_STORAGE_DIR = PROJECT_ROOT / "data" / "processed" / "rag"
INDEX_PATH = RAG_STORAGE_DIR / "knowledge_base.faiss"
METADATA_PATH = RAG_STORAGE_DIR / "metadata.json"


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)

    faiss.normalize_L2(vectors)

    return vectors


class VectorStore:
    def __init__(self):
        self.index = None
        self.metadata = []

    def build(self, chunks: list[dict]) -> None:
        if not chunks:
            raise ValueError("No chunks were provided.")

        texts = [chunk["text"] for chunk in chunks]

        print(f"Embedding {len(texts)} chunks...")

        embeddings = embed_documents(texts)

        matrix = np.asarray(embeddings, dtype=np.float32)

        matrix = _normalize_vectors(matrix)

        dimension = matrix.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(matrix)

        self.metadata = chunks

        RAG_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(INDEX_PATH))

        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

        print(f"FAISS index created successfully.")
        print(f"Vectors: {self.index.ntotal}")
        print(f"Dimension: {dimension}")
        print(f"Index: {INDEX_PATH}")
        print(f"Metadata: {METADATA_PATH}")

    def load(self) -> None:
        if not INDEX_PATH.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {INDEX_PATH}"
            )

        if not METADATA_PATH.exists():
            raise FileNotFoundError(
                f"Metadata not found: {METADATA_PATH}"
            )

        self.index = faiss.read_index(str(INDEX_PATH))

        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                "FAISS index size does not match metadata size."
            )

        print(f"FAISS index loaded: {self.index.ntotal} vectors")

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:

        if self.index is None:
            self.load()

        query_embedding = embed_query(query)

        query_vector = np.asarray(
            [query_embedding],
            dtype=np.float32,
        )

        query_vector = _normalize_vectors(query_vector)

        scores, indices = self.index.search(
            query_vector,
            top_k,
        )

        results = []

        for score, index in zip(scores[0], indices[0]):

            if index < 0:
                continue

            item = dict(self.metadata[index])

            item["similarity"] = float(score)

            results.append(item)

        return results