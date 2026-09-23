from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "rag" / "metadata.json"

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# BM25 hyperparameters (standard defaults from Robertson & Zaragoza, 2009).
_K1 = 1.5
_B = 0.75


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """A minimal, dependency-free BM25 index over the knowledge-base chunks.

    Loads the same chunk metadata the FAISS vector store uses, so lexical
    (BM25) and dense (FAISS cosine) scores are computed over identical
    documents. Built once per process and reused for every rerank.
    """

    def __init__(self) -> None:
        self._chunk_ids: list[str] = []
        self._doc_freqs: list[Counter] = []
        self._doc_lengths: list[int] = []
        self._doc_freq: Counter = Counter()
        self._avg_doc_length = 0.0
        self._n_docs = 0
        self._loaded = False

    def load(self) -> None:
        if not METADATA_PATH.exists():
            raise FileNotFoundError(f"RAG metadata not found: {METADATA_PATH}")

        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        self._chunk_ids = []
        self._doc_freqs = []
        self._doc_lengths = []
        self._doc_freq = Counter()

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            tokens = tokenize(chunk.get("text", ""))
            term_counts = Counter(tokens)

            self._chunk_ids.append(chunk_id)
            self._doc_freqs.append(term_counts)
            self._doc_lengths.append(len(tokens))
            self._doc_freq.update(term_counts.keys())

        self._n_docs = len(self._chunk_ids)
        self._avg_doc_length = (
            sum(self._doc_lengths) / self._n_docs if self._n_docs else 0.0
        )
        self._loaded = True

    def _idf(self, term: str) -> float:
        df = self._doc_freq.get(term, 0)
        # BM25+ style IDF, floored at a small positive value so unseen
        # terms don't zero out the whole score.
        return max(
            math.log((self._n_docs - df + 0.5) / (df + 0.5) + 1.0),
            1e-6,
        )

    def score_by_chunk_id(self, query: str) -> dict[str, float]:
        """Return {chunk_id: bm25_score} for every chunk in the corpus."""

        if not self._loaded:
            self.load()

        query_terms = tokenize(query)
        if not query_terms or self._n_docs == 0:
            return {chunk_id: 0.0 for chunk_id in self._chunk_ids}

        scores: dict[str, float] = {}

        for chunk_id, term_counts, doc_length in zip(
            self._chunk_ids, self._doc_freqs, self._doc_lengths
        ):
            score = 0.0
            for term in query_terms:
                freq = term_counts.get(term, 0)
                if freq == 0:
                    continue
                idf = self._idf(term)
                denom = freq + _K1 * (
                    1 - _B + _B * doc_length / max(self._avg_doc_length, 1e-6)
                )
                score += idf * (freq * (_K1 + 1)) / denom
            scores[chunk_id] = score

        return scores


bm25_index = BM25Index()
