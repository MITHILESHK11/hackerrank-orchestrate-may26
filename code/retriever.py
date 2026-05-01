from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from models import Chunk, RetrievalHit, RetrievalResult


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


@dataclass
class RetrieverConfig:
    bm25_candidates: int = 20
    final_k: int = 5
    semantic_threshold: float = 0.20
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


class HybridRetriever:
    def __init__(self, config: RetrieverConfig | None = None) -> None:
        self.config = config or RetrieverConfig()
        self._chunks: list[Chunk] = []
        self._tokens: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        self._embedder = None
        self._semantic_ready = False

    def build_index(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build index with empty chunks.")
        self._chunks = chunks
        self._tokens = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(self._tokens)
        try:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.config.model_name)
            self._semantic_ready = True
        except Exception:
            self._embedder = None
            self._semantic_ready = False

    def _domain_indexes(self, domain_hint: str | None) -> list[int]:
        if not domain_hint or domain_hint.lower() == "generic":
            return list(range(len(self._chunks)))
        domain = domain_hint.lower().strip()
        indexes = [
            idx for idx, chunk in enumerate(self._chunks) if chunk.source_domain == domain
        ]
        if indexes:
            return indexes
        return list(range(len(self._chunks)))

    def retrieve(self, query: str, domain_hint: str | None, k: int | None = None) -> RetrievalResult:
        if self._bm25 is None:
            raise RuntimeError("Retriever index not built.")
        if not query.strip():
            return RetrievalResult(hits=[], confidence="low")

        final_k = k or self.config.final_k
        query_tokens = _tokenize(query)
        if not query_tokens:
            return RetrievalResult(hits=[], confidence="low")

        domain_indexes = self._domain_indexes(domain_hint)
        all_scores = self._bm25.get_scores(query_tokens)
        scored = [(idx, float(all_scores[idx])) for idx in domain_indexes]
        scored.sort(key=lambda x: x[1], reverse=True)
        top_candidates = scored[: self.config.bm25_candidates]
        if not top_candidates:
            return RetrievalResult(hits=[], confidence="low")

        candidate_indexes = [idx for idx, _ in top_candidates]
        candidate_chunks = [self._chunks[idx] for idx in candidate_indexes]

        if self._semantic_ready and self._embedder is not None:
            query_emb = self._embedder.encode(
                [query],
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]
            doc_emb = self._embedder.encode(
                [chunk.text for chunk in candidate_chunks],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            semantic_scores = np.dot(doc_emb, query_emb).tolist()
            ordered = sorted(
                zip(candidate_chunks, semantic_scores),
                key=lambda x: float(x[1]),
                reverse=True,
            )[:final_k]
            high_count = sum(1 for _, score in ordered if float(score) > self.config.semantic_threshold)
            confidence = "high" if high_count >= 2 else "low"
            hits = [RetrievalHit(chunk=chunk, score=float(score)) for chunk, score in ordered]
            return RetrievalResult(hits=hits, confidence=confidence)

        bm25_values = [score for _, score in top_candidates]
        max_score = max(bm25_values) if bm25_values else 1.0
        if max_score == 0:
            max_score = 1.0
        normalized = [score / max_score for score in bm25_values]
        ordered = list(zip(candidate_chunks, normalized))[:final_k]
        high_count = sum(1 for _, score in ordered if float(score) >= 0.35)
        confidence = "high" if high_count >= 2 else "low"
        hits = [RetrievalHit(chunk=chunk, score=float(score)) for chunk, score in ordered]
        return RetrievalResult(hits=hits, confidence=confidence)


if __name__ == "__main__":
    from pathlib import Path

    from corpus_loader import load_corpus

    base_dir = Path(__file__).resolve().parents[1] / "data"
    chunks_ = load_corpus(str(base_dir))
    retriever = HybridRetriever()
    retriever.build_index(chunks_)
    result = retriever.retrieve("how to add extra time for a candidate", "hackerrank")
    assert result.hits
    print(result.hits[0].chunk.text[:200])
