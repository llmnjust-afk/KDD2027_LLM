"""Baseline 2 — BM25 + LLM (Naive RAG).

Uses a lightweight in-memory BM25 over text snippets.  Falls back to a
simple TF-IDF ranking if pyserini is unavailable.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Sequence

from kg_scaffold.baselines.base import BaseMethod, RunConfig
from kg_scaffold.generation.hypothesis_gen import generate_hypotheses
from kg_scaffold.generation.llm_client import LLMClient

logger = logging.getLogger(__name__)


class BM25Index:
    """Minimal in-memory BM25 (no external deps)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: list[list[str]] = []
        self.df: Counter = Counter()
        self.avg_len = 0.0
        self.N = 0

    def build(self, texts: Sequence[str]) -> None:
        self.docs = [t.lower().split() for t in texts]
        self.N = len(self.docs)
        for d in self.docs:
            seen = set(d)
            for w in seen:
                self.df[w] += 1
        self.avg_len = sum(len(d) for d in self.docs) / max(1, self.N)

    def search(self, query: str, topk: int = 8) -> list[tuple[str, float]]:
        q = query.lower().split()
        scores = []
        for i, d in enumerate(self.docs):
            tf = Counter(d)
            s = 0.0
            for w in q:
                if w not in tf:
                    continue
                idf = math.log((self.N - self.df[w] + 0.5) /
                               (self.df[w] + 0.5) + 1.0)
                denom = tf[w] + self.k1 * (1 - self.b + self.b * len(d) / self.avg_len)
                s += idf * (tf[w] * (self.k1 + 1)) / denom
            scores.append((i, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        orig = self._orig_texts
        return [(orig[i], s) for i, s in scores[:topk] if s > 0]

    @property
    def _orig_texts(self) -> list[str]:
        return [" ".join(d) for d in self.docs]


class BM25RAG(BaseMethod):
    name = "bm25_rag"

    def __init__(self, corpus: Sequence[str], client: LLMClient | None = None):
        self.client = client or LLMClient()
        self.index = BM25Index()
        self.corpus = list(corpus)
        self.index.build(self.corpus)

    def run(self, question: str, seed_entity: str = "",
            cfg: RunConfig | None = None) -> list:
        cfg = cfg or RunConfig()
        results = self.index.search(question, topk=8)
        snippets = [r[0] for r in results]
        hypos = generate_hypotheses(
            question=question,
            subgraph=None,
            snippets=snippets,
            client=self.client,
            num=cfg.num_hypotheses,
            use_kg=False,
        )
        for h in hypos:
            h.source_method = self.name
        return hypos
