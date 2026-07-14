"""Baseline 3 — Dense retrieval + LLM (Dense RAG).

Uses sentence-transformer embeddings to retrieve top-k snippets, then feeds
them to the LLM without KG scaffold.
"""

from __future__ import annotations

from typing import Sequence

from kg_scaffold.baselines.base import BaseMethod, RunConfig
from kg_scaffold.generation.hypothesis_gen import generate_hypotheses
from kg_scaffold.generation.llm_client import LLMClient
from kg_scaffold.retrieval.embed_index import DenseIndex


class DenseRAG(BaseMethod):
    name = "dense_rag"

    def __init__(self, corpus: Sequence[str], client: LLMClient | None = None,
                 model_name: str = "BAAI/bge-small-en-v1.5"):
        self.client = client or LLMClient()
        self.index = DenseIndex(model_name=model_name)
        self.index.build(corpus)
        self.corpus = list(corpus)

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
