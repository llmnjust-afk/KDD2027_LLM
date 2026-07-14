"""Baseline 1 — Vanilla LLM (no retrieval, no KG)."""

from __future__ import annotations

from kg_scaffold.baselines.base import BaseMethod, RunConfig
from kg_scaffold.generation.hypothesis_gen import generate_hypotheses
from kg_scaffold.generation.llm_client import LLMClient


class VanillaLLM(BaseMethod):
    name = "vanilla_llm"

    def __init__(self, client: LLMClient | None = None):
        self.client = client or LLMClient()

    def run(self, question: str, seed_entity: str = "",
            cfg: RunConfig | None = None) -> list:
        cfg = cfg or RunConfig()
        hypos = generate_hypotheses(
            question=question,
            subgraph=None,
            snippets=None,
            client=self.client,
            num=cfg.num_hypotheses,
            use_kg=False,
        )
        for h in hypos:
            h.source_method = self.name
        return hypos
