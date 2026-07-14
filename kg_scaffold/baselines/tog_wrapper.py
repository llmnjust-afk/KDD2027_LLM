"""Baseline 4 — Think-on-Graph (ToG) wrapper.

Re-implements the core ToG reasoning loop (Sun et al., ICLR 2024):
  1. Given a topic entity, ask the LLM to select the most promising relations.
  2. Expand one hop along those relations to get neighbor entities.
  3. Repeat relation selection + entity pruning for ``max_depth`` rounds.
  4. Use the explored subgraph + LLM to answer / generate hypotheses.

This is our strongest KG-augmented baseline.  We keep the interface identical
to our method so they are directly comparable.
"""

from __future__ import annotations

import logging
from typing import Sequence

from kg_scaffold.baselines.base import BaseMethod, RunConfig
from kg_scaffold.generation.hypothesis_gen import generate_hypotheses, Hypothesis
from kg_scaffold.generation.llm_client import LLMClient
from kg_scaffold.retrieval.subgraph import SubgraphRetriever, Subgraph
from kg_scaffold.utils.prompts import RELATION_EXPLORE
from kg_scaffold.utils.semmeddb import Triple, build_graph, is_negative

logger = logging.getLogger(__name__)


class ToGWrapper(BaseMethod):
    """Think-on-Graph: LLM-guided relation exploration over the KG."""

    name = "tog"

    def __init__(self, triples: Sequence[Triple],
                 client: LLMClient | None = None,
                 max_depth: int = 3, beam_width: int = 3,
                 topk_relations: int = 3):
        self.client = client or LLMClient()
        self.retriever = SubgraphRetriever(triples, max_hops=1, topk=1000)
        self.graph = build_graph(triples)
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.topk_relations = topk_relations

    def run(self, question: str, seed_entity: str = "",
            cfg: RunConfig | None = None) -> list[Hypothesis]:
        cfg = cfg or RunConfig()
        subgraph = self._explore(seed_entity, question)
        hypos = generate_hypotheses(
            question=question,
            subgraph=subgraph,
            snippets=None,
            client=self.client,
            num=cfg.num_hypotheses,
            use_kg=True,
        )
        for h in hypos:
            h.source_method = self.name
        return hypos

    def _explore(self, seed_entity: str, question: str) -> Subgraph:
        """LLM-guided beam search over the KG."""
        current = self.retriever._resolve_entity(seed_entity)
        if not current:
            return Subgraph(root=seed_entity)

        collected: list[Triple] = []
        seen_entities = {current}
        frontier = [current]

        for depth in range(self.max_depth):
            if not frontier:
                break
            next_frontier = []
            for ent in frontier:
                relations = self._available_relations(ent)
                if not relations:
                    continue
                chosen = self._select_relations(ent, question, relations)
                for rel in chosen:
                    neighbors = self._expand(ent, rel)
                    for nbr, triple in neighbors:
                        collected.append(triple)
                        if nbr not in seen_entities:
                            seen_entities.add(nbr)
                            next_frontier.append(nbr)
            frontier = next_frontier[: self.beam_width]

        # rank by score and truncate
        collected.sort(key=lambda t: t.score, reverse=True)
        return Subgraph(root=seed_entity, triples=collected[:40])

    def _available_relations(self, entity: str) -> list[str]:
        rels = set()
        if self.graph.out_degree(entity):
            for _, _, d in self.graph.out_edges(entity, data=True):
                p = d.get("predicate", "")
                if p and not is_negative(p):
                    rels.add(p)
        if self.graph.in_degree(entity):
            for _, _, d in self.graph.in_edges(entity, data=True):
                p = d.get("predicate", "")
                if p and not is_negative(p):
                    rels.add(p)
        return sorted(rels)

    def _select_relations(self, entity: str, question: str,
                          relations: list[str]) -> list[str]:
        prompt = RELATION_EXPLORE.format(
            entity=entity,
            relations=", ".join(relations),
            topk=self.topk_relations,
        )
        try:
            raw = self.client.complete(prompt, temperature=0.0)
        except Exception:
            return relations[: self.topk_relations]
        chosen = [r.strip() for r in raw.strip().splitlines() if r.strip()]
        valid = [r for r in chosen if r in relations]
        return (valid or relations)[: self.topk_relations]

    def _expand(self, entity: str, relation: str) -> list[tuple[str, Triple]]:
        out = []
        if self.graph.out_degree(entity):
            for nbr, edges in self.graph[entity].items():
                for d in edges.values():
                    if d.get("predicate") == relation:
                        t = d.get("triple")
                        if t:
                            out.append((nbr, t))
        if self.graph.in_degree(entity):
            for pred in self.graph.predecessors(entity):
                for d in self.graph[pred][entity].values():
                    if d.get("predicate") == relation:
                        t = d.get("triple")
                        if t:
                            out.append((pred, t))
        return out
