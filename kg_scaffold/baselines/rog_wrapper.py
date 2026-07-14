"""Baseline 5 — Reasoning-on-Graphs (RoG) wrapper.

Re-implements the core RoG reasoning loop (Luo et al., ICLR 2024):
  1. Retrieve reasoning paths from the KG using a beam search guided by
     relation priors (which relations are most likely to lead to answers).
  2. Format the retrieved paths as natural-language context.
  3. Feed the context to the LLM to generate answers/hypotheses.

RoG's key distinction from ToG: it retrieves *complete paths* first, then
generates from those paths, rather than exploring step-by-step. It also
explicitly claims "faithful reasoning" — making it the most important
baseline to compare against for our faithfulness contribution.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Sequence

import networkx as nx

from kg_scaffold.baselines.base import BaseMethod, RunConfig
from kg_scaffold.generation.hypothesis_gen import generate_hypotheses, Hypothesis
from kg_scaffold.generation.llm_client import LLMClient
from kg_scaffold.retrieval.subgraph import Subgraph, SubgraphRetriever
from kg_scaffold.utils.semmeddb import Triple, build_graph, is_negative

logger = logging.getLogger(__name__)


class RoGWrapper(BaseMethod):
    """Reasoning-on-Graphs: path-constrained retrieval + LLM generation.

    Differs from ToG in that it:
    - Retrieves complete multi-hop paths (not step-by-step exploration)
    - Formats paths as NL reasoning chains
    - Does NOT refine the KG (same limitation as ToG, which we highlight)
    """

    name = "rog"

    def __init__(self, triples: Sequence[Triple],
                 client: LLMClient | None = None,
                 max_paths: int = 20, max_hops: int = 3,
                 beam_width: int = 5):
        self.client = client or LLMClient()
        self.graph = build_graph(triples)
        self.retriever = SubgraphRetriever(triples, max_hops=max_hops, topk=1000)
        self.max_paths = max_paths
        self.max_hops = max_hops
        self.beam_width = beam_width

    def run(self, question: str, seed_entity: str = "",
            cfg: RunConfig | None = None) -> list[Hypothesis]:
        cfg = cfg or RunConfig()

        # Retrieve complete paths from seed entity
        paths = self._retrieve_paths(seed_entity, question)

        # Format paths as a subgraph for generation
        path_triples = []
        for path in paths:
            path_triples.extend(path)
        # dedup
        seen = set()
        unique_triples = []
        for t in path_triples:
            key = t.as_tuple()
            if key not in seen:
                seen.add(key)
                unique_triples.append(t)

        subgraph = Subgraph(root=seed_entity, triples=unique_triples[:40])

        # Generate using path-constrained context
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

    def _retrieve_paths(self, seed_entity: str,
                        question: str) -> list[list[Triple]]:
        """Retrieve complete multi-hop paths using beam search with relation priors."""
        current = self.retriever._resolve_entity(seed_entity)
        if not current:
            return []

        # BFS-based path retrieval with beam search
        all_paths = []
        frontier = [(current, [])]  # (entity, path_so_far)

        for hop in range(self.max_hops):
            if not frontier:
                break
            next_frontier = []
            for entity, path in frontier:
                # Get neighbors with relation info
                neighbors = self._get_neighbors(entity)
                if not neighbors:
                    continue

                # Score relations by relevance to the question
                scored_neighbors = []
                for nbr, relation, triple in neighbors:
                    if is_negative(relation):
                        continue
                    # Simple relevance: relation name overlap with question
                    rel_score = self._relation_relevance(relation, question)
                    path_score = sum(t.score for t in path) + triple.score + rel_score
                    scored_neighbors.append((nbr, relation, triple, path_score))

                # Keep top-beam_width
                scored_neighbors.sort(key=lambda x: x[3], reverse=True)
                for nbr, relation, triple, _ in scored_neighbors[:self.beam_width]:
                    new_path = path + [triple]
                    all_paths.append(new_path)
                    next_frontier.append((nbr, new_path))

            frontier = next_frontier[:self.max_paths]

        # Rank all paths by total score and return top-k
        all_paths.sort(key=lambda p: sum(t.score for t in p), reverse=True)
        return all_paths[:self.max_paths]

    def _get_neighbors(self, entity: str) -> list[tuple[str, str, Triple]]:
        """Get (neighbor, relation, triple) for all neighbors of entity."""
        results = []
        if self.graph.out_degree(entity):
            for nbr, edges in self.graph[entity].items():
                for d in edges.values():
                    t = d.get("triple")
                    if t:
                        results.append((nbr, d.get("predicate", ""), t))
        if self.graph.in_degree(entity):
            for pred in self.graph.predecessors(entity):
                for d in self.graph[pred][entity].values():
                    t = d.get("triple")
                    if t:
                        results.append((pred, d.get("predicate", ""), t))
        return results

    def _relation_relevance(self, relation: str, question: str) -> float:
        """Score how relevant a relation is to the question (simple keyword overlap)."""
        rel_words = set(relation.lower().replace("_", " ").split())
        q_words = set(question.lower().split())
        overlap = len(rel_words & q_words)
        return overlap * 0.1
