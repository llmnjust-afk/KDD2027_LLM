"""Module B — symbolic multi-hop subgraph retrieval.

Given a query entity (e.g. a disease), explores the KG up to ``max_hops`` hops
to extract a scored subgraph.  Triples are ranked by ComplEx confidence and
relevance, then truncated to ``topk_triples``.

This is the symbolic context that scaffolds the LLM in Module C.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

import networkx as nx

from kg_scaffold.utils.semmeddb import Triple, build_graph, is_negative

logger = logging.getLogger(__name__)


@dataclass
class Subgraph:
    """A retrieved symbolic context."""
    root: str
    triples: list[Triple] = field(default_factory=list)

    def as_text(self) -> str:
        """Render as pipe-delimited text for the LLM prompt."""
        lines = []
        for t in self.triples:
            lines.append(f"{t.subject} | {t.predicate} | {t.object}")
        return "\n".join(lines)

    def paths_from(self, source: str) -> list[list[Triple]]:
        """Return all simple paths (as triple chains) from source to leaves."""
        if not self.triples:
            return []
        g = build_graph(self.triples)
        paths = []
        for node in g.nodes:
            if node == source:
                continue
            try:
                for sp in nx.all_simple_paths(g, source, node,
                                              cutoff=4):
                    path_triples = _path_to_triples(sp, g)
                    if path_triples:
                        paths.append(path_triples)
            except nx.NetworkXError:
                continue
        return paths


def _path_to_triples(path: list[str], g: nx.MultiDiGraph) -> list[Triple]:
    triples = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if g.has_edge(u, v):
            data = g[u][v]
            best = max(data.values(), key=lambda d: d.get("score", 0))
            triples.append(best["triple"])
    return triples


class SubgraphRetriever:
    """Multi-hop symbolic retriever over a KG."""

    def __init__(self, triples: Sequence[Triple], max_hops: int = 2,
                 topk: int = 40):
        self.graph = build_graph(triples)
        self.max_hops = max_hops
        self.topk = topk
        self._entity_index = _build_entity_index(triples)

    def retrieve(self, query_entity: str,
                 direction: str = "both") -> Subgraph:
        """Retrieve a scored subgraph rooted at ``query_entity``.

        Args:
            query_entity: the seed entity (e.g. disease name).
            direction: "out", "in", or "both".
        """
        seed = self._resolve_entity(query_entity)
        if not seed:
            logger.debug("entity not in KG: %s", query_entity)
            return Subgraph(root=query_entity, triples=[])

        collected: list[Triple] = []
        visited_edges = set()

        # BFS expansion
        frontier = {seed}
        for hop in range(self.max_hops):
            next_frontier = set()
            for node in frontier:
                neighbors = self._neighbors(node, direction)
                for nbr, edge_data in neighbors:
                    key = (node, nbr, edge_data.get("predicate"))
                    if key in visited_edges:
                        continue
                    visited_edges.add(key)
                    t = edge_data.get("triple")
                    if t and not is_negative(t.predicate):
                        collected.append(t)
                        next_frontier.add(nbr)
            frontier = next_frontier
            if not frontier:
                break

        # Rank by ComplEx score (desc) and truncate
        collected.sort(key=lambda t: t.score, reverse=True)
        collected = collected[: self.topk]
        return Subgraph(root=seed, triples=collected)

    def retrieve_paths(self, source: str, target: str | None = None,
                       max_paths: int = 20) -> list[list[Triple]]:
        """Retrieve explicit multi-hop paths between source and target.

        If target is None, returns paths to all reachable nodes (top by score).
        """
        src = self._resolve_entity(source)
        if not src:
            return []
        if target:
            tgt = self._resolve_entity(target)
            if not tgt:
                return []
            try:
                raw_paths = nx.all_simple_paths(self.graph, src, tgt,
                                                cutoff=self.max_hops + 2)
            except nx.NetworkXError:
                return []
            return [_path_to_triples(p, self.graph)
                    for p in list(raw_paths)[:max_paths]
                    if _path_to_triples(p, self.graph)]
        else:
            sub = self.retrieve(source)
            return sub.paths_from(src)[:max_paths]

    def _neighbors(self, node: str, direction: str):
        results = []
        if direction in ("out", "both") and self.graph.out_degree(node):
            for nbr, ed in self.graph[node].items():
                for edata in ed.values():
                    results.append((nbr, edata))
        if direction in ("in", "both") and self.graph.in_degree(node):
            for pred in self.graph.predecessors(node):
                for edata in self.graph[pred][node].values():
                    results.append((pred, edata))
        return results

    def _resolve_entity(self, mention: str) -> str | None:
        """Fuzzy entity resolution against the KG."""
        m = mention.lower().strip()
        if m in self._entity_index:
            return self._entity_index[m]
        # substring match
        for ent in self._entity_index:
            if m in ent or ent in m:
                return self._entity_index[ent]
        return None


def _build_entity_index(triples: Sequence[Triple]) -> dict[str, str]:
    """Map lowercase entity -> canonical entity name."""
    idx = {}
    for t in triples:
        idx.setdefault(t.subject.lower(), t.subject)
        idx.setdefault(t.object.lower(), t.object)
    return idx


def merge_subgraphs(subs: Sequence[Subgraph], topk: int = 40) -> Subgraph:
    """Merge multiple subgraphs, dedup, and truncate."""
    seen = set()
    merged: list[Triple] = []
    for sub in subs:
        for t in sub.triples:
            key = t.as_tuple()
            if key in seen:
                continue
            seen.add(key)
            merged.append(t)
    merged.sort(key=lambda t: t.score, reverse=True)
    return Subgraph(root="|".join(s.root for s in subs),
                    triples=merged[:topk])
