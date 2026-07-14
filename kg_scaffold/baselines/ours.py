"""Our method — KG-SCoRE (Knowledge-Graph-Symbolic Co-Refinement).

Ties together all four modules into a single end-to-end pipeline:

  [A] co-refine the KG (ComplEx scoring + LLM verification)
  [B] retrieve a scored multi-hop subgraph for the seed entity
  [C] generate KG-scaffolded hypotheses
  [D] verify faithfulness and attach labels

Supports ablation toggles via RunConfig so the same class produces every row
of the ablation table.
"""

from __future__ import annotations

import logging
from typing import Sequence

from kg_scaffold.baselines.base import BaseMethod, RunConfig
from kg_scaffold.generation.hypothesis_gen import generate_hypotheses, Hypothesis
from kg_scaffold.generation.llm_client import LLMClient
from kg_scaffold.kg.completion import CompletionResult
from kg_scaffold.kg.refinement import co_refine, filter_by_score
from kg_scaffold.retrieval.subgraph import SubgraphRetriever, Subgraph
from kg_scaffold.verification.faithfulness import verify_hypotheses
from kg_scaffold.utils.semmeddb import Triple

logger = logging.getLogger(__name__)


class KGSCoRE(BaseMethod):
    """Full KG-Symbolic Co-Refinement pipeline."""

    name = "kg_score"

    def __init__(
        self,
        triples: Sequence[Triple],
        completion: CompletionResult | None = None,
        client: LLMClient | None = None,
        max_hops: int = 2,
        topk_triples: int = 40,
        min_confidence: float = 0.15,
        text_corpus: Sequence[str] | None = None,
    ):
        self.client = client or LLMClient()
        self.completion = completion
        self.min_confidence = min_confidence
        self.max_hops = max_hops
        self.topk_triples = topk_triples
        self.text_corpus = list(text_corpus) if text_corpus else []
        self._refined_triples: list[Triple] = list(triples)
        self._retriever: SubgraphRetriever | None = None

    @property
    def retriever(self) -> SubgraphRetriever:
        if self._retriever is None:
            self._retriever = SubgraphRetriever(
                self._refined_triples,
                max_hops=self.max_hops,
                topk=self.topk_triples,
            )
        return self._retriever

    def refine_kg(self, triples: Sequence[Triple]) -> list[Triple]:
        """Run Module A co-refinement once (cached)."""
        if self.completion is None:
            logger.info("No completion model — skipping co-refinement.")
            self._refined_triples = list(triples)
            self._retriever = None
            return self._refined_triples

        refined, stats = co_refine(
            triples, self.completion, self.client,
            min_confidence=self.min_confidence,
            context_provider=self._context_for_triple,
        )
        logger.info("KG refined: %d -> %d (%s)", len(triples), len(refined), stats)
        self._refined_triples = refined
        self._retriever = None  # force rebuild
        return refined

    def _context_for_triple(self, triple: Triple) -> str:
        """Find text snippets mentioning the triple entities (for LLM verify)."""
        if not self.text_corpus:
            return ""
        s, o = triple.subject.lower(), triple.object.lower()
        matches = [t for t in self.text_corpus if s in t.lower() and o in t.lower()]
        return " ".join(matches[:2])

    def run(self, question: str, seed_entity: str = "",
            cfg: RunConfig | None = None) -> list[Hypothesis]:
        cfg = cfg or RunConfig()

        # [A] refine if not yet done and completion available
        if cfg.use_completion and self.completion is not None:
            self.refine_kg(self._refined_triples)

        # [B] retrieve subgraph
        if cfg.use_kg:
            subgraph = self.retriever.retrieve(seed_entity)
        else:
            subgraph = None

        # retrieve text snippets via simple keyword match
        snippets = self._keyword_snippets(question, seed_entity)

        # [C] generate
        hypos = generate_hypotheses(
            question=question,
            subgraph=subgraph,
            snippets=snippets,
            client=self.client,
            num=cfg.num_hypotheses,
            use_kg=cfg.use_kg,
        )
        for h in hypos:
            h.source_method = self.name

        # [D] faithfulness
        if cfg.use_faithfulness and cfg.use_kg:
            report = verify_hypotheses(hypos, self.client)
            logger.info("Faithfulness: %s", report.faithfulness_at_k)

        return hypos

    def _keyword_snippets(self, question: str, seed: str,
                          topk: int = 5) -> list[str]:
        if not self.text_corpus:
            return []
        kw = seed.lower()
        matches = [t for t in self.text_corpus if kw in t.lower()]
        return matches[:topk]
