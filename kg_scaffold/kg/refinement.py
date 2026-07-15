"""Module A.3 — KG co-refinement.

Combines the learned ComplEx scores (Module A.2) with LLM verification
(Module A.1) into a single co-refinement loop:

  1. Score all triples with the learned model.
  2. Triples below ``min_confidence`` enter an LLM verification queue.
  3. The LLM decides: keep / delete / correct.
  4. The cleaned KG is returned with updated scores.

This is the core novelty: a *learned* model selects what the *symbolic* LLM
should review, closing the neural-symbolic loop on the KG itself.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Sequence

from kg_scaffold.generation.llm_client import LLMClient
from kg_scaffold.kg.completion import CompletionResult, attach_scores
from kg_scaffold.utils.prompts import TRIPLE_VERIFY
from kg_scaffold.utils.semmeddb import Triple, is_negative

logger = logging.getLogger(__name__)

_VERDICT_RE = re.compile(r"VERDICT:\s*(\w+)", re.IGNORECASE)
_CORRECTED_RE = re.compile(r"CORRECTED:\s*(.+)", re.IGNORECASE)


@dataclass
class RefinementStats:
    total: int
    kept: int
    deleted: int
    corrected: int
    verified_by_llm: int


def co_refine(triples: Sequence[Triple],
              completion: CompletionResult,
              client: LLMClient | None = None,
              min_confidence: float = 0.15,
              context_provider=None,
              strict_delete: bool = True) -> tuple[list[Triple], RefinementStats]:
    """Run the neural-symbolic co-refinement loop.

    Args:
        triples: input (possibly noisy) triples.
        completion: learned scoring result from ComplEx.
        client: LLM client for verification.
        min_confidence: triples below this score are LLM-verified.
        context_provider: optional callable(triple)->str returning context text
            (e.g. PubMed snippets) to help the LLM judge.
        strict_delete: if True, only delete "contradicted" triples (keep
            "unverifiable"). If False, delete both (original behavior).

    Returns:
        (refined_triples, stats)
    """
    client = client or LLMClient()
    scored = attach_scores(list(triples), completion)

    kept: list[Triple] = []
    stats = RefinementStats(total=len(scored), kept=0, deleted=0,
                            corrected=0, verified_by_llm=0)

    for t in scored:
        if is_negative(t.predicate):
            stats.deleted += 1
            continue
        if t.score >= min_confidence:
            kept.append(t)
            stats.kept += 1
            continue
        stats.verified_by_llm += 1
        verdict, corrected = _verify_triple(t, client, context_provider)
        if verdict == "supported":
            kept.append(t)
            stats.kept += 1
        elif verdict == "contradicted" and corrected and corrected.lower() not in ("none", ""):
            kept.append(Triple(t.subject, t.predicate, corrected,
                               t.subject_cui, t.object_cui, score=t.score))
            stats.corrected += 1
        elif verdict == "contradicted":
            stats.deleted += 1
        elif strict_delete:
            kept.append(t)
            stats.kept += 1
        else:
            stats.deleted += 1

    logger.info("Co-refinement: %s", stats)
    return kept, stats


def _verify_triple(triple: Triple, client: LLMClient,
                   context_provider) -> tuple[str, str | None]:
    """Ask the LLM to verify one low-confidence triple.

    Returns (verdict, corrected_object_or_None).
    """
    context = ""
    if context_provider:
        try:
            context = context_provider(triple) or ""
        except Exception:
            context = ""
    if not context:
        context = "(no additional context available)"

    prompt = TRIPLE_VERIFY.format(
        subject=triple.subject,
        predicate=triple.predicate,
        object=triple.object,
        context=context[:2000],
    )
    try:
        raw = client.complete(prompt, temperature=0.0)
    except Exception as exc:
        logger.warning("verify failed (%s): %s", triple, exc)
        return "unverifiable", None

    verdict = "unverifiable"
    m = _VERDICT_RE.search(raw)
    if m:
        verdict = m.group(1).lower()
    corrected = None
    m2 = _CORRECTED_RE.search(raw)
    if m2:
        val = m2.group(1).strip()
        if val.lower() not in ("none", ""):
            corrected = val
    return verdict, corrected


def filter_by_score(triples: Sequence[Triple], min_confidence: float) -> list[Triple]:
    """Pure score-based filtering (no LLM) — used in ablation."""
    return [t for t in triples if t.score >= min_confidence]
