"""Module A.1 — LLM triple extraction from free text.

Takes PubMed-style abstracts and extracts (subject, predicate, object) triples
using the LLM.  This is the *construction* arm of the co-refinement loop.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable

from kg_scaffold.generation.llm_client import LLMClient
from kg_scaffold.utils.prompts import TRIPLE_EXTRACTION
from kg_scaffold.utils.semmeddb import Triple, SEMMED_PREDICATES

logger = logging.getLogger(__name__)

_TRIPLE_RE = re.compile(r"^(.+?)\s*\|\s*(.+?)\s*\|\s*(.+)$")


def extract_triples(abstracts: Iterable[str], client: LLMClient | None = None,
                    predicates: list[str] | None = None) -> list[Triple]:
    """Extract triples from a collection of abstracts.

    Args:
        abstracts: iterable of text abstracts.
        client: LLM client (created if None).
        predicates: allowed predicate vocabulary (defaults to SemMedDB set).
    """
    client = client or LLMClient()
    preds = predicates or SEMMED_PREDICATES
    pred_str = ", ".join(preds[:20])  # keep prompt short
    triples: list[Triple] = []
    for ab in abstracts:
        ab = (ab or "").strip()
        if len(ab) < 20:
            continue
        prompt = TRIPLE_EXTRACTION.format(abstract=ab[:4000], predicates=pred_str)
        try:
            raw = client.complete(prompt, temperature=0.0)
        except Exception as exc:
            logger.warning("extraction failed: %s", exc)
            continue
        for line in raw.splitlines():
            t = _parse_line(line)
            if t:
                triples.append(t)
    logger.info("extracted %d triples from abstracts", len(triples))
    return triples


def _parse_line(line: str) -> Triple | None:
    line = line.strip().strip("-").strip()
    if not line or line.startswith("#"):
        return None
    m = _TRIPLE_RE.match(line)
    if not m:
        return None
    s, p, o = (g.strip() for g in m.groups())
    if not s or not p or not o:
        return None
    return Triple(subject=s, predicate=p.upper(), object=o)
