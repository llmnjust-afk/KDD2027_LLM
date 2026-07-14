"""Module D — faithfulness verification.

Verifies that each generated hypothesis is entailed by its KG path.  Produces
the ``faithfulness@k`` metric: the fraction of top-k hypotheses whose claims are
fully supported by an explicit KG path.

This metric is the paper's novel contribution — it quantifies the
neural-symbolic *grounding* of generated insights, and we validate it with
human annotation (see ``scripts/07_annotation_prep.py``).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Sequence

from kg_scaffold.generation.llm_client import LLMClient
from kg_scaffold.generation.hypothesis_gen import Hypothesis
from kg_scaffold.utils.prompts import FAITHFULNESS_CHECK

logger = logging.getLogger(__name__)

_ENTAIL_RE = re.compile(r"ENTAILMENT:\s*(\w+)", re.IGNORECASE)

ENTAILED = "entailed"
PARTIAL = "partial"
NONE = "none"


@dataclass
class FaithfulnessReport:
    per_hypothesis: list[str]   # entailed|partial|none for each
    faithfulness_at_k: dict[int, float]

    @property
    def mean(self) -> float:
        labels = {ENTAILED: 1.0, PARTIAL: 0.5, NONE: 0.0}
        if not self.per_hypothesis:
            return 0.0
        return sum(labels.get(x, 0.0) for x in self.per_hypothesis) / len(self.per_hypothesis)


def verify_hypotheses(hypotheses: Sequence[Hypothesis],
                      client: LLMClient | None = None,
                      top_k: Sequence[int] = (1, 5, 10)) -> FaithfulnessReport:
    """Check each hypothesis against its KG path.

    Returns a report with per-hypothesis labels and faithfulness@k.
    """
    client = client or LLMClient()
    labels: list[str] = []

    for h in hypotheses:
        if not h.kg_path:
            labels.append(NONE)
            continue
        label = _check_one(h, client)
        labels.append(label)
        h.faithfulness = label

    # faithfulness@k = fraction of top-k that are at least partially entailed
    fk = {}
    for k in top_k:
        top = labels[:k]
        if not top:
            fk[k] = 0.0
            continue
        fk[k] = sum(1 for x in top if x in (ENTAILED, PARTIAL)) / len(top)
    return FaithfulnessReport(per_hypothesis=labels, faithfulness_at_k=fk)


def _check_one(h: Hypothesis, client: LLMClient) -> str:
    prompt = FAITHFULNESS_CHECK.format(
        hypothesis=h.text,
        kg_path=h.kg_path,
    )
    try:
        raw = client.complete(prompt, temperature=0.0)
    except Exception as exc:
        logger.warning("faithfulness check failed: %s", exc)
        return NONE
    m = _ENTAIL_RE.search(raw)
    if not m:
        return NONE
    label = m.group(1).lower()
    if label not in (ENTAILED, PARTIAL, NONE):
        return NONE
    return label


def faithfulness_at_k(hypotheses: Sequence[Hypothesis],
                      k: int) -> float:
    """Convenience: compute faithfulness@k from already-labeled hypotheses."""
    top = hypotheses[:k]
    if not top:
        return 0.0
    good = sum(1 for h in top if h.faithfulness in (ENTAILED, PARTIAL))
    return good / len(top)
