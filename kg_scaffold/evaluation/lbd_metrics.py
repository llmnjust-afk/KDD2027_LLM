"""LBD evaluation metrics.

Computes re-discovery hit@k, MRR, and aggregate accuracy over the
Swanson/Cameron gold-standard pairs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from kg_scaffold.generation.hypothesis_gen import Hypothesis
from kg_scaffold.utils.lbd_gold import LBDPair, hit_at_k, mrr

logger = logging.getLogger(__name__)


@dataclass
class LBDEvalResult:
    method: str
    num_pairs: int
    hit_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    faithfulness_at_k: dict[int, float] = field(default_factory=dict)
    per_pair: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "num_pairs": self.num_pairs,
            "hit_at_k": self.hit_at_k,
            "mrr": self.mrr,
            "faithfulness_at_k": self.faithfulness_at_k,
        }


def evaluate_lbd(pairs: Sequence[LBDPair],
                 predictions: dict[str, list[Hypothesis]],
                 top_k: Sequence[int] = (1, 5, 10),
                 method_name: str = "") -> LBDEvalResult:
    """Evaluate re-discovery performance.

    Args:
        pairs: gold-standard LBD pairs.
        predictions: mapping pair.id -> list of generated Hypothesis.
        top_k: k values for hit@k.
        method_name: label for the result.
    """
    per_pair = []
    hit_counts = {k: 0 for k in top_k}
    mrrs = []
    faith_counts = {k: 0 for k in top_k}
    n = 0

    for pair in pairs:
        hypos = predictions.get(pair.id, [])
        if not hypos:
            continue
        n += 1
        targets = [h.target_entity for h in hypos]
        hits = hit_at_k(targets, pair.target, top_k)
        for k in top_k:
            if hits[k]:
                hit_counts[k] += 1
        mrrs.append(mrr(targets, pair.target))

        # faithfulness (already labeled by Module D)
        for k in top_k:
            top = hypos[:k]
            if top:
                good = sum(1 for h in top
                           if h.faithfulness in ("entailed", "partial"))
                faith_counts[k] += good / len(top)

        per_pair.append({
            "pair_id": pair.id,
            "source": pair.source,
            "target": pair.target,
            "hits": {k: hits[k] for k in top_k},
            "mrr": mrrs[-1],
            "top_targets": targets[:5],
        })

    result = LBDEvalResult(
        method=method_name,
        num_pairs=n,
        hit_at_k={k: (hit_counts[k] / n if n else 0.0) for k in top_k},
        mrr=float(np.mean(mrrs)) if mrrs else 0.0,
        faithfulness_at_k={k: (faith_counts[k] / n if n else 0.0) for k in top_k},
        per_pair=per_pair,
    )
    return result
