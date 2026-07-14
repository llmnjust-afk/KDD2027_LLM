"""LBD gold-standard loader and evaluation helpers.

The Swanson / Cameron benchmark defines (source, target) disease-chemical pairs
that should be re-discoverable via a bridging concept in the KG.  A method
"re-discovers" a pair if the target entity appears in the top-k generated
hypotheses for the source.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from kg_scaffold.utils.config import get_path


@dataclass(frozen=True)
class LBDPair:
    id: str
    source: str
    target: str
    bridge: tuple[str, ...]
    relation: str
    reference: str
    domain: str


def load_lbd_gold(filename: str = "swanson_cameron.json") -> list[LBDPair]:
    """Load LBD gold-standard pairs from the JSON in data/lbd_gold/."""
    p = get_path("lbd_gold_dir") / filename
    if not p.exists():
        raise FileNotFoundError(f"LBD gold file not found: {p}")
    with open(p) as fh:
        data = json.load(fh)
    pairs = []
    for rec in data["pairs"]:
        pairs.append(LBDPair(
            id=rec["id"],
            source=rec["source"],
            target=rec["target"],
            bridge=tuple(rec["bridge"]),
            relation=rec.get("relation", "TREATS"),
            reference=rec.get("reference", ""),
            domain=rec.get("domain", "biomedical"),
        ))
    return pairs


def query_for_pair(pair: LBDPair) -> str:
    """Natural-language research question for a given LBD pair."""
    rel = pair.relation.lower()
    return (f"What {rel} {pair.source}? "
            f"Propose candidate substances that may {rel} {pair.source} "
            f"and explain the mechanistic link.")


def hit_at_k(generated_targets: Sequence[str], gold_target: str,
             top_k: Sequence[int]) -> dict[int, bool]:
    """Check whether the gold target appears in the top-k generated targets."""
    generated_lower = [g.lower().strip() for g in generated_targets]
    gold_lower = gold_target.lower().strip()
    results = {}
    for k in top_k:
        top = generated_lower[:k]
        results[k] = any(gold_lower in g or g in gold_lower for g in top)
    return results


def mrr(generated_targets: Sequence[str], gold_target: str) -> float:
    """Reciprocal rank of the gold target in the generated list."""
    generated_lower = [g.lower().strip() for g in generated_targets]
    gold_lower = gold_target.lower().strip()
    for i, g in enumerate(generated_lower, 1):
        if gold_lower in g or g in gold_lower:
            return 1.0 / i
    return 0.0
