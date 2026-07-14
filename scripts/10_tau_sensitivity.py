#!/usr/bin/env python3
"""Script 10 — Tau sensitivity analysis.

Sweeps the ComplEx score threshold τ to show how it affects:
  - Number of triples sent to LLM verification
  - Final Hit@10 and Faith@10
  - KG size after co-refinement

Addresses reviewer concern: "threshold τ=0.15 is stated but never justified."
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from kg_scaffold.utils.config import load_config, get_path, ensure_dirs
from kg_scaffold.utils.lbd_gold import load_lbd_gold, query_for_pair, hit_at_k
from kg_scaffold.utils.semmeddb import Triple
from kg_scaffold.kg.completion import CompletionResult, load_scores
from kg_scaffold.kg.refinement import co_refine, filter_by_score
from kg_scaffold.generation.llm_client import LLMClient
from kg_scaffold.baselines.base import RunConfig
from kg_scaffold.baselines.ours import KGSCoRE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

TAU_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def load_kg(path: Path) -> list[Triple]:
    with open(path) as fh:
        data = json.load(fh)
    return [Triple(d["subject"], d["predicate"], d["object"],
                   score=d.get("score", 1.0)) for d in data]


def main():
    cfg = load_config()
    ensure_dirs(cfg)

    semmed_dir = get_path("semmeddb_dir")
    triples = load_kg(semmed_dir / "kg_triples.json")
    corpus = [f"{t.subject} {t.predicate.lower().replace('_',' ')} {t.object}."
              for t in triples]

    scores_path = semmed_dir / "completion_scores.tsv"
    if not scores_path.exists():
        log.error("No completion scores found. Run script 02 first.")
        return
    scores = load_scores(scores_path)
    completion = CompletionResult("complex", len(scores), 0, 0, {}, scores)

    client = LLMClient(cfg)
    pairs = load_lbd_gold()
    rc = RunConfig(num_hypotheses=10, use_kg=True, use_completion=True,
                   use_faithfulness=True)

    results = []
    for tau in TAU_VALUES:
        log.info("=== τ = %.2f ===", tau)
        # count how many triples would be verified
        n_below = sum(1 for t in triples if t.score < tau)
        log.info("  triples below τ: %d / %d (%.1f%%)", n_below, len(triples),
                 n_below / len(triples) * 100)

        method = KGSCoRE(
            triples=triples, completion=completion, client=client,
            min_confidence=tau, text_corpus=corpus,
        )

        hits = []
        faiths = []
        for pair in pairs:
            q = query_for_pair(pair)
            try:
                hypos = method.run(question=q, seed_entity=pair.source, cfg=rc)
            except Exception as e:
                log.warning("  failed: %s", e)
                hypos = []
            targets = [h.target_entity for h in hypos]
            h = hit_at_k(targets, pair.target, [10])
            hits.append(1.0 if h[10] else 0.0)
            top10 = hypos[:10]
            if top10:
                faith = sum(1 for x in top10
                            if x.faithfulness in ("entailed", "partial")) / len(top10)
            else:
                faith = 0.0
            faiths.append(faith)

        hit10 = float(np.mean(hits))
        faith10 = float(np.mean(faiths))
        results.append({
            "tau": tau,
            "n_below_threshold": n_below,
            "pct_below": n_below / len(triples) * 100,
            "hit_at_10": hit10,
            "faith_at_10": faith10,
        })
        log.info("  Hit@10=%.3f  Faith@10=%.3f", hit10, faith10)

    # save
    runs_dir = get_path("runs_dir")
    out_path = runs_dir / "tau_sensitivity.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    log.info("Saved τ sensitivity to %s", out_path)

    # summary table
    log.info("\n=== τ Sensitivity Summary ===")
    log.info("%-6s %8s %8s %8s %10s",
             "τ", "#below", "%below", "Hit@10", "Faith@10")
    log.info("-" * 45)
    for r in results:
        log.info("%.2f   %6d   %6.1f   %.3f     %.3f",
                 r["tau"], r["n_below_threshold"], r["pct_below"],
                 r["hit_at_10"], r["faith_at_10"])


if __name__ == "__main__":
    main()
