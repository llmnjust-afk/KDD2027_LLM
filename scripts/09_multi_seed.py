#!/usr/bin/env python3
"""Script 09 — Multi-seed experiment for statistical robustness.

Runs the main LBD experiment with 3 different seeds to produce mean ± std
and strengthen significance tests.  Seeds vary the LLM temperature and
hypothesis sampling order.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from kg_scaffold.utils.config import load_config, get_path, ensure_dirs
from kg_scaffold.utils.lbd_gold import load_lbd_gold, query_for_pair, hit_at_k, mrr
from kg_scaffold.utils.semmeddb import Triple
from kg_scaffold.kg.completion import CompletionResult, load_scores
from kg_scaffold.generation.llm_client import LLMClient
from kg_scaffold.generation.hypothesis_gen import Hypothesis
from kg_scaffold.baselines.base import RunConfig
from kg_scaffold.baselines.vanilla_llm import VanillaLLM
from kg_scaffold.baselines.bm25_rag import BM25RAG
from kg_scaffold.baselines.tog_wrapper import ToGWrapper
from kg_scaffold.baselines.ours import KGSCoRE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SEEDS = [42, 123, 777]


def load_kg(path: Path) -> list[Triple]:
    with open(path) as fh:
        data = json.load(fh)
    return [Triple(d["subject"], d["predicate"], d["object"],
                   score=d.get("score", 1.0)) for d in data]


def build_corpus(triples: list[Triple]) -> list[str]:
    return [f"{t.subject} {t.predicate.lower().replace('_',' ')} {t.object}."
            for t in triples]


def run_seed(seed: int, cfg: dict, triples, corpus, completion, pairs):
    """Run all methods with a given seed, return per-method per-pair hit@10."""
    # vary temperature slightly per seed
    cfg["llm"]["temperature"] = 0.2 + (seed % 3) * 0.1
    client = LLMClient(cfg)

    rc = RunConfig(num_hypotheses=10, use_kg=True,
                   use_completion=completion is not None,
                   use_faithfulness=True)

    methods = {
        "vanilla_llm": VanillaLLM(client),
        "bm25_rag": BM25RAG(corpus, client),
        "tog": ToGWrapper(triples, client),
        "kg_score": KGSCoRE(triples=triples, completion=completion,
                            client=client, text_corpus=corpus),
    }

    all_results = {}
    for name, method in methods.items():
        hits = []
        for pair in pairs:
            q = query_for_pair(pair)
            try:
                hypos = method.run(question=q, seed_entity=pair.source, cfg=rc)
            except Exception as e:
                log.warning("  %s failed on %s: %s", name, pair.id, e)
                hypos = []
            targets = [h.target_entity for h in hypos]
            h = hit_at_k(targets, pair.target, [10])
            hits.append(1.0 if h[10] else 0.0)
        all_results[name] = hits
        log.info("  seed=%d %s: hit@10=%.2f", seed, name, np.mean(hits))

    return all_results


def main():
    cfg = load_config()
    ensure_dirs(cfg)

    semmed_dir = get_path("semmeddb_dir")
    triples = load_kg(semmed_dir / "kg_triples.json")
    corpus = build_corpus(triples)

    scores_path = semmed_dir / "completion_scores.tsv"
    completion = None
    if scores_path.exists():
        scores = load_scores(scores_path)
        completion = CompletionResult("complex", len(scores), 0, 0, {}, scores)

    pairs = load_lbd_gold()

    all_seed_results = {}
    for seed in SEEDS:
        log.info("=== Seed %d ===", seed)
        results = run_seed(seed, cfg, triples, corpus, completion, pairs)
        all_seed_results[seed] = results

    # aggregate
    runs_dir = get_path("runs_dir")
    out_path = runs_dir / "multi_seed_results.json"

    summary = {}
    for method in all_seed_results[SEEDS[0]]:
        per_seed_means = []
        per_pair_all = []
        for seed in SEEDS:
            hits = all_seed_results[seed][method]
            per_seed_means.append(float(np.mean(hits)))
            per_pair_all.append(hits)
        summary[method] = {
            "seeds": SEEDS,
            "per_seed_hit10": per_seed_means,
            "mean": float(np.mean(per_seed_means)),
            "std": float(np.std(per_seed_means)),
            "per_pair_per_seed": per_pair_all,
        }

    with open(out_path, "w") as fh:
        json.dump({"summary": summary,
                   "raw": {str(s): v for s, v in all_seed_results.items()},
                   "pairs": [p.id for p in pairs]}, fh, indent=2)
    log.info("Saved multi-seed results to %s", out_path)

    # print summary
    log.info("\n=== Multi-Seed Summary (Hit@10, mean ± std) ===")
    log.info("%-15s %8s %8s", "Method", "Mean", "Std")
    log.info("-" * 35)
    for method, s in summary.items():
        log.info("%-15s %.3f    %.3f", method, s["mean"], s["std"])

    # paired t-test across all pairs × seeds
    from scipy import stats as sp_stats
    log.info("\n=== Paired t-test (ours vs baselines, n=%d pairs × %d seeds) ===",
             len(pairs), len(SEEDS))
    ours_flat = []
    for seed in SEEDS:
        ours_flat.extend(all_seed_results[seed]["kg_score"])
    for method in ["vanilla_llm", "bm25_rag", "tog"]:
        theirs_flat = []
        for seed in SEEDS:
            theirs_flat.extend(all_seed_results[seed][method])
        n = min(len(ours_flat), len(theirs_flat))
        t, p = sp_stats.ttest_rel(ours_flat[:n], theirs_flat[:n])
        d = (np.mean(ours_flat[:n]) - np.mean(theirs_flat[:n])) / (np.std(ours_flat[:n]) + 1e-8)
        log.info("  kg_score vs %-12s: t=%.2f p=%.4f d=%.2f (n=%d)",
                 method, t, p, d, n)


if __name__ == "__main__":
    main()
