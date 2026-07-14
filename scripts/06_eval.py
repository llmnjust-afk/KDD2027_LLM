#!/usr/bin/env python3
"""Script 06 — Evaluate all runs and produce result tables + significance tests.

Reads all JSON files from experiments/runs/ and produces:
  - main_table.csv     (hit@k, MRR, faithfulness@k per method)
  - ablation_table.csv
  - significance.json  (paired t-test, Wilcoxon, Cohen's d, bootstrap CI)
  - figures/           (bar charts)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from kg_scaffold.utils.config import load_config, get_path, ensure_dirs
from kg_scaffold.utils.lbd_gold import load_lbd_gold
from kg_scaffold.generation.hypothesis_gen import Hypothesis
from kg_scaffold.evaluation.lbd_metrics import evaluate_lbd
from kg_scaffold.evaluation.stats import compare_methods, bootstrap_ci

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_run(path: Path) -> tuple[str, dict[str, list[Hypothesis]]]:
    with open(path) as fh:
        data = json.load(fh)
    method = data.get("method", path.stem)
    preds = {}
    for pid, hlist in data.get("predictions", {}).items():
        preds[pid] = [Hypothesis(
            text=h["text"], kg_path=h.get("kg_path", ""),
            novelty=h.get("novelty", ""), score=h.get("score", 0.0),
            faithfulness=h.get("faithfulness", ""),
            source_method=method) for h in hlist]
    return method, preds


def main():
    cfg = load_config()
    ensure_dirs(cfg)
    runs_dir = get_path("runs_dir")
    results_dir = get_path("results_dir")
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = get_path("figures_dir")
    figures_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_lbd_gold()
    run_files = sorted(runs_dir.glob("*.json"))
    if not run_files:
        log.error("No runs found in %s", runs_dir)
        return

    results = {}
    per_pair_scores = {}
    for rf in run_files:
        method, preds = load_run(rf)
        res = evaluate_lbd(pairs, preds, top_k=(1, 5, 10), method_name=method)
        results[method] = res
        # per-pair binary hit@10 for significance
        per_pair_scores[method] = [
            1.0 if pp["hits"].get(10) else 0.0 for pp in res.per_pair
        ]
        log.info("%s: hit@10=%.3f MRR=%.3f faith@10=%.3f",
                 method, res.hit_at_k.get(10, 0), res.mrr,
                 res.faithfulness_at_k.get(10, 0))

    # --- main table (with ±std from multi-seed if available) ---
    # Load multi-seed results for std calculation
    multi_seed_path = runs_dir / "multi_seed_results.json"
    seed_stds = {}
    if multi_seed_path.exists():
        with open(multi_seed_path) as fh:
            ms_data = json.load(fh)
        for method, s in ms_data.get("summary", {}).items():
            seed_stds[method] = s.get("std", 0.0)

    rows = []
    for m, r in results.items():
        std = seed_stds.get(m, 0.0)
        rows.append({
            "method": m,
            "hit@1": r.hit_at_k.get(1, 0),
            "hit@5": r.hit_at_k.get(5, 0),
            "hit@10": r.hit_at_k.get(10, 0),
            "hit@10_std": std,
            "MRR": r.mrr,
            "faith@1": r.faithfulness_at_k.get(1, 0),
            "faith@5": r.faithfulness_at_k.get(5, 0),
            "faith@10": r.faithfulness_at_k.get(10, 0),
            "num_pairs": r.num_pairs,
        })
    df = pd.DataFrame(rows).sort_values("hit@10", ascending=False)
    main_path = results_dir / "main_table.csv"
    df.to_csv(main_path, index=False)

    # Print with ±std
    log.info("Main table -> %s", main_path)
    for _, row in df.iterrows():
        std_str = f"±{row['hit@10_std']:.2f}" if row['hit@10_std'] > 0 else ""
        log.info("  %-20s Hit@1=%.2f Hit@5=%.2f Hit@10=%.2f%s MRR=%.2f Faith@10=%.2f",
                 row['method'], row['hit@1'], row['hit@5'],
                 row['hit@10'], std_str, row['MRR'], row['faith@10'])

    # --- significance: ours vs each baseline ---
    sig = {}
    ours_key = "kg_score"
    if ours_key in per_pair_scores:
        for m, scores in per_pair_scores.items():
            if m == ours_key:
                continue
            n = min(len(scores), len(per_pair_scores[ours_key]))
            sig[m] = compare_methods(per_pair_scores[ours_key][:n],
                                     scores[:n],
                                     name_a=ours_key, name_b=m)
    sig_path = results_dir / "significance.json"
    with open(sig_path, "w") as fh:
        json.dump(sig, fh, indent=2, default=str)
    log.info("Significance -> %s", sig_path)

    # --- figures ---
    try:
        _plot_bars(df, figures_dir)
    except Exception as exc:
        log.warning("plotting failed: %s", exc)


def _plot_bars(df: pd.DataFrame, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["hit@1", "hit@5", "hit@10", "MRR"]
    x = np.arange(len(df))
    w = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, metric in enumerate(metrics):
        ax.bar(x + i * w, df[metric], w, label=metric)
    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(df["method"], rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("LBD Re-discovery Performance")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_dir / "main_results.png", dpi=150)
    plt.close(fig)

    # faithfulness
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    faith_metrics = ["faith@1", "faith@5", "faith@10"]
    for i, metric in enumerate(faith_metrics):
        ax2.bar(x + i * w, df[metric], w, label=metric)
    ax2.set_xticks(x + w)
    ax2.set_xticklabels(df["method"], rotation=30, ha="right")
    ax2.set_ylabel("Faithfulness")
    ax2.set_title("Faithfulness@k")
    ax2.legend()
    ax2.set_ylim(0, 1.05)
    fig2.tight_layout()
    fig2.savefig(out_dir / "faithfulness.png", dpi=150)
    plt.close(fig2)


if __name__ == "__main__":
    main()
