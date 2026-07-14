#!/usr/bin/env python3
"""Script 04 — Run our method (KG-SCoRE) on the LBD gold standard.

Usage:
    python scripts/04_run_ours.py [--no-completion] [--no-faithfulness]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kg_scaffold.utils.config import load_config, get_path, ensure_dirs
from kg_scaffold.utils.lbd_gold import load_lbd_gold, query_for_pair
from kg_scaffold.utils.semmeddb import Triple
from kg_scaffold.kg.completion import CompletionResult, load_scores
from kg_scaffold.generation.llm_client import LLMClient
from kg_scaffold.baselines.base import RunConfig
from kg_scaffold.baselines.ours import KGSCoRE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_kg(path: Path) -> list[Triple]:
    with open(path) as fh:
        data = json.load(fh)
    return [Triple(d["subject"], d["predicate"], d["object"],
                   score=d.get("score", 1.0)) for d in data]


def build_text_corpus(triples: list[Triple]) -> list[str]:
    return [f"{t.subject} {t.predicate.lower().replace('_',' ')} {t.object}."
            for t in triples]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=10)
    ap.add_argument("--no-completion", action="store_true",
                    help="ablation: disable ComplEx co-refinement")
    ap.add_argument("--no-faithfulness", action="store_true",
                    help="ablation: disable Module D")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)

    semmed_dir = get_path("semmeddb_dir")
    kg_path = semmed_dir / "kg_triples.json"
    triples = load_kg(kg_path)
    corpus = build_text_corpus(triples)
    client = LLMClient(cfg)
    pairs = load_lbd_gold()

    # load ComplEx scores if available
    completion = None
    scores_path = semmed_dir / "completion_scores.tsv"
    if not args.no_completion and scores_path.exists():
        scores = load_scores(scores_path)
        completion = CompletionResult(
            model_name="complex", num_triples=len(scores),
            num_entities=0, num_relations=0,
            metrics={}, scores=scores)
        log.info("Loaded ComplEx scores (%d triples)", len(scores))
    elif not args.no_completion:
        log.warning("No completion scores found at %s — running without.", scores_path)

    method = KGSCoRE(
        triples=triples,
        completion=completion,
        client=client,
        max_hops=cfg["kg"]["max_hops"],
        topk_triples=cfg["kg"]["topk_triples"],
        min_confidence=cfg["kg"]["min_confidence"],
        text_corpus=corpus,
    )

    rc = RunConfig(
        num_hypotheses=args.num,
        use_kg=True,
        use_completion=completion is not None,
        use_faithfulness=not args.no_faithfulness,
        method_name="kg_score",
    )

    log.info("Running KG-SCoRE on %d pairs...", len(pairs))
    predictions = {}
    for pair in pairs:
        q = query_for_pair(pair)
        log.info("[kg_score] %s -> %s", pair.source, pair.target)
        hypos = method.run(question=q, seed_entity=pair.source, cfg=rc)
        predictions[pair.id] = [
            {"text": h.text, "target_entity": h.target_entity,
             "kg_path": h.kg_path, "novelty": h.novelty,
             "faithfulness": h.faithfulness, "score": h.score}
            for h in hypos
        ]

    runs_dir = get_path("runs_dir")
    runs_dir.mkdir(parents=True, exist_ok=True)
    suffix = ""
    if args.no_completion:
        suffix += "_nocomp"
    if args.no_faithfulness:
        suffix += "_nofaith"
    out_path = runs_dir / f"ours{suffix}.json"
    with open(out_path, "w") as fh:
        json.dump({"method": "kg_score", "predictions": predictions}, fh, indent=2)
    log.info("Saved predictions to %s", out_path)


if __name__ == "__main__":
    main()
