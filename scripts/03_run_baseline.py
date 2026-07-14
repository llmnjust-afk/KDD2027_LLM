#!/usr/bin/env python3
"""Script 03 — Run baselines on the LBD gold standard.

Usage:
    python scripts/03_run_baseline.py --method {vanilla_llm,bm25_rag,dense_rag,tog} [--seed]
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
from kg_scaffold.generation.llm_client import LLMClient
from kg_scaffold.baselines.base import RunConfig
from kg_scaffold.baselines.vanilla_llm import VanillaLLM
from kg_scaffold.baselines.bm25_rag import BM25RAG
from kg_scaffold.baselines.dense_rag import DenseRAG
from kg_scaffold.baselines.tog_wrapper import ToGWrapper
from kg_scaffold.baselines.rog_wrapper import RoGWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_kg(path: Path) -> list[Triple]:
    with open(path) as fh:
        data = json.load(fh)
    return [Triple(d["subject"], d["predicate"], d["object"],
                   score=d.get("score", 1.0)) for d in data]


def build_text_corpus(triples: list[Triple]) -> list[str]:
    """Convert triples to pseudo-sentences for text-based RAG baselines."""
    return [f"{t.subject} {t.predicate.lower().replace('_',' ')} {t.object}."
            for t in triples]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["vanilla_llm", "bm25_rag", "dense_rag", "tog", "rog"])
    ap.add_argument("--num", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)

    semmed_dir = get_path("semmeddb_dir")
    kg_path = semmed_dir / "kg_triples.json"
    triples = load_kg(kg_path)
    corpus = build_text_corpus(triples)
    client = LLMClient(cfg)
    pairs = load_lbd_gold()

    rc = RunConfig(num_hypotheses=args.num)

    if args.method == "vanilla_llm":
        method = VanillaLLM(client)
    elif args.method == "bm25_rag":
        method = BM25RAG(corpus, client)
    elif args.method == "dense_rag":
        method = DenseRAG(corpus, client)
    elif args.method == "tog":
        method = ToGWrapper(triples, client)
    elif args.method == "rog":
        method = RoGWrapper(triples, client)
    else:
        raise ValueError(args.method)

    log.info("Running %s on %d LBD pairs...", method.name, len(pairs))
    predictions = {}
    for pair in pairs:
        q = query_for_pair(pair)
        log.info("[%s] %s -> %s", method.name, pair.source, pair.target)
        hypos = method.run(question=q, seed_entity=pair.source, cfg=rc)
        predictions[pair.id] = [
            {"text": h.text, "target_entity": h.target_entity,
             "kg_path": h.kg_path, "novelty": h.novelty,
             "faithfulness": h.faithfulness, "score": h.score}
            for h in hypos
        ]

    runs_dir = get_path("runs_dir")
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_path = runs_dir / f"baseline_{args.method}.json"
    with open(out_path, "w") as fh:
        json.dump({"method": args.method, "predictions": predictions}, fh, indent=2)
    log.info("Saved predictions to %s", out_path)


if __name__ == "__main__":
    main()
