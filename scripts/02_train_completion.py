#!/usr/bin/env python3
"""Script 02 — Train ComplEx KG-completion model and score all triples.

Usage:
    python scripts/02_train_completion.py [--kg PATH] [--out PATH]

Outputs a TSV of triple -> plausibility score used by Module A.3.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kg_scaffold.utils.config import load_config, get_path, ensure_dirs
from kg_scaffold.utils.semmeddb import Triple
from kg_scaffold.kg.completion import train_and_score, save_result

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", default=None, help="path to kg_triples.json")
    ap.add_argument("--out", default=None, help="output scores TSV path")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)

    semmed_dir = get_path("semmeddb_dir")
    kg_path = Path(args.kg) if args.kg else semmed_dir / "kg_triples.json"
    out_path = Path(args.out) if args.out else semmed_dir / "completion_scores.tsv"

    with open(kg_path) as fh:
        data = json.load(fh)
    triples = [Triple(d["subject"], d["predicate"], d["object"],
                      score=d.get("score", 1.0)) for d in data]
    log.info("Loaded %d triples from %s", len(triples), kg_path)

    log.info("Training ComplEx (model=%s, dim=%d, epochs=%d)...",
             cfg["completion"]["model"],
             cfg["completion"]["embedding_dim"],
             cfg["completion"]["epochs"])
    result = train_and_score(triples, cfg)
    log.info("Done. Model=%s metrics=%s", result.model_name, result.metrics)

    save_result(result, out_path)
    log.info("Scores saved to %s", out_path)


if __name__ == "__main__":
    main()
