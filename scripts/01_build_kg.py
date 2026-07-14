#!/usr/bin/env python3
"""Script 01 — Build / load the KG.

Usage:
    python scripts/01_build_kg.py [--sample] [--csv PATH]

With --sample (default), uses the synthetic KG for fast dev.
With --csv PATH, loads a real SemMedDB CSV export.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kg_scaffold.utils.config import load_config, get_path, ensure_dirs
from kg_scaffold.utils.semmeddb import load_or_synthesize, build_graph
from kg_scaffold.utils.lbd_gold import load_lbd_gold

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true",
                    help="use small synthetic KG (147 triples, dev only)")
    ap.add_argument("--large", action="store_true",
                    help="use large synthetic KG (10K+ triples, for real experiments)")
    ap.add_argument("--csv", default=None, help="path to SemMedDB CSV")
    ap.add_argument("--sample-size", type=int, default=200000)
    ap.add_argument("--n-true", type=int, default=6000,
                    help="number of true typed triples for large KG")
    ap.add_argument("--noise-ratio", type=float, default=0.35,
                    help="fraction of noisy triples for large KG")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)

    semmed_dir = get_path("semmeddb_dir")
    csv_path = args.csv or (semmed_dir / "semmeddb_sample.csv")
    if args.sample or args.large:
        csv_path = None  # force synthetic

    if args.large:
        from kg_scaffold.utils.semmeddb import generate_large_synthetic_kg
        log.info("Generating large synthetic KG (n_true=%d, noise=%.0f%%)...",
                 args.n_true, args.noise_ratio * 100)
        triples = generate_large_synthetic_kg(
            seed=42, n_true=args.n_true, noise_ratio=args.noise_ratio)
    else:
        log.info("Loading KG (sample=%s, csv=%s)...", args.sample, csv_path)
        triples = load_or_synthesize(csv_path, sample=args.sample_size, seed=42,
                                     large=args.large)
    log.info("Loaded %d triples.", len(triples))

    # stats
    entities = set()
    relations = set()
    for t in triples:
        entities.add(t.subject)
        entities.add(t.object)
        relations.add(t.predicate)
    log.info("Entities: %d | Relations: %d", len(entities), len(relations))

    # save as JSON for downstream scripts
    out_path = semmed_dir / "kg_triples.json"
    data = [
        {"subject": t.subject, "predicate": t.predicate, "object": t.object,
         "score": t.score}
        for t in triples
    ]
    with open(out_path, "w") as fh:
        json.dump(data, fh)
    log.info("Saved KG to %s", out_path)

    # verify LBD gold pairs have seed entities in the KG
    pairs = load_lbd_gold()
    ent_set = {e.lower() for e in entities}
    coverage = sum(1 for p in pairs if p.source.lower() in ent_set)
    log.info("LBD seed coverage: %d/%d pairs", coverage, len(pairs))


if __name__ == "__main__":
    main()
