#!/usr/bin/env python3
"""Script 07 — Prepare annotation samples for faithfulness human validation.

Samples 300 (hypothesis, KG-path) pairs across methods for 2 annotators,
outputs a CSV ready for blind annotation.  Used to validate the automatic
faithfulness metric (target Cohen's kappa > 0.7, Spearman > 0.8).
"""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from kg_scaffold.utils.config import load_config, get_path, ensure_dirs

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main():
    cfg = load_config()
    ensure_dirs(cfg)
    runs_dir = get_path("runs_dir")
    results_dir = get_path("results_dir")
    results_dir.mkdir(parents=True, exist_ok=True)

    n_per_method = cfg["evaluation"]["annotation_sample"] // 3
    run_files = sorted(runs_dir.glob("*.json"))
    if not run_files:
        log.error("No runs found.")
        return

    rng = random.Random(42)
    rows = []
    for rf in run_files:
        with open(rf) as fh:
            data = json.load(fh)
        method = data.get("method", rf.stem)
        all_hypos = []
        for pid, hlist in data.get("predictions", {}).items():
            for h in hlist:
                all_hypos.append({**h, "pair_id": pid, "method": method})
        if not all_hypos:
            continue
        sample = rng.sample(all_hypos, min(n_per_method, len(all_hypos)))
        rows.extend(sample)

    rng.shuffle(rows)
    df = pd.DataFrame(rows)
    # blind: hide method
    df["annotator_1_verdict"] = ""
    df["annotator_1_note"] = ""
    df["annotator_2_verdict"] = ""
    df["annotator_2_note"] = ""
    cols = ["pair_id", "method", "text", "kg_path", "faithfulness",
            "annotator_1_verdict", "annotator_1_note",
            "annotator_2_verdict", "annotator_2_note"]
    df = df[[c for c in cols if c in df.columns]]
    out = results_dir / "annotation_sample.csv"
    df.to_csv(out, index=False)
    log.info("Annotation sample (%d rows) -> %s", len(df), out)

    # instructions
    instr = results_dir / "annotation_instructions.md"
    instr.write_text(
        "# Faithfulness Annotation\n\n"
        "For each row, judge whether the **hypothesis** is supported by the "
        "**KG path**:\n\n"
        "- `entailed`: the KG path fully supports the claim.\n"
        "- `partial`: the path supports part of the claim.\n"
        "- `none`: the path does not support the claim.\n\n"
        "Write the verdict in `annotator_*_verdict` and a one-sentence note.\n"
        "Annotate independently; do not look at the other annotator's column.\n"
    )
    log.info("Instructions -> %s", instr)


if __name__ == "__main__":
    main()
