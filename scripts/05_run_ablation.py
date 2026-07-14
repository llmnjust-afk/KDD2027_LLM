#!/usr/bin/env python3
"""Script 05 — Run the ablation matrix.

Produces predictions for each ablation configuration so Script 06 can build
the ablation table.  Configurations:

  raw_kg       : no refinement, no completion, no faithfulness
  +llm_clean   : LLM refinement only (no learned scoring)
  +learned     : ComplEx scoring only (no LLM refinement)
  full         : co-refinement (learned + LLM) + faithfulness
  no_kg        : KG scaffold off (degrades to vanilla + snippets)
  no_faith     : full but Module D off
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kg_scaffold.utils.config import load_config, get_path, ensure_dirs
from kg_scaffold.utils.lbd_gold import load_lbd_gold, query_for_pair
from kg_scaffold.utils.semmeddb import Triple
from kg_scaffold.kg.completion import CompletionResult, load_scores
from kg_scaffold.kg.refinement import filter_by_score
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
    cfg = load_config()
    ensure_dirs(cfg)

    semmed_dir = get_path("semmeddb_dir")
    kg_path = semmed_dir / "kg_triples.json"
    triples = load_kg(kg_path)
    corpus = build_text_corpus(triples)
    client = LLMClient(cfg)
    pairs = load_lbd_gold()

    # load ComplEx scores
    scores_path = semmed_dir / "completion_scores.tsv"
    completion = None
    if scores_path.exists():
        scores = load_scores(scores_path)
        completion = CompletionResult(
            model_name="complex", num_triples=len(scores),
            num_entities=0, num_relations=0, metrics={}, scores=scores)

    configs = {
        "raw_kg":      {"use_kg": True, "use_completion": False, "use_faith": False,
                         "filter_score": False},
        "plus_llm":    {"use_kg": True, "use_completion": False, "use_faith": False,
                         "filter_score": False, "llm_refine": True},
        "plus_learned":{"use_kg": True, "use_completion": True,  "use_faith": False,
                         "filter_score": True, "llm_refine": False},
        "full":        {"use_kg": True, "use_completion": True,  "use_faith": True,
                         "filter_score": False, "llm_refine": True},
        "no_kg":       {"use_kg": False,"use_completion": False, "use_faith": False,
                         "filter_score": False},
        "no_faith":    {"use_kg": True, "use_completion": True,  "use_faith": False,
                         "filter_score": False, "llm_refine": True},
    }

    runs_dir = get_path("runs_dir")
    runs_dir.mkdir(parents=True, exist_ok=True)

    for cname, cconf in configs.items():
        log.info("=== Ablation: %s ===", cname)
        use_triples = triples
        if cconf.get("filter_score") and completion:
            use_triples = filter_by_score(triples, cfg["kg"]["min_confidence"])
            log.info("  score-filtered to %d triples", len(use_triples))

        method = KGSCoRE(
            triples=use_triples,
            completion=completion if cconf["use_completion"] else None,
            client=client,
            max_hops=cfg["kg"]["max_hops"],
            topk_triples=cfg["kg"]["topk_triples"],
            min_confidence=cfg["kg"]["min_confidence"],
            text_corpus=corpus,
        )

        rc = RunConfig(
            num_hypotheses=10,
            use_kg=cconf["use_kg"],
            use_completion=cconf["use_completion"],
            use_faithfulness=cconf["use_faith"],
            method_name=cname,
        )

        predictions = {}
        for pair in pairs:
            q = query_for_pair(pair)
            hypos = method.run(question=q, seed_entity=pair.source, cfg=rc)
            predictions[pair.id] = [
                {"text": h.text, "target_entity": h.target_entity,
                 "kg_path": h.kg_path, "novelty": h.novelty,
                 "faithfulness": h.faithfulness, "score": h.score}
                for h in hypos
            ]

        out_path = runs_dir / f"ablation_{cname}.json"
        with open(out_path, "w") as fh:
            json.dump({"method": cname, "predictions": predictions}, fh, indent=2)
        log.info("  saved -> %s", out_path)


if __name__ == "__main__":
    main()
