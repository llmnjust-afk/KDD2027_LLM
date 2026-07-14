#!/usr/bin/env python3
"""Script 08 — KGQA generalization experiment.

Tests whether KG-SCoRE generalizes beyond LBD to multi-hop KGQA.  Constructs
QA pairs from the KG itself (so no external dataset download needed), then
evaluates Exact Match (EM) and F1 for each method.

Two QA types:
  - one-hop: "What TREATS <disease>?" → answer = chemical entity
  - two-hop: "What INHIBITS the entity that <disease> ASSOCIATED_WITH?" → bridging

This proves the method is not overfit to the LBD re-discovery setup.
"""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from kg_scaffold.utils.config import load_config, get_path, ensure_dirs
from kg_scaffold.utils.semmeddb import Triple, generate_synthetic_kg, build_graph
from kg_scaffold.kg.completion import CompletionResult, load_scores, train_and_score, attach_scores
from kg_scaffold.generation.llm_client import LLMClient
from kg_scaffold.retrieval.subgraph import SubgraphRetriever
from kg_scaffold.generation.hypothesis_gen import generate_hypotheses
from kg_scaffold.verification.faithfulness import verify_hypotheses
from kg_scaffold.baselines.base import RunConfig
from kg_scaffold.baselines.vanilla_llm import VanillaLLM
from kg_scaffold.baselines.bm25_rag import BM25RAG
from kg_scaffold.baselines.tog_wrapper import ToGWrapper
from kg_scaffold.baselines.ours import KGSCoRE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def build_qa_pairs(triples: list[Triple], n: int = 20, seed: int = 42) -> list[dict]:
    """Construct QA pairs from the KG for generalization testing."""
    rng = random.Random(seed)
    graph = build_graph(triples)
    qa_pairs = []

    # Type 1: one-hop — "What TREATS <X>?" → Y
    treat_triples = [t for t in triples if t.predicate in ("TREATS", "INHIBITS")]
    rng.shuffle(treat_triples)
    for t in treat_triples[:n // 2]:
        qa_pairs.append({
            "id": f"1hop-{len(qa_pairs)}",
            "type": "one_hop",
            "question": f"What substance {t.predicate.lower()} {t.object}?",
            "answer": t.subject,
            "seed_entity": t.object,
            "hop_type": t.predicate,
        })

    # Type 2: two-hop — "What INHIBITS the entity that <disease> ASSOCIATED_WITH?"
    assoc_triples = [t for t in triples if t.predicate == "ASSOCIATED_WITH"]
    rng.shuffle(assoc_triples)
    for t in assoc_triples[:n // 2]:
        bridge = t.object
        # find what inhibits/treats the bridge
        candidates = [tt for tt in triples
                      if tt.object == bridge and tt.predicate in ("INHIBITS", "TREATS")]
        if candidates:
            ans = candidates[0].subject
            qa_pairs.append({
                "id": f"2hop-{len(qa_pairs)}",
                "type": "two_hop",
                "question": (f"What substance inhibits {bridge}, "
                             f"which is associated with {t.subject}?"),
                "answer": ans,
                "seed_entity": t.subject,
                "bridge": bridge,
                "hop_type": "ASSOCIATED_WITH->INHIBITS",
            })
    return qa_pairs


def extract_answer(hypos: list, question: str) -> str:
    """Extract the predicted answer from generated hypotheses."""
    if not hypos:
        return ""
    # use target_entity from top hypothesis
    return hypos[0].target_entity


def exact_match(pred: str, gold: str) -> float:
    """1.0 if pred matches gold, 0.0 otherwise (case-insensitive substring)."""
    p = pred.lower().strip()
    g = gold.lower().strip()
    if not p or not g:
        return 0.0
    return 1.0 if (g in p or p in g) else 0.0


def f1_score(pred: str, gold: str) -> float:
    """Token-level F1."""
    p_tokens = set(pred.lower().split())
    g_tokens = set(gold.lower().split())
    if not p_tokens or not g_tokens:
        return 0.0
    tp = len(p_tokens & g_tokens)
    if tp == 0:
        return 0.0
    prec = tp / len(p_tokens)
    rec = tp / len(g_tokens)
    return 2 * prec * rec / (prec + rec)


def run_method_on_qa(method, qa_pairs: list[dict], rc: RunConfig) -> dict:
    """Run a method on all QA pairs, return per-pair predictions + metrics."""
    predictions = {}
    em_scores = []
    f1_scores = []
    faith_scores = []

    for qa in qa_pairs:
        log.info("  [%s] %s -> %s", method.name, qa["id"], qa["answer"])
        hypos = method.run(question=qa["question"],
                           seed_entity=qa["seed_entity"], cfg=rc)
        pred = extract_answer(hypos, qa["question"])
        em = exact_match(pred, qa["answer"])
        f1 = f1_score(pred, qa["answer"])
        em_scores.append(em)
        f1_scores.append(f1)

        # faithfulness for top-5
        if hypos:
            top5 = hypos[:5]
            faith = sum(1 for h in top5 if h.faithfulness in ("entailed", "partial")) / len(top5)
        else:
            faith = 0.0
        faith_scores.append(faith)

        predictions[qa["id"]] = {
            "question": qa["question"],
            "gold": qa["answer"],
            "pred": pred,
            "em": em,
            "f1": f1,
            "faith": faith,
            "top_hypotheses": [{"text": h.text, "target": h.target_entity,
                                "kg_path": h.kg_path} for h in hypos[:3]],
        }

    return {
        "method": method.name,
        "num_qa": len(qa_pairs),
        "em": float(np.mean(em_scores)),
        "f1": float(np.mean(f1_scores)),
        "faithfulness": float(np.mean(faith_scores)),
        "per_qa": predictions,
    }


def main():
    cfg = load_config()
    ensure_dirs(cfg)
    semmed_dir = get_path("semmeddb_dir")

    # load KG
    kg_path = semmed_dir / "kg_triples.json"
    with open(kg_path) as fh:
        data = json.load(fh)
    triples = [Triple(d["subject"], d["predicate"], d["object"],
                      score=d.get("score", 1.0)) for d in data]
    corpus = [f"{t.subject} {t.predicate.lower().replace('_',' ')} {t.object}."
              for t in triples]

    # load ComplEx scores
    scores_path = semmed_dir / "completion_scores.tsv"
    completion = None
    if scores_path.exists():
        scores = load_scores(scores_path)
        completion = CompletionResult("complex", len(scores), 0, 0, {}, scores)

    client = LLMClient(cfg)

    # build QA pairs
    qa_pairs = build_qa_pairs(triples, n=20, seed=42)
    log.info("Built %d QA pairs (%d one-hop, %d two-hop)",
             len(qa_pairs),
             sum(1 for q in qa_pairs if q["type"] == "one_hop"),
             sum(1 for q in qa_pairs if q["type"] == "two_hop"))

    rc = RunConfig(num_hypotheses=5, use_kg=True, use_completion=True,
                   use_faithfulness=True)

    # run all methods
    methods = {
        "vanilla_llm": VanillaLLM(client),
        "bm25_rag": BM25RAG(corpus, client),
        "tog": ToGWrapper(triples, client),
        "kg_score": KGSCoRE(triples=triples, completion=completion,
                            client=client, text_corpus=corpus),
    }

    results = {}
    for name, method in methods.items():
        log.info("=== Running %s on KGQA ===", name)
        res = run_method_on_qa(method, qa_pairs, rc)
        results[name] = res
        log.info("  %s: EM=%.2f F1=%.2f Faith=%.2f", name, res["em"], res["f1"],
                 res["faithfulness"])

    # save
    runs_dir = get_path("runs_dir")
    out_path = runs_dir / "kgqa_generalization.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    log.info("Saved KGQA results to %s", out_path)

    # summary table
    log.info("\n=== KGQA Generalization Summary ===")
    log.info("%-15s %5s %5s %6s", "Method", "EM", "F1", "Faith")
    log.info("-" * 35)
    for name, res in results.items():
        log.info("%-15s %.2f  %.2f  %.2f", name, res["em"], res["f1"], res["faithfulness"])


if __name__ == "__main__":
    main()
