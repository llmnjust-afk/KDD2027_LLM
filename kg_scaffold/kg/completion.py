"""Module A.2 — Learned KG completion (ComplEx) via PyKEEN.

Trains a ComplEx embedding model on the (possibly noisy) KG and produces a
plausibility score for every triple.  Low-confidence triples are routed to the
LLM refinement queue (see ``refinement.py``).

This is the *learning* component that addresses the reviewer concern "no
training, no technical depth".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from kg_scaffold.utils.config import get_path, load_config
from kg_scaffold.utils.semmeddb import Triple, is_negative

logger = logging.getLogger(__name__)

_PYKEEN_AVAILABLE = False
try:
    from pykeen.pipeline import pipeline
    from pykeen.triples import TriplesFactory
    import torch
    _PYKEEN_AVAILABLE = True
except ImportError:  # pragma: no cover
    pass


@dataclass
class CompletionResult:
    """Output of training + scoring."""
    model_name: str
    num_triples: int
    num_entities: int
    num_relations: int
    metrics: dict           # MRR, hits@1/5/10 on held-out
    scores: dict[tuple[str, str, str], float]  # triple -> plausibility


def train_and_score(triples: Sequence[Triple],
                    cfg: dict | None = None) -> CompletionResult:
    """Train a ComplEx model and score all triples.

    Falls back to a lightweight heuristic scorer if PyKEEN is unavailable, so
    the pipeline still runs in minimal environments.
    """
    cfg = cfg or load_config()
    comp_cfg = cfg["completion"]

    clean = [t for t in triples if not is_negative(t.predicate)]
    if not clean:
        raise ValueError("No non-negative triples to train on.")

    if _PYKEEN_AVAILABLE:
        return _train_pykeen(clean, comp_cfg)
    return _heuristic_score(clean, comp_cfg)


def _triples_to_array(triples: Sequence[Triple]) -> np.ndarray:
    return np.array(
        [[t.subject, t.predicate, t.object] for t in triples],
        dtype=str,
    )


def _train_pykeen(triples: Sequence[Triple], comp_cfg: dict) -> CompletionResult:
    """Full ComplEx training with PyKEEN."""
    arr = _triples_to_array(triples)
    tf = TriplesFactory.from_labeled_triples(arr)

    # Train/valid/test split
    train, valid, test = tf.split([comp_cfg["train_frac"],
                                   comp_cfg["valid_frac"],
                                   1 - comp_cfg["train_frac"] - comp_cfg["valid_frac"]],
                                  random_state=42)

    ckpt_dir = get_path("checkpoint_dir")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Training ComplEx on %d triples (%d entities, %d relations)...",
                len(triples), tf.num_entities, tf.num_relations)

    result = pipeline(
        training=train,
        validation=valid,
        testing=test,
        model=comp_cfg["model"],          # "complex"
        model_kwargs={"embedding_dim": comp_cfg["embedding_dim"]},
        training_kwargs={
            "batch_size": comp_cfg["batch_size"],
        },
        epochs=comp_cfg["epochs"],
        optimizer_kwargs={"lr": comp_cfg["lr"]},
        negative_sampler_kwargs={"num_negs_per_pos": comp_cfg["negative_samples"]},
        device="cuda" if torch.cuda.is_available() else "cpu",
        random_seed=42,
        use_tqdm=False,
    )

    # Score all triples with the trained model
    model = result.model
    raw_scores: list[float] = []
    raw_keys: list[tuple[str, str, str]] = []
    model.eval()
    with torch.no_grad():
        batch_size = 4096
        for i in range(0, len(triples), batch_size):
            batch = arr[i:i + batch_size]
            tf_batch = TriplesFactory.from_labeled_triples(
                batch, entity_to_id=tf.entity_to_id, relation_to_id=tf.relation_to_id)
            mapped = torch.tensor(tf_batch.mapped_triples, device=model.device)
            scores = model.score_hrt(mapped).cpu().numpy().flatten()
            for row, sc in zip(batch, scores):
                raw_keys.append((row[0], row[1], row[2]))
                raw_scores.append(float(sc))

    # Normalize scores to [0, 1] via min-max so threshold τ is meaningful
    raw_arr = np.array(raw_scores)
    lo, hi = float(raw_arr.min()), float(raw_arr.max())
    score_map: dict[tuple[str, str, str], float] = {}
    if hi > lo:
        for key, sc in zip(raw_keys, raw_scores):
            score_map[key] = (sc - lo) / (hi - lo)
    else:
        for key in raw_keys:
            score_map[key] = 0.5

    metrics = {
        "mrr": float(result.metric_results.get_metric("both.realistic.inverse_harmonic_mean_rank")),
        "hits_at_1": float(result.metric_results.get_metric("both.realistic.hits_at_1")),
        "hits_at_5": float(result.metric_results.get_metric("both.realistic.hits_at_5")),
        "hits_at_10": float(result.metric_results.get_metric("both.realistic.hits_at_10")),
    }
    logger.info("ComplEx metrics: %s", metrics)

    return CompletionResult(
        model_name=comp_cfg["model"],
        num_triples=len(triples),
        num_entities=tf.num_entities,
        num_relations=tf.num_relations,
        metrics=metrics,
        scores=score_map,
    )


def _heuristic_score(triples: Sequence[Triple], comp_cfg: dict) -> CompletionResult:
    """Fallback scorer: uses degree-based + inverse-frequency heuristic.

    This runs when PyKEEN is not installed.  High-degree entities with rare
    predicates get higher scores.  Not as principled as ComplEx but keeps the
    pipeline functional.
    """
    logger.warning("PyKEEN not available — using heuristic KG-completion scorer.")
    from collections import Counter
    ent_deg = Counter()
    rel_freq = Counter()
    for t in triples:
        ent_deg[t.subject] += 1
        ent_deg[t.object] += 1
        rel_freq[t.predicate] += 1

    max_deg = max(ent_deg.values()) if ent_deg else 1
    total = sum(rel_freq.values()) or 1

    score_map: dict[tuple[str, str, str], float] = {}
    for t in triples:
        s_deg = ent_deg[t.subject] / max_deg
        o_deg = ent_deg[t.object] / max_deg
        rarity = -np.log(rel_freq[t.predicate] / total + 1e-8)
        rarity = rarity / (np.log(total) + 1e-8)
        score = 0.4 * s_deg + 0.4 * o_deg + 0.2 * rarity
        score_map[t.as_tuple()] = float(np.clip(score, 0.0, 1.0))

    return CompletionResult(
        model_name="heuristic",
        num_triples=len(triples),
        num_entities=len(ent_deg),
        num_relations=len(rel_freq),
        metrics={"mrr": 0.0, "hits_at_1": 0.0, "hits_at_5": 0.0, "hits_at_10": 0.0},
        scores=score_map,
    )


def attach_scores(triples: list[Triple], result: CompletionResult) -> list[Triple]:
    """Return a copy of triples with ComplEx scores attached."""
    out = []
    for t in triples:
        sc = result.scores.get(t.as_tuple(), t.score)
        out.append(Triple(t.subject, t.predicate, t.object,
                          t.subject_cui, t.object_cui, score=sc))
    return out


def save_result(result: CompletionResult, path: str | Path) -> None:
    """Persist scores to a TSV for reuse without retraining."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write(f"# model={result.model_name} triples={result.num_triples} "
                 f"ents={result.num_entities} rels={result.num_relations}\n")
        fh.write(f"# metrics={result.metrics}\n")
        fh.write("subject\tpredicate\tobject\tscore\n")
        for (s, p, o), sc in result.scores.items():
            fh.write(f"{s}\t{p}\t{o}\t{sc}\n")
    logger.info("saved completion scores to %s", path)


def load_scores(path: str | Path) -> dict[tuple[str, str, str], float]:
    """Reload saved scores."""
    path = Path(path)
    scores: dict[tuple[str, str, str], float] = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) != 4:
                continue
            try:
                scores[(parts[0], parts[1], parts[2])] = float(parts[3])
            except ValueError:
                continue  # header or malformed line
    return scores
