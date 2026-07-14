"""End-to-end smoke test for the KG-SCoRE pipeline.

Runs without API keys (mock LLM) and without real SemMedDB (synthetic KG).
Verifies that every module connects and produces expected output shapes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kg_scaffold.utils.semmeddb import generate_synthetic_kg
from kg_scaffold.utils.lbd_gold import load_lbd_gold, query_for_pair
from kg_scaffold.generation.llm_client import LLMClient
from kg_scaffold.kg.completion import train_and_score, attach_scores
from kg_scaffold.kg.refinement import co_refine
from kg_scaffold.retrieval.subgraph import SubgraphRetriever
from kg_scaffold.generation.hypothesis_gen import generate_hypotheses
from kg_scaffold.verification.faithfulness import verify_hypotheses
from kg_scaffold.baselines.ours import KGSCoRE
from kg_scaffold.baselines.base import RunConfig


def test_synthetic_kg():
    triples = generate_synthetic_kg(seed=42)
    assert len(triples) > 20, f"expected >20 triples, got {len(triples)}"
    print(f"  [OK] synthetic KG: {len(triples)} triples")


def test_completion():
    triples = generate_synthetic_kg(seed=42)
    result = train_and_score(triples)
    assert len(result.scores) > 0, "no scores produced"
    print(f"  [OK] completion ({result.model_name}): {len(result.scores)} scores")


def test_refinement():
    triples = generate_synthetic_kg(seed=42)
    result = train_and_score(triples)
    refined, stats = co_refine(triples, result, LLMClient(), min_confidence=0.15)
    assert stats.total == len(triples)
    assert stats.kept + stats.deleted + stats.corrected == stats.total - sum(
        1 for t in triples if t.predicate.startswith("NEG_"))
    print(f"  [OK] refinement: {stats}")


def test_retrieval():
    triples = generate_synthetic_kg(seed=42)
    result = train_and_score(triples)
    scored = attach_scores(triples, result)
    retriever = SubgraphRetriever(scored, max_hops=2, topk=20)
    sub = retriever.retrieve("Raynaud disease")
    assert len(sub.triples) > 0, "no triples retrieved"
    print(f"  [OK] retrieval: {len(sub.triples)} triples for 'Raynaud disease'")


def test_generation():
    triples = generate_synthetic_kg(seed=42)
    result = train_and_score(triples)
    scored = attach_scores(triples, result)
    retriever = SubgraphRetriever(scored, max_hops=2, topk=20)
    sub = retriever.retrieve("Raynaud disease")
    client = LLMClient()
    hypos = generate_hypotheses(
        question="What treats Raynaud disease?",
        subgraph=sub, snippets=[], client=client, num=5, use_kg=True)
    assert len(hypos) > 0, "no hypotheses generated"
    assert hypos[0].text, "empty hypothesis text"
    print(f"  [OK] generation: {len(hypos)} hypotheses")
    for h in hypos[:2]:
        print(f"       - {h.text[:60]}...")


def test_faithfulness():
    triples = generate_synthetic_kg(seed=42)
    result = train_and_score(triples)
    scored = attach_scores(triples, result)
    retriever = SubgraphRetriever(scored, max_hops=2, topk=20)
    sub = retriever.retrieve("Raynaud disease")
    client = LLMClient()
    hypos = generate_hypotheses(
        question="What treats Raynaud disease?",
        subgraph=sub, client=client, num=5, use_kg=True)
    report = verify_hypotheses(hypos, client, top_k=(1, 5))
    assert len(report.per_hypothesis) == len(hypos)
    print(f"  [OK] faithfulness: {report.faithfulness_at_k}")


def test_e2e_pipeline():
    triples = generate_synthetic_kg(seed=42)
    result = train_and_score(triples)
    client = LLMClient()
    method = KGSCoRE(triples=triples, completion=result, client=client)
    rc = RunConfig(num_hypotheses=5, use_kg=True, use_completion=True,
                   use_faithfulness=True)
    pairs = load_lbd_gold()
    pair = pairs[0]
    hypos = method.run(question=query_for_pair(pair),
                       seed_entity=pair.source, cfg=rc)
    assert len(hypos) > 0
    print(f"  [OK] e2e pipeline: {len(hypos)} hypotheses for {pair.source}")
    print(f"       target was: {pair.target}")
    targets = [h.target_entity for h in hypos]
    print(f"       generated targets: {targets[:3]}")


def main():
    tests = [
        test_synthetic_kg, test_completion, test_refinement,
        test_retrieval, test_generation, test_faithfulness, test_e2e_pipeline,
    ]
    print("Running KG-SCoRE smoke tests...\n")
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            failed += 1
    print(f"\n{'='*40}")
    if failed:
        print(f"FAILED: {failed}/{len(tests)}")
        sys.exit(1)
    print(f"PASSED: {len(tests)}/{len(tests)}")


if __name__ == "__main__":
    main()
