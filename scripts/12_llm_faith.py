#!/usr/bin/env python3
"""Run LLM-Verify+Faith experiment (no ComplEx)."""
import json, sys, time
sys.path.insert(0, '.')
import numpy as np
from kg_scaffold.utils.config import load_config, get_path, ensure_dirs
from kg_scaffold.utils.lbd_gold import load_lbd_gold, query_for_pair, hit_at_k, mrr
from kg_scaffold.utils.semmeddb import Triple, is_negative
from kg_scaffold.generation.llm_client import LLMClient
from kg_scaffold.retrieval.subgraph import SubgraphRetriever, Subgraph
from kg_scaffold.generation.hypothesis_gen import generate_hypotheses
from kg_scaffold.verification.faithfulness import verify_hypotheses
from kg_scaffold.kg.refinement import _verify_triple

cfg = load_config(); ensure_dirs(cfg)
semmed_dir = get_path('semmeddb_dir')
with open(semmed_dir / 'kg_triples.json') as f:
    data = json.load(f)
triples = [Triple(d['subject'], d['predicate'], d['object'], score=d.get('score',1.0)) for d in data]
corpus = [f'{t.subject} {t.predicate.lower().replace("_"," ")} {t.object}.' for t in triples]
client = LLMClient(cfg)
pairs = load_lbd_gold()
retriever = SubgraphRetriever(triples, max_hops=2, topk=40)

def run_llm_verify_faith(question, seed_entity, num=10):
    """LLM-verify subgraph triples + generate + faithfulness (no ComplEx)."""
    subgraph = retriever.retrieve(seed_entity)
    verified_triples = []
    for t in subgraph.triples:
        if is_negative(t.predicate):
            continue
        verdict, corrected = _verify_triple(t, client, None)
        if verdict == "supported":
            verified_triples.append(t)
        elif verdict == "contradicted" and corrected and corrected.lower() not in ("none", ""):
            verified_triples.append(Triple(t.subject, t.predicate, corrected, t.subject_cui, t.object_cui, score=t.score))
    verified_sub = Subgraph(root=seed_entity, triples=verified_triples[:40])
    snippets = [t for t in corpus if seed_entity.lower() in t.lower()][:5]
    hypos = generate_hypotheses(question=question, subgraph=verified_sub, snippets=snippets,
                                client=client, num=num, use_kg=True)
    report = verify_hypotheses(hypos, client, top_k=(1, 5, 10))
    return hypos, report

print("=== LLM-Verify+Faith on 10 LBD pairs ===", flush=True)
predictions = {}
all_hits = {1: [], 5: [], 10: []}
all_mrrs = []
all_faith = {1: [], 5: [], 10: []}

for pair in pairs:
    q = query_for_pair(pair)
    print(f"  {pair.source} -> {pair.target}", flush=True)
    try:
        hypos, report = run_llm_verify_faith(q, pair.source, num=10)
    except Exception as e:
        print(f"    ERROR: {e}", flush=True)
        hypos, report = [], None
    targets = [h.target_entity for h in hypos]
    h = hit_at_k(targets, pair.target, [1, 5, 10])
    for k in [1, 5, 10]:
        all_hits[k].append(1.0 if h[k] else 0.0)
    all_mrrs.append(mrr(targets, pair.target))
    if report:
        for k in [1, 5, 10]:
            all_faith[k].append(report.faithfulness_at_k.get(k, 0.0))
    else:
        for k in [1, 5, 10]:
            all_faith[k].append(0.0)
    predictions[pair.id] = [
        {"text": h.text, "target_entity": h.target_entity, "kg_path": h.kg_path,
         "novelty": h.novelty, "faithfulness": h.faithfulness, "score": h.score}
        for h in hypos
    ]
    print(f"    Hit@10={h[10]} MRR={all_mrrs[-1]:.2f} Faith@10={all_faith[10][-1]:.2f}", flush=True)

print(f"\n=== FINAL RESULTS ===", flush=True)
print(f"Hit@1={np.mean(all_hits[1]):.3f}  Hit@5={np.mean(all_hits[5]):.3f}  Hit@10={np.mean(all_hits[10]):.3f}", flush=True)
print(f"MRR={np.mean(all_mrrs):.3f}", flush=True)
print(f"Faith@1={np.mean(all_faith[1]):.3f}  Faith@5={np.mean(all_faith[5]):.3f}  Faith@10={np.mean(all_faith[10]):.3f}", flush=True)

runs_dir = get_path('runs_dir')
with open(runs_dir / "ours_llm_faith.json", "w") as f:
    json.dump({"method": "llm_faith", "predictions": predictions}, f, indent=2)
print("Saved!", flush=True)
