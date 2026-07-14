#!/usr/bin/env python3
"""Script 13 — Multi-seed for LLM-Verify+Faith (no ComplEx)."""
import json, sys
sys.path.insert(0, '.')
import numpy as np
from scipy import stats as sp_stats
from kg_scaffold.utils.config import load_config, get_path, ensure_dirs
from kg_scaffold.utils.lbd_gold import load_lbd_gold, query_for_pair, hit_at_k
from kg_scaffold.utils.semmeddb import Triple, is_negative
from kg_scaffold.generation.llm_client import LLMClient
from kg_scaffold.retrieval.subgraph import SubgraphRetriever, Subgraph
from kg_scaffold.generation.hypothesis_gen import generate_hypotheses
from kg_scaffold.verification.faithfulness import verify_hypotheses
from kg_scaffold.kg.refinement import _verify_triple
from kg_scaffold.baselines.base import RunConfig
from kg_scaffold.baselines.vanilla_llm import VanillaLLM
from kg_scaffold.baselines.bm25_rag import BM25RAG
from kg_scaffold.baselines.tog_wrapper import ToGWrapper
from kg_scaffold.baselines.rog_wrapper import RoGWrapper

SEEDS = [42, 123, 777]

def load_kg(path):
    with open(path) as f:
        data = json.load(f)
    return [Triple(d['subject'], d['predicate'], d['object'], score=d.get('score',1.0)) for d in data]

def run_llm_faith(triples, corpus, client, question, seed_entity, num=10):
    retriever = SubgraphRetriever(triples, max_hops=2, topk=40)
    subgraph = retriever.retrieve(seed_entity)
    verified = []
    for t in subgraph.triples:
        if is_negative(t.predicate):
            continue
        verdict, corrected = _verify_triple(t, client, None)
        if verdict == "supported":
            verified.append(t)
        elif verdict == "contradicted" and corrected and corrected.lower() not in ("none",""):
            verified.append(Triple(t.subject, t.predicate, corrected, t.subject_cui, t.object_cui, score=t.score))
    verified_sub = Subgraph(root=seed_entity, triples=verified[:40])
    snippets = [t for t in corpus if seed_entity.lower() in t.lower()][:5]
    hypos = generate_hypotheses(question=question, subgraph=verified_sub, snippets=snippets,
                                client=client, num=num, use_kg=True)
    verify_hypotheses(hypos, client, top_k=(1,5,10))
    return hypos

cfg = load_config(); ensure_dirs(cfg)
semmed_dir = get_path('semmeddb_dir')
triples = load_kg(semmed_dir / 'kg_triples.json')
corpus = [f'{t.subject} {t.predicate.lower().replace("_"," ")} {t.object}.' for t in triples]
pairs = load_lbd_gold()
rc = RunConfig(num_hypotheses=10, use_kg=True, use_completion=False, use_faithfulness=True)

all_seed_results = {}
for seed in SEEDS:
    cfg['llm']['temperature'] = 0.2 + (seed % 3) * 0.1
    client = LLMClient(cfg)
    print(f"=== Seed {seed} (temp={cfg['llm']['temperature']}) ===", flush=True)
    
    methods = {
        'vanilla_llm': VanillaLLM(client),
        'bm25_rag': BM25RAG(corpus, client),
        'tog': ToGWrapper(triples, client),
        'rog': RoGWrapper(triples, client),
    }
    
    seed_results = {}
    for name, method in methods.items():
        hits = []
        for pair in pairs:
            q = query_for_pair(pair)
            try:
                hypos = method.run(question=q, seed_entity=pair.source, cfg=rc)
            except:
                hypos = []
            targets = [h.target_entity for h in hypos]
            h = hit_at_k(targets, pair.target, [10])
            hits.append(1.0 if h[10] else 0.0)
        seed_results[name] = hits
        print(f"  {name}: hit@10={np.mean(hits):.2f}", flush=True)
    
    # LLM-faith
    hits = []
    for pair in pairs:
        q = query_for_pair(pair)
        try:
            hypos = run_llm_faith(triples, corpus, client, q, pair.source, num=10)
        except:
            hypos = []
        targets = [h.target_entity for h in hypos]
        h = hit_at_k(targets, pair.target, [10])
        hits.append(1.0 if h[10] else 0.0)
    seed_results['llm_faith'] = hits
    print(f"  llm_faith: hit@10={np.mean(hits):.2f}", flush=True)
    
    all_seed_results[seed] = seed_results

# Summary
summary = {}
for method in all_seed_results[SEEDS[0]]:
    per_seed = [float(np.mean(all_seed_results[s][method])) for s in SEEDS]
    summary[method] = {'seeds': SEEDS, 'per_seed_hit10': per_seed,
                       'mean': float(np.mean(per_seed)), 'std': float(np.std(per_seed))}

runs_dir = get_path('runs_dir')
with open(runs_dir / 'multi_seed_v2.json', 'w') as f:
    json.dump({'summary': summary, 'raw': {str(s): v for s, v in all_seed_results.items()}}, f, indent=2)

print(f"\n=== Multi-Seed Summary (Hit@10, mean ± std) ===", flush=True)
print(f"{'Method':<15} {'Mean':>6} {'Std':>6} {'Per-seed':<25}", flush=True)
for method, s in summary.items():
    per = ', '.join(f'{x:.2f}' for x in s['per_seed_hit10'])
    print(f"{method:<15} {s['mean']:.2f}   {s['std']:.2f}   [{per}]", flush=True)

# Significance
ours_flat = []
for seed in SEEDS:
    ours_flat.extend(all_seed_results[seed]['llm_faith'])
for method in ['vanilla_llm', 'bm25_rag', 'tog', 'rog']:
    theirs = []
    for seed in SEEDS:
        theirs.extend(all_seed_results[seed][method])
    n = min(len(ours_flat), len(theirs))
    t, p = sp_stats.ttest_rel(ours_flat[:n], theirs[:n])
    d = abs((np.mean(ours_flat[:n]) - np.mean(theirs[:n])) / (np.std(ours_flat[:n]) + 1e-8))
    print(f"  llm_faith vs {method:12s}: t={t:.2f} p={p:.4f} d={d:.2f} (n={n})", flush=True)
print("Saved!", flush=True)
