#!/usr/bin/env python3
"""Script 18 — True 20-pair multi-seed experiment.

3 seeds × 20 pairs = n=60. Replaces the 10-pair subset.
"""
import json, sys, random, numpy as np
sys.path.insert(0, '.')
from scipy import stats as sp_stats
from kg_scaffold.utils.config import load_config, get_path, ensure_dirs
from kg_scaffold.utils.lbd_gold import load_lbd_gold, query_for_pair, hit_at_k
from kg_scaffold.utils.semmeddb import Triple, is_negative
from kg_scaffold.generation.llm_client import LLMClient
from kg_scaffold.retrieval.subgraph import SubgraphRetriever, Subgraph
from kg_scaffold.generation.hypothesis_gen import generate_hypotheses
from kg_scaffold.verification.faithfulness import verify_hypotheses, ENTAILED, PARTIAL, NONE
from kg_scaffold.kg.refinement import _verify_triple
from kg_scaffold.baselines.base import RunConfig
from kg_scaffold.baselines.vanilla_llm import VanillaLLM
from kg_scaffold.baselines.bm25_rag import BM25RAG
from kg_scaffold.baselines.tog_wrapper import ToGWrapper
from kg_scaffold.baselines.rog_wrapper import RoGWrapper

cfg = load_config(); ensure_dirs(cfg)
semmed_dir = get_path('semmeddb_dir')
results_dir = get_path('results_dir')
results_dir.mkdir(parents=True, exist_ok=True)

with open(semmed_dir / 'kg_triples.json') as f:
    data = json.load(f)
triples = [Triple(d['subject'], d['predicate'], d['object'], score=d.get('score',1.0)) for d in data]
corpus = [f'{t.subject} {t.predicate.lower().replace("_"," ")} {t.object}.' for t in triples]
pairs = load_lbd_gold()
retriever = SubgraphRetriever(triples, max_hops=2, topk=40)

def run_llm_faith(client, question, seed_entity, num=10):
    subgraph = retriever.retrieve(seed_entity)
    verified = []
    for t in subgraph.triples:
        if is_negative(t.predicate): continue
        verdict, corrected = _verify_triple(t, client, None)
        if verdict == "supported": verified.append(t)
        elif verdict == "contradicted" and corrected and corrected.lower() not in ("none",""):
            verified.append(Triple(t.subject, t.predicate, corrected, t.subject_cui, t.object_cui, score=t.score))
    verified_sub = Subgraph(root=seed_entity, triples=verified[:40])
    snippets = [t for t in corpus if seed_entity.lower() in t.lower()][:5]
    hypos = generate_hypotheses(question=question, subgraph=verified_sub, snippets=snippets, client=client, num=num, use_kg=True)
    verify_hypotheses(hypos, client, top_k=(1,5,10))
    return hypos

SEEDS = [42, 123, 777]
rc = RunConfig(num_hypotheses=10, use_kg=True, use_completion=False, use_faithfulness=True)
all_seed_results = {}

for seed in SEEDS:
    print(f"\n--- Seed {seed} ---", flush=True)
    s_client = LLMClient(cfg)
    methods = {
        'vanilla_llm': VanillaLLM(s_client),
        'bm25_rag': BM25RAG(corpus, s_client),
        'tog': ToGWrapper(triples, s_client),
        'rog': RoGWrapper(triples, s_client),
    }
    seed_res = {}
    for name, method in methods.items():
        hits = []
        for pair in pairs:  # ALL 20 pairs
            q = query_for_pair(pair)
            try:
                hypos = method.run(question=q, seed_entity=pair.source, cfg=rc)
            except Exception as e:
                print(f"  {name} {pair.source}->{pair.target}: ERROR {e}", flush=True)
                hypos = []
            targets = [h.target_entity for h in hypos]
            h = hit_at_k(targets, pair.target, [10])
            hits.append(1.0 if h[10] else 0.0)
        seed_res[name] = hits
        print(f"  {name}: {np.mean(hits):.2f} ({hits})", flush=True)

    # LLM-faith
    hits = []
    for pair in pairs:  # ALL 20 pairs
        q = query_for_pair(pair)
        try:
            hypos = run_llm_faith(s_client, q, pair.source, 10)
        except Exception as e:
            print(f"  llm_faith {pair.source}->{pair.target}: ERROR {e}", flush=True)
            hypos = []
        targets = [h.target_entity for h in hypos]
        h = hit_at_k(targets, pair.target, [10])
        hits.append(1.0 if h[10] else 0.0)
    seed_res['llm_faith'] = hits
    print(f"  llm_faith: {np.mean(hits):.2f} ({hits})", flush=True)
    all_seed_results[seed] = seed_res

# Summary
summary = {}
for m in all_seed_results[SEEDS[0]]:
    per = [float(np.mean(all_seed_results[s][m])) for s in SEEDS]
    all_hits = []
    for s in SEEDS:
        all_hits.extend(all_seed_results[s][m])
    summary[m] = {
        'mean': float(np.mean(per)),
        'std': float(np.std(per)),
        'per_seed': per,
        'n': len(all_hits),
        'all_hits': all_hits,
    }

print(f"\n{'='*60}", flush=True)
print("20-pair Multi-seed Summary (n=60, 3 seeds x 20 pairs):", flush=True)
print(f"{'='*60}", flush=True)
for m, s in summary.items():
    print(f"  {m}: {s['mean']:.2f}±{s['std']:.2f} per_seed={s['per_seed']}", flush=True)

# McNemar tests
print(f"\nMcNemar tests (vs llm_faith):", flush=True)
kg_hits = summary['llm_faith']['all_hits']
for m in ['vanilla_llm', 'bm25_rag', 'tog', 'rog']:
    other = summary[m]['all_hits']
    # McNemar: b = kg yes/other no, c = kg no/other yes
    b = sum(1 for k, o in zip(kg_hits, other) if k == 1 and o == 0)
    c = sum(1 for k, o in zip(kg_hits, other) if k == 0 and o == 1)
    if b + c == 0:
        p = 1.0
    elif b + c < 25:
        # Exact binomial
        from scipy.stats import binom_test
        p = binom_test(min(b, c), b + c, 0.5) if hasattr(sp_stats, 'binom_test') else float(sp_stats.binomtest(min(b, c), b + c, 0.5).pvalue)
    else:
        chi2 = (abs(b - c) - 1)**2 / (b + c)
        p = 1 - sp_stats.chi2.cdf(chi2, 1)
    print(f"  vs {m}: b={b} c={c} p={p:.4f}", flush=True)

# Save
output = {
    'summary': summary,
    'raw': {str(s): v for s, v in all_seed_results.items()},
    'n_pairs': 20,
    'n_seeds': 3,
    'n_total': 60,
}
with open(results_dir / 'multiseed_20pair.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nSaved to {results_dir / 'multiseed_20pair.json'}", flush=True)
print("DONE!", flush=True)
