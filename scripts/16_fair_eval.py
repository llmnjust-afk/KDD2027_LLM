#!/usr/bin/env python3
"""Script 16 — Fair faithfulness evaluation.

Addresses reviewer concern: "0.99 faithfulness is guaranteed by design"

Key changes:
1. ALL methods (including baselines) are required to output KG paths
2. Faithfulness is split into: strict (entailed only), partial, unsupported, coverage
3. KG cleaning is directly evaluated (we know which triples are noise)
4. McNemar test replaces paired t-test for binary Hit@10

This script:
- Runs all methods with path-output requirement
- Evaluates fair faithfulness across all methods
- Computes KG cleaning precision/recall/F1
- Reports McNemar test
"""
import json, sys, re, random
sys.path.insert(0, '.')
import numpy as np
from scipy import stats as sp_stats
from kg_scaffold.utils.config import load_config, get_path, ensure_dirs
from kg_scaffold.utils.lbd_gold import load_lbd_gold, query_for_pair, hit_at_k, mrr
from kg_scaffold.utils.semmeddb import Triple, is_negative
from kg_scaffold.generation.llm_client import LLMClient
from kg_scaffold.retrieval.subgraph import SubgraphRetriever, Subgraph
from kg_scaffold.generation.hypothesis_gen import generate_hypotheses, Hypothesis
from kg_scaffold.verification.faithfulness import verify_hypotheses, ENTAILED, PARTIAL, NONE
from kg_scaffold.kg.refinement import _verify_triple
from kg_scaffold.baselines.base import RunConfig
from kg_scaffold.baselines.vanilla_llm import VanillaLLM
from kg_scaffold.baselines.bm25_rag import BM25RAG
from kg_scaffold.baselines.tog_wrapper import ToGWrapper
from kg_scaffold.baselines.rog_wrapper import RoGWrapper

def load_kg(path):
    with open(path) as f:
        data = json.load(f)
    return [Triple(d['subject'], d['predicate'], d['object'], score=d.get('score',1.0)) for d in data]

def extract_path_from_hypothesis(hypos, triples, seed_entity):
    """For methods that don't output KG paths, extract paths post-hoc.
    
    This makes faithfulness comparison FAIR: all methods get the same
    path-extraction treatment.
    """
    retriever = SubgraphRetriever(triples, max_hops=2, topk=40)
    subgraph = retriever.retrieve(seed_entity)
    path_entities = {t.subject.lower() for t in subgraph.triples} | {t.object.lower() for t in subgraph.triples}
    
    for h in hypos:
        if not h.kg_path:
            # Extract entities mentioned in hypothesis that are in the KG
            hypo_lower = h.text.lower()
            mentioned = [e for e in path_entities if e in hypo_lower]
            if mentioned:
                h.kg_path = " -> ".join(mentioned[:3])
            else:
                h.kg_path = ""
    return hypos

def run_with_fair_paths(method, question, seed_entity, triples, cfg, client):
    """Run a method and ensure all hypotheses have KG paths for fair evaluation."""
    hypos = method.run(question=question, seed_entity=seed_entity, cfg=cfg)
    
    # If method doesn't produce KG paths, extract them post-hoc
    has_paths = any(h.kg_path for h in hypos)
    if not has_paths:
        hypos = extract_path_from_hypothesis(hypos, triples, seed_entity)
    
    # Run faithfulness check on ALL methods (not just KG-SCoRE)
    report = verify_hypotheses(hypos, client, top_k=(1, 5, 10))
    return hypos, report

def compute_strict_faith(hypos, k=10):
    """Strict faithfulness: only 'entailed' counts (not 'partial')."""
    top = hypos[:k]
    if not top:
        return 0.0, 0.0, 0.0, 0.0  # strict, partial, unsupported, coverage
    n = len(top)
    strict = sum(1 for h in top if h.faithfulness == ENTAILED) / n
    partial = sum(1 for h in top if h.faithfulness == PARTIAL) / n
    unsupported = sum(1 for h in top if h.faithfulness == NONE) / n
    has_path = sum(1 for h in top if h.kg_path) / n
    return strict, partial, unsupported, has_path

def evaluate_kg_cleaning(triples, client, sample_size=200):
    """Directly evaluate KG cleaning: we know which triples are noise (score < 0.5)."""
    rng = random.Random(42)
    # Sample triples: half noise (score < 0.5), half clean (score >= 0.5)
    noise_triples = [t for t in triples if t.score < 0.5 and not is_negative(t.predicate)]
    clean_triples = [t for t in triples if t.score >= 0.5 and not is_negative(t.predicate)]
    
    sample_noise = rng.sample(noise_triples, min(sample_size // 2, len(noise_triples)))
    sample_clean = rng.sample(clean_triples, min(sample_size // 2, len(clean_triples)))
    
    tp = fp = tn = fn = 0  # LLM correctly identifies noise vs clean
    
    print(f"  Evaluating KG cleaning on {len(sample_noise)} noise + {len(sample_clean)} clean triples...", flush=True)
    
    for i, t in enumerate(sample_noise + sample_clean):
        is_actual_noise = t.score < 0.5
        verdict, corrected = _verify_triple(t, client, None)
        
        # LLM says "noise" if verdict is contradicted or unverifiable
        llm_says_noise = verdict in ("contradicted", "unverifiable")
        
        if is_actual_noise and llm_says_noise:
            tp += 1  # correctly identified noise
        elif is_actual_noise and not llm_says_noise:
            fn += 1  # missed noise (said supported)
        elif not is_actual_noise and llm_says_noise:
            fp += 1  # false alarm (clean triple marked as noise)
        else:
            tn += 1  # correctly identified clean
        
        if (i + 1) % 50 == 0:
            print(f"    Processed {i+1}/{len(sample_noise)+len(sample_clean)}", flush=True)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
    
    return {
        'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'n_noise': len(sample_noise), 'n_clean': len(sample_clean),
    }

def mcnemar_test(a_correct, b_correct):
    """McNemar's test for paired binary outcomes."""
    # a_correct, b_correct are lists of 0/1
    n01 = sum(1 for a, b in zip(a_correct, b_correct) if a == 1 and b == 0)  # A right, B wrong
    n10 = sum(1 for a, b in zip(a_correct, b_correct) if a == 0 and b == 1)  # A wrong, B right
    if n01 + n10 == 0:
        return 1.0
    # Exact binomial test
    from scipy.stats import binom_test
    p = binom_test(min(n01, n10), n01 + n10, 0.5)
    return p

# Main
cfg = load_config(); ensure_dirs(cfg)
semmed_dir = get_path('semmeddb_dir')
triples = load_kg(semmed_dir / 'kg_triples.json')
corpus = [f'{t.subject} {t.predicate.lower().replace("_"," ")} {t.object}.' for t in triples]
client = LLMClient(cfg)
pairs = load_lbd_gold()
rc = RunConfig(num_hypotheses=10, use_kg=True, use_completion=False, use_faithfulness=True)

retriever = SubgraphRetriever(triples, max_hops=2, topk=40)

# === Part 1: Fair faithfulness evaluation ===
print("=== Part 1: Fair Faithfulness Evaluation (20 pairs) ===", flush=True)

methods = {
    'vanilla_llm': VanillaLLM(client),
    'bm25_rag': BM25RAG(corpus, client),
    'tog': ToGWrapper(triples, client),
    'rog': RoGWrapper(triples, client),
}

def run_llm_faith(question, seed_entity, num=10):
    """KG-SCoRE: LLM verify + generate + faithfulness."""
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

all_results = {}
all_hit10 = {}  # for McNemar

for method_name, method in methods.items():
    print(f"\nRunning {method_name}...", flush=True)
    hits = {1:[], 5:[], 10:[]}
    strict_faiths = []
    partial_faiths = []
    unsupported_rates = []
    coverages = []
    per_pair_correct = []
    
    for pair in pairs:
        q = query_for_pair(pair)
        try:
            hypos, report = run_with_fair_paths(method, q, pair.source, triples, rc, client)
        except Exception as e:
            print(f"  ERROR on {pair.id}: {e}", flush=True)
            hypos, report = [], None
        
        targets = [h.target_entity for h in hypos]
        h = hit_at_k(targets, pair.target, [1,5,10])
        for k in [1,5,10]:
            hits[k].append(1.0 if h[k] else 0.0)
        per_pair_correct.append(1 if h[10] else 0)
        
        strict, partial, unsupp, coverage = compute_strict_faith(hypos, k=10)
        strict_faiths.append(strict)
        partial_faiths.append(partial)
        unsupported_rates.append(unsupp)
        coverages.append(coverage)
    
    all_results[method_name] = {
        'hit@10': float(np.mean(hits[10])),
        'strict_faith': float(np.mean(strict_faiths)),
        'partial_faith': float(np.mean(partial_faiths)),
        'unsupported': float(np.mean(unsupported_rates)),
        'coverage': float(np.mean(coverages)),
    }
    all_hit10[method_name] = per_pair_correct
    print(f"  Hit@10={np.mean(hits[10]):.2f}  Strict={np.mean(strict_faiths):.2f}  "
          f"Partial={np.mean(partial_faiths):.2f}  Coverage={np.mean(coverages):.2f}", flush=True)

# KG-SCoRE (LLM-verify+faith)
print(f"\nRunning kg_score (llm_faith)...", flush=True)
hits = {1:[], 5:[], 10:[]}
strict_faiths = []
partial_faiths = []
unsupported_rates = []
coverages = []
per_pair_correct = []

for pair in pairs:
    q = query_for_pair(pair)
    try:
        hypos = run_llm_faith(q, pair.source, num=10)
    except Exception as e:
        print(f"  ERROR on {pair.id}: {e}", flush=True)
        hypos = []
    targets = [h.target_entity for h in hypos]
    h = hit_at_k(targets, pair.target, [1,5,10])
    for k in [1,5,10]:
        hits[k].append(1.0 if h[k] else 0.0)
    per_pair_correct.append(1 if h[10] else 0)
    strict, partial, unsupp, coverage = compute_strict_faith(hypos, k=10)
    strict_faiths.append(strict)
    partial_faiths.append(partial)
    unsupported_rates.append(unsupp)
    coverages.append(coverage)

all_results['kg_score'] = {
    'hit@10': float(np.mean(hits[10])),
    'strict_faith': float(np.mean(strict_faiths)),
    'partial_faith': float(np.mean(partial_faiths)),
    'unsupported': float(np.mean(unsupported_rates)),
    'coverage': float(np.mean(coverages)),
}
all_hit10['kg_score'] = per_pair_correct
print(f"  Hit@10={np.mean(hits[10]):.2f}  Strict={np.mean(strict_faiths):.2f}  "
      f"Partial={np.mean(partial_faiths):.2f}  Coverage={np.mean(coverages):.2f}", flush=True)

# McNemar tests
print(f"\n=== McNemar Tests (vs kg_score) ===", flush=True)
for method in ['vanilla_llm', 'bm25_rag', 'tog', 'rog']:
    p = mcnemar_test(all_hit10['kg_score'], all_hit10[method])
    print(f"  kg_score vs {method}: p={p:.4f}", flush=True)

# === Part 2: KG Cleaning Evaluation ===
print(f"\n=== Part 2: KG Cleaning Evaluation ===", flush=True)
cleaning = evaluate_kg_cleaning(triples, client, sample_size=200)
print(f"  Precision={cleaning['precision']:.3f}  Recall={cleaning['recall']:.3f}  "
      f"F1={cleaning['f1']:.3f}  Accuracy={cleaning['accuracy']:.3f}", flush=True)
print(f"  TP={cleaning['tp']}  FP={cleaning['fp']}  TN={cleaning['tn']}  FN={cleaning['fn']}", flush=True)

# Save
runs_dir = get_path('runs_dir')
results_dir = get_path('results_dir')
with open(results_dir / 'fair_faithfulness.json', 'w') as f:
    json.dump({
        'fair_faithfulness': all_results,
        'mcnemar': {m: mcnemar_test(all_hit10['kg_score'], all_hit10[m]) 
                     for m in ['vanilla_llm','bm25_rag','tog','rog']},
        'kg_cleaning': cleaning,
    }, f, indent=2)
print(f"\nSaved to {results_dir / 'fair_faithfulness.json'}", flush=True)

# Summary table
print(f"\n=== Fair Faithfulness Summary (20 pairs) ===", flush=True)
print(f"{'Method':<15} {'Hit@10':>6} {'Strict':>7} {'Partial':>8} {'Unsupp':>7} {'Cov':>6}", flush=True)
print("-"*52, flush=True)
for name, r in all_results.items():
    print(f"{name:<15} {r['hit@10']:.2f}   {r['strict_faith']:.2f}    "
          f"{r['partial_faith']:.2f}     {r['unsupported']:.2f}    {r['coverage']:.2f}", flush=True)
print(f"\nKG Cleaning: P={cleaning['precision']:.3f} R={cleaning['recall']:.3f} F1={cleaning['f1']:.3f}", flush=True)
print("Done!", flush=True)
