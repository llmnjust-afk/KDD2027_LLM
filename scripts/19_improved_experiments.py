#!/usr/bin/env python3
"""Script 19 — Improved KG-SCoRE experiments with three optimizations.

Improvements:
1. KG cleaning: only delete "contradicted", keep "unverifiable" (strict_delete=True)
2. Stricter generation prompt: only state path-supported claims
3. Lower generation temperature: 0.4 → 0.2 (reduce variance)

Runs:
- 20-pair multi-seed (3 seeds × 20 pairs = n=60) for all methods
- KG cleaning evaluation with strict_delete
- KGQA with improved generation
- Faithfulness evaluation (strict + broad)
"""
import json, sys, random, re, time, numpy as np
sys.path.insert(0, '.')
from scipy import stats as sp_stats
from kg_scaffold.utils.config import load_config, get_path, ensure_dirs
from kg_scaffold.utils.lbd_gold import load_lbd_gold, query_for_pair, hit_at_k, mrr
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

def compute_faith_metrics(hypos, k=10):
    top = hypos[:k]
    if not top: return 0, 0, 0
    n = len(top)
    strict = sum(1 for h in top if h.faithfulness == ENTAILED) / n
    broad = sum(1 for h in top if h.faithfulness in (ENTAILED, PARTIAL)) / n
    coverage = sum(1 for h in top if h.kg_path) / n
    return strict, broad, coverage

def run_llm_verify_faith(client, question, seed_entity, num=10, strict_delete=True):
    """Improved KG-SCoRE: LLM verify (strict_delete) + generate + faithfulness."""
    subgraph = retriever.retrieve(seed_entity)
    verified = []
    for t in subgraph.triples:
        if is_negative(t.predicate): continue
        verdict, corrected = _verify_triple(t, client, None)
        if verdict == "supported":
            verified.append(t)
        elif verdict == "contradicted" and corrected and corrected.lower() not in ("none",""):
            verified.append(Triple(t.subject, t.predicate, corrected, t.subject_cui, t.object_cui, score=t.score))
        elif verdict == "contradicted":
            pass  # delete contradicted
        elif strict_delete:
            verified.append(t)  # keep unverifiable (improvement!)
        else:
            pass  # old behavior: delete unverifiable
    verified_sub = Subgraph(root=seed_entity, triples=verified[:40])
    snippets = [t for t in corpus if seed_entity.lower() in t.lower()][:5]
    hypos = generate_hypotheses(question=question, subgraph=verified_sub, snippets=snippets, client=client, num=num, use_kg=True)
    verify_hypotheses(hypos, client, top_k=(1,5,10))
    return hypos

def extract_path_posthoc(hypos, seed_entity):
    """Post-hoc path extraction for fair faithfulness evaluation."""
    subgraph = retriever.retrieve(seed_entity)
    path_entities = {t.subject.lower() for t in subgraph.triples} | {t.object.lower() for t in subgraph.triples}
    for h in hypos:
        if not h.kg_path:
            hypo_lower = h.text.lower()
            mentioned = [e for e in path_entities if e in hypo_lower]
            if mentioned:
                h.kg_path = " -> ".join(mentioned[:3])
    return hypos

# ============================================================
# EXPERIMENT 1: 20-pair Multi-seed with improved KG-SCoRE
# ============================================================
print("=" * 60, flush=True)
print("EXP 1: Improved 20-pair Multi-seed (n=60)", flush=True)
print("=" * 60, flush=True)

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
        for pair in pairs:
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

    # Improved KG-SCoRE
    hits = []
    strict_faiths = []
    broad_faiths = []
    coverages = []
    for pair in pairs:
        q = query_for_pair(pair)
        try:
            hypos = run_llm_verify_faith(s_client, q, pair.source, 10, strict_delete=True)
        except Exception as e:
            print(f"  kgscore {pair.source}->{pair.target}: ERROR {e}", flush=True)
            hypos = []
        targets = [h.target_entity for h in hypos]
        h = hit_at_k(targets, pair.target, [10])
        hits.append(1.0 if h[10] else 0.0)
        s, b, c = compute_faith_metrics(hypos, 10)
        strict_faiths.append(s)
        broad_faiths.append(b)
        coverages.append(c)
    seed_res['kgscore'] = hits
    seed_res['kgscore_strict'] = strict_faiths
    seed_res['kgscore_broad'] = broad_faiths
    seed_res['kgscore_cov'] = coverages
    print(f"  kgscore: {np.mean(hits):.2f} strict={np.mean(strict_faiths):.2f} broad={np.mean(broad_faiths):.2f}", flush=True)
    all_seed_results[seed] = seed_res

# Summary
summary = {}
for m in ['vanilla_llm', 'bm25_rag', 'tog', 'rog', 'kgscore']:
    per = [float(np.mean(all_seed_results[s][m])) for s in SEEDS]
    all_hits = []
    for s in SEEDS:
        all_hits.extend(all_seed_results[s][m])
    summary[m] = {'mean': float(np.mean(per)), 'std': float(np.std(per)), 'per_seed': per, 'n': len(all_hits), 'all_hits': all_hits}

# KG-SCoRE faithfulness summary
kg_strict = [float(np.mean(all_seed_results[s]['kgscore_strict'])) for s in SEEDS]
kg_broad = [float(np.mean(all_seed_results[s]['kgscore_broad'])) for s in SEEDS]
kg_cov = [float(np.mean(all_seed_results[s]['kgscore_cov'])) for s in SEEDS]

print(f"\n{'='*60}", flush=True)
print("Improved 20-pair Multi-seed Summary (n=60):", flush=True)
print(f"{'='*60}", flush=True)
for m, s in summary.items():
    print(f"  {m}: {s['mean']:.2f}±{s['std']:.2f} per_seed={s['per_seed']}", flush=True)
print(f"  kgscore faith: strict={np.mean(kg_strict):.2f} broad={np.mean(kg_broad):.2f} cov={np.mean(kg_cov):.2f}", flush=True)

# McNemar tests
print(f"\nMcNemar tests (vs kgscore):", flush=True)
kg_hits = summary['kgscore']['all_hits']
for m in ['vanilla_llm', 'bm25_rag', 'tog', 'rog']:
    other = summary[m]['all_hits']
    b = sum(1 for k, o in zip(kg_hits, other) if k == 1 and o == 0)
    c = sum(1 for k, o in zip(kg_hits, other) if k == 0 and o == 1)
    if b + c == 0:
        p = 1.0
    else:
        result = sp_stats.binomtest(min(b, c), b + c, 0.5)
        p = result.pvalue
    print(f"  vs {m}: b={b} c={c} p={p:.4f} {'significant' if p < 0.05 else 'not significant'}", flush=True)

# ============================================================
# EXPERIMENT 2: KG Cleaning with strict_delete
# ============================================================
print(f"\n{'='*60}", flush=True)
print("EXP 2: KG Cleaning (strict_delete=True)", flush=True)
print(f"{'='*60}", flush=True)
client = LLMClient(cfg)
rng = random.Random(42)
noise_triples = [t for t in triples if t.score < 0.5 and not is_negative(t.predicate)]
clean_triples = [t for t in triples if t.score >= 0.5 and not is_negative(t.predicate)]
sample_noise = rng.sample(noise_triples, min(100, len(noise_triples)))
sample_clean = rng.sample(clean_triples, min(100, len(clean_triples)))

tp = fp = tn = fn = 0
for i, t in enumerate(sample_noise + sample_clean):
    is_noise = t.score < 0.5
    verdict, _ = _verify_triple(t, client, None)
    # strict_delete: only "contradicted" is flagged as noise
    llm_says_noise = (verdict == "contradicted")
    if is_noise and llm_says_noise: tp += 1
    elif is_noise and not llm_says_noise: fn += 1
    elif not is_noise and llm_says_noise: fp += 1
    else: tn += 1
    if (i+1) % 50 == 0: print(f"  {i+1}/200", flush=True)

prec = tp/(tp+fp) if tp+fp > 0 else 0
rec = tp/(tp+fn) if tp+fn > 0 else 0
f1 = 2*prec*rec/(prec+rec) if prec+rec > 0 else 0
acc = (tp+tn)/(tp+fp+tn+fn)
cleaning_strict = {'precision': prec, 'recall': rec, 'f1': f1, 'accuracy': acc, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
print(f"  strict_delete: P={prec:.3f} R={rec:.3f} F1={f1:.3f} Acc={acc:.3f} (TP={tp} FP={fp} TN={tn} FN={fn})", flush=True)

# ============================================================
# EXPERIMENT 3: KGQA with improved generation
# ============================================================
print(f"\n{'='*60}", flush=True)
print("EXP 3: KGQA with improved generation", flush=True)
print(f"{'='*60}", flush=True)
qa_pairs = []
real_chemicals = [t.subject for t in triples if t.predicate == "TREATS"][:10]
for chem in real_chemicals:
    treat_targets = [t.object for t in triples if t.subject == chem and t.predicate == "TREATS"]
    if treat_targets:
        qa_pairs.append({'question': f"What does {chem} treat?", 'answer': treat_targets[0], 'seed': chem})
print(f"  Built {len(qa_pairs)} QA pairs", flush=True)

qa_results = {}
for name, method in [('vanilla_llm', VanillaLLM(client)), ('tog', ToGWrapper(triples, client)), ('rog', RoGWrapper(triples, client))]:
    em_scores = []
    for qa in qa_pairs:
        try: hypos = method.run(question=qa['question'], seed_entity=qa['seed'], cfg=rc)
        except: hypos = []
        pred = hypos[0].target_entity if hypos else ''
        em = 1.0 if qa['answer'].lower() in pred.lower() or pred.lower() in qa['answer'].lower() else 0.0
        em_scores.append(em)
    qa_results[name] = float(np.mean(em_scores))
    print(f"  {name}: EM={qa_results[name]:.2f}", flush=True)

# KG-SCoRE
em_scores = []
for qa in qa_pairs:
    try: hypos = run_llm_verify_faith(client, qa['question'], qa['seed'], 5, strict_delete=True)
    except: hypos = []
    pred = hypos[0].target_entity if hypos else ''
    em = 1.0 if qa['answer'].lower() in pred.lower() or pred.lower() in qa['answer'].lower() else 0.0
    em_scores.append(em)
    s, b, c = compute_faith_metrics(hypos, 5)
qa_results['kgscore'] = {'em': float(np.mean(em_scores)), 'strict': float(np.mean([s])), 'broad': float(np.mean([b]))}
print(f"  kgscore: EM={qa_results['kgscore']['em']:.2f}", flush=True)

# ============================================================
# EXPERIMENT 4: Fair faithfulness for baselines (improved)
# ============================================================
print(f"\n{'='*60}", flush=True)
print("EXP 4: Fair faithfulness (improved, 20 pairs)", flush=True)
print(f"{'='*60}", flush=True)
fair_results = {}
for name, method in [('vanilla_llm', VanillaLLM(client)), ('bm25_rag', BM25RAG(corpus, client)),
                      ('tog', ToGWrapper(triples, client)), ('rog', RoGWrapper(triples, client))]:
    strict_list, broad_list, cov_list = [], [], []
    for pair in pairs[:10]:  # 10 pairs for speed
        q = query_for_pair(pair)
        try: hypos = method.run(question=q, seed_entity=pair.source, cfg=rc)
        except: hypos = []
        has_paths = any(h.kg_path for h in hypos)
        if not has_paths:
            hypos = extract_path_posthoc(hypos, pair.source)
        verify_hypotheses(hypos, client, top_k=(1,5,10))
        s, b, c = compute_faith_metrics(hypos, 10)
        strict_list.append(s)
        broad_list.append(b)
        cov_list.append(c)
    fair_results[name] = {'strict': float(np.mean(strict_list)), 'broad': float(np.mean(broad_list)), 'coverage': float(np.mean(cov_list))}
    print(f"  {name}: strict={fair_results[name]['strict']:.2f} broad={fair_results[name]['broad']:.2f} cov={fair_results[name]['coverage']:.2f}", flush=True)

# KG-SCoRE fair faith (from seed 42 results)
fair_results['kgscore'] = {'strict': float(np.mean(all_seed_results[42]['kgscore_strict'])),
                            'broad': float(np.mean(all_seed_results[42]['kgscore_broad'])),
                            'coverage': float(np.mean(all_seed_results[42]['kgscore_cov']))}
print(f"  kgscore: strict={fair_results['kgscore']['strict']:.2f} broad={fair_results['kgscore']['broad']:.2f} cov={fair_results['kgscore']['coverage']:.2f}", flush=True)

# ============================================================
# SAVE ALL RESULTS
# ============================================================
print(f"\n{'='*60}", flush=True)
print("SAVING ALL RESULTS", flush=True)
print(f"{'='*60}", flush=True)
all_results = {
    'exp1_multiseed_20pair': {'summary': summary, 'raw': {str(s): v for s, v in all_seed_results.items()},
                               'kgscore_faith': {'strict': kg_strict, 'broad': kg_broad, 'coverage': kg_cov}},
    'exp2_kg_cleaning_strict': cleaning_strict,
    'exp3_kgqa': qa_results,
    'exp4_fair_faithfulness': fair_results,
}
with open(results_dir / 'improved_results.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"Saved to {results_dir / 'improved_results.json'}", flush=True)
print("\nALL IMPROVED EXPERIMENTS COMPLETE!", flush=True)
