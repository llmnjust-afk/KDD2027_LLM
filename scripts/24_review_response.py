#!/usr/bin/env python3
"""Script 24 — Review-response experiments.

Addresses KDD reviewer concerns:
1. Rule-based cleaning baseline (type-schema + direction checker) vs LLM verification
2. Reranking before/after StrictFaith comparison (isolate reranking effect)
3. Independent judge (rule-based NLI-style) re-evaluation of reranked StrictFaith
   to address circular-evaluation concern
"""
import json, sys, random, re, numpy as np
sys.path.insert(0, '.')
from kg_scaffold.utils.config import load_config, get_path, ensure_dirs
from kg_scaffold.utils.lbd_gold import load_lbd_gold, query_for_pair, hit_at_k
from kg_scaffold.utils.semmeddb import Triple, is_negative
from kg_scaffold.utils.prompts import KGQA_GEN_WITH_KG
from kg_scaffold.generation.llm_client import LLMClient
from kg_scaffold.retrieval.subgraph import SubgraphRetriever, Subgraph
from kg_scaffold.generation.hypothesis_gen import generate_hypotheses
from kg_scaffold.verification.faithfulness import verify_hypotheses, ENTAILED, PARTIAL, NONE
from kg_scaffold.kg.refinement import _verify_triple

cfg = load_config(); ensure_dirs(cfg)
semmed_dir = get_path('semmeddb_dir')
results_dir = get_path('results_dir')

with open(semmed_dir / 'kg_triples.json') as f:
    data = json.load(f)
triples = [Triple(d['subject'], d['predicate'], d['object'], score=d.get('score',1.0)) for d in data]
corpus = [f'{t.subject} {t.predicate.lower().replace("_"," ")} {t.object}.' for t in triples]
pairs = load_lbd_gold()
retriever = SubgraphRetriever(triples, max_hops=2, topk=40)
client = LLMClient(cfg)

FAITH_RANK = {ENTAILED: 0, PARTIAL: 1, NONE: 2, "": 3, "none": 2}

# ============================================================
# Build entity type map + relation schema (for rule baseline)
# ============================================================
# Infer entity types from the KG generator's naming convention
def entity_type(name):
    n = name.lower()
    # Heuristic type inference from known biomedical entities
    diseases = ['migraine','asthma','psoriasis','hypertension','diabetes','depression',
                'alzheimer','arthritis','cancer','parkinson','raynaud','headache','deficiency','syndrome','disease']
    chemicals = ['vitamin','caffeine','metformin','lithium','omega','aspirin','fish oil',
                 'magnesium','riboflavin','curcumin','sumatriptan','acid','drug']
    if any(d in n for d in diseases): return 'disease'
    if any(c in n for c in chemicals): return 'chemical'
    return 'other'

# Relation schema: valid (subject_type, relation, object_type)
RELATION_SCHEMA = {
    'TREATS': ('chemical', 'disease'),
    'CAUSES': ('chemical', 'disease'),
    'PREVENTS': ('chemical', 'disease'),
    'ISA': ('any', 'any'),
    'CO-OCCURS_WITH': ('any', 'any'),
    'AFFECTS': ('any', 'any'),
}

def rule_based_is_noise(t):
    """Rule-based noise detector: type-schema + direction check.
    Returns True if flagged as noise."""
    pred = t.predicate.upper()
    st = entity_type(t.subject)
    ot = entity_type(t.object)
    if pred in RELATION_SCHEMA:
        exp_s, exp_o = RELATION_SCHEMA[pred]
        # Type violation check
        if exp_s != 'any' and st != 'other' and st != exp_s:
            return True
        if exp_o != 'any' and ot != 'other' and ot != exp_o:
            return True
        # Direction check: TREATS/CAUSES/PREVENTS from disease to chemical is reversed
        if exp_s == 'chemical' and st == 'disease' and ot == 'chemical':
            return True
    return False

# ============================================================
# EXP 1: Rule-based cleaning baseline vs LLM
# ============================================================
print("=" * 60, flush=True)
print("EXP 1: Rule-based cleaning baseline", flush=True)
print("=" * 60, flush=True)
rng = random.Random(42)
noise_triples = [t for t in triples if t.score < 0.5 and not is_negative(t.predicate)]
clean_triples = [t for t in triples if t.score >= 0.5 and not is_negative(t.predicate)]
sample_noise = rng.sample(noise_triples, min(100, len(noise_triples)))
sample_clean = rng.sample(clean_triples, min(100, len(clean_triples)))
sample = sample_noise + sample_clean

# Rule-based
tp = fp = tn = fn = 0
for t in sample:
    is_noise = t.score < 0.5
    flagged = rule_based_is_noise(t)
    if is_noise and flagged: tp += 1
    elif is_noise and not flagged: fn += 1
    elif not is_noise and flagged: fp += 1
    else: tn += 1
prec = tp/(tp+fp) if tp+fp else 0
rec = tp/(tp+fn) if tp+fn else 0
f1 = 2*prec*rec/(prec+rec) if prec+rec else 0
rule_result = {'precision': prec, 'recall': rec, 'f1': f1, 'accuracy': (tp+tn)/len(sample),
               'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
print(f"  Rule-based: P={prec:.3f} R={rec:.3f} F1={f1:.3f} Acc={(tp+tn)/len(sample):.3f} (TP={tp} FP={fp} TN={tn} FN={fn})", flush=True)

# ============================================================
# EXP 2: Reranking before/after + independent judge
# ============================================================
print("\n" + "=" * 60, flush=True)
print("EXP 2: Reranking before/after + independent judge", flush=True)
print("=" * 60, flush=True)

def independent_judge(hypo_text, kg_path):
    """Rule-based independent judge (lexical entailment proxy).
    Different architecture from LLM judge -> breaks circularity.
    entailed: all path entities in hypo AND no extra causal/quant claims
    partial: path entities in hypo but extra claims present
    none: path entities not in hypo
    """
    if not kg_path:
        return NONE
    # Extract entities from path
    path_ents = re.split(r'\s*->\s*|\s*\|\s*|\s+', kg_path.lower())
    path_ents = [e.strip() for e in path_ents if len(e.strip()) > 2]
    hypo_lower = hypo_text.lower()
    matched = sum(1 for e in path_ents if e in hypo_lower)
    coverage = matched / len(path_ents) if path_ents else 0
    # Check for extra causal/quantitative claims
    extra_markers = ['%', 'percent', 'reduce', 'increase', 'by ', 'mechanism', 'because', 'due to']
    has_extra = any(m in hypo_lower for m in extra_markers)
    if coverage >= 0.6 and not has_extra:
        return ENTAILED
    elif coverage >= 0.3:
        return PARTIAL
    else:
        return NONE

def run_kgscore_detailed(client, question, seed_entity, num_gen=20):
    """Return hypotheses with both pre-rerank order and faithfulness labels."""
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
            pass
        else:
            verified.append(t)
    verified_sub = Subgraph(root=seed_entity, triples=verified[:40])
    snippets = [t for t in corpus if seed_entity.lower() in t.lower()][:5]
    hypos = generate_hypotheses(question=question, subgraph=verified_sub, snippets=snippets, client=client, num=num_gen, use_kg=True)
    verify_hypotheses(hypos, client, top_k=(1,5,10,20))
    return hypos

def strict_at_k(hypos, k, use_independent=False):
    top = hypos[:k]
    if not top: return 0
    if use_independent:
        return sum(1 for h in top if independent_judge(h.text, h.kg_path) == ENTAILED) / len(top)
    return sum(1 for h in top if h.faithfulness == ENTAILED) / len(top)

# Run on 10 pairs, single seed
pre_llm, post_llm, post_indep = [], [], []
for pair in pairs[:10]:
    q = query_for_pair(pair)
    try:
        hypos = run_kgscore_detailed(client, q, pair.source, num_gen=20)
    except Exception as e:
        print(f"  {pair.source}: ERROR {e}", flush=True)
        continue
    # Pre-rerank: original LLM order, top-10 StrictFaith (LLM judge)
    pre_llm.append(strict_at_k(hypos, 10, use_independent=False))
    # Post-rerank: sort by faithfulness, top-10 StrictFaith (LLM judge)
    reranked = sorted(hypos, key=lambda h: (FAITH_RANK.get(h.faithfulness, 3), hypos.index(h)))
    post_llm.append(strict_at_k(reranked, 10, use_independent=False))
    # Post-rerank evaluated by INDEPENDENT judge (breaks circularity)
    post_indep.append(strict_at_k(reranked, 10, use_independent=True))
    print(f"  {pair.source}: pre={pre_llm[-1]:.2f} post_llm={post_llm[-1]:.2f} post_indep={post_indep[-1]:.2f}", flush=True)

rerank_result = {
    'pre_rerank_strict_llm': float(np.mean(pre_llm)),
    'post_rerank_strict_llm': float(np.mean(post_llm)),
    'post_rerank_strict_independent': float(np.mean(post_indep)),
    'pre_std': float(np.std(pre_llm)),
    'post_llm_std': float(np.std(post_llm)),
    'post_indep_std': float(np.std(post_indep)),
    'n': len(pre_llm),
}
print(f"\n  Pre-rerank StrictFaith (LLM judge):  {np.mean(pre_llm):.3f}±{np.std(pre_llm):.3f}", flush=True)
print(f"  Post-rerank StrictFaith (LLM judge): {np.mean(post_llm):.3f}±{np.std(post_llm):.3f}", flush=True)
print(f"  Post-rerank StrictFaith (INDEP judge): {np.mean(post_indep):.3f}±{np.std(post_indep):.3f}", flush=True)

# Save
output = {'exp1_rule_baseline': rule_result, 'exp2_rerank_independent': rerank_result}
with open(results_dir / 'review_response.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nSaved to {results_dir / 'review_response.json'}", flush=True)
print("DONE!", flush=True)
