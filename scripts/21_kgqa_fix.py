#!/usr/bin/env python3
"""Script 21 — Re-run KGQA with QA-specific prompt.

Fix: KGQA uses a separate prompt that allows parametric knowledge use,
while still doing faithfulness verification + reranking.
LBD task keeps the strict prompt (only path-supported claims).
"""
import json, sys, random, re, numpy as np
sys.path.insert(0, '.')
from kg_scaffold.utils.config import load_config, get_path, ensure_dirs
from kg_scaffold.utils.lbd_gold import load_lbd_gold, query_for_pair, hit_at_k
from kg_scaffold.utils.semmeddb import Triple, is_negative
from kg_scaffold.utils.prompts import KGQA_GEN_WITH_KG
from kg_scaffold.generation.llm_client import LLMClient
from kg_scaffold.retrieval.subgraph import SubgraphRetriever, Subgraph
from kg_scaffold.generation.hypothesis_gen import Hypothesis, parse_hypotheses
from kg_scaffold.verification.faithfulness import verify_hypotheses, ENTAILED, PARTIAL, NONE
from kg_scaffold.kg.refinement import _verify_triple
from kg_scaffold.baselines.base import RunConfig
from kg_scaffold.baselines.vanilla_llm import VanillaLLM
from kg_scaffold.baselines.tog_wrapper import ToGWrapper
from kg_scaffold.baselines.rog_wrapper import RoGWrapper

cfg = load_config(); ensure_dirs(cfg)
semmed_dir = get_path('semmeddb_dir')
results_dir = get_path('results_dir')

with open(semmed_dir / 'kg_triples.json') as f:
    data = json.load(f)
triples = [Triple(d['subject'], d['predicate'], d['object'], score=d.get('score',1.0)) for d in data]
corpus = [f'{t.subject} {t.predicate.lower().replace("_"," ")} {t.object}.' for t in triples]
retriever = SubgraphRetriever(triples, max_hops=2, topk=40)
client = LLMClient(cfg)

FAITH_RANK = {ENTAILED: 0, PARTIAL: 1, NONE: 2, "": 3, "none": 2}

def faithfulness_rerank(hypos, k=5):
    ranked = sorted(hypos, key=lambda h: (FAITH_RANK.get(h.faithfulness, 3), hypos.index(h)))
    return ranked[:k]

def compute_faith_metrics(hypos, k=5):
    top = hypos[:k]
    if not top: return 0, 0, 0
    n = len(top)
    strict = sum(1 for h in top if h.faithfulness == ENTAILED) / n
    broad = sum(1 for h in top if h.faithfulness in (ENTAILED, PARTIAL)) / n
    coverage = sum(1 for h in top if h.kg_path and h.kg_path != "parametric") / n
    return strict, broad, coverage

def run_kgscore_qa(client, question, seed_entity, num_gen=10, num_ret=5):
    """KG-SCoRE for QA: uses QA-specific prompt (allows parametric knowledge)."""
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
            verified.append(t)  # keep unverifiable (strict_delete)
    verified_sub = Subgraph(root=seed_entity, triples=verified[:40])
    
    # Use QA-specific prompt
    kg_text = verified_sub.as_text()
    prompt = KGQA_GEN_WITH_KG.format(question=question, kg_subgraph=kg_text, num=num_gen)
    
    try:
        raw = client.complete(prompt, temperature=0.2, max_tokens=min(2048, 256*num_gen))
    except Exception as e:
        print(f"  QA gen error: {e}", flush=True)
        return []
    
    hypos = parse_hypotheses(raw)
    for i, h in enumerate(hypos):
        h.score = 1.0 / (i + 1)
    
    # Faithfulness verification + reranking
    verify_hypotheses(hypos, client, top_k=(1,5,10))
    hypos = faithfulness_rerank(hypos, k=num_ret)
    
    return hypos

# ============================================================
# KGQA with QA-specific prompt
# ============================================================
print("=" * 60, flush=True)
print("KGQA with QA-specific prompt (fix)", flush=True)
print("=" * 60, flush=True)

qa_pairs = []
real_chemicals = [t.subject for t in triples if t.predicate == "TREATS"][:10]
for chem in real_chemicals:
    treat_targets = [t.object for t in triples if t.subject == chem and t.predicate == "TREATS"]
    if treat_targets:
        qa_pairs.append({'question': f"What does {chem} treat?", 'answer': treat_targets[0], 'seed': chem})
print(f"  Built {len(qa_pairs)} QA pairs", flush=True)

rc = RunConfig(num_hypotheses=5, use_kg=True, use_completion=False, use_faithfulness=True)
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

# KG-SCoRE with QA prompt + reranking
em_scores = []
strict_list, broad_list, cov_list = [], [], []
for qa in qa_pairs:
    try: hypos = run_kgscore_qa(client, qa['question'], qa['seed'], num_gen=10, num_ret=5)
    except Exception as e:
        print(f"  kgscore error: {e}", flush=True)
        hypos = []
    pred = hypos[0].target_entity if hypos else ''
    em = 1.0 if qa['answer'].lower() in pred.lower() or pred.lower() in qa['answer'].lower() else 0.0
    em_scores.append(em)
    s, b, c = compute_faith_metrics(hypos, 5)
    strict_list.append(s)
    broad_list.append(b)
    cov_list.append(c)
qa_results['kgscore'] = {
    'em': float(np.mean(em_scores)),
    'strict': float(np.mean(strict_list)),
    'broad': float(np.mean(broad_list)),
    'coverage': float(np.mean(cov_list)),
}
print(f"  kgscore: EM={qa_results['kgscore']['em']:.2f} strict={qa_results['kgscore']['strict']:.2f} broad={qa_results['kgscore']['broad']:.2f}", flush=True)

# Save
output = {'kgqa_fixed': qa_results}
with open(results_dir / 'kgqa_fixed.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nSaved to {results_dir / 'kgqa_fixed.json'}", flush=True)
print("DONE!", flush=True)
