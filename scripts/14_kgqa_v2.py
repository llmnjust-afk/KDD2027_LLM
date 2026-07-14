#!/usr/bin/env python3
"""Script 14 — KGQA for LLM-Verify+Faith."""
import json, sys, random
sys.path.insert(0, '.')
import numpy as np
from kg_scaffold.utils.config import load_config, get_path, ensure_dirs
from kg_scaffold.utils.lbd_gold import load_lbd_gold, query_for_pair
from kg_scaffold.utils.semmeddb import Triple, build_graph, is_negative
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

def load_kg(path):
    with open(path) as f:
        data = json.load(f)
    return [Triple(d['subject'], d['predicate'], d['object'], score=d.get('score',1.0)) for d in data]

def build_qa_pairs(triples, n=14, seed=42):
    rng = random.Random(seed)
    qa_pairs = []
    treat_triples = [t for t in triples if t.predicate in ('TREATS','INHIBITS')]
    rng.shuffle(treat_triples)
    for t in treat_triples[:n//2]:
        qa_pairs.append({'id': f'1hop-{len(qa_pairs)}', 'type':'one_hop',
            'question': f'What substance {t.predicate.lower()} {t.object}?',
            'answer': t.subject, 'seed_entity': t.object})
    assoc_triples = [t for t in triples if t.predicate == 'ASSOCIATED_WITH']
    rng.shuffle(assoc_triples)
    for t in assoc_triples[:n//2]:
        bridge = t.object
        candidates = [tt for tt in triples if tt.object == bridge and tt.predicate in ('INHIBITS','TREATS')]
        if candidates:
            qa_pairs.append({'id': f'2hop-{len(qa_pairs)}', 'type':'two_hop',
                'question': f'What substance inhibits {bridge}, which is associated with {t.subject}?',
                'answer': candidates[0].subject, 'seed_entity': t.subject, 'bridge': bridge})
    return qa_pairs

def run_llm_faith_qa(triples, corpus, client, retriever, question, seed_entity, num=5):
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
    verify_hypotheses(hypos, client, top_k=(1,5))
    return hypos

cfg = load_config(); ensure_dirs(cfg)
semmed_dir = get_path('semmeddb_dir')
triples = load_kg(semmed_dir / 'kg_triples.json')
corpus = [f'{t.subject} {t.predicate.lower().replace("_"," ")} {t.object}.' for t in triples]
client = LLMClient(cfg)
qa_pairs = build_qa_pairs(triples, n=14, seed=42)
print(f"Built {len(qa_pairs)} QA pairs", flush=True)
rc = RunConfig(num_hypotheses=5, use_kg=True, use_completion=False, use_faithfulness=True)
retriever = SubgraphRetriever(triples, max_hops=2, topk=40)

methods = {
    'vanilla_llm': VanillaLLM(client),
    'bm25_rag': BM25RAG(corpus, client),
    'tog': ToGWrapper(triples, client),
    'rog': RoGWrapper(triples, client),
}

results = {}
for name, method in methods.items():
    print(f"Running {name}...", flush=True)
    em_scores, f1_scores, faith_scores = [], [], []
    for qa in qa_pairs:
        try:
            hypos = method.run(question=qa['question'], seed_entity=qa['seed_entity'], cfg=rc)
        except:
            hypos = []
        pred = hypos[0].target_entity if hypos else ''
        p, g = pred.lower().strip(), qa['answer'].lower().strip()
        em = 1.0 if (g in p or p in g) else 0.0
        pt, gt = set(p.split()), set(g.split())
        tp = len(pt & gt)
        f1 = 2*tp/(len(pt)+len(gt)) if tp > 0 and pt and gt else 0.0
        em_scores.append(em); f1_scores.append(f1)
        top5 = hypos[:5]
        faith_scores.append(sum(1 for x in top5 if x.faithfulness in ('entailed','partial'))/len(top5) if top5 else 0.0)
    results[name] = {'em': float(np.mean(em_scores)), 'f1': float(np.mean(f1_scores)), 'faithfulness': float(np.mean(faith_scores))}
    print(f"  {name}: EM={results[name]['em']:.2f} F1={results[name]['f1']:.2f} Faith={results[name]['faithfulness']:.2f}", flush=True)

# LLM-faith
print("Running llm_faith...", flush=True)
em_scores, f1_scores, faith_scores = [], [], []
for qa in qa_pairs:
    try:
        hypos = run_llm_faith_qa(triples, corpus, client, retriever, qa['question'], qa['seed_entity'], num=5)
    except:
        hypos = []
    pred = hypos[0].target_entity if hypos else ''
    p, g = pred.lower().strip(), qa['answer'].lower().strip()
    em = 1.0 if (g in p or p in g) else 0.0
    pt, gt = set(p.split()), set(g.split())
    tp = len(pt & gt)
    f1 = 2*tp/(len(pt)+len(gt)) if tp > 0 and pt and gt else 0.0
    em_scores.append(em); f1_scores.append(f1)
    top5 = hypos[:5]
    faith_scores.append(sum(1 for x in top5 if x.faithfulness in ('entailed','partial'))/len(top5) if top5 else 0.0)
results['llm_faith'] = {'em': float(np.mean(em_scores)), 'f1': float(np.mean(f1_scores)), 'faithfulness': float(np.mean(faith_scores))}
print(f"  llm_faith: EM={results['llm_faith']['em']:.2f} F1={results['llm_faith']['f1']:.2f} Faith={results['llm_faith']['faithfulness']:.2f}", flush=True)

runs_dir = get_path('runs_dir')
with open(runs_dir / 'kgqa_v2.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n=== KGQA Summary ===", flush=True)
print(f"{'Method':<15} {'EM':>6} {'F1':>6} {'Faith':>6}", flush=True)
for name, res in results.items():
    print(f"{name:<15} {res['em']:.2f}   {res['f1']:.2f}   {res['faithfulness']:.2f}", flush=True)
print("Saved!", flush=True)
