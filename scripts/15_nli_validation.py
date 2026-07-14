#!/usr/bin/env python3
"""Script 15 — NLI-based faithfulness validation.

Replaces the LLM judge with a trained NLI model (DeBERTa-v3-base-MNLI)
to provide an independent faithfulness assessment.

This addresses reviewer concern: "κ=0.095 means faithfulness is self-reported."
"""
import json, sys, re
sys.path.insert(0, '.')
import numpy as np
from kg_scaffold.utils.config import load_config, get_path, ensure_dirs

cfg = load_config(); ensure_dirs(cfg)
runs_dir = get_path('runs_dir')
results_dir = get_path('results_dir')
results_dir.mkdir(parents=True, exist_ok=True)

# Load all hypotheses with KG paths from ours_llm_faith.json and baseline_tog.json
def load_hypotheses(path):
    with open(path) as f:
        data = json.load(f)
    items = []
    for pid, hlist in data.get('predictions', {}).items():
        for h in hlist:
            if h.get('kg_path'):
                items.append({
                    'pair_id': pid,
                    'hypothesis': h['text'],
                    'kg_path': h['kg_path'],
                    'auto_label': h.get('faithfulness', ''),
                    'method': data.get('method', path.stem),
                })
    return items

all_items = []
for fname in ['ours_llm_faith.json', 'baseline_tog.json', 'baseline_rog.json']:
    p = runs_dir / fname
    if p.exists():
        all_items.extend(load_hypotheses(p))

if not all_items:
    print("No items with KG paths found!")
    sys.exit(1)

print(f"Loaded {len(all_items)} hypotheses with KG paths", flush=True)

# Try to load NLI model
NLI_AVAILABLE = False
try:
    from transformers import pipeline as hf_pipeline
    import torch
    print("Loading DeBERTa-v3-base-MNLI...", flush=True)
    nli_pipe = hf_pipeline(
        "text-classification",
        model="cross-encoder/nli-deberta-v3-base",
        device=0 if torch.cuda.is_available() else -1,
    )
    NLI_AVAILABLE = True
    print("NLI model loaded!", flush=True)
except Exception as e:
    print(f"NLI model not available: {e}", flush=True)
    print("Falling back to rule-based entailment + improved heuristic", flush=True)

# Rule-based entailment (improved)
def rule_based_entailment(hypothesis, kg_path):
    """Check if key entities from hypothesis appear in KG path."""
    if not kg_path or not hypothesis:
        return "none"
    # Extract entities from KG path
    path_entities = set()
    for part in re.split(r'->|\||,', kg_path):
        part = part.strip()
        if part and not part.isupper() and len(part) > 2:
            path_entities.add(part.lower())
    hypo_words = set(w.lower().strip(".,;:!?") for w in hypothesis.split() if len(w) > 3)
    if not path_entities or not hypo_words:
        return "none"
    overlap = path_entities & hypo_words
    ratio = len(overlap) / max(len(path_entities), 1)
    if ratio >= 0.5:
        return "entailed"
    elif ratio > 0:
        return "partial"
    return "none"

# NLI-based entailment
def nli_entailment(hypothesis, kg_path):
    """Use NLI model to check entailment."""
    if not kg_path or not hypothesis:
        return "none"
    # Format as premise-hypothesis pair
    premise = f"Knowledge graph path: {kg_path}"
    hypo = f"Hypothesis: {hypothesis}"
    try:
        result = nli_pipe(f"{premise} [SEP] {hypo}", top_k=None)
        # result is list of {label, score}
        scores = {r['label'].lower(): r['score'] for r in result}
        # Labels are typically: entailed, neutral, contradiction
        if 'entailment' in scores:
            ent_score = scores['entailment']
        elif 'entailed' in scores:
            ent_score = scores['entailed']
        else:
            ent_score = 0.0
        contra_score = scores.get('contradiction', scores.get('contradicted', 0.0))
        
        if ent_score > 0.5:
            return "entailed"
        elif ent_score > 0.2 or contra_score < 0.3:
            return "partial"
        else:
            return "none"
    except Exception as e:
        return "none"

# Run validation
results = []
for i, item in enumerate(all_items):
    if NLI_AVAILABLE:
        nli_label = nli_entailment(item['hypothesis'], item['kg_path'])
    else:
        nli_label = "none"
    rule_label = rule_based_entailment(item['hypothesis'], item['kg_path'])
    
    results.append({
        **item,
        'nli_label': nli_label,
        'rule_label': rule_label,
    })
    
    if (i + 1) % 50 == 0:
        print(f"  Processed {i+1}/{len(all_items)}", flush=True)

# Compute agreement
from scipy import stats as sp_stats

def cohens_kappa(labels1, labels2):
    if len(labels1) != len(labels2) or len(labels1) == 0:
        return 0.0
    labels = sorted(set(labels1) | set(labels2))
    n = len(labels1)
    matrix = np.zeros((len(labels), len(labels)))
    idx = {l: i for i, l in enumerate(labels)}
    for l1, l2 in zip(labels1, labels2):
        matrix[idx[l1]][idx[l2]] += 1
    po = np.trace(matrix) / n
    row = matrix.sum(axis=1) / n
    col = matrix.sum(axis=0) / n
    pe = np.sum(row * col)
    return 1.0 if pe == 1.0 else float((po - pe) / (1 - pe))

def label_to_score(l):
    return {"entailed": 1.0, "partial": 0.5, "none": 0.0}.get(l, 0.0)

auto_labels = [r['auto_label'] or 'none' for r in results]
nli_labels = [r['nli_label'] for r in results]
rule_labels = [r['rule_label'] for r in results]

print(f"\n=== Faithfulness Validation Results ===", flush=True)
if NLI_AVAILABLE:
    kappa_an = cohens_kappa(auto_labels, nli_labels)
    kappa_ar = cohens_kappa(auto_labels, rule_labels)
    kappa_nr = cohens_kappa(nli_labels, rule_labels)
    print(f"Cohen's kappa (auto-LLM vs NLI):     {kappa_an:.3f}", flush=True)
    print(f"Cohen's kappa (auto-LLM vs rule):    {kappa_ar:.3f}", flush=True)
    print(f"Cohen's kappa (NLI vs rule):         {kappa_nr:.3f}", flush=True)
    
    auto_scores = [label_to_score(l) for l in auto_labels]
    nli_scores = [label_to_score(l) for l in nli_labels]
    spearman = sp_stats.spearmanr(auto_scores, nli_scores)
    print(f"Spearman rho (auto-LLM vs NLI):      {spearman.statistic:.3f}", flush=True)
    
    agree = sum(1 for a, b in zip(auto_labels, nli_labels) if a == b) / len(auto_labels)
    print(f"Agreement (auto-LLM vs NLI):         {agree*100:.1f}%", flush=True)
    
    # NLI-based faithfulness@10 for KG-SCoRE
    kg_score_items = [r for r in results if r['method'] in ('llm_faith', 'kg_score')]
    if kg_score_items:
        nli_faith = np.mean([label_to_score(r['nli_label']) for r in kg_score_items])
        auto_faith = np.mean([label_to_score(r['auto_label']) for r in kg_score_items])
        print(f"\nKG-SCoRE Faith@10 (LLM judge):  {auto_faith:.3f}", flush=True)
        print(f"KG-SCoRE Faith@10 (NLI judge):  {nli_faith:.3f}", flush=True)
else:
    kappa_ar = cohens_kappa(auto_labels, rule_labels)
    print(f"Cohen's kappa (auto-LLM vs rule):    {kappa_ar:.3f}", flush=True)

# Save
out = results_dir / "nli_faithfulness_validation.json"
with open(out, 'w') as f:
    json.dump({
        'n_items': len(results),
        'nli_available': NLI_AVAILABLE,
        'cohens_kappa': {
            'auto_vs_nli': kappa_an if NLI_AVAILABLE else None,
            'auto_vs_rule': kappa_ar,
            'nli_vs_rule': kappa_nr if NLI_AVAILABLE else None,
        },
        'per_item': results,
    }, f, indent=2)
print(f"Saved to {out}", flush=True)
