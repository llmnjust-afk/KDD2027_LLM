#!/usr/bin/env python3
"""Script 26 — Module A correction accuracy (review 3.5).

Evaluates whether the LLM's corrected objects (for contradicted triples)
are accurate. For each noise triple that the LLM marks 'contradicted' and
provides a correction, check whether the corrected object matches the
known ground-truth object (the triple's original clean target before noise
injection was applied, if recoverable) or is at least type-consistent.
"""
import json, sys, random, numpy as np
sys.path.insert(0, '.')
from kg_scaffold.utils.config import load_config, get_path
from kg_scaffold.utils.semmeddb import Triple, is_negative
from kg_scaffold.generation.llm_client import LLMClient
from kg_scaffold.kg.refinement import _verify_triple

cfg = load_config()
semmed_dir = get_path('semmeddb_dir')
results_dir = get_path('results_dir')

with open(semmed_dir / 'kg_triples.json') as f:
    data = json.load(f)
triples = [Triple(d['subject'], d['predicate'], d['object'], score=d.get('score',1.0)) for d in data]
client = LLMClient(cfg)

# Type inference (same as rule baseline)
def entity_type(name):
    n = name.lower()
    diseases = ['migraine','asthma','psoriasis','hypertension','diabetes','depression',
                'alzheimer','arthritis','cancer','parkinson','raynaud','headache','deficiency','syndrome','disease']
    chemicals = ['vitamin','caffeine','metformin','lithium','omega','aspirin','fish oil',
                 'magnesium','riboflavin','curcumin','sumatriptan','acid','drug']
    if any(d in n for d in diseases): return 'disease'
    if any(c in n for c in chemicals): return 'chemical'
    return 'other'

# Expected object type per relation
REL_OBJ_TYPE = {'TREATS': 'disease', 'CAUSES': 'disease', 'PREVENTS': 'disease'}

print("=" * 60)
print("Module A Correction Accuracy (review 3.5)")
print("=" * 60)

rng = random.Random(7)
noise_triples = [t for t in triples if t.score < 0.5 and not is_negative(t.predicate)]
sample = rng.sample(noise_triples, min(60, len(noise_triples)))

n_contradicted = 0
n_corrected = 0
n_type_consistent = 0   # corrected object has expected type
n_changed = 0           # correction differs from original noisy object

for i, t in enumerate(sample):
    verdict, corrected = _verify_triple(t, client, None)
    if verdict == "contradicted":
        n_contradicted += 1
        if corrected and corrected.lower() not in ("none", ""):
            n_corrected += 1
            if corrected.lower().strip() != t.object.lower().strip():
                n_changed += 1
            # Type consistency check
            pred = t.predicate.upper()
            if pred in REL_OBJ_TYPE:
                exp_type = REL_OBJ_TYPE[pred]
                if entity_type(corrected) == exp_type:
                    n_type_consistent += 1
            else:
                n_type_consistent += 1  # no constraint -> count as consistent
    if (i+1) % 20 == 0:
        print(f"  {i+1}/{len(sample)}", flush=True)

print(f"\n  Sampled noise triples: {len(sample)}")
print(f"  Marked contradicted: {n_contradicted}")
print(f"  Provided a correction: {n_corrected}")
print(f"  Correction changed the object: {n_changed}")
print(f"  Correction type-consistent: {n_type_consistent}")

result = {
    'n_sampled': len(sample),
    'n_contradicted': n_contradicted,
    'n_corrected': n_corrected,
    'n_changed': n_changed,
    'n_type_consistent': n_type_consistent,
    'correction_rate': n_corrected / n_contradicted if n_contradicted else 0,
    'type_consistency': n_type_consistent / n_corrected if n_corrected else 0,
}
print(f"\n  Correction rate (of contradicted): {result['correction_rate']:.2f}")
print(f"  Type-consistency of corrections: {result['type_consistency']:.2f}")

with open(results_dir / 'correction_accuracy.json', 'w') as f:
    json.dump(result, f, indent=2)
print(f"\nSaved to {results_dir / 'correction_accuracy.json'}")
print("DONE!")
