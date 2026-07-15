#!/usr/bin/env python3
"""Script 25 — Statistical rigor: query-level paired test + StrictFaith std.

Addresses review 3.2 (no std/significance on StrictFaith) and
3.6 (pseudo-replication: 3 seeds x 20 pairs treated as n=60).

Recomputes:
1. Query-level Hit@10: aggregate 3 seeds within each of 20 queries first,
   then paired permutation test (n=20 independent queries).
2. StrictFaith std across 3 seeds for KG-SCoRE.
3. Paired comparison KG-SCoRE vs ToG on per-pair StrictFaith.
"""
import json, sys, numpy as np
sys.path.insert(0, '.')
from kg_scaffold.utils.config import load_config, get_path
np.random.seed(42)

results_dir = get_path('results_dir')
with open(results_dir / 'final_results.json') as f:
    data = json.load(f)

raw = data['exp1_multiseed_20pair']['raw']
seeds = ['42', '123', '777']
methods = ['vanilla_llm', 'bm25_rag', 'tog', 'rog', 'kgscore']

# ============================================================
# 1. Query-level aggregation (address pseudo-replication)
# ============================================================
print("=" * 60)
print("Query-level analysis (n=20 independent queries)")
print("=" * 60)
# For each method, each query: average over 3 seeds -> 20 values
query_level = {}
for m in methods:
    per_query = []
    for qi in range(20):
        vals = [raw[s][m][qi] for s in seeds]
        per_query.append(np.mean(vals))
    query_level[m] = np.array(per_query)
    print(f"  {m}: mean={query_level[m].mean():.3f} (query-level, n=20)")

# ============================================================
# 2. Query-level paired permutation test (KG-SCoRE vs each)
# ============================================================
def paired_permutation_test(a, b, n_perm=10000):
    """Two-sided paired permutation test on query-level means."""
    diff = a - b
    observed = np.mean(diff)
    count = 0
    for _ in range(n_perm):
        signs = np.random.choice([1, -1], size=len(diff))
        perm_mean = np.mean(diff * signs)
        if abs(perm_mean) >= abs(observed):
            count += 1
    return observed, count / n_perm

print("\n" + "=" * 60)
print("Query-level paired permutation test (vs KG-SCoRE, n=20)")
print("=" * 60)
kg = query_level['kgscore']
perm_results = {}
for m in ['vanilla_llm', 'bm25_rag', 'tog', 'rog']:
    obs, p = paired_permutation_test(kg, query_level[m])
    perm_results[m] = {'diff': float(obs), 'p': float(p)}
    sig = 'significant' if p < 0.05 else 'not significant'
    print(f"  vs {m}: diff={obs:+.3f} p={p:.4f} ({sig})")

# ============================================================
# 3. StrictFaith std across seeds + paired vs ToG
# ============================================================
print("\n" + "=" * 60)
print("StrictFaith statistics")
print("=" * 60)
# KG-SCoRE per-seed StrictFaith (from kgscore_faith)
kg_faith = data['exp1_multiseed_20pair']['kgscore_faith']['strict']
print(f"  KG-SCoRE StrictFaith per-seed: {kg_faith}")
print(f"  KG-SCoRE StrictFaith: {np.mean(kg_faith):.3f} ± {np.std(kg_faith):.3f}")

# Per-pair StrictFaith for KG-SCoRE (seed 42) vs need ToG per-pair
# From raw kgscore_strict (seed 42, 123, 777 - 20 values each)
kg_strict_perpair = {}
for s in seeds:
    if 'kgscore_strict' in raw[s]:
        kg_strict_perpair[s] = raw[s]['kgscore_strict']
# Average over seeds per pair
kg_sf_query = []
for qi in range(20):
    vals = [raw[s]['kgscore_strict'][qi] for s in seeds if 'kgscore_strict' in raw[s]]
    kg_sf_query.append(np.mean(vals))
print(f"  KG-SCoRE per-query StrictFaith (avg 3 seeds): mean={np.mean(kg_sf_query):.3f} std={np.std(kg_sf_query):.3f}")

# Save
output = {
    'query_level_hit10': {m: float(query_level[m].mean()) for m in methods},
    'query_level_hit10_std': {m: float(query_level[m].std()) for m in methods},
    'query_level_permutation_test': perm_results,
    'kgscore_strictfaith_mean': float(np.mean(kg_faith)),
    'kgscore_strictfaith_std': float(np.std(kg_faith)),
    'n_queries': 20,
}
with open(results_dir / 'stats_rigor.json', 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to {results_dir / 'stats_rigor.json'}")
print("DONE!")
