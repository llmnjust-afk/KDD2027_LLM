#!/usr/bin/env python3
"""Generate all figures for the paper using final experiment results."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

COLORS = {
    'vanilla_llm': '#999999',
    'bm25_rag': '#66c2a4',
    'tog': '#fc8d62',
    'rog': '#8da0cb',
    'kgscore': '#e78ac3',
}
LABELS = {
    'vanilla_llm': 'Vanilla LLM',
    'bm25_rag': 'BM25-RAG',
    'tog': 'ToG',
    'rog': 'RoG',
    'kgscore': 'KG-SCoRE',
}

with open('/tmp/work/prj_01KXAZFT6R21GYEJMTNF94NW86/KDD2027_LLM/experiments/results/final_results_local.json') as f:
    data = json.load(f)

outdir = '/tmp/work/prj_01KXAZFT6R21GYEJMTNF94NW86/KDD2027_paper/figures/'

# ============================================================
# Figure 2: Hit@10 multi-seed bar chart with error bars
# ============================================================
summary = data['exp1_multiseed_20pair']['summary']
methods = ['vanilla_llm', 'bm25_rag', 'tog', 'rog', 'kgscore']
means = [summary[m]['mean'] for m in methods]
stds = [summary[m]['std'] for m in methods]
labels = [LABELS[m] for m in methods]
colors = [COLORS[m] for m in methods]

fig, ax = plt.subplots(figsize=(5, 3))
bars = ax.bar(labels, means, yerr=stds, capsize=4, color=colors, edgecolor='black', linewidth=0.5, width=0.6)
ax.set_ylabel('Hit@10')
ax.set_title('LBD Re-discovery Hit@10 (n=60, 3 seeds × 20 pairs)')
ax.set_ylim(0, 0.55)
for bar, mean in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.015,
            f'{mean:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.axhline(y=0.40, color='#8da0cb', linestyle='--', alpha=0.5, linewidth=0.8)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(outdir + 'hit10_multiseed.pdf')
plt.close()
print("Saved hit10_multiseed.pdf")

# ============================================================
# Figure 3: Faithfulness comparison (grouped bar chart)
# ============================================================
fair = data['exp3_fair_faithfulness']
metrics = ['strict', 'broad', 'coverage']
metric_labels = ['StrictFaith@10', 'BroadFaith@10', 'Coverage']
x = np.arange(len(metrics))
width = 0.15

fig, ax = plt.subplots(figsize=(6, 3.5))
for i, m in enumerate(methods):
    vals = [fair[m][metric] for metric in metrics]
    bars = ax.bar(x + i * width, vals, width, label=LABELS[m], color=COLORS[m], edgecolor='black', linewidth=0.4)
    for bar, val in zip(bars, vals):
        if val > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=7)

ax.set_ylabel('Score')
ax.set_title('Fair Faithfulness Comparison (10 pairs, post-hoc path extraction)')
ax.set_xticks(x + width * 2)
ax.set_xticklabels(metric_labels)
ax.set_ylim(0, 1.15)
ax.legend(loc='upper left', ncol=2)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(outdir + 'faithfulness_comparison.pdf')
plt.close()
print("Saved faithfulness_comparison.pdf")

# ============================================================
# Figure 4: KGQA EM bar chart
# ============================================================
kgqa = data['exp4_kgqa']
qa_methods = ['vanilla_llm', 'tog', 'rog', 'kgscore']
qa_labels = [LABELS[m] for m in qa_methods]
qa_vals = []
for m in qa_methods:
    v = kgqa[m]
    qa_vals.append(v['em'] if isinstance(v, dict) else v)
qa_colors = [COLORS[m] for m in qa_methods]

fig, ax = plt.subplots(figsize=(4, 3))
bars = ax.bar(qa_labels, qa_vals, color=qa_colors, edgecolor='black', linewidth=0.5, width=0.5)
ax.set_ylabel('Exact Match (EM)')
ax.set_title('KGQA Performance (10 single-hop questions)')
ax.set_ylim(0, 0.40)
for bar, val in zip(bars, qa_vals):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(outdir + 'kgqa_em.pdf')
plt.close()
print("Saved kgqa_em.pdf")

# ============================================================
# Figure 5: KG Cleaning precision-recall tradeoff
# ============================================================
cleaning = data['exp2_kg_cleaning']
cons = cleaning['conservative']
agg = cleaning['aggressive']

fig, ax = plt.subplots(figsize=(4.5, 3.5))
# Plot precision-recall points
modes = ['Conservative\n(delete contradicted\nonly)', 'Aggressive\n(delete contradicted\n+ unverifiable)']
precisions = [cons['precision'], agg['precision']]
recalls = [cons['recall'], agg['recall']]

ax.scatter(recalls, precisions, s=150, c=['#2ca02c', '#d62728'], zorder=5, edgecolors='black')
for i, (r, p, label) in enumerate(zip(recalls, precisions, modes)):
    ax.annotate(label, (r, p), textcoords="offset points", xytext=(15, -10),
                fontsize=8, arrowprops=dict(arrowstyle='->', lw=0.5))

# Connect points
ax.plot(recalls, precisions, 'k--', alpha=0.3, linewidth=0.8)

# F1 labels
f1s = [cons['f1'], agg['f1']]
for r, p, f1 in zip(recalls, precisions, f1s):
    ax.text(r, p - 0.08, f'F1={f1:.3f}', ha='center', fontsize=8, color='gray')

ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('KG Cleaning: Precision-Recall Tradeoff')
ax.set_xlim(-0.1, 1.15)
ax.set_ylim(0, 1.0)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(outdir + 'kg_cleaning_tradeoff.pdf')
plt.close()
print("Saved kg_cleaning_tradeoff.pdf")

# ============================================================
# Figure 6: Radar chart — all metrics for top 3 methods
# ============================================================
# Normalize all metrics to [0,1]
categories = ['Hit@10', 'StrictFaith', 'BroadFaith', 'Coverage', 'KGQA EM']
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

# Get values for each method
def get_method_values(m):
    hit = summary[m]['mean']
    sf = fair[m]['strict']
    bf = fair[m]['broad']
    cov = fair[m]['coverage']
    if m in kgqa:
        v = kgqa[m]
        em = v['em'] if isinstance(v, dict) else v
    else:
        em = 0.0
    return [hit, sf, bf, cov, em]

fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(polar=True))
for m in ['rog', 'tog', 'kgscore']:
    vals = get_method_values(m)
    vals += vals[:1]
    ax.plot(angles, vals, 'o-', linewidth=1.5, label=LABELS[m], color=COLORS[m], markersize=4)
    ax.fill(angles, vals, alpha=0.08, color=COLORS[m])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=9)
ax.set_ylim(0, 1.05)
ax.set_title('Multi-metric Comparison', fontsize=12, pad=20)
ax.legend(loc='lower right', bbox_to_anchor=(1.25, -0.05), fontsize=9)
plt.tight_layout()
plt.savefig(outdir + 'radar_comparison.pdf')
plt.close()
print("Saved radar_comparison.pdf")

print("\nAll figures generated!")
