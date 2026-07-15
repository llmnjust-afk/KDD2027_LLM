#!/usr/bin/env python3
"""Generate improved Figure 4 (KGQA) and Figure 5 (KG cleaning) + combined table."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

with open('/tmp/work/prj_01KXAZFT6R21GYEJMTNF94NW86/KDD2027_LLM/experiments/results/final_results_local.json') as f:
    data = json.load(f)

outdir = '/tmp/work/prj_01KXAZFT6R21GYEJMTNF94NW86/KDD2027_paper/figures/'

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

# ============================================================
# Figure 4: KGQA — combined EM + StrictFaith (side by side)
# ============================================================
kgqa = data['exp4_kgqa']
qa_methods = ['vanilla_llm', 'tog', 'rog', 'kgscore']
qa_labels = [LABELS[m] for m in qa_methods]
em_vals = []
sf_vals = []
for m in qa_methods:
    v = kgqa[m]
    if isinstance(v, dict):
        em_vals.append(v['em'])
        sf_vals.append(v.get('strict', 0))
    else:
        em_vals.append(v)
        sf_vals.append(0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
x = np.arange(len(qa_methods))
width = 0.5

# Left: EM
bars1 = ax1.bar(x, em_vals, width, color=[COLORS[m] for m in qa_methods], edgecolor='black', linewidth=0.5)
ax1.set_ylabel('Exact Match (EM)')
ax1.set_title('(a) KGQA Accuracy')
ax1.set_xticks(x)
ax1.set_xticklabels(qa_labels, rotation=30, ha='right')
ax1.set_ylim(0, 0.40)
for bar, val in zip(bars1, em_vals):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
             f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Right: StrictFaith@5
bars2 = ax2.bar(x, sf_vals, width, color=[COLORS[m] for m in qa_methods], edgecolor='black', linewidth=0.5)
ax2.set_ylabel('StrictFaith@5')
ax2.set_title('(b) Answer Traceability')
ax2.set_xticks(x)
ax2.set_xticklabels(qa_labels, rotation=30, ha='right')
ax2.set_ylim(0, 0.40)
for bar, val in zip(bars2, sf_vals):
    if val > 0:
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(outdir + 'kgqa_combined.pdf')
plt.close()
print("Saved kgqa_combined.pdf")

# ============================================================
# Figure 5: KG Cleaning — confusion matrices + PR tradeoff
# ============================================================
cleaning = data['exp2_kg_cleaning']
cons = cleaning['conservative']
agg = cleaning['aggressive']

fig = plt.figure(figsize=(7, 3.5))
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.2])

# Left: Conservative confusion matrix
ax1 = fig.add_subplot(gs[0])
cm_cons = np.array([[cons['tn'], cons['fp']], [cons['fn'], cons['tp']]])
im1 = ax1.imshow(cm_cons, cmap='Blues', aspect='auto')
ax1.set_title('(a) Conservative', fontsize=10)
ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
ax1.set_xticklabels(['Clean', 'Noise'])
ax1.set_yticklabels(['Clean', 'Noise'])
ax1.set_xlabel('Predicted', fontsize=9)
ax1.set_ylabel('Actual', fontsize=9)
for i in range(2):
    for j in range(2):
        ax1.text(j, i, str(cm_cons[i, j]), ha='center', va='center',
                 fontsize=12, fontweight='bold',
                 color='white' if cm_cons[i, j] > 50 else 'black')
ax1.text(0.5, -0.35, f'P={cons["precision"]:.2f}\nR={cons["recall"]:.2f}\nF1={cons["f1"]:.2f}',
         ha='center', va='top', fontsize=8, transform=ax1.transAxes)

# Middle: Aggressive confusion matrix
ax2 = fig.add_subplot(gs[1])
cm_agg = np.array([[agg['tn'], agg['fp']], [agg['fn'], agg['tp']]])
im2 = ax2.imshow(cm_agg, cmap='Oranges', aspect='auto')
ax2.set_title('(b) Aggressive', fontsize=10)
ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])
ax2.set_xticklabels(['Clean', 'Noise'])
ax2.set_yticklabels(['Clean', 'Noise'])
ax2.set_xlabel('Predicted', fontsize=9)
for i in range(2):
    for j in range(2):
        ax2.text(j, i, str(cm_agg[i, j]), ha='center', va='center',
                 fontsize=12, fontweight='bold',
                 color='white' if cm_agg[i, j] > 50 else 'black')
ax2.text(0.5, -0.35, f'P={agg["precision"]:.2f}\nR={agg["recall"]:.2f}\nF1={agg["f1"]:.2f}',
         ha='center', va='top', fontsize=8, transform=ax2.transAxes)

# Right: PR tradeoff with F1
ax3 = fig.add_subplot(gs[2])
modes = ['Conservative', 'Aggressive']
precisions = [cons['precision'], agg['precision']]
recalls = [cons['recall'], agg['recall']]
f1s = [cons['f1'], agg['f1']]

x_pos = np.arange(len(modes))
width = 0.25
bars_p = ax3.bar(x_pos - width, precisions, width, label='Precision', color='#2ca02c', edgecolor='black', linewidth=0.4)
bars_r = ax3.bar(x_pos, recalls, width, label='Recall', color='#d62728', edgecolor='black', linewidth=0.4)
bars_f = ax3.bar(x_pos + width, f1s, width, label='F1', color='#1f77b4', edgecolor='black', linewidth=0.4)

for bars, vals in [(bars_p, precisions), (bars_r, recalls), (bars_f, f1s)]:
    for bar, val in zip(bars, vals):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=7)

ax3.set_ylabel('Score')
ax3.set_title('(c) Precision-Recall Tradeoff', fontsize=10)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(modes, fontsize=8)
ax3.set_ylim(0, 1.20)
ax3.legend(fontsize=7, loc='upper left')
ax3.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(outdir + 'kg_cleaning_combined.pdf')
plt.close()
print("Saved kg_cleaning_combined.pdf")

print("\nAll improved figures generated!")
