# KG-SCoRE: Knowledge-Graph-Symbolic Co-Refinement for Literature-Based Discovery

Submission target: **SIGKDD 2027 — Applied Data Science (ADS) Track**
Theme: *Modern AI and Big Data — AI's role in knowledge graph construction,
neural-symbolic integration, and automated insight generation from massive data.*

## Overview

KG-SCoRE is a four-module pipeline that pairs a large language model (LLM) with
a symbolic knowledge graph (KG) to generate **traceable, novel scientific
hypotheses** from massive text repositories (PubMed / SemMedDB, 30M+ abstracts,
33M+ triples).

```
PubMed corpus (massive text repository)
   |
   +--[A] KG construction & co-refinement
   |      LLM triple extraction  +  LLM cleaning  +  ComplEx learned scoring
   |
   +--[B] Symbolic multi-hop subgraph retrieval
   |
   +--[C] Neural-symbolic hypothesis generation  (LLM scaffolded by KG paths)
   |
   +--[D] Faithfulness verification  (hypothesis -> KG-path entailment)
```

### Innovation points

1. **KG co-refinement** (Module A): a learned KG-completion model (ComplEx via
   PyKEEN) scores every triple; low-confidence triples enter an LLM
   verification queue that retains / deletes / corrects them. This closes the
   loop between neural scoring and symbolic cleaning.
2. **Symbolic-scaffolded generation** (Module C): the LLM generates hypotheses
   conditioned on a retrieved multi-hop KG subgraph, not free-form text.
3. **Faithfulness metric** (Module D): every generated hypothesis is aligned to
   an explicit KG path; `faithfulness@k` measures the fraction whose claims are
   entailed by the KG — a novel, human-validated metric.
4. **Literature-Based Discovery (LBD) re-discovery**: evaluated on the Swanson /
   Cameron gold-standard hypothesis pairs.

## Repo layout

```
kg_scaffold/
  kg/            Module A — extraction, refinement, ComplEx completion
  retrieval/     Module B — multi-hop subgraph + dense index
  generation/    Module C — LLM client, hypothesis generation
  verification/  Module D — faithfulness scoring
  baselines/     vanilla-LLM, BM25-RAG, dense-RAG, ToG wrapper
  evaluation/    LBD metrics, statistical tests
  utils/         SemMedDB loader, prompts
scripts/         end-to-end pipelines (01..07)
demo/            Streamlit app
```

## Quick start

```bash
pip install -r requirements.txt
python scripts/01_build_kg.py --sample          # build KG (sample mode, no full download)
python scripts/02_train_completion.py           # train ComplEx (1 GPU, ~6-8h full)
python scripts/03_run_baseline.py --method bm25_rag
python scripts/04_run_ours.py                   # KG-SCoRE
python scripts/05_run_ablation.py               # ablation matrix
python scripts/06_eval.py                       # metrics + significance
streamlit run demo/app.py                       # interactive demo
```

## Hardware

- 1 x RTX 5090 / A100 (ComplEx training + dense embedding index)
- CPU + 200GB SSD for SemMedDB subgraph extraction
- LLM via API (OpenAI / Together) or local vLLM

## License

MIT (research code). SemMedDB is governed by the UMLS Metathesaurus license.
