"""Streamlit demo for KG-SCoRE.

Run:  streamlit run demo/app.py

Provides an interactive interface to:
  - enter a research question / seed entity
  - retrieve the KG subgraph (visualized)
  - generate KG-scaffolded hypotheses
  - see faithfulness scores per hypothesis
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import streamlit as st

from kg_scaffold.utils.config import load_config, ensure_dirs
from kg_scaffold.utils.semmeddb import Triple, load_or_synthesize
from kg_scaffold.utils.lbd_gold import load_lbd_gold, query_for_pair
from kg_scaffold.kg.completion import CompletionResult, load_scores
from kg_scaffold.generation.llm_client import LLMClient
from kg_scaffold.retrieval.subgraph import SubgraphRetriever
from kg_scaffold.generation.hypothesis_gen import generate_hypotheses
from kg_scaffold.verification.faithfulness import verify_hypotheses
from kg_scaffold.baselines.ours import KGSCoRE
from kg_scaffold.baselines.base import RunConfig


@st.cache_resource
def get_kg():
    cfg = load_config()
    ensure_dirs(cfg)
    semmed_dir = Path(cfg["paths"]["semmeddb_dir"])
    if not semmed_dir.is_absolute():
        semmed_dir = Path(__file__).resolve().parents[1] / semmed_dir
    kg_path = semmed_dir / "kg_triples.json"
    if kg_path.exists():
        with open(kg_path) as fh:
            data = json.load(fh)
        triples = [Triple(d["subject"], d["predicate"], d["object"],
                          score=d.get("score", 1.0)) for d in data]
    else:
        triples = load_or_synthesize(None, seed=42)
    return triples


@st.cache_resource
def get_completion():
    cfg = load_config()
    semmed_dir = Path(cfg["paths"]["semmeddb_dir"])
    if not semmed_dir.is_absolute():
        semmed_dir = Path(__file__).resolve().parents[1] / semmed_dir
    scores_path = semmed_dir / "completion_scores.tsv"
    if scores_path.exists():
        scores = load_scores(scores_path)
        return CompletionResult("complex", len(scores), 0, 0, {}, scores)
    return None


def main():
    st.set_page_config(page_title="KG-SCoRE", page_icon="🔬", layout="wide")
    st.title("🔬 KG-SCoRE")
    st.caption("Knowledge-Graph-Symbolic Co-Refinement for Literature-Based Discovery")

    triples = get_kg()
    completion = get_completion()
    client = LLMClient()

    # sidebar
    with st.sidebar:
        st.header("Settings")
        num_hypos = st.slider("Number of hypotheses", 3, 20, 10)
        use_kg = st.checkbox("Use KG scaffold", value=True)
        use_completion = st.checkbox("Use ComplEx refinement", value=True)
        use_faith = st.checkbox("Run faithfulness check", value=True)
        st.divider()
        st.write(f"**KG triples:** {len(triples)}")
        st.write(f"**ComplEx scores:** {len(completion.scores) if completion else 0}")
        st.write(f"**LLM:** {client.model} ({'online' if client.available else 'mock'})")

    # main input
    st.subheader("Research Question")
    pairs = load_lbd_gold()
    example = st.selectbox(
        "Or pick an LBD example:",
        ["(custom)"] + [f"{p.source} → ? ({p.id})" for p in pairs],
    )

    col1, col2 = st.columns(2)
    with col1:
        seed_entity = st.text_input("Seed entity (disease/topic)", "Raynaud disease")
    with col2:
        if example != "(custom)":
            pair = next(p for p in pairs if f"({p.id})" in example)
            seed_entity = pair.source
            question = query_for_pair(pair)
            st.text_area("Auto-generated question", question, height=80)
        else:
            question = st.text_area(
                "Research question",
                f"What treats {seed_entity}? Propose candidate substances.")

    if st.button("🚀 Generate Hypotheses", type="primary"):
        with st.spinner("Running KG-SCoRE pipeline..."):
            method = KGSCoRE(
                triples=triples, completion=completion, client=client,
                text_corpus=[f"{t.subject} {t.predicate} {t.object}." for t in triples],
            )
            rc = RunConfig(
                num_hypotheses=num_hypos,
                use_kg=use_kg,
                use_completion=use_completion and completion is not None,
                use_faithfulness=use_faith,
            )
            hypos = method.run(question=question, seed_entity=seed_entity, cfg=rc)

        if not hypos:
            st.warning("No hypotheses generated. Check LLM config / API key.")
            return

        # show subgraph
        if use_kg:
            with st.expander(f"📊 Retrieved KG subgraph ({len(method.retriever.retrieve(seed_entity).triples)} triples)",
                             expanded=False):
                sub = method.retriever.retrieve(seed_entity)
                for t in sub.triples[:20]:
                    st.write(f"- `{t.subject}` **{t.predicate}** `{t.object}` (score={t.score:.3f})")

        # show hypotheses
        st.subheader(f"Generated Hypotheses ({len(hypos)})")
        for i, h in enumerate(hypos, 1):
            emoji = {"entailed": "✅", "partial": "🟡", "none": "❌"}.get(h.faithfulness, "⬜")
            with st.expander(f"{emoji} #{i} {h.text[:80]}...", expanded=i <= 3):
                st.write(f"**Hypothesis:** {h.text}")
                if h.kg_path:
                    st.write(f"**KG Path:** `{h.kg_path}`")
                if h.novelty:
                    st.write(f"**Novelty:** {h.novelty}")
                if h.faithfulness:
                    st.write(f"**Faithfulness:** `{h.faithfulness}`")
                st.write(f"**Score:** {h.score:.3f}")


if __name__ == "__main__":
    main()
