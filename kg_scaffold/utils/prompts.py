"""All LLM prompt templates used across the pipeline.

Kept in one place so prompts are versioned alongside experiments and can be
ablated independently of code.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Module A — KG construction / refinement
# ---------------------------------------------------------------------------

TRIPLE_EXTRACTION = """\
You are a biomedical knowledge extractor. From the abstract below, extract
subject-predicate-object triples that state factual relationships.

Rules:
- Each triple: (subject, predicate, object).
- Predicates must be one of: {predicates}.
- Use canonical entity names; keep them short noun phrases.
- Output one triple per line as: SUBJECT | PREDICATE | OBJECT
- If no clear factual triples exist, output nothing.

Abstract:
{abstract}

Triples:
"""

TRIPLE_VERIFY = """\
You are a biomedical knowledge curator. Decide whether the following triple is
factually supported, contradicted, or unverifiable given the context.

Triple: ({subject}, {predicate}, {object})
Context: {context}

Respond in EXACTLY this format:
VERDICT: <supported|contradicted|unverifiable>
REASON: <one sentence>
CORRECTED: <if contradicted, give the corrected object or "none"; else "none">
"""

# ---------------------------------------------------------------------------
# Module B — retrieval (entity linking)
# ---------------------------------------------------------------------------

ENTITY_LINK = """\
Link the following entity mention to the most likely UMLS concept. Given the
candidate concepts, return the single best CUI.

Mention: {mention}
Candidates: {candidates}

Output ONLY the chosen concept name (no explanation).
"""

# ---------------------------------------------------------------------------
# Module C — neural-symbolic hypothesis generation
# ---------------------------------------------------------------------------

HYPOTHESIS_GEN_WITH_KG = """\
You are a scientific hypothesis generator. Given a research question and a
knowledge-graph subgraph (a set of factual triples), generate novel, testable
hypotheses that connect the question to unstudied relationships.

Research question: {question}

Knowledge-graph subgraph (subject | predicate | object):
{kg_subgraph}

Related literature snippets:
{snippets}

Instructions:
1. Reason step-by-step over the KG paths to find plausible but unstudied links.
2. Generate {num} hypotheses, ranked by plausibility.
3. For EACH hypothesis, provide:
   - HYPOTHESIS: a single declarative sentence.
   - KG_PATH: the chain of triples (A|r1->B|r2->C) that supports it.
   - NOVELTY: why this link is not already in the provided literature.
4. Each hypothesis MUST be traceable to the KG path above it.

Output {num} hypotheses in this exact format:
###
HYPOTHESIS: ...
KG_PATH: ...
NOVELTY: ...
###
"""

HYPOTHESIS_GEN_NO_KG = """\
You are a scientific hypothesis generator. Given a research question and
related literature, generate novel, testable hypotheses.

Research question: {question}

Related literature snippets:
{snippets}

Generate {num} ranked hypotheses. For each:
###
HYPOTHESIS: ...
NOVELTY: ...
###
"""

# ---------------------------------------------------------------------------
# Module D — faithfulness verification
# ---------------------------------------------------------------------------

FAITHFULNESS_CHECK = """\
You are a strict fact-checker. Determine whether the hypothesis is ENTAILED by
the given knowledge-graph path (the path fully supports the claim), PARTIALLY
supported, or NOT supported.

Hypothesis: {hypothesis}
KG path: {kg_path}

Respond in EXACTLY this format:
ENTAILMENT: <entailed|partial|none>
EXPLANATION: <one sentence>
"""

# ---------------------------------------------------------------------------
# ToG-style relation exploration (baseline wrapper uses this)
# ---------------------------------------------------------------------------

RELATION_EXPLORE = """\
Given the topic entity and a set of relations from the KG, select the top-{topk}
relations most likely to lead to an answer.

Topic entity: {entity}
Available relations: {relations}

Output one relation per line, most relevant first.
"""
