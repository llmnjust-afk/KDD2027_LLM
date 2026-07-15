"""Module C — neural-symbolic hypothesis generation.

The LLM generates ranked, traceable hypotheses conditioned on a KG subgraph
(symbolic scaffold) and optional text snippets.  Each hypothesis carries an
explicit KG_PATH that Module D later verifies.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Sequence

from kg_scaffold.generation.llm_client import LLMClient
from kg_scaffold.retrieval.subgraph import Subgraph
from kg_scaffold.utils.prompts import (
    HYPOTHESIS_GEN_WITH_KG,
    HYPOTHESIS_GEN_NO_KG,
)

logger = logging.getLogger(__name__)

_BLOCK_RE = re.compile(r"###\s*\n(.*?)(?=###\s*\n|$)", re.DOTALL)


@dataclass
class Hypothesis:
    """A single generated hypothesis."""
    text: str
    kg_path: str = ""
    novelty: str = ""
    score: float = 0.0           # plausibility / ranking score
    faithfulness: str = ""      # set by Module D: entailed|partial|none
    source_method: str = ""

    @property
    def target_entity(self) -> str:
        """Extract the candidate substance from the hypothesis text.

        Heuristic: the last chemical-like noun phrase before 'may' or 'treats'.
        """
        m = re.search(r"([A-Za-z][A-Za-z\- ]+?)\s+(?:may|might|could|treats?|prevents?)",
                      self.text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return self.text.strip().split(",")[0]


def generate_hypotheses(
    question: str,
    subgraph: Subgraph | None,
    snippets: Sequence[str] | None = None,
    client: LLMClient | None = None,
    num: int = 10,
    use_kg: bool = True,
) -> list[Hypothesis]:
    """Generate ranked hypotheses.

    Args:
        question: the research question.
        subgraph: retrieved KG subgraph (None disables KG scaffold).
        snippets: optional text snippets (for RAG-style context).
        client: LLM client.
        num: number of hypotheses to request.
        use_kg: whether to include the KG scaffold in the prompt.
    """
    client = client or LLMClient()
    snip_text = "\n".join(f"- {s}" for s in (snippets or [])[:8]) or "(none)"

    if use_kg and subgraph and subgraph.triples:
        kg_text = subgraph.as_text()
        prompt = HYPOTHESIS_GEN_WITH_KG.format(
            question=question,
            kg_subgraph=kg_text,
            snippets=snip_text,
            num=num,
        )
    else:
        prompt = HYPOTHESIS_GEN_NO_KG.format(
            question=question,
            snippets=snip_text,
            num=num,
        )

    try:
        raw = client.complete(prompt, temperature=0.2,
                              max_tokens=min(2048, 256 * num))
    except Exception as exc:
        logger.error("hypothesis generation failed: %s", exc)
        return []

    hypos = parse_hypotheses(raw)
    for i, h in enumerate(hypos):
        h.score = 1.0 / (i + 1)  # rank-based prior
    return hypos[:num]


def parse_hypotheses(raw: str) -> list[Hypothesis]:
    """Parse the LLM output into Hypothesis objects."""
    hypos: list[Hypothesis] = []
    for block in _BLOCK_RE.findall(raw):
        h = Hypothesis(text="", kg_path="", novelty="")
        for line in block.strip().splitlines():
            line = line.strip()
            if line.upper().startswith("HYPOTHESIS:"):
                h.text = line.split(":", 1)[1].strip()
            elif line.upper().startswith("KG_PATH:"):
                h.kg_path = line.split(":", 1)[1].strip()
            elif line.upper().startswith("NOVELTY:"):
                h.novelty = line.split(":", 1)[1].strip()
        if h.text:
            hypos.append(h)
    return hypos
