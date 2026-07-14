"""LLM client — unified interface for OpenAI / Together / local vLLM.

All modules call this client; switching providers only requires changing config.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from kg_scaffold.utils.config import env_or_cfg, load_config

logger = logging.getLogger(__name__)

_OPENAI_AVAILABLE = False
try:
    from openai import OpenAI, APIError, RateLimitError
    _OPENAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    pass


class LLMClient:
    """Thin wrapper around the OpenAI Python SDK (works for OpenAI, Together,
    vLLM, Ollama — any OpenAI-compatible endpoint)."""

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or load_config()
        llm_cfg = cfg["llm"]
        self.provider = llm_cfg["provider"]
        self.model = llm_cfg["model"]
        self.temperature = llm_cfg.get("temperature", 0.2)
        self.max_tokens = llm_cfg.get("max_tokens", 1024)
        self.timeout = llm_cfg.get("timeout", 60)

        if self.provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY", "")
            base_url = os.environ.get("OPENAI_BASE_URL", None)
        elif self.provider == "chatanywhere":
            api_key = os.environ.get("OPENAI_API_KEY", "")
            base_url = "https://api.chatanywhere.tech/v1"
        elif self.provider == "together":
            api_key = os.environ.get("TOGETHER_API_KEY", "")
            base_url = "https://api.together.xyz/v1"
        elif self.provider == "local":
            api_key = os.environ.get("LOCAL_LLM_KEY", "EMPTY")
            base_url = env_or_cfg("LOCAL_LLM_BASE_URL", cfg, "llm", "base_url",
                                  default="http://localhost:8000/v1")
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        self._client = None
        if _OPENAI_AVAILABLE and api_key:
            self._client = OpenAI(api_key=api_key, base_url=base_url,
                                  timeout=self.timeout)
        else:
            logger.warning(
                "openai SDK not installed or no API key — LLMClient will run "
                "in MOCK mode (returns canned responses for dev/CI).")

    @property
    def available(self) -> bool:
        return self._client is not None

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(
            (TimeoutError, ConnectionError) +
            ((APIError, RateLimitError) if _OPENAI_AVAILABLE else ())),
    )
    def complete(self, prompt: str, *, temperature: float | None = None,
                 max_tokens: int | None = None,
                 system: str | None = None) -> str:
        """Single-turn completion. Returns the assistant text."""
        if not self.available:
            return _mock_response(prompt)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens or self.max_tokens,
        )
        return resp.choices[0].message.content.strip()

    def batch_complete(self, prompts: list[str], **kw) -> list[str]:
        """Sequential batch (rate-limit safe). Override for async if needed."""
        return [self.complete(p, **kw) for p in prompts]


def _mock_response(prompt: str) -> str:
    """Deterministic canned responses for dev/CI without API keys.

    Detects which prompt template is being used by keyword matching and returns
    a plausible structured response so the pipeline runs end-to-end.
    """
    p = prompt.lower()
    if "triples:" in p and "extract" in p:
        return ("fish oil | INHIBITS | vasoconstriction\n"
                "Raynaud disease | ASSOCIATED_WITH | vasoconstriction\n")
    if "verdict:" in p and "curator" in p:
        return "VERDICT: supported\nREASON: context confirms the relation.\nCORRECTED: none"
    if "hypothesis generator" in p and "kg_path" in p.lower() or "kg path" in p.lower():
        return ("###\nHYPOTHESIS: fish oil may treat Raynaud disease by inhibiting vasoconstriction.\n"
                "KG_PATH: Raynaud disease|ASSOCIATED_WITH->vasoconstriction|INHIBITS->fish oil\n"
                "NOVELTY: No prior literature links fish oil to Raynaud directly.\n###\n"
                "###\nHYPOTHESIS: magnesium may treat migraine by inhibiting vasoconstriction.\n"
                "KG_PATH: migraine|ASSOCIATED_WITH->vasoconstriction|INHIBITS->magnesium\n"
                "NOVELTY: magnesium-migraine link is unstudied in provided text.\n###\n")
    if "hypothesis generator" in p:
        return ("###\nHYPOTHESIS: fish oil may treat Raynaud disease.\n"
                "NOVELTY: unstudied link.\n###\n")
    if "entailment:" in p:
        return "ENTAILMENT: entailed\nEXPLANATION: the path supports the claim."
    if "select the top" in p and "relations" in p:
        return "TREATS\nINHIBITS\nASSOCIATED_WITH\nAFFECTS\nCAUSES"
    if "output only the chosen" in p:
        return prompt.split("Mention:")[1].split("\n")[0].strip() if "Mention:" in prompt else "unknown"
    return "OK"
