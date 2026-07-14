"""Dense embedding index for RAG baselines and entity retrieval.

Wraps sentence-transformers / FlagEmbedding to build a FAISS-free numpy index
(small enough for our datasets).  Used by:
  - Dense RAG baseline
  - Entity linking fallback (when no exact KG match)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

_ST_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:  # pragma: no cover
    pass


class DenseIndex:
    """Embedding index over text passages / entity labels."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self._model = None
        self._texts: list[str] = []
        self._emb: np.ndarray | None = None

    def _ensure_model(self):
        if self._model is not None:
            return
        if _ST_AVAILABLE:
            device = "cuda" if _torch_cuda() else "cpu"
            self._model = SentenceTransformer(self.model_name, device=device)
        else:
            logger.warning("sentence-transformers not available — using "
                           "hashing-based fake embeddings (dev only).")

    def build(self, texts: Sequence[str], batch_size: int = 64) -> None:
        """Embed a list of texts and store them."""
        self._ensure_model()
        self._texts = list(texts)
        if self._model is not None:
            self._emb = self._model.encode(
                self._texts, batch_size=batch_size,
                show_progress_bar=False, convert_to_numpy=True,
                normalize_embeddings=True,
            )
        else:
            self._emb = _hash_embed(self._texts, dim=384)

    def search(self, query: str, topk: int = 10) -> list[tuple[str, float]]:
        """Return top-k (text, score) pairs for the query."""
        if self._emb is None or not self._texts:
            return []
        if self._model is not None:
            q = self._model.encode([query], convert_to_numpy=True,
                                   normalize_embeddings=True)
        else:
            q = _hash_embed([query], dim=self._emb.shape[1])
        scores = (self._emb @ q.T).flatten()
        top_idx = np.argsort(scores)[::-1][:topk]
        return [(self._texts[i], float(scores[i])) for i in top_idx]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, texts=np.array(self._texts, dtype=object),
                 emb=self._emb)
        logger.info("saved dense index (%d items) to %s", len(self._texts), path)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        data = np.load(path, allow_pickle=True)
        self._texts = list(data["texts"])
        self._emb = data["emb"]
        logger.info("loaded dense index (%d items)", len(self._texts))


def _torch_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _hash_embed(texts: list[str], dim: int = 384) -> np.ndarray:
    """Deterministic hashing embedding fallback (bag of character n-grams)."""
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, txt in enumerate(texts):
        for gram in _ngrams(txt.lower(), 3):
            h = hash(gram) % dim
            out[i, h] += 1.0
    # normalize
    norms = np.linalg.norm(out, axis=1, keepdims=True) + 1e-8
    out = out / norms
    return out


def _ngrams(s: str, n: int = 3):
    s = f"^{s}$"
    return [s[i:i + n] for i in range(max(0, len(s) - n + 1))]
