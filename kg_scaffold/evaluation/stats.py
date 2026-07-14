"""Statistical significance tests — bootstrap CI, paired t-test, Wilcoxon."""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


def bootstrap_ci(scores: Sequence[float],
                 n_bootstrap: int = 1000,
                 confidence: float = 0.95,
                 seed: int = 42) -> tuple[float, float, float]:
    """Bootstrap mean with confidence interval.

    Returns (mean, lower, upper).
    """
    scores = np.array(scores)
    if len(scores) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(scores, size=len(scores), replace=True)
        means.append(sample.mean())
    means = np.array(means)
    alpha = (1 - confidence) / 2
    lower = float(np.percentile(means, 100 * alpha))
    upper = float(np.percentile(means, 100 * (1 - alpha)))
    return float(scores.mean()), lower, upper


def paired_ttest(a: Sequence[float], b: Sequence[float]) -> dict:
    """Paired t-test: is method A significantly different from B?"""
    a, b = np.array(a), np.array(b)
    if len(a) != len(b) or len(a) < 2:
        return {"t": 0.0, "p": 1.0, "significant": False}
    t, p = sp_stats.ttest_rel(a, b)
    return {"t": float(t), "p": float(p),
            "significant": bool(p < 0.05)}


def wilcoxon_test(a: Sequence[float], b: Sequence[float]) -> dict:
    """Wilcoxon signed-rank test (non-parametric)."""
    a, b = np.array(a), np.array(b)
    diff = a - b
    if np.all(diff == 0) or len(a) < 2:
        return {"statistic": 0.0, "p": 1.0, "significant": False}
    try:
        stat, p = sp_stats.wilcoxon(a, b)
    except ValueError:
        return {"statistic": 0.0, "p": 1.0, "significant": False}
    return {"statistic": float(stat), "p": float(p),
            "significant": bool(p < 0.05)}


def effect_size_cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Cohen's d effect size."""
    a, b = np.array(a), np.array(b)
    pooled_std = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    if pooled_std == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def compare_methods(scores_a: Sequence[float], scores_b: Sequence[float],
                    name_a: str = "A", name_b: str = "B") -> dict:
    """Full statistical comparison between two methods."""
    mean_a, lo_a, hi_a = bootstrap_ci(scores_a)
    mean_b, lo_b, hi_b = bootstrap_ci(scores_b)
    tt = paired_ttest(scores_a, scores_b)
    wx = wilcoxon_test(scores_a, scores_b)
    d = effect_size_cohens_d(scores_a, scores_b)
    return {
        "a": {"name": name_a, "mean": mean_a, "ci": (lo_a, hi_a)},
        "b": {"name": name_b, "mean": mean_b, "ci": (lo_b, hi_b)},
        "paired_ttest": tt,
        "wilcoxon": wx,
        "cohens_d": d,
        "a_better": mean_a > mean_b,
    }
