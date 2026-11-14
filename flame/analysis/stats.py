"""Small, dependency-light statistics used by the hypothesis tests."""
from __future__ import annotations

import numpy as np
from scipy import stats


def bootstrap_ci(x, y, fn, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI for a paired statistic fn(x, y)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    vals = np.array([fn(x[i], y[i]) for i in idx])
    vals = vals[np.isfinite(vals)]
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def spearman_test(x, y, alternative: str = "two-sided"):
    """Spearman rho with p-value and bootstrap CI. Drops pairs with NaN."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    res = stats.spearmanr(x, y, alternative=alternative)
    lo, hi = bootstrap_ci(x, y, lambda a, b: stats.spearmanr(a, b).statistic)
    return {"n": int(len(x)), "rho": float(res.statistic), "p": float(res.pvalue), "ci95": (lo, hi)}


def agreement(auto, manual, tol: float):
    """Agreement of an automatic measurement against manual reference (Bland-Altman style)."""
    a, m = np.asarray(auto, float), np.asarray(manual, float)
    ok = np.isfinite(a) & np.isfinite(m)
    d = a[ok] - m[ok]
    sd = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
    return {"n": int(len(d)), "mae": float(np.abs(d).mean()), "bias": float(d.mean()),
            "loa": (float(d.mean() - 1.96 * sd), float(d.mean() + 1.96 * sd)),
            "within_tol": float((np.abs(d) <= tol).mean())}


def holm(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm-Bonferroni step-down. Returns reject decision per hypothesis id."""
    order = sorted(pvalues, key=pvalues.get)
    out, still = {}, True
    for k, name in enumerate(order):
        still = still and pvalues[name] <= alpha / (len(order) - k)
        out[name] = still
    return out


def mcnemar_exact(a_correct, b_correct):
    """Exact McNemar test for two methods scored on the same items."""
    a, b = np.asarray(a_correct, bool), np.asarray(b_correct, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    n = n01 + n10
    p = 1.0 if n == 0 else float(stats.binomtest(min(n01, n10), n, 0.5).pvalue)
    return {"a_only": n10, "b_only": n01, "p": p}
