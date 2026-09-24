"""Binary decision metrics, including selective prediction (automation at an error budget)."""
from __future__ import annotations

import numpy as np


def auroc(y, p):
    y, p = np.asarray(y, int), np.asarray(p, float)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann-Whitney U with ties counted as 1/2
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def ece(y, p, bins=10):
    y, p = np.asarray(y, float), np.asarray(p, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def coverage_at_risk(y, p, budget):
    """Largest share of cases we can auto-decide (most confident first) with error rate <= budget.
    Confidence = max(p, 1-p); decision = p >= 0.5."""
    y, p = np.asarray(y, int), np.asarray(p, float)
    conf = np.maximum(p, 1 - p)
    order = np.argsort(-conf, kind="stable")
    err = ((p[order] >= 0.5).astype(int) != y[order]).astype(float)
    cum = np.cumsum(err) / np.arange(1, len(y) + 1)
    ok = np.where(cum <= budget)[0]
    return float((ok.max() + 1) / len(y)) if len(ok) else 0.0


def summary(y, p, thr=0.5):
    y, p = np.asarray(y, int), np.asarray(p, float)
    pred = (p >= thr).astype(int)
    tp, tn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    fp, fn = int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return {
        "n": int(len(y)), "prevalence": float(y.mean()),
        "accuracy": float((pred == y).mean()), "balanced_accuracy": (sens + spec) / 2,
        "sensitivity": sens, "specificity": spec, "auroc": auroc(y, p),
        "brier": float(((p - y) ** 2).mean()), "ece": ece(y, p),
        "cov@5%": coverage_at_risk(y, p, 0.05), "cov@10%": coverage_at_risk(y, p, 0.10),
    }


def bootstrap_ci(y, p, fn, n=1000, seed=0):
    r = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        i = r.integers(0, len(y), len(y))
        vals.append(fn(y[i], p[i]))
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))
