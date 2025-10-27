
from __future__ import annotations
from typing import List, Tuple
from scipy.stats import binomtest, chi2
import numpy as np

def mcnemar_test(y_true: List[str], y_pred_a: List[str], y_pred_b: List[str]) -> Tuple[float, int, int]:
    n01 = sum((ya == yt) and (yb != yt) for yt, ya, yb in zip(y_true, y_pred_a, y_pred_b))
    n10 = sum((ya != yt) and (yb == yt) for yt, ya, yb in zip(y_true, y_pred_a, y_pred_b))
    n = n01 + n10
    if n <= 25:
        p = binomtest(k=min(n01, n10), n=n, p=0.5, alternative='two-sided').pvalue if n > 0 else 1.0
    else:
        stat = (abs(n01 - n10) - 1) ** 2 / max(1e-12, (n01 + n10))
        p = chi2.sf(stat, df=1)
    return float(p), n01, n10

def bh_correction(pvals: List[float]) -> List[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    sorted_p = np.array(pvals)[order]
    q = np.empty(m, dtype=float)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        q[i] = min(prev, sorted_p[i] * m / rank)
        prev = q[i]
    q_final = np.empty(m, dtype=float)
    q_final[order] = q
    return q_final.tolist()
