"""Holm-Bonferroni step-down correction — §3.6.4.

IMPORTANT, read before using this on the study's results. §3.6.4 does NOT
correct across the family of 70. It designates ONE confirmatory comparison
(Condition A vs C, pooled, post-Stage 1, mAP@0.5:0.95), which is a family of
one and is evaluated at alpha=0.05 uncorrected; every other comparison is
labelled exploratory and reported uncorrected with effect sizes and CIs.

family_size() and holm_threshold() exist to reproduce the ARGUMENT in §3.6.4
— that correcting across 70 would demand d_z > 6.30 to reach significance,
which no augmentation intervention plausibly delivers. They are not the
analysis path. Applying holm_adjust to the 70 comparisons would contradict
the frozen methodology.
"""
import numpy as np

N_COMPARISONS, N_SEVERITY_BINS, N_CHECKPOINTS = 5, 7, 2


def family_size() -> int:
    """5 comparisons x 7 severity bins x 2 checkpoints = 70."""
    return N_COMPARISONS * N_SEVERITY_BINS * N_CHECKPOINTS


def holm_threshold(m: int, alpha: float = 0.05) -> float:
    """Threshold the SMALLEST p-value must clear: alpha/m."""
    return alpha / m


def holm_adjust(pvalues, alpha: float = 0.05):
    """Return (reject, adjusted_p) in the caller's original order."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj <= alpha, adj
