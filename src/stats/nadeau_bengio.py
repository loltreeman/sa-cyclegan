"""Corrected resampled t-test (Nadeau & Bengio 2003) — §3.6.4.

Why this exists: with k-fold CV the per-fold scores are NOT independent.
Each pair of training sets shares (k-2)/k of its data, so the naive paired
t-test underestimates the variance and returns p-values that are too small.
Nadeau-Bengio inflates the variance estimate to compensate.
"""
from dataclasses import dataclass
import numpy as np
from scipy import stats
from scipy.integrate import quad


def nb_variance_factor(k: int) -> float:
    """Variance inflation factor: 1/n + n_test/n_train.

    For k-fold CV, n_test/n_train = (1/k) / ((k-1)/k) = 1/(k-1).
    k=5 -> 1/5 + 1/4 = 0.45, against the naive 1/5 = 0.20.
    """
    if k < 2:
        raise ValueError("k must be >= 2")
    return 1.0 / k + 1.0 / (k - 1)


def se_ratio(k: int) -> float:
    """How much wider the corrected standard error is. k=5 -> 1.5x."""
    return float(np.sqrt(nb_variance_factor(k) / (1.0 / k)))


@dataclass(frozen=True)
class NBResult:
    t: float
    p: float
    df: int
    dz: float
    mean_diff: float


def corrected_t_test(diffs) -> NBResult:
    """Two-sided corrected resampled t-test on per-fold differences."""
    d = np.asarray(diffs, dtype=float)
    k = d.size
    if k < 2:
        raise ValueError("need at least 2 folds")
    s = d.std(ddof=1)
    if s == 0:
        raise ValueError("zero variance across folds")
    mean = d.mean()
    t = mean / np.sqrt(s**2 * nb_variance_factor(k))
    df = k - 1
    p = 2 * stats.t.sf(abs(t), df)
    return NBResult(float(t), float(p), df, float(mean / s), float(mean))


def detectable_dz(k: int = 5, alpha: float = 0.05) -> float:
    """Smallest d_z reaching significance. k=5, alpha=.05 -> 1.86.

    This is the d_z whose noncentrality equals the critical value. Note that
    power there is NOT 0.50 but about 0.57 at k=5: E[sqrt(V/df)] < 1 for
    finite df, so the boundary sits slightly below the centre of the
    alternative distribution. Use power_at_dz if you need the actual number.
    """
    t_crit = stats.t.ppf(1 - alpha / 2, k - 1)
    return float(t_crit * np.sqrt(nb_variance_factor(k)))


def _critical_value(k: int, alpha: float) -> float:
    """Rejection boundary expressed in NAIVE-t units.

    t_NB = t_naive / sqrt(n * factor) exactly, so rejecting |t_NB| > t_crit
    is the same as rejecting |t_naive| > t_crit * sqrt(n * factor). We work
    in naive-t units because that is the statistic whose null and alternative
    distributions are standard (central and noncentral t respectively).
    """
    return float(stats.t.ppf(1 - alpha / 2, k - 1) * np.sqrt(k * nb_variance_factor(k)))


def power_at_dz(dz: float, k: int = 5, alpha: float = 0.05) -> float:
    """Two-sided power of the corrected test at a true effect size d_z.

    Computed by integrating the chi-square denominator out by hand rather
    than by calling scipy.stats.nct.

    The reason is not stylistic. Writing t_naive = (Z + ncp) / sqrt(V/df)
    with Z ~ N(0,1), V ~ chi2(df) independent and ncp = d_z * sqrt(n), the
    two-sided power conditional on V = v is

        P(|t| > c | V=v) = Phi(ncp - c*sqrt(v/df)) + Phi(-ncp - c*sqrt(v/df))

    and we average that over the chi2(df) density of V.

    scipy.stats.nct.cdf returns NaN for the far lower tail in this regime.
    The offending term is P(t < -c), whose true value here is around 1e-16 —
    numerically irrelevant, but NaN poisons the sum it is added to. The
    failure is sensitive to the exact float of c rather than confined to a
    clean region: at df=4, alpha=0.05/70, c=14.097400007074867 it is NaN for
    ncp >= 14.1, while the same call at c=14.0974 returns 5.5e-16. An
    intermittent NaN is worse than a consistent one, because a bisection
    comparing `NaN < power` silently takes the False branch and converges to
    a wrong answer without raising. This quadrature agrees with nct to ~1e-11
    wherever nct is finite, and with Monte Carlo where it is not.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    df = k - 1
    c = _critical_value(k, alpha)
    ncp = float(dz) * np.sqrt(k)

    def integrand(v):
        s = np.sqrt(v / df)
        upper = stats.norm.sf(c * s - ncp)     # P(t >  c | V=v)
        lower = stats.norm.cdf(-c * s - ncp)   # P(t < -c | V=v)
        return (upper + lower) * stats.chi2.pdf(v, df)

    value, _abserr = quad(integrand, 0.0, np.inf, limit=400)
    if not np.isfinite(value):
        raise RuntimeError(f"power quadrature failed at dz={dz}, k={k}, alpha={alpha}")
    return float(np.clip(value, 0.0, 1.0))


def dz_for_power(power: float = 0.80, k: int = 5, alpha: float = 0.05) -> float:
    """d_z giving the requested power under the corrected test.

    NOT the naive 1.5x scaling of the uncorrected threshold, and this is
    subtle enough to get wrong: t_NB = t_naive / sqrt(n*factor) exactly, so
    the rejection region expressed in t_naive terms is |t_naive| > t_crit *
    sqrt(n*factor). You must scale the CRITICAL VALUE, not the noncentrality
    parameter. Scaling the ncp instead reproduces the naive answer, because
    under the alternative the statistic is noncentral t whose dispersion
    grows with the ncp.

    k=5, 80% power, alpha=.05 -> 2.40.  Naive 1.5x scaling wrongly gives
    2.52, which actually delivers 84% power. Verified by Monte Carlo.

    At the Bonferroni alpha of 0.05/70 this returns 7.80, the figure quoted
    in Section 3.6.4. An earlier implementation bisected directly on
    scipy.stats.nct and returned 6.17 there, BELOW the 6.31 that only
    reaches 50% power, because nct returns NaN in that regime and `NaN <
    power` is False. See power_at_dz.
    """
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")

    lo, hi = 1e-6, 30.0
    if power_at_dz(hi, k, alpha) < power:
        raise ValueError(f"requested power {power} not reachable below d_z=30")
    for _ in range(200):
        mid = (lo + hi) / 2
        if power_at_dz(mid, k, alpha) < power:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2)
