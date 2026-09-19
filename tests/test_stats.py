"""Regression tests for the statistics module.

These encode values verified independently (including by Monte Carlo).
If one of these breaks, a number in Chapter IV is wrong.
"""
import numpy as np
import pytest
from scipy.stats import wilcoxon

from src.stats.nadeau_bengio import (
    nb_variance_factor, se_ratio, detectable_dz, dz_for_power, corrected_t_test)
from src.stats.holm import family_size, holm_threshold, holm_adjust

approx = pytest.approx


def test_nb_variance_factor():
    assert nb_variance_factor(5) == approx(0.45)      # vs naive 0.20


def test_se_ratio():
    assert se_ratio(5) == approx(1.5)


def test_detectable_dz():
    assert detectable_dz(5, 0.05) == approx(1.86, abs=0.01)


def test_power_is_not_naive_scaling():
    """The trap: scaling the ncp instead of the critical value gives 2.52."""
    assert dz_for_power(0.80, 5, 0.05) == approx(2.40, abs=0.02)
    assert dz_for_power(0.80, 5, 0.05) != approx(2.52, abs=0.02)


def test_power_matches_monte_carlo():
    rng = np.random.default_rng(0)
    dz = dz_for_power(0.80, 5, 0.05)
    x = rng.normal(dz, 1.0, size=(200_000, 5))
    t = x.mean(1) / np.sqrt(x.std(1, ddof=1) ** 2 * nb_variance_factor(5))
    from scipy import stats as st
    assert np.mean(np.abs(t) > st.t.ppf(0.975, 4)) == approx(0.80, abs=0.01)


def test_corrected_p_exceeds_naive():
    """The whole point: the correction makes p LARGER, never smaller."""
    from scipy import stats as st
    d = [0.03, 0.041, 0.028, 0.035, 0.039]
    naive = st.ttest_1samp(d, 0).pvalue
    assert corrected_t_test(d).p > naive


def test_family_size():
    assert family_size() == 70                        # 5 x 7 bins x 2 ckpts


def test_holm_threshold():
    assert holm_threshold(70, 0.05) == approx(0.000714, abs=1e-6)


def test_holm_is_less_conservative_than_bonferroni():
    p = [0.001, 0.02, 0.03, 0.9]
    rej, adj = holm_adjust(p, 0.05)
    assert rej[0] and not rej[-1]
    assert all(adj >= np.array(p))


def test_wilcoxon_cannot_reach_alpha_at_n5():
    """With 5 folds the two-sided Wilcoxon floor is 0.0625.
    It is arithmetically incapable of significance at alpha=0.05."""
    assert wilcoxon([1, 2, 3, 4, 5]).pvalue == approx(0.0625)


# --- regression tests added after the nct NaN defect -------------------------
# scipy.stats.nct returns NaN over part of the (critical value, ncp) plane this
# study lands in. The previous dz_for_power bisected directly on nct and
# returned 6.17 at the Bonferroni alpha -- BELOW the 6.31 that reaches only
# ~57% power, i.e. it claimed 80% power at an effect size that cannot deliver
# it. These tests pin the regime the old code got wrong.

BONFERRONI_ALPHA = 0.05 / 70          # family of 70, Section 3.6.4


def test_scipy_nct_is_nan_in_the_regime_we_care_about():
    """Documents WHY power_at_dz does its own quadrature.

    The NaN is in the lower-tail term P(t < -c), true value ~1e-16, and it
    depends on the exact float of c -- which is why the old bug was
    intermittent. If scipy ever fixes this the test fails, and the
    workaround can then be revisited deliberately rather than by accident.
    """
    from scipy import stats as st
    import numpy as np
    from src.stats.nadeau_bengio import _critical_value
    c = _critical_value(5, BONFERRONI_ALPHA)
    assert np.isnan(st.nct.cdf(-c, 4, 7.8 * np.sqrt(5)))
    assert np.isfinite(st.nct.sf(c, 4, 7.8 * np.sqrt(5)))


def test_bonferroni_detectable_dz_matches_manuscript():
    """Section 3.6.4 states d_z > 6.30 to reach significance at alpha=0.0007."""
    assert detectable_dz(5, BONFERRONI_ALPHA) == approx(6.30, abs=0.01)


def test_bonferroni_dz_for_power_matches_manuscript():
    """Section 3.6.4 states d_z ~ 7.8 for 80% power at alpha=0.0007.
    The old nct-bisection implementation returned 6.17 here."""
    got = dz_for_power(0.80, 5, BONFERRONI_ALPHA)
    assert got == approx(7.8, abs=0.1)
    assert got > detectable_dz(5, BONFERRONI_ALPHA)   # the old value was not


def test_power_is_monotone_and_ordered_against_the_threshold():
    from src.stats.nadeau_bengio import power_at_dz
    p = [power_at_dz(d, 5, 0.05) for d in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)]
    assert all(b > a for a, b in zip(p, p[1:]))
    assert power_at_dz(detectable_dz(5, 0.05), 5, 0.05) > 0.5


def test_power_at_dz_reproduces_manuscript_power_curve():
    """Section 3.6.4: power ~0.39 at d_z=1.5 and ~0.17 at d_z=1.0."""
    from src.stats.nadeau_bengio import power_at_dz
    assert power_at_dz(1.0, 5, 0.05) == approx(0.17, abs=0.01)
    assert power_at_dz(1.5, 5, 0.05) == approx(0.39, abs=0.01)
    assert power_at_dz(2.40, 5, 0.05) == approx(0.80, abs=0.01)


def test_power_at_dz_agrees_with_monte_carlo_at_bonferroni_alpha():
    import numpy as np
    from scipy import stats as st
    from src.stats.nadeau_bengio import power_at_dz, nb_variance_factor
    k, dz = 5, 7.8
    crit = st.t.ppf(1 - BONFERRONI_ALPHA / 2, k - 1) * np.sqrt(k * nb_variance_factor(k))
    rng = np.random.default_rng(1)
    x = rng.normal(dz, 1.0, size=(200_000, k))
    t = x.mean(1) / (x.std(1, ddof=1) / np.sqrt(k))
    assert power_at_dz(dz, k, BONFERRONI_ALPHA) == approx(np.mean(np.abs(t) > crit), abs=0.01)
