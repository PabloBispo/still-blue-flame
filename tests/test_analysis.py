import numpy as np

from flame.analysis.extinction import burn_interval
from flame.analysis.stats import agreement, holm, mcnemar_exact, spearman_test


def test_burn_interval_finds_extinction():
    t = np.arange(0, 210.0)
    px = np.where((t >= 4) & (t <= 200), 20000, 0)
    px[100] = 0  # one-sample flicker must not split the burn
    r = burn_interval(t, px)
    assert r.ignition_s == 4 and r.extinction_s == 200 and not r.censored


def test_burn_interval_censored_and_empty():
    t = np.arange(0, 50.0)
    assert burn_interval(t, np.where(t >= 10, 9000, 0)).censored
    assert burn_interval(t, np.zeros(50)).extinction_s is None


def test_spearman_detects_negative_trend():
    rng = np.random.default_rng(1)
    x = np.linspace(14, 22, 27)
    y = 10 - 0.4 * x + rng.normal(0, 0.4, 27)
    r = spearman_test(x, y, alternative="less")
    assert r["rho"] < -0.7 and r["p"] < 0.01 and r["ci95"][1] < 0


def test_agreement_and_holm_and_mcnemar():
    a = agreement([10, 20, 31], [10, 21, 30], tol=1)
    assert a["within_tol"] == 1.0 and abs(a["bias"]) < 1e-9
    assert holm({"h5": 0.01, "h6": 0.04}) == {"h5": True, "h6": True}
    assert holm({"h5": 0.03, "h6": 0.04}) == {"h5": False, "h6": False}
    m = mcnemar_exact([1] * 20 + [0] * 0, [0] * 12 + [1] * 8)
    assert m["a_only"] == 12 and m["p"] < 0.01
