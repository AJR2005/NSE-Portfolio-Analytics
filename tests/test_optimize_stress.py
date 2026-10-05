import numpy as np
import pandas as pd
import pytest

from portfolio_analytics import metrics as m, optimize as o, stress


@pytest.fixture(scope="module")
def inputs(rets):
    a = rets.drop(columns="^NSEI")
    return a, o.covariance(a), o.expected_returns(a)


@pytest.mark.parametrize("cap", [0.25, 0.35, 1.0])
def test_weights_valid(inputs, cap):
    a, cov, mu = inputs
    for w in (o.min_variance(cov, cap), o.max_sharpe(mu, cov, 0.065, cap)):
        assert w.sum() == pytest.approx(1.0)
        assert (w >= -1e-9).all() and (w <= cap + 1e-6).all()


def test_min_variance_beats_equal_weight(inputs):
    _, cov, mu = inputs
    eq = pd.Series(1 / len(cov), index=cov.index)
    assert (o.portfolio_stats(o.min_variance(cov, 1.0), mu, cov, 0)["Volatility"]
            < o.portfolio_stats(eq, mu, cov, 0)["Volatility"])


def test_max_sharpe_beats_others(inputs):
    _, cov, mu = inputs
    best = o.portfolio_stats(o.max_sharpe(mu, cov, 0.065, 0.35), mu, cov, 0.065)["Sharpe"]
    for w in (o.min_variance(cov, 0.35), o.risk_parity(cov), pd.Series(1 / len(cov), index=cov.index)):
        assert best >= o.portfolio_stats(w, mu, cov, 0.065)["Sharpe"] - 1e-6


def test_risk_parity_equalises_contributions(inputs):
    _, cov, _ = inputs
    w = o.risk_parity(cov)
    contrib = w * (cov @ w) / (w @ cov @ w)
    assert np.allclose(contrib, 1 / len(w), atol=2e-3)


def test_frontier_monotonic(inputs):
    _, cov, mu = inputs
    f = o.efficient_frontier(mu, cov, 0.35, points=15)
    assert f["Expected return"].is_monotonic_increasing


def test_stress_scenarios_and_beta_shock(rets):
    a, bench = rets.drop(columns="^NSEI"), rets["^NSEI"]
    w = pd.Series(1 / a.shape[1], index=a.columns)
    port = m.portfolio_returns(a, w, "D")
    sc = stress.historical_scenarios(a, port, bench)
    assert set(sc["Scenario"]) <= {s["name"] for s in stress.SCENARIOS} and len(sc) >= 3
    covid = sc[sc["Scenario"] == "COVID-19 crash"].iloc[0]
    window = (bench.index > "2020-02-19") & (bench.index <= "2020-03-23")
    assert covid["Benchmark"] == pytest.approx((1 + bench[window]).prod() - 1)
    table, betas = stress.beta_shock(a, w, bench, shocks=(-0.1,))
    assert table["Expected portfolio move"].iloc[0] == pytest.approx(-0.1 * (w * betas).sum())
