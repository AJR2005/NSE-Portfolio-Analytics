import numpy as np
import pandas as pd
import pytest

from portfolio_analytics import metrics as m, risk


@pytest.fixture(scope="module")
def normal_returns():
    rng = np.random.default_rng(7)
    return pd.Series(rng.normal(0, 0.01, 100_000))


def test_methods_agree_on_normal_data(normal_returns):
    # All methods should give z(99%) * sigma ~= 2.326% for normal returns
    for fn in (risk.historical_var, risk.parametric_var, risk.cornish_fisher_var):
        var, es = fn(normal_returns, 0.99)
        assert var == pytest.approx(0.02326, rel=0.03)
        assert es == pytest.approx(0.02665, rel=0.04)  # sigma * pdf(z) / (1 - c)


def test_es_exceeds_var_and_99_exceeds_95(rets):
    r = rets["S0.NS"]
    for fn in (risk.historical_var, risk.parametric_var, risk.cornish_fisher_var):
        v95, e95 = fn(r, 0.95)
        v99, _ = fn(r, 0.99)
        assert e95 > v95 > 0 and v99 > v95


def test_monte_carlo_close_to_parametric(rets):
    a = rets.drop(columns="^NSEI")
    w = pd.Series(1 / a.shape[1], index=a.columns)
    port = a @ w
    mc, _, _ = risk.monte_carlo_var(a, w, 0.99)
    assert mc == pytest.approx(risk.parametric_var(port, 0.99)[0], rel=0.05)


def test_component_var_sums_to_total(rets):
    a = rets.drop(columns="^NSEI")
    w = pd.Series(np.arange(1, a.shape[1] + 1), index=a.columns, dtype=float)
    w /= w.sum()
    rc = risk.risk_contributions(a, w, 0.99)
    sigma = np.sqrt(w @ a.cov() @ w)
    assert rc["Component VaR"].sum() == pytest.approx(2.3263 * sigma, rel=1e-3)
    assert rc["Risk contribution"].sum() == pytest.approx(1.0)


def test_kupiec():
    # Exactly the expected number of exceptions: LR = 0, p = 1
    assert risk._kupiec(1000, 10, 0.01) == pytest.approx((0.0, 1.0))
    # Far too many exceptions: strongly rejected
    assert risk._kupiec(250, 15, 0.01)[1] < 0.001


def test_christoffersen_detects_clustering():
    clustered = np.zeros(1000, dtype=int)
    clustered[500:510] = 1
    spread = np.zeros(1000, dtype=int)
    spread[::100] = 1
    assert risk._christoffersen(clustered)[1] < 0.01
    assert risk._christoffersen(spread)[1] > 0.05


@pytest.mark.parametrize("method,fn", [("historical", risk.historical_var), ("parametric", risk.parametric_var),
                                       ("cornish-fisher", risk.cornish_fisher_var)])
def test_rolling_forecast_uses_only_past_data(rets, method, fn):
    r = rets["S2.NS"]
    f = risk.rolling_var_forecast(r, method, 0.99, 250)
    assert f.index[0] == r.index[250]
    for t in (250, 600, len(r) - 1):
        assert f.loc[r.index[t]] == pytest.approx(fn(r.iloc[t - 250:t], 0.99)[0])


def test_backtest_on_well_specified_model(normal_returns):
    r = pd.Series(normal_returns.to_numpy()[:3000], index=pd.bdate_range("2010-01-01", periods=3000))
    bt = risk.backtest_var(r, "parametric", 0.99, 250)
    assert bt.observations == 2750
    assert bt.kupiec_p > 0.01
    assert bt.traffic_light in {"green", "yellow"}
    assert len(bt.series) == bt.observations and bt.series["Exception"].sum() == bt.exceptions


def test_traffic_light_red_when_model_understates_risk():
    rng = np.random.default_rng(3)
    calm, wild = rng.normal(0, 0.005, 600), rng.normal(0, 0.03, 260)
    r = pd.Series(np.r_[calm, wild], index=pd.bdate_range("2015-01-01", periods=860))
    assert risk.backtest_var(r, "historical", 0.99, 500).traffic_light == "red"
