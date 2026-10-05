import numpy as np
import pandas as pd
import pytest

from portfolio_analytics import metrics as m


def test_sharpe_matches_definition(rets):
    r = rets["S0.NS"]
    expected = (r - 0.065 / 252).mean() / (r - 0.065 / 252).std() * np.sqrt(252)
    assert m.sharpe(r, 0.065) == pytest.approx(expected)


def test_cagr_of_constant_growth():
    r = pd.Series(0.0004, index=pd.bdate_range("2020-01-01", periods=504))
    assert m.cagr(r) == pytest.approx(1.0004 ** 252 - 1)


def test_drawdown_known_path():
    # 100 -> 120 -> 90 -> 130: peak 120, trough 90, max drawdown -25%, recovered on day 3
    idx = pd.bdate_range("2024-01-01", periods=3)
    r = pd.Series([0.20, 90 / 120 - 1, 130 / 90 - 1], index=idx)
    d = m.max_drawdown_details(r)
    assert d["max_drawdown"] == pytest.approx(-0.25)
    assert d["peak"] == idx[0] and d["trough"] == idx[1] and d["recovery"] == idx[2]


def test_capm_recovers_beta(rets):
    c = m.capm(rets["S0.NS"], rets["^NSEI"])
    assert c["beta"] == pytest.approx(1.2, abs=0.08)
    assert m.capm(rets["^NSEI"], rets["^NSEI"])["beta"] == pytest.approx(1.0)


def test_daily_rebalance_is_weighted_sum(rets):
    a = rets.drop(columns="^NSEI")
    w = pd.Series(np.linspace(1, 2, a.shape[1]), index=a.columns)
    w = w / w.sum()
    assert np.allclose(m.portfolio_returns(a, w, "D"), a @ w)


def test_buy_and_hold_equals_value_of_initial_holdings(rets):
    a = rets.drop(columns="^NSEI")
    w = pd.Series(1 / a.shape[1], index=a.columns)
    final_value = ((1 + a).prod() * w).sum()
    assert (1 + m.portfolio_returns(a, w, None)).prod() == pytest.approx(final_value)


def test_monthly_rebalance_compounds_monthly_weighted_returns(rets):
    a = rets.drop(columns="^NSEI")
    w = pd.Series(1 / a.shape[1], index=a.columns)
    monthly_assets = (1 + a).groupby(a.index.to_period("M")).prod() - 1
    expected = (1 + monthly_assets @ w).prod()
    assert (1 + m.portfolio_returns(a, w, "M")).prod() == pytest.approx(expected)


def test_calendar_table_shape(rets):
    t = m.calendar_returns(rets["S1.NS"])
    assert list(t.columns) == m.MONTHS + ["Year"]
    year = t.index[1]
    months = t.loc[year, m.MONTHS].dropna()
    assert (1 + months).prod() - 1 == pytest.approx(t.loc[year, "Year"])
