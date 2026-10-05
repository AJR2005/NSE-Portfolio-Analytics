"""Value at Risk, Expected Shortfall, risk contributions and VaR backtesting.

Sign convention: VaR and ES are reported as positive numbers meaning a LOSS, as a fraction of
portfolio value. "95% 1-day VaR = 1.8%" means: on 95% of days the loss should not exceed 1.8%.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


def horizon_returns(r: pd.Series, horizon: int) -> pd.Series:
    """Overlapping compounded h-day returns, for historical VaR over more than one day."""
    if horizon == 1:
        return r
    return ((1 + r).rolling(horizon).apply(np.prod, raw=True) - 1).dropna()


def historical_var(r: pd.Series, confidence: float = 0.95, horizon: int = 1) -> tuple[float, float]:
    """Empirical quantile of past returns: no distribution assumed, so fat tails are kept."""
    rh = horizon_returns(r, horizon)
    var = -float(np.quantile(rh, 1 - confidence))
    tail = rh[rh <= -var]
    return var, -float(tail.mean())


def parametric_var(r: pd.Series, confidence: float = 0.95, horizon: int = 1) -> tuple[float, float]:
    """Variance-covariance (normal) VaR, scaled by the square-root-of-time rule."""
    mu, sigma = r.mean() * horizon, r.std(ddof=1) * np.sqrt(horizon)
    z = stats.norm.ppf(1 - confidence)
    var = -(mu + z * sigma)
    es = -(mu - sigma * stats.norm.pdf(z) / (1 - confidence))
    return float(var), float(es)


def _cornish_fisher_z(p: float, skew: float, kurt: float) -> float:
    z = stats.norm.ppf(p)
    return (z + (z**2 - 1) * skew / 6 + (z**3 - 3 * z) * kurt / 24
            - (2 * z**3 - 5 * z) * skew**2 / 36)


def cornish_fisher_var(r: pd.Series, confidence: float = 0.95, horizon: int = 1) -> tuple[float, float]:
    """Normal VaR adjusted for the skewness and excess kurtosis actually observed in the returns."""
    mu, sigma = r.mean() * horizon, r.std(ddof=1) * np.sqrt(horizon)
    s, k = float(r.skew()), float(r.kurt())
    var = -(mu + _cornish_fisher_z(1 - confidence, s, k) * sigma)
    # ES = average VaR over all tail probabilities beyond the confidence level.
    ps = np.linspace(1e-4, 1 - confidence, 200)
    es = -float(np.mean([mu + _cornish_fisher_z(p, s, k) * sigma for p in ps]))
    return float(var), es


def monte_carlo_var(asset_returns: pd.DataFrame, weights: pd.Series, confidence: float = 0.95,
                    horizon: int = 1, n_sims: int = 50_000, seed: int = 42) -> tuple[float, float, np.ndarray]:
    """Simulates correlated asset returns (multivariate normal via Cholesky) and revalues the portfolio."""
    w = weights.reindex(asset_returns.columns).fillna(0).to_numpy()
    mu = asset_returns.mean().to_numpy() * horizon
    cov = asset_returns.cov().to_numpy() * horizon
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(cov + np.eye(len(w)) * 1e-12)
    sims = mu + rng.standard_normal((n_sims, len(w))) @ L.T
    port = sims @ w
    var = -float(np.quantile(port, 1 - confidence))
    es = -float(port[port <= -var].mean())
    return var, es, port


def var_table(port_r: pd.Series, asset_returns: pd.DataFrame, weights: pd.Series,
              confidences=(0.95, 0.99), horizon: int = 1) -> pd.DataFrame:
    rows = []
    for c in confidences:
        for name, (v, e) in {
            "Historical": historical_var(port_r, c, horizon),
            "Parametric (normal)": parametric_var(port_r, c, horizon),
            "Cornish-Fisher": cornish_fisher_var(port_r, c, horizon),
            "Monte Carlo": monte_carlo_var(asset_returns, weights, c, horizon)[:2],
        }.items():
            rows.append({"Method": name, "Confidence": c, "VaR": v, "Expected Shortfall": e})
    return pd.DataFrame(rows)


def risk_contributions(asset_returns: pd.DataFrame, weights: pd.Series, confidence: float = 0.95) -> pd.DataFrame:
    """Euler decomposition of parametric VaR: how much of the portfolio's risk each holding adds.

    Component VaR_i = w_i * (Sigma w)_i / sigma_p * z, and the components sum to total VaR (mean ignored).
    A holding with 10% weight can carry 20% of the risk if it is volatile and correlated with the rest.
    """
    w = weights.reindex(asset_returns.columns).fillna(0)
    cov = asset_returns.cov()
    sigma_p = float(np.sqrt(w @ cov @ w))
    z = -stats.norm.ppf(1 - confidence)
    marginal = (cov @ w) / sigma_p * z
    component = w * marginal
    return pd.DataFrame({
        "Weight": w,
        "Marginal VaR": marginal,
        "Component VaR": component,
        "Risk contribution": component / component.sum(),
    }).sort_values("Risk contribution", ascending=False)


@dataclass
class BacktestResult:
    method: str
    confidence: float
    window: int
    observations: int
    exceptions: int
    expected: float
    exception_rate: float
    kupiec_lr: float
    kupiec_p: float
    christoffersen_lr: float
    christoffersen_p: float
    traffic_light: str | None
    series: pd.DataFrame  # date, return, VaR forecast, exception flag


def _kupiec(n: int, x: int, p: float) -> tuple[float, float]:
    """Proportion-of-failures test: is the exception count consistent with the VaR level?"""
    if n == 0:
        return np.nan, np.nan
    phat = x / n

    def ll(q):
        # log-likelihood of x exceptions in n days with exception probability q (0*log0 = 0)
        a = (n - x) * np.log(1 - q) if q < 1 else (0 if x == n else -np.inf)
        b = x * np.log(q) if q > 0 else (0 if x == 0 else -np.inf)
        return a + b

    lr = -2 * (ll(p) - ll(phat))
    return float(lr), float(1 - stats.chi2.cdf(lr, 1))


def _christoffersen(hits: np.ndarray) -> tuple[float, float]:
    """Independence test: do exceptions cluster (a breach today making one tomorrow more likely)?"""
    h0, h1 = hits[:-1], hits[1:]
    n00 = int(np.sum((h0 == 0) & (h1 == 0)))
    n01 = int(np.sum((h0 == 0) & (h1 == 1)))
    n10 = int(np.sum((h0 == 1) & (h1 == 0)))
    n11 = int(np.sum((h0 == 1) & (h1 == 1)))
    if n01 + n11 == 0:
        return 0.0, 1.0
    p01 = n01 / (n00 + n01) if n00 + n01 else 0
    p11 = n11 / (n10 + n11) if n10 + n11 else 0
    p = (n01 + n11) / (n00 + n01 + n10 + n11)

    def xlogy(a, b):
        return a * np.log(b) if a > 0 else 0.0

    l0 = xlogy(n00 + n10, 1 - p) + xlogy(n01 + n11, p)
    l1 = xlogy(n00, 1 - p01) + xlogy(n01, p01) + xlogy(n10, 1 - p11) + xlogy(n11, p11)
    lr = -2 * (l0 - l1)
    return float(lr), float(1 - stats.chi2.cdf(lr, 1))


def rolling_var_forecast(r: pd.Series, method: str = "historical", confidence: float = 0.99,
                         window: int = 250) -> pd.Series:
    """Each day's VaR forecast (positive loss) from only the previous `window` days of returns.
    Vectorised with rolling windows; shifted by one day so a forecast never sees its own day."""
    roll = r.rolling(window)
    p = 1 - confidence
    if method == "historical":
        q = roll.quantile(p, interpolation="linear")
    else:
        mu, sigma = roll.mean(), roll.std(ddof=1)
        if method == "parametric":
            z = stats.norm.ppf(p)
        elif method == "cornish-fisher":
            z = _cornish_fisher_z(p, roll.skew(), roll.kurt())
        else:
            raise ValueError(f"unknown method {method!r}")
        q = mu + z * sigma
    return (-q).shift(1).iloc[window:]


def backtest_var(r: pd.Series, method: str = "historical", confidence: float = 0.99, window: int = 250) -> BacktestResult:
    """Rolling out-of-sample VaR backtest.

    Each day's VaR is estimated from only the previous `window` days, then compared with that day's
    actual return. A day losing more than the forecast is an exception.
    """
    forecasts = rolling_var_forecast(r, method, confidence, window).to_numpy()
    actual = r.to_numpy()[window:]
    hits = (actual < -forecasts).astype(int)
    n, x = len(hits), int(hits.sum())
    p = 1 - confidence
    k_lr, k_p = _kupiec(n, x, p)
    c_lr, c_p = _christoffersen(hits)
    light = None
    if abs(confidence - 0.99) < 1e-9 and n >= 250:
        last = int(hits[-250:].sum())
        light = "green" if last <= 4 else "yellow" if last <= 9 else "red"  # Basel traffic-light zones
    series = pd.DataFrame({"Return": actual, "VaR": -forecasts, "Exception": hits.astype(bool)},
                          index=r.index[window:])
    return BacktestResult(method, confidence, window, n, x, n * p, x / n if n else np.nan,
                          k_lr, k_p, c_lr, c_p, light, series)
