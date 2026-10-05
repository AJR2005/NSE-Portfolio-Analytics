"""Long-only portfolio optimisation: minimum variance, maximum Sharpe, risk parity and the efficient frontier."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .metrics import TRADING_DAYS


def covariance(asset_returns: pd.DataFrame, shrink: bool = True) -> pd.DataFrame:
    """Annualised covariance. Ledoit-Wolf shrinkage pulls noisy sample estimates toward a structured
    target, which gives far more stable optimised weights than the raw sample covariance."""
    if shrink:
        from sklearn.covariance import LedoitWolf

        cov = LedoitWolf().fit(asset_returns.to_numpy()).covariance_
    else:
        cov = asset_returns.cov().to_numpy()
    return pd.DataFrame(cov * TRADING_DAYS, index=asset_returns.columns, columns=asset_returns.columns)


def expected_returns(asset_returns: pd.DataFrame) -> pd.Series:
    """Annualised historical mean. Noisy: the optimiser's biggest source of error, so weights are capped."""
    return asset_returns.mean() * TRADING_DAYS


def _solve(objective, n: int, max_weight: float, extra=()) -> np.ndarray:
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}, *extra]
    bounds = [(0.0, max_weight)] * n
    x0 = np.full(n, 1 / n)
    res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-12})
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def min_variance(cov: pd.DataFrame, max_weight: float = 0.35) -> pd.Series:
    S = cov.to_numpy()
    return pd.Series(_solve(lambda w: w @ S @ w, len(S), max_weight), index=cov.index)


def max_sharpe(mu: pd.Series, cov: pd.DataFrame, rf: float, max_weight: float = 0.35) -> pd.Series:
    S, m = cov.to_numpy(), mu.reindex(cov.index).to_numpy()

    def neg_sharpe(w):
        vol = np.sqrt(w @ S @ w)
        return -(w @ m - rf) / vol

    return pd.Series(_solve(neg_sharpe, len(S), max_weight), index=cov.index)


def risk_parity(cov: pd.DataFrame) -> pd.Series:
    """Equal risk contribution: each holding contributes the same share of portfolio volatility."""
    S = cov.to_numpy()
    n = len(S)

    def spread(w):
        port_var = w @ S @ w
        contrib = w * (S @ w) / port_var
        return float(np.sum((contrib - 1 / n) ** 2))

    return pd.Series(_solve(spread, n, 1.0), index=cov.index)


def portfolio_stats(w: pd.Series, mu: pd.Series, cov: pd.DataFrame, rf: float) -> dict:
    w = w.reindex(cov.index).fillna(0)
    ret = float(w @ mu.reindex(cov.index))
    vol = float(np.sqrt(w @ cov @ w))
    return {"Expected return": ret, "Volatility": vol, "Sharpe": (ret - rf) / vol if vol else np.nan}


def efficient_frontier(mu: pd.Series, cov: pd.DataFrame, max_weight: float = 0.35, points: int = 30) -> pd.DataFrame:
    """Lowest achievable volatility for each target return between the min-variance and max-return portfolios."""
    S, m = cov.to_numpy(), mu.reindex(cov.index).to_numpy()
    w_min = min_variance(cov, max_weight).to_numpy()
    lo = float(w_min @ m)
    # Highest return reachable under the weight cap: fill the best assets up to the cap.
    order = np.argsort(-m)
    w_max, left = np.zeros(len(m)), 1.0
    for i in order:
        w_max[i] = min(max_weight, left)
        left -= w_max[i]
        if left <= 1e-12:
            break
    hi = float(w_max @ m)
    rows = []
    for target in np.linspace(lo, hi, points):
        w = _solve(lambda w: w @ S @ w, len(S), max_weight,
                   extra=({"type": "eq", "fun": lambda w, t=target: w @ m - t},))
        rows.append({"Expected return": float(w @ m), "Volatility": float(np.sqrt(w @ S @ w))})
    return pd.DataFrame(rows).drop_duplicates().sort_values("Volatility")
