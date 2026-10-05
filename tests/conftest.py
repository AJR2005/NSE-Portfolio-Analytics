import numpy as np
import pandas as pd
import pytest


def synthetic_prices(n_days: int = 1500, seed: int = 0) -> pd.DataFrame:
    """Correlated GBM prices with a market factor, for tests only (never shipped as data)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n_days)
    market = rng.standard_normal(n_days) * 0.011 + 0.0004
    cols = {"^NSEI": market}
    for i, beta in enumerate([1.2, 0.9, 1.0, 0.7, 0.4, 1.1]):
        cols[f"S{i}.NS"] = beta * market + rng.standard_normal(n_days) * 0.012 + 0.0001 * i
    cols["GOLD.NS"] = rng.standard_normal(n_days) * 0.008 + 0.0003
    r = pd.DataFrame(cols, index=idx)
    return 100 * (1 + r).cumprod()


@pytest.fixture(scope="session")
def prices():
    return synthetic_prices()


@pytest.fixture(scope="session")
def rets(prices):
    return prices.pct_change().dropna()
