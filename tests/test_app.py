"""Renders the whole dashboard headlessly on synthetic prices and checks no tab raises."""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    from portfolio_analytics.data import DEFAULT_BASKET

    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2019-01-01", periods=1800)
    mkt = rng.standard_t(5, len(idx)) * 0.008 + 0.0004
    cols = {"^NSEI": mkt, "^NSEBANK": 1.2 * mkt + rng.normal(0, 0.006, len(idx))}
    for t in DEFAULT_BASKET:
        cols[t] = rng.uniform(0.5, 1.3) * mkt + rng.normal(0, 0.01, len(idx))
    path = tmp_path_factory.mktemp("data") / "prices.csv"
    (100 * (1 + pd.DataFrame(cols, index=idx)).cumprod()).to_csv(path, index_label="Date")
    return path


def test_dashboard_renders(snapshot, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("NSE_SNAPSHOT", str(snapshot))
    monkeypatch.chdir(ROOT)
    sys.modules.pop("portfolio_analytics.data", None)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 7
    labels = [mt.label for mt in at.metric]
    assert "CAGR" in labels and "Sharpe ratio" in labels
    # Switching period and rebalancing must also render cleanly
    at.selectbox[1].select("Buy and hold").run()
    assert not at.exception, [e.value for e in at.exception]
