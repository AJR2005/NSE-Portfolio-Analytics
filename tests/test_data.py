import numpy as np
import pandas as pd

from portfolio_analytics.data import clean, remove_bad_ticks


def test_bad_ticks_removed_but_real_crash_kept():
    idx = pd.bdate_range("2020-01-01", periods=60)
    etf = pd.Series(100.0, index=idx)
    etf.iloc[20:22] = 1.0  # Yahoo-style 1/100th glitch for two days
    stock = pd.Series(np.r_[np.full(30, 100.0), np.full(30, 75.0)], index=idx)  # genuine 25% gap down
    fixed, counts = remove_bad_ticks(pd.DataFrame({"ETF": etf, "STOCK": stock}))
    assert counts == {"ETF": 2}
    assert fixed["STOCK"].equals(stock)


def test_clean_drops_holiday_rows_and_sparse_tickers():
    idx = pd.bdate_range("2021-01-01", periods=100)
    df = pd.DataFrame({"A": 100.0, "B": 50.0, "C": 10.0, "NEW": np.nan}, index=idx)
    df.loc[idx[-5:], "NEW"] = 20.0
    df.loc[idx[10], ["A", "B", "NEW"]] = np.nan  # holiday: only one stray quote
    out, dropped, repaired = clean(df)
    assert dropped == ["NEW"] and idx[10] not in out.index and not repaired
