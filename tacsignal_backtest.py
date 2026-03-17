#!/usr/bin/env python3
"""
TacSignal Backtest Engine
=========================
Runs the TacSignal framework historically (2010–2025) and measures whether
OW/UW signals predicted subsequent outperformance.

Tests 3 portfolio strategies:
  1. SAA-only (static 60/40-like allocation, no tilts)
  2. SAA + TacSignal tilts (tactical overlay)
  3. Benchmarks: 60/40, 80/20, 100% equity

Outputs:
  - tacsignal-backtest.json (for dashboard import)
  - Console summary with Sharpe, returns, drawdowns, hit rates

Requirements:
    pip install yfinance pandas numpy fredapi

Usage:
    python3 tacsignal_backtest.py --fred-key YOUR_KEY
    python3 tacsignal_backtest.py --fred-key YOUR_KEY --start 2012 --end 2025
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
# ASSET UNIVERSE
# ═══════════════════════════════════════════════════════════════

# Full universe
ASSET_UNIVERSE = {
    "Equity U.S.":   {"price": "SPY",  "type": "equity",    "group": "core"},
    "Equity Intl":   {"price": "EFA",  "type": "equity",    "group": "core"},
    "Equity EM":     {"price": "EEM",  "type": "equity",    "group": "core"},
    "Bond U.S.":     {"price": "AGG",  "type": "bond",      "group": "core"},
    "Bond Intl":     {"price": "BWX",  "type": "bond",      "group": "core"},
    "Gold":          {"price": "GLD",  "type": "commodity",  "group": "alt"},
    "Commodities":   {"price": "DBC",  "type": "commodity",  "group": "alt"},
    "REITs":         {"price": "VNQ",  "type": "equity",    "group": "alt"},
}

# Core = equity + bonds only (for pure signal validation vs benchmarks)
CORE_ASSETS = {k: v for k, v in ASSET_UNIVERSE.items() if v["group"] == "core"}
ALT_ASSETS = {k: v for k, v in ASSET_UNIVERSE.items() if v["group"] == "alt"}

# ═══════════════════════════════════════════════════════════════
# SAA PROFILES — Core (equity+bond only) and Full (with alts)
# ═══════════════════════════════════════════════════════════════

# Core profiles: equity + bonds = 100%, directly comparable to AOR/AOA
CORE_SAA = {
    "Conservative 40/60": {
        "Equity U.S.": 20, "Equity Intl": 12, "Equity EM": 8,
        "Bond U.S.": 45, "Bond Intl": 15,
    },
    "Balanced 60/40": {
        "Equity U.S.": 30, "Equity Intl": 18, "Equity EM": 12,
        "Bond U.S.": 30, "Bond Intl": 10,
    },
    "Growth 80/20": {
        "Equity U.S.": 40, "Equity Intl": 24, "Equity EM": 16,
        "Bond U.S.": 15, "Bond Intl": 5,
    },
}

# Full profiles: with alternatives
FULL_SAA = {
    "Full Conservative": {
        "Equity U.S.": 18, "Equity Intl": 10, "Equity EM": 7,
        "Bond U.S.": 35, "Bond Intl": 10,
        "Gold": 8, "Commodities": 4, "REITs": 8,
    },
    "Full Balanced": {
        "Equity U.S.": 25, "Equity Intl": 15, "Equity EM": 10,
        "Bond U.S.": 20, "Bond Intl": 5,
        "Gold": 8, "Commodities": 4, "REITs": 13,
    },
    "Full Growth": {
        "Equity U.S.": 30, "Equity Intl": 20, "Equity EM": 12,
        "Bond U.S.": 8, "Bond Intl": 2,
        "Gold": 8, "Commodities": 5, "REITs": 15,
    },
}

# For backward compat
DEFAULT_SAA = "Balanced 60/40"
BACKTEST_ASSETS = {name: {**config, "saa": CORE_SAA[DEFAULT_SAA].get(name, 0)}
                   for name, config in ASSET_UNIVERSE.items()}

# FRED series
FRED_SERIES = {
    "US_10Y":       "DGS10",
    "US_2Y":        "DGS2",
    "VIX":          "VIXCLS",
    "US_HY_SPREAD": "BAMLH0A0HYM2",
}

# Real-world fund benchmarks
BENCHMARK_ETFS = {
    "iShares 60/40 (AOR)":   {"ticker": "AOR"},
    "iShares 80/20 (AOA)":   {"ticker": "AOA"},
    "Global Equity (VT)":    {"ticker": "VT"},
}

# Tilt size
TILT_PCT = 5.0

# Sub-periods for analysis
SUB_PERIODS = [
    ("Full Period", 2012, 2025),
    ("Pre-2018",    2012, 2017),
    ("Post-2018",   2018, 2025),
]


# ═══════════════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════════════

def fetch_prices(start_year=2007):
    """Fetch daily prices for all backtest assets."""
    import yfinance as yf

    tickers = list(set(v["price"] for v in ASSET_UNIVERSE.values()))
    # Also fetch benchmark ETFs
    benchmark_tickers = set()
    for bm in BENCHMARK_ETFS.values():
        if "ticker" in bm:
            benchmark_tickers.add(bm["ticker"])
        if "tickers" in bm:
            benchmark_tickers.update(bm["tickers"].keys())
    tickers = list(set(tickers) | benchmark_tickers)
    start = f"{start_year}-01-01"

    print(f"  Fetching {len(tickers)} tickers from {start}...")

    prices = None
    try:
        data = yf.download(tickers, start=start, auto_adjust=True, progress=False)
        if data is not None and len(data) > 0:
            if isinstance(data.columns, pd.MultiIndex):
                prices = data['Close']
            else:
                prices = data[['Close']].copy()
                prices.columns = tickers
    except Exception as e:
        print(f"  Bulk download failed: {e}")

    # Fallback: individual
    if prices is None or prices.isna().all().all():
        print("  Falling back to individual downloads...")
        series_dict = {}
        for ticker in tickers:
            try:
                t = yf.Ticker(ticker)
                hist = t.history(start=start, auto_adjust=True)
                if hist is not None and len(hist) > 50:
                    series_dict[ticker] = hist['Close']
                    print(f"    ✓ {ticker}: {len(hist)} days")
            except Exception as e:
                print(f"    ✗ {ticker}: {e}")
        prices = pd.DataFrame(series_dict)

    prices = prices.ffill().dropna(how='all')
    print(f"  Got {len(prices)} daily observations")
    return prices


def fetch_fred(fred_key, start_year=2007):
    """Fetch FRED data."""
    if not fred_key:
        print("  No FRED key — skipping")
        return pd.DataFrame()
    try:
        from fredapi import Fred
        fred = Fred(api_key=fred_key)
    except ImportError:
        print("  fredapi not installed — skipping")
        return pd.DataFrame()

    start = f"{start_year}-01-01"
    result = {}
    for name, sid in FRED_SERIES.items():
        try:
            s = fred.get_series(sid, observation_start=start)
            result[name] = s
            print(f"    ✓ {name}: {len(s)} obs")
        except Exception as e:
            print(f"    ✗ {name}: {e}")

    if result:
        df = pd.DataFrame(result).ffill()
        return df
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════
# SIGNAL COMPUTATION (same as pipeline, but returns full history)
# ═══════════════════════════════════════════════════════════════

def compute_tech_zscore(prices_series, window=24):
    """Compute monthly technical z-score series."""
    if prices_series is None or len(prices_series) < 300:
        return None

    p = prices_series.dropna()
    sma_50 = p.rolling(50).mean()
    sma_200 = p.rolling(200).mean()

    monthly_price = p.resample('ME').last()
    monthly_sma50 = sma_50.resample('ME').last()
    monthly_sma200 = sma_200.resample('ME').last()

    trend_position = (monthly_price / monthly_sma200) - 1
    ret_12m = monthly_price.pct_change(12)
    ret_1m = monthly_price.pct_change(1)
    mom_12_1 = ret_12m - ret_1m
    crossover = (monthly_sma50 / monthly_sma200) - 1

    raw_tech = 0.5 * trend_position + 0.3 * mom_12_1 + 0.2 * crossover
    raw_tech = raw_tech.dropna()

    if len(raw_tech) < window:
        return None

    mu = raw_tech.rolling(window, min_periods=12).mean()
    sigma = raw_tech.rolling(window, min_periods=12).std()
    z = ((raw_tech - mu) / sigma).clip(-4, 4)
    return z.dropna()


def compute_funda_zscore(prices_series, asset_type, fred_data, window=24, enhanced=False):
    """Compute monthly fundamental z-score series.
    Enhanced mode adds: credit spreads for equities, yield curve for bonds.
    """
    if prices_series is None or len(prices_series) < 300:
        return None

    p = prices_series.dropna()
    monthly_price = p.resample('ME').last()
    log_price = np.log(monthly_price)

    if asset_type in ("equity", "sector"):
        trend = log_price.rolling(24, min_periods=18).mean()
        value_signal = -(log_price - trend)

        components = {'value': value_signal}
        weights = {'value': 0.7, 'vix': 0.3}

        if fred_data is not None and 'VIX' in fred_data.columns:
            components['vix'] = -fred_data['VIX'].resample('ME').last().dropna()

        # Enhanced: add credit spread as a confirmation signal (doesn't dilute, adds)
        if enhanced and fred_data is not None and 'US_HY_SPREAD' in fred_data.columns:
            components['credit'] = -fred_data['US_HY_SPREAD'].resample('ME').last().dropna()
            weights = {'value': 0.55, 'vix': 0.25, 'credit': 0.20}

        combined = pd.DataFrame(components).dropna()
        if len(combined) > 12:
            for col in combined.columns:
                mu = combined[col].rolling(24, min_periods=12).mean()
                sigma = combined[col].rolling(24, min_periods=12).std()
                combined[col] = (combined[col] - mu) / sigma
            combined = combined.dropna()
            if len(combined) > 6:
                raw = sum(weights.get(c, 0) * combined[c] for c in combined.columns if c in weights)
            else:
                raw = value_signal.dropna()
        else:
            raw = value_signal.dropna()

    elif asset_type == "bond":
        if fred_data is not None and 'US_10Y' in fred_data.columns:
            raw_yield = fred_data['US_10Y'].resample('ME').last().dropna()
            # Enhanced: add yield curve slope (steeper = better for bonds when it normalizes)
            if enhanced and 'US_2Y' in fred_data.columns:
                slope = (fred_data['US_10Y'] - fred_data['US_2Y']).resample('ME').last().dropna()
                combined = pd.DataFrame({'yield': raw_yield, 'slope': slope}).dropna()
                if len(combined) > 12:
                    for col in combined.columns:
                        mu = combined[col].rolling(24, min_periods=12).mean()
                        sigma = combined[col].rolling(24, min_periods=12).std()
                        combined[col] = (combined[col] - mu) / sigma
                    combined = combined.dropna()
                    raw = 0.6 * combined['yield'] + 0.4 * combined['slope']
                else:
                    raw = raw_yield
            else:
                raw = raw_yield
        else:
            trend = log_price.rolling(24, min_periods=12).mean()
            raw = -(log_price - trend).dropna()

    elif asset_type == "commodity":
        trend = log_price.rolling(24, min_periods=12).mean()
        raw = -(log_price - trend).dropna()
    else:
        return None

    if raw is None or len(raw) < window:
        return None

    mu = raw.rolling(window, min_periods=12).mean()
    sigma = raw.rolling(window, min_periods=12).std()
    z = ((raw - mu) / sigma).clip(-4, 4)
    return z.dropna()


def compute_wtech_series(prices_series, enhanced=False):
    """Compute monthly w_tech series.
    Original: binary 0 or 100
    Enhanced: continuous 0-100 based on momentum percentile rank
    """
    if prices_series is None or len(prices_series) < 300:
        return None

    monthly = prices_series.resample('ME').last()
    abs_mom = monthly.pct_change(12).abs()

    if enhanced:
        # Smoothed binary w_tech: use 3-month average of binary signals
        # This creates transitions (33%, 67%) instead of instant 0→100 flips
        median_mom = abs_mom.rolling(24, min_periods=12).median()
        binary = (abs_mom > median_mom).astype(float) * 100.0
        w = binary.rolling(3, min_periods=1).mean()  # Smooth over 3 months
    else:
        # Original binary
        median_mom = abs_mom.rolling(24, min_periods=12).median()
        w = (abs_mom > median_mom).astype(float) * 100.0

    return w.dropna()


# ═══════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════

def build_signals(prices, fred_data, window=24, enhanced=False):
    """Build monthly signal history for each asset."""
    signal_data = {}
    mode = "Enhanced" if enhanced else "Original"
    # Enhanced uses slightly lower thresholds since smoothed w_tech
    # produces more moderate composites
    ow_threshold = 0.8 if enhanced else 1.0
    uw_threshold = -0.8 if enhanced else -1.0

    for name, config in ASSET_UNIVERSE.items():
        ticker = config["price"]
        if ticker not in prices.columns:
            print(f"    ✗ {name}: no price data")
            continue

        p = prices[ticker].dropna()
        z_tech = compute_tech_zscore(p, window)
        z_funda = compute_funda_zscore(p, config["type"], fred_data, window, enhanced=enhanced)
        w_tech = compute_wtech_series(p, enhanced=enhanced)

        if z_tech is None or z_funda is None or w_tech is None:
            print(f"    ✗ {name}: insufficient data")
            continue

        df = pd.DataFrame({'tech': z_tech, 'funda': z_funda, 'wTech': w_tech}).dropna()
        df['composite'] = (df['wTech']/100) * df['tech'] + (1 - df['wTech']/100) * df['funda']
        df['signal'] = 'N'
        df.loc[df['composite'] > ow_threshold, 'signal'] = 'OW'
        df.loc[df['composite'] < uw_threshold, 'signal'] = 'UW'

        signal_data[name] = df
        n_ow = (df['signal'] == 'OW').sum()
        n_uw = (df['signal'] == 'UW').sum()
        n_n = (df['signal'] == 'N').sum()
        print(f"    ✓ {name}: {len(df)} mo | OW:{n_ow} UW:{n_uw} N:{n_n}")

    return signal_data


def compute_metrics(rets):
    """Compute standard performance metrics from a monthly return series."""
    cum = (1 + rets).cumprod()
    n_years = len(rets) / 12
    ann_ret = (cum.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else 0
    ann_vol = rets.std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
    return {
        "ann_return": round(float(ann_ret) * 100, 2),
        "ann_vol": round(float(ann_vol) * 100, 2),
        "sharpe": round(float(sharpe), 3),
        "max_drawdown": round(float(max_dd) * 100, 2),
        "calmar": round(float(calmar), 3),
        "cumulative": round(float(cum.iloc[-1] - 1) * 100, 2),
    }


def compute_strategy(returns_df, saa_weights_pct, signal_data=None, tilt_pct=0):
    """Compute portfolio return series for given SAA weights + optional signal tilts."""
    strat_returns = pd.Series(0.0, index=returns_df.index)
    tilt = tilt_pct / 100.0

    # Build weight history for output
    weight_history = []

    for dt in returns_df.index:
        period_return = 0.0
        total_weight = 0.0
        month_weights = {}

        for name, base_pct in saa_weights_pct.items():
            if name not in returns_df.columns:
                continue

            base_w = base_pct / 100.0
            adjusted_w = base_w
            sig = 'N'

            if signal_data and name in signal_data and tilt > 0:
                sig_df = signal_data[name]
                prev_signals = sig_df[sig_df.index < dt]
                if len(prev_signals) > 0:
                    sig = prev_signals.iloc[-1]['signal']
                    if sig == 'OW':
                        adjusted_w = base_w + tilt
                    elif sig == 'UW':
                        adjusted_w = max(0, base_w - tilt)

            adjusted_w = max(0, adjusted_w)
            period_return += adjusted_w * returns_df.loc[dt, name]
            total_weight += adjusted_w
            month_weights[name] = {"weight": round(adjusted_w * 100, 1), "signal": sig}

        if total_weight > 0 and abs(total_weight - 1.0) > 0.001:
            period_return = period_return / total_weight
        strat_returns[dt] = period_return
        weight_history.append({"date": dt.strftime('%Y-%m'), "weights": month_weights})

    return strat_returns, weight_history


def run_backtest(prices, fred_data, start_year=2012, end_year=2025, window=24):
    """Run the full historical backtest."""

    # Build signals for ALL assets
    print(f"\n  Computing signals for {len(ASSET_UNIVERSE)} assets...")
    signals = build_signals(prices, fred_data, window, enhanced=False)

    # Build monthly returns for all assets
    monthly_returns = {}
    for name, config in ASSET_UNIVERSE.items():
        ticker = config["price"]
        if ticker in prices.columns:
            monthly_price = prices[ticker].resample('ME').last()
            monthly_returns[name] = monthly_price.pct_change()

    all_returns = pd.DataFrame(monthly_returns).dropna()

    if all_returns.empty:
        print("\n  ✗ No return data available")
        return None

    # Build benchmark returns
    bm_returns = {}
    for bm in BENCHMARK_ETFS.values():
        for t in ([bm.get("ticker")] if "ticker" in bm else list(bm.get("tickers", {}).keys())):
            if t and t not in bm_returns and t in prices.columns:
                mp = prices[t].resample('ME').last()
                bm_returns[t] = mp.pct_change()

    # ═══════════════════════════════════════════
    # PANEL 1: CORE SIGNAL VALIDATION (no alts)
    # ═══════════════════════════════════════════
    print(f"\n  ══════════════════════════════════════════════════")
    print(f"  PANEL 1: SIGNAL VALIDATION (equity + bonds only)")
    print(f"  ══════════════════════════════════════════════════")

    core_results = {}  # {period_name: {strategy: metrics}}
    core_cumulative = {}  # full period cumulative series

    for period_name, p_start, p_end in SUB_PERIODS:
        start_dt = pd.Timestamp(f"{p_start}-01-01")
        end_dt = pd.Timestamp(f"{p_end}-12-31")
        period_ret = all_returns.loc[start_dt:end_dt]
        if period_ret.empty:
            continue

        print(f"\n  --- {period_name} ({p_start}–{p_end}, {len(period_ret)} months) ---")
        period_metrics = {}
        header = f"  {'Strategy':<28s} {'Return':>8s} {'Vol':>8s} {'Sharpe':>8s} {'MaxDD':>8s}"
        print(header)
        print("  " + "─" * (len(header) - 2))

        for profile_name, weights in CORE_SAA.items():
            # Static SAA
            saa_ret, _ = compute_strategy(period_ret, weights)
            m = compute_metrics(saa_ret)
            period_metrics[profile_name] = m
            print(f"  {profile_name:<28s} {m['ann_return']:>7.1f}% {m['ann_vol']:>7.1f}% {m['sharpe']:>8.3f} {m['max_drawdown']:>7.1f}%")

            # SAA + TacSignal
            tac_name = f"{profile_name} + Tac"
            tac_ret, wh = compute_strategy(period_ret, weights, signals, TILT_PCT)
            m2 = compute_metrics(tac_ret)
            period_metrics[tac_name] = m2
            delta = m2['ann_return'] - m['ann_return']
            print(f"  {tac_name:<28s} {m2['ann_return']:>7.1f}% {m2['ann_vol']:>7.1f}% {m2['sharpe']:>8.3f} {m2['max_drawdown']:>7.1f}%  Δ{delta:>+.1f}%")

            # Save cumulative for full period
            if period_name == "Full Period":
                cum_saa = (1 + saa_ret).cumprod()
                cum_tac = (1 + tac_ret).cumprod()
                core_cumulative[profile_name] = [
                    {"date": dt.strftime('%Y-%m'), "value": round(float(v), 4)}
                    for dt, v in cum_saa.items()
                ]
                core_cumulative[tac_name] = [
                    {"date": dt.strftime('%Y-%m'), "value": round(float(v), 4)}
                    for dt, v in cum_tac.items()
                ]

        # Benchmarks for this period
        for bm_name, bm_config in BENCHMARK_ETFS.items():
            if "ticker" in bm_config:
                t = bm_config["ticker"]
                if t in bm_returns:
                    bm_ret = bm_returns[t].reindex(period_ret.index).fillna(0)
                    m = compute_metrics(bm_ret)
                    period_metrics[bm_name] = m
                    print(f"  {bm_name:<28s} {m['ann_return']:>7.1f}% {m['ann_vol']:>7.1f}% {m['sharpe']:>8.3f} {m['max_drawdown']:>7.1f}%")
                    if period_name == "Full Period":
                        cum = (1 + bm_ret).cumprod()
                        core_cumulative[bm_name] = [
                            {"date": dt.strftime('%Y-%m'), "value": round(float(v), 4)}
                            for dt, v in cum.items()
                        ]

        core_results[period_name] = period_metrics

    # ═══════════════════════════════════════════
    # PANEL 2: FULL PORTFOLIO (with alts)
    # ═══════════════════════════════════════════
    print(f"\n  ══════════════════════════════════════════════════")
    print(f"  PANEL 2: FULL PORTFOLIO (with alternatives)")
    print(f"  ══════════════════════════════════════════════════")

    full_results = {}
    full_cumulative = {}
    full_weight_history = {}

    for period_name, p_start, p_end in SUB_PERIODS:
        start_dt = pd.Timestamp(f"{p_start}-01-01")
        end_dt = pd.Timestamp(f"{p_end}-12-31")
        period_ret = all_returns.loc[start_dt:end_dt]
        if period_ret.empty:
            continue

        print(f"\n  --- {period_name} ({p_start}–{p_end}) ---")
        period_metrics = {}
        header = f"  {'Strategy':<28s} {'Return':>8s} {'Vol':>8s} {'Sharpe':>8s} {'MaxDD':>8s}"
        print(header)
        print("  " + "─" * (len(header) - 2))

        for profile_name, weights in FULL_SAA.items():
            saa_ret, _ = compute_strategy(period_ret, weights)
            m = compute_metrics(saa_ret)
            period_metrics[profile_name] = m
            print(f"  {profile_name:<28s} {m['ann_return']:>7.1f}% {m['ann_vol']:>7.1f}% {m['sharpe']:>8.3f} {m['max_drawdown']:>7.1f}%")

            tac_name = f"{profile_name} + Tac"
            tac_ret, wh = compute_strategy(period_ret, weights, signals, TILT_PCT)
            m2 = compute_metrics(tac_ret)
            period_metrics[tac_name] = m2
            delta = m2['ann_return'] - m['ann_return']
            print(f"  {tac_name:<28s} {m2['ann_return']:>7.1f}% {m2['ann_vol']:>7.1f}% {m2['sharpe']:>8.3f} {m2['max_drawdown']:>7.1f}%  Δ{delta:>+.1f}%")

            if period_name == "Full Period":
                cum_saa = (1 + saa_ret).cumprod()
                cum_tac = (1 + tac_ret).cumprod()
                full_cumulative[profile_name] = [
                    {"date": dt.strftime('%Y-%m'), "value": round(float(v), 4)}
                    for dt, v in cum_saa.items()
                ]
                full_cumulative[tac_name] = [
                    {"date": dt.strftime('%Y-%m'), "value": round(float(v), 4)}
                    for dt, v in cum_tac.items()
                ]
                # Save weight history for the dashboard
                full_weight_history[tac_name] = wh

        full_results[period_name] = period_metrics

    # ═══════════════════════════════════════════
    # SIGNAL QUALITY (per-asset)
    # ═══════════════════════════════════════════
    print(f"\n  ══════════════════════════════════════════════════")
    print(f"  SIGNAL QUALITY (per asset)")
    print(f"  ══════════════════════════════════════════════════\n")

    start_dt = pd.Timestamp(f"{start_year}-01-01")
    end_dt = pd.Timestamp(f"{end_year}-12-31")
    full_returns = all_returns.loc[start_dt:end_dt]

    hit_rates = {}
    ow_returns_all = []
    uw_returns_all = []
    n_returns_all = []

    for name in signals:
        if name not in full_returns.columns:
            continue
        sig_df = signals[name]
        hits = 0
        total = 0
        for i in range(len(sig_df) - 1):
            dt = sig_df.index[i]
            sig = sig_df.iloc[i]['signal']
            if sig == 'N':
                continue
            future_rets = full_returns[name][full_returns.index > dt]
            if len(future_rets) == 0:
                continue
            next_ret = future_rets.iloc[0]
            total += 1
            if sig == 'OW' and next_ret > 0:
                hits += 1
            elif sig == 'UW' and next_ret < 0:
                hits += 1

        hit_rate = hits / total if total > 0 else 0
        group = ASSET_UNIVERSE[name]["group"] if name in ASSET_UNIVERSE else "?"
        hit_rates[name] = {"hits": hits, "total": total, "rate": round(hit_rate * 100, 1), "group": group}
        marker = "●" if group == "core" else "○"
        print(f"  {marker} {name:<20s} Hit rate: {hit_rate:>5.1%} ({hits}/{total}) [{group}]")

        # Collect for spread
        for i in range(len(sig_df) - 1):
            dt = sig_df.index[i]
            sig = sig_df.iloc[i]['signal']
            future_rets = full_returns[name][full_returns.index > dt]
            if len(future_rets) == 0:
                continue
            next_ret = float(future_rets.iloc[0])
            if sig == 'OW':
                ow_returns_all.append(next_ret)
            elif sig == 'UW':
                uw_returns_all.append(next_ret)
            else:
                n_returns_all.append(next_ret)

    avg_ow = np.mean(ow_returns_all) if ow_returns_all else 0
    avg_uw = np.mean(uw_returns_all) if uw_returns_all else 0
    avg_n = np.mean(n_returns_all) if n_returns_all else 0
    spread = avg_ow - avg_uw

    print(f"\n  OW → {avg_ow:>+.2%}/mo ({len(ow_returns_all)})   UW → {avg_uw:>+.2%}/mo ({len(uw_returns_all)})   N → {avg_n:>+.2%}/mo ({len(n_returns_all)})")
    print(f"  OW–UW spread: {spread:>+.2%}/mo {'✅' if spread > 0 else '⚠'}")

    spread_data = {
        "ow_avg": round(avg_ow * 100, 3), "uw_avg": round(avg_uw * 100, 3),
        "n_avg": round(avg_n * 100, 3), "spread": round(spread * 100, 3),
        "ow_count": len(ow_returns_all), "uw_count": len(uw_returns_all),
    }

    # ═══════════════════════════════════════════
    # CURRENT SIGNALS (most recent month)
    # ═══════════════════════════════════════════
    print(f"\n  ══════════════════════════════════════════════════")
    print(f"  CURRENT SIGNALS")
    print(f"  ══════════════════════════════════════════════════\n")

    current_signals = {}
    signal_changes = {}
    for name, sig_df in signals.items():
        if len(sig_df) < 2:
            continue
        current = sig_df.iloc[-1]
        previous = sig_df.iloc[-2]
        sig = current['signal']
        prev_sig = previous['signal']
        changed = sig != prev_sig
        current_signals[name] = {
            "signal": sig,
            "composite": round(float(current['composite']), 2),
            "tech": round(float(current['tech']), 2),
            "funda": round(float(current['funda']), 2),
            "wTech": round(float(current['wTech']), 1),
            "changed": changed,
            "previous": prev_sig,
            "date": sig_df.index[-1].strftime('%Y-%m'),
        }
        marker = "⬥ CHANGED" if changed else ""
        print(f"  {name:<20s} {sig:>2s}  composite:{current['composite']:>+.2f}  {marker}")
        if changed:
            signal_changes[name] = {"from": prev_sig, "to": sig}

    if signal_changes:
        print(f"\n  🔔 Signal changes this month:")
        for name, ch in signal_changes.items():
            print(f"     {name}: {ch['from']} → {ch['to']}")

    # ═══════════════════════════════════════════
    # OUTPUT JSON
    # ═══════════════════════════════════════════
    output = {
        "generated": datetime.now().isoformat(),
        "period": f"{start_year}–{end_year}",
        "months": len(full_returns),
        # Core signal validation (equity + bonds only, vs benchmarks)
        "core": {
            "results": core_results,
            "cumulative": core_cumulative,
            "saa_profiles": CORE_SAA,
        },
        # Full portfolio (with alts)
        "full": {
            "results": full_results,
            "cumulative": full_cumulative,
            "saa_profiles": FULL_SAA,
            "weight_history": full_weight_history,
        },
        # Signal quality
        "hit_rates": hit_rates,
        "spread": spread_data,
        # Current state
        "current_signals": current_signals,
        "signal_changes": signal_changes,
        # For dashboard metrics display
        "metrics": core_results.get("Full Period", {}),
        "cumulative": {**core_cumulative, **full_cumulative},
        "saa_weights": {**CORE_SAA, **FULL_SAA},
    }

    return output


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TacSignal Backtest Engine")
    parser.add_argument("--fred-key", "-k", help="FRED API key")
    parser.add_argument("--start", "-s", type=int, default=2012, help="Start year (default: 2012)")
    parser.add_argument("--end", "-e", type=int, default=2025, help="End year (default: 2025)")
    parser.add_argument("--output", "-o", default="tacsignal-backtest.json", help="Output file")
    args = parser.parse_args()

    fred_key = args.fred_key or os.environ.get("FRED_API_KEY")
    if not fred_key:
        fred_key_file = Path(__file__).parent / ".fred_key"
        if fred_key_file.exists():
            fred_key = fred_key_file.read_text().strip()
            print(f"  (Using FRED key from {fred_key_file})")

    print("\n╔══════════════════════════════════════════════╗")
    print("║       TacSignal Backtest Engine               ║")
    print("╚══════════════════════════════════════════════╝\n")

    print("Step 1: Fetching historical price data")
    # Need data from well before start_year for rolling windows
    fetch_start = min(args.start - 5, 2007)
    prices = fetch_prices(start_year=fetch_start)

    print("\nStep 2: Fetching FRED data")
    fred_data = fetch_fred(fred_key, start_year=fetch_start)

    print("\nStep 3: Running backtest")
    output = run_backtest(prices, fred_data, start_year=args.start, end_year=args.end)

    if output is None:
        print("\n  ✗ Backtest failed — no data available. Run this script on your local machine.")
        sys.exit(1)

    # Save
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n  ✅ Results saved to {args.output}")
    print(f"     Import into the TacSignal dashboard to see charts")
