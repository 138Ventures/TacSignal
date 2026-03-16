#!/usr/bin/env python3
"""
TacSignal Data Pipeline
========================
Pulls market data from Yahoo Finance and FRED, computes technical and fundamental
z-scores, determines w_tech (behavioral weight), and outputs a JSON file that the
TacSignal dashboard can import directly.

Requirements:
    pip install yfinance pandas-datareader fredapi requests pandas numpy

Usage:
    # Basic run — outputs tacsignal-YYYY-MM-DD.json
    python tacsignal_data_pipeline.py

    # Custom output path
    python tacsignal_data_pipeline.py --output my_signals.json

    # Use specific FRED API key (or set FRED_API_KEY env var)
    python tacsignal_data_pipeline.py --fred-key YOUR_API_KEY

    # Custom lookback window (months)
    python tacsignal_data_pipeline.py --window 24

    # Only compute for specific assets
    python tacsignal_data_pipeline.py --assets "Equity U.S.,Gold,Bond U.S."

FRED API Key:
    Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html
    Then either:
      export FRED_API_KEY=your_key_here
    or pass via --fred-key
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

# Yahoo Finance tickers for each asset
ASSET_TICKERS = {
    "Equity CH":        {"price": "^SSMI",   "type": "equity", "region": "CH"},
    "Equity EU ex CH":  {"price": "EZU",     "type": "equity", "region": "EU"},
    "Equity U.S.":      {"price": "^GSPC",   "type": "equity", "region": "US"},
    "Equity Japan":     {"price": "EWJ",     "type": "equity", "region": "JP"},
    "Equity Asia-Pac":  {"price": "AAXJ",    "type": "equity", "region": "APAC"},
    "Equity EM":        {"price": "EEM",     "type": "equity", "region": "EM"},
    "Bond CH":          {"price": "CSBGC0.SW", "type": "bond",  "region": "CH"},
    "Bond EU":          {"price": "IEAG.L",  "type": "bond",   "region": "EU"},
    "Bond U.S.":        {"price": "AGG",     "type": "bond",   "region": "US"},
    "Bond Japan":       {"price": "BNDX",    "type": "bond",   "region": "JP"},
    "Gold":             {"price": "GC=F",    "type": "commodity", "region": "GLOBAL"},
    "Oil":              {"price": "CL=F",    "type": "commodity", "region": "GLOBAL"},
    "Euro":             {"price": "EURCHF=X","type": "currency", "region": "EU"},
    "U.S. Dollar":      {"price": "CHF=X",   "type": "currency", "region": "US"},
}

SECTOR_TICKERS = {
    "Financials":       {"price": "XLF",  "type": "sector"},
    "Cons. Disc.":      {"price": "XLY",  "type": "sector"},
    "Industrials":      {"price": "XLI",  "type": "sector"},
    "Cons. Staples":    {"price": "XLP",  "type": "sector"},
    "Health Care":      {"price": "XLV",  "type": "sector"},
    "Info Tech":        {"price": "XLK",  "type": "sector"},
    "Energy":           {"price": "XLE",  "type": "sector"},
    "Materials":        {"price": "XLB",  "type": "sector"},
    "Telecom":          {"price": "XLC",  "type": "sector"},
    "Utilities":        {"price": "XLU",  "type": "sector"},
}

# FRED series for fundamental indicators
FRED_SERIES = {
    # US yields and spreads
    "US_10Y":           "DGS10",          # 10-year Treasury yield
    "US_2Y":            "DGS2",           # 2-year Treasury yield
    "US_TIPS_10Y":      "DFII10",         # 10-year TIPS (real yield)
    "US_BAA_SPREAD":    "BAMLC0A4CBBB",   # BBB corporate spread
    "US_HY_SPREAD":     "BAMLH0A0HYM2",   # High yield spread
    # Euro
    "EU_10Y":           "IRLTLT01EZM156N", # Euro area 10Y govt yield
    # Japan
    "JP_10Y":           "IRLTLT01JPM156N", # Japan 10Y govt yield
    # Macro
    "US_PMI":           "MANEMP",          # ISM Manufacturing Employment
    "VIX":              "VIXCLS",          # VIX
    "US_UNRATE":        "UNRATE",          # US Unemployment rate
    "US_CPI_YOY":       "CPIAUCSL",        # CPI (we compute YoY)
    # Shiller CAPE
    "CAPE":             "MULTPL/SHILLER_PE_RATIO_MONTH",  # not on FRED, handled separately
}


# ═══════════════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════════════

def fetch_yahoo_prices(tickers_dict, years=3):
    """Fetch daily price data from Yahoo Finance."""
    import yfinance as yf

    all_tickers = list(set(v["price"] for v in tickers_dict.values()))
    end = datetime.now()
    start = end - timedelta(days=years * 365 + 60)  # extra buffer

    print(f"  Fetching {len(all_tickers)} tickers from Yahoo Finance...")

    # Try bulk download first
    prices = None
    try:
        data = yf.download(all_tickers, start=start, end=end, auto_adjust=True, progress=False)
        if data is not None and len(data) > 0:
            if isinstance(data.columns, pd.MultiIndex):
                prices = data['Close']
            else:
                prices = data[['Close']].copy()
                prices.columns = all_tickers

            # Check how many tickers actually have data
            valid_cols = [c for c in prices.columns if prices[c].notna().sum() > 50]
            if len(valid_cols) < len(all_tickers) // 2:
                print(f"  Bulk download only got {len(valid_cols)}/{len(all_tickers)} tickers — falling back to individual downloads...")
                prices = None
    except Exception as e:
        print(f"  Bulk download failed ({e}) — falling back to individual downloads...")

    # Fallback: download tickers one by one (much more reliable)
    if prices is None or len(prices) == 0 or prices.isna().all().all():
        print(f"  Downloading tickers individually...")
        series_dict = {}
        for ticker in all_tickers:
            try:
                t = yf.Ticker(ticker)
                hist = t.history(start=start, end=end, auto_adjust=True)
                if hist is not None and len(hist) > 50:
                    series_dict[ticker] = hist['Close']
                    print(f"    ✓ {ticker}: {len(hist)} days")
                else:
                    print(f"    ✗ {ticker}: no data returned")
            except Exception as e:
                print(f"    ✗ {ticker}: {e}")
        if series_dict:
            prices = pd.DataFrame(series_dict)
        else:
            print("  ⚠ No price data retrieved. Check your internet connection.")
            prices = pd.DataFrame(columns=all_tickers)

    # Forward-fill gaps (holidays etc.)
    prices = prices.ffill()
    print(f"  Got {len(prices)} daily observations, {prices.shape[1]} tickers")
    return prices


def fetch_fred_data(fred_key, years=3):
    """Fetch macro/fundamental data from FRED."""
    try:
        from fredapi import Fred
    except ImportError:
        print("  WARNING: fredapi not installed. Skipping FRED data.")
        print("  Install with: pip install fredapi")
        return pd.DataFrame()

    if not fred_key:
        print("  WARNING: No FRED API key provided. Skipping FRED data.")
        print("  Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html")
        return pd.DataFrame()

    fred = Fred(api_key=fred_key)
    end = datetime.now()
    start = end - timedelta(days=years * 365 + 60)

    series_to_fetch = {k: v for k, v in FRED_SERIES.items() if not v.startswith("MULTPL")}

    result = {}
    for name, series_id in series_to_fetch.items():
        try:
            s = fred.get_series(series_id, observation_start=start, observation_end=end)
            result[name] = s
            print(f"    ✓ {name} ({series_id}): {len(s)} obs")
        except Exception as e:
            print(f"    ✗ {name} ({series_id}): {e}")

    if result:
        df = pd.DataFrame(result)
        df = df.ffill()
        return df
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════
# TECHNICAL SCORE COMPUTATION
# ═══════════════════════════════════════════════════════════════

def compute_technical_indicators(prices_series):
    """
    Compute raw technical composite from daily prices.

    Components:
      1. Price / SMA_200 - 1  (trend position)
      2. 12-1 momentum (12mo return ex last month)
      3. SMA_50 / SMA_200 - 1 (golden/death cross signal)

    Returns a monthly series of the raw technical composite.
    """
    if prices_series is None or len(prices_series) < 250:
        return None

    p = prices_series.copy().dropna()
    if len(p) < 250:
        return None

    # Daily indicators
    sma_50 = p.rolling(50).mean()
    sma_200 = p.rolling(200).mean()

    # Resample to monthly (end of month)
    monthly_price = p.resample('ME').last()
    monthly_sma50 = sma_50.resample('ME').last()
    monthly_sma200 = sma_200.resample('ME').last()

    # Component 1: Price relative to SMA200
    trend_position = (monthly_price / monthly_sma200) - 1

    # Component 2: 12-1 momentum
    ret_12m = monthly_price.pct_change(12)
    ret_1m = monthly_price.pct_change(1)
    mom_12_1 = ret_12m - ret_1m  # approximate 12-1

    # Component 3: SMA crossover
    crossover = (monthly_sma50 / monthly_sma200) - 1

    # Composite (equal weight)
    raw_tech = 0.5 * trend_position + 0.3 * mom_12_1 + 0.2 * crossover

    return raw_tech.dropna()


# ═══════════════════════════════════════════════════════════════
# FUNDAMENTAL SCORE COMPUTATION
# ═══════════════════════════════════════════════════════════════

def compute_fundamental_indicators(asset_name, asset_config, prices_series, fred_data):
    """
    Compute raw fundamental composite.

    Strategy varies by asset type:
    - Equities/Sectors: inverse of price deviation from trend (value proxy) + VIX inverse
    - Bonds: yield level + term spread
    - Commodities: mean reversion of real price
    - Currencies: interest rate differential proxy
    """
    asset_type = asset_config.get("type", "equity")

    if prices_series is None or len(prices_series) < 250:
        return None

    p = prices_series.copy().dropna()
    monthly_price = p.resample('ME').last()

    if asset_type in ("equity", "sector"):
        # Value proxy: negative of price deviation from trend
        # (cheap = high score, expensive = low score)
        log_price = np.log(monthly_price)
        trend = log_price.rolling(24, min_periods=18).mean()
        value_signal = -(log_price - trend)  # below trend = cheap = positive

        # If we have VIX data, add inverse VIX (low vol = favorable)
        # IMPORTANT: normalize both components before combining so VIX doesn't dominate
        if fred_data is not None and 'VIX' in fred_data.columns:
            vix_monthly = fred_data['VIX'].resample('ME').last().dropna()
            inv_vix = -vix_monthly  # lower VIX = more positive
            # Align
            combined = pd.DataFrame({'value': value_signal, 'vix': inv_vix})
            combined = combined.dropna()
            if len(combined) > 12:
                # Normalize each component to zero-mean, unit-variance before combining
                for col in ['value', 'vix']:
                    mu = combined[col].rolling(24, min_periods=12).mean()
                    sigma = combined[col].rolling(24, min_periods=12).std()
                    combined[col] = (combined[col] - mu) / sigma
                combined = combined.dropna()
                if len(combined) > 6:
                    raw_funda = 0.7 * combined['value'] + 0.3 * combined['vix']
                    return raw_funda
        return value_signal.dropna()

    elif asset_type == "bond":
        region = asset_config.get("region", "US")
        # Use yield level as a positive signal (higher yield = more attractive)
        yield_key = {"US": "US_10Y", "EU": "EU_10Y", "JP": "JP_10Y"}.get(region)

        if fred_data is not None and yield_key and yield_key in fred_data.columns:
            yields = fred_data[yield_key].resample('ME').last().dropna()
            if len(yields) > 6:
                return yields

        # Fallback: inverse price momentum (bonds cheap when prices are low)
        log_price = np.log(monthly_price)
        trend = log_price.rolling(24, min_periods=12).mean()
        return -(log_price - trend).dropna()

    elif asset_type == "commodity":
        # Mean reversion: negative of deviation from 2-year average
        log_price = np.log(monthly_price)
        trend = log_price.rolling(24, min_periods=12).mean()
        return -(log_price - trend).dropna()

    elif asset_type == "currency":
        # PPP proxy: negative of deviation from long-term average
        log_price = np.log(monthly_price)
        trend = log_price.rolling(24, min_periods=12).mean()
        return -(log_price - trend).dropna()

    # Fallback
    return None


# ═══════════════════════════════════════════════════════════════
# Z-SCORE COMPUTATION
# ═══════════════════════════════════════════════════════════════

def zscore_rolling(series, window=24):
    """Compute rolling z-score over a window of monthly observations."""
    if series is None or len(series) < window:
        return None
    mu = series.rolling(window, min_periods=max(12, window // 2)).mean()
    sigma = series.rolling(window, min_periods=max(12, window // 2)).std()
    z = (series - mu) / sigma
    # Clip to [-4, 4]
    z = z.clip(-4, 4)
    return z


# ═══════════════════════════════════════════════════════════════
# BEHAVIORAL WEIGHT (w_tech)
# ═══════════════════════════════════════════════════════════════

def compute_w_tech(prices_series, window=12):
    """
    Simple regime-based w_tech approximation.

    Logic:
    - Compute trailing 12-month absolute momentum
    - If momentum is above median → trending regime → w_tech = 100%
    - If momentum is below median → mean-reverting regime → w_tech = 0%

    This is a simplified proxy for BhFS's proprietary behavioral function.
    The real BhFS function likely uses more nuanced behavioral indicators.
    """
    if prices_series is None or len(prices_series) < 300:
        return 50.0  # default

    monthly = prices_series.resample('ME').last()
    abs_mom = monthly.pct_change(12).abs()
    median_mom = abs_mom.rolling(24, min_periods=12).median()

    # Latest values
    current_mom = abs_mom.iloc[-1] if not pd.isna(abs_mom.iloc[-1]) else 0
    med = median_mom.iloc[-1] if not pd.isna(median_mom.iloc[-1]) else current_mom

    if current_mom > med:
        return 100.0  # trending → trust technicals
    else:
        return 0.0    # mean-reverting → trust fundamentals


def compute_w_tech_series(prices_series, window=12):
    """Compute w_tech as a monthly time series (not just latest value)."""
    if prices_series is None or len(prices_series) < 300:
        return None

    monthly = prices_series.resample('ME').last()
    abs_mom = monthly.pct_change(12).abs()
    median_mom = abs_mom.rolling(24, min_periods=12).median()

    w_series = (abs_mom > median_mom).astype(float) * 100.0
    return w_series.dropna()


# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_pipeline(fred_key=None, window=24, output_path=None, asset_filter=None):
    """Run the full data pipeline."""

    print("\n╔══════════════════════════════════════════════╗")
    print("║         TacSignal Data Pipeline              ║")
    print("╚══════════════════════════════════════════════╝\n")

    # 1. Fetch data
    print("Step 1: Fetching price data from Yahoo Finance")
    all_tickers = {**ASSET_TICKERS, **SECTOR_TICKERS}
    fetch_years = max(6, (window / 12) + 4)  # need 6+ years for rolling fundamental indicators
    prices = fetch_yahoo_prices(all_tickers, years=fetch_years)

    print("\nStep 2: Fetching fundamental data from FRED")
    fred_data = fetch_fred_data(fred_key, years=fetch_years)

    # 2. Compute signals for each asset
    print(f"\nStep 3: Computing signals (window={window} months)")

    def process_universe(ticker_dict, label):
        results = []
        history_dict = {}  # name → [{date, tech, funda, wTech, composite, signal}, ...]

        for name, config in ticker_dict.items():
            if asset_filter and name not in asset_filter:
                continue

            ticker = config["price"]
            if ticker not in prices.columns:
                print(f"  ✗ {name}: ticker {ticker} not found in downloaded data")
                results.append({
                    "name": name, "tech": 0, "funda": 0, "wTech": 50,
                    "prevTech": 0, "prevFunda": 0, "_error": f"No data for {ticker}"
                })
                history_dict[name] = []
                continue

            price_series = prices[ticker].dropna()

            # Technical
            raw_tech = compute_technical_indicators(price_series)
            z_tech = zscore_rolling(raw_tech, window)

            # Fundamental
            raw_funda = compute_fundamental_indicators(name, config, price_series, fred_data)
            z_funda = zscore_rolling(raw_funda, window)

            # w_tech (current + series)
            w = compute_w_tech(price_series)
            w_series = compute_w_tech_series(price_series)

            # Extract current and previous values
            tech_current = float(z_tech.iloc[-1]) if z_tech is not None and len(z_tech) > 0 and not pd.isna(z_tech.iloc[-1]) else 0.0
            tech_prev = float(z_tech.iloc[-2]) if z_tech is not None and len(z_tech) > 1 and not pd.isna(z_tech.iloc[-2]) else tech_current

            funda_current = float(z_funda.iloc[-1]) if z_funda is not None and len(z_funda) > 0 and not pd.isna(z_funda.iloc[-1]) else 0.0
            funda_prev = float(z_funda.iloc[-2]) if z_funda is not None and len(z_funda) > 1 and not pd.isna(z_funda.iloc[-2]) else funda_current

            # Compute composite
            w_pct = w / 100
            composite = w_pct * tech_current + (1 - w_pct) * funda_current
            signal = "OW" if composite > 1 else "UW" if composite < -1 else "NEUTRAL"

            print(f"  ✓ {name:22s} | tech={tech_current:+.2f} funda={funda_current:+.2f} w={w:5.1f}% | S={composite:+.2f} → {signal}")

            results.append({
                "name": name,
                "tech": round(tech_current, 4),
                "funda": round(funda_current, 4),
                "wTech": round(w, 1),
                "prevTech": round(tech_prev, 4),
                "prevFunda": round(funda_prev, 4),
            })

            # ── Build monthly history ──
            hist = []
            if z_tech is not None and z_funda is not None and w_series is not None:
                # Align all series
                combined = pd.DataFrame({
                    'tech': z_tech, 'funda': z_funda, 'wTech': w_series
                }).dropna()
                # Keep last 24 months
                combined = combined.tail(24)
                for dt, row in combined.iterrows():
                    wt = row['wTech'] / 100
                    c = wt * row['tech'] + (1 - wt) * row['funda']
                    s = "OW" if c > 1 else "UW" if c < -1 else "N"
                    hist.append({
                        "date": dt.strftime('%Y-%m'),
                        "tech": round(float(row['tech']), 3),
                        "funda": round(float(row['funda']), 3),
                        "wTech": round(float(row['wTech']), 1),
                        "composite": round(float(c), 3),
                        "signal": s,
                    })
            history_dict[name] = hist

        return results, history_dict

    print(f"\n  --- {len(ASSET_TICKERS)} Asset Classes ---")
    asset_results, asset_history = process_universe(ASSET_TICKERS, "Assets")

    print(f"\n  --- {len(SECTOR_TICKERS)} U.S. Sectors ---")
    sector_results, sector_history = process_universe(SECTOR_TICKERS, "Sectors")

    # 3. Build output
    output = {
        "date": datetime.now().isoformat(),
        "generated_by": "tacsignal_data_pipeline.py",
        "window_months": window,
        "data_sources": {
            "prices": "Yahoo Finance",
            "fundamentals": "FRED" if fred_data is not None and len(fred_data) > 0 else "Price-derived (FRED unavailable)",
            "w_tech": "Regime approximation (momentum-based)"
        },
        "thOW": 1.0,
        "thUW": -1.0,
        "assets": asset_results,
        "sectors": sector_results,
        "history": {**asset_history, **sector_history},
    }

    # 4. Save
    if output_path is None:
        output_path = f"tacsignal-{datetime.now().strftime('%Y-%m-%d')}.json"

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Saved to {output_path}")
    print(f"   {len(asset_results)} assets + {len(sector_results)} sectors")
    print(f"   Open the TacSignal dashboard → Import JSON → select this file")

    # Summary
    all_items = asset_results + sector_results
    ow = sum(1 for r in all_items if (r['wTech']/100)*r['tech'] + (1-r['wTech']/100)*r['funda'] > 1)
    uw = sum(1 for r in all_items if (r['wTech']/100)*r['tech'] + (1-r['wTech']/100)*r['funda'] < -1)
    ne = len(all_items) - ow - uw
    print(f"\n   Summary: {ow} Overweight | {uw} Underweight | {ne} Neutral")

    return output


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TacSignal Data Pipeline")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument("--fred-key", "-k", help="FRED API key (or set FRED_API_KEY env var)")
    parser.add_argument("--window", "-w", type=int, default=24, help="Z-score rolling window in months (default: 24)")
    parser.add_argument("--assets", "-a", help="Comma-separated list of asset names to process (default: all)")
    args = parser.parse_args()

    fred_key = args.fred_key or os.environ.get("FRED_API_KEY")
    asset_filter = [a.strip() for a in args.assets.split(",")] if args.assets else None

    try:
        run_pipeline(
            fred_key=fred_key,
            window=args.window,
            output_path=args.output,
            asset_filter=asset_filter,
        )
    except ImportError as e:
        print(f"\n❌ Missing dependency: {e}")
        print("\nInstall required packages:")
        print("  pip install yfinance pandas numpy fredapi")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
