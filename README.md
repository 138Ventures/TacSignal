# TacSignal — Tactical Asset Allocation Dashboard

An interactive tactical asset allocation tool that computes composite z-score signals across 14 asset classes and 10 U.S. equity sectors, combining technical momentum with fundamental/macro indicators.

Built as an enhanced, open-source implementation of the signal framework described in [BhFS](https://www.bhfs-ag.ch/) (Behavioural Finance Solutions) research reports.

![Dashboard Preview](https://img.shields.io/badge/status-active-brightgreen) ![Python](https://img.shields.io/badge/python-3.9+-blue) ![License](https://img.shields.io/badge/license-MIT-green)

## What It Does

**For each asset class, TacSignal computes:**

1. **Technical Z-Score** — SMA crossovers, 12-1 momentum, trend position (price vs SMA200)
2. **Fundamental Z-Score** — Valuation deviations, yield levels, VIX, credit spreads (via FRED)
3. **Behavioral Weight (w_tech)** — Regime detection: trending markets trust technicals, mean-reverting markets trust fundamentals
4. **Composite Signal** — `S = w_tech × Z_tech + (1 − w_tech) × Z_funda`
5. **Allocation Call** — OVERWEIGHT (S > 1), UNDERWEIGHT (S < −1), or NEUTRAL

## Quick Start

### 1. Open the Dashboard
Open `tactical-signal-v2.html` in any browser. It works offline with sample data (Sep 2024 snapshot + 24 months of simulated history).

### 2. Get Live Data (optional)
```bash
# Install dependencies
pip install yfinance pandas numpy fredapi

# Run the pipeline (FRED key is optional but recommended)
python3 tacsignal_data_pipeline.py --fred-key YOUR_FRED_KEY
```
Or just double-click `run_tacsignal.command` (Mac) / `run_tacsignal.bat` (Windows).

### 3. Import into Dashboard
Click **Import** in the dashboard header → select the generated `tacsignal-YYYY-MM-DD.json`.

## Files

| File | Description |
|------|-------------|
| `tactical-signal-v2.html` | Interactive dashboard (7 tabs: Dashboard, Data Grid, History, What-If, Backtest, Methodology, Data Sources) |
| `tacsignal_data_pipeline.py` | Python pipeline — pulls data from Yahoo Finance + FRED, computes all signals |
| `tacsignal_backtest.py` | Historical backtest engine — tests signals vs 60/40, 80/20, 100% equity benchmarks (2012–2025) |
| `run_tacsignal.command` | One-click Mac launcher (installs packages, manages FRED key, runs pipeline) |
| `run_tacsignal.bat` | One-click Windows launcher |
| `SETUP_GUIDE.html` | Visual step-by-step setup guide |
| `TacSignal_Monthly_Playbook.docx` | Detailed methodology reference and monthly update checklist |

## Data Sources

| Source | What | Cost |
|--------|------|------|
| [Yahoo Finance](https://finance.yahoo.com) | Daily prices for 24 ETFs/indices | Free |
| [FRED](https://fred.stlouisfed.org) | Yields, credit spreads, VIX, CPI, unemployment | Free (API key required) |

Get a free FRED API key at: https://fred.stlouisfed.org/docs/api/api_key.html

## Signal Framework

```
Composite = w_tech × Z_technical + (1 − w_tech) × Z_fundamental

where:
  Z = rolling 24-month z-score, clipped to [−4, +4]
  w_tech = 100% if |12m momentum| > median (trending), 0% otherwise (mean-reverting)

Decision:
  S > +1  →  OVERWEIGHT
  S < −1  →  UNDERWEIGHT
  else    →  NEUTRAL
```

### Technical Indicators
- Price / SMA_200 − 1 (trend position)
- 12-month minus 1-month return (cross-sectional momentum)
- SMA_50 / SMA_200 − 1 (golden/death cross)

### Fundamental Indicators
- Equity/Sectors: valuation deviation from trend + inverse VIX
- Bonds: yield levels from FRED (10Y US, EU, JP)
- Commodities: mean-reversion from 24-month average
- Currencies: deviation from 24-month trend (PPP proxy)

## Asset Universe

**14 Asset Classes:** Equity CH, Equity EU, Equity U.S., Equity Japan, Equity Asia-Pac, Equity EM, Bond CH, Bond EU, Bond U.S., Bond Japan, Gold, Oil, Euro, U.S. Dollar

**10 U.S. Sectors:** Financials, Consumer Discretionary, Industrials, Consumer Staples, Health Care, Info Tech, Energy, Materials, Telecom, Utilities

## Dashboard Features

- **Interactive scatter plots** — click any dot for waterfall decomposition
- **Movement arrows** — show signal drift from previous month
- **What-If Scenario** — stress-test signals with global tech/funda shocks
- **History tab** — 24-month time series with composite lines, phase diagrams, signal timeline
- **CSV paste import** — drop in spreadsheet data directly
- **JSON import/export** — save and load complete signal snapshots

## Methodology

Based on the BhFS (Behavioural Finance Solutions) tactical allocation framework. The behavioral weight function (`w_tech`) is approximated using a momentum-regime detection heuristic, as the original function is proprietary. See the Methodology tab in the dashboard or `TacSignal_Monthly_Playbook.docx` for full details.

## License

MIT
