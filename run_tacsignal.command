#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# TacSignal — One-Click Data Update (Mac / Linux)
# ═══════════════════════════════════════════════════════════════
# Double-click this file on Mac to run, or execute in Terminal:
#   bash run_tacsignal.command
# ═══════════════════════════════════════════════════════════════

set -e

# Go to the folder where this script lives
cd "$(dirname "$0")"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║       TacSignal — Monthly Data Updater           ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ─── Step 1: Check Python ───
PYTHON=""
if command -v python3 &>/dev/null; then
    PYTHON="python3"
elif command -v python &>/dev/null; then
    PYTHON="python"
else
    echo "❌ Python not found!"
    echo ""
    echo "   Install Python from: https://www.python.org/downloads/"
    echo "   (Choose Python 3.10 or newer)"
    echo ""
    echo "   On Mac, you can also run:  brew install python3"
    echo ""
    read -p "Press Enter to close..."
    exit 1
fi

echo "✓ Using $($PYTHON --version)"

# ─── Step 2: Install packages if missing ───
echo ""
echo "Checking required packages..."

MISSING=0
$PYTHON -c "import yfinance" 2>/dev/null || MISSING=1
$PYTHON -c "import pandas" 2>/dev/null || MISSING=1
$PYTHON -c "import numpy" 2>/dev/null || MISSING=1
$PYTHON -c "import fredapi" 2>/dev/null || MISSING=1

if [ "$MISSING" -eq 1 ]; then
    echo "Installing required packages (one-time setup)..."
    $PYTHON -m pip install --upgrade pip --quiet 2>/dev/null || true
    $PYTHON -m pip install --upgrade yfinance pandas numpy fredapi --quiet
    echo "✓ Packages installed"
else
    # Always keep yfinance up to date (fixes frequent API changes)
    echo "Upgrading yfinance to latest version..."
    $PYTHON -m pip install --upgrade yfinance --quiet 2>/dev/null || true
    echo "✓ All packages up to date"
fi

# ─── Step 3: FRED API Key ───
echo ""

# Check if key is already saved (local file first, then home dir)
LOCAL_KEY="$(dirname "$0")/.fred_key"
KEY_FILE="$HOME/.tacsignal_fred_key"
if [ -f "$LOCAL_KEY" ]; then
    FRED_KEY=$(cat "$LOCAL_KEY" | tr -d '[:space:]')
    echo "✓ FRED API key loaded from local .fred_key"
elif [ -f "$KEY_FILE" ]; then
    FRED_KEY=$(cat "$KEY_FILE" | tr -d '[:space:]')
    echo "✓ FRED API key loaded from saved config"
elif [ -n "$FRED_API_KEY" ]; then
    FRED_KEY="$FRED_API_KEY"
    echo "✓ FRED API key found in environment"
else
    echo "════════════════════════════════════════════════════"
    echo "  FRED API Key Setup (one-time only)"
    echo "════════════════════════════════════════════════════"
    echo ""
    echo "  You need a free FRED API key for macro data."
    echo "  Get one here (takes 30 seconds):"
    echo ""
    echo "  👉  https://fred.stlouisfed.org/docs/api/api_key.html"
    echo ""
    echo "  1. Click 'Request or view your API keys'"
    echo "  2. Sign in (or create a free account)"
    echo "  3. Copy the 32-character key"
    echo ""
    read -p "  Paste your FRED API key here: " FRED_KEY
    echo ""

    if [ -z "$FRED_KEY" ]; then
        echo "  ⚠ No key entered — will run WITHOUT FRED data."
        echo "    (Fundamental scores will use price-derived proxies)"
        echo ""
    else
        # Save for next time
        echo "$FRED_KEY" > "$KEY_FILE"
        chmod 600 "$KEY_FILE"
        echo "  ✓ Key saved to $KEY_FILE (won't ask again)"
    fi
fi

# ─── Step 4: Run the pipeline ───
echo ""
echo "════════════════════════════════════════════════════"
echo "  Running pipeline..."
echo "════════════════════════════════════════════════════"
echo ""

OUTPUT_FILE="tacsignal-$(date +%Y-%m-%d).json"

if [ -n "$FRED_KEY" ]; then
    $PYTHON tacsignal_data_pipeline.py --output "$OUTPUT_FILE" --fred-key "$FRED_KEY"
else
    $PYTHON tacsignal_data_pipeline.py --output "$OUTPUT_FILE"
fi

# ─── Step 5: Done ───
echo ""
echo "════════════════════════════════════════════════════"
echo "  ✅ DONE!"
echo "════════════════════════════════════════════════════"
echo ""
echo "  Output file:  $(pwd)/$OUTPUT_FILE"
echo ""
echo "  Next step:"
echo "  1. Open tactical-signal-v2.html in your browser"
echo "  2. Click 'Import' (top right)"
echo "  3. Select $OUTPUT_FILE"
echo "  4. All charts + history will update automatically"
echo ""

# Try to open the dashboard
if command -v open &>/dev/null; then
    read -p "Open the dashboard now? [Y/n] " OPEN
    if [ "$OPEN" != "n" ] && [ "$OPEN" != "N" ]; then
        open "tactical-signal-v2.html"
    fi
fi

echo ""
read -p "Press Enter to close..."
