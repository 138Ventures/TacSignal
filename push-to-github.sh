#!/bin/bash
# ═══════════════════════════════════════════════════════════
# Push TacSignal to GitHub Pages
# Target: https://138ventures.github.io/TacSignal/
# ═══════════════════════════════════════════════════════════

set -e

REPO="138ventures/TacSignal"
REMOTE_URL="git@github.com:${REPO}.git"
# If SSH doesn't work, try HTTPS instead:
# REMOTE_URL="https://github.com/${REPO}.git"

cd "$(dirname "$0")"

echo "═══════════════════════════════════════"
echo "  TacSignal → GitHub Pages"
echo "═══════════════════════════════════════"
echo ""

# Check if remote already exists
if git remote get-url origin >/dev/null 2>&1; then
    echo "✓ Remote 'origin' already set: $(git remote get-url origin)"
else
    echo "Adding remote: $REMOTE_URL"
    git remote add origin "$REMOTE_URL"
fi

echo ""
echo "Files to push:"
echo "  $(git log --oneline -1)"
echo "  $(git diff --stat HEAD~1 HEAD 2>/dev/null || echo '12 files, initial commit')"
echo ""

# Push
echo "Pushing to $REPO (main branch)..."
git push -u origin main

echo ""
echo "═══════════════════════════════════════"
echo "  ✓ Done!"
echo ""
echo "  Live at: https://138ventures.github.io/TacSignal/"
echo ""
echo "  If this is a new repo, enable GitHub Pages:"
echo "  1. Go to https://github.com/${REPO}/settings/pages"
echo "  2. Source: Deploy from a branch"
echo "  3. Branch: main / root"
echo "  4. Save"
echo "═══════════════════════════════════════"
