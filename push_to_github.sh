#!/bin/bash
# Push CRANE-X to GitHub
set -e

cd /home/a3/crane-x

# Read token from git-credentials
TOKEN=$(awk -F'[@:]' '{print $3}' ~/.git-credentials)
echo "Token starts with: ${TOKEN:0:10}..."

# Create the repo (if not exists)
CREATE_RESULT=$(curl -s -w "\n%{http_code}" -X POST \
  -H "Authorization: token $TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "craneX",
    "description": "CRANE-X: CPU-Native 3-Signal Sentiment Ensemble with Per-Asset Calibration, Volatility-Normalized Compositing, and LLM-Enhanced Content Understanding for Multi-Asset Financial News Analysis",
    "homepage": "https://tradeflags.com/cranex.html",
    "private": false
  }' \
  https://api.github.com/user/repos)

HTTP_CODE=$(echo "$CREATE_RESULT" | tail -1)
BODY=$(echo "$CREATE_RESULT" | sed '$d')

if [ "$HTTP_CODE" = "201" ]; then
  echo "Repository created: craneX"
elif echo "$BODY" | grep -q "already exists"; then
  echo "Repository craneX already exists"
else
  echo "Create failed ($HTTP_CODE): $(echo $BODY | python3 -c "import sys,json; print(json.load(sys.stdin).get('message',''))")"
  exit 1
fi

# Set up git remote
git remote remove origin 2>/dev/null || true
git remote add origin "https://github.com/a3igner/craneX.git"

# Ensure on main branch
git checkout -B main

# Add and commit
git add -A
git commit -m "Initial release: CRANE-X v1.0.0

CPU-native 3-signal sentiment ensemble for multi-asset financial news analysis."

# Push (credential helper will provide auth)
git push -u origin main

echo ""
echo "PUSHED! https://github.com/a3igner/craneX"
