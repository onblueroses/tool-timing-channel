#!/usr/bin/env bash
set -euo pipefail

export MODELS="${MODELS:-ollama/qwen2.5:3b}"
export TRIALS="${TRIALS:-6}"
export BASELINE_TRIALS="${BASELINE_TRIALS:-4}"
export CANDIDATES="${CANDIDATES:-default}"
export SECRETS="${SECRETS:-SEARCH,SAFETY,SIGNAL}"
export MAX_ITERATIONS="${MAX_ITERATIONS:-20}"

uv run python experiments/overnight_3b_sweep.py
