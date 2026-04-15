#!/usr/bin/env bash
# Wave 2: Extended experiment suite. Runs after wave 1 completes.
# Fills remaining VPS time with deeper analysis across all channels.
#
# Estimated: ~10 hours on CPU-only VPS with qwen2.5 3b/7b.

set -euo pipefail

cd /root/agent-stego
export PATH="$HOME/.local/bin:$PATH"

FINDINGS_BASE="findings"
START_TIME=$(date +%s)
STARTED_AT=$(date -Iseconds)

echo "=============================================="
echo "AGENT-STEGO WAVE 2 - EXTENDED EXPERIMENTS"
echo "Started: $STARTED_AT"
echo "=============================================="
echo ""

run_experiment() {
    local name="$1"
    local desc="$2"
    shift 2
    local exp_start=$(date +%s)
    echo "----------------------------------------------"
    echo "[$(date -Iseconds)] START: $desc"
    echo "----------------------------------------------"
    env "$@" uv run python "experiments/${name}.py" || echo "  WARNING: $name exited non-zero"
    local exp_end=$(date +%s)
    local elapsed=$(( exp_end - exp_start ))
    echo ""
    echo "[$(date -Iseconds)] DONE: $desc (${elapsed}s)"
    echo ""
}

# ============================================================
# PHASE 1: TABLE DEEP DIVE (~2h)
# Mode comparison + message length stress test
# ============================================================

echo "====== PHASE 1: TABLE DEEP DIVE ======"

for MODEL in "ollama/qwen2.5:3b" "ollama/qwen2.5:7b"; do
    MODEL_SHORT="${MODEL##*/}"
    run_experiment table_deep_dive \
        "table_deep_dive model=$MODEL_SHORT trials=10" \
        MODEL="$MODEL" TRIALS=10 BASELINE=4
done

# ============================================================
# PHASE 2: TABLE CORRECTION (~1.5h)
# RepetitionCode(3) and RepetitionCode(5) on longer messages
# ============================================================

echo "====== PHASE 2: TABLE CORRECTION ======"

for MODEL in "ollama/qwen2.5:3b" "ollama/qwen2.5:7b"; do
    MODEL_SHORT="${MODEL##*/}"
    run_experiment table_correction \
        "table_correction model=$MODEL_SHORT trials=10" \
        MODEL="$MODEL" TRIALS=10
done

# ============================================================
# PHASE 3: OVERNIGHT 3B SWEEP (~3h)
# Multi-channel sweep with varied secrets
# ============================================================

echo "====== PHASE 3: OVERNIGHT SWEEP ======"

run_experiment overnight_3b_sweep \
    "overnight_3b_sweep models=3b secrets=HI,HELLO,KEY trials=10" \
    MODELS="ollama/qwen2.5:3b" SECRETS="HI,HELLO,KEY" TRIALS=10 CANDIDATES=default

run_experiment overnight_3b_sweep \
    "overnight_3b_sweep models=7b secrets=HI,HELLO,KEY trials=10" \
    MODELS="ollama/qwen2.5:7b" SECRETS="HI,HELLO,KEY" TRIALS=10 CANDIDATES=default

# ============================================================
# PHASE 4: EXISTING CHANNEL EXPERIMENTS (~2h)
# Run the individual channel experiments we haven't tested
# ============================================================

echo "====== PHASE 4: INDIVIDUAL CHANNELS ======"

# Volume channel
for MODEL in "ollama/qwen2.5:3b" "ollama/qwen2.5:7b"; do
    MODEL_SHORT="${MODEL##*/}"
    run_experiment run_volume \
        "volume model=$MODEL_SHORT trials=10" \
        MODEL="$MODEL" SECRET="HI" TRIALS=10
done

# Ordering channel
for MODEL in "ollama/qwen2.5:3b" "ollama/qwen2.5:7b"; do
    MODEL_SHORT="${MODEL##*/}"
    run_experiment run_ordering \
        "ordering model=$MODEL_SHORT trials=10" \
        MODEL="$MODEL" SECRET="HI" TRIALS=10
done

# Table multiturn
for MODEL in "ollama/qwen2.5:3b" "ollama/qwen2.5:7b"; do
    MODEL_SHORT="${MODEL##*/}"
    run_experiment table_multiturn \
        "table_multiturn model=$MODEL_SHORT trials=10" \
        MODEL="$MODEL" SECRET="HELLO" TRIALS=10
done

# Original table modes (path, query, tri_tool)
for MODEL in "ollama/qwen2.5:3b" "ollama/qwen2.5:7b"; do
    MODEL_SHORT="${MODEL##*/}"
    for MODE in "path" "query" "tri_tool"; do
        run_experiment run_table \
            "table model=$MODEL_SHORT mode=$MODE trials=15" \
            MODEL="$MODEL" MODE="$MODE" SECRET="HELLO" TRIALS=15
    done
done

# ============================================================
# PHASE 5: SCALED TABLE - FINE-GRAINED SIZES (~1.5h)
# Zoom in on the sweet spot found in wave 1
# ============================================================

echo "====== PHASE 5: FINE-GRAINED TABLE SIZES ======"

for MODEL in "ollama/qwen2.5:3b" "ollama/qwen2.5:7b"; do
    MODEL_SHORT="${MODEL##*/}"
    # Zoom in around the 64 sweet spot for 3b
    run_experiment table_scaling \
        "table_scaling model=$MODEL_SHORT secret=HELLO fine-grained sizes" \
        MODEL="$MODEL" SECRET="HELLO" TRIALS=15 SIZES="32,48,64,80,96"

    run_experiment table_scaling \
        "table_scaling model=$MODEL_SHORT secret=SECRETKEY fine-grained sizes" \
        MODEL="$MODEL" SECRET="SECRETKEY" TRIALS=15 SIZES="32,48,64,80,96"
done

# ============================================================
# COMPLETION
# ============================================================

END_TIME=$(date +%s)
TOTAL_ELAPSED=$(( END_TIME - START_TIME ))
HOURS=$(( TOTAL_ELAPSED / 3600 ))
MINUTES=$(( (TOTAL_ELAPSED % 3600) / 60 ))

echo ""
echo "=============================================="
echo "WAVE 2 COMPLETE"
echo "Started:  $STARTED_AT"
echo "Finished: $(date -Iseconds)"
echo "Duration: ${HOURS}h ${MINUTES}m (${TOTAL_ELAPSED}s)"
echo "=============================================="

# Update completion marker
cat > "$FINDINGS_BASE/done.json" <<EOF
{
    "status": "complete",
    "wave": 2,
    "started": "$STARTED_AT",
    "finished": "$(date -Iseconds)",
    "duration_seconds": $TOTAL_ELAPSED,
    "experiments": ["table_deep_dive", "table_correction", "overnight_3b_sweep", "volume", "ordering", "table_multiturn", "run_table", "table_scaling_fine"],
    "models": ["ollama/qwen2.5:3b", "ollama/qwen2.5:7b"]
}
EOF

echo "Results in $FINDINGS_BASE/"

# Chain into wave 3
echo ""
echo "Launching wave 3..."
exec bash remote/run_wave3.sh
