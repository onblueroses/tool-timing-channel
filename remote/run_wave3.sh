#!/usr/bin/env bash
# Wave 3: Comprehensive channel coverage + 1.5b model + capability ladder.
# Fills VPS time after wave 2 completes.
#
# Estimated: ~8-10 hours on CPU-only VPS.

set -euo pipefail

cd /root/agent-stego
export PATH="$HOME/.local/bin:$PATH"

FINDINGS_BASE="findings"
START_TIME=$(date +%s)
STARTED_AT=$(date -Iseconds)

echo "=============================================="
echo "AGENT-STEGO WAVE 3 - FULL CHANNEL COVERAGE"
echo "Started: $STARTED_AT"
echo "=============================================="
echo ""

# Pull 1.5b model if not present
echo "Ensuring qwen2.5:1.5b is available..."
ollama pull qwen2.5:1.5b 2>/dev/null || true
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
# PHASE 1: 1.5B MODEL BASELINE (~2h)
# Test all new channels on the smallest model
# ============================================================

echo "====== PHASE 1: 1.5B MODEL TESTING ======"

# Table scaling on 1.5b
run_experiment table_scaling \
    "table_scaling model=qwen2.5:1.5b secret=HI trials=15" \
    MODEL="ollama/qwen2.5:1.5b" SECRET="HI" TRIALS=15 SIZES="26,64,128,256"

run_experiment table_scaling \
    "table_scaling model=qwen2.5:1.5b secret=HELLO trials=15" \
    MODEL="ollama/qwen2.5:1.5b" SECRET="HELLO" TRIALS=15 SIZES="26,64,128,256"

# Permutation on 1.5b
run_experiment permutation_order \
    "permutation model=qwen2.5:1.5b" \
    MODELS="ollama/qwen2.5:1.5b" TRIALS=15

# Correction on 1.5b
run_experiment correction_comparison \
    "correction model=qwen2.5:1.5b secret=HI trials=15" \
    MODEL="ollama/qwen2.5:1.5b" SECRET="HI" TRIALS=15

# Original table modes on 1.5b
for MODE in "path" "query" "tri_tool"; do
    run_experiment run_table \
        "table model=qwen2.5:1.5b mode=$MODE trials=10" \
        MODEL="ollama/qwen2.5:1.5b" MODE="$MODE" SECRET="HELLO" TRIALS=10
done

# ============================================================
# PHASE 2: REMAINING CHANNELS (~3h)
# Error, argument, volume_binary channels on all 3 models
# ============================================================

echo "====== PHASE 2: ERROR/ARGUMENT/VOLUME_BINARY CHANNELS ======"

for MODEL in "ollama/qwen2.5:1.5b" "ollama/qwen2.5:3b" "ollama/qwen2.5:7b"; do
    MODEL_SHORT="${MODEL##*/}"

    run_experiment run_volume_binary \
        "volume_binary model=$MODEL_SHORT trials=10" \
        MODEL="$MODEL" SECRET="HI" N_TRIALS=10

    run_experiment run_error \
        "error model=$MODEL_SHORT trials=10" \
        MODEL="$MODEL" SECRET="HI" N_TRIALS=10

    run_experiment run_argument \
        "argument model=$MODEL_SHORT trials=10" \
        MODEL="$MODEL" SECRET="HI" N_TRIALS=10
done

# ============================================================
# PHASE 3: CAPABILITY LADDER (~2h)
# Escalating complexity across model sizes (local models only)
# ============================================================

echo "====== PHASE 3: CAPABILITY LADDER ======"

run_experiment capability_ladder \
    "capability_ladder local models levels=1,2,3,4" \
    MODELS="ollama/qwen2.5:1.5b,ollama/qwen2.5:3b,ollama/qwen2.5:7b" \
    SECRET="HI" LEVELS="1,2,3,4"

# ============================================================
# PHASE 4: STRESS TESTS (~2h)
# Longer messages, more trials on best channels
# ============================================================

echo "====== PHASE 4: STRESS TESTS ======"

# Long messages on table (the strongest channel)
for SECRET in "APIKEY" "PASSWORD" "EXFILTRATE" "THEQUICKBROWN"; do
    run_experiment table_scaling \
        "stress model=qwen2.5:3b secret=$SECRET sizes=26,64" \
        MODEL="ollama/qwen2.5:3b" SECRET="$SECRET" TRIALS=10 SIZES="26,64"
done

# Same on 7b
for SECRET in "APIKEY" "PASSWORD" "EXFILTRATE" "THEQUICKBROWN"; do
    run_experiment table_scaling \
        "stress model=qwen2.5:7b secret=$SECRET sizes=26,64" \
        MODEL="ollama/qwen2.5:7b" SECRET="$SECRET" TRIALS=10 SIZES="26,64"
done

# Permutation stress - longer messages (push max_iterations)
run_experiment permutation_order \
    "permutation stress 3b" \
    MODELS="ollama/qwen2.5:3b" TRIALS=15

run_experiment permutation_order \
    "permutation stress 7b" \
    MODELS="ollama/qwen2.5:7b" TRIALS=15

# ============================================================
# PHASE 5: OVERNIGHT SWEEP 1.5B + EXTENDED 3B (~2h)
# ============================================================

echo "====== PHASE 5: EXTENDED SWEEPS ======"

run_experiment overnight_3b_sweep \
    "overnight_sweep model=1.5b secrets=HI,KEY,PASS trials=8" \
    MODELS="ollama/qwen2.5:1.5b" SECRETS="HI,KEY,PASS" TRIALS=8 CANDIDATES=default

# Extended sweep with more secrets on 3b
run_experiment overnight_3b_sweep \
    "overnight_sweep model=3b secrets=HELLO,SECRET,APIKEY trials=8" \
    MODELS="ollama/qwen2.5:3b" SECRETS="HELLO,SECRET,APIKEY" TRIALS=8 CANDIDATES=default

# ============================================================
# COMPLETION
# ============================================================

END_TIME=$(date +%s)
TOTAL_ELAPSED=$(( END_TIME - START_TIME ))
HOURS=$(( TOTAL_ELAPSED / 3600 ))
MINUTES=$(( (TOTAL_ELAPSED % 3600) / 60 ))

echo ""
echo "=============================================="
echo "WAVE 3 COMPLETE"
echo "Started:  $STARTED_AT"
echo "Finished: $(date -Iseconds)"
echo "Duration: ${HOURS}h ${MINUTES}m (${TOTAL_ELAPSED}s)"
echo "=============================================="

cat > "$FINDINGS_BASE/done.json" <<EOF
{
    "status": "complete",
    "wave": 3,
    "started": "$STARTED_AT",
    "finished": "$(date -Iseconds)",
    "duration_seconds": $TOTAL_ELAPSED,
    "experiments": ["table_scaling_1.5b", "permutation_1.5b", "correction_1.5b", "error", "argument", "volume_binary", "capability_ladder", "stress_tests", "extended_sweeps"],
    "models": ["ollama/qwen2.5:1.5b", "ollama/qwen2.5:3b", "ollama/qwen2.5:7b"]
}
EOF

echo "Results in $FINDINGS_BASE/"
