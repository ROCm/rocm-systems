#!/bin/bash
# Probe a subset of ggml ops for partial-instrumentation kernels.
# Captures aegisbit stderr logs (via AEGISBIT_LOG=1) and reports which
# ops produce "Partial instrumentation" lines — the kernels replay targets.
#
# Usage: ./probe_partial_ops.sh [--replay auto|N]

AEGIS_DIR=/home/djavady/aegis_two
TEST_BIN=/home/djavady/llama_cpp/build-hip/bin/test-backend-ops
LIB=$AEGIS_DIR/build/src/libaegisbit.so
LOGDIR=/tmp/aegis_ops_probe
mkdir -p "$LOGDIR"

# Parse --replay option
REPLAY_ENV=""
if [ "$1" = "--replay" ] && [ -n "$2" ]; then
    REPLAY_ENV="AEGISBIT_REPLAY=$2"
fi

# Focused subset: ops whose kernels are most likely to be large enough to
# hit the s_branch-range limit (≥ a few hundred sites).  Keeps total
# runtime bounded while still sampling the "hard" parts of the op space.
OPS=(
    FLASH_ATTN_EXT
    MUL_MAT
    MUL_MAT_ID
    IM2COL
    CONV_2D
    CONV_2D_DW
    CONV_TRANSPOSE_2D
    SOFT_MAX
    RMS_NORM
    ROPE
    SSM_SCAN
    SSM_CONV
    GATED_DELTA_NET
    GATED_LINEAR_ATTN
    TOPK_MOE
    GROUP_NORM
    ARGSORT
    CROSS_ENTROPY_LOSS
    POOL_2D
    CUMSUM
)

echo "========================================="
echo "Probing ${#OPS[@]} ops for partial kernels"
echo "Mode: ${REPLAY_ENV:-replay-off (baseline)}"
echo "========================================="

PARTIAL_OPS=()
FULL_OPS=()
NOKERNEL_OPS=()

for op in "${OPS[@]}"; do
    LOG="$LOGDIR/$op.log"
    printf "  %-22s ... " "$op"
    env $REPLAY_ENV \
        AEGISBIT_LOG=1 \
        LD_PRELOAD=$LIB \
        timeout 60 $TEST_BIN -o "$op" >/dev/null 2>"$LOG"

    # Count discovered and instrumented sites across all kernels for this op.
    partial_lines=$(grep -c "Partial instrumentation:" "$LOG" 2>/dev/null; true)
    cfg_lines=$(grep -c "CFG:" "$LOG" 2>/dev/null; true)
    replay_lines=$(grep -c "Replay: " "$LOG" 2>/dev/null; true)
    partial_lines=${partial_lines:-0}
    cfg_lines=${cfg_lines:-0}
    replay_lines=${replay_lines:-0}

    if [ "$cfg_lines" -eq 0 ]; then
        echo "skipped (no kernels patched)"
        NOKERNEL_OPS+=("$op")
        continue
    fi

    if [ "$partial_lines" -eq 0 ]; then
        kernels=$(grep -c "Patching kernel:" "$LOG" 2>/dev/null; true)
        kernels=${kernels:-0}
        echo "full coverage ($kernels kernels, $replay_lines replay summaries)"
        FULL_OPS+=("$op")
        continue
    fi

    # Partial detected — pull the worst-case ratio.
    worst=$(grep -oE "Partial instrumentation: [0-9]+/[0-9]+" "$LOG" \
            | awk -F'[ /]' '{ printf "%d/%d %s\n", $3, $4, $3/$4 }' \
            | sort -k3 -g | head -1 | awk '{print $1}')
    total_partial=$(grep -oE "Partial instrumentation: [0-9]+/[0-9]+" "$LOG" \
            | awk -F'[ /]' 'BEGIN{inst=0; disc=0} { inst+=$3; disc+=$4 } END { printf "%d/%d", inst, disc }')
    echo "PARTIAL (worst $worst, sum $total_partial over $partial_lines variants)"
    PARTIAL_OPS+=("$op")
done

echo ""
echo "========================================="
echo "SUMMARY"
echo "========================================="
echo "Full coverage: ${#FULL_OPS[@]}"
echo "Partial:       ${#PARTIAL_OPS[@]}"
echo "No kernels:    ${#NOKERNEL_OPS[@]}"
echo ""

if [ ${#PARTIAL_OPS[@]} -gt 0 ]; then
    echo "OPS WITH PARTIAL INSTRUMENTATION (replay candidates):"
    for op in "${PARTIAL_OPS[@]}"; do
        echo "  - $op"
    done
fi
