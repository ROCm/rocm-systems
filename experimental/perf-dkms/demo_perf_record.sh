#!/bin/bash
# Demo script: Combined GPU counter sampling + API tracing via perf
#
# This script captures:
# 1. amdgpu_pmu/config=0x23/ (GRBM_COUNT) - hardware performance counter
# 2. amdgpu_pmu:counter_sample - counter time series samples
# 3. amdgpu_pmu:kernel_dispatch - GPU kernel dispatch events (via rocprofv3)
# 4. amdgpu_pmu:hsa_api - HSA API calls (via rocprofv3)
# 5. amdgpu_pmu:hip_api - HIP API calls (via rocprofv3)
#
# Usage: ./demo_perf_record.sh [application] [args...]
# Example: ./demo_perf_record.sh ./test_32_waves
#          ./demo_perf_record.sh ./my_hip_app --iterations 100

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PERF="${PERF:-$(command -v perf 2>/dev/null || true)}"
ROCPROFV3="${ROCPROFV3:-$(command -v rocprofv3 2>/dev/null || true)}"
SUDO_SCRIPT="${SUDO_SCRIPT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/perf_demo_$$}"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

run_root() {
    if [ -n "$SUDO_SCRIPT" ]; then
        sudo "$SUDO_SCRIPT" "$@"
    elif [ "$EUID" -eq 0 ]; then
        "$@"
    else
        sudo "$@"
    fi
}

if [ -n "$SUDO_SCRIPT" ] && [ ! -x "$SUDO_SCRIPT" ]; then
    die "SUDO_SCRIPT is set but not executable: $SUDO_SCRIPT"
fi

if [ -z "$PERF" ]; then
    die "perf not found. Set PERF=/path/to/perf or install perf tools."
fi

if [ -z "$ROCPROFV3" ]; then
    die "rocprofv3 not found. Set ROCPROFV3=/path/to/rocprofv3."
fi

# Default application
APP="${1:-$SCRIPT_DIR/test_32_waves}"
if [ "$#" -gt 0 ]; then
    shift
fi
APP_ARGS=("$@")

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "GPU Performance Recording Demo"
echo "========================================"
echo "Application: $APP ${APP_ARGS[*]}"
echo "Output dir:  $OUTPUT_DIR"
echo ""
echo "Events being captured:"
echo "  - amdgpu_pmu/config=0x23/ (GRBM_COUNT hardware counter)"
echo "  - amdgpu_pmu:counter_sample (counter time series)"
echo "  - amdgpu_pmu:kernel_dispatch (GPU kernel dispatches)"
echo "  - amdgpu_pmu:hsa_api (HSA API calls)"
echo "  - amdgpu_pmu:hip_api (HIP API calls)"
echo "========================================"
echo ""

# Run perf record with all events
# - GRBM_COUNT counter for hardware performance data
# - amdgpu_pmu tracepoints for kernel/API tracing from rocprofv3
run_root env ROCPROF_OUTPUT_FORMAT=PERF_USER_EVENTS \
    "$PERF" record \
    -e amdgpu_pmu/config=0x23/ \
    -e amdgpu_pmu:* \
    -c 1000000 -a \
    -o "$OUTPUT_DIR/perf.data" \
    -- "$ROCPROFV3" --kernel-trace --hsa-trace --hip-trace -- "$APP" "${APP_ARGS[@]}"

echo ""
echo "========================================"
echo "Recording complete. Analyzing results..."
echo "========================================"
echo ""

# Count events by type
echo "--- Event Summary ---"
PERF_OUTPUT=$(run_root "$PERF" script -i "$OUTPUT_DIR/perf.data" 2>/dev/null || true)
TOTAL=$(echo "$PERF_OUTPUT" | wc -l)
KERNEL_DISPATCH=$(echo "$PERF_OUTPUT" | grep -c "kernel_dispatch" || true)
HSA_API=$(echo "$PERF_OUTPUT" | grep -c "hsa_api" || true)
HIP_API=$(echo "$PERF_OUTPUT" | grep -c "hip_api" || true)
COUNTER_SAMPLE=$(echo "$PERF_OUTPUT" | grep -c "counter_sample" || true)

echo "  Kernel Dispatches:      $KERNEL_DISPATCH"
echo "  HSA API Calls:          $HSA_API"
echo "  HIP API Calls:          $HIP_API"
echo "  Counter Samples:        $COUNTER_SAMPLE"
echo "  Total Events:           $TOTAL"
echo ""

# Show kernel dispatch events
echo "--- Kernel Dispatch Events ---"
run_root "$PERF" script -i "$OUTPUT_DIR/perf.data" 2>/dev/null | grep "kernel_dispatch"
echo ""

# Show sample of HSA API events
echo "--- HSA API Events ---"
run_root "$PERF" script -i "$OUTPUT_DIR/perf.data" 2>/dev/null | grep "hsa_api"
echo ""

# Show sample of HIP API events
echo "--- HIP API Events ---"
run_root "$PERF" script -i "$OUTPUT_DIR/perf.data" 2>/dev/null | grep "hip_api"
echo ""

# Show counter samples
echo "--- Counter Samples ---"
run_root "$PERF" script -i "$OUTPUT_DIR/perf.data" 2>/dev/null | grep "counter_sample"
echo ""

echo "========================================"
echo "Output file: $OUTPUT_DIR/perf.data"
echo ""
echo "To view all events:"
echo "  $PERF script -i $OUTPUT_DIR/perf.data"
echo ""
echo "To view specific event types:"
echo "  $PERF script -i $OUTPUT_DIR/perf.data | grep kernel_dispatch"
echo "  $PERF script -i $OUTPUT_DIR/perf.data | grep hsa_api"
echo "  $PERF script -i $OUTPUT_DIR/perf.data | grep hip_api"
echo "========================================"
