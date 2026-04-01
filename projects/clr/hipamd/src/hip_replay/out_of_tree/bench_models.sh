#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. - MIT License
#
# Benchmark ONNX models with MIGraphX (native) and HRR (kernel-only replay).
#
# Usage: ./bench_models.sh [model1.onnx model2.onnx ...]
# With no args, benchmarks light_{resnet50,squeezenet,densenet121}.onnx.
#
# Environment:
#   HRR_DIR      - directory containing hrr_test workspace (default: detected)
#   HRR_ITERS    - replay iterations (default: 100)
#   HRR_WARMUP   - replay warmup (default: 10)
#   MGX_ITERS    - migraphx-driver perf iterations (default: 50)
#   FORCE        - set to 1 to recompile .mxr files even if they exist

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Locate hrr_test workspace (contains built tools)
: "${HRR_DIR:=$HOME/hrr_test}"

LIBHRR="$HRR_DIR/out_of_tree/libhrr_record.so"
BENCH="$HRR_DIR/src/build/hrr-bench"
RECORD_SCRIPT="$HRR_DIR/out_of_tree/record_model.py"

# Light ONNX models bundled with the onnx package
LIGHT_DIR="$(python3 -c \
  "import onnx, os; print(os.path.join(os.path.dirname(onnx.__file__), \
   'backend/test/data/light'))" 2>/dev/null || true)"

: "${HRR_ITERS:=100}"
: "${HRR_WARMUP:=10}"
: "${MGX_ITERS:=50}"
: "${FORCE:=0}"

# Resolve model list
if [[ $# -gt 0 ]]; then
  MODELS=("$@")
else
  if [[ -z "$LIGHT_DIR" || ! -d "$LIGHT_DIR" ]]; then
    echo "ERROR: Could not find onnx light models. Pass model paths explicitly." >&2
    exit 1
  fi
  MODELS=(
    "$LIGHT_DIR/light_resnet50.onnx"
    "$LIGHT_DIR/light_squeezenet.onnx"
    "$LIGHT_DIR/light_densenet121.onnx"
  )
fi

# Sanity checks
for tool in "$LIBHRR" "$BENCH" "$RECORD_SCRIPT"; do
  if [[ ! -f "$tool" ]]; then
    echo "ERROR: Missing: $tool" >&2
    exit 1
  fi
done

GPU_NAME=$(python3 -c "
import subprocess, re
r = subprocess.run(['rocminfo'], capture_output=True, text=True)
for line in r.stdout.splitlines():
    if 'Marketing' in line:
        print(line.split(':', 1)[-1].strip())
        break
" 2>/dev/null || echo "unknown GPU")

echo "============================================================"
echo "  HRR Model Benchmark"
printf "  GPU:       %s\n" "$GPU_NAME"
echo "  MIGraphX:  $(migraphx-driver --version 2>&1 | grep -oP 'Version: \K\S+' || echo '?')"
printf "  HRR:       %d iters, %d warmup\n" "$HRR_ITERS" "$HRR_WARMUP"
printf "  MGX perf:  %d iters\n" "$MGX_ITERS"
echo "============================================================"
echo ""

# --------------- Phase 1: compile all models (no HRR) ---------------
echo "Phase 1: Compiling models..."
for MODEL_PATH in "${MODELS[@]}"; do
  MODEL_NAME="$(basename "$MODEL_PATH" .onnx)"
  MXR_PATH="${MODEL_PATH%.onnx}.mxr"
  if [[ -f "$MXR_PATH" && "$FORCE" != "1" ]]; then
    echo "  $MODEL_NAME: already compiled ($MXR_PATH)"
  else
    echo "  $MODEL_NAME: compiling..."
    python3 "$RECORD_SCRIPT" --compile "$MODEL_PATH" 2>&1 | sed 's/^/    /'
  fi
done
echo ""

# --------------- Phase 2: record + benchmark each model ---------------
printf "%-20s  %8s  %9s  %9s  %9s  %9s  %7s\n" \
  "Model" "MGX" "HRR_min" "HRR_med" "HRR_p95" "HRR_mean" "kernels"
printf "%-20s  %8s  %9s  %9s  %9s  %9s  %7s\n" \
  "--------------------" "-------" "---------" "---------" "---------" "---------" "-------"

for MODEL_PATH in "${MODELS[@]}"; do
  MODEL_NAME="$(basename "$MODEL_PATH" .onnx)"
  DISPLAY_NAME="${MODEL_NAME#light_}"
  CAPTURE="$HRR_DIR/${DISPLAY_NAME}.hrr"

  echo "" >&2
  echo "=== $DISPLAY_NAME ===" >&2

  # --- Record one clean inference from compiled model ---
  echo "  Recording..." >&2
  rm -rf "$CAPTURE"
  HRR_RECORD=1 HRR_OUTPUT="$CAPTURE" \
    LD_PRELOAD="$LIBHRR" \
    python3 "$RECORD_SCRIPT" --record "$MODEL_PATH" 2>&1 | sed 's/^/    /' >&2

  if [[ ! -d "$CAPTURE" ]]; then
    echo "  ERROR: capture not created — skipping" >&2
    printf "%-20s  %8s  %8s  %8s  %8s  %8s  %7s\n" \
      "$DISPLAY_NAME" "FAIL" "?" "?" "?" "?" "?"
    continue
  fi

  KERNEL_COUNT=$("$BENCH" list "$CAPTURE" 2>/dev/null | tail -n +3 | grep -c '.' || echo "?")

  # --- MIGraphX native perf (measures end-to-end latency incl. framework) ---
  echo "  MIGraphX native perf ($MGX_ITERS iters)..." >&2
  MGX_OUTPUT=$(migraphx-driver perf --onnx --gpu -n "$MGX_ITERS" "$MODEL_PATH" 2>&1)
  echo "$MGX_OUTPUT" | grep -E 'Rate|Batch|batch_size' | sed 's/^/    /' >&2
  MGX_RATE=$(echo "$MGX_OUTPUT" | grep -oP 'Rate: \K[\d.]+' | head -1 || echo "")
  if [[ -n "$MGX_RATE" && "$MGX_RATE" != "0" ]]; then
    MGX_MS=$(python3 -c "print(f'{1000.0/$MGX_RATE:.3f}')" 2>/dev/null || echo "?")
  else
    MGX_MS="?"
  fi

  # --- HRR bench app (measures pure GPU kernel time, no framework overhead) ---
  echo "  HRR bench app ($HRR_ITERS iters)..." >&2
  HRR_OUT=$("$BENCH" app "$CAPTURE" --iterations "$HRR_ITERS" --warmup "$HRR_WARMUP" 2>&1)
  echo "$HRR_OUT" | grep -E 'Kernels|Min|Median|Mean|P95|Throughput' | sed 's/^/    /' >&2

  # Extract value+unit (ms or us)
  HRR_UNIT=$(echo "$HRR_OUT" | grep -oP 'Min:\s+[\d.]+\s+\K(ms|us)' | head -1 || echo "ms")
  HRR_MIN=$(echo  "$HRR_OUT" | grep -oP 'Min:\s+\K[\d.]+' || echo "?")
  HRR_MED=$(echo  "$HRR_OUT" | grep -oP 'Median:\s+\K[\d.]+' || echo "?")
  HRR_P95=$(echo  "$HRR_OUT" | grep -oP 'P95:\s+\K[\d.]+' || echo "?")
  HRR_MEAN=$(echo "$HRR_OUT" | grep -oP 'Mean:\s+\K[\d.]+' || echo "?")

  printf "%-20s  %8s  %9s  %9s  %9s  %9s  %7s\n" \
    "$DISPLAY_NAME" "${MGX_MS}ms" "${HRR_MIN}${HRR_UNIT}" "${HRR_MED}${HRR_UNIT}" \
    "${HRR_P95}${HRR_UNIT}" "${HRR_MEAN}${HRR_UNIT}" "$KERNEL_COUNT"
done

echo ""
echo "MGX = MIGraphX end-to-end latency (includes framework overhead)"
echo "HRR = pure GPU kernel execution (no framework, no CPU dispatch latency)"
