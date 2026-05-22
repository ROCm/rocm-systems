#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Runs hip_perf_bench with and without race detection (RJ_RACE=1) and
# prints a formatted comparison table.
#
# Usage:
#   bash scripts/run_perf_matrix.sh [BUILD_DIR]
#
# BUILD_DIR defaults to ~/workspace/builds/$(basename <repo-root>).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SOURCE_DIR/../.." && pwd)"
BUILD_DIR="${1:-$HOME/workspace/builds/$(basename "$REPO_ROOT")}"

BENCH_BIN="$BUILD_DIR/tests/hip_perf_bench"
KMD_SHIM="$BUILD_DIR/lib/rocjitsu/src/rocjitsu/kmd/librocjitsu_kmd.so"
CONFIG="$SOURCE_DIR/configs/amdgpu_cdna4_kmd.json"
SCHEMA="$SOURCE_DIR/schemas/simulation_config.fbs"

# --- Validation ---

if [ ! -x "$BENCH_BIN" ]; then
    echo "Error: benchmark binary not found at $BENCH_BIN"
    echo "Build it with: cmake --build $BUILD_DIR --target hip_perf_bench_target -j\$(nproc)"
    exit 1
fi

if [ ! -f "$KMD_SHIM" ]; then
    KMD_SHIM="$(find "$BUILD_DIR" -name 'librocjitsu_kmd.so' -print -quit 2>/dev/null || true)"
    if [ -z "$KMD_SHIM" ] || [ ! -f "$KMD_SHIM" ]; then
        echo "Error: librocjitsu_kmd.so not found in $BUILD_DIR"
        exit 1
    fi
fi

echo "Benchmark: $BENCH_BIN"
echo "KMD shim:  $KMD_SHIM"
echo "Source:    $SOURCE_DIR"
echo "Build:     $BUILD_DIR"
echo ""

# --- Run one configuration and capture PERF lines ---

declare -A RESULTS  # RESULTS["kernel,config"] = milliseconds
declare -A PROFILES # PROFILES["config"] = captured HOOK_PROFILE lines

run_config() {
    local label="$1"
    local race_env="$2"

    echo "Running: $label ..."

    local env_vars=(
        "LD_PRELOAD=$KMD_SHIM"
        "RJ_CONFIG=$CONFIG"
        "RJ_SCHEMA=$SCHEMA"
        "HSA_ENABLE_SDMA=1"
    )
    if [ "$race_env" = "1" ]; then
        env_vars+=("RJ_RACE=1")
    fi

    local tmpstderr
    tmpstderr=$(mktemp)

    local output
    output=$(env "${env_vars[@]}" "$BENCH_BIN" 2>"$tmpstderr") || true

    while IFS= read -r line; do
        if [[ "$line" =~ ^PERF\ ([a-z_]+)\ ([0-9.]+) ]]; then
            local kernel="${BASH_REMATCH[1]}"
            local ms="${BASH_REMATCH[2]}"
            RESULTS["$kernel,$label"]="$ms"
        fi
        if [[ "$line" =~ ^CHECK ]]; then
            echo "  $line"
        fi
    done <<< "$output"

    PROFILES["$label"]=$(grep "^HOOK_PROFILE" "$tmpstderr" 2>/dev/null || true)
    rm -f "$tmpstderr"
}

# --- Run all configurations ---

CONFIGS=()

run_config "no-race" "0"
CONFIGS+=("no-race")

run_config "race" "1"
CONFIGS+=("race")

echo ""

# --- Collect kernel names (preserving order) ---

KERNELS=()
for key in "${!RESULTS[@]}"; do
    kernel="${key%%,*}"
    found=0
    for k in "${KERNELS[@]+"${KERNELS[@]}"}"; do
        if [ "$k" = "$kernel" ]; then found=1; break; fi
    done
    if [ "$found" = "0" ]; then
        KERNELS+=("$kernel")
    fi
done

# Sort kernels for stable output
IFS=$'\n' KERNELS=($(sort <<<"${KERNELS[*]}")); unset IFS

# --- Format table ---

NAME_W=20
COL_W=14

printf "%-${NAME_W}s" ""
for cfg in "${CONFIGS[@]}"; do
    printf "%${COL_W}s" "$cfg"
done
printf "\n"

total_w=$((NAME_W + ${#CONFIGS[@]} * COL_W))
printf '%*s\n' "$total_w" '' | tr ' ' '-'

for kernel in "${KERNELS[@]}"; do
    printf "%-${NAME_W}s" "$kernel"
    for cfg in "${CONFIGS[@]}"; do
        ms="${RESULTS["$kernel,$cfg"]:-—}"
        if [ "$ms" != "—" ]; then
            secs=$(awk "BEGIN {printf \"%.1f\", $ms / 1000}")
            printf "%${COL_W}s" "${secs}s"
        else
            printf "%${COL_W}s" "—"
        fi
    done
    printf "\n"
done

echo ""
echo "Times are wall-clock seconds (kernel launch + sync, excluding memcpy)."

# --- Hook profiling output ---

has_profiles=0
for cfg in "${CONFIGS[@]}"; do
    if [ -n "${PROFILES["$cfg"]:-}" ]; then has_profiles=1; break; fi
done

if [ "$has_profiles" = "1" ]; then
    echo ""
    echo "=== Hook profiling ==="
    for cfg in "${CONFIGS[@]}"; do
        profile="${PROFILES["$cfg"]:-}"
        if [ -n "$profile" ]; then
            echo ""
            echo "[$cfg]"
            echo "$profile"
        fi
    done
fi
