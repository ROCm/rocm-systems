#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Runs hip_perf_bench with and without race detection (RJ_RACE=1) and
# prints a formatted comparison table with overhead percentages.
#
# Usage:
#   bash scripts/run_perf_matrix.sh [BUILD_DIR] [--runs=N] [--profile]
#
# BUILD_DIR defaults to ~/workspace/builds/$(basename <repo-root>).
# --runs=N   Run each config N times and report mean±std (default: 1).
# --profile  Enable hook profiling (sets RJ_USE_PROFILED_EXECUTION_PLUGIN_GROUP=1).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SOURCE_DIR/../.." && pwd)"
BUILD_DIR=""
NUM_RUNS=1
PROFILE=0

for arg in "$@"; do
    case "$arg" in
        --runs=*) NUM_RUNS="${arg#--runs=}" ;;
        --profile) PROFILE=1 ;;
        *) BUILD_DIR="$arg" ;;
    esac
done

BUILD_DIR="${BUILD_DIR:-$HOME/workspace/builds/$(basename "$REPO_ROOT")}"

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
echo "Build:     $BUILD_DIR"
echo "Runs:      $NUM_RUNS"
echo "Profiling: $([ "$PROFILE" = "1" ] && echo "on" || echo "off")"
echo ""

# --- Run one configuration once, capture PERF lines ---
# Appends millisecond values to ALL_MS["kernel,config"]

declare -A ALL_MS     # ALL_MS["kernel,config"] = "ms1 ms2 ms3 ..."
declare -A PROFILES   # PROFILES["config"] = captured HOOK_PROFILE lines (last run only)

run_once() {
    local label="$1"
    local race_env="$2"

    local env_vars=(
        "LD_PRELOAD=$KMD_SHIM"
        "RJ_CONFIG=$CONFIG"
        "RJ_SCHEMA=$SCHEMA"
        "HSA_ENABLE_SDMA=1"
    )
    if [ "$race_env" = "1" ]; then
        env_vars+=("RJ_RACE=1")
    fi
    if [ "$PROFILE" = "1" ]; then
        env_vars+=("RJ_USE_PROFILED_EXECUTION_PLUGIN_GROUP=1" "RJ_SINKS=stderr")
    fi
    local tmpstderr
    tmpstderr=$(mktemp)

    local output
    output=$(env "${env_vars[@]}" "$BENCH_BIN" 2>"$tmpstderr") || true

    while IFS= read -r line; do
        if [[ "$line" =~ ^PERF\ ([a-z_]+)\ ([0-9.]+) ]]; then
            local kernel="${BASH_REMATCH[1]}"
            local ms="${BASH_REMATCH[2]}"
            local key="$kernel,$label"
            ALL_MS["$key"]="${ALL_MS["$key"]:-} $ms"
        fi
        if [[ "$line" =~ ^CHECK ]]; then
            echo "  $line"
        fi
    done <<< "$output"

    PROFILES["$label"]=$(grep "^HOOK_PROFILE" "$tmpstderr" 2>/dev/null || true)
    rm -f "$tmpstderr"
}

# --- Run all configurations ---

CONFIGS=("no-race" "race")

for cfg in "${CONFIGS[@]}"; do
    race_val=0
    [ "$cfg" = "race" ] && race_val=1
    for ((run = 1; run <= NUM_RUNS; run++)); do
        echo "Running: $cfg (run $run/$NUM_RUNS) ..."
        run_once "$cfg" "$race_val"
    done
done

echo ""

# --- Compute mean and std from space-separated ms values ---

compute_stats() {
    local values="$1"
    awk '{
        n = NF; sum = 0; sumsq = 0
        for (i = 1; i <= n; i++) { sum += $i; sumsq += $i * $i }
        mean = sum / n
        if (n > 1) std = sqrt((sumsq - sum*sum/n) / (n-1))
        else std = 0
        printf "%.1f %.1f", mean, std
    }' <<< "$values"
}

# --- Collect kernel names ---

KERNELS=()
for key in "${!ALL_MS[@]}"; do
    kernel="${key%%,*}"
    found=0
    for k in "${KERNELS[@]+"${KERNELS[@]}"}"; do
        if [ "$k" = "$kernel" ]; then found=1; break; fi
    done
    if [ "$found" = "0" ]; then
        KERNELS+=("$kernel")
    fi
done
IFS=$'\n' KERNELS=($(sort <<<"${KERNELS[*]}")); unset IFS

# --- Format table ---

NAME_W=20

if [ "$NUM_RUNS" -gt 1 ]; then
    COL_W=18
else
    COL_W=14
fi

# Header
printf "%-${NAME_W}s" ""
for cfg in "${CONFIGS[@]}"; do
    printf "%${COL_W}s" "$cfg"
done
printf "%${COL_W}s\n" "overhead"

total_w=$((NAME_W + (${#CONFIGS[@]} + 1) * COL_W))
printf '%*s\n' "$total_w" '' | tr ' ' '-'

# Rows
declare -A MEANS
for kernel in "${KERNELS[@]}"; do
    printf "%-${NAME_W}s" "$kernel"
    for cfg in "${CONFIGS[@]}"; do
        ms_list="${ALL_MS["$kernel,$cfg"]:-}"
        if [ -n "$ms_list" ]; then
            read -r mean std <<< "$(compute_stats "$ms_list")"
            MEANS["$kernel,$cfg"]="$mean"
            secs=$(awk "BEGIN {printf \"%.1f\", $mean / 1000}")
            if [ "$NUM_RUNS" -gt 1 ]; then
                std_secs=$(awk "BEGIN {printf \"%.2f\", $std / 1000}")
                printf "%${COL_W}s" "${secs}±${std_secs}s"
            else
                printf "%${COL_W}s" "${secs}s"
            fi
        else
            printf "%${COL_W}s" "—"
        fi
    done

    # Overhead column
    no_race_mean="${MEANS["$kernel,no-race"]:-}"
    race_mean="${MEANS["$kernel,race"]:-}"
    if [ -n "$no_race_mean" ] && [ -n "$race_mean" ]; then
        overhead=$(awk "BEGIN {printf \"+%.0f%%\", ($race_mean - $no_race_mean) / $no_race_mean * 100}")
        printf "%${COL_W}s" "$overhead"
    else
        printf "%${COL_W}s" "—"
    fi

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
