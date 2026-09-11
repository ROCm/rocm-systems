#!/usr/bin/env bash
# Re-profile AIPROFCOMP-78/865 CPX golden workloads on MI300X (gfx942).
# Run on alola via: srun --reservation=cpx -w <node> --gres=gpu:gfx942-mi300x:8 ...
set -euo pipefail

WORKDIR="${WORKDIR:-$HOME/aiprofcomp78}"
PROJ="${PROJ:-$WORKDIR/rocm-systems/projects/rocprofiler-compute}"
# Default: grouping coalesce fix (#10912). Override with AIPROFCOMP-865 branch if merged.
GIT_BRANCH="${GIT_BRANCH:-users/feizheng10/fix-counter-grouping-pack}"
NODE="${NODE:-ctr-cx71-mi300x-22}"
RESERVATION="${RESERVATION:-cpx}"
OUT_TAG="${OUT_TAG:-aiprofcomp865-$(date +%Y%m%d)}"
ANALYZE_DIR="${ANALYZE_DIR:-$WORKDIR/$OUT_TAG}"

module load ubuntu-24 rocm/7.15.0.dev.df64a75
export PATH="${ROCM_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"

cd "$WORKDIR/rocm-systems"
git fetch origin "${GIT_BRANCH}"
git checkout "${GIT_BRANCH}"
git pull --ff-only origin "${GIT_BRANCH}" || true

cd "$PROJ"
if [[ -f "$WORKDIR/venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$WORKDIR/venv/bin/activate"
fi
export PYTHONPATH="$PROJ/src:${PYTHONPATH:-}"

# Optional: gfx942 WGM priority from AIPROFCOMP-865 when not yet on GIT_BRANCH.
POLICY="$PROJ/src/rocprof_compute_soc/analysis_configs/profiling_counter_grouping_policy.yaml"
if ! grep -q '"6.1.2"' "$POLICY" 2>/dev/null; then
  echo "Note: 6.1.2 not in grouping policy on this branch (non-fatal)."
fi

echo "=== Code ==="
git log -1 --oneline
python3 -c "import rocprof_compute; print('rocprof_compute OK')"

echo "=== Partition ==="
rocm-smi --showcomputepartition --showmempartition 2>&1 | head -12

mkdir -p "$ANALYZE_DIR"

run_workload() {
  local name="$1"
  local setup_cmd="$2"
  local profile_cmd="$3"
  local analyze_log="$ANALYZE_DIR/analyze_${name}_cpx.log"
  local metrics_log="$ANALYZE_DIR/metrics_${name}_cpx.txt"

  echo ""
  echo "========== PROFILE: $name =========="
  eval "$setup_cmd"
  eval "$profile_cmd"

  local wl_path
  wl_path=$(find "workloads/${name}" -maxdepth 2 -name sysinfo.csv | head -1 | xargs dirname)
  if [[ -z "${wl_path}" ]]; then
    echo "ERROR: no workload output for $name"
    return 1
  fi
  echo "Workload path: $wl_path"

  rocprof-compute analyze -p "$wl_path/" -b 3.1.63 3.1.64 6.1.2 17.2.1 17.2.5 11.2.3 \
    >"$analyze_log" 2>&1

  {
    echo "# $name $(date -Iseconds)"
    grep -E "HBM Read Traffic|HBM Write|Workgroup Manager Utilization|VALU Utilization" \
      "$analyze_log" || true
    echo "--- over 100% (Percent rows) ---"
    grep Percent "$analyze_log" | grep -E "10[0-9]\.|1[1-9][0-9]\." || echo "(none)"
  } | tee "$metrics_log"

  echo "Pass count:"
  ls -1 "$wl_path/perfmon/"*.txt 2>/dev/null | wc -l
}

# occupancy sample
cd "$PROJ/sample"
hipcc -O3 occupancy.hip -o occupancy
cd "$PROJ"
run_workload occupancy_cpx \
  'rm -rf workloads/occupancy_cpx' \
  'rocprof-compute profile -n occupancy_cpx -VV --overwrite -- ./sample/occupancy'

# rocflop (build if missing)
if [[ ! -x "$PROJ/sample/rocflop" ]]; then
  echo "Build rocflop (adjust path if your tree differs)"
fi
if [[ -x "$PROJ/sample/rocflop" ]]; then
  run_workload rocflop \
    'rm -rf workloads/rocflop' \
    'rocprof-compute profile -n rocflop -VV --overwrite -- ./sample/rocflop'
fi

# mat_exp from HPCTrainingExamples if present
MAT_EXP_BIN="${MAT_EXP_BIN:-$WORKDIR/HPCTrainingExamples/HPCTrainingExamples/ManagedMemory/streams_sync/build/mat_exp}"
if [[ -x "$MAT_EXP_BIN" ]]; then
  run_workload mat_exp \
    'rm -rf workloads/mat_exp' \
    "rocprof-compute profile -n mat_exp -VV --overwrite -- $MAT_EXP_BIN"
else
  echo "SKIP mat_exp: set MAT_EXP_BIN to streams_sync mat_exp binary"
fi

echo ""
echo "=== Done. Artifacts: $ANALYZE_DIR ==="
ls -la "$ANALYZE_DIR"
