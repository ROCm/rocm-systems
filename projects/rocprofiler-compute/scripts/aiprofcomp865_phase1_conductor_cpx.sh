#!/usr/bin/env bash
# AIPROFCOMP-865 Phase 1 — CPX re-profile on Conductor (not alola).
#
# Host: hpe-darkstar-ccs-aus-e12-03.cs-aus.dcgpu (or set CONDUCTOR_HOST)
# Use a GPU index in CPX (see rocm-smi --showcomputepartition); default GPU 1.
#
# Prereq: clone rocm-systems @ rocprofiler-compute-develop under WORK_ROOT, e.g.:
#   git clone --depth 1 -b rocprofiler-compute-develop --filter=blob:none \
#     --sparse https://github.com/ROCm/rocm-systems.git "$WORK_ROOT/rocm-systems"
#   cd "$WORK_ROOT/rocm-systems" && git sparse-checkout set projects/rocprofiler-compute
#
set -euo pipefail

CONDUCTOR_HOST="${CONDUCTOR_HOST:-hpe-darkstar-ccs-aus-e12-03.cs-aus.dcgpu}"
WORK_ROOT="${WORK_ROOT:-/home/AMD/feizheng/aiprofcomp78}"
REPO="${REPO:-$WORK_ROOT/rocm-systems}"
PROJ="${PROJ:-$REPO/projects/rocprofiler-compute}"
BRANCH="${BRANCH:-rocprofiler-compute-develop}"
# CPX die on Conductor (GPU[1] and GPU[9] are CPX in typical layout)
ROCR_GPU="${ROCR_GPU:-1}"
OUT_TAG="${OUT_TAG:-phase1-develop-$(date +%Y%m%d_%H%M%S)}"
ARTIFACT_DIR="${ARTIFACT_DIR:-$PROJ/workloads/conductor_phase1_$OUT_TAG}"
LOG="${LOG:-$WORK_ROOT/conductor_phase1_$OUT_TAG.log}"

export PATH=/opt/rocm/bin:${PATH:-}
export LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH:-}
export ROCR_VISIBLE_DEVICES="$ROCR_GPU"
export HIP_VISIBLE_DEVICES=0

exec > >(tee "$LOG") 2>&1

echo "=== Phase 1 Conductor CPX ==="
echo "host=$(hostname) branch=$BRANCH rocr=$ROCR_GPU start=$(date -Is)"

if [[ ! -d "$REPO/.git" ]]; then
  echo "ERROR: missing $REPO — clone rocm-systems ($BRANCH) first."
  exit 1
fi

cd "$REPO"
git fetch origin "$BRANCH"
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH" || true
# Sparse clone often omits vendored blobs; materialize from git or rsync from dev machine.
if [[ ! -f "$PROJ/src/vendored/pyyaml/lib/yaml/__init__.py" ]]; then
  git checkout HEAD -- projects/rocprofiler-compute/src/vendored 2>/dev/null || true
fi
if [[ ! -f "$PROJ/src/vendored/pyyaml/lib/yaml/__init__.py" ]]; then
  echo "ERROR: missing src/vendored (rsync from workstation or use full clone)."
  exit 1
fi
git log -1 --oneline

cd "$PROJ"
python3 -m venv "$WORK_ROOT/venv-phase1" 2>/dev/null || true
# shellcheck disable=SC1091
source "$WORK_ROOT/venv-phase1/bin/activate"
pip install -q -r requirements.txt
export PYTHONPATH="$PROJ/src:${PYTHONPATH:-}"

if [[ -x /opt/rocm/libexec/rocprofiler-compute/rocprof-compute ]]; then
  rocprof() {
    PYTHONPATH="$PROJ/src:${PYTHONPATH:-}" /opt/rocm/libexec/rocprofiler-compute/rocprof-compute "$@"
  }
else
  rocprof() { python3 "$PROJ/src/rocprof-compute" "$@"; }
fi

echo "=== Partition (GPU $ROCR_GPU) ==="
rocm-smi --showcomputepartition 2>&1 | grep -E "GPU\[$ROCR_GPU\]|Partition" | head -5 || true

cd "$PROJ/sample"
hipcc -O3 occupancy.hip -o occupancy
cd "$PROJ"
mkdir -p "$(dirname "$ARTIFACT_DIR")"

echo "=== Profile (full panel, no-roof) ==="
rocprof profile --no-roof --no-native-tool -VV \
  --output-directory "$ARTIFACT_DIR" -- ./sample/occupancy

WL="$ARTIFACT_DIR"
if [[ -f "$ARTIFACT_DIR/0/sysinfo.csv" ]]; then
  WL="$ARTIFACT_DIR/0"
fi

echo "=== sysinfo ==="
grep -E "compute_partition|memory_partition|num_xcd|cu_per_gpu" "$WL/sysinfo.csv" || true
echo "perfmon passes: $(ls -1 "$WL/perfmon/"*.txt 2>/dev/null | wc -l)"

METRICS_LOG="$WORK_ROOT/metrics_${OUT_TAG}.txt"
rocprof analyze --path "$WL" \
  -b 3.1.63 3.1.64 6.1.2 17.2.1 17.2.5 11.2.3 \
  >"$METRICS_LOG" 2>&1

echo "=== Golden metrics ==="
grep -E "HBM Read Traffic|HBM Write|Workgroup Manager|VALU Utilization" "$METRICS_LOG" || true
echo "--- Percent > 100 ---"
grep Percent "$METRICS_LOG" | grep -E "10[0-9]\.|1[1-9][0-9]\." || echo "(none)"

echo "=== Done ==="
echo "workload=$WL log=$LOG metrics=$METRICS_LOG"
