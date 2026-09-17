#!/usr/bin/env bash
# AIPROFCOMP-865 — Compare profile/analyze with coalesce refill on vs off (MI300 CPX).
#
# Run on Conductor (same host as phase1 script). Expect ROCR_GPU=9 for 38-CU CPX die.
#
# Usage (on Conductor):
#   BRANCH=users/feizheng10/aiprofcomp-865-single-run-collection \
#     bash projects/rocprofiler-compute/scripts/aiprofcomp865_refill_verify_conductor_cpx.sh
#
set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/home/AMD/feizheng/aiprofcomp78}"
REPO="${REPO:-$WORK_ROOT/rocm-systems}"
PROJ="${PROJ:-$REPO/projects/rocprofiler-compute}"
BRANCH="${BRANCH:-users/feizheng10/aiprofcomp-865-single-run-collection}"
ROCR_GPU="${ROCR_GPU:-9}"
ROCM_ROOT="${ROCM_ROOT:-/cluster/apps/ubuntu-24/rocm/rocm-7.15.0.dev.df64a75}"
SKIP_GIT="${SKIP_GIT:-0}"
OUT_TAG="${OUT_TAG:-refill_verify_$(date +%Y%m%d_%H%M%S)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$PROJ/workloads/refill_verify_$OUT_TAG}"
METRIC_BLOCKS="${METRIC_BLOCKS:-3.1.63 3.1.64 6.1.2 17.2.1 17.2.5}"

export ROCM_PATH="$ROCM_ROOT"
export PATH="$ROCM_ROOT/bin:${PATH:-}"
export LD_LIBRARY_PATH="$ROCM_ROOT/lib:${ROCM_ROOT}/lib64:${LD_LIBRARY_PATH:-}"
export ROCR_VISIBLE_DEVICES="$ROCR_GPU"
export HIP_VISIBLE_DEVICES=0

mkdir -p "$ARTIFACT_ROOT"
LOG="$ARTIFACT_ROOT/run.log"
exec > >(tee "$LOG") 2>&1

echo "=== Refill verify CPX ==="
echo "host=$(hostname) branch=$BRANCH rocr=$ROCR_GPU start=$(date -Is)"

if [[ "$SKIP_GIT" != "1" ]]; then
  cd "$REPO"
  git fetch origin "$BRANCH"
  git checkout "$BRANCH"
  git pull --ff-only origin "$BRANCH" || true
  if [[ ! -f "$PROJ/src/vendored/pyyaml/lib/yaml/__init__.py" ]]; then
    git checkout HEAD -- projects/rocprofiler-compute/src/vendored 2>/dev/null || true
  fi
fi

cd "$PROJ"
python3 -m venv "$WORK_ROOT/venv-refill-verify" 2>/dev/null || true
# shellcheck disable=SC1091
source "$WORK_ROOT/venv-refill-verify/bin/activate"
pip install -q -r requirements.txt
export PYTHONPATH="$PROJ/src:${PYTHONPATH:-}"

rocprof() { python3 "$PROJ/src/rocprof-compute" "$@"; }

echo "=== Code ==="
git -C "$REPO" log -1 --oneline 2>/dev/null || true
python3 -c "from rocprof_compute_soc.counter_grouping_refill import apply_metric_coalesce_refill_pass; print('refill OK')"

echo "=== Partition / CU ==="
rocm-smi --showcomputepartition 2>&1 | grep -E "GPU\[$ROCR_GPU\]|Partition" | head -5 || true
rocminfo 2>/dev/null | grep -E "Marketing Name:.*MI|Compute Unit:" | tail -4 || true

cd "$PROJ/sample"
hipcc --rocm-path="$ROCM_ROOT" -O3 vcopy.cpp -o vcopy
hipcc --rocm-path="$ROCM_ROOT" -O3 occupancy.hip -o occupancy
cd "$PROJ"

pmc_union() {
  local wl="$1"
  PYTHONPATH="$PROJ/src:${PYTHONPATH:-}" python3 - <<'PY' "$wl"
import sys
from pathlib import Path

from vendored import yaml

wl = Path(sys.argv[1])
pmcs: set[str] = set()
for path in sorted((wl / "perfmon").glob("pmc_perf_*.yaml")):
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    for job in doc.get("jobs", []):
        pmcs.update(job.get("pmc", []))
print(len(pmcs))
print("\n".join(sorted(pmcs)))
PY
}

profile_one() {
  local label="$1"
  local refill="$2"
  local binary="$3"
  shift 3
  local extra_args=("$@")
  local name="${label}_refill_${refill}"
  local out_dir="$ARTIFACT_ROOT/$name"
  rm -rf "$out_dir"
  export ROCPROF_COMPUTE_COALESCE_REFILL="$refill"
  echo ""
  echo "========== PROFILE $name (ROCPROF_COMPUTE_COALESCE_REFILL=$refill) =========="
  if [[ ! -f "$out_dir/sysinfo.csv" ]]; then
    rocprof profile --no-roof --no-native-tool -q \
      --output-directory "$out_dir" -- "$binary" "${extra_args[@]}"
  else
    echo "(skip profile; workload already exists at $out_dir)"
  fi
  local wl="$out_dir"
  if [[ -f "$out_dir/0/sysinfo.csv" ]]; then
    wl="$out_dir/0"
  fi
  local passes
  passes=$(ls -1 "$wl/perfmon/"pmc_perf_*.yaml 2>/dev/null | wc -l)
  echo "passes=$passes wl=$wl"
  {
    echo "name=$name refill=$refill passes=$passes"
    echo "pmc_count=$(pmc_union "$wl" | head -1)"
  } >>"$ARTIFACT_ROOT/summary.txt"
  pmc_union "$wl" | tail -n +2 | sort >"$ARTIFACT_ROOT/${name}_pmcs.txt"
  rocprof analyze --path "$wl" -b $METRIC_BLOCKS \
    >"$ARTIFACT_ROOT/analyze_${name}.log" 2>&1
  grep -E "HBM Read Traffic|HBM Write|Workgroup Manager" \
    "$ARTIFACT_ROOT/analyze_${name}.log" || true
}

: >"$ARTIFACT_ROOT/summary.txt"

VCOPY_N="${VCOPY_N:-1048576}"
VCOPY_B="${VCOPY_B:-256}"
profile_one vcopy 0 "$PROJ/sample/vcopy" -n "$VCOPY_N" -b "$VCOPY_B"
profile_one vcopy 1 "$PROJ/sample/vcopy" -n "$VCOPY_N" -b "$VCOPY_B"
profile_one occupancy 0 "$PROJ/sample/occupancy"
profile_one occupancy 1 "$PROJ/sample/occupancy"

echo ""
echo "=== PMC set diff (should be empty) ==="
for base in vcopy occupancy; do
  if diff -q "$ARTIFACT_ROOT/${base}_refill_0_pmcs.txt" \
    "$ARTIFACT_ROOT/${base}_refill_1_pmcs.txt" >/dev/null; then
    echo "$base: PMC union identical"
  else
    echo "$base: PMC union DIFFERS (unexpected)"
    diff "$ARTIFACT_ROOT/${base}_refill_0_pmcs.txt" \
      "$ARTIFACT_ROOT/${base}_refill_1_pmcs.txt" | head -20 || true
  fi
done

echo ""
echo "=== Pass counts ==="
grep -E '^name=' "$ARTIFACT_ROOT/summary.txt" || true

echo ""
echo "=== Analyze golden rows (side by side) ==="
for base in vcopy occupancy; do
  echo "--- $base refill=0 ---"
  grep -E "HBM Read Traffic|HBM Write|Workgroup Manager" \
    "$ARTIFACT_ROOT/analyze_${base}_refill_0.log" || true
  echo "--- $base refill=1 ---"
  grep -E "HBM Read Traffic|HBM Write|Workgroup Manager" \
    "$ARTIFACT_ROOT/analyze_${base}_refill_1.log" || true
done

echo ""
echo "=== Inspector compare-refill (offline, gfx942) ==="
./tools/counter_grouping_inspector.py --arch gfx942 --compare-refill \
  -o "$ARTIFACT_ROOT/gfx942_refill_compare.txt" || true
head -20 "$ARTIFACT_ROOT/gfx942_refill_compare.txt" || true

echo ""
echo "=== Done artifacts: $ARTIFACT_ROOT ==="
