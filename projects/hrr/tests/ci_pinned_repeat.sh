#!/usr/bin/env bash
# CROSS-CHECK ONLY (hrr-ci/2894gfx90a, do not merge): repeats the PinnedHost
# cases whose capture workloads died with "HW Exception ... GPU Hang" on
# smci250 gfx90a, with controls, then runs every PinnedHost case once.
# $1: JUnit file for the final full PinnedHost run.
set -u
T=./build/hrr/tests/integration/hrr-integration-tests
XML="${1:-pinned.xml}"
REP="${REP:-6}"
mkdir -p rep
SUMMARY=rep/summary.txt
: > "$SUMMARY"
bad=0

echo "host $(hostname) runner ${RUNNER_NAME:-?} kernel $(uname -r) amdgpu $(cat /sys/module/amdgpu/version 2>/dev/null || echo in-box)"
for p in cwsr_enable lockup_timeout sched_policy hws_max_conc_proc queue_preemption_timeout_ms noretry mes; do
  echo "amdgpu.$p=$(cat /sys/module/amdgpu/parameters/$p 2>/dev/null || echo '?')"
done
(rocminfo 2>/dev/null || "${ROCM_SDK}/bin/rocminfo" 2>/dev/null) |
  grep -E "^ *(Name: *gfx|Marketing Name|Compute Unit)" | head -6 || true
env | grep -E '^(HSA_|HIP_|GPU_|ROCR_|AMD_|HRR_)' | sort || true

# run1 LABEL CASE [ENV=VAL ...]
run1() {
  local label="$1" name="$2"; shift 2
  local log="rep/$(echo "$label" | tr -c 'A-Za-z0-9_.-' '_').$RANDOM.log"
  local t0=$(date +%s.%N)
  timeout 300 env "$@" "$T" "$name" > "$log" 2>&1
  local rc=$?
  local dt=$(awk -v a="$t0" -v b="$(date +%s.%N)" 'BEGIN{printf "%.1f", b-a}')
  local hang=""
  grep -q "GPU Hang" "$log" && hang=" GPU-HANG"
  local res=PASS
  [ $rc -ne 0 ] && res=FAIL
  echo "$(date -u +%H:%M:%S) $res rc=$rc ${dt}s$hang  $label" | tee -a "$SUMMARY"
  if [ $rc -ne 0 ]; then
    bad=$((bad + 1))
    grep -E "GPU Hang|exit:|REQUIRE|CHECK|with expansion|== |FAILED|hung|stream B" "$log" | head -25
  fi
}

FAILING=(Unit_HRR_PinnedHost_RestoreWaitsOnStream Unit_HRR_PinnedHost_SptLegacyNoBarrier
         Unit_HRR_PinnedHost_FailedNullLaunchNoBarrier)
COMPANIONS=(Unit_HRR_PinnedHost_SptCoopLegacyNoBarrier Unit_HRR_PinnedHost_WaitsForOtherStream
            Unit_HRR_PinnedHost_NullStreamDrainedFirst Unit_HRR_PinnedHost_NoWaitOnBusyStream)

echo "::group::A. the three cases x${REP}, interleaved"
for i in $(seq 1 "$REP"); do
  for c in "${FAILING[@]}"; do run1 "A $c #$i" "$c"; done
done
echo "::endgroup::"

echo "::group::B. companions x2"
for i in 1 2; do
  for c in "${COMPANIONS[@]}"; do run1 "B $c #$i" "$c"; done
done
echo "::endgroup::"

echo "::group::C. the workloads alone, no capture, x3"
for i in 1 2 3; do
  run1 "C BatchWait_Direct nocap #$i" Unit_HRR_PinnedHost_BatchWait_Direct
  run1 "C NoNullBarrier_Direct v0 nocap #$i" Unit_HRR_PinnedHost_NoNullBarrier_Direct HRR_PINNED_VARIANT=0
  run1 "C NoNullBarrier_Direct v2 nocap #$i" Unit_HRR_PinnedHost_NoNullBarrier_Direct HRR_PINNED_VARIANT=2
done
echo "::endgroup::"

echo "::group::D. the three cases with GPU_STREAMOPS_CP_WAIT=1 x2"
for i in 1 2; do
  for c in "${FAILING[@]}"; do run1 "D $c cpwait #$i" "$c" GPU_STREAMOPS_CP_WAIT=1; done
done
echo "::endgroup::"

echo "::group::E. host control: NullStreamMemsetOrdering x3"
for i in 1 2 3; do run1 "E NullStreamMemsetOrdering #$i" Unit_HRR_NullStreamMemsetOrdering; done
echo "::endgroup::"

echo "::group::F. every PinnedHost case once"
t0=$(date +%s)
python3 projects/hrr/tests/run_catch2.py --xml "$XML" --timeout 1200 -- "$T" "Unit_HRR_PinnedHost_*,~[.]"
frc=$?
echo "F. full PinnedHost run rc=$frc in $(( $(date +%s) - t0 ))s" | tee -a "$SUMMARY"
[ $frc -ne 0 ] && bad=$((bad + 1))
echo "::endgroup::"

echo "dmesg tail:"
dmesg 2>/dev/null | grep -iE "amdgpu|kfd|hang|reset|preempt" | tail -30 || echo "(dmesg not readable)"

echo "===== SUMMARY $(hostname) ${RUNNER_NAME:-?}"
cat "$SUMMARY"
for g in A B C D E; do
  echo "$g: $(grep -c " PASS .* $g " "$SUMMARY") pass, $(grep -c " FAIL .* $g " "$SUMMARY") fail, $(grep " $g " "$SUMMARY" | grep -c GPU-HANG) GPU-HANG"
done
[ "$bad" -eq 0 ]
