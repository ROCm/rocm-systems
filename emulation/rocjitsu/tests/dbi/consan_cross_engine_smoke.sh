#!/usr/bin/env bash
set -euo pipefail

build_dir="${1:-${ROCJITSU_BUILD_DIR:-}}"
if [[ -z "${build_dir}" ]]; then
  cat >&2 <<'EOF'
usage: consan_cross_engine_smoke.sh <rocjitsu-build-dir>

Alternatively set ROCJITSU_BUILD_DIR.
EOF
  exit 2
fi

parallel="${CTEST_PARALLEL_LEVEL:-8}"
regex='^(ConSanMoiHipTest\.DbiAutoReportBufferReplaysConflict|ConSanInlineShadowTest\.DbiReportsCrossWaveRace|ConSanMoiHipTest\.DbiSampledAutoBufferReportsConflict|ConSanMoiHipTest\.DbiBarrierRecordOrdersTwoWaveAccesses|ConSanInlineShadowTest\.DbiBarrierEpochOrdersCrossWaveAccesses|ConSanMoiHipTest\.DbiAtomicHandoffOrdersTwoWaveAccesses|ConSanInlineShadowTest\.DbiAtomicHandoffSameAddressIsClean|ConSanInlineShadowTest\.DbiAtomicHandoffWrongAddressReports)$'

ctest --test-dir "${build_dir}" -j "${parallel}" -R "${regex}" --output-on-failure
