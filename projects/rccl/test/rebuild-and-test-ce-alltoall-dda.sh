#!/usr/bin/env bash
set -euo pipefail

# Fast incremental rebuild + run of the CeAlltoAllDdaDecision unit tests.
#
# Assumes the debug tree has already been configured once via:
#   ./install.sh --debug --tests_build --local_gpu_only
#
# Editing only CeAlltoAllDdaDecisionTests.cpp recompiles just that file.
# Editing rccl_wrap.cc / rccl_common.h (the helper) also relinks librccl.so,
# still far cheaper than a full install.sh run.

rccl_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${rccl_root}/build/debug"
test_binary="${build_dir}/test/rccl-UnitTestsFixturesDebug"

make \
    --directory "${build_dir}" \
    --jobs "$(nproc)" \
    rccl-UnitTestsFixturesDebug

LD_LIBRARY_PATH="${build_dir}:${LD_LIBRARY_PATH:-}" \
    "${test_binary}" \
    --gtest_filter='CeAlltoAllDdaDecisionTest.*' \
    "$@"
