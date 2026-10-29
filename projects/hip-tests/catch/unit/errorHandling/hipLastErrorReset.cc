/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

#include <cstdint>

/**
 * Regression for the cross-test last-error reset listener (hip_test_listener.cc).
 * The two cases must run in declaration order, so run them with --order decl:
 *   ErrorHandlingTest "Unit_hipLastErrorReset_LeaveError,\
 *                      Unit_hipLastErrorReset_SeesCleanState" --order decl
 *
 * Normal run: exit code 0, with a "[HIP last-error]" line on stderr naming the
 * first case. With HIP_TEST_STRICT_LAST_ERROR=1: non-zero exit, first case
 * reported as the strict-mode culprit.
 */

// Leaves a sticky last error on purpose and does not consume it; the listener
// must clear it before the next case runs.
HIP_TEST_CASE(Unit_hipLastErrorReset_LeaveError) {
  void* ptr = nullptr;
  hipError_t err = hipMalloc(&ptr, SIZE_MAX);
  REQUIRE(err != hipSuccess);
}

// Passes only if the listener reset the previous case's leaked error.
HIP_TEST_CASE(Unit_hipLastErrorReset_SeesCleanState) {
  REQUIRE(hipGetLastError() == hipSuccess);
}
