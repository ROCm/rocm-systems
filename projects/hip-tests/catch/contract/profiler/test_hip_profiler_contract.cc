/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

namespace {
// The profiler control entry points are deprecated. A conforming runtime either
// honors them (hipSuccess) or reports that profiler control is unavailable
// (hipErrorNotSupported). No other status satisfies the contract.
void RequireAcceptedOrUnsupported(hipError_t status) {
  if (status == hipSuccess || status == hipErrorNotSupported) {
    return;
  }
  HIP_CHECK(status);
}
}  // namespace

// @asserts: hipProfilerStart - is either honored (hipSuccess) or reported unavailable (hipErrorNotSupported), no other status
HIP_TEST_CASE(Contract_Profiler_Start_IsAcceptedOrUnsupported) {
  RequireAcceptedOrUnsupported(hipProfilerStart());
}

// @asserts: hipProfilerStop - is either honored (hipSuccess) or reported unavailable (hipErrorNotSupported), no other status
HIP_TEST_CASE(Contract_Profiler_Stop_IsAcceptedOrUnsupported) {
  RequireAcceptedOrUnsupported(hipProfilerStop());
}

// @asserts: hipProfilerStart - start/stop report a consistent capability: stop succeeds iff the matching start succeeded
HIP_TEST_CASE(Contract_Profiler_StartStop_PairIsAcceptedOrUnsupported) {
  // Starting then stopping profiling must report a consistent capability: if
  // start is honored the matching stop must also be honored, and if start is
  // unsupported stop must likewise be unsupported. A runtime cannot claim to
  // start profiling and then refuse to stop it.
  const hipError_t start_status = hipProfilerStart();
  RequireAcceptedOrUnsupported(start_status);

  const hipError_t stop_status = hipProfilerStop();
  RequireAcceptedOrUnsupported(stop_status);

  REQUIRE((start_status == hipSuccess) == (stop_status == hipSuccess));
}
