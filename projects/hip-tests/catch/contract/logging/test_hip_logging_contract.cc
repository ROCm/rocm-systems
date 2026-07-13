/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

// The extended logging controls (hipExtEnableLogging, hipExtDisableLogging,
// hipExtSetLoggingParams) are AMD extension APIs, so the whole domain is gated
// like the other AMD extension contracts.
#if HT_AMD

namespace {
// A logging control call must either be honored or reported unsupported. No
// other status satisfies the contract.
void RequireAcceptedOrUnsupported(hipError_t status) {
  if (status == hipSuccess || status == hipErrorNotSupported) {
    return;
  }
  HIP_CHECK(status);
}
}  // namespace

HIP_TEST_CASE(Contract_Logging_EnableThenDisable_IsAcceptedOrUnsupported) {
  // Enabling then disabling logging must report a consistent capability: if
  // enable is honored, the matching disable must also be honored, and if enable
  // is unsupported, disable must likewise be unsupported. A runtime cannot claim
  // to enable logging and then refuse to disable it. Logging is left disabled so
  // this does not perturb output for sibling tests.
  const hipError_t enable_status = hipExtEnableLogging();
  RequireAcceptedOrUnsupported(enable_status);

  const hipError_t disable_status = hipExtDisableLogging();
  RequireAcceptedOrUnsupported(disable_status);

  REQUIRE((enable_status == hipSuccess) == (disable_status == hipSuccess));
}

HIP_TEST_CASE(Contract_Logging_SetLoggingParams_IsAcceptedOrUnsupported) {
  // Configuring the logging level, buffer size, and mask must be accepted or
  // reported unsupported. The parameters chosen are benign (a low level with a
  // small buffer); logging is disabled afterward so no output is emitted for
  // later tests.
  const hipError_t status = hipExtSetLoggingParams(1, 4096, 0);
  RequireAcceptedOrUnsupported(status);

  RequireAcceptedOrUnsupported(hipExtDisableLogging());
}

#endif  // HT_AMD
