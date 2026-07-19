/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

namespace {
// Requires at least one visible device, skipping the test when none is present
// so that driver-style context mutation is only exercised against a real ordinal.
void RequireDevice() {
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count <= 0) {
    HIP_SKIP_TEST(HipTest::SkipReason::kNoGpuDevice);
  }
  // Establish a device context before the driver-style hipCtx* entry points run.
  // On the NVIDIA backend these map to the driver API, which requires a bound
  // primary context; without one, e.g. hipCtxSynchronize returns "invalid device
  // context" rather than succeeding. hipFree(0) is the canonical no-op that forces
  // primary-context initialization, and is a harmless success on AMD where the
  // runtime already auto-initializes.
  HIP_CHECK(hipFree(0));
}

// Resolves the driver-style handle for ordinal zero, which every context
// mutation contract builds on.
hipDevice_t DeviceForOrdinalZero() {
  hipDevice_t device = 0;
  HIP_CHECK(hipDeviceGet(&device, 0));
  return device;
}

// Saves the thread-current driver-style context on construction and restores it
// on destruction so tests that mutate the current context do not leak that state
// into later tests when several run in one process. The destructor cannot use
// Catch assertions, so it ignores the restore status; tests still perform an
// explicit, asserted restore before scope exit so failures on the restore path
// stay visible.
class ScopedCurrentContext {
 public:
  ScopedCurrentContext() { HIP_CHECK(hipCtxGetCurrent(&previous_)); }
  ~ScopedCurrentContext() { static_cast<void>(hipCtxSetCurrent(previous_)); }

  ScopedCurrentContext(const ScopedCurrentContext&) = delete;
  ScopedCurrentContext& operator=(const ScopedCurrentContext&) = delete;

  hipCtx_t previous() const { return previous_; }

 private:
  hipCtx_t previous_ = nullptr;
};
}  // namespace

// @asserts: hipCtxCreate - creating then destroying a context succeeds and the previously current context is restored afterward
HIP_TEST_CASE(Contract_ContextMutation_CreateDestroy_Succeeds) {
  RequireDevice();

  const hipDevice_t device = DeviceForOrdinalZero();
  const ScopedCurrentContext scoped_context;

  hipCtx_t context = nullptr;
  HIP_CHECK(hipCtxCreate(&context, 0, device));
  REQUIRE(context != nullptr);

  HIP_CHECK(hipCtxDestroy(context));

  // Restore the previously current context explicitly so the round trip is an
  // asserted step rather than relying solely on the RAII guard's silent restore.
  HIP_CHECK(hipCtxSetCurrent(scoped_context.previous()));

  hipCtx_t current = nullptr;
  HIP_CHECK(hipCtxGetCurrent(&current));
  REQUIRE(current == scoped_context.previous());
}

// @asserts: hipCtxSetCurrent - a set context round-trips through hipCtxGetCurrent and its device agrees with the runtime current device
HIP_TEST_CASE(Contract_ContextMutation_SetCurrent_RoundTripsThroughGetCurrent) {
  RequireDevice();

  const hipDevice_t device = DeviceForOrdinalZero();
  const ScopedCurrentContext scoped_context;

  hipCtx_t context = nullptr;
  HIP_CHECK(hipCtxCreate(&context, 0, device));
  REQUIRE(context != nullptr);

  HIP_CHECK(hipCtxSetCurrent(context));

  hipCtx_t observed = nullptr;
  HIP_CHECK(hipCtxGetCurrent(&observed));
  REQUIRE(observed == context);

  // With this context current, the context device and the runtime current
  // device must agree.
  hipDevice_t context_device = 0;
  HIP_CHECK(hipCtxGetDevice(&context_device));

  int runtime_device = -1;
  HIP_CHECK(hipGetDevice(&runtime_device));
  REQUIRE(context_device == runtime_device);

  // Restore first so the destroyed context is no longer current, then destroy.
  HIP_CHECK(hipCtxSetCurrent(scoped_context.previous()));
  HIP_CHECK(hipCtxDestroy(context));
}

// @asserts: hipCtxPushCurrent - pushing a context makes it current and popping it returns that context and restores the previous current context
HIP_TEST_CASE(Contract_ContextMutation_PushPop_RestoresPreviousCurrent) {
  RequireDevice();

  const hipDevice_t device = DeviceForOrdinalZero();
  const ScopedCurrentContext scoped_context;

  hipCtx_t context = nullptr;
  HIP_CHECK(hipCtxCreate(&context, 0, device));
  REQUIRE(context != nullptr);

  // hipCtxCreate makes the new context current; normalize back to the saved
  // context so the push/pop round trip is measured against a known baseline.
  HIP_CHECK(hipCtxSetCurrent(scoped_context.previous()));

  HIP_CHECK(hipCtxPushCurrent(context));

  hipCtx_t after_push = nullptr;
  HIP_CHECK(hipCtxGetCurrent(&after_push));
  REQUIRE(after_push == context);

  hipCtx_t popped = nullptr;
  HIP_CHECK(hipCtxPopCurrent(&popped));
  REQUIRE(popped == context);

  hipCtx_t after_pop = nullptr;
  HIP_CHECK(hipCtxGetCurrent(&after_pop));
  REQUIRE(after_pop == scoped_context.previous());

  HIP_CHECK(hipCtxDestroy(context));
}

// @asserts: hipCtxSynchronize - a driver-style context barrier either succeeds or reports accepted-or-unsupported
HIP_TEST_CASE(Contract_ContextMutation_Synchronize_ReportsSupportedOrNotSupported) {
  RequireDevice();

  // hipCtxSynchronize is a driver-style barrier over the current context. It
  // must either succeed or report that the runtime does not support it; no
  // vendor-specific outcome beyond that is assumed.
  const hipError_t status = hipCtxSynchronize();
  REQUIRE((status == hipSuccess || status == hipErrorNotSupported));
}

// @asserts: hipCtxGetApiVersion - the query either succeeds with a positive version or reports accepted-or-unsupported
HIP_TEST_CASE(Contract_ContextMutation_GetApiVersion_ReportsVersionOrNotSupported) {
  RequireDevice();

  const hipDevice_t device = DeviceForOrdinalZero();
  const ScopedCurrentContext scoped_context;

  hipCtx_t context = nullptr;
  HIP_CHECK(hipCtxCreate(&context, 0, device));
  REQUIRE(context != nullptr);

  unsigned int version = 0;
  const hipError_t status = hipCtxGetApiVersion(context, &version);

  // The API version query must either succeed with a positive version or report
  // that the runtime does not support it. No exact version value is assumed.
  REQUIRE((status == hipSuccess || status == hipErrorNotSupported));
  if (status == hipSuccess) {
    REQUIRE(version > 0);
  }

  HIP_CHECK(hipCtxSetCurrent(scoped_context.previous()));
  HIP_CHECK(hipCtxDestroy(context));
}
