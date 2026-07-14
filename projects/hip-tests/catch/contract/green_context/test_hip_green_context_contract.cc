/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <contract_cleanup.hh>

namespace {
int CurrentDevice() {
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  return device;
}

// Queries the current device's SM resource. Returns false (for a capability
// skip) if the runtime path does not support device resource queries.
bool TryGetDeviceSmResource(hipDevResource* resource) {
  const hipError_t status =
      hipDeviceGetDevResource(CurrentDevice(), resource, hipDevResourceTypeSm);
  if (status == hipErrorNotSupported) {
    return false;
  }
  HIP_CHECK(status);
  return true;
}

void SkipIfDevResourceUnsupported() {
  hipDevResource resource{};
  if (!TryGetDeviceSmResource(&resource)) {
    HIP_SKIP_TEST("Device resource queries are not supported by this runtime path.");
  }
}

// Builds a single-partition green execution context from the smallest SM split
// of the current device. Returns false if any step reports the path is
// unsupported; the caller then skips. On success the caller owns the returned
// context and must destroy it.
bool TryCreateGreenContext(hipExecutionCtx_t* ctx) {
  hipDevResource device_resource{};
  if (!TryGetDeviceSmResource(&device_resource)) {
    return false;
  }

  unsigned int group_count = 1;
  hipDevResource groups{};
  hipDevResource remainder{};
  hipError_t status = hipDevSmResourceSplitByCount(&groups, &group_count, &device_resource,
                                                   &remainder, 0, 1);
  if (status == hipErrorNotSupported) {
    return false;
  }
  HIP_CHECK(status);
  if (group_count == 0) {
    return false;
  }

  hipDevResourceDesc_t desc{};
  status = hipDevResourceGenerateDesc(&desc, &groups, 1);
  if (status == hipErrorNotSupported) {
    return false;
  }
  HIP_CHECK(status);

  status = hipGreenCtxCreate(ctx, desc, CurrentDevice(), 0);
  if (status == hipErrorNotSupported) {
    return false;
  }
  HIP_CHECK(status);
  return true;
}
}  // namespace

HIP_TEST_CASE(Contract_GreenContext_GetDevResource_ReportsSmCount) {
  SkipIfDevResourceUnsupported();
  hip::contract::ContractCleanup cleanup;

  // The device SM resource must report a positive SM count. A freshly created
  // stream draws from the whole device, so its SM resource reports the same
  // count.
  hipDevResource device_resource{};
  HIP_CHECK(hipDeviceGetDevResource(CurrentDevice(), &device_resource, hipDevResourceTypeSm));
  REQUIRE(device_resource.type == hipDevResourceTypeSm);
  REQUIRE(device_resource.sm.smCount > 0);

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([&] { (void)hipStreamDestroy(stream); });
  hipDevResource stream_resource{};
  HIP_CHECK(hipStreamGetDevResource(stream, &stream_resource, hipDevResourceTypeSm));
  REQUIRE(stream_resource.sm.smCount == device_resource.sm.smCount);
}

HIP_TEST_CASE(Contract_GreenContext_SplitByCount_ProducesBoundedSubset) {
  SkipIfDevResourceUnsupported();

  hipDevResource device_resource{};
  HIP_CHECK(hipDeviceGetDevResource(CurrentDevice(), &device_resource, hipDevResourceTypeSm));

  // Splitting the device SM resource into groups of at least one SM must yield
  // at least one group whose SM count is within (0, device SM count]. The
  // partition cannot invent SMs the device does not have.
  unsigned int group_count = 1;
  hipDevResource groups{};
  hipDevResource remainder{};
  const hipError_t status = hipDevSmResourceSplitByCount(&groups, &group_count, &device_resource,
                                                         &remainder, 0, 1);
  if (status == hipErrorNotSupported) {
    HIP_SKIP_TEST("SM resource splitting is not supported by this runtime path.");
  }
  HIP_CHECK(status);

  REQUIRE(group_count > 0);
  REQUIRE(groups.sm.smCount > 0);
  REQUIRE(groups.sm.smCount <= device_resource.sm.smCount);
}

HIP_TEST_CASE(Contract_GreenContext_Create_QueriesDeviceAndResource) {
  hipExecutionCtx_t ctx = nullptr;
  if (!TryCreateGreenContext(&ctx)) {
    HIP_SKIP_TEST("Green execution contexts are not supported by this runtime path.");
  }
  hip::contract::ContractCleanup cleanup;
  cleanup.Add([&] { (void)hipExecutionCtxDestroy(ctx); });

  // A green context is bound to the device it was created on and exposes a
  // stable identifier. Its SM resource is a subset of the device's SMs.
  int ctx_device = -1;
  HIP_CHECK(hipExecutionCtxGetDevice(&ctx_device, ctx));
  REQUIRE(ctx_device == CurrentDevice());

  unsigned long long ctx_id = 0;
  HIP_CHECK(hipExecutionCtxGetId(ctx, &ctx_id));

  hipDevResource device_resource{};
  HIP_CHECK(hipDeviceGetDevResource(CurrentDevice(), &device_resource, hipDevResourceTypeSm));

  hipDevResource ctx_resource{};
  HIP_CHECK(hipExecutionCtxGetDevResource(ctx, &ctx_resource, hipDevResourceTypeSm));
  REQUIRE(ctx_resource.sm.smCount > 0);
  REQUIRE(ctx_resource.sm.smCount <= device_resource.sm.smCount);

  // The device's default execution context is also queryable.
  hipExecutionCtx_t default_ctx = nullptr;
  HIP_CHECK(hipDeviceGetExecutionCtx(&default_ctx, CurrentDevice()));
}

HIP_TEST_CASE(Contract_GreenContext_Stream_LaunchesObservableWork) {
  hipExecutionCtx_t ctx = nullptr;
  if (!TryCreateGreenContext(&ctx)) {
    HIP_SKIP_TEST("Green execution contexts are not supported by this runtime path.");
  }
  hip::contract::ContractCleanup cleanup;
  cleanup.Add([&] { (void)hipExecutionCtxDestroy(ctx); });

  // Work submitted to a stream created on the green context must run and
  // complete: a memset on that stream is visible after the context is
  // synchronized.
  hipStream_t stream = nullptr;
  HIP_CHECK(hipExecutionCtxStreamCreate(&stream, ctx, 0, 0));
  cleanup.Add([&] { (void)hipStreamDestroy(stream); });

  void* device_ptr = nullptr;
  HIP_CHECK(hipMalloc(&device_ptr, 64));
  cleanup.Add([&] { (void)hipFree(device_ptr); });
  HIP_CHECK(hipMemsetAsync(device_ptr, 0x5A, 64, stream));
  HIP_CHECK(hipExecutionCtxSynchronize(ctx));

  uint8_t host = 0;
  HIP_CHECK(hipMemcpy(&host, device_ptr, sizeof(host), hipMemcpyDeviceToHost));
  REQUIRE(host == 0x5A);
}

HIP_TEST_CASE(Contract_GreenContext_Event_RecordAndWaitRoundTrips) {
  hipExecutionCtx_t ctx = nullptr;
  if (!TryCreateGreenContext(&ctx)) {
    HIP_SKIP_TEST("Green execution contexts are not supported by this runtime path.");
  }
  hip::contract::ContractCleanup cleanup;
  cleanup.Add([&] { (void)hipExecutionCtxDestroy(ctx); });

  // Recording an event on the green context and then waiting on it must be
  // accepted, and the context must synchronize cleanly afterward.
  hipEvent_t event = nullptr;
  HIP_CHECK(hipEventCreate(&event));
  cleanup.Add([&] { (void)hipEventDestroy(event); });
  HIP_CHECK(hipExecutionCtxRecordEvent(ctx, event));
  HIP_CHECK(hipExecutionCtxWaitEvent(ctx, event));
  HIP_CHECK(hipExecutionCtxSynchronize(ctx));
}
