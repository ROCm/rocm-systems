/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstddef>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <contract_cleanup.hh>

namespace {
constexpr int kHostValue = 0x1234;
constexpr int kDeviceValue = 0x5678;

__global__ void ReadThenWriteKernel(int* data, int expected, int replacement, int* observed) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *observed = (*data == expected) ? 1 : 0;
    *data = replacement;
  }
}

bool ManagedMemorySupported() {
  void* ptr = nullptr;
  const hipError_t status = hipMallocManaged(&ptr, sizeof(int), hipMemAttachGlobal);
  if (status == hipSuccess) {
    HIP_CHECK(hipFree(ptr));
    return true;
  }
  if (status == hipErrorNotSupported || status == hipErrorOutOfMemory) {
    return false;
  }
  HIP_CHECK(status);
  return false;
}

void SkipIfManagedMemoryUnsupported() {
  if (!ManagedMemorySupported()) {
    HIP_SKIP_TEST("hipMallocManaged is not supported by this device/runtime path.");
  }
}
}

HIP_TEST_CASE(Contract_ManagedMemory_MallocManaged_ReturnsUsablePointer) {
  SkipIfManagedMemoryUnsupported();
  hip::contract::ContractCleanup cleanup;
  int* data = nullptr;

  HIP_CHECK(hipMallocManaged(&data, sizeof(*data), hipMemAttachGlobal));
  cleanup.Add([&] { (void)hipFree(data); });

  REQUIRE(data != nullptr);
}

HIP_TEST_CASE(Contract_ManagedMemory_HostWriteDeviceRead_RoundTripsAfterSynchronize) {
  SkipIfManagedMemoryUnsupported();
  hip::contract::ContractCleanup cleanup;
  int* data = nullptr;
  int* observed = nullptr;

  HIP_CHECK(hipMallocManaged(&data, sizeof(*data), hipMemAttachGlobal));
  cleanup.Add([&] { (void)hipFree(data); });
  HIP_CHECK(hipMalloc(&observed, sizeof(*observed)));
  cleanup.Add([&] { (void)hipFree(observed); });
  *data = kHostValue;
  HIP_CHECK(hipMemset(observed, 0, sizeof(*observed)));

  hipLaunchKernelGGL(ReadThenWriteKernel, dim3(1), dim3(1), 0, 0, data, kHostValue,
                     kDeviceValue, observed);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  int host_observed = 0;
  HIP_CHECK(hipMemcpy(&host_observed, observed, sizeof(host_observed), hipMemcpyDeviceToHost));
  REQUIRE(host_observed == 1);
}

HIP_TEST_CASE(Contract_ManagedMemory_DeviceWriteHostRead_RoundTripsAfterSynchronize) {
  SkipIfManagedMemoryUnsupported();
  hip::contract::ContractCleanup cleanup;
  int* data = nullptr;
  int* observed = nullptr;

  HIP_CHECK(hipMallocManaged(&data, sizeof(*data), hipMemAttachGlobal));
  cleanup.Add([&] { (void)hipFree(data); });
  HIP_CHECK(hipMalloc(&observed, sizeof(*observed)));
  cleanup.Add([&] { (void)hipFree(observed); });
  *data = kHostValue;
  HIP_CHECK(hipMemset(observed, 0, sizeof(*observed)));

  hipLaunchKernelGGL(ReadThenWriteKernel, dim3(1), dim3(1), 0, 0, data, kHostValue,
                     kDeviceValue, observed);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  REQUIRE(*data == kDeviceValue);
}

HIP_TEST_CASE(Contract_ManagedMemory_FreeManagedPointer_Succeeds) {
  SkipIfManagedMemoryUnsupported();
  int* data = nullptr;

  HIP_CHECK(hipMallocManaged(&data, sizeof(*data), hipMemAttachGlobal));
  HIP_CHECK(hipFree(data));
}

HIP_TEST_CASE(Contract_ManagedMemory_PrefetchAsync_SucceedsWhenSupported) {
  SkipIfManagedMemoryUnsupported();
  hip::contract::ContractCleanup cleanup;
  int* data = nullptr;
  int device = 0;
  int concurrent_managed_access = 0;
  hipStream_t stream = nullptr;

  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipDeviceGetAttribute(&concurrent_managed_access,
                                  hipDeviceAttributeConcurrentManagedAccess, device));
  if (concurrent_managed_access == 0) {
    HIP_SKIP_TEST("hipMemPrefetchAsync requires concurrent managed access support.");
  }

  HIP_CHECK(hipMallocManaged(&data, sizeof(*data), hipMemAttachGlobal));
  cleanup.Add([&] { (void)hipFree(data); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([&] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipMemPrefetchAsync(data, sizeof(*data), device, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
}
