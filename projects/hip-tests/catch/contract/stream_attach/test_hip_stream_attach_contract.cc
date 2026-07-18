/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <contract_cleanup.hh>

namespace {
constexpr size_t kAttachBytes = sizeof(int);

bool ManagedMemorySupported() {
  void* ptr = nullptr;
  const hipError_t status = hipMallocManaged(&ptr, kAttachBytes, hipMemAttachGlobal);
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

// hipStreamAttachMemAsync takes a `void*` device pointer on AMD but a
// `hipDeviceptr_t*` on the NVIDIA backend (the shim forwards to
// cuStreamAttachMemAsync, whose pointer parameter is CUdeviceptr*). Convert the
// managed allocation to the argument type each backend's signature expects.
#if HT_AMD
void* AttachPtr(void* p) { return p; }
#else
hipDeviceptr_t* AttachPtr(void* p) { return reinterpret_cast<hipDeviceptr_t*>(p); }
#endif
}  // namespace

HIP_TEST_CASE(Contract_StreamAttach_ManagedOnCreatedStream_Succeeds) {
  SkipIfManagedMemoryUnsupported();
  hip::contract::ContractCleanup cleanup;

  int* data = nullptr;
  hipStream_t stream = nullptr;

  HIP_CHECK(hipMallocManaged(&data, kAttachBytes, hipMemAttachHost));
  cleanup.Add([&] { (void)hipFree(data); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([&] { (void)hipStreamDestroy(stream); });

  HIP_CHECK(hipStreamAttachMemAsync(stream, AttachPtr(data), 0, hipMemAttachSingle));
  HIP_CHECK(hipStreamSynchronize(stream));
}

HIP_TEST_CASE(Contract_StreamAttach_NullStream_AttachGlobal_Succeeds) {
  SkipIfManagedMemoryUnsupported();
  hip::contract::ContractCleanup cleanup;

  int* data = nullptr;

  HIP_CHECK(hipMallocManaged(&data, kAttachBytes, hipMemAttachHost));
  cleanup.Add([&] { (void)hipFree(data); });

  HIP_CHECK(hipStreamAttachMemAsync(nullptr, AttachPtr(data), 0, hipMemAttachGlobal));
  HIP_CHECK(hipStreamSynchronize(nullptr));
}

HIP_TEST_CASE(Contract_StreamAttach_NonZeroLengthAttachSingle_Succeeds) {
  SkipIfManagedMemoryUnsupported();
  hip::contract::ContractCleanup cleanup;

  int* data = nullptr;
  hipStream_t stream = nullptr;

  HIP_CHECK(hipMallocManaged(&data, kAttachBytes, hipMemAttachHost));
  cleanup.Add([&] { (void)hipFree(data); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([&] { (void)hipStreamDestroy(stream); });

  HIP_CHECK(hipStreamAttachMemAsync(stream, AttachPtr(data), kAttachBytes, hipMemAttachSingle));
  HIP_CHECK(hipStreamSynchronize(stream));
}

HIP_TEST_CASE(Contract_StreamAttach_NullDevPtr_IsRejected) {
  hip::contract::ContractCleanup cleanup;
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([&] { (void)hipStreamDestroy(stream); });

  REQUIRE(hipStreamAttachMemAsync(stream, nullptr, kAttachBytes, hipMemAttachSingle) != hipSuccess);
}

HIP_TEST_CASE(Contract_StreamAttach_NullStreamAttachSingle_IsRejected) {
  SkipIfManagedMemoryUnsupported();
  hip::contract::ContractCleanup cleanup;

  int* data = nullptr;

  HIP_CHECK(hipMallocManaged(&data, kAttachBytes, hipMemAttachHost));
  cleanup.Add([&] { (void)hipFree(data); });

  REQUIRE(hipStreamAttachMemAsync(nullptr, AttachPtr(data), 0, hipMemAttachSingle) != hipSuccess);
}
