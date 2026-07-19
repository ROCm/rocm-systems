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

// @asserts: hipStreamAttachMemAsync - attaching managed memory with hipMemAttachSingle to a created stream succeeds
HIP_TEST_CASE(Contract_StreamAttach_ManagedOnCreatedStream_Succeeds) {
  SkipIfManagedMemoryUnsupported();
  hip::contract::ContractCleanup cleanup;

  int* data = nullptr;
  hipStream_t stream = nullptr;

  HIP_CHECK(hipMallocManaged(&data, kAttachBytes, hipMemAttachHost));
  cleanup.Add([data] { (void)hipFree(data); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });

  HIP_CHECK(hipStreamAttachMemAsync(stream, AttachPtr(data), 0, hipMemAttachSingle));
  HIP_CHECK(hipStreamSynchronize(stream));
}

// @asserts: hipStreamAttachMemAsync - attaching managed memory with hipMemAttachGlobal on the null stream succeeds
HIP_TEST_CASE(Contract_StreamAttach_NullStream_AttachGlobal_Succeeds) {
  SkipIfManagedMemoryUnsupported();
  hip::contract::ContractCleanup cleanup;

  int* data = nullptr;

  HIP_CHECK(hipMallocManaged(&data, kAttachBytes, hipMemAttachHost));
  cleanup.Add([data] { (void)hipFree(data); });

  HIP_CHECK(hipStreamAttachMemAsync(nullptr, AttachPtr(data), 0, hipMemAttachGlobal));
  HIP_CHECK(hipStreamSynchronize(nullptr));
}

// @asserts: hipStreamAttachMemAsync - a non-zero length with hipMemAttachSingle on a created stream succeeds
HIP_TEST_CASE(Contract_StreamAttach_NonZeroLengthAttachSingle_Succeeds) {
  SkipIfManagedMemoryUnsupported();
  hip::contract::ContractCleanup cleanup;

  int* data = nullptr;
  hipStream_t stream = nullptr;

  HIP_CHECK(hipMallocManaged(&data, kAttachBytes, hipMemAttachHost));
  cleanup.Add([data] { (void)hipFree(data); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });

  HIP_CHECK(hipStreamAttachMemAsync(stream, AttachPtr(data), kAttachBytes, hipMemAttachSingle));
  HIP_CHECK(hipStreamSynchronize(stream));
}

// @asserts: hipStreamAttachMemAsync - rejects a null device pointer with a non-success error
HIP_TEST_CASE(Contract_StreamAttach_NullDevPtr_IsRejected) {
  hip::contract::ContractCleanup cleanup;
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });

  REQUIRE(hipStreamAttachMemAsync(stream, nullptr, kAttachBytes, hipMemAttachSingle) != hipSuccess);
}

// @asserts: hipStreamAttachMemAsync - rejects hipMemAttachSingle on the null stream with a non-success error
HIP_TEST_CASE(Contract_StreamAttach_NullStreamAttachSingle_IsRejected) {
  SkipIfManagedMemoryUnsupported();
  hip::contract::ContractCleanup cleanup;

  int* data = nullptr;

  HIP_CHECK(hipMallocManaged(&data, kAttachBytes, hipMemAttachHost));
  cleanup.Add([data] { (void)hipFree(data); });

  REQUIRE(hipStreamAttachMemAsync(nullptr, AttachPtr(data), 0, hipMemAttachSingle) != hipSuccess);
}
