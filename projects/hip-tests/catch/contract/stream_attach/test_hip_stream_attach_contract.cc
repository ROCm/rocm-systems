/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

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
}  // namespace

HIP_TEST_CASE(Contract_StreamAttach_ManagedOnCreatedStream_Succeeds) {
  SkipIfManagedMemoryUnsupported();

  int* data = nullptr;
  hipStream_t stream = nullptr;

  HIP_CHECK(hipMallocManaged(&data, kAttachBytes, hipMemAttachHost));
  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(hipStreamAttachMemAsync(stream, data, 0, hipMemAttachSingle));
  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(data));
}

HIP_TEST_CASE(Contract_StreamAttach_NullStream_AttachGlobal_Succeeds) {
  SkipIfManagedMemoryUnsupported();

  int* data = nullptr;

  HIP_CHECK(hipMallocManaged(&data, kAttachBytes, hipMemAttachHost));

  HIP_CHECK(hipStreamAttachMemAsync(nullptr, data, 0, hipMemAttachGlobal));
  HIP_CHECK(hipStreamSynchronize(nullptr));

  HIP_CHECK(hipFree(data));
}

HIP_TEST_CASE(Contract_StreamAttach_NonZeroLengthAttachSingle_Succeeds) {
  SkipIfManagedMemoryUnsupported();

  int* data = nullptr;
  hipStream_t stream = nullptr;

  HIP_CHECK(hipMallocManaged(&data, kAttachBytes, hipMemAttachHost));
  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(hipStreamAttachMemAsync(stream, data, kAttachBytes, hipMemAttachSingle));
  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(data));
}

HIP_TEST_CASE(Contract_StreamAttach_NullDevPtr_IsRejected) {
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  REQUIRE(hipStreamAttachMemAsync(stream, nullptr, kAttachBytes, hipMemAttachSingle) != hipSuccess);

  HIP_CHECK(hipStreamDestroy(stream));
}

HIP_TEST_CASE(Contract_StreamAttach_NullStreamAttachSingle_IsRejected) {
  SkipIfManagedMemoryUnsupported();

  int* data = nullptr;

  HIP_CHECK(hipMallocManaged(&data, kAttachBytes, hipMemAttachHost));

  REQUIRE(hipStreamAttachMemAsync(nullptr, data, 0, hipMemAttachSingle) != hipSuccess);

  HIP_CHECK(hipFree(data));
}
