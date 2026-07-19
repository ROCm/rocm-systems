/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <contract_cleanup.hh>

// The external memory and semaphore interop APIs require a valid handle exported
// by another API (Vulkan, a DMA-buf producer, a D3D/NvSci object, etc.), which a
// device-only contract harness cannot construct. These contracts therefore
// exercise only the externally observable invariant that does not need a valid
// handle: an invalid or unsupported import/handle input must be rejected with a
// defined error rather than silently succeeding or corrupting the process. The
// exact error code is backend- and platform-specific, so only a non-success
// status is required.
namespace {
void RequireRejected(hipError_t status) {
  REQUIRE(status != hipSuccess);
  // A rejected call leaves a sticky thread-local error; clear it so it does not
  // leak into later tests.
  (void)hipGetLastError();
}
}  // namespace

// @asserts: hipImportExternalMemory - rejects an invalid (-1) file descriptor with a non-success status and no handle
HIP_TEST_CASE(Contract_ExternalResource_ImportMemory_InvalidFd_IsRejected) {
  // Importing external memory from an invalid file descriptor must not yield a
  // usable handle. On success the runtime would return a non-null handle; the
  // contract requires a non-success status and no handle.
  hipExternalMemoryHandleDesc desc{};
  desc.type = hipExternalMemoryHandleTypeOpaqueFd;
  desc.handle.fd = -1;
  desc.size = 4096;

  hipExternalMemory_t external_memory = nullptr;
  const hipError_t status = hipImportExternalMemory(&external_memory, &desc);
  RequireRejected(status);
  REQUIRE(external_memory == nullptr);
}

// @asserts: hipImportExternalSemaphore - rejects an invalid (-1) file descriptor with a non-success status and no handle
HIP_TEST_CASE(Contract_ExternalResource_ImportSemaphore_InvalidFd_IsRejected) {
  // Importing an external semaphore from an invalid file descriptor must be
  // rejected (or reported unsupported) rather than returning a usable handle.
  hipExternalSemaphoreHandleDesc desc{};
  desc.type = hipExternalSemaphoreHandleTypeOpaqueFd;
  desc.handle.fd = -1;

  hipExternalSemaphore_t external_semaphore = nullptr;
  const hipError_t status = hipImportExternalSemaphore(&external_semaphore, &desc);
  RequireRejected(status);
  REQUIRE(external_semaphore == nullptr);
}

// @asserts: hipExternalMemoryGetMappedBuffer - rejects a null external-memory handle with a non-success status
HIP_TEST_CASE(Contract_ExternalResource_GetMappedBuffer_NullHandle_IsRejected) {
  // Mapping a buffer from a null external-memory handle is invalid input and
  // must be rejected rather than returning a device pointer.
  hipExternalMemoryBufferDesc desc{};
  desc.offset = 0;
  desc.size = 4096;

  void* device_ptr = nullptr;
  const hipError_t status = hipExternalMemoryGetMappedBuffer(&device_ptr, nullptr, &desc);
  RequireRejected(status);
}

// @asserts: hipDestroyExternalMemory - rejects a null external-memory handle with a non-success status
HIP_TEST_CASE(Contract_ExternalResource_DestroyMemory_NullHandle_IsRejected) {
  // Destroying a null external-memory handle is invalid input and must be
  // rejected rather than silently succeeding.
  RequireRejected(hipDestroyExternalMemory(nullptr));
}

// @asserts: hipDestroyExternalSemaphore - rejects a null external-semaphore handle with a non-success status
HIP_TEST_CASE(Contract_ExternalResource_DestroySemaphore_NullHandle_IsRejected) {
  // Destroying a null external-semaphore handle is invalid input and must be
  // rejected rather than silently succeeding.
  RequireRejected(hipDestroyExternalSemaphore(nullptr));
}

// @asserts: hipSignalExternalSemaphoresAsync - rejects a batch containing a null semaphore handle with a non-success status (AMD only)
HIP_TEST_CASE(Contract_ExternalResource_SignalSemaphore_NullHandle_IsRejected) {
  // BACKEND-DIFF: The null-handle rejection contract is only exercised on AMD. On
  // NVIDIA hipSignalExternalSemaphoresAsync maps to
  // cudaSignalExternalSemaphoresAsync, which does not validate the semaphore
  // handle and dereferences it - a null handle faults (SIGSEGV) instead of
  // returning a defined error - so the rejection contract cannot be evaluated
  // safely there, across all tested CUDA versions. (The matching wait path is
  // version-dependent - it validates on CUDA >= 13.0 but faults on older CUDA -
  // so WaitSemaphore_NullHandle is guarded on CUDA_VERSION rather than skipped on
  // all NVIDIA; see below.) Parity would require null-handle validation on the
  // signal path.
#if HT_AMD
  hip::contract::ContractCleanup cleanup;
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });

  // Signalling a batch that contains a null external-semaphore handle must be
  // rejected rather than enqueuing an operation on an invalid object.
  hipExternalSemaphore_t semaphores[1] = {nullptr};
  hipExternalSemaphoreSignalParams params[1] = {};
  RequireRejected(hipSignalExternalSemaphoresAsync(semaphores, params, 1, stream));
#else
  HIP_SKIP_TEST("hipSignalExternalSemaphoresAsync does not validate the semaphore handle on the "
                "NVIDIA backend; the null-handle rejection contract cannot be exercised safely.");
#endif  // HT_AMD
}

// @asserts: hipWaitExternalSemaphoresAsync - rejects a batch containing a null semaphore handle with a non-success status
HIP_TEST_CASE(Contract_ExternalResource_WaitSemaphore_NullHandle_IsRejected) {
  // BACKEND-DIFF: On AMD the wait path validates the handle and rejects null. On
  // NVIDIA the behavior is CUDA-version-dependent: CUDA 13.x validates the handle
  // and returns cudaErrorInvalidValue (probe-confirmed on H100/CUDA 13.1), but
  // CUDA 12.x dereferences the null handle and faults (SIGSEGV, probe-confirmed on
  // V100/CUDA 12.9) - the same non-validating behavior as the signal path above.
  // The rejection contract can therefore be exercised on AMD and on NVIDIA with
  // CUDA >= 13.0, but must be skipped on older CUDA where it cannot be evaluated
  // safely. Parity would require null-handle validation in CUDA < 13.
#if HT_AMD || (defined(CUDA_VERSION) && CUDA_VERSION >= 13000)
  hip::contract::ContractCleanup cleanup;
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });

  // Waiting on a batch that contains a null external-semaphore handle must be
  // rejected rather than enqueuing an operation on an invalid object.
  hipExternalSemaphore_t semaphores[1] = {nullptr};
  hipExternalSemaphoreWaitParams params[1] = {};
  RequireRejected(hipWaitExternalSemaphoresAsync(semaphores, params, 1, stream));
#else
  HIP_SKIP_TEST("hipWaitExternalSemaphoresAsync does not validate the semaphore handle on the "
                "NVIDIA backend before CUDA 13.0; the null-handle rejection contract cannot be "
                "exercised safely.");
#endif  // HT_AMD || CUDA_VERSION >= 13000
}

// @asserts: hipExternalMemoryGetMappedMipmappedArray - rejects a null external-memory handle with a non-success status and no array
HIP_TEST_CASE(Contract_ExternalResource_GetMappedMipmappedArray_NullHandle_IsRejected) {
  // Mapping a mipmapped array from a null external-memory handle is invalid
  // input and must be rejected rather than returning a mipmapped array.
  hipExternalMemoryMipmappedArrayDesc desc{};
  desc.formatDesc = hipCreateChannelDesc(8, 0, 0, 0, hipChannelFormatKindUnsigned);
  desc.extent = make_hipExtent(16, 16, 0);
  desc.numLevels = 1;

  hipMipmappedArray_t mipmap = nullptr;
  const hipError_t status =
      hipExternalMemoryGetMappedMipmappedArray(&mipmap, nullptr, &desc);
  RequireRejected(status);
  REQUIRE(mipmap == nullptr);
}
