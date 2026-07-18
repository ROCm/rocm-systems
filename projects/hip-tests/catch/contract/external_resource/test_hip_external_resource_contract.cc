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

HIP_TEST_CASE(Contract_ExternalResource_DestroyMemory_NullHandle_IsRejected) {
  // Destroying a null external-memory handle is invalid input and must be
  // rejected rather than silently succeeding.
  RequireRejected(hipDestroyExternalMemory(nullptr));
}

HIP_TEST_CASE(Contract_ExternalResource_DestroySemaphore_NullHandle_IsRejected) {
  // Destroying a null external-semaphore handle is invalid input and must be
  // rejected rather than silently succeeding.
  RequireRejected(hipDestroyExternalSemaphore(nullptr));
}

HIP_TEST_CASE(Contract_ExternalResource_SignalSemaphore_NullHandle_IsRejected) {
  // BACKEND-DIFF: The null-handle rejection contract is only exercised on AMD. On
  // NVIDIA hipSignalExternalSemaphoresAsync maps to
  // cudaSignalExternalSemaphoresAsync, which does not validate the semaphore
  // handle and dereferences it - a null handle faults (SIGSEGV) instead of
  // returning a defined error - so the rejection contract cannot be evaluated
  // safely there. (The matching wait API does validate its handle, so
  // WaitSemaphore_NullHandle stays cross-backend.) Parity would require matching
  // null-handle validation on the signal path.
#ifdef __HIP_PLATFORM_AMD__
  hip::contract::ContractCleanup cleanup;
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([&] { (void)hipStreamDestroy(stream); });

  // Signalling a batch that contains a null external-semaphore handle must be
  // rejected rather than enqueuing an operation on an invalid object.
  hipExternalSemaphore_t semaphores[1] = {nullptr};
  hipExternalSemaphoreSignalParams params[1] = {};
  RequireRejected(hipSignalExternalSemaphoresAsync(semaphores, params, 1, stream));
#else
  HIP_SKIP_TEST("hipSignalExternalSemaphoresAsync does not validate the semaphore handle on the "
                "NVIDIA backend; the null-handle rejection contract cannot be exercised safely.");
#endif  // __HIP_PLATFORM_AMD__
}

HIP_TEST_CASE(Contract_ExternalResource_WaitSemaphore_NullHandle_IsRejected) {
  hip::contract::ContractCleanup cleanup;
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([&] { (void)hipStreamDestroy(stream); });

  // Waiting on a batch that contains a null external-semaphore handle must be
  // rejected rather than enqueuing an operation on an invalid object.
  hipExternalSemaphore_t semaphores[1] = {nullptr};
  hipExternalSemaphoreWaitParams params[1] = {};
  RequireRejected(hipWaitExternalSemaphoresAsync(semaphores, params, 1, stream));
}

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
