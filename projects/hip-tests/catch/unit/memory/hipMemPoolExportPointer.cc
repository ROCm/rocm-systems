/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipMemPoolExportPointer hipMemPoolExportPointer
 * @{
 * @ingroup MemoryTest
 * `hipError_t hipMemPoolExportPointer(hipMemPoolPtrExportData* export_data, void* dev_ptr) ` -
 * Export a memory pool allocation to another process.
 */

#include "mempool_common.hh"

constexpr int DATA_SIZE = 1024 * 1024;
constexpr size_t byte_size = DATA_SIZE * sizeof(int);

/**
 * Test Description
 * ------------------------
 *    - Negative Tests for hipMemPoolExportPointer.
 * ------------------------
 *    - unit/memory/hipMemPoolExportPointer.cc
 * Test requirements
 * ------------------------
 *    - Host specific
 *    - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipMemPoolExportPointer_Negative) {
  hipMemPoolPtrExportData ptrExp;
  hipShareableHdl sharedHandle;
  hipMemPoolProps pool_props{};
  hipMemPool_t mempoolPfd;
  checkMempoolSupported(0)

  #if HT_WIN
  hipMemAllocationHandleType handleType = hipMemHandleTypeWin32;
  #else
  hipMemAllocationHandleType handleType = hipMemHandleTypePosixFileDescriptor;
  #endif

  // Create mempool with Posix File Descriptor
  pool_props.allocType = hipMemAllocationTypePinned;
  pool_props.location.id = 0;
  pool_props.location.type = hipMemLocationTypeDevice;
  pool_props.handleTypes = handleType;
  HIP_CHECK(hipMemPoolCreate(&mempoolPfd, &pool_props));
  int* A_d;
  HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&A_d), byte_size, mempoolPfd, 0));
  HIP_CHECK(hipStreamSynchronize(0));
  HIP_CHECK(hipMemPoolExportToShareableHandle(&sharedHandle, mempoolPfd,
                                              handleType, 0));
  SECTION("Passing nullptr as export data") {
    HIP_CHECK_ERROR(hipMemPoolExportPointer(nullptr, A_d), hipErrorInvalidValue);
  }
  SECTION("Passing nullptr as device memory ptr") {
    HIP_CHECK_ERROR(hipMemPoolExportPointer(&ptrExp, nullptr), hipErrorInvalidValue);
  }
  HIP_CHECK(hipFree(reinterpret_cast<void*>(A_d)));
  HIP_CHECK(hipMemPoolDestroy(mempoolPfd));
}

/**
 * Test Description
 * ------------------------
 *    - Test hipMemPoolExportPointer while a stream is capturing. The API is allowed
 * in relaxed capture mode and must return hipErrorStreamCaptureUnsupported in the
 * global and thread-local capture modes.
 * ------------------------
 *    - unit/memory/hipMemPoolExportPointer.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipMemPoolExportPointer_Capture) {
  checkMempoolSupported(0)
  HIP_CHECK(hipSetDevice(0));

#if HT_WIN
  hipMemAllocationHandleType handleType = hipMemHandleTypeWin32;
#else
  hipMemAllocationHandleType handleType = hipMemHandleTypePosixFileDescriptor;
#endif

  hipMemPoolProps pool_props{};
  pool_props.allocType = hipMemAllocationTypePinned;
  pool_props.location.id = 0;
  pool_props.location.type = hipMemLocationTypeDevice;
  pool_props.handleTypes = handleType;
  hipMemPool_t mempool;
  HIP_CHECK(hipMemPoolCreate(&mempool, &pool_props));

  int* A_d = nullptr;
  HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&A_d), byte_size, mempool, 0));
  HIP_CHECK(hipStreamSynchronize(0));

  hipMemPoolPtrExportData ptrExp;
  hipError_t capture_err = hipSuccess;
  constexpr bool kRelaxedModeAllowed = true;
  BEGIN_CAPTURE_SYNC(capture_err, kRelaxedModeAllowed);
  HIP_CHECK_ERROR(hipMemPoolExportPointer(&ptrExp, A_d), capture_err);
  END_CAPTURE_SYNC(capture_err);

  HIP_CHECK(hipFree(reinterpret_cast<void*>(A_d)));
  HIP_CHECK(hipMemPoolDestroy(mempool));
}

/**
 * End doxygen group MemoryTest.
 * @}
 */
