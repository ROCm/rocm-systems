/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipMemPoolImportPointer hipMemPoolImportPointer
 * @{
 * @ingroup MemoryTest
 * `hipError_t hipMemPoolImportPointer(
                                      void**                   dev_ptr,
                                      hipMemPool_t             mem_pool,
                                      hipMemPoolPtrExportData* export_data) ` -
 * Import a memory pool allocation from another process.
 */

#include "mempool_common.hh"

constexpr int DATA_SIZE = 1024 * 1024;
constexpr size_t byte_size = DATA_SIZE * sizeof(int);

/**
 * Test Description
 * ------------------------
 *    - Negative Tests for hipMemPoolImportPointer.
 * ------------------------
 *    - unit/memory/hipMemPoolImportPointer.cc
 * Test requirements
 * ------------------------
 *    - Host specific (LINUX)
 *    - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipMemPoolImportPointer_Negative) {
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

  // Create mempool with OS specific handle type
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
  HIP_CHECK(hipMemPoolExportPointer(&ptrExp, A_d));
  hipMemPool_t mempoolImp;
  HIP_CHECK(hipMemPoolImportFromShareableHandle(&mempoolImp, (void*)sharedHandle,
                                                handleType, 0));
  void* ptrImp;
  SECTION("Passing nullptr as import data") {
    HIP_CHECK_ERROR(hipMemPoolImportPointer(nullptr, mempoolImp, &ptrExp), hipErrorInvalidValue);
  }
  SECTION("Passing nullptr as imported mempool") {
    HIP_CHECK_ERROR(hipMemPoolImportPointer(&ptrImp, nullptr, &ptrExp), hipErrorInvalidValue);
  }
  SECTION("Passing nullptr as exported pointer") {
    HIP_CHECK_ERROR(hipMemPoolImportPointer(&ptrImp, mempoolImp, nullptr), hipErrorInvalidValue);
  }
  HIP_CHECK(hipFree(reinterpret_cast<void*>(A_d)));
  HIP_CHECK(hipMemPoolDestroy(mempoolPfd));
  HIP_CHECK(hipMemPoolDestroy(mempoolImp));
}

/**
 * Test Description
 * ------------------------
 *    - Test hipMemPoolImportPointer while a stream is capturing. The API is allowed
 * in relaxed capture mode and must return hipErrorStreamCaptureUnsupported in the
 * global and thread-local capture modes.
 * ------------------------
 *    - unit/memory/hipMemPoolImportPointer.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipMemPoolImportPointer_Capture) {
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

  hipShareableHdl sharedHandle;
  HIP_CHECK(hipMemPoolExportToShareableHandle(&sharedHandle, mempool, handleType, 0));
  hipMemPoolPtrExportData ptrExp;
  HIP_CHECK(hipMemPoolExportPointer(&ptrExp, A_d));
  hipMemPool_t mempoolImp;
  HIP_CHECK(hipMemPoolImportFromShareableHandle(&mempoolImp, (void*)sharedHandle, handleType, 0));

  void* ptrImp = nullptr;
  hipError_t capture_err = hipSuccess;
  constexpr bool kRelaxedModeAllowed = true;
  BEGIN_CAPTURE_SYNC(capture_err, kRelaxedModeAllowed);
  HIP_CHECK_ERROR(hipMemPoolImportPointer(&ptrImp, mempoolImp, &ptrExp), capture_err);
  END_CAPTURE_SYNC(capture_err);

  if (ptrImp != nullptr) {
    HIP_CHECK(hipFree(ptrImp));
  }
  HIP_CHECK(hipFree(reinterpret_cast<void*>(A_d)));
  HIP_CHECK(hipMemPoolDestroy(mempoolImp));
  HIP_CHECK(hipMemPoolDestroy(mempool));
}

/**
 * End doxygen group MemoryTest.
 * @}
 */
