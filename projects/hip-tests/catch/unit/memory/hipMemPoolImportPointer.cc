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

      // Create mempool with Posix File Descriptor
      pool_props.allocType = hipMemAllocationTypePinned;
  pool_props.location.id = 0;
  pool_props.location.type = hipMemLocationTypeDevice;
  pool_props.handleTypes = hipMemHandleTypePosixFileDescriptor;
  HIP_CHECK(hipMemPoolCreate(&mempoolPfd, &pool_props));
  int* A_d;
  HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&A_d), byte_size, mempoolPfd, 0));
  HIP_CHECK(hipStreamSynchronize(0));
  HIP_CHECK(hipMemPoolExportToShareableHandle(&sharedHandle, mempoolPfd,
                                              hipMemHandleTypePosixFileDescriptor, 0));
  HIP_CHECK(hipMemPoolExportPointer(&ptrExp, A_d));
  hipMemPool_t mempoolImp;
  HIP_CHECK(hipMemPoolImportFromShareableHandle(&mempoolImp, (void*)sharedHandle,
                                                hipMemHandleTypePosixFileDescriptor, 0));
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
 * End doxygen group MemoryTest.
 * @}
 */
