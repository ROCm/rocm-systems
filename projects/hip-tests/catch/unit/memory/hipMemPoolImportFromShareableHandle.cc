/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipMemPoolImportFromShareableHandle hipMemPoolImportFromShareableHandle
 * @{
 * @ingroup MemoryTest
 * `hipError_t hipMemPoolImportFromShareableHandle(
                                                 hipMemPool_t*              mem_pool,
                                                 void*                      shared_handle,
                                                 hipMemAllocationHandleType handle_type,
                                                 unsigned int               flags) ` -
 * Imports a memory pool from a shared handle.
 */

#include "mempool_common.hh"

/**
 * Test Description
 * ------------------------
 *    - Negative Tests for hipMemPoolImportFromShareableHandle.
 * ------------------------
 *    - unit/memory/hipMemPoolImportFromShareableHandle.cc
 * Test requirements
 * ------------------------
 *    - Host specific (LINUX)
 *    - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipMemPoolImportFromShareableHandle_Negative) {
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

  HIP_CHECK(hipMemPoolExportToShareableHandle(&sharedHandle, mempoolPfd,
                                              hipMemHandleTypePosixFileDescriptor, 0));
  hipMemPool_t mempoolImp;
  SECTION("Passing nullptr as imported mempool") {
    HIP_CHECK_ERROR(hipMemPoolImportFromShareableHandle(nullptr, (void*)sharedHandle,
                                                        hipMemHandleTypePosixFileDescriptor, 0),
                    hipErrorInvalidValue);
  }
  SECTION("Passing nullptr as handle") {
    HIP_CHECK_ERROR(hipMemPoolImportFromShareableHandle(&mempoolImp, nullptr,
                                                        hipMemHandleTypePosixFileDescriptor, 0),
                    hipErrorInvalidValue);
  }
  SECTION("Passing invalid handle type") {
    HIP_CHECK_ERROR(hipMemPoolImportFromShareableHandle(&mempoolImp, (void*)sharedHandle,
                                                        hipMemHandleTypeNone, 0),
                    hipErrorInvalidValue);
  }
  HIP_CHECK(hipMemPoolDestroy(mempoolPfd));
}

/**
 * End doxygen group MemoryTest.
 * @}
 */
