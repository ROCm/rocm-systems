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

  HIP_CHECK(hipMemPoolExportToShareableHandle(&sharedHandle, mempoolPfd,
                                              handleType, 0));
  hipMemPool_t mempoolImp;
  SECTION("Passing nullptr as imported mempool") {
    HIP_CHECK_ERROR(hipMemPoolImportFromShareableHandle(nullptr, (void*)sharedHandle,
                                                        handleType, 0),
                    hipErrorInvalidValue);
  }
  SECTION("Passing nullptr as handle") {
    HIP_CHECK_ERROR(hipMemPoolImportFromShareableHandle(&mempoolImp, nullptr,
                                                        handleType, 0),
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
 * Test Description
 * ------------------------
 *    - Test hipMemPoolImportFromShareableHandle while a stream is capturing. The
 * API is allowed in relaxed capture mode and must return
 * hipErrorStreamCaptureUnsupported in the global and thread-local capture modes.
 * ------------------------
 *    - unit/memory/hipMemPoolImportFromShareableHandle.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipMemPoolImportFromShareableHandle_Capture) {
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

  hipShareableHdl sharedHandle;
  HIP_CHECK(hipMemPoolExportToShareableHandle(&sharedHandle, mempool, handleType, 0));

  hipMemPool_t mempoolImp = nullptr;
  hipError_t capture_err = hipSuccess;
  constexpr bool kRelaxedModeAllowed = true;
  BEGIN_CAPTURE_SYNC(capture_err, kRelaxedModeAllowed);
  HIP_CHECK_ERROR(
      hipMemPoolImportFromShareableHandle(&mempoolImp, (void*)sharedHandle, handleType, 0),
      capture_err);
  END_CAPTURE_SYNC(capture_err);

  if (mempoolImp != nullptr) {
    HIP_CHECK(hipMemPoolDestroy(mempoolImp));
  }
  HIP_CHECK(hipMemPoolDestroy(mempool));
}

/**
 * End doxygen group MemoryTest.
 * @}
 */
