/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "mempool_common.hh"

/**
 * @addtogroup hipMemPoolDestroy hipMemPoolDestroy
 * @{
 * @ingroup StreamOTest
 * `hipMemPoolDestroy(hipMemPool_t mem_pool)` -
 * Destroys the specified memory pool
 */

/**
 * Test Description
 * ------------------------
 *  - Test to verify hipMemPoolCreate behavior with invalid arguments:
 *    -# Nullptr mem_pool
 *    -# Double hipMemPoolDestroy
 *    -# Attempt to destroy default mempool
 *
 * Test source
 * ------------------------
 *  - /unit/memory/hipMemPoolDestroy.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE(Unit_hipMemPoolDestroy_Negative_Parameter) {
  checkMempoolSupported(0)

      hipMemPool_t mem_pool = nullptr;

  SECTION("Passing nullptr to mempool") {
    HIP_CHECK_ERROR(hipMemPoolDestroy(nullptr), hipErrorInvalidValue);
  }

  SECTION("Double hipMemPoolDestroy") {
    hipMemPoolProps kPoolProps;
    memset(&kPoolProps, 0, sizeof(kPoolProps));
    kPoolProps.allocType = hipMemAllocationTypePinned;
    kPoolProps.handleTypes = hipMemHandleTypeNone;
    kPoolProps.location.type = hipMemLocationTypeDevice;
    kPoolProps.location.id = 0;
    kPoolProps.win32SecurityAttributes = nullptr;
    HIP_CHECK(hipMemPoolCreate(&mem_pool, &kPoolProps));
    HIP_CHECK(hipMemPoolDestroy(mem_pool));
    HIP_CHECK_ERROR(hipMemPoolDestroy(mem_pool), hipErrorInvalidValue);
  }

  SECTION("Attempt to destroy default mempool") {
    hipMemPool_t default_mem_pool = nullptr;
    int device = 0;
    HIP_CHECK(hipDeviceGetDefaultMemPool(&default_mem_pool, device));
    HIP_CHECK_ERROR(hipMemPoolDestroy(default_mem_pool), hipErrorInvalidValue);
  }
}

/**
 * End doxygen group StreamOTest.
 * @}
 */
