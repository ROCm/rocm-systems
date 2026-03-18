/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>

/**
 * @addtogroup hipDeviceGetDefaultMemPool hipDeviceGetDefaultMemPool
 * @{
 * @ingroup DeviceTest
 * `hipDeviceGetDefaultMemPool(hipMemPool_t* mem_pool, int device)` -
 * Returns the default memory pool of the specified device
 */

/**
 * Test Description
 * ------------------------
 *  - Check that MemPool can be fetched and is not `nullptr`.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceGetDefaultMemPool.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_hipDeviceGetDefaultMemPool_Positive_Basic) {
  const int device = GENERATE(range(0, HipTest::getDeviceCount()));

  int mem_pool_support = 0;
  HIP_CHECK(
      hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, device));
  if (!mem_pool_support) {
    HipTest::HIP_SKIP_TEST("Test only runs on devices with memory pool support");
    return;
  }

  hipMemPool_t mem_pool;
  HIP_CHECK(hipDeviceGetDefaultMemPool(&mem_pool, device));
  REQUIRE(mem_pool != nullptr);
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When output pointer to the MemPool is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When device ID is equal to -1
 *      - Expected output: return 'hipErrorInvalidDevice'
 *    -# When device ID is out of bounds
 *      - Expected output: return 'hipErrorInvalidDevice'
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceGetDefaultMemPool.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_hipDeviceGetDefaultMemPool_Negative_Parameters) {
  hipMemPool_t mem_pool;

  SECTION("mem_pool == nullptr") {
    HIP_CHECK_ERROR(hipDeviceGetDefaultMemPool(nullptr, 0), hipErrorInvalidValue);
  }

  SECTION("device < 0") {
    HIP_CHECK_ERROR(hipDeviceGetDefaultMemPool(&mem_pool, -1), hipErrorInvalidDevice);
  }

  SECTION("device ordinance too large") {
    HIP_CHECK_ERROR(hipDeviceGetDefaultMemPool(&mem_pool, HipTest::getDeviceCount()),
                    hipErrorInvalidDevice);
  }
}

/**
 * End doxygen group DeviceTest.
 * @}
 */
