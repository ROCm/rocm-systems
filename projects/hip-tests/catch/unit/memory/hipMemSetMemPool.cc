/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include "mempool_common.hh"

/**
 * @addtogroup hipMemSetMemPool hipMemSetMemPool
 * @{
 * @ingroup MemoryTest
 * `hipError_t hipMemSetMemPool(hipMemLocation* location, hipMemAllocationType type,
                                hipMemPool_t pool)` -
 *  Sets the current memory pool for the location and allocation type.
 */


HIP_TEST_CASE(Unit_hipMemSetMemPool_Negative) {
  int dev;
  HIP_CHECK(hipGetDevice(&dev));
  checkMempoolSupported(dev);

  hipMemPool_t mem_pool;
  hipMemPoolProps prop{};
  prop.allocType = hipMemAllocationTypePinned;
  prop.location.id = dev;
  prop.location.type = hipMemLocationTypeDevice;
  HIP_CHECK(hipMemPoolCreate(&mem_pool, &prop));

  hipMemLocation location{};
  location.id = dev;
  location.type = hipMemLocationTypeDevice;

  SECTION("Invalid location") {
    HIP_CHECK_ERROR(hipMemSetMemPool(nullptr, hipMemAllocationTypePinned, mem_pool),
                    hipErrorInvalidValue);

    location.id = -1;
    HIP_CHECK_ERROR(hipMemSetMemPool(&location, hipMemAllocationTypePinned, mem_pool),
                    hipErrorInvalidValue);

    location.id = dev;
    location.type = hipMemLocationTypeNone;
    HIP_CHECK_ERROR(hipMemSetMemPool(&location, hipMemAllocationTypePinned, mem_pool),
                    hipErrorInvalidValue);
  }

  SECTION("Invalid pool") {
    HIP_CHECK_ERROR(hipMemSetMemPool(&location, hipMemAllocationTypePinned, nullptr),
                    hipErrorInvalidValue);

    // Pool device and location device do not match
    int dev_cnt = 0;
    HIP_CHECK(hipGetDeviceCount(&dev_cnt));
    if (dev_cnt > 1) {
      hipMemPool_t mem_pool2;
      prop.allocType = hipMemAllocationTypePinned;
      prop.location.id = dev + 1;
      prop.location.type = hipMemLocationTypeDevice;
      HIP_CHECK(hipMemPoolCreate(&mem_pool2, &prop));
      HIP_CHECK_ERROR(hipMemSetMemPool(&location, hipMemAllocationTypePinned, mem_pool2),
                      hipErrorInvalidValue);
      HIP_CHECK(hipMemPoolDestroy(mem_pool2));
    }
  }

  SECTION("Using destroyed pool") {
    // Create a temporary pool
    hipMemPool_t temp_pool;
    prop.allocType = hipMemAllocationTypePinned;
    prop.location.id = dev;
    prop.location.type = hipMemLocationTypeDevice;
    HIP_CHECK(hipMemPoolCreate(&temp_pool, &prop));

    // Destroy it
    HIP_CHECK(hipMemPoolDestroy(temp_pool));

    // Try to set the destroyed pool - should fail
    HIP_CHECK_ERROR(hipMemSetMemPool(&location, hipMemAllocationTypePinned, temp_pool),
                    hipErrorInvalidValue);
  }

  SECTION("Invalid allocation type") {
    HIP_CHECK_ERROR(hipMemSetMemPool(&location, hipMemAllocationTypeInvalid, mem_pool),
                    hipErrorInvalidValue);

    // Different than the one pool got created for
    HIP_CHECK_ERROR(hipMemSetMemPool(&location, hipMemAllocationTypeManaged, mem_pool),
                    hipErrorInvalidValue);
  }

  HIP_CHECK(hipMemPoolDestroy(mem_pool));
}

HIP_TEST_CASE(Unit_hipMemSetMemPool_Basic) {
  int num_devices;
  HIP_CHECK(hipGetDeviceCount(&num_devices));

  auto alloc_type = GENERATE(hipMemAllocationTypePinned, hipMemAllocationTypeManaged);

  for (int dev = 0; dev < num_devices; dev++) {
    checkMempoolSupported(dev);
    HIP_CHECK(hipSetDevice(dev));

    hipMemPool_t mem_pool, curr_mem_pool;
    hipMemPoolProps prop{};
    prop.allocType = alloc_type;
    prop.location.id = dev;
    prop.location.type = hipMemLocationTypeDevice;
    HIP_CHECK(hipMemPoolCreate(&mem_pool, &prop));

    hipMemLocation location{};
    location.id = dev;
    location.type = hipMemLocationTypeDevice;
    HIP_CHECK(hipMemSetMemPool(&location, alloc_type, mem_pool));

    HIP_CHECK(hipMemGetMemPool(&curr_mem_pool, &location, alloc_type));
    REQUIRE(curr_mem_pool == mem_pool);

    HIP_CHECK(hipMemPoolDestroy(mem_pool));
  }
}
