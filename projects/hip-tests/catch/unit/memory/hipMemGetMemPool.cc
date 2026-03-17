/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include "mempool_common.hh"

/**
 * @addtogroup hipMemGetMemPool hipMemGetMemPool
 * @{
 * @ingroup MemoryTest
 * `hipError_t hipMemGetMemPool(hipMemPool_t* pool, hipMemLocation* location,
                                hipMemAllocationType type)` -
 *  Gets the current memory pool for the location and allocation type.
 */

TEST_CASE("Unit_hipMemGetMemPool_Negative") {
  int dev;
  HIP_CHECK(hipGetDevice(&dev));

  hipMemPool_t pool;
  hipMemLocation location{};
  location.id = dev;
  location.type = hipMemLocationTypeDevice;


  SECTION("Invalid pool") {
    HIP_CHECK_ERROR(hipMemGetMemPool(nullptr, &location, hipMemAllocationTypePinned),
                    hipErrorInvalidValue);
  }

  SECTION("Invalid location") {
    HIP_CHECK_ERROR(hipMemGetMemPool(&pool, nullptr, hipMemAllocationTypePinned),
                    hipErrorInvalidValue);

    location.id = -1;
    HIP_CHECK_ERROR(hipMemGetMemPool(&pool, &location, hipMemAllocationTypePinned),
                    hipErrorInvalidValue);

    location.id = dev;
    location.type = hipMemLocationTypeNone;
    HIP_CHECK_ERROR(hipMemGetMemPool(&pool, &location, hipMemAllocationTypePinned),
                    hipErrorInvalidValue);
  }

  SECTION("Invalid allocation type") {
    HIP_CHECK_ERROR(hipMemGetMemPool(&pool, &location, hipMemAllocationTypeInvalid),
                    hipErrorInvalidValue);
  }
}

TEST_CASE("Unit_hipMemGetMemPool_Basic") {
  int dev;
  HIP_CHECK(hipGetDevice(&dev));

  auto alloc_type = GENERATE(hipMemAllocationTypePinned, hipMemAllocationTypeManaged);

  hipMemPool_t mem_pool, curr_mem_pool;
  hipMemPoolProps prop{};
  prop.allocType = alloc_type;
  prop.location.id = dev;
  prop.location.type = hipMemLocationTypeDevice;
  HIP_CHECK(hipMemPoolCreate(&mem_pool, &prop));

  hipMemLocation location{};
  location.id = dev;
  location.type = hipMemLocationTypeDevice;
  HIP_CHECK(hipMemGetMemPool(&curr_mem_pool, &location, alloc_type));
  REQUIRE(curr_mem_pool != nullptr);

  HIP_CHECK(hipMemSetMemPool(&location, alloc_type, mem_pool));
  HIP_CHECK(hipMemGetMemPool(&curr_mem_pool, &location, alloc_type));
  REQUIRE(curr_mem_pool == mem_pool);

  HIP_CHECK(hipMemPoolDestroy(mem_pool));
}
