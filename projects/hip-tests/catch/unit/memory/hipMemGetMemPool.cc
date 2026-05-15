/*
   Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:
   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
   IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
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
