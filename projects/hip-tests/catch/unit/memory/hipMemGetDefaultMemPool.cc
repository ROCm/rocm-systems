/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of intge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>
#include "mempool_common.hh"

/**
 * @addtogroup hipMemGetDefaultMemPool hipMemGetDefaultMemPool
 * @{
 * @ingroup MemoryTest
 * `hipError_t hipMemGetDefaultMemPool(hipMemPool_t* memPool, hipMemLocation* location,
                                       hipMemAllocationType type)` -
 *  Gets the default memory pool for the location and allocation type.
 */

TEST_CASE("Unit_hipMemGetDefaultMemPool_Negative") {
  int dev;
  HIP_CHECK(hipGetDevice(&dev));

  hipMemPool_t memPool;
  hipMemLocation location{};
  location.id = dev;
  location.type = hipMemLocationTypeDevice;
  hipMemAllocationType allocationType = hipMemAllocationTypePinned;

  SECTION("Invalid memPool") {
    HIP_CHECK_ERROR(hipMemGetDefaultMemPool(nullptr, &location, allocationType),
                    hipErrorInvalidValue);
  }

  SECTION("Invalid location") {
    HIP_CHECK_ERROR(hipMemGetDefaultMemPool(&memPool, nullptr, allocationType),
                    hipErrorInvalidValue);

    location.id = -1;
    HIP_CHECK_ERROR(hipMemGetDefaultMemPool(&memPool, &location, allocationType),
                    hipErrorInvalidValue);

    location.id = dev;
    location.type = hipMemLocationTypeNone;
    HIP_CHECK_ERROR(hipMemGetDefaultMemPool(&memPool, &location, allocationType),
                    hipErrorInvalidValue);
  }

  SECTION("Invalid allocation type") {
    HIP_CHECK_ERROR(hipMemGetDefaultMemPool(&memPool, &location, hipMemAllocationTypeInvalid),
                    hipErrorInvalidValue);
  }
}

TEST_CASE("Unit_hipMemGetDefaultMemPool_Basic") {
  int dev;
  HIP_CHECK(hipGetDevice(&dev));

  hipMemLocation location{};
  location.id = dev;
  location.type = hipMemLocationTypeDevice;

  SECTION("Pinned") {
    auto alloc_type = hipMemAllocationTypePinned;
    hipMemPool_t memPool;
    hipMemPool_t deviceMemPool;


    HIP_CHECK(hipMemGetDefaultMemPool(&memPool, &location, alloc_type));
    REQUIRE(memPool != nullptr);
    HIP_CHECK(hipDeviceGetDefaultMemPool(&deviceMemPool, dev));
    REQUIRE(deviceMemPool != nullptr);
    REQUIRE(memPool == deviceMemPool);
  }

  SECTION("Managed") {
    auto alloc_type = hipMemAllocationTypeManaged;
    hipMemPool_t memPool;
    HIP_CHECK(hipMemGetDefaultMemPool(&memPool, &location, alloc_type));
    REQUIRE(memPool != nullptr);
  }
}
