/*
   Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
TEST_CASE("Unit_hipMemPoolDestroy_Negative_Parameter") {
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
