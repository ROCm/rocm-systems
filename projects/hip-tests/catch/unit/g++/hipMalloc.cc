/*
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 * */

#include <hip_test_common.hh>

#include "hipMalloc.h"
/**
 * @addtogroup hipMalloc hipMalloc
 * @{
 * @ingroup MemoryTest
 * `hipError_t hipMalloc(void** ptr, size_t size)` -
 * Allocate memory on the default accelerator.
 * @}
 */

/**
 * Test Description
 * ------------------------
 *    - Allocate memory by using hipMalloc API and verify hipSuccess is returned.

 * Test source
 * ------------------------
 *    - catch/unit/g++/hipMalloc.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */

TEST_CASE("Unit_hipMalloc_gpptest") {
  printf("calling cpp function from here\n");
  int result = MallocFunc();
  REQUIRE(result == 1);
}

/**
 * End doxygen group MemoryTest.
 * @}
 */
