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
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>

/**
 * @addtogroup hipFuncGetAttributes
 * @{
 * @ingroup ModuleTest
 * `hipError_t hipFuncGetAttributes(struct hipFuncAttributes* attr, const void* func)` -
 * Find out attributes for a given function
 */

/**
 * Test Description
 * ------------------------
 * - Test case to Find out attributes for a given function.

 * Test source
 * ------------------------
 * - catch/unit/module/hipFuncGetAttributes.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 5.6
 */

__global__ void getAttrFn(float* px, float* py) {
  *px = *px + 1.0f;
  *py = *py + *px;
}

TEST_CASE("Unit_hipFuncGetAttributes_basic") {
  hipFuncAttributes attr{};

  auto r = hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(&getAttrFn));
  REQUIRE(r == hipSuccess);
  REQUIRE(attr.maxThreadsPerBlock != 0);
}
