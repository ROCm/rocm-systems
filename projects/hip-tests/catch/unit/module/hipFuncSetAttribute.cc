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
 * @addtogroup hipFuncSetAttribute
 * @{
 * @ingroup ModuleTest
 * `hipError_t hipFuncSetAttribute(const void* func, hipFuncAttribute attr, int value)` -
 * Set attributes for a specific function
 */

/**
 * Test Description
 * ------------------------
 * - Test case to set attributes for a specific function

 * Test source
 * ------------------------
 * - catch/unit/module/hipFuncSetAttribute.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 5.6
*/

__global__ void fn(float* px, float* py) {
  *px = *px + 1.0f;
  *py = *py + *px;
}

TEST_CASE("Unit_hipFuncSetAttribute_Basic") {
  HIP_CHECK(hipFuncSetAttribute(reinterpret_cast<const void*>(&fn),
                                hipFuncAttributeMaxDynamicSharedMemorySize, 0));
  HIP_CHECK(hipFuncSetAttribute(reinterpret_cast<const void*>(&fn),
                                hipFuncAttributePreferredSharedMemoryCarveout, 0));
}
