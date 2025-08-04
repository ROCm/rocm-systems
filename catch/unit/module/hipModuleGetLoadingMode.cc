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
 * @addtogroup hipModuleGetLoadingMode hipModuleGetLoadingMode
 * @{
 * @ingroup ModuleTest
 * `hhipError_t hipModuleGetLoadingMode(hipModuleLoadingMode_t* mode)` -
 * Function gets the current module load mode
 */

/**
 * Test Description
 * ------------------------
 * - Test case verifies the positive case of hipModuleGetLoadingMode API.
 * Test source
 * ------------------------
 * - catch/unit/module/hipModuleGetLoadingMode.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipModuleGetLoadingMode_Functional") {
  hipModuleLoadingMode_t mode;
  HIP_CHECK(hipModuleGetLoadingMode(&mode));
  REQUIRE(mode == HIP_MODULE_LAZY_LOADING);  // if env var HIP_MODULE_LOADING = lazy
  setenv("HIP_MODULE_LOADING", "EAGER", 1);
  HIP_CHECK(hipModuleGetLoadingMode(&mode));
  REQUIRE(mode == HIP_MODULE_EAGER_LOADING);  // if env var HIP_MODULE_LOADING = EAGER
  unsetenv("HIP_MODULE_LOADING");
}
/**
 * End doxygen group ModuleTest.
 * @}
 */

