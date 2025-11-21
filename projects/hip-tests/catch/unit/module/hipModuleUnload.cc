/*
Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <hip/hip_runtime_api.h>

TEST_CASE("Unit_hipModuleUnload_Negative_Module_Is_Nullptr") {
  HIP_CHECK(hipFree(nullptr));

  HIP_CHECK_ERROR(hipModuleUnload(nullptr), hipErrorInvalidResourceHandle);
}

TEST_CASE("Unit_hipModuleUnload_Negative_Double_Unload") {
  HIP_CHECK(hipFree(nullptr));

  hipModule_t module = nullptr;
  HIP_CHECK(hipModuleLoad(&module, "empty_module.code"));
  HIP_CHECK(hipModuleUnload(module));
  HIP_CHECK_ERROR(hipModuleUnload(module), hipErrorNotFound);
}
/**
 * @addtogroup hipModuleUnload
 * @{
 * @ingroup ModuleTest
 * `hipError_t hipModuleUnload(hipModule_t module)` -
 * Frees the module
 */

/**
 * Test Description
 * ------------------------
 * - Test case to verify the module release.
 * Test source
 * ------------------------
 * - catch/unit/module/hipModuleUnload.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 5.6
 */
TEST_CASE("Unit_hipModuleLoad_basic") {
  constexpr auto fileName = "vcpy_kernel.code";
  hipModule_t module;
  HIP_CHECK(hipModuleLoad(&module, fileName));
  REQUIRE(module != nullptr);
  HIP_CHECK(hipModuleUnload(module));
}
