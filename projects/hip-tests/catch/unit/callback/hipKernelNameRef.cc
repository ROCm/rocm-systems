/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
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

/**
 * @addtogroup hipKernelNameRef hipKernelNameRef
 * @{
 * @ingroup CallbackTest
 * `hipKernelNameRef(const hipFunction_t f)` -
 * returns the name of passed function object
 */

/**
 * Test Description
 * ------------------------
 *  - Loads the simple kernel function from the matching module
 *  - Checks that the valid name is returned for the loaded kernel function
 * Test source
 * ------------------------
 *  - unit/callback/hipKernelNameRef.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 *  - Platform specific (AMD)
 */
TEST_CASE("Unit_hipKernelNameRef_Positive_Basic") {
  hipModule_t kernel_module{nullptr};
  hipFunction_t kernel_function{nullptr};

  HIP_CHECK(hipModuleLoad(&kernel_module, "SimpleKernel.code"));
  HIP_CHECK(hipModuleGetFunction(&kernel_function, kernel_module, "simple_kernel"));
  REQUIRE(hipKernelNameRef(kernel_function) != nullptr);
  HIP_CHECK(hipModuleUnload(kernel_module));
}

/**
 * Test Description
 * ------------------------
 *  - Checks that the API returns nullptr if the passed function is not loaded
 * Test source
 * ------------------------
 *  - unit/callback/hipKernelNameRef.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 *  - Platform specific (AMD)
 */
TEST_CASE("Unit_hipKernelNameRef_Negative_Parameters") {
  hipFunction_t kernel_function{nullptr};
  REQUIRE(hipKernelNameRef(kernel_function) == nullptr);
}

/**
 * End doxygen group CallbackTest.
 * @}
 */
