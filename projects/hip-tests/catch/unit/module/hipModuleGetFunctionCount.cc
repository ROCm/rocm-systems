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
#include "hip_module_common.hh"
#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
/**
 * @addtogroup hipModuleGetFunctionCount hipModuleGetFunctionCount
 * @{
 * @ingroup ModuleTest
 * `hipError_t hipModuleGetFunctionCount (unsigned int* count, hipModule_t mod)`
 * - Returns the number of functions within a module
 */

/**
 * Test Description
 * ------------------------
 * - Test case verifies the below positive cases of hipModuleGetFunctionCount
 * API.
 * - Get the device count from different kinds of modules.
 * 1. Module is built for Single architecture which contain Single global
 * function
 * 2. Module is built for multi architecture which contain Single global
 * function
 * 3. Empty Module which doesn't contain any function.
 * 4. Module which contain both __global__, __device__ functions in it.
 * 5. RTC module.
 * Test source
 * ------------------------
 * - catch/unit/module/hipModuleGetFunctionCount.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipModuleGetFunctionCount_Functional") {
  hipModule_t moduleSingleArch, moduleEmpty, doubleKernelModule, rtcModule;
  unsigned int count = 0;
  SECTION("Single arch, Single global function") {
    HIP_CHECK(hipModuleLoad(&moduleSingleArch, "vcpy_kernel.code"));
    HIP_CHECK(hipModuleGetFunctionCount(&count, moduleSingleArch));
    REQUIRE(count == 1);
    HIP_CHECK(hipModuleUnload(moduleSingleArch));
  }
#if HT_AMD
  SECTION("Multi arch, Single global function") {
    hipModule_t moduleMultiArch;
    const auto loaded_module =
        LoadModuleIntoBuffer("copyKernelCompressed.code");
    HIP_CHECK(hipModuleLoadData(&moduleMultiArch, loaded_module.data()));
    HIP_CHECK(hipModuleGetFunctionCount(&count, moduleMultiArch));
    REQUIRE(count == 1);
    HIP_CHECK(hipModuleUnload(moduleMultiArch));
  }
#endif
  SECTION("Empty Module Count") {
    HIP_CHECK(hipModuleLoad(&moduleEmpty, "emptyModuleCount.code"));
    HIP_CHECK(hipModuleGetFunctionCount(&count, moduleEmpty));
    REQUIRE(count == 0);
    HIP_CHECK(hipModuleUnload(moduleEmpty));
  }
  SECTION("__global__, __device__ functions module") {
    HIP_CHECK(hipModuleLoad(&doubleKernelModule, "kernel_count.code"));
    HIP_CHECK(hipModuleGetFunctionCount(&count, doubleKernelModule));
    REQUIRE(count == 1);
    HIP_CHECK(hipModuleUnload(doubleKernelModule));
  }

  SECTION("Load RTCd module") {
    const auto rtc =
        CreateRTCCharArray(R"(extern "C" __global__ void kernel() {})");
    HIP_CHECK(hipModuleLoadData(&rtcModule, rtc.data()));
    REQUIRE(rtcModule != nullptr);
    HIP_CHECK(hipModuleGetFunctionCount(&count, rtcModule));
    REQUIRE(count == 1);
    HIP_CHECK(hipModuleUnload(rtcModule));
  }
}
/**
 * Test Description
 * ------------------------
 * - Test case verifies the negative case of hipModuleGetFunctionCount API.
 * Test source
 * ------------------------
 * - catch/unit/module/hipModuleGetFunctionCount.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipModuleGetFunctionCount_NegativeTsts") {
  unsigned int count = 0;
  SECTION("Input module as nullptr") {
    HIP_CHECK_ERROR(hipModuleGetFunctionCount(&count, nullptr),
                    hipErrorInvalidHandle);
  }
}
/**
 * End doxygen group ModuleTest.
 * @}
 */
