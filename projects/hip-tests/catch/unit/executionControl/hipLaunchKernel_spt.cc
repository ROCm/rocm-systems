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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include "execution_control_common.hh"
#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#include <resource_guards.hh>
#include <utils.hh>
/**
 * @addtogroup hipLaunchKernel_spt hipLaunchKernel_spt
 * @{
 * @ingroup ExecutionTest
 * `hipError_t hipLaunchKernel_spt(const void* function_address,
                           dim3 numBlocks,
                           dim3 dimBlocks,
                           void** args,
                           size_t sharedMemBytes __dparm(0),
                           hipStream_t stream __dparm(0))` -
 * C compliant kernel launch API
 */
/**
 * Test Description
 * ------------------------
 * - Basic test to verify the basic positive behavior of hipLaunchKernel_spt..
 * Test source
 * ------------------------
 * - catch\unit\executionControl\hipLaunchKernel_spt.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipLaunchKernel_spt_Positive_Basic") {
  SECTION("Kernel with no arguments") {
    HIP_CHECK(hipLaunchKernel_spt(reinterpret_cast<void*>(kernel), dim3{1, 1, 1}, dim3{1, 1, 1},
                                  nullptr, 0, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
  }
  SECTION("Kernel with arguments using kernelParams") {
    LinearAllocGuard<int> result_dev(LinearAllocs::hipMalloc, sizeof(int));
    HIP_CHECK(hipMemset(result_dev.ptr(), 0, sizeof(*result_dev.ptr())));
    int* result_ptr = result_dev.ptr();
    void* kernel_args[1] = {&result_ptr};
    HIP_CHECK(hipLaunchKernel_spt(reinterpret_cast<void*>(kernel_42), dim3{1, 1, 1}, dim3{1, 1, 1},
                                  kernel_args, 0, nullptr));
    int result = 0;
    HIP_CHECK(hipMemcpy(&result, result_dev.ptr(), sizeof(result), hipMemcpyDefault));
    REQUIRE(result == 42);
  }
}
/**
 * Test Description
 * ------------------------
 * - Basic test to verify the basic functionality with all  positive parameters
 * of hipLaunchKernel_spt. Test source
 * ------------------------
 * - catch\unit\executionControl\hipLaunchKernel_spt.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipLaunchKernel_spt_Positive_Parameters") {
  SECTION("blockDim.x == maxBlockDimX") {
    const unsigned int x = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimX, 0);
    HIP_CHECK(hipLaunchKernel_spt(reinterpret_cast<void*>(kernel), dim3{1, 1, 1}, dim3{x, 1, 1},
                                  nullptr, 0, nullptr));
  }
  SECTION("blockDim.y == maxBlockDimY") {
    const unsigned int y = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimY, 0);
    HIP_CHECK(hipLaunchKernel_spt(reinterpret_cast<void*>(kernel), dim3{1, 1, 1}, dim3{y, 1, 1},
                                  nullptr, 0, nullptr));
  }
  SECTION("blockDim.z == maxBlockDimZ") {
    const unsigned int z = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimZ, 0);
    HIP_CHECK(hipLaunchKernel_spt(reinterpret_cast<void*>(kernel), dim3{1, 1, 1}, dim3{z, 1, 1},
                                  nullptr, 0, nullptr));
  }
}
/**
 * Test Description
 * ------------------------
 * - Basic test to verify the negative cases of hipLaunchKernel_spt.
 * Test source
 * ------------------------
 * - catch\unit\executionControl\hipLaunchKernel_spt.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipLaunchKernel_spt_Negative_Parameters") {
  SECTION("f == nullptr") {
    HIP_CHECK_ERROR(hipLaunchKernel_spt(nullptr, dim3{1, 1, 1}, dim3{1, 1, 1}, nullptr, 0, nullptr),
                    hipErrorInvalidDeviceFunction);
  }
  SECTION("gridDim.x == 0") {
    HIP_CHECK_ERROR(hipLaunchKernel_spt(reinterpret_cast<void*>(kernel), dim3{0, 1, 1},
                                        dim3{1, 1, 1}, nullptr, 0, nullptr),
                    hipErrorInvalidValue);
  }
  SECTION("gridDim.y == 0") {
    HIP_CHECK_ERROR(hipLaunchKernel_spt(reinterpret_cast<void*>(kernel), dim3{1, 0, 1},
                                        dim3{1, 1, 1}, nullptr, 0, nullptr),
                    hipErrorInvalidValue);
  }
  SECTION("gridDim.z == 0") {
    HIP_CHECK_ERROR(hipLaunchKernel_spt(reinterpret_cast<void*>(kernel), dim3{1, 1, 0},
                                        dim3{1, 1, 1}, nullptr, 0, nullptr),
                    hipErrorInvalidValue);
  }
  SECTION("blockDim.x == 0") {
    HIP_CHECK_ERROR(hipLaunchKernel_spt(reinterpret_cast<void*>(kernel), dim3{1, 1, 1},
                                        dim3{0, 1, 1}, nullptr, 0, nullptr),
                    hipErrorInvalidValue);
  }
  SECTION("blockDim.y == 0") {
    HIP_CHECK_ERROR(hipLaunchKernel_spt(reinterpret_cast<void*>(kernel), dim3{1, 1, 1},
                                        dim3{1, 0, 1}, nullptr, 0, nullptr),
                    hipErrorInvalidValue);
  }
  SECTION("blockDim.z == 0") {
    HIP_CHECK_ERROR(hipLaunchKernel_spt(reinterpret_cast<void*>(kernel), dim3{1, 1, 1},
                                        dim3{1, 1, 0}, nullptr, 0, nullptr),
                    hipErrorInvalidValue);
  }
  SECTION("Invalid stream") {
    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK_ERROR(hipLaunchKernel_spt(reinterpret_cast<void*>(kernel), dim3{1, 1, 1},
                                        dim3{1, 1, 1}, nullptr, 0, stream),
                    hipErrorInvalidValue);
  }
}
/**
 * End doxygen group ExecutionTest.
 * @}
 */
