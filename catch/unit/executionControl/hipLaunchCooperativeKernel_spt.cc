/*Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <hip/hip_runtime_api.h>
#include <resource_guards.hh>
#include <hip_test_defgroups.hh>
#include <utils.hh>
#include "execution_control_common.hh"
/**
 * @addtogroup hipLaunchCooperativeKernel_spt hipLaunchCooperativeKernel_spt
 * @{
 * @ingroup ExecutionTest
 * `hipError_t hipLaunchCooperativeKernel_spt(const void* f, dim3 gridDim, dim3 blockDimX,
                                      void** kernelParams, unsigned int sharedMemBytes,
                                      hipStream_t stream);` -
 * launches kernel f with launch parameters and shared memory on stream with arguments passed
 * to kernelparams or extra, where thread blocks can cooperate and synchronize as they execute.
 */
 /**
 * Test Description
 * ------------------------
 * - Basic test to check the functionality of hipLaunchCooperativeKernel_spt.
 * Test source
 * ------------------------
 * - catch\unit\executionControl\hipLaunchCooperativeKernel_spt.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipLaunchCooperativeKernel_spt_Positive_Basic") {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeCooperativeLaunch)) {
    HipTest::HIP_SKIP_TEST("CooperativeLaunch not supported");
    return;
  }
  SECTION("Cooperative kernel with no arguments") {
    HIP_CHECK(hipLaunchCooperativeKernel_spt(reinterpret_cast<void*>(coop_kernel),
                                             dim3{2, 2, 1}, dim3{1, 1, 1},
                                             nullptr, 0, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
  }
  SECTION("Kernel with arguments using kernelParams") {
    LinearAllocGuard<int> result_dev(LinearAllocs::hipMalloc, sizeof(int));
    HIP_CHECK(hipMemset(result_dev.ptr(), 0, sizeof(*result_dev.ptr())));
    int* result_ptr = result_dev.ptr();
    void* kernel_args[1] = {&result_ptr};
    HIP_CHECK(hipLaunchCooperativeKernel_spt(reinterpret_cast<void*>(kernel_42),
                                             dim3{1, 1, 1}, dim3{1, 1, 1},
                                             kernel_args, 0, nullptr));
    int result = 0;
    HIP_CHECK(hipMemcpy(&result, result_dev.ptr(), sizeof(result),
                        hipMemcpyDefault));
    REQUIRE(result == 42);
  }
}
 /**
 * Test Description
 * ------------------------
 * - Basic test to check the functionality of hipLaunchCooperativeKernel_spt
 * with positive parameters.
 * Test source
 * ------------------------
 * - catch\unit\executionControl\hipLaunchCooperativeKernel_spt.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipLaunchCooperativeKernel_spt_Positive_Parameters") {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeCooperativeLaunch)) {
    HipTest::HIP_SKIP_TEST("CooperativeLaunch not supported");
    return;
  }
  SECTION("blockDim.x == maxBlockDimX") {
    const unsigned int x = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimX,
                                              0);
    HIP_CHECK(hipLaunchCooperativeKernel_spt(reinterpret_cast<void*>(kernel),
                                             dim3{1, 1, 1}, dim3{x, 1, 1},
                                             nullptr, 0, nullptr));
  }
  SECTION("blockDim.y == maxBlockDimY") {
    const unsigned int y = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimY,
                                              0);
    HIP_CHECK(hipLaunchCooperativeKernel_spt(reinterpret_cast<void*>(kernel),
                                             dim3{1, 1, 1}, dim3{y, 1, 1},
                                             nullptr, 0, nullptr));
  }
  SECTION("blockDim.z == maxBlockDimZ") {
    const unsigned int z = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimZ,
                                              0);
    HIP_CHECK(hipLaunchCooperativeKernel_spt(reinterpret_cast<void*>(kernel),
                                             dim3{1, 1, 1}, dim3{z, 1, 1},
                                             nullptr, 0, nullptr));
  }
}
 /**
 * Test Description
 * ------------------------
 * - Basic test to check the functionality of hipLaunchCooperativeKernel_spt
 * with negative parameters.
 * Test source
 * ------------------------
 * - catch\unit\executionControl\hipLaunchCooperativeKernel_spt.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipLaunchCooperativeKernel_spt_Negative_Parameters") {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeCooperativeLaunch)) {
    HipTest::HIP_SKIP_TEST("CooperativeLaunch not supported");
    return;
  }
  SECTION("f == nullptr") {
    HIP_CHECK_ERROR(hipLaunchCooperativeKernel_spt(static_cast<void*>(nullptr),
                                                   dim3{1, 1, 1},
                                                   dim3{1, 1, 1},
                                                   nullptr, 0, nullptr),
                    hipErrorInvalidDeviceFunction);
  }
  SECTION("gridDim.x == 0") {
    HIP_CHECK_ERROR(hipLaunchCooperativeKernel_spt(reinterpret_cast<void*>(kernel),
                                               dim3{0, 1, 1},
                                               dim3{1, 1, 1}, nullptr,
                                               0, nullptr),
                    hipErrorInvalidConfiguration);
  }
  SECTION("gridDim.y == 0") {
    HIP_CHECK_ERROR(hipLaunchCooperativeKernel_spt(reinterpret_cast<void*>(kernel),
                                                   dim3{1, 0, 1},
                                                   dim3{1, 1, 1}, nullptr,
                                                   0, nullptr),
                    hipErrorInvalidConfiguration);
  }
  SECTION("gridDim.z == 0") {
    HIP_CHECK_ERROR(hipLaunchCooperativeKernel_spt(reinterpret_cast<void*>(kernel),
                                                   dim3{1, 1, 0},
                                                   dim3{1, 1, 1}, nullptr,
                                                   0, nullptr),
                    hipErrorInvalidConfiguration);
  }
  SECTION("blockDim.x == 0") {
    HIP_CHECK_ERROR(hipLaunchCooperativeKernel_spt(reinterpret_cast<void*>(kernel),
                                                   dim3{1, 1, 1},
                                                   dim3{0, 1, 1}, nullptr,
                                                   0, nullptr),
                    hipErrorInvalidConfiguration);
  }
  SECTION("blockDim.y == 0") {
    HIP_CHECK_ERROR(hipLaunchCooperativeKernel_spt(reinterpret_cast<void*>(kernel),
                                                   dim3{1, 1, 1},
                                                   dim3{1, 0, 1}, nullptr,
                                                   0, nullptr),
                    hipErrorInvalidConfiguration);
  }
  SECTION("blockDim.z == 0") {
    HIP_CHECK_ERROR(hipLaunchCooperativeKernel_spt(reinterpret_cast<void*>(kernel),
                                                   dim3{1, 1, 1},
                                                   dim3{1, 1, 0}, nullptr,
                                                   0, nullptr),
                    hipErrorInvalidConfiguration);
  }
  SECTION("blockDim.x > maxBlockDimX") {
    const unsigned int x = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimX,
                                              0) + 1u;
    HIP_CHECK_ERROR(hipLaunchCooperativeKernel_spt(reinterpret_cast<void*>(kernel),
                                                   dim3{1, 1, 1},
                                                   dim3{x, 1, 1}, nullptr,
                                                   0, nullptr),
                    hipErrorInvalidConfiguration);
  }
  SECTION("blockDim.y > maxBlockDimY") {
    const unsigned int y = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimY,
                                              0) + 1u;
    HIP_CHECK_ERROR(hipLaunchCooperativeKernel_spt(reinterpret_cast<void*>(kernel),
                                                   dim3{1, 1, 1},
                                                   dim3{1, y, 1}, nullptr,
                                                   0, nullptr),
                    hipErrorInvalidConfiguration);
  }
  SECTION("blockDim.z > maxBlockDimZ") {
    const unsigned int z = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimZ,
                                              0) + 1u;
    HIP_CHECK_ERROR(hipLaunchCooperativeKernel_spt(reinterpret_cast<void*>(kernel),
                                                   dim3{1, 1, 1},
                                                   dim3{1, 1, z}, nullptr,
                                                   0, nullptr),
                    hipErrorInvalidConfiguration);
  }
  SECTION("blockDim.x * blockDim.y * blockDim.z > maxThreadsPerBlock") {
    const unsigned int max =
    GetDeviceAttribute(hipDeviceAttributeMaxThreadsPerBlock, 0);
    const unsigned int dim = std::ceil(std::cbrt(max));
    HIP_CHECK_ERROR(hipLaunchCooperativeKernel_spt(reinterpret_cast<void*>(kernel),
                                                   dim3{1, 1, 1},
                                                   dim3{dim, dim, dim},
                                                   nullptr, 0, nullptr),
                    hipErrorInvalidConfiguration);
  }
  SECTION(
      "gridDim.x * gridDim.y * gridDim.z > maxActiveBlocksPerMultiprocessor * "
      "multiProcessorCount") {
    int max_blocks;
    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks,
              reinterpret_cast<void*>(kernel), 1, 0));
    const unsigned int multiproc_count =
        GetDeviceAttribute(hipDeviceAttributeMultiprocessorCount, 0);
    const unsigned int dim = std::ceil(std::cbrt(max_blocks *
                                                 multiproc_count));
    HIP_CHECK_ERROR(hipLaunchCooperativeKernel_spt(reinterpret_cast<void*>(kernel),
                                                   dim3{dim, dim, dim},
                                                   dim3{1, 1, 1}, nullptr, 0,
                                                   nullptr),
                    hipErrorCooperativeLaunchTooLarge);
  }
  SECTION("sharedMemBytes > maxSharedMemoryPerBlock") {
    const unsigned int max =
    GetDeviceAttribute(hipDeviceAttributeMaxSharedMemoryPerBlock, 0) + 1u;
    HIP_CHECK_ERROR(hipLaunchCooperativeKernel_spt(reinterpret_cast<void*>(kernel),
                                                   dim3{1, 1, 1},
                                                   dim3{1, 1, 1}, nullptr, max,
                                                   nullptr),
                    hipErrorCooperativeLaunchTooLarge);
  }
}
/**
* End doxygen group ExecutionTest.
* @}
*/

