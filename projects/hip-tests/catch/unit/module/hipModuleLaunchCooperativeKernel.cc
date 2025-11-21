/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

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

/**
 * @addtogroup hipModuleLaunchCooperativeKernel hipModuleLaunchCooperativeKernel
 * @{
 * @ingroup ModuleTest
 * `hipModuleLaunchCooperativeKernel(hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY,
 * unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
 * unsigned int sharedMemBytes, hipStream_t stream, void ** kernelParams)` -
 * Launches kernel f with launch parameters and shared memory on stream with arguments passed to
 * kernelParams, where thread blocks can cooperate and synchronize as they execute.
 */

#include <hip_test_common.hh>
#include <resource_guards.hh>
#include <utils.hh>

#include "hip_module_launch_kernel_common.hh"

/**
 * Test Description
 * ------------------------
 *  - Tests `hipModuleLaunchCooperativeKernel` for a cooperative kernel with no parameters, and for
 * a normal kernel with parameters.
 * Test source
 * ------------------------
 *  - unit/module/hipModuleLaunchCooperativeKernel.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.5
 */
TEST_CASE("Unit_hipModuleLaunchCooperativeKernel_Positive_Basic") {
  auto mg = ModuleGuard::InitModule("launch_kernel_module.code");
  if (!DeviceAttributesSupport(0, hipDeviceAttributeCooperativeLaunch)) {
    HipTest::HIP_SKIP_TEST("CooperativeLaunch not supported");
    return;
  }

  SECTION("Cooperative kernel with no arguments") {
    hipFunction_t f = GetKernel(mg.module(), "CoopKernel");
    HIP_CHECK(hipModuleLaunchCooperativeKernel(f, 2, 2, 1, 1, 1, 1, 0, nullptr, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
  }

  SECTION("Kernel with arguments using kernelParams") {
    hipFunction_t f = GetKernel(mg.module(), "Kernel42");

    LinearAllocGuard<int> result_dev(LinearAllocs::hipMalloc, sizeof(int));
    HIP_CHECK(hipMemset(result_dev.ptr(), 0, sizeof(*result_dev.ptr())));

    int* result_ptr = result_dev.ptr();
    void* kernel_args[1] = {&result_ptr};
    HIP_CHECK(hipModuleLaunchCooperativeKernel(f, 1, 1, 1, 1, 1, 1, 0, nullptr, kernel_args));

    int result = 0;
    HIP_CHECK(hipMemcpy(&result, result_dev.ptr(), sizeof(result), hipMemcpyDefault));
    REQUIRE(result == 42);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Positive parameters test for `hipModuleLaunchCooperativeKernel`.
 * Test source
 * ------------------------
 *  - unit/module/hipModuleLaunchCooperativeKernel.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.5
 */
TEST_CASE("Unit_hipModuleLaunchCooperativeKernel_Positive_Parameters") {
  auto mg = ModuleGuard::InitModule("launch_kernel_module.code");
  if (!DeviceAttributesSupport(0, hipDeviceAttributeCooperativeLaunch)) {
    HipTest::HIP_SKIP_TEST("CooperativeLaunch not supported");
    return;
  }

  hipFunction_t f = GetKernel(mg.module(), "NOPKernel");

  SECTION("blockDim.x == maxBlockDimX") {
    const unsigned int x = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimX, 0);
    HIP_CHECK(hipModuleLaunchCooperativeKernel(f, 1, 1, 1, x, 1, 1, 0, nullptr, nullptr));
  }

  SECTION("blockDim.y == maxBlockDimY") {
    const unsigned int y = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimY, 0);
    HIP_CHECK(hipModuleLaunchCooperativeKernel(f, 1, 1, 1, y, 1, 1, 0, nullptr, nullptr));
  }

  SECTION("blockDim.z == maxBlockDimZ") {
    const unsigned int z = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimZ, 0);
    HIP_CHECK(hipModuleLaunchCooperativeKernel(f, 1, 1, 1, z, 1, 1, 0, nullptr, nullptr));
  }
}

/**
 * Test Description
 * ------------------------
 *  - Negative parameters test for `hipModuleLaunchCooperativeKernel`.
 * Test source
 * ------------------------
 *  - unit/module/hipModuleLaunchCooperativeKernel.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.5
 */
TEST_CASE("Unit_hipModuleLaunchCooperativeKernel_Negative_Parameters") {
  auto mg = ModuleGuard::InitModule("launch_kernel_module.code");
  if (!DeviceAttributesSupport(0, hipDeviceAttributeCooperativeLaunch)) {
    HipTest::HIP_SKIP_TEST("CooperativeLaunch not supported");
    return;
  }

  hipFunction_t f = GetKernel(mg.module(), "NOPKernel");

  SECTION("f == nullptr") {
    HIP_CHECK_ERROR(
        hipModuleLaunchCooperativeKernel(nullptr, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr),
        hipErrorInvalidResourceHandle);
  }

  SECTION("gridDim.x == 0") {
    HIP_CHECK_ERROR(hipModuleLaunchCooperativeKernel(f, 0, 1, 1, 1, 1, 1, 0, nullptr, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("gridDim.y == 0") {
    HIP_CHECK_ERROR(hipModuleLaunchCooperativeKernel(f, 1, 0, 1, 1, 1, 1, 0, nullptr, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("gridDim.z == 0") {
    HIP_CHECK_ERROR(hipModuleLaunchCooperativeKernel(f, 1, 1, 0, 1, 1, 1, 0, nullptr, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("blockDim.x == 0") {
    HIP_CHECK_ERROR(hipModuleLaunchCooperativeKernel(f, 1, 1, 1, 0, 1, 1, 0, nullptr, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("blockDim.y == 0") {
    HIP_CHECK_ERROR(hipModuleLaunchCooperativeKernel(f, 1, 1, 1, 1, 0, 1, 0, nullptr, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("blockDim.z == 0") {
    HIP_CHECK_ERROR(hipModuleLaunchCooperativeKernel(f, 1, 1, 1, 1, 1, 0, 0, nullptr, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("blockDim.x > maxBlockDimX") {
    const unsigned int x = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimX, 0) + 1u;
    HIP_CHECK_ERROR(hipModuleLaunchCooperativeKernel(f, 1, 1, 1, x, 1, 1, 0, nullptr, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("blockDim.y > maxBlockDimY") {
    const unsigned int y = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimY, 0) + 1u;
    HIP_CHECK_ERROR(hipModuleLaunchCooperativeKernel(f, 1, 1, 1, 1, y, 1, 0, nullptr, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("blockDim.z > maxBlockDimZ") {
    const unsigned int z = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimZ, 0) + 1u;
    HIP_CHECK_ERROR(hipModuleLaunchCooperativeKernel(f, 1, 1, 1, 1, 1, z, 0, nullptr, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("blockDim.x * blockDim.y * blockDim.z > maxThreadsPerBlock") {
    const unsigned int max = GetDeviceAttribute(hipDeviceAttributeMaxThreadsPerBlock, 0);
    const unsigned int dim = std::ceil(std::cbrt(max));
    HIP_CHECK_ERROR(
        hipModuleLaunchCooperativeKernel(f, 1, 1, 1, dim, dim, dim, 0, nullptr, nullptr),
        hipErrorInvalidValue);
  }

#if HT_AMD  // Disabled due to defect EXSWHTEC-351
  SECTION("sharedMemBytes > maxSharedMemoryPerBlock") {
    const unsigned int max = GetDeviceAttribute(hipDeviceAttributeMaxSharedMemoryPerBlock, 0) + 1u;
    HIP_CHECK_ERROR(hipModuleLaunchCooperativeKernel(f, 1, 1, 1, 1, 1, 1, max, nullptr, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("Invalid stream") {
    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK_ERROR(hipModuleLaunchCooperativeKernel(f, 1, 1, 1, 1, 1, 1, 0, stream, nullptr),
                    hipErrorContextIsDestroyed);
  }
#endif
}

/**
 * End doxygen group ModuleTest.
 * @}
 */
