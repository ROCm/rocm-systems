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
#include <hip/hip_cooperative_groups.h>
#include <hip_test_checkers.hh>
#include <hip_test_common.hh>
#include <hip_test_kernels.hh>
#include <utils.hh>

static constexpr int N = 10;
constexpr auto CODE_OBJ_SINGLEARCH = "coopKernel.code";
static constexpr auto kernel_name = "cooperativeKernelEx";

/**
 * @addtogroup hipDrvLaunchKernelEx hipDrvLaunchKernelEx
 * @{
 * @ingroup ModuleTest
 * `hipError_t hipDrvLaunchKernelEx(const HIP_LAUNCH_CONFIG* config,
 hipFunction_t f, void** params, void** extra)`
 * Launches a HIP kernel using the driver API with the specified configuration.
 */

/**
 * Test Description
 * ------------------------
 * This test verfies the different negative scenarios of hipDrvLaunchKernelEx
 * Api Test source
 * ------------------------
 * - catch/unit/module/hipDrvLaunchKernelEx.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.5
 */
TEST_CASE("Unit_hipDrvLaunchKernelEx_NegTsts") {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeCooperativeLaunch)) {
    HipTest::HIP_SKIP_TEST("CooperativeLaunch not supported");
    return;
  }
  int totalThreads = 64;
  int blockSize = 16;
  int numBlocks = (totalThreads + blockSize - 1) / blockSize;

  int* d_output = nullptr;
  HIP_CHECK(hipMalloc(&d_output, totalThreads * sizeof(int)));
  HIP_CHECK(hipMemset(d_output, 0, totalThreads * sizeof(int)));

  // Set up the HIP_LAUNCH_CONFIG structure
  HIP_LAUNCH_CONFIG config = {};
  config.gridDimX = numBlocks;
  config.gridDimY = 1;
  config.gridDimZ = 1;
  config.blockDimX = blockSize;
  config.blockDimY = 1;
  config.blockDimZ = 1;
  config.sharedMemBytes = 0;
  config.hStream = 0;  // default stream

  // Set up a cooperative launch attribute
  hipDrvLaunchAttribute attr;
  attr.id = hipDrvLaunchAttributeCooperative;
  // Zero out the padding bytes
  memset(attr.pad, 0, sizeof(attr.pad));
  attr.value.cooperative = 1;

  config.attrs = &attr;
  config.numAttrs = 1;

  // Kernel parameters: address of d_output and totalThreads.
  void* kernelParams[] = {&d_output, &totalThreads};

  hipModule_t module;
  HIP_CHECK(hipModuleLoad(&module, CODE_OBJ_SINGLEARCH));

  hipFunction_t function;
  HIP_CHECK(hipModuleGetFunction(&function, module, kernel_name));

  SECTION("Kernel config as nullptr") {
    HIP_CHECK_ERROR(hipDrvLaunchKernelEx(nullptr, function, kernelParams, NULL),
                    hipErrorInvalidValue);
  }
  SECTION("Kernel function as nullptr") {
    HIP_CHECK_ERROR(hipDrvLaunchKernelEx(&config, nullptr, kernelParams, NULL),
                    hipErrorInvalidResourceHandle);
  }
  SECTION("Kernel parameter as nullptr") {
    HIP_CHECK_ERROR(hipDrvLaunchKernelEx(&config, function, nullptr, NULL), hipErrorInvalidValue);
  }
  HIP_LAUNCH_CONFIG invalidConfig = {};
  invalidConfig.gridDimX = 0;
  invalidConfig.gridDimY = 1;
  invalidConfig.gridDimZ = 1;
  invalidConfig.blockDimX = 0;
  invalidConfig.blockDimY = 1;
  invalidConfig.blockDimZ = 1;
  invalidConfig.sharedMemBytes = 0;
  invalidConfig.hStream = 0;  // default stream

  // Set up a cooperative launch attribute
  hipDrvLaunchAttribute invalidAttr;
  invalidAttr.id = hipDrvLaunchAttributeCooperative;
  // Zero out the padding bytes
  memset(invalidAttr.pad, 0, sizeof(invalidAttr.pad));
  invalidAttr.value.cooperative = 1;

  invalidConfig.attrs = &invalidAttr;
  invalidConfig.numAttrs = 1;

  SECTION("Invalid Kernel config") {
    HIP_CHECK_ERROR(hipDrvLaunchKernelEx(&invalidConfig, function, kernelParams, NULL),
                    hipErrorInvalidConfiguration);
  }

  HIP_CHECK(hipModuleUnload(module));
  HIP_CHECK(hipFree(d_output));
}

bool runTestDrvLaunch(const char* testName, std::string kernelFunc, int totalThreads, int blockSize,
                      int flagValue) {
  int numBlocks = (totalThreads + blockSize - 1) / blockSize;

  int* d_output = nullptr;
  HIP_CHECK(hipMalloc(&d_output, totalThreads * sizeof(int)));
  HIP_CHECK(hipMemset(d_output, 0, totalThreads * sizeof(int)));

  // Set up the HIP_LAUNCH_CONFIG structure
  HIP_LAUNCH_CONFIG config = {};
  config.gridDimX = numBlocks;
  config.gridDimY = 1;
  config.gridDimZ = 1;
  config.blockDimX = blockSize;
  config.blockDimY = 1;
  config.blockDimZ = 1;
  config.sharedMemBytes = 0;
  config.hStream = 0;  // default stream

  // Set up a cooperative launch attribute
  hipDrvLaunchAttribute attr;
  attr.id = hipDrvLaunchAttributeCooperative;
  // Zero out the padding bytes
  memset(attr.pad, 0, sizeof(attr.pad));
  attr.value.cooperative = 1;

  config.attrs = &attr;
  config.numAttrs = 1;

  // Kernel parameters: address of d_output and totalThreads.
  void* kernelParams[] = {&d_output, &totalThreads};

  hipModule_t module;
  HIP_CHECK(hipModuleLoad(&module, CODE_OBJ_SINGLEARCH));

  hipFunction_t function;
  HIP_CHECK(hipModuleGetFunction(&function, module, kernelFunc.c_str()));

  // Launch the kernel using the driver API function.
  hipError_t err = hipDrvLaunchKernelEx(&config, function, kernelParams, NULL);
  if (err != hipSuccess) {
    printf("%s test failed to launch kernel: error code %d\n", testName, err);
    HIP_CHECK(hipFree(d_output));
    return false;
  }

  HIP_CHECK(hipDeviceSynchronize());

  int* h_output = (int*)malloc(totalThreads * sizeof(int));
  HIP_CHECK(hipMemcpy(h_output, d_output, totalThreads * sizeof(int), hipMemcpyDeviceToHost));

  // Verify results.
  bool success = true;
  if (h_output[0] != flagValue) {
    printf("%s test failed: Expected flag %d at index 0, got %d\n", testName, flagValue,
           h_output[0]);
    success = false;
  }
  for (int i = 1; i < totalThreads; i++) {
    int expectedValue = (flagValue == 1111) ? i : (i * 3);
    if (h_output[i] != expectedValue) {
      printf("%s test failed at index %d: Expected %d, got %d\n", testName, i, expectedValue,
             h_output[i]);
      success = false;
      break;
    }
  }

  HIP_CHECK(hipFree(d_output));
  free(h_output);
  return success;
}
/**
 * Test Description
 * ------------------------
 * This test verfies basic positive test of hipDrvLaunchKernelEx
 * Api Test source
 * ------------------------
 * - catch/unit/module/hipDrvLaunchKernelEx.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.5
 */
TEST_CASE("Unit_hipDrvLaunchKernelEx_Functional") {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeCooperativeLaunch)) {
    HipTest::HIP_SKIP_TEST("CooperativeLaunch not supported");
    return;
  }
  REQUIRE(runTestDrvLaunch("hipDrvLaunchKernelEx", kernel_name, 64, 16, 2222) == true);
}

/**
 * Test Description
 * ------------------------
 *  - This testcase checks following scenarios(Uses kernels from the module)
 *  - 1) Launch Normal kernel which has no arguments using hipDrvLaunchKernelEx
 *  - 2) Launch Normal kernel which has arguments by using hipDrvLaunchKernelEx
 *  - 3) Launch Cooperative kernel which has no arguments by using
 *  -    hipDrvLaunchKernelEx
 * Test source
 * ------------------------
 *  - unit/module/hipDrvLaunchKernelEx.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.5
 */
TEST_CASE("Unit_hipDrvLaunchKernelEx_With_Different_Kernels") {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeCooperativeLaunch)) {
    HipTest::HIP_SKIP_TEST("CooperativeLaunch not supported");
    return;
  }

  hipModule_t module;
  HIP_CHECK(hipModuleLoad(&module, "coopKernel.code"));

  HIP_LAUNCH_CONFIG config = {};
  config.gridDimX = 1;
  config.gridDimY = 1;
  config.gridDimZ = 1;
  config.blockDimX = 1;
  config.blockDimY = 1;
  config.blockDimZ = 1;
  config.sharedMemBytes = 0;
  config.hStream = 0;
  hipDrvLaunchAttribute attr;
  attr.id = hipDrvLaunchAttributeCooperative;
  // Zero out the padding bytes
  memset(attr.pad, 0, sizeof(attr.pad));
  attr.value.cooperative = 1;
  config.attrs = &attr;
  config.numAttrs = 1;

  SECTION("Normal kernel with no arguments") {
    hipFunction_t simpleKernel;
    HIP_CHECK(hipModuleGetFunction(&simpleKernel, module, "emptyKernel"));

    HIP_CHECK(hipDrvLaunchKernelEx(&config, simpleKernel, nullptr, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
  }

  SECTION("Kernel with arguments using kernelParams") {
    int* devMem = nullptr;
    HIP_CHECK(hipMalloc(&devMem, sizeof(int)));

    void* kernel_args[1] = {&devMem};

    hipFunction_t argKernel;
    HIP_CHECK(hipModuleGetFunction(&argKernel, module, "argKernel"));

    HIP_CHECK(hipDrvLaunchKernelEx(&config, argKernel, kernel_args, nullptr));
    HIP_CHECK(hipDeviceSynchronize());

    int result = 0;
    HIP_CHECK(hipMemcpy(&result, devMem, sizeof(result), hipMemcpyDefault));
    REQUIRE(result == 100);

    HIP_CHECK(hipFree(devMem));
  }

  SECTION("Cooperative kernel with no arguments") {
    hipFunction_t coopKernel;
    HIP_CHECK(hipModuleGetFunction(&coopKernel, module, "coopEmptykernel"));

    HIP_CHECK(hipDrvLaunchKernelEx(&config, coopKernel, nullptr, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
  }

  HIP_CHECK(hipModuleUnload(module));
}

/**
 * Test Description
 * ------------------------
 *  - This testcase launches a kernel(Uses kernels from the module) which
 *  - uses the cooperative groups andwith N number of blocks and one thread
 *  - in each block by using the hipDrvLaunchKernelEx and validates
 *  - the kernel output
 * Test source
 * ------------------------
 *  - unit/module/hipDrvLaunchKernelEx.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.5
 */
TEST_CASE("Unit_hipDrvLaunchKernelEx_With_CooperativeKernelWithArgs") {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeCooperativeLaunch)) {
    HipTest::HIP_SKIP_TEST("CooperativeLaunch not supported");
    return;
  }

  hipModule_t module;
  HIP_CHECK(hipModuleLoad(&module, "coopKernel.code"));

  HIP_LAUNCH_CONFIG config = {};
  config.gridDimX = N;
  config.gridDimY = 1;
  config.gridDimZ = 1;
  config.blockDimX = 1;
  config.blockDimY = 1;
  config.blockDimZ = 1;
  config.sharedMemBytes = 0;
  config.hStream = 0;
  hipDrvLaunchAttribute attr;
  attr.id = hipDrvLaunchAttributeCooperative;
  // Zero out the padding bytes
  memset(attr.pad, 0, sizeof(attr.pad));
  attr.value.cooperative = 1;
  config.attrs = &attr;
  config.numAttrs = 1;

  int hostMem[N];
  for (int i = 0; i < N; i++) {
    hostMem[i] = 0;
  }

  int* devMem1 = nullptr;
  HIP_CHECK(hipMalloc(&devMem1, N * sizeof(int)));
  HIP_CHECK(hipMemcpy(devMem1, hostMem, N * sizeof(int), hipMemcpyDefault));
  int* devMem2 = nullptr;
  HIP_CHECK(hipMalloc(&devMem2, N * sizeof(int)));
  HIP_CHECK(hipMemcpy(devMem2, hostMem, N * sizeof(int), hipMemcpyDefault));

  int size = N;
  void* kernel_args[3] = {&devMem1, &devMem2, &size};

  hipFunction_t argKernel;
  HIP_CHECK(hipModuleGetFunction(&argKernel, module, "coopFillArrayKernel"));

  HIP_CHECK(hipDrvLaunchKernelEx(&config, argKernel, kernel_args, nullptr));
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(hostMem, devMem2, N * sizeof(int), hipMemcpyDefault));
  for (int i = 0; i < N; i++) {
    REQUIRE(hostMem[i] == 550);
  }

  HIP_CHECK(hipFree(devMem1));
  HIP_CHECK(hipFree(devMem2));
  HIP_CHECK(hipModuleUnload(module));
}

/**
 * Test Description
 * ------------------------
 *  - This testcase checks following scenarios(Uses kernels from the module)
 *  - 1) Launch a kernel with one block and maximum allowed threads in a block
 *  -    in the x direction by using hipLaunchKernelEx and hipDrvLaunchKernelEx
 *  - 2) Launch a kernel with one block and maximum allowed threads in a block
 *  -    in the y direction by using hipLaunchKernelEx and hipDrvLaunchKernelEx
 *  - 3) Launch a kernel with one block and maximum allowed threads in a block
 *  -    in the z direction by using hipLaunchKernelEx and hipDrvLaunchKernelEx
 * Test source
 * ------------------------
 *  - unit/module/hipDrvLaunchKernelEx.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.5
 */
TEST_CASE("Unit_hipDrvLaunchKernelEx_With_MaxBlockDims") {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeCooperativeLaunch)) {
    HipTest::HIP_SKIP_TEST("CooperativeLaunch not supported");
    return;
  }

  hipModule_t module;
  HIP_CHECK(hipModuleLoad(&module, "coopKernel.code"));

  hipFunction_t kernel;
  HIP_CHECK(hipModuleGetFunction(&kernel, module, "emptyKernel"));

  HIP_LAUNCH_CONFIG config = {};
  config.gridDimX = 1;
  config.gridDimY = 1;
  config.gridDimZ = 1;
  config.blockDimX = 1;
  config.blockDimY = 1;
  config.blockDimZ = 1;
  config.sharedMemBytes = 0;
  config.hStream = 0;
  hipDrvLaunchAttribute attr;
  attr.id = hipDrvLaunchAttributeCooperative;
  // Zero out the padding bytes
  memset(attr.pad, 0, sizeof(attr.pad));
  attr.value.cooperative = 1;
  config.attrs = &attr;
  config.numAttrs = 1;

  SECTION("blockDim.x == maxBlockDimX") {
    const unsigned int x = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimX, 0);
    config.blockDimX = x;

    HIP_CHECK(hipDrvLaunchKernelEx(&config, kernel, nullptr, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
  }

  SECTION("blockDim.y == maxBlockDimY") {
    const unsigned int y = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimY, 0);
    config.blockDimY = y;

    HIP_CHECK(hipDrvLaunchKernelEx(&config, kernel, nullptr, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
  }

  SECTION("blockDim.z == maxBlockDimZ") {
    const unsigned int z = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimZ, 0);
    config.blockDimY = z;

    HIP_CHECK(hipDrvLaunchKernelEx(&config, kernel, nullptr, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
  }

  HIP_CHECK(hipModuleUnload(module));
}
/**
 * End doxygen group ModuleTest.
 * @}
 */
