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
static __device__ int devArr[N];
//------------------------------------------------------------------------------
// Kernel using hipLaunchKernelEx
//------------------------------------------------------------------------------
__global__ void cooperativeKernelEx(int* output, int totalThreads) {
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < totalThreads) {
    output[tid] = tid * 3;
  }
  grid.sync();
  if (tid == 0) {
    output[0] = 2222;
  }
}
//------------------------------------------------------------------------------
// Kernel using hipLaunchKernelExC
//------------------------------------------------------------------------------
__global__ void cooperativeKernelExC(int* output, int totalThreads) {
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < totalThreads) {
    output[tid] = tid;
  }
  grid.sync();
  if (tid == 0) {
    output[0] = 1111;
  }
}
/*
 * Kernel which doesn't use cooperative groups and without any arguments
 */
static __global__ void emptyKernel() {}

/*
 * Kernel which doesn't use cooperative groups and takes an argument
 * and updates the value with 100
 */
static __global__ void argKernel(int* val) { *val = 100; }

/*
 * Kernel which uses cooperative groups and without any arguments
 */
static __global__ void coopEmptykernel() {
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  grid.sync();
}

/*
 * Kernel which uses cooperative groups and takes an array as argument
 * and performs below operations
 * 1) All the threads in the block fills the element in the devArr array
 *    based on the blockIdx.x
 * 2) Wait for all the blocks completes it's operations
 * 3) Fill each element in the output array with sum of elements in devArr
 */
static __global__ void coopFillArrayKernel(int* output) {
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();

  if (blockIdx.x == 0)
    devArr[0] = 10;
  else if (blockIdx.x == 1)
    devArr[1] = 20;
  else if (blockIdx.x == 2)
    devArr[2] = 30;
  else if (blockIdx.x == 3)
    devArr[3] = 40;
  else if (blockIdx.x == 4)
    devArr[4] = 50;
  else if (blockIdx.x == 5)
    devArr[5] = 60;
  else if (blockIdx.x == 6)
    devArr[6] = 70;
  else if (blockIdx.x == 7)
    devArr[7] = 80;
  else if (blockIdx.x == 8)
    devArr[8] = 90;
  else if (blockIdx.x == 9)
    devArr[9] = 100;

  grid.sync();

  for (int i = 0; i < N; i++) {
    output[blockIdx.x] = output[blockIdx.x] + devArr[i];
  }
}

__global__ void normalKernel(int* output, int totalThreads) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < totalThreads) {
    output[tid] = tid * 3;
  }
}
/**
 * @addtogroup hipLaunchKernelExC hipLaunchKernelExC
 * @{
 * @ingroup KernelTest
 * `hipError_t hipLaunchKernelExC(const hipLaunchConfig_t* config, const void*
 * fPtr, void** args)` - Launches a HIP kernel using a generic function pointer
 * and the specified configuration.
 */

/**
 * Test Description
 * ------------------------
 * This test verfies the different negative scenarios of hipLaunchKernelExC

 * Test source
 * ------------------------
 *    - catch/unit/kernel/hipLaunchKernelEx.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.5
 */
TEST_CASE("Unit_hipLaunchKernelExC_NegetiveTsts") {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeCooperativeLaunch)) {
    HipTest::HIP_SKIP_TEST("CooperativeLaunch not supported");
    return;
  }
  int blockSize = 16;
  int totalThreads = 64;
  const int numBlocks = (totalThreads + blockSize - 1) / blockSize;
  hipLaunchConfig_t config = {};
  config.gridDim = dim3{(uint32_t)numBlocks, 1, 1};
  config.blockDim = dim3{(uint32_t)blockSize, 1, 1};
  config.dynamicSmemBytes = 0;
  config.stream = 0;

  hipLaunchAttribute attr;
  attr.id = hipLaunchAttributeCooperative;
  attr.val.cooperative = 1;
  config.attrs = &attr;
  config.numAttrs = 1;

  int* d_output = nullptr;
  HIP_CHECK(hipMalloc(&d_output, totalThreads * sizeof(int)));
  HIP_CHECK(hipMemset(d_output, 0, totalThreads * sizeof(int)));
  void* kernelArgs[] = {&d_output, (void*)&totalThreads};

  SECTION("Kernel function as nullptr") {
    HIP_CHECK_ERROR(hipLaunchKernelExC(&config, nullptr, kernelArgs),
                    hipErrorInvalidDeviceFunction);
  }

  SECTION("Kernel args as nullptr") {
    HIP_CHECK_ERROR(hipLaunchKernelExC(&config, (void*)cooperativeKernelExC, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("Non Cooparative Kernel") {
    HIP_CHECK_ERROR(hipLaunchKernelExC(&config, (void*)normalKernel, kernelArgs), hipSuccess);
  }

  hipLaunchConfig_t invalidConfig = {};
  invalidConfig.gridDim = dim3{0, 1, 1};
  invalidConfig.blockDim = dim3{0, 1, 1};
  invalidConfig.dynamicSmemBytes = 0;
  invalidConfig.stream = 0;

  hipLaunchAttribute invalidAttr;
  invalidAttr.id = hipLaunchAttributeCooperative;
  invalidAttr.val.cooperative = 1;
  invalidConfig.attrs = &invalidAttr;
  invalidConfig.numAttrs = 1;

  SECTION("Invalid Kernel Config") {
    HIP_CHECK_ERROR(hipLaunchKernelExC(&invalidConfig, (void*)cooperativeKernelExC, kernelArgs),
                    hipErrorInvalidConfiguration);
  }
  HIP_CHECK(hipFree(d_output));
}
/**
 * Test Description
 * ------------------------
 * This test verfies the different negative scenarios of hipLaunchKernelEx

 * Test source
 * ------------------------
 *    - catch/unit/kernel/hipLaunchKernelEx.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.5
 */

TEST_CASE("Unit_hipLaunchKernelEx_NegetiveTsts") {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeCooperativeLaunch)) {
    HipTest::HIP_SKIP_TEST("CooperativeLaunch not supported");
    return;
  }
  int blockSize = 16;
  int totalThreads = 64;
  const int numBlocks = (totalThreads + blockSize - 1) / blockSize;
  hipLaunchConfig_t config = {};
  config.gridDim = dim3{(uint32_t)numBlocks, 1, 1};
  config.blockDim = dim3{(uint32_t)blockSize, 1, 1};
  config.dynamicSmemBytes = 0;
  config.stream = 0;

  hipLaunchAttribute attr;
  attr.id = hipLaunchAttributeCooperative;
  attr.val.cooperative = 1;
  config.attrs = &attr;
  config.numAttrs = 1;

  int* d_output = nullptr;
  HIP_CHECK(hipMalloc(&d_output, totalThreads * sizeof(int)));
  HIP_CHECK(hipMemset(d_output, 0, totalThreads * sizeof(int)));

  SECTION("Empty Kernel Args") {
    HIP_CHECK_ERROR(hipLaunchKernelEx(&config, emptyKernel), hipSuccess);
  }

  SECTION("Non Cooparative Kernel") {
    HIP_CHECK_ERROR(
        hipLaunchKernelEx(&config, (void (*)(int*, int))normalKernel, d_output, totalThreads),
        hipSuccess);
  }

  hipLaunchConfig_t invalidConfig = {};
  invalidConfig.gridDim = dim3{0, 1, 1};
  invalidConfig.blockDim = dim3{0, 1, 1};
  invalidConfig.dynamicSmemBytes = 0;
  invalidConfig.stream = 0;

  hipLaunchAttribute attr_1;
  attr_1.id = hipLaunchAttributeCooperative;
  attr_1.val.cooperative = 1;
  invalidConfig.attrs = &attr_1;
  invalidConfig.numAttrs = 1;

  SECTION("Invalid Kernel Config") {
    HIP_CHECK_ERROR(hipLaunchKernelEx(&invalidConfig, (void (*)(int*, int))cooperativeKernelEx,
                                      d_output, totalThreads),
                    hipErrorInvalidConfiguration);
  }
  HIP_CHECK(hipFree(d_output));
}

bool runTest(const char* testName, const void* kernelFunc, int totalThreads, int blockSize,
             int flagValue, bool useTemplate) {
  const int numBlocks = (totalThreads + blockSize - 1) / blockSize;

  int* d_output = nullptr;
  HIP_CHECK(hipMalloc(&d_output, totalThreads * sizeof(int)));
  HIP_CHECK(hipMemset(d_output, 0, totalThreads * sizeof(int)));

  hipLaunchConfig_t config = {};
  config.gridDim = dim3{(uint32_t)numBlocks, 1, 1};
  config.blockDim = dim3{(uint32_t)blockSize, 1, 1};
  config.dynamicSmemBytes = 0;
  config.stream = 0;

  hipLaunchAttribute attr;
  attr.id = hipLaunchAttributeCooperative;
  attr.val.cooperative = 1;
  config.attrs = &attr;
  config.numAttrs = 1;

  // For a kernel parameter declared as "int* output", pass the address of the
  // device pointer.
  void* kernelArgs[] = {&d_output, (void*)&totalThreads};

  if (useTemplate) {
    HIP_CHECK(hipLaunchKernelEx(&config, (void (*)(int*, int))kernelFunc, d_output, totalThreads));
  } else {
    HIP_CHECK(hipLaunchKernelExC(&config, kernelFunc, kernelArgs));
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
 * This test verfies basic positive test scenarios of hipLaunchKernelEx,
 hipLaunchKernelExC

 * Test source
 * ------------------------
 *    - catch/unit/kernel/hipLaunchKernelEx.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.5
 */
TEST_CASE("Unit_hipLaunchKernelEx_Functional") {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeCooperativeLaunch)) {
    HipTest::HIP_SKIP_TEST("CooperativeLaunch not supported");
    return;
  }
  std::string api_type = GENERATE("hipLaunchKernelEx", "hipLaunchKernelExC");
  if (api_type == "hipLaunchKernelEx") {
    REQUIRE(runTest(api_type.c_str(), (void*)cooperativeKernelEx, 64, 16, 2222, true) == true);
  }
  if (api_type == "hipLaunchKernelExC") {
    REQUIRE(runTest(api_type.c_str(), (void*)cooperativeKernelExC, 64, 16, 1111, false) == true);
  }
}
/**
 * Test Description
 * ------------------------
 *  - This testcase checks following scenarios
 *  - 1) Launch Normal kernel which has no arguments using hipLaunchKernelEx
 *  - 2) Launch Normal kernel which has no arguments using hipLaunchKernelExC
 *  - 3) Launch Normal kernel which has arguments by using hipLaunchKernelEx
 *  - 4) Launch Normal kernel which has arguments by using hipLaunchKernelExC
 *  - 5) Launch Cooperative kernel which has no arguments by using
 *  -    hipLaunchKernelEx
 *  - 6) Launch Cooperative kernel which has no arguments by using
 *  -    hipLaunchKernelExC
 * Test source
 * ------------------------
 *  - unit/kernel/hipLaunchKernelEx.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.5
 */
TEST_CASE("Unit_hipLaunchKernelEx_With_Different_Kernels") {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeCooperativeLaunch)) {
    HipTest::HIP_SKIP_TEST("CooperativeLaunch not supported");
    return;
  }

  hipLaunchConfig_t config = {};
  config.gridDim = dim3{2, 2, 1};
  config.blockDim = dim3{2, 2, 1};
  config.dynamicSmemBytes = 0;
  config.stream = 0;

  hipLaunchAttribute attr;
  attr.id = hipLaunchAttributeCooperative;
  memset(attr.pad, 0, sizeof(attr.pad));
  attr.val.cooperative = 1;
  config.attrs = &attr;
  config.numAttrs = 1;

  SECTION("Normal kernel with no arguments") {
    SECTION("hipLaunchKernelEx") { HIP_CHECK(hipLaunchKernelEx(&config, emptyKernel)); }

    SECTION("hipLaunchKernelExC") {
      HIP_CHECK(hipLaunchKernelExC(&config, reinterpret_cast<void*>(emptyKernel), nullptr));
    }
    HIP_CHECK(hipDeviceSynchronize());
  }

  SECTION("Kernel with arguments using kernelParams") {
    config.gridDim = dim3{1, 1, 1};
    config.blockDim = dim3{1, 1, 1};

    int* devMem = nullptr;
    HIP_CHECK(hipMalloc(&devMem, sizeof(int)));

    SECTION("hipLaunchKernelEx") { HIP_CHECK(hipLaunchKernelEx(&config, argKernel, devMem)); }

    SECTION("hipLaunchKernelExC") {
      void* kernel_args[1] = {&devMem};
      HIP_CHECK(hipLaunchKernelExC(&config, reinterpret_cast<void*>(argKernel), kernel_args));
    }
    HIP_CHECK(hipDeviceSynchronize());

    int result = 0;
    HIP_CHECK(hipMemcpy(&result, devMem, sizeof(result), hipMemcpyDefault));
    HIP_CHECK(hipFree(devMem));
    REQUIRE(result == 100);
  }

  SECTION("Cooperative kernel with no arguments") {
    SECTION("hipLaunchKernelEx") { HIP_CHECK(hipLaunchKernelEx(&config, coopEmptykernel)); }

    SECTION("hipLaunchKernelExC") {
      HIP_CHECK(hipLaunchKernelExC(&config, reinterpret_cast<void*>(coopEmptykernel), nullptr));
    }
    HIP_CHECK(hipDeviceSynchronize());
  }
}

/**
 * Test Description
 * ------------------------
 *  - This testcase launches a kernel which uses the cooperative groups and
 *  - with N number of blocks and one thread in each block by using the
 *  - hipLaunchKernelEx and hipLaunchKernelExC and validates the kernel output
 * Test source
 * ------------------------
 *  - unit/kernel/hipLaunchKernelEx.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.5
 */
TEST_CASE("Unit_hipLaunchKernelEx_With_CooperativeKernelWithArgs") {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeCooperativeLaunch)) {
    HipTest::HIP_SKIP_TEST("CooperativeLaunch not supported");
    return;
  }

  hipLaunchConfig_t config = {};
  config.gridDim = dim3{N, 1, 1};
  config.blockDim = dim3{1, 1, 1};
  config.dynamicSmemBytes = 0;
  config.stream = 0;

  hipLaunchAttribute attr;
  attr.id = hipLaunchAttributeCooperative;
  memset(attr.pad, 0, sizeof(attr.pad));
  attr.val.cooperative = 1;
  config.attrs = &attr;
  config.numAttrs = 1;

  int hostMem[N];
  for (int i = 0; i < N; i++) {
    hostMem[i] = 0;
  }

  int* devMem = nullptr;
  HIP_CHECK(hipMalloc(&devMem, N * sizeof(int)));
  HIP_CHECK(hipMemcpy(devMem, hostMem, N * sizeof(int), hipMemcpyDefault));

  SECTION("hipLaunchKernelEx") {
    HIP_CHECK(hipLaunchKernelEx(&config, coopFillArrayKernel, devMem));
  }

  SECTION("hipLaunchKernelExC") {
    void* kernel_args[1] = {&devMem};
    HIP_CHECK(
        hipLaunchKernelExC(&config, reinterpret_cast<void*>(coopFillArrayKernel), kernel_args));
  }
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(hostMem, devMem, N * sizeof(int), hipMemcpyDefault));
  for (int i = 0; i < N; i++) {
    REQUIRE(hostMem[i] == 550);
  }

  HIP_CHECK(hipFree(devMem));
}

/**
 * Test Description
 * ------------------------
 *  - This testcase checks following scenarios
 *  - 1) Launch a kernel with one block and maximum allowed threads in a block
 *  -    in the x direction by using hipLaunchKernelEx and hipLaunchKernelExC
 *  - 2) Launch a kernel with one block and maximum allowed threads in a block
 *  -    in the y direction by using hipLaunchKernelEx and hipLaunchKernelExC
 *  - 3) Launch a kernel with one block and maximum allowed threads in a block
 *  -    in the z direction by using hipLaunchKernelEx and hipLaunchKernelExC
 * Test source
 * ------------------------
 *  - unit/kernel/hipLaunchKernelEx.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.5
 */
TEST_CASE("Unit_hipLaunchKernelEx_With_MaxBlockDims") {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeCooperativeLaunch)) {
    HipTest::HIP_SKIP_TEST("CooperativeLaunch not supported");
    return;
  }

  hipLaunchConfig_t config = {};
  config.gridDim = dim3{1, 1, 1};
  config.blockDim = dim3{1, 1, 1};
  config.dynamicSmemBytes = 0;
  config.stream = 0;

  hipLaunchAttribute attr;
  attr.id = hipLaunchAttributeCooperative;
  memset(attr.pad, 0, sizeof(attr.pad));
  attr.val.cooperative = 1;
  config.attrs = &attr;
  config.numAttrs = 1;

  SECTION("blockDim.x == maxBlockDimX") {
    const unsigned int x = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimX, 0);
    config.blockDim = dim3{x, 1, 1};

    SECTION("hipLaunchKernelEx") { HIP_CHECK(hipLaunchKernelEx(&config, emptyKernel)); }

    SECTION("hipLaunchKernelExC") {
      HIP_CHECK(hipLaunchKernelExC(&config, reinterpret_cast<void*>(emptyKernel), nullptr));
    }
  }

  SECTION("blockDim.y == maxBlockDimY") {
    const unsigned int y = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimY, 0);
    config.blockDim = dim3{1, y, 1};

    SECTION("hipLaunchKernelEx") { HIP_CHECK(hipLaunchKernelEx(&config, emptyKernel)); }

    SECTION("hipLaunchKernelExC") {
      HIP_CHECK(hipLaunchKernelExC(&config, reinterpret_cast<void*>(emptyKernel), nullptr));
    }
  }

  SECTION("blockDim.z == maxBlockDimZ") {
    const unsigned int z = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimZ, 0);
    config.blockDim = dim3{1, 1, z};

    SECTION("hipLaunchKernelEx") { HIP_CHECK(hipLaunchKernelEx(&config, emptyKernel)); }

    SECTION("hipLaunchKernelExC") {
      HIP_CHECK(hipLaunchKernelExC(&config, reinterpret_cast<void*>(emptyKernel), nullptr));
    }
  }
  HIP_CHECK(hipDeviceSynchronize());
}
/**
 * End doxygen group KernelTest.
 * @}
 */
