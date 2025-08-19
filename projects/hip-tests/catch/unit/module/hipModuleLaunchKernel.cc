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
#include "hip_module_launch_kernel_common.hh"

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <hip_test_defgroups.hh>
#include <math.h>

static hipError_t hipModuleLaunchKernelWrapper(hipFunction_t f, uint32_t gridX, uint32_t gridY,
                                               uint32_t gridZ, uint32_t blockX, uint32_t blockY,
                                               uint32_t blockZ, size_t sharedMemBytes,
                                               hipStream_t hStream, void** kernelParams,
                                               void** extra, hipEvent_t, hipEvent_t, uint32_t) {
  return hipModuleLaunchKernel(f, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMemBytes,
                               hStream, kernelParams, extra);
}

TEST_CASE("Unit_hipModuleLaunchKernel_Positive_Basic") {
  HIP_CHECK(hipFree(nullptr));
  ModuleLaunchKernelPositiveBasic<hipModuleLaunchKernelWrapper>();
}

TEST_CASE("Unit_hipModuleLaunchKernel_Positive_Parameters") {
  HIP_CHECK(hipFree(nullptr));
  ModuleLaunchKernelPositiveParameters<hipModuleLaunchKernelWrapper>();
}

TEST_CASE("Unit_hipModuleLaunchKernel_Negative_Parameters") {
  HIP_CHECK(hipFree(nullptr));
  ModuleLaunchKernelNegativeParameters<hipModuleLaunchKernelWrapper>();
}
constexpr auto fileName = "matmul.code";
constexpr auto dummyKernel = "dummyKernel";

/**
* @addtogroup hipModuleLaunchKernel
* @{
* @ingroup ModuleTest
* `hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY,
                                    unsigned int gridDimZ, unsigned int blockDimX,
                                    unsigned int blockDimY, unsigned int blockDimZ,
                                    unsigned int sharedMemBytes, hipStream_t stream,
                                    void** kernelParams, void** extra)` -
* launches kernel f with launch parameters and shared memory on stream with arguments passed
* to kernelparams
*/

/**
 * Test Description
 * ------------------------
 * - Test case to verify Negative tests of hipModuleLaunchKernel API.
 * - Test case to verify hipModuleLaunchKernel API's Corner Scenarios for Grid and Block dimensions.
 * - Test case to verify different work groups of hipModuleLaunchKernel API.

 * Test source
 * ------------------------
 * - catch/unit/module/hipModuleLaunchKernel.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 5.6
*/

struct gridblockDim {
  unsigned int gridX;
  unsigned int gridY;
  unsigned int gridZ;
  unsigned int blockX;
  unsigned int blockY;
  unsigned int blockZ;
};

bool Module_Negative_tests() {
  bool testStatus = true;
  constexpr auto matmulK = "matmulK";
  constexpr auto KernelandExtra = "KernelandExtraParams";
  HIP_CHECK(hipSetDevice(0));
  hipError_t err;
  struct {
    void* _Ad;
    void* _Bd;
    void* _Cd;
    int _n;
  } args1;
  args1._Ad = nullptr;
  args1._Bd = nullptr;
  args1._Cd = nullptr;
  args1._n = 0;
  hipFunction_t MultKernel, KernelandExtraParamKernel;
  size_t size1;
  size1 = sizeof(args1);
  hipModule_t Module;
  hipStream_t stream1;
  hipDeviceptr_t* Ad = nullptr;
#ifdef HT_NVIDIA
  HIP_CHECK(hipInit(0));
  hipCtx_t context;
  HIP_CHECK(hipCtxCreate(&context, 0, 0));
#endif

  HIP_CHECK(hipModuleLoad(&Module, fileName));
  HIP_CHECK(hipModuleGetFunction(&MultKernel, Module, matmulK));
  HIP_CHECK(hipModuleGetFunction(&KernelandExtraParamKernel, Module, KernelandExtra));
  void* config1[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args1, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size1,
                     HIP_LAUNCH_PARAM_END};
  void* params[] = {Ad};
  HIP_CHECK(hipStreamCreate(&stream1));
  // Passing nullptr to kernel function
  err = hipModuleLaunchKernel(nullptr, 1, 1, 1, 1, 1, 1, 0, stream1, NULL,
                              reinterpret_cast<void**>(&config1));
  if (err == hipSuccess) {
    testStatus = false;
  }
  // Passing Max int value to block dimensions
  err = hipModuleLaunchKernel(MultKernel, 1, 1, 1, std::numeric_limits<uint32_t>::max(),
                              std::numeric_limits<uint32_t>::max(),
                              std::numeric_limits<uint32_t>::max(), 0, stream1, NULL,
                              reinterpret_cast<void**>(&config1));
  if (err == hipSuccess) {
    testStatus = false;
  }
  // Passing 0 as value for all dimensions
  err = hipModuleLaunchKernel(MultKernel, 0, 0, 0, 0, 0, 0, 0, stream1, NULL,
                              reinterpret_cast<void**>(&config1));
  if (err == hipSuccess) {
    testStatus = false;
  }
  // Passing 0 as value for x dimension
  err = hipModuleLaunchKernel(MultKernel, 0, 1, 1, 0, 1, 1, 0, stream1, NULL,
                              reinterpret_cast<void**>(&config1));
  if (err == hipSuccess) {
    testStatus = false;
  }
  // Passing 0 as value for y dimension
  err = hipModuleLaunchKernel(MultKernel, 1, 0, 1, 1, 0, 1, 0, stream1, NULL,
                              reinterpret_cast<void**>(&config1));
  if (err == hipSuccess) {
    testStatus = false;
  }
  // Passing 0 as value for z dimension
  err = hipModuleLaunchKernel(MultKernel, 1, 1, 0, 1, 1, 0, 0, stream1, NULL,
                              reinterpret_cast<void**>(&config1));
  if (err == hipSuccess) {
    testStatus = false;
  }
  // Passing both kernel and extra params
  err =
      hipModuleLaunchKernel(KernelandExtraParamKernel, 1, 1, 1, 1, 1, 1, 0, stream1,
                            reinterpret_cast<void**>(&params), reinterpret_cast<void**>(&config1));
  if (err == hipSuccess) {
    testStatus = false;
  }
  // Passing more than maxthreadsperblock to block dimensions
  hipDeviceProp_t deviceProp;
  HIP_CHECK(hipGetDeviceProperties(&deviceProp, 0));
  err = hipModuleLaunchKernel(MultKernel, 1, 1, 1, deviceProp.maxThreadsPerBlock + 1,
                              deviceProp.maxThreadsPerBlock + 1, deviceProp.maxThreadsPerBlock + 1,
                              0, stream1, NULL, reinterpret_cast<void**>(&config1));
  if (err == hipSuccess) {
    testStatus = false;
  }
  // Block dimension X = Max Allowed + 1
  err = hipModuleLaunchKernel(MultKernel, 1, 1, 1, deviceProp.maxThreadsDim[0] + 1, 1, 1, 0,
                              stream1, NULL, reinterpret_cast<void**>(&config1));
  if (err == hipSuccess) {
    testStatus = false;
  }
  // Block dimension Y = Max Allowed + 1
  err = hipModuleLaunchKernel(MultKernel, 1, 1, 1, 1, deviceProp.maxThreadsDim[1] + 1, 1, 0,
                              stream1, NULL, reinterpret_cast<void**>(&config1));
  if (err == hipSuccess) {
    testStatus = false;
  }
  // Block dimension Z = Max Allowed + 1
  err = hipModuleLaunchKernel(MultKernel, 1, 1, 1, 1, 1, deviceProp.maxThreadsDim[2] + 1, 0,
                              stream1, NULL, reinterpret_cast<void**>(&config1));
  if (err == hipSuccess) {
    testStatus = false;
  }
  // Passing invalid config data to extra params
  void* config3[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size1,
                     HIP_LAUNCH_PARAM_END};
  err = hipModuleLaunchKernel(MultKernel, 1, 1, 1, 1, 1, 1, 0, stream1, NULL,
                              reinterpret_cast<void**>(&config3));
  if (err == hipSuccess) {
    testStatus = false;
  }
  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipModuleUnload(Module));
#ifdef HT_NVIDIA
  HIP_CHECK(hipCtxDestroy(context));
#endif
  return testStatus;
}

bool Module_GridBlock_Corner_Tests() {
  bool testStatus = true;
  HIP_CHECK(hipSetDevice(0));
  hipError_t err;
  hipFunction_t DummyKernel;
  hipModule_t Module;
  hipStream_t stream1;
  hipDevice_t device;
#ifdef HT_NVIDIA
  HIP_CHECK(hipInit(0));
  hipCtx_t context;
  HIP_CHECK(hipCtxCreate(&context, 0, 0));
#endif
  HIP_CHECK(hipModuleLoad(&Module, fileName));
  HIP_CHECK(hipModuleGetFunction(&DummyKernel, Module, dummyKernel));
  HIP_CHECK(hipStreamCreate(&stream1));
  // Passing Max int value to block dimensions
  hipDeviceProp_t deviceProp;
  HIP_CHECK(hipDeviceGet(&device, 0));
  HIP_CHECK(hipGetDeviceProperties(&deviceProp, device));
  unsigned int maxblockX = deviceProp.maxThreadsDim[0];
  unsigned int maxblockY = deviceProp.maxThreadsDim[1];
  unsigned int maxblockZ = deviceProp.maxThreadsDim[2];
#ifdef HT_NVIDIA
  unsigned int maxgridX = deviceProp.maxGridSize[0];
  unsigned int maxgridY = deviceProp.maxGridSize[1];
  unsigned int maxgridZ = deviceProp.maxGridSize[2];
#else
  unsigned int maxgridX = deviceProp.maxGridSize[0];
  unsigned int maxgridY = deviceProp.maxGridSize[1];
  unsigned int maxgridZ = deviceProp.maxGridSize[2];
#endif
  struct gridblockDim test[6] = {{1, 1, 1, maxblockX, 1, 1}, {1, 1, 1, 1, maxblockY, 1},
                                 {1, 1, 1, 1, 1, maxblockZ}, {maxgridX, 1, 1, 1, 1, 1},
                                 {1, maxgridY, 1, 1, 1, 1},  {1, 1, maxgridZ, 1, 1, 1}};
  for (int i = 0; i < 6; i++) {
    err = hipModuleLaunchKernel(DummyKernel, test[i].gridX, test[i].gridY, test[i].gridZ,
                                test[i].blockX, test[i].blockY, test[i].blockZ, 0, stream1, NULL,
                                NULL);
    if (err != hipSuccess) {
      testStatus = false;
    }
  }
  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipModuleUnload(Module));
#ifdef HT_NVIDIA
  HIP_CHECK(hipCtxDestroy(context));
#endif
  return testStatus;
}

bool Module_WorkGroup_Test() {
  bool testStatus = true;
  HIP_CHECK(hipSetDevice(0));
  hipError_t err;
  hipFunction_t DummyKernel;
  hipModule_t Module;
  hipStream_t stream1;
#ifdef HT_NVIDIA
  HIP_CHECK(hipInit(0));
  hipCtx_t context;
  HIP_CHECK(hipCtxCreate(&context, 0, 0));
#endif
  HIP_CHECK(hipModuleLoad(&Module, fileName));
  HIP_CHECK(hipModuleGetFunction(&DummyKernel, Module, dummyKernel));
  HIP_CHECK(hipStreamCreate(&stream1));
  // Passing Max int value to block dimensions
  hipDeviceProp_t deviceProp;
  HIP_CHECK(hipGetDeviceProperties(&deviceProp, 0));
  double cuberootVal = cbrt(static_cast<double>(deviceProp.maxThreadsPerBlock));
  uint32_t cuberoot_floor = floor(cuberootVal);
  uint32_t cuberoot_ceil = ceil(cuberootVal);
  // Scenario: (block.x * block.y * block.z) <= Work Group Size where
  // block.x < MaxBlockDimX , block.y < MaxBlockDimY and block.z < MaxBlockDimZ
  err = hipModuleLaunchKernel(DummyKernel, 1, 1, 1, cuberoot_floor, cuberoot_floor, cuberoot_floor,
                              0, stream1, NULL, NULL);
  if (err != hipSuccess) {
    testStatus = false;
  }
  // Scenario: (block.x * block.y * block.z) > Work Group Size where
  // block.x < MaxBlockDimX , block.y < MaxBlockDimY and block.z < MaxBlockDimZ
  err = hipModuleLaunchKernel(DummyKernel, 1, 1, 1, cuberoot_ceil, cuberoot_ceil, cuberoot_ceil + 1,
                              0, stream1, NULL, NULL);
  if (err == hipSuccess) {
    testStatus = false;
  }
  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipModuleUnload(Module));
#ifdef HT_NVIDIA
  HIP_CHECK(hipCtxDestroy(context));
#endif
  return testStatus;
}

TEST_CASE("Unit_hipModuleLaunchKernel_Fntl") {
  bool testStatus = false;
  SECTION("Negative test scenarios") {
    testStatus = Module_Negative_tests();
    REQUIRE(testStatus == true);
  }
  SECTION("Grid Block corner test") {
    testStatus = Module_GridBlock_Corner_Tests();
    REQUIRE(testStatus == true);
  }
  SECTION("Work Group Test") {
    testStatus = Module_WorkGroup_Test();
    REQUIRE(testStatus == true);
  }
}
