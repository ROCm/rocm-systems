/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <hip_test_helper.hh>
#include <utils.hh>
#include "hip/hip_ext.h"
#include "hip_module_common.hh"
#include "../device/hipGetProcAddressHelpers.hh"

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different module management
 *  - (load/unload/GetAttribute/launch) related APIs from the hipGetProcAddress API
 *  - and then validates the basic functionality of that particular API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/module/hipGetProcAddress_Module_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_ModuleApis") {
  void* hipModuleLoad_ptr = nullptr;
  void* hipModuleUnload_ptr = nullptr;
  void* hipModuleGetFunction_ptr = nullptr;
  void* hipModuleLaunchKernel_ptr = nullptr;
  void* hipGetFuncBySymbol_ptr = nullptr;
  void* hipFuncGetAttributes_ptr = nullptr;
  void* hipFuncGetAttribute_ptr = nullptr;
  void* hipModuleGetGlobal_ptr = nullptr;
  void* hipExtModuleLaunchKernel_ptr = nullptr;
  void* hipHccModuleLaunchKernel_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress("hipModuleLoad", &hipModuleLoad_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(
      hipGetProcAddress("hipModuleUnload", &hipModuleUnload_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipModuleGetFunction", &hipModuleGetFunction_ptr, currentHipVersion,
                              0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipModuleLaunchKernel", &hipModuleLaunchKernel_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipGetFuncBySymbol", &hipGetFuncBySymbol_ptr, currentHipVersion, 0,
                              nullptr));
  HIP_CHECK(hipGetProcAddress("hipFuncGetAttributes", &hipFuncGetAttributes_ptr, currentHipVersion,
                              0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipFuncGetAttribute", &hipFuncGetAttribute_ptr, currentHipVersion, 0,
                              nullptr));
  HIP_CHECK(hipGetProcAddress("hipModuleGetGlobal", &hipModuleGetGlobal_ptr, currentHipVersion, 0,
                              nullptr));
  HIP_CHECK(hipGetProcAddress("hipExtModuleLaunchKernel", &hipExtModuleLaunchKernel_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipHccModuleLaunchKernel", &hipHccModuleLaunchKernel_ptr,
                              currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipModuleLoad_ptr)(hipModule_t*, const char*) =
      reinterpret_cast<hipError_t (*)(hipModule_t*, const char*)>(hipModuleLoad_ptr);
  hipError_t (*dyn_hipModuleUnload_ptr)(hipModule_t) =
      reinterpret_cast<hipError_t (*)(hipModule_t)>(hipModuleUnload_ptr);
  hipError_t (*dyn_hipModuleGetFunction_ptr)(hipFunction_t*, hipModule_t, const char*) =
      reinterpret_cast<hipError_t (*)(hipFunction_t*, hipModule_t, const char*)>(
          hipModuleGetFunction_ptr);
  hipError_t (*dyn_hipModuleLaunchKernel_ptr)(
      hipFunction_t, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int,
      unsigned int, unsigned int, hipStream_t, void**, void**) =
      reinterpret_cast<hipError_t (*)(hipFunction_t, unsigned int, unsigned int, unsigned int,
                                      unsigned int, unsigned int, unsigned int, unsigned int,
                                      hipStream_t, void**, void**)>(hipModuleLaunchKernel_ptr);
  hipError_t (*dyn_hipGetFuncBySymbol_ptr)(hipFunction_t*, const void*) =
      reinterpret_cast<hipError_t (*)(hipFunction_t*, const void*)>(hipGetFuncBySymbol_ptr);
  hipError_t (*dyn_hipFuncGetAttributes_ptr)(struct hipFuncAttributes*, const void*) =
      reinterpret_cast<hipError_t (*)(struct hipFuncAttributes*, const void*)>(
          hipFuncGetAttributes_ptr);
  hipError_t (*dyn_hipFuncGetAttribute_ptr)(int*, hipFunction_attribute, hipFunction_t) =
      reinterpret_cast<hipError_t (*)(int*, hipFunction_attribute, hipFunction_t)>(
          hipFuncGetAttribute_ptr);
  hipError_t (*dyn_hipModuleGetGlobal_ptr)(hipDeviceptr_t*, size_t*, hipModule_t, const char*) =
      reinterpret_cast<hipError_t (*)(hipDeviceptr_t*, size_t*, hipModule_t, const char*)>(
          hipModuleGetGlobal_ptr);

  hipError_t (*dyn_hipExtModuleLaunchKernel_ptr)(hipFunction_t, uint32_t, uint32_t, uint32_t,
                                                 uint32_t, uint32_t, uint32_t, size_t, hipStream_t,
                                                 void**, void**, hipEvent_t, hipEvent_t, uint32_t) =
      reinterpret_cast<hipError_t (*)(hipFunction_t, uint32_t, uint32_t, uint32_t, uint32_t,
                                      uint32_t, uint32_t, size_t, hipStream_t, void**, void**,
                                      hipEvent_t, hipEvent_t, uint32_t)>(
          hipExtModuleLaunchKernel_ptr);

  hipError_t (*dyn_hipHccModuleLaunchKernel_ptr)(hipFunction_t, uint32_t, uint32_t, uint32_t,
                                                 uint32_t, uint32_t, uint32_t, size_t, hipStream_t,
                                                 void**, void**, hipEvent_t, hipEvent_t) =
      reinterpret_cast<hipError_t (*)(hipFunction_t, uint32_t, uint32_t, uint32_t, uint32_t,
                                      uint32_t, uint32_t, size_t, hipStream_t, void**, void**,
                                      hipEvent_t, hipEvent_t)>(hipHccModuleLaunchKernel_ptr);

  // Validating hipModuleLoad API
  hipModule_t module;
  HIP_CHECK(dyn_hipModuleLoad_ptr(&module, "addKernel.code"));
  REQUIRE(module != nullptr);

  // Validating hipModuleGetFunction API
  hipFunction_t function;
  HIP_CHECK(dyn_hipModuleGetFunction_ptr(&function, module, "addKernel"));
  REQUIRE(function != nullptr);

  // Validating  hipModuleLaunchKernel API
  const int N = 10;
  const int Nbytes = 10 * sizeof(int);

  int* hostArr = reinterpret_cast<int*>(malloc(Nbytes));
  REQUIRE(hostArr != nullptr);
  fillHostArray(hostArr, N, 10);

  int* devArr = nullptr;
  HIP_CHECK(hipMalloc(&devArr, Nbytes));
  REQUIRE(devArr != nullptr);
  HIP_CHECK(hipMemcpy(devArr, hostArr, Nbytes, hipMemcpyHostToDevice));

  dim3 blocksPerGrid(1, 1, 1);
  dim3 threadsPerBlock(1, 1, N);

  struct kernelParameters {
    void* arr;
    int size;
  };
  kernelParameters kernelParam{};
  kernelParam.arr = devArr;
  kernelParam.size = N;

  auto size = sizeof(kernelParam);
  void* kernel_parameter[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &kernelParam,
                              HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, HIP_LAUNCH_PARAM_END};

  HIP_CHECK(dyn_hipModuleLaunchKernel_ptr(function, blocksPerGrid.x, blocksPerGrid.y,
                                          blocksPerGrid.z, threadsPerBlock.x, threadsPerBlock.y,
                                          threadsPerBlock.z, 0, 0, nullptr, kernel_parameter));

  HIP_CHECK(hipMemcpy(hostArr, devArr, Nbytes, hipMemcpyDeviceToHost));
  REQUIRE(validateHostArray(hostArr, N, 12) == true);

  // Validating hipExtModuleLaunchKernel API
  HIP_CHECK(dyn_hipExtModuleLaunchKernel_ptr(
      function, blocksPerGrid.x, blocksPerGrid.y, blocksPerGrid.z, threadsPerBlock.x,
      threadsPerBlock.y, threadsPerBlock.z, 0, 0, nullptr, kernel_parameter, nullptr, nullptr, 0));

  HIP_CHECK(hipMemcpy(hostArr, devArr, Nbytes, hipMemcpyDeviceToHost));
  REQUIRE(validateHostArray(hostArr, N, 14) == true);

  // Validating hipHccModuleLaunchKernel API
  HIP_CHECK(dyn_hipHccModuleLaunchKernel_ptr(
      function, blocksPerGrid.x, blocksPerGrid.y, blocksPerGrid.z, threadsPerBlock.x,
      threadsPerBlock.y, threadsPerBlock.z, 0, 0, nullptr, kernel_parameter, nullptr, nullptr));

  HIP_CHECK(hipMemcpy(hostArr, devArr, Nbytes, hipMemcpyDeviceToHost));
  REQUIRE(validateHostArray(hostArr, N, 16) == true);

  // Validating hipGetFuncBySymbol API
  hipFunction_t functionWithOrgApi, functionWithFuncPtr;
  HIP_CHECK(hipGetFuncBySymbol(&functionWithOrgApi, reinterpret_cast<const void*>(addOneKernel)));
  REQUIRE(functionWithOrgApi != nullptr);

  HIP_CHECK(dyn_hipGetFuncBySymbol_ptr(&functionWithFuncPtr,
                                       reinterpret_cast<const void*>(addOneKernel)));
  REQUIRE(functionWithFuncPtr != nullptr);

  REQUIRE(functionWithFuncPtr == functionWithOrgApi);

  // Validating hipFuncGetAttributes API
  struct hipFuncAttributes attrWithOrgApi, attrWithFuncPtr;

  HIP_CHECK(hipFuncGetAttributes(&attrWithOrgApi, reinterpret_cast<const void*>(addOneKernel)));
  HIP_CHECK(
      dyn_hipFuncGetAttributes_ptr(&attrWithFuncPtr, reinterpret_cast<const void*>(addOneKernel)));

  REQUIRE(attrWithFuncPtr.binaryVersion == attrWithOrgApi.binaryVersion);
  REQUIRE(attrWithFuncPtr.cacheModeCA == attrWithOrgApi.cacheModeCA);
  REQUIRE(attrWithFuncPtr.constSizeBytes == attrWithOrgApi.constSizeBytes);
  REQUIRE(attrWithFuncPtr.localSizeBytes == attrWithOrgApi.localSizeBytes);
  REQUIRE(attrWithFuncPtr.maxDynamicSharedSizeBytes == attrWithOrgApi.maxDynamicSharedSizeBytes);
  REQUIRE(attrWithFuncPtr.maxThreadsPerBlock == attrWithOrgApi.maxThreadsPerBlock);
  REQUIRE(attrWithFuncPtr.numRegs == attrWithOrgApi.numRegs);
  REQUIRE(attrWithFuncPtr.preferredShmemCarveout == attrWithOrgApi.preferredShmemCarveout);
  REQUIRE(attrWithFuncPtr.ptxVersion == attrWithOrgApi.ptxVersion);
  REQUIRE(attrWithFuncPtr.sharedSizeBytes == attrWithOrgApi.sharedSizeBytes);

  // Validating hipFuncGetAttribute API
  hipFunction_attribute attributes[] = {HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                        HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                                        HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES,
                                        HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                                        HIP_FUNC_ATTRIBUTE_NUM_REGS,
                                        HIP_FUNC_ATTRIBUTE_PTX_VERSION,
                                        HIP_FUNC_ATTRIBUTE_BINARY_VERSION,
                                        HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA,
                                        HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                        HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT};

  for (auto attribute : attributes) {
    int valuewithOrgAPI = 0, valueWithFuncPointer = 0;

    HIP_CHECK(hipFuncGetAttribute(&valuewithOrgAPI, attribute, function));
    HIP_CHECK(dyn_hipFuncGetAttribute_ptr(&valueWithFuncPointer, attribute, function));

    REQUIRE(valueWithFuncPointer == valuewithOrgAPI);
  }

  // Validating hipModuleGetGlobal API
  hipDeviceptr_t dptrWithOrgApi = nullptr;
  size_t bytesWithOrgApi = 0;
  HIP_CHECK(hipModuleGetGlobal(&dptrWithOrgApi, &bytesWithOrgApi, module, "globalDevData"));
  REQUIRE(dptrWithOrgApi != nullptr);

  hipDeviceptr_t dptrWithFuncPtr = nullptr;
  size_t bytesWithFuncPtr = 0;
  HIP_CHECK(
      dyn_hipModuleGetGlobal_ptr(&dptrWithFuncPtr, &bytesWithFuncPtr, module, "globalDevData"));
  REQUIRE(dptrWithFuncPtr != nullptr);
  REQUIRE(bytesWithFuncPtr == 4);

  REQUIRE(dptrWithFuncPtr == dptrWithOrgApi);
  REQUIRE(bytesWithFuncPtr == bytesWithOrgApi);

  // Validating hipModuleUnload API
  HIP_CHECK(dyn_hipModuleUnload_ptr(module));
  REQUIRE(dyn_hipModuleUnload_ptr(module) == hipErrorNotFound);

  free(hostArr);
  HIP_CHECK(hipFree(devArr));
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different module management
 *  - (load data) related APIs from the hipGetProcAddress API
 *  - and then validates the basic functionality of that particular API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/module/hipGetProcAddress_Module_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_ModuleApisLoadData") {
  void* hipModuleLoadData_ptr = nullptr;
  void* hipModuleLoadDataEx_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress("hipModuleLoadData", &hipModuleLoadData_ptr, currentHipVersion, 0,
                              nullptr));
  HIP_CHECK(hipGetProcAddress("hipModuleLoadDataEx", &hipModuleLoadDataEx_ptr, currentHipVersion, 0,
                              nullptr));

  hipError_t (*dyn_hipModuleLoadData_ptr)(hipModule_t*, const void*) =
      reinterpret_cast<hipError_t (*)(hipModule_t*, const void*)>(hipModuleLoadData_ptr);
  hipError_t (*dyn_hipModuleLoadDataEx_ptr)(hipModule_t*, const void*, unsigned int, hipJitOption*,
                                            void**) =
      reinterpret_cast<hipError_t (*)(hipModule_t*, const void*, unsigned int, hipJitOption*,
                                      void**)>(hipModuleLoadDataEx_ptr);

  const auto rtc = CreateRTCCharArray(R"(extern "C" __global__ void simpleKernel() {})");

  // Validating hipModuleLoadData API
  {
    hipModule_t module = nullptr;

    HIP_CHECK(dyn_hipModuleLoadData_ptr(&module, rtc.data()));
    REQUIRE(module != nullptr);

    hipFunction_t function;
    HIP_CHECK(hipModuleGetFunction(&function, module, "simpleKernel"));
    REQUIRE(function != nullptr);
    HIP_CHECK(hipModuleLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, 0, nullptr, nullptr));

    HIP_CHECK(hipModuleUnload(module));
  }

  // Validating hipModuleLoadDataEx API
  {
    hipModule_t module = nullptr;

    HIP_CHECK(dyn_hipModuleLoadDataEx_ptr(&module, rtc.data(), 0, nullptr, nullptr));
    REQUIRE(module != nullptr);

    hipFunction_t function;
    HIP_CHECK(hipModuleGetFunction(&function, module, "simpleKernel"));
    REQUIRE(function != nullptr);
    HIP_CHECK(hipModuleLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, 0, nullptr, nullptr));

    HIP_CHECK(hipModuleUnload(module));
  }
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different module management
 *  - (Cooperative Kernels) related APIs from the hipGetProcAddress API
 *  - and then validates the basic functionality of that particular API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/module/hipGetProcAddress_Module_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_ModuleApisCooperativeKernels") {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeCooperativeLaunch)) {
    HipTest::HIP_SKIP_TEST("CooperativeLaunch not supported");
    return;
  }

  HIP_CHECK(hipSetDevice(0));

  void* hipModuleLaunchCooperativeKernel_ptr = nullptr;
  void* hipModuleLaunchCooperativeKernelMultiDevice_ptr = nullptr;
  void* hipLaunchCooperativeKernel_ptr = nullptr;
  void* hipLaunchCooperativeKernelMultiDevice_ptr = nullptr;
  void* hipExtLaunchMultiKernelMultiDevice_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress("hipModuleLaunchCooperativeKernel",
                              &hipModuleLaunchCooperativeKernel_ptr, currentHipVersion, 0,
                              nullptr));
  HIP_CHECK(hipGetProcAddress("hipModuleLaunchCooperativeKernelMultiDevice",
                              &hipModuleLaunchCooperativeKernelMultiDevice_ptr, currentHipVersion,
                              0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipLaunchCooperativeKernel", &hipLaunchCooperativeKernel_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipLaunchCooperativeKernelMultiDevice",
                              &hipLaunchCooperativeKernelMultiDevice_ptr, currentHipVersion, 0,
                              nullptr));
  HIP_CHECK(hipGetProcAddress("hipExtLaunchMultiKernelMultiDevice",
                              &hipExtLaunchMultiKernelMultiDevice_ptr, currentHipVersion, 0,
                              nullptr));

  hipError_t (*dyn_hipModuleLaunchCooperativeKernel_ptr)(
      hipFunction_t, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int,
      unsigned int, unsigned int, hipStream_t, void**) =
      reinterpret_cast<hipError_t (*)(hipFunction_t, unsigned int, unsigned int, unsigned int,
                                      unsigned int, unsigned int, unsigned int, unsigned int,
                                      hipStream_t, void**)>(hipModuleLaunchCooperativeKernel_ptr);

  hipError_t (*dyn_hipModuleLaunchCooperativeKernelMultiDevice_ptr)(hipFunctionLaunchParams*,
                                                                    unsigned int, unsigned int) =
      reinterpret_cast<hipError_t (*)(hipFunctionLaunchParams*, unsigned int, unsigned int)>(
          hipModuleLaunchCooperativeKernelMultiDevice_ptr);

  hipError_t (*dyn_hipLaunchCooperativeKernel_ptr)(const void*, dim3, dim3, void**, unsigned int,
                                                   hipStream_t) =
      reinterpret_cast<hipError_t (*)(const void*, dim3, dim3, void**, unsigned int, hipStream_t)>(
          hipLaunchCooperativeKernel_ptr);

  hipError_t (*dyn_hipLaunchCooperativeKernelMultiDevice_ptr)(hipLaunchParams*, int, unsigned int) =
      reinterpret_cast<hipError_t (*)(hipLaunchParams*, int, unsigned int)>(
          hipLaunchCooperativeKernelMultiDevice_ptr);

  hipError_t (*dyn_hipExtLaunchMultiKernelMultiDevice_ptr)(hipLaunchParams*, int, unsigned int) =
      reinterpret_cast<hipError_t (*)(hipLaunchParams*, int, unsigned int)>(
          hipExtLaunchMultiKernelMultiDevice_ptr);

  const int N = 10;
  const int Nbytes = 10 * sizeof(int);

  int* hostArr = reinterpret_cast<int*>(malloc(Nbytes));
  REQUIRE(hostArr != nullptr);
  fillHostArray(hostArr, N, 10);

  int* devArr = nullptr;
  HIP_CHECK(hipMalloc(&devArr, Nbytes));
  REQUIRE(devArr != nullptr);
  HIP_CHECK(hipMemcpy(devArr, hostArr, Nbytes, hipMemcpyHostToDevice));

  dim3 blocksPerGrid(1, 1, 1);
  dim3 threadsPerBlock(1, 1, N);

  struct kernelParameters {
    void* arr;
    int size;
  };
  kernelParameters kernelParam;
  kernelParam.arr = devArr;
  kernelParam.size = N;
  void* kernel_parameter[] = {&kernelParam.arr, &kernelParam.size};

  // Validating hipModuleLaunchCooperativeKernel API
  {
    hipModule_t module;
    HIP_CHECK(hipModuleLoad(&module, "addKernel.code"));
    REQUIRE(module != nullptr);

    hipFunction_t function;
    HIP_CHECK(hipModuleGetFunction(&function, module, "addKernel"));
    REQUIRE(function != nullptr);

    HIP_CHECK(dyn_hipModuleLaunchCooperativeKernel_ptr(
        function, blocksPerGrid.x, blocksPerGrid.y, blocksPerGrid.z, threadsPerBlock.x,
        threadsPerBlock.y, threadsPerBlock.z, 0, 0, kernel_parameter));

    HIP_CHECK(hipMemcpy(hostArr, devArr, Nbytes, hipMemcpyDeviceToHost));
    REQUIRE(validateHostArray(hostArr, N, 12) == true);
    HIP_CHECK(hipModuleUnload(module));
  }

  // Validating hipModuleLaunchCooperativeKernelMultiDevice API
  {
    int deviceCount = 0;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));

    auto module = std::make_unique<hipModule_t[]>(deviceCount);
    auto function = std::make_unique<hipFunction_t[]>(deviceCount);
    auto stream_arr = std::make_unique<hipStream_t[]>(deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
      HIP_CHECK(hipSetDevice(i));
      HIP_CHECK(hipStreamCreate(&stream_arr[i]));

      HIP_CHECK(hipModuleLoad(&module[i], "addKernel.code"));
      REQUIRE(module[i] != nullptr);

      HIP_CHECK(hipModuleGetFunction(&function[i], module[i], "sampleModuleKernel"));
      REQUIRE(function[i] != nullptr);
    }

    HIP_CHECK(hipSetDevice(0));

    ::std::vector<hipFunctionLaunchParams> params(deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
      params[i].function = function[i];
      params[i].gridDimX = 1;
      params[i].gridDimY = 1;
      params[i].gridDimZ = 1;
      params[i].blockDimX = 1;
      params[i].blockDimY = 1;
      params[i].blockDimZ = 1;
      params[i].kernelParams = nullptr;
      params[i].sharedMemBytes = 0;
      params[i].hStream = stream_arr[i];
    }

    HIP_CHECK(dyn_hipModuleLaunchCooperativeKernelMultiDevice_ptr(params.data(), deviceCount, 0));

    for (int i = 0; i < deviceCount; ++i) {
      HIP_CHECK(hipStreamSynchronize(params[i].hStream));
    }

    for (int i = 0; i < deviceCount; ++i) {
      HIP_CHECK(hipStreamDestroy(stream_arr[i]));
      HIP_CHECK(hipModuleUnload(module[i]));
    }
  }

  // Validating hipLaunchCooperativeKernel API
  {
    HIP_CHECK(dyn_hipLaunchCooperativeKernel_ptr(reinterpret_cast<void*>(addOneKernel),
                                                 dim3(1, 1, 1), dim3(1, 1, 1), kernel_parameter, 0,
                                                 0));
    HIP_CHECK(hipMemcpy(hostArr, devArr, Nbytes, hipMemcpyDeviceToHost));
    REQUIRE(validateHostArray(hostArr, N, 13) == true);
  }

  // Validating hipLaunchCooperativeKernelMultiDevice API
  {
    int deviceCount = 0;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));

    auto stream_arr = std::make_unique<hipStream_t[]>(deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
      HIP_CHECK(hipSetDevice(i));
      HIP_CHECK(hipStreamCreate(&stream_arr[i]));
    }

    HIP_CHECK(hipSetDevice(0));

    std::vector<hipLaunchParams> params(deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
      params[i].func = reinterpret_cast<void*>(simpleKernel);
      params[i].gridDim = {1, 1, 1};
      params[i].blockDim = {1, 1, 1};
      params[i].args = nullptr;
      params[i].sharedMem = 0;
      params[i].stream = stream_arr[i];
    }

    HIP_CHECK(dyn_hipLaunchCooperativeKernelMultiDevice_ptr(params.data(), deviceCount, 0));

    for (int i = 0; i < deviceCount; ++i) {
      HIP_CHECK(hipStreamSynchronize(params[i].stream));
    }

    for (int i = 0; i < deviceCount; ++i) {
      HIP_CHECK(hipStreamDestroy(stream_arr[i]));
    }
  }

  // Validating hipExtLaunchMultiKernelMultiDevice API
  {
    int deviceCount = 0;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));

    auto stream_arr = std::make_unique<hipStream_t[]>(deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
      HIP_CHECK(hipSetDevice(i));
      HIP_CHECK(hipStreamCreate(&stream_arr[i]));
    }

    HIP_CHECK(hipSetDevice(0));

    std::vector<hipLaunchParams> params(deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
      params[i].func = reinterpret_cast<void*>(simpleKernel);
      params[i].gridDim = {1, 1, 1};
      params[i].blockDim = {1, 1, 1};
      params[i].args = nullptr;
      params[i].sharedMem = 0;
      params[i].stream = stream_arr[i];
    }

    HIP_CHECK(dyn_hipExtLaunchMultiKernelMultiDevice_ptr(params.data(), deviceCount, 0));

    for (int i = 0; i < deviceCount; ++i) {
      HIP_CHECK(hipStreamSynchronize(params[i].stream));
    }

    for (int i = 0; i < deviceCount; ++i) {
      HIP_CHECK(hipStreamDestroy(stream_arr[i]));
    }
  }

  free(hostArr);
  HIP_CHECK(hipFree(devArr));
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different Occupancy
 *  - related APIs from the hipGetProcAddress API
 *  - and then validates the basic functionality of that particular API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/module/hipGetProcAddress_Module_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_ModuleApisOccupancy") {
  void* hipModuleOccupancyMaxPotentialBlockSize_ptr = nullptr;
  void* hipModuleOccupancyMaxPotentialBlockSizeWithFlags_ptr = nullptr;
  void* hipModuleOccupancyMaxActiveBlocksPerMultiprocessor_ptr = nullptr;
  void* hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_ptr = nullptr;
  void* hipOccupancyMaxActiveBlocksPerMultiprocessor_ptr = nullptr;
  void* hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_ptr = nullptr;
  void* hipOccupancyMaxPotentialBlockSize_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress("hipModuleOccupancyMaxPotentialBlockSize",
                              &hipModuleOccupancyMaxPotentialBlockSize_ptr, currentHipVersion, 0,
                              nullptr));
  HIP_CHECK(hipGetProcAddress("hipModuleOccupancyMaxPotentialBlockSizeWithFlags",
                              &hipModuleOccupancyMaxPotentialBlockSizeWithFlags_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipModuleOccupancyMaxActiveBlocksPerMultiprocessor",
                              &hipModuleOccupancyMaxActiveBlocksPerMultiprocessor_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
                              &hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipOccupancyMaxActiveBlocksPerMultiprocessor",
                              &hipOccupancyMaxActiveBlocksPerMultiprocessor_ptr, currentHipVersion,
                              0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
                              &hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipOccupancyMaxPotentialBlockSize",
                              &hipOccupancyMaxPotentialBlockSize_ptr, currentHipVersion, 0,
                              nullptr));

  hipError_t (*dyn_hipModuleOccupancyMaxPotentialBlockSize_ptr)(int*, int*, hipFunction_t, size_t,
                                                                int) =
      reinterpret_cast<hipError_t (*)(int*, int*, hipFunction_t, size_t, int)>(
          hipModuleOccupancyMaxPotentialBlockSize_ptr);

  hipError_t (*dyn_hipModuleOccupancyMaxPotentialBlockSizeWithFlags_ptr)(
      int*, int*, hipFunction_t, size_t, int, unsigned int) =
      reinterpret_cast<hipError_t (*)(int*, int*, hipFunction_t, size_t, int, unsigned int)>(
          hipModuleOccupancyMaxPotentialBlockSizeWithFlags_ptr);

  hipError_t (*dyn_hipModuleOccupancyMaxActiveBlocksPerMultiprocessor_ptr)(int*, hipFunction_t, int,
                                                                           size_t) =
      reinterpret_cast<hipError_t (*)(int*, hipFunction_t, int, size_t)>(
          hipModuleOccupancyMaxActiveBlocksPerMultiprocessor_ptr);

  hipError_t (*dyn_hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_ptr)(
      int*, hipFunction_t, int, size_t, unsigned int) =
      reinterpret_cast<hipError_t (*)(int*, hipFunction_t, int, size_t, unsigned int)>(
          hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_ptr);

  hipError_t (*dyn_hipOccupancyMaxActiveBlocksPerMultiprocessor_ptr)(int*, const void*, int,
                                                                     size_t) =
      reinterpret_cast<hipError_t (*)(int*, const void*, int, size_t)>(
          hipOccupancyMaxActiveBlocksPerMultiprocessor_ptr);

  hipError_t (*dyn_hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_ptr)(
      int*, const void*, int, size_t, unsigned int) =
      reinterpret_cast<hipError_t (*)(int*, const void*, int, size_t, unsigned int)>(
          hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_ptr);

  hipError_t (*dyn_hipOccupancyMaxPotentialBlockSize_ptr)(int*, int*, const void*, size_t, int) =
      reinterpret_cast<hipError_t (*)(int*, int*, const void*, size_t, int)>(
          hipOccupancyMaxPotentialBlockSize_ptr);

  hipModule_t module;
  HIP_CHECK(hipModuleLoad(&module, "addKernel.code"));
  REQUIRE(module != nullptr);
  hipFunction_t function;
  HIP_CHECK(hipModuleGetFunction(&function, module, "addKernel"));
  REQUIRE(function != nullptr);

  int gridSize = 0, blockSize = 0;
  int gridSizeWithFuncPtr = 0, blockSizeWithFuncPtr = 0;

  // Validating hipModuleOccupancyMaxPotentialBlockSize API
  {
    HIP_CHECK(hipModuleOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, function, 0, 0));
    HIP_CHECK(dyn_hipModuleOccupancyMaxPotentialBlockSize_ptr(
        &gridSizeWithFuncPtr, &blockSizeWithFuncPtr, function, 0, 0));

    REQUIRE(gridSizeWithFuncPtr == gridSize);
    REQUIRE(blockSizeWithFuncPtr == blockSize);
  }

  // Validating hipModuleOccupancyMaxPotentialBlockSizeWithFlags API
  {
    gridSize = 0;
    blockSize = 0;
    gridSizeWithFuncPtr = 0;
    blockSizeWithFuncPtr = 0;
    HIP_CHECK(
        hipModuleOccupancyMaxPotentialBlockSizeWithFlags(&gridSize, &blockSize, function, 0, 0, 0));
    HIP_CHECK(dyn_hipModuleOccupancyMaxPotentialBlockSizeWithFlags_ptr(
        &gridSizeWithFuncPtr, &blockSizeWithFuncPtr, function, 0, 0, 0));

    REQUIRE(gridSizeWithFuncPtr == gridSize);
    REQUIRE(blockSizeWithFuncPtr == blockSize);
  }

  int numBlocks = 0, numBlocksWithFuncPtr = 0;
  // Validating hipModuleOccupancyMaxActiveBlocksPerMultiprocessor API
  {
    HIP_CHECK(
        hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, function, blockSize, 0));
    HIP_CHECK(dyn_hipModuleOccupancyMaxActiveBlocksPerMultiprocessor_ptr(&numBlocksWithFuncPtr,
                                                                         function, blockSize, 0));

    REQUIRE(numBlocksWithFuncPtr == numBlocks);
  }

  // Validating hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags API
  {
    numBlocks = 0;
    numBlocksWithFuncPtr = 0;
    HIP_CHECK(hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&numBlocks, function,
                                                                          blockSize, 0, 0));
    HIP_CHECK(dyn_hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_ptr(
        &numBlocksWithFuncPtr, function, blockSize, 0, 0));

    REQUIRE(numBlocksWithFuncPtr == numBlocks);
  }

  // Validating hipOccupancyMaxActiveBlocksPerMultiprocessor API
  {
    numBlocks = 0;
    numBlocksWithFuncPtr = 0;
    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks, reinterpret_cast<const void*>(addOneKernel), blockSize, 0));
    HIP_CHECK(dyn_hipOccupancyMaxActiveBlocksPerMultiprocessor_ptr(
        &numBlocksWithFuncPtr, reinterpret_cast<const void*>(addOneKernel), blockSize, 0));

    REQUIRE(numBlocksWithFuncPtr == numBlocks);
  }

  // Validating hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags API
  {
    numBlocks = 0;
    numBlocksWithFuncPtr = 0;
    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        &numBlocks, reinterpret_cast<const void*>(addOneKernel), blockSize, 0, 0));
    HIP_CHECK(dyn_hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_ptr(
        &numBlocksWithFuncPtr, reinterpret_cast<const void*>(addOneKernel), blockSize, 0, 0));

    REQUIRE(numBlocksWithFuncPtr == numBlocks);
  }

  // Validating hipOccupancyMaxPotentialBlockSize API
  {
    gridSize = 0;
    blockSize = 0;
    gridSizeWithFuncPtr = 0;
    blockSizeWithFuncPtr = 0;
    HIP_CHECK(hipOccupancyMaxPotentialBlockSize(&gridSize, &blockSize,
                                                reinterpret_cast<const void*>(addOneKernel), 0, 0));
    HIP_CHECK(dyn_hipOccupancyMaxPotentialBlockSize_ptr(&gridSizeWithFuncPtr, &blockSizeWithFuncPtr,
                                                        reinterpret_cast<const void*>(addOneKernel),
                                                        0, 0));

    REQUIRE(gridSizeWithFuncPtr == gridSize);
    REQUIRE(blockSizeWithFuncPtr == blockSize);
  }

  HIP_CHECK(hipModuleUnload(module));
}
