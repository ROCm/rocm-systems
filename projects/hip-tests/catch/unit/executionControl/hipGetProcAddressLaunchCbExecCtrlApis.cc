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
#include "../device/hipGetProcAddressHelpers.hh"

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different
 *  - kernel launch APIs from the hipGetProcAddress API
 *  - and then validates the basic functionality of that particular API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/executionControl/hipGetProcAddress_Launch_Callback_ExecCtrl_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_KernelLaunchApis") {
  void* hipConfigureCall_ptr = nullptr;
  void* hipSetupArgument_ptr = nullptr;
  void* hipLaunchByPtr_ptr = nullptr;
  void* __hipPushCallConfiguration_ptr = nullptr;
  void* __hipPopCallConfiguration_ptr = nullptr;
  void* hipExtLaunchKernel_ptr = nullptr;
  void* hipDrvMemcpy2DUnaligned_ptr = nullptr;
  void* hipLaunchKernel_ptr = nullptr;
  void* hipLaunchHostFunc_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(
      hipGetProcAddress("hipConfigureCall", &hipConfigureCall_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(
      hipGetProcAddress("hipSetupArgument", &hipSetupArgument_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(
      hipGetProcAddress("hipLaunchByPtr", &hipLaunchByPtr_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("__hipPushCallConfiguration", &__hipPushCallConfiguration_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("__hipPopCallConfiguration", &__hipPopCallConfiguration_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipExtLaunchKernel", &hipExtLaunchKernel_ptr, currentHipVersion, 0,
                              nullptr));
  HIP_CHECK(hipGetProcAddress("hipDrvMemcpy2DUnaligned", &hipDrvMemcpy2DUnaligned_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(
      hipGetProcAddress("hipLaunchKernel", &hipLaunchKernel_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipLaunchHostFunc", &hipLaunchHostFunc_ptr, currentHipVersion, 0,
                              nullptr));

  hipError_t (*dyn_hipConfigureCall_ptr)(dim3, dim3, size_t, hipStream_t) =
      reinterpret_cast<hipError_t (*)(dim3, dim3, size_t, hipStream_t)>(hipConfigureCall_ptr);
  hipError_t (*dyn_hipSetupArgument_ptr)(const void*, size_t, size_t) =
      reinterpret_cast<hipError_t (*)(const void*, size_t, size_t)>(hipSetupArgument_ptr);
  hipError_t (*dyn_hipLaunchByPtr_ptr)(const void*) =
      reinterpret_cast<hipError_t (*)(const void*)>(hipLaunchByPtr_ptr);
  hipError_t (*dyn__hipPushCallConfiguration_ptr)(dim3, dim3, size_t, hipStream_t) =
      reinterpret_cast<hipError_t (*)(dim3, dim3, size_t, hipStream_t)>(
          __hipPushCallConfiguration_ptr);
  hipError_t (*dyn__hipPopCallConfiguration_ptr)(dim3*, dim3*, size_t*, hipStream_t*) =
      reinterpret_cast<hipError_t (*)(dim3*, dim3*, size_t*, hipStream_t*)>(
          __hipPopCallConfiguration_ptr);
  hipError_t (*dyn_hipExtLaunchKernel_ptr)(const void*, dim3, dim3, void**, size_t, hipStream_t,
                                           hipEvent_t, hipEvent_t, int) =
      reinterpret_cast<hipError_t (*)(const void*, dim3, dim3, void**, size_t, hipStream_t,
                                      hipEvent_t, hipEvent_t, int)>(hipExtLaunchKernel_ptr);
  hipError_t (*dyn_hipDrvMemcpy2DUnaligned_ptr)(const hip_Memcpy2D*) =
      reinterpret_cast<hipError_t (*)(const hip_Memcpy2D*)>(hipDrvMemcpy2DUnaligned_ptr);

  hipError_t (*dyn_hipLaunchKernel_ptr)(const void*, dim3, dim3, void**, size_t, hipStream_t) =
      reinterpret_cast<hipError_t (*)(const void*, dim3, dim3, void**, size_t, hipStream_t)>(
          hipLaunchKernel_ptr);

  hipError_t (*dyn_hipLaunchHostFunc_ptr)(hipStream_t, hipHostFn_t, void*) =
      reinterpret_cast<hipError_t (*)(hipStream_t, hipHostFn_t, void*)>(hipLaunchHostFunc_ptr);

  // Validating hipConfigureCall, hipSetupArgument, hipLaunchByPtr API's
  {
    // hipConfigureCall
    dim3 blocksPerGrid(1, 1, 1);
    dim3 threadsPerBlock(1, 1, 1);

    HIP_CHECK(dyn_hipConfigureCall_ptr(blocksPerGrid, threadsPerBlock, 0, nullptr));

    // hipSetupArgument
    int hostInt = 10;
    int* devInt = nullptr;
    HIP_CHECK(hipMalloc(&devInt, sizeof(int)));
    REQUIRE(devInt != nullptr);
    HIP_CHECK(hipMemcpy(devInt, &hostInt, sizeof(int), hipMemcpyHostToDevice));

    HIP_CHECK(dyn_hipSetupArgument_ptr(&devInt, sizeof(int*), 0));

    // hipLaunchByPtr
    HIP_CHECK(dyn_hipLaunchByPtr_ptr(reinterpret_cast<void*>(addTenKernel)));

    HIP_CHECK(hipMemcpy(&hostInt, devInt, sizeof(int), hipMemcpyDeviceToHost));
    REQUIRE(hostInt == 20);

    HIP_CHECK(hipFree(devInt));
  }

  // Validating __hipPushCallConfiguration, __hipPopCallConfiguration APIs
  {
    // __hipPushCallConfiguration
    dim3 blocksPerGrid(4, 6, 8);
    dim3 threadsPerBlock(20, 30, 40);
    HIP_CHECK(dyn__hipPushCallConfiguration_ptr(blocksPerGrid, threadsPerBlock, 512, nullptr));

    // __hipPopCallConfiguration
    dim3 gridDim, blockDim;
    size_t sharedMem = -1;
    hipStream_t stream;

    HIP_CHECK(dyn__hipPopCallConfiguration_ptr(&gridDim, &blockDim, &sharedMem, &stream));

    REQUIRE(gridDim.x == blocksPerGrid.x);
    REQUIRE(gridDim.y == blocksPerGrid.y);
    REQUIRE(gridDim.z == blocksPerGrid.z);
    REQUIRE(blockDim.x == threadsPerBlock.x);
    REQUIRE(blockDim.y == threadsPerBlock.y);
    REQUIRE(blockDim.z == threadsPerBlock.z);
    REQUIRE(sharedMem == 512);
    REQUIRE(stream == nullptr);
  }

  // Validating hipExtLaunchKernel API
  {
    dim3 blocksPerGrid(1, 1, 1);
    dim3 threadsPerBlock(1, 1, 1);

    int hostInt = 10;
    int* devInt = nullptr;
    HIP_CHECK(hipMalloc(&devInt, sizeof(int)));
    REQUIRE(devInt != nullptr);
    HIP_CHECK(hipMemcpy(devInt, &hostInt, sizeof(int), hipMemcpyHostToDevice));

    void* kernel_args[] = {&devInt};
    HIP_CHECK(dyn_hipExtLaunchKernel_ptr(reinterpret_cast<void*>(addTenKernel), blocksPerGrid,
                                         threadsPerBlock, kernel_args, 0, nullptr, nullptr, nullptr,
                                         0));

    HIP_CHECK(hipMemcpy(&hostInt, devInt, sizeof(int), hipMemcpyDeviceToHost));
    REQUIRE(hostInt == 20);

    HIP_CHECK(hipFree(devInt));
  }

  // Validating hipDrvMemcpy2DUnaligned API
  {
    size_t width = 256;
    size_t height = 256;
    const int N = width * height;
    int value = 10;
    size_t pitch;

    // Host to Host
    {
      char* sHostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(sHostMem != nullptr);
      fillCharHostArray(sHostMem, N, value);

      char* dHostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(dHostMem != nullptr);

      hip_Memcpy2D desc = {};
      desc.srcMemoryType = hipMemoryTypeHost;
      desc.srcHost = sHostMem;
      desc.srcPitch = width;

      desc.dstMemoryType = hipMemoryTypeHost;
      desc.dstHost = dHostMem;
      desc.dstPitch = width;
      desc.WidthInBytes = width * sizeof(char);
      desc.Height = height;

      HIP_CHECK(dyn_hipDrvMemcpy2DUnaligned_ptr(&desc));

      REQUIRE(validateCharHostArray(dHostMem, N, value) == true);

      free(sHostMem);
      free(dHostMem);
    }

    // Host to Device
    {
      char* hostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(hostMem != nullptr);
      fillCharHostArray(hostMem, N, value);

      char* devMem = nullptr;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&devMem), &pitch, width, height));
      REQUIRE(devMem != nullptr);

      hip_Memcpy2D desc = {};
      desc.srcMemoryType = hipMemoryTypeHost;
      desc.srcHost = hostMem;
      desc.srcPitch = width;
      desc.dstMemoryType = hipMemoryTypeDevice;
      desc.dstDevice = hipDeviceptr_t(devMem);
      desc.dstPitch = pitch;
      desc.WidthInBytes = width * sizeof(char);
      desc.Height = height;

      HIP_CHECK(dyn_hipDrvMemcpy2DUnaligned_ptr(&desc));

      REQUIRE(validateCharDeviceArray(devMem, N, value) == true);

      free(hostMem);
      HIP_CHECK(hipFree(devMem));
    }

    // Device To Host
    {
      char* devMem = nullptr;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&devMem), &pitch, width, height));
      REQUIRE(devMem != nullptr);
      HIP_CHECK(hipMemset2D(devMem, pitch, value, width, height));

      char* hostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(hostMem != nullptr);

      hip_Memcpy2D desc = {};
      desc.srcMemoryType = hipMemoryTypeDevice;
      desc.srcDevice = (hipDeviceptr_t)devMem;
      desc.srcPitch = width;

      desc.dstMemoryType = hipMemoryTypeHost;
      desc.dstHost = hostMem;
      desc.dstPitch = width;
      desc.WidthInBytes = width * sizeof(char);
      desc.Height = height;

      HIP_CHECK(dyn_hipDrvMemcpy2DUnaligned_ptr(&desc));

      REQUIRE(validateCharHostArray(hostMem, N, value) == true);

      HIP_CHECK(hipFree(devMem));
      free(hostMem);
    }

    // Device To Device
    {
      char* sDevMem = nullptr;
      size_t sPitch;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&sDevMem), &sPitch, width, height));
      REQUIRE(sDevMem != nullptr);
      HIP_CHECK(hipMemset2D(sDevMem, sPitch, value, width, height));

      char* dDevMem = nullptr;
      size_t dPitch;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&dDevMem), &dPitch, width, height));
      REQUIRE(dDevMem != nullptr);

      hip_Memcpy2D desc = {};
      desc.srcMemoryType = hipMemoryTypeDevice;
      desc.srcDevice = hipDeviceptr_t(sDevMem);
      desc.srcPitch = sPitch;
      desc.dstMemoryType = hipMemoryTypeDevice;
      desc.dstDevice = hipDeviceptr_t(dDevMem);
      desc.dstPitch = dPitch;
      desc.WidthInBytes = width * sizeof(char);
      desc.Height = height;

      HIP_CHECK(dyn_hipDrvMemcpy2DUnaligned_ptr(&desc));

      REQUIRE(validateCharDeviceArray(dDevMem, N, value) == true);

      HIP_CHECK(hipFree(sDevMem));
      HIP_CHECK(hipFree(dDevMem));
    }
  }

  // Validating hipLaunchKernel API
  {
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

    struct kernelArgs {
      void* arr;
      int size;
    };
    kernelArgs kernelArg;
    kernelArg.arr = devArr;
    kernelArg.size = N;
    void* kernel_args[] = {&kernelArg.arr, &kernelArg.size};

    HIP_CHECK(dyn_hipLaunchKernel_ptr(reinterpret_cast<void*>(addOneKernel), blocksPerGrid,
                                      threadsPerBlock, kernel_args, 0, nullptr));

    HIP_CHECK(hipMemcpy(hostArr, devArr, Nbytes, hipMemcpyDeviceToHost));
    REQUIRE(validateHostArray(hostArr, N, 11) == true);

    free(hostArr);
    HIP_CHECK(hipFree(devArr));
  }

  // Validating hipLaunchHostFunc API
  {
    int data = 30;
    hipHostFn_t fn = addTen;

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    HIP_CHECK(dyn_hipLaunchHostFunc_ptr(stream, fn, reinterpret_cast<void*>(&data)));
    HIP_CHECK(hipStreamSynchronize(stream));

    REQUIRE(data == 40);
    HIP_CHECK(hipStreamDestroy(stream));
  }
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different
 *  - call back activity APIs from the hipGetProcAddress API
 *  - and then validates the basic functionality of that particular API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/executionControl/hipGetProcAddress_Launch_Callback_ExecCtrl_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_CallbackActivityAPIs") {
  void* hipGetStreamDeviceId_ptr = nullptr;
  void* hipApiName_ptr = nullptr;
  void* hipKernelNameRef_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress("hipGetStreamDeviceId", &hipGetStreamDeviceId_ptr, currentHipVersion,
                              0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipApiName", &hipApiName_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(
      hipGetProcAddress("hipKernelNameRef", &hipKernelNameRef_ptr, currentHipVersion, 0, nullptr));

  int (*dyn_hipGetStreamDeviceId_ptr)(hipStream_t) =
      reinterpret_cast<int (*)(hipStream_t)>(hipGetStreamDeviceId_ptr);
  const char* (*dyn_hipApiName_ptr)(uint32_t) =
      reinterpret_cast<const char* (*)(uint32_t)>(hipApiName_ptr);
  const char* (*dyn_hipKernelNameRef_ptr)(const hipFunction_t) =
      reinterpret_cast<const char* (*)(const hipFunction_t)>(hipKernelNameRef_ptr);

  // Validating hipGetStreamDeviceId API
  {
    int deviceId = 0;
    HIP_CHECK(hipSetDevice(deviceId));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    REQUIRE(hipGetStreamDeviceId(stream) == dyn_hipGetStreamDeviceId_ptr(stream));

    REQUIRE(dyn_hipGetStreamDeviceId_ptr(stream) == deviceId);

    HIP_CHECK(hipStreamDestroy(stream));
  }

  // Validating hipApiName API
  {
    uint32_t id = 1;
    REQUIRE(strcmp(hipApiName(id), dyn_hipApiName_ptr(id)) == 0);
  }

  // Validating hipKernelNameRef API
  {
    hipFunction_t funcPointer;
    REQUIRE(hipGetFuncBySymbol(&funcPointer, reinterpret_cast<const void*>(addTenKernel)) ==
            hipSuccess);

    REQUIRE(strcmp(hipKernelNameRef(funcPointer), dyn_hipKernelNameRef_ptr(funcPointer)) == 0);
  }
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different
 *  - Execution Control APIs from the hipGetProcAddress API
 *  - and then validates the basic functionality of that particular API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/executionControl/hipGetProcAddress_Launch_Callback_ExecCtrl_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_ExecutionControlAPIs") {
  void* hipFuncSetAttribute_ptr = nullptr;
  void* hipFuncSetCacheConfig_ptr = nullptr;
  void* hipFuncSetSharedMemConfig_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress("hipFuncSetAttribute", &hipFuncSetAttribute_ptr, currentHipVersion, 0,
                              nullptr));
  HIP_CHECK(hipGetProcAddress("hipFuncSetCacheConfig", &hipFuncSetCacheConfig_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipFuncSetSharedMemConfig", &hipFuncSetSharedMemConfig_ptr,
                              currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipFuncSetAttribute_ptr)(const void*, hipFuncAttribute, int) =
      reinterpret_cast<hipError_t (*)(const void*, hipFuncAttribute, int)>(hipFuncSetAttribute_ptr);

  hipError_t (*dyn_hipFuncSetCacheConfig_ptr)(const void*, hipFuncCache_t) =
      reinterpret_cast<hipError_t (*)(const void*, hipFuncCache_t)>(hipFuncSetCacheConfig_ptr);

  hipError_t (*dyn_hipFuncSetSharedMemConfig_ptr)(const void*, hipSharedMemConfig) =
      reinterpret_cast<hipError_t (*)(const void*, hipSharedMemConfig)>(
          hipFuncSetSharedMemConfig_ptr);

  // Validating hipFuncSetAttribute API
  hipFuncAttribute funcAttributes[] = {hipFuncAttributeMaxDynamicSharedMemorySize,
                                       hipFuncAttributePreferredSharedMemoryCarveout,
                                       hipFuncAttributeMax};
  int value = 90;
  for (auto funcAttribute : funcAttributes) {
    HIP_CHECK(
        dyn_hipFuncSetAttribute_ptr(reinterpret_cast<void*>(addOneKernel), funcAttribute, value));
  }

  // Validating hipFuncSetCacheConfig API
  hipFuncCache_t funcCacheConfig[] = {hipFuncCachePreferNone, hipFuncCachePreferShared,
                                      hipFuncCachePreferL1, hipFuncCachePreferEqual};
  for (auto config : funcCacheConfig) {
    HIP_CHECK(dyn_hipFuncSetCacheConfig_ptr(reinterpret_cast<void*>(addOneKernel), config));
  }

  // Validating hipFuncSetSharedMemConfig API
  hipSharedMemConfig sharedMemConfig[] = {hipSharedMemBankSizeDefault, hipSharedMemBankSizeFourByte,
                                          hipSharedMemBankSizeEightByte};
  for (auto config : sharedMemConfig) {
    HIP_CHECK(dyn_hipFuncSetSharedMemConfig_ptr(reinterpret_cast<void*>(addOneKernel), config));
  }
}
