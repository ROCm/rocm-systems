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

#include <hip/hip_runtime_api.h>
#include <hip_test_checkers.hh>
#include <hip_test_common.hh>
#include <hip_test_kernels.hh>
#include "hip_module_common.hh"

TEST_CASE("Unit_hipModuleLoad_Positive_Basic") {
  HIP_CHECK(hipFree(nullptr));
  hipModule_t module = nullptr;
  HIP_CHECK(hipModuleLoad(&module, "empty_module.code"));
  REQUIRE(module != nullptr);
  HIP_CHECK(hipModuleUnload(module));
}

TEST_CASE("Unit_hipModuleLoad_Negative_Parameters") {
  HIP_CHECK(hipFree(nullptr));
  hipModule_t module;

  SECTION("module == nullptr") {
    HIP_CHECK_ERROR(hipModuleLoad(nullptr, "empty_module.code"), hipErrorInvalidValue);
  }

  SECTION("fname == nullptr") {
    HIP_CHECK_ERROR(hipModuleLoad(&module, nullptr), hipErrorInvalidValue);
  }

  SECTION("fname == empty string") {
    HIP_CHECK_ERROR(hipModuleLoad(&module, ""), hipErrorInvalidValue);
  }

  SECTION("fname == non existent file") {
    HIP_CHECK_ERROR(hipModuleLoad(&module, "non existent file"), hipErrorFileNotFound);
  }
}

TEST_CASE("Unit_hipModuleLoad_Negative_Load_From_A_File_That_Is_Not_A_Module") {
  HIP_CHECK(hipFree(nullptr));
  hipModule_t module;

  HIP_CHECK_ERROR(hipModuleLoad(&module, "not_a_module.txt"), hipErrorInvalidImage);
}

/**
 * Test Description
 * ------------------------
 *  - This test case tests the behaviour of hipModuleLoad, hipModuleGetFunction,
 *  - hipModuleGetGlobal, hipModuleGetTexRef API's during the the stream capture
 * Test source
 * ------------------------
 *  - unit/module/hipModuleLoad.cc
 */
TEST_CASE("Unit_ModuleAPIs_StreamCapture_ModuleLoadAndGetFunctions") {
  GENERATE_CAPTURE();

  constexpr int size = 10;
  constexpr int Nbytes = size * sizeof(int);

  std::vector<int> hostArr(size);
  ::std::fill(hostArr.begin(), hostArr.end(), 5);

  int* devArr = nullptr;
  HIP_CHECK(hipMalloc(&devArr, Nbytes));
  REQUIRE(devArr != nullptr);
  HIP_CHECK(hipMemcpy(devArr, hostArr.data(), Nbytes, hipMemcpyHostToDevice));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  BEGIN_CAPTURE(stream);

  // hipModuleLoad
  hipModule_t module;
  HIP_CHECK(hipModuleLoad(&module, "addKernel.code"));
  REQUIRE(module != nullptr);

  // hipModuleGetFunction
  hipFunction_t function;
  HIP_CHECK(hipModuleGetFunction(&function, module, "addKernel"));
  REQUIRE(function != nullptr);

  struct kernelParameters {
    void* arr;
    int size_;
  };
  kernelParameters kernelParam{};
  kernelParam.arr = devArr;
  kernelParam.size_ = size;

  auto kernelParamSize = sizeof(kernelParam);
  void* kernel_parameter[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &kernelParam,
                              HIP_LAUNCH_PARAM_BUFFER_SIZE, &kernelParamSize, HIP_LAUNCH_PARAM_END};

  HIP_CHECK(
      hipModuleLaunchKernel(function, 1, 1, 1, size, 1, 1, 0, stream, nullptr, kernel_parameter));

  // hipModuleGetGlobal
  hipDeviceptr_t devPtrForGlobalDevData;
  size_t sizeOfGlobalDevData = 0;

  HIP_CHECK(
      hipModuleGetGlobal(&devPtrForGlobalDevData, &sizeOfGlobalDevData, module, "globalDevData"));

// hipModuleGetTexRef
#if defined(__HIP_PLATFORM_AMD__) || CUDA_VERSION < CUDA_12000
  hipTexRef texRef = nullptr;
  HIP_CHECK(hipModuleGetTexRef(&texRef, module, "tex"));
  REQUIRE(texRef != nullptr);
#endif

  END_CAPTURE(stream);
  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipMemcpy(hostArr.data(), devArr, Nbytes, hipMemcpyDeviceToHost));
  for (int i = 0; i < size; i++) {
    REQUIRE(hostArr[i] == 7);
  }

  int localHostForGlobalDevData = 0;
  HIP_CHECK(hipMemcpy(&localHostForGlobalDevData,
                      reinterpret_cast<const void*>(devPtrForGlobalDevData), sizeof(int),
                      hipMemcpyDeviceToHost));
  REQUIRE(localHostForGlobalDevData == 10);
  REQUIRE(sizeOfGlobalDevData == 4);

  HIP_CHECK(hipModuleUnload(module));
  HIP_CHECK(hipFree(devArr));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 *  - This test case tests the behaviour of hipModuleUnload API
 *  - during the the stream capture.
 * Test source
 * ------------------------
 *  - unit/module/hipModuleLoad.cc
 */
TEST_CASE("Unit_ModuleAPIs_StreamCapture_Unload") {
  hipModule_t module;
  HIP_CHECK(hipModuleLoad(&module, "addKernel.code"));
  REQUIRE(module != nullptr);

  hipError_t err = hipSuccess;
  BEGIN_CAPTURE_SYNC(err, true);
  // hipModuleUnload
  HIP_CHECK_ERROR(hipModuleUnload(module), err);
  END_CAPTURE_SYNC(err);

  if (err != hipSuccess) {
    HIP_CHECK(hipModuleUnload(module));
  }
}

/**
 * Test Description
 * ------------------------
 *  - This test case tests the behaviour of hipFuncGetAttributes,
 *  - hipGetFuncBySymbol, hipFuncGetAttribute, hipModuleLoadData,
 *  - hipModuleLoadDataEx API's during the the stream capture
 * Test source
 * ------------------------
 *  - unit/module/hipModuleLoad.cc
 */
TEST_CASE("Unit_ModuleAPIs_StreamCapture_GetAttrGetFuncBySymbolLoadData") {
  constexpr auto N = 10;
  constexpr size_t Nbytes = N * sizeof(int);

  int *A_d{nullptr}, *B_d{nullptr}, *C_d{nullptr};
  int *A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr};

  HipTest::initArrays<int>(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  HIP_CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_d, B_h, Nbytes, hipMemcpyHostToDevice));

  GENERATE_CAPTURE();

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  BEGIN_CAPTURE(stream);

  // hipFuncGetAttributes
  hipFuncAttributes attributes{};
  HIP_CHECK(
      hipFuncGetAttributes(&attributes, reinterpret_cast<const void*>(HipTest::vectorADD<int>)));

  REQUIRE(attributes.binaryVersion > 0);
  REQUIRE(attributes.maxThreadsPerBlock > 0);

  // hipGetFuncBySymbol
  hipFunction_t function;
  HIP_CHECK(hipGetFuncBySymbol(&function, reinterpret_cast<const void*>(HipTest::vectorADD<int>)));
  REQUIRE(function != nullptr);

  struct kernelParameters {
    void* a;
    void* b;
    void* c;
    int size;
  };
  kernelParameters kernelParam{};
  kernelParam.a = A_d;
  kernelParam.b = B_d;
  kernelParam.c = C_d;
  kernelParam.size = N;

  auto kernelParamSize = sizeof(kernelParam);
  void* kernel_parameter[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &kernelParam,
                              HIP_LAUNCH_PARAM_BUFFER_SIZE, &kernelParamSize, HIP_LAUNCH_PARAM_END};

  HIP_CHECK(
      hipModuleLaunchKernel(function, 1, 1, 1, N, 1, 1, 0, stream, nullptr, kernel_parameter));

  // hipFuncGetAttribute
  int binaryVersion = 0, maxThreadsPerBlock = 0;
  HIP_CHECK(hipFuncGetAttribute(&binaryVersion, HIP_FUNC_ATTRIBUTE_BINARY_VERSION, function));
  HIP_CHECK(
      hipFuncGetAttribute(&maxThreadsPerBlock, HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function));

  REQUIRE(binaryVersion > 0);
  REQUIRE(maxThreadsPerBlock > 0);

  const auto rtc = CreateRTCCharArray(R"(extern "C" __global__ void simpleKernel() {})");

  // hipModuleLoadData
  hipModule_t moduleForData = nullptr;

  HIP_CHECK(hipModuleLoadData(&moduleForData, rtc.data()));
  REQUIRE(moduleForData != nullptr);

  hipFunction_t function1;
  HIP_CHECK(hipModuleGetFunction(&function1, moduleForData, "simpleKernel"));
  REQUIRE(function1 != nullptr);
  HIP_CHECK(hipModuleLaunchKernel(function1, 1, 1, 1, 1, 1, 1, 0, stream, nullptr, nullptr));

  // hipModuleLoadDataEx
  hipModule_t moduleForDataEx = nullptr;

  HIP_CHECK(hipModuleLoadDataEx(&moduleForDataEx, rtc.data(), 0, nullptr, nullptr));
  REQUIRE(moduleForDataEx != nullptr);

  hipFunction_t function2;
  HIP_CHECK(hipModuleGetFunction(&function2, moduleForDataEx, "simpleKernel"));
  REQUIRE(function2 != nullptr);
  HIP_CHECK(hipModuleLaunchKernel(function2, 1, 1, 1, 1, 1, 1, 0, stream, nullptr, nullptr));

  END_CAPTURE(stream);

  HIP_CHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));
  HipTest::checkVectorADD<int>(A_h, B_h, C_h, N);
  HipTest::freeArrays<int>(A_d, B_d, C_d, A_h, B_h, C_h, false);

  HIP_CHECK(hipModuleUnload(moduleForData));
  HIP_CHECK(hipModuleUnload(moduleForDataEx));
  HIP_CHECK(hipStreamDestroy(stream));
}
