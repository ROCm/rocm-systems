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

#include "hip_module_common.hh"

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <hip_test_defgroups.hh>
#include <fstream>
#include <vector>

TEST_CASE("Unit_hipModuleLoadData_Positive_Basic") {
  HIP_CHECK(hipFree(nullptr));
  hipModule_t module = nullptr;

  SECTION("Load compiled module from file") {
    const auto loaded_module = LoadModuleIntoBuffer("empty_module.code");
    HIP_CHECK(hipModuleLoadData(&module, loaded_module.data()));
    REQUIRE(module != nullptr);
    HIP_CHECK(hipModuleUnload(module));
  }

#if HT_AMD
  SECTION("Load compiled module from file with regular target in compressed fatbin") {
    const auto loaded_module = LoadModuleIntoBuffer("copyKernelCompressed.code");
    HIP_CHECK(hipModuleLoadData(&module, loaded_module.data()));
    REQUIRE(module != nullptr);
    hipFunction_t kernel = nullptr;
    HIP_CHECK(hipModuleGetFunction(&kernel, module, "copy_ker"));
    REQUIRE(kernel != nullptr);
    HIP_CHECK(hipModuleUnload(module));
  }

  SECTION("Load compiled module from file with generic target in regular fatbin") {
    if (!isGenericTargetSupported()) {
      fprintf(stderr, "Generic target test is skipped\n");
      return;
    }
    const auto loaded_module = LoadModuleIntoBuffer("copyKernelGenericTarget.code");
    HIP_CHECK(hipModuleLoadData(&module, loaded_module.data()));
    REQUIRE(module != nullptr);
    hipFunction_t kernel = nullptr;
    HIP_CHECK(hipModuleGetFunction(&kernel, module, "copy_ker"));
    REQUIRE(kernel != nullptr);
    HIP_CHECK(hipModuleUnload(module));
  }

  SECTION("Load compiled module from file with generic target in compressed fatbin") {
    if (!isGenericTargetSupported()) {
      fprintf(stderr, "Generic target test is skipped\n");
      return;
    }
    const auto loaded_module = LoadModuleIntoBuffer("copyKernelGenericTargetCompressed.code");
    HIP_CHECK(hipModuleLoadData(&module, loaded_module.data()));
    REQUIRE(module != nullptr);
    hipFunction_t kernel = nullptr;
    HIP_CHECK(hipModuleGetFunction(&kernel, module, "copy_ker"));
    REQUIRE(kernel != nullptr);
    HIP_CHECK(hipModuleUnload(module));
  }
#endif

  SECTION("Load RTCd module") {
    const auto rtc = CreateRTCCharArray(R"(extern "C" __global__ void kernel() {})");
    HIP_CHECK(hipModuleLoadData(&module, rtc.data()));
    REQUIRE(module != nullptr);
    HIP_CHECK(hipModuleUnload(module));
  }
}

TEST_CASE("Unit_hipModuleLoadData_Negative_Parameters") {
  HIP_CHECK(hipFree(nullptr));
  hipModule_t module;

  SECTION("module == nullptr") {
    const auto loaded_module = LoadModuleIntoBuffer("empty_module.code");
    HIP_CHECK_ERROR(hipModuleLoadData(nullptr, loaded_module.data()), hipErrorInvalidValue);
    LoadModuleIntoBuffer("empty_module.code");
  }

  SECTION("image == nullptr") {
    HIP_CHECK_ERROR(hipModuleLoadData(&module, nullptr), hipErrorInvalidValue);
  }
}

TEST_CASE("Unit_hipModuleLoadData_Negative_Image_Is_An_Empty_String") {
  HIP_CHECK(hipFree(nullptr));
  hipModule_t module;

  HIP_CHECK_ERROR(hipModuleLoadData(&module, ""), hipErrorInvalidImage);
}

/**
 * @addtogroup hipModuleLoad hipModuleGetFunction
 * @{
 * @ingroup ModuleTest
 * `hipError_t hipModuleLoad(hipModule_t* module, const char* fname)` -
 * Loads code object from file into a module
 * `hipError_t hipModuleGetFunction(hipFunction_t* function, hipModule_t module, const char* kname)`
 * - Function with kname will be extracted if present in module
 */

/**
 * Test Description
 * ------------------------
 * - Test case to load data from a code object file through hipModuleLoad and hipModuleGetFunction.

 * Test source
 * ------------------------
 * - catch/unit/module/hipModuleLoadData.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 5.6
*/
#if HT_AMD
// Below test disabled for NVIDIA due to the defect SWDEV-472385
TEST_CASE("Unit_hipModuleLoadData_Functional") {
  constexpr int LEN = 64;
  constexpr int SIZE = LEN << 2;
  constexpr auto FILENAME = "vcpy_kernel.code";
  constexpr auto kernel_name = "hello_world";
  float *A, *B, *Ad, *Bd;
  A = new float[LEN];
  B = new float[LEN];

  for (uint32_t i = 0; i < LEN; i++) {
    A[i] = i * 1.0f;
    B[i] = 0.0f;
  }

  HIP_CHECK(hipMalloc(&Ad, SIZE));
  HIP_CHECK(hipMalloc(&Bd, SIZE));

  HIP_CHECK(hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(Bd, B, SIZE, hipMemcpyHostToDevice));

  hipModule_t Module;
  hipFunction_t Function = nullptr;
  std::ifstream file(FILENAME, std::ios::binary | std::ios::ate);
  std::streamsize fsize = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(fsize);
  if (file.read(buffer.data(), fsize)) {
    HIP_CHECK(hipModuleLoadData(&Module, &buffer[0]));
    HIP_CHECK(hipModuleGetFunction(&Function, Module, kernel_name));
  }
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  struct {
    void* _Ad;
    void* _Bd;
  } args;
  args._Ad = reinterpret_cast<void*>(Ad);
  args._Bd = reinterpret_cast<void*>(Bd);
  size_t size = sizeof(args);

  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                    HIP_LAUNCH_PARAM_END};
  HIP_CHECK(hipModuleLaunchKernel(Function, 1, 1, 1, LEN, 1, 1, 0, stream, NULL,
                                  reinterpret_cast<void**>(&config)));

  HIP_CHECK(hipStreamDestroy(stream));

  HIP_CHECK(hipMemcpy(B, Bd, SIZE, hipMemcpyDeviceToHost));

  for (uint32_t i = 0; i < LEN; i++) {
    REQUIRE(A[i] == B[i]);
  }
  delete[] A;
  delete[] B;
  HIP_CHECK(hipModuleUnload(Module));
  HIP_CHECK(hipFree(Ad));
  HIP_CHECK(hipFree(Bd));
}
#endif
