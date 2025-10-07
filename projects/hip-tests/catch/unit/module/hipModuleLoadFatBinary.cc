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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include "hip_module_common.hh"
#include <array>
#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
/**
 * @addtogroup hipModuleLoadFatBinary hipModuleLoadFatBinary
 * @{
 * @ingroup ModuleTest
 * `hipError_t hipModuleLoadFatBinary(hipModule_t* module, const void* fatbin)`
 * - Loads fatbin object
 */

/**
 * Test Description
 * ------------------------
 * - Test case verifies the negative cases of hipModuleLoadFatBinary API.
 * Test source
 * ------------------------
 * - catch/unit/module/hipModuleLoadFatBinary.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipModuleLoadFatBinary_NegativeTsts") {
  hipModule_t Module;
  SECTION("fatCubin as nullptr") {
    HIP_CHECK_ERROR(hipModuleLoadFatBinary(&Module, nullptr),
                    hipErrorInvalidValue);
  }
  SECTION("Load Module with No Kernel function") {
    const auto loaded_module = LoadModuleIntoBuffer("emptyModuleCount.code");
    HIP_CHECK(hipModuleLoadFatBinary(&Module, loaded_module.data()));
    REQUIRE(Module != nullptr);
    HIP_CHECK(hipModuleUnload(Module));
  }
}
#if HT_AMD
void loadKernelData(hipFunction_t kernel) {
  constexpr int LEN = 64;
  constexpr int SIZE = LEN * sizeof(float);
  float *Ad, *Bd;

  std::array<float, LEN> A;
  std::array<float, LEN> B;

  for (uint32_t i = 0; i < LEN; i++) {
    A.fill(i * 1.0f);
    B.fill(0.0f);
  }
  HIP_CHECK(hipMalloc(&Ad, SIZE));
  HIP_CHECK(hipMalloc(&Bd, SIZE));

  HIP_CHECK(hipMemcpy(Ad, A.data(), SIZE, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(Bd, B.data(), SIZE, hipMemcpyHostToDevice));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  struct {
    void *_Ad;
    void *_Bd;
    size_t size;
  } args;
  args._Ad = reinterpret_cast<void *>(Ad);
  args._Bd = reinterpret_cast<void *>(Bd);
  args.size = LEN;
  size_t size = sizeof(args);

  void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, HIP_LAUNCH_PARAM_END};
  HIP_CHECK(hipModuleLaunchKernel(kernel, 1, 1, 1, LEN, 1, 1, 0, stream, NULL,
                                  reinterpret_cast<void **>(&config)));

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipMemcpy(B.data(), Bd, SIZE, hipMemcpyDeviceToHost));
  // Validation
  for (size_t i = 0; i < A.size(); i++) {
    INFO("Array Failed at index: " << i
                                   << "\nA value at failed index: " << A[i]
                                   << "\nB value at failed index: " << B[i]);
    REQUIRE(A[i] == B[i]);
  }

  HIP_CHECK(hipFree(Ad));
  HIP_CHECK(hipFree(Bd));
}
/**
 * Test Description
 * ------------------------
 * - Test case verifies the below positive cases of hipModuleLoadFatBinary API.
 * case-1 : Loads Compiled module with regular target in compressed fatbin
 * case-2 : Loads Compiled module with Generic target in regular fatbin
 * case-3 : Loads Compiled module with Generic target in compressed fatbin
 * Test source
 * ------------------------
 * - catch/unit/module/hipModuleLoadFatBinary.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipModuleLoadFatBinary_PosiiveTsts") {
  hipModule_t Module;
  SECTION("Compiled module with regular target in compressed fatbin") {
    const auto loaded_module =
        LoadModuleIntoBuffer("copyKernelCompressed.code");
    HIP_CHECK(hipModuleLoadFatBinary(&Module, loaded_module.data()));
    REQUIRE(Module != nullptr);
    hipFunction_t kernel = nullptr;
    HIP_CHECK(hipModuleGetFunction(&kernel, Module, "copy_ker"));
    loadKernelData(kernel);
    REQUIRE(kernel != nullptr);
    HIP_CHECK(hipModuleUnload(Module));
  }
  if (isGenericTargetSupported()) {
    SECTION("Compiled module with Generic target in regular fatbin") {
      const auto loaded_module =
          LoadModuleIntoBuffer("copyKernelGenericTarget.code");
      HIP_CHECK(hipModuleLoadFatBinary(&Module, loaded_module.data()));
      REQUIRE(Module != nullptr);
      hipFunction_t kernel = nullptr;
      HIP_CHECK(hipModuleGetFunction(&kernel, Module, "copy_ker"));
      REQUIRE(kernel != nullptr);
      loadKernelData(kernel);
      HIP_CHECK(hipModuleUnload(Module));
    }

    SECTION("Compiled module with Generic target in compressed fatbin") {
      const auto loaded_module =
          LoadModuleIntoBuffer("copyKernelGenericTargetCompressed.code");
      HIP_CHECK(hipModuleLoadFatBinary(&Module, loaded_module.data()));
      REQUIRE(Module != nullptr);
      hipFunction_t kernel = nullptr;
      HIP_CHECK(hipModuleGetFunction(&kernel, Module, "copy_ker"));
      REQUIRE(kernel != nullptr);
      loadKernelData(kernel);
      HIP_CHECK(hipModuleUnload(Module));
    }
  }
}
#endif
/**
 * End doxygen group ModuleTest.
 * @}
 */
