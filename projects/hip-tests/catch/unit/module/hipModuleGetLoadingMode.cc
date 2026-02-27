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
#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#ifdef __linux__
#include <unistd.h>
#endif
#ifdef _WIN64
#include <windows.h>
#endif

#ifdef _WIN64
#define setenv(x, y, z) _putenv_s(x, y)
#define unsetenv(x) _putenv(x)
#endif

constexpr int LEN = 64;
constexpr auto SIZE_BYTES = (LEN << 2);
constexpr auto codeObjFile = "copyKernel.code";
constexpr auto kernel_name = "copy_ker";
/**
 * @addtogroup hipModuleGetLoadingMode hipModuleGetLoadingMode
 * @{
 * @ingroup ModuleTest
 * `hhipError_t hipModuleGetLoadingMode(hipModuleLoadingMode_t* mode)` -
 * Function gets the current module load mode
 */

/**
 * Test Description
 * ------------------------
 * - Test case verifies the positive case of hipModuleGetLoadingMode API.
 * - Verifies the default mode. Default mode is Lazy.
 * Test source
 * ------------------------
 * - catch/unit/module/hipModuleGetLoadingMode.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 7.2
 */
TEST_CASE("Unit_hipModuleGetLoadingMode_DefaultModeCheck") {
  hipModuleLoadingMode_t mode;
  HIP_CHECK(hipModuleGetLoadingMode(&mode));
  REQUIRE(mode == HIP_MODULE_LAZY_LOADING);
}
/**
 * Test Description
 * ------------------------
 * - Test case verifies the positive case of hipModuleGetLoadingMode API.
 * - Verifies the Eager mode after setting the env var HIP_MODULE_LOADING.
 * Test source
 * ------------------------
 * - catch/unit/module/hipModuleGetLoadingMode.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 7.2
 */
TEST_CASE("Unit_hipModuleGetLoadingMode_EagerModeCheck") {
  hipModuleLoadingMode_t mode;
  if (setenv("HIP_MODULE_LOADING", "EAGER", 1) != 0) {
    HIP_CHECK(hipModuleGetLoadingMode(&mode));
    REQUIRE(mode == HIP_MODULE_LAZY_LOADING); // Default mode is lazy mode
  } else {
    HIP_CHECK(hipModuleGetLoadingMode(&mode));
    REQUIRE(mode ==
            HIP_MODULE_EAGER_LOADING); // if env var HIP_MODULE_LOADING = EAGER
  }
  unsetenv("HIP_MODULE_LOADING");
}

void kernelExecutionFunction(hipModule_t module) {
  float *Ad, *Bd;
  std::vector<float> A(LEN, 1.0f), B(LEN, 0.0f);
  HIP_CHECK(hipMalloc(&Ad, SIZE_BYTES));
  HIP_CHECK(hipMalloc(&Bd, SIZE_BYTES));
  HIP_CHECK(hipMemcpy(Ad, A.data(), SIZE_BYTES, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(Bd, B.data(), SIZE_BYTES, hipMemcpyHostToDevice));
  hipFunction_t Function = nullptr;
  HIP_CHECK(hipModuleGetFunction(&Function, module, kernel_name));
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  struct {
    void *_Ad;
    void *_Bd;
  } args;
  args._Ad = reinterpret_cast<void *>(Ad);
  args._Bd = reinterpret_cast<void *>(Bd);
  size_t size = sizeof(args);

  void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, HIP_LAUNCH_PARAM_END};
  HIP_CHECK(hipModuleLaunchKernel(Function, 1, 1, 1, LEN, 1, 1, 0, stream, NULL,
                                  reinterpret_cast<void **>(&config)));
  HIP_CHECK(hipStreamDestroy(stream));

  HIP_CHECK(hipMemcpy(B.data(), Bd, SIZE_BYTES, hipMemcpyDeviceToHost));

  for (uint32_t i = 0; i < LEN; i++) {
    REQUIRE(A[i] != B[i]);
  }
  HIP_CHECK(hipFree(Bd));
  HIP_CHECK(hipFree(Ad));
}
/**
 * Test Description
 * ------------------------
 * - Test case verifies the positive case with kernel Launch.
 * - Calcualtes the time taken for module load if the mode is Eager.
 * Test source
 * ------------------------
 * - catch/unit/module/hipModuleGetLoadingMode.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 7.2
 */
TEST_CASE("Unit_hipModuleGetLoadingMode_EagerModeKernel") {
  hipModuleLoadingMode_t mode;
  hipModule_t module;
  if (setenv("HIP_MODULE_LOADING", "EAGER", 1) == 0) {
    HIP_CHECK(hipModuleGetLoadingMode(&mode));
    REQUIRE(mode == HIP_MODULE_EAGER_LOADING);
    auto start = std::chrono::high_resolution_clock::now();
    HIP_CHECK(hipModuleLoad(&module, codeObjFile));
    auto stop = std::chrono::high_resolution_clock::now();
    auto result = std::chrono::duration<double, std::milli>(stop - start);
    std::cout << "Time taken for Eager mode:" << result.count()
              << " milliSeconds" << std::endl;
    kernelExecutionFunction(module);
    HIP_CHECK(hipModuleUnload(module));
  }
  unsetenv("HIP_MODULE_LOADING");
}
/**
 * Test Description
 * ------------------------
 * - Test case verifies the positive case with kernel Launch.
 * - Calcualtes the time taken for module load if the mode is Lazy.
 * Test source
 * ------------------------
 * - catch/unit/module/hipModuleGetLoadingMode.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 7.2
 */
TEST_CASE("Unit_hipModuleGetLoadingMode_LazyModeKernel") {
  hipModuleLoadingMode_t mode;
  hipModule_t module;
  HIP_CHECK(hipModuleGetLoadingMode(&mode));
  REQUIRE(mode == HIP_MODULE_LAZY_LOADING);
  auto start = std::chrono::high_resolution_clock::now();
  HIP_CHECK(hipModuleLoad(&module, codeObjFile));
  auto stop = std::chrono::high_resolution_clock::now();
  auto result = std::chrono::duration<double, std::milli>(stop - start);
  std::cout << "Time taken for Lazy mode:" << result.count() << " milliSeconds"
            << std::endl;
  kernelExecutionFunction(module);
  HIP_CHECK(hipModuleUnload(module));
}
void setMode() { setenv("HIP_MODULE_LOADING", "EAGER", 1); }

void ChkMode() {
  hipModuleLoadingMode_t mode;
  HIP_CHECK(hipModuleGetLoadingMode(&mode));
  REQUIRE(mode == HIP_MODULE_EAGER_LOADING);
  unsetenv("HIP_MODULE_LOADING");
}
/**
 * Test Description
 * ------------------------
 * - Test case verifies the multithread Scenario.
 * - Set mode Env in one thread. Verify mode in another thread.
 * Test source
 * ------------------------
 * - catch/unit/module/hipModuleGetLoadingMode.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 7.2
 */
TEST_CASE("Unit_hipModuleGetLoadingMode_MultiThread") {
  // Create Thraed one.
  std::thread t1(setMode);
  t1.join();
  // Create Thread two
  std::thread t2(ChkMode);
  t2.join();
}
/**
 * End doxygen group ModuleTest.
 * @}
 */
