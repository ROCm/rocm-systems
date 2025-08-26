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
#include <hip_test_defgroups.hh>

constexpr int MANAGED_VAR_INIT_VALUE = 10;
constexpr auto fileName = "managed_kernel.code";

/**
 * @addtogroup hipModuleGetGlobal
 * @{
 * @ingroup ModuleTest
 * `hipError_t hipModuleGetGlobal(hipDeviceptr_t* dptr, size_t* bytes, hipModule_t hmod, const char*
 * name)` - Returns a global pointer from a module
 */

/**
 * Test Description
 * ------------------------
 * - Test case to verify global pointer from a module for multiGPU's.

 * Test source
 * ------------------------
 * - catch/unit/module/hipManagedKeyword.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 5.6
*/

TEST_CASE("Unit_hipModuleGetGlobal_Functional") {
  bool testStatus = true;
  int numDevices = 0;
  hipDeviceptr_t x;
  size_t xSize;
  int data;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  for (int i = 0; i < numDevices; i++) {
    int managed_memory = 0;
    HIPCHECK(hipDeviceGetAttribute(&managed_memory, hipDeviceAttributeManagedMemory, i));
    if (!managed_memory) {
      HipTest::HIP_SKIP_TEST("managed memory access not supported on device");
      return;
    }
  }
  for (int i = 0; i < numDevices; i++) {
    HIP_CHECK(hipSetDevice(i));
    hipDevice_t device;
    hipCtx_t context;
    HIP_CHECK(hipDeviceGet(&device, i));
    HIP_CHECK(hipCtxCreate(&context, 0, device));
    hipModule_t Module;
    HIP_CHECK(hipModuleLoad(&Module, fileName));
    hipFunction_t Function;
    HIP_CHECK(hipModuleGetFunction(&Function, Module, "GPU_func"));
    HIP_CHECK(hipModuleLaunchKernel(Function, 1, 1, 1, 1, 1, 1, 0, 0, NULL, NULL));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipModuleGetGlobal(reinterpret_cast<hipDeviceptr_t*>(&x), &xSize, Module, "x"));
    HIP_CHECK(hipMemcpyDtoH(&data, hipDeviceptr_t(x), xSize));
    if (data != (1 + MANAGED_VAR_INIT_VALUE)) {
      HIP_CHECK(hipModuleUnload(Module));
      HIP_CHECK(hipCtxDestroy(context));
      testStatus = false;
    }
    HIP_CHECK(hipModuleUnload(Module));
    HIP_CHECK(hipCtxDestroy(context));
  }
  REQUIRE(testStatus == true);
}
