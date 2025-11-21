/*Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

/**
 * @addtogroup hipHccModuleLaunchKernel hipHccModuleLaunchKernel
 * @{
 * @ingroup KernelTest
 * `HIP_PUBLIC_API hipError_t hipHccModuleLaunchKernel (hipFunction_t f,
 *  uint32_t globalWorkSizeX,
 *  uint32_t globalWorkSizeY,
 *  uint32_t globalWorkSizeZ,
 *  uint32_t localWorkSizeX,
 *  uint32_t localWorkSizeY,
 *  uint32_t localWorkSizeZ,
 *  size_t sharedMemBytes,
 *  hipStream_t hStream,
 *  void ** kernelParams,
 *  void ** extra,
 *  hipEvent_t startEvent,
 *  hipEvent_t stopEvent )` -
 * This HIP API is deprecated, please use hipExtModuleLaunchKernel() instead.
 * Launches kernel with parameters and shared memory on stream with arguments
 * passed to kernel params or extra arguments.
 */

#include <hip_test_common.hh>
#include "hip/hip_ext.h"

#define fileName "copyKernel.code"
#define kernel_name "copy_ker"

/**
 * Test Description
 * ------------------------
 *    - Basic functional testcase to verify kernel launch
 * Test source
 * ------------------------
 *    - catch\unit\module\hipHccModuleLaunchKernel.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.7
 */
TEST_CASE("Unit_hipHccModuleLaunchKernel_basic") {
  size_t width = GENERATE(3, 4, 100);
  size_t widthInBytes = width * sizeof(int);
  int *A_d, *B_d;
  int* A_h = reinterpret_cast<int*>(malloc(widthInBytes));
  int* B_h = reinterpret_cast<int*>(malloc(widthInBytes));
  for (int i = 0; i < width; i++) {
    A_h[i] = i;
  }
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&A_d), widthInBytes));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&B_d), widthInBytes));
  void* kernelArgs[3] = {&A_d, &B_d, &widthInBytes};
  HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)A_d, A_h, widthInBytes));
  hipModule_t module;
  HIP_CHECK(hipModuleLoad(&module, fileName));
  hipFunction_t kernelFunc;
  HIP_CHECK(hipModuleGetFunction(&kernelFunc, module, kernel_name));

  HIP_CHECK(hipHccModuleLaunchKernel(kernelFunc, width, 1, 1, width, 1, 1, 0, 0, kernelArgs,
                                     nullptr, nullptr, nullptr));
  HIP_CHECK(hipMemcpyDtoH(B_h, (hipDeviceptr_t)B_d, widthInBytes));
  for (int i = 0; i < width; i++) {
    REQUIRE(A_h[i] == B_h[i]);
  }
  HIP_CHECK(hipModuleUnload(module));
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(B_d));
  free(A_h);
  free(B_h);
}

/**
 * Test Description
 * ------------------------
 *    - Negative testcases
 * Test source
 * ------------------------
 *    - catch\unit\module\hipHccModuleLaunchKernel.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.7
 */
TEST_CASE("Unit_hipHccModuleLaunchKernel_NegTst") {
  size_t width = GENERATE(3, 4, 100);
  size_t widthInBytes = width * sizeof(int);
  int *A_d, *B_d;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&A_d), widthInBytes));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&B_d), widthInBytes));
  void* kernelArgs[3] = {&A_d, &B_d, &widthInBytes};
  hipModule_t module;
  HIP_CHECK(hipModuleLoad(&module, fileName));
  hipFunction_t kernelFunc;
  HIP_CHECK(hipModuleGetFunction(&kernelFunc, module, kernel_name));
  SECTION("nullptr to f(first argument)") {
    HIP_CHECK_ERROR(hipHccModuleLaunchKernel(nullptr, width, 1, 1, width, 1, 1, 0, 0, kernelArgs,
                                             nullptr, nullptr, nullptr),
                    hipErrorInvalidHandle);
  }
  SECTION("-1 to localWorkSizeX(fifth argument)") {
    HIP_CHECK_ERROR(hipHccModuleLaunchKernel(kernelFunc, width, 1, 1, -1, 1, 1, 0, 0, kernelArgs,
                                             nullptr, nullptr, nullptr),
                    hipErrorInvalidConfiguration);
  }
  SECTION("-1 to localWorkSizeY(sixth argument)") {
    HIP_CHECK_ERROR(hipHccModuleLaunchKernel(kernelFunc, width, 1, 1, width, -1, 1, 0, 0,
                                             kernelArgs, nullptr, nullptr, nullptr),
                    hipErrorInvalidConfiguration);
  }
  SECTION("-1 to localWorkSizeZ(seventh argument)") {
    HIP_CHECK_ERROR(hipHccModuleLaunchKernel(kernelFunc, width, 1, 1, width, 1, -1, 0, 0,
                                             kernelArgs, nullptr, nullptr, nullptr),
                    hipErrorInvalidConfiguration);
  }
  SECTION("-1 to sharedMemBytes(eighth argument)") {
    HIP_CHECK_ERROR(hipHccModuleLaunchKernel(kernelFunc, width, 1, 1, width, 1, 1, -1, 0,
                                             kernelArgs, nullptr, nullptr, nullptr),
                    hipErrorInvalidValue);
  }
  SECTION("nullptr to kernelParams(10th argument)") {
    HIP_CHECK_ERROR(hipHccModuleLaunchKernel(kernelFunc, width, 1, 1, width, 1, 1, 0, 0, nullptr,
                                             nullptr, nullptr, nullptr),
                    hipErrorInvalidValue);
  }
  HIP_CHECK(hipModuleUnload(module));
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(B_d));
}
