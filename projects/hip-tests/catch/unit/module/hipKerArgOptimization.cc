/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hip_test_common.hh>
#include "hip/hip_ext.h"

static constexpr auto fileName = "copiousArgKernel.code";
static constexpr auto kernel_name = "kernelMultipleArgsSaxpy";
static constexpr auto fileName0 = "copiousArgKernel0.code";
static constexpr auto fileName1 = "copiousArgKernel1.code";
static constexpr auto fileName2 = "copiousArgKernel2.code";
static constexpr auto fileName3 = "copiousArgKernel3.code";
static constexpr auto fileName16 = "copiousArgKernel16.code";
static constexpr auto fileName17 = "copiousArgKernel17.code";

static constexpr int coeff[12] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};

static void fillDataTransfer2Dev(int* hostBuf, int* devBuf, size_t len) {
  unsigned int seed = time(nullptr);
  for (size_t i = 0; i < len; i++) {
    hostBuf[i] = (HipTest::RAND_R(&seed) & 0xFF);
  }
  HIP_CHECK(hipMemcpy(devBuf, hostBuf, len * sizeof(int), hipMemcpyHostToDevice));
}

static void verifyDevResult(int* hostBuf, int* devBuf, int coef1, int coef2, size_t len) {
  int* buf = new int[len];
  HIP_CHECK(hipMemcpy(buf, devBuf, len * sizeof(int), hipMemcpyDeviceToHost));
  for (size_t i = 0; i < len; i++) {
    REQUIRE(buf[i] == (coef1 * hostBuf[i] + coef2));
  }
  delete[] buf;
}

TEST_CASE("Unit_KerArgOptimization_Saxpy") {
  constexpr size_t arraylen = 1 << 16;
  constexpr size_t arraylenBytes = arraylen * sizeof(int);
  constexpr auto blocksize = 256;
  // Allocate host resources
  int* x1_h = new int[arraylen];
  REQUIRE(x1_h != nullptr);
  int* x2_h = new int[arraylen];
  REQUIRE(x2_h != nullptr);
  int* x3_h = new int[arraylen];
  REQUIRE(x3_h != nullptr);
  int* x4_h = new int[arraylen];
  REQUIRE(x4_h != nullptr);
  int* x5_h = new int[arraylen];
  REQUIRE(x5_h != nullptr);
  int* x6_h = new int[arraylen];
  REQUIRE(x6_h != nullptr);
  // Allocate device resources
  int *x1_d, *x2_d, *x3_d, *x4_d, *x5_d, *x6_d;
  HIP_CHECK(hipMalloc(&x1_d, arraylenBytes));
  HIP_CHECK(hipMalloc(&x2_d, arraylenBytes));
  HIP_CHECK(hipMalloc(&x3_d, arraylenBytes));
  HIP_CHECK(hipMalloc(&x4_d, arraylenBytes));
  HIP_CHECK(hipMalloc(&x5_d, arraylenBytes));
  HIP_CHECK(hipMalloc(&x6_d, arraylenBytes));
  // Fill data and transfer to device
  fillDataTransfer2Dev(x1_h, x1_d, arraylen);
  fillDataTransfer2Dev(x2_h, x2_d, arraylen);
  fillDataTransfer2Dev(x3_h, x3_d, arraylen);
  fillDataTransfer2Dev(x4_h, x4_d, arraylen);
  fillDataTransfer2Dev(x5_h, x5_d, arraylen);
  fillDataTransfer2Dev(x6_h, x6_d, arraylen);

  // Kernel launch
  struct {
    int a1;
    int a2;
    void* x1;
    int b1;
    int b2;
    void* x2;
    int c1;
    int c2;
    void* x3;
    int d1;
    int d2;
    void* x4;
    int e1;
    int e2;
    void* x5;
    int f1;
    int f2;
    void* x6;
  } args;
  args.a1 = coeff[0];
  args.a2 = coeff[1];
  args.x1 = x1_d;
  args.b1 = coeff[2];
  args.b2 = coeff[3];
  args.x2 = x2_d;
  args.c1 = coeff[4];
  args.c2 = coeff[5];
  args.x3 = x3_d;
  args.d1 = coeff[6];
  args.d2 = coeff[7];
  args.x4 = x4_d;
  args.e1 = coeff[8];
  args.e2 = coeff[9];
  args.x5 = x5_d;
  args.f1 = coeff[10];
  args.f2 = coeff[11];
  args.x6 = x6_d;
  size_t size = sizeof(args);
  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                    HIP_LAUNCH_PARAM_END};

  // Get module and function from module
  hipModule_t Module;
  hipFunction_t Function;
  SECTION("No amdgpu-kernarg-preload-count") {
    HIP_CHECK(hipModuleLoad(&Module, fileName));
    HIP_CHECK(hipModuleGetFunction(&Function, Module, kernel_name));
  }
  SECTION("amdgpu-kernarg-preload-count = 0") {
    HIP_CHECK(hipModuleLoad(&Module, fileName0));
    HIP_CHECK(hipModuleGetFunction(&Function, Module, kernel_name));
  }
  SECTION("amdgpu-kernarg-preload-count = 1") {
    HIP_CHECK(hipModuleLoad(&Module, fileName1));
    HIP_CHECK(hipModuleGetFunction(&Function, Module, kernel_name));
  }
  SECTION("amdgpu-kernarg-preload-count = 2") {
    HIP_CHECK(hipModuleLoad(&Module, fileName2));
    HIP_CHECK(hipModuleGetFunction(&Function, Module, kernel_name));
  }
  SECTION("amdgpu-kernarg-preload-count = 3") {
    HIP_CHECK(hipModuleLoad(&Module, fileName3));
    HIP_CHECK(hipModuleGetFunction(&Function, Module, kernel_name));
  }
  SECTION("amdgpu-kernarg-preload-count = 16") {
    HIP_CHECK(hipModuleLoad(&Module, fileName16));
    HIP_CHECK(hipModuleGetFunction(&Function, Module, kernel_name));
  }
  SECTION("amdgpu-kernarg-preload-count = 17") {
    HIP_CHECK(hipModuleLoad(&Module, fileName17));
    HIP_CHECK(hipModuleGetFunction(&Function, Module, kernel_name));
  }
  HIP_CHECK(hipExtModuleLaunchKernel(Function, arraylen, 1, 1, blocksize, 1, 1, 0, 0, NULL,
                                     reinterpret_cast<void**>(&config), 0));
  HIP_CHECK(hipDeviceSynchronize());
  // Verify results
  verifyDevResult(x1_h, x1_d, coeff[0], coeff[1], arraylen);
  verifyDevResult(x2_h, x2_d, coeff[2], coeff[3], arraylen);
  verifyDevResult(x3_h, x3_d, coeff[4], coeff[5], arraylen);
  verifyDevResult(x4_h, x4_d, coeff[6], coeff[7], arraylen);
  verifyDevResult(x5_h, x5_d, coeff[8], coeff[9], arraylen);
  verifyDevResult(x6_h, x6_d, coeff[10], coeff[11], arraylen);
  // Delete resources
  HIP_CHECK(hipFree(x1_d));
  HIP_CHECK(hipFree(x2_d));
  HIP_CHECK(hipFree(x3_d));
  HIP_CHECK(hipFree(x4_d));
  HIP_CHECK(hipFree(x5_d));
  HIP_CHECK(hipFree(x6_d));
  delete[] x1_h;
  delete[] x2_h;
  delete[] x3_h;
  delete[] x4_h;
  delete[] x5_h;
  delete[] x6_h;
}
