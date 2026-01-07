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
// Test the Grid_Launch syntax.

#include <hip_test_kernels.hh>
#include <hip_test_checkers.hh>
#include <hip_test_common.hh>

#include "hip/hip_ext.h"

static unsigned threadsPerBlock = 256;
static unsigned blocksPerCU = 6;

struct _t {
  double _a, _b, _c, _d, _e, _f, _g, _h, _i, _j;
};

typedef struct _t _T;

__global__ void sKernel(_T s, double* a) {
  *a = s._a + s._b + s._c + s._d + s._e + s._f + s._g + s._h + s._i + s._j;
}

__global__ void mKernel(char f, int16_t a, int b, double c, int16_t d, int e, double* res) {
  *res = a + b + c + d + e + f;
}

void testMixData() {
  double m = 0;
  double* d_m;
  HIP_CHECK(hipMalloc(&d_m, sizeof(double)));
  int a = 1, e = 10;
  int16_t b = 2, d = 4;
  double c = 3.0;
  char ff = 10;
  hipExtLaunchKernelGGL(mKernel, 1, 1, 0, 0, nullptr, nullptr, 0, ff, b, a, c, d, e, d_m);
  HIP_CHECK(hipMemcpy(&m, d_m, sizeof(double), hipMemcpyDeviceToHost));
  REQUIRE(m == 30.0);
  HIP_CHECK(hipFree(d_m));
}

void testStruct() {
  double m = 0;
  double* d_m;
  HIP_CHECK(hipMalloc(&d_m, sizeof(double)));
  _T s{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  hipExtLaunchKernelGGL(sKernel, 1, 1, 0, 0, nullptr, nullptr, 0, s, d_m);
  HIP_CHECK(hipMemcpy(&m, d_m, sizeof(double), hipMemcpyDeviceToHost));
  REQUIRE(m == 55.0);
  HIP_CHECK(hipFree(d_m));
}

void test(size_t N) {
  size_t Nbytes = N * sizeof(int);
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;

  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N);

  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  HIP_CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_d, B_h, Nbytes, hipMemcpyHostToDevice));

  hipExtLaunchKernelGGL(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock), 0, 0, nullptr,
                        nullptr, 0, static_cast<const int*>(A_d), static_cast<const int*>(B_d), C_d,
                        N);

  HIP_CHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());
  HipTest::checkVectorADD(A_h, B_h, C_h, N);
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
}

/**
* @addtogroup hipExtLaunchKernelGGL hipExtLaunchKernelGGL
* @{
* @ingroup KernelTest
* `void hipExtLaunchKernelGGL(F kernel, const dim3& numBlocks, const dim3& dimBlocks,
                              std::uint32_t sharedMemBytes, hipStream_t stream,
                              hipEvent_t startEvent, hipEvent_t stopEvent, std::uint32_t flags,
                              Args... args)` -
* Launches kernel with dimention parameters and shared memory on stream with templated kernel and
arguments
*/

/**
 * Test Description
 * ------------------------
 *    - Test case to verify sample array with hipExtLaunchKernelGGL()
 * and verify the results.
 *    - Test case to verify struct data with hipExtLaunchKernelGGL()
 * and verify the results.
 *    - Test case to verify mix datatypes with hipExtLaunchKernelGGL()
 * and verify the results.

 * Test source
 * ------------------------
 *    - catch/unit/kernel/hipExtLaunchKernelGGL.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.5
 */

TEST_CASE("Unit_hipExtLaunchKernelGGL") {
  SECTION("test run") {
    size_t N = 4 * 1024 * 1024;
    test(N);
  }
  SECTION("testStruct run") { testStruct(); }
  SECTION("testMixData run") { testMixData(); }
}

/**
 * End doxygen group KernelTest.
 * @}
 */
