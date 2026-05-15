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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>

/**
 * @addtogroup hipMemcpyAtoD hipMemcpyAtoD
 * @{
 * @ingroup MemoryTest
 * `hipError_t hipMemcpyAtoD(hipDeviceptr_t dstDevice, hipArray_t srcArray,
 *                           size_t srcOffset, size_t ByteCount)` -
 * Copies from one 1D array to device memory.
 */

/**
 * Test Description
 * ------------------------
 *  - This testcase initially copies data from host to 1D array and then performs
 *  hipMemcpyAtoD api call and copies this device memory to host variable and
 *  verifies with initial host values.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpyAtoD.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMemcpyAtoD_Basic") {
#if HT_NVIDIA
  HipTest::HIP_SKIP_TEST("API currently unsupported on nvidia, skipping...");
  return;
#else
  HIP_CHECK(hipSetDevice(0));
  CHECK_IMAGE_SUPPORT
  int row, col;
  row = 1;
  col = GENERATE(3, 4, 100);
  int* A_h = reinterpret_cast<int*>(malloc(sizeof(int) * row * col));
  int* B_h = reinterpret_cast<int*>(malloc(sizeof(int) * row * col));
  for (int i = 0; i < (row * col); i++) {
    A_h[i] = i;
  }
  hipArray_t A_a;
  int* A_d;
  hipChannelFormatDesc desc = hipCreateChannelDesc<int>();
  HIP_CHECK(hipMallocArray(&A_a, &desc, col, row, hipArrayDefault));
  HIP_CHECK(hipMalloc(&A_d, sizeof(int) * row * col));

  hipError_t memcpy_err = hipSuccess;
  BEGIN_CAPTURE_SYNC(memcpy_err, false);
  HIP_CHECK_ERROR(hipMemcpy2DToArray(A_a, 0, 0, A_h, col * sizeof(int), col * sizeof(int), row,
                                     hipMemcpyHostToDevice),
                  memcpy_err);
  END_CAPTURE_SYNC(memcpy_err);

  if (memcpy_err == hipSuccess) {
    HIP_CHECK(hipMemcpyAtoD(A_d, A_a, 0, sizeof(int) * col * row));
    HIP_CHECK(hipMemcpyDtoH(B_h, A_d, sizeof(int) * row * col));
    for (int i = 0; i < (row * col); i++) {
      REQUIRE(A_h[i] == B_h[i]);
    }
  }
  HIP_CHECK(hipFreeArray(A_a));
  HIP_CHECK(hipFree(A_d));
  free(A_h);
  free(B_h);
#endif
}

/**
 * End doxygen group MemoryTest.
 * @}
 */
