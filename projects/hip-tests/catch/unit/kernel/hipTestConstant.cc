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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <hip_test_common.hh>


#define LEN 512
#define SIZE 2048

__constant__ int Value[LEN];

static __global__ void Get(int* Ad) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  Ad[tid] = Value[tid];
}
/**
* @addtogroup hipLaunchKernelGGL
* @{
* @ingroup KernelTest
* `void hipLaunchKernelGGL(F kernel, const dim3& numBlocks, const dim3& dimBlocks,
   std::uint32_t sharedMemBytes, hipStream_t stream, Args... args)` -
* Method to invocate kernel functions
*/

/**
 * Test Description
 * ------------------------
 * - Test case to check const variable via kernel call.

 * Test source
 * ------------------------
 * - catch/unit/kernel/hipTestConstant.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 5.6
 */

TEST_CASE("Unit_kernel_chkConstantViaKernel") {
  int *A, *B, *Ad;
  A = new int[LEN];
  B = new int[LEN];
  for (unsigned i = 0; i < LEN; i++) {
    A[i] = -1 * i;
    B[i] = 0;
  }

  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ad), SIZE));

  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(Value), A, SIZE, 0, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(Get, dim3(1, 1, 1), dim3(LEN, 1, 1), 0, 0, Ad);
  HIP_CHECK(hipMemcpy(B, Ad, SIZE, hipMemcpyDeviceToHost));

  for (unsigned i = 0; i < LEN; i++) {
    REQUIRE(A[i] == B[i]);
  }
  delete[] A;
  delete[] B;
  HIP_CHECK(hipFree(Ad));
}


/**
 * End doxygen group KernelTest.
 * @}
 */
