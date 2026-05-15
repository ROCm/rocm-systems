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

#include <hip_test_kernels.hh>
#include <hip_test_common.hh>
#include <hip_test_checkers.hh>

#define LEN 512
#define SIZE (LEN * sizeof(int64_t))

static __global__ void kernel1(int64_t* Ad) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  Ad[tid] = clock() + clock64() + __clock() + __clock64();
}

static __global__ void kernel2(int64_t* Ad) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  Ad[tid] = clock() + clock64() + __clock() + __clock64() - Ad[tid];
}

TEST_CASE("Unit_hipTestClock") {
  int64_t *A, *Ad;
  A = new int64_t[LEN];
  for (unsigned i = 0; i < LEN; i++) {
    A[i] = 0;
  }
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ad), SIZE));
  HIP_CHECK(hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(kernel1, dim3(1, 1, 1), dim3(LEN, 1, 1), 0, 0, Ad);
  hipLaunchKernelGGL(kernel2, dim3(1, 1, 1), dim3(LEN, 1, 1), 0, 0, Ad);
  HIP_CHECK(hipMemcpy(A, Ad, SIZE, hipMemcpyDeviceToHost));
  for (unsigned i = 0; i < LEN; i++) {
    assert(0 != A[i]);
  }

  HIP_CHECK(hipFree(Ad));
  delete[] A;
}
