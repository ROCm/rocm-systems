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

#define LEN 1024
#define SIZE (LEN << 2)

__global__ static void cpy(uint32_t* Out, uint32_t* In) {
  int tx = threadIdx.x;
  memcpy(Out + tx, In + tx, sizeof(uint32_t));
}

__global__ static void set(uint32_t* ptr, uint8_t val) {
  int tx = threadIdx.x;
  memset(ptr + tx, val, sizeof(uint32_t));
}

TEST_CASE("Unit_ToAndFroMemCpyToDevice") {
  uint32_t *A, *Ad, *B, *Bd;
  A = new uint32_t[LEN];
  B = new uint32_t[LEN];
  for (int i = 0; i < LEN; i++) {
    A[i] = i;
    B[i] = 0;
  }
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ad), SIZE));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Bd), SIZE));
  HIP_CHECK(hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice));

  hipLaunchKernelGGL(cpy, dim3(1), dim3(LEN), 0, 0, Bd, Ad);

  HIP_CHECK(hipMemcpy(B, Bd, SIZE, hipMemcpyDeviceToHost));
  for (int i = LEN - 16; i < LEN; i++) {
    REQUIRE(A[i] == B[i]);
  }
  hipLaunchKernelGGL(set, dim3(1), dim3(LEN), 0, 0, Bd, 0x1);

  HIP_CHECK(hipMemcpy(B, Bd, SIZE, hipMemcpyDeviceToHost));
  for (int i = LEN - 16; i < LEN; i++) {
    REQUIRE(0x01010101 == B[i]);
  }

  HIP_CHECK(hipFree(Ad));
  HIP_CHECK(hipFree(Bd));
  delete[] A;
  delete[] B;
}
