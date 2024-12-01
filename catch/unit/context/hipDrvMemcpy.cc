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

TEST_CASE("Unit_hipDrvMemcpy_Functional") {
  int *A, *B;
  hipDeviceptr_t Ad, Bd;
  A = new int[LEN];
  B = new int[LEN];

  for (int i = 0; i < LEN; i++) {
    A[i] = i;
  }

  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ad), SIZE));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Bd), SIZE));

  HIP_CHECK(hipMemcpyHtoD(Ad, A, SIZE));
  HIP_CHECK(hipMemcpyDtoD(Bd, Ad, SIZE));
  HIP_CHECK(hipMemcpyDtoH(B, Bd, SIZE));

  for (int i = 0; i < 16; i++) {
    REQUIRE(A[i] == B[i]);
  }

  int *Ah, *Bh;
  HIP_CHECK(hipHostMalloc(&Ah, SIZE, 0));
  HIP_CHECK(hipHostMalloc(&Bh, SIZE, 0));
  memcpy(Ah, A, SIZE);
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(hipMemcpyHtoDAsync(Ad, Ah, SIZE, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipMemcpyDtoDAsync(Bd, Ad, SIZE, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipMemcpyDtoHAsync(Bh, Bd, SIZE, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(reinterpret_cast<void*>(Ad)));
  HIP_CHECK(hipFree(reinterpret_cast<void*>(Bd)));

  REQUIRE(Ah[10] == Bh[10]);
  HIP_CHECK(hipFreeHost(Ah));
  HIP_CHECK(hipFreeHost(Bh));
  delete[] A;
  delete[] B;
}
