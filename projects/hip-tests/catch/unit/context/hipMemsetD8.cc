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
#define N 1024
constexpr char memsetval = 'b';

TEST_CASE("Unit_hipMemsetD8_Functional") {
  size_t Nbytes = N * sizeof(char);
  char* A_h = new char[Nbytes];
  ;

  hipDeviceptr_t A_d;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&A_d), Nbytes));

  HIP_CHECK(hipMemsetD8(A_d, memsetval, Nbytes));

  HIP_CHECK(hipMemcpy(A_h, reinterpret_cast<void*>(A_d), Nbytes, hipMemcpyDeviceToHost));

  for (int i = 0; i < N; i++) {
    REQUIRE(A_h[i] == memsetval);
  }

  HIP_CHECK(hipFree(reinterpret_cast<void*>(A_d)));
  delete[] A_h;
}
