/*
Copyright (c) 2021-25 Advanced Micro Devices, Inc. All rights reserved.
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
TEST_CASE("Unit_hipMemGetInfo_FreeLessThanTotal") {
  unsigned int* A_mem{nullptr};
  size_t freeMemInit, totalMemInit;
  size_t freeMem, totalMem;
  GENERATE_CAPTURE();
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipMemGetInfo(&freeMemInit, &totalMemInit));
  REQUIRE(freeMemInit <= totalMemInit);
  HIP_CHECK(hipMalloc(&A_mem, 1024));
  hipError_t memcpy_err = hipSuccess;
  BEGIN_CAPTURE(stream);
  HIP_CHECK_ERROR(hipMemGetInfo(&freeMem, &totalMem), memcpy_err);
  END_CAPTURE(stream);
  if (memcpy_err == hipSuccess) {
    REQUIRE(freeMem < totalMem);
    REQUIRE(totalMem == totalMemInit);
  }
  HIP_CHECK(hipFree(A_mem));
}
