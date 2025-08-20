/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

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

#include <cmath>
#include <cstring>
#include <hip_test_common.hh>
#include <hip/hip_fp16.h>

#define NElms 5


// Kernel functions
__global__ void HMinMaxHalfOps(__half x, __half y, __half ExptdResult, int* TstResult,
                               int TstToRun) {
  __half OutVal = 0;
  if (TstToRun == 1) {
    OutVal = __hmax(x, y);
    if (!(__heq(OutVal, ExptdResult))) {
      *TstResult = 0;  // Indicates Test failed
    }
  } else if (TstToRun == 2) {
    OutVal = __hmin(x, y);
    if (!(__heq(OutVal, ExptdResult))) {
      *TstResult = 0;  // Indicates Test failed
    }
  } else if (TstToRun == 3) {
    OutVal = __hmax_nan(x, y);
    if ((__hisnan(ExptdResult))) {
      if (!(__hisnan(OutVal))) {
        *TstResult = 0;  // Indicates Test failed
      }
    } else {
      if (!(__heq(OutVal, ExptdResult))) {
        *TstResult = 0;  // Indicates Test failed
      }
    }
  } else if (TstToRun == 4) {
    OutVal = __hmin_nan(x, y);
    if ((__hisnan(ExptdResult))) {
      if (!(__hisnan(OutVal))) {
        *TstResult = 0;  // Indicates Test failed
      }
    } else {
      if (!(__heq(OutVal, ExptdResult))) {
        *TstResult = 0;  // Indicates Test failed
      }
    }
  }
}

__global__ void HMinMaxHalfOpsArray(__half* x, __half* y, __half* ExptdResult, int* TstResult,
                                    int TstToRun) {
  size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
  if (offset < NElms) {
    __half OutVal = 0;
    if (TstToRun == 1) {
      OutVal = __hmax(x[offset], y[offset]);
      if (!(__heq(OutVal, ExptdResult[offset]))) {
        *TstResult = 0;  // Indicates Test failed
      }
    } else if (TstToRun == 2) {
      OutVal = __hmin(x[offset], y[offset]);
      if (!(__heq(OutVal, ExptdResult[offset]))) {
        *TstResult = 0;  // Indicates Test failed
      }
    }
  }
}

// The following tests checks the basic functionality of __hmax(), __hmin()
// __hmax_nan() and __hmin_nan()
TEST_CASE("Unit_hmax_hmin_Tsts") {
  int *Hptr = nullptr, *Dptr = nullptr;
  HIP_CHECK(hipHostMalloc(&Hptr, sizeof(int)));
  *Hptr = 1;
  HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&Dptr), Hptr, 0));

  __half x = GENERATE(0, 0.1, 1.5);
  __half y = GENERATE(1.5, 2, 10);
  SECTION("Running test for __hmax()") {
    HMinMaxHalfOps<<<1, 1>>>(x, y, y, Dptr, 1);
    HIP_CHECK(hipStreamSynchronize(0));
    if (*Hptr == 0) {
      REQUIRE(false);
    }
    HMinMaxHalfOps<<<1, 1>>>(y, x, y, Dptr, 1);
    HIP_CHECK(hipStreamSynchronize(0));
    if (*Hptr == 0) {
      REQUIRE(false);
    }
  }

  *Hptr = 1;
  SECTION("Running test for __hmin()") {
    HMinMaxHalfOps<<<1, 1>>>(x, y, x, Dptr, 2);
    HIP_CHECK(hipStreamSynchronize(0));
    if (*Hptr == 0) {
      REQUIRE(false);
    }
    HMinMaxHalfOps<<<1, 1>>>(y, x, x, Dptr, 2);
    HIP_CHECK(hipStreamSynchronize(0));
    if (*Hptr == 0) {
      REQUIRE(false);
    }
  }

  *Hptr = 1;
  SECTION("Running test for __hmax_nan()") {
    HMinMaxHalfOps<<<1, 1>>>(x, nan("1"), nan("1"), Dptr, 3);
    HIP_CHECK(hipStreamSynchronize(0));

    if (*Hptr == 0) {
      REQUIRE(false);
    }

    HMinMaxHalfOps<<<1, 1>>>(nan("1"), x, nan("1"), Dptr, 3);
    HIP_CHECK(hipStreamSynchronize(0));

    if (*Hptr == 0) {
      REQUIRE(false);
    }

    HMinMaxHalfOps<<<1, 1>>>(nan("1"), nan("1"), nan("1"), Dptr, 3);
    HIP_CHECK(hipStreamSynchronize(0));

    if (*Hptr == 0) {
      REQUIRE(false);
    }

    HMinMaxHalfOps<<<1, 1>>>(x, y, y, Dptr, 3);
    HIP_CHECK(hipStreamSynchronize(0));

    if (*Hptr == 0) {
      REQUIRE(false);
    }

    HMinMaxHalfOps<<<1, 1>>>(y, x, y, Dptr, 3);
    HIP_CHECK(hipStreamSynchronize(0));

    if (*Hptr == 0) {
      REQUIRE(false);
    }
  }

  *Hptr = 1;
  SECTION("Running test for __hmin_nan()") {
    HMinMaxHalfOps<<<1, 1>>>(x, nan("1"), nan("1"), Dptr, 4);
    HIP_CHECK(hipStreamSynchronize(0));

    if (*Hptr == 0) {
      REQUIRE(false);
    }

    HMinMaxHalfOps<<<1, 1>>>(nan("1"), x, nan("1"), Dptr, 4);
    HIP_CHECK(hipStreamSynchronize(0));

    if (*Hptr == 0) {
      REQUIRE(false);
    }

    HMinMaxHalfOps<<<1, 1>>>(nan("1"), nan("1"), nan("1"), Dptr, 4);
    HIP_CHECK(hipStreamSynchronize(0));

    if (*Hptr == 0) {
      REQUIRE(false);
    }

    HMinMaxHalfOps<<<1, 1>>>(x, y, x, Dptr, 4);
    HIP_CHECK(hipStreamSynchronize(0));

    if (*Hptr == 0) {
      REQUIRE(false);
    }

    HMinMaxHalfOps<<<1, 1>>>(y, x, x, Dptr, 4);
    HIP_CHECK(hipStreamSynchronize(0));

    if (*Hptr == 0) {
      REQUIRE(false);
    }
  }
  HIP_CHECK(hipHostFree(Hptr));
}


// The following Tests does negative testing by passing nan
TEST_CASE("Unit_hmax_hmin_Tsts_Negative") {
  int *Hptr = nullptr, *Dptr = nullptr;
  HIP_CHECK(hipHostMalloc(&Hptr, sizeof(int)));
  *Hptr = 1;
  HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&Dptr), Hptr, 0));
  __half x = nan("");
  __half y = 1.5;
  SECTION("Running Negative test for __hmax()") {
    HMinMaxHalfOps<<<1, 1>>>(x, y, y, Dptr, 1);
    HIP_CHECK(hipStreamSynchronize(0));
    if (*Hptr == 0) {
      REQUIRE(false);
    }
  }

  SECTION("Running Negative test for __hmin()") {
    HMinMaxHalfOps<<<1, 1>>>(x, y, y, Dptr, 2);
    HIP_CHECK(hipStreamSynchronize(0));
    if (*Hptr == 0) {
      REQUIRE(false);
    }
  }

  HIP_CHECK(hipHostFree(Hptr));
}

// The following tests the __hmax/min functions over array of memory
TEST_CASE("Unit_hmax_hmin_Tsts_With_Array") {
  int *Hptr = nullptr, *Dptr = nullptr;
  __half *Ad = nullptr, *Bd = nullptr;
  HIP_CHECK(hipHostMalloc(&Hptr, sizeof(int)));
  *Hptr = 1;
  HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&Dptr), Hptr, 0));
  __half x[NElms] = {1, 2, 3, 4, 5};
  __half y[NElms] = {7, 8, 9, 10, 11};
  HIP_CHECK(hipMalloc(&Ad, NElms * sizeof(__half)));
  HIP_CHECK(hipMalloc(&Bd, NElms * sizeof(__half)));
  HIP_CHECK(hipMemcpy(Ad, x, NElms * sizeof(__half), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(Bd, y, NElms * sizeof(__half), hipMemcpyHostToDevice));
  SECTION("Running test for __hmax() with an array") {
    HMinMaxHalfOpsArray<<<1, NElms>>>(Ad, Bd, Bd, Dptr, 1);
    HIP_CHECK(hipStreamSynchronize(0));
    if (*Hptr == 0) {
      REQUIRE(false);
    }
  }

  SECTION("Running test for __hmin() with an array") {
    HMinMaxHalfOpsArray<<<1, NElms>>>(Ad, Bd, Ad, Dptr, 2);
    HIP_CHECK(hipStreamSynchronize(0));
    if (*Hptr == 0) {
      REQUIRE(false);
    }
  }

  HIP_CHECK(hipHostFree(Hptr));
  HIP_CHECK(hipFree(Ad));
  HIP_CHECK(hipFree(Bd));
}
