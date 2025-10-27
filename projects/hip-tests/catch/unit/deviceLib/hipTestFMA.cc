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
#include <iostream>

#define LEN 50
#define SIZE (LEN * sizeof(bool))

__global__ void kernelTestFMA(bool* Ad) {
  float f = 1.0f / 3.0f;
  double d = f;
  int i = 0;
  auto Check = [&](bool Cond) { Ad[i++] = Cond; };
  // f * f + 3.0f will be different if promoted to double.
  float floatResult = fma(f, f, 3.0f);
  double doubleResult = fma(d, d, 3.0);
  Check(floatResult != doubleResult);

  if (sizeof(decltype(fma(f, f, 3))) == 8) {
    // To align with libcxx, if any argument has integral type,
    // it is cast to double.
    // Check type promotes to double.
    Check(fma(f, f, 3) == doubleResult);
    Check(fma(f, f, static_cast<char>(3)) == doubleResult);
    Check(fma(f, f, (unsigned char)3) == doubleResult);
    Check(fma(f, f, (int32_t)3) == doubleResult);
    Check(fma(f, f, (uint32_t)3) == doubleResult);
    Check(fma(f, f, static_cast<int>(3)) == doubleResult);
    Check(fma(f, f, (unsigned int)3) == doubleResult);
    Check(fma(f, f, (int64_t)3) == doubleResult);
    Check(fma(f, f, (uint64_t)3) == doubleResult);
    Check(fma(f, f, true) == fma(static_cast<double>(f), static_cast<double>(f), 1.0));
  } else if (sizeof(decltype(fma(f, f, 3))) == 4) {
    // Previous HIP headers returns float type.
    // Delete this to support backwards compatibility.
    // check promote to float.
    Check(fma(f, f, 3) == floatResult);
    Check(fma(f, f, static_cast<char>(3)) == floatResult);
    Check(fma(f, f, (unsigned char)3) == floatResult);
    Check(fma(f, f, (int32_t)3) == floatResult);
    Check(fma(f, f, (uint32_t)3) == floatResult);
    Check(fma(f, f, static_cast<int>(3)) == floatResult);
    Check(fma(f, f, (unsigned int)3) == floatResult);
    Check(fma(f, f, (int64_t)3) == floatResult);
    Check(fma(f, f, (uint64_t)3) == floatResult);
    Check(fma(f, f, true) == fma(f, f, 1.0f));
  } else {
    Check(false);
  }

  Check(fma(d, static_cast<double>(f), 3) == doubleResult);
  Check(fma(d, static_cast<double>(f), static_cast<char>(3)) == doubleResult);
  Check(fma(d, static_cast<double>(f), (unsigned char)3) == doubleResult);
  Check(fma(d, static_cast<double>(f), (int32_t)3) == doubleResult);
  Check(fma(d, static_cast<double>(f), (uint32_t)3) == doubleResult);
  Check(fma(d, static_cast<double>(f), static_cast<int>(3)) == doubleResult);
  Check(fma(d, static_cast<double>(f), (unsigned int)3) == doubleResult);
  Check(fma(d, static_cast<double>(f), (int64_t)3) == doubleResult);
  Check(fma(d, static_cast<double>(f), (int64_t)3) == doubleResult);
  Check(fma(d, static_cast<double>(f), true) ==
        fma(static_cast<double>(f), static_cast<double>(f), 1.0));

  while (i < LEN) Check(true);
}

void runTestFMA() {
  bool* Ad;
  bool A[LEN];
  for (unsigned i = 0; i < LEN; i++) {
    A[i] = 0;
  }

  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ad), SIZE));
  hipLaunchKernelGGL(kernelTestFMA, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0, Ad);
  HIP_CHECK(hipMemcpy(A, Ad, SIZE, hipMemcpyDeviceToHost));

  for (unsigned i = 0; i < LEN; i++) {
    REQUIRE(A[i] == true);
  }

  HIP_CHECK(hipFree(Ad));
}

__global__ void kernelTestHalfFMA(bool* Ad) {
  _Float16 h = (_Float16)(1.0f / 3.0f);
  float f = h;
  double d = f;
  int i = 0;
  auto Check = [&](bool Cond) { Ad[i++] = Cond; };
  // h * h + 3 will be different if promoted to float.
  _Float16 halfResult = fma(h, h, (_Float16)3);
  float floatResult = fma(f, f, 3.0f);
  double doubleResult = fma(d, d, 3.0);
  Check(halfResult != floatResult);
  Check(halfResult != doubleResult);

  // check promote to half.
  // fma(_Float16, _Float16, int) should resolve to
  // fma(double, double, double). This is similar to
  // fma(float, float, int) resolving to fma(double, double, double)
  // as required Standard C++ header <cmath>.
  if (sizeof(decltype(fma(h, h, 3))) == 8) {
    Check(fma(h, h, 3) == doubleResult);
    Check(fma(h, h, static_cast<char>(3)) == doubleResult);
    Check(fma(h, h, (unsigned char)3) == doubleResult);
    Check(fma(h, h, (int32_t)3) == doubleResult);
    Check(fma(h, h, (uint32_t)3) == doubleResult);
    Check(fma(h, h, static_cast<int>(3)) == doubleResult);
    Check(fma(h, h, (unsigned int)3) == doubleResult);
    Check(fma(h, h, (int64_t)3) == doubleResult);
    Check(fma(h, h, (uint64_t)3) == doubleResult);
    Check(fma(h, h, true) == fma(static_cast<double>(h), static_cast<double>(h), 1.0));
  } else if (sizeof(decltype(fma(h, h, 3))) == 2) {
    // ToDo: Currently there is a bug in clang header
    // __clang_hip_cmath.h due to using
    // std::numeric_limits<T>::is_specified to define
    // overloaded math functions. Since numeric_limits is
    // not specicialized for _Float16, overloaded template
    // functions with argument promotion are not defined
    // for _Float16. As a result, fma(_Float16, _Float16, int)
    // is resolved to fma(_Float16, _Float16, _Float16).
    // This part should be removed after __clang_hip_cmath.h
    // is fixed.
    Check(fma(h, h, 3) == halfResult);
    Check(fma(h, h, static_cast<char>(3)) == halfResult);
    Check(fma(h, h, (unsigned char)3) == halfResult);
    Check(fma(h, h, (int32_t)3) == halfResult);
    Check(fma(h, h, (uint32_t)3) == halfResult);
    Check(fma(h, h, static_cast<int>(3)) == halfResult);
    Check(fma(h, h, (unsigned int)3) == halfResult);
    Check(fma(h, h, (int64_t)3) == halfResult);
    Check(fma(h, h, (int64_t)3) == halfResult);
    Check(fma(h, h, true) == fma(h, h, (_Float16)1));
  } else {
    Check(false);
  }

  while (i < LEN) Check(true);
}

void runTestHalfFMA() {
  bool* Ad;
  bool A[LEN];
  for (unsigned i = 0; i < LEN; i++) {
    A[i] = 0;
  }

  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ad), SIZE));
  hipLaunchKernelGGL(kernelTestHalfFMA, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0, Ad);
  HIP_CHECK(hipMemcpy(A, Ad, SIZE, hipMemcpyDeviceToHost));

  for (unsigned i = 0; i < LEN; i++) {
    REQUIRE(A[i] == true);
  }

  HIP_CHECK(hipFree(Ad));
}

TEST_CASE("Unit_hipTestFMA") {
  SECTION("test FMA") { runTestFMA(); }
  SECTION("test HalfFMA") { runTestHalfFMA(); }
}
