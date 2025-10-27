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

#include "warp_common.hh"
#include <hip_test_common.hh>

template <typename T>
__global__ void matchAll_1(T* Input, unsigned long long* Output, int* Predicate) {
  auto tid = threadIdx.x;

  Output[tid] = __match_all_sync(AllThreads, Input[tid], &Predicate[tid]);
}

template <typename T> static void runTestMatchAll_1() {
  const int size = 64;
  T Input[size] = {
      (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5,
      (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5,
      (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5,
      (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5,
      (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5,
  };
  unsigned long long Output[size];
  unsigned long long Expected[size];
  std::fill_n(Expected, size, -1);

  int Predicate[size];
  int ExpPredicate[size];
  std::fill_n(ExpPredicate, size, true);

  expandPrecision(Input, size);

  int warpSize = getWarpSize();

  T* d_Input;
  unsigned long long* d_Output;
  int* d_Predicate;
  HIP_CHECK(hipMalloc(&d_Input, sizeof(T) * size));
  HIP_CHECK(hipMalloc(&d_Output, 8 * size));
  HIP_CHECK(hipMalloc(&d_Predicate, 4 * size));

  HIP_CHECK(hipMemcpy(d_Input, &Input, sizeof(T) * size, hipMemcpyDefault));
  hipLaunchKernelGGL(matchAll_1<T>, 1, warpSize, 0, 0, d_Input, d_Output, d_Predicate);

  HIP_CHECK(hipMemcpy(&Output, d_Output, 8 * size, hipMemcpyDefault));
  for (int i = 0; i != warpSize; ++i) {
    REQUIRE(compareMaskEqual(Output, Expected, i, warpSize));
  }

  HIP_CHECK(hipMemcpy(&Predicate, d_Predicate, 4 * size, hipMemcpyDefault));
  for (int i = 0; i != warpSize; ++i) {
    REQUIRE(Predicate[i] == ExpPredicate[i]);
  }

  HIP_CHECK(hipFree(d_Input));
  HIP_CHECK(hipFree(d_Output));
  HIP_CHECK(hipFree(d_Predicate));
}

template <typename T>
__global__ void matchAll_2(T* Input, unsigned long long* Output, int* Predicate) {
  auto tid = threadIdx.x;

  Output[tid] = __match_all_sync(AllThreads, Input[tid], &Predicate[tid]);
}

template <typename T> static void runTestMatchAll_2() {
  const int size = 64;
  T Input[size] = {
      (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5,   (T)-5, (T)-5, (T)-5, (T)-5, (T)-5,
      (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5,   (T)-5, (T)-5, (T)-5, (T)-5, (T)-5,
      (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5,   (T)-5, (T)-5, (T)-5, (T)-5, (T)-5,
      (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5,   (T)-5, (T)-5, (T)-5, (T)-5, (T)-5,
      (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-500, (T)-5, (T)-5, (T)-5, (T)-5,
  };
  unsigned long long Output[size];
  unsigned long long Expected[size];

  int warpSize = getWarpSize();

  if (warpSize == 32)
    std::fill_n(Expected, size, -1);
  else
    std::fill_n(Expected, size, 0);

  int Predicate[size];
  int ExpPredicate[size];
  if (warpSize == 32)
    std::fill_n(ExpPredicate, size, true);
  else
    std::fill_n(ExpPredicate, size, false);

  expandPrecision(Input, size);

  T* d_Input;
  unsigned long long* d_Output;
  int* d_Predicate;
  HIP_CHECK(hipMalloc(&d_Input, sizeof(T) * size));
  HIP_CHECK(hipMalloc(&d_Output, 8 * size));
  HIP_CHECK(hipMalloc(&d_Predicate, 4 * size));

  HIP_CHECK(hipMemcpy(d_Input, &Input, sizeof(T) * size, hipMemcpyDefault));
  hipLaunchKernelGGL(matchAll_2<T>, 1, warpSize, 0, 0, d_Input, d_Output, d_Predicate);

  HIP_CHECK(hipMemcpy(&Output, d_Output, 8 * size, hipMemcpyDefault));
  for (int i = 0; i != warpSize; ++i) {
    REQUIRE(compareMaskEqual(Output, Expected, i, warpSize));
  }

  HIP_CHECK(hipMemcpy(&Predicate, d_Predicate, 4 * size, hipMemcpyDefault));
  for (int i = 0; i != warpSize; ++i) {
    REQUIRE(Predicate[i] == ExpPredicate[i]);
  }

  HIP_CHECK(hipFree(d_Input));
  HIP_CHECK(hipFree(d_Output));
  HIP_CHECK(hipFree(d_Predicate));
}

template <typename T>
__global__ void matchAll_3(T* Input, unsigned long long* Output, int* Predicate) {
  auto tid = threadIdx.x;
  // It's okay to use the non-sync__ match, because the purpose of the test is
  // to exercise the mask argument on the sync version.
  auto mask = __match_any_sync(AllThreads, tid / 12);

  Output[tid] = __match_all_sync(mask, Input[tid], &Predicate[tid]);
}

template <typename T> static void runTestMatchAll_3() {
  const int size = 64;
  T Input[size] = {
      (T)-5, (T)-5,   (T)-5, (T)-5, (T)-5,   (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5,
      (T)-5, (T)-500, (T)-5, (T)-5, (T)-5,   (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5,
      (T)-5, (T)-5,   (T)-5, (T)-5, (T)-5,   (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5,
      (T)-5, (T)-5,   (T)-5, (T)-5, (T)-500, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5,
      (T)-5, (T)-5,   (T)-5, (T)-5, (T)-5,   (T)-5, (T)-5, (T)-5, (T)-5, (T)-5, (T)-5,
      (T)-5, (T)-5,   (T)-5, (T)-5, (T)-500, (T)-5, (T)-5, (T)-5, (T)-5,
  };
  unsigned long long Output[size];
  unsigned long long Expected[size] = {0xfff,
                                       0xfff,
                                       0xfff,
                                       0xfff,
                                       0xfff,
                                       0xfff,
                                       0xfff,
                                       0xfff,
                                       0xfff,
                                       0xfff,
                                       0xfff,
                                       0xfff,
                                       0,
                                       0,
                                       0,
                                       0,
                                       0,
                                       0,
                                       0,
                                       0,
                                       0,
                                       0,
                                       0,
                                       0,
                                       0xfff000000,
                                       0xfff000000,
                                       0xfff000000,
                                       0xfff000000,
                                       0xfff000000,
                                       0xfff000000,
                                       0xfff000000,
                                       0xfff000000,
                                       0xfff000000,
                                       0xfff000000,
                                       0xfff000000,
                                       0xfff000000,
                                       0,
                                       0,
                                       0,
                                       0,
                                       0,
                                       0,
                                       0,
                                       0,
                                       0,
                                       0,
                                       0,
                                       0,
                                       0,
                                       0,
                                       0,
                                       0,
                                       0,
                                       0,
                                       0,
                                       0,
                                       0,
                                       0,
                                       0,
                                       0,
                                       0xf000000000000000,
                                       0xf000000000000000,
                                       0xf000000000000000,
                                       0xf000000000000000};

  int Predicate[size];
  int ExpPredicate[size]{
      true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  false,
      false, false, false, false, false, false, false, false, false, false, false, true,  true,
      true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  false, false, false,
      false, false, false, false, false, false, false, false, false, false, false, false, false,
      false, false, false, false, false, false, false, false, true,  true,  true,  true};

  expandPrecision(Input, size);

  int warpSize = getWarpSize();

  T* d_Input;
  unsigned long long* d_Output;
  int* d_Predicate;
  HIP_CHECK(hipMalloc(&d_Input, sizeof(T) * size));
  HIP_CHECK(hipMalloc(&d_Output, 8 * size));
  HIP_CHECK(hipMalloc(&d_Predicate, 4 * size));

  HIP_CHECK(hipMemcpy(d_Input, &Input, sizeof(T) * size, hipMemcpyDefault));
  hipLaunchKernelGGL(matchAll_3<T>, 1, warpSize, 0, 0, d_Input, d_Output, d_Predicate);

  HIP_CHECK(hipMemcpy(&Output, d_Output, 8 * size, hipMemcpyDefault));
  for (int i = 0; i != warpSize; ++i) {
    REQUIRE(compareMaskEqual(Output, Expected, i, warpSize));
  }

  HIP_CHECK(hipMemcpy(&Predicate, d_Predicate, 4 * size, hipMemcpyDefault));
  for (int i = 0; i != warpSize; ++i) {
    REQUIRE(Predicate[i] == ExpPredicate[i]);
  }

  HIP_CHECK(hipFree(d_Input));
  HIP_CHECK(hipFree(d_Output));
  HIP_CHECK(hipFree(d_Predicate));
}

__global__ void matchAll_4(int* Input, int* Output) {
  int tid = threadIdx.x;
  unsigned long long masks[2] = {Every5thBut9th, Every9thBit};

  Output[tid] = -1;
  if (tid % 5 == 0 || tid % 9 == 0) {
    Output[tid] = !!__match_all_sync(masks[tid % 9 == 0], Input[tid], &Input[tid]);
  }
}

static void runTestMatchAll_4() {
  size_t warpSize = getWarpSize();

  auto Input = std::vector<int>(warpSize);

  for (size_t i = 0; i < Input.size(); i++) {
    if (i % 9 == 0 || i % 5 == 0) {
      Input[i] = 0x55;
    } else {
      Input[i] = i;
    }
  }

  auto Output = std::vector<int>(warpSize);
  auto Expected = std::vector<int>(warpSize);

  for (size_t i = 0; i < Expected.size(); i++) {
    if (i % 9 == 0 || i % 5 == 0) {
      Expected[i] = 1;
    } else {
      Expected[i] = -1;
    }
  }

  int* d_Input;
  int* d_Output;
  HIP_CHECK(hipMalloc(&d_Input, Input.size() * sizeof(Input[0])));
  HIP_CHECK(hipMalloc(&d_Output, Output.size() * sizeof(Output[0])));

  HIP_CHECK(hipMemcpy(d_Input, Input.data(), Input.size() * sizeof(Input[0]), hipMemcpyDefault));
  hipLaunchKernelGGL(matchAll_4, 1, warpSize, 0, 0, d_Input, d_Output);

  HIP_CHECK(
      hipMemcpy(Output.data(), d_Output, Output.size() * sizeof(Output[0]), hipMemcpyDefault));
  for (size_t i = 0; i < Output.size(); i++) {
    REQUIRE(Output[i] == Expected[i]);
  }

  HIP_CHECK(hipFree(d_Input));
  HIP_CHECK(hipFree(d_Output));
}

/**
 * @addtogroup __match_sync
 * @{
 * @ingroup MatchSyncTest
 * `unsigned long long __match_all_sync(unsigned long long mask, T value, int *pred)` -
 * Contains warp __match sync functions.
 * @}
 */

/**
 * Test Description
 * ------------------------
 * - Test case to verify __match_sync warp functions for different datatypes.

 * Test source
 * ------------------------
 *    - catch/unit/kernel/hipMatchSyncAllTests.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */

TEST_CASE("Unit_hipMatchSync_All") {
  CHECK_WARP_MATCH_FUNCTIONS_SUPPORT

  SECTION("run test for int") {
    runTestMatchAll_1<int>();
    runTestMatchAll_2<int>();
    runTestMatchAll_3<int>();
  }
  SECTION("run test for unsigned int") {
    runTestMatchAll_1<unsigned int>();
    runTestMatchAll_2<unsigned int>();
    runTestMatchAll_3<unsigned int>();
  }
  SECTION("run test for long") {
    runTestMatchAll_1<long>();
    runTestMatchAll_2<long>();
    runTestMatchAll_3<long>();
  }
  SECTION("run test for unsigned long") {
    runTestMatchAll_1<unsigned long>();
    runTestMatchAll_2<unsigned long>();
    runTestMatchAll_3<unsigned long>();
  }
  SECTION("run test for long long") {
    runTestMatchAll_1<long long>();
    runTestMatchAll_2<long long>();
    runTestMatchAll_3<long long>();
  }
  SECTION("run test for unsigned long long") {
    runTestMatchAll_1<unsigned long long>();
    runTestMatchAll_2<unsigned long long>();
    runTestMatchAll_3<unsigned long long>();
  }
  SECTION("run test for float") {
    runTestMatchAll_1<float>();
    runTestMatchAll_2<float>();
    runTestMatchAll_3<float>();
  }
  SECTION("run test for double") {
    runTestMatchAll_1<double>();
    runTestMatchAll_2<double>();
    runTestMatchAll_3<double>();
  }
  SECTION("run divergent execution tests") { runTestMatchAll_4(); }
}
