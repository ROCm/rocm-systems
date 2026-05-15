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

template <typename T> __global__ void shflUp_1(T* Input, T* Output) {
  auto tid = threadIdx.x;
  int srcLane = (tid > 3) ? 3 : 0;
  Output[tid] = __shfl_up_sync(AllThreads, Input[tid], srcLane);
}

template <typename T> static void runTestShflUp_1() {
  const int size = 64;
  T Input[size];
  T Output[size];
  T Expected[size];
  int Values[] = {0,   -1,  2,  3,   -1, 2,   3,   4,   5,   -6,  7,   8,  -9, 10,  11,  12,
                  13,  -14, 15, 16,  17, -18, 19,  20,  -21, 22,  23,  24, 25, 26,  -27, 28,
                  29,  30,  31, -32, 33, 34,  35,  -36, 37,  38,  -39, 40, 41, 42,  43,  -44,
                  -45, 46,  47, 48,  49, 50,  -51, 52,  53,  -54, 55,  56, 57, -58, 59,  60};

  initializeInput(Input, size);
  initializeExpected(Expected, Values, size);

  int warpSize = getWarpSize();

  T* d_Input;
  T* d_Output;
  HIP_CHECK(hipMalloc(&d_Input, sizeof(T) * size));
  HIP_CHECK(hipMalloc(&d_Output, sizeof(T) * size));

  HIP_CHECK(hipMemcpy(d_Input, &Input, sizeof(T) * size, hipMemcpyDefault));
  hipLaunchKernelGGL(shflUp_1<T>, 1, warpSize, 0, 0, d_Input, d_Output);

  HIP_CHECK(hipMemcpy(&Output, d_Output, sizeof(T) * size, hipMemcpyDefault));
  for (int i = 0; i != warpSize; ++i) {
    REQUIRE(compareEqual(Output[i], Expected[i]));
  }

  HIP_CHECK(hipFree(d_Input));
  HIP_CHECK(hipFree(d_Output));
}

template <typename T> __global__ void shflUp_2(T* Input, T* Output) {
  auto tid = threadIdx.x;
  auto mask = __match_any_sync(AllThreads, tid / 12);
  int srcLane = (tid % 12) < 3 ? 0 : 3;
  Output[tid] = __shfl_up_sync(mask, Input[tid], srcLane);
}

template <typename T> static void runTestShflUp_2() {
  const int size = 64;
  T Input[size];
  T Output[size];
  T Expected[size];
  int Values[size] = {0,  -1,  2,  0,   -1,  2,   3,   4,   5,  -6,  7,   8,  12, 13, -14, 12,
                      13, -14, 15, 16,  17,  -18, 19,  20,  24, 25,  26,  24, 25, 26, -27, 28,
                      29, 30,  31, -32, -36, 37,  38,  -36, 37, 38,  -39, 40, 41, 42, 43,  -44,
                      48, 49,  50, 48,  49,  50,  -51, 52,  53, -54, 55,  56, 60, 61, 62,  60};

  initializeInput(Input, size);
  initializeExpected(Expected, Values, size);

  int warpSize = getWarpSize();

  T* d_Input;
  T* d_Output;
  HIP_CHECK(hipMalloc(&d_Input, sizeof(T) * size));
  HIP_CHECK(hipMalloc(&d_Output, sizeof(T) * size));

  HIP_CHECK(hipMemcpy(d_Input, &Input, sizeof(T) * size, hipMemcpyDefault));
  hipLaunchKernelGGL(shflUp_2<T>, 1, warpSize, 0, 0, d_Input, d_Output);

  HIP_CHECK(hipMemcpy(&Output, d_Output, sizeof(T) * size, hipMemcpyDefault));
  for (int i = 0; i != warpSize; ++i) {
    REQUIRE(compareEqual(Output[i], Expected[i]));
  }

  HIP_CHECK(hipFree(d_Input));
  HIP_CHECK(hipFree(d_Output));
}

template <typename T> __global__ void shflUp_3(T* Input, T* Output) {
  auto tid = threadIdx.x;
  auto mask = __match_any_sync(AllThreads, tid / 12);
  int srcLane = (tid % 12) < 3 ? 0 : 3;
  Output[tid] = __shfl_up_sync(mask, Input[tid], srcLane, 8);
}

template <typename T> static void runTestShflUp_3() {
  const int size = 64;
  T Input[size];
  T Output[size];
  T Expected[size];
  int Values[size] = {0, -1, 2,                   // cannot cross mod-12
                      0, -1, 2, 3, 4, 8, -9, 10,  // cannot cross mod-8
                      8, 12, 13, -14,             // cannot cross mod-12
                      12, 16, 17, -18,            // cannot cross mod-8
                      16, 17, -18, 19, 20,
                      // pattern repeats
                      24, 25, 26, 24, 25, 26, -27, 28, -32, 33, 34, -32, -36, 37, 38, -36, 40, 41,
                      42, 40, 41, 42, 43, -44,
                      // pattern repeats
                      48, 49, 50, 48, 49, 50, -51, 52, 56, 57, -58, 56, 60, 61, 62, 60};

  initializeInput(Input, size);
  initializeExpected(Expected, Values, size);

  int warpSize = getWarpSize();

  T* d_Input;
  T* d_Output;
  HIP_CHECK(hipMalloc(&d_Input, sizeof(T) * size));
  HIP_CHECK(hipMalloc(&d_Output, sizeof(T) * size));

  HIP_CHECK(hipMemcpy(d_Input, &Input, sizeof(T) * size, hipMemcpyDefault));
  hipLaunchKernelGGL(shflUp_3<T>, 1, warpSize, 0, 0, d_Input, d_Output);

  HIP_CHECK(hipMemcpy(&Output, d_Output, sizeof(T) * size, hipMemcpyDefault));
  for (int i = 0; i != warpSize; ++i) {
    REQUIRE(compareEqual(Output[i], Expected[i]));
  }

  HIP_CHECK(hipFree(d_Input));
  HIP_CHECK(hipFree(d_Output));
}

__global__ void shflUp_4(int* Input, int* Output) {
  auto tid = threadIdx.x;
  unsigned long long masks[2] = {Every5thBut9th, Every9thBit};

  Output[tid] = -1;
  if (tid % 5 == 0 || tid % 9 == 0)
    Output[tid] = __shfl_up_sync(masks[tid % 9 == 0], Input[tid], tid);
}

static void runTestShflUp_4() {
  size_t warpSize = getWarpSize();

  auto Input = std::vector<int>(warpSize);

  for (size_t i = 0; i < Input.size(); i++) {
    Input[i] = 0x55 * (i + 1);
  }

  auto Output = std::vector<int>(warpSize);
  auto Expected = std::vector<int>(warpSize);

  for (size_t i = 0; i < Expected.size(); i++) {
    if (i % 9 == 0 || i % 5 == 0) {
      Expected[i] = 0x55;
    } else {
      Expected[i] = -1;
    }
  }

  int* d_Input;
  int* d_Output;
  HIP_CHECK(hipMalloc(&d_Input, Input.size() * sizeof(Input[0])));
  HIP_CHECK(hipMalloc(&d_Output, Output.size() * sizeof(Output[0])));

  HIP_CHECK(hipMemcpy(d_Input, Input.data(), Input.size() * sizeof(Input[0]), hipMemcpyDefault));
  hipLaunchKernelGGL(shflUp_4, 1, warpSize, 0, 0, d_Input, d_Output);

  HIP_CHECK(
      hipMemcpy(Output.data(), d_Output, Output.size() * sizeof(Output[0]), hipMemcpyDefault));
  for (size_t i = 0; i < Output.size(); i++) {
    REQUIRE(Output[i] == Expected[i]);
  }

  HIP_CHECK(hipFree(d_Input));
  HIP_CHECK(hipFree(d_Output));
}

/**
 * @addtogroup __shfl_sync
 * @{
 * @ingroup ShflSyncTest
 * `T  __shfl_up_sync(unsigned long long mask, T var, int delta, int width=warpSize)` -
 * Contains warp __shfl sync functions.
 * @}
 */

/**
 * Test Description
 * ------------------------
 * - Test case to verify __shfl_up_sync warp functions for different datatypes.

 * Test source
 * ------------------------
 *    - catch/unit/kernel/hipShflSyncUpTests.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */

TEST_CASE("Unit_hipShflSync_Up") {
  CHECK_WARP_MATCH_FUNCTIONS_SUPPORT

  SECTION("run test for short") {
    runTestShflUp_1<short>();
    runTestShflUp_2<short>();
    runTestShflUp_3<short>();
  }
  SECTION("run test for unsigned short") {
    runTestShflUp_1<unsigned short>();
    runTestShflUp_2<unsigned short>();
    runTestShflUp_3<unsigned short>();
  }
  SECTION("run test for int") {
    runTestShflUp_1<int>();
    runTestShflUp_2<int>();
    runTestShflUp_3<int>();
  }
  SECTION("run test for unsigned int") {
    runTestShflUp_1<unsigned int>();
    runTestShflUp_2<unsigned int>();
    runTestShflUp_3<unsigned int>();
  }
  SECTION("run test for long") {
    runTestShflUp_1<long>();
    runTestShflUp_2<long>();
    runTestShflUp_3<long>();
  }
  SECTION("run test for unsigned long") {
    runTestShflUp_1<unsigned long>();
    runTestShflUp_2<unsigned long>();
    runTestShflUp_3<unsigned long>();
  }
  SECTION("run test for long long") {
    runTestShflUp_1<long long>();
    runTestShflUp_2<long long>();
    runTestShflUp_3<long long>();
  }
  SECTION("run test for unsigned long long") {
    runTestShflUp_1<unsigned long long>();
    runTestShflUp_2<unsigned long long>();
    runTestShflUp_3<unsigned long long>();
  }
  SECTION("run test for float") {
    runTestShflUp_1<float>();
    runTestShflUp_2<float>();
    runTestShflUp_3<float>();
  }
  SECTION("run test for double") {
    runTestShflUp_1<double>();
    runTestShflUp_2<double>();
    runTestShflUp_3<double>();
  }
  SECTION("run test for __half") {
    runTestShflUp_1<__half>();
    runTestShflUp_2<__half>();
    runTestShflUp_3<__half>();
  }
  SECTION("run test for __half2") {
    runTestShflUp_1<__half2>();
    runTestShflUp_2<__half2>();
    runTestShflUp_3<__half2>();
  }
  SECTION("run divergent execution test") { runTestShflUp_4(); }
}
