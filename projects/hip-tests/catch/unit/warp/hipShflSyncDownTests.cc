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

// For all threads in the warp, shfl the value "down" by three threads. To
// account for the end of the warp, we set the delta to zero near the warp-32
// boundary. This also works for warp-64 since it is a multiple.
template <typename T> __global__ void shflDown_1(T* Input, T* Output) {
  auto tid = threadIdx.x;
  int srcLane = (tid % 32 + 3 < 32) ? 3 : 0;
  Output[tid] = __shfl_down_sync(AllThreads, Input[tid], srcLane);
}

template <typename T> static void runTestShflDown_1() {
  const int size = 64;
  T Input[size];
  T Output[size];
  int Values[size] = {3,   4,   5,   -6,  7,   8,  -9, 10,  11,  12,  13,  -14, 15,  16, 17, -18,
                      19,  20,  -21, 22,  23,  24, 25, 26,  -27, 28,  29,  30,  31,  29, 30, 31,
                      35,  -36, 37,  38,  -39, 40, 41, 42,  43,  -44, -45, 46,  47,  48, 49, 50,
                      -51, 52,  53,  -54, 55,  56, 57, -58, 59,  60,  61,  62,  -63, 61, 62, -63};
  T Expected[size];

  initializeInput(Input, size);
  initializeExpected(Expected, Values, size);

  int warpSize = getWarpSize();

  T* d_Input;
  T* d_Output;
  HIP_CHECK(hipMalloc(&d_Input, sizeof(T) * size));
  HIP_CHECK(hipMalloc(&d_Output, sizeof(T) * size));

  HIP_CHECK(hipMemcpy(d_Input, &Input, sizeof(T) * size, hipMemcpyDefault));
  hipLaunchKernelGGL(shflDown_1<T>, 1, warpSize, 0, 0, d_Input, d_Output);

  HIP_CHECK(hipMemcpy(&Output, d_Output, sizeof(T) * size, hipMemcpyDefault));
  for (int i = 0; i != warpSize; ++i) {
    REQUIRE(compareEqual(Output[i], Expected[i]));
  }

  HIP_CHECK(hipFree(d_Input));
  HIP_CHECK(hipFree(d_Output));
}

// Use the mask argument to divide the warp into groups of 12 threads, and then
// shfl "down" by three threads. Account for the boundary within a group as well
// as withing a warp-32.

template <typename T> __global__ void shflDown_2(T* Input, T* Output) {
  auto tid = threadIdx.x;
  auto mask = __match_any_sync(AllThreads, tid / 12);
  int srcLane = ((tid % 32 + 3 >= 32) || (tid % 12 + 3 >= 12)) ? 0 : 3;
  Output[tid] = __shfl_down_sync(mask, Input[tid], srcLane);
}

template <typename T> static void runTestShflDown_2() {
  const int size = 64;
  T Input[size];
  T Output[size];
  int Values[size] = {3,   4,  5,   -6,  7,   8,   -9, 10,  11,  -9,  10,  11, 15,  16,  17, -18,
                      19,  20, -21, 22,  23,  -21, 22, 23,  -27, 28,  29,  30, 31,  29,  30, 31,
                      35,  33, 34,  35,  -39, 40,  41, 42,  43,  -44, -45, 46, 47,  -45, 46, 47,
                      -51, 52, 53,  -54, 55,  56,  57, -58, 59,  57,  -58, 59, -63, 61,  62, -63};
  T Expected[size];

  initializeInput(Input, size);
  initializeExpected(Expected, Values, size);

  int warpSize = getWarpSize();

  T* d_Input;
  T* d_Output;
  HIP_CHECK(hipMalloc(&d_Input, sizeof(T) * size));
  HIP_CHECK(hipMalloc(&d_Output, sizeof(T) * size));

  HIP_CHECK(hipMemcpy(d_Input, &Input, sizeof(T) * size, hipMemcpyDefault));
  hipLaunchKernelGGL(shflDown_2<T>, 1, warpSize, 0, 0, d_Input, d_Output);

  HIP_CHECK(hipMemcpy(&Output, d_Output, sizeof(T) * size, hipMemcpyDefault));
  for (int i = 0; i != warpSize; ++i) {
    REQUIRE(compareEqual(Output[i], Expected[i]));
  }

  HIP_CHECK(hipFree(d_Input));
  HIP_CHECK(hipFree(d_Output));
}

template <typename T> __global__ void shflDown_3(T* Input, T* Output) {
  auto tid = threadIdx.x;
  auto mask = __match_any_sync(AllThreads, tid / 12);
  int srcLane = ((tid % 12 + 3 >= 12) || (tid % 8 + 3 >= 8)) ? 0 : 3;
  Output[tid] = __shfl_down_sync(mask, Input[tid], srcLane, 8);
}

template <typename T> static void runTestShflDown_3() {
  const int size = 64;
  T Input[size];
  T Output[size];
  int Values[size] = {3, 4, 5, -6, 7, 5, -6, 7,          // cannot cross 8
                      11, -9, 10, 11,                    // cannot cross 12
                      15, 13, -14, 15,                   // cannot cross 8
                      19, 20, -21, 22, 23, -21, 22, 23,  // canot cross 12
                      // pattern repeats
                      -27, 28, 29, 30, 31, 29, 30, 31, 35, 33, 34, 35, -39, 37, 38, -39, 43, -44,
                      -45, 46, 47, -45, 46, 47,
                      // pattern repeats
                      -51, 52, 53, -54, 55, 53, -54, 55, 59, 57, -58, 59, -63, 61, 62, -63};
  T Expected[size];

  initializeInput(Input, size);
  initializeExpected(Expected, Values, size);

  int warpSize = getWarpSize();

  T* d_Input;
  T* d_Output;
  HIP_CHECK(hipMalloc(&d_Input, sizeof(T) * size));
  HIP_CHECK(hipMalloc(&d_Output, sizeof(T) * size));

  HIP_CHECK(hipMemcpy(d_Input, &Input, sizeof(T) * size, hipMemcpyDefault));
  hipLaunchKernelGGL(shflDown_3<T>, 1, warpSize, 0, 0, d_Input, d_Output);

  HIP_CHECK(hipMemcpy(&Output, d_Output, sizeof(T) * size, hipMemcpyDefault));
  for (int i = 0; i != warpSize; ++i) {
    REQUIRE(compareEqual(Output[i], Expected[i]));
  }

  HIP_CHECK(hipFree(d_Input));
  HIP_CHECK(hipFree(d_Output));
}

__global__ void shflDown_4(int* Input, int* Output) {
  auto tid = threadIdx.x;
  unsigned long long masks[2] = {Every5thBut9th, Every9thBit};

  Output[tid] = -1;
  if (tid % 5 == 0 || tid % 9 == 0)
    Output[tid] = __shfl_down_sync(masks[tid % 9 == 0], Input[tid], tid);
}

static void runTestShflDown_4() {
  size_t warpSize = getWarpSize();

  auto Input = std::vector<int>(warpSize);

  for (size_t i = 0; i < Input.size(); i++) {
    Input[i] = 0x55 * i;
  }

  auto Output = std::vector<int>(warpSize);
  auto Expected = std::vector<int>(warpSize);

  for (size_t i = 0; i < Expected.size() / 2; i++) {
    if (i % 5 == 0) {
      Expected[i] = 0x55 * (alignUp(i, 5) * 2);
    } else if (i % 9 == 0) {
      Expected[i] = 0x55 * (alignUp(i, 9) * 2);
    } else {
      Expected[i] = -1;
    }
  }

  for (size_t i = Expected.size() / 2; i < Expected.size(); i++) {
    if (i % 5 == 0) {
      Expected[i] = 0x55 * alignUp(i, 5);
    } else if (i % 9 == 0) {
      Expected[i] = 0x55 * alignUp(i, 9);
    } else {
      Expected[i] = -1;
    }
  }

  int* d_Input;
  int* d_Output;
  HIP_CHECK(hipMalloc(&d_Input, Input.size() * sizeof(Input[0])));
  HIP_CHECK(hipMalloc(&d_Output, Output.size() * sizeof(Output[0])));

  HIP_CHECK(hipMemcpy(d_Input, Input.data(), Input.size() * sizeof(Input[0]), hipMemcpyDefault));
  hipLaunchKernelGGL(shflDown_4, 1, warpSize, 0, 0, d_Input, d_Output);

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
 * `T  __shfl_down_sync(unsigned long long mask, T var, int delta, int width=warpSize)` -
 * Contains warp __shfl sync functions.
 * @}
 */

/**
 * Test Description
 * ------------------------
 * - Test case to verify __shfl_down_sync warp functions for different datatypes.

 * Test source
 * ------------------------
 *    - catch/unit/kernel/hipShflSyncDownTests.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */

TEST_CASE("Unit_hipShflSync_Down") {
  CHECK_WARP_MATCH_FUNCTIONS_SUPPORT

  SECTION("run test for short") {
    runTestShflDown_1<short>();
    runTestShflDown_2<short>();
    runTestShflDown_3<short>();
  }
  SECTION("run test for unsigned short") {
    runTestShflDown_1<unsigned short>();
    runTestShflDown_2<unsigned short>();
    runTestShflDown_3<unsigned short>();
  }
  SECTION("run test for int") {
    runTestShflDown_1<int>();
    runTestShflDown_2<int>();
    runTestShflDown_3<int>();
  }
  SECTION("run test for unsigned int") {
    runTestShflDown_1<unsigned int>();
    runTestShflDown_2<unsigned int>();
    runTestShflDown_3<unsigned int>();
  }
  SECTION("run test for long") {
    runTestShflDown_1<long>();
    runTestShflDown_2<long>();
    runTestShflDown_3<long>();
  }
  SECTION("run test for unsigned long") {
    runTestShflDown_1<unsigned long>();
    runTestShflDown_2<unsigned long>();
    runTestShflDown_3<unsigned long>();
  }
  SECTION("run test for long long") {
    runTestShflDown_1<long long>();
    runTestShflDown_2<long long>();
    runTestShflDown_3<long long>();
  }
  SECTION("run test for unsigned long long") {
    runTestShflDown_1<unsigned long long>();
    runTestShflDown_2<unsigned long long>();
    runTestShflDown_3<unsigned long long>();
  }
  SECTION("run test for float") {
    runTestShflDown_1<float>();
    runTestShflDown_2<float>();
    runTestShflDown_3<float>();
  }
  SECTION("run test for double") {
    runTestShflDown_1<double>();
    runTestShflDown_2<double>();
    runTestShflDown_3<double>();
  }
  SECTION("run test for __half") {
    runTestShflDown_1<__half>();
    runTestShflDown_2<__half>();
    runTestShflDown_3<__half>();
  }
  SECTION("run test for __half2") {
    runTestShflDown_1<__half2>();
    runTestShflDown_2<__half2>();
    runTestShflDown_3<__half2>();
  }
  SECTION("run divergent execution tests") { runTestShflDown_4(); }
}
