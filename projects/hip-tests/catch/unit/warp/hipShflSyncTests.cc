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

template <typename T> __global__ void shfl_1(T* Input, T* Output) {
  int tid = threadIdx.x;
  // Creates groups consisting of every fourth thread.
  auto mask = __match_any_sync(AllThreads, tid % 4);
  int srcLane = tid % 4;

  // Each group reads from the first active thread within that group.
  Output[tid] = __shfl_sync(mask, Input[tid], srcLane);
}

template <typename T> static void runTestShfl_1() {
  const int size = 64;
  T Input[size];
  T Output[size];
  T Expected[size];
  int Values[size] = {0, -1, 2, 3,  0, -1, 2, 3,  0, -1, 2, 3,  0, -1, 2, 3,  0, -1, 2, 3,  0, -1,
                      2, 3,  0, -1, 2, 3,  0, -1, 2, 3,  0, -1, 2, 3,  0, -1, 2, 3,  0, -1, 2, 3,
                      0, -1, 2, 3,  0, -1, 2, 3,  0, -1, 2, 3,  0, -1, 2, 3,  0, -1, 2, 3};

  initializeInput(Input, size);
  initializeExpected(Expected, Values, size);

  int warpSize = getWarpSize();

  T* d_Input;
  T* d_Output;
  HIP_CHECK(hipMalloc(&d_Input, sizeof(T) * size));
  HIP_CHECK(hipMalloc(&d_Output, sizeof(T) * size));

  HIP_CHECK(hipMemcpy(d_Input, &Input, sizeof(T) * size, hipMemcpyDefault));
  hipLaunchKernelGGL(shfl_1<T>, 1, warpSize, 0, 0, d_Input, d_Output);

  HIP_CHECK(hipMemcpy(&Output, d_Output, sizeof(T) * size, hipMemcpyDefault));
  for (int i = 0; i != warpSize; ++i) {
    REQUIRE(compareEqual(Output[i], Expected[i]));
  }

  HIP_CHECK(hipFree(d_Input));
  HIP_CHECK(hipFree(d_Output));
}

template <typename T> __global__ void shfl_2(T* Input, T* Output) {
  int tid = threadIdx.x;
  auto mask = __match_any_sync(AllThreads, tid % 4);
  int srcLane = tid % 4;

  // Each subgroup of eight reads from the first active thread within that
  // subgroup.
  Output[tid] = __shfl_sync(mask, Input[tid], srcLane, 8);
}

template <typename T> static void runTestShfl_2() {
  const int size = 64;
  T Input[size];
  T Output[size];
  T Expected[size];
  int Values[size] = {0,   -1, 2,   3,   0,   -1, 2,   3,   8,  -9, 10,  11,  8,  -9, 10,  11,
                      16,  17, -18, 19,  16,  17, -18, 19,  24, 25, 26,  -27, 24, 25, 26,  -27,
                      -32, 33, 34,  35,  -32, 33, 34,  35,  40, 41, 42,  43,  40, 41, 42,  43,
                      48,  49, 50,  -51, 48,  49, 50,  -51, 56, 57, -58, 59,  56, 57, -58, 59};

  initializeInput(Input, size);
  initializeExpected(Expected, Values, size);

  int warpSize = getWarpSize();

  T* d_Input;
  T* d_Output;
  HIP_CHECK(hipMalloc(&d_Input, sizeof(T) * size));
  HIP_CHECK(hipMalloc(&d_Output, sizeof(T) * size));

  HIP_CHECK(hipMemcpy(d_Input, &Input, sizeof(T) * size, hipMemcpyDefault));
  hipLaunchKernelGGL(shfl_2<T>, 1, warpSize, 0, 0, d_Input, d_Output);

  HIP_CHECK(hipMemcpy(&Output, d_Output, sizeof(T) * size, hipMemcpyDefault));
  for (int i = 0; i != warpSize; ++i) {
    REQUIRE(compareEqual(Output[i], Expected[i]));
  }

  HIP_CHECK(hipFree(d_Input));
  HIP_CHECK(hipFree(d_Output));
}

__global__ void shfl_3(int* Input, int* Output) {
  auto tid = threadIdx.x;
  unsigned long long masks[2] = {Every5thBut9th, Every9thBit};

  Output[tid] = -1;
  if (tid % 5 == 0 || tid % 9 == 0) Output[tid] = __shfl_sync(masks[tid % 9 == 0], Input[tid], tid);
}

static void runTestShfl_3() {
  size_t warpSize = getWarpSize();

  auto Input = std::vector<int>(warpSize);

  for (size_t i = 0; i < Input.size(); i++) {
    Input[i] = i;
  }

  auto Output = std::vector<int>(warpSize);
  auto Expected = std::vector<int>(warpSize);

  for (size_t i = 0; i < Expected.size(); i++) {
    if (i % 9 == 0 || i % 5 == 0) {
      Expected[i] = i;
    } else {
      Expected[i] = -1;
    }
  }

  int* d_Input;
  int* d_Output;
  HIP_CHECK(hipMalloc(&d_Input, Input.size() * sizeof(Input[0])));
  HIP_CHECK(hipMalloc(&d_Output, Output.size() * sizeof(Output[0])));

  HIP_CHECK(hipMemcpy(d_Input, Input.data(), Input.size() * sizeof(Input[0]), hipMemcpyDefault));
  hipLaunchKernelGGL(shfl_3, 1, warpSize, 0, 0, d_Input, d_Output);

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
 * `T  __shfl_sync(unsigned long long mask, T var, int srcLane, int width=warpSize)` -
 * Contains warp __shfl sync functions.
 * @}
 */

/**
 * Test Description
 * ------------------------
 * - Test case to verify __shfl_sync warp functions for different datatypes.

 * Test source
 * ------------------------
 *    - catch/unit/kernel/hipShflSyncTests.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */

TEST_CASE("Unit_hipShflSync") {
  CHECK_WARP_MATCH_FUNCTIONS_SUPPORT

  SECTION("run test for short") {
    runTestShfl_1<short>();
    runTestShfl_2<short>();
  }
  SECTION("run test for unsigned short") {
    runTestShfl_1<unsigned short>();
    runTestShfl_2<unsigned short>();
  }
  SECTION("run test for int") {
    runTestShfl_1<int>();
    runTestShfl_2<int>();
  }
  SECTION("run test for unsigned int") {
    runTestShfl_1<unsigned int>();
    runTestShfl_2<unsigned int>();
  }
  SECTION("run test for long") {
    runTestShfl_1<long>();
    runTestShfl_2<long>();
  }
  SECTION("run test for unsigned long") {
    runTestShfl_1<unsigned long>();
    runTestShfl_2<unsigned long>();
  }
  SECTION("run test for long long") {
    runTestShfl_1<long long>();
    runTestShfl_2<long long>();
  }
  SECTION("run test for unsigned long long") {
    runTestShfl_1<unsigned long long>();
    runTestShfl_2<unsigned long long>();
  }
  SECTION("run test for float") {
    runTestShfl_1<float>();
    runTestShfl_2<float>();
  }
  SECTION("run test for double") {
    runTestShfl_1<double>();
    runTestShfl_2<double>();
  }
  SECTION("divergent execution test") { runTestShfl_3(); }
}
