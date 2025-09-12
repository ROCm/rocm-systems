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

template <typename T> __global__ void shflXor_1(T* Input, T* Output) {
  auto tid = threadIdx.x;
  Output[tid] = __shfl_xor_sync(AllThreads, Input[tid], 16);
}

template <typename T> static void runTestShflXor_1() {
  const int size = 64;
  T Input[size];
  T Output[size];
  int Values[size] = {16,  17, -18, 19,  20,  -21, 22,  23,  24, 25, 26,  -27, 28,  29,  30,  31,
                      0,   -1, 2,   3,   4,   5,   -6,  7,   8,  -9, 10,  11,  12,  13,  -14, 15,
                      48,  49, 50,  -51, 52,  53,  -54, 55,  56, 57, -58, 59,  60,  61,  62,  -63,
                      -32, 33, 34,  35,  -36, 37,  38,  -39, 40, 41, 42,  43,  -44, -45, 46,  47};
  T Expected[size];

  initializeInput(Input, size);
  initializeExpected(Expected, Values, size);

  int warpSize = getWarpSize();

  T* d_Input;
  T* d_Output;
  HIP_CHECK(hipMalloc(&d_Input, sizeof(T) * size));
  HIP_CHECK(hipMalloc(&d_Output, sizeof(T) * size));

  HIP_CHECK(hipMemcpy(d_Input, &Input, sizeof(T) * size, hipMemcpyDefault));
  hipLaunchKernelGGL(shflXor_1<T>, 1, warpSize, 0, 0, d_Input, d_Output);

  HIP_CHECK(hipMemcpy(&Output, d_Output, sizeof(T) * size, hipMemcpyDefault));
  for (int i = 0; i != warpSize; ++i) {
    REQUIRE(compareEqual(Output[i], Expected[i]));
  }

  HIP_CHECK(hipFree(d_Input));
  HIP_CHECK(hipFree(d_Output));
}

template <typename T> __global__ void shflXor_2(T* Input, T* Output) {
  unsigned tid = threadIdx.x;
  auto mask = __match_any_sync(AllThreads, tid / 12);
  int laneMask = 4;
  int section = tid % 24;
  if (section > 7 && section < 16) laneMask = 0;
  Output[tid] = __shfl_xor_sync(mask, Input[tid], laneMask);
}

template <typename T> static void runTestShflXor_2() {
  const int size = 64;
  T Input[size];
  T Output[size];
  int Values[size] = {4, 5, -6, 7, 0, -1, 2, 3, 8, -9, 10, 11, 12, 13, -14, 15,  // disabled around
                                                                                 // mid mod-24
                      20, -21, 22, 23, 16, 17, -18, 19, 28, 29, 30, 31, 24, 25, 26, -27, -32, 33,
                      34, 35, -36, 37, 38, -39,  // disabled around mid mod-24
                      -44, -45, 46, 47, 40, 41, 42, 43, 52, 53, -54, 55, 48, 49, 50, -51, 56, 57,
                      -58, 59, 60, 61, 62, -63};  // disabled around mid mod-24
  T Expected[size];

  initializeInput(Input, size);
  initializeExpected(Expected, Values, size);


  int warpSize = getWarpSize();

  T* d_Input;
  T* d_Output;
  HIP_CHECK(hipMalloc(&d_Input, sizeof(T) * size));
  HIP_CHECK(hipMalloc(&d_Output, sizeof(T) * size));

  HIP_CHECK(hipMemcpy(d_Input, &Input, sizeof(T) * size, hipMemcpyDefault));
  hipLaunchKernelGGL(shflXor_2<T>, 1, warpSize, 0, 0, d_Input, d_Output);

  HIP_CHECK(hipMemcpy(&Output, d_Output, sizeof(T) * size, hipMemcpyDefault));
  for (int i = 0; i != warpSize; ++i) {
    REQUIRE(compareEqual(Output[i], Expected[i]));
  }

  HIP_CHECK(hipFree(d_Input));
  HIP_CHECK(hipFree(d_Output));
}

template <typename T> __global__ void shflXor_3(T* Input, T* Output) {
  auto tid = threadIdx.x;
  auto mask = __match_any_sync(AllThreads, tid / 16);
  Output[tid] = __shfl_xor_sync(mask, Input[tid], 4, 8);
}

template <typename T> static void runTestShflXor_3() {
  const int size = 64;
  T Input[size];
  T Output[size];
  int Values[size] = {4,   5,   -6,  7,   0,   -1, 2,   3,   12,  13,  -14, 15,  8,  -9, 10,  11,
                      20,  -21, 22,  23,  16,  17, -18, 19,  28,  29,  30,  31,  24, 25, 26,  -27,
                      -36, 37,  38,  -39, -32, 33, 34,  35,  -44, -45, 46,  47,  40, 41, 42,  43,
                      52,  53,  -54, 55,  48,  49, 50,  -51, 60,  61,  62,  -63, 56, 57, -58, 59};
  T Expected[size];

  initializeInput(Input, size);
  initializeExpected(Expected, Values, size);


  int warpSize = getWarpSize();

  T* d_Input;
  T* d_Output;
  HIP_CHECK(hipMalloc(&d_Input, sizeof(T) * size));
  HIP_CHECK(hipMalloc(&d_Output, sizeof(T) * size));

  HIP_CHECK(hipMemcpy(d_Input, &Input, sizeof(T) * size, hipMemcpyDefault));
  hipLaunchKernelGGL(shflXor_3<T>, 1, warpSize, 0, 0, d_Input, d_Output);

  HIP_CHECK(hipMemcpy(&Output, d_Output, sizeof(T) * size, hipMemcpyDefault));
  for (int i = 0; i != warpSize; ++i) {
    REQUIRE(compareEqual(Output[i], Expected[i]));
  }

  HIP_CHECK(hipFree(d_Input));
  HIP_CHECK(hipFree(d_Output));
}

template <typename T> __global__ void shflXor_4(T* Input, T* Output) {
  int tid = threadIdx.x;

  Output[tid] = __shfl_xor_sync(AllThreads, Input[tid], 2, 2);
}


template <typename T> static void runTestShflXor_4() {
  const int size = 64;
  T Input[size];
  T Output[size];
  T Expected[size];
  int Values[size] = {0,   -1, 0,   -1, 4,   5,   4,   5,   8,  -9, 8,  -9, 12,  13,  12,  13,
                      16,  17, 16,  17, 20,  -21, 20,  -21, 24, 25, 24, 25, 28,  29,  28,  29,
                      -32, 33, -32, 33, -36, 37,  -36, 37,  40, 41, 40, 41, -44, -45, -44, -45,
                      48,  49, 48,  49, 52,  53,  52,  53,  56, 57, 56, 57, 60,  61,  60,  61};

  initializeInput(Input, size);
  initializeExpected(Expected, Values, size);

  int warpSize = getWarpSize();

  T* d_Input;
  T* d_Output;
  HIP_CHECK(hipMalloc(&d_Input, sizeof(T) * size));
  HIP_CHECK(hipMalloc(&d_Output, sizeof(T) * size));

  HIP_CHECK(hipMemcpy(d_Input, &Input, sizeof(T) * size, hipMemcpyDefault));
  hipLaunchKernelGGL(shflXor_4<T>, 1, warpSize, 0, 0, d_Input, d_Output);

  HIP_CHECK(hipMemcpy(&Output, d_Output, sizeof(T) * size, hipMemcpyDefault));
  for (int i = 0; i != warpSize; ++i) {
    REQUIRE(compareEqual(Output[i], Expected[i]));
  }

  HIP_CHECK(hipFree(d_Input));
  HIP_CHECK(hipFree(d_Output));
}

template <typename T> __global__ void shflXor_5(T* Input, T* Output) {
  int tid = threadIdx.x;
  auto mask = __match_any_sync(AllThreads, (int)(tid < 16));
  int width = (tid < 16) ? 2 : 4;
  Output[tid] = __shfl_xor_sync(mask, Input[tid], width, width);
}


template <typename T> static void runTestShflXor_5() {
  const int size = 64;
  T Input[size];
  T Output[size];
  T Expected[size];
  int Values[size] = {0,   -1, 0,   -1,  4,   5,  4,   5,   8,  -9, 8,   -9,  12, 13, 12,  13,
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
  hipLaunchKernelGGL(shflXor_5<T>, 1, warpSize, 0, 0, d_Input, d_Output);

  HIP_CHECK(hipMemcpy(&Output, d_Output, sizeof(T) * size, hipMemcpyDefault));
  for (int i = 0; i != warpSize; ++i) {
    REQUIRE(compareEqual(Output[i], Expected[i]));
  }

  HIP_CHECK(hipFree(d_Input));
  HIP_CHECK(hipFree(d_Output));
}

__global__ void shflXor_4(int* Input, int* Output) {
  auto tid = threadIdx.x;
  unsigned long long masks[2] = {Every5thBut9th, Every9thBit};

  Output[tid] = -1;
  if (tid % 5 == 0 || tid % 9 == 0)
    Output[tid] = __shfl_xor_sync(masks[tid % 9 == 0], Input[tid], tid);
}

static void runTestShflXor_4() {
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
  hipLaunchKernelGGL(shflXor_4, 1, warpSize, 0, 0, d_Input, d_Output);

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
 * `T  __shfl_xor_sync(unsigned long long mask, T var, unsigned laneMask, int width=warpSize)` -
 * Contains warp __shfl sync functions.
 * @}
 */

/**
 * Test Description
 * ------------------------
 * - Test case to verify __shfl_xor_sync warp functions for different datatypes.

 * Test source
 * ------------------------
 *    - catch/unit/kernel/hipShflSyncXorTests.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */

TEST_CASE("Unit_hipShflSync_Xor") {
  CHECK_WARP_MATCH_FUNCTIONS_SUPPORT

  SECTION("run test for short") {
    runTestShflXor_1<short>();
    runTestShflXor_2<short>();
    runTestShflXor_3<short>();
    runTestShflXor_4<short>();
    runTestShflXor_5<short>();
  }
  SECTION("run test for unsigned short") {
    runTestShflXor_1<unsigned short>();
    runTestShflXor_2<unsigned short>();
    runTestShflXor_3<unsigned short>();
    runTestShflXor_4<unsigned short>();
    runTestShflXor_5<unsigned short>();
  }
  SECTION("run test for int") {
    runTestShflXor_1<int>();
    runTestShflXor_2<int>();
    runTestShflXor_3<int>();
    runTestShflXor_4<int>();
    runTestShflXor_5<int>();
  }
  SECTION("run test for unsigned int") {
    runTestShflXor_1<unsigned int>();
    runTestShflXor_2<unsigned int>();
    runTestShflXor_3<unsigned int>();
    runTestShflXor_4<unsigned int>();
    runTestShflXor_5<unsigned int>();
  }
  SECTION("run test for long") {
    runTestShflXor_1<long>();
    runTestShflXor_2<long>();
    runTestShflXor_3<long>();
    runTestShflXor_4<long>();
    runTestShflXor_5<long>();
  }
  SECTION("run test for unsigned long") {
    runTestShflXor_1<unsigned long>();
    runTestShflXor_2<unsigned long>();
    runTestShflXor_3<unsigned long>();
    runTestShflXor_4<unsigned long>();
    runTestShflXor_5<unsigned long>();
  }
  SECTION("run test for long long") {
    runTestShflXor_1<long long>();
    runTestShflXor_2<long long>();
    runTestShflXor_3<long long>();
    runTestShflXor_4<long long>();
    runTestShflXor_5<long long>();
  }
  SECTION("run test for unsigned long long") {
    runTestShflXor_1<unsigned long long>();
    runTestShflXor_2<unsigned long long>();
    runTestShflXor_3<unsigned long long>();
    runTestShflXor_4<unsigned long long>();
    runTestShflXor_5<unsigned long long>();
  }
  SECTION("run test for float") {
    runTestShflXor_1<float>();
    runTestShflXor_2<float>();
    runTestShflXor_3<float>();
    runTestShflXor_4<float>();
    runTestShflXor_5<float>();
  }
  SECTION("run test for double") {
    runTestShflXor_1<double>();
    runTestShflXor_2<double>();
    runTestShflXor_3<double>();
    runTestShflXor_4<double>();
    runTestShflXor_5<double>();
  }
  SECTION("run test for __half") {
    runTestShflXor_1<__half>();
    runTestShflXor_2<__half>();
    runTestShflXor_3<__half>();
    runTestShflXor_4<__half>();
    runTestShflXor_5<__half>();
  }
  SECTION("run test for __half2") {
    runTestShflXor_1<__half2>();
    runTestShflXor_2<__half2>();
    runTestShflXor_3<__half2>();
    runTestShflXor_4<__half2>();
    runTestShflXor_5<__half2>();
  }
  SECTION("run divergent exec tests") { runTestShflXor_4(); }
}
