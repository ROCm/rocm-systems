/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
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

HIP_TEST_CASE(Unit_hipShflSync) {
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

__global__ void warp_sync_32bit_mask_kernel(unsigned long long* Output) {
  int tid = threadIdx.x;
#if defined(__gfx1250__)
  unsigned int mask = 0xffffffffu;
#else
  unsigned long long mask = ~0ull;
#endif

  __syncwarp(mask);

  unsigned long long ballot = __ballot_sync(mask, (tid % 2) == 0);
  int all = __all_sync(mask, true);
  int any = __any_sync(mask, tid == 0);
  unsigned long long match_any = __match_any_sync(mask, tid % 4);
  int pred = 0;
  unsigned long long match_all = __match_all_sync(mask, 7, &pred);
  int shfl = __shfl_sync(mask, tid, 0);
  int up = __shfl_up_sync(mask, tid, 1);
  int down = __shfl_down_sync(mask, tid, 1);
  int xor_val = __shfl_xor_sync(mask, tid, 1);
#if defined(__HIP_PLATFORM_AMD__)
  unsigned int one = 1;
  unsigned int lane = tid;
  unsigned int reduce_add = __reduce_add_sync(mask, one);
  unsigned int reduce_min = __reduce_min_sync(mask, lane);
  unsigned int reduce_max = __reduce_max_sync(mask, lane);
  unsigned int reduce_and = __reduce_and_sync(mask, one);
  unsigned int reduce_or = __reduce_or_sync(mask, one);
  unsigned int reduce_xor = __reduce_xor_sync(mask, one);
#endif

  if (tid == 0) {
    Output[0] = ballot;
    Output[1] = all;
    Output[2] = any;
    Output[3] = match_any;
    Output[4] = match_all;
    Output[5] = pred;
    Output[6] = shfl;
    Output[7] = down;
    Output[9] = xor_val;
#if defined(__HIP_PLATFORM_AMD__)
    Output[10] = reduce_add;
    Output[11] = reduce_min;
    Output[12] = reduce_max;
    Output[13] = reduce_and;
    Output[14] = reduce_or;
    Output[15] = reduce_xor;
#endif
  } else if (tid == 1) {
    Output[8] = up;
  }
}

/**
 * Test Description
 * ------------------------
 * - Verify explicit-mask warp sync builtins accept a 32-bit mask on wave32
 *   architectures.
 * Test source
 * ------------------------
 * - catch/unit/warp/hipShflSyncTests.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 5.6
 */
HIP_TEST_CASE(Unit_hipWarpSync_32BitMask) {
  const int warpSize = getWarpSize();
  constexpr int kNumResults = 16;
  unsigned long long Output[kNumResults];
  unsigned long long expectedEvenMask = 0;
  unsigned long long expectedMatchAnyMask = 0;
  unsigned long long expectedFullMask = warpSize == 64 ? ~0ull : 0xffffffffull;

  for (int i = 0; i < warpSize; ++i) {
    if ((i % 2) == 0) {
      expectedEvenMask |= 1ull << i;
    }
    if ((i % 4) == 0) {
      expectedMatchAnyMask |= 1ull << i;
    }
  }

  unsigned long long* d_Output = nullptr;
  HIP_CHECK(hipMalloc(&d_Output, sizeof(unsigned long long) * kNumResults));

  hipLaunchKernelGGL(warp_sync_32bit_mask_kernel, 1, warpSize, 0, 0, d_Output);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(&Output, d_Output, sizeof(unsigned long long) * kNumResults,
                      hipMemcpyDeviceToHost));

  REQUIRE(Output[0] == expectedEvenMask);
  REQUIRE(Output[1] == 1ull);
  REQUIRE(Output[2] == 1ull);
  REQUIRE(Output[3] == expectedMatchAnyMask);
  REQUIRE(Output[4] == expectedFullMask);
  REQUIRE(Output[5] == 1ull);
  REQUIRE(Output[6] == 0ull);
  REQUIRE(Output[7] == 1ull);
  REQUIRE(Output[8] == 0ull);
  REQUIRE(Output[9] == 1ull);
#if HT_AMD
  REQUIRE(Output[10] == static_cast<unsigned long long>(warpSize));
  REQUIRE(Output[11] == 0ull);
  REQUIRE(Output[12] == static_cast<unsigned long long>(warpSize - 1));
  REQUIRE(Output[13] == 1ull);
  REQUIRE(Output[14] == 1ull);
  REQUIRE(Output[15] == 0ull);
#endif

  HIP_CHECK(hipFree(d_Output));
}

