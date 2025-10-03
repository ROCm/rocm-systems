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

template <typename T> __global__ void matchAny_1(T* Input, unsigned long long* Output) {
  auto tid = threadIdx.x;
  Output[tid] = __match_any_sync(AllThreads, Input[tid]);
}

template <typename T> static void runTestMatchAny_1() {
  const int size = 64;
  T Input[size] = {
      0, 1, (T)-2, (T)-3, 4, 5, 6,     (T)-7, (T)-3, 4,     5, 6, (T)-7, 0,     1, (T)-2,
      4, 5, 6,     (T)-7, 0, 1, (T)-2, (T)-3, 6,     (T)-7, 0, 1, (T)-2, (T)-3, 4, 5,
      0, 1, (T)-2, (T)-3, 4, 5, 6,     (T)-7, (T)-3, 4,     5, 6, (T)-7, 0,     1, (T)-2,
      4, 5, 6,     (T)-7, 0, 1, (T)-2, (T)-3, 6,     (T)-7, 0, 1, (T)-2, (T)-3, 4, 5};
  unsigned long long Output[size];
  unsigned long long Expected[size] = {
      0x0410200104102001, 0x0820400208204002, 0x1040800410408004, 0x2080010820800108,
      0x4001021040010210, 0x8002042080020420, 0x0104084001040840, 0x0208108002081080,
      0x2080010820800108, 0x4001021040010210, 0x8002042080020420, 0x0104084001040840,
      0x0208108002081080, 0x0410200104102001, 0x0820400208204002, 0x1040800410408004,
      0x4001021040010210, 0x8002042080020420, 0x0104084001040840, 0x0208108002081080,
      0x0410200104102001, 0x0820400208204002, 0x1040800410408004, 0x2080010820800108,
      0x0104084001040840, 0x0208108002081080, 0x0410200104102001, 0x0820400208204002,
      0x1040800410408004, 0x2080010820800108, 0x4001021040010210, 0x8002042080020420,
      0x0410200104102001, 0x0820400208204002, 0x1040800410408004, 0x2080010820800108,
      0x4001021040010210, 0x8002042080020420, 0x0104084001040840, 0x0208108002081080,
      0x2080010820800108, 0x4001021040010210, 0x8002042080020420, 0x0104084001040840,
      0x0208108002081080, 0x0410200104102001, 0x0820400208204002, 0x1040800410408004,
      0x4001021040010210, 0x8002042080020420, 0x0104084001040840, 0x0208108002081080,
      0x0410200104102001, 0x0820400208204002, 0x1040800410408004, 0x2080010820800108,
      0x0104084001040840, 0x0208108002081080, 0x0410200104102001, 0x0820400208204002,
      0x1040800410408004, 0x2080010820800108, 0x4001021040010210, 0x8002042080020420};

  expandPrecision(Input, size);

  int warpSize = getWarpSize();

  T* d_Input;
  unsigned long long* d_Output;
  HIP_CHECK(hipMalloc(&d_Input, sizeof(T) * size));
  HIP_CHECK(hipMalloc(&d_Output, 8 * size));

  HIP_CHECK(hipMemcpy(d_Input, &Input, sizeof(T) * size, hipMemcpyDefault));
  hipLaunchKernelGGL(matchAny_1<T>, 1, warpSize, 0, 0, d_Input, d_Output);

  HIP_CHECK(hipMemcpy(&Output, d_Output, 8 * size, hipMemcpyDefault));
  for (int i = 0; i != warpSize; ++i) {
    REQUIRE(compareMaskEqual(Output, Expected, i, warpSize));
  }

  HIP_CHECK(hipFree(d_Input));
  HIP_CHECK(hipFree(d_Output));
}

template <typename T> __global__ void matchAny_2(T* Input, unsigned long long* Output) {
  auto tid = threadIdx.x;
  // It's okay to use the non-sync__ match, because the purpose of the test is
  // to exercise the mask argument on the sync version.
  auto mask = __match_any_sync(AllThreads, tid / 12);
  Output[tid] = __match_any_sync(mask, Input[tid]);
}

template <typename T> static void runTestMatchAny_2() {
  const int size = 64;
  T Input[size] = {
      0, 1, (T)-2, (T)-3, 4, 5, 6,     (T)-7, (T)-3, 4,     5, 6, (T)-7, 0,     1, (T)-2,
      4, 5, 6,     (T)-7, 0, 1, (T)-2, (T)-3, 6,     (T)-7, 0, 1, (T)-2, (T)-3, 4, 5,
      0, 1, (T)-2, (T)-3, 4, 5, 6,     (T)-7, (T)-3, 4,     5, 6, (T)-7, 0,     1, (T)-2,
      4, 5, 6,     (T)-7, 0, 1, (T)-2, (T)-3, 6,     (T)-7, 0, 1, (T)-2, (T)-3, 4, 5};
  unsigned long long Output[size];
  unsigned long long Expected[size] = {
      0x0000000000000001, 0x0000000000000002, 0x0000000000000004, 0x0000000000000108,
      0x0000000000000210, 0x0000000000000420, 0x0000000000000840, 0x0000000000000080,
      0x0000000000000108, 0x0000000000000210, 0x0000000000000420, 0x0000000000000840,

      0x0000000000081000, 0x0000000000102000, 0x0000000000204000, 0x0000000000408000,
      0x0000000000010000, 0x0000000000020000, 0x0000000000040000, 0x0000000000081000,
      0x0000000000102000, 0x0000000000204000, 0x0000000000408000, 0x0000000000800000,

      0x0000000001000000, 0x0000000002000000, 0x0000000104000000, 0x0000000208000000,
      0x0000000410000000, 0x0000000820000000, 0x0000000040000000, 0x0000000080000000,
      0x0000000104000000, 0x0000000208000000, 0x0000000410000000, 0x0000000820000000,

      0x0000021000000000, 0x0000042000000000, 0x0000084000000000, 0x0000108000000000,
      0x0000010000000000, 0x0000021000000000, 0x0000042000000000, 0x0000084000000000,
      0x0000108000000000, 0x0000200000000000, 0x0000400000000000, 0x0000800000000000,

      0x0001000000000000, 0x0002000000000000, 0x0104000000000000, 0x0208000000000000,
      0x0410000000000000, 0x0820000000000000, 0x0040000000000000, 0x0080000000000000,
      0x0104000000000000, 0x0208000000000000, 0x0410000000000000, 0x0820000000000000,

      0x1000000000000000, 0x2000000000000000, 0x4000000000000000, 0x8000000000000000};

  expandPrecision(Input, size);

  int warpSize = getWarpSize();

  T* d_Input;
  unsigned long long* d_Output;
  HIP_CHECK(hipMalloc(&d_Input, sizeof(T) * size));
  HIP_CHECK(hipMalloc(&d_Output, 8 * size));

  HIP_CHECK(hipMemcpy(d_Input, &Input, sizeof(T) * size, hipMemcpyDefault));
  hipLaunchKernelGGL(matchAny_2<T>, 1, warpSize, 0, 0, d_Input, d_Output);

  HIP_CHECK(hipMemcpy(&Output, d_Output, 8 * size, hipMemcpyDefault));
  for (int i = 0; i != warpSize; ++i) {
    REQUIRE(compareMaskEqual(Output, Expected, i, warpSize));
  }

  HIP_CHECK(hipFree(d_Input));
  HIP_CHECK(hipFree(d_Output));
}

__global__ void matchAny_3(int* Input, int* Output) {
  auto tid = threadIdx.x;
  unsigned long long masks[2] = {Every5thBut9th, Every9thBit};

  Output[tid] = -1;
  if (tid % 5 == 0 || tid % 9 == 0)
    Output[tid] = !!__match_any_sync(masks[tid % 9 == 0], Input[tid]);
}

static void runTestMatchAny_3() {
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
  hipLaunchKernelGGL(matchAny_3, 1, warpSize, 0, 0, d_Input, d_Output);

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
 * `unsigned long long __match_any_sync(unsigned long long mask, T value, int *pred)` -
 * Contains warp __match sync functions.
 * @}
 */

/**
 * Test Description
 * ------------------------
 * - Test case to verify __match_sync warp functions for different datatypes.

 * Test source
 * ------------------------
 *    - catch/unit/kernel/hipMatchSyncAnyTests.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */

TEST_CASE("Unit_hipMatchSync_Any") {
  CHECK_WARP_MATCH_FUNCTIONS_SUPPORT

  SECTION("run test for int") {
    runTestMatchAny_1<int>();
    runTestMatchAny_2<int>();
  }
  SECTION("run test for unsigned int") {
    runTestMatchAny_1<unsigned int>();
    runTestMatchAny_2<unsigned int>();
  }
  SECTION("run test for long") {
    runTestMatchAny_1<long>();
    runTestMatchAny_2<long>();
  }
  SECTION("run test for unsigned long") {
    runTestMatchAny_1<unsigned long>();
    runTestMatchAny_2<unsigned long>();
  }
  SECTION("run test for long long") {
    runTestMatchAny_1<long long>();
    runTestMatchAny_2<long long>();
  }
  SECTION("run test for unsigned long long") {
    runTestMatchAny_1<unsigned long long>();
    runTestMatchAny_2<unsigned long long>();
  }
  SECTION("run test for float") {
    runTestMatchAny_1<float>();
    runTestMatchAny_2<float>();
  }
  SECTION("run test for double") {
    runTestMatchAny_1<double>();
    runTestMatchAny_2<double>();
  }
  SECTION("run divergent execution tests") { runTestMatchAny_3(); }
}
