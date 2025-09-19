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

__global__ void any_1(int* Input, int* Output) {
  auto tid = threadIdx.x;
  Output[tid] = __any_sync(AllThreads, Input[tid]);
}

static void runTestAny_1() {
  const int size = 64;
  int Input[size] = {
      0,
  };
  int Output[size];
  int Expected[size] = {
      0,
  };

  int* d_Input;
  int* d_Output;
  HIP_CHECK(hipMalloc(&d_Input, 4 * size));
  HIP_CHECK(hipMalloc(&d_Output, 4 * size));

  int warpSize = getWarpSize();

  HIP_CHECK(hipMemcpy(d_Input, &Input, 4 * size, hipMemcpyDefault));
  hipLaunchKernelGGL(any_1, 1, warpSize, 0, 0, d_Input, d_Output);

  HIP_CHECK(hipMemcpy(&Output, d_Output, 4 * size, hipMemcpyDefault));
  for (int i = 0; i != warpSize; ++i) {
    REQUIRE(Output[i] == Expected[i]);
  }

  HIP_CHECK(hipFree(d_Input));
  HIP_CHECK(hipFree(d_Output));
}

__global__ void any_2(int* Input, int* Output) {
  auto tid = threadIdx.x;
  Output[tid] = __any_sync(AllThreads, Input[tid]);
}

static void runTestAny_2_w64() {
  const int size = 64;
  int Input[size] = {
      0,
  };
  int Output[size];
  int Expected[size] = {
      0,
  };

  Input[60] = 1;

  int warpSize = getWarpSize();
  if (warpSize == 64) std::fill_n(Expected, size, 1);

  int* d_Input;
  int* d_Output;
  HIP_CHECK(hipMalloc(&d_Input, 4 * size));
  HIP_CHECK(hipMalloc(&d_Output, 4 * size));

  HIP_CHECK(hipMemcpy(d_Input, &Input, 4 * size, hipMemcpyDefault));
  hipLaunchKernelGGL(any_2, 1, warpSize, 0, 0, d_Input, d_Output);

  HIP_CHECK(hipMemcpy(&Output, d_Output, 4 * size, hipMemcpyDefault));
  for (int i = 0; i != warpSize; ++i) {
    REQUIRE(Output[i] == Expected[i]);
  }

  HIP_CHECK(hipFree(d_Input));
  HIP_CHECK(hipFree(d_Output));
}

static void runTestAny_2_w32() {
  const int size = 64;
  int Input[size] = {
      0,
  };
  int Output[size];
  int Expected[size] = {
      0,
  };

  Input[30] = 1;
  std::fill_n(Expected, size, 1);

  int* d_Input;
  int* d_Output;
  HIP_CHECK(hipMalloc(&d_Input, 4 * size));
  HIP_CHECK(hipMalloc(&d_Output, 4 * size));

  int warpSize = getWarpSize();

  HIP_CHECK(hipMemcpy(d_Input, &Input, 4 * size, hipMemcpyDefault));
  hipLaunchKernelGGL(any_2, 1, warpSize, 0, 0, d_Input, d_Output);

  HIP_CHECK(hipMemcpy(&Output, d_Output, 4 * size, hipMemcpyDefault));
  for (int i = 0; i != warpSize; ++i) {
    REQUIRE(Output[i] == Expected[i]);
  }

  HIP_CHECK(hipFree(d_Input));
  HIP_CHECK(hipFree(d_Output));
}

__global__ void any_3(int* Input, int* Output) {
  auto tid = threadIdx.x;
  auto mask = __match_any_sync(AllThreads, tid / 12);
  Output[tid] = __any_sync(mask, Input[tid]);
}

static void runTestAny_3() {
  const int size = 64;
  int Input[size] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0};

  int Output[size];
  int Expected[size] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  int* d_Input;
  int* d_Output;
  HIP_CHECK(hipMalloc(&d_Input, 4 * size));
  HIP_CHECK(hipMalloc(&d_Output, 4 * size));

  int warpSize = getWarpSize();

  HIP_CHECK(hipMemcpy(d_Input, &Input, 4 * size, hipMemcpyDefault));
  hipLaunchKernelGGL(any_3, 1, warpSize, 0, 0, d_Input, d_Output);

  HIP_CHECK(hipMemcpy(&Output, d_Output, 4 * size, hipMemcpyDefault));
  for (int i = 0; i != warpSize; ++i) {
    REQUIRE(Output[i] == Expected[i]);
  }

  HIP_CHECK(hipFree(d_Input));
  HIP_CHECK(hipFree(d_Output));
}

__global__ void any_4(int* Input, int* Output) {
  auto tid = threadIdx.x;
  unsigned long long masks[2] = {Every5thBut9th, Every9thBit};

  Output[tid] = -1;
  if (tid % 5 == 0 || tid % 9 == 0) Output[tid] = __any_sync(masks[tid % 9 == 0], Input[tid]);
}

static void runTestAny_4() {
  const int size = 64;
  int Input[size] = {
      0,
  };
  Input[5] = 1;

  int Output[size];
  int Expected[size];

  for (int i = 0; i != size; ++i) {
    if (i % 9 == 0) {
      Expected[i] = 0;
      continue;
    }

    if (i % 5 == 0) {
      Expected[i] = 1;
      continue;
    }

    Expected[i] = -1;
  }

  int* d_Input;
  int* d_Output;
  HIP_CHECK(hipMalloc(&d_Input, 4 * size));
  HIP_CHECK(hipMalloc(&d_Output, 4 * size));

  int warpSize = getWarpSize();

  HIP_CHECK(hipMemcpy(d_Input, &Input, 4 * size, hipMemcpyDefault));
  hipLaunchKernelGGL(any_4, 1, warpSize, 0, 0, d_Input, d_Output);

  HIP_CHECK(hipMemcpy(&Output, d_Output, 4 * size, hipMemcpyDefault));
  for (int i = 0; i != warpSize; ++i) {
    REQUIRE(Output[i] == Expected[i]);
  }

  HIP_CHECK(hipFree(d_Input));
  HIP_CHECK(hipFree(d_Output));
}

__global__ void all_1(int* Input, int* Output) {
  auto tid = threadIdx.x;
  Output[tid] = __all_sync(AllThreads, Input[tid]);
}

static void runTestAll_1_w64() {
  const int size = 64;
  int Input[size] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1};

  int Output[size];
  int Expected[size] = {
      0,
  };

  int warpSize = getWarpSize();
  if (warpSize == 32) std::fill_n(Expected, size, 1);

  int* d_Input;
  int* d_Output;
  HIP_CHECK(hipMalloc(&d_Input, 4 * size));
  HIP_CHECK(hipMalloc(&d_Output, 4 * size));

  HIP_CHECK(hipMemcpy(d_Input, &Input, 4 * size, hipMemcpyDefault));
  hipLaunchKernelGGL(all_1, 1, warpSize, 0, 0, d_Input, d_Output);

  HIP_CHECK(hipMemcpy(&Output, d_Output, 4 * size, hipMemcpyDefault));
  for (int i = 0; i != warpSize; ++i) {
    REQUIRE(Output[i] == Expected[i]);
  }

  HIP_CHECK(hipFree(d_Input));
  HIP_CHECK(hipFree(d_Output));
}

static void runTestAll_1_w32() {
  const int size = 64;
  int Input[size] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  int Output[size];
  int Expected[size] = {
      0,
  };

  int warpSize = getWarpSize();

  int* d_Input;
  int* d_Output;
  HIP_CHECK(hipMalloc(&d_Input, 4 * size));
  HIP_CHECK(hipMalloc(&d_Output, 4 * size));

  HIP_CHECK(hipMemcpy(d_Input, &Input, 4 * size, hipMemcpyDefault));
  hipLaunchKernelGGL(all_1, 1, warpSize, 0, 0, d_Input, d_Output);

  HIP_CHECK(hipMemcpy(&Output, d_Output, 4 * size, hipMemcpyDefault));
  for (int i = 0; i != warpSize; ++i) {
    REQUIRE(Output[i] == Expected[i]);
  }

  HIP_CHECK(hipFree(d_Input));
  HIP_CHECK(hipFree(d_Output));
}

__global__ void all_2(int* Input, int* Output) {
  auto tid = threadIdx.x;
  Output[tid] = __all_sync(AllThreads, Input[tid]);
}

static void runTestAll_2() {
  const int size = 64;
  int Input[size] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  int Output[size];
  int Expected[size] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  int* d_Input;
  int* d_Output;
  HIP_CHECK(hipMalloc(&d_Input, 4 * size));
  HIP_CHECK(hipMalloc(&d_Output, 4 * size));

  int warpSize = getWarpSize();

  HIP_CHECK(hipMemcpy(d_Input, &Input, 4 * size, hipMemcpyDefault));
  hipLaunchKernelGGL(all_2, 1, warpSize, 0, 0, d_Input, d_Output);

  HIP_CHECK(hipMemcpy(&Output, d_Output, 4 * size, hipMemcpyDefault));
  for (int i = 0; i != warpSize; ++i) {
    REQUIRE(Output[i] == Expected[i]);
  }

  HIP_CHECK(hipFree(d_Input));
  HIP_CHECK(hipFree(d_Output));
}

__global__ void all_3(int* Input, int* Output) {
  auto tid = threadIdx.x;
  auto mask = __match_any_sync(AllThreads, tid / 12);
  Output[tid] = __all_sync(mask, Input[tid]);
}

static void runTestAll_3() {
  const int size = 64;
  int Input[size] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1};

  int Output[size];
  int Expected[size] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  int* d_Input;
  int* d_Output;
  HIP_CHECK(hipMalloc(&d_Input, 4 * size));
  HIP_CHECK(hipMalloc(&d_Output, 4 * size));

  int warpSize = getWarpSize();

  HIP_CHECK(hipMemcpy(d_Input, &Input, 4 * size, hipMemcpyDefault));
  hipLaunchKernelGGL(all_3, 1, warpSize, 0, 0, d_Input, d_Output);

  HIP_CHECK(hipMemcpy(&Output, d_Output, 4 * size, hipMemcpyDefault));
  for (int i = 0; i != warpSize; ++i) {
    REQUIRE(Output[i] == Expected[i]);
  }

  HIP_CHECK(hipFree(d_Input));
  HIP_CHECK(hipFree(d_Output));
}

__global__ void all_4(int* Input, int* Output) {
  auto tid = threadIdx.x;
  unsigned long long masks[2] = {Every5thBut9th, Every9thBit};

  Output[tid] = -1;
  if (tid % 5 == 0 || tid % 9 == 0) Output[tid] = __all_sync(masks[tid % 9 == 0], Input[tid]);
}

static void runTestAll_4() {
  const int size = 64;
  int Input[size];
  std::fill_n(Input, size, 1);
  Input[5] = 0;

  int Output[size];
  int Expected[size];

  for (int i = 0; i != size; ++i) {
    if (i % 9 == 0) {
      Expected[i] = 1;
      continue;
    }

    if (i % 5 == 0) {
      Expected[i] = 0;
      continue;
    }

    Expected[i] = -1;
  }

  int* d_Input;
  int* d_Output;
  HIP_CHECK(hipMalloc(&d_Input, 4 * size));
  HIP_CHECK(hipMalloc(&d_Output, 4 * size));

  int warpSize = getWarpSize();

  HIP_CHECK(hipMemcpy(d_Input, &Input, 4 * size, hipMemcpyDefault));
  hipLaunchKernelGGL(all_4, 1, warpSize, 0, 0, d_Input, d_Output);

  HIP_CHECK(hipMemcpy(&Output, d_Output, 4 * size, hipMemcpyDefault));
  for (int i = 0; i != warpSize; ++i) {
    REQUIRE(Output[i] == Expected[i]);
  }

  HIP_CHECK(hipFree(d_Input));
  HIP_CHECK(hipFree(d_Output));
}

__global__ void ballot_1(int* Input, unsigned long long* Output) {
  auto tid = threadIdx.x;
  Output[tid] = __ballot_sync(AllThreads, Input[tid]);
}

static void runTestBallot_1() {
  const int size = 64;
  int Input[size] = {0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1,
                     0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1,
                     0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1};
  unsigned long long Output[size];
  unsigned long long Expected[size] = {
      0xc9274e72c9274e72, 0xc9274e72c9274e72, 0xc9274e72c9274e72, 0xc9274e72c9274e72,
      0xc9274e72c9274e72, 0xc9274e72c9274e72, 0xc9274e72c9274e72, 0xc9274e72c9274e72,
      0xc9274e72c9274e72, 0xc9274e72c9274e72, 0xc9274e72c9274e72, 0xc9274e72c9274e72,
      0xc9274e72c9274e72, 0xc9274e72c9274e72, 0xc9274e72c9274e72, 0xc9274e72c9274e72,
      0xc9274e72c9274e72, 0xc9274e72c9274e72, 0xc9274e72c9274e72, 0xc9274e72c9274e72,
      0xc9274e72c9274e72, 0xc9274e72c9274e72, 0xc9274e72c9274e72, 0xc9274e72c9274e72,
      0xc9274e72c9274e72, 0xc9274e72c9274e72, 0xc9274e72c9274e72, 0xc9274e72c9274e72,
      0xc9274e72c9274e72, 0xc9274e72c9274e72, 0xc9274e72c9274e72, 0xc9274e72c9274e72,
      0xc9274e72c9274e72, 0xc9274e72c9274e72, 0xc9274e72c9274e72, 0xc9274e72c9274e72,
      0xc9274e72c9274e72, 0xc9274e72c9274e72, 0xc9274e72c9274e72, 0xc9274e72c9274e72,
      0xc9274e72c9274e72, 0xc9274e72c9274e72, 0xc9274e72c9274e72, 0xc9274e72c9274e72,
      0xc9274e72c9274e72, 0xc9274e72c9274e72, 0xc9274e72c9274e72, 0xc9274e72c9274e72,
      0xc9274e72c9274e72, 0xc9274e72c9274e72, 0xc9274e72c9274e72, 0xc9274e72c9274e72,
      0xc9274e72c9274e72, 0xc9274e72c9274e72, 0xc9274e72c9274e72, 0xc9274e72c9274e72,
      0xc9274e72c9274e72, 0xc9274e72c9274e72, 0xc9274e72c9274e72, 0xc9274e72c9274e72,
      0xc9274e72c9274e72, 0xc9274e72c9274e72, 0xc9274e72c9274e72, 0xc9274e72c9274e72};

  int* d_Input;
  unsigned long long* d_Output;
  HIP_CHECK(hipMalloc(&d_Input, 4 * size));
  HIP_CHECK(hipMalloc(&d_Output, 8 * size));

  int warpSize = getWarpSize();

  HIP_CHECK(hipMemcpy(d_Input, &Input, 4 * size, hipMemcpyDefault));
  hipLaunchKernelGGL(ballot_1, 1, warpSize, 0, 0, d_Input, d_Output);

  HIP_CHECK(hipMemcpy(&Output, d_Output, 8 * size, hipMemcpyDefault));
  for (int i = 0; i != warpSize; ++i) {
    REQUIRE(compareMaskEqual(Output, Expected, i, warpSize));
  }

  HIP_CHECK(hipFree(d_Input));
  HIP_CHECK(hipFree(d_Output));
}

__global__ void ballot_2(int* Input, unsigned long long* Output) {
  auto tid = threadIdx.x;
  auto mask = __match_any_sync(AllThreads, tid / 12);
  Output[tid] = __ballot_sync(mask, Input[tid]);
}

static void runTestBallot_2() {
  const int size = 64;
  int Input[size] = {0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1,
                     0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1,
                     0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1};
  unsigned long long Output[size];
  unsigned long long Expected[size] = {
      0x0000000000000e72, 0x0000000000000e72, 0x0000000000000e72, 0x0000000000000e72,
      0x0000000000000e72, 0x0000000000000e72, 0x0000000000000e72, 0x0000000000000e72,
      0x0000000000000e72, 0x0000000000000e72, 0x0000000000000e72, 0x0000000000000e72,
      0x0000000000274000, 0x0000000000274000, 0x0000000000274000, 0x0000000000274000,
      0x0000000000274000, 0x0000000000274000, 0x0000000000274000, 0x0000000000274000,
      0x0000000000274000, 0x0000000000274000, 0x0000000000274000, 0x0000000000274000,
      0x00000002c9000000, 0x00000002c9000000, 0x00000002c9000000, 0x00000002c9000000,
      0x00000002c9000000, 0x00000002c9000000, 0x00000002c9000000, 0x00000002c9000000,
      0x00000002c9000000, 0x00000002c9000000, 0x00000002c9000000, 0x00000002c9000000,
      0x00004e7000000000, 0x00004e7000000000, 0x00004e7000000000, 0x00004e7000000000,
      0x00004e7000000000, 0x00004e7000000000, 0x00004e7000000000, 0x00004e7000000000,
      0x00004e7000000000, 0x00004e7000000000, 0x00004e7000000000, 0x00004e7000000000,
      0x0927000000000000, 0x0927000000000000, 0x0927000000000000, 0x0927000000000000,
      0x0927000000000000, 0x0927000000000000, 0x0927000000000000, 0x0927000000000000,
      0x0927000000000000, 0x0927000000000000, 0x0927000000000000, 0x0927000000000000,
      0xc000000000000000, 0xc000000000000000, 0xc000000000000000, 0xc000000000000000};

  int* d_Input;
  unsigned long long* d_Output;
  HIP_CHECK(hipMalloc(&d_Input, 4 * size));
  HIP_CHECK(hipMalloc(&d_Output, 8 * size));

  int warpSize = getWarpSize();

  HIP_CHECK(hipMemcpy(d_Input, &Input, 4 * size, hipMemcpyDefault));
  hipLaunchKernelGGL(ballot_2, 1, warpSize, 0, 0, d_Input, d_Output);

  HIP_CHECK(hipMemcpy(&Output, d_Output, 8 * size, hipMemcpyDefault));
  for (int i = 0; i != warpSize; ++i) {
    REQUIRE(compareMaskEqual(Output, Expected, i, warpSize));
  }

  HIP_CHECK(hipFree(d_Input));
  HIP_CHECK(hipFree(d_Output));
}

__global__ void ballot_3(int* Input, unsigned long long* Output) {
  auto tid = threadIdx.x;
  unsigned long long masks[2] = {Every5thBut9th, Every9thBit};

  Output[tid] = -1;
  if (tid % 5 == 0 || tid % 9 == 0) Output[tid] = __ballot_sync(masks[tid % 9 == 0], Input[tid]);
}

static void runTestBallot_3() {
  const int size = 64;
  int Input[size];
  std::fill_n(Input, size, 1);

  unsigned long long Output[size];
  unsigned long long Expected[size];

  for (int i = 0; i != size; ++i) {
    if (i % 9 == 0) {
      Expected[i] = Every9thBit;
      continue;
    }

    if (i % 5 == 0) {
      Expected[i] = Every5thBut9th;
      continue;
    }

    Expected[i] = -1;
  }

  int* d_Input;
  unsigned long long* d_Output;
  HIP_CHECK(hipMalloc(&d_Input, 4 * size));
  HIP_CHECK(hipMalloc(&d_Output, 8 * size));

  int warpSize = getWarpSize();

  HIP_CHECK(hipMemcpy(d_Input, &Input, 4 * size, hipMemcpyDefault));
  hipLaunchKernelGGL(ballot_3, 1, warpSize, 0, 0, d_Input, d_Output);

  HIP_CHECK(hipMemcpy(&Output, d_Output, 8 * size, hipMemcpyDefault));
  for (int i = 0; i != warpSize; ++i) {
    REQUIRE(compareMaskEqual(Output, Expected, i, warpSize));
  }

  HIP_CHECK(hipFree(d_Input));
  HIP_CHECK(hipFree(d_Output));
}

/**
 * @addtogroup __vote_sync
 * @{
 * @ingroup VoteSyncTest
 *
 *   - `unsigned long long __any_sync(unsigned long long mask, int predicate)`
 *   - `unsigned long long __all_sync(unsigned long long mask, int predicate)`
 *   - `unsigned long long __ballot_sync(unsigned long long mask, int predicate)`
 *
 * Contains warp vote sync functions.
 * @}
 */

/**
 * Test Description
 * ------------------------
 * - Test cases to verify warp vote functions.

 * Test source
 * ------------------------
 *    - catch/unit/kernel/hipVoteSyncTests.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */

TEST_CASE("Unit_hipVoteSync_Any") {
  CHECK_WARP_MATCH_FUNCTIONS_SUPPORT

  runTestAny_1();
  runTestAny_2_w64();
  runTestAny_2_w32();
  runTestAny_3();
  runTestAny_4();
}

TEST_CASE("Unit_hipVoteSync_All") {
  CHECK_WARP_MATCH_FUNCTIONS_SUPPORT

  runTestAll_1_w64();
  runTestAll_1_w32();
  runTestAll_2();
  runTestAll_3();
  runTestAll_4();
}

TEST_CASE("Unit_hipVoteSync_Ballot") {
  CHECK_WARP_MATCH_FUNCTIONS_SUPPORT

  runTestBallot_1();
  runTestBallot_2();
  runTestBallot_3();
}
