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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#include <vector>
#include <map>
#include "printf_common.h"  // NOLINT

__global__ void kernel_mixed0(int* retval) {
  uint tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  // Three strings passed as divergent values to the same hostcall.
  const char* msg;
  switch (tid % 3) {
    case 0:
      msg = msg_short_dev;
      break;
    case 1:
      msg = msg_long1_dev;
      break;
    case 2:
      msg = msg_long2_dev;
      break;
  }
  retval[tid] = printf("%s\n", msg);
}

static void test_mixed0(int* retval, uint num_blocks, uint threads_per_block) {
  CaptureStream captured(stdout);
  uint num_threads = num_blocks * threads_per_block;
  for (uint i = 0; i != num_threads; ++i) {
    retval[i] = 0x23232323;
  }
  hipLaunchKernelGGL(kernel_mixed0, dim3(num_blocks), dim3(threads_per_block), 0, 0, retval);
  HIP_CHECK(hipStreamSynchronize(0));
  auto CapturedData = captured.getCapturedData();
  for (uint ii = 0; ii != num_threads; ++ii) {
    switch (ii % 3) {
      case 0:
        REQUIRE(retval[ii] == strlen(msg_short) + 1);
        break;
      case 1:
        REQUIRE(retval[ii] == strlen(msg_long1) + 1);
        break;
      case 2:
        REQUIRE(retval[ii] == strlen(msg_long2) + 1);
        break;
    }
  }
  std::map<std::string, int> linecount;
  for (std::string line; std::getline(CapturedData, line);) {
    linecount[line]++;
  }
  REQUIRE(linecount.size() == 3);
  REQUIRE(linecount[msg_short] == (num_threads + 2) / 3);
  REQUIRE(linecount[msg_long1] == (num_threads + 1) / 3);
  REQUIRE(linecount[msg_long2] == (num_threads + 0) / 3);
}

__global__ void kernel_mixed1(int* retval) {
  const uint tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  // Three strings passed to divergent hostcalls.
  switch (tid % 3) {
    case 0:
      retval[tid] = printf("%s\n", msg_short_dev);
      break;
    case 1:
      retval[tid] = printf("%s\n", msg_long1_dev);
      break;
    case 2:
      retval[tid] = printf("%s\n", msg_long2_dev);
      break;
  }
}

static void test_mixed1(int* retval, uint num_blocks, uint threads_per_block) {
  CaptureStream captured(stdout);
  uint num_threads = num_blocks * threads_per_block;
  for (uint i = 0; i != num_threads; ++i) {
    retval[i] = 0x23232323;
  }
  hipLaunchKernelGGL(kernel_mixed1, dim3(num_blocks), dim3(threads_per_block), 0, 0, retval);
  HIP_CHECK(hipStreamSynchronize(0));
  auto CapturedData = captured.getCapturedData();
  for (uint ii = 0; ii != num_threads; ++ii) {
    switch (ii % 3) {
      case 0:
        REQUIRE(retval[ii] == strlen(msg_short) + 1);
        break;
      case 1:
        REQUIRE(retval[ii] == strlen(msg_long1) + 1);
        break;
      case 2:
        REQUIRE(retval[ii] == strlen(msg_long2) + 1);
        break;
    }
  }
  std::map<std::string, int> linecount;
  for (std::string line; std::getline(CapturedData, line);) {
    linecount[line]++;
  }
  REQUIRE(linecount.size() == 3);
  REQUIRE(linecount[msg_short] == (num_threads + 2) / 3);
  REQUIRE(linecount[msg_long1] == (num_threads + 1) / 3);
  REQUIRE(linecount[msg_long2] == (num_threads + 0) / 3);
}

__global__ void kernel_mixed2(int* retval) {
  const uint tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  // Three different strings. All workitems print all three, but
  // in different orders.
  const char* msg[] = {msg_short_dev, msg_long1_dev, msg_long2_dev};
  retval[tid] = printf("%s%s%s\n", msg[tid % 3], msg[(tid + 1) % 3], msg[(tid + 2) % 3]);
}

static void test_mixed2(int* retval, uint num_blocks, uint threads_per_block) {
  CaptureStream captured(stdout);
  uint num_threads = num_blocks * threads_per_block;
  for (uint i = 0; i != num_threads; ++i) {
    retval[i] = 0x23232323;
  }
  hipLaunchKernelGGL(kernel_mixed2, dim3(num_blocks), dim3(threads_per_block), 0, 0, retval);
  HIP_CHECK(hipStreamSynchronize(0));
  auto CapturedData = captured.getCapturedData();
  for (uint ii = 0; ii != num_threads; ++ii) {
    REQUIRE(retval[ii] == strlen(msg_short) + strlen(msg_long1) + strlen(msg_long2) + 1);
  }
  std::map<std::string, int> linecount;
  for (std::string line; std::getline(CapturedData, line);) {
    linecount[line]++;
  }
  std::string str1 = std::string(msg_short) + std::string(msg_long1) + std::string(msg_long2);
  std::string str2 = std::string(msg_long1) + std::string(msg_long2) + std::string(msg_short);
  std::string str3 = std::string(msg_long2) + std::string(msg_short) + std::string(msg_long1);
  REQUIRE(linecount.size() == 3);
  REQUIRE(linecount[str1] == (num_threads + 2) / 3);
  REQUIRE(linecount[str2] == (num_threads + 1) / 3);
  REQUIRE(linecount[str3] == (num_threads + 0) / 3);
}

__global__ void kernel_mixed3(int* retval) {
  const uint tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  int result = 0;
  result += printf("%s\n", msg_long1_dev);
  if (tid % 3 == 0) {
    result += printf("%s\n", msg_short_dev);
  }
  result += printf("%s\n", msg_long2_dev);
  retval[tid] = result;
}

static void test_mixed3(int* retval, uint num_blocks, uint threads_per_block) {
  CaptureStream captured(stdout);
  uint num_threads = num_blocks * threads_per_block;
  for (uint i = 0; i != num_threads; ++i) {
    retval[i] = 0x23232323;
  }
  hipLaunchKernelGGL(kernel_mixed3, dim3(num_blocks), dim3(threads_per_block), 0, 0, retval);
  HIP_CHECK(hipStreamSynchronize(0));
  auto CapturedData = captured.getCapturedData();
  for (uint ii = 0; ii != num_threads; ++ii) {
    if (ii % 3 == 0) {
      REQUIRE(retval[ii] == strlen(msg_long1) + strlen(msg_short) + strlen(msg_long2) + 3);
    } else {
      REQUIRE(retval[ii] == strlen(msg_long1) + strlen(msg_long2) + 2);
    }
  }
  std::map<std::string, int> linecount;
  for (std::string line; std::getline(CapturedData, line);) {
    linecount[line]++;
  }
  REQUIRE(linecount.size() == 3);
  REQUIRE(linecount[msg_long1] == num_threads);
  REQUIRE(linecount[msg_long2] == num_threads);
  REQUIRE(linecount[msg_short] == (num_threads + 2) / 3);
}

__global__ void kernel_numbers() {
  uint tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  for (uint i = 0; i != 7; ++i) {
    uint base = tid * 21 + i * 3;
    printf("%d %d %d\n", base, base + 1, base + 2);
  }
}

static void test_numbers(uint num_blocks, uint threads_per_block) {
  CaptureStream captured(stdout);
  uint num_threads = num_blocks * threads_per_block;
  hipLaunchKernelGGL(kernel_numbers, dim3(num_blocks), dim3(threads_per_block), 0, 0);
  HIP_CHECK(hipStreamSynchronize(0));
  auto CapturedData = captured.getCapturedData();
  std::vector<uint> points;
  while (true) {
    uint i;
    CapturedData >> i;
    if (CapturedData.fail()) break;
    points.push_back(i);
  }
  std::sort(points.begin(), points.end());
  points.erase(std::unique(points.begin(), points.end()), points.end());
  REQUIRE(points.size() == 21 * num_threads);
  REQUIRE(points.back() == 21 * num_threads - 1);
}
/**
 * @addtogroup printf printf
 * @{
 * @ingroup PrintfTest
 * `int printf()` -
 * Method to print the content on output device.
 */
/**
 * Test Description
 * ------------------------
 * - Test case to verify printf API functionality with different strings
 * Test source
 * ------------------------
 * - catch/unit/printf/hipPrintfManyWaves.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_Printf_PrintfManyWaves") {
  int pcieAtomic = 0;
  HIP_CHECK(hipDeviceGetAttribute(&pcieAtomic, hipDeviceAttributeHostNativeAtomicSupported, 0));
  if (!pcieAtomic) {
    HipTest::HIP_SKIP_TEST("Device doesn't support pcie atomic, Skipped");
    return;
  }
  uint num_blocks = 150;
  uint threads_per_block = 250;
  uint num_threads = num_blocks * threads_per_block;
  void* retval_void;
  HIP_CHECK(hipHostMalloc(&retval_void, 4 * num_threads));
  auto retval = reinterpret_cast<int*>(retval_void);
  test_mixed0(retval, num_blocks, threads_per_block);
  test_mixed1(retval, num_blocks, threads_per_block);
  test_mixed2(retval, num_blocks, threads_per_block);
  test_mixed3(retval, num_blocks, threads_per_block);
  test_numbers(num_blocks, threads_per_block);
  HIP_CHECK(hipFree(retval_void));
}
/**
 * End doxygen group PrintfTest.
 * @}
 */
