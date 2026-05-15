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
#include <map>
#include "printf_common.h"  // NOLINT

const char* msg_short = "Carpe diem.";
const char* msg_long1 =
    "Lorem ipsum dolor sit amet, consectetur nullam. In mollis imperdiet nibh nec ullamcorper.";  // NOLINT
const char* msg_long2 =
    "Curabitur nec metus sit amet augue vehicula ultrices ut id leo. Lorem ipsum dolor sit amet, "
    "consectetur adipiscing elit amet.";  // NOLINT

__global__ void kernel_uniform0(int* retval) {
  uint tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  retval[tid] = printf("Hello World\n");
}

static void test_uniform0(int* retval, uint num_blocks, uint threads_per_block) {
  CaptureStream captured(stdout);
  uint num_threads = num_blocks * threads_per_block;
  for (uint i = 0; i != num_threads; ++i) {
    retval[i] = 0x23232323;
  }
  hipLaunchKernelGGL(kernel_uniform0, dim3(num_blocks), dim3(threads_per_block), 0, 0, retval);
  HIP_CHECK(hipStreamSynchronize(0));
  auto CapturedData = captured.getCapturedData();
  for (uint ii = 0; ii != num_threads; ++ii) {
    REQUIRE(retval[ii] == strlen("Hello World\n"));
  }
  std::map<std::string, int> linecount;
  for (std::string line; std::getline(CapturedData, line);) {
    linecount[line]++;
  }
  REQUIRE(linecount.size() == 1);
  REQUIRE(linecount["Hello World"] == num_threads);
}

__global__ void kernel_uniform1(int* retval) {
  uint tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  retval[tid] = printf("Six times Eight is %d\n", 42);
}

static void test_uniform1(int* retval, uint num_blocks, uint threads_per_block) {
  CaptureStream captured(stdout);
  uint num_threads = num_blocks * threads_per_block;
  for (uint i = 0; i != num_threads; ++i) {
    retval[i] = 0x23232323;
  }
  hipLaunchKernelGGL(kernel_uniform1, dim3(num_blocks), dim3(threads_per_block), 0, 0, retval);
  HIP_CHECK(hipStreamSynchronize(0));
  auto CapturedData = captured.getCapturedData();
  for (uint ii = 0; ii != num_threads; ++ii) {
    REQUIRE(retval[ii] == strlen("Six times Eight is 42") + 1);
  }
  std::map<std::string, int> linecount;
  for (std::string line; std::getline(CapturedData, line);) {
    linecount[line]++;
  }
  REQUIRE(linecount.size() == 1);
  REQUIRE(linecount["Six times Eight is 42"] == num_threads);
}

__global__ void kernel_divergent0(int* retval) {
  uint tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  retval[tid] = printf("Thread ID: %d\n", tid);
}

static void test_divergent0(int* retval, uint num_blocks, uint threads_per_block) {
  CaptureStream captured(stdout);
  uint num_threads = num_blocks * threads_per_block;
  for (uint i = 0; i != num_threads; ++i) {
    retval[i] = 0x23232323;
  }
  hipLaunchKernelGGL(kernel_divergent0, dim3(num_blocks), dim3(threads_per_block), 0, 0, retval);
  HIP_CHECK(hipStreamSynchronize(0));
  auto CapturedData = captured.getCapturedData();
  for (uint ii = 0; ii != 10; ++ii) {
    REQUIRE(retval[ii] == 13);
  }
  for (uint ii = 10; ii != num_threads; ++ii) {
    REQUIRE(retval[ii] == 14);
  }
  std::vector<uint> threadIds;
  for (std::string line; std::getline(CapturedData, line);) {
    auto pos = line.find(':');
    REQUIRE(line.substr(0, pos) == "Thread ID");
    threadIds.push_back(std::stoul(line.substr(pos + 2)));
  }
  std::sort(threadIds.begin(), threadIds.end());
  REQUIRE(threadIds.size() == num_threads);
  REQUIRE(threadIds.back() == num_threads - 1);
}

__global__ void kernel_divergent1(int* retval) {
  uint tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  if (tid % 2) {
    retval[tid] = printf("Hello World\n");
  } else {
    retval[tid] = -1;
  }
}

static void test_divergent1(int* retval, uint num_blocks, uint threads_per_block) {
  CaptureStream captured(stdout);
  uint num_threads = num_blocks * threads_per_block;
  for (uint i = 0; i != num_threads; ++i) {
    retval[i] = 0x23232323;
  }
  hipLaunchKernelGGL(kernel_divergent1, dim3(num_blocks), dim3(threads_per_block), 0, 0, retval);
  HIP_CHECK(hipStreamSynchronize(0));
  auto CapturedData = captured.getCapturedData();
  for (uint ii = 0; ii != num_threads; ++ii) {
    if (ii % 2) {
      REQUIRE(retval[ii] == strlen("Hello World\n"));
    } else {
      REQUIRE(retval[ii] == -1);
    }
  }
  std::map<std::string, int> linecount;
  for (std::string line; std::getline(CapturedData, line);) {
    linecount[line]++;
  }
  REQUIRE(linecount.size() == 1);
  REQUIRE(linecount["Hello World"] == num_threads / 2);
}

__global__ void kernel_series(int* retval) {
  const uint tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  int result = 0;
  result += printf("%s\n", msg_long1_dev);
  result += printf("%s\n", msg_short_dev);
  result += printf("%s\n", msg_long2_dev);
  retval[tid] = result;
}

static void test_series(int* retval, uint num_blocks, uint threads_per_block) {
  CaptureStream captured(stdout);
  uint num_threads = num_blocks * threads_per_block;
  for (uint i = 0; i != num_threads; ++i) {
    retval[i] = 0x23232323;
  }
  hipLaunchKernelGGL(kernel_series, dim3(num_blocks), dim3(threads_per_block), 0, 0, retval);
  HIP_CHECK(hipStreamSynchronize(0));
  auto CapturedData = captured.getCapturedData();
  for (uint ii = 0; ii != num_threads; ++ii) {
    REQUIRE(retval[ii] == strlen(msg_long1) + strlen(msg_short) + strlen(msg_long2) + 3);
  }
  std::map<std::string, int> linecount;
  for (std::string line; std::getline(CapturedData, line);) {
    linecount[line]++;
  }
  REQUIRE(linecount.size() == 3);
  REQUIRE(linecount[msg_long1] == num_threads);
  REQUIRE(linecount[msg_long2] == num_threads);
  REQUIRE(linecount[msg_short] == num_threads);
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
 * - Test case to verify basic functionality of printf API.
 * Test source
 * ------------------------
 * - catch/unit/printf/hipPrintfBasic.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_Printf_PrintfBasicTsts") {
  int pcieAtomic = 0;
  HIP_CHECK(hipDeviceGetAttribute(&pcieAtomic, hipDeviceAttributeHostNativeAtomicSupported, 0));
  if (!pcieAtomic) {
    HipTest::HIP_SKIP_TEST("Device doesn't support pcie atomic, Skipped");
    return;
  }
  uint num_blocks = 1;
  uint threads_per_block = 64;
  uint num_threads = num_blocks * threads_per_block;
  void* retval_void;
  HIP_CHECK(hipHostMalloc(&retval_void, 4 * num_threads));
  auto retval = reinterpret_cast<int*>(retval_void);
  test_uniform0(retval, num_blocks, threads_per_block);
  test_uniform1(retval, num_blocks, threads_per_block);
  test_divergent0(retval, num_blocks, threads_per_block);
  test_divergent1(retval, num_blocks, threads_per_block);
  test_series(retval, num_blocks, threads_per_block);
  HIP_CHECK(hipFree(retval_void));
}
/**
 * End doxygen group PrintfTest.
 * @}
 */
