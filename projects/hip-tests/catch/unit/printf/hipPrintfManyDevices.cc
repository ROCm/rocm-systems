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

__global__ void print_things() {
  uint tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  const char* msg[] = {msg_short_dev, msg_long1_dev, msg_long2_dev};
  printf("%s\n", msg[tid % 3]);
  if (tid % 3 == 0) printf("%s\n", msg_short_dev);
  printf("%s\n", msg[(tid + 1) % 3]);
  printf("%s\n", msg[(tid + 2) % 3]);
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
 * - Test case to verify printf API functionality on many devices
 * Test source
 * ------------------------
 * - catch/unit/printf/hipPrintfManyDevices.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_Printf_ManyDevicesTest") {
  int pcieAtomic = 0;
  HIP_CHECK(hipDeviceGetAttribute(&pcieAtomic, hipDeviceAttributeHostNativeAtomicSupported, 0));
  if (!pcieAtomic) {
    HipTest::HIP_SKIP_TEST("Device doesn't support pcie atomic, Skipped");
    return;
  }
  uint num_blocks = 14;
  uint threads_per_block = 250;
  uint threads_per_device = num_blocks * threads_per_block;
  int num_devices = 0;
  HIP_CHECK(hipGetDeviceCount(&num_devices));
  CaptureStream captured(stdout);
  for (int i = 0; i != num_devices; ++i) {
    HIP_CHECK(hipSetDevice(i));
    hipLaunchKernelGGL(print_things, dim3(num_blocks), dim3(threads_per_block), 0, 0);
    HIP_CHECK(hipDeviceSynchronize());
  }
  auto CapturedData = captured.getCapturedData();
  std::map<std::string, int> linecount;
  for (std::string line; std::getline(CapturedData, line);) {
    linecount[line]++;
  }
  uint num_threads = threads_per_device * num_devices;
  REQUIRE(linecount.size() == 3);
  REQUIRE(linecount[msg_long1] == num_threads);
  REQUIRE(linecount[msg_long2] == num_threads);
  REQUIRE(linecount[msg_short] == num_threads + ((threads_per_device + 2) / 3) * num_devices);
}
/**
 * End doxygen group PrintfTest.
 * @}
 */
