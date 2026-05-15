/*
 Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

/**
 * @addtogroup hipMemcpy hipMemcpy
 * @{
 * @ingroup perfMemoryTest
 * `hipMemcpy(void* dst, const void* src, size_t count, hipMemcpyKind kind)` -
 * Copies data between host and device.
 */

#include <hip_test_common.hh>
// #define ENABLE_DEBUG 1
#define NUM_SIZE 14
#define NUM_ITER 1000
// max BW number for DevicetoDeviceNoCU
#define NOCU_MAX_BW 128

class hipPerfMemcpy {
 private:
  size_t totalSizes_[NUM_SIZE];
  void setHostBuffer(int* A, int val, size_t size);

 public:
  hipPerfMemcpy();
  ~hipPerfMemcpy() {}
  void TestResult(unsigned int numTests, std::chrono::duration<double, std::micro> diff,
                  hipMemcpyKind type);
  bool run_h2d(unsigned int numTests);
  bool run_d2h(unsigned int numTests);
  bool run_d2d(unsigned int numTests);
  bool run_d2d_nocu(unsigned int numTests);
};

hipPerfMemcpy::hipPerfMemcpy() {
  for (int i = 0; i < NUM_SIZE; i++) {
    totalSizes_[i] = 1 << (i + 9);
  }
}

void hipPerfMemcpy::setHostBuffer(int* A, int val, size_t size) {
  size_t len = size / sizeof(int);
  for (int i = 0; i < len; i++) {
    A[i] = val;
  }
}

void hipPerfMemcpy::TestResult(unsigned int numTests,
                               std::chrono::duration<double, std::micro> diff, hipMemcpyKind type) {
  // BW in GB/s
  double perf =
      (static_cast<double>(totalSizes_[numTests] * NUM_ITER) * static_cast<double>(1e-03)) /
      diff.count();

  const char* typestr = NULL;

  if (type == hipMemcpyHostToDevice) {
    typestr = "Host to Device";
  } else if (type == hipMemcpyDeviceToHost) {
    typestr = "Device to Host";
  } else if (type == hipMemcpyDeviceToDevice) {
    typestr = "Device to Device";
    perf *= 2.0;
  } else if (type == hipMemcpyDeviceToDeviceNoCU) {
    typestr = "Device to Device No CU";
    perf *= 2.0;
  }

  CONSOLE_PRINT("hipPerfMemcpy[%d] %s copy BW %.2f GB/s for memory size of %zu Bytes.\n", numTests,
                typestr, perf, totalSizes_[numTests]);
}

bool hipPerfMemcpy::run_h2d(unsigned int numTests) {
  int *A, *Ad;
  A = new int[totalSizes_[numTests]];
  HIP_CHECK(hipHostRegister(A, totalSizes_[numTests], hipHostRegisterDefault));
  setHostBuffer(A, 1, totalSizes_[numTests]);
  HIP_CHECK(hipMalloc(&Ad, totalSizes_[numTests]));

  // measure performance based on host time
  auto all_start = std::chrono::steady_clock::now();

  for (int j = 0; j < NUM_ITER; j++) {
    HIP_CHECK(hipMemcpyAsync(Ad, A, totalSizes_[numTests], hipMemcpyHostToDevice, nullptr));
  }

  HIP_CHECK(hipDeviceSynchronize());

  auto all_end = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::micro> diff = all_end - all_start;

  TestResult(numTests, diff, hipMemcpyHostToDevice);

  HIP_CHECK(hipHostUnregister(A));
  delete[] A;
  HIP_CHECK(hipFree(Ad));

  return true;
}

bool hipPerfMemcpy::run_d2h(unsigned int numTests) {
  int *A, *Ad;
  A = new int[totalSizes_[numTests]];
  HIP_CHECK(hipHostRegister(A, totalSizes_[numTests], hipHostRegisterDefault));
  HIP_CHECK(hipMalloc(&Ad, totalSizes_[numTests]));
  HIP_CHECK(hipMemset(Ad, 0x1, totalSizes_[numTests]));

  // measure performance based on host time
  auto all_start = std::chrono::steady_clock::now();

  for (int j = 0; j < NUM_ITER; j++) {
    HIP_CHECK(hipMemcpyAsync(A, Ad, totalSizes_[numTests], hipMemcpyDeviceToHost, nullptr));
  }

  HIP_CHECK(hipDeviceSynchronize());

  auto all_end = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::micro> diff = all_end - all_start;

  TestResult(numTests, diff, hipMemcpyDeviceToHost);

  HIP_CHECK(hipHostUnregister(A));
  delete[] A;
  HIP_CHECK(hipFree(Ad));

  return true;
}

bool hipPerfMemcpy::run_d2d(unsigned int numTests) {
  int *Ad1, *Ad2;
  HIP_CHECK(hipMalloc(&Ad1, totalSizes_[numTests]));
  HIP_CHECK(hipMalloc(&Ad2, totalSizes_[numTests]));
  HIP_CHECK(hipMemset(Ad2, 0x1, totalSizes_[numTests]));


  // measure performance based on host time
  auto all_start = std::chrono::steady_clock::now();

  for (int j = 0; j < NUM_ITER; j++) {
    HIP_CHECK(hipMemcpyAsync(Ad1, Ad2, totalSizes_[numTests], hipMemcpyDeviceToDevice, nullptr));
  }

  HIP_CHECK(hipDeviceSynchronize());

  auto all_end = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::micro> diff = all_end - all_start;

  TestResult(numTests, diff, hipMemcpyDeviceToDevice);

  HIP_CHECK(hipFree(Ad1));
  HIP_CHECK(hipFree(Ad2));

  return true;
}

bool hipPerfMemcpy::run_d2d_nocu(unsigned int numTests) {
  int *Ad1, *Ad2;
  HIP_CHECK(hipMalloc(&Ad1, totalSizes_[numTests]));
  HIP_CHECK(hipMalloc(&Ad2, totalSizes_[numTests]));
  HIP_CHECK(hipMemset(Ad2, 0x1, totalSizes_[numTests]));

  // measure performance based on host time
  auto all_start = std::chrono::steady_clock::now();

  for (int j = 0; j < NUM_ITER; j++) {
    HIP_CHECK(
        hipMemcpyAsync(Ad1, Ad2, totalSizes_[numTests], hipMemcpyDeviceToDeviceNoCU, nullptr));
  }

  HIP_CHECK(hipDeviceSynchronize());

  auto all_end = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::micro> diff = all_end - all_start;

  TestResult(numTests, diff, hipMemcpyDeviceToDeviceNoCU);

  HIP_CHECK(hipFree(Ad1));
  HIP_CHECK(hipFree(Ad2));

  return true;
}

/**
 * Test Description
 * ------------------------
 *  - Verify hipPerfMemcpy status.
 * Test source
 * ------------------------
 *  - perftests/memory/hipPerfMemcpy.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.6
 */

TEST_CASE("Perf_hipPerfMemcpy_test") {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));

  if (numDevices <= 0) {
    SUCCEED("Skipped testcase hipPerfMemcpy as there is no device to test.");
  } else {
    int deviceId = 0;
    HIP_CHECK(hipSetDevice(deviceId));
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, deviceId));

    CONSOLE_PRINT("info: running on bus 0x%x %s with %d CUs and device id: %d\n", props.pciBusID,
                  props.name, props.multiProcessorCount, deviceId);

    hipPerfMemcpy hipPerfMemcpy;
    SECTION("Perf test Host Memory to Device Memory") {
      for (auto testCase = 0; testCase < NUM_SIZE; testCase++) {
        REQUIRE(true == hipPerfMemcpy.run_h2d(testCase));
      }
    }
    SECTION("Perf test Device Memory to Host Memory") {
      for (auto testCase = 0; testCase < NUM_SIZE; testCase++) {
        REQUIRE(true == hipPerfMemcpy.run_d2h(testCase));
      }
    }
    SECTION("Perf test Device Memory to Device Memory") {
      for (auto testCase = 0; testCase < NUM_SIZE; testCase++) {
        REQUIRE(true == hipPerfMemcpy.run_d2d(testCase));
      }
    }
    SECTION("Perf test Device Memory to Device Memory No CU") {
      for (auto testCase = 0; testCase < NUM_SIZE; testCase++) {
        REQUIRE(true == hipPerfMemcpy.run_d2d_nocu(testCase));
      }
    }
  }
}

/**
 * End doxygen group perfMemoryTest.
 * @}
 */
