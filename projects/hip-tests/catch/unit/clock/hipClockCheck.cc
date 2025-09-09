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

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip/hip_ext.h>

/**
 * @addtogroup clock clock
 * @{
 * @ingroup DeviceLanguageTest
 * Contains unit tests for clock, clock64 and wall_clock64 APIs
 */

__global__ void kernel_c64(int clock_rate, uint64_t wait_t) {
  uint64_t start = clock64() / clock_rate, cur = 0;  // in ms
  do {
    cur = clock64() / clock_rate - start;
  } while (cur < wait_t);
}

__global__ void kernel_c(int clock_rate, uint64_t wait_t) {
  uint64_t start = clock() / clock_rate, cur = 0;  // in ms
  do {
    cur = clock() / clock_rate - start;
  } while (cur < wait_t);
}

__global__ void kernel_wc64(int clock_rate, uint64_t wait_t) {
  uint64_t start = wall_clock64() / clock_rate, cur = 0;  // in ms
  do {
    cur = wall_clock64() / clock_rate - start;
  } while (cur < wait_t);
}

bool verify_time_execution(float ratio, float time1, float time2, float expected_time1,
                           float expected_time2) {
  bool test_status = false;

#if (HT_WIN == 1)
  if (time1 > ratio * expected_time1 && time2 > ratio * expected_time2) {
#else
  if (fabs(time1 - expected_time1) < (ratio * expected_time1) &&
      fabs(time2 - expected_time2) < (ratio * expected_time2)) {
#endif
    INFO("Succeeded: Expected Vs Actual: Kernel1 - " << expected_time1 << " Vs " << time1
                                                     << ", Kernel2 - " << expected_time2 << " Vs "
                                                     << time2);
    test_status = true;
  } else {
    INFO("Failed: Expected Vs Actual: Kernel1 -" << expected_time1 << " Vs " << time1
                                                 << ", Kernel2 - " << expected_time2 << " Vs "
                                                 << time2);
    test_status = false;
  }
  return test_status;
}

/*
 * Launching kernel1 and kernel2 and then we try to
 * get the event elapsed time of each kernel using the start and
 * end events.The event elapsed time should return us the kernel
 * execution time for that particular kernel
 */
bool kernel_time_execution(void (*kernel)(int, uint64_t), int clock_rate, uint64_t expected_time1,
                           uint64_t expected_time2) {
  hipStream_t stream;
  hipEvent_t start_event1, end_event1, start_event2, end_event2;
  float time1 = 0, time2 = 0;
  HIP_CHECK(hipEventCreate(&start_event1));
  HIP_CHECK(hipEventCreate(&end_event1));
  HIP_CHECK(hipEventCreate(&start_event2));
  HIP_CHECK(hipEventCreate(&end_event2));
  HIP_CHECK(hipStreamCreate(&stream));
  hipExtLaunchKernelGGL(kernel, dim3(1), dim3(1), 0, stream, start_event1, end_event1, 0,
                        clock_rate, expected_time1);
  hipExtLaunchKernelGGL(kernel, dim3(1), dim3(1), 0, stream, start_event2, end_event2, 0,
                        clock_rate, expected_time2);
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipEventElapsedTime(&time1, start_event1, end_event1));
  HIP_CHECK(hipEventElapsedTime(&time2, start_event2, end_event2));

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipEventDestroy(start_event1));
  HIP_CHECK(hipEventDestroy(end_event1));
  HIP_CHECK(hipEventDestroy(start_event2));
  HIP_CHECK(hipEventDestroy(end_event2));

#if HT_WIN == 1
  float ratio = 1.0f;
#else
  float ratio = kernel == kernel_wc64 ? 0.01 : 0.5;
#endif

  return verify_time_execution(ratio, time1, time2, expected_time1, expected_time2);
}

/**
 * Test Description
 * ------------------------
 *  - Launches two kernels that run for a specified amount of time passed as a kernel argument by
 * using device function clock64. Kernel execution time is calculated through elapsed time between
 * the start and end event, and calculated time is compared with passed time values.
 * Test source
 * ------------------------
 *  - catch/unit/clock/hipClockCheck.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipClock64_Positive_Basic") {
  HIP_CHECK(hipSetDevice(0));
  int clock_rate = 0;  // in kHz
  HIP_CHECK(hipDeviceGetAttribute(&clock_rate, hipDeviceAttributeClockRate, 0));
  if (clock_rate == 0) {
    HipTest::HIP_SKIP_TEST("hipDeviceAttributeClockRate returns 0");
    return;
  }
  if (IsGfx11()) {
    HipTest::HIP_SKIP_TEST("Issue with clock64() function on gfx11 devices!");
    return;
  }

  const auto expected_time1 = GENERATE(1000, 1500, 2000);
  const auto expected_time2 = expected_time1 / 2;

  REQUIRE(kernel_time_execution(kernel_c64, clock_rate, expected_time1, expected_time2));
}

/**
 * Test Description
 * ------------------------
 *  - Launches two kernels that run for a specified amount of time passed as a kernel argument by
 * using device function clock. Kernel execution time is calculated through elapsed time between
 * the start and end event, and calculated time is compared with passed time values.
 * Test source
 * ------------------------
 *  - catch/unit/clock/hipClockCheck.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipClock_Positive_Basic") {
  HIP_CHECK(hipSetDevice(0));
  int clock_rate = 0;  // in kHz
  HIP_CHECK(hipDeviceGetAttribute(&clock_rate, hipDeviceAttributeClockRate, 0));
  if (clock_rate == 0) {
    HipTest::HIP_SKIP_TEST("hipDeviceAttributeClockRate returns 0");
    return;
  }
  if (IsGfx11()) {
    HipTest::HIP_SKIP_TEST("Issue with clock() function on gfx11 devices!");
    return;
  }

  const auto expected_time1 = GENERATE(1000, 1500, 2000);
  const auto expected_time2 = expected_time1 / 2;

  REQUIRE(kernel_time_execution(kernel_c, clock_rate, expected_time1, expected_time2));
}

/**
 * Test Description
 * ------------------------
 *  - Launches two kernels that run for a specified amount of time passed as a kernel argument by
 * using device function wall_clock64. Kernel execution time is calculated through elapsed time
 * between the start and end event, and calculated time is compared with passed time values.
 * Test source
 * ------------------------
 *  - catch/unit/clock/hipClockCheck.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipWallClock64_Positive_Basic") {
  HIP_CHECK(hipSetDevice(0));
  int clock_rate = 0;  // in kHz
  HIP_CHECK(hipDeviceGetAttribute(&clock_rate, hipDeviceAttributeWallClockRate, 0));

  if (!clock_rate) {
    HipTest::HIP_SKIP_TEST("hipDeviceAttributeWallClockRate returns 0");
    return;
  }

  const auto expected_time1 = GENERATE(1000, 1500, 2000);
  const auto expected_time2 = expected_time1 / 2;

  REQUIRE(kernel_time_execution(kernel_wc64, clock_rate, expected_time1, expected_time2));
}

/**
 * End doxygen group DeviceLanguageTest.
 * @}
 */
