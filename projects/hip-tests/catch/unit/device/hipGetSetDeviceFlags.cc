/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <iterator>
#include <vector>
#include <mutex>
#include <condition_variable>

/**
 * @addtogroup hipGetDeviceFlags hipGetDeviceFlags
 * @{
 * @ingroup DeviceTest
 * `hipGetDeviceFlags(unsigned int* flags)` -
 * Gets the flags set for current device.
 */

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When output pointer to the flag is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/device/hipGetSetDeviceFlags.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGetSetDeviceFlags_NullptrFlag") {
  HIP_CHECK_ERROR(hipGetDeviceFlags(nullptr), hipErrorInvalidValue);
}

std::array<unsigned int, 16> getValidFlags() {
  constexpr std::array<unsigned int, 4> scheduleFlags{hipDeviceScheduleAuto, hipDeviceScheduleSpin,
                                                      hipDeviceScheduleYield,
                                                      hipDeviceScheduleBlockingSync};
  constexpr std::array<unsigned int, 2> hostMapFlags{0, hipDeviceMapHost};
  constexpr std::array<unsigned int, 2> localMemResizeFlags{0, 0x10};  // FIXME EXSWCPHIPT-110
  constexpr size_t size = scheduleFlags.size() * hostMapFlags.size() * localMemResizeFlags.size();
  std::array<unsigned int, size> validFlags;
  int i = 0;
  for (auto sf : scheduleFlags) {
    for (auto hf : hostMapFlags) {
      for (auto lf : localMemResizeFlags) {
        validFlags[i] = sf | hf | lf;
        i += 1;
      }
    }
  }
  return validFlags;
}

/**
 * Test Description
 * ------------------------
 *  - Check returned flags against Cartesian product of all
 *    possible valid flag combinations.
 * Test source
 * ------------------------
 *  - unit/device/hipGetSetDeviceFlags.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGetSetDeviceFlags_ValidFlag") {
  auto validFlags = getValidFlags();

  unsigned int flag = 0;
  HIP_CHECK(hipGetDeviceFlags(&flag));
  REQUIRE(std::find(std::begin(validFlags), std::end(validFlags), flag) != std::end(validFlags));
}

/**
 * Test Description
 * ------------------------
 *  - Validate that returned flags are equal to the ones that have
 *    been previously set.
 *  - Perform validation for all connected devices and all flag combinations.
 * Test source
 * ------------------------
 *  - unit/device/hipGetSetDeviceFlags.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGetSetDeviceFlags_SetThenGet") {
  auto validFlags = getValidFlags();

  auto devNo = GENERATE(range(0, HipTest::getDeviceCount()));
  HIP_CHECK(hipSetDevice(devNo));

  const unsigned int flag = GENERATE_COPY(from_range(std::begin(validFlags), std::end(validFlags)));
  HIP_CHECK(hipSetDeviceFlags(flag));

  unsigned int getFlag;
  HIP_CHECK(hipGetDeviceFlags(&getFlag));
// flags other than hipDeviceSchedule* are ignore on the ROCm backend
#if HT_NVIDIA
  // CUDA backend will sometimes set other flags
  getFlag = getFlag & hipDeviceScheduleMask;
#endif
  REQUIRE((flag & hipDeviceScheduleMask) == getFlag);
}

/**
 * Test Description
 * ------------------------
 *  - Validate that the returned flags from the main thread are
 *    equal to the flags that are set from another thread.
 * Test source
 * ------------------------
 *  - unit/device/hipGetSetDeviceFlags.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGetSetDeviceFlags_Threaded") {
  auto validFlags = getValidFlags();

  auto devNo = GENERATE(range(0, HipTest::getDeviceCount()));
  HIP_CHECK(hipSetDevice(devNo));

  std::mutex mut;
  std::condition_variable cv;
  bool ready = false;  // required to avoid spurious wakeups

  const unsigned int flag = GENERATE_COPY(from_range(std::begin(validFlags), std::end(validFlags)));
  std::thread test_thread([&mut, &ready, &cv, devNo, flag]() {
    std::unique_lock<std::mutex> lock(mut);
    cv.wait(lock, [&ready] { return ready; });
    unsigned int getFlag;
    HIP_CHECK_THREAD(hipSetDevice(devNo));
    HIP_CHECK_THREAD(hipGetDeviceFlags(&getFlag));
// flags other than hipDeviceSchedule* are ignore on the ROCm backend
#if HT_NVIDIA
    // CUDA backend will set other flags we aren't concerned about
    getFlag = getFlag & hipDeviceScheduleMask;
#endif
    REQUIRE_THREAD((flag & hipDeviceScheduleMask) == getFlag);
  });

  {
    std::lock_guard<std::mutex> lock(mut);
    HIP_CHECK(hipSetDeviceFlags(flag));
    ready = true;
  }

  cv.notify_one();

  test_thread.join();
  HIP_CHECK_THREAD_FINALIZE();
}

/**
 * Test Description
 * ------------------------
 *  - Create context with flags and validate that valid
 *    flags are returned.
 *  - Perform validation for all connected devices and all flag combinations.
 * Test source
 * ------------------------
 *  - unit/device/hipGetSetDeviceFlags.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGetDeviceFlags_Positive_Context") {
  auto validFlags = getValidFlags();
  const unsigned int flags =
      GENERATE_COPY(from_range(std::begin(validFlags), std::end(validFlags)));

  HIP_CHECK(hipInit(0));

  hipCtx_t ctx;
  HIP_CHECK(hipCtxCreate(&ctx, flags, 0));

  unsigned int actual_flags;
  HIP_CHECK(hipGetDeviceFlags(&actual_flags));

  REQUIRE(actual_flags == flags);

  HIP_CHECK(hipCtxPopCurrent(&ctx));
  HIP_CHECK(hipCtxDestroy(ctx));
}

/**
 * End doxygen group hipGetDeviceFlags.
 * @}
 */

/**
 * @addtogroup hipSetDeviceFlags hipSetDeviceFlags
 * @{
 * @ingroup DeviceTest
 * `hipSetDeviceFlags(unsigned flags)` -
 * The current device behavior is changed according the flags passed.
 * ________________________
 * Test cases from other modules:
 *  - @ref Unit_hipGetSetDeviceFlags_SetThenGet
 *  - @ref Unit_hipGetSetDeviceFlags_Threaded
 */

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When flag combinations are invalid
 *      - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/device/hipGetSetDeviceFlags.cc
 * Test requirements
 * ------------------------
 *  - Platform specific (NVIDIA)
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGetSetDeviceFlags_InvalidFlag") {
  const unsigned int invalidFlag = GENERATE(0xb011,     // schedule flags should not overlap
                                            0xb101,     // schedule flags should not overlap
                                            0xb110,     // schedule flags should not overlap
                                            0xb111,     // schedule flags should not overlap
                                            0xb100000,  // out of bounds
                                            0xFFFF);
  CAPTURE(invalidFlag);
  HIP_CHECK_ERROR(hipSetDeviceFlags(invalidFlag), hipErrorInvalidValue);
}
