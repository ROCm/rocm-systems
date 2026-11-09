/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <resource_guards.hh>
#include <algorithm>
#include <array>
#include <atomic>
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
HIP_TEST_CASE(Unit_hipGetSetDeviceFlags_NullptrFlag) {
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
HIP_TEST_CASE(Unit_hipGetSetDeviceFlags_ValidFlag) {
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
HIP_TEST_CASE(Unit_hipGetSetDeviceFlags_SetThenGet) {
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
HIP_TEST_CASE(Unit_hipGetSetDeviceFlags_Threaded) {
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
 *  - Change the scheduling policy after queues have allocated their completion signals.
 *  - Check that callbacks, memory operations and synchronization still complete as signals
 *    are reused across repeated policy changes.
 * Test source
 * ------------------------
 *  - unit/device/hipGetSetDeviceFlags.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipGetSetDeviceFlags_CompletionAfterPolicyChange) {
  unsigned int saved_flags = 0;
  HIP_CHECK(hipGetDeviceFlags(&saved_flags));
  struct RestoreFlags {
    unsigned int flags;
    ~RestoreFlags() { static_cast<void>(hipSetDeviceFlags(flags)); }
  } restore{saved_flags};

  const auto initial = GENERATE(hipDeviceScheduleSpin, hipDeviceScheduleBlockingSync);
  const auto alternate =
      initial == hipDeviceScheduleSpin ? hipDeviceScheduleBlockingSync : hipDeviceScheduleSpin;
  HIP_CHECK(hipSetDeviceFlags(initial));

  constexpr size_t bytes_per_stream = 4096;
  constexpr size_t iterations = 128;
  LinearAllocGuard<unsigned char> memory(LinearAllocs::hipMalloc, 2 * bytes_per_stream);
  std::atomic<size_t> callbacks{0};
  std::array<StreamGuard, 2> streams{StreamGuard(Streams::withFlags, hipStreamNonBlocking),
                                     StreamGuard(Streams::withFlags, hipStreamNonBlocking)};
  struct DrainStreams {
    const std::array<StreamGuard, 2>& streams;
    ~DrainStreams() {
      // Keep callback data alive even when a test assertion unwinds this scope.
      for (const auto& stream : streams) {
        static_cast<void>(hipStreamSynchronize(stream.stream()));
      }
    }
  } drain{streams};
  for (size_t i = 0; i < streams.size(); ++i) {
    HIP_CHECK(hipMemsetAsync(memory.ptr() + i * bytes_per_stream, 0, bytes_per_stream,
                             streams[i].stream()));
    HIP_CHECK(hipStreamSynchronize(streams[i].stream()));
  }

  size_t expected_callbacks = 0;
  for (auto policy : {alternate, initial, alternate}) {
    CAPTURE(initial, policy);
    HIP_CHECK(hipSetDeviceFlags(policy));
    // Exceed the default signal-pool size, so both existing and reused signals are exercised.
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
      for (size_t i = 0; i < streams.size(); ++i) {
        HIP_CHECK(hipMemsetAsync(memory.ptr() + i * bytes_per_stream, iteration, bytes_per_stream,
                                 streams[i].stream()));
        HIP_CHECK(hipLaunchHostFunc(
            streams[i].stream(),
            [](void* data) { static_cast<std::atomic<size_t>*>(data)->fetch_add(1); }, &callbacks));
        ++expected_callbacks;
      }
    }
    for (const auto& stream : streams) {
      HIP_CHECK(hipStreamSynchronize(stream.stream()));
    }
    REQUIRE(callbacks.load() == expected_callbacks);
    std::array<unsigned char, 2 * bytes_per_stream> result{};
    HIP_CHECK(hipMemcpy(result.data(), memory.ptr(), result.size(), hipMemcpyDeviceToHost));
    REQUIRE(std::all_of(result.begin(), result.end(),
                        [](unsigned char value) { return value == iterations - 1; }));
  }
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
HIP_TEST_CASE(Unit_hipGetDeviceFlags_Positive_Context) {
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
HIP_TEST_CASE(Unit_hipGetSetDeviceFlags_InvalidFlag) {
  const unsigned int invalidFlag = GENERATE(0xb011,     // schedule flags should not overlap
                                            0xb101,     // schedule flags should not overlap
                                            0xb110,     // schedule flags should not overlap
                                            0xb111,     // schedule flags should not overlap
                                            0xb100000,  // out of bounds
                                            0xFFFF);
  CAPTURE(invalidFlag);
  HIP_CHECK_ERROR(hipSetDeviceFlags(invalidFlag), hipErrorInvalidValue);
}
