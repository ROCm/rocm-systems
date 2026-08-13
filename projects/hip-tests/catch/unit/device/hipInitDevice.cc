/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <array>

/**
 * @addtogroup hipInitDevice hipInitDevice
 * @{
 * @ingroup DeviceTest
 * `hipInitDevice(int device, unsigned int deviceFlags, unsigned int flags)` -
 * Initialize the specified device to be used for GPU executions. Unlike `hipSetDevice`,
 * it does not make the device current for the calling thread.
 */

namespace {
constexpr std::array<unsigned int, 4> kScheduleFlags{
    hipDeviceScheduleAuto, hipDeviceScheduleSpin, hipDeviceScheduleYield,
    hipDeviceScheduleBlockingSync};
}  // namespace

/**
 * Test Description
 * ------------------------
 *  - Initialize every device with no flags (flags == 0) and expect success.
 * Test source
 * ------------------------
 *  - unit/device/hipInitDevice.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 10.1
 */
HIP_TEST_CASE(Unit_hipInitDevice_Positive_Basic) {
  const auto deviceCount = HipTest::getDeviceCount();
  for (int dev = 0; dev < deviceCount; ++dev) {
    HIP_CHECK(hipInitDevice(dev, 0, 0));
  }
}

/**
 * Test Description
 * ------------------------
 *  - Initialize every device with valid schedule flags marked as valid, then verify the
 *    flags were applied to that device via `hipGetDeviceFlags` after making it current.
 * Test source
 * ------------------------
 *  - unit/device/hipInitDevice.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 10.1
 */
HIP_TEST_CASE(Unit_hipInitDevice_Positive_Flags) {
  const auto dev = GENERATE(range(0, HipTest::getDeviceCount()));
  const unsigned int flag =
      GENERATE_COPY(from_range(std::begin(kScheduleFlags), std::end(kScheduleFlags)));
  CAPTURE(dev, flag);

  HIP_CHECK(hipInitDevice(dev, flag, hipInitDeviceFlagsAreValid));

  HIP_CHECK(hipSetDevice(dev));
  unsigned int getFlag = 0;
  HIP_CHECK(hipGetDeviceFlags(&getFlag));
#if HT_NVIDIA
  // CUDA backend may set additional flags.
  getFlag = getFlag & hipDeviceScheduleMask;
#endif
  REQUIRE((flag & hipDeviceScheduleMask) == getFlag);
}

/**
 * Test Description
 * ------------------------
 *  - Validate that `hipInitDevice` does not change the current device of the calling thread.
 *  - Requires at least two devices.
 * Test source
 * ------------------------
 *  - unit/device/hipInitDevice.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 10.1
 */
HIP_TEST_CASE(Unit_hipInitDevice_Positive_DoesNotChangeCurrentDevice) {
  const auto deviceCount = HipTest::getDeviceCount();
  if (deviceCount < 2) {
    HIP_SKIP_TEST("Test requires at least two devices");
    return;
  }

  HIP_CHECK(hipSetDevice(0));

  // Initializing a different device must not change the current device.
  HIP_CHECK(hipInitDevice(deviceCount - 1, 0, 0));

  int current = -1;
  HIP_CHECK(hipGetDevice(&current));
  REQUIRE(current == 0);
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid device ordinals:
 *    -# device == -1, device == deviceCount, device == deviceCount + 1
 *      - Expected output: return `hipErrorInvalidDevice`
 * Test source
 * ------------------------
 *  - unit/device/hipInitDevice.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 10.1
 */
HIP_TEST_CASE(Unit_hipInitDevice_Negative_InvalidDevice) {
  const auto deviceCount = HipTest::getDeviceCount();
  const int invalidDevice = GENERATE_COPY(-1, deviceCount, deviceCount + 1);
  CAPTURE(invalidDevice);
  HIP_CHECK_ERROR(hipInitDevice(invalidDevice, 0, 0), hipErrorInvalidDevice);
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of an invalid `flags` argument. The only accepted values are
 *    `0` and `hipInitDeviceFlagsAreValid`.
 *      - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/device/hipInitDevice.cc
 * Test requirements
 * ------------------------
 *  - Platform specific (AMD)
 *  - HIP_VERSION >= 10.1
 */
HIP_TEST_CASE(Unit_hipInitDevice_Negative_InvalidFlags) {
#if HT_AMD
  const unsigned int invalidFlags = GENERATE(0x2u, 0x5u, 0xFFu);
  CAPTURE(invalidFlags);
  HIP_CHECK_ERROR(hipInitDevice(0, 0, invalidFlags), hipErrorInvalidValue);
#else
  HIP_SKIP_TEST("flags validation is AMD-specific");
#endif
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid `deviceFlags` when `flags` is `hipInitDeviceFlagsAreValid`:
 *    -# Multiple mutually-exclusive schedule bits set, or out-of-range bits.
 *      - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/device/hipInitDevice.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 10.1
 */
HIP_TEST_CASE(Unit_hipInitDevice_Negative_InvalidDeviceFlags) {
  const unsigned int invalidDeviceFlags = GENERATE(
      hipDeviceScheduleSpin | hipDeviceScheduleYield,   // two schedule bits set
      hipDeviceScheduleSpin | hipDeviceScheduleBlockingSync,
      0x100000u);                                        // out-of-range bit
  CAPTURE(invalidDeviceFlags);
  HIP_CHECK_ERROR(hipInitDevice(0, invalidDeviceFlags, hipInitDeviceFlagsAreValid),
                  hipErrorInvalidValue);
}

/**
 * End doxygen group hipInitDevice.
 * @}
 */
