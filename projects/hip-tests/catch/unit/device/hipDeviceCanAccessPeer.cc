/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_helper.hh>

/**
 * @addtogroup hipDeviceCanAccessPeer hipDeviceCanAccessPeer
 * @{
 * @ingroup PeerToPeerTest
 * `hipDeviceCanAccessPeer(int* canAccessPeer, int deviceId, int peerDeviceId)` -
 * Determine if a device can access a peer's memory.
 */

/**
 * Test Description
 * ------------------------
 *  - Verifies that each available device can access memory from all other devices.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceCanAccessPeer.cc
 * Test requirements
 * ------------------------
 *  - Multi-device
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_hipDeviceCanAccessPeer_positive) {
  int canAccessPeer = 0;
  int deviceCount = HipTest::getGeviceCount();
  if (deviceCount < 2) {
    HipTest::HIP_SKIP_TEST("Skipping because devices < 2");
    return;
  }

  int dev = GENERATE(range(0, HipTest::getGeviceCount()));
  int peerDev = GENERATE(range(0, HipTest::getGeviceCount()));

  GENERATE_CAPTURE();
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  BEGIN_CAPTURE(stream);
  HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeer, dev, peerDev));
  END_CAPTURE(stream);
  HIP_CHECK(hipStreamDestroy(stream));

  if (dev != peerDev) {
    REQUIRE(canAccessPeer >= 0);
  } else {
    REQUIRE(canAccessPeer == 0);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Verifies handling of invalid arguments:
 *    -# When output pointer to the peer result is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When device ID is invalid (-1 or out of bounds)
 *      - Expected output: return `hipErrorInvalidDevice`
 *    -# When peer device ID is invalid (-1 or out of bounds)
 *      - Expected output: return `hipErrorInvalidDevice`
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceCanAccessPeer.cc
 * Test requirements
 * ------------------------
 *  - Multi-device
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_hipDeviceCanAccessPeer_negative) {
  int canAccessPeer = 0;
  int deviceCount = HipTest::getGeviceCount();
  if (deviceCount < 2) {
    HipTest::HIP_SKIP_TEST("Skipping because devices < 2");
    return;
  }

  SECTION("canAccessPeer is nullptr") {
    HIP_CHECK_ERROR(hipDeviceCanAccessPeer(nullptr, 0, 1), hipErrorInvalidValue);
  }

  SECTION("deviceId is invalid") {
    HIP_CHECK_ERROR(hipDeviceCanAccessPeer(&canAccessPeer, -1, 1), hipErrorInvalidDevice);
    HIP_CHECK_ERROR(hipDeviceCanAccessPeer(&canAccessPeer, deviceCount, 1), hipErrorInvalidDevice);
  }

  SECTION("peerDeviceId is invalid") {
    HIP_CHECK_ERROR(hipDeviceCanAccessPeer(&canAccessPeer, 0, -1), hipErrorInvalidDevice);
    HIP_CHECK_ERROR(hipDeviceCanAccessPeer(&canAccessPeer, 0, deviceCount), hipErrorInvalidDevice);
  }
}

/**
 * End doxygen group PeerToPeerTest.
 * @}
 */
