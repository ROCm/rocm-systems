/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_helper.hh>

/**
 * @addtogroup hipDeviceEnablePeerAccess hipDeviceEnablePeerAccess
 * @{
 * @ingroup PeerToPeerTest
 * `hipDeviceEnablePeerAccess(int peerDeviceId, unsigned int flags)` -
 * Enable direct access from current device's virtual address space to memory allocations
 * physically located on a peer device.
 */

/**
 * Test Description
 * ------------------------
 *  - Enables peer access for each GPU pair.
 *  - Disables peer access for each GPU pair.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceEnableDisablePeerAccess.cc
 * Test requirements
 * ------------------------
 *  - PeerToPeer supported
 *  - Multi-device
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipDeviceEnableDisablePeerAccess_positive) {
  int canAccessPeer = 0;
  int deviceCount = HipTest::getGeviceCount();
  if (deviceCount < 2) {
    HipTest::HIP_SKIP_TEST("Skipping because devices < 2");
    return;
  }

  int dev = GENERATE(range(0, HipTest::getGeviceCount()));
  int peerDev = GENERATE(range(0, HipTest::getGeviceCount()));

  if (dev != peerDev) {
    HIP_CHECK(hipSetDevice(dev));

    GENERATE_CAPTURE();
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    BEGIN_CAPTURE(stream);
    HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeer, dev, peerDev));
    END_CAPTURE(stream);
    HIP_CHECK(hipStreamDestroy(stream));

    if (canAccessPeer == 0) {
      HipTest::HIP_SKIP_TEST("Skipping because no P2P support");
      return;
    }
    HIP_CHECK(hipDeviceEnablePeerAccess(peerDev, 0));
    HIP_CHECK(hipDeviceDisablePeerAccess(peerDev));
  }
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When peer device ID is invalid (-1 or out of bounds)
 *      - Expected output: return `hipErrorInvalidDevice`
 *    -# When flag is invalid (-1)
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When peer access has already been enabled
 *      - Expected output: return `hipErrorPeerAccessAleadyEnabled`
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceEnableDisablePeerAccess.cc
 * Test requirements
 * ------------------------
 *  - Multi-device
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipDeviceEnablePeerAccess_negative) {
  int deviceCount = HipTest::getGeviceCount();
  if (deviceCount < 2) {
    HipTest::HIP_SKIP_TEST("Skipping because devices < 2");
    return;
  }

  SECTION("peerDeviceId is invalid") {
    HIP_CHECK_ERROR(hipDeviceEnablePeerAccess(-1, 0), hipErrorInvalidDevice);
    HIP_CHECK_ERROR(hipDeviceEnablePeerAccess(deviceCount, 0), hipErrorInvalidDevice);
  }
  SECTION("Flag is invalid") {
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK_ERROR(hipDeviceEnablePeerAccess(0, -1), hipErrorInvalidValue);
  }
  SECTION("Peer Access already enabled") {
    HIP_CHECK(hipSetDevice(0));
    int canAccessPeer = 0;
    HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeer, 1, 0));
    if (canAccessPeer == 0) {
      HipTest::HIP_SKIP_TEST("Skipping because no P2P support");
      return;
    }
    HIP_CHECK(hipDeviceEnablePeerAccess(1, 0));
    HIP_CHECK_ERROR(hipDeviceEnablePeerAccess(1, 0), hipErrorPeerAccessAlreadyEnabled);
    HIP_CHECK(hipDeviceDisablePeerAccess(1));
  }
}

/**
 * End doxygen group hipDeviceEnablePeerAccess.
 * @}
 */

/**
 * @addtogroup hipDeviceDisablePeerAccess hipDeviceDisablePeerAccess
 * @{
 * @ingroup PeerToPeerTest
 * hipDeviceDisablePeerAccess(int peerDeviceId)` -
 * Disable direct access from current device's virtual address space
 * to memory allocations physically located on a peer device.
 * ________________________
 * Test cases from other modules:
 *  - @ref Unit_hipDeviceEnableDisablePeerAccess_positive
 */

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When peer device ID is invalid (-1 or out of bounds)
 *      - Expected output: return `hipErrorInvalidDevice`
 *    -# When peer access is not enabled
 *      - Expected output: return `hipErrorPeerAccessNotEnabled`
 *    -# When peer access is already disabled
 *      - Expected output: return `hipErrorPeerAccessNotEnabled`
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceEnableDisablePeerAccess.cc
 * Test requirements
 * ------------------------
 *  - Multi-device
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipDeviceDisablePeerAccess_negative) {
  int deviceCount = HipTest::getGeviceCount();
  if (deviceCount < 2) {
    HipTest::HIP_SKIP_TEST("Skipping because devices < 2");
    return;
  }

  SECTION("peerDeviceId is invalid") {
    HIP_CHECK_ERROR(hipDeviceDisablePeerAccess(-1), hipErrorInvalidDevice);
    HIP_CHECK_ERROR(hipDeviceDisablePeerAccess(deviceCount), hipErrorInvalidDevice);
  }
  SECTION("Peer Access not enabled") {
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK_ERROR(hipDeviceDisablePeerAccess(1), hipErrorPeerAccessNotEnabled);
  }
  SECTION("Peer Access disabled twice") {
    HIP_CHECK(hipSetDevice(0));
    int canAccessPeer = 0;
    HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeer, 1, 0));
    if (canAccessPeer == 0) {
      HipTest::HIP_SKIP_TEST("Skipping because no P2P support");
      return;
    }
    HIP_CHECK(hipDeviceEnablePeerAccess(1, 0));
    HIP_CHECK(hipDeviceDisablePeerAccess(1));
    HIP_CHECK_ERROR(hipDeviceDisablePeerAccess(1), hipErrorPeerAccessNotEnabled);
  }
}
