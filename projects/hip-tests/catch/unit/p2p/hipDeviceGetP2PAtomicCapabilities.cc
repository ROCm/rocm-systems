/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstdlib>
#include <hip_test_common.hh>
#include <hip_test_helper.hh>
#include "hip/hip_runtime_api.h"

/**
 * @addtogroup hipDeviceGetP2PAtomicCapabilities hipDeviceGetP2PAtomicCapabilities
 * @{
 * @ingroup DriverTest
 * `hipDeviceGetP2PAtomicCapabilities(unsigned int* capabilities,
 * const hipAtomicOperation* operations, unsigned int count,
 * int srcDevice, int dstDevice)` -
 * Reports, per atomic operation, which native atomic capabilities are supported
 * over the P2P link between two devices.
 */

namespace {
constexpr hipAtomicOperation kAllOperations[] = {
    hipAtomicOperationIntegerAdd,       hipAtomicOperationIntegerMin,
    hipAtomicOperationIntegerMax,       hipAtomicOperationIntegerIncrement,
    hipAtomicOperationIntegerDecrement, hipAtomicOperationAnd,
    hipAtomicOperationOr,               hipAtomicOperationXOR,
    hipAtomicOperationExchange,         hipAtomicOperationCAS,
    hipAtomicOperationFloatAdd,         hipAtomicOperationFloatMin,
    hipAtomicOperationFloatMax};

constexpr unsigned int kAllCapabilityBits =
    hipAtomicCapabilitySigned | hipAtomicCapabilityUnsigned | hipAtomicCapabilityReduction |
    hipAtomicCapabilityScalar32 | hipAtomicCapabilityScalar64 | hipAtomicCapabilityScalar128 |
    hipAtomicCapabilityVector32x4;
}  // namespace

/**
 * Test Description
 * ------------------------
 *  - Queries the P2P atomic capabilities for every atomic operation across all
 *    pairs of devices and verifies each returned bitmask only contains defined
 *    capability bits.
 * Test source
 * ------------------------
 *  - catch/unit/p2p/hipDeviceGetP2PAtomicCapabilities.cc
 * Test requirements
 * ------------------------
 *  - Multiple devices
 */
HIP_TEST_CASE(Unit_hipDeviceGetP2PAtomicCapabilities_Basic) {
  int deviceCount = HipTest::getGeviceCount();
  if (deviceCount < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }

  constexpr unsigned int count = sizeof(kAllOperations) / sizeof(kAllOperations[0]);

  for (int srcDevice = 0; srcDevice < deviceCount; ++srcDevice) {
    for (int dstDevice = 0; dstDevice < deviceCount; ++dstDevice) {
      if (srcDevice == dstDevice) {
        continue;
      }
      unsigned int capabilities[count] = {};
      HIP_CHECK(
          hipDeviceGetP2PAtomicCapabilities(capabilities, kAllOperations, count, srcDevice, dstDevice));
      for (unsigned int i = 0; i < count; ++i) {
        INFO("operation: " << kAllOperations[i] << "\nsrcDevice: " << srcDevice
                           << "\ndstDevice: " << dstDevice << "\ncapabilities: " << capabilities[i]);
        // Every reported bit must be one of the defined capability flags.
        REQUIRE((capabilities[i] & ~kAllCapabilityBits) == 0u);
      }
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - Verifies handling of invalid arguments:
 *    -# When the output capabilities pointer is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When the operations pointer is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When count is zero
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When an operation value is invalid
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When a device ordinal is negative or out of bounds
 *      - Expected output: return `hipErrorInvalidDevice`
 *    -# When the src and dst devices are the same one
 *      - Expected output: return `hipErrorInvalidDevice`
 * Test source
 * ------------------------
 *  - catch/unit/p2p/hipDeviceGetP2PAtomicCapabilities.cc
 * Test requirements
 * ------------------------
 *  - Multiple devices
 */
HIP_TEST_CASE(Unit_hipDeviceGetP2PAtomicCapabilities_Negative) {
  int deviceCount = HipTest::getGeviceCount();
  if (deviceCount < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }

  const int validSrcDevice = 0;
  const int validDstDevice = 1;
  hipAtomicOperation operations[] = {hipAtomicOperationIntegerAdd};
  unsigned int capabilities[1] = {};

  SECTION("Nullptr capabilities") {
    HIP_CHECK_ERROR(
        hipDeviceGetP2PAtomicCapabilities(nullptr, operations, 1, validSrcDevice, validDstDevice),
        hipErrorInvalidValue);
  }

  SECTION("Nullptr operations") {
    HIP_CHECK_ERROR(
        hipDeviceGetP2PAtomicCapabilities(capabilities, nullptr, 1, validSrcDevice, validDstDevice),
        hipErrorInvalidValue);
  }

  SECTION("Count is zero") {
    HIP_CHECK_ERROR(hipDeviceGetP2PAtomicCapabilities(capabilities, operations, 0, validSrcDevice,
                                                      validDstDevice),
                    hipErrorInvalidValue);
  }

  SECTION("Invalid operation") {
    hipAtomicOperation invalidOps[] = {static_cast<hipAtomicOperation>(1000)};
    HIP_CHECK_ERROR(hipDeviceGetP2PAtomicCapabilities(capabilities, invalidOps, 1, validSrcDevice,
                                                      validDstDevice),
                    hipErrorInvalidValue);
  }

  SECTION("Device is -1") {
    HIP_CHECK_ERROR(
        hipDeviceGetP2PAtomicCapabilities(capabilities, operations, 1, -1, validDstDevice),
        hipErrorInvalidDevice);
    HIP_CHECK_ERROR(
        hipDeviceGetP2PAtomicCapabilities(capabilities, operations, 1, validSrcDevice, -1),
        hipErrorInvalidDevice);
  }

  SECTION("Device is out of bounds") {
    int count = 0;
    HIP_CHECK(hipGetDeviceCount(&count));
    REQUIRE_FALSE(count == 0);
    HIP_CHECK_ERROR(
        hipDeviceGetP2PAtomicCapabilities(capabilities, operations, 1, count, validDstDevice),
        hipErrorInvalidDevice);
    HIP_CHECK_ERROR(
        hipDeviceGetP2PAtomicCapabilities(capabilities, operations, 1, validSrcDevice, count),
        hipErrorInvalidDevice);
  }

  SECTION("Source and destination devices are the same") {
    HIP_CHECK_ERROR(hipDeviceGetP2PAtomicCapabilities(capabilities, operations, 1, validSrcDevice,
                                                      validSrcDevice),
                    hipErrorInvalidDevice);
  }
}

/**
 * End doxygen group DriverTest.
 * @}
 */
