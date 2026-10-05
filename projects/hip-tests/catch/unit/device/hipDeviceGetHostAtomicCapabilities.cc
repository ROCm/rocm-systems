/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <vector>

#include <hip_test_common.hh>

/**
 * @addtogroup hipDeviceGetHostAtomicCapabilities hipDeviceGetHostAtomicCapabilities
 * @{
 * @ingroup DeviceTest
 * `hipDeviceGetHostAtomicCapabilities(unsigned int* capabilities,
 *                                     const hipAtomicOperation* operations,
 *                                     unsigned int count, int device)` -
 * Queries atomic operations supported between a device and the host.
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

constexpr unsigned int kOperationCount = sizeof(kAllOperations) / sizeof(kAllOperations[0]);

// hipAtomicOperationMax is one past the last valid operation.
static_assert(kOperationCount == static_cast<unsigned int>(hipAtomicOperationMax),
              "kAllOperations must list every valid hipAtomicOperation");

constexpr unsigned int kAllCapabilityBits =
    hipAtomicCapabilitySigned | hipAtomicCapabilityUnsigned | hipAtomicCapabilityReduction |
    hipAtomicCapabilityScalar32 | hipAtomicCapabilityScalar64 | hipAtomicCapabilityScalar128 |
    hipAtomicCapabilityVector32x4;

constexpr unsigned int kScalarWidthBits =
    hipAtomicCapabilityScalar32 | hipAtomicCapabilityScalar64 | hipAtomicCapabilityScalar128;

// A value no valid call writes, so it proves the output was left alone.
constexpr unsigned int kSentinel = 0xDEADBEEFu;

}  // namespace

/**
 * Test Description
 * ------------------------
 *  - Query every valid operation: the call succeeds, every entry is written, and no
 *    undefined capability bit is set.
 *  - A non-zero bitmask must name an operand width, otherwise it tells a caller nothing.
 *  - Increment and decrement must never claim signed support - atomicInc/atomicDec exist
 *    only in an unsigned form, which holds on every device.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceGetHostAtomicCapabilities.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 */
HIP_TEST_CASE(Unit_hipDeviceGetHostAtomicCapabilities_Positive_AllOperations) {
  const int device = GENERATE(range(0, HipTest::getDeviceCount()));

  std::vector<unsigned int> capabilities(kOperationCount, kSentinel);
  HIP_CHECK(hipDeviceGetHostAtomicCapabilities(capabilities.data(), kAllOperations,
                                               kOperationCount, device));

  for (unsigned int i = 0; i < kOperationCount; ++i) {
    INFO("operation index " << i);
    REQUIRE(capabilities[i] != kSentinel);
    REQUIRE((capabilities[i] & ~kAllCapabilityBits) == 0);
    if (capabilities[i] != 0) {
      REQUIRE((capabilities[i] & kScalarWidthBits) != 0);

      const hipAtomicOperation op = kAllOperations[i];
      if (op == hipAtomicOperationIntegerIncrement || op == hipAtomicOperationIntegerDecrement) {
        REQUIRE((capabilities[i] & hipAtomicCapabilitySigned) == 0);
      }
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - Null output, null input, and a zero count are rejected with hipErrorInvalidValue.
 *  - A device ordinal below zero or at/above the device count is rejected with
 *    hipErrorInvalidDevice.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceGetHostAtomicCapabilities.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 */
HIP_TEST_CASE(Unit_hipDeviceGetHostAtomicCapabilities_Negative_Parameters) {
  const int deviceCount = HipTest::getDeviceCount();
  const hipAtomicOperation operation = hipAtomicOperationIntegerAdd;
  unsigned int capability = kSentinel;

  SECTION("capabilities is nullptr") {
    REQUIRE(hipDeviceGetHostAtomicCapabilities(nullptr, &operation, 1, 0) ==
            hipErrorInvalidValue);
  }

  SECTION("operations is nullptr") {
    REQUIRE(hipDeviceGetHostAtomicCapabilities(&capability, nullptr, 1, 0) ==
            hipErrorInvalidValue);
  }

  SECTION("count is zero") {
    REQUIRE(hipDeviceGetHostAtomicCapabilities(&capability, &operation, 0, 0) ==
            hipErrorInvalidValue);
  }

  SECTION("device is negative") {
    REQUIRE(hipDeviceGetHostAtomicCapabilities(&capability, &operation, 1, -1) ==
            hipErrorInvalidDevice);
  }

  SECTION("device is out of range") {
    REQUIRE(hipDeviceGetHostAtomicCapabilities(&capability, &operation, 1, deviceCount) ==
            hipErrorInvalidDevice);
  }
}

/**
 * Test Description
 * ------------------------
 *  - An out-of-domain operation, including the hipAtomicOperationMax sentinel, is rejected
 *    with hipErrorInvalidValue.
 *  - Rejection applies to the whole request and leaves the output untouched, so a caller
 *    that checks the return code can trust nothing was partially written.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceGetHostAtomicCapabilities.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 */
HIP_TEST_CASE(Unit_hipDeviceGetHostAtomicCapabilities_Negative_InvalidOperation) {
  unsigned int capability = kSentinel;

  SECTION("sentinel value") {
    const hipAtomicOperation operation = hipAtomicOperationMax;
    REQUIRE(hipDeviceGetHostAtomicCapabilities(&capability, &operation, 1, 0) ==
            hipErrorInvalidValue);
    REQUIRE(capability == kSentinel);
  }

  SECTION("negative value") {
    const hipAtomicOperation operation = static_cast<hipAtomicOperation>(-1);
    REQUIRE(hipDeviceGetHostAtomicCapabilities(&capability, &operation, 1, 0) ==
            hipErrorInvalidValue);
    REQUIRE(capability == kSentinel);
  }

  SECTION("invalid operation after valid ones leaves output untouched") {
    constexpr unsigned int kCount = 3;
    const hipAtomicOperation operations[kCount] = {
        hipAtomicOperationIntegerAdd, hipAtomicOperationCAS, hipAtomicOperationMax};
    unsigned int capabilities[kCount] = {kSentinel, kSentinel, kSentinel};

    REQUIRE(hipDeviceGetHostAtomicCapabilities(capabilities, operations, kCount, 0) ==
            hipErrorInvalidValue);
    for (unsigned int i = 0; i < kCount; ++i) {
      INFO("operation index " << i);
      REQUIRE(capabilities[i] == kSentinel);
    }
  }
}

/**
* End doxygen group DeviceTest.
* @}
*/
