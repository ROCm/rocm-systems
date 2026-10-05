/*
 * Copyright Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstdint>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

namespace {
constexpr hipDeviceP2PAttr kP2PAttribute = hipDevP2PAttrAccessSupported;

bool P2PAttributeQuerySupported(int device) {
  int value = -1;
  const hipError_t status = hipDeviceGetP2PAttribute(&value, kP2PAttribute, device, device);
  return status != hipErrorNotSupported;
}

void SkipIfP2PAttributeUnsupported(int device) {
  if (!P2PAttributeQuerySupported(device)) {
    HIP_SKIP_TEST("hipDeviceGetP2PAttribute is not supported by this runtime path.");
  }
}

constexpr hipAtomicOperation kAtomicOperation = hipAtomicOperationIntegerAdd;
}  // namespace

// @asserts: hipDeviceGetP2PAttribute - rejects a same-device (src==dst) P2P attribute query with a non-success status
HIP_TEST_CASE(Contract_PeerQuery_HipDeviceGetP2PAttribute_SelfDevice_IsRejected) {
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  SkipIfP2PAttributeUnsupported(device);

  int value = -1;
  REQUIRE(hipDeviceGetP2PAttribute(&value, kP2PAttribute, device, device) != hipSuccess);
}

// @asserts: hipDeviceGetP2PAttribute - rejects null output, an unknown attribute enum, and out-of-range device ids with a non-success status
HIP_TEST_CASE(Contract_PeerQuery_HipDeviceGetP2PAttribute_InvalidArgs_AreRejected) {
  int device = 0;
  int device_count = 0;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceCount(&device_count));
  SkipIfP2PAttributeUnsupported(device);

  int value = -1;
  REQUIRE(hipDeviceGetP2PAttribute(nullptr, kP2PAttribute, device, device) != hipSuccess);
  REQUIRE(hipDeviceGetP2PAttribute(&value, static_cast<hipDeviceP2PAttr>(0x7fffffff), device,
                                   device) != hipSuccess);
  REQUIRE(hipDeviceGetP2PAttribute(&value, kP2PAttribute, -1, device) != hipSuccess);
  REQUIRE(hipDeviceGetP2PAttribute(&value, kP2PAttribute, device_count, device) != hipSuccess);
  REQUIRE(hipDeviceGetP2PAttribute(&value, kP2PAttribute, device, -1) != hipSuccess);
  REQUIRE(hipDeviceGetP2PAttribute(&value, kP2PAttribute, device, device_count) != hipSuccess);
}

// @asserts: hipDeviceGetP2PAtomicCapabilities - rejects a null capabilities array, a null operations array, and a zero operation count with a non-success status
HIP_TEST_CASE(Contract_PeerQuery_HipDeviceGetP2PAtomicCapabilities_NullOrEmptyArgs_AreRejected) {
  unsigned int capabilities = 0;
  const hipAtomicOperation operations[] = {kAtomicOperation};

  REQUIRE(hipDeviceGetP2PAtomicCapabilities(nullptr, operations, 1, 0, 1) != hipSuccess);
  REQUIRE(hipDeviceGetP2PAtomicCapabilities(&capabilities, nullptr, 1, 0, 1) != hipSuccess);
  REQUIRE(hipDeviceGetP2PAtomicCapabilities(&capabilities, operations, 0, 0, 1) != hipSuccess);
  REQUIRE(hipDeviceGetP2PAtomicCapabilities(nullptr, nullptr, 0, 0, 1) != hipSuccess);
}

// @asserts: hipDeviceGetP2PAtomicCapabilities - rejects a same-device (src==dst) query and out-of-range or negative device ids with a non-success status
HIP_TEST_CASE(Contract_PeerQuery_HipDeviceGetP2PAtomicCapabilities_SelfOrInvalidDevice_IsRejected) {
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count <= 0) {
    HIP_SKIP_TEST(HipTest::SkipReason::kNoGpuDevice);
  }

  unsigned int capabilities = 0;
  const hipAtomicOperation operations[] = {kAtomicOperation};

  REQUIRE(hipDeviceGetP2PAtomicCapabilities(&capabilities, operations, 1, 0, 0) != hipSuccess);
  REQUIRE(hipDeviceGetP2PAtomicCapabilities(&capabilities, operations, 1, -1, 0) != hipSuccess);
  REQUIRE(hipDeviceGetP2PAtomicCapabilities(&capabilities, operations, 1, 0, -1) != hipSuccess);
  REQUIRE(hipDeviceGetP2PAtomicCapabilities(&capabilities, operations, 1, device_count, 0) !=
          hipSuccess);
  REQUIRE(hipDeviceGetP2PAtomicCapabilities(&capabilities, operations, 1, 0, device_count) !=
          hipSuccess);
}

// The device-id checks run before the per-operation validation, so an invalid
// operation is only reachable with two distinct valid devices.
// @asserts: hipDeviceGetP2PAtomicCapabilities - rejects an operation value outside the hipAtomicOperation enum with a non-success status
HIP_TEST_CASE(Contract_PeerQuery_HipDeviceGetP2PAtomicCapabilities_InvalidOperation_IsRejected) {
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count <= 0) {
    HIP_SKIP_TEST(HipTest::SkipReason::kNoGpuDevice);
  }
  if (device_count < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }

  unsigned int capabilities[2] = {0, 0};
  const hipAtomicOperation below_range[] = {static_cast<hipAtomicOperation>(-1)};
  const hipAtomicOperation above_range[] = {static_cast<hipAtomicOperation>(0x7fffffff)};
  // A valid leading entry must not mask an invalid one later in the array.
  const hipAtomicOperation mixed[] = {kAtomicOperation, static_cast<hipAtomicOperation>(0x7fffffff)};

  REQUIRE(hipDeviceGetP2PAtomicCapabilities(capabilities, below_range, 1, 0, 1) != hipSuccess);
  REQUIRE(hipDeviceGetP2PAtomicCapabilities(capabilities, above_range, 1, 0, 1) != hipSuccess);
  REQUIRE(hipDeviceGetP2PAtomicCapabilities(capabilities, mixed, 2, 0, 1) != hipSuccess);
}

// BACKEND-DIFF: hipExtGetLinkTypeAndHopCount is an AMD extension (link-type and
// hop-count query) with no NVIDIA equivalent, so this contract builds only on
// AMD. Parity would require a NVIDIA-side link-topology query API.
#if HT_AMD
// @asserts: hipExtGetLinkTypeAndHopCount - rejects a same-device (0,0) link-topology query with a non-success status
HIP_TEST_CASE(Contract_PeerQuery_HipExtGetLinkTypeAndHopCount_SameDevice_IsRejected) {
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  REQUIRE(device_count > 0);

  uint32_t link_type = 0;
  uint32_t hop_count = 0;

  REQUIRE(hipExtGetLinkTypeAndHopCount(0, 0, &link_type, &hop_count) != hipSuccess);
}

// @asserts: hipExtGetLinkTypeAndHopCount - rejects out-of-range and negative device ids with a non-success status
HIP_TEST_CASE(Contract_PeerQuery_HipExtGetLinkTypeAndHopCount_InvalidDevice_IsRejected) {
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));

  uint32_t link_type = 0;
  uint32_t hop_count = 0;
  REQUIRE(hipExtGetLinkTypeAndHopCount(device_count, 0, &link_type, &hop_count) != hipSuccess);
  REQUIRE(hipExtGetLinkTypeAndHopCount(0, device_count, &link_type, &hop_count) != hipSuccess);
  REQUIRE(hipExtGetLinkTypeAndHopCount(device_count, device_count + 1, &link_type, &hop_count) !=
          hipSuccess);
  REQUIRE(hipExtGetLinkTypeAndHopCount(-1, 0, &link_type, &hop_count) != hipSuccess);
  REQUIRE(hipExtGetLinkTypeAndHopCount(0, -1, &link_type, &hop_count) != hipSuccess);
  REQUIRE(hipExtGetLinkTypeAndHopCount(-1, -2, &link_type, &hop_count) != hipSuccess);
}

// @asserts: hipExtGetLinkTypeAndHopCount - rejects null link-type and/or hop-count output pointers with a non-success status
HIP_TEST_CASE(Contract_PeerQuery_HipExtGetLinkTypeAndHopCount_NullOutputs_AreRejected) {
  uint32_t link_type = 0;
  uint32_t hop_count = 0;

  REQUIRE(hipExtGetLinkTypeAndHopCount(0, 1, nullptr, &hop_count) != hipSuccess);
  REQUIRE(hipExtGetLinkTypeAndHopCount(0, 1, &link_type, nullptr) != hipSuccess);
  REQUIRE(hipExtGetLinkTypeAndHopCount(0, 1, nullptr, nullptr) != hipSuccess);
}
#endif  // HT_AMD
