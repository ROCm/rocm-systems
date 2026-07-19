/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
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
}  // namespace

// @asserts: hipDeviceGetP2PAttribute - rejects a same-device (src==dst) P2P attribute query with a non-success status
HIP_TEST_CASE(Contract_PeerQuery_GetP2PAttribute_SelfDevice_IsRejected) {
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  SkipIfP2PAttributeUnsupported(device);

  int value = -1;
  REQUIRE(hipDeviceGetP2PAttribute(&value, kP2PAttribute, device, device) != hipSuccess);
}

// @asserts: hipDeviceGetP2PAttribute - rejects null output, an unknown attribute enum, and out-of-range device ids with a non-success status
HIP_TEST_CASE(Contract_PeerQuery_GetP2PAttribute_InvalidArgs_AreRejected) {
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

// BACKEND-DIFF: hipExtGetLinkTypeAndHopCount is an AMD extension (link-type and
// hop-count query) with no NVIDIA equivalent, so this contract builds only on
// AMD. Parity would require a NVIDIA-side link-topology query API.
#if HT_AMD
// @asserts: hipExtGetLinkTypeAndHopCount - rejects a same-device (0,0) link-topology query with a non-success status
HIP_TEST_CASE(Contract_PeerQuery_LinkTypeAndHopCount_SameDevice_IsRejected) {
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  REQUIRE(device_count > 0);

  uint32_t link_type = 0;
  uint32_t hop_count = 0;

  REQUIRE(hipExtGetLinkTypeAndHopCount(0, 0, &link_type, &hop_count) != hipSuccess);
}

// @asserts: hipExtGetLinkTypeAndHopCount - rejects out-of-range and negative device ids with a non-success status
HIP_TEST_CASE(Contract_PeerQuery_LinkTypeAndHopCount_InvalidDevice_IsRejected) {
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
HIP_TEST_CASE(Contract_PeerQuery_LinkTypeAndHopCount_NullOutputs_AreRejected) {
  uint32_t link_type = 0;
  uint32_t hop_count = 0;

  REQUIRE(hipExtGetLinkTypeAndHopCount(0, 1, nullptr, &hop_count) != hipSuccess);
  REQUIRE(hipExtGetLinkTypeAndHopCount(0, 1, &link_type, nullptr) != hipSuccess);
  REQUIRE(hipExtGetLinkTypeAndHopCount(0, 1, nullptr, nullptr) != hipSuccess);
}
#endif  // HT_AMD
