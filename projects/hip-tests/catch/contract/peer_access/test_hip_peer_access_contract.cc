/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

namespace {
void DeviceCount(int* count) {
  HIP_CHECK(hipGetDeviceCount(count));
  if (*count <= 0) {
    HIP_SKIP_TEST(HipTest::SkipReason::kNoGpuDevice);
  }
}
}  // namespace

HIP_TEST_CASE(Contract_PeerAccess_Enable_InvalidPeerId_IsRejected) {
  int device_count = 0;
  DeviceCount(&device_count);

  REQUIRE(hipDeviceEnablePeerAccess(-1, 0) != hipSuccess);
  REQUIRE(hipDeviceEnablePeerAccess(device_count, 0) != hipSuccess);
}

HIP_TEST_CASE(Contract_PeerAccess_Enable_InvalidFlag_IsRejected) {
  int current_device = 0;
  HIP_CHECK(hipGetDevice(&current_device));

  REQUIRE(hipDeviceEnablePeerAccess(current_device, ~0u) != hipSuccess);
}

HIP_TEST_CASE(Contract_PeerAccess_Disable_InvalidPeerId_IsRejected) {
  int device_count = 0;
  DeviceCount(&device_count);

  REQUIRE(hipDeviceDisablePeerAccess(-1) != hipSuccess);
  REQUIRE(hipDeviceDisablePeerAccess(device_count) != hipSuccess);
}

HIP_TEST_CASE(Contract_PeerAccess_Disable_NotEnabled_IsRejected) {
  int current_device = 0;
  HIP_CHECK(hipGetDevice(&current_device));

  REQUIRE(hipDeviceDisablePeerAccess(current_device) != hipSuccess);
}

HIP_TEST_CASE(Contract_PeerAccess_EnableSelf_IsRejected) {
  int current_device = 0;
  HIP_CHECK(hipGetDevice(&current_device));

  REQUIRE(hipDeviceEnablePeerAccess(current_device, 0) != hipSuccess);
}

HIP_TEST_CASE(Contract_PeerAccess_EnableTwice_ThenDisable_RoundTripsWhenAvailable) {
  int device_count = 0;
  DeviceCount(&device_count);
  if (device_count < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }

  int current_device = 0;
  HIP_CHECK(hipGetDevice(&current_device));
  const int peer_device = (current_device == 0) ? 1 : 0;

  int can_access_peer = 0;
  HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, current_device, peer_device));
  if (can_access_peer == 0) {
    HIP_SKIP_TEST(HipTest::SkipReason::kPeerAccessUnavailable);
  }

  HIP_CHECK(hipDeviceEnablePeerAccess(peer_device, 0));
  const hipError_t second_enable = hipDeviceEnablePeerAccess(peer_device, 0);
  HIP_CHECK(hipDeviceDisablePeerAccess(peer_device));
  const hipError_t second_disable = hipDeviceDisablePeerAccess(peer_device);

  REQUIRE(second_enable != hipSuccess);
  REQUIRE(second_disable != hipSuccess);
}
