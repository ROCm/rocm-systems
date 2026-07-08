/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

namespace {
constexpr int kPciBusIdLength = 32;

bool GetDeviceByPciBusIdOrSkip(int* device, const char* pci_bus_id) {
  const hipError_t status = hipDeviceGetByPCIBusId(device, pci_bus_id);
  if (status == hipErrorNotSupported) {
    return false;
  }
  HIP_CHECK(status);
  return true;
}
}  // namespace

HIP_TEST_CASE(Contract_DeviceIdentity_GetByPCIBusId_RoundTripsWithGetPCIBusId) {
  int current_device = 0;
  HIP_CHECK(hipGetDevice(&current_device));

  char pci_bus_id[kPciBusIdLength]{};
  HIP_CHECK(hipDeviceGetPCIBusId(pci_bus_id, kPciBusIdLength, current_device));

  int resolved_device = -1;
  if (!GetDeviceByPciBusIdOrSkip(&resolved_device, pci_bus_id)) {
    HIP_SKIP_TEST("hipDeviceGetByPCIBusId is not supported by this runtime path.");
  }

  REQUIRE(resolved_device == current_device);
}

HIP_TEST_CASE(Contract_DeviceIdentity_GetByPCIBusId_InvalidString_IsRejected) {
  int device = -1;

  REQUIRE(hipDeviceGetByPCIBusId(&device, "") != hipSuccess);
  REQUIRE(hipDeviceGetByPCIBusId(&device, "0000:") != hipSuccess);
}

HIP_TEST_CASE(Contract_DeviceIdentity_GetByPCIBusId_NullArgs_AreRejected) {
  int current_device = 0;
  HIP_CHECK(hipGetDevice(&current_device));

  char pci_bus_id[kPciBusIdLength]{};
  HIP_CHECK(hipDeviceGetPCIBusId(pci_bus_id, kPciBusIdLength, current_device));

  int device = -1;
  REQUIRE(hipDeviceGetByPCIBusId(nullptr, pci_bus_id) != hipSuccess);
  REQUIRE(hipDeviceGetByPCIBusId(&device, nullptr) != hipSuccess);
}

HIP_TEST_CASE(Contract_DeviceIdentity_ChooseDevice_ReturnsInRangeOrdinal) {
  int current_device = 0;
  int device_count = 0;
  HIP_CHECK(hipGetDevice(&current_device));
  HIP_CHECK(hipGetDeviceCount(&device_count));

  hipDeviceProp_t properties{};
  HIP_CHECK(hipGetDeviceProperties(&properties, current_device));

  int chosen_device = -1;
  HIP_CHECK(hipChooseDevice(&chosen_device, &properties));

  REQUIRE(chosen_device >= 0);
  REQUIRE(chosen_device < device_count);
}

HIP_TEST_CASE(Contract_DeviceIdentity_ChooseDevice_NullArgs_AreRejected) {
  hipDeviceProp_t properties{};
  int device = -1;

  REQUIRE(hipChooseDevice(nullptr, &properties) != hipSuccess);
  REQUIRE(hipChooseDevice(&device, nullptr) != hipSuccess);
}

HIP_TEST_CASE(Contract_DeviceIdentity_CanAccessPeer_SelfQueryReturnsBoolean) {
  int current_device = 0;
  HIP_CHECK(hipGetDevice(&current_device));

  int can_access_peer = -1;
  HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, current_device, current_device));

  REQUIRE((can_access_peer == 0 || can_access_peer == 1));
}

HIP_TEST_CASE(Contract_DeviceIdentity_CanAccessPeer_InvalidArgs_AreRejected) {
  int current_device = 0;
  int device_count = 0;
  HIP_CHECK(hipGetDevice(&current_device));
  HIP_CHECK(hipGetDeviceCount(&device_count));

  int can_access_peer = -1;
  REQUIRE(hipDeviceCanAccessPeer(nullptr, current_device, current_device) != hipSuccess);
  REQUIRE(hipDeviceCanAccessPeer(&can_access_peer, -1, current_device) != hipSuccess);
  REQUIRE(hipDeviceCanAccessPeer(&can_access_peer, device_count, current_device) != hipSuccess);
  REQUIRE(hipDeviceCanAccessPeer(&can_access_peer, current_device, -1) != hipSuccess);
  REQUIRE(hipDeviceCanAccessPeer(&can_access_peer, current_device, device_count) != hipSuccess);
}
