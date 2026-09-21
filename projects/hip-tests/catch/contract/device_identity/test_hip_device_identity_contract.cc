/*
 * Copyright Advanced Micro Devices, Inc.
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

// @asserts: hipDeviceGetByPCIBusId - the PCI bus id from hipDeviceGetPCIBusId round-trips back to the same device ordinal
HIP_TEST_CASE(Contract_DeviceIdentity_HipDeviceGetByPCIBusId_Default_RoundTripsWithGetPCIBusId) {
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

// @asserts: hipDeviceGetByPCIBusId - empty and malformed PCI bus id strings are rejected with a non-success status
HIP_TEST_CASE(Contract_DeviceIdentity_HipDeviceGetByPCIBusId_InvalidString_IsRejected) {
  int device = -1;

  REQUIRE(hipDeviceGetByPCIBusId(&device, "") != hipSuccess);
  REQUIRE(hipDeviceGetByPCIBusId(&device, "0000:") != hipSuccess);
}

// @asserts: hipDeviceGetByPCIBusId - a null device out-pointer or null bus-id string is rejected with a non-success status
HIP_TEST_CASE(Contract_DeviceIdentity_HipDeviceGetByPCIBusId_NullArgs_AreRejected) {
  int current_device = 0;
  HIP_CHECK(hipGetDevice(&current_device));

  char pci_bus_id[kPciBusIdLength]{};
  HIP_CHECK(hipDeviceGetPCIBusId(pci_bus_id, kPciBusIdLength, current_device));

  int device = -1;
  REQUIRE(hipDeviceGetByPCIBusId(nullptr, pci_bus_id) != hipSuccess);
  REQUIRE(hipDeviceGetByPCIBusId(&device, nullptr) != hipSuccess);
}

// @asserts: hipChooseDevice - returns a device ordinal within [0, device_count) for a valid property struct
HIP_TEST_CASE(Contract_DeviceIdentity_HipChooseDevice_Default_ReturnsInRangeOrdinal) {
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

// @asserts: hipChooseDevice - a null device out-pointer or null properties pointer is rejected with a non-success status
HIP_TEST_CASE(Contract_DeviceIdentity_HipChooseDevice_NullArgs_AreRejected) {
  hipDeviceProp_t properties{};
  int device = -1;

  REQUIRE(hipChooseDevice(nullptr, &properties) != hipSuccess);
  REQUIRE(hipChooseDevice(&device, nullptr) != hipSuccess);
}

// @asserts: hipDeviceCanAccessPeer - a self-peer query returns a boolean-valued (0 or 1) accessibility result
HIP_TEST_CASE(Contract_DeviceIdentity_HipDeviceCanAccessPeer_Default_SelfQueryReturnsBoolean) {
  int current_device = 0;
  HIP_CHECK(hipGetDevice(&current_device));

  int can_access_peer = -1;
  HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, current_device, current_device));

  REQUIRE((can_access_peer == 0 || can_access_peer == 1));
}

// @asserts: hipDeviceCanAccessPeer - a null out-pointer or out-of-range device ordinal is rejected with a non-success status
HIP_TEST_CASE(Contract_DeviceIdentity_HipDeviceCanAccessPeer_InvalidArgs_AreRejected) {
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

// @asserts: hipDeviceGetLuid - a well-formed query returns hipSuccess or the documented hipErrorNotSupported
HIP_TEST_CASE(Contract_DeviceIdentity_HipDeviceGetLuid_Default_AcceptedOrUnsupported) {
  int current_device = 0;
  HIP_CHECK(hipGetDevice(&current_device));

  hipDevice_t device = 0;
  HIP_CHECK(hipDeviceGet(&device, current_device));

  char luid[8] = {};
  unsigned int node_mask = 0;
  const hipError_t status = hipDeviceGetLuid(luid, &node_mask, device);
  if (status == hipErrorNotSupported) {
    HIP_SKIP_TEST("hipDeviceGetLuid is not supported on this platform.");
  }
  HIP_CHECK(status);
}

// @asserts: hipDeviceGetLuid - a null luid or node-mask out-pointer is rejected with a non-success status
HIP_TEST_CASE(Contract_DeviceIdentity_HipDeviceGetLuid_NullArgs_AreRejected) {
  int current_device = 0;
  HIP_CHECK(hipGetDevice(&current_device));

  hipDevice_t device = 0;
  HIP_CHECK(hipDeviceGet(&device, current_device));

  char luid[8] = {};
  unsigned int node_mask = 0;

  REQUIRE(hipDeviceGetLuid(nullptr, &node_mask, device) != hipSuccess);
  (void)hipGetLastError();
  REQUIRE(hipDeviceGetLuid(luid, nullptr, device) != hipSuccess);
  (void)hipGetLastError();
}
