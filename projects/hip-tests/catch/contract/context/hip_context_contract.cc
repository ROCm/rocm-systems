/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <array>
#include <cstring>
#include <string.h>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

namespace {
// Returns the visible device count, skipping the test when no device is present
// so that driver-style queries are only exercised against a real ordinal.
int RequireDeviceCount() {
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count <= 0) {
    HIP_SKIP_TEST(HipTest::SkipReason::kNoGpuDevice);
  }
  return device_count;
}

// Resolves the driver-style handle for ordinal zero, which every device-backed
// context contract builds on.
hipDevice_t DeviceForOrdinalZero() {
  hipDevice_t device = 0;
  HIP_CHECK(hipDeviceGet(&device, 0));
  return device;
}
}  // namespace

HIP_TEST_CASE(Contract_Context_DeviceGet_ReturnsHandleForOrdinalZero) {
  RequireDeviceCount();

  hipDevice_t device = 0;
  HIP_CHECK(hipDeviceGet(&device, 0));

  std::array<char, 256> name{};
  HIP_CHECK(hipDeviceGetName(name.data(), static_cast<int>(name.size()), device));

  // The name must be a non-empty, NUL-terminated string; exact content is
  // backend-specific and therefore not asserted.
  const size_t name_length = ::strnlen(name.data(), name.size());
  REQUIRE(name_length > 0);
  REQUIRE(name_length < name.size());
}

HIP_TEST_CASE(Contract_Context_DeviceComputeCapability_IsPositive) {
  RequireDeviceCount();

  const hipDevice_t device = DeviceForOrdinalZero();

  int major = -1;
  int minor = -1;
  HIP_CHECK(hipDeviceComputeCapability(&major, &minor, device));

  REQUIRE(major >= 0);
  REQUIRE(minor >= 0);
  // A usable device reports a positive major compute capability.
  REQUIRE(major > 0);
}

HIP_TEST_CASE(Contract_Context_DeviceTotalMem_MatchesProperties) {
  RequireDeviceCount();

  const hipDevice_t device = DeviceForOrdinalZero();

  size_t total_bytes = 0;
  HIP_CHECK(hipDeviceTotalMem(&total_bytes, device));
  REQUIRE(total_bytes > 0);

  hipDeviceProp_t properties{};
  HIP_CHECK(hipGetDeviceProperties(&properties, 0));

  REQUIRE(total_bytes == properties.totalGlobalMem);
}

HIP_TEST_CASE(Contract_Context_DeviceGetUuidAndPciBusId_Succeed) {
  RequireDeviceCount();

  const hipDevice_t device = DeviceForOrdinalZero();

  hipUUID uuid{};
  HIP_CHECK(hipDeviceGetUuid(&uuid, device));

  // hipUUID is a fixed 16-byte structure; a provisioned device exposes at least
  // one non-zero byte, but content/format is backend-specific.
  bool has_non_zero_byte = false;
  for (const char byte : uuid.bytes) {
    if (byte != 0) {
      has_non_zero_byte = true;
      break;
    }
  }
  REQUIRE(has_non_zero_byte);

  std::array<char, 64> pci_bus_id{};
  HIP_CHECK(hipDeviceGetPCIBusId(pci_bus_id.data(),
                                 static_cast<int>(pci_bus_id.size()), 0));

  // The PCI bus id must be a non-empty, NUL-terminated string; the exact format
  // is backend-specific and therefore not asserted.
  const size_t pci_length = ::strnlen(pci_bus_id.data(), pci_bus_id.size());
  REQUIRE(pci_length > 0);
  REQUIRE(pci_length < pci_bus_id.size());
}

HIP_TEST_CASE(Contract_Context_PrimaryCtxRetainRelease_RoundTrips) {
  RequireDeviceCount();

  const hipDevice_t device = DeviceForOrdinalZero();

  hipCtx_t context = nullptr;
  HIP_CHECK(hipDevicePrimaryCtxRetain(&context, device));
  REQUIRE(context != nullptr);

  unsigned int flags = 0;
  int active = -1;
  HIP_CHECK(hipDevicePrimaryCtxGetState(device, &flags, &active));

  // Retain alone does not portably mark the primary context active on AMD HIP;
  // the state query must still return a well-defined boolean active flag.
  REQUIRE((active == 0 || active == 1));

  // Making the device current portably activates the primary context across
  // both AMD HIP and CUDA backends.
  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipDevicePrimaryCtxGetState(device, &flags, &active));
  REQUIRE(active == 1);

  HIP_CHECK(hipDevicePrimaryCtxRelease(device));
}

HIP_TEST_CASE(Contract_Context_CtxGetCurrentAndDevice_AreConsistent) {
  RequireDeviceCount();

  hipCtx_t current = nullptr;
  HIP_CHECK(hipCtxGetCurrent(&current));

  if (current == nullptr) {
    // Some runtime paths report no current context without an active driver-style
    // context; the device-match assertion only applies when one exists.
    return;
  }

  hipDevice_t context_device = 0;
  HIP_CHECK(hipCtxGetDevice(&context_device));

  int runtime_device = -1;
  HIP_CHECK(hipGetDevice(&runtime_device));

  REQUIRE(context_device == runtime_device);
}
