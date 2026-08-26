/*
 * Copyright Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstring>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

namespace {
int CurrentDevice() {
  int device = -1;
  HIP_CHECK(hipGetDevice(&device));
  return device;
}

hipDeviceProp_t CurrentDeviceProperties() {
  hipDeviceProp_t properties{};
  HIP_CHECK(hipGetDeviceProperties(&properties, CurrentDevice()));
  return properties;
}
}

// @asserts: hipGetDeviceProperties - succeeds in populating properties for the current device
HIP_TEST_CASE(Contract_Device_HipGetDeviceProperties_GetProperties_SucceedsForCurrentDevice) {
  hipDeviceProp_t properties{};

  HIP_CHECK(hipGetDeviceProperties(&properties, CurrentDevice()));
}

// @asserts: hipGetDeviceProperties - the device name string is non-empty
HIP_TEST_CASE(Contract_Device_HipGetDeviceProperties_Name_IsNonEmpty) {
  const auto properties = CurrentDeviceProperties();

  REQUIRE(std::strlen(properties.name) > 0);
}

// @asserts: hipGetDeviceProperties - reported total global memory is positive
HIP_TEST_CASE(Contract_Device_HipGetDeviceProperties_TotalGlobalMem_IsPositive) {
  const auto properties = CurrentDeviceProperties();

  REQUIRE(properties.totalGlobalMem > 0);
}

// @asserts: hipGetDeviceProperties - reported multiprocessor count is positive
HIP_TEST_CASE(Contract_Device_HipGetDeviceProperties_MultiProcessorCount_IsPositive) {
  const auto properties = CurrentDeviceProperties();

  REQUIRE(properties.multiProcessorCount > 0);
}

// @asserts: hipGetDeviceProperties - reported warp size is positive
HIP_TEST_CASE(Contract_Device_HipGetDeviceProperties_WarpSize_IsPositive) {
  const auto properties = CurrentDeviceProperties();

  REQUIRE(properties.warpSize > 0);
}

// @asserts: hipDeviceGetAttribute - hipDeviceAttributeWarpSize matches the warp size from hipGetDeviceProperties
HIP_TEST_CASE(Contract_Device_HipDeviceGetAttribute_WarpSize_MatchesProperties) {
  const auto properties = CurrentDeviceProperties();
  int attribute_warp_size = 0;

  HIP_CHECK(hipDeviceGetAttribute(&attribute_warp_size, hipDeviceAttributeWarpSize, CurrentDevice()));

  REQUIRE(attribute_warp_size == properties.warpSize);
}

// @asserts: hipGetDevice - the current device ordinal is in range [0, device_count)
HIP_TEST_CASE(Contract_Device_HipGetDevice_CurrentOrdinal_IsWithinDeviceCount) {
  int device_count = 0;
  const int current_device = CurrentDevice();

  HIP_CHECK(hipGetDeviceCount(&device_count));

  REQUIRE(device_count > 0);
  REQUIRE(current_device >= 0);
  REQUIRE(current_device < device_count);
}

// hipDeviceFlushGPUDirectRDMAWrites is a host-ordered visibility barrier on inbound
// GPUDirect RDMA writes. Proving that remote writes actually became visible needs an
// RDMA-capable NIC writing into device memory, which a device-only harness cannot
// arrange, so that behavioral coverage lives in
// unit/device/hipDeviceFlushGPUDirectRDMAWrites.cc. What is portable here is that a
// well-formed call is accepted or cleanly reports no flush path, and that a malformed
// enumerator is rejected as an invalid argument rather than as a missing capability.
namespace {
void RequireFlushAcceptedOrUnsupported(hipFlushGPUDirectRDMAWritesScope scope) {
  const hipError_t status =
      hipDeviceFlushGPUDirectRDMAWrites(hipFlushGPUDirectRDMAWritesTargetCurrentDevice, scope);

  if (status == hipErrorNotSupported) {
    (void)hipGetLastError();
    HIP_SKIP_TEST("device does not advertise a host GPUDirect RDMA flush path.");
    return;
  }

  REQUIRE(status == hipSuccess);
}
}  // namespace

// @asserts: hipDeviceFlushGPUDirectRDMAWrites - a flush to owner scope is accepted or reports unsupported
HIP_TEST_CASE(Contract_Device_HipDeviceFlushGPUDirectRDMAWrites_ToOwner_AcceptedOrUnsupported) {
  RequireFlushAcceptedOrUnsupported(hipFlushGPUDirectRDMAWritesToOwner);
}

// @asserts: hipDeviceFlushGPUDirectRDMAWrites - a flush to all-devices scope is accepted or reports unsupported
HIP_TEST_CASE(
    Contract_Device_HipDeviceFlushGPUDirectRDMAWrites_ToAllDevices_AcceptedOrUnsupported) {
  RequireFlushAcceptedOrUnsupported(hipFlushGPUDirectRDMAWritesToAllDevices);
}

// @asserts: hipDeviceFlushGPUDirectRDMAWrites - an out-of-range scope is rejected as an invalid argument
HIP_TEST_CASE(Contract_Device_HipDeviceFlushGPUDirectRDMAWrites_InvalidScope_IsRejected) {
  // Argument validation must precede the capability check, so a device with no flush path
  // still reports the bad enumerator rather than hipErrorNotSupported.
  const hipError_t status =
      hipDeviceFlushGPUDirectRDMAWrites(hipFlushGPUDirectRDMAWritesTargetCurrentDevice,
                                        static_cast<hipFlushGPUDirectRDMAWritesScope>(0x7fff));

  REQUIRE(status == hipErrorInvalidValue);
  (void)hipGetLastError();
}

// @asserts: hipDeviceFlushGPUDirectRDMAWrites - an out-of-range target is rejected as an invalid argument
HIP_TEST_CASE(Contract_Device_HipDeviceFlushGPUDirectRDMAWrites_InvalidTarget_IsRejected) {
  const hipError_t status =
      hipDeviceFlushGPUDirectRDMAWrites(static_cast<hipFlushGPUDirectRDMAWritesTarget>(0x7fff),
                                        hipFlushGPUDirectRDMAWritesToOwner);

  REQUIRE(status == hipErrorInvalidValue);
  (void)hipGetLastError();
}
