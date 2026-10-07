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

// hipDeviceGetExecAffinitySupport reports whether a device supports an execution affinity
// type. Which types are supported is a backend and device property, so what is portable
// is the shape of the answer, not its value: a well-formed query writes a strict boolean
// or reports no query path at all, and a malformed argument is rejected. The AMD-specific
// values are pinned by the HT_AMD cases below and by
// unit/device/hipDeviceGetExecAffinitySupport.cc.
namespace {
hipDevice_t CurrentDeviceHandle() {
  hipDevice_t device = 0;
  HIP_CHECK(hipDeviceGet(&device, CurrentDevice()));
  return device;
}

// Returns the reported support flag, having checked it is a strict boolean.
int RequireSupportFlagIsBoolean(hipExecAffinityType type) {
  // -1 is not a value the API may leave behind: the flag must be written on success.
  int supported = -1;
  const hipError_t status =
      hipDeviceGetExecAffinitySupport(&supported, type, CurrentDeviceHandle());

  // BACKEND-DIFF: cuDeviceGetExecAffinitySupport arrived in CUDA 11.4, so the NVIDIA
  // backend reports hipErrorNotSupported for every query when built against an older
  // toolkit. Parity needs a newer CUDA, not a runtime change.
  if (status == hipErrorNotSupported) {
    (void)hipGetLastError();
    HIP_SKIP_TEST("backend provides no execution-affinity support query.");
  }
  HIP_CHECK(status);

  REQUIRE((supported == 0 || supported == 1));
  return supported;
}

// Returns the status of a query that must be rejected.
hipError_t RequireRejectedStatus(int* out_support, hipExecAffinityType type, hipDevice_t device) {
  const hipError_t status = hipDeviceGetExecAffinitySupport(out_support, type, device);
  (void)hipGetLastError();

  if (status == hipErrorNotSupported) {
    HIP_SKIP_TEST("backend provides no execution-affinity support query.");
  }

  REQUIRE(status != hipSuccess);
  return status;
}
}  // namespace

// @asserts: hipDeviceGetExecAffinitySupport - a CU-count query reports a boolean flag or unsupported
HIP_TEST_CASE(Contract_Device_HipDeviceGetExecAffinitySupport_CuCountType_ReportsBooleanFlag) {
  (void)RequireSupportFlagIsBoolean(hipExecAffinityTypeCUCount);
}

// @asserts: hipDeviceGetExecAffinitySupport - a CU-granularity query reports a boolean flag or unsupported
HIP_TEST_CASE(Contract_Device_HipDeviceGetExecAffinitySupport_CuGranularityType_ReportsBooleanFlag) {
  (void)RequireSupportFlagIsBoolean(hipExtExecAffinityTypeGranularityCU);
}

// @asserts: hipDeviceGetExecAffinitySupport - a WGP-granularity query reports a boolean flag or unsupported
HIP_TEST_CASE(
    Contract_Device_HipDeviceGetExecAffinitySupport_WgpGranularityType_ReportsBooleanFlag) {
  (void)RequireSupportFlagIsBoolean(hipExtExecAffinityTypeGranularityWGP);
}

// @asserts: hipDeviceGetExecAffinitySupport - repeating a query reports the same flag
HIP_TEST_CASE(Contract_Device_HipDeviceGetExecAffinitySupport_RepeatedQuery_IsStable) {
  const int first = RequireSupportFlagIsBoolean(hipExtExecAffinityTypeGranularityCU);
  const int second = RequireSupportFlagIsBoolean(hipExtExecAffinityTypeGranularityCU);

  REQUIRE(first == second);
}

#if HT_AMD
// BACKEND-DIFF: the next two cases pin AMD-only guarantees. CU masking exists on every
// AMD GPU, but an NVIDIA device may report SM-count affinity as unavailable; and the
// CU/WGP granularity types are AMD extensions that NVIDIA answers with a constant 0 for
// both, so exactly-one does not hold there. Parity is not expected.

// @asserts: hipDeviceGetExecAffinitySupport - CU-count affinity is supported on every AMD device
HIP_TEST_CASE(Contract_Device_HipDeviceGetExecAffinitySupport_CuCountType_IsAlwaysSupported) {
  REQUIRE(RequireSupportFlagIsBoolean(hipExecAffinityTypeCUCount) == 1);
}

// @asserts: hipDeviceGetExecAffinitySupport - exactly one masking granularity is supported
HIP_TEST_CASE(Contract_Device_HipDeviceGetExecAffinitySupport_MaskGranularity_IsExactlyOneType) {
  const int cu_granularity = RequireSupportFlagIsBoolean(hipExtExecAffinityTypeGranularityCU);
  const int wgp_granularity = RequireSupportFlagIsBoolean(hipExtExecAffinityTypeGranularityWGP);

  // A device masks either per-CU or per-WGP, never both and never neither.
  REQUIRE((cu_granularity + wgp_granularity) == 1);
}
#endif

// @asserts: hipDeviceGetExecAffinitySupport - a null out-pointer is rejected as an invalid argument
HIP_TEST_CASE(Contract_Device_HipDeviceGetExecAffinitySupport_NullOutPointer_IsRejected) {
  const hipError_t status =
      RequireRejectedStatus(nullptr, hipExecAffinityTypeCUCount, CurrentDeviceHandle());

  REQUIRE(status == hipErrorInvalidValue);
}

// @asserts: hipDeviceGetExecAffinitySupport - the type sentinel is rejected as an invalid argument
HIP_TEST_CASE(Contract_Device_HipDeviceGetExecAffinitySupport_SentinelType_IsRejected) {
  int supported = 0;
  const hipError_t status =
      RequireRejectedStatus(&supported, hipExecAffinityTypeMax, CurrentDeviceHandle());

  REQUIRE(status == hipErrorInvalidValue);
}

// @asserts: hipDeviceGetExecAffinitySupport - an out-of-range type is rejected as an invalid argument
HIP_TEST_CASE(Contract_Device_HipDeviceGetExecAffinitySupport_OutOfRangeType_IsRejected) {
  int supported = 0;
  const hipError_t status = RequireRejectedStatus(
      &supported, static_cast<hipExecAffinityType>(0x7fff), CurrentDeviceHandle());

  REQUIRE(status == hipErrorInvalidValue);
}

// @asserts: hipDeviceGetExecAffinitySupport - a negative device ordinal is rejected as an invalid device
HIP_TEST_CASE(Contract_Device_HipDeviceGetExecAffinitySupport_NegativeDevice_IsRejected) {
  int supported = 0;
  const hipError_t status =
      RequireRejectedStatus(&supported, hipExecAffinityTypeCUCount, static_cast<hipDevice_t>(-1));

  REQUIRE(status == hipErrorInvalidDevice);
}

// @asserts: hipDeviceGetExecAffinitySupport - an out-of-range device is rejected as an invalid device
HIP_TEST_CASE(Contract_Device_HipDeviceGetExecAffinitySupport_OutOfRangeDevice_IsRejected) {
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  REQUIRE(device_count > 0);

  int supported = 0;
  const hipError_t status = RequireRejectedStatus(&supported, hipExecAffinityTypeCUCount,
                                                  static_cast<hipDevice_t>(device_count));

  REQUIRE(status == hipErrorInvalidDevice);
}

// @asserts: hipDeviceGetExecAffinitySupport - a null out-pointer outranks a bad device ordinal
HIP_TEST_CASE(Contract_Device_HipDeviceGetExecAffinitySupport_NullOutPointer_OutranksBadDevice) {
  const hipError_t status =
      RequireRejectedStatus(nullptr, hipExecAffinityTypeCUCount, static_cast<hipDevice_t>(-1));

  REQUIRE(status == hipErrorInvalidValue);
}
