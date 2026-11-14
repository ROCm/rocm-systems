/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <hip/hip_runtime.h>
#include <hip_test_common.hh>
#include <hip_test_kernels.hh>

namespace LocalityDomain {

inline constexpr char const kUnsupported[] =
    "Locality domains are not supported by this runtime or device";

/**
 * Queries the number of locality domains on @p device.
 *
 * Returns false when the runtime does not implement the attribute at all, which
 * is the expected state on backends where the feature has not landed yet. Tests
 * use this to skip rather than fail.
 */
inline bool GetDomainCount(int device, int* count) {
  int value = 0;
  if (hipDeviceGetAttribute(&value, hipDeviceAttributeLocalityDomainCount, device) != hipSuccess) {
    return false;
  }
  // A conforming implementation always reports at least one domain.
  if (value < 1) {
    return false;
  }
  *count = value;
  return true;
}

/** Same as GetDomainCount but for the per-domain compute unit count. */
inline bool GetDomainCuCount(int device, int* count) {
  int value = 0;
  if (hipDeviceGetAttribute(&value, hipDeviceAttributeLocalityDomainMultiprocessorCount, device) !=
      hipSuccess) {
    return false;
  }
  *count = value;
  return true;
}

/** Fetches the whole-device SM resource used as the input of every split. */
inline hipDevResource DeviceSmResource(int device_ordinal = 0) {
  hipDevice_t device;
  hipDevResource resource{};
  HIP_CHECK(hipDeviceGet(&device, device_ordinal));
  HIP_CHECK(hipDeviceGetDevResource(device, &resource, hipDevResourceTypeSm));
  return resource;
}

/** Builds split parameters targeting @p domain. */
inline hipDevSmResourceGroupParams LocalityParams(unsigned int domain, unsigned int sm_count = 0,
                                                  unsigned int extra_flags = 0) {
  hipDevSmResourceGroupParams params{};
  params.flags = hipDevSmResourceGroupLocalityDomainId | extra_flags;
  params.localityDomainId = domain;
  params.smCount = sm_count;
  return params;
}

/** Builds a memory location describing a locality domain of a device. */
inline hipMemLocation DomainLocation(unsigned char device, unsigned char domain) {
  hipMemLocation location{};
  location.type = hipMemLocationTypeDeviceLocalityDomain;
  location.localized.deviceId = device;
  location.localized.localityDomainId = domain;
  return location;
}

/** Builds memory pool properties for a locality-domain-backed pool. */
inline hipMemPoolProps DomainPoolProps(unsigned char device, unsigned char domain,
                                       hipMemAllocationType type = hipMemAllocationTypePinned) {
  hipMemPoolProps props{};
  props.allocType = type;
  props.location = DomainLocation(device, domain);
  return props;
}

}  // namespace LocalityDomain
