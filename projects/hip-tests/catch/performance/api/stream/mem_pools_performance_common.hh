/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <hip_test_common.hh>
#include <performance_common.hh>

#if __linux__
  static const hipMemAllocationHandleType kHandleType = hipMemHandleTypePosixFileDescriptor;
#else
  static const hipMemAllocationHandleType kHandleType = hipMemHandleTypeWin32;
#endif

static int AreMemPoolsSupported(int device_id) {
  int mem_pools_supported = 0;
  HIP_CHECK(hipDeviceGetAttribute(&mem_pools_supported,
                                  hipDeviceAttributeMemoryPoolsSupported, 0));
  return mem_pools_supported;
}

static hipMemPoolProps CreateMemPoolProps(const int device_id, const hipMemAllocationHandleType handle_type) {
  hipMemPoolProps kPoolProps;
  memset(&kPoolProps, 0, sizeof(kPoolProps));
  kPoolProps.allocType = hipMemAllocationTypePinned;
  kPoolProps.handleTypes = handle_type;
  kPoolProps.location.type = hipMemLocationTypeDevice;
  kPoolProps.location.id = device_id;
  kPoolProps.win32SecurityAttributes = nullptr;
  return kPoolProps;
}

static std::string GetMemPoolAttrSectionName(const hipMemPoolAttr attribute) {
  switch (attribute) {
    case hipMemPoolReuseFollowEventDependencies:
      return "ReuseFollowEventDependencies";
    case hipMemPoolReuseAllowOpportunistic:
      return "ReuseAllowOpportunistic";
    case hipMemPoolReuseAllowInternalDependencies:
      return "ReuseAllowInternalDependencies";
    case hipMemPoolAttrReleaseThreshold:
      return "AttrReleaseThreshold";
    case hipMemPoolAttrReservedMemCurrent:
      return "AttrReservedMemCurrent";
    case hipMemPoolAttrReservedMemHigh:
      return "AttrReservedMemHigh";
    case hipMemPoolAttrUsedMemCurrent:
      return "AttrUsedMemCurrent";
    case hipMemPoolAttrUsedMemHigh:
      return "AttrUsedMemHigh";
    default:
      return "unknown attribute";
  }
}
