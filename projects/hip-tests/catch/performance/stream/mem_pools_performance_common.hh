/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
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
