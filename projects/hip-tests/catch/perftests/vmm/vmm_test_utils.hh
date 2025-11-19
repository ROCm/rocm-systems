/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

// Common utility functions for VMM tests
#pragma once

#include <hip_test_common.hh>
#include <cassert>

namespace vmm_utils {

// Memory size constants
constexpr size_t kB = 1024;
constexpr size_t kMB = (1024 * 1024);
constexpr size_t kGB = (1024 * 1024 * 1024);

// Check if VMM is supported on device
inline bool CheckVMMSupportedOnDevice(int deviceId) {
  int value = 0;
  hipDeviceAttribute_t attr = hipDeviceAttributeVirtualMemoryManagementSupported;
  HIP_CHECK(hipDeviceGetAttribute(&value, attr, deviceId));
  return static_cast<bool>(value);
}

// Get VMM allocation granularity
inline bool GetVMMGranularityOnDevice(int deviceId, size_t& granularity) {
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = deviceId;
  HIP_CHECK(hipMemGetAllocationGranularity(&granularity, &prop, 
                                           hipMemAllocationGranularityMinimum));
  return true;
}

// Calculate number of elements of type T in given byte size
template <typename T>
inline size_t GetSizeN(size_t total_size) {
  assert(total_size % sizeof(T) == 0);
  return (total_size / sizeof(T));
}

// Round up size to alignment
inline size_t RoundUpToAlignment(size_t size, size_t alignment) {
  return ((size + alignment - 1) / alignment) * alignment;
}

// Get device memory info
inline bool GetDeviceMemoryInfo(int deviceId, size_t& freeMemory, size_t& totalMemory) {
  HIP_CHECK(hipSetDevice(deviceId));
  HIP_CHECK(hipMemGetInfo(&freeMemory, &totalMemory));
  return true;
}

// Format size as human-readable string
inline std::string FormatSize(size_t bytes) {
  if (bytes >= kGB) return std::to_string(bytes / kGB) + " GB";
  if (bytes >= kMB) return std::to_string(bytes / kMB) + " MB";
  if (bytes >= kB) return std::to_string(bytes / kB) + " KB";
  return std::to_string(bytes) + " B";
}

// Create hipMemAllocationProp for device
inline hipMemAllocationProp CreateDeviceMemProp(int deviceId) {
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = deviceId;
  return prop;
}

// Create hipMemAccessDesc for device
inline hipMemAccessDesc CreateDeviceAccessDesc(int deviceId, 
                                                hipMemAccessFlags flags = hipMemAccessFlagsProtReadWrite) {
  hipMemAccessDesc accessDesc{};
  accessDesc.location.type = hipMemLocationTypeDevice;
  accessDesc.location.id = deviceId;
  accessDesc.flags = flags;
  return accessDesc;
}

}  // namespace vmm_utils

