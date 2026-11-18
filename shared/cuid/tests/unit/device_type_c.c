// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// amd_cuid.h compiled as C. A C caller can pass amdcuid_device_type_is_valid()
// any value its enum's integer type holds; a C++ caller cannot legally form
// one outside the enumeration's range.

#include "include/amd_cuid.h"

int amdcuid_test_device_type_is_valid_c(unsigned value) {
  return amdcuid_device_type_is_valid((amdcuid_device_type_t)value);
}
