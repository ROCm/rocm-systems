// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// The same, compiled as C.

#include "amd_cuid.h"

int cuid_header_strict_c(amdcuid_device_type_t type) { return amdcuid_device_type_is_valid(type); }
