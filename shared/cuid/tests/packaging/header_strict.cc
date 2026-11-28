// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// amd_cuid.h is compiled into consumers with their own warning flags. This
// file is built with strict ones and -Werror.

#include "amd_cuid.h"

int cuid_header_strict(amdcuid_device_type_t type) { return amdcuid_device_type_is_valid(type); }
