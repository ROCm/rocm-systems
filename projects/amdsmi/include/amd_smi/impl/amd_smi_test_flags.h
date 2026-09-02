// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMD_SMI_INCLUDE_AMD_SMI_TEST_FLAGS_H_
#define AMD_SMI_INCLUDE_AMD_SMI_TEST_FLAGS_H_

#include <cstdint>

// Reserved test-only init flag. Passed to amdsmi_init() / rsmi_init() to
// switch GPU device mutexes from blocking to non-blocking (trylock) mode,
// making AMDSMI_STATUS_BUSY / RSMI_STATUS_BUSY observable by tests.
//
// MUST NOT overlap with any public AMDSMI_INIT_* flag defined in amdsmi.h.
// Public amdsmi.h flags occupy bits [0:3]; internal rocm_smi flags use bits 58–59.
// This flag uses bit 59 (0x0800_0000_0000_0000).
inline constexpr uint64_t AMD_SMI_INIT_FLAG_RESRV_TEST1 = 0x0800000000000000ULL;

#endif  // AMD_SMI_INCLUDE_AMD_SMI_TEST_FLAGS_H_
