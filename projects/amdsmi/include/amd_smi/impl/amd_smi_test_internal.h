// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMD_SMI_INCLUDE_AMD_SMI_TEST_INTERNAL_H_
#define AMD_SMI_INCLUDE_AMD_SMI_TEST_INTERNAL_H_

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_test_flags.h"

// Internal test-only wrapper around rsmi_test_sleep. Acquires the device mutex
// for |seconds| seconds and returns an amdsmi_status_t so tests do not need to
// extern-declare the rsmi_status_t function directly.
amdsmi_status_t amdsmi_test_sleep(amdsmi_processor_handle processor_handle, uint32_t seconds);

#endif  // AMD_SMI_INCLUDE_AMD_SMI_TEST_INTERNAL_H_
