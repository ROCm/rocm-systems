// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef TESTS_AMD_SMI_TEST_TEST_UTILS_H_
#define TESTS_AMD_SMI_TEST_TEST_UTILS_H_

#include "amd_smi/amdsmi.h"

const char* NameFromFWEnum(amdsmi_fw_block_t blk);
const char* NameFromEvtNotifType(amdsmi_evt_notification_type_t evt);

#endif  // TESTS_AMD_SMI_TEST_TEST_UTILS_H_
