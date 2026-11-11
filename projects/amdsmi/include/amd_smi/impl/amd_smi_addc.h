// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "amd_smi/amdsmi.h"

// Bridge to addc-base, the only source of AFIDs and of the CPER JSON report. Each call takes one
// whole CPER record and parses it strictly: a record addc cannot parse is UNEXPECTED_DATA, a size
// problem is UNEXPECTED_SIZE, and a null or empty buffer is INVAL. Nothing is decoded partly.

// Replaces *afids with the AFID of every event addc decodes from the record.
auto cper_get_afids(const void* cper, size_t size, std::vector<int>* afids) -> amdsmi_status_t;

// Two-call contract: with json == nullptr, *json_size receives the byte count
// needed (trailing NUL included). A buffer that is too small returns
// AMDSMI_STATUS_INSUFFICIENT_SIZE and writes the needed count back.
auto cper_get_report_json(const void* cper, size_t size, char* json, uint32_t* json_size)
    -> amdsmi_status_t;
