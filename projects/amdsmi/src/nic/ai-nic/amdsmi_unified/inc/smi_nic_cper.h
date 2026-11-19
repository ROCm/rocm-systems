// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once

#include "amd_smi/impl/nic/amdsmi_unified/interface/smi_nic_interface.h"
#include "smi_nic.h"

// Internal: read CPER entries for one NIC from its pds_core debugfs ring.
// Called by smi_get_nic_cper_entries after the NIC is resolved from context.
smi_nic_status_t smi_nic_read_cper_entries(const SmiNic& nic, uint32_t severity_mask,
                                           char* cper_data, uint64_t* buf_size,
                                           amdsmi_cper_hdr_t** cper_hdrs, uint64_t* entry_count,
                                           uint64_t* cursor);
