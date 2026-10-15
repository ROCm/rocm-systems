// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMDSMI_UNIFIED_NIC_TESTING_H_
#define AMDSMI_UNIFIED_NIC_TESTING_H_

#include <memory>

// Included bare: the header opens its own extern "C".
#include "smi_nic_interface.h"
#include "smi_nic_system.h"

// Library-local test seam: wraps a prepared system in a context, so the smi_get_nic_* entry
// points can be driven on a host with no NIC. Not part of the C ABI.
smi_nic_status_t smi_nic_create_context_for_testing(smi_nic_ctx_t* ctx,
                                                    std::unique_ptr<SmiNicSystem> system);

#endif  // AMDSMI_UNIFIED_NIC_TESTING_H_
