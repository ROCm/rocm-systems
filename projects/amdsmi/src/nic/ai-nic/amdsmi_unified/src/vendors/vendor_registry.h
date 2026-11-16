// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMDSMI_UNIFIED_VENDORS_REGISTRY_H_
#define AMDSMI_UNIFIED_VENDORS_REGISTRY_H_

#include <memory>
#include <vector>

#include "smi_nic_subsystem.h"

/**
 * Returns one plugin instance per supported NIC vendor. A new partner is added
 * by dropping a src/vendors/<name>/ directory and appending one line here.
 */
std::vector<std::unique_ptr<SmiNicSubsystem>> make_default_vendor_plugins();

#endif  // AMDSMI_UNIFIED_VENDORS_REGISTRY_H_
