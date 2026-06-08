// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/amd_smi_probe.hpp"
#include "core/sdma_feature.hpp"
#include "core/timemory.hpp"

#include <amd_smi/amdsmi.h>

// AMD-SMI >= 26.3 also exposes NIC APIs (gated independently by ROCPROFSYS_USE_AINIC).
#if AMDSMI_LIB_VERSION_MAJOR > 26 ||                                                     \
    (AMDSMI_LIB_VERSION_MAJOR == 26 && AMDSMI_LIB_VERSION_MINOR > 2)
#    if ROCPROFSYS_USE_AINIC > 0
#        define AINIC_SUPPORTED 1
#    endif
#endif

namespace rocprofsys
{
namespace amd_smi
{
bool
ensure_initialized();

metric_availability
probe_processor(amdsmi_processor_handle handle);

std::vector<device_probe_result>
probe_devices();

std::string
format_settings_description(const std::vector<device_probe_result>& devices);

std::vector<metric_entry>
format_avail_entries(const std::vector<device_probe_result>& devices);

void
config_settings(const std::shared_ptr<settings>& config);
}  // namespace amd_smi
}  // namespace rocprofsys
