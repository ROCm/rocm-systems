// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "avail/records.hpp"

#include <concepts>
#include <exception>
#include <utility>
#include <vector>

namespace rocprofsys::avail
{
/// Supplies the GPU agents rocprofiler-sdk reports.
template <typename T>
concept agent_inventory_source = requires(T& source) {
    { source.gpu_agents() } -> std::same_as<std::vector<sdk_agent_entry>>;
};

/// Supplies the GPUs AMD SMI reports. Used only to add detail to agents.
template <typename T>
concept smi_inventory_source = requires(T& source) {
    { source.gpu_devices() } -> std::same_as<std::vector<smi_device_entry>>;
};

/// Correlates the two inventories by PCI BDF. Both sides format BDFs through
/// common::format_pci_bdf, so the canonical strings compare directly.
///
/// The agent list drives the result: an SMI device with no matching agent is
/// dropped, because rocprofiler-sdk decides what is profilable.
[[nodiscard]] std::vector<device_record>
merge_device_inventory(std::vector<sdk_agent_entry>         agents,
                       const std::vector<smi_device_entry>& smi_devices);

/// Builds the device inventory, degrading to whatever each backend could
/// supply. A backend that throws contributes a diagnostic, not a failure.
template <agent_inventory_source Agents, smi_inventory_source Smi>
[[nodiscard]] device_query_result
collect_devices(Agents& agents, Smi& smi)
{
    auto result = device_query_result{};

    auto sdk_agents = std::vector<sdk_agent_entry>{};
    try
    {
        sdk_agents = agents.gpu_agents();
    } catch(const std::exception& e)
    {
        result.diagnostics.push_back({ source_id::rocprofiler_sdk, e.what() });
    }

    // No agents means nothing to enrich, so AMD SMI is never initialized.
    auto smi_devices = std::vector<smi_device_entry>{};
    if(!sdk_agents.empty())
    {
        try
        {
            smi_devices = smi.gpu_devices();
        } catch(const std::exception& e)
        {
            result.diagnostics.push_back({ source_id::amd_smi, e.what() });
        }
    }

    result.devices = merge_device_inventory(std::move(sdk_agents), smi_devices);
    return result;
}

/// Device inventory from the live rocprofiler-sdk and AMD SMI backends.
[[nodiscard]] device_query_result
query_devices();
}  // namespace rocprofsys::avail
