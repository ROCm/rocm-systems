// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "avail/devices.hpp"

#include "backends/amd_smi/backend.hpp"
#include "backends/amd_smi/device.hpp"
#include "backends/amd_smi/wrapper.hpp"
#include "common/pci_bdf.hpp"
#include "core/agent.hpp"
#include "core/agent_manager.hpp"
#include "core/gpu.hpp"

#include <algorithm>
#include <cctype>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocprofsys::avail
{
namespace
{
using smi_backend_t = backends::amd_smi::backend<backends::amd_smi::wrapper>;
using smi_device_t  = backends::amd_smi::device<smi_backend_t>;

[[nodiscard]] std::string
normalize_bdf(std::string bdf)
{
    std::transform(bdf.begin(), bdf.end(), bdf.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return bdf;
}

/// Reads the agents rocprofiler-sdk already discovered. gpu::device_count()
/// performs the one-time SDK agent query and populates the agent manager.
struct sdk_agents
{
    [[nodiscard]] std::vector<sdk_agent_entry> gpu_agents() const
    {
        auto entries = std::vector<sdk_agent_entry>{};
        if(gpu::device_count() == 0) return entries;

        const auto agents =
            get_agent_manager_instance().get_agents_by_type(agent_type::gpu);
        entries.reserve(agents.size());

        for(const auto& agent : agents)
        {
            if(!agent) continue;
            entries.push_back(sdk_agent_entry{
                agent->handle, agent->device_type_index, agent->name,
                agent->product_name, agent->vendor_name,
                common::format_pci_bdf_from_location_id(agent->domain,
                                                        agent->location_id),
                agent->hip_visible });
        }
        return entries;
    }
};

/// Enumerates AMD SMI GPUs for their identification fields only. The session is
/// created here so nothing initializes AMD SMI unless devices are enriched.
struct smi_inventory
{
    [[nodiscard]] std::vector<smi_device_entry> gpu_devices() const
    {
        auto entries = std::vector<smi_device_entry>{};

        auto session =
            backends::amd_smi::backend_factory<backends::amd_smi::wrapper>::
                create_backend();
        session->initialize();

        for(auto handle : session->enumerate_gpu_handles())
        {
            auto device = smi_device_t{ session, handle };
            auto asic   = device.get_gpu_asic_info();
            entries.push_back(smi_device_entry{ device.get_bdf(), asic.product_name,
                                                asic.vendor_name });
        }

        session->shutdown();
        return entries;
    }
};
}  // namespace

std::vector<device_record>
merge_device_inventory(std::vector<sdk_agent_entry>         agents,
                       const std::vector<smi_device_entry>& smi_devices)
{
    auto by_bdf = std::unordered_map<std::string, const smi_device_entry*>{};
    by_bdf.reserve(smi_devices.size());
    for(const auto& smi : smi_devices)
    {
        if(smi.pci_bdf.empty()) continue;
        by_bdf.emplace(normalize_bdf(smi.pci_bdf), &smi);
    }

    auto records = std::vector<device_record>{};
    records.reserve(agents.size());

    for(auto& agent : agents)
    {
        auto record            = device_record{};
        record.agent_handle    = agent.handle;
        record.index           = agent.index;
        record.name            = std::move(agent.name);
        record.product_name    = std::move(agent.product_name);
        record.vendor_name     = std::move(agent.vendor_name);
        record.pci_bdf         = std::move(agent.pci_bdf);
        record.runtime_visible = agent.runtime_visible;

        if(!record.pci_bdf.empty())
        {
            const auto match = by_bdf.find(normalize_bdf(record.pci_bdf));
            if(match != by_bdf.end())
            {
                const auto* smi = match->second;
                if(!smi->market_name.empty()) record.smi_market_name = smi->market_name;
                if(!smi->vendor_name.empty()) record.smi_vendor_name = smi->vendor_name;
            }
        }

        records.push_back(std::move(record));
    }

    std::sort(records.begin(), records.end(),
              [](const device_record& lhs, const device_record& rhs) {
                  if(lhs.index != rhs.index) return lhs.index < rhs.index;
                  return lhs.agent_handle < rhs.agent_handle;
              });

    return records;
}

device_query_result
query_devices()
{
    auto agents = sdk_agents{};
    auto smi    = smi_inventory{};
    return collect_devices(agents, smi);
}
}  // namespace rocprofsys::avail
