// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "avail/devices.hpp"
#include "avail/records.hpp"

#include <exception>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(ROCPROFSYS_AVAIL_HAS_SDK)
#    include "backends/amd_smi/backend.hpp"
#    include "backends/amd_smi/gpu_types.hpp"
#    include "backends/amd_smi/wrapper.hpp"
#    include "backends/rocprofiler_sdk/wrapper.hpp"
#endif

namespace rocprofsys::avail
{
namespace
{

diagnostic
make_diagnostic(capability_kind capability, source_id source, std::string message)
{
    return diagnostic{
        .capability = capability,
        .source     = source,
        .message    = std::move(message),
    };
}

#if defined(ROCPROFSYS_AVAIL_HAS_SDK)
struct smi_session
{
    using backend_t = backends::amd_smi::backend<backends::amd_smi::wrapper>;

    smi_session() { m_backend.initialize(); }
    ~smi_session() { m_backend.shutdown(); }

    smi_session(const smi_session&)            = delete;
    smi_session& operator=(const smi_session&) = delete;
    smi_session(smi_session&&)                 = delete;
    smi_session& operator=(smi_session&&)      = delete;

    [[nodiscard]] backend_t&       backend() { return m_backend; }
    [[nodiscard]] const backend_t& backend() const { return m_backend; }

private:
    backend_t m_backend;
};

using asic_by_bdf = std::unordered_map<std::string, backends::amd_smi::gpu::asic_info>;

[[nodiscard]] asic_by_bdf
smi_asic_by_bdf(smi_session& session)
{
    asic_by_bdf by_bdf;
    for(const auto handle : session.backend().enumerate_gpu_handles())
    {
        smi_session::backend_t::asic_info_t raw{};
        session.backend().get_gpu_asic_info(handle, &raw);
        by_bdf.emplace(session.backend().get_device_bdf(handle),
                       backends::amd_smi::gpu::asic_info{
                           .product_name = c_string_or_empty(raw.market_name),
                           .vendor_name  = c_string_or_empty(raw.vendor_name),
                       });
    }
    return by_bdf;
}

void
apply_asic_info(gpu_agent_info& agent, const backends::amd_smi::gpu::asic_info& asic)
{
    if(!agent.device.product && !asic.product_name.empty())
    {
        agent.device.product = asic.product_name;
        if(agent.device.name.empty())
        {
            agent.device.name = asic.product_name;
        }
    }
    if(!agent.device.vendor && !asic.vendor_name.empty())
    {
        agent.device.vendor = asic.vendor_name;
    }
}

void
enrich_agents_from_smi(std::vector<gpu_agent_info>& agents)
{
    smi_session session;
    const auto  by_bdf = smi_asic_by_bdf(session);
    for(auto& agent : agents)
    {
        if(!agent.device.pci_bdf)
        {
            continue;
        }
        const auto found = by_bdf.find(*agent.device.pci_bdf);
        if(found == by_bdf.end())
        {
            continue;
        }
        apply_asic_info(agent, found->second);
    }
}

[[nodiscard]] gpu_agents_listing_result
query_gpu_agents_from_sdk()
{
    auto listed = inventory::gpu_agents<rocprofiler_sdk::wrapper>();
    if(listed.issue.has_value() || listed.agents.empty())
    {
        return listed;
    }
    try
    {
        enrich_agents_from_smi(listed.agents);
    } catch(const std::exception& err)
    {
        listed.issue =
            make_diagnostic(capability_kind::gpu_devices, source_id::amd_smi, err.what());
    }
    return listed;
}
#endif

}  // namespace

gpu_agents_listing_result
query_gpu_agent_infos()
{
#if defined(ROCPROFSYS_AVAIL_HAS_SDK)
    try
    {
        return query_gpu_agents_from_sdk();
    } catch(const std::exception& err)
    {
        return gpu_agents_listing_result{
            .agents = {},
            .issue  = make_diagnostic(capability_kind::gpu_devices,
                                      source_id::rocprofiler_sdk, err.what()),
        };
    }
#else
    return gpu_agents_listing_result{
        .agents = {},
        .issue = make_diagnostic(capability_kind::gpu_devices, source_id::rocprofiler_sdk,
                                 std::string{ k_sdk_unavailable_message }),
    };
#endif
}

devices_listing_result
query_gpu_devices()
{
    auto                   listed = query_gpu_agent_infos();
    devices_listing_result result;
    result.issue = std::move(listed.issue);
    result.records.reserve(listed.agents.size());
    for(auto& agent : listed.agents)
    {
        result.records.push_back(std::move(agent.device));
    }
    return result;
}

}  // namespace rocprofsys::avail
