// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/pmc/sources.hpp"

#include "backends/amd_smi/backend.hpp"
#include "backends/amd_smi/device.hpp"
#include "backends/amd_smi/wrapper.hpp"
#include "backends/procfs/backend.hpp"
#include "core/agent.hpp"
#include "core/collection/source_interface.hpp"
#include "core/config.hpp"
#include "library/pmc/collectors/common/settings.hpp"
#include "library/pmc/collectors/gpu/device.hpp"
#include "library/pmc/device_providers/amd_smi/provider.hpp"
#include "library/pmc/device_providers/procfs/provider.hpp"
#include "library/pmc/pollers/cpu.hpp"
#include "library/pmc/pollers/gpu_smi.hpp"
#include "logger/debug.hpp"

#include <rocprofiler-sdk/version.h>

#if ROCPROFILER_VERSION >= 600
#    include "backends/rocprofiler_sdk/backend.hpp"
#    include "backends/rocprofiler_sdk/wrapper.hpp"
#    include "library/pmc/collectors/gpu_perf_counter/types.hpp"
#    include "library/pmc/device_providers/rocprofiler_sdk/provider.hpp"
#    include "library/pmc/pollers/gpu_perf_counter.hpp"
#endif

#if defined(ROCPROFSYS_BUILD_AINIC) && ROCPROFSYS_BUILD_AINIC == 1
#    include "library/pmc/collectors/nic/device.hpp"
#    include "library/pmc/pollers/nic_smi.hpp"
#endif

#if defined(ROCPROFSYS_BUILD_HIPFILE) && ROCPROFSYS_BUILD_HIPFILE == 1
#    include "backends/hipfile/backend.hpp"
#    include "backends/hipfile/wrapper.hpp"
#    include "library/pmc/collectors/hipfile/device.hpp"
#    include "library/pmc/device_providers/hipfile/provider.hpp"
#    include "library/pmc/pollers/hipfile.hpp"
#endif

#include <memory>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace rocprofsys::pmc
{
namespace
{
using settings_t = collectors::settings_policy;

using cpu_provider_factory_t =
    device_providers::procfs::provider_factory<backends::procfs::backend_factory>;
using cpu_provider_t = cpu_provider_factory_t::provider_t;

using amd_smi_provider_factory_t = device_providers::amd_smi::provider_factory<
    backends::amd_smi::backend_factory<backends::amd_smi::wrapper>>;
using amd_smi_provider_t = amd_smi_provider_factory_t::provider_t;
using amd_smi_device_t   = backends::amd_smi::device<amd_smi_provider_t::backend_t>;
using gpu_device_t       = collectors::gpu::device<amd_smi_device_t>;

#if defined(ROCPROFSYS_BUILD_AINIC) && ROCPROFSYS_BUILD_AINIC == 1
using nic_device_t = collectors::nic::device<amd_smi_device_t>;
#endif

#if defined(ROCPROFSYS_BUILD_HIPFILE) && ROCPROFSYS_BUILD_HIPFILE == 1
using hipfile_provider_factory_t = device_providers::hipfile::provider_factory<
    backends::hipfile::backend_factory<backends::hipfile::wrapper>>;
using hipfile_provider_t = hipfile_provider_factory_t::provider_t;
using hipfile_device_t   = collectors::hipfile::device<hipfile_provider_t::backend_t>;
#endif

using sources_t = std::vector<std::unique_ptr<collection::source_interface>>;

/** @return the started provider, or null (logged) when it failed to start */
template <typename Factory>
auto
create_provider(std::string_view what) -> decltype(Factory::create())
{
    try
    {
        return Factory::create();
    } catch(const std::runtime_error& error)
    {
        LOG_ERROR("Failed to start the {} provider: {}", what, error.what());
        return nullptr;
    }
}

/** Adds the GPU and NIC pollers. They share the provider; its destructor shuts AMD SMI
 * down after the last poller released it. */
void
add_amd_smi_sources(sources_t& result)
{
    const auto provider = create_provider<amd_smi_provider_factory_t>("AMD SMI");
    if(!provider)
    {
        return;
    }
    result.push_back(
        std::make_unique<pollers::gpu_smi<amd_smi_provider_t, gpu_device_t, settings_t>>(
            provider));
#if defined(ROCPROFSYS_BUILD_AINIC) && ROCPROFSYS_BUILD_AINIC == 1
    result.push_back(
        std::make_unique<pollers::nic_smi<amd_smi_provider_t, nic_device_t, settings_t>>(
            provider));
#endif
}

#if defined(ROCPROFSYS_BUILD_HIPFILE) && ROCPROFSYS_BUILD_HIPFILE == 1
/** Adds the hipFile poller. */
void
add_hipfile_source(sources_t& result)
{
    if(auto provider = create_provider<hipfile_provider_factory_t>("hipFile"))
    {
        result.push_back(
            std::make_unique<
                pollers::hipfile<hipfile_provider_t, hipfile_device_t, settings_t>>(
                std::move(provider)));
    }
}
#endif
}  // namespace

std::vector<std::unique_ptr<collection::source_interface>>
make_default_sources()
{
    sources_t result;
    if(auto provider = create_provider<cpu_provider_factory_t>("CPU"))
    {
        result.push_back(std::make_unique<pollers::cpu<cpu_provider_t, settings_t>>(
            std::move(provider)));
    }
    if(config::get_use_amd_smi())
    {
        add_amd_smi_sources(result);
    }
#if defined(ROCPROFSYS_BUILD_HIPFILE) && ROCPROFSYS_BUILD_HIPFILE == 1
    if(config::get_use_hipfile())
    {
        add_hipfile_source(result);
    }
#endif
    return result;
}

#if ROCPROFILER_VERSION >= 600
namespace
{
using gpu_perf_counter_provider_t = device_providers::rocprofiler_sdk::provider<
    backends::rocprofiler_sdk::backend_factory<::rocprofsys::rocprofiler_sdk::wrapper>>;

/** The configured counters; those without :device= are broadcast to every GPU agent. */
std::vector<collectors::gpu_perf_counter::counter_definition>
broadcast_counters(collectors::gpu_perf_counter::gpu_perf_counter_settings settings,
                   const std::vector<std::shared_ptr<agent>>&              gpu_agents)
{
    auto counters = std::move(settings.explicit_counters);
    for(const auto& name : settings.broadcast_names)
    {
        for(const auto& gpu_agent : gpu_agents)
        {
            counters.push_back(
                { .name = name, .device_index = gpu_agent->device_type_index });
        }
    }
    return counters;
}
}  // namespace

std::unique_ptr<collection::source_interface>
make_gpu_perf_counter_source(const std::vector<std::shared_ptr<agent>>& gpu_agents)
{
    try
    {
        auto counters = broadcast_counters(
            settings_t::get_gpu_perf_counter_enabled_metrics(), gpu_agents);
        if(counters.empty())
        {
            return nullptr;
        }
        auto provider = std::make_shared<gpu_perf_counter_provider_t>(
            gpu_agents,
            collectors::gpu_perf_counter::enabled_metrics{ std::move(counters) });
        LOG_DEBUG("Created the GPU perf counter source");
        return std::make_unique<
            pollers::gpu_perf_counter<gpu_perf_counter_provider_t, settings_t>>(
            std::move(provider));
    } catch(const std::runtime_error& error)
    {
        LOG_ERROR("Failed to create the GPU perf counter source: {}", error.what());
        return nullptr;
    }
}
#else
std::unique_ptr<collection::source_interface>
make_gpu_perf_counter_source(const std::vector<std::shared_ptr<agent>>& /*gpu_agents*/)
{
    return nullptr;
}
#endif
}  // namespace rocprofsys::pmc
