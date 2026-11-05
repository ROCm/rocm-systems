// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Must be first: defines AINIC_SUPPORTED based on the installed AMD SMI version
// and ROCPROFSYS_USE_AINIC before any header gates code on that macro.
#include "backends/amd_smi/ainic_feature.hpp"  // IWYU pragma: keep

#include "library/pmc/sources.hpp"

#include "backends/rocprofiler_sdk/wrapper.hpp"
#include "core/agent.hpp"
#include "core/collection/source_interface.hpp"
#include "core/config.hpp"
#include "library/collection/sources/gpu_smi_source.hpp"
#include "library/pmc/collectors/common/settings.hpp"
#include "library/pmc/collectors/gpu_perf_counter/types.hpp"
#include "library/process_sampler.hpp"
#include "logger/debug.hpp"

#include <rocprofiler-sdk/version.h>
#if ROCPROFILER_VERSION >= 600
#    include "backends/rocprofiler_sdk/backend.hpp"
#    include "library/collection/sources/gpu_perf_counter_source.hpp"
#    include "library/pmc/device_providers/rocprofiler_sdk/provider.hpp"
#endif

#if defined(ROCPROFSYS_BUILD_AINIC)
#    include "library/collection/sources/nic_smi_source.hpp"
#    include "library/pmc/collectors/nic/device.hpp"
#endif

#if defined(ROCPROFSYS_BUILD_HIPFILE)
#    include "library/collection/sources/hipfile_source.hpp"
#    include "library/pmc/collectors/hipfile/device.hpp"
#    include "library/pmc/device_providers/hipfile/provider.hpp"
#endif

#include <chrono>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace rocprofsys::pmc
{
namespace
{
#if ROCPROFILER_VERSION >= 600
using gpu_perf_counter_provider_t = device_providers::rocprofiler_sdk::provider<
    backends::rocprofiler_sdk::backend_factory<::rocprofsys::rocprofiler_sdk::wrapper>>;
#endif

#if defined(ROCPROFSYS_BUILD_AINIC)
using nic_device_t = collectors::nic::device<amd_smi_device_t>;
#endif

#if defined(ROCPROFSYS_BUILD_HIPFILE)
using hipfile_device_t = collectors::hipfile::device<hipfile_provider_t::backend_t>;
#endif
}  // namespace

std::shared_ptr<amd_smi_provider_t>
providers::amd_smi()
{
    const std::lock_guard lock(m_mutex);
    if(!m_amd_smi && config::get_use_amd_smi())
    {
        try
        {
            m_amd_smi = amd_smi_provider_factory_t::create();
        } catch(const std::runtime_error& error)
        {
            LOG_ERROR("Failed to create the AMD SMI provider: {}", error.what());
        }
    }
    return m_amd_smi;
}

#if defined(ROCPROFSYS_BUILD_HIPFILE)
std::shared_ptr<hipfile_provider_t>
providers::hipfile()
{
    const std::lock_guard lock(m_mutex);
    if(!m_hipfile)
    {
        try
        {
            m_hipfile = hipfile_provider_factory_t::create();
        } catch(const std::runtime_error& error)
        {
            LOG_ERROR("Failed to create the hipFile provider: {}", error.what());
        }
    }
    return m_hipfile;
}
#endif

void
providers::shutdown()
{
    const std::lock_guard lock(m_mutex);
    if(m_amd_smi)
    {
        m_amd_smi->shutdown();
    }
#if defined(ROCPROFSYS_BUILD_HIPFILE)
    if(m_hipfile)
    {
        m_hipfile->shutdown();
    }
#endif
}

void
providers::release()
{
    const std::lock_guard lock(m_mutex);
    m_amd_smi.reset();
#if defined(ROCPROFSYS_BUILD_HIPFILE)
    m_hipfile.reset();
#endif
}

std::unique_ptr<collection::source_interface>
make_hipfile_source([[maybe_unused]] std::chrono::nanoseconds interval,
                    [[maybe_unused]] providers&               shared,
                    [[maybe_unused]] scheduler_factory_t      make_scheduler)
{
#if defined(ROCPROFSYS_BUILD_HIPFILE)
    if(!config::get_use_hipfile())
    {
        return nullptr;
    }
    using source_t =
        collection::sources::hipfile_source<hipfile_provider_t, hipfile_device_t,
                                            collectors::settings_policy>;
    return std::make_unique<source_t>(
        interval, [&shared] { return shared.hipfile(); }, std::move(make_scheduler));
#else
    return nullptr;
#endif
}

std::unique_ptr<collection::source_interface>
make_nic_smi_source([[maybe_unused]] std::chrono::nanoseconds interval,
                    [[maybe_unused]] providers&               shared,
                    [[maybe_unused]] scheduler_factory_t      make_scheduler)
{
#if defined(ROCPROFSYS_BUILD_AINIC)
    using source_t = collection::sources::nic_smi_source<amd_smi_provider_t, nic_device_t,
                                                         collectors::settings_policy>;
    return std::make_unique<source_t>(
        interval, [&shared] { return shared.amd_smi(); }, std::move(make_scheduler));
#else
    return nullptr;
#endif
}

std::unique_ptr<collection::source_interface>
make_gpu_smi_source(std::chrono::nanoseconds interval, providers& shared,
                    scheduler_factory_t make_scheduler)
{
    using source_t = collection::sources::gpu_smi_source<amd_smi_provider_t, gpu_device_t,
                                                         collectors::settings_policy>;
    return std::make_unique<source_t>(
        interval, [&shared] { return shared.amd_smi(); }, std::move(make_scheduler));
}

#if ROCPROFILER_VERSION >= 600
namespace
{
// Counters without :device= are broadcast to every GPU agent.
auto
broadcast_counters(collectors::gpu_perf_counter::gpu_perf_counter_settings settings,
                   const std::vector<std::shared_ptr<agent>>&              agent_list)
{
    auto counters = std::move(settings.explicit_counters);
    for(const auto& name : settings.broadcast_names)
    {
        for(const auto& gpu_agent : agent_list)
        {
            counters.push_back(
                { .name = name, .device_index = gpu_agent->device_type_index });
        }
    }
    return counters;
}
}  // namespace

process_sampler::source_factory
register_gpu_perf_counter_source(const std::vector<std::shared_ptr<agent>>& agent_list)
{
    try
    {
        auto settings =
            collectors::settings_policy::get_gpu_perf_counter_enabled_metrics();

        auto counters = broadcast_counters(std::move(settings), agent_list);

        const auto enabled_metrics =
            collectors::gpu_perf_counter::enabled_metrics{ std::move(counters) };

        // SDK contexts can only be created while the SDK is configured, which is now.
        const auto provider =
            std::make_shared<gpu_perf_counter_provider_t>(agent_list, enabled_metrics);

        LOG_DEBUG("Registered GPU Perf Counter PMC source");
        return [provider](std::chrono::nanoseconds interval,
                          scheduler_factory_t      make_scheduler)
                   -> std::unique_ptr<collection::source_interface> {
            using source_t =
                collection::sources::gpu_perf_counter_source<gpu_perf_counter_provider_t,
                                                             collectors::settings_policy>;
            return std::make_unique<source_t>(provider, interval,
                                              std::move(make_scheduler));
        };
    } catch(const std::runtime_error& runtime_exception)
    {
        LOG_ERROR("Failed to register SDK PMC source: {}", runtime_exception.what());
        return nullptr;
    }
}
#endif
}  // namespace rocprofsys::pmc
