// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "library/pmc/collectors/common/collector_slice.hpp"
#include "library/pmc/collectors/common/settings.hpp"
#include "library/pmc/collectors/gpu/cache_policy.hpp"
#include "library/pmc/collectors/gpu/collector.hpp"
#include "library/pmc/collectors/gpu/perfetto_policy.hpp"
#include "library/pmc/device_providers/amd_smi/provider.hpp"

#if defined(ROCPROFSYS_BUILD_AINIC)
#    include "library/pmc/collectors/nic/cache_policy.hpp"
#    include "library/pmc/collectors/nic/collector.hpp"
#    include "library/pmc/collectors/nic/perfetto_policy.hpp"
#endif

#include "core/common.hpp"
#include "core/components/fwd.hpp"
#include "core/state.hpp"
#include "library/pmc/device_providers/amd_smi/drivers/driver.hpp"
#include "library/runtime.hpp"

#include "library/pmc/sampler.hpp"

#include <amd_smi/amdsmi.h>
#include <timemory/backends/threading.hpp>
#include <timemory/components/timing/backends.hpp>
#include <timemory/mpl/type_traits.hpp>
#include <timemory/units.hpp>
#include <timemory/utility/delimit.hpp>
#include <timemory/utility/locking.hpp>

#include <cassert>
#include <memory>
#include <sys/resource.h>
#include <vector>

namespace rocprofsys
{
namespace pmc
{

std::atomic<State>&
get_state()
{
    static std::atomic<State> _v{ State::PreInit };
    return _v;
}

namespace
{

bool&
is_initialized()
{
    static bool _v = false;
    return _v;
}

struct gpu_production_config
{
    using SettingsApi = collectors::settings_policy;
    using PerfettoApi = collectors::gpu::perfetto_policy;
    using CacheApi    = collectors::gpu::cache_policy;
};

#if defined(ROCPROFSYS_BUILD_AINIC)
struct nic_production_config
{
    using SettingsApi = collectors::settings_policy;
    using PerfettoApi = collectors::nic::perfetto_policy;
    using CacheApi    = collectors::nic::cache_policy;
};
#endif

using provider_factory_t =
    device_providers::amd_smi::provider_factory<drivers::amd_smi::driver_factory>;
using provider_t      = provider_factory_t::provider_t;
using gpu_collector_t = collectors::gpu::collector<provider_t, gpu_production_config>;
#if defined(ROCPROFSYS_BUILD_AINIC)
using nic_collector_t = collectors::nic::collector<provider_t, nic_production_config>;
#endif

std::shared_ptr<provider_t> g_device_provider;

std::unique_ptr<gpu_collector_t> g_gpu_collector;
#if defined(ROCPROFSYS_BUILD_AINIC)
std::unique_ptr<nic_collector_t> g_nic_collector;
#endif

std::vector<collectors::collector_slice> g_collector_slices;

}  // namespace

void
set_state(State _v)
{
    pmc::get_state().store(_v);
}

void
config()
{
    for(auto& slice : g_collector_slices)
    {
        slice.config();
    }
    LOG_DEBUG("Setting PMC sampler state to active...");
    pmc::set_state(State::Active);
}

void
sample()
{
    auto_lock_t _lk{ type_mutex<category::amd_smi>() };

    if(pmc::get_state() != State::Active)
    {
        return;
    }

    auto timestamp = static_cast<int64_t>(tim::get_clock_real_now<size_t, std::nano>());

    for(auto& slice : g_collector_slices)
    {
        slice.sample(timestamp);
    }
}

void
setup()
{
    auto_lock_t _lk{ type_mutex<category::amd_smi>() };

    if(is_initialized())
    {
        return;
    }

    ROCPROFSYS_SCOPED_SAMPLING_ON_CHILD_THREADS(false);

    try
    {
        // Create and inject device provider (shared between GPU and NIC collectors)
        g_device_provider = provider_factory_t::create();

        g_gpu_collector = std::make_unique<gpu_collector_t>(g_device_provider);
#if defined(ROCPROFSYS_BUILD_AINIC)
        g_nic_collector = std::make_unique<nic_collector_t>(g_device_provider);
#endif

        g_collector_slices.clear();
        g_collector_slices.emplace_back(*g_gpu_collector);
#if defined(ROCPROFSYS_BUILD_AINIC)
        g_collector_slices.emplace_back(*g_nic_collector);
#endif

        for(auto& slice : g_collector_slices)
        {
            slice.setup();
        }

        is_initialized() = true;
    } catch(const std::runtime_error& _e)
    {
        LOG_ERROR("Exception thrown when initializing PMC sampler: {}", _e.what());
    }
}

void
shutdown()
{
    auto_lock_t _lk{ type_mutex<category::amd_smi>() };

    if(!is_initialized())
    {
        return;
    }

    LOG_DEBUG("Shutting down PMC sampler.");

    try
    {
        for(auto& slice : g_collector_slices)
        {
            slice.shutdown();
        }
    } catch(const std::runtime_error& _e)
    {
        LOG_ERROR("Exception thrown when shutting down PMC sampler: {}", _e.what());
    }

    is_initialized() = false;
}

void
post_process()
{
    LOG_DEBUG("Post-processing PMC samples ({} slices).", g_collector_slices.size());
    for(auto& slice : g_collector_slices)
    {
        slice.post_process();
    }
    g_collector_slices.clear();
    g_device_provider.reset();
}

void
postfork_child_cleanup()
{
    LOG_DEBUG("Disabling PMC sampling in child process after fork.");
    pmc::get_state().store(State::Finalized);
    for(auto& slice : g_collector_slices)
    {
        slice.shutdown();
    }
    g_collector_slices.clear();
    g_gpu_collector.reset();
#if defined(ROCPROFSYS_BUILD_AINIC)
    g_nic_collector.reset();
#endif
    g_device_provider.reset();
    is_initialized() = false;
}

void
postfork_parent_reinit()
{
    LOG_DEBUG("Reinitializing PMC sampling in parent process after fork.");
    shutdown();
    setup();
}

}  // namespace pmc
}  // namespace rocprofsys

ROCPROFSYS_INSTANTIATE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_busy_gfx>),
    true, double)

ROCPROFSYS_INSTANTIATE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_busy_umc>),
    true, double)

ROCPROFSYS_INSTANTIATE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_busy_mm>),
    true, double)

ROCPROFSYS_INSTANTIATE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_temp>), true,
    double)

ROCPROFSYS_INSTANTIATE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_power>), true,
    double)

ROCPROFSYS_INSTANTIATE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_memory>), true,
    double)

ROCPROFSYS_INSTANTIATE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_vcn>), true,
    double)

ROCPROFSYS_INSTANTIATE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_jpeg>), true,
    double)
