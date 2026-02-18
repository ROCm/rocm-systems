// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "library/pmc/collectors/common/collector_slice.hpp"
#include "library/pmc/collectors/common/settings.hpp"
#include "library/pmc/collectors/gpu/collector.hpp"
#include "library/pmc/device_providers/amd_smi/provider.hpp"
#include "library/pmc/output_policies/cache_policy.hpp"
#include "library/pmc/output_policies/perfetto_policy.hpp"

#include "core/common.hpp"
#include "core/components/fwd.hpp"
#include "core/state.hpp"
#include "library/pmc/device_providers/amd_smi/drivers/driver.hpp"
#include "library/runtime.hpp"

#include "library/pmc/sampler.hpp"

// For now let's limit the PMC sampler to ROCM only.
#if defined(ROCPROFSYS_USE_ROCM) && ROCPROFSYS_USE_ROCM > 0

#    if defined(NDEBUG)
#        undef NDEBUG
#    endif

#    include <amd_smi/amdsmi.h>
#    include <timemory/backends/threading.hpp>
#    include <timemory/components/timing/backends.hpp>
#    include <timemory/mpl/type_traits.hpp>
#    include <timemory/units.hpp>
#    include <timemory/utility/delimit.hpp>
#    include <timemory/utility/locking.hpp>

#    include <cassert>
#    include <optional>
#    include <sys/resource.h>
#    include <vector>

namespace rocprofsys
{
namespace pmc
{
namespace
{

bool&
is_initialized()
{
    static bool _v = false;
    return _v;
}

std::atomic<State>&
get_state()
{
    static std::atomic<State> _v{ State::PreInit };
    return _v;
}

struct production_config
{
    using SettingsApi = collectors::settings_policy;
    using PerfettoApi = output_policies::perfetto_policy;
    using CacheApi    = output_policies::cache_policy;
};

using provider_factory_t =
    device_providers::amd_smi::provider_factory<drivers::amd_smi::driver_factory>;
using provider_t      = provider_factory_t::provider_t;
using gpu_collector_t = collectors::gpu::collector<provider_t, production_config>;

/// Storage for actual collector objects (collector_slice is non-owning)
struct collector_storage
{
    std::optional<gpu_collector_t> gpu;
    // Future: std::optional<nic_collector_t> nic;
    // Future: std::optional<cpu_collector_t> cpu;
};

std::shared_ptr<provider_t>              g_device_provider;
collector_storage                        g_collectors;
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
}

void
sample()
{
    auto_lock_t _lk{ type_mutex<category::amd_smi>() };

    if(get_state() != State::Active)
    {
        return;
    }

    for(auto& slice : g_collector_slices)
    {
        slice.sample(tim::get_clock_real_now<size_t, std::nano>);
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
        // Create and inject device provider
        g_device_provider = provider_factory_t::create();

        // Create GPU collector and add to slice vector
        g_collectors.gpu.emplace(g_device_provider);
        g_collector_slices.emplace_back(*g_collectors.gpu);

        // Future: Add NIC collector
        // g_collectors.nic.emplace(g_device_provider);
        // g_collector_slices.emplace_back(*g_collectors.nic);

        // Setup all collectors
        for(auto& slice : g_collector_slices)
        {
            slice.setup();
        }

        is_initialized() = true;
    } catch(std::runtime_error& _e)
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

    LOG_INFO("Shutting down PMC sampler.");

    try
    {
        for(auto& slice : g_collector_slices)
        {
            slice.shutdown();
        }
    } catch(std::runtime_error& _e)
    {
        LOG_ERROR("Exception thrown when shutting down PMC sampler: {}", _e.what());
    }

    is_initialized() = false;
}

void
post_process()
{
    LOG_DEBUG("Post-processing PMC samples.");
    for(auto& slice : g_collector_slices)
    {
        slice.post_process();
    }
    g_collector_slices.clear();
    g_collectors.gpu.reset();
    g_device_provider.reset();
}

void
postfork_child_cleanup()
{
    LOG_DEBUG("Disabling PMC sampling in child process after fork.");
    get_state().store(State::Finalized);
    for(auto& slice : g_collector_slices)
    {
        slice.shutdown();
    }
    g_collector_slices.clear();
    g_collectors.gpu.reset();
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

#endif
