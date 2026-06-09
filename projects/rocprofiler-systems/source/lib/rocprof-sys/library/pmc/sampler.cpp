// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/pmc/collectors/common/collector_slice.hpp"
#include "library/pmc/collectors/common/settings.hpp"
#include "library/pmc/collectors/gpu/cache_policy.hpp"
#include "library/pmc/collectors/gpu/collector.hpp"
#include "library/pmc/collectors/gpu/perfetto_policy.hpp"
#include "library/pmc/device_providers/amd_smi/provider.hpp"
#include <cstdint>
#include <rocprofiler-sdk/version.h>
#if ROCPROFILER_VERSION >= 600
#    include "library/pmc/collectors/gpu_perf_counter/collector.hpp"
#    include "library/pmc/device_providers/rocprofiler_sdk/provider.hpp"
#endif

#if defined(ROCPROFSYS_BUILD_AINIC)
#    include "library/pmc/collectors/nic/cache_policy.hpp"
#    include "library/pmc/collectors/nic/collector.hpp"
#    include "library/pmc/collectors/nic/perfetto_policy.hpp"
#endif

#include "library/pmc/collectors/cpu/cache_policy.hpp"
#include "library/pmc/collectors/cpu/collector.hpp"
#include "library/pmc/collectors/cpu/perfetto_policy.hpp"
#include "library/pmc/device_providers/procfs/provider.hpp"

#include "core/agent.hpp"
#include "core/common.hpp"
#include "core/components/fwd.hpp"
#include "core/state.hpp"
#include "library/pmc/device_providers/amd_smi/drivers/driver.hpp"
#if ROCPROFILER_VERSION >= 600
#    include "library/pmc/device_providers/rocprofiler_sdk/drivers/driver.hpp"
#endif
#include "library/runtime.hpp"

#include "library/pmc/sampler.hpp"

#include "logger/debug.hpp"

#include <amd_smi/amdsmi.h>
#include <timemory/backends/threading.hpp>
#include <timemory/components/timing/backends.hpp>
#include <timemory/mpl/type_traits.hpp>
#include <timemory/units.hpp>
#include <timemory/utility/delimit.hpp>
#include <timemory/utility/locking.hpp>

#include <atomic>
#include <cassert>
#include <cstdint>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <sys/resource.h>
#include <vector>

namespace rocprofsys::pmc
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

struct cpu_production_config
{
    using SettingsApi = collectors::settings_policy;
    using PerfettoApi = collectors::cpu::perfetto_policy;
    using CacheApi    = collectors::cpu::cache_policy;
};

using provider_factory_t =
    device_providers::amd_smi::provider_factory<drivers::amd_smi::driver_factory>;
using provider_t      = provider_factory_t::provider_t;
using gpu_collector_t = collectors::gpu::collector<provider_t, gpu_production_config>;

#if ROCPROFILER_VERSION >= 600
using gpu_perf_counter_provider_t =
    device_providers::rocprofiler_sdk::provider<drivers::rocprofiler_sdk::driver_factory>;
using gpu_perf_counter_collector_t =
    collectors::gpu_perf_counter::collector<gpu_perf_counter_provider_t>;
#endif

#if defined(ROCPROFSYS_BUILD_AINIC)
using nic_collector_t = collectors::nic::collector<provider_t, nic_production_config>;
#endif

using cpu_provider_factory_t =
    device_providers::procfs::provider_factory<drivers::procfs::driver_factory>;
using cpu_provider_t  = cpu_provider_factory_t::provider_t;
using cpu_collector_t = collectors::cpu::collector<cpu_provider_t, cpu_production_config>;

std::shared_ptr<provider_t> g_device_provider;

std::unique_ptr<gpu_collector_t> g_gpu_collector;
#if ROCPROFILER_VERSION >= 600
std::shared_ptr<gpu_perf_counter_provider_t>  g_gpu_perf_counter_provider;
std::unique_ptr<gpu_perf_counter_collector_t> g_gpu_perf_counter_collector;
#endif
#if defined(ROCPROFSYS_BUILD_AINIC)
std::unique_ptr<nic_collector_t> g_nic_collector;
#endif

std::shared_ptr<cpu_provider_t>  g_cpu_provider;
std::unique_ptr<cpu_collector_t> g_cpu_collector;

std::vector<collectors::collector_slice> g_collector_slices;

std::atomic<bool> g_reinit_pending{ false };

// Tears down the AMD SMI collector slices and marks the sampler uninitialized.
// This is the only teardown performed on the post-fork reinit path: that reinit
// exists solely to refresh AMD SMI device handles in the child.
void
shutdown_amd_smi_collectors()
{
    auto_lock_t _lk{ type_mutex<category::amd_smi>() };

    if(!is_initialized())
    {
        return;
    }

    LOG_DEBUG("Shutting down AMD-SMI PMC sampler.");

    try
    {
        for(auto& slice : g_collector_slices)
        {
            slice.shutdown();
        }
    } catch(const std::runtime_error& _e)
    {
        LOG_ERROR("Exception thrown when shutting down AMD-SMI PMC sampler: {}",
                  _e.what());
    }

    is_initialized() = false;
}

// Tears down the rocprofiler-sdk GPU hardware perf-counter collector. This must
// NOT run on the post-fork reinit path: the collector is registered once (from
// sdk_tool_configure) and is never recreated by setup(); its SDK contexts are
// owned by the parent and survive fork(), so tearing it down would permanently
// disable GPU hardware-counter sampling. Only the real finalize resets it.
void
shutdown_gpu_hw_collector()
{
#if ROCPROFILER_VERSION >= 600
    auto_lock_t _lk{ type_mutex<category::amd_smi>() };

    LOG_DEBUG("Shutting down rocprofiler-sdk GPU hardware counter collector.");

    if(g_gpu_perf_counter_collector) g_gpu_perf_counter_collector->shutdown();
    g_gpu_perf_counter_collector.reset();
    g_gpu_perf_counter_provider.reset();
#endif
}

void
reinit_if_pending()
{
    bool _expected = true;
    if(!g_reinit_pending.compare_exchange_strong(_expected, false)) return;

    LOG_DEBUG("Performing deferred PMC reinit after fork.");
    shutdown_amd_smi_collectors();
    setup();
}

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
    reinit_if_pending();

    auto_lock_t _lk{ type_mutex<category::amd_smi>() };

    if(pmc::get_state() != State::Active)
    {
        return;
    }

    auto timestamp =
        static_cast<std::int64_t>(tim::get_clock_real_now<size_t, std::nano>());

    for(auto& slice : g_collector_slices)
    {
        slice.sample(timestamp);
    }
#if ROCPROFILER_VERSION >= 600
    if(g_gpu_perf_counter_collector) g_gpu_perf_counter_collector->sample(timestamp);
#endif
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
        g_collector_slices.clear();

        g_cpu_provider  = cpu_provider_factory_t::create();
        g_cpu_collector = std::make_unique<cpu_collector_t>(g_cpu_provider);

        g_collector_slices.emplace_back(*g_cpu_collector);

        if(config::get_use_amd_smi())
        {
            // Create and inject device provider (shared between GPU and NIC collectors)
            g_device_provider = provider_factory_t::create();

            g_gpu_collector = std::make_unique<gpu_collector_t>(g_device_provider);
#if defined(ROCPROFSYS_BUILD_AINIC)
            g_nic_collector = std::make_unique<nic_collector_t>(g_device_provider);
#endif

            g_collector_slices.emplace_back(*g_gpu_collector);
#if defined(ROCPROFSYS_BUILD_AINIC)
            g_collector_slices.emplace_back(*g_nic_collector);
#endif
        }

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
    shutdown_amd_smi_collectors();
    shutdown_gpu_hw_collector();
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
#if ROCPROFILER_VERSION >= 600
    if(g_gpu_perf_counter_collector) g_gpu_perf_counter_collector->post_process();
    g_gpu_perf_counter_collector.reset();
    g_gpu_perf_counter_provider.reset();
#endif
    g_device_provider.reset();
    g_cpu_provider.reset();
}

void
pause()
{
    auto_lock_t _lk{ type_mutex<category::amd_smi>() };

    if(pmc::get_state() != State::Active || !is_initialized())
    {
        return;
    }

    auto timestamp =
        static_cast<std::int64_t>(tim::get_clock_real_now<size_t, std::nano>());

    for(auto& slice : g_collector_slices)
    {
        slice.pause(timestamp);
    }
#if ROCPROFILER_VERSION >= 600
    if(g_gpu_perf_counter_collector) g_gpu_perf_counter_collector->pause(timestamp);
#endif
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
#if ROCPROFILER_VERSION >= 600
    // Do not call shutdown() or stop() — rocprofiler SDK / HSA mutexes may be
    // inherited locked from the parent. The parent owns SDK context shutdown.
    // gpu_perf_counter is not in g_collector_slices so the loop above is safe.
    g_gpu_perf_counter_collector.reset();
    g_gpu_perf_counter_provider.reset();
#endif
    g_gpu_collector.reset();
#if defined(ROCPROFSYS_BUILD_AINIC)
    g_nic_collector.reset();
#endif
    g_cpu_collector.reset();
    g_device_provider.reset();
    g_cpu_provider.reset();
    is_initialized() = false;
}

void
postfork_parent_reinit()
{
    // Cannot call shutdown()/setup() here: setup() queries AMD SMI, which
    // internally calls fork(), which would re-enter the postfork handler
    // chain while glibc still holds __fork_lock. Defer to next sample().
    g_reinit_pending.store(true);
}

// Intentionally a second-phase setup called after rocprofiler initializes (outside of
// setup()), because the rocprofiler context and agent list are not available at
// setup() time. shutdown() symmetrically stops and resets the provider/collector
// globals created here.
void
prefork_lock_sampler()
{
    // A raw lock must be used, as we must lock in one function and unlock in another
    // Thread that calls this should use postfork_parent_unlock_sampler()
    // Child that inherits the locked mutex needs to use
    // postfork_child_reset_sampler_lock()
    type_mutex<category::amd_smi>().lock();
}

void
postfork_parent_unlock_sampler()
{
    // Same kernel tid as thread that called prefork_lock_sampler(),
    // so the unlock succeeds
    type_mutex<category::amd_smi>().unlock();
}

void
postfork_child_reset_sampler_lock()
{
    // Overwrite the existing mutex with a new one of the same type (placement-new)
    // We cannot unlock it in the child as it has been locked by the parent and has a
    // different kernel TID then what was used to lock it. Destructor cannot be used
    // either, as on a locked mutex, it is undefined behaviour
    using mutex_type =
        std::remove_reference<decltype(type_mutex<category::amd_smi>())>::type;
    auto& _m = type_mutex<category::amd_smi>();
    ::new(static_cast<void*>(&_m)) mutex_type{};
}

#if ROCPROFILER_VERSION >= 600
void
register_gpu_perf_counter_source(const std::vector<std::shared_ptr<agent>>& agent_list)
{
    auto_lock_t _lk{ type_mutex<category::amd_smi>() };

    try
    {
        auto settings =
            collectors::settings_policy::get_gpu_perf_counter_enabled_metrics();

        auto counters = std::move(settings.explicit_counters);
        // counters without :device= are broadcast to every GPU agent
        for(const auto& name : settings.broadcast_names)
        {
            for(const auto& gpu_agent : agent_list)
            {
                counters.push_back({ name, gpu_agent->device_type_index });
            }
        }

        const auto enabled_metrics =
            collectors::gpu_perf_counter::enabled_metrics{ std::move(counters) };

        g_gpu_perf_counter_provider =
            std::make_shared<gpu_perf_counter_provider_t>(agent_list, enabled_metrics);
        g_gpu_perf_counter_collector =
            std::make_unique<gpu_perf_counter_collector_t>(g_gpu_perf_counter_provider);

        g_gpu_perf_counter_collector->setup();
        g_gpu_perf_counter_collector->config();

        LOG_DEBUG("Registered GPU Perf Counter PMC source");
    } catch(const std::runtime_error& runtime_exception)
    {
        LOG_ERROR("Failed to register SDK PMC source: {}", runtime_exception.what());
    }
}
#endif
}  // namespace rocprofsys::pmc
