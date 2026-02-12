// Copyright (c) 2018-2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// with the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// * Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
//
// * Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimers in the
// documentation and/or other materials provided with the distribution.
//
// * Neither the names of Advanced Micro Devices, Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this Software without specific prior written permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.

#include "library/pmc/collectors/common/settings.hpp"
#include "library/pmc/collectors/gpu/collector.hpp"
#include "library/pmc/device_providers/amd_smi/provider.hpp"
#include "library/pmc/output_policies/cache_policy.hpp"
#include "library/pmc/output_policies/perfetto_policy.hpp"

// CPU collector includes (always available, no ROCM dependency)
#include "library/pmc/collectors/cpu/collector.hpp"
#include "library/pmc/device_providers/procfs/provider.hpp"
#include "library/pmc/output_policies/cpu_cache_policy.hpp"
#include "library/pmc/output_policies/cpu_perfetto_policy.hpp"

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
using provider_t        = provider_factory_t::provider_t;
using production_impl_t = collectors::gpu::collector<provider_t, production_config>;

std::shared_ptr<provider_t> g_device_provider;

std::optional<production_impl_t> g_data_collector;

}  // namespace

void
set_state(State _v)
{
    pmc::get_state().store(_v);
}

void
config()
{
    if(g_data_collector) g_data_collector->config();
}

void
sample()
{
    auto_lock_t _lk{ type_mutex<category::amd_smi>() };

    if(get_state() != State::Active)
    {
        return;
    }

    if(g_data_collector)
        g_data_collector->sample(tim::get_clock_real_now<size_t, std::nano>);
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
        g_data_collector.emplace(g_device_provider);

        // Setup the collector
        g_data_collector->setup();
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
        if(g_data_collector) g_data_collector->shutdown();
        g_device_provider.reset();
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
    if(g_data_collector) g_data_collector->post_process();
}

void
postfork_child_cleanup()
{
    LOG_DEBUG("Disabling PMC sampling in child process after fork.");
    get_state().store(State::Finalized);
    if(g_data_collector) g_data_collector->shutdown();
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

// --- CPU PMC collector (always available, no ROCM dependency) ---

#include <timemory/components/timing/backends.hpp>

namespace rocprofsys
{
namespace cpu_pmc
{
namespace
{

struct cpu_production_config
{
    using SettingsApi = pmc::collectors::settings_policy;
    using PerfettoApi = pmc::output_policies::cpu_perfetto_policy;
    using CacheApi    = pmc::output_policies::cpu_cache_policy;
};

using cpu_provider_factory_t =
    pmc::device_providers::procfs::provider_factory<pmc::drivers::procfs::driver_factory>;
using cpu_provider_t = cpu_provider_factory_t::provider_t;
using cpu_collector_t =
    pmc::collectors::cpu::collector<cpu_provider_t, cpu_production_config>;

bool&
is_cpu_initialized()
{
    static bool _v = false;
    return _v;
}

std::shared_ptr<cpu_provider_t> g_cpu_provider;
std::optional<cpu_collector_t>  g_cpu_collector;

}  // namespace

void
setup()
{
    if(is_cpu_initialized()) return;

    try
    {
        g_cpu_provider = cpu_provider_factory_t::create();
        g_cpu_collector.emplace(g_cpu_provider);
        g_cpu_collector->setup();
        is_cpu_initialized() = true;
    } catch(std::runtime_error& _e)
    {
        LOG_ERROR("Exception thrown when initializing CPU PMC sampler: {}", _e.what());
    }
}

void
config()
{
    if(g_cpu_collector) g_cpu_collector->config();
}

void
sample()
{
    if(g_cpu_collector)
        g_cpu_collector->sample(tim::get_clock_real_now<size_t, std::nano>);
}

void
shutdown()
{
    if(!is_cpu_initialized()) return;

    LOG_INFO("Shutting down CPU PMC sampler.");

    try
    {
        if(g_cpu_collector) g_cpu_collector->shutdown();
        g_cpu_provider.reset();
    } catch(std::runtime_error& _e)
    {
        LOG_ERROR("Exception thrown when shutting down CPU PMC sampler: {}", _e.what());
    }

    is_cpu_initialized() = false;
}

void
post_process()
{
    LOG_DEBUG("Post-processing CPU PMC samples.");
    if(g_cpu_collector) g_cpu_collector->post_process();
}

}  // namespace cpu_pmc
}  // namespace rocprofsys
