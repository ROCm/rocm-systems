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

#include "library/amd_smi/amd_smi_driver.hpp"
#include "library/amd_smi/amd_smi_impl.hpp"
#include "library/amd_smi/perfetto_policy.hpp"
#include "library/amd_smi/rocpd_policy.hpp"
#include "library/amd_smi/service.hpp"
#include "library/amd_smi/settings.hpp"

#include "core/common.hpp"
#include "core/components/fwd.hpp"
#include "core/state.hpp"
#include "library/runtime.hpp"

#include "library/amd_smi.hpp"

#if defined(NDEBUG)
#    undef NDEBUG
#endif

#include <amd_smi/amdsmi.h>
#include <timemory/backends/threading.hpp>
#include <timemory/components/timing/backends.hpp>
#include <timemory/mpl/type_traits.hpp>
#include <timemory/units.hpp>
#include <timemory/utility/delimit.hpp>
#include <timemory/utility/locking.hpp>

#include <cassert>
#include <sys/resource.h>

namespace rocprofsys
{
namespace amd_smi
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
    using SmiServiceFactory = smi_service_factory<amd_smi_driver_factory>;
    using SettingsApi       = settings_policy;
    using PerfettoApi       = perfetto_policy;
    using RocpdApi          = rocpd_policy;
};

using production_impl_t = amd_smi_impl<production_config>;

production_impl_t g_smi_impl;

}  // namespace

void
set_state(State _v)
{
    amd_smi::get_state().store(_v);
}

void
config()
{
    g_smi_impl.config();
}

void
sample()
{
    auto_lock_t _lk{ type_mutex<category::amd_smi>() };

    if(get_state() != State::Active)
    {
        return;
    }

    g_smi_impl.sample(tim::get_clock_real_now<size_t, std::nano>);
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
        g_smi_impl.setup();
        is_initialized() = true;
    } catch(std::runtime_error& _e)
    {
        ROCPROFSYS_VERBOSE(0, "Exception thrown when initializing amd-smi: %s\n",
                           _e.what());
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

    ROCPROFSYS_VERBOSE_F(1, "Shutting down amd-smi...\n");

    try
    {
        g_smi_impl.shutdown();
    } catch(std::runtime_error& _e)
    {
        ROCPROFSYS_VERBOSE(0, "Exception thrown when shutting down amd-smi: %s\n",
                           _e.what());
    }

    is_initialized() = false;
}

void
post_process()
{
    g_smi_impl.post_process();
}

void
postfork_child_cleanup()
{
    ROCPROFSYS_VERBOSE_F(2, "Disabling AMD SMI in child process after fork...\n");
    get_state().store(State::Finalized);
    g_smi_impl.shutdown();
    is_initialized() = false;
}

void
postfork_parent_reinit()
{
    ROCPROFSYS_VERBOSE_F(2, "Reinitializing AMD SMI in parent process after fork...\n");
    shutdown();
    setup();
}

}  // namespace amd_smi
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
