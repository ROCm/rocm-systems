// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "library/components/backtrace_metrics.hpp"
#include "library/components/ensure_storage.hpp"
#include "library/components/fwd.hpp"
#include "library/config.hpp"
#include "library/debug.hpp"
#include "library/perfetto.hpp"
#include "library/ptl.hpp"
#include "library/runtime.hpp"
#include "library/sampling.hpp"
#include "library/thread_info.hpp"
#include "library/tracing.hpp"

#include <timemory/backends/papi.hpp>
#include <timemory/backends/threading.hpp>
#include <timemory/components/data_tracker/components.hpp>
#include <timemory/components/macros.hpp>
#include <timemory/components/papi/extern.hpp>
#include <timemory/components/papi/papi_array.hpp>
#include <timemory/components/papi/papi_vector.hpp>
#include <timemory/components/rusage/components.hpp>
#include <timemory/components/rusage/types.hpp>
#include <timemory/components/timing/backends.hpp>
#include <timemory/components/trip_count/extern.hpp>
#include <timemory/macros.hpp>
#include <timemory/math.hpp>
#include <timemory/mpl.hpp>
#include <timemory/mpl/quirks.hpp>
#include <timemory/mpl/type_traits.hpp>
#include <timemory/operations.hpp>
#include <timemory/storage.hpp>
#include <timemory/units.hpp>
#include <timemory/utility/backtrace.hpp>
#include <timemory/utility/demangle.hpp>
#include <timemory/utility/types.hpp>
#include <timemory/variadic.hpp>

#include <array>
#include <cstring>
#include <ctime>
#include <initializer_list>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <pthread.h>
#include <signal.h>

namespace tracing
{
using namespace ::omnitrace::tracing;
}

namespace omnitrace
{
namespace component
{
using hw_counters           = typename backtrace_metrics::hw_counters;
using signal_type_instances = thread_data<std::set<int>, category::sampling>;
using backtrace_metrics_init_instances =
    thread_data<backtrace_metrics, category::sampling>;
using sampler_running_instances = thread_data<bool, category::sampling>;
using papi_vector_instances     = thread_data<hw_counters, category::sampling>;
using papi_label_instances = thread_data<std::vector<std::string>, category::sampling>;

namespace
{
struct perfetto_rusage
{};

unique_ptr_t<std::vector<std::string>>&
get_papi_labels(int64_t _tid)
{
    static auto& _v =
        papi_label_instances::instances(papi_label_instances::construct_on_init{});
    return _v.at(_tid);
}

unique_ptr_t<hw_counters>&
get_papi_vector(int64_t _tid)
{
    static auto& _v = papi_vector_instances::instances();
    if(_tid == threading::get_id()) papi_vector_instances::construct();
    return _v.at(_tid);
}

unique_ptr_t<backtrace_metrics>&
get_backtrace_metrics_init(int64_t _tid)
{
    static auto& _v = backtrace_metrics_init_instances::instances();
    return _v.at(_tid);
}

unique_ptr_t<bool>&
get_sampler_running(int64_t _tid)
{
    static auto& _v = sampler_running_instances::instances(
        sampler_running_instances::construct_on_init{}, false);
    return _v.at(_tid);
}
}  // namespace

std::string
backtrace_metrics::label()
{
    return "backtrace_metrics";
}

std::string
backtrace_metrics::description()
{
    return "Records sampling data";
}

void
backtrace_metrics::start()
{}

void
backtrace_metrics::stop()
{}

void
backtrace_metrics::sample(int)
{
    auto _tid   = threading::get_id();
    auto _cache = tim::rusage_cache{ RUSAGE_THREAD };
    m_cpu       = tim::get_clock_thread_now<int64_t, std::nano>();
    m_mem_peak  = _cache.get_peak_rss();
    m_ctx_swch  = _cache.get_num_priority_context_switch() +
                 _cache.get_num_voluntary_context_switch();
    m_page_flt = _cache.get_num_major_page_faults() + _cache.get_num_minor_page_faults();

    if constexpr(tim::trait::is_available<hw_counters>::value)
    {
        if(tim::trait::runtime_enabled<hw_counters>::get())
        {
            assert(get_papi_vector(_tid).get() != nullptr);
            m_hw_counter = get_papi_vector(_tid)->record();
        }
    }
}

void
backtrace_metrics::configure(bool _setup, int64_t _tid)
{
    auto& _running    = get_sampler_running(_tid);
    bool  _is_running = (!_running) ? false : *_running;

    ensure_storage<comp::trip_count, sampling_wall_clock, sampling_cpu_clock, hw_counters,
                   sampling_percent>{}();

    if(_setup && !_is_running)
    {
        (void) get_debug_sampling();  // make sure query in sampler does not allocate
        assert(_tid == threading::get_id());

        if constexpr(tim::trait::is_available<hw_counters>::value)
        {
            perfetto_counter_track<hw_counters>::init();
            OMNITRACE_DEBUG("HW COUNTER: starting...\n");
            if(get_papi_vector(_tid))
            {
                using common_type_t = typename hw_counters::common_type;
                get_papi_vector(_tid)->start();
                *get_papi_labels(_tid) = comp::papi_common::get_events<common_type_t>();
            }
        }
    }
    else if(!_setup && _is_running)
    {
        OMNITRACE_DEBUG("Destroying sampler for thread %lu...\n", _tid);
        *_running = false;

        if constexpr(tim::trait::is_available<hw_counters>::value)
        {
            if(_tid == threading::get_id())
            {
                if(get_papi_vector(_tid)) get_papi_vector(_tid)->stop();
                OMNITRACE_DEBUG("HW COUNTER: stopped...\n");
            }
        }
        OMNITRACE_DEBUG("Sampler destroyed for thread %lu\n", _tid);
    }
}

void
backtrace_metrics::init_perfetto(int64_t _tid)
{
    auto _hw_cnt_labels = *get_papi_labels(_tid);
    auto _tid_name      = JOIN("", '[', _tid, ']');

    if(!perfetto_counter_track<perfetto_rusage>::exists(_tid))
    {
        perfetto_counter_track<perfetto_rusage>::emplace(
            _tid, JOIN(' ', "Thread Peak Memory Usage", _tid_name, "(S)"), "MB");
        perfetto_counter_track<perfetto_rusage>::emplace(
            _tid, JOIN(' ', "Thread Context Switches", _tid_name, "(S)"));
        perfetto_counter_track<perfetto_rusage>::emplace(
            _tid, JOIN(' ', "Thread Page Faults", _tid_name, "(S)"));
    }

    if(!perfetto_counter_track<hw_counters>::exists(_tid) &&
       tim::trait::runtime_enabled<hw_counters>::get())
    {
        for(auto& itr : _hw_cnt_labels)
        {
            std::string _desc = tim::papi::get_event_info(itr).short_descr;
            if(_desc.empty()) _desc = itr;
            OMNITRACE_CI_THROW(_desc.empty(), "Empty description for %s\n", itr.c_str());
            perfetto_counter_track<hw_counters>::emplace(
                _tid, JOIN(' ', "Thread", _desc, _tid_name, "(S)"));
        }
    }
}

void
backtrace_metrics::fini_perfetto(int64_t _tid)
{
    auto        _hw_cnt_labels = *get_papi_labels(_tid);
    const auto& _thread_info   = thread_info::get(_tid, SequentTID);

    OMNITRACE_CI_THROW(!_thread_info, "Error! missing thread info for tid=%li\n", _tid);
    if(!_thread_info) return;

    uint64_t _ts = _thread_info->get_stop();

    TRACE_COUNTER("thread_peak_memory",
                  perfetto_counter_track<perfetto_rusage>::at(_tid, 0), _ts, 0);

    TRACE_COUNTER("thread_context_switch",
                  perfetto_counter_track<perfetto_rusage>::at(_tid, 1), _ts, 0);

    TRACE_COUNTER("thread_page_fault",
                  perfetto_counter_track<perfetto_rusage>::at(_tid, 2), _ts, 0);

    if(tim::trait::runtime_enabled<hw_counters>::get())
    {
        for(size_t i = 0; i < perfetto_counter_track<hw_counters>::size(_tid); ++i)
        {
            if(i < _hw_cnt_labels.size())
            {
                TRACE_COUNTER("thread_hardware_counter",
                              perfetto_counter_track<hw_counters>::at(_tid, i), _ts, 0.0);
            }
        }
    }
}

void
backtrace_metrics::post_process_perfetto(int64_t _tid, uint64_t _ts) const
{
    TRACE_COUNTER("thread_peak_memory",
                  perfetto_counter_track<perfetto_rusage>::at(_tid, 0), _ts,
                  m_mem_peak / units::megabyte);

    TRACE_COUNTER("thread_context_switch",
                  perfetto_counter_track<perfetto_rusage>::at(_tid, 1), _ts, m_ctx_swch);

    TRACE_COUNTER("thread_page_fault",
                  perfetto_counter_track<perfetto_rusage>::at(_tid, 2), _ts, m_page_flt);

    if(tim::trait::runtime_enabled<hw_counters>::get())
    {
        for(size_t i = 0; i < perfetto_counter_track<hw_counters>::size(_tid); ++i)
        {
            if(i < m_hw_counter.size())
            {
                TRACE_COUNTER("thread_hardware_counter",
                              perfetto_counter_track<hw_counters>::at(_tid, i), _ts,
                              m_hw_counter.at(i));
            }
        }
    }
}
}  // namespace component
}  // namespace omnitrace

OMNITRACE_INSTANTIATE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, omnitrace::component::backtrace_wall_clock>), true,
    double)

OMNITRACE_INSTANTIATE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, omnitrace::component::backtrace_cpu_clock>), true,
    double)

OMNITRACE_INSTANTIATE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, omnitrace::component::backtrace_fraction>), true,
    double)

TIMEMORY_INITIALIZE_STORAGE(omnitrace::component::backtrace_metrics)
