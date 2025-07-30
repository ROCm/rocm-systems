// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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
#include "core/components/fwd.hpp"
#include "core/config.hpp"
#include "core/debug.hpp"
#include "core/perfetto.hpp"
#include "library/components/ensure_storage.hpp"
#include "library/ptl.hpp"
#include "library/runtime.hpp"
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
#include <timemory/mpl/types.hpp>
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
using namespace ::rocprofsys::tracing;
}

namespace rocprofsys
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
    return papi_label_instances::instance(construct_on_thread{ _tid });
}

unique_ptr_t<hw_counters>&
get_papi_vector(int64_t _tid)
{
    return papi_vector_instances::instance(construct_on_thread{ _tid });
}

unique_ptr_t<backtrace_metrics>&
get_backtrace_metrics_init(int64_t _tid)
{
    return backtrace_metrics_init_instances::instance(construct_on_thread{ _tid });
}

unique_ptr_t<bool>&
get_sampler_running(int64_t _tid)
{
    return sampler_running_instances::instance(construct_on_thread{ _tid }, false);
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

std::vector<std::string>
backtrace_metrics::get_hw_counter_labels(int64_t _tid)
{
    auto& _v = get_papi_labels(_tid);
    return (_v) ? *_v : std::vector<std::string>{};
}

void
backtrace_metrics::start()
{}

void
backtrace_metrics::stop()
{}

namespace
{
template <typename... Tp>
auto
get_enabled(tim::type_list<Tp...>)
{
    constexpr size_t N  = sizeof...(Tp);
    auto             _v = std::bitset<N>{};
    size_t           _n = 0;
    (_v.set(_n++, trait::runtime_enabled<Tp>::get()), ...);
    return _v;
}
}  // namespace
void
backtrace_metrics::sample(int)
{
    if(!get_enabled(type_list<category::process_sampling, backtrace_metrics>{}).all())
    {
        m_valid.reset();
        return;
    }

    m_valid = get_enabled(categories_t{});

    // return if everything is disabled
    if(!m_valid.any()) return;

    auto _cache = tim::rusage_cache{ RUSAGE_THREAD };
    m_cpu       = tim::get_clock_thread_now<int64_t, std::nano>();
    m_mem_peak  = _cache.get_peak_rss();
    m_ctx_swch  = _cache.get_num_priority_context_switch() +
                 _cache.get_num_voluntary_context_switch();
    m_page_flt = _cache.get_num_major_page_faults() + _cache.get_num_minor_page_faults();

    if constexpr(tim::trait::is_available<hw_counters>::value)
    {
        constexpr auto hw_counters_idx = tim::index_of<hw_counters, categories_t>::value;
        constexpr auto hw_category_idx =
            tim::index_of<category::thread_hardware_counter, categories_t>::value;

        auto _tid = threading::get_id();
        if(m_valid.test(hw_category_idx) && m_valid.test(hw_counters_idx))
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
            ROCPROFSYS_DEBUG("HW COUNTER: starting...\n");
            if(get_papi_vector(_tid))
            {
                get_papi_vector(_tid)->start();
                *get_papi_labels(_tid) = get_papi_vector(_tid)->get_config()->labels;
            }
        }
    }
    else if(!_setup && _is_running)
    {
        ROCPROFSYS_DEBUG("Destroying sampler for thread %lu...\n", _tid);
        *_running = false;

        if constexpr(tim::trait::is_available<hw_counters>::value)
        {
            if(_tid == threading::get_id())
            {
                if(get_papi_vector(_tid)) get_papi_vector(_tid)->stop();
                ROCPROFSYS_DEBUG("HW COUNTER: stopped...\n");
            }
        }
        ROCPROFSYS_DEBUG("Sampler destroyed for thread %lu\n", _tid);
    }
}

void
backtrace_metrics::init_perfetto(int64_t _tid, valid_array_t _valid)
{
    auto _hw_cnt_labels = *get_papi_labels(_tid);
    auto _tid_name      = JOIN("", '[', _tid, ']');

    if(!perfetto_counter_track<perfetto_rusage>::exists(_tid))
    {
        if(get_valid(category::thread_cpu_time{}, _valid))
            perfetto_counter_track<perfetto_rusage>::emplace(
                _tid, JOIN(' ', "Thread CPU time", _tid_name, "(S)"), "sec");
        if(get_valid(category::thread_peak_memory{}, _valid))
            perfetto_counter_track<perfetto_rusage>::emplace(
                _tid, JOIN(' ', "Thread Peak Memory Usage", _tid_name, "(S)"), "MB");
        if(get_valid(category::thread_context_switch{}, _valid))
            perfetto_counter_track<perfetto_rusage>::emplace(
                _tid, JOIN(' ', "Thread Context Switches", _tid_name, "(S)"));
        if(get_valid(category::thread_page_fault{}, _valid))
            perfetto_counter_track<perfetto_rusage>::emplace(
                _tid, JOIN(' ', "Thread Page Faults", _tid_name, "(S)"));
    }

    if(!perfetto_counter_track<hw_counters>::exists(_tid) &&
       get_valid(type_list<hw_counters>{}, _valid) &&
       get_valid(category::thread_hardware_counter{}, _valid))
    {
        for(auto& itr : _hw_cnt_labels)
        {
            std::string _desc = tim::papi::get_event_info(itr).short_descr;
            if(_desc.empty()) _desc = itr;
            ROCPROFSYS_CI_THROW(_desc.empty(), "Empty description for %s\n", itr.c_str());
            perfetto_counter_track<hw_counters>::emplace(
                _tid, JOIN(' ', "Thread", _desc, _tid_name, "(S)"));
        }
    }
}

void
backtrace_metrics::fini_perfetto(int64_t _tid, valid_array_t _valid)
{
    auto        _hw_cnt_labels = *get_papi_labels(_tid);
    const auto& _thread_info   = thread_info::get(_tid, SequentTID);

    ROCPROFSYS_CI_THROW(!_thread_info, "Error! missing thread info for tid=%li\n", _tid);
    if(!_thread_info) return;

    uint64_t _ts         = _thread_info->get_stop();
    uint64_t _rusage_idx = 0;

    if(get_valid(category::thread_cpu_time{}, _valid))
    {
        TRACE_COUNTER(trait::name<category::thread_cpu_time>::value,
                      perfetto_counter_track<perfetto_rusage>::at(_tid, _rusage_idx++),
                      _ts, 0);
    }

    if(get_valid(category::thread_peak_memory{}, _valid))
    {
        TRACE_COUNTER(trait::name<category::thread_peak_memory>::value,
                      perfetto_counter_track<perfetto_rusage>::at(_tid, _rusage_idx++),
                      _ts, 0);
    }

    if(get_valid(category::thread_context_switch{}, _valid))
    {
        TRACE_COUNTER(trait::name<category::thread_context_switch>::value,
                      perfetto_counter_track<perfetto_rusage>::at(_tid, _rusage_idx++),
                      _ts, 0);
    }

    if(get_valid(category::thread_page_fault{}, _valid))
    {
        TRACE_COUNTER(trait::name<category::thread_page_fault>::value,
                      perfetto_counter_track<perfetto_rusage>::at(_tid, _rusage_idx++),
                      _ts, 0);
    }

    if(get_valid(type_list<hw_counters>{}, _valid) &&
       get_valid(category::thread_hardware_counter{}, _valid))
    {
        for(size_t i = 0; i < perfetto_counter_track<hw_counters>::size(_tid); ++i)
        {
            if(i < _hw_cnt_labels.size())
            {
                TRACE_COUNTER(trait::name<category::thread_hardware_counter>::value,
                              perfetto_counter_track<hw_counters>::at(_tid, i), _ts, 0.0);
            }
        }
    }
}

backtrace_metrics&
backtrace_metrics::operator-=(const backtrace_metrics& _rhs)
{
    auto& _lhs = *this;

    if(_lhs(category::thread_cpu_time{}))
    {
        _lhs.m_cpu -= _rhs.m_cpu;
    }

    if(_lhs(category::thread_peak_memory{}))
    {
        _lhs.m_mem_peak -= _rhs.m_mem_peak;
    }

    if(_lhs(category::thread_context_switch{}))
    {
        _lhs.m_ctx_swch -= _rhs.m_ctx_swch;
    }

    if(_lhs(category::thread_page_fault{}))
    {
        _lhs.m_page_flt -= _rhs.m_page_flt;
    }

    if(_lhs(type_list<hw_counters>{}) && _lhs(category::thread_hardware_counter{}))
    {
        for(size_t i = 0; i < _lhs.m_hw_counter.size(); ++i)
            _lhs.m_hw_counter.at(i) -= _rhs.m_hw_counter.at(i);
    }

    return _lhs;
}

void
backtrace_metrics::post_process_perfetto(int64_t _tid, uint64_t _ts) const
{
    uint64_t _rusage_idx = 0;

    if((*this)(category::thread_cpu_time{}))
    {
        TRACE_COUNTER(trait::name<category::thread_cpu_time>::value,
                      perfetto_counter_track<perfetto_rusage>::at(_tid, _rusage_idx++),
                      _ts, m_cpu / units::sec);
    }

    if((*this)(category::thread_peak_memory{}))
    {
        TRACE_COUNTER(trait::name<category::thread_peak_memory>::value,
                      perfetto_counter_track<perfetto_rusage>::at(_tid, _rusage_idx++),
                      _ts, m_mem_peak / units::megabyte);
    }

    if((*this)(category::thread_context_switch{}))
    {
        TRACE_COUNTER(trait::name<category::thread_context_switch>::value,
                      perfetto_counter_track<perfetto_rusage>::at(_tid, _rusage_idx++),
                      _ts, m_ctx_swch);
    }

    if((*this)(category::thread_page_fault{}))
    {
        TRACE_COUNTER(trait::name<category::thread_page_fault>::value,
                      perfetto_counter_track<perfetto_rusage>::at(_tid, _rusage_idx++),
                      _ts, m_page_flt);
    }

    if((*this)(type_list<hw_counters>{}) && (*this)(category::thread_hardware_counter{}))
    {
        for(size_t i = 0; i < perfetto_counter_track<hw_counters>::size(_tid); ++i)
        {
            if(i < m_hw_counter.size())
            {
                TRACE_COUNTER(trait::name<category::thread_hardware_counter>::value,
                              perfetto_counter_track<hw_counters>::at(_tid, i), _ts,
                              m_hw_counter.at(i));
            }
        }
    }
}
}  // namespace component
}  // namespace rocprofsys

ROCPROFSYS_INSTANTIATE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_wall_clock>), true,
    double)

ROCPROFSYS_INSTANTIATE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_cpu_clock>), true,
    double)

ROCPROFSYS_INSTANTIATE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_fraction>), true,
    double)

TIMEMORY_INITIALIZE_STORAGE(rocprofsys::component::backtrace_metrics)
