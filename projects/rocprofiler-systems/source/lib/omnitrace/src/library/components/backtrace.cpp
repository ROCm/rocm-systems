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

#include "library/components/fwd.hpp"
#include "library/components/rocm_smi.hpp"
#include "library/config.hpp"
#include "library/debug.hpp"
#include "library/perfetto.hpp"
#include "library/ptl.hpp"
#include "library/sampling.hpp"

#include <timemory/backends/papi.hpp>
#include <timemory/backends/threading.hpp>
#include <timemory/components/data_tracker/components.hpp>
#include <timemory/components/macros.hpp>
#include <timemory/components/papi/extern.hpp>
#include <timemory/components/papi/papi_array.hpp>
#include <timemory/components/papi/papi_vector.hpp>
#include <timemory/components/timing/backends.hpp>
#include <timemory/components/trip_count/extern.hpp>
#include <timemory/macros.hpp>
#include <timemory/math.hpp>
#include <timemory/mpl.hpp>
#include <timemory/mpl/quirks.hpp>
#include <timemory/mpl/type_traits.hpp>
#include <timemory/operations.hpp>
#include <timemory/sampling/allocator.hpp>
#include <timemory/sampling/sampler.hpp>
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
#include <type_traits>

#include <pthread.h>
#include <signal.h>

namespace
{
template <typename... Tp>
struct ensure_storage
{
    TIMEMORY_DEFAULT_OBJECT(ensure_storage)

    void operator()() const { TIMEMORY_FOLD_EXPRESSION((*this)(tim::type_list<Tp>{})); }

private:
    template <typename Up, std::enable_if_t<tim::trait::is_available<Up>::value, int> = 0>
    void operator()(tim::type_list<Up>) const
    {
        using namespace tim;
        static thread_local auto _storage = operation::get_storage<Up>{}();
        static thread_local auto _tid     = threading::get_id();
        static thread_local auto _dtor =
            scope::destructor{ []() { operation::set_storage<Up>{}(nullptr, _tid); } };

        tim::operation::set_storage<Up>{}(_storage, _tid);
        if(_tid == 0 && !_storage) tim::trait::runtime_enabled<Up>::set(false);
    }

    template <typename Up,
              std::enable_if_t<!tim::trait::is_available<Up>::value, long> = 0>
    void operator()(tim::type_list<Up>) const
    {
        tim::trait::runtime_enabled<Up>::set(false);
    }
};
}  // namespace

namespace omnitrace
{
namespace component
{
using hw_counters               = typename backtrace::hw_counters;
using signal_type_instances     = thread_data<std::set<int>, api::sampling>;
using backtrace_init_instances  = thread_data<backtrace, api::sampling>;
using sampler_running_instances = thread_data<bool, api::sampling>;
using papi_vector_instances     = thread_data<hw_counters, api::sampling>;

namespace
{
std::unique_ptr<hw_counters>&
get_papi_vector(int64_t _tid)
{
    static auto& _v = papi_vector_instances::instances();
    if(_tid == threading::get_id()) papi_vector_instances::construct();
    return _v.at(_tid);
}

std::unique_ptr<backtrace>&
get_backtrace_init(int64_t _tid)
{
    static auto& _v = backtrace_init_instances::instances();
    return _v.at(_tid);
}

std::unique_ptr<bool>&
get_sampler_running(int64_t _tid)
{
    static auto& _v = sampler_running_instances::instances();
    return _v.at(_tid);
}
}  // namespace

bool
backtrace::operator<(const backtrace& rhs) const
{
    return (m_ts == rhs.m_ts) ? (m_tid < rhs.m_tid) : (m_ts < rhs.m_ts);
}

std::vector<std::string>
backtrace::get() const
{
    std::vector<std::string> _v{};
    _v.reserve(m_size);
    for(size_t i = 0; i < m_size; ++i)
        _v.emplace_back(m_data.at(i));
    return _v;
}

void
backtrace::preinit()
{
    sampling_wall_clock::label()       = "sampling_wall_clock";
    sampling_wall_clock::description() = "Wall clock time (via sampling)";

    sampling_cpu_clock::label()       = "sampling_cpu_clock";
    sampling_cpu_clock::description() = "CPU clock time (via sampling)";

    sampling_percent::label()       = "sampling_percent";
    sampling_percent::description() = "Percentage of samples";

    sampling_gpu_busy::label()       = "sampling_gpu_busy_percent";
    sampling_gpu_busy::description() = "Utilization of GPU(s)";
    sampling_gpu_busy::set_precision(0);
    sampling_gpu_busy::set_format_flags(sampling_gpu_busy::get_format_flags() &
                                        std::ios_base::showpoint);

    sampling_gpu_memory::label()       = "sampling_gpu_memory_usage";
    sampling_gpu_memory::description() = "Memory usage of GPU(s)";

    sampling_gpu_power::label()        = "sampling_gpu_power";
    sampling_gpu_power::description()  = "Power usage of GPU(s)";
    sampling_gpu_power::unit()         = units::watt;
    sampling_gpu_power::display_unit() = "watts";
    sampling_gpu_power::set_precision(2);
    sampling_gpu_power::set_format_flags(sampling_gpu_power::get_format_flags());

    sampling_gpu_temp::label()        = "sampling_gpu_temperature";
    sampling_gpu_temp::description()  = "Temperature of GPU(s)";
    sampling_gpu_temp::unit()         = 1;
    sampling_gpu_temp::display_unit() = "degC";
    sampling_gpu_temp::set_precision(1);
    sampling_gpu_temp::set_format_flags(sampling_gpu_temp::get_format_flags());
}

std::string
backtrace::label()
{
    return "backtrace";
}

std::string
backtrace::description()
{
    return "Records backtrace data";
}

void
backtrace::start()
{}

void
backtrace::stop()
{}

bool
backtrace::empty() const
{
    return (m_size == 0);
}

size_t
backtrace::size() const
{
    return m_size;
}

backtrace::time_point_type
backtrace::get_timestamp() const
{
    return m_ts;
}

int64_t
backtrace::get_thread_cpu_timestamp() const
{
    return m_thr_cpu_ts;
}

void
backtrace::sample(int signum)
{
    if(signum != -1 && get_state() != State::Active)
    {
        OMNITRACE_CONDITIONAL_PRINT(
            get_debug_sampling(),
            "request to sample (signal %i) ignored because omnitrace is not active\n",
            signum);
        return;
    }

    if(get_debug_sampling())
    {
        static auto _timestamp_str = [](const auto& _tp) {
            char _repr[64];
            std::memset(_repr, '\0', sizeof(_repr));
            std::time_t _value = system_clock::to_time_t(_tp);
            // alternative: "%c %Z"
            if(std::strftime(_repr, sizeof(_repr), "%a %b %d %T %Y %Z",
                             std::localtime(&_value)) > 0)
                return std::string{ _repr };
            return std::string{};
        };

        static thread_local size_t _tot  = 0;
        static thread_local auto   _last = system_clock::now();
        auto                       _now  = system_clock::now();
        auto                       _diff = (_now - _last).count();
        _last                            = _now;
        _tot += _diff;

        OMNITRACE_PRINT(
            "Sample on signal %i taken at %s after interval %zu :: total %zu\n", signum,
            _timestamp_str(_now).c_str(), _diff, _tot);
    }

    m_size       = 0;
    m_tid        = threading::get_id();
    m_ts         = clock_type::now();
    m_thr_cpu_ts = tim::get_clock_thread_now<int64_t, std::nano>();
    m_mem_peak   = tim::get_peak_rss(RUSAGE_THREAD);
    m_data       = tim::get_unw_backtrace<128, 4, false>();
    auto* itr    = m_data.begin();
    for(; itr != m_data.end(); ++itr, ++m_size)
    {
        if(strlen(*itr) == 0) break;
    }
    std::reverse(m_data.begin(), itr);
    if(!get_debug_sampling())
    {
        bool _ignore = false;
        for(auto& itr : m_data)
        {
            if(strlen(itr) == 0) break;
            if(strncmp(itr, "funlockfile", 11) == 0) _ignore = true;
            if(_ignore && strlen(itr) > 0)
            {
                OMNITRACE_DEBUG("Discarding sample: '%s'...\n", itr);
                itr[0] = '\0';
                --m_size;
            }
        }
    }

    if constexpr(tim::trait::is_available<hw_counters>::value)
    {
        if(tim::trait::runtime_enabled<hw_counters>::get())
        {
            assert(get_papi_vector(m_tid).get() != nullptr);
            static thread_local auto& _pv         = get_papi_vector(m_tid);
            auto                      _hw_counter = _pv->record();
            auto _num_hw_counters = std::min<size_t>(_hw_counter.size(), num_hw_counters);
            for(size_t i = 0; i < _num_hw_counters; ++i)
            {
                auto& _last     = get_last_hwcounters().at(i);
                auto  itr       = _hw_counter.at(i);
                m_hw_counter[i] = itr - _last;
                _last           = itr;
            }
        }
    }
}

std::set<int>
backtrace::configure(bool _setup, int64_t _tid)
{
    auto& _sampler      = sampling::get_sampler(_tid);
    auto& _running      = get_sampler_running(_tid);
    bool  _is_running   = (!_running) ? false : *_running;
    auto& _signal_types = sampling::get_signal_types(_tid);

    ensure_storage<comp::trip_count, sampling_wall_clock, sampling_cpu_clock, hw_counters,
                   sampling_percent>{}();

    if(_setup && !_sampler && !_is_running)
    {
        (void) get_debug_sampling();  // make sure query in sampler does not allocate
        assert(_tid == threading::get_id());
        sampling::block_signals(*_signal_types);
        if constexpr(tim::trait::is_available<hw_counters>::value)
        {
            perfetto_counter_track<hw_counters>::init();
            OMNITRACE_DEBUG("HW COUNTER: starting...\n");
            if(get_papi_vector(_tid)) get_papi_vector(_tid)->start();
        }

        auto _alrm_freq = 1.0 / std::min<double>(get_sampling_freq(), 5.0);
        auto _prof_freq = 1.0 / get_sampling_freq();
        auto _delay     = std::max<double>(1.0e-3, get_sampling_delay());

        OMNITRACE_DEBUG("Configuring sampler for thread %lu...\n", _tid);
        sampler_running_instances::construct(true);
        sampling::sampler_instances::construct("omnitrace", _tid, *_signal_types);
        _sampler->set_signals(*_signal_types);
        _sampler->set_flags(SA_RESTART);
        _sampler->set_delay(_delay);
        _sampler->set_frequency(_prof_freq, { SIGPROF });
        _sampler->set_frequency(_alrm_freq, { SIGALRM });

        static_assert(tim::trait::buffer_size<sampling::sampler_t>::value > 0,
                      "Error! Zero buffer size");

        OMNITRACE_CONDITIONAL_THROW(
            _sampler->get_buffer_size() <= 0,
            "dynamic sampler requires a positive buffer size: %zu",
            _sampler->get_buffer_size());

        OMNITRACE_DEBUG("Sampler for thread %lu will be triggered %5.1fx per second "
                        "(every %5.2e seconds)...\n",
                        _tid, _sampler->get_frequency(units::sec),
                        _sampler->get_rate(units::sec));

        // (void) sampling::sampler_t::get_samplers(_tid);
        backtrace_init_instances::construct();
        get_backtrace_init(_tid)->sample();
        _sampler->configure(false);
        _sampler->start();
    }
    else if(!_setup && _sampler && _is_running)
    {
        OMNITRACE_DEBUG("Destroying sampler for thread %lu...\n", _tid);
        *_running = false;

        if(_tid == threading::get_id())
        {
            sampling::block_signals(*_signal_types);
        }

        // this propagates to all threads
        if(_tid == 0) _sampler->ignore(*_signal_types);

        _sampler->stop();
        _sampler->swap_data();
        if constexpr(tim::trait::is_available<hw_counters>::value)
        {
            if(_tid == threading::get_id())
            {
                if(get_papi_vector(_tid)) get_papi_vector(_tid)->stop();
                OMNITRACE_DEBUG("HW COUNTER: stopped...\n");
            }
        }
    }

    return (_signal_types) ? *_signal_types : std::set<int>{};
}

backtrace::hw_counter_data_t&
backtrace::get_last_hwcounters()
{
    static thread_local auto _v = hw_counter_data_t{ 0 };
    return _v;
}

void
backtrace::post_process(int64_t _tid)
{
    namespace quirk = tim::quirk;

    configure(false, _tid);

    auto& _sampler = sampling::sampler_instances::instances().at(_tid);
    if(!_sampler)
    {
        // this should be relatively common
        OMNITRACE_DEBUG(
            "Post-processing sampling entries for thread %lu skipped (no sampler)\n",
            _tid);
        return;
    }

    auto& _init = backtrace_init_instances::instances().at(_tid);
    if(!_init)
    {
        // this is not common
        OMNITRACE_PRINT(
            "Post-processing sampling entries for thread %lu skipped (not initialized)\n",
            _tid);
        return;
    }

    // check whether the call-stack entry should be used. -1 means break, 0 means continue
    auto _use_label = [](const std::string& _lbl, bool _check_internal) -> short {
        // debugging feature
        static bool _keep_internal =
            tim::get_env<bool>("OMNITRACE_SAMPLING_KEEP_INTERNAL", get_debug_sampling());
        const auto _npos = std::string::npos;
        if(_keep_internal) return 1;
        if(_lbl.find("omnitrace_init_tooling") != _npos) return -1;
        if(_lbl.find("omnitrace_push_trace") != _npos) return -1;
        if(_lbl.find("omnitrace_pop_trace") != _npos) return -1;
        if(_lbl.find("amd_comgr_") == 0) return -1;
        if(_check_internal)
        {
            if(std::regex_search(
                   _lbl, std::regex("(14pthread_gotcha7wrapper|default_error_condition)",
                                    std::regex_constants::optimize)))
                return 0;
            else if(std::regex_search(
                        _lbl, std::regex("(8sampling9backtrace9configure|"
                                         "8sampling15unblock_signals|pthread_sigmask)",
                                         std::regex_constants::optimize)))
                return 0;
        }
        return 1;
    };

    // in the dyninst binary rewrite runtime, instrumented functions are appended with
    // "_dyninst", i.e. "main" will show up as "main_dyninst" in the backtrace.
    auto _patch_label = [](std::string _lbl) -> std::string {
        // debugging feature
        static bool _keep_suffix = tim::get_env<bool>(
            "OMNITRACE_SAMPLING_KEEP_DYNINST_SUFFIX", get_debug_sampling());
        if(_keep_suffix) return _lbl;
        const std::string _dyninst{ "_dyninst" };
        auto              _pos = _lbl.find(_dyninst);
        if(_pos == std::string::npos) return _lbl;
        return _lbl.replace(_pos, _dyninst.length(), "");
    };

    using common_type_t = typename hw_counters::common_type;
    auto _hw_cnt_labels = (get_papi_vector(_tid))
                              ? comp::papi_common::get_events<common_type_t>()
                              : std::vector<int>{};

    auto _process_perfetto_counters = [&](const std::vector<sampling::bundle_t*>& _data) {
        if(!perfetto_counter_track<comp::peak_rss>::exists(_tid))
        {
            auto _thrname = TIMEMORY_JOIN("", "[Thread ", _tid, "] ");
            auto addendum = [&](const std::string& _v) { return _thrname + _v + " (S)"; };
            perfetto_counter_track<comp::peak_rss>::emplace(
                _tid, addendum("Peak Memory Usage"), "MB");
        }

        if(!perfetto_counter_track<hw_counters>::exists(_tid) &&
           tim::trait::runtime_enabled<hw_counters>::get())
        {
            auto _thrname = TIMEMORY_JOIN("", "[Thread ", _tid, "] ");
            auto addendum = [&](const std::string& _v) { return _thrname + _v + " (S)"; };
            for(auto& itr : _hw_cnt_labels)
            {
                perfetto_counter_track<hw_counters>::emplace(
                    _tid, addendum(tim::papi::get_event_info(itr).short_descr), "");
            }
        }

        for(const auto& ditr : _data)
        {
            const auto* _bt = ditr->get<backtrace>();
            if(_bt->m_tid != _tid) continue;

            auto _ts = static_cast<uint64_t>(_bt->m_ts.time_since_epoch().count());

            TRACE_COUNTER("sampling", perfetto_counter_track<comp::peak_rss>::at(_tid, 0),
                          _ts, _bt->m_mem_peak / units::megabyte);

            if(tim::trait::runtime_enabled<hw_counters>::get())
            {
                for(size_t i = 0; i < perfetto_counter_track<hw_counters>::size(_tid);
                    ++i)
                {
                    if(i < _bt->m_hw_counter.size())
                    {
                        TRACE_COUNTER("sampling",
                                      perfetto_counter_track<hw_counters>::at(_tid, i),
                                      _ts, _bt->m_hw_counter.at(i));
                    }
                }
            }
        }
    };

    auto _process_perfetto = [&](const std::vector<sampling::bundle_t*>& _data,
                                 bool                                    _rename) {
        if(_rename)
            threading::set_thread_name(TIMEMORY_JOIN(" ", "Thread", _tid, "(S)").c_str());
        time_point_type _last_wall_ts = _init->get_timestamp();

        for(const auto& ditr : _data)
        {
            const auto* _bt = ditr->get<backtrace>();
            if(_bt->m_tid != _tid) continue;

            static std::set<std::string> _static_strings{};
            std::string                  _last = {};
            for(const auto& itr : _bt->get())
            {
                auto _name = tim::demangle(_patch_label(itr));
                auto _use =
                    _use_label(_name, !_last.empty() &&
                                          (_last == "start_thread" || _last == "clone"));
                if(_use == -1) break;
                if(_use == 0) continue;
                auto sitr = _static_strings.emplace(_name);
                _last     = *sitr.first;
                auto _ts  = static_cast<uint64_t>(_bt->m_ts.time_since_epoch().count());

                TRACE_EVENT_BEGIN(
                    "sampling", perfetto::StaticString{ sitr.first->c_str() },
                    static_cast<uint64_t>(_last_wall_ts.time_since_epoch().count()));
                TRACE_EVENT_END("sampling", _ts);
            }
            _last_wall_ts = _bt->m_ts;
        }
    };

    auto _raw_data = _sampler->get_allocator().get_data();
    // single sample that is useless (backtrace to unblocking signals)
    if(_raw_data.size() == 1 && _raw_data.front().size() <= 1) _raw_data.clear();

    std::vector<sampling::bundle_t*> _data{};
    for(auto& ditr : _raw_data)
    {
        _data.reserve(_data.size() + ditr.size());
        for(auto& ritr : ditr)
        {
            auto* _bt = ritr.get<backtrace>();
            if(!_bt)
            {
                OMNITRACE_PRINT(
                    "Warning! Nullptr to backtrace instance for thread %lu...\n", _tid);
                continue;
            }
            if(_bt->empty()) continue;
            _data.emplace_back(&ritr);
        }
    }

    if(_data.empty()) return;

    OMNITRACE_CONDITIONAL_PRINT(
        get_verbose() >= 0 || get_debug_sampling(),
        "Post-processing %zu sampling entries for thread %lu...\n", _data.size(), _tid);

    std::sort(_data.begin(), _data.end(),
              [](const sampling::bundle_t* _lhs, const sampling::bundle_t* _rhs) {
                  return _lhs->get<backtrace>()->m_ts < _rhs->get<backtrace>()->m_ts;
              });

    if(get_use_perfetto())
    {
        _process_perfetto_counters(_data);

        if(_tid == 0 && get_mode() == Mode::Sampling)
            _process_perfetto(_data, false);
        else
        {
            pthread_gotcha::push_enable_sampling_on_child_threads(false);
            std::thread{ _process_perfetto, _data, true }.join();
            pthread_gotcha::pop_enable_sampling_on_child_threads();
        }
    }

    if(!get_use_timemory()) return;

    std::map<int64_t, std::map<int64_t, int64_t>> _depth_sum = {};
    auto                                          _scope     = tim::scope::config{};
    if(get_timeline_sampling()) _scope += scope::timeline{};
    if(get_flat_sampling()) _scope += scope::flat{};

    time_point_type _last_wall_ts = _init->get_timestamp();
    int64_t         _last_cpu_ts  = _init->get_thread_cpu_timestamp();
    for(auto& ditr : _data)
    {
        using bundle_t = tim::lightweight_tuple<comp::trip_count, sampling_wall_clock,
                                                sampling_cpu_clock, hw_counters>;

        auto* _bt = ditr->get<backtrace>();

        if(_bt->m_ts < _last_wall_ts) continue;

        double _elapsed_wc = (_bt->m_ts - _last_wall_ts).count();
        double _elapsed_cc = (_bt->m_thr_cpu_ts - _last_cpu_ts);

        std::vector<bundle_t> _tc{};
        _tc.reserve(_bt->size());

        // generate the instances of the tuple of components and start them
        for(const auto& itr : _bt->get())
        {
            auto _lbl = _patch_label(itr);
            auto _use =
                _use_label(_lbl, !_tc.empty() && (_tc.back().key() == "start_thread" ||
                                                  _tc.back().key() == "clone"));
            if(_use == -1) break;
            if(_use == 0) continue;
            _tc.emplace_back(tim::string_view_t{ _lbl }, _scope);
            _tc.back().push(_bt->m_tid);
            _tc.back().start();
        }

        // stop the instances and update the values as needed
        for(size_t i = 0; i < _tc.size(); ++i)
        {
            auto&  itr    = _tc.at(_tc.size() - i - 1);
            size_t _depth = 0;
            _depth_sum[_bt->m_tid][_depth] += 1;
            itr.stop();
            if constexpr(tim::trait::is_available<sampling_wall_clock>::value)
            {
                auto* _sc = itr.get<sampling_wall_clock>();
                if(_sc)
                {
                    auto _value = _elapsed_wc / sampling_wall_clock::get_unit();
                    _sc->set_value(_value);
                    _sc->set_accum(_value);
                }
            }
            if constexpr(tim::trait::is_available<sampling_cpu_clock>::value)
            {
                auto* _cc = itr.get<sampling_cpu_clock>();
                if(_cc)
                {
                    _cc->set_value(_elapsed_cc / sampling_cpu_clock::get_unit());
                    _cc->set_accum(_elapsed_cc / sampling_cpu_clock::get_unit());
                }
            }
            if constexpr(tim::trait::is_available<hw_counters>::value)
            {
                auto* _hw_counter = itr.get<hw_counters>();
                if(_hw_counter)
                {
                    _hw_counter->set_value(_bt->m_hw_counter);
                    _hw_counter->set_accum(_bt->m_hw_counter);
                }
            }
            itr.pop();
        }
        _last_wall_ts = _bt->m_ts;
        _last_cpu_ts  = _bt->m_thr_cpu_ts;
    }

    for(auto&& ditr : _data)
    {
        using bundle_t =
            tim::lightweight_tuple<sampling_percent, quirk::config<quirk::tree_scope>>;

        auto* _bt = ditr->get<backtrace>();

        std::vector<bundle_t> _tc{};
        _tc.reserve(_bt->size());

        // generate the instances of the tuple of components and start them
        for(const auto& itr : _bt->get())
        {
            auto _lbl = _patch_label(itr);
            auto _use =
                _use_label(_lbl, !_tc.empty() && _tc.back().key() == "start_thread");
            if(_use == -1) break;
            if(_use == 0) continue;
            _tc.emplace_back(tim::string_view_t{ _lbl });
            _tc.back().push(_bt->m_tid);
            _tc.back().start();
        }

        // stop the instances and update the values as needed
        for(size_t i = 0; i < _tc.size(); ++i)
        {
            auto&  itr    = _tc.at(_tc.size() - i - 1);
            size_t _depth = 0;
            double _value = (1.0 / _depth_sum[_bt->m_tid][_depth]) * 100.0;
            itr.store(std::plus<double>{}, _value);
            itr.stop();
            itr.pop();
        }
    }
}

}  // namespace component
}  // namespace omnitrace

TIMEMORY_INSTANTIATE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, omnitrace::component::backtrace_wall_clock>), true,
    double)

TIMEMORY_INSTANTIATE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, omnitrace::component::backtrace_cpu_clock>), true,
    double)

TIMEMORY_INSTANTIATE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, omnitrace::component::backtrace_fraction>), true,
    double)

TIMEMORY_INITIALIZE_STORAGE(omnitrace::component::backtrace)
