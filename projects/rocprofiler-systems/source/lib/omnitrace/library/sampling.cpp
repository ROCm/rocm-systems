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

#include "library/sampling.hpp"
#include "library/common.hpp"
#include "library/components/backtrace.hpp"
#include "library/components/backtrace_metrics.hpp"
#include "library/components/backtrace_timestamp.hpp"
#include "library/components/fwd.hpp"
#include "library/config.hpp"
#include "library/debug.hpp"
#include "library/ptl.hpp"
#include "library/runtime.hpp"
#include "library/thread_data.hpp"
#include "library/thread_info.hpp"
#include "library/tracing.hpp"
#include "library/utility.hpp"

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
#include <timemory/utility/backtrace.hpp>
#include <timemory/utility/demangle.hpp>
#include <timemory/utility/types.hpp>
#include <timemory/variadic.hpp>

#include <array>
#include <chrono>
#include <condition_variable>
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

namespace omnitrace
{
namespace sampling
{
using hw_counters               = typename component::backtrace_metrics::hw_counters;
using signal_type_instances     = thread_data<std::set<int>, category::sampling>;
using sampler_running_instances = thread_data<bool, category::sampling>;
using bundle_t =
    tim::lightweight_tuple<backtrace_timestamp, backtrace, backtrace_metrics>;
using sampler_t              = tim::sampling::sampler<bundle_t, tim::sampling::dynamic>;
using sampler_instances      = thread_data<sampler_t, category::sampling>;
using sampler_init_instances = thread_data<bundle_t, category::sampling>;

using tim::sampling::timer;
}  // namespace sampling
}  // namespace omnitrace

OMNITRACE_DEFINE_CONCRETE_TRAIT(prevent_reentry, sampling::sampler_t, std::true_type)

OMNITRACE_DEFINE_CONCRETE_TRAIT(provide_backtrace, sampling::sampler_t, std::false_type)

OMNITRACE_DEFINE_CONCRETE_TRAIT(buffer_size, sampling::sampler_t,
                                TIMEMORY_ESC(std::integral_constant<size_t, 1024>))

namespace omnitrace
{
namespace sampling
{
namespace
{
template <typename... Args>
void
thread_sigmask(Args... _args)
{
    auto _err = pthread_sigmask(_args...);
    if(_err != 0)
    {
        errno = _err;
        perror("pthread_sigmask");
        exit(EXIT_FAILURE);
    }
}

template <typename Tp>
sigset_t
get_signal_set(Tp&& _v)
{
    sigset_t _sigset;
    sigemptyset(&_sigset);
    for(auto itr : _v)
        sigaddset(&_sigset, itr);
    return _sigset;
}

template <typename Tp>
std::string
get_signal_names(Tp&& _v)
{
    std::string _sig_names{};
    for(auto&& itr : _v)
        _sig_names += std::get<0>(tim::signal_settings::get_info(
                          static_cast<tim::sys_signal>(itr))) +
                      " ";
    return _sig_names.substr(0, _sig_names.length() - 1);
}

unique_ptr_t<sampler_t>&
get_sampler(int64_t _tid = threading::get_id())
{
    static auto& _v = sampler_instances::instances();
    return _v.at(_tid);
}

unique_ptr_t<bundle_t>&
get_sampler_init(int64_t _tid = threading::get_id())
{
    static auto& _v = sampler_init_instances::instances();
    if(!_v.at(_tid)) _v.at(_tid) = unique_ptr_t<bundle_t>{ new bundle_t{} };
    return _v.at(_tid);
}

unique_ptr_t<bool>&
get_sampler_running(int64_t _tid)
{
    static auto& _v = sampler_running_instances::instances(
        sampler_running_instances::construct_on_init{}, false);
    return _v.at(_tid);
}

auto&
get_duration_cv()
{
    static auto _v = std::condition_variable{};
    return _v;
}

auto&
get_duration_thread()
{
    static auto _v = std::unique_ptr<std::thread>{};
    return _v;
}

void
start_duration_thread()
{
    static std::mutex            _start_mutex{};
    std::unique_lock<std::mutex> _start_lk{ _start_mutex, std::defer_lock };
    if(!_start_lk.owns_lock()) _start_lk.lock();

    if(!get_duration_thread() && config::get_sampling_duration() > 0.0)
    {
        // we may need to protect against recursion bc of pthread wrapper
        static bool _protect = false;
        if(_protect) return;
        _protect   = true;
        auto _now  = std::chrono::steady_clock::now();
        auto _end  = _now + std::chrono::nanoseconds{ static_cast<uint64_t>(
                               config::get_sampling_duration() * units::sec) };
        auto _func = [_end]() {
            thread_info::init(true);
            std::mutex _mutex{};
            bool       _wait = true;
            while(_wait)
            {
                _wait = false;
                std::unique_lock<std::mutex> _lk{ _mutex };
                get_duration_cv().wait_until(_lk, _end);
                auto _premature = (std::chrono::steady_clock::now() < _end);
                auto _finalized = (get_state() == State::Finalized);
                if(_premature && !_finalized)
                {
                    // protect against spurious wakeups
                    OMNITRACE_VERBOSE(
                        2, "%sSpurious wakeup of sampling duration thread...\n",
                        tim::log::color::warning());
                    _wait = true;
                }
                else if(_finalized)
                {
                    break;
                }
                else
                {
                    OMNITRACE_VERBOSE(1,
                                      "Sampling duration of %f seconds has elapsed. "
                                      "Shutting down sampling...\n",
                                      config::get_sampling_duration());
                    shutdown();
                }
            }
        };

        OMNITRACE_VERBOSE(1, "Sampling will be disabled after %f seconds...\n",
                          config::get_sampling_duration());

        OMNITRACE_SCOPED_SAMPLING_ON_CHILD_THREADS(false);
        get_duration_thread() = std::make_unique<std::thread>(_func);
        _protect              = false;
    }
}

std::set<int>
configure(bool _setup, int64_t _tid = threading::get_id())
{
    const auto& _info         = thread_info::get(_tid, InternalTID);
    auto&       _sampler      = sampling::get_sampler(_tid);
    auto&       _running      = get_sampler_running(_tid);
    bool        _is_running   = (!_running) ? false : *_running;
    auto&       _signal_types = sampling::get_signal_types(_tid);

    OMNITRACE_SCOPED_SAMPLING_ON_CHILD_THREADS(false);

    auto&& _cpu_tids  = get_sampling_cpu_tids();
    auto&& _real_tids = get_sampling_real_tids();

    auto _erase_tid_signal = [_tid, &_signal_types](auto& _tids, int _signum) {
        if(!_tids.empty())
        {
            if(_tids.count(_tid) == 0)
            {
                OMNITRACE_VERBOSE(3, "Disabling SIG%i from thread %li\n", _signum, _tid);
                _signal_types->erase(_signum);
            }
        }
    };

    _erase_tid_signal(_cpu_tids, get_cputime_signal());
    _erase_tid_signal(_real_tids, get_realtime_signal());

    if(_setup && !_sampler && !_is_running && !_signal_types->empty())
    {
        // if this thread has an offset ID, that means it was created internally
        // and is probably here bc it called a function which was instrumented.
        // thus we should not start a sampler for it
        if(_tid > 0 && _info && _info->is_offset) return std::set<int>{};
        // if the thread state is disabled or completed, return
        if(_info && _info->index_data->internal_value == _tid &&
           get_thread_state() == ThreadState::Disabled)
            return std::set<int>{};

        (void) get_debug_sampling();  // make sure query in sampler does not allocate
        assert(_tid == threading::get_id());

        if(trait::runtime_enabled<backtrace_metrics>::get())
            backtrace_metrics::configure(_setup, _tid);

        // NOTE: signals need to be unblocked by calling function
        sampling::block_signals(*_signal_types);

        auto _verbose = std::min<int>(get_verbose() - 2, 2);
        if(get_debug_sampling()) _verbose = 2;

        OMNITRACE_DEBUG("Configuring sampler for thread %lu...\n", _tid);
        sampling::sampler_instances::construct("omnitrace", _tid, _verbose);

        _sampler->set_flags(SA_RESTART);
        _sampler->set_verbose(_verbose);

        if(_signal_types->count(get_realtime_signal()) > 0)
        {
            _sampler->configure(timer{ get_realtime_signal(), CLOCK_REALTIME,
                                       SIGEV_THREAD_ID, get_sampling_real_freq(),
                                       get_sampling_real_delay(), _tid,
                                       threading::get_sys_tid() });
        }

        if(_signal_types->count(get_cputime_signal()) > 0)
        {
            _sampler->configure(timer{ get_cputime_signal(), CLOCK_THREAD_CPUTIME_ID,
                                       SIGEV_THREAD_ID, get_sampling_cpu_freq(),
                                       get_sampling_cpu_delay(), _tid,
                                       threading::get_sys_tid() });
        }

        static_assert(tim::trait::buffer_size<sampling::sampler_t>::value > 0,
                      "Error! Zero buffer size");

        OMNITRACE_CONDITIONAL_THROW(
            _sampler->get_buffer_size() !=
                tim::trait::buffer_size<sampling::sampler_t>::value,
            "dynamic sampler has a buffer size different from static trait: %zu instead "
            "of %zu",
            _sampler->get_buffer_size(),
            tim::trait::buffer_size<sampling::sampler_t>::value);

        OMNITRACE_CONDITIONAL_THROW(
            _sampler->get_buffer_size() <= 0,
            "dynamic sampler requires a positive buffer size: %zu",
            _sampler->get_buffer_size());

        for(auto itr : *_signal_types)
        {
            const char* _type  = (itr == get_realtime_signal()) ? "wall" : "CPU";
            const auto* _timer = _sampler->get_timer(itr);
            if(_timer)
            {
                OMNITRACE_VERBOSE(
                    1,
                    "[SIG%i] Sampler for thread %lu will be triggered %.1fx per "
                    "second of %s-time (every %.3e milliseconds)...\n",
                    itr, _tid, _timer->get_frequency(units::sec), _type,
                    _timer->get_period(units::msec));
            }
        }

        *_running = true;
        sampling::get_sampler_init(_tid)->sample();
        start_duration_thread();
        _sampler->start();
    }
    else if(!_setup && _sampler && _is_running)
    {
        OMNITRACE_DEBUG("Destroying sampler for thread %lu...\n", _tid);
        *_running = false;

        if(_tid == threading::get_id() && !_signal_types->empty())
        {
            sampling::block_signals(*_signal_types);
        }

        get_duration_cv().notify_one();
        if(_tid == 0)
        {
            // this propagates to all threads
            _sampler->ignore(*_signal_types);
            for(int64_t i = 1; i < OMNITRACE_MAX_THREADS; ++i)
            {
                if(sampling::get_sampler(i))
                {
                    sampling::get_sampler(i)->stop();
                    sampling::get_sampler(i)->reset();
                    *get_sampler_running(i) = false;
                }
            }

            if(get_duration_thread())
            {
                get_duration_thread()->join();
                get_duration_thread().reset();
            }
        }

        _sampler->stop();

        if(trait::runtime_enabled<backtrace_metrics>::get())
            backtrace_metrics::configure(_setup, _tid);

        OMNITRACE_DEBUG("Sampler destroyed for thread %lu\n", _tid);
    }

    return (_signal_types) ? *_signal_types : std::set<int>{};
}

void
post_process_perfetto(int64_t _tid, const bundle_t* _init,
                      const std::vector<bundle_t*>& _data);
void
post_process_timemory(int64_t _tid, const bundle_t* _init,
                      const std::vector<bundle_t*>& _data);
}  // namespace

unique_ptr_t<std::set<int>>&
get_signal_types(int64_t _tid)
{
    static auto& _v = signal_type_instances::instances();
    signal_type_instances::construct(omnitrace::get_sampling_signals(_tid));
    return _v.at(_tid);
}

std::set<int>
setup()
{
    if(!get_use_sampling()) return std::set<int>{};
    return configure(true);
}

std::set<int>
shutdown()
{
    return configure(false);
}

void
block_signals(std::set<int> _signals)
{
    if(_signals.empty()) _signals = *get_signal_types(threading::get_id());
    if(_signals.empty())
    {
        OMNITRACE_PRINT("No signals to block...\n");
        return;
    }

    OMNITRACE_DEBUG("Blocking signals [%s] on thread #%lu...\n",
                    get_signal_names(_signals).c_str(), threading::get_id());

    sigset_t _v = get_signal_set(_signals);
    thread_sigmask(SIG_BLOCK, &_v, nullptr);
}

void
unblock_signals(std::set<int> _signals)
{
    if(_signals.empty()) _signals = *get_signal_types(threading::get_id());
    if(_signals.empty())
    {
        OMNITRACE_PRINT("No signals to unblock...\n");
        return;
    }

    OMNITRACE_DEBUG("Unblocking signals [%s] on thread #%lu...\n",
                    get_signal_names(_signals).c_str(), threading::get_id());

    sigset_t _v = get_signal_set(_signals);
    thread_sigmask(SIG_UNBLOCK, &_v, nullptr);
}

void
post_process()
{
    omnitrace::component::backtrace::stop();
    OMNITRACE_VERBOSE(2 || get_debug_sampling(), "Stopping backtrace metrics...\n");

    for(size_t i = 0; i < max_supported_threads; ++i)
        backtrace_metrics::configure(false, i);

    size_t _total_data    = 0;
    size_t _total_threads = 0;
    for(size_t i = 0; i < max_supported_threads; ++i)
    {
        auto& _sampler = get_sampler(i);

        if(!_sampler)
        {
            // this should be relatively common
            OMNITRACE_CONDITIONAL_PRINT(
                get_debug() && get_verbose() >= 2,
                "Post-processing sampling entries for thread %lu skipped (no sampler)\n",
                i);
            continue;
        }

        auto* _init = get_sampler_init(i).get();

        if(!_init)
        {
            // this is not common
            OMNITRACE_PRINT("Post-processing sampling entries for thread %lu skipped "
                            "(not initialized)\n",
                            i);
            continue;
        }

        const auto& _thread_info = thread_info::get(i, InternalTID);

        OMNITRACE_VERBOSE(3 || get_debug_sampling(),
                          "Getting sampler data for thread %lu...\n", i);

        _sampler->stop();
        auto& _raw_data = _sampler->get_data();

        OMNITRACE_VERBOSE(2 || get_debug_sampling(),
                          "Sampler data for thread %lu has %zu initial entries...\n", i,
                          _raw_data.size());

        OMNITRACE_CI_THROW(
            _sampler->get_sample_count() != _raw_data.size(),
            "Error! sampler recorded %zu samples but %zu samples were returned\n",
            _sampler->get_sample_count(), _raw_data.size());
        // single sample that is useless (backtrace to unblocking signals)
        if(_raw_data.size() == 1 && _raw_data.front().size() <= 1) _raw_data.clear();

        std::vector<sampling::bundle_t*> _data{};
        for(auto& itr : _raw_data)
        {
            _data.reserve(_data.size() + itr.size());
            auto* _bt = itr.get<backtrace>();
            auto* _ts = itr.get<backtrace_timestamp>();
            if(!_bt || !_ts) continue;
            if(_bt->empty()) continue;
            if(!_thread_info->is_valid_time(_ts->get_timestamp())) continue;
            _data.emplace_back(&itr);
        }

        if(_data.empty())
        {
            OMNITRACE_VERBOSE(
                3 || get_debug_sampling(),
                "Sampler data for thread %lu has %zu valid entries... (skipped)\n", i,
                _raw_data.size());
            continue;
        }

        OMNITRACE_VERBOSE(2 || get_debug_sampling(),
                          "Sampler data for thread %lu has %zu valid entries...\n", i,
                          _raw_data.size());

        _total_data += _raw_data.size();
        _total_threads += 1;

        if(get_use_perfetto()) post_process_perfetto(i, _init, _data);
        if(get_use_timemory()) post_process_timemory(i, _init, _data);
    }

    OMNITRACE_VERBOSE(3 || get_debug_sampling(), "Destroying samplers...\n");

    for(size_t i = 0; i < max_supported_threads; ++i)
    {
        get_sampler(i).reset();
    }

    OMNITRACE_VERBOSE(1 || get_debug_sampling(),
                      "Collected %zu samples from %zu threads...\n", _total_data,
                      _total_threads);
}

namespace
{
void
post_process_perfetto(int64_t _tid, const bundle_t* _init,
                      const std::vector<bundle_t*>& _data)
{
    if(trait::runtime_enabled<backtrace_metrics>::get())
    {
        OMNITRACE_VERBOSE(3 || get_debug_sampling(),
                          "[%li] Post-processing metrics for perfetto...\n", _tid);
        backtrace_metrics::init_perfetto(_tid);
        for(const auto& itr : _data)
        {
            const auto* _bt_metrics = itr->get<backtrace_metrics>();
            const auto* _bt_time    = itr->get<backtrace_timestamp>();
            if(!_bt_metrics || !_bt_time) continue;
            if(_bt_time->get_tid() != _tid) continue;
            _bt_metrics->post_process_perfetto(_tid, _bt_time->get_timestamp());
        }

        backtrace_metrics::fini_perfetto(_tid);
    }

    auto _process_perfetto = [_tid,
                              _init](const std::vector<sampling::bundle_t*>& _data) {
        thread_info::init(true);
        OMNITRACE_VERBOSE(3 || get_debug_sampling(),
                          "[%li] Post-processing backtraces for perfetto...\n", _tid);

        const auto& _thread_info = thread_info::get(_tid, InternalTID);
        OMNITRACE_CI_THROW(!_thread_info, "No valid thread info for tid=%li\n", _tid);

        if(!_thread_info) return;

        uint64_t _beg_ns  = _thread_info->get_start();
        uint64_t _end_ns  = _thread_info->get_stop();
        uint64_t _last_ts = std::max<uint64_t>(
            _init->get<backtrace_timestamp>()->get_timestamp(), _beg_ns);

        tracing::push_perfetto_ts(category::sampling{}, "samples [omnitrace]", _beg_ns,
                                  "begin_ns", _beg_ns);

        for(const auto& itr : _data)
        {
            const auto* _bt_ts = itr->get<backtrace_timestamp>();
            const auto* _bt_cs = itr->get<backtrace>();

            if(!_bt_ts || !_bt_cs) continue;
            if(_bt_ts->get_tid() != _tid) continue;

            static std::set<std::string> _static_strings{};
            for(const auto& itr : backtrace::filter_and_patch(_bt_cs->get()))
            {
                const auto* _name = _static_strings.emplace(itr).first->c_str();
                uint64_t    _beg  = _last_ts;
                uint64_t    _end  = _bt_ts->get_timestamp();
                if(!_thread_info->is_valid_lifetime({ _beg, _end })) continue;

                tracing::push_perfetto_ts(category::sampling{}, _name, _beg, "begin_ns",
                                          _beg);
                tracing::pop_perfetto_ts(category::sampling{}, _name, _end, "end_ns",
                                         _end);
            }
            _last_ts = _bt_ts->get_timestamp();
        }

        tracing::pop_perfetto_ts(category::sampling{}, "samples [omnitrace]", _end_ns,
                                 "end_ns", _end_ns);
    };

    auto _processing_thread        = threading::get_tid();
    auto _process_perfetto_wrapper = [&]() {
        if(threading::get_tid() != _processing_thread)
            threading::set_thread_name(TIMEMORY_JOIN(" ", "Thread", _tid, "(S)").c_str());

        try
        {
            _process_perfetto(_data);
        } catch(std::runtime_error& _e)
        {
            OMNITRACE_PRINT("[sampling][post_process_perfetto] Exception: %s\n",
                            _e.what());
            OMNITRACE_CI_ABORT(true, "[sampling][post_process_perfetto] Exception: %s\n",
                               _e.what());
        }
    };

    OMNITRACE_SCOPED_SAMPLING_ON_CHILD_THREADS(false);
    std::thread{ _process_perfetto_wrapper }.join();
}

void
post_process_timemory(int64_t _tid, const bundle_t* _init,
                      const std::vector<bundle_t*>& _data)
{
    auto _depth_sum = std::map<int64_t, std::map<int64_t, int64_t>>{};

    OMNITRACE_VERBOSE(3 || get_debug_sampling(),
                      "[%li] Post-processing data for timemory...\n", _tid);

    const auto* _last = _init;
    for(const auto& itr : _data)
    {
        using bundle_t = tim::lightweight_tuple<comp::trip_count, sampling_wall_clock,
                                                sampling_cpu_clock, hw_counters>;

        auto* _bt_data    = itr->get<backtrace>();
        auto* _bt_time    = itr->get<backtrace_timestamp>();
        auto* _bt_metrics = itr->get<backtrace_metrics>();

        if(!_bt_data || !_bt_time || !_bt_metrics) continue;

        double _elapsed_wc = (_bt_time->get_timestamp() -
                              _last->get<backtrace_timestamp>()->get_timestamp());
        double _elapsed_cc = (_bt_metrics->get_cpu_timestamp() -
                              _last->get<backtrace_metrics>()->get_cpu_timestamp());

        std::vector<bundle_t> _tc{};
        _tc.reserve(_bt_data->size());

        // generate the instances of the tuple of components and start them
        for(const auto& itr : backtrace::filter_and_patch(_bt_data->get()))
        {
            _tc.emplace_back(tim::string_view_t{ itr });
            _tc.back().push(_bt_time->get_tid());
            _tc.back().start();
        }

        // stop the instances and update the values as needed
        for(size_t i = 0; i < _tc.size(); ++i)
        {
            auto&  itr    = _tc.at(_tc.size() - i - 1);
            size_t _depth = 0;
            _depth_sum[_bt_time->get_tid()][_depth] += 1;
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
                auto _hw_cnt_vals = _bt_metrics->get_hw_counters();
                if(_last && _bt_metrics->get_hw_counters().size() ==
                                _last->get<backtrace_metrics>()->get_hw_counters().size())
                {
                    for(size_t k = 0; k < _bt_metrics->get_hw_counters().size(); ++k)
                    {
                        if(_last->get<backtrace_metrics>()->get_hw_counters()[k] >
                           _hw_cnt_vals[k])
                            _hw_cnt_vals[k] -=
                                _last->get<backtrace_metrics>()->get_hw_counters()[k];
                    }
                }
                auto* _hw_counter = itr.get<hw_counters>();
                if(_hw_counter)
                {
                    _hw_counter->set_value(_hw_cnt_vals);
                    _hw_counter->set_accum(_hw_cnt_vals);
                }
            }
            itr.pop();
        }
        _last = itr;
    }

    for(auto&& itr : _data)
    {
        using bundle_t =
            tim::lightweight_tuple<sampling_percent, quirk::config<quirk::tree_scope>>;

        auto* _bt_data = itr->get<backtrace>();
        auto* _bt_time = itr->get<backtrace_timestamp>();

        if(!_bt_time || !_bt_data) continue;
        if(_depth_sum.find(_bt_time->get_tid()) == _depth_sum.end()) continue;

        std::vector<bundle_t> _tc{};
        _tc.reserve(_bt_data->size());

        // generate the instances of the tuple of components and start them
        for(const auto& itr : backtrace::filter_and_patch(_bt_data->get()))
        {
            _tc.emplace_back(tim::string_view_t{ itr });
            _tc.back().push(_bt_time->get_tid());
            _tc.back().start();
        }

        // stop the instances and update the values as needed
        for(size_t i = 0; i < _tc.size(); ++i)
        {
            auto&  itr    = _tc.at(_tc.size() - i - 1);
            size_t _depth = 0;
            double _value = (1.0 / _depth_sum[_bt_time->get_tid()][_depth]) * 100.0;
            itr.store(std::plus<double>{}, _value);
            itr.stop();
            itr.pop();
        }
    }
}

struct sampling_initialization
{
    static void preinit()
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
};
}  // namespace
}  // namespace sampling
}  // namespace omnitrace

TIMEMORY_INVOKE_PREINIT(omnitrace::sampling::sampling_initialization)
