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

#include "library/causal/components/backtrace.hpp"
#include "core/concepts.hpp"
#include "core/config.hpp"
#include "core/debug.hpp"
#include "core/state.hpp"
#include "core/utility.hpp"
#include "library/causal/data.hpp"
#include "library/causal/delay.hpp"
#include "library/causal/experiment.hpp"
#include "library/runtime.hpp"
#include "library/thread_data.hpp"
#include "library/thread_info.hpp"
#include "library/tracing.hpp"

#include <timemory/components/timing/backends.hpp>
#include <timemory/components/timing/wall_clock.hpp>
#include <timemory/mpl/concepts.hpp>
#include <timemory/mpl/types.hpp>
#include <timemory/process/threading.hpp>
#include <timemory/units.hpp>
#include <timemory/utility/backtrace.hpp>

#include <atomic>
#include <ctime>
#include <type_traits>

namespace omnitrace
{
namespace causal
{
namespace component
{
namespace
{
using ::tim::backtrace::get_unw_signal_frame_stack_raw;

auto&
get_delay_statistics()
{
    using thread_data_t =
        thread_data<identity<tim::statistics<int64_t>>, category::sampling>;

    static_assert(
        use_placement_new_when_generating_unique_ptr<thread_data_t>::value,
        "delay statistics thread data should use placement new to allocate unique_ptr");

    static auto& _v = thread_data_t::instance(construct_on_init{});
    return _v;
}

auto&
get_in_use()
{
    using thread_data_t = thread_data<identity<bool>, category::sampling>;

    static_assert(
        use_placement_new_when_generating_unique_ptr<thread_data_t>::value,
        "sampling is_use thread data should use placement new to allocate unique_ptr");

    static auto& _v = thread_data_t::instance(construct_on_init{});
    return _v;
}

struct scoped_in_use
{
    scoped_in_use(int64_t _tid = utility::get_thread_index())
    : value{ get_in_use()->at(_tid) }
    {
        value = true;
    }
    ~scoped_in_use() { value = false; }

    bool& value;
};

auto
is_in_use(int64_t _tid = threading::get_id())
{
    return get_in_use()->at(_tid);
}

}  // namespace

void
backtrace::start()
{
    // do not delete these lines. The thread data needs to be allocated
    // before it is called in sampler or else a deadlock will occur when
    // the sample interrupts a malloc call
    (void) get_delay_statistics();
    (void) get_in_use();
}

void
backtrace::stop()
{}

void
backtrace::sample(int _sig)
{
    constexpr size_t  depth        = ::omnitrace::causal::unwind_depth;
    constexpr int64_t ignore_depth = ::omnitrace::causal::unwind_offset;

    // update the last sample for backtrace signal(s) even when in use
    static thread_local int64_t _last_sample = 0;

    if(is_in_use())
    {
        if(_sig == get_realtime_signal()) _last_sample = tracing::now();
        return;
    }
    scoped_in_use _in_use{};

    m_index = causal::experiment::get_index();
    m_stack = get_unw_signal_frame_stack_raw<depth, ignore_depth>();

    // the batch handler timer delivers a signal according to the thread CPU
    // clock, ensuring that setting the current selection and processing the
    // delays only happens when the thread is active
    if(_sig == get_cputime_signal())
    {
        if(!causal::experiment::is_active())
            causal::set_current_selection(m_stack);
        else
            causal::delay::process();
    }
    else if(_sig == get_realtime_signal())
    {
        auto  _this_sample = tracing::now();
        auto& _period_stat = get_delay_statistics()->at(threading::get_id());
        if(_last_sample > 0) _period_stat += (_this_sample - _last_sample);
        _last_sample = _this_sample;

        if(causal::experiment::is_active() && causal::experiment::is_selected(m_stack))
        {
            m_selected = true;
            causal::experiment::add_selected();
            // compute the delay time based on the rate of taking samples,
            // unless we have taken less than 10, in which case, we just
            // use the pre-computed value.
            auto _delay =
                (_period_stat.get_count() < 10)
                    ? causal::experiment::get_delay()
                    : (_period_stat.get_mean() * causal::experiment::get_delay_scaling());
            causal::delay::get_local() += _delay;
        }
    }
    else
    {
        OMNITRACE_THROW("unhandled signal %i\n", _sig);
    }
}

template <typename Tp>
Tp
backtrace::get_period(uint64_t _units)
{
    using cast_type = std::conditional_t<std::is_floating_point<Tp>::value, Tp, double>;

    double _realtime_freq =
        (get_use_sampling_realtime()) ? get_sampling_real_freq() : 0.0;
    double _cputime_freq = (get_use_sampling_cputime()) ? get_sampling_cpu_freq() : 0.0;

    auto    _freq        = std::max<double>(_realtime_freq, _cputime_freq);
    double  _period      = 1.0 / _freq;
    int64_t _period_nsec = static_cast<int64_t>(_period * units::sec) % units::sec;
    return static_cast<Tp>(_period_nsec) / static_cast<cast_type>(_units);
}

tim::statistics<int64_t>
backtrace::get_period_stats()
{
    scoped_in_use _in_use{};
    auto          _data = tim::statistics<int64_t>{};
    if(!get_delay_statistics()) return _data;
    for(size_t i = 0; i < get_delay_statistics()->size(); ++i)
    {
        scoped_in_use _thr_in_use{ static_cast<int64_t>(i) };
        const auto&   itr = get_delay_statistics()->at(i);
        if(itr.get_count() > 1) _data += itr;
    }
    return _data;
}

void
backtrace::reset_period_stats()
{
    scoped_in_use _in_use{};
    for(size_t i = 0; i < get_delay_statistics()->size(); ++i)
    {
        scoped_in_use _thr_in_use{ static_cast<int64_t>(i) };
        get_delay_statistics()->at(i).reset();
    }
}
}  // namespace component
}  // namespace causal
}  // namespace omnitrace

#define INSTANTIATE_BT_CAUSAL_PERIOD(TYPE)                                               \
    template TYPE omnitrace::causal::component::backtrace::get_period<TYPE>(uint64_t);

INSTANTIATE_BT_CAUSAL_PERIOD(float)
INSTANTIATE_BT_CAUSAL_PERIOD(double)
INSTANTIATE_BT_CAUSAL_PERIOD(int64_t)
INSTANTIATE_BT_CAUSAL_PERIOD(uint64_t)
