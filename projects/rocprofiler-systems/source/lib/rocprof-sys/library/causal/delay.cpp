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

#include "library/causal/delay.hpp"
#include "core/state.hpp"
#include "core/utility.hpp"
#include "library/causal/components/causal_gotcha.hpp"
#include "library/causal/experiment.hpp"
#include "library/causal/sampling.hpp"
#include "library/runtime.hpp"
#include "library/thread_data.hpp"
#include "library/thread_info.hpp"
#include "library/tracing.hpp"

#include <timemory/components/macros.hpp>
#include <timemory/mpl/concepts.hpp>
#include <timemory/mpl/types.hpp>
#include <timemory/process/threading.hpp>

#include <atomic>
#include <chrono>
#include <random>

namespace rocprofsys
{
namespace causal
{
namespace
{
auto&
get_delay_data()
{
    using thread_data_t = thread_data<identity<int64_t>, delay>;
    static auto& _v     = thread_data_t::construct(
        construct_on_init{}, []() { return delay::get_global().load(); });
    return _v;
}

int64_t
compute_sleep_for_overhead()
{
    using random_engine_t = std::mt19937_64;
    auto   _engine        = random_engine_t{ std::random_device{}() };
    auto   _dist          = std::uniform_int_distribution<int64_t>{ 0, 5 };
    size_t _ntot          = 250;
    size_t _nwarm         = 50;
    auto   _stats         = tim::statistics<double>{};
    for(size_t i = 0; i < _ntot; ++i)
    {
        auto    _val = _dist(_engine);
        int64_t _beg = tracing::now();
        std::this_thread::sleep_for(std::chrono::nanoseconds{ _val });
        int64_t _end = tracing::now();
        if(i < _nwarm) continue;
        auto _diff = (_end - _beg);
        ROCPROFSYS_CONDITIONAL_THROW(
            _diff < _val, "Error! sleep_for(%zu) [nanoseconds] >= %zu", _val, _diff);
        _stats += (_diff - _val);
    }

    ROCPROFSYS_BASIC_VERBOSE(2,
                             "[causal] overhead of std::this_thread::sleep_for(...) "
                             "invocation = %6.3f usec +/- %e\n",
                             _stats.get_mean() / units::usec,
                             _stats.get_stddev() / units::usec);

    tim::manager::instance()->add_metadata([_stats](auto& ar) {
        ar(tim::cereal::make_nvp("causal thread sleep overhead [nsec]", _stats));
    });

    (void) get_delay_data();

    return _stats.get_mean();
}

int64_t sleep_for_overhead = 0;
}  // namespace

void
delay::setup()
{
    static std::once_flag _once{};
    std::call_once(_once, []() { sleep_for_overhead = compute_sleep_for_overhead(); });
}

void
delay::process()
{
    if(!is_local_available()) return;

    if(causal::experiment::is_active())
    {
        if(get_global() < get_local())
        {
            get_global() += (get_local() - get_global());
        }
        else if(get_global() > get_local())
        {
            ::rocprofsys::causal::sampling::pause();
            auto _beg = tracing::now();
            std::this_thread::sleep_for(
                std::chrono::nanoseconds{ get_global() - get_local() });
            get_local() += (tracing::now() - _beg);
            ::rocprofsys::causal::sampling::resume();
        }
    }
    else
    {
        get_local() = get_global();
    }
}

void
delay::credit()
{
    if(!is_local_available()) return;

    auto _diff = get_global() - get_local();
    if(_diff > 0)
    {
        get_local() += _diff;
    }
}

void
delay::preblock()
{
    if(!is_local_available()) return;

    auto _diff = get_global() - get_local();
    if(_diff > 0)
    {
        get_local() += _diff;
    }
}

void
delay::postblock(int64_t _preblock_global_delay_value)
{
    if(!is_local_available()) return;
    get_local() += (get_global() - _preblock_global_delay_value);
}

int64_t
delay::sync()
{
    auto _v = get_global().load(std::memory_order_seq_cst);
    if(get_delay_data()) get_delay_data()->fill(_v);
    return _v;
}

std::atomic<int64_t>&
delay::get_global()
{
    static auto _v = std::atomic<int64_t>{ 0 };
    return _v;
}

static void
thr_init()
{
    static thread_local auto _thr_init = []() {
        using thread_data_t = thread_data<identity<int64_t>, delay>;
        thread_data_t::construct(construct_on_thread{ threading::get_id() },
                                 delay::get_global().load());
        return true;
    }();
    (void) _thr_init;  // To make compiler happy.
}

bool
delay::is_local_available()
{
    thr_init();
    auto& _data = get_delay_data();
    return _data != nullptr;
}

int64_t&
delay::get_local(int64_t _tid)
{
    thr_init();
    auto& _data = get_delay_data();
    if(_data == nullptr)
    {
        throw std::runtime_error("No data: get_delay_data() returned nullptr");
    }
    return _data->at(_tid);
}

uint64_t
delay::compute_total_delay(uint64_t _baseline)
{
    return get_global().load() - _baseline;
}
}  // namespace causal
}  // namespace rocprofsys
