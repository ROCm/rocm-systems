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

#include "library/causal/sampling.hpp"
#include "core/common.hpp"
#include "core/concepts.hpp"
#include "core/config.hpp"
#include "core/debug.hpp"
#include "core/locking.hpp"
#include "core/state.hpp"
#include "core/utility.hpp"
#include "library/causal/components/backtrace.hpp"
#include "library/causal/data.hpp"
#include "library/ptl.hpp"
#include "library/runtime.hpp"
#include "library/sampling.hpp"
#include "library/thread_data.hpp"
#include "library/thread_info.hpp"

#include <timemory/macros.hpp>
#include <timemory/sampling/allocator.hpp>
#include <timemory/sampling/sampler.hpp>
#include <timemory/units.hpp>
#include <timemory/utility/backtrace.hpp>
#include <timemory/variadic.hpp>

#include <csignal>
#include <cstring>
#include <ctime>
#include <mutex>
#include <sstream>
#include <string>
#include <type_traits>

namespace omnitrace
{
namespace causal
{
namespace sampling
{
using ::tim::sampling::dynamic;
using ::tim::sampling::timer;

using causal_bundle_t =
    tim::lightweight_tuple<causal::component::sample_rate, causal::component::backtrace>;
using causal_sampler_t = tim::sampling::sampler<causal_bundle_t, dynamic>;
}  // namespace sampling
}  // namespace causal
}  // namespace omnitrace

OMNITRACE_DEFINE_CONCRETE_TRAIT(prevent_reentry, causal::sampling::causal_sampler_t,
                                std::true_type)

OMNITRACE_DEFINE_CONCRETE_TRAIT(provide_backtrace, causal::sampling::causal_sampler_t,
                                std::false_type)

OMNITRACE_DEFINE_CONCRETE_TRAIT(buffer_size, causal::sampling::causal_sampler_t,
                                TIMEMORY_ESC(std::integral_constant<size_t, 4096>))

namespace omnitrace
{
namespace causal
{
namespace sampling
{
namespace
{
using causal_sampler_allocator_t = typename causal_sampler_t::allocator_t;
using causal_sampler_bundle_t    = typename causal_sampler_t::bundle_type;
using causal_sampler_buffer_t = tim::data_storage::ring_buffer<causal_sampler_bundle_t>;

struct causal_sampling
{};

std::set<int>
configure(bool _setup, int64_t _tid = threading::get_id());

std::shared_ptr<causal_sampler_allocator_t>&
get_causal_sampler_allocator(bool _construct)
{
    static auto _v = std::shared_ptr<causal_sampler_allocator_t>{};
    if(!_v && _construct) _v = std::make_shared<causal_sampler_allocator_t>();
    return _v;
}

auto&
get_causal_sampler_signals()
{
    using thread_data_t = thread_data<identity<std::set<int>>, causal_sampling>;
    static auto& _v     = thread_data_t::instance(construct_on_init{});
    return _v;
}

auto&
get_causal_sampler_running()
{
    using thread_data_t = thread_data<identity<bool>, causal_sampling>;
    static auto& _v     = thread_data_t::instance(construct_on_init{});
    return _v;
}

auto&
get_causal_samplers()
{
    using thread_data_t =
        thread_data<identity<std::unique_ptr<causal_sampler_t>>, causal_sampling>;
    static auto& _v = thread_data_t::instance(construct_on_init{});
    return _v;
}

std::set<int>&
get_causal_sampler_signals(int64_t _tid)
{
    auto& _data = get_causal_sampler_signals();
    if(static_cast<size_t>(_tid) >= _data->size())
        _data->resize(_tid + 1, std::set<int>{});
    return _data->at(_tid);
}

bool&
get_causal_sampler_running(int64_t _tid)
{
    auto& _data = get_causal_sampler_running();
    if(static_cast<size_t>(_tid) >= _data->size()) _data->resize(_tid + 1, false);
    return _data->at(_tid);
}

auto&
get_causal_sampler(int64_t _tid)
{
    auto& _data = get_causal_samplers();
    if(static_cast<size_t>(_tid) >= _data->size()) _data->resize(_tid + 1);
    return _data->at(_tid);
}

void
causal_offload_buffer(int64_t, causal_sampler_buffer_t&& _buf)
{
    auto _data      = std::move(_buf);
    auto _processed = std::map<uint32_t, std::vector<uintptr_t>>{};
    while(!_data.is_empty())
    {
        auto _bundle = causal_sampler_bundle_t{};
        _data.read(&_bundle);
        auto* _bt_causal = _bundle.get<causal::component::backtrace>();
        if(_bt_causal)
        {
            for(auto&& itr : _bt_causal->get_stack())
            {
                if(itr > 0) _processed[_bt_causal->get_index()].emplace_back(itr);
            }
        }
    }
    _data.destroy();

    if(!_processed.empty())
    {
        static auto _mutex = locking::atomic_mutex{};
        auto        _lk    = locking::atomic_lock{ _mutex };
        for(const auto& itr : _processed)
            add_samples(itr.first, itr.second);
    }
}

std::set<int>
configure(bool _setup, int64_t _tid)
{
    const auto& _info         = thread_info::get(_tid, SequentTID);
    auto&       _causal       = get_causal_sampler(_tid);
    auto&       _running      = get_causal_sampler_running(_tid);
    auto&       _signal_types = get_causal_sampler_signals(_tid);

    OMNITRACE_CONDITIONAL_THROW(get_use_sampling(),
                                "Internal error! configuring causal profiling not "
                                "permitted when sampling is enabled");

    OMNITRACE_SCOPED_SAMPLING_ON_CHILD_THREADS(false);

    if(_setup && _signal_types.empty()) _signal_types = get_sampling_signals(_tid);

    if(_setup && !_causal && !_running && !_signal_types.empty())
    {
        auto _verbose = std::min<int>(get_verbose() - 2, 2);
        if(get_debug_sampling()) _verbose = 2;

        // if this thread has an offset ID, that means it was created internally
        // and is probably here bc it called a function which was instrumented.
        // thus we should not start a sampler for it
        if(_tid > 0 && _info && _info->is_offset) return std::set<int>{};
        // if the thread state is disabled or completed, return
        if(_info && _info->index_data->sequent_value == _tid &&
           get_thread_state() == ThreadState::Disabled)
            return std::set<int>{};

        (void) get_debug_sampling();  // make sure query in sampler does not allocate
        assert(_tid == threading::get_id());

        auto _causal_alloc = get_causal_sampler_allocator(true);
        _causal = std::make_unique<causal_sampler_t>(_causal_alloc, "omnitrace", _tid,
                                                     _verbose);

        TIMEMORY_REQUIRE(_causal) << "nullptr to causal profiling instance";

        _causal->set_flags(SA_RESTART);
        _causal->set_verbose(_verbose);
        _causal->set_offload(&causal_offload_buffer);

        _causal->configure(timer{ get_realtime_signal(), CLOCK_REALTIME, SIGEV_THREAD_ID,
                                  1000.0, 1.0e-6, _tid, threading::get_sys_tid() });

        _causal->configure(timer{ get_cputime_signal(), CLOCK_THREAD_CPUTIME_ID,
                                  SIGEV_THREAD_ID, 1000.0, 1.0e-6, _tid,
                                  threading::get_sys_tid() });

        _running = true;
        if(_tid == 0) causal::component::backtrace::start();
        _causal->start();
    }
    else if(!_setup && _causal && _running)
    {
        OMNITRACE_DEBUG("Destroying causal sampler for thread %lu...\n", _tid);
        _running = false;

        if(_tid == threading::get_id() && !_signal_types.empty())
            block_signals(_signal_types);

        if(_tid == 0)
        {
            block_samples();

            // this propagates to all threads
            _causal->ignore(_signal_types);

            for(int64_t i = 1; i < OMNITRACE_MAX_THREADS; ++i)
            {
                if(get_causal_sampler(i))
                {
                    get_causal_sampler(i)->stop();
                    get_causal_sampler(i)->reset();
                }
            }
        }

        _causal->stop();
        _causal->reset();

        OMNITRACE_DEBUG("Causal sampler destroyed for thread %lu\n", _tid);
    }

    return _signal_types;
}

void
post_process_causal(int64_t _tid, const std::vector<causal_bundle_t>& _data);
}  // namespace

std::set<int>
get_signal_types(int64_t _tid)
{
    return (get_causal_sampler_signals()) ? get_causal_sampler_signals(_tid)
                                          : std::set<int>{};
}

std::set<int>
setup()
{
    if(!get_use_causal()) return std::set<int>{};
    return configure(true);
}

std::set<int>
shutdown()
{
    auto _v = configure(false);
    return _v;
}

void
block_samples()
{
    trait::runtime_enabled<causal_sampler_t>::set(false);
    trait::runtime_enabled<causal::component::backtrace>::set(false);
}

void
unblock_samples()
{
    trait::runtime_enabled<causal::component::backtrace>::set(true);
    trait::runtime_enabled<causal_sampler_t>::set(true);
}

void
block_backtrace_samples()
{
    trait::runtime_enabled<causal::component::backtrace>::set(scope::thread_scope{},
                                                              false);
}

void
unblock_backtrace_samples()
{
    trait::runtime_enabled<causal::component::backtrace>::set(scope::thread_scope{},
                                                              true);
}

void
block_signals(std::set<int> _signals)
{
    if(_signals.empty()) _signals = get_signal_types(threading::get_id());
    if(_signals.empty()) return;

    ::omnitrace::sampling::block_signals(_signals);
}

void
unblock_signals(std::set<int> _signals)
{
    if(_signals.empty()) _signals = get_signal_types(threading::get_id());
    if(_signals.empty()) return;

    ::omnitrace::sampling::unblock_signals(_signals);
}

void
post_process()
{
    OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);

    OMNITRACE_VERBOSE(2 || get_debug_sampling(),
                      "Stopping causal sampling components...\n");

    block_samples();

    for(size_t i = 0; i < max_supported_threads; ++i)
    {
        auto& _causal = get_causal_sampler(i);
        if(_causal) _causal->stop();
    }

    configure(false, 0);

    for(size_t i = 0; i < max_supported_threads; ++i)
    {
        auto& _causal = get_causal_sampler(i);
        auto  _causal_data =
            (_causal) ? _causal->get_data() : std::vector<sampling::causal_bundle_t>{};

        if(!_causal_data.empty()) post_process_causal(i, _causal_data);
    }

    for(size_t i = 0; i < max_supported_threads; ++i)
    {
        get_causal_sampler(i).reset();
    }

    if(get_causal_sampler_allocator(false))
    {
        get_causal_sampler_allocator(false).reset();
    }
}

namespace
{
void
post_process_causal(int64_t, const std::vector<causal_bundle_t>& _data)
{
    for(const auto& itr : _data)
    {
        const auto* _bt_causal = itr.get<causal::component::backtrace>();
        for(auto&& ditr : _bt_causal->get_stack())
        {
            if(ditr > 0) add_sample(_bt_causal->get_index(), ditr);
        }
    }
}
}  // namespace
}  // namespace sampling
}  // namespace causal
}  // namespace omnitrace
