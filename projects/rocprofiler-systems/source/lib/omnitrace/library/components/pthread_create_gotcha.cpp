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

#include "library/components/pthread_create_gotcha.hpp"
#include "library/components/category_region.hpp"
#include "library/components/roctracer.hpp"
#include "library/config.hpp"
#include "library/debug.hpp"
#include "library/runtime.hpp"
#include "library/sampling.hpp"
#include "library/state.hpp"
#include "library/thread_data.hpp"
#include "library/thread_info.hpp"
#include "library/utility.hpp"

#include <timemory/backends/threading.hpp>
#include <timemory/components/macros.hpp>
#include <timemory/components/timing/wall_clock.hpp>
#include <timemory/sampling/allocator.hpp>
#include <timemory/utility/types.hpp>

#include <ostream>
#include <pthread.h>

namespace omnitrace
{
namespace sampling
{
std::set<int>
setup();
std::set<int>
shutdown();
}  // namespace sampling

namespace component
{
using bundle_t          = tim::lightweight_tuple<comp::wall_clock, comp::roctracer_data>;
using category_region_t = tim::lightweight_tuple<category_region<category::pthread>>;

namespace
{
auto* is_shutdown   = new bool{ false };  // intentional data leak
auto* bundles       = new std::map<int64_t, std::shared_ptr<bundle_t>>{};
auto* bundles_mutex = new std::mutex{};
auto  bundles_dtor  = scope::destructor{ []() {
    pthread_create_gotcha::shutdown();
    delete bundles;
    delete bundles_mutex;
    bundles         = nullptr;
    bundles_mutex   = nullptr;
} };

template <typename... Args>
inline void
start_bundle(bundle_t& _bundle, Args&&... _args)
{
    if(!get_use_timemory() && !get_use_perfetto()) return;
    OMNITRACE_BASIC_VERBOSE_F(3, "starting bundle '%s'...\n", _bundle.key().c_str());
    if constexpr(sizeof...(Args) > 0)
    {
        const char* _name = nullptr;
        if(tim::get_hash_identifier(_bundle.hash(), _name) && _name != nullptr)
        {
            category_region_t{}.audit(quirk::config<quirk::perfetto>{},
                                      std::string_view{ _name }, _args...);
        }
    }
    else
    {
        tim::consume_parameters(_args...);
    }
    if(get_use_timemory())
    {
        _bundle.push();
        _bundle.start();
    }
}

template <typename... Args>
inline void
stop_bundle(bundle_t& _bundle, int64_t _tid, Args&&... _args)
{
    if(!get_use_timemory() && !get_use_perfetto()) return;
    OMNITRACE_BASIC_VERBOSE_F(3, "stopping bundle '%s' in thread %li...\n",
                              _bundle.key().c_str(), _tid);
    if(get_use_timemory())
    {
        auto _wc = *_bundle.get<comp::wall_clock>();
        _wc.stop();
        // update roctracer_data
        _bundle.store(std::plus<double>{}, _wc.get() * _wc.unit());
        // stop all
        _bundle.stop();
        // exclude popping wall-clock
        _bundle.pop(_tid);
    }
    if constexpr(sizeof...(Args) > 0)
    {
        const char* _name = nullptr;
        if(tim::get_hash_identifier(_bundle.hash(), _name) && _name != nullptr)
        {
            category_region_t{}.audit(quirk::config<quirk::perfetto>{},
                                      std::string_view{ _name }, _args...);
        }
    }
    else
    {
        tim::consume_parameters(_args...);
    }
}
}  // namespace

//--------------------------------------------------------------------------------------//

pthread_create_gotcha::wrapper::wrapper(routine_t _routine, void* _arg,
                                        bool _enable_sampling, int64_t _parent,
                                        promise_t* _p)
: m_enable_sampling{ _enable_sampling }
, m_parent_tid{ _parent }
, m_routine{ _routine }
, m_arg{ _arg }
, m_promise{ _p }
{}

void*
pthread_create_gotcha::wrapper::operator()() const
{
    using thread_bundle_data_t = thread_data<thread_bundle_t>;

    if(is_shutdown && *is_shutdown)
    {
        if(m_promise) m_promise->set_value();
        // execute the original function
        return m_routine(m_arg);
    }

    push_thread_state(ThreadState::Internal);

    int64_t _tid         = -1;
    void*   _ret         = nullptr;
    auto    _is_sampling = false;
    auto    _bundle      = std::shared_ptr<bundle_t>{};
    auto    _signals     = std::set<int>{};
    auto    _coverage    = (get_mode() == Mode::Coverage);
    // const auto& _parent_info = thread_info::get(m_parent_tid, LookupTID);
    auto _dtor = [&]() {
        set_thread_state(ThreadState::Internal);
        if(_is_sampling)
        {
            sampling::block_signals(_signals);
            sampling::shutdown();
        }

        if(_tid >= 0)
        {
            auto _active = (get_state() == ::omnitrace::State::Active &&
                            bundles != nullptr && bundles_mutex != nullptr);
            if(!_active) return;
            thread_info::set_stop(comp::wall_clock::record());
            auto& _thr_bundle = thread_bundle_data_t::instance();
            if(_thr_bundle && _thr_bundle->get<comp::wall_clock>() &&
               _thr_bundle->get<comp::wall_clock>()->get_is_running())
                _thr_bundle->stop();
            if(_bundle) stop_bundle(*_bundle, _tid);
            pthread_create_gotcha::shutdown(_tid);
        }
    };

    auto _active = (get_state() == ::omnitrace::State::Active && bundles != nullptr &&
                    bundles_mutex != nullptr);

    if(_active && !_coverage)
    {
        const auto& _tid_index = thread_info::init();
        _tid                   = _tid_index->index_data->internal_value;
        threading::set_thread_name(TIMEMORY_JOIN(" ", "Thread", _tid).c_str());
        if(!thread_bundle_data_t::instances().at(_tid))
        {
            thread_data<thread_bundle_t>::construct(
                TIMEMORY_JOIN('/', "omnitrace/process", process::get_id(), "thread",
                              _tid),
                quirk::config<quirk::auto_start>{});
            thread_bundle_data_t::instances().at(_tid)->start();
        }
        if(bundles && bundles_mutex)
        {
            std::unique_lock<std::mutex> _lk{ *bundles_mutex };
            _bundle = bundles->emplace(_tid, std::make_shared<bundle_t>("start_thread"))
                          .first->second;
        }
        if(_bundle) start_bundle(*_bundle);
        get_cpu_cid_stack(_tid, m_parent_tid);
        if(m_enable_sampling)
        {
            _is_sampling = true;
            OMNITRACE_SCOPED_SAMPLING_ON_CHILD_THREADS(false);
            _signals = sampling::setup();
            sampling::unblock_signals();
        }
    }
    else
    {
        thread_info::init(true);
    }

    // notify the wrapper that all internal work is completed
    if(m_promise) m_promise->set_value();

    // Internal -> Enabled
    pop_thread_state();

    push_thread_state(ThreadState::Enabled);

    // execute the original function
    _ret = m_routine(m_arg);

    pop_thread_state();

    // execute the destructor actions
    _dtor();

    set_thread_state(ThreadState::Completed);

    return _ret;
}

void*
pthread_create_gotcha::wrapper::wrap(void* _arg)
{
    if(_arg == nullptr) return nullptr;

    // convert the argument
    wrapper* _wrapper = static_cast<wrapper*>(_arg);

    // execute the original function
    return (*_wrapper)();
}

void
pthread_create_gotcha::configure()
{
    pthread_create_gotcha_t::get_initializer() = []() {
        if(!tim::settings::enabled()) return;
        pthread_create_gotcha_t::template configure<
            0, int, pthread_t*, const pthread_attr_t*, void* (*) (void*), void*>(
            "pthread_create");
    };
}

void
pthread_create_gotcha::shutdown()
{
    if(is_shutdown)
    {
        if(*is_shutdown) return;
        *is_shutdown = true;
    }

    if(!bundles_mutex || !bundles) return;

    std::unique_lock<std::mutex> _lk{ *bundles_mutex };
    unsigned long                _ndangling = 0;
    for(auto itr : *bundles)
    {
        if(itr.second)
        {
            stop_bundle(*itr.second, itr.first);
            ++_ndangling;
        }
        itr.second.reset();
    }

    bundles->clear();

    OMNITRACE_BASIC_VERBOSE(
        2 && _ndangling > 0,
        "[pthread_create_gotcha::shutdown] cleaned up %lu dangling bundles\n",
        _ndangling);
}

void
pthread_create_gotcha::shutdown(int64_t _tid)
{
    if(is_shutdown && *is_shutdown) return;

    if(!bundles_mutex || !bundles) return;

    std::unique_lock<std::mutex> _lk{ *bundles_mutex };
    auto                         itr = bundles->find(_tid);
    if(itr != bundles->end())
    {
        if(itr->second) stop_bundle(*itr->second, itr->first);
        itr->second.reset();
        bundles->erase(itr);
    }
}

void
pthread_create_gotcha::set_data(wrappee_t _v)
{
    m_wrappee = _v;
}

// pthread_create
int
pthread_create_gotcha::operator()(pthread_t* thread, const pthread_attr_t* attr,
                                  void* (*start_routine)(void*), void*     arg) const
{
    auto _disabled = (get_thread_state() == ThreadState::Disabled);
    auto _enabled  = (get_thread_state() == ThreadState::Enabled);
    auto _bundle   = std::optional<bundle_t>{};

    OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);

    auto        _active       = (get_state() == ::omnitrace::State::Active && !_disabled);
    auto        _coverage     = (get_mode() == Mode::Coverage);
    auto        _use_sampling = get_use_sampling();
    auto        _sample_child = sampling_enabled_on_child_threads();
    auto        _tid          = utility::get_thread_index();
    auto        _use_bundle   = (_active && !_coverage);
    const auto& _info = thread_info::init(!_active || !_sample_child || _disabled);
    auto        _enable_sampling =
        (!_disabled && _enabled && _sample_child && _use_sampling && !_info->is_offset);

    if(_active && !_disabled && !_info->is_offset)
    {
        OMNITRACE_VERBOSE(1, "Creating new thread on PID %i (rank: %i), TID %li\n",
                          process::get_id(), dmp::rank(), _tid);
    }

    // ensure that cpu cid stack exists on the parent thread if active
    if(_active && !_coverage) get_cpu_cid_stack();

    set_thread_state(ThreadState::Disabled);
    auto  _blocked = get_sampling_signals();
    auto  _promise = std::promise<void>{};
    auto  _fut     = _promise.get_future();
    auto* _wrap    = new wrapper(start_routine, arg, _enable_sampling, _tid, &_promise);
    set_thread_state(ThreadState::Internal);

    // block the signals in entire process
    if(_enable_sampling && !_blocked.empty())
    {
        OMNITRACE_DEBUG("blocking signals...\n");
        tim::sampling::block_signals(_blocked, tim::sampling::sigmask_scope::process);
    }

    if(_use_bundle)
    {
        _bundle = bundle_t{ "pthread_create" };
        start_bundle(*_bundle, audit::incoming{}, thread, attr, start_routine, arg);
    }

    // create the thread
    auto _ret = (*m_wrappee)(thread, attr, &wrapper::wrap, static_cast<void*>(_wrap));

    // wait for thread to set promise
    OMNITRACE_DEBUG("waiting for child to signal it is setup...\n");
    _fut.wait();

    if(_use_bundle) stop_bundle(*_bundle, threading::get_id(), audit::outgoing{}, _ret);

    // unblock the signals in the entire process
    if(_enable_sampling && !_blocked.empty())
    {
        OMNITRACE_DEBUG("unblocking signals...\n");
        tim::sampling::unblock_signals(_blocked, tim::sampling::sigmask_scope::process);
    }

    OMNITRACE_DEBUG("returning success...\n");
    return _ret;
}
}  // namespace component
}  // namespace omnitrace

TIMEMORY_INITIALIZE_STORAGE(component::roctracer_data)
