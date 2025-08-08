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

#include "library/components/pthread_create_gotcha.hpp"
#include "core/config.hpp"
#include "core/debug.hpp"
#include "core/locking.hpp"
#include "core/state.hpp"
#include "core/utility.hpp"
#include "library/causal/delay.hpp"
#include "library/components/category_region.hpp"
#include "library/runtime.hpp"
#include "library/sampling.hpp"
#include "library/thread_data.hpp"
#include "library/thread_info.hpp"
#include "library/tracing.hpp"

#include <timemory/backends/threading.hpp>
#include <timemory/components/macros.hpp>
#include <timemory/components/timing/wall_clock.hpp>
#include <timemory/hash/types.hpp>
#include <timemory/mpl/types.hpp>
#include <timemory/sampling/allocator.hpp>
#include <timemory/units.hpp>
#include <timemory/utility/types.hpp>

#include <csignal>
#include <ostream>
#include <pthread.h>
#include <utility>

namespace rocprofsys
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
using bundle_t          = tim::lightweight_tuple<comp::wall_clock>;
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
start_bundle(bundle_t& _bundle, int64_t _tid, Args&&... _args)
{
    if(!get_use_timemory() && !get_use_perfetto()) return;
    ROCPROFSYS_BASIC_VERBOSE_F(3, "starting bundle '%s' in thread %li...\n",
                               _bundle.key().c_str(), _tid);
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
        _bundle.push(_tid);
        _bundle.start();
    }
}

template <typename... Args>
inline void
stop_bundle(bundle_t& _bundle, int64_t _tid, Args&&... _args)
{
    if(!get_use_timemory() && !get_use_perfetto()) return;

    auto _main_manager = tim::manager::master_instance();
    auto _this_manager = tim::manager::instance();
    if(!_main_manager || !_this_manager || _main_manager->is_finalized() ||
       _this_manager->is_finalized())
        return;

    ROCPROFSYS_BASIC_VERBOSE_F(3, "stopping bundle '%s' in thread %li...\n",
                               _bundle.key().c_str(), _tid);
    if(get_use_timemory())
    {
        auto _wc = *_bundle.get<comp::wall_clock>();
        _wc.stop();
        // update data
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

using native_handle_set_t = std::set<pthread_create_gotcha::native_handle_t>;

auto native_handles          = native_handle_set_t{};
auto internal_native_handles = native_handle_set_t{};
auto native_handles_mutex    = locking::atomic_mutex{};
}  // namespace

//--------------------------------------------------------------------------------------//

pthread_create_gotcha::wrapper::wrapper(routine_t _routine, void* _arg,
                                        wrapper_config _config)
: m_routine{ _routine }
, m_arg{ _arg }
, m_config{ std::move(_config) }
{}

void*
pthread_create_gotcha::wrapper::operator()() const
{
    using thread_bundle_data_t = thread_data<thread_bundle_t>;

    if(is_shutdown && *is_shutdown)
    {
        if(m_config.promise) m_config.promise->set_value();
        // execute the original function
        return m_routine(m_arg);
    }

    push_thread_state(ThreadState::Internal);

    int64_t     _tid         = -1;
    void*       _ret         = nullptr;
    auto        _is_sampling = false;
    auto        _bundle      = std::shared_ptr<bundle_t>{};
    auto        _signals     = std::set<int>{};
    auto        _coverage    = (get_mode() == Mode::Coverage);
    const auto& _parent_info = thread_info::get(m_config.parent_tid, InternalTID);
    const auto& _info        = thread_info::init(m_config.offset);
    auto        _dtor        = [&]() {
        set_thread_state(ThreadState::Internal);
        if(_is_sampling)
        {
            if(m_config.enable_causal)
            {
                causal::sampling::block_signals(_signals);
                causal::sampling::shutdown();
            }
            else if(m_config.enable_sampling)
            {
                sampling::block_signals(_signals);
                sampling::shutdown();
            }
        }

        if(_tid >= 0)
        {
            auto _active = (get_state() == ::rocprofsys::State::Active &&
                            bundles != nullptr && bundles_mutex != nullptr);
            if(!_active) return;
            thread_info::set_stop(comp::wall_clock::record());
            auto& _thr_bundle = thread_bundle_data_t::instance();
            if(_thr_bundle && _thr_bundle->get<comp::wall_clock>() &&
               _thr_bundle->get<comp::wall_clock>()->get_is_running())
                _thr_bundle->stop();
            if(_bundle) stop_bundle(*_bundle, _tid);
            pthread_create_gotcha::shutdown(_tid);
            ROCPROFSYS_BASIC_VERBOSE(
                1, "[PID=%i][rank=%i] Thread %s (parent: %s) exited\n", process::get_id(),
                dmp::rank(), _info->index_data->as_string().c_str(),
                _parent_info->index_data->as_string().c_str());
        }
    };

    auto _active = (get_state() == ::rocprofsys::State::Active && bundles != nullptr &&
                    bundles_mutex != nullptr);

    if(m_config.offset)
    {
        auto _lk = locking::atomic_lock{ native_handles_mutex };
        internal_native_handles.emplace(pthread_self());
    }

    if(_active && !_coverage && !m_config.offset)
    {
        _tid = _info->index_data->sequent_value;
        ROCPROFSYS_BASIC_VERBOSE(1, "[PID=%i][rank=%i] Thread %s (parent: %s) created\n",
                                 process::get_id(), dmp::rank(),
                                 _info->index_data->as_string().c_str(),
                                 _parent_info->index_data->as_string().c_str());
        threading::set_thread_name(TIMEMORY_JOIN(" ", "Thread", _tid).c_str());
        auto _manager = tim::manager::instance();
        if(_manager) _manager->initialize();
        if(!thread_bundle_data_t::get()->at(_tid))
        {
            thread_data<thread_bundle_t>::construct(
                TIMEMORY_JOIN('/', "rocprofsys/process", process::get_id(), "thread",
                              _tid),
                quirk::config<quirk::auto_start>{});
            thread_bundle_data_t::get()->at(_tid)->start();
        }
        if(bundles && bundles_mutex)
        {
            std::unique_lock<std::mutex> _lk{ *bundles_mutex };
            _bundle = bundles->emplace(_tid, std::make_shared<bundle_t>("start_thread"))
                          .first->second;
        }
        if(_bundle) start_bundle(*_bundle, _tid);
        get_cpu_cid_stack(_tid, m_config.parent_tid);
        if(m_config.enable_causal)
        {
            // children inherit the parent delay data
            if(_parent_info && _parent_info->index_data)
                causal::delay::get_local(_tid) =
                    causal::delay::get_local(_parent_info->index_data->sequent_value);
            _is_sampling = true;
            ROCPROFSYS_SCOPED_SAMPLING_ON_CHILD_THREADS(false);
            _signals = causal::sampling::setup();
            causal::sampling::unblock_signals();
        }
        else if(m_config.enable_sampling)
        {
            _is_sampling = true;
            ROCPROFSYS_SCOPED_SAMPLING_ON_CHILD_THREADS(false);
            _signals = sampling::setup();
            sampling::unblock_signals();
        }
    }
    else if(m_config.offset)
    {
        ROCPROFSYS_BASIC_VERBOSE(
            2,
            "[PID=%i][rank=%i] Thread %s (parent: %s) created [started by rocprof-sys]\n",
            process::get_id(), dmp::rank(), _info->index_data->as_string().c_str(),
            _parent_info->index_data->as_string().c_str());
    }

    // notify the wrapper that all internal work is completed
    if(m_config.promise) m_config.promise->set_value();

    // Internal -> Enabled
    pop_thread_state();

    push_thread_state(ThreadState::Enabled);

    // execute the original function
    _ret = m_routine(m_arg);

    if(get_state() < ::rocprofsys::State::Finalized)
    {
        pop_thread_state();

        // execute the destructor actions
        _dtor();

        set_thread_state(ThreadState::Completed);
    }

    return _ret;
}

void*
pthread_create_gotcha::wrapper::wrap(void* _arg)
{
    if(_arg == nullptr) return nullptr;

    auto _self = pthread_self();

    // convert the argument
    wrapper* _wrapper = static_cast<wrapper*>(_arg);

    // store the handle
    {
        auto _lk = locking::atomic_lock{ native_handles_mutex };
        native_handles.emplace(_self);
    }

    static thread_local auto _remover = scope::destructor{ []() {
        if(get_state() >= rocprofsys::State::Finalized) return;
        // remove the handle even if original function aborts
        auto                 _lk      = locking::atomic_lock{ native_handles_mutex };
        native_handles.erase(pthread_self());
    } };
    (void) _remover;

    // execute the original function
    void* _ret = (*_wrapper)();

    // remove the handle
    if(::pthread_equal(_self, pthread_self()) == 0)
    {
        auto _lk = locking::atomic_lock{ native_handles_mutex };
        native_handles.erase(_self);
    }

    // eliminate memory leak
    if(_ret != _arg) delete _wrapper;

    return _ret;
}

namespace
{
const auto shutdown_signal_v = SIGRTMAX - 1;

size_t shutdown_signals_delivered = 0;

void
pthread_create_gotcha_shutdown_handler(int)
{
    pthread_create_gotcha::shutdown(threading::get_id());
    ++shutdown_signals_delivered;
}
}  // namespace

void
pthread_create_gotcha::configure()
{
    pthread_create_gotcha_t::get_initializer() = []() {
        if(!tim::settings::enabled()) return;
        pthread_create_gotcha_t::template configure<
            0, int, pthread_t*, const pthread_attr_t*, void* (*) (void*), void*>(
            "pthread_create");
    };

    tim::hash::add_hash_id("start_thread");
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

    unsigned long _ndangling = 0;

    for(const auto& itr : *bundles)
    {
        if(itr.second) ++_ndangling;
    }

    tracing::copy_timemory_hash_ids();

    // enable the signal handler for when the timeout is reached
    struct sigaction _action = {};
    struct sigaction _former = {};

    memset(&_action, 0, sizeof(_action));
    memset(&_former, 0, sizeof(_former));
    sigemptyset(&_action.sa_mask);
    sigemptyset(&_former.sa_mask);

    _action.sa_flags   = SA_RESTART;
    _action.sa_handler = pthread_create_gotcha_shutdown_handler;
    // activate signal handler
    sigaction(shutdown_signal_v, &_action, &_former);

    size_t _expected_shutdown_signals_delivered = 0;
    {
        auto _lk = locking::atomic_lock{ native_handles_mutex };
        for(auto itr : native_handles)
        {
            // skip sending signals to internal threads
            if(internal_native_handles.count(itr) != 0) continue;

            bool                         has_bundle = false;
            std::unique_lock<std::mutex> _bundle_lk{ *bundles_mutex };
            const auto&                  thread_info = thread_info::get(itr);
            // Check if this thread has a corresponding bundle entry
            // With the new gotcha update more external threads are tracked
            // but we only want to send signals to threads that have bundles
            for(const auto& bundle_itr : *bundles)
            {
                try
                {
                    if(thread_info && thread_info->index_data &&
                       bundle_itr.first == thread_info->index_data->sequent_value)
                    {
                        has_bundle = true;
                        break;
                    }
                } catch(...)
                {
                    continue;
                }
            }

            if(has_bundle && pthread_equal(pthread_self(), itr) == 0)
            {
                ::pthread_kill(itr, shutdown_signal_v);
                ++_expected_shutdown_signals_delivered;
            }
        }

        auto           _nattempt    = 0U;
        constexpr auto nmax_attempt = 20U;
        while(shutdown_signals_delivered < _expected_shutdown_signals_delivered &&
              _nattempt++ < nmax_attempt)
        {
            std::this_thread::yield();
            std::this_thread::sleep_for(std::chrono::milliseconds{ 50 });
        }

        ROCPROFSYS_CI_BASIC_FAIL(
            shutdown_signals_delivered != _expected_shutdown_signals_delivered,
            "Number of signals delivered (%zu) != expected number of signals delievered "
            "(%zu)",
            shutdown_signals_delivered, _expected_shutdown_signals_delivered);
    }

    // restore existing signal handler
    sigaction(shutdown_signal_v, &_former, nullptr);

    // subtract the bundles that had signals delivered
    _ndangling -= shutdown_signals_delivered;

    // stop any remaining dangling bundles on this thread
    std::unique_lock<std::mutex> _lk{ *bundles_mutex };
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

    if(config::settings_are_configured())
    {
        ROCPROFSYS_VERBOSE(2 && _ndangling > 0,
                           "[pthread_create_gotcha] cleaned up %lu dangling bundles\n",
                           _ndangling);
    }
    else
    {
        ROCPROFSYS_BASIC_VERBOSE(
            2 && _ndangling > 0,
            "[pthread_create_gotcha] cleaned up %lu dangling bundles\n", _ndangling);
    }
}

void
pthread_create_gotcha::shutdown(int64_t _tid)
{
    if(_tid == 0) shutdown();

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

std::set<pthread_create_gotcha::native_handle_t>
pthread_create_gotcha::get_native_handles()
{
    auto _v = native_handles;
    return _v;
}

// pthread_create
int
pthread_create_gotcha::operator()(pthread_t* thread, const pthread_attr_t* attr,
                                  void* (*func)(void*), void*              arg) const
{
    auto        _tid          = utility::get_thread_index();
    auto        _thr_state    = get_thread_state();
    auto        _glob_state   = get_state();
    auto        _mode         = get_mode();
    auto        _disabled     = (_thr_state == ThreadState::Disabled);
    auto        _enabled      = (_thr_state == ThreadState::Enabled);
    auto        _bundle       = std::optional<bundle_t>{};
    auto        _sample_child = sampling_enabled_on_child_threads();
    auto        _active = (_glob_state == ::rocprofsys::State::Active && !_disabled);
    const auto& _info   = thread_info::init(!_active || !_sample_child || _disabled);

    ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);

    auto _coverage     = (_mode == Mode::Coverage);
    auto _use_sampling = config::get_use_sampling();
    auto _use_causal   = config::get_use_causal();
    auto _offset       = (!_enabled || !_active || _info->is_offset);
    auto _use_bundle   = (_active && !_coverage && !_offset);
    auto _enable_sampling =
        (_use_sampling && _sample_child && _active && !_coverage && !_offset);
    auto _enable_causal =
        (_use_causal && _sample_child && _active && !_coverage && !_offset);

    static bool debug_threading_get_id =
        get_env<bool>(TIMEMORY_SETTINGS_PREFIX "DEBUG_THREADING_GET_ID", false);

    auto _verbose = (debug_threading_get_id) ? 0 : 3;
    ROCPROFSYS_VERBOSE(
        _verbose,
        "Creating new thread :: global_state=%s, thread_state=%s, mode=%s, active=%s, "
        "coverage=%s, use_causal=%s, use_sampling=%s, sample_children=%s, tid=%li, "
        "use_bundle=%s, enable_causal=%s, enable_sampling=%s, thread_info=(%s)...\n",
        std::to_string(_glob_state).c_str(), std::to_string(_thr_state).c_str(),
        std::to_string(_mode).c_str(), std::to_string(_active).c_str(),
        std::to_string(_coverage).c_str(), std::to_string(_use_causal).c_str(),
        std::to_string(_use_sampling).c_str(), std::to_string(_sample_child).c_str(),
        _tid, std::to_string(_use_bundle).c_str(), std::to_string(_enable_causal).c_str(),
        std::to_string(_enable_sampling).c_str(), JOIN("", *_info).c_str());

    if(debug_threading_get_id)
    {
        timemory_print_demangled_backtrace<8>(std::cerr, std::string{},
                                              std::string{ "threading::get_id() [id=" } +
                                                  std::to_string(_tid) +
                                                  std::string{ "]" },
                                              std::string{ " " }, false);
    }

    if(_active && !_disabled && !_info->is_offset)
    {
        ROCPROFSYS_BASIC_VERBOSE(2, "[PID=%i][rank=%i] Starting new thread on %s...\n",
                                 process::get_id(), dmp::rank(),
                                 _info->index_data->as_string().c_str());
    }

    // ensure that cpu cid stack exists on the parent thread if active
    if(_active && !_coverage)
    {
        ROCPROFSYS_DEBUG("blocking signals...\n");
        get_cpu_cid_stack();
    }

    set_thread_state(ThreadState::Disabled);
    auto _blocked = get_sampling_signals();
    auto _promise = (_active) ? std::make_shared<std::promise<void>>() : promise_t{};
    auto _config =
        wrapper_config{ _enable_causal, _enable_sampling, _offset, _tid, _promise };
    auto* _wrap = new wrapper{ func, arg, _config };
    set_thread_state(ThreadState::Internal);

    // block the signals in entire process
    if(_enable_sampling && !_blocked.empty())
    {
        ROCPROFSYS_DEBUG("blocking signals...\n");
        tim::signals::block_signals(_blocked, tim::signals::sigmask_scope::process);
    }

    if(_use_bundle)
    {
        _bundle = bundle_t{ "pthread_create" };
        start_bundle(*_bundle, _info->index_data->sequent_value, audit::incoming{},
                     thread, attr, func, arg);
    }

    // threads must process their delays before creating a new thread
    causal::delay::process();

    // create the thread
    auto _ret = (*m_wrappee)(thread, attr, &wrapper::wrap, static_cast<void*>(_wrap));

    // wait for thread to set promise
    if(_promise)
    {
        ROCPROFSYS_DEBUG("waiting for child to signal it is setup...\n");
        _promise->get_future().wait_for(std::chrono::milliseconds{ 500 });
    }

    if(_use_bundle)
        stop_bundle(*_bundle, _info->index_data->sequent_value, audit::outgoing{}, _ret);

    // unblock the signals in the entire process
    if(_enable_sampling && !_blocked.empty())
    {
        ROCPROFSYS_DEBUG("unblocking signals...\n");
        tim::signals::unblock_signals(_blocked, tim::signals::sigmask_scope::process);
    }

    ROCPROFSYS_DEBUG("returning success...\n");
    return _ret;
}
}  // namespace component
}  // namespace rocprofsys
