// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/components/pthread_create_gotcha.hpp"
#include "common/clocks.hpp"
#include "core/config.hpp"
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
#include <cstdint>

#include <timemory/backends/threading.hpp>
#include <timemory/components/macros.hpp>
#include <timemory/components/timing/wall_clock.hpp>
#include <timemory/hash/types.hpp>
#include <timemory/mpl/types.hpp>
#include <timemory/sampling/allocator.hpp>
#include <timemory/utility/types.hpp>

#include "logger/debug.hpp"

#include <dlfcn.h>
#include <ostream>
#include <pthread.h>
#include <string_view>
#include <utility>

namespace rocprofsys
{
namespace sampling
{}  // namespace sampling

namespace component
{
using category_region_t = tim::lightweight_tuple<category_region<category::pthread>>;

namespace
{
constexpr std::string_view k_pthread_create_region = "pthread_create";

auto* is_shutdown   = new bool{ false };  // intentional data leak
auto  shutdown_dtor = scope::destructor{ []() { pthread_create_gotcha::shutdown(); } };

template <typename... Args>
inline void
start_pthread_create_region(std::int64_t _tid, Args&&... _args)
{
    if(!get_use_timemory())
    {
        return;
    }
    LOG_TRACE("Starting '{}' in thread {}...", k_pthread_create_region, _tid);
    category_region_t{}.audit(quirk::config<quirk::cache_only>{}, k_pthread_create_region,
                              _args...);
}

template <typename... Args>
inline void
stop_pthread_create_region(std::int64_t _tid, Args&&... _args)
{
    if(!get_use_timemory())
    {
        return;
    }

    auto const _main_manager = tim::manager::master_instance();
    auto const _this_manager = tim::manager::instance();
    if(!_main_manager || !_this_manager || _main_manager->is_finalized() ||
       _this_manager->is_finalized())
    {
        return;
    }

    LOG_TRACE("Stopping '{}' in thread {}...", k_pthread_create_region, _tid);
    category_region_t{}.audit(quirk::config<quirk::cache_only>{}, k_pthread_create_region,
                              _args...);
}

using native_handle_set_t = std::set<pthread_create_gotcha::native_handle_t>;

auto native_handles          = native_handle_set_t{};
auto internal_native_handles = native_handle_set_t{};
auto native_handles_mutex    = locking::atomic_mutex{};

// Shared across the parent interceptor and the child wrapper so the thread-limit
// warning is emitted exactly once per process, regardless of which gate trips first.
std::once_flag thread_limit_warning_flag;

inline void
warn_thread_limit_once()
{
    std::call_once(thread_limit_warning_flag, []() {
        LOG_WARNING("Maximum allowed thread limit ({}) reached. Further profiling will "
                    "be disabled "
                    "to prevent resource exhaustion. Consider increasing the limit at "
                    "compile time "
                    "using the ROCPROFSYS_MAX_THREADS CMake option.",
                    static_cast<size_t>(ROCPROFSYS_MAX_THREADS));
    });
}
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
    auto _child_internal_tid   = utility::get_thread_index();
    if(static_cast<size_t>(_child_internal_tid) >= ROCPROFSYS_MAX_THREADS ||
       thread_info::get_initialized_thread() >= ROCPROFSYS_MAX_THREADS)
    {
        warn_thread_limit_once();
        if(m_config.promise)
        {
            m_config.promise->set_value();
        }
        return m_routine(m_arg);
    }
    if(is_shutdown && *is_shutdown)
    {
        if(m_config.promise)
        {
            m_config.promise->set_value();
        }
        // execute the original function
        return m_routine(m_arg);
    }

    state::thread::push(state::thread::Internal);

    std::int64_t _tid         = -1;
    void*        _ret         = nullptr;
    auto         _is_sampling = false;
    auto         _signals     = std::set<int>{};
    auto const   _coverage    = (get_mode() == state::process::Mode::coverage);
    const auto&  _parent_info = thread_info::get(m_config.parent_tid, InternalTID);
    const auto&  _info        = thread_info::init(m_config.offset);
    if(!_info)
    {
        if(m_config.promise)
        {
            m_config.promise->set_value();
        }
        return m_routine(m_arg);
    }
    auto const _dtor = [&]() {
        state::thread::set(state::thread::Internal);
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
            if(state::process::get() != ::rocprofsys::state::process::Active)
            {
                return;
            }
            thread_info::set_stop(clocks::timeline::now_ns());
            auto& _thr_bundle = thread_bundle_data_t::instance();
            if(_thr_bundle && _thr_bundle->get<comp::wall_clock>() &&
               _thr_bundle->get<comp::wall_clock>()->get_is_running())
            {
                _thr_bundle->stop();
            }
            LOG_DEBUG("[PID={}][rank={}] Thread {} (parent: {}) exited",
                      process::get_id(), dmp::rank(), _info->index_data->as_string(),
                      _parent_info->index_data->as_string());
        }
    };

    auto const _active = (state::process::get() == ::rocprofsys::state::process::Active);

    if(m_config.offset)
    {
        auto const _lk = locking::atomic_lock{ native_handles_mutex };
        internal_native_handles.emplace(pthread_self());
    }

    if(_active && !_coverage && !m_config.offset)
    {
        _tid = _info->index_data->sequent_value;
        LOG_DEBUG("[PID={}][rank={}] Thread {} (parent: {}) created", process::get_id(),
                  dmp::rank(), _info->index_data->as_string(),
                  _parent_info->index_data->as_string());
        threading::set_thread_name(fmt::format("Thread {}", _tid).c_str());
        auto const _manager = tim::manager::instance();
        if(_manager)
        {
            _manager->initialize();
        }
        if(!thread_bundle_data_t::get()->at(_tid))
        {
            thread_data<thread_bundle_t>::construct(
                fmt::format("rocprofsys/process/{}/thread/{}", process::get_id(), _tid),
                quirk::config<quirk::auto_start>{});
            thread_bundle_data_t::get()->at(_tid)->start();
        }
        // The wall_clock report times every application thread, including threads
        // that never emit a region, so each one must be in the trace-cache metadata
        if(get_use_timemory())
        {
            utility::thread_metadata_source{}.resolve_current_thread();
        }
        get_cpu_cid_stack(_tid, m_config.parent_tid);
        if(m_config.enable_causal)
        {
            // children inherit the parent delay data
            if(_parent_info && _parent_info->index_data)
            {
                causal::delay::get_local(_tid) =
                    causal::delay::get_local(_parent_info->index_data->sequent_value);
            }
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
        LOG_DEBUG(
            "[PID={}][rank={}] Thread {} (parent: {}) created [started by rocprof-sys]",
            process::get_id(), dmp::rank(), _info->index_data->as_string(),
            _parent_info->index_data->as_string());
    }

    // notify the wrapper that all internal work is completed
    if(m_config.promise)
    {
        m_config.promise->set_value();
    }

    // Internal -> Enabled
    state::thread::pop();

    state::thread::push(state::thread::Enabled);

    // execute the original function
    _ret = m_routine(m_arg);

    if(state::process::get() < ::rocprofsys::state::process::Finalized)
    {
        state::thread::pop();

        // execute the destructor actions
        _dtor();

        state::thread::set(state::thread::Completed);
    }

    return _ret;
}

void*
pthread_create_gotcha::wrapper::wrap(void* _arg)
{
    if(_arg == nullptr)
    {
        return nullptr;
    }

    auto const _self = pthread_self();

    // convert the argument
    wrapper const* _wrapper = static_cast<wrapper*>(_arg);

    // store the handle
    {
        auto const _lk = locking::atomic_lock{ native_handles_mutex };
        native_handles.emplace(_self);
    }

    static thread_local auto const _remover = scope::destructor{ []() {
        if(state::process::get() >= rocprofsys::state::process::Finalized)
        {
            return;
        }
        // remove the handle even if original function aborts
        auto const                 _lk      = locking::atomic_lock{ native_handles_mutex };
        native_handles.erase(pthread_self());
    } };
    (void) _remover;

    // execute the original function
    void* _ret = (*_wrapper)();

    // remove the handle
    if(::pthread_equal(_self, pthread_self()) == 0)
    {
        auto const _lk = locking::atomic_lock{ native_handles_mutex };
        native_handles.erase(_self);
    }

    // eliminate memory leak
    if(_ret != _arg)
    {
        delete _wrapper;
    }

    return _ret;
}

void
pthread_create_gotcha::configure()
{
    pthread_create_gotcha_t::get_initializer() = []() {
        if(!tim::settings::enabled())
        {
            return;
        }
        pthread_create_gotcha_t::template configure<
            0, int, pthread_t*, const pthread_attr_t*, void* (*) (void*), void*>(
            "pthread_create");
    };
}

void
pthread_create_gotcha::shutdown()
{
    if(*is_shutdown)
    {
        return;
    }
    *is_shutdown = true;

    tracing::copy_timemory_hash_ids();
}

std::atomic<bool> pthread_create_gotcha::s_is_paused = false;

void
pthread_create_gotcha::pause()
{
    s_is_paused.store(true, std::memory_order_relaxed);
}
void
pthread_create_gotcha::resume()
{
    s_is_paused.store(false, std::memory_order_relaxed);
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

namespace
{
constexpr const std::array k_rocm_internal_libraries = { "libhsa-runtime64",
                                                         "librocprofiler-sdk",
                                                         "libamdhip64" };

bool
is_rocm_internal_thread(void* func_ptr)
{
    if(!func_ptr)
    {
        return false;
    }

    Dl_info info;
    if(dladdr(func_ptr, &info) == 0 || info.dli_fname == nullptr)
    {
        LOG_TRACE("dladdr failed or returned no filename for func_ptr={:p}", func_ptr);
        return false;
    }

    const std::string_view lib_name{ info.dli_fname };

    return std::ranges::any_of(k_rocm_internal_libraries, [lib_name](const auto* lib) {
        return lib_name.find(lib) != std::string_view::npos;
    });
}
}  // namespace

// pthread_create
int
pthread_create_gotcha::operator()(pthread_t* thread, const pthread_attr_t* attr,
                                  void* (*func)(void*), void*              arg) const
{
    // Bypass wrapper for internal ROCm threads to avoid interfering with their event
    // loops
    if(is_rocm_internal_thread(reinterpret_cast<void*>(func)))
    {
        return (*m_wrappee)(thread, attr, func, arg);
    }

    auto tid = utility::get_thread_index();
    if(static_cast<size_t>(tid) >= ROCPROFSYS_MAX_THREADS ||
       thread_info::get_initialized_thread() >= ROCPROFSYS_MAX_THREADS)
    {
        warn_thread_limit_once();
        return (*m_wrappee)(thread, attr, func, arg);
    }
    auto       thr_state    = state::thread::get();
    auto       glob_state   = state::process::get();
    auto       mode         = get_mode();
    auto const disabled     = (thr_state == state::thread::Disabled);
    auto const enabled      = (thr_state == state::thread::Enabled);
    auto const sample_child = sampling_enabled_on_child_threads();
    auto const active = (glob_state == ::rocprofsys::state::process::Active && !disabled);
    const auto& info  = thread_info::init(!active || !sample_child || disabled);
    if(!info)
    {
        // Untracked parent threads cannot safely create wrapped/profiled child threads.
        return (*m_wrappee)(thread, attr, func, arg);
    }

    auto const thread_state_guard = state::thread::scoped(state::thread::Internal);
    auto const coverage           = (mode == state::process::Mode::coverage);
    auto const use_sampling       = config::get_use_sampling();
    auto const use_causal         = config::get_use_causal();
    auto const offset             = (!enabled || !active || info->is_offset);
    auto const record_region      = (active && !coverage && !offset && !is_paused());
    auto const enable_sampling =
        (use_sampling && sample_child && active && !coverage && !offset);
    auto const enable_causal =
        (use_causal && sample_child && active && !coverage && !offset);

    static const bool k_debug_threading_get_id =
        get_env<bool>(TIMEMORY_SETTINGS_PREFIX "DEBUG_THREADING_GET_ID", false);

    if(k_debug_threading_get_id)
    {
        LOG_TRACE(
            "Creating new thread :: global_state={}, thread_state={}, mode={}, "
            "active={}, "
            "coverage={}, use_causal={}, use_sampling={}, sample_children={}, tid={}, "
            "record_region={}, enable_causal={}, enable_sampling={}, thread_info={}...",
            glob_state, thr_state, mode, std::to_string(active), std::to_string(coverage),
            std::to_string(use_causal), std::to_string(use_sampling),
            std::to_string(sample_child), std::to_string(tid),
            std::to_string(record_region), std::to_string(enable_causal),
            std::to_string(enable_sampling), info->as_string());

        std::stringstream backtrace_ss;
        constexpr auto    k_demangle_backtrace_depth = 8;
        timemory_print_demangled_backtrace<k_demangle_backtrace_depth>(
            backtrace_ss, std::string{},
            std::string{ "threading::get_id() [id=" } + std::to_string(tid) +
                std::string{ "]" },
            std::string{ " " }, false);
        LOG_TRACE("Backtrace: {}", backtrace_ss.str());
    }

    if(active && !disabled && !info->is_offset)
    {
        LOG_DEBUG("[PID={}][rank={}] Starting new thread on {}", process::get_id(),
                  dmp::rank(), info->index_data->as_string().c_str());
    }

    // ensure that cpu cid stack exists on the parent thread if active
    if(active && !coverage)
    {
        LOG_DEBUG("Locking signals...");
        get_cpu_cid_stack();
    }

    state::thread::set(state::thread::Disabled);
    auto const blocked = get_sampling_signals();
    auto const promise = active ? std::make_shared<std::promise<void>>() : promise_t{};
    auto const config  = wrapper_config{ .enable_causal   = enable_causal,
                                         .enable_sampling = enable_sampling,
                                         .offset          = offset,
                                         .parent_tid      = tid,
                                         .promise         = promise };
    auto*      wrap    = new wrapper{ func, arg, config };
    state::thread::set(state::thread::Internal);

    // block the signals in entire process
    if(enable_sampling && !blocked.empty())
    {
        LOG_DEBUG("Blocking signals...");
        tim::signals::block_signals(blocked, tim::signals::sigmask_scope::process);
    }

    if(record_region)
    {
        start_pthread_create_region(info->index_data->sequent_value, audit::incoming{},
                                    thread, attr, func, arg);
    }

    // threads must process their delays before creating a new thread
    causal::delay::process();

    // create the thread
    auto ret = (*m_wrappee)(thread, attr, &wrapper::wrap, static_cast<void*>(wrap));

    // wait for thread to set promise
    if(promise)
    {
        LOG_DEBUG("Waiting for child to signal it is setup...");
        promise->get_future().wait_for(500ms);
    }

    if(record_region)
    {
        stop_pthread_create_region(info->index_data->sequent_value, audit::outgoing{},
                                   ret);
    }

    // unblock the signals in the entire process
    if(enable_sampling && !blocked.empty())
    {
        LOG_DEBUG("Unblocking signals...");
        tim::signals::unblock_signals(blocked, tim::signals::sigmask_scope::process);
    }

    LOG_DEBUG("Returning success...");
    return ret;
}
}  // namespace component
}  // namespace rocprofsys
