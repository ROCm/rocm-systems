// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/defines.h"
#include "common/env_vars.hpp"
#include "common/environment.hpp"
#include "core/categories.hpp"
#include "core/locking.hpp"
#include "core/state.hpp"
#include "library/components/pthread_gotcha.hpp"
#include "library/runtime.hpp"
#include "library/thread_info.hpp"
#include "logger/debug.hpp"

#include <timemory/log/color.hpp>
#include <timemory/process/process.hpp>
#include <timemory/signals/types.hpp>
#include <timemory/unwind/backtrace.hpp>

#include <bits/pthreadtypes.h>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <pthread.h>
#include <signal.h>
#include <sstream>
#include <sys/types.h>
#include <thread>
#include <utility>

namespace rocprofsys::timeout
{

void
setup() ROCPROFSYS_INTERNAL_API;

namespace
{
namespace unwind  = ::tim::unwind;
namespace signals = ::tim::signals;
namespace log     = ::tim::log;

constexpr auto k_factor              = 3.0;
constexpr auto k_factor_devider      = 1.25;
constexpr auto k_max_iteration_count = 50;
constexpr auto k_timeout_signal      = signals::sys_signal::Hangup;
constexpr auto k_timeout_signal_v    = static_cast<int>(k_timeout_signal);

auto                       g_main_thread_native_handle         = pthread_self();
bool                       g_ci_timeout_active                 = false;
auto                       g_ci_timeout_mutex                  = locking::atomic_mutex{};
std::uint64_t              g_ci_timeout_backtrace_global_count = 1;
std::uint64_t              g_ci_timeout_backtrace_global_done  = 0;
thread_local std::uint64_t g_ci_timeout_backtrace_local_count  = 0;

void
ci_timeout_backtrace(int)
{
    if(g_ci_timeout_backtrace_local_count >= g_ci_timeout_backtrace_global_count)
    {
        return;
    }
    ++g_ci_timeout_backtrace_local_count;

    auto err            = std::stringstream{};
    auto cfg            = unwind::detailed_backtrace_config{};
    cfg.proc_pid_maps   = false;
    cfg.unwind_lineinfo = false;
    cfg.force_color     = !log::monochrome();

    unwind::detailed_backtrace<0>(err, cfg);

    static auto s_mutex = locking::atomic_mutex{};
    const auto  lock    = locking::atomic_lock{ s_mutex };
    LOG_INFO("{}", err.str());

    ++g_ci_timeout_backtrace_global_done;
}

void
log_pthread_kill_failure(pthread_t handle)
{
    const auto& info = thread_info::get(handle);
    // NOLINTBEGIN
    if(info.has_value())
    {
        LOG_WARNING("pthread_kill({}, {}) failed for thread {} (info: {})",
                    static_cast<std::size_t>(handle), k_timeout_signal_v,
                    info->index_data->sequent_value, info->as_string());
        return;
    }
    // NOLINTEND

    LOG_WARNING("pthread_kill({}, {}) failed. executing generic kill({}, {})...", handle,
                k_timeout_signal_v, process::get_id(), k_timeout_signal_v);
}

void
emit_backtrace_for_thread(pthread_t handle, std::chrono::duration<double> pause)
{
    const auto done_v = g_ci_timeout_backtrace_global_done;
    if(::pthread_kill(handle, k_timeout_signal_v) != 0)
    {
        log_pthread_kill_failure(handle);
        ::kill(process::get_id(), k_timeout_signal_v);
    }

    // wait until ci_timeout_backtrace increments the global done count (or the
    // iteration cap is hit) so that the backtraces do not overlap in the output
    auto iteration = 0;
    while(g_ci_timeout_backtrace_global_done == done_v &&
          iteration++ < k_max_iteration_count)
    {
        std::this_thread::sleep_for(pause);
    }
}

void
emit_backtrace_for_all_threads(double ci_timeout_seconds, double factor)
{
    auto       tids  = pthread_gotcha::get_native_handles();
    const auto pause = std::chrono::duration<double>(factor) / (3 * (tids.size() + 1));

    tids.erase(g_main_thread_native_handle);
    LOG_WARNING("Timeout after {} seconds... Generating backtraces for "
                "{} threads...",
                ci_timeout_seconds, tids.size() + 1);

    for(const auto itr : tids)
    {
        emit_backtrace_for_thread(itr, pause);
    }

    emit_backtrace_for_thread(g_main_thread_native_handle, pause);
}

void
ensure_ci_timeout_backtrace(double             ci_timeout_seconds,
                            std::promise<void> ci_timeout_ready)
{
    ci_timeout_ready.set_value();

    thread_info::init(true);
    const auto thread_state_guard = state::thread::scoped(state::thread::Disabled);

    auto factor = k_factor;
    while(ci_timeout_seconds <= factor)
    {
        factor /= k_factor_devider;
    }

    std::uint64_t      ci_timeout_nitr = 0;
    const std::int64_t ci_timeout_nanosec =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>{ ci_timeout_seconds - factor })
            .count();
    const auto ci_timeout_total_count =
        get_env<std::uint64_t>(env_vars::CI_TIMEOUT_COUNT, 1);
    const auto root_pid = get_env<pid_t>(env_vars::ROOT_PROCESS, process::get_id());

    while(state::process::get() < state::process::Finalized &&
          ci_timeout_nitr < ci_timeout_total_count)
    {
        // sleep until timeout reached
        std::this_thread::sleep_for(std::chrono::nanoseconds{ ci_timeout_nanosec });

        // guard against thread in fork
        if(process::get_id() != root_pid)
        {
            g_ci_timeout_active = false;
            setup();
            return;
        }

        emit_backtrace_for_all_threads(ci_timeout_seconds, factor);

        if(++ci_timeout_nitr >= ci_timeout_total_count)
        {
            // use SIGQUIT because it will generate a core dump
            ::kill(process::get_id(), SIGQUIT);
            return;
        }

        ++g_ci_timeout_backtrace_global_count;
    }

    LOG_WARNING("Timeout thread exiting...");
}
}  // namespace

void
setup()
{
    // make sure there isn't any datarace for ci_timeout_active
    auto lock = locking::atomic_lock{ g_ci_timeout_mutex };

    if(g_ci_timeout_active)
    {
        return;
    }

    // in CI mode, if ROCPROFSYS_CI_TIMEOUT or ROCPROFSYS_CI_TIMEOUT_OVERRIDE is
    // set, start a thread that will print out the backtrace for each thread
    // before the timeout is hit (i.e. killed by CTest) so we can potentially
    // diagnose where the code is stuck
    const auto is_ci = get_env(env_vars::CI, false);
    if(is_ci)
    {
        // set by CTest
        const auto ci_timeout_default = get_env(env_vars::CI_TIMEOUT, -1.0);
        // allow override by user
        const auto ci_timeout_seconds =
            get_env(env_vars::CI_TIMEOUT_OVERRIDE, ci_timeout_default);

        if(ci_timeout_seconds > 0.0)
        {
            // lock served its purpose after setting to true
            g_ci_timeout_active = true;
            lock.unlock();

            const auto thread_state_guard =
                state::thread::scoped(state::thread::Internal);
            ROCPROFSYS_SCOPED_SAMPLING_ON_CHILD_THREADS(false);

            // enable the signal handler for when the timeout is reached
            struct sigaction action = {};
            sigemptyset(&action.sa_mask);
            action.sa_flags   = SA_RESTART;
            action.sa_handler = ci_timeout_backtrace;
            sigaction(k_timeout_signal_v, &action, nullptr);

            // start a background thread that handles waiting for the timeout
            auto       ci_timeout_ready = std::promise<void>{};
            const auto ci_timeout_wait  = ci_timeout_ready.get_future();
            std::thread{ ensure_ci_timeout_backtrace, ci_timeout_seconds,
                         std::move(ci_timeout_ready) }
                .detach();
            ci_timeout_wait.wait_for(std::chrono::seconds{ 1 });
        }
    }
}
}  // namespace rocprofsys::timeout
