// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "api.hpp"

#include "core/config.hpp"
#include "core/perfetto.hpp"
#include "core/perfetto_fwd.hpp"
#include "core/state.hpp"
#include "library/components/fork_gotcha.hpp"
#include "library/runtime.hpp"
#include "library/sampling.hpp"

#include <timemory/backends/process.hpp>
#include <timemory/backends/threading.hpp>
#include <timemory/mpl/types.hpp>
#include <timemory/process/process.hpp>

#include "logger/debug.hpp"

#include <cstdlib>
#include <memory>
#include <pthread.h>
#include <unistd.h>

namespace rocprofsys
{
namespace component
{
namespace
{
// these are used to prevent handlers from executing multiple times
thread_local bool prefork_lock         = false;
thread_local bool postfork_parent_lock = false;
thread_local bool postfork_child_lock  = false;

// this does a quick exit (no cleanup) on child processes
// because perfetto has a tendency to access memory it
// shouldn't during cleanup
void
child_exit(int _ec, void*)
{
    std::quick_exit(_ec);
}

void
prefork_setup()
{
    if(prefork_lock) return;

    ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
    ROCPROFSYS_SCOPED_SAMPLING_ON_CHILD_THREADS(false);

    if(get_state() < State::Active && !config::settings_are_configured())
        rocprofsys_init_library_hidden();

    rocprofsys::set_env("ROCPROFSYS_PRELOAD", "0", 1);
    rocprofsys::set_env("ROCPROFSYS_ROOT_PROCESS", process::get_id(), 0);
    rocprofsys_reset_preload_hidden();
    LOG_INFO("fork() called on PID {} (rank: {}), TID {}", process::get_id(), dmp::rank(),
             threading::get_id());
    LOG_WARNING("Calling fork() within an OpenMPI application using libfabric "
                "may result is segmentation fault");
    // TIMEMORY_CONDITIONAL_DEMANGLED_BACKTRACE(get_debug_env(), 16);

    if(config::get_use_sampling())
    {
        sampling::block_samples();
        sampling::prefork_lock_pmc_sampler();
    }

    rocprofsys::categories::disable_categories(config::get_enabled_categories());

    // prevent re-entry until post-fork routines have been called
    prefork_lock         = true;
    postfork_parent_lock = false;
    postfork_child_lock  = false;
}

void
postfork_parent()
{
    if(postfork_parent_lock) return;

    // Reinitialize AMD SMI in parent process to get fresh device handles before
    // unblocking the shutdown/setup transition. AMD SMI device handles may be corrupted
    // after fork.
    if(config::get_use_sampling())
    {
        sampling::postfork_parent_reinit();
        sampling::postfork_parent_unlock_pmc_sampler();
    }

    rocprofsys::categories::enable_categories(config::get_enabled_categories());

    if(config::get_use_sampling()) sampling::unblock_samples();

    // prevent re-entry until prefork has been called
    postfork_parent_lock = true;
    prefork_lock         = false;
}

void
postfork_child()
{
    if(postfork_child_lock) return;

    if(!is_child_process())
    {
        LOG_ERROR("Child process {} believes it is the root process {}",
                  process::get_id(), get_root_process_id());
        std::exit(1);
    }

    set_state(State::Finalized);

    // Clean up AMD SMI in child process before other shutdowns
    if(config::get_use_sampling())
    {
        sampling::postfork_child_reset_pmc_sampler_lock();
        sampling::postfork_child_cleanup();
    }

    settings::enabled() = false;
    settings::verbose() = -127;
    settings::debug()   = false;
    rocprofsys::sampling::shutdown();
    rocprofsys::categories::shutdown();
    set_thread_state(::rocprofsys::ThreadState::Disabled);

    rocprofsys::get_perfetto_session(process::get_parent_id()).release();

    // register these exit handlers to avoid cleaning up resources
    on_exit(&child_exit, nullptr);
    std::atexit([]() { child_exit(EXIT_SUCCESS, nullptr); });

    // prevent re-entry until prefork has been called
    postfork_child_lock = true;
    prefork_lock        = false;
}
}  // namespace

void
fork_gotcha::configure()
{
    fork_gotcha_t::get_initializer() = []() {
        TIMEMORY_C_GOTCHA(fork_gotcha_t, 0, fork);
    };

    // registering the pthread_atfork and gotcha means that we might execute twice
    // handlers twice, hence the locks
    pthread_atfork(&prefork_setup, &postfork_parent, &postfork_child);
}

pid_t
fork_gotcha::operator()(const gotcha_data_t&, pid_t (*_real_fork)()) const
{
    prefork_setup();

    auto _pid = (*_real_fork)();

    if(_pid != 0)
    {
        LOG_INFO("fork() called on PID {} created PID {}", getpid(), _pid);

        postfork_parent();
    }
    else
    {
        postfork_child();
    }

    if(!settings::use_output_suffix())
    {
        LOG_DEBUG("Application which make calls to fork() should enable using an process "
                  "identifier output suffix (i.e. set ROCPROFSYS_USE_PID=ON)");
    }

    return _pid;
}
}  // namespace component
}  // namespace rocprofsys
