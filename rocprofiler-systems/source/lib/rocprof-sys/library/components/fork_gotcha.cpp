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

#include "api.hpp"

#include "core/config.hpp"
#include "core/debug.hpp"
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
bool prefork_lock         = false;
bool postfork_parent_lock = false;
bool postfork_child_lock  = false;

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

    tim::set_env("ROCPROFSYS_PRELOAD", "0", 1);
    tim::set_env("ROCPROFSYS_ROOT_PROCESS", process::get_id(), 0);
    rocprofsys_reset_preload_hidden();
    ROCPROFSYS_BASIC_VERBOSE(0, "fork() called on PID %i (rank: %i), TID %li\n",
                             process::get_id(), dmp::rank(), threading::get_id());
    ROCPROFSYS_BASIC_DEBUG(
        "Warning! Calling fork() within an OpenMPI application using libfabric "
        "may result is segmentation fault\n");
    TIMEMORY_CONDITIONAL_DEMANGLED_BACKTRACE(get_debug_env(), 16);

    if(config::get_use_sampling()) sampling::block_samples();

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

    ROCPROFSYS_REQUIRE(is_child_process())
        << "Error! child process " << process::get_id()
        << " believes it is the root process " << get_root_process_id() << "\n";

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
        ROCPROFSYS_BASIC_VERBOSE(0, "fork() called on PID %i created PID %i\n", getpid(),
                                 _pid);

        postfork_parent();
    }
    else
    {
        postfork_child();
    }

    if(!settings::use_output_suffix())
    {
        ROCPROFSYS_BASIC_VERBOSE(
            0, "Application which make calls to fork() should enable using an process "
               "identifier output suffix (i.e. set ROCPROFSYS_USE_PID=ON)\n");
    }

    return _pid;
}
}  // namespace component
}  // namespace rocprofsys
