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

#include "core/categories.hpp"
#include "core/config.hpp"
#include "core/debug.hpp"
#include "core/locking.hpp"
#include "core/state.hpp"
#include "library/components/pthread_gotcha.hpp"
#include "library/runtime.hpp"
#include "library/thread_info.hpp"

#include <timemory/log/color.hpp>
#include <timemory/signals/types.hpp>
#include <timemory/unwind/backtrace.hpp>

#include <chrono>
#include <sstream>
#include <thread>

namespace rocprofsys
{
namespace timeout
{
void
setup() ROCPROFSYS_INTERNAL_API;

namespace
{
namespace unwind  = ::tim::unwind;
namespace signals = ::tim::signals;
namespace log     = ::tim::log;

constexpr auto timeout_signal   = signals::sys_signal::Hangup;
constexpr auto timeout_signal_v = static_cast<int>(timeout_signal);

auto                  main_thread_native_handle         = pthread_self();
bool                  ci_timeout_active                 = false;
auto                  ci_timeout_mutex                  = locking::atomic_mutex{};
uint64_t              ci_timeout_backtrace_global_count = 1;
uint64_t              ci_timeout_backtrace_global_done  = 0;
thread_local uint64_t ci_timeout_backtrace_local_count  = 0;

void
ci_timeout_backtrace(int)
{
    if(ci_timeout_backtrace_local_count >= ci_timeout_backtrace_global_count) return;
    ++ci_timeout_backtrace_local_count;

    auto _err            = std::stringstream{};
    auto _cfg            = unwind::detailed_backtrace_config{};
    _cfg.proc_pid_maps   = false;
    _cfg.unwind_lineinfo = false;
    _cfg.force_color     = !log::monochrome();

    unwind::detailed_backtrace<0>(_err, _cfg);

    static auto _mutex = locking::atomic_mutex{};
    auto        _lk    = locking::atomic_lock{ _mutex };
    ROCPROFSYS_PRINT("%s\n", _err.str().c_str());

    ++ci_timeout_backtrace_global_done;
}

void
ensure_ci_timeout_backtrace(double             _ci_timeout_seconds,
                            std::promise<void> _ci_timeout_ready)
{
    _ci_timeout_ready.set_value();

    thread_info::init(true);
    ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Disabled);

    auto _factor = 3.0;
    while(_ci_timeout_seconds <= _factor)
        _factor /= 1.25;

    uint64_t _ci_timeout_nitr    = 0;
    int64_t  _ci_timeout_nanosec = (_ci_timeout_seconds - _factor) * units::sec;
    auto     _ci_timeout_total_count =
        get_env<uint64_t>("ROCPROFSYS_CI_TIMEOUT_COUNT", 1, false);
    const auto root_pid =
        get_env<pid_t>("ROCPROFSYS_ROOT_PROCESS", process::get_id(), false);

    while(get_state() < State::Finalized && _ci_timeout_nitr < _ci_timeout_total_count)
    {
        // sleep until timeout reached
        std::this_thread::sleep_for(std::chrono::nanoseconds{ _ci_timeout_nanosec });

        // guard against thread in fork
        if(process::get_id() != root_pid)
        {
            ci_timeout_active = false;
            setup();
            return;
        }

        auto    _tids             = pthread_gotcha::get_native_handles();
        int64_t _ci_timeout_pause = (_factor * units::sec) / (3 * (_tids.size() + 1));
        auto    _kill_thread      = [_ci_timeout_pause](auto _handle) {
            // execute the pthread_kill and wait until ci_timeout_backtrace increments
            // ci_timeout_backtrace_global_done (or 50 iterations pass) to avoid
            // the backtraces overlapping output
            auto _n      = 0;
            auto _done_v = ci_timeout_backtrace_global_done;
            if(::pthread_kill(_handle, timeout_signal_v) != 0)
            {
                const auto& _info = thread_info::get(_handle);
                if(_info)
                {
                    ROCPROFSYS_WARNING_F(
                        0, "pthread_kill(%zu, %i) failed for thread %zi (info: %s)\n",
                        _handle, timeout_signal_v, _info->index_data->sequent_value,
                        _info->as_string().c_str());
                }
                else
                {
                    ROCPROFSYS_WARNING_F(
                        0,
                        "pthread_kill(%zu, %i) failed. executing generic "
                                "kill(%i, %i)...\n",
                        _handle, timeout_signal_v, process::get_id(), timeout_signal_v);
                }

                ::kill(process::get_id(), timeout_signal_v);
            }

            // wait until the signal has been delivered
            while(ci_timeout_backtrace_global_done == _done_v && _n++ < 50)
                std::this_thread::sleep_for(
                    std::chrono::nanoseconds{ _ci_timeout_pause });
        };

        _tids.erase(main_thread_native_handle);
        ROCPROFSYS_WARNING_F(-127,
                             "timeout after %8.3f seconds... Generating backtraces for "
                             "%zu threads...\n",
                             _ci_timeout_seconds, _tids.size() + 1);

        for(auto itr : _tids)
            _kill_thread(itr);

        _kill_thread(main_thread_native_handle);

        ::rocprofsys::debug::flush();
        ::rocprofsys::debug::lock _debug_lk{};

        if(++_ci_timeout_nitr >= _ci_timeout_total_count)
        {
            // use SIGQUIT because it will generate a core dump
            ::kill(process::get_id(), SIGQUIT);
            return;
        }
        else
        {
            ++ci_timeout_backtrace_global_count;
        }
    }

    ROCPROFSYS_WARNING_F(0, "timeout thread exiting...\n");
}
}  // namespace

void
setup()
{
    // make sure there isn't any datarace for ci_timeout_active
    auto _lk = locking::atomic_lock{ ci_timeout_mutex };

    if(ci_timeout_active) return;

    // in CI mode, if ROCPROFSYS_CI_TIMEOUT or ROCPROFSYS_CI_TIMEOUT_OVERRIDE is
    // set, start a thread that will print out the backtrace for each thread
    // before the timeout is hit (i.e. killed by CTest) so we can potentially
    // diagnose where the code is stuck
    auto _ci = get_env("ROCPROFSYS_CI", false, false);
    if(_ci)
    {
        // set by CTest
        auto _ci_timeout_default = get_env("ROCPROFSYS_CI_TIMEOUT", -1.0, false);
        // allow override by user
        auto _ci_timeout_seconds =
            get_env("ROCPROFSYS_CI_TIMEOUT_OVERRIDE", _ci_timeout_default, false);

        if(_ci_timeout_seconds > 0.0)
        {
            // lock served its purpose after setting to true
            ci_timeout_active = true;
            _lk.unlock();

            ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
            ROCPROFSYS_SCOPED_SAMPLING_ON_CHILD_THREADS(false);

            // enable the signal handler for when the timeout is reached
            struct sigaction _action = {};
            sigemptyset(&_action.sa_mask);
            _action.sa_flags   = SA_RESTART;
            _action.sa_handler = ci_timeout_backtrace;
            sigaction(timeout_signal_v, &_action, nullptr);

            // start a background thread that handles waiting for the timeout
            auto _ci_timeout_ready = std::promise<void>{};
            auto _ci_timeout_wait  = _ci_timeout_ready.get_future();
            std::thread{ ensure_ci_timeout_backtrace, _ci_timeout_seconds,
                         std::move(_ci_timeout_ready) }
                .detach();
            _ci_timeout_wait.wait_for(std::chrono::seconds{ 1 });
        }
    }
}
}  // namespace timeout
}  // namespace rocprofsys
