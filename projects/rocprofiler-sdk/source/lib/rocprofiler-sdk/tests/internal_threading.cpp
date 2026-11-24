// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/internal_threading.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/internal_threading.h>

#include <PTL/ThreadPool.hh>
#include <PTL/Types.hh>

#include <gtest/gtest.h>

#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <future>
#include <mutex>
#include <thread>

namespace
{
namespace internal_threading = ::rocprofiler::internal_threading;

using namespace std::chrono_literals;

constexpr auto lock_hold_time = 200ms;
constexpr auto child_timeout  = 10s;

// Runs one task on the rebuilt default callback thread. _exit skips the inherited atexit
// finalization, which a forked child that never execs must not run.
[[noreturn]] void
child_main()
{
    auto* group = internal_threading::get_task_group(rocprofiler_callback_thread_t{0});
    if(!group) _exit(2);

    auto ran = std::atomic<bool>{false};
    group->exec([&ran]() { ran = true; });
    group->wait();
    _exit(ran ? 0 : 3);
}

::testing::AssertionResult
fork_and_wait_for_child()
{
    pid_t pid = fork();
    if(pid < 0) return ::testing::AssertionFailure() << "fork failed: errno " << errno;
    if(pid == 0) child_main();

    const auto deadline = std::chrono::steady_clock::now() + child_timeout;
    int        status   = 0;
    while(true)
    {
        pid_t ret = waitpid(pid, &status, WNOHANG);
        if(ret == pid) break;
        if(ret < 0) return ::testing::AssertionFailure() << "waitpid failed: errno " << errno;
        if(std::chrono::steady_clock::now() > deadline)
        {
            kill(pid, SIGKILL);
            waitpid(pid, &status, 0);
            return ::testing::AssertionFailure()
                   << "forked child still blocked in the fork child handler after timeout";
        }
        std::this_thread::sleep_for(1ms);
    }

    if(WIFEXITED(status) && WEXITSTATUS(status) == 0) return ::testing::AssertionSuccess();
    if(WIFEXITED(status))
        return ::testing::AssertionFailure() << "child exited with " << WEXITSTATUS(status);
    return ::testing::AssertionFailure() << "child killed by signal " << WTERMSIG(status);
}

struct creation_gate
{
    std::atomic<bool> armed   = false;
    std::atomic<bool> entered = false;
};

creation_gate&
get_creation_gate()
{
    static auto gate = creation_gate{};
    return gate;
}

// Blocks the next internal thread creation only, so it stays inside the creation callback (and
// the lock held around it) while the test forks.
void
blocking_precreate(rocprofiler_runtime_library_t, void* data)
{
    auto* gate = static_cast<creation_gate*>(data);
    if(!gate->armed.exchange(false)) return;
    gate->entered = true;
    std::this_thread::sleep_for(lock_hold_time);
}
}  // namespace

TEST(internal_threading, fork_while_thread_pool_lock_held)
{
    internal_threading::initialize();

    auto held   = std::promise<void>{};
    auto holder = std::thread{[&held]() {
        auto lk = std::unique_lock<std::mutex>{PTL::TypeMutex<PTL::ThreadPool>()};
        held.set_value();
        std::this_thread::sleep_for(lock_hold_time);
    }};
    held.get_future().wait();

    EXPECT_TRUE(fork_and_wait_for_child());
    holder.join();
}

TEST(internal_threading, fork_while_thread_creation_callback_runs)
{
    internal_threading::initialize();

    // registered once per process: creation callbacks cannot be removed
    static auto registered = rocprofiler_at_internal_thread_create(
        blocking_precreate, nullptr, ROCPROFILER_LIBRARY, &get_creation_gate());
    ASSERT_EQ(registered, ROCPROFILER_STATUS_SUCCESS);

    auto& gate   = get_creation_gate();
    gate.entered = false;
    gate.armed   = true;
    auto creator = std::thread{[]() { auto group = internal_threading::create_task_group(1); }};

    const auto deadline = std::chrono::steady_clock::now() + child_timeout;
    while(!gate.entered && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(1ms);
    if(!gate.entered)
    {
        creator.join();
        FAIL() << "creation callback never ran";
    }

    EXPECT_TRUE(fork_and_wait_for_child());
    creator.join();
}
