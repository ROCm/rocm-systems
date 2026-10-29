// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "lib/common/fork_signal_state.hpp"
#include <gtest/gtest.h>
#include <sys/eventfd.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cerrno>
#include <cstdlib>

namespace
{
using rocprofiler::common::fork_signal_mask;
using rocprofiler::common::fork_signal_state;
fork_signal_state*    current   = nullptr;
int                   signal_fd = -1;
volatile sig_atomic_t rejected  = 0;

void
handler(int)
{
    if(!current->usable(getpid()))
    {
        rejected = 1;
        return;
    }
    uint64_t value = 1;
    if(write(signal_fd, &value, sizeof(value)) != sizeof(value)) std::_Exit(31);
}

void
run(bool masked)
{
    alarm(5);
    fork_signal_state state{};
    state.ready(getpid());
    current   = &state;
    signal_fd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    if(signal_fd < 0) std::_Exit(32);
    const auto       parent_fd = signal_fd;
    const int        signo     = masked ? SIGTERM : SIGABRT;
    struct sigaction action
    {};
    action.sa_handler = handler;
    sigemptyset(&action.sa_mask);
    if(sigaction(signo, &action, nullptr) != 0) std::_Exit(33);
    sigset_t handled{};
    sigemptyset(&handled);
    sigaddset(&handled, signo);
    pthread_sigmask(SIG_UNBLOCK, &handled, nullptr);
    fork_signal_mask mask{};
    if(masked && mask.prepare(handled) != 0) std::_Exit(34);
    const auto child = fork();
    if(child == 0)
    {
        if(masked)
        {
            state.begin_child();
            raise(signo);
            sigset_t pending{};
            sigpending(&pending);
            if(!sigismember(&pending, signo)) std::_Exit(35);
            signal_fd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
            if(signal_fd < 0) std::_Exit(36);
            state.ready(getpid());
            mask.child();
            uint64_t value = 0;
            if(read(signal_fd, &value, sizeof(value)) != sizeof(value) || value != 1)
                std::_Exit(37);
        }
        else
        {
            // Deliberately signal before begin_child(): this is the pre-atfork-callback hole.
            raise(signo);
            if(!rejected) std::_Exit(38);
        }
        uint64_t inherited = 0;
        errno              = 0;
        if(read(parent_fd, &inherited, sizeof(inherited)) >= 0 || errno != EAGAIN) std::_Exit(39);
        std::_Exit(0);
    }
    if(masked) mask.parent();
    int status = 0;
    if(child < 0 || waitpid(child, &status, 0) != child || !WIFEXITED(status) ||
       WEXITSTATUS(status) != 0)
        std::_Exit(40);
    uint64_t value = 0;
    errno          = 0;
    if(read(parent_fd, &value, sizeof(value)) >= 0 || errno != EAGAIN) std::_Exit(41);
    std::_Exit(0);
}
}  // namespace

TEST(fork_signal_state, inherited_worker_is_rejected_before_child_callback)
{
    ASSERT_EXIT(run(false), ::testing::ExitedWithCode(0), "");
}

TEST(fork_signal_state, pending_signal_reaches_only_the_fresh_child_eventfd)
{
    ASSERT_EXIT(run(true), ::testing::ExitedWithCode(0), "");
}

TEST(fork_signal_state, readiness_requires_current_pid_and_completed_transition)
{
    fork_signal_state state{};
    EXPECT_FALSE(state.usable(getpid()));
    state.ready(getpid());
    EXPECT_TRUE(state.usable(getpid()));
    state.begin_child();
    EXPECT_FALSE(state.usable(getpid()));
    state.ready(getpid() + 1);
    EXPECT_FALSE(state.usable(getpid()));
    state.ready(getpid());
    EXPECT_TRUE(state.usable(getpid()));
}

TEST(fork_signal_state, nested_parent_masks_restore_the_outer_original_mask)
{
    sigset_t original{};
    pthread_sigmask(SIG_SETMASK, nullptr, &original);
    struct restore_mask
    {
        sigset_t original;
        ~restore_mask() { pthread_sigmask(SIG_SETMASK, &original, nullptr); }
    } restore{original};
    sigset_t handled{};
    sigemptyset(&handled);
    sigaddset(&handled, SIGTERM);
    pthread_sigmask(SIG_UNBLOCK, &handled, nullptr);
    fork_signal_mask mask{};
    ASSERT_EQ(mask.prepare(handled), 0);
    ASSERT_EQ(mask.prepare(handled), 0);
    mask.parent();
    sigset_t now{};
    pthread_sigmask(SIG_SETMASK, nullptr, &now);
    EXPECT_TRUE(sigismember(&now, SIGTERM));
    mask.parent();
    pthread_sigmask(SIG_SETMASK, nullptr, &now);
    EXPECT_FALSE(sigismember(&now, SIGTERM));
}

namespace
{
struct owned_signal_records : rocprofiler::common::signal_dispositions<NSIG>
{
    ~owned_signal_records()
    {
        // Production records deliberately outlive signal readers. These local unit instances
        // have no asynchronous readers and can reclaim their bounded immutable chains.
        for(int signo = 0; signo < NSIG; ++signo)
        {
            const auto* node = load(signo);
            while(node)
            {
                const auto* previous = node->previous;
                delete node;
                node = previous;
            }
        }
    }
};
}  // namespace

TEST(signal_dispositions, first_application_snapshot_replaces_bootstrap_once)
{
    owned_signal_records records{};
    records.save(SIGTERM, {SIG_DFL, std::nullopt, true});
    const auto* bootstrap = records.load(SIGTERM);
    records.save(SIGTERM, {SIG_IGN, std::nullopt, false});
    const auto* application = records.load(SIGTERM);
    EXPECT_NE(application, bootstrap);
    EXPECT_EQ(bootstrap->handler, SIG_DFL);  // old published copies remain immutable and valid
    EXPECT_EQ(application->handler, SIG_IGN);
    records.save(SIGTERM, {SIG_DFL, std::nullopt, false});
    EXPECT_EQ(records.load(SIGTERM), application);
}

TEST(signal_dispositions, explicit_application_default_is_not_a_chained_handler)
{
    owned_signal_records records{};
    records.save(SIGABRT, {SIG_IGN, std::nullopt, true});
    EXPECT_TRUE(records.chained(SIGABRT));
    records.save(SIGABRT, {SIG_DFL, std::nullopt, false});
    EXPECT_FALSE(records.chained(SIGABRT));
}

TEST(signal_dispositions, sigaction_snapshot_preserves_flags_and_mask)
{
    owned_signal_records records{};
    struct sigaction     action
    {};
    action.sa_handler = SIG_IGN;
    action.sa_flags   = SA_RESTART;
    sigemptyset(&action.sa_mask);
    sigaddset(&action.sa_mask, SIGUSR1);
    records.save(SIGTERM, {nullptr, action, false});
    const auto* state = records.load(SIGTERM);
    ASSERT_TRUE(state->action);
    EXPECT_TRUE(state->action->sa_flags & SA_RESTART);
    EXPECT_TRUE(sigismember(&state->action->sa_mask, SIGUSR1));
}
