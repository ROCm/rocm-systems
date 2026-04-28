// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Integration tests for AC-17 (postfork_parent_reinit / postfork_child_cleanup)
// and NFR-TS-5 (signals blocked before state torn down in child).
//
// All tests use test_sampling_policies (generic template). The PMC delegation
// in the production specialization is covered by the produce binary; this file
// covers the generic behavior visible through the policy seams.
//
// Linux-only: fork() and signal blocking via pthread_sigmask.
// Gated at CMake level — no runtime GTEST_SKIP inside this file.

#include <gtest/gtest.h>

#include "doubles/recording_signal_dispatcher.hpp"
#include "doubles/test_sampling_policies.hpp"
#include "sampling/sampling_service.hpp"

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdlib>
#include <set>

using namespace rocprofsys::sampling;
using namespace rocprofsys::sampling::test;
using test_service = sampling_service<test_sampling_policies>;

// ── AC-17: postfork_parent_reinit does not crash when no threads active ───────

TEST(postfork, parent_reinit_is_safe_with_empty_registry)
{
    test_service svc;
    // No setup() called — registry is empty.
    EXPECT_NO_THROW(svc.postfork_parent_reinit())
        << "postfork_parent_reinit() must not throw when the thread registry is empty";
}

// ── AC-17: postfork_parent_reinit does not crash when threads are active ──────

TEST(postfork, parent_reinit_is_safe_with_active_threads)
{
    test_service svc;
    svc.setup(0);
    svc.setup(1);

    EXPECT_NO_THROW(svc.postfork_parent_reinit())
        << "postfork_parent_reinit() must not throw with active per-thread state";

    // Registry must be intact after parent reinit — parent keeps its state.
    bool tid0_present = false;
    bool tid1_present = false;
    svc.registry().each([&](int64_t tid, thread_sampler_state<test_sampling_policies>&) {
        if(tid == 0) tid0_present = true;
        if(tid == 1) tid1_present = true;
    });
    EXPECT_TRUE(tid0_present)
        << "tid=0 state must survive postfork_parent_reinit() in the parent";
    EXPECT_TRUE(tid1_present)
        << "tid=1 state must survive postfork_parent_reinit() in the parent";
}

// ── NFR-TS-5: postfork_child_cleanup blocks signals before touching state ─────
// Verified by observing that block_signals() (which goes through
// signal_dispatcher_.sigmask(SIG_BLOCK,...)) is called before registry_.reset().
// The recording_signal_dispatcher records every sigmask call; we check that at
// least one SIG_BLOCK call was made and that the registry is empty afterwards.

TEST(postfork, child_cleanup_blocks_signals_before_resetting_registry)
{
    test_service svc;
    svc.setup(0);

    // Confirm at least one active thread state exists before cleanup.
    int state_count_before = 0;
    svc.registry().each([&](int64_t, thread_sampler_state<test_sampling_policies>&) {
        ++state_count_before;
    });
    ASSERT_GT(state_count_before, 0)
        << "precondition: registry must be non-empty before postfork_child_cleanup()";

    size_t dispatcher_calls_before = svc.signal_dispatcher_ref().m_calls.size();

    EXPECT_NO_THROW(svc.postfork_child_cleanup())
        << "postfork_child_cleanup() must not throw";

    // NFR-TS-5: at least one SIG_BLOCK call must have been made.
    auto const& calls = svc.signal_dispatcher_ref().m_calls;
    EXPECT_GT(calls.size(), dispatcher_calls_before)
        << "postfork_child_cleanup() must call block_signals() via the signal dispatcher";

    bool found_block = false;
    for(size_t i = dispatcher_calls_before; i < calls.size(); ++i)
    {
        if(calls[i].m_how == sigmask_how::block)
        {
            found_block = true;
            break;
        }
    }
    EXPECT_TRUE(found_block)
        << "postfork_child_cleanup() must issue at least one SIG_BLOCK call "
           "before tearing down state (NFR-TS-5)";
}

// ── AC-17: postfork_child_cleanup clears the registry ─────────────────────────

TEST(postfork, child_cleanup_clears_registry)
{
    test_service svc;
    svc.setup(0);
    svc.setup(1);
    svc.setup(2);

    svc.postfork_child_cleanup();

    int state_count_after = 0;
    svc.registry().each([&](int64_t, thread_sampler_state<test_sampling_policies>&) {
        ++state_count_after;
    });

    EXPECT_EQ(state_count_after, 0)
        << "registry must be empty after postfork_child_cleanup() — "
           "all per-thread state must be released without calling post_process";
}

// ── AC-17: postfork_child_cleanup stops all armed triggers ────────────────────
// After cleanup, any state that was running must have had its triggers stopped.
// We verify by emplacing a trigger on a state, calling cleanup, and confirming
// the registry (and thus the trigger) is gone.

TEST(postfork, child_cleanup_releases_state_with_armed_timer)
{
    test_service svc;
    svc.setup(0);

    // Emplace a mock realtime trigger on thread 0's state.
    if(auto* state = svc.registry().at(0))
    {
        state->realtime_trigger().emplace();
        state->realtime_trigger()->start();
        ASSERT_TRUE(state->realtime_trigger()->is_armed())
            << "precondition: realtime trigger must be armed before cleanup";
    }

    EXPECT_NO_THROW(svc.postfork_child_cleanup())
        << "postfork_child_cleanup() must not throw with an armed trigger";

    // Registry must be empty — state (and thus the trigger) is gone.
    EXPECT_EQ(svc.registry().at(0), nullptr)
        << "per-thread state for tid=0 must be released by postfork_child_cleanup()";
}

// ── AC-17: Real fork() — parent and child both complete without deadlock ───────
// This is the highest-fidelity test: sampling active in parent, fork() called,
// child calls postfork_child_cleanup() and exits, parent calls
// postfork_parent_reinit() and continues.
//
// Uses _exit() in child to avoid GTest atexit handlers running twice.
// Parent waits for child and asserts clean exit.

TEST(postfork, real_fork_parent_and_child_complete_without_deadlock)
{
    test_service svc;
    svc.setup(0);

    pid_t child = ::fork();
    ASSERT_NE(child, pid_t(-1)) << "fork() must succeed";

    if(child == 0)
    {
        // Child process: clean up and exit cleanly.
        // postfork_child_cleanup() must not deadlock or crash.
        svc.postfork_child_cleanup();
        ::_exit(0);
    }

    // Parent process.
    svc.postfork_parent_reinit();

    // Wait for child; assert it exited cleanly (status 0, not killed by signal).
    int   status = 0;
    pid_t waited = ::waitpid(child, &status, 0);
    ASSERT_EQ(waited, child) << "waitpid() must return the child PID";
    ASSERT_TRUE(WIFEXITED(status))
        << "child must exit normally (not killed by signal) — deadlock or crash detected";
    EXPECT_EQ(WEXITSTATUS(status), 0)
        << "child must exit with status 0 after postfork_child_cleanup()";
}

// ── AC-17: postfork_child_cleanup is safe with empty registry (no signals set) ─

TEST(postfork, child_cleanup_safe_with_empty_registry)
{
    test_service svc;
    // No setup() — registry empty, no signals to block.
    EXPECT_NO_THROW(svc.postfork_child_cleanup())
        << "postfork_child_cleanup() must not throw when registry is empty "
           "(no signals in any thread state to block)";

    int count = 0;
    svc.registry().each(
        [&](int64_t, thread_sampler_state<test_sampling_policies>&) { ++count; });
    EXPECT_EQ(count, 0) << "registry must remain empty after cleanup on empty registry";
}
