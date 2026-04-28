// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Direct tests for signal_mask_guard<FatalErrorPolicy>.
// These tests call pthread_sigmask to verify RAII behavior.
// Linux-only: compiled only when ROCPROFSYS_LINUX_SAMPLING is set.

#include <gtest/gtest.h>

#include "doubles/throwing_fatal_error_policy.hpp"
#include "sampling/src/signal_mask_guard.hpp"

#include <csignal>
#include <set>

using namespace rocprofsys::sampling;
using namespace rocprofsys::sampling::test;

// Helper: query whether signal `sig` is currently blocked on this thread.
static bool
is_signal_blocked(int sig)
{
    sigset_t current;
    ::pthread_sigmask(SIG_SETMASK, nullptr, &current);
    return sigismember(&current, sig) != 0;
}

// ─── Constructor blocks the given signals ─────────────────────────────────

TEST(signal_mask_guard, constructor_blocks_signals)
{
    throwing_fatal_error_policy fatal;
    const std::set<int>         sigs = { SIGUSR1, SIGUSR2 };

    // Ensure signals are unblocked before the test.
    {
        sigset_t unblock;
        sigemptyset(&unblock);
        sigaddset(&unblock, SIGUSR1);
        sigaddset(&unblock, SIGUSR2);
        ::pthread_sigmask(SIG_UNBLOCK, &unblock, nullptr);
    }

    {
        signal_mask_guard<throwing_fatal_error_policy> guard(sigs, SIG_BLOCK, fatal);
        EXPECT_TRUE(is_signal_blocked(SIGUSR1))
            << "SIGUSR1 must be blocked while guard is alive";
        EXPECT_TRUE(is_signal_blocked(SIGUSR2))
            << "SIGUSR2 must be blocked while guard is alive";
    }

    // After destruction, signals must be restored (unblocked).
    EXPECT_FALSE(is_signal_blocked(SIGUSR1))
        << "SIGUSR1 must be unblocked after guard destruction";
    EXPECT_FALSE(is_signal_blocked(SIGUSR2))
        << "SIGUSR2 must be unblocked after guard destruction";
}

// ─── Destructor restores the old mask ────────────────────────────────────

TEST(signal_mask_guard, destructor_restores_previous_mask)
{
    throwing_fatal_error_policy fatal;

    // Block SIGUSR2 before the guard.
    {
        sigset_t pre;
        sigemptyset(&pre);
        sigaddset(&pre, SIGUSR2);
        ::pthread_sigmask(SIG_BLOCK, &pre, nullptr);
    }

    {
        // Block SIGUSR1 with the guard.
        signal_mask_guard<throwing_fatal_error_policy> guard({ SIGUSR1 }, SIG_BLOCK,
                                                             fatal);
        EXPECT_TRUE(is_signal_blocked(SIGUSR1)) << "SIGUSR1 blocked by guard";
        EXPECT_TRUE(is_signal_blocked(SIGUSR2))
            << "SIGUSR2 must remain blocked (pre-existing)";
    }

    // Guard destroyed: SIGUSR1 restored (was unblocked before guard).
    EXPECT_FALSE(is_signal_blocked(SIGUSR1))
        << "SIGUSR1 must be unblocked after guard destruction";
    // SIGUSR2 was blocked before the guard — the restored mask had it blocked.
    EXPECT_TRUE(is_signal_blocked(SIGUSR2))
        << "SIGUSR2 was blocked before guard; must still be blocked after";

    // Cleanup: unblock SIGUSR2.
    sigset_t cleanup;
    sigemptyset(&cleanup);
    sigaddset(&cleanup, SIGUSR2);
    ::pthread_sigmask(SIG_UNBLOCK, &cleanup, nullptr);
}

// ─── release() disarms the RAII guard ────────────────────────────────────

TEST(signal_mask_guard, release_disarms_destructor)
{
    throwing_fatal_error_policy fatal;

    // Ensure SIGUSR1 is unblocked before.
    {
        sigset_t unblock;
        sigemptyset(&unblock);
        sigaddset(&unblock, SIGUSR1);
        ::pthread_sigmask(SIG_UNBLOCK, &unblock, nullptr);
    }

    {
        signal_mask_guard<throwing_fatal_error_policy> guard({ SIGUSR1 }, SIG_BLOCK,
                                                             fatal);
        EXPECT_TRUE(is_signal_blocked(SIGUSR1)) << "SIGUSR1 blocked by guard";
        guard.release();
        // After release(), destruction must NOT restore the old mask.
    }

    // Guard was released — signal remains blocked.
    EXPECT_TRUE(is_signal_blocked(SIGUSR1))
        << "After release(), destructor must not restore mask: SIGUSR1 must stay blocked";

    // Cleanup.
    sigset_t cleanup;
    sigemptyset(&cleanup);
    sigaddset(&cleanup, SIGUSR1);
    ::pthread_sigmask(SIG_UNBLOCK, &cleanup, nullptr);
}

// ─── Empty signal set — no-op block, no crash ────────────────────────────

TEST(signal_mask_guard, empty_signal_set_does_not_crash)
{
    throwing_fatal_error_policy fatal;
    EXPECT_NO_THROW({
        signal_mask_guard<throwing_fatal_error_policy> guard({}, SIG_BLOCK, fatal);
    }) << "signal_mask_guard with empty set must not crash";
}
