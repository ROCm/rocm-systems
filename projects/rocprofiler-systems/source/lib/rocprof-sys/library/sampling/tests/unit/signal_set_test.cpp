// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Direct tests for signal_set — a sigset_t value type.
// NFR-PORT-3: this file is Linux-only (uses POSIX sigset_t API) and is
// compiled only on Linux (enforced by CMake ROCPROFSYS_LINUX_SAMPLING gate).

#include <gtest/gtest.h>

#include "sampling/src/linux/signal_set.hpp"

#include <csignal>
#include <set>

using namespace rocprofsys::sampling;

// ─── Empty set ─────────────────────────────────────────────────────────────

TEST(signal_set, default_constructed_set_is_empty)
{
    signal_set ss;
    // sigismember on every standard signal returns 0 for empty set.
    EXPECT_EQ(sigismember(ss.get(), SIGUSR1), 0)
        << "Default-constructed signal_set must have no signals";
    EXPECT_EQ(sigismember(ss.get(), SIGTERM), 0)
        << "Default-constructed signal_set must have no signals";
}

// ─── std::set<int> constructor ────────────────────────────────────────────

TEST(signal_set, constructed_from_set_contains_given_signals)
{
    std::set<int> sigs = { SIGUSR1, SIGUSR2 };
    signal_set    ss(sigs);

    EXPECT_EQ(sigismember(ss.get(), SIGUSR1), 1) << "SIGUSR1 must be in the set";
    EXPECT_EQ(sigismember(ss.get(), SIGUSR2), 1) << "SIGUSR2 must be in the set";
}

TEST(signal_set, constructed_from_set_excludes_absent_signals)
{
    std::set<int> sigs = { SIGUSR1 };
    signal_set    ss(sigs);

    EXPECT_EQ(sigismember(ss.get(), SIGTERM), 0) << "SIGTERM must NOT be in the set";
    EXPECT_EQ(sigismember(ss.get(), SIGUSR2), 0) << "SIGUSR2 must NOT be in the set";
}

TEST(signal_set, empty_std_set_produces_empty_signal_set)
{
    std::set<int> sigs;
    signal_set    ss(sigs);

    EXPECT_EQ(sigismember(ss.get(), SIGUSR1), 0)
        << "Signal_set from empty std::set must contain no signals";
}

// ─── get() non-const and const pointers ──────────────────────────────────

TEST(signal_set, get_returns_non_null_pointer)
{
    signal_set ss;
    EXPECT_NE(ss.get(), nullptr) << "get() must return non-null";
}

TEST(signal_set, const_get_returns_non_null_pointer)
{
    signal_set const ss;
    EXPECT_NE(ss.get(), nullptr) << "const get() must return non-null";
}

TEST(signal_set, mutable_and_const_get_point_to_same_sigset)
{
    signal_set ss({ SIGUSR1 });
    // const cast to verify same address.
    signal_set const& css = ss;
    EXPECT_EQ(ss.get(), css.get())
        << "mutable and const get() must point to same sigset_t";
}

// ─── Multiple signals round-trip ─────────────────────────────────────────

TEST(signal_set, multiple_realtime_signals_round_trip)
{
    std::set<int> sigs;
    // SIGRTMIN through SIGRTMIN+3 are commonly used for sampling.
    for(int i = 0; i < 4 && SIGRTMIN + i <= SIGRTMAX; ++i)
        sigs.insert(SIGRTMIN + i);

    signal_set ss(sigs);
    for(int sig : sigs)
    {
        EXPECT_EQ(sigismember(ss.get(), sig), 1)
            << "Signal " << sig << " must be in the signal_set";
    }
}
