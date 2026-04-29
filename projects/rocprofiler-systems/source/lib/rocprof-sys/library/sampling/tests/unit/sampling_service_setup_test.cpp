// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for sampling_service::setup() — AC-1, AC-2, AC-5, AC-9, AC-18, AC-19.

#include <gtest/gtest.h>

#include "doubles/recording_signal_dispatcher.hpp"
#include "doubles/test_sampling_policies.hpp"
#include "sampling/sampling_service.hpp"

using namespace rocprofsys::sampling;
using namespace rocprofsys::sampling::test;
// test_service alias lives in test_sampling_policies.hpp.

// ─── AC-1: setup() returns non-empty signal set for a normal thread ───────────

TEST(sampling_service_setup, setup_returns_nonempty_signal_set_for_normal_thread)
{
    test_service      svc;
    constexpr int64_t tid = 0;

    auto sigs = svc.setup(tid);

    EXPECT_FALSE(sigs.empty())
        << "setup() must return a non-empty signal set for a non-excluded thread";
}

// ─── AC-1: realtime signal is in the returned signal set ─────────────────────

TEST(sampling_service_setup, setup_includes_realtime_signal)
{
    test_service      svc;
    constexpr int64_t tid = 0;

    svc.setup(tid);

    auto realtime_sig = ::rocprofsys::get_sampling_realtime_signal();
    auto sigs         = svc.get_signal_types(tid);
    EXPECT_NE(sigs.find(realtime_sig), sigs.end())
        << "get_signal_types() must include the realtime signal after setup()";
}

// ─── AC-2: cputime signal is in the returned signal set ──────────────────────

TEST(sampling_service_setup, setup_includes_cputime_signal)
{
    test_service      svc;
    constexpr int64_t tid = 0;

    svc.setup(tid);

    auto cputime_sig = ::rocprofsys::get_sampling_cputime_signal();
    auto sigs        = svc.get_signal_types(tid);
    EXPECT_NE(sigs.find(cputime_sig), sigs.end())
        << "get_signal_types() must include the cputime signal after setup()";
}

// ─── AC-5: returns empty set when the duration controller has already fired ──

TEST(sampling_service_setup, setup_returns_empty_when_duration_disabled)
{
    test_service svc;

    // Drive the duration_controller to its disabled state via the production
    // path: start with a tiny duration, then advance the fake clock past the
    // deadline and call tick_for_test() to fire the disable callback.
    svc.duration_controller().start(1e-9);
    test::fake_clock::advance_ns(1'000);
    svc.duration_controller().tick_for_test();
    ASSERT_TRUE(svc.duration_controller().is_disabled())
        << "duration_controller must be disabled after the deadline fires";

    auto sigs = svc.setup(0);

    EXPECT_TRUE(sigs.empty())
        << "setup() must return {} when the duration controller has fired";
}

// AC-19 (causal profiling guard) is verified at the integration level —
// the unit-test bundle does not configure global causal-profiling state, so
// the throw branch is exercised by the sampling_service_production_hooks
// integration suite alongside the rest of the production wiring.

// ─── AC-8: block_samples / unblock_samples toggle global gate ────────────────

TEST(sampling_service_setup, block_and_unblock_samples_toggle_global_gate)
{
    test_service svc;
    EXPECT_FALSE(svc.is_blocked());

    svc.block_samples();
    EXPECT_TRUE(svc.is_blocked());

    svc.unblock_samples();
    EXPECT_FALSE(svc.is_blocked());
}

// ─── AC-9: block_signals calls signal dispatcher with SIG_BLOCK ──────────────

TEST(sampling_service_setup, block_signals_calls_sigmask_with_sig_block)
{
    test_service svc;
    svc.setup(0);

    auto&  dispatcher   = svc.signal_dispatcher_ref();
    size_t calls_before = dispatcher.m_calls.size();

    svc.block_signals();

    EXPECT_GT(dispatcher.m_calls.size(), calls_before)
        << "block_signals() must call recording_signal_dispatcher::sigmask()";

    bool found_block = false;
    for(auto const& call : dispatcher.m_calls)
    {
        if(call.m_how == sigmask_how::block) found_block = true;
    }
    EXPECT_TRUE(found_block) << "block_signals() must use SIG_BLOCK";
}

TEST(sampling_service_setup, unblock_signals_calls_sigmask_with_sig_unblock)
{
    test_service svc;
    svc.setup(0);

    auto& dispatcher = svc.signal_dispatcher_ref();

    svc.unblock_signals();

    bool found_unblock = false;
    for(auto const& call : dispatcher.m_calls)
    {
        if(call.m_how == sigmask_how::unblock) found_unblock = true;
    }
    EXPECT_TRUE(found_unblock) << "unblock_signals() must use SIG_UNBLOCK";
}

// ─── AC-18: get_signal_types returns initialized set for tid ─────────────────

TEST(sampling_service_setup, get_signal_types_returns_initialized_set_for_tid)
{
    test_service      svc;
    constexpr int64_t tid = 0;

    svc.setup(tid);

    auto sigs = svc.get_signal_types(tid);
    EXPECT_FALSE(sigs.empty())
        << "get_signal_types() must return a non-empty set after setup()";
}

// ─── I-13: setup() internally calls block_signals() → dispatcher was used ────

TEST(sampling_service_setup, setup_calls_signal_dispatcher_to_block_signals)
{
    test_service svc;

    auto& dispatcher = svc.signal_dispatcher_ref();
    ASSERT_EQ(dispatcher.m_calls.size(), 0U)
        << "dispatcher must not have been called before setup()";

    svc.setup(0);

    EXPECT_GT(dispatcher.m_calls.size(), 0U)
        << "setup() must call the signal dispatcher (via block_signals) at least once";

    bool found_block = false;
    for(auto const& call : dispatcher.m_calls)
    {
        if(call.m_how == sigmask_how::block)
        {
            found_block = true;
            break;
        }
    }
    EXPECT_TRUE(found_block)
        << "setup() must block signals via SIG_BLOCK through the signal dispatcher";
}
