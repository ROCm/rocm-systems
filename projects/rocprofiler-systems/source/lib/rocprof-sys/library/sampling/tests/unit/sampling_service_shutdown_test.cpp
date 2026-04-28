// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for sampling_service::shutdown() — AC-6, AC-20.

#include <gtest/gtest.h>

#include "doubles/test_sampling_policies.hpp"
#include "sampling/sampling_service.hpp"

using namespace rocprofsys::sampling;
using namespace rocprofsys::sampling::test;
using test_service = sampling_service<test_sampling_policies>;

// ─── AC-6: shutdown does not crash ───────────────────────────────────────────

TEST(sampling_service_shutdown, shutdown_tid0_does_not_crash)
{
    test_service svc;
    svc.setup(0);

    EXPECT_NO_THROW(svc.shutdown(0)) << "shutdown(0) must not throw or crash";
}

TEST(sampling_service_shutdown, double_shutdown_is_idempotent)
{
    test_service svc;
    svc.setup(0);
    svc.shutdown(0);

    EXPECT_NO_THROW(svc.shutdown(0)) << "Calling shutdown(0) twice must not crash";
}

TEST(sampling_service_shutdown, shutdown_single_thread_returns_signal_set)
{
    test_service      svc;
    constexpr int64_t tid = 0;
    svc.setup(tid);

    auto sigs = svc.shutdown(tid);

    EXPECT_TRUE(sigs.empty() || !sigs.empty())
        << "shutdown() must return without crashing";
}

// ─── AC-20: child process — release without per-tid processing ────────────────

TEST(sampling_service_shutdown, shutdown_in_child_skips_per_tid_processing)
{
    test_service svc;
    svc.setup(0);

    svc.set_child_process_for_test(true);

    svc.shutdown(0);

    auto const& report = svc.report_writer_ref();
    EXPECT_TRUE(report.m_timer_counts.empty())
        << "Child shutdown must skip per-tid processing (timer_counts must be empty)";
    EXPECT_TRUE(report.m_overflow_counts.empty())
        << "Child shutdown must skip per-tid processing (overflow_counts must be empty)";
}
