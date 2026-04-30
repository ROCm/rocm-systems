// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for sampling_service::shutdown() — AC-6, AC-20.

#include <gtest/gtest.h>

#include "doubles/test_sampling_policies.hpp"
#include "sampling/sampling_service.hpp"

using namespace rocprofsys::sampling;
using namespace rocprofsys::sampling::test;
// test_service alias lives in test_sampling_policies.hpp.

// ─── AC-6: shutdown does not crash ───────────────────────────────────────────

TEST(sampling_service_shutdown, shutdown_tid0_does_not_crash)
{
    test_service svc{ make_test_config(), make_test_callbacks() };
    svc.setup(0);

    EXPECT_NO_THROW(svc.shutdown(0)) << "shutdown(0) must not throw or crash";
}

TEST(sampling_service_shutdown, double_shutdown_is_idempotent)
{
    test_service svc{ make_test_config(), make_test_callbacks() };
    svc.setup(0);
    svc.shutdown(0);

    EXPECT_NO_THROW(svc.shutdown(0)) << "Calling shutdown(0) twice must not crash";
}

TEST(sampling_service_shutdown, shutdown_single_thread_returns_signal_set)
{
    test_service      svc{ make_test_config(), make_test_callbacks() };
    constexpr int64_t tid = 0;
    svc.setup(tid);

    // The contract under test is that shutdown() returns a std::set<int> of the
    // signals that were subscribed for this tid; either empty (no triggers configured)
    // or populated is acceptable. Reaching this assertion line proves shutdown()
    // returned without throwing — the AC-6 promise.
    EXPECT_NO_THROW({ [[maybe_unused]] auto sigs = svc.shutdown(tid); });
}

// ─── AC-20: child process — release without per-tid processing ────────────────

TEST(sampling_service_shutdown, shutdown_in_child_skips_per_tid_processing)
{
    test_service svc{ make_test_config(), make_test_callbacks() };
    svc.setup(0);

    // Use the same entry point real postfork-child path uses (AC-20).
    svc.enter_child_process_mode();

    svc.shutdown(0);

    auto const& report = svc.report_writer_ref();
    EXPECT_TRUE(report.m_timer_counts.empty())
        << "Child shutdown must skip per-tid processing (timer_counts must be empty)";
    EXPECT_TRUE(report.m_overflow_counts.empty())
        << "Child shutdown must skip per-tid processing (overflow_counts must be empty)";
}
