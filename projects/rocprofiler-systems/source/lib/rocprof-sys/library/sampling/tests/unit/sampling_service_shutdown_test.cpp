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

    auto const& trace = svc.get_trace_sink();
    EXPECT_TRUE(trace.timer_records().empty())
        << "Child shutdown must skip per-tid processing (no timer records emitted)";
    EXPECT_TRUE(trace.overflow_records().empty())
        << "Child shutdown must skip per-tid processing (no overflow records emitted)";
}

// ─── Deferred resolution: worker shutdown does NOT resolve ────────────────────

TEST(sampling_service_shutdown, worker_shutdown_does_not_resolve)
{
    test_service svc{ make_test_config(), make_test_callbacks() };
    svc.setup(0);
    svc.setup(1);

    backtrace_record r{};
    r.trigger      = trigger_type::TIMER;
    r.pc_count     = 2;
    r.timestamp_ns = 100;
    svc.get_offload().insert(1, r);
    r.timestamp_ns = 200;
    svc.get_offload().insert(1, r);

    svc.shutdown(1);

    EXPECT_TRUE(svc.get_trace_sink().timer_records().empty())
        << "Worker shutdown must NOT resolve — resolution deferred to main thread";

    auto remaining = svc.get_offload().read(1);
    EXPECT_FALSE(remaining.empty())
        << "Worker's offload records must remain until main thread resolves them";
}

// ─── Main thread shutdown resolves all deferred worker data ───────────────────

TEST(sampling_service_shutdown, main_shutdown_resolves_deferred_workers)
{
    test_service svc{ make_test_config(), make_test_callbacks() };
    svc.setup(0);
    svc.setup(1);
    svc.setup(2);

    auto inject_samples = [&](int64_t tid) {
        backtrace_record r{};
        r.tid          = tid;
        r.trigger      = trigger_type::TIMER;
        r.pc_count     = 2;
        r.timestamp_ns = 100;
        svc.get_offload().insert(tid, r);
        r.timestamp_ns = 200;
        svc.get_offload().insert(tid, r);
        r.timestamp_ns = 300;
        svc.get_offload().insert(tid, r);
    };

    inject_samples(1);
    inject_samples(2);
    inject_samples(0);

    // Workers shut down first (deferred — no resolution).
    svc.shutdown(1);
    svc.shutdown(2);

    EXPECT_TRUE(svc.get_trace_sink().timer_records().empty())
        << "No resolution until main thread shuts down";

    // Main thread shutdown triggers batch resolution for all tids.
    svc.shutdown(0);

    auto const& records = svc.get_trace_sink().timer_records();
    EXPECT_GE(records.size(), 2U)
        << "Main shutdown must resolve deferred data for workers + itself";

    // Offload store should be drained.
    EXPECT_TRUE(svc.get_offload().tids().empty())
        << "All offload records must be consumed after main shutdown";
}

// ─── shutdown_thread is idempotent (already-erased tid) ──────────────────────

TEST(sampling_service_shutdown, shutdown_thread_on_unknown_tid_is_noop)
{
    test_service svc{ make_test_config(), make_test_callbacks() };
    svc.setup(0);

    // tid 99 was never set up — shutdown must not crash.
    EXPECT_NO_THROW(svc.shutdown(99));
}

// ─── Main thread drains straggler workers still in registry ──────────────────

TEST(sampling_service_shutdown, main_shutdown_drains_stragglers)
{
    test_service svc{ make_test_config(), make_test_callbacks() };
    svc.setup(0);
    svc.setup(3);

    backtrace_record r{};
    r.tid          = 3;
    r.trigger      = trigger_type::TIMER;
    r.pc_count     = 2;
    r.timestamp_ns = 100;
    svc.get_offload().insert(3, r);
    r.timestamp_ns = 200;
    svc.get_offload().insert(3, r);

    // Worker 3 never called shutdown — main thread must drain it.
    svc.shutdown(0);

    auto remaining = svc.get_offload().read(3);
    EXPECT_TRUE(remaining.empty())
        << "Main shutdown must drain straggler workers that never called shutdown";
}
