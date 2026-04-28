// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// TDD tests for sampling_service<Policies>::post_process().
//
// These tests are intentionally RED: post_process() is currently an empty stub.
// code-writer must implement the fan-out pipeline to make them green.
//
// Spec: architecture.md DEC-9/10/11, requirements.md AC-10/AC-11/AC-13.
//
// Test placement: integration (multi-component, but no OS dependencies).
// Policy bundle: post_process_policies (recording doubles, in-memory offload).
//
// Naming: all test doubles are lower_case per project clang-tidy rules.

#include <gtest/gtest.h>

#include "doubles/post_process_policies.hpp"
#include "sampling/sampling_service.hpp"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

using namespace rocprofsys::sampling;
using namespace rocprofsys::sampling::test;

// Alias for the service under test.
using pp_service = sampling_service<post_process_policies>;

// ── Helpers ──────────────────────────────────────────────────────────────────

// Build a backtrace_record with a given timestamp and enough PCs to survive
// the AC-11 single-sample discard filter (pc_count > 1).
static backtrace_record
make_timer_record(int64_t tid, uint64_t ts_ns, uint8_t pc_count = 2)
{
    backtrace_record r;
    r.tid          = tid;
    r.timestamp_ns = ts_ns;
    r.trigger      = trigger_type::TIMER;
    r.pc_count     = pc_count;
    for(uint8_t i = 0; i < pc_count; ++i)
        r.raw_pcs[i] = static_cast<uintptr_t>(0x1000 + i);
    return r;
}

static backtrace_record
make_overflow_record(int64_t tid, uint64_t ts_ns, uint8_t pc_count = 2)
{
    backtrace_record r;
    r.tid          = tid;
    r.timestamp_ns = ts_ns;
    r.trigger      = trigger_type::OVERFLOW;
    r.pc_count     = pc_count;
    for(uint8_t i = 0; i < pc_count; ++i)
        r.raw_pcs[i] = static_cast<uintptr_t>(0x2000 + i);
    return r;
}

// ── Test 1: Empty offload produces no output ──────────────────────────────────

TEST(post_process, empty_offload_produces_no_report_writer_calls)
{
    pp_service svc;
    svc.setup(0);

    // Offload is empty — no records injected.
    svc.post_process();

    auto& writer = svc.report_writer_ref();
    EXPECT_TRUE(writer.timer_calls.empty())
        << "report_writer.write_timer_samples() must not be called when offload is empty";
    EXPECT_TRUE(writer.overflow_calls.empty())
        << "report_writer.write_overflow_samples() must not be called when offload is "
           "empty";
}

TEST(post_process, empty_offload_flush_called_once)
{
    pp_service svc;
    svc.setup(0);

    svc.post_process();

    auto& writer = svc.report_writer_ref();
    EXPECT_EQ(writer.flush_count, 1) << "report_writer.flush() must be called exactly "
                                        "once per post_process() invocation";
}

TEST(post_process, empty_offload_reset_called_once)
{
    pp_service svc;
    svc.setup(0);

    svc.post_process();

    auto& offload = svc.get_offload();
    EXPECT_EQ(offload.reset_count, 1)
        << "offload.reset() must be called exactly once per post_process() invocation";
}

// ── Test 2: Single-bundle round-trip ─────────────────────────────────────────

// inject_records_for_post_process: pre-populate the in_memory_offload's m_store
// directly so post_process() will read them via offload_.read(tid).
// Two records are needed: init_rec (base timestamp) + one real record.
// AC-11 discards buffers with raw.size()==1 && pc_count<=1.
// With two records, parse_timer produces one sample: [rec0.timestamp, rec1.timestamp).

TEST(post_process, single_bundle_report_writer_called)
{
    pp_service svc;
    svc.setup(0);

    // Inject two timer records for tid 0: t=100 (init/base) and t=200 (sample).
    auto& offload = svc.get_offload();
    offload.inject(0, make_timer_record(0, 100));
    offload.inject(0, make_timer_record(0, 200));

    svc.post_process();

    auto& writer = svc.report_writer_ref();
    ASSERT_FALSE(writer.timer_calls.empty())
        << "report_writer.write_timer_samples() must be called for tid 0 after injecting "
           "2 records";
    EXPECT_EQ(writer.timer_calls.front().tid, 0)
        << "write_timer_samples must carry tid 0";
    EXPECT_FALSE(writer.timer_calls.front().samples.empty())
        << "write_timer_samples must deliver at least one parsed sample";
}

TEST(post_process, single_bundle_perfetto_sink_called)
{
    pp_service svc;
    svc.setup(0);

    auto& offload = svc.get_offload();
    offload.inject(0, make_timer_record(0, 100));
    offload.inject(0, make_timer_record(0, 200));

    svc.post_process();

    auto& psink = svc.get_perfetto_sink();
    ASSERT_FALSE(psink.timer_calls.empty())
        << "perfetto_sink.emit_timer() must be called for tid 0 after injecting 2 "
           "records";
    EXPECT_EQ(psink.timer_calls.front().tid, 0);
    EXPECT_FALSE(psink.timer_calls.front().samples.empty());
}

TEST(post_process, single_bundle_offload_reset_called)
{
    pp_service svc;
    svc.setup(0);

    auto& offload = svc.get_offload();
    offload.inject(0, make_timer_record(0, 100));
    offload.inject(0, make_timer_record(0, 200));

    svc.post_process();

    EXPECT_EQ(offload.reset_count, 1)
        << "offload.reset() must be called after draining all per-thread records";
}

// ── Test 3: Pause-interval filtering ─────────────────────────────────────────
//
// Inject 3 records for tid 0:
//   rec0: t=100 (base / init)
//   rec1: t=300 (sample1 — interval [100,300])
//   rec2: t=500 (sample2 — interval [300,500])
//   rec3: t=700 (sample3 — interval [500,700])
//
// Pause interval: [300, 500] — sample2 overlaps, so only sample1 and sample3 survive.

TEST(post_process, pause_interval_filters_overlapping_sample)
{
    pp_service svc;
    fake_clock::reset(0);
    svc.setup(0);

    // Record the pause interval via the service API.
    // pause() at t=300, resume() at t=500.
    fake_clock::reset(300);
    svc.pause();
    fake_clock::reset(500);
    svc.resume();

    auto& offload = svc.get_offload();
    // init + 3 sample records
    offload.inject(0, make_timer_record(0, 100));  // base
    offload.inject(0, make_timer_record(0, 300));  // sample1: [100,300] — before pause
    offload.inject(0, make_timer_record(0, 500));  // sample2: [300,500] — inside pause
    offload.inject(0, make_timer_record(0, 700));  // sample3: [500,700] — after pause

    svc.post_process();

    auto& writer = svc.report_writer_ref();
    ASSERT_FALSE(writer.timer_calls.empty())
        << "report_writer must be called after pause-filtered post_process";

    auto const& samples = writer.timer_calls.front().samples;
    EXPECT_EQ(samples.size(), 2U)
        << "pause-interval filter must drop sample2 ([300,500]); expected 2 samples, got "
        << samples.size();

    // sample1: [100,300]
    EXPECT_EQ(samples[0].beg_ns, 100U);
    EXPECT_EQ(samples[0].end_ns, 300U);
    // sample3: [500,700]
    EXPECT_EQ(samples[1].beg_ns, 500U);
    EXPECT_EQ(samples[1].end_ns, 700U);
}

// ── Test 4: Multi-thread fan-out ──────────────────────────────────────────────
//
// Inject timer records for tids 1, 2, 3.
// post_process() (no-arg) drains all.
// Expected: report_writer called once per thread, samples isolated per tid.

TEST(post_process, multi_tid_report_writer_called_per_thread)
{
    pp_service svc;
    for(int64_t tid : { 1, 2, 3 })
        svc.setup(tid);

    auto& offload = svc.get_offload();
    for(int64_t tid : { 1, 2, 3 })
    {
        offload.inject(tid, make_timer_record(tid, 100));
        offload.inject(tid, make_timer_record(tid, 200));
    }

    svc.post_process();

    auto& writer = svc.report_writer_ref();
    EXPECT_EQ(writer.timer_calls.size(), 3U)
        << "report_writer.write_timer_samples() must be called once per active thread; "
        << "expected 3 calls for tids 1,2,3";
}

TEST(post_process, multi_tid_samples_isolated_per_thread)
{
    pp_service svc;
    for(int64_t tid : { 1, 2, 3 })
        svc.setup(tid);

    auto& offload = svc.get_offload();
    for(int64_t tid : { 1, 2, 3 })
    {
        offload.inject(tid, make_timer_record(tid, 100));
        offload.inject(tid, make_timer_record(tid, 200));
    }

    svc.post_process();

    auto& writer = svc.report_writer_ref();
    // Each call must reference the correct tid.
    for(auto const& call : writer.timer_calls)
    {
        for(auto const& s : call.samples)
        {
            EXPECT_EQ(s.tid, call.tid) << "sample.tid must match the write call's tid — "
                                          "samples must not bleed across threads";
        }
    }
}

// ── Test 5: Allocator drain ordering ─────────────────────────────────────────
//
// offload.read(tid) must be called BEFORE report_writer.write_timer_samples().
// Verified via the in_memory_offload call_log.

TEST(post_process, offload_read_called_before_report_writer_write)
{
    pp_service svc;
    svc.setup(0);

    auto& offload = svc.get_offload();
    offload.inject(0, make_timer_record(0, 100));
    offload.inject(0, make_timer_record(0, 200));

    svc.post_process();

    // Find the first "read(0)" and the first "write" (approximated by checking reset
    // comes last).
    auto const& log        = offload.call_log;
    bool        found_read = false;
    for(auto const& entry : log)
    {
        if(entry.find("read(") != std::string::npos)
        {
            found_read = true;
        }
        if(entry == "reset()")
        {
            EXPECT_TRUE(found_read)
                << "offload.reset() must not be called before offload.read() — "
                << "reset() appeared before any read() in the call log";
        }
    }
    EXPECT_TRUE(found_read)
        << "offload.read() must be called during post_process() when a tid has records";
}

// ── Test 6: Exception safety — reset() called even when report_writer throws ──
//
// Uses throwing_writer_policies where write_timer_samples() throws.
// offload.reset() must still be called (strong-exception guarantee on teardown).
// The throw is routed through fatal_error_policy (throwing_fatal_error_policy
// re-throws as sampling_fatal_error), so EXPECT_THROW wraps post_process().

using throwing_service       = sampling_service<throwing_writer_policies>;
using throwing_flush_service = sampling_service<throwing_flush_policies>;

TEST(post_process, reset_called_even_when_flush_throws)
{
    throwing_flush_service svc;
    svc.setup(0);

    // No records needed — flush() throws before any write, reset() must still run.
    EXPECT_THROW(svc.post_process(), sampling_fatal_error);

    EXPECT_EQ(svc.get_offload().reset_count, 1)
        << "offload.reset() must be called even when report_writer.flush() throws; "
        << "reset_count was " << svc.get_offload().reset_count;
}

TEST(post_process, reset_called_even_when_report_writer_throws)
{
    throwing_service svc;
    svc.setup(0);

    auto& offload = svc.get_offload();
    offload.inject(0, make_timer_record(0, 100));
    offload.inject(0, make_timer_record(0, 200));

    // post_process() must throw (fatal_error_policy escalates the writer exception).
    // offload.reset() must still have been called.
    EXPECT_THROW(svc.post_process(), sampling_fatal_error);

    EXPECT_EQ(offload.reset_count, 1) << "offload.reset() must be called even when "
                                         "report_writer.write_timer_samples() throws; "
                                      << "reset_count was " << offload.reset_count;
}
