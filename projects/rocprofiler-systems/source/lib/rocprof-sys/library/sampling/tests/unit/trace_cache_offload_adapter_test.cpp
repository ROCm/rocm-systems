// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// TDD tests for trace_cache_offload_adapter — the production EmitterPolicy.
// This adapter replaces tmpfile_offload_store; it stores drained backtrace_records
// in memory without any /tmp file I/O.
//
// The adapter must satisfy the EmitterPolicy named requirement:
//   write(int64_t tid, sample_ring_buffer<N>& buf, FatalErrorPolicy& fatal)
//   read(int64_t tid) -> std::vector<backtrace_record>
//   tids() const     -> std::vector<int64_t>
//   reset() noexcept

#include <gtest/gtest.h>

// The adapter is in its own lightweight header so it is testable without
// libunwind / AMD-SMI / thread_info dependencies.
#include "sampling/data/backtrace_record.hpp"
#include "sampling/policies/trace_cache_offload_adapter.hpp"
#include "sampling/src/sample_ring_buffer.hpp"

using namespace rocprofsys::sampling;

// Minimal fatal policy that aborts — tests that don't exercise the failure path use this.
struct noop_fatal
{
    template <class... Args>
    [[noreturn]] void fatal(char const*, int, std::string_view, Args const&...) noexcept
    {
        std::abort();
    }
};

// Fatal policy that records that it was called, then throws (for failure-path tests).
struct recording_fatal
{
    bool called{ false };

    template <class... Args>
    [[noreturn]] void fatal(char const*, int, std::string_view, Args const&...) noexcept
    {
        called = true;
        throw std::runtime_error("fatal called");
    }
};

static backtrace_record
make_record(int64_t tid, uint64_t ts_ns, uint8_t pc_count = 2)
{
    backtrace_record r{};
    r.tid          = tid;
    r.timestamp_ns = ts_ns;
    r.pc_count     = pc_count;
    for(uint8_t i = 0; i < pc_count; ++i)
        r.raw_pcs[i] = static_cast<uintptr_t>(0x1000 + i);
    return r;
}

// ── Basic round-trip ─────────────────────────────────────────────────────────

TEST(trace_cache_offload_adapter, write_then_read_returns_same_records)
{
    trace_cache_offload_adapter adapter;
    noop_fatal                  fatal;

    sample_ring_buffer<8> ring;
    ring.try_push(make_record(0, 100));
    ring.try_push(make_record(0, 200));
    ring.try_push(make_record(0, 300));

    adapter.write(0, ring, fatal);
    auto records = adapter.read(0);

    ASSERT_EQ(records.size(), 3U);
    EXPECT_EQ(records[0].timestamp_ns, 100U);
    EXPECT_EQ(records[1].timestamp_ns, 200U);
    EXPECT_EQ(records[2].timestamp_ns, 300U);
}

TEST(trace_cache_offload_adapter, read_unknown_tid_returns_empty)
{
    trace_cache_offload_adapter adapter;
    auto                        records = adapter.read(42);
    EXPECT_TRUE(records.empty());
}

TEST(trace_cache_offload_adapter, empty_ring_produces_empty_read)
{
    trace_cache_offload_adapter adapter;
    noop_fatal                  fatal;

    sample_ring_buffer<8> ring;
    adapter.write(0, ring, fatal);
    auto records = adapter.read(0);
    EXPECT_TRUE(records.empty());
}

// ── tids() ───────────────────────────────────────────────────────────────────

TEST(trace_cache_offload_adapter, tids_returns_all_written_tids)
{
    trace_cache_offload_adapter adapter;
    noop_fatal                  fatal;

    for(int64_t tid : { 1, 2, 3 })
    {
        sample_ring_buffer<4> ring;
        ring.try_push(make_record(tid, 100));
        adapter.write(tid, ring, fatal);
    }

    auto tids = adapter.tids();
    ASSERT_EQ(tids.size(), 3U);
    for(int64_t tid : { 1, 2, 3 })
    {
        EXPECT_NE(std::find(tids.begin(), tids.end(), tid), tids.end())
            << "tids() must include tid " << tid;
    }
}

TEST(trace_cache_offload_adapter, tids_empty_before_any_write)
{
    trace_cache_offload_adapter adapter;
    EXPECT_TRUE(adapter.tids().empty());
}

// ── reset() ──────────────────────────────────────────────────────────────────

TEST(trace_cache_offload_adapter, reset_clears_all_stored_records)
{
    trace_cache_offload_adapter adapter;
    noop_fatal                  fatal;

    sample_ring_buffer<4> ring;
    ring.try_push(make_record(1, 100));
    adapter.write(1, ring, fatal);

    adapter.reset();

    EXPECT_TRUE(adapter.read(1).empty()) << "reset() must clear all records";
    EXPECT_TRUE(adapter.tids().empty()) << "reset() must clear the tid set";
}

TEST(trace_cache_offload_adapter, reset_is_noexcept)
{
    static_assert(noexcept(std::declval<trace_cache_offload_adapter>().reset()),
                  "reset() must be noexcept per EmitterPolicy requirement");
}

// ── Multiple write calls accumulate ──────────────────────────────────────────

TEST(trace_cache_offload_adapter, multiple_writes_to_same_tid_accumulate)
{
    trace_cache_offload_adapter adapter;
    noop_fatal                  fatal;

    sample_ring_buffer<4> ring1;
    ring1.try_push(make_record(5, 100));
    adapter.write(5, ring1, fatal);

    sample_ring_buffer<4> ring2;
    ring2.try_push(make_record(5, 200));
    ring2.try_push(make_record(5, 300));
    adapter.write(5, ring2, fatal);

    auto records = adapter.read(5);
    ASSERT_EQ(records.size(), 3U)
        << "successive write() calls for the same tid must accumulate records";
    EXPECT_EQ(records[0].timestamp_ns, 100U);
    EXPECT_EQ(records[1].timestamp_ns, 200U);
    EXPECT_EQ(records[2].timestamp_ns, 300U);
}

// ── Multi-tid isolation ───────────────────────────────────────────────────────

TEST(trace_cache_offload_adapter, records_isolated_per_tid)
{
    trace_cache_offload_adapter adapter;
    noop_fatal                  fatal;

    for(int64_t tid : { 10, 20 })
    {
        sample_ring_buffer<4> ring;
        ring.try_push(make_record(tid, static_cast<uint64_t>(tid * 100)));
        adapter.write(tid, ring, fatal);
    }

    auto r10 = adapter.read(10);
    auto r20 = adapter.read(20);
    ASSERT_EQ(r10.size(), 1U);
    ASSERT_EQ(r20.size(), 1U);
    EXPECT_EQ(r10[0].tid, 10);
    EXPECT_EQ(r20[0].tid, 20);
    EXPECT_EQ(r10[0].timestamp_ns, 1000U);
    EXPECT_EQ(r20[0].timestamp_ns, 2000U);
}

// ── Timestamp ordering preserved ─────────────────────────────────────────────

TEST(trace_cache_offload_adapter, write_preserves_push_order)
{
    trace_cache_offload_adapter adapter;
    noop_fatal                  fatal;

    sample_ring_buffer<8> ring;
    ring.try_push(make_record(0, 10));
    ring.try_push(make_record(0, 20));
    ring.try_push(make_record(0, 30));
    adapter.write(0, ring, fatal);

    auto records = adapter.read(0);
    ASSERT_EQ(records.size(), 3U);
    EXPECT_LT(records[0].timestamp_ns, records[1].timestamp_ns);
    EXPECT_LT(records[1].timestamp_ns, records[2].timestamp_ns);
}

// ── inject() seam (for post_process tests) ───────────────────────────────────

TEST(trace_cache_offload_adapter, inject_adds_record_without_ring_buffer)
{
    trace_cache_offload_adapter adapter;
    adapter.inject(7, make_record(7, 500));
    auto records = adapter.read(7);
    ASSERT_EQ(records.size(), 1U);
    EXPECT_EQ(records[0].tid, 7);
    EXPECT_EQ(records[0].timestamp_ns, 500U);
}

// ── EmitterPolicy concept smoke test ─────────────────────────────────────────
// Verifies that the adapter can be plugged in as the `offload` policy type.

namespace
{
struct minimal_policies
{
    using unwinder          = void;
    using offload           = trace_cache_offload_adapter;
    using trace_sink        = void;
    using timer_trigger     = void;
    using overflow_trigger  = void;
    using clock             = void;
    using signal_dispatcher = void;
    using report_writer     = void;
    using perfetto_sink     = void;
    using fatal_error       = void;
};
static_assert(std::is_same_v<minimal_policies::offload, trace_cache_offload_adapter>,
              "trace_cache_offload_adapter must be usable as the EmitterPolicy type");
}  // namespace
