// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// TDD tests for Task #30: emit_resolved hook (folded into sampling_service).
//
// Verifies that:
//   1. emit_resolved runs as part of shutdown(tid) and clears the offload store
//      for that tid (no records remain after shutdown). The lifecycle hook is
//      no longer policy-substitutable; sampling_service::do_emit_resolved owns
//      the parse/resolve/erase sequence directly.
//   2. trace_cache_offload_adapter::erase(tid) clears records for a tid without
//      affecting other tids.
//   3. in_memory_emitter provides the same erase() seam for test policies.

#include <gtest/gtest.h>

#include "doubles/in_memory_emitter.hpp"
#include "doubles/test_sampling_policies.hpp"
#include "sampling/data/backtrace_record.hpp"
#include "sampling/policies/trace_cache_offload_adapter.hpp"
#include "sampling/sampling_service.hpp"
#include "sampling/src/sample_ring_buffer.hpp"

using namespace rocprofsys::sampling;
using namespace rocprofsys::sampling::test;

// ── shutdown(tid) drains the offload store via the folded emit_resolved ─────
//
// emit_resolved now lives directly on sampling_service (post-fold). For every
// Policies bundle — including the test bundle — shutdown(tid) parses the raw
// records and calls offload_.erase(tid) at the tail of do_emit_resolved.
// Tests therefore observe the offload as empty for that tid after shutdown.

TEST(emit_resolved_hook, shutdown_drains_offload_for_tid)
{
    using svc_t = sampling_service<test_sampling_policies>;
    svc_t svc{ make_test_config() };
    svc.setup(0);

    // Insert two records directly (simulates ring drain result).
    backtrace_record r{};
    r.tid          = 0;
    r.timestamp_ns = 100;
    r.trigger      = trigger_type::TIMER;
    r.pc_count     = 2;
    svc.get_offload().insert(0, r);
    r.timestamp_ns = 200;
    svc.get_offload().insert(0, r);

    svc.shutdown(0);

    auto records = svc.get_offload().read(0);
    EXPECT_TRUE(records.empty())
        << "do_emit_resolved must erase the tid's offload records during shutdown";
}

// ── erase() removes records for a single tid without affecting others ─────────

TEST(emit_resolved_hook, trace_cache_adapter_erase_removes_single_tid)
{
    trace_cache_offload_adapter adapter;
    backtrace_record            rec{};
    rec.tid = 1;
    adapter.insert(1, rec);
    rec.tid = 2;
    adapter.insert(2, rec);

    adapter.erase(1);

    EXPECT_TRUE(adapter.read(1).empty()) << "erase(1) must remove tid 1 records";
    EXPECT_EQ(adapter.read(2).size(), 1U) << "erase(1) must not affect tid 2 records";
}

TEST(emit_resolved_hook, trace_cache_adapter_erase_nonexistent_tid_is_noop)
{
    trace_cache_offload_adapter adapter;
    adapter.erase(99);  // no records for tid 99 — must not throw or crash
    EXPECT_TRUE(adapter.read(99).empty());
}

TEST(emit_resolved_hook, trace_cache_adapter_erase_removes_from_tids_list)
{
    trace_cache_offload_adapter adapter;
    backtrace_record            rec{};
    rec.tid = 5;
    adapter.insert(5, rec);
    adapter.insert(6, rec);

    adapter.erase(5);

    auto tids = adapter.tids();
    EXPECT_EQ(tids.size(), 1U);
    EXPECT_EQ(tids.front(), 6) << "tids() must not include erased tid 5";
}

// ── in_memory_emitter also provides erase() for test completeness ─────────────

TEST(emit_resolved_hook, in_memory_emitter_erase_removes_single_tid)
{
    in_memory_emitter emitter;
    backtrace_record  rec{};
    rec.tid = 10;
    emitter.insert(10, rec);
    rec.tid = 20;
    emitter.insert(20, rec);

    emitter.erase(10);

    EXPECT_TRUE(emitter.read(10).empty()) << "erase(10) must remove tid 10 records";
    EXPECT_EQ(emitter.read(20).size(), 1U) << "erase(10) must not affect tid 20";
}

// ── erase() is noexcept ───────────────────────────────────────────────────────

TEST(emit_resolved_hook, trace_cache_adapter_erase_is_noexcept)
{
    static_assert(noexcept(std::declval<trace_cache_offload_adapter>().erase(0)),
                  "erase() must be noexcept");
}
