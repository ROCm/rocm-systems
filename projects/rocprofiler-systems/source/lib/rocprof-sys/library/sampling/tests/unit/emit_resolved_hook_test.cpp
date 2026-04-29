// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// TDD tests for Task #30: emit_resolved_to_trace_cache hook.
//
// Verifies that:
//   1. emit_resolved_to_trace_cache() is called during shutdown(tid) (generic no-op).
//   2. After shutdown(tid) with generic policies, offload_ records are still accessible
//      (no-op hook leaves the store intact for post_process compatibility).
//   3. trace_cache_offload_adapter::erase(tid) clears records for a tid without
//      affecting other tids.
//   4. in_memory_emitter provides the same erase() seam for test policies.

#include <gtest/gtest.h>

#include "doubles/in_memory_emitter.hpp"
#include "doubles/test_sampling_policies.hpp"
#include "sampling/data/backtrace_record.hpp"
#include "sampling/policies/trace_cache_offload_adapter.hpp"
#include "sampling/sampling_service.hpp"
#include "sampling/src/sample_ring_buffer.hpp"

using namespace rocprofsys::sampling;
using namespace rocprofsys::sampling::test;

// ── Generic no-op hook: records remain in offload_ after shutdown() ───────────
//
// With test_sampling_policies (in_memory_emitter + no production specialization),
// emit_resolved_to_trace_cache() is a no-op. Records injected before shutdown()
// are NOT cleared by the hook — they remain available for post_process().

TEST(emit_resolved_hook, generic_noop_leaves_offload_intact)
{
    using svc_t = sampling_service<test_sampling_policies>;
    svc_t svc;
    svc.setup(0);

    // Inject two records via inject() seam (simulates ring drain result).
    backtrace_record r{};
    r.tid          = 0;
    r.timestamp_ns = 100;
    r.trigger      = trigger_type::TIMER;
    r.pc_count     = 2;
    svc.get_offload().inject(0, r);
    r.timestamp_ns = 200;
    svc.get_offload().inject(0, r);

    // shutdown() calls emit_resolved_to_trace_cache() — no-op for generic template.
    svc.shutdown(0);

    // Records must still be in offload_ (no-op hook did not clear them).
    // post_process() can still read them.
    auto records = svc.get_offload().read(0);
    EXPECT_EQ(records.size(), 2U)
        << "generic emit_resolved_to_trace_cache() must not clear offload records";
}

// ── erase() removes records for a single tid without affecting others ─────────

TEST(emit_resolved_hook, trace_cache_adapter_erase_removes_single_tid)
{
    trace_cache_offload_adapter adapter;
    backtrace_record            rec{};
    rec.tid = 1;
    adapter.inject(1, rec);
    rec.tid = 2;
    adapter.inject(2, rec);

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
    adapter.inject(5, rec);
    adapter.inject(6, rec);

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
    emitter.inject(10, rec);
    rec.tid = 20;
    emitter.inject(20, rec);

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
