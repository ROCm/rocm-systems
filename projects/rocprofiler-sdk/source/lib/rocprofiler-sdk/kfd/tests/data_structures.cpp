// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "lib/rocprofiler-sdk/kfd/correlation_types.hpp"
#include "lib/rocprofiler-sdk/kfd/doorbell_map.hpp"
#include "lib/rocprofiler-sdk/kfd/results_map.hpp"

#include <gtest/gtest.h>

namespace
{
using namespace rocprofiler::kfd;

rocprofiler_queue_id_t
qid(uint64_t h)
{
    return rocprofiler_queue_id_t{h};
}
}  // namespace

// ---------------------------------------------------------------------------
// correlation_key
// ---------------------------------------------------------------------------
TEST(correlation_key, equality_and_hash)
{
    auto a = correlation_key{7, 100, 0};
    auto b = correlation_key{7, 100, 0};
    auto c = correlation_key{7, 100, 1};  // different generation

    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);

    auto hash = correlation_key_hash{};
    EXPECT_EQ(hash(a), hash(b));
    // Different keys should (almost surely) hash differently.
    EXPECT_NE(hash(a), hash(c));
}

// ---------------------------------------------------------------------------
// DoorbellMap
// ---------------------------------------------------------------------------
TEST(DoorbellMap, bind_and_lookup)
{
    auto m = DoorbellMap{};
    m.bind_and_resolve(qid(42), /*doorbell_off=*/7);

    auto e = m.get_by_queue(qid(42));
    ASSERT_TRUE(e.has_value());
    EXPECT_EQ(e->doorbell_off, 7u);
    EXPECT_EQ(e->generation, 0u);
    EXPECT_TRUE(m.is_generation_certain(7));
    EXPECT_EQ(m.get_generation(7), 0u);
}

TEST(DoorbellMap, unknown_queue_returns_nullopt)
{
    auto m = DoorbellMap{};
    EXPECT_FALSE(m.get_by_queue(qid(999)).has_value());
}

TEST(DoorbellMap, destroy_bumps_generation_and_marks_uncertain)
{
    auto m = DoorbellMap{};
    m.bind_and_resolve(qid(42), 7);

    m.on_queue_destroyed(qid(42));

    EXPECT_EQ(m.get_generation(7), 1u);                 // bumped
    EXPECT_FALSE(m.get_by_queue(qid(42)).has_value());  // mapping removed
    EXPECT_FALSE(m.is_generation_certain(7));           // uncertain until rebind
}

TEST(DoorbellMap, doorbell_reuse_gets_new_generation)
{
    auto m = DoorbellMap{};
    // queue 42 on doorbell 7, then destroyed
    m.bind_and_resolve(qid(42), 7);
    m.on_queue_destroyed(qid(42));
    EXPECT_EQ(m.get_generation(7), 1u);

    // a new queue 43 reuses doorbell 7 -> must carry the bumped generation,
    // so records from the old queue can never be attributed to the new one.
    m.bind_and_resolve(qid(43), 7);
    auto e = m.get_by_queue(qid(43));
    ASSERT_TRUE(e.has_value());
    EXPECT_EQ(e->doorbell_off, 7u);
    EXPECT_EQ(e->generation, 1u);
    EXPECT_TRUE(m.is_generation_certain(7));  // rebind clears uncertainty
}

TEST(DoorbellMap, destroy_unknown_queue_is_noop)
{
    auto m = DoorbellMap{};
    m.on_queue_destroyed(qid(123));  // must not crash
    EXPECT_EQ(m.get_generation(7), 0u);
}

// Binding is an upsert (map[key]=), not insert-if-absent: re-binding the SAME
// queue to a NEW doorbell must update its entry (kept []= rather than emplace).
TEST(DoorbellMap, rebind_same_queue_updates_doorbell)
{
    auto m = DoorbellMap{};
    m.bind_and_resolve(qid(42), 7);
    m.bind_and_resolve(qid(42), 9);  // same queue, new doorbell -> must overwrite

    auto e = m.get_by_queue(qid(42));
    ASSERT_TRUE(e.has_value());
    EXPECT_EQ(e->doorbell_off, 9u);  // updated, not stuck at 7
}

// Two distinct queues binding the same doorbell (degenerate, should not happen
// without an intervening destroy) each resolve via the forward map, and the
// shared doorbell generation is preserved (0 here, never bumped without destroy).
TEST(DoorbellMap, two_queues_same_doorbell_forward_resolves)
{
    auto m = DoorbellMap{};
    m.bind_and_resolve(qid(42), 7);
    m.bind_and_resolve(qid(43), 7);

    auto a = m.get_by_queue(qid(42));
    auto b = m.get_by_queue(qid(43));
    ASSERT_TRUE(a.has_value());
    ASSERT_TRUE(b.has_value());
    EXPECT_EQ(a->doorbell_off, 7u);
    EXPECT_EQ(b->doorbell_off, 7u);
    EXPECT_EQ(m.get_generation(7), 0u);  // no destroy -> generation unchanged
}

// Page-relative doorbell slot helpers: capture (from pointer) and reader (from
// record) must reduce to the same per-queue slot so their correlation keys match.
TEST(DoorbellMap, page_relative_slot_capture_matches_reader)
{
    // Reader side: an absolute record doorbell_off masks to its page-relative slot.
    EXPECT_EQ(doorbell_off_to_page_slot(4100u), 4u);
    EXPECT_EQ(doorbell_off_to_page_slot(4104u), 8u);

    // Capture side: a doorbell pointer's in-page byte offset, in dwords, gives the
    // same slot (8-byte doorbells -> 2 dwords apart), independent of the page base.
    constexpr uint64_t kPage = 4096;
    EXPECT_EQ(doorbell_ptr_to_page_slot(0x7f0000004010ull, kPage), 4u);
    EXPECT_EQ(doorbell_ptr_to_page_slot(0x7f0000004020ull, kPage), 8u);

    // Same base index (4096) dropped by both sides -> keys agree.
    EXPECT_EQ(doorbell_off_to_page_slot(4100u),
              doorbell_ptr_to_page_slot(0x7f0000004010ull, kPage));
}

// bind_and_resolve is the per-dispatch hot-path helper: it binds the first time a
// (queue, doorbell) pair is seen, then returns the cached entry on later calls.
TEST(DoorbellMap, bind_and_resolve_binds_then_caches)
{
    auto m = DoorbellMap{};

    // First call binds (queue previously unknown) and resolves.
    auto e1 = m.bind_and_resolve(qid(42), 4u);
    EXPECT_EQ(e1.doorbell_off, 4u);
    EXPECT_EQ(e1.generation, 0u);

    // Subsequent calls return the same entry without changing state.
    auto e2 = m.bind_and_resolve(qid(42), 4u);
    EXPECT_EQ(e2.doorbell_off, 4u);
    EXPECT_EQ(e2.generation, 0u);

    // Equivalent to what get_by_queue reports.
    auto snap = m.get_by_queue(qid(42));
    ASSERT_TRUE(snap.has_value());
    EXPECT_EQ(snap->doorbell_off, 4u);
    EXPECT_EQ(snap->generation, 0u);
}

// After a queue is destroyed and its doorbell reused by a new queue,
// bind_and_resolve must rebind (clearing the uncertain mark) and report the
// bumped generation -- never the stale one.
TEST(DoorbellMap, bind_and_resolve_rebinds_on_doorbell_reuse)
{
    auto m = DoorbellMap{};

    m.bind_and_resolve(qid(1), 4u);  // gen 0
    m.on_queue_destroyed(qid(1));    // doorbell 4 -> uncertain, gen bumped to 1
    EXPECT_FALSE(m.is_generation_certain(4u));

    // New queue reuses doorbell 4: the FIRST resolve rebinds via the slow
    // (write-lock) path because qid(2) is absent from by_queue -- hand back the
    // bumped gen 1, certain.
    auto e = m.bind_and_resolve(qid(2), 4u);
    EXPECT_EQ(e.doorbell_off, 4u);
    EXPECT_EQ(e.generation, 1u);
    EXPECT_TRUE(m.is_generation_certain(4u));

    // A SECOND resolve on the reused queue now takes the fast (read-lock) path
    // (qid(2) bound, doorbell certain). It must still report the bumped gen 1,
    // never a stale 0 -- guards against the fast path serving pre-reuse state.
    auto e2 = m.bind_and_resolve(qid(2), 4u);
    EXPECT_EQ(e2.doorbell_off, 4u);
    EXPECT_EQ(e2.generation, 1u);
}

// A queue that migrates to a different doorbell_off must resolve to the new slot,
// not the cached old one.
TEST(DoorbellMap, bind_and_resolve_follows_queue_to_new_doorbell)
{
    auto m = DoorbellMap{};

    auto a = m.bind_and_resolve(qid(7), 4u);
    EXPECT_EQ(a.doorbell_off, 4u);

    auto b = m.bind_and_resolve(qid(7), 8u);  // same queue, different doorbell
    EXPECT_EQ(b.doorbell_off, 8u);

    auto snap = m.get_by_queue(qid(7));
    ASSERT_TRUE(snap.has_value());
    EXPECT_EQ(snap->doorbell_off, 8u);
}

// ---------------------------------------------------------------------------
// ResultsMap
// ---------------------------------------------------------------------------
TEST(ResultsMap, deposit_take_roundtrip)
{
    auto m   = ResultsMap{};
    auto key = correlation_key{7, 100, 0};
    m.deposit(key, kfd_timing_result{/*start*/ 1000, /*end*/ 2000, /*deposited_at_ns*/ 500});

    auto r = m.take(key);
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->start_gpu_ticks, 1000u);
    EXPECT_EQ(r->end_gpu_ticks, 2000u);
    EXPECT_FALSE(m.take(key).has_value());  // take erased it
}

// deposit() uses emplace (insert-if-absent): a duplicate key keeps the FIRST
// result, not the second.
TEST(ResultsMap, duplicate_deposit_keeps_first)
{
    auto m   = ResultsMap{};
    auto key = correlation_key{7, 100, 0};
    m.deposit(key, kfd_timing_result{1000, 2000, 500});
    m.deposit(key, kfd_timing_result{7777, 8888, 999});  // must be ignored
    auto r = m.take(key);
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->start_gpu_ticks, 1000u);  // first result retained
    EXPECT_EQ(r->end_gpu_ticks, 2000u);
}

TEST(ResultsMap, evict_stale_removes_old_keeps_fresh)
{
    auto m = ResultsMap{};
    m.deposit(correlation_key{7, 1, 0}, kfd_timing_result{1, 2, /*deposited_at_ns*/ 0});
    m.deposit(correlation_key{7, 2, 0}, kfd_timing_result{1, 2, /*deposited_at_ns*/ 9'000});

    // now=10000, max_age=5000 -> entry at t=0 is 10000ns old (evict);
    // entry at t=9000 is 1000ns old (keep).
    auto evicted = m.evict_stale(/*now_ns=*/10'000, /*max_age_ns=*/5'000);
    EXPECT_EQ(evicted, 1u);
    EXPECT_EQ(m.size(), 1u);
    EXPECT_FALSE(m.take(correlation_key{7, 1, 0}).has_value());
    EXPECT_TRUE(m.take(correlation_key{7, 2, 0}).has_value());
}

TEST(ResultsMap, evict_stale_tolerates_future_timestamp)
{
    auto m = ResultsMap{};
    // deposited_at_ns ahead of now_ns (clock skew) must not underflow/evict.
    m.deposit(correlation_key{7, 1, 0}, kfd_timing_result{1, 2, /*deposited_at_ns*/ 20'000});
    auto evicted = m.evict_stale(/*now_ns=*/10'000, /*max_age_ns=*/5'000);
    EXPECT_EQ(evicted, 0u);
    EXPECT_EQ(m.size(), 1u);
}
