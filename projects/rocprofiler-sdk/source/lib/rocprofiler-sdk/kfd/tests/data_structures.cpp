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

#include "lib/rocprofiler-sdk/kfd/correlation_table.hpp"
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
    m.bind(qid(42), /*doorbell_off=*/7);

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
    m.bind(qid(42), 7);

    m.on_queue_destroyed(qid(42));

    EXPECT_EQ(m.get_generation(7), 1u);                 // bumped
    EXPECT_FALSE(m.get_by_queue(qid(42)).has_value());  // mapping removed
    EXPECT_FALSE(m.is_generation_certain(7));           // uncertain until rebind
}

TEST(DoorbellMap, doorbell_reuse_gets_new_generation)
{
    auto m = DoorbellMap{};
    // queue 42 on doorbell 7, then destroyed
    m.bind(qid(42), 7);
    m.on_queue_destroyed(qid(42));
    EXPECT_EQ(m.get_generation(7), 1u);

    // a new queue 43 reuses doorbell 7 -> must carry the bumped generation,
    // so records from the old queue can never be attributed to the new one.
    m.bind(qid(43), 7);
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

// bind() is an upsert (map[key]=), not insert-if-absent: re-binding the SAME
// queue to a NEW doorbell must update its entry. (This is why bind() keeps []=
// rather than emplace.)
TEST(DoorbellMap, rebind_same_queue_updates_doorbell)
{
    auto m = DoorbellMap{};
    m.bind(qid(42), 7);
    m.bind(qid(42), 9);  // same queue, new doorbell -> must overwrite

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
    m.bind(qid(42), 7);
    m.bind(qid(43), 7);

    auto a = m.get_by_queue(qid(42));
    auto b = m.get_by_queue(qid(43));
    ASSERT_TRUE(a.has_value());
    ASSERT_TRUE(b.has_value());
    EXPECT_EQ(a->doorbell_off, 7u);
    EXPECT_EQ(b->doorbell_off, 7u);
    EXPECT_EQ(m.get_generation(7), 0u);  // no destroy -> generation unchanged
}

// Empirical bind: capture notes a pending dispatch for an unbound queue; reader
// binds the doorbell from the matching record; subsequent lookups resolve.
TEST(DoorbellMap, empirical_bind_from_record)
{
    auto m = DoorbellMap{};

    // Queue 42 is unbound: note_pending returns false and records the hint.
    EXPECT_FALSE(m.note_pending_dispatch(qid(42), /*dispatch_idx_low32=*/7));
    EXPECT_FALSE(m.is_bound(/*doorbell_off=*/4100));
    EXPECT_FALSE(m.get_by_queue(qid(42)).has_value());

    // Reader sees record (doorbell=4100, dispatch_id=7): binds 4100 -> queue 42.
    EXPECT_TRUE(m.bind_from_record(/*doorbell_off=*/4100, /*dispatch_id=*/7));
    EXPECT_TRUE(m.is_bound(4100));

    // Now the queue resolves to that doorbell.
    auto e = m.get_by_queue(qid(42));
    ASSERT_TRUE(e.has_value());
    EXPECT_EQ(e->doorbell_off, 4100u);

    // Once bound, note_pending returns true (caller builds key directly).
    EXPECT_TRUE(m.note_pending_dispatch(qid(42), 8));
}

// bind_from_record with no matching hint yet returns false (doorbell stays unknown).
TEST(DoorbellMap, bind_from_record_no_hint)
{
    auto m = DoorbellMap{};
    EXPECT_FALSE(m.bind_from_record(/*doorbell_off=*/4100, /*dispatch_id=*/99));
    EXPECT_FALSE(m.is_bound(4100));
}

// bind_from_record is idempotent: a second record for an already-bound doorbell
// returns true without disturbing the binding.
TEST(DoorbellMap, bind_from_record_idempotent)
{
    auto m = DoorbellMap{};
    m.note_pending_dispatch(qid(42), 7);
    EXPECT_TRUE(m.bind_from_record(4100, 7));
    EXPECT_TRUE(m.bind_from_record(4100, 8));  // already bound -> true, no-op
    auto e = m.get_by_queue(qid(42));
    ASSERT_TRUE(e.has_value());
    EXPECT_EQ(e->doorbell_off, 4100u);
}

// ---------------------------------------------------------------------------
// CorrelationTable
// ---------------------------------------------------------------------------
TEST(CorrelationTable, insert_take_roundtrip)
{
    auto t   = CorrelationTable{};
    auto key = correlation_key{7, 100, 0};
    t.insert(
        key,
        correlation_entry{/*sdk_dispatch_id=*/5, /*kernel_id=*/9, qid(42), /*enqueue_ts=*/123});

    EXPECT_EQ(t.size(), 1u);
    auto e = t.take(key);
    ASSERT_TRUE(e.has_value());
    EXPECT_EQ(e->sdk_dispatch_id, 5u);
    EXPECT_EQ(e->kernel_id, 9u);
    EXPECT_EQ(e->queue_id.handle, 42u);
    EXPECT_EQ(e->enqueue_ts, 123u);
    EXPECT_EQ(t.size(), 0u);  // take erased it
}

TEST(CorrelationTable, take_missing_returns_nullopt)
{
    auto t = CorrelationTable{};
    EXPECT_FALSE(t.take(correlation_key{1, 2, 3}).has_value());
}

TEST(CorrelationTable, erase_is_idempotent)
{
    auto t   = CorrelationTable{};
    auto key = correlation_key{7, 100, 0};
    t.insert(key, correlation_entry{5, 9, qid(42), 123});
    t.erase(key);  // present -> removed
    t.erase(key);  // absent  -> no-op, must not crash
    EXPECT_EQ(t.size(), 0u);
}

// insert() uses emplace (insert-if-absent): a duplicate key keeps the FIRST
// entry, not the second. A correlation_key is unique per in-flight dispatch, so
// this only guards the should-not-happen collision case.
TEST(CorrelationTable, duplicate_insert_keeps_first)
{
    auto t   = CorrelationTable{};
    auto key = correlation_key{7, 100, 0};
    t.insert(key, correlation_entry{5, 9, qid(42), 123});
    t.insert(key, correlation_entry{99, 88, qid(7), 456});  // must be ignored
    EXPECT_EQ(t.size(), 1u);
    auto e = t.take(key);
    ASSERT_TRUE(e.has_value());
    EXPECT_EQ(e->sdk_dispatch_id, 5u);  // first entry retained
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
