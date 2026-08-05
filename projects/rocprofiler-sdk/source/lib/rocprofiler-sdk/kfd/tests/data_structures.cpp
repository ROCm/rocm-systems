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
#include "lib/rocprofiler-sdk/kfd/kfd_correlation.hpp"
#include "lib/rocprofiler-sdk/kfd/results_map.hpp"
#include "lib/rocprofiler-sdk/kfd/signal_less_gate.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>

namespace
{
using namespace rocprofiler::kfd;

rocprofiler_queue_id_t
qid(uint64_t h)
{
    return rocprofiler_queue_id_t{h};
}

// Rendezvous helpers. Deadlines are absolute steady_now_ns() values, so a test
// can hand wait_take() a deadline that is already in the past (no wait), far in
// the future (blocks until notified), or a specific budget.
uint64_t
deadline_in_ms(uint64_t ms)
{
    return steady_now_ns() + ms * 1'000'000ull;
}

uint64_t
elapsed_ms_since(uint64_t start_ns)
{
    return (steady_now_ns() - start_ns) / 1'000'000ull;
}
}  // namespace

// correlation_key
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

// kfd_time_is_sane: the KFD-result-vs-HSA-fallback decision in get_dispatch_time
TEST(kfd_time_is_sane, accepts_interval_inside_the_dispatch_window)
{
    EXPECT_TRUE(kfd_time_is_sane(/*start*/ 150, /*end*/ 250, /*enqueue*/ 100, /*now*/ 300));
    // Exactly on both bounds is still inside the window.
    EXPECT_TRUE(kfd_time_is_sane(100, 300, 100, 300));
}

// A converted firmware end legitimately lands a few ms past a CPU `now`.
TEST(kfd_time_is_sane, accepts_a_converted_end_slightly_past_now)
{
    constexpr uint64_t now = 1'000'000'000;

    EXPECT_TRUE(kfd_time_is_sane(now - 5'000'000, now + 2'700'000, 0, now));  // observed skew
    EXPECT_TRUE(kfd_time_is_sane(0, now + kKfdFutureSlackNs, 0, now));        // exactly the bound

    // Beyond the slack is still rejected: that is not conversion jitter, it is a
    // record from somewhere else.
    EXPECT_FALSE(kfd_time_is_sane(0, now + kKfdFutureSlackNs + 1, 0, now));
    EXPECT_FALSE(kfd_time_is_sane(0, now + 5'000'000'000, 0, now));  // 5 s out
}

TEST(kfd_time_is_sane, rejects_records_outside_the_dispatch_window)
{
    EXPECT_FALSE(kfd_time_is_sane(99, 250, 100, 300));   // starts before enqueue
    // Ends beyond now + the conversion slack (a few hundred ns past now is now
    // tolerated on purpose; see accepts_a_converted_end_slightly_past_now).
    EXPECT_FALSE(kfd_time_is_sane(150, 301 + kKfdFutureSlackNs, 100, 300));
    EXPECT_FALSE(kfd_time_is_sane(250, 150, 100, 300));  // inverted interval
    EXPECT_FALSE(kfd_time_is_sane(150, 150, 100, 300));  // zero-length interval
}

// DoorbellMap
TEST(DoorbellMap, bind_and_lookup)
{
    auto e = DoorbellMap{}.bind_and_resolve(0, qid(42), /*doorbell_off=*/7);
    EXPECT_EQ(e.doorbell_off, 7u);
    EXPECT_EQ(e.generation, 0u);
}

TEST(DoorbellMap, destroy_bumps_generation)
{
    auto m = DoorbellMap{};
    m.bind_and_resolve(0, qid(42), 7);

    m.on_queue_destroyed(qid(42));

    EXPECT_EQ(m.get_generation(0, 7), 1u);  // bumped
}

TEST(DoorbellMap, doorbell_reuse_gets_new_generation)
{
    auto m = DoorbellMap{};
    // queue 42 on doorbell 7, then destroyed
    m.bind_and_resolve(0, qid(42), 7);
    m.on_queue_destroyed(qid(42));
    EXPECT_EQ(m.get_generation(0, 7), 1u);

    // a new queue 43 reuses doorbell 7 -> must carry the bumped generation,
    // so records from the old queue can never be attributed to the new one.
    auto e = m.bind_and_resolve(0, qid(43), 7);
    EXPECT_EQ(e.doorbell_off, 7u);
    EXPECT_EQ(e.generation, 1u);
}

TEST(DoorbellMap, destroy_unknown_queue_is_noop)
{
    auto m = DoorbellMap{};
    m.on_queue_destroyed(qid(123));  // must not crash
    EXPECT_EQ(m.get_generation(0, 7), 0u);
}

// Two distinct queues binding the same doorbell (degenerate, should not happen
// without an intervening destroy) each resolve via the forward map, and the
// shared doorbell generation is preserved (0 here, never bumped without destroy).
TEST(DoorbellMap, two_queues_same_doorbell_forward_resolves)
{
    auto m = DoorbellMap{};
    auto a = m.bind_and_resolve(0, qid(42), 7);
    auto b = m.bind_and_resolve(0, qid(43), 7);

    EXPECT_EQ(a.doorbell_off, 7u);
    EXPECT_EQ(b.doorbell_off, 7u);
    EXPECT_EQ(m.get_generation(0, 7), 0u);  // no destroy -> generation unchanged
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
    auto e1 = m.bind_and_resolve(0, qid(42), 4u);
    EXPECT_EQ(e1.doorbell_off, 4u);
    EXPECT_EQ(e1.generation, 0u);

    // Subsequent calls return the same entry without changing state.
    auto e2 = m.bind_and_resolve(0, qid(42), 4u);
    EXPECT_EQ(e2.doorbell_off, 4u);
    EXPECT_EQ(e2.generation, 0u);
}

// After a queue is destroyed and its doorbell reused by a new queue,
// bind_and_resolve must rebind and report the bumped generation -- never the
// stale one.
TEST(DoorbellMap, bind_and_resolve_rebinds_on_doorbell_reuse)
{
    auto m = DoorbellMap{};

    m.bind_and_resolve(0, qid(1), 4u);  // gen 0
    m.on_queue_destroyed(qid(1));    // doorbell 4 -> gen bumped to 1

    // New queue reuses doorbell 4: the FIRST resolve rebinds via the slow
    // (write-lock) path because qid(2) is absent from by_queue -- hand back the
    // bumped gen 1.
    auto e = m.bind_and_resolve(0, qid(2), 4u);
    EXPECT_EQ(e.doorbell_off, 4u);
    EXPECT_EQ(e.generation, 1u);

    // A SECOND resolve on the reused queue now takes the fast (read-lock) path.
    // It must still report the bumped gen 1, never a stale 0 -- guards against
    // the fast path serving pre-reuse state.
    auto e2 = m.bind_and_resolve(0, qid(2), 4u);
    EXPECT_EQ(e2.doorbell_off, 4u);
    EXPECT_EQ(e2.generation, 1u);
}

// Binding is an upsert: a queue that migrates to a different doorbell_off must
// resolve to the new slot, not the cached old one.
TEST(DoorbellMap, bind_and_resolve_follows_queue_to_new_doorbell)
{
    auto m = DoorbellMap{};

    auto a = m.bind_and_resolve(0, qid(7), 4u);
    EXPECT_EQ(a.doorbell_off, 4u);

    auto b = m.bind_and_resolve(0, qid(7), 8u);  // same queue, different doorbell
    EXPECT_EQ(b.doorbell_off, 8u);
}

// ResultsMap
TEST(ResultsMap, deposit_take_roundtrip)
{
    auto m   = ResultsMap{};
    auto key = correlation_key{7, 100, 0};
    m.deposit(key, kfd_timing_result{/*start*/ 1000, /*end*/ 2000, /*deposited_at_ns*/ 500});

    auto r = m.take(key);
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->start_gpu_ticks, 1000u);
    EXPECT_EQ(r->end_gpu_ticks, 2000u);
    EXPECT_EQ(m.stats().hits, 1u);
    EXPECT_EQ(m.stats().misses, 0u);

    EXPECT_FALSE(m.take(key).has_value());  // take erased it
    EXPECT_EQ(m.stats().hits, 1u);
    EXPECT_EQ(m.stats().misses, 1u);  // the failed take is the HSA-fallback signal
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
    EXPECT_TRUE(m.take(correlation_key{7, 1, 0}).has_value());  // still retained
}

// Phase 1 rendezvous (wait_take). Replaces the one-shot take() that silently
// used HSA timestamps whenever a firmware record was merely late.

// U3a, deposit-before-waiter: the result is already there, so the wait resolves
// immediately (no deadline consumed) and resolves exactly ONCE.
TEST(ResultsMap, rendezvous_deposit_before_waiter)
{
    auto m   = ResultsMap{};
    auto key = correlation_key{7, 100, 0};
    m.deposit(key, kfd_timing_result{1000, 2000, 500});

    auto start = steady_now_ns();
    auto r     = m.wait_take(key, deadline_in_ms(10'000));
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->start_gpu_ticks, 1000u);
    EXPECT_LT(elapsed_ms_since(start), 1000u);  // did not block

    // Exactly once: the entry is gone, and a second wait only burns its deadline.
    EXPECT_FALSE(m.wait_take(key, deadline_in_ms(10)).has_value());
    EXPECT_EQ(m.stats().hits, 1u);
}

// U3a, waiter-before-deposit: the waiter blocks first and the deposit wakes it.
// Resolves exactly once, with the deposited value.
TEST(ResultsMap, rendezvous_waiter_before_deposit)
{
    auto m   = ResultsMap{};
    auto key = correlation_key{7, 101, 0};

    auto producer = std::thread{[&m, key]() {
        std::this_thread::sleep_for(std::chrono::milliseconds{20});
        m.deposit(key, kfd_timing_result{4242, 5252, 1});
    }};

    // Deadline far beyond the deposit, so resolving proves the CV woke the waiter
    // rather than the deadline expiring.
    auto r = m.wait_take(key, deadline_in_ms(60'000));
    producer.join();

    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->start_gpu_ticks, 4242u);
    EXPECT_EQ(r->end_gpu_ticks, 5252u);
    EXPECT_EQ(m.stats().hits, 1u);
    EXPECT_FALSE(m.wait_take(key, 0).has_value());  // not delivered twice
}

// U4, result-vs-deadline: with no deposit the wait ends at the deadline, reports
// a miss (the caller then uses HSA timestamps), and really did wait.
TEST(ResultsMap, rendezvous_deadline_expires)
{
    auto m     = ResultsMap{};
    auto start = steady_now_ns();
    auto r     = m.wait_take(correlation_key{7, 102, 0}, deadline_in_ms(30));

    EXPECT_FALSE(r.has_value());
    EXPECT_GE(elapsed_ms_since(start), 20u);  // waited rather than returning at once
    EXPECT_EQ(m.stats().misses, 1u);
}

// A deadline of 0 means "do not wait": the plain-take behavior, used by any path
// that must not block.
TEST(ResultsMap, rendezvous_zero_deadline_does_not_block)
{
    auto m     = ResultsMap{};
    auto start = steady_now_ns();
    EXPECT_FALSE(m.wait_take(correlation_key{7, 103, 0}, 0).has_value());
    EXPECT_LT(elapsed_ms_since(start), 1000u);
}

// U4, result-vs-reader-dead: a dead reader can never deposit, so abandoning the
// waiters releases them immediately instead of making them burn the deadline.
TEST(ResultsMap, rendezvous_reader_death_wakes_waiter)
{
    auto m   = ResultsMap{};
    auto key = correlation_key{7, 104, 0};

    auto reaper = std::thread{[&m]() {
        std::this_thread::sleep_for(std::chrono::milliseconds{20});
        m.abandon_waiters();
    }};

    auto start = steady_now_ns();
    auto r     = m.wait_take(key, deadline_in_ms(60'000));
    reaper.join();

    EXPECT_FALSE(r.has_value());
    EXPECT_LT(elapsed_ms_since(start), 30'000u);  // woken, not timed out
}

// Phase 1 loss policy: an overrun / dead reader means the SIGNAL fallback, not
// the Phase 2 LEAKED policy. Once abandoned, every wait returns "no result"
// immediately so the dispatch reports HSA timestamps and completes normally.
TEST(ResultsMap, rendezvous_after_abandon_never_waits)
{
    auto m = ResultsMap{};
    m.abandon_waiters();

    auto start = steady_now_ns();
    EXPECT_FALSE(m.wait_take(correlation_key{7, 105, 0}, deadline_in_ms(60'000)).has_value());
    EXPECT_LT(elapsed_ms_since(start), 30'000u);

    // A result deposited after the abandon is still takeable; abandoning only ends
    // waiting, it does not poison the map.
    auto key = correlation_key{7, 106, 0};
    m.deposit(key, kfd_timing_result{11, 22, 0});
    EXPECT_TRUE(m.wait_take(key, deadline_in_ms(60'000)).has_value());
}

// Unrelated deposits notify every waiter; the predicate must send them back to
// sleep rather than let a wakeup end the wait with no result of their own.
TEST(ResultsMap, rendezvous_ignores_unrelated_wakeups)
{
    auto m          = ResultsMap{};
    auto key        = correlation_key{7, 107, 0};
    auto stop       = std::atomic<bool>{false};
    auto noisemaker = std::thread{[&m, &stop]() {
        for(uint32_t i = 0; !stop.load(); ++i)
        {
            m.deposit(correlation_key{9, i, 0}, kfd_timing_result{1, 2, 0});
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
        }
    }};

    auto start = steady_now_ns();
    auto r     = m.wait_take(key, deadline_in_ms(40));
    stop.store(true);
    noisemaker.join();

    EXPECT_FALSE(r.has_value());              // never resolved on someone else's record
    EXPECT_GE(elapsed_ms_since(start), 30u);  // and the wakeups did not cut the wait short
}

// ONE absolute deadline for the whole batch: the first packet may consume it, but
// a later packet whose record is already present is still served immediately, and
// a later packet with no record does not wait a second full deadline.
TEST(ResultsMap, rendezvous_one_deadline_per_batch)
{
    auto m       = ResultsMap{};
    auto present = correlation_key{7, 201, 0};
    m.deposit(present, kfd_timing_result{31, 41, 0});

    const uint64_t batch_deadline = deadline_in_ms(30);

    // Packet 1: never deposited, consumes the batch deadline.
    auto start = steady_now_ns();
    EXPECT_FALSE(m.wait_take(correlation_key{7, 200, 0}, batch_deadline).has_value());
    EXPECT_GE(elapsed_ms_since(start), 20u);

    // Packet 2: already deposited -> served at once even though the deadline is spent.
    start  = steady_now_ns();
    auto r = m.wait_take(present, batch_deadline);
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->start_gpu_ticks, 31u);
    EXPECT_LT(elapsed_ms_since(start), 1000u);

    // Packet 3: no record, and the shared deadline has passed -> no second timeout.
    start = steady_now_ns();
    EXPECT_FALSE(m.wait_take(correlation_key{7, 202, 0}, batch_deadline).has_value());
    EXPECT_LT(elapsed_ms_since(start), 1000u);
}

// The HSA-fallback counter makes source substitution visible instead of silent.
TEST(ResultsMap, hsa_fallback_is_counted)
{
    auto m = ResultsMap{};
    EXPECT_EQ(m.stats().fallbacks, 0u);
    m.note_hsa_fallback();
    m.note_hsa_fallback();
    EXPECT_EQ(m.stats().fallbacks, 2u);
}

// Phase 1 option (b) feature gate

// kfd_selection_enabled() is the env feature flag AND the fully-wired switch.
// The machinery is complete, so what matters now is that it stays OFF unless the
// operator opts in: with the env var unset -- the default in every test process
// -- get_dispatch_time() skips the whole KFD block.
TEST(kfd_selection_gate, off_by_default_without_the_env_opt_in)
{
    static_assert(signal_less_fully_wired(), "the signal-less machinery is complete");
    EXPECT_FALSE(kfd_selection_enabled());
}

// Queue destroy drops stale results for the dead queue's doorbell slot, so a
// queue reusing the slot cannot take one. Other slots are untouched.
TEST(ResultsMap, erase_slot_drops_only_that_slot)
{
    auto m = ResultsMap{};
    m.deposit(correlation_key{7, 1, 0}, kfd_timing_result{1, 2, 0});
    m.deposit(correlation_key{7, 2, 1}, kfd_timing_result{3, 4, 0});  // later generation
    m.deposit(correlation_key{8, 1, 0}, kfd_timing_result{5, 6, 0});

    EXPECT_EQ(m.erase_slot(0, 7), 2u);
    EXPECT_FALSE(m.take(correlation_key{7, 1, 0}).has_value());
    EXPECT_FALSE(m.take(correlation_key{7, 2, 1}).has_value());
    EXPECT_TRUE(m.take(correlation_key{8, 1, 0}).has_value());

    EXPECT_EQ(m.erase_slot(0, 7), 0u);  // idempotent
}

// A firmware record from one GPU must never match a dispatch enqueued on another.
TEST(correlation_key, gpu_id_prevents_cross_gpu_matching)
{
    auto on_gpu0 = correlation_key{7, 100, 0, /*gpu_id=*/0};
    auto on_gpu1 = correlation_key{7, 100, 0, /*gpu_id=*/1};

    EXPECT_NE(on_gpu0, on_gpu1) << "identical slot/index/generation on two GPUs must not be equal";
    EXPECT_EQ(on_gpu0, (correlation_key{7, 100, 0, 0}));

    auto hash = correlation_key_hash{};
    EXPECT_NE(hash(on_gpu0), hash(on_gpu1));

    // Three-field construction still means GPU 0, so existing single-GPU call
    // sites keep their meaning.
    EXPECT_EQ((correlation_key{7, 100, 0}), on_gpu0);
}

// The same proof at the map level: a result deposited for one GPU cannot be
// taken by the other GPU's dispatch, and each is delivered exactly once.
TEST(ResultsMap, a_result_is_never_taken_across_gpus)
{
    auto m       = ResultsMap{};
    auto on_gpu0 = correlation_key{7, 100, 0, 0};
    auto on_gpu1 = correlation_key{7, 100, 0, 1};

    m.deposit(on_gpu0, kfd_timing_result{1000, 2000, 0});

    // GPU 1's dispatch, same slot/index/generation, must not see it.
    EXPECT_FALSE(m.take(on_gpu1).has_value());

    auto r = m.take(on_gpu0);
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->start_gpu_ticks, 1000u);

    // Both GPUs can have a live result on the same slot at once.
    m.deposit(on_gpu0, kfd_timing_result{11, 22, 0});
    m.deposit(on_gpu1, kfd_timing_result{33, 44, 0});
    EXPECT_EQ(m.take(on_gpu1)->start_gpu_ticks, 33u);
    EXPECT_EQ(m.take(on_gpu0)->start_gpu_ticks, 11u);
}

// Doorbell slot numbers repeat across GPUs, so the generation counter must be
// per-GPU: destroying a queue on one GPU must not invalidate another GPU's live
// dispatches on the same slot number.
TEST(DoorbellMap, generations_are_per_gpu)
{
    auto m = DoorbellMap{};

    auto on_gpu0 = m.bind_and_resolve(/*gpu_id=*/0, qid(1), /*doorbell_off=*/7);
    auto on_gpu1 = m.bind_and_resolve(/*gpu_id=*/1, qid(2), /*doorbell_off=*/7);
    EXPECT_EQ(on_gpu0.generation, 0u);
    EXPECT_EQ(on_gpu1.generation, 0u);
    EXPECT_EQ(on_gpu0.gpu_id, 0u);
    EXPECT_EQ(on_gpu1.gpu_id, 1u);

    // Destroying GPU 0's queue bumps only GPU 0's slot 7.
    m.on_queue_destroyed(qid(1));
    EXPECT_EQ(m.get_generation(0, 7), 1u);
    EXPECT_EQ(m.get_generation(1, 7), 0u) << "another GPU's slot must be untouched";

    // GPU 1's queue keeps resolving at its own generation.
    auto again = m.bind_and_resolve(1, qid(2), 7);
    EXPECT_EQ(again.generation, 0u);

    // A new queue reusing GPU 0's slot picks up the bumped generation.
    auto reused = m.bind_and_resolve(0, qid(3), 7);
    EXPECT_EQ(reused.generation, 1u);
}
