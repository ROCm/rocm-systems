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
#include "lib/rocprofiler-sdk/kfd/signal_less_gate.hpp"

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
    EXPECT_FALSE(kfd_time_is_sane(99, 250, 100, 300));  // starts before enqueue
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
    m.on_queue_destroyed(qid(1));       // doorbell 4 -> gen bumped to 1

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
