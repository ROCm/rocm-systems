/*
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

// Unit tests for the pure InterceptQueue retry/overflow decision logic (no HSA/GPU);
// regression coverage for the counter-collection hang and Submit() out-of-bounds fixes.

#include "core/inc/intercept_queue_logic.h"

#include <algorithm>
#include <cstdint>

#include "gtest/gtest.h"

namespace rocr {
namespace core {
namespace intercept_queue_logic {
namespace {

constexpr uint64_t kQSize = 1024;  // power of two, like a real AQL ring

// PlanSubmit: slot accounting. Two invariants prevent the crash: submitted_count never
// exceeds the available non-marker packets, nor free_slots (never overcommits the ring).

TEST(PlanSubmit, EverythingFitsSubmitsAllNoBarrier) {
  // write==read => empty ring, plenty of room for a small rewrite.
  auto p = PlanSubmit(/*write=*/0, /*read=*/0, kQSize, /*count=*/4, /*marker=*/0,
                      /*pending=*/false, /*overflow=*/false);
  EXPECT_EQ(p.submitted_count, 4u);
  EXPECT_FALSE(p.insert_retry_barrier);
}

TEST(PlanSubmit, TransientReadAheadOfWriteDoesNotUnderflow) {
  // Regression: read transiently past the stale write underflowed size-(write-read) -> huge
  // submitted_count -> OOB copy. Must clamp to "queue full" (0 free slots).
  auto p = PlanSubmit(/*write=*/100, /*read=*/130, kQSize, /*count=*/8, /*marker=*/0,
                      /*pending=*/false, /*overflow=*/true);
  EXPECT_EQ(p.submitted_count, 0u);
  EXPECT_FALSE(p.insert_retry_barrier);  // no free slot to hold a barrier
}

TEST(PlanSubmit, InflightLargerThanQueueIsClamped) {
  // (write - read) > qsize must not produce negative/huge free_slots.
  auto p = PlanSubmit(/*write=*/5000, /*read=*/0, kQSize, /*count=*/8, /*marker=*/0,
                      /*pending=*/false, /*overflow=*/true);
  EXPECT_EQ(p.submitted_count, 0u);
}

TEST(PlanSubmit, HugeCountNeverOvercommits) {
  // Regression: count >> qsize must never make submitted_count exceed the input count or
  // the free slots (what walked off the end of packets[]). Ring half full here.
  const uint64_t write = 512, read = 0;  // 512 in flight, 512 free
  auto p = PlanSubmit(write, read, kQSize, /*count=*/16209, /*marker=*/0,
                      /*pending=*/false, /*overflow=*/true);
  EXPECT_LE(p.submitted_count, 16209u);                 // <= available packets
  EXPECT_LE(p.submitted_count, kQSize - (write - read));  // <= free slots
  EXPECT_GT(p.submitted_count, 0u);                     // still makes progress
}

TEST(PlanSubmit, QueueFullNotPendingSubmitsNothingNoBarrier) {
  // Full ring, no pending barrier, small rewrite: must not underflow when reserving
  // a barrier slot, and must not try to insert a barrier into a full queue.
  auto p = PlanSubmit(/*write=*/kQSize, /*read=*/0, kQSize, /*count=*/4, /*marker=*/0,
                      /*pending=*/false, /*overflow=*/false);
  EXPECT_EQ(p.submitted_count, 0u);
  EXPECT_FALSE(p.insert_retry_barrier);
}

TEST(PlanSubmit, PartialFitInsertsRetryBarrier) {
  // More packets than fit, free slot available, none pending: insert a barrier so
  // the remainder drains later.
  const uint64_t write = 1020, read = 0;  // 4 free slots
  auto p = PlanSubmit(write, read, kQSize, /*count=*/100, /*marker=*/0,
                      /*pending=*/false, /*overflow=*/false);
  EXPECT_LT(p.submitted_count, 100u);
  EXPECT_TRUE(p.insert_retry_barrier);
}

TEST(PlanSubmit, PendingBarrierNeverInsertsAnother) {
  // If a retry barrier is already pending, never insert a second one regardless of
  // how little fits.
  const uint64_t write = 1020, read = 0;  // 4 free
  auto p = PlanSubmit(write, read, kQSize, /*count=*/100, /*marker=*/0,
                      /*pending=*/true, /*overflow=*/false);
  EXPECT_FALSE(p.insert_retry_barrier);
}

TEST(PlanSubmit, MarkerPacketsExcludedFromSubmitCount) {
  // Only non-marker packets are copied; submitted_count must be bounded by them.
  auto p = PlanSubmit(/*write=*/0, /*read=*/0, kQSize, /*count=*/10, /*marker=*/3,
                      /*pending=*/false, /*overflow=*/false);
  EXPECT_EQ(p.submitted_count, 7u);  // 10 - 3 markers
}

// Slots the read index has freed may, rarely, still carry the valid header of the packet the
// processor fetched from them (its INVALID store not yet visible). Only retired slots may be
// written.

TEST(PlanSubmit, RetiredSlotsCapWhatIsWritten) {
  // 8 slots free by the read index, but only 4 of them retired: 3 packets are written and
  // the fourth retired slot takes the retry barrier for the rest.
  const uint64_t write = kQSize - 8, read = 0;
  auto p = PlanSubmit(write, read, kQSize, /*count=*/6, /*marker=*/0,
                      /*pending=*/false, /*overflow=*/true, /*retired=*/4);
  EXPECT_EQ(p.submitted_count, 3u);
  EXPECT_TRUE(p.insert_retry_barrier);
  EXPECT_FALSE(p.needs_wakeup);
}

TEST(PlanSubmit, NoRetiredSlotWritesNothingAndNeedsWakeup) {
  // Free by the read index, nothing retired yet: no packet, no barrier, and since no retry
  // is pending nothing on the device will wake the queue.
  const uint64_t write = kQSize - 8, read = 0;
  auto p = PlanSubmit(write, read, kQSize, /*count=*/6, /*marker=*/0,
                      /*pending=*/false, /*overflow=*/true, /*retired=*/0);
  EXPECT_EQ(p.submitted_count, 0u);
  EXPECT_FALSE(p.insert_retry_barrier);
  EXPECT_TRUE(p.needs_wakeup);
}

TEST(PlanSubmit, FullRingWithoutPendingRetryNeedsWakeup) {
  // The retry barrier completed but the read index has not yet caught up (ring reads full):
  // the overflow cannot drain and no barrier can be inserted. This is the stranded state of
  // the HIP graph replay deadlock; the caller must wait or re-arm.
  auto p = PlanSubmit(/*write=*/kQSize, /*read=*/0, kQSize, /*count=*/124, /*marker=*/0,
                      /*pending=*/false, /*overflow=*/true);
  EXPECT_EQ(p.submitted_count, 0u);
  EXPECT_FALSE(p.insert_retry_barrier);
  EXPECT_TRUE(p.needs_wakeup);
}

TEST(PlanSubmit, PendingRetryNeverNeedsWakeup) {
  // An outstanding retry barrier will wake the drain, however little fits now.
  auto p = PlanSubmit(/*write=*/kQSize, /*read=*/0, kQSize, /*count=*/124, /*marker=*/0,
                      /*pending=*/true, /*overflow=*/true);
  EXPECT_EQ(p.submitted_count, 0u);
  EXPECT_FALSE(p.needs_wakeup);
}

TEST(PlanSubmit, NothingLeftNeedsNoWakeup) {
  auto p = PlanSubmit(/*write=*/0, /*read=*/0, kQSize, /*count=*/4, /*marker=*/0,
                      /*pending=*/false, /*overflow=*/false);
  EXPECT_EQ(p.submitted_count, 4u);
  EXPECT_FALSE(p.needs_wakeup);
}

// Exhaustive guard: across a wide range of (possibly inconsistent) snapshots, the two
// safety invariants must always hold so the ring copy can never go OOB or overcommit.
TEST(PlanSubmit, InvariantsHoldOverManySnapshots) {
  const uint64_t qsize = 64;
  for (uint64_t write = 0; write <= 2 * qsize; ++write) {
    for (uint64_t read = 0; read <= 2 * qsize; ++read) {
      for (uint64_t count = 0; count <= 3 * qsize; count += 7) {
        for (uint64_t marker = 0; marker <= count; marker += 5) {
          for (int pending = 0; pending < 2; ++pending) {
            for (int ovf = 0; ovf < 2; ++ovf) {
              for (uint64_t retired : {uint64_t{0}, uint64_t{1}, uint64_t{UINT64_MAX}}) {
                auto p = PlanSubmit(write, read, qsize, count, marker, pending, ovf, retired);
                const uint64_t non_marker = count - marker;
                const uint64_t inflight = (write >= read)
                                              ? std::min(write - read, qsize)
                                              : qsize;
                const uint64_t free_slots = std::min(qsize - inflight, retired);
                EXPECT_LE(p.submitted_count, non_marker);
                EXPECT_LE(p.submitted_count, free_slots);
                if (p.insert_retry_barrier) {
                  EXPECT_GE(free_slots, 1u);
                  EXPECT_FALSE(pending);
                  EXPECT_LT(p.submitted_count, non_marker);
                }
                // Packets left behind are either covered by a retry barrier (pending or
                // inserted now) or reported so the caller waits for the processor; a wakeup is
                // never reported when a barrier covers the remainder, when nothing is left
                // behind, or when a slot could be used.
                const bool left_behind = p.submitted_count < non_marker;
                const bool covered = pending || p.insert_retry_barrier;
                if (left_behind && !covered) EXPECT_TRUE(p.needs_wakeup);
                if (!left_behind) EXPECT_FALSE(p.needs_wakeup);
                if (p.needs_wakeup) {
                  EXPECT_FALSE(covered);
                  EXPECT_EQ(p.submitted_count, 0u);
                  EXPECT_EQ(free_slots, 0u);
                }
              }
            }
          }
        }
      }
    }
  }
}

}  // namespace
}  // namespace intercept_queue_logic
}  // namespace core
}  // namespace rocr
