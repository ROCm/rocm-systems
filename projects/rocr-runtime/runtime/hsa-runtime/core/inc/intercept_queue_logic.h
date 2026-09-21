/*
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

#ifndef HSA_RUNTIME_CORE_INC_INTERCEPT_QUEUE_LOGIC_H_
#define HSA_RUNTIME_CORE_INC_INTERCEPT_QUEUE_LOGIC_H_

#include <cstdint>

// Pure decision logic for InterceptQueue::Submit, factored out so it can be unit
// tested without the HSA runtime or a GPU (regression coverage for the retry/overflow fix).
namespace rocr {
namespace core {
namespace intercept_queue_logic {

struct SubmitPlan {
  // Number of (non-marker) packets to copy to the wrapped queue this iteration.
  uint64_t submitted_count;
  // Whether a fresh retry barrier must be inserted to drain the remaining packets.
  bool insert_retry_barrier;
};

// Decide how many packets fit now and whether a retry barrier is needed. write/read are read
// non-atomically, so a transient read > write is clamped to "queue full" (no underflow/OOB).
inline SubmitPlan PlanSubmit(uint64_t write, uint64_t read, uint64_t qsize, uint64_t count,
                             uint64_t marker_count, bool pending_retry_point,
                             bool overflow_nonempty) {
  uint64_t inflight = (write >= read) ? (write - read) : qsize;
  if (inflight > qsize) inflight = qsize;
  uint64_t free_slots = qsize - inflight;

  const uint64_t non_marker = count - marker_count;
  uint64_t submitted_count = non_marker;

  if (submitted_count >= qsize) {
    // Submit what fits, reserving a slot for the retry barrier (saturating: free_slots
    // may be 0).
    uint64_t reserve = pending_retry_point ? 0 : 1;
    submitted_count = (free_slots > reserve) ? (free_slots - reserve) : 0;
  } else if (free_slots < submitted_count + (pending_retry_point ? 0 : 1)) {
    // Out of space: prefer all-or-nothing, but when draining overflow submit what fits.
    // The minimum footprint to make progress is a data packet plus a retry barrier (2)
    // when none is pending, or just a data packet (1) when a retry is already pending
    // (no new barrier needed). Require strictly more free slots than that footprint so at
    // least one data packet is actually drained beyond the reserved barrier slot.
    if (overflow_nonempty && free_slots > (pending_retry_point ? 1 : 2)) {
      // Reserve one slot for the new retry barrier unless a retry is already pending.
      submitted_count = free_slots - (pending_retry_point ? 0 : 1);
    } else {
      submitted_count = 0;
    }
  }

  // packets[] has exactly non_marker non-marker entries; never copy more than that.
  if (submitted_count > non_marker) submitted_count = non_marker;

  // Need a replacement barrier only when packets remain, none is pending, and a slot is
  // free (a completed-but-unread barrier may still occupy its slot).
  const bool insert_retry_barrier =
      submitted_count < non_marker && !pending_retry_point && free_slots >= 1;

  return SubmitPlan{submitted_count, insert_retry_barrier};
}

}  // namespace intercept_queue_logic
}  // namespace core
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_INTERCEPT_QUEUE_LOGIC_H_
