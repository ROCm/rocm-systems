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
  // Packets remain, none can be written now, and no retry barrier is outstanding or being
  // inserted, so nothing on the device will wake this queue again: the caller has to wait for
  // the packet processor to release a slot (or arrange its own wakeup) before giving up.
  bool needs_wakeup;
};

// Decide how many packets fit now and whether a retry barrier is needed. write/read are read
// non-atomically, so a transient read > write is clamped to "queue full" (no underflow/OOB).
//
// retired_slots caps the free slots. The HSA System Architecture specification (Queues) lets
// a producer write a slot only once read_index has passed it and its format field reads
// INVALID. The packet processor normally stores the INVALID header of a consumed packet before
// it moves the read index past it, but rarely (about 1e-7 per packet on MI250) that store
// becomes visible only after the read index has moved. A slot whose header still reads valid is
// not reusable yet: a header written into it now is overwritten by the late INVALID store,
// which leaves a hole the processor waits at forever. The caller passes the number of
// consecutive slots at the write index whose header already reads INVALID.
inline SubmitPlan PlanSubmit(uint64_t write, uint64_t read, uint64_t qsize, uint64_t count,
                             uint64_t marker_count, bool pending_retry_point,
                             bool overflow_nonempty, uint64_t retired_slots = UINT64_MAX) {
  uint64_t inflight = (write >= read) ? (write - read) : qsize;
  if (inflight > qsize) inflight = qsize;
  uint64_t free_slots = qsize - inflight;
  if (retired_slots < free_slots) free_slots = retired_slots;

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

  const bool needs_wakeup =
      submitted_count < non_marker && !pending_retry_point && !insert_retry_barrier;

  return SubmitPlan{submitted_count, insert_retry_barrier, needs_wakeup};
}

}  // namespace intercept_queue_logic
}  // namespace core
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_INTERCEPT_QUEUE_LOGIC_H_
