/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

// Published-watermark ring publish state (AIRUNTIME-2291).
//
// The one write-index counter is split in two so the GPU frontier can never pass
// an unpublished slot:
//   reserve_    CLR-private allocation head. Producers draw a unique slot index
//               lock-free (fetch_add). The GPU never reads it.
//   committed_  Published watermark: the highest index with all slots below it
//               published. Only this is stored into the real write_dispatch_id.
//
// A producer reserves a slot, writes body and header into the ring, then calls
// MarkReadyAndCommit(index): it sets the ready bit, scans forward over every
// contiguously-ready slot, and pushes committed_ (hence write_dispatch_id) to the
// new frontier. Reserve and the ready-bit store are lock-free; the scan and push
// run under a tiny per-ring spinlock so only one thread advances the frontier at a
// time.

#include <atomic>
#include <cstdint>
#include <vector>

#include "os/os.hpp"

namespace amd::roc {

// Per-slot publication state. A dedicated ready byte per ring slot (not the AQL
// header) so the CPU-side "is slot N ready?" test never races the CP reading the
// header. The byte holds a lap generation so a stale ready flag from a prior lap
// does not count after ring wrap-around.
struct alignas(64) RingPublishState {
  explicit RingPublishState(uint32_t ring_size)
      : slot_mask_(ring_size - 1), slot_bits_(Log2(ring_size)), ready_(ring_size) {
    for (auto& b : ready_) b.store(0, std::memory_order_relaxed);
  }

  // Lock-free: draw a unique, monotonically increasing slot index. Mirrors the
  // semantics of hsa_queue_add_write_index (returns the OLD value) but touches only
  // CLR-private state. The GPU frontier is untouched here.
  uint64_t Reserve(uint64_t count = 1) {
    return reserve_.fetch_add(count, std::memory_order_relaxed);
  }

  // Mark [index, index+count) published, then try to advance the frontier.
  // Returns {advanced, frontier}: advanced == true iff committed_ moved; frontier
  // is the new watermark (== highest contiguously-published index + 1). The caller
  // rings the doorbell to (frontier - 1) only when advanced is true.
  struct CommitResult {
    bool advanced;
    uint64_t frontier;  // one past the last contiguously-published slot
  };

  // \p push_frontier stores the new frontier into the GPU-visible write_dispatch_id.
  // It is invoked WHILE THE COMMIT LOCK IS HELD: if two committers computing frontiers
  // 6 and 10 stored unlocked, 6 could land after 10 and shove the GPU frontier
  // BACKWARDS, re-stranding slots 7..9. The lock makes the push monotonic.
  template <typename PushFrontier>
  CommitResult MarkReadyAndCommit(uint64_t index, uint64_t count, PushFrontier push_frontier) {
    // Stamp this lap's generation into each slot's ready byte (release: the ring
    // body/header stores done by the caller happen-before this becomes visible).
    for (uint64_t i = 0; i < count; ++i) {
      const uint64_t idx = index + i;
      const uint8_t gen = Generation(idx);
      ready_[idx & slot_mask_].store(gen, std::memory_order_release);
    }

    // Every publisher attempts the advance; there is no "only the watermark owner
    // advances" gate. Such a gate can strand a just-published slot (lost-wakeup: the
    // committer reads the ready bit a hair too early, the frontier stops below the
    // slot, the GPU idles at read==write and the waiter hangs). Instead the scan runs
    // under the lock against a FRESH committed_, so whoever takes the lock last sweeps
    // every contiguously-ready slot. Readiness is monotonic within a lap, so no
    // publish is ever lost.
    while (commit_lock_.exchange(1, std::memory_order_acquire) != 0) {
      // CPU pause, not sched_yield(): the critical section is a few loads, so a syscall
      // per failed acquire would dwarf the work it guards.
      amd::Os::spinPause();
    }

    uint64_t frontier = committed_.load(std::memory_order_relaxed);
    const uint64_t start = frontier;
    while (IsReady(frontier)) {
      ++frontier;
    }
    if (frontier != start) {
      committed_.store(frontier, std::memory_order_release);
      // Push the GPU frontier under the lock so it is monotonic (see above).
      push_frontier(frontier);
    }

    commit_lock_.store(0, std::memory_order_release);
    // "advanced" == the frontier now covers this producer's slots (advanced by us or
    // a concurrent committer). The caller rings the doorbell to (frontier - 1) when
    // true; a stale/racing doorbell VALUE is tolerated by MULTI queues, so the
    // doorbell alone stays outside the lock.
    const bool advanced = frontier > index;
    return {advanced, frontier};
  }

 private:
  // The lap number for a given absolute index. ready_ stores (lap % 254) + 1 so 0
  // always means "not published this lap" even after wrap-around. slot_bits_ is
  // precomputed (log2 ring size) so this is a shift, not a per-call bit-count loop.
  // The scan below calls this once per examined slot, so it must be cheap.
  uint8_t Generation(uint64_t index) const {
    return static_cast<uint8_t>(((index >> slot_bits_) % 254) + 1);
  }

  bool IsReady(uint64_t index) const {
    return ready_[index & slot_mask_].load(std::memory_order_acquire) == Generation(index);
  }

  static uint32_t Log2(uint64_t v) {
    uint32_t bits = 0;
    while (v > 1) {
      ++bits;
      v >>= 1;
    }
    return bits;
  }

  const uint64_t slot_mask_;
  const uint32_t slot_bits_;
  std::atomic<uint64_t> reserve_{0};    //!< CLR-private allocation head (GPU never sees)
  std::atomic<uint64_t> committed_{0};  //!< Published watermark (mirrored to write_dispatch_id)
  std::atomic<uint32_t> commit_lock_{0};  //!< Tiny spinlock guarding the frontier scan/store
  std::vector<std::atomic<uint8_t>> ready_;  //!< Per-slot publication generation bytes
};

}  // namespace amd::roc
