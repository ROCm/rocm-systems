// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace rocjitsu::amdgpu {
class ComputeUnitCore;
class Wavefront;

/// Device-global GWS resources, isolated by process and retained across dispatches.
/// KFD grants one queue per process/GPU the whole 64-resource namespace. Direct
/// instruction execution does not require an ioctl grant. The emulator restricts
/// grants to compute queues and releases to the owning queue.
///
/// INIT seeds the shared barrier/semaphore counter. BARRIER requires INIT:
/// positive counters decrement and park; the arrival observing zero releases the
/// queued waves and reloads from its own operand. V/BR add credits, P consumes or
/// parks, and RELEASE_ALL releases existing waiters. Credits belong to queued
/// waiters before later arrivals, including across CUs.
///
/// No residency bound or quiescence observation releases a wait. The CU enforces
/// a configurable inactivity budget and faults a dispatch whose wait exceeds it.
/// A timeout is not proof of deadlock: a later host submission could still signal.
///
/// The store mutex is a leaf lock under the owning CU's wave lock. Signals update
/// shared credits/generations; only the owning CU polls and mutates its waves.
/// No peer CU is dereferenced, and failure callbacks run after releasing the store.
/// Per-ISA qualifications where the manuals describe different behavior:
///  * Barrier release ordering: CDNA1-3 and RDNA1/2 test the counter before
///    decrementing; RDNA3/3.5 (Table 63) decrement first, and their
///    "participants - 1" wording would make @c init(1) followed by the first
///    @c barrier(1) release immediately. The model implements the CDNA/LLVM
///    test-then-decrement reading uniformly; the RDNA3/3.5 decrement-first
///    variant is not modeled.
///  * @c release_all credit accounting: CDNA1-3 and RDNA1/2 assign the queued-
///    wave count to the counter (so with no queued waiter it clears to zero),
///    whereas RDNA3/3.5 (Table 63) add the queued-wave count (leaving prior
///    banked credits intact). The model implements the assignment reading
///    (credits cleared); the RDNA3/3.5 additive variant is not modeled.
class GwsDevice {
public:
  static constexpr uint32_t kResourceCount = 64;
  void init(Wavefront &wf, uint32_t rid, uint32_t count);
  /// Returns false for a barrier on a resource that has not been initialized.
  bool barrier_arrive(ComputeUnitCore &cu, Wavefront &wf, uint32_t rid, uint32_t count);
  void sema_v(ComputeUnitCore &cu, Wavefront &wf, uint32_t rid);
  void sema_p(ComputeUnitCore &cu, Wavefront &wf, uint32_t rid);
  void sema_br(ComputeUnitCore &cu, Wavefront &wf, uint32_t rid, uint32_t count);
  void sema_release_all(ComputeUnitCore &cu, Wavefront &wf, uint32_t rid);
  void notify_parked_wave_gone(const Wavefront &wf);
  /// @brief Poll local releases and refresh inactivity budgets on process instruction progress.
  void poll_waits(ComputeUnitCore &cu);
  /// @brief Reset pending waits' inactivity budgets on real process instruction progress.
  void note_instruction_progress(uint32_t process_id);
  /// @brief Whether any CU has parked GWS waiters; gates polling and progress tracking.
  bool any_parked() const { return parked_total_.load(std::memory_order_acquire) != 0; }
  /// Called only after the process's waves and queues have drained.
  void release_process(uint32_t process_id);
  size_t tracked_process_count();

private:
  struct Resource {
    uint32_t counter = 0; ///< Shared barrier remaining-count and semaphore credits.
    bool armed = false;
    uint64_t release_gen = 0;
    std::vector<uint64_t> sema_wait_seqs;
  };
  void release_sema_waiter(Resource &res, uint64_t seq);
  void park_wave(Wavefront &wf, uint32_t rid, bool is_barrier, uint64_t generation);
  void wake_wave(ComputeUnitCore &cu, Wavefront &wf);
  uint32_t poll_local_waiters(ComputeUnitCore &cu);

  std::mutex mutex_;
  std::unordered_map<uint32_t, std::array<Resource, kResourceCount>> resources_;
  std::unordered_map<uint32_t, uint32_t> parked_;
  std::unordered_map<uint32_t, uint64_t> instruction_progress_;
  std::atomic<uint32_t> parked_total_{0};
  uint64_t sema_seq_next_ = 0;
};
} // namespace rocjitsu::amdgpu
