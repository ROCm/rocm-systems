// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/base/api.h"
#include "rocjitsu/vm/amdgpu/lds.h"

#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

namespace rocjitsu::amdgpu {
class ComputeUnitCore;
class Wavefront;
class VectorMemState;
enum class VmAccessOutcome : uint8_t;

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
/// RDNA3/3.5 Table 63 describes both an initial count of participants minus one
/// and a decrement before testing. Barrier arrivals use the participants-minus-one
/// convention. Semaphore release-all advances the generation to wake queued
/// waiters. RDNA3/3.5 preserve banked credits. Other modeled targets clear them.
/// RDNA 3.5 also shares a 4 KiB GDS allocation and streamout registers across CUs.
class GwsDevice {
public:
  static constexpr uint32_t kResourceCount = 64;
  /// Access the per-process GDS bytes used by indexed instructions on this target.
  /// The same lock serializes RMW operations from independently scheduled CUs.
  template <typename Access>
  void access_gds(rj_code_arch_t arch, uint32_t process_id, Access &&access) {
    std::lock_guard lock(mutex_);
    if (arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5)
      access(data_shares_[process_id].memory);
    else {
      assert(arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2);
      access(gds_.try_emplace(process_id, kGfx10GdsSizeKiB).first->second);
    }
  }

  // GFX10 GDS_COMPUTE_MAX_WAVE_ID and the CS ordered-append SGPR term are 12 bits.
  static constexpr uint32_t kComputeWaveIdMask = 0xfff;
  /// Allocate SGPR launch terms in order on this compute ring.
  uint32_t reserve_ordered_ids(uint32_t process_id, uint32_t queue_id, uint32_t count,
                               uint32_t id_count = kComputeWaveIdMask + 1);
  /// Return no value until the ring's logical wave is eligible.
  std::optional<uint32_t> ordered_count(uint32_t process_id, uint32_t queue_id,
                                        uint32_t logical_wave, uint32_t address, uint32_t value,
                                        bool exchange, bool release);
  /// Reset both ring counters. GDS memory survives queue teardown and an IB wave-ID reset.
  void reset_ordered_ids(uint32_t process_id, uint32_t queue_id);

  static constexpr uint32_t kGfx11GdsBytes = 4096;
  static bool prepare_register_access(Wavefront &wf, VectorMemState &d, uint8_t offset0,
                                      uint8_t offset1, uint32_t src, uint32_t dst, bool ordered,
                                      bool subtract);
  VmAccessOutcome execute_gds(VectorMemState &d, Wavefront &wf);
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
  void retire_ordered_dispatch(uint32_t process_id, uint32_t queue_id, uint32_t dispatch_id);
  size_t tracked_process_count();
  /// The checkpoint format omits GDS backing and the ordered-count wave walkers.
  bool has_gds_checkpoint_state();

  /// Whether an indexed GDS write materialized any process backing.
  bool has_gds_data();

  static constexpr uint32_t kGfx10GdsSizeKiB = 64;

private:
  struct DataShare {
    Lds memory{kGfx11GdsBytes / 1024};
    std::array<uint64_t, 16> streamout{};
    std::unordered_map<uint64_t, uint16_t> ordered_walkers;
  };
  std::unordered_map<uint32_t, DataShare> data_shares_;
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
  /// RDNA1/2 process-local GDS persists across dispatches and shares the device RMW lock.
  std::unordered_map<uint32_t, Lds> gds_;
  struct OrderedRing {
    uint32_t id_count = kComputeWaveIdMask + 1;
    uint32_t next_launch = 0;
    uint32_t oldest = 0;
  };
  std::unordered_map<uint32_t, std::unordered_map<uint32_t, OrderedRing>> ordered_rings_;

  std::unordered_map<uint32_t, std::array<Resource, kResourceCount>> resources_;
  std::unordered_map<uint32_t, uint32_t> parked_;
  std::unordered_map<uint32_t, uint64_t> instruction_progress_;
  std::atomic<uint32_t> parked_total_{0};
  uint64_t sema_seq_next_ = 0;
};
} // namespace rocjitsu::amdgpu
