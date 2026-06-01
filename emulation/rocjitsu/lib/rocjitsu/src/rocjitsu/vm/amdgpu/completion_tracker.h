// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_AMDGPU_COMPLETION_TRACKER_H_
#define ROCJITSU_VM_AMDGPU_COMPLETION_TRACKER_H_

/// @file completion_tracker.h
/// @brief EOP-like completion tracking: per-dispatch WG retirement and
/// in-order signal firing per queue.

#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/dispatch_entry.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <vector>

namespace rocjitsu {
namespace amdgpu {

class CompletionTracker {
public:
  using InterruptCallback = std::function<void(uint32_t process_id, uint32_t event_id)>;

  CompletionTracker(GpuMemory *mem, std::vector<ComputeUnitCore *> &cus)
      : memory_(mem), cus_(cus) {}

  void set_interrupt_callback(InterruptCallback cb) { interrupt_cb_ = std::move(cb); }

  /// @brief Notify that a workgroup has completed all its wavefronts.
  void notify_wg_complete(uint32_t dispatch_id, uint32_t wg_id, std::vector<HwQueueState> &queues);

  /// @brief Scan all queues and fire completion signals for retired dispatches.
  void drain_completions(std::vector<HwQueueState> &queues);

  /// @brief Flush L1/L2 caches before firing a completion signal.
  void flush_caches(uint32_t vmid = 0);

  /// @brief Check if all queues have no pending entries.
  bool all_complete(const std::vector<HwQueueState> &queues) const;

private:
  void fire_signal(const DispatchEntry &entry);

  GpuMemory *memory_;
  std::vector<ComputeUnitCore *> &cus_;
  InterruptCallback interrupt_cb_;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_COMPLETION_TRACKER_H_
