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
#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/plugins/execution_plugin_group.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace rocjitsu {
namespace amdgpu {

struct CompletionDrainFault {
  uint32_t queue_id = 0;
  uint32_t process_id = 0;
  uint64_t dispatch_id = 0;
  VmAccessOutcome outcome = VmAccessOutcome::Faulted;
  bool queue_idle = false;
};

struct CompletionDrainResult {
  bool progress_made = false;
  bool retry_pending = false;
  std::optional<CompletionDrainFault> terminal_fault;
};

class CompletionTracker {
public:
  using DispatchRetiredCallback = std::function<void(const DispatchEntry &entry)>;
  /// Called by the one shard whose publish completes a fanned-out dispatch, so
  /// every participating XCD can be woken: the owner to fire the completion
  /// signal, and any peer parked behind a barrier bit waiting on this dispatch.
  using GridRetiredCallback = std::function<void(const DispatchEntry &entry)>;

  CompletionTracker(GpuMemory *mem, GpuVm *gpu_vm, std::vector<ComputeUnitCore *> &cus)
      : memory_(mem), gpu_vm_(gpu_vm), cus_(cus) {}

  void set_plugin_group(std::shared_ptr<ExecutionPluginGroup> pg) {
    plugin_group_ = pg ? pg : ExecutionPluginGroup::empty_group();
  }

  void set_dispatch_retired_callback(DispatchRetiredCallback cb) {
    dispatch_retired_cb_ = std::move(cb);
  }
  void set_grid_retired_callback(GridRetiredCallback cb) { grid_retired_cb_ = std::move(cb); }

  /// @brief Notify that a workgroup has completed all its wavefronts.
  void notify_wg_complete(uint32_t dispatch_id, uint32_t wg_id, std::vector<HwQueueState> &queues);

  /// @brief Scan all queues and fire completion signals for retired dispatches.
  [[nodiscard]] CompletionDrainResult drain_completions(std::vector<HwQueueState> &queues);

  /// @brief Flush L1/L2 caches before firing a completion signal.
  void flush_caches(uint32_t vmid = 0);

  /// @brief Check if all queues have no pending entries.
  bool all_complete(const std::vector<HwQueueState> &queues) const;

private:
  [[nodiscard]] VmAccessOutcome advance_signal_publication(DispatchEntry &entry);
  [[nodiscard]] VmAccessOutcome advance_queue_idle_publication(QueueIdlePublicationState &state);
  [[nodiscard]] VmAccessOutcome read_gpu(AddressSpaceHandle address_space,
                                         const GpuVmAccess *access, uint64_t address,
                                         void *destination, size_t size, uint32_t process_id) const;
  [[nodiscard]] VmAccessOutcome atomic_store_gpu(AddressSpaceHandle address_space,
                                                 const GpuVmAccess *access, uint64_t address,
                                                 uint32_t width, uint64_t value,
                                                 uint32_t process_id);
  [[nodiscard]] AtomicCompareExchangeResult
  compare_exchange_gpu(AddressSpaceHandle address_space, const GpuVmAccess *access,
                       uint64_t address, uint32_t width, uint64_t expected, uint64_t desired,
                       uint32_t process_id);

  GpuMemory *memory_;
  GpuVm *gpu_vm_;
  std::vector<ComputeUnitCore *> &cus_;
  DispatchRetiredCallback dispatch_retired_cb_;
  GridRetiredCallback grid_retired_cb_;
  std::shared_ptr<ExecutionPluginGroup> plugin_group_ = ExecutionPluginGroup::empty_group();
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_COMPLETION_TRACKER_H_
