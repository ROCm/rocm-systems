// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file completion_tracker.h
/// @brief EOP-like completion tracking: per-dispatch WG retirement and
/// in-order signal firing per queue.

#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/dispatch_entry.h"
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

class L2Cache;

struct CompletionDrainFault {
  uint32_t queue_id = 0;
  uint32_t process_id = 0;
  uint64_t dispatch_id = 0;
  VmAccessOutcome outcome = VmAccessOutcome::Faulted;
  bool queue_idle = false;
};

class CompletionDrainResult {
public:
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

  CompletionTracker(GpuVm &gpu_vm, std::vector<ComputeUnitCore *> &cus,
                    std::vector<L2Cache *> &l2_caches)
      : gpu_vm_(gpu_vm), cus_(cus), l2_caches_(l2_caches) {}

  void set_plugin_group(std::shared_ptr<ExecutionPluginGroup> pg) {
    plugin_group_ = pg ? pg : ExecutionPluginGroup::empty_group();
  }

  void set_dispatch_retired_callback(DispatchRetiredCallback cb) {
    dispatch_retired_cb_ = std::move(cb);
  }
  void set_grid_retired_callback(GridRetiredCallback cb) { grid_retired_cb_ = std::move(cb); }

  /// @brief Notify that a workgroup has completed all its wavefronts.
  void notify_wg_complete(uint32_t dispatch_id, uint32_t wg_id,
                          std::vector<AqlQueueRecord> &queues);

  /// @brief Scan all queues and fire completion signals for retired dispatches.
  [[nodiscard]] CompletionDrainResult drain_completions(std::vector<AqlQueueRecord> &queues);

  /// @brief Deliver an out-of-order-ready non-kernel packet's completion.
  [[nodiscard]] VmAccessOutcome complete_non_kernel(DispatchEntry &entry);

  /// @brief Flush L1/L2 caches before firing a completion signal.
  void flush_caches(uint32_t vmid = 0);

  /// @brief Check if all queues have no pending entries.
  bool all_complete(const std::vector<AqlQueueRecord> &queues) const;

private:
  [[nodiscard]] VmAccessOutcome deliver_completion(DispatchEntry &entry);
  [[nodiscard]] VmAccessOutcome advance_signal_publication(DispatchEntry &entry);
  [[nodiscard]] VmAccessOutcome advance_queue_idle_publication(QueueIdlePublicationState &state);
  [[nodiscard]] VmAccessOutcome read_gpu(const GpuVmAccess &access, uint64_t address,
                                         void *destination, size_t size) const;
  [[nodiscard]] VmAccessOutcome atomic_store_gpu(const GpuVmAccess &access, uint64_t address,
                                                 uint32_t width, uint64_t value);
  [[nodiscard]] AtomicCompareExchangeResult compare_exchange_gpu(const GpuVmAccess &access,
                                                                 uint64_t address, uint32_t width,
                                                                 uint64_t expected,
                                                                 uint64_t desired);

  GpuVm &gpu_vm_;
  std::vector<ComputeUnitCore *> &cus_;
  std::vector<L2Cache *> &l2_caches_;
  DispatchRetiredCallback dispatch_retired_cb_;
  GridRetiredCallback grid_retired_cb_;
  std::shared_ptr<ExecutionPluginGroup> plugin_group_ = ExecutionPluginGroup::empty_group();
};

} // namespace amdgpu
} // namespace rocjitsu
