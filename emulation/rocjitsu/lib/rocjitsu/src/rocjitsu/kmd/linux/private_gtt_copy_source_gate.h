// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file private_gtt_copy_source_gate.h
/// @brief Gate private GTT copy-source reads during alias maintenance.

#pragma once

#include "rocjitsu/vm/amdgpu/gpu_vm.h"

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>

namespace rocjitsu {

/// Counted authority for one KFD-owned private GTT source allocation.
///
/// Admission never waits. A mapping writer closes the gate before changing a
/// MAP grant, host view, or export policy, then waits without allocation or VM
/// locks for synchronous source reads to release their permits.
class PrivateGttCopySourceGate final
    : public std::enable_shared_from_this<PrivateGttCopySourceGate> {
public:
  [[nodiscard]] std::optional<amdgpu::VmAccessOutcome> closed_outcome() const noexcept {
    std::lock_guard lock(mutex_);
    if (revoked_)
      return amdgpu::VmAccessOutcome::Revoked;
    if (closed_count_ != 0)
      return amdgpu::VmAccessOutcome::Unavailable;
    return std::nullopt;
  }

  [[nodiscard]] amdgpu::VmPrivateUcGttCopySourceAdmission admit(uint64_t map_generation,
                                                                uint64_t host_view_generation) {
    std::lock_guard lock(mutex_);
    if (revoked_)
      return {.outcome = amdgpu::VmAccessOutcome::Revoked, .permit = {}};
    if (closed_count_ != 0 || map_generation == 0 || host_view_generation == 0 ||
        map_generation != map_generation_ || host_view_generation != host_view_generation_)
      return {.outcome = amdgpu::VmAccessOutcome::Unavailable, .permit = {}};
    auto owner = shared_from_this();
    amdgpu::MemoryIngressPermit permit([owner = std::move(owner)] { owner->release(); });
    ++active_;
    return {.outcome = amdgpu::VmAccessOutcome::Complete, .permit = std::move(permit)};
  }

  void begin_mutation() noexcept {
    std::lock_guard lock(mutex_);
    ++closed_count_;
  }

  void wait_quiescent() {
    std::unique_lock lock(mutex_);
    drained_.wait(lock, [&] { return active_ == 0; });
  }

  void finish_mutation(uint64_t map_generation, uint64_t host_view_generation) noexcept {
    std::lock_guard lock(mutex_);
    map_generation_ = map_generation;
    host_view_generation_ = host_view_generation;
    if (closed_count_ != 0)
      --closed_count_;
  }

  void revoke() noexcept {
    std::lock_guard lock(mutex_);
    revoked_ = true;
  }

private:
  void release() noexcept {
    std::lock_guard lock(mutex_);
    if (--active_ == 0)
      drained_.notify_all();
  }

  mutable std::mutex mutex_;
  std::condition_variable drained_;
  uint64_t map_generation_ = 0;
  uint64_t host_view_generation_ = 0;
  uint32_t active_ = 0;
  uint32_t closed_count_ = 0;
  bool revoked_ = false;
};

} // namespace rocjitsu
