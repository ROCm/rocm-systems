// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file wf_scheduler.h
/// @brief Pluggable wavefront scheduler for the compute unit.

#ifndef ROCJITSU_VM_AMDGPU_WF_SCHEDULER_H_
#define ROCJITSU_VM_AMDGPU_WF_SCHEDULER_H_

#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <cstdint>
#include <memory>
#include <span>

namespace rocjitsu {
namespace amdgpu {

/// @brief Abstract wavefront scheduler interface.
///
/// @details The CU calls schedule() each step to pick the next wavefront to
/// issue an instruction from. Implementations define the scheduling policy.
class WavefrontScheduler {
public:
  virtual ~WavefrontScheduler() = default;

  /// @brief Pick the next wavefront to issue from.
  /// @param slots All wavefront slots on this CU.
  /// @returns Pointer to the chosen RUNNING wavefront, or nullptr if none ready.
  virtual Wavefront *schedule(std::span<const std::unique_ptr<Wavefront>> slots) = 0;
};

/// @brief Greedy-then-oldest (GTO) scheduler.
///
/// @details Keeps issuing from the same wavefront as long as it stays RUNNING
/// (greedy — maximizes cache locality). On stall, switches to the RUNNING
/// wavefront that became ready earliest (oldest — avoids starvation).
///
/// This is the default scheduler matching AMD's hardware GTO policy.
class OldestFirstScheduler : public WavefrontScheduler {
public:
  Wavefront *schedule(std::span<const std::unique_ptr<Wavefront>> slots) override {
    if (current_ && current_->state() == WfState::RUNNING)
      return current_;

    Wavefront *best = nullptr;
    uint64_t oldest = UINT64_MAX;
    for (auto &w : slots) {
      if (w->state() == WfState::RUNNING && w->ready_cycle() < oldest) {
        oldest = w->ready_cycle();
        best = w.get();
      }
    }
    current_ = best;
    return best;
  }

private:
  Wavefront *current_ = nullptr;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_WF_SCHEDULER_H_
