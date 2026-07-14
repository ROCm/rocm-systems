// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/wavefront.h"

#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/command_processor.h"

namespace rocjitsu {
namespace amdgpu {

void Wavefront::halt() {
  cu_.plugin_group().onAmdgpuWavefrontHalted(*this);
  state_ = WfState::HALTED;
  cu_.release_wf(dispatch_id_, wg_id_);
}

void Wavefront::trap() {
  if (auto *cp = cu_.command_processor())
    cp->notify_wave_trap(dispatch_id_);
  else
    halt();
}

void Wavefront::release_wait_counter(WaitCounterType type) {
  wait_counters_.decrement(type);
  if (state_ == WfState::WAITCNT && wait_satisfied())
    state_ = WfState::RUNNING;
  if (state_ == WfState::ENDING && wait_counters_.empty())
    halt();
}

} // namespace amdgpu
} // namespace rocjitsu
