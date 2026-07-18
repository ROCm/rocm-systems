// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/wavefront.h"

#include "rocjitsu/vm/amdgpu/compute_unit.h"

namespace rocjitsu {
namespace amdgpu {

Lds &Wavefront::lds() { return lds_ ? *lds_ : cu_.lds(); }

const Lds &Wavefront::lds() const { return lds_ ? *lds_ : cu_.lds(); }

void Wavefront::halt() {
  // s_endpgm terminates the wave, frees its resources, and notifies the CP as one
  // action, mirroring hardware. Order matters:
  //   (1) fire the halt hook while registers are still live so observers snapshot
  //       final state before it is freed,
  //   (2) free SGPR/VGPR and reset the slot (sets state HALTED); capture the WG ids
  //       first because reset() zeroes them,
  //   (3) notify the CU/CP of workgroup completion. Freeing before release_wf keeps
  //       has_active_wfs() accurate so the last wave triggers LDS reclaim.
  cu_.plugin_group().onAmdgpuWavefrontHalted(*this);
  const uint32_t dispatch_id = dispatch_id_;
  const uint32_t wg_id = wg_id_;
  cu_.free_wavefront_resources(*this);
  cu_.release_wf(dispatch_id, wg_id);
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
