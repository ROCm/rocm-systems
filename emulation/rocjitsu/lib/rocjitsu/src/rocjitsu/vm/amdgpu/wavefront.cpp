// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/wavefront.h"

#include "rocjitsu/isa/arch/amdgpu/generated/shared/isa_properties.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"

namespace rocjitsu {
namespace amdgpu {

Lds &Wavefront::lds() { return lds_ ? *lds_ : cu_.lds(); }

const Lds &Wavefront::lds() const { return lds_ ? *lds_ : cu_.lds(); }

bool Wavefront::uses_separate_trap_ctrl() const {
  return isa_properties(cu_.arch()).wave_state_layout != WaveStateLayout::Legacy;
}

bool Wavefront::has_gpu_memory() const { return cu_.memory() != nullptr; }

void Wavefront::read_gpu_memory(uint64_t addr, std::span<uint8_t> dst) const {
  assert(has_gpu_memory());
  cu_.memory()->read_block(addr, dst, process_id_);
}

void Wavefront::write_gpu_memory(uint64_t addr, std::span<const uint8_t> src) {
  assert(has_gpu_memory());
  cu_.memory()->write_block(addr, src, process_id_);
}

void Wavefront::barrier_init(int32_t barrier_id, uint32_t member_count) {
  cu_.named_barrier_init(*this, barrier_id, member_count);
}

void Wavefront::barrier_join(int32_t barrier_id) { cu_.named_barrier_join(*this, barrier_id); }

bool Wavefront::barrier_signal(int32_t barrier_id, uint32_t member_count) {
  return cu_.barrier_signal(*this, barrier_id, member_count);
}

uint32_t Wavefront::barrier_state(int32_t barrier_id) const {
  return cu_.barrier_state(*this, barrier_id);
}

void Wavefront::barrier_wait(int32_t barrier_id) { cu_.barrier_wait(*this, barrier_id); }

bool Wavefront::barrier_leave() { return cu_.named_barrier_leave(*this); }

void Wavefront::halt(CpCompletionNotice notice) {
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
  cu_.release_wf(dispatch_id, wg_id, notice);
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
