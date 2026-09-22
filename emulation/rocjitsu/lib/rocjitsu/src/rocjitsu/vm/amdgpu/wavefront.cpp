// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "rocjitsu/vm/amdgpu/graphics_stage.h"

#include "rocjitsu/isa/arch/amdgpu/generated/shared/isa_properties.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/amdgpu/hsa_clock.h"
#include "rocjitsu/vm/amdgpu/pm4.h"

namespace rocjitsu {
namespace amdgpu {

Lds &Wavefront::lds() { return lds_ ? *lds_ : cu_.lds(); }

const Lds &Wavefront::lds() const { return lds_ ? *lds_ : cu_.lds(); }

bool Wavefront::uses_separate_trap_ctrl() const {
  return isa_properties(cu_.arch()).wave_state_layout != WaveStateLayout::Legacy;
}

bool Wavefront::has_gpu_memory() const { return cu_.memory() != nullptr; }

uint64_t Wavefront::realtime_timestamp() const {
  if (use_system_clock_)
    return hsa_system_timestamp();
  auto *engine = cu_.engine();
  return engine ? engine->global_time() : 0;
}

std::optional<GpuVmAccess> Wavefront::snapshot_vm_access() const {
  GpuVm *gpu_vm = cu_.gpu_vm();
  if (gpu_vm == nullptr)
    return std::nullopt;
  return address_space_ ? gpu_vm->snapshot(address_space_) : gpu_vm->snapshot_vmid(process_id_);
}

VmAccessOutcome Wavefront::read_gpu_memory(uint64_t addr, std::span<uint8_t> dst) const {
  assert(has_gpu_memory());
  if (address_space_ || process_id_ != 0) {
    const std::optional<GpuVmAccess> access = snapshot_vm_access();
    if (!access)
      return VmAccessOutcome::Faulted;
    return access->read(
        addr, std::span<std::byte>(reinterpret_cast<std::byte *>(dst.data()), dst.size()));
  }
  cu_.memory()->read_block(addr, dst);
  return VmAccessOutcome::Complete;
}

VmAccessOutcome Wavefront::write_gpu_memory(uint64_t addr, std::span<const uint8_t> src) {
  assert(has_gpu_memory());
  if (address_space_ || process_id_ != 0) {
    const std::optional<GpuVmAccess> access = snapshot_vm_access();
    if (!access)
      return VmAccessOutcome::Faulted;
    return access->write(addr, std::span<const std::byte>(
                                   reinterpret_cast<const std::byte *>(src.data()), src.size()));
  }
  cu_.memory()->write_block(addr, src);
  return VmAccessOutcome::Complete;
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

bool Wavefront::fail_pm4_submission() {
  if (!pm4_failure_)
    return false;
  pm4_failure_->fail();
  return true;
}

void Wavefront::export_graphics(uint32_t target, uint32_t mask,
                                const std::array<uint32_t, 4> &sources, bool row) {
  if (!exec() || (status_raw() & (1u << 18)))
    return;
  if (!graphics_stage_ || row) {
    report_instruction_execution_error(InstructionExecutionError::UnimplementedInstruction);
    return;
  }
  for (uint32_t component = 0; component < 4; ++component)
    if ((mask & (1u << component)) && sources[component] >= num_vgprs()) {
      report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
      return;
    }
  for (uint32_t lane = 0; lane < wf_size(); ++lane) {
    if (!(exec() & (uint64_t{1} << lane)))
      continue;
    std::array<uint32_t, 4> values{};
    for (uint32_t component = 0; component < 4; ++component)
      if (mask & (1u << component))
        values[component] = debug_read_vgpr(sources[component], lane);
    graphics_stage_->export_lane(*this, lane, target, mask, values);
  }
}

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
