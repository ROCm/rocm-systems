// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "rocjitsu/vm/amdgpu/graphics_stage.h"

#include "rocjitsu/isa/arch/amdgpu/generated/shared/isa_properties.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/amdgpu/hsa_clock.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/pm4.h"
#include "rocjitsu/vm/amdgpu/register_access.h"

#include <bit>
#include <cstring>

namespace rocjitsu {
namespace amdgpu {

Wavefront::Wavefront(ComputeUnitCore &cu, uint32_t wf_id, uint32_t default_wf_size,
                     uint32_t max_wf_size, uint32_t max_sgprs, uint32_t max_vgprs,
                     bool mode_has_gpr_idx_en)
    : cu_(cu), cu_view_(cu, *this), wf_id_(wf_id), wf_size_(default_wf_size),
      default_wf_size_(default_wf_size), max_wf_size_(max_wf_size), max_sgprs_(max_sgprs),
      max_vgprs_(max_vgprs), mode_has_gpr_idx_en_(mode_has_gpr_idx_en),
      memory_wait_checks_enabled_(cu.config().memory_wait_diagnostics !=
                                  MemoryWaitDiagnostics::Off) {}

void Wavefront::update_activity_counts(bool was_active, bool was_runnable) {
  if (!activity_tracked_)
    return;
  const bool active = !is_halted();
  const bool runnable = active && !debug_paused();
  const uint64_t delta = (uint64_t(active) - uint64_t(was_active)) +
                         ((uint64_t(runnable) - uint64_t(was_runnable)) << 32);
  const uint64_t previous = cu_.wave_activity_.fetch_add(delta, std::memory_order_release);
  assert((active || !was_active || static_cast<uint32_t>(previous) != 0) &&
         "active wave count underflow");
  assert((runnable || !was_runnable || (previous >> 32) != 0) && "runnable wave count underflow");
  (void)previous;
}

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

void Wavefront::prepare_gs_register(VectorMemState &state, uint32_t offset, uint32_t source,
                                    uint32_t destination, bool subtract) {
  state.gs_registers = graphics_stage_ ? graphics_stage_->gs_registers() : nullptr;
  if (!state.gs_registers) {
    report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return;
  }
  state.gs_register_index = (offset >> 2) & 15;
  state.elem_size = state.gs_register_index < 8 ? 4 : 8;
  state.num_elems = 1;
  if (source >= num_vgprs() || destination >= num_vgprs() ||
      state.elem_size / 4 > num_vgprs() - destination) {
    report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return;
  }
  state.dst_reg_base = vgpr_alloc().base + destination;
  state.atomic_op = subtract ? AtomicOp::SUB : AtomicOp::ADD;
  // Only the first active lane supplies the operand and receives the return value.
  state.exec_mask = state.lane_mask = exec() & (uint64_t{0} - exec());
  state.store_data.resize(4);
  if (state.lane_mask) {
    const uint32_t operand = RegisterAccess(*this).read_vgpr(vgpr_alloc().base + source,
                                                             std::countr_zero(state.lane_mask));
    std::memcpy(state.store_data.data(), &operand, sizeof(operand));
  }
}

void Wavefront::export_graphics(uint32_t target, uint32_t mask,
                                const std::array<uint32_t, 4> &sources, bool row) {
  if (status_raw() & (1u << 18))
    return;
  if (!exec()) {
    if (graphics_stage_)
      graphics_stage_->export_mask(*this, 0);
    return;
  }
  if (!graphics_stage_ || row) {
    report_instruction_execution_error(InstructionExecutionError::UnimplementedInstruction);
    return;
  }
  for (uint32_t component = 0; component < 4; ++component)
    if ((mask & (1u << component)) && sources[component] >= num_vgprs()) {
      report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
      return;
    }
  RegisterAccess regs(*this);
  graphics_stage_->export_mask(*this, exec());
  for (uint32_t lane = 0; lane < wf_size(); ++lane) {
    if (!(exec() & (uint64_t{1} << lane)))
      continue;
    std::array<uint32_t, 4> values{};
    for (uint32_t component = 0; component < 4; ++component)
      if (mask & (1u << component))
        values[component] = regs.read_vgpr(vgpr_alloc().base + sources[component], lane);
    graphics_stage_->export_lane(*this, lane, target, mask, values);
  }
}

void Wavefront::halt(CpCompletionNotice notice) {
  // Observer snapshots are not instruction-side register consumers.
  SuspendedMemoryWaitCheck disable_wait_check;
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
    set_state(WfState::RUNNING);
  if (state_ == WfState::ENDING && wait_counters_.empty())
    halt();
}

} // namespace amdgpu
} // namespace rocjitsu
