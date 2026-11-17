// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/memory_wait_scoreboard.h"
#include "rocjitsu/isa/arch/amdgpu/cdna1/addr_calc.h"
#include "rocjitsu/isa/arch/amdgpu/cdna2/register_destinations.h"
#include "rocjitsu/isa/arch/amdgpu/shared/accvgpr_layout.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_buffer.h"
#include "rocjitsu/isa/arch/amdgpu/shared/branch_stack.h"
#include "rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h"
#include "rocjitsu/isa/arch/amdgpu/shared/flat_address.h"
#include "rocjitsu/isa/arch/amdgpu/shared/image_resource.h"
#include "rocjitsu/isa/arch/amdgpu/shared/register_modifiers.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_read.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_resolve.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/buffer_format.h"
#include "rocjitsu/vm/amdgpu/hwreg.h"
#include "rocjitsu/vm/amdgpu/ordered_count.h"

#include <algorithm>
#include <bit>
#include <limits>
#include <utility>

namespace rocjitsu::amdgpu {

std::optional<uint32_t> RegisterAccess::scalar_control_value(uint32_t selector) const {
  const auto &wf = wavefront();
  const auto m0 = scalar_m0_selector(wf.cu().arch());
  if (!can_resolve_src_scalar(selector, wf.cu().arch()))
    return std::nullopt;
  // Preserve invalid-backing rejection for address planning. Constants and
  // computed aliases have no storage range and use the shared value decoder.
  if (selector < 128 && !resolve_scalar_register_range(wf, selector, 1))
    return std::nullopt;
  return resolve_src_scalar_value<[](const Wavefront &wave, uint32_t reg) {
    return RegisterAccess(wave).cu_->read_sgpr_storage(wave.sgpr_alloc().base + reg);
  },
                                  [](const Wavefront &wave, uint32_t reg) {
                                    return RegisterAccess::shader_can_access_ttmp(wave)
                                               ? wave.ttmp(reg)
                                               : 0u;
                                  }>(wf, selector, m0);
}

std::optional<uint64_t> RegisterAccess::vector_control_value(RegisterRef reg, unsigned lane) const {
  const auto &wf = wavefront();
  if (reg.cls != RegClass::VGPR || !reg.width || reg.width > 2 || lane >= wf.wf_size())
    return std::nullopt;
  const auto base = wf.vgpr_alloc().base + reg.index;
  if (!cu_->owns_vgpr_range(wf, base, reg.width))
    return std::nullopt;
  uint64_t value = cu_->read_vgpr_storage(base, lane);
  if (reg.width == 2)
    value |= uint64_t{cu_->read_vgpr_storage(base + 1, lane)} << 32;
  return value;
}

std::optional<RegisterRef> RegisterAccess::source_register(const Operand &op, bool wordwise) const {
  if (!op.reads_value() || op.size_bits() == 0)
    return std::nullopt;
  const auto &wf = wavefront();
  auto width = static_cast<uint8_t>(std::max(1, op.size_bits() / 32));
  // Immediate encodings can overlap the numeric VGPR selector range.
  if (auto base = op.is_vgpr() ? op.simd_vgpr_base(wf) : std::nullopt) {
    if (auto mapped = remapped_valu_source(op, *base, width)) {
      // GFX11.5 selects the base of the whole operand. CDNA1 can substitute
      // individual words of a pair, which the per-word scanner handles.
      if (cu_->arch() == ROCJITSU_CODE_ARCH_RDNA3_5 || (width == 2 && mapped->hi == mapped->lo + 1))
        return RegisterRef{RegClass::VGPR, static_cast<uint16_t>(mapped->lo - wf.vgpr_alloc().base),
                           width};
      if (width == 1 || mapped->lo == mapped->hi)
        return RegisterRef{RegClass::VGPR, static_cast<uint16_t>(mapped->lo - wf.vgpr_alloc().base),
                           1};
    }
    // Qualified memory bodies acquire source dwords independently. Preserve a
    // backed prefix when a later word lies beyond this wave's ownership block.
    if (wordwise)
      while (width && !cu_->owns_vgpr_range(wf, *base, width))
        --width;
    if (!width || !cu_->owns_vgpr_range(wf, *base, width))
      return std::nullopt;
    return RegisterRef{RegClass::VGPR, static_cast<uint16_t>(*base - wf.vgpr_alloc().base), width};
  }
  if (op.const_value())
    return std::nullopt;
  if (auto reg = op.to_register_ref()) {
    if (reg->cls == RegClass::TTMP && !wf.shader_can_access_ttmp())
      return std::nullopt;
    return reg;
  }
  if (auto special = op.to_special_reg_class())
    return RegisterRef{*special, 0, width};
  if (op.has_register_selector()) {
    const auto selector = op.encoding_value();
    if (auto range = resolve_scalar_register_range(wf, selector, width))
      return range->register_ref();
    if (selector == 251 || selector == 252)
      return RegisterRef{selector == 251 ? RegClass::VCC : RegClass::EXEC, 0,
                         static_cast<uint8_t>(wf.wf_size() / 32)};
    if (selector == 253)
      return RegisterRef{RegClass::SCC, 0, 1};
  }
  return std::nullopt;
}

std::optional<RegisterRef> RegisterAccess::gfx10_valu_source_vgpr(const Operand &op) const {
  const auto &wf = wavefront();
  const auto &allocation = wf.vgpr_alloc();
  if (!op.reads_value() || !op.size_bits() || !allocation.count)
    return std::nullopt;
  const auto base = op.simd_vgpr_base(wf);
  if (!base)
    return std::nullopt;
  const uint32_t first = gfx10_source_word(*base);
  const uint32_t index = first - allocation.base;
  const uint32_t width =
      first == *base ? std::min<uint32_t>(op.vgpr_count(), allocation.count - index) : 1;
  return RegisterRef{RegClass::VGPR, static_cast<uint16_t>(index), static_cast<uint8_t>(width)};
}

std::array<std::optional<RegisterRef>, 4>
RegisterAccess::buffer_resource_registers(const Operand &op, unsigned scalar_words) const {
  std::array<std::optional<RegisterRef>, 4> result{};
  const auto &wf = wavefront();
  const auto selector = op.encoding_value();
  if (scalar_words ? !resolve_scalar_register_range(wf, selector, 2)
                   : !addr_calc::buffer_resource_range_is_backed(wf, selector))
    return result;
  const unsigned words = scalar_words ? scalar_words : 4;
  for (unsigned word = 0; word < words; ++word)
    if (const auto range = resolve_scalar_register_range(wf, selector + word, 1))
      result[word] = range->register_ref();
  return result;
}

std::optional<RegisterRef> RegisterAccess::block_data_register(const Operand &op, uint32_t word,
                                                               bool write) const {
  assert(word < kBlockDwordCount);
  const auto &wf = wavefront();
  const auto base = write ? op.simd_vgpr_base_mut(mutable_wavefront()) : op.simd_vgpr_base(wf);
  if (!base)
    return std::nullopt;
  const uint32_t physical = write ? *base + word : block_store_source_vgpr(*base + word);
  if (!owns_vgpr_range(physical, 1))
    return std::nullopt;
  return RegisterRef{RegClass::VGPR, static_cast<uint16_t>(physical - wf.vgpr_alloc().base), 1};
}

bool MemoryWaitScoreboard::result_is_written(const Instruction &inst, Wavefront &wf) {
  if (inst.mnemonic() == "lds_direct_load" || inst.mnemonic() == "ds_direct_load")
    return valid_lds_direct_operand(wf.m0());
  if (inst.mnemonic() == "s_barrier_signal_isfirst") {
    if (wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA4)
      return wf.cu().in_multiwave_workgroup();
    const auto &source = *inst.src_operand(0);
    const auto constant = source.const_value();
    const auto value =
        constant ? *constant
                 : RegisterAccess(wf).scalar_control_value(source.encoding_value()).value_or(0);
    const bool m0 =
        static_cast<uint32_t>(source.encoding_value()) == scalar_m0_selector(wf.cu().arch());
    return (wf.barrier_state(barrier_operand_id(value, m0)) & 1u) != 0;
  }
  return true;
}

uint64_t MemoryWaitScoreboard::result_lanes(const Instruction &inst, Wavefront &wf) {
  return memory_lanes(inst, wf).result;
}

static bool gfx11_rejects_buffer_load(const Instruction &inst, const Wavefront &wave) {
  const auto arch = wave.cu().arch();
  if (arch != ROCJITSU_CODE_ARCH_RDNA3 && arch != ROCJITSU_CODE_ARCH_RDNA3_5)
    return false;
  const auto name = inst.mnemonic();
  return (name.starts_with("buffer_load_") || name.starts_with("tbuffer_load_")) &&
         !RegisterAccess(wave).vgpr_results_fit_allocation(inst);
}

MemoryWaitScoreboard::MemoryLanes MemoryWaitScoreboard::memory_lanes(const Instruction &inst,
                                                                     Wavefront &wf) {
  if (gfx11_rejects_buffer_load(inst, wf))
    return {wf.exec(), 0};
  if (wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA2 &&
      !RegisterAccess(wf).vgpr_results_fit_allocation(inst))
    return {};
  if (inst.is_memory_op() && wf.memory_destination_out_of_range(inst))
    return {};
  const auto *issue = inst.amdgpu_memory_issue_info();
  if (const auto ordered = decoded_ordered_count_lanes(inst, wf))
    return {ordered->result, ordered->result};
  uint64_t lanes = issue && !issue->exec_masked && !(inst.flags() & FIRST_ACTIVE_MEMORY_LANE)
                       ? (wf.wf_size() == 64 ? ~uint64_t{0} : uint64_t{0xffffffff})
                       : wf.memory_exec(inst);
  if (!(inst.flags() & CONDITIONAL_MEMORY_LANES))
    return {lanes, lanes};
  RegisterModifiers modifiers;
  inst.amdgpu_register_modifiers(modifiers);
  // A known TYPE mismatch has no memory result. Whether it still occupies a
  // counter position is unspecified, so do not use it to prove an older result
  // complete through a later partial wait or counter backpressure.
  if (addr_calc::modern_ignores_buffer_resource(modifiers, wf))
    return {};
  if (modifiers.exec_all_if_nonzero)
    lanes = wave32_exec_all_if_nonzero(lanes);
  if (modifiers.exec_whole_quads)
    lanes = result_is_written(inst, wf) ? whole_active_quads(lanes) : 0;
  if (lanes && modifiers.buffer_resource && !modifiers.scalar_buffer_words) {
    const auto selector = modifiers.buffer_resource->encoding_value();
    if (addr_calc::buffer_resource_range_is_backed(wf, selector)) {
      const auto word3 = RegisterAccess(wf).scalar_control_value(selector + 3);
      // Other targets can suppress the result while retaining an issue position.
      if (word3 && (*word3 >> 30))
        return {lanes, 0};
    }
  }
  return {lanes, lanes};
}

MemoryWaitScoreboard::FlatLanes MemoryWaitScoreboard::flat_lanes(const Instruction &inst,
                                                                 Wavefront &wf,
                                                                 uint64_t shared_base,
                                                                 uint64_t shared_limit) {
  if (wf.memory_destination_out_of_range(inst))
    return {};
  FlatLanes result{wf.memory_exec(inst), 0, 0};
  RegisterModifiers modifiers;
  inst.amdgpu_register_modifiers(modifiers);
  assert(modifiers.flat_address);
  const RegisterAccess registers(wf);
  uint64_t scalar = 0;
  if (const auto *operand = modifiers.flat_scalar_address) {
    const unsigned selector = operand->encoding_value();
    if (!resolve_scalar_register_range(wf, selector, 2))
      return {};
    const auto lo = registers.scalar_control_value(selector);
    const auto hi = registers.scalar_control_value(selector + 1);
    if (!lo || !hi)
      return {};
    scalar = *lo | (uint64_t{*hi} << 32);
  }
  if (!result.requests || (!shared_base && wf.cu().arch() != ROCJITSU_CODE_ARCH_RDNA1 &&
                           wf.cu().arch() != ROCJITSU_CODE_ARCH_RDNA2))
    return result;
  const auto arch = wf.cu().arch();
  const auto layout = flat_private_layout(arch);
  const int32_t segment_offset =
      flat_selects_segment_before_offset(arch) ? modifiers.flat_offset : 0;
  const bool vector_pointer32 = wf.flat_pointer32() && !modifiers.flat_scalar_address;
  const auto address = vector_pointer32
                           ? registers.block_data_register(*modifiers.flat_address, 0, false)
                           : registers.source_register(*modifiers.flat_address);
  uint64_t remaining = result.requests;
  while (remaining) {
    const unsigned lane = std::countr_zero(remaining);
    remaining &= remaining - 1;
    const uint64_t vector =
        address ? registers.vector_control_value(*address, lane).value_or(0) : 0;
    uint64_t value = scalar +
                     flat_vector_offset(
                         vector, modifiers.flat_scalar_address && arch == ROCJITSU_CODE_ARCH_CDNA5,
                         modifiers.flat_scale) +
                     modifiers.flat_offset;
    if (vector_pointer32)
      value = static_cast<uint32_t>(value);
    const uint64_t aperture_pointer = value - segment_offset;
    const auto translated =
        translate_flat_address(wf, aperture_pointer, lane, layout, segment_offset);
    // GFX10.3 retains any complete element of a partial private tuple.
    // Its decoded width is the first element; GFX10.1 requires the complete transfer.
    if ((arch == ROCJITSU_CODE_ARCH_CDNA1 || arch == ROCJITSU_CODE_ARCH_CDNA2 ||
         arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2) &&
        translated.private_address && translated.swizzled &&
        uint64_t{translated.private_offset} + modifiers.flat_access_bytes >
            wf.scratch_lane_size()) {
      result.requests &= ~(uint64_t{1} << lane);
      result.private_oob_results |= uint64_t{1} << lane;
      continue;
    }
    if (!translated.swizzled &&
        flat_in_shared_aperture(aperture_pointer, shared_base, shared_limit))
      result.shared |= uint64_t{1} << lane;
  }
  return result;
}

static bool needs_buffer_payload_lanes(const Instruction &inst, rj_code_arch_t arch) {
  const auto name = inst.mnemonic();
  if (arch == ROCJITSU_CODE_ARCH_CDNA5)
    return name.starts_with("buffer_store_") || name.starts_with("buffer_atomic_");
  return (arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
          arch == ROCJITSU_CODE_ARCH_RDNA4) &&
         (name.starts_with("buffer_store_") || name.starts_with("tbuffer_store_")) &&
         name.find("_format_") != std::string_view::npos;
}

bool MemoryWaitScoreboard::needs_prepared_payload_lanes(const Instruction &inst,
                                                        rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA4 && inst.mnemonic().starts_with("image_atomic_")) {
    RegisterModifiers modifiers;
    inst.amdgpu_register_modifiers(modifiers);
    return modifiers.image_atomic_data != nullptr;
  }
  return needs_buffer_payload_lanes(inst, arch);
}

void MemoryWaitScoreboard::check_image_atomic_payload(const Instruction &inst,
                                                      const Operand &payload, const Wavefront &wf,
                                                      uint64_t lanes) {
  // Bound GFX11 image atomics read VDATA for all EXEC lanes. GFX12 fetches the
  // payload only for in-bounds lanes selected during address preparation.
  if (wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA4) {
    if (!inst.data())
      return;
    lanes = inst.data_as<VectorMemState>()->lane_mask;
  }
  for (unsigned word = 0; word < static_cast<unsigned>(payload.size_bits()) / 32; ++word) {
    const auto reg = gfx115_source_vgpr(payload.encoding_value() + word, wf.num_vgprs());
    access({RegClass::VGPR, reg, 1}, lanes, 0xf, false);
  }
}

void MemoryWaitScoreboard::check_image_store_payload(const Instruction &inst, const Wavefront &wf) {
  const auto &data = *inst.data_as<VectorMemState>();
  RegisterModifiers modifiers;
  inst.amdgpu_register_modifiers(modifiers);
  assert(modifiers.image_store_data);
  const uint32_t first = modifiers.image_store_data->encoding_value();
  const uint32_t words =
      data.buffer_d16 ? (data.buffer_components + 1) / 2 : data.buffer_components;
  for (uint32_t word = 0; word < words; ++word) {
    const auto reg = gfx115_source_vgpr(first + word, wf.num_vgprs());
    const bool half = data.buffer_d16 && 2 * word + 1 == data.buffer_components;
    access({RegClass::VGPR, reg, 1}, data.lane_mask, half ? 0x3 : 0xf, false);
  }
}

void MemoryWaitScoreboard::track_gfx10_scalar_sources(
    const Instruction &inst, Wavefront &wf, const waitcheck_detail::ClassifiedEvent &source) {
  constexpr auto counter = WaitCounterKind::VmVsrc;
  const uint64_t sequence = issue(counter, /*unordered=*/true);
  const bool native_ds = wf.exec() && (inst.flags() & LDS) && inst.mnemonic().starts_with("ds_") &&
                         inst.raw_encoding() && !(inst.raw_encoding()[0] & (1u << 17));
  const RegisterAccess registers(wf);
  auto add_scalar = [&](RegisterRef reg) {
    if (reg.cls == RegClass::VGPR || reg.cls == RegClass::ACC_VGPR)
      return;
    // Each source dword can be overwritten independently.
    for (unsigned r = 0; r < reg.width; ++r)
      add({.sequence = sequence,
           .pc = wf.pc,
           .lanes = ~uint64_t{0},
           .reg = {reg.cls, static_cast<uint16_t>(reg.index + r), 1},
           .counter = counter,
           .bytes = kFullDwordByteMask,
           .gfx10_native_ds_source = native_ds});
  };
  RegisterModifiers modifiers;
  inst.amdgpu_register_modifiers(modifiers);
  for (int i = 0; i < inst.num_src_operands(); ++i) {
    const auto *op = inst.src_operand(i);
    if (!op || !inst.tracks_explicit_source(i) || op->is_vgpr())
      continue;
    if (op == modifiers.buffer_resource) {
      for (const auto reg : registers.buffer_resource_registers(*op, modifiers.scalar_buffer_words))
        if (reg)
          add_scalar(*reg);
    } else if (const auto reg = registers.source_register(*op)) {
      add_scalar(*reg);
    }
  }
  RegisterSet implicit;
  inst.implicit_uses(implicit);
  implicit.for_each(add_scalar);
  if (source.special_reg)
    add_scalar(*source.special_reg);
  add_scalar({RegClass::EXEC, 0, static_cast<uint8_t>(wf.wf_size() / 32)});
}

void MemoryWaitScoreboard::confirm_gfx10_native_ds_source_issue(uint64_t pc) {
  const uint64_t source_sequence = issued_[static_cast<size_t>(WaitCounterKind::VmVsrc)];
  std::vector<Event> issued_sources;
  for (auto &event : events_)
    if (event.counter == WaitCounterKind::VmVsrc && event.sequence == source_sequence &&
        event.pc == pc && event.gfx10_native_ds_source) {
      event.gfx10_native_ds_issued = true;
      issued_sources.push_back(event);
    }
  // A confirmed native DS request releases all of its scalar inputs on a
  // full DS drain. A newer confirmed request for the same input therefore
  // outlives an older one under both DS and VM_VSRC waits. Keep the older
  // record only for cancellation of this latest issue.
  std::erase_if(events_, [&](const Event &old) {
    if (old.counter != WaitCounterKind::VmVsrc || !old.gfx10_native_ds_issued ||
        old.sequence >= source_sequence)
      return false;
    const bool covered = std::ranges::any_of(issued_sources, [&](const Event &newer) {
      return newer.reg.cls == old.reg.cls && newer.reg.index == old.reg.index &&
             newer.reg.width == old.reg.width && (newer.lanes & old.lanes) == old.lanes &&
             (newer.bytes & old.bytes) == old.bytes;
    });
    if (covered)
      superseded_.push_back({old, WaitCounterKind::VmVsrc, source_sequence, pc});
    return covered;
  });
}

static bool disabled_scalar_prefetch(const Instruction &inst, const Wavefront &wf) {
  return wf.cu().arch() == ROCJITSU_CODE_ARCH_CDNA5 && !(wf.mode_raw() & (1u << 24)) &&
         waitcheck_detail::RuntimeWaitTarget::is_scalar_prefetch(inst.mnemonic());
}

void MemoryWaitScoreboard::check_instruction(const Instruction &inst, Wavefront &wf) {
  if (empty() || inst.is_empty_global_transpose(wf.exec()))
    return;
  if (disabled_scalar_prefetch(inst, wf))
    return;
  const bool cdna2_valu = wf.cu().arch() == ROCJITSU_CODE_ARCH_CDNA2 && (inst.flags() & VALU);
  if ((cdna2_valu && !cdna2::valu_destinations_in_range(inst, wf)) ||
      (inst.is_memory_op() && wf.memory_destination_out_of_range(inst)))
    return;
  // A clear register byte proves independence for every lane and byte mask.
  // Only a possible overlap needs the precise instruction-specific footprints.
  if (!pending_scalar_ && (inst.flags() & DIRECT_REGISTER_ACCESSES) && !wf.gpr_idx_en()) {
    auto pending = [&](const Operand &op) {
      const auto reg = op.decoded_vgpr();
      if (!reg)
        return op.is_fieldless() && op.is_vgpr();
      unsigned index = reg->index;
      if (cdna2_valu)
        index = apply_gpr_idx(wf, index, op.vgpr_msb_role());
      else if (index < ACC_VGPR_OFFSET)
        index += wf.vgpr_msb_for_role(op.vgpr_msb_role()) << 8;
      if (index + reg->width > wf.vgpr_alloc().count || index + reg->width > REGISTER_SET_MAX_VGPRS)
        return true;
      return pending_.pending_vgpr(static_cast<uint16_t>(index), reg->width);
    };
    bool hit = false;
    for (int i = 0; i < inst.num_src_operands(); ++i) {
      if (!inst.tracks_explicit_source(i)) {
        // The instruction's implicit uses replace this decoded placeholder, so
        // the explicit-only prefilter cannot prove that the instruction misses.
        hit = true;
        continue;
      }
      hit |= pending(*inst.src_operand(i));
    }
    for (int i = 0; i < inst.num_dst_operands(); ++i)
      hit |= pending(*inst.dst_operand(i));
    if (!hit)
      return;
  }
  if (!pending_scalar_ && inst.mnemonic().starts_with("s_"))
    return;
  checking_valu_ = (inst.flags() & VALU) != 0;
  checking_trans_ = wf.cu().arch() == ROCJITSU_CODE_ARCH_CDNA5 && inst.is_transcendental();
  check_instruction_pending(inst, wf);
  checking_valu_ = false;
  checking_trans_ = false;
}

MemoryWaitScoreboard::SourceRoute MemoryWaitScoreboard::source_route(WaitCounterKind counter,
                                                                     uint64_t sequence,
                                                                     uint64_t lanes) const {
  const uint16_t order = last_order_[static_cast<size_t>(counter)];
  return {counter,
          sequence,
          lanes,
          order,
          order == kUnordered ? 0 : orders_[order].issued,
          order == kUnordered || !last_capacity_eligible_[static_cast<size_t>(counter)]
              ? 0
              : orders_[order].capacity_issued};
}

void MemoryWaitScoreboard::track_expert_vm_sources(const Instruction &inst, Wavefront &wf,
                                                   const waitcheck_detail::ClassifiedEvent &event,
                                                   std::span<const SourceRoute> routes) {
  const uint64_t sequence = issue(event, wf.cu().arch());
  uint64_t actual = 0;
  for (const auto &route : routes)
    actual |= route.lanes;
  if (!actual)
    return;
  auto &tracked = expert_source_completions_[sequence];
  std::ranges::copy(routes, tracked.begin());
  // Native LDS has a single completion, including data from physically
  // routed lanes outside EXEC for DS_BPERMUTE_FI.
  if (event.kind == waitcheck_detail::WaitEventKind::Ds) {
    assert(routes.size() == 1);
    actual = tracked.front().lanes = ~uint64_t{0};
  }
  visit_register_accesses(
      inst, wf,
      [&](RegisterRef ref, uint64_t lanes, uint8_t bytes, bool write, uint16_t) {
        if (write || ref.cls != RegClass::VGPR)
          return;
        add({sequence, wf.pc, lanes & actual, ref, WaitCounterKind::VmVsrc, bytes});
      },
      false);
}

void MemoryWaitScoreboard::leave_expert_scheduling() {
  wait(WaitCounterKind::VaVdst, 0);
  wait(WaitCounterKind::VmVsrc, 0);
}

void MemoryWaitScoreboard::track_expert_valu(const Instruction &inst, Wavefront &wf,
                                             waitcheck_detail::WaitEventKind kind,
                                             rj_code_target_id_t target) {
  using namespace waitcheck_detail;
  if (!wf.exec() && !RuntimeWaitTarget::expert_valu_issues_without_active_lanes(
                        inst, wf.cu().arch(), wf.wf_size()))
    return;
  // One independent VALU slot releases a short TRANS source and permits an
  // ordinary VALU to consume its output. The output still needs an explicit
  // VA wait before a non-VALU consumer. A later TRANS is serialized with the
  // older one. Check other VALUs for dependencies before giving the credit.
  const auto name = inst.mnemonic();
  const bool serialized_trans = kind == kValuTrans;
  if (wf.cu().arch() == ROCJITSU_CODE_ARCH_CDNA5 && !wf.discards_valu_result(inst) &&
      !(name.starts_with("v_dual_") && wf.wf_size() != 32) &&
      std::ranges::any_of(events_, [](const Event &old) { return old.short_trans; })) {
    struct Access {
      RegisterRef reg;
      uint64_t lanes;
      uint8_t bytes;
      bool write;
    };
    std::vector<Access> accesses;
    if (!serialized_trans)
      visit_register_accesses(inst, wf,
                              [&](RegisterRef reg, uint64_t lanes, uint8_t bytes, bool write,
                                  uint16_t) {
                                if (reg.cls == RegClass::VGPR)
                                  accesses.push_back({reg, lanes, bytes, write});
                              });
    const auto overlaps = [](RegisterRef lhs, RegisterRef rhs) {
      return lhs.index < rhs.index + rhs.width && rhs.index < lhs.index + lhs.width;
    };
    // After a separate issued VALU slot, a direct full-byte ordinary VALU
    // write replaces the old TRANS result for reads on the written lanes.
    // Keep the old event for memory writes and its architectural VA position.
    if (kind == kValuCore)
      for (auto &old : events_) {
        if (!old.short_trans || old.valu_source || !old.trans_valu_ready ||
            old.reg.cls != RegClass::VGPR)
          continue;
        const bool reads_old = std::ranges::any_of(accesses, [&](const Access &access) {
          return !access.write && overlaps(old.reg, access.reg);
        });
        if (reads_old)
          continue;
        for (const auto &access : accesses)
          if (access.write && access.bytes == kFullDwordByteMask &&
              access.reg.index <= old.reg.index &&
              access.reg.index + access.reg.width >= old.reg.index + old.reg.width)
            old.trans_use_superseded_lanes |= access.lanes & old.lanes;
      }
    std::vector<uint64_t> dependent;
    for (const Event &old : events_)
      if (old.short_trans && std::ranges::any_of(accesses, [&](const Access &access) {
            return (!old.valu_source || access.write) && overlaps(old.reg, access.reg);
          }))
        dependent.push_back(old.sequence);
    for (auto &old : events_)
      if (old.short_trans && !old.valu_source &&
          std::ranges::find(dependent, old.sequence) == dependent.end())
        old.trans_valu_ready = true;
    if (std::erase_if(events_, [&](const Event &old) {
          if (!old.short_trans || !old.valu_source ||
              std::ranges::find(dependent, old.sequence) != dependent.end())
            return false;
          clear_destination(old);
          return true;
        }))
      rebuild_mask();
  }
  const auto counter = WaitCounterKind::VaVdst;
  const ClassifiedEvent event{counter, kind};
  // A bound on this shared counter also bounds each individual ordered pipeline.
  const auto units = issue_units(inst, event, wf.cu().arch());
  const auto sequence = issue(event, wf.cu().arch(), units);
  // SCHED_MODE[2] lets a CDNA5 wave issue instructions while its WMMA is
  // running. Later core VALU issues therefore cannot prove that its operands
  // have been consumed; retain them until the explicit VA_VDST wait.
  const uint8_t window =
      kind == kValuMatrix && !(wf.wave_sched_mode_raw() & 4u)
          ? RuntimeWaitTarget::wmma_valu_wait_states(inst, wf.cu().arch(), target)
          : 0;
  const auto *core = window ? expert_core_valu_order() : nullptr;
  const uint64_t preceding_core = core ? core->issued : 0;
  visit_register_accesses(
      inst, wf, [&](RegisterRef reg, uint64_t lanes, uint8_t bytes, bool write, uint16_t) {
        if (reg.cls != RegClass::VGPR)
          return;
        Event pending{sequence, wf.pc, lanes, reg, counter, bytes};
        pending.ordered_overwrite = false;
        pending.valu_source = !write;
        pending.short_trans = wf.cu().arch() == ROCJITSU_CODE_ARCH_CDNA5 && kind == kValuTrans &&
                              name.find("_f64") == std::string_view::npos;
        pending.wmma_wait_states = window;
        pending.preceding_core_valu = preceding_core;
        add(pending);
      });
}

namespace {
struct ImageSourceBytes {
  uint8_t active = 0;
  uint8_t quad = 0;
};

/// Return the byte masks for the active and helper lanes of each sampler input.
template <typename Read>
std::array<ImageSourceBytes, 12> image_sample_source_bytes(const RegisterModifiers &modifiers,
                                                           const Wavefront &wf, Read read) {
  std::array<ImageSourceBytes, 12> result{};
  const auto &sample = *modifiers.image_sample_sources;
  const auto backed = [&](uint32_t reg, uint32_t words) {
    return scalar_selector_range_is_backed(wf, reg, words);
  };
  if (!backed(sample.sampler, 4))
    return result;
  const uint32_t resource = modifiers.image_resource->encoding_value();
  const uint32_t word3 = read(resource + 3);
  const auto arch = wf.cu().arch();
  const bool gfx11 = arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5;
  if (has_gfx11_plus_image_operands(arch) &&
      has_unimplemented_image_sampler_controls(
          read_image_sampler_controls(arch, sample.sampler, read)))
    return result;
  const bool volume = (word3 >> 28) == kImage3d;
  if (gfx11 && volume && sample.gather)
    return result;
  const bool native_texture =
      !sample.query && uses_native_2d_sampler(arch, resource, sample.sampler, sample.dim,
                                              sample.unorm, read, backed);
  const bool native_filter = sample.linear_candidate && native_texture;
  const uint32_t filter = read(sample.sampler + 2);
  const uint32_t word1 = read(resource + 1);
  const bool compute_feedback =
      sample.filtered_feedback_candidate &&
      (sample.texel_fault || sample.lod_warning ||
       rdna_image_lod_warning_threshold(arch, read(resource + 5))) &&
      use_linear_image_path(arch, resource, sample.sampler, sample.dim, true, read, backed, true) &&
      compute_filtered_image_feedback(arch, sample.dim, word3, word1, filter, sample.mode,
                                      sample.offset, sample.compare);
  const bool linear =
      !native_filter &&
      (compute_feedback || (volume && !sample.query) ||
       ((!gfx11 && arch != ROCJITSU_CODE_ARCH_RDNA4) ||
        (sample.linear_candidate &&
         use_linear_image_path(arch, resource, sample.sampler, sample.dim, true, read, backed))));
  const bool gradients =
      sample.mode == ImageSampleMode::Derivatives || sample.mode == ImageSampleMode::Derivatives16;
  const auto mip = rdna_image_mip_range(arch, word1, word3);
  const bool needs_gradients =
      linear
          ? gradients || linear_image_needs_implicit_lod(filter, mip.first, mip.last, sample.gather)
          : graphics_image_needs_gradients(sample.query,
                                           graphics_sample_has_fixed_lod(sample.query,
                                                                         sample.lod_warning, filter,
                                                                         native_texture)
                                               ? 0u
                                           : arch == ROCJITSU_CODE_ARCH_RDNA4 ? (word1 >> 12) & 31u
                                                                              : (word1 >> 16) & 15u,
                                           sample.dim, sample.gather, filter);
  const bool quad = needs_gradients && (sample.mode == ImageSampleMode::Implicit ||
                                        sample.mode == ImageSampleMode::Bias);
  const bool a16 = modifiers.image_a16;
  const bool bias = sample.mode == ImageSampleMode::Bias;
  const unsigned gradient_start =
      unsigned(sample.offset) + unsigned(bias) + unsigned(sample.compare);
  const uint32_t gradient_components = sample.mode == ImageSampleMode::Derivatives16
                                           ? image_derivative_component_count(sample.dim)
                                           : 0;
  const uint16_t low16_words =
      image_prefix_low16_words(gradient_start, gradient_components, a16, bias, sample.offset);
  const unsigned prefix = modifiers.image_prefix_words;
  for (unsigned word = 0; word < prefix; ++word) {
    uint8_t bytes = image_prefix_word_bytes(word, low16_words);
    if (!linear && sample.offset && word == 0 && sample.dim == kImageDimCube)
      bytes = 0;
    if (bias && word == unsigned(sample.offset))
      bytes = linear || needs_gradients ? (a16 ? 0x3 : 0xf) : 0;
    if (gradients && word >= gradient_start) {
      if (!needs_gradients)
        bytes = 0;
      else if (sample.mode == ImageSampleMode::Derivatives16 &&
               image_derivative_component_count(sample.dim) == 1)
        bytes = 0x3; // dx and dy each start in the low half of a separate word.
    }
    result[word].active = bytes;
  }
  // Cube unfolding and GFX10's same-face check read the derivative lanes' face selectors.
  const bool gfx10 = arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2;
  const unsigned spatial = sample.dim == kImageDimCube && (!linear || gfx10)
                               ? 3u
                               : image_derivative_component_count(sample.dim);
  const bool array = sample.dim == kImageDim1dArray || sample.dim == kImageDim2dArray;
  const unsigned count = sample.query && array ? spatial : modifiers.image_coordinate_count;
  for (unsigned component = 0; component < count; ++component) {
    auto &word = result[prefix + (a16 ? component / 2 : component)];
    const uint8_t bytes = a16 ? (component % 2 ? 0xc : 0x3) : 0xf;
    word.active |= bytes;
    if (quad && component < spatial)
      word.quad |= bytes;
  }
  return result;
}
} // namespace

void MemoryWaitScoreboard::check_instruction_pending(const Instruction &inst, Wavefront &wf) {
  // These targets can have a physical data dependency when a 16-bit VALU
  // writes one half of a VGPR while a memory result to the other half is
  // pending. Keep the decoded write footprint for source-read protection and
  // expert events; check the other half only against pending results here.
  const auto arch = wf.cu().arch();
  std::vector<std::pair<RegisterRef, uint8_t>> partial_valu_destinations;
  if ((arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
       arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_CDNA5) &&
      (inst.flags() & VALU)) {
    const RegisterAccess registers(wf);
    for (int i = 0; i < inst.num_dst_operands(); ++i) {
      const Operand *operand = inst.dst_operand(i);
      if (!operand || operand->size_bits() != 16)
        continue;
      const uint8_t bytes = operand->register_byte_mask();
      if (bytes != 0x3 && bytes != 0xc)
        continue;
      if (auto reg = registers.destination_register(*operand);
          reg && reg->cls == RegClass::VGPR && reg->width == 1)
        partial_valu_destinations.emplace_back(*reg, bytes);
    }
  }
  visit_register_accesses(
      inst, wf,
      [&](RegisterRef reg, uint64_t lanes, uint8_t bytes, bool write,
          uint16_t ordered_write_order) {
        if (write && std::ranges::any_of(partial_valu_destinations, [&](const auto &destination) {
              return destination.first == reg && destination.second == bytes;
            }))
          access(reg, lanes, static_cast<uint8_t>(0xf & ~bytes), false);
        access(reg, lanes, bytes, write, ordered_write_order);
      });
}

template <typename Visit>
void MemoryWaitScoreboard::visit_register_accesses(const Instruction &inst, Wavefront &wf,
                                                   Visit &&visit, bool check_result) {
  const auto access = [&](RegisterRef reg, uint64_t lanes, uint8_t bytes, bool write,
                          uint16_t order = kUnordered) {
    if (write && reg.cls == RegClass::EXEC && checks_gds_source_lifetime(wf.cu().arch()))
      return;
    visit(reg, lanes, bytes, write, order);
  };
  // These VGPR-returning loads reject the entire tuple, including TFE, before address preparation.
  if (gfx11_rejects_buffer_load(inst, wf))
    return;
  // GFX10 checks synchronous EXEC destinations after successful execution. Static
  // liveness includes untaken conditional moves and control-flow paths.
  RegisterModifiers modifiers;
  inst.amdgpu_register_modifiers(modifiers);
  // Routed image instructions validate their descriptors even with no result lanes.
  // Other RDNA2 memory returns, DS permutations and GET_RESINFO skip their sources
  // when the decoded destination tuple crosses the allocation.
  if (wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA2 && inst.is_memory_wait_producer() &&
      (!inst.is_memory_op() || !modifiers.image_address) &&
      !RegisterAccess(wf).vgpr_results_fit_allocation(inst))
    return;
  if (wf.discards_valu_result(inst))
    return;
  const auto name = inst.mnemonic();
  // VOPD only executes its two operations in a wave32.
  if (wf.wf_size() != 32 && name.starts_with("v_dual_"))
    return;
  const bool gfx10 =
      wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA1 || wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA2;
  const bool gfx10_valu = gfx10 && (inst.flags() & VALU);
  const bool vector = name.starts_with("v_") || name.starts_with("ds_") ||
                      name.starts_with("lds_") || (inst.is_memory_op() && !name.starts_with("s_"));
  const bool scalar_lane_read = name.starts_with("v_readlane_b32");
  const bool scalar_lane_write = name.starts_with("v_writelane_b32");
  const bool first_lane = name.starts_with("v_readfirstlane_b32");
  const bool selected_lane = scalar_lane_read || scalar_lane_write || first_lane;
  const bool matrix = inst.flags() & MATRIX_REGISTER_ACCESSES;
  const auto *memory_issue = inst.amdgpu_memory_issue_info();
  const bool ignores_exec = (inst.flags() & IGNORES_EXEC) || matrix || scalar_lane_read ||
                            scalar_lane_write || (memory_issue && !memory_issue->exec_masked);
  const uint64_t all_lanes = wf.wf_size() == 64 ? ~uint64_t{0} : uint64_t{0xffffffff};
  const auto ordered = decoded_ordered_count_lanes(inst, wf);
  uint64_t lanes = ordered ? ordered->contribution
                   : ignores_exec && !(inst.flags() & FIRST_ACTIVE_MEMORY_LANE)
                       ? all_lanes
                       : wf.memory_exec(inst);
  if ((wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA1 || wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA2) &&
      name.starts_with("ds_gws_"))
    lanes = RegisterAccess(wf).gds_source_lanes(inst);
  else if (!ordered && wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA2 && inst.is_memory_op())
    lanes = result_lanes(inst, wf);
  if (const auto &output = modifiers.gfx10_export) {
    if ((wf.status_raw() & (1u << 18)) || !output->valid() || !wf.cu().shader_export_ready())
      return;
    access({RegClass::EXEC, 0, static_cast<uint8_t>(wf.wf_size() / 32)}, ~uint64_t{0}, 0xf, false);
    output->for_each_source(wf.num_vgprs(), [&](unsigned, uint16_t reg) {
      access({RegClass::VGPR, reg, 1}, wf.exec(), 0xf, false);
    });
    return;
  }
  if (vector && (!ignores_exec || (ordered && ordered->result)))
    access({RegClass::EXEC, 0, static_cast<uint8_t>(wf.wf_size() / 32)}, ~uint64_t{0}, 0xf, false);
  const RegisterAccess registers(wf);
  if (const auto &output = modifiers.shader_export) {
    if (wf.shader_export_issue(*output) != ShaderExportIssue::Accept)
      return;
    access({RegClass::EXEC, 0, static_cast<uint8_t>(wf.wf_size() / 32)}, ~uint64_t{0}, 0xf, false);
    if (output->row_from_m0)
      access({RegClass::M0, 0, 1}, ~uint64_t{0}, 0xf, false);
    const uint64_t exported = output->source_lanes(wf.exec());
    output->for_each_source(wf.num_vgprs(), [&](unsigned, uint16_t reg) {
      access({RegClass::VGPR, reg, 1}, exported, 0xf, false);
    });
    return;
  }
  const bool gfx11 =
      wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA3 || wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA3_5;
  const bool gfx12_image_atomic =
      wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA4 && modifiers.image_atomic_data;
  const bool gfx11_or_gfx12_sample = (gfx11 || wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA4) &&
                                     modifiers.image_sample_sources &&
                                     !modifiers.image_sample_sources->query;
  const bool check_image_result = modifiers.gfx10_lod_query_quad ||
                                  (gfx11 && modifiers.image_address) || gfx12_image_atomic ||
                                  gfx11_or_gfx12_sample || modifiers.image_resource_info;
  if (check_image_result) {
    for (int i = 0; i < inst.num_dst_operands(); ++i) {
      const auto &dst = *inst.dst_operand(i);
      if (dst.decoded_vgpr() && !registers.destination_register(
                                    dst, true, bool(inst.flags() & RESERVES_UNWRITTEN_VGPR))) {
        if (!gfx12_image_atomic && !modifiers.bvh_resource)
          return;
        // GFX11 BVH and GFX12 atomics read the scalar descriptor before suppressing vector work.
        lanes = 0;
        break;
      }
    }
  }
  // Dropped and unbound texture operations do not acquire coordinate or payload VGPRs.
  bool read_image_inputs = modifiers.image_reads_address;
  if (const auto *resource = modifiers.image_resource; resource && read_image_inputs) {
    const auto shape = registers.scalar_control_value(resource->encoding_value() + 3);
    const auto format = registers.scalar_control_value(resource->encoding_value() + 1);
    read_image_inputs =
        shape && format && !image_descriptor_is_unbound(wf.cu().arch(), *format, *shape);
    const auto arch = wf.cu().arch();
    const auto &sample = modifiers.image_sample_sources;
    if (read_image_inputs && has_gfx11_plus_image_operands(arch) &&
        !modifiers.image_resource_info && !(sample && sample->query)) {
      const uint32_t selector = resource->encoding_value();
      const uint32_t words = resource->size_bits() / 32;
      std::array<uint32_t, 8> descriptor{};
      if (!scalar_selector_range_is_backed(wf, selector, words))
        read_image_inputs = false;
      else {
        for (uint32_t i = 0; i < words; ++i)
          descriptor[i] = registers.scalar_control_value(selector + i).value_or(0);
        const auto image = decode_image_resource(descriptor, arch, words == 4);
        const bool resident_mip =
            sample && image.min_lod && !sample->compare && !sample->offset && !sample->clamp &&
            scalar_selector_range_is_backed(wf, sample->sampler, 4) &&
            image.supports_resident_mip(
                sample->dim, sample->mode, sample->gather,
                registers.scalar_control_value(sample->sampler + 2).value_or(0));
        read_image_inputs =
            !image.uav3d &&
            !(sample && image.has_unqualified_sample_controls(arch, resident_mip)) &&
            !(resident_mip && image.min_lod > image.last_level * 256);
        if (words == 8 && image.uses_linear_transfer_pitch() &&
            !native_block_filter(image.data_format)) {
          const auto bytes =
              buffer_format_bytes(image_srgb_storage_format(image.data_format, false));
          if (bytes.succeeded())
            read_image_inputs &= image.supports_linear_pitch(
                arch, sample && sample->dim == kImageDim2d, bytes.value());
        }
      }
    }
  }
  if (modifiers.image_atomic_data && read_image_inputs)
    check_image_atomic_payload(inst, *modifiers.image_atomic_data, wf, lanes);
  if (modifiers.image_sample_sources && read_image_inputs) {
    const auto inputs = image_sample_source_bytes(modifiers, wf, [&](uint32_t selector) {
      return registers.scalar_control_value(selector).value_or(0);
    });
    const uint64_t helpers = ImageDerivativeQuad::source_lanes(lanes) & ~lanes;
    for (unsigned index = 0; index < inputs.size(); ++index) {
      const auto bytes = inputs[index];
      if (!bytes.active)
        continue;
      auto selector = modifiers.image_coordinates[index];
      if (has_gfx11_plus_image_operands(wf.cu().arch()))
        selector = gfx115_source_vgpr(selector, wf.num_vgprs());
      else if (selector >= wf.num_vgprs())
        continue;
      const RegisterRef reg{RegClass::VGPR, static_cast<uint16_t>(selector), 1};
      if (bytes.quad == bytes.active) {
        access(reg, lanes | helpers, bytes.active, false);
      } else {
        access(reg, lanes, bytes.active, false);
        access(reg, helpers, bytes.quad, false);
      }
    }
  } else if (modifiers.image_address && read_image_inputs) {
    auto check_coordinate = [&](uint16_t reg, uint8_t bytes, uint64_t readers) {
      if (has_gfx11_plus_image_operands(wf.cu().arch()))
        reg = gfx115_source_vgpr(reg, wf.num_vgprs());
      if (reg < wf.num_vgprs())
        access({RegClass::VGPR, reg, 1}, readers, bytes, false);
    };
    uint32_t components = modifiers.image_coordinate_count;
    uint32_t prefix = modifiers.image_prefix_words;
    if (const auto dim = modifiers.image_transfer_dim) {
      const uint32_t resource = modifiers.image_resource->encoding_value();
      const auto read = [&](uint32_t index) {
        return registers.scalar_control_value(index).value_or(0);
      };
      const auto backed = [&](uint32_t index, uint32_t count) {
        return scalar_selector_range_is_backed(wf, index, count);
      };
      if (!gfx11 || use_linear_image_path(wf.cu().arch(), resource, 0, *dim, false, read, backed)) {
        const uint32_t type = read(resource + 3) >> 28;
        const auto shape = image_coordinate_shape(type, true);
        const uint32_t slice = image_transfer_slice_coordinate(type, *dim, false, shape.has_y);
        components = image_transfer_coordinate_count(
            type, shape.has_y, shape.has_slice || type == kImage2dMsaaArray, slice);
      }
    }
    const RegisterModifiers::Gfx9ImageTuple &tuple = modifiers.gfx9_image_tuple;
    if (tuple.resource_selector != ~0u) {
      const std::optional<uint32_t> descriptor =
          registers.scalar_control_value(tuple.resource_selector + 3);
      const ImageCoordinateShape shape =
          image_coordinate_shape(descriptor.value_or(0) >> 28, tuple.image_array);
      components = shape.coordinates() + tuple.mip;
      const uint32_t dimensions = shape.spatial_dimensions();
      prefix =
          tuple.derivatives ? 2 * (modifiers.image_a16 ? (dimensions + 1) / 2 : dimensions) : 0;
    }
    for (uint32_t word = 0; word < prefix; ++word) {
      const uint16_t reg = modifiers.image_coordinates[word];
      check_coordinate(reg, image_prefix_word_bytes(word, modifiers.image_prefix_low16_words),
                       lanes);
    }
    const uint64_t coordinate_lanes =
        modifiers.gfx10_lod_query_quad ? ImageDerivativeQuad::source_lanes(lanes) : lanes;
    uint64_t ray_lanes = coordinate_lanes;
    if (const auto resource = modifiers.bvh_resource) {
      const auto word2 = registers.scalar_control_value(*resource + 2);
      const auto word3 = registers.scalar_control_value(*resource + 3);
      const auto last_node = word2 && word3 ? gfx11_bvh_last_node(*word2, *word3) : std::nullopt;
      if (!last_node)
        ray_lanes = 0;
      else {
        const bool pointer_flags = gfx11_bvh_pointer_flags_enabled(*word3);
        for (uint64_t pending = coordinate_lanes; pending; pending &= pending - 1) {
          const uint32_t lane = std::countr_zero(pending);
          const auto pointer_word = [&](unsigned i) {
            const auto index = gfx115_source_vgpr(modifiers.image_coordinates[i], wf.num_vgprs());
            return wf.debug_read_vgpr(index, lane);
          };
          uint64_t pointer = pointer_word(0);
          if (modifiers.bvh_pointer_words == 2)
            pointer |= uint64_t{pointer_word(1)} << 32;
          if (!bvh_node_can_load(pointer, *last_node, pointer_flags))
            ray_lanes &= ~(uint64_t{1} << lane);
        }
      }
    }
    for (uint32_t component = 0; component < components; ++component) {
      const uint32_t word = prefix + (modifiers.image_a16 ? component / 2 : component);
      uint16_t reg = modifiers.image_coordinates[word];
      if (modifiers.gfx10_lod_query_quad && reg >= wf.num_vgprs())
        reg = 0;
      check_coordinate(reg, modifiers.image_a16 ? (component % 2 ? 0xc : 0x3) : 0xf,
                       component < modifiers.bvh_pointer_words ? coordinate_lanes : ray_lanes);
    }
  }
  if (modifiers.exec_all_if_nonzero)
    lanes = wave32_exec_all_if_nonzero(lanes);
  if (modifiers.exec_whole_quads)
    lanes = result_is_written(inst, wf) ? whole_active_quads(lanes) : 0;
  bool read_buffer_address = true;
  bool read_buffer_offset = true;
  bool cdna5_type_mismatch = false;
  if (const auto *resource = modifiers.buffer_resource) {
    const auto arch = wf.cu().arch();
    const bool gfx12 = arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_CDNA5;
    unsigned first_resource_word = 0;
    if ((gfx11 || gfx12) && !modifiers.scalar_buffer_words) {
      const auto selector = resource->encoding_value();
      std::optional<uint32_t> control;
      if (addr_calc::buffer_resource_range_is_backed(wf, selector))
        control = registers.scalar_control_value(selector + 3);
      const bool rdna4 = arch == ROCJITSU_CODE_ARCH_RDNA4;
      const bool typed = name.starts_with("tbuffer_");
      const bool formatted = typed || name.find("_format_") != std::string_view::npos;
      if (!control) {
        first_resource_word = 4;
        read_buffer_offset = read_buffer_address = false;
      } else {
        const bool non_buffer = (*control >> 30) != 0;
        cdna5_type_mismatch = arch == ROCJITSU_CODE_ARCH_CDNA5 && non_buffer;
        const unsigned format = addr_calc::rdna_buffer_resource_format(arch, *control);
        if (formatted && (non_buffer || !format || (typed && !modifiers.encoded_buffer_format))) {
          // Format preparation reads only the descriptor's control word.
          first_resource_word = 3;
          read_buffer_offset = read_buffer_address = false;
        } else if (non_buffer || (rdna4 && !format)) {
          // RDNA4 reads the stride, bounds and control before fetching the base.
          first_resource_word = rdna4 ? 1 : 0;
          read_buffer_offset = read_buffer_address = false;
        } else if (gfx11 && !typed && !format) {
          // GFX11 raw address calculation has already consumed SOFFSET here.
          read_buffer_address = false;
        }
      }
      if (read_buffer_address && modifiers.buffer_offset) {
        const uint32_t offset = modifiers.buffer_offset->encoding_value();
        // MUBUF/MTBUF accept inline constants. VBUFFER uses a 7-bit register
        // selector, and CDNA5 stops before VADDR for encodings above M0.
        read_buffer_address = (gfx11 && resolve_src_scalar_statically(offset).has_value()) ||
                              ((arch != ROCJITSU_CODE_ARCH_CDNA5 || offset <= kModernM0Selector) &&
                               resolve_scalar_register_range(wf, offset, 1).has_value());
      }
    }
    const auto words =
        registers.buffer_resource_registers(*resource, modifiers.scalar_buffer_words);
    for (unsigned word = first_resource_word; word < words.size(); ++word)
      if (words[word])
        access(*words[word], ~uint64_t{0}, 0xf, false);
  }
  // FLAT's result ordering depends on its planned memory domains. The issuer
  // checks those results after checking address sources, still before execution.
  if (check_result && inst.is_memory_wait_producer() && !name.starts_with("flat_")) {
    const uint64_t results = ordered                                     ? ordered->result
                             : (inst.flags() & CONDITIONAL_MEMORY_LANES) ? result_lanes(inst, wf)
                                                                         : lanes;
    check_memory_result(inst, wf, results, modifiers);
  }
  // SUBREV permutes VSRC1, unlike other integer DPP instructions.
  const bool reverse_dpp =
      name.starts_with("v_subrev_") &&
      (name.ends_with("u32") || name.ends_with("u32_e32") || name.ends_with("u32_e64")) &&
      (wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA3 || wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA4);
  const Operand *routed_source = reverse_dpp ? inst.src_operand(1) : modifiers.src0;
  // Floating DOT4 filters its non-routed source observations with its destinations.
  const bool filtered_dot4 =
      wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA4 && name.starts_with("v_dot4_f32_");
  uint64_t source_lanes = lanes;
  uint64_t src0_lanes = lanes;
  uint64_t destination_lanes = lanes;
  if (modifiers.dpp || modifiers.dpp8) {
    const auto plan =
        modifiers.dpp
            ? dpp::make_dpp_plan(wf.wf_size(), modifiers.control, modifiers.row_mask,
                                 modifiers.bank_mask, modifiers.bound_ctrl, modifiers.fi, wf.exec(),
                                 modifiers.inactive_uses_bound_ctrl)
            : dpp::make_dpp8_plan(wf.wf_size(), modifiers.control, modifiers.fi, wf.exec());
    destination_lanes &= plan.row_bank_mask & plan.source_write_mask;
    if (filtered_dot4)
      source_lanes = destination_lanes;
    src0_lanes = dpp::dpp_physical_source_mask(plan, lanes, wf.wf_size());
  }
  if (modifiers.ds_permutation != DsPermutation::None) {
    std::optional<RegisterRef> address;
    if (modifiers.src1) {
      address = registers.source_register(*modifiers.src1);
      if (address)
        access(*address, lanes, 0xf, false);
    }
    const auto data = registers.source_register(*modifiers.src0);
    if (!data || (check_result && !pending_.pending(*data) && !(inst.flags() & VALU)))
      return;
    uint64_t source_lanes = 0;
    if (modifiers.ds_permutation == DsPermutation::Permute) {
      source_lanes = lanes;
    } else {
      const auto group_width = ds_permute_group_width(wf.cu().arch(), wf.wf_size());
      uint64_t remaining = lanes;
      while (remaining) {
        const unsigned lane = std::countr_zero(remaining);
        remaining &= remaining - 1;
        const bool swizzle = modifiers.ds_permutation == DsPermutation::Swizzle;
        const unsigned source =
            swizzle ? ds_swizzle_lane(lane, modifiers.ds_offset)
                    : ds_permute_lane(
                          lane,
                          address ? registers.vector_control_value(*address, lane).value_or(0) : 0,
                          modifiers.ds_offset, group_width);
        if (modifiers.ds_permutation == DsPermutation::BpermuteFi ||
            (lanes & (uint64_t{1} << source)))
          source_lanes |= uint64_t{1} << source;
      }
    }
    access(*data, source_lanes, 0xf, false);
    return;
  }
  // SRC2 has one register input beyond EXEC (checked above). Its other
  // operand comes from the data share. Both pending-state paths use the
  // address register selected by execution, including its VGPR0 fallback.
  if (wf.cu().arch() == ROCJITSU_CODE_ARCH_CDNA1 && name.starts_with("ds_") &&
      name.find("_src2_") != std::string_view::npos) {
    const auto reg = cdna1::ds_src2_address_register(inst.src_operand(0)->encoding_value(),
                                                     wf.vgpr_alloc().count);
    access({RegClass::VGPR, reg, 1}, lanes, 0xf, false);
    return;
  }
  const bool ds = inst.is_memory_op() && name.starts_with("ds_");
  const bool ds_vgpr0 = ds && wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA2 &&
                        RegisterAccess::ds_sources_use_vgpr0(inst);
  const auto check_ds_source = [&](const Operand &operand) {
    registers.for_each_ds_source_range(operand, true, [&](RegisterRef reg) {
      access(reg, lanes, operand.register_byte_mask(), false);
    });
  };
  if (!pending_scalar_ && ds) {
    for (int i = 0; i < inst.num_src_operands(); ++i) {
      if (!inst.tracks_explicit_source(i))
        continue;
      if (const auto *op = inst.src_operand(i); op && op->is_vgpr()) {
        if (ds_vgpr0) {
          check_ds_source(*op);
        } else if (auto reg = registers.source_register(
                       *op, op == modifiers.wordwise_source0 || op == modifiers.wordwise_source1)) {
          access(*reg, lanes, op->register_byte_mask(), false);
        }
      }
    }
    return;
  }
  const bool hwreg_read = name == "s_getreg_b32";
  const bool hwreg_write = name == "s_setreg_b32" || name == "s_setreg_imm32_b32";
  const Operand *hwreg = hwreg_read    ? inst.src_operand(0)
                         : hwreg_write ? inst.dst_operand(0)
                                       : nullptr;
  if (hwreg && hwreg_accesses_scc(wf, hwreg->encoding_value(), hwreg_write))
    access({RegClass::SCC, 0, 1}, ~uint64_t{0}, 0xf, hwreg_write);
  if (name.starts_with("v_dual_") && name.find("cndmask") != std::string_view::npos && lanes)
    access({RegClass::VCC, 0, 1}, ~uint64_t{0}, 0xf, false);
  auto scalar_register = [&](uint32_t selector, uint8_t width,
                             bool write) -> std::optional<RegisterRef> {
    if (width == 2 && scalar_pair_crosses_register_region(wf, selector))
      return std::nullopt;
    const auto base = scalar_register_for_selector(wf, selector);
    if (base && base->storage == ScalarRegisterStorage::SGPR &&
        scalar_gpr_out_of_range(wf, selector, width)) {
      if (write)
        return std::nullopt;
      selector = 0;
    }
    const auto range = write ? resolve_scalar_destination_range(wf, selector, width)
                             : resolve_scalar_register_range(wf, selector, width);
    return range ? range->register_ref() : std::nullopt;
  };
  auto resolve = [&](const Operand &op, bool write) -> std::optional<RegisterRef> {
    if (write && (inst.flags() & COND_BRANCH) && &op == inst.dst_operand(0) &&
        name.starts_with("s_subvector_loop_") &&
        (wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA1 ||
         wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA2)) {
      // BEGIN skips its scratch write on empty raw EXEC. END reads scratch on
      // every path and overwrites it only to start the high pass.
      const auto selector = op.encoding_value();
      const bool writes_scratch =
          name == "s_subvector_loop_begin"
              ? wf.exec_raw() != 0
              : (wf.exec_raw() >> 32) == 0 &&
                    registers.scalar_control_value(selector).value_or(0) != 0;
      if (!writes_scratch)
        return std::nullopt;
      const auto range = resolve_scalar_destination_range(wf, selector, 1);
      return range ? range->register_ref() : std::nullopt;
    }
    // MMA execution applies GPR_IDX only to ordinary VGPRs, never its
    // separate accumulator bank. Keep that bank's decoded identity intact.
    if (matrix && wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA3_5)
      if (auto reg = registers.gfx115_wmma_register(op, write))
        return reg;
    if (matrix)
      if (auto reg = op.to_register_ref(); reg && reg->cls == RegClass::ACC_VGPR) {
        reg->cls = RegClass::VGPR;
        reg->index += ACC_VGPR_OFFSET;
        return reg;
      }
    // Explicit vector mask operands use the wave width before range validation:
    // a wave32 mask may occupy VCC_HI without also accessing the next selector.
    const bool wave_mask = vector && !op.is_vgpr() && op.size_bits() == 64 &&
                           (write || (modifiers.src2_is_wave_mask && &op == inst.src_operand(2)));
    if (wave_mask && op.has_register_selector() && !op.const_value())
      return scalar_register(op.encoding_value(), wf.wf_size() / 32, write);
    if (gfx10_valu && !write && op.is_vgpr())
      return registers.gfx10_valu_source_vgpr(op);
    if (!write && wf.flat_pointer32() && &op == modifiers.flat_address &&
        !modifiers.flat_scalar_address)
      return registers.block_data_register(op, 0, false);
    auto reg = write ? registers.destination_register(op)
                     : registers.source_register(op, &op == modifiers.wordwise_source0 ||
                                                         &op == modifiers.wordwise_source1);
    if (!reg)
      if (auto special = op.to_special_reg_class())
        reg =
            RegisterRef{*special, 0, static_cast<uint8_t>(std::max(1, (op.size_bits() + 31) / 32))};
    if (op.has_register_selector() && op.size_bits() && !op.const_value() &&
        (!reg || is_special_reg_class(reg->cls))) {
      if (auto range = resolve_scalar_register_range(wf, op.encoding_value(),
                                                     std::max(1, (op.size_bits() + 31) / 32)))
        reg = range->register_ref();
      else if (!write && op.encoding_value() == 253)
        reg = RegisterRef{RegClass::SCC, 0, 1};
    }
    if (!reg)
      return std::nullopt;
    if (reg->cls == RegClass::SGPR)
      return scalar_register(reg->index, reg->width, write);
    if (reg->cls == RegClass::TTMP)
      return scalar_register(kTtmpSelectorFirst + reg->index, reg->width, write);
    if (wave_mask)
      reg->width = static_cast<uint8_t>(wf.wf_size() / 32);
    if (op.is_fieldless() && (vector || name.starts_with("s_cbranch")) &&
        (reg->cls == RegClass::EXEC || reg->cls == RegClass::VCC))
      reg->width = static_cast<uint8_t>(wf.wf_size() / 32);
    return reg;
  };
  if (modifiers.valu_permutation != ValuPermutation::None) {
    const auto kind = modifiers.valu_permutation;
    if (kind == ValuPermutation::Swap16 || kind == ValuPermutation::Swap32) {
      const unsigned stride = kind == ValuPermutation::Swap16 ? 16 : 32;
      const auto lower = valu_swap_write_lanes(lanes, wf.wf_size(), stride, false);
      const auto upper = valu_swap_write_lanes(lanes, wf.wf_size(), stride, true);
      for (unsigned i = 0; i < 2; ++i) {
        const auto &operand = *inst.dst_operand(i);
        if (const auto reg = resolve(operand, false))
          access(*reg, i ? upper >> stride : lower << stride, 0xf, false);
        if (const auto reg = resolve(operand, true))
          access(*reg, i ? lower : upper, 0xf, true);
      }
      return;
    }
    const bool variable = kind == ValuPermutation::Perm16Var || kind == ValuPermutation::X16Var;
    const bool row16 = variable || kind == ValuPermutation::Perm16 || kind == ValuPermutation::X16;
    const auto *selector = kind == ValuPermutation::Perm64 ? nullptr : inst.src_operand(1);
    const auto *group = kind == ValuPermutation::Perm64 || variable ? nullptr : inst.src_operand(2);
    auto check_control = [&](const Operand *operand, uint64_t used_lanes, bool per_lane) {
      if (operand && used_lanes)
        if (const auto reg = resolve(*operand, false))
          access(*reg, reg->cls == RegClass::VGPR ? (per_lane ? used_lanes : 1) : ~uint64_t{0}, 0xf,
                 false);
    };
    check_control(selector, row16 && !variable ? lanes & 0x00ff00ff00ff00ffull : lanes, variable);
    check_control(group, row16 ? lanes & 0xff00ff00ff00ff00ull : lanes, false);
    auto control = [&](const Operand *operand, unsigned lane) -> uint32_t {
      if (!operand)
        return 0;
      if (const auto value = operand->const_value())
        return *value;
      if (const auto reg = resolve(*operand, false); reg && reg->cls == RegClass::VGPR)
        return registers.vector_control_value(*reg, lane).value_or(0);
      return registers.scalar_control_value(operand->encoding_value()).value_or(0);
    };
    const uint32_t low = variable ? 0 : control(selector, 0);
    const uint32_t high = control(group, 0);
    const auto permutation = valu_permutation_lanes(
        kind, lanes, wf.wf_size(), high, row16, modifiers.fi, modifiers.bound_ctrl,
        [&](unsigned lane) {
          return variable ? control(selector, lane)
                 : row16  ? (((lane & 15) < 8 ? low : high) >> ((lane & 7) * 4)) & 15
                          : low;
        });
    const auto [read_lanes, write_lanes] = permutation;
    if (read_lanes)
      if (const auto reg = resolve(*inst.src_operand(0), false))
        access(*reg, reg->cls == RegClass::VGPR ? read_lanes : ~uint64_t{0}, 0xf, false);
    if (const auto reg = resolve(*inst.dst_operand(0), true))
      access(*reg, write_lanes, 0xf, true);
    return;
  }
  if (name.starts_with("s_movrel")) {
    access({RegClass::M0, 0, 1}, ~uint64_t{0}, 0xf, false);
    const bool split = name.starts_with("s_movrelsd_2");
    const bool source_relative = name.starts_with("s_movrels");
    const bool destination_relative = split || name.starts_with("s_movreld");
    const auto &destination = *inst.dst_operand(0);
    if (destination_relative &&
        !relative_scalar_selector(wf, destination.encoding_value(), wf.m0(),
                                  std::max(1, destination.size_bits() / 32), true, split))
      return;
    auto check_relative = [&](const Operand &operand, bool relative, bool write) {
      if (!relative) {
        if (auto reg = resolve(operand, write))
          access(*reg, ~uint64_t{0}, 0xf, write);
        return;
      }
      const unsigned width = std::max(1, operand.size_bits() / 32);
      const auto selector =
          relative_scalar_selector(wf, operand.encoding_value(), wf.m0(), width, write, split)
              .value_or(0);
      if (const auto reg = scalar_register(selector, width, write))
        access(*reg, ~uint64_t{0}, 0xf, write);
    };
    check_relative(*inst.src_operand(0), source_relative, false);
    check_relative(*inst.dst_operand(0), destination_relative, true);
    return;
  }
  if (name.starts_with("v_movrel") || name.starts_with("v_swaprel")) {
    access({RegClass::M0, 0, 1}, ~uint64_t{0}, 0xf, false);
    const bool swap = name.starts_with("v_swaprel");
    const bool source_relative = swap || name.starts_with("v_movrels");
    const bool destination_relative =
        swap || name.starts_with("v_movreld") || name.starts_with("v_movrelsd");
    const bool split = swap || name.starts_with("v_movrelsd_2");
    const auto *source = swap ? inst.dst_operand(1) : inst.src_operand(0);
    const auto *destination = inst.dst_operand(0);
    auto relative = [&](const Operand &operand, uint32_t offset) -> std::optional<RegisterRef> {
      const auto reg = operand.to_register_ref();
      std::optional<uint32_t> base;
      if (reg && reg->cls == RegClass::VGPR)
        base = reg->index + (wf.vgpr_msb_for_role(operand.vgpr_msb_role()) << 8);
      const auto index = relative_vgpr_index(base, offset, operand.vgpr_count(),
                                             wf.vgpr_alloc().count, wf.cu().arch());
      if (!index)
        return std::nullopt;
      return RegisterRef{RegClass::VGPR, static_cast<uint16_t>(*index),
                         static_cast<uint8_t>(operand.vgpr_count())};
    };
    auto dst = destination_relative
                   ? relative(*destination, split ? (wf.m0() >> 16) & 0x3ffu : wf.m0())
                   : resolve(*destination, true);
    if (destination_relative && !dst)
      return;
    auto src = source_relative ? relative(*source, split ? wf.m0() & 0x3ffu : wf.m0())
                               : resolve(*source, false);
    if (source_relative && !src) {
      if (swap)
        return;
      // MOVREL's invalid relative source uses the executor's v0 fallback.
      src = RegisterRef{RegClass::VGPR, 0, static_cast<uint8_t>(source->vgpr_count())};
    }
    const auto source_bytes =
        modifiers.sdwa ? sdwa::sdwa_src_byte_mask(modifiers.src0_selection) : 0xf;
    const auto destination_bytes =
        modifiers.sdwa ? sdwa::sdwa_dst_byte_mask(modifiers.dst_selection, modifiers.dst_unused)
                       : 0xf;
    if (src && lanes)
      access(*src, src->cls == RegClass::VGPR ? src0_lanes : ~uint64_t{0}, source_bytes, false);
    if (dst)
      access(*dst, destination_lanes, destination_bytes, true);
    if (swap) {
      access(*dst, lanes, 0xf, false);
      access(*src, lanes, 0xf, true);
    }
    return;
  }
  if (selected_lane) {
    const auto *selector = first_lane ? nullptr : inst.src_operand(inst.num_src_operands() - 1);
    // WRITELANE's tied VDST precedes SRC0 on older targets and follows SRC1 on newer ones.
    if (scalar_lane_write && selector == inst.dst_operand(0))
      selector = inst.src_operand(inst.num_src_operands() - 2);
    if (selector)
      if (auto reg = resolve(*selector, false))
        access(*reg, ~uint64_t{0}, 0xf, false);
    const auto immediate_lane = selector ? selector->const_value() : std::nullopt;
    const unsigned lane =
        first_lane
            ? ((wf.exec_raw() & all_lanes) ? std::countr_zero(wf.exec_raw() & all_lanes) : 0)
            : (immediate_lane
                   ? *immediate_lane
                   : registers.scalar_control_value(selector->encoding_value()).value_or(0)) &
                  (wf.wf_size() - 1);
    for (int i = 0; i < inst.num_src_operands(); ++i) {
      if (!inst.tracks_explicit_source(i))
        continue;
      const auto *op = inst.src_operand(i);
      if (op == selector || (scalar_lane_write && op == inst.dst_operand(0)))
        continue;
      if (auto reg = resolve(*op, false))
        access(*reg, reg->cls == RegClass::VGPR ? uint64_t{1} << lane : ~uint64_t{0}, 0xf, false);
    }
    for (int i = 0; i < inst.num_dst_operands(); ++i)
      if (auto reg = resolve(*inst.dst_operand(i), true))
        access(*reg, reg->cls == RegClass::VGPR ? uint64_t{1} << lane : ~uint64_t{0}, 0xf, true);
    return;
  }
  std::optional<uint64_t> selection;
  if (name.starts_with("v_cndmask_") && lanes &&
      (wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA3 || wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA4 ||
       wf.cu().arch() == ROCJITSU_CODE_ARCH_CDNA3 || wf.cu().arch() == ROCJITSU_CODE_ARCH_CDNA4 ||
       wf.cu().arch() == ROCJITSU_CODE_ARCH_CDNA5)) {
    // Check the condition dependency before using its current value to select sources.
    const auto *condition = inst.num_src_operands() > 2 ? inst.src_operand(2) : nullptr;
    if (!condition || condition->is_fieldless()) {
      if (lanes & 0xffffffffu)
        access({RegClass::VCC, 0, 1}, ~uint64_t{0}, 0xf, false);
      if (lanes >> 32)
        access({RegClass::VCC, 1, 1}, ~uint64_t{0}, 0xf, false);
      selection = wf.vcc_mask();
    } else {
      if (auto reg = resolve(*condition, false))
        access(*reg, ~uint64_t{0}, 0xf, false);
      if (condition->encoding_value() == 124) {
        // NULL denotes zero for the whole mask, not a NULL/M0 register pair.
        selection = 0;
      } else if (auto value = condition->const_value()) {
        selection = *value;
      } else if (auto lo = registers.scalar_control_value(condition->encoding_value())) {
        selection = *lo;
        if (wf.wf_size() == 64) {
          const auto selector = condition->encoding_value();
          if (selector == 251 || selector == 106)
            selection = wf.vcc_mask();
          else if (selector == 252 || selector == 126)
            selection = wf.exec();
          else if (auto hi = registers.scalar_control_value(selector + 1))
            *selection |= uint64_t{*hi} << 32;
          else
            selection.reset();
        }
      }
    }
  }
  const Operand *buffer_payload =
      needs_buffer_payload_lanes(inst, wf.cu().arch()) ? inst.src_operand(0) : nullptr;
  const bool reads_buffer_payload =
      wf.cu().arch() == ROCJITSU_CODE_ARCH_CDNA5 ? !cdna5_type_mismatch : read_buffer_address;
  auto check_operand = [&](const Operand *op, bool write) {
    const bool mix_preservation = !write && op == inst.dst_operand(0) &&
                                  (name.find("mixlo_") != std::string_view::npos ||
                                   name.find("mixhi_") != std::string_view::npos);
    if (mix_preservation || !op || op == hwreg || op == modifiers.buffer_resource ||
        (!write && op == modifiers.buffer_address && !read_buffer_address) ||
        (!write && op == buffer_payload && !reads_buffer_payload) ||
        (!write && op == modifiers.buffer_offset && !read_buffer_offset) ||
        (!write && gfx11_or_gfx12_sample && !modifiers.image_reads_address && !op->is_vgpr()) ||
        op == modifiers.image_address || op == modifiers.image_atomic_data ||
        op == modifiers.image_store_data || (!pending_scalar_ && !op->is_vgpr()))
      return;
    if (!write && ds_vgpr0 && op->decoded_vgpr()) {
      check_ds_source(*op);
      return;
    }
    if (op == modifiers.block_data) {
      for (uint32_t word = 0; word < kBlockDwordCount; ++word)
        if (block_dword_enabled(wf.m0(), word))
          if (const auto reg = registers.block_data_register(*op, word, write))
            access(*reg, lanes, 0xf, write);
      return;
    }
    auto reg = resolve(*op, write);
    if (!reg || reg->cls == RegClass::ACC_VGPR)
      return;
    // gfx1201 DPP8/DPP16 SUBREV substitutes zero for a scalar/inline SRC1.
    if (!write && reverse_dpp && (modifiers.dpp || modifiers.dpp8) && op == routed_source &&
        reg->cls != RegClass::VGPR && wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA4)
      return;
    uint64_t accessed_lanes = reg->cls != RegClass::VGPR ? ~uint64_t{0}
                              : write                    ? destination_lanes
                              : op == routed_source      ? src0_lanes
                                                         : source_lanes;
    if (!write && op == buffer_payload && inst.data() && inst.data()->tag() == GLOBAL_MEM) {
      const auto &data = *inst.data_as<VectorMemState>();
      accessed_lanes = wf.cu().arch() == ROCJITSU_CODE_ARCH_CDNA5 ? data.exec_mask : data.lane_mask;
    }
    if (!write && !destination_lanes && modifiers.dpp && reg->cls != RegClass::VGPR &&
        wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA3_5 && name == "v_add_f32" &&
        op == inst.src_operand(1))
      return;
    if (!write && op == modifiers.dpp_masked_compare_rhs) {
      if (!destination_lanes)
        return;
      if (reg->cls == RegClass::VGPR)
        accessed_lanes &= destination_lanes;
    }
    if (!write && reg->cls == RegClass::VGPR && (inst.flags() & COMPACT_WAVE64_MEMORY_RESULT))
      accessed_lanes &= UINT32_MAX;
    if (!write && modifiers.cvt_scale && op == inst.src_operand(1) && reg->cls == RegClass::VGPR) {
      // Each data half-wave reads its selected byte, including from inactive lanes.
      const CvtScaleSelection &scale = *modifiers.cvt_scale;
      for (unsigned base = 0; base < wf.wf_size(); base += 16) {
        const uint64_t consumers = (accessed_lanes >> base) & 0xffff;
        if (consumers)
          access(*reg, consumers << scale.lane(base), 1u << scale.byte(base), false);
      }
      return;
    }
    // Interpolation coefficients are quad broadcasts, including inactive lanes.
    if (!write && reg->cls == RegClass::VGPR && name.starts_with("v_interp_") &&
        (wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA3 ||
         wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA3_5 ||
         wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA4)) {
      const uint64_t quads =
          (lanes | (lanes >> 1) | (lanes >> 2) | (lanes >> 3)) & 0x1111111111111111ULL;
      const bool p2 = name.starts_with("v_interp_p2");
      if (op == inst.src_operand(0))
        accessed_lanes = quads << (p2 ? 2 : 1);
      else if (op == inst.src_operand(2) && !p2)
        accessed_lanes = quads;
    }
    if (selection && !write && reg->cls == RegClass::VGPR) {
      if (op == inst.src_operand(1))
        accessed_lanes &= *selection;
      else if (op == inst.src_operand(0) && !modifiers.dpp && !modifiers.dpp8)
        accessed_lanes &= ~*selection;
    }
    if (vector && !lanes && !write && !inst.is_memory_op())
      return;
    uint8_t bytes = op->register_byte_mask();
    if (!write && op == modifiers.packed_store_data && reg->width == (op->size_bits() + 31) / 32) {
      const RegisterRef tail{reg->cls, static_cast<uint16_t>(reg->index + reg->width - 1), 1};
      if (--reg->width)
        access(*reg, accessed_lanes, bytes, false);
      access(tail, accessed_lanes, 0x3, false);
      return;
    }
    if (modifiers.sdwa && reg->cls == RegClass::VGPR) {
      if (write)
        bytes = sdwa::sdwa_dst_byte_mask(modifiers.dst_selection, modifiers.dst_unused);
      else if (op == modifiers.src0)
        bytes = sdwa::sdwa_src_byte_mask(modifiers.src0_selection);
      else if (op == modifiers.src1)
        bytes = sdwa::sdwa_src_byte_mask(modifiers.src1_selection);
    }
    if (reg->cls == RegClass::VCC && vector && !write && op->is_fieldless()) {
      if (lanes & 0xffffffffu)
        access({RegClass::VCC, 0, 1}, ~uint64_t{0}, bytes, false);
      if (lanes >> 32)
        access({RegClass::VCC, 1, 1}, ~uint64_t{0}, bytes, false);
    } else {
      if (!write && reg->cls == RegClass::VGPR && reg->width == 2) {
        const auto base = wf.vgpr_alloc().base;
        if (auto mapped = registers.remapped_valu_source(*op, base + reg->index, 2)) {
          for (const auto source : {mapped->lo, mapped->hi})
            access({RegClass::VGPR, static_cast<uint16_t>(source - base), 1}, accessed_lanes, bytes,
                   false);
          return;
        }
      }
      access(*reg, accessed_lanes, bytes, write);
      if (gfx10_valu && !write && reg->cls == RegClass::VGPR && reg->index != 0 &&
          reg->width < op->vgpr_count())
        access({RegClass::VGPR, 0, 1}, accessed_lanes, bytes, false);
    }
  };
  if (!modifiers.constant_float_compare || !wf.dx10_clamp()) {
    for (int i = 0; i < inst.num_src_operands(); ++i) {
      if (!inst.tracks_explicit_source(i))
        continue;
      check_operand(inst.src_operand(i), false);
    }
  }
  const bool partial_destination =
      inst.dst_operand(0) && inst.dst_operand(0)->register_byte_mask() != 0xf;
  // Liveness models preservation as a use; preserving an untouched byte or
  // lane does not consume its pending memory result.
  const bool preserves_destination = partial_destination || modifiers.sdwa || modifiers.dpp ||
                                     modifiers.memory_result_bytes != 0xf ||
                                     modifiers.memory_result_last_bytes != 0xf;
  std::vector<const Operand *> implicit_operands;
  if (!preserves_destination)
    inst.implicit_use_operands(implicit_operands);
  for (const Operand *op : implicit_operands)
    check_operand(op, false);
  // Legacy FLAT selects scratch backing from the active lanes' addresses.
  // Compute that predicate once and check the backing register outside the loop.
  const RegisterRef scratch{RegClass::FLAT_SCRATCH, 0, 2};
  if (lanes && pending_.pending(scratch) && inst.raw_encoding() &&
      (name.starts_with("flat_") || name.starts_with("scratch_"))) {
    const auto arch = wf.cu().arch();
    if (arch == ROCJITSU_CODE_ARCH_CDNA1 || arch == ROCJITSU_CODE_ARCH_CDNA2 ||
        arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4) {
      const unsigned segment = (inst.raw_encoding()[0] >> 14) & 3;
      bool uses_scratch = segment == 1;
      const uint32_t private_hi = wf.private_aperture_base() >> 32;
      if (segment == 0 && private_hi) {
        const auto address = registers.source_register(*inst.src_operand(0));
        const uint32_t offset = inst.raw_encoding()[0] & 0xfff;
        uint64_t remaining = lanes;
        while (remaining && !uses_scratch) {
          const unsigned lane = std::countr_zero(remaining);
          remaining &= remaining - 1;
          if (address)
            uses_scratch =
                (registers.vector_control_value(*address, lane).value_or(0) + offset) >> 32 ==
                private_hi;
        }
      }
      if (uses_scratch)
        access(scratch, ~uint64_t{0}, 0xf, false);
    }
  }
  // A fork or join touches at most one branch stack entry, chosen by MODE.CSP at
  // run time. Its implicit registers are the static superset of every entry, kept
  // for liveness; checking them all would report unrelated pending SGPRs.
  const bool branch_stack = inst.flags() & BRANCH_STACK;
  std::optional<BranchStackAccess> stack_access;
  if (branch_stack && inst.src_operand(0)) {
    const Operand &src0 = *inst.src_operand(0);
    stack_access = name == "s_cbranch_join" ? join_stack_access(wf, registers.read_scalar(src0))
                                            : fork_stack_access(wf, registers.read_scalar64(src0));
  }
  const auto stack_entry = [&]() {
    return RegisterRef{RegClass::SGPR, static_cast<uint16_t>(stack_access->first_wave_sgpr), 4};
  };
  RegisterSet implicit, preservation_uses;
  const bool distinguishes_preservation = inst.implicit_preservation_uses(preservation_uses);
  // Legacy partial-destination hooks do not distinguish preservation from
  // genuine hidden reads. Instructions that do provide the distinction retain
  // their real implicit dependencies while dropping only preservation uses.
  if (!branch_stack && (!preserves_destination || distinguishes_preservation)) {
    inst.implicit_uses(implicit);
    implicit -= preservation_uses;
  }
  // Image coordinates were checked above with their A16 byte masks. Their
  // register-level liveness uses must not turn a packed half into a full read.
  if (modifiers.image_address) {
    RegisterSet coordinate_uses;
    for (uint32_t selector : modifiers.image_coordinates)
      coordinate_uses.expand(RegisterRef{RegClass::VGPR, static_cast<uint16_t>(selector), 1});
    implicit -= coordinate_uses;
  }
  implicit.for_each([&](RegisterRef reg) {
    // Operand-derived reads preserve their bank roles above.
    if (reg.cls == RegClass::VGPR && !implicit_operands.empty())
      return;
    if (vector && !lanes)
      return;
    access(reg, reg.cls == RegClass::VGPR ? lanes : ~uint64_t{0}, 0xf, false);
  });
  if (stack_access && !stack_access->write)
    access(stack_entry(), ~uint64_t{0}, 0xf, false);
  if (!inst.is_memory_wait_producer()) {
    for (int i = 0; i < inst.num_dst_operands(); ++i)
      check_operand(inst.dst_operand(i), true);
    implicit = RegisterSet{};
    if (!branch_stack)
      inst.implicit_defs(implicit);
    implicit.for_each([&](RegisterRef reg) {
      access(reg, reg.cls == RegClass::VGPR ? lanes : ~uint64_t{0}, 0xf, true);
    });
    if (stack_access && stack_access->write)
      access(stack_entry(), ~uint64_t{0}, 0xf, true);
  }
}

namespace {
// Whether loads producing this event write their results to registers in issue
// order within their order class, so overwriting a pending result of the same
// class does not require a wait.
bool has_ordered_writeback(const waitcheck_detail::ClassifiedEvent &event, rj_code_arch_t arch) {
  using namespace waitcheck_detail;
  switch (event.counter) {
  case WaitCounterKind::Load:
    return has_ordered_vmem_writeback(arch);
  case WaitCounterKind::Ds:
    // LDS writeback is in-order on all targets.
    return event.kind == WaitEventKind::Ds ||
           (event.kind == WaitEventKind::Gds && has_ordered_gds_writeback(arch));
  case WaitCounterKind::Exp:
    // LDS direct and parameter loads use EXP independently of exports and
    // indexed LDS accesses.
    return event.kind == WaitEventKind::LdsDirect && has_ordered_lds_direct_writeback(arch);
  default:
    return false;
  }
}
} // namespace

void MemoryWaitScoreboard::check_memory_result(const Instruction &inst, Wavefront &wf,
                                               uint64_t vector_lanes,
                                               const RegisterModifiers &modifiers) {
  using namespace waitcheck_detail;
  const RegisterAccess registers(wf);
  if (!result_is_written(inst, wf))
    return;
  std::array<RegisterRef, kBlockDwordCount> results{};
  unsigned count = 0;
  bool pending = false;
  for (int i = 0; i < inst.num_dst_operands(); ++i) {
    const auto &operand = *inst.dst_operand(i);
    if (&operand == modifiers.block_data) {
      for (uint32_t word = 0; word < kBlockDwordCount; ++word)
        if (block_dword_enabled(wf.m0(), word))
          if (const auto reg = registers.block_data_register(operand, word, /*write=*/true)) {
            assert(count < results.size());
            results[count++] = *reg;
            pending |= pending_.pending(*reg, true);
          }
      continue;
    }
    auto reg = registers.memory_destination_register(inst, operand);
    if (!reg && operand.is_fieldless())
      if (auto special = operand.to_special_reg_class())
        reg = RegisterRef{*special, 0, 1};
    if (!operand.is_vgpr() && operand.has_register_selector()) {
      const auto range = resolve_scalar_register_range(wf, operand.encoding_value(),
                                                       std::max(1, operand.size_bits() / 32));
      reg = range ? range->register_ref() : std::nullopt;
    }
    // Completion validates every vector destination together, including the
    // independent pointer word returned by LDS stack operations.
    if (reg && reg->cls == RegClass::TTMP && !wf.shader_can_write_ttmp())
      continue;
    if (operand.decoded_vgpr() && !reg)
      return;
    if (reg) {
      assert(count < results.size());
      results[count++] = *reg;
      pending |= pending_.pending(*reg, true);
    }
  }
  if (!pending)
    return;
  const auto classified = RuntimeWaitTarget::classify_events(inst, wf.cu().arch());
  if (classified.failed())
    return;
  const auto xevent =
      std::ranges::find(classified.value(), WaitCounterKind::X, &ClassifiedEvent::counter);
  const bool replay = outstanding(WaitCounterKind::X) != 0 && xevent != classified.value().end() &&
                      !RuntimeWaitTarget::is_async_transfer_kind(xevent->kind);
  if (replay)
    xcnt_group(xevent->kind == WaitEventKind::Smem);
  const auto defs = std::ranges::find(classified.value(), TrackedRegisterSource::Defs,
                                      &ClassifiedEvent::registers);
  const bool has_defs = defs != classified.value().end();
  const auto order = has_defs && has_ordered_writeback(*defs, wf.cu().arch())
                         ? ordered_write_order(*defs, wf.cu().arch())
                         : kUnordered;
  const bool vector_replay =
      replay && xevent->kind != WaitEventKind::Smem && (wf.mode_raw() & (1u << 25));
  for (unsigned i = 0; i < count; ++i) {
    auto reg = results[i];
    if (reg.cls != RegClass::VGPR) {
      access(reg, ~uint64_t{0}, 0xf, true);
      continue;
    }
    if (!has_defs)
      continue;
    auto check = [&](RegisterRef range, uint8_t bytes) {
      if (inst.data() && inst.data()->tag() == GLOBAL_MEM) {
        const auto &d = *inst.data_as<VectorMemState>();
        if (d.is_load && (inst.flags() & CONDITIONAL_LAST_VGPR_DEF) &&
            uint32_t{range.index} + range.width - 1 ==
                d.dst_reg_base - wf.vgpr_alloc().base + d.destination_vgpr_count() &&
            !--range.width)
          return;
      }
      if (vector_replay)
        xcnt_ordered_write(range, vector_lanes, bytes);
      access(range, vector_lanes, bytes, true, order);
    };
    for_each_memory_result_range(reg, modifiers, wf.cu().sram_ecc(), check);
  }
}

void MemoryWaitScoreboard::clear() {
  events_.clear();
  superseded_.clear();
  pending_scalar_ = false;
  may_overlap_ = false;
  pending_.reset();
  issued_.fill(0);
  retired_.fill(0);
  unordered_.fill(false);
  ordering_kinds_.fill(0);
  orders_.clear();
  last_order_.fill(kUnordered);
  last_capacity_eligible_.fill(false);
  xcnt_scalar_ = false;
  async_replay_drained_ = 0;
  translations_.clear();
  expert_source_completions_.clear();
}

bool MemoryWaitScoreboard::cancel_last_issue(uint64_t pc, std::span<const WaitCounterType> counters,
                                             rj_code_arch_t arch, bool xcnt, bool vm_vsrc) {
  const auto model = waitcnt_model(arch);
  if (model.failed())
    return false;
  std::array<uint32_t, kCounters> units{};
  for (const auto counter : counters)
    ++units[static_cast<size_t>(canonical_wait_counter(counter, model.value()))];
  if (xcnt)
    units[static_cast<size_t>(WaitCounterKind::X)] = 1;
  if (vm_vsrc)
    units[static_cast<size_t>(WaitCounterKind::VmVsrc)] = 1;
  return cancel_issue_units(pc, units);
}

bool MemoryWaitScoreboard::cancel_last_issue(uint64_t pc, const IssueSnapshot &before) {
  std::array<uint32_t, kCounters> units{};
  for (size_t i = 0; i < kCounters; ++i) {
    if (before.issued[i] > issued_[i] ||
        issued_[i] - before.issued[i] > std::numeric_limits<uint32_t>::max())
      return false;
    units[i] = static_cast<uint32_t>(issued_[i] - before.issued[i]);
  }
  if (!cancel_issue_units(pc, units))
    return false;
  return true;
}

bool MemoryWaitScoreboard::cancel_issue_units(uint64_t pc,
                                              const std::array<uint32_t, kCounters> &units) {
  const auto sequences = issued_;
  for (size_t i = 0; i < kCounters; ++i)
    if (units[i] > issued_[i] - retired_[i])
      return false;

  std::erase_if(events_, [&](const Event &event) {
    const auto i = static_cast<size_t>(event.counter);
    return units[i] && event.pc == pc && event.sequence == sequences[i];
  });
  for (size_t i = 0; i < kCounters; ++i) {
    if (!units[i])
      continue;
    issued_[i] -= units[i];
    if (last_order_[i] != kUnordered) {
      auto &order = orders_[last_order_[i]];
      order.issued -= units[i];
      if (last_capacity_eligible_[i]) {
        order.capacity_issued -= units[i];
        order.capacity_retired = std::min(order.capacity_retired, order.capacity_issued);
      }
    }
    if (issued_[i] == retired_[i]) {
      unordered_[i] = false;
      ordering_kinds_[i] = 0;
    }
  }
  std::erase_if(translations_, [&](const auto &entry) {
    const auto xcounter = static_cast<size_t>(WaitCounterKind::X);
    return units[xcounter] && entry.sequence == sequences[xcounter];
  });
  async_replay_drained_ =
      std::min(async_replay_drained_, issued_[static_cast<size_t>(WaitCounterKind::Async)]);
  std::erase_if(superseded_, [&](const auto &old) {
    const auto i = static_cast<size_t>(old.replacement_counter);
    if (!units[i] || old.replacement_sequence != sequences[i] || old.replacement_pc != pc)
      return false;
    const auto &event = old.event;
    if (!(event.async_replay_source && event.sequence <= async_replay_drained_) &&
        !completed(event.counter, event.sequence, event.order, event.order_sequence,
                   event.capacity_order_sequence))
      events_.push_back(event);
    return true;
  });
  const auto source = static_cast<size_t>(WaitCounterKind::VmVsrc);
  if (units[source])
    expert_source_completions_.erase(sequences[source]);
  // Rebuild all shadow bytes so overlapping, unrelated producers survive.
  pending_.reset();
  pending_scalar_ = false;
  may_overlap_ = false;
  for (const auto &event : events_) {
    pending_scalar_ |= event.reg.cls != RegClass::VGPR;
    may_overlap_ |= pending_.mark(event.reg, event.shadow_bits());
  }
  return true;
}

uint64_t MemoryWaitScoreboard::issue(WaitCounterKind counter, bool unordered,
                                     uint32_t ordering_kind, uint32_t units,
                                     bool capacity_eligible) {
  auto i = static_cast<size_t>(counter);
  std::erase_if(superseded_,
                [counter](const auto &event) { return event.replacement_counter == counter; });
  ordering_kinds_[i] |= ordering_kind;
  unordered_[i] |= unordered || std::popcount(ordering_kinds_[i]) > 1;
  issued_[i] += units;
  last_order_[i] = kUnordered;
  last_capacity_eligible_[i] = false;
  if (!unordered) {
    const auto it = std::ranges::find_if(orders_, [&](const Order &order) {
      return order.counter == counter && order.kind == ordering_kind;
    });
    const auto slot = static_cast<uint16_t>(it - orders_.begin());
    if (it == orders_.end())
      orders_.push_back({counter, VmemWriteback::Ordinary, ordering_kind});
    orders_[slot].last_writeback = VmemWriteback::Ordinary;
    orders_[slot].issued += units;
    if (capacity_eligible)
      orders_[slot].capacity_issued += units;
    last_order_[i] = slot;
    last_capacity_eligible_[i] = capacity_eligible;
  }
  return issued_[i];
}

namespace {
VmemWriteback vmem_writeback(const waitcheck_detail::ClassifiedEvent &event, rj_code_arch_t arch) {
  using namespace waitcheck_detail;
  if (has_ordered_vmem_reads(arch) && event.counter == WaitCounterKind::Load) {
    if (event.kind == WaitEventKind::Sample)
      return event.point_sample_accel ? VmemWriteback::PointSample : VmemWriteback::Sample;
    if (event.kind == WaitEventKind::Bvh)
      return VmemWriteback::Bvh;
  }
  return VmemWriteback::Ordinary;
}

} // namespace

uint64_t MemoryWaitScoreboard::issue(const waitcheck_detail::ClassifiedEvent &event,
                                     rj_code_arch_t arch, uint32_t units, bool capacity_eligible) {
  const auto kind =
      waitcheck_detail::RuntimeWaitTarget::completion_order_kind(event.counter, event.kind, arch);
  const auto sequence = issue(event.counter, !kind, kind.value_or(0), units, capacity_eligible);
  const auto fifo = last_order_[static_cast<size_t>(event.counter)];
  if (fifo != kUnordered)
    orders_[fifo].last_writeback = vmem_writeback(event, arch);
  return sequence;
}

uint32_t MemoryWaitScoreboard::issue_units(const Instruction &inst,
                                           const waitcheck_detail::ClassifiedEvent &event,
                                           rj_code_arch_t arch) {
  return waitcheck_detail::RuntimeWaitTarget::issue_units(inst, event, arch);
}

MemoryWaitScoreboard::WriteOrder
MemoryWaitScoreboard::ordered_write_order(const waitcheck_detail::ClassifiedEvent &event,
                                          rj_code_arch_t arch) const {
  const auto kind =
      waitcheck_detail::RuntimeWaitTarget::completion_order_kind(event.counter, event.kind, arch);
  if (!kind)
    return kUnordered;
  const auto it = std::ranges::find_if(orders_, [&](const Order &order) {
    return order.counter == event.counter && order.kind == *kind;
  });
  if (it == orders_.end())
    return kUnordered;
  return {static_cast<uint16_t>(it - orders_.begin()), vmem_writeback(event, arch)};
}

bool MemoryWaitScoreboard::completed(WaitCounterKind counter, uint64_t sequence, uint16_t order,
                                     uint64_t order_sequence,
                                     uint64_t capacity_order_sequence) const {
  return sequence <= retired_[static_cast<size_t>(counter)] ||
         (order != kUnordered && (order_sequence <= orders_[order].retired ||
                                  (capacity_order_sequence &&
                                   capacity_order_sequence <= orders_[order].capacity_retired)));
}

void MemoryWaitScoreboard::backpressure(WaitCounterKind counter, uint32_t capacity,
                                        uint32_t incoming_units) {
  assert(incoming_units != 0 && incoming_units <= capacity);
  const auto remaining = capacity - incoming_units;
  bool changed = false;
  for (auto &order : orders_) {
    if (order.counter != counter || order.capacity_issued <= remaining)
      continue;
    // Eligible requests alone establish capacity progress. When one FIFO
    // accounts for every outstanding unit, the same proof advances the full
    // counter prefix, including empty issues already retired.
    const auto i = static_cast<size_t>(counter);
    if (!unordered_[i] &&
        order.issued - order.retired == issued_[i] - retired_[i] &&
        order.capacity_issued - order.capacity_retired == issued_[i] - retired_[i] &&
        issued_[i] - retired_[i] > remaining)
      retired_[i] = issued_[i] - remaining;
    // Only requests eligible for this capacity proof can establish that an
    // older request finished. Unused FLAT credits still count for waits.
    const auto through = order.capacity_issued - remaining;
    changed |= through > order.capacity_retired;
    order.capacity_retired = std::max(order.capacity_retired, through);
  }
  if (changed)
    retire_completed();
}

void MemoryWaitScoreboard::stamp_order(Event &event) const {
  const auto counter = static_cast<size_t>(event.counter);
  if (event.sequence == issued_[counter] && last_order_[counter] != kUnordered) {
    event.order = last_order_[counter];
    event.order_sequence = orders_[event.order].issued;
    event.writeback = orders_[event.order].last_writeback;
    if (last_capacity_eligible_[counter])
      event.capacity_order_sequence = orders_[event.order].capacity_issued;
  }
}

uint32_t MemoryWaitScoreboard::required_wait(const Event &event) const {
  if (event.async_replay_source ||
      (event.counter == WaitCounterKind::VmVsrc && event.reg.cls != RegClass::VGPR))
    return 0;
  const auto i = static_cast<size_t>(event.counter);
  const auto required = !unordered_[i] ? issued_[i] - event.sequence
                        : event.order != kUnordered
                            ? orders_[event.order].issued - event.order_sequence
                            : 0;
  if (event.counter == WaitCounterKind::VaVdst) {
    using waitcheck_detail::WaitcheckTarget;
    const auto *core = event.wmma_wait_states ? expert_core_valu_order() : nullptr;
    const uint64_t younger_core =
        core && core->issued > event.preceding_core_valu ? core->issued - event.preceding_core_valu : 0;
    const uint64_t core_bound = waitcheck_detail::RuntimeWaitTarget::wmma_core_wait_bound(
        younger_core, event.wmma_wait_states);
    const auto maximum =
        WaitcheckTarget::maximum_dependency_wait(WaitcntModel::SplitGfx12, event.counter);
    return static_cast<uint32_t>(std::min<uint64_t>(maximum, std::max(required, core_bound)));
  }
  return static_cast<uint32_t>(
      std::min<uint64_t>(required, event.counter == WaitCounterKind::VmVsrc ? 6u : UINT32_MAX));
}

// GFX11 FLAT tracks each lane under both counters. A native, ordered overwrite
// replaces the result on its own pipeline and drops the other counter's register
// dependency on those bytes. Both counters retain their original issue positions.
void MemoryWaitScoreboard::supersede_flat_result(const Event &replacement) {
  const auto other_counter =
      replacement.counter == WaitCounterKind::Ds ? WaitCounterKind::Load : WaitCounterKind::Ds;
  for (size_t i = 0, count = events_.size(); i < count; ++i) {
    const auto &candidate = events_[i];
    if (candidate.counter != other_counter || candidate.reg.cls != RegClass::VGPR ||
        candidate.writeback_lanes == ~uint64_t{0})
      continue;
    const auto old = candidate;
    const uint64_t lanes = replacement.lanes & old.lanes & ~old.writeback_lanes;
    const uint8_t bytes = replacement.bytes & old.bytes;
    const unsigned first = std::max(old.reg.index, replacement.reg.index);
    const unsigned end =
        std::min(old.reg.index + old.reg.width, replacement.reg.index + replacement.reg.width);
    if (!lanes || !bytes || first >= end)
      continue;

    events_[i].lanes &= ~lanes;
    auto part = old;
    part.lanes = lanes;
    const auto retain = [&](unsigned start, unsigned stop, uint8_t mask) {
      if (start < stop && mask) {
        part.reg.index = static_cast<uint16_t>(start);
        part.reg.width = static_cast<uint8_t>(stop - start);
        part.bytes = mask;
        events_.push_back(part);
      }
    };
    retain(old.reg.index, first, old.bytes);
    retain(end, old.reg.index + old.reg.width, old.bytes);
    retain(first, end, old.bytes & ~bytes);
    part.reg.index = static_cast<uint16_t>(first);
    part.reg.width = static_cast<uint8_t>(end - first);
    part.bytes = bytes;
    superseded_.push_back({part, replacement.counter, replacement.sequence, replacement.pc});
  }
}

void MemoryWaitScoreboard::add(Event event, bool ordered_native_result) {
  if (!event.lanes || !event.bytes || !event.reg.width)
    return;
  event.reg = pending_.canonical(event.reg);
  stamp_order(event);
  // A newer dependency can replace an older overlapping one when its
  // completion also proves the older one complete. This bounds repeated
  // writes to unused destinations without hiding differently routed sources.
  const bool overlap = pending_.mark(event.reg, event.shadow_bits());
  may_overlap_ |= overlap;
  if (overlap && ordered_native_result && event.reg.cls == RegClass::VGPR &&
      event.order != kUnordered)
    supersede_flat_result(event);
  if (overlap) {
    const auto expert_source_route_dominates = [&](const Event &old) {
      if (old.counter != WaitCounterKind::VmVsrc || event.counter != WaitCounterKind::VmVsrc ||
          old.reg.cls != RegClass::VGPR)
        return true;
      const auto older = expert_source_completions_.find(old.sequence);
      const auto younger = expert_source_completions_.find(event.sequence);
      if (older == expert_source_completions_.end() || younger == expert_source_completions_.end())
        return false;
      uint64_t remaining = old.lanes;
      for (const auto &old_route : older->second) {
        const uint64_t lanes = old_route.lanes & old.lanes;
        if (!lanes)
          continue;
        remaining &= ~lanes;
        const bool covered =
            std::ranges::any_of(younger->second, [&](const SourceRoute &new_route) {
              if ((new_route.lanes & lanes) != lanes || new_route.counter != old_route.counter ||
                  new_route.sequence < old_route.sequence || new_route.order != old_route.order)
                return false;
              if (old_route.order == kUnordered)
                return true;
              // A capacity proof for the newer route must also prove the older
              // route complete; an ordinary partial wait follows issue order.
              return new_route.order_sequence >= old_route.order_sequence &&
                     (!new_route.capacity_order_sequence ||
                      (old_route.capacity_order_sequence &&
                       new_route.capacity_order_sequence >= old_route.capacity_order_sequence));
            });
        if (!covered)
          return false;
      }
      return !remaining;
    };
    std::erase_if(events_, [&](const Event &old) {
      if (!old.lanes)
        return true;
      const bool ordered = event.order != kUnordered && old.order == event.order &&
                           ordered_vmem_writebacks(old.writeback, event.writeback) &&
                           old.order_sequence <= event.order_sequence &&
                           expert_source_route_dominates(old);
      // Scalar replay sources have a different proof: every sufficient wait
      // drains their entire X group, so its newest covering source is enough.
      const bool scalar_replay =
          xcnt_scalar_ && old.counter == WaitCounterKind::X && event.counter == WaitCounterKind::X;
      // Multiple enabled export channels may select the same physical source.
      const bool same_export =
          event.export_source && old.export_source && old.sequence == event.sequence;
      // A native DS source may retire on LGKMCNT before another GFX10 scalar
      // source. Keep both sides of that possible lifetime difference.
      const bool superseded =
          (ordered || scalar_replay || same_export ||
           (event.gds_source_lock && old.gds_source_lock) ||
           (old.counter == WaitCounterKind::VmVsrc && event.counter == WaitCounterKind::VmVsrc &&
            event.reg.cls != RegClass::VGPR && !old.gfx10_native_ds_source &&
            !event.gfx10_native_ds_source)) &&
          old.protects_source() == event.protects_source() && old.reg.cls == event.reg.cls &&
          (event.writeback_lanes & old.writeback_lanes) == old.writeback_lanes &&
          event.reg.index <= old.reg.index &&
          event.reg.index + event.reg.width >= old.reg.index + old.reg.width &&
          (event.lanes & old.lanes) == old.lanes && (event.bytes & old.bytes) == old.bytes;
      if (superseded && old.sequence < event.sequence) {
        std::erase_if(superseded_, [&](const auto &saved) {
          return saved.replacement_counter == event.counter &&
                 saved.replacement_sequence != event.sequence;
        });
        superseded_.push_back({old, event.counter, event.sequence, event.pc});
      }
      return superseded;
    });
  }
  events_.push_back(event);
  pending_scalar_ |= event.reg.cls != RegClass::VGPR;
}

void MemoryWaitScoreboard::clear_destination(const Event &event) { pending_.clear(event.reg); }

void MemoryWaitScoreboard::rebuild_mask() {
  if (pending_scalar_)
    pending_scalar_ = std::ranges::any_of(
        events_, [](const Event &event) { return event.reg.cls != RegClass::VGPR; });
  if (events_.empty())
    may_overlap_ = false;
  if (!may_overlap_)
    return;
  // Erasing a record cleared its destinations. Restore any overlapping records
  // that remain; unrelated shadow bytes already have the right value.
  for (const auto &event : events_)
    pending_.mark(event.reg, event.shadow_bits());
}

bool MemoryWaitScoreboard::drain_async_replay() {
  const auto issued_async = issued_[static_cast<size_t>(WaitCounterKind::Async)];
  return std::exchange(async_replay_drained_, issued_async) != issued_async;
}

void MemoryWaitScoreboard::wait(WaitCounterKind counter, uint32_t threshold) {
  const bool drained_async_sources =
      counter == WaitCounterKind::X && !threshold && drain_async_replay();
  if (apply_wait(counter, threshold) || drained_async_sources)
    retire_completed();
}

void MemoryWaitScoreboard::wait_ordered_xcnt(uint32_t threshold) {
  if (apply_wait(WaitCounterKind::X, threshold))
    retire_completed();
}

bool MemoryWaitScoreboard::apply_wait(WaitCounterKind counter, uint32_t threshold) {
  // The caller normalizes split and legacy spellings into one sequence
  // domain for each architectural counter.
  const size_t i = static_cast<size_t>(counter);
  const uint64_t through = issued_[i] > threshold ? issued_[i] - threshold : 0;
  bool changed = false;
  if ((!threshold || !unordered_[i]) && through > retired_[i]) {
    retired_[i] = through;
    changed = true;
  }
  // A mixed counter still bounds every ordered class independently. It does
  // not prove readiness of an unordered result such as SMEM.
  for (auto &order : orders_)
    if (order.counter == counter && order.issued > threshold &&
        order.issued - threshold > order.retired) {
      order.retired = order.issued - threshold;
      changed = true;
    }
  if (!threshold) {
    unordered_[i] = false;
    ordering_kinds_[i] = 0;
    for (auto &order : orders_)
      if (order.counter == counter)
        order.retired = order.issued;
    // Only a full native DS drain proves its scalar inputs have been read.
    // VMEM, GDS, generic FLAT, and skipped DS locks retain their own proof.
    if (counter == WaitCounterKind::Ds)
      for (auto &event : events_)
        if (event.gfx10_native_ds_issued && event.lanes) {
          event.lanes = 0;
          changed = true;
        }
  }
  if (counter == WaitCounterKind::X) {
    std::erase_if(translations_, [&](const auto &entry) { return entry.sequence <= through; });
  } else if (xcnt_scalar_ && counter == WaitCounterKind::Km && !threshold) {
    changed |= apply_wait(WaitCounterKind::X, 0);
  }
  return changed;
}

const MemoryWaitScoreboard::Order *MemoryWaitScoreboard::expert_core_valu_order() const {
  constexpr uint32_t kind = 1u << static_cast<unsigned>(waitcheck_detail::kValuCore);
  const auto order = std::ranges::find_if(orders_, [](const Order &entry) {
    return entry.counter == WaitCounterKind::VaVdst && entry.kind == kind;
  });
  return order == orders_.end() ? nullptr : &*order;
}

void MemoryWaitScoreboard::retire_completed() {
  constexpr uint32_t core_kind = 1u << static_cast<unsigned>(waitcheck_detail::kValuCore);
  const auto *core = ordering_kinds_[static_cast<size_t>(WaitCounterKind::VaVdst)] & core_kind
                         ? expert_core_valu_order()
                         : nullptr;
  const uint64_t retired_core = core ? core->retired : 0;
  const auto size = events_.size();
  std::erase_if(events_, [&](Event &event) {
    // Completing the active LDS route of a generic FLAT does not retire
    // VGPR source lanes that the same instruction sent to global memory.
    if (event.counter == WaitCounterKind::VmVsrc && event.reg.cls == RegClass::VGPR) {
      if (const auto it = expert_source_completions_.find(event.sequence);
          it != expert_source_completions_.end())
        for (const auto &route : it->second)
          if (route.lanes && completed(route.counter, route.sequence, route.order,
                                       route.order_sequence, route.capacity_order_sequence))
            event.lanes &= ~route.lanes;
    }
    const bool retire = !event.lanes ||
                        (event.wmma_wait_states &&
                         retired_core > event.preceding_core_valu + event.wmma_wait_states) ||
                        (event.async_replay_source && event.sequence <= async_replay_drained_) ||
                        completed(event.counter, event.sequence, event.order, event.order_sequence,
                                  event.capacity_order_sequence);
    if (retire)
      clear_destination(event);
    return retire;
  });
  if (size != events_.size())
    rebuild_mask();
  std::erase_if(expert_source_completions_, [&](const auto &pair) {
    const auto &[sequence, routes] = pair;
    return sequence <= retired_[static_cast<size_t>(WaitCounterKind::VmVsrc)] ||
           std::ranges::all_of(routes, [&](const auto &route) {
             return !route.lanes || completed(route.counter, route.sequence, route.order,
                                              route.order_sequence, route.capacity_order_sequence);
           });
  });
  if (!xcnt_scalar_ && outstanding(WaitCounterKind::X)) {
    // Completion proves translation. Map the waited counter position back
    // into the ordered X queue; counts from mixed families are not X ages.
    uint64_t translated = 0;
    for (const auto &entry : translations_)
      if (completed(entry.completion, entry.completion_sequence, entry.order, entry.order_sequence,
                    entry.capacity_order_sequence))
        translated = std::max(translated, entry.sequence);
    if (translated > retired_[static_cast<size_t>(WaitCounterKind::X)])
      wait_ordered_xcnt(
          static_cast<uint32_t>(issued_[static_cast<size_t>(WaitCounterKind::X)] - translated));
  }
}

void MemoryWaitScoreboard::check_prefetch_xcnt() const {
  // An enabled scalar prefetch requires a full drain before its source group
  // is selected. Report the outstanding producer once per executed prefetch.
  if (!reporter_ || (!outstanding(WaitCounterKind::X) &&
                     async_replay_drained_ == issued_[static_cast<size_t>(WaitCounterKind::Async)]))
    return;
  const auto event = std::ranges::find_if(events_, &Event::protects_replay_source);
  if (!outstanding(WaitCounterKind::X) && event == events_.end())
    return;
  const RegisterRef pc{RegClass::PC, 0, 1};
  // A PC-relative request with immediate operands has a counter position but
  // may have no tracked register source. The counter still requires a zero wait.
  const Event producer =
      event != events_.end() ? *event : Event{0, 0, 0, pc, WaitCounterKind::X, 0};
  reporter_(context_, {producer, pc_, pc, false, 0});
}

void MemoryWaitScoreboard::check_atomic_xcnt() const {
  if (!reporter_)
    return;
  const Event *newest = nullptr;
  for (const auto &event : events_) {
    if (!event.protects_replay_source() || event.atomic_replay_source)
      continue;
    if (!newest || required_wait(event) < required_wait(*newest))
      newest = &event;
  }
  if (newest)
    reporter_(context_, {*newest, pc_, {RegClass::PC, 0, 1}, false, required_wait(*newest)});
}

void MemoryWaitScoreboard::xcnt_group(bool scalar) {
  if (xcnt_scalar_ != scalar && outstanding(WaitCounterKind::X))
    wait_ordered_xcnt(0);
  xcnt_scalar_ = scalar;
}

uint64_t MemoryWaitScoreboard::issue_xcnt(std::optional<WaitCounterKind> completion, bool scalar) {
  const auto sequence = issue(WaitCounterKind::X, scalar);
  if (!scalar && completion) {
    const auto i = static_cast<size_t>(*completion);
    const auto order = last_order_[i];
    translations_.push_back(
        {sequence, issued_[i], *completion, order, order == kUnordered ? 0 : orders_[order].issued,
         order == kUnordered || !last_capacity_eligible_[i] ? 0 : orders_[order].capacity_issued});
  }
  return sequence;
}

void MemoryWaitScoreboard::xcnt_ordered_write(RegisterRef reg, uint64_t lanes, uint8_t bytes) {
  if (xcnt_scalar_ || !pending_.pending(reg, true))
    return;
  reg = pending_.canonical(reg);
  uint64_t through = 0;
  for (const auto &event : events_)
    if (event.counter == WaitCounterKind::X && event.reg.cls == reg.cls && (event.lanes & lanes) &&
        (event.bytes & bytes) && reg.index < event.reg.index + event.reg.width &&
        event.reg.index < reg.index + reg.width)
      through = std::max(through, event.sequence);
  if (through)
    wait_ordered_xcnt(
        static_cast<uint32_t>(issued_[static_cast<size_t>(WaitCounterKind::X)] - through));
}

bool MemoryWaitScoreboard::issues_flat_counter(const Instruction &inst, Wavefront &wf,
                                               uint64_t requests, uint64_t returned_without_request,
                                               WaitCounterKind counter) const {
  const auto arch = wf.cu().arch();
  if (!waitcheck_detail::RuntimeWaitTarget::flat_issues_both_counters(arch))
    return false;
  // An active private-OOB load still writes a zero result. It has no byte
  // request, but its returned value needs a VM position and the generic FLAT
  // instruction occupies both legacy counter domains.
  if (requests || returned_without_request)
    return true;
  // GFX11 can skip FLAT once physical VM work drains, even while LDS remains
  // pending. A modeled VM event may therefore justify a VM position, but
  // cannot justify a DS position behind an older LDS result.
  if (counter == WaitCounterKind::Ds &&
      (arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5))
    return false;
  // A load whose destination is unallocated issues as if EXEC were zero.
  if (wf.memory_exec(inst) && !wf.memory_destination_out_of_range(inst))
    return false;
  // If the physical queue already drained and this FLAT skips, older events
  // on that same counter are complete. Otherwise it issues a new position.
  // This conditional proof survives the simulator's eager completion, which
  // can leave diagnostic events pending after physical counters reach zero.
  // BVH results have no implemented physical issue path and cannot gate FLAT.
  return flat_issues_with_empty_exec(
      arch, outstanding(WaitCounterKind::Load) || outstanding(WaitCounterKind::Store),
      outstanding(WaitCounterKind::Ds),
      arch == ROCJITSU_CODE_ARCH_CDNA5
          ? outstanding(WaitCounterKind::Async) || outstanding(WaitCounterKind::Tensor)
          : outstanding(WaitCounterKind::Sample));
}

void MemoryWaitScoreboard::before(const Instruction &inst, rj_code_arch_t arch, Wavefront *wf) {
  using namespace waitcheck_detail;
  if (wf && disabled_scalar_prefetch(inst, *wf))
    return;
  if (wf && (inst.flags() & CONDITIONAL_MEMORY_LANES)) {
    RegisterModifiers modifiers;
    inst.amdgpu_register_modifiers(modifiers);
    if (addr_calc::modern_ignores_buffer_resource(modifiers, *wf))
      return; // Uncertain admission cannot prove completion through counter capacity.
  }
  if (wf && pending_scalar_ && WaitCounterPolicy::relative_wait_operand(inst, arch))
    check_instruction(inst, *wf); // The selector is read before this wait can retire its producer.
  if (outstanding(WaitCounterKind::VmVsrc) &&
      RuntimeWaitTarget::releases_gfx10_scalar_sources(inst, arch))
    wait(WaitCounterKind::VmVsrc, 0);
  // Most kernels wait well before a queue is full. Avoid classifying the
  // incoming instruction twice unless some ordered class can force progress.
  const auto model = waitcnt_model(arch);
  const bool full =
      inst.is_memory_wait_producer() && model.succeeded() &&
      std::ranges::any_of(orders_, [&](const Order &order) {
        const auto maximum =
            WaitCounterPolicy::maximum_dependency_wait(model.value(), order.counter);
        return order.issued - order.retired + MemoryCounterObligation::MAX_COUNTER_INCREMENT >
               maximum + 1;
      });
  if (full) {
    const auto events = RuntimeWaitTarget::classify_events(inst, arch);
    if (events.succeeded())
      for (const auto &event : events.value()) {
        if (event.counter == WaitCounterKind::Exp && wf &&
            RuntimeWaitTarget::export_skips_with_empty_exec(inst, arch, wf->exec()))
          continue;
        if (event.counter == WaitCounterKind::X &&
            RuntimeWaitTarget::is_async_transfer_kind(event.kind))
          continue;
        if (event.kind == WaitEventKind::AsyncBarrier &&
            (!wf || (!wf->exec() && !wf->wait_counters().dscnt)))
          continue;
        // Generic FLAT reserves both counter positions before reading its
        // address on targets with dual issue. Capacity progress is therefore
        // available for source checks even when one result route is unused.
        if (inst.mnemonic().starts_with("flat_") &&
            (event.counter == WaitCounterKind::Load || event.counter == WaitCounterKind::Store ||
             event.counter == WaitCounterKind::Ds)) {
          if (!flat_issues_both_counter_domains(arch))
            continue;
          const bool active = wf && wf->memory_exec(inst) &&
                              !wf->memory_destination_out_of_range(inst);
          if (!active && (!wf || !issues_flat_counter(inst, *wf,
                                                     flat_lanes(inst, *wf, 0, 0).requests, 0,
                                                     event.counter)))
            continue;
        }
        if (event.counter == WaitCounterKind::VmVsrc || event.counter == WaitCounterKind::VaVdst ||
            event.counter == WaitCounterKind::Depctr)
          continue;
        const auto maximum = WaitCounterPolicy::maximum_dependency_wait(arch, event.counter);
        if (maximum.succeeded())
          backpressure(event.counter, maximum.value() + 1, issue_units(inst, event, arch));
      }
  }
  if (arch == ROCJITSU_CODE_ARCH_CDNA5 &&
      (outstanding(WaitCounterKind::X) || outstanding(WaitCounterKind::Async)) &&
      WaitcheckTarget::is_xcnt_drain(inst))
    wait(WaitCounterKind::X, 0);
  if (!inst.is_waitcnt() && !inst.has_embedded_memory_wait())
    return;
  const auto fields = inst.is_waitcnt() ? RegisterAccess::explicit_wait_fields(inst, arch, wf)
                                        : WaitcheckTarget::embedded_wait_fields(inst, arch);
  if (fields.succeeded() && fields.value())
    wait(*fields.value(), arch);
}

void MemoryWaitScoreboard::wait(const WaitFields &fields, rj_code_arch_t arch) {
  bool changed = false;
  if (const auto va = fields[static_cast<size_t>(WaitCounterKind::VaVdst)])
    changed |= apply_wait(WaitCounterKind::VaVdst, *va);
  const bool gfx10 = arch == ROCJITSU_CODE_ARCH_RDNA1;
  if (const auto source = fields[static_cast<size_t>(WaitCounterKind::VmVsrc)];
      source && (!gfx10 || *source == 0))
    changed |= apply_wait(WaitCounterKind::VmVsrc, *source);
  for (auto counter :
       {WaitCounterKind::Load, WaitCounterKind::Store, WaitCounterKind::Ds, WaitCounterKind::Km,
        WaitCounterKind::Exp, WaitCounterKind::Sample, WaitCounterKind::Bvh, WaitCounterKind::Async,
        WaitCounterKind::Tensor, WaitCounterKind::X})
    if (const auto value = fields[static_cast<size_t>(counter)]) {
      if (counter == WaitCounterKind::X && !*value)
        changed |= drain_async_replay();
      changed |= apply_wait(counter, *value);
    }
  if (changed)
    retire_completed();
}

std::unique_ptr<MemoryWaitScoreboard::DeferredWrite>
MemoryWaitScoreboard::defer_write(RegisterRef reg) const {
  auto write = std::make_unique<DeferredWrite>();
  if (!pending_.pending(reg, true))
    return write;
  reg = pending_.canonical(reg);
  for (const auto &event : events_) {
    if (reg.cls == event.reg.cls && reg.index < event.reg.index + event.reg.width &&
        event.reg.index < reg.index + reg.width)
      write->producers.push_back({event.counter, event.sequence, required_wait(event)});
  }
  return write;
}

void MemoryWaitScoreboard::complete_deferred_write(DeferredWrite &write, uint64_t lanes) {
  auto &result = write.result;
  result.lanes &= lanes;
  if (result.lanes && !write.producers.empty())
    check_pending(pending_.canonical(result.reg), result.lanes, result.bytes, true,
                  {result.order, result.writeback}, &write);
  add(result, write.ordered_native_result);
}

void MemoryWaitScoreboard::access_pending(RegisterRef reg, uint64_t lanes, uint8_t bytes,
                                          bool write, WriteOrder ordered_write_order) {
  check_pending(reg, lanes, bytes, write, ordered_write_order, nullptr);
}

void MemoryWaitScoreboard::check_pending(RegisterRef reg, uint64_t lanes, uint8_t bytes, bool write,
                                         WriteOrder ordered_write_order,
                                         const DeferredWrite *deferred) {
  for (auto &event : events_) {
    const uint64_t pending_lanes =
        !write && event.short_trans && !event.valu_source
            ? event.lanes & ~event.trans_use_superseded_lanes
            : event.lanes;
    // Sharing a counter does not prove ordered writeback. The old result and
    // incoming write must belong to the same FIFO class, even in a mixed queue.
    if (event.reported ||
        (checking_valu_ && event.counter == WaitCounterKind::VaVdst &&
         (!event.short_trans || checking_trans_ || event.trans_valu_ready)) ||
        (!write && event.protects_source()) ||
        (write && ordered_write_order.fifo != kUnordered && !(event.writeback_lanes & lanes)) ||
        (write && ordered_write_order.fifo != kUnordered &&
         event.order == ordered_write_order.fifo &&
         ordered_vmem_writebacks(event.writeback, ordered_write_order.writeback) &&
         event.ordered_overwrite) ||
        event.reg.cls != reg.cls || !(pending_lanes & lanes) || !(event.bytes & bytes) ||
        reg.index >= event.reg.index + event.reg.width || event.reg.index >= reg.index + reg.width)
      continue;
    uint32_t threshold;
    if (deferred) {
      const auto it = std::ranges::find_if(deferred->producers, [&](const auto &producer) {
        return producer.counter == event.counter && producer.sequence == event.sequence;
      });
      if (it == deferred->producers.end())
        continue;
      threshold = it->required;
    } else {
      threshold = required_wait(event);
    }
    event.reported = true;
    if (reporter_) {
      const auto first = std::max(reg.index, event.reg.index);
      const auto end = std::min(reg.index + reg.width, event.reg.index + event.reg.width);
      const RegisterRef overlap{reg.cls, first, static_cast<uint8_t>(end - first)};
      auto producer = event;
      producer.lanes = pending_lanes & lanes;
      reporter_(context_, {producer, deferred ? deferred->result.pc : pc_, overlap, write,
                           threshold});
    }
  }
  const auto size = events_.size();
  std::erase_if(events_, [&](const Event &event) {
    // Retain reported short TRANS records until an independent VALU or VA
    // wait, so their full source/destination set remains available as a group.
    // Keep the EXEC source as evidence of this possible translation for a
    // later prefetch even after reporting every register overwrite once.
    if (!event.reported || event.short_trans ||
        (event.async_replay_source && event.reg.cls == RegClass::EXEC))
      return false;
    clear_destination(event);
    return true;
  });
  if (size != events_.size())
    rebuild_mask();
}
} // namespace rocjitsu::amdgpu
