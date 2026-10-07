// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/memory_wait_scoreboard.h"
#include "rocjitsu/isa/arch/amdgpu/cdna5/addr_calc.h"
#include "rocjitsu/isa/arch/amdgpu/shared/accvgpr_layout.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_buffer.h"
#include "rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h"
#include "rocjitsu/isa/arch/amdgpu/shared/flat_address.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_read.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_resolve.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/hwreg.h"

#include <algorithm>
#include <bit>

namespace rocjitsu::amdgpu {

std::optional<uint32_t> RegisterAccess::scalar_control_value(uint32_t selector) const {
  const auto &wf = wavefront();
  const auto m0 = scalar_m0_selector(wf.cu().arch());
  if (!can_resolve_src_scalar(selector, m0))
    return std::nullopt;
  // Preserve invalid-backing rejection for address planning. Constants and
  // computed aliases have no storage range and use the shared value decoder.
  if (selector < 128 && !resolve_scalar_register_range(wf, selector, 1))
    return std::nullopt;
  return resolve_src_scalar_value<[](const Wavefront &wave, uint32_t reg) {
    return RegisterAccess(wave).cu_->read_sgpr_storage(wave.sgpr_alloc().base + reg);
  },
                                  [](const Wavefront &wave, uint32_t reg) {
                                    return wave.ttmp(reg);
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
  if (auto reg = op.to_register_ref())
    return reg;
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

bool MemoryWaitScoreboard::result_is_written(const Instruction &inst, Wavefront &wf) {
  if (inst.mnemonic() == "lds_direct_load" || inst.mnemonic() == "ds_direct_load")
    return valid_lds_direct_operand(wf.m0());
  if (inst.mnemonic() == "s_barrier_signal_isfirst") {
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
  const auto *issue = inst.amdgpu_memory_issue_info();
  uint64_t lanes = issue && !issue->exec_masked
                       ? (wf.wf_size() == 64 ? ~uint64_t{0} : uint64_t{0xffffffff})
                       : wf.exec();
  if (!(inst.flags() & CONDITIONAL_MEMORY_LANES))
    return lanes;
  RegisterModifiers modifiers;
  inst.amdgpu_register_modifiers(modifiers);
  if (modifiers.exec_all_if_nonzero)
    lanes = wave32_exec_all_if_nonzero(lanes);
  if (modifiers.exec_whole_quads)
    lanes = result_is_written(inst, wf) ? whole_active_quads(lanes) : 0;
  if (lanes && wf.cu().arch() == ROCJITSU_CODE_ARCH_CDNA5 && modifiers.buffer_resource &&
      !modifiers.scalar_buffer_words) {
    const auto selector = modifiers.buffer_resource->encoding_value();
    if (addr_calc::buffer_resource_range_is_backed(wf, selector))
      if (const auto word3 = RegisterAccess(wf).scalar_control_value(selector + 3))
        if (cdna5::decode_buffer_resource(0, 0, 0, *word3).type != 0)
          return 0;
  }
  return lanes;
}

MemoryWaitScoreboard::FlatLanes MemoryWaitScoreboard::flat_lanes(const Instruction &inst,
                                                                 Wavefront &wf,
                                                                 uint64_t shared_base,
                                                                 uint64_t shared_limit) {
  FlatLanes result{wf.exec(), 0};
  RegisterModifiers modifiers;
  inst.amdgpu_register_modifiers(modifiers);
  assert(modifiers.flat_address);
  const RegisterAccess registers(wf);
  uint64_t scalar = 0;
  if (const auto *operand = modifiers.flat_scalar_address) {
    const unsigned selector = operand->encoding_value();
    if (!resolve_scalar_register_range(wf, selector, 2))
      return {0, 0};
    const auto lo = registers.scalar_control_value(selector);
    const auto hi = registers.scalar_control_value(selector + 1);
    if (!lo || !hi)
      return {0, 0};
    scalar = *lo | (uint64_t{*hi} << 32);
  }
  if (!shared_base || !result.requests)
    return result;
  const auto arch = wf.cu().arch();
  const auto layout = arch == ROCJITSU_CODE_ARCH_CDNA5 ? FlatPrivateLayout::EncodedLane
                      : arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
                              arch == ROCJITSU_CODE_ARCH_RDNA4
                          ? FlatPrivateLayout::Linear
                      : arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2
                          ? FlatPrivateLayout::None
                          : FlatPrivateLayout::Interleaved;
  const auto address = registers.source_register(*modifiers.flat_address);
  uint64_t remaining = result.requests;
  while (remaining) {
    const unsigned lane = std::countr_zero(remaining);
    remaining &= remaining - 1;
    const uint64_t vector =
        address ? registers.vector_control_value(*address, lane).value_or(0) : 0;
    const uint64_t value =
        scalar +
        flat_vector_offset(vector,
                           modifiers.flat_scalar_address && arch == ROCJITSU_CODE_ARCH_CDNA5,
                           modifiers.flat_scale) +
        modifiers.flat_offset;
    const auto translated = translate_flat_address(wf, value, lane, layout);
    if (!translated.swizzled && translated.value >= shared_base && translated.value <= shared_limit)
      result.shared |= uint64_t{1} << lane;
  }
  return result;
}

void MemoryWaitScoreboard::check_instruction(const Instruction &inst, Wavefront &wf) {
  if (empty())
    return;
  // A clear register byte proves independence for every lane and byte mask.
  // Only a possible overlap needs the precise instruction-specific footprints.
  if (!pending_scalar_ && (inst.flags() & DIRECT_REGISTER_ACCESSES) && !wf.gpr_idx_en()) {
    auto pending = [&](const Operand &op) {
      const auto reg = op.decoded_vgpr();
      if (!reg)
        return op.is_fieldless() && op.is_vgpr();
      const unsigned index =
          reg->index +
          (reg->index < ACC_VGPR_OFFSET ? wf.vgpr_msb_for_role(op.vgpr_msb_role()) << 8 : 0);
      if (index + reg->width > wf.vgpr_alloc().count || index + reg->width > REGISTER_SET_MAX_VGPRS)
        return true;
      return pending_.pending_vgpr(static_cast<uint16_t>(index), reg->width);
    };
    bool hit = false;
    for (int i = 0; i < inst.num_src_operands(); ++i)
      hit |= pending(*inst.src_operand(i));
    for (int i = 0; i < inst.num_dst_operands(); ++i)
      hit |= pending(*inst.dst_operand(i));
    if (!hit)
      return;
  }
  if (!pending_scalar_ && inst.mnemonic().starts_with("s_"))
    return;
  check_instruction_pending(inst, wf);
}

void MemoryWaitScoreboard::check_instruction_pending(const Instruction &inst, Wavefront &wf) {
  const auto name = inst.mnemonic();
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
  uint64_t lanes = ignores_exec ? all_lanes : wf.exec();
  if (vector && !ignores_exec)
    access({RegClass::EXEC, 0, static_cast<uint8_t>(wf.wf_size() / 32)}, ~uint64_t{0}, 0xf, false);
  RegisterModifiers modifiers;
  inst.amdgpu_register_modifiers(modifiers);
  const RegisterAccess registers(wf);
  if (modifiers.exec_all_if_nonzero)
    lanes = wave32_exec_all_if_nonzero(lanes);
  if (modifiers.exec_whole_quads)
    lanes = result_is_written(inst, wf) ? whole_active_quads(lanes) : 0;
  if (const auto *resource = modifiers.buffer_resource) {
    for (const auto reg :
         registers.buffer_resource_registers(*resource, modifiers.scalar_buffer_words))
      if (reg)
        access(*reg, ~uint64_t{0}, 0xf, false);
  }
  // FLAT's result ordering depends on its planned memory domains. The issuer
  // checks those results after checking address sources, still before execution.
  if (inst.is_memory_wait_producer() && !name.starts_with("flat_"))
    check_memory_result(inst, wf,
                        (inst.flags() & CONDITIONAL_MEMORY_LANES) ? result_lanes(inst, wf) : lanes,
                        modifiers);
  uint64_t src0_lanes = lanes;
  uint64_t destination_lanes = lanes;
  if (modifiers.dpp || modifiers.dpp8) {
    const auto plan =
        modifiers.dpp
            ? dpp::make_dpp_plan(wf.wf_size(), modifiers.control, modifiers.row_mask,
                                 modifiers.bank_mask, modifiers.bound_ctrl, modifiers.fi, wf.exec(),
                                 modifiers.inactive_uses_bound_ctrl)
            : dpp::make_dpp8_plan(wf.wf_size(), modifiers.control, modifiers.fi, wf.exec());
    src0_lanes = dpp::dpp_physical_source_mask(plan, lanes, wf.wf_size());
    destination_lanes &= plan.row_bank_mask & plan.source_write_mask;
  }
  if (modifiers.ds_permutation != DsPermutation::None) {
    std::optional<RegisterRef> address;
    if (modifiers.src1) {
      address = registers.source_register(*modifiers.src1);
      if (address)
        access(*address, lanes, 0xf, false);
    }
    const auto data = registers.source_register(*modifiers.src0);
    if (!data || !pending_.pending(*data))
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
  if (!pending_scalar_ && inst.is_memory_op() && name.starts_with("ds_")) {
    for (int i = 0; i < inst.num_src_operands(); ++i)
      if (const auto *op = inst.src_operand(i); op && op->is_vgpr())
        if (auto reg = registers.source_register(*op, op == modifiers.wordwise_source0 ||
                                                          op == modifiers.wordwise_source1))
          access(*reg, lanes, op->register_byte_mask(), false);
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
  auto resolve = [&](const Operand &op, bool write) -> std::optional<RegisterRef> {
    // MMA execution applies GPR_IDX only to ordinary VGPRs, never its
    // separate accumulator bank. Keep that bank's decoded identity intact.
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
    if (wave_mask && op.has_register_selector() && !op.const_value()) {
      const auto range = resolve_scalar_register_range(wf, op.encoding_value(), wf.wf_size() / 32);
      return range ? std::optional(range->register_ref()) : std::nullopt;
    }
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
      if (const auto reg = registers.source_register(*operand); reg && reg->cls == RegClass::VGPR)
        return registers.vector_control_value(*reg, lane).value_or(0);
      return registers.scalar_control_value(operand->encoding_value()).value_or(0);
    };
    const uint32_t low = variable ? 0 : control(selector, 0);
    const uint32_t high = control(group, 0);
    uint64_t read_lanes = 0, write_lanes = 0;
    uint64_t remaining = lanes;
    while (remaining) {
      const unsigned lane = std::countr_zero(remaining);
      remaining &= remaining - 1;
      const uint32_t selection = variable ? control(selector, lane)
                                 : row16 ? (((lane & 15) < 8 ? low : high) >> ((lane & 7) * 4)) & 15
                                         : low;
      const unsigned source = valu_permutation_source(kind, lane, selection, high, wf.wf_size());
      if (source >= wf.wf_size()) {
        if (!row16 && kind != ValuPermutation::Perm64)
          write_lanes |= uint64_t{1} << lane;
        continue;
      }
      if (row16 && !modifiers.fi && !(lanes & (uint64_t{1} << source))) {
        if (modifiers.bound_ctrl)
          write_lanes |= uint64_t{1} << lane;
        continue;
      }
      read_lanes |= uint64_t{1} << source;
      write_lanes |= uint64_t{1} << lane;
    }
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
    auto check_relative = [&](const Operand &operand, bool relative, bool write) {
      if (!relative) {
        if (auto reg = resolve(operand, write))
          access(*reg, ~uint64_t{0}, 0xf, write);
        return;
      }
      const unsigned width = std::max(1, operand.size_bits() / 32);
      const auto selector =
          relative_scalar_selector(operand.encoding_value(), width, wf.m0(), write, split);
      if (auto range = resolve_scalar_register_range(wf, selector, width)) {
        if (auto reg = range->register_ref())
          access(*reg, ~uint64_t{0}, 0xf, write);
      } else if (!write) {
        // Indexed sources can select scalar condition aliases or constants.
        if (selector == 251 || selector == 252)
          access({selector == 251 ? RegClass::VCC : RegClass::EXEC, 0,
                  static_cast<uint8_t>(wf.wf_size() / 32)},
                 ~uint64_t{0}, 0xf, false);
        else if (selector == 253)
          access({RegClass::SCC, 0, 1}, ~uint64_t{0}, 0xf, false);
      }
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
      const auto index =
          relative_vgpr_index(base, offset, operand.vgpr_count(), wf.vgpr_alloc().count);
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
  auto check_operand = [&](const Operand *op, bool write) {
    const bool mix_preservation = !write && op == inst.dst_operand(0) &&
                                  (name.find("mixlo_") != std::string_view::npos ||
                                   name.find("mixhi_") != std::string_view::npos);
    if (mix_preservation || !op || op == hwreg || op == modifiers.buffer_resource ||
        (!pending_scalar_ && !op->is_vgpr()))
      return;
    const auto reg = resolve(*op, write);
    if (!reg || reg->cls == RegClass::ACC_VGPR)
      return;
    const uint64_t accessed_lanes = reg->cls != RegClass::VGPR ? ~uint64_t{0}
                                    : write                    ? destination_lanes
                                    : op == modifiers.src0     ? src0_lanes
                                                               : lanes;
    if (vector && !lanes && !write && !inst.is_memory_op())
      return;
    uint8_t bytes = op->register_byte_mask();
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
      access(*reg, accessed_lanes, bytes, write);
    }
  };
  for (int i = 0; i < inst.num_src_operands(); ++i)
    check_operand(inst.src_operand(i), false);
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
  RegisterSet implicit;
  if (!preserves_destination)
    inst.implicit_uses(implicit);
  implicit.for_each([&](RegisterRef reg) {
    // Operand-derived reads preserve their bank roles above.
    if (reg.cls == RegClass::VGPR && !implicit_operands.empty())
      return;
    if (vector && !lanes)
      return;
    access(reg, reg.cls == RegClass::VGPR ? lanes : ~uint64_t{0}, 0xf, false);
  });
  if (!inst.is_memory_wait_producer()) {
    for (int i = 0; i < inst.num_dst_operands(); ++i)
      check_operand(inst.dst_operand(i), true);
    implicit = RegisterSet{};
    inst.implicit_defs(implicit);
    implicit.for_each([&](RegisterRef reg) {
      access(reg, reg.cls == RegClass::VGPR ? lanes : ~uint64_t{0}, 0xf, true);
    });
  }
}

void MemoryWaitScoreboard::check_memory_result(const Instruction &inst, Wavefront &wf,
                                               uint64_t vector_lanes,
                                               const RegisterModifiers &modifiers) {
  using namespace waitcheck_detail;
  const RegisterAccess registers(wf);
  if (!result_is_written(inst, wf))
    return;
  std::array<RegisterRef, 3> results{};
  unsigned count = 0;
  bool pending = false;
  for (int i = 0; i < inst.num_dst_operands(); ++i) {
    const auto &operand = *inst.dst_operand(i);
    auto reg = registers.destination_register(operand);
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
  const auto classified = WaitcheckTarget::classify_events(inst, wf.cu().arch());
  if (classified.failed())
    return;
  const auto xevent =
      std::ranges::find(classified.value(), WaitCounterKind::X, &ClassifiedEvent::counter);
  const bool replay = outstanding(WaitCounterKind::X) != 0 && xevent != classified.value().end();
  if (replay)
    xcnt_group(xevent->kind == WaitEventKind::Smem);
  const auto defs = std::ranges::find(classified.value(), TrackedRegisterSource::Defs,
                                      &ClassifiedEvent::registers);
  const auto model = waitcnt_model(wf.cu().arch());
  const uint16_t order = model.succeeded() && uses_legacy_waitcnt(model.value()) &&
                                 defs != classified.value().end() &&
                                 defs->counter == WaitCounterKind::Load
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
    if (defs == classified.value().end())
      continue;
    auto check = [&](RegisterRef range, uint8_t bytes) {
      if (vector_replay)
        xcnt_ordered_write(range, vector_lanes, bytes);
      access(range, vector_lanes, bytes, true, order);
    };
    const uint8_t bytes = wf.cu().sram_ecc() ? 0xf : modifiers.memory_result_bytes;
    const uint8_t tail = wf.cu().sram_ecc() ? 0xf : modifiers.memory_result_last_bytes;
    if (bytes == tail) {
      check(reg, bytes);
    } else {
      const RegisterRef last{reg.cls, static_cast<uint16_t>(reg.index + reg.width - 1), 1};
      --reg.width;
      if (reg.width)
        check(reg, bytes);
      check(last, tail);
    }
  }
}

void MemoryWaitScoreboard::clear() {
  events_.clear();
  pending_scalar_ = false;
  may_overlap_ = false;
  pending_.reset();
  issued_.fill(0);
  retired_.fill(0);
  unordered_.fill(false);
  ordering_kinds_.fill(0);
  orders_.clear();
  last_order_.fill(kUnordered);
  xcnt_scalar_ = false;
  translations_.clear();
}

uint64_t MemoryWaitScoreboard::issue(WaitCounterKind counter, bool unordered,
                                     uint32_t ordering_kind, uint32_t units,
                                     bool backpressure_ordered) {
  auto i = static_cast<size_t>(counter);
  ordering_kinds_[i] |= ordering_kind;
  unordered_[i] |= unordered || std::popcount(ordering_kinds_[i]) > 1;
  issued_[i] += units;
  last_order_[i] = kUnordered;
  if (!unordered && backpressure_ordered) {
    const auto it = std::ranges::find_if(orders_, [&](const Order &order) {
      return order.counter == counter && order.kind == ordering_kind;
    });
    const auto slot = static_cast<uint16_t>(it - orders_.begin());
    if (it == orders_.end())
      orders_.push_back({counter, ordering_kind});
    orders_[slot].issued += units;
    last_order_[i] = slot;
  }
  return issued_[i];
}

namespace {
std::optional<uint32_t> completion_order_kind(const waitcheck_detail::ClassifiedEvent &event,
                                              rj_code_arch_t arch) {
  using namespace waitcheck_detail;
  const auto model = waitcnt_model(arch).value();
  const bool legacy_flat =
      model == WaitcntModel::LegacyNoVscnt &&
      (event.kind == WaitEventKind::FlatLoad || event.kind == WaitEventKind::FlatStore);
  if (legacy_flat || event.kind == WaitEventKind::Smem || event.kind == WaitEventKind::Gds ||
      event.kind == WaitEventKind::Export || event.kind == WaitEventKind::SqMessage ||
      event.kind == WaitEventKind::SccWrite || event.kind == WaitEventKind::GlobalInv ||
      event.kind == WaitEventKind::Unknown)
    return std::nullopt;
  uint32_t kind = 0;
  if (!(model == WaitcntModel::LegacyNoVscnt && event.counter == WaitCounterKind::Load))
    if (auto normalized =
            WaitcheckTarget::normalized_hardware_event_kind(event.counter, event.kind, model))
      kind = uint32_t{1} << static_cast<unsigned>(*normalized);
  // ASYNC loads and stores share a counter but return done out of order with
  // respect to each other (CDNA5 ISA 10.8). Barrier arrive orders with loads.
  if (event.counter == WaitCounterKind::Async)
    kind = uint32_t{1} << static_cast<unsigned>(event.kind == WaitEventKind::AsyncLdsStore
                                                    ? WaitEventKind::AsyncLdsStore
                                                    : WaitEventKind::AsyncLdsLoad);
  // Older RDNA samples and BVH operations share VMcnt, not a completion FIFO.
  if (model == WaitcntModel::LegacyVscnt && event.counter == WaitCounterKind::Load &&
      (event.kind == WaitEventKind::Sample || event.kind == WaitEventKind::Bvh))
    kind = uint32_t{1} << static_cast<unsigned>(event.kind);
  return kind;
}
} // namespace

uint64_t MemoryWaitScoreboard::issue(const waitcheck_detail::ClassifiedEvent &event,
                                     rj_code_arch_t arch, uint32_t units) {
  const auto kind = completion_order_kind(event, arch);
  return issue(event.counter, !kind, kind.value_or(0), units);
}

uint32_t MemoryWaitScoreboard::issue_units(const Instruction &inst,
                                           const waitcheck_detail::ClassifiedEvent &event,
                                           rj_code_arch_t arch) {
  using namespace waitcheck_detail;
  const auto model = waitcnt_model(arch).value();
  const auto counter_kind = [model](WaitCounterType counter) {
    switch (counter) {
    case WaitCounterType::VMCNT:
    case WaitCounterType::LOADCNT:
      return WaitCounterKind::Load;
    case WaitCounterType::VSCNT:
    case WaitCounterType::STORECNT:
      return vmem_store_wait_counter(model);
    case WaitCounterType::LGKMCNT:
    case WaitCounterType::DSCNT:
      return WaitCounterKind::Ds;
    case WaitCounterType::KMCNT:
      return smem_wait_counter(model);
    case WaitCounterType::EXPCNT:
      return WaitCounterKind::Exp;
    case WaitCounterType::ASYNCCNT:
      return WaitCounterKind::Async;
    case WaitCounterType::TENSORCNT:
      return WaitCounterKind::Tensor;
    }
    return WaitCounterKind::Count;
  };
  if (const auto *info = inst.amdgpu_memory_issue_info())
    for (const auto &obligation : info->counter_obligations())
      if (counter_kind(obligation.wait_counter_type()) == event.counter)
        return obligation.counter_increment();
  // Inline message returns are outside the memory-pipeline metadata: one
  // completion acknowledges the send, the other the returned register value.
  if (event.kind == WaitEventKind::SqMessage && inst.mnemonic().starts_with("s_sendmsg_rtn_"))
    return 2;
  return 1;
}

uint16_t MemoryWaitScoreboard::ordered_write_order(const waitcheck_detail::ClassifiedEvent &event,
                                                   rj_code_arch_t arch) const {
  const auto kind = completion_order_kind(event, arch);
  if (!kind)
    return kUnordered;
  const auto it = std::ranges::find_if(orders_, [&](const Order &order) {
    return order.counter == event.counter && order.kind == *kind;
  });
  return it == orders_.end() ? kUnordered : static_cast<uint16_t>(it - orders_.begin());
}

bool MemoryWaitScoreboard::completed(WaitCounterKind counter, uint64_t sequence, uint16_t order,
                                     uint64_t order_sequence) const {
  return sequence <= retired_[static_cast<size_t>(counter)] ||
         (order != kUnordered && order_sequence <= orders_[order].retired);
}

void MemoryWaitScoreboard::backpressure(WaitCounterKind counter, uint32_t capacity,
                                        uint32_t incoming_units) {
  assert(incoming_units != 0 && incoming_units <= capacity);
  const auto remaining = capacity - incoming_units;
  bool changed = false;
  for (auto &order : orders_) {
    if (order.counter != counter || order.issued <= remaining)
      continue;
    // Even an unordered incoming operation must reserve all its counter units.
    // At most remaining units from any old ordered class can still be pending
    // when the incoming instruction is admitted to read its operands.
    const auto through = order.issued - remaining;
    changed |= through > order.retired;
    order.retired = std::max(order.retired, through);
  }
  if (changed)
    retire_completed();
}

void MemoryWaitScoreboard::stamp_order(Event &event) const {
  const auto counter = static_cast<size_t>(event.counter);
  if (event.sequence == issued_[counter] && last_order_[counter] != kUnordered) {
    event.order = last_order_[counter];
    event.order_sequence = orders_[event.order].issued;
  }
}

uint32_t MemoryWaitScoreboard::required_wait(const Event &event) const {
  const auto i = static_cast<size_t>(event.counter);
  const auto required = !unordered_[i] ? issued_[i] - event.sequence
                        : event.order != kUnordered
                            ? orders_[event.order].issued - event.order_sequence
                            : 0;
  return static_cast<uint32_t>(std::min<uint64_t>(required, UINT32_MAX));
}

void MemoryWaitScoreboard::add(Event event) {
  if (!event.lanes || !event.bytes || !event.reg.width)
    return;
  stamp_order(event);
  // A newer result covering the entire old destination is the stronger
  // dependency within the same FIFO class. Discard the superseded record rather
  // than growing with a loop that repeatedly loads an unused destination.
  const bool overlap =
      pending_.mark(event.reg, event.counter == WaitCounterKind::X ? MemoryWaitShadow::kReplaySource
                                                                   : MemoryWaitShadow::kResult);
  may_overlap_ |= overlap;
  if (overlap)
    std::erase_if(events_, [&](const Event &old) {
      const bool ordered = event.order != kUnordered && old.order == event.order &&
                           old.order_sequence <= event.order_sequence;
      // Scalar replay sources have a different proof: every sufficient wait
      // drains their entire X group, so its newest covering source is enough.
      const bool scalar_replay =
          xcnt_scalar_ && old.counter == WaitCounterKind::X && event.counter == WaitCounterKind::X;
      return (ordered || scalar_replay) && old.reg.cls == event.reg.cls &&
             event.reg.index <= old.reg.index &&
             event.reg.index + event.reg.width >= old.reg.index + old.reg.width &&
             (event.lanes & old.lanes) == old.lanes && (event.bytes & old.bytes) == old.bytes;
    });
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
    pending_.mark(event.reg, event.counter == WaitCounterKind::X ? MemoryWaitShadow::kReplaySource
                                                                 : MemoryWaitShadow::kResult);
}

void MemoryWaitScoreboard::wait(WaitCounterKind counter, uint32_t threshold) {
  if (apply_wait(counter, threshold))
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
  }
  if (counter == WaitCounterKind::X) {
    std::erase_if(translations_, [&](const auto &entry) { return entry.sequence <= through; });
  } else if (xcnt_scalar_ && counter == WaitCounterKind::Km && !threshold) {
    changed |= apply_wait(WaitCounterKind::X, 0);
  }
  return changed;
}

void MemoryWaitScoreboard::retire_completed() {
  const auto size = events_.size();
  std::erase_if(events_, [&](const Event &event) {
    const bool retire = completed(event.counter, event.sequence, event.order, event.order_sequence);
    if (retire)
      clear_destination(event);
    return retire;
  });
  if (size != events_.size())
    rebuild_mask();
  if (!xcnt_scalar_ && outstanding(WaitCounterKind::X)) {
    // Completion proves translation. Map the waited counter position back
    // into the ordered X queue; counts from mixed families are not X ages.
    uint64_t translated = 0;
    for (const auto &entry : translations_)
      if (completed(entry.completion, entry.completion_sequence, entry.order, entry.order_sequence))
        translated = std::max(translated, entry.sequence);
    if (translated > retired_[static_cast<size_t>(WaitCounterKind::X)])
      wait(WaitCounterKind::X,
           static_cast<uint32_t>(issued_[static_cast<size_t>(WaitCounterKind::X)] - translated));
  }
}

void MemoryWaitScoreboard::xcnt_group(bool scalar) {
  if (xcnt_scalar_ != scalar && outstanding(WaitCounterKind::X))
    wait(WaitCounterKind::X, 0);
  xcnt_scalar_ = scalar;
}

uint64_t MemoryWaitScoreboard::issue_xcnt(std::optional<WaitCounterKind> completion, bool scalar) {
  const auto sequence = issue(WaitCounterKind::X, scalar);
  if (!scalar && completion) {
    const auto i = static_cast<size_t>(*completion);
    const auto order = last_order_[i];
    translations_.push_back({sequence, issued_[i], *completion, order,
                             order == kUnordered ? 0 : orders_[order].issued});
  }
  return sequence;
}

void MemoryWaitScoreboard::xcnt_ordered_write(RegisterRef reg, uint64_t lanes, uint8_t bytes) {
  if (xcnt_scalar_ || !pending_.pending(reg, true))
    return;
  uint64_t through = 0;
  for (const auto &event : events_)
    if (event.counter == WaitCounterKind::X && event.reg.cls == reg.cls && (event.lanes & lanes) &&
        (event.bytes & bytes) && reg.index < event.reg.index + event.reg.width &&
        event.reg.index < reg.index + reg.width)
      through = std::max(through, event.sequence);
  if (through)
    wait(WaitCounterKind::X,
         static_cast<uint32_t>(issued_[static_cast<size_t>(WaitCounterKind::X)] - through));
}

void MemoryWaitScoreboard::before(const Instruction &inst, rj_code_arch_t arch) {
  using namespace waitcheck_detail;
  // Most kernels wait well before a queue is full. Avoid classifying the
  // incoming instruction twice unless some ordered class can force progress.
  const auto model = waitcnt_model(arch);
  const bool full =
      inst.is_memory_wait_producer() && model.succeeded() &&
      std::ranges::any_of(orders_, [&](const Order &order) {
        const auto maximum = WaitcheckTarget::maximum_dependency_wait(model.value(), order.counter);
        return order.issued - order.retired + MemoryCounterObligation::MAX_COUNTER_INCREMENT >
               maximum + 1;
      });
  if (full) {
    const auto events = WaitcheckTarget::classify_events(inst, arch);
    if (events.succeeded())
      for (const auto &event : events.value()) {
        // FLAT may not use either memory domain. Its resolved requests apply
        // the corresponding capacity constraint after address calculation.
        if (inst.mnemonic().starts_with("flat_") &&
            (event.counter == WaitCounterKind::Load || event.counter == WaitCounterKind::Store ||
             event.counter == WaitCounterKind::Ds))
          continue;
        if (event.counter == WaitCounterKind::VmVsrc || event.counter == WaitCounterKind::VaVdst ||
            event.counter == WaitCounterKind::Depctr)
          continue;
        const auto maximum = WaitcheckTarget::maximum_dependency_wait(arch, event.counter);
        if (maximum.succeeded())
          backpressure(event.counter, maximum.value() + 1, issue_units(inst, event, arch));
      }
  }
  if (arch == ROCJITSU_CODE_ARCH_CDNA5 && outstanding(WaitCounterKind::X) &&
      WaitcheckTarget::is_xcnt_drain(inst))
    wait(WaitCounterKind::X, 0);
  if (!inst.is_waitcnt() && !inst.has_embedded_memory_wait())
    return;
  const auto fields = inst.is_waitcnt() ? WaitcheckTarget::explicit_wait_fields(inst, arch)
                                        : WaitcheckTarget::embedded_wait_fields(inst, arch);
  if (fields.failed() || !fields.value())
    return;
  bool changed = false;
  for (auto counter :
       {WaitCounterKind::Load, WaitCounterKind::Store, WaitCounterKind::Ds, WaitCounterKind::Km,
        WaitCounterKind::Exp, WaitCounterKind::Sample, WaitCounterKind::Bvh, WaitCounterKind::Async,
        WaitCounterKind::Tensor, WaitCounterKind::X})
    if (const auto value = (*fields.value())[static_cast<size_t>(counter)])
      changed |= apply_wait(counter, *value);
  if (changed)
    retire_completed();
}

void MemoryWaitScoreboard::access_pending(RegisterRef reg, uint64_t lanes, uint8_t bytes,
                                          bool write, uint16_t ordered_write_order) {
  for (auto &event : events_) {
    // Sharing a counter does not prove ordered writeback. The old result and
    // incoming write must belong to the same FIFO class, even in a mixed queue.
    if (event.reported || (!write && event.counter == WaitCounterKind::X) ||
        (write && ordered_write_order != kUnordered && event.order == ordered_write_order) ||
        event.reg.cls != reg.cls || !(event.lanes & lanes) || !(event.bytes & bytes) ||
        reg.index >= event.reg.index + event.reg.width || event.reg.index >= reg.index + reg.width)
      continue;
    event.reported = true;
    if (reporter_) {
      const auto first = std::max(reg.index, event.reg.index);
      const auto end = std::min(reg.index + reg.width, event.reg.index + event.reg.width);
      const RegisterRef overlap{reg.cls, first, static_cast<uint8_t>(end - first)};
      reporter_(context_, {event, pc_, overlap, write, required_wait(event)});
    }
  }
  const auto size = events_.size();
  std::erase_if(events_, [&](const Event &event) {
    if (event.reported)
      clear_destination(event);
    return event.reported;
  });
  if (size != events_.size())
    rebuild_mask();
}
} // namespace rocjitsu::amdgpu
