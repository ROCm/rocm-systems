// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "rocjitsu/code/patch/consan/consan_tensor_access.h"

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/consan/consan_access_target.h"
#include "rocjitsu/code/patch/consan/consan_dispatch_identity_source.h"
#include "rocjitsu/code/patch/consan/consan_internal.h"
#include "rocjitsu/code/patch/consan/consan_native_abi.h"
#include "rocjitsu/code/patch/consan/consan_probe_planning.h"
#include "rocjitsu/code/patch/instruction_sequence.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"

#include <algorithm>

namespace rocjitsu::consan::detail {

std::vector<ProgramContainerId> tensor_execution_owner_kernels(const ProgramInventory &inventory) {
  std::vector<ProgramContainerId> result;
  for (const ProgramSite &site : inventory.access_sites()) {
    if (site.origin != AccessOrigin::TensorLds)
      continue;
    const auto owners = inventory.execution_owner_kernels(site);
    result.insert(result.end(), owners.begin(), owners.end());
  }
  std::ranges::sort(result);
  result.erase(std::ranges::unique(result).begin(), result.end());
  return result;
}

bool site_has_tensor_owner(const ProgramInventory &inventory, const ProgramSite &site,
                           std::span<const ProgramContainerId> tensor_owners) {
  const auto owners = inventory.execution_owner_kernels(site);
  return std::ranges::any_of(owners, [&](ProgramContainerId owner) {
    return std::ranges::binary_search(tensor_owners, owner);
  });
}

bool tensor_identity_sources_are_wave_uniform(const DispatchIdentity &dispatch,
                                              const WorkgroupSources &workgroup,
                                              bool full_wave_private_initialized) {
  const auto uniform_source = [&](const WorkgroupSource &source) {
    return source.is_well_formed() && !source.vector_src &&
           (!source.private_offset || full_wave_private_initialized);
  };
  return dispatch.is_well_formed() && (!dispatch.private_offset || full_wave_private_initialized) &&
         (workgroup.x.scalar_src ||
          (full_wave_private_initialized && workgroup.x.private_offset)) &&
         uniform_source(workgroup.x) && uniform_source(workgroup.y) &&
         uniform_source(workgroup.z) && uniform_source(workgroup.cluster_workgroup_id);
}

std::optional<VgprSpillSequence> tensor_full_wave_spill(const VgprSpillSequence &spill,
                                                        uint16_t exec_save_sgpr,
                                                        rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA5 || exec_save_sgpr > 104u || exec_save_sgpr % 2u != 0u ||
      spill.uses_dynamic_stack_frame || spill.vgpr_count == 0 ||
      !spill.has_complete_slot_metadata() || spill.save_words.empty() ||
      spill.restore_words.empty())
    return std::nullopt;
  const auto save_exec = instrumentation::build_s_mov_b64(exec_save_sgpr, kAmdGpuExecLo, arch);
  const auto full_exec =
      instrumentation::build_s_mov_b64(kAmdGpuExecLo, kScalarInlineNegativeOneOperand, arch);
  const auto restore_exec = instrumentation::build_s_mov_b64(kAmdGpuExecLo, exec_save_sgpr, arch);
  const auto wait_scalar = instrumentation::build_s_wait_scalar_load0(arch);
  if (!save_exec || !full_exec || !restore_exec || !wait_scalar)
    return std::nullopt;
  auto result = spill;
  const auto wrap = [&](std::vector<uint32_t> &words) {
    words.insert(words.begin(), {*wait_scalar, *save_exec, *full_exec});
    words.push_back(*restore_exec);
  };
  wrap(result.save_words);
  wrap(result.restore_words);
  return result;
}

std::optional<SgprSpillSequence> tensor_full_wave_scalar_spill(const SgprSpillSequence &spill,
                                                               uint16_t auxiliary_sgpr,
                                                               rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA5 || auxiliary_sgpr > 104u || auxiliary_sgpr % 2u ||
      !spill.memory_transfer_vgpr || spill.lane_reservoir_vgpr || spill.sgpr_count == 0u ||
      spill.memory_slot_store_words.size() != spill.sgpr_count || spill.save_words.empty() ||
      spill.restore_words.empty() ||
      (auxiliary_sgpr < spill.sgpr_base + spill.sgpr_count &&
       spill.sgpr_base < auxiliary_sgpr + 2u))
    return std::nullopt;
  const auto save = instrumentation::build_s_mov_b64(auxiliary_sgpr, kAmdGpuExecLo, arch);
  const auto full =
      instrumentation::build_s_mov_b64(kAmdGpuExecLo, kScalarInlineNegativeOneOperand, arch);
  const auto restore = instrumentation::build_s_mov_b64(kAmdGpuExecLo, auxiliary_sgpr, arch);
  const auto wait_scalar = instrumentation::build_s_wait_scalar_load0(arch);
  if (!save || !full || !restore || !wait_scalar)
    return std::nullopt;
  auto result = spill;
  for (auto *words : {&result.save_words, &result.restore_words}) {
    words->insert(words->begin(), {*wait_scalar, *save, *full});
    words->push_back(*restore);
  }
  return result;
}

bool prepare_tensor_full_wave_resources(PlannedProbeResources &probe, const OperatingPoint &point,
                                        rj_code_arch_t arch) {
  if (!point.exec_save_sgpr)
    return false;
  uint16_t bootstrap = *point.exec_save_sgpr;
  std::optional<SgprSpillSequence> scalar = probe.scalar_spill;
  if (scalar) {
    if (scalar->lane_reservoir_vgpr)
      return !probe.spill;
    if (!probe.spill || probe.spill->uses_dynamic_stack_frame || !point.scalar_spill_setup)
      return false;
    bootstrap = point.scalar_spill_setup->temporaries.frame_base_sgpr;
    scalar = tensor_full_wave_scalar_spill(*scalar, bootstrap, arch);
    if (!scalar)
      return false;
  }
  std::optional<VgprSpillSequence> vector = probe.spill;
  if (vector) {
    vector = tensor_full_wave_spill(*vector, bootstrap, arch);
    if (!vector)
      return false;
  }
  probe.scalar_spill = std::move(scalar);
  probe.spill = std::move(vector);
  return true;
}

bool append_full_wave_private_identity(std::vector<uint32_t> &words,
                                       const PrivateStateLayout &layout, uint16_t scratch_vgpr,
                                       uint16_t exec_save_sgpr, const VgprSpillSequence &spill,
                                       rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA5 || !layout.is_well_formed() || !layout.owner_offset ||
      scratch_vgpr > 254u || exec_save_sgpr > 104u || exec_save_sgpr % 2u ||
      spill.vgpr_base != scratch_vgpr || spill.vgpr_count != 2u || spill.uses_dynamic_stack_frame ||
      !spill.has_complete_slot_metadata())
    return false;
  const uint16_t archive = scratch_vgpr + 1u;
  InstructionSequence sequence(words);
  if (!sequence.emit_all(
          instrumentation::build_s_mov_b64(exec_save_sgpr, kAmdGpuExecLo, arch),
          instrumentation::build_s_mov_b64(kAmdGpuExecLo, kScalarInlineNegativeOneOperand, arch),
          spill.save_words,
          instrumentation::build_v_writelane_b32(archive, exec_save_sgpr, 0, arch),
          instrumentation::build_v_writelane_b32(archive, exec_save_sgpr + 1u, 1, arch)))
    return false;
  std::vector<uint32_t> offsets{*layout.owner_offset, layout.epoch_offset};
  for (auto offset : layout.exact_workgroup_offsets.values())
    if (offset)
      offsets.push_back(*offset);
  if (layout.dispatch_id_offset) {
    offsets.push_back(*layout.dispatch_id_offset);
    offsets.push_back(*layout.dispatch_id_offset + SpillManager::kSlotBytes);
  }
  for (uint32_t offset : offsets) {
    if (!sequence.emit_all(
            instrumentation::build_private_load_b32(scratch_vgpr, offset, arch),
            instrumentation::build_s_wait_private_load0(arch),
            instrumentation::build_v_readlane_b32(exec_save_sgpr, scratch_vgpr, 0, arch),
            instrumentation::build_valu_to_salu_dependency_wait(arch),
            build_v_mov_b32_e32(scratch_vgpr, exec_save_sgpr, arch),
            instrumentation::build_private_store_b32(scratch_vgpr, offset, arch)))
      return false;
  }
  return sequence.emit_all(
      instrumentation::build_s_wait_private_store0(arch),
      instrumentation::build_v_readlane_b32(exec_save_sgpr, archive, 0, arch),
      instrumentation::build_v_readlane_b32(exec_save_sgpr + 1u, archive, 1, arch),
      instrumentation::build_valu_to_salu_dependency_wait(arch), spill.restore_words,
      instrumentation::build_s_mov_b64(kAmdGpuExecLo, exec_save_sgpr, arch));
}

namespace {
bool append_tensor_divmod(std::vector<uint32_t> &words, uint16_t dividend_vgpr,
                          uint16_t divisor_vgpr, uint16_t quotient_vgpr, uint16_t remainder_vgpr,
                          uint16_t scratch_vgpr, unsigned width, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA5 || scratch_vgpr + width + 2u > 256u)
    return false;
  const std::array<uint16_t, 4> registers{dividend_vgpr, divisor_vgpr, quotient_vgpr,
                                          remainder_vgpr};
  for (size_t i = 0; i < registers.size(); ++i) {
    if (registers[i] + width > 256u ||
        (registers[i] < scratch_vgpr + width + 2u && scratch_vgpr < registers[i] + width))
      return false;
    for (size_t j = 0; j < i; ++j)
      if (registers[i] < registers[j] + width && registers[j] < registers[i] + width)
        return false;
  }
  const uint16_t counter = scratch_vgpr;
  const uint16_t bit = scratch_vgpr + width + 1u;
  const uint16_t reduced = scratch_vgpr + 1u;
  const auto zero = scalar_positive_inline_u32(0);
  const auto one = scalar_positive_inline_u32(1);
  const auto vr = [](uint16_t reg) { return vector_source_vgpr(reg); };
  const auto binary = [&](uint16_t opcode, uint16_t dst, uint16_t lhs, uint16_t rhs) {
    return cdna5::build_vop3(opcode, {.vdst = static_cast<uint8_t>(dst), .src0 = lhs, .src1 = rhs});
  };
  const auto select = [&](uint16_t dst, uint16_t when_false, uint16_t when_true) {
    return cdna5::build_vop3(cdna5::kVCndmaskB32Vop3, {.vdst = static_cast<uint8_t>(dst),
                                                       .src0 = when_false,
                                                       .src1 = when_true,
                                                       .src2 = kAmdGpuVccLo});
  };
  InstructionSequence sequence(words);
  const auto shift_left = [&](uint16_t reg) {
    if (width == 1)
      sequence.append(instrumentation::build_v_lshlrev_b32(reg, one, reg, arch));
    else
      sequence.append(binary(cdna5::kVLshlrevB64Vop3, reg, one, vr(reg)));
  };
  for (unsigned word = 0; word < width; ++word)
    sequence.append(build_v_mov_b32_e32(quotient_vgpr + word, zero, arch),
                    build_v_mov_b32_e32(remainder_vgpr + word, zero, arch));
  // Exact restoring division avoids reciprocal rounding at tensor boundaries.
  // After k steps, the remainder is at most the consumed k-bit numerator
  // prefix. Thus the shift cannot overflow within these width*32 steps,
  // including division by zero (all quotient bits set, dividend retained).
  sequence.append(build_v_mov_b32_e32(counter, scalar_positive_inline_u32(width * 32 - 1), arch));
  const auto loop = sequence.make_label();
  sequence.bind_label(loop);
  if (width == 1)
    sequence.append(instrumentation::build_v_lshrrev_b32(bit, vr(counter), dividend_vgpr, arch),
                    instrumentation::build_v_and_b32(bit, one, bit, arch));
  else
    sequence.append(binary(cdna5::kVLshrrevB64Vop3, reduced, vr(counter), vr(dividend_vgpr)),
                    instrumentation::build_v_and_b32(bit, one, reduced, arch));
  shift_left(remainder_vgpr);
  sequence.append(binary(cdna5::kVOrB32Vop3, remainder_vgpr, vr(remainder_vgpr), vr(bit)),
                  binary(width == 1 ? cdna5::kVSubNcU32Vop3 : cdna5::kVSubNcU64Vop3, reduced,
                         vr(remainder_vgpr), vr(divisor_vgpr)));
  if (width == 1)
    sequence.append(
        instrumentation::build_v_cmp_gt_u32_vcc(vr(divisor_vgpr), remainder_vgpr, arch));
  else
    sequence.append(
        cdna5::build_vopc(cdna5::kVCmpGtU64Vopc, {.src0 = vr(divisor_vgpr),
                                                  .vsrc1 = static_cast<uint8_t>(remainder_vgpr)}));
  sequence.append(select(bit, one, zero));
  for (unsigned word = 0; word < width; ++word)
    sequence.append(select(remainder_vgpr + word, vr(reduced + word), vr(remainder_vgpr + word)));
  shift_left(quotient_vgpr);
  sequence.append(binary(cdna5::kVOrB32Vop3, quotient_vgpr, vr(quotient_vgpr), vr(bit)));
  sequence.append(
      binary(cdna5::kVSubNcU32Vop3, counter, vr(counter), one),
      instrumentation::build_v_cmp_ne_u32_vcc(kScalarInlineNegativeOneOperand, counter, arch));
  sequence.branch(loop, InstructionSequence::BranchKind::VccNonzero);
  return sequence.finish(arch);
}
} // namespace

bool append_tensor_divmod_u32(std::vector<uint32_t> &words, uint16_t dividend_vgpr,
                              uint16_t divisor_vgpr, uint16_t quotient_vgpr,
                              uint16_t remainder_vgpr, uint16_t scratch_vgpr, rj_code_arch_t arch) {
  return append_tensor_divmod(words, dividend_vgpr, divisor_vgpr, quotient_vgpr, remainder_vgpr,
                              scratch_vgpr, 1, arch);
}

bool append_tensor_divmod_u64(std::vector<uint32_t> &words, uint16_t dividend_vgpr,
                              uint16_t divisor_vgpr, uint16_t quotient_vgpr,
                              uint16_t remainder_vgpr, uint16_t scratch_vgpr, rj_code_arch_t arch) {
  return append_tensor_divmod(words, dividend_vgpr, divisor_vgpr, quotient_vgpr, remainder_vgpr,
                              scratch_vgpr, 2, arch);
}

bool append_tensor_iteration_origin(std::vector<uint32_t> &words, const ProgramSite &site,
                                    uint16_t offset_vgpr, uint16_t origin_vgpr,
                                    uint16_t scratch_vgpr, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA5 || site.origin != AccessOrigin::TensorLds ||
      (site.kind != LdsAccessKind::Write && site.kind != LdsAccessKind::Read) ||
      !site.operands.tensor_descriptor_sgprs || offset_vgpr > 254u || origin_vgpr > 250u ||
      scratch_vgpr > 238u)
    return false;
  const auto overlaps = [](uint16_t a, unsigned an, uint16_t b, unsigned bn) {
    return a < b + bn && b < a + an;
  };
  if (overlaps(offset_vgpr, 2, origin_vgpr, 6) || overlaps(offset_vgpr, 2, scratch_vgpr, 18) ||
      overlaps(origin_vgpr, 6, scratch_vgpr, 18))
    return false;
  const auto &groups = *site.operands.tensor_descriptor_sgprs;
  if (groups[0] > 102u || groups[1] > 98u || (groups[2] > 102u && groups[2] != 124u) ||
      (groups[3] > 102u && groups[3] != 124u))
    return false;
  const uint16_t remaining = scratch_vgpr;
  const uint16_t stride1 = scratch_vgpr + 2u;
  const uint16_t stride2 = scratch_vgpr + 4u;
  const uint16_t divisor = scratch_vgpr + 6u;
  const uint16_t order = scratch_vgpr + 8u;
  const uint16_t field = scratch_vgpr + 9u;
  const uint16_t quotient = scratch_vgpr + 10u;
  const uint16_t remainder = scratch_vgpr + 12u;
  const uint16_t division_scratch = scratch_vgpr + 14u;
  const auto zero = scalar_positive_inline_u32(0);
  const auto one = scalar_positive_inline_u32(1);
  const auto vr = [](uint16_t reg) { return vector_source_vgpr(reg); };
  const auto binary = [&](uint16_t opcode, uint16_t dst, uint16_t lhs, uint16_t rhs) {
    return cdna5::build_vop3(opcode, {.vdst = static_cast<uint8_t>(dst), .src0 = lhs, .src1 = rhs});
  };
  const auto select = [&](uint16_t dst, uint16_t when_false, uint16_t when_true) {
    return cdna5::build_vop3(cdna5::kVCndmaskB32Vop3, {.vdst = static_cast<uint8_t>(dst),
                                                       .src0 = when_false,
                                                       .src1 = when_true,
                                                       .src2 = kAmdGpuVccLo});
  };
  InstructionSequence sequence(words);
  const auto read_word = [&](uint16_t dst, size_t group, uint16_t word) {
    sequence.append(
        build_v_mov_b32_e32(dst, groups[group] == 124u ? zero : groups[group] + word, arch));
  };
  // Strides are 48-bit fields, packed consecutively starting at D1 bit 160.
  read_word(stride1, 1, 5);
  read_word(stride1 + 1, 1, 6);
  sequence.append(
      instrumentation::build_v_and_b32_literal(stride1 + 1, 0xffffu, stride1 + 1, arch));
  read_word(stride2, 1, 6);
  read_word(stride2 + 1, 1, 7);
  sequence.append(
      instrumentation::build_v_lshrrev_b32(stride2, scalar_positive_inline_u32(16), stride2, arch),
      instrumentation::build_v_lshlrev_b32(field, scalar_positive_inline_u32(16), stride2 + 1,
                                           arch),
      binary(cdna5::kVOrB32Vop3, stride2, vr(stride2), vr(field)),
      instrumentation::build_v_lshrrev_b32(stride2 + 1, scalar_positive_inline_u32(16), stride2 + 1,
                                           arch));
  // Unit axes are absent from the inverse. A zero stride encodes a skipped
  // axis here; active advancing layouts cannot have a zero non-unit stride.
  for (unsigned axis = 1; axis <= 2; ++axis) {
    const uint16_t stride = axis == 1 ? stride1 : stride2;
    if (axis == 1) {
      read_word(field, 1, 2);
      read_word(divisor, 1, 3);
      sequence.append(
          instrumentation::build_v_lshrrev_b32(field, scalar_positive_inline_u32(16), field, arch),
          instrumentation::build_v_lshlrev_b32(divisor, scalar_positive_inline_u32(16), divisor,
                                               arch),
          binary(cdna5::kVOrB32Vop3, field, vr(field), vr(divisor)));
    } else {
      read_word(field, 2, 0);
    }
    for (const auto extent : {zero, one}) {
      sequence.append(instrumentation::build_v_cmp_eq_u32_vcc(extent, field, arch),
                      select(stride, vr(stride), zero), select(stride + 1, vr(stride + 1), zero));
    }
    if (axis == 2) {
      read_word(field, 1, 4);
      sequence.append(
          instrumentation::build_v_lshrrev_b32(field, scalar_positive_inline_u32(16), field, arch),
          instrumentation::build_v_cmp_eq_u32_vcc(zero, field, arch),
          select(stride, vr(stride), zero), select(stride + 1, vr(stride + 1), zero));
    }
  }
  sequence.append(cdna5::build_vopc(cdna5::kVCmpGtU64Vopc,
                                    {.src0 = vr(stride1), .vsrc1 = static_cast<uint8_t>(stride2)}),
                  select(order, zero, one));
  for (unsigned word = 0; word < 2; ++word)
    sequence.append(build_v_mov_b32_e32(remaining + word, vr(offset_vgpr + word), arch));
  for (unsigned word = 0; word < 6; ++word)
    sequence.append(build_v_mov_b32_e32(origin_vgpr + word, zero, arch));
  for (unsigned step = 0; step < 2; ++step) {
    sequence.append(instrumentation::build_v_cmp_ne_u32_vcc(zero, order, arch));
    for (unsigned word = 0; word < 2; ++word)
      sequence.append(select(divisor + word, vr((step == 0 ? stride2 : stride1) + word),
                             vr((step == 0 ? stride1 : stride2) + word)));
    sequence.require(append_tensor_divmod_u64(words, remaining, divisor, quotient, remainder,
                                              division_scratch, arch));
    sequence.append(binary(cdna5::kVOrB32Vop3, field, vr(divisor), vr(divisor + 1)),
                    instrumentation::build_v_cmp_eq_u32_vcc(zero, field, arch));
    for (unsigned word = 0; word < 2; ++word)
      sequence.append(select(quotient + word, vr(quotient + word), zero),
                      select(remaining + word, vr(remainder + word), vr(remaining + word)));
    sequence.append(instrumentation::build_v_cmp_ne_u32_vcc(zero, order, arch));
    for (unsigned word = 0; word < 2; ++word) {
      const uint16_t a = origin_vgpr + 2u + word;
      const uint16_t b = origin_vgpr + 4u + word;
      sequence.append(
          select(a, vr(step == 0 ? a : quotient + word), vr(step == 0 ? quotient + word : a)),
          select(b, vr(step == 0 ? quotient + word : b), vr(step == 0 ? b : quotient + word)));
    }
  }
  // Dimension zero has stride one. If it is a unit axis, leftover padding
  // still belongs here and must remain visible to subsequent bounds checks.
  for (unsigned word = 0; word < 2; ++word)
    sequence.append(build_v_mov_b32_e32(origin_vgpr + word, vr(remaining + word), arch));
  return sequence.finish();
}

bool append_materialize_tensor_global_address(std::vector<uint32_t> &words, const ProgramSite &site,
                                              uint16_t element_vgpr, uint16_t count_vgpr,
                                              uint16_t address_vgpr, uint16_t in_bounds_vgpr,
                                              uint16_t scratch_vgpr, rj_code_arch_t arch,
                                              std::optional<uint16_t> iteration_hash_vgpr) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA5 || site.origin != AccessOrigin::TensorLds ||
      (site.kind != LdsAccessKind::Write && site.kind != LdsAccessKind::Read) ||
      !site.operands.tensor_descriptor_sgprs ||
      (site.kind == LdsAccessKind::Read && !iteration_hash_vgpr))
    return false;
  const std::array<std::pair<uint16_t, unsigned>, 5> windows{{{element_vgpr, 1},
                                                              {count_vgpr, 1},
                                                              {address_vgpr, 2},
                                                              {in_bounds_vgpr, 1},
                                                              {scratch_vgpr, 30}}};
  for (size_t i = 0; i < windows.size(); ++i) {
    const auto [base, size] = windows[i];
    if (base + size > 256u)
      return false;
    for (size_t j = 0; j < i; ++j)
      if (base < windows[j].first + windows[j].second && windows[j].first < base + size)
        return false;
  }
  if (iteration_hash_vgpr) {
    if (*iteration_hash_vgpr >= 256u)
      return false;
    for (const auto &[base, size] : windows)
      if (*iteration_hash_vgpr >= base && *iteration_hash_vgpr < base + size)
        return false;
  }
  const auto &groups = *site.operands.tensor_descriptor_sgprs;
  if (groups[0] > 102u || groups[1] > 98u || (groups[2] > 102u && groups[2] != 124u) ||
      (groups[3] > 102u && groups[3] != 124u))
    return false;
  const uint16_t global = scratch_vgpr;
  const uint16_t linear = scratch_vgpr + 2;
  const uint16_t iterate = scratch_vgpr + 3;
  const uint16_t gather = scratch_vgpr + 4;
  const uint16_t iteration = scratch_vgpr + 5;
  const uint16_t origin = scratch_vgpr + 6;
  const uint16_t divisor = scratch_vgpr + 12;
  const uint16_t coordinate = scratch_vgpr + 13;
  const uint16_t quotient = scratch_vgpr + 14;
  const uint16_t division_scratch = scratch_vgpr + 15;
  const uint16_t stride = scratch_vgpr + 18;
  const uint16_t temporary = scratch_vgpr + 20;
  const uint16_t extent = scratch_vgpr + 22;
  const uint16_t active_axis = scratch_vgpr + 23;
  const uint16_t field = scratch_vgpr + 24;
  const uint16_t auxiliary = scratch_vgpr + 25;
  const uint16_t wide = scratch_vgpr + 26;
  const auto zero = scalar_positive_inline_u32(0);
  const auto one = scalar_positive_inline_u32(1);
  const auto vr = [](uint16_t reg) { return vector_source_vgpr(reg); };
  const auto binary = [&](uint16_t opcode, uint16_t dst, uint16_t lhs, uint16_t rhs) {
    return cdna5::build_vop3(opcode, {.vdst = static_cast<uint8_t>(dst), .src0 = lhs, .src1 = rhs});
  };
  const auto select = [&](uint16_t dst, uint16_t when_false, uint16_t when_true) {
    return cdna5::build_vop3(cdna5::kVCndmaskB32Vop3, {.vdst = static_cast<uint8_t>(dst),
                                                       .src0 = when_false,
                                                       .src1 = when_true,
                                                       .src2 = kAmdGpuVccLo});
  };
  InstructionSequence sequence(words);
  const auto read_word = [&](uint16_t dst, size_t group, uint16_t word) {
    sequence.append(
        build_v_mov_b32_e32(dst, groups[group] == 124u ? zero : groups[group] + word, arch));
  };
  const auto shr = [&](uint16_t dst, uint16_t src, unsigned bits) {
    sequence.append(
        instrumentation::build_v_lshrrev_b32(dst, scalar_positive_inline_u32(bits), src, arch));
  };
  const auto shl = [&](uint16_t dst, uint16_t src, unsigned bits) {
    sequence.append(
        instrumentation::build_v_lshlrev_b32(dst, scalar_positive_inline_u32(bits), src, arch));
  };
  const auto low_half = [&](uint16_t reg) {
    sequence.append(instrumentation::build_v_and_b32_literal(reg, 0xffffu, reg, arch));
  };
  const auto multiply_32_48 = [&](uint16_t dst, uint16_t coordinate_reg, uint16_t stride_reg,
                                  uint16_t scratch) {
    sequence.append(binary(cdna5::kVMulLoU32Vop3, dst, vr(coordinate_reg), vr(stride_reg)),
                    binary(cdna5::kVMulHiU32Vop3, dst + 1, vr(coordinate_reg), vr(stride_reg)),
                    binary(cdna5::kVMulLoU32Vop3, scratch, vr(coordinate_reg), vr(stride_reg + 1)),
                    instrumentation::build_v_add_u32(dst + 1, vr(scratch), dst + 1, arch));
  };
  read_word(gather, 0, 0);
  shr(gather, gather, 31);
  read_word(iterate, 1, 0);
  shr(iterate, iterate, 19);
  sequence.append(instrumentation::build_v_and_b32(iterate, one, iterate, arch),
                  instrumentation::build_v_cmp_eq_u32_vcc(zero, gather, arch),
                  select(iterate, zero, vr(iterate)));
  read_word(wide, 2, 3);
  shr(wide, wide, 16);      // iteration_count - 1
  read_word(divisor, 2, 1); // unpadded LDS element increment
  if (site.kind == LdsAccessKind::Read) {
    // A store reads the selected iteration, even when later iterations overlap
    // its LDS source. A masked later store must not suppress an earlier read.
    sequence.append(instrumentation::build_v_add_u32(wide, one, wide, arch),
                    binary(cdna5::kVMulHiU32Vop3, iteration, vr(*iteration_hash_vgpr), vr(wide)));
  } else {
    sequence.require(append_tensor_divmod_u32(words, element_vgpr, divisor, iteration, linear,
                                              division_scratch, arch));
    // Loads compare against the final writer of an overlapping LDS element.
    sequence.append(instrumentation::build_v_min_u32(iteration, vr(wide), iteration, arch));
  }
  sequence.append(instrumentation::build_v_cmp_eq_u32_vcc(zero, iterate, arch),
                  select(iteration, vr(iteration), zero),
                  binary(cdna5::kVMulLoU32Vop3, coordinate, vr(iteration), vr(divisor)),
                  binary(cdna5::kVSubNcU32Vop3, linear, vr(element_vgpr), vr(coordinate)));
  read_word(stride, 2, 2);
  read_word(stride + 1, 2, 3);
  low_half(stride + 1);
  multiply_32_48(global, iteration, stride, field);
  // Non-iterating descriptors have zero offset and therefore zero origins,
  // even if their optional fields encode gather indices or unrelated axes.
  sequence.require(append_tensor_iteration_origin(words, site, global, origin, divisor, arch));
  sequence.append(instrumentation::build_v_cmp_ne_u32_vcc(zero, count_vgpr, arch),
                  select(in_bounds_vgpr, zero, one));

  for (unsigned axis = 0; axis < 5; ++axis) {
    const size_t group = axis == 4 ? 3u : axis == 3 ? 2u : 1u;
    const uint16_t word = axis == 4 ? 2u : axis == 3 || axis == 0 ? 3u : 4u;
    read_word(divisor, group, word);
    if (axis == 1)
      low_half(divisor);
    else
      shr(divisor, divisor, 16);
    if (axis >= 2)
      sequence.append(instrumentation::build_v_cmp_ne_u32_vcc(zero, gather, arch),
                      select(divisor, vr(divisor), zero));
    if (axis == 3)
      sequence.append(instrumentation::build_v_cmp_ne_u32_vcc(zero, iterate, arch),
                      select(divisor, vr(divisor), zero));
    sequence.append(instrumentation::build_v_cmp_ne_u32_vcc(zero, divisor, arch),
                    select(active_axis, zero, one), select(divisor, one, vr(divisor)));
    sequence.require(append_tensor_divmod_u32(words, linear, divisor, quotient, coordinate,
                                              division_scratch, arch));
    sequence.append(build_v_mov_b32_e32(linear, vr(quotient), arch));
    if (axis == 1) {
      // Gather rows select from packed halfword or dword indices, not from a
      // tensor coordinate. Only the first 8 indices exist in dword mode.
      read_word(wide, 0, 0);
      shr(wide, wide, 30);
      sequence.append(instrumentation::build_v_and_b32(wide, one, wide, arch),
                      build_v_mov_b32_e32(temporary, zero, arch));
      for (unsigned index = 0; index < 16; ++index) {
        read_word(auxiliary, index < 8 ? 2 : 3, (index % 8) / 2);
        if (index % 2)
          shr(auxiliary, auxiliary, 16);
        else
          low_half(auxiliary);
        if (index < 8)
          read_word(field, index < 4 ? 2 : 3, index % 4);
        else
          sequence.append(build_v_mov_b32_e32(field, zero, arch));
        sequence.append(instrumentation::build_v_cmp_ne_u32_vcc(zero, wide, arch),
                        select(auxiliary, vr(auxiliary), vr(field)),
                        instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(index),
                                                                coordinate, arch),
                        select(temporary, vr(temporary), vr(auxiliary)));
      }
      sequence.append(instrumentation::build_v_cmp_ne_u32_vcc(zero, gather, arch),
                      select(coordinate, vr(coordinate), vr(temporary)));
    }
    if (axis == 2 || axis == 3) {
      read_word(extent, 2, axis - 2);
    } else {
      const size_t extent_group = axis == 4 ? 3u : 1u;
      const uint16_t extent_word = axis == 1 ? 2u : 1u;
      read_word(extent, extent_group, extent_word);
      read_word(field, extent_group, extent_word + 1);
      shr(extent, extent, 16);
      shl(field, field, 16);
      sequence.append(binary(cdna5::kVOrB32Vop3, extent, vr(extent), vr(field)));
    }
    sequence.append(build_v_mov_b32_e32(temporary, vr(coordinate), arch),
                    build_v_mov_b32_e32(temporary + 1, zero, arch));
    if (axis < 3)
      sequence.append(
          binary(cdna5::kVAddNcU64Vop3, temporary, vr(temporary), vr(origin + 2 * axis)));
    sequence.append(
        instrumentation::build_v_cmp_gt_u32_vcc(vr(extent), temporary, arch),
        select(field, zero, one),
        instrumentation::build_v_cmp_eq_u32_vcc(zero, temporary + 1, arch),
        select(field, zero, vr(field)),
        instrumentation::build_v_cmp_eq_u32_vcc(zero, active_axis, arch),
        select(field, vr(field), one),
        instrumentation::build_v_and_b32(in_bounds_vgpr, vr(field), in_bounds_vgpr, arch));
    if (axis == 0) {
      sequence.append(build_v_mov_b32_e32(stride, one, arch),
                      build_v_mov_b32_e32(stride + 1, zero, arch));
    } else if (axis == 2) {
      read_word(stride, 1, 6);
      read_word(stride + 1, 1, 7);
      shr(stride, stride, 16);
      shl(field, stride + 1, 16);
      sequence.append(binary(cdna5::kVOrB32Vop3, stride, vr(stride), vr(field)));
      shr(stride + 1, stride + 1, 16);
    } else {
      const size_t stride_group = axis == 1 ? 1u : axis == 3 ? 2u : 3u;
      const uint16_t stride_word = axis == 1 ? 5u : axis == 3 ? 2u : 0u;
      read_word(stride, stride_group, stride_word);
      read_word(stride + 1, stride_group, stride_word + 1);
      low_half(stride + 1);
    }
    multiply_32_48(temporary, coordinate, stride, field);
    sequence.append(binary(cdna5::kVAddNcU64Vop3, global, vr(global), vr(temporary)));
  }
  read_word(divisor, 1, 0);
  shr(divisor, divisor, 16);
  sequence.append(
      instrumentation::build_v_and_b32(divisor, scalar_positive_inline_u32(3), divisor, arch),
      binary(cdna5::kVLshlrevB64Vop3, address_vgpr, vr(divisor), vr(global)));
  read_word(temporary, 0, 2);
  read_word(temporary + 1, 0, 3);
  sequence.append(
      instrumentation::build_v_and_b32_literal(temporary + 1, (1u << 25) - 1u, temporary + 1, arch),
      binary(cdna5::kVAddNcU64Vop3, address_vgpr, vr(address_vgpr), vr(temporary)));
  return sequence.finish();
}

uint16_t tensor_load_compare_state_sgprs(const ProgramSite &site) {
  if (!site.operands.tensor_descriptor_sgprs)
    return 0;
  const auto &groups = *site.operands.tensor_descriptor_sgprs;
  for (size_t group : {0u, 2u, 3u})
    if (groups[group] != 124u && groups[1] >= groups[group] && groups[1] < groups[group] + 4u)
      return 12;
  return 4;
}

bool append_tensor_load_compare(std::vector<uint32_t> &words, const ProgramSite &site,
                                std::span<const uint32_t> original, uint16_t scratch_vgpr,
                                uint16_t state_sgpr, std::span<const uint32_t> delay_words,
                                std::span<const uint32_t> mismatch_words,
                                uint32_t &guest_word_offset, rj_code_arch_t arch) {
  const uint16_t state_count = tensor_load_compare_state_sgprs(site);
  if (arch != ROCJITSU_CODE_ARCH_CDNA5 || site.origin != AccessOrigin::TensorLds ||
      site.kind != LdsAccessKind::Write || !site.operands.tensor_descriptor_sgprs ||
      original.size() != 3 ||
      static_cast<uint32_t>(scratch_vgpr) + kTensorLoadCompareScratchVgprs > 256u ||
      static_cast<uint32_t>(state_sgpr) + state_count > 106u || state_sgpr % 2u != 0)
    return false;
  const auto &groups = *site.operands.tensor_descriptor_sgprs;
  const std::array<unsigned, 4> sizes{4, 8, 4, 4};
  for (size_t i = 0; i < groups.size(); ++i) {
    if (i >= 2 && groups[i] == 124u)
      continue;
    if (groups[i] + sizes[i] > 106u ||
        (state_sgpr < groups[i] + sizes[i] &&
         groups[i] < state_sgpr + static_cast<uint32_t>(state_count)))
      return false;
  }
  const uint16_t hash = scratch_vgpr;
  const uint16_t iteration_hash = scratch_vgpr + 1;
  const uint16_t element = scratch_vgpr + 2;
  const uint16_t count = scratch_vgpr + 3;
  const uint16_t global = scratch_vgpr + 4;
  const uint16_t bounds = scratch_vgpr + 6;
  const uint16_t lds = scratch_vgpr + 7;
  const uint16_t expected = scratch_vgpr + 8;
  const uint16_t readback = scratch_vgpr + 10;
  const uint16_t width = scratch_vgpr + 12;
  const uint16_t descriptor_archive = scratch_vgpr + 13;
  const uint16_t completion_address = scratch_vgpr + 14;
  const uint16_t work = scratch_vgpr + kTensorLoadCompareWorkspaceOffset;
  const auto zero = scalar_positive_inline_u32(0);
  const auto one = scalar_positive_inline_u32(1);
  const auto vr = [](uint16_t reg) { return vector_source_vgpr(reg); };
  const auto binary = [&](uint16_t opcode, uint16_t dst, uint16_t lhs, uint16_t rhs) {
    return cdna5::build_vop3(opcode, {.vdst = static_cast<uint8_t>(dst), .src0 = lhs, .src1 = rhs});
  };
  const auto select = [&](uint16_t dst, uint16_t when_false, uint16_t when_true) {
    return cdna5::build_vop3(cdna5::kVCndmaskB32Vop3, {.vdst = static_cast<uint8_t>(dst),
                                                       .src0 = when_false,
                                                       .src1 = when_true,
                                                       .src2 = kAmdGpuVccLo});
  };
  InstructionSequence sequence(words);
  sequence.append(
      instrumentation::build_s_mov_b64(state_sgpr, kAmdGpuVccLo, arch),
      instrumentation::build_s_mov_b64(state_sgpr + 2, kAmdGpuExecLo, arch),
      instrumentation::build_s_mov_b64(kAmdGpuExecLo, kScalarInlineNegativeOneOperand, arch),
      instrumentation::build_v_mbcnt_lo_u32_b32(hash, kScalarInlineNegativeOneOperand, zero, arch),
      instrumentation::build_v_mov_b32_literal(work, 0x9e3779b9u, arch),
      binary(cdna5::kVMulLoU32Vop3, hash, vr(hash), vr(work)),
      instrumentation::build_v_xor_b32(hash, groups[0] + 1, hash, arch),
      instrumentation::build_v_xor_b32(hash, groups[0] + 2, hash, arch),
      instrumentation::build_v_mov_b32_literal(work, 0x85ebca6bu, arch),
      binary(cdna5::kVMulLoU32Vop3, iteration_hash, vr(hash), vr(work)));
  sequence.require(
      append_select_tensor_element(words, site, hash, iteration_hash, element, count, work, arch));
  sequence.require(append_materialize_tensor_global_address(words, site, element, count, global,
                                                            bounds, work, arch));
  sequence.require(append_materialize_tensor_lds_address(words, site, element, lds, work, arch));
  sequence.append(
      build_v_mov_b32_e32(descriptor_archive, groups[1], arch),
      instrumentation::build_v_lshrrev_b32(width, scalar_positive_inline_u32(16),
                                           descriptor_archive, arch),
      instrumentation::build_v_and_b32(width, scalar_positive_inline_u32(3), width, arch));
  for (unsigned word = 0; word < 2; ++word)
    sequence.append(build_v_mov_b32_e32(expected + word, zero, arch),
                    build_v_mov_b32_e32(readback + word, zero, arch));
  const auto append_load = [&](bool from_global) {
    const auto done = sequence.make_label();
    sequence.append(
        instrumentation::build_v_cmp_ne_u32_vcc(zero, from_global ? bounds : count, arch),
        instrumentation::build_s_mov_b64(kAmdGpuExecLo, kAmdGpuVccLo, arch));
    const std::array<uint16_t, 4> global_ops{cdna5::kFlatLoadU8Vflat, cdna5::kFlatLoadU16Vflat,
                                             cdna5::kFlatLoadB32Vflat, cdna5::kFlatLoadB64Vflat};
    const std::array<uint16_t, 4> lds_ops{cdna5::kDsLoadU8Vds, cdna5::kDsLoadU16Vds,
                                          cdna5::kDsLoadB32Vds, cdna5::kDsLoadB64Vds};
    for (unsigned size = 0; size < 4; ++size) {
      const auto next = sequence.make_label();
      sequence.append(
          instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(size), width, arch));
      sequence.branch(next, InstructionSequence::BranchKind::VccZero);
      if (from_global)
        sequence.append(
            cdna5::build_vflat(global_ops[size], {.saddr = kGfx1250FlatNoSaddrEncoding,
                                                  .vdst = static_cast<uint8_t>(expected),
                                                  .vaddr = static_cast<uint8_t>(global)}));
      else
        sequence.append(cdna5::build_vds(lds_ops[size], {.addr = static_cast<uint8_t>(lds),
                                                         .vdst = static_cast<uint8_t>(readback)}));
      sequence.branch(done, InstructionSequence::BranchKind::Unconditional);
      sequence.bind_label(next);
    }
    sequence.bind_label(done);
    sequence.append(
        from_global ? instrumentation::build_s_wait_global_load0(arch)
                    : instrumentation::build_s_wait_lds0(arch),
        instrumentation::build_s_mov_b64(kAmdGpuExecLo, kScalarInlineNegativeOneOperand, arch));
  };
  append_load(true);
  // Delay publication until the comparison is complete. Otherwise a consumer
  // released by the DMA's barrier arrival may legally reuse this LDS tile.
  // D1[0] may also name a word of D0/D2/D3: modifying it in place would then
  // change that second operand. Such sites use a private scalar D1 copy.
  const uint16_t dma_descriptor = state_count == 12 ? state_sgpr + 4 : groups[1];
  std::array<uint32_t, 3> dma{original[0], original[1], original[2]};
  if (state_count == 12) {
    for (unsigned word = 0; word < 8; ++word)
      sequence.append(build_s_mov_b32(dma_descriptor + word, groups[1] + word, arch));
    dma[2] = (dma[2] & ~0xff00u) | (static_cast<uint32_t>(dma_descriptor) << 8);
  }
  sequence.append(
      instrumentation::build_v_and_b32_literal(work, ~(1u << 18), descriptor_archive, arch),
      instrumentation::build_v_readlane_b32(dma_descriptor, work, 0, arch),
      instrumentation::build_valu_to_salu_dependency_wait(arch));
  const auto original_offset = static_cast<uint32_t>(words.size());
  sequence.append(dma, build_sopp_encoding(arch, cdna5::kSWaitTensorcntSopp, 0));
  if (state_count == 4)
    sequence.append(instrumentation::build_v_readlane_b32(groups[1], descriptor_archive, 0, arch),
                    instrumentation::build_valu_to_salu_dependency_wait(arch));
  sequence.append(delay_words);
  append_load(false);
  for (unsigned word = 0; word < 2; ++word) {
    const auto matched = sequence.make_label();
    sequence.append(
        instrumentation::build_v_cmp_ne_u32_vcc(vr(expected + word), readback + word, arch));
    sequence.branch(matched, InstructionSequence::BranchKind::VccZero);
    sequence.append(mismatch_words);
    sequence.bind_label(matched);
  }
  // The native arrival operates per active lane. Exactly one arrival is due
  // for an active wave-wide DMA, independently of the original guest EXEC.
  const auto no_arrival = sequence.make_label();
  sequence.append(
      build_v_mov_b32_e32(work, groups[1], arch),
      instrumentation::build_v_lshrrev_b32(work, scalar_positive_inline_u32(18), work, arch),
      instrumentation::build_v_and_b32(work, one, work, arch),
      build_v_mov_b32_e32(work + 1, groups[0], arch),
      instrumentation::build_v_and_b32(work + 1, scalar_positive_inline_u32(3), work + 1, arch),
      instrumentation::build_v_cmp_eq_u32_vcc(one, work + 1, arch), select(work, zero, vr(work)),
      instrumentation::build_v_cmp_ne_u32_vcc(zero, work, arch));
  sequence.branch(no_arrival, InstructionSequence::BranchKind::VccZero);
  sequence.append(instrumentation::build_s_mov_b64(kAmdGpuExecLo, one, arch),
                  build_v_mov_b32_e32(completion_address, groups[1] + 1, arch),
                  instrumentation::build_v_and_b32_literal(completion_address, 0xffffu,
                                                           completion_address, arch),
                  instrumentation::build_v_lshlrev_b32(
                      completion_address, scalar_positive_inline_u32(3), completion_address, arch),
                  cdna5::build_vds(cdna5::kDsAtomicAsyncBarrierArriveB64Vds,
                                   {.addr = static_cast<uint8_t>(completion_address)}),
                  build_sopp_encoding(arch, cdna5::kSWaitAsynccntSopp, 0));
  sequence.bind_label(no_arrival);
  sequence.append(instrumentation::build_s_mov_b64(kAmdGpuExecLo, state_sgpr + 2, arch),
                  instrumentation::build_s_mov_b64(kAmdGpuVccLo, state_sgpr, arch));
  if (!sequence.finish(arch))
    return false;
  guest_word_offset = original_offset;
  return true;
}

bool append_select_tensor_element(std::vector<uint32_t> &words, const ProgramSite &site,
                                  uint16_t element_hash_vgpr, uint16_t iteration_hash_vgpr,
                                  uint16_t element_vgpr, uint16_t count_vgpr, uint16_t scratch_vgpr,
                                  rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA5 || site.origin != AccessOrigin::TensorLds ||
      (site.kind != LdsAccessKind::Write && site.kind != LdsAccessKind::Read) ||
      !site.operands.tensor_descriptor_sgprs || scratch_vgpr > 251u || element_vgpr == count_vgpr)
    return false;
  for (const uint16_t reg : {element_hash_vgpr, iteration_hash_vgpr, element_vgpr, count_vgpr}) {
    if (reg >= 256u || (reg >= scratch_vgpr && reg < scratch_vgpr + 5u))
      return false;
  }
  if (element_vgpr == element_hash_vgpr || element_vgpr == iteration_hash_vgpr ||
      count_vgpr == element_hash_vgpr || count_vgpr == iteration_hash_vgpr)
    return false;
  const auto &groups = *site.operands.tensor_descriptor_sgprs;
  if (groups[0] > 102u || groups[1] > 98u || (groups[2] > 102u && groups[2] != 124u) ||
      (groups[3] > 102u && groups[3] != 124u))
    return false;

  const uint16_t field = scratch_vgpr;
  const uint16_t seen = scratch_vgpr + 1u;
  const uint16_t iterate = scratch_vgpr + 2u;
  const uint16_t gather = scratch_vgpr + 3u;
  const uint16_t iterations = scratch_vgpr + 4u;
  const auto zero = scalar_positive_inline_u32(0);
  const auto one = scalar_positive_inline_u32(1);
  const auto vr = [](uint16_t reg) { return vector_source_vgpr(reg); };
  const auto binary = [&](uint16_t opcode, uint16_t dst, uint16_t lhs, uint16_t rhs) {
    return cdna5::build_vop3(opcode, {.vdst = static_cast<uint8_t>(dst), .src0 = lhs, .src1 = rhs});
  };
  const auto select = [&](uint16_t dst, uint16_t when_false, uint16_t when_true) {
    return cdna5::build_vop3(cdna5::kVCndmaskB32Vop3, {.vdst = static_cast<uint8_t>(dst),
                                                       .src0 = when_false,
                                                       .src1 = when_true,
                                                       .src2 = kAmdGpuVccLo});
  };
  InstructionSequence sequence(words);
  const auto read_word = [&](uint16_t dst, size_t group, uint16_t offset) {
    sequence.append(
        build_v_mov_b32_e32(dst, groups[group] == 124u ? zero : groups[group] + offset, arch));
  };
  read_word(gather, 0, 0);
  sequence.append(
      instrumentation::build_v_lshrrev_b32(gather, scalar_positive_inline_u32(31), gather, arch));
  read_word(iterate, 1, 0);
  sequence.append(
      instrumentation::build_v_lshrrev_b32(iterate, scalar_positive_inline_u32(19), iterate, arch),
      instrumentation::build_v_and_b32(iterate, one, iterate, arch),
      instrumentation::build_v_cmp_eq_u32_vcc(zero, gather, arch),
      select(iterate, zero, vr(iterate)), build_v_mov_b32_e32(count_vgpr, one, arch),
      build_v_mov_b32_e32(seen, zero, arch));

  // Only trailing zero dimensions are absent axes. An interior zero makes the
  // tile empty, even if lower dimensions are nonzero.
  for (int axis = 4; axis >= 0; --axis) {
    const size_t group = axis == 4 ? 3u : axis == 3 ? 2u : 1u;
    const uint16_t word = axis == 4 ? 2u : axis == 3 || axis == 0 ? 3u : 4u;
    read_word(field, group, word);
    if (axis == 1) {
      sequence.append(instrumentation::build_v_and_b32_literal(field, 0xffffu, field, arch));
    } else {
      sequence.append(
          instrumentation::build_v_lshrrev_b32(field, scalar_positive_inline_u32(16), field, arch));
    }
    if (axis == 3) {
      // D2's high halfword is an iteration count in iterate mode, not tile_dim3.
      sequence.append(instrumentation::build_v_cmp_eq_u32_vcc(zero, iterate, arch),
                      select(field, zero, vr(field)));
    }
    sequence.append(binary(cdna5::kVOrB32Vop3, seen, vr(seen), vr(field)),
                    instrumentation::build_v_cmp_eq_u32_vcc(zero, seen, arch),
                    select(field, vr(field), one),
                    binary(cdna5::kVMulLoU32Vop3, count_vgpr, vr(count_vgpr), vr(field)));
  }
  sequence.append(instrumentation::build_v_cmp_eq_u32_vcc(zero, seen, arch),
                  select(count_vgpr, vr(count_vgpr), zero));
  // Gather/scatter descriptors always describe rank-two packed LDS rows.
  read_word(field, 1, 3);
  read_word(iterations, 1, 4);
  sequence.append(
      instrumentation::build_v_lshrrev_b32(field, scalar_positive_inline_u32(16), field, arch),
      instrumentation::build_v_and_b32_literal(iterations, 0xffffu, iterations, arch),
      binary(cdna5::kVMulLoU32Vop3, field, vr(field), vr(iterations)),
      instrumentation::build_v_cmp_eq_u32_vcc(zero, gather, arch),
      select(count_vgpr, vr(field), vr(count_vgpr)));
  read_word(field, 0, 0);
  sequence.append(
      instrumentation::build_v_and_b32(field, scalar_positive_inline_u32(3), field, arch),
      instrumentation::build_v_cmp_eq_u32_vcc(one, field, arch),
      select(count_vgpr, zero, vr(count_vgpr)),
      binary(cdna5::kVMulHiU32Vop3, element_vgpr, vr(element_hash_vgpr), vr(count_vgpr)));
  read_word(iterations, 2, 3);
  sequence.append(instrumentation::build_v_lshrrev_b32(iterations, scalar_positive_inline_u32(16),
                                                       iterations, arch),
                  instrumentation::build_v_add_u32(iterations, one, iterations, arch),
                  instrumentation::build_v_cmp_eq_u32_vcc(zero, iterate, arch),
                  select(iterations, vr(iterations), one),
                  binary(cdna5::kVMulHiU32Vop3, field, vr(iteration_hash_vgpr), vr(iterations)));
  read_word(seen, 2, 1);
  sequence.append(binary(cdna5::kVMulLoU32Vop3, field, vr(field), vr(seen)),
                  instrumentation::build_v_add_u32(element_vgpr, vr(field), element_vgpr, arch));
  return sequence.finish();
}

} // namespace rocjitsu::consan::detail
