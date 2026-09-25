// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "rocjitsu/code/patch/consan/consan_tensor_access.h"

#include "rocjitsu/code/builders/instruction_builder.h"
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

bool append_select_tensor_load_element(std::vector<uint32_t> &words, const ProgramSite &site,
                                       uint16_t element_hash_vgpr, uint16_t iteration_hash_vgpr,
                                       uint16_t element_vgpr, uint16_t count_vgpr,
                                       uint16_t scratch_vgpr, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA5 || site.origin != AccessOrigin::TensorLds ||
      site.kind != LdsAccessKind::Write || !site.operands.tensor_descriptor_sgprs ||
      scratch_vgpr > 251u || element_vgpr == count_vgpr)
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
