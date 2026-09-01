// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi_access_target.h"

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/consan/consan_capability_contract.h"
#include "rocjitsu/code/patch/consan/consan_moi_native_abi.h"
#include "rocjitsu/code/patch/instruction_sequence.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"

namespace rocjitsu::consan_moi_impl {

using consan_detail::range_overlaps;

[[nodiscard]] bool append_compute_effective_lds_byte_offset(std::vector<uint32_t> &words,
                                                            uint16_t dst_vgpr, uint16_t addr_vgpr,
                                                            uint32_t static_byte_offset,
                                                            rj_code_arch_t arch) {
  if (static_byte_offset == 0)
    return true;

  const auto mov_offset =
      instrumentation::build_v_mov_b32_literal(dst_vgpr, static_byte_offset, arch);
  const auto add_offset =
      instrumentation::build_v_add_u32(dst_vgpr, vector_source_vgpr(addr_vgpr), dst_vgpr, arch);
  InstructionSequence sequence(words);
  return sequence.emit_all(mov_offset, add_offset);
}
[[nodiscard]] std::optional<uint16_t>
candidate_lds_byte_offset_vgpr(const ConSanMoiCandidate &candidate,
                               std::vector<std::string> &errors) {
  if (!candidate.lowering.form) {
    errors.emplace_back("ConSan MOI probe requires a normalized access lowering form");
    return std::nullopt;
  }
  const ConSanAccessLoweringForm &form = *candidate.lowering.form;
  // Direct-to-LDS candidates materialize their lane or explicit target
  // address into dedicated scratch before this placeholder is consumed.
  if (form.kind == ConSanAccessLoweringFormKind::DirectToLdsLaneAddressed ||
      form.kind == ConSanAccessLoweringFormKind::DirectToLdsExplicitAddress) {
    return 0u;
  }
  if (!form.address_vgpr || form.address_vgpr_count == 0u) {
    errors.emplace_back("ConSan MOI probe requires an LDS address VGPR");
    return std::nullopt;
  }
  return *form.address_vgpr;
}

[[nodiscard]] bool append_materialize_direct_to_lds_address(std::vector<uint32_t> &words,
                                                            const ConSanMoiCandidate &candidate,
                                                            uint16_t result_vgpr,
                                                            uint16_t exec_save_sgpr,
                                                            rj_code_arch_t arch) {
  if (!candidate.is_direct_to_lds() || !candidate.lowering.form)
    return false;
  const ConSanAccessLoweringForm &form = *candidate.lowering.form;

  // CDNA5 VGLOBAL direct-to-LDS instructions carry the LDS byte destination
  // in an ordinary VGPR. Preserve it before relocating the guest instruction.
  if (consan_uses_gfx12_cdna_execution(arch)) {
    if (!form.address_vgpr || *form.address_vgpr >= 256u || result_vgpr >= 256u)
      return false;
    words.push_back(build_v_mov_b32_e32(result_vgpr, vector_source_vgpr(*form.address_vgpr), arch));
    return true;
  }
  if (!consan_uses_gfx9_cdna_encoding(arch))
    return false;

  const uint16_t lane_stride_shift = form.element_width_bits == 32u    ? 2u
                                     : form.element_width_bits == 96u  ? 4u
                                     : form.element_width_bits == 128u ? 4u
                                                                       : 0u;
  if (lane_stride_shift == 0u)
    return false;
  const auto save_exec = instrumentation::build_s_mov_b64(exec_save_sgpr, kAmdGpuExecLo, arch);
  const auto activate_all_lanes =
      instrumentation::build_s_mov_b64(kAmdGpuExecLo, kScalarInlineNegativeOneOperand, arch);
  const auto lane_lo = instrumentation::build_v_mbcnt_lo_u32_b32(
      result_vgpr, kScalarInlineNegativeOneOperand, scalar_positive_inline_u32(0u), arch);
  const auto lane_hi = instrumentation::build_v_mbcnt_hi_u32_b32(
      result_vgpr, kScalarInlineNegativeOneOperand, vector_source_vgpr(result_vgpr), arch);
  const auto scale = instrumentation::build_v_lshlrev_b32(
      result_vgpr, scalar_positive_inline_u32(lane_stride_shift), result_vgpr, arch);
  const auto add =
      instrumentation::build_v_add_u32(result_vgpr, scalar_operand_m0(arch), result_vgpr, arch);
  const auto restore_exec = instrumentation::build_s_mov_b64(kAmdGpuExecLo, exec_save_sgpr, arch);
  if (!save_exec || !activate_all_lanes || !lane_lo || !lane_hi || !scale || !add ||
      !restore_exec) {
    return false;
  }
  words.push_back(*save_exec);
  words.push_back(*activate_all_lanes);
  words.insert(words.end(), lane_lo->begin(), lane_lo->end());
  words.insert(words.end(), lane_hi->begin(), lane_hi->end());
  words.push_back(*scale);
  words.insert(words.end(), add->begin(), add->end());
  words.push_back(*restore_exec);
  return true;
}

[[nodiscard]] bool candidate_uses_scalar_vector_flat_address(const ConSanMoiCandidate &candidate) {
  return candidate.lowering.form &&
         candidate.lowering.form->kind == ConSanAccessLoweringFormKind::FlatScalarVectorAddress;
}

[[nodiscard]] bool
candidate_requires_flat_address_materialization(const ConSanMoiCandidate &candidate) {
  return candidate.lowering.form &&
         (candidate.lowering.form->kind == ConSanAccessLoweringFormKind::FlatVectorAddress ||
          candidate.lowering.form->kind == ConSanAccessLoweringFormKind::FlatScalarVectorAddress) &&
         (candidate_uses_scalar_vector_flat_address(candidate) ||
          candidate.lowering.form->immediate_byte_offset.value_or(0) != 0);
}

[[nodiscard]] uint16_t flat_access_address_scratch_count(const ConSanMoiCandidate &candidate) {
  if (!candidate_requires_flat_address_materialization(candidate))
    return 0u;
  // A scalar-base VFLAT uses one temporary in addition to the resulting pair.
  // The temporary preserves a load-clobbered vector offset and also carries a
  // SCALE_OFFSET product without overwriting the scalar base.
  return candidate_uses_scalar_vector_flat_address(candidate) ? 3u : 2u;
}

/// Materialize the exact gfx12 VFLAT address calculation used by the ISA:
/// either a vector pair plus signed IOFFSET, or
///   saddr + zero_extend(vaddr * scale) + signed IOFFSET.
/// For group pointers the low result word is the LDS byte offset consumed by
/// ConSan, while retaining the high word lets maybe-group probes perform their
/// ordinary aperture check. The caller preserves VCC/SCC around this sequence.
[[nodiscard]] bool append_materialize_flat_access_address(std::vector<uint32_t> &words,
                                                          const ConSanMoiCandidate &candidate,
                                                          uint16_t input_vgpr, uint16_t result_vgpr,
                                                          rj_code_arch_t arch) {
  const ConSanAccessLoweringForm *form =
      candidate.lowering.form ? &*candidate.lowering.form : nullptr;
  const uint16_t scratch_count = flat_access_address_scratch_count(candidate);
  if (form == nullptr || scratch_count == 0u || !form->address_vgpr || input_vgpr >= 255u ||
      result_vgpr >= 255u || static_cast<uint32_t>(result_vgpr) + scratch_count > 256u)
    return false;

  if (candidate_uses_scalar_vector_flat_address(candidate)) {
    if (!form->scalar_address_sgpr || candidate.width_bits() == 0u)
      return false;
    const uint16_t offset_vgpr = static_cast<uint16_t>(result_vgpr + 2u);
    if (form->scale_immediate) {
      const uint32_t byte_count = (candidate.width_bits() + 7u) / 8u;
      const auto scale = instrumentation::build_v_mul_lo_u32_literal(offset_vgpr, offset_vgpr,
                                                                     byte_count, input_vgpr, arch);
      if (!scale)
        return false;
      words.insert(words.end(), scale->begin(), scale->end());
    } else {
      words.push_back(build_v_mov_b32_e32(offset_vgpr, vector_source_vgpr(input_vgpr), arch));
    }
    words.push_back(build_v_mov_b32_e32(result_vgpr, *form->scalar_address_sgpr, arch));
    words.push_back(build_v_mov_b32_e32(static_cast<uint16_t>(result_vgpr + 1u),
                                        static_cast<uint16_t>(*form->scalar_address_sgpr + 1u),
                                        arch));
    const auto add_vector_offset =
        instrumentation::build_v_add_u64_vgpr_offset(result_vgpr, offset_vgpr, arch);
    if (!add_vector_offset)
      return false;
    words.insert(words.end(), add_vector_offset->begin(), add_vector_offset->end());
  } else {
    words.push_back(build_v_mov_b32_e32(result_vgpr, vector_source_vgpr(input_vgpr), arch));
    words.push_back(build_v_mov_b32_e32(static_cast<uint16_t>(result_vgpr + 1u),
                                        vector_source_vgpr(static_cast<uint16_t>(input_vgpr + 1u)),
                                        arch));
  }
  if (form->immediate_byte_offset.value_or(0) != 0) {
    const auto add = instrumentation::build_v_add_u64_signed_i24(
        result_vgpr, *form->immediate_byte_offset, arch);
    if (!add)
      return false;
    words.insert(words.end(), add->begin(), add->end());
  }
  return true;
}

[[nodiscard]] uint16_t candidate_payload_vgpr_count(const ConSanMoiCandidate &candidate) {
  return candidate.lowering.form ? candidate.lowering.form->data_register_count : 0u;
}

[[nodiscard]] bool moi_load_clobbers_address(const ConSanMoiCandidate &candidate) {
  const ConSanAccessLoweringForm *form =
      candidate.lowering.form ? &*candidate.lowering.form : nullptr;
  if (form == nullptr || !form->address_vgpr || !form->destination_vgpr)
    return false;
  return range_overlaps(*form->address_vgpr, form->address_vgpr_count, *form->destination_vgpr,
                        form->destination_register_count);
}

[[nodiscard]] bool
moi_access_requires_high_bank_address_capture(const ConSanMoiCandidate &candidate,
                                              rj_code_arch_t arch) {
  // gfx1250 DS encodings carry only the low eight bits of each VGPR operand.
  // The current s_set_vgpr_msb mode supplies the high bank independently for
  // SRC0, SRC1, SRC2, and DST. Native LDS addresses are SRC0, so a nonzero
  // SRC0 bank must be copied while that bank is still selected. The appended
  // probe subsequently selects bank zero for its own scratch registers.
  const ConSanAccessLoweringForm *form =
      candidate.lowering.form ? &*candidate.lowering.form : nullptr;
  return consan_arch_has_selectable_vgpr_bank(arch) && form != nullptr &&
         form->kind != ConSanAccessLoweringFormKind::FlatVectorAddress &&
         form->kind != ConSanAccessLoweringFormKind::FlatScalarVectorAddress &&
         form->address_vgpr && (candidate.incoming_vgpr_bank_mode.value_or(0u) & 0x3u) != 0u;
}

[[nodiscard]] bool reject_candidate_scratch_range_overlap(const ConSanMoiCandidate &candidate,
                                                          uint16_t scratch_vgpr,
                                                          uint16_t scratch_count,
                                                          std::vector<std::string> &errors) {
  if (!candidate.lowering.form) {
    errors.emplace_back("ConSan MOI probe requires a normalized access lowering form");
    return true;
  }
  const ConSanAccessLoweringForm &form = *candidate.lowering.form;
  if (form.address_vgpr && form.address_vgpr_count != 0u &&
      range_overlaps(*form.address_vgpr, form.address_vgpr_count, scratch_vgpr, scratch_count)) {
    errors.emplace_back("ConSan MOI probe scratch VGPRs overlap the LDS address VGPRs");
    return true;
  }
  if (form.destination_vgpr && form.destination_register_count != 0u &&
      range_overlaps(*form.destination_vgpr, form.destination_register_count, scratch_vgpr,
                     scratch_count)) {
    errors.emplace_back("ConSan MOI probe scratch VGPRs overlap the destination VGPRs");
    return true;
  }
  const uint16_t data_vgpr_count =
      form.second_data_vgpr ? form.element_register_count : form.data_register_count;
  if (form.data_vgpr &&
      range_overlaps(*form.data_vgpr, data_vgpr_count, scratch_vgpr, scratch_count)) {
    errors.emplace_back("ConSan MOI probe scratch VGPRs overlap the data VGPRs");
    return true;
  }
  if (form.second_data_vgpr &&
      range_overlaps(*form.second_data_vgpr, data_vgpr_count, scratch_vgpr, scratch_count)) {
    errors.emplace_back("ConSan MOI probe scratch VGPRs overlap the second data VGPRs");
    return true;
  }
  return false;
}

} // namespace rocjitsu::consan_moi_impl
