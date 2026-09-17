// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_access_target.h"

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/consan/consan_capability_contract.h"
#include "rocjitsu/code/patch/consan/consan_native_abi.h"
#include "rocjitsu/code/patch/instruction_sequence.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"

namespace rocjitsu::consan::detail {

using detail::range_overlaps;

AccessResourceFacts resolve_access_resource_facts(const OperatingPoint &point,
                                                  const Candidate &candidate, rj_code_arch_t arch) {
  const TargetProfile *target = target_profile(arch);
  const AccessCapability capability = target ? target->access : AccessCapability{};
  const uint16_t flat_address_scratch_count = flat_access_address_scratch_count(candidate);
  const bool needs_address_capture = flat_address_scratch_count != 0u ||
                                     candidate.is_direct_to_lds() ||
                                     load_clobbers_address(candidate) ||
                                     access_requires_high_bank_address_capture(candidate, arch);
  const bool requires_flat_materialization =
      candidate_requires_flat_address_materialization(candidate);
  return {
      .address_scratch_vgpr_count = flat_address_scratch_count != 0u
                                        ? flat_address_scratch_count
                                        : static_cast<uint16_t>(needs_address_capture),
      .two_address_replay_vgpr_count = static_cast<uint16_t>(
          target && target->requires_split_two_address_lds_relocation &&
          candidate.is_native_two_range() && candidate.encoded_offset_scale_bytes() > 8u),
      .dynamic_stack_reservoir_vgpr_count = static_cast<uint16_t>(
          capability.dynamic_stack_uses_scalar_reservoir && point.dynamic_stack_spill
              ? DynamicStackBorrowedSgprSpillSequence::kScalarReservoirCount
              : 0u),
      .has_exec_save = point.exec_save_sgpr.has_value(),
      .initialize_owner_epoch = point.initialize_owner_epoch,
      .has_persistent_owner_vgpr = point.owner_epoch_vgprs.owner().has_value(),
      .uses_private_epoch = point.automatic_private_epoch,
      .has_complete_persistent_sgprs = point.persistent_sgprs.complete(),
      .target_available = target != nullptr,
      .supports_native_lds_spill_recovery = capability.native_lds_spill_recovery &&
                                            candidate.is_native_lds() &&
                                            !requires_flat_materialization,
      .supports_clobbered_address_spill_reload =
          capability.clobbered_address_spill_reload && !candidate.is_flat() &&
          load_clobbers_address(candidate) && !requires_flat_materialization,
      .guest_replay_requires_disjoint_address_scratch =
          target && target->requires_split_two_address_lds_relocation &&
          candidate.is_native_two_range() && candidate.encoded_offset_scale_bytes() > 8u,
  };
}

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
candidate_lds_byte_offset_vgpr(const Candidate &candidate, std::vector<std::string> &errors) {
  if (!candidate.site().lowering.form) {
    errors.emplace_back("ConSan probe requires a normalized access lowering form");
    return std::nullopt;
  }
  const AccessLoweringForm &form = *candidate.site().lowering.form;
  // Direct-to-LDS candidates materialize their lane or explicit target
  // address into dedicated scratch before this placeholder is consumed.
  if (form.kind == AccessLoweringFormKind::DirectToLdsLaneAddressed ||
      form.kind == AccessLoweringFormKind::DirectToLdsExplicitAddress) {
    return 0u;
  }
  if (!form.address_vgpr || form.address_vgpr_count == 0u) {
    errors.emplace_back("ConSan probe requires an LDS address VGPR");
    return std::nullopt;
  }
  return *form.address_vgpr;
}

[[nodiscard]] bool
append_materialize_direct_to_lds_address(std::vector<uint32_t> &words, const ProgramSite &site,
                                         uint16_t result_vgpr, uint16_t temporary_vgpr,
                                         uint16_t /*exec_save_sgpr*/, rj_code_arch_t arch) {
  if (!site.lowering.form)
    return false;
  const AccessLoweringForm &form = *site.lowering.form;

  if (form.kind == AccessLoweringFormKind::DirectToLdsExplicitAddress) {
    if (!form.address_vgpr || *form.address_vgpr >= 256u || result_vgpr >= 256u)
      return false;
    words.push_back(build_v_mov_b32_e32(result_vgpr, vector_source_vgpr(*form.address_vgpr), arch));
    return true;
  }
  if (form.kind != AccessLoweringFormKind::DirectToLdsLaneAddressed)
    return false;

  const uint16_t lane_stride_shift = form.element_width_bits == 32u    ? 2u
                                     : form.element_width_bits == 96u  ? 4u
                                     : form.element_width_bits == 128u ? 4u
                                                                       : 0u;
  if (lane_stride_shift == 0u)
    return false;
  // MBCNT counts the explicit all-ones source mask, so it already produces
  // physical lane IDs under partial EXEC. Expanding EXEC would overwrite
  // inactive spill victims that were never saved by the caller.
  const auto lane_lo = instrumentation::build_v_mbcnt_lo_u32_b32(
      result_vgpr, kScalarInlineNegativeOneOperand, scalar_positive_inline_u32(0u), arch);
  const auto lane_hi = instrumentation::build_v_mbcnt_hi_u32_b32(
      result_vgpr, kScalarInlineNegativeOneOperand, vector_source_vgpr(result_vgpr), arch);
  const auto scale = instrumentation::build_v_lshlrev_b32(
      result_vgpr, scalar_positive_inline_u32(lane_stride_shift), result_vgpr, arch);
  if (temporary_vgpr == result_vgpr || temporary_vgpr >= 256u)
    return false;
  const auto mask = instrumentation::build_v_and_b32_literal(
      temporary_vgpr, form.direct_m0_address_mask, temporary_vgpr, arch);
  const auto add = instrumentation::build_v_add_u32(result_vgpr, vector_source_vgpr(temporary_vgpr),
                                                    result_vgpr, arch);
  const auto add_offset = instrumentation::build_v_add_u32(
      result_vgpr, vector_source_vgpr(temporary_vgpr), result_vgpr, arch);
  const auto offset = instrumentation::build_v_mov_b32_literal(
      temporary_vgpr, static_cast<uint32_t>(form.immediate_byte_offset.value_or(0)), arch);
  if (!lane_lo || !lane_hi || !scale || !mask || !add || !add_offset || !offset) {
    return false;
  }
  words.insert(words.end(), lane_lo->begin(), lane_lo->end());
  words.insert(words.end(), lane_hi->begin(), lane_hi->end());
  words.push_back(*scale);
  words.push_back(build_v_mov_b32_e32(temporary_vgpr, scalar_operand_m0(arch), arch));
  words.insert(words.end(), mask->begin(), mask->end());
  words.insert(words.end(), add->begin(), add->end());
  if (form.immediate_byte_offset.value_or(0) != 0) {
    words.insert(words.end(), offset->begin(), offset->end());
    words.insert(words.end(), add_offset->begin(), add_offset->end());
  }
  return true;
}

[[nodiscard]] bool candidate_uses_scalar_vector_flat_address(const Candidate &candidate) {
  return candidate.site().lowering.form &&
         candidate.site().lowering.form->kind == AccessLoweringFormKind::FlatScalarVectorAddress;
}

[[nodiscard]] bool candidate_requires_flat_address_materialization(const Candidate &candidate) {
  return candidate.site().lowering.form &&
         (candidate.site().lowering.form->kind == AccessLoweringFormKind::FlatVectorAddress ||
          candidate.site().lowering.form->kind ==
              AccessLoweringFormKind::FlatScalarVectorAddress) &&
         (candidate_uses_scalar_vector_flat_address(candidate) ||
          candidate.site().lowering.form->immediate_byte_offset.value_or(0) != 0);
}

[[nodiscard]] uint16_t flat_access_address_scratch_count(const Candidate &candidate) {
  if (!candidate_requires_flat_address_materialization(candidate))
    return 0u;
  // A scalar-base VFLAT uses one temporary in addition to the resulting pair.
  // The temporary preserves a load-clobbered vector offset and also carries a
  // SCALE_OFFSET product without overwriting the scalar base.
  return candidate_uses_scalar_vector_flat_address(candidate) ? 3u : 2u;
}

/// Materialize the exact RDNA4/CDNA5 VFLAT address calculation used by the ISA:
/// either a vector pair plus signed IOFFSET, or
///   saddr + zero_extend(vaddr * scale) + signed IOFFSET.
/// For group pointers the low result word is the LDS byte offset consumed by
/// ConSan, while retaining the high word lets maybe-group probes perform their
/// ordinary aperture check. The caller preserves VCC/SCC around this sequence.
[[nodiscard]] bool append_materialize_flat_access_address(std::vector<uint32_t> &words,
                                                          const Candidate &candidate,
                                                          uint16_t input_vgpr, uint16_t result_vgpr,
                                                          rj_code_arch_t arch) {
  const AccessLoweringForm *form =
      candidate.site().lowering.form ? &*candidate.site().lowering.form : nullptr;
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

[[nodiscard]] uint16_t candidate_payload_vgpr_count(const Candidate &candidate) {
  return candidate.site().lowering.form ? candidate.site().lowering.form->data_register_count : 0u;
}

[[nodiscard]] bool load_clobbers_address(const Candidate &candidate) {
  const AccessLoweringForm *form =
      candidate.site().lowering.form ? &*candidate.site().lowering.form : nullptr;
  if (form == nullptr || !form->address_vgpr || !form->destination_vgpr)
    return false;
  return range_overlaps(*form->address_vgpr, form->address_vgpr_count, *form->destination_vgpr,
                        form->destination_register_count);
}

[[nodiscard]] bool access_requires_high_bank_address_capture(const Candidate &candidate,
                                                             rj_code_arch_t arch) {
  // CDNA5 DS encodings carry only the low eight bits of each VGPR operand.
  // The current s_set_vgpr_msb mode supplies the high bank independently for
  // SRC0, SRC1, SRC2, and DST. Native LDS addresses are SRC0, so a nonzero
  // SRC0 bank must be copied while that bank is still selected. The appended
  // probe subsequently selects bank zero for its own scratch registers.
  const AccessLoweringForm *form =
      candidate.site().lowering.form ? &*candidate.site().lowering.form : nullptr;
  const TargetProfile *target = target_profile(arch);
  return target && target->has_selectable_vgpr_bank && form != nullptr &&
         form->kind != AccessLoweringFormKind::FlatVectorAddress &&
         form->kind != AccessLoweringFormKind::FlatScalarVectorAddress && form->address_vgpr &&
         (candidate.incoming_vgpr_bank_mode.value_or(0u) & 0x3u) != 0u;
}

[[nodiscard]] bool reject_candidate_scratch_range_overlap(const Candidate &candidate,
                                                          uint16_t scratch_vgpr,
                                                          uint16_t scratch_count,
                                                          std::vector<std::string> &errors) {
  if (!candidate.site().lowering.form) {
    errors.emplace_back("ConSan probe requires a normalized access lowering form");
    return true;
  }
  const AccessLoweringForm &form = *candidate.site().lowering.form;
  if (form.address_vgpr && form.address_vgpr_count != 0u &&
      range_overlaps(*form.address_vgpr, form.address_vgpr_count, scratch_vgpr, scratch_count)) {
    errors.emplace_back("ConSan probe scratch VGPRs overlap the LDS address VGPRs");
    return true;
  }
  if (form.destination_vgpr && form.destination_register_count != 0u &&
      range_overlaps(*form.destination_vgpr, form.destination_register_count, scratch_vgpr,
                     scratch_count)) {
    errors.emplace_back("ConSan probe scratch VGPRs overlap the destination VGPRs");
    return true;
  }
  const uint16_t data_vgpr_count =
      form.second_data_vgpr ? form.element_register_count : form.data_register_count;
  if (form.data_vgpr &&
      range_overlaps(*form.data_vgpr, data_vgpr_count, scratch_vgpr, scratch_count)) {
    errors.emplace_back("ConSan probe scratch VGPRs overlap the data VGPRs");
    return true;
  }
  if (form.second_data_vgpr &&
      range_overlaps(*form.second_data_vgpr, data_vgpr_count, scratch_vgpr, scratch_count)) {
    errors.emplace_back("ConSan probe scratch VGPRs overlap the second data VGPRs");
    return true;
  }
  return false;
}

} // namespace rocjitsu::consan::detail
