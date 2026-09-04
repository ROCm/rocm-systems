// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi_runtime_workgroup_gate.h"

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/consan/consan_moi_relocation.h"
#include "rocjitsu/code/patch/consan/consan_moi_report_emission.h"
#include "rocjitsu/code/patch/instruction_sequence.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"

#include <bit>
#include <cstring>

namespace rocjitsu::consan_moi_impl {

std::optional<ConSanMoiWorkgroupSource> moi_runtime_workgroup_selection_source() {
  // A cached Record/Replay selection needs storage distinct from the exact
  // workgroup tuple. Until a dedicated cache slot is assigned, retain the
  // tuple and compute the gate at each probe.
  return std::nullopt;
}

bool moi_has_probe_entry_runtime_workgroup_gate(
    const ConSanMoiWorkgroupSources &workgroup_sources) {
  const auto register_backed = [](const ConSanMoiWorkgroupSource &source) {
    return source.scalar_src.has_value() || source.vector_src.has_value();
  };
  const bool register_backed_tuple = register_backed(workgroup_sources.x) &&
                                     register_backed(workgroup_sources.y) &&
                                     register_backed(workgroup_sources.z) &&
                                     (!workgroup_sources.cluster_workgroup_id.has_value() ||
                                      register_backed(workgroup_sources.cluster_workgroup_id));
  return moi_runtime_workgroup_selection_source().has_value() || register_backed_tuple;
}

std::optional<MoiRuntimeWorkgroupGatePlan>
plan_moi_runtime_workgroup_gate(const MoiRuntimeWorkgroupGateInputs &inputs,
                                const ConSanMoiWorkgroupSources &workgroup_sources) {
  const auto cached_selection = moi_runtime_workgroup_selection_source();
  if (inputs.sample_stride <= 1u ||
      !moi_has_probe_entry_runtime_workgroup_gate(workgroup_sources)) {
    return std::nullopt;
  }
  return MoiRuntimeWorkgroupGatePlan{
      .exec_save_sgpr = inputs.exec_save_sgpr,
      .sample_stride = inputs.sample_stride,
      .sample_offset = inputs.sample_offset,
      .flavor = inputs.flavor,
      .cached_selection = cached_selection,
      .dispatch_id_sgpr = inputs.dispatch_id_sgpr,
      .literal_dispatch_id = inputs.literal_dispatch_id,
      .direct_call_form = inputs.direct_call_form,
  };
}

using consan_moi_detail::append_moi_scc_preserving_indirect_jump;

[[nodiscard]] bool append_moi_runtime_workgroup_residue_compare(std::vector<uint32_t> &words,
                                                                uint16_t residue_sgpr,
                                                                uint16_t temporary_sgpr,
                                                                uint32_t selected_residue,
                                                                rj_code_arch_t arch) {
  InstructionSequence sequence(words);
  if (selected_residue <= 64u) {
    return sequence.emit(instrumentation::build_s_cmp_eq_u32(
        residue_sgpr, scalar_positive_inline_u32(selected_residue), arch));
  }
  constexpr uint16_t kScalarLiteralSource = 255u;
  return sequence.emit_all(build_s_mov_b32(temporary_sgpr, kScalarLiteralSource, arch),
                           selected_residue,
                           instrumentation::build_s_cmp_eq_u32(residue_sgpr, temporary_sgpr, arch));
}

uint64_t moi_runtime_workgroup_gate_reserved_words(uint32_t guest_byte_count,
                                                   bool has_cluster_workgroup_id) {
  // A complete persistent vector tuple needs two more words per coordinate
  // than the scalar form: v_readfirstlane plus its VALU-to-SALU dependency
  // wait. Keep the envelope exact enough that large generated objects do not
  // push otherwise-reachable relays beyond SOPP branch range.
  // The full-tuple Record/Replay gate also restores the saved SCC on its
  // selected path. Account for that word in addition to the common XYZ gate
  // and indirect-return envelope. The previous 26-word bound happened to fit
  // scalar tuples, but rejected a valid vector XYZ+cluster tuple at emission.
  constexpr uint64_t kXyzGateWordsWithoutGuest = 27u;
  constexpr uint64_t kVectorClusterMixWords = 3u;
  return kXyzGateWordsWithoutGuest + (has_cluster_workgroup_id ? kVectorClusterMixWords : 0u) +
         (static_cast<uint64_t>(guest_byte_count) + sizeof(uint32_t) - 1u) / sizeof(uint32_t);
}

[[nodiscard]] bool append_moi_runtime_workgroup_mix(std::vector<uint32_t> &words,
                                                    const ConSanMoiWorkgroupSource &source,
                                                    uint16_t quotient, uint16_t residue,
                                                    rj_code_arch_t arch) {
  if (!source.is_well_formed())
    return false;
  if (!source.has_value())
    return true;
  if (source.private_offset) {
    // The gate runs before a spill-backed probe saves its guest-overlapping
    // scratch window. Loading private state here would require clobbering a
    // VGPR before that save, so callers must select a different lowering.
    return false;
  }

  InstructionSequence sequence(words);
  uint16_t coordinate = 0;
  if (source.scalar_src) {
    coordinate = *source.scalar_src;
  } else if (source.vector_src) {
    if (!sequence.emit_all(
            instrumentation::build_v_readfirstlane_b32(quotient, *source.vector_src, arch),
            instrumentation::build_valu_to_salu_dependency_wait(arch)))
      return false;
    coordinate = quotient;
  } else {
    return false;
  }

  if (source.mask_low_16) {
    (void)sequence.emit_all(
        build_s_lshl_b32(quotient, coordinate, scalar_positive_inline_u32(16), arch),
        build_s_lshr_b32(quotient, quotient, scalar_positive_inline_u32(16), arch));
    coordinate = quotient;
  } else if (source.shift_right_16) {
    (void)sequence.emit(
        build_s_lshr_b32(quotient, coordinate, scalar_positive_inline_u32(16), arch));
    coordinate = quotient;
  }
  return sequence.emit(instrumentation::build_s_sub_u32(residue, residue, coordinate, arch));
}

/// Emit the selection predicate shared by the inline-island and direct-call
/// gate layouts. The caller owns only the surrounding return geometry and
/// chooses registers that do not overlap its live continuation state.
[[nodiscard]] bool append_moi_runtime_workgroup_predicate(
    std::vector<uint32_t> &words, const MoiRuntimeWorkgroupGatePlan &plan,
    const ConSanMoiWorkgroupSources &workgroup_sources, uint16_t saved_scc, uint16_t quotient,
    uint16_t residue, rj_code_arch_t arch) {
  if (plan.sample_stride <= 1u)
    return false;
  InstructionSequence sequence(words);
  if (!sequence.emit(instrumentation::build_s_cselect_b32(saved_scc, scalar_positive_inline_u32(1),
                                                          scalar_positive_inline_u32(0), arch)))
    return false;
  if (plan.cached_selection) {
    uint16_t selected_source = 0u;
    if (plan.cached_selection->scalar_src) {
      selected_source = *plan.cached_selection->scalar_src;
    } else if (plan.cached_selection->vector_src) {
      if (!sequence.emit_all(instrumentation::build_v_readfirstlane_b32(
                                 quotient, *plan.cached_selection->vector_src, arch),
                             instrumentation::build_valu_to_salu_dependency_wait(arch)))
        return false;
      selected_source = quotient;
    } else {
      return false;
    }
    return sequence.emit(
        instrumentation::build_s_cmp_lg_u32(selected_source, scalar_positive_inline_u32(0u), arch));
  }

  if (plan.flavor == MoiRuntimeWorkgroupGatePlan::Flavor::RecordReplay) {
    // Record/Replay samples stable workgroup coordinates. Dispatch identity
    // remains in every record's attribution but does not rotate the selected
    // workgroup or make offset zero omit workgroup zero.
    (void)sequence.emit(build_s_mov_b32(residue, scalar_positive_inline_u32(0u), arch));
  } else {
    const uint16_t dispatch_mix = plan.dispatch_id_sgpr.value_or(
        scalar_positive_inline_u32(static_cast<uint32_t>(plan.literal_dispatch_id) & 63u));
    if (!sequence.emit(instrumentation::build_s_sub_u32(residue, scalar_positive_inline_u32(0),
                                                        dispatch_mix, arch)))
      return false;
  }
  if (!append_moi_runtime_workgroup_mix(words, workgroup_sources.x, quotient, residue, arch) ||
      !append_moi_runtime_workgroup_mix(words, workgroup_sources.y, quotient, residue, arch) ||
      !append_moi_runtime_workgroup_mix(words, workgroup_sources.z, quotient, residue, arch) ||
      !append_moi_runtime_workgroup_mix(words, workgroup_sources.cluster_workgroup_id, quotient,
                                        residue, arch))
    return false;
  const uint16_t shift = scalar_positive_inline_u32(std::countr_zero(plan.sample_stride));
  if (!sequence.emit_all(build_s_lshr_b32(quotient, residue, shift, arch),
                         build_s_lshl_b32(quotient, quotient, shift, arch),
                         instrumentation::build_s_sub_u32(residue, residue, quotient, arch)))
    return false;
  return append_moi_runtime_workgroup_residue_compare(
      words, residue, quotient, plan.sample_offset & (plan.sample_stride - 1u), arch);
}

std::optional<MoiRuntimeWorkgroupGatePrefix>
build_moi_runtime_workgroup_gate_prefix(const MoiRuntimeWorkgroupGatePlan &plan,
                                        const ConSanMoiWorkgroupSources &workgroup_sources,
                                        rj_code_arch_t arch) {
  if (plan.sample_stride <= 1u)
    return std::nullopt;
  const uint16_t saved_scc = static_cast<uint16_t>(plan.exec_save_sgpr + 4u);
  const uint16_t quotient = static_cast<uint16_t>(plan.exec_save_sgpr + 5u);
  const uint16_t residue = static_cast<uint16_t>(plan.exec_save_sgpr + 6u);
  MoiRuntimeWorkgroupGatePrefix result;
  if (!append_moi_runtime_workgroup_predicate(result.words, plan, workgroup_sources, saved_scc,
                                              quotient, residue, arch))
    return std::nullopt;
  InstructionSequence sequence(result.words);
  const InstructionSequence::Label selected = sequence.make_label();
  if (!sequence.emit_branch(selected, InstructionSequence::BranchKind::SccNonzero) ||
      !sequence.emit(
          instrumentation::build_s_cmp_lg_u32(saved_scc, scalar_positive_inline_u32(0), arch)))
    return std::nullopt;
  result.bypass_branch_word = static_cast<uint32_t>(result.words.size());
  result.words.push_back(0u);
  if (!sequence.bind(selected) ||
      !sequence.emit(
          instrumentation::build_s_cmp_lg_u32(saved_scc, scalar_positive_inline_u32(0), arch)) ||
      !sequence.resolve_branches(arch))
    return std::nullopt;
  return result;
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_moi_runtime_workgroup_gate_call_words(
    std::span<const uint8_t> guest_bytes, const MoiRuntimeWorkgroupGatePlan &plan,
    const ConSanMoiWorkgroupSources &workgroup_sources, uint64_t gate_text_offset,
    uint64_t return_text_offset, uint64_t reserved_word_count, rj_code_arch_t arch) {
  if (plan.sample_stride <= 1u)
    return std::nullopt;
  const uint16_t saved_scc = static_cast<uint16_t>(plan.exec_save_sgpr + 4u);
  const uint16_t quotient = static_cast<uint16_t>(plan.exec_save_sgpr + 5u);
  // Dense CDNA5 Record/Replay callers keep their s_call_i64 return PC in
  // base+6:base+7 until the body returns with s_setpc. The indirect-PC pair at
  // base+0 is dead after the dispatcher reaches this gate and is reinitialized
  // by the non-selected return jump or the selected body, so reuse its low
  // word for that engine without destroying the live call return. Sampled
  // returns with its ordinary indirect-PC pair and retains the established
  // residue slot.
  const uint16_t residue = plan.flavor == MoiRuntimeWorkgroupGatePlan::Flavor::RecordReplay &&
                                   plan.direct_call_form == ConSanDirectCallForm::SCallI64
                               ? plan.exec_save_sgpr
                               : static_cast<uint16_t>(plan.exec_save_sgpr + 6u);

  std::vector<uint32_t> words;
  words.reserve(reserved_word_count);
  InstructionSequence sequence(words);
  if (!append_moi_runtime_workgroup_predicate(words, plan, workgroup_sources, saved_scc, quotient,
                                              residue, arch))
    return std::nullopt;
  const InstructionSequence::Label selected = sequence.make_label();
  if (!sequence.emit_branch(selected, InstructionSequence::BranchKind::SccNonzero) ||
      !sequence.emit(
          instrumentation::build_s_cmp_lg_u32(saved_scc, scalar_positive_inline_u32(0), arch)))
    return std::nullopt;
  for (uint64_t offset = 0; offset < guest_bytes.size(); offset += sizeof(uint32_t)) {
    uint32_t word = 0;
    std::memcpy(&word, guest_bytes.data() + offset, sizeof(word));
    (void)sequence.emit(word);
  }
  if (!append_moi_scc_preserving_indirect_jump(words, gate_text_offset, return_text_offset,
                                               plan.exec_save_sgpr, saved_scc,
                                               /*capture_scc=*/false, arch))
    return std::nullopt;
  if (!sequence.bind(selected) ||
      !sequence.emit(
          instrumentation::build_s_cmp_lg_u32(saved_scc, scalar_positive_inline_u32(0), arch)) ||
      !sequence.resolve_branches(arch) || words.size() > reserved_word_count)
    return std::nullopt;
  words.resize(reserved_word_count, build_s_nop(0, arch));
  return words;
}

} // namespace rocjitsu::consan_moi_impl
