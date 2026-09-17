// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_runtime_workgroup_gate.h"

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/consan/consan_relocation.h"
#include "rocjitsu/code/patch/consan/consan_report_emission.h"
#include "rocjitsu/code/patch/instruction_sequence.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"

#include <bit>
#include <cstring>
#include <limits>

namespace rocjitsu::consan::detail {

bool has_probe_entry_runtime_workgroup_gate(const WorkgroupSources &workgroup_sources) {
  const auto register_backed = [](const WorkgroupSource &source) {
    return source.scalar_src.has_value() || source.vector_src.has_value();
  };
  const bool register_backed_tuple = register_backed(workgroup_sources.x) &&
                                     register_backed(workgroup_sources.y) &&
                                     register_backed(workgroup_sources.z) &&
                                     (!workgroup_sources.cluster_workgroup_id.has_value() ||
                                      register_backed(workgroup_sources.cluster_workgroup_id));
  return register_backed_tuple;
}

std::optional<RuntimeWorkgroupGatePlan>
plan_runtime_workgroup_gate(const RuntimeWorkgroupGatePlan &plan,
                            const WorkgroupSources &workgroup_sources) {
  if (plan.sample_stride <= 1u || !has_probe_entry_runtime_workgroup_gate(workgroup_sources)) {
    return std::nullopt;
  }
  return plan;
}

[[nodiscard]] bool append_runtime_workgroup_residue_compare(std::vector<uint32_t> &words,
                                                            uint16_t residue_sgpr,
                                                            uint16_t temporary_sgpr,
                                                            uint32_t selected_residue,
                                                            rj_code_arch_t arch) {
  InstructionSequence sequence(words);
  if (selected_residue <= 64u) {
    sequence.append(instrumentation::build_s_cmp_eq_u32(
        residue_sgpr, scalar_positive_inline_u32(selected_residue), arch));
  } else {
    constexpr uint16_t kScalarLiteralSource = 255u;
    sequence.append(build_s_mov_b32(temporary_sgpr, kScalarLiteralSource, arch), selected_residue,
                    instrumentation::build_s_cmp_eq_u32(residue_sgpr, temporary_sgpr, arch));
  }
  return sequence.finish();
}

[[nodiscard]] bool append_runtime_workgroup_mix(std::vector<uint32_t> &words,
                                                const WorkgroupSource &source, uint16_t quotient,
                                                uint16_t residue, rj_code_arch_t arch) {
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
    sequence.append(instrumentation::build_v_readfirstlane_b32(quotient, *source.vector_src, arch),
                    instrumentation::build_valu_to_salu_dependency_wait(arch));
    coordinate = quotient;
  } else {
    return false;
  }

  if (source.right_shift != 0u) {
    sequence.append(build_s_lshr_b32(quotient, coordinate,
                                     scalar_positive_inline_u32(source.right_shift), arch));
    coordinate = quotient;
  }
  if (source.low_bit_count != 0u && source.low_bit_count < 32u) {
    const uint16_t mask_shift = scalar_positive_inline_u32(32u - source.low_bit_count);
    sequence.append(build_s_lshl_b32(quotient, coordinate, mask_shift, arch),
                    build_s_lshr_b32(quotient, quotient, mask_shift, arch));
    coordinate = quotient;
  }
  sequence.append(instrumentation::build_s_sub_u32(residue, residue, coordinate, arch));
  return sequence.finish();
}

/// Emit the selection predicate shared by the inline-island and direct-call
/// gate layouts. The caller owns only the surrounding return geometry and
/// chooses registers that do not overlap its live continuation state.
[[nodiscard]] bool append_runtime_workgroup_predicate(std::vector<uint32_t> &words,
                                                      const RuntimeWorkgroupGatePlan &plan,
                                                      const WorkgroupSources &workgroup_sources,
                                                      uint16_t saved_scc, uint16_t quotient,
                                                      uint16_t residue, rj_code_arch_t arch) {
  if (plan.sample_stride <= 1u)
    return false;
  InstructionSequence sequence(words);
  sequence.append(instrumentation::build_s_cselect_b32(saved_scc, scalar_positive_inline_u32(1),
                                                       scalar_positive_inline_u32(0), arch));

  const uint16_t dispatch_mix = plan.dispatch_id_sgpr.value_or(
      scalar_positive_inline_u32(static_cast<uint32_t>(plan.literal_dispatch_id) & 63u));
  sequence.append(
      instrumentation::build_s_sub_u32(residue, scalar_positive_inline_u32(0), dispatch_mix, arch));
  sequence
      .require(append_runtime_workgroup_mix(words, workgroup_sources.x, quotient, residue, arch))
      .require(append_runtime_workgroup_mix(words, workgroup_sources.y, quotient, residue, arch))
      .require(append_runtime_workgroup_mix(words, workgroup_sources.z, quotient, residue, arch))
      .require(append_runtime_workgroup_mix(words, workgroup_sources.cluster_workgroup_id, quotient,
                                            residue, arch));
  const uint16_t shift = scalar_positive_inline_u32(std::countr_zero(plan.sample_stride));
  sequence
      .append(build_s_lshr_b32(quotient, residue, shift, arch),
              build_s_lshl_b32(quotient, quotient, shift, arch),
              instrumentation::build_s_sub_u32(residue, residue, quotient, arch))
      .require(append_runtime_workgroup_residue_compare(
          words, residue, quotient, plan.sample_offset & (plan.sample_stride - 1u), arch));
  return sequence.finish();
}

std::optional<RuntimeWorkgroupGatePrefix>
build_runtime_workgroup_gate_prefix(const RuntimeWorkgroupGatePlan &plan,
                                    const WorkgroupSources &workgroup_sources,
                                    rj_code_arch_t arch) {
  if (plan.sample_stride <= 1u)
    return std::nullopt;
  const uint16_t saved_scc = static_cast<uint16_t>(plan.exec_save_sgpr + 4u);
  const uint16_t quotient = static_cast<uint16_t>(plan.exec_save_sgpr + 5u);
  const uint16_t residue = static_cast<uint16_t>(plan.exec_save_sgpr + 6u);
  RuntimeWorkgroupGatePrefix result;
  if (!append_runtime_workgroup_predicate(result.words, plan, workgroup_sources, saved_scc,
                                          quotient, residue, arch))
    return std::nullopt;
  InstructionSequence sequence(result.words);
  const InstructionSequence::Label selected = sequence.make_label();
  sequence.branch(selected, InstructionSequence::BranchKind::SccNonzero)
      .append(instrumentation::build_s_cmp_lg_u32(saved_scc, scalar_positive_inline_u32(0), arch));
  result.bypass_branch_word = static_cast<uint32_t>(result.words.size());
  result.words.push_back(0u);
  sequence.bind_label(selected).append(
      instrumentation::build_s_cmp_lg_u32(saved_scc, scalar_positive_inline_u32(0), arch));
  if (!sequence.finish(arch))
    return std::nullopt;
  return result;
}

bool prepend_runtime_workgroup_gate(TextFragment &fragment, const RuntimeWorkgroupGatePlan &plan,
                                    const WorkgroupSources &workgroup_sources, rj_code_arch_t arch,
                                    std::vector<std::string> &errors, std::string_view patch_name) {
  auto gate = build_runtime_workgroup_gate_prefix(plan, workgroup_sources, arch);
  if (!gate) {
    errors.emplace_back("ConSan " + std::string(patch_name) +
                        " could not encode its runtime workgroup gate");
    return false;
  }
  fragment.branch_fixups.push_back(
      {.word_index = gate->bypass_branch_word, .target = TextFragmentBranchTarget::Bypass});
  fragment.before_words.insert(fragment.before_words.begin(), gate->words.begin(),
                               gate->words.end());
  return true;
}

bool prepend_runtime_workgroup_gate_to_after_words(TextFragment &fragment,
                                                   const RuntimeWorkgroupGatePlan &plan,
                                                   const WorkgroupSources &workgroup_sources,
                                                   rj_code_arch_t arch,
                                                   std::vector<std::string> &errors,
                                                   std::string_view patch_name) {
  auto gate = build_runtime_workgroup_gate_prefix(plan, workgroup_sources, arch);
  if (!gate) {
    errors.emplace_back("ConSan " + std::string(patch_name) +
                        " could not encode its post-guest runtime workgroup gate");
    return false;
  }
  const size_t bypass_words =
      gate->words.size() + fragment.after_words.size() - gate->bypass_branch_word - 1u;
  if (bypass_words > static_cast<size_t>(std::numeric_limits<int16_t>::max())) {
    errors.emplace_back("ConSan " + std::string(patch_name) +
                        " post-guest runtime workgroup gate is out of branch range");
    return false;
  }
  gate->words[gate->bypass_branch_word] = build_s_branch(static_cast<int16_t>(bypass_words), arch);
  fragment.after_words.insert(fragment.after_words.begin(), gate->words.begin(), gate->words.end());
  return true;
}

} // namespace rocjitsu::consan::detail
