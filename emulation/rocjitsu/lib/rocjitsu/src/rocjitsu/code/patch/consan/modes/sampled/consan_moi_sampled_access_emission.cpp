// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/modes/sampled/consan_moi_sampled_access_emission.h"

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/consan/consan_capability_contract.h"
#include "rocjitsu/code/patch/consan/consan_moi_access_target.h"
#include "rocjitsu/code/patch/consan/consan_moi_dynamic_record_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_exact_shadow_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_memory_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_native_abi.h"
#include "rocjitsu/code/patch/consan/consan_moi_probe_contracts.h"
#include "rocjitsu/code/patch/consan/consan_moi_report_emission.h"
#include "rocjitsu/code/patch/instruction_sequence.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace rocjitsu::consan_moi_impl {

using consan_detail::append_moi_relocated_guest_access;
using consan_detail::append_reload_moi_spilled_vgpr;
using consan_detail::moi_guest_access_relocation_requires_adjusted_address;
using consan_detail::moi_spilled_vgpr_reload_result_name;
using consan_detail::MoiSpilledVgprReloadResult;
using consan_detail::MoiWorkitemOwnerDerivationPlan;
using consan_detail::range_overlaps;
using consan_detail::reject_optional_scratch_range_overlap;
using consan_moi_detail::append_atomic_fetch_add_one_u32;
using consan_moi_detail::append_compare_moi_report_dispatch_id_word;
using consan_moi_detail::append_load_u32_vgpr_at_offset;
using consan_moi_detail::append_select_first_lane_in_exec_mask;
using consan_moi_detail::append_store_moi_report_dispatch_id_pair;
using consan_moi_detail::ConSanMoiRecordEmitter;

[[nodiscard]] bool append_sampled_banked_address(std::vector<uint32_t> &words,
                                                 uint64_t first_address, uint32_t element_size,
                                                 uint32_t bank_count, uint16_t bank_vgpr,
                                                 uint16_t address_vgpr, rj_code_arch_t arch);

[[nodiscard]] bool append_direct_sampled_immediate_check(
    std::vector<uint32_t> &words, const MoiSampledAccessEmissionPlan &plan,
    uint32_t prior_record_index, uint16_t bank_vgpr, uint16_t current_low_vgpr,
    uint16_t current_high_vgpr, rj_code_arch_t arch, std::vector<std::string> &errors) {
  if (!plan.sampled_check)
    return true;

  const uint16_t address_lo_vgpr = plan.scratch_vgpr;
  const uint16_t address_hi_vgpr = static_cast<uint16_t>(plan.scratch_vgpr + 1u);
  const uint16_t tmp_vgpr = static_cast<uint16_t>(plan.scratch_vgpr + 4u);
  const uint16_t prior_low_vgpr = static_cast<uint16_t>(plan.scratch_vgpr + 5u);
  const uint16_t prior_high_vgpr = static_cast<uint16_t>(plan.scratch_vgpr + 6u);
  if (plan.window_bank_count == 0) {
    errors.emplace_back("ConSan MOI sampled checker has an invalid prior bank");
    return false;
  }
  const uint64_t prior_address = plan.report_buffer_address + plan.sampled_watchpoints_offset +
                                 static_cast<uint64_t>(prior_record_index) * sizeof(uint64_t);
  InstructionSequence sequence(words);
  MoiStagedEmission emission(sequence, errors, "ConSan MOI sampled checker");
  const auto done_label = sequence.make_label();

  emission.stage("prior-slot snapshot");
  emission.require(
      append_sampled_banked_address(words, prior_address, sizeof(uint64_t), plan.window_bank_count,
                                    bank_vgpr, address_lo_vgpr, arch) &&
      sequence.emit_all(instrumentation::build_v_mov_b32_literal(prior_low_vgpr, 0u, arch),
                        instrumentation::build_v_mov_b32_literal(prior_high_vgpr, 0u, arch),
                        instrumentation::build_flat_atomic_add_u64(
                            address_lo_vgpr, prior_low_vgpr, prior_low_vgpr,
                            /*return_old_value=*/true, kAmdGpuScopeDevice, arch)));
  emission.require(append_moi_global_atomic_wait(words, arch));
  auto append_required_predicate = [&](std::optional<uint32_t> compare) {
    if (!sequence.emit(compare))
      return false;
    return sequence.emit_branch(done_label, InstructionSequence::BranchKind::VccZero);
  };

  emission.stage("validity and owner comparison");
  emission.require(
      sequence.emit(instrumentation::build_v_and_b32_literal(tmp_vgpr, 1u, prior_low_vgpr, arch)));
  emission.require(
      append_required_predicate(
          instrumentation::build_v_cmp_ne_u32_vcc(scalar_positive_inline_u32(0), tmp_vgpr, arch)) &&
      append_extract_exact_shadow_field(words, address_lo_vgpr, current_low_vgpr,
                                        consan_moi_sampled_watchpoint::owner_shift,
                                        consan_moi_sampled_watchpoint::max_owner, arch) &&
      append_extract_exact_shadow_field(words, address_hi_vgpr, prior_low_vgpr,
                                        consan_moi_sampled_watchpoint::owner_shift,
                                        consan_moi_sampled_watchpoint::max_owner, arch) &&
      append_required_predicate(instrumentation::build_v_cmp_ne_u32_vcc(
          vector_source_vgpr(address_lo_vgpr), address_hi_vgpr, arch)));

  const uint32_t low_epoch_generation_mask =
      static_cast<uint32_t>(consan_moi_sampled_watchpoint::epoch_generation_mask);
  const uint32_t high_generation_mask =
      static_cast<uint32_t>(consan_moi_sampled_watchpoint::generation_mask >> 32u);
  const uint32_t high_range_mask =
      static_cast<uint32_t>((consan_moi_sampled_watchpoint::start_byte_mask |
                             consan_moi_sampled_watchpoint::count_mask) >>
                            32u);
  auto append_equal_masked = [&](uint16_t current_vgpr, uint16_t prior_vgpr, uint32_t mask) {
    if (!sequence.emit_all(
            instrumentation::build_v_and_b32_literal(address_lo_vgpr, mask, current_vgpr, arch),
            instrumentation::build_v_and_b32_literal(address_hi_vgpr, mask, prior_vgpr, arch)))
      return false;
    return append_required_predicate(instrumentation::build_v_cmp_eq_u32_vcc(
        vector_source_vgpr(address_lo_vgpr), address_hi_vgpr, arch));
  };
  emission.stage("ordering and range comparison");
  emission.require(
      append_equal_masked(current_low_vgpr, prior_low_vgpr, low_epoch_generation_mask) &&
      append_equal_masked(current_high_vgpr, prior_high_vgpr, high_generation_mask) &&
      append_equal_masked(current_high_vgpr, prior_high_vgpr, high_range_mask));

  emission.stage("access-kind comparison");
  emission.require(
      append_extract_exact_shadow_field(
          words, address_lo_vgpr, current_low_vgpr,
          consan_moi_sampled_watchpoint::access_kind_shift,
          (1u << consan_moi_sampled_watchpoint::access_kind_bits) - 1u, arch) &&
      append_extract_exact_shadow_field(
          words, address_hi_vgpr, prior_low_vgpr, consan_moi_sampled_watchpoint::access_kind_shift,
          (1u << consan_moi_sampled_watchpoint::access_kind_bits) - 1u, arch) &&
      sequence.emit(instrumentation::build_v_and_b32(tmp_vgpr, vector_source_vgpr(address_lo_vgpr),
                                                     address_hi_vgpr, arch)) &&
      append_required_predicate(instrumentation::build_v_cmp_ne_u32_vcc(
          scalar_positive_inline_u32(static_cast<uint32_t>(ConSanMoiShadowAccessKind::Read)),
          tmp_vgpr, arch)));

  const uint64_t immediate_conflict_count_address =
      plan.report_buffer_address + offsetof(ConSanMoiReportHeader, event_counter);
  emission.stage("conflict publication");
  emission.require(append_atomic_fetch_add_one_u32(words, immediate_conflict_count_address,
                                                   tmp_vgpr, plan.scratch_vgpr, arch));

  // VCC is instrumentation-local here. The sampled body's common exit
  // restores guest VCC from its dedicated snapshot, so saving VCC in this
  // helper is both unnecessary and unsafe: moi_exec_save_sgpr is the
  // immutable guest-EXEC snapshot which that exit later consumes.
  emission.stage("control-flow resolution");
  emission.require(sequence.bind(done_label));
  return emission.finish(arch);
}

[[nodiscard]] bool
append_sampled_identity_hash(std::vector<uint32_t> &words,
                             const consan_moi_detail::ConSanMoiReportDispatchIdSource &dispatch,
                             const ConSanMoiWorkgroupSources &workgroup_sources,
                             std::optional<uint16_t> owner_vgpr, uint32_t bucket_count,
                             uint16_t result_vgpr, uint16_t temporary_vgpr,
                             uint16_t coordinate_vgpr, rj_code_arch_t arch) {
  InstructionSequence sequence(words);
  if (bucket_count == 0 || !std::has_single_bit(bucket_count))
    return false;
  sequence.require(
      append_moi_report_dispatch_id_pair(words, dispatch, result_vgpr, temporary_vgpr, arch));
  const auto mix = [&](uint16_t source) {
    sequence.append(instrumentation::build_v_xor_b32(result_vgpr, vector_source_vgpr(source),
                                                     result_vgpr, arch));
  };
  mix(temporary_vgpr);
  const std::array<std::pair<const ConSanMoiWorkgroupSource *, uint32_t>, 4> tuple_sources = {{
      {&workgroup_sources.x, 0u},
      {&workgroup_sources.y, 1u},
      {&workgroup_sources.z, 2u},
      {&workgroup_sources.cluster_workgroup_id, 3u},
  }};
  for (const auto &[source, dimension_shift] : tuple_sources) {
    if (!source->has_value())
      continue;
    sequence.require(
        consan_detail::append_workgroup_source_value(words, *source, coordinate_vgpr, arch));
    if (dimension_shift != 0u)
      sequence.append(instrumentation::build_v_lshlrev_b32(
          coordinate_vgpr, scalar_positive_inline_u32(dimension_shift), coordinate_vgpr, arch));
    mix(coordinate_vgpr);
  }
  if (owner_vgpr)
    mix(*owner_vgpr);
  sequence
      .append(instrumentation::build_v_lshrrev_b32(temporary_vgpr, scalar_positive_inline_u32(16),
                                                   result_vgpr, arch))
      .append(instrumentation::build_v_xor_b32(result_vgpr, vector_source_vgpr(temporary_vgpr),
                                               result_vgpr, arch))
      .append(instrumentation::build_v_mul_lo_u32_literal(result_vgpr, temporary_vgpr, 0x85ebca6bu,
                                                          result_vgpr, arch),
              instrumentation::build_v_and_b32_literal(result_vgpr, bucket_count - 1u, result_vgpr,
                                                       arch));
  return sequence.finish();
}

[[nodiscard]] bool
append_sampled_window_bank_index(std::vector<uint32_t> &words,
                                 const consan_moi_detail::ConSanMoiReportDispatchIdSource &dispatch,
                                 const ConSanMoiWorkgroupSources &workgroup_sources,
                                 uint32_t bank_count, uint16_t bank_vgpr, uint16_t temporary_vgpr,
                                 uint16_t owner_vgpr, rj_code_arch_t arch) {
  if (bank_count == 1)
    return InstructionSequence(words).emit(
        instrumentation::build_v_mov_b32_literal(bank_vgpr, 0, arch));
  // A causal-window bank is an evidence-retention bucket, not a workgroup
  // identity. Include the wave owner so several waves touching the same
  // sampled address can retain independent windows instead of racing for one
  // first-publisher slot. Barrier and atomic paths use the same owner-aware
  // hash, so synchronization metadata remains joined to the corresponding
  // access window.
  return append_sampled_identity_hash(words, dispatch, workgroup_sources, owner_vgpr, bank_count,
                                      bank_vgpr, temporary_vgpr, temporary_vgpr, arch);
}

[[nodiscard]] bool append_sampled_workgroup_residue(std::vector<uint32_t> &words,
                                                    const MoiSampledAccessEmissionPlan &plan,
                                                    uint16_t residue_vgpr, uint16_t temporary_vgpr,
                                                    uint16_t coordinate_vgpr, rj_code_arch_t arch) {
  return append_sampled_identity_hash(words, plan.dispatch_id, plan.workgroup_sources, std::nullopt,
                                      plan.runtime_sample_stride, residue_vgpr, temporary_vgpr,
                                      coordinate_vgpr, arch);
}

[[nodiscard]] bool append_sampled_banked_address(std::vector<uint32_t> &words,
                                                 uint64_t first_address, uint32_t element_size,
                                                 uint32_t bank_count, uint16_t bank_vgpr,
                                                 uint16_t address_vgpr, rj_code_arch_t arch) {
  const ConSanTargetProfile *target = consan_target_profile(arch);
  if (target == nullptr)
    return false;
  if (bank_count != 1u) {
    return consan_detail::append_moi_indexed_address(words,
                                                     {.table_address = first_address,
                                                      .stride_bytes = element_size,
                                                      .address_vgpr = address_vgpr,
                                                      .index_vgpr = bank_vgpr},
                                                     *target);
  }

  return InstructionSequence(words).emit_all(
      instrumentation::build_v_mov_b32_literal(address_vgpr, static_cast<uint32_t>(first_address),
                                               arch),
      instrumentation::build_v_mov_b32_literal(static_cast<uint16_t>(address_vgpr + 1u),
                                               static_cast<uint32_t>(first_address >> 32u), arch));
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_direct_sampled_watchpoint_range_words(
    std::span<const uint8_t> bytes, const ConSanMoiCandidate &candidate,
    uint64_t owner_descriptor_file_offset, const ConSanAccessRange &access_range,
    const MoiSampledAccessEmissionPlan &plan, rj_code_arch_t arch, uint32_t record_index,
    std::span<const uint32_t> prior_record_indices, bool relocate_instruction,
    std::optional<uint16_t> preserved_lds_byte_offset_vgpr, const VgprSpillSequence *spill,
    std::optional<uint16_t> spilled_lds_byte_offset_vgpr, std::vector<std::string> &errors,
    uint32_t *guest_instruction_offset = nullptr,
    uint32_t *guest_instruction_word_count = nullptr) {
  const ConSanTargetProfile *target = consan_target_profile(arch);
  if (target == nullptr) {
    errors.emplace_back("ConSan MOI sampled probe has no target profile");
    return std::nullopt;
  }
  if (plan.pending_acquire_owner_bank_count == 0u ||
      !std::has_single_bit(plan.pending_acquire_owner_bank_count) ||
      plan.pending_acquire_owner_bank_count > kConSanMoiSampledPendingAcquireOwnerBankCount) {
    errors.emplace_back("ConSan MOI sampled probe has an invalid pending-acquire owner bank");
    return std::nullopt;
  }
  // A runtime fast-gate lowers the cave body with stride=1 after the outer
  // scalar gate has made the sampling decision.  Multi-bank publication still
  // needs its bank index to survive all ordinary temporaries in that cave.
  // Reserve the dedicated bank VGPR based on the emitted layout, not solely on
  // whether the inner vector sampler remains enabled.
  const bool dedicated_bank_vgpr = plan.runtime_sample_stride > 1 || plan.window_bank_count > 1;
  const bool reserve_two_address_replay_scratch =
      moi_guest_access_relocation_requires_adjusted_address(candidate, *target);
  if (static_cast<uint32_t>(plan.scratch_vgpr) + plan.scratch_vgpr_count > kMaxVgprs) {
    errors.emplace_back("ConSan MOI sampled probe exceeds the VGPR file");
    return std::nullopt;
  }
  auto lds_byte_offset_vgpr = candidate_lds_byte_offset_vgpr(candidate, errors);
  if (!lds_byte_offset_vgpr)
    return std::nullopt;
  if (spilled_lds_byte_offset_vgpr && spill == nullptr) {
    errors.emplace_back("ConSan MOI sampled probe has no spill plan for its LDS address");
    return std::nullopt;
  }
  if (preserved_lds_byte_offset_vgpr)
    lds_byte_offset_vgpr = preserved_lds_byte_offset_vgpr;
  if (!plan.spill_overlaps_guest_operands &&
      reject_candidate_scratch_range_overlap(candidate, plan.scratch_vgpr, plan.scratch_vgpr_count,
                                             errors))
    return std::nullopt;
  if ((!plan.automatic_private_epoch && !plan.persistent_sgprs.complete() &&
       reject_optional_scratch_range_overlap(plan.owner_epoch_vgprs.owner, plan.scratch_vgpr,
                                             plan.scratch_vgpr_count, "MOI owner", errors)) ||
      (!plan.automatic_private_epoch && !plan.persistent_sgprs.complete() &&
       reject_optional_scratch_range_overlap(plan.owner_epoch_vgprs.epoch, plan.scratch_vgpr,
                                             plan.scratch_vgpr_count, "MOI epoch", errors)))
    return std::nullopt;
  std::optional<uint16_t> derived_owner_vgpr;
  std::optional<uint32_t> derived_owner_word;
  if (!plan.owner_epoch_vgprs.owner) {
    const auto owner_shift =
        moi_descriptor_owner_shift(bytes, owner_descriptor_file_offset, arch, errors);
    if (!owner_shift)
      return std::nullopt;
    const uint16_t value_vgpr = static_cast<uint16_t>(plan.scratch_vgpr + 4u);
    const auto owner_init = instrumentation::build_v_lshrrev_b32(
        value_vgpr, scalar_positive_inline_u32(*owner_shift), kAmdGpuWorkitemIdX, arch);
    if (!owner_init) {
      errors.emplace_back("ConSan MOI sampled probe could not encode owner derivation");
      return std::nullopt;
    }
    derived_owner_vgpr = value_vgpr;
    derived_owner_word = *owner_init;
  }
  const uint16_t low_vgpr = static_cast<uint16_t>(plan.scratch_vgpr + 2u);
  const uint16_t high_vgpr = static_cast<uint16_t>(plan.scratch_vgpr + 3u);
  const uint16_t tmp_vgpr = static_cast<uint16_t>(plan.scratch_vgpr + 4u);
  const uint16_t owner_vgpr = plan.owner_epoch_vgprs.owner.value_or(derived_owner_vgpr.value_or(0));
  const bool runtime_sampled = plan.runtime_sample_stride > 1;
  const uint16_t bank_vgpr =
      dedicated_bank_vgpr
          ? static_cast<uint16_t>(plan.scratch_vgpr + (plan.sampled_check ? 7u : 5u))
          : tmp_vgpr;
  if (runtime_sampled && !plan.owner_epoch_vgprs.owner && !derived_owner_vgpr) {
    errors.emplace_back("ConSan MOI runtime sampled probe could not derive a wave owner");
    return std::nullopt;
  }
  if (!plan.exec_save_sgpr || *plan.exec_save_sgpr > 98u || *plan.exec_save_sgpr % 2u != 0u) {
    errors.emplace_back(
        "ConSan MOI sampled probe needs seven aligned VCC/EXEC/SCC-save SGPRs in 0..104");
    return std::nullopt;
  }
  const auto kind = consan_moi_shadow_kind_from_access_kind(candidate.site().kind);
  const uint32_t generation =
      plan.report_generation & consan_moi_sampled_watchpoint::max_generation;
  const uint64_t generation_field = static_cast<uint64_t>(generation)
                                    << consan_moi_sampled_watchpoint::generation_shift;
  const uint32_t low_literal = static_cast<uint32_t>(
      consan_moi_sampled_watchpoint::valid_mask |
      (static_cast<uint64_t>(kind) << consan_moi_sampled_watchpoint::access_kind_shift) |
      generation_field);
  const uint32_t encoded_byte_count = encode_consan_moi_sampled_byte_count(access_range.byte_width)
                                      << (consan_moi_sampled_watchpoint::count_shift - 32u);
  const uint32_t encoded_generation_high = static_cast<uint32_t>(generation_field >> 32u);
  const uint64_t sampled_entry_address = plan.report_buffer_address +
                                         plan.sampled_watchpoints_offset +
                                         static_cast<uint64_t>(record_index) * sizeof(uint64_t);
  const uint64_t pending_acquire_address =
      plan.report_buffer_address + plan.sampled_pending_acquires_offset +
      static_cast<uint64_t>(record_index) * plan.pending_acquire_owner_bank_count *
          sizeof(ConSanMoiSampledPendingAcquireSlot);

  std::vector<uint32_t> words;
  words.reserve(candidate.size() / sizeof(uint32_t) + 40u + (derived_owner_word ? 1u : 0u) +
                (plan.owner_epoch_vgprs.epoch ? 2u : 0u) +
                (relocate_instruction && moi_load_clobbers_address(candidate) &&
                         !preserved_lds_byte_offset_vgpr
                     ? 1u
                     : 0u) +
                (candidate.lowering_offset(access_range) != 0u ? 3u : 0u));
  InstructionSequence sequence(words);
  MoiEmissionRequirement require_emission(sequence, errors);
  const auto collision_label = sequence.make_label();
  const auto different_identity_label = sequence.make_label();
  const auto restore_label = sequence.make_label();
  ConSanMoiRecordEmitter record(words, plan.scratch_vgpr, tmp_vgpr, arch);
  const auto reload_spilled_lds_byte_offset = [&](uint16_t destination_vgpr) {
    if (!spilled_lds_byte_offset_vgpr)
      return true;
    const auto reload = consan_detail::append_reload_moi_spilled_vgpr(
        words, *spill, destination_vgpr, *spilled_lds_byte_offset_vgpr, arch);
    if (reload == consan_detail::MoiSpilledVgprReloadResult::Appended)
      return true;
    errors.emplace_back("ConSan MOI sampled probe could not recover its spilled LDS address: " +
                        std::string(consan_detail::moi_spilled_vgpr_reload_result_name(reload)));
    return false;
  };
  // A DS load may overwrite its own address VGPR. Snapshot that address before
  // executing the displaced instruction so sampled publication remains tied
  // to the guest access rather than to the loaded payload.
  if (relocate_instruction &&
      (candidate.is_direct_to_lds() || moi_load_clobbers_address(candidate)) &&
      !preserved_lds_byte_offset_vgpr) {
    const uint16_t saved_lds_byte_offset_vgpr =
        static_cast<uint16_t>(plan.scratch_vgpr + plan.base_scratch_vgpr_count);
    if (candidate.is_direct_to_lds()) {
      if (!plan.exec_save_sgpr) {
        errors.emplace_back(
            "ConSan MOI sampled direct-to-LDS probe requires an EXEC-save SGPR pair");
        return std::nullopt;
      }
      if (!append_materialize_direct_to_lds_address(
              words, candidate.site(), saved_lds_byte_offset_vgpr, *plan.exec_save_sgpr, arch)) {
        errors.emplace_back("ConSan MOI sampled probe could not materialize a direct-to-LDS "
                            "destination");
        return std::nullopt;
      }
    } else {
      words.push_back(build_v_mov_b32_e32(saved_lds_byte_offset_vgpr,
                                          vector_source_vgpr(*lds_byte_offset_vgpr), arch));
    }
    lds_byte_offset_vgpr = saved_lds_byte_offset_vgpr;
  }
  if (derived_owner_word)
    words.push_back(*derived_owner_word);
  if (relocate_instruction) {
    const bool capture_high_bank_address =
        moi_access_requires_high_bank_address_capture(candidate, arch);
    const bool select_guest_vgpr_bank = target->has_selectable_vgpr_bank &&
                                        candidate.incoming_vgpr_bank_mode.value_or(0u) != 0u &&
                                        !capture_high_bank_address;
    if (select_guest_vgpr_bank)
      words.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
          0u, static_cast<uint8_t>(*candidate.incoming_vgpr_bank_mode), arch));
    if (guest_instruction_offset)
      *guest_instruction_offset = static_cast<uint32_t>(words.size() * sizeof(uint32_t));
    // Metadata consumes the low-bank captured address, while the guest still
    // names its original encoded SRC0 under the restored application mode.
    const uint16_t guest_address_vgpr = capture_high_bank_address
                                            ? *candidate.site().lowering.form->address_vgpr
                                            : *lds_byte_offset_vgpr;
    if (!append_moi_relocated_guest_access(
            words, bytes, candidate, target, guest_address_vgpr,
            reserve_two_address_replay_scratch
                ? std::optional<uint16_t>(
                      static_cast<uint16_t>(plan.scratch_vgpr + plan.scratch_vgpr_count - 1u))
                : std::nullopt,
            errors, guest_instruction_word_count))
      return std::nullopt;
    if (!append_moi_lds_wait(words, arch))
      return std::nullopt;
    if (select_guest_vgpr_bank)
      words.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
          static_cast<uint8_t>(*candidate.incoming_vgpr_bank_mode), 0u, arch));
  }

  const auto publication_state = moi_sampled_publication_state_sgprs(plan.exec_save_sgpr);
  if (!publication_state) {
    errors.emplace_back("ConSan MOI sampled probe has no scalar publication-state layout");
    return std::nullopt;
  }
  const uint16_t original_exec_save_sgpr = publication_state->original_exec_save_sgpr;
  const uint16_t selection_vcc_save_sgpr = publication_state->selection_vcc_save_sgpr;
  const uint16_t publication_exec_save_sgpr = publication_state->publication_exec_save_sgpr;
  const uint16_t publication_scc_save_sgpr = publication_state->guest_scc_snapshot_sgpr;
  require_emission.append(
      "ConSan MOI sampled probe could not save guest EXEC/VCC/SCC",
      instrumentation::build_s_mov_b64(original_exec_save_sgpr, kAmdGpuExecLo, arch),
      instrumentation::build_s_cselect_b32(publication_scc_save_sgpr, scalar_positive_inline_u32(1),
                                           scalar_positive_inline_u32(0), arch),
      instrumentation::build_s_mov_b64(selection_vcc_save_sgpr, kAmdGpuVccLo, arch));
  // Publication has several nested EXEC-narrowing paths. Some paths reuse the
  // publication-save pair after an earlier narrowing, so that pair is not a
  // reliable copy of the guest mask at the common exit. Keep the otherwise
  // unused leading pair as the immutable guest EXEC snapshot.
  if (candidate_requires_flat_address_materialization(candidate)) {
    if (!preserved_lds_byte_offset_vgpr) {
      errors.emplace_back("ConSan MOI sampled FLAT address has no preserved address pair");
      return std::nullopt;
    }
    if (!append_materialize_flat_access_address(words, candidate, *preserved_lds_byte_offset_vgpr,
                                                *preserved_lds_byte_offset_vgpr, arch)) {
      errors.emplace_back("ConSan MOI sampled probe could not materialize FLAT address");
      return std::nullopt;
    }
  }
  if (runtime_sampled) {
    if (plan.runtime_workgroup_gate_in_body) {
      require_emission(
          append_sampled_workgroup_residue(words, plan, bank_vgpr, high_vgpr, low_vgpr, arch),
          "ConSan MOI runtime sampled probe could not select a private-state workgroup");
      require_emission.append(
          "ConSan MOI runtime sampled probe could not encode its body workgroup gate",
          instrumentation::build_v_mov_b32_literal(high_vgpr, plan.runtime_sample_offset, arch),
          instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(high_vgpr), bank_vgpr, arch));
      sequence.branch(restore_label, InstructionSequence::BranchKind::VccZero)
          .append(instrumentation::build_s_and_saveexec_b64(publication_exec_save_sgpr,
                                                            kAmdGpuVccLo, arch));
    }
    // Sample addresses while retaining every wave which touches the selected
    // cell. Selecting owners here would discard the cross-wave evidence that
    // the sampled checker needs in order to identify a race.
    if (spilled_lds_byte_offset_vgpr) {
      const uint16_t recovered_vgpr =
          candidate.lowering_offset(access_range) == 0u ? low_vgpr : high_vgpr;
      sequence.require(reload_spilled_lds_byte_offset(recovered_vgpr));
      if (candidate.lowering_offset(access_range) != 0u && sequence)
        sequence.require(append_compute_effective_lds_byte_offset(
            words, low_vgpr, recovered_vgpr, candidate.lowering_offset(access_range), arch));
    } else if (candidate.lowering_offset(access_range) != 0u) {
      sequence.require(append_compute_effective_lds_byte_offset(
          words, low_vgpr, *lds_byte_offset_vgpr, candidate.lowering_offset(access_range), arch));
    } else {
      words.push_back(
          build_v_mov_b32_e32(low_vgpr, vector_source_vgpr(*lds_byte_offset_vgpr), arch));
    }
    require_emission.append(
        "ConSan MOI runtime sampled probe could not encode LDS-cell selector",
        instrumentation::build_v_lshrrev_b32(
            low_vgpr, scalar_positive_inline_u32(consan_moi_shadow_cell::granule_shift), low_vgpr,
            arch),
        instrumentation::build_v_and_b32_literal(low_vgpr, plan.runtime_sample_stride - 1u,
                                                 low_vgpr, arch),
        instrumentation::build_v_mov_b32_literal(high_vgpr, plan.runtime_sample_offset, arch),
        instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(high_vgpr), low_vgpr, arch));
    sequence.branch(restore_label, InstructionSequence::BranchKind::VccZero)
        .append(instrumentation::build_s_and_saveexec_b64(publication_exec_save_sgpr, kAmdGpuVccLo,
                                                          arch));
  }
  if (runtime_sampled || plan.window_bank_count > 1) {
    require_emission(append_sampled_window_bank_index(
                         words, plan.dispatch_id, plan.workgroup_sources, plan.window_bank_count,
                         bank_vgpr, high_vgpr, owner_vgpr, arch),
                     "ConSan MOI sampled probe could not select a window bank");
  } else {
    sequence.append(instrumentation::build_v_mov_b32_literal(bank_vgpr, 0, arch));
  }
  sequence.require(append_moi_delay_words(words, arch,
                                          {.mode = plan.delay_mode,
                                           .count = plan.delay_count,
                                           .variable_source = plan.delay_variable_source},
                                          errors, "ConSan MOI sampled probe"));

  // A ready deferred acquire deterministically selects the wave which may
  // publish its associated later read. Empty slots are ordinary reads. Odd,
  // malformed, or other-wave slots take the existing fail-closed drop path,
  // preventing independent first-publisher races from joining unrelated
  // atomic and LDS evidence.
  if (kind == ConSanMoiShadowAccessKind::Read) {
    const auto pending_gate_done_label = sequence.make_label();
    if (plan.pending_acquire_owner_bank_count > 1u) {
      sequence
          .append(instrumentation::build_v_mul_lo_u32_literal(
                      low_vgpr, high_vgpr, plan.pending_acquire_owner_bank_count, bank_vgpr, arch),
                  instrumentation::build_v_and_b32_literal(
                      high_vgpr, plan.pending_acquire_owner_bank_count - 1u, owner_vgpr, arch),
                  instrumentation::build_v_add_u32(low_vgpr, vector_source_vgpr(low_vgpr),
                                                   high_vgpr, arch))
          .require(consan_detail::append_moi_indexed_address(
              words,
              {.table_address = pending_acquire_address,
               .stride_bytes = sizeof(ConSanMoiSampledPendingAcquireSlot),
               .address_vgpr = plan.scratch_vgpr,
               .index_vgpr = low_vgpr},
              *target));
    } else {
      sequence.require(append_sampled_banked_address(
          words, pending_acquire_address, sizeof(ConSanMoiSampledPendingAcquireSlot),
          plan.window_bank_count, bank_vgpr, plan.scratch_vgpr, arch));
    }
    sequence
        .require(append_load_u32_vgpr_at_offset(
            words, plan.scratch_vgpr, offsetof(ConSanMoiSampledPendingAcquireSlot, version),
            low_vgpr, arch))
        .append(
            instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), low_vgpr, arch),
            instrumentation::build_s_cbranch_vccz(1, arch))
        .branch(pending_gate_done_label, InstructionSequence::BranchKind::Unconditional)
        .append(
            instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(2), low_vgpr, arch))
        .branch(different_identity_label, InstructionSequence::BranchKind::VccZero)
        .require(append_load_u32_vgpr_at_offset(
            words, plan.scratch_vgpr, offsetof(ConSanMoiSampledPendingAcquireSlot, owner_id),
            high_vgpr, arch))
        .append(instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(owner_vgpr), high_vgpr,
                                                        arch))
        .branch(different_identity_label, InstructionSequence::BranchKind::VccZero)
        .append(instrumentation::build_s_and_saveexec_b64(publication_exec_save_sgpr, kAmdGpuVccLo,
                                                          arch))
        .bind_label(pending_gate_done_label);
  }

  const uint64_t causal_window_address =
      plan.report_buffer_address + plan.sampled_causal_windows_offset +
      static_cast<uint64_t>(record_index) * sizeof(ConSanMoiSampledCausalWindow);
  const bool claimed_window = append_sampled_banked_address(
      words, causal_window_address + offsetof(ConSanMoiSampledCausalWindow, publication_state),
      sizeof(ConSanMoiSampledCausalWindow), plan.window_bank_count, bank_vgpr, plan.scratch_vgpr,
      arch);
  require_emission(claimed_window, "ConSan MOI sampled probe could not claim a causal window slot");
  require_emission.append(
      "ConSan MOI sampled probe could not claim a causal window slot",
      instrumentation::build_v_mov_b32_literal(
          low_vgpr, static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Publishing),
          arch),
      instrumentation::build_v_mov_b32_literal(
          high_vgpr, static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Empty), arch),
      instrumentation::build_flat_atomic_cmpswap_b32(plan.scratch_vgpr, low_vgpr, low_vgpr,
                                                     /*return_old_value=*/true, kAmdGpuScopeDevice,
                                                     arch));
  sequence.require(append_moi_global_atomic_wait(words, arch))
      .append(instrumentation::build_v_cmp_eq_u32_vcc(
          scalar_positive_inline_u32(
              static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Empty)),
          low_vgpr, arch))
      .branch(collision_label, InstructionSequence::BranchKind::VccZero)
      .append(
          instrumentation::build_s_and_saveexec_b64(publication_exec_save_sgpr, kAmdGpuVccLo, arch))
      .require(append_sampled_banked_address(
          words, causal_window_address, sizeof(ConSanMoiSampledCausalWindow),
          plan.window_bank_count, bank_vgpr, plan.scratch_vgpr, arch));
  const auto store_dynamic_record_index = [&]() {
    if (plan.window_bank_count == 1u) {
      return record.store_literal(offsetof(ConSanMoiSampledCausalWindow, first_entry),
                                  record_index);
    }
    if (!sequence.emit(
            instrumentation::build_v_add_u32_literal(high_vgpr, record_index, bank_vgpr, arch)))
      return false;
    return record.store_vgpr(offsetof(ConSanMoiSampledCausalWindow, first_entry), high_vgpr);
  };
  require_emission(
      record.store_literal(offsetof(ConSanMoiSampledCausalWindow, generation),
                           static_cast<uint32_t>(plan.report_generation)) &&
          record.store_literal(offsetof(ConSanMoiSampledCausalWindow, generation) + 4u,
                               static_cast<uint32_t>(plan.report_generation >> 32u)) &&
          append_store_moi_report_dispatch_id_pair(
              record, plan.dispatch_id, offsetof(ConSanMoiSampledCausalWindow, dispatch_id)) &&
          record.store_workgroup(offsetof(ConSanMoiSampledCausalWindow, workgroup_x),
                                 plan.workgroup_sources.x,
                                 ConSanMoiRecordEmitter::MissingWorkgroupSource::StoreZero) &&
          record.store_workgroup(offsetof(ConSanMoiSampledCausalWindow, workgroup_y),
                                 plan.workgroup_sources.y,
                                 ConSanMoiRecordEmitter::MissingWorkgroupSource::StoreZero) &&
          record.store_workgroup(offsetof(ConSanMoiSampledCausalWindow, workgroup_z),
                                 plan.workgroup_sources.z,
                                 ConSanMoiRecordEmitter::MissingWorkgroupSource::StoreZero) &&
          (plan.owner_epoch_vgprs.epoch
               ? record.store_vgpr(offsetof(ConSanMoiSampledCausalWindow, epoch),
                                   *plan.owner_epoch_vgprs.epoch)
               : record.store_literal(offsetof(ConSanMoiSampledCausalWindow, epoch), 0)) &&
          store_dynamic_record_index() &&
          record.store_literal(offsetof(ConSanMoiSampledCausalWindow, entry_count), 1) &&
          record.store_workgroup(offsetof(ConSanMoiSampledCausalWindow, cluster_workgroup_id),
                                 plan.workgroup_sources.cluster_workgroup_id),
      "ConSan MOI sampled probe could not publish causal window metadata");

  require_emission.append("ConSan MOI sampled probe could not encode sampled entry low word",
                          instrumentation::build_v_mov_b32_literal(low_vgpr, low_literal, arch));
  if (plan.owner_epoch_vgprs.owner || derived_owner_vgpr)
    require_emission(append_add_shifted_vgpr_field(
                         words, low_vgpr, owner_vgpr, consan_moi_sampled_watchpoint::owner_shift,
                         consan_moi_sampled_watchpoint::max_owner, tmp_vgpr, arch),
                     "ConSan MOI sampled probe could not encode owner field");
  if (plan.owner_epoch_vgprs.epoch)
    require_emission(append_add_shifted_vgpr_field(words, low_vgpr, *plan.owner_epoch_vgprs.epoch,
                                                   consan_moi_sampled_watchpoint::epoch_shift,
                                                   consan_moi_sampled_watchpoint::max_epoch,
                                                   tmp_vgpr, arch),
                     "ConSan MOI sampled probe could not encode epoch field");
  uint16_t effective_lds_byte_offset_vgpr = *lds_byte_offset_vgpr;
  if (spilled_lds_byte_offset_vgpr) {
    const uint16_t recovered_vgpr =
        candidate.lowering_offset(access_range) == 0u ? tmp_vgpr : high_vgpr;
    sequence.require(reload_spilled_lds_byte_offset(recovered_vgpr));
    if (candidate.lowering_offset(access_range) != 0u)
      require_emission(
          append_compute_effective_lds_byte_offset(words, tmp_vgpr, recovered_vgpr,
                                                   candidate.lowering_offset(access_range), arch),
          "ConSan MOI sampled probe could not encode effective LDS byte offset");
    effective_lds_byte_offset_vgpr = tmp_vgpr;
  } else if (candidate.lowering_offset(access_range) != 0u) {
    require_emission(
        append_compute_effective_lds_byte_offset(words, tmp_vgpr, *lds_byte_offset_vgpr,
                                                 candidate.lowering_offset(access_range), arch),
        "ConSan MOI sampled probe could not encode effective LDS byte offset");
    effective_lds_byte_offset_vgpr = tmp_vgpr;
  }
  require_emission.append(
      "ConSan MOI sampled probe could not encode sampled entry byte offset",
      instrumentation::build_v_lshlrev_b32(
          high_vgpr,
          scalar_positive_inline_u32(consan_moi_sampled_watchpoint::start_byte_shift - 32u),
          effective_lds_byte_offset_vgpr, arch));
  require_emission(append_add_literal_field(words, high_vgpr,
                                            encoded_byte_count | encoded_generation_high, tmp_vgpr,
                                            arch),
                   "ConSan MOI sampled probe could not encode byte-count field");
  for (uint32_t prior_record_index : prior_record_indices) {
    sequence.require(append_direct_sampled_immediate_check(
        words, plan, prior_record_index, bank_vgpr, low_vgpr, high_vgpr, arch, errors));
  }
  // A site is shared by all selected waves. Publish the complete packed entry
  // with one device-scope atomic exchange so concurrent writers cannot leave a
  // low word from one wave paired with a high word from another.
  require_emission(append_sampled_banked_address(words, sampled_entry_address, sizeof(uint64_t),
                                                 plan.window_bank_count, bank_vgpr,
                                                 plan.scratch_vgpr, arch) &&
                       sequence.emit(instrumentation::build_flat_atomic_swap_b64(
                           plan.scratch_vgpr, low_vgpr, low_vgpr, /*return_old_value=*/true,
                           kAmdGpuScopeDevice, arch)),
                   "ConSan MOI sampled probe could not atomically publish sampled entry");
  sequence.require(append_moi_global_atomic_wait(words, arch))
      .require(append_sampled_banked_address(
          words, causal_window_address, sizeof(ConSanMoiSampledCausalWindow),
          plan.window_bank_count, bank_vgpr, plan.scratch_vgpr, arch));
  require_emission(
      record.store_literal(offsetof(ConSanMoiSampledCausalWindow, publication_state),
                           static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Ready)),
      "ConSan MOI sampled probe could not commit causal window metadata");
  const uint64_t selected_count_address =
      plan.report_buffer_address + offsetof(ConSanMoiReportHeader, sampled_causal_window_count);
  require_emission(append_atomic_fetch_add_one_u32(words, selected_count_address, tmp_vgpr,
                                                   plan.scratch_vgpr, arch),
                   "ConSan MOI sampled probe could not count the claimed causal window");
  sequence.append(instrumentation::build_s_mov_b64(kAmdGpuExecLo, publication_exec_save_sgpr, arch))
      .branch(restore_label, InstructionSequence::BranchKind::Unconditional)
      .bind_label(collision_label)
      .append(instrumentation::build_s_mov_b64(publication_exec_save_sgpr, kAmdGpuExecLo, arch));

  // Repeated executions of the same selected causal window are idempotent:
  // the static slot already retains their representative sample. Only a
  // different exact dynamic identity occupying the slot is capacity loss.
  sequence.require(append_sampled_banked_address(
      words, causal_window_address, sizeof(ConSanMoiSampledCausalWindow), plan.window_bank_count,
      bank_vgpr, plan.scratch_vgpr, arch));
  const auto reject_unless_equal_vgpr = [&](uint32_t offset, uint16_t expected_vgpr) {
    sequence.require(record.load(offset, low_vgpr))
        .append(instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(expected_vgpr), low_vgpr,
                                                        arch))
        .branch(different_identity_label, InstructionSequence::BranchKind::VccZero);
  };
  const auto reject_unless_equal_literal = [&](uint32_t offset, uint32_t literal) {
    sequence.append(instrumentation::build_v_mov_b32_literal(high_vgpr, literal, arch));
    reject_unless_equal_vgpr(offset, high_vgpr);
  };
  const auto reject_unless_equal_dispatch_id = [&](uint32_t offset, bool high_word) {
    sequence.require(record.load(offset, low_vgpr))
        .require(append_compare_moi_report_dispatch_id_word(words, plan.dispatch_id, low_vgpr,
                                                            high_vgpr, high_word, arch))
        .branch(different_identity_label, InstructionSequence::BranchKind::VccZero);
  };
  const auto reject_unless_equal_workgroup = [&](uint32_t offset,
                                                 const ConSanMoiWorkgroupSource &source) {
    if (!source.scalar_src) {
      reject_unless_equal_literal(offset, 0u);
      return;
    }
    sequence.require(consan_detail::append_workgroup_source_value(words, source, high_vgpr, arch));
    reject_unless_equal_vgpr(offset, high_vgpr);
  };
  const auto reject_unless_equal_dynamic_record_index = [&]() {
    if (plan.window_bank_count == 1u) {
      reject_unless_equal_literal(offsetof(ConSanMoiSampledCausalWindow, first_entry),
                                  record_index);
      return;
    }
    sequence.append(
        instrumentation::build_v_add_u32_literal(high_vgpr, record_index, bank_vgpr, arch));
    reject_unless_equal_vgpr(offsetof(ConSanMoiSampledCausalWindow, first_entry), high_vgpr);
  };
  const bool began_identity_comparison = static_cast<bool>(sequence);
  reject_unless_equal_literal(offsetof(ConSanMoiSampledCausalWindow, generation),
                              static_cast<uint32_t>(plan.report_generation));
  reject_unless_equal_literal(offsetof(ConSanMoiSampledCausalWindow, generation) + 4u,
                              static_cast<uint32_t>(plan.report_generation >> 32u));
  reject_unless_equal_dispatch_id(offsetof(ConSanMoiSampledCausalWindow, dispatch_id),
                                  /*high_word=*/false);
  reject_unless_equal_dispatch_id(offsetof(ConSanMoiSampledCausalWindow, dispatch_id) + 4u,
                                  /*high_word=*/true);
  reject_unless_equal_workgroup(offsetof(ConSanMoiSampledCausalWindow, workgroup_x),
                                plan.workgroup_sources.x);
  reject_unless_equal_workgroup(offsetof(ConSanMoiSampledCausalWindow, workgroup_y),
                                plan.workgroup_sources.y);
  reject_unless_equal_workgroup(offsetof(ConSanMoiSampledCausalWindow, workgroup_z),
                                plan.workgroup_sources.z);
  if (plan.owner_epoch_vgprs.epoch)
    reject_unless_equal_vgpr(offsetof(ConSanMoiSampledCausalWindow, epoch),
                             *plan.owner_epoch_vgprs.epoch);
  else
    reject_unless_equal_literal(offsetof(ConSanMoiSampledCausalWindow, epoch), 0u);
  reject_unless_equal_dynamic_record_index();
  reject_unless_equal_literal(offsetof(ConSanMoiSampledCausalWindow, entry_count), 1u);
  reject_unless_equal_literal(offsetof(ConSanMoiSampledCausalWindow, publication_state),
                              static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Ready));
  reject_unless_equal_workgroup(offsetof(ConSanMoiSampledCausalWindow, cluster_workgroup_id),
                                plan.workgroup_sources.cluster_workgroup_id);
  if (began_identity_comparison && !sequence) {
    errors.emplace_back("ConSan MOI sampled probe could not compare a repeated causal identity");
  }
  sequence.append(instrumentation::build_s_mov_b64(kAmdGpuExecLo, publication_exec_save_sgpr, arch))
      .branch(restore_label, InstructionSequence::BranchKind::Unconditional)
      .bind_label(different_identity_label);
  require_emission(append_select_first_lane_in_exec_mask(words, tmp_vgpr,
                                                         publication_exec_save_sgpr,
                                                         publication_exec_save_sgpr, arch),
                   "ConSan MOI sampled probe could not select a collision representative");
  const uint64_t saturated_count_address =
      plan.report_buffer_address + offsetof(ConSanMoiReportHeader, sampled_saturated_window_count);
  require_emission(append_atomic_fetch_add_one_u32(words, saturated_count_address, tmp_vgpr,
                                                   plan.scratch_vgpr, arch),
                   "ConSan MOI sampled probe could not count a saturated causal window claim");
  sequence.append(instrumentation::build_s_mov_b64(kAmdGpuExecLo, publication_exec_save_sgpr, arch))
      .bind_label(restore_label);
  require_emission.append(
      "ConSan MOI sampled probe could not restore guest EXEC/VCC/SCC",
      instrumentation::build_s_mov_b64(kAmdGpuExecLo, original_exec_save_sgpr, arch),
      instrumentation::build_s_mov_b64(kAmdGpuVccLo, selection_vcc_save_sgpr, arch),
      instrumentation::build_s_cmp_lg_u32(publication_scc_save_sgpr, scalar_positive_inline_u32(0),
                                          arch));
  if (!sequence.finish(arch)) {
    errors.emplace_back("ConSan MOI sampled probe local branch is out of range");
    return std::nullopt;
  }
  return words;
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_direct_sampled_watchpoint_words(
    std::span<const uint8_t> bytes, const ConSanMoiCandidate &candidate,
    uint64_t owner_descriptor_file_offset, const MoiSampledAccessEmissionPlan &plan,
    const VgprSpillSequence *spill, rj_code_arch_t arch, std::vector<std::string> &errors,
    uint32_t *guest_instruction_offset, uint32_t *guest_instruction_word_count) {
  const auto &access_ranges = candidate.site().ranges;
  if (access_ranges.empty()) {
    errors.emplace_back("ConSan MOI sampled probe requires at least one LDS access range");
    return std::nullopt;
  }
  const ConSanTargetProfile *target = consan_target_profile(arch);
  if (target == nullptr) {
    errors.emplace_back("ConSan MOI sampled probe has no target profile");
    return std::nullopt;
  }
  std::vector<uint32_t> words;
  InstructionSequence sequence(words);
  const bool select_guest_vgpr_bank =
      target->has_selectable_vgpr_bank && candidate.incoming_vgpr_bank_mode.value_or(0u) != 0u;
  const bool capture_high_bank_address =
      moi_access_requires_high_bank_address_capture(candidate, arch);
  if (plan.spill_backed_operand_recovery &&
      (spill == nullptr || !plan.supports_native_lds_spill_recovery)) {
    errors.emplace_back("ConSan MOI sampled probe has an invalid spill-backed operand plan");
    return std::nullopt;
  }
  if (plan.spill_backed_operand_recovery && select_guest_vgpr_bank) {
    errors.emplace_back(
        "ConSan MOI sampled spill-backed operand recovery cannot change the guest VGPR bank");
    return std::nullopt;
  }
  // A high SRC0 bank is captured by the appended-body wrapper before it
  // selects bank zero for instrumentation scratch. Other banked operands keep
  // the older self-contained transition sequence.
  if (select_guest_vgpr_bank && !capture_high_bank_address)
    if (!sequence.emit(instrumentation::build_s_set_vgpr_msb_transition(
            static_cast<uint8_t>(*candidate.incoming_vgpr_bank_mode), 0u, arch)))
      return std::nullopt;
  std::optional<uint16_t> preserved_lds_byte_offset_vgpr;
  std::optional<uint16_t> spilled_lds_byte_offset_vgpr;
  const bool reserve_two_address_replay_scratch =
      moi_guest_access_relocation_requires_adjusted_address(candidate, *target);
  if (capture_high_bank_address) {
    preserved_lds_byte_offset_vgpr =
        static_cast<uint16_t>(plan.scratch_vgpr + plan.base_scratch_vgpr_count);
  }
  if (plan.spill_backed_operand_recovery) {
    const auto lds_byte_offset_vgpr = candidate_lds_byte_offset_vgpr(candidate, errors);
    if (!lds_byte_offset_vgpr)
      return std::nullopt;
    const bool spilled_address =
        range_overlaps(*lds_byte_offset_vgpr, 1u, spill->vgpr_base, spill->vgpr_count);
    if (candidate.site().lowering.form->destination_vgpr &&
        range_overlaps(*candidate.site().lowering.form->destination_vgpr,
                       candidate_payload_vgpr_count(candidate), spill->vgpr_base,
                       spill->vgpr_count)) {
      errors.emplace_back(
          "ConSan MOI sampled spill-backed operand recovery overlaps the load destination");
      return std::nullopt;
    }
    if (spilled_address) {
      spilled_lds_byte_offset_vgpr = *lds_byte_offset_vgpr;
    } else if (moi_load_clobbers_address(candidate)) {
      preserved_lds_byte_offset_vgpr =
          static_cast<uint16_t>(plan.scratch_vgpr + plan.base_scratch_vgpr_count);
    }
    if (moi_load_clobbers_address(candidate) && !spilled_address) {
      words.push_back(build_v_mov_b32_e32(*preserved_lds_byte_offset_vgpr,
                                          vector_source_vgpr(*lds_byte_offset_vgpr), arch));
    }
    if (guest_instruction_offset)
      *guest_instruction_offset = static_cast<uint32_t>(words.size() * sizeof(uint32_t));
    const uint16_t replay_address_vgpr =
        preserved_lds_byte_offset_vgpr.value_or(*candidate.site().lowering.form->address_vgpr);
    if (!append_moi_relocated_guest_access(
            words, bytes, candidate, target, replay_address_vgpr,
            reserve_two_address_replay_scratch
                ? std::optional<uint16_t>(
                      static_cast<uint16_t>(plan.scratch_vgpr + plan.scratch_vgpr_count - 1u))
                : std::nullopt,
            errors, guest_instruction_word_count))
      return std::nullopt;
    if (!append_moi_lds_wait(words, arch)) {
      errors.emplace_back("ConSan MOI sampled spill-backed operand recovery needs an LDS wait");
      return std::nullopt;
    }
  }
  if (plan.private_epoch_offset) {
    if (!plan.automatic_private_epoch || !plan.owner_epoch_vgprs.epoch) {
      errors.emplace_back("ConSan MOI sampled probe has an invalid private epoch plan");
      return std::nullopt;
    }
    if (!sequence.emit_all(instrumentation::build_private_load_b32(
                               *plan.owner_epoch_vgprs.epoch, *plan.private_epoch_offset, arch),
                           instrumentation::build_s_wait_private_load0(arch))) {
      errors.emplace_back("ConSan MOI sampled probe could not load private epoch state");
      return std::nullopt;
    }
  }
  if (plan.owner_derivation) {
    if (!plan.automatic_private_epoch || !plan.owner_epoch_vgprs.owner ||
        !plan.owner_derivation->entry_workitem_x_private_offset) {
      errors.emplace_back("ConSan MOI sampled probe has an invalid private owner plan");
      return std::nullopt;
    }
    const auto owner = build_moi_workitem_owner_derivation(
        *plan.owner_derivation, *plan.owner_epoch_vgprs.owner, arch, "sampled probe", errors);
    if (!owner || !sequence.emit(owner->words))
      return std::nullopt;
  }
  if (plan.persistent_sgprs.complete()) {
    if (!consan_detail::validate_scalar_state_temporaries(
            plan.persistent_sgprs, plan.owner_epoch_vgprs, "sampled probe", errors))
      return std::nullopt;
    words.push_back(
        build_v_mov_b32_e32(*plan.owner_epoch_vgprs.owner, *plan.persistent_sgprs.owner(), arch));
    words.push_back(
        build_v_mov_b32_e32(*plan.owner_epoch_vgprs.epoch, *plan.persistent_sgprs.epoch(), arch));
  }
  const bool materialize_flat_address = candidate_requires_flat_address_materialization(candidate);
  if (!preserved_lds_byte_offset_vgpr &&
      (moi_load_clobbers_address(candidate) || materialize_flat_address)) {
    preserved_lds_byte_offset_vgpr =
        static_cast<uint16_t>(plan.scratch_vgpr + plan.base_scratch_vgpr_count);
    const auto lds_byte_offset_vgpr = candidate_lds_byte_offset_vgpr(candidate, errors);
    if (!lds_byte_offset_vgpr)
      return std::nullopt;
    words.push_back(build_v_mov_b32_e32(*preserved_lds_byte_offset_vgpr,
                                        vector_source_vgpr(*lds_byte_offset_vgpr), arch));
    if (materialize_flat_address && !candidate_uses_scalar_vector_flat_address(candidate)) {
      words.push_back(build_v_mov_b32_e32(
          static_cast<uint16_t>(*preserved_lds_byte_offset_vgpr + 1u),
          vector_source_vgpr(static_cast<uint16_t>(*lds_byte_offset_vgpr + 1u)), arch));
    }
  }

  for (uint32_t range_index = 0; range_index < access_ranges.size(); ++range_index) {
    std::vector<uint32_t> prior_record_indices;
    if (plan.prior_first_record_index) {
      prior_record_indices.reserve(plan.prior_access_range_count);
      for (uint32_t prior_range_index = 0; prior_range_index < plan.prior_access_range_count;
           ++prior_range_index) {
        prior_record_indices.push_back(*plan.prior_first_record_index +
                                       prior_range_index * plan.window_bank_count);
      }
    }
    uint32_t range_guest_instruction_offset = 0;
    uint32_t range_guest_instruction_word_count = 0;
    auto range_words = build_direct_sampled_watchpoint_range_words(
        bytes, candidate, owner_descriptor_file_offset, access_ranges[range_index], plan, arch,
        plan.first_record_index + range_index * plan.window_bank_count, prior_record_indices,
        range_index == 0u && !plan.spill_backed_operand_recovery, preserved_lds_byte_offset_vgpr,
        spill, spilled_lds_byte_offset_vgpr, errors,
        range_index == 0u ? &range_guest_instruction_offset : nullptr,
        range_index == 0u ? &range_guest_instruction_word_count : nullptr);
    if (!range_words)
      return std::nullopt;
    if (!plan.spill_backed_operand_recovery && range_index == 0u && guest_instruction_offset) {
      *guest_instruction_offset =
          static_cast<uint32_t>(words.size() * sizeof(uint32_t)) + range_guest_instruction_offset;
    }
    if (!plan.spill_backed_operand_recovery && range_index == 0u && guest_instruction_word_count)
      *guest_instruction_word_count = range_guest_instruction_word_count;
    if (!sequence.emit(*range_words))
      return std::nullopt;
  }
  if (select_guest_vgpr_bank && !capture_high_bank_address)
    if (!sequence.emit(instrumentation::build_s_set_vgpr_msb_transition(
            0u, static_cast<uint8_t>(*candidate.incoming_vgpr_bank_mode), arch)))
      return std::nullopt;
  return words;
}

} // namespace rocjitsu::consan_moi_impl
