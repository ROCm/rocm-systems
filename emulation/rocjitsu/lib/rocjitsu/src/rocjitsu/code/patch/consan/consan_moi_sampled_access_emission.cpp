// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi_sampled_access_emission.h"

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

using consan_detail::append_reload_moi_spilled_vgpr;
using consan_detail::build_moi_relocated_guest_access_words;
using consan_detail::moi_guest_access_relocation_requires_adjusted_address;
using consan_detail::moi_spilled_vgpr_reload_result_name;
using consan_detail::MoiSpilledVgprReloadResult;
using consan_detail::MoiWorkitemOwnerDerivationPlan;
using consan_detail::range_overlaps;
using consan_detail::reject_optional_scratch_range_overlap;
using consan_moi_detail::append_atomic_fetch_add_one_u32;
using consan_moi_detail::append_compare_moi_report_dispatch_id_word;
using consan_moi_detail::append_load_u32_vgpr_at_offset;
using consan_moi_detail::append_store_moi_report_dispatch_id_pair;
using consan_moi_detail::ConSanMoiRecordEmitter;

[[nodiscard]] bool append_sampled_banked_address(std::vector<uint32_t> &words,
                                                 uint64_t first_address, uint32_t element_size,
                                                 uint32_t bank_count, uint16_t bank_vgpr,
                                                 uint16_t address_vgpr, rj_code_arch_t arch);

[[nodiscard]] bool append_direct_sampled_immediate_check(
    std::vector<uint32_t> &words, const ConSanRequest &request,
    const BoundRuntimeResources &resources, uint16_t scratch_vgpr, rj_code_arch_t arch,
    uint32_t prior_record_index, uint32_t window_bank_count, uint16_t bank_vgpr,
    size_t sampled_watchpoints_offset, uint16_t current_low_vgpr, uint16_t current_high_vgpr,
    std::vector<std::string> &errors) {
  if (!request.moi_sampled_check)
    return true;
  if (!resources.moi_report_buffer_address) {
    errors.emplace_back("ConSan MOI sampled checker is missing report or vector resources");
    return false;
  }

  const uint16_t address_lo_vgpr = scratch_vgpr;
  const uint16_t address_hi_vgpr = static_cast<uint16_t>(scratch_vgpr + 1u);
  const uint16_t tmp_vgpr = static_cast<uint16_t>(scratch_vgpr + 4u);
  const uint16_t prior_low_vgpr = static_cast<uint16_t>(scratch_vgpr + 5u);
  const uint16_t prior_high_vgpr = static_cast<uint16_t>(scratch_vgpr + 6u);
  if (window_bank_count == 0) {
    errors.emplace_back("ConSan MOI sampled checker has an invalid prior bank");
    return false;
  }
  const uint64_t prior_address = *resources.moi_report_buffer_address + sampled_watchpoints_offset +
                                 static_cast<uint64_t>(prior_record_index) * sizeof(uint64_t);
  InstructionSequence sequence(words);
  const auto done_label = sequence.make_label();

  const auto zero_low = instrumentation::build_v_mov_b32_literal(prior_low_vgpr, 0u, arch);
  const auto zero_high = instrumentation::build_v_mov_b32_literal(prior_high_vgpr, 0u, arch);
  const auto atomic_snapshot = instrumentation::build_flat_atomic_add_u64(
      address_lo_vgpr, prior_low_vgpr, prior_low_vgpr,
      /*return_old_value=*/true, kAmdGpuScopeDevice, arch);
  if (!zero_low || !zero_high || !atomic_snapshot ||
      !append_sampled_banked_address(words, prior_address, sizeof(uint64_t), window_bank_count,
                                     bank_vgpr, address_lo_vgpr, arch)) {
    errors.emplace_back("ConSan MOI sampled checker could not atomically snapshot the prior slot");
    return false;
  }
  words.insert(words.end(), zero_low->begin(), zero_low->end());
  words.insert(words.end(), zero_high->begin(), zero_high->end());
  words.insert(words.end(), atomic_snapshot->begin(), atomic_snapshot->end());
  if (!append_moi_global_atomic_wait(words, arch))
    return false;
  auto append_required_predicate = [&](std::optional<uint32_t> compare) {
    if (!compare)
      return false;
    words.push_back(*compare);
    return sequence.emit_branch(done_label, InstructionSequence::BranchKind::VccZero);
  };

  const auto prior_valid =
      instrumentation::build_v_and_b32_literal(tmp_vgpr, 1u, prior_low_vgpr, arch);
  if (!prior_valid) {
    errors.emplace_back("ConSan MOI sampled checker could not decode validity");
    return false;
  }
  words.insert(words.end(), prior_valid->begin(), prior_valid->end());
  if (!append_required_predicate(
          instrumentation::build_v_cmp_ne_u32_vcc(scalar_positive_inline_u32(0), tmp_vgpr, arch)) ||
      !append_extract_exact_shadow_field(words, address_lo_vgpr, current_low_vgpr,
                                         consan_moi_sampled_watchpoint::owner_shift,
                                         consan_moi_sampled_watchpoint::max_owner, arch) ||
      !append_extract_exact_shadow_field(words, address_hi_vgpr, prior_low_vgpr,
                                         consan_moi_sampled_watchpoint::owner_shift,
                                         consan_moi_sampled_watchpoint::max_owner, arch) ||
      !append_required_predicate(instrumentation::build_v_cmp_ne_u32_vcc(
          vector_source_vgpr(address_lo_vgpr), address_hi_vgpr, arch))) {
    errors.emplace_back("ConSan MOI sampled checker could not compare owners");
    return false;
  }

  const uint32_t low_epoch_generation_mask =
      static_cast<uint32_t>(consan_moi_sampled_watchpoint::epoch_generation_mask);
  const uint32_t high_generation_mask =
      static_cast<uint32_t>(consan_moi_sampled_watchpoint::generation_mask >> 32u);
  const uint32_t high_range_mask =
      static_cast<uint32_t>((consan_moi_sampled_watchpoint::start_byte_mask |
                             consan_moi_sampled_watchpoint::count_mask) >>
                            32u);
  auto append_equal_masked = [&](uint16_t current_vgpr, uint16_t prior_vgpr, uint32_t mask) {
    const auto current =
        instrumentation::build_v_and_b32_literal(address_lo_vgpr, mask, current_vgpr, arch);
    const auto prior =
        instrumentation::build_v_and_b32_literal(address_hi_vgpr, mask, prior_vgpr, arch);
    if (!current || !prior)
      return false;
    words.insert(words.end(), current->begin(), current->end());
    words.insert(words.end(), prior->begin(), prior->end());
    return append_required_predicate(instrumentation::build_v_cmp_eq_u32_vcc(
        vector_source_vgpr(address_lo_vgpr), address_hi_vgpr, arch));
  };
  if (!append_equal_masked(current_low_vgpr, prior_low_vgpr, low_epoch_generation_mask) ||
      !append_equal_masked(current_high_vgpr, prior_high_vgpr, high_generation_mask) ||
      !append_equal_masked(current_high_vgpr, prior_high_vgpr, high_range_mask)) {
    errors.emplace_back("ConSan MOI sampled checker could not compare ordering or ranges");
    return false;
  }

  if (!append_extract_exact_shadow_field(
          words, address_lo_vgpr, current_low_vgpr,
          consan_moi_sampled_watchpoint::access_kind_shift,
          (1u << consan_moi_sampled_watchpoint::access_kind_bits) - 1u, arch) ||
      !append_extract_exact_shadow_field(
          words, address_hi_vgpr, prior_low_vgpr, consan_moi_sampled_watchpoint::access_kind_shift,
          (1u << consan_moi_sampled_watchpoint::access_kind_bits) - 1u, arch)) {
    errors.emplace_back("ConSan MOI sampled checker could not decode access kinds");
    return false;
  }
  const auto both_read = instrumentation::build_v_and_b32(
      tmp_vgpr, vector_source_vgpr(address_lo_vgpr), address_hi_vgpr, arch);
  if (!both_read) {
    errors.emplace_back("ConSan MOI sampled checker could not compare access kinds");
    return false;
  }
  words.push_back(*both_read);
  if (!append_required_predicate(instrumentation::build_v_cmp_ne_u32_vcc(
          scalar_positive_inline_u32(static_cast<uint32_t>(ConSanMoiShadowAccessKind::Read)),
          tmp_vgpr, arch))) {
    errors.emplace_back("ConSan MOI sampled checker could not predicate conflicting kinds");
    return false;
  }

  const uint64_t immediate_conflict_count_address =
      *resources.moi_report_buffer_address + offsetof(ConSanMoiReportHeader, event_counter);
  if (!append_atomic_fetch_add_one_u32(words, immediate_conflict_count_address, tmp_vgpr,
                                       scratch_vgpr, arch)) {
    errors.emplace_back("ConSan MOI sampled checker could not increment the conflict counter");
    return false;
  }

  // VCC is instrumentation-local here. The sampled body's common exit
  // restores guest VCC from its dedicated snapshot, so saving VCC in this
  // helper is both unnecessary and unsafe: moi_exec_save_sgpr is the
  // immutable guest-EXEC snapshot which that exit later consumes.
  if (!sequence.bind(done_label))
    return false;
  if (!sequence.resolve_branches(arch)) {
    errors.emplace_back("ConSan MOI sampled checker skip branch is out of range");
    return false;
  }
  return true;
}

[[nodiscard]] bool
append_sampled_window_bank_index(std::vector<uint32_t> &words, const ConSanMoiOperatingPoint &point,
                                 const BoundRuntimeResources &resources,
                                 const ConSanMoiWorkgroupSources &workgroup_sources,
                                 uint32_t bank_count, uint16_t bank_vgpr, uint16_t temporary_vgpr,
                                 uint16_t owner_vgpr, rj_code_arch_t arch) {
  if (bank_count == 0 || !std::has_single_bit(bank_count))
    return false;
  if (bank_count == 1) {
    const auto zero = instrumentation::build_v_mov_b32_literal(bank_vgpr, 0, arch);
    if (!zero)
      return false;
    words.insert(words.end(), zero->begin(), zero->end());
    return true;
  }
  if (!append_moi_report_dispatch_id_pair(
          words, consan_moi_detail::moi_bound_dispatch_id_sources({point, resources}), bank_vgpr,
          temporary_vgpr, arch)) {
    return false;
  }
  const auto mix = [&](uint16_t source) {
    const auto word =
        instrumentation::build_v_xor_b32(bank_vgpr, vector_source_vgpr(source), bank_vgpr, arch);
    if (!word)
      return false;
    words.push_back(*word);
    return true;
  };
  if (!mix(temporary_vgpr))
    return false;
  const std::array<std::pair<const ConSanMoiWorkgroupSource *, uint32_t>, 4> tuple_sources = {{
      {&workgroup_sources.x, 0u},
      {&workgroup_sources.y, 1u},
      {&workgroup_sources.z, 2u},
      {&workgroup_sources.cluster_workgroup_id, 3u},
  }};
  for (const auto &[source, dimension_shift] : tuple_sources) {
    if (!source->has_value())
      continue;
    if (!consan_detail::append_workgroup_source_value(words, *source, temporary_vgpr, arch))
      return false;
    if (dimension_shift != 0u) {
      const auto distinguish_dimension = instrumentation::build_v_lshlrev_b32(
          temporary_vgpr, scalar_positive_inline_u32(dimension_shift), temporary_vgpr, arch);
      if (!distinguish_dimension)
        return false;
      words.push_back(*distinguish_dimension);
    }
    if (!mix(temporary_vgpr))
      return false;
  }
  // A causal-window bank is an evidence-retention bucket, not a workgroup
  // identity. Include the wave owner so several waves touching the same
  // sampled address can retain independent windows instead of racing for one
  // first-publisher slot. Barrier and atomic paths use the same owner-aware
  // hash, so synchronization metadata remains joined to the corresponding
  // access window.
  if (!mix(owner_vgpr))
    return false;
  const auto shift = instrumentation::build_v_lshrrev_b32(
      temporary_vgpr, scalar_positive_inline_u32(16), bank_vgpr, arch);
  if (!shift)
    return false;
  words.push_back(*shift);
  if (!mix(temporary_vgpr))
    return false;
  const auto multiply = instrumentation::build_v_mul_lo_u32_literal(bank_vgpr, temporary_vgpr,
                                                                    0x85ebca6bu, bank_vgpr, arch);
  if (!multiply)
    return false;
  words.insert(words.end(), multiply->begin(), multiply->end());
  const auto mask =
      instrumentation::build_v_and_b32_literal(bank_vgpr, bank_count - 1u, bank_vgpr, arch);
  if (!mask)
    return false;
  words.insert(words.end(), mask->begin(), mask->end());
  return true;
}

[[nodiscard]] bool append_sampled_workgroup_residue(
    std::vector<uint32_t> &words, const ConSanRequest &request,
    const ConSanMoiOperatingPoint &point, const BoundRuntimeResources &resources,
    const ConSanMoiWorkgroupSources &workgroup_sources, uint16_t residue_vgpr,
    uint16_t temporary_vgpr, uint16_t coordinate_vgpr, rj_code_arch_t arch) {
  if (!append_moi_report_dispatch_id_pair(
          words, consan_moi_detail::moi_bound_dispatch_id_sources({point, resources}), residue_vgpr,
          temporary_vgpr, arch)) {
    return false;
  }
  const auto mix = [&](uint16_t source) {
    const auto word = instrumentation::build_v_xor_b32(residue_vgpr, vector_source_vgpr(source),
                                                       residue_vgpr, arch);
    if (!word)
      return false;
    words.push_back(*word);
    return true;
  };
  if (!mix(temporary_vgpr))
    return false;
  const std::array<std::pair<const ConSanMoiWorkgroupSource *, uint32_t>, 4> tuple_sources = {{
      {&workgroup_sources.x, 0u},
      {&workgroup_sources.y, 1u},
      {&workgroup_sources.z, 2u},
      {&workgroup_sources.cluster_workgroup_id, 3u},
  }};
  for (const auto &[source, dimension_shift] : tuple_sources) {
    if (!source->has_value())
      continue;
    if (!consan_detail::append_workgroup_source_value(words, *source, coordinate_vgpr, arch))
      return false;
    if (dimension_shift != 0u) {
      const auto distinguish_dimension = instrumentation::build_v_lshlrev_b32(
          coordinate_vgpr, scalar_positive_inline_u32(dimension_shift), coordinate_vgpr, arch);
      if (!distinguish_dimension)
        return false;
      words.push_back(*distinguish_dimension);
    }
    if (!mix(coordinate_vgpr))
      return false;
  }
  const auto shift = instrumentation::build_v_lshrrev_b32(
      temporary_vgpr, scalar_positive_inline_u32(16), residue_vgpr, arch);
  if (!shift)
    return false;
  words.push_back(*shift);
  if (!mix(temporary_vgpr))
    return false;
  const auto multiply = instrumentation::build_v_mul_lo_u32_literal(
      residue_vgpr, temporary_vgpr, 0x85ebca6bu, residue_vgpr, arch);
  const auto mask = instrumentation::build_v_and_b32_literal(
      residue_vgpr, request.moi_runtime_sample_stride - 1u, residue_vgpr, arch);
  if (!multiply || !mask)
    return false;
  words.insert(words.end(), multiply->begin(), multiply->end());
  words.insert(words.end(), mask->begin(), mask->end());
  return true;
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

  const auto low = instrumentation::build_v_mov_b32_literal(
      address_vgpr, static_cast<uint32_t>(first_address), arch);
  const auto high = instrumentation::build_v_mov_b32_literal(
      static_cast<uint16_t>(address_vgpr + 1u), static_cast<uint32_t>(first_address >> 32u), arch);
  if (!low || !high)
    return false;
  std::vector<uint32_t> emitted;
  emitted.insert(emitted.end(), low->begin(), low->end());
  emitted.insert(emitted.end(), high->begin(), high->end());
  words.insert(words.end(), emitted.begin(), emitted.end());
  return true;
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_direct_sampled_watchpoint_range_words(
    std::span<const uint8_t> bytes, const ConSanMoiCandidate &candidate,
    const ConSanAccessRange &access_range, const ConSanRequest &request,
    const BoundRuntimeResources &bound_resources, const ConSanMoiOperatingPoint &point,
    const ConSanMoiWorkgroupSources &workgroup_sources,
    const ConSanMoiOwnerEpochVgprSources &owner_epoch_vgprs, uint16_t scratch_vgpr,
    rj_code_arch_t arch, uint32_t record_index, std::span<const uint32_t> prior_record_indices,
    uint32_t window_bank_count, bool spill_overlaps_guest_operands,
    bool spill_backed_operand_recovery, bool relocate_instruction,
    uint32_t pending_acquire_owner_bank_count,
    std::optional<uint16_t> preserved_lds_byte_offset_vgpr, const VgprSpillSequence *spill,
    std::optional<uint16_t> spilled_lds_byte_offset_vgpr, size_t sampled_causal_windows_offset,
    size_t sampled_watchpoints_offset, size_t sampled_pending_acquires_offset,
    bool runtime_workgroup_gate_in_body, std::vector<std::string> &errors,
    uint32_t *guest_instruction_offset = nullptr,
    uint32_t *guest_instruction_word_count = nullptr) {
  const ConSanTargetProfile *target = consan_target_profile(arch);
  if (target == nullptr) {
    errors.emplace_back("ConSan MOI sampled probe has no target profile");
    return std::nullopt;
  }
  if (pending_acquire_owner_bank_count == 0u ||
      !std::has_single_bit(pending_acquire_owner_bank_count) ||
      pending_acquire_owner_bank_count > kConSanMoiSampledPendingAcquireOwnerBankCount) {
    errors.emplace_back("ConSan MOI sampled probe has an invalid pending-acquire owner bank");
    return std::nullopt;
  }
  // A runtime fast-gate lowers the cave body with stride=1 after the outer
  // scalar gate has made the sampling decision.  Multi-bank publication still
  // needs its bank index to survive all ordinary temporaries in that cave.
  // Reserve the dedicated bank VGPR based on the emitted layout, not solely on
  // whether the inner vector sampler remains enabled.
  const bool dedicated_bank_vgpr = request.moi_runtime_sample_stride > 1 || window_bank_count > 1;
  const uint16_t base_scratch_count = static_cast<uint16_t>((request.moi_sampled_check ? 7u : 5u) +
                                                            (dedicated_bank_vgpr ? 1u : 0u));
  const uint16_t scratch_count = static_cast<uint16_t>(
      (spill_backed_operand_recovery
           ? sampled_spill_backed_scratch_count(request, point, candidate, arch)
           : direct_sampled_scratch_count(request, point, candidate, arch)));
  const bool reserve_two_address_replay_scratch =
      moi_guest_access_relocation_requires_adjusted_address(candidate, *target);
  if (static_cast<uint32_t>(scratch_vgpr) + scratch_count > kMaxVgprs) {
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
  if (!spill_overlaps_guest_operands &&
      reject_candidate_scratch_range_overlap(candidate, scratch_vgpr, scratch_count, errors))
    return std::nullopt;
  if ((!point.automatic_moi_private_epoch && !point.moi_persistent_sgprs.complete() &&
       reject_optional_scratch_range_overlap(owner_epoch_vgprs.owner, scratch_vgpr, scratch_count,
                                             "MOI owner", errors)) ||
      (!point.automatic_moi_private_epoch && !point.moi_persistent_sgprs.complete() &&
       reject_optional_scratch_range_overlap(owner_epoch_vgprs.epoch, scratch_vgpr, scratch_count,
                                             "MOI epoch", errors)))
    return std::nullopt;
  std::optional<uint16_t> derived_owner_vgpr;
  std::optional<uint32_t> derived_owner_word;
  if (!owner_epoch_vgprs.owner && candidate.kernel_descriptor_file_offset) {
    const auto owner_shift =
        moi_descriptor_owner_shift(bytes, *candidate.kernel_descriptor_file_offset, arch, errors);
    if (!owner_shift)
      return std::nullopt;
    const uint16_t value_vgpr = static_cast<uint16_t>(scratch_vgpr + 4u);
    const auto owner_init = instrumentation::build_v_lshrrev_b32(
        value_vgpr, scalar_positive_inline_u32(*owner_shift), kAmdGpuWorkitemIdX, arch);
    if (!owner_init) {
      errors.emplace_back("ConSan MOI sampled probe could not encode owner derivation");
      return std::nullopt;
    }
    derived_owner_vgpr = value_vgpr;
    derived_owner_word = *owner_init;
  }
  const uint16_t low_vgpr = static_cast<uint16_t>(scratch_vgpr + 2u);
  const uint16_t high_vgpr = static_cast<uint16_t>(scratch_vgpr + 3u);
  const uint16_t tmp_vgpr = static_cast<uint16_t>(scratch_vgpr + 4u);
  const uint16_t owner_vgpr = owner_epoch_vgprs.owner.value_or(derived_owner_vgpr.value_or(0));
  const bool runtime_sampled = request.moi_runtime_sample_stride > 1;
  const uint16_t bank_vgpr =
      dedicated_bank_vgpr
          ? static_cast<uint16_t>(scratch_vgpr + (request.moi_sampled_check ? 7u : 5u))
          : tmp_vgpr;
  if (runtime_sampled && !owner_epoch_vgprs.owner && !derived_owner_vgpr) {
    errors.emplace_back("ConSan MOI runtime sampled probe could not derive a wave owner");
    return std::nullopt;
  }
  if (!point.moi_exec_save_sgpr || *point.moi_exec_save_sgpr > 98u ||
      *point.moi_exec_save_sgpr % 2u != 0u) {
    errors.emplace_back(
        "ConSan MOI sampled probe needs seven aligned VCC/EXEC/SCC-save SGPRs in 0..104");
    return std::nullopt;
  }
  const auto kind = consan_moi_shadow_kind_from_access_kind(candidate.kind);
  const uint32_t generation =
      bound_resources.moi_report_generation & consan_moi_sampled_watchpoint::max_generation;
  const uint64_t generation_field = static_cast<uint64_t>(generation)
                                    << consan_moi_sampled_watchpoint::generation_shift;
  const uint32_t low_literal = static_cast<uint32_t>(
      consan_moi_sampled_watchpoint::valid_mask |
      (static_cast<uint64_t>(kind) << consan_moi_sampled_watchpoint::access_kind_shift) |
      generation_field);
  const uint32_t encoded_byte_count = encode_consan_moi_sampled_byte_count(access_range.byte_width)
                                      << (consan_moi_sampled_watchpoint::count_shift - 32u);
  const uint32_t encoded_generation_high = static_cast<uint32_t>(generation_field >> 32u);
  const uint64_t sampled_entry_address = *bound_resources.moi_report_buffer_address +
                                         sampled_watchpoints_offset +
                                         static_cast<uint64_t>(record_index) * sizeof(uint64_t);
  const uint64_t pending_acquire_address =
      *bound_resources.moi_report_buffer_address + sampled_pending_acquires_offset +
      static_cast<uint64_t>(record_index) * pending_acquire_owner_bank_count *
          sizeof(ConSanMoiSampledPendingAcquireSlot);

  std::vector<uint32_t> words;
  words.reserve(candidate.size() / sizeof(uint32_t) + 40u + (derived_owner_word ? 1u : 0u) +
                (owner_epoch_vgprs.epoch ? 2u : 0u) +
                (relocate_instruction && moi_load_clobbers_address(candidate) &&
                         !preserved_lds_byte_offset_vgpr
                     ? 1u
                     : 0u) +
                (candidate.lowering_offset(access_range) != 0u ? 3u : 0u));
  InstructionSequence sequence(words);
  const auto collision_label = sequence.make_label();
  const auto different_identity_label = sequence.make_label();
  const auto restore_label = sequence.make_label();
  ConSanMoiRecordEmitter record(words, scratch_vgpr, tmp_vgpr, arch);
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
        static_cast<uint16_t>(scratch_vgpr + base_scratch_count);
    if (candidate.is_direct_to_lds()) {
      if (!point.moi_exec_save_sgpr) {
        errors.emplace_back(
            "ConSan MOI sampled direct-to-LDS probe requires an EXEC-save SGPR pair");
        return std::nullopt;
      }
      if (!append_materialize_direct_to_lds_address(words, candidate, saved_lds_byte_offset_vgpr,
                                                    *point.moi_exec_save_sgpr, arch)) {
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
    const uint16_t guest_address_vgpr =
        capture_high_bank_address ? *candidate.lowering.form->address_vgpr : *lds_byte_offset_vgpr;
    const auto guest_words = build_moi_relocated_guest_access_words(
        {.image = bytes,
         .candidate = &candidate,
         .target = target,
         .replay_address_vgpr = guest_address_vgpr,
         .adjusted_address_vgpr =
             reserve_two_address_replay_scratch
                 ? std::optional<uint16_t>(static_cast<uint16_t>(scratch_vgpr + scratch_count - 1u))
                 : std::nullopt},
        errors);
    if (!guest_words)
      return std::nullopt;
    if (guest_instruction_word_count)
      *guest_instruction_word_count = static_cast<uint32_t>(guest_words->size());
    words.insert(words.end(), guest_words->begin(), guest_words->end());
    if (!append_moi_lds_wait(words, arch))
      return std::nullopt;
    if (select_guest_vgpr_bank)
      words.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
          static_cast<uint8_t>(*candidate.incoming_vgpr_bank_mode), 0u, arch));
  }

  const auto publication_state = moi_sampled_publication_state_sgprs(request, point);
  if (!publication_state) {
    errors.emplace_back("ConSan MOI sampled probe has no scalar publication-state layout");
    return std::nullopt;
  }
  const uint16_t original_exec_save_sgpr = publication_state->original_exec_save_sgpr;
  const uint16_t selection_vcc_save_sgpr = publication_state->selection_vcc_save_sgpr;
  const uint16_t publication_exec_save_sgpr = publication_state->publication_exec_save_sgpr;
  const uint16_t publication_scc_save_sgpr = publication_state->guest_scc_snapshot_sgpr;
  const auto save_scc =
      instrumentation::build_s_cselect_b32(publication_scc_save_sgpr, scalar_positive_inline_u32(1),
                                           scalar_positive_inline_u32(0), arch);
  const auto save_vcc =
      instrumentation::build_s_mov_b64(selection_vcc_save_sgpr, kAmdGpuVccLo, arch);
  const auto save_exec =
      instrumentation::build_s_mov_b64(original_exec_save_sgpr, kAmdGpuExecLo, arch);
  if (!save_scc || !save_vcc || !save_exec) {
    errors.emplace_back("ConSan MOI sampled probe could not save guest EXEC/VCC/SCC");
    return std::nullopt;
  }
  // Publication has several nested EXEC-narrowing paths. Some paths reuse the
  // publication-save pair after an earlier narrowing, so that pair is not a
  // reliable copy of the guest mask at the common exit. Keep the otherwise
  // unused leading pair as the immutable guest EXEC snapshot.
  words.push_back(*save_exec);
  words.push_back(*save_scc);
  words.push_back(*save_vcc);
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
    if (runtime_workgroup_gate_in_body) {
      if (!append_sampled_workgroup_residue(words, request, point, bound_resources,
                                            workgroup_sources, bank_vgpr, high_vgpr, low_vgpr,
                                            arch)) {
        errors.emplace_back(
            "ConSan MOI runtime sampled probe could not select a private-state workgroup");
        return std::nullopt;
      }
      const auto selected_value = instrumentation::build_v_mov_b32_literal(
          high_vgpr, request.moi_runtime_sample_offset, arch);
      const auto selected =
          instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(high_vgpr), bank_vgpr, arch);
      const auto narrow_selected =
          instrumentation::build_s_and_saveexec_b64(publication_exec_save_sgpr, kAmdGpuVccLo, arch);
      if (!selected_value || !selected || !narrow_selected) {
        errors.emplace_back(
            "ConSan MOI runtime sampled probe could not encode its body workgroup gate");
        return std::nullopt;
      }
      words.insert(words.end(), selected_value->begin(), selected_value->end());
      words.push_back(*selected);
      if (!sequence.emit_branch(restore_label, InstructionSequence::BranchKind::VccZero))
        return std::nullopt;
      words.push_back(*narrow_selected);
    }
    // Sample addresses while retaining every wave which touches the selected
    // cell. Selecting owners here would discard the cross-wave evidence that
    // the sampled checker needs in order to identify a race.
    if (spilled_lds_byte_offset_vgpr) {
      const uint16_t recovered_vgpr =
          candidate.lowering_offset(access_range) == 0u ? low_vgpr : high_vgpr;
      if (!reload_spilled_lds_byte_offset(recovered_vgpr))
        return std::nullopt;
      if (candidate.lowering_offset(access_range) != 0u &&
          !append_compute_effective_lds_byte_offset(
              words, low_vgpr, recovered_vgpr, candidate.lowering_offset(access_range), arch)) {
        return std::nullopt;
      }
    } else if (candidate.lowering_offset(access_range) != 0u) {
      if (!append_compute_effective_lds_byte_offset(words, low_vgpr, *lds_byte_offset_vgpr,
                                                    candidate.lowering_offset(access_range), arch))
        return std::nullopt;
    } else {
      words.push_back(
          build_v_mov_b32_e32(low_vgpr, vector_source_vgpr(*lds_byte_offset_vgpr), arch));
    }
    const auto cell = instrumentation::build_v_lshrrev_b32(
        low_vgpr, scalar_positive_inline_u32(consan_moi_exact_shadow::granule_shift), low_vgpr,
        arch);
    const auto residue = instrumentation::build_v_and_b32_literal(
        low_vgpr, request.moi_runtime_sample_stride - 1u, low_vgpr, arch);
    const auto selected_value = instrumentation::build_v_mov_b32_literal(
        high_vgpr, request.moi_runtime_sample_offset, arch);
    const auto selected =
        instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(high_vgpr), low_vgpr, arch);
    const auto narrow_selected =
        instrumentation::build_s_and_saveexec_b64(publication_exec_save_sgpr, kAmdGpuVccLo, arch);
    if (!cell || !residue || !selected_value || !selected || !narrow_selected) {
      errors.emplace_back("ConSan MOI runtime sampled probe could not encode LDS-cell selector");
      return std::nullopt;
    }
    words.push_back(*cell);
    words.insert(words.end(), residue->begin(), residue->end());
    words.insert(words.end(), selected_value->begin(), selected_value->end());
    words.push_back(*selected);
    if (!sequence.emit_branch(restore_label, InstructionSequence::BranchKind::VccZero))
      return std::nullopt;
    words.push_back(*narrow_selected);
  }
  if (runtime_sampled || window_bank_count > 1) {
    if (!append_sampled_window_bank_index(words, point, bound_resources, workgroup_sources,
                                          window_bank_count, bank_vgpr, high_vgpr, owner_vgpr,
                                          arch)) {
      errors.emplace_back("ConSan MOI sampled probe could not select a window bank");
      return std::nullopt;
    }
  } else {
    const auto zero_bank = instrumentation::build_v_mov_b32_literal(bank_vgpr, 0, arch);
    if (!zero_bank)
      return std::nullopt;
    words.insert(words.end(), zero_bank->begin(), zero_bank->end());
  }
  if (!append_moi_delay_words(words, arch, request, errors, "ConSan MOI sampled probe"))
    return std::nullopt;

  // A ready deferred acquire deterministically selects the wave which may
  // publish its associated later read. Empty slots are ordinary reads. Odd,
  // malformed, or other-wave slots take the existing fail-closed drop path,
  // preventing independent first-publisher races from joining unrelated
  // atomic and LDS evidence.
  if (kind == ConSanMoiShadowAccessKind::Read) {
    const auto pending_gate_done_label = sequence.make_label();
    if (pending_acquire_owner_bank_count > 1u) {
      const auto scale_window = instrumentation::build_v_mul_lo_u32_literal(
          low_vgpr, high_vgpr, pending_acquire_owner_bank_count, bank_vgpr, arch);
      const auto owner_bank = instrumentation::build_v_and_b32_literal(
          high_vgpr, pending_acquire_owner_bank_count - 1u, owner_vgpr, arch);
      const auto combine =
          instrumentation::build_v_add_u32(low_vgpr, vector_source_vgpr(low_vgpr), high_vgpr, arch);
      if (!scale_window || !owner_bank || !combine)
        return std::nullopt;
      words.insert(words.end(), scale_window->begin(), scale_window->end());
      words.insert(words.end(), owner_bank->begin(), owner_bank->end());
      words.insert(words.end(), combine->begin(), combine->end());
      if (!consan_detail::append_moi_indexed_address(
              words,
              {.table_address = pending_acquire_address,
               .stride_bytes = sizeof(ConSanMoiSampledPendingAcquireSlot),
               .address_vgpr = scratch_vgpr,
               .index_vgpr = low_vgpr},
              *target))
        return std::nullopt;
    } else if (!append_sampled_banked_address(words, pending_acquire_address,
                                              sizeof(ConSanMoiSampledPendingAcquireSlot),
                                              window_bank_count, bank_vgpr, scratch_vgpr, arch)) {
      return std::nullopt;
    }
    if (!append_load_u32_vgpr_at_offset(words, scratch_vgpr,
                                        offsetof(ConSanMoiSampledPendingAcquireSlot, version),
                                        low_vgpr, arch))
      return std::nullopt;
    const auto empty =
        instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), low_vgpr, arch);
    const auto enter_gate = instrumentation::build_s_cbranch_vccz(1, arch);
    if (!empty || !enter_gate)
      return std::nullopt;
    words.push_back(*empty);
    words.push_back(*enter_gate);
    if (!sequence.emit_branch(pending_gate_done_label,
                              InstructionSequence::BranchKind::Unconditional))
      return std::nullopt;

    const auto ready =
        instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(2), low_vgpr, arch);
    if (!ready)
      return std::nullopt;
    words.push_back(*ready);
    if (!sequence.emit_branch(different_identity_label, InstructionSequence::BranchKind::VccZero))
      return std::nullopt;
    if (!append_load_u32_vgpr_at_offset(words, scratch_vgpr,
                                        offsetof(ConSanMoiSampledPendingAcquireSlot, owner_id),
                                        high_vgpr, arch))
      return std::nullopt;
    const auto owner_equal =
        instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(owner_vgpr), high_vgpr, arch);
    const auto narrow_owner =
        instrumentation::build_s_and_saveexec_b64(publication_exec_save_sgpr, kAmdGpuVccLo, arch);
    if (!owner_equal || !narrow_owner)
      return std::nullopt;
    words.push_back(*owner_equal);
    if (!sequence.emit_branch(different_identity_label, InstructionSequence::BranchKind::VccZero))
      return std::nullopt;
    words.push_back(*narrow_owner);
    if (!sequence.bind(pending_gate_done_label))
      return std::nullopt;
  }

  const uint64_t causal_window_address =
      *bound_resources.moi_report_buffer_address + sampled_causal_windows_offset +
      static_cast<uint64_t>(record_index) * sizeof(ConSanMoiSampledCausalWindow);
  const auto claim_new = instrumentation::build_v_mov_b32_literal(
      low_vgpr, static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Publishing), arch);
  const auto claim_compare = instrumentation::build_v_mov_b32_literal(
      high_vgpr, static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Empty), arch);
  const auto claim = instrumentation::build_flat_atomic_cmpswap_b32(
      scratch_vgpr, low_vgpr, low_vgpr, /*return_old_value=*/true, kAmdGpuScopeDevice, arch);
  const auto claimed =
      instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(static_cast<uint32_t>(
                                                  ConSanMoiSampledCausalPublicationState::Empty)),
                                              low_vgpr, arch);
  const auto narrow_winner =
      instrumentation::build_s_and_saveexec_b64(publication_exec_save_sgpr, kAmdGpuVccLo, arch);
  if (!claim_new || !claim_compare || !claim || !claimed || !narrow_winner ||
      !append_sampled_banked_address(
          words, causal_window_address + offsetof(ConSanMoiSampledCausalWindow, publication_state),
          sizeof(ConSanMoiSampledCausalWindow), window_bank_count, bank_vgpr, scratch_vgpr, arch)) {
    errors.emplace_back("ConSan MOI sampled probe could not claim a causal window slot");
    return std::nullopt;
  }
  words.insert(words.end(), claim_new->begin(), claim_new->end());
  words.insert(words.end(), claim_compare->begin(), claim_compare->end());
  words.insert(words.end(), claim->begin(), claim->end());
  if (!append_moi_global_atomic_wait(words, arch))
    return std::nullopt;
  words.push_back(*claimed);
  if (!sequence.emit_branch(collision_label, InstructionSequence::BranchKind::VccZero))
    return std::nullopt;
  words.push_back(*narrow_winner);

  if (!append_sampled_banked_address(words, causal_window_address,
                                     sizeof(ConSanMoiSampledCausalWindow), window_bank_count,
                                     bank_vgpr, scratch_vgpr, arch))
    return std::nullopt;
  const auto store_dynamic_record_index = [&]() {
    if (window_bank_count == 1u) {
      return record.store_literal(offsetof(ConSanMoiSampledCausalWindow, first_entry),
                                  record_index);
    }
    const auto dynamic_record_index =
        instrumentation::build_v_add_u32_literal(high_vgpr, record_index, bank_vgpr, arch);
    if (!dynamic_record_index)
      return false;
    words.insert(words.end(), dynamic_record_index->begin(), dynamic_record_index->end());
    return record.store_vgpr(offsetof(ConSanMoiSampledCausalWindow, first_entry), high_vgpr);
  };
  if (!record.store_literal(offsetof(ConSanMoiSampledCausalWindow, generation),
                            static_cast<uint32_t>(bound_resources.moi_report_generation)) ||
      !record.store_literal(offsetof(ConSanMoiSampledCausalWindow, generation) + 4u,
                            static_cast<uint32_t>(bound_resources.moi_report_generation >> 32u)) ||
      !append_store_moi_report_dispatch_id_pair(
          record, consan_moi_detail::moi_bound_dispatch_id_sources({point, bound_resources}),
          offsetof(ConSanMoiSampledCausalWindow, dispatch_id)) ||
      !record.store_workgroup(offsetof(ConSanMoiSampledCausalWindow, workgroup_x),
                              workgroup_sources.x,
                              ConSanMoiRecordEmitter::MissingWorkgroupSource::StoreZero) ||
      !record.store_workgroup(offsetof(ConSanMoiSampledCausalWindow, workgroup_y),
                              workgroup_sources.y,
                              ConSanMoiRecordEmitter::MissingWorkgroupSource::StoreZero) ||
      !record.store_workgroup(offsetof(ConSanMoiSampledCausalWindow, workgroup_z),
                              workgroup_sources.z,
                              ConSanMoiRecordEmitter::MissingWorkgroupSource::StoreZero) ||
      !(owner_epoch_vgprs.epoch
            ? record.store_vgpr(offsetof(ConSanMoiSampledCausalWindow, epoch),
                                *owner_epoch_vgprs.epoch)
            : record.store_literal(offsetof(ConSanMoiSampledCausalWindow, epoch), 0)) ||
      !store_dynamic_record_index() ||
      !record.store_literal(offsetof(ConSanMoiSampledCausalWindow, entry_count), 1) ||
      !record.store_workgroup(offsetof(ConSanMoiSampledCausalWindow, cluster_workgroup_id),
                              workgroup_sources.cluster_workgroup_id)) {
    errors.emplace_back("ConSan MOI sampled probe could not publish causal window metadata");
    return std::nullopt;
  }

  const auto mov_low = instrumentation::build_v_mov_b32_literal(low_vgpr, low_literal, arch);
  if (!mov_low) {
    errors.emplace_back("ConSan MOI sampled probe could not encode sampled entry low word");
    return std::nullopt;
  }
  words.insert(words.end(), mov_low->begin(), mov_low->end());
  if ((owner_epoch_vgprs.owner || derived_owner_vgpr) &&
      !append_add_shifted_vgpr_field(words, low_vgpr, owner_vgpr,
                                     consan_moi_sampled_watchpoint::owner_shift,
                                     consan_moi_sampled_watchpoint::max_owner, tmp_vgpr, arch)) {
    errors.emplace_back("ConSan MOI sampled probe could not encode owner field");
    return std::nullopt;
  }
  if (owner_epoch_vgprs.epoch &&
      !append_add_shifted_vgpr_field(words, low_vgpr, *owner_epoch_vgprs.epoch,
                                     consan_moi_sampled_watchpoint::epoch_shift,
                                     consan_moi_sampled_watchpoint::max_epoch, tmp_vgpr, arch)) {
    errors.emplace_back("ConSan MOI sampled probe could not encode epoch field");
    return std::nullopt;
  }
  uint16_t effective_lds_byte_offset_vgpr = *lds_byte_offset_vgpr;
  if (spilled_lds_byte_offset_vgpr) {
    const uint16_t recovered_vgpr =
        candidate.lowering_offset(access_range) == 0u ? tmp_vgpr : high_vgpr;
    if (!reload_spilled_lds_byte_offset(recovered_vgpr)) {
      return std::nullopt;
    }
    if (candidate.lowering_offset(access_range) != 0u &&
        !append_compute_effective_lds_byte_offset(words, tmp_vgpr, recovered_vgpr,
                                                  candidate.lowering_offset(access_range), arch)) {
      errors.emplace_back("ConSan MOI sampled probe could not encode effective LDS byte offset");
      return std::nullopt;
    }
    effective_lds_byte_offset_vgpr = tmp_vgpr;
  } else if (candidate.lowering_offset(access_range) != 0u) {
    if (!append_compute_effective_lds_byte_offset(words, tmp_vgpr, *lds_byte_offset_vgpr,
                                                  candidate.lowering_offset(access_range), arch)) {
      errors.emplace_back("ConSan MOI sampled probe could not encode effective LDS byte offset");
      return std::nullopt;
    }
    effective_lds_byte_offset_vgpr = tmp_vgpr;
  }
  const auto high_from_start = instrumentation::build_v_lshlrev_b32(
      high_vgpr, scalar_positive_inline_u32(consan_moi_sampled_watchpoint::start_byte_shift - 32u),
      effective_lds_byte_offset_vgpr, arch);
  if (!high_from_start) {
    errors.emplace_back("ConSan MOI sampled probe could not encode sampled entry byte offset");
    return std::nullopt;
  }
  words.push_back(*high_from_start);
  if (!append_add_literal_field(words, high_vgpr, encoded_byte_count | encoded_generation_high,
                                tmp_vgpr, arch)) {
    errors.emplace_back("ConSan MOI sampled probe could not encode byte-count field");
    return std::nullopt;
  }
  for (uint32_t prior_record_index : prior_record_indices) {
    if (!append_direct_sampled_immediate_check(
            words, request, bound_resources, scratch_vgpr, arch, prior_record_index,
            window_bank_count, bank_vgpr, sampled_watchpoints_offset, low_vgpr, high_vgpr, errors))
      return std::nullopt;
  }
  // A site is shared by all selected waves. Publish the complete packed entry
  // with one device-scope atomic exchange so concurrent writers cannot leave a
  // low word from one wave paired with a high word from another.
  const auto atomic_publish = instrumentation::build_flat_atomic_swap_b64(
      scratch_vgpr, low_vgpr, low_vgpr, /*return_old_value=*/true, kAmdGpuScopeDevice, arch);
  if (!atomic_publish ||
      !append_sampled_banked_address(words, sampled_entry_address, sizeof(uint64_t),
                                     window_bank_count, bank_vgpr, scratch_vgpr, arch)) {
    errors.emplace_back("ConSan MOI sampled probe could not atomically publish sampled entry");
    return std::nullopt;
  }
  words.insert(words.end(), atomic_publish->begin(), atomic_publish->end());
  if (!append_moi_global_atomic_wait(words, arch))
    return std::nullopt;
  if (!append_sampled_banked_address(words, causal_window_address,
                                     sizeof(ConSanMoiSampledCausalWindow), window_bank_count,
                                     bank_vgpr, scratch_vgpr, arch))
    return std::nullopt;
  if (!record.store_literal(offsetof(ConSanMoiSampledCausalWindow, publication_state),
                            static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Ready))) {
    errors.emplace_back("ConSan MOI sampled probe could not commit causal window metadata");
    return std::nullopt;
  }
  const uint64_t selected_count_address =
      *bound_resources.moi_report_buffer_address +
      offsetof(ConSanMoiReportHeader, sampled_causal_window_count);
  if (!append_atomic_fetch_add_one_u32(words, selected_count_address, tmp_vgpr, scratch_vgpr,
                                       arch)) {
    errors.emplace_back("ConSan MOI sampled probe could not count the claimed causal window");
    return std::nullopt;
  }
  const auto restore_winner_exec =
      instrumentation::build_s_mov_b64(kAmdGpuExecLo, publication_exec_save_sgpr, arch);
  if (!restore_winner_exec)
    return std::nullopt;
  words.push_back(*restore_winner_exec);
  if (!sequence.emit_branch(restore_label, InstructionSequence::BranchKind::Unconditional))
    return std::nullopt;

  const auto save_collision_exec =
      instrumentation::build_s_mov_b64(publication_exec_save_sgpr, kAmdGpuExecLo, arch);
  if (!save_collision_exec || !sequence.bind(collision_label))
    return std::nullopt;
  words.push_back(*save_collision_exec);

  // Repeated executions of the same selected causal window are idempotent:
  // the static slot already retains their representative sample. Only a
  // different exact dynamic identity occupying the slot is capacity loss.
  if (!append_sampled_banked_address(words, causal_window_address,
                                     sizeof(ConSanMoiSampledCausalWindow), window_bank_count,
                                     bank_vgpr, scratch_vgpr, arch))
    return std::nullopt;
  const auto reject_unless_equal_vgpr = [&](uint32_t offset, uint16_t expected_vgpr) -> bool {
    if (!record.load(offset, low_vgpr))
      return false;
    const auto equal =
        instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(expected_vgpr), low_vgpr, arch);
    if (!equal)
      return false;
    words.push_back(*equal);
    return sequence.emit_branch(different_identity_label, InstructionSequence::BranchKind::VccZero);
  };
  const auto reject_unless_equal_literal = [&](uint32_t offset, uint32_t literal) -> bool {
    const auto expected = instrumentation::build_v_mov_b32_literal(high_vgpr, literal, arch);
    if (!expected)
      return false;
    words.insert(words.end(), expected->begin(), expected->end());
    return reject_unless_equal_vgpr(offset, high_vgpr);
  };
  const auto reject_unless_equal_dispatch_id = [&](uint32_t offset, bool high_word) -> bool {
    if (!record.load(offset, low_vgpr) ||
        !append_compare_moi_report_dispatch_id_word(
            words, consan_moi_detail::moi_bound_dispatch_id_sources({point, bound_resources}),
            low_vgpr, high_vgpr, high_word, arch)) {
      return false;
    }
    return sequence.emit_branch(different_identity_label, InstructionSequence::BranchKind::VccZero);
  };
  const auto reject_unless_equal_workgroup = [&](uint32_t offset,
                                                 const ConSanMoiWorkgroupSource &source) -> bool {
    if (!source.scalar_src)
      return reject_unless_equal_literal(offset, 0u);
    if (!consan_detail::append_workgroup_source_value(words, source, high_vgpr, arch))
      return false;
    return reject_unless_equal_vgpr(offset, high_vgpr);
  };
  const auto reject_unless_equal_dynamic_record_index = [&]() {
    if (window_bank_count == 1u) {
      return reject_unless_equal_literal(offsetof(ConSanMoiSampledCausalWindow, first_entry),
                                         record_index);
    }
    const auto dynamic_record_index =
        instrumentation::build_v_add_u32_literal(high_vgpr, record_index, bank_vgpr, arch);
    if (!dynamic_record_index)
      return false;
    words.insert(words.end(), dynamic_record_index->begin(), dynamic_record_index->end());
    return reject_unless_equal_vgpr(offsetof(ConSanMoiSampledCausalWindow, first_entry), high_vgpr);
  };
  if (!reject_unless_equal_literal(offsetof(ConSanMoiSampledCausalWindow, generation),
                                   static_cast<uint32_t>(bound_resources.moi_report_generation)) ||
      !reject_unless_equal_literal(
          offsetof(ConSanMoiSampledCausalWindow, generation) + 4u,
          static_cast<uint32_t>(bound_resources.moi_report_generation >> 32u)) ||
      !reject_unless_equal_dispatch_id(offsetof(ConSanMoiSampledCausalWindow, dispatch_id),
                                       /*high_word=*/false) ||
      !reject_unless_equal_dispatch_id(offsetof(ConSanMoiSampledCausalWindow, dispatch_id) + 4u,
                                       /*high_word=*/true) ||
      !reject_unless_equal_workgroup(offsetof(ConSanMoiSampledCausalWindow, workgroup_x),
                                     workgroup_sources.x) ||
      !reject_unless_equal_workgroup(offsetof(ConSanMoiSampledCausalWindow, workgroup_y),
                                     workgroup_sources.y) ||
      !reject_unless_equal_workgroup(offsetof(ConSanMoiSampledCausalWindow, workgroup_z),
                                     workgroup_sources.z) ||
      !(owner_epoch_vgprs.epoch
            ? reject_unless_equal_vgpr(offsetof(ConSanMoiSampledCausalWindow, epoch),
                                       *owner_epoch_vgprs.epoch)
            : reject_unless_equal_literal(offsetof(ConSanMoiSampledCausalWindow, epoch), 0u)) ||
      !reject_unless_equal_dynamic_record_index() ||
      !reject_unless_equal_literal(offsetof(ConSanMoiSampledCausalWindow, entry_count), 1u) ||
      !reject_unless_equal_literal(
          offsetof(ConSanMoiSampledCausalWindow, publication_state),
          static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Ready)) ||
      !reject_unless_equal_workgroup(offsetof(ConSanMoiSampledCausalWindow, cluster_workgroup_id),
                                     workgroup_sources.cluster_workgroup_id)) {
    errors.emplace_back("ConSan MOI sampled probe could not compare a repeated causal identity");
    return std::nullopt;
  }
  const auto restore_duplicate_exec =
      instrumentation::build_s_mov_b64(kAmdGpuExecLo, publication_exec_save_sgpr, arch);
  if (!restore_duplicate_exec)
    return std::nullopt;
  words.push_back(*restore_duplicate_exec);
  if (!sequence.emit_branch(restore_label, InstructionSequence::BranchKind::Unconditional))
    return std::nullopt;

  if (!sequence.bind(different_identity_label))
    return std::nullopt;
  const auto mbcnt_lo = instrumentation::build_v_mbcnt_lo_u32_b32(
      tmp_vgpr, publication_exec_save_sgpr, scalar_positive_inline_u32(0), arch);
  const auto mbcnt_hi = instrumentation::build_v_mbcnt_hi_u32_b32(
      tmp_vgpr, static_cast<uint16_t>(publication_exec_save_sgpr + 1u),
      vector_source_vgpr(tmp_vgpr), arch);
  const auto first_active_lane =
      instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), tmp_vgpr, arch);
  const auto narrow_collision =
      instrumentation::build_s_and_saveexec_b64(publication_exec_save_sgpr, kAmdGpuVccLo, arch);
  if (!mbcnt_lo || !mbcnt_hi || !first_active_lane || !narrow_collision) {
    errors.emplace_back("ConSan MOI sampled probe could not select a collision representative");
    return std::nullopt;
  }
  words.insert(words.end(), mbcnt_lo->begin(), mbcnt_lo->end());
  words.insert(words.end(), mbcnt_hi->begin(), mbcnt_hi->end());
  words.push_back(*first_active_lane);
  words.push_back(*narrow_collision);
  const uint64_t saturated_count_address =
      *bound_resources.moi_report_buffer_address +
      offsetof(ConSanMoiReportHeader, sampled_saturated_window_count);
  if (!append_atomic_fetch_add_one_u32(words, saturated_count_address, tmp_vgpr, scratch_vgpr,
                                       arch)) {
    errors.emplace_back("ConSan MOI sampled probe could not count a saturated causal window claim");
    return std::nullopt;
  }
  const auto restore_collision_exec =
      instrumentation::build_s_mov_b64(kAmdGpuExecLo, publication_exec_save_sgpr, arch);
  if (!restore_collision_exec)
    return std::nullopt;
  words.push_back(*restore_collision_exec);

  if (!sequence.bind(restore_label))
    return std::nullopt;
  const auto restore_vcc =
      instrumentation::build_s_mov_b64(kAmdGpuVccLo, selection_vcc_save_sgpr, arch);
  const auto restore_exec =
      instrumentation::build_s_mov_b64(kAmdGpuExecLo, original_exec_save_sgpr, arch);
  const auto restore_scc = instrumentation::build_s_cmp_lg_u32(publication_scc_save_sgpr,
                                                               scalar_positive_inline_u32(0), arch);
  if (!restore_exec || !restore_vcc || !restore_scc) {
    errors.emplace_back("ConSan MOI sampled probe could not restore guest EXEC/VCC/SCC");
    return std::nullopt;
  }
  words.push_back(*restore_exec);
  words.push_back(*restore_vcc);
  words.push_back(*restore_scc);
  if (!sequence.resolve_branches(arch)) {
    errors.emplace_back("ConSan MOI sampled probe local branch is out of range");
    return std::nullopt;
  }
  return words;
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_direct_sampled_watchpoint_words(
    std::span<const uint8_t> bytes, const ConSanMoiCandidate &candidate,
    const ConSanRequest &request, const BoundRuntimeResources &bound_resources,
    const ConSanMoiOperatingPoint &point, const ConSanMoiOwnerEpochVgprSources &owner_epoch_vgprs,
    uint16_t scratch_vgpr, rj_code_arch_t arch, uint32_t first_record_index,
    std::optional<uint32_t> prior_first_record_index, uint32_t prior_access_range_count,
    uint32_t window_bank_count, size_t sampled_causal_windows_offset,
    size_t sampled_watchpoints_offset, size_t sampled_pending_acquires_offset,
    uint32_t pending_acquire_owner_bank_count, bool spill_overlaps_guest_operands,
    bool spill_backed_operand_recovery, const VgprSpillSequence *spill,
    std::optional<uint32_t> private_epoch_offset,
    const ConSanMoiPersistentWorkgroupPrivateOffsets *private_workgroup_offsets,
    const std::optional<MoiWorkitemOwnerDerivationPlan> &owner_derivation,
    bool runtime_workgroup_gate_in_body, std::vector<std::string> &errors,
    uint32_t *guest_instruction_offset, uint32_t *guest_instruction_word_count) {
  const auto &access_ranges = candidate.ranges;
  if (access_ranges.empty()) {
    errors.emplace_back("ConSan MOI sampled probe requires at least one LDS access range");
    return std::nullopt;
  }
  const ConSanTargetProfile *target = consan_target_profile(arch);
  if (target == nullptr) {
    errors.emplace_back("ConSan MOI sampled probe has no target profile");
    return std::nullopt;
  }
  if (!candidate.kernel_descriptor_file_offset) {
    errors.emplace_back("ConSan MOI sampled probe requires exact workgroup-id sources");
    return std::nullopt;
  }
  const auto workgroup_sources = moi_persistent_or_descriptor_workgroup_sources(
      bytes, *candidate.kernel_descriptor_file_offset, request.moi_engine, point, arch, errors,
      candidate.container.uses_cluster_workgroup_id, private_workgroup_offsets);
  if (!workgroup_sources)
    return std::nullopt;

  std::vector<uint32_t> words;
  const bool select_guest_vgpr_bank =
      target->has_selectable_vgpr_bank && candidate.incoming_vgpr_bank_mode.value_or(0u) != 0u;
  const bool capture_high_bank_address =
      moi_access_requires_high_bank_address_capture(candidate, arch);
  if (spill_backed_operand_recovery &&
      (spill == nullptr ||
       !sampled_access_supports_spill_backed_operand_recovery(request, candidate, arch))) {
    errors.emplace_back("ConSan MOI sampled probe has an invalid spill-backed operand plan");
    return std::nullopt;
  }
  if (spill_backed_operand_recovery && select_guest_vgpr_bank) {
    errors.emplace_back(
        "ConSan MOI sampled spill-backed operand recovery cannot change the guest VGPR bank");
    return std::nullopt;
  }
  // A high SRC0 bank is captured by the appended-body wrapper before it
  // selects bank zero for instrumentation scratch. Other banked operands keep
  // the older self-contained transition sequence.
  if (select_guest_vgpr_bank && !capture_high_bank_address)
    words.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
        static_cast<uint8_t>(*candidate.incoming_vgpr_bank_mode), 0u, arch));
  std::optional<uint16_t> preserved_lds_byte_offset_vgpr;
  std::optional<uint16_t> spilled_lds_byte_offset_vgpr;
  const uint16_t base_scratch_count = static_cast<uint16_t>(
      (request.moi_sampled_check ? 7u : 5u) +
      (request.moi_runtime_sample_stride > 1 || window_bank_count > 1 ? 1u : 0u));
  const uint16_t scratch_count = static_cast<uint16_t>(
      (spill_backed_operand_recovery
           ? sampled_spill_backed_scratch_count(request, point, candidate, arch)
           : direct_sampled_scratch_count(request, point, candidate, arch)));
  const bool reserve_two_address_replay_scratch =
      moi_guest_access_relocation_requires_adjusted_address(candidate, *target);
  if (capture_high_bank_address) {
    preserved_lds_byte_offset_vgpr = static_cast<uint16_t>(scratch_vgpr + base_scratch_count);
  }
  if (spill_backed_operand_recovery) {
    const auto lds_byte_offset_vgpr = candidate_lds_byte_offset_vgpr(candidate, errors);
    if (!lds_byte_offset_vgpr)
      return std::nullopt;
    const bool spilled_address =
        range_overlaps(*lds_byte_offset_vgpr, 1u, spill->vgpr_base, spill->vgpr_count);
    if (candidate.lowering.form->destination_vgpr &&
        range_overlaps(*candidate.lowering.form->destination_vgpr,
                       candidate_payload_vgpr_count(candidate), spill->vgpr_base,
                       spill->vgpr_count)) {
      errors.emplace_back(
          "ConSan MOI sampled spill-backed operand recovery overlaps the load destination");
      return std::nullopt;
    }
    if (spilled_address) {
      spilled_lds_byte_offset_vgpr = *lds_byte_offset_vgpr;
    } else if (moi_load_clobbers_address(candidate)) {
      preserved_lds_byte_offset_vgpr = static_cast<uint16_t>(scratch_vgpr + base_scratch_count);
    }
    if (moi_load_clobbers_address(candidate) && !spilled_address) {
      words.push_back(build_v_mov_b32_e32(*preserved_lds_byte_offset_vgpr,
                                          vector_source_vgpr(*lds_byte_offset_vgpr), arch));
    }
    if (guest_instruction_offset)
      *guest_instruction_offset = static_cast<uint32_t>(words.size() * sizeof(uint32_t));
    const uint16_t replay_address_vgpr =
        preserved_lds_byte_offset_vgpr.value_or(*candidate.lowering.form->address_vgpr);
    const auto guest_words = build_moi_relocated_guest_access_words(
        {.image = bytes,
         .candidate = &candidate,
         .target = target,
         .replay_address_vgpr = replay_address_vgpr,
         .adjusted_address_vgpr =
             reserve_two_address_replay_scratch
                 ? std::optional<uint16_t>(static_cast<uint16_t>(scratch_vgpr + scratch_count - 1u))
                 : std::nullopt},
        errors);
    if (!guest_words)
      return std::nullopt;
    if (guest_instruction_word_count)
      *guest_instruction_word_count = static_cast<uint32_t>(guest_words->size());
    words.insert(words.end(), guest_words->begin(), guest_words->end());
    if (!append_moi_lds_wait(words, arch)) {
      errors.emplace_back("ConSan MOI sampled spill-backed operand recovery needs an LDS wait");
      return std::nullopt;
    }
  }
  if (private_epoch_offset) {
    if (!point.automatic_moi_private_epoch || !owner_epoch_vgprs.epoch) {
      errors.emplace_back("ConSan MOI sampled probe has an invalid private epoch plan");
      return std::nullopt;
    }
    const auto load = instrumentation::build_private_load_b32(*owner_epoch_vgprs.epoch,
                                                              *private_epoch_offset, arch);
    const auto wait = instrumentation::build_s_wait_private_load0(arch);
    if (!load || !wait) {
      errors.emplace_back("ConSan MOI sampled probe could not load private epoch state");
      return std::nullopt;
    }
    words.insert(words.end(), load->begin(), load->end());
    words.push_back(*wait);
  }
  if (owner_derivation) {
    if (!point.automatic_moi_private_epoch || !owner_epoch_vgprs.owner ||
        !owner_derivation->entry_workitem_x_private_offset) {
      errors.emplace_back("ConSan MOI sampled probe has an invalid private owner plan");
      return std::nullopt;
    }
    const auto owner = build_moi_workitem_owner_derivation(
        *owner_derivation, *owner_epoch_vgprs.owner, arch, "sampled probe", errors);
    if (!owner)
      return std::nullopt;
    words.insert(words.end(), owner->words.begin(), owner->words.end());
  }
  if (point.moi_persistent_sgprs.complete()) {
    if (!consan_detail::validate_scalar_state_temporaries(point, owner_epoch_vgprs, "sampled probe",
                                                          errors))
      return std::nullopt;
    words.push_back(
        build_v_mov_b32_e32(*owner_epoch_vgprs.owner, *point.moi_persistent_sgprs.owner(), arch));
    words.push_back(
        build_v_mov_b32_e32(*owner_epoch_vgprs.epoch, *point.moi_persistent_sgprs.epoch(), arch));
  }
  const bool materialize_flat_address = candidate_requires_flat_address_materialization(candidate);
  if (!preserved_lds_byte_offset_vgpr &&
      (moi_load_clobbers_address(candidate) || materialize_flat_address)) {
    preserved_lds_byte_offset_vgpr = static_cast<uint16_t>(scratch_vgpr + base_scratch_count);
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
    if (prior_first_record_index) {
      prior_record_indices.reserve(prior_access_range_count);
      for (uint32_t prior_range_index = 0; prior_range_index < prior_access_range_count;
           ++prior_range_index) {
        prior_record_indices.push_back(*prior_first_record_index +
                                       prior_range_index * window_bank_count);
      }
    }
    uint32_t range_guest_instruction_offset = 0;
    uint32_t range_guest_instruction_word_count = 0;
    auto range_words = build_direct_sampled_watchpoint_range_words(
        bytes, candidate, access_ranges[range_index], request, bound_resources, point,
        *workgroup_sources, owner_epoch_vgprs, scratch_vgpr, arch,
        first_record_index + range_index * window_bank_count, prior_record_indices,
        window_bank_count, spill_overlaps_guest_operands, spill_backed_operand_recovery,
        range_index == 0u && !spill_backed_operand_recovery, pending_acquire_owner_bank_count,
        preserved_lds_byte_offset_vgpr, spill, spilled_lds_byte_offset_vgpr,
        sampled_causal_windows_offset, sampled_watchpoints_offset, sampled_pending_acquires_offset,
        runtime_workgroup_gate_in_body, errors,
        range_index == 0u ? &range_guest_instruction_offset : nullptr,
        range_index == 0u ? &range_guest_instruction_word_count : nullptr);
    if (!range_words)
      return std::nullopt;
    if (!spill_backed_operand_recovery && range_index == 0u && guest_instruction_offset) {
      *guest_instruction_offset =
          static_cast<uint32_t>(words.size() * sizeof(uint32_t)) + range_guest_instruction_offset;
    }
    if (!spill_backed_operand_recovery && range_index == 0u && guest_instruction_word_count)
      *guest_instruction_word_count = range_guest_instruction_word_count;
    words.insert(words.end(), range_words->begin(), range_words->end());
  }
  if (select_guest_vgpr_bank && !capture_high_bank_address)
    words.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
        0u, static_cast<uint8_t>(*candidate.incoming_vgpr_bank_mode), arch));
  return words;
}

} // namespace rocjitsu::consan_moi_impl
