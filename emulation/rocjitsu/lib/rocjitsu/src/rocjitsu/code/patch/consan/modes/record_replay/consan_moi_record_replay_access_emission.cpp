// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Record/Replay's access-record protocol. Shared target, address, relocation,
// and instruction mechanisms enter only through their published contracts.

#include "rocjitsu/code/patch/consan/modes/record_replay/consan_moi_record_replay.h"

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/consan/consan_moi_access_target.h"
#include "rocjitsu/code/patch/consan/consan_moi_dynamic_record_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_engine_contracts.h"
#include "rocjitsu/code/patch/consan/consan_moi_memory_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_mode_planning.h"
#include "rocjitsu/code/patch/consan/consan_moi_native_abi.h"
#include "rocjitsu/code/patch/consan/consan_moi_record_event_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_relocation.h"
#include "rocjitsu/code/patch/consan/consan_moi_report_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_sync_emission.h"
#include "rocjitsu/code/patch/consan/consan_resource.h"
#include "rocjitsu/code/patch/instruction_sequence.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu {

using consan_detail::append_moi_relocated_guest_access;
using consan_detail::has_recent_saveexec;
using consan_detail::moi_guest_access_relocation_requires_adjusted_address;
using consan_detail::MoiWorkgroupKeyRegisterPlan;
using consan_detail::MoiWorkitemOwnerDerivationPlan;
using consan_detail::reject_optional_scratch_range_overlap;
using consan_moi_detail::append_atomic_fetch_add_one_u32;
using consan_moi_detail::append_atomic_or_u32_literal;
using consan_moi_detail::append_reserve_bounded_dynamic_record_slot;
using consan_moi_detail::append_select_first_lane_in_exec_mask;
using consan_moi_detail::append_store_moi_report_dispatch_id_pair;
using consan_moi_detail::append_store_u32_literal;
using consan_moi_detail::ConSanMoiRecordEmitter;
using consan_moi_detail::DynamicRecordEmitter;
using consan_moi_detail::kAccessRecordLayout;
using consan_moi_detail::moi_has_runtime_hardware_dispatch_id;
using consan_moi_detail::record_replay_uses_automatic_banked_capture;

namespace consan_moi_impl {

[[nodiscard]] std::optional<std::vector<uint32_t>> build_first_light_access_record_words(
    std::span<const uint8_t> bytes, const ConSanMoiCandidate &candidate,
    const ConSanRequest &request, const BoundRuntimeResources &bound_resources,
    const ConSanMoiOperatingPoint &point, uint16_t scratch_vgpr, rj_code_arch_t arch,
    uint32_t record_index, uint32_t record_count, uint32_t logical_range_index,
    const ConSanMoiReportBufferLayout &layout, bool spill_overlaps_guest_operands,
    const VgprSpillSequence *spill, const ConSanMoiPrivateStateLayout *private_layout,
    const std::optional<MoiWorkitemOwnerDerivationPlan> &owner_derivation,
    std::vector<std::string> &errors, uint32_t *guest_instruction_offset,
    uint32_t *guest_instruction_word_count) {
  const std::optional<uint32_t> private_epoch_offset =
      private_layout ? std::optional{private_layout->epoch_offset} : std::nullopt;
  const std::optional<uint32_t> private_dispatch_id_offset =
      private_layout ? private_layout->dispatch_id_offset : std::nullopt;
  const ConSanMoiPersistentWorkgroupPrivateOffsets *private_workgroup_offsets =
      private_layout ? &private_layout->exact_workgroup_offsets : nullptr;
  const auto fail = [&](const char *message) -> std::optional<std::vector<uint32_t>> {
    errors.emplace_back(message);
    return std::nullopt;
  };
  const ConSanTargetProfile *target = consan_target_profile(arch);
  if (target == nullptr)
    return fail("ConSan MOI first-light probe has no target profile");
  const MoiScalarAbiPlan scalar_abi =
      plan_moi_scalar_abi(request.moi_engine, moi_scalar_preservation_state(point));
  const bool automatic_banked_capture = record_replay_uses_automatic_banked_capture(
      request, layout.record_replay_dispatch_token_capacity);
  const uint16_t base_scratch_count = automatic_banked_capture ? 10u : 6u;
  const bool materialize_flat_address = candidate_requires_flat_address_materialization(candidate);
  const bool materialize_direct_to_lds = candidate.is_direct_to_lds();
  const bool load_clobbers_address = moi_load_clobbers_address(candidate);
  const bool capture_high_bank_address =
      moi_access_requires_high_bank_address_capture(candidate, arch);
  const bool save_clobbered_address = load_clobbers_address || capture_high_bank_address;
  const bool reserve_two_address_replay_scratch =
      moi_guest_access_relocation_requires_adjusted_address(candidate, *target);
  const uint16_t address_scratch_count = static_cast<uint16_t>(
      (materialize_flat_address ? flat_access_address_scratch_count(candidate)
                                : (save_clobbered_address ? 1u : 0u)) +
      (materialize_direct_to_lds ? 1u : 0u) + (reserve_two_address_replay_scratch ? 1u : 0u));
  const uint16_t scratch_count = static_cast<uint16_t>(base_scratch_count + address_scratch_count);
  if (static_cast<uint32_t>(scratch_vgpr) + scratch_count > kMaxVgprs) {
    errors.emplace_back(request.moi_dynamic_access_records
                            ? "ConSan MOI dynamic access-record probe needs six scratch VGPRs"
                        : automatic_banked_capture
                            ? "ConSan MOI automatic first-light probe needs ten scratch VGPRs"
                            : "ConSan MOI first-light probe needs six scratch VGPRs");
    return std::nullopt;
  }
  if (request.moi_dynamic_access_records && !point.moi_exec_save_sgpr) {
    errors.emplace_back(
        "ConSan MOI dynamic access-record probe requires RJ_CONSAN_MOI_EXEC_SAVE_SGPR");
    return std::nullopt;
  }
  if (request.moi_dynamic_access_records &&
      (*point.moi_exec_save_sgpr > 100u || *point.moi_exec_save_sgpr % 2u != 0u)) {
    errors.emplace_back(
        "ConSan MOI dynamic access-record probe requires an even RJ_CONSAN_MOI_EXEC_SAVE_SGPR in "
        "0..100");
    return std::nullopt;
  }
  auto lds_byte_offset_vgpr = candidate_lds_byte_offset_vgpr(candidate, errors);
  if (!lds_byte_offset_vgpr)
    return std::nullopt;
  if (!spill_overlaps_guest_operands &&
      reject_candidate_scratch_range_overlap(candidate, scratch_vgpr, scratch_count, errors))
    return std::nullopt;
  if (request.moi_dynamic_access_records && has_recent_saveexec(bytes, candidate)) {
    errors.emplace_back(
        "ConSan MOI dynamic access-record probe skipped a candidate immediately after "
        "s_*_saveexec");
    return std::nullopt;
  }
  if (reject_optional_scratch_range_overlap(point.moi_owner_epoch_vgprs.owner(), scratch_vgpr,
                                            scratch_count, "MOI owner", errors) ||
      reject_optional_scratch_range_overlap(point.moi_owner_epoch_vgprs.epoch(), scratch_vgpr,
                                            scratch_count, "MOI epoch", errors) ||
      reject_optional_scratch_range_overlap(point.moi_workgroup_key_vgpr, scratch_vgpr,
                                            scratch_count, "MOI workgroup key", errors))
    return std::nullopt;
  constexpr std::array<std::string_view, 4> workgroup_names = {
      "MOI Record/Replay workgroup x", "MOI Record/Replay workgroup y",
      "MOI Record/Replay workgroup z", "MOI Record/Replay cluster workgroup ID"};
  const std::array<std::optional<uint16_t>, 4> workgroup_registers =
      point.moi_exact_workgroup_vgprs.values();
  for (size_t index = 0; index < workgroup_registers.size(); ++index) {
    if (reject_optional_scratch_range_overlap(workgroup_registers[index], scratch_vgpr,
                                              scratch_count, workgroup_names[index], errors))
      return std::nullopt;
  }
  const auto &access_ranges = candidate.site().ranges;
  if (access_ranges.empty()) {
    errors.emplace_back("ConSan MOI first-light probe requires a supported LDS access range");
    return std::nullopt;
  }
  if (materialize_flat_address && access_ranges.size() != 1u) {
    errors.emplace_back("ConSan MOI flat-address probe requires one normalized access range");
    return std::nullopt;
  }
  if (capture_high_bank_address && load_clobbers_address && candidate.is_native_two_range()) {
    errors.emplace_back("ConSan MOI first-light probe cannot yet preserve a clobbered high-bank "
                        "CDNA5 two-address LDS operand");
    return std::nullopt;
  }
  std::optional<uint16_t> derived_owner_vgpr;
  std::vector<uint32_t> derived_owner_words;
  if (!point.moi_owner_epoch_vgprs.owner() && !point.moi_persistent_sgprs.owner()) {
    if (!owner_derivation) {
      errors.emplace_back("ConSan MOI first-light probe requires a planned owner derivation");
      return std::nullopt;
    }
    const uint16_t value_vgpr =
        static_cast<uint16_t>(scratch_vgpr + (request.moi_dynamic_access_records ? 4u : 2u));
    const auto owner = build_moi_workitem_owner_derivation(*owner_derivation, value_vgpr, arch,
                                                           "first-light probe", errors);
    if (!owner)
      return std::nullopt;
    const auto owner_mask = instrumentation::build_v_and_b32_literal(
        value_vgpr, consan_moi_exact_shadow::max_owner, value_vgpr, arch);
    if (!owner_mask) {
      errors.emplace_back("ConSan MOI first-light probe could not encode owner derivation");
      return std::nullopt;
    }
    derived_owner_vgpr = owner->vgpr;
    derived_owner_words = owner->words;
    derived_owner_words.insert(derived_owner_words.end(), owner_mask->begin(), owner_mask->end());
  }
  const auto persistent_workgroup_sources =
      moi_exact_entry_workgroup_sources(point, private_workgroup_offsets);
  if (!persistent_workgroup_sources) {
    errors.emplace_back(
        "ConSan MOI Record/Replay access requires one exact entry-captured workgroup tuple");
    return std::nullopt;
  }
  const ConSanMoiWorkgroupSources workgroup_sources = *persistent_workgroup_sources;
  const auto dispatch_id_sources = consan_moi_detail::moi_bound_dispatch_id_sources(
      {point, bound_resources, private_dispatch_id_offset});

  std::vector<uint32_t> words;
  words.reserve(candidate.size() / sizeof(uint32_t) + 1u + 7u * 12u + 9u + 10u + 20u + 24u +
                (point.moi_owner_epoch_vgprs.owner() || derived_owner_vgpr ? 9u : 0u) +
                (point.moi_owner_epoch_vgprs.epoch() ? 9u : 0u) + derived_owner_words.size());
  InstructionSequence sequence(words);
  MoiEmissionRequirement require_emission(sequence, errors);
  const auto restore_exec_label = sequence.make_label();
  // A DS load may overwrite the same VGPR that supplied its address. Preserve
  // that effective address before executing the displaced guest instruction;
  // otherwise RecordReplay publishes the loaded payload as the LDS offset.
  if (materialize_direct_to_lds) {
    if (!point.moi_exec_save_sgpr) {
      errors.emplace_back(
          "ConSan MOI first-light direct-to-LDS probe requires an EXEC-save SGPR pair");
      return std::nullopt;
    }
    const uint16_t materialized_address_vgpr =
        static_cast<uint16_t>(scratch_vgpr + base_scratch_count);
    if (!append_materialize_direct_to_lds_address(words, candidate, materialized_address_vgpr,
                                                  *point.moi_exec_save_sgpr, arch)) {
      errors.emplace_back("ConSan MOI first-light probe could not materialize a direct-to-LDS "
                          "destination");
      return std::nullopt;
    }
    lds_byte_offset_vgpr = materialized_address_vgpr;
  } else if (materialize_flat_address && request.moi_dynamic_access_records) {
    const uint16_t materialized_address_vgpr =
        static_cast<uint16_t>(scratch_vgpr + base_scratch_count);
    words.push_back(build_v_mov_b32_e32(
        materialized_address_vgpr,
        vector_source_vgpr(*candidate.site().lowering.form->address_vgpr), arch));
    if (!candidate_uses_scalar_vector_flat_address(candidate)) {
      words.push_back(build_v_mov_b32_e32(static_cast<uint16_t>(materialized_address_vgpr + 1u),
                                          vector_source_vgpr(static_cast<uint16_t>(
                                              *candidate.site().lowering.form->address_vgpr + 1u)),
                                          arch));
    }
    lds_byte_offset_vgpr = materialized_address_vgpr;
  } else if (save_clobbered_address) {
    const uint16_t saved_lds_byte_offset_vgpr =
        static_cast<uint16_t>(scratch_vgpr + base_scratch_count);
    if (!capture_high_bank_address) {
      words.push_back(build_v_mov_b32_e32(saved_lds_byte_offset_vgpr,
                                          vector_source_vgpr(*lds_byte_offset_vgpr), arch));
    }
    lds_byte_offset_vgpr = saved_lds_byte_offset_vgpr;
  }
  if (request.moi_dynamic_access_records)
    words.insert(words.end(), derived_owner_words.begin(), derived_owner_words.end());
  const auto append_guest_access = [&] {
    if (guest_instruction_offset)
      *guest_instruction_offset = static_cast<uint32_t>(words.size() * sizeof(uint32_t));
    const uint16_t guest_address_vgpr = capture_high_bank_address
                                            ? *candidate.site().lowering.form->address_vgpr
                                            : *lds_byte_offset_vgpr;
    return append_moi_relocated_guest_access(
        words, bytes, candidate, target, guest_address_vgpr,
        reserve_two_address_replay_scratch
            ? std::optional<uint16_t>(static_cast<uint16_t>(scratch_vgpr + scratch_count - 1u))
            : std::nullopt,
        errors, guest_instruction_word_count);
  };
  if (request.moi_dynamic_access_records) {
    if (!append_guest_access())
      return std::nullopt;
    if (!append_moi_lds_wait(words, arch))
      return fail("ConSan MOI dynamic access-record probe could not encode its LDS wait");
  }

  const uint64_t base = *bound_resources.moi_report_buffer_address;
  const uint32_t access_record_capacity = layout.access_record_capacity;
  const uint32_t access_dispatch_bank_count = layout.record_replay_access_dispatch_bank_count;
  const uint32_t access_owner_bank_count = layout.record_replay_access_owner_bank_count;
  const auto kind = consan_moi_shadow_kind_from_access_kind(candidate.site().kind);
  auto append_effective_range_offset = [&](const ConSanAccessRange &range,
                                           uint16_t value_vgpr) -> std::optional<uint16_t> {
    if (candidate.lowering_offset(range) == 0)
      return *lds_byte_offset_vgpr;
    if (!append_compute_effective_lds_byte_offset(words, value_vgpr, *lds_byte_offset_vgpr,
                                                  candidate.lowering_offset(range), arch)) {
      return std::nullopt;
    }
    return value_vgpr;
  };

  if (request.moi_dynamic_access_records) {
    const uint16_t slot_vgpr = static_cast<uint16_t>(scratch_vgpr + 2u);
    const uint16_t value_vgpr = static_cast<uint16_t>(scratch_vgpr + 5u);
    const uint64_t dynamic_record_base = base + layout.access_records_offset;

    for (const ConSanAccessRange &range : access_ranges) {
      require_emission(append_save_moi_special_state(words, scalar_abi.special_state, arch),
                       "ConSan MOI dynamic access-record probe could not save VCC/SCC");
      if (materialize_flat_address) {
        require_emission(
            append_materialize_flat_access_address(words, candidate, *lds_byte_offset_vgpr,
                                                   *lds_byte_offset_vgpr, arch),
            "ConSan MOI dynamic access-record probe could not materialize FLAT address");
      }

      require_emission(
          append_reserve_bounded_dynamic_record_slot(
              words, base + offsetof(ConSanMoiReportHeader, access_record_count),
              access_record_capacity, slot_vgpr, value_vgpr, scratch_vgpr, arch),
          "ConSan MOI dynamic access-record probe could not reserve a bounded record slot");
      require_emission.append(
          "ConSan MOI dynamic access-record probe could not apply its capacity guard",
          instrumentation::build_s_and_saveexec_b64(*point.moi_exec_save_sgpr, kAmdGpuVccLo, arch));

      DynamicRecordEmitter record(words, kAccessRecordLayout, dynamic_record_base, slot_vgpr,
                                  scratch_vgpr, arch);
      if (private_epoch_offset) {
        require_emission.append(
            "ConSan MOI dynamic access-record probe could not load private epoch state",
            instrumentation::build_private_load_b32(value_vgpr, *private_epoch_offset, arch),
            instrumentation::build_s_wait_private_load0(arch));
        record.vgpr(offsetof(ConSanMoiAccessRecord, epoch), value_vgpr);
      } else if (point.moi_persistent_sgprs.epoch()) {
        words.push_back(build_v_mov_b32_e32(value_vgpr, *point.moi_persistent_sgprs.epoch(), arch));
        record.vgpr(offsetof(ConSanMoiAccessRecord, epoch), value_vgpr);
      }

      if (derived_owner_vgpr)
        record.vgpr(offsetof(ConSanMoiAccessRecord, wave_id), *derived_owner_vgpr);
      if (point.moi_persistent_sgprs.owner())
        record.scalar(offsetof(ConSanMoiAccessRecord, wave_id),
                      *point.moi_persistent_sgprs.owner());
      record.dispatch_id(offsetof(ConSanMoiAccessRecord, generation), dispatch_id_sources)
          .event_index(offsetof(ConSanMoiAccessRecord, event_index),
                       base + offsetof(ConSanMoiReportHeader, event_counter))
          .workgroup(offsetof(ConSanMoiAccessRecord, workgroup_x), workgroup_sources.x)
          .workgroup(offsetof(ConSanMoiAccessRecord, workgroup_y), workgroup_sources.y)
          .workgroup(offsetof(ConSanMoiAccessRecord, workgroup_z), workgroup_sources.z)
          .scalar(offsetof(ConSanMoiAccessRecord, lane_mask), *point.moi_exec_save_sgpr)
          .scalar(offsetof(ConSanMoiAccessRecord, lane_mask) + sizeof(uint32_t),
                  static_cast<uint16_t>(*point.moi_exec_save_sgpr + 1u));
      if (point.moi_owner_epoch_vgprs.owner())
        record.vgpr(offsetof(ConSanMoiAccessRecord, wave_id), point.moi_owner_epoch_vgprs->owner);
      if (point.moi_owner_epoch_vgprs.epoch())
        record.vgpr(offsetof(ConSanMoiAccessRecord, epoch), point.moi_owner_epoch_vgprs->epoch);
      record
          .literal(offsetof(ConSanMoiAccessRecord, instruction_offset),
                   static_cast<uint32_t>(candidate.anchor()))
          .literal(offsetof(ConSanMoiAccessRecord, access_kind), static_cast<uint32_t>(kind));

      const std::optional<uint16_t> effective_lds_byte_offset_vgpr =
          append_effective_range_offset(range, value_vgpr);
      if (!effective_lds_byte_offset_vgpr) {
        errors.emplace_back(
            "ConSan MOI dynamic access-record probe could not encode LDS byte offset");
        return std::nullopt;
      }
      record.vgpr(offsetof(ConSanMoiAccessRecord, lds_byte_offset),
                  *effective_lds_byte_offset_vgpr);
      require_emission.append("ConSan MOI dynamic access-record probe could not encode start cell",
                              instrumentation::build_v_lshrrev_b32(
                                  value_vgpr,
                                  scalar_positive_inline_u32(consan_moi_shadow_cell::granule_shift),
                                  *effective_lds_byte_offset_vgpr, arch));
      const ConSanMoiLdsCellRange static_range =
          consan_moi_lds_cell_range_for_bytes(candidate.lowering_offset(range), range.byte_width);
      record.vgpr(offsetof(ConSanMoiAccessRecord, start_cell), value_vgpr)
          .literal(offsetof(ConSanMoiAccessRecord, lds_byte_count), range.byte_width)
          .literal(offsetof(ConSanMoiAccessRecord, cell_count), static_range.cell_count);
      require_emission(record.finish(),
                       "ConSan MOI dynamic access-record probe could not encode record stores");
      require_emission.append(
          "ConSan MOI dynamic access-record probe could not drain stores or restore EXEC",
          instrumentation::build_s_wait_global_store0(arch),
          instrumentation::build_s_mov_b64(kAmdGpuExecLo, *point.moi_exec_save_sgpr, arch));
      // A terminal guest access can return directly to s_endpgm. Drain the
      // injected report publication while the scratch values and narrowed
      // EXEC are still intact, so wave termination cannot race those stores.
      require_emission(append_restore_moi_special_state(words, scalar_abi.special_state, arch),
                       "ConSan MOI dynamic access-record probe could not restore VCC/SCC");
    }
    if (!sequence.finish())
      return std::nullopt;
    return words;
  }

  const uint16_t record_address_vgpr = scratch_vgpr;
  const uint16_t record_value_vgpr = static_cast<uint16_t>(record_address_vgpr + 2u);
  const uint16_t record_value_high_vgpr = static_cast<uint16_t>(record_address_vgpr + 3u);
  const uint16_t record_compare_vgpr = static_cast<uint16_t>(record_address_vgpr + 4u);
  const uint16_t record_compare_high_vgpr = static_cast<uint16_t>(record_address_vgpr + 5u);
  const uint16_t dispatch_bank_vgpr = static_cast<uint16_t>(record_address_vgpr + 6u);
  const uint16_t owner_bank_vgpr = static_cast<uint16_t>(record_address_vgpr + 7u);
  const uint16_t bank_probe_count_vgpr = static_cast<uint16_t>(record_address_vgpr + 8u);
  const uint16_t identity_claim_hash_vgpr = static_cast<uint16_t>(record_address_vgpr + 9u);
  const uint16_t original_exec_sgpr = static_cast<uint16_t>(*point.moi_exec_save_sgpr + 8u);
  const uint16_t address_key_sgpr = static_cast<uint16_t>(*point.moi_exec_save_sgpr + 10u);
  const uint16_t address_group_exec_sgpr = static_cast<uint16_t>(*point.moi_exec_save_sgpr + 12u);
  const std::optional<uint16_t> spilled_lds_byte_offset_vgpr =
      !materialize_flat_address && !save_clobbered_address && spill != nullptr &&
              spill_overlaps_guest_operands && *lds_byte_offset_vgpr >= spill->vgpr_base &&
              *lds_byte_offset_vgpr < static_cast<uint32_t>(spill->vgpr_base) + spill->vgpr_count
          ? lds_byte_offset_vgpr
          : std::nullopt;
  if (spilled_lds_byte_offset_vgpr)
    lds_byte_offset_vgpr = record_compare_high_vgpr;
  ConSanMoiRecordEmitter record(words, record_address_vgpr, record_value_vgpr, arch);
  const uint64_t access_bank_count =
      static_cast<uint64_t>(access_dispatch_bank_count) * access_owner_bank_count;
  if (access_dispatch_bank_count == 0u ||
      (access_dispatch_bank_count & (access_dispatch_bank_count - 1u)) != 0u ||
      access_owner_bank_count == 0u ||
      (access_owner_bank_count & (access_owner_bank_count - 1u)) != 0u ||
      access_bank_count > std::numeric_limits<uint32_t>::max() ||
      (automatic_banked_capture &&
       (layout.record_replay_logical_access_range_count == 0u ||
        static_cast<uint64_t>(layout.record_replay_logical_access_range_count) * access_bank_count *
                layout.record_replay_address_group_headroom >
            access_record_capacity ||
        layout.record_replay_address_group_headroom == 0u ||
        layout.record_replay_address_group_headroom >
            kConSanMoiRecordReplayMaximumAddressGroupsPerWave ||
        (layout.record_replay_address_group_headroom &
         (layout.record_replay_address_group_headroom - 1u)) != 0u ||
        (access_record_capacity & (access_record_capacity - 1u)) != 0u ||
        layout.record_replay_dispatch_token_capacity >
            kConSanMoiRecordReplayMaximumDispatchTokenCount ||
        (layout.record_replay_dispatch_token_capacity &
         (layout.record_replay_dispatch_token_capacity - 1u)) != 0u)) ||
      (!automatic_banked_capture &&
       (access_dispatch_bank_count != 1u || access_owner_bank_count != 1u))) {
    errors.emplace_back("ConSan MOI first-light probe has an invalid identity-table layout");
    return std::nullopt;
  }
  const auto append_claim_token = [&](uint16_t low_vgpr, uint16_t high_vgpr,
                                      uint16_t temporary_vgpr) {
    if (!append_moi_report_dispatch_id_pair(words, dispatch_id_sources, low_vgpr, high_vgpr, arch))
      return false;
    const auto low_xor_literal = instrumentation::build_v_mov_b32_literal(
        temporary_vgpr, static_cast<uint32_t>(kConSanMoiRecordReplayClaimTokenXorMask), arch);
    const auto low_xor = instrumentation::build_v_xor_b32(
        low_vgpr, vector_source_vgpr(temporary_vgpr), low_vgpr, arch);
    const auto high_xor_literal = instrumentation::build_v_mov_b32_literal(
        temporary_vgpr, static_cast<uint32_t>(kConSanMoiRecordReplayClaimTokenXorMask >> 32u),
        arch);
    const auto high_xor = instrumentation::build_v_xor_b32(
        high_vgpr, vector_source_vgpr(temporary_vgpr), high_vgpr, arch);
    InstructionSequence append(words);
    return append.emit_all(low_xor_literal, low_xor, high_xor_literal, high_xor);
  };
  const auto append_owner_id = [&](uint16_t destination_vgpr) {
    if (point.moi_owner_epoch_vgprs.owner()) {
      words.push_back(build_v_mov_b32_e32(
          destination_vgpr, vector_source_vgpr(point.moi_owner_epoch_vgprs->owner), arch));
      return true;
    }
    if (point.moi_persistent_sgprs.owner()) {
      words.push_back(
          build_v_mov_b32_e32(destination_vgpr, *point.moi_persistent_sgprs.owner(), arch));
      return true;
    }
    if (!derived_owner_vgpr)
      return false;
    words.insert(words.end(), derived_owner_words.begin(), derived_owner_words.end());
    if (*derived_owner_vgpr != destination_vgpr) {
      words.push_back(
          build_v_mov_b32_e32(destination_vgpr, vector_source_vgpr(*derived_owner_vgpr), arch));
    }
    return true;
  };
  const auto append_access_claim_token = [&](uint16_t low_vgpr, uint16_t high_vgpr,
                                             uint16_t temporary_vgpr, uint32_t site_token) {
    if (!append_claim_token(low_vgpr, high_vgpr, temporary_vgpr))
      return false;
    return sequence.emit_all(
        instrumentation::build_v_xor_b32(low_vgpr, vector_source_vgpr(low_vgpr),
                                         identity_claim_hash_vgpr, arch),
        instrumentation::build_v_mov_b32_literal(temporary_vgpr, site_token, arch),
        instrumentation::build_v_xor_b32(high_vgpr, vector_source_vgpr(high_vgpr), temporary_vgpr,
                                         arch));
  };
  const auto materialize_banked_record_address = [&](uint64_t first_address) {
    if (!automatic_banked_capture)
      return record.materialize_address(first_address);
    return consan_detail::append_moi_indexed_address(words,
                                                     {.table_address = first_address,
                                                      .stride_bytes = sizeof(ConSanMoiAccessRecord),
                                                      .address_vgpr = record_address_vgpr,
                                                      .index_vgpr = owner_bank_vgpr},
                                                     *target);
  };
  const auto reload_spilled_lds_byte_offset = [&] {
    if (!spilled_lds_byte_offset_vgpr)
      return true;
    if (spill == nullptr) {
      errors.emplace_back("ConSan MOI first-light probe has no spill plan for its LDS address");
      return false;
    }
    const auto reload = consan_detail::append_reload_moi_spilled_vgpr(
        words, *spill, record_compare_high_vgpr, *spilled_lds_byte_offset_vgpr, arch);
    if (reload == consan_detail::MoiSpilledVgprReloadResult::Appended)
      return true;
    errors.emplace_back("ConSan MOI first-light probe could not recover its spilled LDS address: " +
                        std::string(consan_detail::moi_spilled_vgpr_reload_result_name(reload)));
    return false;
  };
  if (!point.moi_exec_save_sgpr) {
    errors.emplace_back("ConSan MOI first-light probe could not save EXEC/VCC/SCC");
    return std::nullopt;
  }
  require_emission(append_save_moi_special_state(words, scalar_abi.special_state, arch),
                   "ConSan MOI first-light probe could not save EXEC/VCC/SCC");
  if (materialize_flat_address) {
    const uint16_t materialized_address_vgpr =
        static_cast<uint16_t>(scratch_vgpr + base_scratch_count);
    require_emission(append_materialize_flat_access_address(words, candidate, *lds_byte_offset_vgpr,
                                                            materialized_address_vgpr, arch),
                     "ConSan MOI first-light probe could not materialize FLAT address");
    lds_byte_offset_vgpr = materialized_address_vgpr;
  }
  std::optional<InstructionSequence::Label> address_group_loop;
  if (automatic_banked_capture) {
    require_emission.append(
        "ConSan MOI first-light probe could not retain its incoming EXEC",
        instrumentation::build_s_mov_b64(original_exec_sgpr, kAmdGpuExecLo, arch));

    // Saturation is a monotonic report-wide latch. Read it once before the
    // distinct-address loop; a group that saturates during this probe still
    // exits through the in-loop saturation paths below.
    require_emission(record.materialize_address(base) &&
                         record.load(offsetof(ConSanMoiReportHeader, flags), record_value_vgpr),
                     "ConSan MOI first-light probe could not read report saturation");
    require_emission.append("ConSan MOI first-light probe could not test report saturation",
                            instrumentation::build_s_wait_global_load0(arch),
                            instrumentation::build_v_and_b32_literal(
                                record_compare_vgpr, kConSanMoiReportFlagRecordReplayBankSaturated,
                                record_value_vgpr, arch),
                            instrumentation::build_v_cmp_ne_u32_vcc(scalar_positive_inline_u32(0),
                                                                    record_compare_vgpr, arch));
    require_emission(
        sequence.emit_branch(restore_exec_label, InstructionSequence::BranchKind::VccNonzero),
        "ConSan MOI first-light probe could not branch around a saturated report");

    address_group_loop = sequence.mark_label();
    require_emission(reload_spilled_lds_byte_offset(),
                     "ConSan MOI first-light probe could not recover its address-group source");
    require_emission(
        sequence.emit_all(
            instrumentation::build_v_readfirstlane_b32(address_key_sgpr, *lds_byte_offset_vgpr,
                                                       arch),
            instrumentation::build_v_cmp_eq_u32_vcc(address_key_sgpr, *lds_byte_offset_vgpr, arch),
            instrumentation::build_s_and_saveexec_b64(*point.moi_exec_save_sgpr, kAmdGpuVccLo,
                                                      arch),
            instrumentation::build_s_mov_b64(address_group_exec_sgpr, kAmdGpuExecLo, arch)) &&
            // Keep the full group mask for replay provenance while one elected lane
            // performs the bounded table transaction.
            append_select_first_lane_in_exec_mask(words, record_value_vgpr, address_group_exec_sgpr,
                                                  address_group_exec_sgpr, arch),
        "ConSan MOI first-light probe could not select one exact LDS address-group owner");
  } else {
    require_emission(sequence.emit(instrumentation::build_s_mov_b64(*point.moi_exec_save_sgpr,
                                                                    kAmdGpuExecLo, arch)) &&
                         append_select_first_lane_in_exec_mask(words, record_value_vgpr,
                                                               *point.moi_exec_save_sgpr,
                                                               *point.moi_exec_save_sgpr, arch),
                     "ConSan MOI first-light probe could not elect a representative lane");
  }
  if (automatic_banked_capture) {
    // The bounded bank hashes the complete persistent tuple. Descriptor SGPRs
    // and launch TTMPs are entry inputs, not probe-lifetime state, so falling
    // back to either here would recreate cross-workgroup LDS aliasing.
    if (!workgroup_sources.x.has_value() || !workgroup_sources.y.has_value() ||
        !workgroup_sources.z.has_value()) {
      errors.emplace_back(
          "ConSan MOI automatic first-light probe requires a persistent exact workgroup tuple");
      return std::nullopt;
    }

    const auto dispatch_probe_label = sequence.make_label();
    const auto dispatch_occupied_label = sequence.make_label();
    const auto dispatch_retry_label = sequence.make_label();
    const auto dispatch_bank_ready_label = sequence.make_label();
    const auto dispatch_saturated_label = sequence.make_label();
    const auto dispatch_token_ready_label = sequence.make_label();

    require_emission(
        append_claim_token(record_value_vgpr, record_value_high_vgpr, record_compare_vgpr),
        "ConSan MOI first-light probe could not form its dispatch claim");
    require_emission.append("ConSan MOI first-light probe could not hash its dispatch claim",
                            instrumentation::build_v_cmp_ne_u32_vcc(scalar_positive_inline_u32(0),
                                                                    record_value_vgpr, arch));
    require_emission(sequence.emit_branch(dispatch_token_ready_label,
                                          InstructionSequence::BranchKind::VccNonzero),
                     "ConSan MOI first-light probe could not validate its dispatch token");
    require_emission(sequence.emit(instrumentation::build_v_cmp_ne_u32_vcc(
                         scalar_positive_inline_u32(0), record_value_high_vgpr, arch)) &&
                         sequence.emit_branch(dispatch_token_ready_label,
                                              InstructionSequence::BranchKind::VccNonzero) &&
                         sequence.emit_branch(dispatch_saturated_label,
                                              InstructionSequence::BranchKind::Unconditional) &&
                         sequence.bind(dispatch_token_ready_label),
                     "ConSan MOI first-light dispatch-token path is invalid");
    require_emission.append(
        "ConSan MOI first-light probe could not hash its dispatch claim",
        instrumentation::build_v_xor_b32(dispatch_bank_vgpr, vector_source_vgpr(record_value_vgpr),
                                         record_value_high_vgpr, arch),
        instrumentation::build_v_and_b32_literal(dispatch_bank_vgpr,
                                                 layout.record_replay_dispatch_token_capacity - 1u,
                                                 dispatch_bank_vgpr, arch),
        build_v_mov_b32_e32(bank_probe_count_vgpr, scalar_positive_inline_u32(0), arch));

    require_emission(sequence.bind(dispatch_probe_label) &&
                         consan_detail::append_moi_indexed_address(
                             words,
                             {.table_address = base + layout.record_replay_dispatch_tokens_offset,
                              .stride_bytes = sizeof(uint64_t),
                              .address_vgpr = record_address_vgpr,
                              .index_vgpr = dispatch_bank_vgpr},
                             *target),
                     "ConSan MOI first-light probe could not address its dispatch table");
    require_emission.append(
        "ConSan MOI first-light probe could not claim a dispatch slot",
        build_v_mov_b32_e32(record_compare_vgpr, scalar_positive_inline_u32(0), arch),
        build_v_mov_b32_e32(record_compare_high_vgpr, scalar_positive_inline_u32(0), arch),
        instrumentation::build_flat_atomic_cmpswap_b64(
            record_address_vgpr, record_value_vgpr, record_value_vgpr,
            /*return_old_value=*/true, kAmdGpuScopeDevice, arch));
    require_emission(append_moi_global_atomic_wait(words, arch),
                     "ConSan MOI first-light probe could not wait for a dispatch slot claim");
    require_emission.append("ConSan MOI first-light probe could not inspect a dispatch slot",
                            instrumentation::build_v_cmp_ne_u32_vcc(scalar_positive_inline_u32(0),
                                                                    record_value_vgpr, arch));
    require_emission(
        sequence.emit_branch(dispatch_occupied_label, InstructionSequence::BranchKind::VccNonzero),
        "ConSan MOI first-light probe could not inspect its dispatch claim");
    require_emission(
        sequence.emit(instrumentation::build_v_cmp_ne_u32_vcc(scalar_positive_inline_u32(0),
                                                              record_value_high_vgpr, arch)) &&
            sequence.emit_branch(dispatch_occupied_label,
                                 InstructionSequence::BranchKind::VccNonzero) &&
            append_atomic_fetch_add_one_u32(
                words, base + offsetof(ConSanMoiReportHeader, record_replay_dispatch_token_count),
                record_compare_vgpr, record_address_vgpr, arch) &&
            sequence.emit_branch(dispatch_bank_ready_label,
                                 InstructionSequence::BranchKind::Unconditional) &&
            sequence.bind(dispatch_occupied_label),
        "ConSan MOI first-light dispatch publication is invalid");

    require_emission(
        sequence.emit_all(instrumentation::build_v_mov_b32_literal(
                              record_compare_vgpr,
                              static_cast<uint32_t>(kConSanMoiRecordReplayClaimTokenXorMask), arch),
                          instrumentation::build_v_xor_b32(record_value_vgpr,
                                                           vector_source_vgpr(record_compare_vgpr),
                                                           record_value_vgpr, arch)) &&
            append_moi_report_dispatch_id_word(words, dispatch_id_sources, record_compare_vgpr,
                                               /*high_word=*/false, arch),
        "ConSan MOI first-light probe could not compare a dispatch slot");
    require_emission.append("ConSan MOI first-light probe could not compare a dispatch low word",
                            instrumentation::build_v_cmp_ne_u32_vcc(
                                vector_source_vgpr(record_compare_vgpr), record_value_vgpr, arch));
    require_emission(
        sequence.emit_branch(dispatch_retry_label, InstructionSequence::BranchKind::VccNonzero),
        "ConSan MOI first-light probe could not branch to its dispatch retry");

    require_emission(
        sequence.emit_all(instrumentation::build_v_mov_b32_literal(
                              record_compare_vgpr,
                              static_cast<uint32_t>(kConSanMoiRecordReplayClaimTokenXorMask >> 32u),
                              arch),
                          instrumentation::build_v_xor_b32(record_value_high_vgpr,
                                                           vector_source_vgpr(record_compare_vgpr),
                                                           record_value_high_vgpr, arch)) &&
            append_moi_report_dispatch_id_word(words, dispatch_id_sources, record_compare_vgpr,
                                               /*high_word=*/true, arch),
        "ConSan MOI first-light probe could not compare a dispatch slot");
    require_emission.append(
        "ConSan MOI first-light probe could not compare a dispatch high word",
        instrumentation::build_v_cmp_ne_u32_vcc(vector_source_vgpr(record_compare_vgpr),
                                                record_value_high_vgpr, arch));
    require_emission(
        sequence.emit_branch(dispatch_retry_label, InstructionSequence::BranchKind::VccNonzero) &&
            sequence.emit_branch(dispatch_bank_ready_label,
                                 InstructionSequence::BranchKind::Unconditional) &&
            sequence.bind(dispatch_retry_label),
        "ConSan MOI first-light dispatch-retry path is invalid");

    // Mirror consan_moi_record_replay_advance_probe(): the incremented probe
    // count produces a triangular walk through the power-of-two directory.
    require_emission.append(
        "ConSan MOI first-light probe could not advance its dispatch probe",
        instrumentation::build_v_add_u32_literal(bank_probe_count_vgpr, record_compare_vgpr, 1u,
                                                 bank_probe_count_vgpr, arch),
        instrumentation::build_v_mov_b32_literal(
            record_compare_vgpr,
            std::min(layout.record_replay_dispatch_token_capacity,
                     kConSanMoiRecordReplayProbeLimit),
            arch),
        instrumentation::build_v_cmp_gt_u32_vcc(vector_source_vgpr(record_compare_vgpr),
                                                bank_probe_count_vgpr, arch));
    require_emission(
        sequence.emit_branch(dispatch_saturated_label, InstructionSequence::BranchKind::VccZero),
        "ConSan MOI first-light probe could not bound its dispatch retry");
    require_emission(
        sequence.emit_all(instrumentation::build_v_add_u32(dispatch_bank_vgpr,
                                                           vector_source_vgpr(dispatch_bank_vgpr),
                                                           bank_probe_count_vgpr, arch),
                          instrumentation::build_v_and_b32_literal(
                              dispatch_bank_vgpr, layout.record_replay_dispatch_token_capacity - 1u,
                              dispatch_bank_vgpr, arch)) &&
            append_claim_token(record_value_vgpr, record_value_high_vgpr, record_compare_vgpr) &&
            sequence.emit_branch(dispatch_probe_label,
                                 InstructionSequence::BranchKind::Unconditional) &&
            sequence.bind(dispatch_saturated_label) &&
            append_atomic_or_u32_literal(words, base + offsetof(ConSanMoiReportHeader, flags),
                                         kConSanMoiReportFlagRecordReplayBankSaturated |
                                             kConSanMoiReportFlagRecordReplayDispatchBankSaturated,
                                         record_address_vgpr, arch) &&
            sequence.emit_branch(restore_exec_label,
                                 InstructionSequence::BranchKind::Unconditional) &&
            sequence.bind(dispatch_bank_ready_label),
        "ConSan MOI first-light dispatch-directory path is invalid");
  }

  const auto append_owner_bank_start = [&](uint32_t site_token) {
    if (!append_owner_id(owner_bank_vgpr))
      return false;
    const std::array<ConSanMoiWorkgroupSource, 3> sources = {
        workgroup_sources.x, workgroup_sources.y, workgroup_sources.z};
    for (const ConSanMoiWorkgroupSource &source : sources) {
      if (!sequence.emit(instrumentation::build_v_mul_lo_u32_literal(
              owner_bank_vgpr, record_value_high_vgpr, kConSanMoiRecordReplayIdentityHashMultiplier,
              owner_bank_vgpr, arch)))
        return false;
      if (source.has_value()) {
        if (!consan_detail::append_workgroup_source_value(words, source, record_value_vgpr, arch)) {
          return false;
        }
      } else {
        words.push_back(
            build_v_mov_b32_e32(record_value_vgpr, scalar_positive_inline_u32(0), arch));
      }
      if (!sequence.emit(instrumentation::build_v_xor_b32(
              owner_bank_vgpr, vector_source_vgpr(owner_bank_vgpr), record_value_vgpr, arch)))
        return false;
    }
    // Preserve the unbounded owner/workgroup/dispatch/address hash for the
    // access publication token. The bounded table slot additionally mixes the
    // static site before it advances independently while probing.
    return sequence.emit_all(
        instrumentation::build_v_mul_lo_u32_literal(owner_bank_vgpr, record_value_high_vgpr,
                                                    kConSanMoiRecordReplayIdentityHashMultiplier,
                                                    owner_bank_vgpr, arch),
        instrumentation::build_v_xor_b32(owner_bank_vgpr, vector_source_vgpr(owner_bank_vgpr),
                                         dispatch_bank_vgpr, arch),
        instrumentation::build_v_mul_lo_u32_literal(owner_bank_vgpr, record_value_high_vgpr,
                                                    kConSanMoiRecordReplayIdentityHashMultiplier,
                                                    owner_bank_vgpr, arch),
        instrumentation::build_v_xor_b32(owner_bank_vgpr, address_key_sgpr, owner_bank_vgpr, arch),
        build_v_mov_b32_e32(identity_claim_hash_vgpr, vector_source_vgpr(owner_bank_vgpr), arch),
        instrumentation::build_v_mul_lo_u32_literal(owner_bank_vgpr, record_value_high_vgpr,
                                                    kConSanMoiRecordReplayIdentityHashMultiplier,
                                                    owner_bank_vgpr, arch),
        instrumentation::build_v_mov_b32_literal(record_value_vgpr, site_token, arch),
        instrumentation::build_v_xor_b32(owner_bank_vgpr, vector_source_vgpr(owner_bank_vgpr),
                                         record_value_vgpr, arch),
        instrumentation::build_v_and_b32_literal(owner_bank_vgpr, access_record_capacity - 1u,
                                                 owner_bank_vgpr, arch),
        build_v_mov_b32_e32(bank_probe_count_vgpr, scalar_positive_inline_u32(0), arch));
  };

  // Each automatic publication represents one exact address group. Direct
  // caller-owned layouts retain their historical one-record capacity and let
  // the per-slot atomic claim elect the first lane that executes the site.
  for (size_t range_index = 0; range_index < access_ranges.size(); ++range_index) {
    const auto publication_done_label = sequence.make_label();
    const auto token_ready_label = sequence.make_label();
    const auto occupied_label = sequence.make_label();
    const auto owner_probe_label = sequence.make_label();
    const auto owner_retry_label = sequence.make_label();
    const auto saturation_label = sequence.make_label();
    const auto owner_saturation_label = sequence.make_label();
    const auto publication_observe_label = sequence.make_label();
    const auto publication_incomplete_label = sequence.make_label();
    const ConSanAccessRange &range = access_ranges[range_index];
    const uint32_t site_token =
        automatic_banked_capture ? logical_range_index + static_cast<uint32_t>(range_index) : 0u;
    const uint64_t access_record_base =
        automatic_banked_capture ? base + layout.access_records_offset
                                 : base + layout.access_records_offset +
                                       (static_cast<uint64_t>(record_index) + range_index) *
                                           sizeof(ConSanMoiAccessRecord);

    if (automatic_banked_capture) {
      require_emission(append_owner_bank_start(site_token) && sequence.bind(owner_probe_label),
                       "ConSan MOI first-light probe could not initialize its access-identity "
                       "probe");
    }

    // Claim one bounded slot with a reversible encoding of the full hardware
    // dispatch ID. The report-wide table keeps the dispatch bank stable across
    // all sites; this per-site probe resolves distinct workgroup/wave owners
    // within that bank. access_kind is committed atomically only after every
    // payload store has drained.
    require_emission(materialize_banked_record_address(
                         access_record_base + offsetof(ConSanMoiAccessRecord, claim_token)) &&
                         (automatic_banked_capture
                              ? append_access_claim_token(record_value_vgpr, record_value_high_vgpr,
                                                          record_compare_vgpr, site_token)
                              : append_claim_token(record_value_vgpr, record_value_high_vgpr,
                                                   record_compare_vgpr)),
                     "ConSan MOI first-light probe could not materialize its bank claim");
    require_emission.append("ConSan MOI first-light probe could not validate its bank claim",
                            instrumentation::build_v_cmp_ne_u32_vcc(scalar_positive_inline_u32(0),
                                                                    record_value_vgpr, arch));
    require_emission(
        sequence.emit_branch(token_ready_label, InstructionSequence::BranchKind::VccNonzero),
        "ConSan MOI first-light probe could not validate its access token");
    require_emission(
        sequence.emit(instrumentation::build_v_cmp_ne_u32_vcc(scalar_positive_inline_u32(0),
                                                              record_value_high_vgpr, arch)) &&
            sequence.emit_branch(token_ready_label, InstructionSequence::BranchKind::VccNonzero) &&
            sequence.emit_branch(saturation_label,
                                 InstructionSequence::BranchKind::Unconditional) &&
            sequence.bind(token_ready_label),
        "ConSan MOI first-light access-token path is invalid");
    words.push_back(build_v_mov_b32_e32(record_compare_vgpr, scalar_positive_inline_u32(0), arch));
    words.push_back(
        build_v_mov_b32_e32(record_compare_high_vgpr, scalar_positive_inline_u32(0), arch));
    require_emission.append("ConSan MOI first-light probe could not encode its publication claim",
                            instrumentation::build_flat_atomic_cmpswap_b64(
                                record_address_vgpr, record_value_vgpr, record_value_vgpr,
                                /*return_old_value=*/true, kAmdGpuScopeDevice, arch));
    require_emission(append_moi_global_atomic_wait(words, arch),
                     "ConSan MOI first-light probe could not wait for its publication claim");
    require_emission.append("ConSan MOI first-light probe could not test its publication claim",
                            instrumentation::build_v_cmp_ne_u32_vcc(scalar_positive_inline_u32(0),
                                                                    record_value_vgpr, arch));
    require_emission(
        sequence.emit_branch(occupied_label, InstructionSequence::BranchKind::VccNonzero),
        "ConSan MOI first-light probe could not inspect its access claim low word");
    require_emission(
        sequence.emit(instrumentation::build_v_cmp_ne_u32_vcc(scalar_positive_inline_u32(0),
                                                              record_value_high_vgpr, arch)) &&
            sequence.emit_branch(occupied_label, InstructionSequence::BranchKind::VccNonzero),
        "ConSan MOI first-light probe could not inspect its access claim high word");
    const uint32_t visible_record_count =
        automatic_banked_capture ? access_record_capacity : record_count;
    require_emission(
        append_store_u32_literal(words, base + offsetof(ConSanMoiReportHeader, access_record_count),
                                 visible_record_count, scratch_vgpr, arch) &&
            append_atomic_fetch_add_one_u32(words,
                                            base + offsetof(ConSanMoiReportHeader, event_counter),
                                            record_compare_vgpr, record_address_vgpr, arch) &&
            materialize_banked_record_address(access_record_base),
        "ConSan MOI first-light probe could not encode record stores");
    // Recover an overlapping guest address only after the final bank address
    // is materialized, so the recorded LDS range cannot accidentally become a
    // transient bank-selection value.
    require_emission(reload_spilled_lds_byte_offset(),
                     "ConSan MOI first-light probe could not recover its spilled LDS address");
    require_emission(
        record.store_vgpr(offsetof(ConSanMoiAccessRecord, event_index), record_compare_vgpr) &&
            append_store_moi_report_dispatch_id_pair(record, dispatch_id_sources,
                                                     offsetof(ConSanMoiAccessRecord, generation)) &&
            record.store_workgroup(offsetof(ConSanMoiAccessRecord, workgroup_x),
                                   workgroup_sources.x) &&
            record.store_workgroup(offsetof(ConSanMoiAccessRecord, workgroup_y),
                                   workgroup_sources.y) &&
            record.store_workgroup(offsetof(ConSanMoiAccessRecord, workgroup_z),
                                   workgroup_sources.z) &&
            record.store_sgpr(offsetof(ConSanMoiAccessRecord, lane_mask),
                              automatic_banked_capture ? address_group_exec_sgpr
                                                       : *point.moi_exec_save_sgpr) &&
            record.store_sgpr(offsetof(ConSanMoiAccessRecord, lane_mask) + sizeof(uint32_t),
                              automatic_banked_capture
                                  ? static_cast<uint16_t>(address_group_exec_sgpr + 1u)
                                  : static_cast<uint16_t>(*point.moi_exec_save_sgpr + 1u)) &&
            (!point.moi_owner_epoch_vgprs.owner() ||
             record.store_vgpr(offsetof(ConSanMoiAccessRecord, wave_id),
                               point.moi_owner_epoch_vgprs->owner)) &&
            (!point.moi_persistent_sgprs.owner() ||
             record.store_sgpr(offsetof(ConSanMoiAccessRecord, wave_id),
                               *point.moi_persistent_sgprs.owner())) &&
            (!point.moi_owner_epoch_vgprs.epoch() ||
             record.store_vgpr(offsetof(ConSanMoiAccessRecord, epoch),
                               point.moi_owner_epoch_vgprs->epoch)) &&
            record.store_literal(offsetof(ConSanMoiAccessRecord, instruction_offset),
                                 static_cast<uint32_t>(candidate.anchor())) &&
            record.store_literal(offsetof(ConSanMoiAccessRecord, site_token), site_token) &&
            record.store_literal(
                offsetof(ConSanMoiAccessRecord, flags),
                automatic_banked_capture ? kConSanMoiAccessRecordFlagExactAddressGroupMask : 0u),
        "ConSan MOI first-light probe could not encode record stores");
    if (derived_owner_vgpr) {
      words.insert(words.end(), derived_owner_words.begin(), derived_owner_words.end());
      require_emission(
          record.store_vgpr(offsetof(ConSanMoiAccessRecord, wave_id), *derived_owner_vgpr),
          "ConSan MOI first-light probe could not encode derived owner");
    }
    if (private_epoch_offset) {
      require_emission.append(
          "ConSan MOI first-light probe could not load private epoch state",
          instrumentation::build_private_load_b32(record_value_vgpr, *private_epoch_offset, arch),
          instrumentation::build_s_wait_private_load0(arch));
      require_emission(record.store_vgpr(offsetof(ConSanMoiAccessRecord, epoch), record_value_vgpr),
                       "ConSan MOI first-light probe could not store private epoch state");
    } else if (point.moi_persistent_sgprs.epoch()) {
      words.push_back(
          build_v_mov_b32_e32(record_value_vgpr, *point.moi_persistent_sgprs.epoch(), arch));
      require_emission(record.store_vgpr(offsetof(ConSanMoiAccessRecord, epoch), record_value_vgpr),
                       "ConSan MOI first-light probe could not store scalar epoch state");
    }

    const std::optional<uint16_t> effective_lds_byte_offset_vgpr =
        append_effective_range_offset(range, record_value_vgpr);
    if (!effective_lds_byte_offset_vgpr) {
      errors.emplace_back("ConSan MOI first-light probe could not encode LDS byte offset");
      return std::nullopt;
    }
    const ConSanMoiLdsCellRange static_range =
        consan_moi_lds_cell_range_for_bytes(candidate.lowering_offset(range), range.byte_width);
    require_emission(record.store_vgpr(offsetof(ConSanMoiAccessRecord, lds_byte_offset),
                                       *effective_lds_byte_offset_vgpr),
                     "ConSan MOI first-light probe could not encode range offset");
    require_emission.append("ConSan MOI first-light probe could not encode start cell",
                            instrumentation::build_v_lshrrev_b32(
                                record_value_vgpr,
                                scalar_positive_inline_u32(consan_moi_shadow_cell::granule_shift),
                                *effective_lds_byte_offset_vgpr, arch));
    require_emission(
        record.store_vgpr(offsetof(ConSanMoiAccessRecord, start_cell), record_value_vgpr) &&
            record.store_literal(offsetof(ConSanMoiAccessRecord, lds_byte_count),
                                 range.byte_width) &&
            record.store_literal(offsetof(ConSanMoiAccessRecord, cell_count),
                                 static_range.cell_count),
        "ConSan MOI first-light probe could not encode range fields");
    require_emission.append(
        "ConSan MOI first-light probe could not drain record stores",
        instrumentation::build_s_wait_global_store0(arch),
        instrumentation::build_v_add_u64_signed_i24(
            record_address_vgpr, offsetof(ConSanMoiAccessRecord, access_kind), arch));
    require_emission.append(
        "ConSan MOI first-light probe could not commit its publication",
        instrumentation::build_v_mov_b32_literal(record_value_vgpr, static_cast<uint32_t>(kind),
                                                 arch),
        instrumentation::build_v_mov_b32_literal(
            record_value_high_vgpr, static_cast<uint32_t>(ConSanMoiShadowAccessKind::Empty), arch),
        instrumentation::build_flat_atomic_cmpswap_b32(
            record_address_vgpr, record_value_vgpr, record_value_vgpr,
            /*return_old_value=*/true, kAmdGpuScopeDevice, arch));
    require_emission(append_moi_global_atomic_wait(words, arch),
                     "ConSan MOI first-light probe could not wait for its publication commit");
    require_emission.append(
        "ConSan MOI first-light probe could not validate its publication commit",
        instrumentation::build_v_cmp_ne_u32_vcc(
            scalar_positive_inline_u32(static_cast<uint32_t>(ConSanMoiShadowAccessKind::Empty)),
            record_value_vgpr, arch));
    require_emission(
        sequence.emit_branch(saturation_label, InstructionSequence::BranchKind::VccNonzero) &&
            sequence.emit_branch(publication_done_label,
                                 InstructionSequence::BranchKind::Unconditional) &&
            sequence.bind(occupied_label),
        "ConSan MOI first-light publication-claim path is invalid");

    if (automatic_banked_capture) {
      // Preserve the returned token in the now-dead address pair, recompute
      // this execution identity's fingerprint, and bypass unrelated
      // in-flight claims without waiting for their publishers.
      words.push_back(
          build_v_mov_b32_e32(record_address_vgpr, vector_source_vgpr(record_value_vgpr), arch));
      words.push_back(build_v_mov_b32_e32(static_cast<uint16_t>(record_address_vgpr + 1u),
                                          vector_source_vgpr(record_value_high_vgpr), arch));
      require_emission(append_access_claim_token(record_compare_vgpr, record_compare_high_vgpr,
                                                 record_value_vgpr, site_token),
                       "ConSan MOI first-light probe could not rebuild its identity claim");
      require_emission.append(
          "ConSan MOI first-light probe could not compare its identity claim",
          instrumentation::build_v_cmp_ne_u32_vcc(vector_source_vgpr(record_compare_vgpr),
                                                  record_address_vgpr, arch));
      require_emission(
          sequence.emit_branch(owner_retry_label, InstructionSequence::BranchKind::VccNonzero),
          "ConSan MOI first-light probe could not retry an access identity");
      require_emission(sequence.emit(instrumentation::build_v_cmp_ne_u32_vcc(
                           vector_source_vgpr(record_compare_high_vgpr),
                           static_cast<uint16_t>(record_address_vgpr + 1u), arch)) &&
                           sequence.emit_branch(owner_retry_label,
                                                InstructionSequence::BranchKind::VccNonzero) &&
                           materialize_banked_record_address(
                               access_record_base + offsetof(ConSanMoiAccessRecord, access_kind)),
                       "ConSan MOI first-light access-identity retry path is invalid");
    } else {
      // Direct capture retains the historical reversible dispatch token.
      require_emission.append(
          "ConSan MOI first-light probe could not compare its dispatch claim",
          instrumentation::build_v_mov_b32_literal(
              record_compare_vgpr, static_cast<uint32_t>(kConSanMoiRecordReplayClaimTokenXorMask),
              arch),
          instrumentation::build_v_xor_b32(
              record_value_vgpr, vector_source_vgpr(record_compare_vgpr), record_value_vgpr, arch));
      require_emission(append_moi_report_dispatch_id_word(words, dispatch_id_sources,
                                                          record_compare_vgpr,
                                                          /*high_word=*/false, arch),
                       "ConSan MOI first-light probe could not materialize a dispatch low word");
      require_emission.append(
          "ConSan MOI first-light probe could not compare a direct dispatch low word",
          instrumentation::build_v_cmp_ne_u32_vcc(vector_source_vgpr(record_compare_vgpr),
                                                  record_value_vgpr, arch));
      require_emission(
          sequence.emit_branch(saturation_label, InstructionSequence::BranchKind::VccNonzero),
          "ConSan MOI first-light probe could not reject a dispatch low mismatch");

      require_emission.append(
          "ConSan MOI first-light probe could not compare its dispatch claim",
          instrumentation::build_v_mov_b32_literal(
              record_compare_vgpr,
              static_cast<uint32_t>(kConSanMoiRecordReplayClaimTokenXorMask >> 32u), arch),
          instrumentation::build_v_xor_b32(record_value_high_vgpr,
                                           vector_source_vgpr(record_compare_vgpr),
                                           record_value_high_vgpr, arch));
      require_emission(append_moi_report_dispatch_id_word(words, dispatch_id_sources,
                                                          record_compare_vgpr,
                                                          /*high_word=*/true, arch),
                       "ConSan MOI first-light probe could not materialize a dispatch high word");
      require_emission.append(
          "ConSan MOI first-light probe could not compare its dispatch claim",
          instrumentation::build_v_cmp_ne_u32_vcc(vector_source_vgpr(record_compare_vgpr),
                                                  record_value_high_vgpr, arch));
      require_emission(
          sequence.emit_branch(saturation_label, InstructionSequence::BranchKind::VccNonzero),
          "ConSan MOI first-light probe could not reject a dispatch high mismatch");
      require_emission.append(
          "ConSan MOI first-light probe could not address a direct access kind",
          instrumentation::build_v_add_u64_signed_i24(
              record_address_vgpr, offsetof(ConSanMoiAccessRecord, access_kind), arch));
    }

    // Use a no-op atomic compare-and-swap as an acquire-style read of the
    // access_kind commit. One lane publishes for each
    // dispatch/workgroup/wave/site/address-group identity, and a wave cannot
    // publish that same first-light identity concurrently with itself.
    // Therefore an incomplete automatic claim with the same compact token
    // belongs to a distinct identity whose fingerprint collided; probe another
    // slot instead of waiting for a wave that may not be scheduled while this
    // wave is resident. A committed record is qualified against the complete
    // identity below. The legacy direct layout has nowhere else to probe and
    // retains its fail-closed result.
    require_emission(sequence.bind(publication_observe_label),
                     "ConSan MOI first-light probe could not bind its publication observer");
    words.push_back(build_v_mov_b32_e32(record_value_vgpr, scalar_positive_inline_u32(0), arch));
    words.push_back(
        build_v_mov_b32_e32(record_value_high_vgpr, scalar_positive_inline_u32(0), arch));
    require_emission.append("ConSan MOI first-light probe could not observe its publication commit",
                            instrumentation::build_flat_atomic_cmpswap_b32(
                                record_address_vgpr, record_value_vgpr, record_value_vgpr,
                                /*return_old_value=*/true, kAmdGpuScopeDevice, arch));
    require_emission(append_moi_global_atomic_wait(words, arch),
                     "ConSan MOI first-light probe could not wait for its publication observation");
    require_emission(
        sequence.emit(instrumentation::build_v_cmp_eq_u32_vcc(
            scalar_positive_inline_u32(static_cast<uint32_t>(ConSanMoiShadowAccessKind::Empty)),
            record_value_vgpr, arch)) &&
            sequence.emit_branch(automatic_banked_capture ? owner_retry_label
                                                          : publication_incomplete_label,
                                 InstructionSequence::BranchKind::VccNonzero),
        "ConSan MOI first-light probe could not handle incomplete publication");
    require_emission(
        sequence.emit(instrumentation::build_v_cmp_ne_u32_vcc(
            scalar_positive_inline_u32(static_cast<uint32_t>(kind)), record_value_vgpr, arch)) &&
            sequence.emit_branch(automatic_banked_capture ? owner_retry_label : saturation_label,
                                 InstructionSequence::BranchKind::VccNonzero),
        "ConSan MOI first-light probe could not reject a publication-kind mismatch");
    require_emission.append(
        "ConSan MOI first-light probe could not recover its record base address",
        instrumentation::build_v_add_u64_signed_i24(
            record_address_vgpr,
            -static_cast<int32_t>(offsetof(ConSanMoiAccessRecord, access_kind)), arch));

    if (automatic_banked_capture) {
      const auto reject_dispatch_mismatch = [&](uint32_t offset, bool high_word) {
        if (!record.load(offset, record_value_vgpr) ||
            !append_moi_report_dispatch_id_word(words, dispatch_id_sources, record_compare_vgpr,
                                                high_word, arch)) {
          return false;
        }
        if (!sequence.emit(instrumentation::build_v_cmp_ne_u32_vcc(
                vector_source_vgpr(record_compare_vgpr), record_value_vgpr, arch)))
          return false;
        return sequence.emit_branch(owner_retry_label, InstructionSequence::BranchKind::VccNonzero);
      };
      require_emission(reject_dispatch_mismatch(offsetof(ConSanMoiAccessRecord, generation),
                                                /*high_word=*/false) &&
                           reject_dispatch_mismatch(offsetof(ConSanMoiAccessRecord, generation) +
                                                        sizeof(uint32_t),
                                                    /*high_word=*/true),
                       "ConSan MOI first-light probe could not qualify its retained dispatch");
    }

    require_emission(record.load(offsetof(ConSanMoiAccessRecord, site_token), record_value_vgpr),
                     "ConSan MOI first-light probe could not load its retained site");
    require_emission.append(
        "ConSan MOI first-light probe could not qualify its retained site",
        instrumentation::build_v_mov_b32_literal(record_compare_vgpr, site_token, arch),
        instrumentation::build_v_cmp_ne_u32_vcc(vector_source_vgpr(record_compare_vgpr),
                                                record_value_vgpr, arch));
    require_emission(
        sequence.emit_branch(automatic_banked_capture ? owner_retry_label : saturation_label,
                             InstructionSequence::BranchKind::VccNonzero),
        "ConSan MOI first-light probe could not reject a retained-site mismatch");

    const auto reject_workgroup_mismatch = [&](uint32_t offset,
                                               const ConSanMoiWorkgroupSource &source) {
      if (!record.load(offset, record_value_vgpr))
        return false;
      if (source.has_value()) {
        if (!consan_detail::append_workgroup_source_value(words, source, record_compare_vgpr, arch))
          return false;
      } else {
        words.push_back(
            build_v_mov_b32_e32(record_compare_vgpr, scalar_positive_inline_u32(0), arch));
      }
      if (!sequence.emit(instrumentation::build_v_cmp_ne_u32_vcc(
              vector_source_vgpr(record_compare_vgpr), record_value_vgpr, arch)))
        return false;
      return sequence.emit_branch(automatic_banked_capture ? owner_retry_label : saturation_label,
                                  InstructionSequence::BranchKind::VccNonzero);
    };
    require_emission(reject_workgroup_mismatch(offsetof(ConSanMoiAccessRecord, workgroup_x),
                                               workgroup_sources.x) &&
                         reject_workgroup_mismatch(offsetof(ConSanMoiAccessRecord, workgroup_y),
                                                   workgroup_sources.y) &&
                         reject_workgroup_mismatch(offsetof(ConSanMoiAccessRecord, workgroup_z),
                                                   workgroup_sources.z),
                     "ConSan MOI first-light probe could not qualify its retained workgroup");

    // Direct first-light capture intentionally coalesces waves within one
    // dispatch/workgroup. The automatic table has spare slots and retains
    // wave identity explicitly so replay can cover every execution owner.
    if (automatic_banked_capture) {
      require_emission(record.load(offsetof(ConSanMoiAccessRecord, wave_id), record_value_vgpr) &&
                           append_owner_id(record_compare_vgpr),
                       "ConSan MOI first-light probe could not qualify its retained owner");
      require_emission.append(
          "ConSan MOI first-light probe could not compare its retained owner",
          instrumentation::build_v_cmp_ne_u32_vcc(vector_source_vgpr(record_compare_vgpr),
                                                  record_value_vgpr, arch));
      require_emission(
          sequence.emit_branch(owner_retry_label, InstructionSequence::BranchKind::VccNonzero),
          "ConSan MOI first-light probe could not qualify its retained owner");

      require_emission(
          record.load(offsetof(ConSanMoiAccessRecord, lds_byte_offset), record_value_vgpr),
          "ConSan MOI first-light probe could not load its retained address group");
      words.push_back(build_v_mov_b32_e32(record_compare_vgpr, address_key_sgpr, arch));
      if (candidate.lowering_offset(range) != 0u) {
        require_emission.append(
            "ConSan MOI first-light probe could not qualify its retained address group",
            instrumentation::build_v_mov_b32_literal(record_compare_high_vgpr,
                                                     candidate.lowering_offset(range), arch),
            instrumentation::build_v_add_u32(record_compare_vgpr,
                                             vector_source_vgpr(record_compare_vgpr),
                                             record_compare_high_vgpr, arch));
      }
      require_emission.append(
          "ConSan MOI first-light probe could not compare its retained address group",
          instrumentation::build_v_cmp_ne_u32_vcc(vector_source_vgpr(record_compare_vgpr),
                                                  record_value_vgpr, arch));
      require_emission(
          sequence.emit_branch(owner_retry_label, InstructionSequence::BranchKind::VccNonzero),
          "ConSan MOI first-light probe could not qualify its retained address group");
    }
    require_emission(sequence.emit_branch(publication_done_label,
                                          InstructionSequence::BranchKind::Unconditional),
                     "ConSan MOI first-light probe could not qualify its retained identity");

    if (automatic_banked_capture) {
      require_emission(sequence.bind(owner_retry_label),
                       "ConSan MOI first-light probe could not bind its access-identity retry");
      // Mirror consan_moi_record_replay_advance_probe(): triangular probing is
      // a permutation of every power-of-two table and avoids the primary
      // clustering of a linear walk at higher load.
      require_emission.append(
          "ConSan MOI first-light probe could not advance its access-identity probe",
          instrumentation::build_v_add_u32_literal(bank_probe_count_vgpr, record_compare_vgpr, 1u,
                                                   bank_probe_count_vgpr, arch),
          instrumentation::build_v_mov_b32_literal(
              record_compare_vgpr,
              std::min(access_record_capacity, kConSanMoiRecordReplayProbeLimit), arch),
          instrumentation::build_v_cmp_gt_u32_vcc(vector_source_vgpr(record_compare_vgpr),
                                                  bank_probe_count_vgpr, arch));
      require_emission(
          sequence.emit_branch(owner_saturation_label, InstructionSequence::BranchKind::VccZero),
          "ConSan MOI first-light probe could not bound its access-identity retry");
      require_emission(
          sequence.emit_all(
              instrumentation::build_v_add_u32(owner_bank_vgpr, vector_source_vgpr(owner_bank_vgpr),
                                               bank_probe_count_vgpr, arch),
              instrumentation::build_v_and_b32_literal(owner_bank_vgpr, access_record_capacity - 1u,
                                                       owner_bank_vgpr, arch)) &&
              sequence.emit_branch(owner_probe_label,
                                   InstructionSequence::BranchKind::Unconditional) &&
              sequence.bind(owner_saturation_label) &&
              append_atomic_or_u32_literal(words, base + offsetof(ConSanMoiReportHeader, flags),
                                           kConSanMoiReportFlagRecordReplayBankSaturated |
                                               kConSanMoiReportFlagRecordReplayOwnerBankSaturated,
                                           record_address_vgpr, arch) &&
              sequence.emit_branch(publication_done_label,
                                   InstructionSequence::BranchKind::Unconditional),
          "ConSan MOI first-light access-identity path is invalid");
    } else {
      require_emission(sequence.bind(publication_incomplete_label) &&
                           append_atomic_or_u32_literal(
                               words, base + offsetof(ConSanMoiReportHeader, flags),
                               kConSanMoiReportFlagRecordReplayBankSaturated |
                                   kConSanMoiReportFlagRecordReplayPublicationIncomplete,
                               record_address_vgpr, arch) &&
                           sequence.emit_branch(publication_done_label,
                                                InstructionSequence::BranchKind::Unconditional),
                       "ConSan MOI first-light direct publication path is invalid");
    }

    require_emission(sequence.bind(saturation_label) &&
                         append_atomic_or_u32_literal(words,
                                                      base + offsetof(ConSanMoiReportHeader, flags),
                                                      kConSanMoiReportFlagRecordReplayBankSaturated,
                                                      record_address_vgpr, arch) &&
                         sequence.bind(publication_done_label),
                     "ConSan MOI first-light publication fast path is invalid");
  }

  if (automatic_banked_capture) {
    if (!address_group_loop) {
      errors.emplace_back("ConSan MOI first-light probe lost its address-group loop");
      return std::nullopt;
    }
    // Keep the scalar-mask dependency explicit before installing the next
    // EXEC. This matches the proven Inline Shadow traversal and prevents the
    // final divergent address group from being skipped on live hardware.
    require_emission(
        sequence.emit_all(
            instrumentation::build_s_xor_b64(*point.moi_exec_save_sgpr, *point.moi_exec_save_sgpr,
                                             address_group_exec_sgpr, arch),
            build_s_nop(0, arch), build_s_nop(0, arch),
            instrumentation::build_s_mov_b64(kAmdGpuExecLo, *point.moi_exec_save_sgpr, arch)) &&
            sequence.emit_branch(*address_group_loop, InstructionSequence::BranchKind::ExecNonzero),
        "ConSan MOI first-light probe could not branch to its next address group");
  }

  const uint16_t restore_exec_sgpr =
      automatic_banked_capture ? original_exec_sgpr : *point.moi_exec_save_sgpr;
  require_emission(sequence.bind(restore_exec_label),
                   "ConSan MOI first-light probe could not bind its EXEC restore");
  require_emission.append("ConSan MOI first-light probe could not restore EXEC",
                          instrumentation::build_s_mov_b64(kAmdGpuExecLo, restore_exec_sgpr, arch));
  require_emission(append_restore_moi_special_state(words, scalar_abi.special_state, arch),
                   "ConSan MOI first-light probe could not restore VCC/SCC");

  // Static first-light records do not consume the guest result. Publish them
  // before a load so that the record scratch window may overlap its destination
  // VGPRs. When the load also overwrites its address VGPR, the saved address
  // above remains outside that window until the displaced instruction executes.
  if (!request.moi_dynamic_access_records)
    sequence.require(append_guest_access());

  if (!sequence.finish(arch)) {
    errors.emplace_back("ConSan MOI first-light local branch is out of reach");
    return std::nullopt;
  }

  return words;
}

} // namespace consan_moi_impl
} // namespace rocjitsu
