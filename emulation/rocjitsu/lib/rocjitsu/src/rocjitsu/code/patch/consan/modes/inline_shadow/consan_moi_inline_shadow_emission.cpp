// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/modes/inline_shadow/consan_moi_inline_shadow_emission.h"

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/consan/consan_capability_contract.h"
#include "rocjitsu/code/patch/consan/consan_moi_access_target.h"
#include "rocjitsu/code/patch/consan/consan_moi_dynamic_record_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_exact_shadow_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_memory_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_native_abi.h"
#include "rocjitsu/code/patch/consan/consan_moi_probe_contracts.h"
#include "rocjitsu/code/patch/consan/consan_moi_relocation.h"
#include "rocjitsu/code/patch/consan/consan_moi_report_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_versioned_publication.h"
#include "rocjitsu/code/patch/consan/modes/inline_shadow/consan_moi_inline_shadow.h"
#include "rocjitsu/code/patch/instruction_sequence.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"
#include "rocjitsu/code/patch/spill_manager.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace rocjitsu::consan_moi_impl {

using consan_detail::append_moi_device_cache_refresh;
using consan_detail::append_moi_relocated_guest_access;
using consan_detail::append_reload_moi_spilled_vgpr;
using consan_detail::moi_guest_access_relocation_requires_adjusted_address;
using consan_detail::moi_spilled_vgpr_reload_result_name;
using consan_detail::MoiSpilledVgprReloadResult;
using consan_detail::MoiWorkgroupKeyRegisterPlan;
using consan_detail::MoiWorkitemOwnerDerivationPlan;
using consan_detail::range_overlaps;
using consan_detail::reject_optional_scratch_range_overlap;
using consan_moi_detail::append_atomic_fetch_add_one_u32;
using consan_moi_detail::append_atomic_load_u32;
using consan_moi_detail::append_compare_moi_report_dispatch_id_word;
using consan_moi_detail::append_dynamic_diagnostic_record_address;
using consan_moi_detail::append_load_u32_vgpr_at_offset;
using consan_moi_detail::append_moi_report_dispatch_id_pair;
using consan_moi_detail::append_moi_report_dispatch_id_word;
using consan_moi_detail::append_publish_first_active_lane_visible_evidence_if_zero;
using consan_moi_detail::append_reserve_bounded_dynamic_record_slot;
using consan_moi_detail::append_select_first_active_lane;
using consan_moi_detail::append_select_first_lane_in_exec_mask;
using consan_moi_detail::append_store_u32_sgpr;
using consan_moi_detail::append_store_u32_vgpr;
using consan_moi_detail::append_store_u32_vgpr_at_offset;
using consan_moi_detail::MoiVisibleEvidencePublicationResult;

[[nodiscard]] bool append_inline_shadow_diagnostic_words(
    std::vector<uint32_t> &words, const ConSanMoiCandidate &candidate,
    const MoiInlineShadowEmissionPlan &plan, rj_code_arch_t arch,
    const ConSanMoiReportBufferLayout &layout, uint16_t old_value_vgpr, uint16_t old_value_hi_vgpr,
    uint16_t current_value_vgpr, uint16_t current_field_vgpr, uint16_t lds_byte_offset_vgpr,
    uint32_t static_byte_offset, uint32_t byte_count, uint32_t shadow_granule_bytes,
    bool special_state_already_saved = false,
    std::optional<uint16_t> diagnostic_lane_mask_sgpr = std::nullopt,
    bool capture_first_diagnostic_only = false, bool workgroup_local_shadow = false,
    std::optional<uint16_t> prior_byte_provenance_vgpr = std::nullopt,
    std::optional<uint16_t> current_byte_provenance_vgpr = std::nullopt,
    std::optional<uint16_t> relative_cell_index_vgpr = std::nullopt,
    uint32_t relative_cell_index = 0u) {
  InstructionSequence sequence(words);
  const ConSanTargetProfile *target = consan_target_profile(arch);
  if (!target)
    return false;
  if (!plan.scalar_state.exec_save_sgpr || layout.diagnostic_capacity == 0)
    return true;
  if (shadow_granule_bytes != 1u && shadow_granule_bytes != consan_moi_shadow_cell::granule_bytes)
    return false;
  if (prior_byte_provenance_vgpr.has_value() != current_byte_provenance_vgpr.has_value())
    return false;

  const uint16_t tmp_vgpr = static_cast<uint16_t>(plan.scratch_vgpr + 4u);
  const uint64_t report_base = plan.report_buffer_address;
  const uint64_t diagnostic_base = report_base + layout.diagnostic_records_offset;

  if (!special_state_already_saved)
    sequence.require(append_save_moi_special_state(words, plan.scalar_abi.special_state, arch));

  const uint16_t workgroup_key = static_cast<uint16_t>(plan.scratch_vgpr + 19u);
  if (plan.track_atomics) {
    if (workgroup_local_shadow) {
      // An LDS shadow is workgroup-local. Its optional generation tag guards
      // reuse of the local cells; it is not the identity used by the global
      // atomic release/token tables. Validate those tables with the persistent
      // workgroup key for every local-shadow representation.
      if (plan.workgroup_key_registers.cached_key_vgpr) {
        words.push_back(build_v_mov_b32_e32(
            workgroup_key, vector_source_vgpr(*plan.workgroup_key_registers.cached_key_vgpr),
            arch));
      } else if (plan.workgroup_key_registers.cached_key_sgpr) {
        words.push_back(build_v_mov_b32_e32(workgroup_key,
                                            *plan.workgroup_key_registers.cached_key_sgpr, arch));
      } else {
        return false;
      }
    } else
      sequence.require(append_extract_exact_shadow_generation(
          words, workgroup_key, current_value_vgpr, static_cast<uint16_t>(current_value_vgpr + 1u),
          tmp_vgpr, arch));
  }

  // Conflict diagnostics are the cold path. Each predicate below can empty
  // EXEC, but merely leaving the remaining vector and memory instructions
  // predicated off still makes scalar execution walk the complete record
  // builder. Large clean kernels can encounter these probes billions of
  // times. Branch directly to the common state restoration whenever no lane
  // remains, preserving identical behavior for every actual conflict.
  const InstructionSequence::Label diagnostic_restore = sequence.make_label();
  const auto skip_diagnostic_if_empty = [&]() {
    sequence.branch(diagnostic_restore, InstructionSequence::BranchKind::ExecZero);
  };

  sequence.append(
      build_v_mov_b32_e32(tmp_vgpr, scalar_positive_inline_u32(0), arch),
      instrumentation::build_v_cmp_gt_u32_vcc(vector_source_vgpr(old_value_vgpr), tmp_vgpr, arch),
      instrumentation::build_s_and_saveexec_b64(*plan.scalar_state.exec_save_sgpr, kAmdGpuVccLo,
                                                arch));
  skip_diagnostic_if_empty();

  const uint16_t address_lo_vgpr = plan.scratch_vgpr;
  // Predicate save destinations are scratch: after each empty-mask branch,
  // only the narrowed EXEC matters. Reuse one pair instead of consuming the
  // VCC/SCC preservation window at higher offsets as predicates are reordered.
  const uint16_t predicate_exec_save_sgpr =
      static_cast<uint16_t>(*plan.scalar_state.exec_save_sgpr + 2u);
  if (!workgroup_local_shadow) {
    sequence
        .require(append_extract_exact_shadow_generation(words, tmp_vgpr, old_value_vgpr,
                                                        old_value_hi_vgpr, address_lo_vgpr, arch))
        .require(append_extract_exact_shadow_generation(
            words, current_field_vgpr, current_value_vgpr,
            static_cast<uint16_t>(current_value_vgpr + 1u), address_lo_vgpr, arch))
        .append(instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(current_field_vgpr),
                                                        tmp_vgpr, arch),
                instrumentation::build_s_and_saveexec_b64(predicate_exec_save_sgpr, kAmdGpuVccLo,
                                                          arch));
    skip_diagnostic_if_empty();
  }

  if (prior_byte_provenance_vgpr) {
    sequence.append(
        instrumentation::build_v_and_b32_literal(tmp_vgpr,
                                                 consan_moi_exact_byte_cell::byte_mask_mask,
                                                 *prior_byte_provenance_vgpr, arch),
        instrumentation::build_v_and_b32_literal(current_field_vgpr,
                                                 consan_moi_exact_byte_cell::byte_mask_mask,
                                                 *current_byte_provenance_vgpr, arch),
        instrumentation::build_v_and_b32(current_field_vgpr, vector_source_vgpr(tmp_vgpr),
                                         current_field_vgpr, arch),
        instrumentation::build_v_cmp_ne_u32_vcc(scalar_positive_inline_u32(0u), current_field_vgpr,
                                                arch),
        instrumentation::build_s_and_saveexec_b64(predicate_exec_save_sgpr, kAmdGpuVccLo, arch));
    skip_diagnostic_if_empty();
  }

  // Reject read/read traffic before extracting either owner or epoch. Preserve
  // owner-before-epoch ordering for the remaining lanes: most repeated
  // accesses by one workitem are ordered, while a same-site access from a
  // different physical lane still represents a distinct participant.
  const auto current_kind = consan_moi_shadow_kind_from_access_kind(candidate.site().kind);
  if (current_kind != ConSanMoiShadowAccessKind::Write) {
    sequence.append(
        instrumentation::build_v_and_b32_literal(
            tmp_vgpr, static_cast<uint32_t>(consan_moi_exact_shadow::access_kind_mask),
            old_value_vgpr, arch),
        instrumentation::build_v_cmp_ne_u32_vcc(
            scalar_positive_inline_u32(static_cast<uint32_t>(current_kind)), tmp_vgpr, arch),
        instrumentation::build_s_and_saveexec_b64(predicate_exec_save_sgpr, kAmdGpuVccLo, arch));
    skip_diagnostic_if_empty();
  }

  sequence
      .require(append_extract_exact_shadow_field(words, tmp_vgpr, old_value_vgpr,
                                                 consan_moi_exact_shadow::owner_shift,
                                                 consan_moi_exact_shadow::max_owner, arch))
      .require(append_extract_exact_shadow_field(words, current_field_vgpr, current_value_vgpr,
                                                 consan_moi_exact_shadow::owner_shift,
                                                 consan_moi_exact_shadow::max_owner, arch));
  if (prior_byte_provenance_vgpr) {
    const uint16_t candidate_exec = static_cast<uint16_t>(*plan.scalar_state.exec_save_sgpr + 2u);
    const uint16_t different_owner_exec =
        static_cast<uint16_t>(*plan.scalar_state.exec_save_sgpr + 4u);
    const uint16_t predicate_exec = static_cast<uint16_t>(*plan.scalar_state.exec_save_sgpr + 6u);
    sequence.append(
        instrumentation::build_s_mov_b64(candidate_exec, kAmdGpuExecLo, arch),
        instrumentation::build_v_cmp_ne_u32_vcc(vector_source_vgpr(current_field_vgpr), tmp_vgpr,
                                                arch),
        instrumentation::build_s_and_saveexec_b64(predicate_exec, kAmdGpuVccLo, arch),
        instrumentation::build_s_mov_b64(different_owner_exec, kAmdGpuExecLo, arch),
        instrumentation::build_s_mov_b64(kAmdGpuExecLo, candidate_exec, arch),
        instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(current_field_vgpr), tmp_vgpr,
                                                arch),
        instrumentation::build_s_and_saveexec_b64(predicate_exec, kAmdGpuVccLo, arch),
        instrumentation::build_v_lshrrev_b32(
            tmp_vgpr,
            scalar_positive_inline_u32(consan_moi_exact_shadow::instruction_offset_shift - 32u),
            old_value_hi_vgpr, arch),
        instrumentation::build_v_mov_b32_literal(current_field_vgpr,
                                                 static_cast<uint32_t>(candidate.anchor()), arch),
        instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(current_field_vgpr), tmp_vgpr,
                                                arch),
        instrumentation::build_s_and_saveexec_b64(predicate_exec, kAmdGpuVccLo, arch),
        instrumentation::build_v_lshrrev_b32(
            tmp_vgpr, scalar_positive_inline_u32(consan_moi_exact_byte_cell::lane_plus_one_shift),
            *prior_byte_provenance_vgpr, arch),
        instrumentation::build_v_lshrrev_b32(
            current_field_vgpr,
            scalar_positive_inline_u32(consan_moi_exact_byte_cell::lane_plus_one_shift),
            *current_byte_provenance_vgpr, arch),
        instrumentation::build_v_cmp_ne_u32_vcc(vector_source_vgpr(current_field_vgpr), tmp_vgpr,
                                                arch),
        instrumentation::build_s_and_saveexec_b64(predicate_exec, kAmdGpuVccLo, arch),
        instrumentation::build_s_xor_b64(kAmdGpuExecLo, different_owner_exec, kAmdGpuExecLo, arch));
  } else {
    sequence.append(
        instrumentation::build_v_cmp_ne_u32_vcc(vector_source_vgpr(current_field_vgpr), tmp_vgpr,
                                                arch),
        instrumentation::build_s_and_saveexec_b64(predicate_exec_save_sgpr, kAmdGpuVccLo, arch));
  }
  skip_diagnostic_if_empty();

  const bool has_epoch = plan.epoch_field.epoch_vgpr || plan.epoch_field.persistent_epoch_sgpr ||
                         plan.epoch_field.automatic_private_epoch;
  if (has_epoch) {
    sequence
        .require(append_extract_exact_shadow_field(words, tmp_vgpr, old_value_vgpr,
                                                   consan_moi_exact_shadow::epoch_shift,
                                                   consan_moi_exact_shadow::max_epoch, arch))
        .require(append_extract_exact_shadow_field(words, current_field_vgpr, current_value_vgpr,
                                                   consan_moi_exact_shadow::epoch_shift,
                                                   consan_moi_exact_shadow::max_epoch, arch))
        .append(instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(current_field_vgpr),
                                                        tmp_vgpr, arch),
                instrumentation::build_s_and_saveexec_b64(predicate_exec_save_sgpr, kAmdGpuVccLo,
                                                          arch));
    skip_diagnostic_if_empty();
  }

  if (plan.track_atomics) {
    const uint16_t base = plan.scratch_vgpr;
    const uint16_t exec_base = *plan.scalar_state.exec_save_sgpr;
    const uint16_t conflict_exec = static_cast<uint16_t>(exec_base + 2u);
    const uint16_t predicate_exec = static_cast<uint16_t>(exec_base + 6u);
    // The wave-coalesced exact-shadow caller keeps +7:+12 live across this
    // helper as its saved address, packed current value, and exact-byte
    // provenance. A multi-address wave returns to the partition loop after
    // token filtering, so borrowing those registers here corrupts the next
    // partition's publication. Keep access-time token state in the cold
    // diagnostic-only tail instead.
    const uint16_t token_version = static_cast<uint16_t>(base + 15u);
    const uint16_t token_kind = static_cast<uint16_t>(base + 16u);
    const uint16_t token_epoch = static_cast<uint16_t>(base + 17u);
    const uint16_t source_version = static_cast<uint16_t>(base + 18u);
    const uint16_t token_hash = static_cast<uint16_t>(base + 14u);
    const uint16_t source_address = static_cast<uint16_t>(base + 21u);
    const uint16_t prior_owner = static_cast<uint16_t>(base + 23u);
    // The packed access value is authoritative here: it was built from the
    // candidate's persistent owner source immediately before publication.
    // Extracting that value below also supports owner-private state without
    // requiring a second persistent VGPR or SGPR solely for diagnostics.
    sequence
        .require(append_extract_exact_shadow_field(words, current_field_vgpr, current_value_vgpr,
                                                   consan_moi_exact_shadow::owner_shift,
                                                   consan_moi_exact_shadow::max_owner, arch))
        .require(append_extract_exact_shadow_field(words, prior_owner, old_value_vgpr,
                                                   consan_moi_exact_shadow::owner_shift,
                                                   consan_moi_exact_shadow::max_owner, arch));

    const auto filter_ordered_pair = [&](uint16_t token_consumer_owner,
                                         uint16_t token_producer_owner,
                                         uint16_t consumer_packed_value,
                                         uint16_t producer_packed_value) {
      sequence.append(instrumentation::build_s_mov_b64(conflict_exec, kAmdGpuExecLo, arch));
      // Acquired-edge history is keyed by consumer segment so reuse of an
      // atomic barrier cannot overwrite evidence for an earlier generation.
      // The packed access supplies the same raw segment epoch used by the
      // acquire publisher when it derived this destination.
      sequence
          .require(append_extract_exact_shadow_field(words, source_version, consumer_packed_value,
                                                     consan_moi_exact_shadow::epoch_shift,
                                                     consan_moi_exact_shadow::max_epoch, arch))
          .require(append_inline_acquired_token_slot_address(
              words, plan.report_buffer_address + layout.inline_acquired_epoch_token_slots_offset,
              layout.inline_acquired_epoch_token_capacity, workgroup_key, token_consumer_owner,
              token_producer_owner, source_version, address_lo_vgpr, token_hash,
              static_cast<uint16_t>(address_lo_vgpr + 1u),
              /*release_sequence=*/false, arch));

      const auto narrow = [&]() {
        sequence.append(
            instrumentation::build_s_and_saveexec_b64(predicate_exec, kAmdGpuVccLo, arch));
      };
      const auto require_equal = [&](size_t offset, uint16_t expected) {
        sequence
            .require(append_load_u32_vgpr_at_offset(words, address_lo_vgpr, offset, tmp_vgpr, arch))
            .append(instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(expected), tmp_vgpr,
                                                            arch));
        narrow();
      };
      const auto require_dispatch_equal = [&](size_t offset, bool high_word) {
        sequence
            .require(append_load_u32_vgpr_at_offset(words, address_lo_vgpr, offset, tmp_vgpr, arch))
            .require(append_compare_moi_report_dispatch_id_word(words, plan.dispatch_id, tmp_vgpr,
                                                                source_version, high_word, arch));
        narrow();
      };
      const auto require_zero = [&](size_t offset) {
        sequence
            .require(append_load_u32_vgpr_at_offset(words, address_lo_vgpr, offset, tmp_vgpr, arch))
            .append(instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), tmp_vgpr,
                                                            arch));
        narrow();
      };

      // The acquiring atomic can populate this direct-mapped slot immediately
      // before the same wave reaches its ordered LDS access. Use a coherent
      // device-scope version read at both ends of the snapshot; an ordinary
      // flat load may otherwise reuse the zero observed while reserving the
      // slot and miss the completed token publication.
      sequence.require(append_atomic_load_u32(words, address_lo_vgpr, token_version, arch));
      sequence.append(instrumentation::build_v_cmp_ne_u32_vcc(scalar_positive_inline_u32(0),
                                                              token_version, arch));
      narrow();
      sequence.append(
          instrumentation::build_v_and_b32_literal(tmp_vgpr, 1u, token_version, arch),
          instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), tmp_vgpr, arch));
      narrow();
      require_equal(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, workgroup_key), workgroup_key);
      require_equal(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, consumer_owner_id),
                    token_consumer_owner);
      require_equal(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, producer_owner_id),
                    token_producer_owner);
      sequence
          .require(append_load_u32_vgpr_at_offset(
              words, address_lo_vgpr,
              offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, producer_epoch_plus_one), token_epoch,
              arch))
          .require(append_load_u32_vgpr_at_offset(
              words, address_lo_vgpr, offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, kind),
              token_kind, arch));
      require_dispatch_equal(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, dispatch_id),
                             /*high_word=*/false);
      require_dispatch_equal(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, dispatch_id) +
                                 sizeof(uint32_t),
                             /*high_word=*/true);
      sequence
          .require(append_load_u32_vgpr_at_offset(
              words, address_lo_vgpr,
              offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, source_release_address),
              source_address, arch))
          .require(append_load_u32_vgpr_at_offset(
              words, address_lo_vgpr,
              offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, source_release_address) +
                  sizeof(uint32_t),
              static_cast<uint16_t>(source_address + 1u), arch))
          .require(append_load_u32_vgpr_at_offset(
              words, address_lo_vgpr,
              offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, source_release_version),
              source_version, arch))
          .require(append_load_u32_vgpr_at_offset(
              words, address_lo_vgpr,
              offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, consumer_epoch_plus_one), token_hash,
              arch));
      require_zero(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, reservation_version));
      sequence.require(
          append_atomic_load_u32(words, address_lo_vgpr, static_cast<uint16_t>(base + 14u), arch));
      sequence.append(instrumentation::build_v_cmp_ne_u32_vcc(scalar_positive_inline_u32(0),
                                                              source_address, arch));
      narrow();
      sequence.append(instrumentation::build_v_cmp_ne_u32_vcc(scalar_positive_inline_u32(0),
                                                              source_version, arch));
      narrow();
      sequence.append(
          instrumentation::build_v_and_b32_literal(tmp_vgpr, 1u, source_version, arch),
          instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), tmp_vgpr, arch));
      narrow();
      sequence.append(instrumentation::build_v_cmp_eq_u32_vcc(
          vector_source_vgpr(token_version), static_cast<uint16_t>(base + 14u), arch));
      narrow();
      sequence.require(append_extract_exact_shadow_field(
          words, source_version, consumer_packed_value, consan_moi_exact_shadow::epoch_shift,
          consan_moi_exact_shadow::max_epoch, arch));
      sequence.append(instrumentation::build_v_mov_b32_literal(
                          tmp_vgpr, consan_moi_exact_shadow::max_epoch, arch),
                      instrumentation::build_v_cmp_ne_u32_vcc(vector_source_vgpr(source_version),
                                                              tmp_vgpr, arch));
      narrow();
      sequence.append(
          instrumentation::build_v_add_u32(source_version, scalar_positive_inline_u32(1),
                                           source_version, arch),
          instrumentation::build_v_cmp_gt_u32_vcc(vector_source_vgpr(token_hash), source_version,
                                                  arch),
          instrumentation::build_s_andn2_b64(kAmdGpuExecLo, kAmdGpuExecLo, kAmdGpuVccLo, arch));
      sequence.require(append_extract_exact_shadow_field(words, token_hash, producer_packed_value,
                                                         consan_moi_exact_shadow::epoch_shift,
                                                         consan_moi_exact_shadow::max_epoch, arch));
      sequence.append(
          instrumentation::build_v_mov_b32_literal(tmp_vgpr, consan_moi_exact_shadow::max_epoch,
                                                   arch),
          instrumentation::build_v_cmp_ne_u32_vcc(vector_source_vgpr(token_hash), tmp_vgpr, arch));
      narrow();
      sequence.append(
          instrumentation::build_v_add_u32(token_hash, scalar_positive_inline_u32(1), token_hash,
                                           arch),
          instrumentation::build_v_cmp_gt_u32_vcc(vector_source_vgpr(token_hash), token_epoch,
                                                  arch),
          instrumentation::build_s_andn2_b64(kAmdGpuExecLo, kAmdGpuExecLo, kAmdGpuVccLo, arch));

      // Direct and inherited tokens are committed only after a stable source
      // release has been acquired and the guest RMW has completed. The token
      // is therefore the durable causal fact. Release-sequence tokens carry
      // ancestry between releases and never authorize an ordinary access.
      sequence.append(
          instrumentation::build_v_cmp_gt_u32_vcc(
              scalar_positive_inline_u32(
                  static_cast<uint32_t>(ConSanMoiInlineTokenEvidenceKind::ReleaseSequence)),
              token_kind, arch),
          instrumentation::build_s_andn2_b64(kAmdGpuExecLo, conflict_exec, kAmdGpuVccLo, arch));
    };

    // A conflicting pair is ordered when either access happens-before the
    // other. The LDS exchange is an observation mechanism, not the memory
    // model's clock: a logically later access can become the stored
    // representative before an earlier wave finishes its own exchange.
    // Check both directed acquired-token identities before reporting.
    filter_ordered_pair(current_field_vgpr, prior_owner, current_value_vgpr, old_value_vgpr);
    filter_ordered_pair(prior_owner, current_field_vgpr, old_value_vgpr, current_value_vgpr);
  }

  // Writes do not have the read/read kind predicate, and atomic ordering can
  // remove the last candidate after all of the common filters. Cover both
  // cases with one final gate before constructing a diagnostic record.
  skip_diagnostic_if_empty();

  // The first-diagnostic claim uses SLOT as both the returned value and the
  // low half of the compare-swap data tuple. Keep it in the dead tail of the
  // ordinary transaction window on every target: exact-range construction
  // reuses current_field_vgpr, so aliasing SLOT with that temporary redirects
  // later record stores through a byte offset or negative range endpoint.
  // Targets with an aligned FLAT compare-swap tuple round the candidate down
  // to the preceding legal register boundary.
  constexpr uint16_t diagnostic_tuple_offset =
      consan_detail::inline_shadow_transaction_scratch_count(/*has_exec_save=*/true,
                                                             /*track_atomics=*/false) -
      2u;
  const uint16_t diagnostic_tuple_candidate =
      static_cast<uint16_t>(plan.scratch_vgpr + diagnostic_tuple_offset);
  const uint16_t slot_vgpr = static_cast<uint16_t>(
      diagnostic_tuple_candidate -
      diagnostic_tuple_candidate % target->flat_compare_swap_data_pair_alignment);
  sequence.require(append_select_first_active_lane(
      words, tmp_vgpr, static_cast<uint16_t>(*plan.scalar_state.exec_save_sgpr + 2u),
      static_cast<uint16_t>(*plan.scalar_state.exec_save_sgpr + 4u), arch));

  if (capture_first_diagnostic_only) {
    // A concurrency fault can make the same static conflict execute in every
    // affected workgroup. Preserve one complete, actionable first-fault
    // record instead of turning those repetitions into report overflow. The
    // returned old count is also the slot index: only the 0 -> 1 winner keeps
    // EXEC and writes slot zero.
    const uint64_t count_address = report_base + offsetof(ConSanMoiReportHeader, diagnostic_count);
    const uint32_t mov_claimed =
        build_v_mov_b32_e32(slot_vgpr, scalar_positive_inline_u32(1), arch);
    const uint32_t mov_unclaimed = build_v_mov_b32_e32(static_cast<uint16_t>(slot_vgpr + 1u),
                                                       scalar_positive_inline_u32(0), arch);
    sequence
        .append(instrumentation::build_v_mov_b32_literal(
                    plan.scratch_vgpr, static_cast<uint32_t>(count_address), arch),
                instrumentation::build_v_mov_b32_literal(
                    static_cast<uint16_t>(plan.scratch_vgpr + 1u),
                    static_cast<uint32_t>(count_address >> 32u), arch),
                mov_claimed, mov_unclaimed,
                instrumentation::build_flat_atomic_cmpswap_b32(plan.scratch_vgpr, slot_vgpr,
                                                               slot_vgpr, /*return_old_value=*/true,
                                                               kAmdGpuScopeDevice, arch))
        .require(append_moi_global_atomic_wait(words, arch))
        .append(
            instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), slot_vgpr, arch),
            instrumentation::build_s_and_saveexec_b64(
                static_cast<uint16_t>(*plan.scalar_state.exec_save_sgpr + 4u), kAmdGpuVccLo, arch));
  } else {
    sequence
        .require(append_reserve_bounded_dynamic_record_slot(
            words, report_base + offsetof(ConSanMoiReportHeader, diagnostic_count),
            layout.diagnostic_capacity, slot_vgpr, tmp_vgpr, plan.scratch_vgpr, arch))
        .append(instrumentation::build_s_and_saveexec_b64(
            static_cast<uint16_t>(*plan.scalar_state.exec_save_sgpr + 4u), kAmdGpuVccLo, arch));
  }

  // SLOT is dead once capacity admission has selected this record. Form its
  // address once and keep it in a pair that every following field store can
  // use with a static offset. The former external-shadow path rebuilt
  // base+slot*stride for every field, replicating substantial address
  // arithmetic in every access probe even though all stores target one cold
  // diagnostic record.
  const uint16_t diagnostic_address_vgpr = plan.scratch_vgpr;
  sequence.require(append_dynamic_diagnostic_record_address(words, diagnostic_base, slot_vgpr,
                                                            diagnostic_address_vgpr, arch));

  const auto store_literal = [&](size_t offset, uint32_t value) {
    if (value <= 64u) {
      sequence.append(build_v_mov_b32_e32(tmp_vgpr, scalar_positive_inline_u32(value), arch));
    } else {
      sequence.append(instrumentation::build_v_mov_b32_literal(tmp_vgpr, value, arch));
    }
    sequence.append(instrumentation::build_flat_store_b32(diagnostic_address_vgpr, tmp_vgpr, arch,
                                                          static_cast<uint32_t>(offset)));
  };
  const auto store_vgpr = [&](size_t offset, uint16_t value_vgpr) {
    sequence.append(instrumentation::build_flat_store_b32(diagnostic_address_vgpr, value_vgpr, arch,
                                                          static_cast<uint32_t>(offset)));
  };
  const auto store_scalar = [&](size_t offset, uint16_t scalar_src) {
    sequence.append(build_v_mov_b32_e32(tmp_vgpr, scalar_src, arch),
                    instrumentation::build_flat_store_b32(diagnostic_address_vgpr, tmp_vgpr, arch,
                                                          static_cast<uint32_t>(offset)));
  };
  if (plan.track_atomics && plan.dispatch_id.sgpr) {
    store_scalar(offsetof(ConSanMoiDiagnosticRecord, generation), *plan.dispatch_id.sgpr);
    store_scalar(offsetof(ConSanMoiDiagnosticRecord, generation) + sizeof(uint32_t),
                 static_cast<uint16_t>(*plan.dispatch_id.sgpr + 1u));
  } else if (plan.track_atomics && plan.dispatch_id.vgpr) {
    store_vgpr(offsetof(ConSanMoiDiagnosticRecord, generation), *plan.dispatch_id.vgpr);
    store_vgpr(offsetof(ConSanMoiDiagnosticRecord, generation) + sizeof(uint32_t),
               static_cast<uint16_t>(*plan.dispatch_id.vgpr + 1u));
  } else {
    const uint64_t generation =
        plan.track_atomics ? plan.report_dispatch_id : plan.report_generation;
    store_literal(offsetof(ConSanMoiDiagnosticRecord, generation),
                  static_cast<uint32_t>(generation));
    store_literal(offsetof(ConSanMoiDiagnosticRecord, generation) + sizeof(uint32_t),
                  static_cast<uint32_t>(generation >> 32u));
  }
  store_literal(offsetof(ConSanMoiDiagnosticRecord, kind),
                static_cast<uint32_t>(ConSanMoiDiagnosticKind::AccessConflict));
  store_literal(offsetof(ConSanMoiDiagnosticRecord, backend),
                static_cast<uint32_t>(ConSanMoiEngine::InlineShadow));
  store_literal(offsetof(ConSanMoiDiagnosticRecord, second_instruction_offset),
                static_cast<uint32_t>(candidate.anchor()));
  store_literal(offsetof(ConSanMoiDiagnosticRecord, second_access_kind),
                static_cast<uint32_t>(current_kind));
  store_literal(offsetof(ConSanMoiDiagnosticRecord, first_lane_mask), 0u);
  store_literal(offsetof(ConSanMoiDiagnosticRecord, first_lane_mask) + sizeof(uint32_t), 0u);
  store_scalar(offsetof(ConSanMoiDiagnosticRecord, second_lane_mask),
               diagnostic_lane_mask_sgpr.value_or(
                   static_cast<uint16_t>(*plan.scalar_state.exec_save_sgpr + 2u)));
  store_scalar(
      offsetof(ConSanMoiDiagnosticRecord, second_lane_mask) + sizeof(uint32_t),
      static_cast<uint16_t>(
          diagnostic_lane_mask_sgpr.value_or(*plan.scalar_state.exec_save_sgpr + 2u) + 1u));
  if (plan.track_atomics)
    store_vgpr(offsetof(ConSanMoiDiagnosticRecord, reserved), workgroup_key);

  sequence.require(append_extract_exact_shadow_field(words, tmp_vgpr, old_value_vgpr,
                                                     consan_moi_exact_shadow::owner_shift,
                                                     consan_moi_exact_shadow::max_owner, arch));
  store_vgpr(offsetof(ConSanMoiDiagnosticRecord, first_owner_id), tmp_vgpr);
  sequence.require(append_extract_exact_shadow_field(words, tmp_vgpr, current_value_vgpr,
                                                     consan_moi_exact_shadow::owner_shift,
                                                     consan_moi_exact_shadow::max_owner, arch));
  store_vgpr(offsetof(ConSanMoiDiagnosticRecord, second_owner_id), tmp_vgpr);

  sequence.append(instrumentation::build_v_lshrrev_b32(
      tmp_vgpr, scalar_positive_inline_u32(consan_moi_exact_shadow::instruction_offset_shift - 32u),
      old_value_hi_vgpr, arch));
  store_vgpr(offsetof(ConSanMoiDiagnosticRecord, first_instruction_offset), tmp_vgpr);
  sequence.append(instrumentation::build_v_and_b32_literal(
      tmp_vgpr, static_cast<uint32_t>(consan_moi_exact_shadow::access_kind_mask), old_value_vgpr,
      arch));
  store_vgpr(offsetof(ConSanMoiDiagnosticRecord, first_access_kind), tmp_vgpr);
  if (has_epoch) {
    sequence.require(append_extract_exact_shadow_field(words, tmp_vgpr, old_value_vgpr,
                                                       consan_moi_exact_shadow::epoch_shift,
                                                       consan_moi_exact_shadow::max_epoch, arch));
    store_vgpr(offsetof(ConSanMoiDiagnosticRecord, first_epoch), tmp_vgpr);
    sequence.require(append_extract_exact_shadow_field(words, tmp_vgpr, current_value_vgpr,
                                                       consan_moi_exact_shadow::epoch_shift,
                                                       consan_moi_exact_shadow::max_epoch, arch));
    store_vgpr(offsetof(ConSanMoiDiagnosticRecord, epoch), tmp_vgpr);
  }

  if (prior_byte_provenance_vgpr) {
    if (static_byte_offset != 0u) {
      sequence.require(append_compute_effective_lds_byte_offset(
          words, current_value_vgpr, lds_byte_offset_vgpr, static_byte_offset, arch));
    } else {
      words.push_back(
          build_v_mov_b32_e32(current_value_vgpr, vector_source_vgpr(lds_byte_offset_vgpr), arch));
    }
    sequence.append(instrumentation::build_v_and_b32_literal(
        current_value_vgpr, ~(shadow_granule_bytes - 1u), current_value_vgpr, arch));
    if (relative_cell_index_vgpr) {
      // CDNA materializes the scale literal in the supplied helper VGPR. The
      // The record address was formed once above, so SLOT is dead and can
      // materialize CDNA's scale literal without clobbering that address pair.
      const uint16_t scale_literal_vgpr = slot_vgpr;
      sequence.append(
          instrumentation::build_v_mul_lo_u32_literal(
              tmp_vgpr, scale_literal_vgpr, shadow_granule_bytes, *relative_cell_index_vgpr, arch),
          instrumentation::build_v_add_u32(current_value_vgpr, vector_source_vgpr(tmp_vgpr),
                                           current_value_vgpr, arch));
    } else if (relative_cell_index != 0u) {
      if (relative_cell_index > std::numeric_limits<uint32_t>::max() / shadow_granule_bytes) {
        sequence.require(false);
      } else {
        sequence.append(instrumentation::build_v_add_u32_literal(
            current_value_vgpr, tmp_vgpr, relative_cell_index * shadow_granule_bytes,
            current_value_vgpr, arch));
      }
    }

    const auto store_exact_range = [&](uint16_t provenance_vgpr, size_t offset_field,
                                       size_t count_field) {
      sequence
          .require(append_extract_exact_shadow_field(
              words, current_field_vgpr, provenance_vgpr,
              consan_moi_exact_byte_cell::byte_offset_shift,
              consan_moi_low_bit_mask(consan_moi_exact_byte_cell::byte_offset_bits), arch))
          .append(instrumentation::build_v_add_u32(current_field_vgpr,
                                                   vector_source_vgpr(current_value_vgpr),
                                                   current_field_vgpr, arch));
      store_vgpr(offset_field, current_field_vgpr);
      sequence
          .require(append_extract_exact_shadow_field(
              words, tmp_vgpr, provenance_vgpr,
              consan_moi_exact_byte_cell::byte_end_minus_one_shift,
              consan_moi_low_bit_mask(consan_moi_exact_byte_cell::byte_end_minus_one_bits), arch))
          .append(instrumentation::build_v_add_u32(tmp_vgpr, scalar_positive_inline_u32(1u),
                                                   tmp_vgpr, arch))
          .require(append_extract_exact_shadow_field(
              words, current_field_vgpr, provenance_vgpr,
              consan_moi_exact_byte_cell::byte_offset_shift,
              consan_moi_low_bit_mask(consan_moi_exact_byte_cell::byte_offset_bits), arch));
      // CDNA4 materializes this literal in the supplied helper VGPR. The
      // already-consumed slot index is dead while the precomputed diagnostic
      // address pair must remain intact for all following field stores.
      const uint16_t negative_start_literal_vgpr = slot_vgpr;
      sequence.append(instrumentation::build_v_mul_lo_u32_literal(
                          current_field_vgpr, negative_start_literal_vgpr,
                          std::numeric_limits<uint32_t>::max(), current_field_vgpr, arch),
                      instrumentation::build_v_add_u32(
                          tmp_vgpr, vector_source_vgpr(current_field_vgpr), tmp_vgpr, arch));
      store_vgpr(count_field, tmp_vgpr);
    };
    if (shadow_granule_bytes == 1u) {
      const auto store_prior_original_range = [&]() {
        sequence
            .require(append_extract_exact_shadow_field(
                words, current_field_vgpr, *prior_byte_provenance_vgpr,
                consan_moi_exact_byte_cell::relative_cell_index_shift,
                consan_moi_low_bit_mask(consan_moi_exact_byte_cell::relative_cell_index_bits),
                arch))
            .append(
                instrumentation::build_v_mul_lo_u32_literal(tmp_vgpr, slot_vgpr,
                                                            std::numeric_limits<uint32_t>::max(),
                                                            current_field_vgpr, arch),
                instrumentation::build_v_add_u32(current_value_vgpr, vector_source_vgpr(tmp_vgpr),
                                                 current_value_vgpr, arch));
        store_vgpr(offsetof(ConSanMoiDiagnosticRecord, first_lds_byte_offset), current_value_vgpr);
        sequence
            .require(append_extract_exact_shadow_field(
                words, tmp_vgpr, *prior_byte_provenance_vgpr,
                consan_moi_exact_byte_cell::access_byte_count_minus_one_shift,
                consan_moi_low_bit_mask(
                    consan_moi_exact_byte_cell::access_byte_count_minus_one_bits),
                arch))
            .append(instrumentation::build_v_add_u32(tmp_vgpr, scalar_positive_inline_u32(1u),
                                                     tmp_vgpr, arch));
        store_vgpr(offsetof(ConSanMoiDiagnosticRecord, first_lds_byte_count), tmp_vgpr);
      };
      store_prior_original_range();
      if (static_byte_offset != 0u) {
        sequence.require(append_compute_effective_lds_byte_offset(
            words, current_value_vgpr, lds_byte_offset_vgpr, static_byte_offset, arch));
      } else {
        words.push_back(build_v_mov_b32_e32(current_value_vgpr,
                                            vector_source_vgpr(lds_byte_offset_vgpr), arch));
      }
      store_vgpr(offsetof(ConSanMoiDiagnosticRecord, second_lds_byte_offset), current_value_vgpr);
      store_literal(offsetof(ConSanMoiDiagnosticRecord, second_lds_byte_count), byte_count);
    } else {
      store_exact_range(*prior_byte_provenance_vgpr,
                        offsetof(ConSanMoiDiagnosticRecord, first_lds_byte_offset),
                        offsetof(ConSanMoiDiagnosticRecord, first_lds_byte_count));
      store_exact_range(*current_byte_provenance_vgpr,
                        offsetof(ConSanMoiDiagnosticRecord, second_lds_byte_offset),
                        offsetof(ConSanMoiDiagnosticRecord, second_lds_byte_count));
    }
  } else {
    uint16_t diagnostic_offset_vgpr = lds_byte_offset_vgpr;
    if (static_byte_offset != 0) {
      sequence.require(append_compute_effective_lds_byte_offset(
          words, current_value_vgpr, lds_byte_offset_vgpr, static_byte_offset, arch));
      diagnostic_offset_vgpr = current_value_vgpr;
    }
    store_vgpr(offsetof(ConSanMoiDiagnosticRecord, first_lds_byte_offset), diagnostic_offset_vgpr);
    store_literal(offsetof(ConSanMoiDiagnosticRecord, first_lds_byte_count), byte_count);
    store_vgpr(offsetof(ConSanMoiDiagnosticRecord, second_lds_byte_offset), diagnostic_offset_vgpr);
    store_literal(offsetof(ConSanMoiDiagnosticRecord, second_lds_byte_count), byte_count);
  }

  sequence.bind_label(diagnostic_restore)
      .append(
          instrumentation::build_s_mov_b64(kAmdGpuExecLo, *plan.scalar_state.exec_save_sgpr, arch))
      .require(append_restore_moi_special_state(words, plan.scalar_abi.special_state, arch));
  return sequence.finish(arch);
}

// Publish one exact slot through an odd-version reservation. EXEC on entry is
// the set of lane publishers selected by the caller. EXEC on return contains
// only successfully committed publishers whose prior slot belonged to the
// same full hardware dispatch; those are the only lanes allowed to diagnose.
// Every address group retries bounded reservation contention. Uniform groups
// publish through one representative; the outer partition loop serializes
// metadata-distinct lanes and each representative uses the same retry
// protocol. Every entering lane that still cannot commit increments
// undercoverage exactly once.
[[nodiscard]] bool append_versioned_exact_shadow_transaction(
    std::vector<uint32_t> &words, const MoiInlineShadowEmissionPlan &plan, rj_code_arch_t arch,
    uint16_t address_lo_vgpr, uint16_t current_low_vgpr, uint16_t old_value_vgpr,
    uint16_t publisher_exec_sgpr, uint16_t committed_exec_sgpr, bool retry_contention,
    std::vector<std::string> &errors) {
  InstructionSequence sequence(words);
  MoiStagedEmission require_emission(sequence, errors,
                                     "ConSan MOI versioned exact-shadow transaction");
  const bool has_hardware_dispatch_id = plan.dispatch_id.sgpr || plan.dispatch_id.vgpr;
  if (!plan.scalar_state.exec_save_sgpr ||
      (!has_hardware_dispatch_id && !plan.dispatch_id.literal) ||
      !plan.dispatch_id.is_well_formed())
    return false;

  const uint16_t exec_base = *plan.scalar_state.exec_save_sgpr;
  const uint16_t temporary_exec_sgpr = static_cast<uint16_t>(exec_base + 2u);
  const uint16_t empty_exec_sgpr = exec_base;
  const uint16_t ready_exec_sgpr = temporary_exec_sgpr;
  // +8:+9 retain the guest VCC across the complete probe. Keep the scalar
  // retry counter in the diagnostic-temporary region, which is consumed only
  // after the transaction has finished.
  const uint16_t retry_count_sgpr = static_cast<uint16_t>(exec_base + 20u);
  const uint16_t low_dispatch_exec_sgpr = static_cast<uint16_t>(exec_base + 18u);
  const uint16_t saved_address_lo_vgpr = static_cast<uint16_t>(old_value_vgpr + 2u);
  const uint16_t saved_address_hi_vgpr = static_cast<uint16_t>(old_value_vgpr + 3u);
  const uint16_t saved_current_low_vgpr = static_cast<uint16_t>(old_value_vgpr + 4u);
  const uint16_t saved_current_high_vgpr = static_cast<uint16_t>(old_value_vgpr + 5u);
  const uint16_t prior_dispatch_low_vgpr = static_cast<uint16_t>(old_value_vgpr + 6u);
  const uint16_t prior_dispatch_high_vgpr = static_cast<uint16_t>(old_value_vgpr + 7u);
  const uint16_t ready_version_vgpr = static_cast<uint16_t>(old_value_vgpr + 8u);
  const uint16_t cas_new_vgpr = static_cast<uint16_t>(old_value_vgpr + 9u);
  const uint16_t cas_expected_vgpr = consan_detail::inline_shadow_cas_expected_vgpr(old_value_vgpr);
  const uint16_t tmp_vgpr = static_cast<uint16_t>(plan.scratch_vgpr + 4u);

  MoiPublicationExec exec_masks(sequence, temporary_exec_sgpr, arch);
  const auto set_stage = [&](std::string_view stage) { require_emission.stage(stage); };
  const auto require_equal_literal = [&](uint16_t value_vgpr, uint32_t literal) {
    return exec_masks.require_literal(value_vgpr, literal, true);
  };
  const auto require_not_equal_literal = [&](uint16_t value_vgpr, uint32_t literal) {
    return exec_masks.require_literal(value_vgpr, literal, false);
  };
  const auto restore_slot_address = [&]() {
    sequence.append(
        build_v_mov_b32_e32(address_lo_vgpr, vector_source_vgpr(saved_address_lo_vgpr), arch),
        build_v_mov_b32_e32(static_cast<uint16_t>(address_lo_vgpr + 1u),
                            vector_source_vgpr(saved_address_hi_vgpr), arch));
  };

  const auto append_exact_candidate = [&]() {
    set_stage("slot snapshot");
    // Use the same coherent first read as the other bounded publishers. This
    // gives the common retry loop one uniform entry point at a small fast-path
    // cost and removes exact-shadow's private refresh/re-entry lifecycle.
    sequence
        .require(append_add_literal_field(words, address_lo_vgpr,
                                          offsetof(ConSanMoiInlineExactShadowSlot, version),
                                          tmp_vgpr, arch))
        .require(append_atomic_load_u32(words, address_lo_vgpr, ready_version_vgpr, arch))
        .require(append_add_literal_field(
            words, address_lo_vgpr,
            0u - static_cast<uint32_t>(offsetof(ConSanMoiInlineExactShadowSlot, version)), tmp_vgpr,
            arch))
        .require(append_moi_publication_loads(
            words, address_lo_vgpr,
            {{offsetof(ConSanMoiInlineExactShadowSlot, packed_access), old_value_vgpr},
             {offsetof(ConSanMoiInlineExactShadowSlot, packed_access) + sizeof(uint32_t),
              static_cast<uint16_t>(old_value_vgpr + 1u)},
             {offsetof(ConSanMoiInlineExactShadowSlot, dispatch_id), prior_dispatch_low_vgpr},
             {offsetof(ConSanMoiInlineExactShadowSlot, dispatch_id) + sizeof(uint32_t),
              prior_dispatch_high_vgpr},
             {offsetof(ConSanMoiInlineExactShadowSlot, byte_provenance), tmp_vgpr},
             {offsetof(ConSanMoiInlineExactShadowSlot, version), cas_expected_vgpr}},
            arch));

    // A changed read, odd predecessor, or terminal version is unusable. Keep
    // narrowing instead of branching so a mixed wave can still publish valid
    // independent slots.
    set_stage("payload validation");
    sequence
        .append(instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(ready_version_vgpr),
                                                        cas_expected_vgpr, arch))
        .require(exec_masks.narrow_vcc())
        .append(
            instrumentation::build_v_and_b32_literal(cas_new_vgpr, 1u, ready_version_vgpr, arch))
        .require(require_equal_literal(cas_new_vgpr, 0))
        .append(instrumentation::build_v_mov_b32_literal(
                    cas_new_vgpr, kConSanMoiInlineExactMaxReadyVersion, arch),
                instrumentation::build_v_cmp_ne_u32_vcc(vector_source_vgpr(cas_new_vgpr),
                                                        ready_version_vgpr, arch))
        .require(exec_masks.narrow_vcc())
        .append(instrumentation::build_s_mov_b64(committed_exec_sgpr, kAmdGpuExecLo, arch));

    // Version zero is empty only when every payload word is zero.
    sequence.require(require_equal_literal(ready_version_vgpr, 0))
        .require(require_equal_literal(old_value_vgpr, 0))
        .require(require_equal_literal(static_cast<uint16_t>(old_value_vgpr + 1u), 0))
        .require(require_equal_literal(prior_dispatch_low_vgpr, 0))
        .require(require_equal_literal(prior_dispatch_high_vgpr, 0))
        .require(require_equal_literal(tmp_vgpr, 0))
        .append(instrumentation::build_s_mov_b64(empty_exec_sgpr, kAmdGpuExecLo, arch),
                instrumentation::build_s_mov_b64(kAmdGpuExecLo, committed_exec_sgpr, arch));

    // A nonempty ready predecessor must itself be structurally valid. Owner and
    // epoch zero are legal packed values; access kind, workgroup key, dispatch,
    // and exact-byte provenance are the discriminants.
    sequence.require(require_not_equal_literal(ready_version_vgpr, 0))
        .require(require_not_equal_literal(tmp_vgpr, 0))
        .append(instrumentation::build_v_and_b32_literal(
            cas_new_vgpr, ~consan_moi_exact_byte_cell::packed_mask, tmp_vgpr, arch))
        .require(require_equal_literal(cas_new_vgpr, 0))
        .append(instrumentation::build_v_and_b32_literal(
            cas_new_vgpr, consan_moi_exact_byte_cell::byte_mask_mask, tmp_vgpr, arch));
    for (uint32_t byte_mask = 0; byte_mask <= consan_moi_exact_byte_cell::byte_mask_mask;
         ++byte_mask) {
      if (!consan_moi_exact_byte_cell_mask_is_contiguous(byte_mask))
        sequence.require(require_not_equal_literal(cas_new_vgpr, byte_mask));
    }
    const auto require_canonical_provenance_field = [&](uint32_t field_shift,
                                                        uint32_t field_lookup) {
      sequence
          .append(instrumentation::build_v_and_b32_literal(
                      cas_new_vgpr, consan_moi_exact_byte_cell::byte_mask_mask, tmp_vgpr, arch),
                  instrumentation::build_v_lshlrev_b32(cas_new_vgpr, scalar_positive_inline_u32(1u),
                                                       cas_new_vgpr, arch),
                  instrumentation::build_v_mov_b32_literal(cas_expected_vgpr, field_lookup, arch),
                  instrumentation::build_v_lshrrev_b32(
                      cas_expected_vgpr, vector_source_vgpr(cas_new_vgpr), cas_expected_vgpr, arch),
                  instrumentation::build_v_and_b32_literal(cas_expected_vgpr, 0x3u,
                                                           cas_expected_vgpr, arch),
                  instrumentation::build_v_lshrrev_b32(
                      cas_new_vgpr, scalar_positive_inline_u32(field_shift), tmp_vgpr, arch),
                  instrumentation::build_v_and_b32_literal(cas_new_vgpr, 0x3u, cas_new_vgpr, arch),
                  instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(cas_expected_vgpr),
                                                          cas_new_vgpr, arch))
          .require(exec_masks.narrow_vcc());
    };
    require_canonical_provenance_field(consan_moi_exact_byte_cell::byte_offset_shift,
                                       consan_moi_exact_byte_cell::byte_offset_by_mask_lookup);
    require_canonical_provenance_field(
        consan_moi_exact_byte_cell::byte_end_minus_one_shift,
        consan_moi_exact_byte_cell::byte_end_minus_one_by_mask_lookup);
    sequence
        .append(instrumentation::build_v_and_b32_literal(
            cas_new_vgpr, consan_moi_exact_byte_cell::lane_plus_one_mask, tmp_vgpr, arch))
        .require(require_not_equal_literal(cas_new_vgpr, 0))
        .append(instrumentation::build_v_lshrrev_b32(
                    cas_new_vgpr,
                    scalar_positive_inline_u32(consan_moi_exact_byte_cell::lane_plus_one_shift),
                    cas_new_vgpr, arch),
                instrumentation::build_v_add_u32_literal(cas_new_vgpr, tmp_vgpr,
                                                         std::numeric_limits<uint32_t>::max(),
                                                         cas_new_vgpr, arch),
                instrumentation::build_v_cmp_gt_u32_vcc(
                    scalar_positive_inline_u32(consan_moi_exact_byte_cell::maximum_lane + 1u),
                    cas_new_vgpr, arch))
        .require(exec_masks.narrow_vcc())
        .append(instrumentation::build_v_and_b32_literal(
            tmp_vgpr, static_cast<uint32_t>(consan_moi_exact_shadow::access_kind_mask),
            old_value_vgpr, arch))
        .require(require_not_equal_literal(tmp_vgpr, 0))
        .append(instrumentation::build_v_cmp_gt_u32_vcc(
            scalar_positive_inline_u32(static_cast<uint32_t>(ConSanMoiShadowAccessKind::Atomic) +
                                       1u),
            tmp_vgpr, arch))
        .require(exec_masks.narrow_vcc())
        .require(append_extract_exact_shadow_generation(words, tmp_vgpr, old_value_vgpr,
                                                        static_cast<uint16_t>(old_value_vgpr + 1u),
                                                        cas_new_vgpr, arch))
        .require(require_not_equal_literal(tmp_vgpr, 0))
        .append(instrumentation::build_s_mov_b64(committed_exec_sgpr, kAmdGpuExecLo, arch));

    // Form the nonzero 64-bit dispatch predicate without truncating it: the low
    // nonzero and low-zero/high-nonzero masks are disjoint, so XOR is their union.
    sequence.require(require_not_equal_literal(prior_dispatch_low_vgpr, 0))
        .append(instrumentation::build_s_mov_b64(low_dispatch_exec_sgpr, kAmdGpuExecLo, arch),
                instrumentation::build_s_mov_b64(kAmdGpuExecLo, committed_exec_sgpr, arch))
        .require(require_equal_literal(prior_dispatch_low_vgpr, 0))
        .require(require_not_equal_literal(prior_dispatch_high_vgpr, 0))
        .append(instrumentation::build_s_mov_b64(committed_exec_sgpr, kAmdGpuExecLo, arch),
                instrumentation::build_s_xor_b64(ready_exec_sgpr, low_dispatch_exec_sgpr,
                                                 committed_exec_sgpr, arch),
                instrumentation::build_s_xor_b64(kAmdGpuExecLo, empty_exec_sgpr, ready_exec_sgpr,
                                                 arch));
    return static_cast<bool>(sequence);
  };

  set_stage("reservation");
  require_emission(append_moi_bounded_version_claim(
      words, sequence, exec_masks,
      {.slot_address_vgpr = address_lo_vgpr,
       .eligible_exec_sgpr = publisher_exec_sgpr,
       .claimed_exec_sgpr = committed_exec_sgpr,
       .retry_exec_sgpr = low_dispatch_exec_sgpr,
       .retry_count_sgpr = retry_count_sgpr,
       .base_version_vgpr = ready_version_vgpr,
       .desired_vgpr = cas_new_vgpr,
       .expected_vgpr = cas_expected_vgpr,
       .version_offset = offsetof(ConSanMoiInlineExactShadowSlot, version),
       .retry_limit = retry_contention ? kConSanMoiInlineMetadataPublicationRetryLimit : 0u,
       .sleep_delay = 1u},
      append_exact_candidate, arch));

  // Qualify the predecessor's complete dispatch identity while its payload
  // registers are still live. The resulting scalar mask survives publication;
  // the vector dispatch pair can then retain exact-byte provenance for the
  // diagnostic path without growing the scratch window.
  set_stage("dispatch qualification");
  sequence.append(instrumentation::build_s_mov_b64(committed_exec_sgpr, kAmdGpuExecLo, arch))
      .require(append_compare_moi_report_dispatch_id_word(words, plan.dispatch_id,
                                                          prior_dispatch_low_vgpr, tmp_vgpr,
                                                          /*high_word=*/false, arch))
      .require(exec_masks.narrow_vcc())
      .require(append_compare_moi_report_dispatch_id_word(words, plan.dispatch_id,
                                                          prior_dispatch_high_vgpr, tmp_vgpr,
                                                          /*high_word=*/true, arch))
      .require(exec_masks.narrow_vcc())
      .append(instrumentation::build_s_mov_b64(low_dispatch_exec_sgpr, kAmdGpuExecLo, arch),
              instrumentation::build_s_mov_b64(kAmdGpuExecLo, committed_exec_sgpr, arch));

  restore_slot_address();
  sequence.require(append_load_u32_vgpr_at_offset(
      words, address_lo_vgpr, offsetof(ConSanMoiInlineExactShadowSlot, byte_provenance),
      prior_dispatch_low_vgpr, arch));
  words.push_back(
      build_v_mov_b32_e32(prior_dispatch_high_vgpr, vector_source_vgpr(current_low_vgpr), arch));
  restore_slot_address();

  set_stage("publication");
  sequence
      .require(append_moi_publication_stores(
          words, address_lo_vgpr,
          {{offsetof(ConSanMoiInlineExactShadowSlot, packed_access), saved_current_low_vgpr},
           {offsetof(ConSanMoiInlineExactShadowSlot, packed_access) + sizeof(uint32_t),
            saved_current_high_vgpr},
           {offsetof(ConSanMoiInlineExactShadowSlot, byte_provenance), current_low_vgpr}},
          arch))
      .require(append_moi_report_dispatch_id_pair(words, plan.dispatch_id, current_low_vgpr,
                                                  static_cast<uint16_t>(current_low_vgpr + 1u),
                                                  arch))
      .require(append_moi_publication_stores(
          words, address_lo_vgpr,
          {{offsetof(ConSanMoiInlineExactShadowSlot, dispatch_id), current_low_vgpr},
           {offsetof(ConSanMoiInlineExactShadowSlot, dispatch_id) + sizeof(uint32_t),
            static_cast<uint16_t>(current_low_vgpr + 1u)}},
          arch))
      .append(instrumentation::build_s_wait_global_store0(arch));

  set_stage("commit");
  sequence
      .require(append_moi_version_transition(words, sequence, exec_masks, address_lo_vgpr,
                                             offsetof(ConSanMoiInlineExactShadowSlot, version),
                                             ready_version_vgpr, cas_new_vgpr, cas_expected_vgpr,
                                             /*desired_delta=*/2u, /*expected_delta=*/1u, arch))
      .append(instrumentation::build_s_mov_b64(committed_exec_sgpr, kAmdGpuExecLo, arch));

  // Count successful publications before excluding cross-dispatch priors from
  // diagnosis. The event counter therefore remains a successful-commit count.
  set_stage("accounting");
  sequence.require(append_atomic_fetch_add_one_u32(
      words, plan.report_buffer_address + offsetof(ConSanMoiReportHeader, event_counter), tmp_vgpr,
      address_lo_vgpr, arch));
  restore_slot_address();

  sequence
      .append(instrumentation::build_s_mov_b64(kAmdGpuExecLo, publisher_exec_sgpr, arch),
              instrumentation::build_s_andn2_b64(kAmdGpuExecLo, publisher_exec_sgpr,
                                                 committed_exec_sgpr, arch))
      .require(append_atomic_fetch_add_one_u32(
          words,
          plan.report_buffer_address + offsetof(ConSanMoiReportHeader, inline_undercoverage_count),
          tmp_vgpr, address_lo_vgpr, arch))
      .append(instrumentation::build_s_mov_b64(kAmdGpuExecLo, committed_exec_sgpr, arch));
  restore_slot_address();
  words.push_back(
      build_v_mov_b32_e32(current_low_vgpr, vector_source_vgpr(saved_current_low_vgpr), arch));
  words.push_back(build_v_mov_b32_e32(static_cast<uint16_t>(current_low_vgpr + 1u),
                                      vector_source_vgpr(saved_current_high_vgpr), arch));
  sequence.append(instrumentation::build_s_and_b64(kAmdGpuExecLo, committed_exec_sgpr,
                                                   low_dispatch_exec_sgpr, arch));
  return require_emission.finish(arch);
}

// Build the target-neutral exact-byte provenance for one external shadow
// cell. Four-byte cells retain compact byte masks for ordinary objects;
// mixed-width objects use one-byte cells so adjacent byte addresses never
// overwrite one another. The caller supplies the relative cell index when a
// wide access is traversed by a runtime loop. MASK remains separate for wave
// partitioning; PROVENANCE carries the redundant range endpoints used by
// diagnostics and coherent snapshot validation. The representative lane is
// added only after the caller elects the publishing lane.
[[nodiscard]] bool append_exact_byte_cell_provenance_base(
    std::vector<uint32_t> &words, rj_code_arch_t arch, uint16_t base_offset_vgpr,
    uint32_t static_byte_offset, uint32_t relative_cell_index,
    std::optional<uint16_t> relative_cell_index_vgpr, uint32_t byte_count,
    uint32_t shadow_granule_bytes, uint16_t provenance_vgpr, uint16_t mask_vgpr, uint16_t high_vgpr,
    uint16_t temporary_vgpr) {
  InstructionSequence sequence(words);
  if (byte_count == 0u ||
      (shadow_granule_bytes != 1u && shadow_granule_bytes != consan_moi_shadow_cell::granule_bytes))
    return false;

  if (static_byte_offset != 0u) {
    sequence.append(instrumentation::build_v_add_u32_literal(
        mask_vgpr, provenance_vgpr, static_byte_offset & (shadow_granule_bytes - 1u),
        base_offset_vgpr, arch));
  } else {
    words.push_back(build_v_mov_b32_e32(mask_vgpr, vector_source_vgpr(base_offset_vgpr), arch));
  }
  sequence.append(instrumentation::build_v_and_b32_literal(mask_vgpr, shadow_granule_bytes - 1u,
                                                           mask_vgpr, arch),
                  instrumentation::build_v_add_u32_literal(high_vgpr, temporary_vgpr, byte_count,
                                                           mask_vgpr, arch));

  if (relative_cell_index_vgpr) {
    sequence.append(instrumentation::build_v_mul_lo_u32_literal(
        temporary_vgpr, provenance_vgpr, shadow_granule_bytes, *relative_cell_index_vgpr, arch));
    // The loop bound covers the worst runtime alignment. Clamp an iteration's
    // cell start to the access end before subtraction so favorable alignments
    // produce an empty mask instead of wrapping to a phantom full cell.
    sequence.append(instrumentation::build_v_min_u32(temporary_vgpr, vector_source_vgpr(high_vgpr),
                                                     temporary_vgpr, arch),
                    instrumentation::build_v_mul_lo_u32_literal(
                        temporary_vgpr, provenance_vgpr, std::numeric_limits<uint32_t>::max(),
                        temporary_vgpr, arch),
                    instrumentation::build_v_add_u32(high_vgpr, vector_source_vgpr(temporary_vgpr),
                                                     high_vgpr, arch),
                    instrumentation::build_v_min_u32_literal(temporary_vgpr, 1u,
                                                             *relative_cell_index_vgpr, arch),
                    instrumentation::build_v_lshlrev_b32(
                        temporary_vgpr, scalar_positive_inline_u32(1u), temporary_vgpr, arch),
                    instrumentation::build_v_lshrrev_b32(
                        temporary_vgpr, vector_source_vgpr(temporary_vgpr), mask_vgpr, arch));
  } else if (relative_cell_index != 0u) {
    if (relative_cell_index > std::numeric_limits<uint32_t>::max() / shadow_granule_bytes) {
      return false;
    }
    sequence.append(instrumentation::build_v_mov_b32_literal(
                        temporary_vgpr, relative_cell_index * shadow_granule_bytes, arch),
                    instrumentation::build_v_min_u32(temporary_vgpr, vector_source_vgpr(high_vgpr),
                                                     temporary_vgpr, arch),
                    instrumentation::build_v_mul_lo_u32_literal(
                        temporary_vgpr, provenance_vgpr, std::numeric_limits<uint32_t>::max(),
                        temporary_vgpr, arch),
                    instrumentation::build_v_add_u32(high_vgpr, vector_source_vgpr(temporary_vgpr),
                                                     high_vgpr, arch),
                    build_v_mov_b32_e32(temporary_vgpr, scalar_positive_inline_u32(0u), arch));
  } else {
    words.push_back(build_v_mov_b32_e32(temporary_vgpr, vector_source_vgpr(mask_vgpr), arch));
  }

  sequence.append(
      instrumentation::build_v_min_u32_literal(high_vgpr, shadow_granule_bytes, high_vgpr, arch),
      instrumentation::build_v_mov_b32_literal(provenance_vgpr, 1u, arch),
      instrumentation::build_v_lshlrev_b32(provenance_vgpr, vector_source_vgpr(high_vgpr),
                                           provenance_vgpr, arch),
      instrumentation::build_v_add_u32_literal(
          provenance_vgpr, mask_vgpr, std::numeric_limits<uint32_t>::max(), provenance_vgpr, arch),
      instrumentation::build_v_mov_b32_literal(mask_vgpr, (1u << shadow_granule_bytes) - 1u, arch),
      instrumentation::build_v_lshlrev_b32(mask_vgpr, vector_source_vgpr(temporary_vgpr), mask_vgpr,
                                           arch),
      instrumentation::build_v_and_b32(mask_vgpr, vector_source_vgpr(provenance_vgpr), mask_vgpr,
                                       arch));

  sequence.append(
      instrumentation::build_v_add_u32_literal(
          high_vgpr, provenance_vgpr, std::numeric_limits<uint32_t>::max(), high_vgpr, arch),
      instrumentation::build_v_lshlrev_b32(
          high_vgpr,
          scalar_positive_inline_u32(consan_moi_exact_byte_cell::byte_end_minus_one_shift),
          high_vgpr, arch),
      instrumentation::build_v_lshlrev_b32(
          temporary_vgpr, scalar_positive_inline_u32(consan_moi_exact_byte_cell::byte_offset_shift),
          temporary_vgpr, arch),
      instrumentation::build_v_add_u32(provenance_vgpr, vector_source_vgpr(high_vgpr),
                                       temporary_vgpr, arch),
      instrumentation::build_v_add_u32(provenance_vgpr, vector_source_vgpr(mask_vgpr),
                                       provenance_vgpr, arch));

  // A byte-granular external slot represents only one byte of a potentially
  // wider access. Retain the original width and this slot's relative byte
  // index so a later conflicting access can report both complete source
  // ranges, rather than reducing the wider side to their one-byte overlap.
  if (shadow_granule_bytes == 1u) {
    if (byte_count > consan_moi_exact_byte_cell::maximum_access_byte_count ||
        relative_cell_index > consan_moi_exact_byte_cell::maximum_relative_cell_index) {
      return false;
    }
    sequence.append(instrumentation::build_v_add_u32_literal(
        provenance_vgpr, temporary_vgpr,
        (byte_count - 1u) << consan_moi_exact_byte_cell::access_byte_count_minus_one_shift,
        provenance_vgpr, arch));
    if (relative_cell_index_vgpr) {
      sequence.append(
          instrumentation::build_v_lshlrev_b32(
              temporary_vgpr,
              scalar_positive_inline_u32(consan_moi_exact_byte_cell::relative_cell_index_shift),
              *relative_cell_index_vgpr, arch),
          instrumentation::build_v_add_u32(provenance_vgpr, vector_source_vgpr(temporary_vgpr),
                                           provenance_vgpr, arch));
    } else if (relative_cell_index != 0u) {
      sequence.append(instrumentation::build_v_add_u32_literal(
          provenance_vgpr, temporary_vgpr,
          relative_cell_index << consan_moi_exact_byte_cell::relative_cell_index_shift,
          provenance_vgpr, arch));
    }
  }
  return sequence.finish();
}

[[nodiscard]] bool append_exact_byte_representative_lane(std::vector<uint32_t> &words,
                                                         rj_code_arch_t arch,
                                                         uint16_t provenance_vgpr,
                                                         uint16_t lane_vgpr) {
  InstructionSequence sequence(words);
  return sequence.emit_all(
      instrumentation::build_v_mbcnt_lo_u32_b32(lane_vgpr, kScalarInlineNegativeOneOperand,
                                                scalar_positive_inline_u32(0), arch),
      instrumentation::build_v_mbcnt_hi_u32_b32(lane_vgpr, kScalarInlineNegativeOneOperand,
                                                vector_source_vgpr(lane_vgpr), arch),
      instrumentation::build_v_add_u32(lane_vgpr, scalar_positive_inline_u32(1u), lane_vgpr, arch),
      instrumentation::build_v_lshlrev_b32(
          lane_vgpr, scalar_positive_inline_u32(consan_moi_exact_byte_cell::lane_plus_one_shift),
          lane_vgpr, arch),
      instrumentation::build_v_add_u32(provenance_vgpr, vector_source_vgpr(lane_vgpr),
                                       provenance_vgpr, arch));
}

// Partition the incoming wave by effective exact-shadow address. Each address
// group with uniform metadata and exact byte coverage publishes through one
// representative lane. A group whose metadata is not uniform submits every
// lane to the versioned transaction rather than selecting arbitrary evidence;
// one CAS winner may publish and every loser poisons coverage. The loop removes
// at least one lane on every iteration, so runtime work is bounded by the
// hardware wave size while generated code size remains constant.
[[nodiscard]] bool append_wave_coalesced_exact_shadow_swap(
    std::vector<uint32_t> &words, const ConSanMoiCandidate &candidate,
    const MoiInlineShadowEmissionPlan &plan, rj_code_arch_t arch,
    const ConSanMoiReportBufferLayout &layout, uint16_t address_lo_vgpr, uint16_t current_low_vgpr,
    uint16_t current_high_vgpr, uint16_t old_value_vgpr, uint16_t lds_byte_offset_vgpr,
    uint32_t static_byte_offset, uint32_t byte_count, uint32_t shadow_granule_bytes,
    uint32_t relative_cell_index, std::optional<uint16_t> relative_cell_index_vgpr,
    const VgprSpillSequence *spill, std::optional<uint16_t> spill_backed_lds_byte_offset_source,
    std::vector<std::string> &errors) {
  InstructionSequence sequence(words);
  MoiStagedEmission require_emission(sequence, errors,
                                     "ConSan MOI wave-coalesced exact-shadow swap");
  if (!plan.scalar_state.exec_save_sgpr)
    return false;

  const auto set_stage = [&](std::string_view stage) { require_emission.stage(stage); };

  const uint16_t base = *plan.scalar_state.exec_save_sgpr;
  const uint16_t temporary_exec_sgpr = static_cast<uint16_t>(base + 2u);
  const uint16_t publisher_exec_sgpr = static_cast<uint16_t>(base + 4u);
  const uint16_t committed_exec_sgpr = static_cast<uint16_t>(base + 6u);
  const uint16_t incoming_exec_sgpr = static_cast<uint16_t>(base + 12u);
  const uint16_t pending_exec_sgpr = static_cast<uint16_t>(base + 14u);
  const uint16_t group_exec_sgpr = static_cast<uint16_t>(base + 16u);
  const uint16_t address_key_sgpr = static_cast<uint16_t>(base + 18u);
  const uint16_t value_key_sgpr = static_cast<uint16_t>(base + 19u);
  const uint16_t lane_rank_vgpr = static_cast<uint16_t>(old_value_vgpr + 1u);
  const uint16_t saved_address_lo_vgpr = static_cast<uint16_t>(old_value_vgpr + 2u);
  const uint16_t saved_address_hi_vgpr = static_cast<uint16_t>(old_value_vgpr + 3u);
  const uint16_t saved_current_low_vgpr = static_cast<uint16_t>(old_value_vgpr + 4u);
  const uint16_t saved_current_high_vgpr = static_cast<uint16_t>(old_value_vgpr + 5u);

  set_stage("partition setup");
  const uint32_t save_address_lo =
      build_v_mov_b32_e32(saved_address_lo_vgpr, vector_source_vgpr(address_lo_vgpr), arch);
  const uint32_t save_address_hi = build_v_mov_b32_e32(
      saved_address_hi_vgpr, vector_source_vgpr(static_cast<uint16_t>(address_lo_vgpr + 1u)), arch);
  const uint32_t save_current_low =
      build_v_mov_b32_e32(saved_current_low_vgpr, vector_source_vgpr(current_low_vgpr), arch);
  const uint32_t save_current_high =
      build_v_mov_b32_e32(saved_current_high_vgpr, vector_source_vgpr(current_high_vgpr), arch);
  const uint32_t restore_address_lo =
      build_v_mov_b32_e32(address_lo_vgpr, vector_source_vgpr(saved_address_lo_vgpr), arch);
  const uint32_t restore_address_hi = build_v_mov_b32_e32(
      static_cast<uint16_t>(address_lo_vgpr + 1u), vector_source_vgpr(saved_address_hi_vgpr), arch);
  const uint32_t restore_current_low =
      build_v_mov_b32_e32(current_low_vgpr, vector_source_vgpr(saved_current_low_vgpr), arch);
  const uint32_t restore_current_high =
      build_v_mov_b32_e32(current_high_vgpr, vector_source_vgpr(saved_current_high_vgpr), arch);
  // Retain validation of the high-word read supported by this ABI even though
  // the current partition key is the low word.
  const auto read_address_high = instrumentation::build_v_readfirstlane_b32(
      value_key_sgpr, static_cast<uint16_t>(address_lo_vgpr + 1u), arch);
  sequence.append(read_address_high,
                  instrumentation::build_s_mov_b64(incoming_exec_sgpr, kAmdGpuExecLo, arch),
                  instrumentation::build_s_mov_b64(pending_exec_sgpr, kAmdGpuExecLo, arch),
                  save_address_lo, save_address_hi, save_current_low, save_current_high);

  const InstructionSequence::Label loop = sequence.mark_label();
  sequence.append(
      instrumentation::build_s_mov_b64(kAmdGpuExecLo, pending_exec_sgpr, arch), restore_address_lo,
      restore_address_hi, restore_current_low, restore_current_high,
      instrumentation::build_v_readfirstlane_b32(address_key_sgpr, address_lo_vgpr, arch),
      instrumentation::build_v_readfirstlane_b32(value_key_sgpr, current_low_vgpr, arch),
      instrumentation::build_v_cmp_eq_u32_vcc(address_key_sgpr, address_lo_vgpr, arch),
      instrumentation::build_s_and_saveexec_b64(temporary_exec_sgpr, kAmdGpuVccLo, arch),
      instrumentation::build_v_cmp_eq_u32_vcc(value_key_sgpr, current_low_vgpr, arch),
      instrumentation::build_s_and_saveexec_b64(temporary_exec_sgpr, kAmdGpuVccLo, arch));
  set_stage("composite-key provenance");
  require_emission(append_exact_byte_cell_provenance_base(
                       words, arch, lds_byte_offset_vgpr, static_byte_offset, relative_cell_index,
                       relative_cell_index_vgpr, byte_count, shadow_granule_bytes, current_low_vgpr,
                       lane_rank_vgpr, current_high_vgpr,
                       static_cast<uint16_t>(plan.scratch_vgpr + 4u)),
                   "ConSan MOI inline-shadow probe could not encode exact byte-cell provenance");
  sequence.append(
      instrumentation::build_v_readfirstlane_b32(value_key_sgpr, lane_rank_vgpr, arch),
      instrumentation::build_v_cmp_eq_u32_vcc(value_key_sgpr, lane_rank_vgpr, arch),
      instrumentation::build_s_and_saveexec_b64(temporary_exec_sgpr, kAmdGpuVccLo, arch),
      instrumentation::build_s_mov_b64(group_exec_sgpr, kAmdGpuExecLo, arch));
  // Partition pending lanes by their complete publication key. Address and
  // packed access were selected above; exact-byte provenance contributes the
  // final mask component here. Saving that subgroup directly avoids emitting
  // a second copy of the complete versioned transaction for a nonuniform
  // fallback. Distinct metadata simply remains in pending EXEC and is visited
  // by the next bounded loop iteration.
  const auto use_group_diagnostic_mask =
      instrumentation::build_s_mov_b64(publisher_exec_sgpr, group_exec_sgpr, arch);
  sequence
      .append(instrumentation::build_v_cmp_ne_u32_vcc(scalar_positive_inline_u32(0u),
                                                      lane_rank_vgpr, arch),
              instrumentation::build_s_and_saveexec_b64(temporary_exec_sgpr, kAmdGpuVccLo, arch),
              use_group_diagnostic_mask)
      .require(append_select_first_lane_in_exec_mask(words, lane_rank_vgpr, group_exec_sgpr,
                                                     temporary_exec_sgpr, arch))
      .append(instrumentation::build_s_mov_b64(publisher_exec_sgpr, kAmdGpuExecLo, arch));
  sequence.require(
      append_exact_byte_representative_lane(words, arch, current_low_vgpr, current_high_vgpr));
  set_stage("composite-key transaction");
  sequence.require(append_versioned_exact_shadow_transaction(
      words, plan, arch, address_lo_vgpr, current_low_vgpr, old_value_vgpr, publisher_exec_sgpr,
      committed_exec_sgpr, /*retry_contention=*/true, errors));
  // The representative stands for the complete metadata-identical subgroup.
  // Retain that exact subgroup as the diagnostic lane mask.
  sequence.append(use_group_diagnostic_mask);

  // Counter updates and version-address CAS operations use the working address
  // pair. Restore it from the loop invariants before diagnostics and the next
  // partition iteration.
  words.push_back(restore_address_lo);
  words.push_back(restore_address_hi);
  if (!sequence)
    return false;
  // Narrow spill-backed paths deliberately pay one reload per partition only
  // when diagnostics can consume the recovered guest address.
  if (layout.diagnostic_capacity != 0 && spill_backed_lds_byte_offset_source) {
    if (spill == nullptr) {
      errors.emplace_back(
          "ConSan MOI inline-shadow probe has no spill plan for its diagnostic LDS address");
      return false;
    }
    const auto reload = consan_detail::append_reload_moi_spilled_vgpr(
        words, *spill, lds_byte_offset_vgpr, *spill_backed_lds_byte_offset_source, arch);
    if (reload != consan_detail::MoiSpilledVgprReloadResult::Appended) {
      errors.emplace_back(
          "ConSan MOI inline-shadow probe could not recover its diagnostic LDS address: " +
          std::string(consan_detail::moi_spilled_vgpr_reload_result_name(reload)));
      return false;
    }
  }

  // The partition operations changed VCC/SCC. Restore the incoming values so
  // the existing diagnostic helper can snapshot and restore the true program
  // state while evaluating every lane in this address group.
  set_stage("pre-diagnostic state restoration");
  sequence.require(append_restore_moi_special_state(words, plan.scalar_abi.special_state, arch));
  set_stage("diagnostic encoding");
  sequence.require(append_inline_shadow_diagnostic_words(
      words, candidate, plan, arch, layout, old_value_vgpr,
      static_cast<uint16_t>(old_value_vgpr + 1u), current_low_vgpr, current_high_vgpr,
      lds_byte_offset_vgpr, static_byte_offset, byte_count, shadow_granule_bytes,
      /*special_state_already_saved=*/true,
      /*diagnostic_lane_mask_sgpr=*/publisher_exec_sgpr,
      /*capture_first_diagnostic_only=*/false,
      /*workgroup_local_shadow=*/false,
      /*prior_byte_provenance_vgpr=*/
      static_cast<uint16_t>(old_value_vgpr + 6u),
      /*current_byte_provenance_vgpr=*/
      static_cast<uint16_t>(old_value_vgpr + 7u), relative_cell_index_vgpr, relative_cell_index));

  // The selected group is a subset of pending EXEC, so XOR removes exactly
  // that group. Keep this independent of the RDNA4 AND-NOT operand convention;
  // live qualification is the authority for this generated control flow.
  set_stage("loop completion");
  sequence.append(instrumentation::build_s_xor_b64(pending_exec_sgpr, pending_exec_sgpr,
                                                   group_exec_sgpr, arch));
  // Install the remaining mask into EXEC and branch on it. Live gfx1201 debug
  // telemetry proved this traversal reaches the final divergent lane when
  // there is scalar distance after the pending-mask XOR. Keep that dependency
  // explicit: without it, both an immediate EXEC move and an immediate scalar
  // compare observed only the first address group in acceptance runs.
  sequence
      .append(build_s_nop(0, arch), build_s_nop(0, arch),
              instrumentation::build_s_mov_b64(kAmdGpuExecLo, pending_exec_sgpr, arch))
      .branch(loop, InstructionSequence::BranchKind::ExecNonzero)
      .append(instrumentation::build_s_mov_b64(kAmdGpuExecLo, incoming_exec_sgpr, arch));
  // The diagnostic path uses current_high_vgpr as a field temporary. Restore
  // both packed words before returning so a following cell of the same wide
  // access starts from the original owner/epoch/workgroup metadata.
  words.push_back(restore_current_low);
  words.push_back(restore_current_high);
  sequence.require(append_restore_moi_special_state(words, plan.scalar_abi.special_state, arch));

  return require_emission.finish(arch);
}

// Make one packed-validity local slot ready without eagerly clearing the full
// exact mirror. Atomic OR claims the cell's initializing bit. The winner
// zeroes the exact slot before publishing its ready bit; every participating
// lane then polls that ready bit before the normal exact exchange. Two bits
// per slot allow neighboring cells to share one atomic state word safely.
[[nodiscard]] bool append_workgroup_local_lazy_shadow_ready(
    std::vector<uint32_t> &words, const MoiInlineShadowEmissionPlan &plan, rj_code_arch_t arch,
    const ConSanMoiWorkgroupShadowLayout &workgroup_shadow, uint16_t saved_cell_vgpr,
    std::vector<std::string> &errors) {
  if (!workgroup_shadow.lazy_initialization)
    return true;
  const ConSanTargetProfile *target = consan_target_profile(arch);
  if (!target || !target->moi_access.lazy_workgroup_shadow || !plan.scalar_state.exec_save_sgpr ||
      workgroup_shadow.validity_size == 0u) {
    errors.emplace_back(
        "ConSan MOI lazy local shadow requires target-supported packed validity state");
    return false;
  }

  const uint16_t scratch = plan.scratch_vgpr;
  const uint16_t address_vgpr = scratch;
  const uint16_t temporary_vgpr = static_cast<uint16_t>(scratch + 4u);
  const uint16_t state_vgpr = static_cast<uint16_t>(scratch + 5u);
  const uint16_t zero_vgpr = static_cast<uint16_t>(scratch + 6u);
  const uint16_t initializing_mask_vgpr = static_cast<uint16_t>(scratch + 8u);
  const uint16_t ready_mask_vgpr = static_cast<uint16_t>(scratch + 11u);
  const uint16_t validity_address_vgpr = static_cast<uint16_t>(scratch + 12u);
  const uint16_t temporary_exec_sgpr =
      static_cast<uint16_t>(*plan.scalar_state.exec_save_sgpr + 14u);
  const uint16_t bounded_exec_sgpr = static_cast<uint16_t>(*plan.scalar_state.exec_save_sgpr + 16u);
  InstructionSequence sequence(words);
  const auto poll = sequence.make_label();
  const auto restore_bounded = sequence.make_label();
  sequence
      .append(instrumentation::build_v_lshrrev_b32(temporary_vgpr, scalar_positive_inline_u32(4u),
                                                   saved_cell_vgpr, arch),
              instrumentation::build_v_lshlrev_b32(temporary_vgpr, scalar_positive_inline_u32(2u),
                                                   temporary_vgpr, arch),
              instrumentation::build_v_add_u32_literal(
                  validity_address_vgpr, workgroup_shadow.validity_base, temporary_vgpr, arch),
              build_v_mov_b32_e32(ready_mask_vgpr, scalar_positive_inline_u32(1u), arch),
              instrumentation::build_v_and_b32_literal(initializing_mask_vgpr, 15u, saved_cell_vgpr,
                                                       arch),
              instrumentation::build_v_lshlrev_b32(initializing_mask_vgpr,
                                                   scalar_positive_inline_u32(1u),
                                                   initializing_mask_vgpr, arch),
              instrumentation::build_v_lshlrev_b32(initializing_mask_vgpr,
                                                   vector_source_vgpr(initializing_mask_vgpr),
                                                   ready_mask_vgpr, arch),
              instrumentation::build_v_lshlrev_b32(ready_mask_vgpr, scalar_positive_inline_u32(1u),
                                                   initializing_mask_vgpr, arch),
              instrumentation::build_ds_load_b32(state_vgpr, validity_address_vgpr, 0u, arch))
      .require(append_moi_lds_wait(words, arch))
      .append(instrumentation::build_v_and_b32(temporary_vgpr, vector_source_vgpr(ready_mask_vgpr),
                                               state_vgpr, arch),
              instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(ready_mask_vgpr),
                                                      temporary_vgpr, arch),
              instrumentation::build_s_andn2_b64(kAmdGpuExecLo, kAmdGpuExecLo, kAmdGpuVccLo, arch))
      .branch(restore_bounded, InstructionSequence::BranchKind::ExecZero)
      .append(instrumentation::build_ds_or_rtn_b32(state_vgpr, validity_address_vgpr,
                                                   initializing_mask_vgpr, 0u, arch))
      .require(append_moi_lds_wait(words, arch))
      .append(instrumentation::build_v_and_b32(
                  temporary_vgpr, vector_source_vgpr(initializing_mask_vgpr), state_vgpr, arch),
              instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0u),
                                                      temporary_vgpr, arch),
              instrumentation::build_s_and_saveexec_b64(temporary_exec_sgpr, kAmdGpuVccLo, arch),
              build_v_mov_b32_e32(state_vgpr, scalar_positive_inline_u32(0u), arch),
              build_v_mov_b32_e32(zero_vgpr, scalar_positive_inline_u32(0u), arch),
              instrumentation::build_v_lshlrev_b32(temporary_vgpr, scalar_positive_inline_u32(3u),
                                                   saved_cell_vgpr, arch),
              instrumentation::build_v_add_u32_literal(address_vgpr, workgroup_shadow.base,
                                                       temporary_vgpr, arch),
              instrumentation::build_ds_store_b64(address_vgpr, state_vgpr, 0u, arch))
      .require(append_moi_lds_wait(words, arch))
      .append(instrumentation::build_ds_or_rtn_b32(state_vgpr, validity_address_vgpr,
                                                   ready_mask_vgpr, 0u, arch))
      .require(append_moi_lds_wait(words, arch))
      .append(instrumentation::build_s_mov_b64(kAmdGpuExecLo, temporary_exec_sgpr, arch))
      .bind_label(poll)
      .append(instrumentation::build_ds_load_b32(state_vgpr, validity_address_vgpr, 0u, arch))
      .require(append_moi_lds_wait(words, arch))
      .append(instrumentation::build_v_and_b32(temporary_vgpr, vector_source_vgpr(ready_mask_vgpr),
                                               state_vgpr, arch),
              instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(ready_mask_vgpr),
                                                      temporary_vgpr, arch),
              instrumentation::build_s_andn2_b64(kAmdGpuExecLo, kAmdGpuExecLo, kAmdGpuVccLo, arch))
      .branch(poll, InstructionSequence::BranchKind::ExecNonzero)
      .bind_label(restore_bounded)
      .append(instrumentation::build_s_mov_b64(kAmdGpuExecLo, bounded_exec_sgpr, arch));
  if (!sequence.finish(arch)) {
    errors.emplace_back(
        "ConSan MOI lazy local shadow could not encode packed RDNA4 state transitions");
    return false;
  }
  return true;
}

// Publish one exact metadata entry into the bounded shadow interval reserved
// in the owning kernel's LDS allocation. Unlike the global versioned table,
// a DS exchange is naturally scoped to one workgroup and atomically returns
// the complete prior 64-bit entry, so it needs neither dispatch qualification
// nor a multi-instruction reservation protocol.
[[nodiscard]] bool append_workgroup_local_exact_shadow_swap(
    std::vector<uint32_t> &words, const ConSanMoiCandidate &candidate,
    const MoiInlineShadowEmissionPlan &plan, rj_code_arch_t arch,
    const ConSanMoiReportBufferLayout &report_layout,
    const ConSanMoiWorkgroupShadowLayout &workgroup_shadow, uint16_t address_vgpr,
    uint16_t current_low_vgpr, uint16_t current_high_vgpr, uint16_t old_value_vgpr,
    uint16_t lds_byte_offset_vgpr, uint32_t static_byte_offset, uint32_t cell_index,
    uint32_t byte_count, uint16_t diagnostic_lds_byte_offset_vgpr,
    uint32_t diagnostic_static_byte_offset, std::optional<uint16_t> relative_cell_index_vgpr,
    std::vector<std::string> &errors) {
  if (!plan.scalar_state.exec_save_sgpr)
    return false;

  const uint16_t temporary_vgpr = static_cast<uint16_t>(plan.scratch_vgpr + 4u);
  const uint16_t saved_current_low_vgpr = static_cast<uint16_t>(old_value_vgpr + 4u);
  const uint16_t saved_current_high_vgpr = static_cast<uint16_t>(old_value_vgpr + 5u);
  const uint16_t saved_cell_vgpr = static_cast<uint16_t>(old_value_vgpr + 2u);
  const uint16_t exec_base = *plan.scalar_state.exec_save_sgpr;
  const uint16_t incoming_exec_sgpr = static_cast<uint16_t>(exec_base + 12u);
  const uint16_t temporary_exec_sgpr = static_cast<uint16_t>(exec_base + 14u);
  const uint16_t bounded_exec_sgpr = static_cast<uint16_t>(exec_base + 16u);
  const uint16_t valid_workgroup_exec_sgpr = static_cast<uint16_t>(exec_base + 22u);
  const uint16_t provenance_temporary_vgpr = static_cast<uint16_t>(old_value_vgpr + 3u);
  InstructionSequence sequence(words);
  MoiEmissionRequirement require_emission(sequence, errors);
  const auto restore_valid_workgroup =
      instrumentation::build_s_mov_b64(kAmdGpuExecLo, valid_workgroup_exec_sgpr, arch);
  require_emission(restore_valid_workgroup.has_value(),
                   "ConSan MOI inline-shadow local publish could not restore its workgroup lanes");

  sequence.append(
      build_v_mov_b32_e32(saved_current_low_vgpr, vector_source_vgpr(current_low_vgpr), arch),
      build_v_mov_b32_e32(saved_current_high_vgpr, vector_source_vgpr(current_high_vgpr), arch));
  require_emission(append_exact_byte_cell_provenance_base(
                       words, arch, diagnostic_lds_byte_offset_vgpr, diagnostic_static_byte_offset,
                       cell_index, relative_cell_index_vgpr, byte_count,
                       consan_moi_shadow_cell::granule_bytes, current_low_vgpr, current_high_vgpr,
                       provenance_temporary_vgpr, temporary_vgpr),
                   "ConSan MOI local publish could not encode exact byte-cell provenance");
  // Worst-case cell loops deliberately include iterations that are empty for
  // some runtime alignments. Remove those lanes before bounds accounting or
  // LDS traffic so only bytes actually touched can consume shadow coverage.
  sequence.append(
      instrumentation::build_v_cmp_ne_u32_vcc(scalar_positive_inline_u32(0u), current_high_vgpr,
                                              arch),
      instrumentation::build_s_and_saveexec_b64(temporary_exec_sgpr, kAmdGpuVccLo, arch));

  uint16_t effective_offset_vgpr = lds_byte_offset_vgpr;
  if (static_byte_offset != 0u) {
    sequence.require(append_compute_effective_lds_byte_offset(
        words, temporary_vgpr, lds_byte_offset_vgpr, static_byte_offset, arch));
    effective_offset_vgpr = temporary_vgpr;
  }

  const uint32_t shadow_cell_capacity = workgroup_shadow.size / sizeof(uint64_t);
  const uint32_t final_cell_index = cell_index;
  const uint32_t cell_capacity =
      final_cell_index < shadow_cell_capacity ? shadow_cell_capacity - final_cell_index : 0u;
  const uint32_t local_base = workgroup_shadow.base + cell_index * sizeof(uint64_t);
  const bool began_bounded_selection = static_cast<bool>(sequence);
  sequence.append(
      instrumentation::build_s_mov_b64(incoming_exec_sgpr, kAmdGpuExecLo, arch),
      instrumentation::build_v_lshrrev_b32(
          temporary_vgpr, scalar_positive_inline_u32(consan_moi_shadow_cell::granule_shift),
          effective_offset_vgpr, arch),
      build_v_mov_b32_e32(saved_cell_vgpr, vector_source_vgpr(temporary_vgpr), arch),
      instrumentation::build_v_mov_b32_literal(address_vgpr, cell_capacity, arch),
      instrumentation::build_v_cmp_gt_u32_vcc(vector_source_vgpr(address_vgpr), temporary_vgpr,
                                              arch),
      instrumentation::build_s_and_saveexec_b64(temporary_exec_sgpr, kAmdGpuVccLo, arch),
      instrumentation::build_s_mov_b64(bounded_exec_sgpr, kAmdGpuExecLo, arch),
      instrumentation::build_s_andn2_b64(kAmdGpuExecLo, incoming_exec_sgpr, bounded_exec_sgpr,
                                         arch));
  if (began_bounded_selection && !sequence)
    errors.emplace_back(
        "ConSan MOI inline-shadow local publish could not encode bounded lane selection");
  const InstructionSequence::Label restore_bounded_exec = sequence.make_label();
  sequence.branch(restore_bounded_exec, InstructionSequence::BranchKind::ExecZero);
  require_emission(
      append_atomic_fetch_add_one_u32(
          words,
          plan.report_buffer_address + offsetof(ConSanMoiReportHeader, inline_undercoverage_count),
          temporary_vgpr, address_vgpr, arch),
      "ConSan MOI inline-shadow local publish could not encode undercoverage "
      "accounting");
  const bool began_restore = static_cast<bool>(sequence);
  sequence.bind_label(restore_bounded_exec)
      .append(instrumentation::build_s_mov_b64(kAmdGpuExecLo, bounded_exec_sgpr, arch));
  if (began_restore && !sequence)
    errors.emplace_back(
        "ConSan MOI inline-shadow local publish could not skip empty undercoverage accounting");
  if (sequence)
    sequence.require(append_workgroup_local_lazy_shadow_ready(words, plan, arch, workgroup_shadow,
                                                              saved_cell_vgpr, errors));
  sequence.append(
      build_v_mov_b32_e32(provenance_temporary_vgpr, vector_source_vgpr(current_low_vgpr), arch));
  require_emission(append_exact_byte_representative_lane(words, arch, provenance_temporary_vgpr,
                                                         current_high_vgpr),
                   "ConSan MOI local publish could not encode its representative lane");
  sequence.append(
      build_v_mov_b32_e32(current_low_vgpr, vector_source_vgpr(saved_current_low_vgpr), arch),
      build_v_mov_b32_e32(current_high_vgpr, vector_source_vgpr(saved_current_high_vgpr), arch));
  require_emission(append_add_exact_shadow_generation(words, current_low_vgpr, current_high_vgpr,
                                                      provenance_temporary_vgpr, temporary_vgpr,
                                                      arch),
                   "ConSan MOI local publish could not embed exact byte-cell provenance");
  const bool began_ds_operations = static_cast<bool>(sequence);
  sequence.append(
      build_v_mov_b32_e32(temporary_vgpr, vector_source_vgpr(saved_cell_vgpr), arch),
      instrumentation::build_v_lshlrev_b32(temporary_vgpr, scalar_positive_inline_u32(3u),
                                           temporary_vgpr, arch),
      instrumentation::build_v_add_u32_literal(address_vgpr, local_base, temporary_vgpr, arch),
      instrumentation::build_ds_storexchg_rtn_b64(old_value_vgpr, address_vgpr, current_low_vgpr,
                                                  0u, arch));
  if (began_ds_operations && !sequence)
    errors.emplace_back(
        "ConSan MOI inline-shadow local publish could not encode CDNA vector/DS operations");
  require_emission(append_moi_lds_wait(words, arch),
                   "ConSan MOI inline-shadow local publish could not encode its LDS result wait");

  // Address formation and conflict filtering use condition state. Reinstall
  // the displaced program's state before the diagnostic helper, whose final
  // restoration makes the local path observationally transparent.
  require_emission(append_restore_moi_special_state(words, plan.scalar_abi.special_state, arch),
                   "ConSan MOI inline-shadow local publish could not restore special scalar state");
  const uint16_t prior_byte_provenance_vgpr = static_cast<uint16_t>(old_value_vgpr + 2u);
  const uint16_t current_byte_provenance_vgpr = static_cast<uint16_t>(old_value_vgpr + 3u);
  require_emission(append_extract_exact_shadow_generation(
                       words, prior_byte_provenance_vgpr, old_value_vgpr,
                       static_cast<uint16_t>(old_value_vgpr + 1u), temporary_vgpr, arch) &&
                       append_extract_exact_shadow_generation(words, current_byte_provenance_vgpr,
                                                              current_low_vgpr, current_high_vgpr,
                                                              temporary_vgpr, arch),
                   "ConSan MOI local publish could not recover exact byte-cell provenance");
  if (sequence)
    require_emission(
        append_inline_shadow_diagnostic_words(
            words, candidate, plan, arch, report_layout, old_value_vgpr,
            static_cast<uint16_t>(old_value_vgpr + 1u), current_low_vgpr, current_high_vgpr,
            diagnostic_lds_byte_offset_vgpr, diagnostic_static_byte_offset, byte_count,
            consan_moi_shadow_cell::granule_bytes,
            /*special_state_already_saved=*/true,
            /*diagnostic_lane_mask_sgpr=*/std::nullopt,
            /*capture_first_diagnostic_only=*/true,
            /*workgroup_local_shadow=*/true, prior_byte_provenance_vgpr,
            current_byte_provenance_vgpr, relative_cell_index_vgpr, cell_index),
        "ConSan MOI inline-shadow local publish could not encode conflict diagnostics");

  sequence.append(
      build_v_mov_b32_e32(current_low_vgpr, vector_source_vgpr(saved_current_low_vgpr), arch),
      build_v_mov_b32_e32(current_high_vgpr, vector_source_vgpr(saved_current_high_vgpr), arch),
      restore_valid_workgroup);
  return sequence.finish(arch);
}

[[nodiscard]] std::optional<std::vector<uint32_t>>
build_inline_shadow_words(std::span<const uint8_t> bytes, const ConSanMoiCandidate &candidate,
                          const MoiInlineShadowEmissionPlan &plan, rj_code_arch_t arch,
                          const ConSanMoiReportBufferLayout &layout, const VgprSpillSequence *spill,
                          std::vector<std::string> &errors,
                          uint32_t *guest_instruction_word_count) {
  const ConSanTargetProfile *target = consan_target_profile(arch);
  const uint16_t scratch_vgpr = plan.scratch_vgpr;
  const uint16_t scratch_count = plan.scratch_count;
  const auto &workgroup_sources = plan.workgroup_sources;
  const auto &workgroup_shadow = plan.workgroup_shadow;
  const auto &private_epoch_offset = plan.private_epoch_offset;
  const auto &owner_derivation = plan.owner_derivation;
  const auto &private_workgroup_key_offset = plan.private_workgroup_key_offset;
  const auto &private_dispatch_id_offset = plan.private_dispatch_id_offset;
  const bool byte_granular_external_shadow = plan.byte_granular_external_shadow;
  const bool spill_overlaps_guest_operands = plan.spill_overlaps_guest_operands;
  const bool has_hardware_dispatch_id = plan.dispatch_id.sgpr || plan.dispatch_id.vgpr;
  if (!target || !plan.scalar_state.exec_save_sgpr ||
      (!has_hardware_dispatch_id && !plan.dispatch_id.literal) ||
      !plan.dispatch_id.is_well_formed()) {
    errors.emplace_back(
        "ConSan MOI versioned exact-shadow probe requires EXEC-save and dispatch-ID state");
    return std::nullopt;
  }
  const MoiAccessResourceFacts &access_resource_facts = plan.access_resource_facts;
  const uint16_t normal_scratch_count =
      inline_shadow_scratch_count(plan.track_atomics, access_resource_facts, candidate);
  const uint16_t spill_scratch_count = inline_shadow_spill_backed_scratch_count(
      plan.track_atomics, access_resource_facts, candidate);
  if (scratch_count != normal_scratch_count &&
      (spill == nullptr || scratch_count != spill_scratch_count)) {
    errors.emplace_back(
        "ConSan MOI inline-shadow probe scratch VGPR count disagrees with its resource plan");
    return std::nullopt;
  }
  const uint16_t expected_spill_count = static_cast<uint16_t>(
      scratch_count + static_cast<uint16_t>(spill != nullptr && !spill->uses_dynamic_stack_frame &&
                                            plan.scalar_state.inline_scalar_spill));
  if (spill != nullptr &&
      (spill->vgpr_base != scratch_vgpr || spill->vgpr_count != expected_spill_count)) {
    errors.emplace_back(
        "ConSan MOI inline-shadow probe spill window disagrees with its resource plan");
    return std::nullopt;
  }
  if (static_cast<uint32_t>(scratch_vgpr) + scratch_count > kMaxVgprs) {
    errors.emplace_back("ConSan MOI inline-shadow probe scratch VGPR window exceeds the limit");
    return std::nullopt;
  }
  if (!validate_inline_shadow_exec_save_sgpr(plan.scalar_state,
                                             /*inline_access_present=*/true,
                                             plan.required_exec_save_sgpr_count, arch, errors))
    return std::nullopt;
  auto lds_byte_offset_vgpr = candidate_lds_byte_offset_vgpr(candidate, errors);
  if (!lds_byte_offset_vgpr)
    return std::nullopt;
  if (!spill_overlaps_guest_operands &&
      reject_candidate_scratch_range_overlap(candidate, scratch_vgpr, scratch_count, errors))
    return std::nullopt;
  if (!plan.owner_field.owner_vgpr && !plan.owner_field.persistent_owner_sgpr &&
      !plan.owner_field.automatic_private_epoch) {
    errors.emplace_back("ConSan MOI inline-shadow probe requires persistent owner state");
    return std::nullopt;
  }
  if (reject_optional_scratch_range_overlap(plan.owner_field.owner_vgpr, scratch_vgpr,
                                            scratch_count, "MOI owner", errors) ||
      reject_optional_scratch_range_overlap(plan.epoch_field.epoch_vgpr, scratch_vgpr,
                                            scratch_count, "MOI epoch", errors) ||
      reject_optional_scratch_range_overlap(plan.workgroup_key_registers.cached_key_vgpr,
                                            scratch_vgpr, scratch_count, "MOI workgroup key",
                                            errors))
    return std::nullopt;
  if (plan.dispatch_id.vgpr &&
      range_overlaps(*plan.dispatch_id.vgpr, 2u, scratch_vgpr, scratch_count)) {
    errors.emplace_back("ConSan MOI scratch VGPRs overlap the dispatch-ID VGPR pair");
    return std::nullopt;
  }

  const auto &access_ranges = candidate.site().ranges;
  if (access_ranges.empty()) {
    errors.emplace_back("ConSan MOI inline-shadow probe requires a supported LDS access range");
    return std::nullopt;
  }

  const uint16_t address_lo_vgpr = scratch_vgpr;
  const uint16_t address_hi_vgpr = static_cast<uint16_t>(scratch_vgpr + 1u);
  const uint16_t low_vgpr = static_cast<uint16_t>(scratch_vgpr + 2u);
  const uint16_t high_vgpr = static_cast<uint16_t>(scratch_vgpr + 3u);
  const uint16_t tmp_vgpr = static_cast<uint16_t>(scratch_vgpr + 4u);
  const uint16_t old_value_vgpr = static_cast<uint16_t>(scratch_vgpr + 5u);
  const uint16_t cas_expected_vgpr = consan_detail::inline_shadow_cas_expected_vgpr(old_value_vgpr);
  if (cas_expected_vgpr >= static_cast<uint32_t>(scratch_vgpr) + scratch_count) {
    errors.emplace_back(
        "ConSan MOI inline-shadow probe has no scratch VGPR for the expected CAS value");
    return std::nullopt;
  }
  const bool materialize_flat_address = candidate_requires_flat_address_materialization(candidate);
  const bool capture_high_bank_address =
      moi_access_requires_high_bank_address_capture(candidate, arch);
  const auto source_is_spilled = [&](uint16_t source) {
    return spill != nullptr && source >= spill->vgpr_base &&
           source < static_cast<uint32_t>(spill->vgpr_base) + spill->vgpr_count;
  };
  const std::optional<uint16_t> spill_backed_lds_byte_offset_source =
      spill != nullptr && !materialize_flat_address &&
              target->moi_access.clobbered_address_spill_reload &&
              source_is_spilled(*lds_byte_offset_vgpr)
          ? lds_byte_offset_vgpr
          : std::nullopt;
  const bool compact_spill_keeps_disjoint_clobbered_address =
      spill != nullptr && scratch_count == spill_scratch_count &&
      access_resource_facts.supports_clobbered_address_spill_reload &&
      !source_is_spilled(*lds_byte_offset_vgpr);
  std::optional<uint16_t> saved_lds_byte_offset_vgpr;
  if (candidate.is_direct_to_lds()) {
    saved_lds_byte_offset_vgpr = static_cast<uint16_t>(
        consan_detail::inline_shadow_loop_counter_vgpr(
            scratch_vgpr, plan.scalar_state.exec_save_sgpr.has_value(), plan.track_atomics) +
        inline_shadow_loop_scratch_count(candidate));
  } else if (materialize_flat_address) {
    saved_lds_byte_offset_vgpr = static_cast<uint16_t>(
        scratch_vgpr + scratch_count - flat_access_address_scratch_count(candidate));
  } else if (spill_backed_lds_byte_offset_source) {
    // Address construction and local exchange consume the guest offset before
    // the transaction initializes its expected CAS value. The authoritative
    // spill slot reloads the offset again before diagnostics. This phase
    // sharing keeps the minimum native CDNA access window at 16 VGPRs.
    saved_lds_byte_offset_vgpr = cas_expected_vgpr;
  } else if ((moi_load_clobbers_address(candidate) || capture_high_bank_address) &&
             !compact_spill_keeps_disjoint_clobbered_address) {
    saved_lds_byte_offset_vgpr = static_cast<uint16_t>(
        consan_detail::inline_shadow_loop_counter_vgpr(
            scratch_vgpr, plan.scalar_state.exec_save_sgpr.has_value(), plan.track_atomics) +
        inline_shadow_loop_scratch_count(candidate));
  }

  const auto kind = consan_moi_shadow_kind_from_access_kind(candidate.site().kind);
  const uint32_t low_literal = static_cast<uint32_t>(kind);
  const uint32_t high_literal =
      static_cast<uint32_t>((candidate.anchor() & consan_moi_exact_shadow::max_instruction_offset)
                            << (consan_moi_exact_shadow::instruction_offset_shift - 32u));
  const uint64_t exact_shadow_base =
      plan.report_buffer_address + layout.exact_shadow_entries_offset;
  const uint32_t external_shadow_granule_bytes =
      byte_granular_external_shadow ? 1u : consan_moi_shadow_cell::granule_bytes;

  std::vector<uint32_t> words;
  words.reserve(candidate.size() / sizeof(uint32_t) + 64u +
                (plan.scalar_state.exec_save_sgpr ? 120u : 0u) +
                (plan.epoch_field.epoch_vgpr ? 2u : 0u));
  InstructionSequence sequence(words);
  if (private_dispatch_id_offset) {
    if (!plan.dispatch_id.sgpr) {
      errors.emplace_back(
          "ConSan MOI private dispatch reload requires a transient scalar destination");
      return std::nullopt;
    }
    if (!sequence.emit_all(
            instrumentation::build_private_load_b32(address_lo_vgpr, *private_dispatch_id_offset,
                                                    arch),
            instrumentation::build_private_load_b32(
                address_hi_vgpr, *private_dispatch_id_offset + SpillManager::kSlotBytes, arch),
            instrumentation::build_s_wait_private_load0(arch),
            instrumentation::build_v_readfirstlane_b32(*plan.dispatch_id.sgpr, address_lo_vgpr,
                                                       arch),
            instrumentation::build_v_readfirstlane_b32(
                static_cast<uint16_t>(*plan.dispatch_id.sgpr + 1u), address_hi_vgpr, arch),
            instrumentation::build_valu_to_salu_dependency_wait(arch))) {
      errors.emplace_back(
          "ConSan MOI inline-shadow probe could not reload private dispatch identity");
      return std::nullopt;
    }
  }
  const auto reload_spill_backed_lds_byte_offset = [&]() {
    if (!spill_backed_lds_byte_offset_source)
      return true;
    if (spill == nullptr) {
      errors.emplace_back("ConSan MOI inline-shadow probe has no spill plan for its LDS address");
      return false;
    }
    const auto reload = consan_detail::append_reload_moi_spilled_vgpr(
        words, *spill, *saved_lds_byte_offset_vgpr, *spill_backed_lds_byte_offset_source, arch);
    if (reload == consan_detail::MoiSpilledVgprReloadResult::Appended)
      return true;
    errors.emplace_back("ConSan MOI inline-shadow probe could not recover its spilled LDS "
                        "address: " +
                        std::string(consan_detail::moi_spilled_vgpr_reload_result_name(reload)));
    return false;
  };
  // A DS load may use the same VGPR for its address and destination. Preserve
  // the effective address before executing the displaced guest instruction;
  // otherwise the loaded payload becomes the exact-shadow cell index.
  if (saved_lds_byte_offset_vgpr) {
    if (!spill_backed_lds_byte_offset_source) {
      if (candidate.is_direct_to_lds()) {
        if (!plan.scalar_state.exec_save_sgpr) {
          errors.emplace_back(
              "ConSan MOI inline-shadow direct-to-LDS probe requires an EXEC-save SGPR pair");
          return std::nullopt;
        }
        if (!append_materialize_direct_to_lds_address(words, candidate, *saved_lds_byte_offset_vgpr,
                                                      *plan.scalar_state.exec_save_sgpr, arch)) {
          errors.emplace_back("ConSan MOI inline-shadow probe could not materialize a "
                              "direct-to-LDS destination");
          return std::nullopt;
        }
      } else if (!capture_high_bank_address && !materialize_flat_address) {
        words.push_back(build_v_mov_b32_e32(*saved_lds_byte_offset_vgpr,
                                            vector_source_vgpr(*lds_byte_offset_vgpr), arch));
      }
    }
    lds_byte_offset_vgpr = saved_lds_byte_offset_vgpr;
  }
  // Publish the shadow before the application access, then execute that access
  // immediately before its continuation. This prevents another wave from
  // observing application data while either side is still in trampoline
  // bookkeeping. Scratch allocation excludes every guest operand, so a
  // deferred store retains its address and payload; an address-clobbering load
  // uses the snapshot above for metadata and produces its application result
  // only after instrumentation is finished.
  // Preserve application condition state before metadata construction. Owner,
  // epoch, address, and partition setup all use VCC/SCC-producing
  // instructions; snapshotting inside the exchange loop would retain those
  // generated values rather than the state at the displaced LDS access.
  const bool preserve_special_state = plan.scalar_state.exec_save_sgpr.has_value();
  if (preserve_special_state &&
      !append_save_moi_special_state(words, plan.scalar_abi.special_state, arch)) {
    errors.emplace_back("ConSan MOI inline-shadow probe could not preserve application VCC/SCC");
    return std::nullopt;
  }
  if (materialize_flat_address) {
    if (!append_materialize_flat_access_address(words, candidate,
                                                *candidate.site().lowering.form->address_vgpr,
                                                *lds_byte_offset_vgpr, arch)) {
      errors.emplace_back("ConSan MOI inline-shadow probe could not materialize FLAT address");
      return std::nullopt;
    }
  }
  if (!append_moi_lds_wait(words, arch))
    return std::nullopt;

  const uint16_t workgroup_key_vgpr = old_value_vgpr;
  const uint16_t original_exec_save_offset = kConSanMoiInlineOriginalExecSaveOffset;
  MoiWorkgroupKeyRegisterPlan key_registers = plan.workgroup_key_registers;
  // Workgroup-local mirrors are physically unreachable from every other live
  // workgroup and omit the generation field.
  if (workgroup_shadow) {
    const auto save_original_exec = instrumentation::build_s_mov_b64(
        static_cast<uint16_t>(*plan.scalar_state.exec_save_sgpr + original_exec_save_offset),
        kAmdGpuExecLo, arch);
    if (!save_original_exec) {
      errors.emplace_back(
          "ConSan MOI inline-shadow probe could not retain its original workgroup lanes");
      return std::nullopt;
    }
    words.push_back(*save_original_exec);
  }
  if (private_workgroup_key_offset && !workgroup_shadow) {
    const auto load = instrumentation::build_private_load_b32(workgroup_key_vgpr,
                                                              *private_workgroup_key_offset, arch);
    const auto wait = instrumentation::build_s_wait_private_load0(arch);
    if (!load || !wait) {
      errors.emplace_back(
          "ConSan MOI inline-shadow probe could not load its private workgroup key");
      return std::nullopt;
    }
    words.insert(words.end(), load->begin(), load->end());
    words.push_back(*wait);
    key_registers.cached_key_vgpr = workgroup_key_vgpr;
    key_registers.cached_key_sgpr = std::nullopt;
  }
  if (!workgroup_shadow && plan.scalar_state.exec_save_sgpr &&
      !append_inline_workgroup_key(words, workgroup_sources, key_registers, workgroup_key_vgpr,
                                   tmp_vgpr, address_lo_vgpr, original_exec_save_offset, arch)) {
    errors.emplace_back("ConSan MOI inline-shadow probe could not derive workgroup identity");
    return std::nullopt;
  }
  if (!workgroup_shadow && plan.scalar_state.exec_save_sgpr) {
    // The packed key is injective only inside its documented dimensional
    // bounds. Keep the guest running, but make every excluded lane-site
    // encounter observable in the Inline-only header flags/count field so a
    // clean acceptance run cannot silently pass with detector undercoverage.
    const uint16_t exec_base = *plan.scalar_state.exec_save_sgpr;
    const auto save_valid_exec = instrumentation::build_s_mov_b64(exec_base, kAmdGpuExecLo, arch);
    const auto select_invalid_exec = instrumentation::build_s_andn2_b64(
        kAmdGpuExecLo, static_cast<uint16_t>(exec_base + original_exec_save_offset), exec_base,
        arch);
    const auto restore_valid_exec =
        instrumentation::build_s_mov_b64(kAmdGpuExecLo, exec_base, arch);
    std::vector<uint32_t> accounting_words;
    if (!save_valid_exec || !select_invalid_exec || !restore_valid_exec ||
        !append_atomic_fetch_add_one_u32(
            accounting_words,
            plan.report_buffer_address + offsetof(ConSanMoiReportHeader, inline_unsupported_count),
            tmp_vgpr, address_lo_vgpr, arch)) {
      errors.emplace_back(
          "ConSan MOI inline-shadow probe could not account unsupported workgroup identity");
      return std::nullopt;
    }
    words.push_back(*save_valid_exec);
    words.push_back(*select_invalid_exec);
    words.insert(words.end(), accounting_words.begin(), accounting_words.end());
    words.push_back(*restore_valid_exec);
  }
  if (candidate.site().container.is_kernel() && candidate.is_flat() &&
      candidate.site().flat_address_space_hint != ConSanFlatAddressSpaceHint::Group) {
    // Likely provenance deliberately admits flat sites whose static dataflow
    // cannot prove one aperture. Resolve that uncertainty with the same high
    // address comparison used by hardware: a 32-bit SHARED_BASE source is the
    // shared aperture's high half. Global lanes bypass shadow bookkeeping and
    // later rejoin for the displaced guest access.
    uint16_t address_hi_source =
        materialize_flat_address
            ? static_cast<uint16_t>(*lds_byte_offset_vgpr + 1u)
            : static_cast<uint16_t>(*candidate.site().lowering.form->address_vgpr + 1u);
    if (!materialize_flat_address && source_is_spilled(address_hi_source)) {
      const uint16_t recovered_address_hi = static_cast<uint16_t>(scratch_vgpr + 1u);
      const auto reload = consan_detail::append_reload_moi_spilled_vgpr(
          words, *spill, recovered_address_hi, address_hi_source, arch);
      if (reload != consan_detail::MoiSpilledVgprReloadResult::Appended) {
        errors.emplace_back(
            "ConSan MOI inline-shadow probe could not recover a spilled flat "
            "address: " +
            std::string(consan_detail::moi_spilled_vgpr_reload_result_name(reload)));
        return std::nullopt;
      }
      address_hi_source = recovered_address_hi;
    }
    const auto in_shared_aperture =
        instrumentation::build_v_cmp_eq_u32_vcc(kScalarOperandSharedBase, address_hi_source, arch);
    const auto retain_shared_lanes = instrumentation::build_s_and_saveexec_b64(
        static_cast<uint16_t>(*plan.scalar_state.exec_save_sgpr + 2u), kAmdGpuVccLo, arch);
    if (!in_shared_aperture || !retain_shared_lanes) {
      errors.emplace_back(
          "ConSan MOI inline-shadow probe could not resolve a maybe-group flat aperture");
      return std::nullopt;
    }
    words.push_back(*in_shared_aperture);
    words.push_back(*retain_shared_lanes);
  }
  if (workgroup_shadow) {
    const auto retain_valid_workgroup = instrumentation::build_s_mov_b64(
        static_cast<uint16_t>(*plan.scalar_state.exec_save_sgpr + 22u), kAmdGpuExecLo, arch);
    if (!retain_valid_workgroup) {
      errors.emplace_back(
          "ConSan MOI inline-shadow probe could not retain validated workgroup lanes");
      return std::nullopt;
    }
    words.push_back(*retain_valid_workgroup);
  }

  const auto mov_low = instrumentation::build_v_mov_b32_literal(low_vgpr, low_literal, arch);
  const auto mov_high = instrumentation::build_v_mov_b32_literal(high_vgpr, high_literal, arch);
  if (!mov_low || !mov_high) {
    errors.emplace_back("ConSan MOI inline-shadow probe could not encode exact-shadow literals");
    return std::nullopt;
  }
  words.insert(words.end(), mov_low->begin(), mov_low->end());
  if (!append_inline_shadow_owner_field(words, plan.owner_field, low_vgpr, tmp_vgpr, high_vgpr,
                                        arch, owner_derivation, errors)) {
    errors.emplace_back("ConSan MOI inline-shadow probe could not encode owner field");
    return std::nullopt;
  }
  if (!append_inline_shadow_epoch_field(words, plan.epoch_field, private_epoch_offset, low_vgpr,
                                        tmp_vgpr, arch, errors)) {
    errors.emplace_back("ConSan MOI inline-shadow probe could not encode epoch field");
    return std::nullopt;
  }
  words.insert(words.end(), mov_high->begin(), mov_high->end());
  if (!workgroup_shadow && plan.scalar_state.exec_save_sgpr &&
      !append_add_exact_shadow_generation(words, low_vgpr, high_vgpr, workgroup_key_vgpr, tmp_vgpr,
                                          arch)) {
    errors.emplace_back("ConSan MOI inline-shadow probe could not encode workgroup identity");
    return std::nullopt;
  }

  // The global versioned transaction keeps old_value at +5 and aligns its CAS
  // tuple downward. The workgroup-local B64 DS exchange instead aligns its
  // return tuple upward into the unused following registers.
  const uint16_t local_old_value_vgpr = static_cast<uint16_t>(
      old_value_vgpr + (target->flat_compare_swap_data_pair_alignment -
                        old_value_vgpr % target->flat_compare_swap_data_pair_alignment) %
                           target->flat_compare_swap_data_pair_alignment);

  for (const ConSanAccessRange &range : access_ranges) {
    ConSanMoiLdsCellRange cell_range;
    if (workgroup_shadow || !byte_granular_external_shadow) {
      cell_range =
          consan_moi_lds_cell_range_for_bytes(candidate.lowering_offset(range), range.byte_width);
      // The register-sourced LDS base can have any byte alignment even when
      // the instruction's static offset is aligned. Emit the target-neutral
      // maximum and let the exact mask remove empty per-lane iterations before
      // coverage accounting and publication.
      cell_range.cell_count = consan_moi_maximum_cell_count_for_unaligned_bytes(range.byte_width);
    } else {
      // A byte-addressed shadow has no runtime alignment expansion: every
      // accessed byte maps to exactly one independent external slot.
      cell_range.start_cell = candidate.lowering_offset(range);
      cell_range.cell_count = range.byte_width;
    }
    const bool use_cell_loop =
        cell_range.cell_count > 1u && inline_shadow_loop_scratch_count(candidate) != 0u;
    for (uint32_t cell_index = 0; cell_index < cell_range.cell_count; ++cell_index) {
      if (workgroup_shadow) {
        if (use_cell_loop) {
          if (cell_index != 0u)
            continue;
          const uint16_t loop_counter_vgpr = consan_detail::inline_shadow_loop_counter_vgpr(
              scratch_vgpr, plan.scalar_state.exec_save_sgpr.has_value(), plan.track_atomics);
          const uint16_t loop_offset_vgpr = static_cast<uint16_t>(loop_counter_vgpr + 1u);
          if (static_cast<uint32_t>(loop_offset_vgpr) >=
              static_cast<uint32_t>(scratch_vgpr) + scratch_count) {
            errors.emplace_back(
                "ConSan MOI inline-shadow local cell loop exceeds its scratch plan");
            return std::nullopt;
          }
          if (!reload_spill_backed_lds_byte_offset()) {
            errors.emplace_back(
                "ConSan MOI inline-shadow probe could not recover its spilled LDS address");
            return std::nullopt;
          }
          if (candidate.lowering_offset(range) == 0u) {
            (void)sequence.emit(build_v_mov_b32_e32(
                loop_offset_vgpr, vector_source_vgpr(*lds_byte_offset_vgpr), arch));
          } else if (!append_compute_effective_lds_byte_offset(
                         words, loop_offset_vgpr, *lds_byte_offset_vgpr,
                         candidate.lowering_offset(range), arch)) {
            errors.emplace_back(
                "ConSan MOI inline-shadow probe could not initialize its local cell loop");
            return std::nullopt;
          }
          (void)sequence.emit(
              build_v_mov_b32_e32(loop_counter_vgpr, scalar_positive_inline_u32(0), arch));
          const InstructionSequence::Label loop_begin = sequence.mark_label();
          if (!append_workgroup_local_exact_shadow_swap(
                  words, candidate, plan, arch, layout, *workgroup_shadow, address_lo_vgpr,
                  low_vgpr, high_vgpr, local_old_value_vgpr, loop_offset_vgpr,
                  /*static_byte_offset=*/0u, /*cell_index=*/0u, range.byte_width,
                  *lds_byte_offset_vgpr, candidate.lowering_offset(range), loop_counter_vgpr,
                  errors)) {
            errors.emplace_back(
                "ConSan MOI inline-shadow probe could not encode looped local LDS shadow publish");
            return std::nullopt;
          }
          const uint16_t loop_exec_save_sgpr =
              static_cast<uint16_t>(*plan.scalar_state.exec_save_sgpr + 14u);
          const uint16_t valid_workgroup_exec_sgpr =
              static_cast<uint16_t>(*plan.scalar_state.exec_save_sgpr + 22u);
          // Vector compares only define VCC for active lanes. Branching on raw
          // VCC can therefore observe stale guest bits from inactive lanes and
          // loop forever. Narrow EXEC to the active lanes that still need a
          // cell, branch on that mask, then restore the validated workgroup
          // mask on loop exit.
          if (!sequence.emit_all(
                  instrumentation::build_v_add_u32(
                      loop_offset_vgpr,
                      scalar_positive_inline_u32(consan_moi_shadow_cell::granule_bytes),
                      loop_offset_vgpr, arch),
                  instrumentation::build_v_add_u32(loop_counter_vgpr, scalar_positive_inline_u32(1),
                                                   loop_counter_vgpr, arch),
                  instrumentation::build_v_cmp_gt_u32_vcc(
                      scalar_positive_inline_u32(cell_range.cell_count), loop_counter_vgpr, arch),
                  instrumentation::build_s_and_saveexec_b64(loop_exec_save_sgpr, kAmdGpuVccLo,
                                                            arch)) ||
              !sequence.emit_branch(loop_begin, InstructionSequence::BranchKind::ExecNonzero) ||
              !sequence.emit(instrumentation::build_s_mov_b64(kAmdGpuExecLo,
                                                              valid_workgroup_exec_sgpr, arch))) {
            errors.emplace_back(
                "ConSan MOI inline-shadow probe could not encode its local cell loop control");
            return std::nullopt;
          }
        } else if (!reload_spill_backed_lds_byte_offset() ||
                   !append_workgroup_local_exact_shadow_swap(
                       words, candidate, plan, arch, layout, *workgroup_shadow, address_lo_vgpr,
                       low_vgpr, high_vgpr, local_old_value_vgpr, *lds_byte_offset_vgpr,
                       candidate.lowering_offset(range), cell_index, range.byte_width,
                       *lds_byte_offset_vgpr, candidate.lowering_offset(range),
                       /*relative_cell_index_vgpr=*/std::nullopt, errors)) {
          errors.emplace_back(
              "ConSan MOI inline-shadow probe could not encode local LDS shadow publish");
          return std::nullopt;
        }
        continue;
      }
      const bool loop_external_cells = use_cell_loop;
      const uint16_t loop_counter_vgpr = consan_detail::inline_shadow_loop_counter_vgpr(
          scratch_vgpr, plan.scalar_state.exec_save_sgpr.has_value(), plan.track_atomics);
      const uint16_t saved_workgroup_key_vgpr = static_cast<uint16_t>(loop_counter_vgpr + 1u);
      if (loop_external_cells) {
        if (static_cast<uint32_t>(saved_workgroup_key_vgpr) >=
            static_cast<uint32_t>(scratch_vgpr) + scratch_count) {
          errors.emplace_back(
              "ConSan MOI inline-shadow external cell loop exceeds its scratch plan");
          return std::nullopt;
        }
        if (cell_index != 0u)
          continue;
        (void)sequence.emit_all(
            build_v_mov_b32_e32(loop_counter_vgpr, scalar_positive_inline_u32(0), arch),
            build_v_mov_b32_e32(saved_workgroup_key_vgpr, vector_source_vgpr(workgroup_key_vgpr),
                                arch));
      }
      // Recompute the complete external address at the top of every wide-cell
      // iteration. The versioned transaction deliberately reuses its address
      // pair, so advancing that pair after the transaction is not valid.
      const InstructionSequence::Label external_loop_begin = sequence.mark_label();
      if (!sequence.emit_all(
              instrumentation::build_v_mov_b32_literal(
                  address_lo_vgpr, static_cast<uint32_t>(exact_shadow_base), arch),
              instrumentation::build_v_mov_b32_literal(
                  address_hi_vgpr, static_cast<uint32_t>(exact_shadow_base >> 32u), arch))) {
        errors.emplace_back("ConSan MOI inline-shadow probe could not encode shadow address");
        return std::nullopt;
      }

      const uint32_t entries_per_dispatch_bank =
          layout.exact_shadow_entry_capacity / layout.inline_exact_dispatch_bank_count;
      const uint32_t dispatch_bank_stride =
          entries_per_dispatch_bank * sizeof(ConSanMoiInlineExactShadowSlot);
      // The versioned transaction reuses old_value_vgpr, where the workgroup
      // key was initially materialized. Wide runtime loops already reserve a
      // dedicated saved key; use it directly on every iteration. Unrolled
      // multi-cell probes have no such register, so reconstruct their key from
      // the packed access before every cell. Both paths prevent a later cell
      // from selecting a different dispatch bank after the first transaction.
      uint16_t dispatch_bank_key_vgpr = saved_workgroup_key_vgpr;
      if (!loop_external_cells) {
        dispatch_bank_key_vgpr = old_value_vgpr;
        if (!append_extract_exact_shadow_generation(words, old_value_vgpr, low_vgpr, high_vgpr,
                                                    tmp_vgpr, arch)) {
          errors.emplace_back(
              "ConSan MOI inline-shadow probe could not recover its exact-shadow bank key");
          return std::nullopt;
        }
      }
      if (layout.inline_exact_dispatch_bank_count == 0u ||
          (layout.inline_exact_dispatch_bank_count &
           (layout.inline_exact_dispatch_bank_count - 1u)) != 0u ||
          entries_per_dispatch_bank == 0u || dispatch_bank_stride == 0u ||
          !append_moi_report_dispatch_id_word(words, plan.dispatch_id, tmp_vgpr,
                                              /*high_word=*/false, arch) ||
          !sequence.emit_all(
              instrumentation::build_v_xor_b32(tmp_vgpr, vector_source_vgpr(dispatch_bank_key_vgpr),
                                               tmp_vgpr, arch),
              instrumentation::build_v_and_b32_literal(
                  tmp_vgpr, layout.inline_exact_dispatch_bank_count - 1u, tmp_vgpr, arch),
              instrumentation::build_v_mul_lo_u32_literal(tmp_vgpr, old_value_vgpr,
                                                          dispatch_bank_stride, tmp_vgpr, arch),
              instrumentation::build_v_add_u64_vgpr_offset(address_lo_vgpr, tmp_vgpr, arch))) {
        errors.emplace_back("ConSan MOI inline-shadow probe could not encode dispatch bank");
        return std::nullopt;
      }

      if (!reload_spill_backed_lds_byte_offset()) {
        errors.emplace_back(
            "ConSan MOI inline-shadow probe could not recover its spilled LDS address");
        return std::nullopt;
      }
      uint16_t effective_lds_byte_offset_vgpr = *lds_byte_offset_vgpr;
      if (candidate.lowering_offset(range) != 0) {
        if (!append_compute_effective_lds_byte_offset(words, tmp_vgpr, *lds_byte_offset_vgpr,
                                                      candidate.lowering_offset(range), arch)) {
          errors.emplace_back("ConSan MOI inline-shadow probe could not encode LDS byte offset");
          return std::nullopt;
        }
        effective_lds_byte_offset_vgpr = tmp_vgpr;
      }

      std::optional<uint32_t> start_cell_shift;
      if (external_shadow_granule_bytes == 1u) {
        start_cell_shift =
            build_v_mov_b32_e32(tmp_vgpr, vector_source_vgpr(effective_lds_byte_offset_vgpr), arch);
      } else {
        start_cell_shift = instrumentation::build_v_lshrrev_b32(
            tmp_vgpr, scalar_positive_inline_u32(consan_moi_shadow_cell::granule_shift),
            effective_lds_byte_offset_vgpr, arch);
      }
      if (!sequence.emit_all(
              start_cell_shift,
              instrumentation::build_v_lshlrev_b32(tmp_vgpr, scalar_positive_inline_u32(3),
                                                   tmp_vgpr, arch),
              instrumentation::build_v_add_u64_vgpr_offset(address_lo_vgpr, tmp_vgpr, arch),
              instrumentation::build_v_lshlrev_b32(tmp_vgpr, scalar_positive_inline_u32(1),
                                                   tmp_vgpr, arch),
              instrumentation::build_v_add_u64_vgpr_offset(address_lo_vgpr, tmp_vgpr, arch))) {
        errors.emplace_back("ConSan MOI inline-shadow probe could not encode shadow publish");
        return std::nullopt;
      }
      if (loop_external_cells) {
        if (!sequence.emit_all(
                instrumentation::build_v_mul_lo_u32_literal(tmp_vgpr, old_value_vgpr,
                                                            sizeof(ConSanMoiInlineExactShadowSlot),
                                                            loop_counter_vgpr, arch),
                instrumentation::build_v_add_u64_vgpr_offset(address_lo_vgpr, tmp_vgpr, arch))) {
          errors.emplace_back("ConSan MOI inline-shadow probe could not encode looped cell offset");
          return std::nullopt;
        }
      } else if (cell_index != 0) {
        if (!sequence.emit(instrumentation::build_v_add_u64_signed_i24(
                address_lo_vgpr,
                static_cast<int32_t>(cell_index * sizeof(ConSanMoiInlineExactShadowSlot)), arch))) {
          errors.emplace_back("ConSan MOI inline-shadow probe could not encode shadow cell offset");
          return std::nullopt;
        }
      }
      if (!append_wave_coalesced_exact_shadow_swap(
              words, candidate, plan, arch, layout, address_lo_vgpr, low_vgpr, high_vgpr,
              old_value_vgpr, *lds_byte_offset_vgpr, candidate.lowering_offset(range),
              range.byte_width, external_shadow_granule_bytes, cell_index,
              loop_external_cells ? std::optional<uint16_t>(loop_counter_vgpr) : std::nullopt,
              spill, spill_backed_lds_byte_offset_source, errors)) {
        errors.emplace_back(
            "ConSan MOI inline-shadow probe could not encode versioned shadow publish with "
            "EXEC base s" +
            std::to_string(*plan.scalar_state.exec_save_sgpr) + " and dispatch base " +
            (plan.dispatch_id.sgpr   ? "s" + std::to_string(*plan.dispatch_id.sgpr)
             : plan.dispatch_id.vgpr ? "v" + std::to_string(*plan.dispatch_id.vgpr)
                                     : std::string("literal")));
        return std::nullopt;
      }
      if (loop_external_cells) {
        const uint16_t temporary_exec_sgpr =
            static_cast<uint16_t>(*plan.scalar_state.exec_save_sgpr + 14u);
        const uint16_t incoming_exec_sgpr =
            static_cast<uint16_t>(*plan.scalar_state.exec_save_sgpr + 12u);
        if (!sequence.emit_all(
                instrumentation::build_v_add_u32(loop_counter_vgpr, scalar_positive_inline_u32(1),
                                                 loop_counter_vgpr, arch),
                instrumentation::build_v_cmp_gt_u32_vcc(
                    scalar_positive_inline_u32(cell_range.cell_count), loop_counter_vgpr, arch),
                instrumentation::build_s_and_saveexec_b64(temporary_exec_sgpr, kAmdGpuVccLo,
                                                          arch)) ||
            !sequence.emit_branch(external_loop_begin,
                                  InstructionSequence::BranchKind::ExecNonzero) ||
            !sequence.emit(
                instrumentation::build_s_mov_b64(kAmdGpuExecLo, incoming_exec_sgpr, arch))) {
          errors.emplace_back(
              "ConSan MOI inline-shadow probe could not encode its external cell loop");
          return std::nullopt;
        }
      }
    }
  }
  if (workgroup_shadow) {
    const std::optional<uint16_t> visible_evidence_sgpr =
        inline_shadow_visible_evidence_sgpr(plan.scalar_state);
    std::optional<InstructionSequence::Label> visible_evidence_done;
    if (visible_evidence_sgpr) {
      visible_evidence_done = sequence.make_label();
      if (!sequence.emit(instrumentation::build_s_cmp_lg_u32(
              *visible_evidence_sgpr, scalar_positive_inline_u32(0), arch)) ||
          !sequence.emit_branch(*visible_evidence_done,
                                InstructionSequence::BranchKind::SccNonzero)) {
        errors.emplace_back("ConSan MOI inline-shadow probe could not test visible evidence");
        return std::nullopt;
      }
    }
    const uint16_t exec_base = *plan.scalar_state.exec_save_sgpr;
    const uint16_t active_exec_sgpr = static_cast<uint16_t>(exec_base + 12u);
    const uint16_t temporary_exec_sgpr = static_cast<uint16_t>(exec_base + 14u);
    const MoiVisibleEvidencePublicationResult publication =
        append_publish_first_active_lane_visible_evidence_if_zero(
            words, plan.report_buffer_address + offsetof(ConSanMoiReportHeader, event_counter),
            tmp_vgpr, address_lo_vgpr, active_exec_sgpr, temporary_exec_sgpr, arch);
    if (publication != MoiVisibleEvidencePublicationResult::Appended) {
      errors.emplace_back(
          publication == MoiVisibleEvidencePublicationResult::ElectionUnsupported
              ? "ConSan MOI inline-shadow probe could not elect visible evidence"
              : "ConSan MOI inline-shadow probe could not publish visible evidence");
      return std::nullopt;
    }
    if (visible_evidence_sgpr) {
      (void)sequence.emit(
          build_s_mov_b32(*visible_evidence_sgpr, scalar_positive_inline_u32(1), arch));
      if (!sequence.bind(*visible_evidence_done))
        return std::nullopt;
    }
  }
  if (plan.scalar_state.exec_save_sgpr) {
    if (!sequence.emit(instrumentation::build_s_mov_b64(
            kAmdGpuExecLo,
            static_cast<uint16_t>(*plan.scalar_state.exec_save_sgpr + original_exec_save_offset),
            arch))) {
      errors.emplace_back(
          "ConSan MOI inline-shadow probe could not restore workgroup-filtered state");
      return std::nullopt;
    }
    if (preserve_special_state &&
        !append_restore_moi_special_state(words, plan.scalar_abi.special_state, arch)) {
      errors.emplace_back(
          "ConSan MOI inline-shadow probe could not restore workgroup-filtered state");
      return std::nullopt;
    }
  }
  if (target == nullptr) {
    errors.emplace_back("ConSan MOI inline-shadow probe has no target profile");
    return std::nullopt;
  }
  const bool reserve_two_address_replay_scratch =
      moi_guest_access_relocation_requires_adjusted_address(candidate, *target);
  const uint16_t guest_address_vgpr = capture_high_bank_address
                                          ? *candidate.site().lowering.form->address_vgpr
                                          : *lds_byte_offset_vgpr;
  if (!append_moi_relocated_guest_access(
          words, bytes, candidate, target, guest_address_vgpr,
          reserve_two_address_replay_scratch
              ? std::optional<uint16_t>(static_cast<uint16_t>(scratch_vgpr + scratch_count - 1u))
              : std::nullopt,
          errors, guest_instruction_word_count))
    return std::nullopt;
  if (!sequence.resolve_branches(arch)) {
    errors.emplace_back("ConSan MOI inline-shadow probe could not resolve local control flow");
    return std::nullopt;
  }
  return words;
}

} // namespace rocjitsu::consan_moi_impl
