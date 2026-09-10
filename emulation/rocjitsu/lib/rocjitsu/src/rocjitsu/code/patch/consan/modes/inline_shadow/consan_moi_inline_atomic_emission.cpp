// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/modes/inline_shadow/consan_moi_inline_atomic_emission.h"
#include "rocjitsu/code/patch/consan/modes/inline_shadow/consan_moi_inline_register_layout.h"

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/consan/consan_growth_policy.h"
#include "rocjitsu/code/patch/consan/consan_instruction_semantics.h"
#include "rocjitsu/code/patch/consan/consan_moi_access_target.h"
#include "rocjitsu/code/patch/consan/consan_moi_dynamic_record_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_engine_contracts.h"
#include "rocjitsu/code/patch/consan/consan_moi_memory_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_native_abi.h"
#include "rocjitsu/code/patch/consan/consan_moi_placement_contracts.h"
#include "rocjitsu/code/patch/consan/consan_moi_probe_contracts.h"
#include "rocjitsu/code/patch/consan/consan_moi_prologue.h"
#include "rocjitsu/code/patch/consan/consan_moi_record_event_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_relocation.h"
#include "rocjitsu/code/patch/consan/consan_moi_report_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_runtime_workgroup_gate.h"
#include "rocjitsu/code/patch/consan/consan_moi_versioned_publication.h"
#include "rocjitsu/code/patch/consan/consan_placement.h"
#include "rocjitsu/code/patch/consan/consan_resource.h"
#include "rocjitsu/code/patch/instruction_sequence.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <ranges>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

namespace rocjitsu {

using consan_detail::MoiSpecialStateSgprs;
using consan_detail::MoiWorkgroupKeyRegisterPlan;
using consan_detail::range_overlaps;
using consan_detail::reject_optional_scratch_range_overlap;
using consan_moi_detail::append_atomic_fetch_add_one_u32;
using consan_moi_detail::append_atomic_load_u32;
using consan_moi_detail::append_compare_moi_report_dispatch_id_word;
using consan_moi_detail::append_load_u32_vgpr_at_offset;
using consan_moi_detail::append_moi_report_dispatch_id_word;
using consan_moi_detail::append_store_u32_vgpr_at_offset;
using consan_moi_detail::moi_has_runtime_hardware_dispatch_id;

namespace consan_moi_impl {

template <typename Entry>
[[nodiscard]] static bool
append_inline_atomic_table_address(std::vector<uint32_t> &words, uint64_t table_base,
                                   uint32_t table_capacity, uint16_t atomic_address_vgpr,
                                   uint16_t entry_address_vgpr, uint16_t hash_vgpr,
                                   uint16_t temporary_vgpr, rj_code_arch_t arch) {
  static_assert(std::is_same_v<Entry, ConSanMoiInlineAtomicReleaseSlot> ||
                std::is_same_v<Entry, ConSanMoiInlineCausalSnapshot>);
  static_assert(sizeof(ConSanMoiInlineAtomicReleaseSlot) == 32u);
  static_assert(sizeof(ConSanMoiInlineCausalSnapshot) == 40u);
  if (table_capacity == 0 || (table_capacity & (table_capacity - 1u)) != 0)
    return false;
  const ConSanTargetProfile *target = consan_target_profile(arch);
  if (target == nullptr)
    return false;
  std::vector<uint32_t> emitted;
  InstructionSequence sequence(emitted);
  sequence
      .append(
          instrumentation::build_v_lshrrev_b32(hash_vgpr, scalar_positive_inline_u32(2),
                                               atomic_address_vgpr, arch),
          instrumentation::build_v_xor_b32(
              hash_vgpr, vector_source_vgpr(static_cast<uint16_t>(atomic_address_vgpr + 1u)),
              hash_vgpr, arch),
          instrumentation::build_v_lshrrev_b32(temporary_vgpr, scalar_positive_inline_u32(16),
                                               hash_vgpr, arch),
          instrumentation::build_v_xor_b32(hash_vgpr, vector_source_vgpr(temporary_vgpr), hash_vgpr,
                                           arch),
          instrumentation::build_v_lshrrev_b32(temporary_vgpr, scalar_positive_inline_u32(8),
                                               hash_vgpr, arch),
          instrumentation::build_v_xor_b32(hash_vgpr, vector_source_vgpr(temporary_vgpr), hash_vgpr,
                                           arch),
          instrumentation::build_v_lshrrev_b32(temporary_vgpr, scalar_positive_inline_u32(4),
                                               hash_vgpr, arch),
          instrumentation::build_v_xor_b32(hash_vgpr, vector_source_vgpr(temporary_vgpr), hash_vgpr,
                                           arch),
          instrumentation::build_v_mul_lo_u32_literal(hash_vgpr, temporary_vgpr, 0x85ebca6bu,
                                                      hash_vgpr, arch),
          instrumentation::build_v_and_b32_literal(hash_vgpr, table_capacity - 1u, hash_vgpr, arch))
      .require(consan_detail::append_moi_indexed_address(emitted,
                                                         {
                                                             .table_address = table_base,
                                                             .stride_bytes = sizeof(Entry),
                                                             .address_vgpr = entry_address_vgpr,
                                                             .index_vgpr = hash_vgpr,
                                                         },
                                                         *target));
  if (!sequence.finish())
    return false;
  words.insert(words.end(), emitted.begin(), emitted.end());
  return true;
}

[[nodiscard]] static bool
append_inline_atomic_release_slot_address(std::vector<uint32_t> &words, uint64_t table_base,
                                          uint32_t table_capacity, uint16_t atomic_address_vgpr,
                                          uint16_t scratch_vgpr, rj_code_arch_t arch) {
  return append_inline_atomic_table_address<ConSanMoiInlineAtomicReleaseSlot>(
      words, table_base, table_capacity, atomic_address_vgpr, scratch_vgpr,
      static_cast<uint16_t>(scratch_vgpr + 2u), scratch_vgpr, arch);
}

[[nodiscard]] static bool
append_inline_causal_snapshot_address(std::vector<uint32_t> &words, uint64_t table_base,
                                      uint32_t table_capacity, uint16_t atomic_address_vgpr,
                                      uint16_t snapshot_address_vgpr, uint16_t hash_vgpr,
                                      uint16_t temporary_vgpr, rj_code_arch_t arch) {
  return append_inline_atomic_table_address<ConSanMoiInlineCausalSnapshot>(
      words, table_base, table_capacity, atomic_address_vgpr, snapshot_address_vgpr, hash_vgpr,
      temporary_vgpr, arch);
}

[[nodiscard]] bool append_inline_workgroup_source_vgpr(std::vector<uint32_t> &words,
                                                       uint16_t destination_vgpr,
                                                       const ConSanMoiWorkgroupSource &source,
                                                       rj_code_arch_t arch) {
  if (!source.is_well_formed())
    return false;
  if (!source.has_value()) {
    const auto zero = instrumentation::build_v_mov_b32_literal(destination_vgpr, 0u, arch);
    InstructionSequence sequence(words);
    return sequence.emit(zero);
  }
  return consan_detail::append_workgroup_source_value(words, source, destination_vgpr, arch);
}

[[nodiscard]] bool
append_inline_workgroup_key(std::vector<uint32_t> &words, const ConSanMoiWorkgroupSources &sources,
                            const MoiWorkgroupKeyRegisterPlan &registers, uint16_t key_vgpr,
                            uint16_t coordinate_vgpr, uint16_t value_vgpr,
                            uint16_t original_exec_save_offset, rj_code_arch_t arch) {
  if (!registers.is_well_formed())
    return false;
  const uint16_t exec_base = *registers.exec_save_sgpr;
  InstructionSequence sequence(words);
  sequence.append(instrumentation::build_s_mov_b64(
      static_cast<uint16_t>(exec_base + original_exec_save_offset), kAmdGpuExecLo, arch));

  if (registers.cached_key_vgpr) {
    words.push_back(
        build_v_mov_b32_e32(key_vgpr, vector_source_vgpr(*registers.cached_key_vgpr), arch));
    return sequence
        .append(
            instrumentation::build_v_cmp_ne_u32_vcc(scalar_positive_inline_u32(0), key_vgpr, arch),
            instrumentation::build_s_and_saveexec_b64(exec_base, kAmdGpuVccLo, arch))
        .finish();
  }
  if (registers.cached_key_sgpr) {
    words.push_back(build_v_mov_b32_e32(key_vgpr, *registers.cached_key_sgpr, arch));
    return sequence
        .append(
            instrumentation::build_v_cmp_ne_u32_vcc(scalar_positive_inline_u32(0), key_vgpr, arch),
            instrumentation::build_s_and_saveexec_b64(exec_base, kAmdGpuVccLo, arch))
        .finish();
  }

  const auto append_coordinate = [&](const ConSanMoiWorkgroupSource &source, uint32_t bits,
                                     uint32_t shift, uint16_t exec_save_offset, bool initialize) {
    sequence.require(append_inline_workgroup_source_vgpr(words, coordinate_vgpr, source, arch));
    if (source.scalar_src) {
      const uint32_t high_mask = ~((uint32_t{1} << bits) - 1u);
      sequence.append(
          instrumentation::build_v_and_b32_literal(value_vgpr, high_mask, coordinate_vgpr, arch),
          instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), value_vgpr, arch),
          instrumentation::build_s_and_saveexec_b64(
              static_cast<uint16_t>(exec_base + exec_save_offset), kAmdGpuVccLo, arch));
    }
    if (shift != 0u)
      sequence.append(instrumentation::build_v_lshlrev_b32(
          coordinate_vgpr, scalar_positive_inline_u32(shift), coordinate_vgpr, arch));
    if (initialize) {
      words.push_back(build_v_mov_b32_e32(key_vgpr, vector_source_vgpr(coordinate_vgpr), arch));
      return;
    }
    sequence.append(instrumentation::build_v_add_u32(key_vgpr, vector_source_vgpr(coordinate_vgpr),
                                                     key_vgpr, arch));
  };

  const bool has_z = sources.z.scalar_src.has_value();
  const bool has_y = sources.y.scalar_src.has_value();
  const uint32_t x_bits = has_z ? 8u : has_y ? 10u : 20u;
  const uint32_t y_bits = has_z ? 6u : 10u;
  append_coordinate(sources.x, x_bits, 0u, 0u, /*initialize=*/true);
  append_coordinate(sources.y, y_bits, x_bits, 2u, /*initialize=*/false);
  append_coordinate(sources.z, 6u, x_bits + y_bits, 4u, /*initialize=*/false);

  // The Y-coordinate predicate journal at +2 is dead after the packed key has
  // been formed. Reuse it here instead of +8, which the surrounding inline
  // transaction reserves for the guest VCC snapshot.
  const uint16_t valid_exec = static_cast<uint16_t>(exec_base + 2u);
  // Keep zero as the persistent invalid sentinel for lanes excluded by a
  // coordinate-width or reserved-key predicate. The caller still receives
  // EXEC narrowed to valid lanes, exactly as before.
  return sequence
      .append(
          instrumentation::build_v_mov_b32_literal(value_vgpr,
                                                   consan_moi_exact_shadow::max_generation, arch),
          instrumentation::build_v_cmp_ne_u32_vcc(vector_source_vgpr(key_vgpr), value_vgpr, arch),
          instrumentation::build_s_and_saveexec_b64(static_cast<uint16_t>(exec_base + 6u),
                                                    kAmdGpuVccLo, arch),
          instrumentation::build_v_add_u32(key_vgpr, scalar_positive_inline_u32(1), key_vgpr, arch),
          instrumentation::build_s_mov_b64(valid_exec, kAmdGpuExecLo, arch),
          instrumentation::build_s_andn2_b64(
              kAmdGpuExecLo, static_cast<uint16_t>(exec_base + original_exec_save_offset),
              valid_exec, arch),
          instrumentation::build_v_mov_b32_literal(key_vgpr, 0u, arch),
          instrumentation::build_s_mov_b64(kAmdGpuExecLo, valid_exec, arch))
      .finish();
}

[[nodiscard]] bool append_inline_acquired_token_slot_address(
    std::vector<uint32_t> &words, uint64_t table_base, uint32_t table_capacity,
    uint16_t workgroup_key_vgpr, uint16_t consumer_owner_vgpr, uint16_t producer_owner_vgpr,
    uint16_t consumer_epoch_vgpr, uint16_t slot_address_vgpr, uint16_t hash_vgpr,
    uint16_t temporary_vgpr, bool release_sequence, rj_code_arch_t arch) {
  if (table_capacity < 2u || (table_capacity & (table_capacity - 1u)) != 0)
    return false;
  const ConSanTargetProfile *target = consan_target_profile(arch);
  if (target == nullptr)
    return false;
  const uint32_t namespace_capacity = table_capacity / 2u;
  const uint64_t namespace_base =
      table_base + (release_sequence ? static_cast<uint64_t>(namespace_capacity) *
                                           sizeof(ConSanMoiInlineAcquiredEpochTokenSlot)
                                     : 0u);
  std::vector<uint32_t> emitted;
  InstructionSequence sequence(emitted);
  sequence.append(
      instrumentation::build_v_mul_lo_u32_literal(hash_vgpr, temporary_vgpr,
                                                  kConSanMoiInlineTokenWorkgroupMultiplier,
                                                  workgroup_key_vgpr, arch),
      instrumentation::build_v_mul_lo_u32_literal(temporary_vgpr, slot_address_vgpr,
                                                  kConSanMoiInlineTokenConsumerMultiplier,
                                                  consumer_owner_vgpr, arch),
      instrumentation::build_v_xor_b32(hash_vgpr, vector_source_vgpr(temporary_vgpr), hash_vgpr,
                                       arch),
      instrumentation::build_v_mul_lo_u32_literal(temporary_vgpr, slot_address_vgpr,
                                                  kConSanMoiInlineTokenProducerMultiplier,
                                                  producer_owner_vgpr, arch),
      instrumentation::build_v_xor_b32(hash_vgpr, vector_source_vgpr(temporary_vgpr), hash_vgpr,
                                       arch),
      instrumentation::build_v_mul_lo_u32_literal(temporary_vgpr, slot_address_vgpr,
                                                  kConSanMoiInlineTokenConsumerEpochMultiplier,
                                                  consumer_epoch_vgpr, arch),
      instrumentation::build_v_xor_b32(hash_vgpr, vector_source_vgpr(temporary_vgpr), hash_vgpr,
                                       arch));
  if (release_sequence)
    sequence.append(instrumentation::build_v_mov_b32_literal(
                        temporary_vgpr, kConSanMoiInlineTokenReleaseSequenceSalt, arch),
                    instrumentation::build_v_xor_b32(hash_vgpr, vector_source_vgpr(temporary_vgpr),
                                                     hash_vgpr, arch));
  sequence
      .append(instrumentation::build_v_lshrrev_b32(temporary_vgpr, scalar_positive_inline_u32(16),
                                                   hash_vgpr, arch),
              instrumentation::build_v_xor_b32(hash_vgpr, vector_source_vgpr(temporary_vgpr),
                                               hash_vgpr, arch),
              instrumentation::build_v_mul_lo_u32_literal(hash_vgpr, temporary_vgpr,
                                                          kConSanMoiInlineTokenAvalancheMultiplier0,
                                                          hash_vgpr, arch),
              instrumentation::build_v_lshrrev_b32(temporary_vgpr, scalar_positive_inline_u32(15),
                                                   hash_vgpr, arch),
              instrumentation::build_v_xor_b32(hash_vgpr, vector_source_vgpr(temporary_vgpr),
                                               hash_vgpr, arch),
              instrumentation::build_v_mul_lo_u32_literal(hash_vgpr, temporary_vgpr,
                                                          kConSanMoiInlineTokenAvalancheMultiplier1,
                                                          hash_vgpr, arch),
              instrumentation::build_v_lshrrev_b32(temporary_vgpr, scalar_positive_inline_u32(16),
                                                   hash_vgpr, arch),
              instrumentation::build_v_xor_b32(hash_vgpr, vector_source_vgpr(temporary_vgpr),
                                               hash_vgpr, arch),
              instrumentation::build_v_and_b32_literal(hash_vgpr, namespace_capacity - 1u,
                                                       hash_vgpr, arch))
      .require(consan_detail::append_moi_indexed_address(
          emitted,
          {
              .table_address = namespace_base,
              .stride_bytes = sizeof(ConSanMoiInlineAcquiredEpochTokenSlot),
              .address_vgpr = slot_address_vgpr,
              .index_vgpr = hash_vgpr,
          },
          *target));
  if (!sequence.finish())
    return false;
  words.insert(words.end(), emitted.begin(), emitted.end());
  return true;
}

// Publishes the direct acquire edge and every inherited snapshot edge as one
// reservation transaction. The reservation-version field is an odd-state
// rollback journal containing the prior even version; it is cleared before
// commit and can never authorize a reader while the slot remains odd.
[[nodiscard]] bool append_inline_acquired_token_transaction(
    std::vector<uint32_t> &words, const MoiInlineAtomicEmissionPlan &plan, uint16_t scratch_vgpr,
    uint16_t source_address_vgpr, uint16_t workgroup_key_vgpr, uint16_t direct_owner_vgpr,
    uint16_t direct_epoch_vgpr, uint16_t source_version_vgpr, uint16_t snapshot_count_vgpr,
    uint16_t snapshot_owners_vgpr, uint16_t snapshot_epochs_vgpr, uint64_t token_table_base,
    uint32_t token_table_capacity, bool release_sequence,
    std::optional<uint16_t> inherited_first_exec, rj_code_arch_t arch) {
  const uint16_t base = scratch_vgpr;
  const bool aligned_cas_pair = plan.requires_aligned_flat_compare_swap_data_pair;
  const uint16_t value = static_cast<uint16_t>(base + (aligned_cas_pair ? 18u : 19u));
  const uint16_t expected = static_cast<uint16_t>(base + (aligned_cas_pair ? 19u : 20u));
  const uint16_t hash = static_cast<uint16_t>(base + 21u);
  const uint16_t saved_direct_owner = static_cast<uint16_t>(base + (aligned_cas_pair ? 22u : 17u));
  const uint16_t saved_direct_epoch = static_cast<uint16_t>(base + (aligned_cas_pair ? 23u : 18u));
  const uint16_t saved_source_version = static_cast<uint16_t>(base + 8u);
  const uint16_t exec_base = plan.exec_save_sgpr;
  const uint16_t winners = static_cast<uint16_t>(exec_base + 16u);
  const uint16_t needed = static_cast<uint16_t>(exec_base + 18u);
  // The enclosing acquire helper's +6 workgroup predicate and +14 owner
  // predicate are dead before this transaction begins. Reuse those pairs for
  // the transaction-wide eligible and skipped masks.
  const uint16_t skipped = static_cast<uint16_t>(exec_base + 14u);
  const uint16_t eligible = static_cast<uint16_t>(exec_base + 6u);

  InstructionSequence sequence(words);
  MoiPublicationExec exec_masks(sequence, exec_base, arch);
  const auto restore_exec = [&](uint16_t source) { sequence.require(exec_masks.restore(source)); };
  const auto save_exec = [&](uint16_t destination) {
    sequence.require(exec_masks.save(destination));
  };
  const auto narrow_vcc = [&] { sequence.require(exec_masks.narrow_vcc()); };
  const auto require_equal = [&](size_t offset, uint16_t expected_vgpr) {
    sequence.require(append_load_u32_vgpr_at_offset(words, base, offset, value, arch))
        .require(exec_masks.narrow(instrumentation::build_v_cmp_eq_u32_vcc(
            vector_source_vgpr(expected_vgpr), value, arch)));
  };
  const auto select_entry = [&](uint16_t index) {
    restore_exec(winners);
    if (index == 0) {
      save_exec(needed);
      return;
    }
    sequence
        .append(instrumentation::build_v_mov_b32_literal(value, static_cast<uint32_t>(index - 1u),
                                                         arch))
        .require(exec_masks.narrow(instrumentation::build_v_cmp_gt_u32_vcc(
            vector_source_vgpr(snapshot_count_vgpr), value, arch)));
    save_exec(needed);
    restore_exec(winners);
    sequence
        .require(exec_masks.narrow(instrumentation::build_v_cmp_gt_u32_vcc(
            scalar_positive_inline_u32(index), snapshot_count_vgpr, arch)))
        .require(exec_masks.narrow(instrumentation::build_v_cmp_eq_u32_vcc(
            scalar_positive_inline_u32(0), static_cast<uint16_t>(snapshot_owners_vgpr + index - 1u),
            arch)))
        .require(exec_masks.narrow(instrumentation::build_v_cmp_eq_u32_vcc(
            scalar_positive_inline_u32(0), static_cast<uint16_t>(snapshot_epochs_vgpr + index - 1u),
            arch)));
    save_exec(skipped);
    restore_exec(needed);
  };
  const auto select_identity = [&](uint16_t index) {
    if (index == 0) {
      words.push_back(
          build_v_mov_b32_e32(direct_owner_vgpr, vector_source_vgpr(saved_direct_owner), arch));
      words.push_back(
          build_v_mov_b32_e32(direct_epoch_vgpr, vector_source_vgpr(saved_direct_epoch), arch));
      return;
    }
    words.push_back(build_v_mov_b32_e32(
        direct_owner_vgpr,
        vector_source_vgpr(static_cast<uint16_t>(snapshot_owners_vgpr + index - 1u)), arch));
    words.push_back(build_v_mov_b32_e32(
        direct_epoch_vgpr,
        vector_source_vgpr(static_cast<uint16_t>(snapshot_epochs_vgpr + index - 1u)), arch));
    sequence.require(exec_masks.narrow(instrumentation::build_v_cmp_ne_u32_vcc(
        scalar_positive_inline_u32(0), direct_owner_vgpr, arch)));
    const auto owner_not_consumer = instrumentation::build_v_cmp_ne_u32_vcc(
        vector_source_vgpr(plan.owner_epoch_vgprs.owner), direct_owner_vgpr, arch);
    const auto owner_not_releaser = instrumentation::build_v_cmp_ne_u32_vcc(
        vector_source_vgpr(saved_direct_owner), direct_owner_vgpr, arch);
    const auto epoch_nonzero = instrumentation::build_v_cmp_ne_u32_vcc(
        scalar_positive_inline_u32(0), direct_epoch_vgpr, arch);
    const auto epoch_limit = instrumentation::build_v_mov_b32_literal(
        value, consan_moi_exact_shadow::max_epoch + 1u, arch);
    const auto epoch_in_range =
        instrumentation::build_v_cmp_gt_u32_vcc(vector_source_vgpr(value), direct_epoch_vgpr, arch);
    if (!owner_not_consumer || !owner_not_releaser || !epoch_nonzero || !epoch_limit ||
        !epoch_in_range) {
      sequence.require(false);
      return;
    }
    words.insert(words.end(), epoch_limit->begin(), epoch_limit->end());
    // A reusable partial barrier can legitimately carry the acquiring
    // consumer's prior epoch in the producer's inherited frontier. That edge
    // is redundant for this consumer, not malformed: omit it while retaining
    // the direct consumer<-producer token. Keep every other malformed-entry
    // predicate fail-closed below.
    save_exec(static_cast<uint16_t>(exec_base + 4u));
    words.push_back(*owner_not_consumer);
    narrow_vcc();
    const auto self_ancestor = instrumentation::build_s_xor_b64(
        needed, static_cast<uint16_t>(exec_base + 4u), kAmdGpuExecLo, arch);
    const auto retain_skipped = instrumentation::build_s_xor_b64(skipped, skipped, needed, arch);
    if (!self_ancestor || !retain_skipped) {
      sequence.require(false);
      return;
    }
    words.push_back(*self_ancestor);
    words.push_back(*retain_skipped);
    for (const auto instruction : {*owner_not_releaser, *epoch_nonzero, *epoch_in_range}) {
      words.push_back(instruction);
      narrow_vcc();
    }
    if (index > 1u) {
      const auto canonical = instrumentation::build_v_cmp_gt_u32_vcc(
          vector_source_vgpr(direct_owner_vgpr),
          static_cast<uint16_t>(snapshot_owners_vgpr + index - 2u), arch);
      if (!canonical) {
        sequence.require(false);
        return;
      }
      words.push_back(*canonical);
      narrow_vcc();
    }
  };
  const auto derive_slot = [&] {
    sequence.require(append_inline_acquired_token_slot_address(
        words, token_table_base, token_table_capacity, workgroup_key_vgpr,
        plan.owner_epoch_vgprs.owner, direct_owner_vgpr, plan.owner_epoch_vgprs.epoch, base, hash,
        value, release_sequence, arch));
  };

  words.push_back(
      build_v_mov_b32_e32(saved_direct_owner, vector_source_vgpr(direct_owner_vgpr), arch));
  words.push_back(
      build_v_mov_b32_e32(saved_direct_epoch, vector_source_vgpr(direct_epoch_vgpr), arch));
  words.push_back(
      build_v_mov_b32_e32(saved_source_version, vector_source_vgpr(source_version_vgpr), arch));
  save_exec(eligible);
  save_exec(winners);

  // Phase one: reserve every required destination odd. Empty and exact stable
  // pair slots are accepted; all other states remove the lane from winners.
  for (uint16_t index = 0; index < 5u; ++index) {
    select_entry(index);
    select_identity(index);
    derive_slot();
    sequence.require(append_load_u32_vgpr_at_offset(
        words, base, offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, version), value, arch));
    save_exec(needed);
    words.push_back(build_v_mov_b32_e32(expected, vector_source_vgpr(value), arch));

    sequence.require(exec_masks.narrow(
        instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), value, arch)));
    for (size_t offset : kConSanMoiInlineAcquiredEpochTokenPayloadOffsets) {
      sequence.require(append_load_u32_vgpr_at_offset(words, base, offset, value, arch))
          .require(exec_masks.narrow(
              instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), value, arch)));
    }
    // `winners` is dead after select_entry materializes this iteration's
    // `needed` mask. Reuse it for the empty-slot subset so +2 retains the
    // incoming same-owner provenance through both transaction phases.
    save_exec(winners);
    restore_exec(needed);

    sequence
        .require(exec_masks.narrow(
            instrumentation::build_v_cmp_ne_u32_vcc(scalar_positive_inline_u32(0), expected, arch)))
        .append(instrumentation::build_v_and_b32_literal(value, 1u, expected, arch))
        .require(exec_masks.narrow(
            instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), value, arch)));
    require_equal(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, workgroup_key),
                  workgroup_key_vgpr);
    require_equal(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, consumer_owner_id),
                  plan.owner_epoch_vgprs.owner);
    require_equal(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, producer_owner_id),
                  direct_owner_vgpr);
    sequence
        .require(append_load_u32_vgpr_at_offset(
            words, base, offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, producer_epoch_plus_one),
            value, arch))
        .append(instrumentation::build_v_mov_b32_literal(
            hash, consan_moi_exact_shadow::max_epoch + 1u, arch))
        .require(exec_masks.narrow(
            instrumentation::build_v_cmp_ne_u32_vcc(scalar_positive_inline_u32(0), value, arch)))
        .require(exec_masks.narrow(
            instrumentation::build_v_cmp_gt_u32_vcc(vector_source_vgpr(hash), value, arch)))
        .require(append_load_u32_vgpr_at_offset(
            words, base, offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, kind), value, arch))
        .require(exec_masks.narrow(
            instrumentation::build_v_cmp_gt_u32_vcc(scalar_positive_inline_u32(3), value, arch)))
        .require(append_load_u32_vgpr_at_offset(
            words, base, offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, dispatch_id), value,
            arch));
    // Slot derivation is complete. `hash` is dead here and its next use
    // overwrites it with the source-version parity bit.
    sequence.require(append_compare_moi_report_dispatch_id_word(words, plan.dispatch_id, value,
                                                                hash,
                                                                /*high_word=*/false, arch));
    narrow_vcc();
    sequence.require(append_load_u32_vgpr_at_offset(
        words, base,
        offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, dispatch_id) + sizeof(uint32_t), value,
        arch));
    sequence.require(append_compare_moi_report_dispatch_id_word(words, plan.dispatch_id, value,
                                                                hash,
                                                                /*high_word=*/true, arch));
    narrow_vcc();
    sequence.require(append_load_u32_vgpr_at_offset(
        words, base, offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, source_release_address), value,
        arch));
    // Requiring the low word to be nonzero is a conservative generated-code
    // subset of the host classifier: aligned 4-GiB boundary addresses fail
    // closed instead of becoming ordering authority.
    sequence
        .require(exec_masks.narrow(
            instrumentation::build_v_cmp_ne_u32_vcc(scalar_positive_inline_u32(0), value, arch)))
        .require(append_load_u32_vgpr_at_offset(
            words, base, offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, source_release_version),
            value, arch))
        .require(exec_masks.narrow(
            instrumentation::build_v_cmp_ne_u32_vcc(scalar_positive_inline_u32(0), value, arch)))
        .append(instrumentation::build_v_and_b32_literal(hash, 1u, value, arch))
        .require(exec_masks.narrow(
            instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), hash, arch)))
        .require(append_load_u32_vgpr_at_offset(
            words, base, offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, consumer_epoch_plus_one),
            value, arch))
        .append(instrumentation::build_v_mov_b32_literal(
            hash, consan_moi_exact_shadow::max_epoch + 1u, arch))
        .require(exec_masks.narrow(
            instrumentation::build_v_cmp_ne_u32_vcc(scalar_positive_inline_u32(0), value, arch)))
        .require(exec_masks.narrow(
            instrumentation::build_v_cmp_gt_u32_vcc(vector_source_vgpr(hash), value, arch)))
        .append(instrumentation::build_v_add_u32(hash, scalar_positive_inline_u32(1),
                                                 plan.owner_epoch_vgprs.epoch, arch),
                instrumentation::build_v_min_u32_literal(hash, consan_moi_exact_shadow::max_epoch,
                                                         hash, arch))
        .require(exec_masks.narrow(
            instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(hash), value, arch)))
        .require(append_load_u32_vgpr_at_offset(
            words, base, offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, reservation_version),
            value, arch))
        .require(exec_masks.narrow(
            instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), value, arch)));
    save_exec(static_cast<uint16_t>(exec_base + 4u));
    const auto valid_union = instrumentation::build_s_xor_b64(
        kAmdGpuExecLo, winners, static_cast<uint16_t>(exec_base + 4u), arch);
    if (!valid_union) {
      sequence.require(false);
      continue;
    }
    words.push_back(*valid_union);
    sequence
        .require(append_moi_version_transition(
            words, sequence, exec_masks, base,
            offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, version), expected, value, expected,
            /*desired_delta=*/1u, /*expected_delta=*/0u, arch))
        .require(append_store_u32_vgpr_at_offset(
            words, base, offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, reservation_version),
            expected, arch));
    save_exec(needed);
    const auto combine = index == 0
                             ? instrumentation::build_s_mov_b64(winners, needed, arch)
                             : instrumentation::build_s_xor_b64(winners, skipped, needed, arch);
    if (combine)
      words.push_back(*combine);
    else
      sequence.require(false);
  }

  sequence.append(instrumentation::build_s_andn2_b64(kAmdGpuExecLo, eligible, winners, arch))
      .require(append_atomic_fetch_add_one_u32(
          words,
          plan.report_buffer_address + offsetof(ConSanMoiReportHeader, inline_overflow_count),
          value, aligned_cas_pair ? base : hash, arch));

  // Roll back every journaled reservation for failed lanes. A CAS whose slot
  // was never claimed simply loses and leaves foreign evidence untouched.
  for (uint16_t index = 0; index < 5u; ++index) {
    restore_exec(eligible);
    sequence.append(instrumentation::build_s_andn2_b64(kAmdGpuExecLo, eligible, winners, arch));
    if (index != 0) {
      sequence
          .append(instrumentation::build_v_mov_b32_literal(value, static_cast<uint32_t>(index - 1u),
                                                           arch))
          .require(exec_masks.narrow(instrumentation::build_v_cmp_gt_u32_vcc(
              vector_source_vgpr(snapshot_count_vgpr), value, arch)));
    }
    select_identity(index);
    derive_slot();
    sequence.require(append_moi_publication_loads(
        words, base,
        {{offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, reservation_version), expected},
         {offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, version), value}},
        arch));
    sequence.append(instrumentation::build_v_and_b32_literal(hash, 1u, value, arch))
        .require(exec_masks.narrow(
            instrumentation::build_v_cmp_ne_u32_vcc(scalar_positive_inline_u32(0), hash, arch)))
        .require(append_moi_version_transition(
            words, sequence, exec_masks, base,
            offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, version), expected, value, expected,
            /*desired_delta=*/0u, /*expected_delta=*/1u, arch));
  }

  // Phase two: winners fill every payload, drain, then commit each odd version
  // even. Readers therefore see either the old frontier or complete tokens.
  for (uint16_t index = 0; index < 5u; ++index) {
    restore_exec(winners);
    if (index != 0) {
      sequence
          .append(instrumentation::build_v_mov_b32_literal(value, static_cast<uint32_t>(index - 1u),
                                                           arch))
          .require(exec_masks.narrow(instrumentation::build_v_cmp_gt_u32_vcc(
              vector_source_vgpr(snapshot_count_vgpr), value, arch)));
    }
    select_identity(index);
    derive_slot();
    sequence
        .require(append_moi_publication_stores(
            words, base,
            {{offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, consumer_owner_id),
              plan.owner_epoch_vgprs.owner},
             {offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, producer_owner_id),
              direct_owner_vgpr},
             {offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, producer_epoch_plus_one),
              direct_epoch_vgpr},
             {offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, workgroup_key), workgroup_key_vgpr}},
            arch))
        .append(instrumentation::build_v_mov_b32_literal(
            value,
            release_sequence
                ? static_cast<uint32_t>(ConSanMoiInlineTokenEvidenceKind::ReleaseSequence)
            : index == 0 ? static_cast<uint32_t>(ConSanMoiInlineTokenEvidenceKind::Direct)
                         : static_cast<uint32_t>(ConSanMoiInlineTokenEvidenceKind::Inherited),
            arch));
    if (!release_sequence && index == 0u && inherited_first_exec) {
      // Same-owner acquires omit their direct self edge and promote the first
      // immutable snapshot ancestor into slot zero. Preserve that distinction
      // in the durable token even though direct and inherited edges authorize
      // the same access-ordering relation.
      // +4 is dead after phase one. +2 remains the incoming same-owner mask:
      // phase one deliberately reuses `winners` and `needed` for its
      // empty/exact and self-ancestor temporaries instead of clobbering it.
      const uint16_t kind_exec = static_cast<uint16_t>(exec_base + 4u);
      save_exec(kind_exec);
      sequence.append(
          instrumentation::build_s_and_b64(kAmdGpuExecLo, kAmdGpuExecLo, *inherited_first_exec,
                                           arch),
          instrumentation::build_v_mov_b32_literal(
              value, static_cast<uint32_t>(ConSanMoiInlineTokenEvidenceKind::Inherited), arch));
      restore_exec(kind_exec);
    }
    sequence.require(append_moi_publication_stores(
        words, base,
        {{offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, kind), value},
         {offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, source_release_address),
          source_address_vgpr},
         {offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, source_release_address) +
              sizeof(uint32_t),
          static_cast<uint16_t>(source_address_vgpr + 1u)},
         {offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, source_release_version),
          saved_source_version}},
        arch));
    sequence
        .append(instrumentation::build_v_add_u32(value, scalar_positive_inline_u32(1),
                                                 plan.owner_epoch_vgprs.epoch, arch),
                instrumentation::build_v_min_u32_literal(value, consan_moi_exact_shadow::max_epoch,
                                                         value, arch))
        .require(append_store_u32_vgpr_at_offset(
            words, base, offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, consumer_epoch_plus_one),
            value, arch))
        .append(instrumentation::build_v_mov_b32_literal(value, 0, arch))
        .require(append_store_u32_vgpr_at_offset(
            words, base, offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, reservation_version),
            value, arch))
        .require(append_moi_report_dispatch_id_word(words, plan.dispatch_id, value,
                                                    /*high_word=*/false, arch))
        .require(append_store_u32_vgpr_at_offset(
            words, base, offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, dispatch_id), value, arch))
        .require(append_moi_report_dispatch_id_word(words, plan.dispatch_id, value,
                                                    /*high_word=*/true, arch))
        .require(append_store_u32_vgpr_at_offset(
            words, base,
            offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, dispatch_id) + sizeof(uint32_t), value,
            arch))
        .require(append_load_u32_vgpr_at_offset(
            words, base, offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, version), value, arch))
        .append(instrumentation::build_s_wait_global_store0(arch));
    words.push_back(build_v_mov_b32_e32(expected, vector_source_vgpr(value), arch));
    sequence.require(append_moi_version_transition(
        words, sequence, exec_masks, base, offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, version),
        expected, value, expected,
        /*desired_delta=*/1u, /*expected_delta=*/0u, arch));
  }
  restore_exec(winners);
  return sequence.finish();
}

[[nodiscard]] bool append_inline_atomic_acquire_import(
    std::vector<uint32_t> &words, const MoiInlineAtomicEmissionPlan &plan, uint16_t scratch_vgpr,
    uint16_t address_vgpr, uint16_t workgroup_key_vgpr, uint16_t producer_owner_vgpr,
    uint16_t token_value_vgpr, uint16_t temporary_vgpr, uint64_t snapshot_table_base,
    uint32_t snapshot_table_capacity, uint64_t token_table_base, uint32_t token_table_capacity,
    bool source_slot_claimed, bool release_sequence_only, bool inline_access_present,
    rj_code_arch_t arch, std::vector<std::string> &errors) {
  const uint16_t value_vgpr = static_cast<uint16_t>(scratch_vgpr + 2u);
  const uint16_t exec_base = plan.exec_save_sgpr;
  InstructionSequence sequence(words);
  MoiPublicationExec valid_exec_masks(sequence, exec_base, arch);
  MoiPublicationExec low_match_exec_masks(sequence, static_cast<uint16_t>(exec_base + 2u), arch);
  MoiPublicationExec high_match_exec_masks(sequence, static_cast<uint16_t>(exec_base + 4u), arch);
  MoiPublicationExec workgroup_match_exec_masks(sequence, static_cast<uint16_t>(exec_base + 6u),
                                                arch);
  MoiPublicationExec other_owner_exec_masks(sequence, static_cast<uint16_t>(exec_base + 14u), arch);
  MoiEmissionRequirement require_emission(sequence, errors);
  // The low/high address-match journals are dead once owner qualification
  // begins. Reuse them for the longer-lived same-owner and validated masks.
  // InlineShadow reserves +8:+9 for guest VCC and +10 for guest SCC; using
  // either range here would corrupt condition state that remains live across
  // an ordered acquire sequence.
  const uint16_t same_owner_exec = static_cast<uint16_t>(exec_base + 2u);
  const uint16_t validated_exec = static_cast<uint16_t>(exec_base + 4u);

  require_emission(append_atomic_load_u32(words, scratch_vgpr, value_vgpr, arch),
                   "ConSan MOI inline atomic acquire patch could not load release version");
  const uint16_t version_before = static_cast<uint16_t>(
      scratch_vgpr + inline_register_layout::AtomicOrdering::release_version_before);
  if (source_slot_claimed)
    sequence.append(instrumentation::build_v_add_u32(temporary_vgpr, scalar_positive_inline_u32(1),
                                                     version_before, arch));
  require_emission(
      valid_exec_masks.narrow(instrumentation::build_v_cmp_eq_u32_vcc(
          vector_source_vgpr(source_slot_claimed ? temporary_vgpr : version_before), value_vgpr,
          arch)) &&
          sequence.emit(
              instrumentation::build_v_and_b32_literal(temporary_vgpr, 1u, value_vgpr, arch)) &&
          valid_exec_masks.narrow(source_slot_claimed
                                      ? instrumentation::build_v_cmp_ne_u32_vcc(
                                            scalar_positive_inline_u32(0), temporary_vgpr, arch)
                                      : instrumentation::build_v_cmp_eq_u32_vcc(
                                            scalar_positive_inline_u32(0), temporary_vgpr, arch)) &&
          valid_exec_masks.narrow(instrumentation::build_v_cmp_ne_u32_vcc(
              scalar_positive_inline_u32(0), value_vgpr, arch)),
      "ConSan MOI inline atomic acquire patch could not classify release version");

  require_emission(
      append_load_u32_vgpr_at_offset(words, scratch_vgpr,
                                     offsetof(ConSanMoiInlineAtomicReleaseSlot, atomic_address),
                                     value_vgpr, arch),
      "ConSan MOI inline atomic acquire patch could not load address low");
  require_emission(low_match_exec_masks.narrow(instrumentation::build_v_cmp_eq_u32_vcc(
                       vector_source_vgpr(address_vgpr), value_vgpr, arch)),
                   "ConSan MOI inline atomic acquire patch could not compare address low");

  require_emission(
      append_load_u32_vgpr_at_offset(words, scratch_vgpr,
                                     offsetof(ConSanMoiInlineAtomicReleaseSlot, atomic_address) +
                                         sizeof(uint32_t),
                                     value_vgpr, arch),
      "ConSan MOI inline atomic acquire patch could not load address high");
  require_emission(
      high_match_exec_masks.narrow(instrumentation::build_v_cmp_eq_u32_vcc(
          vector_source_vgpr(static_cast<uint16_t>(address_vgpr + 1u)), value_vgpr, arch)),
      "ConSan MOI inline atomic acquire patch could not compare address high");

  require_emission(append_load_u32_vgpr_at_offset(
                       words, scratch_vgpr,
                       offsetof(ConSanMoiInlineAtomicReleaseSlot, workgroup_key), value_vgpr, arch),
                   "ConSan MOI inline atomic acquire patch could not load workgroup key");
  require_emission(workgroup_match_exec_masks.narrow(instrumentation::build_v_cmp_eq_u32_vcc(
                       vector_source_vgpr(workgroup_key_vgpr), value_vgpr, arch)),
                   "ConSan MOI inline atomic acquire patch could not compare workgroup key");

  require_emission(append_load_u32_vgpr_at_offset(
                       words, scratch_vgpr, offsetof(ConSanMoiInlineAtomicReleaseSlot, owner_id),
                       producer_owner_vgpr, arch),
                   "ConSan MOI inline atomic acquire patch could not load owner");
  require_emission(
      other_owner_exec_masks.narrow(instrumentation::build_v_cmp_ne_u32_vcc(
          vector_source_vgpr(plan.owner_epoch_vgprs.owner), producer_owner_vgpr, arch)) &&
          sequence.emit_all(
              instrumentation::build_s_andn2_b64(
                  same_owner_exec, static_cast<uint16_t>(exec_base + 14u), kAmdGpuExecLo, arch),
              instrumentation::build_s_mov_b64(kAmdGpuExecLo,
                                               static_cast<uint16_t>(exec_base + 14u), arch)),
      "ConSan MOI inline atomic acquire patch could not compare owner");

  // A polling acquire can observe an empty, unstable, mismatched, or
  // otherwise unusable predecessor. Its validated EXEC mask is then empty.
  // Same-owner predecessors remain eligible because their immutable snapshot
  // can carry the head of a release sequence. Skip the complete causal-import
  // transaction when no lane has either form of authority.
  const InstructionSequence::Label restore_workgroup = sequence.make_label();
  sequence.branch(restore_workgroup, InstructionSequence::BranchKind::ExecZero);

  require_emission(
      append_load_u32_vgpr_at_offset(words, scratch_vgpr,
                                     offsetof(ConSanMoiInlineAtomicReleaseSlot, epoch_plus_one),
                                     token_value_vgpr, arch),
      "ConSan MOI inline atomic acquire patch could not load epoch_plus_one");
  sequence.require(append_load_u32_vgpr_at_offset(
      words, scratch_vgpr, offsetof(ConSanMoiInlineAtomicReleaseSlot, dispatch_id), value_vgpr,
      arch));
  // Release-slot addressing is complete. `temporary_vgpr` is dead here and
  // the later snapshot construction overwrites it before any read.
  sequence
      .require(append_compare_moi_report_dispatch_id_word(words, plan.dispatch_id, value_vgpr,
                                                          temporary_vgpr,
                                                          /*high_word=*/false, arch))
      .require(valid_exec_masks.narrow_vcc())
      .require(append_load_u32_vgpr_at_offset(
          words, scratch_vgpr,
          offsetof(ConSanMoiInlineAtomicReleaseSlot, dispatch_id) + sizeof(uint32_t), value_vgpr,
          arch))
      .require(append_compare_moi_report_dispatch_id_word(words, plan.dispatch_id, value_vgpr,
                                                          temporary_vgpr,
                                                          /*high_word=*/true, arch))
      .require(valid_exec_masks.narrow_vcc());
  // Swap address and temporary lifetimes when the target requires the address
  // pair to remain even-aligned. The token transaction reuses this space only
  // after the snapshot loads have completed.
  const bool aligned_cas_pair = plan.requires_aligned_flat_compare_swap_data_pair;
  const uint16_t snapshot_address =
      static_cast<uint16_t>(scratch_vgpr + (aligned_cas_pair ? 18u : 17u));
  const uint16_t snapshot_temporary =
      static_cast<uint16_t>(scratch_vgpr + (aligned_cas_pair ? 17u : 19u));
  const uint16_t snapshot_hash = static_cast<uint16_t>(scratch_vgpr + 21u);
  const uint16_t snapshot_count = static_cast<uint16_t>(scratch_vgpr + 7u);
  // A claimed release journals its predecessor version in scratch + 8 before
  // entering this helper. An acquire that becomes ineligible before the token
  // transaction must retain that journal so the enclosing publisher can
  // commit its own odd reservation. The generic value register is dead across
  // this immediate flags check and is overwritten below, so use it instead.
  const uint16_t snapshot_flags =
      source_slot_claimed ? value_vgpr : static_cast<uint16_t>(scratch_vgpr + 8u);
  sequence
      .require(append_inline_causal_snapshot_address(
          words, snapshot_table_base, snapshot_table_capacity, address_vgpr, snapshot_address,
          snapshot_temporary, snapshot_hash, arch))
      .require(append_load_u32_vgpr_at_offset(words, snapshot_address,
                                              offsetof(ConSanMoiInlineCausalSnapshot, entry_count),
                                              snapshot_count, arch))
      .require(append_load_u32_vgpr_at_offset(words, snapshot_address,
                                              offsetof(ConSanMoiInlineCausalSnapshot, flags),
                                              snapshot_flags, arch))
      .require(valid_exec_masks.narrow(instrumentation::build_v_cmp_eq_u32_vcc(
          scalar_positive_inline_u32(0), snapshot_flags, arch)))
      .require(valid_exec_masks.narrow(instrumentation::build_v_cmp_gt_u32_vcc(
          scalar_positive_inline_u32(kConSanMoiInlineCausalSnapshotEntryCapacity + 1u),
          snapshot_count, arch)));
  for (uint16_t i = 0; i < kConSanMoiInlineCausalSnapshotEntryCapacity; ++i) {
    const size_t entry_offset = offsetof(ConSanMoiInlineCausalSnapshot, entries) +
                                i * sizeof(ConSanMoiInlineCausalSnapshotEntry);
    sequence
        .require(append_load_u32_vgpr_at_offset(words, snapshot_address, entry_offset,
                                                static_cast<uint16_t>(scratch_vgpr + 9u + i), arch))
        .require(
            append_load_u32_vgpr_at_offset(words, snapshot_address, entry_offset + sizeof(uint32_t),
                                           static_cast<uint16_t>(scratch_vgpr + 13u + i), arch));
  }
  sequence.require(append_atomic_load_u32(words, scratch_vgpr, value_vgpr, arch));
  if (source_slot_claimed)
    sequence.append(instrumentation::build_v_add_u32(temporary_vgpr, scalar_positive_inline_u32(1),
                                                     version_before, arch));
  require_emission(valid_exec_masks.narrow(instrumentation::build_v_cmp_eq_u32_vcc(
                       vector_source_vgpr(source_slot_claimed ? temporary_vgpr : version_before),
                       value_vgpr, arch)) &&
                       valid_exec_masks.narrow(instrumentation::build_v_cmp_ne_u32_vcc(
                           scalar_positive_inline_u32(0), token_value_vgpr, arch)) &&
                       sequence.emit(instrumentation::build_v_mov_b32_literal(
                           value_vgpr, consan_moi_exact_shadow::max_epoch + 1u, arch)) &&
                       valid_exec_masks.narrow(instrumentation::build_v_cmp_gt_u32_vcc(
                           vector_source_vgpr(value_vgpr), token_value_vgpr, arch)),
                   "ConSan MOI inline atomic acquire patch could not validate token epoch");

  // Convert a same-owner acquire into a snapshot-only import. Slot zero takes
  // the first inherited ancestor, the remaining canonical entries shift down,
  // and the self release is never published as a token. Other-owner lanes keep
  // their direct entry and unmodified snapshot. Empty same-owner snapshots do
  // not establish an inter-owner edge and are removed before publication.
  sequence
      .append(
          instrumentation::build_s_mov_b64(validated_exec, kAmdGpuExecLo, arch),
          instrumentation::build_s_and_b64(same_owner_exec, same_owner_exec, kAmdGpuExecLo, arch),
          instrumentation::build_s_mov_b64(kAmdGpuExecLo, same_owner_exec, arch))
      .require(valid_exec_masks.narrow(instrumentation::build_v_cmp_ne_u32_vcc(
          scalar_positive_inline_u32(0), snapshot_count, arch)))
      .append(instrumentation::build_s_mov_b64(static_cast<uint16_t>(exec_base + 14u),
                                               kAmdGpuExecLo, arch));
  words.push_back(build_v_mov_b32_e32(
      producer_owner_vgpr, vector_source_vgpr(static_cast<uint16_t>(scratch_vgpr + 9u)), arch));
  words.push_back(build_v_mov_b32_e32(
      token_value_vgpr, vector_source_vgpr(static_cast<uint16_t>(scratch_vgpr + 13u)), arch));
  for (uint16_t i = 0; i + 1u < kConSanMoiInlineCausalSnapshotEntryCapacity; ++i) {
    words.push_back(build_v_mov_b32_e32(
        static_cast<uint16_t>(scratch_vgpr + 9u + i),
        vector_source_vgpr(static_cast<uint16_t>(scratch_vgpr + 10u + i)), arch));
    words.push_back(build_v_mov_b32_e32(
        static_cast<uint16_t>(scratch_vgpr + 13u + i),
        vector_source_vgpr(static_cast<uint16_t>(scratch_vgpr + 14u + i)), arch));
  }
  sequence.append(instrumentation::build_v_mov_b32_literal(
                      temporary_vgpr, std::numeric_limits<uint32_t>::max(), arch),
                  instrumentation::build_v_add_u32(
                      snapshot_count, vector_source_vgpr(temporary_vgpr), snapshot_count, arch),
                  instrumentation::build_v_mov_b32_literal(temporary_vgpr, 0u, arch));
  words.push_back(build_v_mov_b32_e32(
      static_cast<uint16_t>(scratch_vgpr + 9u + kConSanMoiInlineCausalSnapshotEntryCapacity - 1u),
      vector_source_vgpr(temporary_vgpr), arch));
  words.push_back(build_v_mov_b32_e32(
      static_cast<uint16_t>(scratch_vgpr + 13u + kConSanMoiInlineCausalSnapshotEntryCapacity - 1u),
      vector_source_vgpr(temporary_vgpr), arch));
  sequence
      .append(
          instrumentation::build_s_andn2_b64(kAmdGpuExecLo, validated_exec, same_owner_exec, arch),
          instrumentation::build_s_xor_b64(kAmdGpuExecLo, kAmdGpuExecLo,
                                           static_cast<uint16_t>(exec_base + 14u), arch),
          instrumentation::build_s_mov_b64(same_owner_exec, static_cast<uint16_t>(exec_base + 14u),
                                           arch))
      .branch(restore_workgroup, InstructionSequence::BranchKind::ExecZero);

  if (!release_sequence_only) {
    // Only a fully validated acquire establishes a new consumer segment. In
    // particular, a mismatched or unstable release slot must not make
    // conflicting accesses appear barrier-ordered merely because the guest
    // atomic had acquire semantics.
    //
    // Owner IDs are wave IDs, while an atomic may execute under a single-lane
    // EXEC mask. Advance a VGPR-backed epoch for the whole wave, but preserve
    // the validated acquire mask for token publication. An empty mask branches
    // around the widening so a failed acquire cannot resurrect inactive lanes.
    // Private state was initially loaded only for the guest atomic's lanes, so
    // reload it after widening and write the advanced wave segment back for
    // every lane before later access caves rematerialize it.
    // Scalar persistent state is already wave-uniform and needs no widening.
    // Saturation stays fail-closed in the token readers.
    const bool widen_consumer_segment = inline_access_present && !plan.persistent_sgprs.epoch();
    std::vector<uint32_t> advance_words;
    InstructionSequence advance_sequence(advance_words);
    if (widen_consumer_segment) {
      advance_sequence.append(
          instrumentation::build_s_mov_b64(kAmdGpuExecLo, kScalarInlineNegativeOneOperand, arch));
      if (plan.private_state) {
        advance_sequence.require(advance_sequence.emit_all(
            instrumentation::build_private_load_b32(plan.owner_epoch_vgprs.epoch,
                                                    plan.private_state->epoch_offset, arch),
            instrumentation::build_s_wait_private_load0(arch)));
      }
    }
    advance_sequence.append(instrumentation::build_v_add_u32(plan.owner_epoch_vgprs.epoch,
                                                             scalar_positive_inline_u32(1),
                                                             plan.owner_epoch_vgprs.epoch, arch),
                            instrumentation::build_v_min_u32_literal(
                                plan.owner_epoch_vgprs.epoch, consan_moi_exact_shadow::max_epoch,
                                plan.owner_epoch_vgprs.epoch, arch));
    if (widen_consumer_segment && plan.private_state) {
      advance_sequence.require(advance_sequence.emit_all(
          instrumentation::build_private_store_b32(plan.owner_epoch_vgprs.epoch,
                                                   plan.private_state->epoch_offset, arch),
          instrumentation::build_s_wait_private_store0(arch)));
    }
    if (widen_consumer_segment)
      advance_sequence.append(instrumentation::build_s_mov_b64(
          kAmdGpuExecLo, static_cast<uint16_t>(exec_base + 16u), arch));
    require_emission(advance_sequence.finish(),
                     "ConSan MOI inline acquire could not advance its consumer segment");
    if (plan.persistent_sgprs.epoch()) {
      require_emission(advance_sequence.emit_all(
                           instrumentation::build_v_readfirstlane_b32(
                               *plan.persistent_sgprs.epoch(), plan.owner_epoch_vgprs.epoch, arch),
                           instrumentation::build_valu_to_salu_dependency_wait(arch)),
                       "ConSan MOI inline acquire could not persist its consumer segment");
      // The next atomic/access cave rematerializes this scalar epoch into a
      // VGPR. Make the vector-to-scalar transfer architecturally visible
      // before returning to guest control, rather than relying on elapsed
      // instructions in this cave to resolve the dependency.
    }
    if (widen_consumer_segment)
      require_emission.append("ConSan MOI inline acquire could not save its validated lanes",
                              instrumentation::build_s_mov_b64(
                                  static_cast<uint16_t>(exec_base + 16u), kAmdGpuExecLo, arch));
    const InstructionSequence::Label consumer_segment_done = sequence.make_label();
    require_emission(
        sequence.emit_branch(consumer_segment_done, InstructionSequence::BranchKind::ExecZero) &&
            sequence.emit(advance_words) && sequence.bind(consumer_segment_done),
        "ConSan MOI inline acquire could not guard its consumer segment");
  }

  sequence
      .require(append_inline_acquired_token_transaction(
          words, plan, scratch_vgpr, address_vgpr, workgroup_key_vgpr, producer_owner_vgpr,
          token_value_vgpr, version_before, snapshot_count,
          static_cast<uint16_t>(scratch_vgpr + 9u), static_cast<uint16_t>(scratch_vgpr + 13u),
          token_table_base, token_table_capacity, release_sequence_only, same_owner_exec, arch))
      .bind_label(restore_workgroup)
      .append(instrumentation::build_s_mov_b64(kAmdGpuExecLo,
                                               static_cast<uint16_t>(exec_base + 12u), arch));
  return sequence.finish(arch);
}

// Capture the complete stable causal frontier for the currently active
// release publishers. EXEC on entry and exit is the successful release-claim
// mask. The generated loop scans every ABI-v6 token slot exactly once and
// maintains four canonical (owner, epoch+1) pairs independently per lane.
[[nodiscard]] bool append_inline_release_causal_snapshot_capture(
    std::vector<uint32_t> &words, const MoiInlineAtomicEmissionPlan &plan, uint16_t scratch_vgpr,
    uint16_t atomic_address_vgpr, uint16_t workgroup_key_vgpr, uint64_t token_table_base,
    uint32_t token_table_capacity, uint64_t snapshot_table_base, uint32_t snapshot_table_capacity,
    rj_code_arch_t arch) {
  if (token_table_capacity == 0 ||
      token_table_capacity > kConSanMoiInlineShadowAcquiredEpochTokenSlotCapacity ||
      snapshot_table_capacity == 0 ||
      (snapshot_table_capacity & (snapshot_table_capacity - 1u)) != 0u)
    return false;
  const uint16_t base = scratch_vgpr;
  const bool aligned_cas_pair = plan.requires_aligned_flat_compare_swap_data_pair;
  using SnapshotLayout = inline_register_layout::AtomicCausalSnapshot;
  const uint16_t snapshot_address =
      static_cast<uint16_t>(base + SnapshotLayout::address(aligned_cas_pair));
  const uint16_t count = static_cast<uint16_t>(base + SnapshotLayout::count(aligned_cas_pair));
  const uint16_t flags = static_cast<uint16_t>(base + SnapshotLayout::flags);
  const uint16_t owners = static_cast<uint16_t>(base + SnapshotLayout::owners);
  const uint16_t epochs = static_cast<uint16_t>(base + SnapshotLayout::epochs);
  const uint16_t version_before =
      static_cast<uint16_t>(base + SnapshotLayout::token_version_before);
  const uint16_t producer = static_cast<uint16_t>(base + SnapshotLayout::producer);
  const uint16_t producer_epoch = static_cast<uint16_t>(base + SnapshotLayout::producer_epoch);
  const uint16_t temporary = static_cast<uint16_t>(base + SnapshotLayout::temporary);
  const uint16_t loop_index = static_cast<uint16_t>(base + SnapshotLayout::loop_index);
  const uint16_t exec_base = plan.exec_save_sgpr;
  const uint16_t narrow_save = exec_base;
  const uint16_t empty_exec = static_cast<uint16_t>(exec_base + 2u);
  const uint16_t ready_exec = static_cast<uint16_t>(exec_base + 4u);
  const uint16_t candidate_exec = static_cast<uint16_t>(exec_base + 6u);
  const uint16_t scan_exec = static_cast<uint16_t>(exec_base + 14u);
  const uint16_t mask_a = static_cast<uint16_t>(exec_base + 16u);
  const uint16_t mask_b = static_cast<uint16_t>(exec_base + 18u);
  // Empty is dead before the first flag update. The narrow-save pair is
  // scratch after each saveexec. Alias those two disjoint lifetimes rather
  // than reserving a third long-lived mask pair.
  const uint16_t flag_exec = empty_exec;
  const uint16_t high_nonzero_exec = narrow_save;
  InstructionSequence sequence(words);

  // The vector loop counter cannot advance with EXEC empty. Branch around the
  // complete scan in that case; scalar execution resumes at the caller with
  // the same empty EXEC mask.
  const InstructionSequence::Label scan_done = sequence.make_label();
  sequence.branch(scan_done, InstructionSequence::BranchKind::ExecZero)
      .append(instrumentation::build_s_mov_b64(scan_exec, kAmdGpuExecLo, arch));

  MoiPublicationExec exec_masks(sequence, narrow_save, arch);
  const auto restore_exec = [&](uint16_t source) { sequence.require(exec_masks.restore(source)); };
  const auto save_exec = [&](uint16_t destination) {
    sequence.require(exec_masks.save(destination));
  };
  const auto narrow_vcc = [&] { sequence.require(exec_masks.narrow_vcc()); };
  const auto require_literal = [&](uint16_t value, uint32_t literal, bool equal) {
    sequence.require(exec_masks.require_literal(value, literal, equal));
  };
  const auto load_require_literal = [&](size_t offset, uint32_t literal, bool equal) {
    sequence.require(
        append_load_u32_vgpr_at_offset(words, snapshot_address, offset, temporary, arch));
    require_literal(temporary, literal, equal);
  };
  const auto set_flag = [&](ConSanMoiInlineCausalSnapshotFlag flag) {
    const uint32_t bit = consan_moi_inline_causal_snapshot_flag(flag);
    save_exec(flag_exec);
    sequence.append(instrumentation::build_v_and_b32_literal(temporary, bit, flags, arch))
        .require(exec_masks.narrow(instrumentation::build_v_cmp_eq_u32_vcc(
            scalar_positive_inline_u32(0), temporary, arch)))
        .append(
            instrumentation::build_v_add_u32(flags, scalar_positive_inline_u32(bit), flags, arch));
    restore_exec(flag_exec);
  };
  const auto require_nonzero_u64 = [&](size_t offset) {
    save_exec(mask_a);
    sequence
        .require(append_load_u32_vgpr_at_offset(words, snapshot_address, offset, temporary, arch))
        .require(exec_masks.narrow(instrumentation::build_v_cmp_ne_u32_vcc(
            scalar_positive_inline_u32(0), temporary, arch)));
    save_exec(mask_b);
    restore_exec(mask_a);
    sequence
        .require(exec_masks.narrow(instrumentation::build_v_cmp_eq_u32_vcc(
            scalar_positive_inline_u32(0), temporary, arch)))
        .require(append_load_u32_vgpr_at_offset(words, snapshot_address, offset + sizeof(uint32_t),
                                                temporary, arch))
        .require(exec_masks.narrow(instrumentation::build_v_cmp_ne_u32_vcc(
            scalar_positive_inline_u32(0), temporary, arch)));
    save_exec(high_nonzero_exec);
    sequence.append(
        instrumentation::build_s_xor_b64(kAmdGpuExecLo, mask_b, high_nonzero_exec, arch));
  };

  sequence.append(instrumentation::build_v_mov_b32_literal(temporary, 0, arch));
  words.push_back(build_v_mov_b32_e32(count, vector_source_vgpr(temporary), arch));
  words.push_back(build_v_mov_b32_e32(flags, vector_source_vgpr(temporary), arch));
  for (uint16_t i = 0; i < 4u; ++i) {
    words.push_back(build_v_mov_b32_e32(static_cast<uint16_t>(owners + i),
                                        vector_source_vgpr(temporary), arch));
    words.push_back(build_v_mov_b32_e32(static_cast<uint16_t>(epochs + i),
                                        vector_source_vgpr(temporary), arch));
  }
  words.push_back(build_v_mov_b32_e32(loop_index, vector_source_vgpr(temporary), arch));
  sequence.append(instrumentation::build_v_mov_b32_literal(
                      snapshot_address, static_cast<uint32_t>(token_table_base), arch),
                  instrumentation::build_v_mov_b32_literal(
                      static_cast<uint16_t>(snapshot_address + 1u),
                      static_cast<uint32_t>(token_table_base >> 32u), arch));

  const InstructionSequence::Label loop_begin = sequence.mark_label();
  restore_exec(scan_exec);
  sequence.require(append_load_u32_vgpr_at_offset(
      words, snapshot_address, offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, version),
      version_before, arch));
  require_literal(version_before, 0, /*equal=*/true);
  for (size_t offset : kConSanMoiInlineAcquiredEpochTokenPayloadOffsets) {
    load_require_literal(offset, 0, /*equal=*/true);
  }
  load_require_literal(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, version), 0,
                       /*equal=*/true);
  save_exec(empty_exec);

  restore_exec(scan_exec);
  require_literal(version_before, 0, /*equal=*/false);
  sequence.append(instrumentation::build_v_and_b32_literal(temporary, 1u, version_before, arch));
  require_literal(temporary, 0, /*equal=*/true);
  load_require_literal(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, consumer_owner_id), 0,
                       /*equal=*/false);
  load_require_literal(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, producer_owner_id), 0,
                       /*equal=*/false);
  sequence.require(append_load_u32_vgpr_at_offset(
      words, snapshot_address,
      offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, producer_epoch_plus_one), producer_epoch,
      arch));
  require_literal(producer_epoch, 0, /*equal=*/false);
  sequence.append(instrumentation::build_v_mov_b32_literal(temporary, 1024u, arch))
      .require(exec_masks.narrow(instrumentation::build_v_cmp_gt_u32_vcc(
          vector_source_vgpr(temporary), producer_epoch, arch)));
  load_require_literal(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, workgroup_key), 0,
                       /*equal=*/false);
  sequence
      .require(append_load_u32_vgpr_at_offset(words, snapshot_address,
                                              offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, kind),
                                              temporary, arch))
      .require(exec_masks.narrow(
          instrumentation::build_v_cmp_gt_u32_vcc(scalar_positive_inline_u32(3), temporary, arch)));
  require_nonzero_u64(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, dispatch_id));
  require_nonzero_u64(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, source_release_address));
  sequence.require(append_load_u32_vgpr_at_offset(
      words, snapshot_address,
      offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, source_release_version), temporary, arch));
  require_literal(temporary, 0, /*equal=*/false);
  sequence.append(instrumentation::build_v_and_b32_literal(producer, 1u, temporary, arch));
  require_literal(producer, 0, /*equal=*/true);
  load_require_literal(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, consumer_epoch_plus_one), 0,
                       /*equal=*/false);
  sequence
      .append(instrumentation::build_v_mov_b32_literal(
          producer, consan_moi_exact_shadow::max_epoch + 1u, arch))
      .require(exec_masks.narrow(
          instrumentation::build_v_cmp_gt_u32_vcc(vector_source_vgpr(producer), temporary, arch)));
  load_require_literal(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, reservation_version), 0,
                       /*equal=*/true);
  sequence
      .require(append_load_u32_vgpr_at_offset(
          words, snapshot_address, offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, version),
          temporary, arch))
      .require(exec_masks.narrow(instrumentation::build_v_cmp_eq_u32_vcc(
          vector_source_vgpr(version_before), temporary, arch)));
  save_exec(ready_exec);

  sequence.append(instrumentation::build_s_xor_b64(mask_a, empty_exec, ready_exec, arch),
                  instrumentation::build_s_andn2_b64(kAmdGpuExecLo, scan_exec, mask_a, arch));
  set_flag(ConSanMoiInlineCausalSnapshotFlag::Malformed);
  restore_exec(ready_exec);

  const auto require_current_field = [&](size_t offset, uint16_t expected) {
    sequence
        .require(append_load_u32_vgpr_at_offset(words, snapshot_address, offset, temporary, arch))
        .require(
            exec_masks.narrow(instrumentation::build_v_cmp_eq_u32_vcc(expected, temporary, arch)));
  };
  const auto require_current_dispatch_field = [&](size_t offset, bool high_word) {
    // The stable-version check is complete. `version_before` is dead here and
    // the next scan iteration reloads it before use.
    sequence
        .require(append_load_u32_vgpr_at_offset(words, snapshot_address, offset, temporary, arch))
        .require(append_compare_moi_report_dispatch_id_word(words, plan.dispatch_id, temporary,
                                                            version_before, high_word, arch));
    narrow_vcc();
  };
  require_current_dispatch_field(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, dispatch_id),
                                 /*high_word=*/false);
  require_current_dispatch_field(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, dispatch_id) +
                                     sizeof(uint32_t),
                                 /*high_word=*/true);
  require_current_field(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, workgroup_key),
                        vector_source_vgpr(workgroup_key_vgpr));
  require_current_field(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, consumer_owner_id),
                        vector_source_vgpr(plan.owner_epoch_vgprs.owner));
  sequence.require(append_load_u32_vgpr_at_offset(
      words, snapshot_address, offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, producer_owner_id),
      producer, arch));
  save_exec(candidate_exec);

  sequence.require(exec_masks.narrow(instrumentation::build_v_cmp_eq_u32_vcc(
      vector_source_vgpr(plan.owner_epoch_vgprs.owner), producer, arch)));
  save_exec(mask_b);
  set_flag(ConSanMoiInlineCausalSnapshotFlag::Malformed);
  sequence.append(instrumentation::build_s_andn2_b64(candidate_exec, candidate_exec, mask_b, arch));

  for (uint16_t i = 0; i < 4u; ++i) {
    restore_exec(candidate_exec);
    sequence.append(instrumentation::build_v_mov_b32_literal(temporary, i, arch))
        .require(exec_masks.narrow(
            instrumentation::build_v_cmp_gt_u32_vcc(vector_source_vgpr(count), temporary, arch)))
        .require(exec_masks.narrow(instrumentation::build_v_cmp_eq_u32_vcc(
            vector_source_vgpr(static_cast<uint16_t>(owners + i)), producer, arch)));
    save_exec(mask_b);
    // Direct and release-sequence namespaces can both carry the same causal
    // edge. They are redundant witnesses, not malformed ancestry. Canonicalize
    // the frontier as an owner->maximum-epoch map before removing the duplicate
    // candidate from the insertion path.
    sequence.require(exec_masks.narrow(instrumentation::build_v_cmp_gt_u32_vcc(
        vector_source_vgpr(producer_epoch), static_cast<uint16_t>(epochs + i), arch)));
    words.push_back(build_v_mov_b32_e32(static_cast<uint16_t>(epochs + i),
                                        vector_source_vgpr(producer_epoch), arch));
    sequence.append(
        instrumentation::build_s_andn2_b64(candidate_exec, candidate_exec, mask_b, arch));
  }

  restore_exec(candidate_exec);
  sequence.require(exec_masks.narrow(
      instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(4), count, arch)));
  save_exec(mask_b);
  set_flag(ConSanMoiInlineCausalSnapshotFlag::CapacityOverflow);
  sequence.append(instrumentation::build_s_andn2_b64(candidate_exec, candidate_exec, mask_b, arch));

  for (uint16_t i = 0; i < 4u; ++i) {
    restore_exec(candidate_exec);
    sequence.require(exec_masks.narrow(
        instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(i), count, arch)));
    words.push_back(
        build_v_mov_b32_e32(static_cast<uint16_t>(owners + i), vector_source_vgpr(producer), arch));
    words.push_back(build_v_mov_b32_e32(static_cast<uint16_t>(epochs + i),
                                        vector_source_vgpr(producer_epoch), arch));
  }
  restore_exec(candidate_exec);
  sequence.append(
      instrumentation::build_v_add_u32(count, scalar_positive_inline_u32(1), count, arch));

  for (uint16_t right = 3u; right > 0u; --right) {
    restore_exec(candidate_exec);
    sequence.append(instrumentation::build_v_mov_b32_literal(temporary, right, arch))
        .require(exec_masks.narrow(
            instrumentation::build_v_cmp_gt_u32_vcc(vector_source_vgpr(count), temporary, arch)))
        .require(exec_masks.narrow(instrumentation::build_v_cmp_gt_u32_vcc(
            vector_source_vgpr(static_cast<uint16_t>(owners + right - 1u)),
            static_cast<uint16_t>(owners + right), arch)));
    words.push_back(build_v_mov_b32_e32(
        producer, vector_source_vgpr(static_cast<uint16_t>(owners + right - 1u)), arch));
    words.push_back(build_v_mov_b32_e32(
        producer_epoch, vector_source_vgpr(static_cast<uint16_t>(epochs + right - 1u)), arch));
    words.push_back(build_v_mov_b32_e32(static_cast<uint16_t>(owners + right - 1u),
                                        vector_source_vgpr(static_cast<uint16_t>(owners + right)),
                                        arch));
    words.push_back(build_v_mov_b32_e32(static_cast<uint16_t>(epochs + right - 1u),
                                        vector_source_vgpr(static_cast<uint16_t>(epochs + right)),
                                        arch));
    words.push_back(build_v_mov_b32_e32(static_cast<uint16_t>(owners + right),
                                        vector_source_vgpr(producer), arch));
    words.push_back(build_v_mov_b32_e32(static_cast<uint16_t>(epochs + right),
                                        vector_source_vgpr(producer_epoch), arch));
  }

  restore_exec(scan_exec);
  sequence
      .require(append_add_literal_field(
          words, snapshot_address, sizeof(ConSanMoiInlineAcquiredEpochTokenSlot), temporary, arch))
      .append(
          instrumentation::build_v_add_u32(loop_index, scalar_positive_inline_u32(1), loop_index,
                                           arch),
          instrumentation::build_v_mov_b32_literal(temporary, token_table_capacity, arch),
          instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(temporary), loop_index, arch))
      .branch(loop_begin, InstructionSequence::BranchKind::VccZero);

  restore_exec(scan_exec);
  sequence
      .require(append_inline_causal_snapshot_address(words, snapshot_table_base,
                                                     snapshot_table_capacity, atomic_address_vgpr,
                                                     snapshot_address, temporary, loop_index, arch))
      .require(append_moi_publication_stores(
          words, snapshot_address,
          {{offsetof(ConSanMoiInlineCausalSnapshot, entry_count), count},
           {offsetof(ConSanMoiInlineCausalSnapshot, flags), flags}},
          arch));
  for (uint16_t i = 0; i < 4u; ++i) {
    const size_t entry_offset = offsetof(ConSanMoiInlineCausalSnapshot, entries) +
                                i * sizeof(ConSanMoiInlineCausalSnapshotEntry);
    sequence.require(append_moi_publication_stores(
        words, snapshot_address,
        {{static_cast<uint32_t>(entry_offset), static_cast<uint16_t>(owners + i)},
         {static_cast<uint32_t>(entry_offset + sizeof(uint32_t)),
          static_cast<uint16_t>(epochs + i)}},
        arch));
  }
  sequence.bind_label(scan_done);
  return sequence.finish(arch);
}

[[nodiscard]] bool append_inline_versioned_release_transaction(
    std::vector<uint32_t> &words, std::span<const uint8_t> bytes,
    ConSanMoiAtomicEventKind event_kind, const ConSanAtomicSite &site,
    const ConSanMoiAtomicAddressPlan &address_plan, const MoiInlineAtomicEmissionPlan &plan,
    rj_code_arch_t arch, bool emit_guest_instruction, bool import_claimed_predecessor,
    bool predicate_on_compare_exchange_success, uint32_t &guest_instruction_offset,
    std::vector<std::string> &errors, std::span<const uint32_t> trailing_guest_words = {}) {
  const uint16_t required_scratch_count = plan.scratch_vgpr_count;
  const uint16_t scratch_vgpr = plan.scratch_vgpr;
  const uint64_t release_table_base =
      plan.report_buffer_address + plan.report_layout.inline_atomic_release_slots_offset;
  const uint32_t release_capacity = plan.report_layout.inline_atomic_release_capacity;
  const uint64_t snapshot_table_base =
      plan.report_buffer_address + plan.report_layout.inline_causal_snapshots_offset;
  const uint32_t snapshot_capacity = plan.report_layout.inline_causal_snapshot_capacity;
  const uint64_t token_table_base =
      plan.report_buffer_address + plan.report_layout.inline_acquired_epoch_token_slots_offset;
  const uint32_t token_capacity = plan.report_layout.inline_acquired_epoch_token_capacity;
  const bool inline_access_present = plan.inline_access_present;
  const uint16_t base = plan.scratch_vgpr;
  const uint16_t slot_address = base;
  const uint16_t prior_version = static_cast<uint16_t>(base + 2u);
  const uint16_t workgroup_key = static_cast<uint16_t>(base + 3u);
  // These three registers are recomputed after causal-snapshot capture, so
  // rotating them for a target's aligned CAS pair extends no live range.
  const bool aligned_cas_pair = plan.requires_aligned_flat_compare_swap_data_pair;
  const uint16_t temporary = static_cast<uint16_t>(base + (aligned_cas_pair ? 6u : 4u));
  const uint16_t cas_new = static_cast<uint16_t>(base + (aligned_cas_pair ? 4u : 5u));
  const uint16_t cas_expected = static_cast<uint16_t>(base + (aligned_cas_pair ? 5u : 6u));
  const uint16_t exec_base = plan.exec_save_sgpr;
  const uint16_t narrow_save = exec_base;
  const uint16_t empty_exec = static_cast<uint16_t>(exec_base + 2u);
  const uint16_t ready_exec = static_cast<uint16_t>(exec_base + 4u);
  const uint16_t original_exec = static_cast<uint16_t>(exec_base + 12u);
  const uint16_t claimed_exec = static_cast<uint16_t>(exec_base + 14u);
  const uint16_t committed_exec = static_cast<uint16_t>(exec_base + 16u);
  const uint16_t retry_count_sgpr = static_cast<uint16_t>(exec_base + 20u);
  // The scalar retry counter is dead once the claim loop exits. Ordinary
  // release transactions reuse its pair for the same-wave loser mask. A
  // claimed AcquireRelease instead journals its winning mask there while the
  // acquire helper reuses +14, then reconstructs the loser mask afterward.
  const uint16_t failed_exec = retry_count_sgpr;
  // append_inline_workgroup_key's +6 predicate mask is dead once the key has
  // been formed. The eligible mask dies before causal-snapshot capture, where
  // +6 is reused again as that helper's candidate mask.
  const uint16_t eligible_exec = static_cast<uint16_t>(exec_base + 6u);
  const uint16_t guest_address = address_plan.result_address_vgpr;
  const uint16_t stable_guest_address = static_cast<uint16_t>(base + required_scratch_count - 2u);

  InstructionSequence sequence(words);
  MoiPublicationExec exec_masks(sequence, narrow_save, arch);
  MoiStagedEmission require_emission(sequence, errors,
                                     "ConSan MOI inline versioned release transaction");
  const auto set_stage = [&](std::string_view stage) { require_emission.stage(stage); };
  const auto restore_exec = [&](uint16_t source) { return exec_masks.restore(source); };
  const auto save_exec = [&](uint16_t destination) { return exec_masks.save(destination); };
  const auto narrow_vcc = [&] { return exec_masks.narrow_vcc(); };
  const auto require_literal = [&](uint16_t value, uint32_t literal, bool equal) {
    return exec_masks.require_literal(value, literal, equal);
  };
  const auto require_field = [&](size_t offset, uint16_t expected) -> bool {
    if (!append_load_u32_vgpr_at_offset(words, slot_address, offset, temporary, arch))
      return false;
    return exec_masks.narrow(instrumentation::build_v_cmp_eq_u32_vcc(expected, temporary, arch));
  };
  const auto require_dispatch_field = [&](size_t offset, bool high_word) -> bool {
    // `cas_new` has no live value before the claim. `claim_new` below
    // overwrites it before the first use of the compare-swap operand.
    if (!append_load_u32_vgpr_at_offset(words, slot_address, offset, temporary, arch) ||
        !append_compare_moi_report_dispatch_id_word(words, plan.dispatch_id, temporary, cas_new,
                                                    high_word, arch)) {
      return false;
    }
    return narrow_vcc();
  };

  set_stage("guest-address preservation");
  if (stable_guest_address != guest_address) {
    // Copy high first only when the destination begins at the source high
    // word; otherwise the ordinary low/high order is alias-safe.
    if (static_cast<uint16_t>(guest_address + 1u) == stable_guest_address) {
      words.push_back(
          build_v_mov_b32_e32(static_cast<uint16_t>(stable_guest_address + 1u),
                              vector_source_vgpr(static_cast<uint16_t>(guest_address + 1u)), arch));
      words.push_back(
          build_v_mov_b32_e32(stable_guest_address, vector_source_vgpr(guest_address), arch));
    } else {
      words.push_back(
          build_v_mov_b32_e32(stable_guest_address, vector_source_vgpr(guest_address), arch));
      words.push_back(
          build_v_mov_b32_e32(static_cast<uint16_t>(stable_guest_address + 1u),
                              vector_source_vgpr(static_cast<uint16_t>(guest_address + 1u)), arch));
    }
  }

  set_stage("transaction initialization");
  require_emission(append_save_moi_special_state(words, plan.special_state, arch) &&
                       append_inline_workgroup_key(words, plan.workgroup_sources,
                                                   plan.workgroup_key_registers, workgroup_key,
                                                   static_cast<uint16_t>(base + 6u), temporary,
                                                   /*original_exec_save_offset=*/12u, arch),
                   "ConSan MOI inline release could not initialize its EXEC transaction");
  const auto narrow_compare_exchange_success = [&]() -> bool {
    if (!site.data_vgpr || !site.destination_vgpr) {
      errors.emplace_back("ConSan MOI inline release CAS has no dynamic outcome operands");
      return false;
    }
    const uint16_t value_word_count = static_cast<uint16_t>(site.width_bits / 32u);
    const uint16_t compare_vgpr = static_cast<uint16_t>(*site.data_vgpr + value_word_count);
    if (!exec_masks.narrow(instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(compare_vgpr),
                                                                   *site.destination_vgpr, arch))) {
      errors.emplace_back("ConSan MOI inline release CAS could not compare its dynamic outcome");
      return false;
    }
    if (site.width_bits == 64u && !exec_masks.narrow(instrumentation::build_v_cmp_eq_u32_vcc(
                                      vector_source_vgpr(static_cast<uint16_t>(compare_vgpr + 1u)),
                                      static_cast<uint16_t>(*site.destination_vgpr + 1u), arch))) {
      errors.emplace_back(
          "ConSan MOI inline release 64-bit CAS could not compare its high outcome word");
      return false;
    }
    return true;
  };
  set_stage("guest CAS qualification");
  if (predicate_on_compare_exchange_success && !emit_guest_instruction)
    require_emission(narrow_compare_exchange_success(),
                     "ConSan MOI inline release CAS could not predicate publication");
  set_stage("release-slot addressing");
  require_emission(save_exec(eligible_exec) && append_inline_atomic_release_slot_address(
                                                   words, release_table_base, release_capacity,
                                                   stable_guest_address, slot_address, arch),
                   "ConSan MOI inline release could not initialize its version transaction");

  // Another wave can hold the direct-mapped release slot in its odd
  // publishing state for the complete causal-snapshot capture and guest RMW.
  // Retry only when this invocation has not claimed the slot at all; if one
  // of several local lanes wins, retain the existing fail-closed treatment of
  // the other lanes rather than letting multiple publishers write one slot.
  // The critical section includes a full acquired-token-table causal scan.
  // Coherent version polls plus a short yield let a resident publisher
  // complete, while the finite bound still terminates on a permanently stuck
  // odd slot.
  set_stage("release-slot reservation");
  const auto append_release_candidate = [&]() {
    if (!append_atomic_load_u32(words, slot_address, prior_version, arch)) {
      errors.emplace_back("ConSan MOI inline release could not begin its version retry");
      return false;
    }
    if (!require_literal(prior_version, 0, /*equal=*/true) || !save_exec(empty_exec) ||
        !restore_exec(eligible_exec) || !require_literal(prior_version, 0, /*equal=*/false) ||
        !sequence.emit(
            instrumentation::build_v_and_b32_literal(temporary, 1u, prior_version, arch)) ||
        !require_literal(temporary, 0, /*equal=*/true) ||
        !sequence.emit(instrumentation::build_v_mov_b32_literal(
            temporary, std::numeric_limits<uint32_t>::max() - 1u, arch))) {
      return false;
    }
    // Any stable even direct-mapped record is replaceable. Readers, rather
    // than publishers, own identity qualification.
    return exec_masks.narrow(instrumentation::build_v_cmp_ne_u32_vcc(vector_source_vgpr(temporary),
                                                                     prior_version, arch)) &&
           append_atomic_load_u32(words, slot_address, temporary, arch) &&
           exec_masks.narrow(instrumentation::build_v_cmp_eq_u32_vcc(
               vector_source_vgpr(prior_version), temporary, arch)) &&
           save_exec(ready_exec) &&
           sequence.emit(
               instrumentation::build_s_xor_b64(kAmdGpuExecLo, empty_exec, ready_exec, arch));
  };
  require_emission(append_moi_bounded_version_claim(
      words, sequence, exec_masks,
      {.slot_address_vgpr = slot_address,
       .eligible_exec_sgpr = eligible_exec,
       .claimed_exec_sgpr = claimed_exec,
       .retry_exec_sgpr = committed_exec,
       .retry_count_sgpr = retry_count_sgpr,
       .base_version_vgpr = prior_version,
       .desired_vgpr = cas_new,
       .expected_vgpr = cas_expected,
       .version_offset = offsetof(ConSanMoiInlineAtomicReleaseSlot, version),
       .retry_limit = kConSanMoiInlineMetadataPublicationRetryLimit,
       .sleep_delay = kConSanMoiInlineMetadataPublicationSleepDelay},
      append_release_candidate, arch));

  set_stage("reservation outcome journaling");
  if (import_claimed_predecessor) {
    require_emission(restore_exec(claimed_exec) && save_exec(failed_exec));
  } else {
    sequence.append(
        instrumentation::build_s_andn2_b64(kAmdGpuExecLo, eligible_exec, claimed_exec, arch));
    require_emission(save_exec(failed_exec));
  }

  set_stage("claimed predecessor import");
  if (import_claimed_predecessor) {
    if (!sequence)
      return false;
    const uint16_t source_version = static_cast<uint16_t>(base + 20u);
    // The successful claim returns the authoritative prior even version in
    // cas_new. Keep the acquire source tied to that returned value instead of
    // the retry-loop temporary, whose lifetime ends at the claim boundary.
    words.push_back(build_v_mov_b32_e32(source_version, vector_source_vgpr(cas_new), arch));
    require_emission(restore_exec(original_exec) &&
                     append_restore_moi_special_state(words, plan.special_state, arch));
    guest_instruction_offset = static_cast<uint32_t>(words.size() * sizeof(uint32_t));
    for (uint64_t offset = 0; offset < site.size; offset += sizeof(uint32_t)) {
      uint32_t word = 0;
      std::memcpy(&word, bytes.data() + site.file_offset + offset, sizeof(word));
      words.push_back(word);
    }
    words.insert(words.end(), trailing_guest_words.begin(), trailing_guest_words.end());
    require_emission(append_moi_flat_load_wait(words, arch));
    sequence.append(instrumentation::build_s_wait_flat_store0(arch));
    // The token transaction journals its source version in scratch+8, but an
    // empty predecessor narrows EXEC before that transaction is entered. Seed
    // the same journal slot here so both the empty and nonempty paths retain
    // the claim's prior even version for the successor commit.
    words.push_back(build_v_mov_b32_e32(static_cast<uint16_t>(base + 8u),
                                        vector_source_vgpr(source_version), arch));
    require_emission(append_save_moi_special_state(words, plan.special_state, arch));
    if (predicate_on_compare_exchange_success && sequence) {
      require_emission(
          restore_exec(claimed_exec) && narrow_compare_exchange_success() &&
              save_exec(claimed_exec),
          "ConSan MOI inline release CAS could not predicate its claimed predecessor import");
    }
    require_emission(
        restore_exec(claimed_exec) &&
        append_inline_atomic_acquire_import(
            words, plan, scratch_vgpr, stable_guest_address, workgroup_key, cas_new, cas_expected,
            temporary, snapshot_table_base, snapshot_capacity, token_table_base, token_capacity,
            /*source_slot_claimed=*/true,
            /*release_sequence_only=*/
            event_kind == ConSanMoiAtomicEventKind::Release, inline_access_present, arch, errors));
    // Acquire import reuses the release slot-address and scratch+20 source
    // version for its token transaction. That transaction journals the source
    // version in scratch+8; recover the claim's prior even version from there
    // before snapshotting and committing the still-owned release reservation.
    // The acquire helper uses +14 for its owner predicate and token journal.
    // Recover the outer claim from +20, then rebuild the same-object loser
    // mask from the original invocation before either is needed below. Slot
    // address reconstruction uses scratch+2 as its hash temporary, so recover
    // the prior version from the token journal only after that reconstruction.
    if (predicate_on_compare_exchange_success) {
      // The metadata claim must precede a returning guest CAS, but only lanes
      // whose dynamic compare succeeded may import or publish a release.
      // Restore the prior even slot for failed comparisons, then define
      // detector undercoverage solely over successful guest lanes that could
      // not claim metadata.
      require_emission(restore_exec(original_exec) && narrow_compare_exchange_success() &&
                       save_exec(eligible_exec));
      const auto successful_claims =
          instrumentation::build_s_and_b64(claimed_exec, failed_exec, eligible_exec, arch);
      const auto failed_comparisons =
          instrumentation::build_s_andn2_b64(ready_exec, failed_exec, claimed_exec, arch);
      sequence.append(successful_claims, failed_comparisons);
      require_emission(restore_exec(ready_exec) && append_inline_atomic_release_slot_address(
                                                       words, release_table_base, release_capacity,
                                                       stable_guest_address, slot_address, arch));
      words.push_back(build_v_mov_b32_e32(
          prior_version, vector_source_vgpr(static_cast<uint16_t>(base + 8u)), arch));
      require_emission(
          append_moi_version_transition(words, sequence, exec_masks, slot_address,
                                        offsetof(ConSanMoiInlineAtomicReleaseSlot, version),
                                        prior_version, cas_new, cas_expected, /*desired_delta=*/0u,
                                        /*expected_delta=*/1u, arch) &&
          save_exec(committed_exec));
      const auto restore_failed =
          instrumentation::build_s_andn2_b64(kAmdGpuExecLo, ready_exec, committed_exec, arch);
      sequence.append(restore_failed);
      require_emission(append_atomic_fetch_add_one_u32(
          words,
          plan.report_buffer_address + offsetof(ConSanMoiReportHeader, inline_overflow_count),
          temporary, cas_new, arch));
      const auto rebuild_failed =
          instrumentation::build_s_andn2_b64(kAmdGpuExecLo, eligible_exec, claimed_exec, arch);
      sequence.append(rebuild_failed);
      require_emission(
          save_exec(failed_exec) && restore_exec(claimed_exec) &&
          append_inline_atomic_release_slot_address(words, release_table_base, release_capacity,
                                                    stable_guest_address, slot_address, arch));
    } else {
      require_emission(restore_exec(failed_exec) && save_exec(claimed_exec));
      const auto rebuild_failed =
          instrumentation::build_s_andn2_b64(kAmdGpuExecLo, original_exec, claimed_exec, arch);
      sequence.append(rebuild_failed);
      require_emission(
          save_exec(failed_exec) && restore_exec(claimed_exec) &&
          append_inline_atomic_release_slot_address(words, release_table_base, release_capacity,
                                                    stable_guest_address, slot_address, arch));
    }
    words.push_back(build_v_mov_b32_e32(
        prior_version, vector_source_vgpr(static_cast<uint16_t>(base + 8u)), arch));
  }

  if (event_kind == ConSanMoiAtomicEventKind::Acquire) {
    // A read-only acquire must bind its guest observation to exactly one
    // immutable release transaction. Holding the direct-mapped slot odd
    // across the guest load excludes a producer from linearizing its RMW
    // between our metadata snapshot and import. Restore the prior even version
    // unchanged after the acquired-token transaction; this reader never
    // becomes a release publisher itself.
    set_stage("acquire reservation restore");
    // Guest VCC/SCC were re-saved immediately after the guest load, before
    // acquire import. Do not snapshot them again here: the import changes
    // both condition registers, so a second save would replace guest state
    // with detector state just before the final restore.
    require_emission(restore_exec(claimed_exec));
    require_emission(
        append_moi_version_transition(words, sequence, exec_masks, slot_address,
                                      offsetof(ConSanMoiInlineAtomicReleaseSlot, version),
                                      prior_version, cas_new, cas_expected, /*desired_delta=*/0u,
                                      /*expected_delta=*/1u, arch) &&
        save_exec(committed_exec));
    const auto failed =
        instrumentation::build_s_andn2_b64(kAmdGpuExecLo, original_exec, committed_exec, arch);
    sequence.append(failed);
    // A tight polling acquire can otherwise unlock and immediately reclaim
    // the same direct-mapped slot indefinitely while a release wave is asleep
    // in its bounded claim loop. Yield once after every read-side unlock so a
    // waiting publisher gets a hardware scheduling handoff before the guest
    // poll can re-enter this transaction.
    const uint32_t publisher_handoff =
        build_s_sleep(kConSanMoiInlineMetadataPublicationSleepDelay, arch);
    require_emission(
        append_atomic_fetch_add_one_u32(words,
                                        plan.report_buffer_address +
                                            offsetof(ConSanMoiReportHeader, inline_overflow_count),
                                        temporary, cas_new, arch) &&
        restore_exec(original_exec));
    words.push_back(publisher_handoff);
    require_emission(append_restore_moi_special_state(words, plan.special_state, arch));
    return require_emission.finish(arch);
  }

  set_stage("causal snapshot capture");
  require_emission(
      restore_exec(claimed_exec) &&
      append_inline_release_causal_snapshot_capture(words, plan, scratch_vgpr, stable_guest_address,
                                                    workgroup_key, token_table_base, token_capacity,
                                                    snapshot_table_base, snapshot_capacity, arch));

  set_stage("release metadata publication");
  sequence.append(
      instrumentation::build_v_and_b32_literal(temporary, consan_moi_exact_shadow::max_epoch,
                                               plan.owner_epoch_vgprs.epoch, arch),
      instrumentation::build_v_add_u32(temporary, scalar_positive_inline_u32(1), temporary, arch),
      instrumentation::build_v_min_u32_literal(temporary, consan_moi_exact_shadow::max_epoch,
                                               temporary, arch));
  require_emission(append_moi_publication_stores(
      words, slot_address,
      {{offsetof(ConSanMoiInlineAtomicReleaseSlot, owner_id), plan.owner_epoch_vgprs.owner},
       {offsetof(ConSanMoiInlineAtomicReleaseSlot, epoch_plus_one), temporary},
       {offsetof(ConSanMoiInlineAtomicReleaseSlot, workgroup_key), workgroup_key},
       {offsetof(ConSanMoiInlineAtomicReleaseSlot, atomic_address), stable_guest_address},
       {offsetof(ConSanMoiInlineAtomicReleaseSlot, atomic_address) + sizeof(uint32_t),
        static_cast<uint16_t>(stable_guest_address + 1u)}},
      arch));
  if (!append_moi_report_dispatch_id_word(words, plan.dispatch_id, temporary,
                                          /*high_word=*/false, arch)) {
    require_emission(false, "ConSan MOI inline release could not materialize dispatch ID low");
  }
  require_emission(append_store_u32_vgpr_at_offset(
      words, slot_address, offsetof(ConSanMoiInlineAtomicReleaseSlot, dispatch_id), temporary,
      arch));
  if (!append_moi_report_dispatch_id_word(words, plan.dispatch_id, temporary,
                                          /*high_word=*/true, arch)) {
    require_emission(false, "ConSan MOI inline release could not materialize dispatch ID high");
  }
  require_emission(append_store_u32_vgpr_at_offset(
      words, slot_address,
      offsetof(ConSanMoiInlineAtomicReleaseSlot, dispatch_id) + sizeof(uint32_t), temporary, arch));
  sequence.append(instrumentation::build_s_wait_global_store0(arch));

  // A wave can issue the same release RMW from several lanes. The direct
  // object table can publish only one representative at a time, but the
  // winning transaction represents every losing lane with the same exact
  // dispatch/workgroup/address identity. Preserve fail-closed accounting for
  // actual direct-map collisions instead of treating benign same-object
  // coalescing as detector undercoverage.
  set_stage("same-object collision classification");
  require_emission(
      restore_exec(failed_exec) &&
      require_dispatch_field(offsetof(ConSanMoiInlineAtomicReleaseSlot, dispatch_id),
                             /*high_word=*/false) &&
      require_dispatch_field(offsetof(ConSanMoiInlineAtomicReleaseSlot, dispatch_id) +
                                 sizeof(uint32_t),
                             /*high_word=*/true) &&
      require_field(offsetof(ConSanMoiInlineAtomicReleaseSlot, workgroup_key),
                    vector_source_vgpr(workgroup_key)) &&
      require_field(offsetof(ConSanMoiInlineAtomicReleaseSlot, atomic_address),
                    vector_source_vgpr(stable_guest_address)) &&
      require_field(offsetof(ConSanMoiInlineAtomicReleaseSlot, atomic_address) + sizeof(uint32_t),
                    vector_source_vgpr(static_cast<uint16_t>(stable_guest_address + 1u))) &&
      save_exec(committed_exec));
  sequence.append(
      instrumentation::build_s_andn2_b64(kAmdGpuExecLo, failed_exec, committed_exec, arch));
  require_emission(append_atomic_fetch_add_one_u32(
      words, plan.report_buffer_address + offsetof(ConSanMoiReportHeader, inline_overflow_count),
      temporary, cas_new, arch));
  require_emission(restore_exec(original_exec) &&
                   append_restore_moi_special_state(words, plan.special_state, arch));

  set_stage("guest completion");
  if (emit_guest_instruction && !import_claimed_predecessor) {
    guest_instruction_offset = static_cast<uint32_t>(words.size() * sizeof(uint32_t));
    for (uint64_t offset = 0; offset < site.size; offset += sizeof(uint32_t)) {
      uint32_t word = 0;
      std::memcpy(&word, bytes.data() + site.file_offset + offset, sizeof(word));
      words.push_back(word);
    }
  }
  require_emission(append_moi_flat_load_wait(words, arch));
  sequence.append(instrumentation::build_s_wait_flat_store0(arch));

  set_stage("release commit");
  require_emission(append_save_moi_special_state(words, plan.special_state, arch) &&
                   restore_exec(claimed_exec));
  require_emission(
      append_moi_version_transition(words, sequence, exec_masks, slot_address,
                                    offsetof(ConSanMoiInlineAtomicReleaseSlot, version),
                                    prior_version, cas_new, cas_expected,
                                    /*desired_delta=*/2u, /*expected_delta=*/1u, arch) &&
      save_exec(committed_exec));
  sequence.append(
      instrumentation::build_s_andn2_b64(kAmdGpuExecLo, claimed_exec, committed_exec, arch));
  require_emission(
      append_atomic_fetch_add_one_u32(words,
                                      plan.report_buffer_address +
                                          offsetof(ConSanMoiReportHeader, inline_overflow_count),
                                      temporary, cas_new, arch) &&
      restore_exec(original_exec) &&
      append_restore_moi_special_state(words, plan.special_state, arch));
  return require_emission.finish(arch);
}

[[nodiscard]] bool
inline_atomic_scalar_spill_aliases_guest_address(const ConSanMoiAtomicAddressPlan &address_plan,
                                                 uint16_t scalar_base, uint16_t scalar_count) {
  return (address_plan.scalar_base_sgpr &&
          range_overlaps(scalar_base, scalar_count, *address_plan.scalar_base_sgpr, 2u)) ||
         (address_plan.scalar_offset_sgpr &&
          range_overlaps(scalar_base, scalar_count, *address_plan.scalar_offset_sgpr, 1u));
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_inline_atomic_ordering_cave_words(
    std::span<const uint8_t> bytes, const MoiAtomicEvidenceSourceView &source,
    const ConSanMoiAtomicAddressPlan &address_plan, const MoiInlineAtomicEmissionPlan &plan,
    const VgprSpillSequence *spill, const SgprSpillSequence *scalar_spill, rj_code_arch_t arch,
    uint32_t &guest_instruction_offset, std::vector<std::string> &errors,
    std::span<const uint32_t> trailing_guest_words) {
  const ConSanAtomicSite &site = source.site;
  const std::optional<ConSanMoiAtomicEventKind> event_kind =
      moi_atomic_event_kind(source.sequence->memory_role);
  if (!event_kind)
    return std::nullopt;
  const bool is_rmw = source.is_rmw();
  const uint16_t scratch_vgpr = plan.scratch_vgpr;
  const uint16_t required_scratch_count = plan.scratch_vgpr_count;
  const bool scalar_persistent = plan.persistent_sgprs.complete();

  const uint64_t slot_base =
      plan.report_buffer_address + plan.report_layout.inline_atomic_release_slots_offset;
  const uint64_t snapshot_table_base =
      plan.report_buffer_address + plan.report_layout.inline_causal_snapshots_offset;
  const uint64_t token_table_base =
      plan.report_buffer_address + plan.report_layout.inline_acquired_epoch_token_slots_offset;
  const uint16_t value_vgpr = static_cast<uint16_t>(scratch_vgpr + 2u);
  const uint16_t workgroup_key_vgpr = static_cast<uint16_t>(scratch_vgpr + 3u);
  const uint16_t producer_owner_vgpr = static_cast<uint16_t>(scratch_vgpr + 4u);
  const uint16_t token_value_vgpr = static_cast<uint16_t>(scratch_vgpr + 5u);
  const uint16_t temporary_vgpr = static_cast<uint16_t>(scratch_vgpr + 6u);
  const uint16_t address_vgpr = address_plan.result_address_vgpr;
  const bool is_compare_exchange = consan_atomic_is_compare_exchange(site);

  std::vector<uint32_t> words;
  InstructionSequence sequence(words);
  MoiEmissionRequirement require_emission(sequence, errors);
  words.reserve(
      site.size / sizeof(uint32_t) + trailing_guest_words.size() + 96u +
      (spill ? spill->save_words.size() + spill->restore_words.size() : 0u) +
      (scalar_spill ? scalar_spill->save_words.size() + scalar_spill->restore_words.size() : 0u));
  if (spill)
    words.insert(words.end(), spill->save_words.begin(), spill->save_words.end());
  if (scalar_spill) {
    words.insert(words.end(), scalar_spill->save_words.begin(), scalar_spill->save_words.end());
    // The borrowed scalar window is already durably saved. When it contains
    // an address operand, reconstruct the guest values before address
    // materialization snapshots them into the scratch VGPR tail. The cave is
    // then free to reuse the same scalar window, and the ordinary epilogue
    // restores its guest-visible contents a second time. This is deliberately
    // conditional: most sites do not need the extra reload transaction.
    if (inline_atomic_scalar_spill_aliases_guest_address(address_plan, scalar_spill->sgpr_base,
                                                         scalar_spill->sgpr_count)) {
      words.insert(words.end(), scalar_spill->restore_words.begin(),
                   scalar_spill->restore_words.end());
    }
  }
  const auto append_spill_restore = [&]() {
    // Scalar restore uses the scratch VGPR window, so restore application
    // VGPRs only after every spilled SGPR has been reconstructed.
    if (scalar_spill)
      words.insert(words.end(), scalar_spill->restore_words.begin(),
                   scalar_spill->restore_words.end());
    if (spill)
      words.insert(words.end(), spill->restore_words.begin(), spill->restore_words.end());
  };
  const auto finish_words = [&]() -> std::optional<std::vector<uint32_t>> {
    if (!sequence.finish(arch))
      return std::nullopt;
    append_spill_restore();
    return words;
  };
  if (address_plan.requires_materialization()) {
    const auto address_words = build_consan_moi_atomic_address_materialization(
        address_plan, plan.special_state.vcc_save_sgpr, plan.special_state.scc_save_sgpr, arch);
    if (!address_words) {
      errors.emplace_back("ConSan MOI inline atomic patch could not materialize its address");
      return std::nullopt;
    }
    words.insert(words.end(), address_words->begin(), address_words->end());
  }
  if (scalar_persistent) {
    const uint16_t materialized_owner =
        static_cast<uint16_t>(scratch_vgpr + required_scratch_count - 4u);
    const uint16_t materialized_epoch = static_cast<uint16_t>(materialized_owner + 1u);
    words.push_back(build_v_mov_b32_e32(materialized_owner, *plan.persistent_sgprs.owner(), arch));
    words.push_back(build_v_mov_b32_e32(materialized_epoch, *plan.persistent_sgprs.epoch(), arch));
  }
  if (plan.private_state) {
    const MoiInlineAtomicPrivateStatePlan &private_state = *plan.private_state;
    const uint16_t materialized_workgroup_key =
        static_cast<uint16_t>(scratch_vgpr + required_scratch_count - 6u);
    const uint16_t private_temporary =
        static_cast<uint16_t>(scratch_vgpr + required_scratch_count - 5u);
    const uint16_t materialized_owner =
        static_cast<uint16_t>(scratch_vgpr + required_scratch_count - 4u);
    const uint16_t materialized_epoch = static_cast<uint16_t>(materialized_owner + 1u);
    const auto load_epoch = instrumentation::build_private_load_b32(
        materialized_epoch, private_state.epoch_offset, arch);
    const auto load_workgroup = instrumentation::build_private_load_b32(
        materialized_workgroup_key, private_state.workgroup_key_offset, arch);
    const auto wait_private = instrumentation::build_s_wait_private_load0(arch);
    require_emission.append(
        "ConSan MOI inline atomic patch could not load private epoch/workgroup state", load_epoch,
        load_workgroup, wait_private);
    if (private_state.workitem_owner) {
      const ConSanTargetProfile *target = consan_target_profile(arch);
      require_emission(
          target != nullptr &&
              consan_detail::append_moi_workitem_owner_derivation(
                  words, {.plan = *private_state.workitem_owner, .result_vgpr = materialized_owner},
                  *target),
          "ConSan MOI inline atomic patch could not load private owner state");
    } else if (private_state.resident_wave_owner) {
      const uint16_t owner_sgpr = private_state.resident_wave_owner->destination_sgpr;
      if (private_state.resident_wave_owner_is_borrowed) {
        require_emission.append(
            "ConSan MOI inline atomic patch could not save borrowed owner state",
            instrumentation::build_v_writelane_b32(private_temporary, owner_sgpr, 0u, arch));
      }
      const ConSanTargetProfile *target = consan_target_profile(arch);
      require_emission(target != nullptr && consan_detail::append_moi_resident_wave_owner(
                                                words, *private_state.resident_wave_owner, *target),
                       "ConSan MOI inline atomic patch could not derive resident-wave owner");
      words.push_back(build_v_mov_b32_e32(materialized_owner, owner_sgpr, arch));
      if (private_state.resident_wave_owner_is_borrowed) {
        const auto restore_owner =
            instrumentation::build_v_readlane_b32(owner_sgpr, private_temporary, 0u, arch);
        const auto wait_owner = instrumentation::build_valu_to_salu_dependency_wait(arch);
        require_emission.append(
            "ConSan MOI inline atomic patch could not restore borrowed owner state", restore_owner,
            wait_owner);
      }
    } else {
      errors.emplace_back("ConSan MOI inline atomic patch has no private owner derivation");
      return std::nullopt;
    }
    if (private_state.dispatch_id_offset) {
      const uint16_t dispatch_id_sgpr = *private_state.dispatch_id_sgpr;
      const auto load_dispatch_low = instrumentation::build_private_load_b32(
          private_temporary, *private_state.dispatch_id_offset, arch);
      const auto load_dispatch_high = instrumentation::build_private_load_b32(
          materialized_workgroup_key, *private_state.dispatch_id_offset + SpillManager::kSlotBytes,
          arch);
      const auto wait_dispatch = instrumentation::build_s_wait_private_load0(arch);
      const auto read_dispatch_low =
          instrumentation::build_v_readfirstlane_b32(dispatch_id_sgpr, private_temporary, arch);
      const auto read_dispatch_high = instrumentation::build_v_readfirstlane_b32(
          static_cast<uint16_t>(dispatch_id_sgpr + 1u), materialized_workgroup_key, arch);
      const auto wait_scalar = instrumentation::build_valu_to_salu_dependency_wait(arch);
      require_emission.append("ConSan MOI inline atomic patch could not reload private dispatch ID",
                              load_dispatch_low, load_dispatch_high, wait_dispatch,
                              read_dispatch_low, read_dispatch_high, wait_scalar);
      sequence.append(load_workgroup, wait_private);
    }
  }
  if (*event_kind == ConSanMoiAtomicEventKind::Release && !is_compare_exchange) {
    // An ISA release RMW extends the claimed predecessor's release sequence.
    // A language-level ordinary release store starts a new publication: it
    // must reserve and stage metadata before the guest store, but must not
    // inherit an unrelated predecessor's causal frontier.
    const bool import_claimed_predecessor = is_rmw;
    require_emission(append_inline_versioned_release_transaction(
                         words, bytes, *event_kind, site, address_plan, plan, arch,
                         /*emit_guest_instruction=*/true, import_claimed_predecessor,
                         /*predicate_on_compare_exchange_success=*/false, guest_instruction_offset,
                         errors),
                     "ConSan MOI inline release could not emit a versioned causal transaction");
    return finish_words();
  }
  // CDNA4 can let a following wave reach the guest RMW before the preceding
  // wave has published an initially empty release slot. Serialize non-CAS
  // AcquireRelease RMWs with the release reservation itself: the claimed odd
  // slot retains the predecessor payload across the guest, acquire import,
  // causal snapshot, and successor commit.
  const bool claimed_acquire_release =
      *event_kind == ConSanMoiAtomicEventKind::AcquireRelease && !is_compare_exchange;
  if (claimed_acquire_release) {
    require_emission(
        append_inline_versioned_release_transaction(
            words, bytes, *event_kind, site, address_plan, plan, arch,
            /*emit_guest_instruction=*/true,
            /*import_claimed_predecessor=*/true,
            /*predicate_on_compare_exchange_success=*/false, guest_instruction_offset, errors),
        "ConSan MOI inline acquire-release could not emit a claimed causal transaction");
    return finish_words();
  }

  // A returning release CAS exposes its success result only after the guest
  // instruction, so post-guest publication used to leave a window in which
  // an acquire could observe the new value while the release slot was still
  // empty. Claim the slot first, execute the CAS once inside that transaction,
  // and restore the prior slot unchanged for dynamically failed comparisons.
  const bool claimed_compare_exchange_release =
      is_compare_exchange && *event_kind != ConSanMoiAtomicEventKind::Acquire;
  if (claimed_compare_exchange_release) {
    require_emission(
        append_inline_versioned_release_transaction(
            words, bytes, *event_kind, site, address_plan, plan, arch,
            /*emit_guest_instruction=*/true,
            /*import_claimed_predecessor=*/true,
            /*predicate_on_compare_exchange_success=*/true, guest_instruction_offset, errors),
        "ConSan MOI inline compare-exchange could not emit a claimed causal transaction");
    return finish_words();
  }

  // A compiler-lowered atomic load plus its acquire cache suffix is safe to
  // execute while the metadata slot is reserved and must not race a release
  // publisher through the old pre/post-version fail-closed window. RMW
  // acquires cannot use this path: an acquire-only RMW also extends a language
  // release sequence and needs a separate successor-publication contract.
  const bool claimed_read_only_acquire =
      !is_rmw && *event_kind == ConSanMoiAtomicEventKind::Acquire;
  if (claimed_read_only_acquire) {
    require_emission(
        append_inline_versioned_release_transaction(
            words, bytes, *event_kind, site, address_plan, plan, arch,
            /*emit_guest_instruction=*/true,
            /*import_claimed_predecessor=*/true,
            /*predicate_on_compare_exchange_success=*/false, guest_instruction_offset, errors,
            trailing_guest_words),
        "ConSan MOI inline read-only acquire could not emit a claimed causal transaction");
    return finish_words();
  }

  // A successful guest acquire must observe one immutable release
  // transaction. Retain the first release version across the guest; the
  // importer rechecks it only after loading the complete release payload and
  // causal snapshot. The 24-VGPR spill-backed plan keeps this value disjoint
  // from the final address pair reserved by address materialization.
  if (*event_kind == ConSanMoiAtomicEventKind::Acquire ||
      *event_kind == ConSanMoiAtomicEventKind::AcquireRelease) {
    const uint16_t version_before = static_cast<uint16_t>(
        scratch_vgpr + inline_register_layout::AtomicOrdering::release_version_before);
    const uint16_t version_retry_count = static_cast<uint16_t>(plan.exec_save_sgpr + 20u);
    const auto restore_original_exec = instrumentation::build_s_mov_b64(
        kAmdGpuExecLo, static_cast<uint16_t>(plan.exec_save_sgpr + 12u), arch);
    require_emission(
        restore_original_exec && append_save_moi_special_state(words, plan.special_state, arch) &&
            append_inline_workgroup_key(words, plan.workgroup_sources, plan.workgroup_key_registers,
                                        workgroup_key_vgpr, temporary_vgpr, value_vgpr,
                                        /*original_exec_save_offset=*/12u, arch) &&
            append_inline_atomic_release_slot_address(
                words, slot_base, plan.report_layout.inline_atomic_release_capacity, address_vgpr,
                scratch_vgpr, arch),
        "ConSan MOI inline acquire could not snapshot release version");
    // A release publisher keeps the direct-mapped slot odd while its guest
    // RMW and causal snapshot are in flight. In particular, a consumer can
    // observe the guest RMW before that publisher commits its metadata. Wait
    // for an even version before executing an acquire guest so that its saved
    // source version and a following AcquireRelease publication cannot race
    // the still-active publisher. The bounded retry preserves fail-closed
    // behavior for malformed or abandoned odd slots.
    constexpr uint16_t kScalarLiteralSource = 255u;
    sequence.append(build_s_mov_b32(version_retry_count, kScalarLiteralSource, arch),
                    kConSanMoiInlineMetadataPublicationRetryLimit);
    const InstructionSequence::Label version_retry_begin = sequence.mark_label();
    require_emission(append_atomic_load_u32(words, scratch_vgpr, version_before, arch),
                     "ConSan MOI inline acquire could not load its release version");
    require_emission.append(
        "ConSan MOI inline acquire could not encode its version retry",
        instrumentation::build_v_and_b32_literal(temporary_vgpr, 1u, version_before, arch),
        instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), temporary_vgpr,
                                                arch));
    const InstructionSequence::Label version_retry_exit = sequence.make_label();
    sequence.branch(version_retry_exit, InstructionSequence::BranchKind::VccNonzero)
        .append(build_s_sleep(kConSanMoiInlineMetadataPublicationSleepDelay, arch),
                instrumentation::build_s_sub_u32(version_retry_count, version_retry_count,
                                                 scalar_positive_inline_u32(1), arch),
                instrumentation::build_s_cmp_lg_u32(version_retry_count,
                                                    scalar_positive_inline_u32(0), arch))
        .branch(version_retry_exit, InstructionSequence::BranchKind::SccZero)
        .branch(version_retry_begin, InstructionSequence::BranchKind::Unconditional)
        .bind_label(version_retry_exit);
    if (!sequence.finish(arch)) {
      errors.emplace_back("ConSan MOI inline acquire version exit is out of branch range");
      return std::nullopt;
    }
    words.push_back(*restore_original_exec);
    require_emission(append_restore_moi_special_state(words, plan.special_state, arch),
                     "ConSan MOI inline acquire could not restore pre-guest state");
  }
  guest_instruction_offset = static_cast<uint32_t>(words.size() * sizeof(uint32_t));
  for (uint64_t offset = 0; offset < site.size; offset += sizeof(uint32_t)) {
    uint32_t word = 0;
    std::memcpy(&word, bytes.data() + site.file_offset + offset, sizeof(word));
    words.push_back(word);
  }
  if (is_rmw) {
    sequence.require(append_moi_flat_load_wait(words, arch));
  } else {
    if (trailing_guest_words.empty()) {
      errors.emplace_back(
          "ConSan MOI inline ordinary acquire patch lacks its ordered guest suffix");
      return std::nullopt;
    }
    words.insert(words.end(), trailing_guest_words.begin(), trailing_guest_words.end());
  }

  require_emission(append_save_moi_special_state(words, plan.special_state, arch) &&
                       append_inline_workgroup_key(words, plan.workgroup_sources,
                                                   plan.workgroup_key_registers, workgroup_key_vgpr,
                                                   temporary_vgpr, value_vgpr,
                                                   /*original_exec_save_offset=*/12u, arch) &&
                       append_inline_atomic_release_slot_address(
                           words, slot_base, plan.report_layout.inline_atomic_release_capacity,
                           address_vgpr, scratch_vgpr, arch),
                   "ConSan MOI inline atomic patch could not derive an address-indexed slot");

  if (*event_kind == ConSanMoiAtomicEventKind::Acquire ||
      *event_kind == ConSanMoiAtomicEventKind::AcquireRelease)
    sequence.require(append_inline_atomic_acquire_import(
        words, plan, scratch_vgpr, address_vgpr, workgroup_key_vgpr, producer_owner_vgpr,
        token_value_vgpr, temporary_vgpr, snapshot_table_base,
        plan.report_layout.inline_causal_snapshot_capacity, token_table_base,
        plan.report_layout.inline_acquired_epoch_token_capacity,
        /*source_slot_claimed=*/false,
        /*release_sequence_only=*/false, plan.inline_access_present, arch, errors));

  if (*event_kind != ConSanMoiAtomicEventKind::Acquire) {
    // AcquireRelease must first import the release observed by the guest RMW;
    // all release-capable events then publish that complete ancestry through
    // the same ABI-v6 transaction. CAS publication is narrowed to lanes whose
    // dynamic compare succeeded inside the transaction, after its original
    // EXEC mask has been preserved.
    require_emission(append_restore_moi_special_state(words, plan.special_state, arch) &&
                         append_inline_versioned_release_transaction(
                             words, bytes, *event_kind, site, address_plan, plan, arch,
                             /*emit_guest_instruction=*/false,
                             /*import_claimed_predecessor=*/false,
                             /*predicate_on_compare_exchange_success=*/is_compare_exchange,
                             guest_instruction_offset, errors),
                     "ConSan MOI inline release could not emit a versioned causal transaction");
    return finish_words();
  }

  require_emission.append(
      "ConSan MOI inline atomic patch could not restore original EXEC",
      instrumentation::build_s_mov_b64(kAmdGpuExecLo,
                                       static_cast<uint16_t>(plan.exec_save_sgpr + 12u), arch));
  require_emission(append_restore_moi_special_state(words, plan.special_state, arch),
                   "ConSan MOI inline atomic patch could not restore VCC/SCC");
  return finish_words();
}
} // namespace consan_moi_impl
} // namespace rocjitsu
