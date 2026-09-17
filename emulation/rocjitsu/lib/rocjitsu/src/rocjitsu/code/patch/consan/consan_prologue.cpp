// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_prologue.h"

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/consan/consan_access_target.h"
#include "rocjitsu/code/patch/consan/consan_capability_contract.h"
#include "rocjitsu/code/patch/consan/consan_descriptor.h"
#include "rocjitsu/code/patch/consan/consan_lowering_plan.h"
#include "rocjitsu/code/patch/consan/consan_native_abi.h"
#include "rocjitsu/code/patch/consan/consan_packed_fields.h"
#include "rocjitsu/code/patch/consan/consan_probe_contracts.h"
#include "rocjitsu/code/patch/consan/consan_register_allocation.h"
#include "rocjitsu/code/patch/consan/consan_relocation.h"
#include "rocjitsu/code/patch/consan/consan_report_emission.h"
#include "rocjitsu/code/patch/consan/consan_resource.h"
#include "rocjitsu/code/patch/consan/consan_runtime_workgroup_gate.h"
#include "rocjitsu/code/patch/consan/consan_shared_lowering.h"
#include "rocjitsu/code/patch/consan/consan_text_relocation.h"
#include "rocjitsu/code/patch/instruction_sequence.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"
#include "rocjitsu/code/patch/spill_manager.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <ranges>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace rocjitsu::consan::detail {

using detail::OwnerEpochPrologueEmissionPlan;
using detail::PrivateEpochPrologueEmissionPlan;

[[nodiscard]] DispatchIdCapture dispatch_id_capture(const OperatingPoint &point) {
  if (auto sgpr = point.dispatch_sgpr.base())
    return DispatchIdCapture::in_sgprs(*sgpr);
  return {};
}
using detail::has_runtime_hardware_dispatch_id;
using detail::range_overlaps;

[[nodiscard]] bool append_entry_salu_write(std::vector<uint32_t> &words, uint32_t word,
                                           rj_code_arch_t arch) {
  return InstructionSequence(words).emit_all(word,
                                             instrumentation::build_salu_dependency_delay(arch));
}

[[nodiscard]] bool append_entry_scalar_backup(std::vector<uint32_t> &words,
                                              const EntryScalarBackup &backup, bool restore,
                                              rj_code_arch_t arch,
                                              std::vector<std::string> &errors) {
  if (!is_capability_arch(arch) || !backup.is_well_formed(kMaxVgprs, ordinary_sgpr_limit(arch))) {
    errors.emplace_back("ConSan entry scalar backup has an invalid register layout");
    return false;
  }
  InstructionSequence sequence(words);
  EmissionRequirement require_emission(sequence, errors);
  for (uint16_t index = 0; index < backup.sgpr_count; ++index) {
    const uint16_t sgpr = static_cast<uint16_t>(backup.sgpr_base + index);
    const auto encoded =
        restore ? instrumentation::build_v_readlane_b32(sgpr, backup.vgpr, index, arch)
                : instrumentation::build_v_writelane_b32(backup.vgpr, sgpr, index, arch);
    require_emission.append("ConSan entry scalar backup could not encode a lane transfer", encoded);
  }
  if (restore)
    require_emission.append(
        "ConSan entry scalar backup could not encode its restore dependency wait",
        build_sgpr_restore_dependency_wait(arch));
  return sequence.finish();
}

[[nodiscard]] bool append_dispatch_id_capture_and_restore(
    std::vector<uint32_t> &words, const DispatchIdPreloadPlan &plan, DispatchIdCapture capture,
    bool restore_system_sgprs, rj_code_arch_t arch, std::vector<std::string> &errors) {
  InstructionSequence sequence(words);
  EmissionRequirement require_emission(sequence, errors);
  if (!plan.supported()) {
    errors.emplace_back("ConSan dispatch-ID prologue received an unsupported preload plan");
    return false;
  }
  if (!capture.present() ||
      (capture.sgpr() && (*capture.sgpr() < 20u || *capture.sgpr() % 2u != 0u ||
                          static_cast<uint32_t>(*capture.sgpr()) + 2u > kMaxSgprs)) ||
      (capture.vgpr() && static_cast<uint32_t>(*capture.vgpr()) + 2u > kMaxVgprs) ||
      (plan.identity_salt_sgpr &&
       static_cast<uint32_t>(*plan.identity_salt_sgpr) + 2u > kMaxSgprs) ||
      static_cast<uint32_t>(plan.dispatch_id_sgpr) + 2u > kMaxSgprs) {
    errors.emplace_back("ConSan dispatch-ID prologue has an invalid persistent or source pair");
    return false;
  }

  if (capture.sgpr()) {
    require_emission(
        append_entry_salu_write(
            words, build_s_mov_b32(*capture.sgpr(), plan.dispatch_id_sgpr, arch), arch) &&
            append_entry_salu_write(
                words,
                build_s_mov_b32(static_cast<uint16_t>(*capture.sgpr() + 1u),
                                static_cast<uint16_t>(plan.dispatch_id_sgpr + 1u), arch),
                arch),
        "ConSan dispatch-ID prologue cannot encode its scalar capture");
    if (plan.identity_salt_sgpr) {
      const auto mix = instrumentation::build_s_xor_b64(*capture.sgpr(), *capture.sgpr(),
                                                        *plan.identity_salt_sgpr, arch);
      require_emission(mix && append_entry_salu_write(words, *mix, arch),
                       "ConSan dispatch-ID prologue cannot encode its launch-identity mix");
    }
    // Hardware dispatch ID zero is valid (and common for the first packet),
    // while ConSan reserves zero as the empty metadata sentinel. Store the
    // modulo-2^64 successor used by every downstream comparison.
    require_emission(
        append_entry_salu_write(
            words,
            build_s_add_u32(*capture.sgpr(), *capture.sgpr(), scalar_positive_inline_u32(1), arch),
            arch) &&
            append_entry_salu_write(words,
                                    build_s_addc_u32(static_cast<uint16_t>(*capture.sgpr() + 1u),
                                                     static_cast<uint16_t>(*capture.sgpr() + 1u),
                                                     scalar_positive_inline_u32(0), arch),
                                    arch),
        "ConSan dispatch-ID prologue cannot encode its scalar successor");
  } else {
    require_emission.append("ConSan dispatch-ID prologue cannot encode its temporary VGPR capture",
                            build_v_mov_b32_e32(*capture.vgpr(), plan.dispatch_id_sgpr, arch),
                            build_v_mov_b32_e32(static_cast<uint16_t>(*capture.vgpr() + 1u),
                                                static_cast<uint16_t>(plan.dispatch_id_sgpr + 1u),
                                                arch));
    if (plan.identity_salt_sgpr) {
      require_emission.append(
          "ConSan dispatch-ID prologue cannot encode its launch-identity mix",
          instrumentation::build_v_xor_b32(*capture.vgpr(), *plan.identity_salt_sgpr,
                                           *capture.vgpr(), arch),
          instrumentation::build_v_xor_b32(static_cast<uint16_t>(*capture.vgpr() + 1u),
                                           static_cast<uint16_t>(*plan.identity_salt_sgpr + 1u),
                                           static_cast<uint16_t>(*capture.vgpr() + 1u), arch));
    }
    require_emission.append(
        "ConSan dispatch-ID prologue cannot encode its persistent VGPR successor",
        instrumentation::build_v_add_u64_literal(*capture.vgpr(), 1u, arch));
  }
  for (uint16_t restore_index = 0; restore_index < plan.guest_restore_count; ++restore_index) {
    const uint16_t destination = plan.guest_restore_destinations[restore_index];
    const std::optional<uint16_t> source = plan.guest_restore_sources[restore_index];
    if (!source || *source >= kMaxSgprs) {
      errors.emplace_back("ConSan dispatch-ID prologue has an invalid guest restore source");
      return false;
    }
    require_emission(
        append_entry_salu_write(words, build_s_mov_b32(destination, *source, arch), arch),
        "ConSan dispatch-ID prologue cannot encode a guest-restore dependency delay");
  }
  if (restore_system_sgprs) {
    for (uint16_t i = 0; i < plan.shifted_system_sgpr_count; ++i) {
      const uint16_t destination = static_cast<uint16_t>(plan.original_user_sgpr_count + i);
      require_emission(
          append_entry_salu_write(
              words,
              build_s_mov_b32(destination,
                              static_cast<uint16_t>(destination + plan.system_sgpr_shift), arch),
              arch),
          "ConSan dispatch-ID prologue cannot encode a system-SGPR restore");
    }
  }
  if (plan.requires_kernarg_reload()) {
    if (!arch_supports_kernarg_preload_overflow_recovery(arch) || plan.kernarg_reload_count > 4u) {
      errors.emplace_back("ConSan dispatch-ID prologue has an unsupported kernarg reload");
      return false;
    }
    for (uint16_t i = 0; i < plan.kernarg_reload_count; ++i) {
      require_emission.append(
          "ConSan dispatch-ID prologue cannot encode a kernarg reload",
          instrumentation::build_s_load_dword(
              static_cast<uint16_t>(plan.kernarg_reload_sgpr + i), plan.kernarg_reload_base_sgpr,
              static_cast<uint32_t>(plan.kernarg_reload_offset_dwords + i) * sizeof(uint32_t),
              arch));
    }
    require_emission.append("ConSan dispatch-ID prologue cannot encode its kernarg wait",
                            instrumentation::build_s_wait_scalar_load0(arch));
  }
  return sequence.finish();
}

[[nodiscard]] std::optional<std::vector<std::pair<uint16_t, uint16_t>>>
cdna_guest_workgroup_payload_copies(const WorkgroupSources &sources, rj_code_arch_t arch) {
  if (!sources.cdna_full_payload_base && !sources.cdna_guest_payload_base &&
      sources.cdna_guest_payload_mask == 0u) {
    return std::vector<std::pair<uint16_t, uint16_t>>{};
  }
  if (!sources.cdna_full_payload_base || !sources.cdna_guest_payload_base ||
      !arch_supports_kernarg_preload_overflow_recovery(arch)) {
    return std::nullopt;
  }

  std::vector<std::pair<uint16_t, uint16_t>> copies;
  uint16_t destination = *sources.cdna_guest_payload_base;
  for (uint16_t dimension = 0; dimension < 3u; ++dimension) {
    if ((sources.cdna_guest_payload_mask & (1u << dimension)) == 0u)
      continue;
    const uint16_t source = static_cast<uint16_t>(*sources.cdna_full_payload_base + dimension);
    copies.emplace_back(destination, source);
    ++destination;
  }
  if ((sources.cdna_guest_payload_mask & (1u << 3u)) != 0u) {
    const uint16_t source = static_cast<uint16_t>(*sources.cdna_full_payload_base + 3u);
    copies.emplace_back(destination, source);
  }
  return copies;
}

[[nodiscard]] bool append_cdna_full_workgroup_payload_restore(std::vector<uint32_t> &words,
                                                              const WorkgroupSources &sources,
                                                              rj_code_arch_t arch,
                                                              std::vector<std::string> &errors) {
  const auto copies = cdna_guest_workgroup_payload_copies(sources, arch);
  if (!copies) {
    errors.emplace_back("ConSan found an incomplete AMDHSA workgroup-ID payload repair");
    return false;
  }
  InstructionSequence sequence(words);
  EmissionRequirement require_emission(sequence, errors);
  for (const auto &[destination, source] : *copies) {
    if (destination != source)
      require_emission(
          append_entry_salu_write(words, build_s_mov_b32(destination, source, arch), arch),
          "ConSan could not restore a guest AMDHSA workgroup payload SGPR");
  }
  return sequence.finish();
}

[[nodiscard]] bool append_cdna_semantic_entry_scalar_spill_overrides(
    std::vector<uint32_t> &words, const SgprSpillSequence &spill, const WorkgroupSources &sources,
    rj_code_arch_t arch, std::vector<std::string> &errors) {
  const auto copies = cdna_guest_workgroup_payload_copies(sources, arch);
  if (!copies) {
    errors.emplace_back("ConSan found an incomplete AMDHSA entry-scalar backup repair");
    return false;
  }
  InstructionSequence sequence(words);
  EmissionRequirement require_emission(sequence, errors);
  for (const auto &[destination, source] : *copies) {
    if (destination == source || destination < spill.sgpr_base ||
        destination >= static_cast<uint32_t>(spill.sgpr_base) + spill.sgpr_count) {
      continue;
    }
    require_emission.append(
        "ConSan could not preserve a semantic AMDHSA entry-scalar backup slot",
        build_sgpr_spill_slot_override_sequence(spill, destination, source, arch));
  }
  return sequence.finish();
}

[[nodiscard]] bool append_exact_workgroup_capture(std::vector<uint32_t> &words,
                                                  PersistentWorkgroupRegisters scalar_destinations,
                                                  PersistentWorkgroupRegisters vector_destinations,
                                                  const WorkgroupSources &sources,
                                                  rj_code_arch_t arch,
                                                  std::vector<std::string> &errors) {
  if (!scalar_destinations.empty() && !vector_destinations.empty()) {
    errors.emplace_back(
        "ConSan exact workgroup-tuple prologue requires one complete register tuple");
    return false;
  }
  if (scalar_destinations.empty() && vector_destinations.empty())
    return true;

  const std::array<const WorkgroupSource *, 4> source_tuple = {&sources.x, &sources.y, &sources.z,
                                                               &sources.cluster_workgroup_id};
  const std::array<std::optional<uint16_t>, 4> scalar_tuple = scalar_destinations.values();
  const std::array<std::optional<uint16_t>, 4> vector_tuple = vector_destinations.values();
  InstructionSequence sequence(words);
  EmissionRequirement require_emission(sequence, errors);
  for (size_t index = 0; index < source_tuple.size(); ++index) {
    if (!scalar_tuple[index] && !vector_tuple[index])
      continue;
    const WorkgroupSource &source = *source_tuple[index];
    if (!source.is_well_formed()) {
      errors.emplace_back("ConSan exact workgroup-tuple prologue requires all launch coordinates");
      return false;
    }
    if (!source.has_value()) {
      if (index != 3u) {
        errors.emplace_back(
            "ConSan exact workgroup-tuple prologue requires all launch coordinates");
        return false;
      }
      // A code-object-wide CDNA5 tuple reserves the cluster coordinate when
      // any owning kernel consumes it. Ordinary kernels in the same object do
      // not receive that launch input; zero is their exact cluster identity.
      if (scalar_tuple[index]) {
        require_emission(
            append_entry_salu_write(
                words, build_s_mov_b32(*scalar_tuple[index], scalar_positive_inline_u32(0), arch),
                arch),
            "ConSan exact scalar workgroup-tuple prologue could not zero an absent "
            "cluster coordinate");
      } else {
        require_emission(sequence.emit(
            build_v_mov_b32_e32(*vector_tuple[index], scalar_positive_inline_u32(0), arch)));
      }
      continue;
    }
    if (scalar_tuple[index]) {
      if (!source.scalar_src) {
        errors.emplace_back("ConSan exact scalar workgroup tuple requires scalar launch sources");
        return false;
      }
      const uint16_t destination = *scalar_tuple[index];
      const uint16_t source_operand = *source.scalar_src;
      require_emission(
          append_entry_salu_write(words, build_s_mov_b32(destination, source_operand, arch),
                                  arch) &&
              (source.right_shift == 0u ||
               append_entry_salu_write(
                   words,
                   build_s_lshr_b32(destination, destination,
                                    scalar_positive_inline_u32(source.right_shift), arch),
                   arch)) &&
              ((source.low_bit_count == 0u || source.low_bit_count == 32u) ||
               (append_entry_salu_write(
                    words,
                    build_s_lshl_b32(destination, destination,
                                     scalar_positive_inline_u32(32u - source.low_bit_count), arch),
                    arch) &&
                append_entry_salu_write(
                    words,
                    build_s_lshr_b32(destination, destination,
                                     scalar_positive_inline_u32(32u - source.low_bit_count), arch),
                    arch))),
          "ConSan exact scalar workgroup-tuple prologue could not copy a launch source");
    } else if (vector_tuple[index]) {
      const uint16_t destination = *vector_tuple[index];
      require_emission(
          detail::append_workgroup_source_value(words, source, destination, arch),
          "ConSan exact vector workgroup-tuple prologue could not copy a launch source");
    }
  }
  return sequence.finish();
}

[[nodiscard]] std::optional<std::vector<uint32_t>>
build_owner_epoch_prologue_words(const OwnerEpochPrologueEmissionPlan &plan, rj_code_arch_t arch,
                                 std::vector<std::string> &errors) {
  if (!plan.is_well_formed()) {
    errors.emplace_back("ConSan owner/epoch prologue has an invalid emission plan");
    return std::nullopt;
  }
  const uint16_t owner_vgpr = plan.vgpr_state.owner_epoch.owner;
  const uint16_t epoch_vgpr = plan.vgpr_state.owner_epoch.epoch;
  const uint16_t owner_shift_bits = plan.owner_shift_bits;
  const OwnerSource owner_source = plan.owner_source;
  const std::optional<uint16_t> owner_sgpr = plan.owner_sgpr;
  const std::optional<uint16_t> persistent_owner_sgpr = plan.persistent_sgprs.owner();
  const std::optional<uint16_t> persistent_epoch_sgpr = plan.persistent_sgprs.epoch();
  const PersistentWorkgroupRegisters exact_workgroup_sgprs = plan.persistent_sgprs.exact_workgroup;
  const PersistentWorkgroupRegisters exact_workgroup_vgprs = plan.vgpr_state.exact_workgroup;
  const std::optional<DispatchIdPreloadPlan> &dispatch_plan = plan.dispatch_plan;
  const DispatchIdCapture dispatch_capture = plan.dispatch_capture;
  const std::optional<EntryScalarBackup> &entry_scalar_backup = plan.entry_scalar_backup;
  const std::optional<WorkgroupSources> &workgroup_sources = plan.workgroup_sources;
  std::vector<uint32_t> words;
  words.reserve(48u + (entry_scalar_backup
                           ? static_cast<size_t>(entry_scalar_backup->sgpr_count) * 4u + 1u
                           : 0u));
  InstructionSequence sequence(words);
  // A newly inserted dispatch-ID preload shifts CDNA system SGPRs upward. The
  // dispatch repair below copies those values back to the guest ABI locations
  // and therefore overwrites part of the shifted tuple. Persist the exact
  // workgroup identity before that destructive repair.
  const bool capture_workgroup_before_dispatch =
      dispatch_plan && workgroup_sources && workgroup_sources->cdna_full_payload_base &&
      (exact_workgroup_sgprs.complete() || exact_workgroup_vgprs.complete());
  if (capture_workgroup_before_dispatch &&
      !append_exact_workgroup_capture(words, exact_workgroup_sgprs, exact_workgroup_vgprs,
                                      *workgroup_sources, arch, errors)) {
    return std::nullopt;
  }
  if (dispatch_plan &&
      !append_dispatch_id_capture_and_restore(
          words, *dispatch_plan, dispatch_capture,
          !workgroup_sources || !workgroup_sources->cdna_full_payload_base.has_value(), arch,
          errors)) {
    return std::nullopt;
  }
  // The entry backup preserves the guest-visible ABI values borrowed by the
  // remainder of this prologue.  When ConSan inserted queue/dispatch preloads,
  // those values do not occupy their guest registers until the repair above
  // has completed.  Saving earlier would capture the replacement ABI instead
  // (for example, a queue pointer in the guest's kernarg slot) and restore it
  // over the repaired value immediately before entering the guest kernel.
  if (entry_scalar_backup && !append_entry_scalar_backup(words, *entry_scalar_backup,
                                                         /*restore=*/false, arch, errors)) {
    return std::nullopt;
  }

  // The workgroup-shadow initializer uses the entry-local scratch VGPRs. Capture a
  // scalar owner sourced from v0 before that scratch is allowed to clobber v0.
  if (persistent_owner_sgpr) {
    switch (owner_source) {
    case OwnerSource::Automatic:
      errors.emplace_back("ConSan scalar owner prologue has an unresolved automatic source");
      return std::nullopt;
    case OwnerSource::WorkitemId: {
      if (!sequence.emit_all(instrumentation::build_v_readfirstlane_b32(*persistent_owner_sgpr,
                                                                        kAmdGpuWorkitemIdX, arch),
                             instrumentation::build_valu_to_salu_dependency_wait(arch))) {
        errors.emplace_back("ConSan scalar owner prologue could not read its wave ID");
        return std::nullopt;
      }
      if (!append_entry_salu_write(words,
                                   build_s_lshr_b32(*persistent_owner_sgpr, *persistent_owner_sgpr,
                                                    scalar_positive_inline_u32(owner_shift_bits),
                                                    arch),
                                   arch)) {
        errors.emplace_back("ConSan scalar owner prologue could not normalize its wave ID");
        return std::nullopt;
      }
      break;
    }
    case OwnerSource::HwId: {
      const TargetProfile *target = target_profile(arch);
      const detail::ResidentWaveOwnerRequest request{
          .destination_sgpr = *persistent_owner_sgpr,
      };
      if (target == nullptr || !detail::append_resident_wave_owner(words, request, *target)) {
        errors.emplace_back("ConSan scalar owner prologue could not encode resident-wave identity");
        return std::nullopt;
      }
      break;
    }
    }
  }

  if (!capture_workgroup_before_dispatch &&
      (exact_workgroup_sgprs.complete() || exact_workgroup_vgprs.complete())) {
    if (!workgroup_sources) {
      errors.emplace_back("ConSan exact workgroup-tuple prologue requires launch sources");
      return std::nullopt;
    }
    if (!append_exact_workgroup_capture(words, exact_workgroup_sgprs, exact_workgroup_vgprs,
                                        *workgroup_sources, arch, errors))
      return std::nullopt;
  }

  // The generic dispatch repair treats system SGPRs as one contiguous suffix.
  // That is not sufficient for a sparse guest workgroup payload: after we
  // enable X/Y/Z for ConSan, a Y-only guest still expects Y in its first
  // system-SGPR slot rather than the newly inserted X value.  Always perform
  // the semantic full-to-guest mapping after the generic repair.  It emits no
  // move when the layouts already coincide.
  if (workgroup_sources &&
      !append_cdna_full_workgroup_payload_restore(words, *workgroup_sources, arch, errors)) {
    return std::nullopt;
  }
  if (!persistent_owner_sgpr)
    switch (owner_source) {
    case OwnerSource::Automatic:
      errors.emplace_back("ConSan owner/epoch prologue has an unresolved automatic source");
      return std::nullopt;
    case OwnerSource::WorkitemId: {
      if (!sequence.emit(instrumentation::build_v_lshrrev_b32(
              owner_vgpr, scalar_positive_inline_u32(owner_shift_bits), kAmdGpuWorkitemIdX,
              arch))) {
        errors.emplace_back("ConSan owner/epoch prologue could not encode owner VGPR init");
        return std::nullopt;
      }

      break;
    }
    case OwnerSource::HwId: {
      if (!owner_sgpr) {
        errors.emplace_back("ConSan hw_id owner source requires RJ_CONSAN_OWNER_SGPR");
        return std::nullopt;
      }
      const TargetProfile *target = target_profile(arch);
      const detail::ResidentWaveOwnerRequest request{
          .destination_sgpr = *owner_sgpr,
      };
      if (target == nullptr || !detail::append_resident_wave_owner(words, request, *target)) {
        errors.emplace_back("ConSan owner/epoch prologue could not encode resident-wave identity");
        return std::nullopt;
      }
      (void)sequence.emit(build_v_mov_b32_e32(owner_vgpr, *owner_sgpr, arch));
      break;
    }
    }
  if (persistent_epoch_sgpr) {
    if (!append_entry_salu_write(
            words, build_s_mov_b32(*persistent_epoch_sgpr, scalar_positive_inline_u32(0), arch),
            arch))
      return std::nullopt;
  } else {
    (void)sequence.emit(build_v_mov_b32_e32(epoch_vgpr, scalar_positive_inline_u32(0), arch));
  }
  if (entry_scalar_backup && !append_entry_scalar_backup(words, *entry_scalar_backup,
                                                         /*restore=*/true, arch, errors)) {
    return std::nullopt;
  }
  return words;
}

[[nodiscard]] std::optional<std::vector<uint32_t>>
build_private_epoch_prologue_words(const PrivateEpochPrologueEmissionPlan &plan,
                                   rj_code_arch_t arch, std::vector<std::string> &errors) {
  if (!plan.is_well_formed()) {
    errors.emplace_back("ConSan private-epoch prologue has an invalid emission plan");
    return std::nullopt;
  }
  const uint16_t scratch_vgpr = plan.scratch_vgpr;
  const PrivateStateLayout &private_state_layout = plan.private_state_layout;
  const uint32_t epoch_offset = private_state_layout.epoch_offset;
  const std::optional<uint32_t> owner_offset = private_state_layout.owner_offset;
  const std::optional<uint32_t> dispatch_id_offset = private_state_layout.dispatch_id_offset;
  const PersistentWorkgroupPrivateOffsets &exact_workgroup_offsets =
      private_state_layout.exact_workgroup_offsets;
  const VgprSpillSequence &spill = plan.spill;
  const std::optional<SgprSpillSequence> &entry_scalar_spill = plan.entry_scalar_spill;
  const std::optional<WorkgroupSources> &workgroup_sources = plan.workgroup_sources;
  const std::optional<DispatchIdPreloadPlan> &dispatch_plan = plan.dispatch_plan;
  const DispatchIdCapture dispatch_capture = plan.dispatch_capture;
  const auto epoch_store =
      instrumentation::build_private_store_b32(scratch_vgpr, epoch_offset, arch);
  std::optional<std::vector<uint32_t>> owner_store;
  if (owner_offset)
    owner_store = instrumentation::build_private_store_b32(scratch_vgpr, *owner_offset, arch);
  std::optional<std::vector<uint32_t>> dispatch_id_low_store;
  std::optional<std::vector<uint32_t>> dispatch_id_high_store;
  if (dispatch_id_offset) {
    dispatch_id_low_store =
        instrumentation::build_private_store_b32(scratch_vgpr, *dispatch_id_offset, arch);
    dispatch_id_high_store = instrumentation::build_private_store_b32(
        static_cast<uint16_t>(scratch_vgpr + 1u), *dispatch_id_offset + SpillManager::kSlotBytes,
        arch);
  }
  std::array<std::optional<std::vector<uint32_t>>, 4> exact_workgroup_stores;
  const std::array<std::optional<uint32_t>, 4> exact_workgroup_offset_values =
      exact_workgroup_offsets.values();
  for (size_t index = 0; index < exact_workgroup_offset_values.size(); ++index) {
    if (exact_workgroup_offset_values[index]) {
      exact_workgroup_stores[index] = instrumentation::build_private_store_b32(
          scratch_vgpr, *exact_workgroup_offset_values[index], arch);
    }
  }
  const auto wait_store = instrumentation::build_s_wait_private_store0(arch);
  if (!epoch_store || (owner_offset && !owner_store) ||
      (dispatch_id_offset && (!dispatch_id_low_store || !dispatch_id_high_store)) ||
      std::ranges::any_of(std::views::iota(size_t{0}, exact_workgroup_stores.size()),
                          [&](size_t index) {
                            return exact_workgroup_offset_values[index] &&
                                   !exact_workgroup_stores[index];
                          }) ||
      !wait_store) {
    errors.emplace_back(
        "ConSan private-epoch prologue could not encode persistent-state initialization");
    return std::nullopt;
  }

  std::vector<uint32_t> words;
  words.reserve(spill.save_words.size() + spill.restore_words.size() +
                (entry_scalar_spill ? entry_scalar_spill->save_words.size() +
                                          entry_scalar_spill->restore_words.size()
                                    : 0u) +
                48u);
  InstructionSequence sequence(words);
  if (dispatch_plan && !dispatch_id_offset &&
      !append_dispatch_id_capture_and_restore(
          words, *dispatch_plan, dispatch_capture,
          !workgroup_sources || !workgroup_sources->cdna_full_payload_base.has_value(), arch,
          errors)) {
    return std::nullopt;
  }

  (void)sequence.emit(spill.save_words);
  if (owner_store) {
    // The entry scalar-spill sequence below uses scratch_vgpr to transfer
    // guest SGPRs into private memory. Capture the raw entry workitem ID first;
    // after that sequence scratch_vgpr no longer contains the ABI v0 value.
    (void)sequence.emit_all(
        build_v_mov_b32_e32(scratch_vgpr, vector_source_vgpr(kAmdGpuWorkitemIdX), arch),
        *owner_store);
  }
  if (dispatch_id_offset) {
    if (!dispatch_plan || !dispatch_capture.vgpr() || dispatch_capture.sgpr() ||
        *dispatch_capture.vgpr() != scratch_vgpr ||
        !append_dispatch_id_capture_and_restore(
            words, *dispatch_plan, dispatch_capture,
            !workgroup_sources || !workgroup_sources->cdna_full_payload_base.has_value(), arch,
            errors)) {
      errors.emplace_back(
          "ConSan private-epoch prologue could not capture private dispatch identity");
      return std::nullopt;
    }
    (void)sequence.emit_all(*dispatch_id_low_store, *dispatch_id_high_store, *wait_store);
  }
  // Descriptor surgery may insert queue/dispatch preloads ahead of the guest
  // ABI.  Capture launch identity and repair the guest-visible SGPR layout
  // before preserving any borrowed entry SGPRs; otherwise the eventual spill
  // restore writes replacement-ABI values (notably the queue pointer) over the
  // repaired kernarg pointer.
  if (entry_scalar_spill) {
    (void)sequence.emit(entry_scalar_spill->save_words);
    if (workgroup_sources && !append_cdna_semantic_entry_scalar_spill_overrides(
                                 words, *entry_scalar_spill, *workgroup_sources, arch, errors)) {
      return std::nullopt;
    }
  }

  if (exact_workgroup_offsets.complete()) {
    if (!workgroup_sources) {
      errors.emplace_back("ConSan private exact workgroup tuple requires entry ABI sources");
      return std::nullopt;
    }
    const std::array<WorkgroupSource, 4> sources = {workgroup_sources->x, workgroup_sources->y,
                                                    workgroup_sources->z,
                                                    workgroup_sources->cluster_workgroup_id};
    for (size_t index = 0; index < sources.size(); ++index) {
      if (!exact_workgroup_offset_values[index])
        continue;
      if (!sources[index].is_well_formed()) {
        errors.emplace_back("ConSan private prologue requires every exact workgroup coordinate");
        return std::nullopt;
      }
      if (!sources[index].has_value()) {
        if (index != 3u) {
          errors.emplace_back("ConSan private prologue requires every exact workgroup coordinate");
          return std::nullopt;
        }
        (void)sequence.emit(build_v_mov_b32_e32(scratch_vgpr, scalar_positive_inline_u32(0), arch));
      } else if (!detail::append_workgroup_source_value(words, sources[index], scratch_vgpr,
                                                        arch)) {
        errors.emplace_back(
            "ConSan private prologue could not capture an exact workgroup coordinate");
        return std::nullopt;
      }
      (void)sequence.emit(*exact_workgroup_stores[index]);
    }
  }
  if (workgroup_sources &&
      !append_cdna_full_workgroup_payload_restore(words, *workgroup_sources, arch, errors)) {
    return std::nullopt;
  }
  (void)sequence.emit_all(build_v_mov_b32_e32(scratch_vgpr, scalar_positive_inline_u32(0), arch),
                          *epoch_store, *wait_store);
  if (entry_scalar_spill) {
    (void)sequence.emit(entry_scalar_spill->restore_words);
  }
  (void)sequence.emit(spill.restore_words);

  return words;
}

[[nodiscard]] bool kernel_contains_patch_anchor(const ProgramContainer &kernel,
                                                const CommittedPatchGeometry &patch) {
  return kernel.has_text_range && patch.anchor_offset >= kernel.entry_text_offset &&
         patch.anchor_offset - kernel.entry_text_offset < kernel.code_size;
}

[[nodiscard]] bool kernel_owns_patch(const ProgramContainer &kernel,
                                     const PatchLoweringProduct &patch) {
  if (!patch.owner_descriptor_file_offsets.empty()) {
    return std::ranges::find(patch.owner_descriptor_file_offsets, kernel.descriptor_file_offset) !=
           patch.owner_descriptor_file_offsets.end();
  }
  return kernel_contains_patch_anchor(kernel, patch);
}

template <typename Visitor>
void for_each_patch(const TransformArtifacts &result, Visitor &&visitor) {
  for (const PatchInfo &patch : result.patches)
    visitor(patch);
  for (const TextFragment &fragment : result.staged_text_fragments)
    visitor(fragment.patch);
}

template <typename Predicate>
[[nodiscard]] const PatchInfo *find_patch(const TransformArtifacts &result, Predicate &&predicate) {
  const PatchInfo *found = nullptr;
  for_each_patch(result, [&](const PatchInfo &patch) {
    if (found == nullptr && predicate(patch))
      found = &patch;
  });
  return found;
}

[[nodiscard]] bool enable_full_workgroup_id_payload(rj_code_arch_t arch,
                                                    TransformArtifacts &result) {
  if (result.replacement.empty() || !is_capability_arch(arch)) {
    return true;
  }

  std::unordered_map<std::string_view, uint64_t> owners_by_name;
  for_each_patch(result, [&](const PatchInfo &patch) {
    if (!detail::patch_requires_full_workgroup_id_payload(result.observation_plan().mode, arch,
                                                          patch))
      return;
    for (uint64_t descriptor_offset : patch.owner_descriptor_file_offsets) {
      const ProgramContainer *owner =
          result.program_inventory.find_kernel_by_descriptor(descriptor_offset);
      if (owner == nullptr) {
        result.errors.emplace_back(
            "ConSan could not resolve an owner before enabling workgroup IDs");
        return;
      }
      const auto [it, inserted] =
          owners_by_name.emplace(owner->name, owner->descriptor_file_offset);
      if (!inserted && it->second != owner->descriptor_file_offset) {
        result.errors.emplace_back(
            "ConSan found duplicate owner names while enabling workgroup IDs");
        return;
      }
    }
  });
  if (!result.errors.empty())
    return false;
  if (owners_by_name.empty())
    return true;

  AmdGpuCodeObject code_object(result.replacement.data(), result.replacement.size());
  if (!code_object.is_valid()) {
    result.errors.emplace_back(
        "ConSan could not parse its replacement before enabling workgroup IDs");
    return false;
  }
  CodeObjectPatcher patcher(code_object);
  std::unordered_set<std::string_view> patched_owner_names;
  DescriptorSgprRequirements scalar_requirements;
  for (const AmdGpuKernelInfo &kernel : code_object.kernels()) {
    const auto owner = owners_by_name.find(kernel.name);
    if (owner == owners_by_name.end())
      continue;
    if (!patched_owner_names.insert(kernel.name).second) {
      result.errors.emplace_back("ConSan found duplicate owner names while enabling workgroup IDs");
      return false;
    }
    const uint64_t descriptor_offset = kernel.descriptor_file_offset;
    const std::span<const uint8_t> current_image = patcher.image_bytes();
    auto descriptor_value = read_kernel_descriptor(current_image, descriptor_offset);
    if (!descriptor_value) {
      result.errors.emplace_back("ConSan workgroup-ID descriptor exceeds replacement bytes");
      return false;
    }
    KD &descriptor = *descriptor_value;
    if (arch_supports_kernarg_preload_overflow_recovery(arch)) {
      const uint32_t user_sgpr_count = descriptor_user_sgpr_count(descriptor);
      const uint32_t required_sgpr_count =
          user_sgpr_count + 3u +
          (AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2,
                           kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_INFO) != 0
               ? 1u
               : 0u);
      note_maximum_descriptor_extent(scalar_requirements, owner->second,
                                     static_cast<uint16_t>(required_sgpr_count));
    }
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X,
                    1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y,
                    1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z,
                    1u);
    if (!patcher.patch_kernel_descriptor(descriptor_offset, descriptor)) {
      result.errors.emplace_back("ConSan could not enable its full workgroup-ID payload");
      return false;
    }
  }
  if (patched_owner_names.size() != owners_by_name.size()) {
    result.errors.emplace_back(
        "ConSan could not resolve every replacement owner while enabling workgroup IDs");
    return false;
  }
  if (!apply_descriptor_requirements(patcher, code_object, result.program_inventory,
                                     DescriptorVgprRequirements{}, scalar_requirements,
                                     DescriptorPrivateRequirements{}, nullptr, nullptr, arch,
                                     "ConSan full workgroup-ID payload", result.errors))
    return false;
  result.replacement = std::move(patcher).emit();
  return true;
}

[[nodiscard]] bool apply_dispatch_id_descriptor(CodeObjectPatcher &patcher,
                                                uint64_t descriptor_file_offset,
                                                const DispatchIdPreloadPlan &plan,
                                                DispatchIdCapture capture, rj_code_arch_t arch,
                                                std::vector<std::string> &errors) {
  const std::span<const uint8_t> current_image = patcher.image_bytes();
  auto descriptor_value = read_kernel_descriptor(current_image, descriptor_file_offset);
  if (!descriptor_value) {
    errors.emplace_back("ConSan dispatch-ID descriptor exceeds ELF bytes");
    return false;
  }
  if (!plan.supported() || !capture.present() ||
      (capture.sgpr() && static_cast<uint32_t>(*capture.sgpr()) + 2u > kMaxSgprs) ||
      (capture.vgpr() && static_cast<uint32_t>(*capture.vgpr()) + 2u > kMaxVgprs)) {
    errors.emplace_back("ConSan dispatch-ID descriptor has an unsupported preload plan");
    return false;
  }

  KD &desc = *descriptor_value;
  const DispatchIdPreloadPlan current_plan = descriptor_dispatch_id_preload_plan(desc, arch);
  // The descriptor and its complete ABI rewrite are one transaction.  In
  // particular, queue insertion, the explicit guest source map, and kernarg
  // tail recovery must not drift between planning and descriptor mutation.
  if (current_plan != plan) {
    errors.emplace_back("ConSan dispatch-ID descriptor changed after preload planning");
    return false;
  }

  if (plan.descriptor_change_required()) {
    AMDHSA_BITS_SET(desc.kernel_code_properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR,
                    1u);
    AMDHSA_BITS_SET(desc.kernel_code_properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID,
                    1u);
    AMDHSA_BITS_SET(desc.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT,
                    plan.expanded_user_sgpr_count);
    if (plan.requires_kernarg_reload()) {
      AMDHSA_BITS_SET(desc.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_LENGTH,
                      plan.replacement_kernarg_preload_length);
    }
  }
  if (!patcher.patch_kernel_descriptor(descriptor_file_offset, desc)) {
    errors.emplace_back("ConSan could not patch dispatch-ID descriptor state");
    return false;
  }
  return true;
}

void rollback_dispatch_id_as_unsupported(TransformArtifacts &result, std::string reason) {
  result.outcome = TransformOutcome::Unsupported;
  result.discard_candidate_modification();
  result.warnings.emplace_back("ConSan dispatch-ID capture is unsupported: " + std::move(reason));
}

[[nodiscard]] std::string dispatch_id_preload_rejection_reason(std::string_view kernel_name,
                                                               const DispatchIdPreloadPlan &plan,
                                                               DispatchIdCapture capture) {
  return "kernel '" + std::string(kernel_name) +
         "' preload plan support=" + std::to_string(static_cast<uint32_t>(plan.support)) +
         " user_sgprs=" + std::to_string(plan.original_user_sgpr_count) +
         " required_sgprs=" + std::to_string(plan.required_sgpr_count) +
         (capture.sgpr() ? " persistent_sgpr=" + std::to_string(*capture.sgpr())
                         : " capture_vgpr=" + std::to_string(capture.vgpr().value_or(kMaxVgprs)));
}

void note_dispatch_id_patch_info(PatchAbiEffects &effects, const DispatchIdPreloadPlan &plan,
                                 DispatchIdCapture capture) {
  effects.dispatch_id_prologue = DispatchIdPrologueEffect{
      .preload = plan,
      .capture = capture,
  };
}

void try_apply_private_epoch_prologue_patch(const Request &request,
                                            const BoundRuntimeResources &resources,
                                            const OperatingPoint &operating_point,
                                            rj_code_arch_t arch, TransformArtifacts &result) {
  if (!result.modified() || result.replacement.empty()) {
    result.warnings.emplace_back(
        "ConSan private-epoch prologue skipped because no private-epoch access probe was "
        "emitted");
    return;
  }
  const TargetProfile *target = target_profile(arch);
  if (target == nullptr) {
    result.errors.emplace_back("ConSan private-epoch prologue has no admitted target profile");
    return;
  }

  std::span<const uint8_t> active_bytes(result.replacement.data(), result.replacement.size());
  AmdGpuCodeObject code_object(active_bytes.data(), active_bytes.size());
  CodeObjectPatcher patcher(code_object);
  if (patcher.text_bytes().empty()) {
    result.errors.emplace_back("ConSan private-epoch prologue found no .text section");
    return;
  }
  struct PlannedPrivateEpochPrologue {
    const ProgramContainer *kernel = nullptr;
    uint64_t active_descriptor_file_offset = 0;
    PrivateEpochPrologueEmissionPlan emission;
    PrivateStateLayout private_state_layout;
    uint32_t required_private_bytes = 0;
  };
  std::vector<PlannedPrivateEpochPrologue> planned;
  DescriptorPrivateRequirements private_requirements;
  for (const ProgramContainer &kernel : result.program_inventory.kernels()) {
    const PatchInfo *access_patch = find_patch(result, [&](const PatchInfo &patch) {
      return patch.private_state_layout && patch.scratch_vgpr && kernel_owns_patch(kernel, patch);
    });
    if (access_patch == nullptr)
      continue;

    const auto active_kernel =
        std::ranges::find_if(code_object.kernels(), [&](const AmdGpuKernelInfo &candidate) {
          return candidate.name == kernel.name;
        });
    if (active_kernel == code_object.kernels().end()) {
      result.warnings.emplace_back(
          "ConSan private-epoch prologue could not resolve an active kernel descriptor");
      continue;
    }

    const auto descriptor_value =
        read_kernel_descriptor(active_bytes, active_kernel->descriptor_file_offset);
    if (!descriptor_value) {
      result.errors.emplace_back("ConSan private-epoch descriptor exceeds ELF bytes");
      return;
    }
    const KD &descriptor = *descriptor_value;

    const PrivateStateLayout &layout = *access_patch->private_state_layout;
    if (!layout.is_well_formed()) {
      result.errors.emplace_back("ConSan private-epoch prologue found an invalid layout");
      return;
    }
    OperatingPoint kernel_point = operating_point;
    const std::array<uint64_t, 1> kernel_owner = {kernel.descriptor_file_offset};
    if (!result.operating_point.owner_transient_sgprs.empty() &&
        !apply_transient_sgpr_assignment(request, kernel_point, result.operating_point,
                                         kernel_owner)) {
      result.errors.emplace_back(
          "ConSan private-epoch prologue has no transient scalar assignment for kernel '" +
          kernel.name + "'");
      return;
    }
    const bool has_private_dispatch_id = layout.dispatch_id_offset.has_value();
    const DispatchIdCapture dispatch_capture =
        has_private_dispatch_id ? DispatchIdCapture::in_vgprs(*access_patch->scratch_vgpr)
                                : dispatch_id_capture(kernel_point);
    std::optional<DispatchIdPreloadPlan> dispatch_plan;
    if (dispatch_capture.present()) {
      dispatch_plan = descriptor_dispatch_id_preload_plan(descriptor, arch);
      if (!dispatch_plan->supported() ||
          (dispatch_capture.sgpr() &&
           *dispatch_capture.sgpr() < dispatch_plan->required_sgpr_count)) {
        rollback_dispatch_id_as_unsupported(
            result,
            dispatch_id_preload_rejection_reason(kernel.name, *dispatch_plan, dispatch_capture));
        return;
      }
    }
    const bool has_private_exact_workgroup = layout.exact_workgroup_offsets.complete();

    const auto private_limit = address_free_private_limit(arch);
    if (!private_limit) {
      result.warnings.emplace_back(
          "ConSan private-epoch prologue has no address-free scratch support");
      continue;
    }
    SpillManager manager(layout.ephemeral_base, *private_limit);

    std::optional<WorkgroupSources> workgroup_sources;
    if (has_private_exact_workgroup) {
      std::vector<std::string> source_errors;
      std::optional<uint16_t> full_payload_user_sgpr_count;
      if (has_private_exact_workgroup && arch_supports_kernarg_preload_overflow_recovery(arch)) {
        const auto exact_tuple_descriptor =
            read_kernel_descriptor(active_bytes, active_kernel->descriptor_file_offset);
        if (!exact_tuple_descriptor) {
          result.errors.emplace_back("ConSan private exact-tuple descriptor exceeds ELF bytes");
          return;
        }
        full_payload_user_sgpr_count =
            dispatch_plan
                ? dispatch_plan->expanded_user_sgpr_count
                : static_cast<uint16_t>(descriptor_user_sgpr_count(*exact_tuple_descriptor));
      }
      workgroup_sources = descriptor_workgroup_sources(
          active_bytes, active_kernel->descriptor_file_offset, arch, source_errors,
          kernel.uses_cluster_workgroup_id, full_payload_user_sgpr_count);
      if (!workgroup_sources) {
        result.errors.insert(result.errors.end(), source_errors.begin(), source_errors.end());
        return;
      }
    }
    const uint16_t prologue_temporary_vgpr_count = has_private_dispatch_id ? 2u : 1u;
    const bool fixed_lane_entry_scalar_reservoir = kernel_point.exec_save_sgpr &&
                                                   kernel_point.has_scalar_spill() &&
                                                   !kernel.uses_dynamic_stack.value_or(false);
    std::optional<uint16_t> entry_scalar_reservoir_vgpr;
    uint16_t prologue_spill_vgpr_count = prologue_temporary_vgpr_count;
    if (fixed_lane_entry_scalar_reservoir) {
      const auto access_resource_plan =
          std::ranges::find_if(result.resource_plans, [&](const CandidateResourcePlan &plan) {
            return plan.site_kind == ResourceSiteKind::Access && plan.scratch_vgpr &&
                   *plan.scratch_vgpr == *access_patch->scratch_vgpr &&
                   plan.text_offset == access_patch->anchor_offset &&
                   std::ranges::find(plan.owner_kernel_ids, kernel.id) !=
                       plan.owner_kernel_ids.end();
          });
      if (access_resource_plan == result.resource_plans.end() ||
          access_resource_plan->required_vgpr_count <
              static_cast<uint32_t>(*access_patch->scratch_vgpr) +
                  access_resource_plan->scratch_vgpr_count + 1u) {
        result.warnings.emplace_back(
            "ConSan private-epoch prologue could not resolve its planned scalar reservoir");
        continue;
      }
      prologue_spill_vgpr_count =
          static_cast<uint16_t>(access_resource_plan->scratch_vgpr_count + 1u);
      entry_scalar_reservoir_vgpr = static_cast<uint16_t>(*access_patch->scratch_vgpr +
                                                          access_resource_plan->scratch_vgpr_count);
    }
    auto spill = build_vgpr_spill_sequence(manager, *access_patch->scratch_vgpr,
                                           prologue_spill_vgpr_count, arch);
    if (!spill) {
      result.warnings.emplace_back(
          "ConSan private-epoch prologue could not preserve its temporary VGPR");
      continue;
    }
    std::optional<SgprSpillSequence> entry_scalar_spill;
    if (kernel_point.exec_save_sgpr) {
      const uint16_t guest_entry_sgpr_count = static_cast<uint16_t>(
          descriptor_user_sgpr_count(descriptor) + descriptor_system_sgpr_count(descriptor));
      const uint16_t scalar_base = *kernel_point.exec_save_sgpr;
      const uint16_t scalar_end = static_cast<uint16_t>(std::min<uint32_t>(
          static_cast<uint32_t>(scalar_base) +
              exec_save_sgpr_count(resolve_exec_save_requirement(request, resources, kernel_point),
                                   arch),
          guest_entry_sgpr_count));
      if (scalar_base < scalar_end) {
        entry_scalar_spill =
            entry_scalar_reservoir_vgpr
                ? build_lane_sgpr_spill_sequence(
                      scalar_base, static_cast<uint16_t>(scalar_end - scalar_base),
                      *entry_scalar_reservoir_vgpr, spill->total_private_bytes, arch)
                : build_sgpr_spill_sequence(manager, scalar_base,
                                            static_cast<uint16_t>(scalar_end - scalar_base),
                                            *access_patch->scratch_vgpr, arch);
        if (!entry_scalar_spill) {
          result.warnings.emplace_back(
              "ConSan private-epoch prologue could not preserve its entry ABI SGPRs");
          continue;
        }
      }
    }
    uint32_t required_private_bytes =
        entry_scalar_spill ? entry_scalar_spill->total_private_bytes : spill->total_private_bytes;
    for_each_patch(result, [&](const PatchInfo &patch) {
      if (kernel_owns_patch(kernel, patch))
        required_private_bytes =
            std::max(required_private_bytes, patch.required_private_segment_size);
    });
    private_requirements[kernel.descriptor_file_offset] = required_private_bytes;
    PrivateEpochPrologueEmissionPlan emission{
        .scratch_vgpr = *access_patch->scratch_vgpr,
        .private_state_layout = layout,
        .spill = std::move(*spill),
        .entry_scalar_spill = std::move(entry_scalar_spill),
        .workgroup_sources = std::move(workgroup_sources),
        .dispatch_plan = dispatch_plan,
        .dispatch_capture = dispatch_capture,
    };
    planned.push_back({
        .kernel = &kernel,
        .active_descriptor_file_offset = active_kernel->descriptor_file_offset,
        .emission = std::move(emission),
        .private_state_layout = layout,
        .required_private_bytes = required_private_bytes,
    });
  }

  if (planned.empty()) {
    result.warnings.emplace_back("ConSan private-epoch prologue found no patchable kernels");
    return;
  }
  DescriptorSgprRequirements dispatch_sgpr_requirements;
  for (const PlannedPrivateEpochPrologue &item : planned) {
    if (!item.emission.dispatch_plan || !item.emission.dispatch_capture.present())
      continue;
    uint32_t required = item.emission.dispatch_plan->required_sgpr_count;
    if (const auto capture = item.emission.dispatch_capture.sgpr())
      required = std::max<uint32_t>(required, static_cast<uint32_t>(*capture) + 2u);
    note_maximum_descriptor_extent(dispatch_sgpr_requirements, item.kernel->descriptor_file_offset,
                                   static_cast<uint16_t>(required));
  }
  if (!apply_descriptor_requirements(patcher, code_object, result.program_inventory,
                                     DescriptorVgprRequirements{}, dispatch_sgpr_requirements,
                                     private_requirements, nullptr, nullptr, arch,
                                     "ConSan private-epoch prologue", result.errors))
    return;
  for (const PlannedPrivateEpochPrologue &item : planned) {
    if (item.emission.dispatch_capture.present() &&
        (!item.emission.dispatch_plan ||
         !apply_dispatch_id_descriptor(patcher, item.active_descriptor_file_offset,
                                       *item.emission.dispatch_plan, item.emission.dispatch_capture,
                                       arch, result.errors))) {
      return;
    }
  }

  std::vector<TextFragment> fragments;
  fragments.reserve(planned.size());
  for (const PlannedPrivateEpochPrologue &item : planned) {
    auto words = build_private_epoch_prologue_words(item.emission, arch, result.errors);
    if (!words)
      return;
    PatchInfo info;
    info.kind = PatchKind::KernelEntryPrivateEpochPrologue;
    info.anchor_offset = item.kernel->entry_text_offset;
    info.original_size = 0u;
    info.scratch_vgpr = item.emission.scratch_vgpr;
    info.private_state_layout = item.private_state_layout;
    info.spilled_vgpr_count = item.emission.spill.vgpr_count;
    info.required_private_segment_size = item.required_private_bytes;
    if (item.emission.entry_scalar_spill && item.emission.entry_scalar_spill->lane_reservoir_vgpr) {
      info.entry_scalar_backup = EntryScalarBackup{
          .vgpr = *item.emission.entry_scalar_spill->lane_reservoir_vgpr,
          .sgpr_base = item.emission.entry_scalar_spill->sgpr_base,
          .sgpr_count = item.emission.entry_scalar_spill->sgpr_count,
      };
    }
    note_dynamic_stack_private_requirement(info, &item.emission.spill);
    if (item.emission.dispatch_plan && item.emission.dispatch_capture.present())
      note_dispatch_id_patch_info(info, *item.emission.dispatch_plan,
                                  item.emission.dispatch_capture);
    info.owner_descriptor_file_offsets.push_back(item.kernel->descriptor_file_offset);
    fragments.push_back(TextFragment::entry_prefix(std::move(*words), std::move(info)));
  }
  result.replacement = std::move(patcher).emit();
  (void)stage_text_fragments(std::move(fragments), result);
}

[[nodiscard]] uint32_t
owner_epoch_prologue_required_vgpr_count(const OwnerEpochPrologueEmissionPlan &emission) {
  uint32_t required = emission.vgpr_state.required_vgpr_count();
  if (emission.dispatch_capture.vgpr())
    required = std::max<uint32_t>(required, *emission.dispatch_capture.vgpr() + 2u);
  return required;
}

[[nodiscard]] bool entry_scalar_backup_preserves_persistent_outputs(
    const OwnerEpochPrologueEmissionPlan &emission, bool exec_save_sgprs_persistent,
    bool automatic_owner_sgpr, uint16_t sgpr_base, uint16_t sgpr_count) {
  const auto overlaps = [&](std::optional<uint16_t> reg, uint16_t width = 1u) {
    return reg && range_overlaps(sgpr_base, sgpr_count, *reg, width);
  };
  const bool backs_up_only_automatic_owner = automatic_owner_sgpr && emission.owner_sgpr &&
                                             sgpr_base == *emission.owner_sgpr && sgpr_count == 1u;
  bool conflict = (exec_save_sgprs_persistent && !backs_up_only_automatic_owner) ||
                  overlaps(emission.dispatch_capture.sgpr(), kDispatchStateSgprCount);
  emission.persistent_sgprs.for_each_range(
      [&](std::optional<uint16_t> reg, uint16_t width) { conflict |= overlaps(reg, width); });
  return !conflict;
}

[[nodiscard]] bool owner_epoch_prologue_uses_vgpr(const OwnerEpochPrologueEmissionPlan &emission,
                                                  uint16_t vgpr) {
  const auto overlaps = [vgpr](std::optional<uint16_t> base, uint16_t width = 1u) {
    return base && range_overlaps(vgpr, 1u, *base, width);
  };
  if (overlaps(emission.vgpr_state.owner_epoch.owner))
    return true;
  const uint16_t epoch_width = 1u;
  if (overlaps(emission.vgpr_state.owner_epoch.epoch, epoch_width) ||
      overlaps(emission.dispatch_capture.vgpr(), 2u)) {
    return true;
  }
  for (const std::optional<uint16_t> reg : emission.vgpr_state.exact_workgroup.values()) {
    if (overlaps(reg))
      return true;
  }
  return false;
}

void try_apply_owner_epoch_prologue_patch(
    std::span<const uint8_t> bytes, const Request &request, const BoundRuntimeResources &resources,
    const OperatingPoint &operating_point,
    std::span<const PrologueScratchVgprAssignment> prologue_scratch_assignments,
    rj_code_arch_t arch, TransformArtifacts &result) {
  if (!operating_point.initialize_owner_epoch)
    return;
  if (!is_capability_arch(arch)) {
    result.warnings.emplace_back("ConSan owner/epoch prologue does not support this architecture");
    return;
  }
  const TargetProfile *target = target_profile(arch);
  if (target == nullptr) {
    result.errors.emplace_back("ConSan owner/epoch prologue has no admitted target profile");
    return;
  }
  if (operating_point.automatic_private_epoch) {
    try_apply_private_epoch_prologue_patch(request, resources, operating_point, arch, result);
    if (!result.errors.empty() || result.operating_point.owner_persistent_vgprs.empty())
      return;
  }
  if (!operating_point.owner_epoch_vgprs.complete() &&
      result.operating_point.owner_persistent_vgprs.empty() &&
      !operating_point.persistent_sgprs.complete()) {
    result.errors.emplace_back("ConSan owner/epoch prologue requires RJ_CONSAN_OWNER_VGPR and "
                               "RJ_CONSAN_EPOCH_VGPR");
    return;
  }
  if (operating_point.owner_epoch_vgprs.complete() &&
      operating_point.owner_epoch_vgprs->owner == operating_point.owner_epoch_vgprs->epoch) {
    result.errors.emplace_back("ConSan owner and epoch VGPRs must be distinct");
    return;
  }
  if (request.owner_source == OwnerSource::HwId && !operating_point.owner_sgpr.base()) {
    result.errors.emplace_back("ConSan hw_id owner source requires RJ_CONSAN_OWNER_SGPR");
    return;
  }
  if (result.program_inventory.kernels().empty()) {
    result.warnings.emplace_back("ConSan owner/epoch prologue found no kernel descriptors");
    return;
  }

  const std::span<const uint8_t> active_bytes = result.active_bytes(bytes);
  AmdGpuCodeObject code_object(active_bytes.data(), active_bytes.size());
  CodeObjectPatcher patcher(code_object);
  if (patcher.text_bytes().empty()) {
    result.errors.emplace_back("ConSan owner/epoch prologue found no .text section");
    return;
  }
  const uint32_t owner_required_sgpr_count =
      request.owner_source == OwnerSource::HwId
          ? static_cast<uint32_t>(*operating_point.owner_sgpr.base()) + 1u
          : 0u;
  /// Per-kernel transaction fixed before any descriptor or text mutation.
  ///
  /// Descriptor-derived entry facts, resolved persistent registers, and
  /// preservation/layout choices travel together into emission. The mutable
  /// patching loop may choose among entry routes, but cannot reinterpret the
  /// kernel ABI or change the initialization contract.
  struct PlannedOwnerEpochPrologue {
    ProgramContainer kernel{ProgramContainerKind::Kernel};
    OwnerEpochPrologueEmissionPlan emission;
    uint32_t required_vgpr_count = 0;
    uint16_t required_sgpr_count = 0;
  };
  std::vector<PlannedOwnerEpochPrologue> target_kernels;
  target_kernels.reserve(result.program_inventory.kernels().size());
  for (const ProgramContainer &kernel : result.program_inventory.kernels()) {
    if (!kernel.has_text_range)
      continue;
    const bool owns_emitted_patch = find_patch(result, [&](const PatchInfo &patch) {
                                      return kernel_owns_patch(kernel, patch);
                                    }) != nullptr;
    const bool owns_planned_site =
        std::ranges::any_of(result.resource_plans, [&](const CandidateResourcePlan &plan) {
          return std::ranges::find(plan.owner_kernel_ids, kernel.id) !=
                     plan.owner_kernel_ids.end() &&
                 plan.source != RegisterAllocationSource::Unsupported;
        });
    // Automatic persistent state exists only to serve emitted instrumentation.
    // Resource planning deliberately inventories more sites than selection may
    // admit (for example under max_patches or a kernel filter). Mutating every
    // descriptor/entry that merely owns one of those speculative plans can
    // perturb otherwise untouched kernels and, on real multi-kernel modules,
    // can make execution hang before the selected probe is reached.
    if ((operating_point.automatic_persistent_vgprs ||
         operating_point.persistent_sgprs.complete()) &&
        !owns_emitted_patch)
      continue;
    if (has_runtime_hardware_dispatch_id(operating_point) && !owns_emitted_patch &&
        !owns_planned_site) {
      continue;
    }
    const auto active_kernel =
        std::ranges::find_if(code_object.kernels(), [&](const AmdGpuKernelInfo &candidate) {
          return candidate.name == kernel.name;
        });
    if (active_kernel == code_object.kernels().end()) {
      result.warnings.emplace_back(
          "ConSan owner/epoch prologue could not resolve an active kernel descriptor");
      continue;
    }
    ProgramContainer active = kernel;
    active.descriptor_file_offset = active_kernel->descriptor_file_offset;
    active.entry_text_offset = active_kernel->entry_text_offset;
    active.text_file_offset = active_kernel->text_file_offset;
    OperatingPoint kernel_point = operating_point;
    const std::array<uint64_t, 1> kernel_owner = {kernel.descriptor_file_offset};
    if (!apply_persistent_vgpr_assignment(kernel_point, result.operating_point, kernel_owner)) {
      result.errors.emplace_back(
          "ConSan owner/epoch prologue has no persistent assignment for kernel '" + kernel.name +
          "'");
      return;
    }
    // Its access/synchronization probes and entry initialization were already
    // emitted by the private-state path above. The remaining pass is solely
    // for owner components with a persistent VGPR tuple.
    if (kernel_point.automatic_private_epoch)
      continue;
    if (!result.operating_point.owner_transient_sgprs.empty() &&
        !apply_transient_sgpr_assignment(request, kernel_point, result.operating_point,
                                         kernel_owner)) {
      result.errors.emplace_back(
          "ConSan owner/epoch prologue has no transient scalar assignment for kernel '" +
          kernel.name + "'");
      return;
    }
    const DispatchIdCapture dispatch_capture = dispatch_id_capture(kernel_point);
    OwnerEpochVgprSources owner_epoch_vgprs =
        owner_epoch_vgpr_sources(kernel_point.owner_epoch_vgprs);
    if (operating_point.persistent_sgprs.complete()) {
      const auto scratch =
          std::ranges::find(prologue_scratch_assignments, kernel.descriptor_file_offset,
                            &PrologueScratchVgprAssignment::descriptor_file_offset);
      if (scratch == prologue_scratch_assignments.end()) {
        result.errors.emplace_back(
            "ConSan scalar owner/epoch prologue has no entry-local VGPR scratch for kernel '" +
            kernel.name + "'");
        return;
      }
      owner_epoch_vgprs = {
          .owner = scratch->scratch_vgpr,
          .epoch = static_cast<uint16_t>(scratch->scratch_vgpr + 1u),
      };
    }
    const auto descriptor_value =
        read_kernel_descriptor(active_bytes, active.descriptor_file_offset);
    if (!descriptor_value) {
      if (dispatch_capture.present()) {
        rollback_dispatch_id_as_unsupported(result, "owner descriptor exceeds ELF bytes");
        return;
      }
      result.errors.emplace_back("ConSan owner/epoch descriptor exceeds ELF bytes");
      return;
    }
    const KD &descriptor = *descriptor_value;
    const uint16_t owner_shift_bits = kernel_wavefront_size(arch, descriptor) == 32u ? 5u : 6u;
    std::optional<DispatchIdPreloadPlan> dispatch_plan;
    if (dispatch_capture.present()) {
      dispatch_plan = descriptor_dispatch_id_preload_plan(descriptor, arch);
      if (!dispatch_plan->supported() ||
          (dispatch_capture.sgpr() &&
           *dispatch_capture.sgpr() < dispatch_plan->required_sgpr_count)) {
        rollback_dispatch_id_as_unsupported(
            result,
            dispatch_id_preload_rejection_reason(kernel.name, *dispatch_plan, dispatch_capture));
        return;
      }
    }
    const auto workitem_id_dimensions = descriptor_workitem_id_dimensions(descriptor);
    if (!workitem_id_dimensions) {
      result.errors.emplace_back("ConSan owner/epoch prologue has invalid workitem-ID dimensions");
      return;
    }
    std::optional<WorkgroupSources> workgroup_sources;
    const bool has_exact_workgroup_tuple = !kernel_point.exact_workgroup_vgprs.empty() ||
                                           !kernel_point.persistent_sgprs.exact_workgroup.empty();
    if (has_exact_workgroup_tuple) {
      std::vector<std::string> source_errors;
      std::optional<uint16_t> full_payload_user_sgpr_count;
      if (has_exact_workgroup_tuple && arch_supports_kernarg_preload_overflow_recovery(arch)) {
        full_payload_user_sgpr_count =
            dispatch_plan ? dispatch_plan->expanded_user_sgpr_count
                          : static_cast<uint16_t>(descriptor_user_sgpr_count(descriptor));
      }
      workgroup_sources = descriptor_workgroup_sources(
          active_bytes, active.descriptor_file_offset, arch, source_errors,
          kernel.uses_cluster_workgroup_id, full_payload_user_sgpr_count);
      if (!workgroup_sources) {
        result.errors.insert(result.errors.end(), source_errors.begin(), source_errors.end());
        return;
      }
    }
    OwnerEpochPrologueEmissionPlan emission{
        .vgpr_state =
            {
                .owner_epoch = {*owner_epoch_vgprs.owner, *owner_epoch_vgprs.epoch},
                .exact_workgroup = kernel_point.exact_workgroup_vgprs,
                .owner_epoch_lifetime = owner_epoch_vgpr_lifetime(
                    kernel_point.persistent_sgprs.complete(),
                    !result.operating_point.owner_persistent_vgprs.empty()),
            },
        .owner_shift_bits = owner_shift_bits,
        .owner_source = request.owner_source,
        .owner_sgpr = kernel_point.owner_sgpr.base(),
        .persistent_sgprs = kernel_point.persistent_sgprs,
        .dispatch_plan = dispatch_plan,
        .dispatch_capture = dispatch_capture,
        .entry_scalar_backup = std::nullopt,
        .workgroup_sources = std::move(workgroup_sources),
    };
    std::optional<EntryScalarBackup> entry_scalar_backup;
    const bool needs_branch_only_scalar_backup = kernel_point.branch_only_spill.has_value();
    const bool needs_dynamic_stack_scalar_backup =
        kernel_point.branch_only_spill &&
        kernel_point.branch_only_spill->dynamic_stack_borrowed_sgpr.has_value();
    // Runtime workgroup selection is computed at each eligible probe; it has
    // no entry-cached value. A compact site-local spill therefore introduces
    // no prologue consumer and must not make an entry backup live.
    const bool needs_full_entry_scalar_backup =
        needs_branch_only_scalar_backup || needs_dynamic_stack_scalar_backup;
    const bool needs_automatic_owner_scalar_backup = kernel_point.owner_sgpr.automatic() &&
                                                     request.owner_source == OwnerSource::HwId &&
                                                     kernel_point.owner_sgpr.base().has_value();
    if (needs_full_entry_scalar_backup || needs_automatic_owner_scalar_backup) {
      const bool entry_backup_arch_supported = is_capability_arch(arch);
      const bool has_scalar_spill_contract = kernel_point.has_scalar_spill();
      if (!entry_backup_arch_supported || !kernel_point.exec_save_sgpr ||
          !has_scalar_spill_contract ||
          (needs_dynamic_stack_scalar_backup && !kernel_point.dynamic_stack_spill)) {
        if (needs_full_entry_scalar_backup) {
          result.errors.emplace_back("ConSan entry scalar backup has no valid spill contract");
          return;
        }
      }
      const uint16_t sgpr_base = needs_full_entry_scalar_backup ? *kernel_point.exec_save_sgpr
                                                                : *kernel_point.owner_sgpr.base();
      const uint16_t borrowed_sgpr_count =
          needs_full_entry_scalar_backup
              ? exec_save_sgpr_count(
                    resolve_exec_save_requirement(request, resources, kernel_point), arch)
              : 1u;
      const auto original_descriptor = read_kernel_descriptor(bytes, kernel.descriptor_file_offset);
      if (!original_descriptor) {
        result.errors.emplace_back(
            "ConSan entry scalar backup descriptor exceeds original ELF bytes");
        return;
      }
      if (borrowed_sgpr_count == 0u) {
        result.errors.emplace_back("ConSan entry scalar backup has an empty borrowed window");
        return;
      }
      const uint32_t guest_entry_sgpr_count = descriptor_user_sgpr_count(*original_descriptor) +
                                              descriptor_system_sgpr_count(*original_descriptor);
      const uint32_t backup_end = std::min<uint32_t>(
          static_cast<uint32_t>(sgpr_base) + borrowed_sgpr_count, guest_entry_sgpr_count);
      const uint16_t sgpr_count =
          sgpr_base < backup_end ? static_cast<uint16_t>(backup_end - sgpr_base) : 0u;

      // The hardware initializes only the descriptor-declared user and system
      // SGPR prefix. Values above that prefix are undefined at kernel entry;
      // restoring them later can overwrite guest values written after entry
      // if the prologue's VALU lane transfers are still retiring. Preserve
      // only the initialized intersection of the borrowed scalar window.
      if (sgpr_count != 0u && !entry_scalar_backup_preserves_persistent_outputs(
                                  emission, kernel_point.exec_save_sgprs_persistent,
                                  kernel_point.owner_sgpr.automatic(), sgpr_base, sgpr_count)) {
        result.errors.emplace_back("ConSan entry scalar backup overlaps persistent output state");
        return;
      }
      if (sgpr_count > kernel_wavefront_size(arch, *original_descriptor)) {
        result.outcome = TransformOutcome::Unsupported;
        result.warnings.emplace_back(
            "ConSan entry scalar window exceeds one wave-local VGPR carrier");
        return;
      }

      if (sgpr_count != 0u) {
        const uint32_t entry_abi_vgpr_count = *workitem_id_dimensions;
        const uint32_t original_allocation =
            descriptor_ordinary_vgpr_allocation_count(*original_descriptor, arch);
        const uint32_t unified_allocation =
            descriptor_vgpr_allocation_count(*original_descriptor, arch);
        const bool has_live_accvgpr_bank = original_allocation < unified_allocation;
        const uint32_t tail_begin = std::max(original_allocation, entry_abi_vgpr_count);
        const uint32_t tail_end = has_live_accvgpr_bank ? original_allocation : kMaxVgprs;
        std::optional<uint16_t> backup_vgpr;
        for (uint32_t candidate = tail_begin; candidate < tail_end; ++candidate) {
          if (!owner_epoch_prologue_uses_vgpr(emission, static_cast<uint16_t>(candidate))) {
            backup_vgpr = static_cast<uint16_t>(candidate);
            break;
          }
        }
        // The value of every non-ABI ordinary VGPR is dead at hardware entry.
        // If an AccVGPR boundary or a full ordinary file leaves no fresh tail,
        // reuse a lower entry-local carrier while excluding the exact VGPRs
        // consumed by this prologue. Guest uses are safe because the scalar
        // window is restored before the displaced guest entry executes.
        for (uint32_t candidate = entry_abi_vgpr_count;
             !backup_vgpr && candidate < std::min(original_allocation, kMaxVgprs); ++candidate) {
          if (!owner_epoch_prologue_uses_vgpr(emission, static_cast<uint16_t>(candidate))) {
            backup_vgpr = static_cast<uint16_t>(candidate);
          }
        }
        if (!backup_vgpr) {
          result.outcome = TransformOutcome::Unsupported;
          result.warnings.emplace_back(has_live_accvgpr_bank
                                           ? "ConSan entry scalar backup has no entry-local "
                                             "ordinary VGPR below the AccVGPR boundary"
                                           : "ConSan entry scalar backup has no entry-local "
                                             "VGPR carrier");
          return;
        }
        entry_scalar_backup = EntryScalarBackup{
            .vgpr = *backup_vgpr,
            .sgpr_base = sgpr_base,
            .sgpr_count = sgpr_count,
        };
      }
    }
    emission.entry_scalar_backup = entry_scalar_backup;
    uint32_t required_vgpr_count = owner_epoch_prologue_required_vgpr_count(emission);
    if (entry_scalar_backup) {
      required_vgpr_count = std::max<uint32_t>(required_vgpr_count, entry_scalar_backup->vgpr + 1u);
    }
    uint32_t required_sgpr_count = owner_required_sgpr_count;
    if (emission.dispatch_capture.sgpr()) {
      required_sgpr_count = std::max<uint32_t>(
          required_sgpr_count,
          static_cast<uint32_t>(*emission.dispatch_capture.sgpr()) + kDispatchStateSgprCount);
    }
    emission.persistent_sgprs.for_each_range([&](std::optional<uint16_t> base, uint16_t width) {
      if (base)
        required_sgpr_count = std::max<uint32_t>(required_sgpr_count, *base + width);
    });
    target_kernels.push_back(PlannedOwnerEpochPrologue{
        .kernel = std::move(active),
        .emission = std::move(emission),
        .required_vgpr_count = required_vgpr_count,
        .required_sgpr_count = static_cast<uint16_t>(required_sgpr_count),
    });
  }
  DescriptorVgprRequirements prologue_vgpr_requirements;
  DescriptorSgprRequirements prologue_sgpr_requirements;
  for (const PlannedOwnerEpochPrologue &item : target_kernels) {
    if (item.emission.dispatch_plan && item.emission.dispatch_capture.present() &&
        !apply_dispatch_id_descriptor(patcher, item.kernel.descriptor_file_offset,
                                      *item.emission.dispatch_plan, item.emission.dispatch_capture,
                                      arch, result.errors)) {
      return;
    }
    const ProgramContainer *canonical =
        result.program_inventory.find_kernel_by_name(item.kernel.name);
    if (canonical == nullptr) {
      result.errors.emplace_back("ConSan owner/epoch prologue lost its descriptor owner");
      return;
    }
    note_maximum_descriptor_extent(prologue_vgpr_requirements, canonical->descriptor_file_offset,
                                   static_cast<uint16_t>(item.required_vgpr_count));
    note_maximum_descriptor_extent(prologue_sgpr_requirements, canonical->descriptor_file_offset,
                                   item.required_sgpr_count);
  }
  if (!apply_descriptor_requirements(patcher, code_object, result.program_inventory,
                                     prologue_vgpr_requirements, prologue_sgpr_requirements,
                                     DescriptorPrivateRequirements{}, nullptr, nullptr, arch,
                                     "ConSan owner/epoch prologue", result.errors))
    return;

  std::vector<TextFragment> fragments;
  fragments.reserve(target_kernels.size());
  for (const PlannedOwnerEpochPrologue &item : target_kernels) {
    const ProgramContainer *canonical_kernel =
        result.program_inventory.find_kernel_by_name(item.kernel.name);
    if (canonical_kernel == nullptr) {
      result.errors.emplace_back("ConSan owner/epoch prologue lost its canonical kernel owner");
      return;
    }
    const OwnerEpochPrologueEmissionPlan &prologue_plan = item.emission;
    auto words = build_owner_epoch_prologue_words(prologue_plan, arch, result.errors);
    if (!words)
      return;

    PatchInfo info;
    info.kind = PatchKind::KernelEntryOwnerEpochPrologue;
    info.anchor_offset = canonical_kernel->entry_text_offset;
    info.original_size = 0u;
    info.vgpr_state = prologue_plan.vgpr_state;
    info.persistent_sgpr_state = prologue_plan.persistent_sgprs;
    info.required_sgpr_count = item.required_sgpr_count;
    info.entry_scalar_backup = prologue_plan.entry_scalar_backup;
    if (prologue_plan.dispatch_plan && prologue_plan.dispatch_capture.present())
      note_dispatch_id_patch_info(info, *prologue_plan.dispatch_plan,
                                  prologue_plan.dispatch_capture);
    info.owner_descriptor_file_offsets.push_back(canonical_kernel->descriptor_file_offset);
    fragments.push_back(TextFragment::entry_prefix(std::move(*words), std::move(info)));
  }

  if (fragments.empty()) {
    result.warnings.emplace_back("ConSan owner/epoch prologue found no patchable kernels");
    return;
  }
  result.replacement = std::move(patcher).emit();
  (void)stage_text_fragments(std::move(fragments), result);
}

} // namespace rocjitsu::consan::detail
