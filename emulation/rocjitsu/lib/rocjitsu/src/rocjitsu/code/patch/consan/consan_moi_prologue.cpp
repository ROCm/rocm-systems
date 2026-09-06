// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi_prologue.h"

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/consan/consan_capability_contract.h"
#include "rocjitsu/code/patch/consan/consan_descriptor.h"
#include "rocjitsu/code/patch/consan/consan_moi_access_target.h"
#include "rocjitsu/code/patch/consan/consan_moi_exact_shadow_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_mode_planning.h"
#include "rocjitsu/code/patch/consan/consan_moi_native_abi.h"
#include "rocjitsu/code/patch/consan/consan_moi_placement_contracts.h"
#include "rocjitsu/code/patch/consan/consan_moi_probe_contracts.h"
#include "rocjitsu/code/patch/consan/consan_moi_record_event_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_relocation.h"
#include "rocjitsu/code/patch/consan/consan_moi_report_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_runtime_workgroup_gate.h"
#include "rocjitsu/code/patch/consan/consan_moi_shared_lowering.h"
#include "rocjitsu/code/patch/consan/consan_resource.h"
#include "rocjitsu/code/patch/consan/consan_text_relocation.h"
#include "rocjitsu/code/patch/consan/modes/inline_shadow/consan_moi_inline_shadow.h"
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

namespace rocjitsu::consan_moi_impl {

using consan_detail::moi_workgroup_shadow_initialization_lanes;
using consan_detail::moi_workgroup_shadow_preferred_zero_vgpr_count;
using consan_detail::MoiOwnerEpochPrologueEmissionPlan;
using consan_detail::MoiPrivateEpochPrologueEmissionPlan;
using consan_detail::MoiWorkgroupKeyRegisterPlan;
using consan_detail::MoiWorkgroupShadowClearStoreForm;
using consan_detail::plan_moi_workgroup_shadow_clear;

[[nodiscard]] ConSanMoiDispatchIdCapture dispatch_id_capture(const ConSanMoiOperatingPoint &point) {
  if (auto sgpr = point.moi_dispatch_identity.sgpr())
    return ConSanMoiDispatchIdCapture::in_sgprs(*sgpr);
  if (auto vgpr = point.moi_dispatch_identity.vgpr())
    return ConSanMoiDispatchIdCapture::in_vgprs(*vgpr);
  return {};
}
using consan_detail::range_overlaps;
using consan_moi_detail::moi_has_runtime_hardware_dispatch_id;

[[nodiscard]] bool append_moi_entry_salu_write(std::vector<uint32_t> &words, uint32_t word,
                                               rj_code_arch_t arch) {
  return InstructionSequence(words).emit_all(word,
                                             instrumentation::build_salu_dependency_delay(arch));
}

[[nodiscard]] bool append_moi_entry_scalar_backup(std::vector<uint32_t> &words,
                                                  const ConSanMoiEntryScalarBackup &backup,
                                                  bool restore, rj_code_arch_t arch,
                                                  std::vector<std::string> &errors) {
  if (!consan_is_capability_arch(arch) ||
      !backup.is_well_formed(kMaxVgprs, moi_ordinary_sgpr_limit(arch))) {
    errors.emplace_back("ConSan MOI entry scalar backup has an invalid register layout");
    return false;
  }
  InstructionSequence sequence(words);
  MoiEmissionRequirement require_emission(sequence, errors);
  for (uint16_t index = 0; index < backup.sgpr_count; ++index) {
    const uint16_t sgpr = static_cast<uint16_t>(backup.sgpr_base + index);
    const auto encoded =
        restore ? instrumentation::build_v_readlane_b32(sgpr, backup.vgpr, index, arch)
                : instrumentation::build_v_writelane_b32(backup.vgpr, sgpr, index, arch);
    require_emission.append("ConSan MOI entry scalar backup could not encode a lane transfer",
                            encoded);
  }
  if (restore)
    require_emission.append(
        "ConSan MOI entry scalar backup could not encode its restore dependency wait",
        instrumentation::build_valu_to_salu_dependency_wait(arch));
  return sequence.finish();
}

[[nodiscard]] bool append_dispatch_id_capture_and_restore(
    std::vector<uint32_t> &words, const ConSanMoiDispatchIdPreloadPlan &plan,
    ConSanMoiDispatchIdCapture capture, bool restore_system_sgprs, rj_code_arch_t arch,
    std::vector<std::string> &errors) {
  InstructionSequence sequence(words);
  MoiEmissionRequirement require_emission(sequence, errors);
  if (!plan.supported()) {
    errors.emplace_back("ConSan MOI dispatch-ID prologue received an unsupported preload plan");
    return false;
  }
  if (!capture.present() ||
      (capture.sgpr() && (*capture.sgpr() < 20u || *capture.sgpr() % 2u != 0u ||
                          static_cast<uint32_t>(*capture.sgpr()) + 2u > kMaxSgprs)) ||
      (capture.vgpr() && static_cast<uint32_t>(*capture.vgpr()) + 2u > kMaxVgprs) ||
      static_cast<uint32_t>(plan.dispatch_id_sgpr) + 2u > kMaxSgprs) {
    errors.emplace_back("ConSan MOI dispatch-ID prologue has an invalid persistent or source pair");
    return false;
  }

  if (capture.sgpr()) {
    require_emission(
        append_moi_entry_salu_write(
            words, build_s_mov_b32(*capture.sgpr(), plan.dispatch_id_sgpr, arch), arch) &&
            append_moi_entry_salu_write(
                words,
                build_s_mov_b32(static_cast<uint16_t>(*capture.sgpr() + 1u),
                                static_cast<uint16_t>(plan.dispatch_id_sgpr + 1u), arch),
                arch) &&
            // Hardware dispatch ID zero is valid (and common for the first packet),
            // while ConSan reserves zero as the empty metadata sentinel. Store the
            // injective modulo-2^64 successor used by every downstream comparison.
            append_moi_entry_salu_write(words,
                                        build_s_add_u32(*capture.sgpr(), *capture.sgpr(),
                                                        scalar_positive_inline_u32(1), arch),
                                        arch) &&
            append_moi_entry_salu_write(
                words,
                build_s_addc_u32(static_cast<uint16_t>(*capture.sgpr() + 1u),
                                 static_cast<uint16_t>(*capture.sgpr() + 1u),
                                 scalar_positive_inline_u32(0), arch),
                arch),
        "ConSan MOI dispatch-ID prologue cannot encode its SALU dependency delay");
  } else {
    require_emission.append(
        "ConSan MOI dispatch-ID prologue cannot encode its persistent VGPR successor",
        build_v_mov_b32_e32(*capture.vgpr(), plan.dispatch_id_sgpr, arch),
        build_v_mov_b32_e32(static_cast<uint16_t>(*capture.vgpr() + 1u),
                            static_cast<uint16_t>(plan.dispatch_id_sgpr + 1u), arch),
        instrumentation::build_v_add_u64_literal(*capture.vgpr(), 1u, arch));
  }
  uint32_t shifted_guest_end =
      static_cast<uint32_t>(plan.first_shifted_guest_sgpr) + plan.shifted_guest_sgpr_count;
  if (!restore_system_sgprs) {
    // The ordinary preload plan describes the system SGPRs as the tail of one
    // shifted guest range.  A full workgroup-payload plan restores that tail
    // separately according to the guest's sparse X/Y/Z/info mask.  Stop this
    // positional repair at the user-SGPR boundary so it neither duplicates
    // nor destroys the authoritative full payload before the semantic copy.
    shifted_guest_end = std::min<uint32_t>(shifted_guest_end, plan.original_user_sgpr_count);
  }
  for (uint16_t destination = plan.first_shifted_guest_sgpr; destination < shifted_guest_end;
       ++destination) {
    const auto source = consan_moi_dispatch_id_restore_source(plan, destination);
    if (!source || *source >= kMaxSgprs) {
      errors.emplace_back("ConSan MOI dispatch-ID prologue has an invalid guest restore source");
      return false;
    }
    require_emission(
        append_moi_entry_salu_write(words, build_s_mov_b32(destination, *source, arch), arch),
        "ConSan MOI dispatch-ID prologue cannot encode a guest-restore dependency delay");
  }
  if (restore_system_sgprs) {
    for (uint16_t i = 0; i < plan.shifted_system_sgpr_count; ++i) {
      const uint16_t destination = static_cast<uint16_t>(plan.original_user_sgpr_count + i);
      require_emission(
          append_moi_entry_salu_write(
              words,
              build_s_mov_b32(destination,
                              static_cast<uint16_t>(destination + plan.system_sgpr_shift), arch),
              arch),
          "ConSan MOI dispatch-ID prologue cannot encode a system-SGPR restore");
    }
  }
  if (plan.requires_kernarg_reload()) {
    if (!consan_arch_supports_kernarg_preload_overflow_recovery(arch) ||
        plan.kernarg_reload_count > 2u) {
      errors.emplace_back("ConSan MOI dispatch-ID prologue has an unsupported kernarg reload");
      return false;
    }
    for (uint16_t i = 0; i < plan.kernarg_reload_count; ++i) {
      require_emission.append(
          "ConSan MOI dispatch-ID prologue cannot encode a kernarg reload",
          instrumentation::build_s_load_dword(
              static_cast<uint16_t>(plan.kernarg_reload_sgpr + i), plan.kernarg_reload_base_sgpr,
              static_cast<uint32_t>(plan.kernarg_reload_offset_dwords + i) * sizeof(uint32_t),
              arch));
    }
    require_emission.append("ConSan MOI dispatch-ID prologue cannot encode its kernarg wait",
                            instrumentation::build_s_wait_scalar_load0(arch));
  }
  return sequence.finish();
}

[[nodiscard]] std::optional<std::vector<std::pair<uint16_t, uint16_t>>>
cdna_guest_workgroup_payload_copies(const ConSanMoiWorkgroupSources &sources, rj_code_arch_t arch) {
  if (!sources.cdna_full_payload_base && !sources.cdna_guest_payload_base &&
      sources.cdna_guest_payload_mask == 0u) {
    return std::vector<std::pair<uint16_t, uint16_t>>{};
  }
  if (!sources.cdna_full_payload_base || !sources.cdna_guest_payload_base ||
      !consan_arch_supports_kernarg_preload_overflow_recovery(arch)) {
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

[[nodiscard]] bool
append_cdna_full_workgroup_payload_restore(std::vector<uint32_t> &words,
                                           const ConSanMoiWorkgroupSources &sources,
                                           rj_code_arch_t arch, std::vector<std::string> &errors) {
  const auto copies = cdna_guest_workgroup_payload_copies(sources, arch);
  if (!copies) {
    errors.emplace_back("ConSan MOI found an incomplete AMDHSA workgroup-ID payload repair");
    return false;
  }
  InstructionSequence sequence(words);
  MoiEmissionRequirement require_emission(sequence, errors);
  for (const auto &[destination, source] : *copies) {
    if (destination != source)
      require_emission(
          append_moi_entry_salu_write(words, build_s_mov_b32(destination, source, arch), arch),
          "ConSan MOI could not restore a guest AMDHSA workgroup payload SGPR");
  }
  return sequence.finish();
}

[[nodiscard]] bool append_cdna_semantic_entry_scalar_spill_overrides(
    std::vector<uint32_t> &words, const SgprSpillSequence &spill,
    const ConSanMoiWorkgroupSources &sources, rj_code_arch_t arch,
    std::vector<std::string> &errors) {
  const auto copies = cdna_guest_workgroup_payload_copies(sources, arch);
  if (!copies) {
    errors.emplace_back("ConSan MOI found an incomplete AMDHSA entry-scalar backup repair");
    return false;
  }
  InstructionSequence sequence(words);
  MoiEmissionRequirement require_emission(sequence, errors);
  for (const auto &[destination, source] : *copies) {
    if (destination == source || destination < spill.sgpr_base ||
        destination >= static_cast<uint32_t>(spill.sgpr_base) + spill.sgpr_count) {
      continue;
    }
    require_emission.append(
        "ConSan MOI could not preserve a semantic AMDHSA entry-scalar backup slot",
        build_sgpr_spill_slot_override_sequence(spill, destination, source, arch));
  }
  return sequence.finish();
}

[[nodiscard]] bool
append_exact_workgroup_capture(std::vector<uint32_t> &words,
                               ConSanMoiPersistentWorkgroupRegisters scalar_destinations,
                               ConSanMoiPersistentWorkgroupRegisters vector_destinations,
                               const ConSanMoiWorkgroupSources &sources, rj_code_arch_t arch,
                               std::vector<std::string> &errors) {
  if (!scalar_destinations.empty() && !vector_destinations.empty()) {
    errors.emplace_back(
        "ConSan MOI exact workgroup-tuple prologue requires one complete register tuple");
    return false;
  }
  if (scalar_destinations.empty() && vector_destinations.empty())
    return true;

  const std::array<const ConSanMoiWorkgroupSource *, 4> source_tuple = {
      &sources.x, &sources.y, &sources.z, &sources.cluster_workgroup_id};
  const std::array<std::optional<uint16_t>, 4> scalar_tuple = scalar_destinations.values();
  const std::array<std::optional<uint16_t>, 4> vector_tuple = vector_destinations.values();
  InstructionSequence sequence(words);
  MoiEmissionRequirement require_emission(sequence, errors);
  for (size_t index = 0; index < source_tuple.size(); ++index) {
    if (!scalar_tuple[index] && !vector_tuple[index])
      continue;
    const ConSanMoiWorkgroupSource &source = *source_tuple[index];
    if (!source.is_well_formed()) {
      errors.emplace_back(
          "ConSan MOI exact workgroup-tuple prologue requires all launch coordinates");
      return false;
    }
    if (!source.has_value()) {
      if (index != 3u) {
        errors.emplace_back(
            "ConSan MOI exact workgroup-tuple prologue requires all launch coordinates");
        return false;
      }
      // A code-object-wide CDNA5 tuple reserves the cluster coordinate when
      // any owning kernel consumes it. Ordinary kernels in the same object do
      // not receive that launch input; zero is their exact cluster identity.
      if (scalar_tuple[index]) {
        require_emission(
            append_moi_entry_salu_write(
                words, build_s_mov_b32(*scalar_tuple[index], scalar_positive_inline_u32(0), arch),
                arch),
            "ConSan MOI exact scalar workgroup-tuple prologue could not zero an absent "
            "cluster coordinate");
      } else {
        require_emission(sequence.emit(
            build_v_mov_b32_e32(*vector_tuple[index], scalar_positive_inline_u32(0), arch)));
      }
      continue;
    }
    if (scalar_tuple[index]) {
      if (!source.scalar_src) {
        errors.emplace_back(
            "ConSan MOI exact scalar workgroup tuple requires scalar launch sources");
        return false;
      }
      const uint16_t destination = *scalar_tuple[index];
      const uint16_t source_operand = *source.scalar_src;
      require_emission(
          append_moi_entry_salu_write(words, build_s_mov_b32(destination, source_operand, arch),
                                      arch) &&
              (source.right_shift == 0u ||
               append_moi_entry_salu_write(
                   words,
                   build_s_lshr_b32(destination, destination,
                                    scalar_positive_inline_u32(source.right_shift), arch),
                   arch)) &&
              ((source.low_bit_count == 0u || source.low_bit_count == 32u) ||
               (append_moi_entry_salu_write(
                    words,
                    build_s_lshl_b32(destination, destination,
                                     scalar_positive_inline_u32(32u - source.low_bit_count), arch),
                    arch) &&
                append_moi_entry_salu_write(
                    words,
                    build_s_lshr_b32(destination, destination,
                                     scalar_positive_inline_u32(32u - source.low_bit_count), arch),
                    arch))),
          "ConSan MOI exact scalar workgroup-tuple prologue could not copy a launch source");
    } else if (vector_tuple[index]) {
      const uint16_t destination = *vector_tuple[index];
      require_emission(
          consan_detail::append_workgroup_source_value(words, source, destination, arch),
          "ConSan MOI exact vector workgroup-tuple prologue could not copy a launch source");
    }
  }
  return sequence.finish();
}

[[nodiscard]] bool append_moi_entry_workitem_coordinate(std::vector<uint32_t> &words,
                                                        uint16_t destination_vgpr,
                                                        uint8_t dimension, rj_code_arch_t arch) {
  constexpr uint32_t kPackedCoordinateMask = 0x3ffu;
  constexpr uint16_t kPackedCoordinateBits = 10u;
  if (dimension > 2u)
    return false;

  InstructionSequence sequence(words);
  const auto mask = instrumentation::build_v_and_b32_literal(
      destination_vgpr, kPackedCoordinateMask,
      dimension == 0u ? kAmdGpuWorkitemIdX : destination_vgpr, arch);
  if (dimension == 0u)
    return sequence.emit(mask);
  return sequence.emit_all(
      instrumentation::build_v_lshrrev_b32(
          destination_vgpr,
          scalar_positive_inline_u32(static_cast<uint16_t>(dimension * kPackedCoordinateBits)),
          kAmdGpuWorkitemIdX, arch),
      mask);
}

[[nodiscard]] bool append_moi_parallel_workgroup_shadow_initialization(
    std::vector<uint32_t> &words, const ConSanMoiWorkgroupShadowLayout &layout,
    uint16_t address_vgpr, uint16_t zero_vgpr, bool has_quad_zero_tuple,
    std::optional<uint16_t> state_sgpr, rj_code_arch_t arch, std::vector<std::string> &errors) {
  const uint32_t initialization_base =
      layout.lazy_initialization ? layout.validity_base : layout.base;
  const uint32_t initialization_size =
      layout.lazy_initialization ? layout.validity_size : layout.size;
  const ConSanTargetProfile *target = consan_target_profile(arch);
  const auto clear_plan =
      target ? plan_moi_workgroup_shadow_clear(*target, initialization_size,
                                               layout.initialization_lanes, has_quad_zero_tuple)
             : std::nullopt;
  if (!clear_plan) {
    errors.emplace_back(
        "ConSan MOI parallel workgroup-shadow initializer has no valid target clear plan");
    return false;
  }
  const bool use_wide_store =
      clear_plan->store_form != MoiWorkgroupShadowClearStoreForm::SplitB32Pair;
  const bool use_quad_store =
      clear_plan->store_form == MoiWorkgroupShadowClearStoreForm::PackedB128;
  const uint16_t zero_tuple_size = clear_plan->zero_vgpr_count;
  if (!state_sgpr || *state_sgpr > 118u || initialization_size == 0u ||
      initialization_size % 8u != 0u || layout.workitem_id_dimensions == 0u ||
      layout.workitem_id_dimensions > 3u || layout.initialization_lanes == 0u ||
      layout.initialization_lanes > 64u ||
      static_cast<uint32_t>(zero_vgpr) + zero_tuple_size > kMaxVgprs ||
      (address_vgpr >= zero_vgpr && address_vgpr < zero_vgpr + zero_tuple_size)) {
    errors.emplace_back(
        "ConSan MOI parallel workgroup-shadow initializer requires aligned LDS and vector state");
    return false;
  }
  const uint16_t exec_save = *state_sgpr;
  const uint16_t vcc_save = static_cast<uint16_t>(*state_sgpr + 2u);
  const uint16_t stride = static_cast<uint16_t>(*state_sgpr + 4u);
  const uint16_t saved_scc = static_cast<uint16_t>(*state_sgpr + 5u);
  const uint16_t selected_exec = static_cast<uint16_t>(*state_sgpr + 6u);
  const uint16_t end_vgpr = static_cast<uint16_t>(zero_vgpr + 1u);

  const uint16_t store_shift = use_quad_store ? 4u : 3u;
  std::vector<uint32_t> initialization;
  InstructionSequence sequence(initialization);
  MoiEmissionRequirement require_emission(sequence, errors);
  require_emission(
      sequence.emit_all(instrumentation::build_s_cselect_b32(saved_scc,
                                                             scalar_positive_inline_u32(1),
                                                             scalar_positive_inline_u32(0), arch),
                        instrumentation::build_s_mov_b64(vcc_save, kAmdGpuVccLo, arch),
                        instrumentation::build_s_mov_b64(exec_save, kAmdGpuExecLo, arch)) &&
          append_moi_entry_workitem_coordinate(initialization, end_vgpr, /*dimension=*/0u, arch) &&
          sequence.emit_all(
              instrumentation::build_v_cmp_gt_u32_vcc(
                  scalar_positive_inline_u32(layout.initialization_lanes), end_vgpr, arch),
              instrumentation::build_s_and_saveexec_b64(selected_exec, kAmdGpuVccLo, arch)),
      "ConSan MOI parallel workgroup-shadow initializer could not encode entry selection");
  for (uint8_t dimension = 1u; dimension < layout.workitem_id_dimensions; ++dimension) {
    require_emission(
        append_moi_entry_workitem_coordinate(initialization, address_vgpr, dimension, arch) &&
            sequence.emit_all(
                instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0u),
                                                        address_vgpr, arch),
                instrumentation::build_s_and_saveexec_b64(selected_exec, kAmdGpuVccLo, arch)),
        "ConSan MOI parallel workgroup-shadow initializer could not encode outer selection");
  }
  const InstructionSequence::Label restore_state = sequence.make_label();
  require_emission(
      sequence.emit_branch(restore_state, InstructionSequence::BranchKind::ExecZero) &&
          sequence.emit_all(
              instrumentation::build_s_mov_b64(selected_exec, kAmdGpuExecLo, arch),
              instrumentation::build_s_bcnt1_i32_b64(stride, kAmdGpuExecLo, arch),
              build_s_lshl_b32(stride, stride, scalar_positive_inline_u32(store_shift), arch),
              instrumentation::build_v_lshlrev_b32(
                  end_vgpr, scalar_positive_inline_u32(store_shift), end_vgpr, arch),
              instrumentation::build_v_add_u32_literal(address_vgpr, initialization_base, end_vgpr,
                                                       arch)) &&
          (use_wide_store || sequence.emit(instrumentation::build_v_mov_b32_literal(
                                 end_vgpr, initialization_base + initialization_size, arch))) &&
          sequence.emit(build_v_mov_b32_e32(zero_vgpr, scalar_positive_inline_u32(0), arch)),
      "ConSan MOI parallel workgroup-shadow initializer could not encode clear setup");
  if (use_wide_store) {
    for (uint16_t i = 1u; i < zero_tuple_size; ++i)
      require_emission(sequence.emit(build_v_mov_b32_e32(static_cast<uint16_t>(zero_vgpr + i),
                                                         scalar_positive_inline_u32(0), arch)));
    require_emission.append(
        "ConSan MOI parallel workgroup-shadow initializer could not encode bounds "
        "compare",
        instrumentation::build_v_cmp_gt_u32_literal_vcc(initialization_base + initialization_size,
                                                        address_vgpr, arch));
  } else {
    require_emission.append(
        "ConSan MOI parallel workgroup-shadow initializer could not encode bounds "
        "compare",
        instrumentation::build_v_cmp_gt_u32_vcc(vector_source_vgpr(end_vgpr), address_vgpr, arch));
  }
  require_emission.append(
      "ConSan MOI parallel workgroup-shadow initializer could not retain in-bounds lanes",
      instrumentation::build_s_and_b64(kAmdGpuExecLo, selected_exec, kAmdGpuVccLo, arch));
  const InstructionSequence::Label loop_begin = sequence.mark_label();
  if (use_wide_store) {
    require_emission(
        use_quad_store
            ? sequence.emit(instrumentation::build_ds_store_b128(address_vgpr, zero_vgpr, 0u, arch))
            : sequence.emit(instrumentation::build_ds_store_b64(address_vgpr, zero_vgpr, 0u, arch)),
        "ConSan MOI parallel workgroup-shadow initializer could not encode wide store");
  } else {
    require_emission.append(
        "ConSan MOI parallel workgroup-shadow initializer could not encode split store",
        instrumentation::build_ds_store_b32(address_vgpr, zero_vgpr, 0u, arch),
        instrumentation::build_ds_store_b32(address_vgpr, zero_vgpr, 4u, arch));
  }
  require_emission(
      sequence.emit(instrumentation::build_v_add_u32(address_vgpr, stride, address_vgpr, arch)) &&
          (use_wide_store ? sequence.emit(instrumentation::build_v_cmp_gt_u32_literal_vcc(
                                initialization_base + initialization_size, address_vgpr, arch))
                          : sequence.emit(instrumentation::build_v_cmp_gt_u32_vcc(
                                vector_source_vgpr(end_vgpr), address_vgpr, arch))) &&
          sequence.emit(
              instrumentation::build_s_and_b64(kAmdGpuExecLo, selected_exec, kAmdGpuVccLo, arch)) &&
          sequence.emit_branch(loop_begin, InstructionSequence::BranchKind::ExecNonzero) &&
          sequence.emit(instrumentation::build_s_wait_flat_load0(arch)) &&
          sequence.bind(restore_state) &&
          sequence.emit_all(
              instrumentation::build_s_mov_b64(kAmdGpuExecLo, exec_save, arch),
              instrumentation::build_s_mov_b64(kAmdGpuVccLo, vcc_save, arch),
              instrumentation::build_workgroup_barrier_only(arch),
              instrumentation::build_s_cmp_lg_u32(saved_scc, scalar_positive_inline_u32(0), arch)),
      "ConSan MOI parallel workgroup-shadow initializer could not encode clear loop");
  if (!sequence.finish(arch))
    return false;
  return InstructionSequence(words).emit(initialization);
}

[[nodiscard]] bool append_moi_workgroup_shadow_initialization(
    std::vector<uint32_t> &words, const ConSanMoiWorkgroupShadowLayout &layout,
    uint16_t address_vgpr, uint16_t zero_vgpr, bool has_quad_zero_tuple,
    std::optional<uint16_t> state_sgpr, rj_code_arch_t arch, std::vector<std::string> &errors) {
  if (layout.workitem_id_dimensions >= 1u && layout.workitem_id_dimensions <= 3u) {
    return append_moi_parallel_workgroup_shadow_initialization(
        words, layout, address_vgpr, zero_vgpr, has_quad_zero_tuple, state_sgpr, arch, errors);
  }
  errors.emplace_back("ConSan MOI workgroup-shadow initializer has invalid dimensionality");
  return false;
}

[[nodiscard]] std::optional<ConSanMoiWorkgroupSources>
runtime_selection_workgroup_sources(ConSanMoiPersistentWorkgroupRegisters scalar_registers,
                                    ConSanMoiPersistentWorkgroupRegisters vector_registers) {
  if (!scalar_registers.empty() && !vector_registers.empty()) {
    return std::nullopt;
  }
  const auto scalar_source = [](std::optional<uint16_t> reg) {
    return reg ? ConSanMoiWorkgroupSource::scalar(*reg) : ConSanMoiWorkgroupSource{};
  };
  const auto vector_source = [](std::optional<uint16_t> reg) {
    return reg ? ConSanMoiWorkgroupSource::vector(*reg) : ConSanMoiWorkgroupSource{};
  };
  if (!scalar_registers.empty()) {
    return ConSanMoiWorkgroupSources{
        .x = scalar_source(scalar_registers.x()),
        .y = scalar_source(scalar_registers.y()),
        .z = scalar_source(scalar_registers.z()),
        .cluster_workgroup_id = scalar_source(scalar_registers.cluster_workgroup_id()),
        .cdna_full_payload_base = std::nullopt,
        .cdna_guest_payload_base = std::nullopt,
        .cdna_guest_payload_mask = 0u,
    };
  }
  if (!vector_registers.empty()) {
    return ConSanMoiWorkgroupSources{
        .x = vector_source(vector_registers.x()),
        .y = vector_source(vector_registers.y()),
        .z = vector_source(vector_registers.z()),
        .cluster_workgroup_id = vector_source(vector_registers.cluster_workgroup_id()),
        .cdna_full_payload_base = std::nullopt,
        .cdna_guest_payload_base = std::nullopt,
        .cdna_guest_payload_mask = 0u,
    };
  }
  return std::nullopt;
}

[[nodiscard]] bool append_runtime_workgroup_selection_initialization(
    std::vector<uint32_t> &words, const ConSanMoiWorkgroupSource &selection_destination,
    ConSanMoiDispatchIdCapture dispatch_capture, const ConSanMoiWorkgroupSources &workgroup_sources,
    uint32_t sample_stride, uint32_t sample_offset, uint64_t report_dispatch_id,
    uint16_t scratch_sgpr_base, rj_code_arch_t arch, std::vector<std::string> &errors) {
  if (sample_stride <= 1u || (sample_stride & (sample_stride - 1u)) != 0u) {
    errors.emplace_back("ConSan MOI runtime-selection prologue has an invalid sample stride");
    return false;
  }
  const uint16_t saved_scc = static_cast<uint16_t>(scratch_sgpr_base + 4u);
  const uint16_t quotient = static_cast<uint16_t>(scratch_sgpr_base + 5u);
  const uint16_t residue = static_cast<uint16_t>(scratch_sgpr_base + 6u);
  if (!selection_destination.is_well_formed() || !selection_destination.has_value() ||
      selection_destination.private_offset) {
    errors.emplace_back("ConSan MOI runtime-selection prologue has an invalid destination");
    return false;
  }
  std::vector<uint32_t> selection;
  InstructionSequence sequence(selection);
  MoiEmissionRequirement require_emission(sequence, errors);
  uint16_t dispatch_id_source = 0u;
  if (dispatch_capture.sgpr()) {
    dispatch_id_source = *dispatch_capture.sgpr();
  } else if (dispatch_capture.vgpr()) {
    require_emission.append(
        "ConSan MOI runtime-selection prologue could not read its dispatch identity",
        instrumentation::build_v_readfirstlane_b32(quotient, *dispatch_capture.vgpr(), arch),
        instrumentation::build_valu_to_salu_dependency_wait(arch));
    dispatch_id_source = quotient;
  } else {
    dispatch_id_source =
        scalar_positive_inline_u32(static_cast<uint32_t>(report_dispatch_id) & 63u);
  }
  require_emission.append(
      "ConSan MOI runtime-selection prologue could not initialize its hash",
      instrumentation::build_s_cselect_b32(saved_scc, scalar_positive_inline_u32(1u),
                                           scalar_positive_inline_u32(0u), arch),
      instrumentation::build_s_sub_u32(residue, scalar_positive_inline_u32(0u), dispatch_id_source,
                                       arch));
  require_emission(
      append_moi_runtime_workgroup_mix(selection, workgroup_sources.x, quotient, residue, arch) &&
          append_moi_runtime_workgroup_mix(selection, workgroup_sources.y, quotient, residue,
                                           arch) &&
          append_moi_runtime_workgroup_mix(selection, workgroup_sources.z, quotient, residue,
                                           arch) &&
          append_moi_runtime_workgroup_mix(selection, workgroup_sources.cluster_workgroup_id,
                                           quotient, residue, arch),
      "ConSan MOI runtime-selection prologue could not mix its workgroup tuple");
  const uint16_t shift = scalar_positive_inline_u32(std::countr_zero(sample_stride));
  require_emission.append("ConSan MOI runtime-selection prologue could not reduce its hash",
                          build_s_lshr_b32(quotient, residue, shift, arch),
                          build_s_lshl_b32(quotient, quotient, shift, arch),
                          instrumentation::build_s_sub_u32(residue, residue, quotient, arch));
  require_emission(append_moi_runtime_workgroup_residue_compare(
                       selection, residue, quotient, sample_offset & (sample_stride - 1u), arch),
                   "ConSan MOI runtime-selection prologue could not compare its hash");
  const uint16_t scalar_destination = selection_destination.scalar_src.value_or(residue);
  require_emission.append(
      "ConSan MOI runtime-selection prologue could not materialize its result",
      instrumentation::build_s_cselect_b32(scalar_destination, scalar_positive_inline_u32(1u),
                                           scalar_positive_inline_u32(0u), arch));
  require_emission((!selection_destination.vector_src ||
                    sequence.emit(build_v_mov_b32_e32(*selection_destination.vector_src,
                                                      scalar_destination, arch))) &&
                       sequence.emit(instrumentation::build_s_cmp_lg_u32(
                           saved_scc, scalar_positive_inline_u32(0u), arch)),
                   "ConSan MOI runtime-selection prologue could not restore SCC");
  if (!sequence.finish())
    return false;
  return InstructionSequence(words).emit(selection);
}

[[nodiscard]] std::optional<std::vector<uint32_t>>
build_owner_epoch_prologue_words(const MoiOwnerEpochPrologueEmissionPlan &plan, rj_code_arch_t arch,
                                 std::vector<std::string> &errors) {
  if (!plan.is_well_formed()) {
    errors.emplace_back("ConSan MOI owner/epoch prologue has an invalid emission plan");
    return std::nullopt;
  }
  const uint16_t owner_vgpr = plan.vgpr_state.owner_epoch.owner;
  const uint16_t epoch_vgpr = plan.vgpr_state.owner_epoch.epoch;
  const std::optional<uint16_t> workgroup_key_vgpr = plan.vgpr_state.workgroup_key;
  const uint16_t owner_shift_bits = plan.owner_shift_bits;
  const ConSanMoiOwnerSource owner_source = plan.owner_source;
  const std::optional<uint16_t> owner_sgpr = plan.owner_sgpr;
  const bool one_based_owner_ids = plan.one_based_owner_ids;
  const std::optional<uint16_t> persistent_owner_sgpr = plan.persistent_sgprs.owner();
  const std::optional<uint16_t> persistent_epoch_sgpr = plan.persistent_sgprs.epoch();
  const std::optional<uint16_t> persistent_workgroup_key_sgpr = plan.persistent_sgprs.workgroup_key;
  const ConSanMoiPersistentWorkgroupRegisters exact_workgroup_sgprs =
      plan.persistent_sgprs.exact_workgroup;
  const ConSanMoiPersistentWorkgroupRegisters exact_workgroup_vgprs =
      plan.vgpr_state.exact_workgroup;
  const std::optional<ConSanMoiDispatchIdPreloadPlan> &dispatch_plan = plan.dispatch_plan;
  const ConSanMoiDispatchIdCapture dispatch_capture = plan.dispatch_capture;
  const std::optional<ConSanMoiWorkgroupSource> &runtime_workgroup_selection_source =
      plan.runtime_workgroup_selection_source;
  const std::optional<uint16_t> return_pc_sgpr = plan.return_pc_sgpr;
  const uint32_t runtime_sample_stride = plan.runtime_sample_stride;
  const uint32_t runtime_sample_offset = plan.runtime_sample_offset;
  const uint64_t runtime_report_dispatch_id = plan.runtime_report_dispatch_id;
  const std::optional<ConSanMoiEntryScalarBackup> &entry_scalar_backup = plan.entry_scalar_backup;
  const std::optional<ConSanMoiWorkgroupShadowLayout> &workgroup_shadow = plan.workgroup_shadow;
  const bool has_quad_zero_tuple = plan.has_quad_zero_tuple;
  const std::optional<ConSanMoiWorkgroupSources> &workgroup_sources = plan.workgroup_sources;
  std::vector<uint32_t> words;
  words.reserve(48u + (entry_scalar_backup
                           ? static_cast<size_t>(entry_scalar_backup->sgpr_count) * 4u + 1u
                           : 0u));
  InstructionSequence sequence(words);
  if (entry_scalar_backup && !append_moi_entry_scalar_backup(words, *entry_scalar_backup,
                                                             /*restore=*/false, arch, errors)) {
    return std::nullopt;
  }
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
  if (workgroup_shadow && workgroup_shadow->visible_evidence_sgpr &&
      !append_moi_entry_salu_write(words,
                                   build_s_mov_b32(*workgroup_shadow->visible_evidence_sgpr,
                                                   scalar_positive_inline_u32(0), arch),
                                   arch)) {
    errors.emplace_back("ConSan MOI prologue cannot initialize its visible-evidence latch");
    return std::nullopt;
  }
  // The workgroup-shadow initializer uses the entry-local scratch VGPRs. Capture a
  // scalar owner sourced from v0 before that scratch is allowed to clobber v0.
  if (persistent_owner_sgpr) {
    switch (owner_source) {
    case ConSanMoiOwnerSource::Automatic:
      errors.emplace_back("ConSan MOI scalar owner prologue has an unresolved automatic source");
      return std::nullopt;
    case ConSanMoiOwnerSource::WorkitemId: {
      if (!sequence.emit_all(instrumentation::build_v_readfirstlane_b32(*persistent_owner_sgpr,
                                                                        kAmdGpuWorkitemIdX, arch),
                             instrumentation::build_valu_to_salu_dependency_wait(arch))) {
        errors.emplace_back("ConSan MOI scalar owner prologue could not read its wave ID");
        return std::nullopt;
      }
      if (!append_moi_entry_salu_write(
              words,
              build_s_lshr_b32(*persistent_owner_sgpr, *persistent_owner_sgpr,
                               scalar_positive_inline_u32(owner_shift_bits), arch),
              arch) ||
          (one_based_owner_ids &&
           !append_moi_entry_salu_write(words,
                                        build_s_add_u32(*persistent_owner_sgpr,
                                                        *persistent_owner_sgpr,
                                                        scalar_positive_inline_u32(1), arch),
                                        arch))) {
        errors.emplace_back("ConSan MOI scalar owner prologue could not normalize its wave ID");
        return std::nullopt;
      }
      break;
    }
    case ConSanMoiOwnerSource::HwId: {
      const ConSanTargetProfile *target = consan_target_profile(arch);
      const consan_detail::MoiResidentWaveOwnerRequest request{
          .destination_sgpr = *persistent_owner_sgpr,
          .one_based = one_based_owner_ids,
      };
      if (target == nullptr ||
          !consan_detail::append_moi_resident_wave_owner(words, request, *target)) {
        errors.emplace_back(
            "ConSan MOI scalar owner prologue could not encode resident-wave identity");
        return std::nullopt;
      }
      break;
    }
    }
  }
  if (workgroup_shadow && !append_moi_workgroup_shadow_initialization(
                              words, *workgroup_shadow, owner_vgpr, epoch_vgpr, has_quad_zero_tuple,
                              return_pc_sgpr, arch, errors)) {
    return std::nullopt;
  }
  if (!capture_workgroup_before_dispatch &&
      (exact_workgroup_sgprs.complete() || exact_workgroup_vgprs.complete())) {
    if (!workgroup_sources) {
      errors.emplace_back("ConSan MOI exact workgroup-tuple prologue requires launch sources");
      return std::nullopt;
    }
    if (!append_exact_workgroup_capture(words, exact_workgroup_sgprs, exact_workgroup_vgprs,
                                        *workgroup_sources, arch, errors))
      return std::nullopt;
  }
  if (runtime_workgroup_selection_source) {
    const auto persistent_sources =
        runtime_selection_workgroup_sources(exact_workgroup_sgprs, exact_workgroup_vgprs);
    if (!persistent_sources || !return_pc_sgpr ||
        !append_runtime_workgroup_selection_initialization(
            words, *runtime_workgroup_selection_source, dispatch_capture, *persistent_sources,
            runtime_sample_stride, runtime_sample_offset, runtime_report_dispatch_id,
            *return_pc_sgpr, arch, errors)) {
      errors.emplace_back("ConSan MOI prologue could not cache its runtime workgroup selection");
      return std::nullopt;
    }
  }
  if (persistent_workgroup_key_sgpr) {
    if (!workgroup_sources || !return_pc_sgpr) {
      errors.emplace_back(
          "ConSan MOI scalar workgroup-key prologue requires launch sources and scalar scratch");
      return std::nullopt;
    }
    const uint16_t coordinate = static_cast<uint16_t>(*return_pc_sgpr + 4u);
    const uint16_t temporary = static_cast<uint16_t>(*return_pc_sgpr + 5u);
    if (!append_moi_entry_salu_write(
            words,
            build_s_mov_b32(*persistent_workgroup_key_sgpr, scalar_positive_inline_u32(0), arch),
            arch)) {
      return std::nullopt;
    }
    const InstructionSequence::Label invalid_label = sequence.make_label();
    const InstructionSequence::Label done_label = sequence.make_label();
    const bool has_z = workgroup_sources->z.scalar_src.has_value();
    const bool has_y = workgroup_sources->y.scalar_src.has_value();
    const uint32_t x_bits = has_z ? 8u : has_y ? 10u : 20u;
    const uint32_t y_bits = has_z ? 6u : 10u;
    const auto append_coordinate = [&](const ConSanMoiWorkgroupSource &source, uint32_t bits,
                                       uint32_t shift) -> bool {
      if (!source.scalar_src)
        return true;
      if (!append_moi_entry_salu_write(words, build_s_mov_b32(coordinate, *source.scalar_src, arch),
                                       arch))
        return false;
      if (source.right_shift != 0u &&
          !append_moi_entry_salu_write(
              words,
              build_s_lshr_b32(coordinate, coordinate,
                               scalar_positive_inline_u32(source.right_shift), arch),
              arch))
        return false;
      if (source.low_bit_count != 0u && source.low_bit_count < 32u) {
        const uint16_t mask_shift = scalar_positive_inline_u32(32u - source.low_bit_count);
        if (!append_moi_entry_salu_write(
                words, build_s_lshl_b32(coordinate, coordinate, mask_shift, arch), arch) ||
            !append_moi_entry_salu_write(
                words, build_s_lshr_b32(coordinate, coordinate, mask_shift, arch), arch))
          return false;
      }
      if (!append_moi_entry_salu_write(
              words,
              build_s_lshr_b32(temporary, coordinate, scalar_positive_inline_u32(bits), arch),
              arch))
        return false;
      if (!sequence.emit(instrumentation::build_s_cmp_eq_u32(
              temporary, scalar_positive_inline_u32(0), arch)) ||
          !sequence.emit_branch(invalid_label, InstructionSequence::BranchKind::SccZero))
        return false;
      if (shift != 0u &&
          !append_moi_entry_salu_write(
              words,
              build_s_lshl_b32(coordinate, coordinate,
                               scalar_positive_inline_u32(static_cast<uint16_t>(shift)), arch),
              arch))
        return false;
      return append_moi_entry_salu_write(words,
                                         build_s_add_u32(*persistent_workgroup_key_sgpr,
                                                         *persistent_workgroup_key_sgpr, coordinate,
                                                         arch),
                                         arch);
    };
    if (!append_coordinate(workgroup_sources->x, x_bits, 0u) ||
        !append_coordinate(workgroup_sources->y, y_bits, x_bits) ||
        !append_coordinate(workgroup_sources->z, 6u, x_bits + y_bits) ||
        !append_moi_entry_salu_write(words,
                                     build_s_add_u32(*persistent_workgroup_key_sgpr,
                                                     *persistent_workgroup_key_sgpr,
                                                     scalar_positive_inline_u32(1), arch),
                                     arch) ||
        !append_moi_entry_salu_write(words,
                                     build_s_lshr_b32(temporary, *persistent_workgroup_key_sgpr,
                                                      scalar_positive_inline_u32(20), arch),
                                     arch)) {
      errors.emplace_back("ConSan MOI scalar workgroup-key prologue could not pack launch ID");
      return std::nullopt;
    }
    if (!sequence.emit_all(
            instrumentation::build_s_cmp_eq_u32(temporary, scalar_positive_inline_u32(0), arch),
            instrumentation::build_s_cselect_b32(*persistent_workgroup_key_sgpr,
                                                 *persistent_workgroup_key_sgpr,
                                                 scalar_positive_inline_u32(0), arch)) ||
        !sequence.emit_branch(done_label, InstructionSequence::BranchKind::Unconditional) ||
        !sequence.bind(invalid_label)) {
      errors.emplace_back("ConSan MOI scalar workgroup-key prologue could not validate launch ID");
      return std::nullopt;
    }
    if (!append_moi_entry_salu_write(
            words,
            build_s_mov_b32(*persistent_workgroup_key_sgpr, scalar_positive_inline_u32(0), arch),
            arch) ||
        !sequence.bind(done_label) || !sequence.resolve_branches(arch))
      return std::nullopt;
  } else if (workgroup_key_vgpr) {
    if (!workgroup_sources || !return_pc_sgpr) {
      errors.emplace_back(
          "ConSan MOI workgroup-key prologue requires launch sources and EXEC-save state");
      return std::nullopt;
    }
    (void)sequence.emit(
        build_v_mov_b32_e32(*workgroup_key_vgpr, scalar_positive_inline_u32(0), arch));
    const MoiWorkgroupKeyRegisterPlan key_registers{
        .exec_save_sgpr = return_pc_sgpr,
        .cached_key_vgpr = std::nullopt,
        .cached_key_sgpr = std::nullopt,
    };
    if (!append_inline_workgroup_key(words, *workgroup_sources, key_registers, *workgroup_key_vgpr,
                                     owner_vgpr, epoch_vgpr,
                                     /*original_exec_save_offset=*/20u, arch)) {
      errors.emplace_back("ConSan MOI workgroup-key prologue could not encode launch identity");
      return std::nullopt;
    }
    if (!sequence.emit(instrumentation::build_s_mov_b64(
            kAmdGpuExecLo, static_cast<uint16_t>(*return_pc_sgpr + 20u), arch)))
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
    case ConSanMoiOwnerSource::Automatic:
      errors.emplace_back("ConSan MOI owner/epoch prologue has an unresolved automatic source");
      return std::nullopt;
    case ConSanMoiOwnerSource::WorkitemId: {
      if (!sequence.emit(instrumentation::build_v_lshrrev_b32(
              owner_vgpr, scalar_positive_inline_u32(owner_shift_bits), kAmdGpuWorkitemIdX,
              arch))) {
        errors.emplace_back("ConSan MOI owner/epoch prologue could not encode owner VGPR init");
        return std::nullopt;
      }
      if (one_based_owner_ids) {
        if (!sequence.emit(instrumentation::build_v_add_u32(
                owner_vgpr, scalar_positive_inline_u32(1), owner_vgpr, arch))) {
          errors.emplace_back("ConSan MOI owner/epoch prologue could not bias owner IDs");
          return std::nullopt;
        }
        // Inline metadata reserves zero for absent ownership. Keep its persistent
        // owner IDs one-based; Record/Replay and Sampled retain zero-based IDs
        // because they use owner zero to elect the canonical wave.
      }
      break;
    }
    case ConSanMoiOwnerSource::HwId: {
      if (!owner_sgpr) {
        errors.emplace_back("ConSan MOI hw_id owner source requires RJ_CONSAN_MOI_OWNER_SGPR");
        return std::nullopt;
      }
      const ConSanTargetProfile *target = consan_target_profile(arch);
      const consan_detail::MoiResidentWaveOwnerRequest request{
          .destination_sgpr = *owner_sgpr,
          .one_based = one_based_owner_ids,
      };
      if (target == nullptr ||
          !consan_detail::append_moi_resident_wave_owner(words, request, *target)) {
        errors.emplace_back(
            "ConSan MOI owner/epoch prologue could not encode resident-wave identity");
        return std::nullopt;
      }
      (void)sequence.emit(build_v_mov_b32_e32(owner_vgpr, *owner_sgpr, arch));
      break;
    }
    }
  if (persistent_epoch_sgpr) {
    if (!append_moi_entry_salu_write(
            words, build_s_mov_b32(*persistent_epoch_sgpr, scalar_positive_inline_u32(0), arch),
            arch))
      return std::nullopt;
  } else {
    (void)sequence.emit(build_v_mov_b32_e32(epoch_vgpr, scalar_positive_inline_u32(0), arch));
  }
  if (entry_scalar_backup && !append_moi_entry_scalar_backup(words, *entry_scalar_backup,
                                                             /*restore=*/true, arch, errors)) {
    return std::nullopt;
  }
  return words;
}

[[nodiscard]] std::optional<std::vector<uint32_t>>
build_private_epoch_prologue_words(const MoiPrivateEpochPrologueEmissionPlan &plan,
                                   rj_code_arch_t arch, std::vector<std::string> &errors) {
  if (!plan.is_well_formed()) {
    errors.emplace_back("ConSan MOI private-epoch prologue has an invalid emission plan");
    return std::nullopt;
  }
  const uint16_t scratch_vgpr = plan.scratch_vgpr;
  const ConSanMoiPrivateStateLayout &private_state_layout = plan.private_state_layout;
  const uint32_t epoch_offset = private_state_layout.epoch_offset;
  const std::optional<uint32_t> owner_offset = private_state_layout.owner_offset;
  const std::optional<uint32_t> workgroup_key_offset = private_state_layout.workgroup_key_offset;
  const std::optional<uint32_t> dispatch_id_offset = private_state_layout.dispatch_id_offset;
  const ConSanMoiPersistentWorkgroupPrivateOffsets &exact_workgroup_offsets =
      private_state_layout.exact_workgroup_offsets;
  const VgprSpillSequence &spill = plan.spill;
  const std::optional<SgprSpillSequence> &entry_scalar_spill = plan.entry_scalar_spill;
  const std::optional<ConSanMoiWorkgroupShadowLayout> &workgroup_shadow = plan.workgroup_shadow;
  const std::optional<ConSanMoiWorkgroupSources> &workgroup_sources = plan.workgroup_sources;
  const std::optional<ConSanMoiDispatchIdPreloadPlan> &dispatch_plan = plan.dispatch_plan;
  const ConSanMoiDispatchIdCapture dispatch_capture = plan.dispatch_capture;
  const std::optional<ConSanMoiWorkgroupSource> &runtime_workgroup_selection_source =
      plan.runtime_workgroup_selection_source;
  const std::optional<uint16_t> return_pc_sgpr = plan.return_pc_sgpr;
  const uint32_t runtime_sample_stride = plan.runtime_sample_stride;
  const uint32_t runtime_sample_offset = plan.runtime_sample_offset;
  const uint64_t runtime_report_dispatch_id = plan.runtime_report_dispatch_id;
  const auto epoch_store =
      instrumentation::build_private_store_b32(scratch_vgpr, epoch_offset, arch);
  std::optional<std::vector<uint32_t>> owner_store;
  if (owner_offset)
    owner_store = instrumentation::build_private_store_b32(scratch_vgpr, *owner_offset, arch);
  std::optional<std::vector<uint32_t>> workgroup_key_store;
  if (workgroup_key_offset)
    workgroup_key_store =
        instrumentation::build_private_store_b32(scratch_vgpr, *workgroup_key_offset, arch);
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
      (workgroup_key_offset && !workgroup_key_store) ||
      (dispatch_id_offset && (!dispatch_id_low_store || !dispatch_id_high_store)) ||
      std::ranges::any_of(std::views::iota(size_t{0}, exact_workgroup_stores.size()),
                          [&](size_t index) {
                            return exact_workgroup_offset_values[index] &&
                                   !exact_workgroup_stores[index];
                          }) ||
      !wait_store) {
    errors.emplace_back(
        "ConSan MOI private-epoch prologue could not encode persistent-state initialization");
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
  if (runtime_workgroup_selection_source) {
    if (!workgroup_sources || !return_pc_sgpr ||
        !append_runtime_workgroup_selection_initialization(
            words, *runtime_workgroup_selection_source, dispatch_capture, *workgroup_sources,
            runtime_sample_stride, runtime_sample_offset, runtime_report_dispatch_id,
            *return_pc_sgpr, arch, errors)) {
      errors.emplace_back(
          "ConSan MOI private-epoch prologue could not cache its runtime workgroup selection");
      return std::nullopt;
    }
  }
  if (workgroup_shadow && workgroup_shadow->visible_evidence_sgpr &&
      !append_moi_entry_salu_write(words,
                                   build_s_mov_b32(*workgroup_shadow->visible_evidence_sgpr,
                                                   scalar_positive_inline_u32(0), arch),
                                   arch)) {
    errors.emplace_back(
        "ConSan MOI private-epoch prologue cannot initialize its visible-evidence latch");
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
  if (entry_scalar_spill) {
    (void)sequence.emit(entry_scalar_spill->save_words);
    if (workgroup_sources && !append_cdna_semantic_entry_scalar_spill_overrides(
                                 words, *entry_scalar_spill, *workgroup_sources, arch, errors)) {
      return std::nullopt;
    }
  }
  if (dispatch_id_offset) {
    if (!dispatch_plan || !dispatch_capture.vgpr() || dispatch_capture.sgpr() ||
        *dispatch_capture.vgpr() != scratch_vgpr ||
        !append_dispatch_id_capture_and_restore(
            words, *dispatch_plan, dispatch_capture,
            !workgroup_sources || !workgroup_sources->cdna_full_payload_base.has_value(), arch,
            errors)) {
      errors.emplace_back(
          "ConSan MOI private-epoch prologue could not capture private dispatch identity");
      return std::nullopt;
    }
    (void)sequence.emit_all(*dispatch_id_low_store, *dispatch_id_high_store, *wait_store);
  }
  if (workgroup_shadow &&
      !append_moi_workgroup_shadow_initialization(
          words, *workgroup_shadow, scratch_vgpr, static_cast<uint16_t>(scratch_vgpr + 1u),
          /*has_quad_zero_tuple=*/false, return_pc_sgpr, arch, errors)) {
    return std::nullopt;
  }
  if (workgroup_key_store) {
    if (!workgroup_sources || !return_pc_sgpr) {
      errors.emplace_back("ConSan MOI private workgroup key requires entry ABI sources");
      return std::nullopt;
    }
    const MoiWorkgroupKeyRegisterPlan key_registers{
        .exec_save_sgpr = return_pc_sgpr,
        .cached_key_vgpr = std::nullopt,
        .cached_key_sgpr = std::nullopt,
    };
    if (!append_inline_workgroup_key(words, *workgroup_sources, key_registers, scratch_vgpr,
                                     static_cast<uint16_t>(scratch_vgpr + 1u),
                                     static_cast<uint16_t>(scratch_vgpr + 2u), 20u, arch)) {
      errors.emplace_back("ConSan MOI private prologue could not capture workgroup identity");
      return std::nullopt;
    }
    if (!sequence.emit_all(*workgroup_key_store,
                           instrumentation::build_s_mov_b64(
                               kAmdGpuExecLo, static_cast<uint16_t>(*return_pc_sgpr + 20u), arch)))
      return std::nullopt;
  }
  if (exact_workgroup_offsets.complete()) {
    if (!workgroup_sources) {
      errors.emplace_back("ConSan MOI private exact workgroup tuple requires entry ABI sources");
      return std::nullopt;
    }
    const std::array<ConSanMoiWorkgroupSource, 4> sources = {
        workgroup_sources->x, workgroup_sources->y, workgroup_sources->z,
        workgroup_sources->cluster_workgroup_id};
    for (size_t index = 0; index < sources.size(); ++index) {
      if (!exact_workgroup_offset_values[index])
        continue;
      if (!sources[index].is_well_formed()) {
        errors.emplace_back(
            "ConSan MOI private prologue requires every exact workgroup coordinate");
        return std::nullopt;
      }
      if (!sources[index].has_value()) {
        if (index != 3u) {
          errors.emplace_back(
              "ConSan MOI private prologue requires every exact workgroup coordinate");
          return std::nullopt;
        }
        (void)sequence.emit(build_v_mov_b32_e32(scratch_vgpr, scalar_positive_inline_u32(0), arch));
      } else if (!consan_detail::append_workgroup_source_value(words, sources[index], scratch_vgpr,
                                                               arch)) {
        errors.emplace_back(
            "ConSan MOI private prologue could not capture an exact workgroup coordinate");
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

[[nodiscard]] bool kernel_contains_patch_anchor(const ConSanProgramContainer &kernel,
                                                const ConSanCommittedPatchGeometry &patch) {
  return kernel.has_text_range && patch.anchor_offset >= kernel.entry_text_offset &&
         patch.anchor_offset - kernel.entry_text_offset < kernel.code_size;
}

[[nodiscard]] bool kernel_owns_patch(const ConSanProgramContainer &kernel,
                                     const ConSanPatchLoweringProduct &patch) {
  if (!patch.owner_descriptor_file_offsets.empty()) {
    return std::ranges::find(patch.owner_descriptor_file_offsets, kernel.descriptor_file_offset) !=
           patch.owner_descriptor_file_offsets.end();
  }
  return kernel_contains_patch_anchor(kernel, patch);
}

template <typename Visitor>
void for_each_moi_patch(const ConSanTransformArtifacts &result, Visitor &&visitor) {
  for (const ConSanPatchInfo &patch : result.patches)
    visitor(patch);
  for (const ConSanTextFragment &fragment : result.staged_text_fragments)
    visitor(fragment.patch);
}

template <typename Predicate>
[[nodiscard]] const ConSanPatchInfo *find_moi_patch(const ConSanTransformArtifacts &result,
                                                    Predicate &&predicate) {
  const ConSanPatchInfo *found = nullptr;
  for_each_moi_patch(result, [&](const ConSanPatchInfo &patch) {
    if (found == nullptr && predicate(patch))
      found = &patch;
  });
  return found;
}

[[nodiscard]] bool enable_moi_full_workgroup_id_payload(rj_code_arch_t arch,
                                                        ConSanTransformArtifacts &result) {
  if (result.replacement.empty() || !consan_is_capability_arch(arch)) {
    return true;
  }

  std::unordered_map<std::string_view, uint64_t> owners_by_name;
  for_each_moi_patch(result, [&](const ConSanPatchInfo &patch) {
    if (!consan_detail::patch_requires_full_workgroup_id_payload(result.observation_plan().engine,
                                                                 arch, patch))
      return;
    for (uint64_t descriptor_offset : patch.owner_descriptor_file_offsets) {
      const ConSanProgramContainer *owner =
          result.program_inventory.find_kernel_by_descriptor(descriptor_offset);
      if (owner == nullptr) {
        result.errors.emplace_back(
            "ConSan MOI could not resolve an owner before enabling workgroup IDs");
        return;
      }
      const auto [it, inserted] =
          owners_by_name.emplace(owner->name, owner->descriptor_file_offset);
      if (!inserted && it->second != owner->descriptor_file_offset) {
        result.errors.emplace_back(
            "ConSan MOI found duplicate owner names while enabling workgroup IDs");
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
        "ConSan MOI could not parse its replacement before enabling workgroup IDs");
    return false;
  }
  CodeObjectPatcher patcher(code_object);
  std::unordered_set<std::string_view> patched_owner_names;
  MoiDescriptorSgprRequirements scalar_requirements;
  for (const AmdGpuKernelInfo &kernel : code_object.kernels()) {
    const auto owner = owners_by_name.find(kernel.name);
    if (owner == owners_by_name.end())
      continue;
    if (!patched_owner_names.insert(kernel.name).second) {
      result.errors.emplace_back(
          "ConSan MOI found duplicate owner names while enabling workgroup IDs");
      return false;
    }
    const uint64_t descriptor_offset = kernel.descriptor_file_offset;
    const std::span<const uint8_t> current_image = patcher.image_bytes();
    auto descriptor_value = read_kernel_descriptor(current_image, descriptor_offset);
    if (!descriptor_value) {
      result.errors.emplace_back("ConSan MOI workgroup-ID descriptor exceeds replacement bytes");
      return false;
    }
    KD &descriptor = *descriptor_value;
    if (consan_arch_supports_kernarg_preload_overflow_recovery(arch)) {
      const uint32_t user_sgpr_count = moi_descriptor_user_sgpr_count(descriptor);
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
      result.errors.emplace_back("ConSan MOI could not enable its full workgroup-ID payload");
      return false;
    }
  }
  if (patched_owner_names.size() != owners_by_name.size()) {
    result.errors.emplace_back(
        "ConSan MOI could not resolve every replacement owner while enabling workgroup IDs");
    return false;
  }
  if (!apply_moi_descriptor_requirements(patcher, code_object, result.program_inventory,
                                         MoiDescriptorVgprRequirements{}, scalar_requirements,
                                         MoiDescriptorPrivateRequirements{}, nullptr, nullptr, arch,
                                         "ConSan MOI full workgroup-ID payload", result.errors))
    return false;
  result.replacement = std::move(patcher).emit();
  return true;
}

[[nodiscard]] bool apply_moi_dispatch_id_descriptor(CodeObjectPatcher &patcher,
                                                    uint64_t descriptor_file_offset,
                                                    const ConSanMoiDispatchIdPreloadPlan &plan,
                                                    ConSanMoiDispatchIdCapture capture,
                                                    rj_code_arch_t arch,
                                                    std::vector<std::string> &errors) {
  const std::span<const uint8_t> current_image = patcher.image_bytes();
  auto descriptor_value = read_kernel_descriptor(current_image, descriptor_file_offset);
  if (!descriptor_value) {
    errors.emplace_back("ConSan MOI dispatch-ID descriptor exceeds ELF bytes");
    return false;
  }
  if (!plan.supported() || !capture.present() ||
      (capture.sgpr() && static_cast<uint32_t>(*capture.sgpr()) + 2u > kMaxSgprs) ||
      (capture.vgpr() && static_cast<uint32_t>(*capture.vgpr()) + 2u > kMaxVgprs)) {
    errors.emplace_back("ConSan MOI dispatch-ID descriptor has an unsupported preload plan");
    return false;
  }

  KD &desc = *descriptor_value;
  const ConSanMoiDispatchIdPreloadPlan current_plan =
      moi_descriptor_dispatch_id_preload_plan(desc, arch);
  if (current_plan.support != plan.support ||
      current_plan.dispatch_id_sgpr != plan.dispatch_id_sgpr ||
      current_plan.original_user_sgpr_count != plan.original_user_sgpr_count ||
      current_plan.system_sgpr_count != plan.system_sgpr_count) {
    errors.emplace_back("ConSan MOI dispatch-ID descriptor changed after preload planning");
    return false;
  }

  if (plan.descriptor_change_required()) {
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
    errors.emplace_back("ConSan MOI could not patch dispatch-ID descriptor state");
    return false;
  }
  return true;
}

void rollback_moi_dispatch_id_as_unsupported(ConSanTransformArtifacts &result, std::string reason) {
  result.outcome = ConSanTransformOutcome::Unsupported;
  result.discard_candidate_modification();
  result.warnings.emplace_back("ConSan MOI dispatch-ID capture is unsupported: " +
                               std::move(reason));
}

[[nodiscard]] std::string
dispatch_id_preload_rejection_reason(std::string_view kernel_name,
                                     const ConSanMoiDispatchIdPreloadPlan &plan,
                                     ConSanMoiDispatchIdCapture capture) {
  return "kernel '" + std::string(kernel_name) +
         "' preload plan support=" + std::to_string(static_cast<uint32_t>(plan.support)) +
         " user_sgprs=" + std::to_string(plan.original_user_sgpr_count) +
         " required_sgprs=" + std::to_string(plan.required_sgpr_count) +
         (capture.sgpr()
              ? " persistent_sgpr=" + std::to_string(*capture.sgpr())
              : " persistent_vgpr=" + std::to_string(capture.vgpr().value_or(kMaxVgprs)));
}

void note_dispatch_id_patch_info(ConSanPatchAbiEffects &effects,
                                 const ConSanMoiDispatchIdPreloadPlan &plan,
                                 ConSanMoiDispatchIdCapture capture) {
  effects.dispatch_id_prologue = ConSanMoiDispatchIdPrologueEffect{
      .preload = plan,
      .capture = capture,
  };
}

void try_apply_private_epoch_prologue_patch(const ConSanRequest &request,
                                            const BoundRuntimeResources &resources,
                                            const ConSanMoiOperatingPoint &operating_point,
                                            const MoiObjectModeSemantics &mode_semantics,
                                            rj_code_arch_t arch, ConSanTransformArtifacts &result) {
  if (!result.modified() || result.replacement.empty()) {
    result.warnings.emplace_back(
        "ConSan MOI private-epoch prologue skipped because no private-epoch access probe was "
        "emitted");
    return;
  }
  const ConSanTargetProfile *target = consan_target_profile(arch);
  if (target == nullptr) {
    result.errors.emplace_back("ConSan MOI private-epoch prologue has no admitted target profile");
    return;
  }

  std::span<const uint8_t> active_bytes(result.replacement.data(), result.replacement.size());
  AmdGpuCodeObject code_object(active_bytes.data(), active_bytes.size());
  CodeObjectPatcher patcher(code_object);
  if (patcher.text_bytes().empty()) {
    result.errors.emplace_back("ConSan MOI private-epoch prologue found no .text section");
    return;
  }
  struct PlannedPrivateEpochPrologue {
    const ConSanProgramContainer *kernel = nullptr;
    uint64_t active_descriptor_file_offset = 0;
    MoiPrivateEpochPrologueEmissionPlan emission;
    ConSanMoiPrivateStateLayout private_state_layout;
    uint32_t required_private_bytes = 0;
  };
  std::vector<PlannedPrivateEpochPrologue> planned;
  MoiDescriptorPrivateRequirements private_requirements;
  for (const ConSanProgramContainer &kernel : result.program_inventory.kernels()) {
    const ConSanPatchInfo *access_patch = find_moi_patch(result, [&](const ConSanPatchInfo &patch) {
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
          "ConSan MOI private-epoch prologue could not resolve an active kernel descriptor");
      continue;
    }

    const auto descriptor_value =
        read_kernel_descriptor(active_bytes, active_kernel->descriptor_file_offset);
    if (!descriptor_value) {
      result.errors.emplace_back("ConSan MOI private-epoch descriptor exceeds ELF bytes");
      return;
    }
    const KD &descriptor = *descriptor_value;

    const ConSanMoiPrivateStateLayout &layout = *access_patch->private_state_layout;
    if (!layout.is_well_formed()) {
      result.errors.emplace_back("ConSan MOI private-epoch prologue found an invalid layout");
      return;
    }
    ConSanMoiOperatingPoint kernel_point = operating_point;
    const std::array<uint64_t, 1> kernel_owner = {kernel.descriptor_file_offset};
    if (!result.moi_operating_point.owner_transient_sgprs.empty() &&
        !apply_moi_transient_sgpr_assignment(request, kernel_point, result.moi_operating_point,
                                             kernel_owner)) {
      result.errors.emplace_back(
          "ConSan MOI private-epoch prologue has no transient scalar assignment for kernel '" +
          kernel.name + "'");
      return;
    }
    const bool has_private_dispatch_id = layout.dispatch_id_offset.has_value();
    const ConSanMoiDispatchIdCapture dispatch_capture =
        has_private_dispatch_id ? ConSanMoiDispatchIdCapture::in_vgprs(*access_patch->scratch_vgpr)
                                : dispatch_id_capture(kernel_point);
    std::optional<ConSanMoiDispatchIdPreloadPlan> dispatch_plan;
    if (dispatch_capture.present()) {
      dispatch_plan = moi_descriptor_dispatch_id_preload_plan(descriptor, arch);
      if (!dispatch_plan->supported() ||
          (dispatch_capture.sgpr() &&
           *dispatch_capture.sgpr() < dispatch_plan->required_sgpr_count)) {
        rollback_moi_dispatch_id_as_unsupported(
            result,
            dispatch_id_preload_rejection_reason(kernel.name, *dispatch_plan, dispatch_capture));
        return;
      }
    }
    const bool has_private_workgroup_key = layout.workgroup_key_offset.has_value();
    const bool has_private_exact_workgroup = layout.exact_workgroup_offsets.complete();

    const auto private_limit = consan_address_free_private_limit(arch);
    if (!private_limit) {
      result.warnings.emplace_back(
          "ConSan MOI private-epoch prologue has no address-free scratch support");
      continue;
    }
    SpillManager manager(layout.ephemeral_base, *private_limit);
    std::optional<ConSanMoiWorkgroupShadowLayout> workgroup_shadow = access_patch->workgroup_shadow;
    if (workgroup_shadow) {
      workgroup_shadow->initialization_lanes = moi_workgroup_shadow_initialization_lanes(
          *target, active_kernel->required_workgroup_size);
    }
    std::optional<ConSanMoiWorkgroupSources> workgroup_sources;
    if (has_private_workgroup_key || has_private_exact_workgroup) {
      std::vector<std::string> source_errors;
      std::optional<uint16_t> full_payload_user_sgpr_count;
      if (has_private_exact_workgroup &&
          consan_arch_supports_kernarg_preload_overflow_recovery(arch)) {
        const auto exact_tuple_descriptor =
            read_kernel_descriptor(active_bytes, active_kernel->descriptor_file_offset);
        if (!exact_tuple_descriptor) {
          result.errors.emplace_back("ConSan MOI private exact-tuple descriptor exceeds ELF bytes");
          return;
        }
        full_payload_user_sgpr_count =
            dispatch_plan
                ? dispatch_plan->expanded_user_sgpr_count
                : static_cast<uint16_t>(moi_descriptor_user_sgpr_count(*exact_tuple_descriptor));
      }
      workgroup_sources = moi_descriptor_workgroup_sources(
          active_bytes, active_kernel->descriptor_file_offset, arch, source_errors,
          kernel.uses_cluster_workgroup_id, full_payload_user_sgpr_count);
      if (!workgroup_sources) {
        result.errors.insert(result.errors.end(), source_errors.begin(), source_errors.end());
        return;
      }
    }
    const uint16_t prologue_temporary_vgpr_count =
        has_private_workgroup_key ? 3u : ((has_private_dispatch_id || workgroup_shadow) ? 2u : 1u);
    const bool fixed_lane_entry_scalar_reservoir = kernel_point.moi_exec_save_sgpr &&
                                                   kernel_point.has_moi_scalar_spill() &&
                                                   !kernel.uses_dynamic_stack.value_or(false);
    std::optional<uint16_t> entry_scalar_reservoir_vgpr;
    uint16_t prologue_spill_vgpr_count = prologue_temporary_vgpr_count;
    if (fixed_lane_entry_scalar_reservoir) {
      const auto access_resource_plan =
          std::ranges::find_if(result.resource_plans, [&](const ConSanCandidateResourcePlan &plan) {
            return plan.site_kind == ConSanResourceSiteKind::Access && plan.scratch_vgpr &&
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
            "ConSan MOI private-epoch prologue could not resolve its planned scalar reservoir");
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
          "ConSan MOI private-epoch prologue could not preserve its temporary VGPR");
      continue;
    }
    std::optional<SgprSpillSequence> entry_scalar_spill;
    if (kernel_point.moi_exec_save_sgpr) {
      const uint16_t guest_entry_sgpr_count =
          static_cast<uint16_t>(moi_descriptor_user_sgpr_count(descriptor) +
                                moi_descriptor_system_sgpr_count(descriptor));
      const uint16_t scalar_base = *kernel_point.moi_exec_save_sgpr;
      const uint16_t scalar_end = static_cast<uint16_t>(std::min<uint32_t>(
          static_cast<uint32_t>(scalar_base) +
              moi_exec_save_sgpr_count(resolve_moi_exec_save_requirement(
                                           request, resources, kernel_point, mode_semantics),
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
              "ConSan MOI private-epoch prologue could not preserve its entry ABI SGPRs");
          continue;
        }
      }
    }
    uint32_t required_private_bytes =
        entry_scalar_spill ? entry_scalar_spill->total_private_bytes : spill->total_private_bytes;
    for_each_moi_patch(result, [&](const ConSanPatchInfo &patch) {
      if (kernel_owns_patch(kernel, patch))
        required_private_bytes =
            std::max(required_private_bytes, patch.required_private_segment_size);
    });
    private_requirements[kernel.descriptor_file_offset] = required_private_bytes;
    std::optional<ConSanMoiWorkgroupShadowLayout> prologue_workgroup_shadow = workgroup_shadow;
    if (prologue_workgroup_shadow) {
      prologue_workgroup_shadow->visible_evidence_sgpr =
          inline_shadow_visible_evidence_sgpr(project_inline_shadow_scalar_state(kernel_point));
    }
    MoiPrivateEpochPrologueEmissionPlan emission{
        .scratch_vgpr = *access_patch->scratch_vgpr,
        .private_state_layout = layout,
        .spill = std::move(*spill),
        .entry_scalar_spill = std::move(entry_scalar_spill),
        .workgroup_shadow = std::move(prologue_workgroup_shadow),
        .workgroup_sources = std::move(workgroup_sources),
        .dispatch_plan = dispatch_plan,
        .dispatch_capture = dispatch_capture,
        .runtime_workgroup_selection_source = moi_runtime_workgroup_selection_source(),
        .return_pc_sgpr = kernel_point.moi_exec_save_sgpr,
        .runtime_sample_stride = request.moi_runtime_sample_stride,
        .runtime_sample_offset = request.moi_runtime_sample_offset,
        .runtime_report_dispatch_id = resources.moi_report_dispatch_id,
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
    result.warnings.emplace_back("ConSan MOI private-epoch prologue found no patchable kernels");
    return;
  }
  MoiDescriptorSgprRequirements dispatch_sgpr_requirements;
  for (const PlannedPrivateEpochPrologue &item : planned) {
    if (!item.emission.dispatch_plan || !item.emission.dispatch_capture.present())
      continue;
    uint32_t required = item.emission.dispatch_plan->required_sgpr_count;
    if (const auto capture = item.emission.dispatch_capture.sgpr())
      required = std::max<uint32_t>(required, static_cast<uint32_t>(*capture) + 2u);
    note_maximum_descriptor_extent(dispatch_sgpr_requirements, item.kernel->descriptor_file_offset,
                                   static_cast<uint16_t>(required));
  }
  if (!apply_moi_descriptor_requirements(
          patcher, code_object, result.program_inventory, MoiDescriptorVgprRequirements{},
          dispatch_sgpr_requirements, private_requirements, nullptr, nullptr, arch,
          "ConSan MOI private-epoch prologue", result.errors))
    return;
  for (const PlannedPrivateEpochPrologue &item : planned) {
    if (item.emission.dispatch_capture.present() &&
        (!item.emission.dispatch_plan ||
         !apply_moi_dispatch_id_descriptor(patcher, item.active_descriptor_file_offset,
                                           *item.emission.dispatch_plan,
                                           item.emission.dispatch_capture, arch, result.errors))) {
      return;
    }
  }

  std::vector<ConSanTextFragment> fragments;
  fragments.reserve(planned.size());
  for (const PlannedPrivateEpochPrologue &item : planned) {
    auto words = build_private_epoch_prologue_words(item.emission, arch, result.errors);
    if (!words)
      return;
    ConSanPatchInfo info;
    info.kind = ConSanPatchKind::KernelEntryMoiPrivateEpochPrologue;
    info.anchor_offset = item.kernel->entry_text_offset;
    info.original_size = 0u;
    info.scratch_vgpr = item.emission.scratch_vgpr;
    info.private_state_layout = item.private_state_layout;
    info.spilled_vgpr_count = item.emission.spill.vgpr_count;
    info.required_private_segment_size = item.required_private_bytes;
    note_dynamic_stack_private_requirement(info, &item.emission.spill);
    if (item.emission.dispatch_plan && item.emission.dispatch_capture.present())
      note_dispatch_id_patch_info(info, *item.emission.dispatch_plan,
                                  item.emission.dispatch_capture);
    info.owner_descriptor_file_offsets.push_back(item.kernel->descriptor_file_offset);
    fragments.push_back(ConSanTextFragment::entry_prefix(std::move(*words), std::move(info)));
  }
  result.replacement = std::move(patcher).emit();
  (void)stage_consan_text_fragments(std::move(fragments), result);
}

[[nodiscard]] uint32_t
moi_owner_epoch_prologue_required_vgpr_count(const MoiOwnerEpochPrologueEmissionPlan &emission) {
  uint32_t required = emission.vgpr_state.required_vgpr_count();
  if (emission.workgroup_shadow)
    required = std::max<uint32_t>(required, emission.vgpr_state.owner_epoch.epoch +
                                                (emission.has_quad_zero_tuple ? 4u : 2u));
  if (emission.dispatch_capture.vgpr())
    required = std::max<uint32_t>(required, *emission.dispatch_capture.vgpr() + 2u);
  return required;
}

[[nodiscard]] bool moi_entry_scalar_backup_preserves_persistent_outputs(
    const MoiOwnerEpochPrologueEmissionPlan &emission, bool exec_save_sgprs_persistent,
    bool automatic_owner_sgpr, std::optional<uint16_t> visible_evidence_sgpr, uint16_t sgpr_base,
    uint16_t sgpr_count) {
  const auto overlaps = [&](std::optional<uint16_t> reg, uint16_t width = 1u) {
    return reg && range_overlaps(sgpr_base, sgpr_count, *reg, width);
  };
  const bool backs_up_only_automatic_owner = automatic_owner_sgpr && emission.owner_sgpr &&
                                             sgpr_base == *emission.owner_sgpr && sgpr_count == 1u;
  bool conflict = (exec_save_sgprs_persistent && !backs_up_only_automatic_owner) ||
                  overlaps(emission.dispatch_capture.sgpr(), kMoiDispatchStateSgprCount) ||
                  overlaps(visible_evidence_sgpr);
  emission.persistent_sgprs.for_each_range(
      [&](std::optional<uint16_t> reg, uint16_t width) { conflict |= overlaps(reg, width); });
  return !conflict;
}

[[nodiscard]] bool
moi_owner_epoch_prologue_uses_vgpr(const MoiOwnerEpochPrologueEmissionPlan &emission,
                                   uint16_t vgpr) {
  const auto overlaps = [vgpr](std::optional<uint16_t> base, uint16_t width = 1u) {
    return base && range_overlaps(vgpr, 1u, *base, width);
  };
  if (overlaps(emission.vgpr_state.owner_epoch.owner))
    return true;
  const uint16_t epoch_width = emission.workgroup_shadow
                                   ? static_cast<uint16_t>(emission.has_quad_zero_tuple ? 4u : 2u)
                                   : 1u;
  if (overlaps(emission.vgpr_state.owner_epoch.epoch, epoch_width) ||
      overlaps(emission.vgpr_state.workgroup_key) ||
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
    std::span<const uint8_t> bytes, const ConSanRequest &request,
    const BoundRuntimeResources &resources, const ConSanMoiOperatingPoint &operating_point,
    std::span<const ConSanMoiPrologueScratchVgprAssignment> prologue_scratch_assignments,
    const MoiObjectModeSemantics &mode_semantics, const MoiPrologueModePolicy &mode_policy,
    rj_code_arch_t arch, ConSanTransformArtifacts &result) {
  if (!operating_point.moi_initialize_owner_epoch)
    return;
  if (!consan_is_capability_arch(arch)) {
    result.warnings.emplace_back(
        "ConSan MOI owner/epoch prologue does not support this architecture");
    return;
  }
  const ConSanTargetProfile *target = consan_target_profile(arch);
  if (target == nullptr) {
    result.errors.emplace_back("ConSan MOI owner/epoch prologue has no admitted target profile");
    return;
  }
  if (operating_point.automatic_moi_private_epoch) {
    try_apply_private_epoch_prologue_patch(request, resources, operating_point, mode_semantics,
                                           arch, result);
    if (!result.errors.empty() || result.moi_operating_point.owner_persistent_vgprs.empty())
      return;
  }
  if (!operating_point.moi_owner_epoch_vgprs.complete() &&
      result.moi_operating_point.owner_persistent_vgprs.empty() &&
      !operating_point.moi_persistent_sgprs.complete()) {
    result.errors.emplace_back(
        "ConSan MOI owner/epoch prologue requires RJ_CONSAN_MOI_OWNER_VGPR and "
        "RJ_CONSAN_MOI_EPOCH_VGPR");
    return;
  }
  if (operating_point.moi_owner_epoch_vgprs.complete() &&
      operating_point.moi_owner_epoch_vgprs->owner ==
          operating_point.moi_owner_epoch_vgprs->epoch) {
    result.errors.emplace_back("ConSan MOI owner and epoch VGPRs must be distinct");
    return;
  }
  if (request.moi_owner_source == ConSanMoiOwnerSource::HwId &&
      !operating_point.moi_owner_sgpr.base()) {
    result.errors.emplace_back("ConSan MOI hw_id owner source requires RJ_CONSAN_MOI_OWNER_SGPR");
    return;
  }
  if (result.program_inventory.kernels().empty()) {
    result.warnings.emplace_back("ConSan MOI owner/epoch prologue found no kernel descriptors");
    return;
  }

  const std::span<const uint8_t> active_bytes = result.active_bytes(bytes);
  AmdGpuCodeObject code_object(active_bytes.data(), active_bytes.size());
  CodeObjectPatcher patcher(code_object);
  if (patcher.text_bytes().empty()) {
    result.errors.emplace_back("ConSan MOI owner/epoch prologue found no .text section");
    return;
  }
  const uint32_t owner_required_sgpr_count =
      request.moi_owner_source == ConSanMoiOwnerSource::HwId
          ? static_cast<uint32_t>(*operating_point.moi_owner_sgpr.base()) + 1u
          : 0u;
  /// Per-kernel transaction fixed before any descriptor or text mutation.
  ///
  /// Descriptor-derived entry facts, resolved persistent registers, and
  /// preservation/layout choices travel together into emission. The mutable
  /// patching loop may choose among entry routes, but cannot reinterpret the
  /// kernel ABI or change the initialization contract.
  struct PlannedOwnerEpochPrologue {
    ConSanProgramContainer kernel{ConSanProgramContainerKind::Kernel};
    MoiOwnerEpochPrologueEmissionPlan emission;
    uint32_t required_vgpr_count = 0;
    uint16_t required_sgpr_count = 0;
  };
  std::vector<PlannedOwnerEpochPrologue> target_kernels;
  target_kernels.reserve(result.program_inventory.kernels().size());
  for (const ConSanProgramContainer &kernel : result.program_inventory.kernels()) {
    if (!kernel.has_text_range)
      continue;
    const bool owns_emitted_patch = find_moi_patch(result, [&](const ConSanPatchInfo &patch) {
                                      return kernel_owns_patch(kernel, patch);
                                    }) != nullptr;
    const bool owns_planned_site =
        std::ranges::any_of(result.resource_plans, [&](const ConSanCandidateResourcePlan &plan) {
          return std::ranges::find(plan.owner_kernel_ids, kernel.id) !=
                     plan.owner_kernel_ids.end() &&
                 plan.source != ConSanRegisterAllocationSource::Unsupported;
        });
    // Automatic persistent state exists only to serve emitted instrumentation.
    // Resource planning deliberately inventories more sites than selection may
    // admit (for example under max_patches or a kernel filter). Mutating every
    // descriptor/entry that merely owns one of those speculative plans can
    // perturb otherwise untouched kernels and, on real multi-kernel modules,
    // can make execution hang before the selected probe is reached.
    if ((operating_point.automatic_moi_persistent_vgprs ||
         operating_point.moi_persistent_sgprs.complete()) &&
        !owns_emitted_patch)
      continue;
    if (moi_has_runtime_hardware_dispatch_id(operating_point) && !owns_emitted_patch &&
        !owns_planned_site) {
      continue;
    }
    const auto active_kernel =
        std::ranges::find_if(code_object.kernels(), [&](const AmdGpuKernelInfo &candidate) {
          return candidate.name == kernel.name;
        });
    if (active_kernel == code_object.kernels().end()) {
      result.warnings.emplace_back(
          "ConSan MOI owner/epoch prologue could not resolve an active kernel descriptor");
      continue;
    }
    ConSanProgramContainer active = kernel;
    active.descriptor_file_offset = active_kernel->descriptor_file_offset;
    active.entry_text_offset = active_kernel->entry_text_offset;
    active.text_file_offset = active_kernel->text_file_offset;
    ConSanMoiOperatingPoint kernel_point = operating_point;
    const std::array<uint64_t, 1> kernel_owner = {kernel.descriptor_file_offset};
    if (!apply_moi_persistent_vgpr_assignment(kernel_point, result.moi_operating_point,
                                              kernel_owner)) {
      result.errors.emplace_back(
          "ConSan MOI owner/epoch prologue has no persistent assignment for kernel '" +
          kernel.name + "'");
      return;
    }
    // Its access/synchronization probes and entry initialization were already
    // emitted by the private-state path above. The remaining pass is solely
    // for owner components with a persistent VGPR tuple.
    if (kernel_point.automatic_moi_private_epoch)
      continue;
    if (!result.moi_operating_point.owner_transient_sgprs.empty() &&
        !apply_moi_transient_sgpr_assignment(request, kernel_point, result.moi_operating_point,
                                             kernel_owner)) {
      result.errors.emplace_back(
          "ConSan MOI owner/epoch prologue has no transient scalar assignment for kernel '" +
          kernel.name + "'");
      return;
    }
    const ConSanMoiDispatchIdCapture dispatch_capture = dispatch_id_capture(kernel_point);
    ConSanMoiOwnerEpochVgprSources owner_epoch_vgprs =
        moi_owner_epoch_vgpr_sources(kernel_point.moi_owner_epoch_vgprs);
    if (operating_point.moi_persistent_sgprs.complete()) {
      const auto scratch =
          std::ranges::find(prologue_scratch_assignments, kernel.descriptor_file_offset,
                            &ConSanMoiPrologueScratchVgprAssignment::descriptor_file_offset);
      if (scratch == prologue_scratch_assignments.end()) {
        result.errors.emplace_back(
            "ConSan MOI scalar owner/epoch prologue has no entry-local VGPR scratch for kernel '" +
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
        rollback_moi_dispatch_id_as_unsupported(result, "owner descriptor exceeds ELF bytes");
        return;
      }
      result.errors.emplace_back("ConSan MOI owner/epoch descriptor exceeds ELF bytes");
      return;
    }
    const KD &descriptor = *descriptor_value;
    const uint16_t owner_shift_bits = kernel_wavefront_size(arch, descriptor) == 32u ? 5u : 6u;
    std::optional<ConSanMoiDispatchIdPreloadPlan> dispatch_plan;
    if (dispatch_capture.present()) {
      dispatch_plan = moi_descriptor_dispatch_id_preload_plan(descriptor, arch);
      if (!dispatch_plan->supported() ||
          (dispatch_capture.sgpr() &&
           *dispatch_capture.sgpr() < dispatch_plan->required_sgpr_count)) {
        rollback_moi_dispatch_id_as_unsupported(
            result,
            dispatch_id_preload_rejection_reason(kernel.name, *dispatch_plan, dispatch_capture));
        return;
      }
    }
    std::optional<ConSanMoiWorkgroupShadowLayout> workgroup_shadow;
    const auto workitem_id_dimensions = moi_descriptor_workitem_id_dimensions(descriptor);
    if (!workitem_id_dimensions) {
      result.errors.emplace_back(
          "ConSan MOI owner/epoch prologue has invalid workitem-ID dimensions");
      return;
    }
    for_each_moi_patch(result, [&](const ConSanPatchInfo &patch) {
      if (!kernel_owns_patch(kernel, patch))
        return;
      if (!patch.workgroup_shadow)
        return;
      ConSanMoiWorkgroupShadowLayout candidate = *patch.workgroup_shadow;
      candidate.initialization_lanes =
          moi_workgroup_shadow_initialization_lanes(*target, active.required_workgroup_size);
      if (workgroup_shadow && *workgroup_shadow != candidate) {
        result.errors.emplace_back(
            "ConSan MOI owner/epoch prologue found incompatible workgroup-shadow layouts");
        return;
      }
      workgroup_shadow = candidate;
    });
    if (!result.errors.empty())
      return;
    const bool owns_inline_barrier =
        find_moi_patch(result, [&](const ConSanPatchInfo &patch) {
          return kernel_owns_patch(kernel, patch) &&
                 patch.kind == ConSanPatchKind::TrampolineMoiInlineEpochBarrier;
        }) != nullptr;
    const bool owns_non_barrier_instrumentation =
        find_moi_patch(result, [&](const ConSanPatchInfo &patch) {
          return kernel_owns_patch(kernel, patch) &&
                 patch.kind != ConSanPatchKind::TrampolineMoiInlineEpochBarrier;
        }) != nullptr;
    if (mode_policy.skip_unobserved_barrier_only_initialization &&
        kernel_point.automatic_moi_persistent_vgprs && owns_inline_barrier &&
        !owns_non_barrier_instrumentation) {
      result.warnings.emplace_back(
          "ConSan MOI skipped unobserved owner/epoch initialization for a barrier-only kernel");
      continue;
    }
    std::optional<ConSanMoiWorkgroupSources> workgroup_sources;
    const bool has_exact_workgroup_tuple =
        !kernel_point.moi_exact_workgroup_vgprs.empty() ||
        !kernel_point.moi_persistent_sgprs.exact_workgroup.empty();
    if (kernel_point.moi_workgroup_key_vgpr || kernel_point.moi_persistent_sgprs.workgroup_key ||
        has_exact_workgroup_tuple) {
      std::vector<std::string> source_errors;
      std::optional<uint16_t> full_payload_user_sgpr_count;
      if (has_exact_workgroup_tuple &&
          consan_arch_supports_kernarg_preload_overflow_recovery(arch)) {
        full_payload_user_sgpr_count =
            dispatch_plan ? dispatch_plan->expanded_user_sgpr_count
                          : static_cast<uint16_t>(moi_descriptor_user_sgpr_count(descriptor));
      }
      workgroup_sources = moi_descriptor_workgroup_sources(
          active_bytes, active.descriptor_file_offset, arch, source_errors,
          kernel.uses_cluster_workgroup_id, full_payload_user_sgpr_count);
      if (!workgroup_sources) {
        result.errors.insert(result.errors.end(), source_errors.begin(), source_errors.end());
        return;
      }
    }
    std::optional<ConSanMoiWorkgroupShadowLayout> prologue_workgroup_shadow = workgroup_shadow;
    if (prologue_workgroup_shadow) {
      prologue_workgroup_shadow->visible_evidence_sgpr =
          inline_shadow_visible_evidence_sgpr(project_inline_shadow_scalar_state(kernel_point));
    }
    MoiOwnerEpochPrologueEmissionPlan emission{
        .vgpr_state =
            {
                .owner_epoch = {*owner_epoch_vgprs.owner, *owner_epoch_vgprs.epoch},
                .workgroup_key = kernel_point.moi_workgroup_key_vgpr,
                .exact_workgroup = kernel_point.moi_exact_workgroup_vgprs,
                .owner_epoch_lifetime = consan_moi_owner_epoch_vgpr_lifetime(
                    kernel_point.moi_persistent_sgprs.complete(),
                    !result.moi_operating_point.owner_persistent_vgprs.empty()),
            },
        .owner_shift_bits = owner_shift_bits,
        .owner_source = request.moi_owner_source,
        .owner_sgpr = kernel_point.moi_owner_sgpr.base(),
        .one_based_owner_ids = mode_policy.one_based_owner_ids,
        .persistent_sgprs = kernel_point.moi_persistent_sgprs,
        .dispatch_plan = dispatch_plan,
        .dispatch_capture = dispatch_capture,
        .runtime_workgroup_selection_source = moi_runtime_workgroup_selection_source(),
        .return_pc_sgpr = kernel_point.moi_exec_save_sgpr,
        .runtime_sample_stride = request.moi_runtime_sample_stride,
        .runtime_sample_offset = request.moi_runtime_sample_offset,
        .runtime_report_dispatch_id = resources.moi_report_dispatch_id,
        .entry_scalar_backup = std::nullopt,
        .workgroup_shadow = std::move(prologue_workgroup_shadow),
        .has_quad_zero_tuple = kernel_point.automatic_moi_persistent_vgprs &&
                               moi_workgroup_shadow_preferred_zero_vgpr_count(*target) == 4u,
        .workgroup_sources = std::move(workgroup_sources),
    };
    std::optional<ConSanMoiEntryScalarBackup> entry_scalar_backup;
    const bool needs_branch_only_scalar_backup = kernel_point.moi_branch_only_spill.has_value();
    const bool needs_dynamic_stack_scalar_backup =
        kernel_point.moi_branch_only_spill &&
        kernel_point.moi_branch_only_spill->dynamic_stack_borrowed_sgpr.has_value();
    // A mode whose compact probe protects its borrowed scalar window only at
    // the access body can use that window earlier for runtime workgroup
    // selection. Its policy therefore requests an independent entry save
    // before any site-local spill exists.
    const bool needs_mode_runtime_scalar_backup =
        mode_policy.backup_compact_spill_for_runtime_sampling &&
        kernel_point.has_compact_moi_scalar_spill() && request.moi_runtime_sample_stride > 1u;
    const bool needs_full_entry_scalar_backup = needs_branch_only_scalar_backup ||
                                                needs_dynamic_stack_scalar_backup ||
                                                needs_mode_runtime_scalar_backup;
    const bool needs_automatic_owner_scalar_backup =
        kernel_point.moi_owner_sgpr.automatic() &&
        request.moi_owner_source == ConSanMoiOwnerSource::HwId &&
        kernel_point.moi_owner_sgpr.base().has_value();
    if (needs_full_entry_scalar_backup || needs_automatic_owner_scalar_backup) {
      const bool entry_backup_arch_supported = consan_is_capability_arch(arch);
      const bool has_scalar_spill_contract = kernel_point.has_moi_scalar_spill();
      if (!entry_backup_arch_supported || !kernel_point.moi_exec_save_sgpr ||
          !has_scalar_spill_contract ||
          (needs_dynamic_stack_scalar_backup && !kernel_point.moi_dynamic_stack_spill)) {
        if (needs_full_entry_scalar_backup) {
          result.errors.emplace_back("ConSan MOI entry scalar backup has no valid spill contract");
          return;
        }
      }
      const uint16_t sgpr_base = needs_full_entry_scalar_backup
                                     ? *kernel_point.moi_exec_save_sgpr
                                     : *kernel_point.moi_owner_sgpr.base();
      const uint16_t borrowed_sgpr_count =
          needs_full_entry_scalar_backup
              ? moi_exec_save_sgpr_count(resolve_moi_exec_save_requirement(
                                             request, resources, kernel_point, mode_semantics),
                                         arch)
              : 1u;
      const auto original_descriptor = read_kernel_descriptor(bytes, kernel.descriptor_file_offset);
      if (!original_descriptor) {
        result.errors.emplace_back(
            "ConSan MOI entry scalar backup descriptor exceeds original ELF bytes");
        return;
      }
      if (borrowed_sgpr_count == 0u) {
        result.errors.emplace_back("ConSan MOI entry scalar backup has an empty borrowed window");
        return;
      }
      const uint32_t guest_entry_sgpr_count =
          moi_descriptor_user_sgpr_count(*original_descriptor) +
          moi_descriptor_system_sgpr_count(*original_descriptor);
      const uint32_t backup_end = std::min<uint32_t>(
          static_cast<uint32_t>(sgpr_base) + borrowed_sgpr_count, guest_entry_sgpr_count);
      const uint16_t sgpr_count =
          sgpr_base < backup_end ? static_cast<uint16_t>(backup_end - sgpr_base) : 0u;

      // The hardware initializes only the descriptor-declared user and system
      // SGPR prefix. Values above that prefix are undefined at kernel entry;
      // restoring them later can overwrite guest values written after entry
      // if the prologue's VALU lane transfers are still retiring. Preserve
      // only the initialized intersection of the borrowed scalar window.
      if (sgpr_count != 0u && !moi_entry_scalar_backup_preserves_persistent_outputs(
                                  emission, kernel_point.moi_exec_save_sgprs_persistent,
                                  kernel_point.moi_owner_sgpr.automatic(),
                                  kernel_point.moi_scalar_spill_setup
                                      ? kernel_point.moi_scalar_spill_setup->visible_evidence_sgpr
                                      : std::nullopt,
                                  sgpr_base, sgpr_count)) {
        result.errors.emplace_back(
            "ConSan MOI entry scalar backup overlaps persistent output state");
        return;
      }
      if (sgpr_count > kernel_wavefront_size(arch, *original_descriptor)) {
        result.outcome = ConSanTransformOutcome::Unsupported;
        result.warnings.emplace_back(
            "ConSan MOI entry scalar window exceeds one wave-local VGPR carrier");
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
          if (!moi_owner_epoch_prologue_uses_vgpr(emission, static_cast<uint16_t>(candidate))) {
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
          if (!moi_owner_epoch_prologue_uses_vgpr(emission, static_cast<uint16_t>(candidate))) {
            backup_vgpr = static_cast<uint16_t>(candidate);
          }
        }
        if (!backup_vgpr) {
          result.outcome = ConSanTransformOutcome::Unsupported;
          result.warnings.emplace_back(has_live_accvgpr_bank
                                           ? "ConSan MOI entry scalar backup has no entry-local "
                                             "ordinary VGPR below the AccVGPR boundary"
                                           : "ConSan MOI entry scalar backup has no entry-local "
                                             "VGPR carrier");
          return;
        }
        entry_scalar_backup = ConSanMoiEntryScalarBackup{
            .vgpr = *backup_vgpr,
            .sgpr_base = sgpr_base,
            .sgpr_count = sgpr_count,
        };
      }
    }
    emission.entry_scalar_backup = entry_scalar_backup;
    uint32_t required_vgpr_count = moi_owner_epoch_prologue_required_vgpr_count(emission);
    if (entry_scalar_backup) {
      required_vgpr_count = std::max<uint32_t>(required_vgpr_count, entry_scalar_backup->vgpr + 1u);
    }
    uint32_t required_sgpr_count = owner_required_sgpr_count;
    if (emission.dispatch_capture.sgpr()) {
      required_sgpr_count = std::max<uint32_t>(
          required_sgpr_count,
          static_cast<uint32_t>(*emission.dispatch_capture.sgpr()) + kMoiDispatchStateSgprCount);
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
  MoiDescriptorVgprRequirements prologue_vgpr_requirements;
  MoiDescriptorSgprRequirements prologue_sgpr_requirements;
  for (const PlannedOwnerEpochPrologue &item : target_kernels) {
    if (item.emission.dispatch_plan && item.emission.dispatch_capture.present() &&
        !apply_moi_dispatch_id_descriptor(patcher, item.kernel.descriptor_file_offset,
                                          *item.emission.dispatch_plan,
                                          item.emission.dispatch_capture, arch, result.errors)) {
      return;
    }
    const ConSanProgramContainer *canonical =
        result.program_inventory.find_kernel_by_name(item.kernel.name);
    if (canonical == nullptr) {
      result.errors.emplace_back("ConSan MOI owner/epoch prologue lost its descriptor owner");
      return;
    }
    note_maximum_descriptor_extent(prologue_vgpr_requirements, canonical->descriptor_file_offset,
                                   static_cast<uint16_t>(item.required_vgpr_count));
    note_maximum_descriptor_extent(prologue_sgpr_requirements, canonical->descriptor_file_offset,
                                   item.required_sgpr_count);
  }
  if (!apply_moi_descriptor_requirements(patcher, code_object, result.program_inventory,
                                         prologue_vgpr_requirements, prologue_sgpr_requirements,
                                         MoiDescriptorPrivateRequirements{}, nullptr, nullptr, arch,
                                         "ConSan MOI owner/epoch prologue", result.errors))
    return;

  std::vector<ConSanTextFragment> fragments;
  fragments.reserve(target_kernels.size());
  for (const PlannedOwnerEpochPrologue &item : target_kernels) {
    const ConSanProgramContainer *canonical_kernel =
        result.program_inventory.find_kernel_by_name(item.kernel.name);
    if (canonical_kernel == nullptr) {
      result.errors.emplace_back("ConSan MOI owner/epoch prologue lost its canonical kernel owner");
      return;
    }
    const MoiOwnerEpochPrologueEmissionPlan &prologue_plan = item.emission;
    auto words = build_owner_epoch_prologue_words(prologue_plan, arch, result.errors);
    if (!words)
      return;

    ConSanPatchInfo info;
    info.kind = ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue;
    info.anchor_offset = canonical_kernel->entry_text_offset;
    info.original_size = 0u;
    info.moi_vgpr_state = prologue_plan.vgpr_state;
    info.persistent_sgpr_state = prologue_plan.persistent_sgprs;
    info.required_sgpr_count = item.required_sgpr_count;
    info.entry_scalar_backup = prologue_plan.entry_scalar_backup;
    info.workgroup_shadow = prologue_plan.workgroup_shadow;
    if (prologue_plan.dispatch_plan && prologue_plan.dispatch_capture.present())
      note_dispatch_id_patch_info(info, *prologue_plan.dispatch_plan,
                                  prologue_plan.dispatch_capture);
    info.owner_descriptor_file_offsets.push_back(canonical_kernel->descriptor_file_offset);
    fragments.push_back(ConSanTextFragment::entry_prefix(std::move(*words), std::move(info)));
  }

  if (fragments.empty()) {
    result.warnings.emplace_back("ConSan MOI owner/epoch prologue found no patchable kernels");
    return;
  }
  result.replacement = std::move(patcher).emit();
  (void)stage_consan_text_fragments(std::move(fragments), result);
}

} // namespace rocjitsu::consan_moi_impl
