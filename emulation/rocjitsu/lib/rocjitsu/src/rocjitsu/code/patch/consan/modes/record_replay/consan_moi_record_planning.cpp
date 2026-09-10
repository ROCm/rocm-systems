// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/modes/record_replay/consan_moi_record_planning.h"

#include "rocjitsu/code/patch/consan/consan_capability_contract.h"
#include "rocjitsu/code/patch/consan/consan_moi_mode_planning.h"
#include "rocjitsu/code/patch/consan/consan_moi_placement_contracts.h"
#include "rocjitsu/code/patch/consan/consan_moi_probe_contracts.h"
#include "rocjitsu/code/patch/consan/consan_moi_report_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_runtime_workgroup_gate.h"
#include "rocjitsu/code/patch/consan/consan_moi_shared_lowering.h"

#include <algorithm>
#include <utility>

namespace rocjitsu::consan_moi_impl {

using consan_moi_detail::moi_bound_dispatch_id_sources;

[[nodiscard]] bool
apply_record_replay_entry_workgroup_assignment(ConSanMoiOperatingPoint &point,
                                               MoiOwnerAssignments assignments,
                                               std::span<const uint64_t> owner_descriptor_offsets,
                                               const ConSanMoiPrivateStateLayout *private_layout) {
  const ConSanMoiPersistentWorkgroupPrivateOffsets *private_offsets =
      private_layout ? &private_layout->exact_workgroup_offsets : nullptr;
  if (!consan_moi_detail::moi_exact_entry_workgroup_capture_is_unambiguous(point, private_offsets))
    return false;
  if (consan_moi_detail::moi_has_exact_entry_workgroup_capture(point, private_offsets)) {
    return true;
  }
  return apply_moi_persistent_vgpr_assignment(point, assignments, owner_descriptor_offsets) &&
         consan_moi_detail::moi_has_exact_entry_workgroup_capture(point, private_offsets) &&
         consan_moi_detail::moi_exact_entry_workgroup_capture_is_unambiguous(point,
                                                                             private_offsets);
}

void note_moi_sgpr_requirements(MoiDescriptorSgprRequirements &requirements,
                                const ResolvedMoiScratchPlan &resources,
                                const MoiRecordEventEmissionPlan &plan, rj_code_arch_t /*arch*/) {
  note_maximum_descriptor_extents(requirements, resources.owner_descriptor_file_offsets,
                                  plan.required_sgpr_count);
}

/// Snapshot the already-resolved Record/Replay lowering state needed by a
/// native event emitter. Owner-local assignment remains a planning concern;
/// this function performs no mutation and cannot choose a different binding.
[[nodiscard]] std::optional<MoiRecordEventEmissionPlan> resolve_moi_record_event_emission_plan(
    const ConSanRequest &request, const BoundRuntimeResources &bound_resources,
    const ConSanMoiOperatingPoint &point, const MoiScalarAbiPlan &scalar_abi, uint16_t scratch_vgpr,
    rj_code_arch_t arch, const ConSanMoiPrivateStateLayout *private_layout) {
  const ConSanMoiPersistentWorkgroupPrivateOffsets *private_offsets =
      private_layout ? &private_layout->exact_workgroup_offsets : nullptr;
  const auto workgroup_sources = moi_exact_entry_workgroup_sources(point, private_offsets);
  if (!point.moi_exec_save_sgpr || !bound_resources.moi_report_buffer_address ||
      !workgroup_sources || !scalar_abi.special_state) {
    return std::nullopt;
  }
  std::optional<MoiRuntimeWorkgroupGatePlan> runtime_workgroup_gate;
  if (request.moi_runtime_sample_stride > 1u && !point.has_compact_moi_scalar_spill()) {
    if (const ConSanTargetProfile *target = consan_target_profile(arch)) {
      runtime_workgroup_gate = plan_moi_runtime_workgroup_gate(
          {.exec_save_sgpr = *point.moi_exec_save_sgpr,
           .sample_stride = request.moi_runtime_sample_stride,
           .sample_offset = request.moi_runtime_sample_offset,
           .flavor = MoiRuntimeWorkgroupGatePlan::Flavor::RecordReplay,
           .dispatch_id_sgpr = point.moi_dispatch_identity.sgpr(),
           .literal_dispatch_id = bound_resources.moi_report_dispatch_id,
           .direct_call_form = target->direct_call_form},
          *workgroup_sources);
    }
  }
  return MoiRecordEventEmissionPlan{
      .scratch_vgpr = scratch_vgpr,
      .moi_exec_save_sgpr = point.moi_exec_save_sgpr,
      .moi_owner_epoch_vgprs = point.moi_owner_epoch_vgprs,
      .moi_workgroup_key_vgpr = point.moi_workgroup_key_vgpr,
      .moi_dispatch_id_vgpr = point.moi_dispatch_identity.vgpr(),
      .moi_exact_workgroup_vgprs = point.moi_exact_workgroup_vgprs,
      .moi_persistent_sgprs = point.moi_persistent_sgprs,
      .moi_report_buffer_address = bound_resources.moi_report_buffer_address,
      .workgroup_sources = *workgroup_sources,
      .special_state = *scalar_abi.special_state,
      .dispatch_id_sources = moi_bound_dispatch_id_sources(
          {point, bound_resources,
           private_layout ? private_layout->dispatch_id_offset : std::nullopt}),
      .runtime_workgroup_gate = runtime_workgroup_gate,
      .derived_owner = std::nullopt,
      .selectable_vgpr_bank_save_sgpr = std::nullopt,
  };
}

/// Resolve the resource state shared by Record/Replay synchronization events.
///
/// The caller remains responsible for discovering the event, choosing its
/// scratch plan, and validating any event-specific address or relocation.
/// This function owns the common transaction after scratch selection:
/// owner-local register assignments, optional private identity state, and the
/// VGPR/SGPR preservation sequences needed by emission. Failure rejects only
/// the current event and appends the same contextual warning used by its
/// event-specific planner.
[[nodiscard]] std::optional<MoiPlannedRecordEvent> plan_moi_record_event(
    std::span<const uint8_t> bytes, const ProgramInventory &inventory,
    std::vector<std::string> &warnings, const ResolvedMoiScratchPlan &resources,
    const ConSanRequest &request, const BoundRuntimeResources &bound_resources,
    const ConSanMoiOperatingPoint &base_point, const ConSanMoiOperatingPoint &allocation,
    const MoiObjectModeSemantics &mode_semantics, MoiSpillManagers &spill_managers,
    rj_code_arch_t arch, std::string_view warning_context,
    std::optional<uint32_t> active_private_segment_size) {
  ConSanMoiOperatingPoint event_point = base_point;
  if (!apply_moi_transient_sgpr_assignment(request, event_point, allocation,
                                           resources.owner_descriptor_file_offsets)) {
    warnings.emplace_back(std::string(warning_context) +
                          " found no common transient SGPR assignment");
    return std::nullopt;
  }
  if (!apply_moi_persistent_vgpr_assignment(event_point, allocation,
                                            resources.owner_descriptor_file_offsets)) {
    warnings.emplace_back(std::string(warning_context) + " found no common persistent assignment");
    return std::nullopt;
  }

  std::optional<ConSanMoiPrivateStateLayout> private_layout;
  if (event_point.automatic_moi_private_epoch) {
    private_layout = build_moi_private_state_layout(
        inventory, resources, arch, warnings,
        {.owner = request.moi_owner_source == ConSanMoiOwnerSource::WorkitemId,
         .exact_workgroup = !consan_moi_detail::moi_has_exact_entry_workgroup_capture(event_point),
         .dispatch_id = event_point.moi_dispatch_identity.private_fallback()});
    if (!private_layout)
      return std::nullopt;
  }
  if (!apply_record_replay_entry_workgroup_assignment(
          event_point, allocation, resources.owner_descriptor_file_offsets,
          private_layout ? &*private_layout : nullptr)) {
    warnings.emplace_back(std::string(warning_context) +
                          " found no common entry workgroup assignment");
    return std::nullopt;
  }

  const MoiScalarAbiPlan scalar_abi =
      plan_moi_scalar_abi(request.moi_engine, moi_scalar_preservation_state(event_point));
  auto emission = resolve_moi_record_event_emission_plan(
      request, bound_resources, event_point, scalar_abi, resources.base, arch,
      private_layout ? &*private_layout : nullptr);
  if (!emission) {
    warnings.emplace_back(std::string(warning_context) +
                          " has an incomplete Record/Replay emission plan");
    return std::nullopt;
  }

  std::optional<MoiWorkitemOwnerDerivationPlan> derived_owner;
  if (!event_point.moi_owner_epoch_vgprs.owner() && !event_point.moi_persistent_sgprs.owner()) {
    if (event_point.moi_initialize_owner_epoch && event_point.automatic_moi_private_epoch &&
        request.moi_owner_source == ConSanMoiOwnerSource::WorkitemId) {
      ResolvedMoiScratchPlan owner_resources = resources;
      if (active_private_segment_size)
        owner_resources.original_private_segment_size = *active_private_segment_size;
      derived_owner = resolve_moi_private_workitem_owner(bytes, owner_resources, *private_layout,
                                                         arch, warnings);
    } else {
      const auto shift = common_moi_workitem_owner_shift(bytes, resources, arch, warnings);
      if (shift) {
        derived_owner = MoiWorkitemOwnerDerivationPlan{
            .entry_workitem_x_private_offset = std::nullopt,
            .wave_size_shift = *shift,
        };
      }
    }
    if (!derived_owner)
      return std::nullopt;
  }
  emission->derived_owner = std::move(derived_owner);
  if (consan_arch_has_selectable_vgpr_bank(arch)) {
    const uint16_t count = moi_exec_save_sgpr_count(
        resolve_moi_exec_save_requirement(request, bound_resources, event_point, mode_semantics),
        arch);
    if (count == 0u) {
      warnings.emplace_back(std::string(warning_context) +
                            " has no scalar slot for selectable-VGPR-bank preservation");
      return std::nullopt;
    }
    emission->selectable_vgpr_bank_save_sgpr =
        static_cast<uint16_t>(*event_point.moi_exec_save_sgpr + count - 1u);
  }
  auto probe = plan_moi_probe_resources(
      inventory, resources, request, bound_resources, event_point, mode_semantics, spill_managers,
      arch, std::move(private_layout), event_point.has_compact_moi_scalar_spill(), warnings,
      active_private_segment_size);
  if (!probe)
    return std::nullopt;

  MoiDescriptorSgprRequirements scalar_requirements;
  note_moi_sgpr_requirements(scalar_requirements, resources, request, bound_resources, event_point,
                             mode_semantics, arch);
  uint16_t required_sgpr_count = 0u;
  for (const auto &[descriptor, count] : scalar_requirements) {
    (void)descriptor;
    required_sgpr_count = std::max(required_sgpr_count, count);
  }
  emission->required_sgpr_count = required_sgpr_count;
  return MoiPlannedRecordEvent{std::move(*probe), std::move(*emission)};
}

} // namespace rocjitsu::consan_moi_impl
