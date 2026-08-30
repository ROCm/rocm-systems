// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi_record_planning.h"

#include "rocjitsu/code/patch/consan/consan_moi_placement_contracts.h"
#include "rocjitsu/code/patch/consan/consan_moi_probe_contracts.h"
#include "rocjitsu/code/patch/consan/consan_moi_report_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_runtime_workgroup_gate.h"
#include "rocjitsu/code/patch/consan/consan_moi_shared_lowering.h"

#include <algorithm>
#include <utility>

namespace rocjitsu::consan_moi_impl {

using consan_moi_detail::moi_report_dispatch_id_word_source;

[[nodiscard]] bool moi_transient_sgpr_assignment_uses_borrowed_record_replay_entry(
    const ConSanRequest &request, const ConSanMoiOperatingPoint &allocation,
    std::span<const uint64_t> owner_descriptor_offsets) {
  return request.moi_engine == ConSanMoiEngine::RecordReplay && !owner_descriptor_offsets.empty() &&
         std::ranges::all_of(owner_descriptor_offsets, [&](uint64_t descriptor_offset) {
           const auto assignment =
               std::ranges::find(allocation.owner_transient_sgprs, descriptor_offset,
                                 &ConSanMoiTransientSgprAssignment::descriptor_file_offset);
           return assignment != allocation.owner_transient_sgprs.end() &&
                  assignment->branch_only_scalar_spill && assignment->indirect_pc_sgpr &&
                  assignment->indirect_scc_sgpr && !assignment->dispatch_key_sgpr &&
                  !assignment->call_return_sgpr;
         });
}

[[nodiscard]] bool apply_record_replay_entry_workgroup_assignment(
    const ConSanRequest &request, ConSanMoiOperatingPoint &point,
    const ConSanMoiOperatingPoint &allocation, std::span<const uint64_t> owner_descriptor_offsets) {
  if (!consan_moi_detail::record_replay_entry_workgroup_capture_is_unambiguous(point))
    return false;
  if (!consan_moi_detail::record_replay_requires_entry_workgroup_capture(request.moi_engine) ||
      consan_moi_detail::record_replay_has_entry_workgroup_capture(point)) {
    return true;
  }
  return apply_moi_persistent_vgpr_assignment(point, allocation, owner_descriptor_offsets) &&
         consan_moi_detail::record_replay_has_entry_workgroup_capture(point) &&
         consan_moi_detail::record_replay_entry_workgroup_capture_is_unambiguous(point);
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
    const ConSanMoiOperatingPoint &point, uint16_t scratch_vgpr, rj_code_arch_t arch) {
  const auto workgroup_sources =
      record_replay_persistent_workgroup_sources(request.moi_engine, point);
  const auto special_state = moi_special_state_sgprs(request, point);
  if (!point.moi_exec_save_sgpr || !bound_resources.moi_report_buffer_address ||
      !workgroup_sources || !special_state) {
    return std::nullopt;
  }
  std::optional<MoiRuntimeWorkgroupGatePlan> runtime_workgroup_gate;
  if (request.moi_runtime_sample_stride > 1u && !point.has_compact_moi_scalar_spill()) {
    runtime_workgroup_gate =
        plan_moi_runtime_workgroup_gate(request, bound_resources, point, *workgroup_sources, arch);
  }
  return MoiRecordEventEmissionPlan{
      .scratch_vgpr = scratch_vgpr,
      .moi_exec_save_sgpr = point.moi_exec_save_sgpr,
      .moi_owner_vgpr = point.moi_owner_vgpr,
      .moi_epoch_vgpr = point.moi_epoch_vgpr,
      .moi_workgroup_key_vgpr = point.moi_workgroup_key_vgpr,
      .moi_dispatch_id_vgpr = point.moi_dispatch_id_vgpr,
      .moi_record_replay_workgroup_vgprs = point.moi_record_replay_workgroup_vgprs,
      .moi_persistent_sgprs = point.moi_persistent_sgprs,
      .moi_report_buffer_address = bound_resources.moi_report_buffer_address,
      .automatic_moi_record_replay_sgpr_spill = point.has_compact_moi_scalar_spill(),
      .moi_record_replay_dispatch_key_sgpr = point.moi_record_replay_dispatch_key_sgpr,
      .moi_record_replay_call_return_sgpr = point.moi_record_replay_call_return_sgpr,
      .workgroup_sources = *workgroup_sources,
      .special_state = *special_state,
      .dispatch_id_sources = {moi_report_dispatch_id_word_source(point, bound_resources,
                                                                 /*high_word=*/false),
                              moi_report_dispatch_id_word_source(point, bound_resources,
                                                                 /*high_word=*/true)},
      .runtime_workgroup_gate = runtime_workgroup_gate,
      .indirect_jump = moi_indirect_jump_sgprs(request, point),
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
[[nodiscard]] std::optional<MoiPlannedRecordEvent>
plan_moi_record_event(std::span<const uint8_t> bytes, const ProgramInventory &inventory,
                      std::vector<std::string> &warnings, const ResolvedMoiScratchPlan &resources,
                      const ConSanRequest &request, const BoundRuntimeResources &bound_resources,
                      const ConSanMoiOperatingPoint &base_point,
                      const ConSanMoiOperatingPoint &allocation, MoiSpillManagers &spill_managers,
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

  std::optional<MoiPrivateEpochLayout> private_layout;
  if (event_point.automatic_moi_private_epoch) {
    private_layout = build_moi_private_epoch_layout(inventory, resources, arch, warnings,
                                                    /*include_owner=*/request.moi_owner_source ==
                                                        ConSanMoiOwnerSource::WorkitemId,
                                                    /*include_workgroup_key=*/false,
                                                    /*include_record_replay_workgroup=*/true);
    if (!private_layout)
      return std::nullopt;
    event_point.moi_record_replay_workgroup_private_offsets =
        private_layout->record_replay_workgroup_offsets;
  }
  if (!apply_record_replay_entry_workgroup_assignment(request, event_point, allocation,
                                                      resources.owner_descriptor_file_offsets)) {
    warnings.emplace_back(std::string(warning_context) +
                          " found no common entry workgroup assignment");
    return std::nullopt;
  }

  auto emission = resolve_moi_record_event_emission_plan(request, bound_resources, event_point,
                                                         resources.base, arch);
  if (!emission) {
    warnings.emplace_back(std::string(warning_context) +
                          " has an incomplete Record/Replay emission plan");
    return std::nullopt;
  }

  std::optional<MoiWorkitemOwnerDerivationPlan> derived_owner;
  if (!event_point.moi_owner_vgpr && !event_point.moi_persistent_sgprs.owner) {
    if (moi_record_uses_private_owner(request, event_point)) {
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
  auto probe = plan_moi_probe_resources(inventory, resources, request, bound_resources, event_point,
                                        spill_managers, arch, std::move(private_layout),
                                        event_point.has_compact_moi_scalar_spill(), warnings,
                                        active_private_segment_size);
  if (!probe)
    return std::nullopt;

  MoiDescriptorSgprRequirements scalar_requirements;
  note_moi_sgpr_requirements(scalar_requirements, resources, request, bound_resources, event_point,
                             arch);
  uint16_t required_sgpr_count = 0u;
  for (const auto &[descriptor, count] : scalar_requirements) {
    (void)descriptor;
    required_sgpr_count = std::max(required_sgpr_count, count);
  }
  emission->required_sgpr_count = required_sgpr_count;
  return MoiPlannedRecordEvent{std::move(*probe), std::move(*emission), derived_owner};
}

} // namespace rocjitsu::consan_moi_impl
