// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi_probe_planning.h"

#include "rocjitsu/code/patch/consan/consan_moi_access_apply.h"
#include "rocjitsu/code/patch/consan/consan_moi_shared_lowering.h"

#include <algorithm>
#include <utility>

namespace rocjitsu::consan_moi_impl {

/// Build the guest-state preservation plan shared by MOI probe engines.
///
/// The caller has already selected scratch registers and, when needed, a
/// private epoch/owner layout. This function allocates the compatible VGPR
/// and scalar spill sequences and computes the one descriptor extent that
/// covers every fixed private allocation. An explicit base is used only when
/// an earlier transform has already grown the active descriptor beyond the
/// pristine resource plan. `scalar_spill_required` is an engine decision, not
/// inferred here: Record/Replay and Sampled currently use their record-spill
/// policy, while InlineShadow uses a distinct policy.
[[nodiscard]] std::optional<MoiPlannedProbeResources>
plan_moi_probe_resources(const ProgramInventory &inventory, ResolvedMoiScratchPlan resources,
                         const ConSanRequest &request, const BoundRuntimeResources &bound_resources,
                         const ConSanMoiOperatingPoint &point, MoiSpillManagers &spill_managers,
                         rj_code_arch_t arch, std::optional<MoiPrivateEpochLayout> private_layout,
                         bool scalar_spill_required, std::vector<std::string> &warnings,
                         std::optional<uint32_t> active_private_segment_size) {
  if (active_private_segment_size)
    resources.original_private_segment_size = *active_private_segment_size;
  std::optional<uint32_t> private_layout_base;
  // Keep this as a separate assignment: GCC 13 otherwise diagnoses the
  // equivalent conditional optional construction as maybe-uninitialized.
  if (private_layout)
    private_layout_base = private_layout->ephemeral_base;
  else if (active_private_segment_size)
    private_layout_base = resources.original_private_segment_size;

  std::optional<VgprSpillSequence> spill;
  if (resources.source == ConSanRegisterAllocationSource::SpillRequired ||
      moi_scalar_spill_requires_dynamic_vgpr_frame(inventory, resources, point)) {
    spill = build_moi_spill_sequence(inventory, resources, request, bound_resources, point,
                                     spill_managers, arch, warnings, private_layout_base);
    if (!spill)
      return std::nullopt;
  }
  std::optional<SgprSpillSequence> scalar_spill;
  if (scalar_spill_required) {
    scalar_spill = build_moi_sgpr_spill_sequence(inventory, resources, request, bound_resources,
                                                 point, spill_managers, arch, warnings,
                                                 private_layout_base, spill ? &*spill : nullptr);
    if (!scalar_spill)
      return std::nullopt;
  }
  const uint32_t required_private_bytes =
      std::max({private_layout ? private_layout->ephemeral_base : 0u,
                spill ? spill->total_private_bytes : 0u,
                scalar_spill ? scalar_spill->total_private_bytes : 0u});
  return MoiPlannedProbeResources{.resources = std::move(resources),
                                  .spill = std::move(spill),
                                  .scalar_spill = std::move(scalar_spill),
                                  .private_layout = std::move(private_layout),
                                  .required_private_bytes = required_private_bytes};
}

/// Accumulate the private-memory descriptor requirement imposed by one
/// planned probe's preservation state.
void note_moi_probe_private_requirements(MoiDescriptorPrivateRequirements &requirements,
                                         const MoiPlannedProbeResources &probe) {
  if (probe.spill)
    note_spill_descriptor_requirements(requirements, probe.resources, *probe.spill);
  if (probe.required_private_bytes == 0u)
    return;
  note_maximum_descriptor_extents(requirements, probe.resources.owner_descriptor_file_offsets,
                                  probe.required_private_bytes);
}

/// Publish the mechanical resource facts of one successfully emitted probe.
///
/// The caller owns the patch's semantic kind, placement, covered event, and
/// engine-specific evidence fields. This function owns the common telemetry
/// derived from the preservation plan: scratch and kernel owners, every
/// private identity offset, the complete fixed private extent, and VGPR spill
/// metadata including a dynamic-stack addend. Keeping this projection beside
/// the plan prevents engines from publishing different metadata for the same
/// preservation transaction.
void note_moi_probe_patch_info(ConSanPatchAbiEffects &effects,
                               const MoiPlannedProbeResources &probe) {
  effects.scratch_vgpr = probe.resources.base;
  effects.owner_descriptor_file_offsets = probe.resources.owner_descriptor_file_offsets;
  if (probe.private_layout) {
    effects.persistent_epoch_private_offset = probe.private_layout->epoch_offset;
    effects.persistent_owner_private_offset = probe.private_layout->owner_offset;
    effects.persistent_workgroup_key_private_offset = probe.private_layout->workgroup_key_offset;
    effects.persistent_dispatch_id_private_offset = probe.private_layout->dispatch_id_offset;
    effects.persistent_record_replay_workgroup_private_offsets =
        probe.private_layout->record_replay_workgroup_offsets;
    effects.persistent_private_state_end = probe.private_layout->ephemeral_base;
  }
  effects.required_private_segment_size = probe.required_private_bytes;
  if (probe.spill) {
    effects.spilled_vgpr_count = probe.spill->vgpr_count;
    note_dynamic_stack_private_requirement(effects, &*probe.spill);
  }
}

} // namespace rocjitsu::consan_moi_impl
