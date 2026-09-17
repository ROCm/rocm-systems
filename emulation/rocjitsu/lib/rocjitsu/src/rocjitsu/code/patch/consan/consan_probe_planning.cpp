// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_probe_planning.h"

#include "rocjitsu/code/patch/consan/consan_access_apply.h"
#include "rocjitsu/code/patch/consan/consan_shared_lowering.h"

#include <algorithm>
#include <utility>

namespace rocjitsu::consan::detail {

/// Build the guest-state preservation plan for ConSan probes.
///
/// The caller has already selected scratch registers and, when needed, a
/// private epoch/owner layout. This function allocates the compatible VGPR
/// and scalar spill sequences and computes the one descriptor extent that
/// covers every fixed private allocation. An explicit base is used only when
/// an earlier transform has already grown the active descriptor beyond the
/// pristine resource plan. `scalar_spill_required` comes from the selected
/// preservation policy.
[[nodiscard]] std::optional<PlannedProbeResources>
plan_probe_resources(const ProgramInventory &inventory, ResolvedScratchPlan resources,
                     const Request &request, const BoundRuntimeResources &bound_resources,
                     const OperatingPoint &point, SpillManagers &spill_managers,
                     rj_code_arch_t arch, std::optional<PrivateStateLayout> private_layout,
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
  if (resources.source == RegisterAllocationSource::SpillRequired ||
      scalar_spill_requires_dynamic_vgpr_frame(inventory, resources, point)) {
    spill = build_spill_sequence(inventory, resources, request, bound_resources, point,
                                 spill_managers, arch, warnings, private_layout_base);
    if (!spill)
      return std::nullopt;
  }
  std::optional<SgprSpillSequence> scalar_spill;
  if (scalar_spill_required) {
    scalar_spill = build_sgpr_spill_sequence(inventory, resources, request, bound_resources, point,
                                             spill_managers, arch, warnings, private_layout_base,
                                             spill ? &*spill : nullptr);
    if (!scalar_spill)
      return std::nullopt;
  }
  const uint32_t required_private_bytes =
      std::max({private_layout ? private_layout->ephemeral_base : 0u,
                spill ? spill->total_private_bytes : 0u,
                scalar_spill ? scalar_spill->total_private_bytes : 0u});
  return PlannedProbeResources{.resources = std::move(resources),
                               .spill = std::move(spill),
                               .scalar_spill = std::move(scalar_spill),
                               .private_layout = std::move(private_layout),
                               .required_private_bytes = required_private_bytes};
}

/// Accumulate the private-memory descriptor requirement imposed by one
/// planned probe's preservation state.
void note_probe_private_requirements(DescriptorPrivateRequirements &requirements,
                                     const PlannedProbeResources &probe) {
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
/// evidence fields. This function owns the telemetry
/// derived from the preservation plan: scratch and kernel owners, the complete
/// private identity layout and extent, and VGPR spill metadata including a
/// dynamic-stack addend. Keeping this projection beside the plan ensures that
/// access and synchronization probes publish consistent preservation metadata.
void note_probe_patch_info(PatchAbiEffects &effects, const PlannedProbeResources &probe) {
  effects.scratch_vgpr = probe.resources.base;
  effects.owner_descriptor_file_offsets = probe.resources.owner_descriptor_file_offsets;
  effects.private_state_layout = probe.private_layout;
  effects.required_private_segment_size = probe.required_private_bytes;
  if (probe.spill) {
    effects.spilled_vgpr_count = probe.spill->vgpr_count;
    note_dynamic_stack_private_requirement(effects, &*probe.spill);
  }
}

} // namespace rocjitsu::consan::detail
