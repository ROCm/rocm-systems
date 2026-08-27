/// Common preservation state for one MOI probe whose body temporarily owns a
/// scratch register window.
///
/// Record/Replay and Sampled attach different evidence semantics to their
/// probes, but both must preserve the same guest resources around a selected
/// body. This type is the complete placement-to-emission handoff for that
/// shared responsibility: resolved scratch, optional VGPR and SGPR spill
/// sequences, optional private identity state, and the resulting private
/// segment extent. It deliberately excludes site identity, address semantics,
/// event kind, routing, and evidence indices.
struct MoiPlannedProbeResources {
  ResolvedMoiScratchPlan resources;
  std::optional<VgprSpillSequence> spill;
  std::optional<SgprSpillSequence> scalar_spill;
  std::optional<MoiPrivateEpochLayout> private_layout;
  uint32_t required_private_bytes = 0;
};

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
plan_moi_probe_resources(ConSanTransformArtifacts &result, ResolvedMoiScratchPlan resources,
                         const ConSanOptions &options, MoiSpillManagers &spill_managers,
                         rj_code_arch_t arch, std::optional<MoiPrivateEpochLayout> private_layout,
                         bool scalar_spill_required,
                         std::optional<uint32_t> active_private_segment_size = std::nullopt) {
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
      moi_scalar_spill_requires_dynamic_vgpr_frame(result, resources, options)) {
    spill = build_moi_spill_sequence(result, resources, options, spill_managers, arch,
                                     result.warnings, private_layout_base);
    if (!spill)
      return std::nullopt;
  }
  std::optional<SgprSpillSequence> scalar_spill;
  if (scalar_spill_required) {
    scalar_spill = build_inline_sgpr_spill_sequence(result, resources, options, spill_managers,
                                                    arch, result.warnings, private_layout_base,
                                                    spill ? &*spill : nullptr);
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
  for (uint64_t descriptor_offset : probe.resources.owner_descriptor_file_offsets) {
    auto [it, inserted] = requirements.emplace(descriptor_offset, probe.required_private_bytes);
    if (!inserted)
      it->second = std::max(it->second, probe.required_private_bytes);
  }
}
