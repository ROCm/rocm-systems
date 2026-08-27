/// Common placement-to-emission handoff for one Record/Replay synchronization
/// event.
///
/// An atomic, barrier, or fence has different semantic evidence and native
/// emission, but all three use the same resolved scratch window, private epoch
/// and owner representation, and register-preservation plan. Keeping that
/// transaction in one type prevents the event-specific planners from
/// acquiring subtly different rules for the same Record/Replay resources.
struct MoiPlannedRecordEvent {
  ResolvedMoiScratchPlan resources;
  std::optional<VgprSpillSequence> spill;
  std::optional<SgprSpillSequence> scalar_spill;
  std::optional<MoiPrivateRecordOwnerPlan> private_owner;
  std::optional<MoiPrivateEpochLayout> private_layout;
};

/// Bind one planned Record/Replay event to the still-mutable options accepted
/// by the native emitters.
///
/// This is the single remaining compatibility boundary between the typed
/// event plan and emitters that have not yet replaced `ConSanOptions` with a
/// narrow lowering input. It selects the event's scratch window, private
/// workgroup representation, transient scalar assignment, and entry-captured
/// workgroup tuple. A false result means a previously proved owner-local
/// assignment disappeared and callers must abort rather than emit with a
/// different register interpretation.
[[nodiscard]] bool bind_moi_record_event_options(ConSanOptions &event_options,
                                                 const MoiPlannedRecordEvent &event) {
  event_options.scratch_vgpr = event.resources.base;
  if (event.private_layout) {
    event_options.moi_record_replay_workgroup_private_offsets =
        event.private_layout->record_replay_workgroup_offsets;
  }
  return apply_moi_transient_sgpr_assignment(event_options,
                                             event.resources.owner_descriptor_file_offsets) &&
         apply_record_replay_entry_workgroup_assignment(
             event_options, event.resources.owner_descriptor_file_offsets);
}

/// Accumulate the private-memory descriptor requirement imposed by one
/// planned Record/Replay event.
///
/// VGPR and scalar spills and persistent private identity share one kernel
/// private segment. Their allocations are already non-overlapping; the
/// descriptor therefore needs the maximum end offset, not three independent
/// mutations. Dynamic-stack VGPR spills additionally carry their own loader
/// requirement through `note_spill_descriptor_requirements`.
void note_moi_record_event_private_requirements(MoiDescriptorPrivateRequirements &requirements,
                                                const MoiPlannedRecordEvent &event) {
  if (event.spill)
    note_spill_descriptor_requirements(requirements, event.resources, *event.spill);
  const uint32_t required_private_bytes =
      std::max(event.scalar_spill ? event.scalar_spill->total_private_bytes : 0u,
               event.private_layout ? event.private_layout->ephemeral_base : 0u);
  if (required_private_bytes == 0u)
    return;
  for (uint64_t descriptor_offset : event.resources.owner_descriptor_file_offsets) {
    auto [it, inserted] = requirements.emplace(descriptor_offset, required_private_bytes);
    if (!inserted)
      it->second = std::max(it->second, required_private_bytes);
  }
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
plan_moi_record_event(std::span<const uint8_t> bytes, ConSanTransformArtifacts &result,
                      const ResolvedMoiScratchPlan &resources, const ConSanOptions &options,
                      MoiSpillManagers &spill_managers, rj_code_arch_t arch,
                      std::string_view warning_context,
                      std::optional<uint32_t> active_private_segment_size = std::nullopt) {
  ConSanOptions event_options = options;
  event_options.scratch_vgpr = resources.base;
  if (!apply_moi_transient_sgpr_assignment(event_options,
                                           resources.owner_descriptor_file_offsets)) {
    result.warnings.emplace_back(std::string(warning_context) +
                                 " found no common transient SGPR assignment");
    return std::nullopt;
  }
  if (!apply_moi_persistent_vgpr_assignment(event_options,
                                            resources.owner_descriptor_file_offsets)) {
    result.warnings.emplace_back(std::string(warning_context) +
                                 " found no common persistent assignment");
    return std::nullopt;
  }

  std::optional<MoiPrivateEpochLayout> private_layout;
  if (event_options.automatic_moi_private_epoch) {
    private_layout = build_moi_private_epoch_layout(
        result, resources, arch, result.warnings,
        /*include_owner=*/event_options.moi_owner_source == ConSanMoiOwnerSource::WorkitemId,
        /*include_workgroup_key=*/false,
        /*include_record_replay_workgroup=*/true);
    if (!private_layout)
      return std::nullopt;
    event_options.moi_record_replay_workgroup_private_offsets =
        private_layout->record_replay_workgroup_offsets;
  }
  if (!apply_record_replay_entry_workgroup_assignment(event_options,
                                                      resources.owner_descriptor_file_offsets)) {
    result.warnings.emplace_back(std::string(warning_context) +
                                 " found no common entry workgroup assignment");
    return std::nullopt;
  }

  ResolvedMoiScratchPlan active_resources = resources;
  if (active_private_segment_size)
    active_resources.original_private_segment_size = *active_private_segment_size;
  std::optional<MoiPrivateRecordOwnerPlan> private_owner;
  if (moi_record_uses_private_owner(event_options)) {
    private_owner =
        resolve_moi_private_record_owner(bytes, active_resources, *private_layout, result.warnings);
    if (!private_owner)
      return std::nullopt;
  }

  std::optional<uint32_t> private_layout_base;
  // Keep this as a separate assignment: GCC 13 otherwise diagnoses the
  // equivalent conditional optional construction as maybe-uninitialized.
  if (private_layout)
    private_layout_base = private_layout->ephemeral_base;
  else if (active_private_segment_size)
    private_layout_base = active_resources.original_private_segment_size;
  std::optional<VgprSpillSequence> spill;
  if (active_resources.source == ConSanRegisterAllocationSource::SpillRequired ||
      moi_scalar_spill_requires_dynamic_vgpr_frame(result, active_resources, event_options)) {
    spill = build_moi_spill_sequence(result, active_resources, event_options, spill_managers, arch,
                                     result.warnings, private_layout_base);
    if (!spill)
      return std::nullopt;
  }
  std::optional<SgprSpillSequence> scalar_spill;
  if (event_options.automatic_moi_record_replay_sgpr_spill) {
    scalar_spill = build_inline_sgpr_spill_sequence(result, active_resources, event_options,
                                                    spill_managers, arch, result.warnings,
                                                    private_layout_base, spill ? &*spill : nullptr);
    if (!scalar_spill)
      return std::nullopt;
  }
  return MoiPlannedRecordEvent{.resources = std::move(active_resources),
                               .spill = std::move(spill),
                               .scalar_spill = std::move(scalar_spill),
                               .private_owner = private_owner,
                               .private_layout = private_layout};
}
