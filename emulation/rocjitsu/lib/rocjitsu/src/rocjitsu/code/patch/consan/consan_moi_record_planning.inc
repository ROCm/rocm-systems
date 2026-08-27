/// Common placement-to-emission handoff for one Record/Replay synchronization
/// event.
///
/// An atomic, barrier, or fence has different semantic evidence and native
/// emission, but all three use the same resolved scratch window, private epoch
/// and owner representation, and register-preservation plan. Keeping that
/// transaction in one type prevents the event-specific planners from
/// acquiring subtly different rules for the same Record/Replay resources.
struct MoiPlannedRecordEvent : MoiPlannedProbeResources {
  std::optional<MoiPrivateRecordOwnerPlan> private_owner;
};

/// Bind one planned Record/Replay event to the still-mutable options accepted
/// by the native emitters.
///
/// This is the single remaining compatibility boundary between the typed
/// event plan and emitters that have not yet replaced `MoiOptions` with a
/// narrow lowering input. It selects the event's scratch window, private
/// workgroup representation, transient scalar assignment, and entry-captured
/// workgroup tuple. A false result means a previously proved owner-local
/// assignment disappeared and callers must abort rather than emit with a
/// different register interpretation.
[[nodiscard]] bool bind_moi_record_event_options(MoiOptions &event_options,
                                                 const ConSanMoiOperatingPoint &allocation,
                                                 const MoiPlannedRecordEvent &event) {
  event_options.scratch_vgpr = event.resources.base;
  if (event.private_layout) {
    event_options.moi_record_replay_workgroup_private_offsets =
        event.private_layout->record_replay_workgroup_offsets;
  }
  return apply_moi_transient_sgpr_assignment(event_options, allocation,
                                             event.resources.owner_descriptor_file_offsets) &&
         apply_record_replay_entry_workgroup_assignment(
             event_options, allocation, event.resources.owner_descriptor_file_offsets);
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
                      const ResolvedMoiScratchPlan &resources, const MoiOptions &options,
                      MoiSpillManagers &spill_managers, rj_code_arch_t arch,
                      std::string_view warning_context,
                      std::optional<uint32_t> active_private_segment_size = std::nullopt) {
  MoiOptions event_options = options;
  event_options.scratch_vgpr = resources.base;
  if (!apply_moi_transient_sgpr_assignment(event_options, result.moi_operating_point,
                                           resources.owner_descriptor_file_offsets)) {
    result.warnings.emplace_back(std::string(warning_context) +
                                 " found no common transient SGPR assignment");
    return std::nullopt;
  }
  if (!apply_moi_persistent_vgpr_assignment(event_options, result.moi_operating_point,
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
  if (!apply_record_replay_entry_workgroup_assignment(event_options, result.moi_operating_point,
                                                      resources.owner_descriptor_file_offsets)) {
    result.warnings.emplace_back(std::string(warning_context) +
                                 " found no common entry workgroup assignment");
    return std::nullopt;
  }

  std::optional<MoiPrivateRecordOwnerPlan> private_owner;
  if (moi_record_uses_private_owner(event_options)) {
    ResolvedMoiScratchPlan owner_resources = resources;
    if (active_private_segment_size)
      owner_resources.original_private_segment_size = *active_private_segment_size;
    private_owner = resolve_moi_private_record_owner(bytes, owner_resources, *private_layout, arch,
                                                     result.warnings);
    if (!private_owner)
      return std::nullopt;
  }
  auto probe = plan_moi_probe_resources(
      result, resources, event_options, spill_managers, arch, std::move(private_layout),
      event_options.automatic_moi_record_replay_sgpr_spill, active_private_segment_size);
  if (!probe)
    return std::nullopt;
  return MoiPlannedRecordEvent{std::move(*probe), private_owner};
}
