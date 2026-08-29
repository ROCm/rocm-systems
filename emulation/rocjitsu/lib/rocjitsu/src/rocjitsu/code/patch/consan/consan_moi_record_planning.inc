/// Resolved lowering input shared by Record/Replay atomic, fence, and barrier
/// emitters.
///
/// This deliberately contains only values read by those emitters and their
/// routing boundary. It is immutable after owner-local placement resolves the
/// event and replaces the former pattern of copying and mutating all of
/// `MoiOptions` again at each consumer.
struct MoiRecordEventEmissionPlan {
  std::optional<uint16_t> scratch_vgpr;
  std::optional<uint16_t> moi_exec_save_sgpr;
  std::optional<uint16_t> moi_owner_vgpr;
  std::optional<uint16_t> moi_epoch_vgpr;
  std::optional<uint16_t> moi_workgroup_key_vgpr;
  std::optional<uint16_t> moi_dispatch_id_vgpr;
  ConSanMoiPersistentWorkgroupRegisters moi_record_replay_workgroup_vgprs;
  ConSanMoiPersistentSgprState moi_persistent_sgprs;
  std::optional<uint64_t> moi_report_buffer_address;
  bool automatic_moi_record_replay_sgpr_spill = false;
  std::optional<uint16_t> moi_record_replay_dispatch_key_sgpr;
  std::optional<uint16_t> moi_record_replay_call_return_sgpr;
  ConSanMoiWorkgroupSources workgroup_sources;
  MoiSpecialStateSgprs special_state;
  std::array<ConSanMoiReportDispatchIdWordSource, 2> dispatch_id_sources;
  std::optional<MoiRuntimeWorkgroupGatePlan> runtime_workgroup_gate;
  std::optional<MoiIndirectJumpSgprs> indirect_jump;
  uint16_t required_sgpr_count = 0;
};

[[nodiscard]] std::optional<MoiSpecialStateSgprs>
moi_special_state_sgprs(const MoiRecordEventEmissionPlan &plan) {
  return plan.special_state;
}

[[nodiscard]] std::optional<MoiIndirectJumpSgprs>
moi_indirect_jump_sgprs(const MoiRecordEventEmissionPlan &plan) {
  return plan.indirect_jump;
}

[[nodiscard]] bool append_dynamic_record_store_moi_report_dispatch_id_pair(
    std::vector<uint32_t> &words, const DynamicRecordLayout &layout, uint64_t low_field_address,
    const MoiRecordEventEmissionPlan &plan, uint16_t slot_vgpr, uint16_t scratch_vgpr,
    rj_code_arch_t arch, ConSanMoiLiteralDispatchIdPolicy policy) {
  return append_dynamic_record_store_moi_report_dispatch_id_pair(
      words, layout, low_field_address, plan.dispatch_id_sources, slot_vgpr, scratch_vgpr, arch,
      policy);
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
[[nodiscard]] std::optional<MoiRecordEventEmissionPlan>
resolve_moi_record_event_emission_plan(const MoiOptions &event_options, uint16_t scratch_vgpr,
                                       rj_code_arch_t arch) {
  const auto workgroup_sources = record_replay_persistent_workgroup_sources(event_options, arch);
  const auto special_state = moi_special_state_sgprs(event_options);
  if (!event_options.moi_exec_save_sgpr || !event_options.moi_report_buffer_address ||
      !workgroup_sources || !special_state) {
    return std::nullopt;
  }
  std::optional<MoiRuntimeWorkgroupGatePlan> runtime_workgroup_gate;
  if (event_options.moi_runtime_sample_stride > 1u &&
      !event_options.automatic_moi_record_replay_sgpr_spill) {
    runtime_workgroup_gate = plan_moi_runtime_workgroup_gate(
        event_options, event_options, event_options, *workgroup_sources, arch);
  }
  return MoiRecordEventEmissionPlan{
      .scratch_vgpr = scratch_vgpr,
      .moi_exec_save_sgpr = event_options.moi_exec_save_sgpr,
      .moi_owner_vgpr = event_options.moi_owner_vgpr,
      .moi_epoch_vgpr = event_options.moi_epoch_vgpr,
      .moi_workgroup_key_vgpr = event_options.moi_workgroup_key_vgpr,
      .moi_dispatch_id_vgpr = event_options.moi_dispatch_id_vgpr,
      .moi_record_replay_workgroup_vgprs = event_options.moi_record_replay_workgroup_vgprs,
      .moi_persistent_sgprs = event_options.moi_persistent_sgprs,
      .moi_report_buffer_address = event_options.moi_report_buffer_address,
      .automatic_moi_record_replay_sgpr_spill =
          event_options.automatic_moi_record_replay_sgpr_spill,
      .moi_record_replay_dispatch_key_sgpr = event_options.moi_record_replay_dispatch_key_sgpr,
      .moi_record_replay_call_return_sgpr = event_options.moi_record_replay_call_return_sgpr,
      .workgroup_sources = *workgroup_sources,
      .special_state = *special_state,
      .dispatch_id_sources = {moi_report_dispatch_id_word_source(event_options,
                                                                 /*high_word=*/false),
                              moi_report_dispatch_id_word_source(event_options,
                                                                 /*high_word=*/true)},
      .runtime_workgroup_gate = runtime_workgroup_gate,
      .indirect_jump = moi_indirect_jump_sgprs(event_options),
  };
}

struct MoiPlannedRecordEvent : MoiPlannedProbeResources {
  MoiRecordEventEmissionPlan emission;
  std::optional<MoiWorkitemOwnerDerivationPlan> derived_owner;
};

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

  auto emission = resolve_moi_record_event_emission_plan(event_options, resources.base, arch);
  if (!emission) {
    result.warnings.emplace_back(std::string(warning_context) +
                                 " has an incomplete Record/Replay emission plan");
    return std::nullopt;
  }

  std::optional<MoiWorkitemOwnerDerivationPlan> derived_owner;
  if (!event_options.moi_owner_vgpr && !event_options.moi_persistent_sgprs.owner) {
    if (moi_record_uses_private_owner(event_options)) {
      ResolvedMoiScratchPlan owner_resources = resources;
      if (active_private_segment_size)
        owner_resources.original_private_segment_size = *active_private_segment_size;
      derived_owner = resolve_moi_private_workitem_owner(bytes, owner_resources, *private_layout,
                                                         arch, result.warnings);
    } else {
      const auto shift = common_moi_workitem_owner_shift(bytes, resources, arch, result.warnings);
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
  auto probe = plan_moi_probe_resources(
      result, resources, event_options, spill_managers, arch, std::move(private_layout),
      event_options.automatic_moi_record_replay_sgpr_spill, active_private_segment_size);
  if (!probe)
    return std::nullopt;

  MoiDescriptorSgprRequirements scalar_requirements;
  note_moi_sgpr_requirements(scalar_requirements, resources, event_options, arch);
  uint16_t required_sgpr_count = 0u;
  for (const auto &[descriptor, count] : scalar_requirements) {
    (void)descriptor;
    required_sgpr_count = std::max(required_sgpr_count, count);
  }
  emission->required_sgpr_count = required_sgpr_count;
  return MoiPlannedRecordEvent{std::move(*probe), std::move(*emission), derived_owner};
}
