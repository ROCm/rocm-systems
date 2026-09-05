void try_stage_first_light_access_fragments(std::span<const uint8_t> bytes,
                                            const ConSanOptions &options,
                                            const ConSanMoiOperatingPoint &operating_point,
                                            const ConSanTargetProfile &target,
                                            std::span<const ConSanMoiCandidate> admitted,
                                            const MoiObjectModeSemantics &semantics,
                                            ConSanTransformArtifacts &result) {
  const std::span<const uint8_t> active_bytes = result.active_bytes(bytes);
  AmdGpuCodeObject code_object(active_bytes.data(), active_bytes.size());
  if (!options.moi_report_buffer_address)
    return;
  if (options.moi_report_buffer_size < consan_moi_report_buffer_min_bytes(1, 0, 0, 0)) {
    result.warnings.emplace_back(
        "ConSan MOI first-light probe requires room for the report header and one access record");
    return;
  }
  const ConSanMoiReportBufferLayout &layout = semantics.report_layout;
  if (options.max_patches == 0u || layout.access_record_capacity == 0u)
    return;
  if (!options.moi_dynamic_access_records && layout.record_replay_access_dispatch_bank_count > 1u &&
      !moi_has_runtime_hardware_dispatch_id(operating_point) &&
      !operating_point.moi_dispatch_identity.private_fallback()) {
    result.outcome = ConSanTransformOutcome::Unsupported;
    result.warnings.emplace_back(
        "ConSan MOI automatic Record/Replay identity tables require runtime hardware dispatch "
        "identity");
    return;
  }

  std::vector<const ConSanMoiCandidate *> candidates = order_moi_replay_access_candidates(
      admitted, result.observation_plan(), options.moi_track_atomics);
  if (candidates.empty()) {
    result.warnings.emplace_back("ConSan MOI first-light probe found no native LDS or likely group "
                                 "flat load/store candidate");
    return;
  }

  struct PlannedSourceAccess : MoiPlannedProbeResources {
    const ConSanMoiCandidate *candidate = nullptr;
    ConSanMoiOperatingPoint point;
    std::optional<ConSanMoiPrivateStateLayout> private_state_layout;
    std::optional<MoiWorkitemOwnerDerivationPlan> owner_derivation;
    std::optional<MoiRuntimeWorkgroupGatePlan> runtime_gate;
    std::optional<ConSanMoiWorkgroupSources> runtime_sources;
    uint32_t record_index = 0u;
    uint32_t record_count = 0u;
    uint32_t logical_range_index = 0u;
    uint32_t probe_guest_offset = 0u;
    uint32_t guest_word_count = 0u;
    bool spill_overlaps_guest = false;
    bool wrap_embedded_guest_bank = false;
  };

  std::vector<PlannedSourceAccess> planned;
  const size_t selected_limit =
      options.max_patches_is_expert_limit ? options.max_patches : candidates.size();
  uint32_t record_count = 0u;
  uint32_t logical_range_count = 0u;
  MoiSpillManagers spill_managers;
  MoiPrivateStateLayoutCache private_layouts;
  for (const ConSanMoiCandidate *candidate_ptr : candidates) {
    const ConSanMoiCandidate &candidate = *candidate_ptr;
    const MoiAccessResourceFacts facts =
        resolve_moi_access_resource_facts(operating_point, candidate, target.arch);
    const uint16_t scratch_count =
        record_replay_access_scratch_vgpr_count(options, options, facts, candidate);
    auto scratch =
        resolve_moi_scratch(result.resource_plans, candidate, operating_point, scratch_count);
    if (!scratch)
      continue;
    ConSanMoiOperatingPoint point = operating_point;
    if (!apply_moi_transient_sgpr_assignment(options, point, result.moi_operating_point,
                                             scratch->owner_descriptor_file_offsets) ||
        !apply_moi_persistent_vgpr_assignment(point, result.moi_operating_point,
                                              scratch->owner_descriptor_file_offsets)) {
      continue;
    }
    std::optional<ConSanMoiPrivateStateLayout> private_layout;
    if (point.automatic_moi_private_epoch) {
      const std::optional<uint64_t> descriptor =
          scratch->owner_descriptor_file_offsets.size() == 1u
              ? std::optional{scratch->owner_descriptor_file_offsets.front()}
              : std::nullopt;
      private_layout = private_layouts.resolve(
          descriptor, result.program_inventory, *scratch, target.arch, result.warnings,
          {.owner = options.moi_owner_source == ConSanMoiOwnerSource::WorkitemId,
           .exact_workgroup = !consan_moi_detail::moi_has_exact_entry_workgroup_capture(point),
           .dispatch_id = point.moi_dispatch_identity.private_fallback()});
      if (!private_layout)
        continue;
    }
    if (!apply_record_replay_entry_workgroup_assignment(
            point, result.moi_operating_point, scratch->owner_descriptor_file_offsets,
            private_layout ? &*private_layout : nullptr))
      continue;
    std::optional<MoiWorkitemOwnerDerivationPlan> owner_derivation;
    if (!point.moi_owner_epoch_vgprs.owner() && !point.moi_persistent_sgprs.owner()) {
      const auto owner_shift =
          common_moi_workitem_owner_shift(bytes, *scratch, target.arch, result.warnings);
      if (!owner_shift)
        continue;
      owner_derivation = MoiWorkitemOwnerDerivationPlan{
          .entry_workitem_x_private_offset =
              private_layout ? private_layout->owner_offset : std::nullopt,
          .wave_size_shift = *owner_shift,
      };
    }
    const auto runtime_sources = moi_exact_entry_workgroup_sources(
        point, private_layout ? &private_layout->exact_workgroup_offsets : nullptr);
    const auto runtime_gate =
        runtime_sources && !point.has_compact_moi_scalar_spill() && point.moi_exec_save_sgpr
            ? plan_moi_runtime_workgroup_gate(
                  {.exec_save_sgpr = *point.moi_exec_save_sgpr,
                   .sample_stride = options.moi_runtime_sample_stride,
                   .sample_offset = options.moi_runtime_sample_offset,
                   .flavor = MoiRuntimeWorkgroupGatePlan::Flavor::RecordReplay,
                   .dispatch_id_sgpr = point.moi_dispatch_identity.sgpr(),
                   .literal_dispatch_id = options.moi_report_dispatch_id,
                   .direct_call_form = target.direct_call_form},
                  *runtime_sources)
            : std::nullopt;
    auto probe = plan_moi_probe_resources(
        result.program_inventory, *scratch, options, options, point, semantics, spill_managers,
        target.arch, private_layout, point.has_compact_moi_scalar_spill(), result.warnings);
    if (!probe)
      continue;
    const bool spill_overlaps_guest =
        probe->spill && !options.moi_dynamic_access_records &&
        reject_candidate_scratch_range_overlap(candidate, scratch->base, scratch->count,
                                               result.warnings);
    if (candidate.ranges.empty())
      continue;
    const uint64_t candidate_record_count =
        static_cast<uint64_t>(candidate.ranges.size()) *
        (options.moi_dynamic_access_records
             ? 1u
             : static_cast<uint64_t>(layout.record_replay_access_dispatch_bank_count) *
                   layout.record_replay_access_owner_bank_count);
    if (candidate_record_count > std::numeric_limits<uint32_t>::max() ||
        candidate.ranges.size() > std::numeric_limits<uint32_t>::max() - logical_range_count ||
        (!options.moi_dynamic_access_records &&
         (record_count > layout.access_record_capacity ||
          candidate_record_count > layout.access_record_capacity - record_count))) {
      result.warnings.emplace_back(
          "ConSan MOI first-light probe skipped a supported load/store site without enough "
          "report-buffer access-record slots");
      continue;
    }
    uint32_t guest_offset = 0u;
    uint32_t guest_words = 0u;
    std::vector<std::string> candidate_errors;
    if (!build_first_light_access_record_words(
            bytes, candidate, options, options, point, scratch->base, target.arch, record_count,
            static_cast<uint32_t>(candidate_record_count), logical_range_count, layout,
            spill_overlaps_guest, probe->spill ? &*probe->spill : nullptr,
            private_layout ? &*private_layout : nullptr, owner_derivation, candidate_errors,
            &guest_offset, &guest_words)) {
      result.warnings.insert(result.warnings.end(), candidate_errors.begin(),
                             candidate_errors.end());
      continue;
    }
    const bool wrap_guest_bank = options.moi_dynamic_access_records &&
                                 target.has_selectable_vgpr_bank &&
                                 candidate.incoming_vgpr_bank_mode.value_or(0u) != 0u;
    PlannedSourceAccess entry;
    static_cast<MoiPlannedProbeResources &>(entry) = std::move(*probe);
    entry.candidate = candidate_ptr;
    entry.point = std::move(point);
    entry.private_state_layout = private_layout;
    entry.owner_derivation = std::move(owner_derivation);
    entry.runtime_gate = runtime_gate;
    entry.runtime_sources = runtime_sources;
    entry.record_index = record_count;
    entry.record_count = static_cast<uint32_t>(candidate_record_count);
    entry.logical_range_index = logical_range_count;
    entry.probe_guest_offset = guest_offset;
    entry.guest_word_count = guest_words;
    entry.spill_overlaps_guest = spill_overlaps_guest;
    entry.wrap_embedded_guest_bank = wrap_guest_bank;
    planned.push_back(std::move(entry));
    record_count += static_cast<uint32_t>(candidate_record_count);
    logical_range_count += static_cast<uint32_t>(candidate.ranges.size());
    if (planned.size() == selected_limit)
      break;
  }
  if (planned.empty())
    return;

  MoiDescriptorVgprRequirements vgprs;
  MoiDescriptorSgprRequirements sgprs;
  MoiDescriptorPrivateRequirements private_bytes;
  std::vector<ConSanTextFragment> fragments;
  fragments.reserve(planned.size());
  for (const PlannedSourceAccess &entry : planned) {
    const ConSanMoiCandidate &candidate = *entry.candidate;
    note_descriptor_requirements(vgprs, entry.resources);
    note_moi_sgpr_requirements(sgprs, entry.resources, options, options, entry.point, semantics,
                               target.arch);
    note_moi_probe_private_requirements(private_bytes, entry);
    uint32_t probe_guest_offset = 0u;
    uint32_t guest_words = 0u;
    auto probe_words = build_first_light_access_record_words(
        bytes, candidate, options, options, entry.point, entry.resources.base, target.arch,
        entry.record_index, record_count, entry.logical_range_index, layout,
        entry.spill_overlaps_guest, entry.spill ? &*entry.spill : nullptr,
        entry.private_state_layout ? &*entry.private_state_layout : nullptr, entry.owner_derivation,
        result.errors, &probe_guest_offset, &guest_words);
    if (!probe_words)
      return;
    MoiAppendedBodyOptions body_options(target.arch);
    body_options.scalar_save_words = entry.scalar_spill
                                         ? std::span<const uint32_t>(entry.scalar_spill->save_words)
                                         : std::span<const uint32_t>{};
    body_options.scalar_restore_words =
        entry.scalar_spill ? std::span<const uint32_t>(entry.scalar_spill->restore_words)
                           : std::span<const uint32_t>{};
    body_options.incoming_vgpr_bank_mode = candidate.incoming_vgpr_bank_mode;
    if (moi_access_requires_high_bank_address_capture(candidate, target.arch)) {
      const uint16_t base_count = record_replay_uses_automatic_banked_capture(
                                      options, layout.record_replay_dispatch_token_capacity)
                                      ? 10u
                                      : 6u;
      body_options.high_bank_address_source = candidate.lowering.form->address_vgpr;
      body_options.high_bank_address_destination =
          static_cast<uint16_t>(entry.resources.base + base_count);
    }
    body_options.trailing_guest_word_count = options.moi_dynamic_access_records ? 0u : guest_words;
    body_options.probe_guest_instruction_offset = probe_guest_offset;
    body_options.guest_instruction_word_count = guest_words;
    uint32_t body_guest_offset = 0u;
    body_options.body_guest_instruction_offset = &body_guest_offset;
    body_options.restore_vgpr_spill_before_trailing_guest = entry.spill_overlaps_guest;
    body_options.wrap_embedded_guest_vgpr_bank = entry.wrap_embedded_guest_bank;
    auto body =
        assemble_moi_appended_body({.spill = entry.spill ? &*entry.spill : nullptr,
                                    .displaced_tail_words = {},
                                    .body_size = 0u,
                                    .guest_instruction_size = candidate.size()},
                                   *probe_words, "first-light probe", result.errors, body_options);
    if (!body)
      return;
    ConSanPatchInfo patch;
    patch.kind = ConSanPatchKind::TrampolineMoiAccessRecordStore;
    patch.anchor_offset = candidate.anchor();
    patch.original_size = candidate.size();
    note_moi_probe_patch_info(patch, entry);
    auto access = make_moi_access_attribution(result.observation_plan(), candidate,
                                              ConSanProbeIntentKind::AccessRecord);
    if (!access) {
      result.errors.emplace_back(
          "ConSan MOI first-light probe produced invalid source attribution");
      return;
    }
    auto fragment = make_consan_around_text_fragment(
        std::move(*body), body_guest_offset, candidate.size(), candidate.intent_ids,
        *make_record_replay_access_runtime_mapping(std::move(*access)), std::move(patch),
        result.errors, "MOI first-light probe",
        guest_words * static_cast<uint32_t>(sizeof(uint32_t)));
    if (!fragment)
      return;
    if (entry.runtime_gate && entry.runtime_sources) {
      if (!prepend_moi_runtime_workgroup_gate(*fragment, *entry.runtime_gate,
                                              *entry.runtime_sources, target.arch, result.errors,
                                              "MOI first-light probe"))
        return;
    }
    fragments.push_back(std::move(*fragment));
  }
  const MoiDescriptorLdsRequirements no_lds;
  (void)stage_consan_text_rewrites(code_object, target.arch, {vgprs, sgprs, private_bytes, no_lds},
                                   moi_descriptor_mutation_policy(&options, target.arch),
                                   "MOI first-light probe", std::move(fragments), result);
}
