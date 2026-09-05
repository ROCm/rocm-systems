void try_stage_record_replay_barrier_fragments(std::span<const uint8_t> bytes,
                                               const ConSanOptions &options,
                                               const ConSanMoiOperatingPoint &operating_point,
                                               rj_code_arch_t arch,
                                               const MoiObjectModeSemantics &semantics,
                                               ConSanTransformArtifacts &result) {
  if (!options.moi_track_barriers)
    return;
  if (!consan_is_capability_arch(arch)) {
    result.warnings.emplace_back(
        "ConSan MOI barrier record patch does not support this architecture");
    return;
  }
  if (!options.moi_report_buffer_address) {
    result.warnings.emplace_back("ConSan MOI barrier record patch requires a MOI report buffer");
    return;
  }
  if (options.moi_report_buffer_size < consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 1)) {
    result.warnings.emplace_back(
        "ConSan MOI barrier record patch requires room for access and barrier records");
    return;
  }
  const ConSanMoiReportBufferLayout &layout = semantics.report_layout;
  const uint32_t max_candidates = std::min(options.max_patches, layout.barrier_record_capacity);
  if (max_candidates == 0u || !operating_point.moi_exec_save_sgpr) {
    result.warnings.emplace_back(
        max_candidates == 0u ? "ConSan MOI barrier record patch has no barrier record capacity"
                             : "ConSan MOI barrier record patch requires "
                               "RJ_CONSAN_MOI_EXEC_SAVE_SGPR");
    return;
  }

  std::vector<MoiBarrierEvidenceSitePlan> candidates =
      build_moi_barrier_evidence_site_plans(result.program_inventory, result.observation_plan(),
                                            ConSanProbeIntentKind::BarrierRecord, result.errors);
  if (!result.errors.empty())
    return;
  const std::span<const uint8_t> active_bytes = result.active_bytes(bytes);
  AmdGpuCodeObject code_object(active_bytes.data(), active_bytes.size());
  const std::span<const uint8_t> text =
      code_object.text_sections().size() == 1u
          ? std::span<const uint8_t>(
                reinterpret_cast<const uint8_t *>(code_object.text_sections().front()->data()),
                code_object.text_sections().front()->size())
          : std::span<const uint8_t>{};
  if (text.empty()) {
    result.errors.emplace_back("ConSan MOI barrier record patch found no .text section");
    return;
  }

  struct PlannedBarrierRecord : MoiPlannedRecordEvent {
    MoiBarrierEvidenceSitePlan candidate;
  };
  std::vector<PlannedBarrierRecord> selected;
  std::unordered_set<uint64_t> selected_offsets;
  MoiResourcePlanningStatePtr resources = make_moi_resource_planning_state(
      bytes, arch, result.program_inventory, result.observation_plan(), result.resource_plans,
      options, options, operating_point);
  MoiSpillManagers spill_managers;
  MoiDescriptorVgprRequirements vgprs;
  MoiDescriptorSgprRequirements sgprs;
  MoiDescriptorPrivateRequirements private_bytes;
  for (size_t index = 0u; index < candidates.size() && selected.size() < max_candidates; ++index) {
    MoiBarrierEvidenceSitePlan candidate = candidates[index];
    const ConSanBarrierSite &site = candidate.site;
    if (site.size != sizeof(uint32_t)) {
      result.warnings.emplace_back(
          "ConSan MOI barrier record patch skipped non-32-bit barrier " + site.mnemonic + " in " +
          moi_evidence_container_name(result.program_inventory, candidate.source_site));
      continue;
    }
    if (site.text_offset > text.size() || site.size > text.size() - site.text_offset) {
      result.errors.emplace_back("ConSan MOI barrier record patch site is outside .text");
      return;
    }
    if (!selected_offsets.insert(site.text_offset).second) {
      result.warnings.emplace_back("ConSan MOI barrier record patch skipped an overlapping patch");
      continue;
    }
    const ConSanCandidateResourcePlan *existing = resource_plan_for_site(
        result.resource_plans, ConSanResourceSiteKind::Barrier, site.text_offset);
    ConSanCandidateResourcePlan plan =
        existing ? *existing
                 : plan_moi_resource_site(*resources, options, options, operating_point,
                                          ConSanResourceSiteKind::Barrier, index, site.text_offset,
                                          candidate.source_site, /*scratch_count=*/6u);
    plan.intent_ids = {candidate.evidence_intent};
    if (!existing)
      result.resource_plans.push_back(plan);
    auto scratch = resolve_moi_scratch_plan(plan, operating_point, result.moi_operating_point, 6u);
    if (!scratch) {
      result.warnings.emplace_back(
          "ConSan MOI barrier record patch has no usable resource plan in " +
          moi_evidence_container_name(result.program_inventory, candidate.source_site));
      continue;
    }
    auto event =
        plan_moi_record_event(bytes, result.program_inventory, result.warnings, *scratch, options,
                              options, operating_point, result.moi_operating_point, semantics,
                              spill_managers, arch, "ConSan MOI barrier record patch");
    if (!event)
      continue;
    note_descriptor_requirements(vgprs, event->resources);
    note_moi_sgpr_requirements(sgprs, event->resources, event->emission, arch);
    note_moi_probe_private_requirements(private_bytes, *event);
    selected.push_back({std::move(*event), std::move(candidate)});
  }
  if (selected.empty()) {
    result.warnings.emplace_back(
        candidates.empty() ? "ConSan MOI barrier record patch found no barriers"
                           : "ConSan MOI barrier record patch found no patchable barriers");
    return;
  }

  std::vector<ConSanTextFragment> fragments;
  fragments.reserve(selected.size());
  const bool accesses_are_runtime_gated =
      std::ranges::any_of(result.staged_text_fragments, [](const ConSanTextFragment &fragment) {
        const bool access = fragment.patch.kind == ConSanPatchKind::InlineMoiAccessRecordStore ||
                            fragment.patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore;
        return access && std::ranges::any_of(fragment.branch_fixups, [](const auto &fixup) {
                 return fixup.target == ConSanTextFragmentBranchTarget::Bypass;
               });
      });
  for (const PlannedBarrierRecord &planned : selected) {
    const ConSanProgramContainer *container =
        resolve_moi_evidence_container(result.program_inventory, planned.candidate.source_site);
    if (container == nullptr) {
      result.errors.emplace_back("ConSan MOI barrier record emission lost its source container");
      return;
    }
    uint32_t barrier_word = 0u;
    std::memcpy(&barrier_word, text.data() + planned.candidate.site.text_offset,
                sizeof(barrier_word));
    uint32_t guest_offset = 0u;
    auto words = build_barrier_record_cave_words(
        active_bytes, planned.candidate, *container, planned.emission,
        planned.spill ? &*planned.spill : nullptr,
        planned.scalar_spill ? &*planned.scalar_spill : nullptr, arch,
        layout.barrier_record_capacity, layout.barrier_records_offset, barrier_word, result.errors,
        &guest_offset);
    if (!words)
      return;
    ConSanPatchInfo patch;
    patch.kind = ConSanPatchKind::TrampolineMoiBarrierRecord;
    patch.anchor_offset = planned.candidate.site.text_offset;
    patch.original_size = planned.candidate.site.size;
    note_moi_probe_patch_info(patch, planned);
    auto fragment = make_consan_around_text_fragment(
        std::move(*words), guest_offset, planned.candidate.site.size,
        planned.candidate.intent_ids(), {}, std::move(patch), result.errors,
        "MOI barrier record patch");
    if (!fragment)
      return;
    if (accesses_are_runtime_gated && planned.emission.runtime_workgroup_gate) {
      if (!prepend_moi_runtime_workgroup_gate(*fragment, *planned.emission.runtime_workgroup_gate,
                                              planned.emission.workgroup_sources, arch,
                                              result.errors, "MOI barrier record patch"))
        return;
    }
    fragments.push_back(std::move(*fragment));
  }
  const MoiDescriptorLdsRequirements no_lds;
  (void)stage_consan_text_rewrites(code_object, arch, {vgprs, sgprs, private_bytes, no_lds},
                                   moi_descriptor_mutation_policy(nullptr, arch),
                                   "MOI barrier record patch", std::move(fragments), result);
}
