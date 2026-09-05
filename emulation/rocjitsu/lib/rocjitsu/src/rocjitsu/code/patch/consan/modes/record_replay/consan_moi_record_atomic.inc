void try_stage_atomic_record_fragments(std::span<const uint8_t> bytes, const ConSanOptions &options,
                                       const ConSanMoiOperatingPoint &operating_point,
                                       const MoiObjectModeSemantics &mode_semantics,
                                       rj_code_arch_t arch, ConSanTransformArtifacts &result) {
  if (!options.moi_track_atomics)
    return;
  if (consan_target_profile(arch) == nullptr) {
    result.warnings.emplace_back(
        "ConSan MOI atomic record patch does not support this architecture");
    return;
  }
  if (!options.moi_report_buffer_address) {
    result.warnings.emplace_back("ConSan MOI atomic record patch requires a MOI report buffer");
    return;
  }
  if (options.moi_report_buffer_size < consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 0, 1, 1)) {
    result.warnings.emplace_back(
        "ConSan MOI atomic record patch requires room for access and atomic records");
    return;
  }
  const ConSanMoiReportBufferLayout &layout = mode_semantics.report_layout;
  const uint32_t max_candidates = std::min(options.max_patches, layout.atomic_record_capacity);
  if (max_candidates == 0) {
    result.warnings.emplace_back("ConSan MOI atomic record patch has no atomic record capacity");
    return;
  }
  for (const ConSanAtomicSiteDecision &decision : result.observation_plan().atomic_site_decisions) {
    if (decision.reason != ConSanAtomicPolicyReason::CompareExchangeOutcomeUnavailable)
      continue;
    for (const std::string &container_name : decision.source_containers) {
      result.warnings.emplace_back(
          "ConSan MOI atomic record patch skipped compare-exchange-outcome-unavailable in " +
          container_name);
    }
  }
  std::vector<MoiAtomicEvidenceSitePlan> candidates =
      build_moi_atomic_evidence_site_plans(result.program_inventory, result.observation_plan(),
                                           ConSanProbeIntentKind::AtomicRecord, result.errors);
  if (!result.errors.empty())
    return;
  if (candidates.empty()) {
    result.warnings.emplace_back("ConSan MOI atomic record patch found no supported flat atomics");
    return;
  }

  const std::span<const uint8_t> active_bytes = result.active_bytes(bytes);
  AmdGpuCodeObject code_object(active_bytes.data(), active_bytes.size());
  const std::span<const uint8_t> old_text =
      code_object.text_sections().size() == 1u
          ? std::span<const uint8_t>(
                reinterpret_cast<const uint8_t *>(code_object.text_sections().front()->data()),
                code_object.text_sections().front()->size())
          : std::span<const uint8_t>{};
  if (old_text.empty()) {
    result.errors.emplace_back("ConSan MOI atomic record patch found no .text section");
    return;
  }
  struct PlannedAtomicRecord : MoiPlannedRecordEvent {
    MoiAtomicEvidenceSitePlan candidate;
    ConSanMoiAtomicAddressPlan address_plan;
  };
  std::vector<PlannedAtomicRecord> selected_candidates;
  selected_candidates.reserve(max_candidates);
  MoiResourcePlanningStatePtr resource_state = make_moi_resource_planning_state(
      bytes, arch, result.program_inventory, result.observation_plan(), result.resource_plans,
      options, options, operating_point);
  MoiSpillManagers spill_managers;
  MoiDescriptorVgprRequirements descriptor_requirements;
  MoiDescriptorSgprRequirements scalar_requirements;
  MoiDescriptorPrivateRequirements private_requirements;
  for (size_t candidate_index = 0; candidate_index < candidates.size(); ++candidate_index) {
    const MoiAtomicEvidenceSitePlan &candidate = candidates[candidate_index];
    if (selected_candidates.size() == max_candidates)
      break;
    const ConSanAtomicSite &site = candidate.site;
    if (site.text_offset > old_text.size() || site.size > old_text.size() - site.text_offset) {
      result.errors.emplace_back("ConSan MOI atomic record patch site is outside .text");
      return;
    }
    const bool incompatible_overlap =
        std::ranges::any_of(result.staged_text_fragments, [&](const ConSanTextFragment &fragment) {
          const uint64_t fragment_end = fragment.patch.anchor_offset + fragment.patch.original_size;
          const uint64_t site_end = site.text_offset + site.size;
          return site.text_offset < fragment_end && fragment.patch.anchor_offset < site_end &&
                 (fragment.patch.anchor_offset != site.text_offset ||
                  fragment.patch.original_size != site.size);
        });
    if (incompatible_overlap) {
      result.warnings.emplace_back(
          "ConSan MOI atomic record patch skipped an overlapping patch in " +
          candidate.container_name);
      continue;
    }
    const ConSanCandidateResourcePlan *existing_plan = resource_plan_for_site(
        result.resource_plans, ConSanResourceSiteKind::Atomic, site.text_offset);
    const uint16_t scratch_count = atomic_record_scratch_count(candidate.lowering_form);
    ConSanCandidateResourcePlan plan =
        existing_plan
            ? *existing_plan
            : plan_moi_resource_site(*resource_state, options, options, operating_point,
                                     ConSanResourceSiteKind::Atomic, candidate_index,
                                     site.text_offset, candidate.kernel_descriptor_file_offset,
                                     scratch_count, nullptr, &candidate.lowering_form);
    plan.intent_ids = {candidate.address_capture_intent, candidate.evidence_intent};
    if (!existing_plan)
      result.resource_plans.push_back(plan);
    const auto resources =
        resolve_moi_scratch_plan(plan, operating_point, result.moi_operating_point, scratch_count);
    if (!resources) {
      result.warnings.emplace_back(
          "ConSan MOI atomic record patch has no usable resource plan in " +
          candidate.container_name);
      continue;
    }
    const ConSanMoiAtomicAddressPlan address_plan = plan_consan_moi_atomic_address(
        candidate.lowering_form, resources->base, resources->count, resources->source);
    if (!address_plan.supported()) {
      result.warnings.emplace_back(
          "ConSan MOI atomic record patch has no usable address plan in " +
          candidate.container_name + ": " +
          std::string(consan_moi_atomic_address_support_name(address_plan.support)));
      continue;
    }
    MoiAtomicEvidenceSitePlan planned_candidate = candidate;
    if (!operating_point.moi_owner_epoch_vgprs.owner() &&
        !planned_candidate.kernel_descriptor_file_offset) {
      if (resources->owner_descriptor_file_offsets.empty()) {
        result.warnings.emplace_back(
            "ConSan MOI atomic record patch requires an owning kernel descriptor");
        continue;
      }
      planned_candidate.kernel_descriptor_file_offset =
          resources->owner_descriptor_file_offsets.front();
    }
    auto record_event =
        plan_moi_record_event(bytes, result.program_inventory, result.warnings, *resources, options,
                              options, operating_point, result.moi_operating_point, mode_semantics,
                              spill_managers, arch, "ConSan MOI atomic record patch");
    if (!record_event)
      continue;
    selected_candidates.push_back(
        {std::move(*record_event), std::move(planned_candidate), address_plan});
  }
  // Do not prune a release based on module-local absence of an acquire. Real
  // applications can split producers and consumers across code objects, and
  // Record/Replay must preserve every admitted access in its event stream.
  if (selected_candidates.empty()) {
    result.warnings.emplace_back("ConSan MOI atomic record patch found no patchable atomics");
    return;
  }
  for (const PlannedAtomicRecord &planned : selected_candidates) {
    note_moi_probe_private_requirements(private_requirements, planned);
    note_descriptor_requirements(descriptor_requirements, planned.resources);
    note_moi_sgpr_requirements(scalar_requirements, planned.resources, planned.emission, arch);
  }
  std::vector<ConSanPatchInfo> scalar_clause_patches;
  for (const PlannedAtomicRecord &planned : selected_candidates) {
    if (!append_atomic_scalar_clause_patch(old_text, planned.candidate, arch, scalar_clause_patches,
                                           result.errors))
      return;
  }
  std::vector<ConSanTextFragment> fragments;
  fragments.reserve(selected_candidates.size() + scalar_clause_patches.size());
  for (ConSanPatchInfo &patch : scalar_clause_patches)
    fragments.push_back(ConSanTextFragment::replacement({build_s_nop(0u, arch)}, std::move(patch)));
  uint32_t record_index = 0;
  for (const PlannedAtomicRecord &planned : selected_candidates) {
    const MoiAtomicEvidenceSitePlan &candidate = planned.candidate;
    const ConSanAtomicSite &site = candidate.site;
    uint32_t guest_instruction_offset = 0;
    auto cave_words = build_atomic_record_cave_words(
        active_bytes, candidate, planned.address_plan, planned.emission,
        planned.spill ? &*planned.spill : nullptr,
        planned.scalar_spill ? &*planned.scalar_spill : nullptr, arch, record_index,
        layout.atomic_record_capacity, layout.atomic_records_offset, guest_instruction_offset,
        result.errors);
    if (!cave_words)
      return;

    ConSanPatchInfo info;
    info.kind = ConSanPatchKind::TrampolineMoiAtomicRecord;
    info.anchor_offset = site.text_offset;
    info.original_size = site.size;
    note_moi_probe_patch_info(info, planned);
    auto fragment = make_consan_around_text_fragment(
        std::move(*cave_words), guest_instruction_offset, site.size, candidate.intent_ids(), {},
        std::move(info), result.errors, "MOI atomic record patch");
    if (!fragment)
      return;
    const bool access_gate_already_staged =
        std::ranges::any_of(result.staged_text_fragments, [&](const ConSanTextFragment &existing) {
          return existing.patch.anchor_offset == site.text_offset &&
                 (existing.patch.kind == ConSanPatchKind::InlineMoiAccessRecordStore ||
                  existing.patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore) &&
                 std::ranges::any_of(existing.branch_fixups, [](const auto &fixup) {
                   return fixup.target == ConSanTextFragmentBranchTarget::Bypass;
                 });
        });
    if (!access_gate_already_staged && planned.emission.runtime_workgroup_gate) {
      if (!prepend_moi_runtime_workgroup_gate(*fragment, *planned.emission.runtime_workgroup_gate,
                                              planned.emission.workgroup_sources, arch,
                                              result.errors, "MOI atomic record patch"))
        return;
    }
    fragments.push_back(std::move(*fragment));
    ++record_index;
  }
  const MoiDescriptorLdsRequirements no_lds;
  (void)stage_consan_text_rewrites(
      code_object, arch,
      {descriptor_requirements, scalar_requirements, private_requirements, no_lds},
      moi_descriptor_mutation_policy(nullptr, arch), "MOI atomic record patch",
      std::move(fragments), result);
}
