void try_stage_fence_record_fragments(std::span<const uint8_t> bytes, const ConSanOptions &options,
                                      const ConSanMoiOperatingPoint &operating_point,
                                      const MoiObjectModeSemantics &mode_semantics,
                                      rj_code_arch_t arch, ConSanTransformArtifacts &result) {
  if (!options.moi_track_atomics)
    return;
  if (!consan_is_capability_arch(arch)) {
    result.warnings.emplace_back(
        "ConSan MOI fence record patch does not support this architecture");
    return;
  }
  if (!options.moi_report_buffer_address) {
    result.warnings.emplace_back("ConSan MOI fence record patch requires a MOI report buffer");
    return;
  }
  const ConSanMoiReportBufferLayout &layout = mode_semantics.report_layout;
  const uint32_t max_candidates = std::min(options.max_patches, layout.fence_record_capacity);
  if (max_candidates == 0) {
    result.warnings.emplace_back("ConSan MOI fence record patch has no fence record capacity");
    return;
  }

  std::vector<MoiFenceEvidenceSitePlan> candidates = build_moi_fence_evidence_site_plans(
      result.program_inventory, result.observation_plan(), result.errors);
  if (!result.errors.empty())
    return;
  if (candidates.empty()) {
    result.warnings.emplace_back(
        "ConSan MOI fence record patch rejected all qualified communication events");
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
    result.errors.emplace_back("ConSan MOI fence record patch found no .text section");
    return;
  }
  struct PlannedFenceRecord : MoiPlannedRecordEvent {
    MoiFenceEvidenceSitePlan candidate;
    ConSanMoiAtomicAddressPlan address_plan;
  };
  std::vector<PlannedFenceRecord> selected;
  MoiResourcePlanningStatePtr resource_state = make_moi_resource_planning_state(
      bytes, arch, result.program_inventory, result.observation_plan(), result.resource_plans,
      options, options, operating_point);
  MoiSpillManagers spill_managers;
  MoiDescriptorVgprRequirements descriptor_requirements;
  MoiDescriptorSgprRequirements scalar_requirements;
  MoiDescriptorPrivateRequirements private_requirements;
  for (size_t candidate_index = 0; candidate_index < candidates.size(); ++candidate_index) {
    if (selected.size() == max_candidates)
      break;
    MoiFenceEvidenceSitePlan candidate = candidates[candidate_index];
    const uint64_t fence_text_offset = candidate.semantic_site.physical.original_text_offset;
    if (candidate.patch_text_offset > old_text.size() ||
        candidate.patch_size > old_text.size() - candidate.patch_text_offset) {
      result.errors.emplace_back("ConSan MOI fence record patch site is outside .text");
      return;
    }
    const bool incompatible_overlap =
        std::ranges::any_of(result.staged_text_fragments, [&](const ConSanTextFragment &fragment) {
          const uint64_t fragment_end = fragment.patch.anchor_offset + fragment.patch.original_size;
          const uint64_t site_end = candidate.patch_text_offset + candidate.patch_size;
          return candidate.patch_text_offset < fragment_end &&
                 fragment.patch.anchor_offset < site_end &&
                 (fragment.patch.anchor_offset != candidate.patch_text_offset ||
                  fragment.patch.original_size != candidate.patch_size);
        });
    if (incompatible_overlap) {
      result.warnings.emplace_back("ConSan MOI fence record patch skipped an overlapping patch");
      continue;
    }
    const uint16_t scratch_count =
        fence_record_scratch_count(candidate.communication_lowering_form);
    const ConSanCandidateResourcePlan *existing_plan = resource_plan_for_site(
        result.resource_plans, ConSanResourceSiteKind::Fence, fence_text_offset);
    ConSanCandidateResourcePlan plan =
        existing_plan
            ? *existing_plan
            : plan_moi_resource_site(*resource_state, options, options, operating_point,
                                     ConSanResourceSiteKind::Fence, candidate_index,
                                     candidate.patch_text_offset,
                                     candidate.kernel_descriptor_file_offset, scratch_count,
                                     nullptr, &candidate.communication_lowering_form);
    plan.intent_ids = {candidate.address_capture_intent, candidate.evidence_intent};
    if (!existing_plan) {
      if (candidate.patch_text_offset != fence_text_offset)
        plan.semantic_text_offset = fence_text_offset;
      result.resource_plans.push_back(plan);
    }
    const auto resources =
        resolve_moi_scratch_plan(plan, operating_point, result.moi_operating_point, scratch_count);
    if (!resources) {
      result.warnings.emplace_back("ConSan MOI fence record patch has no usable resource plan");
      continue;
    }
    std::optional<uint32_t> active_private_size;
    if (resources->source == ConSanRegisterAllocationSource::SpillRequired ||
        operating_point.has_compact_moi_scalar_spill()) {
      active_private_size = 0u;
      bool active_descriptor_missing = false;
      for (uint64_t descriptor_offset : resources->owner_descriptor_file_offsets) {
        const ConSanProgramContainer *kernel =
            result.program_inventory.find_kernel_by_descriptor(descriptor_offset);
        const auto active_kernel =
            kernel == nullptr
                ? code_object.kernels().end()
                : std::ranges::find_if(code_object.kernels(), [&](const AmdGpuKernelInfo &item) {
                    return item.name == kernel->name;
                  });
        if (active_kernel == code_object.kernels().end()) {
          result.warnings.emplace_back(
              "ConSan MOI fence record spill could not resolve its active descriptor");
          active_descriptor_missing = true;
          break;
        }
        const auto descriptor =
            read_kernel_descriptor(active_bytes, active_kernel->descriptor_file_offset);
        if (!descriptor) {
          result.warnings.emplace_back(
              "ConSan MOI fence record spill could not resolve its active descriptor");
          active_descriptor_missing = true;
          break;
        }
        *active_private_size =
            std::max(*active_private_size, descriptor->private_segment_fixed_size);
      }
      if (active_descriptor_missing)
        continue;
    }
    const ConSanMoiAtomicAddressPlan address_plan =
        plan_consan_moi_atomic_address(candidate.communication_lowering_form, resources->base,
                                       resources->count, resources->source);
    if (!address_plan.supported()) {
      result.warnings.emplace_back(
          "ConSan MOI fence record patch rejected communication address: " +
          std::string(consan_moi_atomic_address_support_name(address_plan.support)));
      continue;
    }
    if (!operating_point.moi_owner_epoch_vgprs.owner() &&
        !candidate.kernel_descriptor_file_offset) {
      if (resources->owner_descriptor_file_offsets.empty()) {
        result.warnings.emplace_back(
            "ConSan MOI fence record patch requires an owning kernel descriptor");
        continue;
      }
      candidate.kernel_descriptor_file_offset = resources->owner_descriptor_file_offsets.front();
    }
    auto record_event = plan_moi_record_event(
        bytes, result.program_inventory, result.warnings, *resources, options, options,
        operating_point, result.moi_operating_point, mode_semantics, spill_managers, arch,
        "ConSan MOI fence record patch", active_private_size);
    if (!record_event)
      continue;
    note_moi_probe_private_requirements(private_requirements, *record_event);
    note_descriptor_requirements(descriptor_requirements, record_event->resources);
    note_moi_sgpr_requirements(scalar_requirements, record_event->resources, record_event->emission,
                               arch);
    selected.push_back({std::move(*record_event), std::move(candidate), address_plan});
  }
  if (selected.empty()) {
    result.warnings.emplace_back("ConSan MOI fence record patch found no patchable fences");
    return;
  }
  std::vector<ConSanPatchInfo> scalar_clause_patches;
  for (const PlannedFenceRecord &planned : selected) {
    MoiAtomicEvidenceSitePlan scalar_clause_candidate;
    scalar_clause_candidate.site = planned.candidate.communication_site;
    scalar_clause_candidate.scalar_clause_text_offset = planned.candidate.scalar_clause_text_offset;
    if (!append_atomic_scalar_clause_patch(old_text, scalar_clause_candidate, arch,
                                           scalar_clause_patches, result.errors))
      return;
  }
  std::vector<ConSanTextFragment> fragments;
  fragments.reserve(selected.size() + scalar_clause_patches.size());
  for (ConSanPatchInfo &patch : scalar_clause_patches)
    fragments.push_back(ConSanTextFragment::replacement({build_s_nop(0u, arch)}, std::move(patch)));
  uint32_t record_index = 0;
  for (const PlannedFenceRecord &planned : selected) {
    const MoiRecordEventEmissionPlan &candidate_options = planned.emission;
    const uint64_t patch_text_offset = planned.candidate.patch_text_offset;
    const uint32_t patch_size = planned.candidate.patch_size;
    uint32_t guest_instruction_offset = 0u;
    auto cave_words = build_fence_record_cave_words(
        active_bytes, planned.candidate, planned.address_plan, candidate_options,
        planned.spill ? &*planned.spill : nullptr,
        planned.scalar_spill ? &*planned.scalar_spill : nullptr, arch, record_index,
        layout.fence_record_capacity, layout.fence_records_offset,
        /*displaced_tail_words=*/{}, result.errors, &guest_instruction_offset);
    if (!cave_words)
      return;
    ConSanPatchInfo info;
    info.kind = ConSanPatchKind::TrampolineMoiFenceRecord;
    info.anchor_offset = patch_text_offset;
    info.original_size = patch_size;
    note_moi_probe_patch_info(info, planned);
    auto fragment =
        make_consan_around_text_fragment(std::move(*cave_words), guest_instruction_offset,
                                         patch_size, planned.candidate.intent_ids(), {},
                                         std::move(info), result.errors, "MOI fence record patch");
    if (!fragment)
      return;
    if (!planned.candidate.capture_address_before_guest &&
        candidate_options.runtime_workgroup_gate) {
      if (!prepend_moi_runtime_workgroup_gate(*fragment, *candidate_options.runtime_workgroup_gate,
                                              candidate_options.workgroup_sources, arch,
                                              result.errors, "MOI fence record patch"))
        return;
    }
    fragments.push_back(std::move(*fragment));
    ++record_index;
  }
  const MoiDescriptorLdsRequirements no_lds;
  (void)stage_consan_text_rewrites(
      code_object, arch,
      {descriptor_requirements, scalar_requirements, private_requirements, no_lds},
      moi_descriptor_mutation_policy(nullptr, arch), "MOI fence record patch", std::move(fragments),
      result);
}
