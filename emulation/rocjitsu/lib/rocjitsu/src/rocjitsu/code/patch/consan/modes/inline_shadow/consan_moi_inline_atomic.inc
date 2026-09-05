[[nodiscard]] std::optional<MoiInlineAtomicEmissionPlan> plan_inline_atomic_emission(
    std::span<const uint8_t> bytes, const MoiAtomicEvidenceSitePlan &candidate,
    const ConSanOptions &options, const ConSanMoiOperatingPoint &point,
    const ConSanMoiOperatingPoint &allocation, const MoiObjectModeSemantics &semantics,
    uint16_t scratch_vgpr, uint16_t scratch_count,
    const ConSanMoiPrivateStateLayout *private_layout, const MoiTargetFacts &target_facts,
    rj_code_arch_t arch, std::vector<std::string> &errors) {
  if (!options.moi_report_buffer_address || !point.moi_exec_save_sgpr ||
      !candidate.kernel_descriptor_file_offset)
    return std::nullopt;
  const MoiScalarAbiPlan scalar_abi =
      plan_moi_scalar_abi(options.moi_engine, moi_scalar_preservation_state(point));
  const auto workgroup_sources = moi_descriptor_workgroup_sources(
      bytes, *candidate.kernel_descriptor_file_offset, arch, errors);
  if (!scalar_abi.special_state || !workgroup_sources)
    return std::nullopt;
  if (!validate_inline_atomic_exec_save_sgpr(point.moi_exec_save_sgpr, errors) ||
      reject_optional_scratch_range_overlap(point.moi_owner_epoch_vgprs.owner(), scratch_vgpr,
                                            scratch_count, "MOI owner", errors) ||
      reject_optional_scratch_range_overlap(point.moi_owner_epoch_vgprs.epoch(), scratch_vgpr,
                                            scratch_count, "MOI epoch", errors) ||
      reject_optional_scratch_range_overlap(point.moi_workgroup_key_vgpr, scratch_vgpr,
                                            scratch_count, "MOI workgroup key", errors))
    return std::nullopt;

  const uint16_t required_scratch_count =
      inline_shadow_atomic_scratch_vgpr_count(candidate.lowering_form, target_facts);
  ConSanMoiOwnerEpochRegisterState owner_epoch_vgprs = point.moi_owner_epoch_vgprs;
  consan_detail::MoiWorkgroupKeyRegisterPlan workgroup_key_registers =
      moi_workgroup_key_register_plan(point);
  std::optional<MoiInlineAtomicPrivateStatePlan> private_state;
  if (point.moi_persistent_sgprs.complete()) {
    const uint16_t owner = static_cast<uint16_t>(scratch_vgpr + required_scratch_count - 4u);
    owner_epoch_vgprs.set_owner_epoch(owner, static_cast<uint16_t>(owner + 1u));
  }
  if (private_layout) {
    if (!private_layout->workgroup_key_offset) {
      errors.emplace_back("ConSan MOI inline atomic patch has an incomplete private state plan");
      return std::nullopt;
    }
    const uint16_t workgroup_key =
        static_cast<uint16_t>(scratch_vgpr + required_scratch_count - 6u);
    const uint16_t owner = static_cast<uint16_t>(scratch_vgpr + required_scratch_count - 4u);
    owner_epoch_vgprs.set_owner_epoch(owner, static_cast<uint16_t>(owner + 1u));
    workgroup_key_registers.cached_key_vgpr = workgroup_key;
    workgroup_key_registers.cached_key_sgpr.reset();

    MoiInlineAtomicPrivateStatePlan state{
        .epoch_offset = private_layout->epoch_offset,
        .workgroup_key_offset = *private_layout->workgroup_key_offset,
        .workitem_owner = std::nullopt,
        .resident_wave_owner = std::nullopt,
        .resident_wave_owner_is_borrowed = false,
        .dispatch_id_offset = private_layout->dispatch_id_offset,
        .dispatch_id_sgpr =
            private_layout->dispatch_id_offset ? point.moi_dispatch_identity.sgpr() : std::nullopt,
    };
    if (options.moi_owner_source == ConSanMoiOwnerSource::WorkitemId) {
      const auto owner_shift =
          moi_descriptor_owner_shift(bytes, *candidate.kernel_descriptor_file_offset, arch, errors);
      if (!private_layout->owner_offset || !owner_shift)
        return std::nullopt;
      state.workitem_owner = consan_detail::MoiWorkitemOwnerDerivationPlan{
          .entry_workitem_x_private_offset = private_layout->owner_offset,
          .wave_size_shift = *owner_shift,
      };
    } else if (options.moi_owner_source == ConSanMoiOwnerSource::HwId &&
               point.moi_owner_sgpr.base()) {
      state.resident_wave_owner = consan_detail::MoiResidentWaveOwnerRequest{
          .destination_sgpr = *point.moi_owner_sgpr.base(),
          .one_based = true,
      };
      state.resident_wave_owner_is_borrowed = point.moi_owner_sgpr.automatic();
    } else {
      errors.emplace_back("ConSan MOI inline atomic patch has no private owner derivation");
      return std::nullopt;
    }
    if (state.dispatch_id_offset && !state.dispatch_id_sgpr) {
      errors.emplace_back("ConSan MOI inline atomic patch has no private dispatch reload pair");
      return std::nullopt;
    }
    private_state = std::move(state);
  }
  if (!owner_epoch_vgprs.complete() || !workgroup_key_registers.is_well_formed())
    return std::nullopt;
  const auto dispatch_id =
      consan_moi_detail::moi_target_dispatch_id_sources({point, options}, arch);
  if (!dispatch_id.is_well_formed())
    return std::nullopt;

  const ConSanMoiReportBufferLayout &layout = semantics.report_layout;
  MoiInlineAtomicEmissionPlan plan{
      .report_buffer_address = *options.moi_report_buffer_address,
      .report_layout = layout,
      .inline_access_present = semantics.inline_access_present,
      .scratch_vgpr = scratch_vgpr,
      .scratch_vgpr_count = required_scratch_count,
      .requires_aligned_flat_compare_swap_data_pair =
          target_facts.requires_aligned_flat_compare_swap_data_pair,
      .exec_save_sgpr = *point.moi_exec_save_sgpr,
      .owner_epoch_vgprs = *owner_epoch_vgprs.operator->(),
      .persistent_sgprs = point.moi_persistent_sgprs,
      .special_state = *scalar_abi.special_state,
      .workgroup_key_registers = workgroup_key_registers,
      .workgroup_sources = *workgroup_sources,
      .dispatch_id = dispatch_id,
      .private_state = std::move(private_state),
      .patch_abi = {},
  };
  plan.patch_abi.moi_vgpr_state = ConSanMoiVgprStateEffect{
      .owner_epoch = *owner_epoch_vgprs.operator->(),
      .workgroup_key = workgroup_key_registers.cached_key_vgpr,
      .exact_workgroup = point.moi_exact_workgroup_vgprs,
      .owner_epoch_lifetime = consan_moi_owner_epoch_vgpr_lifetime(
          private_layout || point.moi_persistent_sgprs.complete(),
          !allocation.owner_persistent_vgprs.empty()),
  };
  plan.patch_abi.persistent_sgpr_state = point.moi_persistent_sgprs;
  return plan;
}

void try_apply_inline_atomic_ordering_patch(std::span<const uint8_t> bytes,
                                            const ConSanOptions &options,
                                            const ConSanMoiOperatingPoint &operating_point,
                                            rj_code_arch_t arch,
                                            const MoiObjectModeSemantics &semantics,
                                            ConSanTransformArtifacts &result) {
  if (!options.moi_track_atomics)
    return;
  const ConSanTargetProfile *target = consan_target_profile(arch);
  if (target == nullptr) {
    result.warnings.emplace_back(
        "ConSan MOI inline atomic patch does not support this architecture");
    return;
  }
  const MoiTargetFacts target_facts = resolve_moi_target_facts(arch);
  if (!options.moi_report_buffer_address) {
    result.warnings.emplace_back("ConSan MOI inline atomic patch requires a MOI report buffer");
    return;
  }
  const ConSanMoiReportBufferLayout &layout = semantics.report_layout;
  if (layout.inline_atomic_release_capacity == 0 ||
      layout.inline_acquired_epoch_token_capacity == 0) {
    result.warnings.emplace_back(
        "ConSan MOI inline atomic patch has no release or acquired-token metadata slot");
    return;
  }
  if (!operating_point.moi_owner_epoch_vgprs.complete() &&
      !operating_point.moi_persistent_sgprs.complete() &&
      !operating_point.automatic_moi_private_epoch) {
    result.warnings.emplace_back("ConSan MOI inline atomic patch requires "
                                 "RJ_CONSAN_MOI_OWNER_VGPR and RJ_CONSAN_MOI_EPOCH_VGPR");
    return;
  }
  if (!validate_inline_atomic_exec_save_sgpr(operating_point.moi_exec_save_sgpr, result.warnings))
    return;

  std::vector<MoiAtomicEvidenceSitePlan> candidates = build_moi_atomic_evidence_site_plans(
      result.program_inventory, result.observation_plan(),
      ConSanProbeIntentKind::ExactAtomicOrdering, result.errors);
  if (!result.errors.empty())
    return;
  std::sort(candidates.begin(), candidates.end(), [](const auto &lhs, const auto &rhs) {
    return lhs.site.text_offset < rhs.site.text_offset;
  });
  if (candidates.empty()) {
    result.warnings.emplace_back("ConSan MOI inline atomic patch found no supported flat atomics");
    return;
  }

  const std::span<const uint8_t> active_bytes = result.active_bytes(bytes);
  AmdGpuCodeObject code_object(active_bytes.data(), active_bytes.size());
  CodeObjectPatcher patcher(code_object);
  const std::span<const uint8_t> old_text = patcher.text_bytes();
  if (old_text.empty()) {
    result.errors.emplace_back("ConSan MOI inline atomic patch found no .text section");
    return;
  }
  struct PlannedInlineAtomic : MoiPlannedProbeResources {
    MoiAtomicEvidenceSitePlan candidate;
    ConSanMoiAtomicAddressPlan address_plan;
    MoiInlineAtomicEmissionPlan emission;
    uint32_t relocated_guest_size = 0;
    std::vector<uint32_t> trailing_guest_words;
  };
  std::vector<PlannedInlineAtomic> planned_candidates;
  MoiResourcePlanningStatePtr resource_state = make_moi_resource_planning_state(
      bytes, arch, result.program_inventory, result.observation_plan(), result.resource_plans,
      options, options, operating_point);
  MoiSpillManagers spill_managers;
  MoiDescriptorVgprRequirements descriptor_requirements;
  MoiDescriptorSgprRequirements scalar_requirements;
  MoiDescriptorPrivateRequirements private_requirements;
  for (size_t candidate_index = 0; candidate_index < candidates.size(); ++candidate_index) {
    if (planned_candidates.size() == options.max_patches)
      break;
    const MoiAtomicEvidenceSitePlan &candidate = candidates[candidate_index];
    const ConSanAtomicSite &site = candidate.site;
    uint32_t relocated_guest_size = site.size;
    std::vector<uint32_t> trailing_guest_words;
    if (!candidate.is_rmw &&
        candidate.ordered_sequence_end_text_offset > site.text_offset + site.size) {
      const uint64_t sequence_end = candidate.ordered_sequence_end_text_offset;
      const uint64_t sequence_size = sequence_end - site.text_offset;
      if (!moi_resource_offsets_share_block(*resource_state, site.text_offset,
                                            sequence_end - sizeof(uint32_t)) ||
          sequence_size > std::numeric_limits<uint32_t>::max() ||
          sequence_size % sizeof(uint32_t) != 0u || sequence_end > old_text.size()) {
        result.warnings.emplace_back(
            "ConSan MOI inline atomic patch could not relocate an ordinary acquire suffix in " +
            candidate.container_name);
        continue;
      }
      relocated_guest_size = static_cast<uint32_t>(sequence_size);
      trailing_guest_words.resize((relocated_guest_size - site.size) / sizeof(uint32_t));
      std::memcpy(trailing_guest_words.data(), old_text.data() + site.text_offset + site.size,
                  trailing_guest_words.size() * sizeof(uint32_t));
    }
    if (site.text_offset > old_text.size() ||
        relocated_guest_size > old_text.size() - site.text_offset) {
      result.errors.emplace_back("ConSan MOI inline atomic patch site is outside .text");
      return;
    }
    const auto overlaps_existing_patch = std::any_of(
        result.patches.begin(), result.patches.end(), [&](const ConSanPatchLoweringProduct &patch) {
          const bool exact_atomic_mutation =
              patch.phase == ConSanPatchPhase::Mutation &&
              patch.kind == ConSanPatchKind::InlineAtomicAddressRewrite &&
              patch.anchor_offset == site.text_offset && patch.original_size == site.size;
          if (exact_atomic_mutation)
            return false;
          const uint64_t patch_end =
              patch.anchor_offset + std::max<uint32_t>(patch.original_size, 1);
          return site.text_offset < patch_end &&
                 patch.anchor_offset < site.text_offset + relocated_guest_size;
        });
    if (overlaps_existing_patch) {
      result.warnings.emplace_back(
          "ConSan MOI inline atomic patch skipped an overlapping patch in " +
          candidate.container_name);
      continue;
    }
    const ConSanCandidateResourcePlan *existing_plan = resource_plan_for_site(
        result.resource_plans, ConSanResourceSiteKind::Atomic, site.text_offset);
    const uint16_t scratch_count =
        inline_shadow_atomic_scratch_vgpr_count(candidate.lowering_form, target_facts);
    if (existing_plan && existing_plan->scratch_vgpr_count != scratch_count)
      existing_plan = nullptr;
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
          "ConSan MOI inline atomic patch has no usable resource plan in " +
          candidate.container_name);
      continue;
    }
    const ConSanMoiAtomicAddressPlan address_plan = plan_consan_moi_atomic_address(
        candidate.lowering_form, resources->base, resources->count, resources->source);
    if (!address_plan.supported()) {
      result.warnings.emplace_back(
          "ConSan MOI inline atomic patch has no usable address plan in " +
          candidate.container_name + ": " +
          std::string(consan_moi_atomic_address_support_name(address_plan.support)) +
          " raw_saddr=" +
          (site.raw_saddr ? std::to_string(*site.raw_saddr) : std::string("missing")) +
          " saddr_sgpr=" +
          (site.scalar_address_sgpr ? std::to_string(*site.scalar_address_sgpr)
                                    : std::string("missing")) +
          " raw_vaddr=" +
          (site.raw_vaddr ? std::to_string(*site.raw_vaddr) : std::string("missing")) +
          " addr_vgpr=" +
          (site.address_vgpr ? std::to_string(*site.address_vgpr) : std::string("missing")));
      continue;
    }
    MoiAtomicEvidenceSitePlan planned_candidate = candidate;
    if (!planned_candidate.kernel_descriptor_file_offset) {
      planned_candidate.kernel_descriptor_file_offset =
          common_moi_record_owner_descriptor(bytes, *resources, arch, result.warnings);
      if (!planned_candidate.kernel_descriptor_file_offset)
        continue;
    }
    ConSanMoiOperatingPoint candidate_point = operating_point;
    if (!apply_moi_transient_sgpr_assignment(options, candidate_point, result.moi_operating_point,
                                             resources->owner_descriptor_file_offsets)) {
      result.warnings.emplace_back(
          "ConSan MOI inline atomic patch found no common transient SGPR assignment");
      continue;
    }
    if (!result.moi_operating_point.owner_persistent_vgprs.empty() &&
        !apply_moi_persistent_vgpr_assignment(candidate_point, result.moi_operating_point,
                                              resources->owner_descriptor_file_offsets)) {
      result.warnings.emplace_back(
          "ConSan MOI inline atomic patch found no common persistent VGPR assignment");
      continue;
    }
    std::optional<ConSanMoiPrivateStateLayout> private_layout;
    if (candidate_point.automatic_moi_private_epoch) {
      private_layout = build_moi_private_state_layout(
          result.program_inventory, *resources, arch, result.warnings,
          {.owner = options.moi_owner_source == ConSanMoiOwnerSource::WorkitemId,
           .workgroup_key = true,
           .dispatch_id = candidate_point.moi_dispatch_identity.private_fallback()});
      if (!private_layout)
        continue;
      if (private_layout->dispatch_id_offset) {
        if (!candidate_point.moi_scalar_spill_setup) {
          result.warnings.emplace_back(
              "ConSan MOI inline atomic private dispatch identity has no owner-local reload pair");
          continue;
        }
        candidate_point.moi_dispatch_identity.set_sgpr(
            candidate_point.moi_scalar_spill_setup->temporaries.frame_base_sgpr);
      }
    }
    if (candidate_point.has_inline_moi_scalar_spill()) {
      if (!candidate_point.moi_exec_save_sgpr) {
        result.warnings.emplace_back(
            "ConSan MOI inline atomic spill-backed scalar state has no SGPR window in " +
            candidate.container_name);
        continue;
      }
      const uint16_t scalar_base = *candidate_point.moi_exec_save_sgpr;
      const uint16_t scalar_count = kConSanMoiInlineExecSaveSgprCount;
      if (inline_atomic_scalar_spill_aliases_guest_address(address_plan, scalar_base,
                                                           scalar_count)) {
        result.warnings.emplace_back(
            "ConSan MOI inline atomic spill-backed scalar window aliases a guest atomic "
            "address input in " +
            candidate.container_name);
        continue;
      }
    }
    auto emission = plan_inline_atomic_emission(
        active_bytes, planned_candidate, options, candidate_point, result.moi_operating_point,
        semantics, resources->base, address_plan.scratch_vgpr_count,
        private_layout ? &*private_layout : nullptr, target_facts, arch, result.errors);
    if (!emission)
      return;
    auto probe = plan_moi_probe_resources(
        result.program_inventory, *resources, options, options, candidate_point, semantics,
        spill_managers, arch, std::move(private_layout),
        candidate_point.has_inline_moi_scalar_spill(), result.warnings);
    if (!probe)
      continue;
    note_moi_probe_private_requirements(private_requirements, *probe);
    note_descriptor_requirements(descriptor_requirements, probe->resources);
    note_moi_sgpr_requirements(scalar_requirements, probe->resources, options, options,
                               candidate_point, semantics, arch);
    planned_candidates.push_back({std::move(*probe), std::move(planned_candidate), address_plan,
                                  std::move(*emission), relocated_guest_size,
                                  std::move(trailing_guest_words)});
  }
  // Retain every admitted release even when this code object contains no
  // acquire. Its consumer can live in another large code object, and pruning
  // here would make instrumentation depend on an incomplete module-local
  // view of the program.
  if (planned_candidates.empty()) {
    result.warnings.emplace_back("ConSan MOI inline atomic patch found no patchable atomics");
    return;
  }
  std::vector<ConSanPatchInfo> scalar_clause_patches;
  for (const PlannedInlineAtomic &planned : planned_candidates) {
    if (!append_atomic_scalar_clause_patch(old_text, planned.candidate, arch, scalar_clause_patches,
                                           result.errors))
      return;
  }

  std::vector<ConSanTextFragment> fragments;
  fragments.reserve(planned_candidates.size() + scalar_clause_patches.size());
  for (ConSanPatchInfo &patch : scalar_clause_patches)
    fragments.push_back(ConSanTextFragment::replacement({build_s_nop(0u, arch)}, std::move(patch)));

  for (const PlannedInlineAtomic &planned : planned_candidates) {
    const MoiAtomicEvidenceSitePlan &candidate = planned.candidate;
    const ConSanAtomicSite &site = candidate.site;
    uint32_t guest_instruction_offset = 0u;
    auto words = build_inline_atomic_ordering_cave_words(
        active_bytes, candidate, planned.address_plan, planned.emission,
        planned.spill ? &*planned.spill : nullptr,
        planned.scalar_spill ? &*planned.scalar_spill : nullptr, arch, guest_instruction_offset,
        result.errors, planned.trailing_guest_words);
    if (!words)
      return;

    ConSanPatchInfo patch;
    patch.kind = ConSanPatchKind::TrampolineMoiInlineAtomicOrdering;
    patch.anchor_offset = site.text_offset;
    patch.original_size = planned.relocated_guest_size;
    static_cast<ConSanPatchAbiEffects &>(patch) = planned.emission.patch_abi;
    note_moi_probe_patch_info(patch, planned);
    auto fragment = make_consan_around_text_fragment(
        std::move(*words), guest_instruction_offset, planned.relocated_guest_size,
        candidate.intent_ids(), {}, std::move(patch), result.errors, "MOI inline atomic patch");
    if (!fragment)
      return;
    fragments.push_back(std::move(*fragment));
  }

  const MoiDescriptorLdsRequirements no_lds;
  (void)stage_consan_text_rewrites(
      code_object, arch,
      {descriptor_requirements, scalar_requirements, private_requirements, no_lds},
      moi_descriptor_mutation_policy(nullptr, arch), "MOI inline atomic patch",
      std::move(fragments), result);
}
