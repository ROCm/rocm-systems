void try_apply_atomic_record_patch(std::span<const uint8_t> bytes, const ConSanOptions &options,
                                   rj_code_arch_t arch, ConSanResult &result) {
  if (!options.moi_track_atomics)
    return;
  if (options.moi_engine != ConSanMoiEngine::RecordReplay)
    return;
  if (!instrumentation::is_admitted_arch(arch)) {
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
  const ConSanMoiReportBufferLayout layout =
      resolve_record_replay_report_layout(options, options.moi_track_barriers,
                                          /*include_atomics=*/true, /*include_fences=*/true);
  const uint32_t max_candidates = std::min(options.max_patches, layout.atomic_record_capacity);
  if (max_candidates == 0) {
    result.warnings.emplace_back("ConSan MOI atomic record patch has no atomic record capacity");
    return;
  }
  std::vector<AtomicRecordCandidate> candidates;
  const MoiSyncInventoryIndex sync_index(result);
  auto retain_unavailable_cas = [&](std::string_view container_name, bool in_kernel,
                                    const std::vector<ConSanAtomicSite> &sites) {
    for (const ConSanAtomicSite &site : sites) {
      const bool cas = consan_atomic_is_compare_exchange(site);
      if (!cas || site.returns_old_value.value_or(false) ||
          !atomic_event_kind_for_site(sync_index, container_name, in_kernel, site))
        continue;
      result.warnings.emplace_back(
          "ConSan MOI atomic record patch skipped compare-exchange-outcome-unavailable in " +
          std::string(container_name));
    }
  };
  for (const ConSanKernelInfo &kernel : result.kernels) {
    retain_unavailable_cas(kernel.name, true, kernel.atomic_sites);
    append_atomic_record_candidates(kernel, result, sync_index, candidates);
  }
  for (const ConSanFunctionInfo &function : result.functions) {
    retain_unavailable_cas(function.name, false, function.atomic_sites);
    append_atomic_record_candidates(function, result, sync_index, candidates);
  }
  apply_test_container_filter(candidates, options);
  if (!canonicalize_atomic_record_candidates_by_physical_site(candidates, result))
    return;
  std::sort(candidates.begin(), candidates.end(), [](const auto &lhs, const auto &rhs) {
    return lhs.site.text_offset < rhs.site.text_offset;
  });
  if (candidates.empty()) {
    result.warnings.emplace_back("ConSan MOI atomic record patch found no supported flat atomics");
    return;
  }

  std::span<const uint8_t> active_bytes = bytes;
  if (result.modified)
    active_bytes = std::span<const uint8_t>(result.elf_bytes.data(), result.elf_bytes.size());
  AmdGpuCodeObject code_object(active_bytes.data(), active_bytes.size());
  CodeObjectPatcher patcher(code_object);
  const std::span<const uint8_t> old_text = patcher.text_bytes();
  if (old_text.empty()) {
    result.errors.emplace_back("ConSan MOI atomic record patch found no .text section");
    return;
  }
  struct PlannedAtomicRecord {
    AtomicRecordCandidate candidate;
    ResolvedMoiScratchPlan resources;
    ConSanMoiAtomicAddressPlan address_plan;
    std::optional<VgprSpillSequence> spill;
    std::optional<InlineSgprSpillSequence> scalar_spill;
    std::optional<MoiPrivateRecordOwnerPlan> private_owner;
    std::optional<MoiPrivateEpochLayout> private_layout;
  };
  std::vector<PlannedAtomicRecord> selected_candidates;
  selected_candidates.reserve(max_candidates);
  MoiResourcePlanningState resource_state(bytes, arch, options, result);
  MoiSpillManagers spill_managers;
  MoiDescriptorVgprRequirements descriptor_requirements;
  MoiDescriptorSgprRequirements scalar_requirements;
  MoiDescriptorPrivateRequirements private_requirements;
  for (size_t candidate_index = 0; candidate_index < candidates.size(); ++candidate_index) {
    const AtomicRecordCandidate &candidate = candidates[candidate_index];
    if (selected_candidates.size() == max_candidates)
      break;
    const ConSanAtomicSite &site = candidate.site;
    if (site.text_offset > old_text.size() || site.size > old_text.size() - site.text_offset) {
      result.errors.emplace_back("ConSan MOI atomic record patch site is outside .text");
      return;
    }
    const ConSanPatchInfo *composed_access_patch = nullptr;
    for (const ConSanPatchInfo &patch : result.patches) {
      const bool access_patch = patch.kind == ConSanPatchKind::InlineMoiAccessRecordStore ||
                                patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore;
      if (access_patch && patch.anchor_offset == site.text_offset &&
          patch.relocated_guest_instruction_offset) {
        composed_access_patch = &patch;
        break;
      }
    }
    const auto overlaps_existing_patch = std::any_of(
        result.patches.begin(), result.patches.end(), [&](const ConSanPatchInfo &patch) {
          if (&patch == composed_access_patch)
            return false;
          const uint64_t patch_end =
              patch.anchor_offset + std::max<uint32_t>(patch.original_size, 1);
          return site.text_offset < patch_end && patch.anchor_offset < site.text_offset + site.size;
        });
    if (overlaps_existing_patch) {
      result.warnings.emplace_back(
          "ConSan MOI atomic record patch skipped an overlapping patch in " +
          candidate.container_name);
      continue;
    }
    const ConSanCandidateResourcePlan *existing_plan =
        resource_plan_for_site(result, ConSanResourceSiteKind::Atomic, site.text_offset);
    const uint16_t scratch_count = atomic_record_scratch_count(site, arch);
    ConSanCandidateResourcePlan plan =
        existing_plan
            ? *existing_plan
            : plan_moi_resource_site(resource_state, options, ConSanResourceSiteKind::Atomic,
                                     candidate_index, site.text_offset,
                                     candidate.kernel_descriptor_file_offset, scratch_count,
                                     nullptr, &site);
    if (!existing_plan)
      result.resource_plans.push_back(plan);
    const auto resources = resolve_moi_scratch_plan(plan, options, scratch_count);
    if (!resources) {
      result.warnings.emplace_back(
          "ConSan MOI atomic record patch has no usable resource plan in " +
          candidate.container_name);
      continue;
    }
    const ConSanMoiAtomicAddressPlan address_plan = plan_consan_moi_atomic_address(
        site, resources->base, resources->count, resources->source, arch);
    if (!address_plan.supported()) {
      result.warnings.emplace_back(
          "ConSan MOI atomic record patch has no usable address plan in " +
          candidate.container_name + ": " +
          std::string(consan_moi_atomic_address_support_name(address_plan.support)));
      continue;
    }
    AtomicRecordCandidate planned_candidate = candidate;
    if (!options.moi_owner_vgpr && !planned_candidate.kernel_descriptor_file_offset) {
      planned_candidate.kernel_descriptor_file_offset =
          common_moi_record_owner_descriptor(bytes, *resources, arch, result.warnings);
      if (!planned_candidate.kernel_descriptor_file_offset)
        continue;
    }
    ConSanOptions candidate_options = options;
    candidate_options.scratch_vgpr = resources->base;
    std::optional<MoiPrivateEpochLayout> private_layout;
    if (options.automatic_moi_private_epoch) {
      private_layout = build_moi_private_epoch_layout(result, *resources, arch, result.warnings,
                                                      /*include_owner=*/options.moi_owner_source ==
                                                          ConSanMoiOwnerSource::WorkitemId,
                                                      /*include_workgroup_key=*/false,
                                                      /*include_record_replay_workgroup=*/true);
      if (!private_layout)
        continue;
      candidate_options.moi_record_replay_workgroup_private_offsets =
          private_layout->record_replay_workgroup_offsets;
    }
    if (!apply_moi_transient_sgpr_assignment(candidate_options,
                                             resources->owner_descriptor_file_offsets)) {
      result.warnings.emplace_back(
          "ConSan MOI atomic record patch found no common transient SGPR assignment");
      continue;
    }
    if (!apply_record_replay_entry_workgroup_assignment(candidate_options,
                                                        resources->owner_descriptor_file_offsets)) {
      result.warnings.emplace_back(
          "ConSan MOI atomic record patch found no common entry workgroup assignment");
      continue;
    }
    std::optional<MoiPrivateRecordOwnerPlan> private_owner;
    if (moi_record_uses_private_owner(candidate_options)) {
      private_owner =
          resolve_moi_private_record_owner(bytes, *resources, *private_layout, result.warnings);
      if (!private_owner)
        continue;
    }
    std::optional<uint32_t> private_layout_base;
    // Keep this as a separate assignment: GCC 13 otherwise diagnoses the
    // equivalent conditional optional construction as maybe-uninitialized.
    if (private_layout)
      private_layout_base = private_layout->ephemeral_base;
    std::optional<VgprSpillSequence> spill;
    if (resources->source == ConSanRegisterAllocationSource::SpillRequired ||
        moi_scalar_spill_requires_dynamic_vgpr_frame(result, *resources, candidate_options)) {
      spill = build_moi_spill_sequence(result, *resources, candidate_options, spill_managers, arch,
                                       result.warnings, private_layout_base);
      if (!spill)
        continue;
    }
    std::optional<InlineSgprSpillSequence> scalar_spill;
    if (candidate_options.automatic_moi_record_replay_sgpr_spill) {
      scalar_spill = build_inline_sgpr_spill_sequence(
          result, *resources, candidate_options, spill_managers, arch, result.warnings,
          private_layout_base, spill ? &*spill : nullptr);
      if (!scalar_spill)
        continue;
    }
    selected_candidates.push_back({std::move(planned_candidate), *resources, address_plan,
                                   std::move(spill), std::move(scalar_spill), private_owner,
                                   private_layout});
  }
  // Do not prune a release based on module-local absence of an acquire. Real
  // applications can split producers and consumers across code objects, and
  // Record/Replay must preserve every admitted access in its event stream.
  if (selected_candidates.empty()) {
    result.warnings.emplace_back("ConSan MOI atomic record patch found no patchable atomics");
    return;
  }
  for (const PlannedAtomicRecord &planned : selected_candidates) {
    if (planned.spill) {
      note_spill_descriptor_requirements(private_requirements, planned.resources, *planned.spill);
    }
    if (planned.scalar_spill) {
      for (uint64_t descriptor_offset : planned.resources.owner_descriptor_file_offsets) {
        auto [it, inserted] = private_requirements.emplace(
            descriptor_offset, planned.scalar_spill->total_private_bytes);
        if (!inserted)
          it->second = std::max(it->second, planned.scalar_spill->total_private_bytes);
      }
    }
    if (planned.private_layout) {
      for (uint64_t descriptor_offset : planned.resources.owner_descriptor_file_offsets) {
        auto [it, inserted] =
            private_requirements.emplace(descriptor_offset, planned.private_layout->ephemeral_base);
        if (!inserted)
          it->second = std::max(it->second, planned.private_layout->ephemeral_base);
      }
    }
    note_descriptor_requirements(descriptor_requirements, planned.resources);
    ConSanOptions candidate_options = options;
    if (planned.private_layout) {
      candidate_options.moi_record_replay_workgroup_private_offsets =
          planned.private_layout->record_replay_workgroup_offsets;
    }
    if (apply_moi_transient_sgpr_assignment(candidate_options,
                                            planned.resources.owner_descriptor_file_offsets) &&
        apply_record_replay_entry_workgroup_assignment(
            candidate_options, planned.resources.owner_descriptor_file_offsets)) {
      note_moi_sgpr_requirements(scalar_requirements, planned.resources, candidate_options, arch);
    }
  }
  if (!apply_descriptor_requirements(patcher, code_object, active_bytes, result,
                                     descriptor_requirements, arch, result.errors) ||
      !apply_sgpr_descriptor_requirements(patcher, result, code_object, scalar_requirements,
                                          result.errors) ||
      !apply_spill_descriptor_requirements(patcher, code_object, active_bytes, result,
                                           private_requirements, result.errors)) {
    return;
  }

  std::vector<uint8_t> new_text(old_text.begin(), old_text.end());
  DbiPatchPlacementPlanner placement_planner(arch, old_text.size());
  MoiLocalNopIslandAllocator indirect_allocator(old_text, result, arch,
                                                kMoiLocalIndirectIslandWords);
  std::vector<ConSanPatchInfo> patches;
  uint32_t record_index = 0;
  for (const PlannedAtomicRecord &planned : selected_candidates) {
    const AtomicRecordCandidate &candidate = planned.candidate;
    const ConSanAtomicSite &site = candidate.site;
    const auto composed_access_patch = std::ranges::find_if(result.patches, [&](const auto &patch) {
      return (patch.kind == ConSanPatchKind::InlineMoiAccessRecordStore ||
              patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore) &&
             patch.anchor_offset == site.text_offset && patch.relocated_guest_instruction_offset;
    });
    const uint64_t physical_anchor_offset =
        composed_access_patch == result.patches.end()
            ? site.text_offset
            : *composed_access_patch->relocated_guest_instruction_offset;
    const uint64_t cave_text_offset = placement_planner.appended_end();
    const uint64_t return_text_offset = physical_anchor_offset + site.size;
    ConSanOptions candidate_options = options;
    candidate_options.scratch_vgpr = planned.resources.base;
    if (planned.private_layout) {
      candidate_options.moi_record_replay_workgroup_private_offsets =
          planned.private_layout->record_replay_workgroup_offsets;
    }
    if (!apply_moi_transient_sgpr_assignment(candidate_options,
                                             planned.resources.owner_descriptor_file_offsets)) {
      result.errors.emplace_back(
          "ConSan MOI atomic record patch lost its transient SGPR assignment");
      return;
    }
    if (!apply_record_replay_entry_workgroup_assignment(
            candidate_options, planned.resources.owner_descriptor_file_offsets)) {
      result.errors.emplace_back(
          "ConSan MOI atomic record patch lost its entry workgroup assignment");
      return;
    }
    uint32_t guest_instruction_offset = 0;
    auto cave_words = build_atomic_record_cave_words(
        active_bytes, candidate, planned.address_plan, candidate_options,
        planned.spill ? &*planned.spill : nullptr,
        planned.scalar_spill ? &*planned.scalar_spill : nullptr, arch, record_index,
        layout.atomic_record_capacity, layout.atomic_records_offset, cave_text_offset,
        return_text_offset, composed_access_patch != result.patches.end(),
        planned.private_owner ? std::optional<uint32_t>(planned.private_owner->offset)
                              : std::nullopt,
        planned.private_owner ? std::optional<uint16_t>(planned.private_owner->shift)
                              : std::nullopt,
        guest_instruction_offset, result.errors);
    if (!cave_words)
      return;

    const bool direct_entry =
        compute_sopp_branch_simm16(physical_anchor_offset, cave_text_offset).has_value();
    uint64_t indirect_entry_island = 0;
    bool has_indirect_entry_island = false;
    if (!direct_entry) {
      const auto claimed_island = indirect_allocator.claim_reachable(physical_anchor_offset);
      if (!claimed_island) {
        result.warnings.emplace_back(
            "ConSan MOI atomic record patch found no reachable indirect entry island in " +
            candidate.container_name);
        continue;
      }
      indirect_entry_island = *claimed_island;
      has_indirect_entry_island = true;
    }
    std::string placement_error;
    const auto placement = placement_planner.plan_indirect_appended(
        physical_anchor_offset, site.size, cave_words->size() * sizeof(uint32_t), &placement_error);
    if (!placement) {
      result.errors.emplace_back("ConSan MOI atomic record placement failed: " + placement_error);
      return;
    }
    if (placement->body_offset != cave_text_offset || new_text.size() != placement->body_offset) {
      result.errors.emplace_back("ConSan MOI atomic record patch has a stale cave mapping");
      return;
    }

    const uint64_t entry_target =
        has_indirect_entry_island ? indirect_entry_island : placement->body_offset;
    const auto fwd = compute_sopp_branch_simm16(placement->anchor_offset, entry_target);
    if (!fwd) {
      result.errors.emplace_back("ConSan MOI atomic record forward branch is out of range");
      return;
    }
    std::vector<uint32_t> anchor_words(site.size / sizeof(uint32_t), build_s_nop(0, arch));
    if (anchor_words.empty()) {
      result.errors.emplace_back("ConSan MOI atomic record patch empty anchor");
      return;
    }
    anchor_words.front() = build_s_branch(*fwd, arch);
    if (physical_anchor_offset > new_text.size() ||
        anchor_words.size() * sizeof(uint32_t) > new_text.size() - physical_anchor_offset) {
      result.errors.emplace_back("ConSan MOI atomic record anchor exceeds .text");
      return;
    }
    std::memcpy(new_text.data() + physical_anchor_offset, anchor_words.data(),
                anchor_words.size() * sizeof(uint32_t));
    if (has_indirect_entry_island &&
        !emit_moi_local_indirect_entry_island(
            new_text, indirect_entry_island, placement->body_offset, physical_anchor_offset,
            options, planned.resources.owner_descriptor_file_offsets, arch, patches, result.errors,
            "ConSan MOI atomic record patch")) {
      return;
    }
    append_words_bytes(new_text, *cave_words);

    ConSanPatchInfo info;
    info.kind = ConSanPatchKind::TrampolineMoiAtomicRecord;
    info.anchor_offset = site.text_offset;
    info.trampoline_offset = cave_text_offset;
    info.original_size = site.size;
    info.trampoline_size = static_cast<uint32_t>(cave_words->size() * sizeof(uint32_t));
    info.relocated_guest_instruction_offset = cave_text_offset + guest_instruction_offset;
    info.scratch_vgpr = planned.resources.base;
    info.owner_descriptor_file_offsets = planned.resources.owner_descriptor_file_offsets;
    if (planned.private_layout) {
      info.persistent_epoch_private_offset = planned.private_layout->epoch_offset;
      info.persistent_owner_private_offset = planned.private_layout->owner_offset;
      info.persistent_record_replay_workgroup_private_offsets =
          planned.private_layout->record_replay_workgroup_offsets;
      info.persistent_private_state_end = planned.private_layout->ephemeral_base;
      info.required_private_segment_size = planned.private_layout->ephemeral_base;
    }
    if (planned.spill) {
      info.spilled_vgpr_count = planned.spill->vgpr_count;
      info.required_private_segment_size =
          std::max(info.required_private_segment_size, planned.spill->total_private_bytes);
      note_dynamic_stack_private_requirement(info, &*planned.spill);
    }
    if (planned.scalar_spill) {
      info.required_private_segment_size =
          std::max(info.required_private_segment_size, planned.scalar_spill->total_private_bytes);
    }
    patches.push_back(info);
    ++record_index;
  }

  if (!replace_consan_text(patcher, new_text, options, "MOI atomic record patch", result)) {
    return;
  }
  result.elf_bytes = std::move(patcher).emit();
  result.patches.insert(result.patches.end(), patches.begin(), patches.end());
  result.modified = true;
}
