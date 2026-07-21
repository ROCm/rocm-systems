void try_apply_inline_atomic_ordering_patch(std::span<const uint8_t> bytes,
                                            const ConSanOptions &options, rj_code_arch_t arch,
                                            ConSanResult &result) {
  if (!options.moi_track_atomics || options.moi_engine != ConSanMoiEngine::InlineShadow)
    return;
  if (!is_rdna4_family_arch(arch) && arch != ROCJITSU_CODE_ARCH_CDNA4) {
    result.warnings.emplace_back(
        "ConSan MOI inline atomic patch does not support this architecture");
    return;
  }
  if (!options.moi_report_buffer_address) {
    result.warnings.emplace_back("ConSan MOI inline atomic patch requires a MOI report buffer");
    return;
  }
  const ConSanMoiReportBufferLayout layout = resolve_inline_report_layout(options);
  if (layout.inline_atomic_release_capacity == 0 ||
      layout.inline_acquired_epoch_token_capacity == 0) {
    result.warnings.emplace_back(
        "ConSan MOI inline atomic patch has no release or acquired-token metadata slot");
    return;
  }
  if ((!options.moi_owner_vgpr || !options.moi_epoch_vgpr) &&
      (!options.moi_persistent_owner_sgpr || !options.moi_persistent_epoch_sgpr)) {
    result.warnings.emplace_back("ConSan MOI inline atomic patch requires "
                                 "RJ_CONSAN_MOI_OWNER_VGPR and RJ_CONSAN_MOI_EPOCH_VGPR");
    return;
  }
  if (!validate_inline_atomic_exec_save_sgpr(options, result.warnings))
    return;

  std::vector<AtomicRecordCandidate> candidates;
  for (const ConSanKernelInfo &kernel : result.kernels)
    append_inline_atomic_candidates(kernel, result, candidates);
  for (const ConSanFunctionInfo &function : result.functions)
    append_inline_atomic_candidates(function, result, candidates);
  apply_test_container_filter(candidates, options);
  candidates.erase(
      std::remove_if(candidates.begin(), candidates.end(),
                     [&](const auto &candidate) {
                       const ConSanMoiInlineAtomicSupport support =
                           classify_consan_moi_inline_atomic_support(candidate.site,
                                                                     candidate.event_kind);
                       if (support == ConSanMoiInlineAtomicSupport::Supported)
                         return false;
                       result.warnings.emplace_back(
                           "ConSan MOI inline atomic skipped " + candidate.site.mnemonic +
                           " at text offset " + std::to_string(candidate.site.text_offset) + ": " +
                           std::string(consan_moi_inline_atomic_support_name(support)));
                       return true;
                     }),
      candidates.end());
  std::sort(candidates.begin(), candidates.end(), [](const auto &lhs, const auto &rhs) {
    return lhs.site.text_offset < rhs.site.text_offset;
  });
  if (candidates.empty()) {
    result.warnings.emplace_back("ConSan MOI inline atomic patch found no supported flat atomics");
    return;
  }
  if (options.automatic_moi_inline_sgpr_spill) {
    result.warnings.emplace_back(
        "ConSan MOI inline atomic patch does not yet support spill-backed scalar state");
    return;
  }

  std::span<const uint8_t> active_bytes = bytes;
  if (result.modified)
    active_bytes = std::span<const uint8_t>(result.elf_bytes.data(), result.elf_bytes.size());
  AmdGpuCodeObject code_object(active_bytes.data(), active_bytes.size());
  CodeObjectPatcher patcher(code_object);
  const std::span<const uint8_t> old_text = patcher.text_bytes();
  if (old_text.empty()) {
    result.errors.emplace_back("ConSan MOI inline atomic patch found no .text section");
    return;
  }
  struct PlannedInlineAtomic {
    AtomicRecordCandidate candidate;
    ResolvedMoiScratchPlan resources;
    ConSanMoiAtomicAddressPlan address_plan;
    std::optional<VgprSpillSequence> spill;
  };
  std::vector<PlannedInlineAtomic> planned_candidates;
  MoiResourcePlanningState resource_state(bytes, arch, options, result);
  MoiSpillManagers spill_managers;
  MoiDescriptorVgprRequirements descriptor_requirements;
  MoiDescriptorSgprRequirements scalar_requirements;
  MoiDescriptorPrivateRequirements private_requirements;
  for (size_t candidate_index = 0; candidate_index < candidates.size(); ++candidate_index) {
    if (planned_candidates.size() == options.max_patches)
      break;
    const AtomicRecordCandidate &candidate = candidates[candidate_index];
    const ConSanAtomicSite &site = candidate.site;
    if (site.text_offset > old_text.size() || site.size > old_text.size() - site.text_offset) {
      result.errors.emplace_back("ConSan MOI inline atomic patch site is outside .text");
      return;
    }
    const auto overlaps_existing_patch = std::any_of(
        result.patches.begin(), result.patches.end(), [&](const ConSanPatchInfo &patch) {
          const bool exact_atomic_mutation =
              patch.phase == ConSanPatchPhase::Mutation &&
              patch.kind == ConSanPatchKind::InlineAtomicAddressRewrite &&
              patch.anchor_offset == site.text_offset && patch.original_size == site.size;
          if (exact_atomic_mutation)
            return false;
          const uint64_t patch_end =
              patch.anchor_offset + std::max<uint32_t>(patch.original_size, 1);
          return site.text_offset < patch_end && patch.anchor_offset < site.text_offset + site.size;
        });
    if (overlaps_existing_patch) {
      result.warnings.emplace_back(
          "ConSan MOI inline atomic patch skipped an overlapping patch in " +
          candidate.container_name);
      continue;
    }
    const ConSanCandidateResourcePlan *existing_plan =
        resource_plan_for_site(result, ConSanResourceSiteKind::Atomic, site.text_offset);
    const uint16_t scratch_count =
        inline_shadow_atomic_scratch_count(site, candidate.event_kind, arch);
    if (existing_plan && existing_plan->scratch_vgpr_count != scratch_count)
      existing_plan = nullptr;
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
          "ConSan MOI inline atomic patch has no usable resource plan in " +
          candidate.container_name);
      continue;
    }
    const ConSanMoiAtomicAddressPlan address_plan = plan_consan_moi_atomic_address(
        site, resources->base, resources->count, resources->source, arch);
    if (!address_plan.supported()) {
      result.warnings.emplace_back(
          "ConSan MOI inline atomic patch has no usable address plan in " +
          candidate.container_name + ": " +
          std::string(consan_moi_atomic_address_support_name(address_plan.support)) +
          " raw_saddr=" +
          (site.raw_saddr ? std::to_string(*site.raw_saddr) : std::string("missing")) +
          " saddr_sgpr=" +
          (site.saddr_sgpr ? std::to_string(*site.saddr_sgpr) : std::string("missing")) +
          " raw_vaddr=" +
          (site.raw_vaddr ? std::to_string(*site.raw_vaddr) : std::string("missing")) +
          " addr_vgpr=" +
          (site.addr_vgpr ? std::to_string(*site.addr_vgpr) : std::string("missing")));
      continue;
    }
    AtomicRecordCandidate planned_candidate = candidate;
    if (!planned_candidate.kernel_descriptor_file_offset) {
      planned_candidate.kernel_descriptor_file_offset =
          common_moi_record_owner_descriptor(bytes, *resources, arch, result.warnings);
      if (!planned_candidate.kernel_descriptor_file_offset)
        continue;
    }
    std::optional<VgprSpillSequence> spill;
    if (resources->source == ConSanRegisterAllocationSource::SpillRequired) {
      spill = build_moi_spill_sequence(result, *resources, options, spill_managers, arch,
                                       result.warnings);
      if (!spill)
        continue;
    }
    planned_candidates.push_back(
        {std::move(planned_candidate), *resources, address_plan, std::move(spill)});
  }
  // Retain every admitted release even when this code object contains no
  // acquire. Its consumer can live in another large code object, and pruning
  // here would make instrumentation depend on an incomplete module-local
  // view of the program.
  if (planned_candidates.empty()) {
    result.warnings.emplace_back("ConSan MOI inline atomic patch found no patchable atomics");
    return;
  }
  for (const PlannedInlineAtomic &planned : planned_candidates) {
    if (planned.spill) {
      note_spill_descriptor_requirements(private_requirements, planned.resources, *planned.spill);
    }
    note_descriptor_requirements(descriptor_requirements, planned.resources);
    note_moi_sgpr_requirements(scalar_requirements, planned.resources, options, arch);
  }
  if (!apply_descriptor_requirements(patcher, code_object, active_bytes, result,
                                     descriptor_requirements, arch, result.errors) ||
      !apply_spill_descriptor_requirements(patcher, code_object, active_bytes, result,
                                           private_requirements, result.errors) ||
      !apply_sgpr_descriptor_requirements(patcher, result, code_object, scalar_requirements,
                                          result.errors)) {
    return;
  }
  std::vector<uint8_t> new_text(old_text.begin(), old_text.end());
  DbiPatchPlacementPlanner placement_planner(arch, old_text.size());
  MoiLocalNopIslandAllocator indirect_allocator(old_text, result, arch,
                                                kMoiLocalIndirectIslandWords);
  std::vector<MoiLocalNopIsland> appended_indirect_islands;
  const bool needs_appended_island_bank =
      planned_candidates.size() > 1u &&
      (old_text.size() > 64u * 1024u ||
       std::ranges::any_of(planned_candidates, [&](const PlannedInlineAtomic &planned) {
         return !compute_sopp_branch_simm16(planned.candidate.site.text_offset, old_text.size());
       }));
  if (needs_appended_island_bank) {
    const uint64_t island_bytes =
        static_cast<uint64_t>(kMoiLocalIndirectIslandWords) * sizeof(uint32_t);
    const uint64_t bank_offset = placement_planner.appended_end();
    const size_t island_count = planned_candidates.size();
    const uint64_t bank_bytes = island_count * island_bytes;
    std::string bank_error;
    if (!placement_planner.reserve_appended_prefix(bank_bytes, &bank_error)) {
      result.errors.emplace_back("ConSan MOI inline atomic island bank failed: " + bank_error);
      return;
    }
    appended_indirect_islands.reserve(island_count);
    for (size_t index = 0; index < island_count; ++index) {
      appended_indirect_islands.push_back({bank_offset + index * island_bytes, false});
      for (uint32_t word = 0; word < kMoiLocalIndirectIslandWords; ++word)
        append_word_bytes(new_text, build_s_nop(0, arch));
    }
  }
  const auto claim_appended_island = [&](uint64_t anchor_offset) -> std::optional<uint64_t> {
    for (MoiLocalNopIsland &island : appended_indirect_islands) {
      if (island.claimed || !compute_sopp_branch_simm16(anchor_offset, island.text_offset))
        continue;
      island.claimed = true;
      return island.text_offset;
    }
    return std::nullopt;
  };
  std::vector<ConSanPatchInfo> patches;
  for (const PlannedInlineAtomic &planned : planned_candidates) {
    const AtomicRecordCandidate &candidate = planned.candidate;
    const ConSanAtomicSite &site = candidate.site;

    const uint64_t cave_text_offset = placement_planner.appended_end();
    const uint64_t return_text_offset = site.text_offset + site.size;
    ConSanOptions candidate_options = options;
    candidate_options.scratch_vgpr = planned.resources.base;
    uint32_t guest_instruction_offset = 0;
    auto cave_words = build_inline_atomic_ordering_cave_words(
        active_bytes, candidate, planned.address_plan, candidate_options,
        planned.spill ? &*planned.spill : nullptr, arch, layout.inline_atomic_release_slots_offset,
        layout.inline_atomic_release_capacity, layout.inline_causal_snapshots_offset,
        layout.inline_causal_snapshot_capacity, layout.inline_acquired_epoch_token_slots_offset,
        layout.inline_acquired_epoch_token_capacity, cave_text_offset, return_text_offset,
        guest_instruction_offset, result.errors);
    if (!cave_words)
      return;

    const bool direct_entry =
        compute_sopp_branch_simm16(site.text_offset, cave_text_offset).has_value();
    std::optional<uint64_t> indirect_entry_island;
    if (!direct_entry) {
      indirect_entry_island = claim_appended_island(site.text_offset);
      if (!indirect_entry_island)
        indirect_entry_island = indirect_allocator.claim_reachable(site.text_offset);
    }
    if (!direct_entry && !indirect_entry_island) {
      result.warnings.emplace_back(
          "ConSan MOI inline atomic patch found no reachable indirect entry island in " +
          candidate.container_name);
      continue;
    }
    std::string placement_error;
    const auto placement = placement_planner.plan_indirect_appended(
        site.text_offset, site.size, cave_words->size() * sizeof(uint32_t), &placement_error);
    if (!placement) {
      result.errors.emplace_back("ConSan MOI inline atomic placement failed: " + placement_error);
      return;
    }
    if (placement->body_offset != cave_text_offset || new_text.size() != placement->body_offset) {
      result.errors.emplace_back("ConSan MOI inline atomic patch has a stale cave mapping");
      return;
    }

    const uint64_t entry_target = indirect_entry_island.value_or(placement->body_offset);
    const auto fwd = compute_sopp_branch_simm16(placement->anchor_offset, entry_target);
    if (!fwd) {
      result.errors.emplace_back("ConSan MOI inline atomic forward branch is out of range");
      return;
    }
    std::vector<uint32_t> anchor_words(site.size / sizeof(uint32_t), build_s_nop(0, arch));
    if (anchor_words.empty()) {
      result.errors.emplace_back("ConSan MOI inline atomic patch empty anchor");
      return;
    }
    anchor_words.front() = build_s_branch(*fwd, arch);
    std::memcpy(new_text.data() + site.text_offset, anchor_words.data(),
                anchor_words.size() * sizeof(uint32_t));
    if (indirect_entry_island &&
        !emit_moi_local_indirect_entry_island(
            new_text, *indirect_entry_island, placement->body_offset, site.text_offset, options,
            planned.resources.owner_descriptor_file_offsets, arch, patches, result.errors,
            "ConSan MOI inline atomic patch")) {
      return;
    }
    append_words_bytes(new_text, *cave_words);

    ConSanPatchInfo info;
    info.kind = ConSanPatchKind::TrampolineMoiInlineAtomicOrdering;
    info.anchor_offset = site.text_offset;
    info.trampoline_offset = cave_text_offset;
    info.original_size = site.size;
    info.trampoline_size = static_cast<uint32_t>(cave_words->size() * sizeof(uint32_t));
    info.relocated_guest_instruction_offset = cave_text_offset + guest_instruction_offset;
    info.scratch_vgpr = planned.resources.base;
    info.owner_descriptor_file_offsets = planned.resources.owner_descriptor_file_offsets;
    if (planned.spill) {
      info.spilled_vgpr_count = planned.spill->vgpr_count;
      info.required_private_segment_size = planned.spill->total_private_bytes;
    }
    patches.push_back(info);
  }

  if (patches.empty()) {
    result.warnings.emplace_back("ConSan MOI inline atomic patch found no patchable atomics");
    return;
  }
  if (!patcher.replace_text(new_text, kConSanMaxPatchedFileGrowth)) {
    result.errors.emplace_back("ConSan MOI inline atomic patch could not grow .text");
    return;
  }
  result.elf_bytes = patcher.emit();
  result.patches.insert(result.patches.end(), patches.begin(), patches.end());
  result.modified = true;
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_atomic_record_cave_words(
    std::span<const uint8_t> bytes, const AtomicRecordCandidate &candidate,
    const ConSanMoiAtomicAddressPlan &address_plan, const ConSanOptions &options,
    const VgprSpillSequence *spill, const InlineSgprSpillSequence *scalar_spill,
    rj_code_arch_t arch, uint32_t record_index, uint32_t atomic_record_capacity,
    size_t atomic_records_offset, uint64_t cave_text_offset, uint64_t return_text_offset,
    uint32_t &guest_instruction_offset, std::vector<std::string> &errors) {
  if (!options.scratch_vgpr) {
    errors.emplace_back("ConSan MOI atomic record patch requires RJ_CONSAN_TMP_VGPR");
    return std::nullopt;
  }
  if (!options.moi_exec_save_sgpr) {
    errors.emplace_back("ConSan MOI atomic record patch requires scalar EXEC state");
    return std::nullopt;
  }
  if (static_cast<uint32_t>(*options.scratch_vgpr) + address_plan.scratch_vgpr_count > 256u) {
    errors.emplace_back("ConSan MOI atomic record patch has an invalid scratch VGPR window");
    return std::nullopt;
  }
  if (reject_atomic_candidate_scratch_overlap(candidate.site, *options.scratch_vgpr,
                                              address_plan.scratch_vgpr_count, errors) ||
      reject_optional_scratch_range_overlap(options.moi_owner_vgpr, *options.scratch_vgpr,
                                            address_plan.scratch_vgpr_count, "MOI owner", errors) ||
      reject_optional_scratch_range_overlap(options.moi_epoch_vgpr, *options.scratch_vgpr,
                                            address_plan.scratch_vgpr_count, "MOI epoch", errors))
    return std::nullopt;

  if (!address_plan.supported()) {
    errors.emplace_back("ConSan MOI atomic record patch requires a supported address plan");
    return std::nullopt;
  }

  (void)record_index;
  std::optional<uint16_t> derived_owner_vgpr;
  std::optional<uint32_t> derived_owner_word;
  if (!options.moi_owner_vgpr) {
    if (!candidate.kernel_descriptor_file_offset) {
      errors.emplace_back(
          "ConSan MOI atomic record patch requires RJ_CONSAN_MOI_OWNER_VGPR for function atomics");
      return std::nullopt;
    }
    const auto owner_shift =
        moi_descriptor_owner_shift(bytes, *candidate.kernel_descriptor_file_offset, errors);
    if (!owner_shift)
      return std::nullopt;
    const uint16_t value_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 4u);
    const auto owner_init = instrumentation::build_v_lshrrev_b32(
        value_vgpr, scalar_positive_inline_u32(*owner_shift), kRdna4WorkitemIdX, arch);
    if (!owner_init) {
      errors.emplace_back("ConSan MOI atomic record patch could not encode owner derivation");
      return std::nullopt;
    }
    derived_owner_vgpr = value_vgpr;
    derived_owner_word = *owner_init;
  }

  ConSanMoiWorkgroupSources workgroup_sources;
  if (candidate.kernel_descriptor_file_offset) {
    const auto descriptor_workgroup_sources = moi_descriptor_workgroup_sources(
        bytes, *candidate.kernel_descriptor_file_offset, arch, errors);
    if (!descriptor_workgroup_sources)
      return std::nullopt;
    workgroup_sources = *descriptor_workgroup_sources;
  }

  if (candidate.site.file_offset > bytes.size() ||
      candidate.site.size > bytes.size() - candidate.site.file_offset) {
    errors.emplace_back("ConSan MOI atomic record patch site exceeds ELF bytes");
    return std::nullopt;
  }

  std::vector<uint32_t> words;
  words.reserve(candidate.site.size / sizeof(uint32_t) + 96u +
                (spill ? spill->save_words.size() + spill->restore_words.size() : 0u));
  if (spill)
    words.insert(words.end(), spill->save_words.begin(), spill->save_words.end());
  if (scalar_spill)
    words.insert(words.end(), scalar_spill->save_words.begin(), scalar_spill->save_words.end());
  if (address_plan.requires_materialization()) {
    const auto special_state = moi_special_state_sgprs(options);
    if (!special_state) {
      errors.emplace_back(
          "ConSan MOI atomic record address materialization requires special-state SGPRs");
      return std::nullopt;
    }
    const auto address_words = build_consan_moi_atomic_address_materialization(
        address_plan, special_state->vcc_save_sgpr, special_state->scc_save_sgpr, arch);
    if (!address_words) {
      errors.emplace_back("ConSan MOI atomic record patch could not materialize its address");
      return std::nullopt;
    }
    words.insert(words.end(), address_words->begin(), address_words->end());
  }
  const bool is_compare_exchange = candidate.site.mnemonic.find("cmpswap") != std::string::npos ||
                                   candidate.site.mnemonic.find("cmpxchg") != std::string::npos;
  std::vector<uint32_t> guest_atomic_words;
  guest_atomic_words.reserve(candidate.site.size / sizeof(uint32_t));
  for (uint64_t offset = 0; offset < candidate.site.size; offset += sizeof(uint32_t)) {
    uint32_t word = 0;
    std::memcpy(&word, bytes.data() + candidate.site.file_offset + offset, sizeof(word));
    guest_atomic_words.push_back(word);
  }

  const bool record_before_guest_release =
      candidate.event_kind == ConSanMoiAtomicEventKind::Release && !is_compare_exchange;
  if (!record_before_guest_release) {
    guest_instruction_offset = static_cast<uint32_t>(words.size() * sizeof(uint32_t));
    words.insert(words.end(), guest_atomic_words.begin(), guest_atomic_words.end());
    if (!append_moi_flat_load_wait(words, arch))
      return std::nullopt;
  }

  const uint16_t recorded_address_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 5u);
  words.push_back(build_v_mov_b32_e32(recorded_address_vgpr,
                                      vector_source_vgpr(address_plan.result_address_vgpr), arch));
  words.push_back(build_v_mov_b32_e32(
      static_cast<uint16_t>(recorded_address_vgpr + 1u),
      vector_source_vgpr(static_cast<uint16_t>(address_plan.result_address_vgpr + 1u)), arch));

  if (!append_save_moi_special_state(words, options, arch)) {
    errors.emplace_back("ConSan MOI atomic record patch could not save VCC/SCC");
    return std::nullopt;
  }

  const uint64_t base = *options.moi_report_buffer_address;
  const uint64_t atomic_record_base = base + atomic_records_offset;
  const uint16_t slot_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 2u);
  const uint16_t lane_rank_vgpr = *options.scratch_vgpr;
  if (is_compare_exchange) {
    if (!candidate.site.data_vgpr || !candidate.site.dst_vgpr) {
      errors.emplace_back("ConSan MOI atomic record CAS lacks compare/result operands");
      return std::nullopt;
    }
    const uint16_t value_word_count = static_cast<uint16_t>(candidate.site.width_bits / 32u);
    const uint16_t compare_vgpr =
        static_cast<uint16_t>(*candidate.site.data_vgpr + value_word_count);
    const auto compare_low = instrumentation::build_v_cmp_eq_u32_vcc(
        vector_source_vgpr(compare_vgpr), *candidate.site.dst_vgpr, arch);
    if (!compare_low) {
      errors.emplace_back("ConSan MOI atomic record CAS could not capture its outcome mask");
      return std::nullopt;
    }
    const uint16_t success_lo = static_cast<uint16_t>(*options.scratch_vgpr + 3u);
    const uint16_t success_hi = static_cast<uint16_t>(*options.scratch_vgpr + 4u);
    words.push_back(*compare_low);
    words.push_back(build_v_mov_b32_e32(success_lo, kRdna4VccLo, arch));
    words.push_back(build_v_mov_b32_e32(success_hi, kRdna4VccHi, arch));
    if (candidate.site.width_bits == 64u) {
      const auto compare_high = instrumentation::build_v_cmp_eq_u32_vcc(
          vector_source_vgpr(static_cast<uint16_t>(compare_vgpr + 1u)),
          static_cast<uint16_t>(*candidate.site.dst_vgpr + 1u), arch);
      const auto intersect_lo =
          instrumentation::build_v_and_b32(success_lo, kRdna4VccLo, success_lo, arch);
      const auto intersect_hi =
          instrumentation::build_v_and_b32(success_hi, kRdna4VccHi, success_hi, arch);
      if (!compare_high || !intersect_lo || !intersect_hi) {
        errors.emplace_back(
            "ConSan MOI atomic record 64-bit CAS could not combine its outcome masks");
        return std::nullopt;
      }
      words.push_back(*compare_high);
      words.push_back(*intersect_lo);
      words.push_back(*intersect_hi);
    }
  }
  const auto save_active_exec =
      instrumentation::build_s_mov_b64(*options.moi_exec_save_sgpr, kRdna4ExecLo, arch);
  const auto mbcnt_lo = instrumentation::build_v_mbcnt_lo_u32_b32(
      lane_rank_vgpr, *options.moi_exec_save_sgpr, scalar_positive_inline_u32(0), arch);
  const auto mbcnt_hi = instrumentation::build_v_mbcnt_hi_u32_b32(
      lane_rank_vgpr, static_cast<uint16_t>(*options.moi_exec_save_sgpr + 1u),
      vector_source_vgpr(lane_rank_vgpr), arch);
  const auto first_active_lane =
      instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), lane_rank_vgpr, arch);
  const auto save_exec =
      instrumentation::build_s_and_saveexec_b64(*options.moi_exec_save_sgpr, kRdna4VccLo, arch);
  if (!save_active_exec || !mbcnt_lo || !mbcnt_hi || !first_active_lane || !save_exec) {
    errors.emplace_back("ConSan MOI atomic record patch could not select a wave publisher");
    return std::nullopt;
  }
  words.push_back(*save_active_exec);
  words.insert(words.end(), mbcnt_lo->begin(), mbcnt_lo->end());
  words.insert(words.end(), mbcnt_hi->begin(), mbcnt_hi->end());
  words.push_back(*first_active_lane);
  words.push_back(*save_exec);
  if (!options.moi_exec_save_sgpr ||
      !append_atomic_fetch_add_one_u32(words,
                                       base + offsetof(ConSanMoiReportHeader, atomic_record_count),
                                       slot_vgpr, *options.scratch_vgpr, arch)) {
    errors.emplace_back("ConSan MOI atomic record patch could not reserve a dynamic record slot");
    return std::nullopt;
  }
  const auto mov_capacity =
      instrumentation::build_v_mov_b32_literal(lane_rank_vgpr, atomic_record_capacity, arch);
  const auto slot_in_capacity =
      instrumentation::build_v_cmp_gt_u32_vcc(vector_source_vgpr(lane_rank_vgpr), slot_vgpr, arch);
  if (!mov_capacity || !slot_in_capacity) {
    errors.emplace_back("ConSan MOI atomic record patch could not guard dynamic record capacity");
    return std::nullopt;
  }
  words.insert(words.end(), mov_capacity->begin(), mov_capacity->end());
  words.push_back(*slot_in_capacity);
  const ConSanMoiAtomicOperation atomic_operation = is_compare_exchange
                                                        ? ConSanMoiAtomicOperation::CompareExchange
                                                        : ConSanMoiAtomicOperation::Rmw;
  const ConSanMoiAtomicOutcome atomic_outcome = is_compare_exchange
                                                    ? ConSanMoiAtomicOutcome::Unavailable
                                                    : ConSanMoiAtomicOutcome::NotApplicable;
  std::vector<uint32_t> record_words;
  record_words.reserve(128);
  if (is_compare_exchange) {
    if (!append_dynamic_atomic_store_u32_vgpr(
            record_words, atomic_record_base + offsetof(ConSanMoiAtomicRecord, success_lane_mask),
            static_cast<uint16_t>(*options.scratch_vgpr + 3u), slot_vgpr, *options.scratch_vgpr,
            arch) ||
        !append_dynamic_atomic_store_u32_vgpr(
            record_words,
            atomic_record_base + offsetof(ConSanMoiAtomicRecord, success_lane_mask) +
                sizeof(uint32_t),
            static_cast<uint16_t>(*options.scratch_vgpr + 4u), slot_vgpr, *options.scratch_vgpr,
            arch)) {
      errors.emplace_back("ConSan MOI atomic record patch could not preserve CAS outcome");
      return std::nullopt;
    }
  }
  if (derived_owner_word)
    record_words.push_back(*derived_owner_word);
  // Probe-local owner derivation uses the record-emitter temporary. Preserve
  // it before reserving the event index, which reuses that VGPR.
  if ((derived_owner_vgpr &&
       !append_dynamic_atomic_store_u32_vgpr(
           record_words, atomic_record_base + offsetof(ConSanMoiAtomicRecord, owner_id),
           *derived_owner_vgpr, slot_vgpr, *options.scratch_vgpr, arch)) ||
      !append_dynamic_atomic_event_index_store(
          record_words, base + offsetof(ConSanMoiReportHeader, event_counter),
          atomic_record_base + offsetof(ConSanMoiAtomicRecord, event_index), slot_vgpr,
          *options.scratch_vgpr, arch) ||
      !append_dynamic_atomic_store_workgroup_source(
          record_words, atomic_record_base + offsetof(ConSanMoiAtomicRecord, workgroup_x),
          workgroup_sources.x, slot_vgpr, *options.scratch_vgpr, arch) ||
      !append_dynamic_atomic_store_workgroup_source(
          record_words, atomic_record_base + offsetof(ConSanMoiAtomicRecord, workgroup_y),
          workgroup_sources.y, slot_vgpr, *options.scratch_vgpr, arch) ||
      !append_dynamic_atomic_store_workgroup_source(
          record_words, atomic_record_base + offsetof(ConSanMoiAtomicRecord, workgroup_z),
          workgroup_sources.z, slot_vgpr, *options.scratch_vgpr, arch) ||
      (options.moi_owner_vgpr &&
       !append_dynamic_atomic_store_u32_vgpr(
           record_words, atomic_record_base + offsetof(ConSanMoiAtomicRecord, owner_id),
           *options.moi_owner_vgpr, slot_vgpr, *options.scratch_vgpr, arch)) ||
      (options.moi_epoch_vgpr &&
       !append_dynamic_atomic_store_u32_vgpr(
           record_words, atomic_record_base + offsetof(ConSanMoiAtomicRecord, epoch),
           *options.moi_epoch_vgpr, slot_vgpr, *options.scratch_vgpr, arch)) ||
      !append_dynamic_atomic_store_u32_vgpr(
          record_words, atomic_record_base + offsetof(ConSanMoiAtomicRecord, atomic_address),
          recorded_address_vgpr, slot_vgpr, *options.scratch_vgpr, arch) ||
      !append_dynamic_atomic_store_u32_vgpr(
          record_words,
          atomic_record_base + offsetof(ConSanMoiAtomicRecord, atomic_address) + sizeof(uint32_t),
          static_cast<uint16_t>(recorded_address_vgpr + 1u), slot_vgpr, *options.scratch_vgpr,
          arch) ||
      !append_dynamic_atomic_store_u32_literal(
          record_words, atomic_record_base + offsetof(ConSanMoiAtomicRecord, instruction_offset),
          static_cast<uint32_t>(candidate.site.text_offset), slot_vgpr, *options.scratch_vgpr,
          arch) ||
      !append_dynamic_atomic_store_u32_literal(
          record_words, atomic_record_base + offsetof(ConSanMoiAtomicRecord, kind),
          static_cast<uint32_t>(candidate.event_kind), slot_vgpr, *options.scratch_vgpr, arch) ||
      !append_dynamic_atomic_store_u32_literal(
          record_words, atomic_record_base + offsetof(ConSanMoiAtomicRecord, scope),
          *candidate.site.raw_scope, slot_vgpr, *options.scratch_vgpr, arch) ||
      !append_dynamic_atomic_store_u32_literal(
          record_words, atomic_record_base + offsetof(ConSanMoiAtomicRecord, semantics),
          candidate.site.raw_th.value_or(0u), slot_vgpr, *options.scratch_vgpr, arch) ||
      !append_dynamic_atomic_store_u32_literal(
          record_words, atomic_record_base + offsetof(ConSanMoiAtomicRecord, operation),
          static_cast<uint32_t>(atomic_operation), slot_vgpr, *options.scratch_vgpr, arch) ||
      !append_dynamic_atomic_store_u32_literal(
          record_words, atomic_record_base + offsetof(ConSanMoiAtomicRecord, outcome),
          static_cast<uint32_t>(atomic_outcome), slot_vgpr, *options.scratch_vgpr, arch) ||
      (is_compare_exchange
           ? (!append_dynamic_atomic_store_u32_scalar_src(
                  record_words, atomic_record_base + offsetof(ConSanMoiAtomicRecord, lane_mask),
                  *options.moi_exec_save_sgpr, slot_vgpr, *options.scratch_vgpr, arch) ||
              !append_dynamic_atomic_store_u32_scalar_src(
                  record_words,
                  atomic_record_base + offsetof(ConSanMoiAtomicRecord, lane_mask) +
                      sizeof(uint32_t),
                  static_cast<uint16_t>(*options.moi_exec_save_sgpr + 1u), slot_vgpr,
                  *options.scratch_vgpr, arch))
           : false)) {
    errors.emplace_back("ConSan MOI atomic record patch could not encode record stores");
    return std::nullopt;
  }

  const auto wait_store = instrumentation::build_s_wait_flat_store0(arch);
  const auto restore_exec =
      instrumentation::build_s_mov_b64(kRdna4ExecLo, *options.moi_exec_save_sgpr, arch);
  if (!wait_store || !restore_exec) {
    errors.emplace_back("ConSan MOI atomic record patch could not complete dynamic publication");
    return std::nullopt;
  }
  record_words.push_back(*wait_store);
  if (record_words.size() > static_cast<size_t>(std::numeric_limits<int16_t>::max())) {
    errors.emplace_back("ConSan MOI atomic record overflow branch is out of range");
    return std::nullopt;
  }
  const auto skip_record =
      instrumentation::build_s_cbranch_vccz(static_cast<int16_t>(record_words.size()), arch);
  if (!skip_record) {
    errors.emplace_back("ConSan MOI atomic record patch could not encode its capacity branch");
    return std::nullopt;
  }
  words.push_back(*skip_record);
  words.insert(words.end(), record_words.begin(), record_words.end());
  words.push_back(*restore_exec);

  if (!append_restore_moi_special_state(words, options, arch)) {
    errors.emplace_back("ConSan MOI atomic record patch could not restore VCC/SCC");
    return std::nullopt;
  }

  if (scalar_spill)
    words.insert(words.end(), scalar_spill->restore_words.begin(),
                 scalar_spill->restore_words.end());
  if (spill)
    words.insert(words.end(), spill->restore_words.begin(), spill->restore_words.end());
  if (record_before_guest_release) {
    // An ordinary release RMW has no result needed by the record.  Keep its
    // linearization point at the end of the trampoline: placing the large
    // reporting sequence after the guest atomic opens an artificial window in
    // which another wave can observe the release while the releasing wave has
    // not yet resumed its guest bookkeeping.  An acquire that observes this
    // release cannot reserve its replay event until after this pre-release
    // record, so the replay order remains causal.  Restore spilled operands and
    // guest VCC/SCC before executing the atomic, then return immediately.
    guest_instruction_offset = static_cast<uint32_t>(words.size() * sizeof(uint32_t));
    words.insert(words.end(), guest_atomic_words.begin(), guest_atomic_words.end());
  }
  if (!append_moi_direct_or_indirect_return(words, cave_text_offset, return_text_offset, options,
                                            arch)) {
    errors.emplace_back("ConSan MOI atomic record could not encode its return");
    return std::nullopt;
  }
  return words;
}

[[nodiscard]] std::optional<ConSanMoiSampledSyncRole>
sampled_atomic_role(ConSanMoiAtomicEventKind kind) {
  switch (kind) {
  case ConSanMoiAtomicEventKind::Release:
    return ConSanMoiSampledSyncRole::RmwRelease;
  case ConSanMoiAtomicEventKind::Acquire:
    return ConSanMoiSampledSyncRole::RmwAcquire;
  case ConSanMoiAtomicEventKind::AcquireRelease:
    return ConSanMoiSampledSyncRole::RmwAcquireRelease;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<ConSanMoiSampledSyncScope> sampled_atomic_scope(uint32_t raw_scope) {
  if (raw_scope == 0u)
    return ConSanMoiSampledSyncScope::Wavefront;
  if (raw_scope == 1u)
    return ConSanMoiSampledSyncScope::Workgroup;
  if (raw_scope == 2u)
    return ConSanMoiSampledSyncScope::Agent;
  if (raw_scope == 3u)
    return ConSanMoiSampledSyncScope::System;
  return std::nullopt;
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_sampled_pending_acquire_cave_words(
    std::span<const uint8_t> bytes, const AtomicRecordCandidate &candidate,
    const ConSanMoiAtomicAddressPlan &address_plan, const ConSanOptions &options,
    const VgprSpillSequence *spill, rj_code_arch_t arch, uint32_t selected_slot,
    const ConSanMoiReportBufferLayout &layout, uint64_t cave_text_offset,
    uint64_t return_text_offset, std::vector<std::string> &errors,
    uint32_t *guest_instruction_offset = nullptr) {
  if (!options.scratch_vgpr || !options.moi_report_buffer_address || !options.moi_owner_vgpr ||
      !options.moi_epoch_vgpr || !options.moi_exec_save_sgpr ||
      static_cast<uint32_t>(*options.scratch_vgpr) + 10u > kMaxVgprs ||
      selected_slot >= layout.sampled_pending_acquire_capacity || !address_plan.supported() ||
      !candidate.site.raw_scope || candidate.site.width_bits != 32u ||
      !candidate.kernel_descriptor_file_offset)
    return std::nullopt;
  const auto role = sampled_atomic_role(candidate.event_kind);
  const auto scope = sampled_atomic_scope(*candidate.site.raw_scope);
  if (!role || !scope ||
      (static_cast<uint32_t>(*role) & static_cast<uint32_t>(ConSanMoiSampledSyncRole::Acquire)) ==
          0u)
    return std::nullopt;
  const auto workgroup_sources =
      moi_descriptor_workgroup_sources(bytes, *candidate.kernel_descriptor_file_offset, arch,
                                       errors, candidate.uses_gfx1250_cluster_workgroup_id);
  if (!workgroup_sources || candidate.site.file_offset > bytes.size() ||
      candidate.site.size > bytes.size() - candidate.site.file_offset)
    return std::nullopt;

  const bool is_cas = candidate.site.mnemonic.find("cmpswap") != std::string::npos ||
                      candidate.site.mnemonic.find("cmpxchg") != std::string::npos;
  if (is_cas && (!candidate.site.data_vgpr || !candidate.site.dst_vgpr ||
                 !candidate.site.returns_old_value.value_or(false)))
    return std::nullopt;
  const auto descriptor_for = [&](ConSanMoiSampledSyncOutcome outcome) -> std::optional<uint32_t> {
    const auto encoded = encode_consan_moi_sampled_sync_metadata({
        .address = 1,
        .byte_count = 4,
        .kind = ConSanMoiSampledSyncKind::Atomic,
        .role = *role,
        .scope = *scope,
        .outcome = outcome,
    });
    if (encoded.classification != ConSanMoiSampledSyncClassification::Valid)
      return std::nullopt;
    return encoded.packed.descriptor;
  };
  const auto success_descriptor = descriptor_for(is_cas ? ConSanMoiSampledSyncOutcome::CasSuccess
                                                 : candidate.site.returns_old_value.value_or(false)
                                                     ? ConSanMoiSampledSyncOutcome::RmwReturnsOld
                                                     : ConSanMoiSampledSyncOutcome::RmwNoReturn);
  const auto failure_descriptor =
      is_cas ? descriptor_for(ConSanMoiSampledSyncOutcome::CasFailure) : success_descriptor;
  if (!success_descriptor || !failure_descriptor)
    return std::nullopt;

  const uint16_t base = *options.scratch_vgpr;
  const uint16_t value = static_cast<uint16_t>(base + 2u);
  const uint16_t expected = static_cast<uint16_t>(base + 3u);
  const uint16_t saved_address = static_cast<uint16_t>(base + 8u);
  const uint16_t original_exec = static_cast<uint16_t>(*options.moi_exec_save_sgpr + 6u);
  const uint16_t temporary_exec = *options.moi_exec_save_sgpr;
  const uint64_t report_base = *options.moi_report_buffer_address;
  const uint64_t pending_address =
      report_base + layout.sampled_pending_acquires_offset +
      static_cast<uint64_t>(selected_slot) * sizeof(ConSanMoiSampledPendingAcquireSlot);

  std::vector<uint32_t> words;
  if (spill)
    words.insert(words.end(), spill->save_words.begin(), spill->save_words.end());
  if (address_plan.requires_materialization()) {
    const auto special = moi_special_state_sgprs(options);
    if (!special)
      return std::nullopt;
    const auto materialize = build_consan_moi_atomic_address_materialization(
        address_plan, special->vcc_save_sgpr, special->scc_save_sgpr, arch);
    if (!materialize)
      return std::nullopt;
    words.insert(words.end(), materialize->begin(), materialize->end());
  }
  if (guest_instruction_offset)
    *guest_instruction_offset = static_cast<uint32_t>(words.size() * sizeof(uint32_t));
  for (uint64_t offset = 0; offset < candidate.site.size; offset += sizeof(uint32_t)) {
    uint32_t word = 0;
    std::memcpy(&word, bytes.data() + candidate.site.file_offset + offset, sizeof(word));
    words.push_back(word);
  }
  if (!append_moi_flat_load_wait(words, arch))
    return std::nullopt;
  words.push_back(build_v_mov_b32_e32(saved_address,
                                      vector_source_vgpr(address_plan.result_address_vgpr), arch));
  words.push_back(build_v_mov_b32_e32(
      static_cast<uint16_t>(saved_address + 1u),
      vector_source_vgpr(static_cast<uint16_t>(address_plan.result_address_vgpr + 1u)), arch));
  InstructionSequence sequence(words);
  ConSanMoiRecordEmitter record(words, base, value, arch);
  const auto contention_label = sequence.make_label();
  const auto collision_label = sequence.make_label();
  const auto restore_label = sequence.make_label();
  if (!append_save_moi_special_state(words, options, arch)) {
    errors.emplace_back("ConSan MOI sampled barrier failed at special-state save");
    return std::nullopt;
  }
  const auto save_exec = instrumentation::build_s_mov_b64(original_exec, kRdna4ExecLo, arch);
  const auto active_mask = instrumentation::build_s_mov_b64(temporary_exec, kRdna4ExecLo, arch);
  const auto mbcnt_lo = instrumentation::build_v_mbcnt_lo_u32_b32(
      value, temporary_exec, scalar_positive_inline_u32(0), arch);
  const auto mbcnt_hi = instrumentation::build_v_mbcnt_hi_u32_b32(
      value, static_cast<uint16_t>(temporary_exec + 1u), vector_source_vgpr(value), arch);
  const auto first =
      instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), value, arch);
  const auto narrow_first =
      instrumentation::build_s_and_saveexec_b64(temporary_exec, kRdna4VccLo, arch);
  if (!sequence.emit_all(save_exec, active_mask, mbcnt_lo, mbcnt_hi, first, narrow_first))
    return std::nullopt;

  const auto pending_lo =
      instrumentation::build_v_mov_b32_literal(base, static_cast<uint32_t>(pending_address), arch);
  const auto pending_hi = instrumentation::build_v_mov_b32_literal(
      static_cast<uint16_t>(base + 1u), static_cast<uint32_t>(pending_address >> 32u), arch);
  const auto publishing = instrumentation::build_v_mov_b32_literal(value, 1u, arch);
  const auto empty = instrumentation::build_v_mov_b32_literal(expected, 0u, arch);
  const auto claim = instrumentation::build_flat_atomic_cmpswap_b32(
      base, value, value, /*return_old_value=*/true, kRdna4ScopeDevice, arch);
  const auto claim_wait = instrumentation::build_s_wait_flat_load0(arch);
  if (!sequence.emit_all(pending_lo, pending_hi, publishing, empty, claim, claim_wait))
    return std::nullopt;
  const auto claimed =
      instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), value, arch);
  const auto narrow_claimed =
      instrumentation::build_s_and_saveexec_b64(temporary_exec, kRdna4VccLo, arch);
  if (!sequence.emit_all(claimed, narrow_claimed))
    return std::nullopt;
  if (!sequence.emit_branch(contention_label, InstructionSequence::BranchKind::SccZero))
    return std::nullopt;

  if (!record.store_literal(offsetof(ConSanMoiSampledPendingAcquireSlot, selected_slot),
                            selected_slot) ||
      !record.store_literal(offsetof(ConSanMoiSampledPendingAcquireSlot, generation),
                            static_cast<uint32_t>(options.moi_report_generation)) ||
      !record.store_literal(offsetof(ConSanMoiSampledPendingAcquireSlot, generation) + 4u,
                            static_cast<uint32_t>(options.moi_report_generation >> 32u)) ||
      !record.store_literal(offsetof(ConSanMoiSampledPendingAcquireSlot, dispatch_id),
                            static_cast<uint32_t>(options.moi_report_dispatch_id)) ||
      !record.store_literal(offsetof(ConSanMoiSampledPendingAcquireSlot, dispatch_id) + 4u,
                            static_cast<uint32_t>(options.moi_report_dispatch_id >> 32u)) ||
      !record.store_workgroup(offsetof(ConSanMoiSampledPendingAcquireSlot, workgroup_x),
                              workgroup_sources->x) ||
      !record.store_workgroup(offsetof(ConSanMoiSampledPendingAcquireSlot, workgroup_y),
                              workgroup_sources->y) ||
      !record.store_workgroup(offsetof(ConSanMoiSampledPendingAcquireSlot, workgroup_z),
                              workgroup_sources->z) ||
      !record.store_vgpr(offsetof(ConSanMoiSampledPendingAcquireSlot, owner_id),
                         *options.moi_owner_vgpr) ||
      !record.store_vgpr(offsetof(ConSanMoiSampledPendingAcquireSlot, source_epoch),
                         *options.moi_epoch_vgpr) ||
      !record.store_literal(offsetof(ConSanMoiSampledPendingAcquireSlot, reserved), 0u) ||
      !record.store_vgpr(offsetof(ConSanMoiSampledPendingAcquireSlot, metadata) +
                             offsetof(ConSanMoiSampledSyncMetadataPacked, address),
                         saved_address) ||
      !record.store_vgpr(offsetof(ConSanMoiSampledPendingAcquireSlot, metadata) +
                             offsetof(ConSanMoiSampledSyncMetadataPacked, address) + 4u,
                         static_cast<uint16_t>(saved_address + 1u)) ||
      !record.store_literal(offsetof(ConSanMoiSampledPendingAcquireSlot, metadata) +
                                offsetof(ConSanMoiSampledSyncMetadataPacked, byte_count),
                            4u) ||
      !record.store_vgpr(offsetof(ConSanMoiSampledPendingAcquireSlot, metadata) +
                             offsetof(ConSanMoiSampledSyncMetadataPacked, epoch_before),
                         *options.moi_epoch_vgpr) ||
      !record.store_vgpr(offsetof(ConSanMoiSampledPendingAcquireSlot, metadata) +
                             offsetof(ConSanMoiSampledSyncMetadataPacked, epoch_after),
                         *options.moi_epoch_vgpr))
    return std::nullopt;

  if (is_cas) {
    const auto cas_failure_label = sequence.make_label();
    const auto descriptor_done_label = sequence.make_label();
    const uint16_t compare_vgpr = static_cast<uint16_t>(*candidate.site.data_vgpr + 1u);
    const auto success = instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(compare_vgpr),
                                                                 *candidate.site.dst_vgpr, arch);
    if (!success)
      return std::nullopt;
    words.push_back(*success);
    if (!sequence.emit_branch(cas_failure_label, InstructionSequence::BranchKind::VccZero))
      return std::nullopt;
    if (!record.store_literal(offsetof(ConSanMoiSampledPendingAcquireSlot, metadata) +
                                  offsetof(ConSanMoiSampledSyncMetadataPacked, descriptor),
                              *success_descriptor))
      return std::nullopt;
    if (!sequence.emit_branch(descriptor_done_label,
                              InstructionSequence::BranchKind::Unconditional) ||
        !sequence.bind(cas_failure_label))
      return std::nullopt;
    if (!record.store_literal(offsetof(ConSanMoiSampledPendingAcquireSlot, metadata) +
                                  offsetof(ConSanMoiSampledSyncMetadataPacked, descriptor),
                              *failure_descriptor))
      return std::nullopt;
    if (!sequence.bind(descriptor_done_label))
      return std::nullopt;
  } else if (!record.store_literal(offsetof(ConSanMoiSampledPendingAcquireSlot, metadata) +
                                       offsetof(ConSanMoiSampledSyncMetadataPacked, descriptor),
                                   *success_descriptor)) {
    return std::nullopt;
  }

  const auto wait_store = instrumentation::build_s_wait_flat_store0(arch);
  const auto ready = instrumentation::build_v_mov_b32_literal(value, 2u, arch);
  const auto expected_publishing = instrumentation::build_v_mov_b32_literal(expected, 1u, arch);
  const auto commit_lo =
      instrumentation::build_v_mov_b32_literal(base, static_cast<uint32_t>(pending_address), arch);
  const auto commit_hi = instrumentation::build_v_mov_b32_literal(
      static_cast<uint16_t>(base + 1u), static_cast<uint32_t>(pending_address >> 32u), arch);
  const auto commit = instrumentation::build_flat_atomic_cmpswap_b32(
      base, value, value, /*return_old_value=*/true, kRdna4ScopeDevice, arch);
  const auto commit_wait = instrumentation::build_s_wait_flat_load0(arch);
  if (!sequence.emit_all(wait_store, ready, expected_publishing, commit_lo, commit_hi, commit,
                         commit_wait))
    return std::nullopt;
  const auto committed =
      instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(expected), value, arch);
  const auto narrow_committed =
      instrumentation::build_s_and_saveexec_b64(temporary_exec, kRdna4VccLo, arch);
  if (!sequence.emit_all(committed, narrow_committed))
    return std::nullopt;
  if (!sequence.emit_branch(collision_label, InstructionSequence::BranchKind::SccZero))
    return std::nullopt;
  if (!append_atomic_fetch_add_one_u32(
          words, report_base + offsetof(ConSanMoiReportHeader, sampled_pending_acquire_count),
          value, base, arch))
    return std::nullopt;
  if (!sequence.emit_branch(restore_label, InstructionSequence::BranchKind::Unconditional))
    return std::nullopt;

  const auto append_counter_path = [&](size_t counter_offset) -> bool {
    const auto restore = instrumentation::build_s_mov_b64(kRdna4ExecLo, original_exec, arch);
    if (!sequence.emit_all(restore, active_mask, mbcnt_lo, mbcnt_hi, first, narrow_first))
      return false;
    if (!append_atomic_fetch_add_one_u32(words, report_base + counter_offset, value, base, arch))
      return false;
    return sequence.emit_branch(restore_label, InstructionSequence::BranchKind::Unconditional);
  };
  // The sole successful claim chooses one sampled wave. Losing claims are
  // ordinary bounded contention and never write the pending record. The later
  // read requires the exact winner, so winner-without-read remains fail closed.
  if (!sequence.bind(contention_label) ||
      !append_counter_path(
          offsetof(ConSanMoiReportHeader, sampled_pending_acquire_contention_count)) ||
      !sequence.bind(collision_label) ||
      !append_counter_path(
          offsetof(ConSanMoiReportHeader, sampled_pending_acquire_collision_count)))
    return std::nullopt;

  const auto restore_exec = instrumentation::build_s_mov_b64(kRdna4ExecLo, original_exec, arch);
  if (!sequence.bind(restore_label) || !sequence.emit(restore_exec) ||
      !append_restore_moi_special_state(words, options, arch))
    return std::nullopt;
  if (spill)
    words.insert(words.end(), spill->restore_words.begin(), spill->restore_words.end());
  if (!append_moi_direct_or_indirect_return(words, cave_text_offset, return_text_offset, options,
                                            arch))
    return std::nullopt;

  if (!sequence.resolve_branches(arch))
    return std::nullopt;
  return words;
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_sampled_atomic_sync_cave_words(
    std::span<const uint8_t> bytes, const AtomicRecordCandidate &candidate,
    const ConSanMoiAtomicAddressPlan &address_plan, const ConSanOptions &options,
    const VgprSpillSequence *spill, rj_code_arch_t arch, uint32_t selected_slot,
    const ConSanMoiReportBufferLayout &layout, uint64_t cave_text_offset,
    uint64_t return_text_offset, std::vector<std::string> &errors,
    uint32_t *guest_instruction_offset = nullptr) {
  if (!options.scratch_vgpr || !options.moi_report_buffer_address || !options.moi_owner_vgpr ||
      !options.moi_epoch_vgpr || !options.moi_exec_save_sgpr ||
      static_cast<uint32_t>(*options.scratch_vgpr) + 10u > 256u ||
      selected_slot >= layout.sampled_sync_metadata_capacity)
    return std::nullopt;
  if (!address_plan.supported() || !candidate.site.raw_scope || candidate.site.width_bits != 32u)
    return std::nullopt;
  const auto role = sampled_atomic_role(candidate.event_kind);
  const auto scope = sampled_atomic_scope(*candidate.site.raw_scope);
  if (!role || !scope)
    return std::nullopt;
  if (!candidate.kernel_descriptor_file_offset) {
    errors.emplace_back("ConSan MOI sampled atomic metadata lacks an owning kernel descriptor");
    return std::nullopt;
  }
  const auto workgroup_sources =
      moi_descriptor_workgroup_sources(bytes, *candidate.kernel_descriptor_file_offset, arch,
                                       errors, candidate.uses_gfx1250_cluster_workgroup_id);
  if (!workgroup_sources)
    return std::nullopt;
  if (candidate.site.file_offset > bytes.size() ||
      candidate.site.size > bytes.size() - candidate.site.file_offset) {
    errors.emplace_back("ConSan MOI sampled atomic metadata site exceeds ELF bytes");
    return std::nullopt;
  }

  const bool is_cas = candidate.site.mnemonic.find("cmpswap") != std::string::npos ||
                      candidate.site.mnemonic.find("cmpxchg") != std::string::npos;
  if (is_cas && (!candidate.site.data_vgpr || !candidate.site.dst_vgpr ||
                 !candidate.site.returns_old_value.value_or(false))) {
    errors.emplace_back("ConSan MOI sampled atomic metadata requires an exact CAS outcome");
    return std::nullopt;
  }

  const auto descriptor_for = [&](ConSanMoiSampledSyncOutcome outcome) -> std::optional<uint32_t> {
    const ConSanMoiSampledSyncEncodeResult encoded = encode_consan_moi_sampled_sync_metadata({
        .address = 1,
        .byte_count = 4,
        .kind = ConSanMoiSampledSyncKind::Atomic,
        .role = *role,
        .scope = *scope,
        .outcome = outcome,
        .epoch_before = 0,
        .epoch_after = 0,
    });
    if (encoded.classification != ConSanMoiSampledSyncClassification::Valid)
      return std::nullopt;
    return encoded.packed.descriptor;
  };
  const auto success_descriptor = descriptor_for(is_cas ? ConSanMoiSampledSyncOutcome::CasSuccess
                                                 : candidate.site.returns_old_value.value_or(false)
                                                     ? ConSanMoiSampledSyncOutcome::RmwReturnsOld
                                                     : ConSanMoiSampledSyncOutcome::RmwNoReturn);
  const auto failure_descriptor =
      is_cas ? descriptor_for(ConSanMoiSampledSyncOutcome::CasFailure) : success_descriptor;
  if (!success_descriptor || !failure_descriptor)
    return std::nullopt;

  const uint16_t base = *options.scratch_vgpr;
  const uint16_t value = static_cast<uint16_t>(base + 2u);
  const uint16_t expected = static_cast<uint16_t>(base + 3u);
  const uint16_t saved_address = static_cast<uint16_t>(base + 8u);
  const uint16_t original_exec = static_cast<uint16_t>(*options.moi_exec_save_sgpr + 6u);
  const uint16_t temporary_exec = *options.moi_exec_save_sgpr;
  const uint64_t report_base = *options.moi_report_buffer_address;
  const uint64_t window_address =
      report_base + layout.sampled_causal_windows_offset +
      static_cast<uint64_t>(selected_slot) * sizeof(ConSanMoiSampledCausalWindow);
  const uint64_t watchpoint_address = report_base + layout.sampled_watchpoints_offset +
                                      static_cast<uint64_t>(selected_slot) * sizeof(uint64_t);
  const uint64_t metadata_address =
      report_base + layout.sampled_sync_metadata_offset +
      static_cast<uint64_t>(selected_slot) * sizeof(ConSanMoiSampledSyncMetadataPacked);

  std::vector<uint32_t> words;
  if (spill)
    words.insert(words.end(), spill->save_words.begin(), spill->save_words.end());
  if (address_plan.requires_materialization()) {
    const auto special = moi_special_state_sgprs(options);
    if (!special)
      return std::nullopt;
    const auto materialize = build_consan_moi_atomic_address_materialization(
        address_plan, special->vcc_save_sgpr, special->scc_save_sgpr, arch);
    if (!materialize)
      return std::nullopt;
    words.insert(words.end(), materialize->begin(), materialize->end());
  }
  if (guest_instruction_offset)
    *guest_instruction_offset = static_cast<uint32_t>(words.size() * sizeof(uint32_t));
  for (uint64_t offset = 0; offset < candidate.site.size; offset += sizeof(uint32_t)) {
    uint32_t word = 0;
    std::memcpy(&word, bytes.data() + candidate.site.file_offset + offset, sizeof(word));
    words.push_back(word);
  }
  if (!append_moi_flat_load_wait(words, arch))
    return std::nullopt;
  words.push_back(build_v_mov_b32_e32(saved_address,
                                      vector_source_vgpr(address_plan.result_address_vgpr), arch));
  words.push_back(build_v_mov_b32_e32(
      static_cast<uint16_t>(saved_address + 1u),
      vector_source_vgpr(static_cast<uint16_t>(address_plan.result_address_vgpr + 1u)), arch));
  InstructionSequence sequence(words);
  const auto collision_label = sequence.make_label();
  const auto restore_label = sequence.make_label();
  if (!append_save_moi_special_state(words, options, arch))
    return std::nullopt;
  const auto save_exec = instrumentation::build_s_mov_b64(original_exec, kRdna4ExecLo, arch);
  if (!save_exec)
    return std::nullopt;
  words.push_back(*save_exec);

  const auto window_lo =
      instrumentation::build_v_mov_b32_literal(base, static_cast<uint32_t>(window_address), arch);
  const auto window_hi = instrumentation::build_v_mov_b32_literal(
      static_cast<uint16_t>(base + 1u), static_cast<uint32_t>(window_address >> 32u), arch);
  if (!sequence.emit_all(window_lo, window_hi))
    return std::nullopt;

  const auto narrow_equal_literal = [&](uint32_t offset, uint32_t literal) -> bool {
    if (!append_load_u32_vgpr_at_offset(words, base, offset, value, arch))
      return false;
    const auto mov = instrumentation::build_v_mov_b32_literal(expected, literal, arch);
    const auto compare =
        instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(expected), value, arch);
    const auto narrow =
        instrumentation::build_s_and_saveexec_b64(temporary_exec, kRdna4VccLo, arch);
    if (!sequence.emit_all(mov, compare, narrow))
      return false;
    return sequence.emit_branch(restore_label, InstructionSequence::BranchKind::SccZero);
  };
  const auto narrow_equal_vgpr = [&](uint32_t offset, uint16_t expected_vgpr) -> bool {
    if (!append_load_u32_vgpr_at_offset(words, base, offset, value, arch))
      return false;
    const auto compare =
        instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(expected_vgpr), value, arch);
    const auto narrow =
        instrumentation::build_s_and_saveexec_b64(temporary_exec, kRdna4VccLo, arch);
    if (!sequence.emit_all(compare, narrow))
      return false;
    return sequence.emit_branch(restore_label, InstructionSequence::BranchKind::SccZero);
  };
  const auto narrow_equal_workgroup = [&](uint32_t offset,
                                          const ConSanMoiWorkgroupSource &source) -> bool {
    if (!source.scalar_src)
      return narrow_equal_literal(offset, 0u);
    words.push_back(build_v_mov_b32_e32(expected, *source.scalar_src, arch));
    if (source.mask_low_16) {
      const auto shift_left = instrumentation::build_v_lshlrev_b32(
          expected, scalar_positive_inline_u32(16), expected, arch);
      const auto shift_right = instrumentation::build_v_lshrrev_b32(
          expected, scalar_positive_inline_u32(16), expected, arch);
      if (!sequence.emit_all(shift_left, shift_right))
        return false;
    }
    if (source.shift_right_16) {
      const auto shift = instrumentation::build_v_lshrrev_b32(
          expected, scalar_positive_inline_u32(16), expected, arch);
      if (!sequence.emit(shift))
        return false;
    }
    return narrow_equal_vgpr(offset, expected);
  };
  if (!narrow_equal_literal(offsetof(ConSanMoiSampledCausalWindow, generation),
                            static_cast<uint32_t>(options.moi_report_generation)) ||
      !narrow_equal_literal(offsetof(ConSanMoiSampledCausalWindow, generation) + 4u,
                            static_cast<uint32_t>(options.moi_report_generation >> 32u)) ||
      !narrow_equal_literal(offsetof(ConSanMoiSampledCausalWindow, dispatch_id),
                            static_cast<uint32_t>(options.moi_report_dispatch_id)) ||
      !narrow_equal_literal(offsetof(ConSanMoiSampledCausalWindow, dispatch_id) + 4u,
                            static_cast<uint32_t>(options.moi_report_dispatch_id >> 32u)) ||
      !narrow_equal_workgroup(offsetof(ConSanMoiSampledCausalWindow, workgroup_x),
                              workgroup_sources->x) ||
      !narrow_equal_workgroup(offsetof(ConSanMoiSampledCausalWindow, workgroup_y),
                              workgroup_sources->y) ||
      !narrow_equal_workgroup(offsetof(ConSanMoiSampledCausalWindow, workgroup_z),
                              workgroup_sources->z) ||
      !narrow_equal_vgpr(offsetof(ConSanMoiSampledCausalWindow, epoch), *options.moi_epoch_vgpr) ||
      !narrow_equal_literal(offsetof(ConSanMoiSampledCausalWindow, first_entry), selected_slot) ||
      !narrow_equal_literal(offsetof(ConSanMoiSampledCausalWindow, entry_count), 1u) ||
      !narrow_equal_literal(offsetof(ConSanMoiSampledCausalWindow, publication_state),
                            static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Ready)) ||
      !narrow_equal_workgroup(offsetof(ConSanMoiSampledCausalWindow, cluster_workgroup_id),
                              workgroup_sources->cluster_workgroup_id))
    return std::nullopt;

  const auto watch_lo = instrumentation::build_v_mov_b32_literal(
      base, static_cast<uint32_t>(watchpoint_address), arch);
  const auto watch_hi = instrumentation::build_v_mov_b32_literal(
      static_cast<uint16_t>(base + 1u), static_cast<uint32_t>(watchpoint_address >> 32u), arch);
  if (!sequence.emit_all(watch_lo, watch_hi))
    return std::nullopt;
  const auto narrow_current_vcc = [&]() -> bool {
    const auto narrow =
        instrumentation::build_s_and_saveexec_b64(temporary_exec, kRdna4VccLo, arch);
    if (!sequence.emit(narrow))
      return false;
    return sequence.emit_branch(restore_label, InstructionSequence::BranchKind::SccZero);
  };
  const auto narrow_masked_low = [&](uint32_t mask, uint32_t wanted) -> bool {
    if (!append_load_u32_vgpr_at_offset(words, base, 0u, value, arch))
      return false;
    const auto masked = instrumentation::build_v_and_b32_literal(value, mask, value, arch);
    const auto expected_value = instrumentation::build_v_mov_b32_literal(expected, wanted, arch);
    if (!sequence.emit_all(masked, expected_value))
      return false;
    const auto equal =
        instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(expected), value, arch);
    if (!sequence.emit(equal))
      return false;
    return narrow_current_vcc();
  };
  if (!narrow_masked_low(static_cast<uint32_t>(consan_moi_sampled_watchpoint::valid_mask),
                         static_cast<uint32_t>(consan_moi_sampled_watchpoint::valid_mask)) ||
      !narrow_masked_low(static_cast<uint32_t>(consan_moi_sampled_watchpoint::consumed_mask), 0u) ||
      !append_load_u32_vgpr_at_offset(words, base, 0u, value, arch))
    return std::nullopt;
  const auto kind_shift = instrumentation::build_v_lshrrev_b32(
      value, scalar_positive_inline_u32(consan_moi_sampled_watchpoint::access_kind_shift), value,
      arch);
  const auto kind_mask = instrumentation::build_v_and_b32_literal(
      value, (1u << consan_moi_sampled_watchpoint::access_kind_bits) - 1u, value, arch);
  if (!sequence.emit_all(kind_shift, kind_mask))
    return std::nullopt;
  const auto kind_nonempty =
      instrumentation::build_v_cmp_ne_u32_vcc(scalar_positive_inline_u32(0u), value, arch);
  if (!sequence.emit(kind_nonempty))
    return std::nullopt;
  if (!narrow_current_vcc())
    return std::nullopt;
  const auto kind_supported =
      instrumentation::build_v_cmp_gt_u32_vcc(scalar_positive_inline_u32(3u), value, arch);
  if (!sequence.emit(kind_supported))
    return std::nullopt;
  if (!narrow_current_vcc() || !append_load_u32_vgpr_at_offset(words, base, 0u, value, arch))
    return std::nullopt;
  const auto owner_shift = instrumentation::build_v_lshrrev_b32(
      value, scalar_positive_inline_u32(consan_moi_sampled_watchpoint::owner_shift), value, arch);
  const auto owner_mask = instrumentation::build_v_and_b32_literal(
      value, consan_moi_sampled_watchpoint::max_owner, value, arch);
  const auto owner_equal = instrumentation::build_v_cmp_eq_u32_vcc(
      vector_source_vgpr(*options.moi_owner_vgpr), value, arch);
  const auto narrow_owner =
      instrumentation::build_s_and_saveexec_b64(temporary_exec, kRdna4VccLo, arch);
  if (!sequence.emit_all(owner_shift, owner_mask, owner_equal, narrow_owner))
    return std::nullopt;
  if (!sequence.emit_branch(restore_label, InstructionSequence::BranchKind::SccZero))
    return std::nullopt;

  if (!append_load_u32_vgpr_at_offset(words, base, 0u, value, arch))
    return std::nullopt;
  const auto epoch_shift = instrumentation::build_v_lshrrev_b32(
      value, scalar_positive_inline_u32(consan_moi_sampled_watchpoint::epoch_shift), value, arch);
  const auto epoch_mask = instrumentation::build_v_and_b32_literal(
      value, consan_moi_sampled_watchpoint::max_epoch, value, arch);
  const auto epoch_equal = instrumentation::build_v_cmp_eq_u32_vcc(
      vector_source_vgpr(*options.moi_epoch_vgpr), value, arch);
  if (!sequence.emit_all(epoch_shift, epoch_mask, epoch_equal))
    return std::nullopt;
  if (!narrow_current_vcc() || !append_load_u32_vgpr_at_offset(words, base, 0u, value, arch) ||
      !append_load_u32_vgpr_at_offset(words, base, sizeof(uint32_t), expected, arch))
    return std::nullopt;
  const auto generation_low = instrumentation::build_v_lshrrev_b32(
      value, scalar_positive_inline_u32(consan_moi_sampled_watchpoint::generation_shift), value,
      arch);
  const auto generation_high = instrumentation::build_v_lshlrev_b32(
      expected, scalar_positive_inline_u32(32u - consan_moi_sampled_watchpoint::generation_shift),
      expected, arch);
  const auto combine_generation =
      instrumentation::build_v_add_u32(value, vector_source_vgpr(expected), value, arch);
  const auto mask_generation = instrumentation::build_v_and_b32_literal(
      value, consan_moi_sampled_watchpoint::max_generation, value, arch);
  const auto expected_generation = instrumentation::build_v_mov_b32_literal(
      expected,
      static_cast<uint32_t>(options.moi_report_generation) &
          consan_moi_sampled_watchpoint::max_generation,
      arch);
  if (!sequence.emit_all(generation_low, generation_high, combine_generation, mask_generation,
                         expected_generation))
    return std::nullopt;
  const auto generation_equal =
      instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(expected), value, arch);
  if (!sequence.emit(generation_equal))
    return std::nullopt;
  if (!narrow_current_vcc())
    return std::nullopt;

  const auto matching_exec = instrumentation::build_s_mov_b64(temporary_exec, kRdna4ExecLo, arch);
  const auto mbcnt_lo = instrumentation::build_v_mbcnt_lo_u32_b32(
      value, temporary_exec, scalar_positive_inline_u32(0), arch);
  const auto mbcnt_hi = instrumentation::build_v_mbcnt_hi_u32_b32(
      value, static_cast<uint16_t>(temporary_exec + 1u), vector_source_vgpr(value), arch);
  const auto first =
      instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), value, arch);
  const auto narrow_first =
      instrumentation::build_s_and_saveexec_b64(temporary_exec, kRdna4VccLo, arch);
  if (!sequence.emit_all(matching_exec, mbcnt_lo, mbcnt_hi, first, narrow_first))
    return std::nullopt;

  const auto append_publication = [&](uint32_t descriptor) -> bool {
    const auto descriptor_lo = instrumentation::build_v_mov_b32_literal(
        base,
        static_cast<uint32_t>(metadata_address +
                              offsetof(ConSanMoiSampledSyncMetadataPacked, descriptor)),
        arch);
    const auto descriptor_hi = instrumentation::build_v_mov_b32_literal(
        static_cast<uint16_t>(base + 1u),
        static_cast<uint32_t>(
            (metadata_address + offsetof(ConSanMoiSampledSyncMetadataPacked, descriptor)) >> 32u),
        arch);
    const auto sentinel = instrumentation::build_v_mov_b32_literal(
        value, kConSanMoiSampledSyncPublishingDescriptor, arch);
    const auto zero = instrumentation::build_v_mov_b32_literal(expected, 0, arch);
    const auto claim = instrumentation::build_flat_atomic_cmpswap_b32(base, value, value, true,
                                                                      kRdna4ScopeDevice, arch);
    const auto claim_wait = instrumentation::build_s_wait_flat_load0(arch);
    if (!sequence.emit_all(descriptor_lo, descriptor_hi, sentinel, zero, claim, claim_wait))
      return false;
    const auto claimed =
        instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), value, arch);
    const auto narrow_claimed =
        instrumentation::build_s_and_saveexec_b64(temporary_exec, kRdna4VccLo, arch);
    if (!sequence.emit_all(claimed, narrow_claimed))
      return false;
    if (!sequence.emit_branch(collision_label, InstructionSequence::BranchKind::SccZero))
      return false;
    if (!append_store_u32_vgpr(words, metadata_address, saved_address, base, arch) ||
        !append_store_u32_vgpr(words, metadata_address + 4u,
                               static_cast<uint16_t>(saved_address + 1u), base, arch) ||
        !append_store_u32_literal(
            words, metadata_address + offsetof(ConSanMoiSampledSyncMetadataPacked, byte_count), 4u,
            base, arch) ||
        !append_store_u32_vgpr(
            words, metadata_address + offsetof(ConSanMoiSampledSyncMetadataPacked, epoch_before),
            *options.moi_epoch_vgpr, base, arch) ||
        !append_store_u32_vgpr(
            words, metadata_address + offsetof(ConSanMoiSampledSyncMetadataPacked, epoch_after),
            *options.moi_epoch_vgpr, base, arch))
      return false;
    const auto wait = instrumentation::build_s_wait_flat_store0(arch);
    const auto final_value = instrumentation::build_v_mov_b32_literal(value, descriptor, arch);
    const auto final_expected = instrumentation::build_v_mov_b32_literal(
        expected, kConSanMoiSampledSyncPublishingDescriptor, arch);
    const auto final_lo = instrumentation::build_v_mov_b32_literal(
        base,
        static_cast<uint32_t>(metadata_address +
                              offsetof(ConSanMoiSampledSyncMetadataPacked, descriptor)),
        arch);
    const auto final_hi = instrumentation::build_v_mov_b32_literal(
        static_cast<uint16_t>(base + 1u),
        static_cast<uint32_t>(
            (metadata_address + offsetof(ConSanMoiSampledSyncMetadataPacked, descriptor)) >> 32u),
        arch);
    const auto commit = instrumentation::build_flat_atomic_cmpswap_b32(base, value, value, true,
                                                                       kRdna4ScopeDevice, arch);
    const auto commit_wait = instrumentation::build_s_wait_flat_load0(arch);
    if (!sequence.emit_all(wait, final_value, final_expected, final_lo, final_hi, commit,
                           commit_wait))
      return false;
    const auto committed =
        instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(expected), value, arch);
    const auto narrow_committed =
        instrumentation::build_s_and_saveexec_b64(temporary_exec, kRdna4VccLo, arch);
    if (!sequence.emit_all(committed, narrow_committed))
      return false;
    if (!sequence.emit_branch(collision_label, InstructionSequence::BranchKind::SccZero))
      return false;
    if (!append_atomic_fetch_add_one_u32(
            words, report_base + offsetof(ConSanMoiReportHeader, sampled_sync_metadata_count),
            value, base, arch))
      return false;
    return sequence.emit_branch(restore_label, InstructionSequence::BranchKind::Unconditional);
  };

  if (is_cas) {
    const auto cas_failure_label = sequence.make_label();
    const uint16_t compare_vgpr = static_cast<uint16_t>(*candidate.site.data_vgpr + 1u);
    const auto success = instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(compare_vgpr),
                                                                 *candidate.site.dst_vgpr, arch);
    if (!success)
      return std::nullopt;
    words.push_back(*success);
    if (!sequence.emit_branch(cas_failure_label, InstructionSequence::BranchKind::VccZero) ||
        !append_publication(*success_descriptor) || !sequence.bind(cas_failure_label))
      return std::nullopt;
  }
  if (!append_publication(is_cas ? *failure_descriptor : *success_descriptor))
    return std::nullopt;

  // A failed claim/commit reaches here after its cumulative EXEC intersection
  // became empty. Reconstitute the guest mask and select exactly one active
  // lane so the collision counter is neither lost nor multiplied by wave size.
  const auto restore_collision_exec =
      instrumentation::build_s_mov_b64(kRdna4ExecLo, original_exec, arch);
  const auto collision_mask = instrumentation::build_s_mov_b64(temporary_exec, original_exec, arch);
  const auto collision_mbcnt_lo = instrumentation::build_v_mbcnt_lo_u32_b32(
      value, temporary_exec, scalar_positive_inline_u32(0), arch);
  const auto collision_mbcnt_hi = instrumentation::build_v_mbcnt_hi_u32_b32(
      value, static_cast<uint16_t>(temporary_exec + 1u), vector_source_vgpr(value), arch);
  const auto collision_first =
      instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), value, arch);
  const auto narrow_collision =
      instrumentation::build_s_and_saveexec_b64(temporary_exec, kRdna4VccLo, arch);
  if (!sequence.bind(collision_label) ||
      !sequence.emit_all(restore_collision_exec, collision_mask, collision_mbcnt_lo,
                         collision_mbcnt_hi, collision_first, narrow_collision))
    return std::nullopt;
  if (!append_atomic_fetch_add_one_u32(
          words, report_base + offsetof(ConSanMoiReportHeader, sampled_dropped_window_count), value,
          base, arch))
    return std::nullopt;
  const auto restore_exec = instrumentation::build_s_mov_b64(kRdna4ExecLo, original_exec, arch);
  if (!sequence.bind(restore_label) || !sequence.emit(restore_exec) ||
      !append_restore_moi_special_state(words, options, arch))
    return std::nullopt;
  if (spill)
    words.insert(words.end(), spill->restore_words.begin(), spill->restore_words.end());
  if (!append_moi_direct_or_indirect_return(words, cave_text_offset, return_text_offset, options,
                                            arch))
    return std::nullopt;

  if (!sequence.resolve_branches(arch))
    return std::nullopt;
  return words;
}
