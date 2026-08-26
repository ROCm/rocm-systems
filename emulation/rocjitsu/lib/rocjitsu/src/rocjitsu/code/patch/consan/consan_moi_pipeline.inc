void try_apply_fence_record_patch(std::span<const uint8_t> bytes, const ConSanOptions &options,
                                  rj_code_arch_t arch, ConSanResult &result) {
  if (!options.moi_track_atomics || options.moi_engine != ConSanMoiEngine::RecordReplay)
    return;
  const bool has_eligible = std::ranges::any_of(
      result.program_inventory.sync().moi_fence_candidates, &ConSanMoiFenceCandidate::eligible);
  if (!has_eligible)
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
  const ConSanMoiReportBufferLayout layout =
      resolve_record_replay_report_layout(options, options.moi_track_barriers,
                                          /*include_atomics=*/true, /*include_fences=*/true);
  const uint32_t max_candidates = std::min(options.max_patches, layout.fence_record_capacity);
  if (max_candidates == 0) {
    result.warnings.emplace_back("ConSan MOI fence record patch has no fence record capacity");
    return;
  }

  std::vector<FenceRecordCandidate> candidates;
  const MoiSyncInventoryIndex sync_index(result);
  append_fence_record_candidates(result, sync_index, candidates);
  if (!options.test_kernel_name_filter.empty()) {
    std::erase_if(candidates, [&](const FenceRecordCandidate &candidate) {
      return candidate.fence.container_name.find(options.test_kernel_name_filter) ==
             std::string::npos;
    });
  }
  retain_admitted_fence_record_candidates(result.observation_plan, candidates);
  std::ranges::sort(candidates, {}, [](const FenceRecordCandidate &candidate) {
    return candidate.fence.text_offset;
  });
  if (candidates.empty()) {
    result.warnings.emplace_back(
        "ConSan MOI fence record patch rejected all qualified communication events");
    return;
  }

  std::span<const uint8_t> active_bytes = bytes;
  if (result.modified)
    active_bytes = std::span<const uint8_t>(result.elf_bytes.data(), result.elf_bytes.size());
  AmdGpuCodeObject code_object(active_bytes.data(), active_bytes.size());
  CodeObjectPatcher patcher(code_object);
  const std::span<const uint8_t> old_text = patcher.text_bytes();
  if (old_text.empty()) {
    result.errors.emplace_back("ConSan MOI fence record patch found no .text section");
    return;
  }
  struct PlannedFenceRecord {
    FenceRecordCandidate candidate;
    ResolvedMoiScratchPlan resources;
    ConSanMoiAtomicAddressPlan address_plan;
    std::optional<VgprSpillSequence> spill;
    std::optional<InlineSgprSpillSequence> scalar_spill;
    std::optional<MoiPrivateRecordOwnerPlan> private_owner;
    std::optional<MoiPrivateEpochLayout> private_layout;
  };
  std::vector<PlannedFenceRecord> selected;
  MoiResourcePlanningState resource_state(bytes, arch, options, result);
  MoiSpillManagers spill_managers;
  MoiDescriptorVgprRequirements descriptor_requirements;
  MoiDescriptorSgprRequirements scalar_requirements;
  MoiDescriptorPrivateRequirements private_requirements;
  for (size_t candidate_index = 0; candidate_index < candidates.size(); ++candidate_index) {
    if (selected.size() == max_candidates)
      break;
    FenceRecordCandidate candidate = candidates[candidate_index];
    const ConSanMoiFenceCandidate &fence = candidate.fence;
    if (candidate.patch_text_offset > old_text.size() ||
        candidate.patch_size > old_text.size() - candidate.patch_text_offset) {
      result.errors.emplace_back("ConSan MOI fence record patch site is outside .text");
      return;
    }
    const bool overlaps = std::ranges::any_of(result.patches, [&](const ConSanPatchInfo &patch) {
      const uint64_t patch_end = patch.anchor_offset + std::max<uint32_t>(patch.original_size, 1);
      return candidate.patch_text_offset < patch_end &&
             patch.anchor_offset < candidate.patch_text_offset + candidate.patch_size;
    });
    if (overlaps) {
      result.warnings.emplace_back("ConSan MOI fence record patch skipped an overlapping patch");
      continue;
    }
    const uint16_t scratch_count = fence_record_scratch_count(candidate.communication_site, arch);
    const ConSanCandidateResourcePlan *existing_plan =
        resource_plan_for_site(result, ConSanResourceSiteKind::Fence, fence.text_offset);
    ConSanCandidateResourcePlan plan =
        existing_plan
            ? *existing_plan
            : plan_moi_resource_site(resource_state, options, ConSanResourceSiteKind::Fence,
                                     candidate_index, candidate.patch_text_offset,
                                     candidate.kernel_descriptor_file_offset, scratch_count,
                                     nullptr, &candidate.communication_site);
    if (!existing_plan) {
      if (candidate.patch_text_offset != fence.text_offset)
        plan.semantic_text_offset = fence.text_offset;
      result.resource_plans.push_back(plan);
    }
    const auto resources = resolve_moi_scratch_plan(plan, options, scratch_count);
    if (!resources) {
      result.warnings.emplace_back("ConSan MOI fence record patch has no usable resource plan");
      continue;
    }
    ResolvedMoiScratchPlan active_resources = *resources;
    ConSanOptions candidate_options = options;
    candidate_options.scratch_vgpr = resources->base;
    if (!apply_moi_transient_sgpr_assignment(candidate_options,
                                             resources->owner_descriptor_file_offsets)) {
      result.warnings.emplace_back(
          "ConSan MOI fence record patch found no common transient SGPR assignment");
      continue;
    }
    if (!apply_moi_persistent_vgpr_assignment(candidate_options,
                                              resources->owner_descriptor_file_offsets)) {
      result.warnings.emplace_back(
          "ConSan MOI fence record patch found no common persistent assignment");
      continue;
    }
    std::optional<MoiPrivateEpochLayout> private_layout;
    if (candidate_options.automatic_moi_private_epoch) {
      private_layout = build_moi_private_epoch_layout(
          result, active_resources, arch, result.warnings,
          /*include_owner=*/candidate_options.moi_owner_source == ConSanMoiOwnerSource::WorkitemId,
          /*include_workgroup_key=*/false,
          /*include_record_replay_workgroup=*/true);
      if (!private_layout)
        continue;
      candidate_options.moi_record_replay_workgroup_private_offsets =
          private_layout->record_replay_workgroup_offsets;
    }
    if (!apply_record_replay_entry_workgroup_assignment(candidate_options,
                                                        resources->owner_descriptor_file_offsets)) {
      result.warnings.emplace_back(
          "ConSan MOI fence record patch found no common entry workgroup assignment");
      continue;
    }
    std::optional<VgprSpillSequence> spill;
    std::optional<InlineSgprSpillSequence> scalar_spill;
    if (resources->source == ConSanRegisterAllocationSource::SpillRequired ||
        candidate_options.automatic_moi_record_replay_sgpr_spill) {
      uint32_t active_private_size = 0u;
      bool active_descriptor_missing = false;
      for (uint64_t descriptor_offset : resources->owner_descriptor_file_offsets) {
        const ConSanKernelInfo *kernel = kernel_for_descriptor(result, descriptor_offset);
        const auto active_kernel =
            kernel == nullptr
                ? code_object.kernels().end()
                : std::ranges::find_if(code_object.kernels(), [&](const AmdGpuKernelInfo &item) {
                    return item.name == kernel->name;
                  });
        if (active_kernel == code_object.kernels().end() ||
            active_kernel->descriptor_file_offset > active_bytes.size() ||
            sizeof(KD) > active_bytes.size() - active_kernel->descriptor_file_offset) {
          result.warnings.emplace_back(
              "ConSan MOI fence record spill could not resolve its active descriptor");
          active_descriptor_missing = true;
          break;
        }
        KD descriptor{};
        std::memcpy(&descriptor, active_bytes.data() + active_kernel->descriptor_file_offset,
                    sizeof(descriptor));
        active_private_size = std::max(active_private_size, descriptor.private_segment_fixed_size);
      }
      if (active_descriptor_missing)
        continue;
      active_resources.original_private_segment_size = active_private_size;
    }
    const ConSanMoiAtomicAddressPlan address_plan = plan_consan_moi_atomic_address(
        candidate.communication_site, resources->base, resources->count, resources->source, arch);
    if (!address_plan.supported()) {
      result.warnings.emplace_back(
          "ConSan MOI fence record patch rejected communication address: " +
          std::string(consan_moi_atomic_address_support_name(address_plan.support)));
      continue;
    }
    if (!options.moi_owner_vgpr && !candidate.kernel_descriptor_file_offset) {
      candidate.kernel_descriptor_file_offset =
          common_moi_record_owner_descriptor(bytes, *resources, arch, result.warnings);
      if (!candidate.kernel_descriptor_file_offset)
        continue;
    }
    std::optional<MoiPrivateRecordOwnerPlan> private_owner;
    if (moi_record_uses_private_owner(candidate_options)) {
      private_owner = resolve_moi_private_record_owner(bytes, active_resources, *private_layout,
                                                       result.warnings);
      if (!private_owner)
        continue;
    }
    const uint32_t private_layout_base = private_layout
                                             ? private_layout->ephemeral_base
                                             : active_resources.original_private_segment_size;
    if (resources->source == ConSanRegisterAllocationSource::SpillRequired ||
        moi_scalar_spill_requires_dynamic_vgpr_frame(result, active_resources, candidate_options)) {
      spill = build_moi_spill_sequence(result, active_resources, candidate_options, spill_managers,
                                       arch, result.warnings, private_layout_base);
      if (!spill)
        continue;
      note_spill_descriptor_requirements(private_requirements, active_resources, *spill);
    }
    if (candidate_options.automatic_moi_record_replay_sgpr_spill) {
      scalar_spill = build_inline_sgpr_spill_sequence(
          result, active_resources, candidate_options, spill_managers, arch, result.warnings,
          private_layout_base, spill ? &*spill : nullptr);
      if (!scalar_spill)
        continue;
      for (uint64_t descriptor_offset : active_resources.owner_descriptor_file_offsets) {
        auto [it, inserted] =
            private_requirements.emplace(descriptor_offset, scalar_spill->total_private_bytes);
        if (!inserted)
          it->second = std::max(it->second, scalar_spill->total_private_bytes);
      }
    }
    note_descriptor_requirements(descriptor_requirements, active_resources);
    note_moi_sgpr_requirements(scalar_requirements, active_resources, candidate_options, arch);
    if (private_layout) {
      for (uint64_t descriptor_offset : active_resources.owner_descriptor_file_offsets) {
        auto [it, inserted] =
            private_requirements.emplace(descriptor_offset, private_layout->ephemeral_base);
        if (!inserted)
          it->second = std::max(it->second, private_layout->ephemeral_base);
      }
    }
    selected.push_back({std::move(candidate), active_resources, address_plan, std::move(spill),
                        std::move(scalar_spill), private_owner, private_layout});
  }
  if (selected.empty()) {
    result.warnings.emplace_back("ConSan MOI fence record patch found no patchable fences");
    return;
  }
  if (!apply_descriptor_requirements(patcher, code_object, active_bytes, result,
                                     descriptor_requirements, arch, result.errors) ||
      !apply_sgpr_descriptor_requirements(patcher, result, code_object, scalar_requirements,
                                          result.errors) ||
      !apply_spill_descriptor_requirements(patcher, code_object, active_bytes, result,
                                           private_requirements, result.errors))
    return;

  std::vector<uint8_t> new_text(old_text.begin(), old_text.end());
  DbiPatchPlacementPlanner placement_planner(arch, old_text.size());
  MoiLocalNopIslandAllocator indirect_allocator(old_text, result, arch,
                                                kMoiLocalIndirectIslandWords);
  std::vector<ConSanPatchInfo> patches;
  for (const PlannedFenceRecord &planned : selected) {
    AtomicRecordCandidate scalar_clause_candidate;
    scalar_clause_candidate.site = planned.candidate.communication_site;
    scalar_clause_candidate.scalar_clause_text_offset = planned.candidate.scalar_clause_text_offset;
    if (!neutralize_atomic_scalar_clause(new_text, scalar_clause_candidate, arch, patches,
                                         result.errors))
      return;
  }
  uint32_t record_index = 0;
  struct DenseFenceRelay {
    uint64_t island_offset = 0;
    size_t dispatcher_patch_index = 0;
    uint64_t reservation_end = 0;
  };
  const auto find_dense_fence_relay =
      [&](uint64_t descriptor_offset) -> std::optional<DenseFenceRelay> {
    const auto owns_descriptor = [&](const ConSanPatchInfo &patch) {
      return std::ranges::find(patch.owner_descriptor_file_offsets, descriptor_offset) !=
             patch.owner_descriptor_file_offsets.end();
    };
    for (size_t host_index = result.patches.size(); host_index-- > 0u;) {
      const ConSanPatchInfo &host = result.patches[host_index];
      // A relocatable dense host records the complete local indirect island;
      // compact islands without displaced host ownership remain smaller.
      if (host.kind != ConSanPatchKind::TrampolineMoiIndirectBranchIsland ||
          host.original_size < kMoiLocalIndirectIslandWords * sizeof(uint32_t) ||
          !owns_descriptor(host) || host_index + 1u >= result.patches.size())
        continue;
      const ConSanPatchInfo &dispatcher = result.patches[host_index + 1u];
      if (dispatcher.kind != ConSanPatchKind::TrampolineMoiIndirectBranchIsland ||
          dispatcher.original_size != 0u || !owns_descriptor(dispatcher))
        continue;
      uint64_t reservation_end = std::numeric_limits<uint64_t>::max();
      for (const ConSanPatchInfo &patch : result.patches) {
        if ((patch.kind == ConSanPatchKind::TrampolineMoiInlineEpochBarrier ||
             patch.kind == ConSanPatchKind::TrampolineMoiBarrierRecord) &&
            owns_descriptor(patch) && patch.trampoline_offset > dispatcher.trampoline_offset) {
          reservation_end = std::min(reservation_end, patch.trampoline_offset);
        }
      }
      if (reservation_end != std::numeric_limits<uint64_t>::max()) {
        return DenseFenceRelay{.island_offset = host.anchor_offset + sizeof(uint32_t),
                               .dispatcher_patch_index = host_index + 1u,
                               .reservation_end = reservation_end};
      }
    }
    return std::nullopt;
  };
  for (const PlannedFenceRecord &planned : selected) {
    const ConSanMoiFenceCandidate &fence = planned.candidate.fence;
    const uint64_t cave_text_offset = placement_planner.appended_end();
    ConSanOptions candidate_options = options;
    candidate_options.scratch_vgpr = planned.resources.base;
    if (planned.private_layout) {
      candidate_options.moi_record_replay_workgroup_private_offsets =
          planned.private_layout->record_replay_workgroup_offsets;
    }
    if (!apply_moi_transient_sgpr_assignment(candidate_options,
                                             planned.resources.owner_descriptor_file_offsets)) {
      result.errors.emplace_back(
          "ConSan MOI fence record patch lost its transient SGPR assignment");
      return;
    }
    if (!apply_record_replay_entry_workgroup_assignment(
            candidate_options, planned.resources.owner_descriptor_file_offsets)) {
      result.errors.emplace_back(
          "ConSan MOI fence record patch lost its entry workgroup assignment");
      return;
    }
    const std::optional<uint16_t> dense_call_return_sgpr =
        candidate_options.automatic_moi_record_replay_sgpr_spill
            ? candidate_options.moi_record_replay_call_return_sgpr
            : (candidate_options.moi_exec_save_sgpr
                   ? std::optional<uint16_t>(
                         static_cast<uint16_t>(*candidate_options.moi_exec_save_sgpr + 6u))
                   : std::nullopt);
    const std::optional<uint16_t> dense_dispatch_key_sgpr =
        candidate_options.automatic_moi_record_replay_sgpr_spill
            ? candidate_options.moi_record_replay_dispatch_key_sgpr
            : (candidate_options.moi_exec_save_sgpr
                   ? std::optional<uint16_t>(
                         static_cast<uint16_t>(*candidate_options.moi_exec_save_sgpr + 5u))
                   : std::nullopt);
    // These scalars are used only when the optionals are engaged and a dense
    // relay is selected. Materializing them here keeps that invariant explicit
    // without asking GCC 13 to retain optional engagement across the long
    // placement path.
    const uint16_t dense_call_return = dense_call_return_sgpr.value_or(0u);
    const uint16_t dense_dispatch_key = dense_dispatch_key_sgpr.value_or(0u);
    const auto dense_indirect_state = moi_indirect_jump_sgprs(candidate_options);
    const auto dense_special_state = moi_special_state_sgprs(candidate_options);
    const uint64_t patch_text_offset = planned.candidate.patch_text_offset;
    const uint32_t patch_size = planned.candidate.patch_size;
    const bool direct_entry =
        compute_sopp_branch_simm16(patch_text_offset, cave_text_offset).has_value();
    uint64_t indirect_entry_island = 0;
    bool has_indirect_entry_island = false;
    if (!direct_entry) {
      const auto claimed_island = indirect_allocator.claim_reachable(patch_text_offset);
      if (claimed_island) {
        indirect_entry_island = *claimed_island;
        has_indirect_entry_island = true;
      }
    }
    std::vector<uint32_t> displaced_tail_words;
    uint32_t displaced_original_size = patch_size;
    bool entry_island_at_anchor = false;
    bool compact_entry_without_scc = false;
    if (!direct_entry && !has_indirect_entry_island && is_rdna4_family_arch(arch)) {
      constexpr size_t kCompactEntryIslandWords = 7u;
      BasicBlock *entry_block = block_for_offset(resource_state.block_index, patch_text_offset);
      const Instruction *entry_instruction = instruction_at(entry_block, patch_text_offset);
      bool found_owner_context = false;
      bool scc_live = false;
      if (entry_block != nullptr && entry_instruction != nullptr) {
        for (const MoiKernelResourceContext &context : resource_state.contexts) {
          if (!context.block_set.contains(entry_block))
            continue;
          found_owner_context = true;
          scc_live |=
              context.liveness->live_before(*entry_instruction).contains({RegClass::SCC, 0u, 1u});
        }
      }
      compact_entry_without_scc = found_owner_context && !scc_live;
      const size_t required_island_words =
          compact_entry_without_scc ? kCompactEntryIslandWords : kMoiLocalIndirectIslandWords;
      std::vector<uint32_t> displaced_words;
      std::vector<std::string> decode_errors;
      const size_t patch_words = patch_size / sizeof(uint32_t);
      while (displaced_words.size() < std::max(required_island_words, patch_words)) {
        auto decoded = decode_relocatable_entry_instruction(
            old_text, patch_text_offset + displaced_words.size() * sizeof(uint32_t), arch,
            decode_errors, resource_state.decoder.get());
        if (!decoded) {
          displaced_words.clear();
          break;
        }
        displaced_words.insert(displaced_words.end(), decoded->begin(), decoded->end());
      }
      const uint64_t displaced_end = patch_text_offset + displaced_words.size() * sizeof(uint32_t);
      const bool overlaps_existing_patch =
          std::ranges::any_of(result.patches, [&](const ConSanPatchInfo &patch) {
            const uint64_t patch_end =
                patch.anchor_offset + std::max<uint32_t>(patch.original_size, 1u);
            return patch_text_offset < patch_end && patch.anchor_offset < displaced_end;
          });
      const bool swallows_another_fence =
          std::ranges::any_of(candidates, [&](const FenceRecordCandidate &other) {
            return other.patch_text_offset != patch_text_offset &&
                   other.patch_text_offset > patch_text_offset &&
                   other.patch_text_offset < displaced_end;
          });
      if (displaced_words.size() >= required_island_words &&
          patch_words <= displaced_words.size() && !overlaps_existing_patch &&
          !swallows_another_fence &&
          moi_entry_window_is_single_entry(resource_state.block_index, patch_text_offset,
                                           displaced_end - patch_text_offset) &&
          displaced_words.size() * sizeof(uint32_t) <= std::numeric_limits<uint32_t>::max()) {
        displaced_tail_words.assign(displaced_words.begin() + patch_words, displaced_words.end());
        displaced_original_size = static_cast<uint32_t>(displaced_words.size() * sizeof(uint32_t));
        indirect_entry_island = patch_text_offset;
        has_indirect_entry_island = true;
        entry_island_at_anchor = true;
      }
    }
    const uint64_t return_text_offset = patch_text_offset + displaced_original_size;
    auto cave_words = build_fence_record_cave_words(
        active_bytes, planned.candidate, planned.address_plan, candidate_options,
        planned.spill ? &*planned.spill : nullptr,
        planned.scalar_spill ? &*planned.scalar_spill : nullptr, arch, record_index,
        layout.fence_record_capacity, layout.fence_records_offset, cave_text_offset,
        return_text_offset, std::nullopt, displaced_tail_words,
        planned.private_owner ? std::optional<uint32_t>(planned.private_owner->offset)
                              : std::nullopt,
        planned.private_owner ? std::optional<uint16_t>(planned.private_owner->shift)
                              : std::nullopt,
        result.errors);
    if (!cave_words)
      return;
    std::optional<DenseFenceRelay> dense_relay;
    if (!direct_entry && !has_indirect_entry_island && arch == ROCJITSU_CODE_ARCH_CDNA5 &&
        planned.candidate.kernel_descriptor_file_offset && dense_call_return_sgpr &&
        dense_dispatch_key_sgpr && dense_indirect_state && dense_special_state) {
      dense_relay = find_dense_fence_relay(*planned.candidate.kernel_descriptor_file_offset);
      if (dense_relay &&
          !compute_sopp_branch_simm16(patch_text_offset, dense_relay->island_offset)) {
        dense_relay.reset();
      }
      if (dense_relay) {
        cave_words = build_fence_record_cave_words(
            active_bytes, planned.candidate, planned.address_plan, candidate_options,
            planned.spill ? &*planned.spill : nullptr,
            planned.scalar_spill ? &*planned.scalar_spill : nullptr, arch, record_index,
            layout.fence_record_capacity, layout.fence_records_offset, cave_text_offset,
            return_text_offset, dense_call_return,
            /*displaced_tail_words=*/{},
            planned.private_owner ? std::optional<uint32_t>(planned.private_owner->offset)
                                  : std::nullopt,
            planned.private_owner ? std::optional<uint16_t>(planned.private_owner->shift)
                                  : std::nullopt,
            result.errors);
        if (!cave_words)
          return;
      }
    }
    if (!direct_entry && !has_indirect_entry_island && !dense_relay) {
      result.warnings.emplace_back(
          "ConSan MOI fence record patch found no reachable indirect entry island in " +
          fence.container_name);
      continue;
    }
    std::string placement_error;
    const auto placement = placement_planner.plan_indirect_appended(
        patch_text_offset, displaced_original_size, cave_words->size() * sizeof(uint32_t),
        &placement_error);
    if (!placement) {
      result.errors.emplace_back("ConSan MOI fence record placement failed: " + placement_error);
      return;
    }
    if (placement->body_offset != cave_text_offset || new_text.size() != placement->body_offset) {
      result.errors.emplace_back("ConSan MOI fence record patch has a stale cave mapping");
      return;
    }
    const uint64_t entry_target =
        dense_relay ? dense_relay->island_offset
                    : (has_indirect_entry_island ? indirect_entry_island : placement->body_offset);
    const auto fwd = compute_sopp_branch_simm16(placement->anchor_offset, entry_target);
    if (!fwd) {
      result.errors.emplace_back("ConSan MOI fence record forward branch is out of range");
      return;
    }
    std::vector<uint32_t> anchor_words(placement->original_size / sizeof(uint32_t),
                                       build_s_nop(0, arch));
    if (anchor_words.empty()) {
      result.errors.emplace_back("ConSan MOI fence record patch empty anchor");
      return;
    }
    if (entry_island_at_anchor) {
      // The indirect island written below occupies the relocated prefix.
    } else if (dense_relay) {
      const auto call = instrumentation::build_s_call_i64(dense_call_return, *fwd, arch);
      if (!call) {
        result.errors.emplace_back("ConSan MOI fence record patch could not encode dense call");
        return;
      }
      anchor_words.front() = *call;
    } else {
      anchor_words.front() = build_s_branch(*fwd, arch);
    }
    std::memcpy(new_text.data() + patch_text_offset, anchor_words.data(),
                anchor_words.size() * sizeof(uint32_t));
    if (has_indirect_entry_island) {
      if (compact_entry_without_scc && entry_island_at_anchor) {
        constexpr size_t kCompactEntryIslandWords = 7u;
        const auto jump_sgprs = moi_indirect_jump_sgprs(candidate_options);
        std::vector<uint32_t> island_words;
        bool encoded = false;
        if (jump_sgprs) {
          const uint64_t getpc_offset = indirect_entry_island;
          island_words.push_back(build_s_getpc_b64(jump_sgprs->pc_sgpr, arch));
          const int64_t delta = static_cast<int64_t>(placement->body_offset) -
                                static_cast<int64_t>(getpc_offset + sizeof(uint32_t));
          const bool delta_encoded = append_pc_delta_builder(
              island_words, arch, jump_sgprs->pc_sgpr, delta, /*minimum_words=*/0u,
              /*prefer_literal64=*/arch == ROCJITSU_CODE_ARCH_CDNA5);
          if (const auto delay = instrumentation::build_salu_dependency_delay(arch);
              delta_encoded && delay) {
            island_words.push_back(*delay);
            island_words.push_back(build_s_setpc_b64(jump_sgprs->pc_sgpr, arch));
            encoded = true;
          }
        }
        if (!encoded || island_words.size() > kCompactEntryIslandWords) {
          result.errors.emplace_back(
              "ConSan MOI fence record patch could not encode its SCC-dead compact entry");
          return;
        }
        while (island_words.size() < kCompactEntryIslandWords)
          island_words.push_back(build_s_nop(0, arch));
        std::memcpy(new_text.data() + indirect_entry_island, island_words.data(),
                    island_words.size() * sizeof(uint32_t));
        ConSanPatchInfo island_info;
        island_info.kind = ConSanPatchKind::TrampolineMoiIndirectBranchIsland;
        island_info.anchor_offset = patch_text_offset;
        island_info.trampoline_offset = indirect_entry_island;
        // The fence patch owns the entire decoded prefix. A multiword final
        // guest instruction can make that prefix one word larger than the
        // compact island; its trailing NOP is part of the same exact overlap.
        island_info.trampoline_size = displaced_original_size;
        island_info.owner_descriptor_file_offsets = planned.resources.owner_descriptor_file_offsets;
        patches.push_back(std::move(island_info));
      } else if (!emit_moi_local_indirect_entry_island(
                     new_text, indirect_entry_island, placement->body_offset, patch_text_offset,
                     candidate_options, planned.resources.owner_descriptor_file_offsets, arch,
                     patches, result.errors, "ConSan MOI fence record patch")) {
        return;
      }
    }
    append_words_bytes(new_text, *cave_words);

    if (dense_relay) {
      ConSanPatchInfo &dispatcher = result.patches[dense_relay->dispatcher_patch_index];
      if (dispatcher.trampoline_size < sizeof(uint32_t) ||
          dispatcher.trampoline_offset + dispatcher.trampoline_size > new_text.size()) {
        result.errors.emplace_back("ConSan MOI fence record patch found stale dense dispatcher");
        return;
      }
      std::vector<uint32_t> dispatcher_words(dispatcher.trampoline_size / sizeof(uint32_t));
      std::memcpy(dispatcher_words.data(), new_text.data() + dispatcher.trampoline_offset,
                  dispatcher.trampoline_size);
      if (dispatcher_words.empty() || dispatcher_words.back() != build_s_endpgm(arch)) {
        result.errors.emplace_back(
            "ConSan MOI fence record patch found unterminated dense dispatcher");
        return;
      }
      dispatcher_words.pop_back();
      const auto compare =
          instrumentation::build_s_cmp_eq_u32(dense_dispatch_key, /*literal=*/255u, arch);
      if (!compare) {
        result.errors.emplace_back(
            "ConSan MOI fence record patch could not extend dense dispatcher");
        return;
      }
      dispatcher_words.push_back(*compare);
      dispatcher_words.push_back(
          static_cast<uint32_t>(patch_text_offset + sizeof(uint32_t) - dense_relay->island_offset));
      std::vector<uint32_t> matched_words;
      const uint64_t matched_offset =
          dispatcher.trampoline_offset + (dispatcher_words.size() + 1u) * sizeof(uint32_t);
      if (!append_moi_scc_preserving_indirect_jump(
              matched_words, matched_offset, placement->body_offset, dense_indirect_state->pc_sgpr,
              dense_special_state->scc_save_sgpr, /*capture_scc=*/false, arch)) {
        result.errors.emplace_back(
            "ConSan MOI fence record patch could not encode dense dispatch target");
        return;
      }
      const auto skip =
          instrumentation::build_s_cbranch_scc0(static_cast<int16_t>(matched_words.size()), arch);
      if (!skip) {
        result.errors.emplace_back(
            "ConSan MOI fence record patch could not encode dense dispatch branch");
        return;
      }
      dispatcher_words.push_back(*skip);
      dispatcher_words.insert(dispatcher_words.end(), matched_words.begin(), matched_words.end());
      dispatcher_words.push_back(build_s_endpgm(arch));
      const uint64_t dispatcher_bytes = dispatcher_words.size() * sizeof(uint32_t);
      if (dispatcher.trampoline_offset + dispatcher_bytes > dense_relay->reservation_end) {
        result.errors.emplace_back(
            "ConSan MOI fence record patch exhausted dense dispatcher reservation");
        return;
      }
      std::memcpy(new_text.data() + dispatcher.trampoline_offset, dispatcher_words.data(),
                  dispatcher_bytes);
      dispatcher.trampoline_size = static_cast<uint32_t>(dispatcher_bytes);
    }

    ConSanPatchInfo info;
    info.kind = ConSanPatchKind::TrampolineMoiFenceRecord;
    info.anchor_offset = patch_text_offset;
    info.trampoline_offset = cave_text_offset;
    info.original_size = displaced_original_size;
    info.trampoline_size = static_cast<uint32_t>(cave_words->size() * sizeof(uint32_t));
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
    patches.push_back(std::move(info));
    ++record_index;
  }
  if (!replace_consan_text(patcher, new_text, options, "MOI fence record patch", result)) {
    return;
  }
  result.elf_bytes = std::move(patcher).emit();
  result.patches.insert(result.patches.end(), patches.begin(), patches.end());
  result.modified = true;
}

void append_moi_operational_resource_plans(MoiResourcePlanningState &state,
                                           const ConSanOptions &options, ConSanResult &result) {
  const MoiSyncInventoryIndex sync_index(result);
  if (options.moi_track_barriers) {
    std::vector<BarrierRecordCandidate> candidates;
    for (const ConSanKernelInfo &kernel : result.program_inventory.kernels())
      append_barrier_epoch_candidates(kernel, candidates);
    for (const ConSanFunctionInfo &function : result.program_inventory.functions())
      append_barrier_epoch_candidates(function, candidates);
    apply_test_container_filter(candidates, options);
    if (options.moi_engine == ConSanMoiEngine::InlineShadow) {
      std::erase_if(candidates, [](const BarrierRecordCandidate &candidate) {
        return candidate.site.operation == ConSanBarrierSite::Operation::Signal;
      });
    }
    if (!canonicalize_barrier_record_candidates_by_physical_site(candidates, result))
      return;
    retain_admitted_barrier_record_candidates(result.observation_plan, candidates);
    std::ranges::sort(candidates, {}, [](const BarrierRecordCandidate &candidate) {
      return candidate.site.text_offset;
    });
    erase_redundant_adjacent_full_barrier_candidates(candidates);
    for (size_t index = 0; index < candidates.size(); ++index) {
      const BarrierRecordCandidate &candidate = candidates[index];
      const auto scratch_count_for = [](const ConSanOptions &site_options) {
        return static_cast<uint16_t>(site_options.moi_engine == ConSanMoiEngine::Sampled
                                         ? (site_options.automatic_moi_private_epoch ||
                                                    moi_uses_scalar_persistent_state(site_options)
                                                ? 9u
                                                : 7u)
                                     : site_options.moi_engine == ConSanMoiEngine::InlineShadow
                                         ? inline_shadow_barrier_scratch_count(site_options)
                                         : 6u);
      };
      ConSanOptions site_options = options;
      uint16_t scratch_count = scratch_count_for(site_options);
      ConSanCandidateResourcePlan plan = plan_moi_resource_site(
          state, site_options, ConSanResourceSiteKind::Barrier, index, candidate.site.text_offset,
          candidate.kernel_descriptor_file_offset, scratch_count);
      ConSanOptions owner_options = site_options;
      if (apply_moi_persistent_vgpr_assignment(owner_options, plan.owner_descriptor_file_offsets)) {
        const uint16_t owner_scratch_count = scratch_count_for(owner_options);
        if (owner_scratch_count != scratch_count) {
          plan =
              plan_moi_resource_site(state, owner_options, ConSanResourceSiteKind::Barrier, index,
                                     candidate.site.text_offset,
                                     candidate.kernel_descriptor_file_offset, owner_scratch_count);
        }
      }
      result.resource_plans.push_back(std::move(plan));
    }
  }
  if (options.moi_track_atomics) {
    std::vector<AtomicRecordCandidate> candidates;
    if (options.moi_engine == ConSanMoiEngine::InlineShadow) {
      for (const ConSanKernelInfo &kernel : result.program_inventory.kernels())
        append_inline_atomic_candidates(kernel, sync_index, candidates);
      for (const ConSanFunctionInfo &function : result.program_inventory.functions())
        append_inline_atomic_candidates(function, sync_index, candidates);
      candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                                      [&](const auto &candidate) {
                                        return classify_consan_moi_inline_atomic_support(
                                                   candidate.site, candidate.event_kind, state.arch,
                                                   candidate.is_rmw) !=
                                               ConSanMoiInlineAtomicSupport::Supported;
                                      }),
                       candidates.end());
    } else {
      for (const ConSanKernelInfo &kernel : result.program_inventory.kernels())
        append_atomic_record_candidates(kernel, result, sync_index, candidates);
      for (const ConSanFunctionInfo &function : result.program_inventory.functions())
        append_atomic_record_candidates(function, result, sync_index, candidates);
    }
    std::vector<AtomicRecordCandidate> ordinary_candidates;
    append_ordered_ordinary_atomic_candidates(result, sync_index, ordinary_candidates);
    std::ranges::copy_if(ordinary_candidates, std::back_inserter(candidates),
                         [&](const AtomicRecordCandidate &candidate) {
                           return options.moi_engine != ConSanMoiEngine::RecordReplay ||
                                  !ordinary_atomic_candidate_is_covered_by_eligible_fence(
                                      result, sync_index, candidate);
                         });
    apply_test_container_filter(candidates, options);
    if (!canonicalize_atomic_record_candidates_by_physical_site(candidates, result))
      return;
    retain_admitted_atomic_record_candidates(result.observation_plan, candidates);
    for (size_t index = 0; index < candidates.size(); ++index) {
      const AtomicRecordCandidate &candidate = candidates[index];
      const uint16_t scratch_count = options.moi_engine == ConSanMoiEngine::Sampled
                                         ? sampled_atomic_scratch_count()
                                     : options.moi_engine == ConSanMoiEngine::InlineShadow
                                         ? inline_shadow_atomic_scratch_count(state.arch)
                                         : atomic_record_scratch_count(candidate.site, state.arch);
      result.resource_plans.push_back(plan_moi_resource_site(
          state, options, ConSanResourceSiteKind::Atomic, index, candidate.site.text_offset,
          candidate.kernel_descriptor_file_offset, scratch_count, nullptr, &candidate.site));
    }
  }
  if (options.moi_track_atomics && options.moi_engine == ConSanMoiEngine::RecordReplay) {
    std::vector<FenceRecordCandidate> candidates;
    append_fence_record_candidates(result, sync_index, candidates);
    if (!options.test_kernel_name_filter.empty()) {
      std::erase_if(candidates, [&](const FenceRecordCandidate &candidate) {
        return candidate.fence.container_name.find(options.test_kernel_name_filter) ==
               std::string::npos;
      });
    }
    retain_admitted_fence_record_candidates(result.observation_plan, candidates);
    for (size_t index = 0; index < candidates.size(); ++index) {
      const FenceRecordCandidate &candidate = candidates[index];
      ConSanCandidateResourcePlan plan = plan_moi_resource_site(
          state, options, ConSanResourceSiteKind::Fence, index, candidate.patch_text_offset,
          candidate.kernel_descriptor_file_offset,
          fence_record_scratch_count(candidate.communication_site, state.arch), nullptr,
          &candidate.communication_site);
      if (candidate.patch_text_offset != candidate.fence.text_offset)
        plan.semantic_text_offset = candidate.fence.text_offset;
      result.resource_plans.push_back(std::move(plan));
    }
  }
}

[[nodiscard]] std::optional<ConSanResourceSiteKind>
moi_instrumentation_patch_site_kind(const ConSanPatchInfo &patch) {
  switch (patch.kind) {
  case ConSanPatchKind::InlineMoiAccessRecordStore:
  case ConSanPatchKind::TrampolineMoiAccessRecordStore:
  case ConSanPatchKind::InlineMoiExactShadowStore:
  case ConSanPatchKind::TrampolineMoiExactShadowStore:
  case ConSanPatchKind::InlineMoiSampledWatchpointStore:
  case ConSanPatchKind::TrampolineMoiSampledWatchpointStore:
    return ConSanResourceSiteKind::Access;
  case ConSanPatchKind::TrampolineMoiBarrierRecord:
  case ConSanPatchKind::TrampolineMoiInlineEpochBarrier:
  case ConSanPatchKind::InlineMalformedBarrierAbort:
    return ConSanResourceSiteKind::Barrier;
  case ConSanPatchKind::TrampolineMoiInlineAtomicOrdering:
  case ConSanPatchKind::TrampolineMoiAtomicRecord:
    return ConSanResourceSiteKind::Atomic;
  case ConSanPatchKind::TrampolineMoiFenceRecord:
    return ConSanResourceSiteKind::Fence;
  case ConSanPatchKind::TrampolineMoiSampledSyncMetadata:
    return std::nullopt;
  default:
    return std::nullopt;
  }
}

void finalize_moi_site_lowering_outcomes(ConSanResult &result) {
  struct SiteKey {
    ConSanResourceSiteKind kind = ConSanResourceSiteKind::Access;
    uint64_t text_offset = 0;

    bool operator==(const SiteKey &) const = default;
  };
  struct SiteKeyHash {
    size_t operator()(const SiteKey &key) const {
      const size_t offset_hash = std::hash<uint64_t>{}(key.text_offset);
      return offset_hash ^ (static_cast<size_t>(key.kind) + 0x9e3779b9u + (offset_hash << 6u) +
                            (offset_hash >> 2u));
    }
  };

  std::unordered_map<uint64_t, ConSanResourceSiteKind> resource_kind_by_offset;
  std::unordered_map<SiteKey, ConSanRegisterPlanReason, SiteKeyHash> failed_sites;
  resource_kind_by_offset.reserve(result.resource_plans.size());
  failed_sites.reserve(result.resource_plans.size());
  for (const ConSanCandidateResourcePlan &plan : result.resource_plans) {
    const uint64_t semantic_offset = plan.semantic_text_offset.value_or(plan.text_offset);
    resource_kind_by_offset.try_emplace(semantic_offset, plan.site_kind);
    if (plan.source == ConSanRegisterAllocationSource::Unsupported)
      failed_sites.try_emplace({plan.site_kind, semantic_offset}, plan.reason);
  }

  std::unordered_map<std::string_view, const ConSanSyncEvent *> event_by_identity;
  std::unordered_map<uint64_t, const ConSanSyncEvent *> barrier_event_by_offset;
  event_by_identity.reserve(result.program_inventory.sync().sync_events.size());
  barrier_event_by_offset.reserve(result.program_inventory.sync().sync_events.size());
  for (const ConSanSyncEvent &event : result.program_inventory.sync().sync_events) {
    event_by_identity.try_emplace(event.identity, &event);
    if (event.kind == ConSanSyncEventKind::Barrier)
      barrier_event_by_offset.try_emplace(event.text_offset, &event);
  }
  std::unordered_map<std::string_view, const ConSanSyncSequence *> sequence_by_member;
  for (const ConSanSyncSequence &sequence : result.program_inventory.sync().sync_sequences) {
    for (const std::string &identity : sequence.member_event_identities) {
      const auto [found, inserted] = sequence_by_member.try_emplace(identity, &sequence);
      if (!inserted)
        found->second = nullptr;
    }
  }

  std::unordered_set<SiteKey, SiteKeyHash> patched_sites;
  patched_sites.reserve(result.patches.size());
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.phase != ConSanPatchPhase::Instrumentation)
      continue;
    if (patch.kind != ConSanPatchKind::TrampolineMoiSampledSyncMetadata) {
      const auto patch_kind = moi_instrumentation_patch_site_kind(patch);
      if (patch_kind) {
        const auto plan = std::ranges::find_if(result.resource_plans, [&](const auto &candidate) {
          return candidate.site_kind == *patch_kind && candidate.text_offset == patch.anchor_offset;
        });
        const uint64_t semantic_offset =
            plan == result.resource_plans.end()
                ? patch.anchor_offset
                : plan->semantic_text_offset.value_or(plan->text_offset);
        patched_sites.insert({*patch_kind, semantic_offset});
      }
      continue;
    }

    if (patch.sampled_access_kind == ConSanLdsAccessKind::Atomic)
      patched_sites.insert({ConSanResourceSiteKind::Atomic, patch.anchor_offset});
    const auto patch_kind = resource_kind_by_offset.find(patch.anchor_offset);
    if (patch.sampled_access_kind != ConSanLdsAccessKind::Atomic &&
        patch_kind != resource_kind_by_offset.end())
      patched_sites.insert({patch_kind->second, patch.anchor_offset});
    if (patch.covered_sync_event_count == 0)
      continue;
    const auto anchor_event = barrier_event_by_offset.find(patch.anchor_offset);
    if (anchor_event == barrier_event_by_offset.end())
      continue;
    const auto sequence = sequence_by_member.find(anchor_event->second->identity);
    if (sequence == sequence_by_member.end() || sequence->second == nullptr ||
        sequence->second->member_event_identities.size() != patch.covered_sync_event_count)
      continue;
    for (const std::string &identity : sequence->second->member_event_identities) {
      const auto event = event_by_identity.find(identity);
      if (event != event_by_identity.end() && event->second->kind == ConSanSyncEventKind::Barrier)
        patched_sites.insert({ConSanResourceSiteKind::Barrier, event->second->text_offset});
    }
  }

  // Publish every admitted intent directly from resource and patch placement
  // state. Semantic applicability remains immutable in the observation plan;
  // lowering cannot overwrite it with a second disposition vocabulary.
  for (const ConSanProbeIntent &intent : result.observation_plan.probe_intents) {
    ConSanResourceSiteKind site_kind = ConSanResourceSiteKind::Access;
    uint64_t disposition_offset = intent.physical_site.original_text_offset;
    std::string_view subject = "access";
    switch (intent.kind) {
    case ConSanProbeIntentKind::RedundantAccessObservation:
    case ConSanProbeIntentKind::AccessRecord:
    case ConSanProbeIntentKind::SampledAccess:
    case ConSanProbeIntentKind::ExactShadowAccess:
      break;
    case ConSanProbeIntentKind::BarrierRecord:
    case ConSanProbeIntentKind::SampledBarrierEpoch:
    case ConSanProbeIntentKind::ExactBarrierEpoch:
      site_kind = ConSanResourceSiteKind::Barrier;
      subject = "barrier";
      break;
    case ConSanProbeIntentKind::AtomicRecord:
    case ConSanProbeIntentKind::SampledAtomicOrdering:
    case ConSanProbeIntentKind::ExactAtomicOrdering:
      site_kind = ConSanResourceSiteKind::Atomic;
      subject = "atomic";
      break;
    case ConSanProbeIntentKind::FenceRecord:
      site_kind = ConSanResourceSiteKind::Fence;
      subject = "fence";
      break;
    case ConSanProbeIntentKind::AtomicAddressCapture: {
      site_kind = ConSanResourceSiteKind::Atomic;
      subject = "atomic";
      const auto owning_fence = std::ranges::find_if(
          result.observation_plan.fence_site_decisions,
          [&](const ConSanFenceSiteDecision &decision) {
            return decision.kind == ConSanSiteDecisionKind::Admitted &&
                   std::ranges::find(decision.intent_ids, intent.id) != decision.intent_ids.end() &&
                   std::ranges::any_of(decision.intent_ids, [&](ConSanProbeIntentId id) {
                     const ConSanProbeIntent *candidate = result.observation_plan.intent(id);
                     return candidate != nullptr &&
                            candidate->kind == ConSanProbeIntentKind::FenceRecord;
                   });
          });
      if (owning_fence != result.observation_plan.fence_site_decisions.end()) {
        site_kind = ConSanResourceSiteKind::Fence;
        disposition_offset = owning_fence->semantic_site.physical.original_text_offset;
        subject = "fence";
      }
      break;
    }
    case ConSanProbeIntentKind::Count:
      break;
    }
    const SiteKey key{site_kind, disposition_offset};
    if (patched_sites.contains(key)) {
      (void)result.coverage_ledger.set_lowering_outcome(intent.id,
                                                        ConSanLoweringOutcomeKind::Instrumented);
    } else if (failed_sites.contains(key)) {
      (void)result.coverage_ledger.set_lowering_outcome(
          intent.id, ConSanLoweringOutcomeKind::ResourceRejected,
          "MOI resource plan rejected the " + std::string(subject));
    } else {
      (void)result.coverage_ledger.set_lowering_outcome(
          intent.id, ConSanLoweringOutcomeKind::PlacementRejected,
          "MOI lowerer emitted no " + std::string(subject) + " patch");
    }
  }
}

void rebuild_moi_resource_plans(MoiResourcePlanningState &state, const ConSanOptions &options,
                                ConSanResult &result) {
  append_moi_resource_plans(state, options, result);
  append_moi_operational_resource_plans(state, options, result);
  if ((state.arch == ROCJITSU_CODE_ARCH_CDNA5 || state.arch == ROCJITSU_CODE_ARCH_CDNA3 ||
       state.arch == ROCJITSU_CODE_ARCH_CDNA4) &&
      options.automatic_moi_dispatch_id_sgprs && options.moi_dispatch_id_sgpr) {
    for (ConSanCandidateResourcePlan &plan : result.resource_plans) {
      ConSanOptions owner_options = options;
      if (!options.moi_transient_sgpr_assignments.empty() &&
          !apply_moi_transient_sgpr_assignment(owner_options, plan.owner_descriptor_file_offsets)) {
        if (state.arch == ROCJITSU_CODE_ARCH_CDNA5)
          continue;
        owner_options.moi_dispatch_id_sgpr.reset();
      }
      if (plan.owner_descriptor_file_offsets.empty())
        continue;
      // A component-local transient assignment may deliberately omit an
      // overlapping persistent dispatch pair. Its established fallback ABI
      // records generation zero; rejecting the resource plan here would make
      // that safe fallback gfx1250-only even though the report contract is
      // target-neutral.
      if (!owner_options.moi_dispatch_id_sgpr &&
          std::ranges::all_of(plan.owner_descriptor_file_offsets, [&](uint64_t descriptor_offset) {
            const ConSanMoiTransientSgprAssignment *assignment =
                find_moi_transient_sgpr_assignment(options, descriptor_offset);
            return assignment != nullptr && !assignment->dispatch_id_sgpr;
          })) {
        continue;
      }
      const auto owner_contexts =
          resolve_moi_scalar_owner_contexts(state, plan.owner_descriptor_file_offsets);
      const bool missing_dispatch = !owner_options.moi_dispatch_id_sgpr;
      const uint16_t dispatch_base = owner_options.moi_dispatch_id_sgpr.value_or(0u);
      const bool admitted = !missing_dispatch && owner_contexts &&
                            moi_scalar_contexts_admit_reserved_window(
                                owner_contexts->summaries, dispatch_base,
                                moi_dispatch_state_sgpr_count(owner_options), state.arch);
      if (!admitted) {
        plan.source = ConSanRegisterAllocationSource::Unsupported;
        plan.reason = ConSanRegisterPlanReason::ForbiddenOverlap;
        plan.scratch_vgpr.reset();
        plan.required_vgpr_count = plan.current_vgpr_count;
      }
    }
  }
  if (consan_is_capability_arch(state.arch) && options.automatic_moi_partial_exec_save_sgprs &&
      options.moi_exec_save_sgpr) {
    const uint16_t exec_base = *options.moi_exec_save_sgpr;
    const uint16_t exec_count = moi_exec_save_sgpr_count(options, state.arch);
    for (ConSanCandidateResourcePlan &plan : result.resource_plans) {
      const bool has_owner_override =
          std::ranges::any_of(plan.owner_descriptor_file_offsets, [&](uint64_t descriptor_offset) {
            return find_moi_transient_sgpr_assignment(options, descriptor_offset) != nullptr;
          });
      if (has_owner_override) {
        ConSanOptions owner_options = options;
        if (apply_moi_transient_sgpr_assignment(owner_options, plan.owner_descriptor_file_offsets))
          continue;
      }
      const auto owner_contexts =
          resolve_moi_scalar_owner_contexts(state, plan.owner_descriptor_file_offsets);
      if (!plan.owner_descriptor_file_offsets.empty() &&
          (!owner_contexts || !moi_scalar_contexts_admit_reserved_window(
                                  owner_contexts->summaries, exec_base, exec_count, state.arch))) {
        plan.source = ConSanRegisterAllocationSource::Unsupported;
        plan.reason = ConSanRegisterPlanReason::ForbiddenOverlap;
        plan.scratch_vgpr.reset();
        plan.required_vgpr_count = plan.current_vgpr_count;
      }
    }
  }
}

void summarize_moi_resource_plans(ConSanResult &result) {
  result.resource_plan_summary = {};
  for (const ConSanCandidateResourcePlan &plan : result.resource_plans) {
    for (const ConSanResourcePlanAlternative &alternative : plan.alternatives) {
      ++result.resource_plan_summary.alternative_attempts;
      switch (consan_resource_plan_alternative_outcome(plan, alternative)) {
      case ConSanResourcePlanAlternativeOutcome::Selected:
        ++result.resource_plan_summary.alternative_selected;
        break;
      case ConSanResourcePlanAlternativeOutcome::Rejected:
        ++result.resource_plan_summary.alternative_rejected;
        break;
      case ConSanResourcePlanAlternativeOutcome::Superseded:
        ++result.resource_plan_summary.alternative_superseded;
        break;
      case ConSanResourcePlanAlternativeOutcome::Contributed:
        ++result.resource_plan_summary.alternative_contributed;
        break;
      case ConSanResourcePlanAlternativeOutcome::Vetoed:
        ++result.resource_plan_summary.alternative_vetoed;
        break;
      }
    }
    switch (plan.source) {
    case ConSanRegisterAllocationSource::Explicit:
      ++result.resource_plan_summary.explicit_plans;
      break;
    case ConSanRegisterAllocationSource::LivenessDead:
      ++result.resource_plan_summary.dead_plans;
      break;
    case ConSanRegisterAllocationSource::DescriptorGrowth:
      ++result.resource_plan_summary.descriptor_growth_plans;
      break;
    case ConSanRegisterAllocationSource::SpillRequired:
      ++result.resource_plan_summary.spill_plans;
      result.resource_plan_summary.planned_spill_slot_bytes +=
          static_cast<size_t>(plan.scratch_vgpr_count) * SpillManager::kSlotBytes;
      break;
    case ConSanRegisterAllocationSource::Unsupported:
      ++result.resource_plan_summary.unsupported_plans;
      break;
    }
  }
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.spilled_vgpr_count == 0)
      continue;
    ++result.resource_plan_summary.emitted_spill_patches;
    result.resource_plan_summary.emitted_spill_slot_bytes +=
        static_cast<size_t>(patch.spilled_vgpr_count) * SpillManager::kSlotBytes;
  }
}

} // namespace
