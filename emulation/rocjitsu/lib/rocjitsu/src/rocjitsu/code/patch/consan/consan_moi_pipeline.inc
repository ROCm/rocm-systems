void try_apply_fence_record_patch(std::span<const uint8_t> bytes, const ConSanOptions &options,
                                  rj_code_arch_t arch, ConSanResult &result) {
  if (!options.moi_track_atomics || options.moi_engine != ConSanMoiEngine::RecordReplay)
    return;
  const bool has_eligible =
      std::ranges::any_of(result.moi_fence_candidates, &ConSanMoiFenceCandidate::eligible);
  if (!has_eligible)
    return;
  if (!instrumentation::is_admitted_arch(arch)) {
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
  append_fence_record_candidates(result, candidates);
  if (!options.test_kernel_name_filter.empty()) {
    std::erase_if(candidates, [&](const FenceRecordCandidate &candidate) {
      return candidate.fence.container_name.find(options.test_kernel_name_filter) ==
             std::string::npos;
    });
  }
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
    if (fence.text_offset > old_text.size() || fence.size > old_text.size() - fence.text_offset) {
      result.errors.emplace_back("ConSan MOI fence record patch site is outside .text");
      return;
    }
    const bool overlaps = std::ranges::any_of(result.patches, [&](const ConSanPatchInfo &patch) {
      const uint64_t patch_end = patch.anchor_offset + std::max<uint32_t>(patch.original_size, 1);
      return fence.text_offset < patch_end && patch.anchor_offset < fence.text_offset + fence.size;
    });
    if (overlaps) {
      result.warnings.emplace_back("ConSan MOI fence record patch skipped an overlapping patch");
      continue;
    }
    const uint16_t scratch_count = std::max<uint16_t>(
        inline_atomic_scratch_count(candidate.communication_site, arch),
        arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4 ? 8u : 0u);
    const ConSanCandidateResourcePlan *existing_plan =
        resource_plan_for_site(result, ConSanResourceSiteKind::Fence, fence.text_offset);
    ConSanCandidateResourcePlan plan =
        existing_plan
            ? *existing_plan
            : plan_moi_resource_site(resource_state, options, ConSanResourceSiteKind::Fence,
                                     candidate_index, fence.text_offset,
                                     candidate.kernel_descriptor_file_offset, scratch_count,
                                     nullptr, &candidate.communication_site);
    if (!existing_plan)
      result.resource_plans.push_back(plan);
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
      private_owner =
          resolve_moi_private_record_owner(bytes, result, active_resources, result.warnings);
      if (!private_owner)
        continue;
    }
    const uint32_t private_layout_base = private_owner
                                             ? private_owner->private_state_end
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
    selected.push_back({std::move(candidate), active_resources, address_plan, std::move(spill),
                        std::move(scalar_spill), private_owner});
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
  uint32_t record_index = 0;
  const uint32_t record_count = static_cast<uint32_t>(selected.size());
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
      // A relocatable dense host preserves at least one entry word before the
      // local indirect island. Compact eight-word islands are not dispatchers.
      if (host.kind != ConSanPatchKind::TrampolineMoiIndirectBranchIsland ||
          host.original_size < (kMoiLocalIndirectIslandWords + 1u) * sizeof(uint32_t) ||
          !owns_descriptor(host) || host_index + 1u >= result.patches.size())
        continue;
      const ConSanPatchInfo &dispatcher = result.patches[host_index + 1u];
      if (dispatcher.kind != ConSanPatchKind::TrampolineMoiIndirectBranchIsland ||
          dispatcher.original_size != 0u || !owns_descriptor(dispatcher))
        continue;
      uint64_t reservation_end = std::numeric_limits<uint64_t>::max();
      for (const ConSanPatchInfo &patch : result.patches) {
        if (patch.kind == ConSanPatchKind::TrampolineMoiInlineEpochBarrier &&
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
    if (!apply_moi_transient_sgpr_assignment(candidate_options,
                                             planned.resources.owner_descriptor_file_offsets)) {
      result.errors.emplace_back(
          "ConSan MOI fence record patch lost its transient SGPR assignment");
      return;
    }
    const bool direct_entry =
        compute_sopp_branch_simm16(fence.text_offset, cave_text_offset).has_value();
    uint64_t indirect_entry_island = 0;
    bool has_indirect_entry_island = false;
    if (!direct_entry) {
      const auto claimed_island = indirect_allocator.claim_reachable(fence.text_offset);
      if (claimed_island) {
        indirect_entry_island = *claimed_island;
        has_indirect_entry_island = true;
      }
    }
    std::vector<uint32_t> displaced_tail_words;
    uint32_t displaced_original_size = fence.size;
    bool entry_island_at_anchor = false;
    bool compact_entry_without_scc = false;
    if (!direct_entry && !has_indirect_entry_island && is_rdna4_family_arch(arch)) {
      constexpr size_t kCompactEntryIslandWords = 7u;
      BasicBlock *entry_block = block_for_offset(resource_state.block_index, fence.text_offset);
      const Instruction *entry_instruction = instruction_at(entry_block, fence.text_offset);
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
      while (displaced_words.size() < required_island_words) {
        auto decoded = decode_relocatable_entry_instruction(
            old_text, fence.text_offset + displaced_words.size() * sizeof(uint32_t), arch,
            decode_errors);
        if (!decoded) {
          displaced_words.clear();
          break;
        }
        displaced_words.insert(displaced_words.end(), decoded->begin(), decoded->end());
      }
      const size_t fence_words = fence.size / sizeof(uint32_t);
      const uint64_t displaced_end = fence.text_offset + displaced_words.size() * sizeof(uint32_t);
      const bool overlaps_existing_patch =
          std::ranges::any_of(result.patches, [&](const ConSanPatchInfo &patch) {
            const uint64_t patch_end =
                patch.anchor_offset + std::max<uint32_t>(patch.original_size, 1u);
            return fence.text_offset < patch_end && patch.anchor_offset < displaced_end;
          });
      const bool swallows_another_fence =
          std::ranges::any_of(candidates, [&](const FenceRecordCandidate &other) {
            return other.fence.text_offset != fence.text_offset &&
                   other.fence.text_offset > fence.text_offset &&
                   other.fence.text_offset < displaced_end;
          });
      if (displaced_words.size() >= required_island_words &&
          fence_words <= displaced_words.size() && !overlaps_existing_patch &&
          !swallows_another_fence &&
          displaced_words.size() * sizeof(uint32_t) <= std::numeric_limits<uint32_t>::max()) {
        displaced_tail_words.assign(displaced_words.begin() + fence_words, displaced_words.end());
        displaced_original_size = static_cast<uint32_t>(displaced_words.size() * sizeof(uint32_t));
        indirect_entry_island = fence.text_offset;
        has_indirect_entry_island = true;
        entry_island_at_anchor = true;
      }
    }
    const uint64_t return_text_offset = fence.text_offset + displaced_original_size;
    auto cave_words = build_fence_record_cave_words(
        active_bytes, planned.candidate, planned.address_plan, candidate_options,
        planned.spill ? &*planned.spill : nullptr,
        planned.scalar_spill ? &*planned.scalar_spill : nullptr, arch, record_index,
        arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4
            ? layout.fence_record_capacity
            : record_count,
        layout.fence_records_offset, cave_text_offset, return_text_offset, std::nullopt,
        displaced_tail_words,
        planned.private_owner ? std::optional<uint32_t>(planned.private_owner->offset)
                              : std::nullopt,
        planned.private_owner ? std::optional<uint16_t>(planned.private_owner->shift)
                              : std::nullopt,
        result.errors);
    if (!cave_words)
      return;
    std::optional<DenseFenceRelay> dense_relay;
    if (!direct_entry && !has_indirect_entry_island && arch == ROCJITSU_CODE_ARCH_GFX1250 &&
        planned.candidate.kernel_descriptor_file_offset && options.moi_exec_save_sgpr) {
      dense_relay = find_dense_fence_relay(*planned.candidate.kernel_descriptor_file_offset);
      if (dense_relay &&
          !compute_sopp_branch_simm16(fence.text_offset, dense_relay->island_offset)) {
        dense_relay.reset();
      }
      if (dense_relay) {
        cave_words = build_fence_record_cave_words(
            active_bytes, planned.candidate, planned.address_plan, candidate_options,
            planned.spill ? &*planned.spill : nullptr,
            planned.scalar_spill ? &*planned.scalar_spill : nullptr, arch, record_index,
            arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4
                ? layout.fence_record_capacity
                : record_count,
            layout.fence_records_offset, cave_text_offset, return_text_offset,
            static_cast<uint16_t>(*options.moi_exec_save_sgpr + 6u),
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
        fence.text_offset, displaced_original_size, cave_words->size() * sizeof(uint32_t),
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
      const auto call = instrumentation::build_s_call_i64(
          static_cast<uint16_t>(*options.moi_exec_save_sgpr + 6u), *fwd, arch);
      if (!call) {
        result.errors.emplace_back("ConSan MOI fence record patch could not encode dense call");
        return;
      }
      anchor_words.front() = *call;
    } else {
      anchor_words.front() = build_s_branch(*fwd, arch);
    }
    std::memcpy(new_text.data() + fence.text_offset, anchor_words.data(),
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
          const bool delta_encoded =
              append_pc_delta_builder(island_words, arch, jump_sgprs->pc_sgpr, delta);
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
        island_info.anchor_offset = fence.text_offset;
        island_info.trampoline_offset = indirect_entry_island;
        // The fence patch owns the entire decoded prefix. A multiword final
        // guest instruction can make that prefix one word larger than the
        // compact island; its trailing NOP is part of the same exact overlap.
        island_info.trampoline_size = displaced_original_size;
        island_info.owner_descriptor_file_offsets = planned.resources.owner_descriptor_file_offsets;
        patches.push_back(std::move(island_info));
      } else if (!emit_moi_local_indirect_entry_island(
                     new_text, indirect_entry_island, placement->body_offset, fence.text_offset,
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
      const uint16_t base = *options.moi_exec_save_sgpr;
      const uint16_t key_sgpr = static_cast<uint16_t>(base + 5u);
      const auto compare = instrumentation::build_s_cmp_eq_u32(key_sgpr, /*literal=*/255u, arch);
      const auto special_state = moi_special_state_sgprs(options);
      if (!compare || !special_state) {
        result.errors.emplace_back(
            "ConSan MOI fence record patch could not extend dense dispatcher");
        return;
      }
      dispatcher_words.push_back(*compare);
      dispatcher_words.push_back(
          static_cast<uint32_t>(fence.text_offset + sizeof(uint32_t) - dense_relay->island_offset));
      std::vector<uint32_t> matched_words;
      const uint64_t matched_offset =
          dispatcher.trampoline_offset + (dispatcher_words.size() + 1u) * sizeof(uint32_t);
      if (!append_moi_scc_preserving_indirect_jump(
              matched_words, matched_offset, placement->body_offset, base,
              special_state->scc_save_sgpr, /*capture_scc=*/false, arch)) {
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
    info.anchor_offset = fence.text_offset;
    info.trampoline_offset = cave_text_offset;
    info.original_size = displaced_original_size;
    info.trampoline_size = static_cast<uint32_t>(cave_words->size() * sizeof(uint32_t));
    info.scratch_vgpr = planned.resources.base;
    info.owner_descriptor_file_offsets = planned.resources.owner_descriptor_file_offsets;
    if (planned.private_owner) {
      info.persistent_owner_private_offset = planned.private_owner->offset;
      info.persistent_private_state_end = planned.private_owner->private_state_end;
    }
    if (planned.spill) {
      info.spilled_vgpr_count = planned.spill->vgpr_count;
      info.required_private_segment_size = planned.spill->total_private_bytes;
    }
    if (planned.scalar_spill) {
      info.required_private_segment_size =
          std::max(info.required_private_segment_size, planned.scalar_spill->total_private_bytes);
    }
    patches.push_back(std::move(info));
    ++record_index;
  }
  if (!patcher.replace_text(new_text, kConSanMaxPatchedFileGrowth)) {
    result.errors.emplace_back("ConSan MOI fence record patch could not grow .text");
    return;
  }
  result.elf_bytes = patcher.emit();
  result.patches.insert(result.patches.end(), patches.begin(), patches.end());
  result.modified = true;
}

void append_moi_operational_resource_plans(MoiResourcePlanningState &state,
                                           const ConSanOptions &options, ConSanResult &result) {
  if (options.moi_track_barriers) {
    std::vector<BarrierRecordCandidate> candidates;
    for (const ConSanKernelInfo &kernel : result.kernels)
      append_barrier_epoch_candidates(kernel, candidates);
    for (const ConSanFunctionInfo &function : result.functions)
      append_barrier_epoch_candidates(function, candidates);
    apply_test_container_filter(candidates, options);
    const bool uses_coalesced_epoch =
        options.moi_engine == ConSanMoiEngine::InlineShadow ||
        (options.moi_engine == ConSanMoiEngine::RecordReplay && options.moi_init_owner_epoch &&
         (options.moi_epoch_vgpr || options.moi_persistent_epoch_sgpr ||
          options.automatic_moi_private_epoch));
    if (uses_coalesced_epoch) {
      std::erase_if(candidates, [](const BarrierRecordCandidate &candidate) {
        return candidate.site.operation == ConSanBarrierSite::Operation::Signal;
      });
    }
    for (size_t index = 0; index < candidates.size(); ++index) {
      const BarrierRecordCandidate &candidate = candidates[index];
      result.resource_plans.push_back(plan_moi_resource_site(
          state, options, ConSanResourceSiteKind::Barrier, index, candidate.site.text_offset,
          candidate.kernel_descriptor_file_offset,
          /*scratch_count=*/options.moi_engine == ConSanMoiEngine::Sampled
              ? (options.automatic_moi_private_epoch || moi_uses_scalar_persistent_state(options)
                     ? 9u
                     : 7u)
          : options.moi_engine == ConSanMoiEngine::InlineShadow
              ? inline_shadow_barrier_scratch_count(options)
          : uses_coalesced_epoch ? 1u
                                 : 6u));
    }
  }
  if (options.moi_track_atomics) {
    std::vector<AtomicRecordCandidate> candidates;
    if (options.moi_engine == ConSanMoiEngine::InlineShadow) {
      for (const ConSanKernelInfo &kernel : result.kernels)
        append_inline_atomic_candidates(kernel, result, candidates);
      for (const ConSanFunctionInfo &function : result.functions)
        append_inline_atomic_candidates(function, result, candidates);
      candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                                      [](const auto &candidate) {
                                        return classify_consan_moi_inline_atomic_support(
                                                   candidate.site, candidate.event_kind) !=
                                               ConSanMoiInlineAtomicSupport::Supported;
                                      }),
                       candidates.end());
    } else {
      for (const ConSanKernelInfo &kernel : result.kernels)
        append_atomic_record_candidates(kernel, result, candidates);
      for (const ConSanFunctionInfo &function : result.functions)
        append_atomic_record_candidates(function, result, candidates);
    }
    apply_test_container_filter(candidates, options);
    for (size_t index = 0; index < candidates.size(); ++index) {
      const AtomicRecordCandidate &candidate = candidates[index];
      const uint16_t scratch_count =
          options.moi_engine == ConSanMoiEngine::Sampled ? sampled_atomic_scratch_count()
          : options.moi_engine == ConSanMoiEngine::InlineShadow
              ? inline_shadow_atomic_scratch_count(candidate.site, candidate.event_kind, state.arch)
              : atomic_record_scratch_count(candidate.site, state.arch);
      result.resource_plans.push_back(plan_moi_resource_site(
          state, options, ConSanResourceSiteKind::Atomic, index, candidate.site.text_offset,
          candidate.kernel_descriptor_file_offset, scratch_count, nullptr, &candidate.site));
    }
  }
  if (options.moi_track_atomics && options.moi_engine == ConSanMoiEngine::RecordReplay) {
    std::vector<FenceRecordCandidate> candidates;
    append_fence_record_candidates(result, candidates);
    if (!options.test_kernel_name_filter.empty()) {
      std::erase_if(candidates, [&](const FenceRecordCandidate &candidate) {
        return candidate.fence.container_name.find(options.test_kernel_name_filter) ==
               std::string::npos;
      });
    }
    for (size_t index = 0; index < candidates.size(); ++index) {
      const FenceRecordCandidate &candidate = candidates[index];
      result.resource_plans.push_back(plan_moi_resource_site(
          state, options, ConSanResourceSiteKind::Fence, index, candidate.fence.text_offset,
          candidate.kernel_descriptor_file_offset,
          std::max<uint16_t>(inline_atomic_scratch_count(candidate.communication_site, state.arch),
                             state.arch == ROCJITSU_CODE_ARCH_CDNA3 ||
                                     state.arch == ROCJITSU_CODE_ARCH_CDNA4
                                 ? 8u
                                 : 0u),
          nullptr, &candidate.communication_site));
    }
  }
}

void append_moi_site_disposition(ConSanResourceSiteKind kind, std::string_view container_name,
                                 bool in_kernel, uint64_t text_offset, std::string_view mnemonic,
                                 ConSanSiteDispositionReason reason, ConSanResult &result,
                                 bool semantic_not_applicable = false) {
  ConSanSiteDispositionRecord record;
  record.site_kind = kind;
  record.disposition = reason == ConSanSiteDispositionReason::None
                           ? ConSanSiteDisposition::Supported
                       : semantic_not_applicable ? ConSanSiteDisposition::NotApplicable
                                                 : ConSanSiteDisposition::Unsupported;
  record.reason = reason;
  if (record.disposition == ConSanSiteDisposition::Supported) {
    record.lowering_outcome = ConSanSiteLoweringOutcome::Pending;
    record.lowering_reason = ConSanSiteLoweringReason::None;
  } else if (record.disposition == ConSanSiteDisposition::NotApplicable) {
    record.lowering_outcome = ConSanSiteLoweringOutcome::NotApplicable;
    record.lowering_reason = ConSanSiteLoweringReason::SemanticNotApplicable;
  } else {
    record.lowering_outcome = ConSanSiteLoweringOutcome::Unsupported;
    record.lowering_reason = ConSanSiteLoweringReason::SemanticUnsupported;
  }
  record.container_name = container_name;
  record.in_kernel = in_kernel;
  record.text_offset = text_offset;
  record.mnemonic = mnemonic;
  result.site_dispositions.push_back(std::move(record));
}

[[nodiscard]] ConSanSiteDispositionReason
inline_atomic_disposition_reason(ConSanMoiInlineAtomicSupport support) {
  switch (support) {
  case ConSanMoiInlineAtomicSupport::Supported:
    return ConSanSiteDispositionReason::None;
  case ConSanMoiInlineAtomicSupport::NonFlatAddress:
    return ConSanSiteDispositionReason::NonFlatAtomicAddress;
  case ConSanMoiInlineAtomicSupport::NonzeroOffset:
    return ConSanSiteDispositionReason::NonzeroFlatOffset;
  case ConSanMoiInlineAtomicSupport::UnsupportedScope:
    return ConSanSiteDispositionReason::UnsupportedScope;
  case ConSanMoiInlineAtomicSupport::UnsupportedWidth:
    return ConSanSiteDispositionReason::InvalidAccessWidth;
  case ConSanMoiInlineAtomicSupport::CompareExchangeOutcomeUnavailable:
    return ConSanSiteDispositionReason::CompareExchangeOutcomeUnavailable;
  case ConSanMoiInlineAtomicSupport::UnsupportedEncoding:
    return ConSanSiteDispositionReason::UnsupportedAtomicEncoding;
  case ConSanMoiInlineAtomicSupport::MissingOperands:
    return ConSanSiteDispositionReason::MissingAtomicOperands;
  case ConSanMoiInlineAtomicSupport::MissingOrderingMetadata:
    return ConSanSiteDispositionReason::MissingOrderingMetadata;
  }
  return ConSanSiteDispositionReason::UnsupportedAtomicEncoding;
}

[[nodiscard]] ConSanSiteDispositionReason
sampled_atomic_disposition_reason(std::string_view reason) {
  if (reason == "missing-shared-sync-event" || reason == "unqualified-shared-sync-sequence")
    return ConSanSiteDispositionReason::UnqualifiedSyncSequence;
  if (reason == "unsupported-qualified-memory-role")
    return ConSanSiteDispositionReason::UnsupportedMemoryRole;
  if (reason == "missing-qualified-scope")
    return ConSanSiteDispositionReason::MissingScope;
  if (reason == "unsupported-qualified-scope")
    return ConSanSiteDispositionReason::UnsupportedScope;
  if (reason == "unsupported-qualified-byte-range")
    return ConSanSiteDispositionReason::InvalidAccessWidth;
  if (reason == "compare-exchange-dynamic-outcome-unavailable")
    return ConSanSiteDispositionReason::CompareExchangeOutcomeUnavailable;
  if (reason == "unsupported-qualified-rmw-outcome")
    return ConSanSiteDispositionReason::UnsupportedAtomicOutcome;
  return ConSanSiteDispositionReason::UnsupportedMetadataEncoding;
}

[[nodiscard]] ConSanSiteDispositionReason
first_light_atomic_disposition_reason(const ConSanAtomicSite &site, bool allow_gfx1250_lds) {
  if (!site.mnemonic.starts_with("flat_atomic") && !site.mnemonic.starts_with("global_atomic") &&
      !(allow_gfx1250_lds && site.mnemonic.starts_with("ds_")))
    return ConSanSiteDispositionReason::NonFlatAtomicAddress;
  if (!site.addr_vgpr || !site.data_vgpr)
    return ConSanSiteDispositionReason::MissingAtomicOperands;
  if (*site.addr_vgpr >= 255)
    return ConSanSiteDispositionReason::ReservedFlatAddressRegister;
  if (!site.raw_scope)
    return ConSanSiteDispositionReason::MissingOrderingMetadata;
  if (*site.raw_scope < 1u || *site.raw_scope > 3u)
    return ConSanSiteDispositionReason::UnsupportedScope;
  const bool compare_exchange = consan_atomic_is_compare_exchange(site);
  if (compare_exchange && !site.returns_old_value.value_or(false))
    return ConSanSiteDispositionReason::CompareExchangeOutcomeUnavailable;
  return ConSanSiteDispositionReason::None;
}

template <typename Container>
void append_moi_atomic_site_dispositions(const Container &container, const ConSanOptions &options,
                                         const ConSanResult &semantic_result,
                                         ConSanResult &result) {
  if (!options.moi_track_atomics ||
      (!options.test_kernel_name_filter.empty() &&
       container.name.find(options.test_kernel_name_filter) == std::string::npos))
    return;
  const bool in_kernel = std::is_same_v<Container, ConSanKernelInfo>;
  const bool allow_gfx1250_lds = semantic_result.arch_name == "gfx1250";
  if (in_kernel && is_rocclr_runtime_kernel_name(container.name))
    return;
  for (const ConSanAtomicSite &site : container.atomic_sites) {
    const auto event =
        std::ranges::find_if(semantic_result.sync_events, [&](const ConSanSyncEvent &candidate) {
          return candidate.kind == ConSanSyncEventKind::Atomic &&
                 candidate.container_name == container.name && candidate.in_kernel == in_kernel &&
                 candidate.text_offset == site.text_offset;
        });
    const std::optional<ConSanMoiAtomicEventKind> event_kind =
        atomic_event_kind_for_site(semantic_result, container.name, in_kernel, site);
    ConSanSiteDispositionReason reason = ConSanSiteDispositionReason::None;
    if (event == semantic_result.sync_events.end() || event->execution_owners.empty()) {
      // Runtime-library helpers that no inventoried kernel can execute are
      // decoded inventory, but not relevant workload support sites. Keep the
      // operational candidate path stable until report sizing and persistent
      // state selection no longer depend on it; the coverage ledger must not
      // call an impossible MissingOwner plan supported.
      reason = ConSanSiteDispositionReason::UnqualifiedSyncSequence;
    } else if (!event_kind) {
      reason = ConSanSiteDispositionReason::UnqualifiedSyncSequence;
    } else if (options.moi_engine == ConSanMoiEngine::InlineShadow) {
      reason = inline_atomic_disposition_reason(
          classify_consan_moi_inline_atomic_support(site, *event_kind));
    } else {
      const ConSanAtomicSite normalized = normalize_atomic_record_site(site, allow_gfx1250_lds);
      reason = first_light_atomic_disposition_reason(normalized, allow_gfx1250_lds);
      if (reason == ConSanSiteDispositionReason::None &&
          options.moi_engine == ConSanMoiEngine::Sampled) {
        std::string sampled_reason;
        if (!sampled_atomic_semantics_for_site(semantic_result, container.name, in_kernel, site,
                                               sampled_reason))
          reason = sampled_atomic_disposition_reason(sampled_reason);
      }
    }
    append_moi_site_disposition(ConSanResourceSiteKind::Atomic, container.name, in_kernel,
                                site.text_offset, site.mnemonic, reason, result,
                                reason == ConSanSiteDispositionReason::UnqualifiedSyncSequence ||
                                    (reason == ConSanSiteDispositionReason::UnsupportedScope &&
                                     site.raw_scope && *site.raw_scope == 0u));
  }
}

template <typename Container>
void append_moi_barrier_site_dispositions(const Container &container, const ConSanOptions &options,
                                          const ConSanResult &semantic_result,
                                          ConSanResult &result) {
  if (!options.moi_track_barriers ||
      (!options.test_kernel_name_filter.empty() &&
       container.name.find(options.test_kernel_name_filter) == std::string::npos))
    return;
  const bool in_kernel = std::is_same_v<Container, ConSanKernelInfo>;
  if (in_kernel && is_rocclr_runtime_kernel_name(container.name))
    return;
  const bool uses_coalesced_epoch =
      options.moi_engine == ConSanMoiEngine::InlineShadow ||
      (options.moi_engine == ConSanMoiEngine::RecordReplay && options.moi_init_owner_epoch &&
       (options.moi_epoch_vgpr || options.moi_persistent_epoch_sgpr ||
        options.automatic_moi_private_epoch));
  for (const ConSanBarrierSite &site : container.barrier_sites) {
    if (uses_coalesced_epoch && site.operation == ConSanBarrierSite::Operation::Signal)
      continue;
    if (options.moi_engine == ConSanMoiEngine::InlineShadow && !in_kernel &&
        (result.moi_persistent_vgprs_automatic || result.moi_private_epoch_automatic)) {
      const ConSanCandidateResourcePlan *plan =
          resource_plan_for_site(result, ConSanResourceSiteKind::Barrier, site.text_offset);
      if (plan == nullptr || plan->owner_descriptor_file_offsets.empty())
        continue;
    }
    ConSanSiteDispositionReason reason = ConSanSiteDispositionReason::None;
    if (options.moi_engine == ConSanMoiEngine::Sampled) {
      const ConSanSyncEvent *event = nullptr;
      size_t event_count = 0;
      for (const ConSanSyncEvent &candidate : semantic_result.sync_events) {
        if (candidate.kind != ConSanSyncEventKind::Barrier ||
            candidate.container_name != container.name || candidate.in_kernel != in_kernel ||
            candidate.text_offset != site.text_offset || candidate.size != site.size)
          continue;
        ++event_count;
        event = &candidate;
      }
      const bool qualified =
          event_count == 1u &&
          std::ranges::any_of(semantic_result.sync_sequences, [&](const auto &sequence) {
            return consan_moi_sampled_qualifies_barrier_sequence(sequence) &&
                   std::ranges::find(sequence.member_event_identities, event->identity) !=
                       sequence.member_event_identities.end();
          });
      if (!qualified)
        reason = ConSanSiteDispositionReason::UnqualifiedSyncSequence;
    } else if (site.size != sizeof(uint32_t)) {
      reason = ConSanSiteDispositionReason::InvalidBarrierEncoding;
    }
    append_moi_site_disposition(ConSanResourceSiteKind::Barrier, container.name, in_kernel,
                                site.text_offset, site.mnemonic, reason, result);
  }
}

void append_moi_sync_site_dispositions(const ConSanOptions &options, ConSanResult &result) {
  // Keep the semantic inventory immutable while appending to the disposition
  // vector that is part of the same result.
  const ConSanResult &semantic_result = result;
  for (const ConSanKernelInfo &kernel : result.kernels) {
    append_moi_atomic_site_dispositions(kernel, options, semantic_result, result);
    append_moi_barrier_site_dispositions(kernel, options, semantic_result, result);
  }
  for (const ConSanFunctionInfo &function : result.functions) {
    append_moi_atomic_site_dispositions(function, options, semantic_result, result);
    append_moi_barrier_site_dispositions(function, options, semantic_result, result);
  }

  if (!options.moi_track_atomics || options.moi_engine != ConSanMoiEngine::RecordReplay)
    return;
  std::vector<FenceRecordCandidate> admitted;
  append_fence_record_candidates(result, admitted);
  for (const ConSanMoiFenceCandidate &fence : result.moi_fence_candidates) {
    if ((!options.test_kernel_name_filter.empty() &&
         fence.container_name.find(options.test_kernel_name_filter) == std::string::npos) ||
        (fence.in_kernel && is_rocclr_runtime_kernel_name(fence.container_name)))
      continue;
    ConSanSiteDispositionReason reason = ConSanSiteDispositionReason::None;
    const auto communication = std::ranges::find(
        result.sync_events, fence.communication_event_identity, &ConSanSyncEvent::identity);
    const bool unowned_communication =
        communication != result.sync_events.end() && communication->execution_owners.empty();
    const auto fence_event = std::ranges::find(result.sync_events, fence.fence_event_identity,
                                               &ConSanSyncEvent::identity);
    const bool unowned_fence =
        fence_event != result.sync_events.end() && fence_event->execution_owners.empty();
    const bool unassociated_cache_operation =
        fence.rejection_reason == "fence-not-associated-with-addressed-communication-sequence";
    if (!fence.eligible) {
      reason = ConSanSiteDispositionReason::IneligibleFence;
    } else if (communication == result.sync_events.end() || unowned_communication) {
      reason = ConSanSiteDispositionReason::MissingCommunicationEvent;
    } else if (std::ranges::none_of(admitted, [&](const FenceRecordCandidate &candidate) {
                 return candidate.fence.identity == fence.identity;
               })) {
      reason = ConSanSiteDispositionReason::MissingCommunicationEvent;
    }
    append_moi_site_disposition(ConSanResourceSiteKind::Fence, fence.container_name,
                                fence.in_kernel, fence.text_offset, "fence", reason, result,
                                unowned_communication || unowned_fence ||
                                    unassociated_cache_operation ||
                                    reason == ConSanSiteDispositionReason::IneligibleFence);
  }
}

[[nodiscard]] std::optional<ConSanResourceSiteKind>
moi_instrumentation_patch_site_kind(const ConSanPatchInfo &patch, const ConSanResult &result) {
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
    for (const ConSanCandidateResourcePlan &plan : result.resource_plans) {
      if (plan.text_offset == patch.anchor_offset &&
          (plan.site_kind == ConSanResourceSiteKind::Barrier ||
           plan.site_kind == ConSanResourceSiteKind::Atomic))
        return plan.site_kind;
    }
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
  std::unordered_set<SiteKey, SiteKeyHash> failed_sites;
  for (const ConSanCandidateResourcePlan &plan : result.resource_plans) {
    resource_kind_by_offset.try_emplace(plan.text_offset, plan.site_kind);
    if (plan.source == ConSanRegisterAllocationSource::Unsupported)
      failed_sites.insert({plan.site_kind, plan.text_offset});
  }

  std::unordered_map<std::string_view, const ConSanSyncEvent *> event_by_identity;
  event_by_identity.reserve(result.sync_events.size());
  for (const ConSanSyncEvent &event : result.sync_events)
    event_by_identity.try_emplace(event.identity, &event);
  std::unordered_map<std::string_view, const ConSanSyncSequence *> sequence_by_member;
  for (const ConSanSyncSequence &sequence : result.sync_sequences) {
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
      const auto patch_kind = moi_instrumentation_patch_site_kind(patch, result);
      if (patch_kind)
        patched_sites.insert({*patch_kind, patch.anchor_offset});
      continue;
    }

    const auto patch_kind = moi_instrumentation_patch_site_kind(patch, result);
    if (patch_kind)
      patched_sites.insert({*patch_kind, patch.anchor_offset});
    if (patch.covered_sync_event_count == 0)
      continue;
    const auto anchor_event = std::ranges::find_if(result.sync_events, [&](const auto &event) {
      return event.kind == ConSanSyncEventKind::Barrier && event.text_offset == patch.anchor_offset;
    });
    if (anchor_event == result.sync_events.end())
      continue;
    const auto sequence = sequence_by_member.find(anchor_event->identity);
    if (sequence == sequence_by_member.end() || sequence->second == nullptr ||
        sequence->second->member_event_identities.size() != patch.covered_sync_event_count)
      continue;
    for (const std::string &identity : sequence->second->member_event_identities) {
      const auto event = event_by_identity.find(identity);
      if (event != event_by_identity.end() && event->second->kind == ConSanSyncEventKind::Barrier)
        patched_sites.insert({ConSanResourceSiteKind::Barrier, event->second->text_offset});
    }
  }

  for (ConSanSiteDispositionRecord &site : result.site_dispositions) {
    if (site.disposition == ConSanSiteDisposition::NotApplicable) {
      site.lowering_outcome = ConSanSiteLoweringOutcome::NotApplicable;
      site.lowering_reason = ConSanSiteLoweringReason::SemanticNotApplicable;
      continue;
    }
    if (site.disposition == ConSanSiteDisposition::Unsupported) {
      site.lowering_outcome = ConSanSiteLoweringOutcome::Unsupported;
      site.lowering_reason = ConSanSiteLoweringReason::SemanticUnsupported;
      continue;
    }

    const SiteKey key{site.site_kind, site.text_offset};
    if (patched_sites.contains(key)) {
      site.lowering_outcome = ConSanSiteLoweringOutcome::Patched;
      site.lowering_reason = ConSanSiteLoweringReason::None;
      site.resource_reason = ConSanRegisterPlanReason::None;
      continue;
    }

    if (failed_sites.contains(key)) {
      site.lowering_outcome = ConSanSiteLoweringOutcome::ResourceFailed;
      site.lowering_reason = ConSanSiteLoweringReason::UnsupportedResourcePlan;
      const auto failed_plan =
          std::ranges::find_if(result.resource_plans, [&](const ConSanCandidateResourcePlan &plan) {
            return plan.site_kind == site.site_kind && plan.text_offset == site.text_offset &&
                   plan.source == ConSanRegisterAllocationSource::Unsupported;
          });
      site.resource_reason = failed_plan == result.resource_plans.end()
                                 ? ConSanRegisterPlanReason::None
                                 : failed_plan->reason;
      continue;
    }

    site.lowering_outcome = ConSanSiteLoweringOutcome::PlacementOrLoweringFailed;
    site.lowering_reason = ConSanSiteLoweringReason::InstrumentationPatchMissing;
    site.resource_reason = ConSanRegisterPlanReason::None;
  }
}

void rebuild_moi_resource_plans(MoiResourcePlanningState &state, const ConSanOptions &options,
                                ConSanResult &result) {
  append_moi_resource_plans(state, options, result);
  append_moi_operational_resource_plans(state, options, result);
  if ((state.arch == ROCJITSU_CODE_ARCH_GFX1250 || state.arch == ROCJITSU_CODE_ARCH_CDNA3 ||
       state.arch == ROCJITSU_CODE_ARCH_CDNA4) &&
      options.automatic_moi_dispatch_id_sgprs && options.moi_dispatch_id_sgpr) {
    for (ConSanCandidateResourcePlan &plan : result.resource_plans) {
      ConSanOptions owner_options = options;
      if (!options.moi_transient_sgpr_assignments.empty() &&
          !apply_moi_transient_sgpr_assignment(owner_options, plan.owner_descriptor_file_offsets)) {
        if (state.arch == ROCJITSU_CODE_ARCH_GFX1250)
          continue;
        owner_options.moi_dispatch_id_sgpr.reset();
      }
      if (plan.owner_descriptor_file_offsets.empty())
        continue;
      if (state.arch == ROCJITSU_CODE_ARCH_GFX1250 && !owner_options.moi_dispatch_id_sgpr)
        continue;
      const bool missing_dispatch = !owner_options.moi_dispatch_id_sgpr;
      const uint16_t dispatch_base = owner_options.moi_dispatch_id_sgpr.value_or(0u);
      const bool conflicts_with_cdna_vcc =
          !missing_dispatch && instrumentation::is_cdna_family_arch(state.arch) &&
          cdna_sgpr_window_conflicts_with_physical_vcc(
              dispatch_base, moi_dispatch_state_sgpr_count(owner_options), plan.current_sgpr_count);
      // max_referenced_sgpr_count is a count: equality means every guest
      // reference is below dispatch_base, while a greater count reaches or
      // crosses persistent dispatch state and cannot share this global pair.
      if (missing_dispatch || plan.max_referenced_sgpr_count > dispatch_base ||
          conflicts_with_cdna_vcc) {
        plan.source = ConSanRegisterAllocationSource::Unsupported;
        plan.reason = ConSanRegisterPlanReason::ForbiddenOverlap;
        plan.scratch_vgpr.reset();
        plan.required_vgpr_count = plan.current_vgpr_count;
      }
    }
  }
  if ((is_rdna4_family_arch(state.arch) || state.arch == ROCJITSU_CODE_ARCH_CDNA3 ||
       state.arch == ROCJITSU_CODE_ARCH_CDNA4) &&
      options.automatic_moi_partial_exec_save_sgprs && options.moi_exec_save_sgpr) {
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
      const bool conflicts_with_cdna_vcc = instrumentation::is_cdna_family_arch(state.arch) &&
                                           cdna_sgpr_window_conflicts_with_physical_vcc(
                                               exec_base, exec_count, plan.current_sgpr_count);
      if (!plan.owner_descriptor_file_offsets.empty() &&
          (plan.max_referenced_sgpr_count > exec_base || conflicts_with_cdna_vcc)) {
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
