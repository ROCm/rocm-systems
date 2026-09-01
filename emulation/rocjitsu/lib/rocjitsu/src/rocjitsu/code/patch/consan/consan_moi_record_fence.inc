void try_apply_fence_record_patch(std::span<const uint8_t> bytes, const ConSanOptions &options,
                                  const ConSanMoiOperatingPoint &operating_point,
                                  const MoiObjectModeSemantics &mode_semantics, rj_code_arch_t arch,
                                  ConSanTransformArtifacts &result) {
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
  const ConSanMoiReportBufferLayout layout = resolve_moi_report_layout(options, options);
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

  const std::span<const uint8_t> active_bytes = active_moi_bytes(bytes, result);
  AmdGpuCodeObject code_object(active_bytes.data(), active_bytes.size());
  CodeObjectPatcher patcher(code_object);
  const std::span<const uint8_t> old_text = patcher.text_bytes();
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
    const bool overlaps =
        std::ranges::any_of(result.patches, [&](const ConSanPatchLoweringProduct &patch) {
          const uint64_t patch_end =
              patch.anchor_offset + std::max<uint32_t>(patch.original_size, 1);
          return candidate.patch_text_offset < patch_end &&
                 patch.anchor_offset < candidate.patch_text_offset + candidate.patch_size;
        });
    if (overlaps) {
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
        const ConSanKernelInfo *kernel =
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
  if (!apply_moi_descriptor_requirements(patcher, code_object, result, descriptor_requirements,
                                         scalar_requirements, private_requirements, nullptr,
                                         nullptr, arch, "ConSan MOI fence record patch",
                                         result.errors))
    return;

  std::vector<uint8_t> new_text(old_text.begin(), old_text.end());
  DbiPatchPlacementPlanner placement_planner(arch, old_text.size());
  MoiLocalNopIslandAllocator indirect_allocator(old_text, result.program_inventory, result.patches,
                                                arch, kMoiLocalIndirectIslandWords);
  std::vector<ConSanPatchInfo> patches;
  std::vector<ConSanCommittedLowering> lowering_commits;
  for (const PlannedFenceRecord &planned : selected) {
    MoiAtomicEvidenceSitePlan scalar_clause_candidate;
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
    const auto owns_descriptor = [&](const ConSanPatchLoweringProduct &patch) {
      return std::ranges::find(patch.owner_descriptor_file_offsets, descriptor_offset) !=
             patch.owner_descriptor_file_offsets.end();
    };
    for (size_t host_index = result.patches.size(); host_index-- > 0u;) {
      const ConSanPatchLoweringProduct &host = result.patches[host_index];
      // A relocatable dense host records the complete local indirect island;
      // compact islands without displaced host ownership remain smaller.
      if (host.kind != ConSanPatchKind::TrampolineMoiIndirectBranchIsland ||
          host.original_size < kMoiLocalIndirectIslandWords * sizeof(uint32_t) ||
          !owns_descriptor(host) || host_index + 1u >= result.patches.size())
        continue;
      const ConSanPatchLoweringProduct &dispatcher = result.patches[host_index + 1u];
      if (dispatcher.kind != ConSanPatchKind::TrampolineMoiIndirectBranchIsland ||
          dispatcher.original_size != 0u || !owns_descriptor(dispatcher))
        continue;
      uint64_t reservation_end = std::numeric_limits<uint64_t>::max();
      for (const ConSanPatchLoweringProduct &patch : result.patches) {
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
    const uint64_t cave_text_offset = placement_planner.appended_end();
    const MoiRecordEventEmissionPlan &candidate_options = planned.emission;
    const std::optional<uint16_t> dense_call_return_sgpr =
        candidate_options.automatic_moi_record_replay_sgpr_spill
            ? candidate_options.router_call_return_sgpr
            : (candidate_options.moi_exec_save_sgpr
                   ? std::optional<uint16_t>(
                         static_cast<uint16_t>(*candidate_options.moi_exec_save_sgpr + 6u))
                   : std::nullopt);
    const std::optional<uint16_t> dense_dispatch_key_sgpr =
        candidate_options.automatic_moi_record_replay_sgpr_spill
            ? candidate_options.router_dispatch_key_sgpr
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
    if (!direct_entry && !has_indirect_entry_island && consan_uses_gfx12_encoding(arch)) {
      constexpr size_t kCompactEntryIslandWords = 7u;
      compact_entry_without_scc =
          moi_resource_scc_is_dead_before(*resource_state, patch_text_offset);
      const size_t required_island_words =
          compact_entry_without_scc ? kCompactEntryIslandWords : kMoiLocalIndirectIslandWords;
      std::vector<uint32_t> displaced_words;
      std::vector<std::string> decode_errors;
      const size_t patch_words = patch_size / sizeof(uint32_t);
      while (displaced_words.size() < std::max(required_island_words, patch_words)) {
        auto decoded = decode_relocatable_entry_instruction(
            old_text, patch_text_offset + displaced_words.size() * sizeof(uint32_t), arch,
            decode_errors, nullptr);
        if (!decoded) {
          displaced_words.clear();
          break;
        }
        displaced_words.insert(displaced_words.end(), decoded->begin(), decoded->end());
      }
      const uint64_t displaced_end = patch_text_offset + displaced_words.size() * sizeof(uint32_t);
      const bool overlaps_existing_patch =
          std::ranges::any_of(result.patches, [&](const ConSanPatchLoweringProduct &patch) {
            const uint64_t patch_end =
                patch.anchor_offset + std::max<uint32_t>(patch.original_size, 1u);
            return patch_text_offset < patch_end && patch.anchor_offset < displaced_end;
          });
      const bool swallows_another_fence =
          std::ranges::any_of(candidates, [&](const MoiFenceEvidenceSitePlan &other) {
            return other.patch_text_offset != patch_text_offset &&
                   other.patch_text_offset > patch_text_offset &&
                   other.patch_text_offset < displaced_end;
          });
      if (displaced_words.size() >= required_island_words &&
          patch_words <= displaced_words.size() && !overlaps_existing_patch &&
          !swallows_another_fence &&
          moi_resource_entry_window_is_single_entry(*resource_state, patch_text_offset,
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
        return_text_offset, std::nullopt, displaced_tail_words, planned.derived_owner,
        result.errors);
    if (!cave_words)
      return;
    std::optional<DenseFenceRelay> dense_relay;
    if (!direct_entry && !has_indirect_entry_island && consan_arch_has_s_call_i64(arch) &&
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
            /*displaced_tail_words=*/{}, planned.derived_owner, result.errors);
        if (!cave_words)
          return;
      }
    }
    if (!direct_entry && !has_indirect_entry_island && !dense_relay) {
      result.warnings.emplace_back(
          "ConSan MOI fence record patch found no reachable indirect entry island in " +
          planned.candidate.container_name);
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
              /*prefer_literal64=*/consan_arch_has_s_call_i64(arch));
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
      ConSanPatchLoweringProduct &dispatcher = result.patches[dense_relay->dispatcher_patch_index];
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
    note_moi_probe_patch_info(info, planned);
    if (!append_moi_fence_lowering_commit(result, planned.candidate, info, "fence record patch",
                                          lowering_commits))
      return;
    patches.push_back(std::move(info));
    ++record_index;
  }
  (void)publish_moi_sync_patch(result, patcher, new_text, options.patched_image_growth_limit,
                               "MOI fence record patch", "fence record patch",
                               std::move(lowering_commits), std::move(patches));
}
