/// Record/Replay-owned barrier planning, direct/branch-only routing, and
/// fallback coordination over the shared exact-epoch/record barrier mechanism.


// Access planning may materialize a conservative direct-reservoir frontier
// for the later Record/Replay barrier pass. Once barrier routes have been
// emitted, restore every unclaimed reservoir and its recursive route while
// retaining its appended slot as NOP padding. Repeating to a fixed point
// removes dependency-only reservoirs whose sole consumer was itself pruned.
// This preserves stable planned offsets without leaving dead guest-code
// relocations in the final image.
[[nodiscard]] bool
prune_unused_record_replay_direct_reservoirs(std::vector<uint8_t> &text, rj_code_arch_t arch,
                                             std::vector<ConSanPatchInfo> &patches,
                                             std::vector<std::string> &errors) {
  const uint32_t nop = build_s_nop(0u, arch);
  const auto is_direct_reservoir = [](const ConSanPatchLoweringProduct &patch) {
    return patch.kind == ConSanPatchKind::TrampolineBranchRelayReservoir &&
           patch.original_size >= 2u * sizeof(uint32_t) &&
           patch.trampoline_size == patch.original_size + sizeof(uint32_t);
  };
  bool pruned_any = false;
  for (;;) {
    const auto unused = std::ranges::find_if(patches, [&](const ConSanPatchLoweringProduct &patch) {
      if (!is_direct_reservoir(patch) || patch.anchor_offset > text.size() ||
          patch.original_size > text.size() - patch.anchor_offset) {
        return false;
      }
      for (uint64_t offset = patch.anchor_offset + sizeof(uint32_t);
           offset < patch.anchor_offset + patch.original_size; offset += sizeof(uint32_t)) {
        uint32_t word = 0u;
        std::memcpy(&word, text.data() + offset, sizeof(word));
        if (word != nop)
          return false;
      }
      return true;
    });
    if (unused == patches.end())
      break;

    const ConSanPatchInfo reservoir = *unused;
    if (reservoir.trampoline_offset > text.size() ||
        reservoir.trampoline_size > text.size() - reservoir.trampoline_offset ||
        reservoir.original_size > text.size() - reservoir.trampoline_offset) {
      errors.emplace_back("ConSan MOI barrier pass found an invalid provisional reservoir");
      return false;
    }
    std::memcpy(text.data() + reservoir.anchor_offset, text.data() + reservoir.trampoline_offset,
                reservoir.original_size);
    for (uint64_t offset = reservoir.trampoline_offset;
         offset < reservoir.trampoline_offset + reservoir.trampoline_size;
         offset += sizeof(uint32_t)) {
      if (!write_word_bytes(text, offset, nop)) {
        errors.emplace_back("ConSan MOI barrier pass could not clear a provisional reservoir");
        return false;
      }
    }

    std::unordered_set<uint64_t> route_relays;
    if (reservoir.branch_only_route) {
      const auto &entry_relays = reservoir.branch_only_route->entry_relay_offsets();
      const auto &return_relays = reservoir.branch_only_route->return_relay_offsets;
      route_relays.insert(entry_relays.begin(), entry_relays.end());
      route_relays.insert(return_relays.begin(), return_relays.end());
    }
    for (uint64_t relay : route_relays) {
      if (!write_word_bytes(text, relay, nop)) {
        errors.emplace_back("ConSan MOI barrier pass could not restore a provisional relay");
        return false;
      }
    }

    std::erase_if(patches, [&](const ConSanPatchLoweringProduct &patch) {
      if (is_direct_reservoir(patch) && patch.anchor_offset == reservoir.anchor_offset &&
          patch.trampoline_offset == reservoir.trampoline_offset)
        return true;
      const bool is_route_relay =
          patch.kind == ConSanPatchKind::TrampolineNopBranchRelay ||
          (patch.kind == ConSanPatchKind::TrampolineBranchRelayReservoir &&
           patch.original_size == 0u && patch.trampoline_size == sizeof(uint32_t));
      return is_route_relay && route_relays.contains(patch.trampoline_offset);
    });
    pruned_any = true;
  }

  return pruned_any;
}

void try_apply_record_replay_barrier_record_patch(
    std::span<const uint8_t> bytes, const ConSanOptions &options,
    const ConSanMoiOperatingPoint &operating_point, rj_code_arch_t arch,
    const MoiRecordReplayAccessOutput &access_output, const MoiObjectModeSemantics &semantics,
    std::vector<std::string> &warnings, ConSanTransformArtifacts &result) {
  if (!options.moi_track_barriers)
    return;
  if (!consan_is_capability_arch(arch)) {
    warnings.emplace_back("ConSan MOI barrier record patch does not support this architecture");
    return;
  }
  if (!options.moi_report_buffer_address) {
    warnings.emplace_back("ConSan MOI barrier record patch requires a MOI report buffer");
    return;
  }
  if (options.moi_report_buffer_size < consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 1)) {
    warnings.emplace_back(
        "ConSan MOI barrier record patch requires room for access and barrier records");
    return;
  }
  const ConSanMoiReportBufferLayout &layout = semantics.report_layout;
  const uint32_t max_candidates = std::min(options.max_patches, layout.barrier_record_capacity);
  if (max_candidates == 0) {
    warnings.emplace_back("ConSan MOI barrier record patch has no barrier record capacity");
    return;
  }
  if (!operating_point.moi_exec_save_sgpr) {
    warnings.emplace_back("ConSan MOI barrier record patch requires RJ_CONSAN_MOI_EXEC_SAVE_SGPR");
    return;
  }
  if (*operating_point.moi_exec_save_sgpr > 100u ||
      *operating_point.moi_exec_save_sgpr % 2u != 0u) {
    warnings.emplace_back(
        "ConSan MOI barrier record patch requires an even RJ_CONSAN_MOI_EXEC_SAVE_SGPR in 0..100");
    return;
  }

  std::vector<MoiBarrierEvidenceSitePlan> candidates =
      build_moi_barrier_evidence_site_plans(result.program_inventory, result.observation_plan(),
                                            ConSanProbeIntentKind::BarrierRecord, result.errors);
  if (!result.errors.empty())
    return;
  if (candidates.empty()) {
    warnings.emplace_back("ConSan MOI barrier record patch found no barriers");
    return;
  }

  const std::span<const uint8_t> active_bytes = result.active_bytes(bytes);
  AmdGpuCodeObject code_object(active_bytes.data(), active_bytes.size());
  CodeObjectPatcher patcher(code_object);
  const std::span<const uint8_t> old_text = patcher.text_bytes();
  if (old_text.empty()) {
    result.errors.emplace_back("ConSan MOI barrier record patch found no .text section");
    return;
  }
  struct PlannedBarrierRecord : MoiPlannedRecordEvent {
    MoiBarrierEvidenceSitePlan candidate;
  };
  std::vector<PlannedBarrierRecord> selected_candidates;
  selected_candidates.reserve(max_candidates);
  std::unordered_set<uint64_t> selected_offsets;
  const MoiPatchAnchorRangeIndex occupied_anchors(result.patches);
  MoiResourcePlanningStatePtr standalone_resource_state = make_moi_resource_planning_state(
      bytes, arch, result.program_inventory, result.observation_plan(), result.resource_plans,
      options, options, operating_point);
  MoiSpillManagers spill_managers;
  MoiDescriptorVgprRequirements descriptor_requirements;
  MoiDescriptorSgprRequirements scalar_requirements;
  MoiDescriptorPrivateRequirements private_requirements;
  for (size_t candidate_index = 0; candidate_index < candidates.size(); ++candidate_index) {
    const MoiBarrierEvidenceSitePlan &candidate = candidates[candidate_index];
    if (selected_candidates.size() == max_candidates)
      break;
    const ConSanBarrierSite &site = candidate.site;
    if (site.size != sizeof(uint32_t)) {
      warnings.emplace_back("ConSan MOI barrier record patch skipped non-32-bit barrier " +
                            site.mnemonic + " in " + candidate.container_name);
      continue;
    }
    if (site.text_offset > old_text.size() || site.size > old_text.size() - site.text_offset) {
      result.errors.emplace_back("ConSan MOI barrier record patch site is outside .text");
      return;
    }
    if (occupied_anchors.overlaps(site.text_offset, site.size) ||
        !selected_offsets.insert(site.text_offset).second) {
      warnings.emplace_back("ConSan MOI barrier record patch skipped an overlapping patch");
      continue;
    }
    const ConSanCandidateResourcePlan *existing_plan = resource_plan_for_site(
        result.resource_plans, ConSanResourceSiteKind::Barrier, site.text_offset);
    ConSanCandidateResourcePlan plan =
        existing_plan
            ? *existing_plan
            : plan_moi_resource_site(*standalone_resource_state, options, options, operating_point,
                                     ConSanResourceSiteKind::Barrier, candidate_index,
                                     site.text_offset, candidate.kernel_descriptor_file_offset,
                                     /*scratch_count=*/6u);
    plan.intent_ids = {candidate.evidence_intent};
    if (!existing_plan)
      result.resource_plans.push_back(plan);
    const auto resources = resolve_moi_scratch_plan(
        plan, operating_point, result.moi_operating_point, /*expected_count=*/6u);
    if (!resources) {
      warnings.emplace_back("ConSan MOI barrier record patch has no usable resource plan in " +
                            candidate.container_name);
      continue;
    }
    MoiBarrierEvidenceSitePlan planned_candidate = candidate;
    if (!operating_point.moi_owner_epoch_vgprs.owner() &&
        !planned_candidate.kernel_descriptor_file_offset) {
      if (resources->owner_descriptor_file_offsets.empty()) {
        warnings.emplace_back(
            "ConSan MOI barrier record patch requires an owning kernel descriptor");
        continue;
      }
      planned_candidate.kernel_descriptor_file_offset =
          resources->owner_descriptor_file_offsets.front();
    }
    auto record_event =
        plan_moi_record_event(bytes, result.program_inventory, warnings, *resources, options,
                              options, operating_point, result.moi_operating_point, semantics,
                              spill_managers, arch, "ConSan MOI barrier record patch");
    if (!record_event)
      continue;
    selected_candidates.push_back({std::move(*record_event), std::move(planned_candidate)});
  }

  if (selected_candidates.empty()) {
    warnings.emplace_back("ConSan MOI barrier record patch found no patchable barriers");
    return;
  }
  const std::vector<std::pair<uint64_t, uint64_t>> borrowed_barrier_protected_ranges =
      collect_moi_synchronization_ranges(result.program_inventory);
  std::vector<uint8_t> new_text(old_text.begin(), old_text.end());
  DbiPatchPlacementPlanner placement_planner(arch, old_text.size());
  MoiLocalNopIslandAllocator indirect_allocator(old_text, result.program_inventory, result.patches,
                                                arch, kMoiLocalIndirectIslandWords);
  if (operating_point.moi_exec_save_sgpr) {
    const auto reserved_barrier_island_layout = validate_reserved_barrier_island_layout(
        old_text, arch, access_output.reserved_sync_islands, selected_candidates.size(),
        moi_reserved_access_sync_island_count(result.program_inventory, result.observation_plan(),
                                              options, options, semantics));
    if (reserved_barrier_island_layout) {
      const uint64_t reserved_words = reserved_barrier_island_layout->reserved_slot_count *
                                      reserved_barrier_island_layout->slot_words;
      if (reserved_words > std::numeric_limits<uint32_t>::max()) {
        result.errors.emplace_back("ConSan MOI reserved barrier-island layout is too large");
        return;
      }
      indirect_allocator.reserve(reserved_barrier_island_layout->begin,
                                 static_cast<uint32_t>(reserved_words));
    }
  }
  BranchOnlyRelayRouter barrier_branch_router;
  const uint32_t pristine_nop = build_s_nop(0u, arch);
  for (const auto &[begin, end] : access_output.generated_branch_relay_ranges) {
    if (begin > end || begin % sizeof(uint32_t) != 0u || end % sizeof(uint32_t) != 0u ||
        begin > old_text.size() || end > old_text.size()) {
      result.errors.emplace_back("ConSan MOI generated barrier-relay range is invalid");
      return;
    }
    for (uint64_t offset = begin; offset < end; offset += sizeof(uint32_t)) {
      uint32_t word = 0u;
      std::memcpy(&word, old_text.data() + offset, sizeof(word));
      if (word == pristine_nop)
        (void)barrier_branch_router.offer(offset, BranchOnlyRelayProvenance::GeneratedBank);
    }
  }
  // A selected multiword access unconditionally branches away from every
  // recorded tail word in its replaced anchor. The access pass publishes
  // those exact ownership offsets in its patch record; import only tails that
  // remain NOPs after access routing. This gives the later barrier batch a
  // dense, already-paid original-text frontier without rediscovering guest
  // code or weakening capacity-one relay ownership.
  for (const ConSanPatchLoweringProduct &patch : result.patches) {
    if (patch.kind != ConSanPatchKind::TrampolineMoiAccessRecordStore)
      continue;
    for (uint64_t offset : patch.branch_relay_offsets) {
      const bool is_owned_tail = offset >= patch.anchor_offset + sizeof(uint32_t) &&
                                 offset < patch.anchor_offset + patch.original_size;
      if (!is_owned_tail || offset > old_text.size() ||
          sizeof(uint32_t) > old_text.size() - offset) {
        result.errors.emplace_back("ConSan MOI access-owned barrier relay is invalid");
        return;
      }
      uint32_t word = 0u;
      std::memcpy(&word, old_text.data() + offset, sizeof(word));
      if (word == pristine_nop)
        (void)barrier_branch_router.offer(offset, BranchOnlyRelayProvenance::OwnedAnchor);
    }
  }
  // The access pass may have relocated ordinary instruction runs to create
  // direct reservoirs in pristine text.  Its committed routes consume some
  // tail words, but every tail word that remains a NOP is still capacity-one
  // storage owned by the materialized reservoir.  Recover that storage from
  // the durable patch record instead of rediscovering or relocating the
  // original instructions a second time.  These footholds are essential when
  // an original barrier cannot directly reach the appended generated bank.
  for (const ConSanPatchLoweringProduct &patch : result.patches) {
    if (patch.kind != ConSanPatchKind::TrampolineBranchRelayReservoir ||
        patch.original_size < 2u * sizeof(uint32_t)) {
      continue;
    }
    const uint64_t end = patch.anchor_offset + patch.original_size;
    if (end < patch.anchor_offset || end > old_text.size()) {
      result.errors.emplace_back("ConSan MOI materialized barrier-relay reservoir is invalid");
      return;
    }
    for (uint64_t offset = patch.anchor_offset + sizeof(uint32_t); offset < end;
         offset += sizeof(uint32_t)) {
      uint32_t word = 0u;
      std::memcpy(&word, old_text.data() + offset, sizeof(word));
      if (word == pristine_nop) {
        (void)barrier_branch_router.offer_materialized_owner(
            offset, BranchOnlyRelayProvenance::OwnedReservoir,
            BranchOnlyRelayOwnerIdentity::direct_reservoir(patch.anchor_offset));
      }
    }
  }
  const size_t barrier_relay_offer_count = barrier_branch_router.available_count();
  const BranchOnlyRelaySearchLimits barrier_relay_limits =
      branch_only_relay_greedy_pair_limits(barrier_relay_offer_count);

  struct StagedBarrierRecord {
    const PlannedBarrierRecord *planned = nullptr;
    uint32_t original_barrier_word = 0u;
    uint64_t return_text_offset = 0u;
    DbiPatchPlacement placement;
    std::vector<uint32_t> cave_words;
    std::optional<MoiBorrowedRecordReplayEntry> borrowed_entry;
    std::optional<BranchOnlyRelayRoute> branch_only_route;
  };
  std::vector<StagedBarrierRecord> staged_barriers;
  staged_barriers.reserve(selected_candidates.size());
  std::vector<BranchOnlyRelayPairRequest> branch_only_requests;
  std::vector<size_t> branch_only_staged_indices;
  branch_only_requests.reserve(selected_candidates.size());
  branch_only_staged_indices.reserve(selected_candidates.size());
  for (const PlannedBarrierRecord &planned : selected_candidates) {
    const MoiBarrierEvidenceSitePlan &candidate = planned.candidate;
    const ConSanBarrierSite &site = candidate.site;
    uint32_t original_barrier_word = 0u;
    std::memcpy(&original_barrier_word, new_text.data() + site.text_offset,
                sizeof(original_barrier_word));

    const uint64_t cave_text_offset = placement_planner.appended_end();
    uint64_t return_text_offset = site.text_offset + site.size;
    const MoiRecordEventEmissionPlan &emission = planned.emission;
    const bool branch_only_scalar_spill = moi_transient_sgpr_assignment_is_branch_only(
        result.moi_operating_point, planned.resources.owner_descriptor_file_offsets);
    // A routed branch-only body replaces this harmless placeholder after all
    // body coordinates and the shared relay plan are fixed.
    const uint64_t cave_return_target =
        branch_only_scalar_spill ? cave_text_offset : return_text_offset;
    auto cave_words = build_barrier_record_cave_words(
        active_bytes, candidate, emission, planned.spill ? &*planned.spill : nullptr,
        planned.scalar_spill ? &*planned.scalar_spill : nullptr, arch,
        layout.barrier_record_capacity, layout.barrier_records_offset, original_barrier_word,
        cave_text_offset, cave_return_target, result.errors);
    if (!cave_words)
      return;
    std::optional<MoiBorrowedRecordReplayEntry> borrowed_entry;
    uint32_t displaced_original_size = site.size;
    if (branch_only_scalar_spill &&
        moi_transient_sgpr_assignment_uses_borrowed_record_replay_entry(
            options, result.moi_operating_point, planned.resources.owner_descriptor_file_offsets)) {
      std::vector<std::string> ignored_errors;
      std::vector<uint32_t> displaced_words;
      while (displaced_words.size() < kMoiRecordReplayBorrowedEntryIslandWords) {
        auto decoded = decode_relocatable_entry_instruction(
            old_text, site.text_offset + displaced_words.size() * sizeof(uint32_t), arch,
            ignored_errors, moi_resource_decoder(*standalone_resource_state));
        if (!decoded) {
          displaced_words.clear();
          break;
        }
        displaced_words.insert(displaced_words.end(), decoded->begin(), decoded->end());
      }
      const uint64_t displaced_size = displaced_words.size() * sizeof(uint32_t);
      const uint64_t displaced_end = site.text_offset + displaced_size;
      const bool overlaps_existing_patch =
          displaced_size > std::numeric_limits<uint32_t>::max() ||
          occupied_anchors.overlaps(site.text_offset, static_cast<uint32_t>(displaced_size));
      const bool crosses_control_flow_entry = !moi_resource_entry_window_is_single_entry(
          *standalone_resource_state, site.text_offset, displaced_size);
      auto protected_range =
          std::ranges::lower_bound(borrowed_barrier_protected_ranges, site.text_offset, {},
                                   [](const auto &range) { return range.first; });
      bool overlaps_other_sync = false;
      for (; protected_range != borrowed_barrier_protected_ranges.end() &&
             protected_range->first < displaced_end;
           ++protected_range) {
        if (protected_range->first != site.text_offset ||
            protected_range->second != site.text_offset + site.size) {
          overlaps_other_sync = true;
          break;
        }
      }
      const auto jump_sgprs = emission.indirect_jump;
      const auto backup_vgpr = find_moi_borrowed_entry_backup_vgpr(
          *standalone_resource_state, planned.resources.owner_descriptor_file_offsets,
          site.text_offset, planned.resources.required_vgpr_count,
          {
              .owner_epoch = moi_owner_epoch_vgpr_sources(emission.moi_owner_epoch_vgprs),
              .workgroup_key = emission.moi_workgroup_key_vgpr,
              .dispatch_id = emission.moi_dispatch_id_vgpr,
              .record_replay_workgroup = emission.moi_record_replay_workgroup_vgprs,
          });
      if (!displaced_words.empty() && !overlaps_existing_patch && !crosses_control_flow_entry &&
          !overlaps_other_sync && jump_sgprs && backup_vgpr) {
        borrowed_entry = build_moi_borrowed_record_replay_entry(
            site.text_offset, cave_text_offset, *backup_vgpr, *jump_sgprs,
            static_cast<uint32_t>(displaced_words.size()), arch);
      }
      if (borrowed_entry) {
        const size_t site_words = site.size / sizeof(uint32_t);
        cave_words->insert(cave_words->begin(), borrowed_entry->scalar_restore_words.begin(),
                           borrowed_entry->scalar_restore_words.end());
        cave_words->insert(cave_words->end() - 1u, displaced_words.begin() + site_words,
                           displaced_words.end());
        displaced_original_size = static_cast<uint32_t>(displaced_size);
        return_text_offset = displaced_end;
      }
    }
    std::string placement_error;
    auto placement = placement_planner.plan_indirect_appended(
        site.text_offset, displaced_original_size, cave_words->size() * sizeof(uint32_t),
        &placement_error);
    if (!placement || placement->body_offset != cave_text_offset) {
      result.errors.emplace_back("ConSan MOI barrier record placement failed: " + placement_error);
      return;
    }
    const size_t staged_index = staged_barriers.size();
    staged_barriers.push_back({
        .planned = &planned,
        .original_barrier_word = original_barrier_word,
        .return_text_offset = return_text_offset,
        .placement = *placement,
        .cave_words = std::move(*cave_words),
        .borrowed_entry = std::move(borrowed_entry),
        .branch_only_route = std::nullopt,
    });
    if (branch_only_scalar_spill) {
      const StagedBarrierRecord &staged = staged_barriers.back();
      branch_only_requests.push_back({
          .entry_source = site.text_offset,
          .entry_target =
              staged.borrowed_entry ? site.text_offset + sizeof(uint32_t) : cave_text_offset,
          .return_source =
              cave_text_offset + (staged.cave_words.size() - 1u) * sizeof(uint32_t),
          .return_target = return_text_offset,
          .entry_preplaced = staged.borrowed_entry.has_value(),
      });
      branch_only_staged_indices.push_back(staged_index);
    }
  }

  if (!branch_only_requests.empty()) {
    DbiPatchPlacementPlanner tentative_planner = placement_planner;
    BranchOnlyRelayRouter tentative_router = barrier_branch_router;
    std::string relay_error;
    BranchOnlyRelayBatchPlan route_plan = tentative_router.plan_pairs(
        tentative_planner, branch_only_requests, &relay_error, barrier_relay_limits);
    if (!route_plan.complete() || route_plan.routes.size() != branch_only_requests.size() ||
        !tentative_router.commit(route_plan.routes, &relay_error)) {
      warnings.emplace_back(
          "ConSan MOI barrier record patch found no complete routed register-free batch for " +
          std::to_string(branch_only_requests.size()) + " barrier(s)" +
          (relay_error.empty() ? std::string{} : ": " + relay_error));
    } else {
      placement_planner = std::move(tentative_planner);
      barrier_branch_router = std::move(tentative_router);
      for (size_t route_index = 0u; route_index < route_plan.routes.size(); ++route_index) {
        StagedBarrierRecord &staged = staged_barriers[branch_only_staged_indices[route_index]];
        staged.branch_only_route = std::move(route_plan.routes[route_index]);
        const uint64_t return_branch_offset =
            staged.placement.body_offset +
            (staged.cave_words.size() - 1u) * sizeof(uint32_t);
        const uint64_t return_hop_target =
            staged.branch_only_route->return_relay_offsets.empty()
                ? staged.return_text_offset
                : staged.branch_only_route->return_relay_offsets.front();
        const auto return_delta =
            compute_sopp_branch_simm16(return_branch_offset, return_hop_target);
        if (!return_delta) {
          result.errors.emplace_back("ConSan MOI barrier record routed return lost its first hop");
          return;
        }
        staged.cave_words.back() = build_s_branch(*return_delta, arch);
      }
    }
  }

  std::vector<ConSanPatchInfo> patches;
  std::vector<ConSanCommittedLowering> lowering_commits;
  for (StagedBarrierRecord &staged : staged_barriers) {
    const PlannedBarrierRecord &planned = *staged.planned;
    const MoiBarrierEvidenceSitePlan &candidate = planned.candidate;
    const ConSanBarrierSite &site = candidate.site;
    const MoiRecordEventEmissionPlan &emission = planned.emission;
    const uint64_t cave_text_offset = staged.placement.body_offset;
    const uint64_t return_text_offset = staged.return_text_offset;
    const bool branch_only_scalar_spill = moi_transient_sgpr_assignment_is_branch_only(
        result.moi_operating_point, planned.resources.owner_descriptor_file_offsets);
    std::optional<BranchOnlyRelayRoute> &branch_only_route = staged.branch_only_route;
    while (new_text.size() < cave_text_offset)
      append_word_bytes(new_text, build_s_nop(0u, arch));
    if (new_text.size() != cave_text_offset) {
      result.errors.emplace_back("ConSan MOI barrier record patch has a stale cave mapping");
      return;
    }
    if (branch_only_scalar_spill && !branch_only_route) {
      for (size_t word = 0u; word < staged.cave_words.size(); ++word)
        append_word_bytes(new_text, build_s_nop(0u, arch));
      continue;
    }

    const bool direct_entry =
        staged.borrowed_entry ||
        compute_sopp_branch_simm16(site.text_offset, cave_text_offset).has_value();
    uint64_t entry_island = 0;
    bool has_entry_island = false;
    if (!branch_only_scalar_spill && !direct_entry) {
      const auto claimed_island = indirect_allocator.claim_reachable(site.text_offset);
      if (!claimed_island) {
        warnings.emplace_back(
            "ConSan MOI barrier record patch found no reachable indirect entry island in " +
            candidate.container_name);
        for (size_t word = 0u; word < staged.cave_words.size(); ++word)
          append_word_bytes(new_text, build_s_nop(0u, arch));
        continue;
      }
      entry_island = *claimed_island;
      has_entry_island = true;
    }
    std::string placement_error;

    const uint64_t entry_target =
        branch_only_route && !branch_only_route->entry_relay_offsets.empty()
            ? branch_only_route->entry_relay_offsets.front()
            : (has_entry_island ? entry_island : cave_text_offset);
    if (staged.borrowed_entry) {
      const uint64_t island_bytes = staged.borrowed_entry->island_words.size() * sizeof(uint32_t);
      if (site.text_offset > new_text.size() || island_bytes > new_text.size() - site.text_offset) {
        result.errors.emplace_back(
            "ConSan MOI barrier record borrowed entry exceeds reserved text");
        return;
      }
      std::memcpy(new_text.data() + site.text_offset, staged.borrowed_entry->island_words.data(),
                  static_cast<size_t>(island_bytes));
    } else {
      const auto fwd = compute_sopp_branch_simm16(staged.placement.anchor_offset, entry_target);
      if (!fwd) {
        result.errors.emplace_back("ConSan MOI barrier record forward branch is out of range");
        return;
      }
      if (!write_word_bytes(new_text, site.text_offset, build_s_branch(*fwd, arch))) {
        result.errors.emplace_back(
            "ConSan MOI barrier record patch could not rewrite barrier slot");
        return;
      }
    }
    if (has_entry_island) {
      const bool emitted = emit_moi_local_indirect_entry_island(
          new_text, entry_island, cave_text_offset, site.text_offset, emission,
          planned.resources.owner_descriptor_file_offsets, arch, patches, result.errors,
          "ConSan MOI barrier record patch");
      if (!emitted)
        return;
    }
    append_words_bytes(new_text, staged.cave_words);
    if (branch_only_route && !BranchOnlyRelayRouter::emit_and_record(
                                 new_text, *branch_only_route, cave_text_offset, return_text_offset,
                                 arch, patches, &placement_error)) {
      result.errors.emplace_back("ConSan MOI barrier record patch " + placement_error);
      return;
    }

    ConSanPatchInfo info;
    info.kind = ConSanPatchKind::TrampolineMoiBarrierRecord;
    info.anchor_offset = site.text_offset;
    info.trampoline_offset = cave_text_offset;
    info.original_size = staged.placement.original_size;
    info.trampoline_size = static_cast<uint32_t>(staged.cave_words.size() * sizeof(uint32_t));
    note_relocated_barrier_instruction(info, staged.cave_words, staged.original_barrier_word,
                                       cave_text_offset, cave_text_offset,
                                       /*emits_barrier_record=*/true);
    note_moi_probe_patch_info(info, planned);
    if (branch_only_route)
      info.branch_only_route = consan_branch_only_continuation(*branch_only_route);
    if (staged.borrowed_entry) {
      if (!info.branch_only_route) {
        result.errors.emplace_back(
            "ConSan MOI barrier borrowed entry lost its branch-only return route");
        return;
      }
      const ConSanIndirectJumpSgprs jump_sgprs = *emission.indirect_jump;
      info.branch_only_route->entry = ConSanBranchOnlyBorrowedEntry{
          .jump = jump_sgprs,
          .backup_vgpr = staged.borrowed_entry->backup_vgpr,
          .empty_exec_continuation_offset = std::nullopt,
      };
    }
    // Descriptor mutations belong to emitted patches, not merely to
    // resource-admitted candidates.  Accumulating them here keeps a later
    // reachability rejection from changing an uninstrumented owner.
    note_descriptor_requirements(descriptor_requirements, planned.resources);
    note_moi_sgpr_requirements(scalar_requirements, planned.resources, emission, arch);
    note_moi_probe_private_requirements(private_requirements, planned);
    if (!append_moi_sync_lowering_commit(result.observation_plan(), result.errors, candidate, info,
                                         "barrier record patch", lowering_commits))
      return;
    patches.push_back(info);
  }

  const bool pruned_provisional_reservoirs =
      prune_unused_record_replay_direct_reservoirs(new_text, arch, result.patches, result.errors);
  if (!result.errors.empty())
    return;
  if (patches.empty()) {
    result.warnings.emplace_back("ConSan MOI barrier record patch found no patchable barriers");
    if (pruned_provisional_reservoirs) {
      if (!replace_consan_text(patcher, new_text, options.patched_image_growth_limit,
                               "MOI barrier reservoir cleanup",
                               result.program_inventory.code_object_id(), result.errors,
                               &result.transform_failure_cause)) {
        return;
      }
      result.replacement = std::move(patcher).emit();
      result.mark_modified();
    }
    return;
  }
  if (!apply_moi_descriptor_requirements(patcher, code_object, result.program_inventory,
                                         descriptor_requirements, scalar_requirements,
                                         private_requirements, nullptr, nullptr, arch,
                                         "ConSan MOI barrier record patch", result.errors)) {
    return;
  }
  (void)publish_moi_sync_patch(result, patcher, new_text, options.patched_image_growth_limit,
                               "MOI barrier record patch", "barrier record patch",
                               std::move(lowering_commits), std::move(patches));
}

[[nodiscard]] std::optional<MoiRecordEventEmissionPlan> plan_record_replay_barrier_emission(
    const ConSanOptions &options, ConSanMoiOperatingPoint &point,
    const MoiScalarAbiPlan &scalar_abi, uint16_t scratch_vgpr, MoiOwnerAssignments assignments,
    std::span<const uint64_t> owners, rj_code_arch_t arch,
    const ConSanMoiPrivateStateLayout *private_layout = nullptr) {
  return apply_record_replay_entry_workgroup_assignment(options, point, assignments, owners,
                                                         private_layout)
             ? resolve_moi_record_event_emission_plan(options, options, point, scalar_abi,
                                                      scratch_vgpr, arch, private_layout)
             : std::nullopt;
}

[[nodiscard]] std::optional<PlannedBarrierLowering>
plan_record_replay_private_barrier(MoiPrivateBarrierPlanningContext context) {
  auto &[bytes, options, point, candidate, resource_plan, access_patch, semantics, spill_managers,
         arch, inventory, assignments, warnings] = context;
  const std::vector<uint64_t> &owners = resource_plan.owner_descriptor_file_offsets;
  const ConSanMoiPrivateStateLayout &private_state_layout = *access_patch.private_state_layout;
  const auto resources = resolve_moi_scratch_plan(resource_plan, point, assignments,
                                                  /*expected_count=*/6u);
  if (!resources || !private_state_layout.record_replay_workgroup_offsets.complete()) {
    warnings.emplace_back(
        "ConSan MOI Record/Replay dense barrier has no complete private exact-workgroup layout in " +
        candidate.container_name);
    return std::nullopt;
  }
  const MoiScalarAbiPlan scalar_abi =
      plan_moi_scalar_abi(options.moi_engine, moi_scalar_routing_state(point));
  auto record_emission = plan_record_replay_barrier_emission(
      options, point, scalar_abi, resources->base, assignments, owners, arch,
      &private_state_layout);
  if (!record_emission) {
    warnings.emplace_back("ConSan MOI Record/Replay dense barrier has no owner-local emission");
    return std::nullopt;
  }
  std::optional<MoiWorkitemOwnerDerivationPlan> derived_owner;
  if (!point.moi_owner_epoch_vgprs.owner() && !point.moi_persistent_sgprs.owner()) {
    const auto owner_shift = common_moi_workitem_owner_shift(bytes, *resources, arch, warnings);
    if (!owner_shift)
      return std::nullopt;
    derived_owner = MoiWorkitemOwnerDerivationPlan{
        .entry_workitem_x_private_offset = private_state_layout.owner_offset,
        .wave_size_shift = *owner_shift,
    };
  }
  record_emission->derived_owner = std::move(derived_owner);
  const uint32_t ephemeral_base = private_state_layout.ephemeral_base;
  std::optional<VgprSpillSequence> spill;
  if (resources->source == ConSanRegisterAllocationSource::SpillRequired ||
      moi_scalar_spill_requires_dynamic_vgpr_frame(inventory, *resources, point)) {
    spill = build_moi_spill_sequence(inventory, *resources, options, options, point,
                                     semantics, spill_managers, arch, warnings,
                                     ephemeral_base);
    if (!spill)
      return std::nullopt;
  }
  std::optional<SgprSpillSequence> scalar_spill;
  if (point.has_compact_moi_scalar_spill()) {
    scalar_spill = build_moi_sgpr_spill_sequence(
        inventory, *resources, options, options, point, semantics, spill_managers, arch, warnings,
        ephemeral_base, spill ? &*spill : nullptr);
    if (!scalar_spill)
      return std::nullopt;
  }
  uint32_t required_private_bytes = ephemeral_base;
  if (spill)
    required_private_bytes = std::max(required_private_bytes, spill->total_private_bytes);
  if (scalar_spill)
    required_private_bytes = std::max(required_private_bytes, scalar_spill->total_private_bytes);
  ConSanPatchAbiEffects patch_abi = plan_barrier_patch_abi(
      owners, resources->base, private_state_layout, spill ? &*spill : nullptr,
      required_private_bytes);
  return PlannedBarrierLowering{
      .candidate = candidate,
      .body = *record_emission,
      .spill = std::move(spill),
      .scalar_spill = std::move(scalar_spill),
      .patch_abi = std::move(patch_abi),
  };
}

[[nodiscard]] std::optional<MoiBarrierBodyPlan> plan_record_replay_barrier_body(
    const ConSanOptions &options, ConSanMoiOperatingPoint &point,
    const MoiScalarAbiPlan &scalar_abi, std::optional<uint16_t> scratch_vgpr,
    std::span<const uint64_t> owners, rj_code_arch_t arch,
    MoiOwnerAssignments assignments, std::vector<std::string> &warnings,
    std::vector<std::string> &) {
  if (!scratch_vgpr) {
    warnings.emplace_back(
        "ConSan MOI Record/Replay dense barrier fallback has no scratch VGPR state");
    return std::nullopt;
  }
  auto record = plan_record_replay_barrier_emission(options, point, scalar_abi, *scratch_vgpr,
                                                    assignments, owners, arch);
  if (!record) {
    warnings.emplace_back("ConSan MOI Record/Replay dense fallback has no owner-local emission");
    return std::nullopt;
  }
  return MoiBarrierBodyPlan{std::move(*record)};
}

void try_apply_record_replay_barrier_patch(
    std::span<const uint8_t> bytes, const ConSanOptions &options,
    const ConSanMoiOperatingPoint &operating_point, rj_code_arch_t arch,
    MoiResourcePlanningState &resource_state, const MoiRecordReplayAccessOutput &access_output,
    const MoiObjectModeSemantics &semantics, ConSanTransformArtifacts &result) {
  if (!options.moi_track_barriers)
    return;

  std::vector<std::string> record_warnings;
  try_apply_record_replay_barrier_record_patch(bytes, options, operating_point, arch, access_output,
                                               semantics, record_warnings, result);
  const auto publish_record_warnings = [&](bool omit_recovered_route_failures) {
    for (std::string &warning : record_warnings) {
      const bool recovered_route_failure =
          warning.find("no reachable indirect entry island") != std::string::npos ||
          warning == "ConSan MOI barrier record patch found no patchable barriers";
      if (!omit_recovered_route_failures || !recovered_route_failure)
        result.warnings.push_back(std::move(warning));
    }
  };
  if (!result.errors.empty()) {
    publish_record_warnings(false);
    return;
  }
  const bool needs_dense_fallback =
      std::ranges::any_of(record_warnings, [](const std::string &warning) {
        return warning.find("no reachable indirect entry island") != std::string::npos;
      });
  if (!needs_dense_fallback) {
    publish_record_warnings(false);
    return;
  }
  const size_t failed_entry_count =
      std::ranges::count_if(record_warnings, [](const std::string &warning) {
        return warning.find("no reachable indirect entry island") != std::string::npos;
      });
  publish_record_warnings(true);
  const size_t barrier_count_before = std::ranges::count(
      result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord, &ConSanPatchInfo::kind);
  try_apply_shared_barrier_patch(
      bytes, options, ConSanProbeIntentKind::BarrierRecord, plan_record_replay_barrier_body,
      plan_record_replay_private_barrier, operating_point, arch, resource_state,
      access_output.reserved_sync_islands, semantics, result);
  const size_t barrier_count_after = std::ranges::count(
      result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord, &ConSanPatchInfo::kind);
  if (!result.errors.empty() || barrier_count_after - barrier_count_before < failed_entry_count) {
    for (std::string &warning : record_warnings) {
      if (warning.find("no reachable indirect entry island") != std::string::npos ||
          warning == "ConSan MOI barrier record patch found no patchable barriers") {
        result.warnings.push_back(std::move(warning));
      }
    }
  }
}
