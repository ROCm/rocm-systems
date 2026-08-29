void try_apply_inline_atomic_ordering_patch(std::span<const uint8_t> bytes,
                                            const MoiOptions &options, rj_code_arch_t arch,
                                            ConSanTransformArtifacts &result) {
  if (!options.moi_track_atomics || options.moi_engine != ConSanMoiEngine::InlineShadow)
    return;
  const ConSanTargetProfile *target = consan_target_profile(arch);
  if (target == nullptr) {
    result.warnings.emplace_back(
        "ConSan MOI inline atomic patch does not support this architecture");
    return;
  }
  if (!options.moi_report_buffer_address) {
    result.warnings.emplace_back("ConSan MOI inline atomic patch requires a MOI report buffer");
    return;
  }
  const ConSanMoiReportBufferLayout layout = resolve_moi_report_layout(options, options);
  if (layout.inline_atomic_release_capacity == 0 ||
      layout.inline_acquired_epoch_token_capacity == 0) {
    result.warnings.emplace_back(
        "ConSan MOI inline atomic patch has no release or acquired-token metadata slot");
    return;
  }
  if ((!options.moi_owner_vgpr || !options.moi_epoch_vgpr) &&
      (!options.moi_persistent_sgprs.owner || !options.moi_persistent_sgprs.epoch) &&
      !options.automatic_moi_private_epoch) {
    result.warnings.emplace_back("ConSan MOI inline atomic patch requires "
                                 "RJ_CONSAN_MOI_OWNER_VGPR and RJ_CONSAN_MOI_EPOCH_VGPR");
    return;
  }
  if (!validate_inline_atomic_exec_save_sgpr(static_cast<const ConSanMoiOperatingPoint &>(options),
                                             result.warnings))
    return;

  std::vector<MoiAtomicEvidenceSitePlan> candidates = build_moi_atomic_evidence_site_plans(
      result, ConSanProbeIntentKind::ExactAtomicOrdering, result.errors);
  if (!result.errors.empty())
    return;
  std::sort(candidates.begin(), candidates.end(), [](const auto &lhs, const auto &rhs) {
    return lhs.site.text_offset < rhs.site.text_offset;
  });
  if (candidates.empty()) {
    result.warnings.emplace_back("ConSan MOI inline atomic patch found no supported flat atomics");
    return;
  }

  const std::span<const uint8_t> active_bytes = active_moi_bytes(bytes, result);
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
    uint32_t relocated_guest_size = 0;
    std::vector<uint32_t> trailing_guest_words;
  };
  std::vector<PlannedInlineAtomic> planned_candidates;
  MoiResourcePlanningState resource_state(bytes, arch, result.program_inventory,
                                          result.observation_plan, result.resource_plans, options,
                                          options, options);
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
      const BasicBlock *site_block = block_for_offset(resource_state.block_index, site.text_offset);
      const BasicBlock *sequence_end_block =
          block_for_offset(resource_state.block_index, sequence_end - sizeof(uint32_t));
      const uint64_t sequence_size = sequence_end - site.text_offset;
      if (site_block == nullptr || sequence_end_block != site_block ||
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
        result.patches.begin(), result.patches.end(), [&](const ConSanPatchInfo &patch) {
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
    const uint16_t scratch_count = inline_shadow_atomic_scratch_count(arch);
    if (existing_plan && existing_plan->scratch_vgpr_count != scratch_count)
      existing_plan = nullptr;
    ConSanCandidateResourcePlan plan =
        existing_plan
            ? *existing_plan
            : plan_moi_resource_site(resource_state, options, options, options,
                                     ConSanResourceSiteKind::Atomic, candidate_index,
                                     site.text_offset, candidate.kernel_descriptor_file_offset,
                                     scratch_count, nullptr, &candidate.lowering_form);
    plan.intent_ids = {candidate.address_capture_intent, candidate.evidence_intent};
    if (!existing_plan)
      result.resource_plans.push_back(plan);
    const auto resources =
        resolve_moi_scratch_plan(plan, options, result.moi_operating_point, scratch_count);
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
          (site.saddr_sgpr ? std::to_string(*site.saddr_sgpr) : std::string("missing")) +
          " raw_vaddr=" +
          (site.raw_vaddr ? std::to_string(*site.raw_vaddr) : std::string("missing")) +
          " addr_vgpr=" +
          (site.addr_vgpr ? std::to_string(*site.addr_vgpr) : std::string("missing")));
      continue;
    }
    MoiAtomicEvidenceSitePlan planned_candidate = candidate;
    if (!planned_candidate.kernel_descriptor_file_offset) {
      planned_candidate.kernel_descriptor_file_offset =
          common_moi_record_owner_descriptor(bytes, *resources, arch, result.warnings);
      if (!planned_candidate.kernel_descriptor_file_offset)
        continue;
    }
    MoiOptions candidate_options = options;
    candidate_options.scratch_vgpr = resources->base;
    if (!apply_moi_transient_sgpr_assignment(candidate_options, candidate_options,
                                             result.moi_operating_point,
                                             resources->owner_descriptor_file_offsets)) {
      result.warnings.emplace_back(
          "ConSan MOI inline atomic patch found no common transient SGPR assignment");
      continue;
    }
    if (!result.moi_operating_point.owner_persistent_vgprs.empty() &&
        !apply_moi_persistent_vgpr_assignment(candidate_options, result.moi_operating_point,
                                              resources->owner_descriptor_file_offsets)) {
      result.warnings.emplace_back(
          "ConSan MOI inline atomic patch found no common persistent VGPR assignment");
      continue;
    }
    std::optional<MoiPrivateEpochLayout> private_layout;
    if (candidate_options.automatic_moi_private_epoch) {
      private_layout = build_moi_private_epoch_layout(
          result.program_inventory, *resources, arch, result.warnings,
          /*include_owner=*/candidate_options.moi_owner_source == ConSanMoiOwnerSource::WorkitemId,
          /*include_workgroup_key=*/true,
          /*include_record_replay_workgroup=*/false,
          /*include_dispatch_id=*/candidate_options.automatic_moi_private_dispatch_id);
      if (!private_layout)
        continue;
      if (private_layout->dispatch_id_offset) {
        if (!candidate_options.moi_inline_indirect_pc_sgpr) {
          result.warnings.emplace_back(
              "ConSan MOI inline atomic private dispatch identity has no owner-local reload pair");
          continue;
        }
        candidate_options.moi_dispatch_id_sgpr = candidate_options.moi_inline_indirect_pc_sgpr;
      }
    }
    if (candidate_options.automatic_moi_inline_sgpr_spill) {
      if (!candidate_options.moi_exec_save_sgpr) {
        result.warnings.emplace_back(
            "ConSan MOI inline atomic spill-backed scalar state has no SGPR window in " +
            candidate.container_name);
        continue;
      }
      const uint16_t scalar_base = *candidate_options.moi_exec_save_sgpr;
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
    auto probe = plan_moi_probe_resources(result, *resources, candidate_options, spill_managers,
                                          arch, std::move(private_layout),
                                          candidate_options.automatic_moi_inline_sgpr_spill);
    if (!probe)
      continue;
    planned_candidates.push_back({std::move(*probe), std::move(planned_candidate), address_plan,
                                  relocated_guest_size, std::move(trailing_guest_words)});
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
    note_moi_probe_private_requirements(private_requirements, planned);
    note_descriptor_requirements(descriptor_requirements, planned.resources);
    MoiOptions candidate_options = options;
    if (!apply_moi_transient_sgpr_assignment(candidate_options, candidate_options,
                                             result.moi_operating_point,
                                             planned.resources.owner_descriptor_file_offsets)) {
      result.errors.emplace_back(
          "ConSan MOI inline atomic patch lost its planned transient SGPR assignment");
      return;
    }
    if (!result.moi_operating_point.owner_persistent_vgprs.empty() &&
        !apply_moi_persistent_vgpr_assignment(candidate_options, result.moi_operating_point,
                                              planned.resources.owner_descriptor_file_offsets)) {
      result.errors.emplace_back(
          "ConSan MOI inline atomic patch lost its planned persistent VGPR assignment");
      return;
    }
    if (planned.private_layout && planned.private_layout->dispatch_id_offset)
      candidate_options.moi_dispatch_id_sgpr = candidate_options.moi_inline_indirect_pc_sgpr;
    note_moi_sgpr_requirements(scalar_requirements, planned.resources, candidate_options,
                               candidate_options, candidate_options, arch);
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
  MoiLocalNopIslandAllocator indirect_allocator(old_text, result.program_inventory, result.patches,
                                                arch, kMoiLocalIndirectIslandWords);
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
  std::vector<ConSanCommittedLowering> lowering_commits;
  for (const PlannedInlineAtomic &planned : planned_candidates) {
    if (!neutralize_atomic_scalar_clause(new_text, planned.candidate, arch, patches, result.errors))
      return;
  }
  std::vector<bool> routed_by_dense_relay(planned_candidates.size(), false);

  // A large generated CDNA kernel can strand an early atomic beyond SOPP
  // reach of both the pristine-end island bank and every appended body.  A
  // liveness-proven relocated host supplies one shared in-kernel island.  Its
  // scalar key identifies the normal host path or one of the atomic bodies.
  if (target->supports_moi_far_dense_atomic_route) {
    std::vector<consan_detail::MoiDenseRouteSite> route_sites;
    for (size_t index = 0; index < planned_candidates.size(); ++index) {
      const PlannedInlineAtomic &planned = planned_candidates[index];
      const uint64_t anchor = planned.candidate.site.text_offset;
      const bool appended_reachable =
          std::ranges::any_of(appended_indirect_islands, [&](const MoiLocalNopIsland &island) {
            return compute_sopp_branch_simm16(anchor, island.text_offset).has_value();
          });
      const bool stranded = !compute_sopp_branch_simm16(anchor, placement_planner.appended_end()) &&
                            !appended_reachable && !indirect_allocator.has_reachable(anchor);
      if (stranded && planned.relocated_guest_size >= 2u * sizeof(uint32_t)) {
        const std::string_view encoded_owner = planned.candidate.container_name;
        const bool in_kernel = encoded_owner.starts_with("kernel:");
        route_sites.push_back({
            .owner = {in_kernel ? ConSanProgramContainerKind::Kernel
                                : ConSanProgramContainerKind::Function,
                      std::string(encoded_owner.substr(in_kernel ? sizeof("kernel:") - 1u
                                                                 : sizeof("function:") - 1u))},
            .anchor = anchor,
            .source_index = index,
        });
      }
    }
    const auto route_groups = consan_detail::partition_moi_dense_route_sites(route_sites, 31u);

    std::vector<std::pair<uint64_t, uint64_t>> fixed_claims;
    fixed_claims.reserve(result.patches.size() + planned_candidates.size());
    for (const ConSanPatchInfo &patch : result.patches) {
      if (patch.original_size != 0u)
        fixed_claims.emplace_back(patch.anchor_offset, patch.anchor_offset + patch.original_size);
    }
    for (const PlannedInlineAtomic &planned : planned_candidates) {
      fixed_claims.emplace_back(planned.candidate.site.text_offset,
                                planned.candidate.site.text_offset + planned.relocated_guest_size);
    }
    const consan_detail::MoiOccupiedTextRanges occupied(std::move(fixed_claims));
    std::vector<std::pair<uint64_t, uint64_t>> claimed_hosts;
    const auto overlaps_existing = [&](uint64_t begin, uint64_t end) {
      return occupied.overlaps(begin, end) ||
             std::ranges::any_of(claimed_hosts, [&](const auto &range) {
               return range.first < end && begin < range.second;
             });
    };

    for (const consan_detail::MoiDenseRouteIndexGroup &route_group : route_groups) {
      const std::vector<size_t> &group = route_group.source_indices;
      // Keep the route's resource choices and emitted-body descriptors from
      // remaining live while the next independent group is planned.
      {
        std::vector<uint64_t> group_owners;
        std::vector<uint64_t> anchors;
        for (size_t index : group) {
          const PlannedInlineAtomic &planned = planned_candidates[index];
          group_owners.insert(group_owners.end(),
                              planned.resources.owner_descriptor_file_offsets.begin(),
                              planned.resources.owner_descriptor_file_offsets.end());
          anchors.push_back(planned.candidate.site.text_offset);
        }
        std::ranges::sort(group_owners);
        group_owners.erase(std::ranges::unique(group_owners).begin(), group_owners.end());
        MoiOptions group_options = options;
        if (!apply_moi_transient_sgpr_assignment(group_options, group_options,
                                                 result.moi_operating_point, group_owners)) {
          result.warnings.emplace_back(
              "ConSan MOI inline atomic dense relay owners have no common scalar assignment");
          continue;
        }
        if (!result.moi_operating_point.owner_persistent_vgprs.empty() &&
            !apply_moi_persistent_vgpr_assignment(group_options, result.moi_operating_point,
                                                  group_owners)) {
          result.warnings.emplace_back(
              "ConSan MOI inline atomic dense relay owners have no common persistent VGPR "
              "assignment");
          continue;
        }
        const auto scalar_abi =
            moi_inline_dense_router_scalar_abi(group_options, group_options, arch);
        if (!scalar_abi || scalar_abi->call_return_sgpr) {
          result.warnings.emplace_back(
              "ConSan MOI inline atomic dense relay has no explicit-key scalar ABI");
          continue;
        }
        const uint16_t jump_pc_sgpr = scalar_abi->indirect_jump.pc_sgpr;
        const uint16_t saved_scc_sgpr = scalar_abi->indirect_jump.scc_save_sgpr;
        const uint16_t key_sgpr = scalar_abi->dispatch_key_sgpr;
        const bool key_encodes_scc = key_sgpr == saved_scc_sgpr;

        uint64_t owner_begin = 0u;
        uint64_t owner_end = 0u;
        const std::string_view owner_name = route_group.owner.name;
        const ConSanKernelInfo *kernel = result.program_inventory.find_kernel_by_name(owner_name);
        if (kernel != nullptr) {
          owner_begin = kernel->entry_text_offset;
          owner_end = owner_begin + kernel->code_size;
        } else {
          const ConSanFunctionInfo *function =
              result.program_inventory.find_function_by_name(owner_name);
          if (function == nullptr)
            continue;
          owner_begin = function->entry_text_offset;
          owner_end = owner_begin + function->code_size;
        }
        const std::vector<MoiSgprRange> bootstrap_ranges = moi_relocatable_host_scalar_ranges(
            group_options, group_options, MoiRelocatableHostScalarState::AccessRouter, arch);
        MoiDenseOwnerLivenessProof owner_proof =
            resource_state.prepare_dense_liveness_for_owners(group_owners, bootstrap_ranges);
        if (!owner_proof.complete || owner_proof.contexts.empty()) {
          result.warnings.emplace_back(
              "ConSan MOI inline atomic dense relay has incomplete owner liveness");
          continue;
        }
        const auto anchor_admits_bootstrap = [&](uint64_t anchor) {
          BasicBlock *block = block_for_offset(resource_state.block_index, anchor);
          const Instruction *instruction = instruction_at(block, anchor);
          if (block == nullptr || instruction == nullptr)
            return false;
          bool saw_owner = false;
          for (const MoiKernelResourceContext *context : owner_proof.contexts) {
            if (context == nullptr || !context->block_set.contains(block))
              continue;
            saw_owner = true;
            if (context->liveness_is_restricted || context->has_indirect_sgpr_access)
              return false;
            const InstDefUse def_use(*instruction);
            for (const MoiSgprRange &range : bootstrap_ranges) {
              const RegisterRef registers{RegClass::SGPR, range.base,
                                          static_cast<uint8_t>(range.width)};
              if (static_cast<uint32_t>(range.base) + range.width > context->current_sgpr_count ||
                  context->liveness->any_live_before(*instruction, registers) ||
                  def_use.defs.contains(registers) || def_use.uses.contains(registers)) {
                return false;
              }
            }
          }
          return saw_owner;
        };
        if (!std::ranges::all_of(anchors, anchor_admits_bootstrap)) {
          result.warnings.emplace_back(
              "ConSan MOI inline atomic dense relay scalar ABI is live at a call anchor");
          continue;
        }

        constexpr size_t kHostWords = kMoiLocalIndirectIslandWords + 1u;
        auto dense_host = find_moi_dense_relay_host(
            resource_state, old_text,
            {
                .owner_begin = owner_begin,
                .owner_end = owner_end,
                .host_word_count = kHostWords,
                .anchors = anchors,
                .preapplied_reserved_ranges = resource_state.reserved_ranges,
                .owner_proof = owner_proof,
                .bootstrap_ranges = bootstrap_ranges,
            },
            overlaps_existing);
        if (!dense_host) {
          result.warnings.emplace_back(
              "ConSan MOI inline atomic dense relay found no liveness-safe relocatable host");
          continue;
        }
        const uint64_t host_offset = dense_host->host_offset;
        const uint64_t island_offset = host_offset + sizeof(uint32_t);
        claimed_hosts.emplace_back(host_offset, host_offset + kHostWords * sizeof(uint32_t));

        constexpr uint64_t kDispatcherWordsPerRoute = 20u;
        const uint64_t dispatcher_offset = placement_planner.appended_end();
        const uint64_t dispatcher_bytes =
            (group.size() + 1u) * kDispatcherWordsPerRoute * sizeof(uint32_t);
        std::string reservation_error;
        if (!placement_planner.reserve_appended_prefix(dispatcher_bytes, &reservation_error)) {
          result.errors.emplace_back("ConSan MOI inline atomic dense dispatcher failed: " +
                                     reservation_error);
          return;
        }
        while (new_text.size() < placement_planner.appended_end())
          append_word_bytes(new_text, build_s_nop(0, arch));

        struct DenseBody {
          size_t planned_index = 0u;
          uint64_t offset = 0u;
          std::vector<uint32_t> words;
          uint32_t guest_instruction_offset = 0u;
        };
        std::vector<DenseBody> bodies;
        for (size_t index : group) {
          const PlannedInlineAtomic &planned = planned_candidates[index];
          const uint64_t cave_offset = placement_planner.appended_end();
          MoiOptions candidate_options = group_options;
          candidate_options.scratch_vgpr = planned.resources.base;
          if (planned.private_layout && planned.private_layout->dispatch_id_offset)
            candidate_options.moi_dispatch_id_sgpr = candidate_options.moi_inline_indirect_pc_sgpr;
          uint32_t guest_instruction_offset = 0u;
          auto cave_words = build_inline_atomic_ordering_cave_words(
              active_bytes, planned.candidate, planned.address_plan, candidate_options,
              planned.spill ? &*planned.spill : nullptr,
              planned.scalar_spill ? &*planned.scalar_spill : nullptr,
              planned.private_layout ? &*planned.private_layout : nullptr, arch,
              layout.inline_atomic_release_slots_offset, layout.inline_atomic_release_capacity,
              layout.inline_causal_snapshots_offset, layout.inline_causal_snapshot_capacity,
              layout.inline_acquired_epoch_token_slots_offset,
              layout.inline_acquired_epoch_token_capacity, cave_offset,
              planned.candidate.site.text_offset + planned.relocated_guest_size,
              guest_instruction_offset, result.errors, planned.trailing_guest_words);
          if (!cave_words)
            return;
          const auto placement = placement_planner.plan_indirect_appended(
              planned.candidate.site.text_offset, planned.relocated_guest_size,
              cave_words->size() * sizeof(uint32_t), &reservation_error);
          if (!placement || placement->body_offset != cave_offset ||
              new_text.size() != cave_offset) {
            result.errors.emplace_back("ConSan MOI inline atomic dense body placement failed");
            return;
          }
          append_words_bytes(new_text, *cave_words);
          bodies.push_back({index, cave_offset, std::move(*cave_words), guest_instruction_offset});
        }

        const uint64_t host_body_offset = placement_planner.appended_end();
        std::vector<uint32_t> host_words = std::move(dense_host->displaced_words);
        if (!append_moi_scc_preserving_indirect_jump(
                host_words, host_body_offset, host_offset + kHostWords * sizeof(uint32_t),
                jump_pc_sgpr, saved_scc_sgpr, /*capture_scc=*/true, arch) ||
            !placement_planner.reserve_appended_prefix(host_words.size() * sizeof(uint32_t),
                                                       &reservation_error) ||
            new_text.size() != host_body_offset) {
          result.errors.emplace_back("ConSan MOI inline atomic dense host body failed");
          return;
        }
        append_words_bytes(new_text, host_words);

        std::vector<uint32_t> island_words;
        if (key_encodes_scc)
          island_words.push_back(build_s_nop(0, arch));
        if (!append_moi_scc_preserving_indirect_jump(island_words, island_offset, dispatcher_offset,
                                                     jump_pc_sgpr, saved_scc_sgpr,
                                                     /*capture_scc=*/!key_encodes_scc, arch)) {
          result.errors.emplace_back("ConSan MOI inline atomic dense island failed");
          return;
        }
        const auto wait_pc = instrumentation::build_s_wait_indirect_pc0(arch);
        if (!wait_pc) {
          result.errors.emplace_back("ConSan MOI inline atomic dense island wait failed");
          return;
        }
        island_words.insert(island_words.end() - 1u, *wait_pc);
        if (island_words.size() != kMoiLocalIndirectIslandWords) {
          result.errors.emplace_back("ConSan MOI inline atomic dense island has invalid size");
          return;
        }
        std::memcpy(new_text.data() + island_offset, island_words.data(),
                    island_words.size() * sizeof(uint32_t));
        const std::optional<uint32_t> host_key =
            key_encodes_scc
                ? instrumentation::build_s_cselect_b32(key_sgpr, scalar_positive_inline_u32(1),
                                                       scalar_positive_inline_u32(0), arch)
                : std::optional<uint32_t>(
                      build_s_mov_b32(key_sgpr, scalar_positive_inline_u32(0), arch));
        if (!host_key) {
          result.errors.emplace_back("ConSan MOI inline atomic dense host key failed");
          return;
        }
        std::memcpy(new_text.data() + host_offset, &*host_key, sizeof(uint32_t));

        std::vector<uint32_t> dispatcher_words;
        if (key_encodes_scc) {
          dispatcher_words.push_back(
              build_s_lshr_b32(jump_pc_sgpr, key_sgpr, scalar_positive_inline_u32(1), arch));
        }
        std::optional<uint32_t> restore_tagged_scc;
        if (key_encodes_scc) {
          std::vector<uint32_t> restore_words;
          if (!consan_detail::append_restore_moi_scc_from_route_key(
                  restore_words, {.encoded_sgpr = key_sgpr, .normalize_encoded_sgpr = false},
                  *target) ||
              restore_words.size() != 1u) {
            result.errors.emplace_back(
                "ConSan MOI inline atomic dense router could not restore tagged SCC");
            return;
          }
          restore_tagged_scc = restore_words.front();
        }
        const auto append_target = [&](uint16_t route, uint64_t destination) {
          const auto compare = instrumentation::build_s_cmp_eq_u32(
              key_encodes_scc ? jump_pc_sgpr : key_sgpr, scalar_positive_inline_u32(route), arch);
          if (!compare)
            return false;
          dispatcher_words.push_back(*compare);
          std::vector<uint32_t> matched_words;
          if (restore_tagged_scc)
            matched_words.push_back(*restore_tagged_scc);
          const uint64_t matched_offset =
              dispatcher_offset + (dispatcher_words.size() + 1u) * sizeof(uint32_t);
          const uint64_t branch_offset = matched_offset + matched_words.size() * sizeof(uint32_t);
          const auto direct = compute_sopp_branch_simm16(branch_offset, destination);
          if (direct) {
            matched_words.push_back(build_s_branch(*direct, arch));
          } else if (!append_moi_scc_preserving_indirect_jump(
                         matched_words, matched_offset, destination, jump_pc_sgpr, saved_scc_sgpr,
                         /*capture_scc=*/false, arch)) {
            return false;
          } else if (restore_tagged_scc) {
            matched_words[matched_words.size() - 2u] = *restore_tagged_scc;
          }
          const auto skip = instrumentation::build_s_cbranch_scc0(
              static_cast<int16_t>(matched_words.size()), arch);
          if (!skip)
            return false;
          dispatcher_words.push_back(*skip);
          dispatcher_words.insert(dispatcher_words.end(), matched_words.begin(),
                                  matched_words.end());
          return true;
        };
        if (!append_target(/*route=*/0u, host_body_offset)) {
          result.errors.emplace_back("ConSan MOI inline atomic dense host route failed");
          return;
        }
        for (size_t body_index = 0; body_index < bodies.size(); ++body_index) {
          if (!append_target(static_cast<uint16_t>(body_index + 1u), bodies[body_index].offset)) {
            result.errors.emplace_back("ConSan MOI inline atomic dense body route failed");
            return;
          }
        }
        dispatcher_words.push_back(build_s_endpgm(arch));
        if (dispatcher_words.size() * sizeof(uint32_t) > dispatcher_bytes) {
          result.errors.emplace_back("ConSan MOI inline atomic dense dispatcher overflowed");
          return;
        }
        std::memcpy(new_text.data() + dispatcher_offset, dispatcher_words.data(),
                    dispatcher_words.size() * sizeof(uint32_t));

        ConSanPatchInfo host_info;
        host_info.kind = ConSanPatchKind::TrampolineMoiIndirectBranchIsland;
        host_info.anchor_offset = host_offset;
        host_info.original_size = kHostWords * sizeof(uint32_t);
        host_info.trampoline_offset = host_body_offset;
        host_info.trampoline_size = static_cast<uint32_t>(host_words.size() * sizeof(uint32_t));
        host_info.owner_descriptor_file_offsets = group_owners;
        patches.push_back(std::move(host_info));
        ConSanPatchInfo dispatcher_info;
        dispatcher_info.kind = ConSanPatchKind::TrampolineMoiIndirectBranchIsland;
        dispatcher_info.anchor_offset = anchors.front();
        dispatcher_info.trampoline_offset = dispatcher_offset;
        dispatcher_info.trampoline_size =
            static_cast<uint32_t>(dispatcher_words.size() * sizeof(uint32_t));
        dispatcher_info.moi_dense_entry_island_offset = island_offset;
        dispatcher_info.owner_descriptor_file_offsets = group_owners;
        patches.push_back(std::move(dispatcher_info));

        for (size_t body_index = 0; body_index < bodies.size(); ++body_index) {
          const DenseBody &body = bodies[body_index];
          const PlannedInlineAtomic &planned = planned_candidates[body.planned_index];
          const uint16_t route = static_cast<uint16_t>(body_index + 1u);
          const uint64_t branch_offset = planned.candidate.site.text_offset + sizeof(uint32_t);
          const auto branch = compute_sopp_branch_simm16(branch_offset, island_offset);
          const uint16_t tagged_route = static_cast<uint16_t>(2u * route);
          const std::optional<uint32_t> route_key =
              key_encodes_scc ? instrumentation::build_s_cselect_b32(
                                    key_sgpr, scalar_positive_inline_u32(tagged_route + 1u),
                                    scalar_positive_inline_u32(tagged_route), arch)
                              : std::optional<uint32_t>(build_s_mov_b32(
                                    key_sgpr, scalar_positive_inline_u32(route), arch));
          if (!branch || !route_key) {
            result.errors.emplace_back("ConSan MOI inline atomic dense anchor failed");
            return;
          }
          std::vector<uint32_t> anchor_words(planned.relocated_guest_size / sizeof(uint32_t),
                                             build_s_nop(0, arch));
          anchor_words[0] = *route_key;
          anchor_words[1] = build_s_branch(*branch, arch);
          std::memcpy(new_text.data() + planned.candidate.site.text_offset, anchor_words.data(),
                      planned.relocated_guest_size);

          ConSanPatchInfo info;
          info.kind = ConSanPatchKind::TrampolineMoiInlineAtomicOrdering;
          info.anchor_offset = planned.candidate.site.text_offset;
          info.trampoline_offset = body.offset;
          info.original_size = planned.relocated_guest_size;
          info.trampoline_size = static_cast<uint32_t>(body.words.size() * sizeof(uint32_t));
          info.relocated_guest_instruction_offset = body.offset + body.guest_instruction_offset;
          note_moi_probe_patch_info(info, planned);
          note_moi_persistent_vgpr_state(info, group_options, result.moi_operating_point);
          info.persistent_sgpr_state = group_options.moi_persistent_sgprs;
          if (!append_moi_atomic_lowering_commit(result, planned.candidate, info,
                                                 "inline atomic patch", lowering_commits))
            return;
          patches.push_back(std::move(info));
          routed_by_dense_relay[body.planned_index] = true;
        }
      }
    }
  }

  for (size_t planned_index = 0; planned_index < planned_candidates.size(); ++planned_index) {
    if (routed_by_dense_relay[planned_index])
      continue;
    const PlannedInlineAtomic &planned = planned_candidates[planned_index];
    const MoiAtomicEvidenceSitePlan &candidate = planned.candidate;
    const ConSanAtomicSite &site = candidate.site;

    const uint64_t cave_text_offset = placement_planner.appended_end();
    const uint64_t return_text_offset = site.text_offset + planned.relocated_guest_size;
    MoiOptions candidate_options = options;
    candidate_options.scratch_vgpr = planned.resources.base;
    if (!apply_moi_transient_sgpr_assignment(candidate_options, candidate_options,
                                             result.moi_operating_point,
                                             planned.resources.owner_descriptor_file_offsets)) {
      result.errors.emplace_back(
          "ConSan MOI inline atomic patch lost its transient SGPR assignment");
      return;
    }
    if (!result.moi_operating_point.owner_persistent_vgprs.empty() &&
        !apply_moi_persistent_vgpr_assignment(candidate_options, result.moi_operating_point,
                                              planned.resources.owner_descriptor_file_offsets)) {
      result.errors.emplace_back(
          "ConSan MOI inline atomic patch lost its persistent VGPR assignment");
      return;
    }
    if (planned.private_layout && planned.private_layout->dispatch_id_offset)
      candidate_options.moi_dispatch_id_sgpr = candidate_options.moi_inline_indirect_pc_sgpr;
    uint32_t guest_instruction_offset = 0;
    auto cave_words = build_inline_atomic_ordering_cave_words(
        active_bytes, candidate, planned.address_plan, candidate_options,
        planned.spill ? &*planned.spill : nullptr,
        planned.scalar_spill ? &*planned.scalar_spill : nullptr,
        planned.private_layout ? &*planned.private_layout : nullptr, arch,
        layout.inline_atomic_release_slots_offset, layout.inline_atomic_release_capacity,
        layout.inline_causal_snapshots_offset, layout.inline_causal_snapshot_capacity,
        layout.inline_acquired_epoch_token_slots_offset,
        layout.inline_acquired_epoch_token_capacity, cave_text_offset, return_text_offset,
        guest_instruction_offset, result.errors, planned.trailing_guest_words);
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
        site.text_offset, planned.relocated_guest_size, cave_words->size() * sizeof(uint32_t),
        &placement_error);
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
    std::vector<uint32_t> anchor_words(planned.relocated_guest_size / sizeof(uint32_t),
                                       build_s_nop(0, arch));
    if (anchor_words.empty()) {
      result.errors.emplace_back("ConSan MOI inline atomic patch empty anchor");
      return;
    }
    anchor_words.front() = build_s_branch(*fwd, arch);
    std::memcpy(new_text.data() + site.text_offset, anchor_words.data(),
                anchor_words.size() * sizeof(uint32_t));
    if (indirect_entry_island &&
        !emit_moi_local_indirect_entry_island(
            new_text, *indirect_entry_island, placement->body_offset, site.text_offset,
            static_cast<const ConSanRequest &>(candidate_options),
            static_cast<const ConSanMoiOperatingPoint &>(candidate_options),
            planned.resources.owner_descriptor_file_offsets, arch, patches, result.errors,
            "ConSan MOI inline atomic patch")) {
      return;
    }
    append_words_bytes(new_text, *cave_words);

    ConSanPatchInfo info;
    info.kind = ConSanPatchKind::TrampolineMoiInlineAtomicOrdering;
    info.anchor_offset = site.text_offset;
    info.trampoline_offset = cave_text_offset;
    info.original_size = planned.relocated_guest_size;
    info.trampoline_size = static_cast<uint32_t>(cave_words->size() * sizeof(uint32_t));
    info.relocated_guest_instruction_offset = cave_text_offset + guest_instruction_offset;
    note_moi_probe_patch_info(info, planned);
    note_moi_persistent_vgpr_state(info, candidate_options, result.moi_operating_point);
    info.persistent_sgpr_state = candidate_options.moi_persistent_sgprs;
    if (!append_moi_atomic_lowering_commit(result, candidate, info, "inline atomic patch",
                                           lowering_commits))
      return;
    patches.push_back(info);
  }

  if (patches.empty()) {
    result.warnings.emplace_back("ConSan MOI inline atomic patch found no patchable atomics");
    return;
  }
  if (!replace_consan_text(patcher, new_text, options, "MOI inline atomic patch", result)) {
    return;
  }
  if (!publish_moi_sync_lowering_commits(result, std::move(lowering_commits),
                                         "inline atomic patch"))
    return;
  result.replacement = std::move(patcher).emit();
  result.patches.insert(result.patches.end(), patches.begin(), patches.end());
  result.mark_modified();
}

[[nodiscard]] std::optional<ConSanMoiSampledSyncRole>
