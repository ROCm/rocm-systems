void try_apply_inline_atomic_ordering_patch(std::span<const uint8_t> bytes,
                                            const MoiOptions &options, rj_code_arch_t arch,
                                            ConSanTransformArtifacts &result) {
  if (!options.moi_track_atomics || options.moi_engine != ConSanMoiEngine::InlineShadow)
    return;
  if (!consan_is_capability_arch(arch)) {
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
      (!options.moi_persistent_sgprs.owner || !options.moi_persistent_sgprs.epoch) &&
      !options.automatic_moi_private_epoch) {
    result.warnings.emplace_back("ConSan MOI inline atomic patch requires "
                                 "RJ_CONSAN_MOI_OWNER_VGPR and RJ_CONSAN_MOI_EPOCH_VGPR");
    return;
  }
  if (!validate_inline_atomic_exec_save_sgpr(options, result.warnings))
    return;

  std::vector<MoiAtomicEvidenceSitePlan> candidates = build_moi_atomic_evidence_site_plans(
      result, ConSanProbeIntentKind::ExactAtomicOrdering, result.errors);
  if (!result.errors.empty())
    return;
  candidates.erase(
      std::remove_if(candidates.begin(), candidates.end(),
                     [&](const auto &candidate) {
                       const ConSanMoiInlineAtomicSupport support =
                           classify_consan_moi_inline_atomic_support(
                               candidate.site, candidate.event_kind, arch, candidate.is_rmw);
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
  MoiResourcePlanningState resource_state(bytes, arch, result);
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
    const ConSanCandidateResourcePlan *existing_plan =
        resource_plan_for_site(result, ConSanResourceSiteKind::Atomic, site.text_offset);
    const uint16_t scratch_count = inline_shadow_atomic_scratch_count(arch);
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
    const auto resources =
        resolve_moi_scratch_plan(plan, options, result.moi_operating_point, scratch_count);
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
    MoiAtomicEvidenceSitePlan planned_candidate = candidate;
    if (!planned_candidate.kernel_descriptor_file_offset) {
      planned_candidate.kernel_descriptor_file_offset =
          common_moi_record_owner_descriptor(bytes, *resources, arch, result.warnings);
      if (!planned_candidate.kernel_descriptor_file_offset)
        continue;
    }
    MoiOptions candidate_options = options;
    candidate_options.scratch_vgpr = resources->base;
    if (!apply_moi_transient_sgpr_assignment(candidate_options, result.moi_operating_point,
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
          result, *resources, arch, result.warnings,
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
    if (!apply_moi_transient_sgpr_assignment(candidate_options, result.moi_operating_point,
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
    note_moi_sgpr_requirements(scalar_requirements, planned.resources, candidate_options, arch);
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
    if (!neutralize_atomic_scalar_clause(new_text, planned.candidate, arch, patches, result.errors))
      return;
  }
  std::vector<bool> routed_by_dense_relay(planned_candidates.size(), false);

  // A large generated CDNA kernel can strand an early atomic beyond SOPP
  // reach of both the pristine-end island bank and every appended body.  A
  // liveness-proven relocated host supplies one shared in-kernel island.  Its
  // scalar key identifies the normal host path or one of the atomic bodies.
  if (consan_uses_gfx9_cdna_encoding(arch)) {
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
        if (!apply_moi_transient_sgpr_assignment(group_options, result.moi_operating_point,
                                                 group_owners)) {
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
        const auto scalar_abi = moi_inline_dense_router_scalar_abi(group_options, arch);
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
            group_options, MoiRelocatableHostScalarState::AccessRouter, arch);
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
        const uint16_t bitcmp_opcode =
            arch == ROCJITSU_CODE_ARCH_CDNA3 ? cdna3::kSBitcmp1B32Sopc : cdna4::kSBitcmp1B32Sopc;
        const uint32_t restore_tagged_scc =
            build_sopc_encoding(arch, bitcmp_opcode, key_sgpr, scalar_positive_inline_u32(0));
        const auto append_target = [&](uint16_t route, uint64_t target) {
          const auto compare = instrumentation::build_s_cmp_eq_u32(
              key_encodes_scc ? jump_pc_sgpr : key_sgpr, scalar_positive_inline_u32(route), arch);
          if (!compare)
            return false;
          dispatcher_words.push_back(*compare);
          std::vector<uint32_t> matched_words;
          if (key_encodes_scc)
            matched_words.push_back(restore_tagged_scc);
          const uint64_t matched_offset =
              dispatcher_offset + (dispatcher_words.size() + 1u) * sizeof(uint32_t);
          const uint64_t branch_offset = matched_offset + matched_words.size() * sizeof(uint32_t);
          const auto direct = compute_sopp_branch_simm16(branch_offset, target);
          if (direct) {
            matched_words.push_back(build_s_branch(*direct, arch));
          } else if (!append_moi_scc_preserving_indirect_jump(matched_words, matched_offset, target,
                                                              jump_pc_sgpr, saved_scc_sgpr,
                                                              /*capture_scc=*/false, arch)) {
            return false;
          } else if (key_encodes_scc) {
            matched_words[matched_words.size() - 2u] = restore_tagged_scc;
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
    if (!apply_moi_transient_sgpr_assignment(candidate_options, result.moi_operating_point,
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
            candidate_options, planned.resources.owner_descriptor_file_offsets, arch, patches,
            result.errors, "ConSan MOI inline atomic patch")) {
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
    patches.push_back(info);
  }

  if (patches.empty()) {
    result.warnings.emplace_back("ConSan MOI inline atomic patch found no patchable atomics");
    return;
  }
  if (!replace_consan_text(patcher, new_text, options, "MOI inline atomic patch", result)) {
    return;
  }
  result.replacement = std::move(patcher).emit();
  result.patches.insert(result.patches.end(), patches.begin(), patches.end());
  result.mark_modified();
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_atomic_record_cave_words(
    std::span<const uint8_t> bytes, const MoiAtomicEvidenceSitePlan &candidate,
    const ConSanMoiAtomicAddressPlan &address_plan, const MoiOptions &options,
    const VgprSpillSequence *spill, const InlineSgprSpillSequence *scalar_spill,
    rj_code_arch_t arch, uint32_t record_index, uint32_t atomic_record_capacity,
    size_t atomic_records_offset, uint64_t cave_text_offset, uint64_t return_text_offset,
    bool already_runtime_workgroup_gated,
    const std::optional<MoiWorkitemOwnerDerivationPlan> &derived_owner,
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
  std::vector<uint32_t> derived_owner_words;
  if (!options.moi_owner_vgpr && !options.moi_persistent_sgprs.owner) {
    if (!derived_owner) {
      errors.emplace_back("ConSan MOI atomic record patch requires a planned owner derivation");
      return std::nullopt;
    }
    const uint16_t value_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 4u);
    const auto owner = build_moi_workitem_owner_derivation(*derived_owner, value_vgpr, arch,
                                                           "atomic record patch", errors);
    if (!owner)
      return std::nullopt;
    derived_owner_vgpr = owner->vgpr;
    derived_owner_words = owner->words;
  }

  const auto persistent_workgroup_sources =
      record_replay_persistent_workgroup_sources(options, arch);
  if (!persistent_workgroup_sources) {
    errors.emplace_back(
        "ConSan MOI Record/Replay atomic requires one exact entry-captured workgroup tuple");
    return std::nullopt;
  }
  const ConSanMoiWorkgroupSources workgroup_sources = *persistent_workgroup_sources;

  if (candidate.site.file_offset > bytes.size() ||
      candidate.site.size > bytes.size() - candidate.site.file_offset) {
    errors.emplace_back("ConSan MOI atomic record patch site exceeds ELF bytes");
    return std::nullopt;
  }

  std::vector<uint32_t> words;
  const bool runtime_workgroup_gate =
      options.moi_runtime_sample_stride > 1u && !already_runtime_workgroup_gated &&
      !options.automatic_moi_record_replay_sgpr_spill &&
      moi_has_probe_entry_runtime_workgroup_gate(options, workgroup_sources, arch);
  if (runtime_workgroup_gate) {
    auto gate = build_moi_runtime_workgroup_gate_call_words(
        bytes.subspan(candidate.site.file_offset, candidate.site.size), options, workgroup_sources,
        cave_text_offset, return_text_offset,
        moi_runtime_workgroup_gate_reserved_words(
            candidate.site.size, workgroup_sources.cluster_workgroup_id.has_value()),
        arch);
    if (!gate) {
      errors.emplace_back(
          "ConSan MOI atomic record patch could not encode its runtime workgroup gate");
      return std::nullopt;
    }
    words = std::move(*gate);
  }
  words.reserve(words.size() + candidate.site.size / sizeof(uint32_t) + 96u +
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
  const bool is_compare_exchange = consan_atomic_is_compare_exchange(candidate.site);
  std::vector<uint32_t> guest_atomic_words;
  guest_atomic_words.reserve(candidate.site.size / sizeof(uint32_t));
  for (uint64_t offset = 0; offset < candidate.site.size; offset += sizeof(uint32_t)) {
    uint32_t word = 0;
    std::memcpy(&word, bytes.data() + candidate.site.file_offset + offset, sizeof(word));
    guest_atomic_words.push_back(word);
  }

  const uint64_t base = *options.moi_report_buffer_address;
  const uint16_t recorded_address_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 5u);
  words.push_back(build_v_mov_b32_e32(recorded_address_vgpr,
                                      vector_source_vgpr(address_plan.result_address_vgpr), arch));
  words.push_back(build_v_mov_b32_e32(
      static_cast<uint16_t>(recorded_address_vgpr + 1u),
      vector_source_vgpr(static_cast<uint16_t>(address_plan.result_address_vgpr + 1u)), arch));

  const bool reserve_event_before_guest =
      is_compare_exchange && (candidate.event_kind == ConSanMoiAtomicEventKind::Release ||
                              candidate.event_kind == ConSanMoiAtomicEventKind::AcquireRelease);
  const uint16_t reserved_event_index_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 7u);
  if (reserve_event_before_guest) {
    if (!append_save_moi_special_state(words, moi_special_state_sgprs(options), arch)) {
      errors.emplace_back(
          "ConSan MOI release CAS could not preserve VCC/SCC before event reservation");
      return std::nullopt;
    }
    const uint16_t lane_rank_vgpr = *options.scratch_vgpr;
    const auto save_active_exec =
        instrumentation::build_s_mov_b64(*options.moi_exec_save_sgpr, kRdna4ExecLo, arch);
    const auto mbcnt_lo = instrumentation::build_v_mbcnt_lo_u32_b32(
        lane_rank_vgpr, *options.moi_exec_save_sgpr, scalar_positive_inline_u32(0), arch);
    const auto mbcnt_hi = instrumentation::build_v_mbcnt_hi_u32_b32(
        lane_rank_vgpr, static_cast<uint16_t>(*options.moi_exec_save_sgpr + 1u),
        vector_source_vgpr(lane_rank_vgpr), arch);
    const auto first_active_lane = instrumentation::build_v_cmp_eq_u32_vcc(
        scalar_positive_inline_u32(0), lane_rank_vgpr, arch);
    const auto narrow_exec =
        instrumentation::build_s_and_saveexec_b64(*options.moi_exec_save_sgpr, kRdna4VccLo, arch);
    const auto saved_exec_wait = instrumentation::build_salu_to_valu_dependency_wait(arch);
    const auto restore_exec =
        instrumentation::build_s_mov_b64(kRdna4ExecLo, *options.moi_exec_save_sgpr, arch);
    if (!save_active_exec || !mbcnt_lo || !mbcnt_hi || !first_active_lane || !narrow_exec ||
        !saved_exec_wait || !restore_exec) {
      errors.emplace_back("ConSan MOI release CAS could not select its event publisher");
      return std::nullopt;
    }
    words.push_back(*save_active_exec);
    words.push_back(*saved_exec_wait);
    words.insert(words.end(), mbcnt_lo->begin(), mbcnt_lo->end());
    words.insert(words.end(), mbcnt_hi->begin(), mbcnt_hi->end());
    words.push_back(*first_active_lane);
    words.push_back(*narrow_exec);
    if (!append_atomic_fetch_add_one_u32(words,
                                         base + offsetof(ConSanMoiReportHeader, event_counter),
                                         reserved_event_index_vgpr, *options.scratch_vgpr, arch)) {
      errors.emplace_back("ConSan MOI release CAS could not reserve its causal event index");
      return std::nullopt;
    }
    words.push_back(*restore_exec);
    if (!append_restore_moi_special_state(words, moi_special_state_sgprs(options), arch)) {
      errors.emplace_back(
          "ConSan MOI release CAS could not restore VCC/SCC after event reservation");
      return std::nullopt;
    }
  }

  const bool record_before_guest_release =
      candidate.event_kind == ConSanMoiAtomicEventKind::Release && !is_compare_exchange;
  if (!record_before_guest_release) {
    guest_instruction_offset = static_cast<uint32_t>(words.size() * sizeof(uint32_t));
    words.insert(words.end(), guest_atomic_words.begin(), guest_atomic_words.end());
    if (!append_moi_flat_load_wait(words, arch))
      return std::nullopt;
  }

  if (!append_save_moi_special_state(words, moi_special_state_sgprs(options), arch)) {
    errors.emplace_back("ConSan MOI atomic record patch could not save VCC/SCC");
    return std::nullopt;
  }

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
  const auto saved_exec_wait = instrumentation::build_salu_to_valu_dependency_wait(arch);
  if (!save_active_exec || !saved_exec_wait || !mbcnt_lo || !mbcnt_hi || !first_active_lane ||
      !save_exec) {
    errors.emplace_back("ConSan MOI atomic record patch could not select a wave publisher");
    return std::nullopt;
  }
  words.push_back(*save_active_exec);
  words.push_back(*saved_exec_wait);
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
    if (!append_dynamic_record_store_u32_vgpr(
            record_words, kAtomicRecordLayout,
            atomic_record_base + offsetof(ConSanMoiAtomicRecord, success_lane_mask),
            static_cast<uint16_t>(*options.scratch_vgpr + 3u), slot_vgpr, *options.scratch_vgpr,
            arch) ||
        !append_dynamic_record_store_u32_vgpr(
            record_words, kAtomicRecordLayout,
            atomic_record_base + offsetof(ConSanMoiAtomicRecord, success_lane_mask) +
                sizeof(uint32_t),
            static_cast<uint16_t>(*options.scratch_vgpr + 4u), slot_vgpr, *options.scratch_vgpr,
            arch)) {
      errors.emplace_back("ConSan MOI atomic record patch could not preserve CAS outcome");
      return std::nullopt;
    }
  }
  record_words.insert(record_words.end(), derived_owner_words.begin(), derived_owner_words.end());
  const auto append_record_event_index = [&]() {
    if (reserve_event_before_guest) {
      return append_dynamic_record_store_u32_vgpr(
          record_words, kAtomicRecordLayout,
          atomic_record_base + offsetof(ConSanMoiAtomicRecord, event_index),
          reserved_event_index_vgpr, slot_vgpr, *options.scratch_vgpr, arch);
    }
    return append_dynamic_record_event_index_store(
        record_words, kAtomicRecordLayout, base + offsetof(ConSanMoiReportHeader, event_counter),
        atomic_record_base + offsetof(ConSanMoiAtomicRecord, event_index), slot_vgpr,
        *options.scratch_vgpr, arch);
  };
  // Probe-local owner derivation uses the record-emitter temporary. Preserve
  // it before reserving the event index, which reuses that VGPR.
  if ((derived_owner_vgpr && !append_dynamic_record_store_u32_vgpr(
                                 record_words, kAtomicRecordLayout,
                                 atomic_record_base + offsetof(ConSanMoiAtomicRecord, owner_id),
                                 *derived_owner_vgpr, slot_vgpr, *options.scratch_vgpr, arch)) ||
      (options.moi_persistent_sgprs.owner &&
       !append_dynamic_record_store_u32_scalar_src(
           record_words, kAtomicRecordLayout,
           atomic_record_base + offsetof(ConSanMoiAtomicRecord, owner_id),
           *options.moi_persistent_sgprs.owner, slot_vgpr, *options.scratch_vgpr, arch)) ||
      !append_dynamic_record_store_moi_report_dispatch_id_pair(
          record_words, kAtomicRecordLayout,
          atomic_record_base + offsetof(ConSanMoiAtomicRecord, generation), options, slot_vgpr,
          *options.scratch_vgpr, arch, ConSanMoiLiteralDispatchIdPolicy::AnyArchitecture) ||
      !append_record_event_index() ||
      !append_dynamic_record_store_workgroup_source(
          record_words, kAtomicRecordLayout,
          atomic_record_base + offsetof(ConSanMoiAtomicRecord, workgroup_x), workgroup_sources.x,
          slot_vgpr, *options.scratch_vgpr, arch) ||
      !append_dynamic_record_store_workgroup_source(
          record_words, kAtomicRecordLayout,
          atomic_record_base + offsetof(ConSanMoiAtomicRecord, workgroup_y), workgroup_sources.y,
          slot_vgpr, *options.scratch_vgpr, arch) ||
      !append_dynamic_record_store_workgroup_source(
          record_words, kAtomicRecordLayout,
          atomic_record_base + offsetof(ConSanMoiAtomicRecord, workgroup_z), workgroup_sources.z,
          slot_vgpr, *options.scratch_vgpr, arch) ||
      (options.moi_owner_vgpr &&
       !append_dynamic_record_store_u32_vgpr(
           record_words, kAtomicRecordLayout,
           atomic_record_base + offsetof(ConSanMoiAtomicRecord, owner_id), *options.moi_owner_vgpr,
           slot_vgpr, *options.scratch_vgpr, arch)) ||
      (options.moi_epoch_vgpr &&
       !append_dynamic_record_store_u32_vgpr(
           record_words, kAtomicRecordLayout,
           atomic_record_base + offsetof(ConSanMoiAtomicRecord, epoch), *options.moi_epoch_vgpr,
           slot_vgpr, *options.scratch_vgpr, arch)) ||
      (options.moi_persistent_sgprs.epoch &&
       !append_dynamic_record_store_u32_scalar_src(
           record_words, kAtomicRecordLayout,
           atomic_record_base + offsetof(ConSanMoiAtomicRecord, epoch),
           *options.moi_persistent_sgprs.epoch, slot_vgpr, *options.scratch_vgpr, arch)) ||
      !append_dynamic_record_store_u32_vgpr(
          record_words, kAtomicRecordLayout,
          atomic_record_base + offsetof(ConSanMoiAtomicRecord, atomic_address),
          recorded_address_vgpr, slot_vgpr, *options.scratch_vgpr, arch) ||
      !append_dynamic_record_store_u32_vgpr(
          record_words, kAtomicRecordLayout,
          atomic_record_base + offsetof(ConSanMoiAtomicRecord, atomic_address) + sizeof(uint32_t),
          static_cast<uint16_t>(recorded_address_vgpr + 1u), slot_vgpr, *options.scratch_vgpr,
          arch) ||
      !append_dynamic_record_store_u32_literal(
          record_words, kAtomicRecordLayout,
          atomic_record_base + offsetof(ConSanMoiAtomicRecord, instruction_offset),
          static_cast<uint32_t>(candidate.site.text_offset), slot_vgpr, *options.scratch_vgpr,
          arch) ||
      !append_dynamic_record_store_u32_literal(
          record_words, kAtomicRecordLayout,
          atomic_record_base + offsetof(ConSanMoiAtomicRecord, kind),
          static_cast<uint32_t>(candidate.event_kind), slot_vgpr, *options.scratch_vgpr, arch) ||
      !append_dynamic_record_store_u32_literal(
          record_words, kAtomicRecordLayout,
          atomic_record_base + offsetof(ConSanMoiAtomicRecord, scope), *candidate.site.raw_scope,
          slot_vgpr, *options.scratch_vgpr, arch) ||
      !append_dynamic_record_store_u32_literal(
          record_words, kAtomicRecordLayout,
          atomic_record_base + offsetof(ConSanMoiAtomicRecord, semantics),
          candidate.site.raw_th.value_or(0u), slot_vgpr, *options.scratch_vgpr, arch) ||
      !append_dynamic_record_store_u32_literal(
          record_words, kAtomicRecordLayout,
          atomic_record_base + offsetof(ConSanMoiAtomicRecord, operation),
          static_cast<uint32_t>(atomic_operation), slot_vgpr, *options.scratch_vgpr, arch) ||
      !append_dynamic_record_store_u32_literal(
          record_words, kAtomicRecordLayout,
          atomic_record_base + offsetof(ConSanMoiAtomicRecord, outcome),
          static_cast<uint32_t>(atomic_outcome), slot_vgpr, *options.scratch_vgpr, arch) ||
      (is_compare_exchange
           ? (!append_dynamic_record_store_u32_scalar_src(
                  record_words, kAtomicRecordLayout,
                  atomic_record_base + offsetof(ConSanMoiAtomicRecord, lane_mask),
                  *options.moi_exec_save_sgpr, slot_vgpr, *options.scratch_vgpr, arch) ||
              !append_dynamic_record_store_u32_scalar_src(
                  record_words, kAtomicRecordLayout,
                  atomic_record_base + offsetof(ConSanMoiAtomicRecord, lane_mask) +
                      sizeof(uint32_t),
                  static_cast<uint16_t>(*options.moi_exec_save_sgpr + 1u), slot_vgpr,
                  *options.scratch_vgpr, arch))
           : false)) {
    errors.emplace_back("ConSan MOI atomic record patch could not encode record stores");
    return std::nullopt;
  }

  const auto wait_store = instrumentation::build_s_wait_global_store0(arch);
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

  if (!append_restore_moi_special_state(words, moi_special_state_sgprs(options), arch)) {
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
sampled_atomic_role(ConSanMoiAtomicEventKind kind, bool is_rmw) {
  switch (kind) {
  case ConSanMoiAtomicEventKind::Release:
    return is_rmw ? ConSanMoiSampledSyncRole::RmwRelease : ConSanMoiSampledSyncRole::Release;
  case ConSanMoiAtomicEventKind::Acquire:
    return is_rmw ? ConSanMoiSampledSyncRole::RmwAcquire : ConSanMoiSampledSyncRole::Acquire;
  case ConSanMoiAtomicEventKind::AcquireRelease:
    return is_rmw ? std::optional(ConSanMoiSampledSyncRole::RmwAcquireRelease) : std::nullopt;
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

[[nodiscard]] bool sampled_atomic_guest_preserves_address(const ConSanAtomicSite &site,
                                                          bool is_rmw = true) {
  if (!site.addr_vgpr || !site.dst_vgpr)
    return true;
  const uint16_t address_count =
      site.mnemonic.starts_with("buffer_") || site.mnemonic.starts_with("ds_") ? 1u : 2u;
  const bool ordinary_load = !is_rmw && (site.mnemonic.starts_with("flat_load") ||
                                         site.mnemonic.starts_with("global_load"));
  const uint16_t destination_count =
      site.returns_old_value.value_or(false) || ordinary_load
          ? static_cast<uint16_t>(std::max<uint32_t>(site.width_bits / 32u, 1u))
          : 0u;
  return destination_count == 0u ||
         !range_overlaps(*site.addr_vgpr, address_count, *site.dst_vgpr, destination_count);
}

[[nodiscard]] bool sampled_atomic_spill_overlaps_guest_operands(const VgprSpillSequence &spill,
                                                                const ConSanAtomicSite &site) {
  const auto overlaps = [&](std::optional<uint16_t> base, uint16_t count) {
    return base && range_overlaps(spill.vgpr_base, spill.vgpr_count, *base, count);
  };
  const bool is_cas = consan_atomic_is_compare_exchange(site);
  const uint16_t value_count = static_cast<uint16_t>(std::max<uint32_t>(site.width_bits / 32u, 1u));
  const uint16_t address_count =
      site.mnemonic.starts_with("buffer_") || site.mnemonic.starts_with("ds_") ? 1u : 2u;
  return overlaps(site.addr_vgpr, address_count) ||
         overlaps(site.data_vgpr, static_cast<uint16_t>(value_count * (is_cas ? 2u : 1u))) ||
         overlaps(site.dst_vgpr, value_count);
}

[[nodiscard]] bool append_sampled_atomic_guest(std::vector<uint32_t> &words,
                                               std::span<const uint8_t> bytes,
                                               const MoiAtomicEvidenceSitePlan &candidate,
                                               rj_code_arch_t arch,
                                               uint32_t *guest_instruction_offset) {
  if (guest_instruction_offset)
    *guest_instruction_offset = static_cast<uint32_t>(words.size() * sizeof(uint32_t));
  for (uint64_t offset = 0; offset < candidate.site.size; offset += sizeof(uint32_t)) {
    uint32_t word = 0;
    std::memcpy(&word, bytes.data() + candidate.site.file_offset + offset, sizeof(word));
    words.push_back(word);
  }
  // A non-RMW candidate carries its compiler-emitted wait/cache suffix. Do
  // not splice a generic atomic wait into that target-native sequence.
  return !candidate.is_rmw || append_moi_flat_load_wait(words, arch);
}

struct SampledAtomicPreludeState {
  std::optional<uint16_t> cas_compare_vgpr;
  std::optional<uint16_t> cas_result_vgpr;
};

// Spill builders always produce complete metadata. Keep this check defensive
// for internal callers that construct a sequence directly.
[[nodiscard]] bool validate_sampled_atomic_spill_metadata(const VgprSpillSequence &spill,
                                                          std::vector<std::string> &errors) {
  if (spill.has_complete_slot_metadata())
    return true;
  errors.emplace_back("ConSan MOI sampled atomic spill has incomplete slot metadata");
  return false;
}

/// @pre @p spill has complete slot metadata.
[[nodiscard]] bool append_sampled_atomic_snapshot_word(std::vector<uint32_t> &words,
                                                       const VgprSpillSequence &spill,
                                                       uint16_t destination, uint16_t source,
                                                       rj_code_arch_t arch,
                                                       bool &loaded_private_state) {
  const uint32_t spill_end = static_cast<uint32_t>(spill.vgpr_base) + spill.vgpr_count;
  if (source >= spill.vgpr_base && source < spill_end) {
    const uint16_t slot = static_cast<uint16_t>(source - spill.vgpr_base);
    assert(slot < spill.slot_offsets.size());
    // Spill-range sources may have been overwritten by address or evidence
    // scratch, so reload their authoritative post-guest value from private.
    const auto load =
        instrumentation::build_private_load_b32(destination, spill.slot_offsets[slot], arch);
    if (!load)
      return false;
    words.insert(words.end(), load->begin(), load->end());
    loaded_private_state = true;
    return true;
  }
  words.push_back(build_v_mov_b32_e32(destination, vector_source_vgpr(source), arch));
  return true;
}

[[nodiscard]] bool append_sampled_atomic_snapshot_wait(std::vector<uint32_t> &words,
                                                       bool loaded_private_state,
                                                       rj_code_arch_t arch) {
  if (!loaded_private_state)
    return true;
  const auto wait = instrumentation::build_s_wait_private_load0(arch);
  if (!wait)
    return false;
  words.push_back(*wait);
  return true;
}

[[nodiscard]] bool append_sampled_atomic_cas_snapshot(std::vector<uint32_t> &words,
                                                      const VgprSpillSequence &spill,
                                                      uint16_t compare_vgpr, uint16_t result_vgpr,
                                                      uint16_t saved_compare, uint16_t saved_result,
                                                      rj_code_arch_t arch,
                                                      std::vector<std::string> &errors) {
  if (!validate_sampled_atomic_spill_metadata(spill, errors))
    return false;
  bool loaded_private_state = false;
  if (!append_sampled_atomic_snapshot_word(words, spill, saved_compare, compare_vgpr, arch,
                                           loaded_private_state) ||
      !append_sampled_atomic_snapshot_word(words, spill, saved_result, result_vgpr, arch,
                                           loaded_private_state)) {
    errors.emplace_back("ConSan MOI sampled atomic could not preserve CAS outcome evidence");
    return false;
  }
  if (!append_sampled_atomic_snapshot_wait(words, loaded_private_state, arch)) {
    errors.emplace_back("ConSan MOI sampled atomic could not wait for CAS outcome evidence");
    return false;
  }
  return true;
}

[[nodiscard]] bool
append_sampled_atomic_address_snapshot(std::vector<uint32_t> &words, const VgprSpillSequence &spill,
                                       uint16_t source_address, uint16_t saved_address,
                                       rj_code_arch_t arch, std::vector<std::string> &errors) {
  if (!validate_sampled_atomic_spill_metadata(spill, errors))
    return false;
  bool loaded_private_state = false;
  if (!append_sampled_atomic_snapshot_word(words, spill, saved_address, source_address, arch,
                                           loaded_private_state) ||
      !append_sampled_atomic_snapshot_word(words, spill, static_cast<uint16_t>(saved_address + 1u),
                                           static_cast<uint16_t>(source_address + 1u), arch,
                                           loaded_private_state)) {
    errors.emplace_back("ConSan MOI sampled atomic could not preserve its guest address");
    return false;
  }
  if (!append_sampled_atomic_snapshot_wait(words, loaded_private_state, arch)) {
    errors.emplace_back("ConSan MOI sampled atomic could not wait for its guest address");
    return false;
  }
  return true;
}

[[nodiscard]] bool append_sampled_atomic_prelude(
    std::vector<uint32_t> &words, std::span<const uint8_t> bytes,
    const MoiAtomicEvidenceSitePlan &candidate, const ConSanMoiAtomicAddressPlan &address_plan,
    const MoiOptions &options, const VgprSpillSequence *spill,
    const InlineSgprSpillSequence *scalar_spill, const MoiPrivateEpochLayout *private_layout,
    rj_code_arch_t arch, uint16_t saved_address, bool defer_guest, SampledAtomicPreludeState &state,
    std::vector<std::string> &errors, uint32_t *guest_instruction_offset,
    std::span<const uint32_t> trailing_guest_words) {
  const bool is_cas = consan_atomic_is_compare_exchange(candidate.site);
  if (is_cas) {
    assert(candidate.site.data_vgpr && candidate.site.dst_vgpr);
    state.cas_compare_vgpr = static_cast<uint16_t>(*candidate.site.data_vgpr + 1u);
    state.cas_result_vgpr = *candidate.site.dst_vgpr;
  }
  const bool guest_first =
      spill && sampled_atomic_spill_overlaps_guest_operands(*spill, candidate.site);
  assert(!defer_guest || !guest_first);
  const bool guest_preserves_address =
      sampled_atomic_guest_preserves_address(candidate.site, candidate.is_rmw);
  assert(!guest_first || guest_preserves_address);
  if (guest_first && !guest_preserves_address) {
    errors.emplace_back("ConSan MOI sampled atomic overlap spill cannot preserve guest address");
    return false;
  }
  if (guest_first &&
      !append_sampled_atomic_guest(words, bytes, candidate, arch, guest_instruction_offset))
    return false;
  if (guest_first)
    words.insert(words.end(), trailing_guest_words.begin(), trailing_guest_words.end());
  if (spill)
    words.insert(words.end(), spill->save_words.begin(), spill->save_words.end());
  if (scalar_spill)
    words.insert(words.end(), scalar_spill->save_words.begin(), scalar_spill->save_words.end());
  if (address_plan.requires_materialization()) {
    const auto special = moi_special_state_sgprs(options);
    if (!special)
      return false;
    const auto materialize = build_consan_moi_atomic_address_materialization(
        address_plan, special->vcc_save_sgpr, special->scc_save_sgpr, arch);
    if (!materialize)
      return false;
    words.insert(words.end(), materialize->begin(), materialize->end());
  }
  if (guest_first) {
    if (address_plan.requires_materialization()) {
      assert(address_plan.result_address_vgpr == saved_address);
      if (address_plan.result_address_vgpr != saved_address) {
        errors.emplace_back(
            "ConSan MOI sampled atomic materialized address does not use the scratch tail");
        return false;
      }
    } else if (!append_sampled_atomic_address_snapshot(
                   words, *spill, address_plan.result_address_vgpr, saved_address, arch, errors)) {
      return false;
    }
  }
  if (guest_first && is_cas) {
    // Address materialization/snapshot must precede CAS evidence because the
    // evidence destinations may overlap the live guest address pair.
    const uint16_t saved_compare =
        static_cast<uint16_t>(*options.scratch_vgpr + SampledAtomicScratchLayout::kCasCompare);
    const uint16_t saved_result =
        static_cast<uint16_t>(*options.scratch_vgpr + SampledAtomicScratchLayout::kCasResult);
    if (!append_sampled_atomic_cas_snapshot(words, *spill, *state.cas_compare_vgpr,
                                            *state.cas_result_vgpr, saved_compare, saved_result,
                                            arch, errors))
      return false;
    state.cas_compare_vgpr = saved_compare;
    state.cas_result_vgpr = saved_result;
  }
  if (private_layout && !append_sampled_private_owner_epoch_load(
                            words, bytes, *candidate.kernel_descriptor_file_offset, options,
                            *private_layout, arch, errors))
    return false;
  if (options.moi_persistent_sgprs.complete()) {
    if (!consan_detail::validate_scalar_state_temporaries(options, "sampled atomic prelude",
                                                          errors))
      return false;
    words.push_back(
        build_v_mov_b32_e32(*options.moi_owner_vgpr, *options.moi_persistent_sgprs.owner, arch));
    words.push_back(
        build_v_mov_b32_e32(*options.moi_epoch_vgpr, *options.moi_persistent_sgprs.epoch, arch));
  }
  if (!guest_first && !guest_preserves_address) {
    // Compiler-emitted ordinary acquire loads may reuse the low address VGPR
    // as their destination. Preserve the synchronization address before that
    // load replaces it with the acquired value. Atomic RMWs and disjoint
    // ordinary loads retain the post-guest path below.
    words.push_back(build_v_mov_b32_e32(
        saved_address, vector_source_vgpr(address_plan.result_address_vgpr), arch));
    words.push_back(build_v_mov_b32_e32(
        static_cast<uint16_t>(saved_address + 1u),
        vector_source_vgpr(static_cast<uint16_t>(address_plan.result_address_vgpr + 1u)), arch));
  }
  if (!guest_first) {
    if (!defer_guest) {
      if (!append_sampled_atomic_guest(words, bytes, candidate, arch, guest_instruction_offset))
        return false;
      words.insert(words.end(), trailing_guest_words.begin(), trailing_guest_words.end());
    }
    if (guest_preserves_address) {
      words.push_back(build_v_mov_b32_e32(
          saved_address, vector_source_vgpr(address_plan.result_address_vgpr), arch));
      words.push_back(build_v_mov_b32_e32(
          static_cast<uint16_t>(saved_address + 1u),
          vector_source_vgpr(static_cast<uint16_t>(address_plan.result_address_vgpr + 1u)), arch));
    }
  }
  return true;
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_sampled_pending_acquire_cave_words(
    std::span<const uint8_t> bytes, const MoiAtomicEvidenceSitePlan &candidate,
    const ConSanMoiAtomicAddressPlan &address_plan, const MoiOptions &options,
    const VgprSpillSequence *spill, const InlineSgprSpillSequence *scalar_spill,
    const MoiPrivateEpochLayout *private_layout, rj_code_arch_t arch, uint32_t selected_slot,
    uint32_t bank_count, std::optional<uint32_t> release_selected_slot,
    const ConSanMoiReportBufferLayout &layout, uint64_t cave_text_offset,
    uint64_t return_text_offset, std::vector<std::string> &errors,
    uint32_t *guest_instruction_offset = nullptr,
    std::span<const uint32_t> trailing_guest_words = {}) {
  const uint32_t pending_owner_bank_count = consan_moi_sampled_pending_acquire_owner_bank_count(
      layout.sampled_pending_acquire_capacity, layout.sampled_causal_window_capacity);
  if (!options.scratch_vgpr || !options.moi_report_buffer_address || !options.moi_owner_vgpr ||
      !options.moi_epoch_vgpr || !options.moi_exec_save_sgpr ||
      static_cast<uint32_t>(*options.scratch_vgpr) + sampled_atomic_scratch_count() > kMaxVgprs ||
      bank_count == 0u || !std::has_single_bit(bank_count) || pending_owner_bank_count == 0u ||
      selected_slot > layout.sampled_causal_window_capacity ||
      bank_count > layout.sampled_causal_window_capacity - selected_slot ||
      !address_plan.supported() || !candidate.site.raw_scope || candidate.site.width_bits != 32u ||
      !candidate.kernel_descriptor_file_offset)
    return std::nullopt;
  const auto role = sampled_atomic_role(candidate.event_kind, candidate.is_rmw);
  const auto scope = sampled_atomic_scope(*candidate.site.raw_scope);
  if (!role || !scope ||
      (static_cast<uint32_t>(*role) & static_cast<uint32_t>(ConSanMoiSampledSyncRole::Acquire)) ==
          0u)
    return std::nullopt;
  const auto workgroup_sources = moi_persistent_or_descriptor_workgroup_sources(
      bytes, *candidate.kernel_descriptor_file_offset, options, arch, errors,
      candidate.uses_cluster_workgroup_id);
  if (!workgroup_sources || candidate.site.file_offset > bytes.size() ||
      candidate.site.size > bytes.size() - candidate.site.file_offset)
    return std::nullopt;

  const bool is_cas = candidate.is_rmw && consan_atomic_is_compare_exchange(candidate.site);
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
  const auto success_descriptor =
      descriptor_for(!candidate.is_rmw ? ConSanMoiSampledSyncOutcome::NotApplicable
                     : is_cas          ? ConSanMoiSampledSyncOutcome::CasSuccess
                     : candidate.site.returns_old_value.value_or(false)
                         ? ConSanMoiSampledSyncOutcome::RmwReturnsOld
                         : ConSanMoiSampledSyncOutcome::RmwNoReturn);
  const auto failure_descriptor =
      is_cas ? descriptor_for(ConSanMoiSampledSyncOutcome::CasFailure) : success_descriptor;
  if (!success_descriptor || !failure_descriptor)
    return std::nullopt;

  const uint16_t base = *options.scratch_vgpr;
  const uint16_t value = static_cast<uint16_t>(base + SampledAtomicScratchLayout::kValue);
  const uint16_t expected = static_cast<uint16_t>(base + SampledAtomicScratchLayout::kExpected);
  const uint16_t saved_address =
      static_cast<uint16_t>(base + SampledAtomicScratchLayout::kSavedAddress);
  const uint16_t bank = static_cast<uint16_t>(base + SampledAtomicScratchLayout::kBank);
  const uint16_t original_exec = static_cast<uint16_t>(*options.moi_exec_save_sgpr + 6u);
  const uint16_t temporary_exec = *options.moi_exec_save_sgpr;
  const uint64_t report_base = *options.moi_report_buffer_address;
  const uint64_t first_pending_address = report_base + layout.sampled_pending_acquires_offset;

  std::vector<uint32_t> words;
  SampledAtomicPreludeState prelude;
  if (!append_sampled_atomic_prelude(words, bytes, candidate, address_plan, options, spill,
                                     scalar_spill, private_layout, arch, saved_address,
                                     /*defer_guest=*/false, prelude, errors,
                                     guest_instruction_offset, trailing_guest_words))
    return std::nullopt;
  if (!append_sampled_window_bank_index(words, options, *workgroup_sources, bank_count, bank,
                                        expected, *options.moi_owner_vgpr, arch)) {
    errors.emplace_back("ConSan MOI sampled pending acquire failed at bank selection");
    return std::nullopt;
  }
  if (selected_slot != 0u) {
    const auto absolute_slot =
        instrumentation::build_v_add_u32_literal(bank, expected, selected_slot, bank, arch);
    if (!absolute_slot)
      return std::nullopt;
    words.insert(words.end(), absolute_slot->begin(), absolute_slot->end());
  }
  InstructionSequence sequence(words);
  ConSanMoiRecordEmitter record(words, base, value, arch);
  const auto contention_label = sequence.make_label();
  const auto collision_label = sequence.make_label();
  const auto restore_label = sequence.make_label();
  if (!append_save_moi_special_state(words, moi_special_state_sgprs(options), arch)) {
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

  const auto publishing = instrumentation::build_v_mov_b32_literal(value, 1u, arch);
  const auto empty = instrumentation::build_v_mov_b32_literal(expected, 0u, arch);
  const auto claim = instrumentation::build_flat_atomic_cmpswap_b32(
      base, value, value, /*return_old_value=*/true, kRdna4ScopeDevice, arch);
  // BANK still names the causal-window slot stored in the pending payload.
  // Address the pending table through a separate (window, owner) index so all
  // waves at one static last-arriver RMW can retain their exact acquire.
  const auto append_pending_address = [&]() {
    if (pending_owner_bank_count > 1u) {
      const auto scale_window = instrumentation::build_v_mul_lo_u32_literal(
          value, expected, pending_owner_bank_count, bank, arch);
      const auto owner_bank = instrumentation::build_v_and_b32_literal(
          expected, pending_owner_bank_count - 1u, *options.moi_owner_vgpr, arch);
      const auto combine =
          instrumentation::build_v_add_u32(value, vector_source_vgpr(value), expected, arch);
      if (!scale_window || !owner_bank || !combine)
        return false;
      words.insert(words.end(), scale_window->begin(), scale_window->end());
      words.insert(words.end(), owner_bank->begin(), owner_bank->end());
      words.insert(words.end(), combine->begin(), combine->end());
    } else {
      words.push_back(build_v_mov_b32_e32(value, vector_source_vgpr(bank), arch));
    }
    return append_sampled_indexed_address(words, first_pending_address,
                                          sizeof(ConSanMoiSampledPendingAcquireSlot), value, base,
                                          expected, arch);
  };
  if (!append_pending_address() || !sequence.emit_all(publishing, empty, claim) ||
      !append_moi_global_atomic_wait(words, arch)) {
    return std::nullopt;
  }
  const auto claimed =
      instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), value, arch);
  const auto narrow_claimed =
      instrumentation::build_s_and_saveexec_b64(temporary_exec, kRdna4VccLo, arch);
  if (!sequence.emit_all(claimed, narrow_claimed))
    return std::nullopt;
  if (!sequence.emit_branch(contention_label, InstructionSequence::BranchKind::SccZero))
    return std::nullopt;

  if (release_selected_slot) {
    // RESERVED carries release_slot + 1 (zero means no release attachment).
    // BANK is the absolute acquire slot, so the wrapped unsigned delta also
    // handles a release window whose static base precedes the acquire window.
    const uint32_t release_delta = *release_selected_slot - selected_slot + 1u;
    const auto release_route =
        instrumentation::build_v_add_u32_literal(expected, value, release_delta, bank, arch);
    if (!release_route)
      return std::nullopt;
    words.insert(words.end(), release_route->begin(), release_route->end());
  }

  if (!record.store_vgpr(offsetof(ConSanMoiSampledPendingAcquireSlot, selected_slot), bank) ||
      !record.store_literal(offsetof(ConSanMoiSampledPendingAcquireSlot, generation),
                            static_cast<uint32_t>(options.moi_report_generation)) ||
      !record.store_literal(offsetof(ConSanMoiSampledPendingAcquireSlot, generation) + 4u,
                            static_cast<uint32_t>(options.moi_report_generation >> 32u)) ||
      !append_store_moi_report_dispatch_id_pair(
          record, options, offsetof(ConSanMoiSampledPendingAcquireSlot, dispatch_id), arch,
          ConSanMoiLiteralDispatchIdPolicy::AnyArchitecture) ||
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
      !(release_selected_slot
            ? record.store_vgpr(offsetof(ConSanMoiSampledPendingAcquireSlot, reserved), expected)
            : record.store_literal(offsetof(ConSanMoiSampledPendingAcquireSlot, reserved), 0u)) ||
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
    assert(prelude.cas_compare_vgpr && prelude.cas_result_vgpr);
    const auto cas_failure_label = sequence.make_label();
    const auto descriptor_done_label = sequence.make_label();
    const auto success = instrumentation::build_v_cmp_eq_u32_vcc(
        vector_source_vgpr(*prelude.cas_compare_vgpr), *prelude.cas_result_vgpr, arch);
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

  const auto wait_store = instrumentation::build_s_wait_global_store0(arch);
  const auto ready = instrumentation::build_v_mov_b32_literal(value, 2u, arch);
  const auto expected_publishing = instrumentation::build_v_mov_b32_literal(expected, 1u, arch);
  const auto commit = instrumentation::build_flat_atomic_cmpswap_b32(
      base, value, value, /*return_old_value=*/true, kRdna4ScopeDevice, arch);
  if (!sequence.emit(wait_store) || !append_pending_address() ||
      !sequence.emit_all(ready, expected_publishing, commit) ||
      !append_moi_global_atomic_wait(words, arch)) {
    return std::nullopt;
  }
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
      !append_restore_moi_special_state(words, moi_special_state_sgprs(options), arch))
    return std::nullopt;
  if (scalar_spill)
    words.insert(words.end(), scalar_spill->restore_words.begin(),
                 scalar_spill->restore_words.end());
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
    std::span<const uint8_t> bytes, const MoiAtomicEvidenceSitePlan &candidate,
    const ConSanMoiAtomicAddressPlan &address_plan, const MoiOptions &options,
    const VgprSpillSequence *spill, const InlineSgprSpillSequence *scalar_spill,
    const MoiPrivateEpochLayout *private_layout, rj_code_arch_t arch, uint32_t selected_slot,
    uint32_t bank_count, const ConSanMoiReportBufferLayout &layout, uint64_t cave_text_offset,
    uint64_t return_text_offset, std::vector<std::string> &errors,
    uint32_t *guest_instruction_offset = nullptr,
    std::span<const uint32_t> trailing_guest_words = {}, bool preserve_guest_at_anchor = false,
    std::span<const uint32_t> leading_guest_words = {}) {
  if (!options.scratch_vgpr || !options.moi_report_buffer_address || !options.moi_owner_vgpr ||
      !options.moi_epoch_vgpr || !options.moi_exec_save_sgpr ||
      static_cast<uint32_t>(*options.scratch_vgpr) + sampled_atomic_scratch_count() > 256u ||
      bank_count == 0u || !std::has_single_bit(bank_count) ||
      selected_slot > layout.sampled_sync_metadata_capacity ||
      bank_count > layout.sampled_sync_metadata_capacity - selected_slot ||
      selected_slot > layout.sampled_causal_window_capacity ||
      bank_count > layout.sampled_causal_window_capacity - selected_slot ||
      selected_slot > layout.sampled_watchpoint_capacity ||
      bank_count > layout.sampled_watchpoint_capacity - selected_slot)
    return std::nullopt;
  if (!address_plan.supported() || !candidate.site.raw_scope || candidate.site.width_bits != 32u)
    return std::nullopt;
  const auto role = sampled_atomic_role(candidate.event_kind, candidate.is_rmw);
  const auto scope = sampled_atomic_scope(*candidate.site.raw_scope);
  if (!role || !scope)
    return std::nullopt;
  if (!candidate.kernel_descriptor_file_offset) {
    errors.emplace_back("ConSan MOI sampled atomic metadata lacks an owning kernel descriptor");
    return std::nullopt;
  }
  const auto workgroup_sources = moi_persistent_or_descriptor_workgroup_sources(
      bytes, *candidate.kernel_descriptor_file_offset, options, arch, errors,
      candidate.uses_cluster_workgroup_id);
  if (!workgroup_sources)
    return std::nullopt;
  if (candidate.site.file_offset > bytes.size() ||
      candidate.site.size > bytes.size() - candidate.site.file_offset) {
    errors.emplace_back("ConSan MOI sampled atomic metadata site exceeds ELF bytes");
    return std::nullopt;
  }

  const bool is_cas = candidate.is_rmw && consan_atomic_is_compare_exchange(candidate.site);
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
  const auto success_descriptor =
      descriptor_for(!candidate.is_rmw ? ConSanMoiSampledSyncOutcome::NotApplicable
                     : is_cas          ? ConSanMoiSampledSyncOutcome::CasSuccess
                     : candidate.site.returns_old_value.value_or(false)
                         ? ConSanMoiSampledSyncOutcome::RmwReturnsOld
                         : ConSanMoiSampledSyncOutcome::RmwNoReturn);
  const auto failure_descriptor =
      is_cas ? descriptor_for(ConSanMoiSampledSyncOutcome::CasFailure) : success_descriptor;
  if (!success_descriptor || !failure_descriptor)
    return std::nullopt;

  const uint16_t base = *options.scratch_vgpr;
  const uint16_t value = static_cast<uint16_t>(base + SampledAtomicScratchLayout::kValue);
  const uint16_t expected = static_cast<uint16_t>(base + SampledAtomicScratchLayout::kExpected);
  const uint16_t saved_address =
      static_cast<uint16_t>(base + SampledAtomicScratchLayout::kSavedAddress);
  const uint16_t bank = static_cast<uint16_t>(base + SampledAtomicScratchLayout::kBank);
  const uint16_t original_exec = static_cast<uint16_t>(*options.moi_exec_save_sgpr + 6u);
  const uint16_t temporary_exec = *options.moi_exec_save_sgpr;
  const uint64_t report_base = *options.moi_report_buffer_address;
  const uint64_t first_window_address = report_base + layout.sampled_causal_windows_offset;
  const uint64_t first_watchpoint_address = report_base + layout.sampled_watchpoints_offset;
  const uint64_t first_metadata_address = report_base + layout.sampled_sync_metadata_offset;

  // A release-only RMW that does not return a value has no outcome evidence
  // needed by the probe. Publish its shadow metadata before executing the
  // guest release so the release operation orders that metadata and the
  // original code can resume immediately after the externally visible RMW.
  // This is especially important for code that publishes its own side
  // metadata immediately after the atomic. Keep guest-first ordering when a
  // spill overlaps guest operands, and for acquire/CAS operations whose
  // outcome is needed by the probe.
  const bool release_only =
      (static_cast<uint32_t>(*role) & static_cast<uint32_t>(ConSanMoiSampledSyncRole::Release)) !=
          0u &&
      (static_cast<uint32_t>(*role) & static_cast<uint32_t>(ConSanMoiSampledSyncRole::Acquire)) ==
          0u;
  const bool defer_guest =
      (release_only || preserve_guest_at_anchor) && !is_cas &&
      !(spill && sampled_atomic_spill_overlaps_guest_operands(*spill, candidate.site));
  if (preserve_guest_at_anchor && !defer_guest)
    return std::nullopt;

  std::vector<uint32_t> words(leading_guest_words.begin(), leading_guest_words.end());
  SampledAtomicPreludeState prelude;
  if (!append_sampled_atomic_prelude(words, bytes, candidate, address_plan, options, spill,
                                     scalar_spill, private_layout, arch, saved_address, defer_guest,
                                     prelude, errors, guest_instruction_offset,
                                     trailing_guest_words))
    return std::nullopt;
  if (!append_sampled_window_bank_index(words, options, *workgroup_sources, bank_count, bank,
                                        expected, *options.moi_owner_vgpr, arch)) {
    errors.emplace_back("ConSan MOI sampled atomic metadata failed at bank selection");
    return std::nullopt;
  }
  if (selected_slot != 0u) {
    const auto absolute_slot =
        instrumentation::build_v_add_u32_literal(bank, expected, selected_slot, bank, arch);
    if (!absolute_slot)
      return std::nullopt;
    words.insert(words.end(), absolute_slot->begin(), absolute_slot->end());
  }
  InstructionSequence sequence(words);
  ConSanMoiRecordEmitter record(words, base, value, arch);
  const auto collision_label = sequence.make_label();
  const auto restore_label = sequence.make_label();
  if (!append_save_moi_special_state(words, moi_special_state_sgprs(options), arch))
    return std::nullopt;
  const auto save_exec = instrumentation::build_s_mov_b64(original_exec, kRdna4ExecLo, arch);
  if (!save_exec)
    return std::nullopt;
  words.push_back(*save_exec);

  if (!append_sampled_indexed_address(words, first_window_address,
                                      sizeof(ConSanMoiSampledCausalWindow), bank, base, expected,
                                      arch))
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
  const auto narrow_equal_dispatch_id = [&](uint32_t offset, bool high_word) -> bool {
    if (!append_load_u32_vgpr_at_offset(words, base, offset, value, arch) ||
        !append_compare_moi_report_dispatch_id_word(
            words, options, value, expected, high_word, arch,
            ConSanMoiLiteralDispatchIdPolicy::AnyArchitecture)) {
      return false;
    }
    const auto narrow =
        instrumentation::build_s_and_saveexec_b64(temporary_exec, kRdna4VccLo, arch);
    if (!sequence.emit(narrow))
      return false;
    return sequence.emit_branch(restore_label, InstructionSequence::BranchKind::SccZero);
  };
  const auto narrow_equal_workgroup = [&](uint32_t offset,
                                          const ConSanMoiWorkgroupSource &source) -> bool {
    if (!source.has_value())
      return narrow_equal_literal(offset, 0u);
    if (!consan_detail::append_workgroup_source_value(words, source, expected, arch))
      return false;
    return narrow_equal_vgpr(offset, expected);
  };
  if (!narrow_equal_literal(offsetof(ConSanMoiSampledCausalWindow, generation),
                            static_cast<uint32_t>(options.moi_report_generation)) ||
      !narrow_equal_literal(offsetof(ConSanMoiSampledCausalWindow, generation) + 4u,
                            static_cast<uint32_t>(options.moi_report_generation >> 32u)) ||
      !narrow_equal_dispatch_id(offsetof(ConSanMoiSampledCausalWindow, dispatch_id),
                                /*high_word=*/false) ||
      !narrow_equal_dispatch_id(offsetof(ConSanMoiSampledCausalWindow, dispatch_id) + 4u,
                                /*high_word=*/true) ||
      !narrow_equal_workgroup(offsetof(ConSanMoiSampledCausalWindow, workgroup_x),
                              workgroup_sources->x) ||
      !narrow_equal_workgroup(offsetof(ConSanMoiSampledCausalWindow, workgroup_y),
                              workgroup_sources->y) ||
      !narrow_equal_workgroup(offsetof(ConSanMoiSampledCausalWindow, workgroup_z),
                              workgroup_sources->z) ||
      !narrow_equal_vgpr(offsetof(ConSanMoiSampledCausalWindow, epoch), *options.moi_epoch_vgpr) ||
      !narrow_equal_vgpr(offsetof(ConSanMoiSampledCausalWindow, first_entry), bank) ||
      !narrow_equal_literal(offsetof(ConSanMoiSampledCausalWindow, entry_count), 1u) ||
      !narrow_equal_literal(offsetof(ConSanMoiSampledCausalWindow, publication_state),
                            static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Ready)) ||
      !narrow_equal_workgroup(offsetof(ConSanMoiSampledCausalWindow, cluster_workgroup_id),
                              workgroup_sources->cluster_workgroup_id))
    return std::nullopt;

  if (!append_sampled_indexed_address(words, first_watchpoint_address, sizeof(uint64_t), bank, base,
                                      expected, arch))
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
    const auto sentinel = instrumentation::build_v_mov_b32_literal(
        value, kConSanMoiSampledSyncPublishingDescriptor, arch);
    const auto zero = instrumentation::build_v_mov_b32_literal(expected, 0, arch);
    const auto claim = instrumentation::build_flat_atomic_cmpswap_b32(base, value, value, true,
                                                                      kRdna4ScopeDevice, arch);
    if (!append_sampled_indexed_address(
            words,
            first_metadata_address + offsetof(ConSanMoiSampledSyncMetadataPacked, descriptor),
            sizeof(ConSanMoiSampledSyncMetadataPacked), bank, base, expected, arch) ||
        !sequence.emit_all(sentinel, zero, claim) || !append_moi_global_atomic_wait(words, arch)) {
      return false;
    }
    const auto claimed =
        instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), value, arch);
    const auto narrow_claimed =
        instrumentation::build_s_and_saveexec_b64(temporary_exec, kRdna4VccLo, arch);
    if (!sequence.emit_all(claimed, narrow_claimed))
      return false;
    if (!sequence.emit_branch(collision_label, InstructionSequence::BranchKind::SccZero))
      return false;
    if (!append_sampled_indexed_address(words, first_metadata_address,
                                        sizeof(ConSanMoiSampledSyncMetadataPacked), bank, base,
                                        expected, arch) ||
        !record.store_vgpr(offsetof(ConSanMoiSampledSyncMetadataPacked, address), saved_address) ||
        !record.store_vgpr(offsetof(ConSanMoiSampledSyncMetadataPacked, address) + 4u,
                           static_cast<uint16_t>(saved_address + 1u)) ||
        !record.store_literal(offsetof(ConSanMoiSampledSyncMetadataPacked, byte_count), 4u) ||
        !record.store_vgpr(offsetof(ConSanMoiSampledSyncMetadataPacked, epoch_before),
                           *options.moi_epoch_vgpr) ||
        !record.store_vgpr(offsetof(ConSanMoiSampledSyncMetadataPacked, epoch_after),
                           *options.moi_epoch_vgpr))
      return false;
    const auto wait = instrumentation::build_s_wait_global_store0(arch);
    const auto final_value = instrumentation::build_v_mov_b32_literal(value, descriptor, arch);
    const auto final_expected = instrumentation::build_v_mov_b32_literal(
        expected, kConSanMoiSampledSyncPublishingDescriptor, arch);
    const auto commit = instrumentation::build_flat_atomic_cmpswap_b32(base, value, value, true,
                                                                       kRdna4ScopeDevice, arch);
    if (!sequence.emit(wait) ||
        !append_sampled_indexed_address(
            words,
            first_metadata_address + offsetof(ConSanMoiSampledSyncMetadataPacked, descriptor),
            sizeof(ConSanMoiSampledSyncMetadataPacked), bank, base, expected, arch) ||
        !sequence.emit_all(final_value, final_expected, commit) ||
        !append_moi_global_atomic_wait(words, arch)) {
      return false;
    }
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
    assert(prelude.cas_compare_vgpr && prelude.cas_result_vgpr);
    const auto cas_failure_label = sequence.make_label();
    const auto success = instrumentation::build_v_cmp_eq_u32_vcc(
        vector_source_vgpr(*prelude.cas_compare_vgpr), *prelude.cas_result_vgpr, arch);
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
      !append_restore_moi_special_state(words, moi_special_state_sgprs(options), arch))
    return std::nullopt;
  if (scalar_spill)
    words.insert(words.end(), scalar_spill->restore_words.begin(),
                 scalar_spill->restore_words.end());
  if (spill)
    words.insert(words.end(), spill->restore_words.begin(), spill->restore_words.end());
  if (preserve_guest_at_anchor) {
    if (!append_moi_direct_or_indirect_return(words, cave_text_offset, return_text_offset, options,
                                              arch)) {
      return std::nullopt;
    }
  } else if (defer_guest) {
    std::vector<uint32_t> deferred_guest_words;
    if (!append_sampled_atomic_guest(deferred_guest_words, bytes, candidate, arch,
                                     /*guest_instruction_offset=*/nullptr)) {
      return std::nullopt;
    }
    deferred_guest_words.insert(deferred_guest_words.end(), trailing_guest_words.begin(),
                                trailing_guest_words.end());
    const uint64_t branch_text_offset =
        cave_text_offset + (words.size() + deferred_guest_words.size()) * sizeof(uint32_t);
    if (const auto direct = compute_sopp_branch_simm16(branch_text_offset, return_text_offset)) {
      if (guest_instruction_offset)
        *guest_instruction_offset = static_cast<uint32_t>(words.size() * sizeof(uint32_t));
      words.insert(words.end(), deferred_guest_words.begin(), deferred_guest_words.end());
      words.push_back(build_s_branch(*direct, arch));
    } else {
      const auto jump_sgprs = moi_indirect_jump_sgprs(options);
      if (!jump_sgprs || !append_moi_prepare_scc_preserving_indirect_jump(
                             words, cave_text_offset, return_text_offset, jump_sgprs->pc_sgpr,
                             jump_sgprs->scc_save_sgpr, arch)) {
        return std::nullopt;
      }
      if (guest_instruction_offset)
        *guest_instruction_offset = static_cast<uint32_t>(words.size() * sizeof(uint32_t));
      words.insert(words.end(), deferred_guest_words.begin(), deferred_guest_words.end());
      words.push_back(build_s_setpc_b64(jump_sgprs->pc_sgpr, arch));
    }
  } else if (!append_moi_direct_or_indirect_return(words, cave_text_offset, return_text_offset,
                                                   options, arch)) {
    return std::nullopt;
  }

  if (!sequence.resolve_branches(arch))
    return std::nullopt;
  return words;
}
