using InlineSgprSpillSequence = SgprSpillSequence;

[[nodiscard]] bool
moi_scalar_spill_requires_dynamic_vgpr_frame(const ConSanResult &result,
                                             const ResolvedMoiScratchPlan &resources,
                                             const ConSanOptions &options) {
  if (!options.automatic_moi_inline_sgpr_spill && !options.automatic_moi_record_replay_sgpr_spill) {
    return false;
  }
  return std::ranges::any_of(resources.owner_descriptor_file_offsets, [&](uint64_t offset) {
    const ConSanKernelInfo *kernel = kernel_for_descriptor(result, offset);
    return kernel != nullptr && kernel->uses_dynamic_stack.value_or(false);
  });
}

[[nodiscard]] std::optional<InlineSgprSpillSequence> build_inline_sgpr_spill_sequence(
    const ConSanResult &result, const ResolvedMoiScratchPlan &resources,
    const ConSanOptions &options, MoiSpillManagers &managers, rj_code_arch_t arch,
    std::vector<std::string> &warnings, std::optional<uint32_t> private_layout_base,
    const VgprSpillSequence *vgpr_spill = nullptr) {
  if ((!options.automatic_moi_inline_sgpr_spill &&
       !options.automatic_moi_record_replay_sgpr_spill) ||
      !options.moi_exec_save_sgpr)
    return std::nullopt;
  if (!instrumentation::is_admitted_arch(arch) || resources.owner_descriptor_file_offsets.empty()) {
    warnings.emplace_back(
        "ConSan MOI spill-backed scalar state requires at least one supported kernel owner");
    return std::nullopt;
  }
  const auto private_limit = address_free_scratch_private_limit(arch);
  if (!private_limit) {
    warnings.emplace_back("ConSan MOI spill-backed Inline scalar state has no private capacity");
    return std::nullopt;
  }
  const uint16_t count = moi_exec_save_sgpr_count(options, arch);
  if (moi_scalar_spill_requires_dynamic_vgpr_frame(result, resources, options)) {
    if (vgpr_spill == nullptr || !vgpr_spill->uses_dynamic_stack_frame) {
      warnings.emplace_back(
          "ConSan MOI dynamic-stack scalar spill requires an established VGPR spill frame");
      return std::nullopt;
    }
    auto sequence = build_dynamic_stack_sgpr_spill_sequence(
        *options.moi_exec_save_sgpr, count, resources.base, *vgpr_spill,
        static_cast<uint32_t>(resources.count) * SpillManager::kSlotBytes, arch);
    if (!sequence)
      warnings.emplace_back("ConSan MOI could not encode the dynamic-stack scalar spill sequence");
    return sequence;
  }
  const uint32_t layout_base =
      private_layout_base.value_or(resources.original_private_segment_size);
  SpillManager planned_manager(layout_base, *private_limit);
  SpillManager *committed_manager = nullptr;
  if (resources.owner_descriptor_file_offsets.size() == 1u) {
    const uint64_t descriptor_offset = resources.owner_descriptor_file_offsets.front();
    auto [it, inserted] = managers.try_emplace(descriptor_offset, layout_base, *private_limit);
    (void)inserted;
    planned_manager = it->second;
    committed_manager = &it->second;
  }
  if (!planned_manager.allocate_slots(RegisterRef{RegClass::SGPR, *options.moi_exec_save_sgpr, 1u},
                                      count)) {
    warnings.emplace_back("ConSan MOI could not reserve the Inline scalar spill slots");
    return std::nullopt;
  }
  const auto required_private_bytes =
      normalize_address_free_scratch_private_size(arch, planned_manager.total_private_bytes());
  const auto wait_store = instrumentation::build_s_wait_private_store0(arch);
  const auto wait_load = instrumentation::build_s_wait_private_load0(arch);
  if (!required_private_bytes || !wait_store || !wait_load) {
    warnings.emplace_back("ConSan MOI could not encode the Inline scalar spill waits");
    return std::nullopt;
  }

  InlineSgprSpillSequence sequence;
  sequence.sgpr_base = *options.moi_exec_save_sgpr;
  sequence.sgpr_count = count;
  sequence.save_words.reserve(static_cast<size_t>(count) * 4u + 1u);
  sequence.restore_words.reserve(static_cast<size_t>(count) * 5u);
  for (uint16_t i = 0; i < count; ++i) {
    const uint16_t sgpr = static_cast<uint16_t>(sequence.sgpr_base + i);
    const auto offset = planned_manager.offset_for(RegisterRef{RegClass::SGPR, sgpr, 1u});
    if (!offset)
      return std::nullopt;
    sequence.save_words.push_back(build_v_mov_b32_e32(resources.base, sgpr, arch));
    const auto store = instrumentation::build_private_store_b32(resources.base, *offset, arch);
    const auto load = instrumentation::build_private_load_b32(resources.base, *offset, arch);
    const auto restore = instrumentation::build_v_readfirstlane_b32(sgpr, resources.base, arch);
    if (!store || !load || !restore) {
      warnings.emplace_back("ConSan MOI could not encode the Inline scalar spill sequence");
      return std::nullopt;
    }
    sequence.save_words.insert(sequence.save_words.end(), store->begin(), store->end());
    sequence.restore_words.insert(sequence.restore_words.end(), load->begin(), load->end());
    sequence.restore_words.push_back(*wait_load);
    sequence.restore_words.push_back(*restore);
  }
  sequence.save_words.push_back(*wait_store);
  sequence.total_private_bytes = *required_private_bytes;
  if (committed_manager)
    *committed_manager = std::move(planned_manager);
  return sequence;
}

void try_apply_inline_shadow_patch(std::span<const uint8_t> bytes, const ConSanOptions &options,
                                   rj_code_arch_t arch, MoiResourcePlanningState &resource_state,
                                   ConSanResult &result) {
  if (!options.moi_report_buffer_address)
    return;
  if (!instrumentation::is_admitted_arch(arch)) {
    result.warnings.emplace_back(
        "ConSan MOI inline-shadow probe does not support this architecture");
    return;
  }
  std::unique_ptr<Decoder> relocation_decoder = Decoder::create(arch);
  if (!relocation_decoder) {
    result.warnings.emplace_back("ConSan MOI inline-shadow could not create a relocation decoder");
    return;
  }

  const ConSanMoiReportBufferLayout layout = resolve_inline_report_layout(options);
  if (options.max_patches == 0)
    return;
  if (!options.moi_inline_workgroup_shadow &&
      layout.exact_shadow_entry_capacity < kConSanMoiInlineShadowConservativeExactShadowEntries) {
    result.warnings.emplace_back(
        "ConSan MOI inline-shadow probe requires exact-shadow capacity for the full 64 KiB "
        "LDS address range");
    return;
  }
  if (!validate_inline_shadow_exec_save_sgpr(options, arch, result.warnings))
    return;
  if (!options.moi_owner_vgpr && options.moi_persistent_vgpr_assignments.empty() &&
      !options.moi_persistent_owner_sgpr && !options.automatic_moi_private_epoch) {
    result.warnings.emplace_back("ConSan MOI inline-shadow probe requires persistent owner state");
    return;
  }
  std::vector<const ConSanMoiCandidate *> candidates =
      find_inline_shadow_access_candidates(result, bytes, arch);
  apply_test_kernel_filter(candidates, options);
  std::ranges::sort(candidates, [](const ConSanMoiCandidate *lhs, const ConSanMoiCandidate *rhs) {
    return lhs->text_offset < rhs->text_offset;
  });
  if (candidates.empty()) {
    result.warnings.emplace_back(
        "ConSan MOI inline-shadow probe found no supported native LDS or normalized group-flat "
        "load/store candidate");
    return;
  }

  struct PlannedInlineShadowPatch {
    const ConSanMoiCandidate *candidate = nullptr;
    DbiPatchPlacement placement;
    std::optional<uint64_t> entry_island_offset;
    bool entry_island_at_anchor = false;
    bool dense_call_anchor = false;
    std::optional<uint64_t> dense_dispatcher_offset;
    bool branch_only_continuation = false;
    std::vector<uint64_t> entry_relay_offsets;
    std::vector<uint64_t> return_relay_offsets;
    std::optional<BranchOnlyRelayRoute> branch_only_route;
    std::vector<uint32_t> displaced_tail_words;
    ResolvedMoiScratchPlan resources;
    ConSanMoiWorkgroupSources workgroup_sources;
    std::optional<VgprSpillSequence> spill;
    std::optional<InlineSgprSpillSequence> scalar_spill;
    std::optional<uint32_t> private_epoch_offset;
    std::optional<uint32_t> private_owner_offset;
    std::optional<uint32_t> private_workgroup_key_offset;
    std::optional<uint16_t> private_owner_shift;
    std::optional<ConSanMoiWorkgroupShadowLayout> workgroup_shadow;
    uint16_t workgroup_shadow_compact_token = 0;
    uint32_t required_private_bytes = 0;
  };
  std::vector<PlannedInlineShadowPatch> planned_patches;
  planned_patches.reserve(candidates.size());
  AmdGpuCodeObject code_object(bytes.data(), bytes.size());
  const uint64_t original_text_size =
      code_object.text_sections().size() == 1 ? code_object.text_sections().front()->size() : 0;
  constexpr uint64_t kIndirectIslandWords = 8u;
  // A dense dispatcher may need one comparison, one conditional branch, and
  // a long jump per site, plus a fixed relocated-host route. Twenty words per
  // site safely amortizes that fixed route even for small owner groups.
  constexpr uint64_t access_relay_words = 20u;
  const bool has_function_candidate = std::ranges::any_of(
      candidates, [](const ConSanMoiCandidate *candidate) { return !candidate->in_kernel; });
  std::unordered_map<uint64_t, size_t> candidate_count_by_descriptor;
  std::unordered_set<uint64_t> dynamic_lds_descriptors;
  for (const AmdGpuKernelInfo &kernel : code_object.kernels()) {
    if (kernel.has_dynamic_lds)
      dynamic_lds_descriptors.insert(kernel.descriptor_file_offset);
  }
  std::unordered_map<uint64_t, std::optional<ConSanOptions>> direct_options_by_descriptor;
  std::unordered_map<uint64_t, std::optional<ConSanMoiWorkgroupShadowLayout>>
      direct_workgroup_shadow_by_descriptor;
  std::unordered_map<uint64_t, std::optional<MoiPrivateEpochLayout>>
      direct_private_layout_by_descriptor;
  std::unordered_map<uint64_t, std::optional<uint16_t>> direct_owner_shift_by_descriptor;
  std::unordered_map<uint64_t, std::optional<ConSanMoiWorkgroupSources>>
      direct_workgroup_sources_by_descriptor;
  bool warned_dynamic_lds_external_shadow = false;
  for (const ConSanMoiCandidate *candidate : candidates) {
    // Direct-kernel compact tokens live in descriptor-local domains. Shared
    // function candidates remain on the external table because one helper may
    // have several owning descriptors and therefore no unambiguous local token
    // domain; their presence must not disable compact state for direct sites.
    if (candidate->in_kernel && candidate->kernel_descriptor_file_offset)
      ++candidate_count_by_descriptor[*candidate->kernel_descriptor_file_offset];
  }
  const bool has_branch_only_dynamic_assignment =
      std::ranges::any_of(options.moi_transient_sgpr_assignments,
                          [](const ConSanMoiTransientSgprAssignment &assignment) {
                            return assignment.dynamic_stack_borrowed_sgpr.has_value();
                          });
  const bool use_indirect_appended =
      options.moi_exec_save_sgpr.has_value() &&
      (candidates.size() >= 8u || has_function_candidate || has_branch_only_dynamic_assignment);
  uint64_t reserved_barrier_island_count = 0u;
  if (use_indirect_appended && options.moi_track_barriers) {
    for (const ConSanKernelInfo &kernel : result.kernels)
      reserved_barrier_island_count += kernel.barrier_sites.size();
    for (const ConSanFunctionInfo &function : result.functions)
      reserved_barrier_island_count += function.barrier_sites.size();
    reserved_barrier_island_count =
        std::min<uint64_t>(reserved_barrier_island_count, options.max_patches);
  }
  const uint64_t indirect_body_begin =
      original_text_size + reserved_barrier_island_count * kIndirectIslandWords * sizeof(uint32_t) +
      candidates.size() * access_relay_words * sizeof(uint32_t);
  const uint64_t access_island_begin =
      original_text_size + reserved_barrier_island_count * kIndirectIslandWords * sizeof(uint32_t);
  DbiPatchPlacementPlanner placement_planner(arch, use_indirect_appended ? indirect_body_begin
                                                                         : original_text_size);

  std::span<const uint8_t> original_text;
  if (code_object.text_sections().size() == 1) {
    const Section *text = code_object.text_sections().front();
    original_text =
        std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(text->data()), text->size());
  }
  std::vector<std::pair<uint64_t, uint64_t>> synchronization_ranges;
  const auto append_synchronization_ranges = [&](const auto &sites) {
    for (const auto &site : sites) {
      if (site.size != 0u)
        synchronization_ranges.emplace_back(site.text_offset, site.text_offset + site.size);
    }
  };
  for (const ConSanKernelInfo &kernel : result.kernels) {
    append_synchronization_ranges(kernel.barrier_sites);
    append_synchronization_ranges(kernel.atomic_sites);
    append_synchronization_ranges(kernel.fence_sites);
  }
  for (const ConSanFunctionInfo &function : result.functions) {
    append_synchronization_ranges(function.barrier_sites);
    append_synchronization_ranges(function.atomic_sites);
    append_synchronization_ranges(function.fence_sites);
  }
  std::ranges::sort(synchronization_ranges);
  size_t merged_synchronization_range_count = 0u;
  for (const auto &range : synchronization_ranges) {
    if (merged_synchronization_range_count != 0u &&
        synchronization_ranges[merged_synchronization_range_count - 1u].second >= range.first) {
      synchronization_ranges[merged_synchronization_range_count - 1u].second = std::max(
          synchronization_ranges[merged_synchronization_range_count - 1u].second, range.second);
    } else {
      synchronization_ranges[merged_synchronization_range_count++] = range;
    }
  }
  synchronization_ranges.resize(merged_synchronization_range_count);
  const auto relocation_overlaps_synchronization = [&](uint64_t begin, uint64_t end) {
    const auto after = std::ranges::lower_bound(synchronization_ranges, end, {},
                                                [](const auto &range) { return range.first; });
    return after != synchronization_ranges.begin() && std::prev(after)->second > begin;
  };
  std::optional<MoiLocalNopIslandAllocator> local_entry_islands;
  if (use_indirect_appended && !original_text.empty())
    local_entry_islands.emplace(original_text, result, arch, kIndirectIslandWords,
                                options.preapplied_reserved_ranges);
  BranchOnlyRelayRouter branch_router;
  if (local_entry_islands) {
    for (uint64_t offset : local_entry_islands->available_word_offsets())
      (void)branch_router.offer(offset, BranchOnlyRelayProvenance::PristineNop);
  }
  const auto retire_branch_relay_range = [&](uint64_t offset, uint32_t words) {
    branch_router.retire_range(offset, static_cast<uint64_t>(words) * sizeof(uint32_t));
  };

  std::unordered_map<const ConSanMoiCandidate *, uint64_t> dense_entry_islands;
  std::unordered_map<const ConSanMoiCandidate *, uint64_t> dense_dispatchers;
  std::unordered_map<const ConSanMoiCandidate *, size_t> candidate_indices;
  candidate_indices.reserve(candidates.size());
  for (size_t index = 0; index < candidates.size(); ++index)
    candidate_indices.emplace(candidates[index], index);
  struct DenseEntryHost {
    uint64_t host_offset = 0;
    std::vector<uint32_t> displaced_words;
    uint64_t body_offset = 0;
  };
  std::map<uint64_t, DenseEntryHost> dense_entry_hosts;
  // Dense routing is partitioned by owning container, SOPP reach, and a
  // bounded number of candidates per dispatcher below. Its safety therefore
  // depends on the local group shape, not the total number of sites in an
  // unrelated code object. Retain the planning state's pristine CFG and
  // liveness facts so a relocated relay host cannot clobber transient state.
  const auto &dense_host_blocks = resource_state.blocks;
  if (is_rdna4_family_arch(arch) && use_indirect_appended && local_entry_islands) {
    struct DenseGroup {
      std::vector<const ConSanMoiCandidate *> candidates;
      size_t first_candidate_index = 0;
    };
    std::map<std::pair<bool, std::string>, std::vector<const ConSanMoiCandidate *>> owner_groups;
    for (size_t candidate_index = 0; candidate_index < candidates.size(); ++candidate_index) {
      const ConSanMoiCandidate *candidate = candidates[candidate_index];
      owner_groups[{candidate->in_kernel, candidate->container_name}].push_back(candidate);
    }
    constexpr uint64_t kMaxCommonRelayAnchorSpan =
        static_cast<uint64_t>(static_cast<int64_t>(std::numeric_limits<int16_t>::max()) -
                              static_cast<int64_t>(std::numeric_limits<int16_t>::min())) *
        sizeof(uint32_t);
    std::vector<std::pair<std::pair<bool, std::string>, DenseGroup>> groups;
    for (const auto &[key, owner_candidates] : owner_groups) {
      for (const ConSanMoiCandidate *candidate : owner_candidates) {
        if (groups.empty() || groups.back().first != key ||
            (arch != ROCJITSU_CODE_ARCH_GFX1250 && groups.back().second.candidates.size() == 64u) ||
            candidate->text_offset - groups.back().second.candidates.front()->text_offset >
                kMaxCommonRelayAnchorSpan) {
          groups.push_back(
              {key, {.candidates = {}, .first_candidate_index = candidate_indices.at(candidate)}});
        }
        groups.back().second.candidates.push_back(candidate);
      }
    }
    for (const auto &[key, group] : groups) {
      const ConSanCandidateResourcePlan *group_resource =
          group.candidates.empty() ? nullptr
                                   : resource_plan_for_candidate(result, *group.candidates.front());
      ConSanOptions group_options = options;
      if (group_resource != nullptr)
        (void)apply_moi_transient_sgpr_assignment(group_options,
                                                  group_resource->owner_descriptor_file_offsets);
      if (!moi_inline_dense_router_scalar_abi(group_options, arch))
        continue;
      bool has_dense_neighbors = false;
      for (size_t index = 1; index < group.candidates.size(); ++index) {
        has_dense_neighbors |=
            group.candidates[index]->text_offset <
            group.candidates[index - 1u]->text_offset + kIndirectIslandWords * sizeof(uint32_t);
      }
      const bool has_stranded_entry =
          std::ranges::any_of(group.candidates, [&](const ConSanMoiCandidate *candidate) {
            const size_t candidate_index = candidate_indices.at(candidate);
            const uint64_t default_entry =
                access_island_begin + candidate_index * access_relay_words * sizeof(uint32_t);
            return !compute_sopp_branch_simm16(candidate->text_offset, default_entry);
          });
      const bool explicit_anchors_fit =
          arch == ROCJITSU_CODE_ARCH_GFX1250 ||
          std::ranges::all_of(group.candidates, [](const ConSanMoiCandidate *candidate) {
            return candidate->size >= 2u * sizeof(uint32_t);
          });
      const bool singleton_has_direct_island =
          group.candidates.size() == 1u &&
          local_entry_islands->has_reachable(group.candidates.front()->text_offset,
                                             kIndirectIslandWords);
      // gfx1250 also uses the shared call relay to compact nearby anchors.
      // RDNA4 has no scalar call instruction, so use an explicit key plus
      // branch only when the ordinary per-site relay is stranded.
      if (!explicit_anchors_fit || singleton_has_direct_island ||
          (!(arch == ROCJITSU_CODE_ARCH_GFX1250 && has_dense_neighbors) && !has_stranded_entry))
        continue;
      std::vector<uint64_t> anchors;
      anchors.reserve(group.candidates.size());
      for (const ConSanMoiCandidate *candidate : group.candidates)
        anchors.push_back(candidate->text_offset);
      const uint64_t dispatcher =
          access_island_begin + group.first_candidate_index * access_relay_words * sizeof(uint32_t);
      auto island = local_entry_islands->claim_reachable_from_all(anchors);
      if (island)
        retire_branch_relay_range(*island, kIndirectIslandWords);
      std::vector<MoiSgprRange> host_bootstrap_ranges;
      MoiDenseOwnerLivenessProof host_owner_proof;
      if (!island && key.first) {
        if (group_resource != nullptr) {
          host_bootstrap_ranges = moi_relocatable_host_scalar_ranges(
              group_options, MoiRelocatableHostScalarState::AccessRouter, arch);
          host_owner_proof = resource_state.prepare_dense_liveness_for_owners(
              group_resource->owner_descriptor_file_offsets, host_bootstrap_ranges);
        }
        const auto kernel = std::ranges::find(result.kernels, key.second, &ConSanKernelInfo::name);
        if (kernel != result.kernels.end()) {
          constexpr size_t kHostWords = kIndirectIslandWords + 1u;
          constexpr uint64_t kHostBytes = kHostWords * sizeof(uint32_t);
          const uint64_t kernel_end = kernel->entry_text_offset + kernel->code_size;
          std::vector<std::pair<uint64_t, uint64_t>> occupied_ranges;
          occupied_ranges.reserve(candidates.size() + kernel->barrier_sites.size() +
                                  kernel->fence_sites.size() + kernel->atomic_sites.size() +
                                  options.preapplied_reserved_ranges.size());
          const auto append_occupied_range = [&](uint64_t begin, uint64_t size) {
            if (size != 0u)
              occupied_ranges.emplace_back(begin, begin + size);
          };
          // A kernel can be split into several relay groups. A host for one
          // group must not consume an access anchor owned by another group.
          const auto owner_candidates = owner_groups.find(key);
          if (owner_candidates != owner_groups.end()) {
            for (const ConSanMoiCandidate *candidate : owner_candidates->second)
              append_occupied_range(candidate->text_offset, candidate->size);
          }
          for (const ConSanBarrierSite &site : kernel->barrier_sites)
            append_occupied_range(site.text_offset, site.size);
          for (const ConSanFenceSite &site : kernel->fence_sites)
            append_occupied_range(site.text_offset, site.size);
          for (const ConSanAtomicSite &site : kernel->atomic_sites)
            append_occupied_range(site.text_offset, site.size);
          for (const ConSanPreappliedReservedRange &reserved : options.preapplied_reserved_ranges)
            append_occupied_range(reserved.text_offset, reserved.size);
          std::ranges::sort(occupied_ranges);
          size_t merged_range_count = 0u;
          for (const auto &range : occupied_ranges) {
            if (merged_range_count != 0u &&
                occupied_ranges[merged_range_count - 1u].second >= range.first) {
              occupied_ranges[merged_range_count - 1u].second =
                  std::max(occupied_ranges[merged_range_count - 1u].second, range.second);
            } else {
              occupied_ranges[merged_range_count++] = range;
            }
          }
          occupied_ranges.resize(merged_range_count);
          const auto overlaps_instrumented_site = [&](uint64_t begin, uint64_t end) {
            const auto after = std::ranges::lower_bound(
                occupied_ranges, end, {}, [](const auto &range) { return range.first; });
            return after != occupied_ranges.begin() && std::prev(after)->second > begin;
          };

          std::optional<DenseEntryHost> best_host;
          uint64_t best_max_distance = std::numeric_limits<uint64_t>::max();
          auto block = std::lower_bound(
              dense_host_blocks.begin(), dense_host_blocks.end(), kernel->entry_text_offset,
              [](const std::unique_ptr<BasicBlock> &candidate, uint64_t entry) {
                return candidate->end_offset() <= entry;
              });
          for (; block != dense_host_blocks.end() && (*block)->start_offset() < kernel_end;
               ++block) {
            if ((*block)->start_offset() < kernel->entry_text_offset ||
                (*block)->end_offset() > kernel_end || (*block)->size() < kHostBytes)
              continue;
            const auto block_end = (*block)->instructions().end();
            for (auto instruction_it = (*block)->instructions().begin();
                 instruction_it != block_end; ++instruction_it) {
              const Instruction &instruction = *instruction_it;
              const uint64_t host_offset = instruction.src_loc();
              if (host_offset + kHostBytes > (*block)->end_offset() ||
                  overlaps_instrumented_site(host_offset, host_offset + kHostBytes) ||
                  group_resource == nullptr)
                continue;
              RegisterSet relocated_references;
              bool has_indirect_scalar_reference = false;
              for (auto relocated = instruction_it;
                   relocated != block_end && (*relocated).src_loc() < host_offset + kHostBytes;
                   ++relocated) {
                has_indirect_scalar_reference |= (*relocated).mnemonic().starts_with("s_movrel");
                const InstDefUse def_use(*relocated);
                relocated_references |= def_use.defs | def_use.uses;
              }
              if (has_indirect_scalar_reference ||
                  !moi_relocatable_host_preserves_transient_sgprs(host_owner_proof, **block,
                                                                  instruction, relocated_references,
                                                                  host_bootstrap_ranges)) {
                continue;
              }
              const uint64_t relay_offset = host_offset + sizeof(uint32_t);
              uint64_t max_distance = 0;
              const bool relay_reachable = std::ranges::all_of(anchors, [&](uint64_t anchor) {
                if (!compute_sopp_branch_simm16(anchor, relay_offset))
                  return false;
                max_distance =
                    std::max(max_distance,
                             anchor < relay_offset ? relay_offset - anchor : anchor - relay_offset);
                return true;
              });
              if (!relay_reachable || max_distance >= best_max_distance)
                continue;

              std::vector<uint32_t> displaced_words;
              std::vector<std::string> decode_errors;
              while (displaced_words.size() < kHostWords) {
                auto decoded = decode_relocatable_entry_instruction(
                    original_text, host_offset + displaced_words.size() * sizeof(uint32_t), arch,
                    decode_errors, relocation_decoder.get());
                if (!decoded) {
                  displaced_words.clear();
                  break;
                }
                displaced_words.insert(displaced_words.end(), decoded->begin(), decoded->end());
              }
              if (displaced_words.size() != kHostWords)
                continue;
              best_max_distance = max_distance;
              best_host = DenseEntryHost{.host_offset = host_offset,
                                         .displaced_words = std::move(displaced_words)};
            }
          }
          if (best_host) {
            island = best_host->host_offset + sizeof(uint32_t);
            dense_entry_hosts.emplace(dispatcher, std::move(*best_host));
          }
        }
      }
      if (!island)
        continue;
      for (const ConSanMoiCandidate *candidate : group.candidates) {
        dense_entry_islands.emplace(candidate, *island);
        dense_dispatchers.emplace(candidate, dispatcher);
      }
    }
  }
  if (use_indirect_appended) {
    // Every ordinary access owns an eight-word entry-island prefix. The
    // remaining per-site bank is explicit branch-relay capacity unless a
    // dense dispatcher owns that complete slice.
    for (const ConSanMoiCandidate *candidate : candidates) {
      if (dense_entry_islands.contains(candidate))
        continue;
      const uint64_t slice_begin = access_island_begin + candidate_indices.at(candidate) *
                                                             access_relay_words * sizeof(uint32_t);
      for (uint64_t word = kIndirectIslandWords; word < access_relay_words; ++word) {
        (void)branch_router.offer(slice_begin + word * sizeof(uint32_t),
                                  BranchOnlyRelayProvenance::GeneratedBank);
      }
    }
  }
  BranchOnlyDirectRelayReservoirSet direct_branch_reservoirs;
  if (has_branch_only_dynamic_assignment && use_indirect_appended && !original_text.empty()) {
    std::vector<std::pair<uint64_t, uint64_t>> protected_ranges = synchronization_ranges;
    protected_ranges.reserve(protected_ranges.size() + candidates.size() + result.kernels.size() +
                             options.preapplied_reserved_ranges.size() + dense_entry_hosts.size());
    for (const ConSanMoiCandidate *candidate : candidates)
      protected_ranges.emplace_back(candidate->text_offset,
                                    candidate->text_offset + candidate->size);
    for (const ConSanKernelInfo &kernel : result.kernels) {
      protected_ranges.emplace_back(
          kernel.entry_text_offset,
          std::min<uint64_t>(original_text.size(), kernel.entry_text_offset + 256u));
    }
    for (const ConSanPreappliedReservedRange &range : options.preapplied_reserved_ranges)
      protected_ranges.emplace_back(range.text_offset, range.text_offset + range.size);
    for (const auto &[dispatcher, host] : dense_entry_hosts) {
      (void)dispatcher;
      protected_ranges.emplace_back(
          host.host_offset, host.host_offset + host.displaced_words.size() * sizeof(uint32_t));
    }
    const uint64_t latest_source =
        std::ranges::max(candidates, {}, &ConSanMoiCandidate::text_offset)->text_offset;
    const uint64_t route_midpoint = latest_source + (indirect_body_begin - latest_source) / 2u;
    std::vector<BasicBlock *> reservoir_blocks;
    reservoir_blocks.reserve(dense_host_blocks.size());
    for (const std::unique_ptr<BasicBlock> &block : dense_host_blocks) {
      if (block)
        reservoir_blocks.push_back(block.get());
    }
    std::string relay_error;
    if (!branch_router.plan_direct_reservoirs(
            reservoir_blocks, original_text, protected_ranges, arch, route_midpoint,
            2u * candidates.size(), placement_planner, direct_branch_reservoirs, &relay_error)) {
      result.errors.emplace_back("ConSan MOI inline-shadow probe " + relay_error);
      return;
    }
  }

  MoiSpillManagers spill_managers;
  std::vector<std::pair<uint64_t, uint64_t>> planned_anchor_ranges;
  bool warned_access_relay_sync_overlap = false;
  std::unordered_map<uint64_t, uint16_t> next_compact_token_by_descriptor;
  std::unordered_set<uint64_t> selected_inline_anchor_relay_offsets;
  const auto publish_inline_anchor_relays = [&](const ConSanMoiCandidate &candidate,
                                                uint32_t original_size) {
    if (original_size <= sizeof(uint32_t))
      return;
    branch_router.retire_range(candidate.text_offset, original_size);
    for (uint64_t offset = candidate.text_offset + sizeof(uint32_t);
         offset < candidate.text_offset + original_size; offset += sizeof(uint32_t)) {
      if (selected_inline_anchor_relay_offsets.insert(offset).second)
        (void)branch_router.offer(offset, BranchOnlyRelayProvenance::OwnedAnchor);
    }
  };
  const auto uses_branch_only_dynamic_stack = [&](const ConSanMoiCandidate *candidate) {
    const ConSanCandidateResourcePlan *plan = resource_plan_for_candidate(result, *candidate);
    if (plan == nullptr || plan->owner_descriptor_file_offsets.empty())
      return false;
    return std::ranges::all_of(
        plan->owner_descriptor_file_offsets, [&](uint64_t descriptor_offset) {
          const auto *assignment = find_moi_transient_sgpr_assignment(options, descriptor_offset);
          return assignment != nullptr && assignment->dynamic_stack_borrowed_sgpr.has_value();
        });
  };
  std::vector<const ConSanMoiCandidate *> planning_candidates = candidates;
  if (has_branch_only_dynamic_assignment && options.max_patches >= candidates.size()) {
    // Plan branch-only sites nearest appended text first. Their unconditional
    // source branch makes every remaining original-instruction word a proven
    // capacity-one relay host. Publishing those hosts before planning earlier
    // sites avoids depending on padding or workload-specific NOP caves while
    // preserving ordinary candidate order when a caller requests a subset.
    std::ranges::stable_sort(planning_candidates,
                             [&](const ConSanMoiCandidate *lhs, const ConSanMoiCandidate *rhs) {
                               const bool lhs_branch_only = uses_branch_only_dynamic_stack(lhs);
                               const bool rhs_branch_only = uses_branch_only_dynamic_stack(rhs);
                               if (lhs_branch_only != rhs_branch_only)
                                 return lhs_branch_only;
                               if (lhs_branch_only)
                                 return lhs->text_offset > rhs->text_offset;
                               return lhs->text_offset < rhs->text_offset;
                             });
  }
  for (const ConSanMoiCandidate *candidate_ptr : planning_candidates) {
    const ConSanMoiCandidate &candidate = *candidate_ptr;
    const bool use_dense_call = dense_entry_islands.contains(candidate_ptr);
    const uint64_t candidate_end = candidate.text_offset + candidate.size;
    if (use_indirect_appended && !use_dense_call &&
        std::ranges::any_of(planned_anchor_ranges, [&](const auto &range) {
          return candidate.text_offset < range.second && range.first < candidate_end;
        })) {
      result.warnings.emplace_back(
          "ConSan MOI inline-shadow probe skipped access site inside a relocated prefix");
      continue;
    }
    const uint16_t scratch_count = inline_shadow_scratch_count(options, candidate, arch);
    const auto resources = resolve_moi_scratch(result, candidate, options, scratch_count);
    if (!resources)
      continue;
    const std::optional<uint64_t> direct_descriptor =
        resources->owner_descriptor_file_offsets.size() == 1u
            ? std::optional<uint64_t>(resources->owner_descriptor_file_offsets.front())
            : std::nullopt;
    ConSanOptions candidate_options;
    bool persistent_assignment_applied = false;
    if (direct_descriptor) {
      auto [cached, inserted] =
          direct_options_by_descriptor.try_emplace(*direct_descriptor, std::nullopt);
      if (inserted) {
        ConSanOptions resolved = options;
        if (apply_moi_transient_sgpr_assignment(resolved,
                                                resources->owner_descriptor_file_offsets) &&
            (resolved.moi_persistent_vgpr_assignments.empty() ||
             apply_moi_persistent_vgpr_assignment(resolved,
                                                  resources->owner_descriptor_file_offsets))) {
          // The descriptor-specific scalar and persistent choices are now in
          // the scalar option fields.  Dropping the object-wide assignment
          // vectors keeps the per-site copy constant-size in dense objects.
          resolved.moi_transient_sgpr_assignments.clear();
          resolved.moi_persistent_vgpr_assignments.clear();
          resolved.moi_prologue_scratch_vgpr_assignments.clear();
          cached->second = std::move(resolved);
        }
      }
      if (!cached->second)
        continue;
      candidate_options = *cached->second;
      persistent_assignment_applied = true;
    } else {
      candidate_options = options;
      if (!apply_moi_transient_sgpr_assignment(candidate_options,
                                               resources->owner_descriptor_file_offsets))
        continue;
    }

    std::optional<ConSanMoiWorkgroupShadowLayout> workgroup_shadow;
    uint16_t workgroup_shadow_compact_token = 0u;
    const bool has_unspilled_component_override = std::ranges::any_of(
        resources->owner_descriptor_file_offsets, [&](uint64_t descriptor_offset) {
          const auto *assignment = find_moi_transient_sgpr_assignment(options, descriptor_offset);
          return assignment != nullptr && !assignment->spill_backed;
        });
    // The spill-backed scalar window is preserved only after the access body
    // establishes its private frame. Kernel-entry local-shadow initialization
    // cannot borrow that guest window. Keep the semantically equivalent
    // external exact-shadow table for every spill-backed component, whether
    // or not it also found a fresh visible-evidence latch.
    const bool spill_backed_scalar_state = candidate_options.automatic_moi_inline_sgpr_spill;
    if (options.moi_inline_workgroup_shadow && !has_unspilled_component_override &&
        !spill_backed_scalar_state) {
      uint16_t next_compact_token = 0u;
      if (candidate.in_kernel && candidate.kernel_descriptor_file_offset) {
        uint16_t &next = next_compact_token_by_descriptor[*candidate.kernel_descriptor_file_offset];
        if (next == 0u)
          next = 1u;
        next_compact_token = next;
      }
      const bool compact_token_available =
          direct_descriptor && next_compact_token != 0u &&
          next_compact_token <= consan_moi_exact_shadow::max_compact_token &&
          candidate_count_by_descriptor[*direct_descriptor] <=
              consan_moi_exact_shadow::max_compact_token;
      const bool has_dynamic_lds_owner = std::ranges::any_of(
          resources->owner_descriptor_file_offsets, [&](uint64_t descriptor_offset) {
            return dynamic_lds_descriptors.contains(descriptor_offset);
          });
      if (!has_dynamic_lds_owner) {
        if (direct_descriptor) {
          auto [cached, inserted] =
              direct_workgroup_shadow_by_descriptor.try_emplace(*direct_descriptor, std::nullopt);
          if (inserted) {
            cached->second = common_moi_workgroup_shadow_layout(
                bytes, *resources, arch, options, result.warnings, compact_token_available);
          }
          workgroup_shadow = cached->second;
        } else {
          workgroup_shadow = common_moi_workgroup_shadow_layout(
              bytes, *resources, arch, options, result.warnings, compact_token_available);
        }
      }
      if (has_dynamic_lds_owner && !warned_dynamic_lds_external_shadow) {
        result.warnings.emplace_back(
            "ConSan MOI inline-shadow dynamic-LDS owner uses the external exact-shadow table");
        warned_dynamic_lds_external_shadow = true;
      } else if (!workgroup_shadow) {
        result.warnings.emplace_back(
            "ConSan MOI inline-shadow access falls back to the external exact-shadow table");
      } else if (workgroup_shadow->compact) {
        workgroup_shadow_compact_token = next_compact_token;
      }
    }

    std::optional<MoiPrivateEpochLayout> private_layout;
    std::optional<uint16_t> private_owner_shift;
    if (options.automatic_moi_private_epoch) {
      const bool persist_workitem_owner =
          options.moi_owner_source == ConSanMoiOwnerSource::WorkitemId;
      // Every supported target exposes workgroup identity as an entry ABI
      // value. Preserve the complete key before arbitrary guest code can reuse
      // the ordinary SGPR or TTMP inputs.
      const bool persist_workgroup_key = true;
      if (direct_descriptor) {
        auto [cached, inserted] =
            direct_private_layout_by_descriptor.try_emplace(*direct_descriptor, std::nullopt);
        if (inserted) {
          cached->second = build_moi_private_epoch_layout(result, *resources, arch, result.warnings,
                                                          /*include_owner=*/persist_workitem_owner,
                                                          /*include_workgroup_key=*/
                                                          persist_workgroup_key);
        }
        private_layout = cached->second;
      } else {
        private_layout = build_moi_private_epoch_layout(result, *resources, arch, result.warnings,
                                                        /*include_owner=*/persist_workitem_owner,
                                                        /*include_workgroup_key=*/
                                                        persist_workgroup_key);
      }
      if (!private_layout)
        continue;
      if (options.moi_owner_source == ConSanMoiOwnerSource::WorkitemId) {
        if (direct_descriptor) {
          auto [cached, inserted] =
              direct_owner_shift_by_descriptor.try_emplace(*direct_descriptor, std::nullopt);
          if (inserted) {
            cached->second = common_moi_private_owner_shift(bytes, *resources, result.warnings);
          }
          private_owner_shift = cached->second;
        } else {
          private_owner_shift = common_moi_private_owner_shift(bytes, *resources, result.warnings);
        }
        if (!private_owner_shift)
          continue;
      }
    }

    if (!persistent_assignment_applied &&
        !candidate_options.moi_persistent_vgpr_assignments.empty() &&
        !apply_moi_persistent_vgpr_assignment(candidate_options,
                                              resources->owner_descriptor_file_offsets)) {
      result.warnings.emplace_back(
          "ConSan MOI inline-shadow access has no common persistent VGPR assignment");
      continue;
    }

    std::vector<std::string> workgroup_errors;
    std::optional<ConSanMoiWorkgroupSources> workgroup_sources;
    const bool has_persistent_workgroup_key =
        (private_layout && private_layout->workgroup_key_offset) ||
        candidate_options.moi_workgroup_key_vgpr ||
        candidate_options.moi_persistent_workgroup_key_sgpr;
    if (has_persistent_workgroup_key) {
      // The owner-specific entry prologue captured the descriptor-valid
      // coordinates. Shared physical text can therefore consume one stable
      // key even when its owning descriptors request different coordinate
      // subsets.
      workgroup_sources = ConSanMoiWorkgroupSources{};
    } else {
      std::optional<uint64_t> workgroup_descriptor = direct_descriptor;
      if (!workgroup_descriptor) {
        workgroup_descriptor =
            common_moi_record_owner_descriptor(bytes, *resources, arch, result.warnings);
      }
      if (!workgroup_descriptor)
        continue;
      if (direct_descriptor) {
        auto [cached, inserted] =
            direct_workgroup_sources_by_descriptor.try_emplace(*direct_descriptor, std::nullopt);
        if (inserted) {
          cached->second =
              moi_descriptor_workgroup_sources(bytes, *direct_descriptor, arch, workgroup_errors);
        }
        workgroup_sources = cached->second;
      } else {
        workgroup_sources =
            moi_descriptor_workgroup_sources(bytes, *workgroup_descriptor, arch, workgroup_errors);
      }
    }
    if (!workgroup_sources) {
      result.warnings.insert(result.warnings.end(), workgroup_errors.begin(),
                             workgroup_errors.end());
      continue;
    }

    std::optional<VgprSpillSequence> spill;
    if (resources->source == ConSanRegisterAllocationSource::SpillRequired ||
        moi_scalar_spill_requires_dynamic_vgpr_frame(result, *resources, candidate_options)) {
      if (candidate_options.moi_inline_dynamic_stack_borrowed_sgpr) {
        constexpr uint16_t reservoir_count =
            DynamicStackBorrowedSgprSpillSequence::kScalarReservoirCount;
        const uint32_t scalar_frame_bytes =
            static_cast<uint32_t>(moi_exec_save_sgpr_count(candidate_options, arch)) *
            SpillManager::kSlotBytes;
        if (resources->count < reservoir_count) {
          result.warnings.emplace_back(
              "ConSan MOI dynamic-stack bootstrap has no scalar reservoir window");
          continue;
        }
        const uint16_t reservoir_base =
            static_cast<uint16_t>(resources->base + resources->count - reservoir_count);
        const auto borrowed = build_dynamic_stack_borrowed_sgpr_spill_sequence(
            resources->base, resources->count,
            *candidate_options.moi_inline_dynamic_stack_borrowed_sgpr, reservoir_base,
            resources->original_private_segment_size, arch, scalar_frame_bytes);
        if (!borrowed) {
          result.warnings.emplace_back(
              "ConSan MOI could not encode the branch-only dynamic-stack bootstrap");
          continue;
        }
        spill = borrowed->as_vgpr_spill_sequence();
      } else {
        spill = build_moi_spill_sequence(
            result, *resources, candidate_options, spill_managers, arch, result.warnings,
            private_layout ? std::optional<uint32_t>(private_layout->ephemeral_base)
                           : std::nullopt);
      }
      if (!spill)
        continue;
    }
    std::optional<InlineSgprSpillSequence> scalar_spill;
    if (candidate_options.automatic_moi_inline_sgpr_spill) {
      scalar_spill = build_inline_sgpr_spill_sequence(
          result, *resources, candidate_options, spill_managers, arch, result.warnings,
          private_layout ? std::optional<uint32_t>(private_layout->ephemeral_base) : std::nullopt,
          spill ? &*spill : nullptr);
      if (!scalar_spill)
        continue;
    }

    candidate_options.moi_inline_workgroup_shadow = workgroup_shadow.has_value();
    candidate_options.scratch_vgpr = resources->base;
    candidate_options.moi_spill_overlaps_guest_operands = spill.has_value();
    std::vector<std::string> candidate_errors;
    auto words = build_inline_shadow_words(
        bytes, candidate, candidate_options, arch, layout, *workgroup_sources, workgroup_shadow,
        private_layout ? std::optional<uint32_t>(private_layout->epoch_offset) : std::nullopt,
        private_layout ? private_layout->owner_offset : std::nullopt, private_owner_shift,
        private_layout ? private_layout->workgroup_key_offset : std::nullopt,
        workgroup_shadow_compact_token, candidate_errors);
    if (!words) {
      result.warnings.insert(result.warnings.end(), candidate_errors.begin(),
                             candidate_errors.end());
      continue;
    }

    const uint64_t spill_bytes =
        spill ? static_cast<uint64_t>((spill->save_words.size() + spill->restore_words.size()) *
                                      sizeof(uint32_t))
              : 0;
    const uint64_t scalar_spill_bytes =
        scalar_spill ? static_cast<uint64_t>(scalar_spill->save_words.size() +
                                             scalar_spill->restore_words.size()) *
                           sizeof(uint32_t)
                     : 0;
    const uint64_t vgpr_msb_bytes =
        arch == ROCJITSU_CODE_ARCH_GFX1250 && candidate.gfx1250_vgpr_msb_mode.value_or(0u) != 0u
            ? (spill ? 4u : 2u) * sizeof(uint32_t)
            : 0u;
    const uint32_t required_private_bytes =
        std::max({private_layout ? private_layout->ephemeral_base : 0u,
                  spill ? spill->total_private_bytes : 0u,
                  scalar_spill ? scalar_spill->total_private_bytes : 0u});
    const uint32_t available_padding =
        count_nop_padding(bytes, candidate.file_offset + candidate.size, arch);
    const uint64_t available_bytes =
        candidate.size + static_cast<uint64_t>(available_padding) * sizeof(uint32_t);
    std::optional<DbiPatchPlacement> placement;
    std::optional<uint64_t> entry_island_offset;
    bool entry_island_at_anchor = false;
    std::vector<uint32_t> displaced_tail_words;
    bool branch_only_continuation = false;
    std::vector<uint64_t> entry_relay_offsets;
    std::vector<uint64_t> return_relay_offsets;
    std::optional<BranchOnlyRelayRoute> branch_only_route;
    uint32_t displaced_original_size = candidate.size;
    bool placed_through_indirect_island = false;
    if (use_indirect_appended && candidate_options.moi_inline_dynamic_stack_borrowed_sgpr) {
      const uint64_t patch_bytes = static_cast<uint64_t>(words->size() * sizeof(uint32_t)) +
                                   spill_bytes + scalar_spill_bytes + vgpr_msb_bytes +
                                   sizeof(uint32_t);
      DbiPatchPlacementPlanner tentative_planner = placement_planner;
      const auto reservation = tentative_planner.plan_indirect_appended(
          candidate.text_offset, candidate.size, patch_bytes + sizeof(uint32_t));
      if (reservation) {
        std::string relay_error;
        branch_only_route = branch_router.plan_pair(
            tentative_planner, candidate.text_offset, reservation->body_offset,
            reservation->body_offset + patch_bytes, candidate.text_offset + candidate.size,
            &relay_error);
        if (branch_only_route && branch_router.commit(*branch_only_route, &relay_error)) {
          entry_relay_offsets = branch_only_route->entry_relay_offsets;
          return_relay_offsets = branch_only_route->return_relay_offsets;
          for (const BranchOnlyRelayClaim &claim : branch_only_route->claims) {
            if (local_entry_islands && claim.provenance == BranchOnlyRelayProvenance::PristineNop)
              local_entry_islands->reserve(claim.offset, 1u);
          }
          if (!direct_branch_reservoirs.mark_claims_used(branch_only_route->claims, &relay_error)) {
            result.errors.emplace_back("ConSan MOI inline-shadow probe " + relay_error);
            return;
          }
          placement_planner = std::move(tentative_planner);
          placement = *reservation;
          placement->body_size = patch_bytes;
          placement->return_target = candidate.text_offset + candidate.size;
          branch_only_continuation = true;
          placed_through_indirect_island = true;
        }
      }
    }
    if (use_indirect_appended && !placed_through_indirect_island) {
      if (candidate_options.moi_inline_dynamic_stack_borrowed_sgpr) {
        result.warnings.emplace_back(
            "ConSan MOI inline-shadow probe could not route its branch-only dynamic-stack body");
        continue;
      }
      std::optional<uint64_t> candidate_entry_island_offset;
      if (use_dense_call) {
        candidate_entry_island_offset = dense_entry_islands.at(candidate_ptr);
      } else {
        candidate_entry_island_offset = access_island_begin + candidate_indices.at(candidate_ptr) *
                                                                  access_relay_words *
                                                                  sizeof(uint32_t);
      }
      bool entry_island_reachable =
          candidate.size >= sizeof(uint32_t) &&
          compute_sopp_branch_simm16(candidate.text_offset, *candidate_entry_island_offset);
      if (!use_dense_call && !entry_island_reachable) {
        if (const auto local_island =
                local_entry_islands ? local_entry_islands->claim_reachable(candidate.text_offset)
                                    : std::nullopt) {
          candidate_entry_island_offset = local_island;
          entry_island_reachable = true;
          retire_branch_relay_range(*local_island, kIndirectIslandWords);
        }
      }
      if (!use_dense_call && !entry_island_reachable && !original_text.empty()) {
        std::vector<uint32_t> displaced_words;
        while (displaced_words.size() < kIndirectIslandWords) {
          auto decoded = decode_relocatable_entry_instruction(
              original_text, candidate.text_offset + displaced_words.size() * sizeof(uint32_t),
              arch, candidate_errors, relocation_decoder.get());
          if (!decoded) {
            displaced_words.clear();
            break;
          }
          displaced_words.insert(displaced_words.end(), decoded->begin(), decoded->end());
        }
        const uint64_t displaced_end =
            candidate.text_offset + displaced_words.size() * sizeof(uint32_t);
        if (displaced_words.size() >= kIndirectIslandWords &&
            displaced_words.size() * sizeof(uint32_t) <= std::numeric_limits<uint32_t>::max() &&
            !relocation_overlaps_synchronization(candidate.text_offset + candidate.size,
                                                 displaced_end)) {
          const size_t candidate_words = candidate.size / sizeof(uint32_t);
          if (candidate_words <= displaced_words.size()) {
            displaced_tail_words.assign(displaced_words.begin() + candidate_words,
                                        displaced_words.end());
            displaced_original_size =
                static_cast<uint32_t>(displaced_words.size() * sizeof(uint32_t));
            candidate_entry_island_offset = candidate.text_offset;
            entry_island_reachable = true;
            entry_island_at_anchor = true;
          }
        } else if (!warned_access_relay_sync_overlap &&
                   displaced_words.size() >= kIndirectIslandWords &&
                   relocation_overlaps_synchronization(candidate.text_offset + candidate.size,
                                                       displaced_end)) {
          result.warnings.emplace_back(
              "ConSan MOI inline-shadow probe did not relocate a synchronization site through "
              "an uninstrumented access entry relay");
          warned_access_relay_sync_overlap = true;
        }
      }
      if (entry_island_reachable) {
        entry_island_offset = *candidate_entry_island_offset;
        const uint64_t patch_bytes =
            static_cast<uint64_t>((words->size() + displaced_tail_words.size()) *
                                  sizeof(uint32_t)) +
            spill_bytes + scalar_spill_bytes + vgpr_msb_bytes;
        // The long return is getpc, a three-word signed delta builder, SCC
        // restore, a scalar-destination dependency wait, and setpc. The fixed
        // reservation keeps later body addresses stable while their
        // instruction words are regenerated during emission.
        const uint64_t return_bytes =
            (use_dense_call && arch == ROCJITSU_CODE_ARCH_GFX1250 ? 1u : 7u) * sizeof(uint32_t);
        std::string placement_error;
        const auto reservation =
            placement_planner.plan_indirect_appended(candidate.text_offset, displaced_original_size,
                                                     patch_bytes + return_bytes, &placement_error);
        if (reservation) {
          placement = *reservation;
          placement->body_size = patch_bytes;
          placement->return_target = candidate.text_offset + displaced_original_size;
          placed_through_indirect_island = true;
        }
      }
    }
    if (!placed_through_indirect_island) {
      const uint64_t patch_bytes = static_cast<uint64_t>(words->size() * sizeof(uint32_t)) +
                                   spill_bytes + scalar_spill_bytes + vgpr_msb_bytes;
      DbiPatchPlacementRequest placement_request;
      placement_request.anchor_offset = candidate.text_offset;
      placement_request.original_size = candidate.size;
      placement_request.body_size = patch_bytes;
      placement_request.inline_capacity =
          spill || scalar_spill || private_layout || workgroup_shadow ? 0u : available_bytes;
      std::string placement_error;
      placement = placement_planner.plan(placement_request, &placement_error);
      if (!placement) {
        result.warnings.emplace_back("ConSan MOI inline-shadow probe skipped access site: " +
                                     placement_error);
        continue;
      }
    }

    PlannedInlineShadowPatch planned;
    planned.candidate = candidate_ptr;
    planned.placement = *placement;
    planned.entry_island_offset = entry_island_offset;
    planned.entry_island_at_anchor = entry_island_at_anchor;
    planned.dense_call_anchor = use_dense_call;
    planned.branch_only_continuation = branch_only_continuation;
    planned.entry_relay_offsets = std::move(entry_relay_offsets);
    planned.return_relay_offsets = std::move(return_relay_offsets);
    planned.branch_only_route = std::move(branch_only_route);
    if (use_dense_call)
      planned.dense_dispatcher_offset = dense_dispatchers.at(candidate_ptr);
    planned.displaced_tail_words = std::move(displaced_tail_words);
    planned.resources = *resources;
    planned.workgroup_sources = *workgroup_sources;
    planned.spill = spill;
    planned.scalar_spill = scalar_spill;
    if (private_layout)
      planned.private_epoch_offset = private_layout->epoch_offset;
    if (private_layout)
      planned.private_owner_offset = private_layout->owner_offset;
    if (private_layout)
      planned.private_workgroup_key_offset = private_layout->workgroup_key_offset;
    planned.private_owner_shift = private_owner_shift;
    planned.workgroup_shadow = workgroup_shadow;
    planned.workgroup_shadow_compact_token = workgroup_shadow_compact_token;
    planned.required_private_bytes = required_private_bytes;
    planned_patches.push_back(std::move(planned));
    if (branch_only_continuation)
      publish_inline_anchor_relays(candidate, placement->original_size);
    if (use_indirect_appended && !use_dense_call)
      planned_anchor_ranges.emplace_back(candidate.text_offset,
                                         candidate.text_offset + placement->original_size);
    if (workgroup_shadow && workgroup_shadow->compact && candidate.kernel_descriptor_file_offset) {
      ++next_compact_token_by_descriptor[*candidate.kernel_descriptor_file_offset];
    }
    if (planned_patches.size() == options.max_patches)
      break;
  }

  for (auto &[dispatcher, host] : dense_entry_hosts) {
    (void)dispatcher;
    host.body_offset = placement_planner.appended_end();
    std::string reservation_error;
    if (!placement_planner.reserve_appended_prefix(
            (host.displaced_words.size() + kIndirectIslandWords) * sizeof(uint32_t),
            &reservation_error)) {
      result.errors.emplace_back("ConSan MOI inline-shadow probe could not reserve a dense "
                                 "entry-host body: " +
                                 reservation_error);
      return;
    }
  }

  if (planned_patches.empty())
    return;

  std::unordered_map<uint64_t, size_t> inline_anchor_relay_donors;
  for (size_t index = 0; index < planned_patches.size(); ++index) {
    const PlannedInlineShadowPatch &planned = planned_patches[index];
    for (uint64_t offset = planned.candidate->text_offset + sizeof(uint32_t);
         offset < planned.candidate->text_offset + planned.placement.original_size;
         offset += sizeof(uint32_t)) {
      if (selected_inline_anchor_relay_offsets.contains(offset))
        inline_anchor_relay_donors.emplace(offset, index);
    }
  }
  for (size_t index = 0; index < planned_patches.size(); ++index) {
    const PlannedInlineShadowPatch &planned = planned_patches[index];
    const auto donor_precedes_borrower = [&](uint64_t relay) {
      if (!selected_inline_anchor_relay_offsets.contains(relay))
        return true;
      const auto donor = inline_anchor_relay_donors.find(relay);
      return donor != inline_anchor_relay_donors.end() && donor->second < index;
    };
    if (!std::ranges::all_of(planned.entry_relay_offsets, donor_precedes_borrower) ||
        !std::ranges::all_of(planned.return_relay_offsets, donor_precedes_borrower)) {
      result.errors.emplace_back(
          "ConSan MOI inline-shadow probe planned an anchor-tail relay before its donor");
      return;
    }
  }

  MoiDescriptorVgprRequirements descriptor_requirements;
  MoiDescriptorSgprRequirements scalar_requirements;
  MoiDescriptorPrivateRequirements private_requirements;
  MoiDescriptorLdsRequirements lds_requirements;
  for (const PlannedInlineShadowPatch &planned_patch : planned_patches) {
    note_descriptor_requirements(descriptor_requirements, planned_patch.resources);
    ConSanOptions candidate_options = options;
    if (!apply_moi_transient_sgpr_assignment(candidate_options,
                                             planned_patch.resources.owner_descriptor_file_offsets))
      continue;
    note_moi_sgpr_requirements(scalar_requirements, planned_patch.resources, candidate_options,
                               arch);
    if (planned_patch.workgroup_shadow) {
      note_moi_lds_requirements(lds_requirements, planned_patch.resources,
                                *planned_patch.workgroup_shadow);
    }
    if (planned_patch.required_private_bytes != 0) {
      for (uint64_t descriptor_offset : planned_patch.resources.owner_descriptor_file_offsets) {
        auto [it, inserted] =
            private_requirements.emplace(descriptor_offset, planned_patch.required_private_bytes);
        if (!inserted)
          it->second = std::max(it->second, planned_patch.required_private_bytes);
      }
    }
  }

  const bool uses_appended_cave =
      std::ranges::any_of(planned_patches, [](const PlannedInlineShadowPatch &patch) {
        return patch.placement.kind == DbiPatchPlacementKind::AppendedCave;
      });
  if (uses_appended_cave) {
    CodeObjectPatcher patcher(code_object);
    if (!apply_descriptor_requirements(patcher, code_object, bytes, result, descriptor_requirements,
                                       arch, result.errors))
      return;
    if (!apply_spill_descriptor_requirements(patcher, code_object, bytes, result,
                                             private_requirements, result.errors))
      return;
    if (!apply_sgpr_descriptor_requirements(patcher, result, code_object, scalar_requirements,
                                            result.errors))
      return;
    if (!apply_lds_descriptor_requirements(patcher, result, code_object, lds_requirements, options,
                                           arch, result.errors))
      return;
    const std::span<const uint8_t> old_text = patcher.text_bytes();
    if (old_text.empty()) {
      result.errors.emplace_back("ConSan MOI inline-shadow probe found no .text section");
      return;
    }
    std::vector<uint8_t> new_text(old_text.begin(), old_text.end());
    if (use_indirect_appended) {
      while (new_text.size() < indirect_body_begin)
        append_word_bytes(new_text, build_s_nop(0, arch));
    }
    std::vector<ConSanPatchInfo> patches;
    patches.reserve(planned_patches.size());
    for (const BranchOnlyDirectRelayReservoir &reservoir : direct_branch_reservoirs.reservoirs) {
      std::string relay_error;
      if (!BranchOnlyRelayRouter::emit_direct_reservoir(new_text, reservoir, arch, patches,
                                                        &relay_error)) {
        result.errors.emplace_back("ConSan MOI inline-shadow probe " + relay_error);
        return;
      }
    }
    std::map<uint64_t, std::vector<const PlannedInlineShadowPatch *>> dense_groups;
    for (const PlannedInlineShadowPatch &planned_patch : planned_patches) {
      if (planned_patch.dense_call_anchor)
        dense_groups[*planned_patch.dense_dispatcher_offset].push_back(&planned_patch);
    }
    for (const auto &[dispatcher_offset, group] : dense_groups) {
      if (group.empty() || !group.front()->entry_island_offset)
        continue;
      ConSanOptions group_options = options;
      if (!apply_moi_transient_sgpr_assignment(
              group_options, group.front()->resources.owner_descriptor_file_offsets))
        continue;
      const auto scalar_abi = moi_inline_dense_router_scalar_abi(group_options, arch);
      if (!scalar_abi)
        continue;
      const bool explicit_dense_key = !scalar_abi->call_return_sgpr.has_value();
      const uint16_t key_sgpr = scalar_abi->dispatch_key_sgpr;
      const std::optional<uint16_t> call_return_sgpr = scalar_abi->call_return_sgpr;
      const uint64_t island_offset = *group.front()->entry_island_offset;
      const MoiIndirectJumpSgprs jump_sgprs = scalar_abi->indirect_jump;

      std::vector<uint32_t> island_words;
      if (!append_moi_scc_preserving_indirect_jump(island_words, island_offset, dispatcher_offset,
                                                   jump_sgprs.pc_sgpr, jump_sgprs.scc_save_sgpr,
                                                   /*capture_scc=*/true, arch) ||
          island_words.empty()) {
        result.errors.emplace_back(
            "ConSan MOI inline-shadow probe could not encode its dense entry island");
        return;
      }
      const auto wait_pc = instrumentation::build_s_wait_indirect_pc0(arch);
      if (!wait_pc) {
        result.errors.emplace_back(
            "ConSan MOI inline-shadow probe could not encode its dense entry dependency wait");
        return;
      }
      island_words.insert(island_words.end() - 1, *wait_pc);
      if (island_words.size() != kIndirectIslandWords) {
        result.errors.emplace_back(
            "ConSan MOI inline-shadow probe encoded an invalid dense entry island size");
        return;
      }
      const uint64_t island_bytes = island_words.size() * sizeof(uint32_t);
      if (island_offset > new_text.size() || island_bytes > new_text.size() - island_offset) {
        result.errors.emplace_back(
            "ConSan MOI inline-shadow probe dense entry island exceeds .text");
        return;
      }
      std::memcpy(new_text.data() + island_offset, island_words.data(),
                  static_cast<size_t>(island_bytes));
      ConSanPatchInfo island_info;
      island_info.kind = ConSanPatchKind::TrampolineMoiIndirectBranchIsland;
      const auto entry_host = dense_entry_hosts.find(dispatcher_offset);
      if (entry_host != dense_entry_hosts.end()) {
        island_info.anchor_offset = entry_host->second.host_offset;
        island_info.original_size =
            static_cast<uint32_t>(entry_host->second.displaced_words.size() * sizeof(uint32_t));
        island_info.trampoline_offset = entry_host->second.body_offset;
        island_info.trampoline_size = static_cast<uint32_t>(
            (entry_host->second.displaced_words.size() + kIndirectIslandWords) * sizeof(uint32_t));
      } else {
        island_info.anchor_offset = group.front()->candidate->text_offset;
        island_info.trampoline_offset = island_offset;
        island_info.trampoline_size = static_cast<uint32_t>(island_bytes);
      }
      island_info.owner_descriptor_file_offsets =
          group.front()->resources.owner_descriptor_file_offsets;
      patches.push_back(std::move(island_info));

      std::vector<uint32_t> dispatcher_words;
      if (!explicit_dense_key) {
        const auto subtract_pc =
            instrumentation::build_s_sub_u32(key_sgpr, *call_return_sgpr, jump_sgprs.pc_sgpr, arch);
        if (!subtract_pc) {
          result.errors.emplace_back(
              "ConSan MOI inline-shadow probe could not encode its dense dispatch key");
          return;
        }
        dispatcher_words.push_back(*subtract_pc);
        dispatcher_words.push_back(build_s_add_u32(key_sgpr, key_sgpr, /*literal=*/255u, arch));
        dispatcher_words.push_back(static_cast<uint32_t>(dispatcher_offset - island_offset));
      }

      if (entry_host != dense_entry_hosts.end()) {
        const auto compare_entry =
            instrumentation::build_s_cmp_eq_u32(key_sgpr, scalar_positive_inline_u32(0), arch);
        if (!compare_entry) {
          result.errors.emplace_back(
              "ConSan MOI inline-shadow probe could not encode its dense entry-host comparison");
          return;
        }
        dispatcher_words.push_back(*compare_entry);
        std::vector<uint32_t> matched_words;
        const uint64_t matched_offset =
            dispatcher_offset + (dispatcher_words.size() + 1u) * sizeof(uint32_t);
        const auto direct_entry =
            explicit_dense_key
                ? compute_sopp_branch_simm16(matched_offset, entry_host->second.body_offset)
                : std::nullopt;
        if (direct_entry) {
          matched_words.push_back(build_s_branch(*direct_entry, arch));
        } else if (!append_moi_scc_preserving_indirect_jump(
                       matched_words, matched_offset, entry_host->second.body_offset,
                       jump_sgprs.pc_sgpr, jump_sgprs.scc_save_sgpr,
                       /*capture_scc=*/false, arch)) {
          result.errors.emplace_back(
              "ConSan MOI inline-shadow probe could not encode its dense entry-host target");
          return;
        }
        const auto skip =
            instrumentation::build_s_cbranch_scc0(static_cast<int16_t>(matched_words.size()), arch);
        if (!skip) {
          result.errors.emplace_back(
              "ConSan MOI inline-shadow probe could not encode its dense entry-host branch");
          return;
        }
        dispatcher_words.push_back(*skip);
        dispatcher_words.insert(dispatcher_words.end(), matched_words.begin(), matched_words.end());
      }

      for (size_t group_index = 0; group_index < group.size(); ++group_index) {
        const PlannedInlineShadowPatch *planned_patch = group[group_index];
        const ConSanMoiCandidate &candidate = *planned_patch->candidate;
        const auto compare = instrumentation::build_s_cmp_eq_u32(
            key_sgpr,
            explicit_dense_key ? scalar_positive_inline_u32(static_cast<uint16_t>(group_index + 1u))
                               : 255u,
            arch);
        if (!compare) {
          result.errors.emplace_back(
              "ConSan MOI inline-shadow probe could not encode its dense dispatch comparison");
          return;
        }
        dispatcher_words.push_back(*compare);
        if (!explicit_dense_key) {
          dispatcher_words.push_back(
              static_cast<uint32_t>(candidate.text_offset + sizeof(uint32_t) - island_offset));
        }

        std::vector<uint32_t> matched_words;
        if (!explicit_dense_key && candidate.size > sizeof(uint32_t)) {
          const uint32_t adjustment = candidate.size - sizeof(uint32_t);
          matched_words.push_back(build_s_add_u32(*call_return_sgpr, *call_return_sgpr,
                                                  /*literal=*/255u, arch));
          matched_words.push_back(adjustment);
          matched_words.push_back(build_s_addc_u32(static_cast<uint16_t>(*call_return_sgpr + 1u),
                                                   static_cast<uint16_t>(*call_return_sgpr + 1u),
                                                   scalar_positive_inline_u32(0), arch));
        }
        const uint64_t matched_offset =
            dispatcher_offset + (dispatcher_words.size() + 1u) * sizeof(uint32_t);
        const auto direct_body =
            explicit_dense_key
                ? compute_sopp_branch_simm16(matched_offset, planned_patch->placement.body_offset)
                : std::nullopt;
        if (direct_body) {
          matched_words.push_back(build_s_branch(*direct_body, arch));
        } else if (!append_moi_scc_preserving_indirect_jump(
                       matched_words, matched_offset, planned_patch->placement.body_offset,
                       jump_sgprs.pc_sgpr, jump_sgprs.scc_save_sgpr,
                       /*capture_scc=*/false, arch)) {
          result.errors.emplace_back(
              "ConSan MOI inline-shadow probe could not encode a dense dispatch target");
          return;
        }
        if (matched_words.size() > static_cast<size_t>(std::numeric_limits<int16_t>::max())) {
          result.errors.emplace_back(
              "ConSan MOI inline-shadow probe dense dispatch stub is too large");
          return;
        }
        const auto skip =
            instrumentation::build_s_cbranch_scc0(static_cast<int16_t>(matched_words.size()), arch);
        if (!skip) {
          result.errors.emplace_back(
              "ConSan MOI inline-shadow probe could not encode a dense dispatch branch");
          return;
        }
        dispatcher_words.push_back(*skip);
        dispatcher_words.insert(dispatcher_words.end(), matched_words.begin(), matched_words.end());
      }
      dispatcher_words.push_back(build_s_endpgm(arch));
      const uint64_t reserved_words = group.size() * access_relay_words;
      if (dispatcher_words.size() > reserved_words || dispatcher_offset > new_text.size() ||
          dispatcher_words.size() * sizeof(uint32_t) > new_text.size() - dispatcher_offset) {
        result.errors.emplace_back(
            "ConSan MOI inline-shadow probe dense dispatcher exceeds its reserved relay space");
        return;
      }
      std::memcpy(new_text.data() + dispatcher_offset, dispatcher_words.data(),
                  dispatcher_words.size() * sizeof(uint32_t));
      ConSanPatchInfo dispatcher_info;
      dispatcher_info.kind = ConSanPatchKind::TrampolineMoiIndirectBranchIsland;
      dispatcher_info.anchor_offset = group.front()->candidate->text_offset;
      dispatcher_info.trampoline_offset = dispatcher_offset;
      dispatcher_info.trampoline_size =
          static_cast<uint32_t>(dispatcher_words.size() * sizeof(uint32_t));
      dispatcher_info.moi_dense_entry_island_offset = island_offset;
      dispatcher_info.owner_descriptor_file_offsets =
          group.front()->resources.owner_descriptor_file_offsets;
      patches.push_back(std::move(dispatcher_info));

      if (entry_host != dense_entry_hosts.end()) {
        const std::optional<uint32_t> host_entry =
            explicit_dense_key ? std::optional<uint32_t>(
                                     build_s_mov_b32(key_sgpr, scalar_positive_inline_u32(0), arch))
                               : instrumentation::build_s_call_i64(*call_return_sgpr,
                                                                   /*next instruction=*/0, arch);
        if (!host_entry || entry_host->second.host_offset > new_text.size() ||
            sizeof(uint32_t) > new_text.size() - entry_host->second.host_offset) {
          result.errors.emplace_back(
              "ConSan MOI inline-shadow probe could not encode its dense entry host");
          return;
        }
        std::memcpy(new_text.data() + entry_host->second.host_offset, &*host_entry,
                    sizeof(uint32_t));
      }

      for (size_t group_index = 0; group_index < group.size(); ++group_index) {
        const PlannedInlineShadowPatch *planned_patch = group[group_index];
        const ConSanMoiCandidate &candidate = *planned_patch->candidate;
        const uint64_t branch_offset =
            candidate.text_offset + (explicit_dense_key ? sizeof(uint32_t) : 0u);
        const auto call_delta = compute_sopp_branch_simm16(branch_offset, island_offset);
        const std::optional<uint32_t> call =
            !call_delta ? std::nullopt
                        : instrumentation::build_s_call_i64(*call_return_sgpr, *call_delta, arch);
        if ((!explicit_dense_key && !call) ||
            candidate.size < (explicit_dense_key ? 2u : 1u) * sizeof(uint32_t) ||
            candidate.text_offset > new_text.size() ||
            candidate.size > new_text.size() - candidate.text_offset) {
          result.errors.emplace_back(
              "ConSan MOI inline-shadow probe could not encode a dense call anchor");
          return;
        }
        std::vector<uint32_t> anchor_words(candidate.size / sizeof(uint32_t), build_s_nop(0, arch));
        if (explicit_dense_key) {
          if (!call_delta || group_index >= 64u) {
            result.errors.emplace_back(
                "ConSan MOI inline-shadow probe could not encode an explicit dense branch key");
            return;
          }
          anchor_words[0] = build_s_mov_b32(
              key_sgpr, scalar_positive_inline_u32(static_cast<uint16_t>(group_index + 1u)), arch);
          anchor_words[1] = build_s_branch(*call_delta, arch);
        } else {
          anchor_words.front() = *call;
        }
        std::memcpy(new_text.data() + candidate.text_offset, anchor_words.data(), candidate.size);
      }
    }
    for (const PlannedInlineShadowPatch &planned_patch : planned_patches) {
      const ConSanMoiCandidate &candidate = *planned_patch.candidate;
      ConSanOptions candidate_options = options;
      if (!apply_moi_transient_sgpr_assignment(
              candidate_options, planned_patch.resources.owner_descriptor_file_offsets)) {
        result.errors.emplace_back(
            "ConSan MOI inline-shadow access lost its transient scalar assignment");
        return;
      }
      if (!candidate_options.moi_persistent_vgpr_assignments.empty() &&
          !apply_moi_persistent_vgpr_assignment(
              candidate_options, planned_patch.resources.owner_descriptor_file_offsets)) {
        result.errors.emplace_back(
            "ConSan MOI inline-shadow access lost its persistent VGPR assignment");
        return;
      }
      candidate_options.moi_inline_workgroup_shadow = planned_patch.workgroup_shadow.has_value();
      candidate_options.scratch_vgpr = planned_patch.resources.base;
      candidate_options.moi_spill_overlaps_guest_operands = planned_patch.spill.has_value();
      auto words = build_inline_shadow_words(
          bytes, candidate, candidate_options, arch, layout, planned_patch.workgroup_sources,
          planned_patch.workgroup_shadow, planned_patch.private_epoch_offset,
          planned_patch.private_owner_offset, planned_patch.private_owner_shift,
          planned_patch.private_workgroup_key_offset, planned_patch.workgroup_shadow_compact_token,
          result.errors);
      if (!words)
        return;

      // Keep every displaced memory access as close as possible to the
      // continuation. In particular, appended caves must restore spills and
      // prepare a long return before touching application data; otherwise
      // another wave can observe the access while this wave is still in
      // trampoline bookkeeping.
      std::vector<uint32_t> deferred_guest_words;
      if (planned_patch.placement.kind == DbiPatchPlacementKind::AppendedCave &&
          (candidate.kind == ConSanLdsAccessKind::Read ||
           candidate.kind == ConSanLdsAccessKind::Write)) {
        const size_t guest_word_count = candidate.size / sizeof(uint32_t);
        if (words->size() < guest_word_count) {
          result.errors.emplace_back(
              "ConSan MOI inline-shadow probe lost its deferred guest write");
          return;
        }
        deferred_guest_words.assign(words->end() - guest_word_count, words->end());
        words->erase(words->end() - guest_word_count, words->end());
      }

      ConSanPatchInfo info;
      info.kind = planned_patch.placement.kind == DbiPatchPlacementKind::AppendedCave
                      ? ConSanPatchKind::TrampolineMoiExactShadowStore
                      : ConSanPatchKind::InlineMoiExactShadowStore;
      info.anchor_offset = candidate.text_offset;
      info.scratch_vgpr = planned_patch.resources.base;
      info.owner_descriptor_file_offsets = planned_patch.resources.owner_descriptor_file_offsets;
      info.persistent_epoch_private_offset = planned_patch.private_epoch_offset;
      info.persistent_owner_private_offset = planned_patch.private_owner_offset;
      info.persistent_workgroup_key_private_offset = planned_patch.private_workgroup_key_offset;
      if (planned_patch.private_epoch_offset) {
        uint32_t persistent_state_end =
            *planned_patch.private_epoch_offset + SpillManager::kSlotBytes;
        if (planned_patch.private_owner_offset)
          persistent_state_end = std::max(
              persistent_state_end, *planned_patch.private_owner_offset + SpillManager::kSlotBytes);
        if (planned_patch.private_workgroup_key_offset)
          persistent_state_end =
              std::max(persistent_state_end,
                       *planned_patch.private_workgroup_key_offset + SpillManager::kSlotBytes);
        info.persistent_private_state_end = persistent_state_end;
      }
      info.required_private_segment_size = planned_patch.required_private_bytes;
      note_dynamic_stack_private_requirement(info,
                                             planned_patch.spill ? &*planned_patch.spill : nullptr);
      if (planned_patch.workgroup_shadow) {
        info.workgroup_shadow_base = planned_patch.workgroup_shadow->base;
        info.workgroup_shadow_size = planned_patch.workgroup_shadow->size;
        info.workgroup_shadow_validity_base = planned_patch.workgroup_shadow->validity_base;
        info.workgroup_shadow_validity_size = planned_patch.workgroup_shadow->validity_size;
        info.workgroup_shadow_lazy_initialization =
            planned_patch.workgroup_shadow->lazy_initialization;
        info.workgroup_shadow_compact = planned_patch.workgroup_shadow->compact;
        info.workgroup_shadow_compact_token = planned_patch.workgroup_shadow_compact_token;
        info.required_group_segment_size =
            planned_patch.workgroup_shadow->required_group_segment_size;
      }
      if (planned_patch.spill)
        info.spilled_vgpr_count = planned_patch.spill->vgpr_count;
      info.branch_only_continuation = planned_patch.branch_only_continuation;
      info.branch_only_entry_relay_offsets = planned_patch.entry_relay_offsets;
      info.branch_only_return_relay_offsets = planned_patch.return_relay_offsets;

      if (planned_patch.placement.kind == DbiPatchPlacementKind::AppendedCave) {
        const uint64_t cave_text_offset = planned_patch.placement.body_offset;
        if (new_text.size() != cave_text_offset) {
          result.errors.emplace_back(
              "ConSan MOI inline-shadow probe emitted a stale appended-cave mapping: expected " +
              std::to_string(cave_text_offset) + " actual " + std::to_string(new_text.size()));
          return;
        }
        const uint64_t entry_target =
            planned_patch.branch_only_continuation && !planned_patch.entry_relay_offsets.empty()
                ? planned_patch.entry_relay_offsets.front()
                : planned_patch.entry_island_offset.value_or(planned_patch.placement.body_offset);
        if (!planned_patch.dense_call_anchor &&
            !write_moi_appended_anchor(new_text, planned_patch, entry_target, arch,
                                       "inline-shadow probe", result.errors))
          return;

        if (planned_patch.entry_island_offset && !planned_patch.dense_call_anchor) {
          const auto jump_sgprs = moi_indirect_jump_sgprs(candidate_options);
          if (!jump_sgprs) {
            result.errors.emplace_back(
                "ConSan MOI inline-shadow probe has no indirect jump scalar state");
            return;
          }
          std::vector<uint32_t> island_words;
          if (!append_moi_scc_preserving_indirect_jump(
                  island_words, *planned_patch.entry_island_offset, cave_text_offset,
                  jump_sgprs->pc_sgpr, jump_sgprs->scc_save_sgpr,
                  /*capture_scc=*/true, arch) ||
              island_words.empty()) {
            result.errors.emplace_back(
                "ConSan MOI inline-shadow probe could not encode its entry island");
            return;
          }
          const auto wait_pc = instrumentation::build_s_wait_indirect_pc0(arch);
          if (!wait_pc) {
            result.errors.emplace_back(
                "ConSan MOI inline-shadow probe could not encode its entry dependency wait");
            return;
          }
          island_words.insert(island_words.end() - 1, *wait_pc);
          if (island_words.size() != kIndirectIslandWords) {
            result.errors.emplace_back(
                "ConSan MOI inline-shadow probe encoded an invalid entry island size");
            return;
          }
          if (!write_moi_entry_island(new_text, planned_patch, island_words, patches,
                                      "inline-shadow probe", result.errors))
            return;
        }

        MoiAppendedBodyOptions body_options(arch);
        const std::array<uint32_t, 1> dynamic_stack_guard_placeholder = {
            build_s_nop(0, arch),
        };
        if (planned_patch.branch_only_continuation)
          body_options.entry_prefix_words = dynamic_stack_guard_placeholder;
        body_options.deferred_guest_word_count = deferred_guest_words.size();
        body_options.scalar_save_words = planned_patch.scalar_spill
                                             ? planned_patch.scalar_spill->save_words
                                             : std::span<const uint32_t>{};
        body_options.scalar_restore_words = planned_patch.scalar_spill
                                                ? planned_patch.scalar_spill->restore_words
                                                : std::span<const uint32_t>{};
        body_options.gfx1250_vgpr_msb_mode = candidate.gfx1250_vgpr_msb_mode;
        auto cave_words = assemble_moi_appended_body(planned_patch, *words, "inline-shadow probe",
                                                     result.errors, body_options);
        if (!cave_words)
          return;
        // An anchor-resident indirect island may displace instructions after
        // the memory access. Those instructions are the access's continuation
        // and must remain after the deferred guest operation. In particular,
        // a load's relocated consumers may overwrite its address VGPRs.
        std::vector<uint32_t> displaced_tail_words;
        if (!deferred_guest_words.empty() && !planned_patch.displaced_tail_words.empty()) {
          if (cave_words->size() < planned_patch.displaced_tail_words.size()) {
            result.errors.emplace_back(
                "ConSan MOI inline-shadow probe lost its displaced continuation");
            return;
          }
          const auto displaced_begin =
              cave_words->end() - planned_patch.displaced_tail_words.size();
          displaced_tail_words.assign(displaced_begin, cave_words->end());
          cave_words->erase(displaced_begin, cave_words->end());
        }
        if (planned_patch.branch_only_continuation) {
          if (!candidate_options.moi_inline_dynamic_stack_borrowed_sgpr ||
              planned_patch.entry_island_offset || planned_patch.dense_call_anchor ||
              !planned_patch.displaced_tail_words.empty()) {
            result.errors.emplace_back(
                "ConSan MOI inline-shadow probe found inconsistent branch-only state");
            return;
          }
          if (cave_words->empty()) {
            result.errors.emplace_back(
                "ConSan MOI inline-shadow dynamic-stack body lost its guard slot");
            return;
          }
          const uint64_t guarded_word_count =
              cave_words->size() - 1u + deferred_guest_words.size() + displaced_tail_words.size();
          if (guarded_word_count > std::numeric_limits<int16_t>::max()) {
            result.errors.emplace_back(
                "ConSan MOI inline-shadow dynamic-stack guard exceeds conditional-branch range");
            return;
          }
          const auto skip_empty_wave = instrumentation::build_s_cbranch_execz(
              static_cast<int16_t>(guarded_word_count), arch);
          if (!skip_empty_wave) {
            result.errors.emplace_back(
                "ConSan MOI inline-shadow probe could not encode its empty-wave guard");
            return;
          }
          cave_words->front() = *skip_empty_wave;
          cave_words->insert(cave_words->end(), deferred_guest_words.begin(),
                             deferred_guest_words.end());
          cave_words->insert(cave_words->end(), displaced_tail_words.begin(),
                             displaced_tail_words.end());
          if (cave_words->size() * sizeof(uint32_t) != planned_patch.placement.body_size) {
            result.errors.emplace_back(
                "ConSan MOI inline-shadow branch-only body size changed after placement");
            return;
          }
          const uint64_t return_source = cave_text_offset + cave_words->size() * sizeof(uint32_t);
          const uint64_t return_target = planned_patch.return_relay_offsets.empty()
                                             ? planned_patch.placement.return_target
                                             : planned_patch.return_relay_offsets.front();
          const auto ret = compute_sopp_branch_simm16(return_source, return_target);
          if (!ret) {
            result.errors.emplace_back(
                "ConSan MOI inline-shadow branch-only return is unreachable");
            return;
          }
          cave_words->push_back(build_s_branch(*ret, arch));
        } else if (planned_patch.dense_call_anchor) {
          if (arch == ROCJITSU_CODE_ARCH_GFX1250) {
            cave_words->insert(cave_words->end(), deferred_guest_words.begin(),
                               deferred_guest_words.end());
            cave_words->insert(cave_words->end(), displaced_tail_words.begin(),
                               displaced_tail_words.end());
            const uint16_t call_return_sgpr =
                candidate_options.automatic_moi_inline_sgpr_spill
                    ? *candidate_options.moi_inline_call_return_sgpr
                    : static_cast<uint16_t>(*candidate_options.moi_exec_save_sgpr + 28u);
            cave_words->push_back(build_s_setpc_b64(call_return_sgpr, arch));
          } else {
            const auto jump_sgprs = moi_indirect_jump_sgprs(candidate_options);
            if (!jump_sgprs ||
                !append_moi_scc_preserving_indirect_jump(
                    *cave_words, cave_text_offset, planned_patch.placement.return_target,
                    jump_sgprs->pc_sgpr, jump_sgprs->scc_save_sgpr,
                    /*capture_scc=*/false, arch)) {
              result.errors.emplace_back(
                  "ConSan MOI inline-shadow dense explicit-key probe could not encode its long "
                  "return");
              return;
            }
            const auto wait_pc = instrumentation::build_s_wait_indirect_pc0(arch);
            if (!wait_pc) {
              result.errors.emplace_back(
                  "ConSan MOI inline-shadow dense explicit-key probe could not encode its return "
                  "dependency wait");
              return;
            }
            cave_words->insert(cave_words->end() - 1, *wait_pc);
            cave_words->insert(cave_words->end() - 1, deferred_guest_words.begin(),
                               deferred_guest_words.end());
            cave_words->insert(cave_words->end() - 1, displaced_tail_words.begin(),
                               displaced_tail_words.end());
          }
        } else if (planned_patch.entry_island_offset) {
          const auto jump_sgprs = moi_indirect_jump_sgprs(candidate_options);
          if (!jump_sgprs) {
            result.errors.emplace_back(
                "ConSan MOI inline-shadow probe has no return jump scalar state");
            return;
          }
          if (!append_moi_scc_preserving_indirect_jump(
                  *cave_words, cave_text_offset, planned_patch.placement.return_target,
                  jump_sgprs->pc_sgpr, jump_sgprs->scc_save_sgpr,
                  /*capture_scc=*/false, arch)) {
            result.errors.emplace_back(
                "ConSan MOI inline-shadow probe could not encode its long return");
            return;
          }
          const auto wait_pc = instrumentation::build_s_wait_indirect_pc0(arch);
          if (!wait_pc) {
            result.errors.emplace_back(
                "ConSan MOI inline-shadow probe could not encode its return dependency wait");
            return;
          }
          cave_words->insert(cave_words->end() - 1, *wait_pc);
          cave_words->insert(cave_words->end() - 1, deferred_guest_words.begin(),
                             deferred_guest_words.end());
          cave_words->insert(cave_words->end() - 1, displaced_tail_words.begin(),
                             displaced_tail_words.end());
        } else {
          const auto ret = compute_sopp_branch_simm16(planned_patch.placement.return_branch_offset,
                                                      planned_patch.placement.return_target);
          if (!ret) {
            result.errors.emplace_back(
                "ConSan MOI inline-shadow probe appended cave return is out of range");
            return;
          }
          cave_words->insert(cave_words->end(), deferred_guest_words.begin(),
                             deferred_guest_words.end());
          cave_words->insert(cave_words->end(), displaced_tail_words.begin(),
                             displaced_tail_words.end());
          cave_words->push_back(build_s_branch(*ret, arch));
        }
        append_words_bytes(new_text, *cave_words);
        info.trampoline_offset = cave_text_offset;
        info.original_size = planned_patch.placement.original_size;
        info.trampoline_size = static_cast<uint32_t>(cave_words->size() * sizeof(uint32_t));

      } else {
        const uint64_t patch_bytes = static_cast<uint64_t>(words->size() * sizeof(uint32_t));
        if (patch_bytes != planned_patch.placement.body_size) {
          result.errors.emplace_back("ConSan MOI inline-shadow probe final patch size changed");
          return;
        }
        if (candidate.text_offset > new_text.size() ||
            patch_bytes > new_text.size() - candidate.text_offset) {
          result.errors.emplace_back("ConSan MOI inline-shadow probe exceeds .text");
          return;
        }
        std::memcpy(new_text.data() + candidate.text_offset, words->data(),
                    static_cast<size_t>(patch_bytes));
        info.trampoline_offset = candidate.text_offset + candidate.size;
        info.original_size = static_cast<uint32_t>(patch_bytes);
        info.trampoline_size = 0;
      }
      patches.push_back(info);
      if (planned_patch.branch_only_route) {
        std::string relay_error;
        if (!BranchOnlyRelayRouter::emit_and_record(
                new_text, *planned_patch.branch_only_route, info.trampoline_offset,
                planned_patch.placement.return_target, arch, patches, &relay_error)) {
          result.errors.emplace_back("ConSan MOI inline-shadow probe " + relay_error);
          return;
        }
      }
    }

    for (const auto &[dispatcher, host] : dense_entry_hosts) {
      if (new_text.size() != host.body_offset) {
        result.errors.emplace_back(
            "ConSan MOI inline-shadow probe emitted a stale dense entry-host mapping");
        return;
      }
      if (!dense_groups.contains(dispatcher)) {
        // Host reservations are assigned before resource filtering.  If all
        // sites in an owner group are later rejected, retain its stable
        // offsets as padding rather than emitting an unreachable relay body
        // with no corresponding patch record.
        const size_t reserved_words = host.displaced_words.size() + kIndirectIslandWords;
        for (size_t word = 0; word < reserved_words; ++word)
          append_word_bytes(new_text, build_s_nop(0, arch));
        continue;
      }
      const auto &group = dense_groups.at(dispatcher);
      ConSanOptions group_options = options;
      if (group.empty() ||
          !apply_moi_transient_sgpr_assignment(
              group_options, group.front()->resources.owner_descriptor_file_offsets)) {
        result.errors.emplace_back(
            "ConSan MOI inline-shadow probe lost its dense host scalar assignment");
        return;
      }
      const auto jump_sgprs = moi_indirect_jump_sgprs(group_options);
      std::vector<uint32_t> host_words = host.displaced_words;
      if (!jump_sgprs ||
          !append_moi_scc_preserving_indirect_jump(
              host_words, host.body_offset,
              host.host_offset + host.displaced_words.size() * sizeof(uint32_t),
              jump_sgprs->pc_sgpr, jump_sgprs->scc_save_sgpr, /*capture_scc=*/true, arch)) {
        result.errors.emplace_back(
            "ConSan MOI inline-shadow probe could not encode its dense entry-host return");
        return;
      }
      const auto wait_pc = instrumentation::build_s_wait_indirect_pc0(arch);
      if (!wait_pc) {
        result.errors.emplace_back(
            "ConSan MOI inline-shadow probe could not encode its dense entry-host dependency wait");
        return;
      }
      host_words.insert(host_words.end() - 1, *wait_pc);
      if (host_words.size() != host.displaced_words.size() + kIndirectIslandWords) {
        result.errors.emplace_back(
            "ConSan MOI inline-shadow probe encoded an invalid dense entry-host return size");
        return;
      }
      append_words_bytes(new_text, host_words);
    }

    if (!replace_consan_text(patcher, new_text, options, "MOI inline-shadow probe", result)) {
      return;
    }
    result.elf_bytes = std::move(patcher).emit();
    result.patches.insert(result.patches.end(), patches.begin(), patches.end());
    result.modified = true;
    return;
  }

  if (!apply_inline_moi_access_patches(
          bytes, planned_patches, descriptor_requirements, scalar_requirements,
          "inline-shadow probe", arch,
          [&](const PlannedInlineShadowPatch &planned_patch) {
            ConSanOptions candidate_options = options;
            if (!candidate_options.moi_persistent_vgpr_assignments.empty() &&
                !apply_moi_persistent_vgpr_assignment(
                    candidate_options, planned_patch.resources.owner_descriptor_file_offsets)) {
              result.errors.emplace_back(
                  "ConSan MOI inline-shadow access lost its persistent VGPR assignment");
              return std::optional<std::vector<uint32_t>>{};
            }
            candidate_options.moi_inline_workgroup_shadow =
                planned_patch.workgroup_shadow.has_value();
            candidate_options.scratch_vgpr = planned_patch.resources.base;
            candidate_options.moi_spill_overlaps_guest_operands = planned_patch.spill.has_value();
            return build_inline_shadow_words(
                bytes, *planned_patch.candidate, candidate_options, arch, layout,
                planned_patch.workgroup_sources, planned_patch.workgroup_shadow,
                planned_patch.private_epoch_offset, planned_patch.private_owner_offset,
                planned_patch.private_owner_shift, planned_patch.private_workgroup_key_offset,
                planned_patch.workgroup_shadow_compact_token, result.errors);
          },
          [](const PlannedInlineShadowPatch &planned_patch) {
            const ConSanMoiCandidate &candidate = *planned_patch.candidate;
            ConSanPatchInfo info;
            info.kind = ConSanPatchKind::InlineMoiExactShadowStore;
            info.anchor_offset = candidate.text_offset;
            info.trampoline_offset = candidate.text_offset + candidate.size;
            info.original_size = static_cast<uint32_t>(planned_patch.placement.body_size);
            info.scratch_vgpr = planned_patch.resources.base;
            info.owner_descriptor_file_offsets =
                planned_patch.resources.owner_descriptor_file_offsets;
            if (planned_patch.workgroup_shadow) {
              info.workgroup_shadow_base = planned_patch.workgroup_shadow->base;
              info.workgroup_shadow_size = planned_patch.workgroup_shadow->size;
              info.workgroup_shadow_validity_base = planned_patch.workgroup_shadow->validity_base;
              info.workgroup_shadow_validity_size = planned_patch.workgroup_shadow->validity_size;
              info.workgroup_shadow_lazy_initialization =
                  planned_patch.workgroup_shadow->lazy_initialization;
              info.workgroup_shadow_compact = planned_patch.workgroup_shadow->compact;
              info.workgroup_shadow_compact_token = planned_patch.workgroup_shadow_compact_token;
              info.required_group_segment_size =
                  planned_patch.workgroup_shadow->required_group_segment_size;
            }
            return info;
          },
          [&](std::vector<uint8_t> &image, ConSanResult &current_result) {
            return apply_lds_descriptor_requirements(image, current_result, lds_requirements,
                                                     options, arch, current_result.errors);
          },
          result))
    return;
}

[[nodiscard]] bool append_sampled_banked_address(std::vector<uint32_t> &words,
                                                 uint64_t first_address, uint32_t element_size,
                                                 uint32_t bank_count, uint16_t bank_vgpr,
                                                 uint16_t address_vgpr, uint16_t temporary_vgpr,
                                                 rj_code_arch_t arch);

[[nodiscard]] bool append_direct_sampled_immediate_check(
    std::vector<uint32_t> &words, const ConSanOptions &options, rj_code_arch_t arch,
    uint32_t prior_record_index, uint32_t window_bank_count, uint16_t bank_vgpr,
    size_t sampled_watchpoints_offset, uint16_t current_low_vgpr, uint16_t current_high_vgpr,
    std::vector<std::string> &errors) {
  if (!options.moi_sampled_check)
    return true;
  if (!options.moi_report_buffer_address || !options.scratch_vgpr || !options.moi_exec_save_sgpr) {
    errors.emplace_back("ConSan MOI sampled checker is missing report or scalar resources");
    return false;
  }

  const uint16_t address_lo_vgpr = *options.scratch_vgpr;
  const uint16_t address_hi_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 1u);
  const uint16_t tmp_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 4u);
  const uint16_t prior_low_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 5u);
  const uint16_t prior_high_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 6u);
  if (window_bank_count == 0) {
    errors.emplace_back("ConSan MOI sampled checker has an invalid prior bank");
    return false;
  }
  const uint64_t prior_address = *options.moi_report_buffer_address + sampled_watchpoints_offset +
                                 static_cast<uint64_t>(prior_record_index) * sizeof(uint64_t);
  InstructionSequence sequence(words);
  const auto restore_vcc_label = sequence.make_label();

  const auto save_vcc =
      instrumentation::build_s_mov_b64(*options.moi_exec_save_sgpr, kRdna4VccLo, arch);
  const auto zero_low = instrumentation::build_v_mov_b32_literal(prior_low_vgpr, 0u, arch);
  const auto zero_high = instrumentation::build_v_mov_b32_literal(prior_high_vgpr, 0u, arch);
  const auto atomic_snapshot = instrumentation::build_flat_atomic_add_u64(
      address_lo_vgpr, prior_low_vgpr, prior_low_vgpr,
      /*return_old_value=*/true, kRdna4ScopeDevice, arch);
  if (!save_vcc || !zero_low || !zero_high || !atomic_snapshot ||
      !append_sampled_banked_address(words, prior_address, sizeof(uint64_t), window_bank_count,
                                     bank_vgpr, address_lo_vgpr, tmp_vgpr, arch)) {
    errors.emplace_back("ConSan MOI sampled checker could not atomically snapshot the prior slot");
    return false;
  }
  words.insert(words.end(), zero_low->begin(), zero_low->end());
  words.insert(words.end(), zero_high->begin(), zero_high->end());
  words.insert(words.end(), atomic_snapshot->begin(), atomic_snapshot->end());
  if (!append_moi_flat_load_wait(words, arch))
    return false;
  words.push_back(*save_vcc);

  auto append_required_predicate = [&](std::optional<uint32_t> compare) {
    if (!compare)
      return false;
    words.push_back(*compare);
    return sequence.emit_branch(restore_vcc_label, InstructionSequence::BranchKind::VccZero);
  };

  const auto prior_valid =
      instrumentation::build_v_and_b32_literal(tmp_vgpr, 1u, prior_low_vgpr, arch);
  if (!prior_valid) {
    errors.emplace_back("ConSan MOI sampled checker could not decode validity");
    return false;
  }
  words.insert(words.end(), prior_valid->begin(), prior_valid->end());
  if (!append_required_predicate(
          instrumentation::build_v_cmp_ne_u32_vcc(scalar_positive_inline_u32(0), tmp_vgpr, arch)) ||
      !append_extract_exact_shadow_field(words, address_lo_vgpr, current_low_vgpr,
                                         consan_moi_sampled_watchpoint::owner_shift,
                                         consan_moi_sampled_watchpoint::max_owner, arch) ||
      !append_extract_exact_shadow_field(words, address_hi_vgpr, prior_low_vgpr,
                                         consan_moi_sampled_watchpoint::owner_shift,
                                         consan_moi_sampled_watchpoint::max_owner, arch) ||
      !append_required_predicate(instrumentation::build_v_cmp_ne_u32_vcc(
          vector_source_vgpr(address_lo_vgpr), address_hi_vgpr, arch))) {
    errors.emplace_back("ConSan MOI sampled checker could not compare owners");
    return false;
  }

  const uint32_t low_epoch_generation_mask =
      static_cast<uint32_t>(consan_moi_sampled_watchpoint::epoch_generation_mask);
  const uint32_t high_generation_mask =
      static_cast<uint32_t>(consan_moi_sampled_watchpoint::generation_mask >> 32u);
  const uint32_t high_range_mask = static_cast<uint32_t>(
      (consan_moi_sampled_watchpoint::start_mask | consan_moi_sampled_watchpoint::count_mask) >>
      32u);
  auto append_equal_masked = [&](uint16_t current_vgpr, uint16_t prior_vgpr, uint32_t mask) {
    const auto current =
        instrumentation::build_v_and_b32_literal(address_lo_vgpr, mask, current_vgpr, arch);
    const auto prior =
        instrumentation::build_v_and_b32_literal(address_hi_vgpr, mask, prior_vgpr, arch);
    if (!current || !prior)
      return false;
    words.insert(words.end(), current->begin(), current->end());
    words.insert(words.end(), prior->begin(), prior->end());
    return append_required_predicate(instrumentation::build_v_cmp_eq_u32_vcc(
        vector_source_vgpr(address_lo_vgpr), address_hi_vgpr, arch));
  };
  if (!append_equal_masked(current_low_vgpr, prior_low_vgpr, low_epoch_generation_mask) ||
      !append_equal_masked(current_high_vgpr, prior_high_vgpr, high_generation_mask) ||
      !append_equal_masked(current_high_vgpr, prior_high_vgpr, high_range_mask)) {
    errors.emplace_back("ConSan MOI sampled checker could not compare ordering or ranges");
    return false;
  }

  if (!append_extract_exact_shadow_field(
          words, address_lo_vgpr, current_low_vgpr,
          consan_moi_sampled_watchpoint::access_kind_shift,
          (1u << consan_moi_sampled_watchpoint::access_kind_bits) - 1u, arch) ||
      !append_extract_exact_shadow_field(
          words, address_hi_vgpr, prior_low_vgpr, consan_moi_sampled_watchpoint::access_kind_shift,
          (1u << consan_moi_sampled_watchpoint::access_kind_bits) - 1u, arch)) {
    errors.emplace_back("ConSan MOI sampled checker could not decode access kinds");
    return false;
  }
  const auto both_read = instrumentation::build_v_and_b32(
      tmp_vgpr, vector_source_vgpr(address_lo_vgpr), address_hi_vgpr, arch);
  if (!both_read) {
    errors.emplace_back("ConSan MOI sampled checker could not compare access kinds");
    return false;
  }
  words.push_back(*both_read);
  if (!append_required_predicate(instrumentation::build_v_cmp_ne_u32_vcc(
          scalar_positive_inline_u32(static_cast<uint32_t>(ConSanMoiShadowAccessKind::Read)),
          tmp_vgpr, arch))) {
    errors.emplace_back("ConSan MOI sampled checker could not predicate conflicting kinds");
    return false;
  }

  const uint64_t immediate_conflict_count_address =
      *options.moi_report_buffer_address + offsetof(ConSanMoiReportHeader, event_counter);
  if (!append_atomic_fetch_add_one_u32(words, immediate_conflict_count_address, tmp_vgpr,
                                       *options.scratch_vgpr, arch)) {
    errors.emplace_back("ConSan MOI sampled checker could not increment the conflict counter");
    return false;
  }

  const auto restore_vcc =
      instrumentation::build_s_mov_b64(kRdna4VccLo, *options.moi_exec_save_sgpr, arch);
  if (!restore_vcc) {
    errors.emplace_back("ConSan MOI sampled checker could not restore VCC");
    return false;
  }
  if (!sequence.bind(restore_vcc_label))
    return false;
  words.push_back(*restore_vcc);
  if (!sequence.resolve_branches(arch)) {
    errors.emplace_back("ConSan MOI sampled checker skip branch is out of range");
    return false;
  }
  return true;
}

[[nodiscard]] bool
append_sampled_window_bank_index(std::vector<uint32_t> &words, const ConSanOptions &options,
                                 const ConSanMoiWorkgroupSources &workgroup_sources,
                                 uint32_t bank_count, uint16_t bank_vgpr, uint16_t temporary_vgpr,
                                 rj_code_arch_t arch) {
  if (bank_count == 0 || !std::has_single_bit(bank_count))
    return false;
  if (bank_count == 1) {
    const auto zero = instrumentation::build_v_mov_b32_literal(bank_vgpr, 0, arch);
    if (!zero)
      return false;
    words.insert(words.end(), zero->begin(), zero->end());
    return true;
  }
  if (!append_moi_report_dispatch_id_pair(words, options, bank_vgpr, temporary_vgpr, arch,
                                          ConSanMoiLiteralDispatchIdPolicy::Rdna4FamilyOnly)) {
    return false;
  }
  const auto mix = [&](uint16_t source) {
    const auto word =
        instrumentation::build_v_xor_b32(bank_vgpr, vector_source_vgpr(source), bank_vgpr, arch);
    if (!word)
      return false;
    words.push_back(*word);
    return true;
  };
  if (!mix(temporary_vgpr))
    return false;
  for (const ConSanMoiWorkgroupSource &source :
       {workgroup_sources.x, workgroup_sources.y, workgroup_sources.z,
        workgroup_sources.cluster_workgroup_id}) {
    if (!source.scalar_src)
      continue;
    if (!consan_detail::append_workgroup_source_value(words, source, temporary_vgpr, arch) ||
        !mix(temporary_vgpr))
      return false;
  }
  const auto shift = instrumentation::build_v_lshrrev_b32(
      temporary_vgpr, scalar_positive_inline_u32(16), bank_vgpr, arch);
  if (!shift)
    return false;
  words.push_back(*shift);
  if (!mix(temporary_vgpr))
    return false;
  const auto multiply = instrumentation::build_v_mul_lo_u32_literal(bank_vgpr, temporary_vgpr,
                                                                    0x85ebca6bu, bank_vgpr, arch);
  if (!multiply)
    return false;
  words.insert(words.end(), multiply->begin(), multiply->end());
  const auto mask =
      instrumentation::build_v_and_b32_literal(bank_vgpr, bank_count - 1u, bank_vgpr, arch);
  if (!mask)
    return false;
  words.insert(words.end(), mask->begin(), mask->end());
  return true;
}

[[nodiscard]] bool append_sampled_banked_address(std::vector<uint32_t> &words,
                                                 uint64_t first_address, uint32_t element_size,
                                                 uint32_t bank_count, uint16_t bank_vgpr,
                                                 uint16_t address_vgpr, uint16_t temporary_vgpr,
                                                 rj_code_arch_t arch) {
  const auto low = instrumentation::build_v_mov_b32_literal(
      address_vgpr, static_cast<uint32_t>(first_address), arch);
  const auto high = instrumentation::build_v_mov_b32_literal(
      static_cast<uint16_t>(address_vgpr + 1u), static_cast<uint32_t>(first_address >> 32u), arch);
  if (!low || !high)
    return false;
  words.insert(words.end(), low->begin(), low->end());
  words.insert(words.end(), high->begin(), high->end());
  if (bank_count == 1)
    return true;
  const auto offset = instrumentation::build_v_mul_lo_u32_literal(temporary_vgpr, temporary_vgpr,
                                                                  element_size, bank_vgpr, arch);
  const auto add = instrumentation::build_v_add_u64_vgpr_offset(address_vgpr, temporary_vgpr, arch);
  if (!offset || !add)
    return false;
  words.insert(words.end(), offset->begin(), offset->end());
  words.insert(words.end(), add->begin(), add->end());
  return true;
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_direct_sampled_watchpoint_range_words(
    std::span<const uint8_t> bytes, const ConSanMoiCandidate &candidate,
    const ConSanMoiAccessRange &access_range, const ConSanOptions &options, rj_code_arch_t arch,
    uint32_t record_index, std::span<const uint32_t> prior_record_indices,
    uint32_t window_bank_count, bool relocate_instruction,
    std::optional<uint16_t> preserved_lds_byte_offset_vgpr, size_t sampled_causal_windows_offset,
    size_t sampled_watchpoints_offset, size_t sampled_pending_acquires_offset,
    std::vector<std::string> &errors, uint32_t *guest_instruction_offset = nullptr) {
  if (!options.scratch_vgpr) {
    errors.emplace_back("ConSan MOI sampled probe requires RJ_CONSAN_TMP_VGPR");
    return std::nullopt;
  }
  // A runtime fast-gate lowers the cave body with stride=1 after the outer
  // scalar gate has made the sampling decision.  Multi-bank publication still
  // needs its bank index to survive all ordinary temporaries in that cave.
  // Reserve the dedicated bank VGPR based on the emitted layout, not solely on
  // whether the inner vector sampler remains enabled.
  const bool dedicated_bank_vgpr = options.moi_runtime_sample_stride > 1 || window_bank_count > 1;
  const uint16_t base_scratch_count = static_cast<uint16_t>((options.moi_sampled_check ? 7u : 5u) +
                                                            (dedicated_bank_vgpr ? 1u : 0u));
  const uint16_t scratch_count = static_cast<uint16_t>(
      direct_sampled_scratch_count(options, candidate) +
      (window_bank_count > 1 && options.moi_runtime_sample_stride <= 1 ? 1u : 0u));
  if (static_cast<uint32_t>(*options.scratch_vgpr) + scratch_count > kMaxVgprs) {
    errors.emplace_back("ConSan MOI sampled probe exceeds the VGPR file");
    return std::nullopt;
  }
  auto lds_byte_offset_vgpr = candidate_lds_byte_offset_vgpr(candidate, errors);
  if (!lds_byte_offset_vgpr)
    return std::nullopt;
  if (preserved_lds_byte_offset_vgpr)
    lds_byte_offset_vgpr = preserved_lds_byte_offset_vgpr;
  if (!options.moi_spill_overlaps_guest_operands &&
      reject_candidate_scratch_range_overlap(candidate, *options.scratch_vgpr, scratch_count,
                                             errors))
    return std::nullopt;
  if ((!options.automatic_moi_private_epoch && !moi_uses_scalar_persistent_state(options) &&
       reject_optional_scratch_range_overlap(options.moi_owner_vgpr, *options.scratch_vgpr,
                                             scratch_count, "MOI owner", errors)) ||
      (!options.automatic_moi_private_epoch && !moi_uses_scalar_persistent_state(options) &&
       reject_optional_scratch_range_overlap(options.moi_epoch_vgpr, *options.scratch_vgpr,
                                             scratch_count, "MOI epoch", errors)))
    return std::nullopt;
  const ConSanMoiLdsCellRange static_range =
      consan_moi_lds_cell_range_for_bytes(access_range.static_byte_offset, access_range.byte_count);
  std::optional<uint16_t> derived_owner_vgpr;
  std::optional<uint32_t> derived_owner_word;
  if (!options.moi_owner_vgpr && candidate.kernel_descriptor_file_offset) {
    const auto owner_shift =
        moi_descriptor_owner_shift(bytes, *candidate.kernel_descriptor_file_offset, errors);
    if (!owner_shift)
      return std::nullopt;
    const uint16_t value_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 4u);
    const auto owner_init = instrumentation::build_v_lshrrev_b32(
        value_vgpr, scalar_positive_inline_u32(*owner_shift), kRdna4WorkitemIdX, arch);
    if (!owner_init) {
      errors.emplace_back("ConSan MOI sampled probe could not encode owner derivation");
      return std::nullopt;
    }
    derived_owner_vgpr = value_vgpr;
    derived_owner_word = *owner_init;
  }
  if (!candidate.kernel_descriptor_file_offset) {
    errors.emplace_back("ConSan MOI sampled probe requires exact workgroup-id sources");
    return std::nullopt;
  }
  const auto workgroup_sources =
      moi_descriptor_workgroup_sources(bytes, *candidate.kernel_descriptor_file_offset, arch,
                                       errors, candidate.uses_gfx1250_cluster_workgroup_id);
  if (!workgroup_sources)
    return std::nullopt;

  const uint16_t low_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 2u);
  const uint16_t high_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 3u);
  const uint16_t tmp_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 4u);
  const uint16_t owner_vgpr = options.moi_owner_vgpr.value_or(derived_owner_vgpr.value_or(0));
  const bool runtime_sampled = options.moi_runtime_sample_stride > 1;
  const uint16_t bank_vgpr =
      dedicated_bank_vgpr
          ? static_cast<uint16_t>(*options.scratch_vgpr + (options.moi_sampled_check ? 7u : 5u))
          : tmp_vgpr;
  if (runtime_sampled && !options.moi_owner_vgpr && !derived_owner_vgpr) {
    errors.emplace_back("ConSan MOI runtime sampled probe could not derive a wave owner");
    return std::nullopt;
  }
  if (runtime_sampled && !options.moi_dispatch_id_sgpr && !is_rdna4_family_arch(arch)) {
    errors.emplace_back("ConSan MOI runtime sampled probe requires a hardware dispatch ID");
    return std::nullopt;
  }
  if (!options.moi_exec_save_sgpr || *options.moi_exec_save_sgpr > 98u ||
      *options.moi_exec_save_sgpr % 2u != 0u) {
    errors.emplace_back(
        "ConSan MOI sampled probe needs seven aligned VCC/EXEC/SCC-save SGPRs in 0..104");
    return std::nullopt;
  }
  const auto kind = consan_moi_shadow_kind_from_access_kind(candidate.kind);
  const uint32_t generation =
      options.moi_report_generation & consan_moi_sampled_watchpoint::max_generation;
  const uint64_t generation_field = static_cast<uint64_t>(generation)
                                    << consan_moi_sampled_watchpoint::generation_shift;
  const uint32_t low_literal = static_cast<uint32_t>(
      consan_moi_sampled_watchpoint::valid_mask |
      (static_cast<uint64_t>(kind) << consan_moi_sampled_watchpoint::access_kind_shift) |
      generation_field);
  const uint32_t encoded_cell_count = encode_consan_moi_sampled_cell_count(static_range.cell_count)
                                      << (consan_moi_sampled_watchpoint::count_shift - 32u);
  const uint32_t encoded_generation_high = static_cast<uint32_t>(generation_field >> 32u);
  const uint64_t sampled_entry_address = *options.moi_report_buffer_address +
                                         sampled_watchpoints_offset +
                                         static_cast<uint64_t>(record_index) * sizeof(uint64_t);
  const uint64_t pending_acquire_address =
      *options.moi_report_buffer_address + sampled_pending_acquires_offset +
      static_cast<uint64_t>(record_index) * sizeof(ConSanMoiSampledPendingAcquireSlot);

  std::vector<uint32_t> words;
  words.reserve(candidate.size / sizeof(uint32_t) + 40u + (derived_owner_word ? 1u : 0u) +
                (options.moi_epoch_vgpr ? 2u : 0u) +
                (relocate_instruction && moi_load_clobbers_address(candidate) &&
                         !preserved_lds_byte_offset_vgpr
                     ? 1u
                     : 0u) +
                (access_range.static_byte_offset != 0u ? 3u : 0u));
  InstructionSequence sequence(words);
  const auto collision_label = sequence.make_label();
  const auto different_identity_label = sequence.make_label();
  const auto restore_label = sequence.make_label();
  ConSanMoiRecordEmitter record(words, *options.scratch_vgpr, tmp_vgpr, arch);
  // A DS load may overwrite its own address VGPR. Snapshot that address before
  // executing the displaced instruction so sampled publication remains tied
  // to the guest access rather than to the loaded payload.
  if (relocate_instruction && moi_load_clobbers_address(candidate) &&
      !preserved_lds_byte_offset_vgpr) {
    const uint16_t saved_lds_byte_offset_vgpr =
        static_cast<uint16_t>(*options.scratch_vgpr + base_scratch_count);
    words.push_back(build_v_mov_b32_e32(saved_lds_byte_offset_vgpr,
                                        vector_source_vgpr(*lds_byte_offset_vgpr), arch));
    lds_byte_offset_vgpr = saved_lds_byte_offset_vgpr;
  }
  if (derived_owner_word)
    words.push_back(*derived_owner_word);
  if (relocate_instruction) {
    const bool select_guest_vgpr_bank =
        arch == ROCJITSU_CODE_ARCH_GFX1250 && candidate.gfx1250_vgpr_msb_mode.value_or(0u) != 0u;
    if (select_guest_vgpr_bank)
      words.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
          0u, static_cast<uint8_t>(*candidate.gfx1250_vgpr_msb_mode), arch));
    if (guest_instruction_offset)
      *guest_instruction_offset = static_cast<uint32_t>(words.size() * sizeof(uint32_t));
    for (uint64_t offset = 0; offset < candidate.size; offset += sizeof(uint32_t)) {
      uint32_t word = 0;
      std::memcpy(&word, bytes.data() + candidate.file_offset + offset, sizeof(word));
      words.push_back(word);
    }
    if (!append_moi_lds_wait(words, arch))
      return std::nullopt;
    if (select_guest_vgpr_bank)
      words.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
          static_cast<uint8_t>(*candidate.gfx1250_vgpr_msb_mode), 0u, arch));
  }

  const uint16_t original_exec_save_sgpr = *options.moi_exec_save_sgpr;
  const uint16_t selection_vcc_save_sgpr = static_cast<uint16_t>(*options.moi_exec_save_sgpr + 2u);
  const uint16_t publication_exec_save_sgpr =
      static_cast<uint16_t>(*options.moi_exec_save_sgpr + 4u);
  const uint16_t publication_scc_save_sgpr =
      static_cast<uint16_t>(*options.moi_exec_save_sgpr + 6u);
  const auto save_scc =
      instrumentation::build_s_cselect_b32(publication_scc_save_sgpr, scalar_positive_inline_u32(1),
                                           scalar_positive_inline_u32(0), arch);
  const auto save_vcc =
      instrumentation::build_s_mov_b64(selection_vcc_save_sgpr, kRdna4VccLo, arch);
  const auto save_exec =
      instrumentation::build_s_mov_b64(original_exec_save_sgpr, kRdna4ExecLo, arch);
  if (!save_scc || !save_vcc || !save_exec) {
    errors.emplace_back("ConSan MOI sampled probe could not save guest EXEC/VCC/SCC");
    return std::nullopt;
  }
  // Publication has several nested EXEC-narrowing paths. Some paths reuse the
  // publication-save pair after an earlier narrowing, so that pair is not a
  // reliable copy of the guest mask at the common exit. Keep the otherwise
  // unused leading pair as the immutable guest EXEC snapshot.
  words.push_back(*save_exec);
  words.push_back(*save_scc);
  words.push_back(*save_vcc);
  if (candidate_requires_gfx1250_flat_offset_materialization(candidate)) {
    if (!preserved_lds_byte_offset_vgpr) {
      errors.emplace_back("ConSan MOI sampled FLAT offset has no preserved address pair");
      return std::nullopt;
    }
    const auto add = instrumentation::build_v_add_u64_signed_i24(*preserved_lds_byte_offset_vgpr,
                                                                 *candidate.raw_ioffset, arch);
    if (!add) {
      errors.emplace_back("ConSan MOI sampled probe could not materialize FLAT offset");
      return std::nullopt;
    }
    words.insert(words.end(), add->begin(), add->end());
  }
  if (runtime_sampled) {
    // Sample addresses while retaining every wave which touches the selected
    // cell. Selecting owners here would discard the cross-wave evidence that
    // the sampled checker needs in order to identify a race.
    if (access_range.static_byte_offset != 0u) {
      if (!append_compute_effective_lds_byte_offset(words, low_vgpr, *lds_byte_offset_vgpr,
                                                    access_range.static_byte_offset, arch))
        return std::nullopt;
    } else {
      words.push_back(
          build_v_mov_b32_e32(low_vgpr, vector_source_vgpr(*lds_byte_offset_vgpr), arch));
    }
    const auto cell = instrumentation::build_v_lshrrev_b32(
        low_vgpr, scalar_positive_inline_u32(consan_moi_sampled_watchpoint::granule_shift),
        low_vgpr, arch);
    const auto residue = instrumentation::build_v_and_b32_literal(
        low_vgpr, options.moi_runtime_sample_stride - 1u, low_vgpr, arch);
    const auto selected_value = instrumentation::build_v_mov_b32_literal(
        high_vgpr, options.moi_runtime_sample_offset, arch);
    const auto selected =
        instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(high_vgpr), low_vgpr, arch);
    const auto narrow_selected =
        instrumentation::build_s_and_saveexec_b64(publication_exec_save_sgpr, kRdna4VccLo, arch);
    if (!cell || !residue || !selected_value || !selected || !narrow_selected) {
      errors.emplace_back("ConSan MOI runtime sampled probe could not encode LDS-cell selector");
      return std::nullopt;
    }
    words.push_back(*cell);
    words.insert(words.end(), residue->begin(), residue->end());
    words.insert(words.end(), selected_value->begin(), selected_value->end());
    words.push_back(*selected);
    if (!sequence.emit_branch(restore_label, InstructionSequence::BranchKind::VccZero))
      return std::nullopt;
    words.push_back(*narrow_selected);
  }
  if (runtime_sampled || window_bank_count > 1) {
    if (!append_sampled_window_bank_index(words, options, *workgroup_sources, window_bank_count,
                                          bank_vgpr, high_vgpr, arch)) {
      errors.emplace_back("ConSan MOI sampled probe could not select a window bank");
      return std::nullopt;
    }
  } else {
    const auto zero_bank = instrumentation::build_v_mov_b32_literal(bank_vgpr, 0, arch);
    if (!zero_bank)
      return std::nullopt;
    words.insert(words.end(), zero_bank->begin(), zero_bank->end());
  }
  if (!append_moi_delay_words(words, arch, options, errors, "ConSan MOI sampled probe"))
    return std::nullopt;

  // A ready deferred acquire deterministically selects the wave which may
  // publish its associated later read. Empty slots are ordinary reads. Odd,
  // malformed, or other-wave slots take the existing fail-closed drop path,
  // preventing independent first-publisher races from joining unrelated
  // atomic and LDS evidence.
  if (kind == ConSanMoiShadowAccessKind::Read) {
    const auto pending_gate_done_label = sequence.make_label();
    if (!append_sampled_banked_address(
            words, pending_acquire_address, sizeof(ConSanMoiSampledPendingAcquireSlot),
            window_bank_count, bank_vgpr, *options.scratch_vgpr, high_vgpr, arch))
      return std::nullopt;
    if (!append_load_u32_vgpr_at_offset(words, *options.scratch_vgpr,
                                        offsetof(ConSanMoiSampledPendingAcquireSlot, version),
                                        low_vgpr, arch))
      return std::nullopt;
    const auto empty =
        instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), low_vgpr, arch);
    const auto enter_gate = instrumentation::build_s_cbranch_vccz(1, arch);
    if (!empty || !enter_gate)
      return std::nullopt;
    words.push_back(*empty);
    words.push_back(*enter_gate);
    if (!sequence.emit_branch(pending_gate_done_label,
                              InstructionSequence::BranchKind::Unconditional))
      return std::nullopt;

    const auto ready =
        instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(2), low_vgpr, arch);
    if (!ready)
      return std::nullopt;
    words.push_back(*ready);
    if (!sequence.emit_branch(different_identity_label, InstructionSequence::BranchKind::VccZero))
      return std::nullopt;
    if (!append_load_u32_vgpr_at_offset(words, *options.scratch_vgpr,
                                        offsetof(ConSanMoiSampledPendingAcquireSlot, owner_id),
                                        high_vgpr, arch))
      return std::nullopt;
    const auto owner_equal =
        instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(owner_vgpr), high_vgpr, arch);
    const auto narrow_owner =
        instrumentation::build_s_and_saveexec_b64(publication_exec_save_sgpr, kRdna4VccLo, arch);
    if (!owner_equal || !narrow_owner)
      return std::nullopt;
    words.push_back(*owner_equal);
    if (!sequence.emit_branch(different_identity_label, InstructionSequence::BranchKind::VccZero))
      return std::nullopt;
    words.push_back(*narrow_owner);
    if (!sequence.bind(pending_gate_done_label))
      return std::nullopt;
  }

  const uint64_t causal_window_address =
      *options.moi_report_buffer_address + sampled_causal_windows_offset +
      static_cast<uint64_t>(record_index) * sizeof(ConSanMoiSampledCausalWindow);
  const auto claim_new = instrumentation::build_v_mov_b32_literal(
      low_vgpr, static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Publishing), arch);
  const auto claim_compare = instrumentation::build_v_mov_b32_literal(
      high_vgpr, static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Empty), arch);
  const auto claim = instrumentation::build_flat_atomic_cmpswap_b32(
      *options.scratch_vgpr, low_vgpr, low_vgpr, /*return_old_value=*/true, kRdna4ScopeDevice,
      arch);
  const auto claimed =
      instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(static_cast<uint32_t>(
                                                  ConSanMoiSampledCausalPublicationState::Empty)),
                                              low_vgpr, arch);
  const auto narrow_winner =
      instrumentation::build_s_and_saveexec_b64(publication_exec_save_sgpr, kRdna4VccLo, arch);
  if (!claim_new || !claim_compare || !claim || !claimed || !narrow_winner ||
      !append_sampled_banked_address(
          words, causal_window_address + offsetof(ConSanMoiSampledCausalWindow, publication_state),
          sizeof(ConSanMoiSampledCausalWindow), window_bank_count, bank_vgpr, *options.scratch_vgpr,
          high_vgpr, arch)) {
    errors.emplace_back("ConSan MOI sampled probe could not claim a causal window slot");
    return std::nullopt;
  }
  words.insert(words.end(), claim_new->begin(), claim_new->end());
  words.insert(words.end(), claim_compare->begin(), claim_compare->end());
  words.insert(words.end(), claim->begin(), claim->end());
  if (!append_moi_flat_load_wait(words, arch))
    return std::nullopt;
  words.push_back(*claimed);
  if (!sequence.emit_branch(collision_label, InstructionSequence::BranchKind::VccZero))
    return std::nullopt;
  words.push_back(*narrow_winner);

  if (!append_sampled_banked_address(words, causal_window_address,
                                     sizeof(ConSanMoiSampledCausalWindow), window_bank_count,
                                     bank_vgpr, *options.scratch_vgpr, high_vgpr, arch))
    return std::nullopt;
  const auto store_dynamic_record_index = [&]() {
    const auto dynamic_record_index =
        instrumentation::build_v_add_u32_literal(high_vgpr, record_index, bank_vgpr, arch);
    if (!dynamic_record_index)
      return false;
    words.insert(words.end(), dynamic_record_index->begin(), dynamic_record_index->end());
    return record.store_vgpr(offsetof(ConSanMoiSampledCausalWindow, first_entry), high_vgpr);
  };
  if (!record.store_literal(offsetof(ConSanMoiSampledCausalWindow, generation),
                            static_cast<uint32_t>(options.moi_report_generation)) ||
      !record.store_literal(offsetof(ConSanMoiSampledCausalWindow, generation) + 4u,
                            static_cast<uint32_t>(options.moi_report_generation >> 32u)) ||
      !append_store_moi_report_dispatch_id_pair(
          record, options, offsetof(ConSanMoiSampledCausalWindow, dispatch_id), arch,
          ConSanMoiLiteralDispatchIdPolicy::AnyArchitecture) ||
      !record.store_workgroup(offsetof(ConSanMoiSampledCausalWindow, workgroup_x),
                              workgroup_sources->x,
                              ConSanMoiRecordEmitter::MissingWorkgroupSource::StoreZero) ||
      !record.store_workgroup(offsetof(ConSanMoiSampledCausalWindow, workgroup_y),
                              workgroup_sources->y,
                              ConSanMoiRecordEmitter::MissingWorkgroupSource::StoreZero) ||
      !record.store_workgroup(offsetof(ConSanMoiSampledCausalWindow, workgroup_z),
                              workgroup_sources->z,
                              ConSanMoiRecordEmitter::MissingWorkgroupSource::StoreZero) ||
      !(options.moi_epoch_vgpr
            ? record.store_vgpr(offsetof(ConSanMoiSampledCausalWindow, epoch),
                                *options.moi_epoch_vgpr)
            : record.store_literal(offsetof(ConSanMoiSampledCausalWindow, epoch), 0)) ||
      !store_dynamic_record_index() ||
      !record.store_literal(offsetof(ConSanMoiSampledCausalWindow, entry_count), 1) ||
      !record.store_workgroup(offsetof(ConSanMoiSampledCausalWindow, cluster_workgroup_id),
                              workgroup_sources->cluster_workgroup_id)) {
    errors.emplace_back("ConSan MOI sampled probe could not publish causal window metadata");
    return std::nullopt;
  }

  const auto mov_low = instrumentation::build_v_mov_b32_literal(low_vgpr, low_literal, arch);
  if (!mov_low) {
    errors.emplace_back("ConSan MOI sampled probe could not encode sampled entry low word");
    return std::nullopt;
  }
  words.insert(words.end(), mov_low->begin(), mov_low->end());
  if ((options.moi_owner_vgpr || derived_owner_vgpr) &&
      !append_add_shifted_vgpr_field(words, low_vgpr, owner_vgpr,
                                     consan_moi_sampled_watchpoint::owner_shift,
                                     consan_moi_sampled_watchpoint::max_owner, tmp_vgpr, arch)) {
    errors.emplace_back("ConSan MOI sampled probe could not encode owner field");
    return std::nullopt;
  }
  if (options.moi_epoch_vgpr &&
      !append_add_shifted_vgpr_field(words, low_vgpr, *options.moi_epoch_vgpr,
                                     consan_moi_sampled_watchpoint::epoch_shift,
                                     consan_moi_sampled_watchpoint::max_epoch, tmp_vgpr, arch)) {
    errors.emplace_back("ConSan MOI sampled probe could not encode epoch field");
    return std::nullopt;
  }
  uint16_t effective_lds_byte_offset_vgpr = *lds_byte_offset_vgpr;
  if (access_range.static_byte_offset != 0u) {
    if (!append_compute_effective_lds_byte_offset(words, tmp_vgpr, *lds_byte_offset_vgpr,
                                                  access_range.static_byte_offset, arch)) {
      errors.emplace_back("ConSan MOI sampled probe could not encode effective LDS byte offset");
      return std::nullopt;
    }
    effective_lds_byte_offset_vgpr = tmp_vgpr;
  }
  const auto start_cell_shift = instrumentation::build_v_lshrrev_b32(
      tmp_vgpr, scalar_positive_inline_u32(consan_moi_sampled_watchpoint::granule_shift),
      effective_lds_byte_offset_vgpr, arch);
  const auto high_from_start = instrumentation::build_v_lshlrev_b32(
      high_vgpr, scalar_positive_inline_u32(consan_moi_sampled_watchpoint::start_shift - 32u),
      tmp_vgpr, arch);
  if (!start_cell_shift || !high_from_start) {
    errors.emplace_back("ConSan MOI sampled probe could not encode sampled entry start cell");
    return std::nullopt;
  }
  words.push_back(*start_cell_shift);
  words.push_back(*high_from_start);
  if (!append_add_literal_field(words, high_vgpr, encoded_cell_count | encoded_generation_high,
                                tmp_vgpr, arch)) {
    errors.emplace_back("ConSan MOI sampled probe could not encode cell-count field");
    return std::nullopt;
  }
  for (uint32_t prior_record_index : prior_record_indices) {
    if (!append_direct_sampled_immediate_check(
            words, options, arch, prior_record_index, window_bank_count, bank_vgpr,
            sampled_watchpoints_offset, low_vgpr, high_vgpr, errors))
      return std::nullopt;
  }
  // A site is shared by all selected waves. Publish the complete packed entry
  // with one device-scope atomic exchange so concurrent writers cannot leave a
  // low word from one wave paired with a high word from another.
  const auto atomic_publish = instrumentation::build_flat_atomic_swap_b64(
      *options.scratch_vgpr, low_vgpr, low_vgpr, /*return_old_value=*/true, kRdna4ScopeDevice,
      arch);
  if (!atomic_publish || !append_sampled_banked_address(
                             words, sampled_entry_address, sizeof(uint64_t), window_bank_count,
                             bank_vgpr, *options.scratch_vgpr, tmp_vgpr, arch)) {
    errors.emplace_back("ConSan MOI sampled probe could not atomically publish sampled entry");
    return std::nullopt;
  }
  words.insert(words.end(), atomic_publish->begin(), atomic_publish->end());
  if (!append_moi_flat_load_wait(words, arch))
    return std::nullopt;
  if (!append_sampled_banked_address(words, causal_window_address,
                                     sizeof(ConSanMoiSampledCausalWindow), window_bank_count,
                                     bank_vgpr, *options.scratch_vgpr, tmp_vgpr, arch))
    return std::nullopt;
  if (!record.store_literal(offsetof(ConSanMoiSampledCausalWindow, publication_state),
                            static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Ready))) {
    errors.emplace_back("ConSan MOI sampled probe could not commit causal window metadata");
    return std::nullopt;
  }
  const uint64_t selected_count_address =
      *options.moi_report_buffer_address +
      offsetof(ConSanMoiReportHeader, sampled_causal_window_count);
  if (!append_atomic_fetch_add_one_u32(words, selected_count_address, tmp_vgpr,
                                       *options.scratch_vgpr, arch)) {
    errors.emplace_back("ConSan MOI sampled probe could not count the claimed causal window");
    return std::nullopt;
  }
  const auto restore_winner_exec =
      instrumentation::build_s_mov_b64(kRdna4ExecLo, publication_exec_save_sgpr, arch);
  if (!restore_winner_exec)
    return std::nullopt;
  words.push_back(*restore_winner_exec);
  if (!sequence.emit_branch(restore_label, InstructionSequence::BranchKind::Unconditional))
    return std::nullopt;

  const auto save_collision_exec =
      instrumentation::build_s_mov_b64(publication_exec_save_sgpr, kRdna4ExecLo, arch);
  if (!save_collision_exec || !sequence.bind(collision_label))
    return std::nullopt;
  words.push_back(*save_collision_exec);

  // Repeated executions of the same selected causal window are idempotent:
  // the static slot already retains their representative sample. Only a
  // different exact dynamic identity occupying the slot is capacity loss.
  if (!append_sampled_banked_address(words, causal_window_address,
                                     sizeof(ConSanMoiSampledCausalWindow), window_bank_count,
                                     bank_vgpr, *options.scratch_vgpr, high_vgpr, arch))
    return std::nullopt;
  const auto reject_unless_equal_vgpr = [&](uint32_t offset, uint16_t expected_vgpr) -> bool {
    if (!record.load(offset, low_vgpr))
      return false;
    const auto equal =
        instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(expected_vgpr), low_vgpr, arch);
    if (!equal)
      return false;
    words.push_back(*equal);
    return sequence.emit_branch(different_identity_label, InstructionSequence::BranchKind::VccZero);
  };
  const auto reject_unless_equal_literal = [&](uint32_t offset, uint32_t literal) -> bool {
    const auto expected = instrumentation::build_v_mov_b32_literal(high_vgpr, literal, arch);
    if (!expected)
      return false;
    words.insert(words.end(), expected->begin(), expected->end());
    return reject_unless_equal_vgpr(offset, high_vgpr);
  };
  const auto reject_unless_equal_dispatch_id = [&](uint32_t offset, bool high_word) -> bool {
    if (!record.load(offset, low_vgpr) || !append_compare_moi_report_dispatch_id_word(
                                              words, options, low_vgpr, high_vgpr, high_word, arch,
                                              ConSanMoiLiteralDispatchIdPolicy::AnyArchitecture)) {
      return false;
    }
    return sequence.emit_branch(different_identity_label, InstructionSequence::BranchKind::VccZero);
  };
  const auto reject_unless_equal_workgroup = [&](uint32_t offset,
                                                 const ConSanMoiWorkgroupSource &source) -> bool {
    if (!source.scalar_src)
      return reject_unless_equal_literal(offset, 0u);
    if (!consan_detail::append_workgroup_source_value(words, source, high_vgpr, arch))
      return false;
    return reject_unless_equal_vgpr(offset, high_vgpr);
  };
  const auto reject_unless_equal_dynamic_record_index = [&]() {
    const auto dynamic_record_index =
        instrumentation::build_v_add_u32_literal(high_vgpr, record_index, bank_vgpr, arch);
    if (!dynamic_record_index)
      return false;
    words.insert(words.end(), dynamic_record_index->begin(), dynamic_record_index->end());
    return reject_unless_equal_vgpr(offsetof(ConSanMoiSampledCausalWindow, first_entry), high_vgpr);
  };
  if (!reject_unless_equal_literal(offsetof(ConSanMoiSampledCausalWindow, generation),
                                   static_cast<uint32_t>(options.moi_report_generation)) ||
      !reject_unless_equal_literal(offsetof(ConSanMoiSampledCausalWindow, generation) + 4u,
                                   static_cast<uint32_t>(options.moi_report_generation >> 32u)) ||
      !reject_unless_equal_dispatch_id(offsetof(ConSanMoiSampledCausalWindow, dispatch_id),
                                       /*high_word=*/false) ||
      !reject_unless_equal_dispatch_id(offsetof(ConSanMoiSampledCausalWindow, dispatch_id) + 4u,
                                       /*high_word=*/true) ||
      !reject_unless_equal_workgroup(offsetof(ConSanMoiSampledCausalWindow, workgroup_x),
                                     workgroup_sources->x) ||
      !reject_unless_equal_workgroup(offsetof(ConSanMoiSampledCausalWindow, workgroup_y),
                                     workgroup_sources->y) ||
      !reject_unless_equal_workgroup(offsetof(ConSanMoiSampledCausalWindow, workgroup_z),
                                     workgroup_sources->z) ||
      !(options.moi_epoch_vgpr
            ? reject_unless_equal_vgpr(offsetof(ConSanMoiSampledCausalWindow, epoch),
                                       *options.moi_epoch_vgpr)
            : reject_unless_equal_literal(offsetof(ConSanMoiSampledCausalWindow, epoch), 0u)) ||
      !reject_unless_equal_dynamic_record_index() ||
      !reject_unless_equal_literal(offsetof(ConSanMoiSampledCausalWindow, entry_count), 1u) ||
      !reject_unless_equal_literal(
          offsetof(ConSanMoiSampledCausalWindow, publication_state),
          static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Ready)) ||
      !reject_unless_equal_workgroup(offsetof(ConSanMoiSampledCausalWindow, cluster_workgroup_id),
                                     workgroup_sources->cluster_workgroup_id)) {
    errors.emplace_back("ConSan MOI sampled probe could not compare a repeated causal identity");
    return std::nullopt;
  }
  const auto restore_duplicate_exec =
      instrumentation::build_s_mov_b64(kRdna4ExecLo, publication_exec_save_sgpr, arch);
  if (!restore_duplicate_exec)
    return std::nullopt;
  words.push_back(*restore_duplicate_exec);
  if (!sequence.emit_branch(restore_label, InstructionSequence::BranchKind::Unconditional))
    return std::nullopt;

  if (!sequence.bind(different_identity_label))
    return std::nullopt;
  const auto mbcnt_lo = instrumentation::build_v_mbcnt_lo_u32_b32(
      tmp_vgpr, publication_exec_save_sgpr, scalar_positive_inline_u32(0), arch);
  const auto mbcnt_hi = instrumentation::build_v_mbcnt_hi_u32_b32(
      tmp_vgpr, static_cast<uint16_t>(publication_exec_save_sgpr + 1u),
      vector_source_vgpr(tmp_vgpr), arch);
  const auto first_active_lane =
      instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), tmp_vgpr, arch);
  const auto narrow_collision =
      instrumentation::build_s_and_saveexec_b64(publication_exec_save_sgpr, kRdna4VccLo, arch);
  if (!mbcnt_lo || !mbcnt_hi || !first_active_lane || !narrow_collision) {
    errors.emplace_back("ConSan MOI sampled probe could not select a collision representative");
    return std::nullopt;
  }
  words.insert(words.end(), mbcnt_lo->begin(), mbcnt_lo->end());
  words.insert(words.end(), mbcnt_hi->begin(), mbcnt_hi->end());
  words.push_back(*first_active_lane);
  words.push_back(*narrow_collision);
  const uint64_t saturated_count_address =
      *options.moi_report_buffer_address +
      offsetof(ConSanMoiReportHeader, sampled_saturated_window_count);
  if (!append_atomic_fetch_add_one_u32(words, saturated_count_address, tmp_vgpr,
                                       *options.scratch_vgpr, arch)) {
    errors.emplace_back("ConSan MOI sampled probe could not count a saturated causal window claim");
    return std::nullopt;
  }
  const auto restore_collision_exec =
      instrumentation::build_s_mov_b64(kRdna4ExecLo, publication_exec_save_sgpr, arch);
  if (!restore_collision_exec)
    return std::nullopt;
  words.push_back(*restore_collision_exec);

  if (!sequence.bind(restore_label))
    return std::nullopt;
  const auto restore_vcc =
      instrumentation::build_s_mov_b64(kRdna4VccLo, selection_vcc_save_sgpr, arch);
  const auto restore_exec =
      instrumentation::build_s_mov_b64(kRdna4ExecLo, original_exec_save_sgpr, arch);
  const auto restore_scc = instrumentation::build_s_cmp_lg_u32(publication_scc_save_sgpr,
                                                               scalar_positive_inline_u32(0), arch);
  if (!restore_exec || !restore_vcc || !restore_scc) {
    errors.emplace_back("ConSan MOI sampled probe could not restore guest EXEC/VCC/SCC");
    return std::nullopt;
  }
  words.push_back(*restore_exec);
  words.push_back(*restore_vcc);
  words.push_back(*restore_scc);
  if (!sequence.resolve_branches(arch)) {
    errors.emplace_back("ConSan MOI sampled probe local branch is out of range");
    return std::nullopt;
  }
  return words;
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_direct_sampled_watchpoint_words(
    std::span<const uint8_t> bytes, const ConSanMoiCandidate &candidate,
    const ConSanOptions &options, rj_code_arch_t arch, uint32_t first_record_index,
    std::optional<uint32_t> prior_first_record_index, uint32_t prior_access_range_count,
    uint32_t window_bank_count, size_t sampled_causal_windows_offset,
    size_t sampled_watchpoints_offset, size_t sampled_pending_acquires_offset,
    std::optional<uint32_t> private_epoch_offset, std::optional<uint32_t> private_owner_offset,
    std::optional<uint16_t> private_owner_shift, std::vector<std::string> &errors,
    uint32_t *guest_instruction_offset = nullptr) {
  const auto access_ranges = candidate_access_ranges(bytes, candidate, arch);
  if (!access_ranges || access_ranges->empty()) {
    errors.emplace_back("ConSan MOI sampled probe requires at least one LDS access range");
    return std::nullopt;
  }

  std::vector<uint32_t> words;
  const bool select_guest_vgpr_bank =
      arch == ROCJITSU_CODE_ARCH_GFX1250 && candidate.gfx1250_vgpr_msb_mode.value_or(0u) != 0u;
  if (select_guest_vgpr_bank)
    words.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
        static_cast<uint8_t>(*candidate.gfx1250_vgpr_msb_mode), 0u, arch));
  if (private_epoch_offset) {
    if (!options.automatic_moi_private_epoch || !options.moi_epoch_vgpr) {
      errors.emplace_back("ConSan MOI sampled probe has an invalid private epoch plan");
      return std::nullopt;
    }
    const auto load = instrumentation::build_private_load_b32(*options.moi_epoch_vgpr,
                                                              *private_epoch_offset, arch);
    const auto wait = instrumentation::build_s_wait_private_load0(arch);
    if (!load || !wait) {
      errors.emplace_back("ConSan MOI sampled probe could not load private epoch state");
      return std::nullopt;
    }
    words.insert(words.end(), load->begin(), load->end());
    words.push_back(*wait);
  }
  if (private_owner_offset) {
    if (!options.automatic_moi_private_epoch || !options.moi_owner_vgpr || !private_owner_shift) {
      errors.emplace_back("ConSan MOI sampled probe has an invalid private owner plan");
      return std::nullopt;
    }
    const auto load = instrumentation::build_private_load_b32(*options.moi_owner_vgpr,
                                                              *private_owner_offset, arch);
    const auto wait = instrumentation::build_s_wait_private_load0(arch);
    const auto owner = instrumentation::build_v_lshrrev_b32(
        *options.moi_owner_vgpr, scalar_positive_inline_u32(*private_owner_shift),
        *options.moi_owner_vgpr, arch);
    if (!load || !wait || !owner) {
      errors.emplace_back("ConSan MOI sampled probe could not load private owner state");
      return std::nullopt;
    }
    words.insert(words.end(), load->begin(), load->end());
    words.push_back(*wait);
    words.push_back(*owner);
  }
  if (moi_uses_scalar_persistent_state(options)) {
    if (!consan_detail::validate_scalar_state_temporaries(options, "sampled probe", errors))
      return std::nullopt;
    words.push_back(
        build_v_mov_b32_e32(*options.moi_owner_vgpr, *options.moi_persistent_owner_sgpr, arch));
    words.push_back(
        build_v_mov_b32_e32(*options.moi_epoch_vgpr, *options.moi_persistent_epoch_sgpr, arch));
  }
  std::optional<uint16_t> preserved_lds_byte_offset_vgpr;
  const bool materialize_flat_offset =
      candidate_requires_gfx1250_flat_offset_materialization(candidate);
  if (moi_load_clobbers_address(candidate) || materialize_flat_offset) {
    if (!options.scratch_vgpr)
      return std::nullopt;
    const uint16_t base_scratch_count = static_cast<uint16_t>(
        (options.moi_sampled_check ? 7u : 5u) +
        (options.moi_runtime_sample_stride > 1 || window_bank_count > 1 ? 1u : 0u));
    preserved_lds_byte_offset_vgpr =
        static_cast<uint16_t>(*options.scratch_vgpr + base_scratch_count);
    const auto lds_byte_offset_vgpr = candidate_lds_byte_offset_vgpr(candidate, errors);
    if (!lds_byte_offset_vgpr)
      return std::nullopt;
    words.push_back(build_v_mov_b32_e32(*preserved_lds_byte_offset_vgpr,
                                        vector_source_vgpr(*lds_byte_offset_vgpr), arch));
    if (materialize_flat_offset) {
      words.push_back(build_v_mov_b32_e32(
          static_cast<uint16_t>(*preserved_lds_byte_offset_vgpr + 1u),
          vector_source_vgpr(static_cast<uint16_t>(*lds_byte_offset_vgpr + 1u)), arch));
    }
  }

  for (uint32_t range_index = 0; range_index < access_ranges->size(); ++range_index) {
    std::vector<uint32_t> prior_record_indices;
    if (prior_first_record_index) {
      prior_record_indices.reserve(prior_access_range_count);
      for (uint32_t prior_range_index = 0; prior_range_index < prior_access_range_count;
           ++prior_range_index) {
        prior_record_indices.push_back(*prior_first_record_index +
                                       prior_range_index * window_bank_count);
      }
    }
    uint32_t range_guest_instruction_offset = 0;
    auto range_words = build_direct_sampled_watchpoint_range_words(
        bytes, candidate, (*access_ranges)[range_index], options, arch,
        first_record_index + range_index * window_bank_count, prior_record_indices,
        window_bank_count, range_index == 0u, preserved_lds_byte_offset_vgpr,
        sampled_causal_windows_offset, sampled_watchpoints_offset, sampled_pending_acquires_offset,
        errors, range_index == 0u ? &range_guest_instruction_offset : nullptr);
    if (!range_words)
      return std::nullopt;
    if (range_index == 0u && guest_instruction_offset) {
      *guest_instruction_offset =
          static_cast<uint32_t>(words.size() * sizeof(uint32_t)) + range_guest_instruction_offset;
    }
    words.insert(words.end(), range_words->begin(), range_words->end());
  }
  if (select_guest_vgpr_bank)
    words.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
        0u, static_cast<uint8_t>(*candidate.gfx1250_vgpr_msb_mode), arch));
  return words;
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_sampled_runtime_fast_gate_island_words(
    std::span<const uint8_t> bytes, const ConSanMoiCandidate &candidate,
    const ConSanOptions &options, const ConSanMoiWorkgroupSources &workgroup_sources,
    [[maybe_unused]] uint32_t first_record_index, uint64_t island_text_offset,
    uint64_t cave_text_offset, uint64_t return_text_offset, uint64_t island_word_count,
    rj_code_arch_t arch) {
  if (!options.moi_exec_save_sgpr || options.moi_runtime_sample_stride <= 1u ||
      (!options.moi_dispatch_id_sgpr && !is_rdna4_family_arch(arch)))
    return std::nullopt;
  const uint16_t saved_scc = static_cast<uint16_t>(*options.moi_exec_save_sgpr + 4u);
  const uint16_t quotient = static_cast<uint16_t>(*options.moi_exec_save_sgpr + 5u);
  const uint16_t residue = static_cast<uint16_t>(*options.moi_exec_save_sgpr + 6u);
  const uint16_t shift =
      scalar_positive_inline_u32(std::countr_zero(options.moi_runtime_sample_stride));
  const uint32_t mask = options.moi_runtime_sample_stride - 1u;
  const uint32_t selected_residue = options.moi_runtime_sample_offset & mask;

  std::vector<uint32_t> words;
  words.reserve(island_word_count);
  const auto save_scc_word = instrumentation::build_s_cselect_b32(
      saved_scc, scalar_positive_inline_u32(1), scalar_positive_inline_u32(0), arch);
  const uint16_t dispatch_mix = options.moi_dispatch_id_sgpr.value_or(
      scalar_positive_inline_u32(static_cast<uint32_t>(options.moi_report_dispatch_id) & 63u));
  const auto mix_dispatch =
      instrumentation::build_s_sub_u32(residue, scalar_positive_inline_u32(0), dispatch_mix, arch);
  if (!save_scc_word || !mix_dispatch)
    return std::nullopt;
  words.push_back(*save_scc_word);
  words.push_back(*mix_dispatch);

  // Sample a dispatch/workgroup consistently at every site. A per-site sample
  // counter used to send producer and consumer sites into unrelated banks, so
  // the checker could never compare their evidence. Workgroup coordinates keep
  // the configured stride spatial without turning a selected dispatch into an
  // all-workgroups-at-once event.
  const auto append_workgroup_mix = [&](const ConSanMoiWorkgroupSource &source) -> bool {
    if (!source.scalar_src)
      return true;
    uint16_t coordinate = *source.scalar_src;
    if (source.mask_low_16) {
      words.push_back(build_s_lshl_b32(quotient, coordinate, scalar_positive_inline_u32(16), arch));
      words.push_back(build_s_lshr_b32(quotient, quotient, scalar_positive_inline_u32(16), arch));
      coordinate = quotient;
    } else if (source.shift_right_16) {
      words.push_back(build_s_lshr_b32(quotient, coordinate, scalar_positive_inline_u32(16), arch));
      coordinate = quotient;
    }
    const auto mix = instrumentation::build_s_sub_u32(residue, residue, coordinate, arch);
    if (!mix)
      return false;
    words.push_back(*mix);
    return true;
  };
  if (!append_workgroup_mix(workgroup_sources.x) || !append_workgroup_mix(workgroup_sources.y) ||
      !append_workgroup_mix(workgroup_sources.z))
    return std::nullopt;

  words.push_back(build_s_lshr_b32(quotient, residue, shift, arch));
  words.push_back(build_s_lshl_b32(quotient, quotient, shift, arch));
  const auto subtract = instrumentation::build_s_sub_u32(residue, residue, quotient, arch);
  const auto compare =
      instrumentation::build_s_cmp_eq_u32(residue, scalar_positive_inline_u32(0), arch);
  const auto restore_scc =
      instrumentation::build_s_cmp_lg_u32(saved_scc, scalar_positive_inline_u32(0), arch);
  if (!subtract || !compare || !restore_scc)
    return std::nullopt;
  words.push_back(*subtract);
  for (uint32_t remaining = selected_residue; remaining != 0u;) {
    const uint32_t chunk = std::min<uint32_t>(remaining, 64u);
    const auto subtract_selected =
        instrumentation::build_s_sub_u32(residue, residue, scalar_positive_inline_u32(chunk), arch);
    if (!subtract_selected)
      return std::nullopt;
    words.push_back(*subtract_selected);
    remaining -= chunk;
  }
  words.push_back(*compare);
  const size_t selected_branch_index = words.size();
  words.push_back(build_s_nop(0, arch));
  words.push_back(*restore_scc);
  for (uint64_t offset = 0; offset < candidate.size; offset += sizeof(uint32_t)) {
    uint32_t word = 0;
    std::memcpy(&word, bytes.data() + candidate.file_offset + offset, sizeof(word));
    words.push_back(word);
  }
  const size_t return_branch_index = words.size();
  words.push_back(build_s_nop(0, arch));
  const size_t selected_index = words.size();
  const auto branch_delta = [](size_t branch, size_t target) -> std::optional<int16_t> {
    const int64_t delta = static_cast<int64_t>(target) - static_cast<int64_t>(branch) - 1;
    if (delta < std::numeric_limits<int16_t>::min() || delta > std::numeric_limits<int16_t>::max())
      return std::nullopt;
    return static_cast<int16_t>(delta);
  };
  const auto selected_offset = branch_delta(selected_branch_index, selected_index);
  const uint64_t return_branch_text_offset =
      island_text_offset + return_branch_index * sizeof(uint32_t);
  const auto return_offset =
      compute_sopp_branch_simm16(return_branch_text_offset, return_text_offset);
  if (!selected_offset || !return_offset)
    return std::nullopt;
  const auto select = instrumentation::build_s_cbranch_scc1(*selected_offset, arch);
  if (!select)
    return std::nullopt;
  words[selected_branch_index] = *select;
  words[return_branch_index] = build_s_branch(*return_offset, arch);
  if (!append_moi_scc_preserving_indirect_jump(words, island_text_offset, cave_text_offset,
                                               *options.moi_exec_save_sgpr, saved_scc,
                                               /*capture_scc=*/false, arch) ||
      words.size() > island_word_count)
    return std::nullopt;
  words.resize(island_word_count, build_s_nop(0, arch));
  return words;
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_sampled_runtime_fast_gate_call_words(
    std::span<const uint8_t> bytes, const ConSanMoiCandidate &candidate,
    const ConSanOptions &options, const ConSanMoiWorkgroupSources &workgroup_sources,
    uint64_t gate_text_offset, uint64_t return_text_offset, uint64_t reserved_word_count,
    rj_code_arch_t arch) {
  if (!options.moi_exec_save_sgpr || options.moi_runtime_sample_stride <= 1u ||
      (!options.moi_dispatch_id_sgpr && !is_rdna4_family_arch(arch)))
    return std::nullopt;
  const uint16_t saved_scc = static_cast<uint16_t>(*options.moi_exec_save_sgpr + 4u);
  const uint16_t quotient = static_cast<uint16_t>(*options.moi_exec_save_sgpr + 5u);
  const uint16_t residue = static_cast<uint16_t>(*options.moi_exec_save_sgpr + 6u);
  const uint16_t shift =
      scalar_positive_inline_u32(std::countr_zero(options.moi_runtime_sample_stride));
  const uint32_t mask = options.moi_runtime_sample_stride - 1u;
  const uint32_t selected_residue = options.moi_runtime_sample_offset & mask;

  std::vector<uint32_t> words;
  words.reserve(reserved_word_count);
  const auto save_scc_word = instrumentation::build_s_cselect_b32(
      saved_scc, scalar_positive_inline_u32(1), scalar_positive_inline_u32(0), arch);
  const uint16_t dispatch_mix = options.moi_dispatch_id_sgpr.value_or(
      scalar_positive_inline_u32(static_cast<uint32_t>(options.moi_report_dispatch_id) & 63u));
  const auto mix_dispatch =
      instrumentation::build_s_sub_u32(residue, scalar_positive_inline_u32(0), dispatch_mix, arch);
  if (!save_scc_word || !mix_dispatch)
    return std::nullopt;
  words.push_back(*save_scc_word);
  words.push_back(*mix_dispatch);
  const auto append_workgroup_mix = [&](const ConSanMoiWorkgroupSource &source) -> bool {
    if (!source.scalar_src)
      return true;
    uint16_t coordinate = *source.scalar_src;
    if (source.mask_low_16) {
      words.push_back(build_s_lshl_b32(quotient, coordinate, scalar_positive_inline_u32(16), arch));
      words.push_back(build_s_lshr_b32(quotient, quotient, scalar_positive_inline_u32(16), arch));
      coordinate = quotient;
    } else if (source.shift_right_16) {
      words.push_back(build_s_lshr_b32(quotient, coordinate, scalar_positive_inline_u32(16), arch));
      coordinate = quotient;
    }
    const auto mix = instrumentation::build_s_sub_u32(residue, residue, coordinate, arch);
    if (!mix)
      return false;
    words.push_back(*mix);
    return true;
  };
  if (!append_workgroup_mix(workgroup_sources.x) || !append_workgroup_mix(workgroup_sources.y) ||
      !append_workgroup_mix(workgroup_sources.z))
    return std::nullopt;
  words.push_back(build_s_lshr_b32(quotient, residue, shift, arch));
  words.push_back(build_s_lshl_b32(quotient, quotient, shift, arch));
  const auto subtract = instrumentation::build_s_sub_u32(residue, residue, quotient, arch);
  const auto compare =
      instrumentation::build_s_cmp_eq_u32(residue, scalar_positive_inline_u32(0), arch);
  const auto restore_scc =
      instrumentation::build_s_cmp_lg_u32(saved_scc, scalar_positive_inline_u32(0), arch);
  if (!subtract || !compare || !restore_scc)
    return std::nullopt;
  words.push_back(*subtract);
  for (uint32_t remaining = selected_residue; remaining != 0u;) {
    const uint32_t chunk = std::min<uint32_t>(remaining, 64u);
    const auto subtract_selected =
        instrumentation::build_s_sub_u32(residue, residue, scalar_positive_inline_u32(chunk), arch);
    if (!subtract_selected)
      return std::nullopt;
    words.push_back(*subtract_selected);
    remaining -= chunk;
  }
  words.push_back(*compare);
  const size_t selected_branch_index = words.size();
  words.push_back(build_s_nop(0, arch));
  words.push_back(*restore_scc);
  for (uint64_t offset = 0; offset < candidate.size; offset += sizeof(uint32_t)) {
    uint32_t word = 0;
    std::memcpy(&word, bytes.data() + candidate.file_offset + offset, sizeof(word));
    words.push_back(word);
  }
  if (!append_moi_scc_preserving_indirect_jump(words, gate_text_offset, return_text_offset,
                                               *options.moi_exec_save_sgpr, saved_scc,
                                               /*capture_scc=*/false, arch))
    return std::nullopt;
  const size_t selected_index = words.size();
  words.push_back(*restore_scc);
  const int64_t branch_delta =
      static_cast<int64_t>(selected_index) - static_cast<int64_t>(selected_branch_index) - 1;
  if (branch_delta < std::numeric_limits<int16_t>::min() ||
      branch_delta > std::numeric_limits<int16_t>::max() || words.size() > reserved_word_count)
    return std::nullopt;
  const auto select =
      instrumentation::build_s_cbranch_scc1(static_cast<int16_t>(branch_delta), arch);
  if (!select)
    return std::nullopt;
  words[selected_branch_index] = *select;
  words.resize(reserved_word_count, build_s_nop(0, arch));
  return words;
}
