[[nodiscard]] std::optional<ConSanMoiAtomicEventKind>
atomic_event_kind_for_site(const MoiSyncInventoryIndex &sync_index,
                           std::string_view container_name, bool in_kernel,
                           const ConSanAtomicSite &site);

void try_apply_direct_sampled_watchpoint_patch(std::span<const uint8_t> bytes,
                                               const ConSanOptions &options, rj_code_arch_t arch,
                                               const MoiResourcePlanningState &resource_state,
                                               ConSanResult &result) {
  if (!options.moi_report_buffer_address)
    return;
  if (!instrumentation::is_admitted_arch(arch)) {
    result.warnings.emplace_back("ConSan MOI sampled probe does not support this architecture");
    return;
  }

  const ConSanMoiReportBufferLayout layout = resolve_sampled_report_layout(options);
  const uint32_t total_budget = std::min(options.max_patches, layout.sampled_watchpoint_capacity);
  uint32_t qualified_atomic_sequences = 0;
  if (options.moi_track_atomics) {
    for (const ConSanSyncSequence &sequence : result.sync_sequences) {
      if (sequence.kind != ConSanSyncSequenceKind::Atomic || sequence.width_bits != 32u ||
          !consan_sync_confidence_meets(sequence.confidence,
                                        ConSanSemanticConfidence::Conservative) ||
          !consan_sync_confidence_meets(sequence.memory_role_confidence,
                                        ConSanSemanticConfidence::Conservative) ||
          (sequence.memory_role != ConSanSyncMemoryRole::Release &&
           sequence.memory_role != ConSanSyncMemoryRole::Acquire &&
           sequence.memory_role != ConSanSyncMemoryRole::AcquireRelease) ||
          (sequence.rmw_outcome != ConSanSyncRmwOutcome::NoReturn &&
           sequence.rmw_outcome != ConSanSyncRmwOutcome::ReturnsOldValue &&
           sequence.rmw_outcome != ConSanSyncRmwOutcome::CompareExchange))
        continue;
      bool eligible = false;
      for (const std::string &identity : sequence.member_event_identities) {
        const auto event =
            std::ranges::find(result.sync_events, identity, &ConSanSyncEvent::identity);
        if (event == result.sync_events.end() || event->kind != ConSanSyncEventKind::Atomic)
          continue;
        const ConSanAtomicSite *site = nullptr;
        if (event->in_kernel) {
          const auto kernel =
              std::ranges::find(result.kernels, event->container_name, &ConSanKernelInfo::name);
          if (kernel != result.kernels.end()) {
            const auto found = std::ranges::find(kernel->atomic_sites, event->text_offset,
                                                 &ConSanAtomicSite::text_offset);
            if (found != kernel->atomic_sites.end())
              site = &*found;
          }
        } else {
          const auto function =
              std::ranges::find(result.functions, event->container_name, &ConSanFunctionInfo::name);
          if (function != result.functions.end()) {
            const auto found = std::ranges::find(function->atomic_sites, event->text_offset,
                                                 &ConSanAtomicSite::text_offset);
            if (found != function->atomic_sites.end())
              site = &*found;
          }
        }
        const bool supported_scope = sequence.raw_scope
                                         ? *sequence.raw_scope >= 1u && *sequence.raw_scope <= 3u
                                         : result.arch == ROCJITSU_CODE_ARCH_GFX1250 &&
                                               site != nullptr && site->mnemonic.starts_with("ds_");
        if (site != nullptr && supported_scope &&
            (site->mnemonic.starts_with("flat_atomic") ||
             site->mnemonic.starts_with("global_atomic") ||
             (result.arch == ROCJITSU_CODE_ARCH_GFX1250 && site->mnemonic.starts_with("ds_"))) &&
            (sequence.rmw_outcome != ConSanSyncRmwOutcome::CompareExchange ||
             site->returns_old_value.value_or(false))) {
          eligible = true;
          break;
        }
      }
      if (eligible && qualified_atomic_sequences != std::numeric_limits<uint32_t>::max())
        ++qualified_atomic_sequences;
    }
  }
  const uint32_t reserved_atomic_slots = std::min(total_budget, qualified_atomic_sequences);
  const uint32_t max_candidates = total_budget - reserved_atomic_slots;
  if (max_candidates == 0) {
    if (total_budget == 0) {
      result.warnings.emplace_back(
          "ConSan MOI sampled probe requires room for the report header and one sampled entry");
    }
    return;
  }
  if (options.moi_sample_stride == 0) {
    result.warnings.emplace_back("ConSan MOI sampled probe requires a nonzero sample stride");
    return;
  }
  if (options.moi_sample_offset >= options.moi_sample_stride) {
    result.warnings.emplace_back("ConSan MOI sampled probe sample offset must be less than stride");
    return;
  }
  if (options.moi_runtime_sample_stride == 0 || options.moi_runtime_sample_stride > (1u << 24u) ||
      (options.moi_runtime_sample_stride & (options.moi_runtime_sample_stride - 1u)) != 0 ||
      options.moi_runtime_sample_offset >= options.moi_runtime_sample_stride) {
    result.warnings.emplace_back(
        "ConSan MOI runtime sample stride must be a power of two in 1..16777216 and offset must "
        "be less than stride");
    return;
  }
  std::vector<const ConSanMoiCandidate *> candidates =
      find_first_light_access_record_candidates(result, bytes, arch);
  apply_test_kernel_filter(candidates, options);
  std::ranges::sort(candidates, [](const ConSanMoiCandidate *lhs, const ConSanMoiCandidate *rhs) {
    return lhs->text_offset < rhs->text_offset;
  });
  // Sampled atomic metadata wraps the relocated guest instruction in a
  // second text-growth pass. Keep dual-role access/ordering sites at the tail
  // so their one-word nested anchors remain branch-reachable in large images.
  const MoiSyncInventoryIndex sync_index(result);
  std::unordered_set<uint64_t> ordering_atomic_offsets;
  if (options.moi_track_atomics) {
    const auto append_ordering_offsets = [&](const auto &container, bool in_kernel) {
      for (const ConSanAtomicSite &site : container.atomic_sites) {
        if (atomic_event_kind_for_site(sync_index, container.name, in_kernel, site))
          ordering_atomic_offsets.insert(site.text_offset);
      }
    };
    for (const ConSanKernelInfo &kernel : result.kernels)
      append_ordering_offsets(kernel, true);
    for (const ConSanFunctionInfo &function : result.functions)
      append_ordering_offsets(function, false);
  }
  std::stable_partition(candidates.begin(), candidates.end(), [&](const auto *candidate) {
    return !ordering_atomic_offsets.contains(candidate->text_offset);
  });
  if (candidates.empty()) {
    result.warnings.emplace_back(
        "ConSan MOI sampled probe found no native LDS or likely group flat load/store candidate");
    return;
  }

  uint32_t sampled_filter_candidate_count = 0;
  uint32_t sampled_filter_range_count = 0;
  for (uint32_t i = 0; i < candidates.size(); ++i) {
    if (i % options.moi_sample_stride == options.moi_sample_offset) {
      ++sampled_filter_candidate_count;
      const auto ranges = candidate_access_ranges(bytes, *candidates[i], arch);
      if (ranges &&
          ranges->size() <= std::numeric_limits<uint32_t>::max() - sampled_filter_range_count)
        sampled_filter_range_count += static_cast<uint32_t>(ranges->size());
    }
  }
  uint32_t window_bank_count = 1;
  if (options.moi_runtime_sample_stride > 1 && sampled_filter_range_count != 0) {
    const uint32_t available_per_range = max_candidates / sampled_filter_range_count;
    while (window_bank_count < 8u && window_bank_count * 2u <= available_per_range)
      window_bank_count *= 2u;
  }

  struct PlannedSampledPatch {
    const ConSanMoiCandidate *candidate = nullptr;
    uint32_t first_record_index = 0;
    uint32_t access_range_count = 0;
    std::optional<uint32_t> prior_first_record_index;
    uint32_t prior_access_range_count = 0;
    uint32_t runtime_sample_index = 0;
    DbiPatchPlacement placement;
    std::optional<uint64_t> entry_island_offset;
    uint64_t entry_island_word_count = 0;
    bool entry_island_at_anchor = false;
    bool dense_call_anchor = false;
    std::optional<uint64_t> dense_dispatcher_offset;
    std::vector<uint32_t> displaced_tail_words;
    uint64_t workgroup_descriptor_file_offset = 0;
    ResolvedMoiScratchPlan resources;
    std::optional<VgprSpillSequence> spill;
    std::optional<InlineSgprSpillSequence> scalar_spill;
    std::optional<uint16_t> gfx1250_vgpr_msb_mode;
    bool runtime_fast_gate = false;
    bool dense_runtime_fast_gate = false;
    std::optional<uint32_t> private_epoch_offset;
    std::optional<uint32_t> private_owner_offset;
    std::optional<uint16_t> private_owner_shift;
    uint32_t persistent_private_state_end = 0;
    uint32_t required_private_bytes = 0;
    uint32_t probe_guest_instruction_offset = 0;
    std::optional<uint16_t> transient_exec_save_sgpr;
    bool spill_overlaps_guest_operands = false;
    bool wrap_embedded_guest_vgpr_bank = false;
  };
  std::vector<PlannedSampledPatch> planned_patches;
  planned_patches.reserve(candidates.size());
  AmdGpuCodeObject code_object(bytes.data(), bytes.size());
  const uint64_t original_text_size =
      code_object.text_sections().size() == 1 ? code_object.text_sections().front()->size() : 0;
  DbiPatchPlacementPlanner placement_planner(arch, original_text_size);
  constexpr uint64_t kIndirectIslandWords = 7u;
  // The fast gate has 21 fixed worst-case words: dispatch/workgroup mixing for
  // all three gfx1250 coordinates, branches, SCC preservation, and the long
  // jump.  Add the displaced instruction and one subtract per 64 positions in
  // the configured residue.  Reserving the old unconditional 40 words for an
  // offset-zero profile made a thousand-site image exceed direct-branch reach
  // even though each emitted gate only needed about two dozen words.
  uint64_t max_candidate_instruction_words = 0u;
  for (const ConSanMoiCandidate *candidate : candidates) {
    max_candidate_instruction_words = std::max(
        max_candidate_instruction_words,
        (static_cast<uint64_t>(candidate->size) + sizeof(uint32_t) - 1u) / sizeof(uint32_t));
  }
  const uint64_t selected_runtime_residue =
      options.moi_runtime_sample_offset & (options.moi_runtime_sample_stride - 1u);
  const uint64_t sampled_access_island_words =
      21u + max_candidate_instruction_words + (selected_runtime_residue + 63u) / 64u;
  const uint64_t access_island_word_count =
      options.moi_runtime_sample_stride > 1u ? sampled_access_island_words : kIndirectIslandWords;
  const uint64_t access_relay_words = arch == ROCJITSU_CODE_ARCH_GFX1250
                                          ? std::max<uint64_t>(20u, access_island_word_count)
                                          : access_island_word_count;
  const bool has_function_candidate = std::ranges::any_of(
      candidates, [](const ConSanMoiCandidate *candidate) { return !candidate->in_kernel; });
  const bool has_spill_backed_scalar_assignment = std::ranges::any_of(
      options.moi_transient_sgpr_assignments, &ConSanMoiTransientSgprAssignment::spill_backed);
  const bool use_indirect_appended =
      options.moi_exec_save_sgpr.has_value() &&
      (candidates.size() > 8u || has_function_candidate || has_spill_backed_scalar_assignment);
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
  uint64_t indirect_body_cursor = indirect_body_begin;
  std::span<const uint8_t> original_text;
  if (code_object.text_sections().size() == 1) {
    const Section *text = code_object.text_sections().front();
    original_text =
        std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(text->data()), text->size());
  }
  std::optional<MoiLocalNopIslandAllocator> local_entry_islands;
  if (use_indirect_appended && !original_text.empty()) {
    const uint32_t local_entry_island_words =
        has_spill_backed_scalar_assignment ? kIndirectIslandWords + 1u : kIndirectIslandWords;
    local_entry_islands.emplace(original_text, result, arch, local_entry_island_words,
                                options.preapplied_reserved_ranges);
  }
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
  // Resource planning has already decoded the complete pristine CFG and keeps
  // it alive through lowering. Rebuilding it here solely to find a relay host
  // is particularly expensive for generated libraries containing thousands
  // of kernels, and cannot reveal any instruction that is absent from the
  // planning graph.
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
    for (auto &[key, owner_candidates] : owner_groups) {
      std::ranges::sort(owner_candidates, {}, &ConSanMoiCandidate::text_offset);
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
      const uint32_t group_entry_island_words = group_options.automatic_moi_record_replay_sgpr_spill
                                                    ? kIndirectIslandWords + 1u
                                                    : kIndirectIslandWords;
      bool has_dense_neighbors = false;
      for (size_t index = 1; index < group.candidates.size(); ++index) {
        has_dense_neighbors |=
            group.candidates[index]->text_offset <
            group.candidates[index - 1u]->text_offset + group_entry_island_words * sizeof(uint32_t);
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
                                             group_entry_island_words);
      // gfx1250 also uses the call relay to compact nearby anchors. RDNA4 has
      // no scalar call instruction, so use an explicit key plus branch only
      // when the ordinary per-site relay is stranded.
      if (!explicit_anchors_fit || singleton_has_direct_island ||
          (!(arch == ROCJITSU_CODE_ARCH_GFX1250 && has_dense_neighbors) && !has_stranded_entry))
        continue;
      std::vector<uint64_t> anchors;
      anchors.reserve(group.candidates.size());
      for (const ConSanMoiCandidate *candidate : group.candidates)
        anchors.push_back(candidate->text_offset);
      const uint64_t dispatcher =
          access_island_begin + group.first_candidate_index * access_relay_words * sizeof(uint32_t);
      auto island =
          local_entry_islands->claim_reachable_from_all(anchors, group_entry_island_words);
      if (!island && key.first) {
        const auto kernel = std::ranges::find(result.kernels, key.second, &ConSanKernelInfo::name);
        if (kernel != result.kernels.end()) {
          const size_t host_words_count = group_entry_island_words + 1u;
          const uint64_t host_bytes = host_words_count * sizeof(uint32_t);
          const uint64_t kernel_end = kernel->entry_text_offset + kernel->code_size;
          std::vector<std::pair<uint64_t, uint64_t>> occupied_ranges;
          occupied_ranges.reserve(group.candidates.size() + kernel->barrier_sites.size() +
                                  kernel->fence_sites.size() + kernel->atomic_sites.size());
          const auto append_occupied_range = [&](uint64_t begin, uint64_t size) {
            if (size != 0u)
              occupied_ranges.emplace_back(begin, begin + size);
          };
          for (const ConSanMoiCandidate *candidate : group.candidates)
            append_occupied_range(candidate->text_offset, candidate->size);
          for (const ConSanBarrierSite &site : kernel->barrier_sites)
            append_occupied_range(site.text_offset, site.size);
          for (const ConSanFenceSite &site : kernel->fence_sites)
            append_occupied_range(site.text_offset, site.size);
          for (const ConSanAtomicSite &site : kernel->atomic_sites)
            append_occupied_range(site.text_offset, site.size);
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
                (*block)->end_offset() > kernel_end || (*block)->size() < host_bytes)
              continue;
            for (const Instruction &instruction : (*block)->instructions()) {
              const uint64_t host_offset = instruction.src_loc();
              if (host_offset + host_bytes > (*block)->end_offset() ||
                  overlaps_instrumented_site(host_offset, host_offset + host_bytes))
                continue;
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
              while (displaced_words.size() < host_words_count) {
                auto decoded = decode_relocatable_entry_instruction(
                    original_text, host_offset + displaced_words.size() * sizeof(uint32_t), arch,
                    decode_errors, resource_state.decoder.get());
                if (!decoded) {
                  displaced_words.clear();
                  break;
                }
                displaced_words.insert(displaced_words.end(), decoded->begin(), decoded->end());
              }
              if (displaced_words.size() != host_words_count)
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
  uint32_t sampled_candidate_index = 0;
  uint32_t next_record_index = 0;
  bool capacity_limited = false;
  MoiSpillManagers spill_managers;
  std::unordered_map<uint64_t, std::optional<MoiPrivateEpochLayout>> private_layout_by_descriptor;
  uint64_t planned_anchor_end = 0;
  for (const ConSanMoiCandidate *candidate_ptr : candidates) {
    const uint32_t current_sample_index = sampled_candidate_index++;
    if (current_sample_index % options.moi_sample_stride != options.moi_sample_offset)
      continue;

    const ConSanMoiCandidate &candidate = *candidate_ptr;
    const bool use_dense_call = dense_entry_islands.contains(candidate_ptr);
    if (use_indirect_appended && !use_dense_call && candidate.text_offset < planned_anchor_end) {
      result.warnings.emplace_back(
          "ConSan MOI sampled probe skipped load/store site inside a relocated prefix");
      continue;
    }
    const auto access_ranges = candidate_access_ranges(bytes, candidate, arch);
    if (!access_ranges || access_ranges->empty())
      continue;
    const uint32_t range_count = static_cast<uint32_t>(access_ranges->size());
    const uint32_t required_slots = range_count * window_bank_count;
    if (required_slots > max_candidates - next_record_index) {
      capacity_limited = true;
      continue;
    }
    const auto resources = resolve_moi_scratch(result, candidate, options,
                                               direct_sampled_scratch_count(options, candidate));
    if (!resources) {
      continue;
    }
    std::optional<uint64_t> workgroup_descriptor = candidate.kernel_descriptor_file_offset;
    if (!workgroup_descriptor) {
      workgroup_descriptor =
          common_moi_record_owner_descriptor(bytes, *resources, arch, result.warnings);
    }
    if (!workgroup_descriptor)
      continue;
    std::optional<MoiPrivateEpochLayout> private_layout;
    std::optional<uint16_t> private_owner_shift;
    if (options.automatic_moi_private_epoch) {
      if (resources->owner_descriptor_file_offsets.size() == 1u) {
        const uint64_t descriptor = resources->owner_descriptor_file_offsets.front();
        auto [cached, inserted] =
            private_layout_by_descriptor.try_emplace(descriptor, std::nullopt);
        if (inserted) {
          cached->second = build_sampled_private_epoch_layout(result, *resources, options, arch,
                                                              result.warnings);
        }
        private_layout = cached->second;
      } else {
        private_layout =
            build_sampled_private_epoch_layout(result, *resources, options, arch, result.warnings);
      }
      if (private_layout && private_layout->owner_offset)
        private_owner_shift = common_moi_private_owner_shift(bytes, *resources, result.warnings);
      if (!private_layout || (private_layout->owner_offset && !private_owner_shift))
        continue;
    }
    ConSanOptions candidate_options = options;
    if (!apply_moi_transient_sgpr_assignment(candidate_options,
                                             resources->owner_descriptor_file_offsets)) {
      result.warnings.emplace_back(
          "ConSan MOI sampled probe lost its owner-local transient scalar assignment");
      continue;
    }
    candidate_options.scratch_vgpr = resources->base;
    std::optional<VgprSpillSequence> spill;
    if (resources->source == ConSanRegisterAllocationSource::SpillRequired ||
        moi_scalar_spill_requires_dynamic_vgpr_frame(result, *resources, candidate_options)) {
      spill = build_moi_spill_sequence(
          result, *resources, candidate_options, spill_managers, arch, result.warnings,
          private_layout ? std::optional<uint32_t>(private_layout->ephemeral_base) : std::nullopt);
      if (!spill)
        continue;
    }
    uint32_t required_private_bytes = private_layout ? private_layout->ephemeral_base : 0u;
    if (spill)
      required_private_bytes = std::max(required_private_bytes, spill->total_private_bytes);
    bool spill_overlaps_guest_operands = false;
    if (spill &&
        sampled_access_can_emit_spill_over_guest_operands(candidate_options, candidate)) {
      std::vector<std::string> ignored_overlap_errors;
      spill_overlaps_guest_operands = reject_candidate_scratch_range_overlap(
          candidate, resources->base, resources->count, ignored_overlap_errors);
      candidate_options.moi_spill_overlaps_guest_operands = spill_overlaps_guest_operands;
    }
    const uint32_t candidate_entry_island_words =
        candidate_options.automatic_moi_record_replay_sgpr_spill ? kIndirectIslandWords + 1u
                                                                 : kIndirectIslandWords;
    std::optional<InlineSgprSpillSequence> scalar_spill;
    if (candidate_options.automatic_moi_record_replay_sgpr_spill) {
      scalar_spill = build_inline_sgpr_spill_sequence(
          result, *resources, candidate_options, spill_managers, arch, result.warnings,
          private_layout ? std::optional<uint32_t>(private_layout->ephemeral_base) : std::nullopt,
          spill ? &*spill : nullptr);
      if (!scalar_spill)
        continue;
      required_private_bytes = std::max(required_private_bytes, scalar_spill->total_private_bytes);
    }
    std::optional<uint64_t> entry_island_offset;
    bool uses_local_entry_island = false;
    bool entry_island_at_anchor = false;
    std::vector<uint32_t> displaced_tail_words;
    uint32_t displaced_original_size = candidate.size;
    std::vector<std::string> candidate_errors;
    if (use_indirect_appended) {
      entry_island_offset =
          use_dense_call ? std::optional<uint64_t>(dense_entry_islands.at(candidate_ptr))
                         : std::optional<uint64_t>(access_island_begin +
                                                   candidate_indices.at(candidate_ptr) *
                                                       access_relay_words * sizeof(uint32_t));
      if (!use_dense_call && candidate.size >= sizeof(uint32_t) &&
          !compute_sopp_branch_simm16(candidate.text_offset, *entry_island_offset)) {
        entry_island_offset =
            local_entry_islands ? local_entry_islands->claim_reachable(candidate.text_offset,
                                                                       candidate_entry_island_words)
                                : std::nullopt;
        uses_local_entry_island = entry_island_offset.has_value();
      }
      if (!use_dense_call && !entry_island_offset && !original_text.empty()) {
        std::vector<uint32_t> displaced_words;
        while (displaced_words.size() < candidate_entry_island_words) {
          auto decoded = decode_relocatable_entry_instruction(
              original_text, candidate.text_offset + displaced_words.size() * sizeof(uint32_t),
              arch, candidate_errors, resource_state.decoder.get());
          if (!decoded) {
            displaced_words.clear();
            break;
          }
          displaced_words.insert(displaced_words.end(), decoded->begin(), decoded->end());
        }
        if (displaced_words.size() >= candidate_entry_island_words &&
            displaced_words.size() * sizeof(uint32_t) <= std::numeric_limits<uint32_t>::max()) {
          const size_t candidate_words = candidate.size / sizeof(uint32_t);
          if (candidate_words <= displaced_words.size()) {
            displaced_tail_words.assign(displaced_words.begin() + candidate_words,
                                        displaced_words.end());
            displaced_original_size =
                static_cast<uint32_t>(displaced_words.size() * sizeof(uint32_t));
            entry_island_offset = candidate.text_offset;
            uses_local_entry_island = true;
            entry_island_at_anchor = true;
          }
        }
      }
      if (candidate.size < sizeof(uint32_t) || !entry_island_offset) {
        result.warnings.emplace_back(
            "ConSan MOI sampled probe skipped load/store site: entry island is unreachable");
        continue;
      }
    }
    if (private_layout || moi_uses_scalar_persistent_state(candidate_options)) {
      uint16_t state_vgpr = static_cast<uint16_t>(resources->base + resources->count);
      if (!private_layout || private_layout->owner_offset)
        candidate_options.moi_owner_vgpr = --state_vgpr;
      candidate_options.moi_epoch_vgpr = --state_vgpr;
    }
    const bool runtime_fast_gate =
        !candidate_options.automatic_moi_record_replay_sgpr_spill && use_indirect_appended &&
        !use_dense_call && !uses_local_entry_island && options.moi_runtime_sample_stride > 1u &&
        options.moi_runtime_sample_offset <= 1024u;
    const bool dense_runtime_fast_gate =
        !candidate_options.automatic_moi_record_replay_sgpr_spill && use_dense_call &&
        options.moi_runtime_sample_stride > 1u && options.moi_runtime_sample_offset <= 1024u;
    ConSanMoiCandidate lowered_candidate = candidate;
    lowered_candidate.kernel_descriptor_file_offset = workgroup_descriptor;
    const auto prior_patch = std::find_if(
        planned_patches.rbegin(), planned_patches.rend(), [&](const PlannedSampledPatch &planned) {
          return planned.workgroup_descriptor_file_offset == *workgroup_descriptor;
        });
    const std::optional<uint32_t> prior_first_record_index =
        prior_patch == planned_patches.rend()
            ? std::nullopt
            : std::optional<uint32_t>(prior_patch->first_record_index);
    const uint32_t prior_access_range_count =
        prior_patch == planned_patches.rend() ? 0u : prior_patch->access_range_count;
    uint32_t probe_guest_instruction_offset = 0;
    auto words = build_direct_sampled_watchpoint_words(
        bytes, lowered_candidate, candidate_options, arch, next_record_index,
        prior_first_record_index, prior_access_range_count, window_bank_count,
        layout.sampled_causal_windows_offset, layout.sampled_watchpoints_offset,
        layout.sampled_pending_acquires_offset,
        private_layout ? std::optional<uint32_t>(private_layout->epoch_offset) : std::nullopt,
        private_layout ? private_layout->owner_offset : std::nullopt, private_owner_shift,
        candidate_errors, &probe_guest_instruction_offset);
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
                     : 0u;
    const bool wrap_embedded_guest_vgpr_bank = use_indirect_appended &&
                                               arch == ROCJITSU_CODE_ARCH_GFX1250 &&
                                               candidate.gfx1250_vgpr_msb_mode.value_or(0u) != 0u;
    const uint64_t vgpr_msb_bytes = use_indirect_appended
                                        ? moi_appended_body_vgpr_msb_transition_word_count(
                                              arch, candidate.gfx1250_vgpr_msb_mode,
                                              spill.has_value(), wrap_embedded_guest_vgpr_bank) *
                                              sizeof(uint32_t)
                                        : 0u;
    const uint64_t patch_bytes =
        static_cast<uint64_t>((words->size() + displaced_tail_words.size()) * sizeof(uint32_t)) +
        spill_bytes + scalar_spill_bytes + vgpr_msb_bytes +
        (dense_runtime_fast_gate ? access_island_word_count * sizeof(uint32_t) : 0u);
    const uint32_t available_padding =
        count_nop_padding(bytes, candidate.file_offset + candidate.size, arch);
    const uint64_t available_bytes =
        candidate.size + static_cast<uint64_t>(available_padding) * sizeof(uint32_t);
    std::optional<DbiPatchPlacement> placement;
    if (use_indirect_appended) {
      placement = DbiPatchPlacement{
          .kind = DbiPatchPlacementKind::AppendedCave,
          .anchor_offset = candidate.text_offset,
          .original_size = displaced_original_size,
          .body_offset = indirect_body_cursor,
          .body_size = patch_bytes,
          .return_branch_offset = 0,
          .return_target = candidate.text_offset + displaced_original_size,
      };
      const uint64_t return_words =
          use_dense_call && candidate_options.automatic_moi_record_replay_sgpr_spill
              ? kIndirectIslandWords
              : 6u;
      indirect_body_cursor += patch_bytes + return_words * sizeof(uint32_t);
    } else {
      DbiPatchPlacementRequest placement_request;
      placement_request.anchor_offset = candidate.text_offset;
      placement_request.original_size = candidate.size;
      placement_request.body_size = patch_bytes;
      placement_request.inline_capacity =
          spill || scalar_spill || private_layout ? 0u : available_bytes;
      std::string placement_error;
      placement = placement_planner.plan(placement_request, &placement_error);
      if (!placement) {
        result.warnings.emplace_back("ConSan MOI sampled probe skipped load/store site: " +
                                     placement_error);
        continue;
      }
    }

    PlannedSampledPatch planned;
    planned.candidate = candidate_ptr;
    planned.first_record_index = next_record_index;
    planned.access_range_count = range_count;
    planned.prior_first_record_index = prior_first_record_index;
    planned.prior_access_range_count = prior_access_range_count;
    planned.runtime_sample_index = current_sample_index;
    planned.placement = *placement;
    planned.entry_island_offset = entry_island_offset;
    planned.entry_island_word_count =
        uses_local_entry_island ? candidate_entry_island_words : access_island_word_count;
    planned.entry_island_at_anchor = entry_island_at_anchor;
    planned.dense_call_anchor = use_dense_call;
    if (use_dense_call)
      planned.dense_dispatcher_offset = dense_dispatchers.at(candidate_ptr);
    planned.displaced_tail_words = std::move(displaced_tail_words);
    planned.workgroup_descriptor_file_offset = *workgroup_descriptor;
    planned.resources = *resources;
    planned.spill = spill;
    planned.scalar_spill = scalar_spill;
    planned.gfx1250_vgpr_msb_mode = candidate.gfx1250_vgpr_msb_mode;
    planned.runtime_fast_gate = runtime_fast_gate;
    planned.dense_runtime_fast_gate = dense_runtime_fast_gate;
    if (private_layout)
      planned.private_epoch_offset = private_layout->epoch_offset;
    if (private_layout)
      planned.private_owner_offset = private_layout->owner_offset;
    planned.private_owner_shift = private_owner_shift;
    if (private_layout)
      planned.persistent_private_state_end = private_layout->ephemeral_base;
    planned.required_private_bytes = required_private_bytes;
    planned.probe_guest_instruction_offset = probe_guest_instruction_offset;
    planned.transient_exec_save_sgpr = candidate_options.moi_exec_save_sgpr;
    planned.spill_overlaps_guest_operands = spill_overlaps_guest_operands;
    planned.wrap_embedded_guest_vgpr_bank = wrap_embedded_guest_vgpr_bank;
    planned_patches.push_back(std::move(planned));
    if (use_indirect_appended && !use_dense_call)
      planned_anchor_end = candidate.text_offset + placement->original_size;
    next_record_index += required_slots;
    if (next_record_index == max_candidates)
      break;
  }

  for (auto &[dispatcher, host] : dense_entry_hosts) {
    (void)dispatcher;
    host.body_offset = indirect_body_cursor;
    indirect_body_cursor += (host.displaced_words.size() + kIndirectIslandWords) * sizeof(uint32_t);
  }

  if (planned_patches.empty()) {
    if (capacity_limited) {
      result.warnings.emplace_back(
          "ConSan MOI sampled probe report capacity cannot retain every range of a selected "
          "load/store site");
    } else if (options.moi_sample_stride != 1 || options.moi_sample_offset != 0) {
      result.warnings.emplace_back(
          "ConSan MOI sampled probe selected no candidate after applying the sample filter");
    }
    return;
  }

  if ((capacity_limited || next_record_index == max_candidates) &&
      sampled_filter_candidate_count > planned_patches.size()) {
    result.warnings.emplace_back(
        "ConSan MOI sampled probe limited sampled patches to " +
        std::to_string(planned_patches.size()) + " of " +
        std::to_string(sampled_filter_candidate_count) +
        " sample-filter-selected candidates by report capacity or max-patch budget");
  }

  MoiDescriptorVgprRequirements descriptor_requirements;
  MoiDescriptorSgprRequirements scalar_requirements;
  MoiDescriptorPrivateRequirements private_requirements;
  for (const PlannedSampledPatch &planned_patch : planned_patches) {
    note_descriptor_requirements(descriptor_requirements, planned_patch.resources);
    ConSanOptions candidate_options = options;
    if (!apply_moi_transient_sgpr_assignment(candidate_options,
                                             planned_patch.resources.owner_descriptor_file_offsets))
      continue;
    note_moi_sgpr_requirements(scalar_requirements, planned_patch.resources, candidate_options,
                               arch);
    if (planned_patch.required_private_bytes != 0u) {
      for (uint64_t descriptor_offset : planned_patch.resources.owner_descriptor_file_offsets) {
        auto [it, inserted] =
            private_requirements.emplace(descriptor_offset, planned_patch.required_private_bytes);
        if (!inserted)
          it->second = std::max(it->second, planned_patch.required_private_bytes);
      }
    }
  }

  const bool uses_appended_cave =
      std::ranges::any_of(planned_patches, [](const PlannedSampledPatch &patch) {
        return patch.placement.kind == DbiPatchPlacementKind::AppendedCave;
      });
  if (uses_appended_cave) {
    CodeObjectPatcher patcher(code_object);
    if (!apply_descriptor_requirements(patcher, code_object, bytes, result, descriptor_requirements,
                                       arch, result.errors)) {
      return;
    }
    if (!apply_spill_descriptor_requirements(patcher, code_object, bytes, result,
                                             private_requirements, result.errors)) {
      return;
    }
    if (!apply_sgpr_descriptor_requirements(patcher, result, code_object, scalar_requirements,
                                            result.errors))
      return;
    const std::span<const uint8_t> old_text = patcher.text_bytes();
    if (old_text.empty()) {
      result.errors.emplace_back("ConSan MOI sampled probe found no .text section");
      return;
    }
    std::vector<uint8_t> new_text(old_text.begin(), old_text.end());
    if (use_indirect_appended) {
      while (new_text.size() < indirect_body_begin)
        append_word_bytes(new_text, build_s_nop(0, arch));
    }
    std::vector<ConSanPatchInfo> patches;
    patches.reserve(planned_patches.size());
    std::map<uint64_t, std::vector<const PlannedSampledPatch *>> dense_groups;
    for (const PlannedSampledPatch &planned_patch : planned_patches) {
      if (planned_patch.dense_call_anchor)
        dense_groups[*planned_patch.dense_dispatcher_offset].push_back(&planned_patch);
    }
    for (const auto &[dispatcher_offset, group] : dense_groups) {
      if (group.empty() || !group.front()->transient_exec_save_sgpr ||
          !group.front()->entry_island_offset)
        continue;
      ConSanOptions group_options = options;
      if (!apply_moi_transient_sgpr_assignment(
              group_options, group.front()->resources.owner_descriptor_file_offsets))
        continue;
      const uint16_t base = *group_options.moi_exec_save_sgpr;
      const uint16_t pc_sgpr = group_options.automatic_moi_record_replay_sgpr_spill
                                   ? *group_options.moi_inline_indirect_pc_sgpr
                                   : base;
      const uint16_t key_sgpr = group_options.automatic_moi_record_replay_sgpr_spill
                                    ? *group_options.moi_record_replay_dispatch_key_sgpr
                                    : static_cast<uint16_t>(base + 5u);
      const uint16_t call_return_sgpr = group_options.automatic_moi_record_replay_sgpr_spill
                                            ? *group_options.moi_record_replay_call_return_sgpr
                                            : static_cast<uint16_t>(base + 6u);
      const bool collapsed_spill_router =
          group_options.automatic_moi_record_replay_sgpr_spill && call_return_sgpr == pc_sgpr;
      const uint32_t dense_entry_island_words =
          collapsed_spill_router ? kIndirectIslandWords + 1u : kIndirectIslandWords;
      const bool explicit_dense_key = arch != ROCJITSU_CODE_ARCH_GFX1250;
      const uint64_t island_offset = *group.front()->entry_island_offset;
      const MoiSpecialStateSgprs special_state = *moi_special_state_sgprs(group_options);

      std::vector<uint32_t> island_words;
      if (collapsed_spill_router) {
        const auto save_scc = instrumentation::build_s_cselect_b32(
            special_state.scc_save_sgpr, scalar_positive_inline_u32(1),
            scalar_positive_inline_u32(0), arch);
        if (!save_scc) {
          result.errors.emplace_back("ConSan MOI sampled probe could not save its dense entry SCC");
          return;
        }
        island_words.push_back(*save_scc);
        island_words.push_back(build_s_mov_b32(key_sgpr, call_return_sgpr, arch));
      }
      if (!append_moi_scc_preserving_indirect_jump(island_words, island_offset, dispatcher_offset,
                                                   pc_sgpr, special_state.scc_save_sgpr,
                                                   /*capture_scc=*/!collapsed_spill_router, arch) ||
          island_words.size() > dense_entry_island_words) {
        result.errors.emplace_back(
            "ConSan MOI sampled probe could not encode its dense entry island");
        return;
      }
      while (island_words.size() < dense_entry_island_words)
        island_words.push_back(build_s_nop(0, arch));
      const uint64_t island_bytes = island_words.size() * sizeof(uint32_t);
      if (island_offset > new_text.size() || island_bytes > new_text.size() - island_offset) {
        result.errors.emplace_back("ConSan MOI sampled probe dense entry island exceeds .text");
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
        const auto subtract_pc = instrumentation::build_s_sub_u32(
            key_sgpr, collapsed_spill_router ? key_sgpr : call_return_sgpr, pc_sgpr, arch);
        if (!subtract_pc) {
          result.errors.emplace_back(
              "ConSan MOI sampled probe could not encode its dense dispatch key");
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
              "ConSan MOI sampled probe could not encode its dense entry-host comparison");
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
                       matched_words, matched_offset, entry_host->second.body_offset, pc_sgpr,
                       special_state.scc_save_sgpr, /*capture_scc=*/false, arch)) {
          result.errors.emplace_back(
              "ConSan MOI sampled probe could not encode its dense entry-host target");
          return;
        }
        const auto skip =
            instrumentation::build_s_cbranch_scc0(static_cast<int16_t>(matched_words.size()), arch);
        if (!skip) {
          result.errors.emplace_back(
              "ConSan MOI sampled probe could not encode its dense entry-host branch");
          return;
        }
        dispatcher_words.push_back(*skip);
        dispatcher_words.insert(dispatcher_words.end(), matched_words.begin(), matched_words.end());
      }
      for (size_t group_index = 0; group_index < group.size(); ++group_index) {
        const PlannedSampledPatch *planned_patch = group[group_index];
        const ConSanMoiCandidate &candidate = *planned_patch->candidate;
        const auto compare = instrumentation::build_s_cmp_eq_u32(
            key_sgpr,
            explicit_dense_key ? scalar_positive_inline_u32(static_cast<uint16_t>(group_index + 1u))
                               : 255u,
            arch);
        if (!compare) {
          result.errors.emplace_back(
              "ConSan MOI sampled probe could not encode its dense dispatch comparison");
          return;
        }
        dispatcher_words.push_back(*compare);
        if (!explicit_dense_key) {
          dispatcher_words.push_back(
              static_cast<uint32_t>(candidate.text_offset + sizeof(uint32_t) - island_offset));
        }
        std::vector<uint32_t> matched_words;
        if (!explicit_dense_key && !collapsed_spill_router && candidate.size > sizeof(uint32_t)) {
          const uint32_t adjustment = candidate.size - sizeof(uint32_t);
          matched_words.push_back(
              build_s_add_u32(call_return_sgpr, call_return_sgpr, /*literal=*/255u, arch));
          matched_words.push_back(adjustment);
          matched_words.push_back(build_s_addc_u32(static_cast<uint16_t>(call_return_sgpr + 1u),
                                                   static_cast<uint16_t>(call_return_sgpr + 1u),
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
                       matched_words, matched_offset, planned_patch->placement.body_offset, pc_sgpr,
                       special_state.scc_save_sgpr, /*capture_scc=*/false, arch)) {
          result.errors.emplace_back(
              "ConSan MOI sampled probe could not encode a dense dispatch target");
          return;
        }
        if (matched_words.size() > static_cast<size_t>(std::numeric_limits<int16_t>::max())) {
          result.errors.emplace_back("ConSan MOI sampled probe dense dispatch stub is too large");
          return;
        }
        const auto skip =
            instrumentation::build_s_cbranch_scc0(static_cast<int16_t>(matched_words.size()), arch);
        if (!skip) {
          result.errors.emplace_back(
              "ConSan MOI sampled probe could not encode a dense dispatch branch");
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
            "ConSan MOI sampled probe dense dispatcher exceeds its reserved relay space");
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
      dispatcher_info.owner_descriptor_file_offsets =
          group.front()->resources.owner_descriptor_file_offsets;
      patches.push_back(std::move(dispatcher_info));

      if (entry_host != dense_entry_hosts.end()) {
        const std::optional<uint32_t> host_entry =
            explicit_dense_key
                ? std::optional<uint32_t>(
                      build_s_mov_b32(key_sgpr, scalar_positive_inline_u32(0), arch))
                : instrumentation::build_s_call_i64(call_return_sgpr, /*next instruction=*/0, arch);
        if (!host_entry || entry_host->second.host_offset > new_text.size() ||
            sizeof(uint32_t) > new_text.size() - entry_host->second.host_offset) {
          result.errors.emplace_back(
              "ConSan MOI sampled probe could not encode its dense entry host");
          return;
        }
        std::memcpy(new_text.data() + entry_host->second.host_offset, &*host_entry,
                    sizeof(uint32_t));
      }

      for (size_t group_index = 0; group_index < group.size(); ++group_index) {
        const PlannedSampledPatch *planned_patch = group[group_index];
        const ConSanMoiCandidate &candidate = *planned_patch->candidate;
        const uint64_t branch_offset =
            candidate.text_offset + (explicit_dense_key ? sizeof(uint32_t) : 0u);
        const auto call_delta = compute_sopp_branch_simm16(branch_offset, island_offset);
        const std::optional<uint32_t> call =
            !call_delta ? std::nullopt
                        : instrumentation::build_s_call_i64(call_return_sgpr, *call_delta, arch);
        if ((!explicit_dense_key && !call) ||
            candidate.size < (explicit_dense_key ? 2u : 1u) * sizeof(uint32_t) ||
            candidate.text_offset > new_text.size() ||
            candidate.size > new_text.size() - candidate.text_offset) {
          result.errors.emplace_back(
              "ConSan MOI sampled probe could not encode a dense call anchor");
          return;
        }
        std::vector<uint32_t> anchor_words(candidate.size / sizeof(uint32_t), build_s_nop(0, arch));
        if (explicit_dense_key) {
          if (!call_delta || group_index >= 64u) {
            result.errors.emplace_back(
                "ConSan MOI sampled probe could not encode an explicit dense branch key");
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
    for (const PlannedSampledPatch &planned_patch : planned_patches) {
      const ConSanMoiCandidate &candidate = *planned_patch.candidate;
      ConSanOptions candidate_options = options;
      if (!apply_moi_transient_sgpr_assignment(
              candidate_options, planned_patch.resources.owner_descriptor_file_offsets)) {
        result.errors.emplace_back(
            "ConSan MOI sampled probe lost its final transient scalar assignment");
        return;
      }
      candidate_options.scratch_vgpr = planned_patch.resources.base;
      candidate_options.moi_spill_overlaps_guest_operands =
          planned_patch.spill_overlaps_guest_operands;
      if (planned_patch.private_epoch_offset ||
          moi_uses_scalar_persistent_state(candidate_options)) {
        uint16_t state_vgpr =
            static_cast<uint16_t>(planned_patch.resources.base + planned_patch.resources.count);
        if (!planned_patch.private_epoch_offset || planned_patch.private_owner_offset)
          candidate_options.moi_owner_vgpr = --state_vgpr;
        candidate_options.moi_epoch_vgpr = --state_vgpr;
      }
      ConSanMoiCandidate lowered_candidate = candidate;
      lowered_candidate.kernel_descriptor_file_offset =
          planned_patch.workgroup_descriptor_file_offset;
      auto words = build_direct_sampled_watchpoint_words(
          bytes, lowered_candidate, candidate_options, arch, planned_patch.first_record_index,
          planned_patch.prior_first_record_index, planned_patch.prior_access_range_count,
          window_bank_count, layout.sampled_causal_windows_offset,
          layout.sampled_watchpoints_offset, layout.sampled_pending_acquires_offset,
          planned_patch.private_epoch_offset, planned_patch.private_owner_offset,
          planned_patch.private_owner_shift, result.errors);
      if (!words)
        return;
      const uint64_t spill_bytes =
          planned_patch.spill ? static_cast<uint64_t>((planned_patch.spill->save_words.size() +
                                                       planned_patch.spill->restore_words.size()) *
                                                      sizeof(uint32_t))
                              : 0;
      const uint64_t scalar_spill_bytes =
          planned_patch.scalar_spill
              ? static_cast<uint64_t>(planned_patch.scalar_spill->save_words.size() +
                                      planned_patch.scalar_spill->restore_words.size()) *
                    sizeof(uint32_t)
              : 0u;
      const uint64_t displaced_tail_bytes =
          planned_patch.displaced_tail_words.size() * sizeof(uint32_t);
      const uint64_t vgpr_msb_bytes =
          moi_appended_body_vgpr_msb_transition_word_count(
              arch, planned_patch.gfx1250_vgpr_msb_mode, planned_patch.spill.has_value(),
              planned_patch.wrap_embedded_guest_vgpr_bank) *
          sizeof(uint32_t);
      const uint64_t dense_gate_bytes =
          planned_patch.dense_runtime_fast_gate ? access_island_word_count * sizeof(uint32_t) : 0u;
      if (spill_bytes + scalar_spill_bytes + displaced_tail_bytes + dense_gate_bytes +
              vgpr_msb_bytes >
          planned_patch.placement.body_size) {
        result.errors.emplace_back("ConSan MOI sampled probe has an invalid relocated body size");
        return;
      }
      const uint64_t planned_probe_bytes = planned_patch.placement.body_size - spill_bytes -
                                           scalar_spill_bytes - displaced_tail_bytes -
                                           dense_gate_bytes - vgpr_msb_bytes;
      if (words->size() * sizeof(uint32_t) > planned_probe_bytes) {
        result.errors.emplace_back("ConSan MOI sampled probe final patch grew after planning");
        return;
      }
      words->resize(static_cast<size_t>(planned_probe_bytes / sizeof(uint32_t)),
                    build_s_nop(0, arch));

      ConSanPatchInfo info;
      info.kind = planned_patch.placement.kind == DbiPatchPlacementKind::AppendedCave
                      ? ConSanPatchKind::TrampolineMoiSampledWatchpointStore
                      : ConSanPatchKind::InlineMoiSampledWatchpointStore;
      info.anchor_offset = candidate.text_offset;
      info.sampled_first_slot = planned_patch.first_record_index;
      info.sampled_window_bank_count = window_bank_count;
      info.sampled_access_kind = candidate.kind;
      info.sampled_access_range_count = planned_patch.access_range_count;
      info.scratch_vgpr = planned_patch.resources.base;
      info.persistent_epoch_private_offset = planned_patch.private_epoch_offset;
      info.persistent_owner_private_offset = planned_patch.private_owner_offset;
      if (planned_patch.private_epoch_offset)
        info.persistent_private_state_end = planned_patch.persistent_private_state_end;
      info.required_private_segment_size = planned_patch.required_private_bytes;
      info.owner_descriptor_file_offsets = planned_patch.resources.owner_descriptor_file_offsets;
      if (planned_patch.spill) {
        info.spilled_vgpr_count = planned_patch.spill->vgpr_count;
        note_dynamic_stack_private_requirement(info, &*planned_patch.spill);
      }

      if (planned_patch.placement.kind == DbiPatchPlacementKind::AppendedCave) {
        const uint64_t cave_text_offset = planned_patch.placement.body_offset;
        if (new_text.size() != cave_text_offset) {
          result.errors.emplace_back(
              "ConSan MOI sampled probe emitted a stale appended-cave mapping");
          return;
        }
        const uint64_t entry_target =
            planned_patch.entry_island_offset.value_or(planned_patch.placement.body_offset);
        const auto ret =
            planned_patch.entry_island_offset
                ? std::optional<int16_t>()
                : compute_sopp_branch_simm16(planned_patch.placement.return_branch_offset,
                                             planned_patch.placement.return_target);
        if ((!planned_patch.dense_call_anchor &&
             !write_moi_appended_anchor(new_text, planned_patch, entry_target, arch,
                                        "sampled probe", result.errors)) ||
            (!planned_patch.entry_island_offset && !ret)) {
          result.errors.emplace_back(
              "ConSan MOI sampled probe appended cave branch is out of range");
          return;
        }

        if (planned_patch.entry_island_offset && !planned_patch.dense_call_anchor) {
          const MoiSpecialStateSgprs special_state = *moi_special_state_sgprs(candidate_options);
          std::vector<uint32_t> island_words;
          if (planned_patch.runtime_fast_gate) {
            std::vector<std::string> workgroup_errors;
            const auto workgroup_sources = moi_descriptor_workgroup_sources(
                bytes, planned_patch.workgroup_descriptor_file_offset, arch, workgroup_errors);
            if (!workgroup_sources) {
              result.errors.insert(result.errors.end(), workgroup_errors.begin(),
                                   workgroup_errors.end());
              return;
            }
            auto fast_gate = build_sampled_runtime_fast_gate_island_words(
                bytes, candidate, candidate_options, *workgroup_sources,
                planned_patch.runtime_sample_index, *planned_patch.entry_island_offset,
                cave_text_offset, planned_patch.placement.return_target,
                planned_patch.entry_island_word_count, arch);
            if (fast_gate)
              island_words = std::move(*fast_gate);
          } else if (append_moi_scc_preserving_indirect_jump(
                         island_words, *planned_patch.entry_island_offset, cave_text_offset,
                         moi_indirect_jump_sgprs(candidate_options)->pc_sgpr,
                         special_state.scc_save_sgpr,
                         /*capture_scc=*/true, arch)) {
            island_words.resize(planned_patch.entry_island_word_count, build_s_nop(0, arch));
          }
          if (island_words.size() != planned_patch.entry_island_word_count) {
            result.errors.emplace_back(
                "ConSan MOI sampled probe could not encode its entry island");
            return;
          }
          if (!write_moi_entry_island(new_text, planned_patch, island_words, patches,
                                      "sampled probe", result.errors))
            return;
        }

        std::vector<uint32_t> dense_gate_words;
        if (planned_patch.dense_runtime_fast_gate) {
          std::vector<std::string> workgroup_errors;
          const auto workgroup_sources = moi_descriptor_workgroup_sources(
              bytes, planned_patch.workgroup_descriptor_file_offset, arch, workgroup_errors);
          if (!workgroup_sources) {
            result.errors.insert(result.errors.end(), workgroup_errors.begin(),
                                 workgroup_errors.end());
            return;
          }
          auto gate = build_sampled_runtime_fast_gate_call_words(
              bytes, candidate, candidate_options, *workgroup_sources, cave_text_offset,
              planned_patch.placement.return_target, access_island_word_count, arch);
          if (!gate) {
            result.errors.emplace_back(
                "ConSan MOI sampled probe could not encode its dense runtime gate");
            return;
          }
          dense_gate_words = std::move(*gate);
        }
        uint32_t body_guest_instruction_offset = 0;
        MoiAppendedBodyOptions body_options(arch);
        body_options.scalar_save_words = planned_patch.scalar_spill
                                             ? planned_patch.scalar_spill->save_words
                                             : std::span<const uint32_t>{};
        body_options.scalar_restore_words = planned_patch.scalar_spill
                                                ? planned_patch.scalar_spill->restore_words
                                                : std::span<const uint32_t>{};
        body_options.gfx1250_vgpr_msb_mode = planned_patch.gfx1250_vgpr_msb_mode;
        body_options.entry_prefix_words = dense_gate_words;
        body_options.probe_guest_instruction_offset = planned_patch.probe_guest_instruction_offset;
        body_options.body_guest_instruction_offset = &body_guest_instruction_offset;
        body_options.wrap_embedded_guest_vgpr_bank = planned_patch.wrap_embedded_guest_vgpr_bank;
        auto cave_words = assemble_moi_appended_body(planned_patch, *words, "sampled probe",
                                                     result.errors, body_options);
        if (!cave_words)
          return;
        if (planned_patch.dense_call_anchor) {
          const MoiSpecialStateSgprs special_state = *moi_special_state_sgprs(candidate_options);
          const size_t return_begin = cave_words->size();
          if (!append_moi_scc_preserving_indirect_jump(
                  *cave_words, cave_text_offset, planned_patch.placement.return_target,
                  moi_indirect_jump_sgprs(candidate_options)->pc_sgpr, special_state.scc_save_sgpr,
                  /*capture_scc=*/candidate_options.automatic_moi_record_replay_sgpr_spill, arch)) {
            result.errors.emplace_back(
                "ConSan MOI sampled probe could not encode its dense long return");
            return;
          }
          if (candidate_options.automatic_moi_record_replay_sgpr_spill)
            while (cave_words->size() < return_begin + kIndirectIslandWords)
              cave_words->push_back(build_s_nop(0, arch));
        } else if (planned_patch.entry_island_offset) {
          const MoiSpecialStateSgprs special_state = *moi_special_state_sgprs(candidate_options);
          if (!append_moi_scc_preserving_indirect_jump(
                  *cave_words, cave_text_offset, planned_patch.placement.return_target,
                  moi_indirect_jump_sgprs(candidate_options)->pc_sgpr, special_state.scc_save_sgpr,
                  /*capture_scc=*/false, arch)) {
            result.errors.emplace_back("ConSan MOI sampled probe could not encode its long return");
            return;
          }
        } else {
          cave_words->push_back(build_s_branch(*ret, arch));
        }
        append_words_bytes(new_text, *cave_words);
        info.trampoline_offset = cave_text_offset;
        info.original_size = planned_patch.placement.original_size;
        info.trampoline_size = static_cast<uint32_t>(cave_words->size() * sizeof(uint32_t));
        info.relocated_guest_instruction_offset = cave_text_offset + body_guest_instruction_offset;
      } else {
        const uint64_t patch_bytes = static_cast<uint64_t>(words->size() * sizeof(uint32_t));
        if (patch_bytes != planned_patch.placement.body_size) {
          result.errors.emplace_back("ConSan MOI sampled probe final patch size changed");
          return;
        }
        if (candidate.text_offset > new_text.size() ||
            patch_bytes > new_text.size() - candidate.text_offset) {
          result.errors.emplace_back("ConSan MOI sampled probe exceeds .text");
          return;
        }
        std::memcpy(new_text.data() + candidate.text_offset, words->data(),
                    static_cast<size_t>(patch_bytes));
        info.trampoline_offset = candidate.text_offset + candidate.size;
        info.original_size = static_cast<uint32_t>(patch_bytes);
        info.trampoline_size = 0;
        info.relocated_guest_instruction_offset =
            candidate.text_offset + planned_patch.probe_guest_instruction_offset;
      }
      patches.push_back(info);
    }

    for (const auto &[dispatcher, host] : dense_entry_hosts) {
      if (new_text.size() != host.body_offset) {
        result.errors.emplace_back(
            "ConSan MOI sampled probe emitted a stale dense entry-host mapping");
        return;
      }
      if (!dense_groups.contains(dispatcher)) {
        // Dense hosts are discovered before per-site resource filtering so
        // their appended offsets remain stable.  A whole owner group can
        // subsequently lose every usable site (for example when persistent
        // scalar state overlaps a full-pressure kernel).  Preserve that
        // reservation as inert padding instead of emitting an orphan relay
        // body that no anchor can reach and no patch record can account for.
        const size_t reserved_words = host.displaced_words.size() + kIndirectIslandWords;
        for (size_t word = 0; word < reserved_words; ++word)
          append_word_bytes(new_text, build_s_nop(0, arch));
        continue;
      }
      const PlannedSampledPatch *representative = dense_groups.at(dispatcher).front();
      ConSanOptions group_options = options;
      if (!apply_moi_transient_sgpr_assignment(
              group_options, representative->resources.owner_descriptor_file_offsets)) {
        result.errors.emplace_back(
            "ConSan MOI sampled probe lost its dense-host transient scalar assignment");
        return;
      }
      std::vector<uint32_t> host_words = host.displaced_words;
      if (!append_moi_scc_preserving_indirect_jump(
              host_words, host.body_offset,
              host.host_offset + host.displaced_words.size() * sizeof(uint32_t),
              moi_indirect_jump_sgprs(group_options)->pc_sgpr,
              moi_special_state_sgprs(group_options)->scc_save_sgpr,
              /*capture_scc=*/true, arch)) {
        result.errors.emplace_back(
            "ConSan MOI sampled probe could not encode its dense entry-host return");
        return;
      }
      append_words_bytes(new_text, host_words);
    }

    if (!replace_consan_text(patcher, new_text, options, "MOI sampled probe", result)) {
      return;
    }
    result.elf_bytes = std::move(patcher).emit();
    result.patches.insert(result.patches.end(), patches.begin(), patches.end());
    result.modified = true;
    return;
  }

  if (!apply_inline_moi_access_patches(
          bytes, planned_patches, descriptor_requirements, scalar_requirements, "sampled probe",
          arch,
          [&](const PlannedSampledPatch &planned_patch) {
            ConSanOptions candidate_options = options;
            if (!apply_moi_transient_sgpr_assignment(
                    candidate_options, planned_patch.resources.owner_descriptor_file_offsets)) {
              result.errors.emplace_back(
                  "ConSan MOI sampled probe lost its inline transient scalar assignment");
              return std::optional<std::vector<uint32_t>>{};
            }
            candidate_options.scratch_vgpr = planned_patch.resources.base;
            candidate_options.moi_spill_overlaps_guest_operands =
                planned_patch.spill_overlaps_guest_operands;
            if (planned_patch.private_epoch_offset ||
                moi_uses_scalar_persistent_state(candidate_options)) {
              uint16_t state_vgpr = static_cast<uint16_t>(planned_patch.resources.base +
                                                          planned_patch.resources.count);
              if (!planned_patch.private_epoch_offset || planned_patch.private_owner_offset)
                candidate_options.moi_owner_vgpr = --state_vgpr;
              candidate_options.moi_epoch_vgpr = --state_vgpr;
            }
            ConSanMoiCandidate lowered_candidate = *planned_patch.candidate;
            lowered_candidate.kernel_descriptor_file_offset =
                planned_patch.workgroup_descriptor_file_offset;
            auto words = build_direct_sampled_watchpoint_words(
                bytes, lowered_candidate, candidate_options, arch, planned_patch.first_record_index,
                planned_patch.prior_first_record_index, planned_patch.prior_access_range_count,
                window_bank_count, layout.sampled_causal_windows_offset,
                layout.sampled_watchpoints_offset, layout.sampled_pending_acquires_offset,
                planned_patch.private_epoch_offset, planned_patch.private_owner_offset,
                planned_patch.private_owner_shift, result.errors);
            if (words) {
              const size_t planned_words =
                  static_cast<size_t>(planned_patch.placement.body_size / sizeof(uint32_t));
              if (words->size() > planned_words) {
                result.errors.emplace_back(
                    "ConSan MOI sampled probe final patch grew after planning");
                return std::optional<std::vector<uint32_t>>{};
              }
              words->resize(planned_words, build_s_nop(0, arch));
            }
            return words;
          },
          [&](const PlannedSampledPatch &planned_patch) {
            const ConSanMoiCandidate &candidate = *planned_patch.candidate;
            ConSanPatchInfo info;
            info.kind = ConSanPatchKind::InlineMoiSampledWatchpointStore;
            info.anchor_offset = candidate.text_offset;
            info.trampoline_offset = candidate.text_offset + candidate.size;
            info.original_size = static_cast<uint32_t>(planned_patch.placement.body_size);
            info.sampled_first_slot = planned_patch.first_record_index;
            info.sampled_window_bank_count = window_bank_count;
            info.sampled_access_kind = candidate.kind;
            info.sampled_access_range_count = planned_patch.access_range_count;
            info.scratch_vgpr = planned_patch.resources.base;
            info.persistent_epoch_private_offset = planned_patch.private_epoch_offset;
            info.persistent_owner_private_offset = planned_patch.private_owner_offset;
            if (planned_patch.private_epoch_offset)
              info.persistent_private_state_end = planned_patch.persistent_private_state_end;
            info.required_private_segment_size = planned_patch.required_private_bytes;
            info.owner_descriptor_file_offsets =
                planned_patch.resources.owner_descriptor_file_offsets;
            info.relocated_guest_instruction_offset =
                candidate.text_offset + planned_patch.probe_guest_instruction_offset;
            note_dynamic_stack_private_requirement(
                info, planned_patch.spill ? &*planned_patch.spill : nullptr);
            return info;
          },
          [](std::vector<uint8_t> &, ConSanResult &) { return true; }, result))
    return;
}
