[[nodiscard]] std::optional<ConSanMoiAtomicEventKind>
atomic_event_kind_for_site(const ConSanResult &result, std::string_view container_name,
                           bool in_kernel, const ConSanAtomicSite &site);

void try_apply_first_light_access_record_patch(std::span<const uint8_t> bytes,
                                               const ConSanOptions &options, rj_code_arch_t arch,
                                               const MoiResourcePlanningState &resource_state,
                                               ConSanResult &result) {
  if (!options.moi_report_buffer_address)
    return;
  if (!instrumentation::is_admitted_arch(arch)) {
    result.warnings.emplace_back("ConSan MOI first-light probe does not support this architecture");
    return;
  }
  if (options.moi_report_buffer_size < consan_moi_report_buffer_min_bytes(1, 0, 0, 0)) {
    result.warnings.emplace_back(
        "ConSan MOI first-light probe requires room for the report header and one access record");
    return;
  }

  const ConSanMoiReportBufferLayout layout = resolve_record_replay_report_layout(
      options, options.moi_track_barriers, options.moi_track_atomics, options.moi_track_atomics);
  if (options.max_patches == 0 || layout.access_record_capacity == 0)
    return;
  if (options.moi_dynamic_access_records && !options.moi_exec_save_sgpr) {
    result.warnings.emplace_back(
        "ConSan MOI dynamic access-record probe requires RJ_CONSAN_MOI_EXEC_SAVE_SGPR");
    return;
  }
  if (options.moi_dynamic_access_records &&
      (*options.moi_exec_save_sgpr > 100u || *options.moi_exec_save_sgpr % 2u != 0u)) {
    result.warnings.emplace_back(
        "ConSan MOI dynamic access-record probe requires an even RJ_CONSAN_MOI_EXEC_SAVE_SGPR in "
        "0..100");
    return;
  }
  std::vector<const ConSanMoiCandidate *> candidates =
      find_first_light_access_record_candidates(result, bytes, arch);
  apply_test_kernel_filter(candidates, options);
  std::ranges::sort(candidates, [](const ConSanMoiCandidate *lhs, const ConSanMoiCandidate *rhs) {
    return lhs->text_offset < rhs->text_offset;
  });
  // Ordering instrumentation composes around the relocated guest access in a
  // second text-growth pass. Keep those access bodies at the tail so their
  // one-word branch anchors remain in direct reach of the appended ordering
  // caves even for very large code objects.
  std::unordered_set<uint64_t> ordering_atomic_offsets;
  if (options.moi_track_atomics) {
    const auto append_ordering_offsets = [&](const auto &container, bool in_kernel) {
      for (const ConSanAtomicSite &site : container.atomic_sites) {
        if (atomic_event_kind_for_site(result, container.name, in_kernel, site))
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
    result.warnings.emplace_back("ConSan MOI first-light probe found no native LDS or likely group "
                                 "flat load/store candidate");
    return;
  }
  const size_t selected_candidate_limit =
      options.max_patches_is_expert_limit ? options.max_patches : candidates.size();

  struct PlannedAccessRecordPatch {
    const ConSanMoiCandidate *candidate = nullptr;
    DbiPatchPlacement placement;
    std::optional<uint64_t> entry_island_offset;
    bool entry_island_at_anchor = false;
    bool dense_call_anchor = false;
    std::optional<uint64_t> dense_dispatcher_offset;
    std::vector<uint32_t> displaced_tail_words;
    uint32_t record_index = 0;
    uint32_t record_count = 0;
    ResolvedMoiScratchPlan resources;
    std::optional<VgprSpillSequence> spill;
    std::optional<InlineSgprSpillSequence> scalar_spill;
    std::optional<uint16_t> gfx1250_vgpr_msb_mode;
    std::optional<uint32_t> private_epoch_offset;
    uint32_t persistent_private_state_end = 0;
    uint32_t required_private_bytes = 0;
    uint32_t probe_guest_instruction_offset = 0;
    bool spill_overlaps_guest_operands = false;
    bool wrap_embedded_guest_vgpr_bank = false;
  };
  std::vector<PlannedAccessRecordPatch> planned_patches;
  uint32_t planned_record_count = 0;
  planned_patches.reserve(candidates.size());
  AmdGpuCodeObject code_object(bytes.data(), bytes.size());
  const uint64_t original_text_size =
      code_object.text_sections().size() == 1 ? code_object.text_sections().front()->size() : 0;
  DbiPatchPlacementPlanner placement_planner(arch, original_text_size);
  constexpr uint64_t kIndirectIslandWords = 7u;
  const bool has_spill_backed_scalar_assignment = std::ranges::any_of(
      options.moi_transient_sgpr_assignments, &ConSanMoiTransientSgprAssignment::spill_backed);
  // Spill-backed groups need room for the extra scalar-save arm, while truly
  // large generated objects need a conservative envelope to keep every
  // routing target reachable. Applying that spacing to every ordinary RDNA4
  // object needlessly displaces access bodies from their compact entry and
  // return layout. Retain the compact envelope for moderate objects.
  constexpr size_t kLargeRdna4RelayCandidateCount = 1024u;
  const bool needs_large_rdna4_relay =
      candidates.size() >= kLargeRdna4RelayCandidateCount ||
      (has_spill_backed_scalar_assignment && candidates.size() <= 2u);
  const uint64_t access_relay_words =
      arch == ROCJITSU_CODE_ARCH_GFX1250
          ? 24u
          : (needs_large_rdna4_relay ? 20u : 10u) + (has_spill_backed_scalar_assignment ? 1u : 0u);
  const bool has_unreachable_function_candidate =
      std::ranges::any_of(candidates, [&](const ConSanMoiCandidate *candidate) {
        return !candidate->in_kernel &&
               !compute_sopp_branch_simm16(candidate->text_offset, original_text_size);
      });
  const bool use_indirect_appended =
      options.moi_exec_save_sgpr.has_value() &&
      (candidates.size() > 8u || has_unreachable_function_candidate ||
       has_spill_backed_scalar_assignment);
  uint64_t reserved_barrier_island_count = 0u;
  if (use_indirect_appended && options.moi_track_barriers) {
    // The later persistent-epoch pass excludes split-barrier signals and
    // runtime kernels. Reserve the same candidate set here; counting the full
    // inventory leaves a large unused prefix before every access relay.
    const auto completion_count = [](const auto &container) {
      return std::ranges::count_if(container.barrier_sites, [](const ConSanBarrierSite &site) {
        return site.operation != ConSanBarrierSite::Operation::Signal;
      });
    };
    for (const ConSanKernelInfo &kernel : result.kernels) {
      if (!kernel.name.starts_with("__amd_rocclr_"))
        reserved_barrier_island_count += completion_count(kernel);
    }
    for (const ConSanFunctionInfo &function : result.functions)
      reserved_barrier_island_count += completion_count(function);
    reserved_barrier_island_count =
        std::min<uint64_t>(reserved_barrier_island_count, options.max_patches);
  }
  const uint64_t indirect_body_begin =
      original_text_size + reserved_barrier_island_count * kIndirectIslandWords * sizeof(uint32_t) +
      candidates.size() * access_relay_words * sizeof(uint32_t);
  const uint64_t access_island_begin =
      original_text_size + reserved_barrier_island_count * kIndirectIslandWords * sizeof(uint32_t);
  uint64_t indirect_body_cursor = indirect_body_begin;
  std::optional<MoiLocalNopIslandAllocator> local_entry_islands;
  std::span<const uint8_t> original_text;
  if (use_indirect_appended && code_object.text_sections().size() == 1) {
    const Section *text = code_object.text_sections().front();
    const auto text_bytes =
        std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(text->data()), text->size());
    original_text = text_bytes;
    local_entry_islands.emplace(text_bytes, result, arch, kIndirectIslandWords);
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
  // The retained planning state already owns the pristine CFG and liveness
  // facts needed by dense relay selection. Re-decoding the whole code object
  // here (and a third time for component-local scalar assignments) is both
  // redundant and dominant on multi-thousand-kernel generated libraries.
  const auto &dense_host_blocks = resource_state.blocks;
  if ((is_rdna4_family_arch(arch) || arch == ROCJITSU_CODE_ARCH_CDNA3 ||
       arch == ROCJITSU_CODE_ARCH_CDNA4) &&
      use_indirect_appended && local_entry_islands) {
    const auto merge_ranges = [](std::vector<std::pair<uint64_t, uint64_t>> &ranges) {
      std::ranges::sort(ranges);
      size_t merged_range_count = 0u;
      for (const auto &range : ranges) {
        if (merged_range_count != 0u && ranges[merged_range_count - 1u].second >= range.first) {
          ranges[merged_range_count - 1u].second =
              std::max(ranges[merged_range_count - 1u].second, range.second);
        } else {
          ranges[merged_range_count++] = range;
        }
      }
      ranges.resize(merged_range_count);
    };
    const auto overlaps_ranges = [](const std::vector<std::pair<uint64_t, uint64_t>> &ranges,
                                    uint64_t begin, uint64_t end) {
      const auto after =
          std::ranges::lower_bound(ranges, end, {}, [](const auto &range) { return range.first; });
      return after != ranges.begin() && std::prev(after)->second > begin;
    };
    std::vector<std::pair<uint64_t, uint64_t>> global_access_ranges;
    global_access_ranges.reserve(candidates.size());
    for (const ConSanMoiCandidate *candidate : candidates) {
      if (candidate->size != 0u)
        global_access_ranges.emplace_back(candidate->text_offset,
                                          candidate->text_offset + candidate->size);
    }
    merge_ranges(global_access_ranges);

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
      std::vector<const MoiKernelResourceContext *> group_owner_contexts;
      if (group_resource != nullptr) {
        group_owner_contexts.reserve(group_resource->owner_descriptor_file_offsets.size());
        for (uint64_t descriptor_offset : group_resource->owner_descriptor_file_offsets) {
          const auto context = resource_state.context_by_descriptor.find(descriptor_offset);
          if (context != resource_state.context_by_descriptor.end())
            group_owner_contexts.push_back(context->second);
        }
      }
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
      // gfx1250 also uses the shared call relay as a code-size optimization for
      // nearby sites. On RDNA4, keep the simpler per-site relay whenever it is
      // reachable and use dense routing only to recover otherwise-stranded
      // sites.
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
      if (!island) {
        const auto kernel =
            key.first ? std::ranges::find(result.kernels, key.second, &ConSanKernelInfo::name)
                      : result.kernels.end();
        const auto function =
            !key.first ? std::ranges::find(result.functions, key.second, &ConSanFunctionInfo::name)
                       : result.functions.end();
        const bool has_owner_range =
            (kernel != result.kernels.end() && kernel->has_text_range && kernel->code_size != 0u) ||
            (function != result.functions.end() && function->code_size != 0u);
        if (has_owner_range) {
          const uint64_t host_words_count = group_entry_island_words + 1u;
          const uint64_t host_bytes = host_words_count * sizeof(uint32_t);
          const uint64_t owner_begin =
              key.first ? kernel->entry_text_offset : function->entry_text_offset;
          const uint64_t owner_end =
              owner_begin + (key.first ? kernel->code_size : function->code_size);
          std::vector<std::pair<uint64_t, uint64_t>> owner_sync_ranges;
          const size_t owner_barrier_count =
              key.first ? kernel->barrier_sites.size() : function->barrier_sites.size();
          const size_t owner_fence_count =
              key.first ? kernel->fence_sites.size() : function->fence_sites.size();
          const size_t owner_atomic_count =
              key.first ? kernel->atomic_sites.size() : function->atomic_sites.size();
          owner_sync_ranges.reserve(owner_barrier_count + owner_fence_count + owner_atomic_count);
          const auto append_occupied_range = [&](uint64_t begin, uint64_t size) {
            if (size != 0u)
              owner_sync_ranges.emplace_back(begin, begin + size);
          };
          const auto append_owner_sites = [&](const auto &owner) {
            for (const ConSanBarrierSite &site : owner.barrier_sites)
              append_occupied_range(site.text_offset, site.size);
            for (const ConSanFenceSite &site : owner.fence_sites)
              append_occupied_range(site.text_offset, site.size);
            for (const ConSanAtomicSite &site : owner.atomic_sites)
              append_occupied_range(site.text_offset, site.size);
          };
          if (key.first)
            append_owner_sites(*kernel);
          else
            append_owner_sites(*function);
          merge_ranges(owner_sync_ranges);
          const auto overlaps_instrumented_site = [&](uint64_t begin, uint64_t end) {
            return overlaps_ranges(global_access_ranges, begin, end) ||
                   overlaps_ranges(owner_sync_ranges, begin, end);
          };

          std::optional<DenseEntryHost> best_host;
          uint64_t best_max_distance = std::numeric_limits<uint64_t>::max();
          auto block =
              std::lower_bound(dense_host_blocks.begin(), dense_host_blocks.end(), owner_begin,
                               [](const std::unique_ptr<BasicBlock> &candidate, uint64_t entry) {
                                 return candidate->end_offset() <= entry;
                               });
          for (; block != dense_host_blocks.end() && (*block)->start_offset() < owner_end;
               ++block) {
            if ((*block)->start_offset() < owner_begin || (*block)->end_offset() > owner_end ||
                (*block)->size() < host_bytes)
              continue;
            for (const Instruction &instruction : (*block)->instructions()) {
              const uint64_t host_offset = instruction.src_loc();
              if (host_offset + host_bytes > (*block)->end_offset() ||
                  overlaps_instrumented_site(host_offset, host_offset + host_bytes) ||
                  group_resource == nullptr ||
                  !moi_relocatable_host_preserves_transient_sgprs(
                      resource_state, group_options, *group_resource, group_owner_contexts, **block,
                      instruction, host_bytes))
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
                    decode_errors);
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
  MoiSpillManagers spill_managers;
  std::unordered_map<uint64_t, std::optional<MoiPrivateEpochLayout>> private_layout_by_descriptor;
  uint64_t planned_anchor_end = 0;
  for (const ConSanMoiCandidate *candidate_ptr : candidates) {
    const ConSanMoiCandidate &candidate = *candidate_ptr;
    const bool use_dense_call = dense_entry_islands.contains(candidate_ptr);
    if (use_indirect_appended && !use_dense_call && candidate.text_offset < planned_anchor_end) {
      result.warnings.emplace_back(
          "ConSan MOI first-light probe skipped load/store site inside a relocated prefix");
      continue;
    }
    const uint16_t access_scratch_count = moi_access_scratch_vgpr_count(options, candidate, arch);
    const auto resources = resolve_moi_scratch(result, candidate, options, access_scratch_count);
    if (!resources) {
      const ConSanCandidateResourcePlan *plan = resource_plan_for_candidate(result, candidate);
      if (plan != nullptr && plan->reason == ConSanRegisterPlanReason::ForbiddenOverlap &&
          options.scratch_vgpr) {
        std::vector<std::string> overlap_errors;
        const bool rejected = reject_candidate_scratch_range_overlap(
            candidate, *options.scratch_vgpr, access_scratch_count, overlap_errors);
        if (!rejected)
          overlap_errors.emplace_back("ConSan MOI probe scratch VGPRs overlap a forbidden range");
        result.warnings.insert(result.warnings.end(), overlap_errors.begin(), overlap_errors.end());
      }
      continue;
    }
    std::optional<MoiPrivateEpochLayout> private_layout;
    if (options.automatic_moi_private_epoch) {
      if (resources->owner_descriptor_file_offsets.size() == 1u) {
        const uint64_t descriptor = resources->owner_descriptor_file_offsets.front();
        auto [cached, inserted] =
            private_layout_by_descriptor.try_emplace(descriptor, std::nullopt);
        if (inserted) {
          cached->second =
              build_moi_private_epoch_layout(result, *resources, arch, result.warnings);
        }
        private_layout = cached->second;
      } else {
        private_layout = build_moi_private_epoch_layout(result, *resources, arch, result.warnings);
      }
      if (!private_layout) {
        result.warnings.emplace_back(
            "ConSan MOI first-light probe could not build private epoch state");
        continue;
      }
    }
    std::optional<VgprSpillSequence> spill;
    if (resources->source == ConSanRegisterAllocationSource::SpillRequired) {
      spill = build_moi_spill_sequence(
          result, *resources, options, spill_managers, arch, result.warnings,
          private_layout ? std::optional<uint32_t>(private_layout->ephemeral_base) : std::nullopt);
      if (!spill) {
        result.warnings.emplace_back(
            "ConSan MOI first-light probe could not build its VGPR spill sequence");
        continue;
      }
    }
    uint32_t required_private_bytes = private_layout ? private_layout->ephemeral_base : 0u;
    if (spill)
      required_private_bytes = std::max(required_private_bytes, spill->total_private_bytes);
    ConSanOptions candidate_options = options;
    candidate_options.scratch_vgpr = resources->base;
    if (!apply_moi_transient_sgpr_assignment(candidate_options,
                                             resources->owner_descriptor_file_offsets)) {
      result.warnings.emplace_back(
          "ConSan MOI first-light probe found no common transient SGPR assignment");
      continue;
    }
    bool spill_overlaps_guest_operands = false;
    if (spill && (arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4) &&
        !options.moi_dynamic_access_records) {
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
          *resources, candidate_options, spill_managers, arch, result.warnings,
          private_layout ? std::optional<uint32_t>(private_layout->ephemeral_base) : std::nullopt,
          /*spill_count=*/4u);
      if (!scalar_spill)
        continue;
      required_private_bytes = std::max(required_private_bytes, scalar_spill->total_private_bytes);
    }
    const std::optional<std::vector<ConSanMoiAccessRange>> access_ranges =
        candidate_access_ranges(bytes, candidate, arch);
    if (!access_ranges || access_ranges->empty()) {
      result.warnings.emplace_back(
          "ConSan MOI first-light probe could not recover an admitted access range");
      continue;
    }
    const uint32_t candidate_record_count = static_cast<uint32_t>(access_ranges->size());
    if (!options.moi_dynamic_access_records &&
        candidate_record_count > layout.access_record_capacity - planned_record_count) {
      result.warnings.emplace_back(
          "ConSan MOI first-light probe skipped a supported load/store site without enough "
          "report-buffer access-record slots");
      continue;
    }
    std::vector<std::string> candidate_errors;
    uint32_t probe_guest_instruction_offset = 0;
    auto words = build_first_light_access_record_words(
        bytes, candidate, candidate_options, arch, 0, 0, layout.access_record_capacity,
        private_layout ? std::optional<uint32_t>(private_layout->epoch_offset) : std::nullopt,
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
    const bool wrap_embedded_guest_vgpr_bank =
        use_indirect_appended && options.moi_dynamic_access_records &&
        arch == ROCJITSU_CODE_ARCH_GFX1250 && candidate.gfx1250_vgpr_msb_mode.value_or(0u) != 0u;
    const uint64_t vgpr_msb_bytes = moi_appended_body_vgpr_msb_transition_word_count(
                                        arch, candidate.gfx1250_vgpr_msb_mode, spill.has_value(),
                                        wrap_embedded_guest_vgpr_bank) *
                                    sizeof(uint32_t);
    std::vector<uint32_t> displaced_tail_words;
    uint32_t displaced_original_size = candidate.size;
    bool entry_island_at_anchor = false;
    const uint32_t available_padding =
        count_nop_padding(bytes, candidate.file_offset + candidate.size, arch);
    const uint64_t available_bytes =
        candidate.size + static_cast<uint64_t>(available_padding) * sizeof(uint32_t);
    std::optional<DbiPatchPlacement> placement;
    std::optional<uint64_t> entry_island_offset;
    if (use_indirect_appended) {
      if (use_dense_call) {
        entry_island_offset = dense_entry_islands.at(candidate_ptr);
      } else {
        const size_t candidate_index = candidate_indices.at(candidate_ptr);
        entry_island_offset =
            access_island_begin + candidate_index * access_relay_words * sizeof(uint32_t);
      }
      if (!use_dense_call &&
          (candidate.size < sizeof(uint32_t) ||
           !compute_sopp_branch_simm16(candidate.text_offset, *entry_island_offset))) {
        entry_island_offset =
            local_entry_islands ? local_entry_islands->claim_reachable(candidate.text_offset,
                                                                       candidate_entry_island_words)
                                : std::nullopt;
      }
      if (!use_dense_call && !entry_island_offset && !original_text.empty()) {
        std::vector<uint32_t> displaced_words;
        while (displaced_words.size() < candidate_entry_island_words) {
          auto decoded = decode_relocatable_entry_instruction(
              original_text, candidate.text_offset + displaced_words.size() * sizeof(uint32_t),
              arch, candidate_errors);
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
            entry_island_at_anchor = true;
          }
        }
      }
      if (candidate.size < sizeof(uint32_t) || !entry_island_offset) {
        result.warnings.emplace_back(
            "ConSan MOI first-light probe skipped load/store site: entry island is unreachable");
        continue;
      }
      const uint64_t patch_bytes =
          static_cast<uint64_t>((words->size() + displaced_tail_words.size()) * sizeof(uint32_t)) +
          spill_bytes + scalar_spill_bytes + vgpr_msb_bytes;
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
          use_dense_call ? (arch != ROCJITSU_CODE_ARCH_GFX1250 ||
                                    candidate_options.automatic_moi_record_replay_sgpr_spill
                                ? 7u
                                : 1u)
                         : 6u;
      indirect_body_cursor += patch_bytes + return_words * sizeof(uint32_t);
    } else {
      const uint64_t patch_bytes = static_cast<uint64_t>(words->size() * sizeof(uint32_t)) +
                                   spill_bytes + scalar_spill_bytes + vgpr_msb_bytes;
      DbiPatchPlacementRequest placement_request;
      placement_request.anchor_offset = candidate.text_offset;
      placement_request.original_size = candidate.size;
      placement_request.body_size = patch_bytes;
      placement_request.inline_capacity = spill || private_layout ? 0u : available_bytes;
      std::string placement_error;
      placement = placement_planner.plan(placement_request, &placement_error);
      if (!placement) {
        result.warnings.emplace_back("ConSan MOI first-light probe skipped load/store site: " +
                                     placement_error);
        continue;
      }
    }

    PlannedAccessRecordPatch planned;
    planned.candidate = candidate_ptr;
    planned.placement = *placement;
    planned.entry_island_offset = entry_island_offset;
    planned.entry_island_at_anchor = entry_island_at_anchor;
    planned.dense_call_anchor = use_dense_call;
    if (use_dense_call)
      planned.dense_dispatcher_offset = dense_dispatchers.at(candidate_ptr);
    planned.displaced_tail_words = std::move(displaced_tail_words);
    planned.record_index = planned_record_count;
    planned.record_count = candidate_record_count;
    planned.probe_guest_instruction_offset = probe_guest_instruction_offset;
    planned.resources = *resources;
    planned.spill = spill;
    planned.scalar_spill = scalar_spill;
    planned.gfx1250_vgpr_msb_mode = candidate.gfx1250_vgpr_msb_mode;
    if (private_layout) {
      planned.private_epoch_offset = private_layout->epoch_offset;
      planned.persistent_private_state_end = private_layout->ephemeral_base;
    }
    planned.required_private_bytes = required_private_bytes;
    planned.spill_overlaps_guest_operands = spill_overlaps_guest_operands;
    planned.wrap_embedded_guest_vgpr_bank = wrap_embedded_guest_vgpr_bank;
    planned_patches.push_back(std::move(planned));
    if (use_indirect_appended && !use_dense_call)
      planned_anchor_end = candidate.text_offset + placement->original_size;
    planned_record_count += candidate_record_count;
    if (planned_patches.size() == selected_candidate_limit)
      break;
  }

  for (auto &[dispatcher, host] : dense_entry_hosts) {
    (void)dispatcher;
    host.body_offset = indirect_body_cursor;
    indirect_body_cursor += (host.displaced_words.size() + kIndirectIslandWords) * sizeof(uint32_t);
  }

  if (planned_patches.empty())
    return;

  MoiDescriptorVgprRequirements descriptor_requirements;
  MoiDescriptorSgprRequirements scalar_requirements;
  MoiDescriptorPrivateRequirements private_requirements;
  for (const PlannedAccessRecordPatch &planned_patch : planned_patches) {
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
      std::ranges::any_of(planned_patches, [](const PlannedAccessRecordPatch &patch) {
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
      result.errors.emplace_back("ConSan MOI first-light probe found no .text section");
      return;
    }
    std::vector<uint8_t> new_text(old_text.begin(), old_text.end());
    if (use_indirect_appended) {
      while (new_text.size() < indirect_body_begin)
        append_word_bytes(new_text, build_s_nop(0, arch));
    }
    std::vector<ConSanPatchInfo> patches;
    patches.reserve(planned_patches.size());
    std::map<uint64_t, std::vector<const PlannedAccessRecordPatch *>> dense_groups;
    for (const PlannedAccessRecordPatch &planned_patch : planned_patches) {
      if (planned_patch.dense_call_anchor)
        dense_groups[*planned_patch.dense_dispatcher_offset].push_back(&planned_patch);
    }
    for (const auto &[dispatcher_offset, group] : dense_groups) {
      if (group.empty() || !group.front()->entry_island_offset)
        continue;
      ConSanOptions group_options = options;
      if (!apply_moi_transient_sgpr_assignment(
              group_options, group.front()->resources.owner_descriptor_file_offsets) ||
          !group_options.moi_exec_save_sgpr)
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
      const bool explicit_dense_key = arch != ROCJITSU_CODE_ARCH_GFX1250;
      const bool collapsed_spill_router = !explicit_dense_key &&
                                          group_options.automatic_moi_record_replay_sgpr_spill &&
                                          call_return_sgpr == pc_sgpr;
      const uint32_t dense_entry_island_words =
          collapsed_spill_router ? kIndirectIslandWords + 1u : kIndirectIslandWords;
      const uint64_t island_offset = *group.front()->entry_island_offset;
      const MoiSpecialStateSgprs special_state = *moi_special_state_sgprs(group_options);

      std::vector<uint32_t> island_words;
      if (collapsed_spill_router) {
        const auto save_scc = instrumentation::build_s_cselect_b32(
            special_state.scc_save_sgpr, scalar_positive_inline_u32(1),
            scalar_positive_inline_u32(0), arch);
        if (!save_scc) {
          result.errors.emplace_back(
              "ConSan MOI first-light probe could not save its dense entry SCC");
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
            "ConSan MOI first-light probe could not encode its dense entry island");
        return;
      }
      while (island_words.size() < dense_entry_island_words)
        island_words.push_back(build_s_nop(0, arch));
      const uint64_t island_bytes = island_words.size() * sizeof(uint32_t);
      if (island_offset > new_text.size() || island_bytes > new_text.size() - island_offset) {
        result.errors.emplace_back("ConSan MOI first-light probe dense entry island exceeds .text");
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
              "ConSan MOI first-light probe could not encode its dense dispatch key");
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
              "ConSan MOI first-light probe could not encode its dense entry-host comparison");
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
              "ConSan MOI first-light probe could not encode its dense entry-host target");
          return;
        }
        const auto skip =
            instrumentation::build_s_cbranch_scc0(static_cast<int16_t>(matched_words.size()), arch);
        if (!skip) {
          result.errors.emplace_back(
              "ConSan MOI first-light probe could not encode its dense entry-host branch");
          return;
        }
        dispatcher_words.push_back(*skip);
        dispatcher_words.insert(dispatcher_words.end(), matched_words.begin(), matched_words.end());
      }

      for (size_t group_index = 0; group_index < group.size(); ++group_index) {
        const PlannedAccessRecordPatch *planned_patch = group[group_index];
        const ConSanMoiCandidate &candidate = *planned_patch->candidate;
        const auto compare = instrumentation::build_s_cmp_eq_u32(
            key_sgpr,
            explicit_dense_key ? scalar_positive_inline_u32(static_cast<uint16_t>(group_index + 1u))
                               : 255u,
            arch);
        if (!compare) {
          result.errors.emplace_back(
              "ConSan MOI first-light probe could not encode its dense dispatch comparison");
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
              "ConSan MOI first-light probe could not encode a dense dispatch target");
          return;
        }
        if (matched_words.size() > static_cast<size_t>(std::numeric_limits<int16_t>::max())) {
          result.errors.emplace_back(
              "ConSan MOI first-light probe dense dispatch stub is too large");
          return;
        }
        const auto skip =
            instrumentation::build_s_cbranch_scc0(static_cast<int16_t>(matched_words.size()), arch);
        if (!skip) {
          result.errors.emplace_back(
              "ConSan MOI first-light probe could not encode a dense dispatch branch");
          return;
        }
        dispatcher_words.push_back(*skip);
        dispatcher_words.insert(dispatcher_words.end(), matched_words.begin(), matched_words.end());
      }
      dispatcher_words.push_back(build_s_endpgm(arch));
      const uint64_t reserved_words = group.size() * access_relay_words;
      if (dispatcher_words.size() > reserved_words || dispatcher_offset > new_text.size() ||
          dispatcher_words.size() * sizeof(uint32_t) > new_text.size() - dispatcher_offset) {
        result.errors.emplace_back("ConSan MOI first-light probe dense dispatcher requires " +
                                   std::to_string(dispatcher_words.size()) + " words for " +
                                   std::to_string(group.size()) + " sites, but reserved " +
                                   std::to_string(reserved_words));
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
            explicit_dense_key ? std::optional<uint32_t>(
                                     build_s_mov_b32(key_sgpr, scalar_positive_inline_u32(0), arch))
                               : instrumentation::build_s_call_i64(call_return_sgpr,
                                                                   /*next instruction=*/0, arch);
        if (!host_entry || entry_host->second.host_offset > new_text.size() ||
            sizeof(uint32_t) > new_text.size() - entry_host->second.host_offset) {
          result.errors.emplace_back(
              "ConSan MOI first-light probe could not encode its dense entry host");
          return;
        }
        std::memcpy(new_text.data() + entry_host->second.host_offset, &*host_entry,
                    sizeof(uint32_t));
      }

      for (size_t group_index = 0; group_index < group.size(); ++group_index) {
        const PlannedAccessRecordPatch *planned_patch = group[group_index];
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
              "ConSan MOI first-light probe could not encode a dense call anchor");
          return;
        }
        std::vector<uint32_t> anchor_words(candidate.size / sizeof(uint32_t), build_s_nop(0, arch));
        if (explicit_dense_key) {
          if (!call_delta || group_index >= 64u) {
            result.errors.emplace_back(
                "ConSan MOI first-light probe could not encode an explicit dense branch key");
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
    for (const PlannedAccessRecordPatch &planned_patch : planned_patches) {
      const ConSanMoiCandidate &candidate = *planned_patch.candidate;
      ConSanOptions candidate_options = options;
      candidate_options.scratch_vgpr = planned_patch.resources.base;
      candidate_options.moi_spill_overlaps_guest_operands =
          planned_patch.spill_overlaps_guest_operands;
      if (!apply_moi_transient_sgpr_assignment(
              candidate_options, planned_patch.resources.owner_descriptor_file_offsets) ||
          !candidate_options.moi_exec_save_sgpr) {
        result.errors.emplace_back(
            "ConSan MOI first-light probe lost its owner-local transient SGPR assignment");
        return;
      }
      auto words = build_first_light_access_record_words(
          bytes, candidate, candidate_options, arch, planned_patch.record_index,
          planned_record_count, layout.access_record_capacity, planned_patch.private_epoch_offset,
          result.errors);
      if (!words)
        return;

      ConSanPatchInfo info;
      info.kind = planned_patch.placement.kind == DbiPatchPlacementKind::AppendedCave
                      ? ConSanPatchKind::TrampolineMoiAccessRecordStore
                      : ConSanPatchKind::InlineMoiAccessRecordStore;
      info.anchor_offset = candidate.text_offset;
      info.scratch_vgpr = planned_patch.resources.base;
      info.owner_descriptor_file_offsets = planned_patch.resources.owner_descriptor_file_offsets;
      info.persistent_epoch_private_offset = planned_patch.private_epoch_offset;
      if (planned_patch.private_epoch_offset)
        info.persistent_private_state_end = planned_patch.persistent_private_state_end;
      info.required_private_segment_size = planned_patch.required_private_bytes;
      if (planned_patch.spill) {
        info.spilled_vgpr_count = planned_patch.spill->vgpr_count;
      }

      if (planned_patch.placement.kind == DbiPatchPlacementKind::AppendedCave) {
        const uint64_t cave_text_offset = planned_patch.placement.body_offset;
        if (new_text.size() != cave_text_offset) {
          result.errors.emplace_back(
              "ConSan MOI first-light probe emitted a stale appended-cave mapping (expected " +
              std::to_string(cave_text_offset) + ", got " + std::to_string(new_text.size()) + ")");
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
                                        "first-light probe", result.errors)) ||
            (!planned_patch.entry_island_offset && !ret)) {
          result.errors.emplace_back(
              "ConSan MOI first-light probe appended cave branch is out of range");
          return;
        }

        if (planned_patch.entry_island_offset && !planned_patch.dense_call_anchor) {
          const MoiSpecialStateSgprs special_state = *moi_special_state_sgprs(candidate_options);
          const uint32_t candidate_entry_island_words =
              candidate_options.automatic_moi_record_replay_sgpr_spill ? kIndirectIslandWords + 1u
                                                                       : kIndirectIslandWords;
          std::vector<uint32_t> island_words;
          if (!append_moi_scc_preserving_indirect_jump(
                  island_words, *planned_patch.entry_island_offset, cave_text_offset,
                  moi_indirect_jump_sgprs(candidate_options)->pc_sgpr, special_state.scc_save_sgpr,
                  /*capture_scc=*/true, arch) ||
              island_words.size() > candidate_entry_island_words) {
            result.errors.emplace_back(
                "ConSan MOI first-light probe could not encode its entry island");
            return;
          }
          while (island_words.size() < candidate_entry_island_words)
            island_words.push_back(build_s_nop(0, arch));
          if (!write_moi_entry_island(new_text, planned_patch, island_words, patches,
                                      "first-light probe", result.errors))
            return;
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
        body_options.trailing_guest_word_count =
            options.moi_dynamic_access_records ? 0u : candidate.size / sizeof(uint32_t);
        body_options.probe_guest_instruction_offset = planned_patch.probe_guest_instruction_offset;
        body_options.body_guest_instruction_offset = &body_guest_instruction_offset;
        body_options.restore_vgpr_spill_before_trailing_guest =
            planned_patch.spill_overlaps_guest_operands;
        body_options.wrap_embedded_guest_vgpr_bank = planned_patch.wrap_embedded_guest_vgpr_bank;
        auto cave_words = assemble_moi_appended_body(planned_patch, *words, "first-light probe",
                                                     result.errors, body_options);
        if (!cave_words)
          return;
        if (planned_patch.dense_call_anchor) {
          if (arch != ROCJITSU_CODE_ARCH_GFX1250 ||
              candidate_options.automatic_moi_record_replay_sgpr_spill) {
            const MoiSpecialStateSgprs special_state = *moi_special_state_sgprs(candidate_options);
            const size_t return_begin = cave_words->size();
            if (!append_moi_scc_preserving_indirect_jump(
                    *cave_words, cave_text_offset, planned_patch.placement.return_target,
                    moi_indirect_jump_sgprs(candidate_options)->pc_sgpr,
                    special_state.scc_save_sgpr, /*capture_scc=*/true, arch)) {
              result.errors.emplace_back(
                  "ConSan MOI first-light probe could not encode its dense long return");
              return;
            }
            while (cave_words->size() < return_begin + kIndirectIslandWords)
              cave_words->push_back(build_s_nop(0, arch));
          } else {
            const uint16_t call_return_sgpr =
                static_cast<uint16_t>(*candidate_options.moi_exec_save_sgpr + 6u);
            cave_words->push_back(build_s_setpc_b64(call_return_sgpr, arch));
          }
        } else if (planned_patch.entry_island_offset) {
          const MoiSpecialStateSgprs special_state = *moi_special_state_sgprs(candidate_options);
          if (!append_moi_scc_preserving_indirect_jump(
                  *cave_words, cave_text_offset, planned_patch.placement.return_target,
                  moi_indirect_jump_sgprs(candidate_options)->pc_sgpr, special_state.scc_save_sgpr,
                  /*capture_scc=*/false, arch)) {
            result.errors.emplace_back(
                "ConSan MOI first-light probe could not encode its long return");
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
          result.errors.emplace_back("ConSan MOI first-light probe final patch size changed");
          return;
        }
        if (candidate.text_offset > new_text.size() ||
            patch_bytes > new_text.size() - candidate.text_offset) {
          result.errors.emplace_back("ConSan MOI first-light probe exceeds .text");
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
            "ConSan MOI first-light probe emitted a stale dense entry-host mapping");
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
              group_options, group.front()->resources.owner_descriptor_file_offsets) ||
          !group_options.moi_exec_save_sgpr) {
        result.errors.emplace_back(
            "ConSan MOI first-light probe lost its dense-host transient SGPR assignment");
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
            "ConSan MOI first-light probe could not encode its dense entry-host return");
        return;
      }
      while (host_words.size() < host.displaced_words.size() + kIndirectIslandWords)
        host_words.push_back(build_s_nop(0, arch));
      append_words_bytes(new_text, host_words);
    }

    if (!patcher.replace_text(new_text, kConSanMaxPatchedFileGrowth)) {
      result.errors.emplace_back("ConSan MOI first-light probe could not grow .text");
      return;
    }
    result.elf_bytes = patcher.emit();
    result.patches.insert(result.patches.end(), patches.begin(), patches.end());
    result.modified = true;
    return;
  }

  if (!apply_inline_moi_access_patches(
          bytes, planned_patches, descriptor_requirements, scalar_requirements, "first-light probe",
          arch,
          [&](const PlannedAccessRecordPatch &planned_patch) {
            ConSanOptions candidate_options = options;
            candidate_options.scratch_vgpr = planned_patch.resources.base;
            candidate_options.moi_spill_overlaps_guest_operands =
                planned_patch.spill_overlaps_guest_operands;
            if (!apply_moi_transient_sgpr_assignment(
                    candidate_options, planned_patch.resources.owner_descriptor_file_offsets))
              return std::optional<std::vector<uint32_t>>{};
            return build_first_light_access_record_words(
                bytes, *planned_patch.candidate, candidate_options, arch,
                planned_patch.record_index, planned_record_count, layout.access_record_capacity,
                planned_patch.private_epoch_offset, result.errors);
          },
          [](const PlannedAccessRecordPatch &planned_patch) {
            const ConSanMoiCandidate &candidate = *planned_patch.candidate;
            ConSanPatchInfo info;
            info.kind = ConSanPatchKind::InlineMoiAccessRecordStore;
            info.anchor_offset = candidate.text_offset;
            info.trampoline_offset = candidate.text_offset + candidate.size;
            info.original_size = static_cast<uint32_t>(planned_patch.placement.body_size);
            info.scratch_vgpr = planned_patch.resources.base;
            info.owner_descriptor_file_offsets =
                planned_patch.resources.owner_descriptor_file_offsets;
            info.persistent_epoch_private_offset = planned_patch.private_epoch_offset;
            if (planned_patch.private_epoch_offset)
              info.persistent_private_state_end = planned_patch.persistent_private_state_end;
            info.required_private_segment_size = planned_patch.required_private_bytes;
            info.relocated_guest_instruction_offset =
                candidate.text_offset + planned_patch.probe_guest_instruction_offset;
            if (planned_patch.spill)
              info.spilled_vgpr_count = planned_patch.spill->vgpr_count;
            return info;
          },
          [](std::vector<uint8_t> &, ConSanResult &) { return true; }, result))
    return;
}
