[[nodiscard]] std::optional<ConSanMoiAtomicEventKind>
atomic_event_kind_for_site(const MoiSyncInventoryIndex &sync_index, std::string_view container_name,
                           bool in_kernel, const ConSanAtomicSite &site);

void try_apply_first_light_access_record_patch(std::span<const uint8_t> bytes,
                                               const ConSanOptions &options, rj_code_arch_t arch,
                                               MoiResourcePlanningState &resource_state,
                                               MoiRecordReplayAccessOutput &access_output,
                                               std::span<const ConSanMoiCandidate> admitted,
                                               ConSanTransformArtifacts &result) {
  if (!options.moi_report_buffer_address)
    return;
  if (!consan_is_capability_arch(arch)) {
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
  if (options.moi_runtime_sample_stride == 0 || options.moi_runtime_sample_stride > (1u << 24u) ||
      (options.moi_runtime_sample_stride & (options.moi_runtime_sample_stride - 1u)) != 0u ||
      options.moi_runtime_sample_offset >= options.moi_runtime_sample_stride) {
    result.warnings.emplace_back(
        "ConSan MOI Record/Replay runtime stride must be a power of two in 1..16777216 and "
        "offset must be less than stride");
    return;
  }
  if (options.max_patches == 0 || layout.access_record_capacity == 0)
    return;
  if (!options.moi_dynamic_access_records && layout.record_replay_access_dispatch_bank_count > 1u &&
      !moi_has_runtime_hardware_dispatch_id(options)) {
    result.outcome = ConSanTransformOutcome::Unsupported;
    result.warnings.emplace_back(
        "ConSan MOI automatic Record/Replay identity tables require runtime hardware dispatch "
        "identity");
    return;
  }
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
      find_first_light_access_record_candidates(admitted);
  apply_test_kernel_filter(candidates, options);
  std::ranges::sort(candidates, [](const ConSanMoiCandidate *lhs, const ConSanMoiCandidate *rhs) {
    return lhs->anchor() < rhs->anchor();
  });
  // Ordering instrumentation composes around the relocated guest access in a
  // second text-growth pass. Keep those access bodies at the tail so their
  // one-word branch anchors remain in direct reach of the appended ordering
  // caves even for very large code objects.
  const MoiSyncInventoryIndex sync_index(result);
  std::unordered_set<uint64_t> ordering_atomic_offsets;
  if (options.moi_track_atomics) {
    const auto append_ordering_offsets = [&](const auto &container, bool in_kernel) {
      for (const ConSanAtomicSite &site : container.atomic_sites) {
        if (atomic_event_kind_for_site(sync_index, container.name, in_kernel, site))
          ordering_atomic_offsets.insert(site.text_offset);
      }
    };
    for (const ConSanKernelInfo &kernel : result.program_inventory.kernels())
      append_ordering_offsets(kernel, true);
    for (const ConSanFunctionInfo &function : result.program_inventory.functions())
      append_ordering_offsets(function, false);
  }
  std::stable_partition(candidates.begin(), candidates.end(), [&](const auto *candidate) {
    return !ordering_atomic_offsets.contains(candidate->anchor());
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
    uint64_t entry_island_word_count = 0;
    bool entry_island_at_anchor = false;
    bool dense_call_anchor = false;
    bool runtime_workgroup_gate = false;
    bool dense_runtime_workgroup_gate = false;
    std::optional<uint64_t> dense_dispatcher_offset;
    std::optional<ConSanMoiWorkgroupSources> runtime_workgroup_sources;
    std::vector<uint32_t> displaced_tail_words;
    uint32_t record_index = 0;
    uint32_t record_count = 0;
    uint32_t logical_range_index = 0;
    ResolvedMoiScratchPlan resources;
    std::optional<VgprSpillSequence> spill;
    std::optional<InlineSgprSpillSequence> scalar_spill;
    std::optional<uint16_t> gfx1250_vgpr_msb_mode;
    std::optional<uint32_t> private_epoch_offset;
    std::optional<uint32_t> private_owner_offset;
    std::optional<uint16_t> private_owner_shift;
    ConSanMoiPersistentWorkgroupPrivateOffsets private_record_replay_workgroup_offsets;
    uint32_t persistent_private_state_end = 0;
    uint32_t required_private_bytes = 0;
    uint32_t probe_guest_instruction_offset = 0;
    uint32_t guest_instruction_word_count = 0;
    bool spill_overlaps_guest_operands = false;
    bool wrap_embedded_guest_vgpr_bank = false;
    bool branch_only_scalar_spill = false;
    std::vector<uint64_t> appended_branch_relay_offsets;
    std::optional<MoiBorrowedRecordReplayEntry> borrowed_entry;
    std::optional<BranchOnlyRelayRoute> branch_only_route;
  };
  std::vector<PlannedAccessRecordPatch> planned_patches;
  uint32_t planned_record_count = 0;
  uint32_t planned_logical_range_count = 0;
  planned_patches.reserve(candidates.size());
  AmdGpuCodeObject code_object(bytes.data(), bytes.size());
  const uint64_t original_text_size =
      code_object.text_sections().size() == 1 ? code_object.text_sections().front()->size() : 0;
  DbiPatchPlacementPlanner placement_planner(arch, original_text_size);
  constexpr uint64_t kIndirectIslandWords = kMoiRecordReplayIndirectIslandWords;
  // A branch-only appended body cannot use a live scalar PC pair. Keep one
  // capacity-one entry relay and one capacity-one return relay after its
  // unconditional return branch. Earlier bodies can then extend both sides
  // of the route through already-owned appended storage as the body region
  // grows beyond one SOPP hop.
  constexpr uint64_t kBranchOnlyAppendedRelayWords = 2u;
  uint64_t max_candidate_instruction_words = 0u;
  for (const ConSanMoiCandidate *candidate : candidates) {
    max_candidate_instruction_words = std::max(
        max_candidate_instruction_words,
        (static_cast<uint64_t>(candidate->size()) + sizeof(uint32_t) - 1u) / sizeof(uint32_t));
  }
  const bool has_cluster_workgroup_id =
      std::ranges::any_of(candidates, [](const ConSanMoiCandidate *candidate) {
        return candidate->container.uses_gfx1250_cluster_workgroup_id;
      });
  const uint64_t runtime_gate_island_words = moi_runtime_workgroup_gate_reserved_words(
      static_cast<uint32_t>(max_candidate_instruction_words * sizeof(uint32_t)),
      has_cluster_workgroup_id);
  const bool has_spill_backed_scalar_assignment = std::ranges::any_of(
      options.moi_transient_sgpr_assignments, &ConSanMoiTransientSgprAssignment::spill_backed);
  const bool has_branch_only_scalar_assignment =
      std::ranges::any_of(options.moi_transient_sgpr_assignments,
                          &ConSanMoiTransientSgprAssignment::branch_only_scalar_spill);
  // Spill-backed groups need room for the extra scalar-save arm, while truly
  // large generated objects need a conservative envelope to keep every
  // routing target reachable. Applying that spacing to every ordinary RDNA4
  // object needlessly displaces access bodies from their compact entry and
  // return layout. Retain the compact envelope for moderate objects.
  constexpr size_t kLargeRdna4RelayCandidateCount = 1024u;
  const bool needs_large_rdna4_relay =
      candidates.size() >= kLargeRdna4RelayCandidateCount ||
      (has_spill_backed_scalar_assignment && candidates.size() <= 2u);
  const uint64_t ordinary_access_relay_words =
      (arch == ROCJITSU_CODE_ARCH_CDNA5 ? 24u : (needs_large_rdna4_relay ? 20u : 10u)) +
      (has_spill_backed_scalar_assignment ? 1u : 0u);
  const uint64_t access_relay_words =
      options.moi_runtime_sample_stride > 1u
          ? std::max(ordinary_access_relay_words, runtime_gate_island_words)
          : ordinary_access_relay_words;
  // Branch-only scalar assignments cannot use the ordinary indirect-entry
  // ABI. Keep two words per access slot outside that ABI as shared relay
  // capacity. Dense dispatchers retain a contiguous prefix of their combined
  // slots, leaving the corresponding tail available to the branch router.
  const uint64_t access_slot_words =
      access_relay_words + (has_branch_only_scalar_assignment ? 2u : 0u);
  const bool has_unreachable_function_candidate =
      std::ranges::any_of(candidates, [&](const ConSanMoiCandidate *candidate) {
        return !candidate->container.is_kernel() &&
               !compute_sopp_branch_simm16(candidate->anchor(), original_text_size);
      });
  const bool use_indirect_appended =
      options.moi_exec_save_sgpr.has_value() &&
      (options.moi_runtime_sample_stride > 1u || candidates.size() > 8u ||
       has_unreachable_function_candidate || has_spill_backed_scalar_assignment);
  // Compact lowering patches every split-barrier member, while the dense
  // router shares one entry per logical signal/wait pair. Reserve the same
  // member-accurate or shared barrier inventory selected by that lowering,
  // plus every fence candidate; otherwise the access pass can consume every
  // reachable island before a later synchronization pass runs.
  const uint64_t reserved_sync_island_count =
      use_indirect_appended ? moi_reserved_record_replay_sync_island_count(result, options) : 0u;
  const uint64_t reserved_sync_slot_words =
      has_branch_only_scalar_assignment
          ? kMoiRecordReplayBarrierRelaySlotWords
          : (options.moi_track_atomics ? kMoiLocalIndirectIslandWords : kIndirectIslandWords);
  const uint64_t indirect_body_begin =
      original_text_size +
      reserved_sync_island_count * reserved_sync_slot_words * sizeof(uint32_t) +
      candidates.size() * access_slot_words * sizeof(uint32_t);
  const uint64_t access_island_begin =
      original_text_size + reserved_sync_island_count * reserved_sync_slot_words * sizeof(uint32_t);
  uint64_t indirect_body_cursor = indirect_body_begin;
  std::optional<MoiLocalNopIslandAllocator> local_entry_islands;
  std::span<const uint8_t> original_text;
  if (use_indirect_appended && code_object.text_sections().size() == 1) {
    const Section *text = code_object.text_sections().front();
    const auto text_bytes =
        std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(text->data()), text->size());
    original_text = text_bytes;
    const uint32_t local_entry_island_words =
        moi_record_replay_entry_island_words(has_spill_backed_scalar_assignment);
    local_entry_islands.emplace(text_bytes, result, arch, local_entry_island_words,
                                options.preapplied_reserved_ranges);
  }
  std::unordered_map<const ConSanMoiCandidate *, uint64_t> dense_entry_islands;
  std::unordered_map<const ConSanMoiCandidate *, uint64_t> dense_dispatchers;
  const auto dense_candidates = consan_detail::partition_moi_dense_candidates(
      candidates, arch == ROCJITSU_CODE_ARCH_CDNA5 ? 0u : 64u);
  const auto &candidate_indices = dense_candidates.index_by_candidate;
  std::map<uint64_t, MoiDenseEntryHost> dense_entry_hosts;
  // The retained planning state already owns the pristine CFG and liveness
  // facts needed by dense relay selection. Re-decoding the whole code object
  // here (and a third time for component-local scalar assignments) is both
  // redundant and dominant on multi-thousand-kernel generated libraries.
  if (consan_is_capability_arch(arch) && use_indirect_appended && local_entry_islands) {
    std::vector<std::pair<uint64_t, uint64_t>> global_access_ranges;
    global_access_ranges.reserve(candidates.size());
    for (const ConSanMoiCandidate *candidate : candidates) {
      if (candidate->size() != 0u)
        global_access_ranges.emplace_back(candidate->anchor(),
                                          candidate->anchor() + candidate->size());
    }
    const consan_detail::MoiOccupiedTextRanges occupied_accesses(std::move(global_access_ranges));
    using DenseHostCandidateCacheKey =
        std::tuple<ConSanProgramContainerKind, std::string, uint64_t, std::vector<uint64_t>,
                   std::vector<std::pair<uint16_t, uint16_t>>>;
    std::map<DenseHostCandidateCacheKey, std::vector<uint64_t>> dense_host_candidate_cache;
    std::vector<std::pair<uint64_t, uint64_t>> claimed_dense_host_ranges;
    for (const auto &group : dense_candidates.groups) {
      const bool owner_is_kernel = group.owner.kind == ConSanProgramContainerKind::Kernel;
      const ConSanCandidateResourcePlan *group_resource =
          group.candidates.empty() ? nullptr
                                   : resource_plan_for_candidate(result, *group.candidates.front());
      ConSanOptions group_options = options;
      if (group_resource != nullptr)
        (void)apply_moi_transient_sgpr_assignment(group_options,
                                                  group_resource->owner_descriptor_file_offsets);
      if (group_resource != nullptr &&
          moi_transient_sgpr_assignment_is_branch_only(
              options, group_resource->owner_descriptor_file_offsets)) {
        continue;
      }
      const uint32_t group_entry_island_words = moi_record_replay_entry_island_words(
          group_options.automatic_moi_record_replay_sgpr_spill);
      const bool has_dense_neighbors =
          group.has_overlapping_entries(group_entry_island_words * sizeof(uint32_t));
      const bool has_stranded_entry =
          std::ranges::any_of(group.candidates, [&](const ConSanMoiCandidate *candidate) {
            const size_t candidate_index = candidate_indices.at(candidate);
            const uint64_t default_entry =
                access_island_begin + candidate_index * access_slot_words * sizeof(uint32_t);
            return !compute_sopp_branch_simm16(candidate->anchor(), default_entry);
          });
      const bool explicit_anchors_fit =
          arch == ROCJITSU_CODE_ARCH_CDNA5 || group.every_site_has_size(2u * sizeof(uint32_t));
      const bool singleton_has_direct_island =
          group.candidates.size() == 1u &&
          local_entry_islands->has_reachable(group.candidates.front()->anchor(),
                                             group_entry_island_words);
      const bool has_spill_backed_access =
          std::ranges::any_of(group.candidates, [&](const ConSanMoiCandidate *candidate) {
            const ConSanCandidateResourcePlan *plan =
                resource_plan_for_candidate(result, *candidate);
            return plan != nullptr && plan->source == ConSanRegisterAllocationSource::SpillRequired;
          });
      // gfx1250 uses the shared call relay as a code-size optimization for
      // nearby sites. Spill-backed access bodies need the same return-PC path
      // on every family: a compact wave64 object can otherwise append enough
      // save/probe/restore bodies to strand later returns even though every
      // original site can still reach its nearby entry slot. Keep the simpler
      // per-site relay for moderate non-spilling RDNA3/RDNA4/CDNA3/CDNA4
      // objects and use dense routing only for stranded or spill-dense groups.
      const bool prefer_dense_neighbors =
          arch == ROCJITSU_CODE_ARCH_CDNA5 || has_spill_backed_access;
      if (!explicit_anchors_fit || singleton_has_direct_island ||
          (!(prefer_dense_neighbors && has_dense_neighbors) && !has_stranded_entry))
        continue;
      const std::vector<uint64_t> anchors = group.anchors();
      const uint64_t dispatcher =
          access_island_begin + group.first_candidate_index * access_slot_words * sizeof(uint32_t);
      auto island =
          local_entry_islands->claim_reachable_from_all(anchors, group_entry_island_words);
      if (!island && group_resource != nullptr) {
        const ConSanKernelInfo *kernel =
            owner_is_kernel ? result.program_inventory.find_kernel_by_name(group.owner.name)
                            : nullptr;
        const ConSanFunctionInfo *function =
            !owner_is_kernel ? result.program_inventory.find_function_by_name(group.owner.name)
                             : nullptr;
        const bool has_owner_range =
            (kernel != nullptr && kernel->has_text_range && kernel->code_size != 0u) ||
            (function != nullptr && function->code_size != 0u);
        if (has_owner_range) {
          const std::vector<MoiSgprRange> host_bootstrap_ranges =
              moi_relocatable_host_scalar_ranges(group_options,
                                                 MoiRelocatableHostScalarState::AccessRouter, arch);
          const uint64_t host_words_count = group_entry_island_words + 1u;
          const uint64_t owner_begin =
              owner_is_kernel ? kernel->entry_text_offset : function->entry_text_offset;
          const uint64_t owner_end =
              owner_begin + (owner_is_kernel ? kernel->code_size : function->code_size);
          std::vector<std::pair<uint16_t, uint16_t>> bootstrap_key;
          bootstrap_key.reserve(host_bootstrap_ranges.size());
          for (const MoiSgprRange &range : host_bootstrap_ranges)
            bootstrap_key.emplace_back(range.base, range.width);
          DenseHostCandidateCacheKey cache_key{group.owner.kind, group.owner.name, host_words_count,
                                               group_resource->owner_descriptor_file_offsets,
                                               std::move(bootstrap_key)};
          auto cached_candidates = dense_host_candidate_cache.find(cache_key);
          if (cached_candidates == dense_host_candidate_cache.end()) {
            MoiDenseOwnerLivenessProof host_owner_proof =
                resource_state.prepare_dense_liveness_for_owners(
                    group_resource->owner_descriptor_file_offsets, host_bootstrap_ranges);
            const auto &owner_candidates = dense_candidates.candidates_by_owner.at(group.owner);
            const consan_detail::MoiOccupiedTextRanges occupied_owner_sites =
                owner_is_kernel
                    ? collect_moi_dense_owner_occupied_ranges(*kernel, owner_candidates)
                    : collect_moi_dense_owner_occupied_ranges(*function, owner_candidates);
            std::vector<std::pair<uint64_t, uint64_t>> owner_sync_ranges;
            owner_sync_ranges.reserve(result.program_inventory.sync().sync_sequences.size());
            const auto append_occupied_range = [&](uint64_t begin, uint64_t size) {
              if (size != 0u)
                owner_sync_ranges.emplace_back(begin, begin + size);
            };
            // Compiler atomics can be expressed as ordinary loads/stores plus
            // exact wait/cache clauses. They are downstream synchronization
            // sites even though they do not appear in atomic_sites or
            // fence_sites. Preserve the complete qualified sequence: a dense
            // access router that borrows any part of it as a relocatable host
            // otherwise prevents the ordering pass from emitting its record.
            for (const ConSanSyncSequence &sequence :
                 result.program_inventory.sync().sync_sequences) {
              if (sequence.kind != ConSanSyncSequenceKind::OrdinaryMemory ||
                  sequence.in_kernel != owner_is_kernel ||
                  sequence.container_name != group.owner.name ||
                  !consan_sync_confidence_meets(sequence.confidence,
                                                ConSanSemanticConfidence::Conservative) ||
                  !consan_sync_confidence_meets(sequence.memory_role_confidence,
                                                ConSanSemanticConfidence::Conservative) ||
                  sequence.memory_role == ConSanSyncMemoryRole::Unknown ||
                  sequence.memory_role == ConSanSyncMemoryRole::None ||
                  sequence.end_text_offset <= sequence.begin_text_offset) {
                continue;
              }
              append_occupied_range(sequence.begin_text_offset,
                                    sequence.end_text_offset - sequence.begin_text_offset);
            }
            const consan_detail::MoiOccupiedTextRanges occupied_ordering_sequences(
                std::move(owner_sync_ranges));
            const auto overlaps_instrumented_site = [&](uint64_t begin, uint64_t end) {
              return occupied_accesses.overlaps(begin, end) ||
                     occupied_owner_sites.overlaps(begin, end) ||
                     occupied_ordering_sequences.overlaps(begin, end);
            };
            std::vector<uint64_t> host_candidates = collect_moi_dense_relay_host_candidates(
                resource_state,
                {
                    .owner_begin = owner_begin,
                    .owner_end = owner_end,
                    .host_word_count = static_cast<size_t>(host_words_count),
                    .anchors = anchors,
                    .preapplied_reserved_ranges = options.preapplied_reserved_ranges,
                    .owner_proof = host_owner_proof,
                    .bootstrap_ranges = host_bootstrap_ranges,
                },
                overlaps_instrumented_site);
            cached_candidates =
                dense_host_candidate_cache.emplace(std::move(cache_key), std::move(host_candidates))
                    .first;
          }
          auto best_host = select_moi_dense_relay_host(
              resource_state, original_text, static_cast<size_t>(host_words_count), anchors,
              cached_candidates->second, owner_begin, claimed_dense_host_ranges);
          if (best_host) {
            claimed_dense_host_ranges.emplace_back(best_host->host_offset,
                                                   best_host->host_offset +
                                                       host_words_count * sizeof(uint32_t));
            island = best_host->host_offset + sizeof(uint32_t);
            dense_entry_hosts.emplace(dispatcher,
                                      MoiDenseEntryHost{
                                          .host_offset = best_host->host_offset,
                                          .displaced_words = std::move(best_host->displaced_words),
                                      });
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
  BranchOnlyRelayRouter branch_router;
  std::vector<uint64_t> generated_branch_relay_offers;
  std::vector<std::pair<uint64_t, uint64_t>> generated_branch_relay_ranges;
  if (use_indirect_appended && access_slot_words > access_relay_words) {
    generated_branch_relay_offers.reserve(2u * static_cast<size_t>(reserved_sync_island_count) +
                                          2u * candidates.size());
    generated_branch_relay_ranges.reserve(static_cast<size_t>(reserved_sync_island_count) +
                                          candidates.size());
    // The barrier prefix can itself be longer than one SOPP hop. Keep two
    // words per slot outside the seven-word barrier island so branch-only
    // reservoir discovery and final access routing have a continuous first
    // foothold without consuming storage owned by the later barrier pass.
    // Only the branch-only layout owns a dedicated two-word spine. An
    // eight-word synchronization slot must remain intact for a later fence.
    if (reserved_sync_slot_words == kMoiRecordReplayBarrierRelaySlotWords) {
      for (uint64_t index = 0u; index < reserved_sync_island_count; ++index) {
        const uint64_t begin =
            original_text_size +
            (index * reserved_sync_slot_words + kIndirectIslandWords) * sizeof(uint32_t);
        const uint64_t end =
            original_text_size + (index + 1u) * reserved_sync_slot_words * sizeof(uint32_t);
        generated_branch_relay_ranges.emplace_back(begin, end);
        for (uint64_t relay = begin; relay < end; relay += sizeof(uint32_t))
          generated_branch_relay_offers.push_back(relay);
      }
    }
    std::map<uint64_t, size_t> dense_candidate_counts;
    for (const auto &[candidate, dispatcher] : dense_dispatchers) {
      (void)candidate;
      ++dense_candidate_counts[dispatcher];
    }
    std::vector<std::pair<uint64_t, uint64_t>> occupied_access_relay_ranges;
    occupied_access_relay_ranges.reserve(candidates.size());
    for (size_t index = 0u; index < candidates.size(); ++index) {
      const ConSanMoiCandidate *candidate = candidates[index];
      if (dense_dispatchers.contains(candidate))
        continue;
      const uint64_t begin = access_island_begin + index * access_slot_words * sizeof(uint32_t);
      occupied_access_relay_ranges.emplace_back(begin,
                                                begin + access_relay_words * sizeof(uint32_t));
    }
    for (const auto &[dispatcher, count] : dense_candidate_counts) {
      occupied_access_relay_ranges.emplace_back(
          dispatcher, dispatcher + count * access_relay_words * sizeof(uint32_t));
    }
    std::ranges::sort(occupied_access_relay_ranges);
    uint64_t cursor = access_island_begin;
    const uint64_t end =
        access_island_begin + candidates.size() * access_slot_words * sizeof(uint32_t);
    const auto append_generated_range = [&](uint64_t begin, uint64_t range_end) {
      if (begin >= range_end)
        return;
      generated_branch_relay_ranges.emplace_back(begin, range_end);
      for (uint64_t relay = begin; relay < range_end; relay += sizeof(uint32_t))
        generated_branch_relay_offers.push_back(relay);
    };
    for (const auto &[begin, range_end] : occupied_access_relay_ranges) {
      append_generated_range(cursor, begin);
      cursor = std::max(cursor, range_end);
    }
    append_generated_range(cursor, end);
  }
  // Seed the immutable appended relay bank before reservoir discovery. A
  // reservoir that cannot reach its displaced body directly may use this bank
  // and previously planned reservoir tails to extend the frontier recursively.
  for (uint64_t relay : generated_branch_relay_offers)
    (void)branch_router.offer(relay, BranchOnlyRelayProvenance::GeneratedBank);
  DbiPatchPlacementPlanner branch_only_placement_planner(arch, original_text_size);
  if (use_indirect_appended && indirect_body_begin > original_text_size) {
    std::string reservation_error;
    if (!branch_only_placement_planner.reserve_appended_prefix(
            indirect_body_begin - original_text_size, &reservation_error)) {
      result.errors.emplace_back("ConSan MOI first-light probe " + reservation_error);
      return;
    }
  }
  BranchOnlyDirectRelayReservoirSet direct_branch_reservoirs;
  const auto uses_branch_only_scalar_spill = [&](const ConSanMoiCandidate *candidate) {
    const ConSanCandidateResourcePlan *plan = resource_plan_for_candidate(result, *candidate);
    return plan != nullptr && !plan->owner_descriptor_file_offsets.empty() &&
           moi_transient_sgpr_assignment_is_branch_only(options,
                                                        plan->owner_descriptor_file_offsets);
  };
  struct BorrowedEntryProtectedRange {
    uint64_t begin = 0u;
    uint64_t end = 0u;
    const ConSanMoiCandidate *access = nullptr;
  };
  std::vector<BorrowedEntryProtectedRange> borrowed_entry_protected_ranges;
  borrowed_entry_protected_ranges.reserve(
      candidates.size() + result.program_inventory.kernels().size() +
      result.program_inventory.functions().size() + options.preapplied_reserved_ranges.size());
  for (const ConSanMoiCandidate *candidate : candidates) {
    borrowed_entry_protected_ranges.push_back(
        {candidate->anchor(), candidate->anchor() + candidate->size(), candidate});
  }
  const auto append_borrowed_entry_sync_ranges = [&](const auto &container) {
    for (const ConSanBarrierSite &site : container.barrier_sites)
      borrowed_entry_protected_ranges.push_back(
          {site.text_offset, site.text_offset + site.size, nullptr});
    for (const ConSanFenceSite &site : container.fence_sites)
      borrowed_entry_protected_ranges.push_back(
          {site.text_offset, site.text_offset + site.size, nullptr});
    for (const ConSanAtomicSite &site : container.atomic_sites)
      borrowed_entry_protected_ranges.push_back(
          {site.text_offset, site.text_offset + site.size, nullptr});
  };
  for (const ConSanKernelInfo &kernel : result.program_inventory.kernels())
    append_borrowed_entry_sync_ranges(kernel);
  for (const ConSanFunctionInfo &function : result.program_inventory.functions())
    append_borrowed_entry_sync_ranges(function);
  for (const ConSanPreappliedReservedRange &range : options.preapplied_reserved_ranges) {
    borrowed_entry_protected_ranges.push_back(
        {range.text_offset, range.text_offset + range.size, nullptr});
  }
  std::ranges::sort(borrowed_entry_protected_ranges, {}, &BorrowedEntryProtectedRange::begin);
  const auto borrowed_entry_words_for_candidate =
      [&](const ConSanMoiCandidate *candidate) -> std::optional<std::vector<uint32_t>> {
    const ConSanCandidateResourcePlan *plan = resource_plan_for_candidate(result, *candidate);
    if (plan == nullptr || original_text.empty() ||
        !moi_transient_sgpr_assignment_uses_borrowed_record_replay_entry(
            options, plan->owner_descriptor_file_offsets)) {
      return std::nullopt;
    }
    std::vector<std::string> ignored_errors;
    std::vector<uint32_t> displaced_words;
    while (displaced_words.size() < kMoiRecordReplayBorrowedEntryIslandWords) {
      auto decoded = decode_relocatable_entry_instruction(
          original_text, candidate->anchor() + displaced_words.size() * sizeof(uint32_t), arch,
          ignored_errors, resource_state.decoder.get());
      if (!decoded)
        return std::nullopt;
      displaced_words.insert(displaced_words.end(), decoded->begin(), decoded->end());
    }
    const uint64_t displaced_end = candidate->anchor() + displaced_words.size() * sizeof(uint32_t);
    if (!moi_borrowed_entry_window_is_single_entry(resource_state, candidate->anchor(),
                                                   displaced_words.size() * sizeof(uint32_t))) {
      return std::nullopt;
    }
    auto protected_range =
        std::ranges::lower_bound(borrowed_entry_protected_ranges, candidate->anchor(), {},
                                 &BorrowedEntryProtectedRange::begin);
    for (; protected_range != borrowed_entry_protected_ranges.end() &&
           protected_range->begin < displaced_end;
         ++protected_range) {
      if (protected_range->access != candidate)
        return std::nullopt;
    }
    return displaced_words;
  };
  const size_t branch_only_candidate_count =
      std::ranges::count_if(candidates, uses_branch_only_scalar_spill);
  const auto resource_plan_uses_branch_only_scalar_spill = [&](const auto &plan) {
    return !plan.owner_descriptor_file_offsets.empty() &&
           moi_transient_sgpr_assignment_is_branch_only(options,
                                                        plan.owner_descriptor_file_offsets);
  };
  const size_t branch_only_barrier_count =
      std::ranges::count_if(result.resource_plans, [&](const ConSanCandidateResourcePlan &plan) {
        return plan.site_kind == ConSanResourceSiteKind::Barrier &&
               resource_plan_uses_branch_only_scalar_spill(plan);
      });
  const size_t branch_only_routing_demand_count =
      saturated_add(branch_only_candidate_count, branch_only_barrier_count);
  if (branch_only_routing_demand_count != 0u && use_indirect_appended && !original_text.empty()) {
    std::vector<std::pair<uint64_t, uint64_t>> protected_ranges;
    protected_ranges.reserve(candidates.size() + result.program_inventory.kernels().size() +
                             result.program_inventory.functions().size() +
                             options.preapplied_reserved_ranges.size() + dense_entry_hosts.size());
    const auto append_synchronization_ranges = [&](const auto &container) {
      for (const ConSanBarrierSite &site : container.barrier_sites)
        protected_ranges.emplace_back(site.text_offset, site.text_offset + site.size);
      for (const ConSanFenceSite &site : container.fence_sites)
        protected_ranges.emplace_back(site.text_offset, site.text_offset + site.size);
      for (const ConSanAtomicSite &site : container.atomic_sites)
        protected_ranges.emplace_back(site.text_offset, site.text_offset + site.size);
    };
    for (const ConSanKernelInfo &kernel : result.program_inventory.kernels())
      append_synchronization_ranges(kernel);
    for (const ConSanFunctionInfo &function : result.program_inventory.functions())
      append_synchronization_ranges(function);
    for (const ConSanMoiCandidate *candidate : candidates) {
      const auto borrowed_words = borrowed_entry_words_for_candidate(candidate);
      protected_ranges.emplace_back(
          candidate->anchor(),
          candidate->anchor() +
              (borrowed_words ? borrowed_words->size() * sizeof(uint32_t) : candidate->size()));
    }
    for (const ConSanKernelInfo &kernel : result.program_inventory.kernels()) {
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
    std::vector<BasicBlock *> reservoir_blocks;
    reservoir_blocks.reserve(resource_state.blocks.size());
    for (const std::unique_ptr<BasicBlock> &block : resource_state.blocks) {
      if (block)
        reservoir_blocks.push_back(block.get());
    }
    uint64_t route_frontier_source = std::numeric_limits<uint64_t>::max();
    for (const ConSanMoiCandidate *candidate : candidates) {
      if (uses_branch_only_scalar_spill(candidate))
        route_frontier_source = std::min(route_frontier_source, candidate->anchor());
    }
    for (const ConSanCandidateResourcePlan &plan : result.resource_plans) {
      if (plan.site_kind == ConSanResourceSiteKind::Barrier &&
          resource_plan_uses_branch_only_scalar_spill(plan)) {
        route_frontier_source = std::min(route_frontier_source, plan.text_offset);
      }
    }
    if (route_frontier_source == std::numeric_limits<uint64_t>::max()) {
      result.errors.emplace_back(
          "ConSan MOI first-light probe lost its branch-only routing frontier");
      return;
    }
    std::string relay_error;
    if (!branch_router.plan_direct_reservoirs(
            reservoir_blocks, original_text, protected_ranges, arch, route_frontier_source,
            saturated_multiply(2u, branch_only_routing_demand_count), branch_only_placement_planner,
            direct_branch_reservoirs, &relay_error,
            {.discovery = options.direct_reservoir_planning_work_limit})) {
      result.errors.emplace_back("ConSan MOI first-light probe " + relay_error);
      return;
    }
    indirect_body_cursor = branch_only_placement_planner.appended_end();
  }
  std::vector<uint64_t> pristine_branch_relay_offers =
      local_entry_islands ? local_entry_islands->available_word_offsets() : std::vector<uint64_t>{};
  std::set<uint64_t> branch_relay_offsets(pristine_branch_relay_offers.begin(),
                                          pristine_branch_relay_offers.end());
  branch_relay_offsets.insert(generated_branch_relay_offers.begin(),
                              generated_branch_relay_offers.end());
  for (const auto &[relay, reservoir] : direct_branch_reservoirs.reservoir_by_relay) {
    (void)reservoir;
    branch_relay_offsets.insert(relay);
  }
  for (const BranchOnlyDirectRelayReservoir &reservoir : direct_branch_reservoirs.reservoirs) {
    if (!reservoir.route)
      continue;
    for (const BranchOnlyRelayClaim &claim : reservoir.route->claims)
      branch_relay_offsets.erase(claim.offset);
  }
  const auto direct_or_forward_relay = [&](uint64_t source, uint64_t target) {
    if (compute_sopp_branch_simm16(source, target))
      return true;
    const auto relay = branch_relay_offsets.upper_bound(source);
    return relay != branch_relay_offsets.end() && *relay < target &&
           compute_sopp_branch_simm16(source, *relay).has_value();
  };
  const auto direct_or_backward_relay = [&](uint64_t source, uint64_t target) {
    if (compute_sopp_branch_simm16(source, target))
      return true;
    auto relay = branch_relay_offsets.lower_bound(source);
    if (relay == branch_relay_offsets.begin())
      return false;
    --relay;
    return *relay > target && compute_sopp_branch_simm16(source, *relay).has_value();
  };
  std::unordered_set<uint64_t> selected_record_replay_anchor_relay_offsets;
  const auto publish_record_replay_anchor_relays =
      [&](uint64_t anchor_offset, uint32_t original_size,
          std::unordered_set<uint64_t> &selected_relay_offsets) {
        if (original_size <= sizeof(uint32_t))
          return;
        branch_router.retire_range(anchor_offset, original_size);
        for (uint64_t offset = anchor_offset + sizeof(uint32_t);
             offset < anchor_offset + original_size; offset += sizeof(uint32_t)) {
          if (selected_relay_offsets.insert(offset).second)
            branch_relay_offsets.insert(offset);
        }
      };
  // Every branch-only request needs independent entry and return capacity at
  // each cut through a long appended region. Per-body tails preserve local
  // continuity, but a dense prefix of ordinary bodies can otherwise offer
  // fewer words per SOPP window than the later batch has routes. Periodic
  // banks make that cut capacity explicit and scale it with the actual batch.
  constexpr uint64_t kBranchOnlySpineSpacing = kSoppBranchMaximumForwardReachBytes / 2u;
  const size_t branch_only_spine_bank_word_count =
      saturated_multiply(2u, branch_only_routing_demand_count);
  const bool branch_only_spine_has_bounded_density =
      branch_only_periodic_relay_bank_has_bounded_density(branch_only_spine_bank_word_count,
                                                          kBranchOnlySpineSpacing);
  if (branch_only_routing_demand_count != 0u && !branch_only_spine_has_bounded_density) {
    result.warnings.emplace_back(
        "ConSan MOI first-light omitted oversized periodic branch-only relay banks");
  }
  uint64_t branch_only_spine_frontier = indirect_body_begin;
  const auto ensure_branch_only_appended_spine = [&](uint64_t next_body_bytes) {
    if (branch_only_routing_demand_count == 0u || !branch_only_spine_has_bounded_density)
      return true;
    if (next_body_bytes > kBranchOnlySpineSpacing) {
      result.errors.emplace_back(
          "ConSan MOI first-light branch-only appended body exceeds its relay-spine spacing");
      return false;
    }
    if (indirect_body_cursor <=
        branch_only_spine_frontier + kBranchOnlySpineSpacing - next_body_bytes) {
      return true;
    }
    if (branch_only_placement_planner.appended_end() > indirect_body_cursor) {
      result.errors.emplace_back(
          "ConSan MOI first-light branch-only relay spine has invalid geometry");
      return false;
    }
    std::string reservation_error;
    if (branch_only_placement_planner.appended_end() < indirect_body_cursor &&
        !branch_only_placement_planner.reserve_appended_prefix(
            indirect_body_cursor - branch_only_placement_planner.appended_end(),
            &reservation_error)) {
      result.errors.emplace_back("ConSan MOI first-light probe " + reservation_error);
      return false;
    }
    const uint64_t bank_bytes =
        saturated_multiply(branch_only_spine_bank_word_count, sizeof(uint32_t));
    if (branch_only_spine_bank_word_count == std::numeric_limits<size_t>::max() ||
        bank_bytes == std::numeric_limits<uint64_t>::max() ||
        !branch_only_placement_planner.reserve_appended_prefix(bank_bytes, &reservation_error)) {
      result.errors.emplace_back(
          "ConSan MOI first-light branch-only relay spine could not reserve its route bank" +
          (reservation_error.empty() ? std::string{} : ": " + reservation_error));
      return false;
    }
    const uint64_t bank_begin = indirect_body_cursor;
    for (size_t word = 0u; word < branch_only_spine_bank_word_count; ++word) {
      const uint64_t relay = bank_begin + word * sizeof(uint32_t);
      if (!branch_router.offer(relay, BranchOnlyRelayProvenance::GeneratedBank)) {
        result.errors.emplace_back(
            "ConSan MOI first-light probe could not publish its appended branch-only spine");
        return false;
      }
      branch_relay_offsets.insert(relay);
    }
    generated_branch_relay_ranges.emplace_back(bank_begin, bank_begin + bank_bytes);
    indirect_body_cursor = branch_only_placement_planner.appended_end();
    // Spacing applies to appended body bytes after the newest complete bank.
    // Measuring again from bank_begin makes a wide demand-scaled bank consume
    // its own interval and can trigger another equally wide bank after only a
    // few body words.
    branch_only_spine_frontier = indirect_body_cursor;
    return true;
  };
  std::vector<const ConSanMoiCandidate *> planning_candidates = candidates;
  if (branch_only_candidate_count != 0u && selected_candidate_limit >= candidates.size()) {
    // A selected multiword access unconditionally branches away from its
    // original instruction. Every remaining word in that instruction is then
    // a capacity-one relay host. Plan ordinary appended accesses first so
    // their paid tails form the complete-text routing frontier; then plan
    // branch-only sites from high to low text offsets so each new tail can
    // extend that frontier toward earlier sites. Subset selection retains the
    // caller's original order because an omitted donor cannot be assumed.
    std::ranges::stable_sort(planning_candidates,
                             [&](const ConSanMoiCandidate *lhs, const ConSanMoiCandidate *rhs) {
                               const bool lhs_branch_only = uses_branch_only_scalar_spill(lhs);
                               const bool rhs_branch_only = uses_branch_only_scalar_spill(rhs);
                               if (lhs_branch_only != rhs_branch_only)
                                 return !lhs_branch_only;
                               if (lhs_branch_only)
                                 return lhs->anchor() > rhs->anchor();
                               return lhs->anchor() < rhs->anchor();
                             });
  }
  MoiSpillManagers spill_managers;
  std::unordered_map<uint64_t, std::optional<MoiPrivateEpochLayout>> private_layout_by_descriptor;
  uint64_t planned_anchor_end = 0;
  for (const ConSanMoiCandidate *candidate_ptr : planning_candidates) {
    const ConSanMoiCandidate &candidate = *candidate_ptr;
    const bool use_dense_call = dense_entry_islands.contains(candidate_ptr);
    const bool candidate_uses_branch_only_scalar_spill =
        uses_branch_only_scalar_spill(candidate_ptr);
    if (use_indirect_appended && !use_dense_call && !candidate_uses_branch_only_scalar_spill &&
        candidate.anchor() < planned_anchor_end) {
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
    ConSanOptions candidate_options = options;
    candidate_options.scratch_vgpr = resources->base;
    if (!apply_moi_transient_sgpr_assignment(candidate_options,
                                             resources->owner_descriptor_file_offsets)) {
      result.warnings.emplace_back(
          "ConSan MOI first-light probe found no common transient SGPR assignment");
      continue;
    }
    // Persistent placement can deliberately mix private state for a fixed-
    // stack component with an owner-local VGPR tuple for a disconnected
    // dynamic-stack component. Resolve that choice before building private
    // offsets; the code-object-wide marker alone does not describe this site.
    if (!apply_moi_persistent_vgpr_assignment(candidate_options,
                                              resources->owner_descriptor_file_offsets)) {
      result.warnings.emplace_back(
          "ConSan MOI first-light probe found no common persistent assignment");
      continue;
    }
    std::optional<MoiPrivateEpochLayout> private_layout;
    std::optional<uint16_t> private_owner_shift;
    if (candidate_options.automatic_moi_private_epoch) {
      const bool persist_workitem_owner =
          candidate_options.moi_owner_source == ConSanMoiOwnerSource::WorkitemId;
      if (resources->owner_descriptor_file_offsets.size() == 1u) {
        const uint64_t descriptor = resources->owner_descriptor_file_offsets.front();
        auto [cached, inserted] =
            private_layout_by_descriptor.try_emplace(descriptor, std::nullopt);
        if (inserted) {
          cached->second = build_moi_private_epoch_layout(result, *resources, arch, result.warnings,
                                                          /*include_owner=*/persist_workitem_owner,
                                                          /*include_workgroup_key=*/false,
                                                          /*include_record_replay_workgroup=*/true);
        }
        private_layout = cached->second;
      } else {
        private_layout = build_moi_private_epoch_layout(result, *resources, arch, result.warnings,
                                                        /*include_owner=*/persist_workitem_owner,
                                                        /*include_workgroup_key=*/false,
                                                        /*include_record_replay_workgroup=*/true);
      }
      if (private_layout && private_layout->owner_offset)
        private_owner_shift = common_moi_private_owner_shift(bytes, *resources, result.warnings);
      if (!private_layout) {
        result.warnings.emplace_back(
            "ConSan MOI first-light probe could not build private epoch state");
        continue;
      }
      if (private_layout->owner_offset && !private_owner_shift)
        continue;
    }
    if (private_layout) {
      candidate_options.moi_record_replay_workgroup_private_offsets =
          private_layout->record_replay_workgroup_offsets;
    }
    const bool branch_only_scalar_spill = moi_transient_sgpr_assignment_is_branch_only(
        options, resources->owner_descriptor_file_offsets);
    if (!apply_record_replay_entry_workgroup_assignment(candidate_options,
                                                        resources->owner_descriptor_file_offsets)) {
      result.warnings.emplace_back(
          "ConSan MOI first-light probe found no common entry workgroup assignment");
      continue;
    }
    const std::optional<ConSanMoiWorkgroupSources> runtime_workgroup_sources =
        record_replay_persistent_workgroup_sources(candidate_options, arch);
    const bool has_probe_entry_runtime_workgroup_gate =
        runtime_workgroup_sources && moi_has_probe_entry_runtime_workgroup_gate(
                                         candidate_options, *runtime_workgroup_sources, arch);
    std::optional<VgprSpillSequence> spill;
    if (resources->source == ConSanRegisterAllocationSource::SpillRequired ||
        moi_scalar_spill_requires_dynamic_vgpr_frame(result, *resources, candidate_options)) {
      spill = build_moi_spill_sequence(
          result, *resources, candidate_options, spill_managers, arch, result.warnings,
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
    bool spill_overlaps_guest_operands = false;
    if (spill && !options.moi_dynamic_access_records) {
      std::vector<std::string> ignored_overlap_errors;
      spill_overlaps_guest_operands = reject_candidate_scratch_range_overlap(
          candidate, resources->base, resources->count, ignored_overlap_errors);
    }
    const uint32_t candidate_entry_island_words = moi_record_replay_entry_island_words(
        candidate_options.automatic_moi_record_replay_sgpr_spill);
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
    const auto &access_ranges = candidate.ranges;
    if (access_ranges.empty()) {
      result.warnings.emplace_back(
          "ConSan MOI first-light probe could not recover an admitted access range");
      continue;
    }
    const uint64_t candidate_record_count_wide =
        static_cast<uint64_t>(access_ranges.size()) *
        (options.moi_dynamic_access_records
             ? 1u
             : static_cast<uint64_t>(layout.record_replay_access_dispatch_bank_count) *
                   layout.record_replay_access_owner_bank_count);
    if (access_ranges.size() > std::numeric_limits<uint32_t>::max() - planned_logical_range_count ||
        candidate_record_count_wide > std::numeric_limits<uint32_t>::max()) {
      result.warnings.emplace_back(
          "ConSan MOI first-light probe skipped a supported load/store site whose "
          "logical-range inventory exceeds the report ABI");
      continue;
    }
    const uint32_t candidate_record_count = static_cast<uint32_t>(candidate_record_count_wide);
    if (!options.moi_dynamic_access_records &&
        (planned_record_count > layout.access_record_capacity ||
         candidate_record_count > layout.access_record_capacity - planned_record_count)) {
      result.warnings.emplace_back(
          "ConSan MOI first-light probe skipped a supported load/store site without enough "
          "report-buffer access-record slots");
      continue;
    }
    std::vector<std::string> candidate_errors;
    uint32_t probe_guest_instruction_offset = 0;
    uint32_t guest_instruction_word_count = 0;
    auto words = build_first_light_access_record_words(
        bytes, candidate, candidate_options, arch, 0, 0, planned_logical_range_count, layout,
        spill_overlaps_guest_operands, spill ? &*spill : nullptr,
        private_layout ? std::optional<uint32_t>(private_layout->epoch_offset) : std::nullopt,
        private_layout ? private_layout->owner_offset : std::nullopt, private_owner_shift,
        candidate_errors, &probe_guest_instruction_offset, &guest_instruction_word_count);
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
    const bool wrap_embedded_guest_vgpr_bank = options.moi_dynamic_access_records &&
                                               arch == ROCJITSU_CODE_ARCH_CDNA5 &&
                                               candidate.gfx1250_vgpr_msb_mode.value_or(0u) != 0u;
    const bool capture_high_bank_address =
        moi_access_requires_gfx1250_high_bank_address_capture(candidate, arch);
    const uint64_t vgpr_msb_bytes =
        moi_appended_body_vgpr_msb_transition_word_count(arch, candidate.gfx1250_vgpr_msb_mode,
                                                         spill.has_value(),
                                                         wrap_embedded_guest_vgpr_bank) *
            sizeof(uint32_t) +
        (capture_high_bank_address ? sizeof(uint32_t) : 0u) +
        (capture_high_bank_address
             ? (spill                                                          ? 2u
                : (candidate.gfx1250_vgpr_msb_mode.value_or(0u) & 0xfcu) != 0u ? 1u
                                                                               : 0u) *
                   sizeof(uint32_t)
             : 0u);
    std::vector<uint32_t> displaced_tail_words;
    uint32_t displaced_original_size = candidate.size();
    bool entry_island_at_anchor = false;
    const uint32_t available_padding =
        count_nop_padding(bytes, candidate.file_offset + candidate.size(), arch);
    const uint64_t available_bytes =
        candidate.size() + static_cast<uint64_t>(available_padding) * sizeof(uint32_t);
    std::optional<DbiPatchPlacement> placement;
    std::optional<uint64_t> entry_island_offset;
    std::optional<MoiBorrowedRecordReplayEntry> borrowed_entry;
    std::optional<BranchOnlyRelayRoute> branch_only_route;
    std::vector<uint64_t> appended_branch_relay_offsets;
    bool uses_reserved_entry_relay = false;
    if (use_indirect_appended) {
      if (branch_only_scalar_spill) {
        const bool assignment_supports_borrowed_entry =
            moi_transient_sgpr_assignment_uses_borrowed_record_replay_entry(
                options, resources->owner_descriptor_file_offsets);
        std::optional<MoiBorrowedRecordReplayEntry> borrowed_entry_prototype;
        if (assignment_supports_borrowed_entry && !original_text.empty()) {
          std::vector<uint32_t> displaced_words =
              borrowed_entry_words_for_candidate(candidate_ptr).value_or(std::vector<uint32_t>{});
          const size_t candidate_words = candidate.size() / sizeof(uint32_t);
          const auto jump_sgprs = moi_indirect_jump_sgprs(candidate_options);
          const auto backup_vgpr = find_moi_borrowed_entry_backup_vgpr(
              resource_state, resources->owner_descriptor_file_offsets, candidate.anchor(),
              resources->required_vgpr_count, candidate_options);
          if (jump_sgprs && backup_vgpr && candidate_words <= displaced_words.size() &&
              displaced_words.size() * sizeof(uint32_t) <= std::numeric_limits<uint32_t>::max()) {
            borrowed_entry_prototype = build_moi_borrowed_record_replay_entry(
                candidate.anchor(), original_text_size, *backup_vgpr, *jump_sgprs,
                static_cast<uint32_t>(displaced_words.size()), arch);
          }
          if (borrowed_entry_prototype) {
            displaced_tail_words.assign(displaced_words.begin() + candidate_words,
                                        displaced_words.end());
            displaced_original_size =
                static_cast<uint32_t>(displaced_words.size() * sizeof(uint32_t));
            entry_island_offset = candidate.anchor();
            entry_island_at_anchor = true;
          }
        }
        const uint64_t borrowed_body_prefix_bytes =
            borrowed_entry_prototype
                ? static_cast<uint64_t>(borrowed_entry_prototype->scalar_restore_words.size() +
                                        1u) *
                      sizeof(uint32_t)
                : 0u;
        const uint64_t patch_bytes =
            static_cast<uint64_t>((words->size() + displaced_tail_words.size()) *
                                  sizeof(uint32_t)) +
            spill_bytes + scalar_spill_bytes + vgpr_msb_bytes + borrowed_body_prefix_bytes;
        const uint64_t reserved_body_bytes =
            patch_bytes + (1u + kBranchOnlyAppendedRelayWords) * sizeof(uint32_t);
        if (!ensure_branch_only_appended_spine(reserved_body_bytes))
          return;
        if (candidate.size() < sizeof(uint32_t) ||
            branch_only_placement_planner.appended_end() > indirect_body_cursor) {
          result.warnings.emplace_back(
              "ConSan MOI first-light branch-only scalar-spill body has invalid geometry");
          continue;
        }
        DbiPatchPlacementPlanner tentative_planner = branch_only_placement_planner;
        std::string relay_error;
        if (indirect_body_cursor > tentative_planner.appended_end() &&
            !tentative_planner.reserve_appended_prefix(
                indirect_body_cursor - tentative_planner.appended_end(), &relay_error)) {
          result.warnings.emplace_back("ConSan MOI first-light probe " + relay_error);
          continue;
        }
        const auto reservation = tentative_planner.plan_indirect_appended(
            candidate.anchor(), displaced_original_size, reserved_body_bytes, &relay_error);
        if (!reservation) {
          result.warnings.emplace_back(
              "ConSan MOI first-light branch-only scalar-spill body could not be placed" +
              (relay_error.empty() ? std::string{} : ": " + relay_error));
          continue;
        }
        std::optional<MoiBorrowedRecordReplayEntry> actual_borrowed_entry;
        if (borrowed_entry_prototype) {
          actual_borrowed_entry = build_moi_borrowed_record_replay_entry(
              candidate.anchor(), reservation->body_offset, borrowed_entry_prototype->backup_vgpr,
              *moi_indirect_jump_sgprs(candidate_options),
              displaced_original_size / sizeof(uint32_t), arch);
        }
        if ((borrowed_entry_prototype && !actual_borrowed_entry) ||
            (!actual_borrowed_entry &&
             !direct_or_forward_relay(candidate.anchor(), reservation->body_offset)) ||
            !direct_or_backward_relay(reservation->body_offset + patch_bytes,
                                      candidate.anchor() + displaced_original_size)) {
          result.warnings.emplace_back(
              "ConSan MOI first-light branch-only scalar-spill body has no first-hop relay");
          continue;
        }
        branch_only_placement_planner = std::move(tentative_planner);
        placement = *reservation;
        placement->body_size = patch_bytes;
        const uint64_t relay_begin =
            placement->body_offset + placement->body_size + sizeof(uint32_t);
        for (uint64_t word = 0u; word < kBranchOnlyAppendedRelayWords; ++word)
          appended_branch_relay_offsets.push_back(relay_begin + word * sizeof(uint32_t));
        borrowed_entry = std::move(actual_borrowed_entry);
        indirect_body_cursor = branch_only_placement_planner.appended_end();
      } else if (use_dense_call) {
        entry_island_offset = dense_entry_islands.at(candidate_ptr);
      } else {
        const size_t candidate_index = candidate_indices.at(candidate_ptr);
        entry_island_offset =
            access_island_begin + candidate_index * access_slot_words * sizeof(uint32_t);
        uses_reserved_entry_relay = true;
      }
      if (!branch_only_scalar_spill && !use_dense_call &&
          (candidate.size() < sizeof(uint32_t) ||
           !compute_sopp_branch_simm16(candidate.anchor(), *entry_island_offset))) {
        entry_island_offset =
            local_entry_islands ? local_entry_islands->claim_reachable(candidate.anchor(),
                                                                       candidate_entry_island_words)
                                : std::nullopt;
        uses_reserved_entry_relay = false;
      }
      if (!branch_only_scalar_spill && !use_dense_call && !entry_island_offset &&
          !original_text.empty()) {
        std::vector<uint32_t> displaced_words;
        while (displaced_words.size() < candidate_entry_island_words) {
          auto decoded = decode_relocatable_entry_instruction(
              original_text, candidate.anchor() + displaced_words.size() * sizeof(uint32_t), arch,
              candidate_errors, resource_state.decoder.get());
          if (!decoded) {
            displaced_words.clear();
            break;
          }
          displaced_words.insert(displaced_words.end(), decoded->begin(), decoded->end());
        }
        const uint64_t displaced_size = displaced_words.size() * sizeof(uint32_t);
        if (displaced_words.size() >= candidate_entry_island_words &&
            displaced_size <= std::numeric_limits<uint32_t>::max() &&
            moi_entry_window_is_single_entry(resource_state.block_index, candidate.anchor(),
                                             displaced_size)) {
          const size_t candidate_words = candidate.size() / sizeof(uint32_t);
          if (candidate_words <= displaced_words.size()) {
            displaced_tail_words.assign(displaced_words.begin() + candidate_words,
                                        displaced_words.end());
            displaced_original_size = static_cast<uint32_t>(displaced_size);
            entry_island_offset = candidate.anchor();
            entry_island_at_anchor = true;
            uses_reserved_entry_relay = false;
          }
        }
      }
      if (!branch_only_scalar_spill &&
          (candidate.size() < sizeof(uint32_t) || !entry_island_offset)) {
        result.warnings.emplace_back(
            "ConSan MOI first-light probe skipped load/store site: entry island is unreachable");
        continue;
      }
      if (!branch_only_scalar_spill && !use_dense_call && entry_island_offset) {
        branch_router.retire_range(*entry_island_offset,
                                   entry_island_at_anchor
                                       ? displaced_original_size
                                       : candidate_entry_island_words * sizeof(uint32_t));
      }
      const bool dense_runtime_workgroup_gate =
          use_dense_call && options.moi_runtime_sample_stride > 1u &&
          !candidate_options.automatic_moi_record_replay_sgpr_spill &&
          has_probe_entry_runtime_workgroup_gate;
      const uint64_t patch_bytes =
          static_cast<uint64_t>((words->size() + displaced_tail_words.size()) * sizeof(uint32_t)) +
          spill_bytes + scalar_spill_bytes + vgpr_msb_bytes +
          (dense_runtime_workgroup_gate ? runtime_gate_island_words * sizeof(uint32_t) : 0u);
      if (!branch_only_scalar_spill) {
        const uint64_t return_words =
            use_dense_call ? (arch != ROCJITSU_CODE_ARCH_CDNA5 ||
                                      candidate_options.automatic_moi_record_replay_sgpr_spill
                                  ? 7u
                                  : 1u)
                           : 6u;
        const uint64_t appended_relay_words =
            has_branch_only_scalar_assignment ? kBranchOnlyAppendedRelayWords : 0u;
        const uint64_t reserved_body_bytes =
            patch_bytes + (return_words + appended_relay_words) * sizeof(uint32_t);
        if (!ensure_branch_only_appended_spine(reserved_body_bytes))
          return;
        placement = DbiPatchPlacement{
            .kind = DbiPatchPlacementKind::AppendedCave,
            .anchor_offset = candidate.anchor(),
            .original_size = displaced_original_size,
            .body_offset = indirect_body_cursor,
            .body_size = patch_bytes,
            .return_branch_offset = 0,
            .return_target = candidate.anchor() + displaced_original_size,
        };
        indirect_body_cursor += patch_bytes + return_words * sizeof(uint32_t);
        if (has_branch_only_scalar_assignment) {
          for (uint64_t word = 0u; word < kBranchOnlyAppendedRelayWords; ++word)
            appended_branch_relay_offsets.push_back(indirect_body_cursor + word * sizeof(uint32_t));
          indirect_body_cursor += kBranchOnlyAppendedRelayWords * sizeof(uint32_t);
        }
      }
    } else {
      const uint64_t patch_bytes = static_cast<uint64_t>(words->size() * sizeof(uint32_t)) +
                                   spill_bytes + scalar_spill_bytes + vgpr_msb_bytes;
      DbiPatchPlacementRequest placement_request;
      placement_request.anchor_offset = candidate.anchor();
      placement_request.original_size = candidate.size();
      placement_request.body_size = patch_bytes;
      placement_request.inline_capacity =
          spill || private_layout || capture_high_bank_address ? 0u : available_bytes;
      std::string placement_error;
      placement = placement_planner.plan(placement_request, &placement_error);
      if (!placement) {
        result.warnings.emplace_back("ConSan MOI first-light probe skipped load/store site: " +
                                     placement_error);
        continue;
      }
    }
    if (capture_high_bank_address && placement->kind != DbiPatchPlacementKind::AppendedCave) {
      result.warnings.emplace_back(
          "ConSan MOI first-light probe skipped a gfx1250 high-bank LDS address without an "
          "appended capture body");
      continue;
    }

    PlannedAccessRecordPatch planned;
    planned.candidate = candidate_ptr;
    planned.placement = *placement;
    planned.entry_island_offset = entry_island_offset;
    planned.entry_island_word_count = borrowed_entry ? borrowed_entry->island_words.size()
                                      : uses_reserved_entry_relay ? access_relay_words
                                                                  : candidate_entry_island_words;
    planned.entry_island_at_anchor = entry_island_at_anchor;
    planned.dense_call_anchor = use_dense_call;
    planned.runtime_workgroup_gate = uses_reserved_entry_relay &&
                                     options.moi_runtime_sample_stride > 1u &&
                                     !candidate_options.automatic_moi_record_replay_sgpr_spill &&
                                     has_probe_entry_runtime_workgroup_gate;
    planned.dense_runtime_workgroup_gate =
        use_dense_call && options.moi_runtime_sample_stride > 1u &&
        !candidate_options.automatic_moi_record_replay_sgpr_spill &&
        has_probe_entry_runtime_workgroup_gate;
    if (planned.runtime_workgroup_gate || planned.dense_runtime_workgroup_gate)
      planned.runtime_workgroup_sources = runtime_workgroup_sources;
    if (use_dense_call)
      planned.dense_dispatcher_offset = dense_dispatchers.at(candidate_ptr);
    planned.displaced_tail_words = std::move(displaced_tail_words);
    planned.record_index = planned_record_count;
    planned.record_count = candidate_record_count;
    planned.logical_range_index = planned_logical_range_count;
    planned.probe_guest_instruction_offset = probe_guest_instruction_offset;
    planned.guest_instruction_word_count = guest_instruction_word_count;
    planned.resources = *resources;
    planned.spill = spill;
    planned.scalar_spill = scalar_spill;
    planned.gfx1250_vgpr_msb_mode = candidate.gfx1250_vgpr_msb_mode;
    if (private_layout) {
      planned.private_epoch_offset = private_layout->epoch_offset;
      planned.private_owner_offset = private_layout->owner_offset;
      planned.private_owner_shift = private_owner_shift;
      planned.private_record_replay_workgroup_offsets =
          private_layout->record_replay_workgroup_offsets;
      planned.persistent_private_state_end = private_layout->ephemeral_base;
    }
    planned.required_private_bytes = required_private_bytes;
    planned.spill_overlaps_guest_operands = spill_overlaps_guest_operands;
    planned.wrap_embedded_guest_vgpr_bank = wrap_embedded_guest_vgpr_bank;
    planned.branch_only_scalar_spill = branch_only_scalar_spill;
    planned.appended_branch_relay_offsets = std::move(appended_branch_relay_offsets);
    planned.borrowed_entry = std::move(borrowed_entry);
    planned.branch_only_route = std::move(branch_only_route);
    planned_patches.push_back(std::move(planned));
    if (!planned_patches.back().appended_branch_relay_offsets.empty()) {
      generated_branch_relay_ranges.emplace_back(
          planned_patches.back().appended_branch_relay_offsets.front(),
          planned_patches.back().appended_branch_relay_offsets.back() + sizeof(uint32_t));
    }
    for (uint64_t relay : planned_patches.back().appended_branch_relay_offsets) {
      if (!branch_router.offer(relay, BranchOnlyRelayProvenance::GeneratedBank)) {
        result.errors.emplace_back(
            "ConSan MOI first-light probe could not publish an appended branch-only relay");
        return;
      }
      branch_relay_offsets.insert(relay);
    }
    const bool owns_appended_anchor_tail = use_indirect_appended && !use_dense_call &&
                                           !entry_island_at_anchor &&
                                           placement->kind == DbiPatchPlacementKind::AppendedCave;
    if (owns_appended_anchor_tail) {
      publish_record_replay_anchor_relays(candidate.anchor(), placement->original_size,
                                          selected_record_replay_anchor_relay_offsets);
    }
    if (use_indirect_appended && !use_dense_call && !branch_only_scalar_spill)
      planned_anchor_end = candidate.anchor() + placement->original_size;
    planned_record_count += candidate_record_count;
    planned_logical_range_count += static_cast<uint32_t>(access_ranges.size());
    if (planned_patches.size() == selected_candidate_limit)
      break;
  }

  // Route branch-only accesses together after all bodies have fixed positions.
  // Large E2E objects can contain thousands of such sites and millions of
  // possible relays. One bounded router call qualifies and indexes that shared
  // inventory once; per-site calls (or small batches) repeatedly rebuild it.
  std::vector<size_t> branch_only_patch_indices;
  for (size_t index = 0; index < planned_patches.size(); ++index) {
    const PlannedAccessRecordPatch &patch = planned_patches[index];
    if (patch.branch_only_scalar_spill)
      branch_only_patch_indices.push_back(index);
  }
  if (local_entry_islands)
    pristine_branch_relay_offers = local_entry_islands->available_word_offsets();
  size_t branch_relay_offer_count = branch_router.available_count();
  if (!branch_only_patch_indices.empty()) {
    for (uint64_t relay : pristine_branch_relay_offers)
      (void)branch_router.offer(relay, BranchOnlyRelayProvenance::PristineNop);
    // The generated bank was seeded before direct-reservoir discovery.
    // Retain that router state: re-offering its full source inventory here
    // would resurrect relays already consumed by routed reservoirs and allow
    // two emitted continuations to overwrite the same capacity-one word.
    for (uint64_t relay : selected_record_replay_anchor_relay_offsets)
      (void)branch_router.offer(relay, BranchOnlyRelayProvenance::OwnedAnchor);
    branch_relay_offer_count = branch_router.available_count();
  }
  // A very large simultaneous exact solve is not useful for this monotonic
  // corridor. Fall through promptly to the pair-atomic greedy tier, but size
  // its one-time ordered inventory budget to the actual offered domain.
  const BranchOnlyRelaySearchLimits relay_limits =
      branch_only_relay_greedy_pair_limits(branch_relay_offer_count);
  for (size_t batch_begin = 0; batch_begin < branch_only_patch_indices.size();) {
    const size_t batch_end = branch_only_patch_indices.size();
    std::vector<BranchOnlyRelayPairRequest> requests;
    requests.reserve(batch_end - batch_begin);
    for (size_t position = batch_begin; position < batch_end; ++position) {
      const PlannedAccessRecordPatch &patch = planned_patches[branch_only_patch_indices[position]];
      requests.push_back({
          .entry_source = patch.candidate->anchor(),
          .entry_target = patch.borrowed_entry ? patch.candidate->anchor() + sizeof(uint32_t)
                                               : patch.placement.body_offset,
          .return_source = patch.placement.body_offset + patch.placement.body_size,
          .return_target = patch.candidate->anchor() + patch.placement.original_size,
          .entry_preplaced = patch.borrowed_entry.has_value(),
      });
    }

    DbiPatchPlacementPlanner tentative_planner = branch_only_placement_planner;
    BranchOnlyRelayRouter tentative_router = branch_router;
    std::string relay_error;
    BranchOnlyRelayBatchPlan route_plan =
        tentative_router.plan_pairs(tentative_planner, requests, &relay_error, relay_limits);
    if (!route_plan.complete() || route_plan.routes.size() != requests.size() ||
        !tentative_router.commit(route_plan.routes, &relay_error)) {
      result.warnings.emplace_back(
          "ConSan MOI first-light branch-only scalar-spill route batch was rejected" +
          (relay_error.empty() ? std::string{} : ": " + relay_error));
      batch_begin = batch_end;
      continue;
    }

    for (const BranchOnlyRelayRoute &route : route_plan.routes) {
      for (const BranchOnlyRelayClaim &claim : route.claims) {
        if (local_entry_islands && claim.provenance == BranchOnlyRelayProvenance::PristineNop)
          local_entry_islands->reserve(claim.offset, 1u);
      }
      if (!direct_branch_reservoirs.mark_claims_used(route.claims, &relay_error)) {
        result.errors.emplace_back("ConSan MOI first-light probe " + relay_error);
        return;
      }
    }
    for (size_t route_index = 0; route_index < route_plan.routes.size(); ++route_index) {
      planned_patches[branch_only_patch_indices[batch_begin + route_index]].branch_only_route =
          std::move(route_plan.routes[route_index]);
    }
    branch_only_placement_planner = std::move(tentative_planner);
    branch_router = std::move(tentative_router);
    batch_begin = batch_end;
  }
  std::erase_if(planned_patches, [&](const PlannedAccessRecordPatch &patch) {
    if (!patch.branch_only_scalar_spill || patch.branch_only_route)
      return false;
    result.warnings.emplace_back(
        "ConSan MOI first-light branch-only scalar-spill body is out of routed reach");
    return true;
  });

  // Reservoir discovery is intentionally performed before final access
  // selection so a branch-only candidate can establish a complete-text
  // routing frontier even when that candidate is later omitted. The sync
  // passes are independent downstream consumers of that frontier. Preserve
  // every planned reservoir when synchronization is enabled; otherwise an
  // access batch that happens not to claim one can strand a later pass.
  if (moi_reserved_record_replay_sync_island_count(result, options) != 0u) {
    for (BranchOnlyDirectRelayReservoir &reservoir : direct_branch_reservoirs.reservoirs)
      reservoir.used = true;
  }

  std::unordered_map<uint64_t, size_t> record_replay_anchor_relay_donors;
  for (size_t index = 0; index < planned_patches.size(); ++index) {
    const PlannedAccessRecordPatch &planned = planned_patches[index];
    for (uint64_t offset = planned.candidate->anchor() + sizeof(uint32_t);
         offset < planned.candidate->anchor() + planned.placement.original_size;
         offset += sizeof(uint32_t)) {
      if (selected_record_replay_anchor_relay_offsets.contains(offset))
        record_replay_anchor_relay_donors.emplace(offset, index);
    }
  }
  for (size_t index = 0; index < planned_patches.size(); ++index) {
    const PlannedAccessRecordPatch &planned = planned_patches[index];
    const auto donor_is_emitted_first = [&](uint64_t relay) {
      if (!selected_record_replay_anchor_relay_offsets.contains(relay))
        return true;
      const auto donor = record_replay_anchor_relay_donors.find(relay);
      return donor != record_replay_anchor_relay_donors.end() && donor->second <= index;
    };
    if (planned.branch_only_route &&
        (!std::ranges::all_of(planned.branch_only_route->entry_relay_offsets,
                              donor_is_emitted_first) ||
         !std::ranges::all_of(planned.branch_only_route->return_relay_offsets,
                              donor_is_emitted_first))) {
      result.errors.emplace_back(
          "ConSan MOI first-light probe planned an anchor-tail relay before its donor");
      return;
    }
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
    candidate_options.moi_record_replay_workgroup_private_offsets =
        planned_patch.private_record_replay_workgroup_offsets;
    if (!apply_moi_transient_sgpr_assignment(candidate_options,
                                             planned_patch.resources.owner_descriptor_file_offsets))
      continue;
    if (!apply_record_replay_entry_workgroup_assignment(
            candidate_options, planned_patch.resources.owner_descriptor_file_offsets))
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
    for (const BranchOnlyDirectRelayReservoir &reservoir : direct_branch_reservoirs.reservoirs) {
      std::string relay_error;
      if (!BranchOnlyRelayRouter::emit_direct_reservoir(new_text, reservoir, arch, patches,
                                                        &relay_error)) {
        result.errors.emplace_back("ConSan MOI first-light probe " + relay_error);
        return;
      }
    }
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
      if (!emit_moi_dense_access_group(std::span<const PlannedAccessRecordPatch *const>(group),
                                       dispatcher_offset, dense_entry_hosts, group_options, arch,
                                       access_slot_words,
                                       /*collapse_spill_router_for_explicit_key=*/false,
                                       "first-light probe", new_text, patches, result.errors)) {
        return;
      }
    }
    for (const PlannedAccessRecordPatch &planned_patch : planned_patches) {
      const ConSanMoiCandidate &candidate = *planned_patch.candidate;
      ConSanOptions candidate_options = options;
      candidate_options.scratch_vgpr = planned_patch.resources.base;
      candidate_options.moi_record_replay_workgroup_private_offsets =
          planned_patch.private_record_replay_workgroup_offsets;
      if (!apply_moi_transient_sgpr_assignment(
              candidate_options, planned_patch.resources.owner_descriptor_file_offsets) ||
          !apply_record_replay_entry_workgroup_assignment(
              candidate_options, planned_patch.resources.owner_descriptor_file_offsets) ||
          !candidate_options.moi_exec_save_sgpr) {
        result.errors.emplace_back(
            "ConSan MOI first-light probe lost its owner-local register assignment");
        return;
      }
      auto words = build_first_light_access_record_words(
          bytes, candidate, candidate_options, arch, planned_patch.record_index,
          planned_record_count, planned_patch.logical_range_index, layout,
          planned_patch.spill_overlaps_guest_operands,
          planned_patch.spill ? &*planned_patch.spill : nullptr, planned_patch.private_epoch_offset,
          planned_patch.private_owner_offset, planned_patch.private_owner_shift, result.errors);
      if (!words)
        return;

      ConSanPatchInfo info;
      info.kind = planned_patch.placement.kind == DbiPatchPlacementKind::AppendedCave
                      ? ConSanPatchKind::TrampolineMoiAccessRecordStore
                      : ConSanPatchKind::InlineMoiAccessRecordStore;
      info.anchor_offset = candidate.anchor();
      info.scratch_vgpr = planned_patch.resources.base;
      info.owner_descriptor_file_offsets = planned_patch.resources.owner_descriptor_file_offsets;
      info.persistent_epoch_private_offset = planned_patch.private_epoch_offset;
      info.persistent_owner_private_offset = planned_patch.private_owner_offset;
      info.persistent_record_replay_workgroup_private_offsets =
          planned_patch.private_record_replay_workgroup_offsets;
      if (planned_patch.private_epoch_offset)
        info.persistent_private_state_end = planned_patch.persistent_private_state_end;
      info.required_private_segment_size = planned_patch.required_private_bytes;
      if (planned_patch.spill) {
        info.spilled_vgpr_count = planned_patch.spill->vgpr_count;
        note_dynamic_stack_private_requirement(info, &*planned_patch.spill);
      }
      info.branch_only_continuation = planned_patch.branch_only_route.has_value();
      info.branch_only_borrowed_indirect_entry = planned_patch.borrowed_entry.has_value();
      if (planned_patch.borrowed_entry) {
        const MoiIndirectJumpSgprs jump_sgprs = *moi_indirect_jump_sgprs(candidate_options);
        info.indirect_pc_sgpr = jump_sgprs.pc_sgpr;
        info.indirect_saved_scc_sgpr = jump_sgprs.scc_save_sgpr;
        info.branch_only_borrowed_backup_vgpr = planned_patch.borrowed_entry->backup_vgpr;
      }
      if (planned_patch.branch_only_route) {
        info.branch_only_entry_relay_offsets = planned_patch.branch_only_route->entry_relay_offsets;
        info.branch_only_return_relay_offsets =
            planned_patch.branch_only_route->return_relay_offsets;
      }

      if (planned_patch.placement.kind == DbiPatchPlacementKind::AppendedCave) {
        const uint64_t cave_text_offset = planned_patch.placement.body_offset;
        while (new_text.size() < cave_text_offset)
          append_word_bytes(new_text, build_s_nop(0u, arch));
        if (new_text.size() != cave_text_offset) {
          result.errors.emplace_back(
              "ConSan MOI first-light probe emitted a stale appended-cave mapping (expected " +
              std::to_string(cave_text_offset) + ", got " + std::to_string(new_text.size()) + ")");
          return;
        }
        const uint64_t entry_target =
            planned_patch.branch_only_route &&
                    !planned_patch.branch_only_route->entry_relay_offsets.empty()
                ? planned_patch.branch_only_route->entry_relay_offsets.front()
                : planned_patch.entry_island_offset.value_or(planned_patch.placement.body_offset);
        const auto ret =
            planned_patch.entry_island_offset || planned_patch.branch_only_scalar_spill
                ? std::optional<int16_t>()
                : compute_sopp_branch_simm16(planned_patch.placement.return_branch_offset,
                                             planned_patch.placement.return_target);
        if ((!planned_patch.dense_call_anchor &&
             !write_moi_appended_anchor(new_text, planned_patch, entry_target, arch,
                                        "first-light probe", result.errors)) ||
            (!planned_patch.entry_island_offset && !planned_patch.branch_only_scalar_spill &&
             !ret)) {
          result.errors.emplace_back(
              "ConSan MOI first-light probe appended cave branch is out of range");
          return;
        }

        if (planned_patch.entry_island_offset && !planned_patch.dense_call_anchor) {
          const MoiSpecialStateSgprs special_state = *moi_special_state_sgprs(candidate_options);
          std::vector<uint32_t> island_words;
          if (planned_patch.borrowed_entry) {
            island_words = planned_patch.borrowed_entry->island_words;
          } else if (planned_patch.runtime_workgroup_gate) {
            if (!planned_patch.runtime_workgroup_sources) {
              result.errors.emplace_back(
                  "ConSan MOI first-light probe lost its runtime workgroup sources");
              return;
            }
            auto gate = build_moi_runtime_workgroup_gate_island_words(
                bytes, candidate, candidate_options, *planned_patch.runtime_workgroup_sources,
                planned_patch.record_index, *planned_patch.entry_island_offset, cave_text_offset,
                planned_patch.placement.return_target, planned_patch.entry_island_word_count, arch);
            if (gate)
              island_words = std::move(*gate);
          } else if (append_moi_scc_preserving_indirect_jump(
                         island_words, *planned_patch.entry_island_offset, cave_text_offset,
                         moi_indirect_jump_sgprs(candidate_options)->pc_sgpr,
                         special_state.scc_save_sgpr,
                         /*capture_scc=*/true, arch)) {
            island_words.resize(planned_patch.entry_island_word_count, build_s_nop(0, arch));
          }
          if (island_words.size() != planned_patch.entry_island_word_count) {
            result.errors.emplace_back(
                "ConSan MOI first-light probe could not encode its entry island");
            return;
          }
          if (!write_moi_entry_island(new_text, planned_patch, island_words, patches,
                                      "first-light probe", result.errors))
            return;
        }

        std::vector<uint32_t> dense_gate_words;
        if (planned_patch.dense_runtime_workgroup_gate) {
          if (!planned_patch.runtime_workgroup_sources) {
            result.errors.emplace_back(
                "ConSan MOI first-light probe lost its dense runtime workgroup sources");
            return;
          }
          auto gate = build_moi_runtime_workgroup_gate_call_words(
              bytes.subspan(candidate.file_offset, candidate.size()), candidate_options,
              *planned_patch.runtime_workgroup_sources, cave_text_offset,
              planned_patch.placement.return_target, runtime_gate_island_words, arch);
          if (!gate) {
            const auto describe_source = [](const ConSanMoiWorkgroupSource &source) {
              if (source.scalar_src)
                return std::string("s") + std::to_string(*source.scalar_src);
              if (source.vector_src)
                return std::string("v") + std::to_string(*source.vector_src);
              if (source.private_offset)
                return std::string("private+") + std::to_string(*source.private_offset);
              return std::string("absent");
            };
            result.errors.emplace_back(
                "ConSan MOI first-light probe could not encode its dense runtime gate at " +
                std::to_string(candidate.anchor()) + " with exec-save s" +
                std::to_string(*candidate_options.moi_exec_save_sgpr) + " and workgroup sources " +
                describe_source(planned_patch.runtime_workgroup_sources->x) + "," +
                describe_source(planned_patch.runtime_workgroup_sources->y) + "," +
                describe_source(planned_patch.runtime_workgroup_sources->z) + "," +
                describe_source(planned_patch.runtime_workgroup_sources->cluster_workgroup_id));
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
        if (moi_access_requires_gfx1250_high_bank_address_capture(candidate, arch)) {
          const uint16_t base_scratch_count =
              record_replay_uses_automatic_banked_capture(
                  candidate_options, layout.record_replay_dispatch_token_capacity)
                  ? 10u
                  : 6u;
          body_options.gfx1250_high_bank_address_source = candidate.operands.address_vgpr;
          body_options.gfx1250_high_bank_address_destination =
              static_cast<uint16_t>(planned_patch.resources.base + base_scratch_count);
        }
        body_options.entry_prefix_words = dense_gate_words;
        body_options.trailing_guest_word_count =
            options.moi_dynamic_access_records ? 0u : planned_patch.guest_instruction_word_count;
        body_options.probe_guest_instruction_offset = planned_patch.probe_guest_instruction_offset;
        body_options.guest_instruction_word_count = planned_patch.guest_instruction_word_count;
        body_options.body_guest_instruction_offset = &body_guest_instruction_offset;
        body_options.restore_vgpr_spill_before_trailing_guest =
            planned_patch.spill_overlaps_guest_operands;
        body_options.wrap_embedded_guest_vgpr_bank = planned_patch.wrap_embedded_guest_vgpr_bank;
        std::vector<uint32_t> borrowed_entry_prefix;
        std::optional<size_t> borrowed_empty_exec_guard_word;
        if (planned_patch.borrowed_entry) {
          borrowed_entry_prefix = planned_patch.borrowed_entry->scalar_restore_words;
          borrowed_empty_exec_guard_word = borrowed_entry_prefix.size();
          borrowed_entry_prefix.push_back(build_s_nop(0u, arch));
          body_options.entry_prefix_words = borrowed_entry_prefix;
        }
        auto cave_words = assemble_moi_appended_body(planned_patch, *words, "first-light probe",
                                                     result.errors, body_options);
        if (!cave_words)
          return;
        if (borrowed_empty_exec_guard_word) {
          const size_t displaced_tail_begin =
              cave_words->size() - planned_patch.displaced_tail_words.size();
          if (displaced_tail_begin <= *borrowed_empty_exec_guard_word ||
              displaced_tail_begin - *borrowed_empty_exec_guard_word - 1u >
                  static_cast<size_t>(std::numeric_limits<int16_t>::max())) {
            result.errors.emplace_back(
                "ConSan MOI first-light borrowed-entry empty-EXEC guard is out of range");
            return;
          }
          const auto guard = instrumentation::build_s_cbranch_execz(
              static_cast<int16_t>(displaced_tail_begin - *borrowed_empty_exec_guard_word - 1u),
              arch);
          if (!guard) {
            result.errors.emplace_back(
                "ConSan MOI first-light could not encode its borrowed-entry empty-EXEC guard");
            return;
          }
          (*cave_words)[*borrowed_empty_exec_guard_word] = *guard;
          info.branch_only_borrowed_continuation_offset =
              cave_text_offset + displaced_tail_begin * sizeof(uint32_t);
        }
        if (planned_patch.dense_call_anchor) {
          if (arch != ROCJITSU_CODE_ARCH_CDNA5 ||
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
        } else if (planned_patch.borrowed_entry) {
          const uint64_t return_target =
              planned_patch.branch_only_route &&
                      !planned_patch.branch_only_route->return_relay_offsets.empty()
                  ? planned_patch.branch_only_route->return_relay_offsets.front()
                  : planned_patch.placement.return_target;
          const auto direct_return = compute_sopp_branch_simm16(
              cave_text_offset + cave_words->size() * sizeof(uint32_t), return_target);
          if (!direct_return) {
            result.errors.emplace_back(
                "ConSan MOI first-light borrowed-entry return is unreachable");
            return;
          }
          cave_words->push_back(build_s_branch(*direct_return, arch));
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
        } else if (planned_patch.branch_only_scalar_spill) {
          const uint64_t return_target =
              planned_patch.branch_only_route &&
                      !planned_patch.branch_only_route->return_relay_offsets.empty()
                  ? planned_patch.branch_only_route->return_relay_offsets.front()
                  : planned_patch.placement.return_target;
          const auto direct_return = compute_sopp_branch_simm16(
              cave_text_offset + cave_words->size() * sizeof(uint32_t), return_target);
          if (!direct_return) {
            result.errors.emplace_back(
                "ConSan MOI first-light branch-only scalar-spill return is unreachable");
            return;
          }
          cave_words->push_back(build_s_branch(*direct_return, arch));
        } else {
          cave_words->push_back(build_s_branch(*ret, arch));
        }
        append_words_bytes(new_text, *cave_words);
        for (uint64_t relay : planned_patch.appended_branch_relay_offsets) {
          (void)relay;
          append_word_bytes(new_text, build_s_nop(0u, arch));
        }
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
        if (candidate.anchor() > new_text.size() ||
            patch_bytes > new_text.size() - candidate.anchor()) {
          result.errors.emplace_back("ConSan MOI first-light probe exceeds .text");
          return;
        }
        std::memcpy(new_text.data() + candidate.anchor(), words->data(),
                    static_cast<size_t>(patch_bytes));
        info.trampoline_offset = candidate.anchor() + candidate.size();
        info.original_size = static_cast<uint32_t>(patch_bytes);
        info.trampoline_size = 0;
        info.relocated_guest_instruction_offset =
            candidate.anchor() + planned_patch.probe_guest_instruction_offset;
      }
      for (uint64_t offset = info.anchor_offset + sizeof(uint32_t);
           offset < info.anchor_offset + info.original_size; offset += sizeof(uint32_t)) {
        if (selected_record_replay_anchor_relay_offsets.contains(offset))
          info.branch_relay_offsets.push_back(offset);
      }
      patches.push_back(info);
      if (planned_patch.branch_only_route) {
        std::string relay_error;
        if (!BranchOnlyRelayRouter::emit_and_record(
                new_text, *planned_patch.branch_only_route, info.trampoline_offset,
                planned_patch.placement.return_target, arch, patches, &relay_error)) {
          result.errors.emplace_back("ConSan MOI first-light probe " + relay_error);
          return;
        }
      }
    }

    for (const auto &[dispatcher, host] : dense_entry_hosts) {
      while (new_text.size() < host.body_offset)
        append_word_bytes(new_text, build_s_nop(0u, arch));
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

    if (!replace_consan_text(patcher, new_text, options, "MOI first-light probe", result)) {
      return;
    }
    result.replacement = std::move(patcher).emit();
    result.patches.insert(result.patches.end(), patches.begin(), patches.end());
    if (reserved_sync_island_count != 0u) {
      access_output.reserved_sync_islands = MoiReservedBarrierIslandLayout{
          .begin = original_text_size,
          .slot_words = static_cast<uint32_t>(reserved_sync_slot_words),
          .reserved_slot_count = reserved_sync_island_count,
      };
    }
    access_output.generated_branch_relay_ranges = std::move(generated_branch_relay_ranges);
    result.mark_modified();
    return;
  }

  if (!apply_inline_moi_access_patches(
          bytes, planned_patches, descriptor_requirements, scalar_requirements, "first-light probe",
          arch,
          [&](const PlannedAccessRecordPatch &planned_patch) {
            ConSanOptions candidate_options = options;
            candidate_options.scratch_vgpr = planned_patch.resources.base;
            candidate_options.moi_record_replay_workgroup_private_offsets =
                planned_patch.private_record_replay_workgroup_offsets;
            if (!apply_moi_transient_sgpr_assignment(
                    candidate_options, planned_patch.resources.owner_descriptor_file_offsets) ||
                !apply_record_replay_entry_workgroup_assignment(
                    candidate_options, planned_patch.resources.owner_descriptor_file_offsets))
              return std::optional<std::vector<uint32_t>>{};
            return build_first_light_access_record_words(
                bytes, *planned_patch.candidate, candidate_options, arch,
                planned_patch.record_index, planned_record_count, planned_patch.logical_range_index,
                layout, planned_patch.spill_overlaps_guest_operands,
                planned_patch.spill ? &*planned_patch.spill : nullptr,
                planned_patch.private_epoch_offset, planned_patch.private_owner_offset,
                planned_patch.private_owner_shift, result.errors);
          },
          [](const PlannedAccessRecordPatch &planned_patch) {
            const ConSanMoiCandidate &candidate = *planned_patch.candidate;
            ConSanPatchInfo info;
            info.kind = ConSanPatchKind::InlineMoiAccessRecordStore;
            info.anchor_offset = candidate.anchor();
            info.trampoline_offset = candidate.anchor() + candidate.size();
            info.original_size = static_cast<uint32_t>(planned_patch.placement.body_size);
            info.scratch_vgpr = planned_patch.resources.base;
            info.owner_descriptor_file_offsets =
                planned_patch.resources.owner_descriptor_file_offsets;
            info.persistent_epoch_private_offset = planned_patch.private_epoch_offset;
            info.persistent_owner_private_offset = planned_patch.private_owner_offset;
            info.persistent_record_replay_workgroup_private_offsets =
                planned_patch.private_record_replay_workgroup_offsets;
            if (planned_patch.private_epoch_offset)
              info.persistent_private_state_end = planned_patch.persistent_private_state_end;
            info.required_private_segment_size = planned_patch.required_private_bytes;
            info.relocated_guest_instruction_offset =
                candidate.anchor() + planned_patch.probe_guest_instruction_offset;
            if (planned_patch.spill) {
              info.spilled_vgpr_count = planned_patch.spill->vgpr_count;
              note_dynamic_stack_private_requirement(info, &*planned_patch.spill);
            }
            return info;
          },
          [](std::vector<uint8_t> &, ConSanTransformArtifacts &) { return true; }, result))
    return;
}
