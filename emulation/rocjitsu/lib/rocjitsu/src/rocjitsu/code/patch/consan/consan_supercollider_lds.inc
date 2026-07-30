void try_apply_lds_load_check_trap_patch(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                                         const ConSanOptions &options, ConSanResult &result) {
  if (!instrumentation::is_admitted_arch(arch)) {
    result.warnings.emplace_back("ConSan LDS check/trap proof has unsupported architecture");
    return;
  }

  const auto delay_words =
      delay_instruction_word_count(options, result.errors, "ConSan LDS check/trap proof");
  if (!delay_words)
    return;

  struct CanonicalLdsSite {
    std::string_view container_name;
    uint64_t container_entry_text_offset = 0;
    ConSanLdsSite site;
  };
  size_t discovered_site_count = 0;
  for (const ConSanKernelInfo &kernel : result.kernels)
    discovered_site_count += kernel.lds_sites.size();
  for (const ConSanFunctionInfo &function : result.functions)
    discovered_site_count += function.lds_sites.size();
  auto canonicalizer = consan_detail::make_physical_site_alias_canonicalizer<CanonicalLdsSite>(
      result.errors, "ConSan LDS physical site",
      [](const CanonicalLdsSite &candidate) { return candidate.site.file_offset; },
      [](const CanonicalLdsSite &candidate) { return candidate.container_name; },
      [&](const CanonicalLdsSite &lhs, const CanonicalLdsSite &rhs) {
        return lhs.container_entry_text_offset == rhs.container_entry_text_offset &&
               sc_lds_site_semantics_equal(lhs.site, rhs.site);
      },
      [](CanonicalLdsSite &retained, const CanonicalLdsSite &alias) {
        // Plan and patch one physical instruction, but prove resource
        // safety and descriptor growth for every aliasing dispatch.
        retained.site.owner_descriptor_file_offsets.insert(
            retained.site.owner_descriptor_file_offsets.end(),
            alias.site.owner_descriptor_file_offsets.begin(),
            alias.site.owner_descriptor_file_offsets.end());
      },
      discovered_site_count);
  for (const ConSanKernelInfo &kernel : result.kernels) {
    for (const ConSanLdsSite &site : kernel.lds_sites) {
      CanonicalLdsSite candidate{
          .container_name = kernel.name,
          .container_entry_text_offset = kernel.entry_text_offset,
          .site = site,
      };
      candidate.site.owner_descriptor_file_offsets.push_back(kernel.descriptor_file_offset);
      if (!canonicalizer.insert(std::move(candidate)))
        return;
    }
  }
  for (const ConSanFunctionInfo &function : result.functions) {
    for (const ConSanLdsSite &site : function.lds_sites) {
      if (!canonicalizer.insert({
              .container_name = function.name,
              .container_entry_text_offset = function.entry_text_offset,
              .site = site,
          }))
        return;
    }
  }
  std::optional<std::vector<CanonicalLdsSite>> canonical_sites_result =
      std::move(canonicalizer).take();
  if (!canonical_sites_result)
    return;
  std::vector<CanonicalLdsSite> canonical_sites = std::move(*canonical_sites_result);
  for (CanonicalLdsSite &canonical : canonical_sites) {
    auto &owners = canonical.site.owner_descriptor_file_offsets;
    std::ranges::sort(owners);
    owners.erase(std::ranges::unique(owners).begin(), owners.end());
  }

  std::unique_ptr<Decoder> decoder = Decoder::create(arch);
  const bool has_plannable_supported_site = [&] {
    for (const ConSanKernelInfo &kernel : result.kernels) {
      if (kernel.preflight_action == ConSanPreflightAction::Reject)
        continue;
      if (std::ranges::any_of(kernel.lds_sites, [&](const auto &site) {
            return is_supported_check_trap_site(site, arch);
          }))
        return true;
    }
    for (const ConSanFunctionInfo &function : result.functions) {
      for (const ConSanLdsSite &site : function.lds_sites) {
        if (!is_supported_check_trap_site(site, arch) || site.owner_descriptor_file_offsets.empty())
          continue;
        const bool every_owner_is_plannable =
            std::ranges::all_of(site.owner_descriptor_file_offsets, [&](uint64_t owner) {
              const auto kernel = std::ranges::find(result.kernels, owner,
                                                    &ConSanKernelInfo::descriptor_file_offset);
              return kernel != result.kernels.end() &&
                     kernel->preflight_action != ConSanPreflightAction::Reject;
            });
        if (every_owner_is_plannable)
          return true;
      }
    }
    return false;
  }();
  if (!has_plannable_supported_site) {
    publish_sc_access_coverage_support(result, ConSanScAccessCoverageKind::NativeLds,
                                       [](uint64_t) { return false; });
    result.warnings.emplace_back("ConSan LDS check/trap proof found no supported "
                                 "native-LDS load or store site with complete owners");
    return;
  }

  const auto original_bytes = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(code_object.image_data()), code_object.image_size());
  uint64_t appended_cave_offset = 0;
  std::span<const uint8_t> pristine_text;
  if (code_object.text_sections().size() == 1) {
    appended_cave_offset = code_object.text_sections().front()->size();
    pristine_text = std::span<const uint8_t>(
        reinterpret_cast<const uint8_t *>(code_object.text_sections().front()->data()),
        code_object.text_sections().front()->size());
  }
  std::vector<LocalNopCave> local_nop_caves = find_uncovered_nop_caves(code_object, result, arch);
  std::erase_if(local_nop_caves, [&](const LocalNopCave &cave) {
    return overlaps_preapplied_reserved_range(
        options, {cave.text_offset,
                  cave.text_offset + static_cast<uint64_t>(cave.word_count) * sizeof(uint32_t)});
  });
  uint32_t max_uncovered_nop_cave_words = 0;
  for (const LocalNopCave &cave : local_nop_caves)
    max_uncovered_nop_cave_words = std::max(max_uncovered_nop_cave_words, cave.word_count);

  // A full check body consumes far more padding than its seven-word absolute
  // entry jump. Large generated objects commonly have enough aggregate local
  // padding for every site's entry island, but spread it across a smaller
  // number of long caves. Treat each non-overlapping seven-word slice as an
  // independently reservable island instead of wasting the remainder after
  // the first site claims a cave.
  size_t local_entry_island_capacity = 0;
  for (const LocalNopCave &cave : local_nop_caves)
    local_entry_island_capacity += cave.word_count / kLdsCheckTrapIndirectJumpWords;
  std::vector<LocalNopCave> local_entry_islands;
  local_entry_islands.reserve(local_entry_island_capacity);
  for (const LocalNopCave &cave : local_nop_caves) {
    const uint32_t island_count = cave.word_count / kLdsCheckTrapIndirectJumpWords;
    for (uint32_t island_index = 0; island_index < island_count; ++island_index) {
      const uint64_t word_offset =
          static_cast<uint64_t>(island_index) * kLdsCheckTrapIndirectJumpWords;
      local_entry_islands.push_back(LocalNopCave{
          .text_offset = cave.text_offset + word_offset * sizeof(uint32_t),
          .file_offset = cave.file_offset + word_offset * sizeof(uint32_t),
          .word_count = static_cast<uint32_t>(kLdsCheckTrapIndirectJumpWords),
          .owner_entry_text_offset = cave.owner_entry_text_offset,
      });
    }
  }

  std::vector<std::unique_ptr<BasicBlock>> blocks;
  std::vector<BasicBlock *> block_ptrs;
  struct ScReservoirOwnerContext {
    uint64_t descriptor_file_offset = 0;
    KernelCfgScope scope;
    std::unique_ptr<LivenessAnalysis> liveness;
    uint32_t wavefront_size = 0;
  };
  std::vector<ScReservoirOwnerContext> reservoir_owner_contexts;
  if (decoder) {
    // Decoding only preflight candidates is a scalability fallback for truly
    // large generated libraries.  Smaller bundled libraries still need the
    // surrounding functions as potential relay donors; excluding them made a
    // 1.8 MiB PyTorch helper lose all 27 otherwise-placeable sites.  Keep the
    // complete CFG below this bound and retain the bounded path for the 23 MiB
    // and 40 MiB top-k objects that motivated it.
    constexpr uint64_t kScBoundedCfgMinimumImageBytes = 4u * 1024u * 1024u;
    const bool restrict_to_preflight_candidates =
        result.functions.empty() && options.flavor == ConSanFlavor::SuperCollider &&
        !options.fault_dry_run && options.sc_perturb_kind == ConSanPerturbationKind::None &&
        code_object.image_size() >= kScBoundedCfgMinimumImageBytes;
    std::vector<uint64_t> extra_leaders;
    if (restrict_to_preflight_candidates) {
      for (const ConSanKernelInfo &kernel : result.kernels) {
        if (kernel.has_text_range && kernel.preflight_action == ConSanPreflightAction::Candidate)
          extra_leaders.push_back(kernel.entry_text_offset);
      }
    } else {
      extra_leaders = kernel_entry_offsets(result);
    }
    extra_leaders.reserve(extra_leaders.size() + result.functions.size());
    for (const ConSanFunctionInfo &function : result.functions)
      extra_leaders.push_back(function.entry_text_offset);
    std::ranges::sort(extra_leaders);
    extra_leaders.erase(std::ranges::unique(extra_leaders).begin(), extra_leaders.end());

    std::vector<BasicBlock::CodeRange> code_ranges;
    if (restrict_to_preflight_candidates) {
      code_ranges.reserve(extra_leaders.size());
      for (const ConSanKernelInfo &kernel : result.kernels) {
        if (kernel.has_text_range && kernel.code_size != 0 &&
            kernel.preflight_action == ConSanPreflightAction::Candidate) {
          code_ranges.push_back(
              {.start_offset = kernel.entry_text_offset, .size = kernel.code_size});
        }
      }
    } else {
      code_ranges.reserve(code_object.functions().size());
      for (const AmdGpuFunctionInfo &function : code_object.functions()) {
        if (function.code_size != 0)
          code_ranges.push_back(
              {.start_offset = function.entry_text_offset, .size = function.code_size});
      }
    }
    blocks = BasicBlock::build(code_object, *decoder, arch, extra_leaders, code_ranges);
    block_ptrs = block_ptrs_for(blocks);

    const BlockOffsetIndex block_index = build_block_offset_index(blocks);
    std::vector<uint64_t> kernel_entries;
    for (const ConSanKernelInfo &kernel : result.kernels) {
      if (kernel.has_text_range && (!restrict_to_preflight_candidates ||
                                    kernel.preflight_action == ConSanPreflightAction::Candidate))
        kernel_entries.push_back(kernel.entry_text_offset);
    }
    std::ranges::sort(kernel_entries);
    kernel_entries.erase(std::ranges::unique(kernel_entries).begin(), kernel_entries.end());
    reservoir_owner_contexts.reserve(result.kernels.size());
    for (const ConSanKernelInfo &kernel : result.kernels) {
      // With no shared functions, a preflight-skipped kernel cannot own a
      // supported access. Avoid constructing thousands of unused liveness
      // analyses for generated operator-library code objects.
      if (restrict_to_preflight_candidates &&
          kernel.preflight_action != ConSanPreflightAction::Candidate)
        continue;
      if (!kernel.has_text_range || kernel.descriptor_file_offset > original_bytes.size() ||
          sizeof(KD) > original_bytes.size() - kernel.descriptor_file_offset)
        continue;
      auto scope =
          build_kernel_cfg_scope(blocks, block_index,
                                 KernelScopeRequest{.entry_offset = kernel.entry_text_offset,
                                                    .additional_entry_offsets = {}},
                                 kernel_entries, pristine_text);
      if (!scope)
        continue;
      KD descriptor{};
      std::memcpy(&descriptor, original_bytes.data() + kernel.descriptor_file_offset,
                  sizeof(descriptor));
      reservoir_owner_contexts.push_back(
          {.descriptor_file_offset = kernel.descriptor_file_offset,
           .scope = std::move(*scope),
           .liveness = nullptr,
           .wavefront_size = descriptor_wavefront_size(arch, descriptor)});
      ScReservoirOwnerContext &context = reservoir_owner_contexts.back();
      context.liveness = std::make_unique<LivenessAnalysis>(KernelBlockScope(context.scope.blocks),
                                                            LivenessAnalysisOptions{},
                                                            context.scope.liveness_edges);
    }
  }

  struct LdsCheckTrapCandidate {
    const ConSanLdsSite *site = nullptr;
    const LocalNopCave *local_cave = nullptr;
    const LocalNopCave *entry_island = nullptr;
    bool entry_island_is_appended = false;
    uint64_t kernel_entry_text_offset = 0;
    uint64_t descriptor_file_offset = 0;
    uint64_t appended_cave_text_offset = 0;
    uint16_t scratch_vgpr = 0;
    uint16_t vcc_save_sgpr = 0;
    std::optional<uint16_t> indirect_pc_sgpr;
    std::optional<uint16_t> indirect_scc_save_sgpr;
    uint64_t requested_words = 0;
    bool use_local_cave = false;
    bool use_appended_cave = false;
    std::optional<uint16_t> required_vgpr_allocation_count;
    std::optional<DbiPatchPlacement> placement;
    std::vector<uint64_t> entry_relay_offsets = {};
    std::optional<uint16_t> gfx1250_vgpr_msb_mode;
    std::optional<size_t> dense_call_group_index;
    std::optional<uint16_t> dense_call_return_sgpr;
    std::optional<uint16_t> dense_explicit_key;
    std::optional<uint16_t> scalar_vcc_spill_vgpr;
    bool requires_branch_only_route = false;
    std::optional<BranchOnlyRelayRoute> branch_only_route = std::nullopt;
  };

  // The save and restore described by this type bracket one instrumentation
  // body and complete before control returns to guest code. That lifetime
  // contract permits independent owner sets to reuse byte offsets. A future
  // spill that survives a call, relay, or guest-visible control-flow edge
  // needs a separate plan type backed by a globally disjoint allocator.
  struct ScBracketLocalPrivateSpillPlan {
    std::vector<uint64_t> owner_descriptor_file_offsets;
    uint32_t original_private_bytes = 0;
    uint32_t private_limit = 0;
    uint16_t vgpr_base = 0;
    uint16_t vgpr_count = 0;
  };

  struct ScDenseCallGroup {
    uint64_t kernel_entry_text_offset = 0;
    uint64_t descriptor_file_offset = 0;
    uint64_t entry_host_offset = 0;
    std::vector<uint32_t> displaced_host_words;
    uint64_t dispatcher_offset = 0;
    uint64_t entry_host_body_offset = 0;
    uint16_t jump_pc_sgpr = 0;
    uint16_t saved_scc_sgpr = 0;
    uint16_t call_return_sgpr = 0;
    std::optional<uint16_t> explicit_key_sgpr;
    uint16_t required_sgpr_count = 0;
    std::vector<const ConSanLdsSite *> sites;
  };
  constexpr uint16_t kDenseExplicitKeyMaximumSites = 64u;
  std::vector<ScDenseCallGroup> dense_call_groups;

  // A wide relay donor combines a relocatable two-dword instruction with a
  // pristine six-dword sequence. The sequence's appended body preserves its
  // ordinary execution path; its skipped payload preserves the two-dword
  // instruction and contributes three capacity-one relay words.
  struct ScRelayAnchorCandidate {
    uint64_t anchor_offset = 0;
    std::array<uint32_t, 2> anchor_original_words{};
  };
  struct ScWideRelayHostCandidate {
    uint64_t host_offset = 0;
    std::vector<uint32_t> host_original_words;
    std::vector<uint32_t> prefix_word_counts;
  };
  struct ScWideCompositeBranchRelayDonor {
    uint64_t anchor_offset = 0;
    std::array<uint32_t, 2> anchor_original_words{};
    uint64_t host_offset = 0;
    std::vector<uint32_t> host_original_words;
    DbiPatchPlacement host_placement;
  };
  struct ScRelayReservoirCandidate {
    uint64_t anchor_offset = 0;
    uint32_t original_size = 0;
    const BasicBlock *block = nullptr;
    uint16_t pc_sgpr_pair = 0;
    uint16_t saved_vcc_sgpr = 0;
    uint16_t return_pc_sgpr_pair = 0;
    uint16_t return_saved_vcc_sgpr = 0;
    uint16_t required_sgpr_count = 0;
    std::vector<uint64_t> owner_descriptor_file_offsets;
  };
  struct ScRelayReservoir {
    uint64_t anchor_offset = 0;
    uint32_t original_size = 0;
    uint16_t pc_sgpr_pair = 0;
    uint16_t saved_vcc_sgpr = 0;
    uint16_t return_pc_sgpr_pair = 0;
    uint16_t return_saved_vcc_sgpr = 0;
    uint16_t required_sgpr_count = 0;
    std::vector<uint64_t> owner_descriptor_file_offsets;
    DbiPatchPlacement placement;
    bool used = false;
  };

  // Candidate discovery visits every decoded instruction, which can number in
  // the hundreds of thousands in generated library code objects. Index the
  // immutable container and probe ranges once instead of rescanning every
  // kernel and every site for each instruction. In the sorted representation,
  // `end` is the greatest end seen at or before that entry; this supports both
  // containment and overlap queries in logarithmic time, including nested
  // ranges.
  struct ScRangeIndex {
    std::vector<ByteRange> ranges;

    void add(uint64_t begin, uint64_t size) {
      if (size != 0 && begin <= std::numeric_limits<uint64_t>::max() - size)
        ranges.push_back({begin, begin + size});
    }

    void finalize() {
      std::ranges::sort(ranges, [](ByteRange lhs, ByteRange rhs) {
        return std::tie(lhs.begin, lhs.end) < std::tie(rhs.begin, rhs.end);
      });
      uint64_t greatest_end = 0;
      for (ByteRange &range : ranges) {
        greatest_end = std::max(greatest_end, range.end);
        range.end = greatest_end;
      }
    }

    [[nodiscard]] bool contains(uint64_t begin, uint64_t size) const {
      if (size == 0 || begin > std::numeric_limits<uint64_t>::max() - size)
        return false;
      const auto it = std::ranges::upper_bound(ranges, begin, {}, &ByteRange::begin);
      return it != ranges.begin() && std::prev(it)->end >= begin + size;
    }

    [[nodiscard]] bool overlaps(uint64_t begin, uint64_t size) const {
      if (size == 0 || begin > std::numeric_limits<uint64_t>::max() - size)
        return false;
      const auto it = std::ranges::lower_bound(ranges, begin + size, {}, &ByteRange::begin);
      return it != ranges.begin() && std::prev(it)->end > begin;
    }
  };

  ScRangeIndex selected_container_ranges;
  ScRangeIndex future_flat_probe_ranges;
  ScRangeIndex supported_lds_probe_ranges;
  const auto container_is_selected = [&](const auto &container) {
    if (options.test_kernel_name_filter.empty() ||
        container.name.find(options.test_kernel_name_filter) != std::string::npos)
      return true;
    const auto owner_matches = [&](uint64_t descriptor_offset) {
      const auto kernel = std::ranges::find(result.kernels, descriptor_offset,
                                            &ConSanKernelInfo::descriptor_file_offset);
      return kernel != result.kernels.end() &&
             kernel->name.find(options.test_kernel_name_filter) != std::string::npos;
    };
    const auto site_has_matching_owner = [&](const auto &site) {
      return std::ranges::any_of(site.owner_descriptor_file_offsets, owner_matches);
    };
    return std::ranges::any_of(container.lds_sites, site_has_matching_owner) ||
           std::ranges::any_of(container.flat_sites, site_has_matching_owner);
  };
  const auto index_container = [&](const auto &container, bool add_selected_range) {
    if (!container_is_selected(container))
      return;
    if (add_selected_range)
      selected_container_ranges.add(container.entry_text_offset, container.code_size);
    if (options.probe_flat_check_trap) {
      for (const ConSanFlatSite &site : container.flat_sites) {
        if (is_supported_flat_check_trap_site(site, options.flat_provenance_mode))
          future_flat_probe_ranges.add(site.text_offset, site.size);
      }
    }
    for (const ConSanLdsSite &site : container.lds_sites) {
      if (is_supported_check_trap_site(site, arch))
        supported_lds_probe_ranges.add(site.text_offset, site.size);
    }
  };
  for (const ConSanKernelInfo &kernel : result.kernels)
    index_container(kernel, kernel.preflight_action != ConSanPreflightAction::Reject);
  for (const ConSanFunctionInfo &function : result.functions)
    index_container(function, true);
  selected_container_ranges.finalize();
  future_flat_probe_ranges.finalize();
  supported_lds_probe_ranges.finalize();

  const auto range_is_in_selected_container = [&](uint64_t offset, uint64_t size) {
    return selected_container_ranges.contains(offset, size);
  };
  const auto overlaps_future_flat_probe = [&](uint64_t offset, uint64_t size) {
    return future_flat_probe_ranges.overlaps(offset, size);
  };
  const auto overlaps_supported_lds_probe = [&](uint64_t offset, uint64_t size) {
    return supported_lds_probe_ranges.overlaps(offset, size);
  };

  std::vector<const Instruction *> ordered_instructions;
  std::unordered_map<const Instruction *, BasicBlock *> instruction_owner_blocks;
  for (BasicBlock *block : block_ptrs) {
    if (block == nullptr)
      continue;
    for (const Instruction &inst : block->instructions()) {
      ordered_instructions.push_back(&inst);
      instruction_owner_blocks.emplace(&inst, block);
    }
  }
  std::ranges::sort(ordered_instructions, {},
                    [](const Instruction *inst) { return inst->src_loc(); });
  std::unordered_set<uint64_t> s_clause_blocked_offsets;
  uint32_t clause_remaining = 0;
  std::optional<uint64_t> preceding_end;
  for (const Instruction *inst : ordered_instructions) {
    if (preceding_end && inst->src_loc() != *preceding_end)
      clause_remaining = 0;
    if (clause_remaining > 0) {
      s_clause_blocked_offsets.insert(inst->src_loc());
      --clause_remaining;
    }
    if (is_s_clause(*inst))
      clause_remaining = s_clause_following_instruction_count(*inst);
    preceding_end = inst->src_loc() + static_cast<uint64_t>(inst->size());
  }

  std::vector<ScRelayAnchorCandidate> relay_anchor_candidates;
  for (BasicBlock *block : block_ptrs) {
    if (block == nullptr)
      continue;
    for (const Instruction &inst : block->instructions()) {
      const uint64_t offset = inst.src_loc();
      if (inst.size() != 2 * static_cast<int>(sizeof(uint32_t)) ||
          std::string_view(inst.mnemonic()).starts_with("ds_") || inst.raw_encoding() == nullptr ||
          !is_relocatable_anchor(inst, offset, pristine_text, arch) ||
          offset % sizeof(uint32_t) != 0 || !range_is_in_selected_container(offset, inst.size()) ||
          s_clause_blocked_offsets.contains(offset) ||
          overlaps_future_flat_probe(offset, inst.size()) ||
          overlaps_preapplied_reserved_range(
              options, {offset, offset + static_cast<uint64_t>(inst.size())})) {
        continue;
      }
      ScRelayAnchorCandidate candidate;
      candidate.anchor_offset = offset;
      std::memcpy(candidate.anchor_original_words.data(), pristine_text.data() + offset,
                  inst.size());
      relay_anchor_candidates.push_back(candidate);
    }
  }
  std::ranges::sort(relay_anchor_candidates, {}, &ScRelayAnchorCandidate::anchor_offset);

  std::vector<ScWideRelayHostCandidate> relay_wide_host_candidates;
  for (BasicBlock *block : block_ptrs) {
    if (block == nullptr)
      continue;
    auto &instructions = block->instructions();
    for (auto host_it = instructions.begin(); host_it != instructions.end(); ++host_it) {
      const Instruction &host = *host_it;
      const uint64_t host_offset = host.src_loc();
      if (host.size() != 3 * static_cast<int>(sizeof(uint32_t)) ||
          (!std::string_view(host.mnemonic()).starts_with("flat_load") &&
           !std::string_view(host.mnemonic()).starts_with("flat_store")) ||
          host.raw_encoding() == nullptr || s_clause_blocked_offsets.contains(host_offset))
        continue;
      constexpr uint32_t kMaxWideRelayHostWords = 512u;
      uint32_t accumulated_size = 0;
      std::vector<uint32_t> prefix_word_counts;
      auto sequence_it = host_it;
      while (sequence_it != instructions.end() &&
             accumulated_size < kMaxWideRelayHostWords * sizeof(uint32_t)) {
        const Instruction &inst = *sequence_it;
        if (inst.src_loc() != host_offset + accumulated_size || inst.size() <= 0 ||
            !is_consan_branch_relay_reservoir_instruction(inst, inst.src_loc(), pristine_text,
                                                          arch) ||
            s_clause_blocked_offsets.contains(inst.src_loc()) ||
            overlaps_supported_lds_probe(inst.src_loc(), static_cast<uint64_t>(inst.size())) ||
            overlaps_future_flat_probe(inst.src_loc(), static_cast<uint64_t>(inst.size())) ||
            overlaps_preapplied_reserved_range(
                options, {inst.src_loc(), inst.src_loc() + static_cast<uint64_t>(inst.size())}) ||
            !range_is_in_selected_container(inst.src_loc(), static_cast<uint64_t>(inst.size())) ||
            static_cast<uint32_t>(inst.size()) >
                kMaxWideRelayHostWords * sizeof(uint32_t) - accumulated_size) {
          break;
        }
        accumulated_size += static_cast<uint32_t>(inst.size());
        if (accumulated_size >= 6u * sizeof(uint32_t))
          prefix_word_counts.push_back(accumulated_size / sizeof(uint32_t));
        ++sequence_it;
      }
      if (prefix_word_counts.empty())
        continue;
      ScWideRelayHostCandidate candidate;
      candidate.host_offset = host_offset;
      candidate.host_original_words.resize(accumulated_size / sizeof(uint32_t));
      candidate.prefix_word_counts = std::move(prefix_word_counts);
      std::memcpy(candidate.host_original_words.data(), pristine_text.data() + host_offset,
                  accumulated_size);
      relay_wide_host_candidates.push_back(candidate);
    }
  }
  std::ranges::sort(relay_wide_host_candidates, {}, &ScWideRelayHostCandidate::host_offset);

  // A reservoir relocates one pristine straight-line sequence and turns every
  // skipped word after its entry branch into an independent relay vertex. Keep
  // candidate windows bounded to avoid pathological code growth, while still
  // admitting the hundreds of vertices required by dense generated kernels.
  constexpr uint32_t kMaxScRelayReservoirWords = 512u;
  std::vector<ScRelayReservoirCandidate> relay_reservoir_candidates;
  const auto reservoir_instruction_is_admissible = [&](const Instruction &inst) {
    const uint64_t offset = inst.src_loc();
    const int size = inst.size();
    const std::string_view mnemonic = inst.mnemonic();
    if (size <= 0 || size % static_cast<int>(sizeof(uint32_t)) != 0 ||
        inst.raw_encoding() == nullptr || offset % sizeof(uint32_t) != 0 ||
        mnemonic.starts_with("ds_") || mnemonic == "s_clause" || mnemonic == "s_delay_alu" ||
        s_clause_blocked_offsets.contains(offset) ||
        !range_is_in_selected_container(offset, static_cast<uint64_t>(size)) ||
        overlaps_supported_lds_probe(offset, static_cast<uint64_t>(size)) ||
        overlaps_future_flat_probe(offset, static_cast<uint64_t>(size)) ||
        overlaps_preapplied_reserved_range(options,
                                           {offset, offset + static_cast<uint64_t>(size)})) {
      return false;
    }
    return is_consan_branch_relay_reservoir_instruction(inst, offset, pristine_text, arch);
  };
  for (BasicBlock *block : block_ptrs) {
    if (block == nullptr)
      continue;
    std::vector<std::pair<uint64_t, uint32_t>> run;
    const auto flush_run = [&]() {
      for (size_t begin = 0; begin < run.size(); ++begin) {
        uint32_t bytes = 0;
        size_t end = begin;
        while (end < run.size() &&
               run[end].second <= kMaxScRelayReservoirWords * sizeof(uint32_t) - bytes) {
          bytes += run[end].second;
          ++end;
        }
        if (bytes >= 2u * sizeof(uint32_t))
          relay_reservoir_candidates.push_back({.anchor_offset = run[begin].first,
                                                .original_size = bytes,
                                                .block = block,
                                                .pc_sgpr_pair = 0,
                                                .saved_vcc_sgpr = 0,
                                                .return_pc_sgpr_pair = 0,
                                                .return_saved_vcc_sgpr = 0,
                                                .required_sgpr_count = 0,
                                                .owner_descriptor_file_offsets = {}});
      }
      run.clear();
    };
    std::optional<uint64_t> expected_offset;
    for (const Instruction &inst : block->instructions()) {
      if (!reservoir_instruction_is_admissible(inst) ||
          (expected_offset && inst.src_loc() != *expected_offset)) {
        flush_run();
      }
      if (!reservoir_instruction_is_admissible(inst)) {
        expected_offset.reset();
        continue;
      }
      run.emplace_back(inst.src_loc(), static_cast<uint32_t>(inst.size()));
      expected_offset = inst.src_loc() + static_cast<uint64_t>(inst.size());
    }
    flush_run();
  }
  std::unordered_map<uint64_t, const Instruction *> reservoir_instruction_by_offset;
  for (BasicBlock *block : block_ptrs) {
    if (block == nullptr)
      continue;
    for (const Instruction &instruction : block->instructions())
      reservoir_instruction_by_offset.emplace(instruction.src_loc(), &instruction);
  }
  std::unordered_map<const BasicBlock *, std::vector<const ScReservoirOwnerContext *>>
      reservoir_owners_by_block;
  for (const ScReservoirOwnerContext &context : reservoir_owner_contexts) {
    for (const BasicBlock *block : context.scope.blocks)
      reservoir_owners_by_block[block].push_back(&context);
  }
  std::erase_if(relay_reservoir_candidates, [&](ScRelayReservoirCandidate &candidate) {
    if (candidate.original_size <
            (kScRelayEntryWords + kScRelayTailRestoreWords + 1u) * sizeof(uint32_t) ||
        reservoir_owner_contexts.empty())
      return true;
    const auto entry_it = reservoir_instruction_by_offset.find(candidate.anchor_offset);
    const Instruction *entry_instruction =
        entry_it == reservoir_instruction_by_offset.end() ? nullptr : entry_it->second;
    if (entry_instruction == nullptr)
      return true;
    const auto boundary_it =
        reservoir_instruction_by_offset.find(candidate.anchor_offset + candidate.original_size);
    const Instruction *boundary =
        boundary_it == reservoir_instruction_by_offset.end() ? nullptr : boundary_it->second;
    const BasicBlock *candidate_block = candidate.block;
    if (candidate_block == nullptr)
      return true;
    const auto owners_it = reservoir_owners_by_block.find(candidate_block);
    if (owners_it == reservoir_owners_by_block.end() || owners_it->second.empty())
      return true;
    const std::vector<const ScReservoirOwnerContext *> &owners = owners_it->second;
    if (!std::ranges::all_of(owners,
                             [](const auto *owner) { return owner->wavefront_size == 32u; }))
      return true;
    const auto register_is_dead_at_endpoint = [&](const ScReservoirOwnerContext &owner,
                                                  RegisterRef reg, bool entry) {
      if (entry)
        return !owner.liveness->any_live_before(*entry_instruction, reg);
      if (boundary != nullptr)
        return !owner.liveness->any_live_before(*boundary, reg);
      return !owner.liveness->any_live_out(*candidate_block, reg);
    };
    const auto find_saved_vcc_sgpr = [&](bool entry) -> std::optional<uint16_t> {
      for (uint16_t sgpr = 0u; sgpr < 104u; ++sgpr) {
        const RegisterRef ref{RegClass::SGPR, sgpr, 1};
        if (std::ranges::all_of(owners, [&](const auto *owner) {
              return register_is_dead_at_endpoint(*owner, ref, entry);
            }))
          return sgpr;
      }
      return std::nullopt;
    };
    const auto entry_saved_vcc = find_saved_vcc_sgpr(true);
    if (!entry_saved_vcc)
      return true;
    const std::optional<uint16_t> return_saved_vcc = find_saved_vcc_sgpr(false);
    if (!return_saved_vcc)
      return true;
    candidate.pc_sgpr_pair = kRdna4VccLo;
    candidate.saved_vcc_sgpr = *entry_saved_vcc;
    candidate.return_pc_sgpr_pair = kRdna4VccLo;
    candidate.return_saved_vcc_sgpr = *return_saved_vcc;
    candidate.required_sgpr_count =
        std::max<uint16_t>(static_cast<uint16_t>(*entry_saved_vcc + 1u),
                           static_cast<uint16_t>(*return_saved_vcc + 1u));
    for (const ScReservoirOwnerContext *owner : owners)
      candidate.owner_descriptor_file_offsets.push_back(owner->descriptor_file_offset);
    std::ranges::sort(candidate.owner_descriptor_file_offsets);
    return false;
  });
  std::ranges::sort(relay_reservoir_candidates, [](const auto &lhs, const auto &rhs) {
    if (lhs.anchor_offset != rhs.anchor_offset)
      return lhs.anchor_offset < rhs.anchor_offset;
    return lhs.original_size < rhs.original_size;
  });
  relay_reservoir_candidates.erase(std::ranges::unique(relay_reservoir_candidates, {},
                                                       [](const auto &candidate) {
                                                         return std::pair(candidate.anchor_offset,
                                                                          candidate.original_size);
                                                       })
                                       .begin(),
                                   relay_reservoir_candidates.end());

  struct LdsSitePlanningContext {
    std::string_view container_name;
    uint64_t container_entry_text_offset = 0;
    const ConSanLdsSite *site = nullptr;
    const Instruction *instruction = nullptr;
    std::vector<uint64_t> owner_descriptor_file_offsets;
    std::vector<const ScReservoirOwnerContext *> owner_contexts;
    std::optional<uint16_t> common_allocated_vgpr_count;
    std::optional<uint16_t> common_allocated_sgpr_count;
    std::optional<uint16_t> min_auto_scratch_vgpr;
    std::optional<uint16_t> min_preferred_vcc_save_sgpr;
    uint32_t max_owner_private_segment_size = 0;
    bool uses_dynamic_stack = false;
  };
  std::unordered_map<uint64_t, const ConSanKernelInfo *> kernels_by_descriptor;
  std::unordered_map<uint64_t, KernelMaxRegisterRefs> max_refs_by_descriptor;
  kernels_by_descriptor.reserve(result.kernels.size());
  max_refs_by_descriptor.reserve(result.kernels.size());
  for (const ConSanKernelInfo &kernel : result.kernels) {
    kernels_by_descriptor.emplace(kernel.descriptor_file_offset, &kernel);
    const auto owner_context =
        std::ranges::find(reservoir_owner_contexts, kernel.descriptor_file_offset,
                          &ScReservoirOwnerContext::descriptor_file_offset);
    if (owner_context != reservoir_owner_contexts.end()) {
      max_refs_by_descriptor.emplace(
          kernel.descriptor_file_offset,
          max_register_refs_in_kernel(kernel, KernelBlockScope(owner_context->scope.blocks)));
    }
  }

  std::vector<LdsSitePlanningContext> site_contexts;
  const auto add_site_context = [&](std::string_view container_name,
                                    uint64_t container_entry_text_offset, const ConSanLdsSite &site,
                                    std::vector<uint64_t> owners) {
    std::ranges::sort(owners);
    owners.erase(std::ranges::unique(owners).begin(), owners.end());
    if (owners.empty())
      return;

    LdsSitePlanningContext context;
    context.container_name = container_name;
    context.container_entry_text_offset = container_entry_text_offset;
    context.site = &site;
    context.owner_descriptor_file_offsets = std::move(owners);
    context.instruction =
        find_instruction_at_text_offset(KernelBlockScope(block_ptrs), site.text_offset);
    bool complete = true;
    bool first_owner = true;
    std::optional<uint16_t> max_owner_vgpr_ref;
    std::optional<uint16_t> max_owner_sgpr_ref;
    for (uint64_t owner : context.owner_descriptor_file_offsets) {
      const auto owner_kernel = kernels_by_descriptor.find(owner);
      const auto owner_context = std::ranges::find(
          reservoir_owner_contexts, owner, &ScReservoirOwnerContext::descriptor_file_offset);
      const auto allocated_vgprs = descriptor_vgpr_allocation_count(original_bytes, owner, arch);
      const Instruction *owner_site_inst =
          owner_context == reservoir_owner_contexts.end()
              ? nullptr
              : find_instruction_at_text_offset(KernelBlockScope(owner_context->scope.blocks),
                                                site.text_offset);
      if (owner_kernel == kernels_by_descriptor.end() ||
          owner_context == reservoir_owner_contexts.end() || owner_context->liveness == nullptr ||
          context.instruction == nullptr || owner_site_inst != context.instruction ||
          !allocated_vgprs || owner > original_bytes.size() ||
          sizeof(KD) > original_bytes.size() - owner ||
          owner_kernel->second->preflight_action == ConSanPreflightAction::Reject) {
        complete = false;
        break;
      }
      context.owner_contexts.push_back(&*owner_context);
      KD descriptor{};
      std::memcpy(&descriptor, original_bytes.data() + owner, sizeof(descriptor));
      const uint16_t allocated_sgprs = descriptor_sgpr_allocation_count(descriptor, arch);
      context.max_owner_private_segment_size =
          std::max(context.max_owner_private_segment_size, descriptor.private_segment_fixed_size);
      context.common_allocated_vgpr_count =
          first_owner ? *allocated_vgprs
                      : std::min(*context.common_allocated_vgpr_count, *allocated_vgprs);
      context.common_allocated_sgpr_count =
          first_owner ? allocated_sgprs
                      : std::min(*context.common_allocated_sgpr_count, allocated_sgprs);
      const auto refs_it = max_refs_by_descriptor.find(owner);
      if (refs_it == max_refs_by_descriptor.end()) {
        complete = false;
        break;
      }
      const KernelMaxRegisterRefs &refs = refs_it->second;
      if (refs.vgpr)
        max_owner_vgpr_ref =
            max_owner_vgpr_ref ? std::max(*max_owner_vgpr_ref, *refs.vgpr) : *refs.vgpr;
      if (refs.sgpr)
        max_owner_sgpr_ref =
            max_owner_sgpr_ref ? std::max(*max_owner_sgpr_ref, *refs.sgpr) : *refs.sgpr;
      context.uses_dynamic_stack |= owner_kernel->second->uses_dynamic_stack.value_or(false);
      first_owner = false;
    }
    if (!complete)
      return;
    if (max_owner_vgpr_ref && *max_owner_vgpr_ref < 255u)
      context.min_auto_scratch_vgpr = static_cast<uint16_t>(*max_owner_vgpr_ref + 1u);
    if (max_owner_sgpr_ref && *max_owner_sgpr_ref + 1u < REGISTER_SET_ALLOCATABLE_SGPRS)
      context.min_preferred_vcc_save_sgpr = static_cast<uint16_t>(*max_owner_sgpr_ref + 1u);
    site_contexts.push_back(std::move(context));
  };
  for (const CanonicalLdsSite &canonical : canonical_sites)
    add_site_context(canonical.container_name, canonical.container_entry_text_offset,
                     canonical.site, canonical.site.owner_descriptor_file_offsets);

  // Coverage is a physical decoded-site inventory, not an inventory of sites
  // that happened to survive owner, CFG, liveness, or placement planning.
  // Retain every canonical native-LDS access in the denominator and advertise
  // support only after its complete planning context has been established.
  std::unordered_set<const ConSanLdsSite *> sites_with_complete_context;
  sites_with_complete_context.reserve(site_contexts.size());
  for (const LdsSitePlanningContext &context : site_contexts) {
    if (context.site != nullptr && !context.owner_contexts.empty())
      sites_with_complete_context.insert(context.site);
  }
  std::unordered_set<uint64_t> supported_physical_sites;
  supported_physical_sites.reserve(sites_with_complete_context.size());
  if (decoder != nullptr) {
    for (const ConSanLdsSite *site : sites_with_complete_context) {
      if (site != nullptr && is_supported_check_trap_site(*site, arch))
        supported_physical_sites.insert(site->file_offset);
    }
  }
  publish_sc_access_coverage_support(
      result, ConSanScAccessCoverageKind::NativeLds,
      [&](uint64_t file_offset) { return supported_physical_sites.contains(file_offset); });
  if (options.max_patches_is_expert_limit && options.max_patches == 0u) {
    result.warnings.emplace_back(
        "ConSan LDS check/trap proof skipped: shared patch budget consumed");
    return;
  }

  std::vector<LdsCheckTrapCandidate> inline_candidates;
  std::vector<LdsCheckTrapCandidate> local_cave_candidates;
  std::vector<LdsCheckTrapCandidate> appended_cave_candidates;
  std::vector<LdsCheckTrapCandidate> generated_island_seed_candidates;
  std::vector<LdsCheckTrapCandidate> branch_only_candidates;
  bool skipped_supported_site_for_excessive_delay = false;
  size_t supported_candidate_count = 0;
  size_t scratchable_candidate_count = 0;
  size_t append_cave_reachable_candidate_count = 0;
  size_t indirect_pc_resource_failed_site_count = 0;
  size_t indirect_island_unreachable_site_count = 0;
  size_t indirect_candidate_placement_failure_count = 0;
  size_t local_cave_reachable_candidate_count = 0;
  size_t s_clause_blocked_candidate_count = 0;
  size_t missing_width_plan_site_count = 0;
  size_t missing_instruction_site_count = 0;
  size_t missing_vcc_save_site_count = 0;
  size_t missing_scratch_site_count = 0;
  size_t invalid_scratch_allocation_site_count = 0;
  size_t dynamic_stack_private_spill_rejection_count = 0;
  size_t private_spill_encoding_failure_count = 0;
  uint32_t max_observed_padding_words = 0;
  // Candidate discovery records immutable requirements only. Stable private
  // slots are assigned transactionally when a concrete placement is selected,
  // so rejected candidates cannot inflate any descriptor.
  std::unordered_map<const ConSanLdsSite *, ScBracketLocalPrivateSpillPlan> private_spill_plans;
  auto reachable_local_caves = [&](const ConSanLdsSite &site, uint64_t owner_entry_text_offset,
                                   uint64_t requested_words) {
    std::vector<const LocalNopCave *> caves;
    const uint64_t cave_words = requested_words + 1u;
    for (const LocalNopCave &cave : local_nop_caves) {
      if (cave.owner_entry_text_offset != owner_entry_text_offset || cave.word_count < cave_words)
        continue;
      const uint64_t return_branch_pc = cave.text_offset + requested_words * sizeof(uint32_t);
      if (compute_sopp_branch_simm16(site.text_offset, cave.text_offset) &&
          compute_sopp_branch_simm16(return_branch_pc, site.text_offset + site.size))
        caves.push_back(&cave);
    }
    return caves;
  };
  for (const LdsSitePlanningContext &site_context : site_contexts) {
    if (!options.test_kernel_name_filter.empty()) {
      const bool container_matches =
          site_context.container_name.find(options.test_kernel_name_filter) != std::string::npos;
      const bool owner_matches =
          std::ranges::any_of(site_context.owner_descriptor_file_offsets, [&](uint64_t owner) {
            const auto kernel = kernels_by_descriptor.find(owner);
            return kernel != kernels_by_descriptor.end() &&
                   kernel->second->name.find(options.test_kernel_name_filter) != std::string::npos;
          });
      if (!container_matches && !owner_matches)
        continue;
    }
    if (site_context.site == nullptr || site_context.owner_contexts.empty())
      continue;
    const KernelBlockScope kernel_scope(site_context.owner_contexts.front()->scope.blocks);
    std::vector<const LivenessAnalysis *> owner_liveness;
    owner_liveness.reserve(site_context.owner_contexts.size());
    for (const ScReservoirOwnerContext *owner : site_context.owner_contexts)
      owner_liveness.push_back(owner->liveness.get());
    const auto max_auto_scratch_vgpr = site_context.common_allocated_vgpr_count;
    const auto allocated_sgpr_count = site_context.common_allocated_sgpr_count;
    const auto min_auto_scratch_vgpr = site_context.min_auto_scratch_vgpr;
    const auto min_preferred_vcc_save_sgpr = site_context.min_preferred_vcc_save_sgpr;
    const uint64_t primary_descriptor_file_offset =
        site_context.owner_descriptor_file_offsets.front();
    {
      const ConSanLdsSite &site = *site_context.site;
      if (!is_supported_check_trap_site(site, arch))
        continue;
      ++supported_candidate_count;
      const std::optional<uint16_t> gfx1250_vgpr_msb_mode =
          arch == ROCJITSU_CODE_ARCH_GFX1250 && site.file_offset >= site.text_offset
              ? consan_gfx1250_vgpr_msb_mode_at(original_bytes, site.file_offset - site.text_offset,
                                                site_context.container_entry_text_offset,
                                                site.file_offset)
              : std::nullopt;
      auto required_vgprs = lds_dword_count(site);
      if (!required_vgprs) {
        ++missing_width_plan_site_count;
        continue;
      }
      auto action_words =
          mismatch_action_word_count(options, arch, result.errors, "ConSan LDS check/trap proof");
      if (!action_words)
        return;
      const uint32_t report_alignment_padding =
          options.report_buffer_address &&
                  (arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4)
              ? 1u
              : 0u;
      const Instruction *site_inst = site_context.instruction;
      if (site_inst == nullptr) {
        ++missing_instruction_site_count;
        continue;
      }
      // The complete physical-site owner set is known here, so a dead scalar
      // above one owner's current allocation is safe when every owner
      // descriptor is grown transactionally. Liveness across all owners,
      // rather than the smallest preexisting descriptor, defines the reusable
      // register set.
      constexpr uint16_t dead_vcc_selection_limit = REGISTER_SET_ALLOCATABLE_SGPRS;
      auto vcc_save =
          allocated_sgpr_count
              ? choose_sc_dead_vcc_save_sgpr(site_inst, owner_liveness, *allocated_sgpr_count,
                                             min_preferred_vcc_save_sgpr, dead_vcc_selection_limit,
                                             arch, options)
              : std::nullopt;
      std::optional<uint16_t> scalar_vcc_spill_sgpr;
      if (!vcc_save && is_rdna4_family_arch(arch) && allocated_sgpr_count) {
        scalar_vcc_spill_sgpr = choose_sc_borrowed_scalar_sgpr(*allocated_sgpr_count, options);
        vcc_save = scalar_vcc_spill_sgpr;
      }
      const bool spill_vcc_save_to_vgpr = scalar_vcc_spill_sgpr.has_value();
      if (!vcc_save) {
        ++missing_vcc_save_site_count;
        continue;
      }
      const uint16_t vcc_save_sgpr_width = sc_vcc_save_sgpr_width(arch);
      const uint32_t probe_scratch_vgprs =
          *required_vgprs + lds_check_trap_extra_scratch_vgprs(site) + report_alignment_padding +
          report_action_scratch_vgprs(options) + (lds_load_clobbers_address(site) ? 1u : 0u);
      const uint32_t scratch_vgprs = probe_scratch_vgprs + (spill_vcc_save_to_vgpr ? 1u : 0u);
      uint64_t requested_words = (lds_load_clobbers_address(site) ? 1u : 0u) + 2u +
                                 lds_check_trap_predelay_setup_words(site) +
                                 static_cast<uint64_t>(*delay_words) + 2u + 1u +
                                 lds_check_trap_precompare_words(site) + 1u +
                                 (2u + *action_words) * *required_vgprs + 1u;
      if (site.dst_accvgpr &&
          (arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4))
        requested_words += 2u * *required_vgprs;
      if (spill_vcc_save_to_vgpr)
        requested_words += 4u;
      if (spill_vcc_save_to_vgpr && gfx1250_vgpr_msb_mode.value_or(0u) != 0u)
        requested_words += 2u;
      if (arch == ROCJITSU_CODE_ARCH_GFX1250 && site.kind == ConSanLdsAccessKind::Read &&
          site.dst_vgpr) {
        const uint32_t dst_bank =
            static_cast<uint32_t>((gfx1250_vgpr_msb_mode.value_or(0u) >> 6u) & 0x3u);
        for (uint16_t i = 0; i < *required_vgprs; ++i) {
          if (dst_bank * 256u + *site.dst_vgpr + i >= 256u)
            requested_words += 2u;
        }
      }
      if (gfx1250_vgpr_msb_mode.value_or(0u) != 0u)
        requested_words += 2u;
      if (gfx1250_vgpr_msb_mode.value_or(0u) != 0u && lds_load_clobbers_address(site))
        requested_words += 2u;
      const auto choose_context_scratch =
          [&](std::optional<uint16_t> allocation_limit) -> std::optional<uint16_t> {
        if (options.scratch_vgpr) {
          return choose_scratch_vgpr(site, options, site_inst, owner_liveness.front(),
                                     min_auto_scratch_vgpr, allocation_limit, scratch_vgprs);
        }
        // A shared helper can reuse any run proved dead in every owner. An
        // owner-wide maximum-reference floor would discard those common holes
        // and needlessly grow every descriptor.
        const uint32_t begin = site_context.owner_descriptor_file_offsets.size() > 1u
                                   ? 0u
                                   : min_auto_scratch_vgpr.value_or(0u);
        const uint32_t limit = allocation_limit.value_or(256u);
        for (uint32_t base = begin; base + scratch_vgprs <= limit && base + scratch_vgprs <= 256u;
             ++base) {
          if (is_forbidden_scratch_vgpr_run(site, static_cast<uint16_t>(base), scratch_vgprs))
            continue;
          const RegisterRef run{RegClass::VGPR, static_cast<uint16_t>(base),
                                static_cast<uint8_t>(scratch_vgprs)};
          if (std::ranges::all_of(owner_liveness, [&](const LivenessAnalysis *liveness) {
                return !liveness->any_live_before(*site_inst, run);
              }))
            return static_cast<uint16_t>(base);
        }
        return std::nullopt;
      };
      uint16_t scratch_vgpr = 0;
      bool scratch_selected = false;
      if (const auto candidate = choose_context_scratch(max_auto_scratch_vgpr)) {
        scratch_vgpr = *candidate;
        scratch_selected = true;
      }
      if (!scratch_selected && !options.scratch_vgpr && max_auto_scratch_vgpr) {
        if (const auto candidate = choose_context_scratch(std::nullopt)) {
          scratch_vgpr = *candidate;
          scratch_selected = true;
        }
      }
      if (!scratch_selected && !options.scratch_vgpr && max_auto_scratch_vgpr) {
        if (site_context.uses_dynamic_stack) {
          ++dynamic_stack_private_spill_rejection_count;
        } else {
          const auto spill_scratch =
              choose_spill_scratch_vgpr(site, *max_auto_scratch_vgpr, scratch_vgprs);
          const auto private_limit = address_free_scratch_private_limit(arch);
          if (spill_scratch && private_limit) {
            scratch_vgpr = *spill_scratch;
            SpillManager preview_manager(site_context.max_owner_private_segment_size,
                                         *private_limit);
            auto spill =
                build_vgpr_spill_sequence(preview_manager, scratch_vgpr, scratch_vgprs, arch);
            if (spill) {
              requested_words += spill->save_words.size() + spill->restore_words.size();
              if (gfx1250_vgpr_msb_mode.value_or(0u) != 0u) {
                requested_words += 2u;
                if (spill_vcc_save_to_vgpr)
                  requested_words -= 2u;
              }
              private_spill_plans.insert_or_assign(
                  &site,
                  ScBracketLocalPrivateSpillPlan{
                      .owner_descriptor_file_offsets = site_context.owner_descriptor_file_offsets,
                      .original_private_bytes = site_context.max_owner_private_segment_size,
                      .private_limit = *private_limit,
                      .vgpr_base = scratch_vgpr,
                      .vgpr_count = static_cast<uint16_t>(scratch_vgprs),
                  });
              scratch_selected = true;
            } else {
              ++private_spill_encoding_failure_count;
            }
          } else {
            ++private_spill_encoding_failure_count;
          }
        }
      }
      if (!scratch_selected) {
        ++missing_scratch_site_count;
        continue;
      }
      if (requested_words > 256u) {
        skipped_supported_site_for_excessive_delay = true;
        continue;
      }
      const uint32_t padding_words = static_cast<uint32_t>(requested_words - 2u);
      auto required_allocation =
          required_descriptor_vgpr_allocation_for_scratch(site, scratch_vgpr, scratch_vgprs);
      if (!required_allocation) {
        ++invalid_scratch_allocation_site_count;
        continue;
      }
      std::optional<uint16_t> descriptor_growth;
      if (max_auto_scratch_vgpr && *required_allocation > *max_auto_scratch_vgpr)
        descriptor_growth = *required_allocation;
      const std::optional<uint16_t> scalar_vcc_spill_vgpr =
          spill_vcc_save_to_vgpr
              ? std::optional<uint16_t>(static_cast<uint16_t>(scratch_vgpr + probe_scratch_vgprs))
              : std::nullopt;
      ++scratchable_candidate_count;
      uint32_t observed_padding =
          count_nop_padding(original_bytes, site.file_offset + site.size, padding_words, arch);
      if (overlaps_preapplied_reserved_range(
              options,
              {site.text_offset,
               site.text_offset + requested_words * static_cast<uint64_t>(sizeof(uint32_t))})) {
        observed_padding = 0;
      }
      max_observed_padding_words = std::max(max_observed_padding_words, observed_padding);
      if (text_offset_is_inside_s_clause(kernel_scope, site.text_offset)) {
        ++s_clause_blocked_candidate_count;
      } else {
        bool appended_cave_reachable = false;
        if (appended_cave_offset != 0) {
          const uint64_t return_branch_pc =
              appended_cave_offset + requested_words * sizeof(uint32_t);
          if (compute_sopp_branch_simm16(site.text_offset, appended_cave_offset) &&
              compute_sopp_branch_simm16(return_branch_pc, site.text_offset + site.size)) {
            appended_cave_reachable = true;
            LdsCheckTrapCandidate candidate{&site,
                                            nullptr,
                                            nullptr,
                                            false,
                                            site_context.container_entry_text_offset,
                                            primary_descriptor_file_offset,
                                            appended_cave_offset,
                                            scratch_vgpr,
                                            *vcc_save,
                                            std::nullopt,
                                            std::nullopt,
                                            requested_words,
                                            false,
                                            true,
                                            descriptor_growth,
                                            std::nullopt,
                                            {},
                                            gfx1250_vgpr_msb_mode,
                                            std::nullopt,
                                            std::nullopt,
                                            std::nullopt,
                                            std::nullopt};
            candidate.scalar_vcc_spill_vgpr = scalar_vcc_spill_vgpr;
            appended_cave_candidates.push_back(candidate);
          }
        }
        const std::vector<const LocalNopCave *> caves =
            reachable_local_caves(site, site_context.container_entry_text_offset, requested_words);
        if (!caves.empty())
          ++local_cave_reachable_candidate_count;
        for (const LocalNopCave *cave : caves) {
          LdsCheckTrapCandidate candidate{&site,
                                          cave,
                                          nullptr,
                                          false,
                                          site_context.container_entry_text_offset,
                                          primary_descriptor_file_offset,
                                          0,
                                          scratch_vgpr,
                                          *vcc_save,
                                          std::nullopt,
                                          std::nullopt,
                                          requested_words,
                                          true,
                                          false,
                                          descriptor_growth,
                                          std::nullopt,
                                          {},
                                          gfx1250_vgpr_msb_mode,
                                          std::nullopt,
                                          std::nullopt,
                                          std::nullopt,
                                          std::nullopt};
          candidate.scalar_vcc_spill_vgpr = scalar_vcc_spill_vgpr;
          local_cave_candidates.push_back(candidate);
        }
        if (appended_cave_offset != 0 && !spill_vcc_save_to_vgpr) {
          // Every scalar temporary survives a shared helper's indirect entry
          // or return, so its complete range must be dead in every owner.
          uint16_t indirect_vcc_save_sgpr = *vcc_save;
          auto indirect_pc_sgpr = choose_lds_check_trap_indirect_pc_sgpr(
              site_inst, owner_liveness, indirect_vcc_save_sgpr, vcc_save_sgpr_width,
              min_preferred_vcc_save_sgpr.value_or(0));
          std::optional<uint16_t> indirect_scc_save_sgpr;
          if (indirect_pc_sgpr) {
            indirect_scc_save_sgpr = choose_lds_check_trap_indirect_scc_save_sgpr(
                site_inst, owner_liveness, indirect_vcc_save_sgpr, vcc_save_sgpr_width,
                *indirect_pc_sgpr, min_preferred_vcc_save_sgpr.value_or(0));
          }
          if (!indirect_pc_sgpr || !indirect_scc_save_sgpr) {
            if (auto joint_scratch = choose_lds_check_trap_indirect_scalar_scratch(
                    site_inst, owner_liveness, min_preferred_vcc_save_sgpr, arch)) {
              indirect_vcc_save_sgpr = joint_scratch->vcc_save_sgpr;
              indirect_pc_sgpr = joint_scratch->pc_sgpr_pair;
              indirect_scc_save_sgpr = joint_scratch->scc_save_sgpr;
            }
          }
          // Prefer four fresh scalar words above every guest reference. Keep
          // dead-register reuse as a bounded fallback for synthetic or
          // maximally allocated kernels until scalar spilling is available.
          if ((!indirect_pc_sgpr || !indirect_scc_save_sgpr) && min_preferred_vcc_save_sgpr) {
            if (auto joint_scratch = choose_lds_check_trap_indirect_scalar_scratch(
                    site_inst, owner_liveness, std::nullopt, arch)) {
              indirect_vcc_save_sgpr = joint_scratch->vcc_save_sgpr;
              indirect_pc_sgpr = joint_scratch->pc_sgpr_pair;
              indirect_scc_save_sgpr = joint_scratch->scc_save_sgpr;
            }
          }
          if (!indirect_pc_sgpr || !indirect_scc_save_sgpr) {
            if (!appended_cave_reachable)
              ++indirect_pc_resource_failed_site_count;
            LdsCheckTrapCandidate candidate{&site,
                                            nullptr,
                                            nullptr,
                                            false,
                                            site_context.container_entry_text_offset,
                                            primary_descriptor_file_offset,
                                            appended_cave_offset,
                                            scratch_vgpr,
                                            *vcc_save,
                                            std::nullopt,
                                            std::nullopt,
                                            requested_words,
                                            false,
                                            true,
                                            descriptor_growth,
                                            std::nullopt,
                                            {},
                                            gfx1250_vgpr_msb_mode,
                                            std::nullopt,
                                            std::nullopt,
                                            std::nullopt,
                                            scalar_vcc_spill_vgpr};
            candidate.requires_branch_only_route = true;
            branch_only_candidates.push_back(std::move(candidate));
          } else {
            LdsCheckTrapCandidate generated_seed{&site,
                                                 nullptr,
                                                 nullptr,
                                                 true,
                                                 site_context.container_entry_text_offset,
                                                 primary_descriptor_file_offset,
                                                 appended_cave_offset,
                                                 scratch_vgpr,
                                                 indirect_vcc_save_sgpr,
                                                 *indirect_pc_sgpr,
                                                 *indirect_scc_save_sgpr,
                                                 requested_words,
                                                 false,
                                                 true,
                                                 descriptor_growth,
                                                 std::nullopt,
                                                 {},
                                                 gfx1250_vgpr_msb_mode,
                                                 std::nullopt,
                                                 std::nullopt,
                                                 std::nullopt,
                                                 std::nullopt};
            generated_island_seed_candidates.push_back(generated_seed);

            if (!appended_cave_reachable) {
              bool found_reachable_island = false;
              for (const LocalNopCave &cave : local_entry_islands) {
                if (cave.owner_entry_text_offset != site_context.container_entry_text_offset ||
                    !compute_sopp_branch_simm16(site.text_offset, cave.text_offset)) {
                  continue;
                }
                found_reachable_island = true;
                LdsCheckTrapCandidate candidate{&site,
                                                nullptr,
                                                &cave,
                                                false,
                                                site_context.container_entry_text_offset,
                                                primary_descriptor_file_offset,
                                                appended_cave_offset,
                                                scratch_vgpr,
                                                indirect_vcc_save_sgpr,
                                                *indirect_pc_sgpr,
                                                *indirect_scc_save_sgpr,
                                                requested_words,
                                                false,
                                                true,
                                                descriptor_growth,
                                                std::nullopt,
                                                {},
                                                gfx1250_vgpr_msb_mode,
                                                std::nullopt,
                                                std::nullopt,
                                                std::nullopt,
                                                std::nullopt};
                appended_cave_candidates.push_back(candidate);
              }
              if (found_reachable_island)
                appended_cave_reachable = true;
              else
                ++indirect_island_unreachable_site_count;
            }
          }
        } else if (appended_cave_offset != 0 && !appended_cave_reachable) {
          ++indirect_pc_resource_failed_site_count;
          LdsCheckTrapCandidate candidate{&site,
                                          nullptr,
                                          nullptr,
                                          false,
                                          site_context.container_entry_text_offset,
                                          primary_descriptor_file_offset,
                                          appended_cave_offset,
                                          scratch_vgpr,
                                          *vcc_save,
                                          std::nullopt,
                                          std::nullopt,
                                          requested_words,
                                          false,
                                          true,
                                          descriptor_growth,
                                          std::nullopt,
                                          {},
                                          gfx1250_vgpr_msb_mode,
                                          std::nullopt,
                                          std::nullopt,
                                          std::nullopt,
                                          scalar_vcc_spill_vgpr};
          candidate.requires_branch_only_route = true;
          branch_only_candidates.push_back(std::move(candidate));
        }
        if (appended_cave_reachable)
          ++append_cave_reachable_candidate_count;
      }
      if (observed_padding < padding_words)
        continue;
      LdsCheckTrapCandidate candidate{&site,
                                      nullptr,
                                      nullptr,
                                      false,
                                      site_context.container_entry_text_offset,
                                      primary_descriptor_file_offset,
                                      0,
                                      scratch_vgpr,
                                      *vcc_save,
                                      std::nullopt,
                                      std::nullopt,
                                      requested_words,
                                      false,
                                      false,
                                      descriptor_growth,
                                      std::nullopt,
                                      {},
                                      gfx1250_vgpr_msb_mode,
                                      std::nullopt,
                                      std::nullopt,
                                      std::nullopt,
                                      std::nullopt};
      candidate.scalar_vcc_spill_vgpr = scalar_vcc_spill_vgpr;
      inline_candidates.push_back(candidate);
    }
  }
  auto candidate_order = [](const LdsCheckTrapCandidate &lhs, const LdsCheckTrapCandidate &rhs) {
    const bool lhs_grows_descriptor = lhs.required_vgpr_allocation_count.has_value();
    const bool rhs_grows_descriptor = rhs.required_vgpr_allocation_count.has_value();
    if (lhs_grows_descriptor != rhs_grows_descriptor)
      return !lhs_grows_descriptor;
    if (lhs.site->file_offset != rhs.site->file_offset)
      return lhs.site->file_offset < rhs.site->file_offset;
    if (lhs.use_local_cave != rhs.use_local_cave)
      return !lhs.use_local_cave;
    if ((lhs.entry_island != nullptr) != (rhs.entry_island != nullptr))
      return lhs.entry_island == nullptr;
    const uint64_t lhs_cave = lhs.local_cave ? lhs.local_cave->file_offset : 0;
    const uint64_t rhs_cave = rhs.local_cave ? rhs.local_cave->file_offset : 0;
    if (lhs_cave != rhs_cave)
      return lhs_cave < rhs_cave;
    const uint64_t lhs_island = lhs.entry_island ? lhs.entry_island->file_offset : 0;
    const uint64_t rhs_island = rhs.entry_island ? rhs.entry_island->file_offset : 0;
    return lhs_island < rhs_island;
  };
  std::ranges::sort(inline_candidates, candidate_order);
  std::ranges::sort(local_cave_candidates, candidate_order);
  std::ranges::sort(appended_cave_candidates, candidate_order);
  std::ranges::sort(generated_island_seed_candidates, [](const auto &lhs, const auto &rhs) {
    return lhs.site->text_offset < rhs.site->text_offset;
  });
  std::ranges::sort(branch_only_candidates, candidate_order);

  const size_t max_patches =
      options.max_patches_is_expert_limit ? options.max_patches : supported_candidate_count;
  std::vector<LdsCheckTrapCandidate> selected_candidates;
  selected_candidates.reserve(std::min(max_patches, supported_candidate_count));
  std::unordered_set<const ConSanLdsSite *> selected_sites;
  struct ScSelectedSpillState {
    // Within this bracket-local lifetime domain, an exact owner set shares
    // stable register-to-slot mappings while distinct sets may reuse offsets.
    std::map<std::vector<uint64_t>, SpillManager> managers;
    std::unordered_map<const ConSanLdsSite *, VgprSpillSequence> sequences;
  };
  ScSelectedSpillState selected_spills;
  std::unordered_set<const ConSanLdsSite *> selection_spill_failure_sites;
  const auto try_add_selected_spill = [&](ScSelectedSpillState &state,
                                          const LdsCheckTrapCandidate &candidate) {
    const auto plan = private_spill_plans.find(candidate.site);
    if (plan == private_spill_plans.end())
      return true;
    const std::vector<uint64_t> &key = plan->second.owner_descriptor_file_offsets;
    std::optional<SpillManager> manager;
    if (const auto existing = state.managers.find(key); existing != state.managers.end())
      manager = existing->second;
    else
      manager.emplace(plan->second.original_private_bytes, plan->second.private_limit);
    auto sequence =
        build_vgpr_spill_sequence(*manager, plan->second.vgpr_base, plan->second.vgpr_count, arch);
    if (!sequence)
      return false;
    state.managers.insert_or_assign(key, std::move(*manager));
    state.sequences.insert_or_assign(candidate.site, std::move(*sequence));
    return true;
  };
  const auto note_selection_spill_failure = [&](const ConSanLdsSite *site) {
    if (selection_spill_failure_sites.insert(site).second)
      ++private_spill_encoding_failure_count;
  };
  DbiPatchPlacementPlanner placement_planner(arch, appended_cave_offset);
  std::unordered_map<const ConSanLdsSite *, std::vector<uint64_t>> direct_body_relay_slots;

  // Spill-backed sites have no scalar pair for an indirect continuation, so
  // their bodies must remain close enough to pristine text for SOPP relay
  // chains. Give those rare bodies the front of the appended region before
  // the ordinary direct bodies make that region many megabytes deep.
  std::ranges::sort(branch_only_candidates, [](const auto &lhs, const auto &rhs) {
    return lhs.site->text_offset < rhs.site->text_offset;
  });
  // Branch-only continuation uses only the shared GFX12 SOPP branch contract.
  // Reserving its appended bodies on another architecture would leave phantom
  // planner allocations when that route is not emitted, shifting every later
  // body away from the emitter's append cursor.
  if (is_rdna4_family_arch(arch)) {
    for (LdsCheckTrapCandidate &candidate : branch_only_candidates) {
      DbiPatchPlacementPlanner tentative_planner = placement_planner;
      auto placement = tentative_planner.plan_indirect_appended(
          candidate.site->text_offset, candidate.site->size,
          (candidate.requested_words + 1u) * sizeof(uint32_t));
      if (!placement)
        continue;
      candidate.placement = *placement;
      placement_planner = std::move(tentative_planner);
    }
  }

  // A forward SOPP relay graph has at most 32,768 distinct instruction slots
  // across any one reach window.  Dense generated objects can contain more
  // independent LDS sites than that, making a one-relay-per-site layout
  // mathematically impossible even when pristine text has ample relocatable
  // instructions. gfx1250 can identify a site with its one-word s_call_i64;
  // RDNA4 uses an explicit inline key plus s_branch in the two-word LDS
  // anchor. Both forms let all sites in one small kernel share an eight-word
  // local entry host. The appended dispatcher uses that identity to select the
  // site's ordinary instrumentation body.
  // Keep this path for either the proven absolute min-cut bound or a large
  // object where most supported sites are already reach-stranded.  Requiring
  // at least 1,024 stranded sites amortizes the shared dispatcher and keeps
  // sparse objects on the simpler direct/island layout.
  constexpr size_t kDenseCallMinimumStrandedSites = 1024u;
  const bool exceeds_absolute_relay_bound =
      indirect_island_unreachable_site_count >
      static_cast<size_t>(std::numeric_limits<int16_t>::max()) + 1u;
  const bool is_large_majority_stranded =
      indirect_island_unreachable_site_count >= kDenseCallMinimumStrandedSites &&
      indirect_island_unreachable_site_count > supported_candidate_count / 2u;
  if (is_rdna4_family_arch(arch) && local_nop_caves.empty() &&
      (exceeds_absolute_relay_bound || is_large_majority_stranded)) {
    struct DenseSeedGroup {
      uint64_t kernel_entry_text_offset = 0;
      uint64_t descriptor_file_offset = 0;
      uint16_t jump_pc_sgpr = 0;
      uint16_t saved_scc_sgpr = 0;
      std::vector<LdsCheckTrapCandidate> candidates;
    };
    std::vector<DenseSeedGroup> seed_groups;
    for (const LdsCheckTrapCandidate &candidate : generated_island_seed_candidates) {
      if (!candidate.indirect_pc_sgpr || !candidate.indirect_scc_save_sgpr)
        continue;
      auto group = std::ranges::find_if(seed_groups, [&](const DenseSeedGroup &entry) {
        return entry.kernel_entry_text_offset == candidate.kernel_entry_text_offset &&
               entry.descriptor_file_offset == candidate.descriptor_file_offset &&
               entry.jump_pc_sgpr == *candidate.indirect_pc_sgpr &&
               entry.saved_scc_sgpr == *candidate.indirect_scc_save_sgpr;
      });
      if (group == seed_groups.end()) {
        seed_groups.push_back({candidate.kernel_entry_text_offset,
                               candidate.descriptor_file_offset,
                               *candidate.indirect_pc_sgpr,
                               *candidate.indirect_scc_save_sgpr,
                               {}});
        group = std::prev(seed_groups.end());
      }
      group->candidates.push_back(candidate);
    }

    // s_mov_b32 can encode keys 0..64 inline in one word. RDNA4 always uses
    // this two-word s_mov-plus-s_branch anchor; gfx1250 uses it when no fresh
    // call-return pair exists. Partition explicit-key groups by their
    // already-dead VCC-save SGPR and into at most 64 sites.
    std::vector<DenseSeedGroup> normalized_seed_groups;
    for (DenseSeedGroup &seed_group : seed_groups) {
      uint16_t highest_instrumentation_sgpr = std::max<uint16_t>(
          static_cast<uint16_t>(seed_group.jump_pc_sgpr + 1u), seed_group.saved_scc_sgpr);
      for (const LdsCheckTrapCandidate &candidate : seed_group.candidates)
        highest_instrumentation_sgpr =
            std::max<uint16_t>(highest_instrumentation_sgpr, candidate.vcc_save_sgpr);
      const uint16_t fresh_call_return_sgpr =
          static_cast<uint16_t>((highest_instrumentation_sgpr + 2u) & ~uint16_t{1u});
      if (arch == ROCJITSU_CODE_ARCH_GFX1250 &&
          fresh_call_return_sgpr + 1u < REGISTER_SET_ALLOCATABLE_SGPRS) {
        normalized_seed_groups.push_back(std::move(seed_group));
        continue;
      }
      for (const LdsCheckTrapCandidate &candidate : seed_group.candidates) {
        auto subgroup = std::ranges::find_if(normalized_seed_groups, [&](const auto &entry) {
          return entry.kernel_entry_text_offset == seed_group.kernel_entry_text_offset &&
                 entry.descriptor_file_offset == seed_group.descriptor_file_offset &&
                 entry.jump_pc_sgpr == seed_group.jump_pc_sgpr &&
                 entry.saved_scc_sgpr == seed_group.saved_scc_sgpr && !entry.candidates.empty() &&
                 entry.candidates.front().vcc_save_sgpr == candidate.vcc_save_sgpr &&
                 entry.candidates.size() < kDenseExplicitKeyMaximumSites;
        });
        if (subgroup == normalized_seed_groups.end()) {
          normalized_seed_groups.push_back({seed_group.kernel_entry_text_offset,
                                            seed_group.descriptor_file_offset,
                                            seed_group.jump_pc_sgpr,
                                            seed_group.saved_scc_sgpr,
                                            {}});
          subgroup = std::prev(normalized_seed_groups.end());
        }
        subgroup->candidates.push_back(candidate);
      }
    }
    seed_groups = std::move(normalized_seed_groups);

    constexpr uint32_t kDenseEntryHostWords = 8u;
    constexpr uint32_t kDenseDispatcherSetupWords = 3u;
    constexpr uint32_t kDenseDispatcherHostWords = 9u;
    constexpr uint32_t kDenseDispatcherSiteWords = 10u;
    size_t dense_seed_group_count = 0;
    size_t dense_selected_group_count = 0;
    size_t dense_missing_kernel_count = 0;
    size_t dense_missing_host_count = 0;
    size_t dense_missing_owner_context_count = 0;
    size_t dense_missing_call_sgpr_count = 0;
    size_t dense_placement_failure_count = 0;
    size_t dense_largest_missing_host_prefix_words = 0;
    std::vector<std::pair<uint64_t, uint64_t>> dense_reserved_host_ranges;
    for (DenseSeedGroup &seed_group : seed_groups) {
      if (seed_group.candidates.empty())
        continue;
      ++dense_seed_group_count;
      const auto kernel = std::ranges::find(result.kernels, seed_group.kernel_entry_text_offset,
                                            &ConSanKernelInfo::entry_text_offset);
      if (kernel == result.kernels.end() || !kernel->has_text_range) {
        ++dense_missing_kernel_count;
        continue;
      }
      uint16_t highest_instrumentation_sgpr = std::max<uint16_t>(
          static_cast<uint16_t>(seed_group.jump_pc_sgpr + 1u), seed_group.saved_scc_sgpr);
      for (const LdsCheckTrapCandidate &candidate : seed_group.candidates)
        highest_instrumentation_sgpr =
            std::max<uint16_t>(highest_instrumentation_sgpr, candidate.vcc_save_sgpr);
      const uint16_t call_return_sgpr =
          static_cast<uint16_t>((highest_instrumentation_sgpr + 2u) & ~uint16_t{1u});
      const bool use_explicit_key = arch != ROCJITSU_CODE_ARCH_GFX1250 ||
                                    call_return_sgpr + 1u >= REGISTER_SET_ALLOCATABLE_SGPRS;
      const std::optional<uint16_t> explicit_key_sgpr =
          use_explicit_key && !seed_group.candidates.empty()
              ? std::optional<uint16_t>(seed_group.candidates.front().vcc_save_sgpr)
              : std::nullopt;
      if (use_explicit_key &&
          (!explicit_key_sgpr || seed_group.candidates.size() > kDenseExplicitKeyMaximumSites ||
           !std::ranges::all_of(seed_group.candidates, [&](const auto &candidate) {
             return candidate.vcc_save_sgpr == *explicit_key_sgpr;
           }))) {
        ++dense_missing_call_sgpr_count;
        continue;
      }

      std::vector<const ScReservoirOwnerContext *> dense_owner_contexts;
      std::unordered_set<uint64_t> dense_owner_descriptors;
      for (const LdsCheckTrapCandidate &candidate : seed_group.candidates) {
        for (uint64_t owner : candidate.site->owner_descriptor_file_offsets) {
          if (!dense_owner_descriptors.insert(owner).second)
            continue;
          const auto context = std::ranges::find(reservoir_owner_contexts, owner,
                                                 &ScReservoirOwnerContext::descriptor_file_offset);
          if (context != reservoir_owner_contexts.end() && context->liveness != nullptr)
            dense_owner_contexts.push_back(&*context);
        }
      }
      if (dense_owner_contexts.size() != dense_owner_descriptors.size()) {
        ++dense_missing_owner_context_count;
        continue;
      }
      const auto dense_host_scratch_is_dead_at_entry = [&](const Instruction &instruction) {
        std::array<RegisterRef, 3> scratch = {
            RegisterRef{RegClass::SGPR, seed_group.jump_pc_sgpr, 2u},
            RegisterRef{RegClass::SGPR, seed_group.saved_scc_sgpr, 1u},
            RegisterRef{RegClass::SGPR, use_explicit_key ? *explicit_key_sgpr : call_return_sgpr,
                        static_cast<uint8_t>(use_explicit_key ? 1u : 2u)},
        };
        for (const ScReservoirOwnerContext *owner : dense_owner_contexts) {
          // Canonical sites can have descriptors whose kernel scopes overlap
          // the access but not this donor host. Only scopes that can execute
          // the host constrain its scratch registers; a host absent from every
          // owner scope is unreachable guest text and imposes no live state.
          if (!owner->liveness->has_live_before(instruction))
            continue;
          if (std::ranges::any_of(scratch, [&](RegisterRef reg) {
                return owner->liveness->any_live_before(instruction, reg);
              }))
            return false;
        }
        return true;
      };
      const auto dense_host_return_scratch_is_dead = [&](const Instruction *boundary,
                                                         const BasicBlock &block) {
        const std::array<RegisterRef, 2> return_scratch = {
            RegisterRef{RegClass::SGPR, seed_group.jump_pc_sgpr, 2u},
            RegisterRef{RegClass::SGPR, seed_group.saved_scc_sgpr, 1u},
        };
        for (const ScReservoirOwnerContext *owner : dense_owner_contexts) {
          const bool owns_boundary = boundary != nullptr
                                         ? owner->liveness->has_live_before(*boundary)
                                         : owner->liveness->contains_block(block);
          if (!owns_boundary)
            continue;
          if (std::ranges::any_of(return_scratch, [&](RegisterRef reg) {
                return boundary != nullptr ? owner->liveness->any_live_before(*boundary, reg)
                                           : owner->liveness->any_live_out(block, reg);
              }))
            return false;
        }
        return true;
      };

      const uint64_t kernel_end = kernel->entry_text_offset + kernel->code_size;
      std::optional<uint64_t> best_host_offset;
      std::vector<uint32_t> best_displaced_words;
      uint64_t best_max_distance = std::numeric_limits<uint64_t>::max();
      size_t largest_host_prefix_words = 0;
      for (const Instruction *instruction : ordered_instructions) {
        if (instruction == nullptr)
          continue;
        const uint64_t host_offset = instruction->src_loc();
        const uint64_t host_bytes = kDenseEntryHostWords * sizeof(uint32_t);
        if (host_offset < kernel->entry_text_offset || host_offset > kernel_end ||
            host_bytes > kernel_end - host_offset ||
            std::ranges::any_of(dense_reserved_host_ranges,
                                [&](const auto &range) {
                                  return host_offset < range.second &&
                                         range.first < host_offset + host_bytes;
                                }) ||
            overlaps_supported_lds_probe(host_offset, host_bytes) ||
            overlaps_future_flat_probe(host_offset, host_bytes) ||
            overlaps_preapplied_reserved_range(options, {host_offset, host_offset + host_bytes}) ||
            !dense_host_scratch_is_dead_at_entry(*instruction)) {
          continue;
        }
        const uint64_t island_offset = host_offset + sizeof(uint32_t);
        uint64_t max_distance = 0;
        const bool reachable =
            std::ranges::all_of(seed_group.candidates, [&](const LdsCheckTrapCandidate &candidate) {
              if (!compute_sopp_branch_simm16(candidate.site->text_offset, island_offset))
                return false;
              const uint64_t distance = candidate.site->text_offset < island_offset
                                            ? island_offset - candidate.site->text_offset
                                            : candidate.site->text_offset - island_offset;
              max_distance = std::max(max_distance, distance);
              return true;
            });
        if (!reachable || max_distance >= best_max_distance)
          continue;
        std::vector<uint32_t> displaced_words;
        uint64_t cursor = host_offset;
        const auto owner_block_it = instruction_owner_blocks.find(instruction);
        if (owner_block_it == instruction_owner_blocks.end())
          continue;
        auto &host_instructions = owner_block_it->second->instructions();
        auto sequence_it = std::ranges::find_if(
            host_instructions, [&](const Instruction &entry) { return &entry == instruction; });
        while (sequence_it != host_instructions.end() &&
               displaced_words.size() < kDenseEntryHostWords) {
          const Instruction &entry = *sequence_it;
          if (entry.src_loc() != cursor || entry.size() <= 0 || entry.raw_encoding() == nullptr ||
              !is_consan_branch_relay_reservoir_instruction(entry, cursor, pristine_text, arch) ||
              displaced_words.size() + entry.size() / sizeof(uint32_t) > kDenseEntryHostWords) {
            break;
          }
          const size_t old_size = displaced_words.size();
          displaced_words.resize(old_size + entry.size() / sizeof(uint32_t));
          std::memcpy(displaced_words.data() + old_size, pristine_text.data() + cursor,
                      entry.size());
          cursor += static_cast<uint64_t>(entry.size());
          ++sequence_it;
        }
        largest_host_prefix_words = std::max(largest_host_prefix_words, displaced_words.size());
        if (displaced_words.size() != kDenseEntryHostWords)
          continue;
        const Instruction *boundary =
            sequence_it == host_instructions.end() ? nullptr : &*sequence_it;
        if (!dense_host_return_scratch_is_dead(boundary, *owner_block_it->second))
          continue;
        best_host_offset = host_offset;
        best_displaced_words = std::move(displaced_words);
        best_max_distance = max_distance;
      }
      if (!best_host_offset) {
        ++dense_missing_host_count;
        dense_largest_missing_host_prefix_words =
            std::max(dense_largest_missing_host_prefix_words, largest_host_prefix_words);
        continue;
      }

      DbiPatchPlacementPlanner tentative_planner = placement_planner;
      const uint64_t host_bytes = kDenseEntryHostWords * sizeof(uint32_t);
      if (!tentative_planner.reserve_existing_range(*best_host_offset, host_bytes)) {
        ++dense_placement_failure_count;
        continue;
      }
      const uint64_t dispatcher_offset = tentative_planner.appended_end();
      const uint64_t dispatcher_words =
          (use_explicit_key ? 0u : kDenseDispatcherSetupWords) + kDenseDispatcherHostWords +
          seed_group.candidates.size() *
              (use_explicit_key ? kDenseDispatcherHostWords : kDenseDispatcherSiteWords) +
          1u;
      if (!tentative_planner.reserve_appended_prefix(dispatcher_words * sizeof(uint32_t))) {
        ++dense_placement_failure_count;
        continue;
      }
      const uint64_t host_body_offset = tentative_planner.appended_end();
      if (!tentative_planner.reserve_appended_prefix(
              (kDenseEntryHostWords + kLdsCheckTrapIndirectJumpWords) * sizeof(uint32_t))) {
        ++dense_placement_failure_count;
        continue;
      }

      const size_t group_index = dense_call_groups.size();
      std::vector<LdsCheckTrapCandidate> planned_candidates;
      planned_candidates.reserve(seed_group.candidates.size());
      bool complete_group = true;
      for (size_t candidate_index = 0; candidate_index < seed_group.candidates.size();
           ++candidate_index) {
        LdsCheckTrapCandidate candidate = seed_group.candidates[candidate_index];
        auto placement = tentative_planner.plan_indirect_appended(
            candidate.site->text_offset, candidate.site->size,
            (candidate.requested_words + kLdsCheckTrapIndirectJumpWords) * sizeof(uint32_t));
        if (!placement) {
          complete_group = false;
          break;
        }
        candidate.placement = *placement;
        candidate.dense_call_group_index = group_index;
        if (use_explicit_key)
          candidate.dense_explicit_key = static_cast<uint16_t>(candidate_index + 1u);
        else
          candidate.dense_call_return_sgpr = call_return_sgpr;
        planned_candidates.push_back(std::move(candidate));
      }
      if (!complete_group) {
        ++dense_placement_failure_count;
        continue;
      }
      std::optional<ScSelectedSpillState> tentative_spills;
      if (std::ranges::any_of(planned_candidates, [&](const LdsCheckTrapCandidate &candidate) {
            return private_spill_plans.contains(candidate.site);
          })) {
        tentative_spills = selected_spills;
        for (const LdsCheckTrapCandidate &candidate : planned_candidates) {
          if (!try_add_selected_spill(*tentative_spills, candidate)) {
            note_selection_spill_failure(candidate.site);
            complete_group = false;
            break;
          }
        }
      }
      if (!complete_group)
        continue;

      ScDenseCallGroup group;
      group.kernel_entry_text_offset = seed_group.kernel_entry_text_offset;
      group.descriptor_file_offset = seed_group.descriptor_file_offset;
      group.entry_host_offset = *best_host_offset;
      group.displaced_host_words = std::move(best_displaced_words);
      group.dispatcher_offset = dispatcher_offset;
      group.entry_host_body_offset = host_body_offset;
      group.jump_pc_sgpr = seed_group.jump_pc_sgpr;
      group.saved_scc_sgpr = seed_group.saved_scc_sgpr;
      group.call_return_sgpr = call_return_sgpr;
      group.explicit_key_sgpr = explicit_key_sgpr;
      group.required_sgpr_count = use_explicit_key
                                      ? static_cast<uint16_t>(highest_instrumentation_sgpr + 1u)
                                      : static_cast<uint16_t>(call_return_sgpr + 2u);
      for (const LdsCheckTrapCandidate &candidate : planned_candidates)
        group.sites.push_back(candidate.site);
      dense_call_groups.push_back(std::move(group));
      dense_reserved_host_ranges.emplace_back(*best_host_offset, *best_host_offset + host_bytes);
      ++dense_selected_group_count;
      placement_planner = std::move(tentative_planner);
      if (tentative_spills)
        selected_spills = std::move(*tentative_spills);
      for (LdsCheckTrapCandidate &candidate : planned_candidates) {
        selected_sites.insert(candidate.site);
        selected_candidates.push_back(std::move(candidate));
      }
    }
    if (dense_selected_group_count < dense_seed_group_count) {
      result.warnings.emplace_back(
          "ConSan LDS check/trap dense-call planning selected " +
          std::to_string(dense_selected_group_count) + "/" +
          std::to_string(dense_seed_group_count) +
          " groups; missing_kernel=" + std::to_string(dense_missing_kernel_count) +
          " missing_host=" + std::to_string(dense_missing_host_count) +
          " missing_owner_context=" + std::to_string(dense_missing_owner_context_count) +
          " missing_call_sgpr=" + std::to_string(dense_missing_call_sgpr_count) +
          " placement_failure=" + std::to_string(dense_placement_failure_count) +
          " largest_missing_host_prefix_words=" +
          std::to_string(dense_largest_missing_host_prefix_words));
    }
  }

  auto try_select_candidate = [&](LdsCheckTrapCandidate candidate, bool allow_descriptor_growth) {
    if (selected_candidates.size() >= max_patches || selected_sites.contains(candidate.site))
      return;
    if (!allow_descriptor_growth && candidate.required_vgpr_allocation_count)
      return;
    const ConSanLdsSite &site = *candidate.site;
    std::optional<ScSelectedSpillState> tentative_spills;
    if (private_spill_plans.contains(candidate.site)) {
      tentative_spills = selected_spills;
      if (!try_add_selected_spill(*tentative_spills, candidate)) {
        note_selection_spill_failure(candidate.site);
        return;
      }
    }
    DbiPatchPlacementPlanner tentative_planner = placement_planner;
    DbiPatchPlacementRequest request;
    request.anchor_offset = site.text_offset;
    request.original_size = site.size;
    request.body_size =
        (candidate.requested_words +
         (candidate.entry_island != nullptr ? kLdsCheckTrapIndirectJumpWords : 0u)) *
        sizeof(uint32_t);
    request.inline_capacity =
        !candidate.use_local_cave && !candidate.use_appended_cave ? request.body_size : 0u;
    request.allow_appended_cave = candidate.use_appended_cave;
    if (candidate.use_local_cave && candidate.local_cave)
      request.local_cave = DbiPatchLocalCave{
          candidate.local_cave->text_offset,
          static_cast<uint64_t>(candidate.local_cave->word_count) * sizeof(uint32_t)};
    std::optional<DbiPatchPlacement> placement;
    if (candidate.entry_island != nullptr) {
      if (!candidate.indirect_pc_sgpr || !candidate.indirect_scc_save_sgpr)
        return;
      if (!candidate.entry_island_is_appended &&
          !tentative_planner.reserve_existing_range(candidate.entry_island->text_offset,
                                                    kLdsCheckTrapIndirectJumpWords *
                                                        sizeof(uint32_t))) {
        ++indirect_candidate_placement_failure_count;
        return;
      }
      placement =
          tentative_planner.plan_indirect_appended(site.text_offset, site.size, request.body_size);
      if (!placement)
        ++indirect_candidate_placement_failure_count;
    } else if (auto relay_slot = direct_body_relay_slots.find(candidate.site);
               relay_slot != direct_body_relay_slots.end() && candidate.use_appended_cave) {
      placement = tentative_planner.plan(request);
      if (!placement || relay_slot->second.empty() ||
          tentative_planner.appended_end() != relay_slot->second.front() ||
          !tentative_planner.reserve_appended_prefix(relay_slot->second.size() *
                                                     sizeof(uint32_t))) {
        placement.reset();
      }
    } else {
      placement = tentative_planner.plan(request);
    }
    if (!placement)
      return;
    if ((candidate.use_local_cave && placement->kind != DbiPatchPlacementKind::LocalCave) ||
        (candidate.use_appended_cave && placement->kind != DbiPatchPlacementKind::AppendedCave) ||
        (!candidate.use_local_cave && !candidate.use_appended_cave &&
         placement->kind != DbiPatchPlacementKind::Inline))
      return;
    candidate.placement = placement;

    if (candidate.use_local_cave) {
      if (candidate.local_cave == nullptr)
        return;
    }
    placement_planner = std::move(tentative_planner);
    if (tentative_spills)
      selected_spills = std::move(*tentative_spills);
    selected_candidates.push_back(candidate);
    selected_sites.insert(candidate.site);
  };

  for (const LdsCheckTrapCandidate &candidate : inline_candidates)
    try_select_candidate(candidate, false);
  for (const LdsCheckTrapCandidate &candidate : inline_candidates)
    try_select_candidate(candidate, true);

  const auto site_is_selected = [&](const ConSanLdsSite *site) {
    return selected_sites.contains(site);
  };
  bool needs_generated_island_bank = false;
  {
    DbiPatchPlacementPlanner simulated_planner = placement_planner;
    size_t simulated_selected_count = selected_candidates.size();
    std::unordered_set<const ConSanLdsSite *> simulated_selected_sites;
    for (const LdsCheckTrapCandidate &selected : selected_candidates)
      simulated_selected_sites.insert(selected.site);

    // Model the density-maximizing local-island phase before deciding whether
    // the subsequent direct-appended sequence will drift out of SOPP range.
    // Use a tentative planner so a failed island/body pair reserves neither
    // half of the placement.
    for (const LdsCheckTrapCandidate &candidate : appended_cave_candidates) {
      if (candidate.entry_island == nullptr || simulated_selected_sites.contains(candidate.site) ||
          simulated_selected_count >= max_patches)
        continue;
      DbiPatchPlacementPlanner tentative_planner = simulated_planner;
      if (!tentative_planner.reserve_existing_range(candidate.entry_island->text_offset,
                                                    kLdsCheckTrapIndirectJumpWords *
                                                        sizeof(uint32_t)) ||
          !tentative_planner.plan_indirect_appended(
              candidate.site->text_offset, candidate.site->size,
              (candidate.requested_words + kLdsCheckTrapIndirectJumpWords) * sizeof(uint32_t)))
        continue;
      simulated_planner = std::move(tentative_planner);
      simulated_selected_sites.insert(candidate.site);
      ++simulated_selected_count;
    }
    for (const LdsCheckTrapCandidate &candidate : appended_cave_candidates) {
      if (candidate.entry_island != nullptr || simulated_selected_sites.contains(candidate.site) ||
          simulated_selected_count >= max_patches)
        continue;
      DbiPatchPlacementRequest request;
      request.anchor_offset = candidate.site->text_offset;
      request.original_size = candidate.site->size;
      request.body_size = candidate.requested_words * sizeof(uint32_t);
      request.allow_appended_cave = true;
      if (!simulated_planner.plan(request)) {
        needs_generated_island_bank = true;
        break;
      }
      simulated_selected_sites.insert(candidate.site);
      ++simulated_selected_count;
    }
  }
  std::vector<LocalNopCave> generated_entry_islands;
  std::vector<LdsCheckTrapCandidate> generated_island_candidates;
  std::vector<ScWideCompositeBranchRelayDonor> selected_wide_composite_relay_donors;
  std::vector<ScRelayReservoir> selected_relay_reservoirs;
  generated_entry_islands.reserve(generated_island_seed_candidates.size());
  generated_island_candidates.reserve(generated_island_seed_candidates.size());
  uint64_t generated_island_bank_offset = appended_cave_offset;
  uint64_t generated_island_cursor = generated_island_bank_offset;
  uint64_t generated_relay_bank_offset = appended_cave_offset;
  size_t generated_relay_bank_word_count = 0;
  std::unordered_set<const ConSanLdsSite *> relay_routed_sites;
  bool use_relay_island_bank = local_nop_caves.empty() &&
                               indirect_island_unreachable_site_count != 0 &&
                               max_patches >= supported_candidate_count;
  if (use_relay_island_bank) {
    std::vector<LdsCheckTrapCandidate> relay_owner_seeds;
    relay_owner_seeds.reserve(generated_island_seed_candidates.size());
    for (const LdsCheckTrapCandidate &candidate : generated_island_seed_candidates) {
      if (!site_is_selected(candidate.site))
        relay_owner_seeds.push_back(candidate);
    }
    std::unordered_set<const ConSanLdsSite *> direct_sites;
    for (const LdsCheckTrapCandidate &candidate : appended_cave_candidates) {
      if (candidate.entry_island == nullptr)
        direct_sites.insert(candidate.site);
    }
    for (const LdsCheckTrapCandidate &candidate : relay_owner_seeds) {
      if (!direct_sites.contains(candidate.site))
        relay_routed_sites.insert(candidate.site);
    }

    // Place wide-host bodies at the pristine append cursor, before the much
    // larger direct SC bodies consume their SOPP reach. Each host keeps an
    // intermediate eight-byte instruction's original bytes and local return
    // in its otherwise skipped payload, creating three relay vertices while
    // preserving both normal execution paths.
    DbiPatchPlacementPlanner relay_prefix_planner = placement_planner;
    std::unordered_set<uint64_t> prefixed_anchor_offsets;
    std::unordered_set<uint64_t> prefixed_host_offsets;
    const size_t wide_prefix_limit = relay_owner_seeds.size();
    for (auto host_it = relay_wide_host_candidates.rbegin();
         host_it != relay_wide_host_candidates.rend(); ++host_it) {
      if (selected_wide_composite_relay_donors.size() >= wide_prefix_limit)
        break;
      const ScWideRelayHostCandidate &host = *host_it;
      if (prefixed_host_offsets.contains(host.host_offset))
        continue;
      DbiPatchPlacementPlanner host_planner = relay_prefix_planner;
      std::optional<DbiPatchPlacement> host_placement;
      uint32_t selected_host_words = 0;
      for (auto prefix_it = host.prefix_word_counts.rbegin();
           prefix_it != host.prefix_word_counts.rend(); ++prefix_it) {
        DbiPatchPlacementPlanner tentative_host_planner = relay_prefix_planner;
        DbiPatchPlacementRequest host_request;
        host_request.anchor_offset = host.host_offset;
        host_request.original_size = *prefix_it * sizeof(uint32_t);
        host_request.body_size = *prefix_it * sizeof(uint32_t);
        host_request.allow_appended_cave = true;
        auto placement = tentative_host_planner.plan(host_request);
        if (!placement || placement->kind != DbiPatchPlacementKind::AppendedCave)
          continue;
        host_planner = std::move(tentative_host_planner);
        host_placement = *placement;
        selected_host_words = *prefix_it;
        break;
      }
      if (!host_placement)
        continue;
      const uint64_t target_distance =
          selected_wide_composite_relay_donors.empty() ? 110000u : 65536u;
      const uint64_t target_anchor =
          host.host_offset > target_distance ? host.host_offset - target_distance : 0u;
      auto anchor_it = std::ranges::lower_bound(relay_anchor_candidates, target_anchor, {},
                                                &ScRelayAnchorCandidate::anchor_offset);
      for (; anchor_it != relay_anchor_candidates.end() &&
             anchor_it->anchor_offset < host.host_offset;
           ++anchor_it) {
        const ScRelayAnchorCandidate &anchor = *anchor_it;
        if (prefixed_anchor_offsets.contains(anchor.anchor_offset) ||
            !compute_sopp_branch_simm16(anchor.anchor_offset,
                                        host.host_offset + sizeof(uint32_t)) ||
            !compute_sopp_branch_simm16(host.host_offset + 3u * sizeof(uint32_t),
                                        anchor.anchor_offset + 2u * sizeof(uint32_t)))
          continue;
        DbiPatchPlacementPlanner tentative_planner = host_planner;
        if (!tentative_planner.reserve_existing_range(anchor.anchor_offset, 2u * sizeof(uint32_t)))
          continue;
        ScWideCompositeBranchRelayDonor wide;
        wide.anchor_offset = anchor.anchor_offset;
        wide.anchor_original_words = anchor.anchor_original_words;
        wide.host_offset = host.host_offset;
        wide.host_original_words.assign(host.host_original_words.begin(),
                                        host.host_original_words.begin() + selected_host_words);
        wide.host_placement = *host_placement;
        selected_wide_composite_relay_donors.push_back(wide);
        prefixed_anchor_offsets.insert(anchor.anchor_offset);
        prefixed_host_offsets.insert(host.host_offset);
        relay_prefix_planner = std::move(tentative_planner);
        break;
      }
    }

    const auto replay_support_prefix =
        [&](std::vector<ScWideCompositeBranchRelayDonor> &wide_donors,
            std::string *error_out) -> std::optional<DbiPatchPlacementPlanner> {
      DbiPatchPlacementPlanner planner = placement_planner;
      for (ScWideCompositeBranchRelayDonor &donor : wide_donors) {
        const uint32_t host_size =
            static_cast<uint32_t>(donor.host_original_words.size() * sizeof(uint32_t));
        DbiPatchPlacementRequest request;
        request.anchor_offset = donor.host_offset;
        request.original_size = host_size;
        request.body_size = host_size;
        request.allow_appended_cave = true;
        auto placement = planner.plan(request, error_out);
        if (!placement || placement->kind != DbiPatchPlacementKind::AppendedCave ||
            !planner.reserve_existing_range(donor.anchor_offset, 2u * sizeof(uint32_t), error_out))
          return std::nullopt;
        donor.host_placement = *placement;
      }
      return planner;
    };

    // Every support prefix changes the later direct bodies and island bank.
    // Rebuild that complete layout for every trial and let exact max flow,
    // rather than geometric capacity guesses, decide whether a reservoir is
    // useful.
    struct RelayLayoutAttempt {
      std::unordered_set<const ConSanLdsSite *> routed_sites;
      std::unordered_map<const ConSanLdsSite *, std::vector<uint64_t>> body_relay_slots;
      std::vector<LdsCheckTrapCandidate> relay_seeds;
      std::vector<uint64_t> source_offsets;
      uint64_t relay_bank_offset = 0;
      size_t relay_bank_word_count = 0;
      uint64_t island_bank_offset = 0;
      SoppBranchRelayPlan relay_plan;
    };
    const std::unordered_set<const ConSanLdsSite *> required_relay_routed_sites =
        relay_routed_sites;
    const auto evaluate_relay_layout =
        [&](const DbiPatchPlacementPlanner &prefix_planner, size_t relay_bank_word_count,
            std::string *error_out) -> std::optional<RelayLayoutAttempt> {
      std::unordered_set<const ConSanLdsSite *> routed_sites = required_relay_routed_sites;
      DbiPatchPlacementPlanner direct_layout = prefix_planner;
      std::unordered_map<const ConSanLdsSite *, std::vector<uint64_t>> body_relay_slots;
      for (;;) {
        DbiPatchPlacementPlanner direct_simulation = prefix_planner;
        std::vector<const ConSanLdsSite *> newly_routed;
        std::unordered_map<const ConSanLdsSite *, std::vector<uint64_t>> simulated_slots;
        for (const LdsCheckTrapCandidate &candidate : appended_cave_candidates) {
          if (candidate.entry_island != nullptr || routed_sites.contains(candidate.site) ||
              site_is_selected(candidate.site))
            continue;
          DbiPatchPlacementRequest request;
          request.anchor_offset = candidate.site->text_offset;
          request.original_size = candidate.site->size;
          request.body_size = candidate.requested_words * sizeof(uint32_t);
          request.allow_appended_cave = true;
          if (!direct_simulation.plan(request)) {
            newly_routed.push_back(candidate.site);
            continue;
          }
          const uint64_t relay_slot_begin = direct_simulation.appended_end();
          if (!direct_simulation.reserve_appended_prefix(sizeof(uint32_t))) {
            if (error_out != nullptr)
              *error_out = "could not reserve a direct-body relay slot";
            return std::nullopt;
          }
          std::vector<uint64_t> relay_slots;
          relay_slots.push_back(relay_slot_begin);
          simulated_slots.emplace(candidate.site, std::move(relay_slots));
        }
        if (newly_routed.empty()) {
          direct_layout = std::move(direct_simulation);
          body_relay_slots = std::move(simulated_slots);
          break;
        }
        for (const ConSanLdsSite *site : newly_routed)
          routed_sites.insert(site);
      }

      std::vector<LdsCheckTrapCandidate> relay_seeds;
      relay_seeds.reserve(routed_sites.size());
      for (const LdsCheckTrapCandidate &candidate : relay_owner_seeds) {
        if (routed_sites.contains(candidate.site))
          relay_seeds.push_back(candidate);
      }
      std::vector<uint64_t> source_offsets;
      source_offsets.reserve(relay_seeds.size());
      for (const LdsCheckTrapCandidate &candidate : relay_seeds)
        source_offsets.push_back(candidate.site->text_offset);

      std::vector<uint64_t> relay_offsets;
      size_t reservoir_slot_count = 0;
      for (const ScRelayReservoir &reservoir : selected_relay_reservoirs)
        reservoir_slot_count +=
            sc_relay_reservoir_slot_offsets(reservoir.anchor_offset, reservoir.original_size)
                .size();
      relay_offsets.reserve(relay_owner_seeds.size() + body_relay_slots.size() +
                            3u * selected_wide_composite_relay_donors.size() +
                            reservoir_slot_count);
      for (const LdsCheckTrapCandidate &candidate : relay_owner_seeds)
        relay_offsets.push_back(candidate.site->text_offset + sizeof(uint32_t));
      for (const auto &[site, relay_site_offsets] : body_relay_slots) {
        (void)site;
        relay_offsets.insert(relay_offsets.end(), relay_site_offsets.begin(),
                             relay_site_offsets.end());
      }
      for (const ScWideCompositeBranchRelayDonor &donor : selected_wide_composite_relay_donors) {
        relay_offsets.push_back(donor.anchor_offset + sizeof(uint32_t));
        for (size_t word = 4u; word < donor.host_original_words.size(); ++word)
          relay_offsets.push_back(donor.host_offset + word * sizeof(uint32_t));
      }
      for (const ScRelayReservoir &reservoir : selected_relay_reservoirs) {
        const std::vector<uint64_t> slots =
            sc_relay_reservoir_slot_offsets(reservoir.anchor_offset, reservoir.original_size);
        relay_offsets.insert(relay_offsets.end(), slots.begin(), slots.end());
      }
      const uint64_t relay_bank_offset = direct_layout.appended_end();
      if (!direct_layout.reserve_appended_prefix(relay_bank_word_count * sizeof(uint32_t),
                                                 error_out))
        return std::nullopt;
      for (size_t index = 0; index < relay_bank_word_count; ++index)
        relay_offsets.push_back(relay_bank_offset + index * sizeof(uint32_t));
      std::vector<uint64_t> island_offsets;
      island_offsets.reserve(relay_seeds.size());
      uint64_t cursor = direct_layout.appended_end();
      const uint64_t island_bank_offset = cursor;
      for (size_t index = 0; index < relay_seeds.size(); ++index) {
        island_offsets.push_back(cursor);
        cursor += kLdsCheckTrapIndirectJumpWords * sizeof(uint32_t);
      }
      const uint64_t island_bank_bytes = cursor - island_bank_offset;
      if (!direct_layout.reserve_appended_prefix(island_bank_bytes, error_out))
        return std::nullopt;
      for (ScRelayReservoir &reservoir : selected_relay_reservoirs) {
        auto placement = direct_layout.plan_indirect_appended(
            reservoir.anchor_offset, reservoir.original_size,
            reservoir.original_size + kScRelayAppendedOverheadWords * sizeof(uint32_t), error_out);
        if (!placement || placement->kind != DbiPatchPlacementKind::AppendedCave)
          return std::nullopt;
        placement->return_target = reservoir.anchor_offset + reservoir.original_size -
                                   kScRelayTailRestoreWords * sizeof(uint32_t);
        reservoir.placement = *placement;
      }
      auto plan =
          plan_forward_sopp_branch_relays(source_offsets, relay_offsets, island_offsets, error_out);
      if (!plan)
        return std::nullopt;
      return RelayLayoutAttempt{std::move(routed_sites), std::move(body_relay_slots),
                                std::move(relay_seeds),  std::move(source_offsets),
                                relay_bank_offset,       relay_bank_word_count,
                                island_bank_offset,      std::move(*plan)};
    };

    std::string relay_error;
    size_t relay_bank_word_count = 0;
    auto relay_attempt =
        evaluate_relay_layout(relay_prefix_planner, relay_bank_word_count, &relay_error);
    if (!relay_attempt) {
      result.errors.emplace_back("ConSan LDS check/trap relay planning failed: " + relay_error);
      return;
    }

    // A cut in pristine text needs more pristine vertices. Relocate a
    // straight-line wave32 range with an indirect entry/return pair. Each
    // endpoint uses VCC as the PC pair, saves VCC_LO in one endpoint-dead
    // SGPR, and uses architecturally inactive EXEC_HI to preserve SCC. Expose
    // only the skipped interior words as relay
    // capacity. Exact replay and max flow remain the acceptance test for every
    // trial.
    size_t last_ranked_reservoir_count = 0;
    size_t last_exact_reservoir_trial_count = 0;
    while (!relay_attempt->relay_plan.rejected_source_indices.empty()) {
      struct RankedReservoir {
        const ScRelayReservoirCandidate *candidate = nullptr;
        size_t cut_words = 0;
      };
      std::vector<RankedReservoir> ranked;
      constexpr uint64_t kMaximumForwardSoppBranchBytes = 131072u;
      uint64_t cut_neighborhood_begin = 0u;
      uint64_t cut_neighborhood_end = 0u;
      if (!relay_attempt->relay_plan.min_cut_relay_offsets.empty()) {
        const auto [minimum_cut_relay, maximum_cut_relay] =
            std::ranges::minmax(relay_attempt->relay_plan.min_cut_relay_offsets);
        cut_neighborhood_begin = minimum_cut_relay > kMaximumForwardSoppBranchBytes
                                     ? minimum_cut_relay - kMaximumForwardSoppBranchBytes
                                     : 0u;
        cut_neighborhood_end = maximum_cut_relay > std::numeric_limits<uint64_t>::max() -
                                                       kMaximumForwardSoppBranchBytes
                                   ? std::numeric_limits<uint64_t>::max()
                                   : maximum_cut_relay + kMaximumForwardSoppBranchBytes;
      } else {
        // A graph with no existing relay on its source-to-island frontier has
        // no saturated relay vertex to report. Seed that empty frontier from
        // the rejected sources through the reserved island bank; exact replay
        // below still decides whether any ranked reservoir adds real flow.
        cut_neighborhood_begin = std::numeric_limits<uint64_t>::max();
        for (size_t rejected : relay_attempt->relay_plan.rejected_source_indices) {
          if (rejected >= relay_attempt->source_offsets.size())
            continue;
          cut_neighborhood_begin =
              std::min(cut_neighborhood_begin, relay_attempt->source_offsets[rejected]);
        }
        if (cut_neighborhood_begin == std::numeric_limits<uint64_t>::max())
          break;
        cut_neighborhood_end =
            relay_attempt->island_bank_offset +
            relay_attempt->relay_seeds.size() * kLdsCheckTrapIndirectJumpWords * sizeof(uint32_t);
      }
      for (const ScRelayReservoirCandidate &candidate : relay_reservoir_candidates) {
        const bool already_selected =
            std::ranges::any_of(selected_relay_reservoirs, [&](const auto &selected) {
              return candidate.anchor_offset < selected.anchor_offset + selected.original_size &&
                     selected.anchor_offset < candidate.anchor_offset + candidate.original_size;
            });
        if (already_selected)
          continue;
        const uint64_t payload_begin =
            candidate.anchor_offset + kScRelayEntryWords * sizeof(uint32_t);
        const uint64_t payload_end = candidate.anchor_offset + candidate.original_size -
                                     kScRelayTailRestoreWords * sizeof(uint32_t);
        const uint64_t overlap_begin = std::max(payload_begin, cut_neighborhood_begin);
        const uint64_t overlap_end = std::min(payload_end, cut_neighborhood_end);
        const size_t cut_words =
            overlap_end > overlap_begin
                ? static_cast<size_t>((overlap_end - overlap_begin) / sizeof(uint32_t))
                : 0u;
        if (cut_words != 0)
          ranked.push_back({&candidate, cut_words});
      }
      std::ranges::sort(ranked, [](const auto &lhs, const auto &rhs) {
        if (lhs.cut_words != rhs.cut_words)
          return lhs.cut_words > rhs.cut_words;
        if (lhs.candidate->original_size != rhs.candidate->original_size)
          return lhs.candidate->original_size > rhs.candidate->original_size;
        return lhs.candidate->anchor_offset < rhs.candidate->anchor_offset;
      });
      last_ranked_reservoir_count = ranked.size();
      const size_t old_rejected = relay_attempt->relay_plan.rejected_source_indices.size();
      std::optional<RelayLayoutAttempt> best_attempt;
      std::optional<ScRelayReservoir> best_reservoir;
      DbiPatchPlacementPlanner best_prefix = relay_prefix_planner;
      const auto append_candidate_reservoir = [&](const ScRelayReservoirCandidate &candidate) {
        selected_relay_reservoirs.push_back({candidate.anchor_offset,
                                             candidate.original_size,
                                             candidate.pc_sgpr_pair,
                                             candidate.saved_vcc_sgpr,
                                             candidate.return_pc_sgpr_pair,
                                             candidate.return_saved_vcc_sgpr,
                                             candidate.required_sgpr_count,
                                             candidate.owner_descriptor_file_offsets,
                                             {}});
      };
      constexpr size_t kMaxExactReservoirTrialsPerIteration = 8u;
      size_t exact_reservoir_trials = 0;
      for (const RankedReservoir &ranked_candidate : ranked) {
        const ScRelayReservoirCandidate &candidate = *ranked_candidate.candidate;
        append_candidate_reservoir(candidate);
        std::string trial_error;
        auto trial_prefix =
            replay_support_prefix(selected_wide_composite_relay_donors, &trial_error);
        if (!trial_prefix) {
          selected_relay_reservoirs.pop_back();
          continue;
        }
        ++exact_reservoir_trials;
        last_exact_reservoir_trial_count = exact_reservoir_trials;
        auto trial = evaluate_relay_layout(*trial_prefix, 0u, &trial_error);
        ScRelayReservoir trial_reservoir = selected_relay_reservoirs.back();
        selected_relay_reservoirs.pop_back();
        if (!trial || trial->relay_plan.rejected_source_indices.size() >= old_rejected) {
          if (exact_reservoir_trials >= kMaxExactReservoirTrialsPerIteration)
            break;
          else
            continue;
        }
        if (!best_attempt || trial->relay_plan.rejected_source_indices.size() <
                                 best_attempt->relay_plan.rejected_source_indices.size()) {
          best_attempt = std::move(trial);
          best_reservoir = std::move(trial_reservoir);
          best_prefix = std::move(*trial_prefix);
          if (best_attempt->relay_plan.rejected_source_indices.empty())
            break;
        }
        if (exact_reservoir_trials >= kMaxExactReservoirTrialsPerIteration)
          break;
      }
      bool accepted_batch = false;
      if (!best_attempt) {
        const size_t old_reservoir_count = selected_relay_reservoirs.size();
        size_t batch_size = 0;
        size_t last_trial_batch_size = 0;
        size_t next_trial_size = 2u;
        // A long generated object can require a complete chain across many
        // empty SOPP bands before maximum flow increases at all. Keep the
        // exponential trials, but allow one trial to span that full chain.
        constexpr size_t kMaxBatchReservoirs = 4096u;
        for (const RankedReservoir &ranked_candidate : ranked) {
          const ScRelayReservoirCandidate &candidate = *ranked_candidate.candidate;
          const bool overlaps_batch =
              std::ranges::any_of(selected_relay_reservoirs, [&](const auto &selected) {
                return candidate.anchor_offset < selected.anchor_offset + selected.original_size &&
                       selected.anchor_offset < candidate.anchor_offset + candidate.original_size;
              });
          if (overlaps_batch)
            continue;
          append_candidate_reservoir(candidate);
          ++batch_size;
          if (batch_size < next_trial_size && batch_size < kMaxBatchReservoirs)
            continue;
          std::string trial_error;
          last_trial_batch_size = batch_size;
          auto trial_prefix =
              replay_support_prefix(selected_wide_composite_relay_donors, &trial_error);
          ++exact_reservoir_trials;
          last_exact_reservoir_trial_count = exact_reservoir_trials;
          auto trial =
              trial_prefix ? evaluate_relay_layout(*trial_prefix, 0u, &trial_error) : std::nullopt;
          if (trial && trial->relay_plan.rejected_source_indices.size() < old_rejected) {
            relay_prefix_planner = std::move(*trial_prefix);
            relay_attempt = std::move(trial);
            accepted_batch = true;
            break;
          }
          if (batch_size >= kMaxBatchReservoirs)
            break;
          next_trial_size = std::min(kMaxBatchReservoirs, next_trial_size * 2u);
        }
        if (!accepted_batch && batch_size != last_trial_batch_size) {
          std::string trial_error;
          auto trial_prefix =
              replay_support_prefix(selected_wide_composite_relay_donors, &trial_error);
          ++exact_reservoir_trials;
          last_exact_reservoir_trial_count = exact_reservoir_trials;
          auto trial =
              trial_prefix ? evaluate_relay_layout(*trial_prefix, 0u, &trial_error) : std::nullopt;
          if (trial && trial->relay_plan.rejected_source_indices.size() < old_rejected) {
            relay_prefix_planner = std::move(*trial_prefix);
            relay_attempt = std::move(trial);
            accepted_batch = true;
          }
        }
        if (!accepted_batch)
          selected_relay_reservoirs.resize(old_reservoir_count);
      }
      if (accepted_batch)
        continue;
      if (!best_attempt || !best_reservoir)
        break;
      selected_relay_reservoirs.push_back(*best_reservoir);
      relay_prefix_planner = std::move(best_prefix);
      relay_attempt = std::move(best_attempt);
    }

    if (relay_attempt->relay_plan.rejected_source_indices.empty()) {
      std::unordered_set<uint64_t> used_support_slots;
      for (const SoppBranchRelayRoute &route : relay_attempt->relay_plan.routes) {
        for (uint64_t offset : route.relay_offsets)
          used_support_slots.insert(offset);
      }
      const auto wide_is_used = [&](const ScWideCompositeBranchRelayDonor &donor) {
        if (used_support_slots.contains(donor.anchor_offset + sizeof(uint32_t)))
          return true;
        for (size_t word = 4u; word < donor.host_original_words.size(); ++word) {
          if (used_support_slots.contains(donor.host_offset + word * sizeof(uint32_t)))
            return true;
        }
        return false;
      };
      const auto reservoir_is_used = [&](const ScRelayReservoir &reservoir) {
        for (uint64_t offset :
             sc_relay_reservoir_slot_offsets(reservoir.anchor_offset, reservoir.original_size)) {
          if (used_support_slots.contains(offset))
            return true;
        }
        return false;
      };
      std::vector<ScWideCompositeBranchRelayDonor> pruned_wide;
      std::ranges::copy_if(selected_wide_composite_relay_donors, std::back_inserter(pruned_wide),
                           wide_is_used);
      std::vector<ScRelayReservoir> pruned_reservoirs;
      std::ranges::copy_if(selected_relay_reservoirs, std::back_inserter(pruned_reservoirs),
                           reservoir_is_used);
      if (pruned_wide.size() != selected_wide_composite_relay_donors.size() ||
          pruned_reservoirs.size() != selected_relay_reservoirs.size()) {
        auto original_wide = selected_wide_composite_relay_donors;
        auto original_reservoirs = selected_relay_reservoirs;
        selected_wide_composite_relay_donors = std::move(pruned_wide);
        selected_relay_reservoirs = std::move(pruned_reservoirs);
        std::string prune_error;
        auto pruned_prefix =
            replay_support_prefix(selected_wide_composite_relay_donors, &prune_error);
        auto pruned_attempt =
            pruned_prefix
                ? evaluate_relay_layout(*pruned_prefix, relay_bank_word_count, &prune_error)
                : std::nullopt;
        if (pruned_attempt && pruned_attempt->relay_plan.rejected_source_indices.empty()) {
          relay_prefix_planner = std::move(*pruned_prefix);
          relay_attempt = std::move(pruned_attempt);
        } else {
          selected_wide_composite_relay_donors = std::move(original_wide);
          selected_relay_reservoirs = std::move(original_reservoirs);
        }
      }
    }

    relay_routed_sites = relay_attempt->routed_sites;
    direct_body_relay_slots = relay_attempt->body_relay_slots;
    std::vector<LdsCheckTrapCandidate> relay_seeds = relay_attempt->relay_seeds;
    std::vector<uint64_t> source_offsets = relay_attempt->source_offsets;
    generated_relay_bank_offset = relay_attempt->relay_bank_offset;
    generated_relay_bank_word_count = relay_attempt->relay_bank_word_count;
    generated_island_bank_offset = relay_attempt->island_bank_offset;
    auto relay_plan = std::move(relay_attempt->relay_plan);

    if (!relay_plan.rejected_source_indices.empty()) {
      uint64_t min_rejected_source = std::numeric_limits<uint64_t>::max();
      uint64_t max_rejected_source = 0;
      uint64_t min_cut_from = std::numeric_limits<uint64_t>::max();
      uint64_t max_cut_from = 0;
      uint64_t min_cut_to = std::numeric_limits<uint64_t>::max();
      uint64_t max_cut_to = 0;
      uint64_t min_cut_relay = std::numeric_limits<uint64_t>::max();
      uint64_t max_cut_relay = 0;
      size_t min_cut_direct_relay_count = 0;
      size_t min_cut_owner_relay_count = 0;
      size_t min_cut_wide_relay_count = 0;
      size_t selected_wide_relay_slot_count = 0;
      size_t selected_reservoir_relay_slot_count = 0;
      size_t max_selected_wide_host_words = 0;
      for (const ScWideCompositeBranchRelayDonor &donor : selected_wide_composite_relay_donors) {
        selected_wide_relay_slot_count += donor.host_original_words.size() - 3u;
        max_selected_wide_host_words =
            std::max(max_selected_wide_host_words, donor.host_original_words.size());
      }
      for (const ScRelayReservoir &reservoir : selected_relay_reservoirs) {
        selected_reservoir_relay_slot_count +=
            sc_relay_reservoir_slot_offsets(reservoir.anchor_offset, reservoir.original_size)
                .size();
      }
      for (size_t rejected_index : relay_plan.rejected_source_indices) {
        min_rejected_source = std::min(min_rejected_source, source_offsets[rejected_index]);
        max_rejected_source = std::max(max_rejected_source, source_offsets[rejected_index]);
      }
      for (const auto &[from, to] : relay_plan.min_cut_hops) {
        min_cut_from = std::min(min_cut_from, from);
        max_cut_from = std::max(max_cut_from, from);
        min_cut_to = std::min(min_cut_to, to);
        max_cut_to = std::max(max_cut_to, to);
      }
      for (uint64_t relay : relay_plan.min_cut_relay_offsets) {
        min_cut_relay = std::min(min_cut_relay, relay);
        max_cut_relay = std::max(max_cut_relay, relay);
        for (const auto &[site, offsets] : direct_body_relay_slots) {
          (void)site;
          min_cut_direct_relay_count += std::ranges::find(offsets, relay) != offsets.end();
        }
        min_cut_owner_relay_count +=
            std::ranges::any_of(relay_owner_seeds, [&](const LdsCheckTrapCandidate &candidate) {
              return candidate.site->text_offset + sizeof(uint32_t) == relay;
            });
        min_cut_wide_relay_count +=
            std::ranges::any_of(selected_wide_composite_relay_donors,
                                [&](const ScWideCompositeBranchRelayDonor &donor) {
                                  return donor.anchor_offset + sizeof(uint32_t) == relay ||
                                         donor.host_offset + 4u * sizeof(uint32_t) == relay ||
                                         donor.host_offset + 5u * sizeof(uint32_t) == relay;
                                });
      }
      result.warnings.emplace_back(
          "ConSan LDS check/trap relay bank could not close " +
          std::to_string(relay_plan.rejected_source_indices.size()) +
          " source route(s); retaining ordinary placement; "
          "relay_anchor_candidates=" +
          std::to_string(relay_anchor_candidates.size()) + " wide_host_candidates=" +
          std::to_string(relay_wide_host_candidates.size()) + " selected_wide_composite_donors=" +
          std::to_string(selected_wide_composite_relay_donors.size()) + " rejected_source_range=" +
          std::to_string(min_rejected_source) + "-" + std::to_string(max_rejected_source) +
          " relay_reservoir_candidates=" + std::to_string(relay_reservoir_candidates.size()) +
          " selected_relay_reservoirs=" + std::to_string(selected_relay_reservoirs.size()) +
          " ranked_relay_reservoirs=" + std::to_string(last_ranked_reservoir_count) +
          " exact_relay_reservoir_trials=" + std::to_string(last_exact_reservoir_trial_count) +
          " generated_relay_bank_words=" + std::to_string(generated_relay_bank_word_count) +
          " selected_wide_relay_slots=" + std::to_string(selected_wide_relay_slot_count) +
          " selected_reservoir_relay_slots=" + std::to_string(selected_reservoir_relay_slot_count) +
          " max_selected_wide_host_words=" + std::to_string(max_selected_wide_host_words) +
          " min_cut_hops=" + std::to_string(relay_plan.min_cut_hops.size()) +
          " min_cut_from_range=" + std::to_string(min_cut_from) + "-" +
          std::to_string(max_cut_from) + " min_cut_to_range=" + std::to_string(min_cut_to) + "-" +
          std::to_string(max_cut_to) +
          " min_cut_relays=" + std::to_string(relay_plan.min_cut_relay_offsets.size()) +
          " min_cut_direct_relays=" + std::to_string(min_cut_direct_relay_count) +
          " min_cut_owner_relays=" + std::to_string(min_cut_owner_relay_count) +
          " min_cut_wide_relays=" + std::to_string(min_cut_wide_relay_count) +
          " min_cut_relay_range=" + std::to_string(min_cut_relay) + "-" +
          std::to_string(max_cut_relay));
      generated_entry_islands.clear();
      selected_wide_composite_relay_donors.clear();
      selected_relay_reservoirs.clear();
      generated_island_bank_offset = appended_cave_offset;
      generated_island_cursor = generated_island_bank_offset;
      generated_relay_bank_offset = appended_cave_offset;
      generated_relay_bank_word_count = 0;
      use_relay_island_bank = false;
    } else {
      generated_island_cursor = generated_island_bank_offset;
      for (const LdsCheckTrapCandidate &candidate : relay_seeds) {
        generated_entry_islands.push_back(LocalNopCave{
            .text_offset = generated_island_cursor,
            .file_offset = 0,
            .word_count = static_cast<uint32_t>(kLdsCheckTrapIndirectJumpWords),
            .owner_entry_text_offset = candidate.kernel_entry_text_offset,
        });
        generated_island_cursor += kLdsCheckTrapIndirectJumpWords * sizeof(uint32_t);
      }
      for (const SoppBranchRelayRoute &route : relay_plan.routes) {
        LdsCheckTrapCandidate candidate = relay_seeds[route.source_index];
        const uint64_t island_index = (route.island_offset - generated_island_bank_offset) /
                                      (kLdsCheckTrapIndirectJumpWords * sizeof(uint32_t));
        if (island_index >= generated_entry_islands.size() ||
            generated_entry_islands[island_index].text_offset != route.island_offset) {
          result.errors.emplace_back(
              "ConSan LDS check/trap relay planner returned an unknown island");
          return;
        }
        candidate.entry_island = &generated_entry_islands[island_index];
        candidate.entry_relay_offsets = route.relay_offsets;
        generated_island_candidates.push_back(std::move(candidate));
      }
      indirect_island_unreachable_site_count = 0;
    }
  }
  if (!use_relay_island_bank && needs_generated_island_bank) {
    generated_island_bank_offset = placement_planner.appended_end();
    generated_island_cursor = generated_island_bank_offset;
    for (LdsCheckTrapCandidate candidate : generated_island_seed_candidates) {
      if (site_is_selected(candidate.site) ||
          !compute_sopp_branch_simm16(candidate.site->text_offset, generated_island_cursor))
        continue;
      generated_entry_islands.push_back(LocalNopCave{
          .text_offset = generated_island_cursor,
          .file_offset = 0,
          .word_count = static_cast<uint32_t>(kLdsCheckTrapIndirectJumpWords),
          .owner_entry_text_offset = candidate.kernel_entry_text_offset,
      });
      candidate.entry_island = &generated_entry_islands.back();
      generated_island_candidates.push_back(candidate);
      generated_island_cursor += kLdsCheckTrapIndirectJumpWords * sizeof(uint32_t);
    }
  }
  const uint64_t generated_island_bank_bytes =
      generated_entry_islands.size() * kLdsCheckTrapIndirectJumpWords * sizeof(uint32_t);
  if (!use_relay_island_bank &&
      !placement_planner.reserve_appended_prefix(generated_island_bank_bytes)) {
    result.errors.emplace_back(
        "ConSan LDS check/trap proof could not reserve its entry-island bank");
    return;
  }
  std::ranges::sort(generated_island_candidates, candidate_order);

  // A long local NOP cave can hold one complete check body or several
  // seven-word entry islands whose bodies live in appended text. Prefer the
  // latter representation before consuming any cave with a full body: it is
  // the only choice that can maximize covered sites in large generated
  // objects.
  for (const LdsCheckTrapCandidate &candidate : appended_cave_candidates) {
    if (candidate.entry_island != nullptr)
      try_select_candidate(candidate, true);
  }
  for (const LdsCheckTrapCandidate &candidate : local_cave_candidates)
    try_select_candidate(candidate, false);
  for (const LdsCheckTrapCandidate &candidate : local_cave_candidates)
    try_select_candidate(candidate, true);
  if (use_relay_island_bank) {
    for (ScWideCompositeBranchRelayDonor &donor : selected_wide_composite_relay_donors) {
      const uint32_t host_size =
          static_cast<uint32_t>(donor.host_original_words.size() * sizeof(uint32_t));
      DbiPatchPlacementRequest request;
      request.anchor_offset = donor.host_offset;
      request.original_size = host_size;
      request.body_size = host_size;
      request.allow_appended_cave = true;
      auto placement = placement_planner.plan(request);
      if (!placement || placement->kind != DbiPatchPlacementKind::AppendedCave ||
          placement->body_offset != donor.host_placement.body_offset ||
          placement->return_target != donor.host_placement.return_target ||
          !placement_planner.reserve_existing_range(donor.anchor_offset, 2u * sizeof(uint32_t))) {
        result.errors.emplace_back(
            "ConSan LDS check/trap direct sequence diverged from its wide composite relay plan");
        return;
      }
      donor.host_placement = *placement;
    }
    for (const LdsCheckTrapCandidate &candidate : appended_cave_candidates) {
      if (candidate.entry_island == nullptr && !relay_routed_sites.contains(candidate.site))
        try_select_candidate(candidate, true);
    }
    if (placement_planner.appended_end() != generated_relay_bank_offset ||
        !placement_planner.reserve_appended_prefix(generated_relay_bank_word_count *
                                                   sizeof(uint32_t)) ||
        placement_planner.appended_end() != generated_island_bank_offset ||
        !placement_planner.reserve_appended_prefix(generated_island_bank_bytes)) {
      result.errors.emplace_back(
          "ConSan LDS check/trap direct sequence diverged from its relay-bank plan");
      return;
    }
    for (ScRelayReservoir &reservoir : selected_relay_reservoirs) {
      auto placement = placement_planner.plan_indirect_appended(
          reservoir.anchor_offset, reservoir.original_size,
          reservoir.original_size + kScRelayAppendedOverheadWords * sizeof(uint32_t));
      if (placement)
        placement->return_target = reservoir.anchor_offset + reservoir.original_size -
                                   kScRelayTailRestoreWords * sizeof(uint32_t);
      if (!placement || placement->kind != DbiPatchPlacementKind::AppendedCave ||
          placement->body_offset != reservoir.placement.body_offset ||
          placement->body_size != reservoir.placement.body_size ||
          placement->return_branch_offset != reservoir.placement.return_branch_offset ||
          placement->return_target != reservoir.placement.return_target) {
        result.errors.emplace_back(
            "ConSan LDS check/trap direct sequence diverged from its relay-reservoir plan");
        return;
      }
      reservoir.placement = *placement;
    }
    for (const LdsCheckTrapCandidate &candidate : generated_island_candidates)
      try_select_candidate(candidate, true);
  } else {
    for (const LdsCheckTrapCandidate &candidate : generated_island_candidates)
      try_select_candidate(candidate, true);
    for (const LdsCheckTrapCandidate &candidate : appended_cave_candidates) {
      if (candidate.entry_island == nullptr)
        try_select_candidate(candidate, true);
    }
  }

  // A spill-backed VCC-save scalar solves the data-state problem at a
  // full-pressure site, but intentionally supplies no indirect PC pair.  For
  // far appended bodies, route both control-flow edges through capacity-one
  // words in selected eight-byte anchors and, when those are insufficient,
  // the same proven straight-line relay reservoirs used by ordinary entry
  // islands.  The two chains are disjoint, preserve every ordinary return,
  // and never make a live guest scalar carry a continuation.
  if (is_rdna4_family_arch(arch) && !branch_only_candidates.empty() &&
      selected_candidates.size() < max_patches) {
    std::string branch_only_route_error;
    ConSanBranchOnlyRoutingTelemetry &routing_telemetry = result.lds_branch_only_routing_telemetry;
    std::ranges::sort(branch_only_candidates, [](const auto &lhs, const auto &rhs) {
      return lhs.site->text_offset < rhs.site->text_offset;
    });

    // Reservoir discovery emits overlapping suffixes of each admissible run.
    // Keep one maximum-size, non-overlapping candidate at each frontier so an
    // optimistic route cannot borrow two payload words whose ordinary
    // sequences could not both be relocated.
    std::vector<const ScRelayReservoirCandidate *> branch_reservoir_candidates;
    branch_reservoir_candidates.reserve(relay_reservoir_candidates.size());
    for (const ScRelayReservoirCandidate &candidate : relay_reservoir_candidates)
      branch_reservoir_candidates.push_back(&candidate);
    std::ranges::sort(branch_reservoir_candidates, [](const auto *lhs, const auto *rhs) {
      if (lhs->anchor_offset != rhs->anchor_offset)
        return lhs->anchor_offset < rhs->anchor_offset;
      return lhs->original_size > rhs->original_size;
    });
    uint64_t reservoir_frontier = 0;
    std::erase_if(branch_reservoir_candidates, [&](const auto *candidate) {
      const bool overlaps_selected =
          std::ranges::any_of(selected_relay_reservoirs, [&](const auto &selected) {
            return candidate->anchor_offset < selected.anchor_offset + selected.original_size &&
                   selected.anchor_offset < candidate->anchor_offset + candidate->original_size;
          });
      if (overlaps_selected || candidate->anchor_offset < reservoir_frontier)
        return true;
      reservoir_frontier = candidate->anchor_offset + candidate->original_size;
      return false;
    });
    // A dense object can offer tens of thousands of tiny relocatable runs.
    // Routing against all of them makes independently greedy paths choose a
    // different run at nearly every hop. Bound the candidate pool to the
    // capacity needed for two disjoint relay lanes per site in each half-range
    // window.
    constexpr uint64_t kBranchReservoirBinBytes = 65536u;
    const size_t required_slots_per_bin = 2u * branch_only_candidates.size() + 8u;
    std::unordered_map<uint64_t, size_t> reservoir_slots_by_bin;
    std::erase_if(branch_reservoir_candidates, [&](const auto *candidate) {
      const uint64_t bin = candidate->anchor_offset / kBranchReservoirBinBytes;
      size_t &slots = reservoir_slots_by_bin[bin];
      if (slots >= required_slots_per_bin)
        return true;
      slots +=
          sc_relay_reservoir_all_slot_offsets(candidate->anchor_offset, candidate->original_size)
              .size();
      return false;
    });

    DbiPatchPlacementPlanner branch_prefix_planner = placement_planner;
    std::vector<ScRelayReservoir> branch_selected_reservoirs;
    std::unordered_set<const ScRelayReservoirCandidate *> selected_branch_reservoir_candidates;
    std::unordered_set<const ScRelayReservoirCandidate *> unavailable_branch_reservoir_candidates;
    std::vector<LdsCheckTrapCandidate> accepted_candidates;
    DbiPatchPlacementPlanner accepted_planner = placement_planner;
    bool routes_complete = false;

    // Route optimistically through every non-overlapping proven reservoir,
    // then materialize exactly the reservoirs used by that route and replay.
    // Relocating a reservoir moves the appended bodies, so the replay may
    // expose one more frontier. Every noninitial replay either commits, stops,
    // or permanently classifies at least one previously optimistic candidate
    // as selected or unavailable. The defensive attempt cap admits the
    // initial pass, one promotion per finite candidate, and one final
    // materialized-only replay; correct progress never reaches it.
    const size_t convergence_attempt_limit =
        branch_reservoir_candidates.size() <= std::numeric_limits<size_t>::max() - 2u
            ? branch_reservoir_candidates.size() + 2u
            : std::numeric_limits<size_t>::max();
    [[maybe_unused]] bool convergence_terminated = false;
    for (size_t iteration = 0u; iteration < convergence_attempt_limit; ++iteration) {
      const bool initial_iteration = iteration == 0u;
      std::vector<LdsCheckTrapCandidate> tentative_candidates;
      DbiPatchPlacementPlanner tentative_planner = branch_prefix_planner;
      for (LdsCheckTrapCandidate candidate : branch_only_candidates) {
        if (selected_sites.contains(candidate.site) ||
            selected_candidates.size() + tentative_candidates.size() >= max_patches)
          continue;
        if (!candidate.placement)
          continue;
        tentative_candidates.push_back(std::move(candidate));
      }
      if (tentative_candidates.empty()) {
        convergence_terminated = true;
        break;
      }

      std::vector<uint64_t> available_relays;
      std::unordered_set<uint64_t> owned_anchor_relays;
      const auto offer_anchor_relay = [&](const LdsCheckTrapCandidate &candidate) {
        if ((candidate.use_local_cave || candidate.use_appended_cave) &&
            !candidate.dense_call_group_index && candidate.site->size == 2u * sizeof(uint32_t)) {
          const uint64_t relay = candidate.site->text_offset + sizeof(uint32_t);
          available_relays.push_back(relay);
          owned_anchor_relays.insert(relay);
        }
      };
      for (const LdsCheckTrapCandidate &candidate : selected_candidates)
        offer_anchor_relay(candidate);
      for (const LdsCheckTrapCandidate &candidate : tentative_candidates)
        offer_anchor_relay(candidate);
      for (const ScWideCompositeBranchRelayDonor &donor : selected_wide_composite_relay_donors) {
        available_relays.push_back(donor.anchor_offset + sizeof(uint32_t));
        for (size_t word = 4u; word < donor.host_original_words.size(); ++word)
          available_relays.push_back(donor.host_offset + word * sizeof(uint32_t));
      }
      for (const ScRelayReservoir &reservoir : selected_relay_reservoirs) {
        const auto slots =
            sc_relay_reservoir_all_slot_offsets(reservoir.anchor_offset, reservoir.original_size);
        available_relays.insert(available_relays.end(), slots.begin(), slots.end());
      }
      for (const ScRelayReservoir &reservoir : branch_selected_reservoirs) {
        const auto slots =
            sc_relay_reservoir_all_slot_offsets(reservoir.anchor_offset, reservoir.original_size);
        available_relays.insert(available_relays.end(), slots.begin(), slots.end());
      }
      // The ordinary appended-body layout deliberately leaves one relay word
      // after each direct body. These words bridge the otherwise empty span
      // between pristine text and late branch-only bodies just as they do for
      // the generated entry-island bank.
      for (const auto &[site, slots] : direct_body_relay_slots) {
        (void)site;
        available_relays.insert(available_relays.end(), slots.begin(), slots.end());
      }
      for (size_t index = 0; index < generated_relay_bank_word_count; ++index) {
        const uint64_t relay = generated_relay_bank_offset + index * sizeof(uint32_t);
        available_relays.push_back(relay);
      }

      std::unordered_map<uint64_t, const ScRelayReservoirCandidate *> optimistic_relay_owner;
      for (const ScRelayReservoirCandidate *candidate : branch_reservoir_candidates) {
        if (initial_iteration || selected_branch_reservoir_candidates.contains(candidate) ||
            unavailable_branch_reservoir_candidates.contains(candidate))
          continue;
        const auto slots =
            sc_relay_reservoir_all_slot_offsets(candidate->anchor_offset, candidate->original_size);
        for (uint64_t slot : slots) {
          available_relays.push_back(slot);
          optimistic_relay_owner.emplace(slot, candidate);
        }
      }
      std::ranges::sort(available_relays);
      available_relays.erase(std::ranges::unique(available_relays).begin(), available_relays.end());
      for (const LdsCheckTrapCandidate &candidate : selected_candidates) {
        for (uint64_t used : candidate.entry_relay_offsets)
          std::erase(available_relays, used);
        if (candidate.branch_only_route) {
          for (const BranchOnlyRelayClaim &claim : candidate.branch_only_route->claims)
            std::erase(available_relays, claim.offset);
        }
      }

      std::unordered_set<const ScRelayReservoirCandidate *> used_optimistic_reservoirs;
      BranchOnlyRelayRouter branch_router;
      for (uint64_t relay : available_relays) {
        const BranchOnlyRelayProvenance provenance =
            owned_anchor_relays.contains(relay) ? BranchOnlyRelayProvenance::OwnedAnchor
                                                : BranchOnlyRelayProvenance::OwnedReservoir;
        const auto optimistic_owner = optimistic_relay_owner.find(relay);
        if (optimistic_owner == optimistic_relay_owner.end()) {
          (void)branch_router.offer(relay, provenance);
        } else {
          (void)branch_router.offer(
              relay, provenance,
              BranchOnlyRelayOwnerIdentity::lds_reservoir(optimistic_owner->second->anchor_offset));
        }
      }
      std::vector<BranchOnlyRelayPairRequest> route_requests;
      route_requests.reserve(tentative_candidates.size());
      for (const LdsCheckTrapCandidate &candidate : tentative_candidates) {
        route_requests.push_back({
            .entry_source = candidate.site->text_offset,
            .entry_target = candidate.placement->body_offset,
            .return_source =
                candidate.placement->body_offset + candidate.requested_words * sizeof(uint32_t),
            .return_target = candidate.site->text_offset + candidate.site->size,
        });
      }
      std::string route_error;
      // Only retain the current convergence iteration's routing cause. A
      // later successful route must not relabel a spill-selection failure.
      branch_only_route_error.clear();
      BranchOnlyRelayBatchPlan route_plan =
          branch_router.plan_pairs(tentative_planner, route_requests, &route_error);
      record_branch_only_relay_plan(routing_telemetry, route_plan.plan_outcome(),
                                    route_plan.pair_strategies);
      if (!route_plan.complete()) {
        branch_only_route_error = route_error;
        for (BranchOnlyRelayPairRejection rejection : route_plan.rejection_reasons)
          record_branch_only_relay_rejection(routing_telemetry, rejection);
        if (route_plan.failure == BranchOnlyRelayPlanFailure::Reservation)
          record_branch_only_relay_failure(routing_telemetry, route_plan.failure);
      }
      for (const BranchOnlyRelayRoute &route : route_plan.routes) {
        for (const BranchOnlyRelayClaim &claim : route.claims) {
          if (const auto owner = optimistic_relay_owner.find(claim.offset);
              owner != optimistic_relay_owner.end())
            used_optimistic_reservoirs.insert(owner->second);
        }
      }
      const bool complete =
          route_plan.complete() && route_plan.routes.size() == tentative_candidates.size();
      bool committed = false;
      if (complete && used_optimistic_reservoirs.empty()) {
        committed = branch_router.commit(route_plan.routes, &route_error);
        if (!committed)
          ++routing_telemetry.reservation_failure_count;
      }
      if (committed) {
        for (size_t index = 0; index < tentative_candidates.size(); ++index) {
          tentative_candidates[index].branch_only_route = std::move(route_plan.routes[index]);
        }
        accepted_candidates = std::move(tentative_candidates);
        accepted_planner = std::move(tentative_planner);
        routes_complete = true;
        convergence_terminated = true;
        break;
      }
      if (used_optimistic_reservoirs.empty() && initial_iteration &&
          !branch_reservoir_candidates.empty())
        continue;
      if (used_optimistic_reservoirs.empty()) {
        convergence_terminated = true;
        break;
      }

      // Promote every optimistic owner referenced by the current complete or
      // partial plan together. Owner minimization reduces the promoted set for
      // complete plans; bounded partial plans may still expose capacity needed
      // by the next attempt. Replanning remains necessary because
      // materializing the set advances appended placement and can expose a new
      // reach frontier.
      std::vector<const ScRelayReservoirCandidate *> newly_used(used_optimistic_reservoirs.begin(),
                                                                used_optimistic_reservoirs.end());
      std::ranges::sort(newly_used, {}, &ScRelayReservoirCandidate::anchor_offset);
      bool selected_any = false;
      for (const ScRelayReservoirCandidate *candidate : newly_used) {
        auto placement = branch_prefix_planner.plan_indirect_appended(
            candidate->anchor_offset, candidate->original_size,
            candidate->original_size + kScRelayAppendedOverheadWords * sizeof(uint32_t));
        if (!placement || placement->kind != DbiPatchPlacementKind::AppendedCave) {
          unavailable_branch_reservoir_candidates.insert(candidate);
          continue;
        }
        placement->return_target = candidate->anchor_offset + candidate->original_size -
                                   kScRelayTailRestoreWords * sizeof(uint32_t);
        branch_selected_reservoirs.push_back(
            {candidate->anchor_offset, candidate->original_size, candidate->pc_sgpr_pair,
             candidate->saved_vcc_sgpr, candidate->return_pc_sgpr_pair,
             candidate->return_saved_vcc_sgpr, candidate->required_sgpr_count,
             candidate->owner_descriptor_file_offsets, *placement});
        selected_branch_reservoir_candidates.insert(candidate);
        selected_any = true;
      }
      if (!selected_any) {
        convergence_terminated = true;
        break;
      }
    }
    assert(convergence_terminated &&
           "branch-only reservoir replay must converge before its defensive limit");
    std::optional<ScSelectedSpillState> accepted_spills;
    if (routes_complete &&
        std::ranges::any_of(accepted_candidates, [&](const LdsCheckTrapCandidate &candidate) {
          return private_spill_plans.contains(candidate.site);
        })) {
      accepted_spills = selected_spills;
      for (const LdsCheckTrapCandidate &candidate : accepted_candidates) {
        if (!try_add_selected_spill(*accepted_spills, candidate)) {
          note_selection_spill_failure(candidate.site);
          routes_complete = false;
          break;
        }
      }
    }
    if (routes_complete) {
      placement_planner = std::move(accepted_planner);
      if (accepted_spills)
        selected_spills = std::move(*accepted_spills);
      selected_relay_reservoirs.insert(selected_relay_reservoirs.end(),
                                       branch_selected_reservoirs.begin(),
                                       branch_selected_reservoirs.end());
      for (LdsCheckTrapCandidate &candidate : accepted_candidates) {
        selected_sites.insert(candidate.site);
        selected_candidates.push_back(std::move(candidate));
      }
    } else {
      result.warnings.emplace_back(
          "ConSan LDS check/trap branch-only continuation could not route all spill-backed sites" +
          (branch_only_route_error.empty() ? std::string{} : ": " + branch_only_route_error));
    }
  }

  if (!selected_candidates.empty() && selected_candidates.size() < supported_candidate_count &&
      max_patches >= supported_candidate_count) {
    result.warnings.emplace_back(
        "ConSan LDS check/trap proof selected only " + std::to_string(selected_candidates.size()) +
        "/" + std::to_string(supported_candidate_count) +
        " supported sites; scratchable_candidates=" + std::to_string(scratchable_candidate_count) +
        " missing_width_plan_sites=" + std::to_string(missing_width_plan_site_count) +
        " missing_instruction_sites=" + std::to_string(missing_instruction_site_count) +
        " missing_vcc_save_sites=" + std::to_string(missing_vcc_save_site_count) +
        " missing_scratch_sites=" + std::to_string(missing_scratch_site_count) +
        " invalid_scratch_allocation_sites=" +
        std::to_string(invalid_scratch_allocation_site_count) +
        " append_cave_reachable_candidates=" +
        std::to_string(append_cave_reachable_candidate_count) +
        " indirect_pc_resource_failed_sites=" +
        std::to_string(indirect_pc_resource_failed_site_count) +
        " indirect_island_unreachable_sites=" +
        std::to_string(indirect_island_unreachable_site_count) +
        " indirect_candidate_placement_failures=" +
        std::to_string(indirect_candidate_placement_failure_count) +
        " dynamic_stack_private_spill_rejections=" +
        std::to_string(dynamic_stack_private_spill_rejection_count) +
        " private_spill_encoding_failures=" + std::to_string(private_spill_encoding_failure_count) +
        " uncovered_nop_caves=" + std::to_string(local_nop_caves.size()) +
        " local_entry_islands=" + std::to_string(local_entry_islands.size()) +
        " local_cave_reachable_candidates=" + std::to_string(local_cave_reachable_candidate_count));
  }

  if (selected_candidates.empty()) {
    if (skipped_supported_site_for_excessive_delay) {
      result.warnings.emplace_back(
          "ConSan LDS check/trap proof skipped: requested delay needs too much "
          "padding for the supported LDS site");
      return;
    }
    result.warnings.emplace_back(
        "ConSan LDS check/trap proof found no padded or local-cave-reachable "
        "ds_load_b{32,64,128}, ds_load_tr{8_b64,16_b128}, ds_load_2addr_b{32,64}, "
        "ds_load_{i,u}{8,16}, ds_load_{i,u}8_d16(_hi), ds_load_u16_d16(_hi), or "
        "ds_read_{i,u}8_d16(_hi), ds_read_u16_d16(_hi), "
        "ds_store_b{32,64,96,128}, or ds_store_2addr_b{32,64} site; supported_candidates=" +
        std::to_string(supported_candidate_count) +
        " scratchable_candidates=" + std::to_string(scratchable_candidate_count) +
        " missing_width_plan_sites=" + std::to_string(missing_width_plan_site_count) +
        " missing_instruction_sites=" + std::to_string(missing_instruction_site_count) +
        " missing_vcc_save_sites=" + std::to_string(missing_vcc_save_site_count) +
        " missing_scratch_sites=" + std::to_string(missing_scratch_site_count) +
        " invalid_scratch_allocation_sites=" +
        std::to_string(invalid_scratch_allocation_site_count) + " max_observed_padding_words=" +
        std::to_string(max_observed_padding_words) + " append_cave_reachable_candidates=" +
        std::to_string(append_cave_reachable_candidate_count) +
        " indirect_pc_resource_failed_sites=" +
        std::to_string(indirect_pc_resource_failed_site_count) +
        " indirect_island_unreachable_sites=" +
        std::to_string(indirect_island_unreachable_site_count) +
        " indirect_candidate_placement_failures=" +
        std::to_string(indirect_candidate_placement_failure_count) +
        " dynamic_stack_private_spill_rejections=" +
        std::to_string(dynamic_stack_private_spill_rejection_count) +
        " private_spill_encoding_failures=" + std::to_string(private_spill_encoding_failure_count) +
        " uncovered_nop_caves=" + std::to_string(local_nop_caves.size()) +
        " max_uncovered_nop_cave_words=" + std::to_string(max_uncovered_nop_cave_words) +
        " local_cave_reachable_candidates=" + std::to_string(local_cave_reachable_candidate_count) +
        " s_clause_blocked_candidates=" + std::to_string(s_clause_blocked_candidate_count));
    return;
  }

  struct PlannedLdsCheckTrapPatch {
    const ConSanLdsSite *site = nullptr;
    const LocalNopCave *local_cave = nullptr;
    const LocalNopCave *entry_island = nullptr;
    std::vector<uint32_t> words;
    std::vector<uint32_t> cave_words;
    std::vector<uint32_t> entry_island_words;
    std::array<uint32_t, 2> anchor_words{};
    ConSanPatchInfo info;
    bool use_local_cave = false;
    bool use_appended_cave = false;
    uint64_t descriptor_file_offset = 0;
    std::optional<uint16_t> required_vgpr_allocation_count;
    std::optional<DbiPatchPlacement> placement;
    std::vector<uint64_t> entry_relay_offsets = {};
    std::optional<BranchOnlyRelayRoute> branch_only_route = std::nullopt;
    std::vector<uint64_t> trailing_relay_offsets;
    std::optional<size_t> dense_call_group_index;
    std::optional<uint16_t> dense_call_return_sgpr;
  };

  std::vector<PlannedLdsCheckTrapPatch> planned_patches;
  planned_patches.reserve(selected_candidates.size());
  for (const LdsCheckTrapCandidate &candidate : selected_candidates) {
    const ConSanLdsSite &site = *candidate.site;
    auto words = build_lds_check_trap_words(
        original_bytes, site, arch, options, *delay_words, candidate.scratch_vgpr,
        candidate.vcc_save_sgpr, candidate.gfx1250_vgpr_msb_mode.value_or(0u), result.errors);
    if (!words)
      return;
    const auto spill = selected_spills.sequences.find(&site);
    std::optional<std::array<uint32_t, 2>> scalar_vcc_save_words;
    std::optional<std::array<uint32_t, 2>> scalar_vcc_restore_words;
    if (candidate.scalar_vcc_spill_vgpr) {
      scalar_vcc_save_words = instrumentation::build_v_writelane_b32(
          *candidate.scalar_vcc_spill_vgpr, candidate.vcc_save_sgpr, 0u, arch);
      scalar_vcc_restore_words = instrumentation::build_v_readlane_b32(
          candidate.vcc_save_sgpr, *candidate.scalar_vcc_spill_vgpr, 0u, arch);
      if (!scalar_vcc_save_words || !scalar_vcc_restore_words) {
        result.errors.emplace_back(
            "ConSan LDS check/trap could not encode its scalar VCC-save spill");
        return;
      }
    }
    if (candidate.gfx1250_vgpr_msb_mode.value_or(0u) != 0u) {
      const auto select_low = instrumentation::build_s_set_vgpr_msb_transition(
          static_cast<uint8_t>(*candidate.gfx1250_vgpr_msb_mode), 0u, arch);
      const auto restore_guest = instrumentation::build_s_set_vgpr_msb_transition(
          0u, static_cast<uint8_t>(*candidate.gfx1250_vgpr_msb_mode), arch);
      const size_t guest_prefix_words = lds_load_clobbers_address(site) ? 3u : 2u;
      if (!select_low || !restore_guest || words->size() < guest_prefix_words) {
        result.errors.emplace_back(
            "ConSan LDS check/trap proof could not preserve gfx1250 VGPR-bank mode");
        return;
      }
      std::vector<uint32_t> wrapped;
      const bool has_spill = spill != selected_spills.sequences.end();
      wrapped.reserve(
          words->size() +
          (has_spill ? spill->second.save_words.size() + spill->second.restore_words.size() + 4u
                     : 2u) +
          (scalar_vcc_save_words ? 6u : 0u));
      if (has_spill || scalar_vcc_save_words) {
        wrapped.push_back(*select_low);
        if (has_spill)
          wrapped.insert(wrapped.end(), spill->second.save_words.begin(),
                         spill->second.save_words.end());
        if (scalar_vcc_save_words)
          wrapped.insert(wrapped.end(), scalar_vcc_save_words->begin(),
                         scalar_vcc_save_words->end());
        wrapped.push_back(*restore_guest);
      }
      if (lds_load_clobbers_address(site)) {
        wrapped.push_back(*select_low);
        wrapped.push_back(words->front());
        wrapped.push_back(*restore_guest);
        wrapped.insert(wrapped.end(), words->begin() + 1u, words->begin() + guest_prefix_words);
      } else {
        wrapped.insert(wrapped.end(), words->begin(), words->begin() + guest_prefix_words);
      }
      wrapped.push_back(*select_low);
      wrapped.insert(wrapped.end(), words->begin() + guest_prefix_words, words->end());
      if (scalar_vcc_restore_words)
        wrapped.insert(wrapped.end(), scalar_vcc_restore_words->begin(),
                       scalar_vcc_restore_words->end());
      if (has_spill)
        wrapped.insert(wrapped.end(), spill->second.restore_words.begin(),
                       spill->second.restore_words.end());
      wrapped.push_back(*restore_guest);
      words = std::move(wrapped);
    } else if (spill != selected_spills.sequences.end()) {
      std::vector<uint32_t> wrapped;
      wrapped.reserve(spill->second.save_words.size() + words->size() +
                      spill->second.restore_words.size() + (scalar_vcc_save_words ? 4u : 0u));
      wrapped.insert(wrapped.end(), spill->second.save_words.begin(),
                     spill->second.save_words.end());
      if (scalar_vcc_save_words)
        wrapped.insert(wrapped.end(), scalar_vcc_save_words->begin(), scalar_vcc_save_words->end());
      wrapped.insert(wrapped.end(), words->begin(), words->end());
      if (scalar_vcc_restore_words)
        wrapped.insert(wrapped.end(), scalar_vcc_restore_words->begin(),
                       scalar_vcc_restore_words->end());
      wrapped.insert(wrapped.end(), spill->second.restore_words.begin(),
                     spill->second.restore_words.end());
      words = std::move(wrapped);
    } else if (scalar_vcc_save_words && scalar_vcc_restore_words) {
      std::vector<uint32_t> wrapped;
      wrapped.reserve(words->size() + 4u);
      wrapped.insert(wrapped.end(), scalar_vcc_save_words->begin(), scalar_vcc_save_words->end());
      wrapped.insert(wrapped.end(), words->begin(), words->end());
      wrapped.insert(wrapped.end(), scalar_vcc_restore_words->begin(),
                     scalar_vcc_restore_words->end());
      words = std::move(wrapped);
    }

    PlannedLdsCheckTrapPatch planned;
    planned.site = &site;
    planned.local_cave = candidate.local_cave;
    planned.entry_island = candidate.entry_island;
    planned.words = std::move(*words);
    planned.use_local_cave = candidate.use_local_cave;
    planned.use_appended_cave = candidate.use_appended_cave;
    planned.descriptor_file_offset = candidate.descriptor_file_offset;
    planned.required_vgpr_allocation_count = candidate.required_vgpr_allocation_count;
    planned.placement = candidate.placement;
    planned.entry_relay_offsets = candidate.entry_relay_offsets;
    planned.branch_only_route = candidate.branch_only_route;
    planned.dense_call_group_index = candidate.dense_call_group_index;
    planned.dense_call_return_sgpr = candidate.dense_call_return_sgpr;
    if (auto relay = direct_body_relay_slots.find(candidate.site);
        candidate.entry_island == nullptr && relay != direct_body_relay_slots.end()) {
      planned.trailing_relay_offsets = relay->second;
    }
    planned.info.kind =
        lds_check_trap_patch_kind(site, candidate.use_local_cave || candidate.use_appended_cave);
    planned.info.anchor_offset = site.text_offset;
    planned.info.branch_relay_offsets = planned.trailing_relay_offsets;
    planned.info.scratch_vgpr = candidate.scratch_vgpr;
    planned.info.scalar_vcc_spill_sgpr = candidate.scalar_vcc_spill_vgpr
                                             ? std::optional<uint16_t>(candidate.vcc_save_sgpr)
                                             : std::nullopt;
    planned.info.scalar_vcc_spill_vgpr = candidate.scalar_vcc_spill_vgpr;
    planned.info.scalar_vcc_spill_vgpr_count = candidate.scalar_vcc_spill_vgpr ? 1u : 0u;
    planned.info.required_sgpr_count =
        static_cast<uint16_t>(candidate.vcc_save_sgpr + sc_vcc_save_sgpr_width(arch));
    planned.info.owner_descriptor_file_offsets = site.owner_descriptor_file_offsets;
    if (planned.info.owner_descriptor_file_offsets.empty())
      planned.info.owner_descriptor_file_offsets = {candidate.descriptor_file_offset};
    if (spill != selected_spills.sequences.end()) {
      planned.info.spilled_vgpr_count = spill->second.vgpr_count;
      planned.info.required_private_segment_size = spill->second.total_private_bytes;
      if (spill->second.uses_dynamic_stack_frame) {
        planned.info.dynamic_private_segment_addend = spill->second.dynamic_frame_bytes;
      }
    }

    if (candidate.use_local_cave || candidate.use_appended_cave) {
      if (!candidate.placement) {
        result.errors.emplace_back("ConSan LDS check/trap proof selected missing placement");
        return;
      }
      if (candidate.use_local_cave && candidate.local_cave == nullptr) {
        result.errors.emplace_back("ConSan LDS check/trap proof selected missing local cave");
        return;
      }
      if (site.size != 2u * sizeof(uint32_t)) {
        result.errors.emplace_back(
            "ConSan LDS check/trap proof local cave expects an 8-byte LDS site");
        return;
      }

      const uint64_t trampoline_text_offset = candidate.placement->body_offset;
      planned.cave_words = planned.words;
      uint64_t entry_target = candidate.placement->body_offset;
      if (candidate.requires_branch_only_route) {
        if (!candidate.branch_only_route) {
          result.errors.emplace_back(
              "ConSan LDS check/trap shared router lost its branch-only route");
          return;
        }
        const BranchOnlyRelayRoute &route = *candidate.branch_only_route;
        planned.cave_words.push_back(build_s_nop(0, arch));
        entry_target = route.entry_relay_offsets.empty() ? candidate.placement->body_offset
                                                         : route.entry_relay_offsets.front();
        const uint64_t return_source =
            candidate.placement->body_offset + planned.words.size() * sizeof(uint32_t);
        const uint64_t return_target = route.return_relay_offsets.empty()
                                           ? candidate.placement->return_target
                                           : route.return_relay_offsets.front();
        const auto ret = compute_sopp_branch_simm16(return_source, return_target);
        if (!ret ||
            planned.cave_words.size() * sizeof(uint32_t) != candidate.placement->body_size) {
          result.errors.emplace_back(
              "ConSan LDS check/trap could not encode its branch-only return");
          return;
        }
        planned.cave_words.back() = build_s_branch(*ret, arch);
        planned.info.branch_only_continuation = true;
        planned.info.branch_only_entry_relay_offsets = route.entry_relay_offsets;
        planned.info.branch_only_return_relay_offsets = route.return_relay_offsets;
      } else if (candidate.entry_island != nullptr || candidate.dense_call_group_index) {
        if (!candidate.indirect_pc_sgpr || !candidate.indirect_scc_save_sgpr) {
          result.errors.emplace_back(
              "ConSan LDS check/trap proof selected missing indirect SGPR scratch");
          return;
        }
        const uint16_t required_sgpr_count = std::max<uint16_t>(
            std::max<uint16_t>(
                static_cast<uint16_t>(*candidate.indirect_pc_sgpr + 2u),
                static_cast<uint16_t>(candidate.vcc_save_sgpr + sc_vcc_save_sgpr_width(arch))),
            static_cast<uint16_t>(*candidate.indirect_scc_save_sgpr + 1u));
        planned.info.indirect_pc_sgpr = *candidate.indirect_pc_sgpr;
        planned.info.indirect_saved_scc_sgpr = *candidate.indirect_scc_save_sgpr;
        planned.info.indirect_saved_vcc_sgpr = candidate.vcc_save_sgpr;
        planned.info.indirect_return_offset =
            candidate.placement->body_offset + planned.cave_words.size() * sizeof(uint32_t);
        planned.info.indirect_required_sgpr_count =
            candidate.dense_call_return_sgpr
                ? std::max<uint16_t>(required_sgpr_count,
                                     static_cast<uint16_t>(*candidate.dense_call_return_sgpr + 2u))
                : required_sgpr_count;
        if (candidate.dense_call_group_index) {
          const ScDenseCallGroup &group = dense_call_groups[*candidate.dense_call_group_index];
          planned.info.sc_dense_dispatcher_offset = group.dispatcher_offset;
          planned.info.sc_dense_call_return_sgpr = candidate.dense_call_return_sgpr;
          planned.info.sc_dense_explicit_key_sgpr = group.explicit_key_sgpr;
        }
        if (!append_lds_check_trap_indirect_return(
                planned.cave_words, candidate.placement->body_offset,
                candidate.placement->return_target, *candidate.indirect_pc_sgpr,
                *candidate.indirect_scc_save_sgpr, arch) ||
            planned.cave_words.size() * sizeof(uint32_t) != candidate.placement->body_size) {
          result.errors.emplace_back(
              "ConSan LDS check/trap proof could not encode its long return");
          return;
        }
        if (candidate.entry_island != nullptr) {
          entry_target = candidate.entry_relay_offsets.empty()
                             ? candidate.entry_island->text_offset
                             : candidate.entry_relay_offsets.front();
          if (!append_lds_check_trap_indirect_jump(
                  planned.entry_island_words, candidate.entry_island->text_offset,
                  candidate.placement->body_offset, *candidate.indirect_pc_sgpr,
                  *candidate.indirect_scc_save_sgpr, arch) ||
              planned.entry_island_words.size() != kLdsCheckTrapIndirectJumpWords) {
            result.errors.emplace_back(
                "ConSan LDS check/trap proof could not encode its entry island");
            return;
          }
        }
      } else {
        if (planned.words.size() * sizeof(uint32_t) != candidate.placement->body_size) {
          result.errors.emplace_back(
              "ConSan LDS check/trap proof body size changed after placement");
          return;
        }
        const auto ret = compute_sopp_branch_simm16(candidate.placement->return_branch_offset,
                                                    candidate.placement->return_target);
        if (!ret) {
          result.errors.emplace_back(
              "ConSan LDS check/trap proof direct trampoline return exceeds s_branch simm16");
          return;
        }
        planned.cave_words.push_back(build_s_branch(*ret, arch));
      }
      const auto fwd = compute_sopp_branch_simm16(candidate.placement->anchor_offset, entry_target);
      if (!candidate.dense_call_group_index && !fwd) {
        result.errors.emplace_back(
            "ConSan LDS check/trap proof trampoline entry exceeds s_branch simm16");
        return;
      }
      if (candidate.use_local_cave &&
          candidate.local_cave->word_count < planned.cave_words.size()) {
        result.errors.emplace_back("ConSan LDS check/trap proof local cave is too small");
        return;
      }

      if (candidate.dense_call_group_index) {
        if (*candidate.dense_call_group_index >= dense_call_groups.size()) {
          result.errors.emplace_back("ConSan LDS check/trap proof found stale dense-call state");
          return;
        }
        const ScDenseCallGroup &group = dense_call_groups[*candidate.dense_call_group_index];
        const uint64_t island_offset = group.entry_host_offset + sizeof(uint32_t);
        if (group.explicit_key_sgpr) {
          const auto branch_delta = compute_sopp_branch_simm16(
              candidate.placement->anchor_offset + sizeof(uint32_t), island_offset);
          if (!candidate.dense_explicit_key || *candidate.dense_explicit_key == 0u ||
              *candidate.dense_explicit_key > kDenseExplicitKeyMaximumSites || !branch_delta) {
            result.errors.emplace_back(
                "ConSan LDS check/trap proof could not encode its dense-key anchor");
            return;
          }
          planned.anchor_words = {
              build_s_mov_b32(*group.explicit_key_sgpr,
                              scalar_positive_inline_u32(*candidate.dense_explicit_key), arch),
              build_s_branch(*branch_delta, arch)};
        } else {
          const auto call_delta =
              compute_sopp_branch_simm16(candidate.placement->anchor_offset, island_offset);
          const auto call = candidate.dense_call_return_sgpr && call_delta
                                ? instrumentation::build_s_call_i64(
                                      *candidate.dense_call_return_sgpr, *call_delta, arch)
                                : std::nullopt;
          if (!call) {
            result.errors.emplace_back(
                "ConSan LDS check/trap proof could not encode its dense-call anchor");
            return;
          }
          planned.anchor_words = {*call, build_s_nop(0, arch)};
        }
      } else {
        planned.anchor_words = {build_s_branch(*fwd, arch), build_s_nop(0, arch)};
      }
      const uint64_t anchor_bytes = planned.anchor_words.size() * sizeof(uint32_t);
      const uint64_t cave_bytes = planned.cave_words.size() * sizeof(uint32_t);
      if (site.file_offset > original_bytes.size() ||
          anchor_bytes > original_bytes.size() - site.file_offset ||
          (candidate.use_local_cave &&
           (candidate.local_cave->file_offset > original_bytes.size() ||
            cave_bytes > original_bytes.size() - candidate.local_cave->file_offset)) ||
          (candidate.entry_island != nullptr &&
           (candidate.entry_island->file_offset > original_bytes.size() ||
            kLdsCheckTrapIndirectJumpWords * sizeof(uint32_t) >
                original_bytes.size() - candidate.entry_island->file_offset))) {
        result.errors.emplace_back("ConSan LDS check/trap proof exceeds ELF bytes");
        return;
      }

      planned.info.trampoline_offset = trampoline_text_offset;
      planned.info.original_size = site.size;
      planned.info.trampoline_size =
          static_cast<uint32_t>(planned.cave_words.size() * sizeof(uint32_t) +
                                planned.trailing_relay_offsets.size() * sizeof(uint32_t));
    } else {
      const uint64_t patch_bytes = static_cast<uint64_t>(planned.words.size() * sizeof(uint32_t));
      if (site.file_offset > original_bytes.size() ||
          patch_bytes > original_bytes.size() - site.file_offset) {
        result.errors.emplace_back("ConSan LDS check/trap proof exceeds ELF bytes");
        return;
      }
      planned.info.trampoline_offset = site.text_offset + site.size;
      planned.info.original_size = static_cast<uint32_t>(patch_bytes);
      planned.info.trampoline_size = 0;
    }

    planned_patches.push_back(std::move(planned));
  }

  // Selection order is a policy order, not necessarily appended-layout
  // order. Emit all in-place/local patches first, then appended bodies in the
  // exact order committed by the placement planner.
  std::ranges::stable_sort(planned_patches, [](const PlannedLdsCheckTrapPatch &lhs,
                                               const PlannedLdsCheckTrapPatch &rhs) {
    if (lhs.use_appended_cave != rhs.use_appended_cave)
      return !lhs.use_appended_cave;
    if (!lhs.use_appended_cave)
      return false;
    return lhs.placement->body_offset < rhs.placement->body_offset;
  });

  std::unordered_set<uint64_t> selected_relay_slots;
  for (const PlannedLdsCheckTrapPatch &planned : planned_patches) {
    if ((planned.use_local_cave || planned.use_appended_cave) &&
        planned.site->size == 2u * sizeof(uint32_t)) {
      selected_relay_slots.insert(planned.site->text_offset + sizeof(uint32_t));
    }
    selected_relay_slots.insert(planned.trailing_relay_offsets.begin(),
                                planned.trailing_relay_offsets.end());
  }
  for (const ScWideCompositeBranchRelayDonor &donor : selected_wide_composite_relay_donors) {
    selected_relay_slots.insert(donor.anchor_offset + sizeof(uint32_t));
    for (size_t word = 4u; word < donor.host_original_words.size(); ++word)
      selected_relay_slots.insert(donor.host_offset + word * sizeof(uint32_t));
  }
  for (const ScRelayReservoir &reservoir : selected_relay_reservoirs) {
    for (uint64_t offset :
         sc_relay_reservoir_all_slot_offsets(reservoir.anchor_offset, reservoir.original_size))
      selected_relay_slots.insert(offset);
  }
  for (size_t index = 0; index < generated_relay_bank_word_count; ++index)
    selected_relay_slots.insert(generated_relay_bank_offset + index * sizeof(uint32_t));
  std::unordered_set<uint64_t> branch_only_claimed_relay_slots;
  for (const PlannedLdsCheckTrapPatch &planned : planned_patches) {
    if (!planned.branch_only_route)
      continue;
    for (const BranchOnlyRelayClaim &claim : planned.branch_only_route->claims) {
      if (!selected_relay_slots.contains(claim.offset) ||
          !branch_only_claimed_relay_slots.insert(claim.offset).second) {
        result.errors.emplace_back(
            "ConSan LDS check/trap shared router claimed an unowned or duplicate relay");
        return;
      }
    }
  }
  std::unordered_map<uint64_t, uint32_t> entry_relay_words;
  for (const PlannedLdsCheckTrapPatch &planned : planned_patches) {
    if (planned.branch_only_route) {
      if (!planned.entry_relay_offsets.empty()) {
        result.errors.emplace_back(
            "ConSan LDS check/trap patch mixes island relays with a branch-only route");
        return;
      }
      continue;
    }
    if (!planned.entry_relay_offsets.empty() && planned.entry_island == nullptr) {
      result.errors.emplace_back("ConSan LDS check/trap relay path lost its entry island");
      return;
    }
    for (size_t relay_index = 0; relay_index < planned.entry_relay_offsets.size(); ++relay_index) {
      const uint64_t relay_offset = planned.entry_relay_offsets[relay_index];
      if (!selected_relay_slots.contains(relay_offset)) {
        result.errors.emplace_back("ConSan LDS check/trap relay belongs to an unselected anchor");
        return;
      }
      const uint64_t target = relay_index + 1u < planned.entry_relay_offsets.size()
                                  ? planned.entry_relay_offsets[relay_index + 1u]
                                  : planned.entry_island->text_offset;
      const auto branch = compute_sopp_branch_simm16(relay_offset, target);
      if (!branch ||
          !entry_relay_words.emplace(relay_offset, build_s_branch(*branch, arch)).second) {
        result.errors.emplace_back(
            "ConSan LDS check/trap relay path is unreachable or overlapping");
        return;
      }
    }
  }
  for (ScRelayReservoir &reservoir : selected_relay_reservoirs) {
    reservoir.used = std::ranges::any_of(
        sc_relay_reservoir_all_slot_offsets(reservoir.anchor_offset, reservoir.original_size),
        [&](uint64_t offset) {
          return entry_relay_words.contains(offset) ||
                 branch_only_claimed_relay_slots.contains(offset);
        });
  }
  ConSanBranchOnlyReservoirTelemetry lds_reservoir_telemetry;
  for (const ScRelayReservoir &reservoir : selected_relay_reservoirs) {
    // Derive the footprint from the same structural inputs as planning and
    // emission. The final overhead word is the return-control word; keeping it
    // separate avoids depending on the placement body-size convention.
    const uint64_t retained_payload_bytes =
        reservoir.original_size +
        static_cast<uint64_t>(kScRelayAppendedOverheadWords - 1u) * sizeof(uint32_t);
    accumulate_branch_only_reservoir_telemetry(lds_reservoir_telemetry, retained_payload_bytes,
                                               sizeof(uint32_t), reservoir.used);
  }
  const bool generated_relay_bank_used =
      std::ranges::any_of(selected_relay_slots, [&](uint64_t offset) {
        const bool used =
            entry_relay_words.contains(offset) || branch_only_claimed_relay_slots.contains(offset);
        return offset >= generated_relay_bank_offset && used &&
               (offset - generated_relay_bank_offset) / sizeof(uint32_t) <
                   generated_relay_bank_word_count;
      });
  const bool uses_appended_cave = std::ranges::any_of(planned_patches,
                                                      [](const PlannedLdsCheckTrapPatch &patch) {
                                                        return patch.use_appended_cave;
                                                      }) ||
                                  !selected_relay_reservoirs.empty();
  std::vector<DescriptorVgprGrowth> descriptor_growths;
  for (const PlannedLdsCheckTrapPatch &planned : planned_patches) {
    if (planned.required_vgpr_allocation_count) {
      for (uint64_t owner : planned.info.owner_descriptor_file_offsets)
        descriptor_growths.push_back({owner, *planned.required_vgpr_allocation_count});
    }
  }
  std::vector<DescriptorSgprGrowth> descriptor_sgpr_growths;
  for (const PlannedLdsCheckTrapPatch &planned : planned_patches) {
    const uint16_t required =
        std::max(planned.info.required_sgpr_count, planned.info.indirect_required_sgpr_count);
    if (required != 0)
      for (uint64_t owner : planned.info.owner_descriptor_file_offsets)
        descriptor_sgpr_growths.push_back({owner, required});
  }
  for (const ScRelayReservoir &reservoir : selected_relay_reservoirs) {
    if (!reservoir.used)
      continue;
    for (uint64_t descriptor_file_offset : reservoir.owner_descriptor_file_offsets) {
      descriptor_sgpr_growths.push_back({descriptor_file_offset, reservoir.required_sgpr_count});
    }
  }
  std::unordered_map<uint64_t, uint32_t> descriptor_private_requirements;
  for (const PlannedLdsCheckTrapPatch &planned : planned_patches) {
    if (planned.info.required_private_segment_size != 0) {
      for (uint64_t owner : planned.info.owner_descriptor_file_offsets) {
        uint32_t &required = descriptor_private_requirements[owner];
        required = std::max(required, planned.info.required_private_segment_size);
      }
    }
  }
  if (uses_appended_cave) {
    CodeObjectPatcher patcher(code_object);
    if (!apply_descriptor_vgpr_growths_to_patcher(patcher, original_bytes, descriptor_growths, arch,
                                                  result.errors)) {
      result.patches.clear();
      return;
    }
    if (!apply_descriptor_sgpr_growths_to_patcher(patcher, descriptor_sgpr_growths, arch,
                                                  result.errors)) {
      result.patches.clear();
      return;
    }
    for (const auto &[descriptor_offset, required_private_bytes] :
         descriptor_private_requirements) {
      const std::span<const uint8_t> image = patcher.image_bytes();
      if (descriptor_offset > image.size() || sizeof(KD) > image.size() - descriptor_offset) {
        result.errors.emplace_back("ConSan LDS check/trap spill descriptor exceeds ELF bytes");
        result.patches.clear();
        return;
      }
      std::array<uint8_t, sizeof(KD)> descriptor{};
      std::memcpy(descriptor.data(), image.data() + descriptor_offset, descriptor.size());
      const SpillDescriptorUpdate update =
          update_kernel_descriptor_for_spills(descriptor, 0, required_private_bytes);
      if ((update != SpillDescriptorUpdate::Updated &&
           update != SpillDescriptorUpdate::Unchanged) ||
          !patcher.patch_kernel_descriptor(descriptor_offset, descriptor)) {
        result.errors.emplace_back("ConSan LDS check/trap could not grow its spill descriptor");
        result.patches.clear();
        return;
      }
    }
    std::span<const uint8_t> text_bytes = patcher.text_bytes();
    std::vector<uint8_t> new_text(text_bytes.begin(), text_bytes.end());
    if (new_text.size() != appended_cave_offset ||
        generated_island_bank_offset < appended_cave_offset ||
        generated_island_bank_offset > std::numeric_limits<size_t>::max() ||
        generated_island_bank_bytes >
            std::numeric_limits<size_t>::max() - generated_island_bank_offset) {
      result.errors.emplace_back("ConSan LDS check/trap proof found an invalid entry-island bank");
      result.patches.clear();
      return;
    }
    const uint32_t padding_nop = build_s_nop(0, arch);
    const size_t relay_bank_begin = static_cast<size_t>(generated_relay_bank_offset);
    const size_t relay_bank_end =
        relay_bank_begin + generated_relay_bank_word_count * sizeof(uint32_t);
    const size_t bank_begin = static_cast<size_t>(generated_island_bank_offset);
    const size_t bank_end = bank_begin + static_cast<size_t>(generated_island_bank_bytes);
    const auto append_island_bank_if_due = [&]() {
      if (new_text.size() != bank_begin)
        return;
      new_text.resize(bank_end);
      for (size_t offset = bank_begin; offset < bank_end; offset += sizeof(uint32_t))
        std::memcpy(new_text.data() + offset, &padding_nop, sizeof(padding_nop));
    };
    bool generated_relay_bank_emitted = false;
    const auto append_generated_relay_bank_if_due = [&]() {
      if (generated_relay_bank_emitted || generated_relay_bank_word_count == 0)
        return;
      if (new_text.size() != relay_bank_begin)
        return;
      new_text.resize(relay_bank_end);
      for (size_t offset = relay_bank_begin; offset < relay_bank_end; offset += sizeof(uint32_t))
        std::memcpy(new_text.data() + offset, &padding_nop, sizeof(padding_nop));
      ConSanPatchInfo info;
      info.kind = ConSanPatchKind::TrampolineBranchRelayReservoir;
      info.trampoline_offset = generated_relay_bank_offset;
      info.trampoline_size =
          static_cast<uint32_t>(generated_relay_bank_word_count * sizeof(uint32_t));
      if (generated_relay_bank_used)
        result.patches.push_back(std::move(info));
      generated_relay_bank_emitted = true;
    };
    bool wide_composite_relay_donors_emitted = false;
    const auto append_wide_composite_relay_donors_if_due = [&]() -> bool {
      if (wide_composite_relay_donors_emitted || selected_wide_composite_relay_donors.empty())
        return true;
      if (new_text.size() !=
          selected_wide_composite_relay_donors.front().host_placement.body_offset)
        return true;
      for (const ScWideCompositeBranchRelayDonor &donor : selected_wide_composite_relay_donors) {
        if (new_text.size() != donor.host_placement.body_offset)
          return false;
        const auto anchor_to_host =
            compute_sopp_branch_simm16(donor.anchor_offset, donor.host_offset + sizeof(uint32_t));
        const auto anchor_return = compute_sopp_branch_simm16(
            donor.host_offset + 3u * sizeof(uint32_t), donor.anchor_offset + 2u * sizeof(uint32_t));
        const auto host_to_body =
            compute_sopp_branch_simm16(donor.host_offset, donor.host_placement.body_offset);
        const auto host_return = compute_sopp_branch_simm16(
            donor.host_placement.return_branch_offset, donor.host_placement.return_target);
        if (!anchor_to_host || !anchor_return || !host_to_body || !host_return ||
            donor.host_original_words.size() < 6u ||
            donor.host_placement.return_target !=
                donor.host_offset + donor.host_original_words.size() * sizeof(uint32_t)) {
          result.errors.emplace_back(
              "ConSan LDS check/trap wide composite relay exceeds branch reach");
          return false;
        }
        const std::array<uint32_t, 2> anchor_words = {build_s_branch(*anchor_to_host, arch),
                                                      padding_nop};
        std::vector<uint32_t> host_words(donor.host_original_words.size(), padding_nop);
        host_words[0] = build_s_branch(*host_to_body, arch);
        host_words[1] = donor.anchor_original_words[0];
        host_words[2] = donor.anchor_original_words[1];
        host_words[3] = build_s_branch(*anchor_return, arch);
        std::memcpy(new_text.data() + donor.anchor_offset, anchor_words.data(),
                    anchor_words.size() * sizeof(uint32_t));
        std::memcpy(new_text.data() + donor.host_offset, host_words.data(),
                    host_words.size() * sizeof(uint32_t));
        append_words(new_text, donor.host_original_words);
        const uint32_t host_return_word = build_s_branch(*host_return, arch);
        append_words(new_text, std::span<const uint32_t>(&host_return_word, 1));

        ConSanPatchInfo anchor_info;
        anchor_info.kind = ConSanPatchKind::TrampolineScBranchRelayDonor;
        anchor_info.anchor_offset = donor.anchor_offset;
        anchor_info.trampoline_offset = donor.host_offset + sizeof(uint32_t);
        anchor_info.original_size = 2u * sizeof(uint32_t);
        anchor_info.trampoline_size = 3u * sizeof(uint32_t);
        result.patches.push_back(std::move(anchor_info));
        ConSanPatchInfo host_info;
        host_info.kind = ConSanPatchKind::TrampolineScBranchRelayDonor;
        host_info.anchor_offset = donor.host_offset;
        host_info.trampoline_offset = donor.host_placement.body_offset;
        host_info.original_size =
            static_cast<uint32_t>(donor.host_original_words.size() * sizeof(uint32_t));
        host_info.trampoline_size = host_info.original_size + sizeof(uint32_t);
        result.patches.push_back(std::move(host_info));
      }
      wide_composite_relay_donors_emitted = true;
      return true;
    };
    bool relay_reservoirs_emitted = false;
    const auto append_relay_reservoirs_if_due = [&]() -> bool {
      if (relay_reservoirs_emitted || selected_relay_reservoirs.empty())
        return true;
      if (new_text.size() != selected_relay_reservoirs.front().placement.body_offset)
        return true;
      for (const ScRelayReservoir &reservoir : selected_relay_reservoirs) {
        if (new_text.size() != reservoir.placement.body_offset ||
            reservoir.anchor_offset > pristine_text.size() ||
            reservoir.original_size > pristine_text.size() - reservoir.anchor_offset)
          return false;
        if (!reservoir.used) {
          new_text.resize(new_text.size() + reservoir.placement.body_size);
          for (uint64_t offset = reservoir.placement.body_offset; offset < new_text.size();
               offset += sizeof(uint32_t))
            std::memcpy(new_text.data() + offset, &padding_nop, sizeof(padding_nop));
          continue;
        }
        std::vector<uint32_t> entry_words;
        entry_words.push_back(build_s_mov_b32(reservoir.saved_vcc_sgpr, kRdna4VccLo, arch));
        if (!append_lds_check_trap_indirect_jump(entry_words, reservoir.anchor_offset,
                                                 reservoir.placement.body_offset,
                                                 reservoir.pc_sgpr_pair, kRdna4ExecHi, arch) ||
            entry_words.size() != kScRelayEntryWords ||
            reservoir.original_size < entry_words.size() * sizeof(uint32_t) ||
            reservoir.placement.return_target != reservoir.anchor_offset + reservoir.original_size -
                                                     kScRelayTailRestoreWords * sizeof(uint32_t)) {
          result.errors.emplace_back(
              "ConSan LDS check/trap could not encode its relay-reservoir entry");
          return false;
        }
        std::memcpy(new_text.data() + reservoir.anchor_offset, entry_words.data(),
                    entry_words.size() * sizeof(uint32_t));
        for (uint64_t offset = reservoir.anchor_offset + entry_words.size() * sizeof(uint32_t);
             offset < reservoir.placement.return_target; offset += sizeof(uint32_t))
          std::memcpy(new_text.data() + offset, &padding_nop, sizeof(padding_nop));
        const uint32_t restore_return_vcc =
            build_s_mov_b32(kRdna4VccLo, reservoir.return_saved_vcc_sgpr, arch);
        std::memcpy(new_text.data() + reservoir.placement.return_target, &restore_return_vcc,
                    sizeof(uint32_t));
        const uint32_t restore_entry_vcc =
            build_s_mov_b32(kRdna4VccLo, reservoir.saved_vcc_sgpr, arch);
        append_words(new_text, std::span<const uint32_t>(&restore_entry_vcc, 1u));
        new_text.insert(new_text.end(), pristine_text.begin() + reservoir.anchor_offset,
                        pristine_text.begin() + reservoir.anchor_offset + reservoir.original_size);
        std::vector<uint32_t> return_words;
        return_words.push_back(build_s_mov_b32(reservoir.return_saved_vcc_sgpr, kRdna4VccLo, arch));
        if (!append_lds_check_trap_indirect_jump(
                return_words,
                reservoir.placement.body_offset + sizeof(uint32_t) + reservoir.original_size,
                reservoir.placement.return_target, reservoir.return_pc_sgpr_pair, kRdna4ExecHi,
                arch) ||
            return_words.size() + 1u != kScRelayAppendedOverheadWords) {
          result.errors.emplace_back(
              "ConSan LDS check/trap could not encode its relay-reservoir return");
          return false;
        }
        append_words(new_text, return_words);

        ConSanPatchInfo info;
        info.kind = ConSanPatchKind::TrampolineBranchRelayReservoir;
        info.anchor_offset = reservoir.anchor_offset;
        info.trampoline_offset = reservoir.placement.body_offset;
        info.original_size = reservoir.original_size;
        info.trampoline_size =
            reservoir.original_size + kScRelayAppendedOverheadWords * sizeof(uint32_t);
        info.indirect_pc_sgpr = reservoir.pc_sgpr_pair;
        info.indirect_saved_vcc_sgpr = reservoir.saved_vcc_sgpr;
        info.indirect_return_pc_sgpr = reservoir.return_pc_sgpr_pair;
        info.indirect_return_saved_vcc_sgpr = reservoir.return_saved_vcc_sgpr;
        info.indirect_required_sgpr_count = reservoir.required_sgpr_count;
        info.owner_descriptor_file_offsets = reservoir.owner_descriptor_file_offsets;
        result.patches.push_back(std::move(info));
      }
      relay_reservoirs_emitted = true;
      return true;
    };
    size_t next_dense_group = 0;
    const auto append_dense_call_groups_if_due = [&]() -> bool {
      while (next_dense_group < dense_call_groups.size()) {
        const ScDenseCallGroup &group = dense_call_groups[next_dense_group];
        if (new_text.size() != group.dispatcher_offset)
          return true;
        const uint64_t island_offset = group.entry_host_offset + sizeof(uint32_t);
        std::vector<uint32_t> island_words;
        if (!append_lds_check_trap_indirect_jump(island_words, island_offset,
                                                 group.dispatcher_offset, group.jump_pc_sgpr,
                                                 group.saved_scc_sgpr, arch) ||
            island_words.size() != kLdsCheckTrapIndirectJumpWords) {
          result.errors.emplace_back(
              "ConSan LDS check/trap could not encode its dense-call entry host");
          return false;
        }
        const std::optional<uint32_t> host_entry =
            group.explicit_key_sgpr
                ? std::optional<uint32_t>(build_s_mov_b32(*group.explicit_key_sgpr,
                                                          scalar_positive_inline_u32(0), arch))
                : instrumentation::build_s_call_i64(group.call_return_sgpr, 0, arch);
        if (!host_entry || group.entry_host_offset > new_text.size() ||
            (1u + island_words.size()) * sizeof(uint32_t) >
                new_text.size() - group.entry_host_offset) {
          result.errors.emplace_back(
              "ConSan LDS check/trap dense-call entry host exceeds executable text");
          return false;
        }
        std::memcpy(new_text.data() + group.entry_host_offset, &*host_entry, sizeof(uint32_t));
        std::memcpy(new_text.data() + island_offset, island_words.data(),
                    island_words.size() * sizeof(uint32_t));

        std::vector<uint32_t> dispatcher_words;
        const uint16_t dispatch_key_sgpr = group.explicit_key_sgpr.value_or(group.call_return_sgpr);
        if (!group.explicit_key_sgpr) {
          const auto subtract_pc = instrumentation::build_s_sub_u32(
              group.call_return_sgpr, group.call_return_sgpr, group.jump_pc_sgpr, arch);
          if (!subtract_pc) {
            result.errors.emplace_back(
                "ConSan LDS check/trap could not encode its dense-call dispatch key");
            return false;
          }
          dispatcher_words.push_back(*subtract_pc);
          dispatcher_words.push_back(
              build_s_add_u32(group.call_return_sgpr, group.call_return_sgpr, 255u, arch));
          dispatcher_words.push_back(
              static_cast<uint32_t>(group.dispatcher_offset - island_offset));
        }

        const auto append_target = [&](uint32_t key, uint64_t target) -> bool {
          const bool explicit_key = group.explicit_key_sgpr.has_value();
          const auto compare = instrumentation::build_s_cmp_eq_u32(
              dispatch_key_sgpr, explicit_key || key == 0u ? scalar_positive_inline_u32(key) : 255u,
              arch);
          if (!compare)
            return false;
          dispatcher_words.push_back(*compare);
          if (!explicit_key && key != 0u)
            dispatcher_words.push_back(key);
          std::vector<uint32_t> matched_words;
          const uint64_t matched_offset =
              group.dispatcher_offset + (dispatcher_words.size() + 1u) * sizeof(uint32_t);
          if (!append_lds_check_trap_indirect_return(matched_words, matched_offset, target,
                                                     group.jump_pc_sgpr, group.saved_scc_sgpr,
                                                     arch) ||
              matched_words.size() != kLdsCheckTrapIndirectJumpWords)
            return false;
          const auto skip = instrumentation::build_s_cbranch_scc0(
              static_cast<int16_t>(matched_words.size()), arch);
          if (!skip)
            return false;
          dispatcher_words.push_back(*skip);
          dispatcher_words.insert(dispatcher_words.end(), matched_words.begin(),
                                  matched_words.end());
          return true;
        };
        if (!append_target(0u, group.entry_host_body_offset)) {
          result.errors.emplace_back(
              "ConSan LDS check/trap could not encode its dense-call host dispatch");
          return false;
        }
        for (size_t site_index = 0; site_index < group.sites.size(); ++site_index) {
          const ConSanLdsSite *site = group.sites[site_index];
          const auto planned =
              std::ranges::find(planned_patches, site, &PlannedLdsCheckTrapPatch::site);
          if (planned == planned_patches.end() || !planned->placement ||
              !append_target(
                  group.explicit_key_sgpr
                      ? static_cast<uint32_t>(site_index + 1u)
                      : static_cast<uint32_t>(site->text_offset + sizeof(uint32_t) - island_offset),
                  planned->placement->body_offset)) {
            result.errors.emplace_back(
                "ConSan LDS check/trap could not encode a dense-call site dispatch");
            return false;
          }
        }
        dispatcher_words.push_back(build_s_endpgm(arch));
        if (dispatcher_words.size() * sizeof(uint32_t) !=
                group.entry_host_body_offset - group.dispatcher_offset ||
            new_text.size() != group.dispatcher_offset) {
          result.errors.emplace_back(
              "ConSan LDS check/trap dense-call dispatcher changed after planning");
          return false;
        }
        append_words(new_text, dispatcher_words);

        std::vector<uint32_t> host_body = group.displaced_host_words;
        if (!append_lds_check_trap_indirect_jump(
                host_body, group.entry_host_body_offset,
                group.entry_host_offset + group.displaced_host_words.size() * sizeof(uint32_t),
                group.jump_pc_sgpr, group.saved_scc_sgpr, arch) ||
            host_body.size() !=
                group.displaced_host_words.size() + kLdsCheckTrapIndirectJumpWords) {
          result.errors.emplace_back(
              "ConSan LDS check/trap could not encode its dense-call host return");
          return false;
        }
        append_words(new_text, host_body);

        ConSanPatchInfo dispatcher_info;
        dispatcher_info.kind = ConSanPatchKind::TrampolineScDenseCallDispatcher;
        dispatcher_info.anchor_offset = group.sites.front()->text_offset;
        dispatcher_info.trampoline_offset = group.dispatcher_offset;
        dispatcher_info.trampoline_size =
            static_cast<uint32_t>(dispatcher_words.size() * sizeof(uint32_t));
        dispatcher_info.indirect_pc_sgpr = group.jump_pc_sgpr;
        dispatcher_info.indirect_saved_scc_sgpr = group.saved_scc_sgpr;
        if (!group.explicit_key_sgpr)
          dispatcher_info.indirect_return_pc_sgpr = group.call_return_sgpr;
        dispatcher_info.indirect_required_sgpr_count = group.required_sgpr_count;
        dispatcher_info.sc_dense_dispatcher_offset = group.dispatcher_offset;
        if (group.explicit_key_sgpr)
          dispatcher_info.sc_dense_explicit_key_sgpr = group.explicit_key_sgpr;
        else
          dispatcher_info.sc_dense_call_return_sgpr = group.call_return_sgpr;
        dispatcher_info.owner_descriptor_file_offsets = {group.descriptor_file_offset};
        result.patches.push_back(std::move(dispatcher_info));
        ConSanPatchInfo host_info;
        host_info.kind = ConSanPatchKind::TrampolineScDenseEntryHost;
        host_info.anchor_offset = group.entry_host_offset;
        host_info.trampoline_offset = group.entry_host_body_offset;
        host_info.original_size =
            static_cast<uint32_t>(group.displaced_host_words.size() * sizeof(uint32_t));
        host_info.trampoline_size = static_cast<uint32_t>(host_body.size() * sizeof(uint32_t));
        host_info.indirect_pc_sgpr = group.jump_pc_sgpr;
        host_info.indirect_saved_scc_sgpr = group.saved_scc_sgpr;
        if (!group.explicit_key_sgpr)
          host_info.indirect_return_pc_sgpr = group.call_return_sgpr;
        host_info.indirect_required_sgpr_count = group.required_sgpr_count;
        host_info.sc_dense_dispatcher_offset = group.dispatcher_offset;
        if (group.explicit_key_sgpr)
          host_info.sc_dense_explicit_key_sgpr = group.explicit_key_sgpr;
        else
          host_info.sc_dense_call_return_sgpr = group.call_return_sgpr;
        host_info.owner_descriptor_file_offsets = {group.descriptor_file_offset};
        result.patches.push_back(std::move(host_info));
        ++next_dense_group;
      }
      return true;
    };
    for (const PlannedLdsCheckTrapPatch &planned : planned_patches) {
      const ConSanLdsSite &site = *planned.site;
      if (planned.use_local_cave) {
        std::memcpy(new_text.data() + site.text_offset, planned.anchor_words.data(),
                    planned.anchor_words.size() * sizeof(uint32_t));
        std::memcpy(new_text.data() + planned.local_cave->text_offset, planned.cave_words.data(),
                    planned.cave_words.size() * sizeof(uint32_t));
      } else if (planned.use_appended_cave) {
        if (!append_dense_call_groups_if_due()) {
          result.patches.clear();
          return;
        }
        if (!append_wide_composite_relay_donors_if_due()) {
          result.patches.clear();
          return;
        }
        append_generated_relay_bank_if_due();
        append_island_bank_if_due();
        if (!append_relay_reservoirs_if_due()) {
          result.patches.clear();
          return;
        }
        if (!planned.placement || new_text.size() != planned.placement->body_offset) {
          const uint64_t expected_offset = planned.placement ? planned.placement->body_offset : 0u;
          result.errors.emplace_back(
              "ConSan LDS check/trap proof emitted a stale appended-cave mapping: site=" +
              std::to_string(site.text_offset) + " expected=" + std::to_string(expected_offset) +
              " actual=" + std::to_string(new_text.size()));
          result.patches.clear();
          return;
        }
        if (planned.entry_island != nullptr) {
          const uint64_t island_bytes = planned.entry_island_words.size() * sizeof(uint32_t);
          if (island_bytes != kLdsCheckTrapIndirectJumpWords * sizeof(uint32_t) ||
              planned.entry_island->text_offset > new_text.size() ||
              island_bytes > new_text.size() - planned.entry_island->text_offset) {
            result.errors.emplace_back(
                "ConSan LDS check/trap proof emitted an invalid entry-island mapping");
            result.patches.clear();
            return;
          }
          std::memcpy(new_text.data() + planned.entry_island->text_offset,
                      planned.entry_island_words.data(), island_bytes);
        }
        std::memcpy(new_text.data() + site.text_offset, planned.anchor_words.data(),
                    planned.anchor_words.size() * sizeof(uint32_t));
        append_words(new_text, planned.cave_words);
        for (uint64_t trailing_relay_offset : planned.trailing_relay_offsets) {
          if (new_text.size() != trailing_relay_offset) {
            result.errors.emplace_back(
                "ConSan LDS check/trap emitted a stale direct-body relay slot: site=" +
                std::to_string(site.text_offset) +
                " expected=" + std::to_string(trailing_relay_offset) +
                " actual=" + std::to_string(new_text.size()));
            result.patches.clear();
            return;
          }
          append_words(new_text, std::span<const uint32_t>(&padding_nop, 1));
        }
      } else {
        std::memcpy(new_text.data() + site.text_offset, planned.words.data(),
                    planned.words.size() * sizeof(uint32_t));
      }
      if (planned.entry_island != nullptr) {
        ConSanPatchInfo island_info;
        island_info.kind = ConSanPatchKind::TrampolineScIndirectBranchIsland;
        island_info.anchor_offset = site.text_offset;
        island_info.trampoline_offset = planned.entry_island->text_offset;
        island_info.trampoline_size =
            static_cast<uint32_t>(kLdsCheckTrapIndirectJumpWords * sizeof(uint32_t));
        result.patches.push_back(std::move(island_info));
      }
      result.patches.push_back(planned.info);
    }
    if (!append_dense_call_groups_if_due()) {
      result.patches.clear();
      return;
    }
    if (!append_wide_composite_relay_donors_if_due()) {
      result.patches.clear();
      return;
    }
    append_generated_relay_bank_if_due();
    append_island_bank_if_due();
    if (!append_relay_reservoirs_if_due()) {
      result.patches.clear();
      return;
    }
    for (const PlannedLdsCheckTrapPatch &planned : planned_patches) {
      if (!planned.branch_only_route)
        continue;
      if (!planned.placement) {
        result.errors.emplace_back(
            "ConSan LDS check/trap shared router lost its appended placement");
        result.patches.clear();
        return;
      }
      std::string relay_error;
      if (!BranchOnlyRelayRouter::emit_and_record(
              new_text, *planned.branch_only_route, planned.placement->body_offset,
              planned.site->text_offset + planned.site->size, arch, result.patches, &relay_error)) {
        result.errors.emplace_back("ConSan LDS check/trap " + relay_error);
        result.patches.clear();
        return;
      }
    }
    for (const auto &[relay_offset, relay_word] : entry_relay_words) {
      if (relay_offset > new_text.size() || sizeof(relay_word) > new_text.size() - relay_offset) {
        result.errors.emplace_back("ConSan LDS check/trap relay exceeds executable text");
        result.patches.clear();
        return;
      }
      std::memcpy(new_text.data() + relay_offset, &relay_word, sizeof(relay_word));
    }
    if (!replace_consan_text(patcher, new_text, options, "LDS check/trap proof", result)) {
      result.patches.clear();
      return;
    }
    const std::span<const uint8_t> patched_image = patcher.image_bytes();
    AmdGpuCodeObject patched_probe(patched_image.data(), patched_image.size());
    if (!patched_probe.is_valid()) {
      result.errors.emplace_back(
          "ConSan LDS check/trap text replacement produced an invalid descriptor image");
      result.patches.clear();
      return;
    }
    for (const auto &[descriptor_offset, required_private_bytes] :
         descriptor_private_requirements) {
      const auto original_kernel = std::ranges::find(result.kernels, descriptor_offset,
                                                     &ConSanKernelInfo::descriptor_file_offset);
      const auto moved_kernel =
          original_kernel == result.kernels.end()
              ? patched_probe.kernels().end()
              : std::ranges::find(patched_probe.kernels(), original_kernel->name,
                                  &AmdGpuKernelInfo::name);
      if (moved_kernel == patched_probe.kernels().end()) {
        result.errors.emplace_back(
            "ConSan LDS check/trap text replacement lost a spill owner descriptor");
        result.patches.clear();
        return;
      }
      std::vector<uint8_t> descriptor_probe(patched_image.begin(), patched_image.end());
      const SpillDescriptorUpdate update = update_kernel_descriptor_for_spills(
          descriptor_probe, moved_kernel->descriptor_file_offset, required_private_bytes);
      if (update != SpillDescriptorUpdate::Unchanged) {
        result.errors.emplace_back(
            "ConSan LDS check/trap text replacement lost its spill descriptor update at offset " +
            std::to_string(descriptor_offset));
        result.patches.clear();
        return;
      }
    }
    result.elf_bytes = std::move(patcher).emit();
    result.lds_relay_reservoir_telemetry = lds_reservoir_telemetry;
    result.modified = true;
    return;
  }

  if (result.elf_bytes.empty())
    result.elf_bytes.assign(original_bytes.begin(), original_bytes.end());
  if (!apply_descriptor_vgpr_growths_to_bytes(result.elf_bytes, descriptor_growths, arch,
                                              result.errors)) {
    result.patches.clear();
    return;
  }
  if (!apply_descriptor_sgpr_growths_to_bytes(result.elf_bytes, descriptor_sgpr_growths, arch,
                                              result.errors)) {
    result.patches.clear();
    return;
  }
  for (const auto &[descriptor_offset, required_private_bytes] : descriptor_private_requirements) {
    const SpillDescriptorUpdate update = update_kernel_descriptor_for_spills(
        result.elf_bytes, descriptor_offset, required_private_bytes);
    if (update != SpillDescriptorUpdate::Updated && update != SpillDescriptorUpdate::Unchanged) {
      result.errors.emplace_back("ConSan LDS check/trap could not grow its spill descriptor");
      result.patches.clear();
      return;
    }
  }
  for (const PlannedLdsCheckTrapPatch &planned : planned_patches) {
    const ConSanLdsSite &site = *planned.site;
    if (planned.use_local_cave) {
      const uint64_t anchor_bytes = planned.anchor_words.size() * sizeof(uint32_t);
      const uint64_t cave_bytes = planned.cave_words.size() * sizeof(uint32_t);
      std::memcpy(result.elf_bytes.data() + site.file_offset, planned.anchor_words.data(),
                  static_cast<size_t>(anchor_bytes));
      std::memcpy(result.elf_bytes.data() + planned.local_cave->file_offset,
                  planned.cave_words.data(), static_cast<size_t>(cave_bytes));
    } else {
      const uint64_t patch_bytes = static_cast<uint64_t>(planned.words.size() * sizeof(uint32_t));
      std::memcpy(result.elf_bytes.data() + site.file_offset, planned.words.data(),
                  static_cast<size_t>(patch_bytes));
    }
    result.patches.push_back(planned.info);
  }
  result.modified = true;
}
