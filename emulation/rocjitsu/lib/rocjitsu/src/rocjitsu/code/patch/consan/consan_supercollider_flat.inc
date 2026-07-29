[[nodiscard]] uint16_t flat_check_trap_comparison_scratch_vgprs(const ConSanFlatSite &site) {
  if (const auto load = consan_flat_load_subword_semantics(site.mnemonic))
    return load->placement == ConSanFlatSubwordPlacement::High16 ? 1u : 0u;
  if (const auto store = consan_flat_store_subword_semantics(site.mnemonic))
    return store->placement == ConSanFlatSubwordPlacement::High16 || store->memory_width_bits == 8u
               ? 1u
               : 0u;
  return 0u;
}

[[nodiscard]] uint32_t flat_check_trap_comparison_setup_words(const ConSanFlatSite &site) {
  if (const auto load = consan_flat_load_subword_semantics(site.mnemonic))
    return load->placement == ConSanFlatSubwordPlacement::High16 ? 2u : 0u;
  if (const auto store = consan_flat_store_subword_semantics(site.mnemonic)) {
    return (store->placement == ConSanFlatSubwordPlacement::High16 ? 1u : 0u) +
           (store->memory_width_bits == 8u ? 2u : 0u);
  }
  return 0u;
}

[[nodiscard]] std::optional<uint32_t>
build_flat_check_trap_vcc_move(uint16_t dst_sgpr, uint16_t src_sgpr, rj_code_arch_t arch) {
  return instrumentation::build_s_mov_b64(dst_sgpr, src_sgpr, arch);
}

[[nodiscard]] bool flat_check_trap_sgpr_ranges_overlap(uint16_t lhs_base, uint16_t lhs_width,
                                                       uint16_t rhs_base, uint16_t rhs_width) {
  return sc_sgpr_ranges_overlap(lhs_base, lhs_width, rhs_base, rhs_width);
}

[[nodiscard]] std::optional<uint16_t>
flat_check_trap_delay_source_sgpr(const ConSanOptions &options) {
  return sc_delay_source_sgpr(options);
}

[[nodiscard]] bool flat_check_trap_overlaps_delay_source(uint16_t base, uint16_t width,
                                                         const ConSanOptions &options) {
  const auto delay_source = flat_check_trap_delay_source_sgpr(options);
  return delay_source && flat_check_trap_sgpr_ranges_overlap(base, width, *delay_source, 1u);
}

[[nodiscard]] std::optional<uint16_t> choose_flat_check_trap_vcc_save_sgpr(
    const Instruction *inst, std::span<const LivenessAnalysis *const> owner_liveness,
    std::optional<uint16_t> min_preferred_sgpr, const ConSanOptions &options,
    bool preserve_dynamic_stack_abi) {
  if (inst == nullptr || owner_liveness.empty() ||
      std::ranges::any_of(owner_liveness, [](const auto *liveness) { return liveness == nullptr; }))
    return std::nullopt;
  const auto find_pair = [&](uint16_t start) -> std::optional<uint16_t> {
    while (auto pair = owner_liveness.front()->find_free_sgpr_pair(inst, start)) {
      const bool overlaps_dynamic_stack =
          preserve_dynamic_stack_abi &&
          flat_check_trap_sgpr_ranges_overlap(*pair, 2u, kDynamicStackTopSgpr, 2u);
      const RegisterRef ref{RegClass::SGPR, *pair, 2u};
      const bool dead_for_every_owner =
          std::ranges::all_of(owner_liveness, [&](const LivenessAnalysis *liveness) {
            return !liveness->is_live_before(*inst, ref);
          });
      if (!flat_check_trap_overlaps_delay_source(*pair, 2u, options) && !overlaps_dynamic_stack &&
          dead_for_every_owner)
        return pair;
      if (*pair + 2u >= REGISTER_SET_ALLOCATABLE_SGPRS)
        return std::nullopt;
      start = static_cast<uint16_t>(*pair + 2u);
    }
    return std::nullopt;
  };
  if (min_preferred_sgpr) {
    if (auto high_pair = find_pair(*min_preferred_sgpr))
      return high_pair;
  }
  return find_pair(0);
}

[[nodiscard]] std::optional<uint16_t> choose_flat_check_trap_scalar_spill_pair(
    uint16_t allocated_sgpr_count, const ConSanOptions &options, bool preserve_dynamic_stack_abi) {
  // The caller spills and restores the returned live pair, so any ordinary
  // allocated pair is legal except the active delay source. Staying within
  // the descriptor allocation (and every owner's common minimum for shared
  // code) avoids growing an owner around a site-local choice.
  const uint16_t limit = std::min<uint16_t>(allocated_sgpr_count, REGISTER_SET_ALLOCATABLE_SGPRS);
  for (uint16_t pair = 0; static_cast<uint32_t>(pair) + 2u <= limit; pair += 2u) {
    const bool overlaps_dynamic_stack =
        preserve_dynamic_stack_abi &&
        flat_check_trap_sgpr_ranges_overlap(pair, 2u, kDynamicStackTopSgpr, 2u);
    if (!flat_check_trap_overlaps_delay_source(pair, 2u, options) && !overlaps_dynamic_stack)
      return pair;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<LdsCheckTrapIndirectScalarScratch>
choose_flat_check_trap_indirect_scalar_scratch(
    const Instruction *inst, std::span<const LivenessAnalysis *const> owner_liveness,
    std::optional<uint16_t> min_preferred_sgpr, const ConSanOptions &options) {
  if (inst == nullptr || owner_liveness.empty() ||
      std::ranges::any_of(owner_liveness, [](const auto *liveness) { return liveness == nullptr; }))
    return std::nullopt;

  const auto dead_for_every_owner = [&](uint16_t base, uint8_t width) {
    const RegisterRef ref{RegClass::SGPR, base, width};
    return std::ranges::all_of(owner_liveness, [&](const LivenessAnalysis *liveness) {
      return !liveness->is_live_before(*inst, ref);
    });
  };
  const auto search = [&](uint16_t start) -> std::optional<LdsCheckTrapIndirectScalarScratch> {
    uint16_t vcc_start = start;
    while (auto vcc_pair = owner_liveness.front()->find_free_sgpr_pair(inst, vcc_start)) {
      if (flat_check_trap_overlaps_delay_source(*vcc_pair, 2u, options) ||
          !dead_for_every_owner(*vcc_pair, 2u)) {
        if (*vcc_pair + 2u >= REGISTER_SET_ALLOCATABLE_SGPRS)
          return std::nullopt;
        vcc_start = static_cast<uint16_t>(*vcc_pair + 2u);
        continue;
      }

      uint16_t pc_start = start;
      while (auto pc_pair = owner_liveness.front()->find_free_sgpr_pair(inst, pc_start)) {
        if (flat_check_trap_sgpr_ranges_overlap(*pc_pair, 2u, *vcc_pair, 2u) ||
            flat_check_trap_overlaps_delay_source(*pc_pair, 2u, options) ||
            !dead_for_every_owner(*pc_pair, 2u)) {
          if (*pc_pair + 2u >= REGISTER_SET_ALLOCATABLE_SGPRS)
            break;
          pc_start = static_cast<uint16_t>(*pc_pair + 2u);
          continue;
        }

        uint16_t scc_start = start;
        while (auto scc_save = owner_liveness.front()->find_free_sgpr(inst, scc_start)) {
          if (!flat_check_trap_sgpr_ranges_overlap(*scc_save, 1u, *vcc_pair, 2u) &&
              !flat_check_trap_sgpr_ranges_overlap(*scc_save, 1u, *pc_pair, 2u) &&
              !flat_check_trap_overlaps_delay_source(*scc_save, 1u, options) &&
              dead_for_every_owner(*scc_save, 1u)) {
            return LdsCheckTrapIndirectScalarScratch{*vcc_pair, *pc_pair, *scc_save};
          }
          if (*scc_save + 1u >= REGISTER_SET_ALLOCATABLE_SGPRS)
            break;
          scc_start = static_cast<uint16_t>(*scc_save + 1u);
        }

        if (*pc_pair + 2u >= REGISTER_SET_ALLOCATABLE_SGPRS)
          break;
        pc_start = static_cast<uint16_t>(*pc_pair + 2u);
      }

      if (*vcc_pair + 2u >= REGISTER_SET_ALLOCATABLE_SGPRS)
        break;
      vcc_start = static_cast<uint16_t>(*vcc_pair + 2u);
    }
    return std::nullopt;
  };

  if (min_preferred_sgpr) {
    if (auto high_scratch = search(*min_preferred_sgpr))
      return high_scratch;
  }
  return search(0);
}

[[nodiscard]] std::optional<uint16_t> choose_flat_check_trap_scratch_vgpr(
    const ConSanFlatSite &site, const ConSanOptions &options, const Instruction *inst,
    std::span<const LivenessAnalysis *const> owner_liveness,
    std::optional<uint16_t> min_auto_scratch_vgpr, std::optional<uint16_t> max_auto_scratch_vgpr,
    uint16_t required_vgprs) {
  if (options.scratch_vgpr) {
    return choose_flat_scratch_vgpr(site, options, inst, nullptr, min_auto_scratch_vgpr,
                                    max_auto_scratch_vgpr, required_vgprs);
  }
  if (inst == nullptr || owner_liveness.empty() ||
      std::ranges::any_of(owner_liveness,
                          [](const auto *liveness) { return liveness == nullptr; }) ||
      required_vgprs > std::numeric_limits<uint8_t>::max())
    return std::nullopt;

  uint16_t search_start =
      std::max(flat_scratch_search_start(site), min_auto_scratch_vgpr.value_or(0));
  while (search_start < REGISTER_SET_MAX_VGPRS) {
    const auto candidate =
        owner_liveness.front()->find_free_run(inst, required_vgprs, search_start);
    if (!candidate)
      return std::nullopt;
    const uint32_t candidate_end = static_cast<uint32_t>(*candidate) + required_vgprs;
    if (max_auto_scratch_vgpr && candidate_end > *max_auto_scratch_vgpr)
      return std::nullopt;
    const RegisterRef ref{RegClass::VGPR, *candidate, static_cast<uint8_t>(required_vgprs)};
    const bool dead_for_every_owner =
        std::ranges::all_of(owner_liveness, [&](const LivenessAnalysis *liveness) {
          return !liveness->is_live_before(*inst, ref);
        });
    if (candidate_end <= REGISTER_SET_MAX_VGPRS && dead_for_every_owner &&
        !is_forbidden_flat_scratch_vgpr_run(site, *candidate, required_vgprs))
      return candidate;
    if (*candidate == UINT16_MAX)
      return std::nullopt;
    search_start = static_cast<uint16_t>(*candidate + 1u);
  }
  return std::nullopt;
}

void try_apply_flat_check_trap_patch(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                                     const ConSanOptions &options, ConSanResult &result,
                                     std::span<const ByteRange> initial_reserved_ranges = {}) {
  if (!instrumentation::is_admitted_arch(arch)) {
    result.warnings.emplace_back("ConSan flat check/trap proof has unsupported architecture");
    return;
  }

  struct CanonicalFlatSite {
    std::string_view container_name;
    uint64_t container_entry_text_offset = 0;
    ConSanFlatSite site;
  };
  size_t discovered_site_count = 0;
  for (const ConSanKernelInfo &kernel : result.kernels)
    discovered_site_count += kernel.flat_sites.size();
  for (const ConSanFunctionInfo &function : result.functions)
    discovered_site_count += function.flat_sites.size();
  auto canonicalizer = consan_detail::make_physical_site_alias_canonicalizer<CanonicalFlatSite>(
      result.errors, "ConSan FLAT physical site",
      [](const CanonicalFlatSite &candidate) { return candidate.site.file_offset; },
      [](const CanonicalFlatSite &candidate) { return candidate.container_name; },
      [&](const CanonicalFlatSite &lhs, const CanonicalFlatSite &rhs) {
        return lhs.container_entry_text_offset == rhs.container_entry_text_offset &&
               sc_flat_site_semantics_equal(lhs.site, rhs.site);
      },
      [](CanonicalFlatSite &retained, const CanonicalFlatSite &alias) {
        retained.site.owner_descriptor_file_offsets.insert(
            retained.site.owner_descriptor_file_offsets.end(),
            alias.site.owner_descriptor_file_offsets.begin(),
            alias.site.owner_descriptor_file_offsets.end());
      },
      discovered_site_count);
  for (const ConSanKernelInfo &kernel : result.kernels) {
    for (const ConSanFlatSite &site : kernel.flat_sites) {
      if (!sc_flat_site_is_applicable(site, options.flat_provenance_mode))
        continue;
      CanonicalFlatSite candidate{
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
    for (const ConSanFlatSite &site : function.flat_sites) {
      if (!sc_flat_site_is_applicable(site, options.flat_provenance_mode))
        continue;
      if (!canonicalizer.insert({
              .container_name = function.name,
              .container_entry_text_offset = function.entry_text_offset,
              .site = site,
          }))
        return;
    }
  }
  std::optional<std::vector<CanonicalFlatSite>> canonical_sites_result =
      std::move(canonicalizer).take();
  if (!canonical_sites_result)
    return;
  std::vector<CanonicalFlatSite> canonical_sites = std::move(*canonical_sites_result);
  for (CanonicalFlatSite &canonical : canonical_sites) {
    auto &owners = canonical.site.owner_descriptor_file_offsets;
    std::ranges::sort(owners);
    owners.erase(std::ranges::unique(owners).begin(), owners.end());
  }

  if (std::ranges::none_of(canonical_sites, [&](const CanonicalFlatSite &canonical) {
        return is_supported_flat_check_trap_site(canonical.site, options.flat_provenance_mode);
      })) {
    publish_sc_access_coverage_support(result, ConSanScAccessCoverageKind::FlatGroup,
                                       [](uint64_t) { return false; });
    result.warnings.emplace_back(
        "ConSan flat check/trap proof found no supported likely group-FLAT load/store site");
    return;
  }

  const auto delay_words =
      delay_instruction_word_count(options, result.errors, "ConSan flat check/trap proof");
  if (!delay_words)
    return;

  const auto original_bytes = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(code_object.image_data()), code_object.image_size());
  uint64_t appended_cave_offset = 0;
  if (code_object.text_sections().size() == 1)
    appended_cave_offset = code_object.text_sections().front()->size();
  std::vector<LocalNopCave> local_nop_caves = find_uncovered_nop_caves(code_object, result, arch);
  std::erase_if(local_nop_caves, [&](const LocalNopCave &cave) {
    return overlaps_preapplied_reserved_range(
        options, {cave.text_offset,
                  cave.text_offset + static_cast<uint64_t>(cave.word_count) * sizeof(uint32_t)});
  });
  uint32_t max_uncovered_nop_cave_words = 0;
  for (const LocalNopCave &cave : local_nop_caves)
    max_uncovered_nop_cave_words = std::max(max_uncovered_nop_cave_words, cave.word_count);
  size_t branch_relay_word_count = 0;
  for (const LocalNopCave &cave : local_nop_caves)
    branch_relay_word_count += cave.word_count;
  BranchOnlyRelayRouter branch_router;
  for (const LocalNopCave &cave : local_nop_caves) {
    for (uint64_t word = 0; word < cave.word_count; ++word) {
      (void)branch_router.offer(cave.text_offset + word * sizeof(uint32_t),
                                BranchOnlyRelayProvenance::PristineNop);
    }
  }

  struct FlatOwnerPlanningContext {
    uint64_t descriptor_file_offset = 0;
    KernelCfgScope scope;
    std::unique_ptr<LivenessAnalysis> liveness;
  };
  std::vector<std::unique_ptr<BasicBlock>> blocks;
  std::vector<BasicBlock *> block_ptrs;
  BlockOffsetIndex block_index;
  std::vector<uint64_t> kernel_entries;
  std::span<const uint8_t> pristine_text;
  std::unordered_map<uint64_t, FlatOwnerPlanningContext> owner_contexts;
  std::unordered_set<uint64_t> failed_owner_contexts;
  std::unique_ptr<Decoder> decoder = Decoder::create(arch);
  if (decoder) {
    std::vector<uint64_t> extra_leaders = kernel_entry_offsets(result);
    extra_leaders.reserve(extra_leaders.size() + result.functions.size());
    for (const ConSanFunctionInfo &function : result.functions)
      extra_leaders.push_back(function.entry_text_offset);
    std::ranges::sort(extra_leaders);
    extra_leaders.erase(std::ranges::unique(extra_leaders).begin(), extra_leaders.end());

    std::vector<BasicBlock::CodeRange> code_ranges;
    code_ranges.reserve(code_object.functions().size());
    for (const AmdGpuFunctionInfo &function : code_object.functions()) {
      if (function.code_size != 0)
        code_ranges.push_back(
            {.start_offset = function.entry_text_offset, .size = function.code_size});
    }
    blocks = BasicBlock::build(code_object, *decoder, arch, extra_leaders, code_ranges);
    block_ptrs = block_ptrs_for(blocks);

    block_index = build_block_offset_index(blocks);
    for (const ConSanKernelInfo &kernel : result.kernels) {
      if (kernel.has_text_range)
        kernel_entries.push_back(kernel.entry_text_offset);
    }
    std::ranges::sort(kernel_entries);
    kernel_entries.erase(std::ranges::unique(kernel_entries).begin(), kernel_entries.end());
    if (code_object.text_sections().size() == 1) {
      const Section *text = code_object.text_sections().front();
      pristine_text =
          std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(text->data()), text->size());
    }
  }
  owner_contexts.reserve(result.kernels.size());
  failed_owner_contexts.reserve(result.kernels.size());

  struct FlatCheckTrapCandidate {
    const ConSanFlatSite *site = nullptr;
    const LocalNopCave *local_cave = nullptr;
    const LocalNopCave *entry_island = nullptr;
    uint16_t scratch_vgpr = 0;
    uint64_t requested_words = 0;
    bool use_local_cave = false;
    bool use_appended_cave = false;
    bool branch_only_continuation = false;
    std::optional<BranchOnlyRelayRoute> branch_only_route;
    std::optional<LdsCheckTrapIndirectScalarScratch> indirect_scratch;
    std::vector<uint64_t> owner_descriptor_file_offsets;
    std::optional<DbiPatchPlacement> placement;
    bool entry_island_at_anchor = false;
    uint32_t displaced_original_size = 0;
    std::vector<uint32_t> displaced_tail_words;
    uint16_t vcc_save_sgpr = 0;
    std::optional<uint16_t> scalar_vcc_spill_vgpr = std::nullopt;
    uint16_t scalar_vcc_spill_vgpr_count = 0;
    std::optional<VgprSpillSequence> vgpr_spill;
    std::optional<DynamicStackBorrowedSgprSpillSequence> dynamic_stack_spill;
  };

  enum class FlatOwnerStackKind : uint8_t {
    Fixed,
    Dynamic,
    Mixed,
  };

  const auto make_candidate = [](const ConSanFlatSite *site, const LocalNopCave *local_cave,
                                 const LocalNopCave *entry_island, uint16_t scratch_vgpr,
                                 uint64_t requested_words, bool use_local_cave,
                                 bool use_appended_cave,
                                 std::optional<LdsCheckTrapIndirectScalarScratch> indirect_scratch,
                                 const std::vector<uint64_t> &owner_descriptor_file_offsets) {
    FlatCheckTrapCandidate candidate;
    candidate.site = site;
    candidate.local_cave = local_cave;
    candidate.entry_island = entry_island;
    candidate.scratch_vgpr = scratch_vgpr;
    candidate.requested_words = requested_words;
    candidate.use_local_cave = use_local_cave;
    candidate.use_appended_cave = use_appended_cave;
    candidate.indirect_scratch = indirect_scratch;
    candidate.owner_descriptor_file_offsets = owner_descriptor_file_offsets;
    return candidate;
  };

  std::vector<FlatCheckTrapCandidate> inline_candidates;
  std::vector<FlatCheckTrapCandidate> local_cave_candidates;
  std::vector<FlatCheckTrapCandidate> appended_cave_candidates;
  std::vector<FlatCheckTrapCandidate> branch_only_appended_candidates;
  std::vector<FlatCheckTrapCandidate> indirect_appended_candidates;
  std::vector<LocalNopCave> flat_entry_islands;
  size_t flat_entry_island_count = 0;
  for (const LocalNopCave &cave : local_nop_caves)
    flat_entry_island_count += cave.word_count / kLdsCheckTrapIndirectJumpWords;
  flat_entry_islands.reserve(flat_entry_island_count);
  for (const LocalNopCave &cave : local_nop_caves) {
    for (uint64_t word = 0; word + kLdsCheckTrapIndirectJumpWords <= cave.word_count;
         word += kLdsCheckTrapIndirectJumpWords) {
      flat_entry_islands.push_back(
          {.text_offset = cave.text_offset + word * sizeof(uint32_t),
           .file_offset = cave.file_offset + word * sizeof(uint32_t),
           .word_count = static_cast<uint32_t>(kLdsCheckTrapIndirectJumpWords)});
    }
  }
  bool skipped_supported_site_for_excessive_delay = false;
  size_t supported_candidate_count = 0;
  size_t scratchable_candidate_count = 0;
  size_t spill_backed_candidate_count = 0;
  size_t fixed_stack_candidate_count = 0;
  size_t dynamic_stack_candidate_count = 0;
  size_t mixed_stack_candidate_count = 0;
  size_t missing_vcc_save_candidate_count = 0;
  size_t missing_scratch_candidate_count = 0;
  size_t mixed_stack_owner_spill_rejection_count = 0;
  size_t dynamic_stack_spill_encoding_failure_count = 0;
  size_t private_spill_encoding_failure_count = 0;
  size_t append_cave_reachable_candidate_count = 0;
  size_t local_cave_reachable_candidate_count = 0;
  uint32_t max_observed_padding_words = 0;
  auto reachable_local_caves = [&](const ConSanFlatSite &site, uint64_t requested_words) {
    std::vector<const LocalNopCave *> caves;
    const uint64_t cave_words = requested_words + 1u;
    for (const LocalNopCave &cave : local_nop_caves) {
      // find_uncovered_nop_caves already proves that the complete run lies
      // outside every decoded kernel/function range. Ownership describes the
      // adjacent symbol, not an execution constraint: a direct branch may use
      // any such nonoverlapping run when both outbound and return edges reach.
      if (cave.word_count < cave_words)
        continue;
      const uint64_t return_branch_pc = cave.text_offset + requested_words * sizeof(uint32_t);
      if (compute_sopp_branch_simm16(site.text_offset, cave.text_offset) &&
          compute_sopp_branch_simm16(return_branch_pc, site.text_offset + site.size))
        caves.push_back(&cave);
    }
    return caves;
  };
  auto visit_site = [&](const ConSanFlatSite &site, std::optional<uint16_t> min_auto_scratch_vgpr,
                        std::optional<uint16_t> max_auto_scratch_vgpr,
                        std::optional<uint16_t> min_fresh_sgpr,
                        std::optional<uint16_t> allocated_sgpr_count,
                        uint32_t original_private_segment_size, FlatOwnerStackKind owner_stack_kind,
                        std::span<const LivenessAnalysis *const> owner_liveness,
                        std::vector<uint64_t> owner_descriptor_file_offsets) {
    if (!is_supported_flat_check_trap_site(site, options.flat_provenance_mode))
      return;
    ++supported_candidate_count;
    switch (owner_stack_kind) {
    case FlatOwnerStackKind::Fixed:
      ++fixed_stack_candidate_count;
      break;
    case FlatOwnerStackKind::Dynamic:
      ++dynamic_stack_candidate_count;
      break;
    case FlatOwnerStackKind::Mixed:
      ++mixed_stack_candidate_count;
      break;
    }
    auto required_vgprs = flat_dword_count(site);
    if (!required_vgprs)
      return;
    auto action_words =
        mismatch_action_word_count(options, arch, result.errors, "ConSan flat check/trap proof");
    if (!action_words)
      return;
    const uint32_t comparison_scratch_vgprs = flat_check_trap_comparison_scratch_vgprs(site);
    const uint32_t probe_scratch_vgprs = *required_vgprs + comparison_scratch_vgprs;
    const uint32_t report_alignment_padding =
        options.report_buffer_address &&
                (arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4)
            ? 1u
            : 0u;
    const uint32_t instrumentation_scratch_vgprs =
        probe_scratch_vgprs + report_alignment_padding + report_action_scratch_vgprs(options);
    const uint64_t comparison_setup_words = flat_check_trap_comparison_setup_words(site);
    const uint64_t mismatch_words =
        *required_vgprs > 1u ? 2u * *required_vgprs + 1u + *action_words
                             : comparison_setup_words + (2u + *action_words) * *required_vgprs;
    const uint64_t access_words = site.size / sizeof(uint32_t);
    const bool needs_runtime_group_gate =
        arch == ROCJITSU_CODE_ARCH_GFX1250 &&
        site.address_space_hint == ConSanFlatAddressSpaceHint::MaybeGroup;
    uint64_t requested_words = access_words + static_cast<uint64_t>(*delay_words) +
                               (needs_runtime_group_gate ? 2u : 0u) + access_words + 1u +
                               mismatch_words + 2u;
    const Instruction *site_inst =
        find_instruction_at_text_offset(KernelBlockScope(block_ptrs), site.text_offset);
    const bool uses_dynamic_stack = owner_stack_kind == FlatOwnerStackKind::Dynamic;
    const bool preserve_dynamic_stack_abi = owner_stack_kind != FlatOwnerStackKind::Fixed;
    auto vcc_save = choose_flat_check_trap_vcc_save_sgpr(site_inst, owner_liveness, min_fresh_sgpr,
                                                         options, preserve_dynamic_stack_abi);
    bool spill_vcc_save_pair_to_vgprs = false;
    if (!vcc_save && is_rdna4_family_arch(arch) && allocated_sgpr_count) {
      vcc_save = choose_flat_check_trap_scalar_spill_pair(*allocated_sgpr_count, options,
                                                          preserve_dynamic_stack_abi);
      spill_vcc_save_pair_to_vgprs = vcc_save.has_value();
    }
    if (!vcc_save) {
      ++missing_vcc_save_candidate_count;
      return;
    }
    uint32_t scratch_vgprs =
        instrumentation_scratch_vgprs + (spill_vcc_save_pair_to_vgprs ? 1u : 0u);
    auto scratch = choose_flat_check_trap_scratch_vgpr(site, options, site_inst, owner_liveness,
                                                       min_auto_scratch_vgpr, max_auto_scratch_vgpr,
                                                       scratch_vgprs);
    std::optional<VgprSpillSequence> vgpr_spill;
    std::optional<DynamicStackBorrowedSgprSpillSequence> dynamic_stack_spill;
    uint16_t scalar_vcc_spill_vgpr_count = spill_vcc_save_pair_to_vgprs ? 1u : 0u;
    if (!scratch && !options.scratch_vgpr && max_auto_scratch_vgpr) {
      if (owner_stack_kind == FlatOwnerStackKind::Mixed) {
        ++mixed_stack_owner_spill_rejection_count;
      } else {
        // Once the VGPR window itself is live, preserve it before using it as
        // instrumentation scratch. A borrowed scalar pair additionally needs
        // one active-lane reservoir per scalar. Dynamic-stack bootstrapping
        // always preserves the selected pair because that pair temporarily
        // carries both frame state and VCC, even when it was dead at the site.
        // An EXECZ guard leaves empty waves untouched.
        const uint16_t scalar_reservoir_count =
            uses_dynamic_stack ? DynamicStackBorrowedSgprSpillSequence::kScalarReservoirCount
                               : static_cast<uint16_t>(spill_vcc_save_pair_to_vgprs ? 2u : 0u);
        scratch_vgprs = instrumentation_scratch_vgprs + scalar_reservoir_count;
        scratch = choose_flat_spill_scratch_vgpr(site, *max_auto_scratch_vgpr, scratch_vgprs);
        if (scratch && uses_dynamic_stack) {
          dynamic_stack_spill = build_dynamic_stack_borrowed_sgpr_spill_sequence(
              *scratch, scratch_vgprs, *vcc_save,
              static_cast<uint16_t>(*scratch + instrumentation_scratch_vgprs),
              original_private_segment_size, arch);
        } else if (scratch) {
          const auto private_limit = address_free_scratch_private_limit(arch);
          if (private_limit) {
            SpillManager manager(original_private_segment_size, *private_limit);
            vgpr_spill = build_vgpr_spill_sequence(manager, *scratch, scratch_vgprs, arch);
          }
        }
        if (vgpr_spill || dynamic_stack_spill) {
          if (dynamic_stack_spill)
            spill_vcc_save_pair_to_vgprs = true;
          scalar_vcc_spill_vgpr_count = scalar_reservoir_count;
          if (dynamic_stack_spill) {
            requested_words += 1u + dynamic_stack_spill->save_words.size() +
                               dynamic_stack_spill->restore_words.size();
          } else {
            requested_words += 1u + vgpr_spill->save_words.size() + 2u * scalar_reservoir_count +
                               vgpr_spill->restore_words.size();
          }
          ++spill_backed_candidate_count;
        } else {
          scratch.reset();
          if (uses_dynamic_stack)
            ++dynamic_stack_spill_encoding_failure_count;
          else
            ++private_spill_encoding_failure_count;
        }
      }
    } else if (spill_vcc_save_pair_to_vgprs) {
      requested_words += 8u;
    }
    if (!scratch) {
      ++missing_scratch_candidate_count;
      return;
    }
    if (requested_words > 256u) {
      skipped_supported_site_for_excessive_delay = true;
      return;
    }
    const uint32_t padding_words = static_cast<uint32_t>(requested_words - 3u);
    const std::optional<uint16_t> scalar_vcc_spill_vgpr =
        spill_vcc_save_pair_to_vgprs ? std::optional<uint16_t>(static_cast<uint16_t>(
                                           *scratch + instrumentation_scratch_vgprs))
                                     : std::nullopt;
    const auto finish_candidate = [&](FlatCheckTrapCandidate &candidate) {
      candidate.vcc_save_sgpr = *vcc_save;
      candidate.scalar_vcc_spill_vgpr = scalar_vcc_spill_vgpr;
      candidate.scalar_vcc_spill_vgpr_count = scalar_vcc_spill_vgpr_count;
      candidate.vgpr_spill = vgpr_spill;
      candidate.dynamic_stack_spill = dynamic_stack_spill;
    };
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
    // Text growth must start from the same image whose earlier patches we retain.
    // The combined native-LDS + flat path currently carries those earlier edits
    // in result.elf_bytes, so leave it to inline/local placement for now.
    if (appended_cave_offset != 0 && result.elf_bytes.empty()) {
      const uint64_t return_branch_pc = appended_cave_offset + requested_words * sizeof(uint32_t);
      if (compute_sopp_branch_simm16(site.text_offset, appended_cave_offset) &&
          compute_sopp_branch_simm16(return_branch_pc, site.text_offset + site.size)) {
        ++append_cave_reachable_candidate_count;
        appended_cave_candidates.push_back(
            make_candidate(&site, nullptr, nullptr, *scratch, requested_words,
                           /*use_local_cave=*/false, /*use_appended_cave=*/true, std::nullopt,
                           owner_descriptor_file_offsets));
        finish_candidate(appended_cave_candidates.back());
      }
      branch_only_appended_candidates.push_back(
          make_candidate(&site, nullptr, nullptr, *scratch, requested_words,
                         /*use_local_cave=*/false, /*use_appended_cave=*/true, std::nullopt,
                         owner_descriptor_file_offsets));
      branch_only_appended_candidates.back().branch_only_continuation = true;
      finish_candidate(branch_only_appended_candidates.back());
    }
    const std::vector<const LocalNopCave *> caves = reachable_local_caves(site, requested_words);
    if (!caves.empty())
      ++local_cave_reachable_candidate_count;
    for (const LocalNopCave *cave : caves) {
      local_cave_candidates.push_back(
          make_candidate(&site, cave, nullptr, *scratch, requested_words,
                         /*use_local_cave=*/true, /*use_appended_cave=*/false, std::nullopt,
                         owner_descriptor_file_offsets));
      finish_candidate(local_cave_candidates.back());
    }
    if (!scalar_vcc_spill_vgpr && appended_cave_offset != 0 && result.elf_bytes.empty() &&
        !owner_descriptor_file_offsets.empty()) {
      auto scalar_scratch = choose_flat_check_trap_indirect_scalar_scratch(
          site_inst, owner_liveness, min_fresh_sgpr, options);
      // Prefer fresh scalar words above every owner reference.  A maximally
      // allocated owner may leave too little room there for the PC pair plus
      // SCC/VCC saves even though liveness proves a disjoint dead window at
      // the shared function site.  Match the LDS far-trampoline policy and
      // use that bounded fallback before rejecting the supported access.
      if (!scalar_scratch && min_fresh_sgpr) {
        scalar_scratch = choose_flat_check_trap_indirect_scalar_scratch(site_inst, owner_liveness,
                                                                        std::nullopt, options);
      }
      if (scalar_scratch) {
        for (const LocalNopCave &island : flat_entry_islands) {
          if (!compute_sopp_branch_simm16(site.text_offset, island.text_offset))
            continue;
          indirect_appended_candidates.push_back(
              make_candidate(&site, nullptr, &island, *scratch, requested_words,
                             /*use_local_cave=*/false, /*use_appended_cave=*/true, scalar_scratch,
                             owner_descriptor_file_offsets));
          indirect_appended_candidates.back().vcc_save_sgpr = scalar_scratch->vcc_save_sgpr;
        }
        if (code_object.text_sections().size() == 1) {
          const Section *text = code_object.text_sections().front();
          const auto text_bytes = std::span<const uint8_t>(
              reinterpret_cast<const uint8_t *>(text->data()), text->size());
          std::vector<uint32_t> displaced_words;
          uint64_t cursor = site.text_offset;
          while (displaced_words.size() < kLdsCheckTrapIndirectJumpWords) {
            const Instruction *instruction =
                find_instruction_at_text_offset(KernelBlockScope(block_ptrs), cursor);
            if (instruction == nullptr || instruction->size() <= 0 ||
                instruction->size() % static_cast<int>(sizeof(uint32_t)) != 0 ||
                instruction->raw_encoding() == nullptr ||
                (cursor != site.text_offset &&
                 (!is_relocatable_anchor(*instruction, cursor, text_bytes, arch) ||
                  overlaps_preapplied_reserved_range(
                      options, {cursor, cursor + static_cast<uint64_t>(instruction->size())})))) {
              displaced_words.clear();
              break;
            }
            const size_t old_size = displaced_words.size();
            displaced_words.resize(old_size + instruction->size() / sizeof(uint32_t));
            std::memcpy(displaced_words.data() + old_size, text_bytes.data() + cursor,
                        instruction->size());
            cursor += static_cast<uint64_t>(instruction->size());
          }
          if (displaced_words.size() >= kLdsCheckTrapIndirectJumpWords &&
              displaced_words.size() * sizeof(uint32_t) <= std::numeric_limits<uint32_t>::max()) {
            FlatCheckTrapCandidate direct =
                make_candidate(&site, nullptr, nullptr, *scratch, requested_words,
                               /*use_local_cave=*/false, /*use_appended_cave=*/true, scalar_scratch,
                               owner_descriptor_file_offsets);
            direct.entry_island_at_anchor = true;
            direct.vcc_save_sgpr = scalar_scratch->vcc_save_sgpr;
            direct.displaced_original_size =
                static_cast<uint32_t>(displaced_words.size() * sizeof(uint32_t));
            direct.displaced_tail_words.assign(
                displaced_words.begin() + site.size / sizeof(uint32_t), displaced_words.end());
            indirect_appended_candidates.push_back(std::move(direct));
          }
        }
      }
    }
    if (observed_padding < padding_words)
      return;
    inline_candidates.push_back(make_candidate(&site, nullptr, nullptr, *scratch, requested_words,
                                               /*use_local_cave=*/false,
                                               /*use_appended_cave=*/false, std::nullopt,
                                               owner_descriptor_file_offsets));
    finish_candidate(inline_candidates.back());
  };

  // Shared functions can contribute hundreds of candidate sites to the same
  // owner.  Computing the owner's maximum register references for every site
  // turns otherwise modest multi-kernel objects into a quadratic scan.  The
  // answer depends only on the immutable decoded blocks and kernel range, so
  // compute it once per descriptor and reuse it for both kernel-local and
  // shared-function candidates.
  std::unordered_map<uint64_t, const ConSanKernelInfo *> kernels_by_descriptor;
  std::unordered_map<uint64_t, KernelMaxRegisterRefs> max_refs_by_descriptor;
  kernels_by_descriptor.reserve(result.kernels.size());
  max_refs_by_descriptor.reserve(result.kernels.size());
  for (const ConSanKernelInfo &kernel : result.kernels)
    kernels_by_descriptor.emplace(kernel.descriptor_file_offset, &kernel);
  const auto get_owner_context = [&](const ConSanKernelInfo &kernel) -> FlatOwnerPlanningContext * {
    const uint64_t descriptor_file_offset = kernel.descriptor_file_offset;
    if (const auto existing = owner_contexts.find(descriptor_file_offset);
        existing != owner_contexts.end())
      return &existing->second;
    if (decoder == nullptr || !kernel.has_text_range ||
        failed_owner_contexts.contains(descriptor_file_offset))
      return nullptr;
    auto scope = build_kernel_cfg_scope(blocks, block_index,
                                        KernelScopeRequest{.entry_offset = kernel.entry_text_offset,
                                                           .additional_entry_offsets = {}},
                                        kernel_entries, pristine_text);
    if (!scope) {
      failed_owner_contexts.insert(descriptor_file_offset);
      return nullptr;
    }
    FlatOwnerPlanningContext context{
        .descriptor_file_offset = descriptor_file_offset,
        .scope = std::move(*scope),
        .liveness = nullptr,
    };
    context.liveness =
        std::make_unique<LivenessAnalysis>(KernelBlockScope(context.scope.blocks),
                                           LivenessAnalysisOptions{}, context.scope.liveness_edges);
    return &owner_contexts.emplace(descriptor_file_offset, std::move(context)).first->second;
  };

  std::unordered_set<uint64_t> supported_physical_sites;
  supported_physical_sites.reserve(canonical_sites.size());
  for (CanonicalFlatSite &canonical : canonical_sites) {
    const ConSanFlatSite &site = canonical.site;
    const Instruction *site_inst =
        find_instruction_at_text_offset(KernelBlockScope(block_ptrs), site.text_offset);
    std::vector<uint64_t> owners = site.owner_descriptor_file_offsets;
    for (const ConSanCandidateResourcePlan &plan : result.resource_plans) {
      if (plan.site_kind == ConSanResourceSiteKind::Access &&
          plan.text_offset == site.text_offset) {
        owners.insert(owners.end(), plan.owner_descriptor_file_offsets.begin(),
                      plan.owner_descriptor_file_offsets.end());
      }
    }
    for (const ConSanFaultSite &fault : result.fault_sites) {
      if (fault.in_kernel || fault.container_name != canonical.container_name)
        continue;
      for (const ConSanExecutionOwner &owner : fault.execution_owners)
        owners.push_back(owner.descriptor_file_offset);
    }
    std::ranges::sort(owners);
    owners.erase(std::ranges::unique(owners).begin(), owners.end());

    bool complete_context = decoder != nullptr && site_inst != nullptr && !owners.empty();
    std::vector<const LivenessAnalysis *> owner_liveness;
    owner_liveness.reserve(owners.size());
    std::optional<uint16_t> common_allocated_vgpr_count;
    std::optional<uint16_t> common_allocated_sgpr_count;
    std::optional<uint16_t> max_owner_vgpr_ref;
    std::optional<uint16_t> max_owner_sgpr_ref;
    uint32_t max_owner_private_segment_size = 0;
    bool any_owner_uses_dynamic_stack = false;
    bool any_owner_uses_fixed_stack = false;
    for (uint64_t owner : owners) {
      const auto owner_kernel = kernels_by_descriptor.find(owner);
      FlatOwnerPlanningContext *owner_context = owner_kernel == kernels_by_descriptor.end()
                                                    ? nullptr
                                                    : get_owner_context(*owner_kernel->second);
      auto refs = max_refs_by_descriptor.find(owner);
      if (refs == max_refs_by_descriptor.end() && owner_context != nullptr) {
        refs =
            max_refs_by_descriptor
                .emplace(owner,
                         max_register_refs_in_kernel(*owner_kernel->second,
                                                     KernelBlockScope(owner_context->scope.blocks)))
                .first;
      }
      const auto allocated_vgprs = descriptor_vgpr_allocation_count(original_bytes, owner, arch);
      const Instruction *owner_site_inst =
          owner_context == nullptr
              ? nullptr
              : find_instruction_at_text_offset(KernelBlockScope(owner_context->scope.blocks),
                                                site.text_offset);
      if (owner_kernel == kernels_by_descriptor.end() ||
          owner_kernel->second->preflight_action == ConSanPreflightAction::Reject ||
          owner_context == nullptr || owner_context->liveness == nullptr ||
          owner_site_inst != site_inst || refs == max_refs_by_descriptor.end() ||
          !allocated_vgprs || owner > original_bytes.size() ||
          sizeof(KD) > original_bytes.size() - owner) {
        complete_context = false;
        break;
      }
      owner_liveness.push_back(owner_context->liveness.get());

      KD descriptor{};
      std::memcpy(&descriptor, original_bytes.data() + owner, sizeof(descriptor));
      const uint16_t allocated_sgprs = descriptor_sgpr_allocation_count(descriptor, arch);
      common_allocated_vgpr_count = common_allocated_vgpr_count
                                        ? std::min(*common_allocated_vgpr_count, *allocated_vgprs)
                                        : *allocated_vgprs;
      common_allocated_sgpr_count = common_allocated_sgpr_count
                                        ? std::min(*common_allocated_sgpr_count, allocated_sgprs)
                                        : allocated_sgprs;
      max_owner_private_segment_size =
          std::max(max_owner_private_segment_size, descriptor.private_segment_fixed_size);
      if (refs->second.vgpr)
        max_owner_vgpr_ref = max_owner_vgpr_ref ? std::max(*max_owner_vgpr_ref, *refs->second.vgpr)
                                                : *refs->second.vgpr;
      if (refs->second.sgpr)
        max_owner_sgpr_ref = max_owner_sgpr_ref ? std::max(*max_owner_sgpr_ref, *refs->second.sgpr)
                                                : *refs->second.sgpr;
      if (owner_kernel->second->uses_dynamic_stack.value_or(false))
        any_owner_uses_dynamic_stack = true;
      else
        any_owner_uses_fixed_stack = true;
    }
    if (!complete_context || !common_allocated_vgpr_count || !common_allocated_sgpr_count)
      continue;

    if (is_supported_flat_check_trap_site(site, options.flat_provenance_mode))
      supported_physical_sites.insert(site.file_offset);

    const bool selected_by_filter =
        options.test_kernel_name_filter.empty() ||
        canonical.container_name.find(options.test_kernel_name_filter) != std::string::npos ||
        std::ranges::any_of(owners, [&](uint64_t owner) {
          const auto kernel = kernels_by_descriptor.find(owner);
          return kernel != kernels_by_descriptor.end() &&
                 kernel->second->name.find(options.test_kernel_name_filter) != std::string::npos;
        });
    if (!selected_by_filter)
      continue;

    std::optional<uint16_t> min_auto_scratch_vgpr;
    if (max_owner_vgpr_ref && *max_owner_vgpr_ref < 255u)
      min_auto_scratch_vgpr = static_cast<uint16_t>(*max_owner_vgpr_ref + 1u);
    std::optional<uint16_t> min_fresh_sgpr;
    if (max_owner_sgpr_ref && *max_owner_sgpr_ref + 1u < REGISTER_SET_ALLOCATABLE_SGPRS)
      min_fresh_sgpr = static_cast<uint16_t>(*max_owner_sgpr_ref + 1u);
    const FlatOwnerStackKind owner_stack_kind =
        any_owner_uses_dynamic_stack && any_owner_uses_fixed_stack ? FlatOwnerStackKind::Mixed
        : any_owner_uses_dynamic_stack                             ? FlatOwnerStackKind::Dynamic
                                                                   : FlatOwnerStackKind::Fixed;
    visit_site(site, min_auto_scratch_vgpr, common_allocated_vgpr_count, min_fresh_sgpr,
               common_allocated_sgpr_count, max_owner_private_segment_size, owner_stack_kind,
               owner_liveness, std::move(owners));
  }
  publish_sc_access_coverage_support(
      result, ConSanScAccessCoverageKind::FlatGroup,
      [&](uint64_t file_offset) { return supported_physical_sites.contains(file_offset); });
  if (!result.errors.empty())
    return;
  if (options.max_patches_is_expert_limit && options.max_patches == 0u) {
    result.warnings.emplace_back(
        "ConSan flat check/trap proof skipped: shared patch budget consumed");
    return;
  }

  auto candidate_order = [](const FlatCheckTrapCandidate &lhs, const FlatCheckTrapCandidate &rhs) {
    if (lhs.site->file_offset != rhs.site->file_offset)
      return lhs.site->file_offset < rhs.site->file_offset;
    if (lhs.use_local_cave != rhs.use_local_cave)
      return !lhs.use_local_cave;
    if (lhs.use_appended_cave != rhs.use_appended_cave)
      return !lhs.use_appended_cave;
    if (lhs.entry_island_at_anchor != rhs.entry_island_at_anchor)
      return lhs.entry_island_at_anchor;
    const uint64_t lhs_cave = lhs.local_cave ? lhs.local_cave->file_offset : 0;
    const uint64_t rhs_cave = rhs.local_cave ? rhs.local_cave->file_offset : 0;
    return lhs_cave < rhs_cave;
  };
  std::ranges::sort(inline_candidates, candidate_order);
  std::ranges::sort(local_cave_candidates, candidate_order);
  std::ranges::sort(appended_cave_candidates, candidate_order);
  std::ranges::sort(branch_only_appended_candidates, candidate_order);
  std::ranges::sort(indirect_appended_candidates, candidate_order);

  const size_t max_patches =
      options.max_patches_is_expert_limit ? options.max_patches : supported_candidate_count;
  std::vector<FlatCheckTrapCandidate> selected_candidates;
  selected_candidates.reserve(std::min(max_patches, supported_candidate_count));
  DbiPatchPlacementPlanner placement_planner(arch, appended_cave_offset);
  BranchOnlyDirectRelayReservoirSet direct_branch_reservoirs;
  const auto prepare_direct_branch_reservoirs = [&]() {
    if (selected_candidates.size() >= max_patches || branch_only_appended_candidates.empty() ||
        pristine_text.empty()) {
      return true;
    }
    std::vector<std::pair<uint64_t, uint64_t>> protected_ranges;
    protected_ranges.reserve(canonical_sites.size() + local_nop_caves.size() +
                             result.kernels.size() + result.functions.size() +
                             options.preapplied_reserved_ranges.size() + result.patches.size());
    for (const CanonicalFlatSite &canonical : canonical_sites) {
      protected_ranges.emplace_back(canonical.site.text_offset,
                                    canonical.site.text_offset + canonical.site.size);
    }
    for (const LocalNopCave &cave : local_nop_caves) {
      protected_ranges.emplace_back(cave.text_offset,
                                    cave.text_offset +
                                        static_cast<uint64_t>(cave.word_count) * sizeof(uint32_t));
    }
    const auto protect_sync_sites = [&](const auto &container) {
      const auto append = [&](const auto &sites) {
        for (const auto &site : sites)
          protected_ranges.emplace_back(site.text_offset, site.text_offset + site.size);
      };
      append(container.barrier_sites);
      append(container.atomic_sites);
      append(container.fence_sites);
    };
    for (const ConSanKernelInfo &kernel : result.kernels) {
      protect_sync_sites(kernel);
      protected_ranges.emplace_back(
          kernel.entry_text_offset,
          std::min<uint64_t>(pristine_text.size(), kernel.entry_text_offset + 256u));
    }
    for (const ConSanFunctionInfo &function : result.functions)
      protect_sync_sites(function);
    for (const ConSanPreappliedReservedRange &range : options.preapplied_reserved_ranges)
      protected_ranges.emplace_back(range.text_offset, range.text_offset + range.size);
    for (const ConSanPatchInfo &patch : result.patches) {
      protected_ranges.emplace_back(patch.anchor_offset,
                                    patch.anchor_offset +
                                        std::max<uint32_t>(patch.original_size, sizeof(uint32_t)));
      if (patch.trampoline_size != 0u) {
        protected_ranges.emplace_back(patch.trampoline_offset,
                                      patch.trampoline_offset + patch.trampoline_size);
      }
    }
    const auto latest_candidate =
        std::ranges::max_element(branch_only_appended_candidates, {},
                                 [](const auto &candidate) { return candidate.site->text_offset; });
    const uint64_t latest_source = latest_candidate->site->text_offset;
    const uint64_t route_midpoint =
        latest_source + (placement_planner.appended_end() - latest_source) / 2u;
    std::string relay_error;
    if (!branch_router.plan_direct_reservoirs(
            block_ptrs, pristine_text, protected_ranges, arch, route_midpoint,
            2u * (max_patches - selected_candidates.size()), placement_planner,
            direct_branch_reservoirs, &relay_error)) {
      result.errors.emplace_back("ConSan flat check/trap proof " + relay_error);
      return false;
    }
    return true;
  };
  std::unordered_set<uint64_t> selected_flat_anchor_relay_offsets;
  size_t branch_only_placement_failures = 0;
  size_t branch_only_entry_route_failures = 0;
  size_t branch_only_return_route_failures = 0;
  size_t branch_only_search_budget_failures = 0;
  size_t branch_only_reservation_failures = 0;
  size_t branch_only_selected = 0;
  const auto publish_flat_anchor_relays = [&](const FlatCheckTrapCandidate &candidate,
                                              uint32_t original_size) {
    branch_router.retire_range(candidate.site->text_offset, original_size);
    if (candidate.entry_island_at_anchor || original_size <= sizeof(uint32_t))
      return;
    for (uint64_t offset = candidate.site->text_offset + sizeof(uint32_t);
         offset < candidate.site->text_offset + original_size; offset += sizeof(uint32_t)) {
      if (selected_flat_anchor_relay_offsets.insert(offset).second) {
        (void)branch_router.offer(offset, BranchOnlyRelayProvenance::OwnedAnchor);
      }
    }
  };
  std::unordered_set<uint64_t> selected_flat_site_offsets;
  auto try_select_candidate = [&](FlatCheckTrapCandidate candidate) {
    if (selected_candidates.size() >= max_patches)
      return;
    const ConSanFlatSite &site = *candidate.site;
    if (selected_flat_site_offsets.contains(site.text_offset))
      return;
    const uint64_t body_bytes =
        (candidate.requested_words +
         (candidate.entry_island_at_anchor
              ? 1u + static_cast<uint64_t>(candidate.displaced_tail_words.size())
              : 0u)) *
        sizeof(uint32_t);
    const uint32_t original_size =
        candidate.entry_island_at_anchor ? candidate.displaced_original_size : site.size;
    const ByteRange anchor_file_range{site.file_offset, site.file_offset + original_size};
    if (overlaps_reserved_range(initial_reserved_ranges, anchor_file_range))
      return;
    if (candidate.use_local_cave) {
      if (candidate.local_cave == nullptr)
        return;
      const ByteRange cave_file_range{candidate.local_cave->file_offset,
                                      candidate.local_cave->file_offset + body_bytes +
                                          sizeof(uint32_t)};
      if (overlaps_reserved_range(initial_reserved_ranges, cave_file_range))
        return;
    }
    if (candidate.entry_island != nullptr || candidate.entry_island_at_anchor) {
      if (candidate.entry_island != nullptr) {
        const ByteRange island_file_range{candidate.entry_island->file_offset,
                                          candidate.entry_island->file_offset +
                                              kLdsCheckTrapIndirectJumpWords * sizeof(uint32_t)};
        if (overlaps_reserved_range(initial_reserved_ranges, island_file_range))
          return;
        const ByteRange island_text_range{candidate.entry_island->text_offset,
                                          candidate.entry_island->text_offset +
                                              kLdsCheckTrapIndirectJumpWords * sizeof(uint32_t)};
        if (std::ranges::any_of(placement_planner.occupied_ranges(), [&](const auto &range) {
              return island_text_range.begin < range.second && range.first < island_text_range.end;
            }))
          return;
      }
    }
    if (!candidate.use_local_cave && !candidate.use_appended_cave &&
        overlaps_reserved_range(initial_reserved_ranges,
                                {site.file_offset, site.file_offset + body_bytes}))
      return;

    if (candidate.branch_only_continuation) {
      DbiPatchPlacementPlanner tentative_planner = placement_planner;
      const auto placement = tentative_planner.plan_indirect_appended(
          site.text_offset, original_size, body_bytes + sizeof(uint32_t));
      if (!placement) {
        ++branch_only_placement_failures;
        return;
      }

      std::string relay_error;
      BranchOnlyRelayPlanFailure relay_failure = BranchOnlyRelayPlanFailure::None;
      auto route =
          branch_router.plan_pair(tentative_planner, site.text_offset, placement->body_offset,
                                  placement->body_offset + body_bytes,
                                  site.text_offset + original_size, &relay_error, &relay_failure);
      if (!route) {
        if (relay_failure == BranchOnlyRelayPlanFailure::ReturnRoute)
          ++branch_only_return_route_failures;
        else if (relay_failure == BranchOnlyRelayPlanFailure::SearchBudget)
          ++branch_only_search_budget_failures;
        else if (relay_failure == BranchOnlyRelayPlanFailure::Reservation)
          ++branch_only_reservation_failures;
        else
          ++branch_only_entry_route_failures;
        return;
      }
      if (!branch_router.commit(*route, &relay_error)) {
        ++branch_only_reservation_failures;
        return;
      }
      if (!direct_branch_reservoirs.mark_claims_used(route->claims, &relay_error)) {
        result.errors.emplace_back("ConSan flat check/trap proof " + relay_error);
        return;
      }
      candidate.branch_only_route = std::move(*route);
      placement_planner = std::move(tentative_planner);
      candidate.placement = placement;
      selected_candidates.push_back(std::move(candidate));
      selected_flat_site_offsets.insert(site.text_offset);
      publish_flat_anchor_relays(selected_candidates.back(), original_size);
      ++branch_only_selected;
      return;
    }

    if (candidate.entry_island != nullptr || candidate.entry_island_at_anchor) {
      DbiPatchPlacementPlanner tentative_planner = placement_planner;
      const auto placement = tentative_planner.plan_indirect_appended(
          site.text_offset, original_size,
          body_bytes + kLdsCheckTrapIndirectJumpWords * sizeof(uint32_t));
      if (!placement || (candidate.entry_island != nullptr &&
                         !tentative_planner.reserve_existing_range(
                             candidate.entry_island->text_offset,
                             kLdsCheckTrapIndirectJumpWords * sizeof(uint32_t)))) {
        return;
      }
      placement_planner = std::move(tentative_planner);
      candidate.placement = placement;
      selected_candidates.push_back(std::move(candidate));
      selected_flat_site_offsets.insert(site.text_offset);
      publish_flat_anchor_relays(selected_candidates.back(), original_size);
      if (selected_candidates.back().entry_island != nullptr) {
        branch_router.retire_range(selected_candidates.back().entry_island->text_offset,
                                   kLdsCheckTrapIndirectJumpWords * sizeof(uint32_t));
      }
      return;
    }

    DbiPatchPlacementRequest request;
    request.anchor_offset = site.text_offset;
    request.original_size = site.size;
    request.body_size = body_bytes;
    request.inline_capacity =
        (candidate.use_local_cave || candidate.use_appended_cave) ? 0u : body_bytes;
    request.allow_appended_cave = candidate.use_appended_cave;
    if (candidate.use_local_cave) {
      request.local_cave = DbiPatchLocalCave{
          candidate.local_cave->text_offset,
          static_cast<uint64_t>(candidate.local_cave->word_count) * sizeof(uint32_t)};
    }
    DbiPatchPlacementPlanner tentative_planner = placement_planner;
    const auto placement = tentative_planner.plan(request);
    if (!placement)
      return;
    if ((candidate.use_local_cave && placement->kind != DbiPatchPlacementKind::LocalCave) ||
        (candidate.use_appended_cave && placement->kind != DbiPatchPlacementKind::AppendedCave) ||
        (!candidate.use_local_cave && !candidate.use_appended_cave &&
         placement->kind != DbiPatchPlacementKind::Inline)) {
      return;
    }
    placement_planner = std::move(tentative_planner);
    candidate.placement = placement;
    selected_candidates.push_back(std::move(candidate));
    selected_flat_site_offsets.insert(site.text_offset);
    if (placement->kind == DbiPatchPlacementKind::Inline) {
      branch_router.retire_range(placement->body_offset, placement->body_size);
    } else {
      publish_flat_anchor_relays(selected_candidates.back(), original_size);
      if (placement->kind == DbiPatchPlacementKind::LocalCave) {
        branch_router.retire_range(placement->body_offset, placement->return_branch_offset +
                                                               sizeof(uint32_t) -
                                                               placement->body_offset);
      }
    }
  };

  for (const FlatCheckTrapCandidate &candidate : inline_candidates)
    try_select_candidate(candidate);
  for (const FlatCheckTrapCandidate &candidate : appended_cave_candidates)
    try_select_candidate(candidate);
  for (const FlatCheckTrapCandidate &candidate : local_cave_candidates)
    try_select_candidate(candidate);

  const auto branch_failure_baseline =
      std::tuple{branch_only_placement_failures, branch_only_entry_route_failures,
                 branch_only_return_route_failures, branch_only_search_budget_failures,
                 branch_only_reservation_failures};
  for (const FlatCheckTrapCandidate &candidate : branch_only_appended_candidates)
    try_select_candidate(candidate);
  if (selected_candidates.size() < max_patches && !branch_only_appended_candidates.empty()) {
    BranchOnlyRelayRouter router_without_direct_reservoirs = branch_router;
    DbiPatchPlacementPlanner planner_without_direct_reservoirs = placement_planner;
    const size_t selected_before_direct_reservoirs = branch_only_selected;
    if (!prepare_direct_branch_reservoirs())
      return;

    std::tie(branch_only_placement_failures, branch_only_entry_route_failures,
             branch_only_return_route_failures, branch_only_search_budget_failures,
             branch_only_reservation_failures) = branch_failure_baseline;
    for (const FlatCheckTrapCandidate &candidate : branch_only_appended_candidates)
      try_select_candidate(candidate);

    if (branch_only_selected == selected_before_direct_reservoirs) {
      branch_router = std::move(router_without_direct_reservoirs);
      placement_planner = std::move(planner_without_direct_reservoirs);
      direct_branch_reservoirs = {};
    }
  }
  for (const FlatCheckTrapCandidate &candidate : indirect_appended_candidates)
    try_select_candidate(candidate);

  const size_t selection_target = std::min(max_patches, supported_candidate_count);
  result.flat_selection_telemetry = ConSanFlatSelectionTelemetry{
      .supported_candidate_count = supported_candidate_count,
      .selection_target = selection_target,
      .selected_candidate_count = selected_candidates.size(),
      .branch_only_candidate_count = branch_only_appended_candidates.size(),
      .fixed_stack_candidate_count = fixed_stack_candidate_count,
      .dynamic_stack_candidate_count = dynamic_stack_candidate_count,
      .mixed_stack_candidate_count = mixed_stack_candidate_count,
      .missing_vcc_save_candidate_count = missing_vcc_save_candidate_count,
      .missing_scratch_candidate_count = missing_scratch_candidate_count,
      .spill_backed_candidate_count = spill_backed_candidate_count,
      .mixed_stack_spill_rejection_count = mixed_stack_owner_spill_rejection_count,
      .dynamic_stack_spill_encoding_failure_count = dynamic_stack_spill_encoding_failure_count,
      .private_spill_encoding_failure_count = private_spill_encoding_failure_count,
      .branch_only_selected_count = branch_only_selected,
      .branch_only_placement_failure_count = branch_only_placement_failures,
      .branch_only_entry_route_failure_count = branch_only_entry_route_failures,
      .branch_only_return_route_failure_count = branch_only_return_route_failures,
      .branch_only_search_budget_failure_count = branch_only_search_budget_failures,
      .branch_only_reservation_failure_count = branch_only_reservation_failures,
      .original_nop_relay_slot_count = branch_relay_word_count,
      .selected_anchor_relay_slot_count = selected_flat_anchor_relay_offsets.size(),
  };
  if (selected_candidates.size() < selection_target) {
    result.warnings.emplace_back("ConSan flat check/trap proof selected only " +
                                 std::to_string(selected_candidates.size()) + "/" +
                                 std::to_string(selection_target) +
                                 " supported sites; structured selection telemetry is available");
  }

  if (selected_candidates.empty()) {
    if (skipped_supported_site_for_excessive_delay) {
      result.warnings.emplace_back(
          "ConSan flat check/trap proof skipped: requested delay needs too much "
          "padding for the supported flat site");
      return;
    }
    result.warnings.emplace_back(
        "ConSan flat check/trap proof found no selectable padded or "
        "local- or appended-cave-reachable likely group-FLAT load/store site; "
        "supported_candidates=" +
        std::to_string(supported_candidate_count) +
        " scratchable_candidates=" + std::to_string(scratchable_candidate_count) +
        " spill_backed_candidates=" + std::to_string(spill_backed_candidate_count) +
        " mixed_stack_owner_spill_rejections=" +
        std::to_string(mixed_stack_owner_spill_rejection_count) +
        " dynamic_stack_spill_encoding_failures=" +
        std::to_string(dynamic_stack_spill_encoding_failure_count) +
        " private_spill_encoding_failures=" + std::to_string(private_spill_encoding_failure_count) +
        " inline_candidates=" + std::to_string(inline_candidates.size()) +
        " local_cave_candidates=" + std::to_string(local_cave_candidates.size()) +
        " branch_only_appended_candidates=" +
        std::to_string(branch_only_appended_candidates.size()) + " max_observed_padding_words=" +
        std::to_string(max_observed_padding_words) + " append_cave_reachable_candidates=" +
        std::to_string(append_cave_reachable_candidate_count) +
        " uncovered_nop_caves=" + std::to_string(local_nop_caves.size()) +
        " max_uncovered_nop_cave_words=" + std::to_string(max_uncovered_nop_cave_words) +
        " local_cave_reachable_candidates=" + std::to_string(local_cave_reachable_candidate_count));
    return;
  }

  struct PlannedFlatCheckTrapPatch {
    const ConSanFlatSite *site = nullptr;
    const LocalNopCave *local_cave = nullptr;
    const LocalNopCave *entry_island = nullptr;
    std::vector<uint32_t> words;
    std::vector<uint32_t> cave_words;
    std::vector<uint32_t> entry_island_words;
    std::vector<uint32_t> anchor_words;
    ConSanPatchInfo info;
    bool use_local_cave = false;
    bool use_appended_cave = false;
    bool branch_only_continuation = false;
    std::optional<BranchOnlyRelayRoute> branch_only_route;
    std::optional<DbiPatchPlacement> placement;
    bool entry_island_at_anchor = false;
    std::vector<uint32_t> displaced_tail_words;
  };

  const auto wait_flat = arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4
                             ? instrumentation::build_s_wait_flat_load_lds0(arch)
                             : build_s_wait_dscnt_word(0, arch);
  if (!wait_flat) {
    result.errors.emplace_back("ConSan flat check/trap proof could not encode sequence");
    return;
  }

  std::vector<PlannedFlatCheckTrapPatch> planned_patches;
  planned_patches.reserve(selected_candidates.size());
  for (const FlatCheckTrapCandidate &candidate : selected_candidates) {
    if (candidate.site == nullptr) {
      result.errors.emplace_back("ConSan flat check/trap proof selected missing site");
      return;
    }
    const ConSanFlatSite &site = *candidate.site;
    const auto required_vgprs = flat_dword_count(site);
    const auto compare_vgpr = flat_check_trap_compare_vgpr(site);
    if (!required_vgprs || !compare_vgpr) {
      result.errors.emplace_back("ConSan flat check/trap proof selected unsupported site");
      return;
    }

    const size_t access_words = site.size / sizeof(uint32_t);
    if (access_words < 2u || access_words > 3u) {
      result.errors.emplace_back("ConSan flat check/trap proof selected invalid access size");
      return;
    }
    std::array<uint32_t, 3> original_access{};
    if (!read_words_at(original_bytes, site.file_offset,
                       std::span<uint32_t>(original_access).first(access_words))) {
      result.errors.emplace_back("ConSan flat check/trap proof could not read access");
      return;
    }

    std::optional<std::array<uint32_t, 3>> duplicate_load;
    const bool use_trampoline = candidate.use_local_cave || candidate.use_appended_cave;
    ConSanPatchKind patch_kind = use_trampoline ? ConSanPatchKind::LocalCaveFlatLoadCheckTrap
                                                : ConSanPatchKind::InlineFlatLoadCheckTrap;
    if (site.kind == ConSanLdsAccessKind::Read) {
      duplicate_load = retarget_flat_load_vdst(original_access, candidate.scratch_vgpr, arch);
    } else if (site.kind == ConSanLdsAccessKind::Write) {
      duplicate_load = build_flat_load_from_flat_store(original_access, site.width_bits,
                                                       candidate.scratch_vgpr, arch);
      patch_kind = use_trampoline ? ConSanPatchKind::LocalCaveFlatStoreCheckTrap
                                  : ConSanPatchKind::InlineFlatStoreCheckTrap;
    }
    if (!duplicate_load) {
      result.errors.emplace_back("ConSan flat check/trap proof could not encode readback");
      return;
    }

    PlannedFlatCheckTrapPatch planned;
    planned.site = &site;
    planned.local_cave = candidate.local_cave;
    planned.entry_island = candidate.entry_island;
    planned.use_local_cave = candidate.use_local_cave;
    planned.use_appended_cave = candidate.use_appended_cave;
    planned.branch_only_continuation = candidate.branch_only_continuation;
    planned.branch_only_route = candidate.branch_only_route;
    planned.placement = candidate.placement;
    planned.entry_island_at_anchor = candidate.entry_island_at_anchor;
    planned.displaced_tail_words = candidate.displaced_tail_words;
    planned.info.kind = patch_kind;
    planned.info.anchor_offset = site.text_offset;
    planned.info.scratch_vgpr = candidate.scratch_vgpr;
    planned.info.scalar_vcc_spill_sgpr = candidate.scalar_vcc_spill_vgpr
                                             ? std::optional<uint16_t>(candidate.vcc_save_sgpr)
                                             : std::nullopt;
    planned.info.scalar_vcc_spill_vgpr = candidate.scalar_vcc_spill_vgpr;
    planned.info.scalar_vcc_spill_vgpr_count = candidate.scalar_vcc_spill_vgpr_count;
    planned.info.owner_descriptor_file_offsets = candidate.owner_descriptor_file_offsets;
    planned.info.branch_only_continuation = candidate.branch_only_continuation;
    if (candidate.branch_only_route) {
      planned.info.branch_only_entry_relay_offsets =
          candidate.branch_only_route->entry_relay_offsets;
      planned.info.branch_only_return_relay_offsets =
          candidate.branch_only_route->return_relay_offsets;
    }
    planned.info.required_sgpr_count = static_cast<uint16_t>(candidate.vcc_save_sgpr + 2u);
    if (candidate.vgpr_spill) {
      planned.info.spilled_vgpr_count = candidate.vgpr_spill->vgpr_count;
      planned.info.required_private_segment_size = candidate.vgpr_spill->total_private_bytes;
    } else if (candidate.dynamic_stack_spill) {
      planned.info.spilled_vgpr_count = candidate.dynamic_stack_spill->vgpr_count;
      planned.info.required_private_segment_size =
          candidate.dynamic_stack_spill->total_private_bytes;
      planned.info.dynamic_private_segment_addend =
          candidate.dynamic_stack_spill->dynamic_frame_bytes;
    }
    planned.words.reserve(static_cast<size_t>(candidate.requested_words));
    const auto save_vcc =
        build_flat_check_trap_vcc_move(candidate.vcc_save_sgpr, kRdna4VccLo, arch);
    const auto restore_vcc =
        build_flat_check_trap_vcc_move(kRdna4VccLo, candidate.vcc_save_sgpr, arch);
    if (!save_vcc || !restore_vcc) {
      result.errors.emplace_back("ConSan flat check/trap proof could not preserve VCC");
      return;
    }
    planned.words.push_back(*save_vcc);
    planned.words.insert(planned.words.end(), original_access.begin(),
                         original_access.begin() + access_words);
    if (!append_delay_words(planned.words, arch, options, result.errors,
                            "ConSan flat check/trap proof"))
      return;
    const bool needs_runtime_group_gate =
        arch == ROCJITSU_CODE_ARCH_GFX1250 &&
        site.address_space_hint == ConSanFlatAddressSpaceHint::MaybeGroup;
    std::optional<size_t> runtime_group_gate_branch_index;
    if (needs_runtime_group_gate) {
      constexpr uint16_t kScalarInlineMinusOne = 0xC1;
      const auto group_aperture_high = instrumentation::build_v_cmp_eq_u32_vcc(
          kScalarInlineMinusOne, static_cast<uint16_t>(*site.addr_vgpr + 1u), arch);
      if (!group_aperture_high) {
        result.errors.emplace_back(
            "ConSan flat check/trap proof could not encode runtime group gate");
        return;
      }
      planned.words.push_back(*group_aperture_high);
      runtime_group_gate_branch_index = planned.words.size();
      planned.words.push_back(build_s_nop(0, arch));
    }
    planned.words.insert(planned.words.end(), duplicate_load->begin(),
                         duplicate_load->begin() + access_words);
    planned.words.push_back(*wait_flat);
    const auto load_subword = consan_flat_load_subword_semantics(site.mnemonic);
    const auto store_subword = consan_flat_store_subword_semantics(site.mnemonic);
    const bool compare_high_load =
        load_subword && load_subword->placement == ConSanFlatSubwordPlacement::High16;
    const uint16_t comparison_scratch_vgprs = flat_check_trap_comparison_scratch_vgprs(site);
    uint16_t report_scratch_vgpr =
        static_cast<uint16_t>(candidate.scratch_vgpr + *required_vgprs + comparison_scratch_vgprs);
    if (options.report_buffer_address &&
        (arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4) &&
        report_scratch_vgpr % 2u != 0)
      ++report_scratch_vgpr;
    auto mismatch_action = build_mismatch_action_words(
        options, arch, report_scratch_vgpr, result.errors, "ConSan flat check/trap proof");
    if (!mismatch_action)
      return;
    if (mismatch_action->size() > static_cast<size_t>(std::numeric_limits<int16_t>::max())) {
      result.errors.emplace_back("ConSan flat check/trap proof mismatch action is too "
                                 "large");
      return;
    }
    if (*required_vgprs > 1u) {
      std::vector<size_t> mismatch_branch_indices;
      mismatch_branch_indices.reserve(*required_vgprs);
      for (uint16_t i = 0; i < *required_vgprs; ++i) {
        const auto chunk_cmp_ne =
            build_v_cmp_ne_u32_e32_word(static_cast<uint16_t>(*compare_vgpr + i),
                                        static_cast<uint16_t>(candidate.scratch_vgpr + i), arch);
        if (!chunk_cmp_ne) {
          result.errors.emplace_back("ConSan flat check/trap proof could not encode chunk compare");
          return;
        }
        planned.words.push_back(*chunk_cmp_ne);
        mismatch_branch_indices.push_back(planned.words.size());
        planned.words.push_back(build_s_nop(0, arch));
      }
      const auto skip_action = build_s_branch(static_cast<int16_t>(mismatch_action->size()), arch);
      planned.words.push_back(skip_action);
      const size_t action_index = planned.words.size();
      planned.words.insert(planned.words.end(), mismatch_action->begin(), mismatch_action->end());
      for (size_t branch_index : mismatch_branch_indices) {
        const size_t distance = action_index - (branch_index + 1u);
        if (distance > static_cast<size_t>(std::numeric_limits<int16_t>::max())) {
          result.errors.emplace_back(
              "ConSan flat check/trap proof shared mismatch branch is out of range");
          return;
        }
        const auto branch =
            instrumentation::build_s_cbranch_vccnz(static_cast<int16_t>(distance), arch);
        if (!branch) {
          result.errors.emplace_back(
              "ConSan flat check/trap proof could not encode shared mismatch branch");
          return;
        }
        planned.words[branch_index] = *branch;
      }
    } else {
      uint16_t compare_lhs_vgpr = *compare_vgpr;
      uint16_t compare_rhs_vgpr = candidate.scratch_vgpr;
      const uint16_t comparison_scratch_vgpr =
          static_cast<uint16_t>(candidate.scratch_vgpr + *required_vgprs);
      if (compare_high_load) {
        const uint16_t original_high_vgpr = comparison_scratch_vgpr;
        const auto select_original_high = instrumentation::build_v_lshrrev_b32(
            original_high_vgpr, scalar_positive_inline_u32(16u), *compare_vgpr, arch);
        const auto select_duplicate_high = instrumentation::build_v_lshrrev_b32(
            candidate.scratch_vgpr, scalar_positive_inline_u32(16u), candidate.scratch_vgpr, arch);
        if (!select_original_high || !select_duplicate_high) {
          result.errors.emplace_back(
              "ConSan flat check/trap proof could not select high-half load values");
          return;
        }
        planned.words.push_back(*select_original_high);
        planned.words.push_back(*select_duplicate_high);
        compare_lhs_vgpr = original_high_vgpr;
      } else if (store_subword) {
        if (store_subword->placement == ConSanFlatSubwordPlacement::High16) {
          const auto select_store_high = instrumentation::build_v_lshrrev_b32(
              comparison_scratch_vgpr, scalar_positive_inline_u32(16u), *compare_vgpr, arch);
          if (!select_store_high) {
            result.errors.emplace_back(
                "ConSan flat check/trap proof could not select high-half store value");
            return;
          }
          planned.words.push_back(*select_store_high);
          compare_lhs_vgpr = comparison_scratch_vgpr;
        }
        if (store_subword->memory_width_bits == 8u) {
          const auto mask_store_byte = instrumentation::build_v_and_b32_literal(
              comparison_scratch_vgpr, 0xffu, compare_lhs_vgpr, arch);
          if (!mask_store_byte) {
            result.errors.emplace_back(
                "ConSan flat check/trap proof could not mask byte store value");
            return;
          }
          planned.words.insert(planned.words.end(), mask_store_byte->begin(),
                               mask_store_byte->end());
          compare_lhs_vgpr = comparison_scratch_vgpr;
        }
      }
      const auto skip_action =
          build_s_cbranch_vccz_word(static_cast<int16_t>(mismatch_action->size()), arch);
      if (!skip_action) {
        result.errors.emplace_back("ConSan flat check/trap proof could not encode "
                                   "mismatch-action branch");
        return;
      }
      const auto chunk_cmp_ne =
          load_subword || store_subword || site.width_bits == 16
              ? build_v_cmp_ne_u16_e32_word(compare_lhs_vgpr, compare_rhs_vgpr, arch)
              : build_v_cmp_ne_u32_e32_word(compare_lhs_vgpr, compare_rhs_vgpr, arch);
      if (!chunk_cmp_ne) {
        result.errors.emplace_back("ConSan flat check/trap proof could not encode chunk compare");
        return;
      }
      planned.words.push_back(*chunk_cmp_ne);
      planned.words.push_back(*skip_action);
      planned.words.insert(planned.words.end(), mismatch_action->begin(), mismatch_action->end());
    }
    const size_t runtime_group_gate_target_index = planned.words.size();
    planned.words.push_back(*restore_vcc);
    if (runtime_group_gate_branch_index) {
      const size_t distance =
          runtime_group_gate_target_index - (*runtime_group_gate_branch_index + 1u);
      if (distance > static_cast<size_t>(std::numeric_limits<int16_t>::max())) {
        result.errors.emplace_back(
            "ConSan flat check/trap proof runtime group gate is out of range");
        return;
      }
      const auto skip_probe = build_s_cbranch_vccz_word(static_cast<int16_t>(distance), arch);
      if (!skip_probe) {
        result.errors.emplace_back(
            "ConSan flat check/trap proof could not encode runtime group branch");
        return;
      }
      planned.words[*runtime_group_gate_branch_index] = *skip_probe;
    }
    if (candidate.vgpr_spill || candidate.dynamic_stack_spill) {
      if (candidate.dynamic_stack_spill) {
        if (!candidate.scalar_vcc_spill_vgpr ||
            candidate.scalar_vcc_spill_vgpr_count !=
                DynamicStackBorrowedSgprSpillSequence::kScalarReservoirCount) {
          result.errors.emplace_back(
              "ConSan flat check/trap proof lost its dynamic scalar spill window");
          return;
        }
        const size_t guarded_words = candidate.dynamic_stack_spill->save_words.size() +
                                     planned.words.size() +
                                     candidate.dynamic_stack_spill->restore_words.size();
        if (guarded_words > static_cast<size_t>(std::numeric_limits<int16_t>::max())) {
          result.errors.emplace_back(
              "ConSan flat check/trap proof dynamic spill guard exceeds s_cbranch_execz range");
          return;
        }
        const auto skip_empty_wave =
            instrumentation::build_s_cbranch_execz(static_cast<int16_t>(guarded_words), arch);
        if (!skip_empty_wave) {
          result.errors.emplace_back(
              "ConSan flat check/trap proof could not encode its dynamic empty-wave spill guard");
          return;
        }
        std::vector<uint32_t> wrapped;
        wrapped.reserve(1u + guarded_words);
        wrapped.push_back(*skip_empty_wave);
        wrapped.insert(wrapped.end(), candidate.dynamic_stack_spill->save_words.begin(),
                       candidate.dynamic_stack_spill->save_words.end());
        wrapped.insert(wrapped.end(), planned.words.begin(), planned.words.end());
        wrapped.insert(wrapped.end(), candidate.dynamic_stack_spill->restore_words.begin(),
                       candidate.dynamic_stack_spill->restore_words.end());
        planned.words = std::move(wrapped);
      } else {
        const bool spills_scalar_pair = candidate.scalar_vcc_spill_vgpr.has_value();
        if ((spills_scalar_pair && candidate.scalar_vcc_spill_vgpr_count != 2u) ||
            (!spills_scalar_pair && candidate.scalar_vcc_spill_vgpr_count != 0u)) {
          result.errors.emplace_back(
              "ConSan flat check/trap proof has an invalid scalar spill window");
          return;
        }
        std::array<uint32_t, 2> save_scalar_pair{};
        std::array<uint32_t, 2> restore_scalar_pair{};
        if (spills_scalar_pair) {
          for (uint16_t i = 0; i < 2u; ++i) {
            save_scalar_pair[i] =
                build_v_mov_b32_e32(static_cast<uint16_t>(*candidate.scalar_vcc_spill_vgpr + i),
                                    static_cast<uint16_t>(candidate.vcc_save_sgpr + i), arch);
            const auto restore = instrumentation::build_v_readfirstlane_b32(
                static_cast<uint16_t>(candidate.vcc_save_sgpr + i),
                static_cast<uint16_t>(*candidate.scalar_vcc_spill_vgpr + i), arch);
            if (!restore) {
              result.errors.emplace_back(
                  "ConSan flat check/trap proof could not encode its active-lane scalar restore");
              return;
            }
            restore_scalar_pair[i] = *restore;
          }
        }
        const size_t guarded_words = candidate.vgpr_spill->save_words.size() +
                                     (spills_scalar_pair ? save_scalar_pair.size() : 0u) +
                                     planned.words.size() +
                                     (spills_scalar_pair ? restore_scalar_pair.size() : 0u) +
                                     candidate.vgpr_spill->restore_words.size();
        if (guarded_words > static_cast<size_t>(std::numeric_limits<int16_t>::max())) {
          result.errors.emplace_back(
              "ConSan flat check/trap proof spill guard exceeds s_cbranch_execz range");
          return;
        }
        const auto skip_empty_wave =
            instrumentation::build_s_cbranch_execz(static_cast<int16_t>(guarded_words), arch);
        if (!skip_empty_wave) {
          result.errors.emplace_back(
              "ConSan flat check/trap proof could not encode its empty-wave spill guard");
          return;
        }
        std::vector<uint32_t> wrapped;
        wrapped.reserve(1u + guarded_words);
        wrapped.push_back(*skip_empty_wave);
        wrapped.insert(wrapped.end(), candidate.vgpr_spill->save_words.begin(),
                       candidate.vgpr_spill->save_words.end());
        if (spills_scalar_pair)
          wrapped.insert(wrapped.end(), save_scalar_pair.begin(), save_scalar_pair.end());
        wrapped.insert(wrapped.end(), planned.words.begin(), planned.words.end());
        if (spills_scalar_pair)
          wrapped.insert(wrapped.end(), restore_scalar_pair.begin(), restore_scalar_pair.end());
        wrapped.insert(wrapped.end(), candidate.vgpr_spill->restore_words.begin(),
                       candidate.vgpr_spill->restore_words.end());
        planned.words = std::move(wrapped);
      }
    } else if (candidate.scalar_vcc_spill_vgpr) {
      if (candidate.scalar_vcc_spill_vgpr_count != 1u) {
        result.errors.emplace_back(
            "ConSan flat check/trap proof has an invalid fixed-lane scalar spill window");
        return;
      }
      std::array<uint32_t, 4> save_scalar_pair{};
      std::array<uint32_t, 4> restore_scalar_pair{};
      for (uint16_t i = 0; i < 2u; ++i) {
        const auto save = instrumentation::build_v_writelane_b32(
            *candidate.scalar_vcc_spill_vgpr, static_cast<uint16_t>(candidate.vcc_save_sgpr + i), i,
            arch);
        const auto restore = instrumentation::build_v_readlane_b32(
            static_cast<uint16_t>(candidate.vcc_save_sgpr + i), *candidate.scalar_vcc_spill_vgpr, i,
            arch);
        if (!save || !restore) {
          result.errors.emplace_back(
              "ConSan flat check/trap proof could not encode its scalar VCC-save spill");
          return;
        }
        std::copy(save->begin(), save->end(), save_scalar_pair.begin() + 2u * i);
        std::copy(restore->begin(), restore->end(), restore_scalar_pair.begin() + 2u * i);
      }
      std::vector<uint32_t> wrapped;
      wrapped.reserve(save_scalar_pair.size() + planned.words.size() + restore_scalar_pair.size());
      wrapped.insert(wrapped.end(), save_scalar_pair.begin(), save_scalar_pair.end());
      wrapped.insert(wrapped.end(), planned.words.begin(), planned.words.end());
      wrapped.insert(wrapped.end(), restore_scalar_pair.begin(), restore_scalar_pair.end());
      planned.words = std::move(wrapped);
    }

    if (use_trampoline) {
      if (!candidate.placement || (candidate.use_local_cave && candidate.local_cave == nullptr)) {
        result.errors.emplace_back("ConSan flat check/trap proof selected missing cave placement");
        return;
      }
      planned.cave_words = planned.words;
      uint64_t entry_target = candidate.placement->body_offset;
      if (candidate.branch_only_continuation) {
        if (candidate.entry_island != nullptr || candidate.entry_island_at_anchor ||
            candidate.indirect_scratch || !candidate.branch_only_route ||
            candidate.placement->body_size !=
                planned.words.size() * sizeof(uint32_t) + sizeof(uint32_t)) {
          result.errors.emplace_back(
              "ConSan flat check/trap proof found inconsistent branch-only placement");
          return;
        }
        const uint64_t return_source =
            candidate.placement->body_offset + planned.words.size() * sizeof(uint32_t);
        const uint64_t return_target =
            candidate.branch_only_route->return_relay_offsets.empty()
                ? candidate.placement->return_target
                : candidate.branch_only_route->return_relay_offsets.front();
        const auto ret = compute_sopp_branch_simm16(return_source, return_target);
        if (!ret) {
          result.errors.emplace_back(
              "ConSan flat check/trap proof branch-only return is unreachable");
          return;
        }
        planned.cave_words.push_back(build_s_branch(*ret, arch));
        entry_target = candidate.branch_only_route->entry_relay_offsets.empty()
                           ? candidate.placement->body_offset
                           : candidate.branch_only_route->entry_relay_offsets.front();
      } else if (candidate.entry_island != nullptr || candidate.entry_island_at_anchor) {
        if (!candidate.indirect_scratch) {
          result.errors.emplace_back("ConSan flat check/trap proof lost indirect scalar scratch");
          return;
        }
        const auto &scalar = *candidate.indirect_scratch;
        planned.info.indirect_pc_sgpr = scalar.pc_sgpr_pair;
        planned.info.indirect_saved_scc_sgpr = scalar.scc_save_sgpr;
        planned.info.indirect_saved_vcc_sgpr = scalar.vcc_save_sgpr;
        if (candidate.entry_island_at_anchor) {
          const auto restore_scc = instrumentation::build_s_cmp_lg_u32(
              scalar.scc_save_sgpr, scalar_positive_inline_u32(0), arch);
          if (!restore_scc) {
            result.errors.emplace_back(
                "ConSan flat check/trap proof could not restore SCC before displaced tail");
            return;
          }
          planned.cave_words.push_back(*restore_scc);
          planned.cave_words.insert(planned.cave_words.end(),
                                    candidate.displaced_tail_words.begin(),
                                    candidate.displaced_tail_words.end());
        }
        planned.info.indirect_return_offset =
            candidate.placement->body_offset + planned.cave_words.size() * sizeof(uint32_t);
        planned.info.indirect_required_sgpr_count =
            std::max<uint16_t>(std::max<uint16_t>(static_cast<uint16_t>(scalar.pc_sgpr_pair + 2u),
                                                  static_cast<uint16_t>(scalar.scc_save_sgpr + 1u)),
                               static_cast<uint16_t>(scalar.vcc_save_sgpr + 2u));
        const bool return_encoded = candidate.entry_island_at_anchor
                                        ? append_lds_check_trap_indirect_jump(
                                              planned.cave_words, candidate.placement->body_offset,
                                              candidate.placement->return_target,
                                              scalar.pc_sgpr_pair, scalar.scc_save_sgpr, arch)
                                        : append_lds_check_trap_indirect_return(
                                              planned.cave_words, candidate.placement->body_offset,
                                              candidate.placement->return_target,
                                              scalar.pc_sgpr_pair, scalar.scc_save_sgpr, arch);
        const uint64_t entry_island_offset = candidate.entry_island_at_anchor
                                                 ? site.text_offset
                                                 : candidate.entry_island->text_offset;
        if (!return_encoded ||
            planned.cave_words.size() * sizeof(uint32_t) != candidate.placement->body_size ||
            !append_lds_check_trap_indirect_jump(planned.entry_island_words, entry_island_offset,
                                                 candidate.placement->body_offset,
                                                 scalar.pc_sgpr_pair, scalar.scc_save_sgpr, arch)) {
          result.errors.emplace_back(
              "ConSan flat check/trap proof could not encode indirect relay");
          return;
        }
        entry_target = entry_island_offset;
      } else {
        if (planned.words.size() * sizeof(uint32_t) != candidate.placement->body_size) {
          result.errors.emplace_back(
              "ConSan flat check/trap proof body size changed after placement");
          return;
        }
        const auto ret = compute_sopp_branch_simm16(candidate.placement->return_branch_offset,
                                                    candidate.placement->return_target);
        if (!ret) {
          result.errors.emplace_back(
              "ConSan flat check/trap proof cave return exceeds s_branch simm16");
          return;
        }
        planned.cave_words.push_back(build_s_branch(*ret, arch));
      }
      const auto fwd =
          candidate.entry_island_at_anchor
              ? std::optional<int16_t>(0)
              : compute_sopp_branch_simm16(candidate.placement->anchor_offset, entry_target);
      if (!fwd) {
        result.errors.emplace_back("ConSan flat check/trap proof relay entry is unreachable");
        return;
      }
      if (candidate.use_local_cave &&
          candidate.local_cave->word_count < planned.cave_words.size()) {
        result.errors.emplace_back("ConSan flat check/trap proof local cave is too small");
        return;
      }

      if (!candidate.entry_island_at_anchor) {
        planned.anchor_words.assign(access_words, build_s_nop(0, arch));
        planned.anchor_words.front() = build_s_branch(*fwd, arch);
      }
      planned.info.trampoline_offset = candidate.placement->body_offset;
      planned.info.original_size = candidate.placement->original_size;
      planned.info.trampoline_size =
          static_cast<uint32_t>(planned.cave_words.size() * sizeof(uint32_t));
    } else {
      planned.info.trampoline_offset = site.text_offset + site.size;
      planned.info.original_size = static_cast<uint32_t>(planned.words.size() * sizeof(uint32_t));
      planned.info.trampoline_size = 0;
    }

    planned_patches.push_back(std::move(planned));
  }

  std::unordered_map<uint64_t, size_t> flat_anchor_relay_donors;
  for (size_t index = 0; index < planned_patches.size(); ++index) {
    const PlannedFlatCheckTrapPatch &planned = planned_patches[index];
    for (uint64_t offset = planned.info.anchor_offset + sizeof(uint32_t);
         offset < planned.info.anchor_offset + planned.info.original_size;
         offset += sizeof(uint32_t)) {
      if (selected_flat_anchor_relay_offsets.contains(offset))
        flat_anchor_relay_donors.emplace(offset, index);
    }
  }
  for (size_t index = 0; index < planned_patches.size(); ++index) {
    const PlannedFlatCheckTrapPatch &planned = planned_patches[index];
    if (!planned.branch_only_route)
      continue;
    const auto donor_precedes_borrower = [&](uint64_t relay) {
      if (!selected_flat_anchor_relay_offsets.contains(relay))
        return true;
      const auto donor = flat_anchor_relay_donors.find(relay);
      return donor != flat_anchor_relay_donors.end() && donor->second < index;
    };
    if (!std::ranges::all_of(planned.branch_only_route->entry_relay_offsets,
                             donor_precedes_borrower) ||
        !std::ranges::all_of(planned.branch_only_route->return_relay_offsets,
                             donor_precedes_borrower)) {
      result.errors.emplace_back(
          "ConSan flat check/trap proof planned an anchor-tail relay before its donor");
      return;
    }
  }

  std::vector<uint8_t> patched_bytes;
  if (result.elf_bytes.empty())
    patched_bytes.assign(original_bytes.begin(), original_bytes.end());
  else
    patched_bytes = result.elf_bytes;

  std::vector<ConSanPatchInfo> new_patch_infos;
  new_patch_infos.reserve(planned_patches.size());
  for (const PlannedFlatCheckTrapPatch &planned : planned_patches) {
    const ConSanFlatSite &site = *planned.site;
    if (planned.use_local_cave) {
      if (planned.local_cave == nullptr) {
        result.errors.emplace_back("ConSan flat check/trap proof selected missing local cave");
        return;
      }
      const uint64_t anchor_bytes = planned.anchor_words.size() * sizeof(uint32_t);
      const uint64_t cave_bytes = planned.cave_words.size() * sizeof(uint32_t);
      if (site.file_offset > patched_bytes.size() ||
          anchor_bytes > patched_bytes.size() - site.file_offset ||
          planned.local_cave->file_offset > patched_bytes.size() ||
          cave_bytes > patched_bytes.size() - planned.local_cave->file_offset) {
        result.errors.emplace_back("ConSan flat check/trap proof exceeds ELF bytes");
        return;
      }

      std::memcpy(patched_bytes.data() + site.file_offset, planned.anchor_words.data(),
                  static_cast<size_t>(anchor_bytes));
      std::memcpy(patched_bytes.data() + planned.local_cave->file_offset, planned.cave_words.data(),
                  static_cast<size_t>(cave_bytes));
    } else if (!planned.use_appended_cave) {
      const uint64_t patch_bytes = static_cast<uint64_t>(planned.words.size() * sizeof(uint32_t));
      if (site.file_offset > patched_bytes.size() ||
          patch_bytes > patched_bytes.size() - site.file_offset) {
        result.errors.emplace_back("ConSan flat check/trap proof exceeds ELF bytes");
        return;
      }
      std::memcpy(patched_bytes.data() + site.file_offset, planned.words.data(),
                  static_cast<size_t>(patch_bytes));
    }

    new_patch_infos.push_back(planned.info);
    if (planned.entry_island != nullptr && !planned.entry_island_at_anchor) {
      ConSanPatchInfo island_info;
      island_info.kind = ConSanPatchKind::TrampolineScIndirectBranchIsland;
      island_info.anchor_offset = planned.info.anchor_offset;
      island_info.trampoline_offset = planned.entry_island->text_offset;
      island_info.original_size = 0;
      island_info.trampoline_size =
          static_cast<uint32_t>(planned.entry_island_words.size() * sizeof(uint32_t));
      new_patch_infos.push_back(std::move(island_info));
    }
  }
  const bool has_appended_patch =
      std::ranges::any_of(planned_patches, &PlannedFlatCheckTrapPatch::use_appended_cave);
  std::vector<DescriptorSgprGrowth> descriptor_sgpr_growths;
  std::unordered_map<uint64_t, uint32_t> descriptor_private_requirements;
  for (const PlannedFlatCheckTrapPatch &planned : planned_patches) {
    for (uint64_t owner : planned.info.owner_descriptor_file_offsets) {
      descriptor_sgpr_growths.push_back(
          {owner,
           std::max(planned.info.required_sgpr_count, planned.info.indirect_required_sgpr_count)});
      if (planned.info.required_private_segment_size != 0) {
        uint32_t &required = descriptor_private_requirements[owner];
        required = std::max(required, planned.info.required_private_segment_size);
      }
    }
  }
  if (has_appended_patch) {
    CodeObjectPatcher patcher(code_object);
    if (!apply_descriptor_sgpr_growths_to_patcher(patcher, descriptor_sgpr_growths, arch,
                                                  result.errors))
      return;
    for (const auto &[descriptor_offset, required_private_bytes] :
         descriptor_private_requirements) {
      const std::span<const uint8_t> image = patcher.image_bytes();
      if (descriptor_offset > image.size() || sizeof(KD) > image.size() - descriptor_offset) {
        result.errors.emplace_back("ConSan flat check/trap spill descriptor exceeds ELF bytes");
        return;
      }
      std::array<uint8_t, sizeof(KD)> descriptor{};
      std::memcpy(descriptor.data(), image.data() + descriptor_offset, descriptor.size());
      const SpillDescriptorUpdate update =
          update_kernel_descriptor_for_spills(descriptor, 0, required_private_bytes);
      if ((update != SpillDescriptorUpdate::Updated &&
           update != SpillDescriptorUpdate::Unchanged) ||
          !patcher.patch_kernel_descriptor(descriptor_offset, descriptor)) {
        result.errors.emplace_back(
            "ConSan flat check/trap could not grow its private spill descriptor");
        return;
      }
    }
    std::vector<uint8_t> new_text(patcher.text_bytes().begin(), patcher.text_bytes().end());
    for (const BranchOnlyDirectRelayReservoir &reservoir : direct_branch_reservoirs.reservoirs) {
      std::string relay_error;
      if (!BranchOnlyRelayRouter::emit_direct_reservoir(new_text, reservoir, arch, new_patch_infos,
                                                        &relay_error)) {
        result.errors.emplace_back("ConSan flat check/trap proof " + relay_error);
        return;
      }
    }
    for (const PlannedFlatCheckTrapPatch &planned : planned_patches) {
      const ConSanFlatSite &site = *planned.site;
      const uint64_t anchor_bytes = planned.entry_island_at_anchor
                                        ? planned.entry_island_words.size() * sizeof(uint32_t)
                                        : planned.anchor_words.size() * sizeof(uint32_t);
      if (planned.use_appended_cave) {
        if (!planned.placement) {
          result.errors.emplace_back("ConSan flat check/trap proof lost appended placement");
          return;
        }
        const uint64_t body_offset = planned.placement->body_offset;
        const uint64_t body_bytes = planned.cave_words.size() * sizeof(uint32_t);
        if (body_offset > std::numeric_limits<size_t>::max() - body_bytes) {
          result.errors.emplace_back("ConSan flat check/trap proof appended body overflows text");
          return;
        }
        new_text.resize(std::max<uint64_t>(new_text.size(), body_offset + body_bytes), 0);
        const uint32_t *anchor_data = planned.entry_island_at_anchor
                                          ? planned.entry_island_words.data()
                                          : planned.anchor_words.data();
        std::memcpy(new_text.data() + site.text_offset, anchor_data, anchor_bytes);
        std::memcpy(new_text.data() + body_offset, planned.cave_words.data(), body_bytes);
        if (planned.entry_island != nullptr) {
          std::memcpy(new_text.data() + planned.entry_island->text_offset,
                      planned.entry_island_words.data(),
                      planned.entry_island_words.size() * sizeof(uint32_t));
        }
      } else if (planned.use_local_cave) {
        const uint64_t body_bytes = planned.cave_words.size() * sizeof(uint32_t);
        std::memcpy(new_text.data() + site.text_offset, planned.anchor_words.data(), anchor_bytes);
        std::memcpy(new_text.data() + planned.local_cave->text_offset, planned.cave_words.data(),
                    body_bytes);
      } else {
        std::memcpy(new_text.data() + site.text_offset, planned.words.data(),
                    planned.words.size() * sizeof(uint32_t));
      }
      if (planned.branch_only_route) {
        std::string relay_error;
        if (!BranchOnlyRelayRouter::emit_and_record(
                new_text, *planned.branch_only_route, planned.info.trampoline_offset,
                planned.info.anchor_offset + planned.info.original_size, arch, new_patch_infos,
                &relay_error)) {
          result.errors.emplace_back("ConSan flat check/trap proof " + relay_error);
          return;
        }
      }
    }
    if (!replace_consan_text(patcher, new_text, options, "flat check/trap proof", result)) {
      return;
    }
    result.elf_bytes = std::move(patcher).emit();
  } else {
    std::optional<AmdGpuCodeObject> current_object;
    if (!result.elf_bytes.empty())
      current_object.emplace(patched_bytes.data(), patched_bytes.size());
    const auto current_descriptor_offset =
        [&](uint64_t original_descriptor_offset) -> std::optional<uint64_t> {
      if (!current_object)
        return original_descriptor_offset;
      const auto original_kernel =
          std::ranges::find(code_object.kernels(), original_descriptor_offset,
                            &AmdGpuKernelInfo::descriptor_file_offset);
      if (original_kernel == code_object.kernels().end())
        return std::nullopt;
      const auto current_kernel = std::ranges::find(current_object->kernels(),
                                                    original_kernel->name, &AmdGpuKernelInfo::name);
      if (current_kernel == current_object->kernels().end())
        return std::nullopt;
      return current_kernel->descriptor_file_offset;
    };
    for (const auto &[descriptor_offset, required_count] :
         merge_descriptor_sgpr_growths(descriptor_sgpr_growths)) {
      const auto active_descriptor_offset = current_descriptor_offset(descriptor_offset);
      if (!active_descriptor_offset) {
        result.errors.emplace_back(
            "ConSan flat check/trap proof could not resolve SGPR descriptor owner");
        return;
      }
      if (*active_descriptor_offset > patched_bytes.size() ||
          sizeof(KD) > patched_bytes.size() - *active_descriptor_offset) {
        result.errors.emplace_back(
            "ConSan flat check/trap proof SGPR descriptor exceeds ELF bytes");
        return;
      }
      KD descriptor{};
      std::memcpy(&descriptor, patched_bytes.data() + *active_descriptor_offset,
                  sizeof(descriptor));
      if (!grow_descriptor_sgpr_allocation(descriptor, required_count, arch)) {
        result.errors.emplace_back("ConSan flat check/trap proof could not grow SGPR allocation");
        return;
      }
      std::memcpy(patched_bytes.data() + *active_descriptor_offset, &descriptor,
                  sizeof(descriptor));
    }
    for (const auto &[descriptor_offset, required_private_bytes] :
         descriptor_private_requirements) {
      const auto active_descriptor_offset = current_descriptor_offset(descriptor_offset);
      if (!active_descriptor_offset) {
        result.errors.emplace_back(
            "ConSan flat check/trap proof could not resolve private spill descriptor owner");
        return;
      }
      const SpillDescriptorUpdate update = update_kernel_descriptor_for_spills(
          patched_bytes, *active_descriptor_offset, required_private_bytes);
      if (update != SpillDescriptorUpdate::Updated && update != SpillDescriptorUpdate::Unchanged) {
        result.errors.emplace_back(
            "ConSan flat check/trap could not grow its private spill descriptor");
        return;
      }
    }
    result.elf_bytes = std::move(patched_bytes);
  }
  result.patches.insert(result.patches.end(), new_patch_infos.begin(), new_patch_infos.end());
  result.modified = true;
}

void try_apply_flat_trap_patch(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                               const ConSanOptions &options, ConSanResult &result) {
  if (arch != ROCJITSU_CODE_ARCH_RDNA4) {
    result.warnings.emplace_back("ConSan flat trap proof currently supports only RDNA4");
    return;
  }

  auto trap = build_s_trap_word(0, arch);
  if (!trap) {
    result.errors.emplace_back("ConSan flat trap proof could not encode s_trap");
    return;
  }

  std::vector<const ConSanFlatSite *> candidates;
  for (const ConSanKernelInfo &kernel : result.kernels) {
    if (kernel.preflight_action == ConSanPreflightAction::Reject)
      continue;
    for (const ConSanFlatSite &site : kernel.flat_sites) {
      if (is_supported_flat_trap_site(site, options.flat_provenance_mode))
        candidates.push_back(&site);
    }
  }
  for (const ConSanFunctionInfo &function : result.functions) {
    for (const ConSanFlatSite &site : function.flat_sites) {
      if (is_supported_flat_trap_site(site, options.flat_provenance_mode))
        candidates.push_back(&site);
    }
  }

  if (candidates.empty()) {
    result.warnings.emplace_back(
        "ConSan flat trap proof found no likely group flat_load/store site");
    return;
  }

  std::ranges::sort(candidates, [](const ConSanFlatSite *lhs, const ConSanFlatSite *rhs) {
    if (lhs->file_offset != rhs->file_offset)
      return lhs->file_offset < rhs->file_offset;
    return lhs->text_offset < rhs->text_offset;
  });

  result.elf_bytes.assign(reinterpret_cast<const uint8_t *>(code_object.image_data()),
                          reinterpret_cast<const uint8_t *>(code_object.image_data()) +
                              code_object.image_size());
  const std::array<uint32_t, 3> replacement = {*trap, build_s_nop(0, arch), build_s_nop(0, arch)};
  for (const ConSanFlatSite *site : candidates) {
    if (site->file_offset > result.elf_bytes.size() ||
        site->size > result.elf_bytes.size() - site->file_offset) {
      result.errors.emplace_back("ConSan flat trap proof exceeds ELF bytes");
      result.elf_bytes.clear();
      result.patches.clear();
      result.modified = false;
      return;
    }
    std::memcpy(result.elf_bytes.data() + site->file_offset, replacement.data(),
                replacement.size() * sizeof(uint32_t));

    ConSanPatchInfo info;
    info.kind = ConSanPatchKind::InlineFlatTrapRewrite;
    info.anchor_offset = site->text_offset;
    info.trampoline_offset = site->text_offset + site->size;
    info.original_size = site->size;
    result.patches.push_back(info);
  }
  result.modified = true;
}

} // namespace
