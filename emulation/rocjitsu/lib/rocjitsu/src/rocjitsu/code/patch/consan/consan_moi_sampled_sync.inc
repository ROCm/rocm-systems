struct SampledBarrierWindow {
  uint32_t first_slot = 0;
  uint32_t bank_count = 1;

  auto operator<=>(const SampledBarrierWindow &) const = default;
};

struct SampledBarrierCandidate {
  const ConSanSyncSequence *sequence = nullptr;
  BarrierRecordCandidate barrier;
  std::vector<SampledBarrierWindow> windows;
  bool uses_gfx1250_cluster_workgroup_id = false;
};

struct SampledCfgDistance {
  uint32_t block_edges = 0;
  uint64_t text_distance = 0;

  auto operator<=>(const SampledCfgDistance &) const = default;
};

struct SampledCfgForwardDistanceIndex {
  BasicBlock *target = nullptr;
  uint64_t target_text_offset = 0;
  std::unordered_map<BasicBlock *, uint32_t> block_edges;
};

[[nodiscard]] bool append_sampled_indexed_address(std::vector<uint32_t> &words,
                                                  uint64_t first_address, uint32_t element_size,
                                                  uint16_t index_vgpr, uint16_t address_vgpr,
                                                  uint16_t temporary_vgpr, rj_code_arch_t arch) {
  const auto low = instrumentation::build_v_mov_b32_literal(
      address_vgpr, static_cast<uint32_t>(first_address), arch);
  const auto high = instrumentation::build_v_mov_b32_literal(
      static_cast<uint16_t>(address_vgpr + 1u), static_cast<uint32_t>(first_address >> 32u), arch);
  const auto offset = instrumentation::build_v_mul_lo_u32_literal(temporary_vgpr, temporary_vgpr,
                                                                  element_size, index_vgpr, arch);
  const auto add = instrumentation::build_v_add_u64_vgpr_offset(address_vgpr, temporary_vgpr, arch);
  if (!low || !high || !offset || !add)
    return false;
  words.insert(words.end(), low->begin(), low->end());
  words.insert(words.end(), high->begin(), high->end());
  words.insert(words.end(), offset->begin(), offset->end());
  words.insert(words.end(), add->begin(), add->end());
  return true;
}

[[nodiscard]] std::optional<SampledCfgForwardDistanceIndex>
build_sampled_cfg_forward_distance_index(const MoiResourcePlanningState &state,
                                         const MoiKernelResourceContext &context,
                                         uint64_t target_text_offset) {
  BasicBlock *target = block_for_offset(state.block_index, target_text_offset);
  if (target == nullptr)
    return std::nullopt;
  const std::unordered_set<BasicBlock *> scope(context.scope.blocks.begin(),
                                               context.scope.blocks.end());
  if (!scope.contains(target))
    return std::nullopt;

  std::unordered_map<BasicBlock *, std::vector<BasicBlock *>> predecessors;
  predecessors.reserve(scope.size());
  for (BasicBlock *block : context.scope.blocks) {
    for (BasicBlock *successor : block->successors()) {
      if (scope.contains(successor))
        predecessors[successor].push_back(block);
    }
  }
  for (const ScopedCfgEdge &edge : context.scope.liveness_edges) {
    if (edge.from != nullptr && edge.to != nullptr && scope.contains(edge.from) &&
        scope.contains(edge.to))
      predecessors[edge.to].push_back(edge.from);
  }

  SampledCfgForwardDistanceIndex index;
  index.target = target;
  index.target_text_offset = target_text_offset;
  index.block_edges.reserve(scope.size());
  std::vector<BasicBlock *> pending{target};
  index.block_edges.emplace(target, 0u);
  for (size_t cursor = 0; cursor < pending.size(); ++cursor) {
    BasicBlock *block = pending[cursor];
    const uint32_t next_distance = index.block_edges.at(block) + 1u;
    auto found = predecessors.find(block);
    if (found == predecessors.end())
      continue;
    std::ranges::sort(found->second);
    found->second.erase(std::ranges::unique(found->second).begin(), found->second.end());
    for (BasicBlock *predecessor : found->second) {
      if (index.block_edges.emplace(predecessor, next_distance).second)
        pending.push_back(predecessor);
    }
  }
  return index;
}

[[nodiscard]] std::optional<SampledCfgDistance>
sampled_cfg_forward_distance(const MoiResourcePlanningState &state,
                             const SampledCfgForwardDistanceIndex &index,
                             uint64_t from_text_offset) {
  BasicBlock *from = block_for_offset(state.block_index, from_text_offset);
  if (from == nullptr)
    return std::nullopt;
  if (from == index.target && from_text_offset > index.target_text_offset)
    return std::nullopt;
  const auto found = index.block_edges.find(from);
  if (found == index.block_edges.end())
    return std::nullopt;
  const uint64_t text_distance = from_text_offset < index.target_text_offset
                                     ? index.target_text_offset - from_text_offset
                                     : from_text_offset - index.target_text_offset;
  return SampledCfgDistance{found->second, text_distance};
}

[[nodiscard]] std::optional<SampledCfgDistance>
sampled_cfg_forward_distance(const MoiResourcePlanningState &state,
                             const MoiKernelResourceContext &context, uint64_t from_text_offset,
                             uint64_t to_text_offset) {
  const auto index = build_sampled_cfg_forward_distance_index(state, context, to_text_offset);
  if (!index)
    return std::nullopt;
  return sampled_cfg_forward_distance(state, *index, from_text_offset);
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_sampled_barrier_sync_cave_words(
    std::span<const uint8_t> bytes, const SampledBarrierCandidate &candidate,
    const ConSanOptions &options, const VgprSpillSequence *spill,
    const InlineSgprSpillSequence *scalar_spill, rj_code_arch_t arch,
    const ConSanMoiReportBufferLayout &layout, uint64_t cave_text_offset,
    uint64_t return_text_offset, const MoiPrivateEpochLayout *private_layout,
    std::optional<uint16_t> gfx1250_vgpr_msb_mode, std::vector<std::string> &errors) {
  const bool scalar_persistent = moi_uses_scalar_persistent_state(options);
  if (!options.scratch_vgpr || !options.moi_report_buffer_address || !options.moi_epoch_vgpr ||
      !options.moi_exec_save_sgpr || !options.moi_owner_vgpr ||
      options.moi_owner_source != ConSanMoiOwnerSource::WorkitemId ||
      !candidate.barrier.kernel_descriptor_file_offset ||
      static_cast<uint32_t>(*options.scratch_vgpr) + (scalar_persistent ? 9u : 7u) > kMaxVgprs ||
      candidate.windows.empty() ||
      std::ranges::any_of(candidate.windows, [&](const SampledBarrierWindow &window) {
        return window.first_slot >= layout.sampled_sync_metadata_capacity ||
               window.bank_count == 0 ||
               window.bank_count > layout.sampled_sync_metadata_capacity - window.first_slot;
      }))
    return std::nullopt;
  const auto workgroup_sources =
      moi_descriptor_workgroup_sources(bytes, *candidate.barrier.kernel_descriptor_file_offset,
                                       arch, errors, candidate.uses_gfx1250_cluster_workgroup_id);
  if (!workgroup_sources)
    return std::nullopt;
  if (candidate.barrier.site.file_offset > bytes.size() ||
      candidate.barrier.site.size > bytes.size() - candidate.barrier.site.file_offset) {
    errors.emplace_back("ConSan MOI sampled barrier sequence exceeds ELF bytes");
    return std::nullopt;
  }
  const auto encoded = encode_consan_moi_sampled_sync_metadata({
      .kind = ConSanMoiSampledSyncKind::Barrier,
      .role = ConSanMoiSampledSyncRole::AcquireRelease,
      .scope = candidate.sequence->barrier_scope == ConSanBarrierSite::Scope::Cluster
                   ? ConSanMoiSampledSyncScope::Cluster
                   : ConSanMoiSampledSyncScope::Workgroup,
      .outcome = ConSanMoiSampledSyncOutcome::NotApplicable,
      .epoch_before = 0,
      .epoch_after = 1,
  });
  if (encoded.classification != ConSanMoiSampledSyncClassification::Valid)
    return std::nullopt;

  const uint16_t base = *options.scratch_vgpr;
  const uint16_t value = static_cast<uint16_t>(base + 2u);
  const uint16_t expected = static_cast<uint16_t>(base + 3u);
  const uint16_t next_epoch = static_cast<uint16_t>(base + 4u);
  const uint16_t lane_rank = static_cast<uint16_t>(base + 5u);
  const uint16_t bank_vgpr = static_cast<uint16_t>(base + 6u);
  const uint16_t temporary_exec = *options.moi_exec_save_sgpr;
  const uint16_t original_exec = static_cast<uint16_t>(*options.moi_exec_save_sgpr + 6u);
  const uint64_t report_base = *options.moi_report_buffer_address;
  const uint64_t first_window_address = report_base + layout.sampled_causal_windows_offset;
  const uint64_t first_metadata_address = report_base + layout.sampled_sync_metadata_offset;

  std::vector<uint32_t> words;
  InstructionSequence sequence(words);
  ConSanMoiRecordEmitter record(words, base, value, arch);
  const auto collision_label = sequence.make_label();
  const auto restore_label = sequence.make_label();
  const auto fast_return_label = sequence.make_label();
  const auto matched_window_label = sequence.make_label();
  const bool control_gfx1250_vgpr_bank = arch == ROCJITSU_CODE_ARCH_GFX1250;
  const bool restore_guest_vgpr_bank =
      control_gfx1250_vgpr_bank && gfx1250_vgpr_msb_mode.value_or(0u) != 0u;
  // A patched s_call_i64 can be immediately followed by the guest's next
  // s_set_vgpr_msb.  That program-state update may take effect as control
  // transfers, even when the replaced barrier itself ran in mode zero.  Make
  // every gfx1250 cave establish its low-register invariant explicitly before
  // touching scratch VGPRs.
  if (control_gfx1250_vgpr_bank)
    words.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
        static_cast<uint8_t>(gfx1250_vgpr_msb_mode.value_or(0u)), 0u, arch));
  if (spill)
    words.insert(words.end(), spill->save_words.begin(), spill->save_words.end());
  if (scalar_spill)
    words.insert(words.end(), scalar_spill->save_words.begin(), scalar_spill->save_words.end());
  if (restore_guest_vgpr_bank)
    words.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
        0u, static_cast<uint8_t>(*gfx1250_vgpr_msb_mode), arch));
  for (uint64_t offset = 0; offset < candidate.barrier.site.size; offset += sizeof(uint32_t)) {
    uint32_t word = 0;
    std::memcpy(&word, bytes.data() + candidate.barrier.site.file_offset + offset, sizeof(word));
    words.push_back(word);
  }
  if (restore_guest_vgpr_bank)
    words.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
        static_cast<uint8_t>(*gfx1250_vgpr_msb_mode), 0u, arch));
  // Runtime sampling selects LDS addresses, not workgroups. Every workgroup
  // which reaches the guest barrier must therefore inspect its banked causal
  // window and advance its epoch. A separate workgroup-residue gate can skip a
  // workgroup which touched a selected address, leaving its recorded accesses
  // in one epoch with no synchronization metadata.
  if (private_layout && !append_sampled_private_owner_epoch_load(
                            words, bytes, *candidate.barrier.kernel_descriptor_file_offset, options,
                            *private_layout, arch, errors))
    return std::nullopt;
  if (scalar_persistent) {
    words.push_back(
        build_v_mov_b32_e32(*options.moi_owner_vgpr, *options.moi_persistent_owner_sgpr, arch));
    words.push_back(
        build_v_mov_b32_e32(*options.moi_epoch_vgpr, *options.moi_persistent_epoch_sgpr, arch));
  }
  if (!append_save_moi_special_state(words, options, arch)) {
    errors.emplace_back("ConSan MOI sampled barrier failed at special-state save");
    return std::nullopt;
  }
  const auto save_exec = instrumentation::build_s_mov_b64(original_exec, kRdna4ExecLo, arch);
  if (!save_exec) {
    errors.emplace_back("ConSan MOI sampled barrier failed at EXEC snapshot");
    return std::nullopt;
  }
  words.push_back(*save_exec);
  for (const SampledBarrierWindow &window : candidate.windows) {
    const auto next_window_label = sequence.make_label();
    const auto restore_scan_exec =
        instrumentation::build_s_mov_b64(kRdna4ExecLo, original_exec, arch);
    if (!restore_scan_exec) {
      errors.emplace_back("ConSan MOI sampled barrier failed at window-scan EXEC restoration");
      return std::nullopt;
    }
    words.push_back(*restore_scan_exec);
    if (!append_sampled_window_bank_index(words, options, *workgroup_sources, window.bank_count,
                                          bank_vgpr, value, arch)) {
      errors.emplace_back("ConSan MOI sampled barrier failed at causal-window bank selection");
      return std::nullopt;
    }
    if (window.first_slot != 0u) {
      // VALUE is dead after bank selection and is distinct from BANK. Use it
      // as CDNA4's explicit literal temporary while keeping the result in BANK.
      const auto absolute_slot = instrumentation::build_v_add_u32_literal(
          bank_vgpr, value, window.first_slot, bank_vgpr, arch);
      if (!absolute_slot) {
        errors.emplace_back("ConSan MOI sampled barrier failed at absolute slot materialization");
        return std::nullopt;
      }
      words.insert(words.end(), absolute_slot->begin(), absolute_slot->end());
    }
    if (!append_sampled_indexed_address(words, first_window_address,
                                        sizeof(ConSanMoiSampledCausalWindow), bank_vgpr, base,
                                        value, arch)) {
      errors.emplace_back("ConSan MOI sampled barrier failed at causal-window addressing");
      return std::nullopt;
    }

    const auto narrow_current = [&]() -> bool {
      const auto narrow =
          instrumentation::build_s_and_saveexec_b64(temporary_exec, kRdna4VccLo, arch);
      if (!sequence.emit(narrow))
        return false;
      return sequence.emit_branch(next_window_label, InstructionSequence::BranchKind::SccZero);
    };
    const auto narrow_equal_literal = [&](uint32_t offset, uint32_t literal) -> bool {
      if (!record.load(offset, value))
        return false;
      const auto mov = instrumentation::build_v_mov_b32_literal(expected, literal, arch);
      const auto compare =
          instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(expected), value, arch);
      if (!mov || !compare)
        return false;
      words.insert(words.end(), mov->begin(), mov->end());
      words.push_back(*compare);
      return narrow_current();
    };
    const auto narrow_equal_vgpr = [&](uint32_t offset, uint16_t expected_vgpr) -> bool {
      if (!record.load(offset, value))
        return false;
      const auto compare =
          instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(expected_vgpr), value, arch);
      if (!compare)
        return false;
      words.push_back(*compare);
      return narrow_current();
    };
    const auto narrow_equal_dispatch_id = [&](uint32_t offset, bool high_word) -> bool {
      if (!record.load(offset, value) || !append_compare_moi_report_dispatch_id_word(
                                             words, options, value, expected, high_word, arch,
                                             ConSanMoiLiteralDispatchIdPolicy::AnyArchitecture)) {
        return false;
      }
      return narrow_current();
    };
    const auto narrow_equal_workgroup = [&](uint32_t offset,
                                            const ConSanMoiWorkgroupSource &source) -> bool {
      if (!source.scalar_src)
        return narrow_equal_literal(offset, 0u);
      words.push_back(build_v_mov_b32_e32(expected, *source.scalar_src, arch));
      if (source.mask_low_16) {
        const auto left = instrumentation::build_v_lshlrev_b32(
            expected, scalar_positive_inline_u32(16), expected, arch);
        const auto right = instrumentation::build_v_lshrrev_b32(
            expected, scalar_positive_inline_u32(16), expected, arch);
        if (!left || !right)
          return false;
        words.push_back(*left);
        words.push_back(*right);
      }
      if (source.shift_right_16) {
        const auto shift = instrumentation::build_v_lshrrev_b32(
            expected, scalar_positive_inline_u32(16), expected, arch);
        if (!shift)
          return false;
        words.push_back(*shift);
      }
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
        !narrow_equal_vgpr(offsetof(ConSanMoiSampledCausalWindow, epoch),
                           *options.moi_epoch_vgpr) ||
        !narrow_equal_vgpr(offsetof(ConSanMoiSampledCausalWindow, first_entry), bank_vgpr) ||
        !narrow_equal_literal(offsetof(ConSanMoiSampledCausalWindow, entry_count), 1u) ||
        !narrow_equal_literal(
            offsetof(ConSanMoiSampledCausalWindow, publication_state),
            static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Ready)) ||
        !narrow_equal_workgroup(offsetof(ConSanMoiSampledCausalWindow, cluster_workgroup_id),
                                workgroup_sources->cluster_workgroup_id)) {
      errors.emplace_back("ConSan MOI sampled barrier failed at causal-window validation");
      return std::nullopt;
    }
    if (!sequence.emit_branch(matched_window_label,
                              InstructionSequence::BranchKind::Unconditional) ||
        !sequence.bind(next_window_label))
      return std::nullopt;
  }
  if (!sequence.emit_branch(restore_label, InstructionSequence::BranchKind::Unconditional) ||
      !sequence.bind(matched_window_label))
    return std::nullopt;

  const auto narrow_current = [&]() -> bool {
    const auto narrow =
        instrumentation::build_s_and_saveexec_b64(temporary_exec, kRdna4VccLo, arch);
    if (!sequence.emit(narrow))
      return false;
    return sequence.emit_branch(restore_label, InstructionSequence::BranchKind::SccZero);
  };
  // A workgroup barrier resumes every wave. Elect the first logical owner
  // before the wave-local lane election below so exactly one wave can publish
  // the workgroup-wide transition. Without this narrowing, every wave races
  // the same metadata slot and the losing waves look like unsupported sync.
  const auto owner_zero = instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0),
                                                                  *options.moi_owner_vgpr, arch);
  if (!owner_zero) {
    errors.emplace_back("ConSan MOI sampled barrier failed at owner election");
    return std::nullopt;
  }
  words.push_back(*owner_zero);
  if (!narrow_current()) {
    errors.emplace_back("ConSan MOI sampled barrier failed at owner narrowing");
    return std::nullopt;
  }
  const auto max_epoch = instrumentation::build_v_mov_b32_literal(
      expected, consan_moi_sampled_watchpoint::max_epoch, arch);
  const auto epoch_has_successor = instrumentation::build_v_cmp_ne_u32_vcc(
      vector_source_vgpr(expected), *options.moi_epoch_vgpr, arch);
  if (!max_epoch || !epoch_has_successor) {
    errors.emplace_back("ConSan MOI sampled barrier failed at epoch successor check");
    return std::nullopt;
  }
  words.insert(words.end(), max_epoch->begin(), max_epoch->end());
  words.push_back(*epoch_has_successor);
  if (!narrow_current()) {
    errors.emplace_back("ConSan MOI sampled barrier failed at epoch narrowing");
    return std::nullopt;
  }

  const auto matching_exec = instrumentation::build_s_mov_b64(temporary_exec, kRdna4ExecLo, arch);
  const auto mbcnt_lo = instrumentation::build_v_mbcnt_lo_u32_b32(
      lane_rank, temporary_exec, scalar_positive_inline_u32(0), arch);
  const auto mbcnt_hi = instrumentation::build_v_mbcnt_hi_u32_b32(
      lane_rank, static_cast<uint16_t>(temporary_exec + 1u), vector_source_vgpr(lane_rank), arch);
  const auto first =
      instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), lane_rank, arch);
  const auto narrow_first =
      instrumentation::build_s_and_saveexec_b64(temporary_exec, kRdna4VccLo, arch);
  if (!matching_exec || !mbcnt_lo || !mbcnt_hi || !first || !narrow_first) {
    errors.emplace_back("ConSan MOI sampled barrier failed at lane election");
    return std::nullopt;
  }
  words.push_back(*matching_exec);
  words.insert(words.end(), mbcnt_lo->begin(), mbcnt_lo->end());
  words.insert(words.end(), mbcnt_hi->begin(), mbcnt_hi->end());
  words.push_back(*first);
  words.push_back(*narrow_first);

  const auto sentinel = instrumentation::build_v_mov_b32_literal(
      value, kConSanMoiSampledSyncPublishingDescriptor, arch);
  const auto zero = instrumentation::build_v_mov_b32_literal(expected, 0u, arch);
  const auto claim = instrumentation::build_flat_atomic_cmpswap_b32(base, value, value, true,
                                                                    kRdna4ScopeDevice, arch);
  if (!sentinel || !zero || !claim ||
      !append_sampled_indexed_address(
          words, first_metadata_address + offsetof(ConSanMoiSampledSyncMetadataPacked, descriptor),
          sizeof(ConSanMoiSampledSyncMetadataPacked), bank_vgpr, base, next_epoch, arch)) {
    errors.emplace_back("ConSan MOI sampled barrier failed at metadata claim setup");
    return std::nullopt;
  }
  words.insert(words.end(), sentinel->begin(), sentinel->end());
  words.insert(words.end(), zero->begin(), zero->end());
  words.insert(words.end(), claim->begin(), claim->end());
  if (!append_moi_flat_load_wait(words, arch))
    return std::nullopt;
  const auto claimed =
      instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), value, arch);
  const auto narrow_claimed =
      instrumentation::build_s_and_saveexec_b64(temporary_exec, kRdna4VccLo, arch);
  if (!claimed || !narrow_claimed) {
    errors.emplace_back("ConSan MOI sampled barrier failed at metadata claim result");
    return std::nullopt;
  }
  words.push_back(*claimed);
  words.push_back(*narrow_claimed);
  if (!sequence.emit_branch(collision_label, InstructionSequence::BranchKind::SccZero))
    return std::nullopt;

  const auto increment = instrumentation::build_v_add_u32(next_epoch, scalar_positive_inline_u32(1),
                                                          *options.moi_epoch_vgpr, arch);
  if (!increment) {
    errors.emplace_back("ConSan MOI sampled barrier failed at next-epoch materialization");
    return std::nullopt;
  }
  words.insert(words.end(), increment->begin(), increment->end());
  if (!append_sampled_indexed_address(words, first_metadata_address,
                                      sizeof(ConSanMoiSampledSyncMetadataPacked), bank_vgpr, base,
                                      value, arch)) {
    errors.emplace_back("ConSan MOI sampled barrier failed at metadata addressing");
    return std::nullopt;
  }
  if (!record.store_literal(0u, 0u) || !record.store_literal(4u, 0u) ||
      !record.store_literal(offsetof(ConSanMoiSampledSyncMetadataPacked, byte_count), 0u) ||
      !record.store_vgpr(offsetof(ConSanMoiSampledSyncMetadataPacked, epoch_before),
                         *options.moi_epoch_vgpr) ||
      !record.store_vgpr(offsetof(ConSanMoiSampledSyncMetadataPacked, epoch_after), next_epoch)) {
    errors.emplace_back("ConSan MOI sampled barrier failed at metadata body stores");
    return std::nullopt;
  }
  const auto wait_store = instrumentation::build_s_wait_flat_store0(arch);
  const auto final_value =
      instrumentation::build_v_mov_b32_literal(value, encoded.packed.descriptor, arch);
  const auto final_expected = instrumentation::build_v_mov_b32_literal(
      expected, kConSanMoiSampledSyncPublishingDescriptor, arch);
  const auto commit = instrumentation::build_flat_atomic_cmpswap_b32(base, value, value, true,
                                                                     kRdna4ScopeDevice, arch);
  if (!wait_store || !final_value || !final_expected || !commit) {
    errors.emplace_back("ConSan MOI sampled barrier failed at metadata commit setup");
    return std::nullopt;
  }
  words.push_back(*wait_store);
  words.insert(words.end(), final_value->begin(), final_value->end());
  words.insert(words.end(), final_expected->begin(), final_expected->end());
  if (!append_sampled_indexed_address(
          words, first_metadata_address + offsetof(ConSanMoiSampledSyncMetadataPacked, descriptor),
          sizeof(ConSanMoiSampledSyncMetadataPacked), bank_vgpr, base, next_epoch, arch)) {
    errors.emplace_back("ConSan MOI sampled barrier failed at metadata commit addressing");
    return std::nullopt;
  }
  words.insert(words.end(), commit->begin(), commit->end());
  if (!append_moi_flat_load_wait(words, arch))
    return std::nullopt;
  const auto committed =
      instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(expected), value, arch);
  const auto narrow_committed =
      instrumentation::build_s_and_saveexec_b64(temporary_exec, kRdna4VccLo, arch);
  if (!committed || !narrow_committed) {
    errors.emplace_back("ConSan MOI sampled barrier failed at metadata commit result");
    return std::nullopt;
  }
  words.push_back(*committed);
  words.push_back(*narrow_committed);
  if (!sequence.emit_branch(collision_label, InstructionSequence::BranchKind::SccZero))
    return std::nullopt;
  if (!append_atomic_fetch_add_one_u32(
          words, report_base + offsetof(ConSanMoiReportHeader, sampled_sync_metadata_count), value,
          base, arch)) {
    errors.emplace_back("ConSan MOI sampled barrier failed at success accounting");
    return std::nullopt;
  }
  if (!sequence.emit_branch(restore_label, InstructionSequence::BranchKind::Unconditional))
    return std::nullopt;

  if (!sequence.bind(collision_label))
    return std::nullopt;
  const auto restore_collision_exec =
      instrumentation::build_s_mov_b64(kRdna4ExecLo, original_exec, arch);
  const auto collision_mask = instrumentation::build_s_mov_b64(temporary_exec, original_exec, arch);
  const auto collision_lo = instrumentation::build_v_mbcnt_lo_u32_b32(
      lane_rank, temporary_exec, scalar_positive_inline_u32(0), arch);
  const auto collision_hi = instrumentation::build_v_mbcnt_hi_u32_b32(
      lane_rank, static_cast<uint16_t>(temporary_exec + 1u), vector_source_vgpr(lane_rank), arch);
  const auto collision_first =
      instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), lane_rank, arch);
  const auto narrow_collision =
      instrumentation::build_s_and_saveexec_b64(temporary_exec, kRdna4VccLo, arch);
  if (!restore_collision_exec || !collision_mask || !collision_lo || !collision_hi ||
      !collision_first || !narrow_collision) {
    errors.emplace_back("ConSan MOI sampled barrier failed at collision lane election");
    return std::nullopt;
  }
  words.push_back(*restore_collision_exec);
  words.push_back(*collision_mask);
  words.insert(words.end(), collision_lo->begin(), collision_lo->end());
  words.insert(words.end(), collision_hi->begin(), collision_hi->end());
  words.push_back(*collision_first);
  words.push_back(*narrow_collision);
  if (!append_atomic_fetch_add_one_u32(
          words, report_base + offsetof(ConSanMoiReportHeader, sampled_unsupported_sync_count),
          value, base, arch)) {
    errors.emplace_back("ConSan MOI sampled barrier failed at collision accounting");
    return std::nullopt;
  }

  if (!sequence.bind(restore_label))
    return std::nullopt;
  const auto restore_exec = instrumentation::build_s_mov_b64(kRdna4ExecLo, original_exec, arch);
  const auto restore_after_advance =
      instrumentation::build_s_mov_b64(kRdna4ExecLo, original_exec, arch);
  if (!restore_exec || !restore_after_advance) {
    errors.emplace_back("ConSan MOI sampled barrier failed at EXEC restoration");
    return std::nullopt;
  }
  words.push_back(*restore_exec);
  words.insert(words.end(), max_epoch->begin(), max_epoch->end());
  words.push_back(*epoch_has_successor);
  const auto narrow_advance =
      instrumentation::build_s_and_saveexec_b64(temporary_exec, kRdna4VccLo, arch);
  if (!narrow_advance) {
    errors.emplace_back("ConSan MOI sampled barrier failed at epoch-advance narrowing");
    return std::nullopt;
  }
  words.push_back(*narrow_advance);
  const auto advance_epoch = instrumentation::build_v_add_u32(
      *options.moi_epoch_vgpr, scalar_positive_inline_u32(1), *options.moi_epoch_vgpr, arch);
  if (!advance_epoch) {
    errors.emplace_back("ConSan MOI sampled barrier failed at epoch advance");
    return std::nullopt;
  }
  words.insert(words.end(), advance_epoch->begin(), advance_epoch->end());
  words.push_back(*restore_after_advance);
  if (scalar_persistent) {
    const auto persist_epoch = instrumentation::build_v_readfirstlane_b32(
        *options.moi_persistent_epoch_sgpr, *options.moi_epoch_vgpr, arch);
    const auto delay = instrumentation::build_salu_dependency_delay(arch);
    if (!persist_epoch || !delay) {
      errors.emplace_back("ConSan MOI sampled barrier failed at scalar epoch persistence");
      return std::nullopt;
    }
    words.push_back(*persist_epoch);
    words.push_back(*delay);
  }
  if (private_layout) {
    const auto store = instrumentation::build_private_store_b32(*options.moi_epoch_vgpr,
                                                                private_layout->epoch_offset, arch);
    const auto wait = instrumentation::build_s_wait_private_store0(arch);
    if (!store || !wait) {
      errors.emplace_back("ConSan MOI sampled barrier failed at private epoch store");
      return std::nullopt;
    }
    words.insert(words.end(), store->begin(), store->end());
    words.push_back(*wait);
  }
  if (!append_restore_moi_special_state(words, options, arch)) {
    errors.emplace_back("ConSan MOI sampled barrier failed at special-state restore");
    return std::nullopt;
  }
  if (!sequence.bind(fast_return_label))
    return std::nullopt;
  if (scalar_spill)
    words.insert(words.end(), scalar_spill->restore_words.begin(),
                 scalar_spill->restore_words.end());
  if (spill)
    words.insert(words.end(), spill->restore_words.begin(), spill->restore_words.end());
  if (restore_guest_vgpr_bank)
    words.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
        0u, static_cast<uint8_t>(*gfx1250_vgpr_msb_mode), arch));
  const uint64_t branch_pc = cave_text_offset + words.size() * sizeof(uint32_t);
  const auto ret = compute_sopp_branch_simm16(branch_pc, return_text_offset);
  if (ret) {
    words.push_back(build_s_branch(*ret, arch));
  } else {
    const auto jump_sgprs = moi_indirect_jump_sgprs(options);
    if (!jump_sgprs || !append_moi_scc_preserving_indirect_jump(
                           words, cave_text_offset, return_text_offset, jump_sgprs->pc_sgpr,
                           jump_sgprs->scc_save_sgpr, /*capture_scc=*/true, arch)) {
      errors.emplace_back("ConSan MOI sampled barrier could not encode its long return");
      return std::nullopt;
    }
  }

  if (!sequence.resolve_branches(arch)) {
    errors.emplace_back("ConSan MOI sampled barrier local branch is out of range");
    return std::nullopt;
  }
  return words;
}

void try_apply_sampled_barrier_sync_patch(std::span<const uint8_t> bytes,
                                          const ConSanOptions &options, rj_code_arch_t arch,
                                          ConSanResult &result) {
  if (!options.moi_track_barriers || options.moi_engine != ConSanMoiEngine::Sampled)
    return;
  if (!options.moi_report_buffer_address || !instrumentation::is_admitted_arch(arch)) {
    result.warnings.emplace_back(
        "ConSan MOI sampled barrier sync requires a supported report-buffer target");
    return;
  }
  const ConSanMoiReportBufferLayout layout = resolve_sampled_report_layout(options);
  struct SelectedWindow {
    const ConSanPatchInfo *patch = nullptr;
    const ConSanMoiCandidate *access = nullptr;
    uint32_t first_slot = 0;
    uint32_t bank_count = 1;
  };
  std::vector<SelectedWindow> selected_windows;
  selected_windows.reserve(result.patches.size());
  std::unordered_map<uint64_t, const ConSanMoiCandidate *> access_by_offset;
  access_by_offset.reserve(result.moi_candidates.size());
  for (const ConSanMoiCandidate &candidate : result.moi_candidates) {
    if (candidate.kind == ConSanLdsAccessKind::Read || candidate.kind == ConSanLdsAccessKind::Write)
      access_by_offset.try_emplace(candidate.text_offset, &candidate);
  }
  bool has_multi_range_window = false;
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.kind != ConSanPatchKind::InlineMoiSampledWatchpointStore &&
        patch.kind != ConSanPatchKind::TrampolineMoiSampledWatchpointStore)
      continue;
    const auto access = access_by_offset.find(patch.anchor_offset);
    if (access == access_by_offset.end())
      continue;
    const auto ranges = candidate_access_ranges(bytes, *access->second, arch);
    if (!ranges || ranges->empty())
      continue;
    selected_windows.push_back({&patch, access->second, patch.sampled_first_slot,
                                std::max(patch.sampled_window_bank_count, 1u)});
    has_multi_range_window |= ranges->size() > 1u;
  }
  if (selected_windows.empty() || layout.sampled_sync_metadata_capacity == 0) {
    result.warnings.emplace_back(
        "ConSan MOI sampled barrier sync found no selected Ready causal-window capacity");
    return;
  }
  if (has_multi_range_window) {
    result.warnings.emplace_back(
        "ConSan MOI sampled barrier sync currently attaches only the first logical range of a "
        "multi-range access");
  }
  MoiResourcePlanningState resource_state(bytes, arch, options, result);
  if (!resource_state.valid) {
    result.warnings.emplace_back("ConSan MOI sampled barrier sync lacks a complete kernel CFG");
    return;
  }
  std::unordered_map<uint64_t, const MoiKernelResourceContext *> context_by_descriptor;
  context_by_descriptor.reserve(resource_state.contexts.size());
  for (const MoiKernelResourceContext &context : resource_state.contexts) {
    if (context.kernel != nullptr)
      context_by_descriptor.try_emplace(context.kernel->descriptor_file_offset, &context);
  }
  std::unordered_map<uint64_t, std::vector<const SelectedWindow *>> windows_by_descriptor;
  for (const SelectedWindow &window : selected_windows) {
    if (!window.patch->owner_descriptor_file_offsets.empty()) {
      for (uint64_t descriptor : window.patch->owner_descriptor_file_offsets)
        windows_by_descriptor[descriptor].push_back(&window);
    } else if (window.access->kernel_descriptor_file_offset) {
      windows_by_descriptor[*window.access->kernel_descriptor_file_offset].push_back(&window);
    }
  }

  std::unordered_map<std::string_view, const ConSanKernelInfo *> kernel_by_name;
  kernel_by_name.reserve(result.kernels.size());
  for (const ConSanKernelInfo &kernel : result.kernels)
    kernel_by_name.try_emplace(kernel.name, &kernel);
  std::unordered_map<std::string_view, const ConSanFunctionInfo *> function_by_name;
  function_by_name.reserve(result.functions.size());
  for (const ConSanFunctionInfo &function : result.functions)
    function_by_name.try_emplace(function.name, &function);

  std::vector<SampledBarrierCandidate> candidates;
  for (const ConSanSyncSequence &sequence : result.sync_sequences) {
    if (!consan_moi_sampled_qualifies_barrier_sequence(sequence))
      continue;
    const std::vector<ConSanBarrierSite> *barrier_sites = nullptr;
    uint64_t container_entry_text_offset = 0;
    uint64_t container_code_size = 0;
    uint64_t container_text_file_offset = 0;
    if (sequence.in_kernel) {
      const auto found = kernel_by_name.find(sequence.container_name);
      if (found == kernel_by_name.end())
        continue;
      barrier_sites = &found->second->barrier_sites;
      container_entry_text_offset = found->second->entry_text_offset;
      container_code_size = found->second->code_size;
      container_text_file_offset = found->second->text_file_offset;
    } else {
      const auto found = function_by_name.find(sequence.container_name);
      if (found == function_by_name.end())
        continue;
      barrier_sites = &found->second->barrier_sites;
      container_entry_text_offset = found->second->entry_text_offset;
      container_code_size = found->second->code_size;
      container_text_file_offset = found->second->text_file_offset;
    }
    std::vector<ConSanBarrierSite> completing_sites;
    std::ranges::copy_if(*barrier_sites, std::back_inserter(completing_sites),
                         [&](const ConSanBarrierSite &site) {
                           return site.text_offset + site.size == sequence.end_text_offset;
                         });
    if (completing_sites.size() != 1u) {
      result.warnings.emplace_back("ConSan MOI sampled barrier sync rejected ambiguous completing "
                                   "site in " +
                                   sequence.container_name);
      continue;
    }
    if (sequence.member_event_identities.size() >
        std::numeric_limits<uint32_t>::max() - result.sampled_barrier_applicable_event_count) {
      result.errors.emplace_back("ConSan MOI sampled barrier applicable event count overflow");
      return;
    }
    result.sampled_barrier_applicable_event_count +=
        static_cast<uint32_t>(sequence.member_event_identities.size());
    std::vector<SampledBarrierWindow> windows;
    bool every_owner_has_window = true;
    std::string owner_lowering_failure;
    bool uses_cluster_workgroup_id = false;
    bool first_owner = true;
    std::optional<ConSanMoiWorkgroupSources> common_workgroup_sources;
    std::optional<uint16_t> common_owner_shift;
    for (const ConSanExecutionOwner &owner : sequence.execution_owners) {
      const auto context = context_by_descriptor.find(owner.descriptor_file_offset);
      if (context == context_by_descriptor.end() || context->second->kernel == nullptr) {
        owner_lowering_failure = "missing kernel resource context";
        break;
      }
      if (!first_owner &&
          uses_cluster_workgroup_id != context->second->kernel->uses_gfx1250_cluster_workgroup_id) {
        owner_lowering_failure = "incompatible cluster workgroup ABI";
        break;
      }
      uses_cluster_workgroup_id = context->second->kernel->uses_gfx1250_cluster_workgroup_id;
      first_owner = false;
      const auto workgroup_sources = moi_descriptor_workgroup_sources(
          bytes, owner.descriptor_file_offset, arch, result.errors, uses_cluster_workgroup_id);
      const auto owner_shift =
          options.automatic_moi_private_epoch
              ? moi_descriptor_owner_shift(bytes, owner.descriptor_file_offset, result.errors)
              : std::optional<uint16_t>(0u);
      if (!workgroup_sources || !owner_shift ||
          (common_workgroup_sources && *common_workgroup_sources != *workgroup_sources) ||
          (common_owner_shift && *common_owner_shift != *owner_shift)) {
        owner_lowering_failure = "incompatible descriptor ABI inputs";
        break;
      }
      common_workgroup_sources = *workgroup_sources;
      common_owner_shift = *owner_shift;
      const auto distance_index = build_sampled_cfg_forward_distance_index(
          resource_state, *context->second, sequence.begin_text_offset);
      if (!distance_index) {
        owner_lowering_failure = "missing owner CFG distance proof";
        break;
      }
      const auto owner_windows = windows_by_descriptor.find(owner.descriptor_file_offset);
      if (owner_windows == windows_by_descriptor.end()) {
        every_owner_has_window = false;
        break;
      }
      std::optional<SampledBarrierWindow> selected_window;
      std::optional<SampledCfgDistance> closest;
      for (const SelectedWindow *window : owner_windows->second) {
        const ConSanPatchInfo &patch = *window->patch;
        if (!window->access || !kernel_owns_patch(*context->second->kernel, patch))
          continue;
        const auto distance =
            sampled_cfg_forward_distance(resource_state, *distance_index, patch.anchor_offset);
        if (!distance || (closest && *closest <= *distance))
          continue;
        closest = *distance;
        selected_window = SampledBarrierWindow{window->first_slot, window->bank_count};
      }
      if (!selected_window) {
        every_owner_has_window = false;
        break;
      }
      windows.push_back(*selected_window);
    }
    std::ranges::sort(windows);
    windows.erase(std::ranges::unique(windows).begin(), windows.end());
    if (!owner_lowering_failure.empty()) {
      // The sequence is semantically applicable and must remain in the strict
      // coverage denominator. Leaving its Supported disposition unpatched
      // records a durable placement/lowering failure instead of hiding a
      // descriptor or CFG incompatibility as a missing causal window.
      result.warnings.emplace_back("ConSan MOI sampled barrier sync cannot lower every owner in " +
                                   sequence.container_name + ": " + owner_lowering_failure);
      continue;
    }
    if (!every_owner_has_window || windows.empty()) {
      result.warnings.emplace_back(
          "ConSan MOI sampled barrier sync has no preceding selected causal window for every "
          "owner in " +
          sequence.container_name);
      if (!sequence.in_kernel)
        continue;
      for (const std::string &identity : sequence.member_event_identities) {
        const auto event =
            std::ranges::find(result.sync_events, identity, &ConSanSyncEvent::identity);
        if (event == result.sync_events.end())
          continue;
        const auto disposition = std::ranges::find_if(
            result.site_dispositions, [&](const ConSanSiteDispositionRecord &site) {
              return site.site_kind == ConSanResourceSiteKind::Barrier &&
                     site.container_name == event->container_name &&
                     site.in_kernel == event->in_kernel && site.text_offset == event->text_offset;
            });
        if (disposition == result.site_dispositions.end() ||
            disposition->disposition != ConSanSiteDisposition::Supported)
          continue;
        // No memory-access window exists before this synchronization point,
        // so there is no causal state for Sampled to publish. This is a typed
        // non-applicable site, not lost coverage.
        disposition->disposition = ConSanSiteDisposition::NotApplicable;
        disposition->reason = ConSanSiteDispositionReason::NoPrecedingSampledWindow;
        --result.sampled_barrier_applicable_event_count;
      }
      continue;
    }
    const uint64_t representative_descriptor =
        sequence.execution_owners.front().descriptor_file_offset;
    candidates.push_back(
        {&sequence,
         {(sequence.in_kernel ? "kernel:" : "function:") + sequence.container_name,
          sequence.in_kernel, completing_sites.front(), representative_descriptor,
          container_entry_text_offset, container_code_size, container_text_file_offset},
         std::move(windows),
         uses_cluster_workgroup_id});
  }
  if (candidates.empty()) {
    result.warnings.emplace_back(
        "ConSan MOI sampled barrier sync found no qualified complete barrier sequence");
    return;
  }
  std::ranges::sort(candidates, {}, [](const SampledBarrierCandidate &candidate) {
    return candidate.barrier.site.text_offset;
  });
  std::vector<SampledBarrierCandidate> unambiguous;
  for (size_t begin = 0; begin < candidates.size();) {
    size_t end = begin + 1u;
    while (end < candidates.size() &&
           candidates[end].barrier.site.text_offset == candidates[begin].barrier.site.text_offset)
      ++end;
    if (end == begin + 1u) {
      unambiguous.push_back(std::move(candidates[begin]));
    } else {
      result.warnings.emplace_back(
          "ConSan MOI sampled barrier sync rejected ambiguous sequence ownership at offset " +
          std::to_string(candidates[begin].barrier.site.text_offset));
    }
    begin = end;
  }
  candidates = std::move(unambiguous);
  if (candidates.size() > options.max_patches)
    candidates.resize(options.max_patches);

  std::span<const uint8_t> active_bytes = bytes;
  if (result.modified)
    active_bytes = {result.elf_bytes.data(), result.elf_bytes.size()};
  AmdGpuCodeObject code_object(active_bytes.data(), active_bytes.size());
  CodeObjectPatcher patcher(code_object);
  const std::span<const uint8_t> old_text = patcher.text_bytes();
  if (old_text.empty()) {
    result.errors.emplace_back("ConSan MOI sampled barrier sync found no .text section");
    return;
  }

  struct Planned {
    SampledBarrierCandidate candidate;
    ResolvedMoiScratchPlan resources;
    std::optional<VgprSpillSequence> spill;
    std::optional<InlineSgprSpillSequence> scalar_spill;
    std::optional<MoiPrivateEpochLayout> private_layout;
    uint32_t required_private_bytes = 0;
  };
  std::vector<Planned> selected;
  MoiSpillManagers spill_managers;
  MoiDescriptorVgprRequirements descriptor_requirements;
  MoiDescriptorPrivateRequirements private_requirements;
  std::vector<std::pair<uint64_t, uint64_t>> patch_claims;
  patch_claims.reserve(result.patches.size());
  for (const ConSanPatchInfo &patch : result.patches) {
    const uint64_t size = std::max<uint32_t>(patch.original_size, 1u);
    patch_claims.emplace_back(patch.anchor_offset, patch.anchor_offset + size);
  }
  std::ranges::sort(patch_claims);
  size_t merged_patch_claim_count = 0u;
  for (const auto &claim : patch_claims) {
    if (merged_patch_claim_count != 0u &&
        patch_claims[merged_patch_claim_count - 1u].second >= claim.first) {
      patch_claims[merged_patch_claim_count - 1u].second =
          std::max(patch_claims[merged_patch_claim_count - 1u].second, claim.second);
    } else {
      patch_claims[merged_patch_claim_count++] = claim;
    }
  }
  patch_claims.resize(merged_patch_claim_count);
  const auto overlaps_patch = [&](uint64_t begin, uint64_t end) {
    const auto after = std::ranges::lower_bound(patch_claims, end, {},
                                                [](const auto &range) { return range.first; });
    return after != patch_claims.begin() && std::prev(after)->second > begin;
  };
  std::unordered_map<uint64_t, size_t> barrier_plan_by_offset;
  barrier_plan_by_offset.reserve(result.resource_plans.size());
  for (size_t index = 0; index < result.resource_plans.size(); ++index) {
    const ConSanCandidateResourcePlan &plan = result.resource_plans[index];
    if (plan.site_kind == ConSanResourceSiteKind::Barrier)
      barrier_plan_by_offset.try_emplace(plan.text_offset, index);
  }
  for (size_t index = 0; index < candidates.size(); ++index) {
    SampledBarrierCandidate candidate = candidates[index];
    const ConSanBarrierSite &site = candidate.barrier.site;
    if (site.size == 0 || site.size % sizeof(uint32_t) != 0 || site.text_offset > old_text.size() ||
        site.size > old_text.size() - site.text_offset)
      continue;
    if (overlaps_patch(site.text_offset, site.text_offset + site.size))
      continue;
    const auto existing = barrier_plan_by_offset.find(site.text_offset);
    ConSanCandidateResourcePlan plan;
    if (existing != barrier_plan_by_offset.end()) {
      plan = result.resource_plans[existing->second];
    } else {
      plan =
          plan_moi_resource_site(resource_state, options, ConSanResourceSiteKind::Barrier, index,
                                 site.text_offset, candidate.barrier.kernel_descriptor_file_offset,
                                 options.automatic_moi_private_epoch ? 9u : 7u);
      result.resource_plans.push_back(plan);
      barrier_plan_by_offset.emplace(site.text_offset, result.resource_plans.size() - 1u);
    }
    const uint16_t scratch_count =
        options.automatic_moi_private_epoch || moi_uses_scalar_persistent_state(options) ? 9u : 7u;
    const auto resources = resolve_moi_scratch_plan(plan, options, scratch_count);
    if (!resources)
      continue;
    ConSanOptions candidate_options = options;
    if (!apply_moi_transient_sgpr_assignment(candidate_options,
                                             resources->owner_descriptor_file_offsets)) {
      result.warnings.emplace_back(
          "ConSan MOI sampled barrier lost its owner-local transient scalar assignment");
      continue;
    }
    candidate_options.scratch_vgpr = resources->base;
    std::optional<MoiPrivateEpochLayout> private_layout;
    if (options.automatic_moi_private_epoch) {
      private_layout =
          build_sampled_private_epoch_layout(result, *resources, options, arch, result.warnings);
      if (!private_layout)
        continue;
    }
    std::optional<VgprSpillSequence> spill;
    if (resources->source == ConSanRegisterAllocationSource::SpillRequired ||
        moi_scalar_spill_requires_dynamic_vgpr_frame(result, *resources, candidate_options)) {
      spill = build_moi_spill_sequence(
          result, *resources, candidate_options, spill_managers, arch, result.warnings,
          private_layout ? std::optional<uint32_t>(private_layout->ephemeral_base) : std::nullopt);
      if (!spill)
        continue;
    }
    std::optional<InlineSgprSpillSequence> scalar_spill;
    if (candidate_options.automatic_moi_record_replay_sgpr_spill) {
      scalar_spill = build_inline_sgpr_spill_sequence(
          result, *resources, candidate_options, spill_managers, arch, result.warnings,
          private_layout ? std::optional<uint32_t>(private_layout->ephemeral_base) : std::nullopt,
          spill ? &*spill : nullptr);
      if (!scalar_spill)
        continue;
    }
    const uint32_t required_private_bytes =
        std::max({private_layout ? private_layout->ephemeral_base : 0u,
                  spill ? spill->total_private_bytes : 0u,
                  scalar_spill ? scalar_spill->total_private_bytes : 0u});
    if (required_private_bytes != 0u) {
      for (uint64_t descriptor_offset : resources->owner_descriptor_file_offsets) {
        auto [it, inserted] =
            private_requirements.emplace(descriptor_offset, required_private_bytes);
        if (!inserted)
          it->second = std::max(it->second, required_private_bytes);
      }
    }
    note_descriptor_requirements(descriptor_requirements, *resources);
    selected.push_back({std::move(candidate), *resources, std::move(spill), std::move(scalar_spill),
                        std::move(private_layout), required_private_bytes});
  }
  if (selected.empty()) {
    result.warnings.emplace_back("ConSan MOI sampled barrier sync found no patchable sequence");
    return;
  }
  if (!apply_descriptor_requirements(patcher, code_object, active_bytes, result,
                                     descriptor_requirements, arch, result.errors) ||
      !apply_spill_descriptor_requirements(patcher, code_object, active_bytes, result,
                                           private_requirements, result.errors))
    return;

  std::vector<uint8_t> new_text(old_text.begin(), old_text.end());
  AmdGpuCodeObject original_code_object(bytes.data(), bytes.size());
  const uint64_t original_text_size = original_code_object.text_sections().size() == 1
                                          ? original_code_object.text_sections().front()->size()
                                          : 0u;
  constexpr uint64_t kIndirectIslandWords = 7u;
  std::vector<ConSanPatchInfo> patches;
  const bool use_dense_barrier_relay = arch == ROCJITSU_CODE_ARCH_GFX1250 ||
                                       (arch == ROCJITSU_CODE_ARCH_RDNA4 && selected.size() >= 8u);
  if (use_dense_barrier_relay) {
    const auto special_state = moi_special_state_sgprs(options);
    if (!options.moi_exec_save_sgpr || !special_state) {
      result.errors.emplace_back(
          "ConSan MOI sampled dense barrier relay requires scalar special-state storage");
      return;
    }
    const uint16_t base = *options.moi_exec_save_sgpr;
    const uint16_t key_sgpr = static_cast<uint16_t>(base + 5u);
    const uint16_t call_return_sgpr = static_cast<uint16_t>(base + 6u);
    const auto build_direct_call = [&](int16_t simm16) -> std::optional<uint32_t> {
      if (arch == ROCJITSU_CODE_ARCH_GFX1250)
        return instrumentation::build_s_call_i64(call_return_sgpr, simm16, arch);
      if (arch == ROCJITSU_CODE_ARCH_RDNA4)
        return build_s_call_b64(call_return_sgpr, simm16, arch);
      return std::nullopt;
    };
    constexpr uint64_t kHostWords = kIndirectIslandWords + 1u;
    constexpr uint64_t kHostBytes = kHostWords * sizeof(uint32_t);
    constexpr uint64_t kDispatcherWordsPerSite = 20u;

    std::vector<uint64_t> leaders;
    std::vector<BasicBlock::CodeRange> code_ranges;
    for (const ConSanKernelInfo &kernel : result.kernels) {
      if (!kernel.has_text_range)
        continue;
      leaders.push_back(kernel.entry_text_offset);
      if (kernel.code_size != 0u)
        code_ranges.push_back({.start_offset = kernel.entry_text_offset, .size = kernel.code_size});
    }
    for (const ConSanFunctionInfo &function : result.functions) {
      leaders.push_back(function.entry_text_offset);
      if (function.code_size != 0u)
        code_ranges.push_back(
            {.start_offset = function.entry_text_offset, .size = function.code_size});
    }
    std::ranges::sort(leaders);
    leaders.erase(std::ranges::unique(leaders).begin(), leaders.end());
    std::unique_ptr<Decoder> decoder = Decoder::create(arch);
    if (!decoder) {
      result.errors.emplace_back(
          "ConSan MOI sampled dense barrier relay could not create a decoder");
      return;
    }
    const auto blocks = BasicBlock::build(code_object, *decoder, arch, leaders, code_ranges);
    using ContainerKey = std::pair<bool, std::string>;
    std::map<ContainerKey, std::vector<size_t>> owner_groups;
    for (size_t index = 0; index < selected.size(); ++index) {
      const SampledBarrierCandidate &candidate = selected[index].candidate;
      owner_groups[{candidate.sequence->in_kernel, candidate.sequence->container_name}].push_back(
          index);
    }
    constexpr uint64_t kMaxCommonRelayAnchorSpan =
        static_cast<uint64_t>(static_cast<int64_t>(std::numeric_limits<int16_t>::max()) -
                              static_cast<int64_t>(std::numeric_limits<int16_t>::min())) *
        sizeof(uint32_t);
    std::vector<std::pair<ContainerKey, std::vector<size_t>>> groups;
    for (auto &[key, owner_indices] : owner_groups) {
      std::ranges::sort(owner_indices, [&](size_t lhs, size_t rhs) {
        return selected[lhs].candidate.barrier.site.text_offset <
               selected[rhs].candidate.barrier.site.text_offset;
      });
      for (size_t index : owner_indices) {
        const uint64_t anchor = selected[index].candidate.barrier.site.text_offset;
        if (groups.empty() || groups.back().first != key ||
            anchor - selected[groups.back().second.front()].candidate.barrier.site.text_offset >
                kMaxCommonRelayAnchorSpan) {
          groups.push_back({key, {}});
        }
        groups.back().second.push_back(index);
      }
    }
    std::vector<std::pair<uint64_t, uint64_t>> claimed_hosts;
    std::vector<std::pair<uint64_t, uint64_t>> fixed_claims;
    fixed_claims.reserve(result.patches.size() + options.preapplied_reserved_ranges.size() +
                         selected.size());
    for (const ConSanPatchInfo &patch : result.patches) {
      if (patch.original_size != 0u)
        fixed_claims.emplace_back(patch.anchor_offset, patch.anchor_offset + patch.original_size);
    }
    for (const auto &reserved : options.preapplied_reserved_ranges) {
      if (reserved.size != 0u)
        fixed_claims.emplace_back(reserved.text_offset, reserved.text_offset + reserved.size);
    }
    for (const Planned &planned : selected) {
      const ConSanBarrierSite &site = planned.candidate.barrier.site;
      if (site.size != 0u)
        fixed_claims.emplace_back(site.text_offset, site.text_offset + site.size);
    }
    std::ranges::sort(fixed_claims);
    size_t merged_claim_count = 0u;
    for (const auto &claim : fixed_claims) {
      if (merged_claim_count != 0u && fixed_claims[merged_claim_count - 1u].second >= claim.first) {
        fixed_claims[merged_claim_count - 1u].second =
            std::max(fixed_claims[merged_claim_count - 1u].second, claim.second);
      } else {
        fixed_claims[merged_claim_count++] = claim;
      }
    }
    fixed_claims.resize(merged_claim_count);
    const auto overlaps_fixed_claim = [&](uint64_t begin, uint64_t end) {
      const auto found = std::ranges::lower_bound(fixed_claims, begin, {},
                                                  [](const auto &range) { return range.second; });
      return found != fixed_claims.end() && found->first < end;
    };
    const auto overlaps_existing = [&](uint64_t begin, uint64_t end) {
      if (overlaps_fixed_claim(begin, end) ||
          std::ranges::any_of(claimed_hosts, [&](const auto &range) {
            return range.first < end && range.second > begin;
          }))
        return true;
      return false;
    };

    for (const auto &[container, indices] : groups) {
      if (indices.empty())
        continue;
      uint64_t container_begin = 0u;
      uint64_t container_end = 0u;
      std::string_view container_name = container.second;
      if (container.first) {
        const auto kernel =
            std::ranges::find(result.kernels, container.second, &ConSanKernelInfo::name);
        if (kernel == result.kernels.end())
          continue;
        container_begin = kernel->entry_text_offset;
        container_end = container_begin + kernel->code_size;
      } else {
        const auto function =
            std::ranges::find(result.functions, container.second, &ConSanFunctionInfo::name);
        if (function == result.functions.end())
          continue;
        container_begin = function->entry_text_offset;
        container_end = container_begin + function->code_size;
      }
      std::vector<uint64_t> group_owners;
      for (size_t index : indices) {
        const auto &owners = selected[index].resources.owner_descriptor_file_offsets;
        group_owners.insert(group_owners.end(), owners.begin(), owners.end());
      }
      std::ranges::sort(group_owners);
      group_owners.erase(std::ranges::unique(group_owners).begin(), group_owners.end());
      std::vector<uint64_t> anchors;
      anchors.reserve(indices.size());
      for (size_t index : indices)
        anchors.push_back(selected[index].candidate.barrier.site.text_offset);
      std::optional<uint64_t> host_offset;
      std::vector<uint32_t> displaced_words;
      uint64_t best_max_distance = std::numeric_limits<uint64_t>::max();
      auto block =
          std::lower_bound(blocks.begin(), blocks.end(), container_begin,
                           [](const std::unique_ptr<BasicBlock> &candidate, uint64_t entry) {
                             return candidate->end_offset() <= entry;
                           });
      for (; block != blocks.end() && (*block)->start_offset() < container_end; ++block) {
        if ((*block)->start_offset() < container_begin || (*block)->end_offset() > container_end ||
            (*block)->size() < kHostBytes)
          continue;
        for (const Instruction &instruction : (*block)->instructions()) {
          const uint64_t candidate_host = instruction.src_loc();
          if (candidate_host + kHostBytes > (*block)->end_offset() ||
              overlaps_existing(candidate_host, candidate_host + kHostBytes))
            continue;
          const uint64_t relay_offset = candidate_host + sizeof(uint32_t);
          uint64_t max_distance = 0u;
          const bool reachable = std::ranges::all_of(anchors, [&](uint64_t anchor) {
            if (!compute_sopp_branch_simm16(anchor, relay_offset))
              return false;
            max_distance = std::max(max_distance, anchor < relay_offset ? relay_offset - anchor
                                                                        : anchor - relay_offset);
            return true;
          });
          if (!reachable || max_distance >= best_max_distance)
            continue;
          std::vector<uint32_t> words;
          std::vector<std::string> decode_errors;
          while (words.size() < kHostWords) {
            auto decoded = decode_relocatable_entry_instruction(
                old_text, candidate_host + words.size() * sizeof(uint32_t), arch, decode_errors);
            if (!decoded) {
              words.clear();
              break;
            }
            words.insert(words.end(), decoded->begin(), decoded->end());
          }
          if (words.size() != kHostWords)
            continue;
          host_offset = candidate_host;
          displaced_words = std::move(words);
          best_max_distance = max_distance;
        }
      }
      if (!host_offset) {
        result.errors.emplace_back(
            "ConSan MOI sampled barrier sync found no relocatable dense relay host in " +
            std::string(container_name));
        return;
      }
      claimed_hosts.emplace_back(*host_offset, *host_offset + kHostBytes);

      const uint64_t dispatcher_offset = new_text.size();
      const uint64_t dispatcher_reservation =
          (indices.size() * kDispatcherWordsPerSite + 12u) * sizeof(uint32_t);
      while (new_text.size() < dispatcher_offset + dispatcher_reservation)
        append_word_bytes(new_text, build_s_nop(0, arch));
      struct DenseBody {
        size_t selected_index = 0;
        uint64_t offset = 0;
        std::vector<uint32_t> words;
      };
      std::vector<DenseBody> bodies;
      for (size_t index : indices) {
        const Planned &planned = selected[index];
        const ConSanBarrierSite &site = planned.candidate.barrier.site;
        ConSanOptions candidate_options = options;
        candidate_options.scratch_vgpr = planned.resources.base;
        if (!apply_moi_transient_sgpr_assignment(candidate_options,
                                                 planned.resources.owner_descriptor_file_offsets)) {
          result.errors.emplace_back(
              "ConSan MOI sampled dense barrier lost its transient scalar assignment");
          return;
        }
        if (planned.private_layout || moi_uses_scalar_persistent_state(candidate_options)) {
          candidate_options.moi_owner_vgpr = static_cast<uint16_t>(planned.resources.base + 7u);
          candidate_options.moi_epoch_vgpr = static_cast<uint16_t>(planned.resources.base + 8u);
        } else if (!apply_moi_persistent_vgpr_assignment(
                       candidate_options, planned.resources.owner_descriptor_file_offsets)) {
          result.errors.emplace_back(
              "ConSan MOI sampled dense barrier lost its persistent VGPR assignment");
          return;
        }
        const uint64_t body_offset = new_text.size();
        const auto vgpr_msb_mode = gfx1250_vgpr_msb_mode_at(
            active_bytes, planned.candidate.barrier.text_file_offset,
            planned.candidate.barrier.container_entry_text_offset, site.file_offset);
        auto body_words = build_sampled_barrier_sync_cave_words(
            bytes, planned.candidate, candidate_options, planned.spill ? &*planned.spill : nullptr,
            planned.scalar_spill ? &*planned.scalar_spill : nullptr, arch, layout, body_offset,
            site.text_offset + site.size,
            planned.private_layout ? &*planned.private_layout : nullptr, vgpr_msb_mode,
            result.errors);
        if (!body_words)
          return;
        append_words_bytes(new_text, *body_words);
        bodies.push_back({index, body_offset, std::move(*body_words)});
      }
      const uint64_t host_body_offset = new_text.size();
      std::vector<uint32_t> host_words = displaced_words;
      if (!append_moi_scc_preserving_indirect_jump(
              host_words, host_body_offset, *host_offset + kHostBytes, base,
              special_state->scc_save_sgpr, /*capture_scc=*/true, arch)) {
        result.errors.emplace_back(
            "ConSan MOI sampled dense barrier relay could not return from its host");
        return;
      }
      append_words_bytes(new_text, host_words);

      const uint64_t island_offset = *host_offset + sizeof(uint32_t);
      std::vector<uint32_t> island_words;
      if (!append_moi_scc_preserving_indirect_jump(island_words, island_offset, dispatcher_offset,
                                                   base, special_state->scc_save_sgpr,
                                                   /*capture_scc=*/true, arch) ||
          island_words.size() != kIndirectIslandWords) {
        result.errors.emplace_back(
            "ConSan MOI sampled dense barrier relay could not encode its island");
        return;
      }
      std::memcpy(new_text.data() + island_offset, island_words.data(),
                  island_words.size() * sizeof(uint32_t));
      const auto host_call = build_direct_call(/*next instruction=*/0);
      if (!host_call) {
        result.errors.emplace_back(
            "ConSan MOI sampled dense barrier relay could not encode its host call");
        return;
      }
      std::memcpy(new_text.data() + *host_offset, &*host_call, sizeof(uint32_t));

      std::vector<uint32_t> dispatcher_words;
      const auto subtract_pc =
          instrumentation::build_s_sub_u32(key_sgpr, call_return_sgpr, base, arch);
      if (!subtract_pc)
        return;
      dispatcher_words.push_back(*subtract_pc);
      dispatcher_words.push_back(build_s_add_u32(key_sgpr, key_sgpr, /*literal=*/255u, arch));
      dispatcher_words.push_back(static_cast<uint32_t>(dispatcher_offset - island_offset));
      const auto append_target = [&](uint32_t key, uint64_t target) -> bool {
        const auto compare = instrumentation::build_s_cmp_eq_u32(key_sgpr, /*literal=*/255u, arch);
        if (!compare)
          return false;
        dispatcher_words.push_back(*compare);
        dispatcher_words.push_back(key);
        std::vector<uint32_t> matched_words;
        const uint64_t matched_offset =
            dispatcher_offset + (dispatcher_words.size() + 1u) * sizeof(uint32_t);
        if (!append_moi_scc_preserving_indirect_jump(matched_words, matched_offset, target, base,
                                                     special_state->scc_save_sgpr,
                                                     /*capture_scc=*/false, arch))
          return false;
        const auto skip =
            instrumentation::build_s_cbranch_scc0(static_cast<int16_t>(matched_words.size()), arch);
        if (!skip)
          return false;
        dispatcher_words.push_back(*skip);
        dispatcher_words.insert(dispatcher_words.end(), matched_words.begin(), matched_words.end());
        return true;
      };
      if (!append_target(/*entry host key=*/0u, host_body_offset))
        return;
      for (const DenseBody &body : bodies) {
        const ConSanBarrierSite &site = selected[body.selected_index].candidate.barrier.site;
        if (!append_target(
                static_cast<uint32_t>(site.text_offset + sizeof(uint32_t) - island_offset),
                body.offset))
          return;
      }
      dispatcher_words.push_back(build_s_endpgm(arch));
      if (dispatcher_words.size() * sizeof(uint32_t) > dispatcher_reservation) {
        result.errors.emplace_back(
            "ConSan MOI sampled dense barrier dispatcher exceeds its reserved relay space");
        return;
      }
      std::memcpy(new_text.data() + dispatcher_offset, dispatcher_words.data(),
                  dispatcher_words.size() * sizeof(uint32_t));

      ConSanPatchInfo host_info;
      host_info.kind = ConSanPatchKind::TrampolineMoiIndirectBranchIsland;
      host_info.anchor_offset = *host_offset;
      host_info.original_size = static_cast<uint32_t>(kHostBytes);
      host_info.trampoline_offset = host_body_offset;
      host_info.trampoline_size = static_cast<uint32_t>(host_words.size() * sizeof(uint32_t));
      host_info.owner_descriptor_file_offsets = group_owners;
      patches.push_back(std::move(host_info));
      ConSanPatchInfo dispatcher_info;
      dispatcher_info.kind = ConSanPatchKind::TrampolineMoiIndirectBranchIsland;
      dispatcher_info.anchor_offset = selected[indices.front()].candidate.barrier.site.text_offset;
      dispatcher_info.trampoline_offset = dispatcher_offset;
      dispatcher_info.trampoline_size =
          static_cast<uint32_t>(dispatcher_words.size() * sizeof(uint32_t));
      dispatcher_info.owner_descriptor_file_offsets = group_owners;
      patches.push_back(std::move(dispatcher_info));
      for (const DenseBody &body : bodies) {
        const Planned &planned = selected[body.selected_index];
        const ConSanBarrierSite &site = planned.candidate.barrier.site;
        const auto call_delta = compute_sopp_branch_simm16(site.text_offset, island_offset);
        const auto call = call_delta ? build_direct_call(*call_delta) : std::nullopt;
        if (!call) {
          result.errors.emplace_back(
              "ConSan MOI sampled dense barrier relay could not encode a barrier call");
          return;
        }
        std::vector<uint32_t> anchor(site.size / sizeof(uint32_t), build_s_nop(0, arch));
        anchor.front() = *call;
        std::memcpy(new_text.data() + site.text_offset, anchor.data(), site.size);
        ConSanPatchInfo info;
        info.kind = ConSanPatchKind::TrampolineMoiSampledSyncMetadata;
        info.anchor_offset = site.text_offset;
        info.trampoline_offset = body.offset;
        info.original_size = site.size;
        info.trampoline_size = static_cast<uint32_t>(body.words.size() * sizeof(uint32_t));
        info.covered_sync_event_count =
            static_cast<uint32_t>(planned.candidate.sequence->member_event_identities.size());
        info.scratch_vgpr = planned.resources.base;
        if (planned.private_layout) {
          info.persistent_epoch_private_offset = planned.private_layout->epoch_offset;
          info.persistent_owner_private_offset = planned.private_layout->owner_offset;
          info.persistent_private_state_end = planned.private_layout->ephemeral_base;
        }
        info.required_private_segment_size = planned.required_private_bytes;
        info.owner_descriptor_file_offsets = planned.resources.owner_descriptor_file_offsets;
        if (planned.spill) {
          info.spilled_vgpr_count = planned.spill->vgpr_count;
          note_dynamic_stack_private_requirement(info, &*planned.spill);
        }
        patches.push_back(std::move(info));
      }
    }
    if (!replace_consan_text(patcher, new_text, options, "MOI sampled barrier sync", result)) {
      return;
    }
    result.elf_bytes = patcher.emit();
    result.patches.insert(result.patches.end(), patches.begin(), patches.end());
    result.modified = true;
    return;
  }
  MoiLocalNopIslandAllocator indirect_allocator(old_text, result, arch,
                                                kMoiLocalIndirectIslandWords);
  for (size_t barrier_index = 0; barrier_index < selected.size(); ++barrier_index) {
    const Planned &planned = selected[barrier_index];
    const ConSanBarrierSite &site = planned.candidate.barrier.site;
    const uint64_t island_offset =
        original_text_size + barrier_index * kIndirectIslandWords * sizeof(uint32_t);
    const uint64_t cave_offset = new_text.size();
    ConSanOptions candidate_options = options;
    candidate_options.scratch_vgpr = planned.resources.base;
    if (!apply_moi_transient_sgpr_assignment(candidate_options,
                                             planned.resources.owner_descriptor_file_offsets)) {
      result.errors.emplace_back(
          "ConSan MOI sampled barrier sync lost its transient scalar assignment");
      return;
    }
    if (planned.private_layout || moi_uses_scalar_persistent_state(candidate_options)) {
      candidate_options.moi_owner_vgpr = static_cast<uint16_t>(planned.resources.base + 7u);
      candidate_options.moi_epoch_vgpr = static_cast<uint16_t>(planned.resources.base + 8u);
    } else if (!apply_moi_persistent_vgpr_assignment(
                   candidate_options, planned.resources.owner_descriptor_file_offsets)) {
      result.errors.emplace_back(
          "ConSan MOI sampled barrier sync lost its persistent VGPR assignment");
      return;
    }
    auto cave_words = build_sampled_barrier_sync_cave_words(
        bytes, planned.candidate, candidate_options, planned.spill ? &*planned.spill : nullptr,
        planned.scalar_spill ? &*planned.scalar_spill : nullptr, arch, layout, cave_offset,
        site.text_offset + site.size, planned.private_layout ? &*planned.private_layout : nullptr,
        std::nullopt, result.errors);
    if (!cave_words) {
      result.errors.emplace_back(
          "ConSan MOI sampled barrier sync could not encode its probe with scratch v" +
          std::to_string(planned.resources.base) + ":v" +
          std::to_string(planned.resources.base + planned.resources.count - 1u) + " epoch v" +
          (candidate_options.moi_epoch_vgpr ? std::to_string(*candidate_options.moi_epoch_vgpr)
                                            : std::string("<unassigned>")) +
          " EXEC-save s" +
          (candidate_options.moi_exec_save_sgpr
               ? std::to_string(*candidate_options.moi_exec_save_sgpr)
               : std::string("<unassigned>")) +
          " across " + std::to_string(planned.candidate.windows.size()) + " owner window(s)");
      return;
    }
    auto fwd = compute_sopp_branch_simm16(site.text_offset, cave_offset);
    if (!fwd) {
      const auto reserved_fwd = compute_sopp_branch_simm16(site.text_offset, island_offset);
      const auto special_state = moi_special_state_sgprs(options);
      std::vector<uint32_t> island_words;
      const bool reserved_island_valid =
          reserved_fwd && special_state && options.moi_exec_save_sgpr &&
          append_moi_scc_preserving_indirect_jump(
              island_words, island_offset, cave_offset, *options.moi_exec_save_sgpr,
              special_state->scc_save_sgpr, /*capture_scc=*/true, arch) &&
          island_words.size() == kIndirectIslandWords && island_offset <= new_text.size() &&
          island_words.size() * sizeof(uint32_t) <= new_text.size() - island_offset;
      if (reserved_island_valid) {
        fwd = reserved_fwd;
        std::memcpy(new_text.data() + island_offset, island_words.data(),
                    island_words.size() * sizeof(uint32_t));
        ConSanPatchInfo island_info;
        island_info.kind = ConSanPatchKind::TrampolineMoiIndirectBranchIsland;
        island_info.anchor_offset = site.text_offset;
        island_info.trampoline_offset = island_offset;
        island_info.trampoline_size = static_cast<uint32_t>(island_words.size() * sizeof(uint32_t));
        island_info.owner_descriptor_file_offsets = planned.resources.owner_descriptor_file_offsets;
        patches.push_back(std::move(island_info));
      } else {
        const auto local_island = indirect_allocator.claim_reachable(site.text_offset);
        fwd = local_island ? compute_sopp_branch_simm16(site.text_offset, *local_island)
                           : std::nullopt;
        if (!local_island || !fwd) {
          result.warnings.emplace_back(
              "ConSan MOI sampled barrier sync skipped barrier at text offset " +
              std::to_string(site.text_offset) + ": no reachable entry island");
          continue;
        }
        if (!emit_moi_local_indirect_entry_island(
                new_text, *local_island, cave_offset, site.text_offset, options,
                planned.resources.owner_descriptor_file_offsets, arch, patches, result.errors,
                "ConSan MOI sampled barrier sync patch")) {
          return;
        }
      }
    }
    std::vector<uint32_t> anchor(site.size / sizeof(uint32_t), build_s_nop(0, arch));
    anchor.front() = build_s_branch(*fwd, arch);
    std::memcpy(new_text.data() + site.text_offset, anchor.data(), site.size);
    append_words_bytes(new_text, *cave_words);
    ConSanPatchInfo info;
    info.kind = ConSanPatchKind::TrampolineMoiSampledSyncMetadata;
    info.anchor_offset = site.text_offset;
    info.trampoline_offset = cave_offset;
    info.original_size = site.size;
    info.trampoline_size = static_cast<uint32_t>(cave_words->size() * sizeof(uint32_t));
    info.covered_sync_event_count =
        static_cast<uint32_t>(planned.candidate.sequence->member_event_identities.size());
    info.scratch_vgpr = planned.resources.base;
    if (planned.private_layout) {
      info.persistent_epoch_private_offset = planned.private_layout->epoch_offset;
      info.persistent_owner_private_offset = planned.private_layout->owner_offset;
      info.persistent_private_state_end = planned.private_layout->ephemeral_base;
    }
    info.required_private_segment_size = planned.required_private_bytes;
    info.owner_descriptor_file_offsets = planned.resources.owner_descriptor_file_offsets;
    if (planned.spill) {
      info.spilled_vgpr_count = planned.spill->vgpr_count;
      note_dynamic_stack_private_requirement(info, &*planned.spill);
    }
    patches.push_back(std::move(info));
  }
  if (!replace_consan_text(patcher, new_text, options, "MOI sampled barrier sync", result)) {
    return;
  }
  result.elf_bytes = patcher.emit();
  result.patches.insert(result.patches.end(), patches.begin(), patches.end());
  result.modified = true;
}

void try_apply_sampled_atomic_sync_patch(std::span<const uint8_t> bytes,
                                         const ConSanOptions &options, rj_code_arch_t arch,
                                         ConSanResult &result) {
  if (!options.moi_track_atomics || options.moi_engine != ConSanMoiEngine::Sampled)
    return;
  if (!options.moi_report_buffer_address || !instrumentation::is_admitted_arch(arch)) {
    result.warnings.emplace_back(
        "ConSan MOI sampled atomic sync requires a supported report-buffer target");
    return;
  }
  const ConSanMoiReportBufferLayout layout = resolve_sampled_report_layout(options);
  const uint32_t selected_window_count = static_cast<uint32_t>(
      std::count_if(result.patches.begin(), result.patches.end(), [](const ConSanPatchInfo &patch) {
        return patch.kind == ConSanPatchKind::InlineMoiSampledWatchpointStore ||
               patch.kind == ConSanPatchKind::TrampolineMoiSampledWatchpointStore;
      }));
  if (selected_window_count == 0 || options.max_patches == 0 ||
      layout.sampled_sync_metadata_capacity == 0) {
    result.warnings.emplace_back(
        "ConSan MOI sampled atomic sync found no selected Ready causal-window capacity");
    return;
  }

  struct Candidate {
    AtomicRecordCandidate atomic;
    SampledAtomicSemantics semantics;
    uint32_t selected_slot = 0;
    bool deferred_acquire = false;
  };
  std::vector<Candidate> candidates;
  const MoiSyncInventoryIndex sync_index(result);
  const bool allow_gfx1250_lds = result.arch == ROCJITSU_CODE_ARCH_GFX1250;
  const auto append = [&](std::string_view name, bool in_kernel, std::optional<uint64_t> descriptor,
                          const std::vector<ConSanAtomicSite> &sites) {
    for (const ConSanAtomicSite &site : sites) {
      const bool cas = consan_atomic_is_compare_exchange(site);
      if (cas && !site.returns_old_value.value_or(false)) {
        result.warnings.emplace_back("ConSan MOI sampled atomic sync skipped " + std::string(name) +
                                     ": compare-exchange-dynamic-outcome-unavailable");
        continue;
      }
      const ConSanAtomicSite normalized = normalize_atomic_record_site(site, allow_gfx1250_lds);
      if (!is_first_light_atomic_record_candidate(normalized, allow_gfx1250_lds))
        continue;
      const auto event_kind = atomic_event_kind_for_site(sync_index, name, in_kernel, normalized);
      if (!event_kind)
        continue;
      std::string reason;
      const auto semantics = sampled_atomic_semantics_for_site(result, sync_index, name, in_kernel,
                                                               normalized, reason);
      if (!semantics) {
        result.warnings.emplace_back("ConSan MOI sampled atomic sync skipped " + std::string(name) +
                                     ": " + reason);
        continue;
      }
      const ConSanSyncEvent *event =
          sync_index.event(ConSanSyncEventKind::Atomic, name, in_kernel, site.text_offset);
      if (descriptor && (event == nullptr || event->execution_owners.size() != 1u ||
                         event->execution_owners.front().descriptor_file_offset != *descriptor))
        descriptor.reset();
      candidates.push_back({{(in_kernel ? "kernel:" : "function:") + std::string(name), normalized,
                             descriptor, *event_kind},
                            *semantics,
                            0,
                            false});
    }
  };
  for (const ConSanKernelInfo &kernel : result.kernels) {
    if (!is_rocclr_runtime_kernel_name(kernel.name))
      append(kernel.name, true, kernel.descriptor_file_offset, kernel.atomic_sites);
  }
  for (const ConSanFunctionInfo &function : result.functions)
    append(function.name, false, std::nullopt, function.atomic_sites);
  if (!options.test_kernel_name_filter.empty()) {
    std::erase_if(candidates, [&](const Candidate &candidate) {
      return candidate.atomic.container_name.find(options.test_kernel_name_filter) ==
             std::string::npos;
    });
  }
  std::unordered_map<uint64_t, size_t> canonical_index_by_file_offset;
  canonical_index_by_file_offset.reserve(candidates.size());
  std::vector<Candidate> canonical_candidates;
  canonical_candidates.reserve(candidates.size());
  for (Candidate &candidate : candidates) {
    const auto [entry, inserted] = canonical_index_by_file_offset.emplace(
        candidate.atomic.site.file_offset, canonical_candidates.size());
    if (inserted) {
      canonical_candidates.push_back(std::move(candidate));
      continue;
    }
    Candidate &retained = canonical_candidates[entry->second];
    if (!atomic_record_candidate_semantics_equal(retained.atomic, candidate.atomic) ||
        std::tie(retained.semantics.role, retained.semantics.scope, retained.semantics.outcome,
                 retained.semantics.byte_count, retained.semantics.descriptor,
                 retained.semantics.cas_failure_descriptor) !=
            std::tie(candidate.semantics.role, candidate.semantics.scope,
                     candidate.semantics.outcome, candidate.semantics.byte_count,
                     candidate.semantics.descriptor, candidate.semantics.cas_failure_descriptor)) {
      result.errors.emplace_back("ConSan MOI sampled atomic site at file offset " +
                                 std::to_string(candidate.atomic.site.file_offset) +
                                 " was decoded inconsistently through aliases '" +
                                 retained.atomic.container_name + "' and '" +
                                 candidate.atomic.container_name + "'");
      return;
    }
    retained.atomic.kernel_descriptor_file_offset.reset();
  }
  candidates = std::move(canonical_candidates);
  std::sort(candidates.begin(), candidates.end(), [](const Candidate &lhs, const Candidate &rhs) {
    return lhs.atomic.site.text_offset < rhs.atomic.site.text_offset;
  });
  if (candidates.empty()) {
    result.warnings.emplace_back(
        "ConSan MOI sampled atomic sync found no qualified atomic sequence");
    return;
  }

  std::span<const uint8_t> active_bytes = bytes;
  if (result.modified)
    active_bytes = {result.elf_bytes.data(), result.elf_bytes.size()};
  AmdGpuCodeObject code_object(active_bytes.data(), active_bytes.size());
  CodeObjectPatcher patcher(code_object);
  const std::span<const uint8_t> old_text = patcher.text_bytes();
  if (old_text.empty()) {
    result.errors.emplace_back("ConSan MOI sampled atomic sync found no .text section");
    return;
  }

  // Direct sampled access instrumentation may already have grown .text and
  // recorded the then-active descriptor offsets on prologue patches. A second
  // growth would otherwise leave those intermediate offsets orphaned. Keep
  // owner evidence canonical in the original inventory coordinate system.
  std::unordered_map<uint64_t, uint64_t> active_to_canonical_owner;
  for (const AmdGpuKernelInfo &active_kernel : code_object.kernels()) {
    const auto canonical =
        std::ranges::find(result.kernels, active_kernel.name, &ConSanKernelInfo::name);
    if (canonical != result.kernels.end())
      active_to_canonical_owner.emplace(active_kernel.descriptor_file_offset,
                                        canonical->descriptor_file_offset);
  }
  const auto canonicalize_owners = [&](std::vector<uint64_t> &owners) {
    for (uint64_t &owner : owners) {
      const auto canonical = active_to_canonical_owner.find(owner);
      if (canonical != active_to_canonical_owner.end())
        owner = canonical->second;
    }
  };
  for (ConSanCandidateResourcePlan &plan : result.resource_plans)
    canonicalize_owners(plan.owner_descriptor_file_offsets);
  for (ConSanPatchInfo &patch : result.patches)
    canonicalize_owners(patch.owner_descriptor_file_offsets);

  MoiResourcePlanningState resource_state(bytes, arch, options, result);
  if (!resource_state.valid) {
    result.warnings.emplace_back("ConSan MOI sampled atomic sync lacks a complete kernel CFG");
    return;
  }

  struct Window {
    uint32_t slot = 0;
    const ConSanPatchInfo *patch = nullptr;
    ConSanLdsAccessKind access_kind = ConSanLdsAccessKind::Other;
    const MoiKernelResourceContext *owner = nullptr;
  };
  std::vector<Window> windows;
  windows.reserve(selected_window_count);
  bool has_multi_range_window = false;
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.kind != ConSanPatchKind::InlineMoiSampledWatchpointStore &&
        patch.kind != ConSanPatchKind::TrampolineMoiSampledWatchpointStore)
      continue;
    if (patch.sampled_access_kind != ConSanLdsAccessKind::Read &&
        patch.sampled_access_kind != ConSanLdsAccessKind::Write &&
        patch.sampled_access_kind != ConSanLdsAccessKind::Atomic)
      continue;
    has_multi_range_window |= patch.sampled_access_range_count > 1u;
    for (const MoiKernelResourceContext &context : resource_state.contexts) {
      if (context.kernel == nullptr || !kernel_owns_patch(*context.kernel, patch))
        continue;
      // One physical access in a shared helper is one sampled slot, but it has
      // an independent causal view in every kernel that can execute it. Those
      // owners are mutually exclusive dispatches, so each may safely attach
      // one synchronization edge to the same metadata slot.
      windows.push_back({patch.sampled_first_slot, &patch, patch.sampled_access_kind, &context});
    }
  }
  if (has_multi_range_window) {
    result.warnings.emplace_back(
        "ConSan MOI sampled atomic sync currently attaches only the first logical range of a "
        "multi-range access");
  }

  struct Association {
    size_t candidate_index = 0;
    size_t window_index = 0;
    SampledCfgDistance distance;
  };
  std::vector<Association> proposed;
  const auto owner_executes_candidate = [&](const MoiKernelResourceContext &owner,
                                            const Candidate &candidate) {
    if (owner.kernel == nullptr)
      return false;
    if (candidate.atomic.kernel_descriptor_file_offset)
      return *candidate.atomic.kernel_descriptor_file_offset ==
             owner.kernel->descriptor_file_offset;
    if (const ConSanCandidateResourcePlan *plan = resource_plan_for_site(
            result, ConSanResourceSiteKind::Atomic, candidate.atomic.site.text_offset)) {
      return std::ranges::find(plan->owner_descriptor_file_offsets,
                               owner.kernel->descriptor_file_offset) !=
             plan->owner_descriptor_file_offsets.end();
    }
    return std::ranges::any_of(owner.scope.blocks, [&](BasicBlock *block) {
      return instruction_at(block, candidate.atomic.site.text_offset) != nullptr;
    });
  };
  for (size_t window_index = 0; window_index < windows.size(); ++window_index) {
    const Window &window = windows[window_index];
    std::optional<Association> best;
    bool ambiguous = false;
    for (size_t candidate_index = 0; candidate_index < candidates.size(); ++candidate_index) {
      const Candidate &candidate = candidates[candidate_index];
      if (!owner_executes_candidate(*window.owner, candidate))
        continue;
      const bool release_side =
          (window.access_kind == ConSanLdsAccessKind::Write ||
           window.access_kind == ConSanLdsAccessKind::Atomic) &&
          (candidate.semantics.role == ConSanMoiSampledSyncRole::RmwRelease ||
           candidate.semantics.role == ConSanMoiSampledSyncRole::RmwAcquireRelease);
      const bool acquire_side =
          (window.access_kind == ConSanLdsAccessKind::Read ||
           window.access_kind == ConSanLdsAccessKind::Atomic) &&
          (candidate.semantics.role == ConSanMoiSampledSyncRole::RmwAcquire ||
           candidate.semantics.role == ConSanMoiSampledSyncRole::RmwAcquireRelease);
      std::optional<SampledCfgDistance> distance;
      if (release_side) {
        distance =
            sampled_cfg_forward_distance(resource_state, *window.owner, window.patch->anchor_offset,
                                         candidate.atomic.site.text_offset);
      } else if (acquire_side) {
        distance = sampled_cfg_forward_distance(resource_state, *window.owner,
                                                candidate.atomic.site.text_offset,
                                                window.patch->anchor_offset);
      }
      if (!distance)
        continue;
      Association current{candidate_index, window_index, *distance};
      if (!best || current.distance < best->distance) {
        best = current;
        ambiguous = false;
      } else if (current.distance == best->distance) {
        ambiguous = true;
      }
    }
    if (best && !ambiguous)
      proposed.push_back(*best);
  }

  // Generic access and atomic helpers can both have recovered kernel owners
  // without an interprocedural CFG edge connecting their reusable bodies.
  // The emitted probe still validates dispatch, workgroup, owner, epoch,
  // Ready-window identity, and watchpoint state before it publishes. Attach
  // otherwise-unmatched atomics to the closest role-compatible window owned
  // by the same kernel; concurrent/static aliases fail through the existing
  // transactional collision path rather than overwriting metadata.
  for (size_t candidate_index = 0; candidate_index < candidates.size(); ++candidate_index) {
    if (std::ranges::any_of(proposed, [&](const Association &association) {
          return association.candidate_index == candidate_index;
        }))
      continue;
    const Candidate &candidate = candidates[candidate_index];
    std::optional<Association> best;
    for (size_t window_index = 0; window_index < windows.size(); ++window_index) {
      const Window &window = windows[window_index];
      if (!owner_executes_candidate(*window.owner, candidate))
        continue;
      const bool release_side =
          (window.access_kind == ConSanLdsAccessKind::Write ||
           window.access_kind == ConSanLdsAccessKind::Atomic) &&
          (candidate.semantics.role == ConSanMoiSampledSyncRole::RmwRelease ||
           candidate.semantics.role == ConSanMoiSampledSyncRole::RmwAcquireRelease);
      const bool acquire_side =
          (window.access_kind == ConSanLdsAccessKind::Read ||
           window.access_kind == ConSanLdsAccessKind::Atomic) &&
          (candidate.semantics.role == ConSanMoiSampledSyncRole::RmwAcquire ||
           candidate.semantics.role == ConSanMoiSampledSyncRole::RmwAcquireRelease);
      if (!release_side && !acquire_side)
        continue;
      const uint64_t text_distance =
          window.patch->anchor_offset < candidate.atomic.site.text_offset
              ? candidate.atomic.site.text_offset - window.patch->anchor_offset
              : window.patch->anchor_offset - candidate.atomic.site.text_offset;
      Association current{
          candidate_index, window_index, {std::numeric_limits<uint32_t>::max(), text_distance}};
      if (!best || current.distance < best->distance)
        best = current;
    }
    if (best)
      proposed.push_back(*best);
  }

  // A synchronization edge may order only one selected access window. If two
  // windows independently choose the same edge, retain a strictly closer
  // owner/CFG match and fail closed on equal-distance aliases.
  std::vector<bool> rejected(proposed.size(), false);
  for (size_t i = 0; i < proposed.size(); ++i) {
    for (size_t j = i + 1u; j < proposed.size(); ++j) {
      if (proposed[i].candidate_index != proposed[j].candidate_index)
        continue;
      const Window &lhs = windows[proposed[i].window_index];
      const Window &rhs = windows[proposed[j].window_index];
      if (lhs.patch == rhs.patch && lhs.slot == rhs.slot) {
        // The same physical access and atomic in a shared helper appear once
        // per owning kernel CFG. They need one physical patch, not competing
        // duplicate attachments.
        rejected[j] = true;
        continue;
      }
      if (proposed[i].distance < proposed[j].distance)
        rejected[j] = true;
      else if (proposed[j].distance < proposed[i].distance)
        rejected[i] = true;
      else
        rejected[i] = rejected[j] = true;
    }
  }
  std::vector<Candidate> associated;
  // Metadata capacity limits distinct selected slots, not static attachment
  // sites. Multiple sites targeting one slot use the transactional collision
  // path if they execute in the same sampled window.
  const size_t static_patch_budget = std::min<size_t>(options.max_patches, proposed.size());
  associated.reserve(std::min(static_patch_budget, proposed.size()));
  for (size_t i = 0; i < proposed.size() && associated.size() < static_patch_budget; ++i) {
    if (rejected[i])
      continue;
    Candidate candidate = candidates[proposed[i].candidate_index];
    candidate.selected_slot = windows[proposed[i].window_index].slot;
    candidate.deferred_acquire =
        windows[proposed[i].window_index].access_kind == ConSanLdsAccessKind::Read ||
        (windows[proposed[i].window_index].access_kind == ConSanLdsAccessKind::Atomic &&
         candidate.semantics.role == ConSanMoiSampledSyncRole::RmwAcquire);
    associated.push_back(std::move(candidate));
  }
  std::ranges::sort(associated, {}, &Candidate::selected_slot);
  candidates = std::move(associated);
  if (candidates.empty()) {
    result.warnings.emplace_back(
        "ConSan MOI sampled atomic sync found no unambiguous owner/CFG-associated edge");
    return;
  }

  struct Planned {
    Candidate candidate;
    ResolvedMoiScratchPlan resources;
    ConSanMoiAtomicAddressPlan address_plan;
    std::optional<VgprSpillSequence> spill;
    std::optional<MoiPrivateEpochLayout> private_layout;
    uint32_t required_private_bytes = 0;
    uint64_t physical_anchor_offset = 0;
  };
  std::vector<Planned> selected;
  MoiSpillManagers spill_managers;
  MoiDescriptorVgprRequirements descriptor_requirements;
  MoiDescriptorPrivateRequirements private_requirements;
  const size_t remaining = static_patch_budget;
  const uint16_t scratch_count = sampled_atomic_scratch_count();
  for (size_t index = 0; index < candidates.size() && selected.size() < remaining; ++index) {
    Candidate candidate = candidates[index];
    const ConSanAtomicSite &site = candidate.atomic.site;
    const auto composed_access_patch = std::ranges::find_if(result.patches, [&](const auto &patch) {
      return (patch.kind == ConSanPatchKind::InlineMoiSampledWatchpointStore ||
              patch.kind == ConSanPatchKind::TrampolineMoiSampledWatchpointStore) &&
             patch.anchor_offset == site.text_offset && patch.relocated_guest_instruction_offset;
    });
    const uint64_t physical_anchor_offset =
        composed_access_patch == result.patches.end()
            ? site.text_offset
            : *composed_access_patch->relocated_guest_instruction_offset;
    if (physical_anchor_offset > old_text.size() ||
        site.size > old_text.size() - physical_anchor_offset)
      continue;
    const bool overlap = std::any_of(
        result.patches.begin(), result.patches.end(), [&](const ConSanPatchInfo &patch) {
          if (composed_access_patch != result.patches.end() && &patch == &*composed_access_patch)
            return false;
          const uint64_t patch_end =
              patch.anchor_offset + std::max<uint32_t>(patch.original_size, 1u);
          return site.text_offset < patch_end && patch.anchor_offset < site.text_offset + site.size;
        });
    if (overlap)
      continue;
    const ConSanCandidateResourcePlan *existing =
        resource_plan_for_site(result, ConSanResourceSiteKind::Atomic, site.text_offset);
    ConSanCandidateResourcePlan plan =
        existing ? *existing
                 : plan_moi_resource_site(resource_state, options, ConSanResourceSiteKind::Atomic,
                                          index, site.text_offset,
                                          candidate.atomic.kernel_descriptor_file_offset,
                                          scratch_count, nullptr, &site);
    if (!existing)
      result.resource_plans.push_back(plan);
    const auto resources = resolve_moi_scratch_plan(plan, options, scratch_count);
    if (!resources) {
      result.warnings.emplace_back("ConSan MOI sampled atomic sync has no " +
                                   std::to_string(scratch_count) + "-VGPR resource plan in " +
                                   candidate.atomic.container_name);
      continue;
    }
    const bool allow_post_guest_spill_operand_overlap =
        resources->source == ConSanRegisterAllocationSource::SpillRequired &&
        sampled_atomic_guest_preserves_address(site);
    const ConSanMoiAtomicAddressPlan address_plan =
        plan_consan_moi_atomic_address(site, resources->base, resources->count, resources->source,
                                       arch, allow_post_guest_spill_operand_overlap);
    if (!address_plan.supported()) {
      result.warnings.emplace_back(
          "ConSan MOI sampled atomic sync rejected exact address plan: " +
          std::string(consan_moi_atomic_address_support_name(address_plan.support)));
      continue;
    }
    if (!candidate.atomic.kernel_descriptor_file_offset) {
      candidate.atomic.kernel_descriptor_file_offset =
          common_moi_record_owner_descriptor(bytes, *resources, arch, result.warnings);
      if (!candidate.atomic.kernel_descriptor_file_offset)
        continue;
    }
    std::optional<MoiPrivateEpochLayout> private_layout;
    if (options.automatic_moi_private_epoch) {
      private_layout =
          build_sampled_private_epoch_layout(result, *resources, options, arch, result.warnings);
      if (!private_layout)
        continue;
    }
    std::optional<VgprSpillSequence> spill;
    if (resources->source == ConSanRegisterAllocationSource::SpillRequired) {
      spill = build_moi_spill_sequence(
          result, *resources, options, spill_managers, arch, result.warnings,
          private_layout ? std::optional<uint32_t>(private_layout->ephemeral_base) : std::nullopt);
      if (!spill)
        continue;
    }
    const uint32_t required_private_bytes =
        std::max(private_layout ? private_layout->ephemeral_base : 0u,
                 spill ? spill->total_private_bytes : 0u);
    if (required_private_bytes != 0u) {
      for (uint64_t descriptor_offset : resources->owner_descriptor_file_offsets) {
        auto [it, inserted] =
            private_requirements.emplace(descriptor_offset, required_private_bytes);
        if (!inserted)
          it->second = std::max(it->second, required_private_bytes);
      }
    }
    note_descriptor_requirements(descriptor_requirements, *resources);
    selected.push_back({std::move(candidate), *resources, address_plan, std::move(spill),
                        std::move(private_layout), required_private_bytes, physical_anchor_offset});
  }
  if (selected.empty()) {
    result.warnings.emplace_back("ConSan MOI sampled atomic sync found no patchable sequence");
    return;
  }
  if (!apply_descriptor_requirements(patcher, code_object, active_bytes, result,
                                     descriptor_requirements, arch, result.errors) ||
      !apply_spill_descriptor_requirements(patcher, code_object, active_bytes, result,
                                           private_requirements, result.errors))
    return;

  std::vector<uint8_t> new_text(old_text.begin(), old_text.end());
  DbiPatchPlacementPlanner placement_planner(arch, old_text.size());
  MoiLocalNopIslandAllocator indirect_allocator(old_text, result, arch,
                                                kMoiLocalIndirectIslandWords);
  std::vector<MoiLocalNopIsland> appended_indirect_islands;
  if (selected.size() > 1u) {
    const uint64_t island_bytes =
        static_cast<uint64_t>(kMoiLocalIndirectIslandWords) * sizeof(uint32_t);
    const uint64_t bank_offset = placement_planner.appended_end();
    const uint64_t bank_bytes = selected.size() * island_bytes;
    std::string bank_error;
    if (!placement_planner.reserve_appended_prefix(bank_bytes, &bank_error)) {
      result.errors.emplace_back("ConSan MOI sampled atomic island bank failed: " + bank_error);
      return;
    }
    appended_indirect_islands.reserve(selected.size());
    for (size_t index = 0; index < selected.size(); ++index) {
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
  for (const Planned &planned : selected) {
    const ConSanAtomicSite &site = planned.candidate.atomic.site;
    const uint64_t physical_anchor_offset = planned.physical_anchor_offset;
    const uint64_t cave_offset = placement_planner.appended_end();
    ConSanOptions candidate_options = options;
    candidate_options.scratch_vgpr = planned.resources.base;
    if (!apply_moi_transient_sgpr_assignment(candidate_options,
                                             planned.resources.owner_descriptor_file_offsets)) {
      result.errors.emplace_back(
          "ConSan MOI sampled atomic sync lost its transient scalar assignment");
      return;
    }
    if (planned.private_layout || moi_uses_scalar_persistent_state(candidate_options)) {
      candidate_options.moi_owner_vgpr =
          static_cast<uint16_t>(planned.resources.base + SampledAtomicScratchLayout::kOwner);
      candidate_options.moi_epoch_vgpr =
          static_cast<uint16_t>(planned.resources.base + SampledAtomicScratchLayout::kEpoch);
    } else if (!apply_moi_persistent_vgpr_assignment(
                   candidate_options, planned.resources.owner_descriptor_file_offsets)) {
      result.errors.emplace_back(
          "ConSan MOI sampled atomic sync lost its persistent VGPR assignment");
      return;
    }
    uint32_t guest_instruction_offset = 0;
    auto cave_words =
        planned.candidate.deferred_acquire
            ? build_sampled_pending_acquire_cave_words(
                  bytes, planned.candidate.atomic, planned.address_plan, candidate_options,
                  planned.spill ? &*planned.spill : nullptr,
                  planned.private_layout ? &*planned.private_layout : nullptr, arch,
                  planned.candidate.selected_slot, layout, cave_offset,
                  physical_anchor_offset + site.size, result.errors, &guest_instruction_offset)
            : build_sampled_atomic_sync_cave_words(
                  bytes, planned.candidate.atomic, planned.address_plan, candidate_options,
                  planned.spill ? &*planned.spill : nullptr,
                  planned.private_layout ? &*planned.private_layout : nullptr, arch,
                  planned.candidate.selected_slot, layout, cave_offset,
                  physical_anchor_offset + site.size, result.errors, &guest_instruction_offset);
    if (!cave_words) {
      result.warnings.emplace_back(
          "ConSan MOI sampled atomic sync could not lower associated " +
          std::string(planned.candidate.deferred_acquire ? "deferred-acquire" : "release") +
          " site at text offset " + std::to_string(site.text_offset));
      continue;
    }
    const bool direct_entry =
        compute_sopp_branch_simm16(physical_anchor_offset, cave_offset).has_value();
    std::optional<uint64_t> indirect_entry_island;
    if (!direct_entry) {
      indirect_entry_island = claim_appended_island(physical_anchor_offset);
      if (!indirect_entry_island)
        indirect_entry_island = indirect_allocator.claim_reachable(physical_anchor_offset);
    }
    if (!direct_entry && !indirect_entry_island) {
      result.warnings.emplace_back(
          "ConSan MOI sampled atomic sync found no reachable indirect entry island in " +
          planned.candidate.atomic.container_name);
      continue;
    }
    std::string placement_error;
    const auto placement = placement_planner.plan_indirect_appended(
        physical_anchor_offset, site.size, cave_words->size() * sizeof(uint32_t), &placement_error);
    if (!placement) {
      result.errors.emplace_back("ConSan MOI sampled atomic sync placement failed: " +
                                 placement_error);
      return;
    }
    if (placement->body_offset != cave_offset || new_text.size() != cave_offset) {
      result.errors.emplace_back("ConSan MOI sampled atomic sync has a stale cave mapping");
      return;
    }
    const uint64_t entry_target = indirect_entry_island.value_or(placement->body_offset);
    const auto fwd = compute_sopp_branch_simm16(placement->anchor_offset, entry_target);
    if (!fwd) {
      result.errors.emplace_back("ConSan MOI sampled atomic sync forward branch is out of range");
      return;
    }
    std::vector<uint32_t> anchor(site.size / sizeof(uint32_t), build_s_nop(0, arch));
    if (anchor.empty())
      return;
    anchor.front() = build_s_branch(*fwd, arch);
    std::memcpy(new_text.data() + physical_anchor_offset, anchor.data(),
                anchor.size() * sizeof(uint32_t));
    if (indirect_entry_island &&
        !emit_moi_local_indirect_entry_island(
            new_text, *indirect_entry_island, placement->body_offset, physical_anchor_offset,
            options, planned.resources.owner_descriptor_file_offsets, arch, patches, result.errors,
            "ConSan MOI sampled atomic sync patch")) {
      return;
    }
    append_words_bytes(new_text, *cave_words);

    ConSanPatchInfo info;
    info.kind = ConSanPatchKind::TrampolineMoiSampledSyncMetadata;
    info.anchor_offset = site.text_offset;
    info.trampoline_offset = cave_offset;
    info.original_size = site.size;
    info.trampoline_size = static_cast<uint32_t>(cave_words->size() * sizeof(uint32_t));
    info.relocated_guest_instruction_offset = cave_offset + guest_instruction_offset;
    info.sampled_access_kind = ConSanLdsAccessKind::Atomic;
    info.scratch_vgpr = planned.resources.base;
    info.owner_descriptor_file_offsets = planned.resources.owner_descriptor_file_offsets;
    if (planned.private_layout) {
      info.persistent_epoch_private_offset = planned.private_layout->epoch_offset;
      info.persistent_owner_private_offset = planned.private_layout->owner_offset;
      info.persistent_private_state_end = planned.private_layout->ephemeral_base;
    }
    info.required_private_segment_size = planned.required_private_bytes;
    if (planned.spill) {
      info.spilled_vgpr_count = planned.spill->vgpr_count;
      note_dynamic_stack_private_requirement(info, &*planned.spill);
    }
    patches.push_back(std::move(info));
  }
  if (!replace_consan_text(patcher, new_text, options, "MOI sampled atomic sync", result)) {
    return;
  }
  result.elf_bytes = patcher.emit();
  result.patches.insert(result.patches.end(), patches.begin(), patches.end());
  result.modified = true;
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_fence_record_cave_words(
    std::span<const uint8_t> bytes, const FenceRecordCandidate &candidate,
    const ConSanMoiAtomicAddressPlan &address_plan, const ConSanOptions &options,
    const VgprSpillSequence *spill, const InlineSgprSpillSequence *scalar_spill,
    rj_code_arch_t arch, uint32_t record_index, uint32_t record_capacity_or_count,
    size_t fence_records_offset, uint64_t cave_text_offset, uint64_t return_text_offset,
    std::optional<uint16_t> call_return_sgpr, std::span<const uint32_t> displaced_tail_words,
    std::optional<uint32_t> private_owner_offset, std::optional<uint16_t> private_owner_shift,
    std::vector<std::string> &errors) {
  if (!options.scratch_vgpr) {
    errors.emplace_back("ConSan MOI fence record patch requires RJ_CONSAN_TMP_VGPR");
    return std::nullopt;
  }
  const bool requires_raw_th = arch != ROCJITSU_CODE_ARCH_CDNA3 && arch != ROCJITSU_CODE_ARCH_CDNA4;
  if (!address_plan.supported() || !candidate.fence.raw_scope ||
      (requires_raw_th && !candidate.communication_site.raw_th) ||
      candidate.fence.communication_event_identity.empty()) {
    errors.emplace_back("ConSan MOI fence record patch requires qualified address semantics");
    return std::nullopt;
  }
  if (static_cast<uint32_t>(*options.scratch_vgpr) + address_plan.scratch_vgpr_count > 256u ||
      reject_atomic_candidate_scratch_overlap(candidate.communication_site, *options.scratch_vgpr,
                                              address_plan.scratch_vgpr_count, errors) ||
      reject_optional_scratch_range_overlap(options.moi_owner_vgpr, *options.scratch_vgpr,
                                            address_plan.scratch_vgpr_count, "MOI owner", errors) ||
      reject_optional_scratch_range_overlap(options.moi_epoch_vgpr, *options.scratch_vgpr,
                                            address_plan.scratch_vgpr_count, "MOI epoch", errors))
    return std::nullopt;
  if (candidate.fence.file_offset > bytes.size() ||
      candidate.fence.size > bytes.size() - candidate.fence.file_offset) {
    errors.emplace_back("ConSan MOI fence record patch site exceeds ELF bytes");
    return std::nullopt;
  }

  std::optional<uint16_t> derived_owner_vgpr;
  std::vector<uint32_t> derived_owner_words;
  if (!options.moi_owner_vgpr && !options.moi_persistent_owner_sgpr) {
    const uint16_t value_vgpr = static_cast<uint16_t>(
        *options.scratch_vgpr +
        (arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4 ? 5u : 2u));
    const auto owner = build_moi_record_owner_derivation(
        bytes, candidate.kernel_descriptor_file_offset, private_owner_offset, private_owner_shift,
        value_vgpr, arch, "fence record patch", errors);
    if (!owner)
      return std::nullopt;
    derived_owner_vgpr = owner->vgpr;
    derived_owner_words = owner->words;
  }

  const auto persistent_workgroup_sources =
      record_replay_persistent_workgroup_sources(options, arch);
  if (!persistent_workgroup_sources) {
    errors.emplace_back(
        "ConSan MOI Record/Replay fence requires one exact entry-captured workgroup tuple");
    return std::nullopt;
  }
  const ConSanMoiWorkgroupSources workgroup_sources = *persistent_workgroup_sources;

  std::vector<uint32_t> words;
  const std::optional<uint16_t> vgpr_msb_mode =
      arch == ROCJITSU_CODE_ARCH_GFX1250
          ? gfx1250_vgpr_msb_mode_at(bytes, candidate.text_file_offset,
                                     candidate.container_entry_text_offset,
                                     candidate.fence.file_offset)
          : std::nullopt;
  const bool select_low_vgpr_bank = vgpr_msb_mode.value_or(0u) != 0u;
  words.reserve(candidate.fence.size / sizeof(uint32_t) + 80u +
                (spill ? spill->save_words.size() + spill->restore_words.size() : 0u) +
                (select_low_vgpr_bank ? 2u : 0u));
  if (select_low_vgpr_bank)
    words.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
        static_cast<uint8_t>(*vgpr_msb_mode), 0u, arch));
  if (spill)
    words.insert(words.end(), spill->save_words.begin(), spill->save_words.end());
  if (scalar_spill)
    words.insert(words.end(), scalar_spill->save_words.begin(), scalar_spill->save_words.end());
  if (address_plan.requires_materialization()) {
    const auto special_state = moi_special_state_sgprs(options);
    if (!special_state) {
      errors.emplace_back(
          "ConSan MOI fence record address materialization requires special-state SGPRs");
      return std::nullopt;
    }
    const auto address_words = build_consan_moi_atomic_address_materialization(
        address_plan, special_state->vcc_save_sgpr, special_state->scc_save_sgpr, arch);
    if (!address_words) {
      errors.emplace_back("ConSan MOI fence record patch could not materialize its address");
      return std::nullopt;
    }
    words.insert(words.end(), address_words->begin(), address_words->end());
  }
  if (!derived_owner_words.empty() && arch != ROCJITSU_CODE_ARCH_CDNA3 &&
      arch != ROCJITSU_CODE_ARCH_CDNA4) {
    words.insert(words.end(), derived_owner_words.begin(), derived_owner_words.end());
  }
  for (uint64_t offset = 0; offset < candidate.fence.size; offset += sizeof(uint32_t)) {
    uint32_t word = 0;
    std::memcpy(&word, bytes.data() + candidate.fence.file_offset + offset, sizeof(word));
    words.push_back(word);
  }
  const uint16_t address_vgpr = address_plan.result_address_vgpr;
  const uint16_t recorded_address_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 6u);
  if (arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4) {
    words.push_back(
        build_v_mov_b32_e32(recorded_address_vgpr, vector_source_vgpr(address_vgpr), arch));
    words.push_back(
        build_v_mov_b32_e32(static_cast<uint16_t>(recorded_address_vgpr + 1u),
                            vector_source_vgpr(static_cast<uint16_t>(address_vgpr + 1u)), arch));
  }
  if (!append_save_moi_special_state(words, options, arch)) {
    errors.emplace_back("ConSan MOI fence record patch could not save VCC/SCC");
    return std::nullopt;
  }

  const uint64_t base = *options.moi_report_buffer_address;
  const uint64_t record_base =
      base + fence_records_offset +
      ((arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4)
           ? 0u
           : static_cast<uint64_t>(record_index) * sizeof(ConSanMoiFenceRecord));
  const ConSanMoiFenceEventKind kind = candidate.fence.memory_role == ConSanSyncMemoryRole::Release
                                           ? ConSanMoiFenceEventKind::Release
                                           : ConSanMoiFenceEventKind::Acquire;
  if (arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4) {
    if (!options.moi_exec_save_sgpr) {
      errors.emplace_back("ConSan MOI dynamic fence record requires scalar EXEC state");
      return std::nullopt;
    }
    const uint16_t lane_rank_vgpr = *options.scratch_vgpr;
    const uint16_t slot_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 2u);
    const uint16_t value_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 5u);
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
    if (!save_active_exec || !mbcnt_lo || !mbcnt_hi || !first_active_lane || !narrow_exec) {
      errors.emplace_back("ConSan MOI dynamic fence record could not select a wave publisher");
      return std::nullopt;
    }
    words.push_back(*save_active_exec);
    words.insert(words.end(), mbcnt_lo->begin(), mbcnt_lo->end());
    words.insert(words.end(), mbcnt_hi->begin(), mbcnt_hi->end());
    words.push_back(*first_active_lane);
    words.push_back(*narrow_exec);
    if (!append_atomic_fetch_add_one_u32(words,
                                         base + offsetof(ConSanMoiReportHeader, fence_record_count),
                                         slot_vgpr, *options.scratch_vgpr, arch)) {
      errors.emplace_back("ConSan MOI dynamic fence record could not reserve a wave slot");
      return std::nullopt;
    }
    const auto mov_capacity =
        instrumentation::build_v_mov_b32_literal(value_vgpr, record_capacity_or_count, arch);
    const auto slot_in_capacity =
        instrumentation::build_v_cmp_gt_u32_vcc(vector_source_vgpr(value_vgpr), slot_vgpr, arch);
    if (!mov_capacity || !slot_in_capacity) {
      errors.emplace_back("ConSan MOI dynamic fence record could not guard capacity");
      return std::nullopt;
    }
    words.insert(words.end(), mov_capacity->begin(), mov_capacity->end());
    words.push_back(*slot_in_capacity);
    std::vector<uint32_t> record_words;
    record_words.insert(record_words.end(), derived_owner_words.begin(), derived_owner_words.end());
    if ((derived_owner_vgpr &&
         !append_dynamic_fence_store_u32_vgpr(
             record_words, record_base + offsetof(ConSanMoiFenceRecord, owner_id),
             *derived_owner_vgpr, slot_vgpr, *options.scratch_vgpr, arch)) ||
        !append_dynamic_fence_event_index_store(
            record_words, base + offsetof(ConSanMoiReportHeader, event_counter),
            record_base + offsetof(ConSanMoiFenceRecord, event_index), slot_vgpr,
            *options.scratch_vgpr, arch) ||
        !append_dynamic_record_store_moi_report_dispatch_id_pair(
            record_words, kDynamicFenceRecordLayout,
            record_base + offsetof(ConSanMoiFenceRecord, generation), options, slot_vgpr,
            *options.scratch_vgpr, arch, ConSanMoiLiteralDispatchIdPolicy::AnyArchitecture) ||
        !append_dynamic_fence_store_workgroup_source(
            record_words, record_base + offsetof(ConSanMoiFenceRecord, workgroup_x),
            workgroup_sources.x, slot_vgpr, *options.scratch_vgpr, arch) ||
        !append_dynamic_fence_store_workgroup_source(
            record_words, record_base + offsetof(ConSanMoiFenceRecord, workgroup_y),
            workgroup_sources.y, slot_vgpr, *options.scratch_vgpr, arch) ||
        !append_dynamic_fence_store_workgroup_source(
            record_words, record_base + offsetof(ConSanMoiFenceRecord, workgroup_z),
            workgroup_sources.z, slot_vgpr, *options.scratch_vgpr, arch) ||
        (options.moi_owner_vgpr &&
         !append_dynamic_fence_store_u32_vgpr(
             record_words, record_base + offsetof(ConSanMoiFenceRecord, owner_id),
             *options.moi_owner_vgpr, slot_vgpr, *options.scratch_vgpr, arch)) ||
        (options.moi_persistent_owner_sgpr &&
         !append_dynamic_fence_store_u32_scalar_src(
             record_words, record_base + offsetof(ConSanMoiFenceRecord, owner_id),
             *options.moi_persistent_owner_sgpr, slot_vgpr, *options.scratch_vgpr, arch)) ||
        (options.moi_epoch_vgpr &&
         !append_dynamic_fence_store_u32_vgpr(
             record_words, record_base + offsetof(ConSanMoiFenceRecord, epoch),
             *options.moi_epoch_vgpr, slot_vgpr, *options.scratch_vgpr, arch)) ||
        (options.moi_persistent_epoch_sgpr &&
         !append_dynamic_fence_store_u32_scalar_src(
             record_words, record_base + offsetof(ConSanMoiFenceRecord, epoch),
             *options.moi_persistent_epoch_sgpr, slot_vgpr, *options.scratch_vgpr, arch)) ||
        !append_dynamic_fence_store_u32_literal(
            record_words, record_base + offsetof(ConSanMoiFenceRecord, instruction_offset),
            static_cast<uint32_t>(candidate.fence.text_offset), slot_vgpr, *options.scratch_vgpr,
            arch) ||
        !append_dynamic_fence_store_u32_literal(
            record_words, record_base + offsetof(ConSanMoiFenceRecord, kind),
            static_cast<uint32_t>(kind), slot_vgpr, *options.scratch_vgpr, arch) ||
        !append_dynamic_fence_store_u32_literal(
            record_words, record_base + offsetof(ConSanMoiFenceRecord, scope),
            *candidate.fence.raw_scope, slot_vgpr, *options.scratch_vgpr, arch) ||
        !append_dynamic_fence_store_u32_literal(
            record_words, record_base + offsetof(ConSanMoiFenceRecord, semantics),
            candidate.communication_site.raw_th.value_or(0u), slot_vgpr, *options.scratch_vgpr,
            arch) ||
        !append_dynamic_fence_store_u32_vgpr(
            record_words, record_base + offsetof(ConSanMoiFenceRecord, communication_token),
            recorded_address_vgpr, slot_vgpr, *options.scratch_vgpr, arch) ||
        !append_dynamic_fence_store_u32_vgpr(
            record_words,
            record_base + offsetof(ConSanMoiFenceRecord, communication_token) + sizeof(uint32_t),
            static_cast<uint16_t>(recorded_address_vgpr + 1u), slot_vgpr, *options.scratch_vgpr,
            arch)) {
      errors.emplace_back("ConSan MOI dynamic fence record could not encode record stores");
      return std::nullopt;
    }
    const auto wait_store = instrumentation::build_s_wait_flat_store0(arch);
    const auto restore_exec =
        instrumentation::build_s_mov_b64(kRdna4ExecLo, *options.moi_exec_save_sgpr, arch);
    if (!wait_store || !restore_exec ||
        record_words.size() > static_cast<size_t>(std::numeric_limits<int16_t>::max()))
      return std::nullopt;
    record_words.push_back(*wait_store);
    const auto skip_record =
        instrumentation::build_s_cbranch_vccz(static_cast<int16_t>(record_words.size()), arch);
    if (!skip_record)
      return std::nullopt;
    words.push_back(*skip_record);
    words.insert(words.end(), record_words.begin(), record_words.end());
    words.push_back(*restore_exec);
  } else {
    ConSanMoiRecordEmitter record(words, *options.scratch_vgpr,
                                  static_cast<uint16_t>(*options.scratch_vgpr + 2u), arch);
    if ((derived_owner_vgpr &&
         !append_store_u32_vgpr(words, record_base + offsetof(ConSanMoiFenceRecord, owner_id),
                                *derived_owner_vgpr, *options.scratch_vgpr, arch)) ||
        !append_atomic_event_index_store(words,
                                         base + offsetof(ConSanMoiReportHeader, event_counter),
                                         record_base + offsetof(ConSanMoiFenceRecord, event_index),
                                         *options.scratch_vgpr, arch) ||
        !append_store_u32_literal(words, base + offsetof(ConSanMoiReportHeader, fence_record_count),
                                  record_capacity_or_count, *options.scratch_vgpr, arch) ||
        !record.materialize_address(record_base) ||
        !append_store_moi_report_dispatch_id_pair(
            record, options, offsetof(ConSanMoiFenceRecord, generation), arch,
            ConSanMoiLiteralDispatchIdPolicy::AnyArchitecture) ||
        !record.store_workgroup(offsetof(ConSanMoiFenceRecord, workgroup_x), workgroup_sources.x) ||
        !record.store_workgroup(offsetof(ConSanMoiFenceRecord, workgroup_y), workgroup_sources.y) ||
        !record.store_workgroup(offsetof(ConSanMoiFenceRecord, workgroup_z), workgroup_sources.z) ||
        (options.moi_owner_vgpr &&
         !record.store_vgpr(offsetof(ConSanMoiFenceRecord, owner_id), *options.moi_owner_vgpr)) ||
        (options.moi_persistent_owner_sgpr &&
         !record.store_sgpr(offsetof(ConSanMoiFenceRecord, owner_id),
                            *options.moi_persistent_owner_sgpr)) ||
        (options.moi_epoch_vgpr &&
         !record.store_vgpr(offsetof(ConSanMoiFenceRecord, epoch), *options.moi_epoch_vgpr)) ||
        (options.moi_persistent_epoch_sgpr &&
         !record.store_sgpr(offsetof(ConSanMoiFenceRecord, epoch),
                            *options.moi_persistent_epoch_sgpr)) ||
        !record.store_literal(offsetof(ConSanMoiFenceRecord, instruction_offset),
                              static_cast<uint32_t>(candidate.fence.text_offset)) ||
        !record.store_literal(offsetof(ConSanMoiFenceRecord, kind), static_cast<uint32_t>(kind)) ||
        !record.store_literal(offsetof(ConSanMoiFenceRecord, scope), *candidate.fence.raw_scope) ||
        !record.store_literal(offsetof(ConSanMoiFenceRecord, semantics),
                              candidate.communication_site.raw_th.value_or(0u)) ||
        !record.store_vgpr(offsetof(ConSanMoiFenceRecord, communication_token), address_vgpr) ||
        !record.store_vgpr(offsetof(ConSanMoiFenceRecord, communication_token) + sizeof(uint32_t),
                           static_cast<uint16_t>(address_vgpr + 1u))) {
      errors.emplace_back("ConSan MOI fence record patch could not encode record stores");
      return std::nullopt;
    }
  }
  if (!append_restore_moi_special_state(words, options, arch)) {
    errors.emplace_back("ConSan MOI fence record patch could not restore VCC/SCC");
    return std::nullopt;
  }
  if (scalar_spill)
    words.insert(words.end(), scalar_spill->restore_words.begin(),
                 scalar_spill->restore_words.end());
  if (spill)
    words.insert(words.end(), spill->restore_words.begin(), spill->restore_words.end());
  if (select_low_vgpr_bank)
    words.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
        0u, static_cast<uint8_t>(*vgpr_msb_mode), arch));
  words.insert(words.end(), displaced_tail_words.begin(), displaced_tail_words.end());
  if (call_return_sgpr) {
    words.push_back(build_s_setpc_b64(*call_return_sgpr, arch));
  } else if (!append_moi_direct_or_indirect_return(words, cave_text_offset, return_text_offset,
                                                   options, arch)) {
    errors.emplace_back("ConSan MOI fence record could not encode its return");
    return std::nullopt;
  }
  return words;
}
