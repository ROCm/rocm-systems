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
  const auto workgroup_sources = moi_persistent_or_descriptor_workgroup_sources(
      bytes, *candidate.barrier.kernel_descriptor_file_offset, options, arch, errors,
      candidate.uses_gfx1250_cluster_workgroup_id);
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
  const auto existing_label = sequence.make_label();
  const auto collision_label = sequence.make_label();
  const auto restore_label = sequence.make_label();
  const auto fast_return_label = sequence.make_label();
  const auto matched_window_label = sequence.make_label();
  const bool control_gfx1250_vgpr_bank = consan_arch_has_selectable_vgpr_bank(arch);
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
                                          bank_vgpr, value, *options.moi_owner_vgpr, arch)) {
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
  if (!append_moi_global_atomic_wait(words, arch))
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
  if (!sequence.emit_branch(existing_label, InstructionSequence::BranchKind::SccZero)) {
    errors.emplace_back("ConSan MOI sampled barrier failed at existing metadata branch");
    return std::nullopt;
  }

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
  const auto wait_store = instrumentation::build_s_wait_global_store0(arch);
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
  if (!append_moi_global_atomic_wait(words, arch))
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

  // The host model admits an exact pre-existing metadata record as an
  // idempotent publication. Shared barrier helpers can expose several
  // equivalent physical sequences for one logical transition, so the device
  // path must implement the same contract instead of treating every failed
  // empty-slot claim as lost coverage. A different or partially published
  // record remains a collision.
  if (!sequence.bind(existing_label)) {
    errors.emplace_back("ConSan MOI sampled barrier failed to bind existing metadata");
    return std::nullopt;
  }
  const auto restore_existing_exec =
      instrumentation::build_s_mov_b64(kRdna4ExecLo, temporary_exec, arch);
  const auto existing_descriptor =
      instrumentation::build_v_mov_b32_literal(expected, encoded.packed.descriptor, arch);
  const auto descriptor_equal =
      instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(expected), value, arch);
  if (!restore_existing_exec || !existing_descriptor || !descriptor_equal) {
    errors.emplace_back("ConSan MOI sampled barrier failed at existing metadata descriptor");
    return std::nullopt;
  }
  words.push_back(*restore_existing_exec);
  words.insert(words.end(), existing_descriptor->begin(), existing_descriptor->end());
  words.push_back(*descriptor_equal);
  if (!sequence.emit_branch(collision_label, InstructionSequence::BranchKind::VccZero))
    return std::nullopt;
  words.insert(words.end(), increment->begin(), increment->end());
  if (!append_sampled_indexed_address(words, first_metadata_address,
                                      sizeof(ConSanMoiSampledSyncMetadataPacked), bank_vgpr, base,
                                      value, arch)) {
    errors.emplace_back("ConSan MOI sampled barrier failed at existing metadata addressing");
    return std::nullopt;
  }
  const auto require_existing_vgpr = [&](uint32_t offset, uint16_t expected_vgpr) -> bool {
    if (!record.load(offset, value))
      return false;
    const auto equal =
        instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(expected_vgpr), value, arch);
    return equal && sequence.emit(equal) &&
           sequence.emit_branch(collision_label, InstructionSequence::BranchKind::VccZero);
  };
  const auto require_existing_literal = [&](uint32_t offset, uint32_t literal) -> bool {
    const auto materialize = instrumentation::build_v_mov_b32_literal(expected, literal, arch);
    return sequence.emit(materialize) && require_existing_vgpr(offset, expected);
  };
  if (!require_existing_literal(offsetof(ConSanMoiSampledSyncMetadataPacked, address), 0u) ||
      !require_existing_literal(offsetof(ConSanMoiSampledSyncMetadataPacked, address) + 4u, 0u) ||
      !require_existing_literal(offsetof(ConSanMoiSampledSyncMetadataPacked, byte_count), 0u) ||
      !require_existing_vgpr(offsetof(ConSanMoiSampledSyncMetadataPacked, epoch_before),
                             *options.moi_epoch_vgpr) ||
      !require_existing_vgpr(offsetof(ConSanMoiSampledSyncMetadataPacked, epoch_after),
                             next_epoch) ||
      !sequence.emit_branch(restore_label, InstructionSequence::BranchKind::Unconditional)) {
    errors.emplace_back("ConSan MOI sampled barrier failed at existing metadata comparison");
    return std::nullopt;
  }

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
                                          MoiResourcePlanningState &resource_state,
                                          std::span<const ConSanMoiCandidate> admitted,
                                          ConSanTransformArtifacts &result) {
  if (!options.moi_track_barriers || options.moi_engine != ConSanMoiEngine::Sampled)
    return;
  if (!options.moi_report_buffer_address || !consan_is_capability_arch(arch)) {
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
  access_by_offset.reserve(admitted.size());
  for (const ConSanMoiCandidate &candidate : admitted) {
    if (candidate.kind == ConSanLdsAccessKind::Read || candidate.kind == ConSanLdsAccessKind::Write)
      access_by_offset.try_emplace(candidate.anchor(), &candidate);
  }
  bool has_multi_range_window = false;
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.kind != ConSanPatchKind::InlineMoiSampledWatchpointStore &&
        patch.kind != ConSanPatchKind::TrampolineMoiSampledWatchpointStore)
      continue;
    const auto access = access_by_offset.find(patch.anchor_offset);
    if (access == access_by_offset.end())
      continue;
    const auto &ranges = access->second->ranges;
    if (ranges.empty())
      continue;
    selected_windows.push_back({&patch, access->second, patch.sampled_first_slot,
                                std::max(patch.sampled_window_bank_count, 1u)});
    has_multi_range_window |= ranges.size() > 1u;
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
  kernel_by_name.reserve(result.program_inventory.kernels().size());
  for (const ConSanKernelInfo &kernel : result.program_inventory.kernels())
    kernel_by_name.try_emplace(kernel.name, &kernel);
  std::unordered_map<std::string_view, const ConSanFunctionInfo *> function_by_name;
  function_by_name.reserve(result.program_inventory.functions().size());
  for (const ConSanFunctionInfo &function : result.program_inventory.functions())
    function_by_name.try_emplace(function.name, &function);

  std::vector<SampledBarrierCandidate> candidates;
  for (const ConSanSyncSequence &sequence : result.program_inventory.sync().sync_sequences) {
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
    if (!plan_admits_barrier_insertion(result.observation_plan,
                                       completing_sites.front().text_offset))
      continue;
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
      ConSanOptions owner_options = options;
      const std::array<uint64_t, 1> owner_descriptor = {owner.descriptor_file_offset};
      if (!apply_moi_persistent_vgpr_assignment(owner_options, owner_descriptor)) {
        owner_lowering_failure = "missing owner-local persistent assignment";
        break;
      }
      const auto owner_shift =
          owner_options.automatic_moi_private_epoch
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

  const std::span<const uint8_t> active_bytes = active_moi_bytes(bytes, result);
  AmdGpuCodeObject code_object(active_bytes.data(), active_bytes.size());
  CodeObjectPatcher patcher(code_object);
  const std::span<const uint8_t> old_text = patcher.text_bytes();
  if (old_text.empty()) {
    result.errors.emplace_back("ConSan MOI sampled barrier sync found no .text section");
    return;
  }

  struct Planned : MoiPlannedProbeResources {
    SampledBarrierCandidate candidate;
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
    size_t resource_plan_index = 0u;
    if (existing != barrier_plan_by_offset.end()) {
      resource_plan_index = existing->second;
      plan = result.resource_plans[resource_plan_index];
    } else {
      plan =
          plan_moi_resource_site(resource_state, options, ConSanResourceSiteKind::Barrier, index,
                                 site.text_offset, candidate.barrier.kernel_descriptor_file_offset,
                                 options.automatic_moi_private_epoch ? 9u : 7u);
      result.resource_plans.push_back(plan);
      resource_plan_index = result.resource_plans.size() - 1u;
      barrier_plan_by_offset.emplace(site.text_offset, resource_plan_index);
    }
    ConSanOptions candidate_options = options;
    if (!apply_moi_transient_sgpr_assignment(candidate_options,
                                             plan.owner_descriptor_file_offsets) ||
        !apply_moi_persistent_vgpr_assignment(candidate_options,
                                              plan.owner_descriptor_file_offsets)) {
      result.warnings.emplace_back(
          "ConSan MOI sampled barrier lost its owner-local persistent or transient assignment");
      continue;
    }
    const uint16_t scratch_count = candidate_options.automatic_moi_private_epoch ||
                                           moi_uses_scalar_persistent_state(candidate_options)
                                       ? 9u
                                       : 7u;
    if (plan.scratch_vgpr_count != scratch_count) {
      plan = plan_moi_resource_site(resource_state, candidate_options,
                                    ConSanResourceSiteKind::Barrier, index, site.text_offset,
                                    candidate.barrier.kernel_descriptor_file_offset, scratch_count);
      result.resource_plans[resource_plan_index] = plan;
    }
    const auto resources = resolve_moi_scratch_plan(plan, candidate_options, scratch_count);
    if (!resources)
      continue;
    candidate_options.scratch_vgpr = resources->base;
    std::optional<MoiPrivateEpochLayout> private_layout;
    if (candidate_options.automatic_moi_private_epoch) {
      private_layout = build_sampled_private_epoch_layout(result, *resources, candidate_options,
                                                          arch, result.warnings);
      if (!private_layout)
        continue;
    }
    auto probe = plan_moi_probe_resources(result, *resources, candidate_options, spill_managers,
                                          arch, std::move(private_layout),
                                          candidate_options.automatic_moi_record_replay_sgpr_spill);
    if (!probe)
      continue;
    note_moi_probe_private_requirements(private_requirements, *probe);
    note_descriptor_requirements(descriptor_requirements, probe->resources);
    selected.push_back({std::move(*probe), std::move(candidate)});
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
  constexpr uint64_t kIndirectIslandWords = kMoiRecordReplayIndirectIslandWords;
  std::vector<ConSanPatchInfo> patches;
  MoiLocalNopIslandAllocator indirect_allocator(old_text, result, arch,
                                                kMoiLocalIndirectIslandWords);
  const auto try_emit_direct_barrier = [&](size_t barrier_index, bool allow_local_island) -> bool {
    const Planned &planned = selected[barrier_index];
    const ConSanBarrierSite &site = planned.candidate.barrier.site;
    const uint64_t island_offset =
        original_text_size + barrier_index * kIndirectIslandWords * sizeof(uint32_t);
    const uint64_t cave_offset = new_text.size();
    ConSanOptions candidate_options = options;
    candidate_options.scratch_vgpr = planned.resources.base;
    if (planned.private_layout) {
      candidate_options.moi_record_replay_workgroup_private_offsets =
          planned.private_layout->record_replay_workgroup_offsets;
    }
    if (!apply_moi_transient_sgpr_assignment(candidate_options,
                                             planned.resources.owner_descriptor_file_offsets)) {
      result.errors.emplace_back(
          "ConSan MOI sampled barrier sync lost its transient scalar assignment");
      return false;
    }
    if (planned.private_layout || moi_uses_scalar_persistent_state(candidate_options)) {
      candidate_options.moi_owner_vgpr = static_cast<uint16_t>(planned.resources.base + 7u);
      candidate_options.moi_epoch_vgpr = static_cast<uint16_t>(planned.resources.base + 8u);
    } else if (!apply_moi_persistent_vgpr_assignment(
                   candidate_options, planned.resources.owner_descriptor_file_offsets)) {
      result.errors.emplace_back(
          "ConSan MOI sampled barrier sync lost its persistent VGPR assignment");
      return false;
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
      return false;
    }
    auto fwd = compute_sopp_branch_simm16(site.text_offset, cave_offset);
    if (!fwd) {
      const auto reserved_fwd = compute_sopp_branch_simm16(site.text_offset, island_offset);
      const auto indirect_state = moi_indirect_jump_sgprs(candidate_options);
      std::vector<uint32_t> island_words;
      const bool reserved_island_valid =
          reserved_fwd && indirect_state &&
          append_moi_scc_preserving_indirect_jump(
              island_words, island_offset, cave_offset, indirect_state->pc_sgpr,
              indirect_state->scc_save_sgpr, /*capture_scc=*/true, arch) &&
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
      } else if (allow_local_island) {
        const auto local_island = indirect_allocator.claim_reachable(site.text_offset);
        fwd = local_island ? compute_sopp_branch_simm16(site.text_offset, *local_island)
                           : std::nullopt;
        if (local_island && fwd) {
          if (!emit_moi_local_indirect_entry_island(
                  new_text, *local_island, cave_offset, site.text_offset, candidate_options,
                  planned.resources.owner_descriptor_file_offsets, arch, patches, result.errors,
                  "ConSan MOI sampled barrier sync patch")) {
            return false;
          }
        }
      }
    }
    if (!fwd) {
      result.warnings.emplace_back(
          "ConSan MOI sampled barrier sync skipped barrier at text offset " +
          std::to_string(site.text_offset) + ": no reachable entry island");
      return true;
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
      info.persistent_record_replay_workgroup_private_offsets =
          planned.private_layout->record_replay_workgroup_offsets;
      info.persistent_private_state_end = planned.private_layout->ephemeral_base;
    }
    info.required_private_segment_size = planned.required_private_bytes;
    info.owner_descriptor_file_offsets = planned.resources.owner_descriptor_file_offsets;
    if (planned.spill) {
      info.spilled_vgpr_count = planned.spill->vgpr_count;
      note_dynamic_stack_private_requirement(info, &*planned.spill);
    }
    patches.push_back(std::move(info));
    return true;
  };
  // A single nearby gfx1250 barrier needs no shared call router. Keeping that
  // case on the ordinary direct-cave path is also important for a shared
  // helper: B0-to-A0 translation clones the helper per kernel owner, whereas
  // an absolute call-return dispatcher only names one physical clone.
  const bool gfx1250_single_direct_barrier =
      arch == ROCJITSU_CODE_ARCH_CDNA5 && selected.size() == 1u &&
      compute_sopp_branch_simm16(selected.front().candidate.barrier.site.text_offset,
                                 new_text.size())
          .has_value();
  const bool has_stranded_barrier =
      (consan_uses_gfx9_cdna_encoding(arch) || arch == ROCJITSU_CODE_ARCH_RDNA4) &&
      std::ranges::any_of(std::views::iota(size_t{0}, selected.size()), [&](size_t index) {
        const uint64_t anchor = selected[index].candidate.barrier.site.text_offset;
        const uint64_t reserved_island =
            original_text_size + index * kIndirectIslandWords * sizeof(uint32_t);
        return !compute_sopp_branch_simm16(anchor, new_text.size()) &&
               !compute_sopp_branch_simm16(anchor, reserved_island);
      });
  const bool use_dense_barrier_relay =
      (arch == ROCJITSU_CODE_ARCH_CDNA5 && !gfx1250_single_direct_barrier) ||
      (arch == ROCJITSU_CODE_ARCH_RDNA4 && selected.size() >= 8u) || has_stranded_barrier;
  if (use_dense_barrier_relay) {
    constexpr uint64_t kDispatcherWordsPerSite = 20u;

    std::vector<consan_detail::MoiDenseRouteSite> route_sites;
    route_sites.reserve(selected.size());
    for (size_t index = 0; index < selected.size(); ++index) {
      const SampledBarrierCandidate &candidate = selected[index].candidate;
      route_sites.push_back({
          .owner = {candidate.sequence->in_kernel ? ConSanProgramContainerKind::Kernel
                                                  : ConSanProgramContainerKind::Function,
                    candidate.sequence->container_name},
          .anchor = candidate.barrier.site.text_offset,
          .source_index = index,
      });
    }
    // GFX1250 helpers can be cloned independently during B0-to-A0 translation.
    // Keep every caller identity in the non-negative scalar inline-constant
    // range so a clone-local two-word entry stub can carry it to the common
    // appended dispatcher without embedding a clone address.
    constexpr size_t kMaxGfx1250CloneLocalRoutes = 63u;
    const auto groups = consan_detail::partition_moi_dense_route_sites(
        route_sites, arch == ROCJITSU_CODE_ARCH_CDNA5 ? kMaxGfx1250CloneLocalRoutes : 0u);
    std::vector<std::pair<uint64_t, uint64_t>> claimed_hosts;
    std::vector<std::pair<uint64_t, uint64_t>> fixed_claims;
    fixed_claims.reserve(result.patches.size() + selected.size());
    for (const ConSanPatchInfo &patch : result.patches) {
      if (patch.original_size != 0u)
        fixed_claims.emplace_back(patch.anchor_offset, patch.anchor_offset + patch.original_size);
    }
    for (const Planned &planned : selected) {
      const ConSanBarrierSite &site = planned.candidate.barrier.site;
      if (site.size != 0u)
        fixed_claims.emplace_back(site.text_offset, site.text_offset + site.size);
    }
    const consan_detail::MoiOccupiedTextRanges occupied(std::move(fixed_claims));
    const auto overlaps_existing = [&](uint64_t begin, uint64_t end) {
      if (occupied.overlaps(begin, end) ||
          std::ranges::any_of(claimed_hosts, [&](const auto &range) {
            return range.first < end && range.second > begin;
          }))
        return true;
      return false;
    };

    for (const auto &group : groups) {
      const auto &indices = group.source_indices;
      if (indices.empty())
        continue;
      uint64_t container_begin = 0u;
      uint64_t container_end = 0u;
      std::string_view container_name = group.owner.name;
      if (group.owner.kind == ConSanProgramContainerKind::Kernel) {
        const ConSanKernelInfo *kernel =
            result.program_inventory.find_kernel_by_name(group.owner.name);
        if (kernel == nullptr)
          continue;
        container_begin = kernel->entry_text_offset;
        container_end = container_begin + kernel->code_size;
      } else {
        const ConSanFunctionInfo *function =
            result.program_inventory.find_function_by_name(group.owner.name);
        if (function == nullptr)
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
      ConSanOptions group_options = options;
      if (!apply_moi_transient_sgpr_assignment(group_options, group_owners)) {
        result.warnings.emplace_back("ConSan MOI sampled barrier sync skipped dense relay in " +
                                     std::string(container_name) +
                                     ": owners do not share one transient scalar assignment");
        continue;
      }
      const auto special_state = moi_special_state_sgprs(group_options);
      const auto indirect_state = moi_indirect_jump_sgprs(group_options);
      const bool spill_backed_router = group_options.automatic_moi_record_replay_sgpr_spill;
      if ((!spill_backed_router && (!group_options.moi_exec_save_sgpr || !special_state)) ||
          !indirect_state ||
          (spill_backed_router && (!group_options.moi_record_replay_dispatch_key_sgpr ||
                                   !group_options.moi_record_replay_call_return_sgpr))) {
        result.warnings.emplace_back("ConSan MOI sampled barrier sync skipped dense relay in " +
                                     std::string(container_name) +
                                     ": incomplete scalar router state");
        continue;
      }
      const uint16_t jump_pc_sgpr =
          spill_backed_router ? indirect_state->pc_sgpr : *group_options.moi_exec_save_sgpr;
      const uint16_t saved_scc_sgpr =
          spill_backed_router ? indirect_state->scc_save_sgpr : special_state->scc_save_sgpr;
      const uint16_t key_sgpr = spill_backed_router
                                    ? *group_options.moi_record_replay_dispatch_key_sgpr
                                    : static_cast<uint16_t>(*group_options.moi_exec_save_sgpr + 5u);
      const uint16_t call_return_sgpr =
          spill_backed_router ? *group_options.moi_record_replay_call_return_sgpr
                              : static_cast<uint16_t>(*group_options.moi_exec_save_sgpr + 6u);
      // GFX1250 uses an explicit key in a clone-local entry stub. Other
      // architectures retain the call-return-derived entry key when their
      // scalar assignment is spill backed.
      const bool use_gfx1250_clone_local_routes = arch == ROCJITSU_CODE_ARCH_CDNA5;
      const bool derive_key_at_entry = spill_backed_router && !use_gfx1250_clone_local_routes;
      const bool key_encodes_scc = derive_key_at_entry && key_sgpr == saved_scc_sgpr;
      const auto build_direct_call = [&](int16_t simm16) -> std::optional<uint32_t> {
        if (consan_arch_has_s_call_i64(arch))
          return instrumentation::build_s_call_i64(call_return_sgpr, simm16, arch);
        if (consan_arch_has_s_call_b64(arch) &&
            (arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_CDNA3 ||
             arch == ROCJITSU_CODE_ARCH_CDNA4))
          return build_s_call_b64(call_return_sgpr, simm16, arch);
        return std::nullopt;
      };
      const uint64_t island_words_count =
          use_gfx1250_clone_local_routes
              ? 2u * (indices.size() + 1u) + kMoiRecordReplayIndirectIslandWords
              : moi_dense_entry_island_words(derive_key_at_entry, key_encodes_scc);
      const uint64_t host_words_count = island_words_count + 1u;
      const uint64_t host_bytes = host_words_count * sizeof(uint32_t);
      const std::vector<MoiSgprRange> host_bootstrap_ranges = moi_relocatable_host_scalar_ranges(
          group_options, MoiRelocatableHostScalarState::BarrierRouter, arch);
      MoiDenseOwnerLivenessProof host_owner_proof =
          resource_state.prepare_dense_liveness_for_owners(group_owners, host_bootstrap_ranges);
      std::vector<uint64_t> anchors;
      anchors.reserve(indices.size());
      for (size_t index : indices)
        anchors.push_back(selected[index].candidate.barrier.site.text_offset);
      auto dense_host = find_moi_dense_relay_host(
          resource_state, old_text,
          {
              .owner_begin = container_begin,
              .owner_end = container_end,
              .host_word_count = static_cast<size_t>(host_words_count),
              .anchors = anchors,
              .preapplied_reserved_ranges = options.preapplied_reserved_ranges,
              .owner_proof = host_owner_proof,
              .bootstrap_ranges = host_bootstrap_ranges,
          },
          overlaps_existing);
      if (!dense_host) {
        result.warnings.emplace_back("ConSan MOI sampled barrier sync fell back from dense relay "
                                     "in " +
                                     std::string(container_name) +
                                     ": no liveness-safe relocatable host");
        for (size_t index : indices) {
          if (!try_emit_direct_barrier(index, /*allow_local_island=*/true))
            return;
        }
        continue;
      }
      const uint64_t host_offset = dense_host->host_offset;
      std::vector<uint32_t> displaced_words = std::move(dense_host->displaced_words);
      claimed_hosts.emplace_back(host_offset, host_offset + host_bytes);
      indirect_allocator.reserve(host_offset, static_cast<uint32_t>(host_words_count));

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
        if (planned.private_layout) {
          candidate_options.moi_record_replay_workgroup_private_offsets =
              planned.private_layout->record_replay_workgroup_offsets;
        }
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
      if (!append_moi_scc_preserving_indirect_jump(host_words, host_body_offset,
                                                   host_offset + host_bytes, jump_pc_sgpr,
                                                   saved_scc_sgpr, /*capture_scc=*/true, arch)) {
        result.errors.emplace_back(
            "ConSan MOI sampled dense barrier relay could not return from its host");
        return;
      }
      append_words_bytes(new_text, host_words);

      const uint64_t island_offset = host_offset + sizeof(uint32_t);
      std::optional<std::vector<uint32_t>> island_words;
      if (use_gfx1250_clone_local_routes) {
        std::vector<uint32_t> clone_local_words;
        clone_local_words.reserve(island_words_count);
        const uint64_t common_entry_offset =
            island_offset + 2u * (indices.size() + 1u) * sizeof(uint32_t);
        for (size_t route_id = 0u; route_id <= indices.size(); ++route_id) {
          clone_local_words.push_back(build_s_mov_b32(
              key_sgpr, scalar_positive_inline_u32(static_cast<uint16_t>(route_id)), arch));
          const uint64_t branch_offset =
              island_offset + clone_local_words.size() * sizeof(uint32_t);
          const auto branch_delta = compute_sopp_branch_simm16(branch_offset, common_entry_offset);
          if (!branch_delta) {
            result.errors.emplace_back(
                "ConSan MOI sampled dense barrier relay could not encode a clone-local route");
            return;
          }
          clone_local_words.push_back(build_s_branch(*branch_delta, arch));
        }
        if (!append_moi_scc_preserving_indirect_jump(
                clone_local_words, island_offset, dispatcher_offset, jump_pc_sgpr, saved_scc_sgpr,
                /*capture_scc=*/true, arch)) {
          result.errors.emplace_back(
              "ConSan MOI sampled dense barrier relay could not encode its common entry");
          return;
        }
        island_words = std::move(clone_local_words);
      } else {
        island_words = build_moi_dense_barrier_entry_island(island_offset, dispatcher_offset,
                                                            {.jump_pc_sgpr = jump_pc_sgpr,
                                                             .saved_scc_sgpr = saved_scc_sgpr,
                                                             .key_sgpr = key_sgpr,
                                                             .call_return_sgpr = call_return_sgpr},
                                                            derive_key_at_entry, arch);
      }
      if (!island_words || island_words->size() != island_words_count) {
        result.errors.emplace_back(
            "ConSan MOI sampled dense barrier relay could not encode its island");
        return;
      }
      std::memcpy(new_text.data() + island_offset, island_words->data(),
                  island_words->size() * sizeof(uint32_t));
      const auto host_call = build_direct_call(/*next instruction=*/0);
      if (!host_call) {
        result.errors.emplace_back(
            "ConSan MOI sampled dense barrier relay could not encode its host call");
        return;
      }
      std::memcpy(new_text.data() + host_offset, &*host_call, sizeof(uint32_t));

      std::vector<uint32_t> dispatcher_words;
      if (key_encodes_scc) {
        dispatcher_words.push_back(
            build_s_lshr_b32(jump_pc_sgpr, key_sgpr, scalar_positive_inline_u32(1), arch));
      }
      if (!derive_key_at_entry && arch != ROCJITSU_CODE_ARCH_CDNA5) {
        const auto subtract_pc =
            instrumentation::build_s_sub_u32(key_sgpr, call_return_sgpr, jump_pc_sgpr, arch);
        if (!subtract_pc) {
          result.errors.emplace_back(
              "ConSan MOI sampled dense barrier relay could not encode its dispatch key");
          return;
        }
        dispatcher_words.push_back(*subtract_pc);
        dispatcher_words.push_back(build_s_add_u32(key_sgpr, key_sgpr, /*literal=*/255u, arch));
        dispatcher_words.push_back(static_cast<uint32_t>(dispatcher_offset - island_offset));
      }
      const auto append_target = [&](size_t route_id, uint64_t caller_return,
                                     uint64_t target) -> bool {
        if (use_gfx1250_clone_local_routes) {
          const auto compare = instrumentation::build_s_cmp_eq_u32(
              key_sgpr, scalar_positive_inline_u32(static_cast<uint16_t>(route_id)), arch);
          if (!compare)
            return false;
          dispatcher_words.push_back(*compare);
        } else {
          const auto compare = instrumentation::build_s_cmp_eq_u32(
              key_encodes_scc ? jump_pc_sgpr : key_sgpr, /*literal=*/255u, arch);
          if (!compare)
            return false;
          dispatcher_words.push_back(*compare);
          dispatcher_words.push_back(static_cast<uint32_t>(caller_return - island_offset) /
                                     (key_encodes_scc ? 2u : 1u));
        }
        std::vector<uint32_t> matched_words;
        if (key_encodes_scc) {
          const uint16_t bitcmp_opcode =
              arch == ROCJITSU_CODE_ARCH_CDNA3 ? cdna3::kSBitcmp1B32Sopc : cdna4::kSBitcmp1B32Sopc;
          const auto normalize = instrumentation::build_s_cselect_b32(
              key_sgpr, scalar_positive_inline_u32(1), scalar_positive_inline_u32(0), arch);
          if (!normalize)
            return false;
          matched_words.push_back(
              build_sopc_encoding(arch, bitcmp_opcode, key_sgpr, scalar_positive_inline_u32(0)));
          matched_words.push_back(*normalize);
        }
        const uint64_t matched_offset =
            dispatcher_offset + (dispatcher_words.size() + 1u) * sizeof(uint32_t);
        if (!append_moi_scc_preserving_indirect_jump(matched_words, matched_offset, target,
                                                     jump_pc_sgpr, saved_scc_sgpr,
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
      if (!append_target(/*route_id=*/0u, /*caller_return=*/island_offset, host_body_offset))
        return;
      for (size_t body_index = 0u; body_index < bodies.size(); ++body_index) {
        const DenseBody &body = bodies[body_index];
        const ConSanBarrierSite &site = selected[body.selected_index].candidate.barrier.site;
        if (!append_target(body_index + 1u, site.text_offset + sizeof(uint32_t), body.offset))
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
      host_info.anchor_offset = host_offset;
      host_info.original_size = static_cast<uint32_t>(host_bytes);
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
      for (size_t body_index = 0u; body_index < bodies.size(); ++body_index) {
        const DenseBody &body = bodies[body_index];
        const Planned &planned = selected[body.selected_index];
        const ConSanBarrierSite &site = planned.candidate.barrier.site;
        const uint64_t call_target = use_gfx1250_clone_local_routes
                                         ? island_offset + 2u * (body_index + 1u) * sizeof(uint32_t)
                                         : island_offset;
        const auto call_delta = compute_sopp_branch_simm16(site.text_offset, call_target);
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
          info.persistent_record_replay_workgroup_private_offsets =
              planned.private_layout->record_replay_workgroup_offsets;
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
    result.replacement = std::move(patcher).emit();
    result.patches.insert(result.patches.end(), patches.begin(), patches.end());
    result.mark_modified();
    return;
  }
  for (size_t barrier_index = 0; barrier_index < selected.size(); ++barrier_index) {
    if (!try_emit_direct_barrier(barrier_index, /*allow_local_island=*/true))
      return;
  }
  if (!replace_consan_text(patcher, new_text, options, "MOI sampled barrier sync", result)) {
    return;
  }
  result.replacement = std::move(patcher).emit();
  result.patches.insert(result.patches.end(), patches.begin(), patches.end());
  result.mark_modified();
}

void try_apply_sampled_atomic_sync_patch(std::span<const uint8_t> bytes,
                                         const ConSanOptions &options, rj_code_arch_t arch,
                                         MoiResourcePlanningState &resource_state,
                                         ConSanTransformArtifacts &result) {
  if (!options.moi_track_atomics || options.moi_engine != ConSanMoiEngine::Sampled)
    return;
  if (!options.moi_report_buffer_address || !consan_is_capability_arch(arch)) {
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
    consan_detail::SampledAtomicSemantics semantics;
    uint64_t ordered_sequence_end_text_offset = 0;
    uint32_t selected_slot = 0;
    uint32_t bank_count = 1;
    bool deferred_acquire = false;
    std::optional<uint32_t> release_selected_slot;
  };
  std::vector<Candidate> candidates;
  const MoiSyncInventoryIndex sync_index(result);
  const bool allow_ordered_lds_atomic = consan_arch_supports_capability_form(
      result.program_inventory.arch(), ConSanCapabilityForm::OrderedLdsAtomic);
  const auto append = [&](std::string_view name, bool in_kernel, std::optional<uint64_t> descriptor,
                          const std::vector<ConSanAtomicSite> &sites) {
    for (const ConSanAtomicSite &site : sites) {
      const bool cas = consan_atomic_is_compare_exchange(site);
      if (cas && !site.returns_old_value.value_or(false)) {
        result.warnings.emplace_back("ConSan MOI sampled atomic sync skipped " + std::string(name) +
                                     ": compare-exchange-dynamic-outcome-unavailable");
        continue;
      }
      const ConSanAtomicSite normalized =
          normalize_atomic_record_site(site, allow_ordered_lds_atomic);
      if (!is_first_light_atomic_record_candidate(normalized, allow_ordered_lds_atomic))
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
      const ConSanSyncSequence *sequence =
          event == nullptr ? nullptr : sync_index.sequence(event->identity);
      if (descriptor && (event == nullptr || event->execution_owners.size() != 1u ||
                         event->execution_owners.front().descriptor_file_offset != *descriptor))
        descriptor.reset();
      candidates.push_back(
          {{(in_kernel ? "kernel:" : "function:") + std::string(name), normalized, descriptor,
            *event_kind, /*uses_gfx1250_cluster_workgroup_id=*/false, /*is_rmw=*/true,
            /*ordered_sequence_end_text_offset=*/0,
            atomic_scalar_clause_for_site(sync_index, name, in_kernel, site)},
           *semantics,
           sequence == nullptr ? site.text_offset + site.size : sequence->end_text_offset,
           0,
           1,
           false,
           std::nullopt});
    }
  };
  for (const ConSanKernelInfo &kernel : result.program_inventory.kernels()) {
    if (!is_rocclr_runtime_kernel_name(kernel.name))
      append(kernel.name, true, kernel.descriptor_file_offset, kernel.atomic_sites);
  }
  for (const ConSanFunctionInfo &function : result.program_inventory.functions())
    append(function.name, false, std::nullopt, function.atomic_sites);
  std::vector<AtomicRecordCandidate> ordinary_candidates;
  append_ordered_ordinary_atomic_candidates(result, sync_index, ordinary_candidates);
  for (AtomicRecordCandidate &atomic : ordinary_candidates) {
    const auto role = sampled_atomic_role(atomic.event_kind, /*is_rmw=*/false);
    const auto scope =
        atomic.site.raw_scope ? sampled_atomic_scope(*atomic.site.raw_scope) : std::nullopt;
    if (!role || !scope || atomic.site.width_bits == 0u || atomic.site.width_bits % 8u != 0u) {
      result.warnings.emplace_back(
          "ConSan MOI sampled atomic sync skipped an unqualified ordinary atomic sequence in " +
          atomic.container_name);
      continue;
    }
    consan_detail::SampledAtomicSemantics semantics{
        .role = *role,
        .scope = *scope,
        .outcome = ConSanMoiSampledSyncOutcome::NotApplicable,
        .byte_count = atomic.site.width_bits / 8u,
        .descriptor = 0u,
        .cas_failure_descriptor = std::nullopt,
    };
    const auto encoded = encode_consan_moi_sampled_sync_metadata({
        .address = 1u,
        .byte_count = semantics.byte_count,
        .kind = ConSanMoiSampledSyncKind::Atomic,
        .role = semantics.role,
        .scope = semantics.scope,
        .outcome = semantics.outcome,
    });
    if (encoded.classification != ConSanMoiSampledSyncClassification::Valid) {
      result.warnings.emplace_back(
          "ConSan MOI sampled atomic sync rejected ordinary atomic metadata in " +
          atomic.container_name);
      continue;
    }
    semantics.descriptor = encoded.packed.descriptor;
    const uint64_t sequence_end = atomic.ordered_sequence_end_text_offset;
    candidates.push_back({std::move(atomic), semantics, sequence_end, 0, 1, false, std::nullopt});
  }
  if (!options.test_kernel_name_filter.empty()) {
    std::erase_if(candidates, [&](const Candidate &candidate) {
      return candidate.atomic.container_name.find(options.test_kernel_name_filter) ==
             std::string::npos;
    });
  }
  if (!consan_detail::canonicalize_physical_site_aliases(
          candidates, result.errors, "ConSan MOI sampled atomic site",
          [](const Candidate &candidate) { return candidate.atomic.site.file_offset; },
          [](const Candidate &candidate) -> std::string_view {
            return candidate.atomic.container_name;
          },
          [](const Candidate &lhs, const Candidate &rhs) {
            return atomic_record_candidate_semantics_equal(lhs.atomic, rhs.atomic) &&
                   lhs.semantics == rhs.semantics &&
                   lhs.ordered_sequence_end_text_offset == rhs.ordered_sequence_end_text_offset;
          },
          [](Candidate &retained, const Candidate &) {
            // The hint is nested here, so the common direct-member invalidator
            // cannot be used.
            retained.atomic.kernel_descriptor_file_offset.reset();
          })) {
    return;
  }
  std::erase_if(candidates, [&](const Candidate &candidate) {
    return !plan_admits_atomic_insertion(result.observation_plan,
                                         candidate.atomic.site.text_offset);
  });
  std::sort(candidates.begin(), candidates.end(), [](const Candidate &lhs, const Candidate &rhs) {
    return lhs.atomic.site.text_offset < rhs.atomic.site.text_offset;
  });
  if (candidates.empty()) {
    result.warnings.emplace_back(
        "ConSan MOI sampled atomic sync found no qualified atomic sequence");
    return;
  }

  const std::span<const uint8_t> active_bytes = active_moi_bytes(bytes, result);
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
    const ConSanKernelInfo *canonical =
        result.program_inventory.find_kernel_by_name(active_kernel.name);
    if (canonical != nullptr)
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

  if (!resource_state.valid) {
    result.warnings.emplace_back("ConSan MOI sampled atomic sync lacks a complete kernel CFG");
    return;
  }

  struct Window {
    uint32_t slot = 0;
    uint32_t bank_count = 1;
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
      windows.push_back({patch.sampled_first_slot, std::max(patch.sampled_window_bank_count, 1u),
                         &patch, patch.sampled_access_kind, &context});
    }
  }
  if (has_multi_range_window) {
    result.warnings.emplace_back(
        "ConSan MOI sampled atomic sync currently attaches only the first logical range of a "
        "multi-range access");
  }

  struct Association {
    enum class Side : uint8_t { Release, Acquire };

    size_t candidate_index = 0;
    size_t window_index = 0;
    SampledCfgDistance distance;
    Side side = Side::Release;
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
      const uint32_t semantic_role = static_cast<uint32_t>(candidate.semantics.role);
      const bool release_side =
          (window.access_kind == ConSanLdsAccessKind::Write ||
           window.access_kind == ConSanLdsAccessKind::Atomic) &&
          (semantic_role & static_cast<uint32_t>(ConSanMoiSampledSyncRole::Release)) != 0u;
      const bool acquire_side =
          (window.access_kind == ConSanLdsAccessKind::Read ||
           window.access_kind == ConSanLdsAccessKind::Atomic) &&
          (semantic_role & static_cast<uint32_t>(ConSanMoiSampledSyncRole::Acquire)) != 0u;
      std::optional<SampledCfgDistance> distance;
      Association::Side side = Association::Side::Release;
      if (release_side) {
        distance =
            sampled_cfg_forward_distance(resource_state, *window.owner, window.patch->anchor_offset,
                                         candidate.atomic.site.text_offset);
      } else if (acquire_side) {
        side = Association::Side::Acquire;
        distance = sampled_cfg_forward_distance(resource_state, *window.owner,
                                                candidate.atomic.site.text_offset,
                                                window.patch->anchor_offset);
      }
      if (!distance)
        continue;
      Association current{candidate_index, window_index, *distance, side};
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
    const Candidate &candidate = candidates[candidate_index];
    for (const Association::Side missing_side :
         {Association::Side::Release, Association::Side::Acquire}) {
      if (std::ranges::any_of(proposed, [&](const Association &association) {
            return association.candidate_index == candidate_index &&
                   association.side == missing_side;
          }))
        continue;
      const uint32_t semantic_role = static_cast<uint32_t>(candidate.semantics.role);
      const bool role_has_side =
          (semantic_role & static_cast<uint32_t>(missing_side == Association::Side::Release
                                                     ? ConSanMoiSampledSyncRole::Release
                                                     : ConSanMoiSampledSyncRole::Acquire)) != 0u;
      if (!role_has_side)
        continue;
      std::optional<Association> best;
      for (size_t window_index = 0; window_index < windows.size(); ++window_index) {
        const Window &window = windows[window_index];
        if (!owner_executes_candidate(*window.owner, candidate))
          continue;
        const bool window_has_side = missing_side == Association::Side::Release
                                         ? window.access_kind == ConSanLdsAccessKind::Write ||
                                               window.access_kind == ConSanLdsAccessKind::Atomic
                                         : window.access_kind == ConSanLdsAccessKind::Read ||
                                               window.access_kind == ConSanLdsAccessKind::Atomic;
        if (!window_has_side)
          continue;
        const uint64_t text_distance =
            window.patch->anchor_offset < candidate.atomic.site.text_offset
                ? candidate.atomic.site.text_offset - window.patch->anchor_offset
                : window.patch->anchor_offset - candidate.atomic.site.text_offset;
        Association current{candidate_index,
                            window_index,
                            {std::numeric_limits<uint32_t>::max(), text_distance},
                            missing_side};
        if (!best || current.distance < best->distance)
          best = current;
      }
      if (best)
        proposed.push_back(*best);
    }
  }

  // One sampled window has one metadata slot. Multiple synchronization sites
  // must not race to claim that slot: the earliest dynamic publisher would
  // otherwise win even when a later, CFG-nearer acquire is the edge that
  // actually orders the payload. Prefer the strongest static association and
  // reject equal evidence rather than making execution order part of the
  // sanitizer's answer. Window indices include their recovered kernel-owner
  // context, so mutually exclusive shared-helper owners remain independent.
  std::vector<bool> rejected(proposed.size(), false);
  for (size_t i = 0; i < proposed.size(); ++i) {
    for (size_t j = i + 1u; j < proposed.size(); ++j) {
      if (proposed[i].window_index != proposed[j].window_index)
        continue;
      if (proposed[i].distance < proposed[j].distance)
        rejected[j] = true;
      else if (proposed[j].distance < proposed[i].distance)
        rejected[i] = true;
      else
        rejected[i] = rejected[j] = true;
    }
  }

  // Each directional half may order only one selected access window. An
  // acquire-release RMW may legitimately attach its release half to a
  // preceding write and its acquire half to a later read.
  for (size_t i = 0; i < proposed.size(); ++i) {
    for (size_t j = i + 1u; j < proposed.size(); ++j) {
      if (rejected[i] || rejected[j])
        continue;
      if (proposed[i].candidate_index != proposed[j].candidate_index ||
          proposed[i].side != proposed[j].side)
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
  // One physical atomic patch can carry both directional associations.
  const size_t static_patch_budget = std::min<size_t>(options.max_patches, candidates.size());
  associated.reserve(static_patch_budget);
  for (size_t candidate_index = 0;
       candidate_index < candidates.size() && associated.size() < static_patch_budget;
       ++candidate_index) {
    std::optional<size_t> release_association;
    std::optional<size_t> acquire_association;
    for (size_t i = 0; i < proposed.size(); ++i) {
      if (rejected[i] || proposed[i].candidate_index != candidate_index)
        continue;
      if (proposed[i].side == Association::Side::Release)
        release_association = i;
      else
        acquire_association = i;
    }
    const std::optional<size_t> primary =
        acquire_association ? acquire_association : release_association;
    if (!primary)
      continue;
    Candidate candidate = candidates[candidate_index];
    const Window &primary_window = windows[proposed[*primary].window_index];
    candidate.selected_slot = primary_window.slot;
    candidate.bank_count = primary_window.bank_count;
    candidate.deferred_acquire = acquire_association.has_value();
    if (candidate.deferred_acquire && release_association) {
      const Window &release_window = windows[proposed[*release_association].window_index];
      if (release_window.bank_count == candidate.bank_count)
        candidate.release_selected_slot = release_window.slot;
      else
        result.warnings.emplace_back(
            "ConSan MOI sampled acquire-release edge has incompatible access-window banks");
    }
    associated.push_back(std::move(candidate));
  }
  std::ranges::sort(associated, {}, &Candidate::selected_slot);
  candidates = std::move(associated);
  if (candidates.empty()) {
    result.warnings.emplace_back(
        "ConSan MOI sampled atomic sync found no unambiguous owner/CFG-associated edge");
    return;
  }

  struct Planned : MoiPlannedProbeResources {
    Candidate candidate;
    ConSanMoiAtomicAddressPlan address_plan;
    uint64_t physical_anchor_offset = 0;
    uint32_t relocated_guest_size = 0;
    std::vector<uint32_t> trailing_guest_words;
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
    uint32_t relocated_guest_size = site.size;
    std::vector<uint32_t> trailing_guest_words;
    if (candidate.deferred_acquire && composed_access_patch == result.patches.end() &&
        candidate.ordered_sequence_end_text_offset > site.text_offset + site.size) {
      const uint64_t sequence_end = candidate.ordered_sequence_end_text_offset;
      const BasicBlock *atomic_block =
          block_for_offset(resource_state.block_index, site.text_offset);
      const BasicBlock *sequence_end_block =
          block_for_offset(resource_state.block_index, sequence_end - sizeof(uint32_t));
      const uint64_t sequence_size = sequence_end - site.text_offset;
      if (atomic_block == nullptr || sequence_end_block != atomic_block ||
          sequence_size > std::numeric_limits<uint32_t>::max() ||
          sequence_size % sizeof(uint32_t) != 0u || sequence_end > old_text.size()) {
        result.warnings.emplace_back(
            "ConSan MOI sampled atomic sync could not relocate its bounded acquire suffix");
        continue;
      }
      relocated_guest_size = static_cast<uint32_t>(sequence_size);
      trailing_guest_words.resize((relocated_guest_size - site.size) / sizeof(uint32_t));
      std::memcpy(trailing_guest_words.data(), old_text.data() + site.text_offset + site.size,
                  trailing_guest_words.size() * sizeof(uint32_t));
    }
    if (physical_anchor_offset > old_text.size() ||
        relocated_guest_size > old_text.size() - physical_anchor_offset)
      continue;
    const bool overlap = std::any_of(
        result.patches.begin(), result.patches.end(), [&](const ConSanPatchInfo &patch) {
          if (composed_access_patch != result.patches.end() && &patch == &*composed_access_patch)
            return false;
          const uint64_t patch_end =
              patch.anchor_offset + std::max<uint32_t>(patch.original_size, 1u);
          return site.text_offset < patch_end &&
                 patch.anchor_offset < site.text_offset + relocated_guest_size;
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
    ConSanOptions candidate_options = options;
    if (!apply_moi_transient_sgpr_assignment(candidate_options,
                                             resources->owner_descriptor_file_offsets)) {
      result.warnings.emplace_back(
          "ConSan MOI sampled atomic sync lost its owner-local transient scalar assignment");
      continue;
    }
    candidate_options.scratch_vgpr = resources->base;
    const bool allow_post_guest_spill_operand_overlap =
        resources->source == ConSanRegisterAllocationSource::SpillRequired &&
        sampled_atomic_guest_preserves_address(site, candidate.atomic.is_rmw);
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
      private_layout = build_sampled_private_epoch_layout(result, *resources, candidate_options,
                                                          arch, result.warnings);
      if (!private_layout)
        continue;
    }
    auto probe = plan_moi_probe_resources(result, *resources, candidate_options, spill_managers,
                                          arch, std::move(private_layout),
                                          candidate_options.automatic_moi_record_replay_sgpr_spill);
    if (!probe)
      continue;
    note_moi_probe_private_requirements(private_requirements, *probe);
    note_descriptor_requirements(descriptor_requirements, probe->resources);
    selected.push_back({std::move(*probe), std::move(candidate), address_plan,
                        physical_anchor_offset, relocated_guest_size,
                        std::move(trailing_guest_words)});
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
  AmdGpuCodeObject original_code_object(bytes.data(), bytes.size());
  const uint64_t original_text_size = original_code_object.text_sections().size() == 1
                                          ? original_code_object.text_sections().front()->size()
                                          : 0u;
  const uint64_t reserved_barrier_island_count =
      sampled_reserved_barrier_island_count(result, options);
  const uint64_t reserved_atomic_island_count =
      sampled_reserved_atomic_island_count(result, options);
  const uint64_t reserved_atomic_island_begin =
      original_text_size +
      reserved_barrier_island_count * kMoiRecordReplayIndirectIslandWords * sizeof(uint32_t);
  std::vector<ConSanPreappliedReservedRange> local_island_reserved_ranges =
      options.preapplied_reserved_ranges;
  const uint64_t reserved_sync_island_bytes =
      reserved_barrier_island_count * kMoiRecordReplayIndirectIslandWords * sizeof(uint32_t) +
      reserved_atomic_island_count * kMoiLocalIndirectIslandWords * sizeof(uint32_t);
  if (reserved_sync_island_bytes != 0u) {
    local_island_reserved_ranges.push_back(
        {.text_offset = original_text_size,
         .size = static_cast<uint32_t>(reserved_sync_island_bytes)});
  }
  MoiLocalNopIslandAllocator indirect_allocator(
      old_text, result, arch, kMoiLocalIndirectIslandWords, local_island_reserved_ranges);
  std::vector<ConSanPatchInfo> patches;
  for (const Planned &planned : selected) {
    if (!neutralize_atomic_scalar_clause(new_text, planned.candidate.atomic, arch, patches,
                                         result.errors))
      return;
  }
  std::vector<bool> routed_by_dense_relay(selected.size(), false);

  const auto build_atomic_cave =
      [&](const Planned &planned, uint64_t cave_offset, uint32_t &guest_instruction_offset,
          const ConSanOptions *resolved_options = nullptr, bool preserve_guest_at_anchor = false,
          std::span<const uint32_t> leading_guest_words = {})
      -> std::optional<std::vector<uint32_t>> {
    ConSanOptions candidate_options = resolved_options ? *resolved_options : options;
    candidate_options.scratch_vgpr = planned.resources.base;
    if (planned.private_layout) {
      // Sampled probes run after arbitrary guest code. Use the tuple captured
      // by the automatic private-state prologue instead of descriptor SGPRs
      // whose ABI lifetime ended at kernel entry.
      candidate_options.moi_record_replay_workgroup_private_offsets =
          planned.private_layout->record_replay_workgroup_offsets;
    }
    if (resolved_options == nullptr &&
        !apply_moi_transient_sgpr_assignment(candidate_options,
                                             planned.resources.owner_descriptor_file_offsets)) {
      result.errors.emplace_back(
          "ConSan MOI sampled atomic sync lost its transient scalar assignment");
      return std::nullopt;
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
      return std::nullopt;
    }
    const uint64_t return_text_offset =
        preserve_guest_at_anchor ? planned.physical_anchor_offset
                                 : planned.physical_anchor_offset + planned.relocated_guest_size;
    return planned.candidate.deferred_acquire
               ? build_sampled_pending_acquire_cave_words(
                     bytes, planned.candidate.atomic, planned.address_plan, candidate_options,
                     planned.spill ? &*planned.spill : nullptr,
                     planned.scalar_spill ? &*planned.scalar_spill : nullptr,
                     planned.private_layout ? &*planned.private_layout : nullptr, arch,
                     planned.candidate.selected_slot, planned.candidate.bank_count,
                     planned.candidate.release_selected_slot, layout, cave_offset,
                     return_text_offset, result.errors, &guest_instruction_offset,
                     planned.trailing_guest_words)
               : build_sampled_atomic_sync_cave_words(
                     bytes, planned.candidate.atomic, planned.address_plan, candidate_options,
                     planned.spill ? &*planned.spill : nullptr,
                     planned.scalar_spill ? &*planned.scalar_spill : nullptr,
                     planned.private_layout ? &*planned.private_layout : nullptr, arch,
                     planned.candidate.selected_slot, planned.candidate.bank_count, layout,
                     cave_offset, return_text_offset, result.errors, &guest_instruction_offset,
                     planned.trailing_guest_words, preserve_guest_at_anchor, leading_guest_words);
  };
  const auto append_atomic_patch =
      [&](const Planned &planned, uint64_t cave_offset, std::span<const uint32_t> cave_words,
          uint32_t guest_instruction_offset, bool guest_preserved_at_anchor = false) {
        const ConSanAtomicSite &site = planned.candidate.atomic.site;
        ConSanPatchInfo info;
        info.kind = ConSanPatchKind::TrampolineMoiSampledSyncMetadata;
        info.anchor_offset = site.text_offset;
        info.trampoline_offset = cave_offset;
        info.original_size = guest_preserved_at_anchor ? 0u : planned.relocated_guest_size;
        info.trampoline_size = static_cast<uint32_t>(cave_words.size() * sizeof(uint32_t));
        info.relocated_guest_instruction_offset =
            guest_preserved_at_anchor ? site.text_offset : cave_offset + guest_instruction_offset;
        info.sampled_access_kind = ConSanLdsAccessKind::Atomic;
        info.sampled_first_slot = planned.candidate.selected_slot;
        info.sampled_window_bank_count = planned.candidate.bank_count;
        info.scratch_vgpr = planned.resources.base;
        info.owner_descriptor_file_offsets = planned.resources.owner_descriptor_file_offsets;
        if (planned.private_layout) {
          info.persistent_epoch_private_offset = planned.private_layout->epoch_offset;
          info.persistent_owner_private_offset = planned.private_layout->owner_offset;
          info.persistent_record_replay_workgroup_private_offsets =
              planned.private_layout->record_replay_workgroup_offsets;
          info.persistent_private_state_end = planned.private_layout->ephemeral_base;
        }
        info.required_private_segment_size = planned.required_private_bytes;
        if (planned.spill) {
          info.spilled_vgpr_count = planned.spill->vgpr_count;
          note_dynamic_stack_private_requirement(info, &*planned.spill);
        }
        patches.push_back(std::move(info));
      };

  // An ordered atomic can sit near the front of a large generated kernel,
  // beyond SOPP reach of both its pristine-end relay and every appended
  // body. CDNA atomic instructions have enough anchor space for a scalar
  // call, so share the same liveness-proven, call-return-keyed relay model as
  // far barriers instead of silently dropping the causal edge.
  if (consan_uses_gfx9_cdna_encoding(arch)) {
    std::vector<consan_detail::MoiDenseRouteSite> route_sites;
    for (size_t index = 0u; index < selected.size(); ++index) {
      const Planned &planned = selected[index];
      const uint64_t anchor = planned.physical_anchor_offset;
      const uint64_t reserved_island =
          reserved_atomic_island_begin + index * kMoiLocalIndirectIslandWords * sizeof(uint32_t);
      const bool reserved_reachable =
          index < reserved_atomic_island_count &&
          compute_sopp_branch_simm16(anchor, reserved_island).has_value();
      const bool stranded = !compute_sopp_branch_simm16(anchor, new_text.size()) &&
                            !reserved_reachable && !indirect_allocator.has_reachable(anchor);
      if (stranded && anchor == planned.candidate.atomic.site.text_offset &&
          planned.relocated_guest_size >= sizeof(uint32_t)) {
        const std::string_view encoded_owner = planned.candidate.atomic.container_name;
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
    const auto groups = consan_detail::partition_moi_dense_route_sites(route_sites, 0u);

    std::vector<std::pair<uint64_t, uint64_t>> fixed_claims;
    fixed_claims.reserve(result.patches.size() + selected.size());
    for (const ConSanPatchInfo &patch : result.patches) {
      if (patch.original_size != 0u)
        fixed_claims.emplace_back(patch.anchor_offset, patch.anchor_offset + patch.original_size);
    }
    for (const Planned &planned : selected) {
      const ConSanAtomicSite &site = planned.candidate.atomic.site;
      if (planned.relocated_guest_size != 0u)
        fixed_claims.emplace_back(site.text_offset,
                                  site.text_offset + planned.relocated_guest_size);
    }
    const consan_detail::MoiOccupiedTextRanges occupied(std::move(fixed_claims));
    std::vector<std::pair<uint64_t, uint64_t>> claimed_hosts;
    const auto overlaps_existing = [&](uint64_t begin, uint64_t end) {
      return occupied.overlaps(begin, end) ||
             std::ranges::any_of(claimed_hosts, [&](const auto &range) {
               return range.first < end && begin < range.second;
             });
    };

    for (const auto &group : groups) {
      const auto &indices = group.source_indices;
      const bool in_kernel = group.owner.kind == ConSanProgramContainerKind::Kernel;
      const std::string_view owner_name = group.owner.name;
      const std::string container =
          std::string(in_kernel ? "kernel:" : "function:") + std::string(owner_name);
      uint64_t owner_begin = 0u;
      uint64_t owner_end = 0u;
      if (in_kernel) {
        const ConSanKernelInfo *kernel = result.program_inventory.find_kernel_by_name(owner_name);
        if (kernel == nullptr)
          continue;
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
      std::vector<uint64_t> group_owners;
      std::vector<uint64_t> anchors;
      for (size_t index : indices) {
        const Planned &planned = selected[index];
        group_owners.insert(group_owners.end(),
                            planned.resources.owner_descriptor_file_offsets.begin(),
                            planned.resources.owner_descriptor_file_offsets.end());
        anchors.push_back(planned.physical_anchor_offset);
      }
      std::ranges::sort(group_owners);
      group_owners.erase(std::ranges::unique(group_owners).begin(), group_owners.end());
      ConSanOptions group_options = options;
      if (!apply_moi_transient_sgpr_assignment(group_options, group_owners)) {
        result.warnings.emplace_back("ConSan MOI sampled atomic sync skipped dense relay in " +
                                     container +
                                     ": owners do not share one transient scalar assignment");
        continue;
      }
      const bool spill_backed_router = group_options.automatic_moi_record_replay_sgpr_spill;
      auto special_state = moi_special_state_sgprs(group_options);
      auto indirect_state = moi_indirect_jump_sgprs(group_options);
      if ((!spill_backed_router && (!group_options.moi_exec_save_sgpr || !special_state)) ||
          !indirect_state ||
          (spill_backed_router && (!group_options.moi_record_replay_dispatch_key_sgpr ||
                                   !group_options.moi_record_replay_call_return_sgpr))) {
        result.warnings.emplace_back("ConSan MOI sampled atomic sync skipped dense relay in " +
                                     container + ": incomplete scalar router state");
        continue;
      }
      // Component-local scalar spilling qualifies its bootstrap registers at
      // the access and barrier sites known during the first resource pass.
      // Atomic associations are selected later, after causal windows exist.
      // Re-qualify the live-through router tuple at those actual call anchors:
      // s_call overwrites the return pair before the body can back up borrowed
      // guest SGPRs, and the entry island similarly consumes its PC, SCC, and
      // key temporaries before entering the spill-protected body.
      const uint16_t ordinary_sgpr_limit = moi_ordinary_sgpr_limit(arch);
      const std::array complete_scalar_range{
          MoiSgprRange{.base = 0u, .width = ordinary_sgpr_limit}};
      MoiDenseOwnerLivenessProof host_owner_proof =
          resource_state.prepare_dense_liveness_for_owners(group_owners, complete_scalar_range);
      std::vector<consan_detail::ScalarOwnerContextSummary> scalar_summaries;
      scalar_summaries.reserve(host_owner_proof.contexts.size());
      for (const MoiKernelResourceContext *context : host_owner_proof.contexts) {
        if (context != nullptr)
          scalar_summaries.push_back(context->scalar_summary());
      }
      const auto overlaps_reserved_scalar_state = [&](uint16_t base, uint16_t width) {
        const auto overlaps_optional = [&](std::optional<uint16_t> reg, uint16_t reg_width = 1u) {
          return reg && range_overlaps(base, width, *reg, reg_width);
        };
        bool overlaps = persistent_sgpr_range_overlaps_reserved_ordinary_range(base, width, arch) ||
                        overlaps_optional(group_options.moi_owner_sgpr) ||
                        overlaps_optional(group_options.moi_persistent_owner_sgpr) ||
                        overlaps_optional(group_options.moi_persistent_epoch_sgpr) ||
                        overlaps_optional(group_options.moi_persistent_workgroup_key_sgpr) ||
                        overlaps_optional(group_options.moi_dispatch_id_sgpr,
                                          moi_dispatch_state_sgpr_count(group_options));
        for (const std::optional<uint16_t> reg :
             {group_options.moi_record_replay_workgroup_sgprs.x,
              group_options.moi_record_replay_workgroup_sgprs.y,
              group_options.moi_record_replay_workgroup_sgprs.z,
              group_options.moi_record_replay_workgroup_sgprs.cluster_workgroup_id}) {
          overlaps |= overlaps_optional(reg);
        }
        if (group_options.delay_mode == ConSanDelayMode::SleepVar)
          overlaps |= range_overlaps(base, width, group_options.delay_var_ssrc, 1u);
        return overlaps;
      };
      const auto dead_at_atomic_anchors = [&](uint16_t base, uint16_t width) {
        if (!host_owner_proof.complete || host_owner_proof.contexts.empty() || width == 0u ||
            static_cast<uint32_t>(base) + width > ordinary_sgpr_limit ||
            overlaps_reserved_scalar_state(base, width)) {
          return false;
        }
        const RegisterRef range{RegClass::SGPR, base, static_cast<uint8_t>(width)};
        for (size_t index : indices) {
          const uint64_t anchor = selected[index].physical_anchor_offset;
          BasicBlock *block = block_for_offset(resource_state.block_index, anchor);
          const Instruction *instruction = instruction_at(block, anchor);
          if (block == nullptr || instruction == nullptr)
            return false;
          bool saw_owner = false;
          for (const MoiKernelResourceContext *context : host_owner_proof.contexts) {
            if (context == nullptr || !context->block_set.contains(block))
              continue;
            saw_owner = true;
            if (context->liveness_is_restricted || context->has_indirect_sgpr_access ||
                static_cast<uint32_t>(base) + width > context->current_sgpr_count ||
                context->liveness->any_live_before(*instruction, range)) {
              return false;
            }
            const InstDefUse def_use(*instruction);
            if (def_use.defs.contains(range) || def_use.uses.contains(range))
              return false;
            // Some generated helpers have a decoded join whose local CFG
            // edge is intentionally absent from the owner scope. Liveness is
            // then conservative about the disconnected block as a whole, but
            // cannot carry a value across the missing edge. Reject a tuple
            // when the text-order continuation consumes one of its registers
            // before redefining it. This is a safety supplement only: normal
            // CFG liveness above remains authoritative across non-lexical
            // successors.
            std::vector<const Instruction *> lexical_tail;
            for (BasicBlock *owner_block : context->scope.blocks) {
              if (owner_block == nullptr || owner_block->end_offset() <= anchor)
                continue;
              for (const Instruction &tail_instruction : owner_block->instructions()) {
                if (tail_instruction.src_loc() > anchor)
                  lexical_tail.push_back(&tail_instruction);
              }
            }
            std::ranges::sort(lexical_tail, {}, &Instruction::src_loc);
            std::vector<bool> redefined(width, false);
            for (const Instruction *tail_instruction : lexical_tail) {
              const InstDefUse tail_def_use(*tail_instruction);
              for (uint16_t lane = 0u; lane < width; ++lane) {
                const RegisterRef scalar{RegClass::SGPR, static_cast<uint16_t>(base + lane), 1u};
                if (!redefined[lane] && tail_def_use.uses.contains(scalar))
                  return false;
                if (tail_def_use.defs.contains(scalar))
                  redefined[lane] = true;
              }
              if (std::ranges::all_of(redefined, std::identity{}))
                break;
            }
          }
          if (!saw_owner)
            return false;
        }
        return true;
      };
      const auto tuple_is_safe = [&](uint16_t pc, uint16_t scc, uint16_t key,
                                     uint16_t call_return) {
        const std::array ranges{MoiSgprRange{pc, 2u}, MoiSgprRange{scc, 1u}, MoiSgprRange{key, 1u},
                                MoiSgprRange{call_return, 2u}};
        return dead_at_atomic_anchors(pc, 2u) && dead_at_atomic_anchors(scc, 1u) &&
               dead_at_atomic_anchors(key, 1u) && dead_at_atomic_anchors(call_return, 2u) &&
               !range_overlaps(pc, 2u, scc, 1u) && !range_overlaps(pc, 2u, key, 1u) &&
               (pc == call_return || !range_overlaps(pc, 2u, call_return, 2u)) &&
               !range_overlaps(scc, 1u, key, 1u) && !range_overlaps(scc, 1u, call_return, 2u) &&
               !range_overlaps(key, 1u, call_return, 2u) &&
               !moi_scalar_contexts_conflict_with_physical_vcc(scalar_summaries, ranges, arch);
      };

      uint16_t jump_pc_sgpr =
          spill_backed_router ? indirect_state->pc_sgpr : *group_options.moi_exec_save_sgpr;
      uint16_t saved_scc_sgpr =
          spill_backed_router ? indirect_state->scc_save_sgpr : special_state->scc_save_sgpr;
      uint16_t key_sgpr = spill_backed_router
                              ? *group_options.moi_record_replay_dispatch_key_sgpr
                              : static_cast<uint16_t>(*group_options.moi_exec_save_sgpr + 5u);
      uint16_t call_return_sgpr =
          spill_backed_router ? *group_options.moi_record_replay_call_return_sgpr
                              : static_cast<uint16_t>(*group_options.moi_exec_save_sgpr + 6u);
      if (spill_backed_router &&
          !tuple_is_safe(jump_pc_sgpr, saved_scc_sgpr, key_sgpr, call_return_sgpr)) {
        bool found = false;
        if (spill_backed_router) {
          for (uint32_t pair = 0u; !found && pair + 2u <= ordinary_sgpr_limit; pair += 2u) {
            const uint16_t candidate_pair = static_cast<uint16_t>(pair);
            if (!dead_at_atomic_anchors(candidate_pair, 2u))
              continue;
            for (uint32_t scc = 0u; !found && scc < ordinary_sgpr_limit; ++scc) {
              const uint16_t candidate_scc = static_cast<uint16_t>(scc);
              if (range_overlaps(candidate_scc, 1u, candidate_pair, 2u) ||
                  !dead_at_atomic_anchors(candidate_scc, 1u)) {
                continue;
              }
              for (uint32_t key = 0u; key < ordinary_sgpr_limit; ++key) {
                const uint16_t candidate_key = static_cast<uint16_t>(key);
                if (range_overlaps(candidate_key, 1u, candidate_pair, 2u) ||
                    candidate_key == candidate_scc ||
                    !tuple_is_safe(candidate_pair, candidate_scc, candidate_key, candidate_pair)) {
                  continue;
                }
                jump_pc_sgpr = call_return_sgpr = candidate_pair;
                saved_scc_sgpr = candidate_scc;
                key_sgpr = candidate_key;
                group_options.moi_inline_indirect_pc_sgpr = jump_pc_sgpr;
                group_options.moi_inline_indirect_scc_sgpr = saved_scc_sgpr;
                group_options.moi_record_replay_dispatch_key_sgpr = key_sgpr;
                group_options.moi_record_replay_call_return_sgpr = call_return_sgpr;
                special_state = moi_special_state_sgprs(group_options);
                indirect_state = moi_indirect_jump_sgprs(group_options);
                found = special_state.has_value() && indirect_state.has_value();
                break;
              }
            }
          }
        }
        if (!found) {
          result.warnings.emplace_back("ConSan MOI sampled atomic sync skipped dense relay in " +
                                       container + ": no call-anchor-safe scalar router state");
          continue;
        }
      }
      const bool derive_key_at_entry = spill_backed_router;
      const bool key_encodes_scc = derive_key_at_entry && key_sgpr == saved_scc_sgpr;
      const uint64_t island_words_count =
          moi_dense_entry_island_words(derive_key_at_entry, key_encodes_scc);
      const uint64_t host_words_count = island_words_count + 1u;
      const uint64_t host_bytes = host_words_count * sizeof(uint32_t);
      const std::vector<MoiSgprRange> host_bootstrap_ranges = moi_relocatable_host_scalar_ranges(
          group_options, MoiRelocatableHostScalarState::BarrierRouter, arch);
      auto dense_host = find_moi_dense_relay_host(
          resource_state, old_text,
          {
              .owner_begin = owner_begin,
              .owner_end = owner_end,
              .host_word_count = static_cast<size_t>(host_words_count),
              .anchors = anchors,
              .preapplied_reserved_ranges = options.preapplied_reserved_ranges,
              .owner_proof = host_owner_proof,
              .bootstrap_ranges = host_bootstrap_ranges,
          },
          overlaps_existing);
      if (!dense_host) {
        result.warnings.emplace_back("ConSan MOI sampled atomic sync skipped dense relay in " +
                                     container + ": no liveness-safe relocatable host");
        continue;
      }
      const uint64_t host_offset = dense_host->host_offset;
      const uint64_t island_offset = host_offset + sizeof(uint32_t);
      claimed_hosts.emplace_back(host_offset, host_offset + host_bytes);

      constexpr uint64_t kDispatcherWordsPerSite = 20u;
      const uint64_t dispatcher_offset = new_text.size();
      const uint64_t dispatcher_reservation =
          (indices.size() * kDispatcherWordsPerSite + 12u) * sizeof(uint32_t);
      while (new_text.size() < dispatcher_offset + dispatcher_reservation)
        append_word_bytes(new_text, build_s_nop(0, arch));
      struct DenseBody {
        size_t selected_index = 0u;
        uint64_t offset = 0u;
        std::vector<uint32_t> words;
        uint32_t guest_instruction_offset = 0u;
      };
      std::vector<DenseBody> bodies;
      for (size_t index : indices) {
        uint32_t guest_instruction_offset = 0u;
        const uint64_t body_offset = new_text.size();
        auto body_words = build_atomic_cave(selected[index], body_offset, guest_instruction_offset,
                                            &group_options);
        if (!body_words)
          return;
        append_words_bytes(new_text, *body_words);
        bodies.push_back({index, body_offset, std::move(*body_words), guest_instruction_offset});
      }
      const uint64_t host_body_offset = new_text.size();
      std::vector<uint32_t> host_words = std::move(dense_host->displaced_words);
      if (!append_moi_scc_preserving_indirect_jump(host_words, host_body_offset,
                                                   host_offset + host_bytes, jump_pc_sgpr,
                                                   saved_scc_sgpr, /*capture_scc=*/true, arch)) {
        result.errors.emplace_back(
            "ConSan MOI sampled dense atomic relay could not return from its host");
        return;
      }
      append_words_bytes(new_text, host_words);

      auto island_words =
          build_moi_dense_barrier_entry_island(island_offset, dispatcher_offset,
                                               {.jump_pc_sgpr = jump_pc_sgpr,
                                                .saved_scc_sgpr = saved_scc_sgpr,
                                                .key_sgpr = key_sgpr,
                                                .call_return_sgpr = call_return_sgpr},
                                               derive_key_at_entry, arch);
      if (!island_words || island_words->size() != island_words_count) {
        result.errors.emplace_back(
            "ConSan MOI sampled dense atomic relay could not encode its island");
        return;
      }
      std::memcpy(new_text.data() + island_offset, island_words->data(),
                  island_words->size() * sizeof(uint32_t));
      const auto host_call = build_s_call_b64(call_return_sgpr, /*next instruction=*/0, arch);
      std::memcpy(new_text.data() + host_offset, &host_call, sizeof(uint32_t));

      std::vector<uint32_t> dispatcher_words;
      if (key_encodes_scc) {
        dispatcher_words.push_back(
            build_s_lshr_b32(jump_pc_sgpr, key_sgpr, scalar_positive_inline_u32(1), arch));
      }
      if (!derive_key_at_entry) {
        const auto subtract_pc =
            instrumentation::build_s_sub_u32(key_sgpr, call_return_sgpr, jump_pc_sgpr, arch);
        if (!subtract_pc) {
          result.errors.emplace_back(
              "ConSan MOI sampled dense atomic relay could not encode its dispatch key");
          return;
        }
        dispatcher_words.push_back(*subtract_pc);
        dispatcher_words.push_back(build_s_add_u32(key_sgpr, key_sgpr, /*literal=*/255u, arch));
        dispatcher_words.push_back(static_cast<uint32_t>(dispatcher_offset - island_offset));
      }
      const auto append_target = [&](uint64_t caller_return, uint64_t target) {
        const auto compare = instrumentation::build_s_cmp_eq_u32(
            key_encodes_scc ? jump_pc_sgpr : key_sgpr, /*literal=*/255u, arch);
        if (!compare)
          return false;
        dispatcher_words.push_back(*compare);
        dispatcher_words.push_back(static_cast<uint32_t>(caller_return - island_offset) /
                                   (key_encodes_scc ? 2u : 1u));
        std::vector<uint32_t> matched_words;
        if (key_encodes_scc) {
          const uint16_t bitcmp_opcode =
              arch == ROCJITSU_CODE_ARCH_CDNA3 ? cdna3::kSBitcmp1B32Sopc : cdna4::kSBitcmp1B32Sopc;
          const auto normalize = instrumentation::build_s_cselect_b32(
              key_sgpr, scalar_positive_inline_u32(1), scalar_positive_inline_u32(0), arch);
          if (!normalize)
            return false;
          matched_words.push_back(
              build_sopc_encoding(arch, bitcmp_opcode, key_sgpr, scalar_positive_inline_u32(0)));
          matched_words.push_back(*normalize);
        }
        const uint64_t matched_offset =
            dispatcher_offset + (dispatcher_words.size() + 1u) * sizeof(uint32_t);
        if (!append_moi_scc_preserving_indirect_jump(matched_words, matched_offset, target,
                                                     jump_pc_sgpr, saved_scc_sgpr,
                                                     /*capture_scc=*/false, arch)) {
          return false;
        }
        const auto skip =
            instrumentation::build_s_cbranch_scc0(static_cast<int16_t>(matched_words.size()), arch);
        if (!skip)
          return false;
        dispatcher_words.push_back(*skip);
        dispatcher_words.insert(dispatcher_words.end(), matched_words.begin(), matched_words.end());
        return true;
      };
      if (!append_target(island_offset, host_body_offset))
        return;
      for (const DenseBody &body : bodies) {
        const Planned &planned = selected[body.selected_index];
        if (!append_target(planned.physical_anchor_offset + sizeof(uint32_t), body.offset))
          return;
      }
      dispatcher_words.push_back(build_s_endpgm(arch));
      if (dispatcher_words.size() * sizeof(uint32_t) > dispatcher_reservation) {
        result.errors.emplace_back(
            "ConSan MOI sampled dense atomic dispatcher exceeds its reserved relay space");
        return;
      }
      std::memcpy(new_text.data() + dispatcher_offset, dispatcher_words.data(),
                  dispatcher_words.size() * sizeof(uint32_t));

      ConSanPatchInfo host_info;
      host_info.kind = ConSanPatchKind::TrampolineMoiIndirectBranchIsland;
      host_info.anchor_offset = host_offset;
      host_info.original_size = static_cast<uint32_t>(host_bytes);
      host_info.trampoline_offset = host_body_offset;
      host_info.trampoline_size = static_cast<uint32_t>(host_words.size() * sizeof(uint32_t));
      host_info.owner_descriptor_file_offsets = group_owners;
      patches.push_back(std::move(host_info));
      ConSanPatchInfo dispatcher_info;
      dispatcher_info.kind = ConSanPatchKind::TrampolineMoiIndirectBranchIsland;
      dispatcher_info.anchor_offset = selected[indices.front()].candidate.atomic.site.text_offset;
      dispatcher_info.trampoline_offset = dispatcher_offset;
      dispatcher_info.trampoline_size =
          static_cast<uint32_t>(dispatcher_words.size() * sizeof(uint32_t));
      dispatcher_info.owner_descriptor_file_offsets = group_owners;
      patches.push_back(std::move(dispatcher_info));
      for (const DenseBody &body : bodies) {
        const Planned &planned = selected[body.selected_index];
        const auto call_delta =
            compute_sopp_branch_simm16(planned.physical_anchor_offset, island_offset);
        if (!call_delta) {
          result.errors.emplace_back(
              "ConSan MOI sampled dense atomic relay could not encode an atomic call");
          return;
        }
        const uint32_t call = build_s_call_b64(call_return_sgpr, *call_delta, arch);
        std::vector<uint32_t> anchor(planned.relocated_guest_size / sizeof(uint32_t),
                                     build_s_nop(0, arch));
        anchor.front() = call;
        std::memcpy(new_text.data() + planned.physical_anchor_offset, anchor.data(),
                    planned.relocated_guest_size);
        append_atomic_patch(planned, body.offset, body.words, body.guest_instruction_offset);
        routed_by_dense_relay[body.selected_index] = true;
      }
    }
  }

  DbiPatchPlacementPlanner placement_planner(arch, old_text.size());
  if (!placement_planner.reserve_appended_prefix(new_text.size() - old_text.size())) {
    result.errors.emplace_back("ConSan MOI sampled atomic dense relay reservation failed");
    return;
  }
  std::vector<MoiLocalNopIsland> appended_indirect_islands;
  const size_t ordinary_selected_count =
      static_cast<size_t>(std::ranges::count(routed_by_dense_relay, false));
  if (ordinary_selected_count > 1u) {
    const uint64_t island_bytes =
        static_cast<uint64_t>(kMoiLocalIndirectIslandWords) * sizeof(uint32_t);
    const uint64_t bank_offset = placement_planner.appended_end();
    const uint64_t bank_bytes = ordinary_selected_count * island_bytes;
    std::string bank_error;
    if (!placement_planner.reserve_appended_prefix(bank_bytes, &bank_error)) {
      result.errors.emplace_back("ConSan MOI sampled atomic island bank failed: " + bank_error);
      return;
    }
    appended_indirect_islands.reserve(ordinary_selected_count);
    for (size_t index = 0; index < ordinary_selected_count; ++index) {
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
  for (size_t planned_index = 0; planned_index < selected.size(); ++planned_index) {
    if (routed_by_dense_relay[planned_index])
      continue;
    const Planned &planned = selected[planned_index];
    const ConSanAtomicSite &site = planned.candidate.atomic.site;
    const uint64_t physical_anchor_offset = planned.physical_anchor_offset;
    uint64_t entry_anchor_offset = physical_anchor_offset;
    uint32_t entry_original_size = planned.relocated_guest_size;
    std::vector<uint32_t> leading_guest_words;
    bool preserve_guest_at_anchor = false;
    const bool release_only_no_return =
        !planned.candidate.deferred_acquire && !consan_atomic_is_compare_exchange(site) &&
        !site.returns_old_value.value_or(false) &&
        !(planned.spill && sampled_atomic_spill_overlaps_guest_operands(*planned.spill, site));
    if (release_only_no_return && physical_anchor_offset == site.text_offset &&
        planned.relocated_guest_size == site.size) {
      BasicBlock *block = block_for_offset(resource_state.block_index, physical_anchor_offset);
      const Instruction *prefix = nullptr;
      if (block != nullptr) {
        for (const Instruction &instruction : block->instructions()) {
          if (instruction.size() > 0 &&
              instruction.src_loc() + static_cast<uint64_t>(instruction.size()) ==
                  physical_anchor_offset) {
            prefix = &instruction;
          }
        }
      }
      if (prefix != nullptr && prefix->mnemonic() != "s_clause" &&
          prefix->size() >= static_cast<int>(sizeof(uint32_t)) &&
          prefix->size() % static_cast<int>(sizeof(uint32_t)) == 0) {
        const uint64_t prefix_begin = prefix->src_loc();
        const uint64_t prefix_end = physical_anchor_offset;
        const bool overlaps_patch = std::ranges::any_of(result.patches, [&](const auto &patch) {
          const uint64_t patch_end =
              patch.anchor_offset + std::max<uint32_t>(patch.original_size, 1u);
          return prefix_begin < patch_end && patch.anchor_offset < prefix_end;
        });
        if (!overlaps_patch && prefix_end <= old_text.size()) {
          entry_anchor_offset = prefix_begin;
          entry_original_size = static_cast<uint32_t>(prefix->size());
          leading_guest_words.resize(entry_original_size / sizeof(uint32_t));
          std::memcpy(leading_guest_words.data(), old_text.data() + entry_anchor_offset,
                      entry_original_size);
          preserve_guest_at_anchor = true;
        }
      }
    }
    ConSanOptions route_options = options;
    if (!apply_moi_transient_sgpr_assignment(route_options,
                                             planned.resources.owner_descriptor_file_offsets)) {
      result.warnings.emplace_back(
          "ConSan MOI sampled atomic sync lost its ordinary-route scalar assignment");
      continue;
    }
    if (route_options.automatic_moi_record_replay_sgpr_spill) {
      const uint16_t ordinary_sgpr_limit = moi_ordinary_sgpr_limit(arch);
      const std::array complete_scalar_range{
          MoiSgprRange{.base = 0u, .width = ordinary_sgpr_limit}};
      MoiDenseOwnerLivenessProof route_owner_proof =
          resource_state.prepare_dense_liveness_for_owners(
              planned.resources.owner_descriptor_file_offsets, complete_scalar_range);
      std::vector<consan_detail::ScalarOwnerContextSummary> route_scalar_summaries;
      route_scalar_summaries.reserve(route_owner_proof.contexts.size());
      for (const MoiKernelResourceContext *context : route_owner_proof.contexts) {
        if (context != nullptr)
          route_scalar_summaries.push_back(context->scalar_summary());
      }
      const auto route_overlaps_reserved_scalar_state = [&](uint16_t base, uint16_t width) {
        const auto overlaps_optional = [&](std::optional<uint16_t> reg, uint16_t reg_width = 1u) {
          return reg && range_overlaps(base, width, *reg, reg_width);
        };
        bool overlaps = persistent_sgpr_range_overlaps_reserved_ordinary_range(base, width, arch) ||
                        overlaps_optional(route_options.moi_owner_sgpr) ||
                        overlaps_optional(route_options.moi_persistent_owner_sgpr) ||
                        overlaps_optional(route_options.moi_persistent_epoch_sgpr) ||
                        overlaps_optional(route_options.moi_persistent_workgroup_key_sgpr) ||
                        overlaps_optional(route_options.moi_dispatch_id_sgpr,
                                          moi_dispatch_state_sgpr_count(route_options));
        for (const std::optional<uint16_t> reg :
             {route_options.moi_record_replay_workgroup_sgprs.x,
              route_options.moi_record_replay_workgroup_sgprs.y,
              route_options.moi_record_replay_workgroup_sgprs.z,
              route_options.moi_record_replay_workgroup_sgprs.cluster_workgroup_id}) {
          overlaps |= overlaps_optional(reg);
        }
        if (route_options.delay_mode == ConSanDelayMode::SleepVar)
          overlaps |= range_overlaps(base, width, route_options.delay_var_ssrc, 1u);
        return overlaps;
      };
      const auto route_dead_at_anchor = [&](uint16_t base, uint16_t width) {
        if (!route_owner_proof.complete || route_owner_proof.contexts.empty() || width == 0u ||
            static_cast<uint32_t>(base) + width > ordinary_sgpr_limit ||
            route_overlaps_reserved_scalar_state(base, width)) {
          return false;
        }
        const RegisterRef range{RegClass::SGPR, base, static_cast<uint8_t>(width)};
        BasicBlock *block = block_for_offset(resource_state.block_index, physical_anchor_offset);
        const Instruction *instruction = instruction_at(block, physical_anchor_offset);
        if (block == nullptr || instruction == nullptr)
          return false;
        bool saw_owner = false;
        for (const MoiKernelResourceContext *context : route_owner_proof.contexts) {
          if (context == nullptr || !context->block_set.contains(block))
            continue;
          saw_owner = true;
          if (context->liveness_is_restricted || context->has_indirect_sgpr_access ||
              static_cast<uint32_t>(base) + width > context->current_sgpr_count ||
              context->liveness->any_live_before(*instruction, range)) {
            return false;
          }
          const InstDefUse def_use(*instruction);
          if (def_use.defs.contains(range) || def_use.uses.contains(range))
            return false;
          std::vector<const Instruction *> lexical_tail;
          for (BasicBlock *owner_block : context->scope.blocks) {
            if (owner_block == nullptr || owner_block->end_offset() <= physical_anchor_offset)
              continue;
            for (const Instruction &tail_instruction : owner_block->instructions()) {
              if (tail_instruction.src_loc() > physical_anchor_offset)
                lexical_tail.push_back(&tail_instruction);
            }
          }
          std::ranges::sort(lexical_tail, {}, &Instruction::src_loc);
          std::vector<bool> redefined(width, false);
          for (const Instruction *tail_instruction : lexical_tail) {
            const InstDefUse tail_def_use(*tail_instruction);
            for (uint16_t lane = 0u; lane < width; ++lane) {
              const RegisterRef scalar{RegClass::SGPR, static_cast<uint16_t>(base + lane), 1u};
              if (!redefined[lane] && tail_def_use.uses.contains(scalar))
                return false;
              if (tail_def_use.defs.contains(scalar))
                redefined[lane] = true;
            }
            if (std::ranges::all_of(redefined, std::identity{}))
              break;
          }
        }
        return saw_owner;
      };
      const auto route_tuple_is_safe = [&](uint16_t pc, uint16_t scc) {
        const std::array ranges{MoiSgprRange{pc, 2u}, MoiSgprRange{scc, 1u}};
        return route_dead_at_anchor(pc, 2u) && route_dead_at_anchor(scc, 1u) &&
               !range_overlaps(pc, 2u, scc, 1u) &&
               !moi_scalar_contexts_conflict_with_physical_vcc(route_scalar_summaries, ranges,
                                                               arch);
      };
      auto route_state = moi_indirect_jump_sgprs(route_options);
      if (!route_state || !route_tuple_is_safe(route_state->pc_sgpr, route_state->scc_save_sgpr)) {
        bool found = false;
        for (uint32_t pair = 0u; !found && pair + 2u <= ordinary_sgpr_limit; pair += 2u) {
          const uint16_t candidate_pair = static_cast<uint16_t>(pair);
          if (!route_dead_at_anchor(candidate_pair, 2u))
            continue;
          for (uint32_t scc = 0u; scc < ordinary_sgpr_limit; ++scc) {
            const uint16_t candidate_scc = static_cast<uint16_t>(scc);
            if (!route_tuple_is_safe(candidate_pair, candidate_scc))
              continue;
            route_options.moi_inline_indirect_pc_sgpr = candidate_pair;
            route_options.moi_inline_indirect_scc_sgpr = candidate_scc;
            route_state = moi_indirect_jump_sgprs(route_options);
            found = route_state.has_value();
            if (found)
              break;
          }
        }
        if (!found) {
          result.warnings.emplace_back(
              "ConSan MOI sampled atomic sync has no call-anchor-safe ordinary route");
          continue;
        }
      }
    }
    const uint64_t cave_offset = placement_planner.appended_end();
    uint32_t guest_instruction_offset = 0;
    auto cave_words =
        build_atomic_cave(planned, cave_offset, guest_instruction_offset, &route_options,
                          preserve_guest_at_anchor, leading_guest_words);
    if (!cave_words) {
      result.warnings.emplace_back(
          "ConSan MOI sampled atomic sync could not lower associated " +
          std::string(planned.candidate.deferred_acquire ? "deferred-acquire" : "release") +
          " site at text offset " + std::to_string(site.text_offset));
      continue;
    }
    const bool direct_entry =
        compute_sopp_branch_simm16(entry_anchor_offset, cave_offset).has_value();
    std::optional<uint64_t> indirect_entry_island;
    if (!direct_entry) {
      const uint64_t reserved_island =
          reserved_atomic_island_begin +
          planned_index * kMoiLocalIndirectIslandWords * sizeof(uint32_t);
      if (planned_index < reserved_atomic_island_count && reserved_island <= new_text.size() &&
          kMoiLocalIndirectIslandWords * sizeof(uint32_t) <= new_text.size() - reserved_island &&
          compute_sopp_branch_simm16(entry_anchor_offset, reserved_island)) {
        indirect_entry_island = reserved_island;
      }
    }
    if (!direct_entry && !indirect_entry_island) {
      indirect_entry_island = claim_appended_island(entry_anchor_offset);
      if (!indirect_entry_island)
        indirect_entry_island = indirect_allocator.claim_reachable(entry_anchor_offset);
    }
    if (!direct_entry && !indirect_entry_island) {
      result.warnings.emplace_back(
          "ConSan MOI sampled atomic sync found no reachable indirect entry island in " +
          planned.candidate.atomic.container_name);
      continue;
    }
    std::string placement_error;
    const auto placement = placement_planner.plan_indirect_appended(
        entry_anchor_offset, entry_original_size, cave_words->size() * sizeof(uint32_t),
        &placement_error);
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
    std::vector<uint32_t> anchor(entry_original_size / sizeof(uint32_t), build_s_nop(0, arch));
    if (anchor.empty())
      return;
    anchor.front() = build_s_branch(*fwd, arch);
    std::memcpy(new_text.data() + entry_anchor_offset, anchor.data(),
                anchor.size() * sizeof(uint32_t));
    if (indirect_entry_island &&
        !emit_moi_local_indirect_entry_island(
            new_text, *indirect_entry_island, placement->body_offset, entry_anchor_offset,
            route_options, planned.resources.owner_descriptor_file_offsets, arch, patches,
            result.errors, "ConSan MOI sampled atomic sync patch")) {
      return;
    }
    append_words_bytes(new_text, *cave_words);

    if (preserve_guest_at_anchor) {
      ConSanPatchInfo prefix_info;
      prefix_info.kind = ConSanPatchKind::TrampolineMoiIndirectBranchIsland;
      prefix_info.anchor_offset = entry_anchor_offset;
      prefix_info.original_size = entry_original_size;
      prefix_info.trampoline_offset = cave_offset;
      prefix_info.trampoline_size = static_cast<uint32_t>(cave_words->size() * sizeof(uint32_t));
      prefix_info.owner_descriptor_file_offsets = planned.resources.owner_descriptor_file_offsets;
      patches.push_back(std::move(prefix_info));
    }
    append_atomic_patch(planned, cave_offset, *cave_words, guest_instruction_offset,
                        preserve_guest_at_anchor);
  }
  if (!replace_consan_text(patcher, new_text, options, "MOI sampled atomic sync", result)) {
    return;
  }
  result.replacement = std::move(patcher).emit();
  result.patches.insert(result.patches.end(), patches.begin(), patches.end());
  result.mark_modified();
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
  (void)record_index;
  if (!options.scratch_vgpr) {
    errors.emplace_back("ConSan MOI fence record patch requires RJ_CONSAN_TMP_VGPR");
    return std::nullopt;
  }
  const bool requires_raw_th = !consan_uses_gfx9_cdna_encoding(arch);
  if (!address_plan.supported() || !candidate.communication_site.raw_scope ||
      (requires_raw_th && !candidate.communication_site.raw_th) ||
      candidate.association == nullptr || !candidate.association->communication_event ||
      candidate.fence_event == nullptr) {
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
  if (candidate.patch_file_offset > bytes.size() ||
      candidate.patch_size > bytes.size() - candidate.patch_file_offset) {
    errors.emplace_back("ConSan MOI fence record patch site exceeds ELF bytes");
    return std::nullopt;
  }

  std::optional<uint16_t> derived_owner_vgpr;
  std::vector<uint32_t> derived_owner_words;
  if (!options.moi_owner_vgpr && !options.moi_persistent_owner_sgpr) {
    const uint16_t value_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 5u);
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
  const uint64_t displaced_tail_bytes =
      static_cast<uint64_t>(displaced_tail_words.size()) * sizeof(uint32_t);
  if (displaced_tail_bytes > std::numeric_limits<uint32_t>::max() - candidate.patch_size ||
      candidate.patch_file_offset > bytes.size() ||
      candidate.patch_size + displaced_tail_bytes > bytes.size() - candidate.patch_file_offset) {
    errors.emplace_back("ConSan MOI fence record runtime guest exceeds ELF bytes");
    return std::nullopt;
  }
  const bool runtime_workgroup_gate =
      !candidate.capture_address_before_guest && options.moi_runtime_sample_stride > 1u &&
      !options.automatic_moi_record_replay_sgpr_spill &&
      moi_has_probe_entry_runtime_workgroup_gate(options, workgroup_sources, arch);
  if (runtime_workgroup_gate) {
    if (candidate.capture_address_before_guest) {
      errors.emplace_back(
          "ConSan MOI pre-acquire fence capture does not support a runtime workgroup gate");
      return std::nullopt;
    }
    const uint32_t guest_byte_count =
        static_cast<uint32_t>(candidate.patch_size + displaced_tail_bytes);
    auto gate = build_moi_runtime_workgroup_gate_call_words(
        bytes.subspan(candidate.patch_file_offset, guest_byte_count), options, workgroup_sources,
        cave_text_offset, return_text_offset,
        moi_runtime_workgroup_gate_reserved_words(
            guest_byte_count, workgroup_sources.cluster_workgroup_id.has_value()),
        arch);
    if (!gate) {
      errors.emplace_back(
          "ConSan MOI fence record patch could not encode its runtime workgroup gate");
      return std::nullopt;
    }
    words = std::move(*gate);
  }
  const std::optional<uint16_t> vgpr_msb_mode =
      arch == ROCJITSU_CODE_ARCH_CDNA5
          ? gfx1250_vgpr_msb_mode_at(bytes, candidate.text_file_offset,
                                     candidate.container_entry_text_offset,
                                     candidate.patch_file_offset)
          : std::nullopt;
  const bool select_low_vgpr_bank = vgpr_msb_mode.value_or(0u) != 0u;
  words.reserve(words.size() + candidate.patch_size / sizeof(uint32_t) + 80u +
                (spill ? spill->save_words.size() + spill->restore_words.size() : 0u) +
                (select_low_vgpr_bank ? 2u : 0u));
  if (select_low_vgpr_bank)
    words.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
        static_cast<uint8_t>(*vgpr_msb_mode), 0u, arch));
  if (spill)
    words.insert(words.end(), spill->save_words.begin(), spill->save_words.end());
  if (scalar_spill)
    words.insert(words.end(), scalar_spill->save_words.begin(), scalar_spill->save_words.end());
  const uint16_t recorded_address_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 6u);
  const auto append_address_capture = [&]() {
    if (address_plan.requires_materialization()) {
      const auto special_state = moi_special_state_sgprs(options);
      if (!special_state) {
        errors.emplace_back(
            "ConSan MOI fence record address materialization requires special-state SGPRs");
        return false;
      }
      const auto address_words = build_consan_moi_atomic_address_materialization(
          address_plan, special_state->vcc_save_sgpr, special_state->scc_save_sgpr, arch);
      if (!address_words) {
        errors.emplace_back("ConSan MOI fence record patch could not materialize its address");
        return false;
      }
      words.insert(words.end(), address_words->begin(), address_words->end());
    }
    const uint16_t address_vgpr = address_plan.result_address_vgpr;
    words.push_back(
        build_v_mov_b32_e32(recorded_address_vgpr, vector_source_vgpr(address_vgpr), arch));
    words.push_back(
        build_v_mov_b32_e32(static_cast<uint16_t>(recorded_address_vgpr + 1u),
                            vector_source_vgpr(static_cast<uint16_t>(address_vgpr + 1u)), arch));
    return true;
  };
  if (candidate.capture_address_before_guest && !append_address_capture())
    return std::nullopt;
  for (uint64_t offset = 0; offset < candidate.patch_size; offset += sizeof(uint32_t)) {
    uint32_t word = 0;
    std::memcpy(&word, bytes.data() + candidate.patch_file_offset + offset, sizeof(word));
    words.push_back(word);
  }
  if (!candidate.capture_address_before_guest && !append_address_capture())
    return std::nullopt;
  if (!append_save_moi_special_state(words, options, arch)) {
    errors.emplace_back("ConSan MOI fence record patch could not save VCC/SCC");
    return std::nullopt;
  }

  const uint64_t base = *options.moi_report_buffer_address;
  const uint64_t record_base = base + fence_records_offset;
  const ConSanMoiFenceEventKind kind =
      candidate.association->memory_role == ConSanSyncMemoryRole::Release
          ? ConSanMoiFenceEventKind::Release
          : ConSanMoiFenceEventKind::Acquire;
  if (!options.moi_exec_save_sgpr) {
    errors.emplace_back("ConSan MOI dynamic fence record requires scalar EXEC state");
    return std::nullopt;
  }
  {
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
    const auto saved_exec_wait = instrumentation::build_salu_to_valu_dependency_wait(arch);
    if (!save_active_exec || !saved_exec_wait || !mbcnt_lo || !mbcnt_hi || !first_active_lane ||
        !narrow_exec) {
      errors.emplace_back("ConSan MOI dynamic fence record could not select a wave publisher");
      return std::nullopt;
    }
    words.push_back(*save_active_exec);
    words.push_back(*saved_exec_wait);
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
            static_cast<uint32_t>(candidate.fence_event->text_offset), slot_vgpr,
            *options.scratch_vgpr, arch) ||
        !append_dynamic_fence_store_u32_literal(
            record_words, record_base + offsetof(ConSanMoiFenceRecord, kind),
            static_cast<uint32_t>(kind), slot_vgpr, *options.scratch_vgpr, arch) ||
        !append_dynamic_fence_store_u32_literal(
            record_words, record_base + offsetof(ConSanMoiFenceRecord, scope),
            *candidate.communication_site.raw_scope, slot_vgpr, *options.scratch_vgpr, arch) ||
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
    const auto wait_store = instrumentation::build_s_wait_global_store0(arch);
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
