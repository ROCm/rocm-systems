struct BarrierRecordCandidate {
  std::string container_name;
  bool in_kernel = true;
  ConSanBarrierSite site;
  std::optional<uint64_t> kernel_descriptor_file_offset;
  uint64_t container_entry_text_offset = 0;
  uint64_t container_code_size = 0;
  uint64_t text_file_offset = 0;
};

[[nodiscard]] bool barrier_record_candidate_semantics_equal(const BarrierRecordCandidate &lhs,
                                                            const BarrierRecordCandidate &rhs) {
  const ConSanBarrierSite &left = lhs.site;
  const ConSanBarrierSite &right = rhs.site;
  return lhs.in_kernel == rhs.in_kernel &&
         lhs.container_entry_text_offset == rhs.container_entry_text_offset &&
         lhs.container_code_size == rhs.container_code_size &&
         lhs.text_file_offset == rhs.text_file_offset &&
         std::tie(left.operation, left.text_offset, left.file_offset, left.size, left.barrier_id,
                  left.operand_source, left.raw_operand_selector, left.literal_width_bits,
                  left.literal_value, left.raw_simm16, left.scope, left.participant_encoding,
                  left.participant_count, left.participant_mask, left.participant_setup_text_offset,
                  left.participant_setup_file_offset, left.participant_setup_literal,
                  left.mnemonic) ==
             std::tie(right.operation, right.text_offset, right.file_offset, right.size,
                      right.barrier_id, right.operand_source, right.raw_operand_selector,
                      right.literal_width_bits, right.literal_value, right.raw_simm16, right.scope,
                      right.participant_encoding, right.participant_count, right.participant_mask,
                      right.participant_setup_text_offset, right.participant_setup_file_offset,
                      right.participant_setup_literal, right.mnemonic);
}

[[nodiscard]] bool canonicalize_barrier_record_candidates_by_physical_site(
    std::vector<BarrierRecordCandidate> &candidates, ConSanResult &result) {
  return consan_detail::canonicalize_physical_site_aliases(
      candidates, result.errors, "ConSan MOI barrier site",
      [](const BarrierRecordCandidate &candidate) { return candidate.site.file_offset; },
      [](const BarrierRecordCandidate &candidate) -> std::string_view {
        return candidate.container_name;
      },
      barrier_record_candidate_semantics_equal,
      consan_detail::invalidate_physical_site_single_owner_hint);
}

class MoiPatchAnchorRangeIndex {
public:
  explicit MoiPatchAnchorRangeIndex(const ConSanResult &result) {
    for (const ConSanPatchInfo &patch : result.patches) {
      if (patch.original_size != 0u)
        ranges_.emplace_back(patch.anchor_offset, patch.anchor_offset + patch.original_size);
    }
    std::ranges::sort(ranges_);
    std::vector<std::pair<uint64_t, uint64_t>> merged;
    for (const auto &range : ranges_) {
      if (merged.empty() || merged.back().second < range.first)
        merged.push_back(range);
      else
        merged.back().second = std::max(merged.back().second, range.second);
    }
    ranges_ = std::move(merged);
  }

  [[nodiscard]] bool overlaps(uint64_t begin, uint64_t size) const {
    if (size == 0u || begin > std::numeric_limits<uint64_t>::max() - size)
      return false;
    const auto it =
        std::ranges::lower_bound(ranges_, begin + size, {}, &std::pair<uint64_t, uint64_t>::first);
    return it != ranges_.begin() && std::prev(it)->second > begin;
  }

private:
  std::vector<std::pair<uint64_t, uint64_t>> ranges_;
};

[[nodiscard]] bool is_rocclr_runtime_kernel_name(std::string_view name) {
  return name.starts_with("__amd_rocclr_");
}

void append_barrier_epoch_candidates(const ConSanKernelInfo &kernel,
                                     std::vector<BarrierRecordCandidate> &candidates) {
  if (is_rocclr_runtime_kernel_name(kernel.name))
    return;
  for (const ConSanBarrierSite &site : kernel.barrier_sites)
    candidates.push_back({"kernel:" + kernel.name, true, site, kernel.descriptor_file_offset,
                          kernel.entry_text_offset, kernel.has_text_range ? kernel.code_size : 0u,
                          kernel.text_file_offset});
}

void append_barrier_epoch_candidates(const ConSanFunctionInfo &function,
                                     std::vector<BarrierRecordCandidate> &candidates) {
  for (const ConSanBarrierSite &site : function.barrier_sites)
    candidates.push_back({"function:" + function.name, false, site, std::nullopt,
                          function.entry_text_offset, function.code_size,
                          function.text_file_offset});
}

[[nodiscard]] uint16_t inline_shadow_barrier_scratch_count(const ConSanOptions &options) {
  if (options.moi_engine == ConSanMoiEngine::RecordReplay)
    return 6u;
  // An object with no Inline access probes has no other way to prove that its
  // epoch-only barrier probe executed. Reserve a result/rank VGPR plus an
  // address pair for one aggregate event-counter publication. Ordinary
  // access-bearing objects publish through their access probes and retain the
  // single-register epoch-only barrier plan.
  return options.moi_engine == ConSanMoiEngine::InlineShadow && options.moi_report_buffer_address &&
                 !options.moi_inline_access_present
             ? 3u
             : 1u;
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_barrier_record_cave_words(
    std::span<const uint8_t> bytes, const BarrierRecordCandidate &candidate,
    const ConSanOptions &options, const VgprSpillSequence *spill,
    const InlineSgprSpillSequence *scalar_spill, rj_code_arch_t arch,
    uint32_t barrier_record_capacity, size_t barrier_records_offset, uint32_t original_barrier_word,
    uint64_t cave_text_offset, uint64_t return_text_offset,
    std::optional<uint32_t> private_owner_offset, std::optional<uint16_t> private_owner_shift,
    std::vector<std::string> &errors) {
  if (!options.scratch_vgpr) {
    errors.emplace_back("ConSan MOI barrier record patch requires RJ_CONSAN_TMP_VGPR");
    return std::nullopt;
  }
  if (static_cast<uint32_t>(*options.scratch_vgpr) + 6u > kMaxVgprs) {
    errors.emplace_back("ConSan MOI barrier record patch needs six scratch VGPRs");
    return std::nullopt;
  }
  if (!options.moi_exec_save_sgpr) {
    errors.emplace_back("ConSan MOI barrier record patch requires RJ_CONSAN_MOI_EXEC_SAVE_SGPR");
    return std::nullopt;
  }
  if (*options.moi_exec_save_sgpr > 100u || *options.moi_exec_save_sgpr % 2u != 0u) {
    errors.emplace_back(
        "ConSan MOI barrier record patch requires an even RJ_CONSAN_MOI_EXEC_SAVE_SGPR in 0..100");
    return std::nullopt;
  }
  if (reject_optional_scratch_range_overlap(options.moi_owner_vgpr, *options.scratch_vgpr, 6,
                                            "MOI owner", errors))
    return std::nullopt;

  std::optional<uint16_t> derived_owner_vgpr;
  std::vector<uint32_t> derived_owner_words;
  if (!options.moi_owner_vgpr && !options.moi_persistent_owner_sgpr) {
    const uint16_t value_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 5u);
    const auto owner = build_moi_record_owner_derivation(
        bytes, candidate.kernel_descriptor_file_offset, private_owner_offset, private_owner_shift,
        value_vgpr, arch, "barrier record patch", errors);
    if (!owner)
      return std::nullopt;
    derived_owner_vgpr = owner->vgpr;
    derived_owner_words = owner->words;
  }
  const auto persistent_workgroup_sources =
      record_replay_persistent_workgroup_sources(options, arch);
  if (!persistent_workgroup_sources) {
    errors.emplace_back(
        "ConSan MOI Record/Replay barrier requires one exact entry-captured workgroup tuple");
    return std::nullopt;
  }
  const ConSanMoiWorkgroupSources workgroup_sources = *persistent_workgroup_sources;

  std::vector<uint32_t> words;
  const uint64_t base = *options.moi_report_buffer_address;
  const uint64_t barrier_record_base = base + barrier_records_offset;
  const uint16_t slot_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 2u);
  const uint16_t lane_rank_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 5u);
  constexpr uint16_t kScalarInlineMinusOne = 0xC1;

  const std::optional<uint16_t> vgpr_msb_mode =
      arch == ROCJITSU_CODE_ARCH_GFX1250
          ? gfx1250_vgpr_msb_mode_at(bytes, candidate.text_file_offset,
                                     candidate.container_entry_text_offset,
                                     candidate.site.file_offset)
          : std::nullopt;
  const bool select_low_vgpr_bank = vgpr_msb_mode.value_or(0u) != 0u;
  words.reserve(
      186 + (spill ? spill->save_words.size() + spill->restore_words.size() : 0u) +
      (scalar_spill ? scalar_spill->save_words.size() + scalar_spill->restore_words.size() : 0u));
  const bool runtime_workgroup_gate =
      options.moi_runtime_sample_stride > 1u && !options.automatic_moi_record_replay_sgpr_spill &&
      workgroup_sources.x.scalar_src && workgroup_sources.y.scalar_src &&
      workgroup_sources.z.scalar_src;
  if (runtime_workgroup_gate) {
    ConSanMoiCandidate barrier_guest;
    barrier_guest.file_offset = candidate.site.file_offset;
    barrier_guest.size = candidate.site.size;
    auto gate = build_moi_runtime_workgroup_gate_call_words(
        bytes, barrier_guest, options, workgroup_sources, cave_text_offset, return_text_offset,
        moi_runtime_workgroup_gate_reserved_words(barrier_guest.size), arch);
    if (!gate) {
      errors.emplace_back(
          "ConSan MOI barrier record patch could not encode its runtime workgroup gate");
      return std::nullopt;
    }
    words = std::move(*gate);
  }
  if (select_low_vgpr_bank)
    words.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
        static_cast<uint8_t>(*vgpr_msb_mode), 0u, arch));
  if (spill)
    words.insert(words.end(), spill->save_words.begin(), spill->save_words.end());
  if (scalar_spill)
    words.insert(words.end(), scalar_spill->save_words.begin(), scalar_spill->save_words.end());
  if (!append_save_moi_special_state(words, options, arch)) {
    errors.emplace_back("ConSan MOI barrier record patch could not save VCC/SCC");
    return std::nullopt;
  }
  const auto mbcnt_lo = instrumentation::build_v_mbcnt_lo_u32_b32(
      lane_rank_vgpr, kScalarInlineMinusOne, scalar_positive_inline_u32(0), arch);
  const auto mbcnt_hi = instrumentation::build_v_mbcnt_hi_u32_b32(
      lane_rank_vgpr, kScalarInlineMinusOne, vector_source_vgpr(lane_rank_vgpr), arch);
  const auto first_active_lane =
      instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), lane_rank_vgpr, arch);
  const auto save_exec =
      instrumentation::build_s_and_saveexec_b64(*options.moi_exec_save_sgpr, kRdna4VccLo, arch);
  if (!mbcnt_lo || !mbcnt_hi || !first_active_lane || !save_exec) {
    errors.emplace_back("ConSan MOI barrier record patch could not encode EXEC narrowing");
    return std::nullopt;
  }
  words.insert(words.end(), mbcnt_lo->begin(), mbcnt_lo->end());
  words.insert(words.end(), mbcnt_hi->begin(), mbcnt_hi->end());
  words.push_back(*first_active_lane);
  words.push_back(*save_exec);

  if (!append_atomic_fetch_add_one_u32(words,
                                       base + offsetof(ConSanMoiReportHeader, barrier_record_count),
                                       slot_vgpr, *options.scratch_vgpr, arch)) {
    errors.emplace_back("ConSan MOI barrier record patch could not encode dynamic slot reserve");
    return std::nullopt;
  }

  const uint16_t value_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 5u);
  const auto mov_capacity =
      instrumentation::build_v_mov_b32_literal(value_vgpr, barrier_record_capacity, arch);
  const auto slot_in_capacity =
      instrumentation::build_v_cmp_gt_u32_vcc(vector_source_vgpr(value_vgpr), slot_vgpr, arch);
  if (!mov_capacity || !slot_in_capacity) {
    errors.emplace_back("ConSan MOI barrier record patch could not encode capacity guard");
    return std::nullopt;
  }
  words.insert(words.end(), mov_capacity->begin(), mov_capacity->end());
  words.push_back(*slot_in_capacity);

  std::vector<uint32_t> record_words;
  record_words.reserve(128);
  record_words.insert(record_words.end(), derived_owner_words.begin(), derived_owner_words.end());
  if ((derived_owner_vgpr &&
       !append_dynamic_barrier_store_u32_vgpr(
           record_words, barrier_record_base + offsetof(ConSanMoiBarrierRecord, wave_id),
           *derived_owner_vgpr, slot_vgpr, *options.scratch_vgpr, arch)) ||
      !append_dynamic_record_store_moi_report_dispatch_id_pair(
          record_words, kDynamicBarrierRecordLayout,
          barrier_record_base + offsetof(ConSanMoiBarrierRecord, generation), options, slot_vgpr,
          *options.scratch_vgpr, arch, ConSanMoiLiteralDispatchIdPolicy::AnyArchitecture) ||
      !append_dynamic_barrier_event_index_store(
          record_words, base + offsetof(ConSanMoiReportHeader, event_counter),
          barrier_record_base + offsetof(ConSanMoiBarrierRecord, event_index), slot_vgpr,
          *options.scratch_vgpr, arch) ||
      !append_dynamic_barrier_store_workgroup_source(
          record_words, barrier_record_base + offsetof(ConSanMoiBarrierRecord, workgroup_x),
          workgroup_sources.x, slot_vgpr, *options.scratch_vgpr, arch) ||
      !append_dynamic_barrier_store_workgroup_source(
          record_words, barrier_record_base + offsetof(ConSanMoiBarrierRecord, workgroup_y),
          workgroup_sources.y, slot_vgpr, *options.scratch_vgpr, arch) ||
      !append_dynamic_barrier_store_workgroup_source(
          record_words, barrier_record_base + offsetof(ConSanMoiBarrierRecord, workgroup_z),
          workgroup_sources.z, slot_vgpr, *options.scratch_vgpr, arch) ||
      !append_dynamic_barrier_store_u32_scalar_src(
          record_words, barrier_record_base + offsetof(ConSanMoiBarrierRecord, lane_mask),
          *options.moi_exec_save_sgpr, slot_vgpr, *options.scratch_vgpr, arch) ||
      !append_dynamic_barrier_store_u32_scalar_src(
          record_words,
          barrier_record_base + offsetof(ConSanMoiBarrierRecord, lane_mask) + sizeof(uint32_t),
          static_cast<uint16_t>(*options.moi_exec_save_sgpr + 1u), slot_vgpr, *options.scratch_vgpr,
          arch) ||
      (options.moi_owner_vgpr &&
       !append_dynamic_barrier_store_u32_vgpr(
           record_words, barrier_record_base + offsetof(ConSanMoiBarrierRecord, wave_id),
           *options.moi_owner_vgpr, slot_vgpr, *options.scratch_vgpr, arch)) ||
      (options.moi_persistent_owner_sgpr &&
       !append_dynamic_barrier_store_u32_scalar_src(
           record_words, barrier_record_base + offsetof(ConSanMoiBarrierRecord, wave_id),
           *options.moi_persistent_owner_sgpr, slot_vgpr, *options.scratch_vgpr, arch)) ||
      !append_dynamic_barrier_store_u32_literal(
          record_words, barrier_record_base + offsetof(ConSanMoiBarrierRecord, instruction_offset),
          static_cast<uint32_t>(candidate.site.text_offset), slot_vgpr, *options.scratch_vgpr,
          arch)) {
    errors.emplace_back("ConSan MOI barrier record patch could not encode record stores");
    return std::nullopt;
  }

  if (record_words.size() > static_cast<size_t>(std::numeric_limits<int16_t>::max())) {
    errors.emplace_back("ConSan MOI barrier record overflow branch is out of range");
    return std::nullopt;
  }
  const auto skip_record =
      instrumentation::build_s_cbranch_vccz(static_cast<int16_t>(record_words.size()), arch);
  const auto restore_exec =
      instrumentation::build_s_mov_b64(kRdna4ExecLo, *options.moi_exec_save_sgpr, arch);
  if (!skip_record || !restore_exec) {
    errors.emplace_back("ConSan MOI barrier record patch could not encode EXEC restore");
    return std::nullopt;
  }
  words.push_back(*skip_record);
  words.insert(words.end(), record_words.begin(), record_words.end());
  words.push_back(*restore_exec);
  if (!append_restore_moi_special_state(words, options, arch)) {
    errors.emplace_back("ConSan MOI barrier record patch could not restore VCC/SCC");
    return std::nullopt;
  }

  if (scalar_spill) {
    words.insert(words.end(), scalar_spill->restore_words.begin(),
                 scalar_spill->restore_words.end());
  }
  if (spill)
    words.insert(words.end(), spill->restore_words.begin(), spill->restore_words.end());
  if (select_low_vgpr_bank)
    words.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
        0u, static_cast<uint8_t>(*vgpr_msb_mode), arch));
  words.push_back(original_barrier_word);

  if (!append_moi_direct_or_indirect_return(words, cave_text_offset, return_text_offset, options,
                                            arch)) {
    errors.emplace_back("ConSan MOI barrier record could not encode its return");
    return std::nullopt;
  }
  return words;
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_inline_shadow_barrier_epoch_cave_words(
    const ConSanBarrierSite &site, const ConSanOptions &options, rj_code_arch_t arch,
    std::optional<uint16_t> visible_evidence_scratch_vgpr, const VgprSpillSequence *spill,
    const InlineSgprSpillSequence *scalar_spill, uint32_t original_barrier_word,
    uint64_t cave_text_offset, uint64_t return_text_offset,
    std::optional<uint16_t> call_return_sgpr, std::optional<uint16_t> gfx1250_vgpr_msb_mode,
    std::vector<std::string> &errors) {
  const bool publish_visible_evidence = visible_evidence_scratch_vgpr.has_value();
  const bool control_gfx1250_vgpr_bank = arch == ROCJITSU_CODE_ARCH_GFX1250;
  const bool restore_guest_vgpr_bank =
      control_gfx1250_vgpr_bank && gfx1250_vgpr_msb_mode.value_or(0u) != 0u;
  if (publish_visible_evidence &&
      (!options.moi_report_buffer_address || !options.moi_exec_save_sgpr)) {
    errors.emplace_back(
        "ConSan MOI inline barrier visible evidence requires report and scalar state");
    return std::nullopt;
  }
  const auto append_visible_evidence = [&](std::vector<uint32_t> &words) {
    if (!publish_visible_evidence)
      return true;
    const uint16_t scratch = *visible_evidence_scratch_vgpr;
    const uint16_t active_exec_sgpr = static_cast<uint16_t>(*options.moi_exec_save_sgpr + 12u);
    const uint16_t temporary_exec_sgpr = static_cast<uint16_t>(*options.moi_exec_save_sgpr + 14u);
    const auto save_active_exec =
        instrumentation::build_s_mov_b64(active_exec_sgpr, kRdna4ExecLo, arch);
    const auto mbcnt_lo = instrumentation::build_v_mbcnt_lo_u32_b32(
        scratch, active_exec_sgpr, scalar_positive_inline_u32(0), arch);
    const auto mbcnt_hi = instrumentation::build_v_mbcnt_hi_u32_b32(
        scratch, static_cast<uint16_t>(active_exec_sgpr + 1u), vector_source_vgpr(scratch), arch);
    const auto first_active =
        instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), scratch, arch);
    const auto narrow_first =
        instrumentation::build_s_and_saveexec_b64(temporary_exec_sgpr, kRdna4VccLo, arch);
    const auto restore_exec =
        instrumentation::build_s_mov_b64(kRdna4ExecLo, active_exec_sgpr, arch);
    const auto saved_exec_wait = instrumentation::build_salu_to_valu_dependency_wait(arch);
    if (!save_active_exec || !saved_exec_wait || !mbcnt_lo || !mbcnt_hi || !first_active ||
        !narrow_first || !restore_exec)
      return false;
    words.push_back(*save_active_exec);
    words.push_back(*saved_exec_wait);
    words.insert(words.end(), mbcnt_lo->begin(), mbcnt_lo->end());
    words.insert(words.end(), mbcnt_hi->begin(), mbcnt_hi->end());
    words.push_back(*first_active);
    words.push_back(*narrow_first);
    if (!append_publish_visible_evidence_if_zero(
            words,
            *options.moi_report_buffer_address + offsetof(ConSanMoiReportHeader, event_counter),
            scratch, static_cast<uint16_t>(scratch + 1u), temporary_exec_sgpr, arch))
      return false;
    words.push_back(*restore_exec);
    return true;
  };
  if (options.moi_persistent_epoch_sgpr) {
    const auto special_state = moi_special_state_sgprs(options);
    if (!options.moi_exec_save_sgpr || !special_state) {
      errors.emplace_back("ConSan MOI scalar inline barrier requires scalar special-state storage");
      return std::nullopt;
    }
    const auto save_scc = instrumentation::build_s_cselect_b32(special_state->scc_save_sgpr,
                                                               scalar_positive_inline_u32(1),
                                                               scalar_positive_inline_u32(0), arch);
    const auto at_maximum = instrumentation::build_s_cmp_eq_u32(*options.moi_persistent_epoch_sgpr,
                                                                /*literal=*/255u, arch);
    // The compare consumes the following literal word. If the epoch is already
    // saturated, skip the two-word CDNA4-compatible increment sequence. This
    // needs no temporary SGPRs: in particular, the SCC snapshot at exec-save
    // +4 must remain intact until restore_scc.
    const auto skip_increment = instrumentation::build_s_cbranch_scc1(/*offset_dwords=*/2, arch);
    const auto restore_scc = instrumentation::build_s_cmp_lg_u32(
        special_state->scc_save_sgpr, scalar_positive_inline_u32(0), arch);
    if (!save_scc || !at_maximum || !skip_increment || !restore_scc) {
      errors.emplace_back("ConSan MOI scalar inline barrier could not encode epoch state");
      return std::nullopt;
    }
    std::vector<uint32_t> words;
    words.reserve(
        48u + (spill ? spill->save_words.size() + spill->restore_words.size() : 0u) +
        (scalar_spill ? scalar_spill->save_words.size() + scalar_spill->restore_words.size() : 0u));
    // The guest instruction after a patched control transfer may update the
    // VGPR-bank mode as the cave begins executing.  Establish the cave's
    // low-bank invariant even when the replaced barrier itself used mode 0.
    if (control_gfx1250_vgpr_bank)
      words.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
          static_cast<uint8_t>(gfx1250_vgpr_msb_mode.value_or(0u)), 0u, arch));
    if (spill)
      words.insert(words.end(), spill->save_words.begin(), spill->save_words.end());
    if (scalar_spill)
      words.insert(words.end(), scalar_spill->save_words.begin(), scalar_spill->save_words.end());
    if (publish_visible_evidence) {
      if (!append_save_moi_special_state(words, options, arch)) {
        errors.emplace_back("ConSan MOI inline barrier could not save visible-evidence state");
        return std::nullopt;
      }
    } else {
      words.push_back(*save_scc);
    }
    words.push_back(original_barrier_word);
    words.push_back(*at_maximum);
    words.push_back(consan_moi_exact_shadow::max_epoch);
    words.push_back(*skip_increment);
    if (!append_moi_entry_salu_write(words,
                                     build_s_add_u32(*options.moi_persistent_epoch_sgpr,
                                                     *options.moi_persistent_epoch_sgpr,
                                                     scalar_positive_inline_u32(1), arch),
                                     arch)) {
      errors.emplace_back("ConSan MOI scalar inline barrier could not encode saturating add");
      return std::nullopt;
    }
    if (publish_visible_evidence) {
      if (!append_visible_evidence(words) ||
          !append_restore_moi_special_state(words, options, arch)) {
        errors.emplace_back("ConSan MOI inline barrier could not publish visible evidence");
        return std::nullopt;
      }
    } else {
      words.push_back(*restore_scc);
    }
    if (scalar_spill)
      words.insert(words.end(), scalar_spill->restore_words.begin(),
                   scalar_spill->restore_words.end());
    if (spill)
      words.insert(words.end(), spill->restore_words.begin(), spill->restore_words.end());
    if (restore_guest_vgpr_bank)
      words.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
          0u, static_cast<uint8_t>(*gfx1250_vgpr_msb_mode), arch));
    const uint64_t branch_pc =
        cave_text_offset + static_cast<uint64_t>(words.size()) * sizeof(uint32_t);
    if (call_return_sgpr) {
      words.push_back(build_s_setpc_b64(*call_return_sgpr, arch));
    } else if (const auto ret = compute_sopp_branch_simm16(branch_pc, return_text_offset)) {
      words.push_back(build_s_branch(*ret, arch));
    } else {
      const auto jump_sgprs = moi_indirect_jump_sgprs(options);
      if (!jump_sgprs || !append_moi_scc_preserving_indirect_jump(
                             words, cave_text_offset, return_text_offset, jump_sgprs->pc_sgpr,
                             jump_sgprs->scc_save_sgpr, /*capture_scc=*/true, arch)) {
        errors.emplace_back("ConSan MOI scalar inline barrier could not encode long return");
        return std::nullopt;
      }
    }
    return words;
  }
  if (!options.moi_epoch_vgpr) {
    errors.emplace_back(
        "ConSan MOI inline-shadow barrier epoch patch requires RJ_CONSAN_MOI_EPOCH_VGPR");
    return std::nullopt;
  }

  const auto increment_epoch = instrumentation::build_v_add_u32(
      *options.moi_epoch_vgpr, scalar_positive_inline_u32(1), *options.moi_epoch_vgpr, arch);
  const auto saturate_epoch = instrumentation::build_v_min_u32_literal(
      *options.moi_epoch_vgpr, consan_moi_exact_shadow::max_epoch, *options.moi_epoch_vgpr, arch);
  if (!increment_epoch || !saturate_epoch) {
    errors.emplace_back(
        "ConSan MOI inline-shadow barrier epoch patch could not encode saturating epoch add");
    return std::nullopt;
  }

  std::vector<uint32_t> words;
  words.reserve(
      40u + (spill ? spill->save_words.size() + spill->restore_words.size() : 0u) +
      (scalar_spill ? scalar_spill->save_words.size() + scalar_spill->restore_words.size() : 0u) +
      (control_gfx1250_vgpr_bank ? 1u : 0u) + (restore_guest_vgpr_bank ? 1u : 0u));
  words.push_back(original_barrier_word);
  if (control_gfx1250_vgpr_bank)
    words.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
        static_cast<uint8_t>(gfx1250_vgpr_msb_mode.value_or(0u)), 0u, arch));
  if (spill)
    words.insert(words.end(), spill->save_words.begin(), spill->save_words.end());
  if (scalar_spill)
    words.insert(words.end(), scalar_spill->save_words.begin(), scalar_spill->save_words.end());
  if (publish_visible_evidence && !append_save_moi_special_state(words, options, arch)) {
    errors.emplace_back("ConSan MOI inline barrier could not save visible-evidence state");
    return std::nullopt;
  }
  words.insert(words.end(), increment_epoch->begin(), increment_epoch->end());
  words.insert(words.end(), saturate_epoch->begin(), saturate_epoch->end());
  if (publish_visible_evidence && (!append_visible_evidence(words) ||
                                   !append_restore_moi_special_state(words, options, arch))) {
    errors.emplace_back("ConSan MOI inline barrier could not publish visible evidence");
    return std::nullopt;
  }
  if (scalar_spill)
    words.insert(words.end(), scalar_spill->restore_words.begin(),
                 scalar_spill->restore_words.end());
  if (spill)
    words.insert(words.end(), spill->restore_words.begin(), spill->restore_words.end());
  if (restore_guest_vgpr_bank)
    words.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
        0u, static_cast<uint8_t>(*gfx1250_vgpr_msb_mode), arch));

  const uint64_t branch_pc =
      cave_text_offset + static_cast<uint64_t>(words.size()) * sizeof(uint32_t);
  const auto ret = compute_sopp_branch_simm16(branch_pc, return_text_offset);
  if (call_return_sgpr) {
    words.push_back(build_s_setpc_b64(*call_return_sgpr, arch));
  } else if (ret) {
    words.push_back(build_s_branch(*ret, arch));
  } else {
    const auto jump_sgprs = moi_indirect_jump_sgprs(options);
    if (!jump_sgprs || !append_moi_scc_preserving_indirect_jump(
                           words, cave_text_offset, return_text_offset, jump_sgprs->pc_sgpr,
                           jump_sgprs->scc_save_sgpr, /*capture_scc=*/true, arch)) {
      errors.emplace_back(
          "ConSan MOI inline-shadow barrier epoch could not encode long return at " +
          std::to_string(site.text_offset));
      return std::nullopt;
    }
  }
  return words;
}

[[nodiscard]] std::optional<std::vector<uint32_t>>
build_inline_shadow_private_epoch_barrier_cave_words(
    const ConSanBarrierSite &site, const ConSanOptions &options, uint16_t scratch_vgpr,
    uint32_t epoch_offset, const VgprSpillSequence &spill,
    const InlineSgprSpillSequence *scalar_spill, rj_code_arch_t arch,
    uint32_t original_barrier_word, uint64_t cave_text_offset, uint64_t return_text_offset,
    std::optional<uint16_t> call_return_sgpr, std::optional<uint16_t> gfx1250_vgpr_msb_mode,
    std::vector<std::string> &errors) {
  const auto load_epoch = instrumentation::build_private_load_b32(scratch_vgpr, epoch_offset, arch);
  const auto wait_load = instrumentation::build_s_wait_private_load0(arch);
  const auto increment_epoch = instrumentation::build_v_add_u32(
      scratch_vgpr, scalar_positive_inline_u32(1), scratch_vgpr, arch);
  const auto saturate_epoch = instrumentation::build_v_min_u32_literal(
      scratch_vgpr, consan_moi_exact_shadow::max_epoch, scratch_vgpr, arch);
  const auto store_epoch =
      instrumentation::build_private_store_b32(scratch_vgpr, epoch_offset, arch);
  const auto wait_store = instrumentation::build_s_wait_private_store0(arch);
  if (!load_epoch || !wait_load || !increment_epoch || !saturate_epoch || !store_epoch ||
      !wait_store) {
    errors.emplace_back("ConSan MOI inline-shadow barrier could not encode private epoch update");
    return std::nullopt;
  }

  std::vector<uint32_t> words;
  const bool control_gfx1250_vgpr_bank = arch == ROCJITSU_CODE_ARCH_GFX1250;
  const bool restore_guest_vgpr_bank =
      control_gfx1250_vgpr_bank && gfx1250_vgpr_msb_mode.value_or(0u) != 0u;
  words.reserve(
      2u + spill.save_words.size() + load_epoch->size() + store_epoch->size() +
      saturate_epoch->size() + spill.restore_words.size() +
      (scalar_spill ? scalar_spill->save_words.size() + scalar_spill->restore_words.size() : 0u) +
      3u + (control_gfx1250_vgpr_bank ? 1u : 0u) + (restore_guest_vgpr_bank ? 1u : 0u));
  words.push_back(original_barrier_word);
  if (control_gfx1250_vgpr_bank)
    words.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
        static_cast<uint8_t>(gfx1250_vgpr_msb_mode.value_or(0u)), 0u, arch));
  words.insert(words.end(), spill.save_words.begin(), spill.save_words.end());
  if (scalar_spill)
    words.insert(words.end(), scalar_spill->save_words.begin(), scalar_spill->save_words.end());
  words.insert(words.end(), load_epoch->begin(), load_epoch->end());
  words.push_back(*wait_load);
  words.insert(words.end(), increment_epoch->begin(), increment_epoch->end());
  words.insert(words.end(), saturate_epoch->begin(), saturate_epoch->end());
  words.insert(words.end(), store_epoch->begin(), store_epoch->end());
  words.push_back(*wait_store);
  if (scalar_spill)
    words.insert(words.end(), scalar_spill->restore_words.begin(),
                 scalar_spill->restore_words.end());
  words.insert(words.end(), spill.restore_words.begin(), spill.restore_words.end());
  if (restore_guest_vgpr_bank)
    words.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
        0u, static_cast<uint8_t>(*gfx1250_vgpr_msb_mode), arch));

  const uint64_t branch_pc =
      cave_text_offset + static_cast<uint64_t>(words.size()) * sizeof(uint32_t);
  const auto ret = compute_sopp_branch_simm16(branch_pc, return_text_offset);
  if (call_return_sgpr) {
    words.push_back(build_s_setpc_b64(*call_return_sgpr, arch));
  } else if (ret) {
    words.push_back(build_s_branch(*ret, arch));
  } else {
    const auto jump_sgprs = moi_indirect_jump_sgprs(options);
    if (!jump_sgprs || !append_moi_scc_preserving_indirect_jump(
                           words, cave_text_offset, return_text_offset, jump_sgprs->pc_sgpr,
                           jump_sgprs->scc_save_sgpr, /*capture_scc=*/true, arch)) {
      errors.emplace_back(
          "ConSan MOI inline-shadow private epoch could not encode long return at " +
          std::to_string(site.text_offset));
      return std::nullopt;
    }
  }
  return words;
}

void try_apply_inline_shadow_barrier_epoch_patch(std::span<const uint8_t> bytes,
                                                 const ConSanOptions &options, rj_code_arch_t arch,
                                                 MoiResourcePlanningState &resource_state,
                                                 ConSanResult &result) {
  if (!options.moi_track_barriers)
    return;
  if (!instrumentation::is_admitted_arch(arch)) {
    result.warnings.emplace_back(
        "ConSan MOI inline-shadow barrier epoch patch does not support this architecture");
    return;
  }
  if (!options.moi_epoch_vgpr && options.moi_persistent_vgpr_assignments.empty() &&
      !options.moi_persistent_epoch_sgpr && !options.automatic_moi_private_epoch) {
    result.warnings.emplace_back(
        "ConSan MOI inline-shadow barrier epoch patch has no persistent epoch state");
    return;
  }
  if (options.moi_epoch_vgpr && static_cast<uint32_t>(*options.moi_epoch_vgpr) + 1u > kMaxVgprs) {
    result.warnings.emplace_back(
        "ConSan MOI inline-shadow barrier epoch patch needs one epoch VGPR");
    return;
  }

  std::vector<BarrierRecordCandidate> candidates;
  for (const ConSanKernelInfo &kernel : result.kernels)
    append_barrier_epoch_candidates(kernel, candidates);
  for (const ConSanFunctionInfo &function : result.functions)
    append_barrier_epoch_candidates(function, candidates);
  if (options.moi_engine == ConSanMoiEngine::InlineShadow) {
    std::erase_if(candidates, [](const BarrierRecordCandidate &candidate) {
      return candidate.site.operation == ConSanBarrierSite::Operation::Signal;
    });
  }
  apply_test_container_filter(candidates, options);
  if (!canonicalize_barrier_record_candidates_by_physical_site(candidates, result))
    return;
  std::sort(candidates.begin(), candidates.end(), [](const auto &lhs, const auto &rhs) {
    return lhs.site.text_offset < rhs.site.text_offset;
  });
  if (candidates.empty()) {
    result.warnings.emplace_back("ConSan MOI inline-shadow barrier epoch patch found no barriers");
    return;
  }

  std::span<const uint8_t> active_bytes = bytes;
  if (result.modified)
    active_bytes = std::span<const uint8_t>(result.elf_bytes.data(), result.elf_bytes.size());
  AmdGpuCodeObject code_object(active_bytes.data(), active_bytes.size());
  CodeObjectPatcher patcher(code_object);
  const std::span<const uint8_t> old_text = patcher.text_bytes();
  if (old_text.empty()) {
    result.errors.emplace_back(
        "ConSan MOI inline-shadow barrier epoch patch found no .text section");
    return;
  }
  std::vector<BarrierRecordCandidate> selected_candidates;
  selected_candidates.reserve(options.max_patches);
  std::unordered_set<uint64_t> selected_offsets;
  const MoiPatchAnchorRangeIndex occupied_anchors(result);
  for (const BarrierRecordCandidate &candidate : candidates) {
    if (selected_candidates.size() == options.max_patches)
      break;
    const ConSanBarrierSite &site = candidate.site;
    if (site.size != sizeof(uint32_t)) {
      result.warnings.emplace_back(
          "ConSan MOI inline-shadow barrier epoch patch skipped non-32-bit barrier " +
          site.mnemonic + " in " + candidate.container_name);
      continue;
    }
    if (site.text_offset > old_text.size() || site.size > old_text.size() - site.text_offset) {
      result.errors.emplace_back("ConSan MOI inline-shadow barrier epoch site is outside .text");
      return;
    }
    if (occupied_anchors.overlaps(site.text_offset, site.size) ||
        !selected_offsets.insert(site.text_offset).second) {
      result.warnings.emplace_back(
          "ConSan MOI inline-shadow barrier epoch skipped an overlapping barrier patch");
      continue;
    }
    selected_candidates.push_back(candidate);
  }

  if (selected_candidates.empty()) {
    result.warnings.emplace_back(
        "ConSan MOI inline-shadow barrier epoch patch found no patchable barriers");
    return;
  }

  struct PlannedInlineEpochBarrier {
    BarrierRecordCandidate candidate;
    std::vector<uint64_t> owner_descriptor_file_offsets;
    std::optional<uint16_t> scratch_vgpr;
    std::optional<uint32_t> private_epoch_offset;
    std::optional<uint32_t> private_owner_offset;
    std::optional<uint16_t> private_owner_shift;
    ConSanMoiPersistentWorkgroupPrivateOffsets private_record_replay_workgroup_offsets;
    std::optional<VgprSpillSequence> spill;
    std::optional<InlineSgprSpillSequence> scalar_spill;
    uint32_t persistent_private_state_end = 0;
    uint32_t required_private_bytes = 0;
  };
  std::vector<PlannedInlineEpochBarrier> planned_candidates;
  MoiDescriptorPrivateRequirements private_requirements;
  MoiSpillManagers visible_evidence_spill_managers;
  MoiDescriptorVgprRequirements visible_evidence_vgpr_requirements;
  MoiDescriptorSgprRequirements visible_evidence_sgpr_requirements;
  MoiDescriptorPrivateRequirements visible_evidence_private_requirements;
  if (options.automatic_moi_private_epoch) {
    for (const BarrierRecordCandidate &candidate : selected_candidates) {
      const ConSanCandidateResourcePlan *plan = resource_plan_for_site(
          result, ConSanResourceSiteKind::Barrier, candidate.site.text_offset);
      if (plan == nullptr || plan->owner_descriptor_file_offsets.empty()) {
        result.warnings.emplace_back(
            "ConSan MOI private epoch barrier has no owned resource scope in " +
            candidate.container_name);
        continue;
      }
      const std::vector<uint64_t> &owners = plan->owner_descriptor_file_offsets;
      const auto owns_every_descriptor = [&](const ConSanPatchInfo &patch) {
        return std::ranges::all_of(owners, [&](uint64_t descriptor_offset) {
          return std::ranges::find(patch.owner_descriptor_file_offsets, descriptor_offset) !=
                 patch.owner_descriptor_file_offsets.end();
        });
      };
      const auto access_patch =
          std::ranges::find_if(result.patches, [&](const ConSanPatchInfo &patch) {
            return (patch.kind == ConSanPatchKind::InlineMoiExactShadowStore ||
                    patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore ||
                    patch.kind == ConSanPatchKind::InlineMoiAccessRecordStore ||
                    patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore) &&
                   patch.scratch_vgpr && patch.persistent_epoch_private_offset &&
                   owns_every_descriptor(patch);
          });
      if (access_patch == result.patches.end()) {
        result.warnings.emplace_back(
            "ConSan MOI private epoch barrier has no common access-probe state in " +
            candidate.container_name);
        continue;
      }
      if (options.moi_engine == ConSanMoiEngine::RecordReplay) {
        const auto resources = resolve_moi_scratch_plan(*plan, options, /*expected_count=*/6u);
        if (!resources ||
            !access_patch->persistent_record_replay_workgroup_private_offsets.complete()) {
          result.warnings.emplace_back(
              "ConSan MOI Record/Replay dense barrier has no complete private exact-workgroup "
              "layout in " +
              candidate.container_name);
          continue;
        }
        ConSanOptions candidate_options = options;
        candidate_options.scratch_vgpr = resources->base;
        candidate_options.moi_record_replay_workgroup_private_offsets =
            access_patch->persistent_record_replay_workgroup_private_offsets;
        if (!apply_moi_transient_sgpr_assignment(candidate_options, owners) ||
            !apply_record_replay_entry_workgroup_assignment(candidate_options, owners)) {
          result.warnings.emplace_back(
              "ConSan MOI Record/Replay dense barrier lost its private owner-local assignment");
          continue;
        }
        std::optional<uint16_t> private_owner_shift;
        if (access_patch->persistent_owner_private_offset) {
          private_owner_shift = common_moi_private_owner_shift(bytes, *resources, result.warnings);
          if (!private_owner_shift)
            continue;
        }
        const uint32_t persistent_end = access_patch->persistent_private_state_end.value_or(0u);
        std::optional<VgprSpillSequence> spill;
        if (resources->source == ConSanRegisterAllocationSource::SpillRequired ||
            moi_scalar_spill_requires_dynamic_vgpr_frame(result, *resources, candidate_options)) {
          spill = build_moi_spill_sequence(result, *resources, candidate_options,
                                           visible_evidence_spill_managers, arch, result.warnings,
                                           persistent_end);
          if (!spill)
            continue;
        }
        std::optional<InlineSgprSpillSequence> scalar_spill;
        if (candidate_options.automatic_moi_record_replay_sgpr_spill) {
          scalar_spill = build_inline_sgpr_spill_sequence(
              result, *resources, candidate_options, visible_evidence_spill_managers, arch,
              result.warnings, persistent_end, spill ? &*spill : nullptr);
          if (!scalar_spill)
            continue;
        }
        uint32_t required_private_bytes = persistent_end;
        if (spill)
          required_private_bytes = std::max(required_private_bytes, spill->total_private_bytes);
        if (scalar_spill) {
          required_private_bytes =
              std::max(required_private_bytes, scalar_spill->total_private_bytes);
        }
        for (uint64_t descriptor_offset : owners) {
          auto [it, inserted] =
              private_requirements.emplace(descriptor_offset, required_private_bytes);
          if (!inserted)
            it->second = std::max(it->second, required_private_bytes);
        }
        planned_candidates.push_back(PlannedInlineEpochBarrier{
            .candidate = candidate,
            .owner_descriptor_file_offsets = owners,
            .scratch_vgpr = resources->base,
            .private_epoch_offset = access_patch->persistent_epoch_private_offset,
            .private_owner_offset = access_patch->persistent_owner_private_offset,
            .private_owner_shift = private_owner_shift,
            .private_record_replay_workgroup_offsets =
                access_patch->persistent_record_replay_workgroup_private_offsets,
            .spill = std::move(spill),
            .scalar_spill = std::move(scalar_spill),
            .persistent_private_state_end = persistent_end,
            .required_private_bytes = required_private_bytes,
        });
        continue;
      }
      const uint32_t ephemeral_base = access_patch->persistent_private_state_end.value_or(
          *access_patch->persistent_epoch_private_offset + SpillManager::kSlotBytes);
      const auto private_limit = address_free_scratch_private_limit(arch);
      if (!private_limit) {
        result.warnings.emplace_back(
            "ConSan MOI private epoch barrier has no address-free scratch support");
        continue;
      }
      SpillManager manager(ephemeral_base, *private_limit);
      auto spill = build_vgpr_spill_sequence(manager, *access_patch->scratch_vgpr,
                                             /*vgpr_count=*/1, arch);
      if (!spill) {
        result.warnings.emplace_back(
            "ConSan MOI private epoch barrier could not preserve its temporary VGPR");
        continue;
      }
      const uint32_t required_private_bytes =
          std::max(access_patch->required_private_segment_size, spill->total_private_bytes);
      for (uint64_t descriptor_offset : owners) {
        auto [it, inserted] =
            private_requirements.emplace(descriptor_offset, required_private_bytes);
        if (!inserted)
          it->second = std::max(it->second, required_private_bytes);
      }
      planned_candidates.push_back(PlannedInlineEpochBarrier{
          .candidate = candidate,
          .owner_descriptor_file_offsets = owners,
          .scratch_vgpr = access_patch->scratch_vgpr,
          .private_epoch_offset = access_patch->persistent_epoch_private_offset,
          .private_owner_offset = std::nullopt,
          .private_owner_shift = std::nullopt,
          .private_record_replay_workgroup_offsets = {},
          .spill = std::move(spill),
          .scalar_spill = std::nullopt,
          .persistent_private_state_end = 0u,
          .required_private_bytes = required_private_bytes,
      });
    }
    if (!apply_spill_descriptor_requirements(patcher, code_object, active_bytes, result,
                                             private_requirements, result.errors))
      return;
  } else if (options.automatic_moi_persistent_sgprs) {
    // The owner-kernel entry prologue grows each descriptor for this scalar
    // tuple after all emitted patch ownership is known.
  } else if (options.automatic_moi_persistent_vgprs) {
    MoiDescriptorVgprRequirements requirements;
    for (const BarrierRecordCandidate &candidate : selected_candidates) {
      const ConSanCandidateResourcePlan *plan = resource_plan_for_site(
          result, ConSanResourceSiteKind::Barrier, candidate.site.text_offset);
      if (plan == nullptr || plan->owner_descriptor_file_offsets.empty()) {
        result.warnings.emplace_back(
            "ConSan MOI inline barrier has no owned automatic persistent-state scope");
        continue;
      }
      for (uint64_t descriptor_offset : plan->owner_descriptor_file_offsets) {
        const auto *assignment = find_moi_persistent_vgpr_assignment(options, descriptor_offset);
        const uint16_t required = assignment ? static_cast<uint16_t>(assignment->epoch_vgpr + 1u)
                                             : static_cast<uint16_t>(*options.moi_epoch_vgpr + 1u);
        requirements[descriptor_offset] = required;
      }
    }
    if (!apply_descriptor_requirements(patcher, code_object, active_bytes, result, requirements,
                                       arch, result.errors)) {
      return;
    }
  } else {
    MoiDescriptorVgprRequirements requirements;
    for (const ConSanKernelInfo &kernel : result.kernels) {
      requirements.emplace(kernel.descriptor_file_offset,
                           static_cast<uint32_t>(*options.moi_epoch_vgpr) + 1u);
    }
    if (!apply_descriptor_requirements(patcher, code_object, active_bytes, result, requirements,
                                       arch, result.errors)) {
      return;
    }
  }
  if (!options.automatic_moi_private_epoch) {
    for (const BarrierRecordCandidate &candidate : selected_candidates) {
      std::vector<uint64_t> owners;
      const ConSanCandidateResourcePlan *plan = resource_plan_for_site(
          result, ConSanResourceSiteKind::Barrier, candidate.site.text_offset);
      if (plan) {
        owners = plan->owner_descriptor_file_offsets;
      }
      if (owners.empty() && candidate.kernel_descriptor_file_offset)
        owners.push_back(*candidate.kernel_descriptor_file_offset);
      if (automatic_moi_persistent_state(options) && owners.empty())
        continue;
      ConSanOptions candidate_options = options;
      if (!apply_moi_transient_sgpr_assignment(candidate_options, owners))
        continue;
      candidate_options.moi_inline_workgroup_shadow =
          options.moi_inline_workgroup_shadow &&
          std::ranges::any_of(result.patches, [&](const ConSanPatchInfo &patch) {
            return patch.workgroup_shadow_size != 0u &&
                   std::ranges::any_of(owners, [&](uint64_t descriptor_offset) {
                     return std::ranges::find(patch.owner_descriptor_file_offsets,
                                              descriptor_offset) !=
                            patch.owner_descriptor_file_offsets.end();
                   });
          });
      std::optional<uint16_t> scratch_vgpr;
      std::optional<VgprSpillSequence> spill;
      std::optional<InlineSgprSpillSequence> scalar_spill;
      uint32_t required_private_bytes = 0u;
      const uint16_t scratch_count = inline_shadow_barrier_scratch_count(candidate_options);
      const bool needs_record_replay_scalar_spill =
          candidate_options.automatic_moi_record_replay_sgpr_spill;
      const bool needs_scalar_spill =
          needs_record_replay_scalar_spill ||
          (scratch_count > 1u && candidate_options.automatic_moi_inline_sgpr_spill);
      if (scratch_count > 1u || needs_scalar_spill) {
        if (plan == nullptr) {
          result.warnings.emplace_back(
              "ConSan MOI inline barrier visible evidence has no resource plan in " +
              candidate.container_name);
          continue;
        }
        const auto resources = resolve_moi_scratch_plan(*plan, candidate_options, scratch_count);
        if (!resources) {
          result.warnings.emplace_back(
              "ConSan MOI inline barrier visible evidence has no usable resource plan in " +
              candidate.container_name);
          continue;
        }
        // A one-VGPR Record/Replay resource can be used solely as the scalar
        // spill transfer register. It is not Inline visible-evidence scratch:
        // passing it as such would select Inline's much wider +12/+14 SGPR
        // layout and reject an otherwise valid compact Record/Replay probe.
        if (scratch_count > 1u)
          scratch_vgpr = resources->base;
        owners = resources->owner_descriptor_file_offsets;
        if (resources->source == ConSanRegisterAllocationSource::SpillRequired ||
            moi_scalar_spill_requires_dynamic_vgpr_frame(result, *resources, candidate_options)) {
          spill = build_moi_spill_sequence(result, *resources, candidate_options,
                                           visible_evidence_spill_managers, arch, result.warnings);
          if (!spill)
            continue;
          required_private_bytes = spill->total_private_bytes;
          note_spill_descriptor_requirements(visible_evidence_private_requirements, *resources,
                                             *spill);
        }
        if (needs_scalar_spill) {
          scalar_spill = build_inline_sgpr_spill_sequence(
              result, *resources, candidate_options, visible_evidence_spill_managers, arch,
              result.warnings, std::nullopt, spill ? &*spill : nullptr);
          if (!scalar_spill)
            continue;
          required_private_bytes =
              std::max(required_private_bytes, scalar_spill->total_private_bytes);
          for (uint64_t descriptor_offset : resources->owner_descriptor_file_offsets) {
            auto [it, inserted] = visible_evidence_private_requirements.emplace(
                descriptor_offset, scalar_spill->total_private_bytes);
            if (!inserted)
              it->second = std::max(it->second, scalar_spill->total_private_bytes);
          }
        }
        note_descriptor_requirements(visible_evidence_vgpr_requirements, *resources);
        note_moi_sgpr_requirements(visible_evidence_sgpr_requirements, *resources,
                                   candidate_options, arch);
      }
      planned_candidates.push_back(PlannedInlineEpochBarrier{
          .candidate = candidate,
          .owner_descriptor_file_offsets = std::move(owners),
          .scratch_vgpr = scratch_vgpr,
          .private_epoch_offset = std::nullopt,
          .private_owner_offset = std::nullopt,
          .private_owner_shift = std::nullopt,
          .private_record_replay_workgroup_offsets = {},
          .spill = std::move(spill),
          .scalar_spill = std::move(scalar_spill),
          .persistent_private_state_end = 0u,
          .required_private_bytes = required_private_bytes,
      });
    }
  }
  if (planned_candidates.empty()) {
    result.warnings.emplace_back(
        "ConSan MOI inline-shadow barrier epoch patch found no compatible persistent state");
    return;
  }
  if (!apply_descriptor_requirements(patcher, code_object, active_bytes, result,
                                     visible_evidence_vgpr_requirements, arch, result.errors) ||
      !apply_spill_descriptor_requirements(patcher, code_object, active_bytes, result,
                                           visible_evidence_private_requirements, result.errors) ||
      !apply_sgpr_descriptor_requirements(patcher, result, code_object,
                                          visible_evidence_sgpr_requirements, result.errors)) {
    return;
  }

  std::vector<uint8_t> new_text(old_text.begin(), old_text.end());
  DbiPatchPlacementPlanner placement_planner(arch, old_text.size());
  MoiLocalNopIslandAllocator local_indirect_allocator(old_text, result, arch,
                                                      kMoiLocalIndirectIslandWords);
  std::vector<ConSanPatchInfo> patches;
  patches.reserve(planned_candidates.size() * 2u);
  constexpr uint64_t kIndirectIslandWords = 7u;
  const uint64_t indirect_island_bytes = kIndirectIslandWords * sizeof(uint32_t);
  std::optional<uint64_t> barrier_island_begin;
  if (options.moi_exec_save_sgpr) {
    uint64_t first_access_body = std::numeric_limits<uint64_t>::max();
    uint64_t access_island_begin = std::numeric_limits<uint64_t>::max();
    uint64_t access_island_end = 0u;
    for (const ConSanPatchInfo &patch : result.patches) {
      if (patch.kind == ConSanPatchKind::InlineMoiExactShadowStore ||
          patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore ||
          patch.kind == ConSanPatchKind::InlineMoiAccessRecordStore ||
          patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore) {
        first_access_body = std::min(first_access_body, patch.trampoline_offset);
      } else if (patch.kind == ConSanPatchKind::TrampolineMoiIndirectBranchIsland) {
        access_island_begin = std::min(access_island_begin, patch.trampoline_offset);
        access_island_end =
            std::max(access_island_end,
                     patch.trampoline_offset + static_cast<uint64_t>(patch.trampoline_size));
      }
    }
    const uint64_t required_island_bytes = planned_candidates.size() * indirect_island_bytes;
    const auto is_reserved_nop_range = [&](uint64_t candidate_begin) {
      bool all_reserved_nops = candidate_begin + required_island_bytes <= old_text.size();
      for (uint64_t offset = candidate_begin;
           all_reserved_nops && offset < candidate_begin + required_island_bytes;
           offset += sizeof(uint32_t)) {
        uint32_t word = 0;
        std::memcpy(&word, old_text.data() + offset, sizeof(word));
        all_reserved_nops = word == build_s_nop(0, arch);
      }
      return all_reserved_nops;
    };
    // Keep barrier islands first: early synchronization sites are more likely
    // to be near the start of a large generated object, while sorted access
    // islands retain the same monotonic anchor-to-island mapping after them.
    if (access_island_begin != std::numeric_limits<uint64_t>::max() &&
        access_island_begin >= required_island_bytes) {
      const uint64_t candidate_begin = access_island_begin - required_island_bytes;
      if (is_reserved_nop_range(candidate_begin))
        barrier_island_begin = candidate_begin;
    }
    // Accept the older trailing reservation layout as well so independently
    // composed exact-shadow producers remain compatible.
    if (!barrier_island_begin && first_access_body != std::numeric_limits<uint64_t>::max() &&
        first_access_body >= required_island_bytes &&
        first_access_body - required_island_bytes >= access_island_end) {
      const uint64_t candidate_begin = first_access_body - required_island_bytes;
      if (is_reserved_nop_range(candidate_begin))
        barrier_island_begin = candidate_begin;
    }
  }
  if (barrier_island_begin) {
    local_indirect_allocator.reserve(
        *barrier_island_begin,
        static_cast<uint32_t>(planned_candidates.size() * kIndirectIslandWords));
  }
  std::vector<size_t> dense_fallback_indices;
  const ConSanMoiReportBufferLayout record_layout = resolve_record_replay_report_layout(
      options, /*include_barriers=*/true, options.moi_track_atomics, options.moi_track_atomics);
  const auto build_engine_barrier_body =
      [&](const PlannedInlineEpochBarrier &planned, ConSanOptions candidate_options,
          uint32_t original_barrier_word, uint64_t body_offset, uint64_t return_offset,
          std::optional<uint16_t> call_return_sgpr,
          std::optional<uint16_t> gfx1250_vgpr_msb_mode) -> std::optional<std::vector<uint32_t>> {
    if (options.moi_engine == ConSanMoiEngine::RecordReplay) {
      if (!planned.scratch_vgpr) {
        result.warnings.emplace_back(
            "ConSan MOI Record/Replay dense barrier fallback requires scratch VGPR state");
        return std::nullopt;
      }
      candidate_options.scratch_vgpr = planned.scratch_vgpr;
      if (planned.private_epoch_offset) {
        candidate_options.moi_record_replay_workgroup_private_offsets =
            planned.private_record_replay_workgroup_offsets;
      }
      if (!apply_moi_transient_sgpr_assignment(candidate_options,
                                               planned.owner_descriptor_file_offsets) ||
          !apply_record_replay_entry_workgroup_assignment(candidate_options,
                                                          planned.owner_descriptor_file_offsets)) {
        result.warnings.emplace_back(
            "ConSan MOI Record/Replay dense barrier fallback lost its owner-local assignment");
        return std::nullopt;
      }
      return build_barrier_record_cave_words(
          active_bytes, planned.candidate, candidate_options,
          planned.spill ? &*planned.spill : nullptr,
          planned.scalar_spill ? &*planned.scalar_spill : nullptr, arch,
          record_layout.barrier_record_capacity, record_layout.barrier_records_offset,
          original_barrier_word, body_offset, return_offset, planned.private_owner_offset,
          planned.private_owner_shift, result.errors);
    }
    if (planned.private_epoch_offset && planned.scratch_vgpr && planned.spill) {
      return build_inline_shadow_private_epoch_barrier_cave_words(
          planned.candidate.site, candidate_options, *planned.scratch_vgpr,
          *planned.private_epoch_offset, *planned.spill,
          planned.scalar_spill ? &*planned.scalar_spill : nullptr, arch, original_barrier_word,
          body_offset, return_offset, call_return_sgpr, gfx1250_vgpr_msb_mode, result.errors);
    }
    return build_inline_shadow_barrier_epoch_cave_words(
        planned.candidate.site, candidate_options, arch, planned.scratch_vgpr,
        planned.spill ? &*planned.spill : nullptr,
        planned.scalar_spill ? &*planned.scalar_spill : nullptr, original_barrier_word, body_offset,
        return_offset, call_return_sgpr, gfx1250_vgpr_msb_mode, result.errors);
  };

  for (size_t planned_index = 0; planned_index < planned_candidates.size(); ++planned_index) {
    const PlannedInlineEpochBarrier &planned = planned_candidates[planned_index];
    const BarrierRecordCandidate &candidate = planned.candidate;
    const ConSanBarrierSite &site = candidate.site;
    uint32_t original_barrier_word = 0;
    std::memcpy(&original_barrier_word, new_text.data() + site.text_offset,
                sizeof(original_barrier_word));
    const std::optional<uint16_t> gfx1250_vgpr_msb_mode =
        arch == ROCJITSU_CODE_ARCH_GFX1250
            ? gfx1250_vgpr_msb_mode_at(active_bytes, candidate.text_file_offset,
                                       candidate.container_entry_text_offset, site.file_offset)
            : std::nullopt;
    ConSanOptions candidate_options = options;
    if (!apply_moi_transient_sgpr_assignment(candidate_options,
                                             planned.owner_descriptor_file_offsets)) {
      result.errors.emplace_back("ConSan MOI inline barrier lost its transient scalar assignment");
      return;
    }
    std::optional<uint64_t> oversized_local_body_island;
    if (!planned.private_epoch_offset && !candidate_options.moi_persistent_epoch_sgpr &&
        !apply_moi_persistent_vgpr_assignment(candidate_options,
                                              planned.owner_descriptor_file_offsets)) {
      result.errors.emplace_back("ConSan MOI inline barrier lost its persistent VGPR assignment");
      return;
    }

    // The persistent-VGPR epoch update is small enough to live entirely in a
    // local NOP island. Prefer that layout over an indirect island followed by
    // an appended body: besides being smaller, it avoids perturbing SCC and
    // consuming temporary SGPR state on both the entry and return paths. The
    // private-epoch form is larger and continues to use the general appended
    // placement below.
    if (!planned.private_epoch_offset && !candidate_options.moi_persistent_epoch_sgpr) {
      if (const auto local_body_offset =
              local_indirect_allocator.claim_reachable(site.text_offset)) {
        const uint64_t return_text_offset = site.text_offset + site.size;
        auto local_words = build_engine_barrier_body(
            planned, candidate_options, original_barrier_word, *local_body_offset,
            return_text_offset, std::nullopt, gfx1250_vgpr_msb_mode);
        if (!local_words)
          return;
        if (local_words->size() <= kMoiLocalIndirectIslandWords &&
            *local_body_offset + local_words->size() * sizeof(uint32_t) <= new_text.size()) {
          const auto fwd = compute_sopp_branch_simm16(site.text_offset, *local_body_offset);
          if (!fwd || !write_word_bytes(new_text, site.text_offset, build_s_branch(*fwd, arch))) {
            result.errors.emplace_back(
                "ConSan MOI inline-shadow barrier epoch could not enter its local body");
            return;
          }
          std::memcpy(new_text.data() + *local_body_offset, local_words->data(),
                      local_words->size() * sizeof(uint32_t));

          ConSanPatchInfo info;
          info.kind = options.moi_engine == ConSanMoiEngine::RecordReplay
                          ? ConSanPatchKind::TrampolineMoiBarrierRecord
                          : ConSanPatchKind::TrampolineMoiInlineEpochBarrier;
          info.anchor_offset = site.text_offset;
          info.trampoline_offset = *local_body_offset;
          info.original_size = site.size;
          info.trampoline_size = static_cast<uint32_t>(local_words->size() * sizeof(uint32_t));
          info.relocated_guest_instruction_offset =
              *local_body_offset +
              (candidate_options.moi_persistent_epoch_sgpr ? sizeof(uint32_t) : 0u);
          info.owner_descriptor_file_offsets = planned.owner_descriptor_file_offsets;
          patches.push_back(std::move(info));
          continue;
        }
        // gfx1250 may need VGPR-bank setup around this otherwise-small body.
        // A claimed short island is therefore only an optimization; if the
        // final body does not fit, retain the island as the indirect entry to
        // the general appended body below.
        oversized_local_body_island = *local_body_offset;
      }
    }

    const uint64_t cave_text_offset = placement_planner.appended_end();
    const uint64_t return_text_offset = site.text_offset + site.size;
    std::optional<std::vector<uint32_t>> cave_words = build_engine_barrier_body(
        planned, candidate_options, original_barrier_word, cave_text_offset, return_text_offset,
        std::nullopt, gfx1250_vgpr_msb_mode);
    if (!cave_words)
      return;

    std::optional<uint64_t> island_offset =
        barrier_island_begin
            ? std::optional<uint64_t>(*barrier_island_begin + planned_index * indirect_island_bytes)
            : std::nullopt;
    if (island_offset && !compute_sopp_branch_simm16(site.text_offset, *island_offset))
      island_offset.reset();
    if (!island_offset &&
        !compute_sopp_branch_simm16(site.text_offset, cave_text_offset).has_value()) {
      island_offset = oversized_local_body_island
                          ? oversized_local_body_island
                          : local_indirect_allocator.claim_reachable(site.text_offset);
      if (!island_offset) {
        const bool has_spill_backed_inline_router =
            candidate_options.automatic_moi_inline_sgpr_spill &&
            moi_indirect_jump_sgprs(candidate_options) &&
            candidate_options.moi_inline_visible_evidence_sgpr;
        const bool has_spill_backed_record_replay_router =
            candidate_options.automatic_moi_record_replay_sgpr_spill &&
            moi_indirect_jump_sgprs(candidate_options) &&
            candidate_options.moi_record_replay_dispatch_key_sgpr &&
            candidate_options.moi_record_replay_call_return_sgpr;
        const bool has_spill_backed_dense_router =
            has_spill_backed_inline_router || has_spill_backed_record_replay_router;
        const bool has_dense_router =
            arch == ROCJITSU_CODE_ARCH_GFX1250 ||
            (arch == ROCJITSU_CODE_ARCH_RDNA4 &&
             (candidate_options.moi_engine == ConSanMoiEngine::InlineShadow ||
              candidate_options.moi_engine == ConSanMoiEngine::RecordReplay)) ||
            ((arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA3 ||
              arch == ROCJITSU_CODE_ARCH_CDNA4) &&
             candidate_options.moi_engine == ConSanMoiEngine::RecordReplay &&
             candidate_options.moi_record_replay_dense_barrier_router);
        if (has_dense_router &&
            (candidate_options.moi_exec_save_sgpr || has_spill_backed_dense_router) &&
            !planned.owner_descriptor_file_offsets.empty()) {
          dense_fallback_indices.push_back(planned_index);
        } else {
          result.warnings.emplace_back(
              "ConSan MOI inline-shadow barrier epoch found no reachable indirect entry island "
              "in " +
              candidate.container_name);
        }
        continue;
      }
    }
    std::optional<DbiPatchPlacement> placement;
    if (island_offset) {
      std::string placement_error;
      placement = placement_planner.plan_indirect_appended(
          site.text_offset, site.size, cave_words->size() * sizeof(uint32_t), &placement_error);
      if (!placement)
        result.errors.emplace_back("ConSan MOI inline-shadow barrier epoch placement failed: " +
                                   placement_error);
    } else {
      placement = plan_prebuilt_appended_cave(placement_planner, site.text_offset, site.size,
                                              *cave_words, result.errors,
                                              "ConSan MOI inline-shadow barrier epoch patch");
    }
    if (!placement || placement->body_offset != cave_text_offset ||
        new_text.size() != placement->body_offset) {
      result.errors.emplace_back(
          "ConSan MOI inline-shadow barrier epoch patch has a stale cave mapping");
      return;
    }

    const uint64_t entry_target = island_offset.value_or(placement->body_offset);
    const auto fwd = compute_sopp_branch_simm16(placement->anchor_offset, entry_target);
    if (!fwd) {
      result.errors.emplace_back(
          "ConSan MOI inline-shadow barrier epoch forward branch is out of range");
      return;
    }
    if (!write_word_bytes(new_text, site.text_offset, build_s_branch(*fwd, arch))) {
      result.errors.emplace_back(
          "ConSan MOI inline-shadow barrier epoch patch could not rewrite barrier slot");
      return;
    }
    if (island_offset) {
      const auto jump_sgprs = moi_indirect_jump_sgprs(candidate_options);
      std::vector<uint32_t> island_words;
      if (!jump_sgprs ||
          !append_moi_scc_preserving_indirect_jump(
              island_words, *island_offset, placement->body_offset, jump_sgprs->pc_sgpr,
              jump_sgprs->scc_save_sgpr, /*capture_scc=*/true, arch) ||
          island_words.size() != kIndirectIslandWords ||
          *island_offset + indirect_island_bytes > new_text.size()) {
        result.errors.emplace_back(
            "ConSan MOI inline-shadow barrier epoch could not encode its entry island");
        return;
      }
      std::memcpy(new_text.data() + *island_offset, island_words.data(), indirect_island_bytes);
      ConSanPatchInfo island_info;
      island_info.kind = ConSanPatchKind::TrampolineMoiIndirectBranchIsland;
      island_info.anchor_offset = site.text_offset;
      island_info.trampoline_offset = *island_offset;
      island_info.trampoline_size = static_cast<uint32_t>(indirect_island_bytes);
      island_info.owner_descriptor_file_offsets = planned.owner_descriptor_file_offsets;
      patches.push_back(std::move(island_info));
    }
    append_words_bytes(new_text, *cave_words);

    ConSanPatchInfo info;
    info.kind = options.moi_engine == ConSanMoiEngine::RecordReplay
                    ? ConSanPatchKind::TrampolineMoiBarrierRecord
                    : ConSanPatchKind::TrampolineMoiInlineEpochBarrier;
    info.anchor_offset = site.text_offset;
    info.trampoline_offset = cave_text_offset;
    info.original_size = site.size;
    info.trampoline_size = static_cast<uint32_t>(cave_words->size() * sizeof(uint32_t));
    if (options.moi_engine == ConSanMoiEngine::RecordReplay) {
      const auto guest =
          std::ranges::find(cave_words->rbegin(), cave_words->rend(), original_barrier_word);
      if (guest != cave_words->rend()) {
        info.relocated_guest_instruction_offset =
            cave_text_offset +
            static_cast<uint64_t>(std::distance(guest, cave_words->rend()) - 1u) * sizeof(uint32_t);
      }
    } else {
      info.relocated_guest_instruction_offset =
          cave_text_offset + (candidate_options.moi_persistent_epoch_sgpr ? sizeof(uint32_t) : 0u);
    }
    info.scratch_vgpr = planned.scratch_vgpr;
    info.persistent_epoch_private_offset = planned.private_epoch_offset;
    info.persistent_owner_private_offset = planned.private_owner_offset;
    info.persistent_record_replay_workgroup_private_offsets =
        planned.private_record_replay_workgroup_offsets;
    if (planned.persistent_private_state_end != 0u)
      info.persistent_private_state_end = planned.persistent_private_state_end;
    info.required_private_segment_size = planned.required_private_bytes;
    info.owner_descriptor_file_offsets = planned.owner_descriptor_file_offsets;
    if (planned.spill) {
      info.spilled_vgpr_count = planned.spill->vgpr_count;
      note_dynamic_stack_private_requirement(info, &*planned.spill);
    }
    patches.push_back(info);
  }

  if (!dense_fallback_indices.empty()) {
    constexpr uint64_t kDispatcherWordsPerSite = 20u;

    std::map<std::pair<bool, std::string>, std::vector<size_t>> owner_groups;
    for (size_t index : dense_fallback_indices) {
      const BarrierRecordCandidate &candidate = planned_candidates[index].candidate;
      owner_groups[{candidate.in_kernel, candidate.container_name}].push_back(index);
    }
    constexpr uint64_t kMaxCommonRelayAnchorSpan =
        static_cast<uint64_t>(static_cast<int64_t>(std::numeric_limits<int16_t>::max()) -
                              static_cast<int64_t>(std::numeric_limits<int16_t>::min())) *
        sizeof(uint32_t);
    std::vector<std::pair<std::pair<bool, std::string>, std::vector<size_t>>> groups;
    for (auto &[owner_key, owner_indices] : owner_groups) {
      std::ranges::sort(owner_indices, [&](size_t lhs, size_t rhs) {
        return planned_candidates[lhs].candidate.site.text_offset <
               planned_candidates[rhs].candidate.site.text_offset;
      });
      for (size_t index : owner_indices) {
        const uint64_t anchor = planned_candidates[index].candidate.site.text_offset;
        if (groups.empty() || groups.back().first != owner_key ||
            anchor - planned_candidates[groups.back().second.front()].candidate.site.text_offset >
                kMaxCommonRelayAnchorSpan) {
          groups.push_back({owner_key, {}});
        }
        groups.back().second.push_back(index);
      }
    }
    std::vector<std::pair<uint64_t, uint64_t>> occupied_ranges;
    occupied_ranges.reserve(result.patches.size() + patches.size() + selected_candidates.size());
    const auto append_occupied_range = [&](uint64_t begin, uint64_t size) {
      if (size != 0u && begin <= std::numeric_limits<uint64_t>::max() - size)
        occupied_ranges.emplace_back(begin, begin + size);
    };
    for (const ConSanPatchInfo &patch : result.patches)
      append_occupied_range(patch.anchor_offset, patch.original_size);
    for (const ConSanPatchInfo &patch : patches)
      append_occupied_range(patch.anchor_offset, patch.original_size);
    for (const BarrierRecordCandidate &candidate : selected_candidates)
      append_occupied_range(candidate.site.text_offset, candidate.site.size);
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
    std::map<uint64_t, uint64_t> claimed_hosts;
    const auto overlaps_existing = [&](uint64_t begin, uint64_t end) {
      const auto occupied_after =
          std::ranges::lower_bound(occupied_ranges, end, {}, &std::pair<uint64_t, uint64_t>::first);
      if (occupied_after != occupied_ranges.begin() && std::prev(occupied_after)->second > begin)
        return true;
      const auto claimed_after = claimed_hosts.lower_bound(end);
      return claimed_after != claimed_hosts.begin() && std::prev(claimed_after)->second > begin;
    };

    struct ReusableDenseAccessRelay {
      uint64_t island_offset = 0;
      size_t dispatcher_patch_index = 0;
      uint64_t reservation_end = 0;
    };
    std::vector<uint64_t> access_relay_boundaries;
    access_relay_boundaries.reserve(result.patches.size());
    for (const ConSanPatchInfo &patch : result.patches) {
      if (patch.trampoline_size != 0u)
        access_relay_boundaries.push_back(patch.trampoline_offset);
    }
    std::ranges::sort(access_relay_boundaries);
    const auto unique_boundary_end = std::ranges::unique(access_relay_boundaries);
    access_relay_boundaries.erase(unique_boundary_end.begin(), unique_boundary_end.end());

    std::map<std::vector<uint64_t>, std::vector<ReusableDenseAccessRelay>> reusable_access_relays;
    if (arch == ROCJITSU_CODE_ARCH_GFX1250) {
      for (size_t patch_index = 0; patch_index < result.patches.size(); ++patch_index) {
        const ConSanPatchInfo &dispatcher = result.patches[patch_index];
        if (dispatcher.kind != ConSanPatchKind::TrampolineMoiIndirectBranchIsland ||
            dispatcher.original_size != 0u || !dispatcher.moi_dense_entry_island_offset ||
            dispatcher.owner_descriptor_file_offsets.empty()) {
          continue;
        }
        const auto boundary =
            std::ranges::upper_bound(access_relay_boundaries, dispatcher.trampoline_offset);
        if (boundary == access_relay_boundaries.end() ||
            dispatcher.trampoline_offset + dispatcher.trampoline_size > *boundary) {
          continue;
        }
        reusable_access_relays[dispatcher.owner_descriptor_file_offsets].push_back(
            {.island_offset = *dispatcher.moi_dense_entry_island_offset,
             .dispatcher_patch_index = patch_index,
             .reservation_end = *boundary});
      }
    }
    const auto find_reusable_access_relay =
        [&](const std::vector<size_t> &indices,
            const std::vector<uint64_t> &owners) -> std::optional<ReusableDenseAccessRelay> {
      // gfx1250 access relays already derive their dispatch key from the call
      // return PC. RDNA4 access relays use an explicit two-word key and must
      // retain their separate one-word barrier router.
      if (arch != ROCJITSU_CODE_ARCH_GFX1250)
        return std::nullopt;
      const auto found = reusable_access_relays.find(owners);
      if (found == reusable_access_relays.end())
        return std::nullopt;
      for (const ReusableDenseAccessRelay &relay : found->second) {
        if (!std::ranges::all_of(indices, [&](size_t index) {
              return compute_sopp_branch_simm16(
                         planned_candidates[index].candidate.site.text_offset, relay.island_offset)
                  .has_value();
            })) {
          continue;
        }
        return relay;
      }
      return std::nullopt;
    };

    enum class ReusedDenseRelayResult : uint8_t {
      Unavailable,
      Applied,
      Failed,
    };
    const auto apply_reused_access_relay =
        [&](const std::vector<size_t> &indices,
            const std::vector<uint64_t> &owners) -> ReusedDenseRelayResult {
      const auto relay = find_reusable_access_relay(indices, owners);
      if (!relay)
        return ReusedDenseRelayResult::Unavailable;

      ConSanOptions group_options = options;
      if (!apply_moi_transient_sgpr_assignment(group_options, owners)) {
        result.errors.emplace_back(
            "ConSan MOI reused barrier relay lost its transient scalar assignment");
        return ReusedDenseRelayResult::Failed;
      }
      const auto scalar_abi = moi_inline_dense_router_scalar_abi(group_options, arch);
      if (!scalar_abi || !scalar_abi->call_return_sgpr) {
        result.errors.emplace_back("ConSan MOI reused barrier relay has no compatible scalar ABI");
        return ReusedDenseRelayResult::Failed;
      }
      const MoiIndirectJumpSgprs indirect_state = scalar_abi->indirect_jump;
      const uint16_t key_sgpr = scalar_abi->dispatch_key_sgpr;
      const uint16_t call_return_sgpr = *scalar_abi->call_return_sgpr;
      const auto build_direct_call = [&](int16_t simm16) -> std::optional<uint32_t> {
        if (arch == ROCJITSU_CODE_ARCH_GFX1250)
          return instrumentation::build_s_call_i64(call_return_sgpr, simm16, arch);
        if (arch == ROCJITSU_CODE_ARCH_RDNA4)
          return build_s_call_b64(call_return_sgpr, simm16, arch);
        return std::nullopt;
      };

      ConSanPatchInfo &dispatcher = result.patches[relay->dispatcher_patch_index];
      if (dispatcher.trampoline_size < sizeof(uint32_t) ||
          dispatcher.trampoline_offset + dispatcher.trampoline_size > new_text.size()) {
        result.errors.emplace_back(
            "ConSan MOI reused barrier relay found a stale access dispatcher");
        return ReusedDenseRelayResult::Failed;
      }
      std::vector<uint32_t> dispatcher_words(dispatcher.trampoline_size / sizeof(uint32_t));
      std::memcpy(dispatcher_words.data(), new_text.data() + dispatcher.trampoline_offset,
                  dispatcher.trampoline_size);
      if (dispatcher_words.empty() || dispatcher_words.back() != build_s_endpgm(arch)) {
        result.errors.emplace_back(
            "ConSan MOI reused barrier relay found an unterminated access dispatcher");
        return ReusedDenseRelayResult::Failed;
      }
      dispatcher_words.pop_back();

      struct StagedReusedBarrier {
        uint64_t site_offset = 0;
        uint32_t call_word = 0;
        std::vector<uint32_t> body_words;
        ConSanPatchInfo patch;
      };
      std::vector<StagedReusedBarrier> staged_barriers;
      staged_barriers.reserve(indices.size());
      uint64_t staged_body_bytes = 0u;
      for (size_t index : indices) {
        const PlannedInlineEpochBarrier &planned = planned_candidates[index];
        const BarrierRecordCandidate &candidate = planned.candidate;
        const ConSanBarrierSite &site = candidate.site;
        uint32_t original_barrier_word = 0;
        std::memcpy(&original_barrier_word, old_text.data() + site.text_offset,
                    sizeof(original_barrier_word));
        const std::optional<uint16_t> gfx1250_vgpr_msb_mode =
            arch == ROCJITSU_CODE_ARCH_GFX1250
                ? gfx1250_vgpr_msb_mode_at(active_bytes, candidate.text_file_offset,
                                           candidate.container_entry_text_offset, site.file_offset)
                : std::nullopt;
        ConSanOptions candidate_options = options;
        if (!apply_moi_transient_sgpr_assignment(candidate_options,
                                                 planned.owner_descriptor_file_offsets)) {
          result.errors.emplace_back(
              "ConSan MOI reused barrier relay lost a site scalar assignment");
          return ReusedDenseRelayResult::Failed;
        }
        if (!planned.private_epoch_offset && !candidate_options.moi_persistent_epoch_sgpr &&
            !apply_moi_persistent_vgpr_assignment(candidate_options,
                                                  planned.owner_descriptor_file_offsets)) {
          result.errors.emplace_back(
              "ConSan MOI reused barrier relay lost a persistent VGPR assignment");
          return ReusedDenseRelayResult::Failed;
        }

        if (staged_body_bytes > std::numeric_limits<uint64_t>::max() - new_text.size()) {
          result.errors.emplace_back("ConSan MOI reused barrier relay body offsets overflowed");
          return ReusedDenseRelayResult::Failed;
        }
        const uint64_t body_offset = new_text.size() + staged_body_bytes;
        std::optional<std::vector<uint32_t>> body_words = build_engine_barrier_body(
            planned, candidate_options, original_barrier_word, body_offset,
            site.text_offset + site.size, call_return_sgpr, gfx1250_vgpr_msb_mode);
        if (!body_words)
          return ReusedDenseRelayResult::Failed;
        const uint64_t body_bytes = body_words->size() * sizeof(uint32_t);
        if (body_bytes > std::numeric_limits<uint64_t>::max() - staged_body_bytes) {
          result.errors.emplace_back("ConSan MOI reused barrier relay body size overflowed");
          return ReusedDenseRelayResult::Failed;
        }
        staged_body_bytes += body_bytes;

        const auto compare = instrumentation::build_s_cmp_eq_u32(key_sgpr, /*literal=*/255u, arch);
        if (!compare) {
          result.errors.emplace_back(
              "ConSan MOI reused barrier relay could not encode its dispatch comparison");
          return ReusedDenseRelayResult::Failed;
        }
        dispatcher_words.push_back(*compare);
        dispatcher_words.push_back(
            static_cast<uint32_t>(site.text_offset + sizeof(uint32_t) - relay->island_offset));
        std::vector<uint32_t> matched_words;
        const uint64_t matched_offset =
            dispatcher.trampoline_offset + (dispatcher_words.size() + 1u) * sizeof(uint32_t);
        if (!append_moi_scc_preserving_indirect_jump(
                matched_words, matched_offset, body_offset, indirect_state.pc_sgpr,
                indirect_state.scc_save_sgpr, /*capture_scc=*/false, arch)) {
          result.errors.emplace_back(
              "ConSan MOI reused barrier relay could not encode a dispatch target");
          return ReusedDenseRelayResult::Failed;
        }
        const auto skip =
            instrumentation::build_s_cbranch_scc0(static_cast<int16_t>(matched_words.size()), arch);
        if (!skip) {
          result.errors.emplace_back(
              "ConSan MOI reused barrier relay could not encode a dispatch branch");
          return ReusedDenseRelayResult::Failed;
        }
        dispatcher_words.push_back(*skip);
        dispatcher_words.insert(dispatcher_words.end(), matched_words.begin(), matched_words.end());

        const auto call_delta = compute_sopp_branch_simm16(site.text_offset, relay->island_offset);
        const auto call = call_delta ? build_direct_call(*call_delta) : std::nullopt;
        if (!call) {
          result.errors.emplace_back(
              "ConSan MOI reused barrier relay could not encode a barrier call");
          return ReusedDenseRelayResult::Failed;
        }

        ConSanPatchInfo info;
        info.kind = options.moi_engine == ConSanMoiEngine::RecordReplay
                        ? ConSanPatchKind::TrampolineMoiBarrierRecord
                        : ConSanPatchKind::TrampolineMoiInlineEpochBarrier;
        info.anchor_offset = site.text_offset;
        info.trampoline_offset = body_offset;
        info.original_size = site.size;
        info.trampoline_size = static_cast<uint32_t>(body_words->size() * sizeof(uint32_t));
        if (options.moi_engine == ConSanMoiEngine::RecordReplay) {
          const auto guest =
              std::ranges::find(body_words->rbegin(), body_words->rend(), original_barrier_word);
          if (guest != body_words->rend()) {
            info.relocated_guest_instruction_offset =
                body_offset + static_cast<uint64_t>(std::distance(guest, body_words->rend()) - 1u) *
                                  sizeof(uint32_t);
          }
        } else {
          info.relocated_guest_instruction_offset =
              body_offset + (candidate_options.moi_persistent_epoch_sgpr ? sizeof(uint32_t) : 0u);
        }
        info.scratch_vgpr = planned.scratch_vgpr;
        info.persistent_epoch_private_offset = planned.private_epoch_offset;
        info.persistent_owner_private_offset = planned.private_owner_offset;
        info.persistent_record_replay_workgroup_private_offsets =
            planned.private_record_replay_workgroup_offsets;
        if (planned.persistent_private_state_end != 0u)
          info.persistent_private_state_end = planned.persistent_private_state_end;
        info.required_private_segment_size = planned.required_private_bytes;
        info.owner_descriptor_file_offsets = planned.owner_descriptor_file_offsets;
        if (planned.spill) {
          info.spilled_vgpr_count = planned.spill->vgpr_count;
          note_dynamic_stack_private_requirement(info, &*planned.spill);
        }
        staged_barriers.push_back({.site_offset = site.text_offset,
                                   .call_word = *call,
                                   .body_words = std::move(*body_words),
                                   .patch = std::move(info)});
      }

      dispatcher_words.push_back(build_s_endpgm(arch));
      const uint64_t dispatcher_bytes = dispatcher_words.size() * sizeof(uint32_t);
      if (dispatcher.trampoline_offset + dispatcher_bytes > relay->reservation_end) {
        return ReusedDenseRelayResult::Unavailable;
      }

      // Everything above is side-effect-free with respect to the transformed
      // image. Commit only after the exact dispatcher encoding is known to fit
      // its access-relay reservation.
      new_text.reserve(new_text.size() + staged_body_bytes);
      for (StagedReusedBarrier &staged : staged_barriers) {
        append_words_bytes(new_text, staged.body_words);
        std::memcpy(new_text.data() + staged.site_offset, &staged.call_word, sizeof(uint32_t));
        patches.push_back(std::move(staged.patch));
      }
      std::memcpy(new_text.data() + dispatcher.trampoline_offset, dispatcher_words.data(),
                  dispatcher_bytes);
      dispatcher.trampoline_size = static_cast<uint32_t>(dispatcher_bytes);
      return ReusedDenseRelayResult::Applied;
    };

    for (const auto &group : groups) {
      const std::vector<size_t> &indices = group.second;
      if (indices.empty())
        continue;
      const BarrierRecordCandidate &first_candidate = planned_candidates[indices.front()].candidate;
      if (first_candidate.container_code_size == 0u)
        continue;
      const uint64_t owner_begin = first_candidate.container_entry_text_offset;
      const uint64_t owner_end = owner_begin + first_candidate.container_code_size;
      const std::string &owner_name = first_candidate.container_name;
      const std::vector<uint64_t> &group_owner =
          planned_candidates[indices.front()].owner_descriptor_file_offsets;
      if (group_owner.empty() || std::ranges::any_of(indices, [&](size_t index) {
            return planned_candidates[index].owner_descriptor_file_offsets != group_owner;
          })) {
        result.errors.emplace_back(
            "ConSan MOI dense barrier relay has inconsistent dispatch ownership");
        return;
      }
      switch (apply_reused_access_relay(indices, group_owner)) {
      case ReusedDenseRelayResult::Applied:
        continue;
      case ReusedDenseRelayResult::Failed:
        return;
      case ReusedDenseRelayResult::Unavailable:
        break;
      }
      ConSanOptions group_options = options;
      if (!apply_moi_transient_sgpr_assignment(group_options, group_owner)) {
        result.errors.emplace_back(
            "ConSan MOI dense barrier relay lost its transient scalar assignment");
        return;
      }
      const auto special_state = moi_special_state_sgprs(group_options);
      const auto indirect_state = moi_indirect_jump_sgprs(group_options);
      const bool spill_backed_inline_router = group_options.automatic_moi_inline_sgpr_spill;
      const bool spill_backed_record_replay_router =
          group_options.automatic_moi_record_replay_sgpr_spill;
      const bool spill_backed_router =
          spill_backed_inline_router || spill_backed_record_replay_router;
      if ((!spill_backed_router && (!group_options.moi_exec_save_sgpr || !special_state)) ||
          !indirect_state ||
          (spill_backed_inline_router && !group_options.moi_inline_visible_evidence_sgpr) ||
          (spill_backed_record_replay_router &&
           (!group_options.moi_record_replay_dispatch_key_sgpr ||
            !group_options.moi_record_replay_call_return_sgpr))) {
        result.errors.emplace_back(
            "ConSan MOI dense barrier relay requires scalar special-state storage");
        return;
      }
      const uint16_t jump_pc_sgpr =
          spill_backed_router ? indirect_state->pc_sgpr : *group_options.moi_exec_save_sgpr;
      const uint16_t saved_scc_sgpr =
          spill_backed_router ? indirect_state->scc_save_sgpr : special_state->scc_save_sgpr;
      uint16_t key_sgpr = static_cast<uint16_t>(*group_options.moi_exec_save_sgpr + 5u);
      uint16_t call_return_sgpr = static_cast<uint16_t>(*group_options.moi_exec_save_sgpr + 6u);
      if (spill_backed_inline_router) {
        key_sgpr = *group_options.moi_inline_visible_evidence_sgpr;
        call_return_sgpr = jump_pc_sgpr;
      } else if (spill_backed_record_replay_router) {
        key_sgpr = *group_options.moi_record_replay_dispatch_key_sgpr;
        call_return_sgpr = *group_options.moi_record_replay_call_return_sgpr;
      }
      const auto build_direct_call = [&](int16_t simm16) -> std::optional<uint32_t> {
        if (arch == ROCJITSU_CODE_ARCH_GFX1250)
          return instrumentation::build_s_call_i64(call_return_sgpr, simm16, arch);
        if (arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA4 ||
            arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4)
          return build_s_call_b64(call_return_sgpr, simm16, arch);
        return std::nullopt;
      };
      const uint64_t island_words_count = moi_dense_barrier_entry_island_words(spill_backed_router);
      const uint64_t host_words_count = island_words_count + 1u;
      const uint64_t host_bytes = host_words_count * sizeof(uint32_t);
      std::vector<uint64_t> anchors;
      anchors.reserve(indices.size());
      for (size_t index : indices)
        anchors.push_back(planned_candidates[index].candidate.site.text_offset);
      const ConSanCandidateResourcePlan *group_resource =
          resource_plan_for_site(result, ConSanResourceSiteKind::Barrier,
                                 planned_candidates[indices.front()].candidate.site.text_offset);
      if (group_resource == nullptr) {
        result.warnings.emplace_back(
            "ConSan MOI inline-shadow barrier epoch has no dense relay resource plan in " +
            owner_name);
        continue;
      }
      const std::vector<MoiSgprRange> host_bootstrap_ranges = moi_relocatable_host_scalar_ranges(
          group_options, MoiRelocatableHostScalarState::BarrierRouter, arch);
      MoiDenseOwnerLivenessProof host_owner_proof =
          resource_state.prepare_dense_liveness_for_owners(
              group_resource->owner_descriptor_file_offsets, host_bootstrap_ranges);

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
        result.warnings.emplace_back(
            "ConSan MOI inline-shadow barrier epoch found no relocatable dense relay host in " +
            owner_name);
        continue;
      }
      const uint64_t host_offset = dense_host->host_offset;
      std::vector<uint32_t> displaced_words = std::move(dense_host->displaced_words);
      claimed_hosts.emplace(host_offset, host_offset + host_bytes);

      const uint64_t dispatcher_offset = new_text.size();
      const uint64_t dispatcher_reservation =
          (indices.size() + 1u) * kDispatcherWordsPerSite * sizeof(uint32_t);
      while (new_text.size() < dispatcher_offset + dispatcher_reservation)
        append_word_bytes(new_text, build_s_nop(0, arch));

      struct DenseBarrierBody {
        size_t planned_index = 0;
        uint64_t offset = 0;
        uint64_t relocated_guest_instruction_offset = 0;
        std::vector<uint32_t> words;
      };
      std::vector<DenseBarrierBody> bodies;
      bodies.reserve(indices.size());
      for (size_t index : indices) {
        const PlannedInlineEpochBarrier &planned = planned_candidates[index];
        const ConSanBarrierSite &site = planned.candidate.site;
        uint32_t original_barrier_word = 0;
        std::memcpy(&original_barrier_word, old_text.data() + site.text_offset,
                    sizeof(original_barrier_word));
        const BarrierRecordCandidate &candidate = planned.candidate;
        const std::optional<uint16_t> gfx1250_vgpr_msb_mode =
            arch == ROCJITSU_CODE_ARCH_GFX1250
                ? gfx1250_vgpr_msb_mode_at(active_bytes, candidate.text_file_offset,
                                           candidate.container_entry_text_offset, site.file_offset)
                : std::nullopt;
        ConSanOptions candidate_options = options;
        if (!apply_moi_transient_sgpr_assignment(candidate_options,
                                                 planned.owner_descriptor_file_offsets)) {
          result.errors.emplace_back(
              "ConSan MOI dense barrier relay lost its transient scalar assignment");
          return;
        }
        if (!planned.private_epoch_offset && !candidate_options.moi_persistent_epoch_sgpr &&
            !apply_moi_persistent_vgpr_assignment(candidate_options,
                                                  planned.owner_descriptor_file_offsets)) {
          result.errors.emplace_back(
              "ConSan MOI dense barrier relay lost its persistent VGPR assignment");
          return;
        }
        const uint64_t body_offset = new_text.size();
        std::optional<std::vector<uint32_t>> body_words = build_engine_barrier_body(
            planned, candidate_options, original_barrier_word, body_offset,
            site.text_offset + site.size,
            spill_backed_router ? std::nullopt : std::optional<uint16_t>(call_return_sgpr),
            gfx1250_vgpr_msb_mode);
        if (!body_words)
          return;
        append_words_bytes(new_text, *body_words);
        bodies.push_back({.planned_index = index,
                          .offset = body_offset,
                          .relocated_guest_instruction_offset =
                              body_offset +
                              (candidate_options.moi_persistent_epoch_sgpr ? sizeof(uint32_t) : 0u),
                          .words = std::move(*body_words)});
      }

      const uint64_t host_body_offset = new_text.size();
      std::vector<uint32_t> host_words = displaced_words;
      if (!append_moi_scc_preserving_indirect_jump(host_words, host_body_offset,
                                                   host_offset + host_bytes, jump_pc_sgpr,
                                                   saved_scc_sgpr, /*capture_scc=*/true, arch)) {
        result.errors.emplace_back("ConSan MOI dense barrier relay could not return from its host");
        return;
      }
      append_words_bytes(new_text, host_words);

      const uint64_t island_offset = host_offset + sizeof(uint32_t);
      const auto island_words =
          build_moi_dense_barrier_entry_island(island_offset, dispatcher_offset,
                                               {.jump_pc_sgpr = jump_pc_sgpr,
                                                .saved_scc_sgpr = saved_scc_sgpr,
                                                .key_sgpr = key_sgpr,
                                                .call_return_sgpr = call_return_sgpr},
                                               spill_backed_router, arch);
      if (!island_words || island_words->size() != island_words_count) {
        result.errors.emplace_back("ConSan MOI dense barrier relay could not encode its island");
        return;
      }
      std::memcpy(new_text.data() + island_offset, island_words->data(),
                  island_words->size() * sizeof(uint32_t));
      const auto host_call = build_direct_call(/*next instruction=*/0);
      if (!host_call) {
        result.errors.emplace_back("ConSan MOI dense barrier relay could not encode its host call");
        return;
      }
      std::memcpy(new_text.data() + host_offset, &*host_call, sizeof(uint32_t));

      std::vector<uint32_t> dispatcher_words;
      if (!spill_backed_router) {
        const auto subtract_pc =
            instrumentation::build_s_sub_u32(key_sgpr, call_return_sgpr, jump_pc_sgpr, arch);
        if (!subtract_pc) {
          result.errors.emplace_back("ConSan MOI dense barrier relay could not encode its key");
          return;
        }
        dispatcher_words.push_back(*subtract_pc);
        dispatcher_words.push_back(build_s_add_u32(key_sgpr, key_sgpr, /*literal=*/255u, arch));
        dispatcher_words.push_back(static_cast<uint32_t>(dispatcher_offset - island_offset));
      }

      const auto append_dispatch_target = [&](uint32_t key, uint64_t target) -> bool {
        const auto compare = instrumentation::build_s_cmp_eq_u32(key_sgpr, /*literal=*/255u, arch);
        if (!compare)
          return false;
        dispatcher_words.push_back(*compare);
        dispatcher_words.push_back(key);
        std::vector<uint32_t> matched_words;
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
      if (!append_dispatch_target(/*entry host key=*/0u, host_body_offset)) {
        result.errors.emplace_back(
            "ConSan MOI dense barrier relay could not encode its host dispatch");
        return;
      }
      for (const DenseBarrierBody &body : bodies) {
        const ConSanBarrierSite &site = planned_candidates[body.planned_index].candidate.site;
        if (!append_dispatch_target(
                static_cast<uint32_t>(site.text_offset + sizeof(uint32_t) - island_offset),
                body.offset)) {
          result.errors.emplace_back(
              "ConSan MOI dense barrier relay could not encode a barrier dispatch");
          return;
        }
      }
      dispatcher_words.push_back(build_s_endpgm(arch));
      if (dispatcher_words.size() * sizeof(uint32_t) > dispatcher_reservation) {
        result.errors.emplace_back(
            "ConSan MOI dense barrier dispatcher exceeds its reserved relay space");
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
      host_info.owner_descriptor_file_offsets = group_owner;
      patches.push_back(std::move(host_info));
      ConSanPatchInfo dispatcher_info;
      dispatcher_info.kind = ConSanPatchKind::TrampolineMoiIndirectBranchIsland;
      dispatcher_info.anchor_offset =
          planned_candidates[indices.front()].candidate.site.text_offset;
      dispatcher_info.trampoline_offset = dispatcher_offset;
      dispatcher_info.trampoline_size =
          static_cast<uint32_t>(dispatcher_words.size() * sizeof(uint32_t));
      dispatcher_info.owner_descriptor_file_offsets = group_owner;
      patches.push_back(std::move(dispatcher_info));

      for (const DenseBarrierBody &body : bodies) {
        const PlannedInlineEpochBarrier &planned = planned_candidates[body.planned_index];
        const ConSanBarrierSite &site = planned.candidate.site;
        const auto call_delta = compute_sopp_branch_simm16(site.text_offset, island_offset);
        const auto call = call_delta ? build_direct_call(*call_delta) : std::nullopt;
        if (!call) {
          result.errors.emplace_back(
              "ConSan MOI dense barrier relay could not encode a barrier call");
          return;
        }
        std::memcpy(new_text.data() + site.text_offset, &*call, sizeof(uint32_t));
        ConSanPatchInfo info;
        info.kind = options.moi_engine == ConSanMoiEngine::RecordReplay
                        ? ConSanPatchKind::TrampolineMoiBarrierRecord
                        : ConSanPatchKind::TrampolineMoiInlineEpochBarrier;
        info.anchor_offset = site.text_offset;
        info.trampoline_offset = body.offset;
        info.original_size = site.size;
        info.trampoline_size = static_cast<uint32_t>(body.words.size() * sizeof(uint32_t));
        if (options.moi_engine == ConSanMoiEngine::RecordReplay) {
          const uint32_t original_barrier_word =
              planned_candidates[body.planned_index].candidate.site.text_offset <=
                      old_text.size() - sizeof(uint32_t)
                  ? [&] {
                      uint32_t word = 0u;
                      std::memcpy(
                          &word,
                          old_text.data() +
                              planned_candidates[body.planned_index].candidate.site.text_offset,
                          sizeof(word));
                      return word;
                    }()
                  : 0u;
          const auto guest =
              std::ranges::find(body.words.rbegin(), body.words.rend(), original_barrier_word);
          if (guest != body.words.rend()) {
            info.relocated_guest_instruction_offset =
                body.offset + static_cast<uint64_t>(std::distance(guest, body.words.rend()) - 1u) *
                                  sizeof(uint32_t);
          }
        } else {
          info.relocated_guest_instruction_offset = body.relocated_guest_instruction_offset;
        }
        info.scratch_vgpr = planned.scratch_vgpr;
        info.persistent_epoch_private_offset = planned.private_epoch_offset;
        info.persistent_owner_private_offset = planned.private_owner_offset;
        info.persistent_record_replay_workgroup_private_offsets =
            planned.private_record_replay_workgroup_offsets;
        if (planned.persistent_private_state_end != 0u)
          info.persistent_private_state_end = planned.persistent_private_state_end;
        info.required_private_segment_size = planned.required_private_bytes;
        info.owner_descriptor_file_offsets = planned.owner_descriptor_file_offsets;
        if (planned.spill) {
          info.spilled_vgpr_count = planned.spill->vgpr_count;
          note_dynamic_stack_private_requirement(info, &*planned.spill);
        }
        patches.push_back(std::move(info));
      }
    }
  }

  if (!replace_consan_text(patcher, new_text, options, "MOI inline-shadow barrier epoch patch",
                           result)) {
    return;
  }

  result.elf_bytes = std::move(patcher).emit();
  result.patches.insert(result.patches.end(), patches.begin(), patches.end());
  result.modified = true;
}

void try_apply_record_replay_barrier_record_patch(std::span<const uint8_t> bytes,
                                                  const ConSanOptions &options, rj_code_arch_t arch,
                                                  ConSanResult &result);

void try_apply_barrier_epoch_patch(std::span<const uint8_t> bytes, const ConSanOptions &options,
                                   rj_code_arch_t arch, MoiResourcePlanningState &resource_state,
                                   ConSanResult &result) {
  if (!options.moi_track_barriers || options.moi_engine == ConSanMoiEngine::Sampled)
    return;
  if (options.moi_engine == ConSanMoiEngine::InlineShadow) {
    ConSanOptions barrier_options = options;
    if (barrier_options.automatic_moi_inline_sgpr_spill)
      barrier_options.moi_exec_save_sgpr.reset();
    try_apply_inline_shadow_barrier_epoch_patch(bytes, barrier_options, arch, resource_state,
                                                result);
    return;
  }

  const size_t warning_count = result.warnings.size();
  try_apply_record_replay_barrier_record_patch(bytes, options, arch, result);
  if (!result.errors.empty())
    return;
  const bool needs_dense_fallback = std::ranges::any_of(
      std::span<const std::string>(result.warnings).subspan(warning_count),
      [](const std::string &warning) {
        return warning.find("no reachable indirect entry island") != std::string::npos;
      });
  if (needs_dense_fallback) {
    const size_t failed_entry_count = std::ranges::count_if(
        std::span<const std::string>(result.warnings).subspan(warning_count),
        [](const std::string &warning) {
          return warning.find("no reachable indirect entry island") != std::string::npos;
        });
    const size_t barrier_count_before = std::ranges::count(
        result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord, &ConSanPatchInfo::kind);
    try_apply_inline_shadow_barrier_epoch_patch(bytes, options, arch, resource_state, result);
    const size_t barrier_count_after = std::ranges::count(
        result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord, &ConSanPatchInfo::kind);
    if (result.errors.empty() && barrier_count_after - barrier_count_before >= failed_entry_count) {
      auto first_new_warning = result.warnings.begin() + static_cast<std::ptrdiff_t>(warning_count);
      result.warnings.erase(
          std::remove_if(first_new_warning, result.warnings.end(),
                         [](const std::string &warning) {
                           return warning.find("no reachable indirect entry island") !=
                                      std::string::npos ||
                                  warning ==
                                      "ConSan MOI barrier record patch found no patchable barriers";
                         }),
          result.warnings.end());
    }
  }
}

void try_apply_record_replay_barrier_record_patch(std::span<const uint8_t> bytes,
                                                  const ConSanOptions &options, rj_code_arch_t arch,
                                                  ConSanResult &result) {
  if (!options.moi_track_barriers)
    return;
  if (arch != ROCJITSU_CODE_ARCH_RDNA3 && arch != ROCJITSU_CODE_ARCH_RDNA4 &&
      arch != ROCJITSU_CODE_ARCH_GFX1250 && arch != ROCJITSU_CODE_ARCH_CDNA3 &&
      arch != ROCJITSU_CODE_ARCH_CDNA4) {
    result.warnings.emplace_back(
        "ConSan MOI barrier record patch does not support this architecture");
    return;
  }
  if (!options.moi_report_buffer_address) {
    result.warnings.emplace_back("ConSan MOI barrier record patch requires a MOI report buffer");
    return;
  }
  if (options.moi_report_buffer_size < consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 1)) {
    result.warnings.emplace_back(
        "ConSan MOI barrier record patch requires room for access and barrier records");
    return;
  }
  const ConSanMoiReportBufferLayout layout = resolve_record_replay_report_layout(
      options, /*include_barriers=*/true, options.moi_track_atomics, options.moi_track_atomics);
  const uint32_t max_candidates = std::min(options.max_patches, layout.barrier_record_capacity);
  if (max_candidates == 0) {
    result.warnings.emplace_back("ConSan MOI barrier record patch has no barrier record capacity");
    return;
  }
  if (!options.moi_exec_save_sgpr) {
    result.warnings.emplace_back(
        "ConSan MOI barrier record patch requires RJ_CONSAN_MOI_EXEC_SAVE_SGPR");
    return;
  }
  if (*options.moi_exec_save_sgpr > 100u || *options.moi_exec_save_sgpr % 2u != 0u) {
    result.warnings.emplace_back(
        "ConSan MOI barrier record patch requires an even RJ_CONSAN_MOI_EXEC_SAVE_SGPR in 0..100");
    return;
  }

  std::vector<BarrierRecordCandidate> candidates;
  for (const ConSanKernelInfo &kernel : result.kernels)
    append_barrier_epoch_candidates(kernel, candidates);
  for (const ConSanFunctionInfo &function : result.functions)
    append_barrier_epoch_candidates(function, candidates);
  apply_test_container_filter(candidates, options);
  if (!canonicalize_barrier_record_candidates_by_physical_site(candidates, result))
    return;
  std::sort(candidates.begin(), candidates.end(), [](const auto &lhs, const auto &rhs) {
    return lhs.site.text_offset < rhs.site.text_offset;
  });
  if (candidates.empty()) {
    result.warnings.emplace_back("ConSan MOI barrier record patch found no barriers");
    return;
  }

  std::span<const uint8_t> active_bytes = bytes;
  if (result.modified)
    active_bytes = std::span<const uint8_t>(result.elf_bytes.data(), result.elf_bytes.size());
  AmdGpuCodeObject code_object(active_bytes.data(), active_bytes.size());
  CodeObjectPatcher patcher(code_object);
  const std::span<const uint8_t> old_text = patcher.text_bytes();
  if (old_text.empty()) {
    result.errors.emplace_back("ConSan MOI barrier record patch found no .text section");
    return;
  }
  struct PlannedBarrierRecord {
    BarrierRecordCandidate candidate;
    ResolvedMoiScratchPlan resources;
    std::optional<VgprSpillSequence> spill;
    std::optional<InlineSgprSpillSequence> scalar_spill;
    std::optional<MoiPrivateRecordOwnerPlan> private_owner;
    std::optional<MoiPrivateEpochLayout> private_layout;
  };
  std::vector<PlannedBarrierRecord> selected_candidates;
  selected_candidates.reserve(max_candidates);
  std::unordered_set<uint64_t> selected_offsets;
  const MoiPatchAnchorRangeIndex occupied_anchors(result);
  MoiResourcePlanningState standalone_resource_state(bytes, arch, options, result);
  MoiSpillManagers spill_managers;
  MoiDescriptorVgprRequirements descriptor_requirements;
  MoiDescriptorSgprRequirements scalar_requirements;
  MoiDescriptorPrivateRequirements private_requirements;
  for (size_t candidate_index = 0; candidate_index < candidates.size(); ++candidate_index) {
    const BarrierRecordCandidate &candidate = candidates[candidate_index];
    if (selected_candidates.size() == max_candidates)
      break;
    const ConSanBarrierSite &site = candidate.site;
    if (site.size != sizeof(uint32_t)) {
      result.warnings.emplace_back("ConSan MOI barrier record patch skipped non-32-bit barrier " +
                                   site.mnemonic + " in " + candidate.container_name);
      continue;
    }
    if (site.text_offset > old_text.size() || site.size > old_text.size() - site.text_offset) {
      result.errors.emplace_back("ConSan MOI barrier record patch site is outside .text");
      return;
    }
    if (occupied_anchors.overlaps(site.text_offset, site.size) ||
        !selected_offsets.insert(site.text_offset).second) {
      result.warnings.emplace_back("ConSan MOI barrier record patch skipped an overlapping patch");
      continue;
    }
    const ConSanCandidateResourcePlan *existing_plan =
        resource_plan_for_site(result, ConSanResourceSiteKind::Barrier, site.text_offset);
    ConSanCandidateResourcePlan plan =
        existing_plan
            ? *existing_plan
            : plan_moi_resource_site(standalone_resource_state, options,
                                     ConSanResourceSiteKind::Barrier, candidate_index,
                                     site.text_offset, candidate.kernel_descriptor_file_offset,
                                     /*scratch_count=*/6u);
    if (!existing_plan)
      result.resource_plans.push_back(plan);
    const auto resources = resolve_moi_scratch_plan(plan, options, /*expected_count=*/6u);
    if (!resources) {
      result.warnings.emplace_back(
          "ConSan MOI barrier record patch has no usable resource plan in " +
          candidate.container_name);
      continue;
    }
    BarrierRecordCandidate planned_candidate = candidate;
    if (!options.moi_owner_vgpr && !planned_candidate.kernel_descriptor_file_offset) {
      planned_candidate.kernel_descriptor_file_offset =
          common_moi_record_owner_descriptor(bytes, *resources, arch, result.warnings);
      if (!planned_candidate.kernel_descriptor_file_offset)
        continue;
    }
    ConSanOptions candidate_options = options;
    candidate_options.scratch_vgpr = resources->base;
    std::optional<MoiPrivateEpochLayout> private_layout;
    if (options.automatic_moi_private_epoch) {
      private_layout = build_moi_private_epoch_layout(result, *resources, arch, result.warnings,
                                                      /*include_owner=*/options.moi_owner_source ==
                                                          ConSanMoiOwnerSource::WorkitemId,
                                                      /*include_workgroup_key=*/false,
                                                      /*include_record_replay_workgroup=*/true);
      if (!private_layout)
        continue;
      candidate_options.moi_record_replay_workgroup_private_offsets =
          private_layout->record_replay_workgroup_offsets;
    }
    if (!apply_moi_transient_sgpr_assignment(candidate_options,
                                             resources->owner_descriptor_file_offsets)) {
      result.warnings.emplace_back(
          "ConSan MOI barrier record patch found no common transient SGPR assignment");
      continue;
    }
    if (!apply_record_replay_entry_workgroup_assignment(candidate_options,
                                                        resources->owner_descriptor_file_offsets)) {
      result.warnings.emplace_back(
          "ConSan MOI barrier record patch found no common entry workgroup assignment");
      continue;
    }
    std::optional<MoiPrivateRecordOwnerPlan> private_owner;
    if (moi_record_uses_private_owner(candidate_options)) {
      private_owner =
          resolve_moi_private_record_owner(bytes, *resources, *private_layout, result.warnings);
      if (!private_owner)
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
      note_spill_descriptor_requirements(private_requirements, *resources, *spill);
    }
    std::optional<InlineSgprSpillSequence> scalar_spill;
    if (candidate_options.automatic_moi_record_replay_sgpr_spill) {
      scalar_spill = build_inline_sgpr_spill_sequence(
          result, *resources, candidate_options, spill_managers, arch, result.warnings,
          private_layout ? std::optional<uint32_t>(private_layout->ephemeral_base) : std::nullopt,
          spill ? &*spill : nullptr);
      if (!scalar_spill)
        continue;
      for (uint64_t descriptor_offset : resources->owner_descriptor_file_offsets) {
        auto [it, inserted] =
            private_requirements.emplace(descriptor_offset, scalar_spill->total_private_bytes);
        if (!inserted)
          it->second = std::max(it->second, scalar_spill->total_private_bytes);
      }
    }
    if (private_layout) {
      for (uint64_t descriptor_offset : resources->owner_descriptor_file_offsets) {
        auto [it, inserted] =
            private_requirements.emplace(descriptor_offset, private_layout->ephemeral_base);
        if (!inserted)
          it->second = std::max(it->second, private_layout->ephemeral_base);
      }
    }
    note_descriptor_requirements(descriptor_requirements, *resources);
    note_moi_sgpr_requirements(scalar_requirements, *resources, candidate_options, arch);
    selected_candidates.push_back({std::move(planned_candidate), *resources, std::move(spill),
                                   std::move(scalar_spill), private_owner, private_layout});
  }

  if (selected_candidates.empty()) {
    result.warnings.emplace_back("ConSan MOI barrier record patch found no patchable barriers");
    return;
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
  std::vector<ConSanPatchInfo> patches;
  for (const PlannedBarrierRecord &planned : selected_candidates) {
    const BarrierRecordCandidate &candidate = planned.candidate;
    const ConSanBarrierSite &site = candidate.site;
    uint32_t original_barrier_word = 0;
    std::memcpy(&original_barrier_word, new_text.data() + site.text_offset,
                sizeof(original_barrier_word));

    const uint64_t cave_text_offset = placement_planner.appended_end();
    const uint64_t return_text_offset = site.text_offset + site.size;
    ConSanOptions candidate_options = options;
    candidate_options.scratch_vgpr = planned.resources.base;
    if (planned.private_layout) {
      candidate_options.moi_record_replay_workgroup_private_offsets =
          planned.private_layout->record_replay_workgroup_offsets;
    }
    if (!apply_moi_transient_sgpr_assignment(candidate_options,
                                             planned.resources.owner_descriptor_file_offsets)) {
      result.errors.emplace_back(
          "ConSan MOI barrier record patch lost its transient SGPR assignment");
      return;
    }
    if (!apply_record_replay_entry_workgroup_assignment(
            candidate_options, planned.resources.owner_descriptor_file_offsets)) {
      result.errors.emplace_back(
          "ConSan MOI barrier record patch lost its entry workgroup assignment");
      return;
    }
    auto cave_words = build_barrier_record_cave_words(
        active_bytes, candidate, candidate_options, planned.spill ? &*planned.spill : nullptr,
        planned.scalar_spill ? &*planned.scalar_spill : nullptr, arch,
        layout.barrier_record_capacity, layout.barrier_records_offset, original_barrier_word,
        cave_text_offset, return_text_offset,
        planned.private_owner ? std::optional<uint32_t>(planned.private_owner->offset)
                              : std::nullopt,
        planned.private_owner ? std::optional<uint16_t>(planned.private_owner->shift)
                              : std::nullopt,
        result.errors);
    if (!cave_words)
      return;

    const bool direct_entry =
        compute_sopp_branch_simm16(site.text_offset, cave_text_offset).has_value();
    uint64_t indirect_entry_island = 0;
    bool has_indirect_entry_island = false;
    if (!direct_entry) {
      const auto claimed_island = indirect_allocator.claim_reachable(site.text_offset);
      if (!claimed_island) {
        result.warnings.emplace_back(
            "ConSan MOI barrier record patch found no reachable indirect entry island in " +
            candidate.container_name);
        continue;
      }
      indirect_entry_island = *claimed_island;
      has_indirect_entry_island = true;
    }
    std::string placement_error;
    const auto placement = placement_planner.plan_indirect_appended(
        site.text_offset, site.size, cave_words->size() * sizeof(uint32_t), &placement_error);
    if (!placement) {
      result.errors.emplace_back("ConSan MOI barrier record placement failed: " + placement_error);
      return;
    }
    if (placement->body_offset != cave_text_offset || new_text.size() != placement->body_offset) {
      result.errors.emplace_back("ConSan MOI barrier record patch has a stale cave mapping");
      return;
    }

    const uint64_t entry_target =
        has_indirect_entry_island ? indirect_entry_island : placement->body_offset;
    const auto fwd = compute_sopp_branch_simm16(placement->anchor_offset, entry_target);
    if (!fwd) {
      result.errors.emplace_back("ConSan MOI barrier record forward branch is out of range");
      return;
    }
    if (!write_word_bytes(new_text, site.text_offset, build_s_branch(*fwd, arch))) {
      result.errors.emplace_back("ConSan MOI barrier record patch could not rewrite barrier slot");
      return;
    }
    if (has_indirect_entry_island &&
        !emit_moi_local_indirect_entry_island(
            new_text, indirect_entry_island, placement->body_offset, site.text_offset, options,
            planned.resources.owner_descriptor_file_offsets, arch, patches, result.errors,
            "ConSan MOI barrier record patch")) {
      return;
    }
    append_words_bytes(new_text, *cave_words);

    ConSanPatchInfo info;
    info.kind = ConSanPatchKind::TrampolineMoiBarrierRecord;
    info.anchor_offset = site.text_offset;
    info.trampoline_offset = cave_text_offset;
    info.original_size = site.size;
    info.trampoline_size = static_cast<uint32_t>(cave_words->size() * sizeof(uint32_t));
    const auto guest =
        std::ranges::find(cave_words->rbegin(), cave_words->rend(), original_barrier_word);
    if (guest != cave_words->rend()) {
      info.relocated_guest_instruction_offset =
          cave_text_offset +
          static_cast<uint64_t>(std::distance(guest, cave_words->rend()) - 1u) * sizeof(uint32_t);
    }
    info.scratch_vgpr = planned.resources.base;
    info.owner_descriptor_file_offsets = planned.resources.owner_descriptor_file_offsets;
    if (planned.private_layout) {
      info.persistent_epoch_private_offset = planned.private_layout->epoch_offset;
      info.persistent_owner_private_offset = planned.private_layout->owner_offset;
      info.persistent_record_replay_workgroup_private_offsets =
          planned.private_layout->record_replay_workgroup_offsets;
      info.persistent_private_state_end = planned.private_layout->ephemeral_base;
      info.required_private_segment_size = planned.private_layout->ephemeral_base;
    }
    if (planned.spill) {
      info.spilled_vgpr_count = planned.spill->vgpr_count;
      info.required_private_segment_size =
          std::max(info.required_private_segment_size, planned.spill->total_private_bytes);
      note_dynamic_stack_private_requirement(info, &*planned.spill);
    }
    if (planned.scalar_spill) {
      info.required_private_segment_size =
          std::max(info.required_private_segment_size, planned.scalar_spill->total_private_bytes);
    }
    patches.push_back(info);
  }

  if (patches.empty()) {
    result.warnings.emplace_back("ConSan MOI barrier record patch found no patchable barriers");
    return;
  }
  if (!replace_consan_text(patcher, new_text, options, "MOI barrier record patch", result)) {
    return;
  }

  result.elf_bytes = std::move(patcher).emit();
  result.patches.insert(result.patches.end(), patches.begin(), patches.end());
  result.modified = true;
}
