[[nodiscard]] bool append_moi_runtime_workgroup_residue_compare(std::vector<uint32_t> &words,
                                                                uint16_t residue_sgpr,
                                                                uint16_t temporary_sgpr,
                                                                uint32_t selected_residue,
                                                                rj_code_arch_t arch) {
  std::optional<uint32_t> compare;
  if (selected_residue <= 64u) {
    compare = instrumentation::build_s_cmp_eq_u32(
        residue_sgpr, scalar_positive_inline_u32(selected_residue), arch);
  } else {
    constexpr uint16_t kScalarLiteralSource = 255u;
    words.push_back(build_s_mov_b32(temporary_sgpr, kScalarLiteralSource, arch));
    words.push_back(selected_residue);
    compare = instrumentation::build_s_cmp_eq_u32(residue_sgpr, temporary_sgpr, arch);
  }
  if (!compare)
    return false;
  words.push_back(*compare);
  return true;
}

[[nodiscard]] constexpr uint64_t
moi_runtime_workgroup_gate_reserved_words(uint32_t guest_byte_count,
                                          bool has_cluster_workgroup_id = false) {
  // A complete persistent vector tuple needs two more words per coordinate
  // than the scalar form: v_readfirstlane plus its VALU-to-SALU dependency
  // wait. Keep the envelope exact enough that large generated objects do not
  // push otherwise-reachable relays beyond SOPP branch range.
  // The full-tuple Record/Replay gate also restores the saved SCC on its
  // selected path. Account for that word in addition to the common XYZ gate
  // and indirect-return envelope. The previous 26-word bound happened to fit
  // scalar tuples, but rejected a valid vector XYZ+cluster tuple at emission.
  constexpr uint64_t kXyzGateWordsWithoutGuest = 27u;
  constexpr uint64_t kVectorClusterMixWords = 3u;
  return kXyzGateWordsWithoutGuest + (has_cluster_workgroup_id ? kVectorClusterMixWords : 0u) +
         (static_cast<uint64_t>(guest_byte_count) + sizeof(uint32_t) - 1u) / sizeof(uint32_t);
}

[[nodiscard]] bool append_moi_runtime_workgroup_mix(std::vector<uint32_t> &words,
                                                    const ConSanMoiWorkgroupSource &source,
                                                    uint16_t quotient, uint16_t residue,
                                                    rj_code_arch_t arch) {
  if (!source.is_well_formed())
    return false;
  if (!source.has_value())
    return true;
  if (source.private_offset) {
    // The gate runs before a spill-backed probe saves its guest-overlapping
    // scratch window. Loading private state here would require clobbering a
    // VGPR before that save, so callers must select a different lowering.
    return false;
  }

  uint16_t coordinate = 0;
  if (source.scalar_src) {
    coordinate = *source.scalar_src;
  } else if (source.vector_src) {
    const auto read =
        instrumentation::build_v_readfirstlane_b32(quotient, *source.vector_src, arch);
    const auto wait = instrumentation::build_valu_to_salu_dependency_wait(arch);
    if (!read || !wait)
      return false;
    words.push_back(*read);
    words.push_back(*wait);
    coordinate = quotient;
  } else {
    return false;
  }

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
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_moi_runtime_workgroup_gate_island_words(
    std::span<const uint8_t> bytes, const ConSanMoiCandidate &candidate, const MoiOptions &options,
    const ConSanMoiWorkgroupSources &workgroup_sources,
    [[maybe_unused]] uint32_t first_record_index, uint64_t island_text_offset,
    uint64_t cave_text_offset, uint64_t return_text_offset, uint64_t island_word_count,
    rj_code_arch_t arch) {
  const std::optional<ConSanMoiWorkgroupSource> cached_selection =
      moi_runtime_workgroup_selection_source();
  const bool record_replay = options.moi_engine == ConSanMoiEngine::RecordReplay;
  const ConSanTargetProfile *target = consan_target_profile(arch);
  if (!options.moi_exec_save_sgpr || options.moi_runtime_sample_stride <= 1u || target == nullptr ||
      (!record_replay && !cached_selection && !options.moi_dispatch_id_sgpr &&
       !moi_permits_literal_dispatch_identity(options, arch)))
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
  if (!save_scc_word)
    return std::nullopt;
  words.push_back(*save_scc_word);
  if (cached_selection) {
    uint16_t selected_source = 0u;
    if (cached_selection->scalar_src) {
      selected_source = *cached_selection->scalar_src;
    } else if (cached_selection->vector_src) {
      const auto read =
          instrumentation::build_v_readfirstlane_b32(quotient, *cached_selection->vector_src, arch);
      const auto wait = instrumentation::build_valu_to_salu_dependency_wait(arch);
      if (!read || !wait)
        return std::nullopt;
      words.push_back(*read);
      words.push_back(*wait);
      selected_source = quotient;
    } else {
      return std::nullopt;
    }
    const auto selected =
        instrumentation::build_s_cmp_lg_u32(selected_source, scalar_positive_inline_u32(0u), arch);
    if (!selected)
      return std::nullopt;
    words.push_back(*selected);
  } else {
    if (record_replay) {
      // Record/Replay samples stable workgroup coordinates. Dispatch identity
      // remains part of every record's attribution, but must not rotate the
      // selected workgroup or make offset zero omit workgroup zero.
      words.push_back(build_s_mov_b32(residue, scalar_positive_inline_u32(0u), arch));
    } else {
      const uint16_t dispatch_mix = options.moi_dispatch_id_sgpr.value_or(
          scalar_positive_inline_u32(static_cast<uint32_t>(options.moi_report_dispatch_id) & 63u));
      const auto mix_dispatch = instrumentation::build_s_sub_u32(
          residue, scalar_positive_inline_u32(0), dispatch_mix, arch);
      if (!mix_dispatch)
        return std::nullopt;
      words.push_back(*mix_dispatch);
    }

    // Sample one stable workgroup consistently at every site. Sampled also
    // mixes dispatch identity above; Record/Replay keeps dispatch identity in
    // its records instead of using it to rotate this workgroup predicate.
    if (!append_moi_runtime_workgroup_mix(words, workgroup_sources.x, quotient, residue, arch) ||
        !append_moi_runtime_workgroup_mix(words, workgroup_sources.y, quotient, residue, arch) ||
        !append_moi_runtime_workgroup_mix(words, workgroup_sources.z, quotient, residue, arch) ||
        !append_moi_runtime_workgroup_mix(words, workgroup_sources.cluster_workgroup_id, quotient,
                                          residue, arch))
      return std::nullopt;

    words.push_back(build_s_lshr_b32(quotient, residue, shift, arch));
    words.push_back(build_s_lshl_b32(quotient, quotient, shift, arch));
    const auto subtract = instrumentation::build_s_sub_u32(residue, residue, quotient, arch);
    if (!subtract)
      return std::nullopt;
    words.push_back(*subtract);
    if (!append_moi_runtime_workgroup_residue_compare(words, residue, quotient, selected_residue,
                                                      arch))
      return std::nullopt;
  }
  const auto restore_scc =
      instrumentation::build_s_cmp_lg_u32(saved_scc, scalar_positive_inline_u32(0), arch);
  if (!restore_scc)
    return std::nullopt;
  const size_t selected_branch_index = words.size();
  words.push_back(build_s_nop(0, arch));
  words.push_back(*restore_scc);
  for (uint64_t offset = 0; offset < candidate.size(); offset += sizeof(uint32_t)) {
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

[[nodiscard]] std::optional<std::vector<uint32_t>> build_moi_runtime_workgroup_gate_call_words(
    std::span<const uint8_t> guest_bytes, const MoiRuntimeWorkgroupGatePlan &plan,
    const ConSanMoiWorkgroupSources &workgroup_sources, uint64_t gate_text_offset,
    uint64_t return_text_offset, uint64_t reserved_word_count, rj_code_arch_t arch) {
  if (plan.sample_stride <= 1u)
    return std::nullopt;
  const uint16_t saved_scc = static_cast<uint16_t>(plan.exec_save_sgpr + 4u);
  const uint16_t quotient = static_cast<uint16_t>(plan.exec_save_sgpr + 5u);
  // Dense gfx1250 Record/Replay callers keep their s_call_i64 return PC in
  // base+6:base+7 until the body returns with s_setpc. The indirect-PC pair at
  // base+0 is dead after the dispatcher reaches this gate and is reinitialized
  // by the non-selected return jump or the selected body, so reuse its low
  // word for that engine without destroying the live call return. Sampled
  // returns with its ordinary indirect-PC pair and retains the established
  // residue slot.
  const uint16_t residue =
      plan.record_replay && plan.direct_call_form == ConSanDirectCallForm::SCallI64
          ? plan.exec_save_sgpr
          : static_cast<uint16_t>(plan.exec_save_sgpr + 6u);
  const uint16_t shift = scalar_positive_inline_u32(std::countr_zero(plan.sample_stride));
  const uint32_t mask = plan.sample_stride - 1u;
  const uint32_t selected_residue = plan.sample_offset & mask;

  std::vector<uint32_t> words;
  words.reserve(reserved_word_count);
  const auto save_scc_word = instrumentation::build_s_cselect_b32(
      saved_scc, scalar_positive_inline_u32(1), scalar_positive_inline_u32(0), arch);
  if (!save_scc_word)
    return std::nullopt;
  words.push_back(*save_scc_word);
  if (plan.cached_selection) {
    uint16_t selected_source = 0u;
    if (plan.cached_selection->scalar_src) {
      selected_source = *plan.cached_selection->scalar_src;
    } else if (plan.cached_selection->vector_src) {
      const auto read = instrumentation::build_v_readfirstlane_b32(
          quotient, *plan.cached_selection->vector_src, arch);
      const auto wait = instrumentation::build_valu_to_salu_dependency_wait(arch);
      if (!read || !wait)
        return std::nullopt;
      words.push_back(*read);
      words.push_back(*wait);
      selected_source = quotient;
    } else {
      return std::nullopt;
    }
    const auto selected =
        instrumentation::build_s_cmp_lg_u32(selected_source, scalar_positive_inline_u32(0u), arch);
    if (!selected)
      return std::nullopt;
    words.push_back(*selected);
  } else {
    if (plan.record_replay) {
      words.push_back(build_s_mov_b32(residue, scalar_positive_inline_u32(0u), arch));
    } else {
      const uint16_t dispatch_mix = plan.dispatch_id_sgpr.value_or(
          scalar_positive_inline_u32(static_cast<uint32_t>(plan.literal_dispatch_id) & 63u));
      const auto mix_dispatch = instrumentation::build_s_sub_u32(
          residue, scalar_positive_inline_u32(0), dispatch_mix, arch);
      if (!mix_dispatch)
        return std::nullopt;
      words.push_back(*mix_dispatch);
    }
    if (!append_moi_runtime_workgroup_mix(words, workgroup_sources.x, quotient, residue, arch) ||
        !append_moi_runtime_workgroup_mix(words, workgroup_sources.y, quotient, residue, arch) ||
        !append_moi_runtime_workgroup_mix(words, workgroup_sources.z, quotient, residue, arch) ||
        !append_moi_runtime_workgroup_mix(words, workgroup_sources.cluster_workgroup_id, quotient,
                                          residue, arch))
      return std::nullopt;
    words.push_back(build_s_lshr_b32(quotient, residue, shift, arch));
    words.push_back(build_s_lshl_b32(quotient, quotient, shift, arch));
    const auto subtract = instrumentation::build_s_sub_u32(residue, residue, quotient, arch);
    if (!subtract)
      return std::nullopt;
    words.push_back(*subtract);
    if (!append_moi_runtime_workgroup_residue_compare(words, residue, quotient, selected_residue,
                                                      arch))
      return std::nullopt;
  }
  const auto restore_scc =
      instrumentation::build_s_cmp_lg_u32(saved_scc, scalar_positive_inline_u32(0), arch);
  if (!restore_scc)
    return std::nullopt;
  const size_t selected_branch_index = words.size();
  words.push_back(build_s_nop(0, arch));
  words.push_back(*restore_scc);
  for (uint64_t offset = 0; offset < guest_bytes.size(); offset += sizeof(uint32_t)) {
    uint32_t word = 0;
    std::memcpy(&word, guest_bytes.data() + offset, sizeof(word));
    words.push_back(word);
  }
  if (!append_moi_scc_preserving_indirect_jump(words, gate_text_offset, return_text_offset,
                                               plan.exec_save_sgpr, saved_scc,
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
