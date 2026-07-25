struct ByteRange {
  uint64_t begin = 0;
  uint64_t end = 0;
};

[[nodiscard]] bool sc_sgpr_ranges_overlap(uint16_t lhs_base, uint16_t lhs_width, uint16_t rhs_base,
                                          uint16_t rhs_width) {
  return static_cast<uint32_t>(lhs_base) < static_cast<uint32_t>(rhs_base) + rhs_width &&
         static_cast<uint32_t>(rhs_base) < static_cast<uint32_t>(lhs_base) + lhs_width;
}

[[nodiscard]] std::optional<uint16_t> sc_delay_source_sgpr(const ConSanOptions &options) {
  if (options.delay_nops == 0 || options.delay_mode != ConSanDelayMode::SleepVar ||
      options.delay_var_ssrc >= REGISTER_SET_ALLOCATABLE_SGPRS)
    return std::nullopt;
  return options.delay_var_ssrc;
}

[[nodiscard]] bool sc_sgpr_range_overlaps_delay_source(uint16_t base, uint16_t width,
                                                       const ConSanOptions &options) {
  const auto delay_source = sc_delay_source_sgpr(options);
  return delay_source && sc_sgpr_ranges_overlap(base, width, *delay_source, 1u);
}

[[nodiscard]] uint16_t sc_vcc_save_sgpr_width(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4 ? 2u : 1u;
}

[[nodiscard]] std::optional<uint16_t> choose_sc_borrowed_scalar_sgpr(uint16_t allocated_sgpr_count,
                                                                     const ConSanOptions &options) {
  // This scalar is live and explicitly preserved, so the only constraints are
  // that every owner already allocates it and that it is not the active
  // runtime delay source.
  const uint16_t limit = std::min<uint16_t>(allocated_sgpr_count, REGISTER_SET_ALLOCATABLE_SGPRS);
  for (uint16_t sgpr = 0; sgpr < limit; ++sgpr) {
    if (!sc_sgpr_range_overlaps_delay_source(sgpr, 1u, options))
      return sgpr;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<uint16_t> choose_sc_dead_vcc_save_sgpr(
    const Instruction *inst, std::span<const LivenessAnalysis *const> owner_liveness,
    std::optional<uint16_t> min_preferred_sgpr, uint16_t selection_limit, rj_code_arch_t arch,
    const ConSanOptions &options) {
  if (inst == nullptr || owner_liveness.empty() ||
      std::ranges::any_of(owner_liveness, [](const auto *liveness) { return liveness == nullptr; }))
    return std::nullopt;
  const uint16_t width = sc_vcc_save_sgpr_width(arch);
  const uint16_t limit =
      std::min<uint16_t>(selection_limit, REGISTER_SET_ALLOCATABLE_SGPRS);
  const auto find_free = [&](uint16_t start) -> std::optional<uint16_t> {
    if (width == 2u && (start & 1u) != 0u)
      ++start;
    for (uint16_t base = start; static_cast<uint32_t>(base) + width <= limit;
         base = static_cast<uint16_t>(base + width)) {
      if (sc_sgpr_range_overlaps_delay_source(base, width, options))
        continue;
      const RegisterRef ref{RegClass::SGPR, base, static_cast<uint8_t>(width)};
      if (std::ranges::all_of(owner_liveness, [&](const LivenessAnalysis *liveness) {
            return !liveness->is_live_before(*inst, ref);
          }))
        return base;
    }
    return std::nullopt;
  };
  if (min_preferred_sgpr) {
    if (auto high_sgpr = find_free(*min_preferred_sgpr))
      return high_sgpr;
  }
  return find_free(0);
}

[[nodiscard]] std::optional<uint64_t>
text_offset_to_file_offset(const AmdGpuCodeObject &code_object, uint64_t text_offset,
                           uint64_t byte_count) {
  for (const Section *section : code_object.text_sections()) {
    if (text_offset > section->size())
      continue;
    if (byte_count > section->size() - text_offset)
      continue;
    return section->sectionOffset() + text_offset;
  }
  return std::nullopt;
}

[[nodiscard]] bool ranges_overlap(ByteRange lhs, ByteRange rhs) {
  return lhs.begin < rhs.end && rhs.begin < lhs.end;
}

[[nodiscard]] bool overlaps_preapplied_reserved_range(const ConSanOptions &options,
                                                      ByteRange range) {
  return std::ranges::any_of(
      options.preapplied_reserved_ranges, [&](const ConSanPreappliedReservedRange &reserved) {
        if (reserved.size == 0 ||
            reserved.text_offset > std::numeric_limits<uint64_t>::max() - reserved.size)
          return false;
        return ranges_overlap(range, {reserved.text_offset, reserved.text_offset + reserved.size});
      });
}

[[nodiscard]] bool overlaps_reserved_range(std::span<const ByteRange> ranges, ByteRange range) {
  for (ByteRange reserved : ranges) {
    if (ranges_overlap(reserved, range))
      return true;
  }
  return false;
}

[[nodiscard]] bool reserve_byte_range(std::vector<ByteRange> &ranges, ByteRange range) {
  if (range.begin >= range.end || overlaps_reserved_range(ranges, range))
    return false;
  ranges.push_back(range);
  return true;
}

[[nodiscard]] std::optional<std::vector<ByteRange>>
reserved_ranges_for_existing_patches(const AmdGpuCodeObject &code_object,
                                     const ConSanResult &result) {
  std::vector<ByteRange> ranges;
  for (const ConSanPatchInfo &patch : result.patches) {
    // Some structural records describe appended dispatch machinery without
    // replacing pristine text.  Their zero-byte anchors are identities, not
    // reservations, and must not make a later composed patch fail closed.
    if (patch.original_size != 0) {
      const auto anchor_file_offset =
          text_offset_to_file_offset(code_object, patch.anchor_offset, patch.original_size);
      if (!anchor_file_offset)
        return std::nullopt;
      if (!reserve_byte_range(ranges,
                              {*anchor_file_offset, *anchor_file_offset + patch.original_size}))
        return std::nullopt;
    }

    if (patch.trampoline_size == 0)
      continue;
    const auto trampoline_file_offset =
        text_offset_to_file_offset(code_object, patch.trampoline_offset, patch.trampoline_size);
    // Appended bodies are outside the pristine image and cannot overlap the
    // inline/local-only flat selector. They are reserved later against the
    // current composed text image.
    if (!trampoline_file_offset)
      continue;
    if (!reserve_byte_range(
            ranges, {*trampoline_file_offset, *trampoline_file_offset + patch.trampoline_size}))
      return std::nullopt;
  }
  return ranges;
}

[[nodiscard]] ConSanPatchKind lds_check_trap_patch_kind(const ConSanLdsSite &site,
                                                        bool use_local_cave) {
  if (site.kind == ConSanLdsAccessKind::Write)
    return use_local_cave ? ConSanPatchKind::LocalCaveLdsStoreCheckTrap
                          : ConSanPatchKind::InlineLdsStoreCheckTrap;
  return use_local_cave ? ConSanPatchKind::LocalCaveLdsLoadCheckTrap
                        : ConSanPatchKind::InlineLdsLoadCheckTrap;
}

[[nodiscard]] bool is_d16_lds_load(const ConSanLdsSite &site) {
  return site.mnemonic == "ds_load_u16_d16" || site.mnemonic == "ds_load_u16_d16_hi";
}

[[nodiscard]] uint32_t lds_check_trap_predelay_setup_words(const ConSanLdsSite &site) {
  return is_d16_lds_load(site) ? 2u : 0u;
}

[[nodiscard]] bool is_subdword_lds_store(const ConSanLdsSite &site) {
  return site.kind == ConSanLdsAccessKind::Write && site.width_bits < 32u;
}

[[nodiscard]] bool is_high_d16_lds_store(const ConSanLdsSite &site) {
  return site.mnemonic == "ds_store_b8_d16_hi" || site.mnemonic == "ds_store_b16_d16_hi";
}

[[nodiscard]] uint16_t lds_check_trap_extra_scratch_vgprs(const ConSanLdsSite &site) {
  return is_subdword_lds_store(site) || site.dst_accvgpr ? 1u : 0u;
}

[[nodiscard]] uint32_t lds_check_trap_precompare_words(const ConSanLdsSite &site) {
  if (!is_subdword_lds_store(site))
    return 0u;
  return is_high_d16_lds_store(site) ? 3u : 2u;
}

[[nodiscard]] std::optional<std::array<uint32_t, 2>>
build_cdna_accvgpr_read_b32(uint16_t dst_vgpr, uint16_t src_accvgpr, rj_code_arch_t arch) {
  if ((arch != ROCJITSU_CODE_ARCH_CDNA3 && arch != ROCJITSU_CODE_ARCH_CDNA4) || dst_vgpr > 255u ||
      src_accvgpr > 255u)
    return std::nullopt;
  if (arch == ROCJITSU_CODE_ARCH_CDNA3) {
    return cdna3::build_vop3p(cdna3::kVAccvgprReadVop3p,
                              {.vdst = static_cast<uint8_t>(dst_vgpr),
                               .op_sel_hi_2 = 1u,
                               .src0 = static_cast<uint16_t>(256u + src_accvgpr),
                               .op_sel_hi = 3u});
  }
  return cdna4::build_vop3p(cdna4::kVAccvgprReadVop3p,
                            {.vdst = static_cast<uint8_t>(dst_vgpr),
                             .op_sel_hi_2 = 1u,
                             .src0 = static_cast<uint16_t>(256u + src_accvgpr),
                             .op_sel_hi = 3u});
}

constexpr uint64_t kLdsCheckTrapIndirectJumpWords = 7u;
constexpr uint32_t kScRelayEntryWords = 8u;
constexpr uint32_t kScRelayTailRestoreWords = 1u;
constexpr uint32_t kScRelayAppendedOverheadWords = 9u;
constexpr uint32_t kScRelayReservoirSlotStrideWords = 16u;
constexpr uint32_t kScRelayReservoirDenseEdgeWords = 16u;
constexpr uint16_t kRdna4VccLo = 106u;
constexpr uint16_t kRdna4ExecHi = 127u;

[[nodiscard]] std::vector<uint64_t> sc_relay_reservoir_slot_offsets(uint64_t anchor_offset,
                                                                    uint32_t original_size) {
  const uint64_t payload_begin = anchor_offset + kScRelayEntryWords * sizeof(uint32_t);
  const uint64_t payload_end =
      anchor_offset + original_size - kScRelayTailRestoreWords * sizeof(uint32_t);
  std::vector<uint64_t> slots;
  if (payload_end <= payload_begin)
    return slots;
  for (uint64_t offset = payload_begin; offset < payload_end;
       offset += kScRelayReservoirSlotStrideWords * sizeof(uint32_t))
    slots.push_back(offset);
  const uint64_t dense_edge_bytes = kScRelayReservoirDenseEdgeWords * sizeof(uint32_t);
  for (uint64_t offset = payload_begin;
       offset < std::min(payload_end, payload_begin + dense_edge_bytes); offset += sizeof(uint32_t))
    slots.push_back(offset);
  const uint64_t dense_tail_begin = payload_end > dense_edge_bytes
                                        ? std::max(payload_begin, payload_end - dense_edge_bytes)
                                        : payload_begin;
  for (uint64_t offset = dense_tail_begin; offset < payload_end; offset += sizeof(uint32_t))
    slots.push_back(offset);
  std::ranges::sort(slots);
  slots.erase(std::ranges::unique(slots).begin(), slots.end());
  return slots;
}

[[nodiscard]] std::vector<uint64_t> sc_relay_reservoir_all_slot_offsets(uint64_t anchor_offset,
                                                                        uint32_t original_size) {
  const uint64_t payload_begin = anchor_offset + kScRelayEntryWords * sizeof(uint32_t);
  const uint64_t payload_end =
      anchor_offset + original_size - kScRelayTailRestoreWords * sizeof(uint32_t);
  std::vector<uint64_t> slots;
  if (payload_end <= payload_begin)
    return slots;
  slots.reserve(static_cast<size_t>((payload_end - payload_begin) / sizeof(uint32_t)));
  for (uint64_t offset = payload_begin; offset < payload_end; offset += sizeof(uint32_t))
    slots.push_back(offset);
  return slots;
}

[[nodiscard]] bool append_lds_check_trap_indirect_jump(std::vector<uint32_t> &words,
                                                       uint64_t words_text_offset,
                                                       uint64_t target_text_offset,
                                                       uint16_t pc_sgpr, uint16_t saved_scc_sgpr,
                                                       rj_code_arch_t arch) {
  const auto save_scc = instrumentation::build_s_cselect_b32(
      saved_scc_sgpr, scalar_positive_inline_u32(1), scalar_positive_inline_u32(0), arch);
  if (!save_scc)
    return false;
  words.push_back(*save_scc);
  const uint64_t getpc_text_offset =
      words_text_offset + static_cast<uint64_t>(words.size()) * sizeof(uint32_t);
  words.push_back(build_s_getpc_b64(pc_sgpr, arch));
  const uint64_t pc_after_getpc = getpc_text_offset + sizeof(uint32_t);
  if (target_text_offset > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
      pc_after_getpc > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    return false;
  }
  const int64_t delta =
      static_cast<int64_t>(target_text_offset) - static_cast<int64_t>(pc_after_getpc);
  if (!append_pc_delta_builder(words, arch, pc_sgpr, delta))
    return false;
  const auto restore_scc =
      instrumentation::build_s_cmp_lg_u32(saved_scc_sgpr, scalar_positive_inline_u32(0), arch);
  if (!restore_scc)
    return false;
  words.push_back(*restore_scc);
  words.push_back(build_s_setpc_b64(pc_sgpr, arch));
  return true;
}

[[nodiscard]] bool
append_lds_check_trap_indirect_return(std::vector<uint32_t> &words, uint64_t words_text_offset,
                                      uint64_t target_text_offset, uint16_t pc_sgpr,
                                      uint16_t captured_scc_sgpr, rj_code_arch_t arch) {
  // Keep the same fixed geometry as an entry jump, but do not recapture SCC:
  // the entry island's value must survive the complete instrumentation body.
  words.push_back(build_s_nop(0, arch));
  const uint64_t getpc_text_offset =
      words_text_offset + static_cast<uint64_t>(words.size()) * sizeof(uint32_t);
  words.push_back(build_s_getpc_b64(pc_sgpr, arch));
  const uint64_t pc_after_getpc = getpc_text_offset + sizeof(uint32_t);
  if (target_text_offset > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
      pc_after_getpc > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    return false;
  }
  const int64_t delta =
      static_cast<int64_t>(target_text_offset) - static_cast<int64_t>(pc_after_getpc);
  if (!append_pc_delta_builder(words, arch, pc_sgpr, delta))
    return false;
  const auto restore_scc =
      instrumentation::build_s_cmp_lg_u32(captured_scc_sgpr, scalar_positive_inline_u32(0), arch);
  if (!restore_scc)
    return false;
  words.push_back(*restore_scc);
  words.push_back(build_s_setpc_b64(pc_sgpr, arch));
  return true;
}

[[nodiscard]] std::optional<uint16_t>
choose_lds_check_trap_indirect_pc_sgpr(
    const Instruction *inst, std::span<const LivenessAnalysis *const> owner_liveness,
    uint16_t excluded_vcc_sgpr, uint16_t excluded_vcc_width, uint16_t search_start = 0) {
  if (inst == nullptr || owner_liveness.empty() ||
      std::ranges::any_of(owner_liveness, [](const auto *liveness) { return liveness == nullptr; }))
    return std::nullopt;
  // find_free_sgpr_pair scans the full ordinary SGPR file and depends only on
  // liveness, not descriptor allocation. The first owner is therefore a
  // complete candidate enumerator; the all-owner predicate below computes the
  // actual intersection without making the result owner-order-dependent.
  while (auto pair = owner_liveness.front()->find_free_sgpr_pair(inst, search_start)) {
    const RegisterRef ref{RegClass::SGPR, *pair, 2u};
    if (!sc_sgpr_ranges_overlap(*pair, 2u, excluded_vcc_sgpr, excluded_vcc_width) &&
        std::ranges::all_of(owner_liveness, [&](const LivenessAnalysis *liveness) {
          return !liveness->is_live_before(*inst, ref);
        }))
      return pair;
    if (*pair + 2u >= REGISTER_SET_ALLOCATABLE_SGPRS)
      return std::nullopt;
    search_start = static_cast<uint16_t>(*pair + 2u);
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<uint16_t> choose_lds_check_trap_indirect_scc_save_sgpr(
    const Instruction *inst, std::span<const LivenessAnalysis *const> owner_liveness,
    uint16_t excluded_vcc_sgpr, uint16_t excluded_vcc_width, uint16_t excluded_pair,
    uint16_t search_start = 0) {
  if (inst == nullptr || owner_liveness.empty() ||
      std::ranges::any_of(owner_liveness, [](const auto *liveness) { return liveness == nullptr; }))
    return std::nullopt;
  // As above, the front owner enumerates every ordinary SGPR that can be dead
  // in the intersection; it does not impose that owner's allocation range.
  while (auto single = owner_liveness.front()->find_free_sgpr(inst, search_start)) {
    const RegisterRef ref{RegClass::SGPR, *single, 1u};
    if (!sc_sgpr_ranges_overlap(*single, 1u, excluded_vcc_sgpr, excluded_vcc_width) &&
        !sc_sgpr_ranges_overlap(*single, 1u, excluded_pair, 2u) &&
        std::ranges::all_of(owner_liveness, [&](const LivenessAnalysis *liveness) {
          return !liveness->is_live_before(*inst, ref);
        })) {
      return single;
    }
    if (*single + 1u >= REGISTER_SET_ALLOCATABLE_SGPRS)
      return std::nullopt;
    search_start = static_cast<uint16_t>(*single + 1u);
  }
  return std::nullopt;
}

struct LdsCheckTrapIndirectScalarScratch {
  uint16_t vcc_save_sgpr = 0;
  uint16_t pc_sgpr_pair = 0;
  uint16_t scc_save_sgpr = 0;
};

[[nodiscard]] std::optional<LdsCheckTrapIndirectScalarScratch>
choose_lds_check_trap_indirect_scalar_scratch(const Instruction *inst,
                                              std::span<const LivenessAnalysis *const> owner_liveness,
                                              std::optional<uint16_t> min_preferred_vcc_save,
                                              rj_code_arch_t arch) {
  if (inst == nullptr || owner_liveness.empty() ||
      std::ranges::any_of(owner_liveness, [](const auto *liveness) { return liveness == nullptr; }))
    return std::nullopt;

  uint16_t pair_search_start = min_preferred_vcc_save.value_or(0);
  while (auto pair = owner_liveness.front()->find_free_sgpr_pair(inst, pair_search_start)) {
    const RegisterRef pair_ref{RegClass::SGPR, *pair, 2u};
    if (!std::ranges::all_of(owner_liveness, [&](const LivenessAnalysis *liveness) {
          return !liveness->is_live_before(*inst, pair_ref);
        })) {
      if (*pair + 2u >= REGISTER_SET_ALLOCATABLE_SGPRS)
        return std::nullopt;
      pair_search_start = static_cast<uint16_t>(*pair + 2u);
      continue;
    }
    const uint16_t vcc_width = sc_vcc_save_sgpr_width(arch);
    const auto find_disjoint_vcc = [&](uint16_t start) -> std::optional<uint16_t> {
      if (vcc_width == 2u) {
        while (auto vcc = owner_liveness.front()->find_free_sgpr_pair(inst, start)) {
          const RegisterRef vcc_ref{RegClass::SGPR, *vcc, 2u};
          if (!sc_sgpr_ranges_overlap(*vcc, 2u, *pair, 2u) &&
              std::ranges::all_of(owner_liveness, [&](const LivenessAnalysis *liveness) {
                return !liveness->is_live_before(*inst, vcc_ref);
              }))
            return vcc;
          if (*vcc + 2u >= REGISTER_SET_ALLOCATABLE_SGPRS)
            return std::nullopt;
          start = static_cast<uint16_t>(*vcc + 2u);
        }
        return std::nullopt;
      }
      while (auto vcc = owner_liveness.front()->find_free_sgpr(inst, start)) {
        const RegisterRef vcc_ref{RegClass::SGPR, *vcc, 1u};
        if (!sc_sgpr_ranges_overlap(*vcc, 1u, *pair, 2u) &&
            std::ranges::all_of(owner_liveness, [&](const LivenessAnalysis *liveness) {
              return !liveness->is_live_before(*inst, vcc_ref);
            }))
          return vcc;
        if (*vcc + 1u >= REGISTER_SET_ALLOCATABLE_SGPRS)
          return std::nullopt;
        start = static_cast<uint16_t>(*vcc + 1u);
      }
      return std::nullopt;
    };

    std::optional<uint16_t> vcc_save;
    if (min_preferred_vcc_save)
      vcc_save = find_disjoint_vcc(*min_preferred_vcc_save);
    if (!vcc_save)
      vcc_save = find_disjoint_vcc(0);
    if (vcc_save) {
      const auto scc_save = choose_lds_check_trap_indirect_scc_save_sgpr(
          inst, owner_liveness, *vcc_save, vcc_width, *pair,
          min_preferred_vcc_save.value_or(0));
      if (scc_save)
        return LdsCheckTrapIndirectScalarScratch{*vcc_save, *pair, *scc_save};
    }

    if (*pair + 2u >= REGISTER_SET_ALLOCATABLE_SGPRS)
      return std::nullopt;
    pair_search_start = static_cast<uint16_t>(*pair + 2u);
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<uint32_t>
build_v_mov_b32_e32_vgpr_word(uint16_t vdst, uint16_t src_vgpr, rj_code_arch_t arch) {
  if (vdst > 255 || src_vgpr > 255)
    return std::nullopt;
  return build_v_mov_b32_e32(vdst, vector_source_vgpr(src_vgpr), arch);
}

[[nodiscard]] uint32_t report_action_scratch_vgprs(const ConSanOptions &options) {
  return options.report_buffer_address ? 3u : 0u;
}

[[nodiscard]] std::optional<uint32_t> mismatch_action_word_count(const ConSanOptions &options,
                                                                 rj_code_arch_t arch,
                                                                 std::vector<std::string> &errors,
                                                                 std::string_view context) {
  if (!options.report_buffer_address)
    return 1u;
  if (!instrumentation::is_admitted_arch(arch)) {
    errors.emplace_back(std::string(context) + " report-buffer action has unsupported target");
    return std::nullopt;
  }
  return arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4 ? 8u : 12u;
}

[[nodiscard]] std::optional<std::vector<uint32_t>>
build_mismatch_action_words(const ConSanOptions &options, rj_code_arch_t arch,
                            uint16_t report_scratch_vgpr, std::vector<std::string> &errors,
                            std::string_view context) {
  if (!options.report_buffer_address) {
    auto trap = build_s_trap_word(0, arch);
    if (!trap) {
      errors.emplace_back(std::string(context) + " could not encode s_trap");
      return std::nullopt;
    }
    return std::vector<uint32_t>{*trap};
  }

  if (static_cast<uint32_t>(report_scratch_vgpr) + report_action_scratch_vgprs(options) > 256u) {
    errors.emplace_back(std::string(context) +
                        " report-buffer action has insufficient scratch VGPR headroom");
    return std::nullopt;
  }

  const uint64_t address = *options.report_buffer_address;
  const uint16_t address_lo_vgpr = report_scratch_vgpr;
  const uint16_t address_hi_vgpr = static_cast<uint16_t>(report_scratch_vgpr + 1u);
  const uint16_t marker_vgpr = static_cast<uint16_t>(report_scratch_vgpr + 2u);

  const auto mov_address_lo = instrumentation::build_v_mov_b32_literal(
      address_lo_vgpr, static_cast<uint32_t>(address), arch);
  const auto mov_address_hi = instrumentation::build_v_mov_b32_literal(
      address_hi_vgpr, static_cast<uint32_t>(address >> 32u), arch);
  const auto mov_marker =
      instrumentation::build_v_mov_b32_literal(marker_vgpr, options.report_marker, arch);
  const auto store = instrumentation::build_flat_store_b32(address_lo_vgpr, marker_vgpr, arch);
  if (!mov_address_lo || !mov_address_hi || !mov_marker || !store) {
    errors.emplace_back(std::string(context) + " could not encode report-buffer action");
    return std::nullopt;
  }

  std::vector<uint32_t> words;
  words.reserve(12);
  words.insert(words.end(), mov_address_lo->begin(), mov_address_lo->end());
  words.insert(words.end(), mov_address_hi->begin(), mov_address_hi->end());
  words.insert(words.end(), mov_marker->begin(), mov_marker->end());
  words.insert(words.end(), store->begin(), store->end());
  return words;
}

[[nodiscard]] std::optional<std::vector<uint32_t>>
build_lds_check_trap_words(std::span<const uint8_t> original_bytes, const ConSanLdsSite &site,
                           rj_code_arch_t arch, const ConSanOptions &options, uint32_t delay_words,
                           uint16_t scratch_vgpr, uint16_t vcc_save_sgpr,
                           uint16_t gfx1250_vgpr_msb_mode, std::vector<std::string> &errors) {
  auto required_vgprs = lds_dword_count(site);
  if (!required_vgprs) {
    errors.emplace_back("ConSan LDS check/trap proof selected unsupported width");
    return std::nullopt;
  }

  std::array<uint32_t, 2> original_access{};
  if (!read_words_at(original_bytes, site.file_offset, std::span<uint32_t>(original_access))) {
    errors.emplace_back("ConSan LDS check/trap proof could not read LDS access");
    return std::nullopt;
  }

  std::array<uint32_t, 2> duplicate_load = original_access;
  const bool preserve_address = lds_load_clobbers_address(site);
  const uint16_t compare_scratch_vgpr =
      static_cast<uint16_t>(scratch_vgpr + static_cast<uint16_t>(*required_vgprs));
  uint16_t report_scratch_vgpr =
      static_cast<uint16_t>(compare_scratch_vgpr + lds_check_trap_extra_scratch_vgprs(site));
  if (options.report_buffer_address &&
      (arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4) &&
      report_scratch_vgpr % 2u != 0)
    ++report_scratch_vgpr;
  const uint16_t saved_address_vgpr = static_cast<uint16_t>(
      report_scratch_vgpr + static_cast<uint16_t>(report_action_scratch_vgprs(options)));
  if (site.kind == ConSanLdsAccessKind::Read) {
    if (site.dst_accvgpr && (arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4))
      duplicate_load[0] &= ~(uint32_t{1} << 25u);
    duplicate_load[1] =
        (duplicate_load[1] & 0x00FFFFFFu) | (static_cast<uint32_t>(scratch_vgpr) << 24u);
    if (preserve_address)
      duplicate_load[1] = (duplicate_load[1] & 0xFFFFFF00u) | saved_address_vgpr;
  } else if (site.kind == ConSanLdsAccessKind::Write) {
    const bool two_address = is_two_address_lds_store(site);
    const uint32_t element_width =
        two_address ? two_address_lds_element_width_bits(site).value_or(0u) : site.width_bits;
    auto word0 =
        build_ds_load_word0_from_vds_word0(original_access[0], element_width, two_address, arch);
    if (!word0) {
      errors.emplace_back("ConSan LDS check/trap proof could not encode store readback");
      return std::nullopt;
    }
    duplicate_load[0] = *word0;
    duplicate_load[1] = build_ds_load_word1(*site.addr_vgpr, scratch_vgpr);
  } else {
    errors.emplace_back("ConSan LDS check/trap proof selected unsupported site");
    return std::nullopt;
  }

  auto wait_dscnt = build_s_wait_dscnt_word(0, arch);
  auto first_compare_vgpr =
      site.dst_accvgpr && (arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4)
          ? std::optional<uint16_t>(compare_scratch_vgpr)
          : check_trap_compare_vgpr(site, 0u, arch, gfx1250_vgpr_msb_mode);
  if (!first_compare_vgpr) {
    errors.emplace_back("ConSan LDS check/trap proof selected site without compare VGPR");
    return std::nullopt;
  }
  auto mismatch_action = build_mismatch_action_words(options, arch, report_scratch_vgpr, errors,
                                                     "ConSan LDS check/trap proof");
  if (!mismatch_action)
    return std::nullopt;
  if (mismatch_action->size() > static_cast<size_t>(std::numeric_limits<int16_t>::max())) {
    errors.emplace_back("ConSan LDS check/trap proof mismatch action is too large");
    return std::nullopt;
  }
  auto skip_action = build_s_cbranch_vccz_word(static_cast<int16_t>(mismatch_action->size()), arch);
  constexpr uint16_t kRdna4VccLo = 106;
  const std::optional<uint32_t> save_vcc =
      arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4
          ? instrumentation::build_s_mov_b64(vcc_save_sgpr, kRdna4VccLo, arch)
          : std::optional<uint32_t>(build_s_mov_b32(vcc_save_sgpr, kRdna4VccLo, arch));
  const std::optional<uint32_t> restore_vcc =
      arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4
          ? instrumentation::build_s_mov_b64(kRdna4VccLo, vcc_save_sgpr, arch)
          : std::optional<uint32_t>(build_s_mov_b32(kRdna4VccLo, vcc_save_sgpr, arch));
  if (!wait_dscnt || !skip_action || !save_vcc || !restore_vcc) {
    errors.emplace_back("ConSan LDS check/trap proof could not encode sequence");
    return std::nullopt;
  }

  std::optional<std::vector<uint32_t>> masked_store_value;
  if (is_subdword_lds_store(site)) {
    std::vector<uint32_t> narrowed;
    uint16_t narrow_source = *first_compare_vgpr;
    if (is_high_d16_lds_store(site)) {
      const auto shift = instrumentation::build_v_lshrrev_b32(
          compare_scratch_vgpr, scalar_positive_inline_u32(16u), *first_compare_vgpr, arch);
      if (!shift) {
        errors.emplace_back("ConSan LDS check/trap proof could not select high store subword");
        return std::nullopt;
      }
      narrowed.push_back(*shift);
      narrow_source = compare_scratch_vgpr;
    }
    const auto mask = instrumentation::build_v_and_b32_literal(
        compare_scratch_vgpr, (uint32_t{1} << site.width_bits) - 1u, narrow_source, arch);
    if (!mask) {
      errors.emplace_back("ConSan LDS check/trap proof could not mask sub-dword store value");
      return std::nullopt;
    }
    narrowed.insert(narrowed.end(), mask->begin(), mask->end());
    masked_store_value = std::move(narrowed);
  }

  std::optional<uint32_t> d16_seed_scratch;
  if (is_d16_lds_load(site)) {
    d16_seed_scratch = build_v_mov_b32_e32_vgpr_word(scratch_vgpr, *first_compare_vgpr, arch);
    if (!d16_seed_scratch) {
      errors.emplace_back("ConSan LDS check/trap proof could not encode d16 scratch "
                          "seed move");
      return std::nullopt;
    }
  }

  uint32_t compare_mode_switch_words = 0;
  if (arch == ROCJITSU_CODE_ARCH_GFX1250 && site.kind == ConSanLdsAccessKind::Read) {
    for (uint16_t i = 0; i < *required_vgprs; ++i) {
      const auto compare_vgpr = check_trap_compare_vgpr(site, i, arch, gfx1250_vgpr_msb_mode);
      if (compare_vgpr && *compare_vgpr >= 256u)
        compare_mode_switch_words += 2u;
    }
  }
  const uint32_t total_words = static_cast<uint32_t>(
      (preserve_address ? 1u : 0u) + 2u + lds_check_trap_predelay_setup_words(site) + delay_words +
      2u + 1u + lds_check_trap_precompare_words(site) + 1u +
      (2u + mismatch_action->size()) * *required_vgprs + 1u + compare_mode_switch_words +
      (site.dst_accvgpr ? 2u * *required_vgprs : 0u));
  std::vector<uint32_t> words;
  words.reserve(total_words);
  if (preserve_address) {
    auto save_address = build_v_mov_b32_e32_vgpr_word(saved_address_vgpr, *site.addr_vgpr, arch);
    if (!save_address) {
      errors.emplace_back("ConSan LDS check/trap proof could not preserve a clobbered address");
      return std::nullopt;
    }
    words.push_back(*save_address);
  }
  words.push_back(original_access[0]);
  words.push_back(original_access[1]);
  if (d16_seed_scratch) {
    words.push_back(*wait_dscnt);
    words.push_back(*d16_seed_scratch);
  }
  if (!append_delay_words(words, arch, options, errors, "ConSan LDS check/trap proof"))
    return std::nullopt;
  words.push_back(duplicate_load[0]);
  words.push_back(duplicate_load[1]);
  words.push_back(*wait_dscnt);
  if (masked_store_value)
    words.insert(words.end(), masked_store_value->begin(), masked_store_value->end());
  words.push_back(*save_vcc);
  for (uint16_t i = 0; i < *required_vgprs; ++i) {
    auto compare_vgpr = check_trap_compare_vgpr(site, i, arch, gfx1250_vgpr_msb_mode);
    if (site.dst_accvgpr &&
        (arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4)) {
      const auto acc_read = build_cdna_accvgpr_read_b32(
          compare_scratch_vgpr, static_cast<uint16_t>(*site.dst_accvgpr + i), arch);
      if (!acc_read) {
        errors.emplace_back("ConSan LDS check/trap proof could not encode accumulator read");
        return std::nullopt;
      }
      words.insert(words.end(), acc_read->begin(), acc_read->end());
      compare_vgpr = compare_scratch_vgpr;
    }
    const uint16_t compare_mode = arch == ROCJITSU_CODE_ARCH_GFX1250 && compare_vgpr
                                      ? static_cast<uint16_t>(*compare_vgpr / 256u)
                                      : 0u;
    if (compare_mode != 0u) {
      auto select_compare_bank = instrumentation::build_s_set_vgpr_msb_transition(
          0u, static_cast<uint8_t>(compare_mode), arch);
      if (!select_compare_bank) {
        errors.emplace_back("ConSan LDS check/trap proof could not select compare VGPR bank");
        return std::nullopt;
      }
      words.push_back(*select_compare_bank);
    }
    const std::optional<uint16_t> effective_compare_vgpr =
        is_subdword_lds_store(site) ? std::optional<uint16_t>(compare_scratch_vgpr) : compare_vgpr;
    auto chunk_cmp_ne =
        effective_compare_vgpr
            ? build_v_cmp_ne_u32_e32_word(static_cast<uint16_t>(*effective_compare_vgpr & 0xFFu),
                                          static_cast<uint16_t>(scratch_vgpr + i), arch)
            : std::nullopt;
    if (!chunk_cmp_ne) {
      errors.emplace_back("ConSan LDS check/trap proof could not encode chunk compare");
      return std::nullopt;
    }
    words.push_back(*chunk_cmp_ne);
    if (compare_mode != 0u) {
      auto restore_low_bank = instrumentation::build_s_set_vgpr_msb_transition(
          static_cast<uint8_t>(compare_mode), 0u, arch);
      if (!restore_low_bank) {
        errors.emplace_back("ConSan LDS check/trap proof could not restore compare VGPR bank");
        return std::nullopt;
      }
      words.push_back(*restore_low_bank);
    }
    words.push_back(*skip_action);
    words.insert(words.end(), mismatch_action->begin(), mismatch_action->end());
  }
  words.push_back(*restore_vcc);
  return words;
}
