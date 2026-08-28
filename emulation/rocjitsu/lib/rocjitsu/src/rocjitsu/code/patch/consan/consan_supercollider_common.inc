[[nodiscard]] std::unordered_set<uint64_t>
sc_admitted_access_file_offsets(const ConSanTransformArtifacts &result, ConSanAccessOrigin origin) {
  std::unordered_set<uint64_t> admitted_text_offsets;
  admitted_text_offsets.reserve(result.observation_plan.probe_intents.size());
  for (const ConSanProbeIntent &intent : result.observation_plan.probe_intents) {
    if (intent.physical_site.code_object == result.program_inventory.code_object_id())
      admitted_text_offsets.insert(intent.physical_site.original_text_offset);
  }

  std::unordered_set<uint64_t> offsets;
  offsets.reserve(admitted_text_offsets.size());
  for (const ConSanAccessInventorySite &access : result.program_inventory.access_sites())
    if (access.origin == origin &&
        admitted_text_offsets.contains(access.physical_id.original_text_offset))
      offsets.insert(access.file_offset);
  return offsets;
}

[[nodiscard]] bool initialize_sc_access_coverage(const ConSanOptions &options,
                                                 ConSanTransformArtifacts &result) {
  ConSanObservationProduct observation = assemble_consan_observation_product(
      result.program_inventory, static_cast<const ConSanRequest &>(options),
      static_cast<const ConSanDebugOverrides &>(options));
  const bool valid = observation.valid();
  result.observation_plan = std::move(observation.plan);
  result.coverage_ledger = std::move(observation.initial_coverage);
  result.errors.insert(result.errors.end(),
                       std::make_move_iterator(observation.diagnostics.begin()),
                       std::make_move_iterator(observation.diagnostics.end()));
  return valid;
}

[[nodiscard]] std::vector<ConSanProbeIntentId>
sc_pending_access_intent_ids(const ConSanTransformArtifacts &result, const PhysicalSiteId &site) {
  std::vector<ConSanProbeIntentId> intent_ids;
  for (const ConSanProbeIntent &intent : result.observation_plan.probe_intents) {
    if (intent.kind != ConSanProbeIntentKind::RedundantAccessObservation ||
        intent.physical_site != site)
      continue;
    const ConSanIntentCoverageEntry *entry = result.coverage_ledger.intent_entry(intent.id);
    if (entry != nullptr && entry->lowering == ConSanLoweringOutcomeKind::Pending)
      intent_ids.push_back(intent.id);
  }
  return intent_ids;
}

[[nodiscard]] bool publish_sc_access_rejection(ConSanTransformArtifacts &result,
                                               const PhysicalSiteId &site,
                                               ConSanLoweringOutcomeKind outcome,
                                               std::string detail) {
  std::vector<ConSanProbeIntentId> intent_ids = sc_pending_access_intent_ids(result, site);
  if (intent_ids.empty())
    return true;
  auto commit = make_consan_committed_lowering(result.observation_plan, intent_ids,
                                               std::span<const ConSanCommittedLoweringLocation>{},
                                               outcome, std::move(detail));
  if (!commit || !result.publish_lowering_commit(std::move(*commit))) {
    result.errors.emplace_back(
        "ConSan SuperCollider could not publish an intent-bound access rejection");
    return false;
  }
  return true;
}

[[nodiscard]] std::optional<ConSanCommittedLowering> make_sc_access_lowering_commit(
    const ConSanTransformArtifacts &result, std::span<const ConSanProbeIntentId> intent_ids,
    const ConSanAccessInventorySite &site, const ConSanPatchInfo &patch) {
  std::vector<ConSanCommittedLoweringLocation> locations;
  locations.push_back({
      .original_site = site.physical_id,
      .emitted_text_offset = patch.anchor_offset,
      .emitted_size = patch.original_size,
      .relocated_guest_text_offset =
          patch.trampoline_size == 0u ? patch.relocated_guest_instruction_offset : std::nullopt,
  });
  if (patch.trampoline_size != 0u) {
    locations.push_back({
        .original_site = site.physical_id,
        .emitted_text_offset = patch.trampoline_offset,
        .emitted_size = patch.trampoline_size,
        .relocated_guest_text_offset = patch.relocated_guest_instruction_offset,
    });
  }
  return make_consan_committed_lowering(result.observation_plan, intent_ids, locations,
                                        ConSanLoweringOutcomeKind::Instrumented);
}

/// Publish an explicit placement rejection for every SuperCollider access
/// intent that no site-local lowering transaction consumed. This fallback
/// consults only immutable intent identity and the ledger's pending state; it
/// never reconstructs success or failure from patch telemetry.
void publish_unplaced_sc_access_rejections(ConSanTransformArtifacts &result) {
  for (const ConSanProbeIntent &intent : result.observation_plan.probe_intents) {
    if (intent.kind != ConSanProbeIntentKind::RedundantAccessObservation)
      continue;
    const ConSanIntentCoverageEntry *entry = result.coverage_ledger.intent_entry(intent.id);
    if (entry == nullptr || entry->lowering != ConSanLoweringOutcomeKind::Pending)
      continue;
    (void)publish_sc_access_rejection(result, intent.physical_site,
                                      ConSanLoweringOutcomeKind::PlacementRejected,
                                      "SuperCollider lowerer selected no access placement");
  }
}

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

[[nodiscard]] uint16_t sc_vcc_save_sgpr_width(const ConSanTargetProfile &target) {
  return target.encoding_family == ConSanEncodingFamily::Gfx9Cdna3 ||
                 target.encoding_family == ConSanEncodingFamily::Gfx9Cdna4
             ? 2u
             : 1u;
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
    uint16_t allocated_sgpr_count, std::optional<uint16_t> min_preferred_sgpr,
    uint16_t selection_limit, const ConSanTargetProfile &target, const ConSanOptions &options) {
  if (inst == nullptr || owner_liveness.empty() ||
      std::ranges::any_of(owner_liveness, [](const auto *liveness) { return liveness == nullptr; }))
    return std::nullopt;
  const uint16_t width = sc_vcc_save_sgpr_width(target);
  const uint16_t limit = std::min<uint16_t>(selection_limit, REGISTER_SET_ALLOCATABLE_SGPRS);
  const auto find_free = [&](uint16_t start, uint16_t end) -> std::optional<uint16_t> {
    if (width == 2u && (start & 1u) != 0u)
      ++start;
    for (uint16_t base = start; static_cast<uint32_t>(base) + width <= end;
         base = static_cast<uint16_t>(base + width)) {
      if (sc_sgpr_range_overlaps_delay_source(base, width, options))
        continue;
      const RegisterRef ref{RegClass::SGPR, base, static_cast<uint8_t>(width)};
      if (std::ranges::all_of(owner_liveness, [&](const LivenessAnalysis *liveness) {
            return !liveness->any_live_before(*inst, ref);
          }))
        return base;
    }
    return std::nullopt;
  };
  const uint16_t allocated_limit = std::min(allocated_sgpr_count, limit);
  if (target.encoding_family == ConSanEncodingFamily::Gfx9Cdna3 ||
      target.encoding_family == ConSanEncodingFamily::Gfx9Cdna4) {
    // CDNA SGPR allocation participates in occupancy. Reuse an all-owner dead
    // hole inside the current allocation before growing every descriptor.
    if (auto existing = find_free(0, allocated_limit))
      return existing;
    const uint16_t fresh_start =
        std::max(allocated_limit, min_preferred_sgpr.value_or(allocated_limit));
    return find_free(fresh_start, limit);
  }

  // RDNA allocation is effectively fixed, so retain the high-first policy.
  // Besides avoiding churn in established physical workloads, this keeps
  // instrumentation temporaries away from the guest's referenced scalar
  // range whenever a fresh register is available.
  if (min_preferred_sgpr) {
    if (auto preferred = find_free(*min_preferred_sgpr, limit))
      return preferred;
  }
  return find_free(0, limit);
}

[[nodiscard]] ConSanPatchKind lds_check_trap_patch_kind(const ConSanAccessInventorySite &access,
                                                        bool use_local_cave) {
  if (access.kind == ConSanLdsAccessKind::Write)
    return use_local_cave ? ConSanPatchKind::LocalCaveLdsStoreCheckTrap
                          : ConSanPatchKind::InlineLdsStoreCheckTrap;
  return use_local_cave ? ConSanPatchKind::LocalCaveLdsLoadCheckTrap
                        : ConSanPatchKind::InlineLdsLoadCheckTrap;
}

[[nodiscard]] bool is_d16_lds_load(const ConSanAccessInventorySite &access) {
  return access.mnemonic == "ds_load_u8_d16" || access.mnemonic == "ds_load_u8_d16_hi" ||
         access.mnemonic == "ds_load_i8_d16" || access.mnemonic == "ds_load_i8_d16_hi" ||
         access.mnemonic == "ds_load_u16_d16" || access.mnemonic == "ds_load_u16_d16_hi" ||
         access.mnemonic == "ds_read_u8_d16" || access.mnemonic == "ds_read_u8_d16_hi" ||
         access.mnemonic == "ds_read_i8_d16" || access.mnemonic == "ds_read_i8_d16_hi" ||
         access.mnemonic == "ds_read_u16_d16" || access.mnemonic == "ds_read_u16_d16_hi";
}

[[nodiscard]] uint32_t
lds_check_trap_predelay_setup_words(const ConSanAccessInventorySite &access) {
  return is_d16_lds_load(access) ? 2u : 0u;
}

[[nodiscard]] bool is_subdword_lds_store(const ConSanAccessInventorySite &access) {
  return access.kind == ConSanLdsAccessKind::Write && access.decoded_width_bits < 32u;
}

[[nodiscard]] bool is_high_d16_lds_store(const ConSanAccessInventorySite &access) {
  return access.mnemonic == "ds_store_b8_d16_hi" || access.mnemonic == "ds_store_b16_d16_hi";
}

struct SplitTwoAddressLdsExpansionPlan {
  TwoAddressLdsByteOffsets offsets;
  uint16_t element_dwords = 0;

  [[nodiscard]] uint16_t large_offset_count() const {
    return static_cast<uint16_t>(offsets.first > UINT16_MAX) +
           static_cast<uint16_t>(offsets.second > UINT16_MAX);
  }

  [[nodiscard]] uint32_t access_word_count() const { return 4u + 2u * large_offset_count(); }

  [[nodiscard]] uint32_t total_additional_word_count() const {
    // The baseline sequence replays one two-word instruction before and after
    // the delay. Splitting both replays adds two single-address instructions
    // plus any address-adjustment literal pair for each out-of-range offset.
    return 2u * (access_word_count() - 2u);
  }
};

[[nodiscard]] std::optional<SplitTwoAddressLdsExpansionPlan>
split_two_address_lds_expansion_plan(std::span<const uint8_t> original_bytes,
                                     const ConSanAccessInventorySite &access,
                                     const ConSanTargetProfile &target) {
  if (!target.requires_split_two_address_lds_relocation ||
      (!is_two_address_lds_load(access.mnemonic) && !is_two_address_lds_store(access.mnemonic)))
    return std::nullopt;
  const auto width_bits = two_address_lds_element_width_bits(access.mnemonic);
  std::array<uint32_t, 1> words{};
  if (!width_bits || !read_words_at(original_bytes, access.file_offset, std::span<uint32_t>(words)))
    return std::nullopt;
  const auto offsets = two_address_lds_byte_offsets(access.mnemonic, words.front());
  if (!offsets)
    return std::nullopt;
  return SplitTwoAddressLdsExpansionPlan{
      .offsets = *offsets,
      .element_dwords = static_cast<uint16_t>(*width_bits / 32u),
  };
}

[[nodiscard]] uint16_t lds_check_trap_extra_scratch_vgprs(
    const ConSanAccessInventorySite &site,
    const std::optional<SplitTwoAddressLdsExpansionPlan> &expansion) {
  return static_cast<uint16_t>(
      (is_subdword_lds_store(site) || site.operands.destination_accvgpr ? 1u : 0u) +
      (expansion && expansion->large_offset_count() != 0u ? 1u : 0u));
}

[[nodiscard]] uint32_t lds_check_trap_precompare_words(const ConSanAccessInventorySite &access) {
  if (!is_subdword_lds_store(access))
    return 0u;
  return is_high_d16_lds_store(access) ? 3u : 2u;
}

[[nodiscard]] std::optional<std::array<uint32_t, 2>>
build_cdna_accvgpr_read_b32(uint16_t dst_vgpr, uint16_t src_accvgpr,
                            const ConSanTargetProfile &target) {
  if ((target.encoding_family != ConSanEncodingFamily::Gfx9Cdna3 &&
       target.encoding_family != ConSanEncodingFamily::Gfx9Cdna4) ||
      dst_vgpr > 255u || src_accvgpr > 255u)
    return std::nullopt;
  if (target.encoding_family == ConSanEncodingFamily::Gfx9Cdna3) {
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
                                                       const ConSanTargetProfile &target) {
  const rj_code_arch_t arch = target.arch;
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
  if (!append_pc_delta_builder(words, arch, pc_sgpr, delta, /*minimum_words=*/3u,
                               /*prefer_literal64=*/
                               target.direct_call_form == ConSanDirectCallForm::SCallI64))
    return false;
  const auto restore_scc =
      instrumentation::build_s_cmp_lg_u32(saved_scc_sgpr, scalar_positive_inline_u32(0), arch);
  if (!restore_scc)
    return false;
  words.push_back(*restore_scc);
  words.push_back(build_s_setpc_b64(pc_sgpr, arch));
  return true;
}

[[nodiscard]] bool append_lds_check_trap_indirect_return(
    std::vector<uint32_t> &words, uint64_t words_text_offset, uint64_t target_text_offset,
    uint16_t pc_sgpr, uint16_t captured_scc_sgpr, const ConSanTargetProfile &target) {
  const rj_code_arch_t arch = target.arch;
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
  if (!append_pc_delta_builder(words, arch, pc_sgpr, delta, /*minimum_words=*/3u,
                               /*prefer_literal64=*/
                               target.direct_call_form == ConSanDirectCallForm::SCallI64))
    return false;
  const auto restore_scc =
      instrumentation::build_s_cmp_lg_u32(captured_scc_sgpr, scalar_positive_inline_u32(0), arch);
  if (!restore_scc)
    return false;
  words.push_back(*restore_scc);
  words.push_back(build_s_setpc_b64(pc_sgpr, arch));
  return true;
}

[[nodiscard]] std::optional<uint16_t> choose_lds_check_trap_indirect_pc_sgpr(
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
          return !liveness->any_live_before(*inst, ref);
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
          return !liveness->any_live_before(*inst, ref);
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
choose_lds_check_trap_indirect_scalar_scratch(
    const Instruction *inst, std::span<const LivenessAnalysis *const> owner_liveness,
    std::optional<uint16_t> min_preferred_vcc_save, const ConSanTargetProfile &target) {
  if (inst == nullptr || owner_liveness.empty() ||
      std::ranges::any_of(owner_liveness, [](const auto *liveness) { return liveness == nullptr; }))
    return std::nullopt;

  uint16_t pair_search_start = min_preferred_vcc_save.value_or(0);
  while (auto pair = owner_liveness.front()->find_free_sgpr_pair(inst, pair_search_start)) {
    const RegisterRef pair_ref{RegClass::SGPR, *pair, 2u};
    if (!std::ranges::all_of(owner_liveness, [&](const LivenessAnalysis *liveness) {
          return !liveness->any_live_before(*inst, pair_ref);
        })) {
      if (*pair + 2u >= REGISTER_SET_ALLOCATABLE_SGPRS)
        return std::nullopt;
      pair_search_start = static_cast<uint16_t>(*pair + 2u);
      continue;
    }
    const uint16_t vcc_width = sc_vcc_save_sgpr_width(target);
    const auto find_disjoint_vcc = [&](uint16_t start) -> std::optional<uint16_t> {
      if (vcc_width == 2u) {
        while (auto vcc = owner_liveness.front()->find_free_sgpr_pair(inst, start)) {
          const RegisterRef vcc_ref{RegClass::SGPR, *vcc, 2u};
          if (!sc_sgpr_ranges_overlap(*vcc, 2u, *pair, 2u) &&
              std::ranges::all_of(owner_liveness, [&](const LivenessAnalysis *liveness) {
                return !liveness->any_live_before(*inst, vcc_ref);
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
              return !liveness->any_live_before(*inst, vcc_ref);
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
          inst, owner_liveness, *vcc_save, vcc_width, *pair, min_preferred_vcc_save.value_or(0));
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

[[nodiscard]] bool lds_report_action_reuses_readback(const ConSanOptions &options,
                                                     uint16_t readback_vgprs) {
  // The automatic SuperCollider report is an idempotent one-word marker. Once
  // a comparison has found a mismatch, its LDS readback values need not
  // survive the marker store: clobbering them can only cause another marker
  // store in a later chunk. Reuse a wide readback tuple instead of requiring
  // three additional ordinary VGPRs at an AccVGPR boundary.
  return options.report_buffer_address.has_value() && readback_vgprs >= 3u;
}

[[nodiscard]] std::optional<uint32_t> mismatch_action_word_count(const ConSanOptions &options,
                                                                 const ConSanTargetProfile &target,
                                                                 std::vector<std::string> &errors,
                                                                 std::string_view context) {
  const rj_code_arch_t arch = target.arch;
  if (!options.report_buffer_address)
    return 1u;
  const auto literal = instrumentation::build_v_mov_b32_literal(0u, 0u, arch);
  const auto store = instrumentation::build_flat_store_b32(0u, 2u, arch);
  if (!literal || !store) {
    errors.emplace_back(std::string(context) + " could not size report-buffer action");
    return std::nullopt;
  }
  return static_cast<uint32_t>(3u * literal->size() + store->size());
}

[[nodiscard]] std::optional<std::vector<uint32_t>>
build_mismatch_action_words(const ConSanOptions &options, const ConSanTargetProfile &target,
                            uint16_t report_scratch_vgpr, std::vector<std::string> &errors,
                            std::string_view context) {
  const rj_code_arch_t arch = target.arch;
  if (!options.report_buffer_address) {
    auto trap = instrumentation::build_s_trap(0, arch);
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
  if (target.requires_even_vgpr_tuples && report_scratch_vgpr % 2u != 0u) {
    errors.emplace_back(std::string(context) +
                        " report-buffer action selected a misaligned address tuple");
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
build_split_single_address_lds_pair(const SplitTwoAddressLdsExpansionPlan &plan,
                                    uint16_t address_vgpr, uint16_t first_data_vgpr,
                                    uint16_t second_data_vgpr, uint16_t adjusted_address_vgpr,
                                    bool load, const ConSanTargetProfile &target) {
  if (address_vgpr > 255u || adjusted_address_vgpr > 255u || first_data_vgpr > 255u ||
      second_data_vgpr > 255u ||
      (plan.element_dwords == 2u && (first_data_vgpr > 254u || second_data_vgpr > 254u)))
    return std::nullopt;

  const uint16_t op =
      load ? (plan.element_dwords == 1u ? cdna5::kDsLoadB32Vds : cdna5::kDsLoadB64Vds)
           : (plan.element_dwords == 1u ? cdna5::kDsStoreB32Vds : cdna5::kDsStoreB64Vds);
  std::vector<uint32_t> words;
  words.reserve(plan.access_word_count());
  const auto append_access = [&](uint32_t byte_offset, uint16_t data_vgpr) -> bool {
    uint16_t effective_address_vgpr = address_vgpr;
    uint16_t immediate = 0;
    if (byte_offset > UINT16_MAX) {
      const auto adjust = instrumentation::build_v_add_u32_literal(
          adjusted_address_vgpr, adjusted_address_vgpr, byte_offset, address_vgpr, target.arch);
      if (!adjust)
        return false;
      words.insert(words.end(), adjust->begin(), adjust->end());
      effective_address_vgpr = adjusted_address_vgpr;
    } else {
      immediate = static_cast<uint16_t>(byte_offset);
    }
    cdna5::VdsBuilderFields fields{
        .offset0 = static_cast<uint8_t>(immediate),
        .offset1 = static_cast<uint8_t>(immediate >> 8u),
        .addr = static_cast<uint8_t>(effective_address_vgpr),
    };
    if (load)
      fields.vdst = static_cast<uint8_t>(data_vgpr);
    else
      fields.data0 = static_cast<uint8_t>(data_vgpr);
    const auto access = cdna5::build_vds(op, fields);
    words.insert(words.end(), access.begin(), access.end());
    return true;
  };
  if (!append_access(plan.offsets.first, first_data_vgpr) ||
      !append_access(plan.offsets.second, second_data_vgpr) ||
      words.size() != plan.access_word_count())
    return std::nullopt;
  return words;
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_lds_check_trap_words(
    std::span<const uint8_t> original_bytes, const ConSanAccessInventorySite &site,
    const ConSanTargetProfile &target, const ConSanOptions &options, uint32_t delay_words,
    uint16_t scratch_vgpr, uint16_t vcc_save_sgpr, uint16_t gfx1250_vgpr_msb_mode,
    std::optional<uint32_t> &guest_instruction_word_offset, std::vector<std::string> &errors) {
  const rj_code_arch_t arch = target.arch;
  guest_instruction_word_offset.reset();
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
  const auto split_two_address = split_two_address_lds_expansion_plan(original_bytes, site, target);
  const uint16_t compare_scratch_vgpr =
      static_cast<uint16_t>(scratch_vgpr + static_cast<uint16_t>(*required_vgprs));
  const uint16_t core_scratch_end = static_cast<uint16_t>(
      compare_scratch_vgpr + lds_check_trap_extra_scratch_vgprs(site, split_two_address));
  const bool reuse_readback_for_report =
      lds_report_action_reuses_readback(options, static_cast<uint16_t>(*required_vgprs));
  uint16_t report_scratch_vgpr = reuse_readback_for_report ? scratch_vgpr : core_scratch_end;
  if (!reuse_readback_for_report && options.report_buffer_address &&
      target.requires_even_vgpr_tuples && report_scratch_vgpr % 2u != 0)
    ++report_scratch_vgpr;
  const uint16_t report_scratch_end = static_cast<uint16_t>(
      report_scratch_vgpr + static_cast<uint16_t>(report_action_scratch_vgprs(options)));
  const uint16_t saved_address_vgpr = std::max(core_scratch_end, report_scratch_end);
  if (site.kind == ConSanLdsAccessKind::Read) {
    if (site.operands.destination_accvgpr &&
        target.accumulator_model == ConSanAccumulatorModel::DescriptorPartitioned)
      duplicate_load[0] &= ~(uint32_t{1} << 25u);
    duplicate_load[1] =
        (duplicate_load[1] & 0x00FFFFFFu) | (static_cast<uint32_t>(scratch_vgpr) << 24u);
    if (preserve_address)
      duplicate_load[1] = (duplicate_load[1] & 0xFFFFFF00u) | saved_address_vgpr;
  } else if (site.kind == ConSanLdsAccessKind::Write) {
    const bool two_address = is_two_address_lds_store(site.mnemonic);
    const uint32_t element_width =
        two_address ? two_address_lds_element_width_bits(site.mnemonic).value_or(0u)
                    : site.decoded_width_bits;
    auto word0 =
        build_ds_load_word0_from_vds_word0(original_access[0], element_width, two_address, arch);
    if (!word0) {
      errors.emplace_back("ConSan LDS check/trap proof could not encode store readback");
      return std::nullopt;
    }
    duplicate_load[0] = *word0;
    duplicate_load[1] = build_ds_load_word1(*site.operands.address_vgpr, scratch_vgpr);
  } else {
    errors.emplace_back("ConSan LDS check/trap proof selected unsupported site");
    return std::nullopt;
  }

  if (!lds_scratch_tuple_base_is_valid(site, arch, scratch_vgpr)) {
    errors.emplace_back("ConSan LDS check/trap proof selected a misaligned tuple scratch base");
    return std::nullopt;
  }

  auto wait_dscnt = build_s_wait_dscnt_word(0, arch);
  auto first_compare_vgpr =
      site.operands.destination_accvgpr &&
              target.accumulator_model == ConSanAccumulatorModel::DescriptorPartitioned
          ? std::optional<uint16_t>(compare_scratch_vgpr)
          : check_trap_compare_vgpr(site, 0u, arch, gfx1250_vgpr_msb_mode);
  if (!first_compare_vgpr) {
    errors.emplace_back("ConSan LDS check/trap proof selected site without compare VGPR");
    return std::nullopt;
  }
  auto mismatch_action = build_mismatch_action_words(options, target, report_scratch_vgpr, errors,
                                                     "ConSan LDS check/trap proof");
  if (!mismatch_action)
    return std::nullopt;
  if (mismatch_action->size() > static_cast<size_t>(std::numeric_limits<int16_t>::max())) {
    errors.emplace_back("ConSan LDS check/trap proof mismatch action is too large");
    return std::nullopt;
  }
  auto skip_action =
      instrumentation::build_s_cbranch_vccz(static_cast<int16_t>(mismatch_action->size()), arch);
  constexpr uint16_t kRdna4VccLo = 106;
  const bool uses_gfx9_cdna_encoding = target.encoding_family == ConSanEncodingFamily::Gfx9Cdna3 ||
                                       target.encoding_family == ConSanEncodingFamily::Gfx9Cdna4;
  const std::optional<uint32_t> save_vcc =
      uses_gfx9_cdna_encoding
          ? instrumentation::build_s_mov_b64(vcc_save_sgpr, kRdna4VccLo, arch)
          : std::optional<uint32_t>(build_s_mov_b32(vcc_save_sgpr, kRdna4VccLo, arch));
  const std::optional<uint32_t> restore_vcc =
      uses_gfx9_cdna_encoding
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
        compare_scratch_vgpr, (uint32_t{1} << site.decoded_width_bits) - 1u, narrow_source, arch);
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
  if (target.has_selectable_vgpr_bank && site.kind == ConSanLdsAccessKind::Read) {
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
      (site.operands.destination_accvgpr ? 2u * *required_vgprs : 0u) +
      (split_two_address ? split_two_address->total_additional_word_count() : 0u));
  std::vector<uint32_t> words;
  words.reserve(total_words);
  if (preserve_address) {
    auto save_address =
        build_v_mov_b32_e32_vgpr_word(saved_address_vgpr, *site.operands.address_vgpr, arch);
    if (!save_address) {
      errors.emplace_back("ConSan LDS check/trap proof could not preserve a clobbered address");
      return std::nullopt;
    }
    words.push_back(*save_address);
  }
  if (split_two_address) {
    const uint16_t replay_address_vgpr =
        preserve_address ? saved_address_vgpr : *site.operands.address_vgpr;
    const std::optional<uint16_t> first_data_vgpr = site.kind == ConSanLdsAccessKind::Read
                                                        ? site.operands.destination_vgpr
                                                        : site.operands.data_vgpr;
    const std::optional<uint16_t> second_data_vgpr =
        site.kind == ConSanLdsAccessKind::Read && first_data_vgpr
            ? std::optional<uint16_t>(
                  static_cast<uint16_t>(*first_data_vgpr + split_two_address->element_dwords))
            : site.operands.second_data_vgpr;
    const auto split =
        first_data_vgpr && second_data_vgpr
            ? build_split_single_address_lds_pair(
                  *split_two_address, replay_address_vgpr, *first_data_vgpr, *second_data_vgpr,
                  compare_scratch_vgpr, site.kind == ConSanLdsAccessKind::Read, target)
            : std::nullopt;
    if (!split) {
      errors.emplace_back("ConSan LDS check/trap proof could not split gfx1250 two-address access");
      return std::nullopt;
    }
    words.insert(words.end(), split->begin(), split->end());
  } else {
    guest_instruction_word_offset = static_cast<uint32_t>(words.size());
    words.push_back(original_access[0]);
    words.push_back(original_access[1]);
  }
  if (d16_seed_scratch) {
    words.push_back(*wait_dscnt);
    words.push_back(*d16_seed_scratch);
  }
  if (!append_delay_words(words, arch, options, errors, "ConSan LDS check/trap proof"))
    return std::nullopt;
  if (split_two_address) {
    const uint16_t replay_address_vgpr =
        preserve_address ? saved_address_vgpr : *site.operands.address_vgpr;
    const auto split = build_split_single_address_lds_pair(
        *split_two_address, replay_address_vgpr, scratch_vgpr,
        static_cast<uint16_t>(scratch_vgpr + split_two_address->element_dwords),
        compare_scratch_vgpr, true, target);
    if (!split) {
      errors.emplace_back(
          "ConSan LDS check/trap proof could not split gfx1250 two-address readback");
      return std::nullopt;
    }
    words.insert(words.end(), split->begin(), split->end());
  } else {
    words.push_back(duplicate_load[0]);
    words.push_back(duplicate_load[1]);
  }
  words.push_back(*wait_dscnt);
  if (masked_store_value)
    words.insert(words.end(), masked_store_value->begin(), masked_store_value->end());
  words.push_back(*save_vcc);
  for (uint16_t i = 0; i < *required_vgprs; ++i) {
    auto compare_vgpr = check_trap_compare_vgpr(site, i, arch, gfx1250_vgpr_msb_mode);
    if (site.operands.destination_accvgpr && uses_gfx9_cdna_encoding) {
      const auto acc_read = build_cdna_accvgpr_read_b32(
          compare_scratch_vgpr, static_cast<uint16_t>(*site.operands.destination_accvgpr + i),
          target);
      if (!acc_read) {
        errors.emplace_back("ConSan LDS check/trap proof could not encode accumulator read");
        return std::nullopt;
      }
      words.insert(words.end(), acc_read->begin(), acc_read->end());
      compare_vgpr = compare_scratch_vgpr;
    }
    const uint16_t compare_mode = target.has_selectable_vgpr_bank && compare_vgpr
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
