template <typename PlannedPatch, typename BuildWords, typename MakePatchInfo,
          typename ApplyExtraRequirements>
[[nodiscard]] bool apply_inline_moi_access_patches(
    std::span<const uint8_t> bytes, const std::vector<PlannedPatch> &planned_patches,
    const MoiDescriptorVgprRequirements &descriptor_requirements,
    const MoiDescriptorSgprRequirements &scalar_requirements, std::string_view probe_name,
    rj_code_arch_t arch, BuildWords build_words, MakePatchInfo make_patch_info,
    ApplyExtraRequirements apply_extra_requirements, ConSanTransformArtifacts &result) {
  result.replacement.assign(bytes.begin(), bytes.end());
  for (const PlannedPatch &planned_patch : planned_patches) {
    const ConSanMoiCandidate &candidate = *planned_patch.candidate;
    auto words = build_words(planned_patch);
    if (!words) {
      result.replacement.clear();
      return false;
    }
    const uint64_t patch_bytes = static_cast<uint64_t>(words->size() * sizeof(uint32_t));
    if (patch_bytes != planned_patch.placement.body_size) {
      result.errors.emplace_back("ConSan MOI " + std::string(probe_name) +
                                 " final patch size changed");
      result.replacement.clear();
      return false;
    }
    if (candidate.file_offset > result.replacement.size() ||
        patch_bytes > result.replacement.size() - candidate.file_offset) {
      result.errors.emplace_back("ConSan MOI " + std::string(probe_name) +
                                 " inline patch exceeds the code object");
      result.replacement.clear();
      return false;
    }
    std::memcpy(result.replacement.data() + candidate.file_offset, words->data(),
                static_cast<size_t>(patch_bytes));
  }

  if (!apply_descriptor_requirements(result.replacement, result, descriptor_requirements, arch,
                                     result.errors) ||
      !apply_sgpr_descriptor_requirements(result.replacement, result, scalar_requirements,
                                          result.errors) ||
      !apply_extra_requirements(result.replacement, result)) {
    result.replacement.clear();
    return false;
  }

  for (const PlannedPatch &planned_patch : planned_patches)
    result.patches.push_back(make_patch_info(planned_patch));
  result.mark_modified();
  return true;
}

/// Apply the descriptor growth shared by every appended MOI access engine.
///
/// VGPR, SGPR, and private-memory requirements come from the common access
/// plan. InlineShadow may additionally supply an LDS requirement for its
/// workgroup-local shadow; replay-style engines pass no LDS requirement.
[[nodiscard]] bool apply_moi_appended_access_descriptor_requirements(
    CodeObjectPatcher &patcher, const AmdGpuCodeObject &code_object, std::span<const uint8_t> bytes,
    const MoiOptions &options, const MoiDescriptorVgprRequirements &descriptor_requirements,
    const MoiDescriptorSgprRequirements &scalar_requirements,
    const MoiDescriptorPrivateRequirements &private_requirements,
    const MoiDescriptorLdsRequirements *lds_requirements, rj_code_arch_t arch,
    ConSanTransformArtifacts &result) {
  return apply_descriptor_requirements(patcher, code_object, bytes, result, descriptor_requirements,
                                       arch, result.errors) &&
         apply_spill_descriptor_requirements(patcher, code_object, bytes, result,
                                             private_requirements, result.errors) &&
         apply_sgpr_descriptor_requirements(patcher, result, code_object, scalar_requirements,
                                            result.errors) &&
         (lds_requirements == nullptr ||
          apply_lds_descriptor_requirements(patcher, result, code_object, *lds_requirements,
                                            options, arch, result.errors));
}

/// Initialize one appended MOI access image and materialize its shared direct
/// relay reservoirs.
///
/// This is byte-placement mechanics only: the caller has already selected the
/// engine's evidence and patch plans. The body prefix is filled with target
/// NOPs so every subsequently planned offset remains stable.
[[nodiscard]] bool initialize_moi_appended_access_text(
    CodeObjectPatcher &patcher, bool use_indirect_appended, uint64_t indirect_body_begin,
    const BranchOnlyDirectRelayReservoirSet &direct_reservoirs, std::string_view probe_name,
    rj_code_arch_t arch, std::vector<uint8_t> &new_text, std::vector<ConSanPatchInfo> &patches,
    ConSanTransformArtifacts &result) {
  const std::span<const uint8_t> old_text = patcher.text_bytes();
  if (old_text.empty()) {
    result.errors.emplace_back("ConSan MOI " + std::string(probe_name) + " found no .text section");
    return false;
  }
  new_text.assign(old_text.begin(), old_text.end());
  if (use_indirect_appended) {
    while (new_text.size() < indirect_body_begin)
      append_word_bytes(new_text, build_s_nop(0, arch));
  }
  for (const BranchOnlyDirectRelayReservoir &reservoir : direct_reservoirs.reservoirs) {
    std::string relay_error;
    if (!BranchOnlyRelayRouter::emit_direct_reservoir(new_text, reservoir, arch, patches,
                                                      &relay_error)) {
      result.errors.emplace_back("ConSan MOI " + std::string(probe_name) + " " + relay_error);
      return false;
    }
  }
  return true;
}

/// Group planned access patches by the dense dispatcher selected during
/// placement. The map preserves dispatcher order so generated bodies retain
/// the offsets proved by planning.
template <typename PlannedPatch>
[[nodiscard]] std::map<uint64_t, std::vector<const PlannedPatch *>>
group_moi_dense_access_patches(const std::vector<PlannedPatch> &planned_patches) {
  std::map<uint64_t, std::vector<const PlannedPatch *>> groups;
  for (const PlannedPatch &planned_patch : planned_patches) {
    if (planned_patch.dense_call_anchor)
      groups[*planned_patch.dense_dispatcher_offset].push_back(&planned_patch);
  }
  return groups;
}

/// Emit the shared dense access-dispatcher mechanics for one MOI engine.
///
/// `eligible` retains the few engine-specific ABI preconditions that are not
/// implied by the common patch plan. It observes the representative patch and
/// its owner-bound options but does not perform placement or native emission.
template <typename PlannedPatch, typename Eligible>
[[nodiscard]] bool emit_moi_dense_access_groups(
    const std::map<uint64_t, std::vector<const PlannedPatch *>> &groups,
    const std::map<uint64_t, MoiDenseEntryHost> &dense_entry_hosts, const MoiOptions &options,
    const ConSanMoiOperatingPoint &operating_point, uint64_t access_slot_words,
    bool collapse_spill_router_for_explicit_key, std::string_view probe_name, rj_code_arch_t arch,
    std::vector<uint8_t> &new_text, std::vector<ConSanPatchInfo> &patches,
    std::vector<std::string> &errors, Eligible eligible) {
  for (const auto &[dispatcher_offset, group] : groups) {
    if (group.empty() || !group.front()->entry_island_offset)
      continue;
    MoiOptions group_options = options;
    if (!apply_moi_transient_sgpr_assignment(
            group_options, operating_point,
            group.front()->resources.owner_descriptor_file_offsets) ||
        !eligible(*group.front(), group_options)) {
      continue;
    }
    if (!emit_moi_dense_access_group(std::span<const PlannedPatch *const>(group), dispatcher_offset,
                                     dense_entry_hosts, group_options, arch, access_slot_words,
                                     collapse_spill_router_for_explicit_key, probe_name, new_text,
                                     patches, errors)) {
      return false;
    }
  }
  return true;
}

/// Emit the relocated entry-host bodies shared by dense MOI access routers.
///
/// Every engine restores the displaced host prefix and returns through the
/// same owner-bound indirect-jump ABI. `eligible` retains any stronger flavor
/// precondition, while `finalize_host_words` adds only a genuinely different
/// suffix or fixed-size check before the common byte publication. InlineShadow
/// keeps SCC in the indirect-jump assignment itself; the other engines use the
/// shared special-state assignment. `use_indirect_jump_scc_save` states that
/// remaining semantic distinction without duplicating the emission loop.
template <typename PlannedPatch, typename Eligible, typename FinalizeHostWords>
[[nodiscard]] bool emit_moi_dense_entry_hosts(
    const std::map<uint64_t, MoiDenseEntryHost> &dense_entry_hosts,
    const std::map<uint64_t, std::vector<const PlannedPatch *>> &groups, const MoiOptions &options,
    const ConSanMoiOperatingPoint &operating_point, uint32_t indirect_island_words,
    bool pad_to_planned_offset, bool use_indirect_jump_scc_save, std::string_view probe_name,
    std::string_view lost_assignment_error, std::string_view return_error, rj_code_arch_t arch,
    std::vector<uint8_t> &new_text, std::vector<std::string> &errors, Eligible eligible,
    FinalizeHostWords finalize_host_words) {
  for (const auto &[dispatcher, host] : dense_entry_hosts) {
    if (pad_to_planned_offset) {
      while (new_text.size() < host.body_offset)
        append_word_bytes(new_text, build_s_nop(0u, arch));
    }
    if (new_text.size() != host.body_offset) {
      errors.emplace_back("ConSan MOI " + std::string(probe_name) +
                          " emitted a stale dense entry-host mapping");
      return false;
    }
    const auto group = groups.find(dispatcher);
    if (group == groups.end()) {
      const size_t reserved_words = host.displaced_words.size() + indirect_island_words;
      for (size_t word = 0; word < reserved_words; ++word)
        append_word_bytes(new_text, build_s_nop(0, arch));
      continue;
    }
    const PlannedPatch &representative = *group->second.front();
    MoiOptions group_options = options;
    if (!apply_moi_transient_sgpr_assignment(
            group_options, operating_point,
            representative.resources.owner_descriptor_file_offsets) ||
        !eligible(representative, group_options)) {
      errors.emplace_back(std::string(lost_assignment_error));
      return false;
    }
    const auto jump_sgprs = moi_indirect_jump_sgprs(group_options);
    const auto special_state = moi_special_state_sgprs(group_options);
    std::vector<uint32_t> host_words = host.displaced_words;
    if (!jump_sgprs || (!use_indirect_jump_scc_save && !special_state) ||
        !append_moi_scc_preserving_indirect_jump(
            host_words, host.body_offset,
            host.host_offset + host.displaced_words.size() * sizeof(uint32_t), jump_sgprs->pc_sgpr,
            use_indirect_jump_scc_save ? jump_sgprs->scc_save_sgpr : special_state->scc_save_sgpr,
            /*capture_scc=*/true, arch) ||
        !finalize_host_words(host, host_words)) {
      errors.emplace_back(std::string(return_error));
      return false;
    }
    append_words_bytes(new_text, host_words);
  }
  return true;
}

template <typename PlannedPatch>
[[nodiscard]] bool
write_moi_appended_anchor(std::vector<uint8_t> &text, const PlannedPatch &planned_patch,
                          uint64_t entry_target, rj_code_arch_t arch, std::string_view probe_name,
                          std::vector<std::string> &errors) {
  const ConSanMoiCandidate &candidate = *planned_patch.candidate;
  const auto forward =
      compute_sopp_branch_simm16(planned_patch.placement.anchor_offset, entry_target);
  if (!planned_patch.entry_island_at_anchor && !forward) {
    errors.emplace_back("ConSan MOI " + std::string(probe_name) +
                        " appended cave entry is out of range");
    return false;
  }
  std::vector<uint32_t> anchor_words(planned_patch.placement.original_size / sizeof(uint32_t),
                                     build_s_nop(0, arch));
  if (anchor_words.empty()) {
    errors.emplace_back("ConSan MOI " + std::string(probe_name) + " has an empty anchor");
    return false;
  }
  if (!planned_patch.entry_island_at_anchor)
    anchor_words.front() = build_s_branch(*forward, arch);
  const uint64_t anchor_bytes = anchor_words.size() * sizeof(uint32_t);
  if (candidate.anchor() > text.size() || anchor_bytes > text.size() - candidate.anchor()) {
    errors.emplace_back("ConSan MOI " + std::string(probe_name) + " anchor exceeds .text");
    return false;
  }
  std::memcpy(text.data() + candidate.anchor(), anchor_words.data(),
              static_cast<size_t>(anchor_bytes));
  return true;
}

template <typename PlannedPatch>
[[nodiscard]] bool
write_moi_entry_island(std::vector<uint8_t> &text, const PlannedPatch &planned_patch,
                       std::span<const uint32_t> island_words,
                       std::vector<ConSanPatchInfo> &patches, std::string_view probe_name,
                       std::vector<std::string> &errors) {
  if (!planned_patch.entry_island_offset)
    return true;
  const uint64_t island_bytes = island_words.size() * sizeof(uint32_t);
  const uint64_t island_offset = *planned_patch.entry_island_offset;
  if (island_offset > text.size() || island_bytes > text.size() - island_offset) {
    errors.emplace_back("ConSan MOI " + std::string(probe_name) +
                        " entry island exceeds reserved text");
    return false;
  }
  std::memcpy(text.data() + island_offset, island_words.data(), static_cast<size_t>(island_bytes));
  if (!planned_patch.entry_island_at_anchor) {
    ConSanPatchInfo island_info;
    island_info.kind = ConSanPatchKind::TrampolineMoiIndirectBranchIsland;
    island_info.anchor_offset = planned_patch.candidate->anchor();
    island_info.trampoline_offset = island_offset;
    island_info.trampoline_size = static_cast<uint32_t>(island_bytes);
    island_info.owner_descriptor_file_offsets =
        planned_patch.resources.owner_descriptor_file_offsets;
    patches.push_back(std::move(island_info));
  }
  return true;
}

/// Complete target-state and composition contract for one relocated MOI
/// access body.
///
/// The access planners decide guest placement, preservation, and target-state
/// requirements before calling `assemble_moi_appended_body`. This value then
/// gives the common assembler everything needed to establish the probe's
/// required register-bank state, preserve scalar/vector scratch, place the
/// relocated guest instruction, and restore the incoming state. It contains no
/// engine evidence policy and owns no storage; every span and output pointer is
/// valid only for the duration of the assembly call.
struct MoiAppendedBodyOptions {
  explicit MoiAppendedBodyOptions(rj_code_arch_t arch) : arch(arch) {}

  /// Target whose native instruction builders assemble this body.
  rj_code_arch_t arch;
  /// Guest words emitted after the assembled body by its caller.
  size_t deferred_guest_word_count = 0u;
  /// Scalar preservation sequences surrounding the probe body.
  std::span<const uint32_t> scalar_save_words = {};
  std::span<const uint32_t> scalar_restore_words = {};
  /// Selectable-VGPR-bank state observed on entry. Absence and zero both need
  /// no transition; a nonzero value is valid only on a target whose profile
  /// advertises selectable banks.
  std::optional<uint16_t> incoming_vgpr_bank_mode = std::nullopt;
  /// Guest address operand to copy before selecting the low instrumentation
  /// bank, and the low-bank scratch register that receives the copy. They are
  /// either both present or both absent.
  std::optional<uint16_t> high_bank_address_source = std::nullopt;
  std::optional<uint16_t> high_bank_address_destination = std::nullopt;
  /// Probe suffix that must remain after vector-spill restoration.
  size_t trailing_guest_word_count = 0u;
  /// Already-planned entry gate or preservation words prepended to the body.
  std::span<const uint32_t> entry_prefix_words = {};
  /// Guest location within `probe_words`, if the probe embeds it.
  std::optional<uint32_t> probe_guest_instruction_offset = std::nullopt;
  std::optional<size_t> guest_instruction_word_count = std::nullopt;
  /// Receives the final byte offset of the relocated guest within the body.
  uint32_t *body_guest_instruction_offset = nullptr;
  /// Whether overlapping guest operands require restoring vector spill state
  /// before the trailing probe suffix rather than at the ordinary exit.
  bool restore_vgpr_spill_before_trailing_guest = false;
  /// Whether the guest instruction itself must run in the incoming selectable
  /// bank while the surrounding instrumentation runs in the low bank.
  bool wrap_embedded_guest_vgpr_bank = false;
};

[[nodiscard]] constexpr bool
moi_appended_body_uses_selectable_vgpr_bank(rj_code_arch_t arch,
                                            std::optional<uint16_t> incoming_vgpr_bank_mode) {
  return consan_arch_has_selectable_vgpr_bank(arch) && incoming_vgpr_bank_mode.value_or(0u) != 0u;
}

[[nodiscard]] constexpr bool
moi_appended_body_vgpr_bank_mode_is_valid(rj_code_arch_t arch,
                                          std::optional<uint16_t> incoming_vgpr_bank_mode) {
  return incoming_vgpr_bank_mode.value_or(0u) == 0u || consan_arch_has_selectable_vgpr_bank(arch);
}

[[nodiscard]] constexpr size_t
moi_appended_body_vgpr_bank_transition_word_count(rj_code_arch_t arch,
                                                  std::optional<uint16_t> incoming_vgpr_bank_mode,
                                                  bool has_vgpr_spill, bool wraps_embedded_guest) {
  if (!moi_appended_body_uses_selectable_vgpr_bank(arch, incoming_vgpr_bank_mode))
    return 0u;
  return (has_vgpr_spill ? 4u : 2u) + (wraps_embedded_guest ? 2u : 0u);
}

[[nodiscard]] constexpr bool
moi_embedded_guest_vgpr_bank_plan_is_valid(bool wraps_embedded_guest, bool selects_low_vgpr_bank,
                                           bool has_guest_instruction,
                                           size_t trailing_guest_word_count) {
  return !wraps_embedded_guest ||
         (selects_low_vgpr_bank && has_guest_instruction && trailing_guest_word_count == 0u);
}

static_assert(moi_appended_body_vgpr_bank_mode_is_valid(ROCJITSU_CODE_ARCH_CDNA5, uint16_t{4u}));
static_assert(!moi_appended_body_vgpr_bank_mode_is_valid(ROCJITSU_CODE_ARCH_CDNA4, uint16_t{4u}));
static_assert(moi_appended_body_vgpr_bank_transition_word_count(ROCJITSU_CODE_ARCH_CDNA5,
                                                                std::nullopt, false, false) == 0u);
static_assert(moi_appended_body_vgpr_bank_transition_word_count(ROCJITSU_CODE_ARCH_CDNA5,
                                                                uint16_t{4u}, false, false) == 2u);
static_assert(moi_appended_body_vgpr_bank_transition_word_count(ROCJITSU_CODE_ARCH_CDNA5,
                                                                uint16_t{4u}, false, true) == 4u);
static_assert(moi_appended_body_vgpr_bank_transition_word_count(ROCJITSU_CODE_ARCH_CDNA5,
                                                                uint16_t{4u}, true, false) == 4u);
static_assert(moi_appended_body_vgpr_bank_transition_word_count(ROCJITSU_CODE_ARCH_CDNA5,
                                                                uint16_t{4u}, true, true) == 6u);
static_assert(moi_appended_body_vgpr_bank_transition_word_count(ROCJITSU_CODE_ARCH_CDNA4,
                                                                uint16_t{4u}, true, true) == 0u);
static_assert(moi_embedded_guest_vgpr_bank_plan_is_valid(false, false, false, 1u));
static_assert(moi_embedded_guest_vgpr_bank_plan_is_valid(true, true, true, 0u));
static_assert(!moi_embedded_guest_vgpr_bank_plan_is_valid(true, false, true, 0u));
static_assert(!moi_embedded_guest_vgpr_bank_plan_is_valid(true, true, false, 0u));
static_assert(!moi_embedded_guest_vgpr_bank_plan_is_valid(true, true, true, 1u));

template <typename PlannedPatch>
[[nodiscard]] std::optional<std::vector<uint32_t>>
assemble_moi_appended_body(const PlannedPatch &planned_patch, std::span<const uint32_t> probe_words,
                           std::string_view probe_name, std::vector<std::string> &errors,
                           const MoiAppendedBodyOptions &options) {
  std::vector<uint32_t> body;
  if (options.trailing_guest_word_count > probe_words.size()) {
    errors.emplace_back("ConSan MOI " + std::string(probe_name) +
                        " VGPR-bank guest split exceeds the probe body");
    return std::nullopt;
  }
  // These checks are defensive backstops for plans assembled by ConSan's
  // internal callers. Public options cannot directly construct either invalid
  // combination, but release builds must not silently emit a corrupt body if a
  // future planner violates the contract.
  if (!moi_appended_body_vgpr_bank_mode_is_valid(options.arch, options.incoming_vgpr_bank_mode)) {
    errors.emplace_back("ConSan MOI " + std::string(probe_name) +
                        " has a selectable-VGPR-bank mode on an unsupported target body");
    return std::nullopt;
  }
  if (options.high_bank_address_source.has_value() !=
      options.high_bank_address_destination.has_value()) {
    errors.emplace_back("ConSan MOI " + std::string(probe_name) +
                        " has an incomplete high-bank address capture");
    return std::nullopt;
  }
  const bool select_low_vgpr_bank =
      moi_appended_body_uses_selectable_vgpr_bank(options.arch, options.incoming_vgpr_bank_mode);
  const bool capture_high_bank_address = options.high_bank_address_source.has_value();
  const uint8_t incoming_vgpr_msb_mode =
      static_cast<uint8_t>(options.incoming_vgpr_bank_mode.value_or(0u));
  const uint8_t address_capture_mode = static_cast<uint8_t>(incoming_vgpr_msb_mode & 0x3u);
  if (capture_high_bank_address &&
      (!select_low_vgpr_bank || address_capture_mode == 0u ||
       *options.high_bank_address_source > 255u || *options.high_bank_address_destination > 255u)) {
    errors.emplace_back("ConSan MOI " + std::string(probe_name) +
                        " has an invalid high-bank address capture");
    return std::nullopt;
  }
  const size_t probe_prefix_words = probe_words.size() - options.trailing_guest_word_count;
  body.insert(body.end(), options.entry_prefix_words.begin(), options.entry_prefix_words.end());
  uint8_t current_vgpr_msb_mode = incoming_vgpr_msb_mode;
  const bool save_spill_before_high_bank_capture = capture_high_bank_address && planned_patch.spill;
  if (save_spill_before_high_bank_capture) {
    body.push_back(
        *instrumentation::build_s_set_vgpr_msb_transition(current_vgpr_msb_mode, 0u, options.arch));
    body.insert(body.end(), planned_patch.spill->save_words.begin(),
                planned_patch.spill->save_words.end());
    current_vgpr_msb_mode = 0u;
  }
  if (capture_high_bank_address) {
    if (current_vgpr_msb_mode != address_capture_mode) {
      body.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
          current_vgpr_msb_mode, address_capture_mode, options.arch));
      current_vgpr_msb_mode = address_capture_mode;
    }
    body.push_back(build_v_mov_b32_e32(*options.high_bank_address_destination,
                                       vector_source_vgpr(*options.high_bank_address_source),
                                       options.arch));
  }
  if (select_low_vgpr_bank && current_vgpr_msb_mode != 0u)
    body.push_back(
        *instrumentation::build_s_set_vgpr_msb_transition(current_vgpr_msb_mode, 0u, options.arch));
  if (planned_patch.spill && !save_spill_before_high_bank_capture)
    body.insert(body.end(), planned_patch.spill->save_words.begin(),
                planned_patch.spill->save_words.end());
  body.insert(body.end(), options.scalar_save_words.begin(), options.scalar_save_words.end());
  const size_t probe_body_begin = body.size();
  std::optional<size_t> guest_word;
  size_t guest_word_count = 0u;
  if (options.probe_guest_instruction_offset) {
    if (*options.probe_guest_instruction_offset % sizeof(uint32_t) != 0u) {
      errors.emplace_back("ConSan MOI " + std::string(probe_name) +
                          " has an unaligned relocated guest offset");
      return std::nullopt;
    }
    guest_word = *options.probe_guest_instruction_offset / sizeof(uint32_t);
    guest_word_count = options.guest_instruction_word_count.value_or(
        planned_patch.candidate->size() / sizeof(uint32_t));
    if (*guest_word >= probe_words.size() || guest_word_count == 0u ||
        guest_word_count > probe_words.size() - *guest_word) {
      errors.emplace_back("ConSan MOI " + std::string(probe_name) +
                          " has an invalid relocated guest range");
      return std::nullopt;
    }
  }
  const bool valid_embedded_guest_plan = moi_embedded_guest_vgpr_bank_plan_is_valid(
      options.wrap_embedded_guest_vgpr_bank, select_low_vgpr_bank, guest_word.has_value(),
      options.trailing_guest_word_count);
  if (!valid_embedded_guest_plan) {
    errors.emplace_back("ConSan MOI " + std::string(probe_name) +
                        " has an invalid embedded-guest VGPR-bank plan");
    return std::nullopt;
  }
  const bool wrap_embedded_guest = options.wrap_embedded_guest_vgpr_bank;
  if (wrap_embedded_guest) {
    body.insert(body.end(), probe_words.begin(), probe_words.begin() + *guest_word);
    body.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
        0u, static_cast<uint8_t>(*options.incoming_vgpr_bank_mode), options.arch));
    if (options.body_guest_instruction_offset)
      *options.body_guest_instruction_offset =
          static_cast<uint32_t>(body.size() * sizeof(uint32_t));
    body.insert(body.end(), probe_words.begin() + *guest_word,
                probe_words.begin() + *guest_word + guest_word_count);
    body.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
        static_cast<uint8_t>(*options.incoming_vgpr_bank_mode), 0u, options.arch));
    body.insert(body.end(), probe_words.begin() + *guest_word + guest_word_count,
                probe_words.begin() + probe_prefix_words);
  } else {
    body.insert(body.end(), probe_words.begin(), probe_words.begin() + probe_prefix_words);
  }
  body.insert(body.end(), options.scalar_restore_words.begin(), options.scalar_restore_words.end());
  if (options.restore_vgpr_spill_before_trailing_guest && planned_patch.spill)
    body.insert(body.end(), planned_patch.spill->restore_words.begin(),
                planned_patch.spill->restore_words.end());
  if (select_low_vgpr_bank) {
    body.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
        0u, static_cast<uint8_t>(*options.incoming_vgpr_bank_mode), options.arch));
  }
  const size_t trailing_probe_body_begin = body.size();
  body.insert(body.end(), probe_words.begin() + probe_prefix_words, probe_words.end());
  if (guest_word && options.body_guest_instruction_offset && !wrap_embedded_guest) {
    const size_t body_word = *guest_word < probe_prefix_words
                                 ? probe_body_begin + *guest_word
                                 : trailing_probe_body_begin + *guest_word - probe_prefix_words;
    *options.body_guest_instruction_offset = static_cast<uint32_t>(body_word * sizeof(uint32_t));
  }
  if (select_low_vgpr_bank && planned_patch.spill)
    body.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
        static_cast<uint8_t>(*options.incoming_vgpr_bank_mode), 0u, options.arch));
  if (planned_patch.spill && !options.restore_vgpr_spill_before_trailing_guest)
    body.insert(body.end(), planned_patch.spill->restore_words.begin(),
                planned_patch.spill->restore_words.end());
  if (select_low_vgpr_bank && planned_patch.spill) {
    body.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
        0u, static_cast<uint8_t>(*options.incoming_vgpr_bank_mode), options.arch));
  }
  body.insert(body.end(), planned_patch.displaced_tail_words.begin(),
              planned_patch.displaced_tail_words.end());
  if ((body.size() + options.deferred_guest_word_count) * sizeof(uint32_t) !=
      planned_patch.placement.body_size) {
    errors.emplace_back(
        "ConSan MOI " + std::string(probe_name) + " body size changed after placement: emitted=" +
        std::to_string((body.size() + options.deferred_guest_word_count) * sizeof(uint32_t)) +
        " planned=" + std::to_string(planned_patch.placement.body_size));
    return std::nullopt;
  }
  return body;
}

[[nodiscard]] bool
moi_scalar_spill_requires_dynamic_vgpr_frame(const ConSanTransformArtifacts &result,
                                             const ResolvedMoiScratchPlan &resources,
                                             const MoiOptions &options) {
  if (!options.automatic_moi_inline_sgpr_spill && !options.automatic_moi_record_replay_sgpr_spill) {
    return false;
  }
  return std::ranges::any_of(resources.owner_descriptor_file_offsets, [&](uint64_t offset) {
    const ConSanKernelInfo *kernel = result.program_inventory.find_kernel_by_descriptor(offset);
    return kernel != nullptr && kernel->uses_dynamic_stack.value_or(false);
  });
}

/// Build the common scalar-preservation transaction selected by resource
/// planning. Dynamic-stack owners borrow the already-established VGPR spill
/// frame; fixed-stack owners use the planned lane-backed reservoir. Engine
/// policy decides whether scalar spilling is enabled, but no engine privately
/// interprets the owner stack model or reservoir geometry.
[[nodiscard]] std::optional<SgprSpillSequence> build_moi_sgpr_spill_sequence(
    const ConSanTransformArtifacts &result, const ResolvedMoiScratchPlan &resources,
    const MoiOptions &options, MoiSpillManagers &managers, rj_code_arch_t arch,
    std::vector<std::string> &warnings, std::optional<uint32_t> private_layout_base,
    const VgprSpillSequence *vgpr_spill = nullptr) {
  if ((!options.automatic_moi_inline_sgpr_spill &&
       !options.automatic_moi_record_replay_sgpr_spill) ||
      !options.moi_exec_save_sgpr)
    return std::nullopt;
  if (!consan_is_capability_arch(arch) || resources.owner_descriptor_file_offsets.empty()) {
    warnings.emplace_back(
        "ConSan MOI spill-backed scalar state requires at least one supported kernel owner");
    return std::nullopt;
  }
  const auto private_limit = consan_address_free_private_limit(arch);
  if (!private_limit) {
    warnings.emplace_back("ConSan MOI spill-backed Inline scalar state has no private capacity");
    return std::nullopt;
  }
  const uint16_t count = moi_exec_save_sgpr_count(options, arch);
  if (moi_scalar_spill_requires_dynamic_vgpr_frame(result, resources, options)) {
    if (vgpr_spill == nullptr || !vgpr_spill->uses_dynamic_stack_frame) {
      warnings.emplace_back(
          "ConSan MOI dynamic-stack scalar spill requires an established VGPR spill frame");
      return std::nullopt;
    }
    auto sequence = build_dynamic_stack_sgpr_spill_sequence(
        *options.moi_exec_save_sgpr, count, resources.base, *vgpr_spill,
        static_cast<uint32_t>(resources.count) * SpillManager::kSlotBytes, arch);
    if (!sequence)
      warnings.emplace_back("ConSan MOI could not encode the dynamic-stack scalar spill sequence");
    return sequence;
  }
  (void)managers;
  (void)private_layout_base;
  const uint32_t reservoir_end = static_cast<uint32_t>(resources.base) + resources.count + 1u;
  if (reservoir_end > REGISTER_SET_MAX_VGPRS || resources.required_vgpr_count < reservoir_end ||
      (vgpr_spill && (vgpr_spill->vgpr_base != resources.base ||
                      vgpr_spill->vgpr_count != resources.count + 1u))) {
    warnings.emplace_back(
        "ConSan MOI fixed-stack scalar spill has no planned lane-backed VGPR reservoir");
    return std::nullopt;
  }
  const uint16_t reservoir_vgpr = static_cast<uint16_t>(reservoir_end - 1u);
  const uint32_t total_private_bytes =
      vgpr_spill ? vgpr_spill->total_private_bytes : resources.original_private_segment_size;
  auto sequence = build_lane_sgpr_spill_sequence(*options.moi_exec_save_sgpr, count, reservoir_vgpr,
                                                 total_private_bytes, arch);
  if (!sequence) {
    warnings.emplace_back("ConSan MOI could not encode the lane-backed scalar spill sequence");
    return std::nullopt;
  }
  return sequence;
}
