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
