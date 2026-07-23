template <typename PlannedPatch, typename BuildWords, typename MakePatchInfo,
          typename ApplyExtraRequirements>
[[nodiscard]] bool apply_inline_moi_access_patches(
    std::span<const uint8_t> bytes, const std::vector<PlannedPatch> &planned_patches,
    const MoiDescriptorVgprRequirements &descriptor_requirements,
    const MoiDescriptorSgprRequirements &scalar_requirements, std::string_view probe_name,
    rj_code_arch_t arch, BuildWords build_words, MakePatchInfo make_patch_info,
    ApplyExtraRequirements apply_extra_requirements, ConSanResult &result) {
  result.elf_bytes.assign(bytes.begin(), bytes.end());
  for (const PlannedPatch &planned_patch : planned_patches) {
    const ConSanMoiCandidate &candidate = *planned_patch.candidate;
    auto words = build_words(planned_patch);
    if (!words) {
      result.elf_bytes.clear();
      return false;
    }
    const uint64_t patch_bytes = static_cast<uint64_t>(words->size() * sizeof(uint32_t));
    if (patch_bytes != planned_patch.placement.body_size) {
      result.errors.emplace_back("ConSan MOI " + std::string(probe_name) +
                                 " final patch size changed");
      result.elf_bytes.clear();
      return false;
    }
    if (candidate.file_offset > result.elf_bytes.size() ||
        patch_bytes > result.elf_bytes.size() - candidate.file_offset) {
      result.errors.emplace_back("ConSan MOI " + std::string(probe_name) +
                                 " inline patch exceeds the code object");
      result.elf_bytes.clear();
      return false;
    }
    std::memcpy(result.elf_bytes.data() + candidate.file_offset, words->data(),
                static_cast<size_t>(patch_bytes));
  }

  if (!apply_descriptor_requirements(result.elf_bytes, result, descriptor_requirements, arch,
                                     result.errors) ||
      !apply_sgpr_descriptor_requirements(result.elf_bytes, result, scalar_requirements,
                                          result.errors) ||
      !apply_extra_requirements(result.elf_bytes, result)) {
    result.elf_bytes.clear();
    return false;
  }

  for (const PlannedPatch &planned_patch : planned_patches)
    result.patches.push_back(make_patch_info(planned_patch));
  result.modified = true;
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
  if (candidate.text_offset > text.size() || anchor_bytes > text.size() - candidate.text_offset) {
    errors.emplace_back("ConSan MOI " + std::string(probe_name) + " anchor exceeds .text");
    return false;
  }
  std::memcpy(text.data() + candidate.text_offset, anchor_words.data(),
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
    island_info.anchor_offset = planned_patch.candidate->text_offset;
    island_info.trampoline_offset = island_offset;
    island_info.trampoline_size = static_cast<uint32_t>(island_bytes);
    island_info.owner_descriptor_file_offsets =
        planned_patch.resources.owner_descriptor_file_offsets;
    patches.push_back(std::move(island_info));
  }
  return true;
}

template <typename PlannedPatch>
[[nodiscard]] std::optional<std::vector<uint32_t>> assemble_moi_appended_body(
    const PlannedPatch &planned_patch, std::span<const uint32_t> probe_words,
    std::string_view probe_name, std::vector<std::string> &errors,
    size_t deferred_guest_word_count = 0u, std::span<const uint32_t> scalar_save_words = {},
    std::span<const uint32_t> scalar_restore_words = {},
    std::optional<uint16_t> gfx1250_vgpr_msb_mode = std::nullopt,
    size_t trailing_guest_word_count = 0u, std::span<const uint32_t> entry_prefix_words = {},
    std::optional<uint32_t> probe_guest_instruction_offset = std::nullopt,
    uint32_t *body_guest_instruction_offset = nullptr) {
  std::vector<uint32_t> body;
  if (trailing_guest_word_count > probe_words.size()) {
    errors.emplace_back("ConSan MOI " + std::string(probe_name) +
                        " VGPR-bank guest split exceeds the probe body");
    return std::nullopt;
  }
  const bool select_low_vgpr_bank = gfx1250_vgpr_msb_mode.value_or(0u) != 0u;
  const size_t probe_prefix_words = probe_words.size() - trailing_guest_word_count;
  body.insert(body.end(), entry_prefix_words.begin(), entry_prefix_words.end());
  if (select_low_vgpr_bank)
    body.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
        static_cast<uint8_t>(*gfx1250_vgpr_msb_mode), 0u, ROCJITSU_CODE_ARCH_GFX1250));
  if (planned_patch.spill)
    body.insert(body.end(), planned_patch.spill->save_words.begin(),
                planned_patch.spill->save_words.end());
  body.insert(body.end(), scalar_save_words.begin(), scalar_save_words.end());
  const size_t probe_body_begin = body.size();
  body.insert(body.end(), probe_words.begin(), probe_words.begin() + probe_prefix_words);
  body.insert(body.end(), scalar_restore_words.begin(), scalar_restore_words.end());
  if (select_low_vgpr_bank) {
    body.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
        0u, static_cast<uint8_t>(*gfx1250_vgpr_msb_mode), ROCJITSU_CODE_ARCH_GFX1250));
  }
  const size_t trailing_probe_body_begin = body.size();
  body.insert(body.end(), probe_words.begin() + probe_prefix_words, probe_words.end());
  if (probe_guest_instruction_offset && body_guest_instruction_offset) {
    const size_t guest_word = *probe_guest_instruction_offset / sizeof(uint32_t);
    if (*probe_guest_instruction_offset % sizeof(uint32_t) != 0u ||
        guest_word >= probe_words.size()) {
      errors.emplace_back("ConSan MOI " + std::string(probe_name) +
                          " has an invalid relocated guest offset");
      return std::nullopt;
    }
    const size_t body_word = guest_word < probe_prefix_words
                                 ? probe_body_begin + guest_word
                                 : trailing_probe_body_begin + guest_word - probe_prefix_words;
    *body_guest_instruction_offset = static_cast<uint32_t>(body_word * sizeof(uint32_t));
  }
  if (select_low_vgpr_bank && planned_patch.spill)
    body.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
        static_cast<uint8_t>(*gfx1250_vgpr_msb_mode), 0u, ROCJITSU_CODE_ARCH_GFX1250));
  if (planned_patch.spill)
    body.insert(body.end(), planned_patch.spill->restore_words.begin(),
                planned_patch.spill->restore_words.end());
  if (select_low_vgpr_bank && planned_patch.spill) {
    body.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
        0u, static_cast<uint8_t>(*gfx1250_vgpr_msb_mode), ROCJITSU_CODE_ARCH_GFX1250));
  }
  body.insert(body.end(), planned_patch.displaced_tail_words.begin(),
              planned_patch.displaced_tail_words.end());
  if ((body.size() + deferred_guest_word_count) * sizeof(uint32_t) !=
      planned_patch.placement.body_size) {
    errors.emplace_back(
        "ConSan MOI " + std::string(probe_name) + " body size changed after placement: emitted=" +
        std::to_string((body.size() + deferred_guest_word_count) * sizeof(uint32_t)) +
        " planned=" + std::to_string(planned_patch.placement.body_size));
    return std::nullopt;
  }
  return body;
}
