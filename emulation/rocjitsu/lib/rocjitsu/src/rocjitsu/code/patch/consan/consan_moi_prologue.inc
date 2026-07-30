[[nodiscard]] bool append_moi_entry_salu_write(std::vector<uint32_t> &words, uint32_t word,
                                               rj_code_arch_t arch) {
  const auto delay = instrumentation::build_salu_dependency_delay(arch);
  if (!delay)
    return false;
  words.push_back(word);
  words.push_back(*delay);
  return true;
}

struct ConSanMoiDispatchIdCapture {
  std::optional<uint16_t> sgpr;
  std::optional<uint16_t> vgpr;

  [[nodiscard]] bool present() const { return sgpr || vgpr; }
  [[nodiscard]] bool unambiguous() const {
    return static_cast<bool>(sgpr) != static_cast<bool>(vgpr);
  }
};

struct MoiEntryScalarBackup {
  uint16_t vgpr = 0;
  uint16_t sgpr_base = 0;
  uint16_t sgpr_count = 0;
};

[[nodiscard]] bool append_moi_entry_scalar_backup(std::vector<uint32_t> &words,
                                                  const MoiEntryScalarBackup &backup, bool restore,
                                                  rj_code_arch_t arch,
                                                  std::vector<std::string> &errors) {
  if (!instrumentation::is_admitted_arch(arch) || backup.vgpr >= kMaxVgprs ||
      backup.sgpr_count == 0u || backup.sgpr_count > 64u ||
      static_cast<uint32_t>(backup.sgpr_base) + backup.sgpr_count > moi_ordinary_sgpr_limit(arch)) {
    errors.emplace_back("ConSan MOI entry scalar backup has an invalid register layout");
    return false;
  }
  for (uint16_t index = 0; index < backup.sgpr_count; ++index) {
    const uint16_t sgpr = static_cast<uint16_t>(backup.sgpr_base + index);
    const auto encoded =
        restore ? instrumentation::build_v_readlane_b32(sgpr, backup.vgpr, index, arch)
                : instrumentation::build_v_writelane_b32(backup.vgpr, sgpr, index, arch);
    if (!encoded) {
      errors.emplace_back("ConSan MOI entry scalar backup could not encode a lane transfer");
      return false;
    }
    words.insert(words.end(), encoded->begin(), encoded->end());
  }
  if (restore) {
    const auto wait = instrumentation::build_valu_to_salu_dependency_wait(arch);
    if (!wait) {
      errors.emplace_back(
          "ConSan MOI entry scalar backup could not encode its restore dependency wait");
      return false;
    }
    words.push_back(*wait);
  }
  return true;
}

[[nodiscard]] bool append_dispatch_id_capture_and_restore(
    std::vector<uint32_t> &words, const ConSanMoiDispatchIdPreloadPlan &plan,
    ConSanMoiDispatchIdCapture capture, bool restore_system_sgprs, rj_code_arch_t arch,
    std::vector<std::string> &errors) {
  if (!plan.supported()) {
    errors.emplace_back("ConSan MOI dispatch-ID prologue received an unsupported preload plan");
    return false;
  }
  if (!capture.unambiguous() ||
      (capture.sgpr && (*capture.sgpr < 20u || *capture.sgpr % 2u != 0u ||
                        static_cast<uint32_t>(*capture.sgpr) + 2u > kMaxSgprs)) ||
      (capture.vgpr && static_cast<uint32_t>(*capture.vgpr) + 2u > kMaxVgprs) ||
      static_cast<uint32_t>(plan.dispatch_id_sgpr) + 2u > kMaxSgprs) {
    errors.emplace_back("ConSan MOI dispatch-ID prologue has an invalid persistent or source pair");
    return false;
  }

  if (capture.sgpr) {
    if (!append_moi_entry_salu_write(
            words, build_s_mov_b32(*capture.sgpr, plan.dispatch_id_sgpr, arch), arch) ||
        !append_moi_entry_salu_write(
            words,
            build_s_mov_b32(static_cast<uint16_t>(*capture.sgpr + 1u),
                            static_cast<uint16_t>(plan.dispatch_id_sgpr + 1u), arch),
            arch) ||
        // Hardware dispatch ID zero is valid (and common for the first packet),
        // while ConSan reserves zero as the empty metadata sentinel. Store the
        // injective modulo-2^64 successor used by every downstream comparison.
        !append_moi_entry_salu_write(
            words,
            build_s_add_u32(*capture.sgpr, *capture.sgpr, scalar_positive_inline_u32(1), arch),
            arch) ||
        !append_moi_entry_salu_write(words,
                                     build_s_addc_u32(static_cast<uint16_t>(*capture.sgpr + 1u),
                                                      static_cast<uint16_t>(*capture.sgpr + 1u),
                                                      scalar_positive_inline_u32(0), arch),
                                     arch)) {
      errors.emplace_back(
          "ConSan MOI dispatch-ID prologue cannot encode its SALU dependency delay");
      return false;
    }
  } else {
    words.push_back(build_v_mov_b32_e32(*capture.vgpr, plan.dispatch_id_sgpr, arch));
    words.push_back(build_v_mov_b32_e32(static_cast<uint16_t>(*capture.vgpr + 1u),
                                        static_cast<uint16_t>(plan.dispatch_id_sgpr + 1u), arch));
    const auto encode_nonzero = instrumentation::build_v_add_u64_literal(*capture.vgpr, 1u, arch);
    if (!encode_nonzero) {
      errors.emplace_back(
          "ConSan MOI dispatch-ID prologue cannot encode its persistent VGPR successor");
      return false;
    }
    words.insert(words.end(), encode_nonzero->begin(), encode_nonzero->end());
  }
  for (uint16_t destination = plan.first_shifted_guest_sgpr;
       destination < plan.first_shifted_guest_sgpr + plan.shifted_guest_sgpr_count; ++destination) {
    const auto source = consan_moi_dispatch_id_restore_source(plan, destination);
    if (!source || *source >= kMaxSgprs) {
      errors.emplace_back("ConSan MOI dispatch-ID prologue has an invalid guest restore source");
      return false;
    }
    if (!append_moi_entry_salu_write(words, build_s_mov_b32(destination, *source, arch), arch)) {
      errors.emplace_back(
          "ConSan MOI dispatch-ID prologue cannot encode a guest-restore dependency delay");
      return false;
    }
  }
  if (restore_system_sgprs) {
    for (uint16_t i = 0; i < plan.shifted_system_sgpr_count; ++i) {
      const uint16_t destination = static_cast<uint16_t>(plan.original_user_sgpr_count + i);
      if (!append_moi_entry_salu_write(
              words,
              build_s_mov_b32(destination,
                              static_cast<uint16_t>(destination + plan.system_sgpr_shift), arch),
              arch)) {
        errors.emplace_back("ConSan MOI dispatch-ID prologue cannot encode a system-SGPR restore");
        return false;
      }
    }
  }
  if (plan.requires_kernarg_reload()) {
    if ((arch != ROCJITSU_CODE_ARCH_CDNA3 && arch != ROCJITSU_CODE_ARCH_CDNA4) ||
        plan.kernarg_reload_count > 2u) {
      errors.emplace_back("ConSan MOI dispatch-ID prologue has an unsupported kernarg reload");
      return false;
    }
    for (uint16_t i = 0; i < plan.kernarg_reload_count; ++i) {
      const auto load = instrumentation::build_s_load_dword(
          static_cast<uint16_t>(plan.kernarg_reload_sgpr + i), plan.kernarg_reload_base_sgpr,
          static_cast<uint32_t>(plan.kernarg_reload_offset_dwords + i) * sizeof(uint32_t), arch);
      if (!load) {
        errors.emplace_back("ConSan MOI dispatch-ID prologue cannot encode a kernarg reload");
        return false;
      }
      words.insert(words.end(), load->begin(), load->end());
    }
    const auto wait = instrumentation::build_s_wait_scalar_load0(arch);
    if (!wait) {
      errors.emplace_back("ConSan MOI dispatch-ID prologue cannot encode its kernarg wait");
      return false;
    }
    words.push_back(*wait);
  }
  return true;
}

[[nodiscard]] bool
append_cdna_full_workgroup_payload_restore(std::vector<uint32_t> &words,
                                           const ConSanMoiWorkgroupSources &sources,
                                           rj_code_arch_t arch, std::vector<std::string> &errors) {
  if (!sources.cdna_full_payload_base && !sources.cdna_guest_payload_base &&
      sources.cdna_guest_payload_mask == 0u) {
    return true;
  }
  if (!sources.cdna_full_payload_base || !sources.cdna_guest_payload_base ||
      (arch != ROCJITSU_CODE_ARCH_CDNA3 && arch != ROCJITSU_CODE_ARCH_CDNA4)) {
    errors.emplace_back("ConSan MOI found an incomplete CDNA workgroup-ID payload repair");
    return false;
  }

  uint16_t destination = *sources.cdna_guest_payload_base;
  for (uint16_t dimension = 0; dimension < 3u; ++dimension) {
    if ((sources.cdna_guest_payload_mask & (1u << dimension)) == 0u)
      continue;
    const uint16_t source = static_cast<uint16_t>(*sources.cdna_full_payload_base + dimension);
    if (destination != source &&
        !append_moi_entry_salu_write(words, build_s_mov_b32(destination, source, arch), arch)) {
      errors.emplace_back("ConSan MOI could not restore a guest CDNA workgroup-ID SGPR");
      return false;
    }
    ++destination;
  }
  if ((sources.cdna_guest_payload_mask & (1u << 3u)) != 0u) {
    const uint16_t source = static_cast<uint16_t>(*sources.cdna_full_payload_base + 3u);
    if (destination != source &&
        !append_moi_entry_salu_write(words, build_s_mov_b32(destination, source, arch), arch)) {
      errors.emplace_back("ConSan MOI could not restore the guest CDNA workgroup-info SGPR");
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool append_moi_prologue_return(std::vector<uint32_t> &words,
                                              uint64_t prologue_text_offset,
                                              uint64_t original_entry_text_offset,
                                              std::optional<uint16_t> pc_sgpr, rj_code_arch_t arch,
                                              std::string_view prologue_name,
                                              std::vector<std::string> &errors) {
  const uint64_t branch_pc =
      prologue_text_offset + static_cast<uint64_t>(words.size()) * sizeof(uint32_t);
  if (const auto branch = compute_sopp_branch_simm16(branch_pc, original_entry_text_offset)) {
    words.push_back(build_s_branch(*branch, arch));
    return true;
  }

  // Large all-site transformations can place the entry prologue beyond a
  // SOPP branch's signed 16-bit dword range. A probe-local EXEC-save window is
  // dead at the selected probe sites, but that does not make it safe at kernel
  // entry: it may still hold guest ABI input. RDNA-family VCC and SCC are not
  // kernel-entry inputs, so use VCC for the absolute return and let the guest
  // establish both special registers from its normal entry state afterward.
  const std::optional<uint16_t> long_return_pc_sgpr =
      is_rdna4_family_arch(arch) ? std::optional<uint16_t>(kRdna4VccLo) : pc_sgpr;
  if (!long_return_pc_sgpr || *long_return_pc_sgpr > 126u || *long_return_pc_sgpr % 2u != 0u) {
    errors.emplace_back("ConSan MOI " + std::string(prologue_name) +
                        " branch target is out of range and has no valid long-return SGPR pair");
    return false;
  }
  words.push_back(build_s_getpc_b64(*long_return_pc_sgpr, arch));
  const uint64_t pc_after_getpc = branch_pc + sizeof(uint32_t);
  if (original_entry_text_offset > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
      pc_after_getpc > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    errors.emplace_back("ConSan MOI " + std::string(prologue_name) +
                        " long-return address exceeds the signed builder range");
    return false;
  }
  const int64_t delta =
      static_cast<int64_t>(original_entry_text_offset) - static_cast<int64_t>(pc_after_getpc);
  if (!append_pc_delta_builder(words, arch, *long_return_pc_sgpr, delta)) {
    errors.emplace_back("ConSan MOI " + std::string(prologue_name) +
                        " could not encode its long return");
    return false;
  }
  const auto wait_pc = instrumentation::build_s_wait_indirect_pc0(arch);
  if (!wait_pc) {
    errors.emplace_back("ConSan MOI " + std::string(prologue_name) +
                        " could not encode its long-return scalar dependency wait");
    return false;
  }
  words.push_back(*wait_pc);
  words.push_back(build_s_setpc_b64(*long_return_pc_sgpr, arch));
  return true;
}

[[nodiscard]] bool append_moi_scc_preserving_indirect_jump(
    std::vector<uint32_t> &words, uint64_t words_text_offset, uint64_t target_text_offset,
    uint16_t pc_sgpr, uint16_t saved_scc_sgpr, bool capture_scc, rj_code_arch_t arch) {
  if (capture_scc) {
    const auto save_scc = instrumentation::build_s_cselect_b32(
        saved_scc_sgpr, scalar_positive_inline_u32(1), scalar_positive_inline_u32(0), arch);
    if (!save_scc)
      return false;
    words.push_back(*save_scc);
  }
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

[[nodiscard]] bool append_moi_direct_or_indirect_return(std::vector<uint32_t> &words,
                                                        uint64_t cave_text_offset,
                                                        uint64_t return_text_offset,
                                                        const ConSanOptions &options,
                                                        rj_code_arch_t arch) {
  const uint64_t branch_pc =
      cave_text_offset + static_cast<uint64_t>(words.size()) * sizeof(uint32_t);
  if (const auto direct = compute_sopp_branch_simm16(branch_pc, return_text_offset)) {
    words.push_back(build_s_branch(*direct, arch));
    return true;
  }
  const auto jump_sgprs = moi_indirect_jump_sgprs(options);
  if (!jump_sgprs || !append_moi_scc_preserving_indirect_jump(
                         words, cave_text_offset, return_text_offset, jump_sgprs->pc_sgpr,
                         jump_sgprs->scc_save_sgpr, /*capture_scc=*/false, arch)) {
    return false;
  }
  const auto wait_pc = instrumentation::build_s_wait_indirect_pc0(arch);
  if (!wait_pc)
    return false;
  words.insert(words.end() - 1, *wait_pc);
  return true;
}

[[nodiscard]] bool
emit_moi_local_indirect_entry_island(std::vector<uint8_t> &text, uint64_t island_text_offset,
                                     uint64_t cave_text_offset, uint64_t anchor_text_offset,
                                     const ConSanOptions &options,
                                     std::span<const uint64_t> owner_descriptor_file_offsets,
                                     rj_code_arch_t arch, std::vector<ConSanPatchInfo> &patches,
                                     std::vector<std::string> &errors, std::string_view context) {
  const auto jump_sgprs = moi_indirect_jump_sgprs(options);
  std::vector<uint32_t> words;
  if (!jump_sgprs || !append_moi_scc_preserving_indirect_jump(
                         words, island_text_offset, cave_text_offset, jump_sgprs->pc_sgpr,
                         jump_sgprs->scc_save_sgpr, /*capture_scc=*/true, arch)) {
    errors.emplace_back(std::string(context) + " could not encode its indirect entry island");
    return false;
  }
  const auto dependency_delay = instrumentation::build_salu_dependency_delay(arch);
  if (!dependency_delay)
    return false;
  words.insert(words.end() - 1, *dependency_delay);
  if (words.size() != kMoiLocalIndirectIslandWords || island_text_offset > text.size() ||
      words.size() * sizeof(uint32_t) > text.size() - island_text_offset) {
    errors.emplace_back(std::string(context) + " has an invalid indirect entry island");
    return false;
  }
  std::memcpy(text.data() + island_text_offset, words.data(), words.size() * sizeof(uint32_t));

  ConSanPatchInfo island_info;
  island_info.kind = ConSanPatchKind::TrampolineMoiIndirectBranchIsland;
  island_info.anchor_offset = anchor_text_offset;
  island_info.trampoline_offset = island_text_offset;
  island_info.trampoline_size = static_cast<uint32_t>(words.size() * sizeof(uint32_t));
  island_info.owner_descriptor_file_offsets.assign(owner_descriptor_file_offsets.begin(),
                                                   owner_descriptor_file_offsets.end());
  patches.push_back(std::move(island_info));
  return true;
}

[[nodiscard]] bool emit_moi_local_indirect_kernel_entry_island(
    std::vector<uint8_t> &text, uint64_t island_text_offset, uint64_t cave_text_offset,
    uint64_t anchor_text_offset, std::span<const uint64_t> owner_descriptor_file_offsets,
    rj_code_arch_t arch, std::vector<ConSanPatchInfo> &patches, std::vector<std::string> &errors,
    std::string_view context) {
  // Ordinary SGPRs at a hardware entry carry ABI inputs even when later
  // liveness selects them as transient instrumentation state. VCC and SCC
  // have no incoming kernel-argument value, so use VCC as the PC pair and
  // leave every ordinary input untouched.
  std::vector<uint32_t> words;
  const uint64_t getpc_text_offset = island_text_offset;
  words.push_back(build_s_getpc_b64(kRdna4VccLo, arch));
  const uint64_t pc_after_getpc = getpc_text_offset + sizeof(uint32_t);
  if (cave_text_offset > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
      pc_after_getpc > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
      !append_pc_delta_builder(words, arch, kRdna4VccLo,
                               static_cast<int64_t>(cave_text_offset) -
                                   static_cast<int64_t>(pc_after_getpc))) {
    errors.emplace_back(std::string(context) + " could not encode its indirect entry island");
    return false;
  }
  const auto dependency_delay = instrumentation::build_salu_dependency_delay(arch);
  if (!dependency_delay)
    return false;
  words.push_back(*dependency_delay);
  words.push_back(build_s_nop(0, arch));
  words.push_back(build_s_setpc_b64(kRdna4VccLo, arch));
  if (island_text_offset > text.size() || words.size() != 7u ||
      words.size() * sizeof(uint32_t) > text.size() - island_text_offset) {
    errors.emplace_back(std::string(context) + " has an invalid indirect entry island");
    return false;
  }
  std::memcpy(text.data() + island_text_offset, words.data(), words.size() * sizeof(uint32_t));

  ConSanPatchInfo island_info;
  island_info.kind = ConSanPatchKind::TrampolineMoiIndirectBranchIsland;
  island_info.anchor_offset = anchor_text_offset;
  island_info.trampoline_offset = island_text_offset;
  island_info.trampoline_size = static_cast<uint32_t>(words.size() * sizeof(uint32_t));
  island_info.owner_descriptor_file_offsets.assign(owner_descriptor_file_offsets.begin(),
                                                   owner_descriptor_file_offsets.end());
  patches.push_back(std::move(island_info));
  return true;
}

[[nodiscard]] uint16_t moi_workgroup_shadow_initialization_lanes(
    const std::optional<std::array<uint32_t, 3>> &required_workgroup_size, rj_code_arch_t arch) {
  constexpr uint16_t kWave32Lanes = 32u;
  constexpr uint16_t kMaximumInlineLanes = 64u;
  if (arch != ROCJITSU_CODE_ARCH_GFX1250 || !required_workgroup_size ||
      (*required_workgroup_size)[0] <= kWave32Lanes) {
    return kWave32Lanes;
  }
  return static_cast<uint16_t>(
      std::min<uint32_t>((*required_workgroup_size)[0], kMaximumInlineLanes));
}

[[nodiscard]] bool append_moi_entry_workitem_coordinate(std::vector<uint32_t> &words,
                                                        uint16_t destination_vgpr,
                                                        uint8_t dimension, rj_code_arch_t arch) {
  constexpr uint32_t kPackedCoordinateMask = 0x3ffu;
  constexpr uint16_t kPackedCoordinateBits = 10u;
  if (dimension > 2u)
    return false;

  if (dimension != 0u) {
    const auto shift = instrumentation::build_v_lshrrev_b32(
        destination_vgpr,
        scalar_positive_inline_u32(static_cast<uint16_t>(dimension * kPackedCoordinateBits)),
        kRdna4WorkitemIdX, arch);
    if (!shift)
      return false;
    words.push_back(*shift);
  }
  const auto mask = instrumentation::build_v_and_b32_literal(
      destination_vgpr, kPackedCoordinateMask,
      dimension == 0u ? kRdna4WorkitemIdX : destination_vgpr, arch);
  if (!mask)
    return false;
  words.insert(words.end(), mask->begin(), mask->end());
  return true;
}

[[nodiscard]] bool append_moi_parallel_workgroup_shadow_initialization(
    std::vector<uint32_t> &words, const ConSanMoiWorkgroupShadowLayout &layout,
    uint16_t address_vgpr, uint16_t zero_vgpr, bool has_quad_zero_tuple,
    std::optional<uint16_t> state_sgpr, rj_code_arch_t arch, std::vector<std::string> &errors) {
  const uint32_t initialization_base =
      layout.lazy_initialization ? layout.validity_base : layout.base;
  const uint32_t initialization_size =
      layout.lazy_initialization ? layout.validity_size : layout.size;
  const bool use_quad_store =
      arch == ROCJITSU_CODE_ARCH_GFX1250 && has_quad_zero_tuple && initialization_size % 16u == 0u;
  const uint16_t zero_tuple_size = use_quad_store ? 4u : 2u;
  if (!state_sgpr || *state_sgpr > 118u || initialization_size == 0u ||
      initialization_size % 8u != 0u || layout.workitem_id_dimensions == 0u ||
      layout.workitem_id_dimensions > 3u || layout.initialization_lanes == 0u ||
      layout.initialization_lanes > 64u ||
      static_cast<uint32_t>(zero_vgpr) + zero_tuple_size > kMaxVgprs ||
      (address_vgpr >= zero_vgpr && address_vgpr < zero_vgpr + zero_tuple_size)) {
    errors.emplace_back(
        "ConSan MOI parallel workgroup-shadow initializer requires aligned LDS and vector state");
    return false;
  }
  const uint16_t exec_save = *state_sgpr;
  const uint16_t vcc_save = static_cast<uint16_t>(*state_sgpr + 2u);
  const uint16_t stride = static_cast<uint16_t>(*state_sgpr + 4u);
  const uint16_t saved_scc = static_cast<uint16_t>(*state_sgpr + 5u);
  const uint16_t selected_exec = static_cast<uint16_t>(*state_sgpr + 6u);
  const uint16_t end_vgpr = static_cast<uint16_t>(zero_vgpr + 1u);

  const auto save_scc = instrumentation::build_s_cselect_b32(
      saved_scc, scalar_positive_inline_u32(1), scalar_positive_inline_u32(0), arch);
  const auto save_vcc = instrumentation::build_s_mov_b64(vcc_save, kRdna4VccLo, arch);
  const auto save_exec = instrumentation::build_s_mov_b64(exec_save, kRdna4ExecLo, arch);
  std::vector<uint32_t> extract_initialization_x;
  const bool valid_initialization_x =
      append_moi_entry_workitem_coordinate(extract_initialization_x, end_vgpr,
                                           /*dimension=*/0u, arch);
  const auto select_initialization_x_lanes = instrumentation::build_v_cmp_gt_u32_vcc(
      scalar_positive_inline_u32(layout.initialization_lanes), end_vgpr, arch);
  const auto narrow_initialization_x_lanes =
      instrumentation::build_s_and_saveexec_b64(selected_exec, kRdna4VccLo, arch);
  std::array<std::vector<uint32_t>, 2> extract_outer_coordinate;
  std::array<std::optional<uint32_t>, 2> select_zero_outer_coordinate;
  std::array<std::optional<uint32_t>, 2> narrow_zero_outer_coordinate;
  bool valid_outer_extraction = true;
  for (uint8_t dimension = 1u; dimension < layout.workitem_id_dimensions; ++dimension) {
    valid_outer_extraction &= append_moi_entry_workitem_coordinate(
        extract_outer_coordinate[dimension - 1u], address_vgpr, dimension, arch);
    select_zero_outer_coordinate[dimension - 1u] =
        instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0u), address_vgpr, arch);
    narrow_zero_outer_coordinate[dimension - 1u] =
        instrumentation::build_s_and_saveexec_b64(selected_exec, kRdna4VccLo, arch);
  }
  const auto skip_empty_wave = instrumentation::build_s_cbranch_execz(0, arch);
  const auto save_selected_exec =
      instrumentation::build_s_mov_b64(selected_exec, kRdna4ExecLo, arch);
  const auto count_selected = instrumentation::build_s_bcnt1_i32_b64(stride, kRdna4ExecLo, arch);
  const uint16_t store_shift = use_quad_store ? 4u : 3u;
  const uint32_t scale_stride =
      build_s_lshl_b32(stride, stride, scalar_positive_inline_u32(store_shift), arch);
  const auto scale_x = instrumentation::build_v_lshlrev_b32(
      end_vgpr, scalar_positive_inline_u32(store_shift), end_vgpr, arch);
  const auto address =
      instrumentation::build_v_add_u32_literal(address_vgpr, initialization_base, end_vgpr, arch);
  const bool use_wide_store = arch != ROCJITSU_CODE_ARCH_CDNA4;
  const auto end = instrumentation::build_v_mov_b32_literal(
      end_vgpr, initialization_base + initialization_size, arch);
  const auto store_low = instrumentation::build_ds_store_b32(address_vgpr, zero_vgpr, 0u, arch);
  const auto store_high = instrumentation::build_ds_store_b32(address_vgpr, zero_vgpr, 4u, arch);
  const auto store_wide = instrumentation::build_ds_store_b64(address_vgpr, zero_vgpr, 0u, arch);
  const auto store_quad = instrumentation::build_ds_store_b128(address_vgpr, zero_vgpr, 0u, arch);
  const auto advance = instrumentation::build_v_add_u32(address_vgpr, stride, address_vgpr, arch);
  const auto more =
      instrumentation::build_v_cmp_gt_u32_vcc(vector_source_vgpr(end_vgpr), address_vgpr, arch);
  const auto more_literal = instrumentation::build_v_cmp_gt_u32_literal_vcc(
      initialization_base + initialization_size, address_vgpr, arch);
  const auto retain_in_bounds =
      instrumentation::build_s_and_b64(kRdna4ExecLo, selected_exec, kRdna4VccLo, arch);
  const auto loop = instrumentation::build_s_cbranch_execnz(0, arch);
  const auto wait_ds = instrumentation::build_s_wait_flat_load_lds0(arch);
  const auto restore_exec = instrumentation::build_s_mov_b64(kRdna4ExecLo, exec_save, arch);
  const auto restore_vcc = instrumentation::build_s_mov_b64(kRdna4VccLo, vcc_save, arch);
  const auto barrier = instrumentation::build_workgroup_barrier_only(arch);
  const auto restore_scc =
      instrumentation::build_s_cmp_lg_u32(saved_scc, scalar_positive_inline_u32(0), arch);
  bool valid_outer_selection = true;
  for (uint8_t dimension = 1u; dimension < layout.workitem_id_dimensions; ++dimension) {
    valid_outer_selection &= select_zero_outer_coordinate[dimension - 1u].has_value() &&
                             narrow_zero_outer_coordinate[dimension - 1u].has_value();
  }
  if (!save_scc || !save_vcc || !save_exec || !valid_initialization_x ||
      !select_initialization_x_lanes || !narrow_initialization_x_lanes || !valid_outer_extraction ||
      !valid_outer_selection || !skip_empty_wave || !save_selected_exec || !count_selected ||
      !scale_x || !address ||
      (use_wide_store ? (!(use_quad_store ? store_quad : store_wide) || !more_literal)
                      : (!end || !store_low || !store_high || !more)) ||
      !advance || !retain_in_bounds || !loop || !wait_ds || !restore_exec || !restore_vcc ||
      !barrier || !restore_scc) {
    const std::array<std::pair<bool, std::string_view>, 22> encodings = {{
        {save_scc.has_value(), "save-scc"},
        {save_vcc.has_value(), "save-vcc"},
        {save_exec.has_value(), "save-exec"},
        {valid_initialization_x, "extract-initialization-x"},
        {select_initialization_x_lanes.has_value(), "select-initialization-x-lanes"},
        {narrow_initialization_x_lanes.has_value(), "narrow-initialization-x-lanes"},
        {valid_outer_extraction, "outer-coordinate-extraction"},
        {valid_outer_selection, "outer-coordinate-selection"},
        {skip_empty_wave.has_value(), "skip-empty-wave"},
        {save_selected_exec.has_value(), "save-selected-exec"},
        {count_selected.has_value(), "count-selected"},
        {scale_x.has_value(), "scale-x"},
        {address.has_value(), "address"},
        {use_wide_store ? more_literal.has_value() : end.has_value(), "bounds-limit"},
        {use_wide_store ? (use_quad_store ? store_quad.has_value() : store_wide.has_value())
                        : store_low.has_value(),
         "store-low-or-wide"},
        {use_wide_store || store_high.has_value(), "store-high"},
        {advance.has_value(), "advance"},
        {use_wide_store ? more_literal.has_value() : more.has_value(), "bounds-compare"},
        {retain_in_bounds.has_value(), "retain-in-bounds"},
        {loop.has_value(), "loop"},
        {wait_ds.has_value(), "wait-ds"},
        {barrier.has_value(), "barrier"},
    }};
    for (const auto &[valid, name] : encodings) {
      if (!valid)
        errors.emplace_back("ConSan MOI parallel workgroup-shadow initializer cannot encode " +
                            std::string(name));
    }
    errors.emplace_back(
        "ConSan MOI parallel workgroup-shadow initializer could not encode target state");
    return false;
  }

  words.push_back(*save_scc);
  words.push_back(*save_vcc);
  words.push_back(*save_exec);
  words.insert(words.end(), extract_initialization_x.begin(), extract_initialization_x.end());
  words.push_back(*select_initialization_x_lanes);
  words.push_back(*narrow_initialization_x_lanes);
  for (uint8_t dimension = 1u; dimension < layout.workitem_id_dimensions; ++dimension) {
    words.insert(words.end(), extract_outer_coordinate[dimension - 1u].begin(),
                 extract_outer_coordinate[dimension - 1u].end());
    words.push_back(*select_zero_outer_coordinate[dimension - 1u]);
    words.push_back(*narrow_zero_outer_coordinate[dimension - 1u]);
  }
  const size_t skip_empty_wave_index = words.size();
  words.push_back(*skip_empty_wave);
  words.push_back(*save_selected_exec);
  words.push_back(*count_selected);
  words.push_back(scale_stride);
  words.push_back(*scale_x);
  words.insert(words.end(), address->begin(), address->end());
  if (!use_wide_store)
    words.insert(words.end(), end->begin(), end->end());
  words.push_back(build_v_mov_b32_e32(zero_vgpr, scalar_positive_inline_u32(0), arch));
  if (use_wide_store) {
    for (uint16_t i = 1u; i < zero_tuple_size; ++i) {
      words.push_back(build_v_mov_b32_e32(static_cast<uint16_t>(zero_vgpr + i),
                                          scalar_positive_inline_u32(0), arch));
    }
    words.insert(words.end(), more_literal->begin(), more_literal->end());
  } else {
    words.push_back(*more);
  }
  words.push_back(*retain_in_bounds);
  const size_t loop_begin = words.size();
  if (use_wide_store) {
    const auto &store = use_quad_store ? *store_quad : *store_wide;
    words.insert(words.end(), store.begin(), store.end());
  } else {
    words.insert(words.end(), store_low->begin(), store_low->end());
    words.insert(words.end(), store_high->begin(), store_high->end());
  }
  words.insert(words.end(), advance->begin(), advance->end());
  if (use_wide_store)
    words.insert(words.end(), more_literal->begin(), more_literal->end());
  else
    words.push_back(*more);
  words.push_back(*retain_in_bounds);
  const size_t loop_branch_index = words.size();
  words.push_back(*loop);
  words.push_back(*wait_ds);
  const size_t restore_state_index = words.size();
  words.push_back(*restore_exec);
  words.push_back(*restore_vcc);
  words.insert(words.end(), barrier->begin(), barrier->end());
  words.push_back(*restore_scc);

  const int64_t skip_delta =
      static_cast<int64_t>(restore_state_index) - static_cast<int64_t>(skip_empty_wave_index + 1u);
  if (skip_delta < std::numeric_limits<int16_t>::min() ||
      skip_delta > std::numeric_limits<int16_t>::max()) {
    errors.emplace_back("ConSan MOI parallel workgroup-shadow branch exceeds SOPP range");
    return false;
  }
  words[skip_empty_wave_index] =
      *instrumentation::build_s_cbranch_execz(static_cast<int16_t>(skip_delta), arch);
  const int64_t loop_delta =
      static_cast<int64_t>(loop_begin) - static_cast<int64_t>(loop_branch_index + 1u);
  if (loop_delta < std::numeric_limits<int16_t>::min() ||
      loop_delta > std::numeric_limits<int16_t>::max()) {
    errors.emplace_back("ConSan MOI parallel workgroup-shadow loop exceeds SOPP range");
    return false;
  }
  words[loop_branch_index] =
      *instrumentation::build_s_cbranch_execnz(static_cast<int16_t>(loop_delta), arch);
  return true;
}

[[nodiscard]] bool append_moi_workgroup_shadow_initialization(
    std::vector<uint32_t> &words, const ConSanMoiWorkgroupShadowLayout &layout,
    uint16_t address_vgpr, uint16_t zero_vgpr, bool has_quad_zero_tuple,
    std::optional<uint16_t> state_sgpr, rj_code_arch_t arch, std::vector<std::string> &errors) {
  if (consan_moi_generation_tagged_workgroup_shadow(layout))
    return true;
  if (layout.workitem_id_dimensions >= 1u && layout.workitem_id_dimensions <= 3u) {
    return append_moi_parallel_workgroup_shadow_initialization(
        words, layout, address_vgpr, zero_vgpr, has_quad_zero_tuple, state_sgpr, arch, errors);
  }
  errors.emplace_back("ConSan MOI workgroup-shadow initializer has invalid dimensionality");
  return false;
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_owner_epoch_prologue_words(
    uint64_t prologue_text_offset, uint64_t original_entry_text_offset, uint16_t owner_vgpr,
    uint16_t epoch_vgpr, std::optional<uint16_t> workgroup_key_vgpr, uint16_t owner_shift_bits,
    ConSanMoiOwnerSource owner_source, std::optional<uint16_t> owner_sgpr, bool one_based_owner_ids,
    std::optional<uint16_t> persistent_owner_sgpr, std::optional<uint16_t> persistent_epoch_sgpr,
    std::optional<uint16_t> persistent_workgroup_key_sgpr,
    ConSanMoiPersistentWorkgroupRegisters record_replay_workgroup_sgprs,
    ConSanMoiPersistentWorkgroupRegisters record_replay_workgroup_vgprs,
    std::optional<ConSanMoiDispatchIdPreloadPlan> dispatch_plan,
    ConSanMoiDispatchIdCapture dispatch_capture,
    std::optional<uint16_t> runtime_sample_counter_sgpr, std::optional<uint16_t> return_pc_sgpr,
    std::optional<MoiEntryScalarBackup> entry_scalar_backup,
    std::optional<ConSanMoiWorkgroupShadowLayout> workgroup_shadow, bool has_quad_zero_tuple,
    std::optional<ConSanMoiWorkgroupSources> workgroup_sources,
    std::span<const uint32_t> displaced_entry_words, rj_code_arch_t arch,
    std::vector<std::string> &errors) {
  std::vector<uint32_t> words;
  words.reserve(48u + (entry_scalar_backup
                           ? static_cast<size_t>(entry_scalar_backup->sgpr_count) * 4u + 1u
                           : 0u));
  if (entry_scalar_backup && !append_moi_entry_scalar_backup(words, *entry_scalar_backup,
                                                             /*restore=*/false, arch, errors)) {
    return std::nullopt;
  }
  if (dispatch_plan &&
      !append_dispatch_id_capture_and_restore(
          words, *dispatch_plan, dispatch_capture,
          !workgroup_sources || !workgroup_sources->cdna_full_payload_base.has_value(), arch,
          errors)) {
    return std::nullopt;
  }
  if (runtime_sample_counter_sgpr) {
    if (!append_moi_entry_salu_write(
            words,
            build_s_mov_b32(*runtime_sample_counter_sgpr, scalar_positive_inline_u32(0), arch),
            arch)) {
      errors.emplace_back("ConSan MOI prologue cannot encode its sample-counter dependency delay");
      return std::nullopt;
    }
  }
  if (workgroup_shadow && workgroup_shadow->visible_evidence_sgpr &&
      !append_moi_entry_salu_write(words,
                                   build_s_mov_b32(*workgroup_shadow->visible_evidence_sgpr,
                                                   scalar_positive_inline_u32(0), arch),
                                   arch)) {
    errors.emplace_back("ConSan MOI prologue cannot initialize its visible-evidence latch");
    return std::nullopt;
  }
  // The workgroup-shadow initializer uses the entry-local scratch VGPRs. Capture a
  // scalar owner sourced from v0 before that scratch is allowed to clobber v0.
  if (persistent_owner_sgpr) {
    switch (owner_source) {
    case ConSanMoiOwnerSource::Automatic:
      errors.emplace_back("ConSan MOI scalar owner prologue has an unresolved automatic source");
      return std::nullopt;
    case ConSanMoiOwnerSource::WorkitemId: {
      const auto read_owner = instrumentation::build_v_readfirstlane_b32(*persistent_owner_sgpr,
                                                                         kRdna4WorkitemIdX, arch);
      if (!read_owner) {
        errors.emplace_back("ConSan MOI scalar owner prologue could not read its wave ID");
        return std::nullopt;
      }
      words.push_back(*read_owner);
      const auto wait = instrumentation::build_valu_to_salu_dependency_wait(arch);
      if (!wait)
        return std::nullopt;
      words.push_back(*wait);
      if (!append_moi_entry_salu_write(
              words,
              build_s_lshr_b32(*persistent_owner_sgpr, *persistent_owner_sgpr,
                               scalar_positive_inline_u32(owner_shift_bits), arch),
              arch) ||
          (one_based_owner_ids &&
           !append_moi_entry_salu_write(words,
                                        build_s_add_u32(*persistent_owner_sgpr,
                                                        *persistent_owner_sgpr,
                                                        scalar_positive_inline_u32(1), arch),
                                        arch))) {
        errors.emplace_back("ConSan MOI scalar owner prologue could not normalize its wave ID");
        return std::nullopt;
      }
      break;
    }
    case ConSanMoiOwnerSource::HwId: {
      if (!append_moi_resident_wave_owner(words, *persistent_owner_sgpr, one_based_owner_ids,
                                          arch)) {
        errors.emplace_back(
            "ConSan MOI scalar owner prologue could not encode resident-wave identity");
        return std::nullopt;
      }
      break;
    }
    }
  }
  if (workgroup_shadow && !append_moi_workgroup_shadow_initialization(
                              words, *workgroup_shadow, owner_vgpr, epoch_vgpr, has_quad_zero_tuple,
                              return_pc_sgpr, arch, errors)) {
    return std::nullopt;
  }
  if ((!record_replay_workgroup_sgprs.empty() && !record_replay_workgroup_sgprs.complete()) ||
      (!record_replay_workgroup_vgprs.empty() && !record_replay_workgroup_vgprs.complete()) ||
      (record_replay_workgroup_sgprs.complete() && record_replay_workgroup_vgprs.complete())) {
    errors.emplace_back(
        "ConSan MOI exact workgroup-tuple prologue requires one complete register tuple");
    return std::nullopt;
  }
  if (record_replay_workgroup_sgprs.complete() || record_replay_workgroup_vgprs.complete()) {
    if (!workgroup_sources) {
      errors.emplace_back("ConSan MOI exact workgroup-tuple prologue requires launch sources");
      return std::nullopt;
    }
    const std::array<const ConSanMoiWorkgroupSource *, 3> sources = {
        &workgroup_sources->x, &workgroup_sources->y, &workgroup_sources->z};
    const std::array<std::optional<uint16_t>, 3> scalar_destinations = {
        record_replay_workgroup_sgprs.x, record_replay_workgroup_sgprs.y,
        record_replay_workgroup_sgprs.z};
    const std::array<std::optional<uint16_t>, 3> vector_destinations = {
        record_replay_workgroup_vgprs.x, record_replay_workgroup_vgprs.y,
        record_replay_workgroup_vgprs.z};
    for (size_t index = 0; index < sources.size(); ++index) {
      const ConSanMoiWorkgroupSource &source = *sources[index];
      if (!source.is_well_formed() || !source.has_value()) {
        errors.emplace_back(
            "ConSan MOI exact workgroup-tuple prologue requires all launch coordinates");
        return std::nullopt;
      }
      if (scalar_destinations[index]) {
        if (!source.scalar_src) {
          errors.emplace_back(
              "ConSan MOI exact scalar workgroup tuple requires scalar launch sources");
          return std::nullopt;
        }
        const uint16_t destination = *scalar_destinations[index];
        const uint16_t source_operand = *source.scalar_src;
        if (!append_moi_entry_salu_write(words, build_s_mov_b32(destination, source_operand, arch),
                                         arch) ||
            (source.mask_low_16 &&
             (!append_moi_entry_salu_write(
                  words,
                  build_s_lshl_b32(destination, destination, scalar_positive_inline_u32(16), arch),
                  arch) ||
              !append_moi_entry_salu_write(
                  words,
                  build_s_lshr_b32(destination, destination, scalar_positive_inline_u32(16), arch),
                  arch))) ||
            (source.shift_right_16 &&
             !append_moi_entry_salu_write(
                 words,
                 build_s_lshr_b32(destination, destination, scalar_positive_inline_u32(16), arch),
                 arch))) {
          errors.emplace_back(
              "ConSan MOI exact scalar workgroup-tuple prologue could not copy a launch source");
          return std::nullopt;
        }
      } else if (vector_destinations[index]) {
        const uint16_t destination = *vector_destinations[index];
        if (source.has_value()) {
          if (!consan_detail::append_workgroup_source_value(words, source, destination, arch)) {
            errors.emplace_back(
                "ConSan MOI exact vector workgroup-tuple prologue could not copy a launch source");
            return std::nullopt;
          }
        } else {
          words.push_back(build_v_mov_b32_e32(destination, scalar_positive_inline_u32(0u), arch));
        }
      }
    }
  }
  if (persistent_workgroup_key_sgpr) {
    if (!workgroup_sources || !return_pc_sgpr) {
      errors.emplace_back(
          "ConSan MOI scalar workgroup-key prologue requires launch sources and scalar scratch");
      return std::nullopt;
    }
    const uint16_t coordinate = static_cast<uint16_t>(*return_pc_sgpr + 4u);
    const uint16_t temporary = static_cast<uint16_t>(*return_pc_sgpr + 5u);
    if (!append_moi_entry_salu_write(
            words,
            build_s_mov_b32(*persistent_workgroup_key_sgpr, scalar_positive_inline_u32(0), arch),
            arch)) {
      return std::nullopt;
    }
    std::vector<size_t> invalid_branches;
    const bool has_z = workgroup_sources->z.scalar_src.has_value();
    const bool has_y = workgroup_sources->y.scalar_src.has_value();
    const uint32_t x_bits = has_z ? 8u : has_y ? 10u : 20u;
    const uint32_t y_bits = has_z ? 6u : 10u;
    const auto append_coordinate = [&](const ConSanMoiWorkgroupSource &source, uint32_t bits,
                                       uint32_t shift) -> bool {
      if (!source.scalar_src)
        return true;
      if (!append_moi_entry_salu_write(words, build_s_mov_b32(coordinate, *source.scalar_src, arch),
                                       arch))
        return false;
      if (source.mask_low_16) {
        if (!append_moi_entry_salu_write(
                words,
                build_s_lshl_b32(coordinate, coordinate, scalar_positive_inline_u32(16), arch),
                arch) ||
            !append_moi_entry_salu_write(
                words,
                build_s_lshr_b32(coordinate, coordinate, scalar_positive_inline_u32(16), arch),
                arch))
          return false;
      }
      if (source.shift_right_16 &&
          !append_moi_entry_salu_write(
              words, build_s_lshr_b32(coordinate, coordinate, scalar_positive_inline_u32(16), arch),
              arch))
        return false;
      if (!append_moi_entry_salu_write(
              words,
              build_s_lshr_b32(temporary, coordinate, scalar_positive_inline_u32(bits), arch),
              arch))
        return false;
      const auto fits =
          instrumentation::build_s_cmp_eq_u32(temporary, scalar_positive_inline_u32(0), arch);
      if (!fits)
        return false;
      words.push_back(*fits);
      invalid_branches.push_back(words.size());
      words.push_back(*instrumentation::build_s_cbranch_scc0(0, arch));
      if (shift != 0u &&
          !append_moi_entry_salu_write(
              words,
              build_s_lshl_b32(coordinate, coordinate,
                               scalar_positive_inline_u32(static_cast<uint16_t>(shift)), arch),
              arch))
        return false;
      return append_moi_entry_salu_write(words,
                                         build_s_add_u32(*persistent_workgroup_key_sgpr,
                                                         *persistent_workgroup_key_sgpr, coordinate,
                                                         arch),
                                         arch);
    };
    if (!append_coordinate(workgroup_sources->x, x_bits, 0u) ||
        !append_coordinate(workgroup_sources->y, y_bits, x_bits) ||
        !append_coordinate(workgroup_sources->z, 6u, x_bits + y_bits) ||
        !append_moi_entry_salu_write(words,
                                     build_s_add_u32(*persistent_workgroup_key_sgpr,
                                                     *persistent_workgroup_key_sgpr,
                                                     scalar_positive_inline_u32(1), arch),
                                     arch) ||
        !append_moi_entry_salu_write(words,
                                     build_s_lshr_b32(temporary, *persistent_workgroup_key_sgpr,
                                                      scalar_positive_inline_u32(20), arch),
                                     arch)) {
      errors.emplace_back("ConSan MOI scalar workgroup-key prologue could not pack launch ID");
      return std::nullopt;
    }
    const auto key_fits =
        instrumentation::build_s_cmp_eq_u32(temporary, scalar_positive_inline_u32(0), arch);
    const auto select_key = instrumentation::build_s_cselect_b32(
        *persistent_workgroup_key_sgpr, *persistent_workgroup_key_sgpr,
        scalar_positive_inline_u32(0), arch);
    if (!key_fits || !select_key) {
      errors.emplace_back("ConSan MOI scalar workgroup-key prologue could not validate launch ID");
      return std::nullopt;
    }
    words.push_back(*key_fits);
    words.push_back(*select_key);
    const size_t skip_invalid = words.size();
    words.push_back(build_s_branch(0, arch));
    const size_t invalid_label = words.size();
    if (!append_moi_entry_salu_write(
            words,
            build_s_mov_b32(*persistent_workgroup_key_sgpr, scalar_positive_inline_u32(0), arch),
            arch))
      return std::nullopt;
    const size_t done_label = words.size();
    for (size_t branch_index : invalid_branches) {
      const int64_t delta =
          static_cast<int64_t>(invalid_label) - static_cast<int64_t>(branch_index + 1u);
      words[branch_index] =
          *instrumentation::build_s_cbranch_scc0(static_cast<int16_t>(delta), arch);
    }
    words[skip_invalid] =
        build_s_branch(static_cast<int16_t>(static_cast<int64_t>(done_label) -
                                            static_cast<int64_t>(skip_invalid + 1u)),
                       arch);
  } else if (workgroup_key_vgpr) {
    if (!workgroup_sources || !return_pc_sgpr) {
      errors.emplace_back(
          "ConSan MOI workgroup-key prologue requires launch sources and EXEC-save state");
      return std::nullopt;
    }
    words.push_back(build_v_mov_b32_e32(*workgroup_key_vgpr, scalar_positive_inline_u32(0), arch));
    ConSanOptions key_options;
    key_options.moi_exec_save_sgpr = return_pc_sgpr;
    if (!append_inline_workgroup_key(words, *workgroup_sources, key_options, *workgroup_key_vgpr,
                                     owner_vgpr, epoch_vgpr,
                                     /*original_exec_save_offset=*/20u, arch)) {
      errors.emplace_back("ConSan MOI workgroup-key prologue could not encode launch identity");
      return std::nullopt;
    }
    const auto restore_exec = instrumentation::build_s_mov_b64(
        kRdna4ExecLo, static_cast<uint16_t>(*return_pc_sgpr + 20u), arch);
    if (!restore_exec)
      return std::nullopt;
    words.push_back(*restore_exec);
  }
  if (workgroup_sources &&
      !append_cdna_full_workgroup_payload_restore(words, *workgroup_sources, arch, errors)) {
    return std::nullopt;
  }
  if (!persistent_owner_sgpr)
    switch (owner_source) {
    case ConSanMoiOwnerSource::Automatic:
      errors.emplace_back("ConSan MOI owner/epoch prologue has an unresolved automatic source");
      return std::nullopt;
    case ConSanMoiOwnerSource::WorkitemId: {
      auto owner_init = instrumentation::build_v_lshrrev_b32(
          owner_vgpr, scalar_positive_inline_u32(owner_shift_bits), kRdna4WorkitemIdX, arch);
      if (!owner_init) {
        errors.emplace_back("ConSan MOI owner/epoch prologue could not encode owner VGPR init");
        return std::nullopt;
      }
      words.push_back(*owner_init);
      if (one_based_owner_ids) {
        const auto make_nonzero = instrumentation::build_v_add_u32(
            owner_vgpr, scalar_positive_inline_u32(1), owner_vgpr, arch);
        if (!make_nonzero) {
          errors.emplace_back("ConSan MOI owner/epoch prologue could not bias owner IDs");
          return std::nullopt;
        }
        // Inline metadata reserves zero for absent ownership. Keep its persistent
        // owner IDs one-based; Record/Replay and Sampled retain zero-based IDs
        // because they use owner zero to elect the canonical wave.
        words.insert(words.end(), make_nonzero->begin(), make_nonzero->end());
      }
      break;
    }
    case ConSanMoiOwnerSource::HwId: {
      if (!owner_sgpr) {
        errors.emplace_back("ConSan MOI hw_id owner source requires RJ_CONSAN_MOI_OWNER_SGPR");
        return std::nullopt;
      }
      if (!append_moi_resident_wave_owner(words, *owner_sgpr, one_based_owner_ids, arch)) {
        errors.emplace_back(
            "ConSan MOI owner/epoch prologue could not encode resident-wave identity");
        return std::nullopt;
      }
      words.push_back(build_v_mov_b32_e32(owner_vgpr, *owner_sgpr, arch));
      break;
    }
    }
  if (persistent_epoch_sgpr) {
    if (!append_moi_entry_salu_write(
            words, build_s_mov_b32(*persistent_epoch_sgpr, scalar_positive_inline_u32(0), arch),
            arch))
      return std::nullopt;
  } else {
    words.push_back(build_v_mov_b32_e32(epoch_vgpr, scalar_positive_inline_u32(0), arch));
  }
  if (entry_scalar_backup && !append_moi_entry_scalar_backup(words, *entry_scalar_backup,
                                                             /*restore=*/true, arch, errors)) {
    return std::nullopt;
  }
  words.insert(words.end(), displaced_entry_words.begin(), displaced_entry_words.end());

  if (!append_moi_prologue_return(words, prologue_text_offset, original_entry_text_offset,
                                  return_pc_sgpr, arch, "owner/epoch prologue", errors)) {
    return std::nullopt;
  }
  return words;
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_private_epoch_prologue_words(
    uint64_t prologue_text_offset, uint64_t original_entry_text_offset, uint16_t scratch_vgpr,
    uint32_t epoch_offset, std::optional<uint32_t> owner_offset,
    std::optional<uint32_t> workgroup_key_offset,
    const ConSanMoiPersistentWorkgroupPrivateOffsets &record_replay_workgroup_offsets,
    const VgprSpillSequence &spill, std::optional<ConSanMoiWorkgroupShadowLayout> workgroup_shadow,
    std::optional<ConSanMoiWorkgroupSources> workgroup_sources,
    std::optional<ConSanMoiDispatchIdPreloadPlan> dispatch_plan,
    ConSanMoiDispatchIdCapture dispatch_capture,
    std::optional<uint16_t> runtime_sample_counter_sgpr, std::optional<uint16_t> return_pc_sgpr,
    rj_code_arch_t arch, std::vector<std::string> &errors) {
  const auto epoch_store =
      instrumentation::build_private_store_b32(scratch_vgpr, epoch_offset, arch);
  std::optional<std::vector<uint32_t>> owner_store;
  if (owner_offset)
    owner_store = instrumentation::build_private_store_b32(scratch_vgpr, *owner_offset, arch);
  std::optional<std::vector<uint32_t>> workgroup_key_store;
  if (workgroup_key_offset)
    workgroup_key_store =
        instrumentation::build_private_store_b32(scratch_vgpr, *workgroup_key_offset, arch);
  std::array<std::optional<std::vector<uint32_t>>, 3> record_replay_workgroup_stores;
  const std::array<std::optional<uint32_t>, 3> record_replay_workgroup_offset_values = {
      record_replay_workgroup_offsets.x, record_replay_workgroup_offsets.y,
      record_replay_workgroup_offsets.z};
  for (size_t index = 0; index < record_replay_workgroup_offset_values.size(); ++index) {
    if (record_replay_workgroup_offset_values[index]) {
      record_replay_workgroup_stores[index] = instrumentation::build_private_store_b32(
          scratch_vgpr, *record_replay_workgroup_offset_values[index], arch);
    }
  }
  const auto wait_store = instrumentation::build_s_wait_private_store0(arch);
  if (!epoch_store || (owner_offset && !owner_store) ||
      (workgroup_key_offset && !workgroup_key_store) ||
      std::ranges::any_of(std::views::iota(size_t{0}, record_replay_workgroup_stores.size()),
                          [&](size_t index) {
                            return record_replay_workgroup_offset_values[index] &&
                                   !record_replay_workgroup_stores[index];
                          }) ||
      !wait_store) {
    errors.emplace_back(
        "ConSan MOI private-epoch prologue could not encode persistent-state initialization");
    return std::nullopt;
  }

  std::vector<uint32_t> words;
  words.reserve(spill.save_words.size() + spill.restore_words.size() + 48u);
  if (dispatch_plan &&
      !append_dispatch_id_capture_and_restore(
          words, *dispatch_plan, dispatch_capture,
          !workgroup_sources || !workgroup_sources->cdna_full_payload_base.has_value(), arch,
          errors)) {
    return std::nullopt;
  }
  if (runtime_sample_counter_sgpr) {
    if (!append_moi_entry_salu_write(
            words,
            build_s_mov_b32(*runtime_sample_counter_sgpr, scalar_positive_inline_u32(0), arch),
            arch)) {
      errors.emplace_back(
          "ConSan MOI private-epoch prologue cannot encode its sample-counter dependency delay");
      return std::nullopt;
    }
  }
  if (workgroup_shadow && workgroup_shadow->visible_evidence_sgpr &&
      !append_moi_entry_salu_write(words,
                                   build_s_mov_b32(*workgroup_shadow->visible_evidence_sgpr,
                                                   scalar_positive_inline_u32(0), arch),
                                   arch)) {
    errors.emplace_back(
        "ConSan MOI private-epoch prologue cannot initialize its visible-evidence latch");
    return std::nullopt;
  }
  words.insert(words.end(), spill.save_words.begin(), spill.save_words.end());
  if (workgroup_shadow &&
      !append_moi_workgroup_shadow_initialization(
          words, *workgroup_shadow, scratch_vgpr, static_cast<uint16_t>(scratch_vgpr + 1u),
          /*has_quad_zero_tuple=*/false, return_pc_sgpr, arch, errors)) {
    return std::nullopt;
  }
  if (owner_store) {
    words.push_back(build_v_mov_b32_e32(scratch_vgpr, vector_source_vgpr(kRdna4WorkitemIdX), arch));
    words.insert(words.end(), owner_store->begin(), owner_store->end());
  }
  if (workgroup_key_store) {
    if (!workgroup_sources || !return_pc_sgpr) {
      errors.emplace_back("ConSan MOI private workgroup key requires entry ABI sources");
      return std::nullopt;
    }
    ConSanOptions key_options;
    key_options.moi_exec_save_sgpr = return_pc_sgpr;
    if (!append_inline_workgroup_key(words, *workgroup_sources, key_options, scratch_vgpr,
                                     static_cast<uint16_t>(scratch_vgpr + 1u),
                                     static_cast<uint16_t>(scratch_vgpr + 2u), 20u, arch)) {
      errors.emplace_back("ConSan MOI private prologue could not capture workgroup identity");
      return std::nullopt;
    }
    words.insert(words.end(), workgroup_key_store->begin(), workgroup_key_store->end());
    const auto restore_exec = instrumentation::build_s_mov_b64(
        kRdna4ExecLo, static_cast<uint16_t>(*return_pc_sgpr + 20u), arch);
    if (!restore_exec)
      return std::nullopt;
    words.push_back(*restore_exec);
  }
  if (record_replay_workgroup_offsets.complete()) {
    if (!workgroup_sources) {
      errors.emplace_back("ConSan MOI private exact workgroup tuple requires entry ABI sources");
      return std::nullopt;
    }
    const std::array<ConSanMoiWorkgroupSource, 3> sources = {
        workgroup_sources->x, workgroup_sources->y, workgroup_sources->z};
    for (size_t index = 0; index < sources.size(); ++index) {
      if (!sources[index].has_value() || !consan_detail::append_workgroup_source_value(
                                             words, sources[index], scratch_vgpr, arch)) {
        errors.emplace_back(
            "ConSan MOI private prologue requires every exact workgroup coordinate");
        return std::nullopt;
      }
      words.insert(words.end(), record_replay_workgroup_stores[index]->begin(),
                   record_replay_workgroup_stores[index]->end());
    }
  }
  if (workgroup_sources &&
      !append_cdna_full_workgroup_payload_restore(words, *workgroup_sources, arch, errors)) {
    return std::nullopt;
  }
  words.push_back(build_v_mov_b32_e32(scratch_vgpr, scalar_positive_inline_u32(0), arch));
  words.insert(words.end(), epoch_store->begin(), epoch_store->end());
  words.push_back(*wait_store);
  words.insert(words.end(), spill.restore_words.begin(), spill.restore_words.end());

  if (!append_moi_prologue_return(words, prologue_text_offset, original_entry_text_offset,
                                  return_pc_sgpr, arch, "private-epoch prologue", errors)) {
    return std::nullopt;
  }
  return words;
}

[[nodiscard]] bool kernel_contains_patch_anchor(const ConSanKernelInfo &kernel,
                                                const ConSanPatchInfo &patch) {
  return kernel.has_text_range && patch.anchor_offset >= kernel.entry_text_offset &&
         patch.anchor_offset - kernel.entry_text_offset < kernel.code_size;
}

[[nodiscard]] bool kernel_owns_patch(const ConSanKernelInfo &kernel, const ConSanPatchInfo &patch) {
  if (!patch.owner_descriptor_file_offsets.empty()) {
    return std::ranges::find(patch.owner_descriptor_file_offsets, kernel.descriptor_file_offset) !=
           patch.owner_descriptor_file_offsets.end();
  }
  return kernel_contains_patch_anchor(kernel, patch);
}

[[nodiscard]] bool enable_moi_full_workgroup_id_payload(rj_code_arch_t arch, ConSanResult &result) {
  result.moi_full_workgroup_id_descriptor_offsets.clear();
  if (result.elf_bytes.empty() ||
      (arch != ROCJITSU_CODE_ARCH_CDNA3 && arch != ROCJITSU_CODE_ARCH_CDNA4 &&
       arch != ROCJITSU_CODE_ARCH_RDNA4 && arch != ROCJITSU_CODE_ARCH_GFX1250)) {
    return true;
  }

  std::unordered_set<uint64_t> owner_descriptor_offsets;
  std::unordered_set<std::string_view> owner_names;
  for (const ConSanPatchInfo &patch : result.patches) {
    const bool exact_cdna_workgroup_capture =
        patch.kind == ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue ||
        patch.kind == ConSanPatchKind::KernelEntryMoiPrivateEpochPrologue;
    if ((arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4) &&
        (!exact_cdna_workgroup_capture ||
         (!result.resolved_moi_record_replay_workgroup_sgprs.complete() &&
          !patch.persistent_record_replay_workgroup_vgprs.complete() &&
          !patch.persistent_record_replay_workgroup_private_offsets.complete()))) {
      continue;
    }
    for (uint64_t descriptor_offset : patch.owner_descriptor_file_offsets) {
      owner_descriptor_offsets.insert(descriptor_offset);
      const auto owner = std::ranges::find(result.kernels, descriptor_offset,
                                           &ConSanKernelInfo::descriptor_file_offset);
      if (owner == result.kernels.end()) {
        result.errors.emplace_back(
            "ConSan MOI could not resolve an owner before enabling workgroup IDs");
        return false;
      }
      owner_names.insert(owner->name);
    }
  }
  if (owner_names.empty())
    return true;

  AmdGpuCodeObject code_object(result.elf_bytes.data(), result.elf_bytes.size());
  if (!code_object.is_valid()) {
    result.errors.emplace_back(
        "ConSan MOI could not parse its replacement before enabling workgroup IDs");
    return false;
  }
  CodeObjectPatcher patcher(code_object);
  std::unordered_set<std::string_view> patched_owner_names;
  for (const AmdGpuKernelInfo &kernel : code_object.kernels()) {
    if (!owner_names.contains(kernel.name))
      continue;
    if (!patched_owner_names.insert(kernel.name).second) {
      result.errors.emplace_back(
          "ConSan MOI found duplicate owner names while enabling workgroup IDs");
      return false;
    }
    const uint64_t descriptor_offset = kernel.descriptor_file_offset;
    const std::span<const uint8_t> current_image = patcher.image_bytes();
    if (descriptor_offset > current_image.size() ||
        sizeof(KD) > current_image.size() - descriptor_offset) {
      result.errors.emplace_back("ConSan MOI workgroup-ID descriptor exceeds replacement bytes");
      return false;
    }
    KD descriptor{};
    std::memcpy(&descriptor, current_image.data() + descriptor_offset, sizeof(descriptor));
    if (arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4) {
      const uint32_t user_sgpr_count = moi_descriptor_user_sgpr_count(descriptor);
      const uint32_t required_sgpr_count =
          user_sgpr_count + 3u +
          (AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2,
                           kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_INFO) != 0
               ? 1u
               : 0u);
      if (!grow_moi_descriptor_sgpr_allocation(descriptor, required_sgpr_count, arch)) {
        result.errors.emplace_back(
            "ConSan MOI could not allocate the full CDNA workgroup-ID payload");
        return false;
      }
    }
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X,
                    1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y,
                    1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z,
                    1u);
    if (!patcher.patch_kernel_descriptor(
            descriptor_offset,
            std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(&descriptor),
                                     sizeof(descriptor)))) {
      result.errors.emplace_back("ConSan MOI could not enable its full workgroup-ID payload");
      return false;
    }
  }
  if (patched_owner_names.size() != owner_names.size()) {
    result.errors.emplace_back(
        "ConSan MOI could not resolve every replacement owner while enabling workgroup IDs");
    return false;
  }
  result.elf_bytes = std::move(patcher).emit();
  result.moi_full_workgroup_id_descriptor_offsets.assign(owner_descriptor_offsets.begin(),
                                                         owner_descriptor_offsets.end());
  std::ranges::sort(result.moi_full_workgroup_id_descriptor_offsets);
  return true;
}

[[nodiscard]] bool
grow_moi_kernel_descriptor_registers(CodeObjectPatcher &patcher, const ConSanKernelInfo &kernel,
                                     uint32_t required_vgpr_count, uint32_t required_sgpr_count,
                                     rj_code_arch_t arch, std::vector<std::string> &errors) {
  const std::span<const uint8_t> current_image = patcher.image_bytes();
  if (kernel.descriptor_file_offset > current_image.size() ||
      sizeof(KD) > current_image.size() - kernel.descriptor_file_offset) {
    errors.emplace_back("ConSan MOI descriptor register growth exceeds ELF bytes");
    return false;
  }

  KD desc{};
  std::memcpy(&desc, current_image.data() + kernel.descriptor_file_offset, sizeof(desc));
  if (!grow_moi_descriptor_vgpr_allocation(desc, required_vgpr_count, arch)) {
    errors.emplace_back("ConSan MOI could not grow descriptor VGPR allocation");
    return false;
  }
  if (required_sgpr_count != 0 &&
      !grow_moi_descriptor_sgpr_allocation(desc, required_sgpr_count, arch)) {
    errors.emplace_back("ConSan MOI could not grow descriptor SGPR allocation");
    return false;
  }
  if (!patcher.patch_kernel_descriptor(kernel.descriptor_file_offset,
                                       {reinterpret_cast<const uint8_t *>(&desc), sizeof(desc)})) {
    errors.emplace_back("ConSan MOI could not patch descriptor register allocation");
    return false;
  }
  return true;
}

[[nodiscard]] bool apply_moi_dispatch_id_descriptor(CodeObjectPatcher &patcher,
                                                    uint64_t descriptor_file_offset,
                                                    const ConSanMoiDispatchIdPreloadPlan &plan,
                                                    ConSanMoiDispatchIdCapture capture,
                                                    rj_code_arch_t arch,
                                                    std::vector<std::string> &errors) {
  const std::span<const uint8_t> current_image = patcher.image_bytes();
  if (descriptor_file_offset > current_image.size() ||
      sizeof(KD) > current_image.size() - descriptor_file_offset) {
    errors.emplace_back("ConSan MOI dispatch-ID descriptor exceeds ELF bytes");
    return false;
  }
  if (!plan.supported() || !capture.unambiguous() ||
      (capture.sgpr && static_cast<uint32_t>(*capture.sgpr) + 2u > kMaxSgprs) ||
      (capture.vgpr && static_cast<uint32_t>(*capture.vgpr) + 2u > kMaxVgprs)) {
    errors.emplace_back("ConSan MOI dispatch-ID descriptor has an unsupported preload plan");
    return false;
  }

  KD desc{};
  std::memcpy(&desc, current_image.data() + descriptor_file_offset, sizeof(desc));
  const ConSanMoiDispatchIdPreloadPlan current_plan =
      moi_descriptor_dispatch_id_preload_plan(desc, arch);
  if (current_plan.support != plan.support ||
      current_plan.dispatch_id_sgpr != plan.dispatch_id_sgpr ||
      current_plan.original_user_sgpr_count != plan.original_user_sgpr_count ||
      current_plan.system_sgpr_count != plan.system_sgpr_count) {
    errors.emplace_back("ConSan MOI dispatch-ID descriptor changed after preload planning");
    return false;
  }

  if (plan.descriptor_change_required()) {
    AMDHSA_BITS_SET(desc.kernel_code_properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID,
                    1u);
    AMDHSA_BITS_SET(desc.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT,
                    plan.expanded_user_sgpr_count);
    if (plan.requires_kernarg_reload()) {
      AMDHSA_BITS_SET(desc.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_LENGTH,
                      plan.replacement_kernarg_preload_length);
    }
  }
  const uint32_t required_sgpr_count =
      capture.sgpr
          ? std::max<uint32_t>(plan.required_sgpr_count, static_cast<uint32_t>(*capture.sgpr) + 2u)
          : plan.required_sgpr_count;
  if (!grow_moi_descriptor_sgpr_allocation(desc, required_sgpr_count, arch)) {
    errors.emplace_back("ConSan MOI could not grow dispatch-ID descriptor SGPR allocation");
    return false;
  }
  if (!patcher.patch_kernel_descriptor(
          descriptor_file_offset,
          std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(&desc), sizeof(desc)))) {
    errors.emplace_back("ConSan MOI could not patch dispatch-ID descriptor state");
    return false;
  }
  return true;
}

void rollback_moi_dispatch_id_as_unsupported(ConSanResult &result, std::string reason) {
  result.outcome = ConSanTransformOutcome::Unsupported;
  result.modified = false;
  result.final_validation_passed = false;
  result.elf_bytes.clear();
  result.patches.clear();
  result.warnings.emplace_back("ConSan MOI dispatch-ID capture is unsupported: " +
                               std::move(reason));
}

[[nodiscard]] std::string
dispatch_id_preload_rejection_reason(std::string_view kernel_name,
                                     const ConSanMoiDispatchIdPreloadPlan &plan,
                                     ConSanMoiDispatchIdCapture capture) {
  return "kernel '" + std::string(kernel_name) +
         "' preload plan support=" + std::to_string(static_cast<uint32_t>(plan.support)) +
         " user_sgprs=" + std::to_string(plan.original_user_sgpr_count) +
         " required_sgprs=" + std::to_string(plan.required_sgpr_count) +
         (capture.sgpr ? " persistent_sgpr=" + std::to_string(*capture.sgpr)
                       : " persistent_vgpr=" + std::to_string(capture.vgpr.value_or(kMaxVgprs)));
}

void note_dispatch_id_patch_info(ConSanPatchInfo &info, const ConSanMoiDispatchIdPreloadPlan &plan,
                                 ConSanMoiDispatchIdCapture capture) {
  info.dispatch_id_capture_sgpr = capture.sgpr;
  info.dispatch_id_capture_vgpr = capture.vgpr;
  info.dispatch_id_source_sgpr = plan.dispatch_id_sgpr;
  info.dispatch_id_original_user_sgpr_count = plan.original_user_sgpr_count;
  info.dispatch_id_expanded_user_sgpr_count = plan.expanded_user_sgpr_count;
  info.dispatch_id_system_sgpr_count = plan.system_sgpr_count;
  info.dispatch_id_kernarg_reload_sgpr = plan.kernarg_reload_sgpr;
  info.dispatch_id_kernarg_reload_base_sgpr = plan.kernarg_reload_base_sgpr;
  info.dispatch_id_kernarg_reload_offset_dwords = plan.kernarg_reload_offset_dwords;
  info.dispatch_id_kernarg_reload_count = plan.kernarg_reload_count;
  info.dispatch_id_shifted_system_sgpr_count = plan.shifted_system_sgpr_count;
  info.dispatch_id_system_sgpr_shift = plan.system_sgpr_shift;
  info.dispatch_id_original_kernarg_preload_length = plan.original_kernarg_preload_length;
  info.dispatch_id_replacement_kernarg_preload_length = plan.replacement_kernarg_preload_length;
  info.dispatch_id_required_sgpr_count = static_cast<uint16_t>(
      capture.sgpr
          ? std::max<uint32_t>(plan.required_sgpr_count, static_cast<uint32_t>(*capture.sgpr) + 2u)
          : plan.required_sgpr_count);
  info.dispatch_id_preload_inserted = plan.descriptor_change_required();
}

void try_apply_private_epoch_prologue_patch(const ConSanOptions &options, rj_code_arch_t arch,
                                            ConSanResult &result) {
  if (!result.modified || result.elf_bytes.empty()) {
    result.warnings.emplace_back(
        "ConSan MOI private-epoch prologue skipped because no private-epoch access probe was "
        "emitted");
    return;
  }

  std::span<const uint8_t> active_bytes(result.elf_bytes.data(), result.elf_bytes.size());
  AmdGpuCodeObject code_object(active_bytes.data(), active_bytes.size());
  CodeObjectPatcher patcher(code_object);
  const std::span<const uint8_t> old_text = patcher.text_bytes();
  if (old_text.empty()) {
    result.errors.emplace_back("ConSan MOI private-epoch prologue found no .text section");
    return;
  }
  const ConSanMoiDispatchIdCapture dispatch_capture{.sgpr = options.moi_dispatch_id_sgpr,
                                                    .vgpr = options.moi_dispatch_id_vgpr};
  if (dispatch_capture.present() && !dispatch_capture.unambiguous()) {
    result.errors.emplace_back(
        "ConSan MOI private-epoch prologue has multiple dispatch-ID representations");
    return;
  }

  struct PlannedPrivateEpochPrologue {
    const ConSanKernelInfo *kernel = nullptr;
    uint64_t active_descriptor_file_offset = 0;
    uint64_t active_entry_text_offset = 0;
    uint16_t scratch_vgpr = 0;
    MoiPrivateEpochLayout layout;
    VgprSpillSequence spill;
    std::optional<ConSanMoiWorkgroupShadowLayout> workgroup_shadow;
    std::optional<ConSanMoiWorkgroupSources> workgroup_sources;
    uint32_t required_private_bytes = 0;
    std::optional<ConSanMoiDispatchIdPreloadPlan> dispatch_plan;
    bool has_kernarg_preload = false;
  };
  std::vector<PlannedPrivateEpochPrologue> planned;
  MoiDescriptorPrivateRequirements private_requirements;
  for (const ConSanKernelInfo &kernel : result.kernels) {
    const auto access_patch =
        std::ranges::find_if(result.patches, [&](const ConSanPatchInfo &patch) {
          return patch.persistent_epoch_private_offset && patch.scratch_vgpr &&
                 kernel_owns_patch(kernel, patch);
        });
    if (access_patch == result.patches.end())
      continue;

    const auto active_kernel =
        std::ranges::find_if(code_object.kernels(), [&](const AmdGpuKernelInfo &candidate) {
          return candidate.name == kernel.name;
        });
    if (active_kernel == code_object.kernels().end()) {
      result.warnings.emplace_back(
          "ConSan MOI private-epoch prologue could not resolve an active kernel descriptor");
      continue;
    }

    if (!access_patch->persistent_epoch_private_offset)
      continue;
    std::optional<ConSanMoiDispatchIdPreloadPlan> dispatch_plan;
    bool has_kernarg_preload = false;
    if (dispatch_capture.present()) {
      if (active_kernel->descriptor_file_offset > active_bytes.size() ||
          sizeof(KD) > active_bytes.size() - active_kernel->descriptor_file_offset) {
        rollback_moi_dispatch_id_as_unsupported(result,
                                                "private-epoch descriptor exceeds ELF bytes");
        return;
      }
      KD descriptor{};
      std::memcpy(&descriptor, active_bytes.data() + active_kernel->descriptor_file_offset,
                  sizeof(descriptor));
      dispatch_plan = moi_descriptor_dispatch_id_preload_plan(descriptor, arch);
      if (!dispatch_plan->supported() ||
          (dispatch_capture.sgpr && *dispatch_capture.sgpr < dispatch_plan->required_sgpr_count)) {
        rollback_moi_dispatch_id_as_unsupported(
            result,
            dispatch_id_preload_rejection_reason(kernel.name, *dispatch_plan, dispatch_capture));
        return;
      }
      has_kernarg_preload = moi_descriptor_has_kernarg_preload(descriptor);
      if (has_kernarg_preload && kernel.code_size <= kAmdhsaKernelEntryAlignment) {
        rollback_moi_dispatch_id_as_unsupported(
            result, "kernarg-preload kernel has no valid secondary hardware entry");
        return;
      }
    }
    const bool has_private_owner = access_patch->persistent_owner_private_offset.has_value();
    const bool has_private_workgroup_key =
        access_patch->persistent_workgroup_key_private_offset.has_value();
    const bool has_private_record_replay_workgroup =
        access_patch->persistent_record_replay_workgroup_private_offsets.complete();
    uint32_t persistent_end =
        *access_patch->persistent_epoch_private_offset + SpillManager::kSlotBytes;
    if (has_private_owner)
      persistent_end = std::max(persistent_end, *access_patch->persistent_owner_private_offset +
                                                    SpillManager::kSlotBytes);
    if (has_private_workgroup_key)
      persistent_end =
          std::max(persistent_end, *access_patch->persistent_workgroup_key_private_offset +
                                       SpillManager::kSlotBytes);
    for (const std::optional<uint32_t> offset :
         {access_patch->persistent_record_replay_workgroup_private_offsets.x,
          access_patch->persistent_record_replay_workgroup_private_offsets.y,
          access_patch->persistent_record_replay_workgroup_private_offsets.z}) {
      if (offset)
        persistent_end = std::max(persistent_end, *offset + SpillManager::kSlotBytes);
    }
    MoiPrivateEpochLayout layout{
        .epoch_offset = *access_patch->persistent_epoch_private_offset,
        .owner_offset = access_patch->persistent_owner_private_offset,
        .workgroup_key_offset = access_patch->persistent_workgroup_key_private_offset,
        .record_replay_workgroup_offsets =
            access_patch->persistent_record_replay_workgroup_private_offsets,
        .ephemeral_base = access_patch->persistent_private_state_end.value_or(persistent_end)};

    const auto private_limit = address_free_scratch_private_limit(arch);
    if (!private_limit) {
      result.warnings.emplace_back(
          "ConSan MOI private-epoch prologue has no address-free scratch support");
      continue;
    }
    SpillManager manager(layout.ephemeral_base, *private_limit);
    uint8_t workitem_id_dimensions = 1u;
    if (access_patch->workgroup_shadow_size != 0u) {
      if (active_kernel->descriptor_file_offset > active_bytes.size() ||
          sizeof(KD) > active_bytes.size() - active_kernel->descriptor_file_offset) {
        result.errors.emplace_back(
            "ConSan MOI private-epoch prologue workitem descriptor exceeds ELF bytes");
        return;
      }
      KD descriptor{};
      std::memcpy(&descriptor, active_bytes.data() + active_kernel->descriptor_file_offset,
                  sizeof(descriptor));
      const auto dimensions = moi_descriptor_workitem_id_dimensions(descriptor);
      if (!dimensions) {
        result.errors.emplace_back(
            "ConSan MOI private-epoch prologue has invalid workitem-ID dimensions");
        return;
      }
      workitem_id_dimensions = *dimensions;
    }
    const std::optional<ConSanMoiWorkgroupShadowLayout> workgroup_shadow =
        access_patch->workgroup_shadow_size == 0u
            ? std::nullopt
            : std::optional<ConSanMoiWorkgroupShadowLayout>(ConSanMoiWorkgroupShadowLayout{
                  .base = access_patch->workgroup_shadow_base,
                  .size = access_patch->workgroup_shadow_size,
                  .validity_base = access_patch->workgroup_shadow_validity_base,
                  .validity_size = access_patch->workgroup_shadow_validity_size,
                  .required_group_segment_size = access_patch->required_group_segment_size,
                  .workitem_id_dimensions = workitem_id_dimensions,
                  .initialization_lanes = moi_workgroup_shadow_initialization_lanes(
                      active_kernel->required_workgroup_size, arch),
                  .lazy_initialization = access_patch->workgroup_shadow_lazy_initialization,
                  .compact = access_patch->workgroup_shadow_compact,
                  .visible_evidence_sgpr = std::nullopt,
              });
    std::optional<ConSanMoiWorkgroupSources> workgroup_sources;
    if (has_private_workgroup_key || has_private_record_replay_workgroup) {
      std::vector<std::string> source_errors;
      std::optional<uint16_t> full_payload_user_sgpr_count;
      if (has_private_record_replay_workgroup &&
          (arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4)) {
        if (active_kernel->descriptor_file_offset > active_bytes.size() ||
            sizeof(KD) > active_bytes.size() - active_kernel->descriptor_file_offset) {
          result.errors.emplace_back("ConSan MOI private exact-tuple descriptor exceeds ELF bytes");
          return;
        }
        KD descriptor{};
        std::memcpy(&descriptor, active_bytes.data() + active_kernel->descriptor_file_offset,
                    sizeof(descriptor));
        full_payload_user_sgpr_count =
            dispatch_plan ? dispatch_plan->expanded_user_sgpr_count
                          : static_cast<uint16_t>(moi_descriptor_user_sgpr_count(descriptor));
      }
      workgroup_sources = moi_descriptor_workgroup_sources(
          active_bytes, active_kernel->descriptor_file_offset, arch, source_errors,
          /*uses_gfx1250_cluster_workgroup_id=*/false, full_payload_user_sgpr_count);
      if (!workgroup_sources) {
        result.errors.insert(result.errors.end(), source_errors.begin(), source_errors.end());
        return;
      }
    }
    auto spill = build_vgpr_spill_sequence(
        manager, *access_patch->scratch_vgpr,
        has_private_workgroup_key ? 3u : (workgroup_shadow ? 2u : 1u), arch);
    if (!spill) {
      result.warnings.emplace_back(
          "ConSan MOI private-epoch prologue could not preserve its temporary VGPR");
      continue;
    }
    uint32_t required_private_bytes = spill->total_private_bytes;
    for (const ConSanPatchInfo &patch : result.patches) {
      if (kernel_owns_patch(kernel, patch))
        required_private_bytes =
            std::max(required_private_bytes, patch.required_private_segment_size);
    }
    private_requirements[kernel.descriptor_file_offset] = required_private_bytes;
    planned.push_back({&kernel, active_kernel->descriptor_file_offset,
                       active_kernel->entry_text_offset, *access_patch->scratch_vgpr, layout,
                       *spill, workgroup_shadow, workgroup_sources, required_private_bytes,
                       dispatch_plan, has_kernarg_preload});
  }

  if (planned.empty()) {
    result.warnings.emplace_back("ConSan MOI private-epoch prologue found no patchable kernels");
    return;
  }
  if (!apply_spill_descriptor_requirements(patcher, code_object, active_bytes, result,
                                           private_requirements, result.errors))
    return;
  if (dispatch_capture.present()) {
    for (const PlannedPrivateEpochPrologue &item : planned) {
      if (!item.dispatch_plan || !apply_moi_dispatch_id_descriptor(
                                     patcher, item.active_descriptor_file_offset,
                                     *item.dispatch_plan, dispatch_capture, arch, result.errors)) {
        return;
      }
    }
  }

  std::vector<uint8_t> new_text(old_text.begin(), old_text.end());
  std::vector<ConSanPatchInfo> patches;
  patches.reserve(planned.size());
  for (const PlannedPrivateEpochPrologue &item : planned) {
    append_nop_padding_to_alignment(new_text, kAmdhsaKernelEntryAlignment, arch);
    const uint64_t prologue_text_offset = static_cast<uint64_t>(new_text.size());
    std::optional<ConSanMoiWorkgroupShadowLayout> prologue_workgroup_shadow = item.workgroup_shadow;
    if (prologue_workgroup_shadow) {
      prologue_workgroup_shadow->visible_evidence_sgpr =
          inline_shadow_visible_evidence_sgpr(options);
    }
    auto words = build_private_epoch_prologue_words(
        prologue_text_offset, item.active_entry_text_offset, item.scratch_vgpr,
        item.layout.epoch_offset, item.layout.owner_offset, item.layout.workgroup_key_offset,
        item.layout.record_replay_workgroup_offsets, item.spill, prologue_workgroup_shadow,
        item.workgroup_sources, item.dispatch_plan, dispatch_capture,
        moi_runtime_sample_counter_sgpr(options), options.moi_exec_save_sgpr, arch, result.errors);
    if (!words)
      return;
    if (!patcher.redirect_kernel_entry(item.active_descriptor_file_offset,
                                       item.active_entry_text_offset, prologue_text_offset)) {
      result.errors.emplace_back(
          "ConSan MOI private-epoch prologue could not redirect kernel entry");
      return;
    }
    std::optional<uint64_t> primary_body_offset;
    std::optional<uint64_t> secondary_body_offset;
    if (item.has_kernarg_preload) {
      const uint64_t secondary_prologue_offset = util::align_up(
          prologue_text_offset + words->size() * sizeof(uint32_t), kAmdhsaKernelEntryAlignment);
      auto secondary_words = build_private_epoch_prologue_words(
          secondary_prologue_offset, item.active_entry_text_offset + kAmdhsaKernelEntryAlignment,
          item.scratch_vgpr, item.layout.epoch_offset, item.layout.owner_offset,
          item.layout.workgroup_key_offset, item.layout.record_replay_workgroup_offsets, item.spill,
          prologue_workgroup_shadow, item.workgroup_sources, item.dispatch_plan, dispatch_capture,
          moi_runtime_sample_counter_sgpr(options), options.moi_exec_save_sgpr, arch,
          result.errors);
      if (!secondary_words)
        return;
      if (words->size() * sizeof(uint32_t) <= kAmdhsaKernelEntryAlignment &&
          secondary_words->size() * sizeof(uint32_t) <= kAmdhsaKernelEntryAlignment) {
        append_words_bytes(new_text, *words);
        append_nop_padding_to_alignment(new_text, kAmdhsaKernelEntryAlignment, arch);
        append_words_bytes(new_text, *secondary_words);
      } else {
        constexpr uint64_t kPairedEntryWindow = 2u * kAmdhsaKernelEntryAlignment;
        primary_body_offset = prologue_text_offset + kPairedEntryWindow;
        words = build_private_epoch_prologue_words(
            *primary_body_offset, item.active_entry_text_offset, item.scratch_vgpr,
            item.layout.epoch_offset, item.layout.owner_offset, item.layout.workgroup_key_offset,
            item.layout.record_replay_workgroup_offsets, item.spill, prologue_workgroup_shadow,
            item.workgroup_sources, item.dispatch_plan, dispatch_capture,
            moi_runtime_sample_counter_sgpr(options), options.moi_exec_save_sgpr, arch,
            result.errors);
        if (!words)
          return;
        secondary_body_offset = *primary_body_offset + words->size() * sizeof(uint32_t);
        secondary_words = build_private_epoch_prologue_words(
            *secondary_body_offset, item.active_entry_text_offset + kAmdhsaKernelEntryAlignment,
            item.scratch_vgpr, item.layout.epoch_offset, item.layout.owner_offset,
            item.layout.workgroup_key_offset, item.layout.record_replay_workgroup_offsets,
            item.spill, prologue_workgroup_shadow, item.workgroup_sources, item.dispatch_plan,
            dispatch_capture, moi_runtime_sample_counter_sgpr(options), options.moi_exec_save_sgpr,
            arch, result.errors);
        const auto primary_relay =
            compute_sopp_branch_simm16(prologue_text_offset, *primary_body_offset);
        const auto secondary_relay = compute_sopp_branch_simm16(
            prologue_text_offset + kAmdhsaKernelEntryAlignment, *secondary_body_offset);
        if (!secondary_words || !primary_relay || !secondary_relay) {
          rollback_moi_dispatch_id_as_unsupported(
              result, "could not encode private-epoch kernarg-preload prologue relays");
          return;
        }
        std::vector<uint32_t> relays(kPairedEntryWindow / sizeof(uint32_t), build_s_nop(0, arch));
        relays.front() = build_s_branch(*primary_relay, arch);
        relays[kAmdhsaKernelEntryAlignment / sizeof(uint32_t)] =
            build_s_branch(*secondary_relay, arch);
        append_words_bytes(new_text, relays);
        append_words_bytes(new_text, *words);
        append_words_bytes(new_text, *secondary_words);
      }
    } else {
      append_words_bytes(new_text, *words);
    }

    ConSanPatchInfo info;
    info.kind = ConSanPatchKind::KernelEntryMoiPrivateEpochPrologue;
    info.anchor_offset = item.kernel->entry_text_offset;
    info.trampoline_offset = prologue_text_offset;
    info.trampoline_size = static_cast<uint32_t>(words->size() * sizeof(uint32_t));
    info.dispatch_id_primary_prologue_offset = primary_body_offset;
    info.dispatch_id_secondary_prologue_offset = secondary_body_offset;
    info.scratch_vgpr = item.scratch_vgpr;
    info.persistent_epoch_private_offset = item.layout.epoch_offset;
    info.persistent_owner_private_offset = item.layout.owner_offset;
    info.persistent_workgroup_key_private_offset = item.layout.workgroup_key_offset;
    info.persistent_record_replay_workgroup_private_offsets =
        item.layout.record_replay_workgroup_offsets;
    info.persistent_private_state_end = item.layout.ephemeral_base;
    info.spilled_vgpr_count = item.spill.vgpr_count;
    info.required_private_segment_size = item.required_private_bytes;
    note_dynamic_stack_private_requirement(info, &item.spill);
    if (item.dispatch_plan && dispatch_capture.present())
      note_dispatch_id_patch_info(info, *item.dispatch_plan, dispatch_capture);
    info.trampoline_size = static_cast<uint32_t>(new_text.size() - prologue_text_offset);
    info.owner_descriptor_file_offsets.push_back(item.kernel->descriptor_file_offset);
    patches.push_back(info);
  }

  if (!replace_consan_text(patcher, new_text, options, "MOI private-epoch prologue", result)) {
    return;
  }
  result.elf_bytes = std::move(patcher).emit();
  result.patches.insert(result.patches.end(), patches.begin(), patches.end());
  result.modified = true;
}

[[nodiscard]] uint32_t moi_owner_epoch_prologue_required_vgpr_count(
    const ConSanOptions &options,
    const std::optional<ConSanMoiWorkgroupShadowLayout> &workgroup_shadow, rj_code_arch_t arch) {
  if (!options.moi_owner_vgpr || !options.moi_epoch_vgpr)
    return 0u;
  uint32_t required = std::max<uint32_t>(*options.moi_owner_vgpr, *options.moi_epoch_vgpr) + 1u;
  if (workgroup_shadow) {
    const uint32_t shadow_zero_tuple_count =
        options.automatic_moi_persistent_vgprs && arch == ROCJITSU_CODE_ARCH_GFX1250 ? 4u : 2u;
    required = std::max<uint32_t>(required, *options.moi_epoch_vgpr + shadow_zero_tuple_count);
  }
  if (options.moi_workgroup_key_vgpr)
    required = std::max<uint32_t>(required, *options.moi_workgroup_key_vgpr + 1u);
  if (options.moi_dispatch_id_vgpr)
    required = std::max<uint32_t>(required, *options.moi_dispatch_id_vgpr + 2u);
  for (const std::optional<uint16_t> reg :
       {options.moi_record_replay_workgroup_vgprs.x, options.moi_record_replay_workgroup_vgprs.y,
        options.moi_record_replay_workgroup_vgprs.z}) {
    if (reg)
      required = std::max<uint32_t>(required, *reg + 1u);
  }
  return required;
}

[[nodiscard]] bool
moi_entry_scalar_backup_preserves_persistent_outputs(const ConSanOptions &options,
                                                     uint16_t sgpr_base, uint16_t sgpr_count) {
  const auto overlaps = [&](std::optional<uint16_t> reg, uint16_t width = 1u) {
    return reg && range_overlaps(sgpr_base, sgpr_count, *reg, width);
  };
  const bool backs_up_only_automatic_owner =
      options.automatic_moi_owner_sgpr && options.moi_owner_sgpr &&
      sgpr_base == *options.moi_owner_sgpr && sgpr_count == 1u;
  bool conflict = (options.moi_exec_save_sgprs_persistent && !backs_up_only_automatic_owner) ||
                  overlaps(options.moi_persistent_owner_sgpr) ||
                  overlaps(options.moi_persistent_epoch_sgpr) ||
                  overlaps(options.moi_persistent_workgroup_key_sgpr) ||
                  overlaps(options.moi_dispatch_id_sgpr, moi_dispatch_state_sgpr_count(options)) ||
                  overlaps(options.moi_inline_visible_evidence_sgpr);
  for (const std::optional<uint16_t> reg :
       {options.moi_record_replay_workgroup_sgprs.x, options.moi_record_replay_workgroup_sgprs.y,
        options.moi_record_replay_workgroup_sgprs.z}) {
    conflict |= overlaps(reg);
  }
  return !conflict;
}

void try_apply_owner_epoch_prologue_patch(std::span<const uint8_t> bytes,
                                          const ConSanOptions &options, rj_code_arch_t arch,
                                          ConSanResult &result) {
  if (!options.moi_init_owner_epoch)
    return;
  if (!instrumentation::is_admitted_arch(arch)) {
    result.warnings.emplace_back(
        "ConSan MOI owner/epoch prologue does not support this architecture");
    return;
  }
  if (options.automatic_moi_private_epoch) {
    try_apply_private_epoch_prologue_patch(options, arch, result);
    return;
  }
  if ((!options.moi_owner_vgpr || !options.moi_epoch_vgpr) &&
      options.moi_persistent_vgpr_assignments.empty() &&
      (!options.moi_persistent_owner_sgpr || !options.moi_persistent_epoch_sgpr)) {
    result.errors.emplace_back(
        "ConSan MOI owner/epoch prologue requires RJ_CONSAN_MOI_OWNER_VGPR and "
        "RJ_CONSAN_MOI_EPOCH_VGPR");
    return;
  }
  if (options.moi_owner_vgpr && options.moi_epoch_vgpr &&
      *options.moi_owner_vgpr == *options.moi_epoch_vgpr) {
    result.errors.emplace_back("ConSan MOI owner and epoch VGPRs must be distinct");
    return;
  }
  if (options.moi_owner_source == ConSanMoiOwnerSource::HwId && !options.moi_owner_sgpr) {
    result.errors.emplace_back("ConSan MOI hw_id owner source requires RJ_CONSAN_MOI_OWNER_SGPR");
    return;
  }
  if (result.kernels.empty()) {
    result.warnings.emplace_back("ConSan MOI owner/epoch prologue found no kernel descriptors");
    return;
  }

  std::span<const uint8_t> active_bytes = bytes;
  if (result.modified)
    active_bytes = std::span<const uint8_t>(result.elf_bytes.data(), result.elf_bytes.size());
  AmdGpuCodeObject code_object(active_bytes.data(), active_bytes.size());
  CodeObjectPatcher patcher(code_object);
  const std::span<const uint8_t> old_text = patcher.text_bytes();
  if (old_text.empty()) {
    result.errors.emplace_back("ConSan MOI owner/epoch prologue found no .text section");
    return;
  }
  std::unique_ptr<Decoder> relocation_decoder = Decoder::create(arch);
  if (!relocation_decoder) {
    result.errors.emplace_back("ConSan MOI owner/epoch prologue could not create decoder");
    return;
  }
  const uint32_t owner_required_sgpr_count =
      options.moi_owner_source == ConSanMoiOwnerSource::HwId
          ? static_cast<uint32_t>(*options.moi_owner_sgpr) + 1u
          : 0u;
  struct PlannedOwnerEpochPrologue {
    ConSanKernelInfo kernel;
    ConSanOptions persistent_options;
    std::optional<ConSanMoiDispatchIdPreloadPlan> dispatch_plan;
    std::optional<ConSanMoiWorkgroupShadowLayout> workgroup_shadow;
    std::optional<ConSanMoiWorkgroupSources> workgroup_sources;
    std::optional<MoiEntryScalarBackup> entry_scalar_backup;
    bool has_kernarg_preload = false;
  };
  std::vector<PlannedOwnerEpochPrologue> target_kernels;
  target_kernels.reserve(result.kernels.size());
  for (const ConSanKernelInfo &kernel : result.kernels) {
    if (!kernel.has_text_range)
      continue;
    const bool owns_emitted_patch =
        std::ranges::any_of(result.patches, [&](const ConSanPatchInfo &patch) {
          return kernel_owns_patch(kernel, patch);
        });
    const bool owns_planned_site =
        std::ranges::any_of(result.resource_plans, [&](const ConSanCandidateResourcePlan &plan) {
          return std::ranges::find(plan.owner_descriptor_file_offsets,
                                   kernel.descriptor_file_offset) !=
                     plan.owner_descriptor_file_offsets.end() &&
                 plan.source != ConSanRegisterAllocationSource::Unsupported;
        });
    // Automatic persistent state exists only to serve emitted instrumentation.
    // Resource planning deliberately inventories more sites than selection may
    // admit (for example under max_patches or a kernel filter). Mutating every
    // descriptor/entry that merely owns one of those speculative plans can
    // perturb otherwise untouched kernels and, on real multi-kernel modules,
    // can make execution hang before the selected probe is reached.
    if ((options.automatic_moi_persistent_vgprs || options.automatic_moi_persistent_sgprs) &&
        !owns_emitted_patch)
      continue;
    if (moi_has_runtime_hardware_dispatch_id(options) && !owns_emitted_patch &&
        !owns_planned_site) {
      continue;
    }
    const auto active_kernel =
        std::ranges::find_if(code_object.kernels(), [&](const AmdGpuKernelInfo &candidate) {
          return candidate.name == kernel.name;
        });
    if (active_kernel == code_object.kernels().end()) {
      result.warnings.emplace_back(
          "ConSan MOI owner/epoch prologue could not resolve an active kernel descriptor");
      continue;
    }
    ConSanKernelInfo active = kernel;
    active.descriptor_file_offset = active_kernel->descriptor_file_offset;
    active.entry_text_offset = active_kernel->entry_text_offset;
    active.text_file_offset = active_kernel->text_file_offset;
    ConSanOptions kernel_options = options;
    const std::array<uint64_t, 1> kernel_owner = {kernel.descriptor_file_offset};
    if (!options.moi_persistent_vgpr_assignments.empty() &&
        !apply_moi_persistent_vgpr_assignment(kernel_options, kernel_owner)) {
      result.errors.emplace_back(
          "ConSan MOI owner/epoch prologue has no persistent assignment for kernel '" +
          kernel.name + "'");
      return;
    }
    if (!options.moi_transient_sgpr_assignments.empty() &&
        !apply_moi_transient_sgpr_assignment(kernel_options, kernel_owner)) {
      result.errors.emplace_back(
          "ConSan MOI owner/epoch prologue has no transient scalar assignment for kernel '" +
          kernel.name + "'");
      return;
    }
    const ConSanMoiDispatchIdCapture dispatch_capture{.sgpr = kernel_options.moi_dispatch_id_sgpr,
                                                      .vgpr = kernel_options.moi_dispatch_id_vgpr};
    if (dispatch_capture.present() && !dispatch_capture.unambiguous()) {
      result.errors.emplace_back("ConSan MOI owner/epoch prologue has multiple dispatch-ID "
                                 "representations for kernel '" +
                                 kernel.name + "'");
      return;
    }
    if (options.automatic_moi_persistent_sgprs) {
      const auto *scratch =
          find_moi_prologue_scratch_vgpr_assignment(options, kernel.descriptor_file_offset);
      if (!scratch) {
        result.errors.emplace_back(
            "ConSan MOI scalar owner/epoch prologue has no entry-local VGPR scratch for kernel '" +
            kernel.name + "'");
        return;
      }
      kernel_options.moi_owner_vgpr = scratch->scratch_vgpr;
      kernel_options.moi_epoch_vgpr = static_cast<uint16_t>(scratch->scratch_vgpr + 1u);
    }
    if (active.descriptor_file_offset > active_bytes.size() ||
        sizeof(KD) > active_bytes.size() - active.descriptor_file_offset) {
      if (dispatch_capture.present()) {
        rollback_moi_dispatch_id_as_unsupported(result, "owner descriptor exceeds ELF bytes");
        return;
      }
      result.errors.emplace_back("ConSan MOI owner/epoch descriptor exceeds ELF bytes");
      return;
    }
    KD descriptor{};
    std::memcpy(&descriptor, active_bytes.data() + active.descriptor_file_offset,
                sizeof(descriptor));
    std::optional<ConSanMoiDispatchIdPreloadPlan> dispatch_plan;
    if (dispatch_capture.present()) {
      dispatch_plan = moi_descriptor_dispatch_id_preload_plan(descriptor, arch);
      if (!dispatch_plan->supported() ||
          (dispatch_capture.sgpr && *dispatch_capture.sgpr < dispatch_plan->required_sgpr_count)) {
        rollback_moi_dispatch_id_as_unsupported(
            result,
            dispatch_id_preload_rejection_reason(kernel.name, *dispatch_plan, dispatch_capture));
        return;
      }
    }
    const bool has_kernarg_preload = moi_descriptor_has_kernarg_preload(descriptor);
    if (has_kernarg_preload && kernel.code_size <= kAmdhsaKernelEntryAlignment) {
      if (dispatch_capture.present()) {
        rollback_moi_dispatch_id_as_unsupported(
            result, "kernel '" + kernel.name + "' has no valid secondary hardware entry");
      } else {
        result.errors.emplace_back(
            "ConSan MOI kernarg-preload kernel has no valid secondary hardware entry");
      }
      return;
    }
    std::optional<ConSanMoiWorkgroupShadowLayout> workgroup_shadow;
    const auto workitem_id_dimensions = moi_descriptor_workitem_id_dimensions(descriptor);
    if (!workitem_id_dimensions) {
      result.errors.emplace_back(
          "ConSan MOI owner/epoch prologue has invalid workitem-ID dimensions");
      return;
    }
    for (const ConSanPatchInfo &patch : result.patches) {
      if (!kernel_owns_patch(kernel, patch))
        continue;
      if (patch.workgroup_shadow_size == 0u)
        continue;
      const ConSanMoiWorkgroupShadowLayout candidate{
          .base = patch.workgroup_shadow_base,
          .size = patch.workgroup_shadow_size,
          .validity_base = patch.workgroup_shadow_validity_base,
          .validity_size = patch.workgroup_shadow_validity_size,
          .required_group_segment_size = patch.required_group_segment_size,
          .workitem_id_dimensions = *workitem_id_dimensions,
          .initialization_lanes =
              moi_workgroup_shadow_initialization_lanes(active.required_workgroup_size, arch),
          .lazy_initialization = patch.workgroup_shadow_lazy_initialization,
          .compact = patch.workgroup_shadow_compact,
          .visible_evidence_sgpr = std::nullopt,
      };
      if (workgroup_shadow &&
          (workgroup_shadow->base != candidate.base || workgroup_shadow->size != candidate.size ||
           workgroup_shadow->validity_base != candidate.validity_base ||
           workgroup_shadow->validity_size != candidate.validity_size ||
           workgroup_shadow->required_group_segment_size != candidate.required_group_segment_size ||
           workgroup_shadow->workitem_id_dimensions != candidate.workitem_id_dimensions ||
           workgroup_shadow->initialization_lanes != candidate.initialization_lanes ||
           workgroup_shadow->lazy_initialization != candidate.lazy_initialization ||
           workgroup_shadow->compact != candidate.compact)) {
        result.errors.emplace_back(
            "ConSan MOI owner/epoch prologue found incompatible workgroup-shadow layouts");
        return;
      }
      workgroup_shadow = candidate;
    }
    const bool owns_inline_barrier =
        std::ranges::any_of(result.patches, [&](const ConSanPatchInfo &patch) {
          return kernel_owns_patch(kernel, patch) &&
                 patch.kind == ConSanPatchKind::TrampolineMoiInlineEpochBarrier;
        });
    const bool owns_non_barrier_instrumentation =
        std::ranges::any_of(result.patches, [&](const ConSanPatchInfo &patch) {
          return kernel_owns_patch(kernel, patch) &&
                 patch.kind != ConSanPatchKind::TrampolineMoiInlineEpochBarrier &&
                 patch.kind != ConSanPatchKind::TrampolineMoiIndirectBranchIsland;
        });
    if (options.moi_engine == ConSanMoiEngine::InlineShadow &&
        options.automatic_moi_persistent_vgprs && owns_inline_barrier &&
        !owns_non_barrier_instrumentation) {
      result.warnings.emplace_back(
          "ConSan MOI skipped unobserved owner/epoch initialization for a barrier-only kernel");
      continue;
    }
    std::optional<ConSanMoiWorkgroupSources> workgroup_sources;
    const bool has_record_replay_workgroup_tuple =
        !kernel_options.moi_record_replay_workgroup_vgprs.empty() ||
        !kernel_options.moi_record_replay_workgroup_sgprs.empty();
    if (kernel_options.moi_workgroup_key_vgpr || kernel_options.moi_persistent_workgroup_key_sgpr ||
        has_record_replay_workgroup_tuple) {
      std::vector<std::string> source_errors;
      std::optional<uint16_t> full_payload_user_sgpr_count;
      if (has_record_replay_workgroup_tuple &&
          (arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4)) {
        full_payload_user_sgpr_count =
            dispatch_plan ? dispatch_plan->expanded_user_sgpr_count
                          : static_cast<uint16_t>(moi_descriptor_user_sgpr_count(descriptor));
      }
      workgroup_sources = moi_descriptor_workgroup_sources(
          active_bytes, active.descriptor_file_offset, arch, source_errors,
          /*uses_gfx1250_cluster_workgroup_id=*/false, full_payload_user_sgpr_count);
      if (!workgroup_sources) {
        result.errors.insert(result.errors.end(), source_errors.begin(), source_errors.end());
        return;
      }
    }
    std::optional<MoiEntryScalarBackup> entry_scalar_backup;
    const bool needs_dynamic_stack_scalar_backup =
        kernel_options.moi_inline_dynamic_stack_borrowed_sgpr.has_value();
    const bool needs_automatic_owner_scalar_backup =
        kernel_options.automatic_moi_owner_sgpr &&
        kernel_options.moi_owner_source == ConSanMoiOwnerSource::HwId &&
        kernel_options.moi_owner_sgpr.has_value();
    if (needs_dynamic_stack_scalar_backup || needs_automatic_owner_scalar_backup) {
      if (!is_rdna4_family_arch(arch) || !kernel_options.moi_exec_save_sgpr ||
          !kernel_options.automatic_moi_inline_sgpr_spill ||
          !kernel_options.moi_dynamic_stack_spill) {
        if (needs_dynamic_stack_scalar_backup) {
          result.errors.emplace_back(
              "ConSan MOI dynamic-stack entry scalar backup has no valid Inline spill contract");
          return;
        }
      }
      const uint16_t sgpr_base = needs_dynamic_stack_scalar_backup
                                     ? *kernel_options.moi_exec_save_sgpr
                                     : *kernel_options.moi_owner_sgpr;
      const uint16_t sgpr_count =
          needs_dynamic_stack_scalar_backup ? moi_exec_save_sgpr_count(kernel_options, arch) : 1u;
      if (sgpr_count == 0u || !moi_entry_scalar_backup_preserves_persistent_outputs(
                                  kernel_options, sgpr_base, sgpr_count)) {
        result.errors.emplace_back(
            "ConSan MOI entry scalar backup overlaps persistent output state");
        return;
      }
      if (kernel.descriptor_file_offset > bytes.size() ||
          sizeof(KD) > bytes.size() - kernel.descriptor_file_offset) {
        result.errors.emplace_back(
            "ConSan MOI entry scalar backup descriptor exceeds original ELF bytes");
        return;
      }
      KD original_descriptor{};
      std::memcpy(&original_descriptor, bytes.data() + kernel.descriptor_file_offset,
                  sizeof(original_descriptor));
      if (sgpr_count > moi_descriptor_wavefront_size(original_descriptor)) {
        result.outcome = ConSanTransformOutcome::Unsupported;
        result.warnings.emplace_back(
            "ConSan MOI entry scalar window exceeds one wave-local VGPR carrier");
        return;
      }

      const uint32_t persistent_end =
          moi_owner_epoch_prologue_required_vgpr_count(kernel_options, workgroup_shadow, arch);
      const uint32_t entry_abi_vgpr_count = *workitem_id_dimensions;
      const uint32_t entry_dead_floor = std::max(persistent_end, entry_abi_vgpr_count);
      const uint32_t original_allocation =
          moi_descriptor_vgpr_allocation_count(original_descriptor, arch);
      uint32_t backup_vgpr = std::max(original_allocation, entry_dead_floor);
      if (backup_vgpr >= kMaxVgprs) {
        // Only the enabled workitem-ID tuple is defined at hardware entry.
        // When the original descriptor already allocates the full vector
        // file, a non-ABI VGPR is still valid entry-local storage: its value
        // is dead before the first guest instruction and no persistent
        // instrumentation state uses it.
        backup_vgpr = entry_dead_floor;
      }
      if (backup_vgpr >= kMaxVgprs) {
        result.outcome = ConSanTransformOutcome::Unsupported;
        result.warnings.emplace_back(
            "ConSan MOI entry scalar backup has no entry-local VGPR carrier");
        return;
      }
      entry_scalar_backup = MoiEntryScalarBackup{
          .vgpr = static_cast<uint16_t>(backup_vgpr),
          .sgpr_base = sgpr_base,
          .sgpr_count = sgpr_count,
      };
    }
    target_kernels.push_back({std::move(active), std::move(kernel_options), dispatch_plan,
                              workgroup_shadow, workgroup_sources, entry_scalar_backup,
                              has_kernarg_preload});
  }
  for (const PlannedOwnerEpochPrologue &item : target_kernels) {
    const ConSanOptions &kernel_options = item.persistent_options;
    uint32_t required_vgpr_count =
        moi_owner_epoch_prologue_required_vgpr_count(kernel_options, item.workgroup_shadow, arch);
    if (item.entry_scalar_backup)
      required_vgpr_count =
          std::max<uint32_t>(required_vgpr_count, item.entry_scalar_backup->vgpr + 1u);
    uint32_t required_sgpr_count = owner_required_sgpr_count;
    if (kernel_options.moi_dispatch_id_sgpr) {
      required_sgpr_count = std::max<uint32_t>(
          required_sgpr_count, static_cast<uint32_t>(*kernel_options.moi_dispatch_id_sgpr) +
                                   moi_dispatch_state_sgpr_count(kernel_options));
    }
    if (options.moi_persistent_owner_sgpr)
      required_sgpr_count = std::max<uint32_t>(
          required_sgpr_count, static_cast<uint32_t>(*options.moi_persistent_owner_sgpr) + 1u);
    if (options.moi_persistent_epoch_sgpr)
      required_sgpr_count = std::max<uint32_t>(
          required_sgpr_count, static_cast<uint32_t>(*options.moi_persistent_epoch_sgpr) + 1u);
    if (options.moi_persistent_workgroup_key_sgpr)
      required_sgpr_count = std::max<uint32_t>(
          required_sgpr_count,
          static_cast<uint32_t>(*options.moi_persistent_workgroup_key_sgpr) + 1u);
    for (const std::optional<uint16_t> reg : {kernel_options.moi_record_replay_workgroup_sgprs.x,
                                              kernel_options.moi_record_replay_workgroup_sgprs.y,
                                              kernel_options.moi_record_replay_workgroup_sgprs.z}) {
      if (reg)
        required_sgpr_count = std::max<uint32_t>(required_sgpr_count, *reg + 1u);
    }
    const ConSanMoiDispatchIdCapture dispatch_capture{.sgpr = kernel_options.moi_dispatch_id_sgpr,
                                                      .vgpr = kernel_options.moi_dispatch_id_vgpr};
    if (item.dispatch_plan && dispatch_capture.present() &&
        !apply_moi_dispatch_id_descriptor(patcher, item.kernel.descriptor_file_offset,
                                          *item.dispatch_plan, dispatch_capture, arch,
                                          result.errors)) {
      return;
    }
    if (!grow_moi_kernel_descriptor_registers(patcher, item.kernel, required_vgpr_count,
                                              required_sgpr_count, arch, result.errors)) {
      return;
    }
  }

  std::vector<uint8_t> new_text(old_text.begin(), old_text.end());
  std::vector<ConSanPatchInfo> patches;
  MoiLocalNopIslandAllocator entry_islands(old_text, result, arch, 7u);
  for (const PlannedOwnerEpochPrologue &item : target_kernels) {
    const ConSanKernelInfo &kernel = item.kernel;
    const ConSanOptions &kernel_options = item.persistent_options;
    const ConSanMoiDispatchIdCapture dispatch_capture{.sgpr = kernel_options.moi_dispatch_id_sgpr,
                                                      .vgpr = kernel_options.moi_dispatch_id_vgpr};
    std::optional<ConSanMoiWorkgroupShadowLayout> prologue_workgroup_shadow = item.workgroup_shadow;
    if (prologue_workgroup_shadow) {
      prologue_workgroup_shadow->visible_evidence_sgpr =
          inline_shadow_visible_evidence_sgpr(kernel_options);
    }

    // Keep the hardware entry stable and displace its first instruction
    // through the same local-island mechanism used by ordinary long-range
    // probes. Besides being required by dynamic-stack kernels, this avoids a
    // gfx12 runtime failure observed when a large instrumented module redirects
    // a descriptor entry across particular late .text placement boundaries.
    const bool requires_in_place_entry = kernel.uses_dynamic_stack.value_or(false);
    bool instrument_entry_in_place =
        requires_in_place_entry ||
        (options.moi_engine == ConSanMoiEngine::InlineShadow &&
         (options.automatic_moi_persistent_vgprs || options.automatic_moi_persistent_sgprs));
    if (instrument_entry_in_place && item.has_kernarg_preload) {
      if (options.automatic_moi_persistent_vgprs) {
        // The in-place relay currently represents one hardware entry. Retain
        // the paired descriptor-redirection path when kernarg preloading
        // exposes a second entry that must run identical initialization.
        instrument_entry_in_place = false;
      } else {
        result.errors.emplace_back(
            "ConSan MOI dynamic-stack owner/epoch entry trampoline does not yet support a "
            "secondary kernarg-preload entry");
        return;
      }
    }
    std::vector<uint32_t> displaced_entry_words;
    std::optional<uint64_t> entry_island_offset;
    bool indirect_at_entry = false;
    uint64_t prologue_return_target = kernel.entry_text_offset;
    const auto original_kernel =
        std::ranges::find(result.kernels, kernel.name, &ConSanKernelInfo::name);
    const auto composed_entry_patch =
        std::ranges::find_if(result.patches, [&](const ConSanPatchInfo &patch) {
          const bool chainable_instrumentation =
              patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore ||
              patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore ||
              patch.kind == ConSanPatchKind::TrampolineMoiSampledWatchpointStore ||
              patch.kind == ConSanPatchKind::TrampolineMoiAtomicRecord ||
              patch.kind == ConSanPatchKind::TrampolineMoiBarrierRecord ||
              patch.kind == ConSanPatchKind::TrampolineMoiFenceRecord;
          return chainable_instrumentation && patch.anchor_offset == kernel.entry_text_offset &&
                 patch.original_size != 0u && patch.trampoline_size != 0u &&
                 original_kernel != result.kernels.end() &&
                 kernel_owns_patch(*original_kernel, patch);
        });
    const bool chains_existing_entry_trampoline =
        instrument_entry_in_place && composed_entry_patch != result.patches.end();
    const uint32_t composed_entry_original_size =
        chains_existing_entry_trampoline ? composed_entry_patch->original_size : 0u;
    const std::optional<uint64_t> composed_branch_only_trampoline_offset =
        chains_existing_entry_trampoline && composed_entry_patch->branch_only_continuation
            ? std::optional<uint64_t>(composed_entry_patch->trampoline_offset)
            : std::nullopt;
    const std::vector<uint64_t> composed_branch_only_entry_relays =
        composed_branch_only_trampoline_offset
            ? composed_entry_patch->branch_only_entry_relay_offsets
            : std::vector<uint64_t>{};
    if (instrument_entry_in_place) {
      if (chains_existing_entry_trampoline) {
        // An earlier instrumentation phase already displaced the guest entry
        // instruction and owns the continuation back into the kernel. Chain
        // initialization directly into that generated body instead of trying
        // to relocate its PC-relative anchor branch as though it were guest
        // control flow.
        prologue_return_target = composed_entry_patch->trampoline_offset;
      } else {
        std::vector<std::string> relocation_errors;
        auto decoded = decode_relocatable_entry_instruction(
            old_text, kernel.entry_text_offset, arch, relocation_errors, relocation_decoder.get());
        if (!decoded) {
          if (requires_in_place_entry) {
            result.errors.insert(result.errors.end(), relocation_errors.begin(),
                                 relocation_errors.end());
            return;
          }
          // A non-dynamic kernel with a PC-relative first instruction cannot
          // safely displace that instruction. Retain descriptor redirection as
          // the compatibility fallback for this uncommon entry shape.
          instrument_entry_in_place = false;
        } else {
          displaced_entry_words = std::move(*decoded);
          prologue_return_target += displaced_entry_words.size() * sizeof(uint32_t);
        }
      }
    }

    append_nop_padding_to_alignment(new_text, kAmdhsaKernelEntryAlignment, arch);
    const uint64_t prologue_text_offset = static_cast<uint64_t>(new_text.size());
    if (instrument_entry_in_place &&
        !compute_sopp_branch_simm16(kernel.entry_text_offset, prologue_text_offset)) {
      entry_island_offset = entry_islands.claim_reachable(kernel.entry_text_offset);
      if (!entry_island_offset) {
        if (chains_existing_entry_trampoline) {
          result.errors.emplace_back(
              "ConSan MOI dynamic-stack owner/epoch prologue found no reachable entry relay "
              "without clobbering an existing entry instrumentation patch");
          return;
        }
        constexpr size_t kDirectEntryRelayWords = 7u;
        while (displaced_entry_words.size() < kDirectEntryRelayWords) {
          std::vector<std::string> relocation_errors;
          auto decoded = decode_relocatable_entry_instruction(
              old_text, kernel.entry_text_offset + displaced_entry_words.size() * sizeof(uint32_t),
              arch, relocation_errors, relocation_decoder.get());
          if (!decoded) {
            if (requires_in_place_entry) {
              result.errors.insert(result.errors.end(), relocation_errors.begin(),
                                   relocation_errors.end());
              return;
            }
            // The expanded in-place relay is only an optimization for ordinary
            // kernels. If a later entry instruction is PC-relative, leave the
            // hardware entry intact and use descriptor redirection instead.
            instrument_entry_in_place = false;
            displaced_entry_words.clear();
            prologue_return_target = kernel.entry_text_offset;
            break;
          }
          displaced_entry_words.insert(displaced_entry_words.end(), decoded->begin(),
                                       decoded->end());
        }
        if (instrument_entry_in_place) {
          indirect_at_entry = true;
          prologue_return_target =
              kernel.entry_text_offset + displaced_entry_words.size() * sizeof(uint32_t);
        }
      }
    }
    const std::optional<uint16_t> owner_shift =
        moi_kernel_owner_shift(active_bytes, kernel, result.errors);
    if (!owner_shift)
      return;
    auto words = build_owner_epoch_prologue_words(
        prologue_text_offset, prologue_return_target, *kernel_options.moi_owner_vgpr,
        *kernel_options.moi_epoch_vgpr, kernel_options.moi_workgroup_key_vgpr, *owner_shift,
        kernel_options.moi_owner_source, kernel_options.moi_owner_sgpr,
        kernel_options.moi_engine == ConSanMoiEngine::InlineShadow,
        kernel_options.moi_persistent_owner_sgpr, kernel_options.moi_persistent_epoch_sgpr,
        kernel_options.moi_persistent_workgroup_key_sgpr,
        kernel_options.moi_record_replay_workgroup_sgprs,
        kernel_options.moi_record_replay_workgroup_vgprs, item.dispatch_plan, dispatch_capture,
        moi_runtime_sample_counter_sgpr(kernel_options), kernel_options.moi_exec_save_sgpr,
        item.entry_scalar_backup, prologue_workgroup_shadow,
        options.automatic_moi_persistent_vgprs && arch == ROCJITSU_CODE_ARCH_GFX1250,
        item.workgroup_sources, displaced_entry_words, arch, result.errors);
    if (!words)
      return;
    if (instrument_entry_in_place) {
      size_t first_padding_word = 1u;
      if (indirect_at_entry) {
        const size_t patch_count_before_entry_relay = patches.size();
        if (!emit_moi_local_indirect_kernel_entry_island(
                new_text, kernel.entry_text_offset, prologue_text_offset, kernel.entry_text_offset,
                std::span<const uint64_t>(&kernel.descriptor_file_offset, 1u), arch, patches,
                result.errors, "ConSan MOI dynamic-stack owner/epoch prologue")) {
          return;
        }
        // The prologue record below owns this same entry anchor, so retain
        // only one authoritative range for final overlap validation.
        patches.resize(patch_count_before_entry_relay);
        first_padding_word = 7u;
      } else {
        const uint64_t entry_target = entry_island_offset.value_or(prologue_text_offset);
        const auto branch = compute_sopp_branch_simm16(kernel.entry_text_offset, entry_target);
        if (!branch ||
            !write_word_bytes(new_text, kernel.entry_text_offset, build_s_branch(*branch, arch))) {
          result.errors.emplace_back(
              "ConSan MOI owner/epoch prologue could not instrument the dynamic-stack entry");
          return;
        }
      }
      for (size_t word = first_padding_word; word < displaced_entry_words.size(); ++word) {
        if (!write_word_bytes(new_text, kernel.entry_text_offset + word * sizeof(uint32_t),
                              build_s_nop(0, arch))) {
          result.errors.emplace_back(
              "ConSan MOI owner/epoch prologue could not pad its displaced entry instruction");
          return;
        }
      }
      if (!indirect_at_entry && entry_island_offset &&
          !emit_moi_local_indirect_kernel_entry_island(
              new_text, *entry_island_offset, prologue_text_offset, kernel.entry_text_offset,
              std::span<const uint64_t>(&kernel.descriptor_file_offset, 1u), arch, patches,
              result.errors, "ConSan MOI dynamic-stack owner/epoch prologue")) {
        return;
      }
    } else if (!patcher.redirect_kernel_entry(kernel.descriptor_file_offset,
                                              kernel.entry_text_offset, prologue_text_offset)) {
      result.errors.emplace_back("ConSan MOI owner/epoch prologue could not redirect kernel entry");
      return;
    }
    std::optional<uint64_t> dispatch_id_primary_prologue_offset;
    std::optional<uint64_t> dispatch_id_secondary_prologue_offset;
    if (item.has_kernarg_preload) {
      const uint64_t secondary_prologue_offset = util::align_up(
          prologue_text_offset + words->size() * sizeof(uint32_t), kAmdhsaKernelEntryAlignment);
      auto secondary_words = build_owner_epoch_prologue_words(
          secondary_prologue_offset, kernel.entry_text_offset + kAmdhsaKernelEntryAlignment,
          *kernel_options.moi_owner_vgpr, *kernel_options.moi_epoch_vgpr,
          kernel_options.moi_workgroup_key_vgpr, *owner_shift, kernel_options.moi_owner_source,
          kernel_options.moi_owner_sgpr, kernel_options.moi_engine == ConSanMoiEngine::InlineShadow,
          kernel_options.moi_persistent_owner_sgpr, kernel_options.moi_persistent_epoch_sgpr,
          kernel_options.moi_persistent_workgroup_key_sgpr,
          kernel_options.moi_record_replay_workgroup_sgprs,
          kernel_options.moi_record_replay_workgroup_vgprs, item.dispatch_plan, dispatch_capture,
          moi_runtime_sample_counter_sgpr(kernel_options), kernel_options.moi_exec_save_sgpr,
          item.entry_scalar_backup, prologue_workgroup_shadow,
          options.automatic_moi_persistent_vgprs && arch == ROCJITSU_CODE_ARCH_GFX1250,
          item.workgroup_sources, std::span<const uint32_t>{}, arch, result.errors);
      if (!secondary_words)
        return;

      if (words->size() * sizeof(uint32_t) <= kAmdhsaKernelEntryAlignment &&
          secondary_words->size() * sizeof(uint32_t) <= kAmdhsaKernelEntryAlignment) {
        append_words_bytes(new_text, *words);
        append_nop_padding_to_alignment(new_text, kAmdhsaKernelEntryAlignment, arch);
        append_words_bytes(new_text, *secondary_words);
      } else {
        // A kernarg-preload descriptor exposes paired hardware entries exactly
        // 256 bytes apart. Keep those slots as short relays when either full
        // initialization body no longer fits, then place both bodies after the
        // paired-entry window. This preserves identical initialization at both
        // entries without imposing a 256-byte ceiling on future persistent
        // state.
        constexpr uint64_t kPairedEntryWindow = 2u * kAmdhsaKernelEntryAlignment;
        const uint64_t primary_body_offset = prologue_text_offset + kPairedEntryWindow;
        words = build_owner_epoch_prologue_words(
            primary_body_offset, prologue_return_target, *kernel_options.moi_owner_vgpr,
            *kernel_options.moi_epoch_vgpr, kernel_options.moi_workgroup_key_vgpr, *owner_shift,
            kernel_options.moi_owner_source, kernel_options.moi_owner_sgpr,
            kernel_options.moi_engine == ConSanMoiEngine::InlineShadow,
            kernel_options.moi_persistent_owner_sgpr, kernel_options.moi_persistent_epoch_sgpr,
            kernel_options.moi_persistent_workgroup_key_sgpr,
            kernel_options.moi_record_replay_workgroup_sgprs,
            kernel_options.moi_record_replay_workgroup_vgprs, item.dispatch_plan, dispatch_capture,
            moi_runtime_sample_counter_sgpr(kernel_options), kernel_options.moi_exec_save_sgpr,
            item.entry_scalar_backup, prologue_workgroup_shadow,
            options.automatic_moi_persistent_vgprs && arch == ROCJITSU_CODE_ARCH_GFX1250,
            item.workgroup_sources, displaced_entry_words, arch, result.errors);
        if (!words)
          return;
        const uint64_t secondary_body_offset =
            primary_body_offset + words->size() * sizeof(uint32_t);
        secondary_words = build_owner_epoch_prologue_words(
            secondary_body_offset, kernel.entry_text_offset + kAmdhsaKernelEntryAlignment,
            *kernel_options.moi_owner_vgpr, *kernel_options.moi_epoch_vgpr,
            kernel_options.moi_workgroup_key_vgpr, *owner_shift, kernel_options.moi_owner_source,
            kernel_options.moi_owner_sgpr,
            kernel_options.moi_engine == ConSanMoiEngine::InlineShadow,
            kernel_options.moi_persistent_owner_sgpr, kernel_options.moi_persistent_epoch_sgpr,
            kernel_options.moi_persistent_workgroup_key_sgpr,
            kernel_options.moi_record_replay_workgroup_sgprs,
            kernel_options.moi_record_replay_workgroup_vgprs, item.dispatch_plan, dispatch_capture,
            moi_runtime_sample_counter_sgpr(kernel_options), kernel_options.moi_exec_save_sgpr,
            item.entry_scalar_backup, prologue_workgroup_shadow,
            options.automatic_moi_persistent_vgprs && arch == ROCJITSU_CODE_ARCH_GFX1250,
            item.workgroup_sources, std::span<const uint32_t>{}, arch, result.errors);
        const auto primary_relay =
            compute_sopp_branch_simm16(prologue_text_offset, primary_body_offset);
        const auto secondary_relay = compute_sopp_branch_simm16(
            prologue_text_offset + kAmdhsaKernelEntryAlignment, secondary_body_offset);
        if (!secondary_words || !primary_relay || !secondary_relay) {
          result.errors.emplace_back(
              "ConSan MOI could not encode its kernarg-preload owner prologue relays");
          return;
        }
        std::vector<uint32_t> relays(kPairedEntryWindow / sizeof(uint32_t), build_s_nop(0, arch));
        relays.front() = build_s_branch(*primary_relay, arch);
        relays[kAmdhsaKernelEntryAlignment / sizeof(uint32_t)] =
            build_s_branch(*secondary_relay, arch);
        append_words_bytes(new_text, relays);
        append_words_bytes(new_text, *words);
        append_words_bytes(new_text, *secondary_words);
        dispatch_id_primary_prologue_offset = primary_body_offset;
        dispatch_id_secondary_prologue_offset = secondary_body_offset;
      }
    } else {
      append_words_bytes(new_text, *words);
    }

    ConSanPatchInfo info;
    info.kind = ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue;
    info.anchor_offset = kernel.entry_text_offset;
    info.trampoline_offset = prologue_text_offset;
    info.original_size =
        instrument_entry_in_place
            ? (chains_existing_entry_trampoline ? composed_entry_original_size
                                                : displaced_entry_words.size() * sizeof(uint32_t))
            : 0u;
    info.trampoline_size = static_cast<uint32_t>(new_text.size() - prologue_text_offset);
    info.entry_prologue_chained_trampoline_offset = composed_branch_only_trampoline_offset;
    info.dispatch_id_primary_prologue_offset = dispatch_id_primary_prologue_offset;
    info.dispatch_id_secondary_prologue_offset = dispatch_id_secondary_prologue_offset;
    info.persistent_owner_vgpr = kernel_options.moi_owner_vgpr;
    info.persistent_epoch_vgpr = kernel_options.moi_epoch_vgpr;
    info.persistent_workgroup_key_vgpr = kernel_options.moi_workgroup_key_vgpr;
    info.persistent_record_replay_workgroup_vgprs =
        kernel_options.moi_record_replay_workgroup_vgprs;
    if (item.entry_scalar_backup) {
      info.entry_scalar_backup_vgpr = item.entry_scalar_backup->vgpr;
      info.entry_scalar_backup_sgpr_base = item.entry_scalar_backup->sgpr_base;
      info.entry_scalar_backup_sgpr_count = item.entry_scalar_backup->sgpr_count;
    }
    if (item.workgroup_shadow) {
      info.workgroup_shadow_base = item.workgroup_shadow->base;
      info.workgroup_shadow_size = item.workgroup_shadow->size;
      info.workgroup_shadow_validity_base = item.workgroup_shadow->validity_base;
      info.workgroup_shadow_validity_size = item.workgroup_shadow->validity_size;
      info.workgroup_shadow_lazy_initialization = item.workgroup_shadow->lazy_initialization;
      info.workgroup_shadow_compact = item.workgroup_shadow->compact;
      info.required_group_segment_size = item.workgroup_shadow->required_group_segment_size;
    }
    if (item.dispatch_plan && dispatch_capture.present())
      note_dispatch_id_patch_info(info, *item.dispatch_plan, dispatch_capture);
    const auto canonical_kernel =
        std::ranges::find(result.kernels, kernel.name, &ConSanKernelInfo::name);
    if (canonical_kernel == result.kernels.end()) {
      result.errors.emplace_back("ConSan MOI owner/epoch prologue lost its canonical kernel owner");
      return;
    }
    info.owner_descriptor_file_offsets.push_back(canonical_kernel->descriptor_file_offset);
    if (composed_branch_only_trampoline_offset) {
      for (uint64_t relay : composed_branch_only_entry_relays) {
        if (!write_word_bytes(new_text, relay, build_s_nop(0, arch))) {
          result.errors.emplace_back(
              "ConSan MOI owner/epoch prologue could not retire a superseded entry relay");
          return;
        }
      }
      composed_entry_patch->branch_only_entry_relay_offsets.clear();
      composed_entry_patch->branch_only_entry_prologue_offset = prologue_text_offset;
      // The erase below invalidates composed_entry_patch. Keep it last in this
      // composition block so later maintenance cannot accidentally reuse it.
      std::erase_if(result.patches, [&](const ConSanPatchInfo &patch) {
        return patch.kind == ConSanPatchKind::TrampolineNopBranchRelay &&
               std::ranges::find(composed_branch_only_entry_relays, patch.anchor_offset) !=
                   composed_branch_only_entry_relays.end();
      });
    }
    patches.push_back(info);
  }

  if (patches.empty()) {
    result.warnings.emplace_back("ConSan MOI owner/epoch prologue found no patchable kernels");
    return;
  }
  if (!replace_consan_text(patcher, new_text, options, "MOI owner/epoch prologue", result)) {
    return;
  }

  result.elf_bytes = std::move(patcher).emit();
  result.patches.insert(result.patches.end(), patches.begin(), patches.end());
  result.modified = true;
}
