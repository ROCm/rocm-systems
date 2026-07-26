struct MoiPrivateEpochLayout {
  uint32_t epoch_offset = 0;
  std::optional<uint32_t> owner_offset;
  std::optional<uint32_t> workgroup_key_offset;
  uint32_t ephemeral_base = 0;
};

[[nodiscard]] std::optional<uint8_t> moi_descriptor_workitem_id_dimensions(const KD &descriptor) {
  const uint32_t encoded =
      AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_VGPR_WORKITEM_ID);
  if (encoded > 2u)
    return std::nullopt;
  return static_cast<uint8_t>(encoded + 1u);
}

[[nodiscard]] std::optional<uint16_t>
common_moi_private_owner_shift(std::span<const uint8_t> image,
                               const ResolvedMoiScratchPlan &resources,
                               std::vector<std::string> &warnings) {
  std::optional<uint16_t> common_shift;
  for (uint64_t descriptor_offset : resources.owner_descriptor_file_offsets) {
    std::vector<std::string> errors;
    const auto shift = moi_descriptor_owner_shift(image, descriptor_offset, errors);
    if (!shift) {
      warnings.insert(warnings.end(), errors.begin(), errors.end());
      return std::nullopt;
    }
    if (common_shift && *common_shift != *shift) {
      warnings.emplace_back("ConSan MOI shared private owner has incompatible owner wave sizes");
      return std::nullopt;
    }
    common_shift = shift;
  }
  if (!common_shift)
    warnings.emplace_back("ConSan MOI private owner requires an owning kernel descriptor");
  return common_shift;
}

struct MoiPrivateRecordOwnerPlan {
  uint32_t offset = 0;
  uint16_t shift = 0;
  uint32_t private_state_end = 0;
};

[[nodiscard]] bool moi_record_uses_private_owner(const ConSanOptions &options) {
  return options.moi_init_owner_epoch && options.automatic_moi_private_epoch &&
         options.moi_owner_source == ConSanMoiOwnerSource::WorkitemId && !options.moi_owner_vgpr &&
         !options.moi_persistent_owner_sgpr;
}

[[nodiscard]] std::optional<MoiPrivateRecordOwnerPlan>
resolve_moi_private_record_owner(std::span<const uint8_t> image, const ConSanResult &result,
                                 const ResolvedMoiScratchPlan &resources,
                                 std::vector<std::string> &warnings) {
  std::optional<uint32_t> common_offset;
  std::optional<uint32_t> common_state_end;
  for (uint64_t descriptor_offset : resources.owner_descriptor_file_offsets) {
    const auto owner_patch =
        std::ranges::find_if(result.patches, [&](const ConSanPatchInfo &patch) {
          return patch.persistent_owner_private_offset && patch.persistent_private_state_end &&
                 std::ranges::find(patch.owner_descriptor_file_offsets, descriptor_offset) !=
                     patch.owner_descriptor_file_offsets.end();
        });
    if (owner_patch == result.patches.end()) {
      warnings.emplace_back(
          "ConSan MOI private owner has no captured state for an owning kernel descriptor");
      return std::nullopt;
    }
    if (common_offset && *common_offset != *owner_patch->persistent_owner_private_offset) {
      warnings.emplace_back("ConSan MOI shared private owner has incompatible state offsets");
      return std::nullopt;
    }
    if (common_state_end && *common_state_end != *owner_patch->persistent_private_state_end) {
      warnings.emplace_back("ConSan MOI shared private owner has incompatible state sizes");
      return std::nullopt;
    }
    common_offset = owner_patch->persistent_owner_private_offset;
    common_state_end = owner_patch->persistent_private_state_end;
  }
  if (!common_offset || !common_state_end) {
    warnings.emplace_back("ConSan MOI private owner requires an owning kernel descriptor");
    return std::nullopt;
  }
  const auto common_shift = common_moi_private_owner_shift(image, resources, warnings);
  if (!common_shift)
    return std::nullopt;
  return MoiPrivateRecordOwnerPlan{
      .offset = *common_offset, .shift = *common_shift, .private_state_end = *common_state_end};
}

struct MoiRecordOwnerDerivation {
  uint16_t vgpr = 0;
  std::vector<uint32_t> words;
};

[[nodiscard]] std::optional<MoiRecordOwnerDerivation> build_moi_record_owner_derivation(
    std::span<const uint8_t> bytes, std::optional<uint64_t> descriptor_file_offset,
    std::optional<uint32_t> private_owner_offset, std::optional<uint16_t> private_owner_shift,
    uint16_t value_vgpr, rj_code_arch_t arch, std::string_view consumer,
    std::vector<std::string> &errors) {
  if (private_owner_offset.has_value() != private_owner_shift.has_value()) {
    errors.emplace_back("ConSan MOI " + std::string(consumer) +
                        " has an incomplete private owner plan");
    return std::nullopt;
  }
  std::optional<uint16_t> owner_shift = private_owner_shift;
  if (!owner_shift) {
    if (!descriptor_file_offset) {
      errors.emplace_back("ConSan MOI " + std::string(consumer) +
                          " requires an owning kernel descriptor");
      return std::nullopt;
    }
    owner_shift = moi_descriptor_owner_shift(bytes, *descriptor_file_offset, errors);
    if (!owner_shift)
      return std::nullopt;
  }

  MoiRecordOwnerDerivation derivation{.vgpr = value_vgpr, .words = {}};
  if (private_owner_offset) {
    // Guest code may repurpose v0 after kernel entry, so reload the raw
    // workitem ID captured by the private-state prologue before deriving the wave owner.
    const auto owner_load =
        instrumentation::build_private_load_b32(value_vgpr, *private_owner_offset, arch);
    const auto owner_wait = instrumentation::build_s_wait_private_load0(arch);
    if (!owner_load || !owner_wait) {
      errors.emplace_back("ConSan MOI " + std::string(consumer) +
                          " could not load its captured entry owner");
      return std::nullopt;
    }
    derivation.words.insert(derivation.words.end(), owner_load->begin(), owner_load->end());
    derivation.words.push_back(*owner_wait);
  }
  const uint16_t owner_source = private_owner_offset ? value_vgpr : kRdna4WorkitemIdX;
  const auto owner_init = instrumentation::build_v_lshrrev_b32(
      value_vgpr, scalar_positive_inline_u32(*owner_shift), owner_source, arch);
  if (!owner_init) {
    errors.emplace_back("ConSan MOI " + std::string(consumer) +
                        " could not encode owner derivation");
    return std::nullopt;
  }
  derivation.words.push_back(*owner_init);
  return derivation;
}

[[nodiscard]] bool same_workgroup_source(const ConSanMoiWorkgroupSource &lhs,
                                         const ConSanMoiWorkgroupSource &rhs) {
  return lhs.scalar_src == rhs.scalar_src && lhs.shift_right_16 == rhs.shift_right_16 &&
         lhs.mask_low_16 == rhs.mask_low_16;
}

[[nodiscard]] bool same_workgroup_sources(const ConSanMoiWorkgroupSources &lhs,
                                          const ConSanMoiWorkgroupSources &rhs) {
  return same_workgroup_source(lhs.x, rhs.x) && same_workgroup_source(lhs.y, rhs.y) &&
         same_workgroup_source(lhs.z, rhs.z) &&
         same_workgroup_source(lhs.cluster_workgroup_id, rhs.cluster_workgroup_id);
}

[[nodiscard]] std::optional<uint64_t>
common_moi_record_owner_descriptor(std::span<const uint8_t> image,
                                   const ResolvedMoiScratchPlan &resources, rj_code_arch_t arch,
                                   std::vector<std::string> &warnings) {
  std::optional<uint16_t> common_shift;
  std::optional<ConSanMoiWorkgroupSources> common_workgroup_sources;
  std::optional<uint64_t> representative;
  for (uint64_t descriptor_offset : resources.owner_descriptor_file_offsets) {
    std::vector<std::string> errors;
    const auto shift = moi_descriptor_owner_shift(image, descriptor_offset, errors);
    const auto workgroup_sources =
        moi_descriptor_workgroup_sources(image, descriptor_offset, arch, errors);
    if (!shift || !workgroup_sources) {
      warnings.insert(warnings.end(), errors.begin(), errors.end());
      return std::nullopt;
    }
    if ((common_shift && *common_shift != *shift) ||
        (common_workgroup_sources &&
         !same_workgroup_sources(*common_workgroup_sources, *workgroup_sources))) {
      warnings.emplace_back(
          "ConSan MOI shared record owner has incompatible descriptor ABI inputs");
      return std::nullopt;
    }
    common_shift = shift;
    common_workgroup_sources = workgroup_sources;
    if (!representative)
      representative = descriptor_offset;
  }
  if (!representative)
    warnings.emplace_back("ConSan MOI shared record owner has no owning descriptor");
  return representative;
}

[[nodiscard]] std::optional<ConSanMoiWorkgroupShadowLayout> common_moi_workgroup_shadow_layout(
    std::span<const uint8_t> image, const ResolvedMoiScratchPlan &resources, rj_code_arch_t arch,
    const ConSanOptions &options, std::vector<std::string> &warnings, bool prefer_compact = false) {
  std::optional<ConSanMoiWorkgroupShadowLayout> common;
  const uint32_t max_workgroup_lds_bytes = consan_moi_max_workgroup_lds_bytes(options, arch);
  for (uint64_t descriptor_offset : resources.owner_descriptor_file_offsets) {
    if (descriptor_offset > image.size() || sizeof(KD) > image.size() - descriptor_offset) {
      warnings.emplace_back("ConSan MOI workgroup shadow descriptor exceeds ELF bytes");
      return std::nullopt;
    }
    KD descriptor{};
    std::memcpy(&descriptor, image.data() + descriptor_offset, sizeof(descriptor));
    auto layout = plan_consan_moi_workgroup_shadow(descriptor.group_segment_fixed_size,
                                                   max_workgroup_lds_bytes);
    if (!layout) {
      warnings.emplace_back(
          "ConSan MOI workgroup shadow requires fixed LDS that fits with its exact mirror");
      return std::nullopt;
    }
    // RDNA4 LDS contents are not guaranteed to begin zeroed. Qualify every
    // local cell with a dispatch-scoped workgroup generation rather than
    // depending on a large entry-prologue clear. Atomic token tables retain
    // the persistent coordinate key separately, so synchronized local cells
    // no longer need to share their bounded generation with global metadata.
    if (arch == ROCJITSU_CODE_ARCH_RDNA4) {
      layout = plan_consan_moi_generation_tagged_workgroup_shadow(
          descriptor.group_segment_fixed_size, max_workgroup_lds_bytes);
    }
    // A near-capacity exact mirror is still useful when either its cells are
    // compacted or its eager clear is replaced by a validity bitmap. Preserve
    // exact local identity and select the representation requested by the
    // object-wide planner.
    constexpr uint32_t kGfx1250PreferredLocalShadowLimit = 48u * 1024u;
    if (arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4) {
      // LDS is reusable storage whose initial contents are unspecified. Make
      // CDNA validity explicit in each exact cell instead of depending on a
      // workgroup-wide eager clear before the instrumented program starts.
      layout = plan_consan_moi_generation_tagged_workgroup_shadow(
          descriptor.group_segment_fixed_size, max_workgroup_lds_bytes);
      if (!layout) {
        warnings.emplace_back("ConSan MOI CDNA generation-tagged workgroup shadow does not fit in "
                              "architectural LDS");
        return std::nullopt;
      }
    } else if (arch == ROCJITSU_CODE_ARCH_GFX1250 &&
               layout->required_group_segment_size > kGfx1250PreferredLocalShadowLimit) {
      layout = consan_moi_prefer_generation_tagged_workgroup_shadow(
                   descriptor.group_segment_fixed_size, prefer_compact)
                   ? plan_consan_moi_generation_tagged_workgroup_shadow(
                         descriptor.group_segment_fixed_size, max_workgroup_lds_bytes)
                   : plan_consan_moi_compact_workgroup_shadow(descriptor.group_segment_fixed_size,
                                                              max_workgroup_lds_bytes);
      if (!layout) {
        warnings.emplace_back(
            "ConSan MOI gfx1250 generation-tagged/compact workgroup shadow does not fit in "
            "architectural LDS");
        return std::nullopt;
      }
    }
    const auto workitem_id_dimensions = moi_descriptor_workitem_id_dimensions(descriptor);
    if (!workitem_id_dimensions) {
      warnings.emplace_back("ConSan MOI workgroup shadow has invalid workitem-ID dimensions");
      return std::nullopt;
    }
    layout->workitem_id_dimensions = *workitem_id_dimensions;
    if (common && (common->base != layout->base || common->size != layout->size ||
                   common->validity_base != layout->validity_base ||
                   common->validity_size != layout->validity_size ||
                   common->lazy_initialization != layout->lazy_initialization ||
                   common->compact != layout->compact ||
                   common->workitem_id_dimensions != layout->workitem_id_dimensions)) {
      warnings.emplace_back(
          "ConSan MOI shared access text has incompatible owning-kernel LDS layouts or IDs");
      return std::nullopt;
    }
    common = layout;
  }
  if (!common)
    warnings.emplace_back("ConSan MOI workgroup shadow has no owning kernel descriptor");
  return common;
}

[[nodiscard]] std::optional<MoiPrivateEpochLayout>
build_moi_private_epoch_layout(const ConSanResult &result, const ResolvedMoiScratchPlan &resources,
                               rj_code_arch_t arch, std::vector<std::string> &warnings,
                               bool include_owner = false, bool include_workgroup_key = false) {
  if (resources.owner_descriptor_file_offsets.empty()) {
    warnings.emplace_back("ConSan MOI private epoch requires an owning kernel descriptor");
    return std::nullopt;
  }
  for (uint64_t descriptor_offset : resources.owner_descriptor_file_offsets) {
    const ConSanKernelInfo *kernel = kernel_for_descriptor(result, descriptor_offset);
    if (kernel == nullptr) {
      warnings.emplace_back("ConSan MOI private epoch references an unknown kernel descriptor");
      return std::nullopt;
    }
    if (kernel->uses_dynamic_stack.value_or(false)) {
      warnings.emplace_back(
          "ConSan MOI private epoch does not support a dynamic-stack owning kernel");
      return std::nullopt;
    }
  }
  const uint64_t epoch_offset =
      util::align_up(resources.original_private_segment_size, SpillManager::kDbiZoneAlignment);
  uint64_t persistent_end = epoch_offset + SpillManager::kSlotBytes;
  const std::optional<uint64_t> owner_offset =
      include_owner ? std::optional<uint64_t>(persistent_end) : std::nullopt;
  if (owner_offset)
    persistent_end += SpillManager::kSlotBytes;
  const std::optional<uint64_t> workgroup_key_offset =
      include_workgroup_key ? std::optional<uint64_t>(persistent_end) : std::nullopt;
  if (workgroup_key_offset)
    persistent_end += SpillManager::kSlotBytes;
  const auto private_limit = address_free_scratch_private_limit(arch);
  if (!private_limit) {
    warnings.emplace_back(
        "ConSan MOI private epoch has no address-free scratch support for this architecture");
    return std::nullopt;
  }
  if (persistent_end > std::numeric_limits<uint32_t>::max()) {
    warnings.emplace_back("ConSan MOI private epoch exceeds address-free scratch capacity");
    return std::nullopt;
  }
  const auto ephemeral_base =
      normalize_address_free_scratch_private_size(arch, static_cast<uint32_t>(persistent_end));
  if (!ephemeral_base) {
    warnings.emplace_back("ConSan MOI private epoch exceeds address-free scratch capacity");
    return std::nullopt;
  }
  return MoiPrivateEpochLayout{
      .epoch_offset = static_cast<uint32_t>(epoch_offset),
      .owner_offset = owner_offset ? std::optional<uint32_t>(static_cast<uint32_t>(*owner_offset))
                                   : std::nullopt,
      .workgroup_key_offset =
          workgroup_key_offset
              ? std::optional<uint32_t>(static_cast<uint32_t>(*workgroup_key_offset))
              : std::nullopt,
      .ephemeral_base = *ephemeral_base};
}

[[nodiscard]] std::optional<MoiPrivateEpochLayout> build_sampled_private_epoch_layout(
    const ConSanResult &result, const ResolvedMoiScratchPlan &resources,
    const ConSanOptions &options, rj_code_arch_t arch, std::vector<std::string> &warnings) {
  return build_moi_private_epoch_layout(result, resources, arch, warnings,
                                        /*include_owner=*/options.moi_owner_source ==
                                            ConSanMoiOwnerSource::WorkitemId);
}

[[nodiscard]] bool append_sampled_private_owner_epoch_load(
    std::vector<uint32_t> &words, std::span<const uint8_t> bytes, uint64_t descriptor_file_offset,
    const ConSanOptions &options, const MoiPrivateEpochLayout &layout, rj_code_arch_t arch,
    std::vector<std::string> &errors) {
  if (!options.automatic_moi_private_epoch || !options.moi_owner_vgpr || !options.moi_epoch_vgpr ||
      !layout.owner_offset) {
    errors.emplace_back("ConSan MOI sampled sync has an invalid private owner/epoch plan");
    return false;
  }
  const auto owner_shift = moi_descriptor_owner_shift(bytes, descriptor_file_offset, errors);
  const auto load_epoch =
      instrumentation::build_private_load_b32(*options.moi_epoch_vgpr, layout.epoch_offset, arch);
  const auto load_owner =
      instrumentation::build_private_load_b32(*options.moi_owner_vgpr, *layout.owner_offset, arch);
  const auto wait = instrumentation::build_s_wait_private_load0(arch);
  const auto owner =
      owner_shift ? instrumentation::build_v_lshrrev_b32(*options.moi_owner_vgpr,
                                                         scalar_positive_inline_u32(*owner_shift),
                                                         *options.moi_owner_vgpr, arch)
                  : std::nullopt;
  if (!owner_shift || !load_epoch || !load_owner || !wait || !owner) {
    errors.emplace_back("ConSan MOI sampled sync could not load private owner/epoch state");
    return false;
  }
  words.insert(words.end(), load_epoch->begin(), load_epoch->end());
  words.insert(words.end(), load_owner->begin(), load_owner->end());
  words.push_back(*wait);
  words.push_back(*owner);
  return true;
}

[[nodiscard]] std::optional<VgprSpillSequence>
build_moi_spill_sequence(const ConSanResult &result, const ResolvedMoiScratchPlan &resources,
                         const ConSanOptions &options, MoiSpillManagers &managers,
                         rj_code_arch_t arch, std::vector<std::string> &warnings,
                         std::optional<uint32_t> private_layout_base = std::nullopt) {
  if (resources.owner_descriptor_file_offsets.empty()) {
    warnings.emplace_back("ConSan MOI spill requires an owning kernel descriptor");
    return std::nullopt;
  }
  const auto private_limit = address_free_scratch_private_limit(arch);
  if (!private_limit) {
    warnings.emplace_back(
        "ConSan MOI spill has no address-free scratch support for this architecture");
    return std::nullopt;
  }

  bool has_dynamic_owner = false;
  bool has_fixed_owner = false;
  for (uint64_t descriptor_offset : resources.owner_descriptor_file_offsets) {
    const ConSanKernelInfo *kernel = kernel_for_descriptor(result, descriptor_offset);
    if (kernel == nullptr) {
      warnings.emplace_back("ConSan MOI spill references an unknown kernel descriptor");
      return std::nullopt;
    }
    if (kernel->uses_dynamic_stack.value_or(false))
      has_dynamic_owner = true;
    else
      has_fixed_owner = true;
  }
  const bool needs_dynamic_scalar_frame =
      has_dynamic_owner &&
      (options.automatic_moi_inline_sgpr_spill || options.automatic_moi_record_replay_sgpr_spill);
  if (resources.source != ConSanRegisterAllocationSource::SpillRequired &&
      !needs_dynamic_scalar_frame) {
    return std::nullopt;
  }

  if (has_dynamic_owner) {
    if (has_fixed_owner) {
      warnings.emplace_back(
          "ConSan MOI spill cannot share one frame recipe across dynamic and fixed owners");
      return std::nullopt;
    }
    if (!moi_supports_dynamic_stack_spill(arch, options.moi_engine)) {
      warnings.emplace_back(
          "ConSan MOI dynamic-stack spill has no backend for this target and engine");
      return std::nullopt;
    }
    if (!options.moi_dynamic_stack_spill) {
      warnings.emplace_back(
          "ConSan MOI dynamic-stack spill was not included in scalar-state planning");
      return std::nullopt;
    }
    if (!options.moi_exec_save_sgpr) {
      warnings.emplace_back("ConSan MOI dynamic-stack spill has no EXEC-save scalar window");
      return std::nullopt;
    }
    const bool spill_backed_scalar_window =
        options.automatic_moi_inline_sgpr_spill || options.automatic_moi_record_replay_sgpr_spill;
    const auto special_state = moi_special_state_sgprs(options);
    const auto indirect_state = moi_indirect_jump_sgprs(options);
    if (!special_state || (spill_backed_scalar_window && !indirect_state)) {
      warnings.emplace_back("ConSan MOI dynamic-stack spill has no SCC-save register");
      return std::nullopt;
    }
    const uint16_t saved_scc_sgpr =
        spill_backed_scalar_window ? indirect_state->scc_save_sgpr : special_state->scc_save_sgpr;
    const auto saved_frame_offset = moi_dynamic_stack_frame_save_sgpr_offset(options.moi_engine);
    if (!spill_backed_scalar_window &&
        (!saved_frame_offset || *saved_frame_offset >= moi_exec_save_sgpr_count(options, arch))) {
      warnings.emplace_back("ConSan MOI dynamic-stack spill has no reserved frame-base save slot");
      return std::nullopt;
    }
    const uint16_t saved_frame_base_sgpr =
        spill_backed_scalar_window
            ? indirect_state->pc_sgpr
            : static_cast<uint16_t>(*options.moi_exec_save_sgpr + *saved_frame_offset);
    const uint32_t additional_frame_bytes =
        spill_backed_scalar_window
            ? static_cast<uint32_t>(moi_exec_save_sgpr_count(options, arch)) *
                  SpillManager::kSlotBytes
            : 0u;
    auto spill = build_dynamic_stack_vgpr_spill_sequence(
        resources.base, resources.count, kDynamicStackTopSgpr, kDynamicFrameBaseSgpr,
        saved_frame_base_sgpr, saved_scc_sgpr, arch, additional_frame_bytes);
    const uint64_t requested_private_bytes =
        static_cast<uint64_t>(resources.original_private_segment_size) +
        static_cast<uint64_t>(resources.count) * SpillManager::kSlotBytes + additional_frame_bytes;
    const auto required_private_bytes =
        requested_private_bytes <= std::numeric_limits<uint32_t>::max()
            ? normalize_address_free_scratch_private_size(
                  arch, static_cast<uint32_t>(requested_private_bytes))
            : std::nullopt;
    if (!spill || !required_private_bytes) {
      warnings.emplace_back("ConSan MOI could not encode the dynamic-stack VGPR spill frame");
      return std::nullopt;
    }
    // The offsets are frame-relative, but the loader still needs enough
    // backing scratch for the extra maximum stack depth.
    spill->total_private_bytes = *required_private_bytes;
    return spill;
  }

  if (resources.owner_descriptor_file_offsets.size() > 1) {
    // Shared text needs one immediate layout. Start above the caller's
    // persistent-state boundary when supplied, otherwise above the maximum
    // original private size. That keeps one save/fill sequence legal for
    // every owner without overlapping epoch, owner, or sequence state.
    SpillManager shared_manager(
        private_layout_base.value_or(resources.original_private_segment_size), *private_limit);
    auto spill = build_vgpr_spill_sequence(shared_manager, resources.base, resources.count, arch);
    if (!spill)
      warnings.emplace_back("ConSan MOI could not encode the shared-owner VGPR spill window");
    return spill;
  }

  const uint64_t descriptor_offset = resources.owner_descriptor_file_offsets.front();
  auto [it, inserted] = managers.try_emplace(
      descriptor_offset, private_layout_base.value_or(resources.original_private_segment_size),
      *private_limit);
  (void)inserted;
  auto spill = build_vgpr_spill_sequence(it->second, resources.base, resources.count, arch);
  if (!spill)
    warnings.emplace_back("ConSan MOI could not encode or reserve the planned VGPR spill window");
  return spill;
}

[[nodiscard]] bool
apply_descriptor_requirements(CodeObjectPatcher &patcher, const AmdGpuCodeObject &code_object,
                              std::span<const uint8_t> image, const ConSanResult &result,
                              const MoiDescriptorVgprRequirements &requirements,
                              rj_code_arch_t arch, std::vector<std::string> &errors) {
  for (const auto &[descriptor_offset, required_count] : requirements) {
    const ConSanKernelInfo *kernel = kernel_for_descriptor(result, descriptor_offset);
    if (kernel == nullptr) {
      errors.emplace_back("ConSan MOI resource plan references an unknown kernel descriptor");
      return false;
    }
    const auto active_kernel =
        std::ranges::find_if(code_object.kernels(), [&](const AmdGpuKernelInfo &candidate) {
          return candidate.name == kernel->name;
        });
    if (active_kernel == code_object.kernels().end()) {
      errors.emplace_back("ConSan MOI resource plan could not resolve an active descriptor");
      return false;
    }
    ConSanKernelInfo active = *kernel;
    active.descriptor_file_offset = active_kernel->descriptor_file_offset;
    if (!grow_moi_kernel_descriptor_vgprs(patcher, image, active, required_count, arch, errors))
      return false;
  }
  return true;
}

[[nodiscard]] bool
apply_spill_descriptor_requirements(CodeObjectPatcher &patcher, const AmdGpuCodeObject &code_object,
                                    std::span<const uint8_t> image, const ConSanResult &result,
                                    const MoiDescriptorPrivateRequirements &requirements,
                                    std::vector<std::string> &errors) {
  (void)image;
  for (const auto &[descriptor_offset, required_private_bytes] : requirements) {
    const ConSanKernelInfo *kernel = kernel_for_descriptor(result, descriptor_offset);
    const auto active_kernel =
        kernel == nullptr
            ? code_object.kernels().end()
            : std::ranges::find_if(code_object.kernels(), [&](const AmdGpuKernelInfo &candidate) {
                return candidate.name == kernel->name;
              });
    const std::span<const uint8_t> current_image = patcher.image_bytes();
    if (kernel == nullptr || active_kernel == code_object.kernels().end() ||
        active_kernel->descriptor_file_offset > current_image.size() ||
        sizeof(KD) > current_image.size() - active_kernel->descriptor_file_offset) {
      errors.emplace_back("ConSan MOI spill descriptor exceeds ELF bytes or has no kernel owner");
      return false;
    }

    std::vector<uint8_t> descriptor(sizeof(KD));
    std::memcpy(descriptor.data(), current_image.data() + active_kernel->descriptor_file_offset,
                sizeof(KD));
    const SpillDescriptorUpdate update = update_kernel_descriptor_for_spills(
        descriptor, /*descriptor_file_offset=*/0, required_private_bytes);
    if (update != SpillDescriptorUpdate::Updated && update != SpillDescriptorUpdate::Unchanged) {
      errors.emplace_back("ConSan MOI could not grow the owning kernel's private spill segment");
      return false;
    }
    if (!patcher.patch_kernel_descriptor(active_kernel->descriptor_file_offset, descriptor)) {
      errors.emplace_back("ConSan MOI could not patch the owning kernel spill descriptor");
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool apply_sgpr_descriptor_requirements(
    CodeObjectPatcher &patcher, const ConSanResult &result, const AmdGpuCodeObject &code_object,
    const MoiDescriptorSgprRequirements &requirements, std::vector<std::string> &errors) {
  for (const auto &[descriptor_offset, required_count] : requirements) {
    const ConSanKernelInfo *kernel = kernel_for_descriptor(result, descriptor_offset);
    const auto active_kernel =
        kernel == nullptr
            ? code_object.kernels().end()
            : std::ranges::find_if(code_object.kernels(), [&](const AmdGpuKernelInfo &candidate) {
                return candidate.name == kernel->name;
              });
    if (active_kernel == code_object.kernels().end()) {
      errors.emplace_back("ConSan MOI scalar plan could not resolve an active descriptor");
      return false;
    }
    const uint64_t active_descriptor_offset = active_kernel->descriptor_file_offset;
    const std::span<const uint8_t> current_image = patcher.image_bytes();
    if (active_descriptor_offset > current_image.size() ||
        sizeof(KD) > current_image.size() - active_descriptor_offset) {
      errors.emplace_back("ConSan MOI scalar-plan descriptor exceeds ELF bytes");
      return false;
    }
    KD descriptor{};
    std::memcpy(&descriptor, current_image.data() + active_descriptor_offset, sizeof(descriptor));
    if (!grow_moi_descriptor_sgpr_allocation(descriptor, required_count)) {
      errors.emplace_back("ConSan MOI could not satisfy planned descriptor SGPR allocation");
      return false;
    }
    if (!patcher.patch_kernel_descriptor(
            active_descriptor_offset,
            std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(&descriptor),
                                     sizeof(descriptor)))) {
      errors.emplace_back("ConSan MOI could not patch the planned descriptor SGPR allocation");
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool apply_lds_descriptor_requirements(
    CodeObjectPatcher &patcher, const ConSanResult &result, const AmdGpuCodeObject &code_object,
    const MoiDescriptorLdsRequirements &requirements, const ConSanOptions &options,
    rj_code_arch_t arch, std::vector<std::string> &errors) {
  for (const auto &[descriptor_offset, required_bytes] : requirements) {
    const ConSanKernelInfo *kernel = kernel_for_descriptor(result, descriptor_offset);
    const auto active_kernel =
        kernel == nullptr
            ? code_object.kernels().end()
            : std::ranges::find_if(code_object.kernels(), [&](const AmdGpuKernelInfo &candidate) {
                return candidate.name == kernel->name;
              });
    if (active_kernel == code_object.kernels().end()) {
      errors.emplace_back("ConSan MOI LDS plan could not resolve an active descriptor");
      return false;
    }
    const uint64_t active_descriptor_offset = active_kernel->descriptor_file_offset;
    const std::span<const uint8_t> current_image = patcher.image_bytes();
    if (active_descriptor_offset > current_image.size() ||
        sizeof(KD) > current_image.size() - active_descriptor_offset) {
      errors.emplace_back("ConSan MOI LDS-plan descriptor exceeds ELF bytes");
      return false;
    }
    KD descriptor{};
    std::memcpy(&descriptor, current_image.data() + active_descriptor_offset, sizeof(descriptor));
    if (required_bytes > consan_moi_max_workgroup_lds_bytes(options, arch) ||
        descriptor.group_segment_fixed_size > required_bytes) {
      errors.emplace_back("ConSan MOI LDS-plan descriptor has an incompatible fixed-LDS size");
      return false;
    }
    descriptor.group_segment_fixed_size = required_bytes;
    if (!patcher.patch_kernel_descriptor(
            active_descriptor_offset,
            std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(&descriptor),
                                     sizeof(descriptor)))) {
      errors.emplace_back("ConSan MOI could not patch the planned fixed-LDS reservation");
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool apply_descriptor_requirements(std::vector<uint8_t> &image,
                                                 const ConSanResult &result,
                                                 const MoiDescriptorVgprRequirements &requirements,
                                                 rj_code_arch_t arch,
                                                 std::vector<std::string> &errors) {
  for (const auto &[descriptor_offset, required_count] : requirements) {
    if (descriptor_offset > image.size() || sizeof(KD) > image.size() - descriptor_offset) {
      errors.emplace_back("ConSan MOI resource-plan descriptor exceeds ELF bytes");
      return false;
    }
    if (kernel_for_descriptor(result, descriptor_offset) == nullptr) {
      errors.emplace_back("ConSan MOI resource plan references an unknown kernel descriptor");
      return false;
    }
    KD descriptor{};
    std::memcpy(&descriptor, image.data() + descriptor_offset, sizeof(descriptor));
    if (!grow_moi_descriptor_vgpr_allocation(descriptor, required_count, arch)) {
      errors.emplace_back("ConSan MOI could not satisfy planned descriptor VGPR allocation");
      return false;
    }
    std::memcpy(image.data() + descriptor_offset, &descriptor, sizeof(descriptor));
  }
  return true;
}

[[nodiscard]] bool
apply_sgpr_descriptor_requirements(std::vector<uint8_t> &image, const ConSanResult &result,
                                   const MoiDescriptorSgprRequirements &requirements,
                                   std::vector<std::string> &errors) {
  for (const auto &[descriptor_offset, required_count] : requirements) {
    if (descriptor_offset > image.size() || sizeof(KD) > image.size() - descriptor_offset) {
      errors.emplace_back("ConSan MOI scalar-plan descriptor exceeds ELF bytes");
      return false;
    }
    if (kernel_for_descriptor(result, descriptor_offset) == nullptr) {
      errors.emplace_back("ConSan MOI scalar plan references an unknown kernel descriptor");
      return false;
    }
    KD descriptor{};
    std::memcpy(&descriptor, image.data() + descriptor_offset, sizeof(descriptor));
    if (!grow_moi_descriptor_sgpr_allocation(descriptor, required_count)) {
      errors.emplace_back("ConSan MOI could not satisfy planned descriptor SGPR allocation");
      return false;
    }
    std::memcpy(image.data() + descriptor_offset, &descriptor, sizeof(descriptor));
  }
  return true;
}

[[nodiscard]] bool
apply_lds_descriptor_requirements(std::vector<uint8_t> &image, const ConSanResult &result,
                                  const MoiDescriptorLdsRequirements &requirements,
                                  const ConSanOptions &options, rj_code_arch_t arch,
                                  std::vector<std::string> &errors) {
  for (const auto &[descriptor_offset, required_bytes] : requirements) {
    if (descriptor_offset > image.size() || sizeof(KD) > image.size() - descriptor_offset ||
        kernel_for_descriptor(result, descriptor_offset) == nullptr) {
      errors.emplace_back("ConSan MOI LDS-plan descriptor exceeds ELF bytes or has no owner");
      return false;
    }
    KD descriptor{};
    std::memcpy(&descriptor, image.data() + descriptor_offset, sizeof(descriptor));
    if (required_bytes > consan_moi_max_workgroup_lds_bytes(options, arch) ||
        descriptor.group_segment_fixed_size > required_bytes) {
      errors.emplace_back("ConSan MOI LDS-plan descriptor has an incompatible fixed-LDS size");
      return false;
    }
    descriptor.group_segment_fixed_size = required_bytes;
    std::memcpy(image.data() + descriptor_offset, &descriptor, sizeof(descriptor));
  }
  return true;
}

[[nodiscard]] bool append_compute_effective_lds_byte_offset(std::vector<uint32_t> &words,
                                                            uint16_t dst_vgpr, uint16_t addr_vgpr,
                                                            uint32_t static_byte_offset,
                                                            rj_code_arch_t arch) {
  if (static_byte_offset == 0)
    return true;

  const auto mov_offset =
      instrumentation::build_v_mov_b32_literal(dst_vgpr, static_byte_offset, arch);
  const auto add_offset =
      instrumentation::build_v_add_u32(dst_vgpr, vector_source_vgpr(addr_vgpr), dst_vgpr, arch);
  InstructionSequence sequence(words);
  return sequence.emit_all(mov_offset, add_offset);
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_first_light_access_record_words(
    std::span<const uint8_t> bytes, const ConSanMoiCandidate &candidate,
    const ConSanOptions &options, rj_code_arch_t arch, uint32_t record_index, uint32_t record_count,
    uint32_t access_record_capacity, uint32_t access_dispatch_bank_count,
    uint32_t access_owner_bank_count, const VgprSpillSequence *spill,
    std::optional<uint32_t> private_epoch_offset, std::optional<uint32_t> private_owner_offset,
    std::optional<uint16_t> private_owner_shift, std::vector<std::string> &errors,
    uint32_t *guest_instruction_offset = nullptr) {
  if (!options.scratch_vgpr) {
    errors.emplace_back("ConSan MOI first-light probe requires RJ_CONSAN_TMP_VGPR");
    return std::nullopt;
  }
  const uint16_t base_scratch_count = 6u;
  const bool materialize_flat_offset =
      candidate_requires_gfx1250_flat_offset_materialization(candidate);
  const bool save_clobbered_address = moi_load_clobbers_address(candidate);
  const uint16_t address_scratch_count =
      materialize_flat_offset ? 2u : (save_clobbered_address ? 1u : 0u);
  const uint16_t scratch_count = static_cast<uint16_t>(base_scratch_count + address_scratch_count);
  if (static_cast<uint32_t>(*options.scratch_vgpr) + scratch_count > 256u) {
    errors.emplace_back(options.moi_dynamic_access_records
                            ? "ConSan MOI dynamic access-record probe needs six scratch VGPRs"
                            : "ConSan MOI first-light probe needs six scratch VGPRs");
    return std::nullopt;
  }
  if (options.moi_dynamic_access_records && !options.moi_exec_save_sgpr) {
    errors.emplace_back(
        "ConSan MOI dynamic access-record probe requires RJ_CONSAN_MOI_EXEC_SAVE_SGPR");
    return std::nullopt;
  }
  if (options.moi_dynamic_access_records &&
      (*options.moi_exec_save_sgpr > 100u || *options.moi_exec_save_sgpr % 2u != 0u)) {
    errors.emplace_back(
        "ConSan MOI dynamic access-record probe requires an even RJ_CONSAN_MOI_EXEC_SAVE_SGPR in "
        "0..100");
    return std::nullopt;
  }
  auto lds_byte_offset_vgpr = candidate_lds_byte_offset_vgpr(candidate, errors);
  if (!lds_byte_offset_vgpr)
    return std::nullopt;
  if (!options.moi_spill_overlaps_guest_operands &&
      reject_candidate_scratch_range_overlap(candidate, *options.scratch_vgpr, scratch_count,
                                             errors))
    return std::nullopt;
  if (options.moi_dynamic_access_records && has_recent_saveexec(bytes, candidate)) {
    errors.emplace_back(
        "ConSan MOI dynamic access-record probe skipped a candidate immediately after "
        "s_*_saveexec");
    return std::nullopt;
  }
  if (reject_optional_scratch_range_overlap(options.moi_owner_vgpr, *options.scratch_vgpr,
                                            scratch_count, "MOI owner", errors) ||
      reject_optional_scratch_range_overlap(options.moi_epoch_vgpr, *options.scratch_vgpr,
                                            scratch_count, "MOI epoch", errors) ||
      reject_optional_scratch_range_overlap(options.moi_workgroup_key_vgpr, *options.scratch_vgpr,
                                            scratch_count, "MOI workgroup key", errors))
    return std::nullopt;
  auto byte_count = byte_count_for_candidate(candidate);
  if (!byte_count) {
    errors.emplace_back("ConSan MOI first-light probe could not determine LDS byte count");
    return std::nullopt;
  }
  const std::optional<std::vector<ConSanMoiAccessRange>> access_ranges =
      candidate_access_ranges(bytes, candidate, arch);
  if (!access_ranges || access_ranges->empty()) {
    errors.emplace_back("ConSan MOI first-light probe requires a supported LDS access range");
    return std::nullopt;
  }
  if (materialize_flat_offset && access_ranges->size() != 1u) {
    errors.emplace_back("ConSan MOI flat-offset probe requires one normalized access range");
    return std::nullopt;
  }
  std::optional<uint16_t> derived_owner_vgpr;
  std::vector<uint32_t> derived_owner_words;
  if (private_owner_offset.has_value() != private_owner_shift.has_value()) {
    errors.emplace_back("ConSan MOI first-light probe has an incomplete private owner plan");
    return std::nullopt;
  }
  if (!options.moi_owner_vgpr && !options.moi_persistent_owner_sgpr &&
      (private_owner_offset || candidate.kernel_descriptor_file_offset)) {
    std::optional<uint16_t> owner_shift = private_owner_shift;
    if (!owner_shift && candidate.kernel_descriptor_file_offset) {
      owner_shift =
          moi_descriptor_owner_shift(bytes, *candidate.kernel_descriptor_file_offset, errors);
    }
    if (!owner_shift)
      return std::nullopt;
    const uint16_t value_vgpr = static_cast<uint16_t>(
        *options.scratch_vgpr + (options.moi_dynamic_access_records ? 4u : 2u));
    if (private_owner_offset) {
      const auto owner_load =
          instrumentation::build_private_load_b32(value_vgpr, *private_owner_offset, arch);
      const auto owner_wait = instrumentation::build_s_wait_private_load0(arch);
      const auto owner_init = instrumentation::build_v_lshrrev_b32(
          value_vgpr, scalar_positive_inline_u32(*owner_shift), value_vgpr, arch);
      if (!owner_load || !owner_wait || !owner_init) {
        errors.emplace_back(
            "ConSan MOI first-light probe could not load its persistent workitem-id owner");
        return std::nullopt;
      }
      derived_owner_words.insert(derived_owner_words.end(), owner_load->begin(), owner_load->end());
      derived_owner_words.push_back(*owner_wait);
      derived_owner_words.push_back(*owner_init);
    } else {
      const auto owner_init = instrumentation::build_v_lshrrev_b32(
          value_vgpr, scalar_positive_inline_u32(*owner_shift), kRdna4WorkitemIdX, arch);
      if (!owner_init) {
        errors.emplace_back("ConSan MOI first-light probe could not encode owner derivation");
        return std::nullopt;
      }
      derived_owner_words.push_back(*owner_init);
    }
    const auto owner_mask = instrumentation::build_v_and_b32_literal(
        value_vgpr, consan_moi_exact_shadow::max_owner, value_vgpr, arch);
    if (!owner_mask) {
      errors.emplace_back("ConSan MOI first-light probe could not encode owner derivation");
      return std::nullopt;
    }
    derived_owner_vgpr = value_vgpr;
    derived_owner_words.insert(derived_owner_words.end(), owner_mask->begin(), owner_mask->end());
  }
  const auto persistent_workgroup_sources =
      record_replay_persistent_workgroup_sources(options, arch);
  ConSanMoiWorkgroupSources workgroup_sources =
      persistent_workgroup_sources.value_or(ConSanMoiWorkgroupSources{});
  if (!persistent_workgroup_sources && candidate.kernel_descriptor_file_offset) {
    const auto descriptor_workgroup_sources = moi_descriptor_workgroup_sources(
        bytes, *candidate.kernel_descriptor_file_offset, arch, errors);
    if (!descriptor_workgroup_sources)
      return std::nullopt;
    workgroup_sources =
        record_replay_workgroup_sources(*descriptor_workgroup_sources, options, arch);
  }

  std::vector<uint32_t> words;
  words.reserve(candidate.size / sizeof(uint32_t) + 1u + 7u * 12u + 9u + 10u + 20u + 24u +
                (options.moi_owner_vgpr || derived_owner_vgpr ? 9u : 0u) +
                (options.moi_epoch_vgpr ? 9u : 0u) + derived_owner_words.size());
  InstructionSequence sequence(words);
  const auto restore_exec_label = sequence.make_label();
  // A DS load may overwrite the same VGPR that supplied its address. Preserve
  // that effective address before executing the displaced guest instruction;
  // otherwise RecordReplay publishes the loaded payload as the LDS offset.
  if (materialize_flat_offset && options.moi_dynamic_access_records) {
    const uint16_t materialized_address_vgpr =
        static_cast<uint16_t>(*options.scratch_vgpr + base_scratch_count);
    words.push_back(build_v_mov_b32_e32(materialized_address_vgpr,
                                        vector_source_vgpr(*candidate.addr_vgpr), arch));
    words.push_back(build_v_mov_b32_e32(
        static_cast<uint16_t>(materialized_address_vgpr + 1u),
        vector_source_vgpr(static_cast<uint16_t>(*candidate.addr_vgpr + 1u)), arch));
    lds_byte_offset_vgpr = materialized_address_vgpr;
  } else if (save_clobbered_address) {
    const uint16_t saved_lds_byte_offset_vgpr =
        static_cast<uint16_t>(*options.scratch_vgpr + base_scratch_count);
    words.push_back(build_v_mov_b32_e32(saved_lds_byte_offset_vgpr,
                                        vector_source_vgpr(*lds_byte_offset_vgpr), arch));
    lds_byte_offset_vgpr = saved_lds_byte_offset_vgpr;
  }
  if (options.moi_dynamic_access_records)
    words.insert(words.end(), derived_owner_words.begin(), derived_owner_words.end());
  const auto append_guest_access = [&] {
    if (guest_instruction_offset)
      *guest_instruction_offset = static_cast<uint32_t>(words.size() * sizeof(uint32_t));
    for (uint64_t offset = 0; offset < candidate.size; offset += sizeof(uint32_t)) {
      uint32_t word = 0;
      std::memcpy(&word, bytes.data() + candidate.file_offset + offset, sizeof(word));
      words.push_back(word);
    }
  };
  if (options.moi_dynamic_access_records) {
    append_guest_access();
    if (!append_moi_lds_wait(words, arch))
      return std::nullopt;
  }

  const uint64_t base = *options.moi_report_buffer_address;
  const auto kind = consan_moi_shadow_kind_from_access_kind(candidate.kind);
  auto append_effective_range_offset = [&](const ConSanMoiAccessRange &range,
                                           uint16_t value_vgpr) -> std::optional<uint16_t> {
    if (range.static_byte_offset == 0)
      return *lds_byte_offset_vgpr;
    if (!append_compute_effective_lds_byte_offset(words, value_vgpr, *lds_byte_offset_vgpr,
                                                  range.static_byte_offset, arch)) {
      return std::nullopt;
    }
    return value_vgpr;
  };

  if (options.moi_dynamic_access_records) {
    const uint16_t slot_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 2u);
    const uint16_t value_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 5u);
    const uint64_t dynamic_record_base = base + sizeof(ConSanMoiReportHeader);

    for (const ConSanMoiAccessRange &range : *access_ranges) {
      if (!append_save_moi_special_state(words, options, arch)) {
        errors.emplace_back("ConSan MOI dynamic access-record probe could not save VCC/SCC");
        return std::nullopt;
      }
      if (materialize_flat_offset) {
        const auto add = instrumentation::build_v_add_u64_signed_i24(*lds_byte_offset_vgpr,
                                                                     *candidate.raw_ioffset, arch);
        if (!add) {
          errors.emplace_back(
              "ConSan MOI dynamic access-record probe could not materialize FLAT offset");
          return std::nullopt;
        }
        words.insert(words.end(), add->begin(), add->end());
      }

      if (!append_atomic_fetch_add_one_u32(
              words, base + offsetof(ConSanMoiReportHeader, access_record_count), slot_vgpr,
              *options.scratch_vgpr, arch)) {
        errors.emplace_back(
            "ConSan MOI dynamic access-record probe could not reserve a record slot");
        return std::nullopt;
      }

      const auto mov_capacity =
          instrumentation::build_v_mov_b32_literal(value_vgpr, access_record_capacity, arch);
      const auto slot_in_capacity =
          instrumentation::build_v_cmp_gt_u32_vcc(vector_source_vgpr(value_vgpr), slot_vgpr, arch);
      const auto narrow_exec =
          instrumentation::build_s_and_saveexec_b64(*options.moi_exec_save_sgpr, kRdna4VccLo, arch);
      if (!mov_capacity || !slot_in_capacity || !narrow_exec) {
        errors.emplace_back(
            "ConSan MOI dynamic access-record probe could not encode capacity guard");
        return std::nullopt;
      }
      words.insert(words.end(), mov_capacity->begin(), mov_capacity->end());
      words.push_back(*slot_in_capacity);
      words.push_back(*narrow_exec);

      if (private_epoch_offset) {
        const auto load_epoch =
            instrumentation::build_private_load_b32(value_vgpr, *private_epoch_offset, arch);
        const auto wait_epoch = instrumentation::build_s_wait_private_load0(arch);
        if (!load_epoch || !wait_epoch) {
          errors.emplace_back(
              "ConSan MOI dynamic access-record probe could not load private epoch state");
          return std::nullopt;
        }
        words.insert(words.end(), load_epoch->begin(), load_epoch->end());
        words.push_back(*wait_epoch);
        if (!append_dynamic_access_store_u32_vgpr(
                words, dynamic_record_base + offsetof(ConSanMoiAccessRecord, epoch), value_vgpr,
                slot_vgpr, *options.scratch_vgpr, arch)) {
          errors.emplace_back(
              "ConSan MOI dynamic access-record probe could not store private epoch state");
          return std::nullopt;
        }
      } else if (options.moi_persistent_epoch_sgpr) {
        words.push_back(build_v_mov_b32_e32(value_vgpr, *options.moi_persistent_epoch_sgpr, arch));
        if (!append_dynamic_access_store_u32_vgpr(
                words, dynamic_record_base + offsetof(ConSanMoiAccessRecord, epoch), value_vgpr,
                slot_vgpr, *options.scratch_vgpr, arch)) {
          errors.emplace_back(
              "ConSan MOI dynamic access-record probe could not store scalar epoch state");
          return std::nullopt;
        }
      }

      if ((derived_owner_vgpr &&
           !append_dynamic_access_store_u32_vgpr(
               words, dynamic_record_base + offsetof(ConSanMoiAccessRecord, wave_id),
               *derived_owner_vgpr, slot_vgpr, *options.scratch_vgpr, arch)) ||
          (options.moi_persistent_owner_sgpr &&
           !append_dynamic_access_store_u32_scalar_src(
               words, dynamic_record_base + offsetof(ConSanMoiAccessRecord, wave_id),
               *options.moi_persistent_owner_sgpr, slot_vgpr, *options.scratch_vgpr, arch)) ||
          !append_dynamic_record_store_moi_report_dispatch_id_pair(
              words, kDynamicAccessRecordLayout,
              dynamic_record_base + offsetof(ConSanMoiAccessRecord, generation), options,
              slot_vgpr, *options.scratch_vgpr, arch,
              ConSanMoiLiteralDispatchIdPolicy::AnyArchitecture) ||
          !append_dynamic_access_event_index_store(
              words, base + offsetof(ConSanMoiReportHeader, event_counter),
              dynamic_record_base + offsetof(ConSanMoiAccessRecord, event_index), slot_vgpr,
              *options.scratch_vgpr, arch) ||
          !append_dynamic_access_store_workgroup_source(
              words, dynamic_record_base + offsetof(ConSanMoiAccessRecord, workgroup_x),
              workgroup_sources.x, slot_vgpr, *options.scratch_vgpr, arch) ||
          !append_dynamic_access_store_workgroup_source(
              words, dynamic_record_base + offsetof(ConSanMoiAccessRecord, workgroup_y),
              workgroup_sources.y, slot_vgpr, *options.scratch_vgpr, arch) ||
          !append_dynamic_access_store_workgroup_source(
              words, dynamic_record_base + offsetof(ConSanMoiAccessRecord, workgroup_z),
              workgroup_sources.z, slot_vgpr, *options.scratch_vgpr, arch) ||
          !append_dynamic_access_store_u32_scalar_src(
              words, dynamic_record_base + offsetof(ConSanMoiAccessRecord, lane_mask),
              *options.moi_exec_save_sgpr, slot_vgpr, *options.scratch_vgpr, arch) ||
          !append_dynamic_access_store_u32_scalar_src(
              words,
              dynamic_record_base + offsetof(ConSanMoiAccessRecord, lane_mask) + sizeof(uint32_t),
              static_cast<uint16_t>(*options.moi_exec_save_sgpr + 1u), slot_vgpr,
              *options.scratch_vgpr, arch) ||
          (options.moi_owner_vgpr &&
           !append_dynamic_access_store_u32_vgpr(
               words, dynamic_record_base + offsetof(ConSanMoiAccessRecord, wave_id),
               *options.moi_owner_vgpr, slot_vgpr, *options.scratch_vgpr, arch)) ||
          (options.moi_epoch_vgpr &&
           !append_dynamic_access_store_u32_vgpr(
               words, dynamic_record_base + offsetof(ConSanMoiAccessRecord, epoch),
               *options.moi_epoch_vgpr, slot_vgpr, *options.scratch_vgpr, arch)) ||
          !append_dynamic_access_store_u32_literal(
              words, dynamic_record_base + offsetof(ConSanMoiAccessRecord, instruction_offset),
              static_cast<uint32_t>(candidate.text_offset), slot_vgpr, *options.scratch_vgpr,
              arch) ||
          !append_dynamic_access_store_u32_literal(
              words, dynamic_record_base + offsetof(ConSanMoiAccessRecord, access_kind),
              static_cast<uint32_t>(kind), slot_vgpr, *options.scratch_vgpr, arch)) {
        errors.emplace_back(
            "ConSan MOI dynamic access-record probe could not encode record stores");
        return std::nullopt;
      }

      const std::optional<uint16_t> effective_lds_byte_offset_vgpr =
          append_effective_range_offset(range, value_vgpr);
      if (!effective_lds_byte_offset_vgpr) {
        errors.emplace_back(
            "ConSan MOI dynamic access-record probe could not encode LDS byte offset");
        return std::nullopt;
      }
      if (!append_dynamic_access_store_u32_vgpr(
              words, dynamic_record_base + offsetof(ConSanMoiAccessRecord, lds_byte_offset),
              *effective_lds_byte_offset_vgpr, slot_vgpr, *options.scratch_vgpr, arch)) {
        errors.emplace_back("ConSan MOI dynamic access-record probe could not encode range fields");
        return std::nullopt;
      }
      const auto start_cell_shift = instrumentation::build_v_lshrrev_b32(
          value_vgpr, scalar_positive_inline_u32(consan_moi_exact_shadow::granule_shift),
          *effective_lds_byte_offset_vgpr, arch);
      if (!start_cell_shift) {
        errors.emplace_back("ConSan MOI dynamic access-record probe could not encode start cell");
        return std::nullopt;
      }
      words.push_back(*start_cell_shift);
      const ConSanMoiLdsCellRange static_range =
          consan_moi_lds_cell_range_for_bytes(range.static_byte_offset, range.byte_count);
      if (!append_dynamic_access_store_u32_vgpr(
              words, dynamic_record_base + offsetof(ConSanMoiAccessRecord, start_cell), value_vgpr,
              slot_vgpr, *options.scratch_vgpr, arch) ||
          !append_dynamic_access_store_u32_literal(
              words, dynamic_record_base + offsetof(ConSanMoiAccessRecord, lds_byte_count),
              range.byte_count, slot_vgpr, *options.scratch_vgpr, arch) ||
          !append_dynamic_access_store_u32_literal(
              words, dynamic_record_base + offsetof(ConSanMoiAccessRecord, cell_count),
              static_range.cell_count, slot_vgpr, *options.scratch_vgpr, arch)) {
        errors.emplace_back("ConSan MOI dynamic access-record probe could not encode range fields");
        return std::nullopt;
      }
      const auto wait_store = instrumentation::build_s_wait_flat_store0(arch);
      const auto restore_exec =
          instrumentation::build_s_mov_b64(kRdna4ExecLo, *options.moi_exec_save_sgpr, arch);
      if (!wait_store || !restore_exec) {
        errors.emplace_back(
            "ConSan MOI dynamic access-record probe could not drain stores or restore EXEC");
        return std::nullopt;
      }
      // A terminal guest access can return directly to s_endpgm. Drain the
      // injected report publication while the scratch values and narrowed
      // EXEC are still intact, so wave termination cannot race those stores.
      words.push_back(*wait_store);
      words.push_back(*restore_exec);
      if (!append_restore_moi_special_state(words, options, arch)) {
        errors.emplace_back("ConSan MOI dynamic access-record probe could not restore VCC/SCC");
        return std::nullopt;
      }
    }
    return words;
  }

  const uint16_t record_address_vgpr = *options.scratch_vgpr;
  const uint16_t record_value_vgpr = static_cast<uint16_t>(record_address_vgpr + 2u);
  const uint16_t record_value_high_vgpr = static_cast<uint16_t>(record_address_vgpr + 3u);
  const uint16_t record_compare_vgpr = static_cast<uint16_t>(record_address_vgpr + 4u);
  const uint16_t record_compare_high_vgpr = static_cast<uint16_t>(record_address_vgpr + 5u);
  const std::optional<uint16_t> spilled_lds_byte_offset_vgpr =
      !materialize_flat_offset && !save_clobbered_address && spill != nullptr &&
              options.moi_spill_overlaps_guest_operands &&
              *lds_byte_offset_vgpr >= spill->vgpr_base &&
              *lds_byte_offset_vgpr < static_cast<uint32_t>(spill->vgpr_base) + spill->vgpr_count
          ? lds_byte_offset_vgpr
          : std::nullopt;
  if (spilled_lds_byte_offset_vgpr)
    lds_byte_offset_vgpr = record_compare_high_vgpr;
  ConSanMoiRecordEmitter record(words, record_address_vgpr, record_value_vgpr, arch);
  constexpr uint16_t kScalarInlineMinusOne = 0xC1;
  const uint64_t access_bank_count =
      static_cast<uint64_t>(access_dispatch_bank_count) * access_owner_bank_count;
  if (access_dispatch_bank_count == 0u ||
      (access_dispatch_bank_count & (access_dispatch_bank_count - 1u)) != 0u ||
      access_owner_bank_count == 0u ||
      (access_owner_bank_count & (access_owner_bank_count - 1u)) != 0u ||
      access_bank_count > std::numeric_limits<uint32_t>::max() ||
      access_record_capacity % access_bank_count != 0u) {
    errors.emplace_back("ConSan MOI first-light probe has an invalid dispatch/owner-bank layout");
    return std::nullopt;
  }
  const auto append_claim_token = [&](uint16_t low_vgpr, uint16_t high_vgpr,
                                      uint16_t temporary_vgpr) {
    if (!append_moi_report_dispatch_id_pair(words, options, low_vgpr, high_vgpr, arch,
                                            ConSanMoiLiteralDispatchIdPolicy::AnyArchitecture))
      return false;
    const auto low_xor_literal = instrumentation::build_v_mov_b32_literal(
        temporary_vgpr, static_cast<uint32_t>(kConSanMoiRecordReplayClaimTokenXorMask), arch);
    const auto low_xor = instrumentation::build_v_xor_b32(
        low_vgpr, vector_source_vgpr(temporary_vgpr), low_vgpr, arch);
    const auto high_xor_literal = instrumentation::build_v_mov_b32_literal(
        temporary_vgpr, static_cast<uint32_t>(kConSanMoiRecordReplayClaimTokenXorMask >> 32u),
        arch);
    const auto high_xor = instrumentation::build_v_xor_b32(
        high_vgpr, vector_source_vgpr(temporary_vgpr), high_vgpr, arch);
    InstructionSequence append(words);
    return append.emit_all(low_xor_literal, low_xor, high_xor_literal, high_xor);
  };
  const auto append_owner_id = [&](uint16_t destination_vgpr) {
    if (options.moi_owner_vgpr) {
      words.push_back(
          build_v_mov_b32_e32(destination_vgpr, vector_source_vgpr(*options.moi_owner_vgpr), arch));
      return true;
    }
    if (options.moi_persistent_owner_sgpr) {
      words.push_back(
          build_v_mov_b32_e32(destination_vgpr, *options.moi_persistent_owner_sgpr, arch));
      return true;
    }
    if (!derived_owner_vgpr)
      return false;
    words.insert(words.end(), derived_owner_words.begin(), derived_owner_words.end());
    if (*derived_owner_vgpr != destination_vgpr) {
      words.push_back(
          build_v_mov_b32_e32(destination_vgpr, vector_source_vgpr(*derived_owner_vgpr), arch));
    }
    return true;
  };
  const auto materialize_banked_record_address = [&](uint64_t first_address) {
    if (!record.materialize_address(first_address))
      return false;
    if (access_bank_count == 1u)
      return true;
    if (access_dispatch_bank_count > 1u && !options.moi_dispatch_id_sgpr)
      return false;
    if (access_dispatch_bank_count > 1u) {
      words.push_back(build_v_mov_b32_e32(record_value_vgpr, *options.moi_dispatch_id_sgpr, arch));
    } else {
      words.push_back(build_v_mov_b32_e32(record_value_vgpr, scalar_positive_inline_u32(0), arch));
    }
    const auto select_dispatch_bank = instrumentation::build_v_and_b32_literal(
        record_value_vgpr, access_dispatch_bank_count - 1u, record_value_vgpr, arch);
    const auto shift_dispatch_bank =
        instrumentation::build_v_lshlrev_b32(record_value_vgpr,
                                             scalar_positive_inline_u32(static_cast<uint16_t>(
                                                 std::countr_zero(access_owner_bank_count))),
                                             record_value_vgpr, arch);
    std::optional<std::vector<uint32_t>> add_owner_bank;
    if (access_owner_bank_count > 1u) {
      if (!append_owner_id(record_compare_high_vgpr))
        return false;
      add_owner_bank = instrumentation::build_v_add_u32(
          record_value_vgpr, vector_source_vgpr(record_value_vgpr), record_compare_high_vgpr, arch);
    }
    const auto scale_bank = instrumentation::build_v_mul_lo_u32_literal(
        record_value_vgpr, record_value_high_vgpr, sizeof(ConSanMoiAccessRecord), record_value_vgpr,
        arch);
    const auto add_bank =
        instrumentation::build_v_add_u64_vgpr_offset(record_address_vgpr, record_value_vgpr, arch);
    InstructionSequence append(words);
    return append.emit_all(select_dispatch_bank, shift_dispatch_bank, add_owner_bank, scale_bank,
                           add_bank);
  };
  const auto reload_spilled_lds_byte_offset = [&] {
    if (!spilled_lds_byte_offset_vgpr)
      return true;
    if (spill == nullptr || !spill->has_complete_slot_metadata()) {
      return false;
    }
    const size_t slot_index = static_cast<size_t>(*spilled_lds_byte_offset_vgpr - spill->vgpr_base);
    if (slot_index >= spill->slot_offsets.size())
      return false;
    const uint32_t slot_offset = spill->slot_offsets[slot_index];
    if (spill->uses_dynamic_stack_frame) {
      if (arch == ROCJITSU_CODE_ARCH_CDNA3) {
        const auto load = build_cdna3_scratch_load_b32_saddr(
            record_compare_high_vgpr, spill->dynamic_frame_base_sgpr, slot_offset, arch);
        if (!load)
          return false;
        words.insert(words.end(), load->begin(), load->end());
      } else if (arch == ROCJITSU_CODE_ARCH_CDNA4) {
        const auto load = build_cdna4_scratch_load_b32_saddr(
            record_compare_high_vgpr, spill->dynamic_frame_base_sgpr, slot_offset, arch);
        if (!load)
          return false;
        words.insert(words.end(), load->begin(), load->end());
      } else {
        const auto load = build_scratch_load_b32_saddr(
            record_compare_high_vgpr, spill->dynamic_frame_base_sgpr, slot_offset, arch);
        if (!load)
          return false;
        words.insert(words.end(), load->begin(), load->end());
      }
    } else {
      const auto load =
          instrumentation::build_private_load_b32(record_compare_high_vgpr, slot_offset, arch);
      if (!load)
        return false;
      words.insert(words.end(), load->begin(), load->end());
    }
    const auto wait = instrumentation::build_s_wait_private_load0(arch);
    if (!wait)
      return false;
    words.push_back(*wait);
    return true;
  };
  if (!options.moi_exec_save_sgpr || !append_save_moi_special_state(words, options, arch)) {
    errors.emplace_back("ConSan MOI first-light probe could not save EXEC/VCC/SCC");
    return std::nullopt;
  }
  if (materialize_flat_offset) {
    const uint16_t materialized_address_vgpr =
        static_cast<uint16_t>(*options.scratch_vgpr + base_scratch_count);
    if (!append_materialize_gfx1250_flat_access_address(words, candidate, materialized_address_vgpr,
                                                        arch)) {
      errors.emplace_back("ConSan MOI first-light probe could not materialize FLAT offset");
      return std::nullopt;
    }
    lds_byte_offset_vgpr = materialized_address_vgpr;
  }
  const auto lane_rank_lo = instrumentation::build_v_mbcnt_lo_u32_b32(
      record_value_vgpr, kScalarInlineMinusOne, scalar_positive_inline_u32(0), arch);
  const auto lane_rank_hi = instrumentation::build_v_mbcnt_hi_u32_b32(
      record_value_vgpr, kScalarInlineMinusOne, vector_source_vgpr(record_value_vgpr), arch);
  const auto first_active = instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0),
                                                                    record_value_vgpr, arch);
  const auto narrow_first =
      instrumentation::build_s_and_saveexec_b64(*options.moi_exec_save_sgpr, kRdna4VccLo, arch);
  if (!lane_rank_lo || !lane_rank_hi || !first_active || !narrow_first) {
    errors.emplace_back("ConSan MOI first-light probe could not elect a representative lane");
    return std::nullopt;
  }
  words.insert(words.end(), lane_rank_lo->begin(), lane_rank_lo->end());
  words.insert(words.end(), lane_rank_hi->begin(), lane_rank_hi->end());
  words.push_back(*first_active);
  words.push_back(*narrow_first);

  // Fixed first-light records need one representative execution per site, but
  // the site may be control-dependent on a nonzero wave or workgroup. Let the
  // per-slot atomic claim below elect the first lane that actually executes the
  // access instead of suppressing valid evidence through an identity gate.
  for (size_t range_index = 0; range_index < access_ranges->size(); ++range_index) {
    const auto publication_done_label = sequence.make_label();
    const auto token_ready_label = sequence.make_label();
    const auto occupied_label = sequence.make_label();
    const auto saturation_label = sequence.make_label();
    const ConSanMoiAccessRange &range = (*access_ranges)[range_index];
    const uint64_t access_record_base =
        base + sizeof(ConSanMoiReportHeader) +
        (static_cast<uint64_t>(record_index) + range_index * access_bank_count) *
            sizeof(ConSanMoiAccessRecord);

    if (access_owner_bank_count > 1u) {
      if (!append_owner_id(record_compare_high_vgpr)) {
        errors.emplace_back("ConSan MOI first-light probe could not materialize its owner bank");
        return std::nullopt;
      }
      const auto owner_in_range = instrumentation::build_v_cmp_gt_u32_vcc(
          scalar_positive_inline_u32(access_owner_bank_count), record_compare_high_vgpr, arch);
      if (!owner_in_range) {
        errors.emplace_back("ConSan MOI first-light probe could not validate its owner bank");
        return std::nullopt;
      }
      words.push_back(*owner_in_range);
      if (!sequence.emit_branch(saturation_label, InstructionSequence::BranchKind::VccZero))
        return std::nullopt;
    }

    // Claim one bounded slot with a reversible encoding of the full hardware
    // dispatch ID. Zero is reserved for unpublished storage, so its one
    // preimage fails closed as saturation. A full-width CAS elects exactly one
    // publisher when different workgroups or overlapping dispatches reach the
    // same site. access_kind is committed atomically only after every payload
    // store has drained; a repeated publisher may reuse the immutable record
    // only after observing that commit and matching all three recorded
    // workgroup coordinates.
    if (!materialize_banked_record_address(access_record_base +
                                           offsetof(ConSanMoiAccessRecord, claim_token)) ||
        !append_claim_token(record_value_vgpr, record_value_high_vgpr, record_compare_vgpr)) {
      errors.emplace_back("ConSan MOI first-light probe could not materialize its bank claim");
      return std::nullopt;
    }
    const auto token_low_nonzero = instrumentation::build_v_cmp_ne_u32_vcc(
        scalar_positive_inline_u32(0), record_value_vgpr, arch);
    const auto token_high_nonzero = instrumentation::build_v_cmp_ne_u32_vcc(
        scalar_positive_inline_u32(0), record_value_high_vgpr, arch);
    if (!token_low_nonzero || !token_high_nonzero) {
      errors.emplace_back("ConSan MOI first-light probe could not validate its bank claim");
      return std::nullopt;
    }
    words.push_back(*token_low_nonzero);
    if (!sequence.emit_branch(token_ready_label, InstructionSequence::BranchKind::VccNonzero))
      return std::nullopt;
    words.push_back(*token_high_nonzero);
    if (!sequence.emit_branch(token_ready_label, InstructionSequence::BranchKind::VccNonzero) ||
        !sequence.emit_branch(saturation_label, InstructionSequence::BranchKind::Unconditional) ||
        !sequence.bind(token_ready_label)) {
      return std::nullopt;
    }
    words.push_back(build_v_mov_b32_e32(record_compare_vgpr, scalar_positive_inline_u32(0), arch));
    words.push_back(
        build_v_mov_b32_e32(record_compare_high_vgpr, scalar_positive_inline_u32(0), arch));
    const auto claim = instrumentation::build_flat_atomic_cmpswap_b64(
        record_address_vgpr, record_value_vgpr, record_value_vgpr,
        /*return_old_value=*/true, kRdna4ScopeDevice, arch);
    const auto claim_wait = instrumentation::build_s_wait_flat_load0(arch);
    if (!claim || !claim_wait) {
      errors.emplace_back("ConSan MOI first-light probe could not encode its publication claim");
      return std::nullopt;
    }
    words.insert(words.end(), claim->begin(), claim->end());
    words.push_back(*claim_wait);
    const auto prior_low_nonzero = instrumentation::build_v_cmp_ne_u32_vcc(
        scalar_positive_inline_u32(0), record_value_vgpr, arch);
    const auto prior_high_nonzero = instrumentation::build_v_cmp_ne_u32_vcc(
        scalar_positive_inline_u32(0), record_value_high_vgpr, arch);
    if (!prior_low_nonzero || !prior_high_nonzero) {
      errors.emplace_back("ConSan MOI first-light probe could not test its publication claim");
      return std::nullopt;
    }
    words.push_back(*prior_low_nonzero);
    if (!sequence.emit_branch(occupied_label, InstructionSequence::BranchKind::VccNonzero))
      return std::nullopt;
    words.push_back(*prior_high_nonzero);
    if (!sequence.emit_branch(occupied_label, InstructionSequence::BranchKind::VccNonzero))
      return std::nullopt;
    if (!append_store_u32_literal(words,
                                  base + offsetof(ConSanMoiReportHeader, access_record_count),
                                  record_count, *options.scratch_vgpr, arch) ||
        !append_atomic_fetch_add_one_u32(words,
                                         base + offsetof(ConSanMoiReportHeader, event_counter),
                                         record_compare_vgpr, record_address_vgpr, arch) ||
        !materialize_banked_record_address(access_record_base)) {
      errors.emplace_back("ConSan MOI first-light probe could not encode record stores");
      return std::nullopt;
    }
    // Bank selection reuses the high compare VGPR for the owner index. Recover
    // an overlapping guest address only after the final address
    // materialization so the recorded LDS range cannot accidentally become
    // the owner number.
    if (!reload_spilled_lds_byte_offset()) {
      errors.emplace_back("ConSan MOI first-light probe could not recover its spilled LDS address");
      return std::nullopt;
    }
    if (!record.store_vgpr(offsetof(ConSanMoiAccessRecord, event_index), record_compare_vgpr) ||
        !append_store_moi_report_dispatch_id_pair(
            record, options, offsetof(ConSanMoiAccessRecord, generation), arch,
            ConSanMoiLiteralDispatchIdPolicy::AnyArchitecture) ||
        !record.store_workgroup(offsetof(ConSanMoiAccessRecord, workgroup_x),
                                workgroup_sources.x) ||
        !record.store_workgroup(offsetof(ConSanMoiAccessRecord, workgroup_y),
                                workgroup_sources.y) ||
        !record.store_workgroup(offsetof(ConSanMoiAccessRecord, workgroup_z),
                                workgroup_sources.z) ||
        !record.store_sgpr(offsetof(ConSanMoiAccessRecord, lane_mask),
                           *options.moi_exec_save_sgpr) ||
        !record.store_sgpr(offsetof(ConSanMoiAccessRecord, lane_mask) + sizeof(uint32_t),
                           static_cast<uint16_t>(*options.moi_exec_save_sgpr + 1u)) ||
        (options.moi_owner_vgpr &&
         !record.store_vgpr(offsetof(ConSanMoiAccessRecord, wave_id), *options.moi_owner_vgpr)) ||
        (options.moi_persistent_owner_sgpr &&
         !record.store_sgpr(offsetof(ConSanMoiAccessRecord, wave_id),
                            *options.moi_persistent_owner_sgpr)) ||
        (options.moi_epoch_vgpr &&
         !record.store_vgpr(offsetof(ConSanMoiAccessRecord, epoch), *options.moi_epoch_vgpr)) ||
        !record.store_literal(offsetof(ConSanMoiAccessRecord, instruction_offset),
                              static_cast<uint32_t>(candidate.text_offset))) {
      errors.emplace_back("ConSan MOI first-light probe could not encode record stores");
      return std::nullopt;
    }
    if (derived_owner_vgpr) {
      words.insert(words.end(), derived_owner_words.begin(), derived_owner_words.end());
      if (!record.store_vgpr(offsetof(ConSanMoiAccessRecord, wave_id), *derived_owner_vgpr)) {
        errors.emplace_back("ConSan MOI first-light probe could not encode derived owner");
        return std::nullopt;
      }
    }
    if (private_epoch_offset) {
      const auto load_epoch =
          instrumentation::build_private_load_b32(record_value_vgpr, *private_epoch_offset, arch);
      const auto wait_epoch = instrumentation::build_s_wait_private_load0(arch);
      if (!load_epoch || !wait_epoch) {
        errors.emplace_back("ConSan MOI first-light probe could not load private epoch state");
        return std::nullopt;
      }
      words.insert(words.end(), load_epoch->begin(), load_epoch->end());
      words.push_back(*wait_epoch);
      if (!record.store_vgpr(offsetof(ConSanMoiAccessRecord, epoch), record_value_vgpr)) {
        errors.emplace_back("ConSan MOI first-light probe could not store private epoch state");
        return std::nullopt;
      }
    } else if (options.moi_persistent_epoch_sgpr) {
      words.push_back(
          build_v_mov_b32_e32(record_value_vgpr, *options.moi_persistent_epoch_sgpr, arch));
      if (!record.store_vgpr(offsetof(ConSanMoiAccessRecord, epoch), record_value_vgpr)) {
        errors.emplace_back("ConSan MOI first-light probe could not store scalar epoch state");
        return std::nullopt;
      }
    }

    const std::optional<uint16_t> effective_lds_byte_offset_vgpr =
        append_effective_range_offset(range, record_value_vgpr);
    if (!effective_lds_byte_offset_vgpr) {
      errors.emplace_back("ConSan MOI first-light probe could not encode LDS byte offset");
      return std::nullopt;
    }
    const ConSanMoiLdsCellRange static_range =
        consan_moi_lds_cell_range_for_bytes(range.static_byte_offset, range.byte_count);
    if (!record.store_vgpr(offsetof(ConSanMoiAccessRecord, lds_byte_offset),
                           *effective_lds_byte_offset_vgpr)) {
      errors.emplace_back("ConSan MOI first-light probe could not encode range offset");
      return std::nullopt;
    }
    const auto start_cell = instrumentation::build_v_lshrrev_b32(
        record_value_vgpr, scalar_positive_inline_u32(consan_moi_exact_shadow::granule_shift),
        *effective_lds_byte_offset_vgpr, arch);
    if (!start_cell) {
      errors.emplace_back("ConSan MOI first-light probe could not encode start cell");
      return std::nullopt;
    }
    words.push_back(*start_cell);
    if (!record.store_vgpr(offsetof(ConSanMoiAccessRecord, start_cell), record_value_vgpr) ||
        !record.store_literal(offsetof(ConSanMoiAccessRecord, lds_byte_count), range.byte_count) ||
        !record.store_literal(offsetof(ConSanMoiAccessRecord, cell_count),
                              static_range.cell_count)) {
      errors.emplace_back("ConSan MOI first-light probe could not encode range fields");
      return std::nullopt;
    }
    const auto payload_wait = instrumentation::build_s_wait_flat_store0(arch);
    const auto commit_address = instrumentation::build_v_add_u64_signed_i24(
        record_address_vgpr, offsetof(ConSanMoiAccessRecord, access_kind), arch);
    if (!payload_wait || !commit_address) {
      errors.emplace_back("ConSan MOI first-light probe could not drain record stores");
      return std::nullopt;
    }
    words.push_back(*payload_wait);
    words.insert(words.end(), commit_address->begin(), commit_address->end());
    const auto commit_value = instrumentation::build_v_mov_b32_literal(
        record_value_vgpr, static_cast<uint32_t>(kind), arch);
    const auto commit_expected = instrumentation::build_v_mov_b32_literal(
        record_value_high_vgpr, static_cast<uint32_t>(ConSanMoiShadowAccessKind::Empty), arch);
    const auto commit = instrumentation::build_flat_atomic_cmpswap_b32(
        record_address_vgpr, record_value_vgpr, record_value_vgpr,
        /*return_old_value=*/true, kRdna4ScopeDevice, arch);
    const auto commit_wait = instrumentation::build_s_wait_flat_load0(arch);
    if (!commit_value || !commit_expected || !commit || !commit_wait) {
      errors.emplace_back("ConSan MOI first-light probe could not commit its publication");
      return std::nullopt;
    }
    words.insert(words.end(), commit_value->begin(), commit_value->end());
    words.insert(words.end(), commit_expected->begin(), commit_expected->end());
    words.insert(words.end(), commit->begin(), commit->end());
    words.push_back(*commit_wait);
    const auto commit_collision = instrumentation::build_v_cmp_ne_u32_vcc(
        scalar_positive_inline_u32(static_cast<uint32_t>(ConSanMoiShadowAccessKind::Empty)),
        record_value_vgpr, arch);
    if (!commit_collision) {
      errors.emplace_back("ConSan MOI first-light probe could not validate its publication commit");
      return std::nullopt;
    }
    words.push_back(*commit_collision);
    if (!sequence.emit_branch(saturation_label, InstructionSequence::BranchKind::VccNonzero) ||
        !sequence.emit_branch(publication_done_label,
                              InstructionSequence::BranchKind::Unconditional) ||
        !sequence.bind(occupied_label)) {
      return std::nullopt;
    }

    // Decode the prior reversible token in place. This keeps the claimed
    // record address live and avoids rebuilding the complete bank-selection
    // expression on the occupied hot path.
    const auto low_xor_literal = instrumentation::build_v_mov_b32_literal(
        record_compare_vgpr, static_cast<uint32_t>(kConSanMoiRecordReplayClaimTokenXorMask), arch);
    const auto low_xor = instrumentation::build_v_xor_b32(
        record_value_vgpr, vector_source_vgpr(record_compare_vgpr), record_value_vgpr, arch);
    if (!low_xor_literal || !low_xor) {
      errors.emplace_back("ConSan MOI first-light probe could not compare its bank owner");
      return std::nullopt;
    }
    words.insert(words.end(), low_xor_literal->begin(), low_xor_literal->end());
    words.push_back(*low_xor);
    if (!append_moi_report_dispatch_id_word(words, options, record_compare_vgpr,
                                            /*high_word=*/false, arch,
                                            ConSanMoiLiteralDispatchIdPolicy::AnyArchitecture))
      return std::nullopt;
    const auto low_mismatch = instrumentation::build_v_cmp_ne_u32_vcc(
        vector_source_vgpr(record_compare_vgpr), record_value_vgpr, arch);
    if (!low_mismatch)
      return std::nullopt;
    words.push_back(*low_mismatch);
    if (!sequence.emit_branch(saturation_label, InstructionSequence::BranchKind::VccNonzero))
      return std::nullopt;

    const auto high_xor_literal = instrumentation::build_v_mov_b32_literal(
        record_compare_vgpr, static_cast<uint32_t>(kConSanMoiRecordReplayClaimTokenXorMask >> 32u),
        arch);
    const auto high_xor = instrumentation::build_v_xor_b32(record_value_high_vgpr,
                                                           vector_source_vgpr(record_compare_vgpr),
                                                           record_value_high_vgpr, arch);
    if (!high_xor_literal || !high_xor) {
      errors.emplace_back("ConSan MOI first-light probe could not compare its bank owner");
      return std::nullopt;
    }
    words.insert(words.end(), high_xor_literal->begin(), high_xor_literal->end());
    words.push_back(*high_xor);
    if (!append_moi_report_dispatch_id_word(words, options, record_compare_vgpr,
                                            /*high_word=*/true, arch,
                                            ConSanMoiLiteralDispatchIdPolicy::AnyArchitecture))
      return std::nullopt;
    const auto high_mismatch = instrumentation::build_v_cmp_ne_u32_vcc(
        vector_source_vgpr(record_compare_vgpr), record_value_high_vgpr, arch);
    const auto access_kind_address = instrumentation::build_v_add_u64_signed_i24(
        record_address_vgpr, offsetof(ConSanMoiAccessRecord, access_kind), arch);
    if (!high_mismatch || !access_kind_address) {
      errors.emplace_back("ConSan MOI first-light probe could not compare its bank owner");
      return std::nullopt;
    }
    words.push_back(*high_mismatch);
    if (!sequence.emit_branch(saturation_label, InstructionSequence::BranchKind::VccNonzero)) {
      return std::nullopt;
    }
    words.insert(words.end(), access_kind_address->begin(), access_kind_address->end());

    // Use a no-op atomic compare-and-swap as an acquire-style read of the
    // access_kind commit. A zero result means the winning publisher is still
    // filling the payload; fail closed instead of inspecting partial
    // workgroup coordinates.
    words.push_back(build_v_mov_b32_e32(record_value_vgpr, scalar_positive_inline_u32(0), arch));
    words.push_back(
        build_v_mov_b32_e32(record_value_high_vgpr, scalar_positive_inline_u32(0), arch));
    const auto read_commit = instrumentation::build_flat_atomic_cmpswap_b32(
        record_address_vgpr, record_value_vgpr, record_value_vgpr,
        /*return_old_value=*/true, kRdna4ScopeDevice, arch);
    const auto read_commit_wait = instrumentation::build_s_wait_flat_load0(arch);
    const auto publication_incomplete = instrumentation::build_v_cmp_eq_u32_vcc(
        scalar_positive_inline_u32(static_cast<uint32_t>(ConSanMoiShadowAccessKind::Empty)),
        record_value_vgpr, arch);
    const auto publication_kind_mismatch = instrumentation::build_v_cmp_ne_u32_vcc(
        scalar_positive_inline_u32(static_cast<uint32_t>(kind)), record_value_vgpr, arch);
    if (!read_commit || !read_commit_wait || !publication_incomplete ||
        !publication_kind_mismatch) {
      errors.emplace_back("ConSan MOI first-light probe could not observe its publication commit");
      return std::nullopt;
    }
    words.insert(words.end(), read_commit->begin(), read_commit->end());
    words.push_back(*read_commit_wait);
    words.push_back(*publication_incomplete);
    if (!sequence.emit_branch(saturation_label, InstructionSequence::BranchKind::VccNonzero)) {
      return std::nullopt;
    }
    words.push_back(*publication_kind_mismatch);
    const auto record_base_address = instrumentation::build_v_add_u64_signed_i24(
        record_address_vgpr, -static_cast<int32_t>(offsetof(ConSanMoiAccessRecord, access_kind)),
        arch);
    if (!record_base_address ||
        !sequence.emit_branch(saturation_label, InstructionSequence::BranchKind::VccNonzero)) {
      return std::nullopt;
    }
    words.insert(words.end(), record_base_address->begin(), record_base_address->end());

    const auto reject_workgroup_mismatch = [&](uint32_t offset,
                                               const ConSanMoiWorkgroupSource &source) {
      if (!record.load(offset, record_value_vgpr))
        return false;
      if (source.scalar_src) {
        if (!append_workgroup_source_value(words, source, record_compare_vgpr, arch))
          return false;
      } else {
        words.push_back(
            build_v_mov_b32_e32(record_compare_vgpr, scalar_positive_inline_u32(0), arch));
      }
      const auto mismatch = instrumentation::build_v_cmp_ne_u32_vcc(
          vector_source_vgpr(record_compare_vgpr), record_value_vgpr, arch);
      if (!mismatch)
        return false;
      words.push_back(*mismatch);
      return sequence.emit_branch(saturation_label, InstructionSequence::BranchKind::VccNonzero);
    };
    if (!reject_workgroup_mismatch(offsetof(ConSanMoiAccessRecord, workgroup_x),
                                   workgroup_sources.x) ||
        !reject_workgroup_mismatch(offsetof(ConSanMoiAccessRecord, workgroup_y),
                                   workgroup_sources.y) ||
        !reject_workgroup_mismatch(offsetof(ConSanMoiAccessRecord, workgroup_z),
                                   workgroup_sources.z) ||
        !sequence.emit_branch(publication_done_label,
                              InstructionSequence::BranchKind::Unconditional)) {
      errors.emplace_back("ConSan MOI first-light probe could not qualify its retained workgroup");
      return std::nullopt;
    }

    if (!sequence.bind(saturation_label) ||
        !append_atomic_or_u32_literal(words, base + offsetof(ConSanMoiReportHeader, flags),
                                      kConSanMoiReportFlagRecordReplayBankSaturated,
                                      record_address_vgpr, arch) ||
        !sequence.bind(publication_done_label)) {
      errors.emplace_back("ConSan MOI first-light publication fast path is invalid");
      return std::nullopt;
    }
  }

  const auto restore_exec =
      instrumentation::build_s_mov_b64(kRdna4ExecLo, *options.moi_exec_save_sgpr, arch);
  if (!restore_exec) {
    errors.emplace_back("ConSan MOI first-light probe could not restore EXEC");
    return std::nullopt;
  }
  if (!sequence.bind(restore_exec_label))
    return std::nullopt;
  words.push_back(*restore_exec);
  if (!append_restore_moi_special_state(words, options, arch)) {
    errors.emplace_back("ConSan MOI first-light probe could not restore VCC/SCC");
    return std::nullopt;
  }

  // Static first-light records do not consume the guest result. Publish them
  // before a load so that the record scratch window may overlap its destination
  // VGPRs. When the load also overwrites its address VGPR, the saved address
  // above remains outside that window until the displaced instruction executes.
  if (!options.moi_dynamic_access_records)
    append_guest_access();

  if (!sequence.resolve_branches(arch)) {
    errors.emplace_back("ConSan MOI first-light local branch is out of reach");
    return std::nullopt;
  }

  return words;
}

[[nodiscard]] bool append_add_shifted_vgpr_field(std::vector<uint32_t> &words,
                                                 uint16_t destination_vgpr, uint16_t field_vgpr,
                                                 uint16_t shift, uint32_t mask, uint16_t tmp_vgpr,
                                                 rj_code_arch_t arch) {
  const auto mask_words =
      instrumentation::build_v_and_b32_literal(tmp_vgpr, mask, field_vgpr, arch);
  const auto shift_word = instrumentation::build_v_lshlrev_b32(
      tmp_vgpr, scalar_positive_inline_u32(shift), tmp_vgpr, arch);
  const auto add_word = instrumentation::build_v_add_u32(
      destination_vgpr, vector_source_vgpr(destination_vgpr), tmp_vgpr, arch);
  InstructionSequence sequence(words);
  return sequence.emit_all(mask_words, shift_word, add_word);
}

[[nodiscard]] bool append_add_literal_field(std::vector<uint32_t> &words, uint16_t destination_vgpr,
                                            uint32_t value, uint16_t tmp_vgpr,
                                            rj_code_arch_t arch) {
  if (value == 0)
    return true;
  const auto mov_value = instrumentation::build_v_mov_b32_literal(tmp_vgpr, value, arch);
  const auto add_word = instrumentation::build_v_add_u32(
      destination_vgpr, vector_source_vgpr(destination_vgpr), tmp_vgpr, arch);
  InstructionSequence sequence(words);
  return sequence.emit_all(mov_value, add_word);
}

[[nodiscard]] bool append_inline_shadow_owner_field(std::vector<uint32_t> &words,
                                                    const ConSanOptions &options, uint16_t low_vgpr,
                                                    uint16_t tmp_vgpr, rj_code_arch_t arch,
                                                    std::optional<uint32_t> private_owner_offset,
                                                    std::optional<uint16_t> private_owner_shift,
                                                    std::vector<std::string> &errors) {
  if (options.moi_owner_vgpr) {
    return append_add_shifted_vgpr_field(words, low_vgpr, *options.moi_owner_vgpr,
                                         consan_moi_exact_shadow::owner_shift,
                                         consan_moi_exact_shadow::max_owner, tmp_vgpr, arch);
  }
  if (options.moi_persistent_owner_sgpr) {
    words.push_back(build_v_mov_b32_e32(tmp_vgpr, *options.moi_persistent_owner_sgpr, arch));
    return append_add_shifted_vgpr_field(words, low_vgpr, tmp_vgpr,
                                         consan_moi_exact_shadow::owner_shift,
                                         consan_moi_exact_shadow::max_owner, tmp_vgpr, arch);
  }
  if (!options.automatic_moi_private_epoch) {
    errors.emplace_back("ConSan MOI inline-shadow probe has no persistent owner representation");
    return false;
  }

  switch (options.moi_owner_source) {
  case ConSanMoiOwnerSource::WorkitemId: {
    if (!private_owner_offset || !private_owner_shift) {
      errors.emplace_back(
          "ConSan MOI private owner has no persistent descriptor-compatible workitem-id state");
      return false;
    }
    const auto load =
        instrumentation::build_private_load_b32(tmp_vgpr, *private_owner_offset, arch);
    const auto wait = instrumentation::build_s_wait_private_load0(arch);
    const auto owner = instrumentation::build_v_lshrrev_b32(
        tmp_vgpr, scalar_positive_inline_u32(*private_owner_shift), tmp_vgpr, arch);
    if (!load || !wait || !owner) {
      errors.emplace_back(
          "ConSan MOI inline-shadow probe could not load its persistent workitem-id owner");
      return false;
    }
    words.insert(words.end(), load->begin(), load->end());
    words.push_back(*wait);
    words.push_back(*owner);
    break;
  }
  case ConSanMoiOwnerSource::HwId: {
    if (!options.moi_owner_sgpr) {
      errors.emplace_back("ConSan MOI private owner hw_id source requires "
                          "RJ_CONSAN_MOI_OWNER_SGPR");
      return false;
    }
    const auto hwreg = build_hwreg_imm(kGfx12HwRegHwId1, /*offset=*/0, kGfx12HwIdOwnerBits);
    const auto get_hw_id =
        hwreg ? build_s_getreg_b32(*options.moi_owner_sgpr, *hwreg, arch) : std::nullopt;
    if (!get_hw_id) {
      errors.emplace_back("ConSan MOI inline-shadow probe could not derive owner from HW_ID1");
      return false;
    }
    words.push_back(*get_hw_id);
    words.push_back(build_v_mov_b32_e32(tmp_vgpr, *options.moi_owner_sgpr, arch));
    break;
  }
  }

  return append_add_shifted_vgpr_field(words, low_vgpr, tmp_vgpr,
                                       consan_moi_exact_shadow::owner_shift,
                                       consan_moi_exact_shadow::max_owner, tmp_vgpr, arch);
}

[[nodiscard]] bool append_inline_shadow_epoch_field(std::vector<uint32_t> &words,
                                                    const ConSanOptions &options,
                                                    std::optional<uint32_t> private_epoch_offset,
                                                    uint16_t low_vgpr, uint16_t tmp_vgpr,
                                                    rj_code_arch_t arch,
                                                    std::vector<std::string> &errors) {
  if (options.moi_epoch_vgpr) {
    return append_add_shifted_vgpr_field(words, low_vgpr, *options.moi_epoch_vgpr,
                                         consan_moi_exact_shadow::epoch_shift,
                                         consan_moi_exact_shadow::max_epoch, tmp_vgpr, arch);
  }
  if (options.moi_persistent_epoch_sgpr) {
    words.push_back(build_v_mov_b32_e32(tmp_vgpr, *options.moi_persistent_epoch_sgpr, arch));
    return append_add_shifted_vgpr_field(words, low_vgpr, tmp_vgpr,
                                         consan_moi_exact_shadow::epoch_shift,
                                         consan_moi_exact_shadow::max_epoch, tmp_vgpr, arch);
  }
  if (!options.automatic_moi_private_epoch || !private_epoch_offset) {
    errors.emplace_back("ConSan MOI inline-shadow probe has no persistent epoch representation");
    return false;
  }
  const auto load = instrumentation::build_private_load_b32(tmp_vgpr, *private_epoch_offset, arch);
  const auto wait = instrumentation::build_s_wait_private_load0(arch);
  InstructionSequence sequence(words);
  if (!sequence.emit_all(load, wait)) {
    errors.emplace_back("ConSan MOI inline-shadow probe could not load private epoch state");
    return false;
  }
  return append_add_shifted_vgpr_field(words, low_vgpr, tmp_vgpr,
                                       consan_moi_exact_shadow::epoch_shift,
                                       consan_moi_exact_shadow::max_epoch, tmp_vgpr, arch);
}

[[nodiscard]] bool is_inline_shadow_access_candidate(const ConSanMoiCandidate &candidate,
                                                     std::span<const uint8_t> bytes,
                                                     rj_code_arch_t arch) {
  const bool native_lds = candidate.source == ConSanMoiCandidateSource::NativeLds;
  const bool normalized_flat = (candidate.source == ConSanMoiCandidateSource::FlatGroup ||
                                candidate.source == ConSanMoiCandidateSource::FlatMaybeGroup) &&
                               is_first_light_flat_candidate(candidate, arch);
  if (!native_lds && !normalized_flat)
    return false;
  if (candidate.kind != ConSanLdsAccessKind::Read && candidate.kind != ConSanLdsAccessKind::Write &&
      candidate.kind != ConSanLdsAccessKind::Atomic)
    return false;
  if (candidate.size == 0 || candidate.size % sizeof(uint32_t) != 0)
    return false;
  const std::optional<std::vector<ConSanMoiAccessRange>> ranges =
      candidate_access_ranges(bytes, candidate, arch);
  if (!ranges || ranges->empty())
    return false;
  for (const ConSanMoiAccessRange &range : *ranges) {
    const ConSanMoiLdsCellRange cell_range =
        consan_moi_lds_cell_range_for_bytes(range.static_byte_offset, range.byte_count);
    if (cell_range.cell_count == 0)
      return false;
  }
  return true;
}

[[nodiscard]] std::vector<const ConSanMoiCandidate *>
find_inline_shadow_access_candidates(const ConSanResult &result, std::span<const uint8_t> bytes,
                                     rj_code_arch_t arch) {
  std::vector<const ConSanMoiCandidate *> candidates;
  for (const ConSanMoiCandidate &candidate : result.moi_candidates) {
    if (is_inline_shadow_access_candidate(candidate, bytes, arch))
      candidates.push_back(&candidate);
  }
  return candidates;
}

[[nodiscard]] bool append_extract_exact_shadow_field(std::vector<uint32_t> &words,
                                                     uint16_t destination_vgpr,
                                                     uint16_t packed_vgpr, uint16_t shift,
                                                     uint32_t mask, rj_code_arch_t arch) {
  const auto shifted = instrumentation::build_v_lshrrev_b32(
      destination_vgpr, scalar_positive_inline_u32(shift), packed_vgpr, arch);
  const auto masked =
      instrumentation::build_v_and_b32_literal(destination_vgpr, mask, destination_vgpr, arch);
  InstructionSequence sequence(words);
  return sequence.emit_all(shifted, masked);
}

[[nodiscard]] bool
append_extract_exact_shadow_generation(std::vector<uint32_t> &words, uint16_t destination_vgpr,
                                       uint16_t packed_low_vgpr, uint16_t packed_high_vgpr,
                                       uint16_t temporary_vgpr, rj_code_arch_t arch) {
  constexpr uint32_t kLowBits = 32u - consan_moi_exact_shadow::generation_shift;
  constexpr uint32_t kHighBits = consan_moi_exact_shadow::generation_bits - kLowBits;
  // Read the high fragment first: the destination is intentionally allowed to
  // alias packed_high_vgpr in the compact diagnostic register assignment.
  const auto high = instrumentation::build_v_and_b32_literal(
      temporary_vgpr, (uint32_t{1} << kHighBits) - 1u, packed_high_vgpr, arch);
  const auto low = instrumentation::build_v_lshrrev_b32(
      destination_vgpr, scalar_positive_inline_u32(consan_moi_exact_shadow::generation_shift),
      packed_low_vgpr, arch);
  const auto shift_high = instrumentation::build_v_lshlrev_b32(
      temporary_vgpr, scalar_positive_inline_u32(kLowBits), temporary_vgpr, arch);
  const auto combine = instrumentation::build_v_add_u32(
      destination_vgpr, vector_source_vgpr(destination_vgpr), temporary_vgpr, arch);
  InstructionSequence sequence(words);
  return sequence.emit_all(high, low, shift_high, combine);
}

[[nodiscard]] bool
append_add_exact_shadow_generation(std::vector<uint32_t> &words, uint16_t packed_low_vgpr,
                                   uint16_t packed_high_vgpr, uint16_t generation_vgpr,
                                   uint16_t temporary_vgpr, rj_code_arch_t arch) {
  constexpr uint32_t kLowBits = 32u - consan_moi_exact_shadow::generation_shift;
  constexpr uint32_t kHighBits = consan_moi_exact_shadow::generation_bits - kLowBits;
  const auto low = instrumentation::build_v_and_b32_literal(
      temporary_vgpr, (uint32_t{1} << kLowBits) - 1u, generation_vgpr, arch);
  const auto shift_low = instrumentation::build_v_lshlrev_b32(
      temporary_vgpr, scalar_positive_inline_u32(consan_moi_exact_shadow::generation_shift),
      temporary_vgpr, arch);
  const auto add_low = instrumentation::build_v_add_u32(
      packed_low_vgpr, vector_source_vgpr(packed_low_vgpr), temporary_vgpr, arch);
  const auto high = instrumentation::build_v_lshrrev_b32(
      temporary_vgpr, scalar_positive_inline_u32(kLowBits), generation_vgpr, arch);
  const auto mask_high = instrumentation::build_v_and_b32_literal(
      temporary_vgpr, (uint32_t{1} << kHighBits) - 1u, temporary_vgpr, arch);
  const auto add_high = instrumentation::build_v_add_u32(
      packed_high_vgpr, vector_source_vgpr(packed_high_vgpr), temporary_vgpr, arch);
  InstructionSequence sequence(words);
  return sequence.emit_all(low, shift_low, add_low, high, mask_high, add_high);
}

[[nodiscard]] bool append_inline_workgroup_key(std::vector<uint32_t> &words,
                                               const ConSanMoiWorkgroupSources &sources,
                                               const ConSanOptions &options, uint16_t key_vgpr,
                                               uint16_t coordinate_vgpr, uint16_t value_vgpr,
                                               uint16_t original_exec_save_offset,
                                               rj_code_arch_t arch);

[[nodiscard]] bool append_inline_acquired_token_slot_address(
    std::vector<uint32_t> &words, uint64_t table_base, uint32_t table_capacity,
    uint16_t workgroup_key_vgpr, uint16_t consumer_owner_vgpr, uint16_t producer_owner_vgpr,
    uint16_t slot_address_vgpr, uint16_t hash_vgpr, uint16_t temporary_vgpr, bool release_sequence,
    rj_code_arch_t arch);

[[nodiscard]] bool append_inline_atomic_slot_address(std::vector<uint32_t> &words,
                                                     uint64_t table_base, uint32_t table_capacity,
                                                     uint16_t atomic_address_vgpr,
                                                     uint16_t scratch_vgpr, rj_code_arch_t arch);

[[nodiscard]] bool
append_inline_atomic_snapshot_address(std::vector<uint32_t> &words, uint64_t table_base,
                                      uint32_t table_capacity, uint16_t atomic_address_vgpr,
                                      uint16_t snapshot_address_vgpr, uint16_t hash_vgpr,
                                      uint16_t temporary_vgpr, rj_code_arch_t arch);
