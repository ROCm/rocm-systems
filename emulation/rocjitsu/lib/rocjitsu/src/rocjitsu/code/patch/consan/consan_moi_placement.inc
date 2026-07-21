struct MoiLocalNopIsland {
  uint64_t text_offset = 0;
  bool claimed = false;
};

constexpr uint32_t kMoiLocalIndirectIslandWords = 8u;

class MoiLocalNopIslandAllocator {
public:
  MoiLocalNopIslandAllocator(std::span<const uint8_t> text, const ConSanResult &result,
                             rj_code_arch_t arch, uint32_t island_words,
                             std::span<const ConSanPreappliedReservedRange> reserved_ranges = {})
      : text_(text), nop_(build_s_nop(0, arch)), island_words_(island_words) {
    if (island_words == 0)
      return;
    struct OwnerRange {
      uint64_t entry = 0;
      uint64_t code_end = 0;
    };
    std::vector<OwnerRange> owners;
    owners.reserve(result.kernels.size() + result.functions.size());
    const auto append_owner = [&](uint64_t entry, uint64_t code_size) {
      if (code_size == 0 || entry >= text.size())
        return;
      owners.push_back({entry, entry + std::min<uint64_t>(code_size, text.size() - entry)});
    };
    for (const ConSanKernelInfo &kernel : result.kernels) {
      if (kernel.has_text_range)
        append_owner(kernel.entry_text_offset, kernel.code_size);
    }
    for (const ConSanFunctionInfo &function : result.functions)
      append_owner(function.entry_text_offset, function.code_size);
    std::ranges::sort(owners, {}, &OwnerRange::entry);
    std::vector<OwnerRange> unique;
    for (const OwnerRange &owner : owners) {
      if (!unique.empty() && unique.back().entry == owner.entry) {
        unique.back().code_end = std::max(unique.back().code_end, owner.code_end);
      } else {
        unique.push_back(owner);
      }
    }

    const uint64_t island_bytes = static_cast<uint64_t>(island_words) * sizeof(uint32_t);
    std::vector<std::pair<uint64_t, uint64_t>> occupied_ranges;
    occupied_ranges.reserve(result.patches.size() * 2u + reserved_ranges.size());
    for (const ConSanPatchInfo &patch : result.patches) {
      occupied_ranges.emplace_back(
          patch.anchor_offset, patch.anchor_offset + std::max<uint32_t>(patch.original_size, 1u));
      if (patch.trampoline_size != 0u) {
        occupied_ranges.emplace_back(patch.trampoline_offset,
                                     patch.trampoline_offset + patch.trampoline_size);
      }
    }
    for (const ConSanPreappliedReservedRange &reserved : reserved_ranges) {
      if (reserved.size != 0u)
        occupied_ranges.emplace_back(reserved.text_offset, reserved.text_offset + reserved.size);
    }
    std::ranges::sort(occupied_ranges);
    std::vector<std::pair<uint64_t, uint64_t>> merged_occupied_ranges;
    for (const auto &range : occupied_ranges) {
      if (merged_occupied_ranges.empty() || merged_occupied_ranges.back().second < range.first) {
        merged_occupied_ranges.push_back(range);
      } else {
        merged_occupied_ranges.back().second =
            std::max(merged_occupied_ranges.back().second, range.second);
      }
    }
    const auto overlaps_patch = [&](uint64_t begin, uint64_t end) {
      const auto it = std::ranges::lower_bound(merged_occupied_ranges, end, {},
                                               &std::pair<uint64_t, uint64_t>::first);
      return it != merged_occupied_ranges.begin() && std::prev(it)->second > begin;
    };
    for (size_t owner_index = 0; owner_index < unique.size(); ++owner_index) {
      uint64_t cursor = util::align_up(unique[owner_index].code_end, uint64_t{4});
      const uint64_t gap_end =
          owner_index + 1u < unique.size() ? unique[owner_index + 1u].entry : text.size();
      while (cursor + island_bytes <= gap_end) {
        bool all_nops = !overlaps_patch(cursor, cursor + island_bytes);
        for (uint64_t offset = 0; all_nops && offset < island_bytes; offset += sizeof(uint32_t)) {
          uint32_t word = 0;
          std::memcpy(&word, text.data() + cursor + offset, sizeof(word));
          all_nops = word == nop_;
        }
        if (all_nops) {
          available_offsets_.insert(cursor);
          cursor += island_bytes;
        } else {
          cursor += sizeof(uint32_t);
        }
      }
    }
  }

  [[nodiscard]] std::optional<uint64_t> claim_reachable(uint64_t anchor_offset,
                                                        uint32_t claim_words = 0u) {
    claim_words = claim_words == 0u ? island_words_ : claim_words;
    auto best = available_offsets_.end();
    uint64_t best_distance = std::numeric_limits<uint64_t>::max();
    const auto [first_reachable, last_reachable] = reachable_target_bounds(anchor_offset);
    for (auto candidate = available_offsets_.lower_bound(first_reachable);
         candidate != available_offsets_.end() && *candidate <= last_reachable; ++candidate) {
      if (!claim_fits(*candidate, claim_words) ||
          !compute_sopp_branch_simm16(anchor_offset, *candidate))
        continue;
      const uint64_t distance =
          anchor_offset < *candidate ? *candidate - anchor_offset : anchor_offset - *candidate;
      if (distance < best_distance) {
        best = candidate;
        best_distance = distance;
      }
    }
    if (best == available_offsets_.end())
      return std::nullopt;
    const uint64_t offset = *best;
    retire_overlaps(offset, claim_words);
    return offset;
  }

  [[nodiscard]] bool has_reachable(uint64_t anchor_offset, uint32_t claim_words = 0u) const {
    claim_words = claim_words == 0u ? island_words_ : claim_words;
    const auto [first_reachable, last_reachable] = reachable_target_bounds(anchor_offset);
    for (auto candidate = available_offsets_.lower_bound(first_reachable);
         candidate != available_offsets_.end() && *candidate <= last_reachable; ++candidate) {
      if (claim_fits(*candidate, claim_words) &&
          compute_sopp_branch_simm16(anchor_offset, *candidate))
        return true;
    }
    return false;
  }

  [[nodiscard]] std::optional<uint64_t>
  claim_reachable_from_all(std::span<const uint64_t> anchor_offsets, uint32_t claim_words = 0u) {
    if (anchor_offsets.empty())
      return std::nullopt;
    claim_words = claim_words == 0u ? island_words_ : claim_words;
    uint64_t first_reachable = 0u;
    uint64_t last_reachable = std::numeric_limits<uint64_t>::max();
    for (uint64_t anchor_offset : anchor_offsets) {
      const auto [first, last] = reachable_target_bounds(anchor_offset);
      first_reachable = std::max(first_reachable, first);
      last_reachable = std::min(last_reachable, last);
    }
    if (first_reachable > last_reachable)
      return std::nullopt;
    auto best = available_offsets_.end();
    uint64_t best_max_distance = std::numeric_limits<uint64_t>::max();
    for (auto candidate = available_offsets_.lower_bound(first_reachable);
         candidate != available_offsets_.end() && *candidate <= last_reachable; ++candidate) {
      if (!claim_fits(*candidate, claim_words))
        continue;
      uint64_t max_distance = 0;
      bool reachable = true;
      for (uint64_t anchor_offset : anchor_offsets) {
        if (!compute_sopp_branch_simm16(anchor_offset, *candidate)) {
          reachable = false;
          break;
        }
        max_distance =
            std::max(max_distance, anchor_offset < *candidate ? *candidate - anchor_offset
                                                              : anchor_offset - *candidate);
      }
      if (reachable && max_distance < best_max_distance) {
        best = candidate;
        best_max_distance = max_distance;
      }
    }
    if (best == available_offsets_.end())
      return std::nullopt;
    const uint64_t offset = *best;
    retire_overlaps(offset, claim_words);
    return offset;
  }

  void reserve(uint64_t offset, uint32_t words) { retire_overlaps(offset, words); }

private:
  [[nodiscard]] static constexpr std::pair<uint64_t, uint64_t>
  reachable_target_bounds(uint64_t anchor_offset) {
    constexpr uint64_t kBackwardReach =
        static_cast<uint64_t>(-static_cast<int64_t>(std::numeric_limits<int16_t>::min())) *
        sizeof(uint32_t);
    constexpr uint64_t kForwardReach =
        static_cast<uint64_t>(std::numeric_limits<int16_t>::max()) * sizeof(uint32_t);
    const uint64_t branch_base =
        anchor_offset > std::numeric_limits<uint64_t>::max() - sizeof(uint32_t)
            ? std::numeric_limits<uint64_t>::max()
            : anchor_offset + sizeof(uint32_t);
    const uint64_t first = branch_base > kBackwardReach ? branch_base - kBackwardReach : 0u;
    const uint64_t last = branch_base > std::numeric_limits<uint64_t>::max() - kForwardReach
                              ? std::numeric_limits<uint64_t>::max()
                              : branch_base + kForwardReach;
    return {first, last};
  }

  [[nodiscard]] bool claim_fits(uint64_t offset, uint32_t claim_words) const {
    const uint64_t claim_bytes = static_cast<uint64_t>(claim_words) * sizeof(uint32_t);
    if (offset > text_.size() || claim_bytes > text_.size() - offset)
      return false;
    for (uint64_t byte = 0; byte < claim_bytes; byte += sizeof(uint32_t)) {
      uint32_t word = 0;
      std::memcpy(&word, text_.data() + offset + byte, sizeof(word));
      if (word != nop_)
        return false;
    }
    return true;
  }

  void retire_overlaps(uint64_t offset, uint32_t claim_words) {
    const uint64_t claim_end = offset + static_cast<uint64_t>(claim_words) * sizeof(uint32_t);
    const uint64_t island_bytes = static_cast<uint64_t>(island_words_) * sizeof(uint32_t);
    const uint64_t first_possible_overlap =
        offset >= island_bytes ? offset - island_bytes + 1u : 0u;
    auto candidate = available_offsets_.lower_bound(first_possible_overlap);
    while (candidate != available_offsets_.end() && *candidate < claim_end) {
      if (*candidate + island_bytes > offset) {
        candidate = available_offsets_.erase(candidate);
      } else {
        ++candidate;
      }
    }
  }

  std::span<const uint8_t> text_;
  uint32_t nop_ = 0;
  uint32_t island_words_ = 0;
  std::set<uint64_t> available_offsets_;
};

void append_nop_padding_to_alignment(std::vector<uint8_t> &bytes, uint64_t alignment,
                                     rj_code_arch_t arch) {
  if (alignment == 0)
    return;
  while (bytes.size() % alignment != 0)
    append_word_bytes(bytes, build_s_nop(0, arch));
}

[[nodiscard]] bool write_word_bytes(std::vector<uint8_t> &bytes, uint64_t offset, uint32_t word) {
  if (offset > bytes.size() || sizeof(word) > bytes.size() - offset)
    return false;
  std::memcpy(bytes.data() + offset, &word, sizeof(word));
  return true;
}

[[nodiscard]] uint32_t moi_descriptor_wavefront_size(const KD &desc) {
  const bool wave32 = AMDHSA_BITS_GET(desc.kernel_code_properties,
                                      kd::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32);
  return wave32 ? 32 : 64;
}

[[nodiscard]] uint32_t moi_descriptor_user_sgpr_count(const KD &desc) {
  return AMDHSA_BITS_GET(desc.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT);
}

[[nodiscard]] uint16_t moi_descriptor_system_sgpr_count(const KD &desc) {
  const uint32_t rsrc2 = desc.compute_pgm_rsrc2;
  return static_cast<uint16_t>(
      (AMDHSA_BITS_GET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X) != 0 ? 1u : 0u) +
      (AMDHSA_BITS_GET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y) != 0 ? 1u : 0u) +
      (AMDHSA_BITS_GET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z) != 0 ? 1u : 0u) +
      (AMDHSA_BITS_GET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_INFO) != 0 ? 1u : 0u));
}

[[nodiscard]] uint16_t moi_descriptor_dispatch_id_prefix_sgpr_count(const KD &desc) {
  const uint16_t properties = desc.kernel_code_properties;
  return consan_moi_amdhsa_dispatch_id_prefix_sgpr_count(
      AMDHSA_BITS_GET(properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER) != 0,
      AMDHSA_BITS_GET(properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR) != 0,
      AMDHSA_BITS_GET(properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR) != 0,
      AMDHSA_BITS_GET(properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR) != 0);
}

[[nodiscard]] bool moi_descriptor_has_kernarg_preload(const KD &desc) {
  return AMDHSA_BITS_GET(desc.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_LENGTH) != 0;
}

[[nodiscard]] ConSanMoiDispatchIdPreloadPlan
moi_descriptor_dispatch_id_preload_plan(const KD &desc, rj_code_arch_t arch) {
  const uint16_t properties = desc.kernel_code_properties;
  const uint16_t original_user_count = static_cast<uint16_t>(moi_descriptor_user_sgpr_count(desc));
  const uint16_t system_count = moi_descriptor_system_sgpr_count(desc);
  const bool dispatch_enabled =
      AMDHSA_BITS_GET(properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID) != 0;
  const uint16_t prefix = moi_descriptor_dispatch_id_prefix_sgpr_count(desc);
  const uint16_t named_suffix = static_cast<uint16_t>(
      (AMDHSA_BITS_GET(properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_FLAT_SCRATCH_INIT) != 0
           ? 2u
           : 0u) +
      (AMDHSA_BITS_GET(properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE) != 0
           ? 1u
           : 0u) +
      AMDHSA_BITS_GET(desc.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_LENGTH));
  const uint32_t minimum_declared_user_count =
      static_cast<uint32_t>(prefix) + (dispatch_enabled ? 2u : 0u) + named_suffix;
  if (minimum_declared_user_count > original_user_count) {
    ConSanMoiDispatchIdPreloadPlan invalid;
    invalid.support = ConSanMoiDispatchIdPreloadSupport::InvalidDispatchPosition;
    invalid.dispatch_id_sgpr = prefix;
    invalid.original_user_sgpr_count = original_user_count;
    invalid.expanded_user_sgpr_count = original_user_count;
    invalid.system_sgpr_count = system_count;
    return invalid;
  }
  const uint16_t user_sgpr_initialization_limit = arch == ROCJITSU_CODE_ARCH_GFX1250 ? 32u : 16u;
  return consan_moi_plan_dispatch_id_preload(original_user_count, system_count, prefix,
                                             dispatch_enabled, kMaxSgprs,
                                             user_sgpr_initialization_limit);
}

[[nodiscard]] std::optional<uint16_t> moi_descriptor_workgroup_id_sgpr(const KD &desc,
                                                                       uint32_t dimension) {
  const uint32_t rsrc2 = desc.compute_pgm_rsrc2;
  const bool enabled[3] = {
      AMDHSA_BITS_GET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X) != 0,
      AMDHSA_BITS_GET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y) != 0,
      AMDHSA_BITS_GET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z) != 0,
  };

  uint32_t sgpr = moi_descriptor_user_sgpr_count(desc);
  for (uint32_t i = 0; i < 3; ++i) {
    if (!enabled[i])
      continue;
    if (i == dimension)
      return static_cast<uint16_t>(sgpr);
    ++sgpr;
  }
  return std::nullopt;
}

[[nodiscard]] bool moi_descriptor_enables_workgroup_id(const KD &desc, uint32_t dimension) {
  switch (dimension) {
  case 0:
    return AMDHSA_BITS_GET(desc.compute_pgm_rsrc2,
                           kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X) != 0;
  case 1:
    return AMDHSA_BITS_GET(desc.compute_pgm_rsrc2,
                           kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y) != 0;
  case 2:
    return AMDHSA_BITS_GET(desc.compute_pgm_rsrc2,
                           kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z) != 0;
  default:
    return false;
  }
}

[[nodiscard]] std::optional<ConSanMoiWorkgroupSources>
moi_descriptor_workgroup_sources(std::span<const uint8_t> image, uint64_t descriptor_file_offset,
                                 rj_code_arch_t arch, std::vector<std::string> &errors,
                                 bool uses_gfx1250_cluster_workgroup_id = false) {
  if (descriptor_file_offset > image.size() || sizeof(KD) > image.size() - descriptor_file_offset) {
    errors.emplace_back("ConSan MOI workgroup-id descriptor exceeds ELF bytes");
    return std::nullopt;
  }

  KD desc{};
  std::memcpy(&desc, image.data() + descriptor_file_offset, sizeof(desc));
  if (arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_GFX1250) {
    // These are the translated RDNA command-processor ABI TTMPs, not the
    // descriptor-selected entry SGPRs copied from them by the kernel prologue.
    // They remain dispatch inputs for the lifetime of the wave and are outside
    // both guest SGPR allocation and ConSan's ordinary SGPR scratch planner.
    return ConSanMoiWorkgroupSources{
        moi_descriptor_enables_workgroup_id(desc, 0)
            ? ConSanMoiWorkgroupSource{ttmp_scalar_operand(kTtmpRdna4GridX), false}
            : ConSanMoiWorkgroupSource{},
        moi_descriptor_enables_workgroup_id(desc, 1)
            ? ConSanMoiWorkgroupSource{ttmp_scalar_operand(kTtmpRdna4GridYz),
                                       /*shift_right_16=*/false, /*mask_low_16=*/true}
            : ConSanMoiWorkgroupSource{},
        moi_descriptor_enables_workgroup_id(desc, 2)
            ? ConSanMoiWorkgroupSource{ttmp_scalar_operand(kTtmpRdna4GridYz),
                                       /*shift_right_16=*/true}
            : ConSanMoiWorkgroupSource{},
        arch == ROCJITSU_CODE_ARCH_GFX1250 && uses_gfx1250_cluster_workgroup_id
            ? ConSanMoiWorkgroupSource{ttmp_scalar_operand(kTtmpGfx1250ClusterWorkgroupId), false,
                                       false}
            : ConSanMoiWorkgroupSource{},
    };
  }
  const auto sgpr_source = [](std::optional<uint16_t> sgpr) {
    return sgpr ? ConSanMoiWorkgroupSource{*sgpr, false} : ConSanMoiWorkgroupSource{};
  };
  return ConSanMoiWorkgroupSources{
      sgpr_source(moi_descriptor_workgroup_id_sgpr(desc, 0)),
      sgpr_source(moi_descriptor_workgroup_id_sgpr(desc, 1)),
      sgpr_source(moi_descriptor_workgroup_id_sgpr(desc, 2)),
      ConSanMoiWorkgroupSource{},
  };
}

[[nodiscard]] uint32_t moi_descriptor_vgpr_granularity(const KD &desc, rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_CDNA4)
    return 8;
  if (arch == ROCJITSU_CODE_ARCH_GFX1250)
    return moi_descriptor_wavefront_size(desc) == 32 ? 16 : 8;
  return moi_descriptor_wavefront_size(desc) == 32 ? 8 : 4;
}

[[nodiscard]] uint32_t moi_descriptor_vgpr_allocation_count(const KD &desc, rj_code_arch_t arch) {
  const uint32_t granularity = moi_descriptor_vgpr_granularity(desc, arch);
  const uint32_t granulated =
      AMDHSA_BITS_GET(desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
  const uint32_t allocated = (granulated + 1u) * granularity;
  return std::min<uint32_t>(allocated, kMaxVgprs);
}

[[nodiscard]] uint32_t moi_descriptor_sgpr_allocation_count(const KD &desc) {
  const uint32_t granulated = AMDHSA_BITS_GET(
      desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT);
  const uint32_t allocated = (granulated + 1u) * 8u;
  return std::min<uint32_t>(allocated, kMaxSgprs);
}

[[nodiscard]] std::optional<uint16_t> moi_descriptor_owner_shift(std::span<const uint8_t> image,
                                                                 uint64_t descriptor_file_offset,
                                                                 std::vector<std::string> &errors) {
  if (descriptor_file_offset > image.size() || sizeof(KD) > image.size() - descriptor_file_offset) {
    errors.emplace_back("ConSan MOI owner derivation descriptor exceeds ELF bytes");
    return std::nullopt;
  }

  KD desc{};
  std::memcpy(&desc, image.data() + descriptor_file_offset, sizeof(desc));
  return moi_descriptor_wavefront_size(desc) == 32 ? 5 : 6;
}

[[nodiscard]] std::optional<uint16_t> moi_kernel_owner_shift(std::span<const uint8_t> image,
                                                             const ConSanKernelInfo &kernel,
                                                             std::vector<std::string> &errors) {
  return moi_descriptor_owner_shift(image, kernel.descriptor_file_offset, errors);
}

[[nodiscard]] bool grow_moi_descriptor_vgpr_allocation(KD &desc, uint32_t required_count,
                                                       rj_code_arch_t arch) {
  if (required_count == 0 || required_count > kMaxVgprs)
    return false;

  if (required_count <= moi_descriptor_vgpr_allocation_count(desc, arch))
    return true;

  const uint32_t granularity = std::max<uint32_t>(moi_descriptor_vgpr_granularity(desc, arch), 1u);
  const uint32_t rounded = ((required_count + granularity - 1u) / granularity) * granularity;
  if (rounded > kMaxVgprs)
    return false;
  const uint32_t granulated = rounded / granularity - 1u;
  AMDHSA_BITS_SET(desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                  granulated);
  return true;
}

[[nodiscard]] bool grow_moi_descriptor_sgpr_allocation(KD &desc, uint32_t required_count) {
  if (required_count == 0 || required_count > kMaxSgprs)
    return false;
  if (required_count <= moi_descriptor_sgpr_allocation_count(desc))
    return true;

  constexpr uint32_t kSgprGranularity = 8;
  const uint32_t rounded =
      ((required_count + kSgprGranularity - 1u) / kSgprGranularity) * kSgprGranularity;
  // s0:s105 are the ordinary scalar operand file; the descriptor expresses
  // allocation in eight-register granules. The final legal request therefore
  // encodes the 112-register granule even though instruction operands stop at
  // s105. Decoding clamps that granule back to kMaxSgprs above.
  constexpr uint32_t kMaxEncodedSgprs =
      ((kMaxSgprs + kSgprGranularity - 1u) / kSgprGranularity) * kSgprGranularity;
  if (rounded > kMaxEncodedSgprs)
    return false;
  const uint32_t granulated = rounded / kSgprGranularity - 1u;
  AMDHSA_BITS_SET(desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  granulated);
  return true;
}

[[nodiscard]] bool grow_moi_kernel_descriptor_vgprs(CodeObjectPatcher &patcher,
                                                    std::span<const uint8_t> image,
                                                    const ConSanKernelInfo &kernel,
                                                    uint32_t required_count, rj_code_arch_t arch,
                                                    std::vector<std::string> &errors) {
  if (kernel.descriptor_file_offset > image.size() ||
      sizeof(KD) > image.size() - kernel.descriptor_file_offset) {
    errors.emplace_back("ConSan MOI descriptor VGPR growth exceeds ELF bytes");
    return false;
  }

  KD desc{};
  std::memcpy(&desc, image.data() + kernel.descriptor_file_offset, sizeof(desc));
  if (!grow_moi_descriptor_vgpr_allocation(desc, required_count, arch)) {
    errors.emplace_back("ConSan MOI could not grow descriptor VGPR allocation");
    return false;
  }
  if (!patcher.patch_kernel_descriptor(kernel.descriptor_file_offset,
                                       {reinterpret_cast<const uint8_t *>(&desc), sizeof(desc)})) {
    errors.emplace_back("ConSan MOI could not patch descriptor VGPR allocation");
    return false;
  }
  return true;
}

[[nodiscard]] bool append_moi_delay_words(std::vector<uint32_t> &words, rj_code_arch_t arch,
                                          const ConSanOptions &options,
                                          std::vector<std::string> &errors,
                                          std::string_view context) {
  if (options.delay_nops == 0)
    return true;

  switch (options.delay_mode) {
  case ConSanDelayMode::Nop:
    for (uint32_t i = 0; i < options.delay_nops; ++i)
      words.push_back(build_s_nop(0, arch));
    return true;
  case ConSanDelayMode::Sleep:
    if (options.delay_nops > std::numeric_limits<uint16_t>::max()) {
      errors.emplace_back(std::string(context) +
                          " sleep delay immediate exceeds the 16-bit s_sleep field");
      return false;
    }
    words.push_back(build_s_sleep(static_cast<uint16_t>(options.delay_nops), arch));
    return true;
  case ConSanDelayMode::SleepVar:
    if (options.delay_var_ssrc > std::numeric_limits<uint8_t>::max()) {
      errors.emplace_back(std::string(context) +
                          " sleep_var source exceeds the 8-bit scalar source field");
      return false;
    }
    words.push_back(build_s_sleep_var(options.delay_var_ssrc, arch));
    return true;
  }

  errors.emplace_back(std::string(context) + " has unknown delay mode '" +
                      consan_delay_mode_name(options.delay_mode) + "'");
  return false;
}

[[nodiscard]] uint16_t inline_shadow_scratch_count(const ConSanOptions &options) {
  // Versioned exact publication retains the slot address, new packed access,
  // prior dispatch, ready version, and CAS operands through commit. The token
  // hash aliases the now-larger transaction window.
  return options.moi_exec_save_sgpr ? (options.moi_track_atomics ? 24u : 16u) : 11u;
}

[[nodiscard]] uint16_t inline_shadow_loop_scratch_count(const ConSanOptions &options,
                                                        const ConSanMoiCandidate &candidate) {
  (void)options;
  // Wide local accesses retain an offset and iteration counter. Wide external
  // accesses retain an iteration counter and the workgroup key, since the
  // versioned transaction reuses all of its ordinary address temporaries.
  return candidate.width_bits > consan_moi_exact_shadow::granule_bytes * 8u ? 2u : 0u;
}

[[nodiscard]] bool validate_inline_shadow_exec_save_sgpr(const ConSanOptions &options,
                                                         rj_code_arch_t arch,
                                                         std::vector<std::string> &errors) {
  if (!options.moi_exec_save_sgpr)
    return true;
  if (options.automatic_moi_inline_sgpr_spill && options.moi_inline_access_present &&
      (!options.moi_inline_call_return_sgpr || !options.moi_inline_dispatch_key_sgpr)) {
    errors.emplace_back(
        "ConSan MOI spill-backed inline-shadow probes require external dense-router state");
    return false;
  }
  // Diagnostics use s[base:base+10]. Atomic publication and causal-token
  // handling retain masks through +20:+21. Access probes additionally use
  // local/dynamic state through +25 and preserve application EXEC in the
  // dedicated +26:+27 pair.
  const uint16_t ordinary_sgpr_count = arch == ROCJITSU_CODE_ARCH_CDNA4 ? 102u : kMaxSgprs;
  const uint16_t required_sgpr_count =
      options.moi_inline_access_present ? kConSanMoiInlineExecSaveSgprCount : 22u;
  const uint16_t max_exec_save_sgpr =
      static_cast<uint16_t>(ordinary_sgpr_count - required_sgpr_count);
  if (*options.moi_exec_save_sgpr > max_exec_save_sgpr || *options.moi_exec_save_sgpr % 2u != 0u) {
    errors.emplace_back("ConSan MOI inline-shadow diagnostics require an even "
                        "RJ_CONSAN_MOI_EXEC_SAVE_SGPR in 0.." +
                        std::to_string(max_exec_save_sgpr));
    return false;
  }
  return true;
}

[[nodiscard]] std::optional<uint16_t>
candidate_lds_byte_offset_vgpr(const ConSanMoiCandidate &candidate,
                               std::vector<std::string> &errors) {
  if (!candidate.addr_vgpr) {
    errors.emplace_back("ConSan MOI probe requires an LDS address VGPR");
    return std::nullopt;
  }
  return *candidate.addr_vgpr;
}

[[nodiscard]] bool
candidate_requires_gfx1250_flat_offset_materialization(const ConSanMoiCandidate &candidate) {
  return (candidate.source == ConSanMoiCandidateSource::FlatGroup ||
          candidate.source == ConSanMoiCandidateSource::FlatMaybeGroup) &&
         candidate.raw_ioffset && *candidate.raw_ioffset != 0;
}

[[nodiscard]] bool
append_materialize_gfx1250_flat_access_address(std::vector<uint32_t> &words,
                                               const ConSanMoiCandidate &candidate,
                                               uint16_t result_vgpr, rj_code_arch_t arch) {
  if (!candidate_requires_gfx1250_flat_offset_materialization(candidate) ||
      arch != ROCJITSU_CODE_ARCH_GFX1250 || !candidate.addr_vgpr || *candidate.addr_vgpr >= 255u ||
      result_vgpr >= 255u)
    return false;
  words.push_back(build_v_mov_b32_e32(result_vgpr, vector_source_vgpr(*candidate.addr_vgpr), arch));
  words.push_back(build_v_mov_b32_e32(
      static_cast<uint16_t>(result_vgpr + 1u),
      vector_source_vgpr(static_cast<uint16_t>(*candidate.addr_vgpr + 1u)), arch));
  const auto add =
      instrumentation::build_v_add_u64_signed_i24(result_vgpr, *candidate.raw_ioffset, arch);
  if (!add)
    return false;
  words.insert(words.end(), add->begin(), add->end());
  return true;
}

[[nodiscard]] uint16_t candidate_payload_vgpr_count(const ConSanMoiCandidate &candidate) {
  if (candidate.width_bits == 0)
    return 1;
  const bool two_address = equals_any(
      candidate.mnemonic, {"ds_load_2addr_b32", "ds_load_2addr_b64", "ds_load_2addr_stride64_b32",
                           "ds_load_2addr_stride64_b64", "ds_store_2addr_b32", "ds_store_2addr_b64",
                           "ds_store_2addr_stride64_b32", "ds_store_2addr_stride64_b64"});
  const uint32_t per_address = std::max<uint32_t>((candidate.width_bits + 31u) / 32u, 1u);
  return static_cast<uint16_t>(per_address * (two_address ? 2u : 1u));
}

[[nodiscard]] uint16_t
candidate_payload_vgpr_count_per_address(const ConSanMoiCandidate &candidate) {
  if (candidate.width_bits == 0)
    return 1;
  return static_cast<uint16_t>(std::max<uint32_t>((candidate.width_bits + 31u) / 32u, 1u));
}

[[nodiscard]] bool moi_load_clobbers_address(const ConSanMoiCandidate &candidate) {
  if (!candidate.addr_vgpr || !candidate.dst_vgpr)
    return false;
  const uint16_t address_vgpr_count =
      candidate.source == ConSanMoiCandidateSource::NativeLds ? 1u : 2u;
  return range_overlaps(*candidate.addr_vgpr, address_vgpr_count, *candidate.dst_vgpr,
                        candidate_payload_vgpr_count(candidate));
}

[[nodiscard]] uint16_t inline_shadow_scratch_count(const ConSanOptions &options,
                                                   const ConSanMoiCandidate &candidate,
                                                   rj_code_arch_t arch) {
  // A CDNA4 spill window may overlap the displaced guest operands. Reserve
  // two otherwise-unused registers at the end of the window so emission can
  // snapshot the DS address before the probe starts clobbering that window.
  // Two registers ensure the chosen snapshot is distinct even when the guest
  // address itself happens to be one of the reserved registers.
  const bool cdna4_spill_capable =
      arch == ROCJITSU_CODE_ARCH_CDNA4 &&
      (options.force_vgpr_spill || options.automatic_moi_private_epoch ||
       options.moi_dynamic_stack_spill);
  const uint16_t spill_address_snapshot_count = cdna4_spill_capable ? 2u : 0u;
  const uint16_t flat_offset_address_count =
      candidate_requires_gfx1250_flat_offset_materialization(candidate) ? 2u : 0u;
  return static_cast<uint16_t>(
      inline_shadow_scratch_count(options) + inline_shadow_loop_scratch_count(options, candidate) +
      (moi_load_clobbers_address(candidate) && flat_offset_address_count == 0u ? 1u : 0u) +
      std::max(spill_address_snapshot_count, flat_offset_address_count));
}

[[nodiscard]] uint16_t direct_sampled_scratch_count(const ConSanOptions &options,
                                                    const ConSanMoiCandidate &candidate) {
  const uint16_t base_scratch_count = options.moi_sampled_check ? 7u : 5u;
  return static_cast<uint16_t>(
      base_scratch_count + (options.moi_runtime_sample_stride > 1 ? 1u : 0u) +
      (candidate_requires_gfx1250_flat_offset_materialization(candidate) ? 2u
       : moi_load_clobbers_address(candidate)                            ? 1u
                                                                         : 0u) +
      (options.automatic_moi_private_epoch ? 1u : 0u));
}

[[nodiscard]] bool reject_candidate_scratch_range_overlap(const ConSanMoiCandidate &candidate,
                                                          uint16_t scratch_vgpr,
                                                          uint16_t scratch_count,
                                                          std::vector<std::string> &errors) {
  auto address_vgpr = candidate_lds_byte_offset_vgpr(candidate, errors);
  if (!address_vgpr)
    return true;
  const uint16_t address_vgpr_count =
      candidate.source == ConSanMoiCandidateSource::NativeLds ? 1u : 2u;
  if (range_overlaps(*address_vgpr, address_vgpr_count, scratch_vgpr, scratch_count)) {
    errors.emplace_back("ConSan MOI probe scratch VGPRs overlap the LDS address VGPRs");
    return true;
  }
  const uint16_t payload_vgpr_count = candidate_payload_vgpr_count(candidate);
  if (candidate.dst_vgpr &&
      range_overlaps(*candidate.dst_vgpr, payload_vgpr_count, scratch_vgpr, scratch_count)) {
    errors.emplace_back("ConSan MOI probe scratch VGPRs overlap the destination VGPRs");
    return true;
  }
  const uint16_t data_vgpr_count = candidate.second_data_vgpr
                                       ? candidate_payload_vgpr_count_per_address(candidate)
                                       : payload_vgpr_count;
  if (candidate.data_vgpr &&
      range_overlaps(*candidate.data_vgpr, data_vgpr_count, scratch_vgpr, scratch_count)) {
    errors.emplace_back("ConSan MOI probe scratch VGPRs overlap the data VGPRs");
    return true;
  }
  if (candidate.second_data_vgpr &&
      range_overlaps(*candidate.second_data_vgpr, data_vgpr_count, scratch_vgpr, scratch_count)) {
    errors.emplace_back("ConSan MOI probe scratch VGPRs overlap the second data VGPRs");
    return true;
  }
  return false;
}

struct MoiKernelResourceContext {
  const ConSanKernelInfo *kernel = nullptr;
  KernelCfgScope scope;
  std::unordered_set<BasicBlock *> block_set;
  std::unique_ptr<LivenessAnalysis> liveness;
  uint16_t current_vgpr_count = 0;
  uint16_t max_referenced_vgpr_count = 0;
  uint16_t ordinary_vgpr_limit = kMaxVgprs;
  uint16_t current_sgpr_count = 0;
  uint16_t max_referenced_sgpr_count = 0;
  uint32_t original_private_segment_size = 0;
  bool has_indirect_vgpr_access = false;
  bool descriptor_valid = false;
};

[[nodiscard]] const ConSanMoiPersistentVgprAssignment *
find_moi_persistent_vgpr_assignment(const ConSanOptions &options, uint64_t descriptor_offset) {
  const auto assignment =
      std::ranges::find(options.moi_persistent_vgpr_assignments, descriptor_offset,
                        &ConSanMoiPersistentVgprAssignment::descriptor_file_offset);
  return assignment == options.moi_persistent_vgpr_assignments.end() ? nullptr : &*assignment;
}

[[nodiscard]] const ConSanMoiPrologueScratchVgprAssignment *
find_moi_prologue_scratch_vgpr_assignment(const ConSanOptions &options,
                                          uint64_t descriptor_offset) {
  const auto assignment =
      std::ranges::find(options.moi_prologue_scratch_vgpr_assignments, descriptor_offset,
                        &ConSanMoiPrologueScratchVgprAssignment::descriptor_file_offset);
  return assignment == options.moi_prologue_scratch_vgpr_assignments.end() ? nullptr : &*assignment;
}

[[nodiscard]] const ConSanMoiTransientSgprAssignment *
find_moi_transient_sgpr_assignment(const ConSanOptions &options, uint64_t descriptor_offset) {
  const auto assignment =
      std::ranges::find(options.moi_transient_sgpr_assignments, descriptor_offset,
                        &ConSanMoiTransientSgprAssignment::descriptor_file_offset);
  return assignment == options.moi_transient_sgpr_assignments.end() ? nullptr : &*assignment;
}

/// Applies the one transient scalar assignment shared by every owner of a
/// site. Owners connected through a shared helper must either all carry the
/// same assignment or keep the code-object-wide fallback.
[[nodiscard]] bool
apply_moi_transient_sgpr_assignment(ConSanOptions &options,
                                    std::span<const uint64_t> owner_descriptor_offsets) {
  const ConSanMoiTransientSgprAssignment *selected = nullptr;
  bool saw_unassigned_owner = false;
  for (uint64_t descriptor_offset : owner_descriptor_offsets) {
    const auto *assignment = find_moi_transient_sgpr_assignment(options, descriptor_offset);
    if (!assignment) {
      saw_unassigned_owner = true;
      continue;
    }
    if (selected && (selected->exec_save_sgpr != assignment->exec_save_sgpr ||
                     selected->dispatch_id_sgpr != assignment->dispatch_id_sgpr ||
                     selected->spill_backed != assignment->spill_backed ||
                     selected->indirect_pc_sgpr != assignment->indirect_pc_sgpr ||
                     selected->indirect_scc_sgpr != assignment->indirect_scc_sgpr ||
                     selected->dispatch_key_sgpr != assignment->dispatch_key_sgpr ||
                     selected->call_return_sgpr != assignment->call_return_sgpr ||
                     selected->visible_evidence_sgpr != assignment->visible_evidence_sgpr))
      return false;
    selected = assignment;
  }
  if (!selected) {
    return options.moi_exec_save_sgpr.has_value() ||
           (options.moi_engine == ConSanMoiEngine::InlineShadow &&
            options.automatic_moi_inline_sgpr_spill && options.moi_inline_indirect_pc_sgpr &&
            options.moi_inline_indirect_scc_sgpr && options.moi_inline_dispatch_key_sgpr &&
            options.moi_inline_call_return_sgpr);
  }
  if (saw_unassigned_owner)
    return false;
  options.moi_exec_save_sgpr = selected->exec_save_sgpr;
  options.moi_dispatch_id_sgpr = selected->dispatch_id_sgpr;
  options.moi_inline_indirect_pc_sgpr = selected->indirect_pc_sgpr;
  options.moi_inline_indirect_scc_sgpr = selected->indirect_scc_sgpr;
  if (options.moi_engine == ConSanMoiEngine::InlineShadow) {
    options.automatic_moi_inline_sgpr_spill = selected->spill_backed;
    options.automatic_moi_record_replay_sgpr_spill = false;
    options.moi_inline_dispatch_key_sgpr = selected->dispatch_key_sgpr;
    options.moi_inline_call_return_sgpr = selected->call_return_sgpr;
    options.moi_inline_visible_evidence_sgpr = selected->visible_evidence_sgpr;
  } else {
    options.automatic_moi_inline_sgpr_spill = false;
    options.automatic_moi_record_replay_sgpr_spill = selected->spill_backed;
    options.moi_record_replay_dispatch_key_sgpr = selected->dispatch_key_sgpr;
    options.moi_record_replay_call_return_sgpr = selected->call_return_sgpr;
  }
  return true;
}

[[nodiscard]] bool
apply_moi_persistent_vgpr_assignment(ConSanOptions &options,
                                     std::span<const uint64_t> owner_descriptor_offsets) {
  if (options.moi_persistent_vgpr_assignments.empty())
    return options.moi_owner_vgpr.has_value();
  const ConSanMoiPersistentVgprAssignment *selected = nullptr;
  for (uint64_t descriptor_offset : owner_descriptor_offsets) {
    const auto *assignment = find_moi_persistent_vgpr_assignment(options, descriptor_offset);
    if (!assignment)
      continue;
    if (selected && (selected->owner_vgpr != assignment->owner_vgpr ||
                     selected->epoch_vgpr != assignment->epoch_vgpr ||
                     selected->workgroup_key_vgpr != assignment->workgroup_key_vgpr ||
                     selected->sample_sequence_vgpr != assignment->sample_sequence_vgpr)) {
      return false;
    }
    selected = assignment;
  }
  if (!selected)
    return false;
  options.moi_owner_vgpr = selected->owner_vgpr;
  options.moi_epoch_vgpr = selected->epoch_vgpr;
  options.moi_workgroup_key_vgpr = selected->workgroup_key_vgpr;
  options.moi_sample_sequence_vgpr = selected->sample_sequence_vgpr;
  return true;
}

[[nodiscard]] bool has_indirect_vgpr_access(const KernelCfgScope &scope) {
  for (BasicBlock *block : scope.blocks) {
    if (block == nullptr)
      continue;
    if (std::ranges::any_of(block->instructions(), [](const Instruction &inst) {
          return inst.mnemonic().starts_with("v_movrel");
        })) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] uint16_t max_referenced_vgpr_count(const KernelCfgScope &scope) {
  uint32_t max_count = 0;
  for (BasicBlock *block : scope.blocks) {
    if (block == nullptr)
      continue;
    for (const Instruction &inst : block->instructions()) {
      const InstDefUse du(inst);
      const RegisterSet referenced = du.defs | du.uses;
      referenced.for_each([&](RegisterRef ref) {
        if (ref.cls == RegClass::VGPR)
          max_count = std::max<uint32_t>(max_count, static_cast<uint32_t>(ref.index) + 1u);
      });
    }
  }
  return static_cast<uint16_t>(std::min<uint32_t>(max_count, kMaxVgprs));
}

[[nodiscard]] uint16_t max_referenced_sgpr_count(const KernelCfgScope &scope) {
  uint32_t max_count = 0;
  for (BasicBlock *block : scope.blocks) {
    if (block == nullptr)
      continue;
    for (const Instruction &inst : block->instructions()) {
      const InstDefUse du(inst);
      const RegisterSet referenced = du.defs | du.uses;
      referenced.for_each([&](RegisterRef ref) {
        if (ref.cls == RegClass::SGPR)
          max_count = std::max<uint32_t>(max_count, static_cast<uint32_t>(ref.index) + 1u);
      });
    }
  }
  return static_cast<uint16_t>(std::min<uint32_t>(max_count, kMaxSgprs));
}

[[nodiscard]] const Instruction *instruction_at(BasicBlock *block, uint64_t text_offset) {
  if (block == nullptr)
    return nullptr;
  const auto it =
      std::ranges::find_if(block->instructions(), [text_offset](const Instruction &inst) {
        return inst.src_loc() == text_offset;
      });
  return it == block->instructions().end() ? nullptr : &*it;
}

[[nodiscard]] uint16_t moi_access_scratch_vgpr_count(const ConSanOptions &options,
                                                     const ConSanMoiCandidate &candidate,
                                                     rj_code_arch_t arch) {
  switch (options.moi_engine) {
  case ConSanMoiEngine::RecordReplay:
    return static_cast<uint16_t>((options.moi_dynamic_access_records ? 6u : 3u) +
                                 (candidate_requires_gfx1250_flat_offset_materialization(candidate)
                                      ? 2u
                                  : moi_load_clobbers_address(candidate) ? 1u
                                                                         : 0u));
  case ConSanMoiEngine::InlineShadow:
    return inline_shadow_scratch_count(options, candidate, arch);
  case ConSanMoiEngine::Sampled:
    return direct_sampled_scratch_count(options, candidate);
  }
  return 0;
}

struct MoiResourcePlanningState {
  std::span<const uint8_t> bytes;
  rj_code_arch_t arch;
  AmdGpuCodeObject code_object;
  std::unique_ptr<Decoder> decoder;
  std::vector<std::unique_ptr<BasicBlock>> blocks;
  BlockOffsetIndex block_index;
  std::vector<MoiKernelResourceContext> contexts;
  std::unordered_map<uint64_t, MoiKernelResourceContext *> context_by_descriptor;
  bool valid = false;

  MoiResourcePlanningState(std::span<const uint8_t> input, rj_code_arch_t arch,
                           const ConSanOptions &options, const ConSanResult &result)
      : bytes(input), arch(arch), code_object(input.data(), input.size()),
        decoder(Decoder::create(arch)) {
    if (!code_object.is_valid() || !decoder)
      return;

    std::vector<uint64_t> leaders;
    leaders.reserve(result.kernels.size() + result.functions.size());
    std::vector<uint64_t> kernel_entries;
    kernel_entries.reserve(result.kernels.size());
    for (const ConSanKernelInfo &kernel : result.kernels) {
      if (!kernel.has_text_range)
        continue;
      leaders.push_back(kernel.entry_text_offset);
      kernel_entries.push_back(kernel.entry_text_offset);
    }
    for (const ConSanFunctionInfo &function : result.functions)
      leaders.push_back(function.entry_text_offset);
    for (const ConSanPreappliedCodeRange &range : options.preapplied_code_ranges) {
      leaders.push_back(range.text_offset);
      if (range.continuation_text_offset)
        leaders.push_back(*range.continuation_text_offset);
    }
    std::ranges::sort(leaders);
    leaders.erase(std::ranges::unique(leaders).begin(), leaders.end());
    std::ranges::sort(kernel_entries);
    kernel_entries.erase(std::ranges::unique(kernel_entries).begin(), kernel_entries.end());

    std::vector<BasicBlock::CodeRange> code_ranges;
    code_ranges.reserve(code_object.functions().size());
    for (const AmdGpuFunctionInfo &function : code_object.functions()) {
      if (function.code_size != 0)
        code_ranges.push_back(
            {.start_offset = function.entry_text_offset, .size = function.code_size});
    }
    for (const ConSanPreappliedCodeRange &range : options.preapplied_code_ranges) {
      if (range.size != 0)
        code_ranges.push_back({.start_offset = range.text_offset, .size = range.size});
    }

    blocks = BasicBlock::build(code_object, *decoder, arch, leaders, code_ranges);
    block_index = build_block_offset_index(blocks);
    CodeObjectPatcher patcher(code_object);
    const std::span<const uint8_t> text = patcher.text_bytes();

    contexts.reserve(result.kernels.size());
    for (const ConSanKernelInfo &kernel : result.kernels) {
      if (!kernel.has_text_range)
        continue;
      std::vector<uint64_t> additional_entry_offsets;
      for (const ConSanPreappliedCodeRange &range : options.preapplied_code_ranges) {
        if (range.kernel_name != kernel.name)
          continue;
        if (block_for_offset(block_index, range.text_offset) != nullptr)
          additional_entry_offsets.push_back(range.text_offset);
        if (range.continuation_text_offset &&
            block_for_offset(block_index, *range.continuation_text_offset) != nullptr)
          additional_entry_offsets.push_back(*range.continuation_text_offset);
      }
      auto scope = build_kernel_cfg_scope(
          blocks, block_index,
          KernelScopeRequest{.entry_offset = kernel.entry_text_offset,
                             .additional_entry_offsets = additional_entry_offsets},
          kernel_entries, text);
      if (!scope)
        continue;

      MoiKernelResourceContext context;
      context.kernel = &kernel;
      context.scope = std::move(*scope);
      contexts.push_back(std::move(context));
      MoiKernelResourceContext &stored = contexts.back();
      stored.block_set.insert(stored.scope.blocks.begin(), stored.scope.blocks.end());
      stored.liveness = std::make_unique<LivenessAnalysis>(KernelBlockScope(stored.scope.blocks),
                                                           LivenessAnalysisOptions{},
                                                           stored.scope.liveness_edges);
      stored.max_referenced_vgpr_count = max_referenced_vgpr_count(stored.scope);
      stored.max_referenced_sgpr_count = std::max<uint16_t>(max_referenced_sgpr_count(stored.scope),
                                                            kernel.sgpr_count.value_or(0u));
      stored.has_indirect_vgpr_access = has_indirect_vgpr_access(stored.scope);

      if (kernel.descriptor_file_offset > bytes.size() ||
          sizeof(KD) > bytes.size() - kernel.descriptor_file_offset) {
        continue;
      }
      KD descriptor{};
      std::memcpy(&descriptor, bytes.data() + kernel.descriptor_file_offset, sizeof(descriptor));
      stored.current_vgpr_count =
          static_cast<uint16_t>(moi_descriptor_vgpr_allocation_count(descriptor, arch));
      if (arch == ROCJITSU_CODE_ARCH_CDNA4) {
        const uint32_t encoded_accum_offset = AMDHSA_BITS_GET(
            descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET);
        // A nonzero ACCUM_OFFSET is the first physical AccVGPR and therefore
        // the hard upper bound for ordinary instrumentation VGPRs. Encoded
        // zero remains ambiguous for kernels without an accumulator bank.
        if (encoded_accum_offset != 0u) {
          stored.ordinary_vgpr_limit = static_cast<uint16_t>(
              std::min<uint32_t>(kMaxVgprs, (encoded_accum_offset + 1u) * 4u));
        }
      }
      stored.current_sgpr_count =
          static_cast<uint16_t>(moi_descriptor_sgpr_allocation_count(descriptor));
      stored.original_private_segment_size = descriptor.private_segment_fixed_size;
      stored.descriptor_valid = true;
    }
    context_by_descriptor.reserve(contexts.size());
    for (MoiKernelResourceContext &context : contexts)
      context_by_descriptor.try_emplace(context.kernel->descriptor_file_offset, &context);
    valid = true;
  }
};

[[nodiscard]] uint16_t moi_exec_save_sgpr_count(const ConSanOptions &options, rj_code_arch_t arch);

[[nodiscard]] bool moi_relocatable_host_preserves_transient_sgprs(
    const MoiResourcePlanningState &state, const ConSanOptions &options,
    const ConSanCandidateResourcePlan &resource_plan,
    std::span<const MoiKernelResourceContext *const> owner_contexts, BasicBlock &block,
    const Instruction &host_instruction, uint64_t host_bytes) {
  const bool has_owner_override = std::ranges::any_of(
      resource_plan.owner_descriptor_file_offsets, [&](uint64_t descriptor_offset) {
        return find_moi_transient_sgpr_assignment(options, descriptor_offset) != nullptr;
      });
  if (!has_owner_override)
    return true;
  if (!state.valid || !options.moi_exec_save_sgpr)
    return false;

  const auto overlaps_range = [](const RegisterSet &set, uint16_t base, uint16_t count) {
    for (uint16_t index = 0; index < count; ++index) {
      if (set.contains({RegClass::SGPR, static_cast<uint16_t>(base + index), 1u}))
        return true;
    }
    return false;
  };
  const auto overlaps_window = [&](const RegisterSet &set) {
    if (options.automatic_moi_record_replay_sgpr_spill) {
      if (!options.moi_inline_indirect_pc_sgpr || !options.moi_inline_indirect_scc_sgpr ||
          !options.moi_record_replay_dispatch_key_sgpr ||
          !options.moi_record_replay_call_return_sgpr)
        return true;
      return overlaps_range(set, *options.moi_inline_indirect_pc_sgpr, 2u) ||
             overlaps_range(set, *options.moi_inline_indirect_scc_sgpr, 1u) ||
             overlaps_range(set, *options.moi_record_replay_dispatch_key_sgpr, 1u) ||
             overlaps_range(set, *options.moi_record_replay_call_return_sgpr, 2u);
    }
    if (options.automatic_moi_inline_sgpr_spill) {
      if (!options.moi_inline_indirect_pc_sgpr || !options.moi_inline_indirect_scc_sgpr ||
          !options.moi_inline_visible_evidence_sgpr)
        return true;
      return overlaps_range(set, *options.moi_inline_indirect_pc_sgpr, 2u) ||
             overlaps_range(set, *options.moi_inline_indirect_scc_sgpr, 1u) ||
             overlaps_range(set, *options.moi_inline_visible_evidence_sgpr, 1u);
    }
    return overlaps_range(set, *options.moi_exec_save_sgpr,
                          moi_exec_save_sgpr_count(options, state.arch));
  };

  bool found_owner_context = false;
  for (const MoiKernelResourceContext *context : owner_contexts) {
    if (context == nullptr || !context->block_set.contains(&block))
      continue;
    if (overlaps_window(context->liveness->live_before(host_instruction)))
      return false;
    found_owner_context = true;
  }
  if (!found_owner_context)
    return false;
  for (const Instruction &instruction : block.instructions()) {
    if (instruction.src_loc() < host_instruction.src_loc() ||
        instruction.src_loc() >= host_instruction.src_loc() + host_bytes)
      continue;
    const InstDefUse def_use(instruction);
    if (overlaps_window(def_use.defs | def_use.uses))
      return false;
  }
  return true;
}

[[nodiscard]] ConSanCandidateResourcePlan
plan_moi_resource_site(MoiResourcePlanningState &state, const ConSanOptions &options,
                       ConSanResourceSiteKind site_kind, size_t candidate_index,
                       uint64_t text_offset, std::optional<uint64_t> kernel_descriptor_file_offset,
                       uint16_t scratch_count, const ConSanMoiCandidate *access_candidate = nullptr,
                       const ConSanAtomicSite *atomic_site = nullptr) {
  ConSanCandidateResourcePlan plan;
  plan.site_kind = site_kind;
  plan.candidate_index = candidate_index;
  plan.text_offset = text_offset;
  plan.scratch_vgpr_count = scratch_count;
  if (!state.valid) {
    plan.reason = ConSanRegisterPlanReason::MissingInstruction;
    return plan;
  }

  BasicBlock *anchor_block = block_for_offset(state.block_index, text_offset);
  const Instruction *anchor = instruction_at(anchor_block, text_offset);
  if (anchor == nullptr) {
    plan.reason = ConSanRegisterPlanReason::MissingInstruction;
    return plan;
  }

  std::vector<MoiKernelResourceContext *> owners;
  const auto add_owner = [&](MoiKernelResourceContext &context) {
    if (!context.block_set.contains(anchor_block))
      return;
    owners.push_back(&context);
    plan.owner_descriptor_file_offsets.push_back(context.kernel->descriptor_file_offset);
  };
  if (kernel_descriptor_file_offset) {
    const auto owner = state.context_by_descriptor.find(*kernel_descriptor_file_offset);
    if (owner != state.context_by_descriptor.end())
      add_owner(*owner->second);
  } else {
    for (MoiKernelResourceContext &context : state.contexts)
      add_owner(context);
  }
  if (owners.empty()) {
    plan.reason = ConSanRegisterPlanReason::MissingOwner;
    return plan;
  }
  plan.current_vgpr_count = kMaxVgprs;
  plan.ordinary_vgpr_limit = kMaxVgprs;
  plan.current_sgpr_count = kMaxSgprs;
  RegisterSet live_before_all_owners;
  bool descriptors_valid = true;
  for (MoiKernelResourceContext *owner : owners) {
    plan.current_vgpr_count = std::min(plan.current_vgpr_count, owner->current_vgpr_count);
    plan.ordinary_vgpr_limit = std::min(plan.ordinary_vgpr_limit, owner->ordinary_vgpr_limit);
    plan.max_referenced_vgpr_count =
        std::max(plan.max_referenced_vgpr_count, owner->max_referenced_vgpr_count);
    plan.current_sgpr_count = std::min(plan.current_sgpr_count, owner->current_sgpr_count);
    plan.max_referenced_sgpr_count =
        std::max(plan.max_referenced_sgpr_count, owner->max_referenced_sgpr_count);
    plan.original_private_segment_size =
        std::max(plan.original_private_segment_size, owner->original_private_segment_size);
    live_before_all_owners |= owner->liveness->live_before(*anchor);
    // Relative VGPR instructions address a register array through M0. Their
    // dynamically selected elements are intentionally absent from ordinary
    // explicit-operand def/use sets, so a normal liveness proof can otherwise
    // misclassify a live array element as instrumentation scratch. Conservatively
    // reserve the whole guest allocation; the planner can grow the descriptor or
    // use the existing spill path instead of clobbering indirect guest state.
    if (owner->has_indirect_vgpr_access) {
      plan.has_indirect_vgpr_access = true;
      for (uint16_t index = 0; index < owner->current_vgpr_count; ++index)
        live_before_all_owners.expand({RegClass::VGPR, index, 1});
    }
    descriptors_valid &= owner->descriptor_valid;
  }
  plan.required_vgpr_count = plan.current_vgpr_count;
  if (!descriptors_valid) {
    plan.reason = ConSanRegisterPlanReason::InvalidDescriptor;
    return plan;
  }

  const InstDefUse anchor_def_use(*anchor);
  ConSanRegisterRequest request;
  request.reg_class = RegClass::VGPR;
  request.count = scratch_count;
  // Every ConSan engine uses the first two scratch registers as a FLAT
  // address pair. CDNA4 requires that pair to begin at an even VGPR.
  request.alignment = state.arch == ROCJITSU_CODE_ARCH_CDNA4 ? 2u : 1u;
  request.current_allocation_count = std::min(plan.current_vgpr_count, plan.ordinary_vgpr_limit);
  request.max_referenced_count = plan.max_referenced_vgpr_count;
  request.architecture_limit = plan.ordinary_vgpr_limit;
  request.explicit_base = options.scratch_vgpr;
  request.force_spill = options.force_vgpr_spill;
  request.forbidden = anchor_def_use.defs | anchor_def_use.uses;
  // Candidate decoding is the authoritative operand-width source for native
  // two-address LDS operations.  Some generic instruction def/use models only
  // expose the first destination register, so supplement them before choosing
  // a liveness-dead window rather than rejecting the chosen window later.
  if (access_candidate != nullptr) {
    if (access_candidate->addr_vgpr) {
      const uint8_t address_count =
          access_candidate->source == ConSanMoiCandidateSource::NativeLds ? 1u : 2u;
      request.forbidden.expand({RegClass::VGPR, *access_candidate->addr_vgpr, address_count});
    }
    const uint8_t payload_count =
        static_cast<uint8_t>(candidate_payload_vgpr_count(*access_candidate));
    if (access_candidate->dst_vgpr)
      request.forbidden.expand({RegClass::VGPR, *access_candidate->dst_vgpr, payload_count});
    const uint8_t data_count =
        access_candidate->second_data_vgpr
            ? static_cast<uint8_t>(candidate_payload_vgpr_count_per_address(*access_candidate))
            : payload_count;
    if (access_candidate->data_vgpr)
      request.forbidden.expand({RegClass::VGPR, *access_candidate->data_vgpr, data_count});
    if (access_candidate->second_data_vgpr) {
      request.forbidden.expand({RegClass::VGPR, *access_candidate->second_data_vgpr, data_count});
    }
  }
  // Atomic address/payload widths come from ConSan's decoded atomic site.
  // Reserve them explicitly: generic def/use can understate native two-address
  // operations, and rejecting an overlap only during emission invalidates the
  // whole code-object transform instead of selecting another scratch window.
  if (atomic_site != nullptr) {
    if (atomic_site->addr_vgpr) {
      const uint8_t address_count =
          atomic_site->mnemonic.starts_with("buffer_") || atomic_site->mnemonic.starts_with("ds_")
              ? 1u
              : 2u;
      request.forbidden.expand({RegClass::VGPR, *atomic_site->addr_vgpr, address_count});
    }
    if (atomic_site->data_vgpr)
      request.forbidden.expand({RegClass::VGPR, *atomic_site->data_vgpr, 1});
    if (atomic_site->dst_vgpr)
      request.forbidden.expand({RegClass::VGPR, *atomic_site->dst_vgpr, 1});
  }
  if (!options.moi_owner_vgpr && options.moi_persistent_vgpr_assignments.empty() &&
      options.moi_owner_source == ConSanMoiOwnerSource::WorkitemId)
    request.forbidden.expand({RegClass::VGPR, kRdna4WorkitemIdX, 1});
  if (options.moi_owner_vgpr)
    request.forbidden.expand({RegClass::VGPR, *options.moi_owner_vgpr, 1});
  if (options.moi_epoch_vgpr)
    request.forbidden.expand({RegClass::VGPR, *options.moi_epoch_vgpr, 1});
  if (options.moi_workgroup_key_vgpr)
    request.forbidden.expand({RegClass::VGPR, *options.moi_workgroup_key_vgpr, 1});
  for (uint64_t descriptor_offset : plan.owner_descriptor_file_offsets) {
    const auto *assignment = find_moi_persistent_vgpr_assignment(options, descriptor_offset);
    if (!assignment)
      continue;
    request.forbidden.expand({RegClass::VGPR, assignment->owner_vgpr, 1});
    request.forbidden.expand({RegClass::VGPR, assignment->epoch_vgpr, 1});
    if (assignment->workgroup_key_vgpr)
      request.forbidden.expand({RegClass::VGPR, *assignment->workgroup_key_vgpr, 1});
    if (assignment->sample_sequence_vgpr)
      request.forbidden.expand({RegClass::VGPR, *assignment->sample_sequence_vgpr, 1});
  }

  ConSanRegisterPlan register_plan = plan_consan_registers(request, live_before_all_owners);
  if (register_plan.source == ConSanRegisterAllocationSource::Unsupported &&
      register_plan.reason == ConSanRegisterPlanReason::NoLegalWindow &&
      options.moi_engine == ConSanMoiEngine::InlineShadow &&
      site_kind == ConSanResourceSiteKind::Access && !request.explicit_base) {
    // Inline access bodies execute in appended caves as save -> probe ->
    // restore -> displaced guest access. The spill tier may therefore use a
    // window overlapping the guest instruction's operands: those values are
    // restored before that instruction executes. Persistent state remains
    // forbidden because it must survive the complete kernel lifetime.
    ConSanRegisterRequest spill_request = request;
    spill_request.force_spill = true;
    spill_request.forbidden = {};
    if (options.moi_owner_vgpr)
      spill_request.forbidden.expand({RegClass::VGPR, *options.moi_owner_vgpr, 1});
    if (options.moi_epoch_vgpr)
      spill_request.forbidden.expand({RegClass::VGPR, *options.moi_epoch_vgpr, 1});
    if (options.moi_workgroup_key_vgpr)
      spill_request.forbidden.expand({RegClass::VGPR, *options.moi_workgroup_key_vgpr, 1});
    for (uint64_t descriptor_offset : plan.owner_descriptor_file_offsets) {
      const auto *assignment = find_moi_persistent_vgpr_assignment(options, descriptor_offset);
      if (!assignment)
        continue;
      spill_request.forbidden.expand({RegClass::VGPR, assignment->owner_vgpr, 1});
      spill_request.forbidden.expand({RegClass::VGPR, assignment->epoch_vgpr, 1});
      if (assignment->workgroup_key_vgpr)
        spill_request.forbidden.expand({RegClass::VGPR, *assignment->workgroup_key_vgpr, 1});
      if (assignment->sample_sequence_vgpr)
        spill_request.forbidden.expand({RegClass::VGPR, *assignment->sample_sequence_vgpr, 1});
    }
    register_plan = plan_consan_registers(spill_request, live_before_all_owners);
  }
  plan.source = register_plan.source;
  plan.reason = register_plan.reason;
  plan.scratch_vgpr = register_plan.base;
  plan.required_vgpr_count = register_plan.required_descriptor_count;
  if (plan.source == ConSanRegisterAllocationSource::SpillRequired &&
      options.moi_engine != ConSanMoiEngine::InlineShadow &&
      std::ranges::any_of(owners, [](const MoiKernelResourceContext *owner) {
        return owner->kernel->uses_dynamic_stack.value_or(false);
      })) {
    plan.source = ConSanRegisterAllocationSource::Unsupported;
    plan.reason = ConSanRegisterPlanReason::DynamicStack;
    plan.scratch_vgpr.reset();
    plan.required_vgpr_count = plan.current_vgpr_count;
  }
  return plan;
}

void append_moi_resource_plans(MoiResourcePlanningState &state, const ConSanOptions &options,
                               ConSanResult &result) {
  result.resource_plans.clear();
  result.resource_plans.reserve(result.moi_candidates.size());
  for (size_t candidate_index = 0; candidate_index < result.moi_candidates.size();
       ++candidate_index) {
    const ConSanMoiCandidate &candidate = result.moi_candidates[candidate_index];
    const uint16_t scratch_count = moi_access_scratch_vgpr_count(options, candidate, state.arch);
    result.resource_plans.push_back(plan_moi_resource_site(
        state, options, ConSanResourceSiteKind::Access, candidate_index, candidate.text_offset,
        candidate.kernel_descriptor_file_offset, scratch_count, &candidate));
  }
}

struct ResolvedMoiScratchPlan {
  uint16_t base = 0;
  uint16_t count = 0;
  uint16_t required_vgpr_count = 0;
  uint32_t original_private_segment_size = 0;
  std::vector<uint64_t> owner_descriptor_file_offsets;
  ConSanRegisterAllocationSource source = ConSanRegisterAllocationSource::Unsupported;
};

[[nodiscard]] const ConSanCandidateResourcePlan *
resource_plan_for_candidate(const ConSanResult &result, const ConSanMoiCandidate &candidate) {
  if (result.moi_candidates.empty())
    return nullptr;
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(result.moi_candidates.data());
  const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(&candidate);
  if (address < begin)
    return nullptr;
  const std::uintptr_t byte_offset = address - begin;
  if (byte_offset % sizeof(ConSanMoiCandidate) != 0u)
    return nullptr;
  const size_t index = static_cast<size_t>(byte_offset / sizeof(ConSanMoiCandidate));
  if (index >= result.resource_plans.size() || index >= result.moi_candidates.size() ||
      &result.moi_candidates[index] != &candidate ||
      result.resource_plans[index].candidate_index != index) {
    return nullptr;
  }
  return &result.resource_plans[index];
}

[[nodiscard]] const ConSanCandidateResourcePlan *
resource_plan_for_site(const ConSanResult &result, ConSanResourceSiteKind site_kind,
                       uint64_t text_offset) {
  const auto it = std::ranges::find_if(result.resource_plans, [&](const auto &plan) {
    return plan.site_kind == site_kind && plan.text_offset == text_offset;
  });
  return it == result.resource_plans.end() ? nullptr : &*it;
}

[[nodiscard]] std::optional<ResolvedMoiScratchPlan>
resolve_moi_scratch_plan(const ConSanCandidateResourcePlan &plan, const ConSanOptions &options,
                         uint16_t expected_count) {
  if (plan.scratch_vgpr && plan.scratch_vgpr_count == expected_count &&
      (plan.source == ConSanRegisterAllocationSource::Explicit ||
       plan.source == ConSanRegisterAllocationSource::LivenessDead ||
       plan.source == ConSanRegisterAllocationSource::DescriptorGrowth ||
       plan.source == ConSanRegisterAllocationSource::SpillRequired)) {
    ResolvedMoiScratchPlan resolved;
    resolved.base = *plan.scratch_vgpr;
    resolved.count = expected_count;
    resolved.required_vgpr_count = plan.required_vgpr_count;
    resolved.original_private_segment_size = plan.original_private_segment_size;
    resolved.owner_descriptor_file_offsets = plan.owner_descriptor_file_offsets;
    resolved.source = plan.source;
    if (options.moi_owner_vgpr) {
      resolved.required_vgpr_count = std::max<uint16_t>(
          resolved.required_vgpr_count, static_cast<uint16_t>(*options.moi_owner_vgpr + 1u));
    }
    if (options.moi_epoch_vgpr) {
      resolved.required_vgpr_count = std::max<uint16_t>(
          resolved.required_vgpr_count, static_cast<uint16_t>(*options.moi_epoch_vgpr + 1u));
    }
    if (options.moi_workgroup_key_vgpr) {
      resolved.required_vgpr_count =
          std::max<uint16_t>(resolved.required_vgpr_count,
                             static_cast<uint16_t>(*options.moi_workgroup_key_vgpr + 1u));
    }
    for (uint64_t descriptor_offset : plan.owner_descriptor_file_offsets) {
      const auto *assignment = find_moi_persistent_vgpr_assignment(options, descriptor_offset);
      if (!assignment)
        continue;
      uint16_t end = static_cast<uint16_t>(assignment->epoch_vgpr + 1u);
      if (assignment->workgroup_key_vgpr)
        end = std::max<uint16_t>(end, static_cast<uint16_t>(*assignment->workgroup_key_vgpr + 1u));
      if (assignment->sample_sequence_vgpr)
        end =
            std::max<uint16_t>(end, static_cast<uint16_t>(*assignment->sample_sequence_vgpr + 1u));
      resolved.required_vgpr_count = std::max(resolved.required_vgpr_count, end);
    }
    return resolved;
  }

  return std::nullopt;
}

[[nodiscard]] std::optional<ResolvedMoiScratchPlan>
resolve_moi_scratch(const ConSanResult &result, const ConSanMoiCandidate &candidate,
                    const ConSanOptions &options, uint16_t expected_count) {
  const ConSanCandidateResourcePlan *plan = resource_plan_for_candidate(result, candidate);
  return plan == nullptr ? std::nullopt : resolve_moi_scratch_plan(*plan, options, expected_count);
}

void apply_test_kernel_filter(std::vector<const ConSanMoiCandidate *> &candidates,
                              const ConSanOptions &options) {
  if (options.test_kernel_name_filter.empty())
    return;
  std::erase_if(candidates, [&](const ConSanMoiCandidate *candidate) {
    return candidate == nullptr ||
           candidate->container_name.find(options.test_kernel_name_filter) == std::string::npos;
  });
}

template <typename Candidate>
void apply_test_container_filter(std::vector<Candidate> &candidates, const ConSanOptions &options) {
  if (options.test_kernel_name_filter.empty())
    return;
  std::erase_if(candidates, [&](const Candidate &candidate) {
    return candidate.container_name.find(options.test_kernel_name_filter) == std::string::npos;
  });
}

[[nodiscard]] uint16_t moi_exec_save_sgpr_count(const ConSanOptions &options, rj_code_arch_t arch) {
  if (options.moi_engine == ConSanMoiEngine::Sampled)
    // Claim selection and optional wave sampling predicate on VCC. A winning
    // lane narrows EXEC while publishing, and every scalar predicate may
    // change SCC. The runtime fast gate and sampled-sync cave are mutually
    // exclusive bodies and share the same eight-register layout through +7.
    // gfx1250 dense access streams additionally reserve an aligned temporary
    // call-return pair through +7. The body returns with its ordinary
    // indirect-PC pair, so the probe may reuse that temporary after dispatch.
    return options.moi_report_buffer_address
               ? (arch == ROCJITSU_CODE_ARCH_GFX1250 ? 8u : (options.moi_track_atomics ? 8u : 7u))
               : 0u;
  if (options.moi_engine == ConSanMoiEngine::InlineShadow) {
    if (!options.moi_report_buffer_address)
      return 0;
    // Access probes reserve every transient mask plus the dedicated
    // application EXEC pair. Atomic-only objects use masks through +21 and do
    // not pay for access state they cannot execute.
    return options.moi_inline_access_present ? kConSanMoiInlineExecSaveSgprCount : 22u;
  }
  if (options.automatic_moi_record_replay_sgpr_spill)
    return 4u;
  // One EXEC-save pair, one VCC-save pair, and one SCC snapshot. Barrier-dense
  // RDNA-family objects additionally reserve a dispatch key and an aligned
  // call-return pair for the relocated router.
  return options.moi_report_buffer_address
             ? (is_rdna4_family_arch(arch) && options.moi_record_replay_dense_barrier_router ? 8u
                                                                                             : 5u)
             : 0u;
}

[[nodiscard]] std::optional<uint16_t>
inline_shadow_visible_evidence_sgpr(const ConSanOptions &options) {
  if (options.moi_engine != ConSanMoiEngine::InlineShadow || !options.moi_inline_workgroup_shadow ||
      !options.moi_exec_save_sgpr) {
    return std::nullopt;
  }
  if (options.automatic_moi_inline_sgpr_spill)
    return options.moi_inline_visible_evidence_sgpr;
  if (!options.moi_exec_save_sgprs_persistent)
    return std::nullopt;
  return static_cast<uint16_t>(*options.moi_exec_save_sgpr +
                               (options.moi_dynamic_stack_spill ? 25u : 24u));
}

[[nodiscard]] constexpr uint16_t moi_ordinary_sgpr_limit(rj_code_arch_t arch) {
  // On CDNA4, encoded SGPR numbers 102:103 and 104:105 name FLAT_SCRATCH and
  // XNACK_MASK rather than ordinary scalar registers.  RDNA4 retains ordinary
  // SGPRs through s105 before VCC at s106.  Instrumentation state must never
  // treat either CDNA4 special pair as allocatable storage.
  return arch == ROCJITSU_CODE_ARCH_CDNA4 ? 102u : kMaxSgprs;
}

[[nodiscard]] uint16_t moi_dispatch_state_sgpr_count(const ConSanOptions &options) {
  (void)options;
  return 2u;
}

[[nodiscard]] std::optional<uint16_t>
moi_runtime_sample_counter_sgpr(const ConSanOptions &options) {
  // Runtime sampling is a stable hash of dispatch/workgroup identity. The old
  // per-site scalar counter was removed from the gate because it separated
  // producer and consumer evidence, so no persistent third SGPR is needed.
  (void)options;
  return std::nullopt;
}

struct MoiSpecialStateSgprs {
  uint16_t vcc_save_sgpr = 0;
  uint16_t scc_save_sgpr = 0;
};

struct MoiIndirectJumpSgprs {
  uint16_t pc_sgpr = 0;
  uint16_t scc_save_sgpr = 0;
};

[[nodiscard]] std::optional<MoiSpecialStateSgprs>
moi_special_state_sgprs(const ConSanOptions &options) {
  if (!options.moi_exec_save_sgpr)
    return std::nullopt;
  const uint16_t vcc_offset = options.moi_engine == ConSanMoiEngine::InlineShadow ? 8u : 2u;
  return MoiSpecialStateSgprs{
      .vcc_save_sgpr = static_cast<uint16_t>(*options.moi_exec_save_sgpr + vcc_offset),
      .scc_save_sgpr =
          options.automatic_moi_record_replay_sgpr_spill && options.moi_inline_indirect_scc_sgpr
              ? *options.moi_inline_indirect_scc_sgpr
              : static_cast<uint16_t>(*options.moi_exec_save_sgpr + vcc_offset + 2u),
  };
}

[[nodiscard]] std::optional<MoiIndirectJumpSgprs>
moi_indirect_jump_sgprs(const ConSanOptions &options) {
  if (options.automatic_moi_inline_sgpr_spill || options.automatic_moi_record_replay_sgpr_spill) {
    if (!options.moi_inline_indirect_pc_sgpr || !options.moi_inline_indirect_scc_sgpr)
      return std::nullopt;
    return MoiIndirectJumpSgprs{.pc_sgpr = *options.moi_inline_indirect_pc_sgpr,
                                .scc_save_sgpr = *options.moi_inline_indirect_scc_sgpr};
  }
  const auto special_state = moi_special_state_sgprs(options);
  if (!options.moi_exec_save_sgpr || !special_state)
    return std::nullopt;
  if (options.moi_engine == ConSanMoiEngine::InlineShadow) {
    return MoiIndirectJumpSgprs{
        .pc_sgpr = static_cast<uint16_t>(*options.moi_exec_save_sgpr + 12u),
        .scc_save_sgpr = static_cast<uint16_t>(*options.moi_exec_save_sgpr + 10u),
    };
  }
  // Record/Replay reserves five SGPRs. Its indirect island runs before the
  // body snapshots EXEC/VCC/SCC, and its return runs after those snapshots
  // have been restored, so the EXEC-save pair and SCC slot are free scratch
  // at both boundaries. Do not address Inline's wider +10/+12 slots here.
  return MoiIndirectJumpSgprs{
      .pc_sgpr = *options.moi_exec_save_sgpr,
      .scc_save_sgpr = special_state->scc_save_sgpr,
  };
}

[[nodiscard]] bool append_save_moi_special_state(std::vector<uint32_t> &words,
                                                 const ConSanOptions &options,
                                                 rj_code_arch_t arch) {
  const auto registers = moi_special_state_sgprs(options);
  if (!registers)
    return false;
  const auto save_scc = instrumentation::build_s_cselect_b32(
      registers->scc_save_sgpr, scalar_positive_inline_u32(1), scalar_positive_inline_u32(0), arch);
  const auto save_vcc =
      instrumentation::build_s_mov_b64(registers->vcc_save_sgpr, kRdna4VccLo, arch);
  if (!save_scc || !save_vcc)
    return false;
  // Snapshot SCC before any instrumentation instruction can modify it.
  words.push_back(*save_scc);
  words.push_back(*save_vcc);
  return true;
}

[[nodiscard]] bool append_restore_moi_special_state(std::vector<uint32_t> &words,
                                                    const ConSanOptions &options,
                                                    rj_code_arch_t arch) {
  const auto registers = moi_special_state_sgprs(options);
  if (!registers)
    return false;
  const auto restore_vcc =
      instrumentation::build_s_mov_b64(kRdna4VccLo, registers->vcc_save_sgpr, arch);
  const auto restore_scc = instrumentation::build_s_cmp_lg_u32(registers->scc_save_sgpr,
                                                               scalar_positive_inline_u32(0), arch);
  if (!restore_vcc || !restore_scc)
    return false;
  words.push_back(*restore_vcc);
  // Keep this last: every scalar comparison or saveexec before it may write SCC.
  words.push_back(*restore_scc);
  return true;
}

[[nodiscard]] bool
configure_automatic_moi_transient_sgpr_assignments(ConSanOptions &options, ConSanResult &result,
                                                   std::span<const uint8_t> bytes,
                                                   rj_code_arch_t arch) {
  if ((options.moi_engine != ConSanMoiEngine::RecordReplay &&
       options.moi_engine != ConSanMoiEngine::InlineShadow) ||
      !options.automatic_moi_partial_exec_save_sgprs || !options.moi_exec_save_sgpr)
    return false;

  MoiResourcePlanningState state(bytes, arch, options, result);
  if (!state.valid)
    return false;

  std::unordered_map<uint64_t, uint64_t> parent;
  const auto ensure_owner = [&](uint64_t descriptor_offset) {
    parent.try_emplace(descriptor_offset, descriptor_offset);
  };
  for (const ConSanCandidateResourcePlan &plan : result.resource_plans)
    for (uint64_t descriptor_offset : plan.owner_descriptor_file_offsets)
      ensure_owner(descriptor_offset);
  const auto find_root = [&](uint64_t owner) {
    uint64_t root = owner;
    while (parent.at(root) != root)
      root = parent.at(root);
    uint64_t cursor = owner;
    while (parent.at(cursor) != cursor) {
      const uint64_t next = parent.at(cursor);
      parent[cursor] = root;
      cursor = next;
    }
    return root;
  };
  for (const ConSanCandidateResourcePlan &plan : result.resource_plans) {
    if (plan.owner_descriptor_file_offsets.empty())
      continue;
    const uint64_t first = plan.owner_descriptor_file_offsets.front();
    for (uint64_t owner : plan.owner_descriptor_file_offsets) {
      const uint64_t lhs = find_root(first);
      const uint64_t rhs = find_root(owner);
      if (lhs != rhs)
        parent[rhs] = lhs;
    }
  }

  std::unordered_map<uint64_t, RegisterSet> unavailable_by_owner;
  const auto include_site = [&](uint64_t text_offset) {
    BasicBlock *block = block_for_offset(state.block_index, text_offset);
    const Instruction *instruction = instruction_at(block, text_offset);
    if (block == nullptr || instruction == nullptr)
      return;
    for (const MoiKernelResourceContext &context : state.contexts) {
      if (context.kernel == nullptr ||
          std::ranges::find(context.scope.blocks, block) == context.scope.blocks.end())
        continue;
      RegisterSet &unavailable = unavailable_by_owner[context.kernel->descriptor_file_offset];
      unavailable |= context.liveness->live_before(*instruction);
      const InstDefUse def_use(*instruction);
      unavailable |= def_use.defs | def_use.uses;
    }
  };
  for (const ConSanMoiCandidate &candidate : result.moi_candidates)
    include_site(candidate.text_offset);
  for (const ConSanCandidateResourcePlan &plan : result.resource_plans)
    include_site(plan.text_offset);
  for (const ConSanSiteDispositionRecord &site : result.site_dispositions) {
    if (site.disposition == ConSanSiteDisposition::Supported)
      include_site(site.text_offset);
  }
  if (options.moi_init_owner_epoch) {
    for (const ConSanKernelInfo &kernel : result.kernels)
      include_site(kernel.entry_text_offset);
  }

  std::unordered_map<uint64_t, RegisterSet> unavailable_by_component;
  std::unordered_map<uint64_t, std::vector<uint64_t>> owners_by_component;
  for (const auto &[owner, ignored] : parent) {
    (void)ignored;
    const uint64_t root = find_root(owner);
    owners_by_component[root].push_back(owner);
    if (const auto it = unavailable_by_owner.find(owner); it != unavailable_by_owner.end())
      unavailable_by_component[root] |= it->second;
  }

  const uint16_t count = moi_exec_save_sgpr_count(options, arch);
  const uint16_t limit = moi_ordinary_sgpr_limit(arch);
  const auto context_for_owner = [&](uint64_t owner) -> const MoiKernelResourceContext * {
    const auto it = std::ranges::find_if(state.contexts, [&](const auto &context) {
      return context.kernel != nullptr && context.kernel->descriptor_file_offset == owner;
    });
    return it == state.contexts.end() ? nullptr : &*it;
  };
  const auto intersects = [](const RegisterSet &set, uint16_t base, uint16_t width) {
    for (uint16_t i = 0; i < width; ++i)
      if (set.contains({RegClass::SGPR, static_cast<uint16_t>(base + i), 1u}))
        return true;
    return false;
  };

  bool assigned_any = false;
  for (auto &[root, owners] : owners_by_component) {
    const bool needs_override = std::ranges::any_of(owners, [&](uint64_t owner) {
      const MoiKernelResourceContext *context = context_for_owner(owner);
      return context != nullptr && context->max_referenced_sgpr_count > *options.moi_exec_save_sgpr;
    });
    if (!needs_override)
      continue;
    const RegisterSet &unavailable = unavailable_by_component[root];
    std::optional<uint16_t> selected;
    std::optional<uint16_t> spill_pc;
    std::optional<uint16_t> spill_call_return;
    std::optional<uint16_t> spill_scc;
    std::optional<uint16_t> spill_key;
    std::optional<uint16_t> spill_visible;
    if (options.moi_engine == ConSanMoiEngine::RecordReplay) {
      for (uint32_t base = 0; base + count <= limit; base += 2u) {
        const uint16_t candidate = static_cast<uint16_t>(base);
        if (intersects(unavailable, candidate, count) ||
            (options.moi_owner_sgpr &&
             range_overlaps(candidate, count, *options.moi_owner_sgpr, 1u)) ||
            (options.moi_dispatch_id_sgpr &&
             range_overlaps(candidate, count, *options.moi_dispatch_id_sgpr,
                            moi_dispatch_state_sgpr_count(options))) ||
            (options.moi_dynamic_stack_spill &&
             range_overlaps(candidate, count, /*stack top=*/32u, /*stack registers=*/2u)))
          continue;
        const bool overlaps_original_cdna4_vcc =
            arch == ROCJITSU_CODE_ARCH_CDNA4 && std::ranges::any_of(owners, [&](uint64_t owner) {
              const MoiKernelResourceContext *context = context_for_owner(owner);
              return context != nullptr && context->current_sgpr_count >= 2u &&
                     range_overlaps(candidate, count,
                                    static_cast<uint16_t>(context->current_sgpr_count - 2u), 2u);
            });
        if (overlaps_original_cdna4_vcc)
          continue;
        selected = candidate;
        break;
      }
    }
    if (!selected && options.moi_engine == ConSanMoiEngine::InlineShadow &&
        is_rdna4_family_arch(arch) && !options.moi_dynamic_stack_spill &&
        address_free_scratch_private_limit(arch)) {
      const auto reserved = [&](uint16_t base, uint16_t width) {
        return (options.moi_owner_sgpr &&
                range_overlaps(base, width, *options.moi_owner_sgpr, 1u)) ||
               (options.moi_dispatch_id_sgpr &&
                range_overlaps(base, width, *options.moi_dispatch_id_sgpr,
                               moi_dispatch_state_sgpr_count(options)));
      };
      uint32_t initialized_end = 0u;
      uint32_t referenced_end = 0u;
      for (uint64_t owner : owners) {
        const MoiKernelResourceContext *context = context_for_owner(owner);
        if (context)
          referenced_end = std::max<uint32_t>(referenced_end, context->max_referenced_sgpr_count);
        if (owner > bytes.size() || sizeof(KD) > bytes.size() - owner)
          continue;
        KD descriptor{};
        std::memcpy(&descriptor, bytes.data() + owner, sizeof(descriptor));
        initialized_end =
            std::max<uint32_t>(initialized_end, moi_descriptor_user_sgpr_count(descriptor) +
                                                    moi_descriptor_system_sgpr_count(descriptor));
      }
      const auto unavailable_or_reserved = [&](uint16_t base, uint16_t width, uint16_t spill_base) {
        return range_overlaps(base, width, spill_base, count) || reserved(base, width) ||
               intersects(unavailable, base, width);
      };
      for (uint32_t base = util::align_up(initialized_end, 2u); !selected && base + count <= limit;
           base += 2u) {
        const uint16_t spill_base = static_cast<uint16_t>(base);
        if (reserved(spill_base, count))
          continue;
        for (uint32_t reg = referenced_end; reg < limit; ++reg) {
          const uint16_t candidate = static_cast<uint16_t>(reg);
          if (!range_overlaps(candidate, 1u, spill_base, count) && !reserved(candidate, 1u)) {
            spill_visible = candidate;
            break;
          }
        }
        for (uint32_t pc = 0; pc + 2u <= limit; pc += 2u) {
          const uint16_t candidate = static_cast<uint16_t>(pc);
          if (!unavailable_or_reserved(candidate, 2u, spill_base) &&
              (!spill_visible || !range_overlaps(candidate, 2u, *spill_visible, 1u))) {
            spill_pc = candidate;
            break;
          }
        }
        for (uint32_t reg = 0; spill_pc && reg < limit; ++reg) {
          const uint16_t candidate = static_cast<uint16_t>(reg);
          if (!unavailable_or_reserved(candidate, 1u, spill_base) && candidate != *spill_visible &&
              !range_overlaps(candidate, 1u, *spill_pc, 2u)) {
            spill_scc = candidate;
            break;
          }
        }
        for (uint32_t reg = 0; spill_scc && reg < limit; ++reg) {
          const uint16_t candidate = static_cast<uint16_t>(reg);
          if (!unavailable_or_reserved(candidate, 1u, spill_base) &&
              (!spill_visible || candidate != *spill_visible) &&
              !range_overlaps(candidate, 1u, *spill_pc, 2u) && candidate != *spill_scc) {
            spill_key = candidate;
            break;
          }
        }
        for (uint32_t call = 0; spill_key && call + 2u <= limit; call += 2u) {
          const uint16_t candidate = static_cast<uint16_t>(call);
          if (!unavailable_or_reserved(candidate, 2u, spill_base) &&
              (!spill_visible || !range_overlaps(candidate, 2u, *spill_visible, 1u)) &&
              !range_overlaps(candidate, 2u, *spill_pc, 2u) &&
              !range_overlaps(candidate, 2u, *spill_scc, 1u) &&
              !range_overlaps(candidate, 2u, *spill_key, 1u)) {
            spill_call_return = candidate;
            break;
          }
        }
        if (spill_call_return)
          selected = spill_base;
        else
          spill_visible = spill_pc = spill_scc = spill_key = std::nullopt;
      }
      // If no complete spill router exists, an owner-local window that is
      // dead at every instrumentation site remains useful for external-shadow
      // probes.  It cannot carry the entry-to-probe visible-evidence latch;
      // access selection therefore keeps such owners off the local LDS mirror.
      if (!selected) {
        for (uint32_t base = util::align_up(initialized_end, 2u); base + count <= limit;
             base += 2u) {
          const uint16_t candidate = static_cast<uint16_t>(base);
          if (!reserved(candidate, count) && !intersects(unavailable, candidate, count)) {
            selected = candidate;
            break;
          }
        }
      }
    }
    if (!selected && options.moi_engine == ConSanMoiEngine::RecordReplay &&
        is_rdna4_family_arch(arch) && !options.moi_dynamic_stack_spill &&
        address_free_scratch_private_limit(arch)) {
      const auto reserved = [&](uint16_t base, uint16_t width) {
        return (options.moi_owner_sgpr &&
                range_overlaps(base, width, *options.moi_owner_sgpr, 1u)) ||
               (options.moi_dispatch_id_sgpr &&
                range_overlaps(base, width, *options.moi_dispatch_id_sgpr,
                               moi_dispatch_state_sgpr_count(options)));
      };
      for (uint32_t base = 0; base + 2u <= limit; base += 2u) {
        if (!intersects(unavailable, static_cast<uint16_t>(base), 2u) &&
            !reserved(static_cast<uint16_t>(base), 2u)) {
          spill_pc = static_cast<uint16_t>(base);
          break;
        }
      }
      for (uint32_t reg = 0; spill_pc && reg < limit; ++reg) {
        const uint16_t candidate = static_cast<uint16_t>(reg);
        if (!intersects(unavailable, candidate, 1u) && !reserved(candidate, 1u) &&
            !range_overlaps(candidate, 1u, *spill_pc, 2u)) {
          spill_scc = candidate;
          break;
        }
      }
      for (uint32_t reg = 0; spill_scc && reg < limit; ++reg) {
        const uint16_t candidate = static_cast<uint16_t>(reg);
        if (!intersects(unavailable, candidate, 1u) && !reserved(candidate, 1u) &&
            !range_overlaps(candidate, 1u, *spill_pc, 2u) && candidate != *spill_scc) {
          spill_key = candidate;
          break;
        }
      }
      // Dense s_call anchors first write their return PC into this pair.  The
      // entry island copies its low word into the key before reusing the pair
      // for the long jump, and the body reconstructs its known continuation.
      // One dead pair is therefore sufficient for both transient roles.
      if (spill_key)
        spill_call_return = spill_pc;
      constexpr uint16_t kSpilledExecVccCount = 4u;
      for (uint32_t base = 0; spill_call_return && base + kSpilledExecVccCount <= limit;
           base += 2u) {
        const uint16_t candidate = static_cast<uint16_t>(base);
        if (!reserved(candidate, kSpilledExecVccCount) &&
            !range_overlaps(candidate, kSpilledExecVccCount, *spill_pc, 2u) &&
            !range_overlaps(candidate, kSpilledExecVccCount, *spill_scc, 1u) &&
            !range_overlaps(candidate, kSpilledExecVccCount, *spill_key, 1u) &&
            !range_overlaps(candidate, kSpilledExecVccCount, *spill_call_return, 2u)) {
          selected = candidate;
          break;
        }
      }
    }
    if (!selected)
      continue;
    const bool spill_backed =
        spill_pc && spill_scc && spill_key && spill_call_return &&
        (options.moi_engine != ConSanMoiEngine::InlineShadow || spill_visible);
    uint32_t component_max_referenced_sgpr_count = 0u;
    for (uint64_t owner : owners) {
      const MoiKernelResourceContext *context = context_for_owner(owner);
      if (context) {
        component_max_referenced_sgpr_count = std::max<uint32_t>(
            component_max_referenced_sgpr_count, context->max_referenced_sgpr_count);
      }
    }
    const std::optional<uint16_t> dispatch_id_sgpr =
        options.moi_dispatch_id_sgpr &&
                component_max_referenced_sgpr_count <= *options.moi_dispatch_id_sgpr
            ? options.moi_dispatch_id_sgpr
            : std::nullopt;
    for (uint64_t owner : owners) {
      options.moi_transient_sgpr_assignments.push_back({.descriptor_file_offset = owner,
                                                        .exec_save_sgpr = *selected,
                                                        .dispatch_id_sgpr = dispatch_id_sgpr,
                                                        .spill_backed = spill_backed,
                                                        .indirect_pc_sgpr = spill_pc,
                                                        .indirect_scc_sgpr = spill_scc,
                                                        .dispatch_key_sgpr = spill_key,
                                                        .call_return_sgpr = spill_call_return,
                                                        .visible_evidence_sgpr = spill_visible});
    }
    assigned_any = true;
  }
  if (!assigned_any)
    return false;
  result.resolved_moi_transient_sgpr_assignments = options.moi_transient_sgpr_assignments;
  result.warnings.emplace_back(
      "ConSan MOI assigned owner-local transient SGPR windows to independent owner components");
  return true;
}

[[nodiscard]] bool configure_automatic_moi_exec_save_sgprs(ConSanOptions &options,
                                                           ConSanResult &result,
                                                           std::span<const uint8_t> bytes,
                                                           rj_code_arch_t arch) {
  if (options.moi_exec_save_sgpr)
    return false;
  const uint16_t count = moi_exec_save_sgpr_count(options, arch);
  if (count == 0)
    return false;

  uint32_t first = 0;
  bool found_owned_scope = false;
  for (const ConSanCandidateResourcePlan &plan : result.resource_plans) {
    if (plan.owner_descriptor_file_offsets.empty())
      continue;
    found_owned_scope = true;
    first = std::max<uint32_t>(first, plan.max_referenced_sgpr_count);
  }
  if (!found_owned_scope) {
    result.warnings.emplace_back(
        "ConSan MOI could not derive automatic EXEC-save SGPRs without an owned candidate "
        "scope");
    return false;
  }

  first = util::align_up(first, 2u);
  const uint16_t ordinary_sgpr_limit = moi_ordinary_sgpr_limit(arch);
  const auto overlaps_original_cdna4_vcc = [&](uint16_t base) {
    if (arch != ROCJITSU_CODE_ARCH_CDNA4)
      return false;
    return std::ranges::any_of(result.resource_plans, [&](const ConSanCandidateResourcePlan &plan) {
      if (plan.owner_descriptor_file_offsets.empty() || plan.current_sgpr_count < 2u)
        return false;
      // CDNA4 physically places VCC in the highest two allocated user SGPRs.
      // Existing code may name that physical pair by logical SGPR number, so
      // descriptor growth does not prove the old pair dead even when decoded
      // ordinary operands stop below it.
      return range_overlaps(base, count, static_cast<uint16_t>(plan.current_sgpr_count - 2u), 2u);
    });
  };
  for (uint32_t base = first; base + count <= ordinary_sgpr_limit; base += 2u) {
    // Dynamic-stack kernels use s32 as the stack top and s33 as the current
    // frame base even when those architectural roles are absent from ordinary
    // decoded operand liveness. Never assign either register to persistent
    // Inline scalar state.
    if (overlaps_original_cdna4_vcc(static_cast<uint16_t>(base)) ||
        (options.moi_dynamic_stack_spill &&
         range_overlaps(static_cast<uint16_t>(base), count, /*stack top=*/32u,
                        /*stack registers=*/2u))) {
      continue;
    }
    if (options.moi_owner_sgpr &&
        range_overlaps(static_cast<uint16_t>(base), count, *options.moi_owner_sgpr, 1u)) {
      continue;
    }
    if (options.moi_dispatch_id_sgpr &&
        range_overlaps(static_cast<uint16_t>(base), count, *options.moi_dispatch_id_sgpr,
                       moi_dispatch_state_sgpr_count(options))) {
      continue;
    }
    options.moi_exec_save_sgpr = static_cast<uint16_t>(base);
    options.automatic_moi_exec_save_sgprs = true;
    options.moi_exec_save_sgprs_persistent = options.moi_inline_workgroup_shadow;
    result.resolved_moi_exec_save_sgpr = options.moi_exec_save_sgpr;
    result.moi_exec_save_sgprs_automatic = true;
    result.warnings.emplace_back("ConSan MOI automatically assigned EXEC-save SGPRs s" +
                                 std::to_string(base) + ":s" + std::to_string(base + count - 1u));
    return true;
  }

  // A high-numbered scalar operand anywhere in a kernel does not imply that
  // every lower SGPR is live at every instrumentation site. Large generated
  // kernels commonly use s104 for a short-lived address calculation while a
  // five-register window is dead throughout all LDS/synchronization sites.
  // Prove one common window with the existing CFG liveness before falling
  // back to an unsupported resource result.
  MoiResourcePlanningState state(bytes, arch, options, result);
  std::unordered_map<const BasicBlock *, std::vector<const MoiKernelResourceContext *>>
      contexts_by_block;
  contexts_by_block.reserve(state.blocks.size());
  for (const MoiKernelResourceContext &context : state.contexts) {
    for (const BasicBlock *block : context.scope.blocks)
      contexts_by_block[block].push_back(&context);
  }
  RegisterSet unavailable;
  bool saw_site = false;
  const auto include_site = [&](uint64_t text_offset) {
    BasicBlock *block = block_for_offset(state.block_index, text_offset);
    const Instruction *instruction = instruction_at(block, text_offset);
    if (block == nullptr || instruction == nullptr)
      return;
    const auto owners = contexts_by_block.find(block);
    if (owners == contexts_by_block.end())
      return;
    for (const MoiKernelResourceContext *context : owners->second) {
      unavailable |= context->liveness->live_before(*instruction);
      const InstDefUse def_use(*instruction);
      unavailable |= def_use.defs | def_use.uses;
      saw_site = true;
    }
  };
  for (const ConSanMoiCandidate &candidate : result.moi_candidates)
    include_site(candidate.text_offset);
  for (const ConSanCandidateResourcePlan &plan : result.resource_plans)
    include_site(plan.text_offset);
  for (const ConSanSiteDispositionRecord &site : result.site_dispositions) {
    if (site.disposition == ConSanSiteDisposition::Supported)
      include_site(site.text_offset);
  }
  if (options.moi_init_owner_epoch) {
    for (const ConSanKernelInfo &kernel : result.kernels)
      include_site(kernel.entry_text_offset);
  }
  if (state.valid && saw_site) {
    const auto unavailable_intersects = [&](uint16_t base) {
      for (uint16_t i = 0; i < count; ++i) {
        if (unavailable.contains({RegClass::SGPR, static_cast<uint16_t>(base + i), 1u})) {
          return true;
        }
      }
      return false;
    };
    for (uint32_t base = 0; base + count <= ordinary_sgpr_limit; base += 2u) {
      if (unavailable_intersects(static_cast<uint16_t>(base)) ||
          overlaps_original_cdna4_vcc(static_cast<uint16_t>(base)) ||
          (options.moi_dynamic_stack_spill &&
           range_overlaps(static_cast<uint16_t>(base), count, /*stack top=*/32u,
                          /*stack registers=*/2u)) ||
          (options.moi_owner_sgpr &&
           range_overlaps(static_cast<uint16_t>(base), count, *options.moi_owner_sgpr, 1u)) ||
          (options.moi_dispatch_id_sgpr &&
           range_overlaps(static_cast<uint16_t>(base), count, *options.moi_dispatch_id_sgpr,
                          moi_dispatch_state_sgpr_count(options)))) {
        continue;
      }
      options.moi_exec_save_sgpr = static_cast<uint16_t>(base);
      options.automatic_moi_exec_save_sgprs = true;
      result.resolved_moi_exec_save_sgpr = options.moi_exec_save_sgpr;
      result.moi_exec_save_sgprs_automatic = true;
      result.warnings.emplace_back(
          "ConSan MOI automatically assigned liveness-dead EXEC-save SGPRs s" +
          std::to_string(base) + ":s" + std::to_string(base + count - 1u));
      return true;
    }
  }

  // Prefer a high window for owners that already fit and component-scoped
  // assignments for the independent owners that do not.  Trying this before
  // the monolithic Inline spill fallback avoids spilling every kernel in a
  // generated library merely because one component has high scalar pressure.
  if (is_rdna4_family_arch(arch) && options.moi_engine == ConSanMoiEngine::InlineShadow) {
    const int32_t last_base = static_cast<int32_t>((ordinary_sgpr_limit - count) & ~1u);
    for (int32_t base = last_base; base >= 0; base -= 2) {
      const uint16_t candidate = static_cast<uint16_t>(base);
      if ((options.moi_owner_sgpr &&
           range_overlaps(candidate, count, *options.moi_owner_sgpr, 1u)) ||
          (options.moi_dispatch_id_sgpr &&
           range_overlaps(candidate, count, *options.moi_dispatch_id_sgpr,
                          moi_dispatch_state_sgpr_count(options))) ||
          (options.moi_dynamic_stack_spill &&
           range_overlaps(candidate, count, /*stack top=*/32u, /*stack registers=*/2u))) {
        continue;
      }
      const bool admits_owner =
          std::ranges::any_of(result.resource_plans, [&](const ConSanCandidateResourcePlan &plan) {
            return plan.source != ConSanRegisterAllocationSource::Unsupported &&
                   !plan.owner_descriptor_file_offsets.empty() &&
                   plan.max_referenced_sgpr_count <= candidate;
          });
      const bool excludes_owner =
          std::ranges::any_of(result.resource_plans, [&](const ConSanCandidateResourcePlan &plan) {
            return !plan.owner_descriptor_file_offsets.empty() &&
                   plan.max_referenced_sgpr_count > candidate;
          });
      if (!admits_owner || !excludes_owner)
        continue;
      options.moi_exec_save_sgpr = candidate;
      options.automatic_moi_exec_save_sgprs = true;
      options.automatic_moi_partial_exec_save_sgprs = true;
      options.moi_exec_save_sgprs_persistent = true;
      if (configure_automatic_moi_transient_sgpr_assignments(options, result, bytes, arch)) {
        result.resolved_moi_exec_save_sgpr = options.moi_exec_save_sgpr;
        result.moi_exec_save_sgprs_automatic = true;
        result.warnings.emplace_back(
            "ConSan MOI reserved a high EXEC-save SGPR window and assigned "
            "component-local scalar state to incompatible owners");
        return true;
      }
      options.moi_transient_sgpr_assignments.clear();
      result.resolved_moi_transient_sgpr_assignments.clear();
      options.moi_exec_save_sgpr.reset();
      options.automatic_moi_exec_save_sgprs = false;
      options.automatic_moi_partial_exec_save_sgprs = false;
      options.moi_exec_save_sgprs_persistent = false;
    }
  }

  // gfx12 can expose 28-register Inline access probes in kernels whose
  // scalar pressure leaves no equally large dead window. The access patcher
  // can preserve a transient window in private memory, but entry islands and
  // cross-site visible-evidence state still require genuinely dead/fresh
  // registers outside that window.
  if (is_rdna4_family_arch(arch) && options.moi_engine == ConSanMoiEngine::InlineShadow &&
      options.moi_inline_access_present && !options.moi_dynamic_stack_spill && state.valid &&
      saw_site) {
    const size_t ordinary_access_count =
        std::ranges::count_if(result.moi_candidates, [](const ConSanMoiCandidate &candidate) {
          return candidate.kind == ConSanLdsAccessKind::Read ||
                 candidate.kind == ConSanLdsAccessKind::Write;
        });
    const bool needs_dense_router =
        ordinary_access_count >= 8u ||
        std::ranges::any_of(result.moi_candidates, [](const ConSanMoiCandidate &candidate) {
          return !candidate.in_kernel && (candidate.kind == ConSanLdsAccessKind::Read ||
                                          candidate.kind == ConSanLdsAccessKind::Write);
        });
    uint32_t initialized_end = 0;
    for (const ConSanCandidateResourcePlan &plan : result.resource_plans) {
      for (uint64_t descriptor_offset : plan.owner_descriptor_file_offsets) {
        if (descriptor_offset > bytes.size() || sizeof(KD) > bytes.size() - descriptor_offset)
          continue;
        KD descriptor{};
        std::memcpy(&descriptor, bytes.data() + descriptor_offset, sizeof(descriptor));
        initialized_end =
            std::max<uint32_t>(initialized_end, moi_descriptor_user_sgpr_count(descriptor) +
                                                    moi_descriptor_system_sgpr_count(descriptor));
      }
    }
    const auto overlaps_reserved = [&](uint16_t base, uint16_t width) {
      return (options.moi_owner_sgpr && range_overlaps(base, width, *options.moi_owner_sgpr, 1u)) ||
             (options.moi_dispatch_id_sgpr &&
              range_overlaps(base, width, *options.moi_dispatch_id_sgpr,
                             moi_dispatch_state_sgpr_count(options)));
    };
    std::optional<uint16_t> spill_base;
    for (uint32_t base = util::align_up(initialized_end, 2u); base + count <= ordinary_sgpr_limit;
         base += 2u) {
      if (!overlaps_reserved(static_cast<uint16_t>(base), count)) {
        spill_base = static_cast<uint16_t>(base);
        break;
      }
    }
    if (spill_base) {
      const auto unavailable_or_reserved = [&](uint16_t base, uint16_t width) {
        if (range_overlaps(base, width, *spill_base, count) || overlaps_reserved(base, width))
          return true;
        for (uint16_t i = 0; i < width; ++i) {
          if (unavailable.contains({RegClass::SGPR, static_cast<uint16_t>(base + i), 1u}))
            return true;
        }
        return false;
      };
      std::optional<uint16_t> visible;
      for (uint32_t reg = first; reg < ordinary_sgpr_limit; ++reg) {
        if (!range_overlaps(static_cast<uint16_t>(reg), 1u, *spill_base, count) &&
            !overlaps_reserved(static_cast<uint16_t>(reg), 1u) &&
            !unavailable.contains({RegClass::SGPR, static_cast<uint16_t>(reg), 1u})) {
          visible = static_cast<uint16_t>(reg);
          break;
        }
      }
      std::optional<uint16_t> pc;
      std::optional<uint16_t> call_return;
      std::optional<uint16_t> dispatch_key;
      std::optional<uint16_t> scc;
      for (uint32_t base = 0; base + 2u <= ordinary_sgpr_limit; base += 2u) {
        if (!unavailable_or_reserved(static_cast<uint16_t>(base), 2u) &&
            (!visible || !range_overlaps(static_cast<uint16_t>(base), 2u, *visible, 1u))) {
          pc = static_cast<uint16_t>(base);
          break;
        }
      }
      if (needs_dense_router) {
        for (uint32_t base = util::align_up(first, 2u); base + 2u <= ordinary_sgpr_limit;
             base += 2u) {
          if (!unavailable_or_reserved(static_cast<uint16_t>(base), 2u) &&
              (!visible || !range_overlaps(static_cast<uint16_t>(base), 2u, *visible, 1u)) &&
              (!pc || !range_overlaps(static_cast<uint16_t>(base), 2u, *pc, 2u))) {
            call_return = static_cast<uint16_t>(base);
            break;
          }
        }
      } else {
        call_return = pc;
      }
      if (needs_dense_router) {
        for (uint32_t reg = first; reg < ordinary_sgpr_limit; ++reg) {
          if (!unavailable_or_reserved(static_cast<uint16_t>(reg), 1u) &&
              (!visible || reg != *visible) &&
              (!pc || !range_overlaps(static_cast<uint16_t>(reg), 1u, *pc, 2u)) &&
              (!call_return || !range_overlaps(static_cast<uint16_t>(reg), 1u, *call_return, 2u))) {
            dispatch_key = static_cast<uint16_t>(reg);
            break;
          }
        }
      } else {
        dispatch_key = visible;
      }
      for (uint32_t reg = 0; reg < ordinary_sgpr_limit; ++reg) {
        if (!unavailable_or_reserved(static_cast<uint16_t>(reg), 1u) &&
            (!visible || reg != *visible) &&
            (!pc || !range_overlaps(static_cast<uint16_t>(reg), 1u, *pc, 2u)) &&
            (!call_return || !range_overlaps(static_cast<uint16_t>(reg), 1u, *call_return, 2u)) &&
            (!dispatch_key || reg != *dispatch_key)) {
          scc = static_cast<uint16_t>(reg);
          break;
        }
      }
      if (pc && call_return && dispatch_key && scc) {
        options.moi_exec_save_sgpr = spill_base;
        options.automatic_moi_exec_save_sgprs = true;
        options.automatic_moi_inline_sgpr_spill = true;
        options.moi_inline_visible_evidence_sgpr = visible;
        options.moi_inline_indirect_pc_sgpr = pc;
        options.moi_inline_call_return_sgpr = call_return;
        options.moi_inline_dispatch_key_sgpr = dispatch_key;
        options.moi_inline_indirect_scc_sgpr = scc;
        result.resolved_moi_exec_save_sgpr = spill_base;
        result.moi_exec_save_sgprs_automatic = true;
        result.warnings.emplace_back(
            "ConSan MOI automatically assigned spill-backed Inline SGPRs s" +
            std::to_string(*spill_base) + ":s" + std::to_string(*spill_base + count - 1u) +
            ", dense call-return s" + std::to_string(*call_return) + ":s" +
            std::to_string(*call_return + 1u) + ", key s" + std::to_string(*dispatch_key));
        return true;
      }
    }
  }

  // A monolithic generated library can contain a few full-pressure kernels
  // without making every independent owner unusable.  Keep one global window
  // (and therefore the existing probe ABI), but choose the highest legal
  // window that at least one owner can preserve.  Resource-plan rebuilding
  // rejects every owner group whose compiler allocation reaches the window.
  // This is deliberately more conservative than site-local liveness: the
  // retained window is safe across entry relays and every guest instruction
  // in each admitted kernel.
  if (is_rdna4_family_arch(arch)) {
    const int32_t last_base = static_cast<int32_t>((ordinary_sgpr_limit - count) & ~1u);
    for (int32_t base = last_base; base >= 0; base -= 2) {
      const uint16_t candidate = static_cast<uint16_t>(base);
      if ((options.moi_owner_sgpr &&
           range_overlaps(candidate, count, *options.moi_owner_sgpr, 1u)) ||
          (options.moi_dispatch_id_sgpr &&
           range_overlaps(candidate, count, *options.moi_dispatch_id_sgpr,
                          moi_dispatch_state_sgpr_count(options))) ||
          (options.moi_dynamic_stack_spill &&
           range_overlaps(candidate, count, /*stack top=*/32u, /*stack registers=*/2u))) {
        continue;
      }
      const bool admits_owner =
          std::ranges::any_of(result.resource_plans, [&](const ConSanCandidateResourcePlan &plan) {
            return plan.source != ConSanRegisterAllocationSource::Unsupported &&
                   !plan.owner_descriptor_file_offsets.empty() &&
                   plan.max_referenced_sgpr_count <= candidate;
          });
      if (!admits_owner)
        continue;
      options.moi_exec_save_sgpr = candidate;
      options.automatic_moi_exec_save_sgprs = true;
      options.automatic_moi_partial_exec_save_sgprs = true;
      options.moi_exec_save_sgprs_persistent = true;
      result.resolved_moi_exec_save_sgpr = options.moi_exec_save_sgpr;
      result.moi_exec_save_sgprs_automatic = true;
      result.warnings.emplace_back(
          "ConSan MOI reserved a high EXEC-save SGPR window and excluded only "
          "incompatible full-pressure owner groups");
      (void)configure_automatic_moi_transient_sgpr_assignments(options, result, bytes, arch);
      return true;
    }
  }

  // A code object made entirely of full-pressure owners has no owner from
  // which to seed the ordinary high-window fallback. Seed component planning
  // with a placeholder; every admitted owner must replace it with either a
  // liveness-dead window or the spill-backed Record/Replay layout.
  if (is_rdna4_family_arch(arch) && (options.moi_engine == ConSanMoiEngine::RecordReplay ||
                                     options.moi_engine == ConSanMoiEngine::InlineShadow)) {
    options.moi_exec_save_sgpr = 0u;
    options.automatic_moi_exec_save_sgprs = true;
    options.automatic_moi_partial_exec_save_sgprs = true;
    if (configure_automatic_moi_transient_sgpr_assignments(options, result, bytes, arch)) {
      result.resolved_moi_exec_save_sgpr = options.moi_exec_save_sgpr;
      result.moi_exec_save_sgprs_automatic = true;
      result.warnings.emplace_back(
          "ConSan MOI admitted full-pressure owners with component-local scalar state");
      return true;
    }
    options.moi_exec_save_sgpr.reset();
    options.automatic_moi_exec_save_sgprs = false;
    options.automatic_moi_partial_exec_save_sgprs = false;
  }

  result.warnings.emplace_back(
      "ConSan MOI could not place a fresh automatic EXEC-save SGPR window");
  return false;
}

[[nodiscard]] bool configure_automatic_moi_owner_sgpr(ConSanOptions &options, ConSanResult &result,
                                                      rj_code_arch_t arch) {
  if (options.moi_owner_source != ConSanMoiOwnerSource::HwId || options.moi_owner_sgpr)
    return false;

  uint32_t first = 0;
  bool found_owned_scope = false;
  for (const ConSanCandidateResourcePlan &plan : result.resource_plans) {
    if (plan.owner_descriptor_file_offsets.empty())
      continue;
    found_owned_scope = true;
    first = std::max<uint32_t>(first, plan.max_referenced_sgpr_count);
  }
  if (!found_owned_scope) {
    result.warnings.emplace_back(
        "ConSan MOI could not derive an automatic hw_id owner SGPR without an owned candidate "
        "scope");
    return false;
  }

  for (uint32_t sgpr = first; sgpr < moi_ordinary_sgpr_limit(arch); ++sgpr) {
    if (options.moi_exec_save_sgpr &&
        range_overlaps(static_cast<uint16_t>(sgpr), 1u, *options.moi_exec_save_sgpr,
                       moi_exec_save_sgpr_count(options, arch))) {
      continue;
    }
    options.moi_owner_sgpr = static_cast<uint16_t>(sgpr);
    options.automatic_moi_owner_sgpr = true;
    result.resolved_moi_owner_sgpr = options.moi_owner_sgpr;
    result.moi_owner_sgpr_automatic = true;
    result.warnings.emplace_back("ConSan MOI automatically assigned hw_id owner temporary s" +
                                 std::to_string(sgpr));
    return true;
  }

  result.warnings.emplace_back("ConSan MOI could not place a fresh automatic hw_id owner SGPR");
  return false;
}

[[nodiscard]] bool configure_automatic_moi_dispatch_id_sgprs(ConSanOptions &options,
                                                             ConSanResult &result,
                                                             rj_code_arch_t arch) {
  const bool has_inline_dispatch_consumer =
      options.moi_engine == ConSanMoiEngine::InlineShadow &&
      !(arch == ROCJITSU_CODE_ARCH_GFX1250 && !options.moi_track_atomics) &&
      std::ranges::any_of(result.resource_plans, [](const ConSanCandidateResourcePlan &plan) {
        return !plan.owner_descriptor_file_offsets.empty() &&
               (plan.site_kind == ConSanResourceSiteKind::Access ||
                plan.site_kind == ConSanResourceSiteKind::Atomic);
      });
  const bool needs_dispatch_id =
      has_inline_dispatch_consumer ||
      (options.moi_engine == ConSanMoiEngine::RecordReplay &&
       arch == ROCJITSU_CODE_ARCH_GFX1250) ||
      (options.moi_engine == ConSanMoiEngine::Sampled && options.moi_runtime_sample_stride > 1u &&
       arch != ROCJITSU_CODE_ARCH_GFX1250);
  if (!needs_dispatch_id || !options.moi_report_buffer_address || options.moi_dispatch_id_sgpr) {
    return false;
  }

  // Sixteen initialized user SGPRs plus X/Y/Z and WORKGROUP_INFO is the
  // largest legal entry-input window after insertion. Keeping the persistent
  // pair above that window makes it safe for every per-kernel preload map.
  uint32_t first = 20u;
  bool found_owned_scope = false;
  for (const ConSanCandidateResourcePlan &plan : result.resource_plans) {
    if (plan.owner_descriptor_file_offsets.empty())
      continue;
    found_owned_scope = true;
    first = std::max<uint32_t>(first, plan.max_referenced_sgpr_count);
  }
  if (!found_owned_scope) {
    result.warnings.emplace_back(
        "ConSan MOI could not derive automatic dispatch-ID SGPRs without an owned candidate "
        "scope");
    return false;
  }

  const auto qualify_owner_local_assignments = [&](uint16_t dispatch_base) {
    for (ConSanMoiTransientSgprAssignment &assignment :
         options.moi_transient_sgpr_assignments) {
      uint32_t max_referenced_sgpr_count = 0u;
      for (const ConSanCandidateResourcePlan &plan : result.resource_plans) {
        if (std::ranges::find(plan.owner_descriptor_file_offsets,
                              assignment.descriptor_file_offset) ==
            plan.owner_descriptor_file_offsets.end()) {
          continue;
        }
        max_referenced_sgpr_count =
            std::max<uint32_t>(max_referenced_sgpr_count, plan.max_referenced_sgpr_count);
      }
      assignment.dispatch_id_sgpr =
          max_referenced_sgpr_count <= dispatch_base ? std::optional(dispatch_base) : std::nullopt;
    }
    result.resolved_moi_transient_sgpr_assignments = options.moi_transient_sgpr_assignments;
  };

  first = util::align_up(first, 2u);
  const uint16_t dispatch_state_count = moi_dispatch_state_sgpr_count(options);
  const uint16_t ordinary_sgpr_limit = moi_ordinary_sgpr_limit(arch);
  for (uint32_t base = first; base + dispatch_state_count <= ordinary_sgpr_limit; base += 2u) {
    if (options.moi_owner_sgpr && range_overlaps(static_cast<uint16_t>(base), dispatch_state_count,
                                                 *options.moi_owner_sgpr, 1u)) {
      continue;
    }
    if (options.moi_exec_save_sgpr &&
        range_overlaps(static_cast<uint16_t>(base), dispatch_state_count,
                       *options.moi_exec_save_sgpr, moi_exec_save_sgpr_count(options, arch))) {
      continue;
    }
    options.moi_dispatch_id_sgpr = static_cast<uint16_t>(base);
    options.automatic_moi_dispatch_id_sgprs = true;
    result.resolved_moi_dispatch_id_sgpr = options.moi_dispatch_id_sgpr;
    result.moi_dispatch_id_sgprs_automatic = true;
    qualify_owner_local_assignments(static_cast<uint16_t>(base));
    result.warnings.emplace_back("ConSan MOI automatically assigned persistent dispatch-ID "
                                 "state SGPRs s" +
                                 std::to_string(base) + ":s" +
                                 std::to_string(base + dispatch_state_count - 1u));
    return true;
  }

  // A large generated code object can contain one full-SGPR kernel alongside
  // many independent lower-pressure kernels.  Requiring one pair above the
  // maximum reference in the entire object would reject every site because
  // of that one owner.  gfx1250 instead reserves the highest legal pair and
  // lets resource-plan rebuilding reject only site groups whose owners reach
  // that pair.  The pair remains code-object-wide, so this does not change the
  // report ABI or any emitted probe sequence.
  if (arch == ROCJITSU_CODE_ARCH_GFX1250) {
    const int32_t last_base =
        static_cast<int32_t>((ordinary_sgpr_limit - dispatch_state_count) & ~1u);
    for (int32_t base = last_base; base >= 20; base -= 2) {
      if (options.moi_owner_sgpr &&
          range_overlaps(static_cast<uint16_t>(base), dispatch_state_count, *options.moi_owner_sgpr,
                         1u)) {
        continue;
      }
      if (options.moi_exec_save_sgpr &&
          range_overlaps(static_cast<uint16_t>(base), dispatch_state_count,
                         *options.moi_exec_save_sgpr, moi_exec_save_sgpr_count(options, arch))) {
        continue;
      }
      options.moi_dispatch_id_sgpr = static_cast<uint16_t>(base);
      options.automatic_moi_dispatch_id_sgprs = true;
      result.resolved_moi_dispatch_id_sgpr = options.moi_dispatch_id_sgpr;
      result.moi_dispatch_id_sgprs_automatic = true;
      qualify_owner_local_assignments(static_cast<uint16_t>(base));
      result.warnings.emplace_back(
          "ConSan MOI reserved a high dispatch-ID SGPR pair and excluded only "
          "incompatible full-pressure owner groups");
      return true;
    }
  }

  if (arch == ROCJITSU_CODE_ARCH_RDNA4) {
    result.warnings.emplace_back(
        "ConSan MOI is using a literal report dispatch ID because no persistent hardware "
        "dispatch-ID SGPR pair fits");
    return false;
  }
  result.outcome = ConSanTransformOutcome::Unsupported;
  result.warnings.emplace_back(
      "ConSan MOI could not place a persistent hardware dispatch-ID SGPR pair");
  return false;
}

[[nodiscard]] bool validate_moi_dispatch_id_sgprs(const ConSanOptions &options,
                                                  ConSanResult &result, rj_code_arch_t arch) {
  if (!options.moi_dispatch_id_sgpr)
    return true;
  const uint16_t base = *options.moi_dispatch_id_sgpr;
  const uint16_t dispatch_state_count = moi_dispatch_state_sgpr_count(options);
  const bool overlaps_owner = options.moi_owner_sgpr && range_overlaps(base, dispatch_state_count,
                                                                       *options.moi_owner_sgpr, 1u);
  const bool overlaps_exec = options.moi_exec_save_sgpr &&
                             range_overlaps(base, dispatch_state_count, *options.moi_exec_save_sgpr,
                                            moi_exec_save_sgpr_count(options, arch));
  if (base < 20u || base % 2u != 0u ||
      static_cast<uint32_t>(base) + dispatch_state_count > moi_ordinary_sgpr_limit(arch) ||
      overlaps_owner || overlaps_exec) {
    result.outcome = ConSanTransformOutcome::Unsupported;
    result.warnings.emplace_back(
        "ConSan MOI persistent dispatch-ID SGPR pair is unaligned, overlaps entry/scratch "
        "state, or exceeds the architectural SGPR file");
    return false;
  }
  return true;
}

[[nodiscard]] bool validate_moi_ordinary_scalar_state(const ConSanOptions &options,
                                                      ConSanResult &result, rj_code_arch_t arch) {
  const uint16_t limit = moi_ordinary_sgpr_limit(arch);
  const bool exec_invalid =
      options.moi_exec_save_sgpr &&
      static_cast<uint32_t>(*options.moi_exec_save_sgpr) + moi_exec_save_sgpr_count(options, arch) >
          limit;
  const bool owner_invalid = options.moi_owner_sgpr && *options.moi_owner_sgpr >= limit;
  const bool persistent_invalid =
      (options.moi_persistent_owner_sgpr && *options.moi_persistent_owner_sgpr >= limit) ||
      (options.moi_persistent_epoch_sgpr && *options.moi_persistent_epoch_sgpr >= limit) ||
      (options.moi_persistent_workgroup_key_sgpr &&
       *options.moi_persistent_workgroup_key_sgpr >= limit);
  const bool inline_external_invalid =
      (options.moi_inline_indirect_pc_sgpr &&
       static_cast<uint32_t>(*options.moi_inline_indirect_pc_sgpr) + 2u > limit) ||
      (options.moi_inline_call_return_sgpr &&
       static_cast<uint32_t>(*options.moi_inline_call_return_sgpr) + 2u > limit) ||
      (options.moi_inline_dispatch_key_sgpr && *options.moi_inline_dispatch_key_sgpr >= limit) ||
      (options.moi_inline_indirect_scc_sgpr && *options.moi_inline_indirect_scc_sgpr >= limit) ||
      (options.moi_inline_visible_evidence_sgpr &&
       *options.moi_inline_visible_evidence_sgpr >= limit) ||
      (options.moi_record_replay_dispatch_key_sgpr &&
       *options.moi_record_replay_dispatch_key_sgpr >= limit) ||
      (options.moi_record_replay_call_return_sgpr &&
       static_cast<uint32_t>(*options.moi_record_replay_call_return_sgpr) + 2u > limit);
  if (!exec_invalid && !owner_invalid && !persistent_invalid && !inline_external_invalid)
    return true;
  result.outcome = ConSanTransformOutcome::Unsupported;
  result.warnings.emplace_back(
      "ConSan MOI scalar instrumentation state aliases an architectural special SGPR");
  return false;
}

[[nodiscard]] bool configure_automatic_moi_persistent_vgprs(ConSanOptions &options,
                                                            ConSanResult &result,
                                                            std::span<const uint8_t> bytes,
                                                            rj_code_arch_t arch) {
  if (options.moi_owner_vgpr || options.moi_epoch_vgpr || options.moi_workgroup_key_vgpr ||
      options.moi_sample_sequence_vgpr)
    return false;
  const bool has_supported_sampled_sync =
      options.moi_engine == ConSanMoiEngine::Sampled &&
      std::ranges::any_of(result.resource_plans,
                          [](const ConSanCandidateResourcePlan &plan) {
                            return plan.site_kind == ConSanResourceSiteKind::Barrier ||
                                   plan.site_kind == ConSanResourceSiteKind::Atomic;
                          }) &&
      std::ranges::any_of(result.site_dispositions, [&](const ConSanSiteDispositionRecord &site) {
        if ((site.site_kind != ConSanResourceSiteKind::Barrier &&
             site.site_kind != ConSanResourceSiteKind::Atomic) ||
            site.disposition != ConSanSiteDisposition::Supported)
          return false;
        return std::ranges::any_of(
            result.resource_plans, [&](const ConSanCandidateResourcePlan &plan) {
              return plan.site_kind == site.site_kind && plan.text_offset == site.text_offset &&
                     plan.source != ConSanRegisterAllocationSource::Unsupported &&
                     !plan.owner_descriptor_file_offsets.empty();
            });
      });
  // Access-only Sampled objects select waves with the persistent dispatch
  // counter. A per-workitem sequence is only needed when causal
  // synchronization state is actually represented; owner/epoch still live
  // across the guest and are handled below.
  const bool needs_sample_sequence = options.moi_engine == ConSanMoiEngine::Sampled &&
                                     options.moi_runtime_sample_stride > 1u &&
                                     has_supported_sampled_sync;
  const bool needs_workgroup_key =
      options.moi_engine == ConSanMoiEngine::InlineShadow &&
      std::ranges::any_of(result.resource_plans, [&](const ConSanCandidateResourcePlan &plan) {
        // Inline sync dispositions are appended only after persistent-state
        // placement. Its operational resource plans have already filtered
        // out semantically unsupported atomics, so they are the authoritative
        // pre-placement signal here.
        if (plan.source == ConSanRegisterAllocationSource::Unsupported ||
            plan.owner_descriptor_file_offsets.empty())
          return false;
        if (plan.site_kind == ConSanResourceSiteKind::Atomic)
          return true;
        // CDNA4 exposes workgroup IDs through ordinary entry SGPRs that guest
        // code may reuse before any DS or group-FLAT access. Preserve the key
        // at kernel entry for every Inline access instead of limiting the
        // lifetime fix to shared helper/group-FLAT candidates.
        return result.arch_name == "cdna4" && plan.site_kind == ConSanResourceSiteKind::Access;
      });
  // Sampled access probes execute after arbitrary guest instructions. Their
  // owner must therefore come from entry-captured persistent state even when
  // the code object has no semantically admitted synchronization site; v0 is
  // only the ABI workitem ID at kernel entry and may be clobbered later.
  const bool needs_persistent_state =
      options.moi_init_owner_epoch ||
      (options.moi_track_atomics && options.moi_engine != ConSanMoiEngine::RecordReplay) ||
      (options.moi_engine == ConSanMoiEngine::Sampled &&
       (options.moi_track_barriers || options.moi_runtime_sample_stride > 1u)) ||
      options.moi_engine == ConSanMoiEngine::InlineShadow;
  if (!needs_persistent_state)
    return false;
  if (!options.moi_report_buffer_address)
    return false;

  uint32_t persistent_base = 0;
  bool found_owned_scope = false;
  for (const ConSanCandidateResourcePlan &plan : result.resource_plans) {
    if (plan.owner_descriptor_file_offsets.empty())
      continue;
    found_owned_scope = true;
    // Persistent state crosses arbitrary guest code, so it must sit above
    // every statically referenced ordinary VGPR. Indirect v_movrel access has
    // no complete static bound and conservatively reserves the ordinary guest
    // allocation instead. On CDNA4, ACCUM_OFFSET bounds this search below the
    // physical AccVGPR bank even when the descriptor's total VGPR allocation
    // extends through that bank.
    const uint32_t safe_base =
        plan.has_indirect_vgpr_access
            ? std::min<uint32_t>(plan.current_vgpr_count, plan.ordinary_vgpr_limit)
            : plan.max_referenced_vgpr_count;
    persistent_base = std::max(persistent_base, safe_base);
  }

  if (!found_owned_scope) {
    result.warnings.emplace_back(
        "ConSan MOI could not derive automatic persistent VGPRs without an owned candidate "
        "scope");
    return false;
  }
  // The gfx1250 Inline local-shadow prologue uses three additional tail VGPRs
  // only while clearing with a 128-bit LDS store. They must lie above every
  // guest reference even though owner/epoch remain the only persistent values.
  const uint32_t named_persistent_count = 2u + static_cast<uint32_t>(needs_sample_sequence) +
                                          static_cast<uint32_t>(needs_workgroup_key);
  const uint32_t persistent_count =
      options.moi_engine == ConSanMoiEngine::InlineShadow && arch == ROCJITSU_CODE_ARCH_GFX1250
          ? std::max<uint32_t>(named_persistent_count, 5u)
          : named_persistent_count;
  const uint32_t persistent_end = persistent_base + persistent_count;
  const bool crosses_cdna4_accvgpr_boundary =
      result.arch_name == "cdna4" &&
      std::ranges::any_of(result.resource_plans, [&](const ConSanCandidateResourcePlan &plan) {
        return !plan.owner_descriptor_file_offsets.empty() &&
               plan.source != ConSanRegisterAllocationSource::Unsupported &&
               plan.ordinary_vgpr_limit != kMaxVgprs && persistent_end > plan.ordinary_vgpr_limit;
      });
  if (crosses_cdna4_accvgpr_boundary) {
    // ACCUM_OFFSET defines the compiler's ordinary/accumulator register split.
    // Moving that split after code generation changes which physical registers
    // existing MFMA and accumulator operands address and has corrupted native
    // gfx950 workloads. Until persistent tuples are selected per owning kernel,
    // reject a code object whose one global tuple would cross any such boundary.
    std::vector<std::vector<uint64_t>> components;
    for (const ConSanCandidateResourcePlan &plan : result.resource_plans) {
      if (plan.source == ConSanRegisterAllocationSource::Unsupported)
        continue;
      for (uint64_t descriptor_offset : plan.owner_descriptor_file_offsets) {
        if (std::ranges::none_of(components, [&](const auto &component) {
              return std::ranges::find(component, descriptor_offset) != component.end();
            })) {
          components.push_back({descriptor_offset});
        }
      }
      std::optional<size_t> merged_index;
      for (uint64_t descriptor_offset : plan.owner_descriptor_file_offsets) {
        const auto component = std::ranges::find_if(components, [&](const auto &candidate) {
          return std::ranges::find(candidate, descriptor_offset) != candidate.end();
        });
        if (component == components.end())
          continue;
        const size_t index = static_cast<size_t>(component - components.begin());
        if (!merged_index) {
          merged_index = index;
        } else if (*merged_index != index) {
          auto &destination = components[*merged_index];
          destination.insert(destination.end(), components[index].begin(), components[index].end());
          components.erase(components.begin() + static_cast<ptrdiff_t>(index));
          if (index < *merged_index)
            --*merged_index;
        }
      }
    }

    for (const std::vector<uint64_t> &component : components) {
      uint32_t component_base = 0u;
      uint32_t component_limit = kMaxVgprs;
      for (const ConSanCandidateResourcePlan &plan : result.resource_plans) {
        if (plan.source == ConSanRegisterAllocationSource::Unsupported ||
            std::ranges::none_of(plan.owner_descriptor_file_offsets, [&](uint64_t offset) {
              return std::ranges::find(component, offset) != component.end();
            })) {
          continue;
        }
        const uint32_t safe_base =
            plan.has_indirect_vgpr_access
                ? std::min<uint32_t>(plan.current_vgpr_count, plan.ordinary_vgpr_limit)
                : plan.max_referenced_vgpr_count;
        component_base = std::max(component_base, safe_base);
        component_limit = std::min<uint32_t>(component_limit, plan.ordinary_vgpr_limit);
      }
      if (component_base + persistent_count > component_limit) {
        const bool has_dynamic_stack_owner =
            std::ranges::any_of(component, [&](uint64_t descriptor_offset) {
              const auto kernel = std::ranges::find(result.kernels, descriptor_offset,
                                                    &ConSanKernelInfo::descriptor_file_offset);
              return kernel != result.kernels.end() && kernel->uses_dynamic_stack.value_or(false);
            });
        if (options.moi_engine == ConSanMoiEngine::InlineShadow && has_dynamic_stack_owner) {
          // Owner, epoch, and the optional workgroup key are uniform within a
          // wave. A fresh SGPR tail preserves them across arbitrary guest code
          // without changing ACCUM_OFFSET or consuming address-based private
          // state, so it composes with compiler-managed dynamic stacks.
          const uint16_t scalar_count = static_cast<uint16_t>(2u + needs_workgroup_key);
          uint32_t first = 0u;
          for (const ConSanCandidateResourcePlan &scalar_plan : result.resource_plans) {
            if (scalar_plan.source == ConSanRegisterAllocationSource::Unsupported ||
                scalar_plan.owner_descriptor_file_offsets.empty())
              continue;
            first = std::max<uint32_t>(first, scalar_plan.max_referenced_sgpr_count);
          }
          const uint16_t scalar_limit = moi_ordinary_sgpr_limit(ROCJITSU_CODE_ARCH_CDNA4);
          const auto unavailable = [&](uint16_t base) {
            if ((options.moi_exec_save_sgpr &&
                 range_overlaps(base, scalar_count, *options.moi_exec_save_sgpr,
                                moi_exec_save_sgpr_count(options, ROCJITSU_CODE_ARCH_CDNA4))) ||
                (options.moi_dispatch_id_sgpr &&
                 range_overlaps(base, scalar_count, *options.moi_dispatch_id_sgpr,
                                moi_dispatch_state_sgpr_count(options))) ||
                (options.moi_owner_sgpr &&
                 range_overlaps(base, scalar_count, *options.moi_owner_sgpr, 1u)) ||
                range_overlaps(base, scalar_count, 32u, 2u))
              return true;
            return std::ranges::any_of(
                result.resource_plans, [&](const ConSanCandidateResourcePlan &scalar_plan) {
                  if (scalar_plan.owner_descriptor_file_offsets.empty() ||
                      scalar_plan.current_sgpr_count < 2u)
                    return false;
                  return range_overlaps(base, scalar_count,
                                        static_cast<uint16_t>(scalar_plan.current_sgpr_count - 2u),
                                        2u);
                });
          };
          for (uint32_t base = first; base + scalar_count <= scalar_limit; ++base) {
            if (unavailable(static_cast<uint16_t>(base)))
              continue;
            MoiResourcePlanningState state(bytes, arch, options, result);
            if (!state.valid) {
              result.outcome = ConSanTransformOutcome::Unsupported;
              result.warnings.emplace_back(
                  "ConSan MOI could not analyze entry-local VGPR scratch for scalar persistent "
                  "state");
              return false;
            }
            std::vector<ConSanMoiPrologueScratchVgprAssignment> prologue_scratch;
            for (uint64_t descriptor_offset : component) {
              const auto context = std::ranges::find_if(
                  state.contexts, [&](const MoiKernelResourceContext &candidate) {
                    return candidate.kernel != nullptr &&
                           candidate.kernel->descriptor_file_offset == descriptor_offset;
                  });
              if (context == state.contexts.end() || context->kernel == nullptr) {
                result.outcome = ConSanTransformOutcome::Unsupported;
                result.warnings.emplace_back(
                    "ConSan MOI could not resolve a scalar-state owner entry CFG");
                return false;
              }
              BasicBlock *entry_block =
                  block_for_offset(state.block_index, context->kernel->entry_text_offset);
              const Instruction *entry =
                  instruction_at(entry_block, context->kernel->entry_text_offset);
              if (entry_block == nullptr || entry == nullptr) {
                result.outcome = ConSanTransformOutcome::Unsupported;
                result.warnings.emplace_back(
                    "ConSan MOI could not decode a scalar-state owner entry instruction");
                return false;
              }
              RegisterSet unavailable_vgprs = context->liveness->live_before(*entry);
              const InstDefUse entry_def_use(*entry);
              unavailable_vgprs |= entry_def_use.defs | entry_def_use.uses;
              std::optional<uint16_t> scratch;
              for (uint16_t candidate = 0; candidate + 2u <= context->ordinary_vgpr_limit;
                   ++candidate) {
                if (!unavailable_vgprs.contains({RegClass::VGPR, candidate, 1u}) &&
                    !unavailable_vgprs.contains(
                        {RegClass::VGPR, static_cast<uint16_t>(candidate + 1u), 1u})) {
                  scratch = candidate;
                  break;
                }
              }
              if (!scratch) {
                result.outcome = ConSanTransformOutcome::Unsupported;
                result.warnings.emplace_back(
                    "ConSan MOI could not place entry-local VGPR scratch for scalar persistent "
                    "state");
                return false;
              }
              prologue_scratch.push_back({descriptor_offset, *scratch});
            }
            options.moi_persistent_owner_sgpr = static_cast<uint16_t>(base);
            options.moi_persistent_epoch_sgpr = static_cast<uint16_t>(base + 1u);
            if (needs_workgroup_key)
              options.moi_persistent_workgroup_key_sgpr = static_cast<uint16_t>(base + 2u);
            options.moi_init_owner_epoch = true;
            options.automatic_moi_persistent_sgprs = true;
            options.moi_prologue_scratch_vgpr_assignments = std::move(prologue_scratch);
            result.resolved_moi_persistent_owner_sgpr = options.moi_persistent_owner_sgpr;
            result.resolved_moi_persistent_epoch_sgpr = options.moi_persistent_epoch_sgpr;
            result.resolved_moi_persistent_workgroup_key_sgpr =
                options.moi_persistent_workgroup_key_sgpr;
            result.moi_persistent_sgprs_automatic = true;
            result.resolved_moi_prologue_scratch_vgpr_assignments =
                options.moi_prologue_scratch_vgpr_assignments;
            result.warnings.emplace_back("ConSan MOI assigned persistent scalar owner/epoch" +
                                         std::string(needs_workgroup_key ? "/workgroup-key" : "") +
                                         " state s" + std::to_string(base) + ":s" +
                                         std::to_string(base + scalar_count - 1u) +
                                         " below a full CDNA4 ordinary VGPR bank");
            return true;
          }
          result.outcome = ConSanTransformOutcome::Unsupported;
          result.warnings.emplace_back(
              "ConSan MOI cannot place persistent scalar state for a dynamic-stack CDNA4 "
              "owner with a full ordinary VGPR bank");
          return false;
        }
        if (options.moi_engine == ConSanMoiEngine::InlineShadow && !has_dynamic_stack_owner) {
          options.moi_persistent_vgpr_assignments.clear();
          options.moi_init_owner_epoch = true;
          options.automatic_moi_private_epoch = true;
          result.moi_private_epoch_automatic = true;
          result.warnings.emplace_back(
              "ConSan MOI selected entry-snapshotted private owner/epoch state below a full "
              "CDNA4 ordinary VGPR bank");
          return true;
        }
        result.outcome = ConSanTransformOutcome::Unsupported;
        result.warnings.emplace_back(
            "ConSan MOI cannot place persistent VGPR state below a connected CDNA4 AccVGPR "
            "boundary");
        return false;
      }
      for (uint64_t descriptor_offset : component) {
        ConSanMoiPersistentVgprAssignment assignment;
        assignment.descriptor_file_offset = descriptor_offset;
        assignment.owner_vgpr = static_cast<uint16_t>(component_base);
        assignment.epoch_vgpr = static_cast<uint16_t>(component_base + 1u);
        uint16_t next = static_cast<uint16_t>(component_base + 2u);
        if (needs_sample_sequence)
          assignment.sample_sequence_vgpr = next++;
        if (needs_workgroup_key)
          assignment.workgroup_key_vgpr = next++;
        options.moi_persistent_vgpr_assignments.push_back(assignment);
      }
    }
    options.moi_init_owner_epoch = true;
    options.automatic_moi_persistent_vgprs = true;
    result.resolved_moi_persistent_vgpr_assignments = options.moi_persistent_vgpr_assignments;
    result.moi_persistent_vgprs_automatic = true;
    result.warnings.emplace_back(
        "ConSan MOI assigned per-owner persistent VGPR tuples below CDNA4 AccVGPR boundaries");
    return true;
  }
  const bool would_grow_descriptor =
      std::ranges::any_of(result.resource_plans, [&](const ConSanCandidateResourcePlan &plan) {
        return !plan.owner_descriptor_file_offsets.empty() &&
               plan.source != ConSanRegisterAllocationSource::Unsupported &&
               persistent_end > plan.current_vgpr_count;
      });
  const bool has_dynamic_stack_owner =
      std::ranges::any_of(result.kernels, [&](const ConSanKernelInfo &kernel) {
        if (!kernel.uses_dynamic_stack.value_or(false))
          return false;
        return std::ranges::any_of(
            result.resource_plans, [&](const ConSanCandidateResourcePlan &plan) {
              return plan.source != ConSanRegisterAllocationSource::Unsupported &&
                     std::ranges::find(plan.owner_descriptor_file_offsets,
                                       kernel.descriptor_file_offset) !=
                         plan.owner_descriptor_file_offsets.end();
            });
      });
  const bool has_dynamic_lds_owner =
      options.moi_engine == ConSanMoiEngine::InlineShadow &&
      std::ranges::any_of(result.kernels, [&](const ConSanKernelInfo &kernel) {
        if (!kernel.has_dynamic_lds)
          return false;
        return std::ranges::any_of(
            result.resource_plans, [&](const ConSanCandidateResourcePlan &plan) {
              return plan.source != ConSanRegisterAllocationSource::Unsupported &&
                     std::ranges::find(plan.owner_descriptor_file_offsets,
                                       kernel.descriptor_file_offset) !=
                         plan.owner_descriptor_file_offsets.end();
            });
      });
  const size_t record_replay_barrier_sites =
      options.moi_engine == ConSanMoiEngine::RecordReplay && options.moi_track_barriers
          ? std::ranges::count_if(result.resource_plans,
                                  [](const ConSanCandidateResourcePlan &plan) {
                                    return plan.site_kind == ConSanResourceSiteKind::Barrier &&
                                           plan.source !=
                                               ConSanRegisterAllocationSource::Unsupported &&
                                           !plan.owner_descriptor_file_offsets.empty();
                                  })
          : 0u;
  const size_t record_replay_access_sites =
      options.moi_engine == ConSanMoiEngine::RecordReplay
          ? std::ranges::count_if(result.resource_plans,
                                  [](const ConSanCandidateResourcePlan &plan) {
                                    return plan.site_kind == ConSanResourceSiteKind::Access &&
                                           plan.source !=
                                               ConSanRegisterAllocationSource::Unsupported &&
                                           !plan.owner_descriptor_file_offsets.empty();
                                  })
          : 0u;
  // This is an automatic operating-point choice, not a user-facing limit.
  // Small barrier inventories benefit from compact persistent-epoch barriers;
  // larger barrier-dense Record/Replay objects benefit more from private-epoch
  // access coalescing. Qwen's 28 barrier sites and the access-heavy TP
  // workloads avoid millions of dynamic barrier records.
  constexpr size_t kCompactRecordReplayBarrierSiteLimit = 32u;
  const bool prefer_compact_record_replay_barriers =
      record_replay_barrier_sites > 0u &&
      (record_replay_barrier_sites <= kCompactRecordReplayBarrierSiteLimit ||
       record_replay_access_sites >= 2u * record_replay_barrier_sites);
  // Private epoch storage is address-free scratch and therefore cannot be
  // composed with a guest dynamic stack. If persistent state fits in the
  // architectural VGPR file, grow those owner descriptors instead. The
  // rebuilt resource plans reserve the selected tail VGPRs before allocating
  // per-site scratch, so this growth cannot overlap instrumentation scratch.
  // A bounded Record/Replay barrier inventory is another exception: without
  // persistent epoch state it requires one dynamic record per participating
  // workitem and barrier execution. That can turn a small static inventory
  // into millions of records. Prefer bounded descriptor growth so those
  // barriers use the compact inline-epoch path instead.
  // A synchronization-aware Sampled probe must preserve one owner identity
  // from kernel entry through both access and sync sites. Its private-epoch
  // fallback otherwise derives the access owner from guest-reusable v0 and
  // can publish the reserved owner zero on real compiler output. Access-only
  // Sampled objects retain the cheaper private-state choice.
  const bool sampled_requires_persistent_owner =
      options.moi_engine == ConSanMoiEngine::Sampled && has_supported_sampled_sync;
  // InlineShadow reads owner state on every instrumented LDS access. A
  // scratch-backed owner made a complete multi-kernel CLIP invocation spend
  // more than a minute repeatedly loading that state, whereas two persistent
  // tail VGPRs complete the same clean row in about 1.5 seconds. Prefer the
  // descriptor-backed representation whenever it fits; private state remains
  // the capacity fallback for near-full VGPR files. Dynamic-LDS kernels are an
  // exception: their external exact-shadow path exercises arbitrary guest
  // VGPR-bank transitions, so entry-snapshotted private state avoids treating
  // a remapped tail VGPR as a different owner while retaining the same exact
  // shadow semantics.
  const bool prefer_private_epoch =
      !has_dynamic_stack_owner && !prefer_compact_record_replay_barriers && !needs_workgroup_key &&
      !sampled_requires_persistent_owner &&
      (has_dynamic_lds_owner ||
       (would_grow_descriptor && options.moi_engine != ConSanMoiEngine::InlineShadow));
  const bool rdna4_record_replay_needs_scalar_epoch =
      is_rdna4_family_arch(arch) && options.moi_engine == ConSanMoiEngine::RecordReplay &&
      !options.force_private_epoch && (persistent_end > kMaxVgprs || prefer_private_epoch);
  if (rdna4_record_replay_needs_scalar_epoch) {
    constexpr uint16_t scalar_count = 2u;
    uint32_t first = 0u;
    std::vector<uint64_t> owner_descriptors;
    for (const ConSanCandidateResourcePlan &plan : result.resource_plans) {
      if (plan.source == ConSanRegisterAllocationSource::Unsupported ||
          plan.owner_descriptor_file_offsets.empty()) {
        continue;
      }
      first = std::max<uint32_t>(first, plan.max_referenced_sgpr_count);
      for (uint64_t descriptor_offset : plan.owner_descriptor_file_offsets) {
        if (std::ranges::find(owner_descriptors, descriptor_offset) == owner_descriptors.end())
          owner_descriptors.push_back(descriptor_offset);
      }
    }
    const auto overlaps_reserved_state = [&](uint16_t base) {
      return (options.moi_exec_save_sgpr &&
              range_overlaps(base, scalar_count, *options.moi_exec_save_sgpr,
                             moi_exec_save_sgpr_count(options, arch))) ||
             (options.moi_dispatch_id_sgpr &&
              range_overlaps(base, scalar_count, *options.moi_dispatch_id_sgpr,
                             moi_dispatch_state_sgpr_count(options))) ||
             (options.moi_owner_sgpr &&
              range_overlaps(base, scalar_count, *options.moi_owner_sgpr, 1u));
    };
    for (uint32_t base = first; base + scalar_count <= moi_ordinary_sgpr_limit(arch); ++base) {
      if (overlaps_reserved_state(static_cast<uint16_t>(base)))
        continue;
      std::vector<ConSanMoiPrologueScratchVgprAssignment> prologue_scratch;
      // Record/Replay's scalar prologue initializes both persistent values
      // directly in SGPRs. The VGPR pair is part of the shared prologue API
      // but is not read or written in this mode, so it does not require an
      // entry-liveness allocation even for a full VGPR bank.
      for (uint64_t descriptor_offset : owner_descriptors)
        prologue_scratch.push_back({descriptor_offset, 0u});
      options.moi_persistent_owner_sgpr = static_cast<uint16_t>(base);
      options.moi_persistent_epoch_sgpr = static_cast<uint16_t>(base + 1u);
      options.moi_init_owner_epoch = true;
      options.automatic_moi_persistent_sgprs = true;
      options.moi_prologue_scratch_vgpr_assignments = std::move(prologue_scratch);
      result.resolved_moi_persistent_owner_sgpr = options.moi_persistent_owner_sgpr;
      result.resolved_moi_persistent_epoch_sgpr = options.moi_persistent_epoch_sgpr;
      result.moi_persistent_sgprs_automatic = true;
      result.resolved_moi_prologue_scratch_vgpr_assignments =
          options.moi_prologue_scratch_vgpr_assignments;
      result.warnings.emplace_back(
          "ConSan MOI assigned persistent scalar owner/epoch state for RDNA4 "
          "Record/Replay without growing the VGPR allocation");
      return true;
    }
    result.warnings.emplace_back(
        "ConSan MOI could not place persistent scalar owner/epoch state for RDNA4 "
        "Record/Replay");
  }
  if (options.force_private_epoch || persistent_end > kMaxVgprs || prefer_private_epoch) {
    options.moi_init_owner_epoch = true;
    options.automatic_moi_private_epoch = true;
    result.moi_private_epoch_automatic = true;
    result.warnings.emplace_back(
        std::string("ConSan MOI automatically selected derived-owner/private-"
                    "epoch persistent state") +
        (would_grow_descriptor ? " to preserve the guest VGPR allocation" : ""));
    return true;
  }

  options.moi_owner_vgpr = static_cast<uint16_t>(persistent_base);
  options.moi_epoch_vgpr = static_cast<uint16_t>(persistent_base + 1u);
  uint16_t next_persistent_vgpr = static_cast<uint16_t>(persistent_base + 2u);
  if (needs_sample_sequence)
    options.moi_sample_sequence_vgpr = next_persistent_vgpr++;
  if (needs_workgroup_key)
    options.moi_workgroup_key_vgpr = next_persistent_vgpr++;
  options.moi_init_owner_epoch = true;
  options.automatic_moi_persistent_vgprs = true;
  result.resolved_moi_owner_vgpr = options.moi_owner_vgpr;
  result.resolved_moi_epoch_vgpr = options.moi_epoch_vgpr;
  result.resolved_moi_workgroup_key_vgpr = options.moi_workgroup_key_vgpr;
  result.resolved_moi_sample_sequence_vgpr = options.moi_sample_sequence_vgpr;
  result.moi_persistent_vgprs_automatic = true;
  result.warnings.emplace_back(
      "ConSan MOI automatically assigned persistent owner/epoch" +
      std::string(needs_sample_sequence ? "/sample-sequence" : "") +
      std::string(needs_workgroup_key ? "/workgroup-key" : "") + " VGPRs v" +
      std::to_string(*options.moi_owner_vgpr) + ":v" + std::to_string(next_persistent_vgpr - 1u) +
      (would_grow_descriptor && has_dynamic_stack_owner
           ? " with descriptor growth because private epoch storage is incompatible with a "
             "dynamic-stack owner"
           : ""));
  return true;
}

[[nodiscard]] bool automatic_moi_persistent_state(const ConSanOptions &options) {
  return options.automatic_moi_persistent_vgprs || options.automatic_moi_persistent_sgprs ||
         options.automatic_moi_private_epoch;
}

using MoiDescriptorVgprRequirements = std::unordered_map<uint64_t, uint16_t>;
using MoiDescriptorSgprRequirements = std::unordered_map<uint64_t, uint16_t>;
using MoiDescriptorPrivateRequirements = std::unordered_map<uint64_t, uint32_t>;
using MoiDescriptorLdsRequirements = std::unordered_map<uint64_t, uint32_t>;
using MoiSpillManagers = std::unordered_map<uint64_t, SpillManager>;

void note_descriptor_requirements(MoiDescriptorVgprRequirements &requirements,
                                  const ResolvedMoiScratchPlan &plan) {
  for (uint64_t descriptor_offset : plan.owner_descriptor_file_offsets) {
    auto [it, inserted] = requirements.emplace(descriptor_offset, plan.required_vgpr_count);
    if (!inserted)
      it->second = std::max(it->second, plan.required_vgpr_count);
  }
}

void note_moi_sgpr_requirements(MoiDescriptorSgprRequirements &requirements,
                                const ResolvedMoiScratchPlan &plan, const ConSanOptions &options,
                                rj_code_arch_t arch) {
  uint16_t required_count = 0;
  if (options.moi_exec_save_sgpr) {
    const uint16_t count = moi_exec_save_sgpr_count(options, arch);
    required_count = static_cast<uint16_t>(*options.moi_exec_save_sgpr + count);
  }
  if (options.moi_owner_source == ConSanMoiOwnerSource::HwId && options.moi_owner_sgpr) {
    required_count =
        std::max<uint16_t>(required_count, static_cast<uint16_t>(*options.moi_owner_sgpr + 1u));
  }
  if (options.moi_dispatch_id_sgpr) {
    required_count = std::max<uint16_t>(
        required_count, static_cast<uint16_t>(*options.moi_dispatch_id_sgpr +
                                              moi_dispatch_state_sgpr_count(options)));
  }
  if (options.moi_persistent_owner_sgpr)
    required_count = std::max<uint16_t>(
        required_count, static_cast<uint16_t>(*options.moi_persistent_owner_sgpr + 1u));
  if (options.moi_persistent_epoch_sgpr)
    required_count = std::max<uint16_t>(
        required_count, static_cast<uint16_t>(*options.moi_persistent_epoch_sgpr + 1u));
  if (options.moi_persistent_workgroup_key_sgpr)
    required_count = std::max<uint16_t>(
        required_count, static_cast<uint16_t>(*options.moi_persistent_workgroup_key_sgpr + 1u));
  if (options.moi_inline_indirect_pc_sgpr)
    required_count = std::max<uint16_t>(
        required_count, static_cast<uint16_t>(*options.moi_inline_indirect_pc_sgpr + 2u));
  if (options.moi_inline_call_return_sgpr)
    required_count = std::max<uint16_t>(
        required_count, static_cast<uint16_t>(*options.moi_inline_call_return_sgpr + 2u));
  if (options.moi_inline_dispatch_key_sgpr)
    required_count = std::max<uint16_t>(
        required_count, static_cast<uint16_t>(*options.moi_inline_dispatch_key_sgpr + 1u));
  if (options.moi_inline_indirect_scc_sgpr)
    required_count = std::max<uint16_t>(
        required_count, static_cast<uint16_t>(*options.moi_inline_indirect_scc_sgpr + 1u));
  if (options.moi_inline_visible_evidence_sgpr)
    required_count = std::max<uint16_t>(
        required_count, static_cast<uint16_t>(*options.moi_inline_visible_evidence_sgpr + 1u));
  if (options.moi_record_replay_dispatch_key_sgpr)
    required_count = std::max<uint16_t>(
        required_count, static_cast<uint16_t>(*options.moi_record_replay_dispatch_key_sgpr + 1u));
  if (options.moi_record_replay_call_return_sgpr)
    required_count = std::max<uint16_t>(
        required_count, static_cast<uint16_t>(*options.moi_record_replay_call_return_sgpr + 2u));
  if (required_count == 0)
    return;
  for (uint64_t descriptor_offset : plan.owner_descriptor_file_offsets) {
    auto [it, inserted] = requirements.emplace(descriptor_offset, required_count);
    if (!inserted)
      it->second = std::max(it->second, required_count);
  }
}

void note_spill_descriptor_requirements(MoiDescriptorPrivateRequirements &requirements,
                                        const ResolvedMoiScratchPlan &plan,
                                        const VgprSpillSequence &spill) {
  for (uint64_t descriptor_offset : plan.owner_descriptor_file_offsets) {
    auto [it, inserted] = requirements.emplace(descriptor_offset, spill.total_private_bytes);
    if (!inserted)
      it->second = std::max(it->second, spill.total_private_bytes);
  }
}

void note_moi_lds_requirements(MoiDescriptorLdsRequirements &requirements,
                               const ResolvedMoiScratchPlan &plan,
                               const ConSanMoiWorkgroupShadowLayout &layout) {
  for (uint64_t descriptor_offset : plan.owner_descriptor_file_offsets) {
    auto [it, inserted] =
        requirements.emplace(descriptor_offset, layout.required_group_segment_size);
    if (!inserted)
      it->second = std::max(it->second, layout.required_group_segment_size);
  }
}

[[nodiscard]] const ConSanKernelInfo *kernel_for_descriptor(const ConSanResult &result,
                                                            uint64_t descriptor_offset) {
  const auto it = std::ranges::find_if(result.kernels, [descriptor_offset](const auto &kernel) {
    return kernel.descriptor_file_offset == descriptor_offset;
  });
  return it == result.kernels.end() ? nullptr : &*it;
}
