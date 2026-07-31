struct MoiLocalNopIsland {
  uint64_t text_offset = 0;
  bool claimed = false;
};

[[nodiscard]] constexpr bool moi_supports_dynamic_stack_spill(rj_code_arch_t arch,
                                                              ConSanMoiEngine engine) {
  if (engine == ConSanMoiEngine::InlineShadow)
    return true;
  return (is_rdna4_family_arch(arch) || instrumentation::is_cdna_family_arch(arch)) &&
         (engine == ConSanMoiEngine::RecordReplay || engine == ConSanMoiEngine::Sampled);
}

[[nodiscard]] constexpr bool moi_supports_full_pressure_component_state(rj_code_arch_t arch,
                                                                        ConSanMoiEngine engine) {
  return (is_rdna4_family_arch(arch) || instrumentation::is_cdna_family_arch(arch)) &&
         (engine == ConSanMoiEngine::RecordReplay || engine == ConSanMoiEngine::InlineShadow ||
          engine == ConSanMoiEngine::Sampled);
}

[[nodiscard]] bool moi_sgpr_range_overlaps_dynamic_stack_state(uint16_t base, uint16_t width) {
  return range_overlaps(base, width, kDynamicStackTopSgpr, 1u) ||
         range_overlaps(base, width, kDynamicFrameBaseSgpr, 1u);
}

constexpr uint32_t kMoiLocalIndirectIslandWords = 8u;
constexpr uint32_t kMoiRecordReplayIndirectIslandWords = 7u;
constexpr uint32_t kMoiDenseBarrierEntryKeyPrologueWords = 6u;

[[nodiscard]] constexpr uint32_t
moi_record_replay_entry_island_words(bool spill_backed_scalar_assignment) {
  return kMoiRecordReplayIndirectIslandWords + (spill_backed_scalar_assignment ? 1u : 0u);
}

[[nodiscard]] constexpr uint32_t
moi_dense_barrier_entry_island_words(bool spill_backed_scalar_assignment) {
  // Spill-backed dense barrier routers derive the dispatcher key at entry:
  // key = call return - island offset. The prologue captures SCC itself, so
  // the following indirect jump omits its ordinary SCC-capture word.
  return spill_backed_scalar_assignment
             ? kMoiDenseBarrierEntryKeyPrologueWords + (kMoiRecordReplayIndirectIslandWords - 1u)
             : kMoiRecordReplayIndirectIslandWords;
}

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
      const uint64_t gap_begin = util::align_up(unique[owner_index].code_end, uint64_t{4});
      const uint64_t gap_end =
          owner_index + 1u < unique.size() ? unique[owner_index + 1u].entry : text.size();
      for (uint64_t word_offset = gap_begin; word_offset + sizeof(uint32_t) <= gap_end;
           word_offset += sizeof(uint32_t)) {
        uint32_t word = 0;
        std::memcpy(&word, text.data() + word_offset, sizeof(word));
        if (word == nop_ && !overlaps_patch(word_offset, word_offset + sizeof(uint32_t)))
          available_word_offsets_.insert(word_offset);
      }
      uint64_t cursor = gap_begin;
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

  [[nodiscard]] std::vector<uint64_t> available_word_offsets() const {
    return {available_word_offsets_.begin(), available_word_offsets_.end()};
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
    const uint64_t claim_end = offset + claim_bytes;
    if (std::ranges::any_of(claimed_ranges_, [&](const auto &range) {
          return range.first < claim_end && offset < range.second;
        })) {
      return false;
    }
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
    claimed_ranges_.emplace_back(offset, claim_end);
    available_word_offsets_.erase(available_word_offsets_.lower_bound(offset),
                                  available_word_offsets_.lower_bound(claim_end));
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
  std::set<uint64_t> available_word_offsets_;
  std::vector<std::pair<uint64_t, uint64_t>> claimed_ranges_;
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

[[nodiscard]] std::optional<uint16_t> moi_descriptor_kernarg_segment_ptr_sgpr(const KD &desc) {
  const uint16_t properties = desc.kernel_code_properties;
  if (AMDHSA_BITS_GET(properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR) == 0)
    return std::nullopt;
  return consan_moi_amdhsa_dispatch_id_prefix_sgpr_count(
      AMDHSA_BITS_GET(properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER) != 0,
      AMDHSA_BITS_GET(properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR) != 0,
      AMDHSA_BITS_GET(properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR) != 0, false);
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
  ConSanMoiDispatchIdPreloadPlan plan = consan_moi_plan_dispatch_id_preload(
      original_user_count, system_count, prefix, dispatch_enabled, kMaxSgprs,
      user_sgpr_initialization_limit);
  const uint16_t preload_length =
      AMDHSA_BITS_GET(desc.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_LENGTH);
  plan.original_kernarg_preload_length = preload_length;
  plan.replacement_kernarg_preload_length = preload_length;
  const auto kernarg_ptr = moi_descriptor_kernarg_segment_ptr_sgpr(desc);
  // CDNA3/4 initialize at most 16 user SGPRs. When dispatch ID would exceed
  // that window by one or two dwords, shorten the hardware kernarg preload by
  // exactly that overflow and recover those guest values from the still-live
  // kernarg pointer in the entry prologue. System SGPRs move only by the
  // remaining user-window growth, so record that smaller repair separately.
  const uint16_t overflow =
      original_user_count + 2u > user_sgpr_initialization_limit
          ? static_cast<uint16_t>(original_user_count + 2u - user_sgpr_initialization_limit)
          : 0u;
  if ((arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4) &&
      plan.support == ConSanMoiDispatchIdPreloadSupport::UserSgprInitializationLimit &&
      !dispatch_enabled && overflow >= 1u && overflow <= 2u && kernarg_ptr &&
      preload_length > overflow && prefix + named_suffix == original_user_count) {
    plan.support = ConSanMoiDispatchIdPreloadSupport::SupportedInsert;
    plan.expanded_user_sgpr_count = user_sgpr_initialization_limit;
    plan.first_shifted_guest_sgpr = prefix;
    plan.shifted_guest_sgpr_count = static_cast<uint16_t>(original_user_count - prefix - overflow);
    plan.kernarg_reload_sgpr = static_cast<uint16_t>(original_user_count - overflow);
    plan.kernarg_reload_base_sgpr = *kernarg_ptr;
    plan.kernarg_reload_offset_dwords = static_cast<uint16_t>(
        AMDHSA_BITS_GET(desc.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_OFFSET) + preload_length -
        overflow);
    plan.kernarg_reload_count = overflow;
    plan.system_sgpr_shift = static_cast<uint16_t>(2u - overflow);
    plan.shifted_system_sgpr_count = plan.system_sgpr_shift != 0u ? system_count : 0u;
    plan.original_kernarg_preload_length = preload_length;
    plan.replacement_kernarg_preload_length = static_cast<uint16_t>(preload_length - overflow);
    plan.required_sgpr_count = static_cast<uint16_t>(plan.expanded_user_sgpr_count + system_count);
  }
  return plan;
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

[[nodiscard]] std::optional<ConSanMoiWorkgroupSources> moi_descriptor_workgroup_sources(
    std::span<const uint8_t> image, uint64_t descriptor_file_offset, rj_code_arch_t arch,
    std::vector<std::string> &errors, bool uses_gfx1250_cluster_workgroup_id = false,
    std::optional<uint16_t> cdna_full_payload_user_sgpr_count = std::nullopt) {
  if (descriptor_file_offset > image.size() || sizeof(KD) > image.size() - descriptor_file_offset) {
    errors.emplace_back("ConSan MOI workgroup-id descriptor exceeds ELF bytes");
    return std::nullopt;
  }

  KD desc{};
  std::memcpy(&desc, image.data() + descriptor_file_offset, sizeof(desc));
  if (arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_GFX1250) {
    // These are the translated RDNA command-processor ABI TTMPs, not the
    // descriptor-selected entry SGPRs copied from them by the kernel prologue.
    // MOI finalization enables the complete command-processor grid payload on
    // every patched owner descriptor. Preserve that full identity: an x-only
    // guest kernel can still be dispatched over a multidimensional grid, and
    // each workgroup owns a distinct LDS instance.
    return ConSanMoiWorkgroupSources{
        .x = ConSanMoiWorkgroupSource::scalar(ttmp_scalar_operand(kTtmpRdna4GridX)),
        .y = ConSanMoiWorkgroupSource::scalar(ttmp_scalar_operand(kTtmpRdna4GridYz),
                                              /*shift_right_16=*/false,
                                              /*mask_low_16=*/true),
        .z = ConSanMoiWorkgroupSource::scalar(ttmp_scalar_operand(kTtmpRdna4GridYz),
                                              /*shift_right_16=*/true),
        .cluster_workgroup_id =
            arch == ROCJITSU_CODE_ARCH_GFX1250 && uses_gfx1250_cluster_workgroup_id
                ? ConSanMoiWorkgroupSource::scalar(
                      ttmp_scalar_operand(kTtmpGfx1250ClusterWorkgroupId))
                : ConSanMoiWorkgroupSource{},
        .cdna_full_payload_base = std::nullopt,
        .cdna_guest_payload_base = std::nullopt,
        .cdna_guest_payload_mask = 0u,
    };
  }
  if (cdna_full_payload_user_sgpr_count &&
      (arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4)) {
    const uint16_t full_base = *cdna_full_payload_user_sgpr_count;
    if (static_cast<uint32_t>(full_base) + 3u > kMaxSgprs) {
      errors.emplace_back("ConSan MOI full CDNA workgroup-ID payload exceeds scalar registers");
      return std::nullopt;
    }
    const uint32_t rsrc2 = desc.compute_pgm_rsrc2;
    uint8_t guest_mask = 0;
    guest_mask |= AMDHSA_BITS_GET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X) != 0
                      ? 1u << 0u
                      : 0u;
    guest_mask |= AMDHSA_BITS_GET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y) != 0
                      ? 1u << 1u
                      : 0u;
    guest_mask |= AMDHSA_BITS_GET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z) != 0
                      ? 1u << 2u
                      : 0u;
    guest_mask |= AMDHSA_BITS_GET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_INFO) != 0
                      ? 1u << 3u
                      : 0u;
    return ConSanMoiWorkgroupSources{
        .x = ConSanMoiWorkgroupSource::scalar(full_base),
        .y = ConSanMoiWorkgroupSource::scalar(static_cast<uint16_t>(full_base + 1u)),
        .z = ConSanMoiWorkgroupSource::scalar(static_cast<uint16_t>(full_base + 2u)),
        .cluster_workgroup_id = {},
        .cdna_full_payload_base = full_base,
        .cdna_guest_payload_base = static_cast<uint16_t>(moi_descriptor_user_sgpr_count(desc)),
        .cdna_guest_payload_mask = guest_mask,
    };
  }
  const auto sgpr_source = [](std::optional<uint16_t> sgpr) {
    return sgpr ? ConSanMoiWorkgroupSource::scalar(*sgpr) : ConSanMoiWorkgroupSource{};
  };
  return ConSanMoiWorkgroupSources{
      .x = sgpr_source(moi_descriptor_workgroup_id_sgpr(desc, 0)),
      .y = sgpr_source(moi_descriptor_workgroup_id_sgpr(desc, 1)),
      .z = sgpr_source(moi_descriptor_workgroup_id_sgpr(desc, 2)),
      .cluster_workgroup_id = {},
      .cdna_full_payload_base = std::nullopt,
      .cdna_guest_payload_base = std::nullopt,
      .cdna_guest_payload_mask = 0u,
  };
}

/// Record/Replay probes execute after arbitrary guest code. Capture the
/// complete workgroup tuple at entry instead of depending on descriptor SGPRs
/// or RDNA launch TTMPs retaining their entry values.
[[nodiscard]] std::optional<ConSanMoiWorkgroupSources>
record_replay_persistent_workgroup_sources(const ConSanOptions &options, rj_code_arch_t /*arch*/) {
  if (!record_replay_requires_entry_workgroup_capture(options) ||
      !record_replay_has_entry_workgroup_capture(options) ||
      !record_replay_entry_workgroup_capture_is_unambiguous(options)) {
    return std::nullopt;
  }
  const auto scalar_source = [](uint16_t sgpr) { return ConSanMoiWorkgroupSource::scalar(sgpr); };
  const auto vector_source = [](uint16_t vgpr) { return ConSanMoiWorkgroupSource::vector(vgpr); };
  const auto private_source = [](uint32_t offset) {
    return ConSanMoiWorkgroupSource::private_state(offset);
  };
  if (options.moi_record_replay_workgroup_sgprs.complete()) {
    return ConSanMoiWorkgroupSources{
        .x = scalar_source(*options.moi_record_replay_workgroup_sgprs.x),
        .y = scalar_source(*options.moi_record_replay_workgroup_sgprs.y),
        .z = scalar_source(*options.moi_record_replay_workgroup_sgprs.z),
        .cluster_workgroup_id = {},
        .cdna_full_payload_base = std::nullopt,
        .cdna_guest_payload_base = std::nullopt,
        .cdna_guest_payload_mask = 0u,
    };
  }
  if (options.moi_record_replay_workgroup_vgprs.complete()) {
    return ConSanMoiWorkgroupSources{
        .x = vector_source(*options.moi_record_replay_workgroup_vgprs.x),
        .y = vector_source(*options.moi_record_replay_workgroup_vgprs.y),
        .z = vector_source(*options.moi_record_replay_workgroup_vgprs.z),
        .cluster_workgroup_id = {},
        .cdna_full_payload_base = std::nullopt,
        .cdna_guest_payload_base = std::nullopt,
        .cdna_guest_payload_mask = 0u,
    };
  }
  if (options.moi_record_replay_workgroup_private_offsets.complete()) {
    return ConSanMoiWorkgroupSources{
        .x = private_source(*options.moi_record_replay_workgroup_private_offsets.x),
        .y = private_source(*options.moi_record_replay_workgroup_private_offsets.y),
        .z = private_source(*options.moi_record_replay_workgroup_private_offsets.z),
        .cluster_workgroup_id = {},
        .cdna_full_payload_base = std::nullopt,
        .cdna_guest_payload_base = std::nullopt,
        .cdna_guest_payload_mask = 0u,
    };
  }
  return std::nullopt;
}

[[nodiscard]] uint32_t moi_descriptor_vgpr_granularity(const KD &desc, rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4)
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

[[nodiscard]] uint32_t moi_descriptor_unified_vgpr_allocation_count(const KD &desc,
                                                                    rj_code_arch_t arch) {
  const uint32_t granularity = moi_descriptor_vgpr_granularity(desc, arch);
  const uint32_t granulated =
      AMDHSA_BITS_GET(desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
  return (granulated + 1u) * granularity;
}

[[nodiscard]] uint32_t moi_descriptor_sgpr_allocation_count(const KD &desc, rj_code_arch_t arch) {
  const uint32_t granulated = AMDHSA_BITS_GET(
      desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT);
  const uint32_t allocated = amdgpu_kernel_descriptor_sgpr_count(granulated, arch);
  return std::min<uint32_t>(allocated, kMaxSgprs);
}

struct MoiSgprRange {
  uint16_t base = 0;
  uint16_t width = 0;
};

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
                                                       rj_code_arch_t arch,
                                                       bool proven_empty_accumulator_bank = false) {
  if (required_count == 0 || required_count > kMaxVgprs)
    return false;

  if (arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4) {
    const uint32_t encoded_accum_offset =
        AMDHSA_BITS_GET(desc.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET);
    if (encoded_accum_offset != 0u) {
      constexpr uint32_t kAccvgprOffsetGranularity = 4u;
      const uint32_t accvgpr_base = (encoded_accum_offset + 1u) * kAccvgprOffsetGranularity;
      if (required_count <= accvgpr_base)
        return true;

      const uint32_t unified_count = moi_descriptor_unified_vgpr_allocation_count(desc, arch);
      // Instrumentation may move only an empty accumulator boundary. Moving
      // live accumulator storage would require relocating guest register
      // contents and rewriting every accumulator operand.
      if (unified_count > accvgpr_base && !proven_empty_accumulator_bank)
        return false;
      const uint32_t granularity = moi_descriptor_vgpr_granularity(desc, arch);
      const uint32_t new_unified_count =
          std::max(unified_count, util::align_up(required_count, granularity));
      const uint32_t new_accvgpr_base = new_unified_count;
      if (new_accvgpr_base > kMaxVgprs || new_accvgpr_base % kAccvgprOffsetGranularity != 0u ||
          new_unified_count / granularity > 64u)
        return false;
      AMDHSA_BITS_SET(desc.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET,
                      ((new_accvgpr_base / kAccvgprOffsetGranularity) - 1u));
      AMDHSA_BITS_SET(desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                      ((new_unified_count / granularity) - 1u));
      return true;
    }
  }

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

[[nodiscard]] bool grow_moi_descriptor_sgpr_allocation(KD &desc, uint32_t required_count,
                                                       rj_code_arch_t arch) {
  if (required_count == 0 || required_count > kMaxSgprs)
    return false;
  if (required_count <= moi_descriptor_sgpr_allocation_count(desc, arch))
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
  if (!grow_moi_descriptor_vgpr_allocation(desc, required_count, arch, kernel.agpr_count == 0u)) {
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
  return consan_detail::inline_shadow_transaction_scratch_count(
      options.moi_exec_save_sgpr.has_value(), options.moi_track_atomics);
}

[[nodiscard]] uint16_t inline_shadow_loop_scratch_count(const ConSanOptions &options,
                                                        const ConSanMoiCandidate &candidate) {
  (void)options;
  // Wide local accesses retain an offset and iteration counter. Wide external
  // accesses retain an iteration counter and the workgroup key, since the
  // versioned transaction reuses all of its ordinary address temporaries.
  return consan_detail::inline_shadow_loop_scratch_count(candidate.width_bits,
                                                         consan_moi_exact_shadow::granule_bytes);
}

[[nodiscard]] uint16_t moi_exec_save_sgpr_count(const ConSanOptions &options, rj_code_arch_t arch);

[[nodiscard]] bool moi_inline_uses_planned_owner_sgpr(const ConSanOptions &options) {
  return options.moi_engine == ConSanMoiEngine::InlineShadow &&
         options.moi_owner_source == ConSanMoiOwnerSource::HwId &&
         (!options.moi_owner_sgpr || options.automatic_moi_owner_sgpr);
}

[[nodiscard]] bool validate_inline_shadow_exec_save_sgpr(const ConSanOptions &options,
                                                         rj_code_arch_t arch,
                                                         std::vector<std::string> &errors) {
  if (!options.moi_exec_save_sgpr)
    return true;
  if (options.automatic_moi_inline_sgpr_spill && options.moi_inline_access_present &&
      ((!options.moi_inline_call_return_sgpr || !options.moi_inline_dispatch_key_sgpr) &&
       !options.moi_inline_branch_only_scalar_spill &&
       !options.moi_inline_dynamic_stack_borrowed_sgpr)) {
    errors.emplace_back(
        "ConSan MOI spill-backed inline-shadow probes require a dense router or branch-only "
        "scalar spill");
    return false;
  }
  // Diagnostics use s[base:base+10]. Atomic publication and causal-token
  // handling retain masks through +20:+21. Access probes additionally use
  // local/dynamic state through +25 and preserve application EXEC in the
  // dedicated +26:+27 pair.
  const uint16_t ordinary_sgpr_count =
      arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4 ? 102u : kMaxSgprs;
  const uint16_t required_sgpr_count = moi_exec_save_sgpr_count(options, arch);
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

// A Sampled store executes its displaced guest instruction before the probe
// uses ordinary scratch VGPRs. That ordering permits a spill-backed scratch
// window to overlap the store's address or payload: the spill save preserves
// the live values, the guest consumes them, and the final restore cannot erase
// a guest result because stores have no destination VGPR. Keep the proof
// deliberately narrow. Address materialization and scratch-backed owner state
// both write the scratch window before the guest, while loads produce a value
// that the final restore would overwrite.
[[nodiscard]] bool
sampled_access_can_emit_spill_over_guest_operands(const ConSanOptions &options,
                                                  const ConSanMoiCandidate &candidate) {
  return options.moi_engine == ConSanMoiEngine::Sampled && !candidate.dst_vgpr &&
         options.moi_owner_vgpr && !options.automatic_moi_private_epoch &&
         !moi_uses_scalar_persistent_state(options) &&
         !candidate_requires_gfx1250_flat_offset_materialization(candidate);
}

[[nodiscard]] bool
sampled_access_can_plan_spill_over_guest_operands(const ConSanOptions &options,
                                                  const ConSanMoiCandidate &candidate) {
  if (sampled_access_can_emit_spill_over_guest_operands(options, candidate))
    return true;
  if (options.moi_engine != ConSanMoiEngine::Sampled || candidate.dst_vgpr ||
      options.automatic_moi_private_epoch || moi_uses_scalar_persistent_state(options) ||
      candidate_requires_gfx1250_flat_offset_materialization(candidate)) {
    return false;
  }
  // Automatic persistent-state placement runs after the first resource pass.
  // Before that pass, v0 remains forbidden as the provisional workitem owner,
  // which can hide an otherwise valid forced-spill window. Admit the store
  // overlap provisionally when this transform requires entry-persistent
  // owner state. The post-placement resource rebuild and emission predicate
  // above still require a concrete persistent owner VGPR, so private/scalar
  // fallbacks or failed placement cannot turn this planning allowance into
  // generated code.
  return options.moi_init_owner_epoch || options.moi_track_atomics || options.moi_track_barriers ||
         options.moi_runtime_sample_stride > 1u;
}

[[nodiscard]] bool sampled_access_supports_spill_backed_operand_recovery(
    const ConSanOptions &options, const ConSanMoiCandidate &candidate, rj_code_arch_t arch) {
  return options.moi_engine == ConSanMoiEngine::Sampled &&
         candidate.source == ConSanMoiCandidateSource::NativeLds &&
         (arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_CDNA3 ||
          arch == ROCJITSU_CODE_ARCH_CDNA4) &&
         !candidate_requires_gfx1250_flat_offset_materialization(candidate);
}

[[nodiscard]] uint16_t inline_shadow_scratch_count(const ConSanOptions &options,
                                                   const ConSanMoiCandidate &candidate,
                                                   rj_code_arch_t arch) {
  const uint16_t flat_offset_address_count =
      candidate_requires_gfx1250_flat_offset_materialization(candidate) ? 2u : 0u;
  const uint16_t dynamic_stack_reservoir_count =
      is_rdna4_family_arch(arch) && options.moi_dynamic_stack_spill
          ? DynamicStackBorrowedSgprSpillSequence::kScalarReservoirCount
          : 0u;
  const uint16_t two_address_replay_count =
      arch == ROCJITSU_CODE_ARCH_GFX1250 &&
              two_address_native_lds_offset_scale(candidate.mnemonic).value_or(0u) > 8u
          ? 1u
          : 0u;
  return static_cast<uint16_t>(
      inline_shadow_scratch_count(options) + inline_shadow_loop_scratch_count(options, candidate) +
      (moi_load_clobbers_address(candidate) && flat_offset_address_count == 0u ? 1u : 0u) +
      flat_offset_address_count + dynamic_stack_reservoir_count + two_address_replay_count);
}

[[nodiscard]] uint16_t inline_shadow_spill_backed_scratch_count(const ConSanOptions &options,
                                                                const ConSanMoiCandidate &candidate,
                                                                rj_code_arch_t arch) {
  const uint16_t normal_count = inline_shadow_scratch_count(options, candidate, arch);
  // A spill-backed CDNA probe can recover an overlapping guest address from
  // its authoritative private slot into a phase-shared transaction register.
  // Only the non-spill path must retain a dedicated address snapshot when a
  // load overwrites its address operand.
  const bool can_reload_clobbered_address =
      (arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4) &&
      candidate.source == ConSanMoiCandidateSource::NativeLds &&
      moi_load_clobbers_address(candidate) &&
      !candidate_requires_gfx1250_flat_offset_materialization(candidate);
  return static_cast<uint16_t>(normal_count - (can_reload_clobbered_address ? 1u : 0u));
}

[[nodiscard]] uint16_t direct_sampled_scratch_count(const ConSanOptions &options,
                                                    const ConSanMoiCandidate &candidate,
                                                    rj_code_arch_t arch) {
  const uint16_t base_scratch_count = options.moi_sampled_check ? 7u : 5u;
  const uint16_t two_address_replay_count =
      arch == ROCJITSU_CODE_ARCH_GFX1250 &&
              two_address_native_lds_offset_scale(candidate.mnemonic).value_or(0u) > 8u
          ? 1u
          : 0u;
  return static_cast<uint16_t>(
      base_scratch_count + (options.moi_runtime_sample_stride > 1 ? 1u : 0u) +
      (candidate_requires_gfx1250_flat_offset_materialization(candidate) ? 2u
       : moi_load_clobbers_address(candidate)                            ? 1u
                                                                         : 0u) +
      (options.automatic_moi_private_epoch
           ? (options.moi_owner_source == ConSanMoiOwnerSource::WorkitemId ? 2u : 1u)
       : moi_uses_scalar_persistent_state(options) ? 2u
                                                   : 0u) +
      two_address_replay_count);
}

[[nodiscard]] uint16_t sampled_spill_backed_scratch_count(const ConSanOptions &options,
                                                          const ConSanMoiCandidate &candidate,
                                                          rj_code_arch_t arch) {
  // The publication phases recover an overlapping address into an ordinary
  // temporary only while that phase consumes it. This keeps the fallback
  // within the same scratch budget instead of requiring a ninth live value at
  // full-pressure eight-register Sampled sites.
  return direct_sampled_scratch_count(options, candidate, arch);
}

struct SampledAtomicScratchLayout {
  static constexpr uint16_t kValue = 2u;
  static constexpr uint16_t kExpected = 3u;
  // Sampled atomic lowering currently admits only 32-bit CAS, so each
  // outcome-evidence field occupies one VGPR. Wider support must also widen
  // append_sampled_atomic_cas_snapshot(), not only this reservation.
  static constexpr uint16_t kCasFieldWords = 1u;
  static_assert(kCasFieldWords == 1u);
  static constexpr uint16_t kCasCompare = 4u;
  static constexpr uint16_t kCasResult = kCasCompare + kCasFieldWords;
  static constexpr uint16_t kOwner = kCasResult + kCasFieldWords;
  static constexpr uint16_t kEpoch = kOwner + 1u;
  static constexpr uint16_t kSavedAddress = kEpoch + 1u;
  static constexpr uint16_t kSavedAddressCount = 2u;
  static constexpr uint16_t kCount = kSavedAddress + kSavedAddressCount;
};

[[nodiscard]] constexpr uint16_t sampled_atomic_scratch_count() {
  return SampledAtomicScratchLayout::kCount;
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
  bool cdna_accvgpr_boundary_is_allocation_tail = false;
  bool can_move_empty_cdna_accvgpr_boundary = false;
  uint16_t current_sgpr_count = 0;
  uint16_t max_referenced_sgpr_count = 0;
  uint32_t original_private_segment_size = 0;
  bool has_indirect_sgpr_access = false;
  bool sgpr_reference_coverage_complete = false;
  bool has_indirect_vgpr_access = false;
  bool descriptor_valid = false;
  bool liveness_is_restricted = false;

  [[nodiscard]] consan_detail::ScalarOwnerContextSummary scalar_summary() const {
    return {
        .descriptor_file_offset =
            kernel ? std::optional{kernel->descriptor_file_offset} : std::nullopt,
        .current_sgpr_count = current_sgpr_count,
        .max_referenced_sgpr_count = max_referenced_sgpr_count,
        .has_indirect_sgpr_access = has_indirect_sgpr_access,
        .sgpr_reference_coverage_complete = sgpr_reference_coverage_complete,
        .descriptor_valid = descriptor_valid,
    };
  }
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

/// Returns true when automatic scalar spilling selected after the initial
/// resource pass must share a site-local frame with a dynamic-stack owner.
/// The first pass is still useful: it avoids reserving the dynamic frame ABI
/// for ordinary owners that can keep their transient state in registers.
[[nodiscard]] bool
automatic_moi_scalar_spill_needs_dynamic_stack_planning(const ConSanOptions &options,
                                                        const ConSanResult &result) {
  const auto owner_uses_dynamic_stack = [&](uint64_t descriptor_offset) {
    const auto kernel = std::ranges::find(result.kernels, descriptor_offset,
                                          &ConSanKernelInfo::descriptor_file_offset);
    return kernel != result.kernels.end() && kernel->uses_dynamic_stack.value_or(false);
  };
  if (std::ranges::any_of(options.moi_transient_sgpr_assignments,
                          [&](const ConSanMoiTransientSgprAssignment &assignment) {
                            return assignment.spill_backed &&
                                   owner_uses_dynamic_stack(assignment.descriptor_file_offset);
                          })) {
    return true;
  }
  if (!options.automatic_moi_inline_sgpr_spill && !options.automatic_moi_record_replay_sgpr_spill) {
    return false;
  }
  return std::ranges::any_of(result.resource_plans, [&](const ConSanCandidateResourcePlan &plan) {
    return plan.source != ConSanRegisterAllocationSource::Unsupported &&
           std::ranges::any_of(plan.owner_descriptor_file_offsets, owner_uses_dynamic_stack);
  });
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
    if (selected &&
        (selected->exec_save_sgpr != assignment->exec_save_sgpr ||
         selected->owner_sgpr != assignment->owner_sgpr ||
         selected->dispatch_id_sgpr != assignment->dispatch_id_sgpr ||
         selected->spill_backed != assignment->spill_backed ||
         selected->indirect_pc_sgpr != assignment->indirect_pc_sgpr ||
         selected->indirect_scc_sgpr != assignment->indirect_scc_sgpr ||
         selected->dispatch_key_sgpr != assignment->dispatch_key_sgpr ||
         selected->call_return_sgpr != assignment->call_return_sgpr ||
         selected->visible_evidence_sgpr != assignment->visible_evidence_sgpr ||
         selected->branch_only_scalar_spill != assignment->branch_only_scalar_spill ||
         selected->dynamic_stack_borrowed_sgpr != assignment->dynamic_stack_borrowed_sgpr))
      return false;
    selected = assignment;
  }
  if (!selected) {
    return options.moi_exec_save_sgpr.has_value() ||
           (options.moi_engine == ConSanMoiEngine::InlineShadow &&
            options.automatic_moi_inline_sgpr_spill &&
            ((options.moi_inline_indirect_pc_sgpr && options.moi_inline_indirect_scc_sgpr &&
              options.moi_inline_dispatch_key_sgpr && options.moi_inline_call_return_sgpr) ||
             options.moi_inline_branch_only_scalar_spill ||
             options.moi_inline_dynamic_stack_borrowed_sgpr));
  }
  if (saw_unassigned_owner)
    return false;
  options.moi_exec_save_sgpr = selected->exec_save_sgpr;
  options.moi_owner_sgpr = selected->owner_sgpr;
  // A component assignment replaces the object-wide scalar ABI. Its exact
  // persistent outputs are represented by the fields copied below, so the
  // broad object-wide persistence marker must not reject a spill-backed
  // window whose incoming contents are deliberately backed up at entry.
  if (selected->spill_backed)
    options.moi_exec_save_sgprs_persistent = false;
  options.moi_dispatch_id_sgpr = selected->dispatch_id_sgpr;
  options.moi_inline_indirect_pc_sgpr = selected->indirect_pc_sgpr;
  options.moi_inline_indirect_scc_sgpr = selected->indirect_scc_sgpr;
  if (options.moi_engine == ConSanMoiEngine::InlineShadow) {
    options.automatic_moi_inline_sgpr_spill = selected->spill_backed;
    options.automatic_moi_record_replay_sgpr_spill = false;
    options.moi_inline_dispatch_key_sgpr = selected->dispatch_key_sgpr;
    options.moi_inline_call_return_sgpr = selected->call_return_sgpr;
    options.moi_inline_visible_evidence_sgpr = selected->visible_evidence_sgpr;
    options.moi_inline_branch_only_scalar_spill = selected->branch_only_scalar_spill;
    options.moi_inline_dynamic_stack_borrowed_sgpr = selected->dynamic_stack_borrowed_sgpr;
  } else {
    options.automatic_moi_inline_sgpr_spill = false;
    options.moi_inline_branch_only_scalar_spill = false;
    options.moi_inline_dynamic_stack_borrowed_sgpr.reset();
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
    if (selected &&
        (selected->owner_vgpr != assignment->owner_vgpr ||
         selected->epoch_vgpr != assignment->epoch_vgpr ||
         selected->workgroup_key_vgpr != assignment->workgroup_key_vgpr ||
         selected->record_replay_workgroup_vgprs != assignment->record_replay_workgroup_vgprs ||
         selected->dispatch_id_vgpr != assignment->dispatch_id_vgpr)) {
      return false;
    }
    selected = assignment;
  }
  if (!selected)
    return false;
  options.moi_owner_vgpr = selected->owner_vgpr;
  options.moi_epoch_vgpr = selected->epoch_vgpr;
  options.moi_workgroup_key_vgpr = selected->workgroup_key_vgpr;
  options.moi_record_replay_workgroup_vgprs = selected->record_replay_workgroup_vgprs;
  options.moi_dispatch_id_vgpr = selected->dispatch_id_vgpr;
  return true;
}

/// Restore an owner-local persistent assignment when placement could not use a
/// code-object-wide tuple, then verify that the probe has one exact
/// entry-captured workgroup source. A scalar tuple is already global and needs
/// no owner-local VGPR assignment.
[[nodiscard]] bool
apply_record_replay_entry_workgroup_assignment(ConSanOptions &options,
                                               std::span<const uint64_t> owner_descriptor_offsets) {
  if (!record_replay_entry_workgroup_capture_is_unambiguous(options))
    return false;
  if (!record_replay_requires_entry_workgroup_capture(options) ||
      record_replay_has_entry_workgroup_capture(options)) {
    return true;
  }
  return apply_moi_persistent_vgpr_assignment(options, owner_descriptor_offsets) &&
         record_replay_has_entry_workgroup_capture(options) &&
         record_replay_entry_workgroup_capture_is_unambiguous(options);
}

[[nodiscard]] bool is_indirect_sgpr_access(const Instruction &instruction) {
  return instruction.mnemonic().starts_with("s_movrel");
}

[[nodiscard]] bool is_indirect_vgpr_access(const Instruction &instruction) {
  return instruction.mnemonic().starts_with("v_movrel");
}

[[nodiscard]] bool has_indirect_vgpr_access(const KernelCfgScope &scope) {
  for (BasicBlock *block : scope.blocks) {
    if (block == nullptr)
      continue;
    if (std::ranges::any_of(block->instructions(), is_indirect_vgpr_access)) {
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

enum class MoiUnclosedControlKind {
  Branch,
  Call,
  Fallthrough,
};

struct MoiOwnerRegisterReferenceSummary {
  RegisterSet references;
  bool uses_indirect_sgprs = false;
  bool uses_indirect_vgprs = false;
  bool control_flow_closed = true;
  std::optional<uint64_t> first_unclosed_control_offset;
  std::string first_unclosed_control_mnemonic;
  MoiUnclosedControlKind first_unclosed_control_kind = MoiUnclosedControlKind::Branch;
};

[[nodiscard]] MoiOwnerRegisterReferenceSummary summarize_moi_owner_register_references(
    std::span<const MoiKernelResourceContext *const> owner_contexts) {
  MoiOwnerRegisterReferenceSummary result;
  std::unordered_set<BasicBlock *> visited;
  const auto classify_unclosed = [](const Instruction &instruction) {
    if ((instruction.flags() & INDIRECT_CALL) != 0)
      return MoiUnclosedControlKind::Call;
    if ((instruction.flags() & (BRANCH | COND_BRANCH | INDIRECT_BRANCH)) != 0)
      return MoiUnclosedControlKind::Branch;
    return MoiUnclosedControlKind::Fallthrough;
  };
  const auto mark_unclosed = [&](const Instruction &instruction,
                                 std::optional<MoiUnclosedControlKind> kind = std::nullopt) {
    result.control_flow_closed = false;
    if (!result.first_unclosed_control_offset) {
      result.first_unclosed_control_offset = instruction.src_loc();
      result.first_unclosed_control_mnemonic = instruction.mnemonic();
      result.first_unclosed_control_kind = kind.value_or(classify_unclosed(instruction));
    }
  };
  const auto classify_static_issue = [](BasicBlock::StaticSuccessorIssue issue) {
    switch (issue) {
    case BasicBlock::StaticSuccessorIssue::MissingCallTarget:
    case BasicBlock::StaticSuccessorIssue::MissingCallContinuation:
      return MoiUnclosedControlKind::Call;
    case BasicBlock::StaticSuccessorIssue::MissingBranchTarget:
      return MoiUnclosedControlKind::Branch;
    case BasicBlock::StaticSuccessorIssue::MissingFallthrough:
    case BasicBlock::StaticSuccessorIssue::None:
      return MoiUnclosedControlKind::Fallthrough;
    }
    return MoiUnclosedControlKind::Fallthrough;
  };
  for (const MoiKernelResourceContext *context : owner_contexts) {
    if (context == nullptr)
      continue;

    // Closure is context-specific: a setpc can be a validated return for one
    // owner while remaining an unproven branch for another. Require every CFG
    // edge and every exhaustive recovered target to remain inside this owner's
    // scope before using owner-local references as a whole-program proof.
    std::unordered_set<const BasicBlock *> scope_blocks;
    std::unordered_set<uint64_t> scope_block_offsets;
    for (BasicBlock *block : context->scope.blocks) {
      if (block == nullptr)
        continue;
      scope_blocks.insert(block);
      scope_block_offsets.insert(block->start_offset());
    }

    for (BasicBlock *block : context->scope.blocks) {
      if (block == nullptr)
        continue;

      const Instruction *terminator = block->terminator();
      const auto mark_unclosed_terminator = [&]() {
        if (terminator != nullptr)
          mark_unclosed(*terminator);
      };
      if (!block->static_successors_complete() && terminator != nullptr)
        mark_unclosed(*terminator, classify_static_issue(block->static_successor_issue()));
      if (std::ranges::any_of(block->successors(), [&](const BasicBlock *successor) {
            return successor == nullptr || !scope_blocks.contains(successor);
          })) {
        mark_unclosed_terminator();
      }
      if (std::ranges::any_of(block->call_edges(), [&](const BasicBlock::CallEdge &call) {
            return call.callee == nullptr || call.continuation == nullptr ||
                   !scope_blocks.contains(call.callee) || !scope_blocks.contains(call.continuation);
          })) {
        mark_unclosed_terminator();
      }

      const bool scan_references = visited.insert(block).second;
      for (const Instruction &instruction : block->instructions()) {
        const uint64_t offset = instruction.src_loc();
        const bool indirect_call =
            (instruction.flags() & INDIRECT_CALL) != 0 && !instruction.branch_offset_bytes();
        const bool indirect_branch = (instruction.flags() & INDIRECT_BRANCH) != 0;
        const bool validated_return = context->scope.call_return_offsets.contains(offset);
        if (indirect_call || (indirect_branch && !validated_return)) {
          bool found_fixup = false;
          bool targets_exhaustive = true;
          for (const IndirectCallFixup &fixup : block->static_indirect_call_fixups()) {
            if (fixup.source_call_offset != offset)
              continue;
            found_fixup = true;
            targets_exhaustive = targets_exhaustive && fixup.source_targets_exhaustive &&
                                 scope_block_offsets.contains(fixup.source_target_offset);
          }
          if (!found_fixup || !targets_exhaustive)
            mark_unclosed(instruction);
        }

        if (!scan_references)
          continue;
        if (is_indirect_sgpr_access(instruction))
          result.uses_indirect_sgprs = true;
        if (is_indirect_vgpr_access(instruction))
          result.uses_indirect_vgprs = true;
        const InstDefUse def_use(instruction);
        result.references |= def_use.defs | def_use.uses;
      }
    }
  }
  return result;
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
    return static_cast<uint16_t>(
        (record_replay_uses_automatic_banked_capture(options) ? 10u : 6u) +
        (candidate_requires_gfx1250_flat_offset_materialization(candidate) ? 2u
         : moi_load_clobbers_address(candidate)                            ? 1u
                                                                           : 0u) +
        (arch == ROCJITSU_CODE_ARCH_GFX1250 &&
                 two_address_native_lds_offset_scale(candidate.mnemonic).value_or(0u) > 8u
             ? 1u
             : 0u));
  case ConSanMoiEngine::InlineShadow:
    return inline_shadow_scratch_count(options, candidate, arch);
  case ConSanMoiEngine::Sampled:
    return direct_sampled_scratch_count(options, candidate, arch);
  }
  return 0;
}

[[nodiscard]] bool moi_context_references_sgpr_ranges(const MoiKernelResourceContext &context,
                                                      std::span<const MoiSgprRange> ranges) {
  const auto intersects = [&](const RegisterSet &set) {
    return std::ranges::any_of(ranges, [&](const MoiSgprRange &range) {
      for (uint16_t index = 0; index < range.width; ++index) {
        if (set.contains({RegClass::SGPR, static_cast<uint16_t>(range.base + index), 1u})) {
          return true;
        }
      }
      return false;
    });
  };
  for (BasicBlock *block : context.scope.blocks) {
    if (block == nullptr)
      continue;
    for (const Instruction &instruction : block->instructions()) {
      if (is_indirect_sgpr_access(instruction))
        return true;
      const InstDefUse def_use(instruction);
      if (intersects(def_use.defs | def_use.uses))
        return true;
    }
  }
  return false;
}

struct MoiDenseOwnerLivenessProof {
  bool complete = false;
  std::vector<const MoiKernelResourceContext *> contexts;
  std::unordered_set<const MoiKernelResourceContext *> reference_free_contexts;
};

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

    // Resource planning only asks for liveness at instrumentation anchors and
    // kernel entries. Keeping a snapshot for every decoded instruction makes
    // large generated libraries scale with their complete instruction count,
    // even though most instructions are never queried. Per-owner transient
    // assignments are the one exception: their dense relay search may inspect
    // any relocatable instruction in an owning block.
    const bool restrict_liveness_queries = options.moi_transient_sgpr_assignments.empty();
    std::unordered_map<BasicBlock *, std::vector<const Instruction *>> liveness_queries_by_block;
    if (restrict_liveness_queries) {
      std::unordered_set<uint64_t> query_offsets;
      query_offsets.reserve(result.moi_candidates.size() + result.sync_events.size() +
                            result.resource_plans.size() + result.kernels.size());
      for (const ConSanMoiCandidate &candidate : result.moi_candidates)
        query_offsets.insert(candidate.text_offset);
      for (const ConSanSyncEvent &event : result.sync_events)
        query_offsets.insert(event.text_offset);
      for (const ConSanCandidateResourcePlan &plan : result.resource_plans)
        query_offsets.insert(plan.text_offset);
      for (const ConSanSiteDispositionRecord &site : result.site_dispositions)
        query_offsets.insert(site.text_offset);
      for (const ConSanMoiFenceCandidate &fence : result.moi_fence_candidates)
        query_offsets.insert(fence.text_offset);
      for (const ConSanKernelInfo &kernel : result.kernels) {
        if (kernel.has_text_range)
          query_offsets.insert(kernel.entry_text_offset);
      }
      for (uint64_t offset : query_offsets) {
        BasicBlock *block = block_for_offset(block_index, offset);
        if (const Instruction *instruction = instruction_at(block, offset))
          liveness_queries_by_block[block].push_back(instruction);
      }
    }

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
      std::vector<const Instruction *> context_liveness_queries;
      LivenessAnalysisOptions liveness_options;
      if (restrict_liveness_queries) {
        for (BasicBlock *block : stored.scope.blocks) {
          const auto queries = liveness_queries_by_block.find(block);
          if (queries == liveness_queries_by_block.end())
            continue;
          context_liveness_queries.insert(context_liveness_queries.end(), queries->second.begin(),
                                          queries->second.end());
        }
        liveness_options.restrict_live_before_to_instructions = true;
        liveness_options.live_before_instructions = context_liveness_queries;
      }
      stored.liveness = std::make_unique<LivenessAnalysis>(
          KernelBlockScope(stored.scope.blocks), liveness_options, stored.scope.liveness_edges);
      stored.liveness_is_restricted = restrict_liveness_queries;
      stored.max_referenced_vgpr_count = max_referenced_vgpr_count(stored.scope);
      stored.max_referenced_sgpr_count = std::max<uint16_t>(max_referenced_sgpr_count(stored.scope),
                                                            kernel.sgpr_count.value_or(0u));
      const std::array<const MoiKernelResourceContext *, 1> owner = {&stored};
      const MoiOwnerRegisterReferenceSummary scalar_references =
          summarize_moi_owner_register_references(owner);
      stored.has_indirect_sgpr_access = scalar_references.uses_indirect_sgprs;
      stored.sgpr_reference_coverage_complete = scalar_references.control_flow_closed;
      stored.has_indirect_vgpr_access = has_indirect_vgpr_access(stored.scope);

      if (kernel.descriptor_file_offset > bytes.size() ||
          sizeof(KD) > bytes.size() - kernel.descriptor_file_offset) {
        continue;
      }
      KD descriptor{};
      std::memcpy(&descriptor, bytes.data() + kernel.descriptor_file_offset, sizeof(descriptor));
      stored.current_vgpr_count =
          static_cast<uint16_t>(moi_descriptor_vgpr_allocation_count(descriptor, arch));
      if (arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4) {
        const uint32_t encoded_accum_offset = AMDHSA_BITS_GET(
            descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET);
        // A nonzero ACCUM_OFFSET is the first physical AccVGPR and therefore
        // the hard upper bound for ordinary instrumentation VGPRs. Encoded
        // zero remains ambiguous for kernels without an accumulator bank.
        if (encoded_accum_offset != 0u) {
          const uint32_t accvgpr_base =
              std::min<uint32_t>(kMaxVgprs, (encoded_accum_offset + 1u) * 4u);
          const uint32_t unified_vgpr_count =
              moi_descriptor_unified_vgpr_allocation_count(descriptor, arch);
          stored.ordinary_vgpr_limit = static_cast<uint16_t>(accvgpr_base);
          stored.cdna_accvgpr_boundary_is_allocation_tail = unified_vgpr_count <= accvgpr_base;
          stored.can_move_empty_cdna_accvgpr_boundary =
              stored.cdna_accvgpr_boundary_is_allocation_tail || kernel.agpr_count == 0u;
        }
      }
      stored.current_sgpr_count =
          static_cast<uint16_t>(moi_descriptor_sgpr_allocation_count(descriptor, arch));
      stored.original_private_segment_size = descriptor.private_segment_fixed_size;
      stored.descriptor_valid = true;
    }
    context_by_descriptor.reserve(contexts.size());
    for (MoiKernelResourceContext &context : contexts)
      context_by_descriptor.try_emplace(context.kernel->descriptor_file_offset, &context);
    valid = true;
  }

  [[nodiscard]] MoiDenseOwnerLivenessProof
  prepare_dense_liveness_for_owners(std::span<const uint64_t> owners,
                                    std::span<const MoiSgprRange> bootstrap_ranges) {
    MoiDenseOwnerLivenessProof proof;
    proof.complete = !owners.empty() && !bootstrap_ranges.empty();
    proof.contexts.reserve(owners.size());
    proof.reference_free_contexts.reserve(owners.size());
    for (uint64_t owner : owners) {
      const auto found = context_by_descriptor.find(owner);
      if (found == context_by_descriptor.end() || found->second == nullptr) {
        proof.complete = false;
        continue;
      }
      MoiKernelResourceContext &context = *found->second;
      proof.contexts.push_back(&context);
      if (!moi_context_references_sgpr_ranges(context, bootstrap_ranges)) {
        proof.reference_free_contexts.insert(&context);
        continue;
      }
      if (!context.liveness_is_restricted)
        continue;
      context.liveness = std::make_unique<LivenessAnalysis>(KernelBlockScope(context.scope.blocks),
                                                            LivenessAnalysisOptions{},
                                                            context.scope.liveness_edges);
      context.liveness_is_restricted = false;
    }
    return proof;
  }
};

/// Conservatively summarize every instruction in a code object's declared
/// function ranges when those ranges cover the complete text section except
/// for verified compiler/linker padding.
///
/// This is the fail-safe fallback for generated kernels whose local-label
/// calls cannot be reconstructed by CFG analysis. Scanning unrelated kernels
/// can only make a hole unavailable. An unresolved target is treated as
/// represented by the summary only when every byte outside a declared
/// function is an aligned zero or s_nop word, so an uncovered instruction that
/// could clobber the selected state still fails closed.
[[nodiscard]] std::optional<MoiOwnerRegisterReferenceSummary>
summarize_moi_complete_text_register_references(const MoiResourcePlanningState &state) {
  // Function offsets are currently normalized against the code object's sole
  // executable text span. Multi-text objects need section-aware function
  // provenance before their ranges can be tiled independently, so retain the
  // conservative fail-closed boundary here.
  if (!state.valid || state.code_object.text_sections().size() != 1u)
    return std::nullopt;

  const Section &text_section = *state.code_object.text_sections().front();
  const uint64_t text_size = text_section.size();
  const std::span<const uint8_t> text(reinterpret_cast<const uint8_t *>(text_section.data()),
                                      text_section.size());
  const uint32_t padding_nop = build_s_nop(0, state.arch);
  const auto is_padding = [&](uint64_t begin, uint64_t end) {
    if (begin > end || end > text.size() || begin % sizeof(uint32_t) != 0u ||
        end % sizeof(uint32_t) != 0u) {
      return false;
    }
    for (uint64_t offset = begin; offset < end; offset += sizeof(uint32_t)) {
      uint32_t word = 0u;
      std::memcpy(&word, text.data() + offset, sizeof(word));
      if (word != 0u && word != padding_nop)
        return false;
    }
    return true;
  };
  std::vector<std::pair<uint64_t, uint64_t>> ranges;
  ranges.reserve(state.code_object.functions().size());
  for (const AmdGpuFunctionInfo &function : state.code_object.functions()) {
    if (function.code_size == 0u || function.entry_text_offset > text_size ||
        function.code_size > text_size - function.entry_text_offset) {
      return std::nullopt;
    }
    ranges.emplace_back(function.entry_text_offset,
                        function.entry_text_offset + function.code_size);
  }
  std::ranges::sort(ranges);
  uint64_t covered_end = 0u;
  for (const auto &[begin, end] : ranges) {
    if (begin > covered_end && !is_padding(covered_end, begin))
      return std::nullopt;
    covered_end = std::max(covered_end, end);
  }
  if (covered_end > text_size || !is_padding(covered_end, text_size))
    return std::nullopt;

  MoiOwnerRegisterReferenceSummary result;
  for (const std::unique_ptr<BasicBlock> &block : state.blocks) {
    if (!block)
      continue;
    for (const Instruction &instruction : block->instructions()) {
      if (is_indirect_sgpr_access(instruction))
        result.uses_indirect_sgprs = true;
      if (is_indirect_vgpr_access(instruction))
        result.uses_indirect_vgprs = true;
      const InstDefUse def_use(instruction);
      result.references |= def_use.defs | def_use.uses;
    }
  }
  return result;
}

[[nodiscard]] std::vector<uint64_t> moi_scalar_owner_descriptors(const ConSanResult &result) {
  std::vector<uint64_t> owners;
  for (const ConSanCandidateResourcePlan &plan : result.resource_plans) {
    if (plan.source == ConSanRegisterAllocationSource::Unsupported)
      continue;
    for (uint64_t descriptor_offset : plan.owner_descriptor_file_offsets) {
      if (std::ranges::find(owners, descriptor_offset) == owners.end())
        owners.push_back(descriptor_offset);
    }
  }
  return owners;
}

struct MoiScalarOwnerContextResolution {
  std::vector<const MoiKernelResourceContext *> contexts;
  std::vector<consan_detail::ScalarOwnerContextSummary> summaries;
  uint32_t tail_floor = 0;
};

[[nodiscard]] std::optional<MoiScalarOwnerContextResolution>
resolve_moi_scalar_owner_contexts(const MoiResourcePlanningState &state,
                                  std::span<const uint64_t> owners) {
  std::vector<consan_detail::ScalarOwnerContextSummary> summaries;
  summaries.reserve(state.contexts.size());
  for (const MoiKernelResourceContext &context : state.contexts)
    summaries.push_back(context.scalar_summary());
  const auto resolved =
      consan_detail::resolve_scalar_owner_contexts(state.valid, summaries, owners);
  if (!resolved)
    return std::nullopt;

  MoiScalarOwnerContextResolution result;
  result.contexts.reserve(resolved->context_indices.size());
  result.summaries.reserve(resolved->context_indices.size());
  for (size_t context_index : resolved->context_indices) {
    result.contexts.push_back(&state.contexts[context_index]);
    result.summaries.push_back(summaries[context_index]);
  }
  result.tail_floor = resolved->tail_floor;
  return result;
}

struct MoiScalarOwnerContextGroup {
  bool supported = false;
  MoiScalarOwnerContextResolution owners;
};

struct MoiScalarOwnerContextGroupsResolution {
  bool planning_state_valid = false;
  std::vector<MoiScalarOwnerContextGroup> groups;
  std::optional<uint64_t> unresolved_owner;
};

[[nodiscard]] MoiScalarOwnerContextGroupsResolution
resolve_moi_resource_plan_scalar_owner_contexts(const MoiResourcePlanningState &state,
                                                const ConSanResult &result) {
  MoiScalarOwnerContextGroupsResolution resolved;
  resolved.planning_state_valid = state.valid;
  if (!state.valid)
    return resolved;
  resolved.groups.reserve(result.resource_plans.size());
  for (const ConSanCandidateResourcePlan &plan : result.resource_plans) {
    if (plan.owner_descriptor_file_offsets.empty())
      continue;
    auto owners = resolve_moi_scalar_owner_contexts(state, plan.owner_descriptor_file_offsets);
    if (!owners) {
      resolved.unresolved_owner = plan.owner_descriptor_file_offsets.front();
      for (uint64_t owner : plan.owner_descriptor_file_offsets) {
        const auto found = state.context_by_descriptor.find(owner);
        if (found == state.context_by_descriptor.end() || found->second == nullptr ||
            !found->second->descriptor_valid) {
          resolved.unresolved_owner = owner;
          break;
        }
      }
      return resolved;
    }
    resolved.groups.push_back({
        .supported = plan.source != ConSanRegisterAllocationSource::Unsupported,
        .owners = std::move(*owners),
    });
  }
  return resolved;
}

[[nodiscard]] MoiScalarOwnerContextResolution
collect_moi_scalar_owner_contexts(std::span<const MoiScalarOwnerContextGroup> groups,
                                  bool include_unsupported) {
  MoiScalarOwnerContextResolution result;
  for (const MoiScalarOwnerContextGroup &group : groups) {
    if (!include_unsupported && !group.supported)
      continue;
    for (size_t index = 0; index < group.owners.contexts.size(); ++index) {
      const MoiKernelResourceContext *context = group.owners.contexts[index];
      if (std::ranges::find(result.contexts, context) != result.contexts.end())
        continue;
      result.contexts.push_back(context);
      result.summaries.push_back(group.owners.summaries[index]);
      result.tail_floor = std::max<uint32_t>(result.tail_floor, group.owners.tail_floor);
    }
  }
  return result;
}

[[nodiscard]] bool moi_scalar_contexts_conflict_with_physical_vcc(
    std::span<const consan_detail::ScalarOwnerContextSummary> contexts, uint16_t base,
    uint16_t width, rj_code_arch_t arch) {
  if (!instrumentation::is_cdna_family_arch(arch))
    return false;
  const std::array ranges{consan_detail::ScalarOwnerSgprRange{base, width}};
  return consan_detail::scalar_owner_contexts_conflict_with_physical_vcc(contexts, ranges);
}

[[nodiscard]] bool moi_scalar_contexts_conflict_with_physical_vcc(
    std::span<const consan_detail::ScalarOwnerContextSummary> contexts,
    std::span<const MoiSgprRange> ranges, rj_code_arch_t arch) {
  if (!instrumentation::is_cdna_family_arch(arch))
    return false;
  std::vector<consan_detail::ScalarOwnerSgprRange> scalar_ranges;
  scalar_ranges.reserve(ranges.size());
  for (const MoiSgprRange &range : ranges)
    scalar_ranges.push_back({range.base, range.width});
  return consan_detail::scalar_owner_contexts_conflict_with_physical_vcc(contexts, scalar_ranges);
}

[[nodiscard]] bool moi_scalar_contexts_admit_reserved_window(
    std::span<const consan_detail::ScalarOwnerContextSummary> contexts, uint16_t base,
    uint16_t width, rj_code_arch_t arch) {
  return consan_detail::scalar_owner_contexts_admit_reserved_window(
      contexts, base, width, instrumentation::is_cdna_family_arch(arch));
}

[[nodiscard]] std::optional<std::vector<ConSanMoiPrologueScratchVgprAssignment>>
plan_moi_scalar_prologue_scratch(const MoiResourcePlanningState &state,
                                 const MoiScalarOwnerContextResolution &resolution,
                                 bool needs_entry_vgpr_pair) {
  std::vector<ConSanMoiPrologueScratchVgprAssignment> assignments;
  assignments.reserve(resolution.contexts.size());
  for (const MoiKernelResourceContext *context : resolution.contexts) {
    if (!needs_entry_vgpr_pair) {
      assignments.push_back({context->kernel->descriptor_file_offset, 0u});
      continue;
    }
    BasicBlock *entry_block =
        block_for_offset(state.block_index, context->kernel->entry_text_offset);
    const Instruction *entry = instruction_at(entry_block, context->kernel->entry_text_offset);
    if (entry_block == nullptr || entry == nullptr)
      return std::nullopt;
    RegisterSet unavailable_vgprs = context->liveness->live_before(*entry);
    const InstDefUse entry_def_use(*entry);
    unavailable_vgprs |= entry_def_use.defs | entry_def_use.uses;
    // The same trusted zero-AGPR metadata that permits site-local descriptor
    // growth also permits an entry-only scalar-transfer pair above the
    // compiler's rounded ordinary boundary. Descriptor mutation repeats that
    // proof before moving ACCUM_OFFSET. Without it, remain below the original
    // boundary even though the pair is dead before the first guest instruction.
    const uint16_t scratch_limit =
        context->can_move_empty_cdna_accvgpr_boundary ? kMaxVgprs : context->ordinary_vgpr_limit;
    std::optional<uint16_t> scratch;
    for (uint16_t candidate = 0; candidate + 2u <= scratch_limit; ++candidate) {
      if (!unavailable_vgprs.contains({RegClass::VGPR, candidate, 1u}) &&
          !unavailable_vgprs.contains(
              {RegClass::VGPR, static_cast<uint16_t>(candidate + 1u), 1u})) {
        scratch = candidate;
        break;
      }
    }
    if (!scratch)
      return std::nullopt;
    assignments.push_back({context->kernel->descriptor_file_offset, *scratch});
  }
  return assignments;
}

enum class MoiRelocatableHostScalarState : uint8_t {
  AccessRouter,
  BarrierRouter,
};

struct MoiIndirectJumpSgprs {
  uint16_t pc_sgpr = 0;
  uint16_t scc_save_sgpr = 0;
};

[[nodiscard]] std::optional<MoiIndirectJumpSgprs>
moi_indirect_jump_sgprs(const ConSanOptions &options);

struct MoiInlineDenseRouterScalarAbi {
  MoiIndirectJumpSgprs indirect_jump;
  uint16_t dispatch_key_sgpr = 0;
  std::optional<uint16_t> call_return_sgpr;
};

[[nodiscard]] std::optional<MoiInlineDenseRouterScalarAbi>
moi_inline_dense_router_scalar_abi(const ConSanOptions &options, rj_code_arch_t arch) {
  if (options.moi_engine != ConSanMoiEngine::InlineShadow ||
      (arch != ROCJITSU_CODE_ARCH_RDNA4 && arch != ROCJITSU_CODE_ARCH_GFX1250)) {
    return std::nullopt;
  }
  if (options.moi_inline_branch_only_scalar_spill)
    return std::nullopt;
  const auto indirect_jump = moi_indirect_jump_sgprs(options);
  if (!indirect_jump)
    return std::nullopt;

  if (options.automatic_moi_inline_sgpr_spill) {
    if (!options.moi_inline_dispatch_key_sgpr)
      return std::nullopt;
    if (arch == ROCJITSU_CODE_ARCH_GFX1250 && !options.moi_inline_call_return_sgpr)
      return std::nullopt;
    return MoiInlineDenseRouterScalarAbi{
        .indirect_jump = *indirect_jump,
        .dispatch_key_sgpr = *options.moi_inline_dispatch_key_sgpr,
        .call_return_sgpr =
            arch == ROCJITSU_CODE_ARCH_GFX1250 ? options.moi_inline_call_return_sgpr : std::nullopt,
    };
  }

  if (!options.moi_exec_save_sgpr)
    return std::nullopt;
  const uint16_t base = *options.moi_exec_save_sgpr;
  return MoiInlineDenseRouterScalarAbi{
      .indirect_jump = *indirect_jump,
      // GFX1250 derives its key from the scalar-call return PC. RDNA4 has no
      // scalar call and keeps an explicit key in the otherwise-unused return
      // pair's low half.
      .dispatch_key_sgpr = arch == ROCJITSU_CODE_ARCH_GFX1250 ? indirect_jump->pc_sgpr
                                                              : static_cast<uint16_t>(base + 28u),
      .call_return_sgpr = arch == ROCJITSU_CODE_ARCH_GFX1250
                              ? std::optional<uint16_t>(static_cast<uint16_t>(base + 28u))
                              : std::nullopt,
  };
}

[[nodiscard]] std::vector<MoiSgprRange> moi_relocatable_host_scalar_ranges(
    const ConSanOptions &options, MoiRelocatableHostScalarState scalar_state, rj_code_arch_t arch) {
  // These are the ConSan-owned scalar values that remain live across the
  // emitted router, not a guest register window assumed to be preserved.
  // Spill-backed layouts carry their allocated registers explicitly. The
  // fixed-layout offsets below are the matching emission ABI fields.
  if (options.automatic_moi_record_replay_sgpr_spill) {
    if (!options.moi_inline_indirect_pc_sgpr || !options.moi_inline_indirect_scc_sgpr ||
        !options.moi_record_replay_dispatch_key_sgpr ||
        !options.moi_record_replay_call_return_sgpr) {
      return {};
    }
    return {{*options.moi_inline_indirect_pc_sgpr, 2u},
            {*options.moi_inline_indirect_scc_sgpr, 1u},
            {*options.moi_record_replay_dispatch_key_sgpr, 1u},
            {*options.moi_record_replay_call_return_sgpr, 2u}};
  }
  if (options.automatic_moi_inline_sgpr_spill) {
    if (scalar_state == MoiRelocatableHostScalarState::BarrierRouter) {
      if (!options.moi_inline_indirect_pc_sgpr || !options.moi_inline_indirect_scc_sgpr)
        return {};
      std::vector<MoiSgprRange> ranges{{*options.moi_inline_indirect_pc_sgpr, 2u},
                                       {*options.moi_inline_indirect_scc_sgpr, 1u}};
      if (options.moi_inline_visible_evidence_sgpr)
        ranges.push_back({*options.moi_inline_visible_evidence_sgpr, 1u});
      return ranges;
    }
    const auto abi = moi_inline_dense_router_scalar_abi(options, arch);
    if (!abi)
      return {};
    std::vector<MoiSgprRange> ranges{{abi->indirect_jump.pc_sgpr, 2u},
                                     {abi->indirect_jump.scc_save_sgpr, 1u},
                                     {abi->dispatch_key_sgpr, 1u}};
    if (abi->call_return_sgpr)
      ranges.push_back({*abi->call_return_sgpr, 2u});
    return ranges;
  }
  if (!options.moi_exec_save_sgpr)
    return {};
  const uint16_t base = *options.moi_exec_save_sgpr;
  if (options.moi_engine == ConSanMoiEngine::InlineShadow &&
      scalar_state == MoiRelocatableHostScalarState::AccessRouter) {
    const auto abi = moi_inline_dense_router_scalar_abi(options, arch);
    if (!abi)
      return {};
    std::vector<MoiSgprRange> ranges{{abi->indirect_jump.pc_sgpr, 2u},
                                     {abi->indirect_jump.scc_save_sgpr, 1u},
                                     {abi->dispatch_key_sgpr, 1u}};
    if (abi->call_return_sgpr)
      ranges.push_back({*abi->call_return_sgpr, 2u});
    return ranges;
  }
  const uint16_t scc_offset = options.moi_engine == ConSanMoiEngine::InlineShadow ? 10u : 4u;
  return {{base, 2u},
          {static_cast<uint16_t>(base + scc_offset), 1u},
          {static_cast<uint16_t>(base + 5u), 1u},
          {static_cast<uint16_t>(base + 6u), 2u}};
}

[[nodiscard]] bool moi_relocatable_host_preserves_transient_sgprs(
    const MoiDenseOwnerLivenessProof &owner_proof, BasicBlock &block,
    const Instruction &host_instruction, const RegisterSet &relocated_references,
    std::span<const MoiSgprRange> bootstrap_ranges) {
  // Every declared owner must contribute either complete liveness or a
  // complete proof that it never references the router ABI registers.
  if (!owner_proof.complete || owner_proof.contexts.empty() || bootstrap_ranges.empty())
    return false;

  const auto overlaps_range = [](const RegisterSet &set, uint16_t base, uint16_t count) {
    for (uint16_t index = 0; index < count; ++index) {
      if (set.contains({RegClass::SGPR, static_cast<uint16_t>(base + index), 1u}))
        return true;
    }
    return false;
  };
  const auto overlaps_window = [&](const RegisterSet &set) {
    return std::ranges::any_of(bootstrap_ranges, [&](const MoiSgprRange &range) {
      return overlaps_range(set, range.base, range.width);
    });
  };

  for (const MoiKernelResourceContext *context : owner_proof.contexts) {
    if (context == nullptr)
      return false;
    const bool reference_free = owner_proof.reference_free_contexts.contains(context);
    // A declared kernel range can contain an intentionally unreachable block
    // used as a relocation host. Such a block has no owner-local live-before
    // state. Admit it only when the complete reachable owner does not
    // reference the instrumentation scalar ranges; the relocated instructions
    // themselves are checked below.
    if (!context->block_set.contains(&block)) {
      if (!reference_free)
        return false;
      continue;
    }
    if (reference_free)
      continue;
    // Any owner that references the router ABI has complete liveness
    // materialized before the host scan. A filtered snapshot must never be
    // interpreted as proof that the scalar state is dead.
    if (context->liveness_is_restricted ||
        overlaps_window(context->liveness->live_before(host_instruction))) {
      return false;
    }
  }
  return !overlaps_window(relocated_references);
}

// A host discovered in existing text. Appended-body placement is deliberately
// absent because the search has not reserved any emitted bytes.
struct MoiDenseRelayHost {
  uint64_t host_offset = 0;
  std::vector<uint32_t> displaced_words;
};

// Entry-relay state retained after a caller reserves the appended host body.
struct MoiDenseEntryHost {
  uint64_t host_offset = 0;
  std::vector<uint32_t> displaced_words;
  uint64_t body_offset = 0;
};

struct MoiDenseRelayHostSearch {
  uint64_t owner_begin = 0;
  uint64_t owner_end = 0;
  size_t host_word_count = 0;
  std::span<const uint64_t> anchors;
  // Mutation-stage ownership is universal; caller predicates add only
  // engine-specific access, synchronization, and already-claimed ranges.
  std::span<const ConSanPreappliedReservedRange> preapplied_reserved_ranges;
  const MoiDenseOwnerLivenessProof &owner_proof;
  std::span<const MoiSgprRange> bootstrap_ranges;
};

template <typename OverlapsReserved>
[[nodiscard]] std::optional<MoiDenseRelayHost>
find_moi_dense_relay_host(const MoiResourcePlanningState &resource_state,
                          std::span<const uint8_t> text, const MoiDenseRelayHostSearch &search,
                          OverlapsReserved &&overlaps_reserved) {
  if (search.owner_begin >= search.owner_end || search.host_word_count == 0u ||
      search.anchors.empty())
    return std::nullopt;
  if (search.host_word_count > std::numeric_limits<uint64_t>::max() / sizeof(uint32_t))
    return std::nullopt;
  const uint64_t host_bytes = static_cast<uint64_t>(search.host_word_count) * sizeof(uint32_t);

  std::optional<MoiDenseRelayHost> best_host;
  uint64_t best_max_distance = std::numeric_limits<uint64_t>::max();
  auto block = std::lower_bound(resource_state.blocks.begin(), resource_state.blocks.end(),
                                search.owner_begin,
                                [](const std::unique_ptr<BasicBlock> &candidate, uint64_t entry) {
                                  return candidate->end_offset() <= entry;
                                });
  for (; block != resource_state.blocks.end() && (*block)->start_offset() < search.owner_end;
       ++block) {
    // BasicBlock::build receives every preapplied cave and continuation as a
    // leader. Requiring the complete host to remain in one owner-contained
    // block makes that CFG split an executable relocation boundary.
    if ((*block)->start_offset() < search.owner_begin ||
        (*block)->end_offset() > search.owner_end || (*block)->size() < host_bytes)
      continue;
    const auto block_end = (*block)->instructions().end();
    for (auto instruction_it = (*block)->instructions().begin(); instruction_it != block_end;
         ++instruction_it) {
      const Instruction &instruction = *instruction_it;
      const uint64_t host_offset = instruction.src_loc();
      const auto overlaps_preapplied = [&](const ConSanPreappliedReservedRange &reserved) {
        if (reserved.size == 0u ||
            reserved.text_offset > std::numeric_limits<uint64_t>::max() - reserved.size)
          return false;
        return host_offset < reserved.text_offset + reserved.size &&
               reserved.text_offset < host_offset + host_bytes;
      };
      if (host_offset > std::numeric_limits<uint64_t>::max() - host_bytes ||
          host_offset + host_bytes > (*block)->end_offset() ||
          std::ranges::any_of(search.preapplied_reserved_ranges, overlaps_preapplied) ||
          overlaps_reserved(host_offset, host_offset + host_bytes)) {
        continue;
      }

      RegisterSet relocated_references;
      bool has_indirect_scalar_reference = false;
      for (auto relocated = instruction_it;
           relocated != block_end && (*relocated).src_loc() < host_offset + host_bytes;
           ++relocated) {
        has_indirect_scalar_reference |= (*relocated).mnemonic().starts_with("s_movrel");
        const InstDefUse def_use(*relocated);
        relocated_references |= def_use.defs | def_use.uses;
      }
      if (has_indirect_scalar_reference || !moi_relocatable_host_preserves_transient_sgprs(
                                               search.owner_proof, **block, instruction,
                                               relocated_references, search.bootstrap_ranges)) {
        continue;
      }

      const uint64_t relay_offset = host_offset + sizeof(uint32_t);
      uint64_t max_distance = 0u;
      const bool reachable = std::ranges::all_of(search.anchors, [&](uint64_t anchor) {
        if (!compute_sopp_branch_simm16(anchor, relay_offset))
          return false;
        max_distance = std::max(max_distance, anchor < relay_offset ? relay_offset - anchor
                                                                    : anchor - relay_offset);
        return true;
      });
      if (!reachable || max_distance >= best_max_distance)
        continue;

      std::vector<uint32_t> displaced_words;
      std::vector<std::string> decode_errors;
      while (displaced_words.size() < search.host_word_count) {
        auto decoded = decode_relocatable_entry_instruction(
            text, host_offset + displaced_words.size() * sizeof(uint32_t), resource_state.arch,
            decode_errors, resource_state.decoder.get());
        if (!decoded) {
          displaced_words.clear();
          break;
        }
        displaced_words.insert(displaced_words.end(), decoded->begin(), decoded->end());
      }
      if (displaced_words.size() != search.host_word_count)
        continue;
      best_max_distance = max_distance;
      best_host = MoiDenseRelayHost{
          .host_offset = host_offset,
          .displaced_words = std::move(displaced_words),
      };
    }
  }
  return best_host;
}

[[nodiscard]] RegisterSet
dynamic_frame_save_registers_for_site(const MoiResourcePlanningState &state, uint64_t text_offset) {
  RegisterSet saved;
  constexpr uint64_t kMaxDynamicFramePrologueDistance = 16u * 1024u;
  CodeObjectPatcher patcher(state.code_object);
  const std::span<const uint8_t> text = patcher.text_bytes();
  if (text_offset < sizeof(uint32_t) || text_offset > text.size())
    return saved;
  const uint64_t begin = text_offset > kMaxDynamicFramePrologueDistance
                             ? text_offset - kMaxDynamicFramePrologueDistance
                             : 0u;
  for (uint64_t offset = text_offset - sizeof(uint32_t);; offset -= sizeof(uint32_t)) {
    if (offset + 2u * sizeof(uint32_t) <= text.size()) {
      uint32_t save_word = 0;
      uint32_t establish_word = 0;
      std::memcpy(&save_word, text.data() + offset, sizeof(save_word));
      std::memcpy(&establish_word, text.data() + offset + sizeof(uint32_t), sizeof(establish_word));
      if ((save_word & 0xff800000u) == 0xbe800000u && (save_word & 0xffu) == 33u &&
          establish_word == build_s_mov_b32(/*dst=*/33u, /*src=*/32u, state.arch)) {
        saved.expand({RegClass::SGPR, static_cast<uint16_t>((save_word >> 16u) & 0x7fu), 1u});
        break;
      }
    }
    if (offset < begin + sizeof(uint32_t))
      break;
  }
  return saved;
}

[[nodiscard]] ConSanCandidateResourcePlan
plan_moi_resource_site(MoiResourcePlanningState &state, const ConSanOptions &options,
                       ConSanResourceSiteKind site_kind, size_t candidate_index,
                       uint64_t text_offset, std::optional<uint64_t> kernel_descriptor_file_offset,
                       uint16_t scratch_count, const ConSanMoiCandidate *access_candidate = nullptr,
                       const ConSanAtomicSite *atomic_site = nullptr, bool require_spill = false) {
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
  plan.sgpr_reference_coverage_complete = true;
  bool every_owner_can_move_empty_cdna_accvgpr_boundary =
      state.arch == ROCJITSU_CODE_ARCH_CDNA3 || state.arch == ROCJITSU_CODE_ARCH_CDNA4;
  bool every_owner_cdna_accvgpr_boundary_is_allocation_tail =
      every_owner_can_move_empty_cdna_accvgpr_boundary;
  RegisterSet live_before_all_owners;
  bool descriptors_valid = true;
  for (MoiKernelResourceContext *owner : owners) {
    plan.current_vgpr_count = std::min(plan.current_vgpr_count, owner->current_vgpr_count);
    plan.ordinary_vgpr_limit = std::min(plan.ordinary_vgpr_limit, owner->ordinary_vgpr_limit);
    every_owner_cdna_accvgpr_boundary_is_allocation_tail &=
        owner->cdna_accvgpr_boundary_is_allocation_tail;
    every_owner_can_move_empty_cdna_accvgpr_boundary &= owner->can_move_empty_cdna_accvgpr_boundary;
    plan.max_referenced_vgpr_count =
        std::max(plan.max_referenced_vgpr_count, owner->max_referenced_vgpr_count);
    plan.current_sgpr_count = std::min(plan.current_sgpr_count, owner->current_sgpr_count);
    plan.max_referenced_sgpr_count =
        std::max(plan.max_referenced_sgpr_count, owner->max_referenced_sgpr_count);
    const consan_detail::ScalarOwnerContextSummary scalar_owner = {
        .descriptor_file_offset = owner->kernel->descriptor_file_offset,
        .current_sgpr_count = owner->current_sgpr_count,
        .max_referenced_sgpr_count = owner->max_referenced_sgpr_count,
        .has_indirect_sgpr_access = owner->has_indirect_sgpr_access,
        .sgpr_reference_coverage_complete = owner->sgpr_reference_coverage_complete,
        .descriptor_valid = owner->descriptor_valid,
    };
    plan.scalar_tail_floor =
        std::max(plan.scalar_tail_floor, consan_detail::scalar_owner_tail_floor(scalar_owner));
    // These two fields summarize diagnostics only. The actionable bound above
    // is computed per owner before aggregation.
    plan.has_indirect_sgpr_access |= owner->has_indirect_sgpr_access;
    plan.sgpr_reference_coverage_complete &= owner->sgpr_reference_coverage_complete;
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
  // address pair. CDNA3/4 require that pair to begin at an even VGPR.
  request.alignment =
      state.arch == ROCJITSU_CODE_ARCH_CDNA3 || state.arch == ROCJITSU_CODE_ARCH_CDNA4 ? 2u : 1u;
  request.current_allocation_count = std::min(plan.current_vgpr_count, plan.ordinary_vgpr_limit);
  request.max_referenced_count = plan.max_referenced_vgpr_count;
  request.architecture_limit =
      every_owner_cdna_accvgpr_boundary_is_allocation_tail ? kMaxVgprs : plan.ordinary_vgpr_limit;
  request.explicit_base = options.scratch_vgpr;
  request.force_spill = options.force_vgpr_spill || require_spill;
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
  if (options.moi_dispatch_id_vgpr)
    request.forbidden.expand({RegClass::VGPR, *options.moi_dispatch_id_vgpr, 2});
  for (const std::optional<uint16_t> reg :
       {options.moi_record_replay_workgroup_vgprs.x, options.moi_record_replay_workgroup_vgprs.y,
        options.moi_record_replay_workgroup_vgprs.z}) {
    if (reg)
      request.forbidden.expand({RegClass::VGPR, *reg, 1});
  }
  for (uint64_t descriptor_offset : plan.owner_descriptor_file_offsets) {
    const auto *assignment = find_moi_persistent_vgpr_assignment(options, descriptor_offset);
    if (!assignment)
      continue;
    request.forbidden.expand({RegClass::VGPR, assignment->owner_vgpr, 1});
    request.forbidden.expand({RegClass::VGPR, assignment->epoch_vgpr, 1});
    if (assignment->workgroup_key_vgpr)
      request.forbidden.expand({RegClass::VGPR, *assignment->workgroup_key_vgpr, 1});
    if (assignment->dispatch_id_vgpr)
      request.forbidden.expand({RegClass::VGPR, *assignment->dispatch_id_vgpr, 2});
    for (const std::optional<uint16_t> reg :
         {assignment->record_replay_workgroup_vgprs.x, assignment->record_replay_workgroup_vgprs.y,
          assignment->record_replay_workgroup_vgprs.z}) {
      if (reg)
        request.forbidden.expand({RegClass::VGPR, *reg, 1});
    }
  }

  ConSanRegisterPlan register_plan = plan_consan_registers(request, live_before_all_owners);
  if (register_plan.source == ConSanRegisterAllocationSource::Unsupported &&
      register_plan.reason == ConSanRegisterPlanReason::NoLegalWindow &&
      every_owner_can_move_empty_cdna_accvgpr_boundary && !request.explicit_base &&
      !request.force_spill) {
    // Trusted metadata can prove that a rounded unified-VGPR tail contains no
    // accumulator state even when ACCUM_OFFSET conservatively places the
    // ordinary-register boundary below it. Retry descriptor growth only after
    // ordinary placement fails. This keeps already-supported kernels on their
    // original allocation while avoiding hot-path private-memory spills for
    // the otherwise unplaceable sites.
    ConSanRegisterRequest growth_request = request;
    growth_request.architecture_limit = kMaxVgprs;
    growth_request.allow_spill = false;
    register_plan = plan_consan_registers(growth_request, live_before_all_owners);
    plan.alternatives.push_back({
        .kind = ConSanResourcePlanAlternativeKind::EmptyAccumulatorDescriptorGrowth,
        .source = register_plan.source,
        .reason = register_plan.reason,
        .scratch_vgpr_count = scratch_count,
        .outcome = register_plan.source == ConSanRegisterAllocationSource::DescriptorGrowth
                       ? ConSanResourcePlanAlternativeOutcome::Selected
                       : ConSanResourcePlanAlternativeOutcome::Rejected,
    });
  }
  const bool guest_replay_needs_disjoint_address_scratch =
      access_candidate != nullptr && state.arch == ROCJITSU_CODE_ARCH_GFX1250 &&
      two_address_native_lds_offset_scale(access_candidate->mnemonic).value_or(0u) > 8u;
  const bool supports_access_operand_overlap_spill =
      (options.moi_engine == ConSanMoiEngine::InlineShadow &&
       !guest_replay_needs_disjoint_address_scratch) ||
      ((is_rdna4_family_arch(state.arch) || instrumentation::is_cdna_family_arch(state.arch)) &&
       ((options.moi_engine == ConSanMoiEngine::RecordReplay &&
         !options.moi_dynamic_access_records && !guest_replay_needs_disjoint_address_scratch) ||
        (access_candidate != nullptr && !guest_replay_needs_disjoint_address_scratch &&
         (sampled_access_can_plan_spill_over_guest_operands(options, *access_candidate) ||
          sampled_access_supports_spill_backed_operand_recovery(options, *access_candidate,
                                                                state.arch)))));
  const bool supports_sampled_atomic_operand_overlap_spill =
      (is_rdna4_family_arch(state.arch) || instrumentation::is_cdna_family_arch(state.arch)) &&
      options.moi_engine == ConSanMoiEngine::Sampled && site_kind == ConSanResourceSiteKind::Atomic;
  if (register_plan.source == ConSanRegisterAllocationSource::Unsupported &&
      register_plan.reason == ConSanRegisterPlanReason::NoLegalWindow &&
      ((supports_access_operand_overlap_spill && site_kind == ConSanResourceSiteKind::Access) ||
       supports_sampled_atomic_operand_overlap_spill) &&
      !request.explicit_base) {
    // These appended-cave bodies preserve guest operands even when no
    // disjoint ordinary-VGPR window exists. Sampled atomics and the admitted
    // Sampled stores execute the guest before clobbering an overlapping saved
    // window; Record/Replay access bodies restore the window before the
    // displaced guest. Persistent state remains forbidden because it must
    // survive the complete kernel lifetime. Shared physical code can have
    // owners with different descriptor allocations, so this capability-gated
    // fallback may also grow the smaller descriptors to hold the spill window.
    ConSanRegisterRequest spill_request = request;
    spill_request.force_spill = true;
    spill_request.allow_spill_descriptor_growth = true;
    if (every_owner_can_move_empty_cdna_accvgpr_boundary)
      spill_request.architecture_limit = kMaxVgprs;
    spill_request.forbidden = {};
    if (options.moi_owner_vgpr)
      spill_request.forbidden.expand({RegClass::VGPR, *options.moi_owner_vgpr, 1});
    if (options.moi_epoch_vgpr)
      spill_request.forbidden.expand({RegClass::VGPR, *options.moi_epoch_vgpr, 1});
    if (options.moi_workgroup_key_vgpr)
      spill_request.forbidden.expand({RegClass::VGPR, *options.moi_workgroup_key_vgpr, 1});
    if (options.moi_dispatch_id_vgpr)
      spill_request.forbidden.expand({RegClass::VGPR, *options.moi_dispatch_id_vgpr, 2});
    for (const std::optional<uint16_t> reg :
         {options.moi_record_replay_workgroup_vgprs.x, options.moi_record_replay_workgroup_vgprs.y,
          options.moi_record_replay_workgroup_vgprs.z}) {
      if (reg)
        spill_request.forbidden.expand({RegClass::VGPR, *reg, 1});
    }
    for (uint64_t descriptor_offset : plan.owner_descriptor_file_offsets) {
      const auto *assignment = find_moi_persistent_vgpr_assignment(options, descriptor_offset);
      if (!assignment)
        continue;
      spill_request.forbidden.expand({RegClass::VGPR, assignment->owner_vgpr, 1});
      spill_request.forbidden.expand({RegClass::VGPR, assignment->epoch_vgpr, 1});
      if (assignment->workgroup_key_vgpr)
        spill_request.forbidden.expand({RegClass::VGPR, *assignment->workgroup_key_vgpr, 1});
      if (assignment->dispatch_id_vgpr)
        spill_request.forbidden.expand({RegClass::VGPR, *assignment->dispatch_id_vgpr, 2});
      for (const std::optional<uint16_t> reg : {assignment->record_replay_workgroup_vgprs.x,
                                                assignment->record_replay_workgroup_vgprs.y,
                                                assignment->record_replay_workgroup_vgprs.z}) {
        if (reg)
          spill_request.forbidden.expand({RegClass::VGPR, *reg, 1});
      }
    }
    if (options.moi_engine == ConSanMoiEngine::Sampled && access_candidate != nullptr &&
        sampled_access_supports_spill_backed_operand_recovery(options, *access_candidate,
                                                              state.arch) &&
        access_candidate->dst_vgpr) {
      spill_request.forbidden.expand(
          {RegClass::VGPR, *access_candidate->dst_vgpr,
           static_cast<uint8_t>(candidate_payload_vgpr_count(*access_candidate))});
    }
    register_plan = plan_consan_registers(spill_request, live_before_all_owners);
    plan.alternatives.push_back({
        .kind = ConSanResourcePlanAlternativeKind::GuestOperandOverlapSpill,
        .source = register_plan.source,
        .reason = register_plan.reason,
        .scratch_vgpr_count = scratch_count,
        .outcome = register_plan.source == ConSanRegisterAllocationSource::Unsupported
                       ? ConSanResourcePlanAlternativeOutcome::Rejected
                       : ConSanResourcePlanAlternativeOutcome::Selected,
    });
  }
  plan.source = register_plan.source;
  plan.reason = register_plan.reason;
  plan.scratch_vgpr = register_plan.base;
  plan.required_vgpr_count = register_plan.required_descriptor_count;
  if (plan.source == ConSanRegisterAllocationSource::SpillRequired) {
    const bool has_dynamic_owner =
        std::ranges::any_of(owners, [](const MoiKernelResourceContext *owner) {
          return owner->kernel->uses_dynamic_stack.value_or(false);
        });
    const bool every_owner_dynamic =
        std::ranges::all_of(owners, [](const MoiKernelResourceContext *owner) {
          return owner->kernel->uses_dynamic_stack.value_or(false);
        });
    const bool supported_dynamic_spill =
        moi_supports_dynamic_stack_spill(state.arch, options.moi_engine) && every_owner_dynamic;
    if (has_dynamic_owner && options.moi_engine != ConSanMoiEngine::InlineShadow &&
        !supported_dynamic_spill) {
      plan.source = ConSanRegisterAllocationSource::Unsupported;
      plan.reason = ConSanRegisterPlanReason::DynamicStack;
      plan.scratch_vgpr.reset();
      plan.required_vgpr_count = plan.current_vgpr_count;
    }
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
    ConSanCandidateResourcePlan plan = plan_moi_resource_site(
        state, options, ConSanResourceSiteKind::Access, candidate_index, candidate.text_offset,
        candidate.kernel_descriptor_file_offset, scratch_count, &candidate);
    bool initial_sampled_spill_overlaps_guest = false;
    if (plan.source == ConSanRegisterAllocationSource::SpillRequired && plan.scratch_vgpr) {
      std::vector<std::string> ignored_overlap_errors;
      initial_sampled_spill_overlaps_guest = reject_candidate_scratch_range_overlap(
          candidate, *plan.scratch_vgpr, plan.scratch_vgpr_count, ignored_overlap_errors);
    }
    if (sampled_access_supports_spill_backed_operand_recovery(options, candidate, state.arch) &&
        ((plan.source == ConSanRegisterAllocationSource::Unsupported &&
          plan.reason == ConSanRegisterPlanReason::NoLegalWindow) ||
         initial_sampled_spill_overlaps_guest)) {
      const uint16_t spill_scratch_count =
          sampled_spill_backed_scratch_count(options, candidate, state.arch);
      ConSanCandidateResourcePlan spill_plan = plan_moi_resource_site(
          state, options, ConSanResourceSiteKind::Access, candidate_index, candidate.text_offset,
          candidate.kernel_descriptor_file_offset, spill_scratch_count, &candidate,
          /*atomic_site=*/nullptr, /*require_spill=*/true);
      const bool selected = spill_plan.source == ConSanRegisterAllocationSource::SpillRequired;
      const ConSanResourcePlanAlternative alternative{
          .kind = ConSanResourcePlanAlternativeKind::SpillBackedOperandRecovery,
          .source = spill_plan.source,
          .reason = spill_plan.reason,
          .scratch_vgpr_count = spill_scratch_count,
          .outcome = selected ? ConSanResourcePlanAlternativeOutcome::Selected
                              : ConSanResourcePlanAlternativeOutcome::Rejected,
      };
      std::vector<ConSanResourcePlanAlternative> initial_alternatives =
          std::move(plan.alternatives);
      std::vector<ConSanResourcePlanAlternative> nested_alternatives =
          std::move(spill_plan.alternatives);
      if (selected) {
        for (ConSanResourcePlanAlternative &initial : initial_alternatives) {
          if (initial.outcome == ConSanResourcePlanAlternativeOutcome::Selected)
            initial.outcome = ConSanResourcePlanAlternativeOutcome::Superseded;
        }
        for (ConSanResourcePlanAlternative &nested : nested_alternatives) {
          if (nested.outcome == ConSanResourcePlanAlternativeOutcome::Selected)
            nested.outcome = ConSanResourcePlanAlternativeOutcome::Contributed;
        }
      } else {
        for (ConSanResourcePlanAlternative &nested : nested_alternatives) {
          if (nested.outcome == ConSanResourcePlanAlternativeOutcome::Selected)
            nested.outcome = ConSanResourcePlanAlternativeOutcome::Vetoed;
        }
      }
      std::vector<ConSanResourcePlanAlternative> alternatives;
      alternatives.reserve(initial_alternatives.size() + 1u + nested_alternatives.size());
      alternatives.insert(alternatives.end(), std::make_move_iterator(initial_alternatives.begin()),
                          std::make_move_iterator(initial_alternatives.end()));
      alternatives.push_back(alternative);
      alternatives.insert(alternatives.end(), std::make_move_iterator(nested_alternatives.begin()),
                          std::make_move_iterator(nested_alternatives.end()));
      if (selected)
        plan = std::move(spill_plan);
      plan.alternatives = std::move(alternatives);
    }
    if (options.moi_engine == ConSanMoiEngine::InlineShadow &&
        (state.arch == ROCJITSU_CODE_ARCH_CDNA3 || state.arch == ROCJITSU_CODE_ARCH_CDNA4) &&
        candidate.source == ConSanMoiCandidateSource::NativeLds &&
        moi_load_clobbers_address(candidate) &&
        !candidate_requires_gfx1250_flat_offset_materialization(candidate) &&
        (plan.source == ConSanRegisterAllocationSource::SpillRequired ||
         (plan.source == ConSanRegisterAllocationSource::Unsupported &&
          plan.reason == ConSanRegisterPlanReason::NoLegalWindow))) {
      const uint16_t spill_scratch_count =
          inline_shadow_spill_backed_scratch_count(options, candidate, state.arch);
      ConSanCandidateResourcePlan spill_plan = plan_moi_resource_site(
          state, options, ConSanResourceSiteKind::Access, candidate_index, candidate.text_offset,
          candidate.kernel_descriptor_file_offset, spill_scratch_count, &candidate,
          /*atomic_site=*/nullptr, /*require_spill=*/true);
      const bool selected = spill_plan.source == ConSanRegisterAllocationSource::SpillRequired;
      const ConSanResourcePlanAlternative alternative{
          .kind = ConSanResourcePlanAlternativeKind::SpillBackedOperandRecovery,
          .source = spill_plan.source,
          .reason = spill_plan.reason,
          .scratch_vgpr_count = spill_scratch_count,
          .outcome = selected ? ConSanResourcePlanAlternativeOutcome::Selected
                              : ConSanResourcePlanAlternativeOutcome::Rejected,
      };
      std::vector<ConSanResourcePlanAlternative> initial_alternatives =
          std::move(plan.alternatives);
      std::vector<ConSanResourcePlanAlternative> nested_alternatives =
          std::move(spill_plan.alternatives);
      if (selected) {
        for (ConSanResourcePlanAlternative &initial : initial_alternatives) {
          if (initial.outcome == ConSanResourcePlanAlternativeOutcome::Selected)
            initial.outcome = ConSanResourcePlanAlternativeOutcome::Superseded;
        }
        for (ConSanResourcePlanAlternative &nested : nested_alternatives) {
          if (nested.outcome == ConSanResourcePlanAlternativeOutcome::Selected)
            nested.outcome = ConSanResourcePlanAlternativeOutcome::Contributed;
        }
      } else {
        for (ConSanResourcePlanAlternative &nested : nested_alternatives) {
          if (nested.outcome == ConSanResourcePlanAlternativeOutcome::Selected)
            nested.outcome = ConSanResourcePlanAlternativeOutcome::Vetoed;
        }
      }

      // The attempt numbers form a stable planning trace: attempts retained
      // from the initial plan, the outer recovery, then its nested attempts.
      std::vector<ConSanResourcePlanAlternative> alternatives;
      alternatives.reserve(initial_alternatives.size() + 1u + nested_alternatives.size());
      alternatives.insert(alternatives.end(), std::make_move_iterator(initial_alternatives.begin()),
                          std::make_move_iterator(initial_alternatives.end()));
      alternatives.push_back(alternative);
      alternatives.insert(alternatives.end(), std::make_move_iterator(nested_alternatives.begin()),
                          std::make_move_iterator(nested_alternatives.end()));
      if (selected)
        plan = std::move(spill_plan);
      plan.alternatives = std::move(alternatives);
    }
    result.resource_plans.push_back(std::move(plan));
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
    if (options.moi_dispatch_id_vgpr) {
      resolved.required_vgpr_count = std::max<uint16_t>(
          resolved.required_vgpr_count, static_cast<uint16_t>(*options.moi_dispatch_id_vgpr + 2u));
    }
    for (const std::optional<uint16_t> reg :
         {options.moi_record_replay_workgroup_vgprs.x, options.moi_record_replay_workgroup_vgprs.y,
          options.moi_record_replay_workgroup_vgprs.z}) {
      if (reg) {
        resolved.required_vgpr_count =
            std::max<uint16_t>(resolved.required_vgpr_count, static_cast<uint16_t>(*reg + 1u));
      }
    }
    for (uint64_t descriptor_offset : plan.owner_descriptor_file_offsets) {
      const auto *assignment = find_moi_persistent_vgpr_assignment(options, descriptor_offset);
      if (!assignment)
        continue;
      uint16_t end = static_cast<uint16_t>(assignment->epoch_vgpr + 1u);
      if (assignment->workgroup_key_vgpr)
        end = std::max<uint16_t>(end, static_cast<uint16_t>(*assignment->workgroup_key_vgpr + 1u));
      if (assignment->dispatch_id_vgpr)
        end = std::max<uint16_t>(end, static_cast<uint16_t>(*assignment->dispatch_id_vgpr + 2u));
      for (const std::optional<uint16_t> reg : {assignment->record_replay_workgroup_vgprs.x,
                                                assignment->record_replay_workgroup_vgprs.y,
                                                assignment->record_replay_workgroup_vgprs.z}) {
        if (reg)
          end = std::max<uint16_t>(end, static_cast<uint16_t>(*reg + 1u));
      }
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

[[nodiscard]] constexpr std::optional<uint16_t>
moi_dynamic_stack_frame_save_sgpr_offset(ConSanMoiEngine engine) {
  switch (engine) {
  case ConSanMoiEngine::InlineShadow:
    return 24u;
  case ConSanMoiEngine::RecordReplay:
    return 5u;
  case ConSanMoiEngine::Sampled:
    return 8u;
  }
  return std::nullopt;
}

[[nodiscard]] constexpr std::optional<uint16_t>
moi_dynamic_stack_exec_save_sgpr_count(ConSanMoiEngine engine) {
  const auto frame_save_offset = moi_dynamic_stack_frame_save_sgpr_offset(engine);
  if (!frame_save_offset)
    return std::nullopt;
  // The displaced frame base is the final value in every dynamic-stack
  // scalar layout. Keep that contract explicit beside the offset table.
  return static_cast<uint16_t>(*frame_save_offset + 1u);
}

[[nodiscard]] uint16_t moi_exec_save_sgpr_count(const ConSanOptions &options, rj_code_arch_t arch) {
  if (options.moi_engine == ConSanMoiEngine::Sampled) {
    // Claim selection and optional wave sampling predicate on VCC. A winning
    // lane narrows EXEC while publishing, and every scalar predicate may
    // change SCC. The runtime fast gate and sampled-sync cave are mutually
    // exclusive bodies and share the same eight-register layout through +7.
    // gfx1250 dense access streams additionally reserve an aligned temporary
    // call-return pair through +7. The body returns with its ordinary
    // indirect-PC pair, so the probe may reuse that temporary after dispatch.
    if (!options.moi_report_buffer_address)
      return 0u;
    if (options.automatic_moi_record_replay_sgpr_spill)
      return 8u;
    if (options.moi_dynamic_stack_spill)
      return moi_dynamic_stack_exec_save_sgpr_count(options.moi_engine).value_or(0u);
    if (arch == ROCJITSU_CODE_ARCH_GFX1250)
      return 8u;
    return options.moi_track_atomics ? 8u : 7u;
  }
  if (options.moi_engine == ConSanMoiEngine::InlineShadow) {
    if (!options.moi_report_buffer_address)
      return 0;
    // Access probes reserve every transient mask plus the dedicated
    // application EXEC pair. Atomic-only objects use masks through +21 and do
    // not pay for access state they cannot execute. Dynamic-stack spills still
    // need the final frame-base save slot even in atomic-only objects.
    const uint16_t engine_count =
        options.moi_inline_access_present ? kConSanMoiInlineExecSaveSgprCount : 22u;
    const uint16_t dynamic_stack_count =
        options.moi_dynamic_stack_spill
            ? moi_dynamic_stack_exec_save_sgpr_count(options.moi_engine).value_or(0u)
            : 0u;
    return std::max(engine_count, dynamic_stack_count);
  }
  if (record_replay_uses_automatic_banked_capture(options)) {
    // Automatic access capture keeps the application EXEC pair through the
    // full distinct-address traversal, uses one scalar address key, and keeps
    // the selected group's EXEC in an aligned pair through +13 while one lane
    // performs publication.
    // Dense routing already occupies the call-return pair through +7, while
    // dynamic stack owners retain their frame-base snapshot at +5.
    return 14u;
  }
  constexpr uint16_t kRuntimeWorkgroupGateSgprCount = 7u;
  const uint16_t runtime_workgroup_gate_count =
      options.moi_engine == ConSanMoiEngine::RecordReplay &&
              options.moi_runtime_sample_stride > 1u &&
              !options.automatic_moi_record_replay_sgpr_spill
          ? kRuntimeWorkgroupGateSgprCount
          : 0u;
  if (options.moi_engine == ConSanMoiEngine::RecordReplay &&
      options.moi_record_replay_dense_barrier_router)
    return std::max<uint16_t>(8u, runtime_workgroup_gate_count);
  if (options.automatic_moi_record_replay_sgpr_spill) {
    // Sampled access transfer and dense routing use slots +0..+7. Compact
    // Record/Replay access, fence, and atomic bodies use only +0..+3.
    return options.moi_engine == ConSanMoiEngine::Sampled ? 8u : 4u;
  }
  if (options.moi_engine == ConSanMoiEngine::RecordReplay && options.moi_dynamic_stack_spill)
    // Preserve the compiler frame base in addition to EXEC, VCC, and SCC.
    // The dynamic spill setup reuses the ordinary SCC-save slot.
    return std::max(moi_dynamic_stack_exec_save_sgpr_count(options.moi_engine).value_or(0u),
                    runtime_workgroup_gate_count);
  // One EXEC-save pair, one VCC-save pair, and one SCC snapshot. Barrier-dense
  // RDNA-family objects additionally reserve a dispatch key and an aligned
  // call-return pair for the relocated router.
  return options.moi_report_buffer_address ? std::max<uint16_t>(5u, runtime_workgroup_gate_count)
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

struct MoiArchitecturalSgprRange {
  uint16_t base = kMaxSgprs;
  uint16_t width = 0u;
};

inline constexpr std::array kGfx1250ReservedOrdinarySgprRanges = {
    MoiArchitecturalSgprRange{.base = 102u, .width = 2u},
};

struct MoiArchitecturalSgprLayout {
  uint16_t ordinary_limit = kMaxSgprs;
  // Subranges of [0, ordinary_limit) that remain unsafe for persistent SALU
  // state even though they are encodable as ordinary scalar operands. Special
  // selectors at or above ordinary_limit are rejected by the limit itself.
  // Referenced range tables must have static storage duration.
  std::span<const MoiArchitecturalSgprRange> reserved_ordinary_ranges = {};
};

[[nodiscard]] constexpr MoiArchitecturalSgprLayout
moi_architectural_sgpr_layout(rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4) {
    return {
        .ordinary_limit = 102u,
    };
  }
  if (arch == ROCJITSU_CODE_ARCH_GFX1250) {
    // RDNA4 retains ordinary physical SGPRs through s105, but scalar
    // selectors s102:s103 alias FLAT_SCRATCH in persistent SALU state.
    return {
        .ordinary_limit = kMaxSgprs,
        .reserved_ordinary_ranges = kGfx1250ReservedOrdinarySgprRanges,
    };
  }
  return {};
}

[[nodiscard]] constexpr uint16_t moi_ordinary_sgpr_limit(rj_code_arch_t arch) {
  return moi_architectural_sgpr_layout(arch).ordinary_limit;
}

[[nodiscard]] constexpr bool moi_sgpr_ranges_overlap(uint16_t base, uint16_t width,
                                                     MoiArchitecturalSgprRange range) {
  return range.width != 0u &&
         static_cast<uint32_t>(base) < static_cast<uint32_t>(range.base) + range.width &&
         static_cast<uint32_t>(range.base) < static_cast<uint32_t>(base) + width;
}

[[nodiscard]] constexpr bool
persistent_sgpr_range_overlaps_reserved_ordinary_range(uint16_t base, uint16_t width,
                                                       rj_code_arch_t arch) {
  const MoiArchitecturalSgprLayout layout = moi_architectural_sgpr_layout(arch);
  return std::ranges::any_of(layout.reserved_ordinary_ranges, [=](MoiArchitecturalSgprRange range) {
    return moi_sgpr_ranges_overlap(base, width, range);
  });
}

static_assert(!persistent_sgpr_range_overlaps_reserved_ordinary_range(102u, 2u,
                                                                      ROCJITSU_CODE_ARCH_CDNA4));
static_assert(persistent_sgpr_range_overlaps_reserved_ordinary_range(102u, 2u,
                                                                     ROCJITSU_CODE_ARCH_GFX1250));
static_assert(!persistent_sgpr_range_overlaps_reserved_ordinary_range(104u, 2u,
                                                                      ROCJITSU_CODE_ARCH_GFX1250));
static_assert(!persistent_sgpr_range_overlaps_reserved_ordinary_range(106u, 2u,
                                                                      ROCJITSU_CODE_ARCH_GFX1250));
static_assert(moi_architectural_sgpr_layout(ROCJITSU_CODE_ARCH_CDNA3).ordinary_limit == 102u);
static_assert(moi_architectural_sgpr_layout(ROCJITSU_CODE_ARCH_CDNA4).ordinary_limit == 102u);
static_assert(moi_architectural_sgpr_layout(ROCJITSU_CODE_ARCH_GFX1250).ordinary_limit == 106u);

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
  // For Sampled this is entry-boundary SCC scratch that aliases the
  // publication EXEC-save pair. It is not the post-body guest SCC snapshot.
  uint16_t scc_save_sgpr = 0;
};

struct MoiSampledPublicationStateSgprs {
  uint16_t original_exec_save_sgpr = 0;
  uint16_t selection_vcc_save_sgpr = 0;
  uint16_t publication_exec_save_sgpr = 0;
  uint16_t guest_scc_snapshot_sgpr = 0;
};

[[nodiscard]] std::optional<MoiSampledPublicationStateSgprs>
moi_sampled_publication_state_sgprs(const ConSanOptions &options) {
  if (options.moi_engine != ConSanMoiEngine::Sampled || !options.moi_exec_save_sgpr)
    return std::nullopt;
  const uint16_t base = *options.moi_exec_save_sgpr;
  return MoiSampledPublicationStateSgprs{
      .original_exec_save_sgpr = base,
      .selection_vcc_save_sgpr = static_cast<uint16_t>(base + 2u),
      .publication_exec_save_sgpr = static_cast<uint16_t>(base + 4u),
      .guest_scc_snapshot_sgpr = static_cast<uint16_t>(base + 6u),
  };
}

[[nodiscard]] std::optional<MoiSpecialStateSgprs>
moi_special_state_sgprs(const ConSanOptions &options) {
  if (!options.moi_exec_save_sgpr)
    return std::nullopt;
  if (options.moi_engine == ConSanMoiEngine::Sampled) {
    const auto publication = moi_sampled_publication_state_sgprs(options);
    if (!publication)
      return std::nullopt;
    return MoiSpecialStateSgprs{
        .vcc_save_sgpr = publication->selection_vcc_save_sgpr,
        .scc_save_sgpr =
            options.automatic_moi_record_replay_sgpr_spill &&
                    options.moi_inline_indirect_scc_sgpr
                ? *options.moi_inline_indirect_scc_sgpr
                : publication->publication_exec_save_sgpr,
    };
  }
  const uint16_t vcc_offset = options.moi_engine == ConSanMoiEngine::InlineShadow ? 8u : 2u;
  return MoiSpecialStateSgprs{
      .vcc_save_sgpr = static_cast<uint16_t>(*options.moi_exec_save_sgpr + vcc_offset),
      .scc_save_sgpr =
          options.automatic_moi_record_replay_sgpr_spill && options.moi_inline_indirect_scc_sgpr
              ? *options.moi_inline_indirect_scc_sgpr
              : static_cast<uint16_t>(*options.moi_exec_save_sgpr + vcc_offset + 2u),
  };
}

/// Returns the SCC snapshot that remains valid after a sampled access body.
///
/// Fixed layouts return after the sampled body has overwritten the runtime
/// gate's residue field with the guest SCC snapshot. Spill-backed layouts
/// restore the complete transient window before returning, so their separately
/// allocated indirect-jump SCC register remains the stable snapshot.
[[nodiscard]] std::optional<uint16_t>
moi_sampled_access_return_scc_sgpr(const ConSanOptions &options) {
  if (options.moi_engine != ConSanMoiEngine::Sampled || !options.moi_exec_save_sgpr)
    return std::nullopt;
  if (options.automatic_moi_record_replay_sgpr_spill) {
    const auto indirect = moi_indirect_jump_sgprs(options);
    return indirect ? std::optional<uint16_t>(indirect->scc_save_sgpr) : std::nullopt;
  }
  const auto publication = moi_sampled_publication_state_sgprs(options);
  return publication ? std::optional<uint16_t>(publication->guest_scc_snapshot_sgpr) : std::nullopt;
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
  // Record/Replay's indirect island runs before the body snapshots
  // EXEC/VCC/SCC, and its return runs after those snapshots have been
  // restored, so the EXEC-save pair and SCC slot are free scratch at both
  // boundaries. The wider automatic-access window remains body-local.
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
                                                   const MoiResourcePlanningState &planning_state) {
  const std::span<const uint8_t> bytes = planning_state.bytes;
  const rj_code_arch_t arch = planning_state.arch;
  const bool component_local_sampled =
      options.moi_engine == ConSanMoiEngine::Sampled &&
      (is_rdna4_family_arch(arch) || arch == ROCJITSU_CODE_ARCH_CDNA3 ||
       arch == ROCJITSU_CODE_ARCH_CDNA4);
  if ((options.moi_engine != ConSanMoiEngine::RecordReplay &&
       options.moi_engine != ConSanMoiEngine::InlineShadow && !component_local_sampled) ||
      !options.automatic_moi_partial_exec_save_sgprs || !options.moi_exec_save_sgpr)
    return false;

  MoiResourcePlanningState state(bytes, arch, options, result);
  if (!state.valid)
    return false;

  std::unordered_map<uint64_t, uint64_t> parent;
  const auto ensure_owner = [&](uint64_t descriptor_offset) {
    parent.try_emplace(descriptor_offset, descriptor_offset);
  };
  for (const ConSanCandidateResourcePlan &plan : result.resource_plans) {
    if ((arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4) &&
        plan.source == ConSanRegisterAllocationSource::Unsupported)
      continue;
    for (uint64_t descriptor_offset : plan.owner_descriptor_file_offsets)
      ensure_owner(descriptor_offset);
  }
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
    if (((arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4) &&
         plan.source == ConSanRegisterAllocationSource::Unsupported) ||
        plan.owner_descriptor_file_offsets.empty())
      continue;
    const uint64_t first = plan.owner_descriptor_file_offsets.front();
    for (uint64_t owner : plan.owner_descriptor_file_offsets) {
      const uint64_t lhs = find_root(first);
      const uint64_t rhs = find_root(owner);
      if (lhs != rhs)
        parent[rhs] = lhs;
    }
  }

  std::unordered_map<BasicBlock *, std::vector<const MoiKernelResourceContext *>> contexts_by_block;
  for (const MoiKernelResourceContext &context : state.contexts) {
    if (context.kernel == nullptr)
      continue;
    for (BasicBlock *block : context.scope.blocks) {
      if (block != nullptr)
        contexts_by_block[block].push_back(&context);
    }
  }

  std::unordered_map<uint64_t, RegisterSet> unavailable_by_owner;
  std::unordered_set<uint64_t> included_site_offsets;
  included_site_offsets.reserve(result.moi_candidates.size() + result.resource_plans.size() +
                                result.site_dispositions.size() + result.kernels.size());
  const auto include_site = [&](uint64_t text_offset) {
    // Site dispositions describe logical cells, so one physical instruction can
    // appear many times.  Liveness and def/use are set unions and need only be
    // incorporated once per instruction.
    if (!included_site_offsets.insert(text_offset).second)
      return;
    BasicBlock *block = block_for_offset(state.block_index, text_offset);
    const Instruction *instruction = instruction_at(block, text_offset);
    if (block == nullptr || instruction == nullptr)
      return;
    const auto owners = contexts_by_block.find(block);
    if (owners == contexts_by_block.end())
      return;
    for (const MoiKernelResourceContext *context : owners->second) {
      RegisterSet &unavailable = unavailable_by_owner[context->kernel->descriptor_file_offset];
      unavailable |= context->liveness->live_before(*instruction);
      const InstDefUse def_use(*instruction);
      unavailable |= def_use.defs | def_use.uses;
      // Dynamic-stack helpers save their caller's frame base before establishing
      // a callee frame.  Recovered CFGs do not always connect a helper return to
      // its entry, so ordinary liveness can miss that long-lived scalar.  A
      // transient instrumentation window must never overwrite it.
      if (context->kernel->uses_dynamic_stack.value_or(false))
        unavailable |= dynamic_frame_save_registers_for_site(state, text_offset);
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
  // Inline's component-local transient window is established only inside an
  // access or synchronization body. Its RDNA entry relay and return use
  // special state, while persistent owner/epoch values live in VGPRs. Do not
  // turn entry ABI inputs into false live-through constraints for that
  // site-local window.
  if (options.moi_init_owner_epoch && options.moi_engine != ConSanMoiEngine::InlineShadow) {
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
  const auto intersects = [](const RegisterSet &set, uint16_t base, uint16_t width) {
    for (uint16_t i = 0; i < width; ++i)
      if (set.contains({RegClass::SGPR, static_cast<uint16_t>(base + i), 1u}))
        return true;
    return false;
  };

  bool assigned_any = false;
  bool missed_required_assignment = false;
  for (auto &[root, owners] : owners_by_component) {
    const auto component_contexts = resolve_moi_scalar_owner_contexts(state, owners);
    if (!component_contexts) {
      missed_required_assignment = true;
      continue;
    }
    const uint32_t liveness_search_floor =
        std::ranges::any_of(component_contexts->summaries,
                            [](const auto &context) {
                              return consan_detail::scalar_owner_tail_floor(context) >
                                     context.max_referenced_sgpr_count;
                            })
            ? component_contexts->tail_floor
            : 0u;
    const bool needs_override = !moi_scalar_contexts_admit_reserved_window(
        component_contexts->summaries, *options.moi_exec_save_sgpr, count, arch);
    if (!needs_override)
      continue;
    const RegisterSet &unavailable = unavailable_by_component[root];
    std::optional<uint16_t> selected;
    std::optional<uint16_t> spill_pc;
    std::optional<uint16_t> spill_call_return;
    std::optional<uint16_t> spill_scc;
    std::optional<uint16_t> spill_key;
    std::optional<uint16_t> spill_visible;
    bool branch_only_scalar_spill = false;
    std::optional<uint16_t> dynamic_stack_borrowed_sgpr;
    if (options.moi_engine == ConSanMoiEngine::RecordReplay || component_local_sampled) {
      for (uint32_t base = util::align_up(liveness_search_floor, 2u); base + count <= limit;
           base += 2u) {
        const uint16_t candidate = static_cast<uint16_t>(base);
        if (intersects(unavailable, candidate, count) ||
            (options.moi_owner_sgpr &&
             range_overlaps(candidate, count, *options.moi_owner_sgpr, 1u)) ||
            (options.moi_dispatch_id_sgpr &&
             range_overlaps(candidate, count, *options.moi_dispatch_id_sgpr,
                            moi_dispatch_state_sgpr_count(options))) ||
            (options.moi_dynamic_stack_spill &&
             moi_sgpr_range_overlaps_dynamic_stack_state(candidate, count)))
          continue;
        if (moi_scalar_contexts_conflict_with_physical_vcc(component_contexts->summaries, candidate,
                                                           count, arch))
          continue;
        selected = candidate;
        break;
      }
    }
    if (!selected && options.moi_engine == ConSanMoiEngine::InlineShadow &&
        (is_rdna4_family_arch(arch) || arch == ROCJITSU_CODE_ARCH_CDNA3 ||
         arch == ROCJITSU_CODE_ARCH_CDNA4) &&
        address_free_scratch_private_limit(arch)) {
      const auto reserved = [&](uint16_t base, uint16_t width) {
        return (options.moi_owner_sgpr &&
                range_overlaps(base, width, *options.moi_owner_sgpr, 1u)) ||
               (options.moi_dispatch_id_sgpr &&
                range_overlaps(base, width, *options.moi_dispatch_id_sgpr,
                               moi_dispatch_state_sgpr_count(options))) ||
               (options.moi_dynamic_stack_spill &&
                moi_sgpr_range_overlaps_dynamic_stack_state(base, width));
      };
      uint32_t initialized_end = 0u;
      for (uint64_t owner : owners) {
        if (owner > bytes.size() || sizeof(KD) > bytes.size() - owner)
          continue;
        KD descriptor{};
        std::memcpy(&descriptor, bytes.data() + owner, sizeof(descriptor));
        initialized_end =
            std::max<uint32_t>(initialized_end, moi_descriptor_user_sgpr_count(descriptor) +
                                                    moi_descriptor_system_sgpr_count(descriptor));
      }
      const auto architecturally_reserved = [&](uint16_t base, uint16_t width) {
        return reserved(base, width) ||
               ((arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4) &&
                range_overlaps(base, width, /*initialized SGPR base=*/0u, initialized_end));
      };
      const auto conflicts_with_cdna_vcc = [&](std::span<const MoiSgprRange> ranges) {
        return moi_scalar_contexts_conflict_with_physical_vcc(component_contexts->summaries, ranges,
                                                              arch);
      };
      const uint32_t fresh_begin =
          arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4
              ? std::max(initialized_end, component_contexts->tail_floor)
              : component_contexts->tail_floor;
      // The Inline scalar window is saved after the VGPR frame is established
      // and restored before that frame is released. RDNA4 may therefore
      // borrow initialized guest SGPRs; only the live-through router state
      // must remain liveness-dead. CDNA retains its stricter preload boundary.
      const uint32_t scalar_spill_begin =
          is_rdna4_family_arch(arch) ? 0u : std::max(initialized_end, liveness_search_floor);
      const uint32_t router_begin =
          is_rdna4_family_arch(arch) ? liveness_search_floor : fresh_begin;
      for (uint32_t base = util::align_up(scalar_spill_begin, 2u);
           !selected && base + count <= limit; base += 2u) {
        const uint16_t spill_base = static_cast<uint16_t>(base);
        std::vector<MoiSgprRange> selected_ranges{{spill_base, count}};
        if (architecturally_reserved(spill_base, count) || conflicts_with_cdna_vcc(selected_ranges))
          continue;
        const auto unavailable_or_reserved = [&](uint16_t candidate, uint16_t width) {
          return range_overlaps(candidate, width, spill_base, count) ||
                 architecturally_reserved(candidate, width) ||
                 intersects(unavailable, candidate, width);
        };
        const auto can_add_range = [&](uint16_t candidate, uint16_t width) {
          std::vector<MoiSgprRange> proposed = selected_ranges;
          proposed.push_back({candidate, width});
          return !conflicts_with_cdna_vcc(proposed);
        };
        for (uint32_t reg = fresh_begin; reg < limit; ++reg) {
          const uint16_t candidate = static_cast<uint16_t>(reg);
          if (!unavailable_or_reserved(candidate, 1u) && can_add_range(candidate, 1u)) {
            spill_visible = candidate;
            selected_ranges.push_back({candidate, 1u});
            break;
          }
        }
        for (uint32_t pc = util::align_up(liveness_search_floor, 2u); pc + 2u <= limit; pc += 2u) {
          const uint16_t candidate = static_cast<uint16_t>(pc);
          if (!unavailable_or_reserved(candidate, 2u) &&
              (!spill_visible || !range_overlaps(candidate, 2u, *spill_visible, 1u)) &&
              can_add_range(candidate, 2u)) {
            spill_pc = candidate;
            selected_ranges.push_back({candidate, 2u});
            break;
          }
        }
        for (uint32_t reg = liveness_search_floor; spill_pc && reg < limit; ++reg) {
          const uint16_t candidate = static_cast<uint16_t>(reg);
          if (!unavailable_or_reserved(candidate, 1u) &&
              (!spill_visible || candidate != *spill_visible) &&
              !range_overlaps(candidate, 1u, *spill_pc, 2u) && can_add_range(candidate, 1u)) {
            spill_scc = candidate;
            selected_ranges.push_back({candidate, 1u});
            break;
          }
        }
        // CDNA access and barrier sites retain independent local entry
        // islands. They need only the dead indirect-PC pair and SCC slot; if
        // no fresh cross-site latch exists, use the external exact-shadow
        // table and let the unused dense-router fields alias that bootstrap
        // state.
        if (!spill_visible && spill_pc && spill_scc &&
            (arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4)) {
          spill_key = spill_scc;
          spill_call_return = spill_pc;
          selected = spill_base;
          break;
        }
        for (uint32_t reg = router_begin; spill_scc && reg < limit; ++reg) {
          const uint16_t candidate = static_cast<uint16_t>(reg);
          if (!unavailable_or_reserved(candidate, 1u) &&
              (!spill_visible || candidate != *spill_visible) &&
              !range_overlaps(candidate, 1u, *spill_pc, 2u) && candidate != *spill_scc &&
              can_add_range(candidate, 1u)) {
            spill_key = candidate;
            selected_ranges.push_back({candidate, 1u});
            break;
          }
        }
        for (uint32_t call = util::align_up(router_begin, 2u); spill_key && call + 2u <= limit;
             call += 2u) {
          const uint16_t candidate = static_cast<uint16_t>(call);
          if (!unavailable_or_reserved(candidate, 2u) &&
              (!spill_visible || !range_overlaps(candidate, 2u, *spill_visible, 1u)) &&
              !range_overlaps(candidate, 2u, *spill_pc, 2u) &&
              !range_overlaps(candidate, 2u, *spill_scc, 1u) &&
              !range_overlaps(candidate, 2u, *spill_key, 1u) && can_add_range(candidate, 2u)) {
            spill_call_return = candidate;
            selected_ranges.push_back({candidate, 2u});
            break;
          }
        }
        if (spill_call_return)
          selected = spill_base;
        else
          spill_visible = spill_pc = spill_scc = spill_key = std::nullopt;
      }
      // A full-pressure owner may have no scalar register that remains dead
      // from an access anchor through an appended router. Route with
      // branch-only hops instead and preserve the complete Inline window
      // before using it. Prefer this path to a site-local dead window because
      // the latter cannot retain the entry-to-probe visible-evidence latch and
      // therefore cannot preserve local LDS exact-shadow eligibility.
      // Dynamic-stack owners additionally need one ordinary bootstrap pair
      // while establishing their private frame. Every choice stays inside
      // every owner's existing allocation and is derived from
      // descriptor/runtime ABI state rather than a fixed range.
      const auto owner_uses_dynamic_stack = [&](uint64_t descriptor_offset) -> std::optional<bool> {
        const auto kernel = std::ranges::find(result.kernels, descriptor_offset,
                                              &ConSanKernelInfo::descriptor_file_offset);
        if (kernel == result.kernels.end())
          return std::nullopt;
        if (kernel->uses_dynamic_stack)
          return kernel->uses_dynamic_stack;
        if (descriptor_offset > state.bytes.size() ||
            sizeof(KD) > state.bytes.size() - descriptor_offset) {
          return std::nullopt;
        }
        KD descriptor{};
        std::memcpy(&descriptor, state.bytes.data() + descriptor_offset, sizeof(descriptor));
        // A runtime-sized frame requires the private-segment ABI. Its absence
        // is an architectural fixed-stack proof even when older metadata omits
        // the optional stack marker. A scratch-enabled owner without a marker
        // remains unknown and must fail closed.
        if (AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2,
                            kd::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT) == 0u) {
          return false;
        }
        return std::nullopt;
      };
      const bool all_owners_use_dynamic_stack =
          is_rdna4_family_arch(arch) && !owners.empty() &&
          std::ranges::all_of(owners, [&](uint64_t descriptor_offset) {
            return owner_uses_dynamic_stack(descriptor_offset).value_or(false);
          });
      const bool all_owners_use_fixed_stack =
          is_rdna4_family_arch(arch) && !owners.empty() &&
          std::ranges::all_of(owners, [&](uint64_t descriptor_offset) {
            const std::optional<bool> uses_dynamic_stack =
                owner_uses_dynamic_stack(descriptor_offset);
            return uses_dynamic_stack.has_value() && !*uses_dynamic_stack;
          });
      if (!selected && (all_owners_use_dynamic_stack || all_owners_use_fixed_stack)) {
        uint16_t allocated_limit = limit;
        for (const auto &summary : component_contexts->summaries)
          allocated_limit = std::min(allocated_limit, summary.current_sgpr_count);

        const auto overlaps_optional = [](uint16_t base, uint16_t width,
                                          std::optional<uint16_t> reg, uint16_t reg_width = 1u) {
          return reg && range_overlaps(base, width, *reg, reg_width);
        };
        const auto overlaps_instrumentation_state = [&](uint16_t base, uint16_t width) {
          bool overlaps =
              architecturally_reserved(base, width) ||
              moi_sgpr_range_overlaps_dynamic_stack_state(base, width) ||
              persistent_sgpr_range_overlaps_reserved_ordinary_range(base, width, arch) ||
              overlaps_optional(base, width, options.moi_persistent_owner_sgpr) ||
              overlaps_optional(base, width, options.moi_persistent_epoch_sgpr) ||
              overlaps_optional(base, width, options.moi_persistent_workgroup_key_sgpr) ||
              overlaps_optional(base, width, options.moi_dispatch_id_sgpr,
                                moi_dispatch_state_sgpr_count(options)) ||
              overlaps_optional(base, width, options.moi_inline_visible_evidence_sgpr);
          for (const std::optional<uint16_t> reg : {options.moi_record_replay_workgroup_sgprs.x,
                                                    options.moi_record_replay_workgroup_sgprs.y,
                                                    options.moi_record_replay_workgroup_sgprs.z}) {
            overlaps |= overlaps_optional(base, width, reg);
          }
          if (options.delay_mode == ConSanDelayMode::SleepVar)
            overlaps |= range_overlaps(base, width, options.delay_var_ssrc, 1u);
          return overlaps;
        };

        for (uint32_t base = 0; !selected && base + count <= allocated_limit; base += 2u) {
          const uint16_t candidate = static_cast<uint16_t>(base);
          const std::array scalar_window{MoiSgprRange{candidate, count}};
          if (overlaps_instrumentation_state(candidate, count) ||
              conflicts_with_cdna_vcc(scalar_window)) {
            continue;
          }
          if (all_owners_use_fixed_stack) {
            selected = candidate;
            branch_only_scalar_spill = true;
            break;
          }
          for (uint32_t pair = 0; pair + 2u <= allocated_limit; pair += 2u) {
            const uint16_t borrowed = static_cast<uint16_t>(pair);
            const std::array combined_ranges{
                MoiSgprRange{candidate, count},
                MoiSgprRange{borrowed, 2u},
            };
            if (range_overlaps(borrowed, 2u, candidate, count) ||
                overlaps_instrumentation_state(borrowed, 2u) ||
                conflicts_with_cdna_vcc(combined_ranges)) {
              continue;
            }
            selected = candidate;
            branch_only_scalar_spill = true;
            dynamic_stack_borrowed_sgpr = borrowed;
            break;
          }
        }
      }
      // If no complete spill router exists, an owner-local window that is
      // dead at every instrumentation site remains useful for external-shadow
      // probes.  It cannot carry the entry-to-probe visible-evidence latch;
      // access selection therefore keeps such owners off the local LDS mirror.
      if (!selected) {
        for (uint32_t base = util::align_up(std::max(initialized_end, liveness_search_floor), 2u);
             base + count <= limit; base += 2u) {
          const uint16_t candidate = static_cast<uint16_t>(base);
          const std::array ranges{MoiSgprRange{candidate, count}};
          if (!architecturally_reserved(candidate, count) &&
              !intersects(unavailable, candidate, count) && !conflicts_with_cdna_vcc(ranges)) {
            selected = candidate;
            break;
          }
        }
      }
    }
    const bool supports_compact_scalar_spill =
        (options.moi_engine == ConSanMoiEngine::RecordReplay || component_local_sampled) &&
        (is_rdna4_family_arch(arch) || arch == ROCJITSU_CODE_ARCH_CDNA3 ||
         arch == ROCJITSU_CODE_ARCH_CDNA4);
    if (!selected && supports_compact_scalar_spill && address_free_scratch_private_limit(arch)) {
      const auto conflicts_with_cdna_vcc = [&](uint16_t base, uint16_t width) {
        return moi_scalar_contexts_conflict_with_physical_vcc(component_contexts->summaries, base,
                                                              width, arch);
      };
      const auto reserved = [&](uint16_t base, uint16_t width) {
        return (options.moi_owner_sgpr &&
                range_overlaps(base, width, *options.moi_owner_sgpr, 1u)) ||
               (options.moi_dispatch_id_sgpr &&
                range_overlaps(base, width, *options.moi_dispatch_id_sgpr,
                               moi_dispatch_state_sgpr_count(options))) ||
               (options.moi_dynamic_stack_spill &&
                moi_sgpr_range_overlaps_dynamic_stack_state(base, width)) ||
               conflicts_with_cdna_vcc(base, width);
      };
      // This path preserves every borrowed guest SGPR in address-free private
      // storage before using it. Unlike a merely liveness-dead window, it is
      // safe below a scalar-relative owner's allocation tail.
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
      const uint16_t spilled_scalar_count = component_local_sampled ? 8u : count;
      for (uint32_t base = 0; spill_call_return && base + spilled_scalar_count <= limit;
           base += 2u) {
        const uint16_t candidate = static_cast<uint16_t>(base);
        if (!reserved(candidate, spilled_scalar_count) &&
            !range_overlaps(candidate, spilled_scalar_count, *spill_pc, 2u) &&
            !range_overlaps(candidate, spilled_scalar_count, *spill_scc, 1u) &&
            !range_overlaps(candidate, spilled_scalar_count, *spill_key, 1u) &&
            !range_overlaps(candidate, spilled_scalar_count, *spill_call_return, 2u)) {
          selected = candidate;
          break;
        }
      }
    }
    if (!selected) {
      missed_required_assignment = true;
      continue;
    }
    const bool spill_backed = (spill_pc && spill_scc && spill_key && spill_call_return) ||
                              branch_only_scalar_spill || dynamic_stack_borrowed_sgpr.has_value();
    const std::optional<uint16_t> dispatch_id_sgpr =
        options.moi_dispatch_id_sgpr &&
                moi_scalar_contexts_admit_reserved_window(
                    component_contexts->summaries, *options.moi_dispatch_id_sgpr,
                    moi_dispatch_state_sgpr_count(options), arch)
            ? options.moi_dispatch_id_sgpr
            : std::nullopt;
    for (uint64_t owner : owners) {
      options.moi_transient_sgpr_assignments.push_back(
          {.descriptor_file_offset = owner,
           .exec_save_sgpr = *selected,
           .owner_sgpr = moi_inline_uses_planned_owner_sgpr(options)
                             ? std::optional<uint16_t>(*selected)
                             : options.moi_owner_sgpr,
           .dispatch_id_sgpr = dispatch_id_sgpr,
           .spill_backed = spill_backed,
           .indirect_pc_sgpr = spill_pc,
           .indirect_scc_sgpr = spill_scc,
           .dispatch_key_sgpr = spill_key,
           .call_return_sgpr = spill_call_return,
           .visible_evidence_sgpr = spill_visible,
           .branch_only_scalar_spill = branch_only_scalar_spill,
           .dynamic_stack_borrowed_sgpr = dynamic_stack_borrowed_sgpr});
    }
    assigned_any = true;
  }
  const bool can_exclude_incompatible_component =
      arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4;
  if (missed_required_assignment && !can_exclude_incompatible_component) {
    options.moi_transient_sgpr_assignments.clear();
    result.resolved_moi_transient_sgpr_assignments.clear();
    return false;
  }
  if (!assigned_any)
    return false;
  result.resolved_moi_transient_sgpr_assignments = options.moi_transient_sgpr_assignments;
  result.warnings.emplace_back(
      "ConSan MOI assigned owner-local transient SGPR windows to independent owner components");
  return true;
}

[[nodiscard]] bool
configure_automatic_moi_exec_save_sgprs(ConSanOptions &options, ConSanResult &result,
                                        const MoiResourcePlanningState &planning_state) {
  const std::span<const uint8_t> bytes = planning_state.bytes;
  const rj_code_arch_t arch = planning_state.arch;
  if (options.moi_exec_save_sgpr)
    return false;
  const uint16_t count = moi_exec_save_sgpr_count(options, arch);
  if (count == 0)
    return false;

  const auto plan_contexts =
      resolve_moi_resource_plan_scalar_owner_contexts(planning_state, result);
  if (!plan_contexts.planning_state_valid || plan_contexts.unresolved_owner) {
    std::string warning = "ConSan MOI could not resolve an owner context for automatic EXEC-save "
                          "SGPR placement";
    if (plan_contexts.unresolved_owner)
      warning += " at descriptor file offset " + std::to_string(*plan_contexts.unresolved_owner);
    result.warnings.push_back(std::move(warning));
    return false;
  }
  const bool include_unsupported =
      arch != ROCJITSU_CODE_ARCH_CDNA3 && arch != ROCJITSU_CODE_ARCH_CDNA4;
  const MoiScalarOwnerContextResolution scalar_contexts =
      collect_moi_scalar_owner_contexts(plan_contexts.groups, include_unsupported);
  if (scalar_contexts.contexts.empty()) {
    result.warnings.emplace_back(
        "ConSan MOI could not derive automatic EXEC-save SGPRs without an owned candidate "
        "scope");
    return false;
  }

  uint32_t first = util::align_up(scalar_contexts.tail_floor, 2u);
  const uint16_t ordinary_sgpr_limit = moi_ordinary_sgpr_limit(arch);
  const auto conflicts_with_cdna_vcc = [&](uint16_t base) {
    return moi_scalar_contexts_conflict_with_physical_vcc(scalar_contexts.summaries, base, count,
                                                          arch);
  };
  std::vector<uint16_t> fresh_candidates;
  for (uint32_t base = first; base + count <= ordinary_sgpr_limit; base += 2u)
    fresh_candidates.push_back(static_cast<uint16_t>(base));
  const bool scalar_tail_covers_implicit_references =
      std::ranges::any_of(scalar_contexts.summaries, [](const auto &context) {
        return consan_detail::scalar_owner_tail_floor(context) > context.max_referenced_sgpr_count;
      });
  // Scalar-relative and open-CFG owners may still need a smaller persistent
  // tuple above the same tail. Prefer the high end for the wide transient
  // window so lower legal holes remain available to that later placement.
  if (scalar_tail_covers_implicit_references)
    std::ranges::reverse(fresh_candidates);
  for (uint16_t base : fresh_candidates) {
    // Dynamic-stack kernels use s32 as the stack top and s33 as the current
    // frame base even when those architectural roles are absent from ordinary
    // decoded operand liveness. Never assign either register to persistent
    // Inline scalar state.
    if (conflicts_with_cdna_vcc(base) ||
        (options.moi_dynamic_stack_spill &&
         moi_sgpr_range_overlaps_dynamic_stack_state(base, count))) {
      continue;
    }
    if (options.moi_owner_sgpr && range_overlaps(base, count, *options.moi_owner_sgpr, 1u)) {
      continue;
    }
    if (options.moi_dispatch_id_sgpr && range_overlaps(base, count, *options.moi_dispatch_id_sgpr,
                                                       moi_dispatch_state_sgpr_count(options))) {
      continue;
    }
    options.moi_exec_save_sgpr = base;
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
  const bool has_dynamic_stack_owner =
      std::ranges::any_of(result.kernels, [](const ConSanKernelInfo &kernel) {
        return kernel.uses_dynamic_stack.value_or(false);
      });
  const auto include_site = [&](uint64_t text_offset) {
    BasicBlock *block = block_for_offset(state.block_index, text_offset);
    const Instruction *instruction = instruction_at(block, text_offset);
    if (block == nullptr || instruction == nullptr)
      return;
    if (has_dynamic_stack_owner) {
      const RegisterSet frame_saves = dynamic_frame_save_registers_for_site(state, text_offset);
      unavailable |= frame_saves;
    }
    const auto owners = contexts_by_block.find(block);
    if (owners == contexts_by_block.end())
      return;
    for (const MoiKernelResourceContext *context : owners->second) {
      unavailable |= context->liveness->live_before(*instruction);
      const InstDefUse def_use(*instruction);
      unavailable |= def_use.defs | def_use.uses;
      if (context->kernel != nullptr && context->kernel->uses_dynamic_stack.value_or(false))
        unavailable |= dynamic_frame_save_registers_for_site(state, text_offset);
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
  for (const ConSanCandidateResourcePlan &plan : result.resource_plans) {
    if (!plan.owner_descriptor_file_offsets.empty())
      include_site(plan.text_offset);
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
    const uint32_t liveness_search_floor = scalar_tail_covers_implicit_references ? first : 0u;
    for (uint32_t base = liveness_search_floor; base + count <= ordinary_sgpr_limit; base += 2u) {
      if (unavailable_intersects(static_cast<uint16_t>(base)) ||
          conflicts_with_cdna_vcc(static_cast<uint16_t>(base)) ||
          (options.moi_dynamic_stack_spill &&
           moi_sgpr_range_overlaps_dynamic_stack_state(static_cast<uint16_t>(base), count)) ||
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
  const bool supports_partial_owner_exec =
      (is_rdna4_family_arch(arch) || instrumentation::is_cdna_family_arch(arch)) &&
      (options.moi_engine == ConSanMoiEngine::RecordReplay ||
       options.moi_engine == ConSanMoiEngine::InlineShadow ||
       options.moi_engine == ConSanMoiEngine::Sampled);
  if (supports_partial_owner_exec) {
    const int32_t last_base = static_cast<int32_t>((ordinary_sgpr_limit - count) & ~1u);
    for (int32_t base = last_base; base >= 0; base -= 2) {
      const uint16_t candidate = static_cast<uint16_t>(base);
      if ((options.moi_owner_sgpr &&
           range_overlaps(candidate, count, *options.moi_owner_sgpr, 1u)) ||
          (options.moi_dispatch_id_sgpr &&
           range_overlaps(candidate, count, *options.moi_dispatch_id_sgpr,
                          moi_dispatch_state_sgpr_count(options))) ||
          (options.moi_dynamic_stack_spill &&
           moi_sgpr_range_overlaps_dynamic_stack_state(candidate, count))) {
        continue;
      }
      const auto group_admits_window = [&](const MoiScalarOwnerContextGroup &group) {
        return group.supported && moi_scalar_contexts_admit_reserved_window(group.owners.summaries,
                                                                            candidate, count, arch);
      };
      const bool admits_owner = std::ranges::any_of(plan_contexts.groups, group_admits_window);
      const bool excludes_owner =
          std::ranges::any_of(plan_contexts.groups, [&](const MoiScalarOwnerContextGroup &group) {
            return !group_admits_window(group);
          });
      if (!admits_owner || !excludes_owner)
        continue;
      options.moi_exec_save_sgpr = candidate;
      options.automatic_moi_exec_save_sgprs = true;
      options.automatic_moi_partial_exec_save_sgprs = true;
      options.moi_exec_save_sgprs_persistent = true;
      const bool assigned_component_local =
          configure_automatic_moi_transient_sgpr_assignments(options, result, state);
      const bool can_exclude_incompatible_component =
          arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4;
      if (assigned_component_local || can_exclude_incompatible_component) {
        result.resolved_moi_exec_save_sgpr = options.moi_exec_save_sgpr;
        result.moi_exec_save_sgprs_automatic = true;
        result.warnings.emplace_back(
            "ConSan MOI reserved a high EXEC-save SGPR window, assigned component-local "
            "scalar state where available, and excluded incompatible owners");
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
      options.moi_inline_access_present && state.valid && saw_site) {
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
                             moi_dispatch_state_sgpr_count(options))) ||
             (options.moi_dynamic_stack_spill &&
              moi_sgpr_range_overlaps_dynamic_stack_state(base, width));
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
      // A singleton/sparse appended body uses only the indirect PC pair and
      // SCC save.  The dispatch key is validated as part of the spill-backed
      // scalar ABI, but is never live in this non-dense path, so let it alias
      // the SCC save instead of demanding another dead scalar.
      if (!needs_dense_router && scc)
        dispatch_key = scc;
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
           moi_sgpr_range_overlaps_dynamic_stack_state(candidate, count))) {
        continue;
      }
      const bool admits_owner =
          std::ranges::any_of(plan_contexts.groups, [&](const MoiScalarOwnerContextGroup &group) {
            return group.supported && consan_detail::scalar_owner_contexts_admit_reserved_window(
                                          group.owners.summaries, candidate, count,
                                          /*protect_physical_vcc=*/false);
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
      (void)configure_automatic_moi_transient_sgpr_assignments(options, result, state);
      return true;
    }
  }

  // A code object made entirely of full-pressure owners has no owner from
  // which to seed the ordinary high-window fallback. Seed component planning
  // with a placeholder; every admitted owner must replace it with either a
  // liveness-dead window or a spill-backed access-probe layout.
  if (moi_supports_full_pressure_component_state(arch, options.moi_engine)) {
    options.moi_exec_save_sgpr = 0u;
    options.automatic_moi_exec_save_sgprs = true;
    options.automatic_moi_partial_exec_save_sgprs = true;
    if (configure_automatic_moi_transient_sgpr_assignments(options, result, state)) {
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
                                                      const MoiResourcePlanningState &state) {
  const rj_code_arch_t arch = state.arch;
  if (options.moi_owner_source != ConSanMoiOwnerSource::HwId || options.moi_owner_sgpr)
    return false;
  if (options.moi_engine == ConSanMoiEngine::InlineShadow)
    return false;

  const auto plan_contexts = resolve_moi_resource_plan_scalar_owner_contexts(state, result);
  if (!plan_contexts.planning_state_valid || plan_contexts.unresolved_owner) {
    std::string warning =
        "ConSan MOI could not resolve an owner context for automatic hw_id owner SGPR placement";
    if (plan_contexts.unresolved_owner)
      warning += " at descriptor file offset " + std::to_string(*plan_contexts.unresolved_owner);
    result.warnings.push_back(std::move(warning));
    return false;
  }
  const MoiScalarOwnerContextResolution scalar_contexts =
      collect_moi_scalar_owner_contexts(plan_contexts.groups, true);
  if (scalar_contexts.contexts.empty()) {
    result.warnings.emplace_back(
        "ConSan MOI could not derive an automatic hw_id owner SGPR without an owned candidate "
        "scope");
    return false;
  }

  for (uint32_t sgpr = scalar_contexts.tail_floor; sgpr < moi_ordinary_sgpr_limit(arch); ++sgpr) {
    if (options.moi_exec_save_sgpr &&
        range_overlaps(static_cast<uint16_t>(sgpr), 1u, *options.moi_exec_save_sgpr,
                       moi_exec_save_sgpr_count(options, arch))) {
      continue;
    }
    if (!moi_scalar_contexts_admit_reserved_window(scalar_contexts.summaries,
                                                   static_cast<uint16_t>(sgpr), 1u, arch)) {
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

[[nodiscard]] bool configure_inline_moi_owner_sgpr(ConSanOptions &options, ConSanResult &result) {
  if (!moi_inline_uses_planned_owner_sgpr(options) || options.automatic_moi_owner_sgpr)
    return false;
  if (!options.moi_exec_save_sgpr) {
    result.warnings.emplace_back(
        "ConSan MOI could not place its Inline resident-wave owner in a transient scalar window");
    return false;
  }
  options.moi_owner_sgpr = *options.moi_exec_save_sgpr;
  options.automatic_moi_owner_sgpr = true;
  std::optional<uint16_t> resolved_owner = options.moi_owner_sgpr;
  if (!options.moi_transient_sgpr_assignments.empty()) {
    resolved_owner = options.moi_transient_sgpr_assignments.front().owner_sgpr;
    for (const ConSanMoiTransientSgprAssignment &assignment :
         options.moi_transient_sgpr_assignments) {
      if (!resolved_owner || assignment.owner_sgpr != resolved_owner) {
        resolved_owner.reset();
        break;
      }
    }
  }
  result.resolved_moi_owner_sgpr = resolved_owner;
  result.moi_owner_sgpr_automatic = true;
  result.resolved_moi_transient_sgpr_assignments = options.moi_transient_sgpr_assignments;
  result.warnings.emplace_back(
      "ConSan MOI assigned its Inline resident-wave owner inside the saved scalar window");
  return true;
}

[[nodiscard]] bool
configure_automatic_moi_dispatch_id_sgprs(ConSanOptions &options, ConSanResult &result,
                                          const MoiResourcePlanningState &state) {
  const rj_code_arch_t arch = state.arch;
  const bool has_inline_dispatch_consumer =
      options.moi_engine == ConSanMoiEngine::InlineShadow &&
      !(arch == ROCJITSU_CODE_ARCH_GFX1250 && !options.moi_track_atomics) &&
      std::ranges::any_of(result.resource_plans, [](const ConSanCandidateResourcePlan &plan) {
        return !plan.owner_descriptor_file_offsets.empty() &&
               (plan.site_kind == ConSanResourceSiteKind::Access ||
                plan.site_kind == ConSanResourceSiteKind::Atomic);
      });
  const bool needs_dispatch_id =
      has_inline_dispatch_consumer || options.moi_engine == ConSanMoiEngine::RecordReplay ||
      (options.moi_engine == ConSanMoiEngine::Sampled && options.moi_runtime_sample_stride > 1u &&
       arch != ROCJITSU_CODE_ARCH_GFX1250);
  if (!needs_dispatch_id || !options.moi_report_buffer_address || options.moi_dispatch_id_sgpr ||
      options.moi_dispatch_id_vgpr) {
    return false;
  }

  const auto plan_contexts = resolve_moi_resource_plan_scalar_owner_contexts(state, result);
  if (!plan_contexts.planning_state_valid || plan_contexts.unresolved_owner) {
    std::string warning =
        "ConSan MOI could not resolve an owner context for automatic dispatch-ID SGPR placement";
    if (plan_contexts.unresolved_owner)
      warning += " at descriptor file offset " + std::to_string(*plan_contexts.unresolved_owner);
    result.warnings.push_back(std::move(warning));
    return false;
  }
  const MoiScalarOwnerContextResolution scalar_contexts =
      collect_moi_scalar_owner_contexts(plan_contexts.groups, true);
  if (scalar_contexts.contexts.empty()) {
    result.warnings.emplace_back(
        "ConSan MOI could not derive automatic dispatch-ID SGPRs without an owned candidate "
        "scope");
    return false;
  }

  // Sixteen initialized user SGPRs plus X/Y/Z and WORKGROUP_INFO is the
  // largest legal entry-input window after insertion. Keeping the persistent
  // pair above that window makes it safe for every per-kernel preload map.
  uint32_t first = std::max<uint32_t>(20u, scalar_contexts.tail_floor);

  const auto qualify_owner_local_assignments = [&](uint16_t dispatch_base) {
    for (ConSanMoiTransientSgprAssignment &assignment : options.moi_transient_sgpr_assignments) {
      const std::array owner = {assignment.descriptor_file_offset};
      const auto contexts = resolve_moi_scalar_owner_contexts(state, owner);
      assignment.dispatch_id_sgpr = contexts && moi_scalar_contexts_admit_reserved_window(
                                                    contexts->summaries, dispatch_base, 2u, arch)
                                        ? std::optional(dispatch_base)
                                        : std::nullopt;
    }
    result.resolved_moi_transient_sgpr_assignments = options.moi_transient_sgpr_assignments;
  };

  first = util::align_up(first, 2u);
  const uint16_t dispatch_state_count = moi_dispatch_state_sgpr_count(options);
  const uint16_t ordinary_sgpr_limit = moi_ordinary_sgpr_limit(arch);
  for (uint32_t base = first; base + dispatch_state_count <= ordinary_sgpr_limit; base += 2u) {
    if (persistent_sgpr_range_overlaps_reserved_ordinary_range(static_cast<uint16_t>(base),
                                                               dispatch_state_count, arch)) {
      continue;
    }
    if (moi_scalar_contexts_conflict_with_physical_vcc(
            scalar_contexts.summaries, static_cast<uint16_t>(base), dispatch_state_count, arch)) {
      continue;
    }
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
  // of that one owner. Reserve the highest pair that is legal for at least one
  // owner and let resource-plan rebuilding reject only incompatible groups.
  // The pair remains code-object-wide, so this does not change the report ABI
  // or any emitted probe sequence.
  if (arch == ROCJITSU_CODE_ARCH_GFX1250 || arch == ROCJITSU_CODE_ARCH_CDNA3 ||
      arch == ROCJITSU_CODE_ARCH_CDNA4) {
    const int32_t last_base =
        static_cast<int32_t>((ordinary_sgpr_limit - dispatch_state_count) & ~1u);
    for (int32_t base = last_base; base >= 20; base -= 2) {
      if (persistent_sgpr_range_overlaps_reserved_ordinary_range(static_cast<uint16_t>(base),
                                                                 dispatch_state_count, arch)) {
        continue;
      }
      const bool admits_owner =
          std::ranges::any_of(plan_contexts.groups, [&](const MoiScalarOwnerContextGroup &group) {
            return group.supported && moi_scalar_contexts_admit_reserved_window(
                                          group.owners.summaries, static_cast<uint16_t>(base),
                                          dispatch_state_count, arch);
          });
      if ((arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4) && !admits_owner)
        continue;
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
    // A dynamic-stack kernel can allocate the full ordinary scalar file while
    // leaving a lower pair completely untouched. Dispatch identity crosses
    // arbitrary guest code, so site-local liveness is not sufficient: require
    // a closed owner CFG, or conservatively scan every declared function when
    // those ranges cover the complete text section. The pair also remains
    // above the largest possible entry-input window so dispatch-ID insertion
    // and guest-input restoration cannot overwrite it.
    MoiOwnerRegisterReferenceSummary references =
        summarize_moi_owner_register_references(scalar_contexts.contexts);
    bool used_complete_text_summary = false;
    if (!references.control_flow_closed) {
      if (auto complete = summarize_moi_complete_text_register_references(state)) {
        references = std::move(*complete);
        used_complete_text_summary = true;
      }
    }
    const bool has_dynamic_stack_owner =
        std::ranges::any_of(result.kernels, [](const ConSanKernelInfo &kernel) {
          return kernel.uses_dynamic_stack.value_or(false);
        });
    const auto referenced = [&](uint16_t base) {
      for (uint16_t index = 0; index < dispatch_state_count; ++index) {
        if (references.references.contains(
                {RegClass::SGPR, static_cast<uint16_t>(base + index), 1u})) {
          return true;
        }
      }
      return false;
    };
    if (references.control_flow_closed && !references.uses_indirect_sgprs) {
      for (uint32_t base = 20u; base + dispatch_state_count <= ordinary_sgpr_limit; base += 2u) {
        const uint16_t candidate = static_cast<uint16_t>(base);
        if (referenced(candidate) ||
            persistent_sgpr_range_overlaps_reserved_ordinary_range(candidate, dispatch_state_count,
                                                                   arch) ||
            moi_scalar_contexts_conflict_with_physical_vcc(scalar_contexts.summaries, candidate,
                                                           dispatch_state_count, arch) ||
            (options.moi_owner_sgpr &&
             range_overlaps(candidate, dispatch_state_count, *options.moi_owner_sgpr, 1u)) ||
            (options.moi_exec_save_sgpr &&
             range_overlaps(candidate, dispatch_state_count, *options.moi_exec_save_sgpr,
                            moi_exec_save_sgpr_count(options, arch))) ||
            (has_dynamic_stack_owner &&
             moi_sgpr_range_overlaps_dynamic_stack_state(candidate, dispatch_state_count))) {
          continue;
        }
        options.moi_dispatch_id_sgpr = candidate;
        options.automatic_moi_dispatch_id_sgprs = true;
        result.resolved_moi_dispatch_id_sgpr = candidate;
        result.moi_dispatch_id_sgprs_automatic = true;
        qualify_owner_local_assignments(candidate);
        result.warnings.emplace_back(
            "ConSan MOI assigned persistent dispatch-ID state to a proven-unused RDNA4 SGPR pair" +
            std::string(used_complete_text_summary ? " using complete-text coverage" : ""));
        return true;
      }
    }
  }

  if (arch == ROCJITSU_CODE_ARCH_RDNA4) {
    // Automatic banked Record/Replay can capture the preload into persistent
    // descriptor-grown VGPRs during the later owner-state placement pass.
    if (record_replay_uses_automatic_banked_capture(options))
      return false;
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

[[nodiscard]] bool
configure_gfx1250_record_replay_dispatch_id_overrides(ConSanOptions &options, ConSanResult &result,
                                                      const MoiResourcePlanningState &state) {
  const rj_code_arch_t arch = state.arch;
  if (arch != ROCJITSU_CODE_ARCH_GFX1250 || options.moi_engine != ConSanMoiEngine::RecordReplay ||
      !options.automatic_moi_dispatch_id_sgprs || !options.moi_dispatch_id_sgpr ||
      !options.moi_exec_save_sgpr) {
    return false;
  }

  // Dispatch placement runs before transient EXEC-save placement. Ask the
  // existing component planner to prove a complete owner-local scalar ABI for
  // any newly included full-pressure component. Do not infer that the global
  // EXEC-save window is safe merely because dispatch placement excluded the
  // component before its liveness was considered.
  const auto prior_assignments = options.moi_transient_sgpr_assignments;
  const auto prior_resolved_assignments = result.resolved_moi_transient_sgpr_assignments;
  const bool prior_partial_exec = options.automatic_moi_partial_exec_save_sgprs;
  options.moi_transient_sgpr_assignments.clear();
  result.resolved_moi_transient_sgpr_assignments.clear();
  options.automatic_moi_partial_exec_save_sgprs = true;
  if (!configure_automatic_moi_transient_sgpr_assignments(options, result, state)) {
    options.moi_transient_sgpr_assignments = prior_assignments;
    result.resolved_moi_transient_sgpr_assignments = prior_resolved_assignments;
    options.automatic_moi_partial_exec_save_sgprs = prior_partial_exec;
    return false;
  }

  // The component planner connects all callers of shared text and fails
  // closed on gfx1250 unless it can assign every component. It also retains
  // the hardware pair per owner exactly when that owner's scalar pressure
  // permits it; otherwise the assignment's established contract is to record
  // zero generation.
  const bool selected_zero_generation =
      std::ranges::any_of(options.moi_transient_sgpr_assignments,
                          [](const auto &assignment) { return !assignment.dispatch_id_sgpr; });
  if (!selected_zero_generation) {
    options.moi_transient_sgpr_assignments = prior_assignments;
    result.resolved_moi_transient_sgpr_assignments = prior_resolved_assignments;
    options.automatic_moi_partial_exec_save_sgprs = prior_partial_exec;
    return false;
  }

  result.warnings.emplace_back(
      "ConSan MOI selected owner-local zero-generation records where the gfx1250 hardware "
      "dispatch-ID pair overlaps guest scalar state");
  return true;
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
  const bool below_entry_window = base < 20u;
  const bool unaligned = base % 2u != 0u;
  const bool beyond_ordinary_file =
      static_cast<uint32_t>(base) + dispatch_state_count > moi_ordinary_sgpr_limit(arch);
  const bool overlaps_reserved =
      persistent_sgpr_range_overlaps_reserved_ordinary_range(base, dispatch_state_count, arch);
  if (below_entry_window || unaligned || beyond_ordinary_file || overlaps_reserved ||
      overlaps_owner || overlaps_exec) {
    result.outcome = ConSanTransformOutcome::Unsupported;
    std::string reason;
    const auto append_reason = [&](std::string_view item) {
      if (!reason.empty())
        reason += ", ";
      reason += item;
    };
    if (below_entry_window)
      append_reason("entry window");
    if (unaligned)
      append_reason("alignment");
    if (beyond_ordinary_file)
      append_reason("ordinary-file bound");
    if (overlaps_reserved)
      append_reason("architectural reservation");
    if (overlaps_owner)
      append_reason("owner temporary");
    if (overlaps_exec)
      append_reason("EXEC-save window");
    result.warnings.emplace_back(
        "ConSan MOI persistent dispatch-ID SGPR pair s" + std::to_string(base) + ":s" +
        std::to_string(base + dispatch_state_count - 1u) + " violates " + reason);
    return false;
  }
  return true;
}

[[nodiscard]] bool validate_moi_dispatch_id_vgprs(const ConSanOptions &options,
                                                  ConSanResult &result) {
  if (options.moi_dispatch_id_sgpr && options.moi_dispatch_id_vgpr) {
    result.outcome = ConSanTransformOutcome::Unsupported;
    result.warnings.emplace_back(
        "ConSan MOI has multiple persistent hardware dispatch-ID representations");
    return false;
  }
  if (!options.moi_dispatch_id_vgpr)
    return true;

  const uint16_t base = *options.moi_dispatch_id_vgpr;
  const auto overlaps = [&](std::optional<uint16_t> reg) {
    return reg && range_overlaps(base, 2u, *reg, 1u);
  };
  bool overlaps_persistent_state = overlaps(options.moi_owner_vgpr) ||
                                   overlaps(options.moi_epoch_vgpr) ||
                                   overlaps(options.moi_workgroup_key_vgpr);
  for (const std::optional<uint16_t> reg :
       {options.moi_record_replay_workgroup_vgprs.x, options.moi_record_replay_workgroup_vgprs.y,
        options.moi_record_replay_workgroup_vgprs.z}) {
    overlaps_persistent_state |= overlaps(reg);
  }
  if (static_cast<uint32_t>(base) + 2u > kMaxVgprs || overlaps_persistent_state) {
    result.outcome = ConSanTransformOutcome::Unsupported;
    result.warnings.emplace_back(
        "ConSan MOI persistent dispatch-ID VGPR pair overlaps persistent state or exceeds the "
        "architectural VGPR file");
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
  const auto persistent_workgroup_tuple_invalid = [&] {
    for (const std::optional<uint16_t> reg :
         {options.moi_record_replay_workgroup_sgprs.x, options.moi_record_replay_workgroup_sgprs.y,
          options.moi_record_replay_workgroup_sgprs.z}) {
      if (reg && (*reg >= limit ||
                  persistent_sgpr_range_overlaps_reserved_ordinary_range(*reg, 1u, arch))) {
        return true;
      }
    }
    return false;
  }();
  const bool persistent_invalid =
      (options.moi_persistent_owner_sgpr && *options.moi_persistent_owner_sgpr >= limit) ||
      (options.moi_persistent_epoch_sgpr && *options.moi_persistent_epoch_sgpr >= limit) ||
      (options.moi_persistent_workgroup_key_sgpr &&
       *options.moi_persistent_workgroup_key_sgpr >= limit) ||
      (options.moi_persistent_owner_sgpr && persistent_sgpr_range_overlaps_reserved_ordinary_range(
                                                *options.moi_persistent_owner_sgpr, 1u, arch)) ||
      (options.moi_persistent_epoch_sgpr && persistent_sgpr_range_overlaps_reserved_ordinary_range(
                                                *options.moi_persistent_epoch_sgpr, 1u, arch)) ||
      (options.moi_persistent_workgroup_key_sgpr &&
       persistent_sgpr_range_overlaps_reserved_ordinary_range(
           *options.moi_persistent_workgroup_key_sgpr, 1u, arch)) ||
      persistent_workgroup_tuple_invalid;
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
  const bool branch_only_scalar_invalid =
      options.moi_inline_branch_only_scalar_spill &&
      (options.moi_engine != ConSanMoiEngine::InlineShadow ||
       !options.automatic_moi_inline_sgpr_spill || !is_rdna4_family_arch(arch) ||
       !options.moi_exec_save_sgpr || options.moi_inline_indirect_pc_sgpr ||
       options.moi_inline_indirect_scc_sgpr || options.moi_inline_dispatch_key_sgpr ||
       options.moi_inline_call_return_sgpr);
  const bool dynamic_bootstrap_invalid = [&] {
    if (!options.moi_inline_dynamic_stack_borrowed_sgpr)
      return false;
    const uint16_t base = *options.moi_inline_dynamic_stack_borrowed_sgpr;
    const uint16_t exec_count = moi_exec_save_sgpr_count(options, arch);
    const bool overlaps_persistent =
        (options.moi_persistent_owner_sgpr &&
         range_overlaps(base, 2u, *options.moi_persistent_owner_sgpr, 1u)) ||
        (options.moi_persistent_epoch_sgpr &&
         range_overlaps(base, 2u, *options.moi_persistent_epoch_sgpr, 1u)) ||
        (options.moi_persistent_workgroup_key_sgpr &&
         range_overlaps(base, 2u, *options.moi_persistent_workgroup_key_sgpr, 1u));
    bool overlaps_workgroup_tuple = false;
    for (const std::optional<uint16_t> reg :
         {options.moi_record_replay_workgroup_sgprs.x, options.moi_record_replay_workgroup_sgprs.y,
          options.moi_record_replay_workgroup_sgprs.z}) {
      overlaps_workgroup_tuple |= reg && range_overlaps(base, 2u, *reg, 1u);
    }
    return options.moi_engine != ConSanMoiEngine::InlineShadow ||
           !options.automatic_moi_inline_sgpr_spill ||
           !options.moi_inline_branch_only_scalar_spill || !options.moi_dynamic_stack_spill ||
           !is_rdna4_family_arch(arch) || base % 2u != 0u ||
           static_cast<uint32_t>(base) + 2u > limit ||
           moi_sgpr_range_overlaps_dynamic_stack_state(base, 2u) ||
           persistent_sgpr_range_overlaps_reserved_ordinary_range(base, 2u, arch) ||
           (options.moi_exec_save_sgpr &&
            range_overlaps(base, 2u, *options.moi_exec_save_sgpr, exec_count)) ||
           options.moi_inline_indirect_pc_sgpr || options.moi_inline_indirect_scc_sgpr ||
           options.moi_inline_dispatch_key_sgpr || options.moi_inline_call_return_sgpr ||
           overlaps_persistent || overlaps_workgroup_tuple;
  }();
  if (!exec_invalid && !owner_invalid && !persistent_invalid && !inline_external_invalid &&
      !branch_only_scalar_invalid && !dynamic_bootstrap_invalid)
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
  // Runtime sampling is spatial: dispatch and workgroup identity select a
  // bank, while the LDS cell selects an address. No per-workitem sequence is
  // read by the emitted probes, so persistent Sampled state is only
  // owner/epoch.
  const bool needs_inline_workgroup_key =
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
        // Capture the full workgroup identity at kernel entry for every Inline
        // access. CDNA ordinary entry SGPRs and RDNA-family TTMP launch inputs
        // are both entry ABI values; probes must not rely on either remaining
        // available after arbitrary guest code.
        return plan.site_kind == ConSanResourceSiteKind::Access;
      });
  // Every Record/Replay layout publishes exact workgroup coordinates. Banking
  // changes retention and scratch requirements, but a caller-owned single-bank
  // probe executes under the same entry-state lifetime rule as an automatic
  // probe.
  const bool has_supported_record_replay_site =
      std::ranges::any_of(result.resource_plans, [](const ConSanCandidateResourcePlan &plan) {
        return plan.source != ConSanRegisterAllocationSource::Unsupported &&
               !plan.owner_descriptor_file_offsets.empty() &&
               (plan.site_kind == ConSanResourceSiteKind::Access ||
                plan.site_kind == ConSanResourceSiteKind::Barrier ||
                plan.site_kind == ConSanResourceSiteKind::Atomic ||
                plan.site_kind == ConSanResourceSiteKind::Fence);
      });
  const bool needs_record_replay_workgroup_tuple =
      record_replay_requires_entry_workgroup_capture(options) && has_supported_record_replay_site &&
      !record_replay_has_entry_workgroup_capture(options);
  const bool has_supported_record_replay_access =
      options.moi_engine == ConSanMoiEngine::RecordReplay &&
      std::ranges::any_of(result.resource_plans, [](const ConSanCandidateResourcePlan &plan) {
        return plan.source != ConSanRegisterAllocationSource::Unsupported &&
               !plan.owner_descriptor_file_offsets.empty() &&
               plan.site_kind == ConSanResourceSiteKind::Access;
      });
  const bool needs_persistent_dispatch_vgprs =
      is_rdna4_family_arch(arch) && has_supported_record_replay_access &&
      record_replay_uses_automatic_banked_capture(options) &&
      !moi_has_runtime_hardware_dispatch_id(options);
  const bool needs_workgroup_key = needs_inline_workgroup_key;
  const bool has_explicit_owner_epoch = options.moi_owner_vgpr && options.moi_epoch_vgpr;
  if (!record_replay_entry_workgroup_capture_is_unambiguous(options)) {
    result.outcome = ConSanTransformOutcome::Unsupported;
    result.warnings.emplace_back(
        "ConSan MOI Record/Replay has multiple persistent workgroup-tuple representations");
    return false;
  }
  if ((options.moi_owner_vgpr || options.moi_epoch_vgpr) && !has_explicit_owner_epoch) {
    result.outcome = ConSanTransformOutcome::Unsupported;
    result.warnings.emplace_back(
        "ConSan MOI persistent VGPR overrides require both owner and epoch registers");
    return false;
  }
  if ((options.moi_workgroup_key_vgpr && !needs_record_replay_workgroup_tuple) ||
      (has_explicit_owner_epoch && !needs_record_replay_workgroup_tuple))
    return false;
  // Sampled access probes execute after arbitrary guest instructions. Their
  // owner must therefore come from entry-captured persistent state even when
  // the code object has no semantically admitted synchronization site; v0 is
  // only the ABI workitem ID at kernel entry and may be clobbered later.
  const bool needs_persistent_state =
      options.moi_init_owner_epoch || needs_record_replay_workgroup_tuple ||
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
    // allocation instead. On CDNA3/4, ACCUM_OFFSET bounds this search below the
    // physical AccVGPR bank even when the descriptor's total VGPR allocation
    // extends through that bank.
    const uint32_t safe_base =
        plan.has_indirect_vgpr_access
            ? std::min<uint32_t>(plan.current_vgpr_count, plan.ordinary_vgpr_limit)
            : plan.max_referenced_vgpr_count;
    persistent_base = std::max(persistent_base, safe_base);
  }
  if (has_explicit_owner_epoch) {
    persistent_base = std::max<uint32_t>(
        persistent_base, std::max(*options.moi_owner_vgpr, *options.moi_epoch_vgpr) + 1u);
  }

  if (!found_owned_scope) {
    result.warnings.emplace_back(
        "ConSan MOI could not derive automatic persistent VGPRs without an owned candidate "
        "scope");
    return false;
  }
  if (is_rdna4_family_arch(arch) && options.moi_engine == ConSanMoiEngine::RecordReplay) {
    // Persistent owner/epoch state and a site-local Record/Replay probe share
    // the ordinary VGPR file. Keep the tuple above the largest operational
    // scratch window so a small kernel can still spill v0..vN when no
    // liveness-dead window exists. Guest access operands may overlap that
    // spilled window because emission recovers them from the spill frame;
    // persistent values cannot overlap it because they must survive for the
    // complete dispatch.
    uint32_t scratch_floor = 0;
    for (const ConSanCandidateResourcePlan &plan : result.resource_plans) {
      if (plan.source != ConSanRegisterAllocationSource::Unsupported &&
          !plan.owner_descriptor_file_offsets.empty()) {
        const uint32_t scratch_end =
            plan.scratch_vgpr ? static_cast<uint32_t>(*plan.scratch_vgpr) + plan.scratch_vgpr_count
                              : plan.scratch_vgpr_count;
        scratch_floor = std::max(scratch_floor, scratch_end);
      }
    }
    persistent_base = std::max(persistent_base, scratch_floor);
  }
  // The gfx1250 Inline local-shadow prologue uses three additional tail VGPRs
  // only while clearing with a 128-bit LDS store. They must lie above every
  // guest reference even though owner/epoch remain the only persistent values.
  const uint32_t automatic_owner_epoch_count = has_explicit_owner_epoch ? 0u : 2u;
  const uint32_t record_replay_workgroup_count = needs_record_replay_workgroup_tuple ? 3u : 0u;
  const uint32_t dispatch_id_count = needs_persistent_dispatch_vgprs ? 2u : 0u;
  const uint32_t named_persistent_count = automatic_owner_epoch_count +
                                          static_cast<uint32_t>(needs_workgroup_key) +
                                          record_replay_workgroup_count + dispatch_id_count;
  const uint32_t persistent_count =
      options.moi_engine == ConSanMoiEngine::InlineShadow && arch == ROCJITSU_CODE_ARCH_GFX1250
          ? std::max<uint32_t>(named_persistent_count, 5u)
          : named_persistent_count;
  const uint32_t persistent_end = persistent_base + persistent_count;
  const auto has_dynamic_stack_resource_owner = [&] {
    return std::ranges::any_of(result.kernels, [&](const ConSanKernelInfo &kernel) {
      return kernel.uses_dynamic_stack.value_or(false) &&
             std::ranges::any_of(
                 result.resource_plans, [&](const ConSanCandidateResourcePlan &plan) {
                   return plan.source != ConSanRegisterAllocationSource::Unsupported &&
                          std::ranges::find(plan.owner_descriptor_file_offsets,
                                            kernel.descriptor_file_offset) !=
                              plan.owner_descriptor_file_offsets.end();
                 });
    });
  };
  const auto assign_scalar_persistent_state = [&](uint16_t base) {
    uint16_t next = base;
    if (!has_explicit_owner_epoch) {
      options.moi_persistent_owner_sgpr = next++;
      options.moi_persistent_epoch_sgpr = next++;
    }
    if (needs_workgroup_key)
      options.moi_persistent_workgroup_key_sgpr = next++;
    if (needs_record_replay_workgroup_tuple) {
      options.moi_record_replay_workgroup_sgprs = {
          .x = next,
          .y = static_cast<uint16_t>(next + 1u),
          .z = static_cast<uint16_t>(next + 2u),
      };
    }
  };
  const auto publish_scalar_persistent_state = [&] {
    result.resolved_moi_persistent_owner_sgpr = options.moi_persistent_owner_sgpr;
    result.resolved_moi_persistent_epoch_sgpr = options.moi_persistent_epoch_sgpr;
    result.resolved_moi_persistent_workgroup_key_sgpr = options.moi_persistent_workgroup_key_sgpr;
    result.resolved_moi_record_replay_workgroup_sgprs = options.moi_record_replay_workgroup_sgprs;
  };
  const bool crosses_cdna_accvgpr_boundary =
      (result.arch == ROCJITSU_CODE_ARCH_CDNA3 || result.arch == ROCJITSU_CODE_ARCH_CDNA4) &&
      std::ranges::any_of(result.resource_plans, [&](const ConSanCandidateResourcePlan &plan) {
        return plan.source != ConSanRegisterAllocationSource::Unsupported &&
               !plan.owner_descriptor_file_offsets.empty() &&
               ((plan.ordinary_vgpr_limit != kMaxVgprs &&
                 persistent_end > plan.ordinary_vgpr_limit) ||
                persistent_end > kMaxVgprs);
      });
  if (crosses_cdna_accvgpr_boundary) {
    // ACCUM_OFFSET defines the compiler's ordinary/accumulator register split.
    // Moving that split after code generation changes which physical registers
    // existing MFMA and accumulator operands address and has corrupted native
    // gfx950 workloads. Until persistent tuples are selected per owning kernel,
    // reject a code object whose one global tuple would cross any such boundary.
    const bool has_dynamic_stack_owner = has_dynamic_stack_resource_owner();
    if (options.moi_engine == ConSanMoiEngine::RecordReplay && options.force_private_epoch &&
        !has_dynamic_stack_owner) {
      options.moi_init_owner_epoch = true;
      options.automatic_moi_private_epoch = true;
      result.moi_private_epoch_automatic = true;
      result.warnings.emplace_back(
          "ConSan MOI selected entry-captured private Record/Replay state below a CDNA "
          "ordinary-VGPR boundary");
      return true;
    }
    if (options.moi_engine == ConSanMoiEngine::Sampled &&
        options.moi_owner_source == ConSanMoiOwnerSource::WorkitemId && !has_dynamic_stack_owner) {
      options.moi_init_owner_epoch = true;
      options.automatic_moi_private_epoch = true;
      result.moi_private_epoch_automatic = true;
      result.warnings.emplace_back(
          "ConSan MOI selected entry-snapshotted private owner/epoch state below a CDNA ordinary "
          "VGPR boundary for Sampled instrumentation");
      return true;
    }
    if (options.moi_engine == ConSanMoiEngine::RecordReplay && !options.force_private_epoch) {
      const uint16_t scalar_count = static_cast<uint16_t>(named_persistent_count);
      MoiResourcePlanningState scalar_state(bytes, arch, options, result);
      const std::vector<uint64_t> scalar_owners = moi_scalar_owner_descriptors(result);
      const auto resolved_contexts = resolve_moi_scalar_owner_contexts(scalar_state, scalar_owners);
      const std::vector<const MoiKernelResourceContext *> scalar_contexts =
          resolved_contexts ? resolved_contexts->contexts
                            : std::vector<const MoiKernelResourceContext *>{};
      // Per-context allocation floors and VCC checks are intentionally
      // stricter than the former plan-min scan. Resolution failure leaves the
      // candidate range empty so placement fails closed.
      const uint32_t first =
          resolved_contexts ? resolved_contexts->tail_floor : moi_ordinary_sgpr_limit(arch);
      const auto unavailable = [&](uint16_t base) {
        if ((options.moi_exec_save_sgpr &&
             range_overlaps(base, scalar_count, *options.moi_exec_save_sgpr,
                            moi_exec_save_sgpr_count(options, arch))) ||
            (options.moi_dispatch_id_sgpr &&
             range_overlaps(base, scalar_count, *options.moi_dispatch_id_sgpr,
                            moi_dispatch_state_sgpr_count(options))) ||
            (options.moi_owner_sgpr &&
             range_overlaps(base, scalar_count, *options.moi_owner_sgpr, 1u))) {
          return true;
        }
        return resolved_contexts && moi_scalar_contexts_conflict_with_physical_vcc(
                                        resolved_contexts->summaries, base, scalar_count, arch);
      };
      for (uint32_t base = first; base + scalar_count <= moi_ordinary_sgpr_limit(arch); ++base) {
        if (unavailable(static_cast<uint16_t>(base)))
          continue;
        std::vector<ConSanMoiPrologueScratchVgprAssignment> prologue_scratch;
        // Record/Replay initializes both persistent values directly in SGPRs;
        // its shared prologue API does not use the placeholder VGPR pair.
        // Resource plans are rebuilt after persistent-state selection. Keep a
        // placeholder for every possible kernel owner so a site admitted by
        // that rebuild can still receive its scalar prologue. Prologue
        // lowering itself filters this list to kernels with emitted patches.
        for (const ConSanKernelInfo &kernel : result.kernels) {
          if (kernel.has_text_range)
            prologue_scratch.push_back({kernel.descriptor_file_offset, 0u});
        }
        assign_scalar_persistent_state(static_cast<uint16_t>(base));
        options.moi_init_owner_epoch = true;
        options.automatic_moi_persistent_sgprs = true;
        options.moi_prologue_scratch_vgpr_assignments = std::move(prologue_scratch);
        publish_scalar_persistent_state();
        result.moi_persistent_sgprs_automatic = true;
        result.resolved_moi_prologue_scratch_vgpr_assignments =
            options.moi_prologue_scratch_vgpr_assignments;
        result.warnings.emplace_back(
            "ConSan MOI assigned persistent scalar " +
            std::string(has_explicit_owner_epoch ? "" : "owner/epoch") +
            std::string(needs_workgroup_key ? "/workgroup-key" : "") +
            std::string(needs_record_replay_workgroup_tuple ? "/exact-workgroup-tuple" : "") +
            " state for CDNA Record/Replay");
        return true;
      }

      // A linked kernel can reference a high SGPR transiently while leaving a
      // lower window completely untouched throughout every owning kernel and
      // shared helper. Persistent state may use such a hole only after proving
      // that the pair is absent from the complete owner scopes; site-local
      // liveness is insufficient because the values cross arbitrary guest
      // code between the entry prologue and every probe.
      ConSanOptions scalar_options = options;
      const bool scalar_state_valid =
          resolved_contexts.has_value() &&
          apply_moi_transient_sgpr_assignment(scalar_options, scalar_owners);
      bool every_owner_resolved = scalar_state_valid;
      MoiOwnerRegisterReferenceSummary scalar_references =
          summarize_moi_owner_register_references(scalar_contexts);
      bool used_complete_text_summary = false;
      if (resolved_contexts && !scalar_references.control_flow_closed) {
        // Some generated kernels use indirect local-label control flow whose
        // recovered target set is absent or explicitly non-exhaustive. When
        // declared function ranges cover the whole text section, scanning all
        // of them is a conservative substitute for a closed owner CFG:
        // unrelated code may reject a hole, but omitted destinations cannot
        // make an occupied register look free.
        if (auto complete = summarize_moi_complete_text_register_references(scalar_state)) {
          scalar_references = std::move(*complete);
          used_complete_text_summary = true;
        }
      }
      for (const MoiKernelResourceContext *context : scalar_contexts) {
        const uint64_t descriptor_offset = context->kernel->descriptor_file_offset;
        if (context->kernel->uses_dynamic_stack.value_or(false)) {
          scalar_references.references.expand({RegClass::SGPR, kDynamicStackTopSgpr, 1u});
          scalar_references.references.expand({RegClass::SGPR, kDynamicFrameBaseSgpr, 1u});
        }
        if (descriptor_offset > bytes.size() || sizeof(KD) > bytes.size() - descriptor_offset) {
          every_owner_resolved = false;
          break;
        }
        KD descriptor{};
        std::memcpy(&descriptor, bytes.data() + descriptor_offset, sizeof(descriptor));
        const uint32_t initialized_count = moi_descriptor_user_sgpr_count(descriptor) +
                                           moi_descriptor_system_sgpr_count(descriptor);
        for (uint32_t sgpr = 0u;
             sgpr < std::min<uint32_t>(initialized_count, moi_ordinary_sgpr_limit(arch)); ++sgpr) {
          scalar_references.references.expand({RegClass::SGPR, static_cast<uint16_t>(sgpr), 1u});
        }
      }
      const auto hole_unavailable = [&](uint16_t base) {
        for (uint16_t index = 0; index < scalar_count; ++index) {
          if (scalar_references.references.contains(
                  {RegClass::SGPR, static_cast<uint16_t>(base + index), 1u})) {
            return true;
          }
        }
        if ((scalar_options.moi_exec_save_sgpr &&
             range_overlaps(base, scalar_count, *scalar_options.moi_exec_save_sgpr,
                            moi_exec_save_sgpr_count(scalar_options, arch))) ||
            (scalar_options.moi_dispatch_id_sgpr &&
             range_overlaps(base, scalar_count, *scalar_options.moi_dispatch_id_sgpr,
                            moi_dispatch_state_sgpr_count(scalar_options))) ||
            (scalar_options.moi_owner_sgpr &&
             range_overlaps(base, scalar_count, *scalar_options.moi_owner_sgpr, 1u)) ||
            (scalar_options.moi_record_replay_dispatch_key_sgpr &&
             range_overlaps(base, scalar_count, *scalar_options.moi_record_replay_dispatch_key_sgpr,
                            1u)) ||
            (scalar_options.moi_record_replay_call_return_sgpr &&
             range_overlaps(base, scalar_count, *scalar_options.moi_record_replay_call_return_sgpr,
                            2u))) {
          return true;
        }
        return std::ranges::any_of(scalar_contexts,
                                   [&](const MoiKernelResourceContext *context) {
                                     return static_cast<uint32_t>(base) + scalar_count >
                                            context->current_sgpr_count;
                                   }) ||
               (resolved_contexts && moi_scalar_contexts_conflict_with_physical_vcc(
                                         resolved_contexts->summaries, base, scalar_count, arch));
      };
      std::optional<uint16_t> persistent_hole;
      // A closed KernelCfgScope includes every direct edge and every member of
      // an explicitly exhaustive recovered target set. Unrelated linked
      // kernels must not veto a safe owner-local hole, while an open control
      // transfer or dynamic scalar index still makes the proof fail closed.
      if (every_owner_resolved && scalar_references.control_flow_closed &&
          !scalar_references.uses_indirect_sgprs) {
        for (uint32_t base = 0u; base + scalar_count <= first; ++base) {
          if (!hole_unavailable(static_cast<uint16_t>(base))) {
            persistent_hole = static_cast<uint16_t>(base);
            break;
          }
        }
      }
      if (persistent_hole) {
        std::vector<ConSanMoiPrologueScratchVgprAssignment> prologue_scratch;
        // Resource plans are rebuilt after persistent-state selection. Keep
        // the same placeholder used by the ordinary scalar fallback for every
        // kernel that the rebuild can newly admit. Record/Replay initializes
        // its persistent values directly in SGPRs, so the shared prologue API
        // does not actually clobber this placeholder VGPR pair.
        for (const ConSanKernelInfo &kernel : result.kernels) {
          if (kernel.has_text_range)
            prologue_scratch.push_back({kernel.descriptor_file_offset, 0u});
        }
        assign_scalar_persistent_state(*persistent_hole);
        options.moi_init_owner_epoch = true;
        options.automatic_moi_persistent_sgprs = true;
        options.moi_prologue_scratch_vgpr_assignments = std::move(prologue_scratch);
        publish_scalar_persistent_state();
        result.moi_persistent_sgprs_automatic = true;
        result.resolved_moi_prologue_scratch_vgpr_assignments =
            options.moi_prologue_scratch_vgpr_assignments;
        result.warnings.emplace_back(
            "ConSan MOI assigned persistent scalar " +
            std::string(has_explicit_owner_epoch ? "" : "owner/epoch") +
            std::string(needs_workgroup_key ? "/workgroup-key" : "") +
            std::string(needs_record_replay_workgroup_tuple ? "/exact-workgroup-tuple" : "") +
            " state to a proven-unused CDNA SGPR window below the "
            "maximum guest reference" +
            std::string(used_complete_text_summary ? " using complete-text coverage" : ""));
        return true;
      }
      // Some full-pressure CDNA kernels leave independently proven holes in
      // both register files without leaving five consecutive registers in
      // either one. Owner and epoch are vector-uniform values, so retaining
      // them in an unused ordinary VGPR pair while retaining the exact
      // workgroup tuple in an unused SGPR triple is semantically identical to
      // either homogeneous representation. Require whole-owner reference and
      // indirect-index closure for both files; this is not a calling-
      // convention or preserved-register assumption.
      if (!has_explicit_owner_epoch && needs_record_replay_workgroup_tuple &&
          every_owner_resolved && scalar_references.control_flow_closed &&
          !scalar_references.uses_indirect_sgprs && !scalar_references.uses_indirect_vgprs) {
        constexpr uint16_t kWorkgroupScalarCount = 3u;
        constexpr uint16_t kOwnerEpochVectorCount = 2u;
        const auto scalar_tuple_unavailable = [&](uint16_t base) {
          for (uint16_t index = 0u; index < kWorkgroupScalarCount; ++index) {
            if (scalar_references.references.contains(
                    {RegClass::SGPR, static_cast<uint16_t>(base + index), 1u})) {
              return true;
            }
          }
          if ((scalar_options.moi_exec_save_sgpr &&
               range_overlaps(base, kWorkgroupScalarCount, *scalar_options.moi_exec_save_sgpr,
                              moi_exec_save_sgpr_count(scalar_options, arch))) ||
              (scalar_options.moi_dispatch_id_sgpr &&
               range_overlaps(base, kWorkgroupScalarCount, *scalar_options.moi_dispatch_id_sgpr,
                              moi_dispatch_state_sgpr_count(scalar_options))) ||
              (scalar_options.moi_owner_sgpr &&
               range_overlaps(base, kWorkgroupScalarCount, *scalar_options.moi_owner_sgpr, 1u)) ||
              (scalar_options.moi_record_replay_dispatch_key_sgpr &&
               range_overlaps(base, kWorkgroupScalarCount,
                              *scalar_options.moi_record_replay_dispatch_key_sgpr, 1u)) ||
              (scalar_options.moi_record_replay_call_return_sgpr &&
               range_overlaps(base, kWorkgroupScalarCount,
                              *scalar_options.moi_record_replay_call_return_sgpr, 2u)) ||
              persistent_sgpr_range_overlaps_reserved_ordinary_range(base, kWorkgroupScalarCount,
                                                                     arch)) {
            return true;
          }
          return resolved_contexts &&
                 moi_scalar_contexts_conflict_with_physical_vcc(resolved_contexts->summaries, base,
                                                                kWorkgroupScalarCount, arch);
        };
        std::optional<uint16_t> workgroup_scalar_base;
        for (uint32_t base = 0u; !workgroup_scalar_base &&
                                 base + kWorkgroupScalarCount <= moi_ordinary_sgpr_limit(arch);
             ++base) {
          if (!scalar_tuple_unavailable(static_cast<uint16_t>(base)))
            workgroup_scalar_base = static_cast<uint16_t>(base);
        }

        uint16_t common_ordinary_vgpr_limit = kMaxVgprs;
        for (const MoiKernelResourceContext *context : scalar_contexts)
          common_ordinary_vgpr_limit =
              std::min(common_ordinary_vgpr_limit, context->ordinary_vgpr_limit);
        // Automatic site-local scratch plans are provisional here and are
        // rebuilt after persistent placement, at which point the allocator
        // excludes this selected pair. A caller-pinned scratch base cannot
        // move during that rebuild, so exclude its widest supported window
        // now.
        const auto vector_pair_unavailable = [&](uint16_t base) {
          for (uint16_t index = 0u; index < kOwnerEpochVectorCount; ++index) {
            if (scalar_references.references.contains(
                    {RegClass::VGPR, static_cast<uint16_t>(base + index), 1u})) {
              return true;
            }
          }
          if (!options.scratch_vgpr)
            return false;
          const uint16_t fixed_scratch_count =
              std::accumulate(result.resource_plans.begin(), result.resource_plans.end(),
                              uint16_t{0}, [](uint16_t count, const auto &plan) {
                                return plan.source == ConSanRegisterAllocationSource::Unsupported
                                           ? count
                                           : std::max(count, plan.scratch_vgpr_count);
                              });
          return range_overlaps(base, kOwnerEpochVectorCount, *options.scratch_vgpr,
                                fixed_scratch_count);
        };
        std::optional<uint16_t> owner_epoch_vector_base;
        for (uint32_t base = 0u; !owner_epoch_vector_base &&
                                 base + kOwnerEpochVectorCount <= common_ordinary_vgpr_limit;
             ++base) {
          if (!vector_pair_unavailable(static_cast<uint16_t>(base)))
            owner_epoch_vector_base = static_cast<uint16_t>(base);
        }

        if (workgroup_scalar_base && owner_epoch_vector_base) {
          const uint16_t scalar_base = *workgroup_scalar_base;
          const uint16_t vector_base = *owner_epoch_vector_base;
          options.moi_owner_vgpr = vector_base;
          options.moi_epoch_vgpr = static_cast<uint16_t>(vector_base + 1u);
          options.moi_record_replay_workgroup_sgprs = {
              .x = scalar_base,
              .y = static_cast<uint16_t>(scalar_base + 1u),
              .z = static_cast<uint16_t>(scalar_base + 2u),
          };
          options.moi_init_owner_epoch = true;
          options.automatic_moi_persistent_vgprs = true;
          options.moi_prologue_scratch_vgpr_assignments.clear();
          result.resolved_moi_owner_vgpr = options.moi_owner_vgpr;
          result.resolved_moi_epoch_vgpr = options.moi_epoch_vgpr;
          result.resolved_moi_record_replay_workgroup_sgprs =
              options.moi_record_replay_workgroup_sgprs;
          result.moi_persistent_vgprs_automatic = true;
          result.moi_persistent_sgprs_automatic = true;
          result.resolved_moi_prologue_scratch_vgpr_assignments.clear();
          result.warnings.emplace_back(
              "ConSan MOI assigned hybrid persistent vector owner/epoch and scalar "
              "exact-workgroup-tuple state to proven-unused CDNA register holes" +
              std::string(used_complete_text_summary ? " using complete-text coverage" : ""));
          return true;
        }
      }
      if (!scalar_state_valid) {
        result.warnings.emplace_back(
            "ConSan MOI could not build owner scopes for a proven-unused persistent SGPR window");
      } else if (!every_owner_resolved) {
        result.warnings.emplace_back(
            "ConSan MOI could not resolve every owner descriptor for a proven-unused persistent "
            "SGPR window");
      } else if (!scalar_references.control_flow_closed) {
        std::string control_description;
        switch (scalar_references.first_unclosed_control_kind) {
        case MoiUnclosedControlKind::Call:
          control_description = "call ";
          break;
        case MoiUnclosedControlKind::Branch:
          control_description = "branch ";
          break;
        case MoiUnclosedControlKind::Fallthrough:
          control_description = "fallthrough after ";
          break;
        }
        result.warnings.emplace_back(
            "ConSan MOI cannot prove a persistent SGPR hole across unresolved guest " +
            control_description + scalar_references.first_unclosed_control_mnemonic +
            " at text offset " +
            std::to_string(scalar_references.first_unclosed_control_offset.value_or(0u)));
      } else if (scalar_references.uses_indirect_sgprs) {
        result.warnings.emplace_back(
            "ConSan MOI cannot prove a persistent SGPR hole across scalar-relative guest access");
      } else {
        result.warnings.emplace_back(
            "ConSan MOI found no persistent SGPR window unused by every admitted owner");
      }
      if (!has_dynamic_stack_owner) {
        options.moi_init_owner_epoch = true;
        options.automatic_moi_private_epoch = true;
        result.moi_private_epoch_automatic = true;
        result.warnings.emplace_back(
            "ConSan MOI retained exact Record/Replay entry state in private storage below a "
            "CDNA AccVGPR boundary");
        return true;
      }
      result.outcome = ConSanTransformOutcome::Unsupported;
      result.warnings.emplace_back(
          "ConSan MOI cannot place persistent scalar Record/Replay state below a CDNA "
          "AccVGPR boundary for a dynamic-stack owner");
      return false;
    }

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

    std::string scalar_persistence_failure;
    const auto try_assign_scalar_persistent_state = [&] {
      // Owner, epoch, and the optional workgroup key are uniform within a
      // wave. A fresh SGPR tail preserves them across arbitrary guest code
      // without moving ACCUM_OFFSET or adding a private-memory transaction at
      // every barrier. Prove the choice across the complete owner union: this
      // is a code-object-wide representation, not a component-local shortcut.
      const uint16_t scalar_count = static_cast<uint16_t>(named_persistent_count);
      const std::vector<uint64_t> scalar_owners = moi_scalar_owner_descriptors(result);
      MoiResourcePlanningState state(bytes, arch, options, result);
      const auto resolved_contexts = resolve_moi_scalar_owner_contexts(state, scalar_owners);
      if (!resolved_contexts) {
        scalar_persistence_failure = "could not resolve every scalar-state owner descriptor";
        return false;
      }
      auto prologue_scratch = plan_moi_scalar_prologue_scratch(state, *resolved_contexts,
                                                               /*needs_entry_vgpr_pair=*/true);
      if (!prologue_scratch) {
        scalar_persistence_failure =
            "could not place entry-local VGPR scratch for scalar persistent state";
        return false;
      }
      // This state crosses arbitrary guest code. Closed direct-only owners may
      // reuse descriptor padding above their highest named operand. Owners
      // with scalar-relative access or open reference coverage start beyond
      // their original allocation. The candidate filter separately preserves
      // physical VCC before and after descriptor growth.
      const uint32_t first = resolved_contexts->tail_floor;
      const uint16_t scalar_limit = moi_ordinary_sgpr_limit(arch);
      const auto unavailable = [&](uint16_t base) {
        if ((options.moi_exec_save_sgpr &&
             range_overlaps(base, scalar_count, *options.moi_exec_save_sgpr,
                            moi_exec_save_sgpr_count(options, arch))) ||
            (options.moi_dispatch_id_sgpr &&
             range_overlaps(base, scalar_count, *options.moi_dispatch_id_sgpr,
                            moi_dispatch_state_sgpr_count(options))) ||
            (options.moi_owner_sgpr &&
             range_overlaps(base, scalar_count, *options.moi_owner_sgpr, 1u)) ||
            moi_sgpr_range_overlaps_dynamic_stack_state(base, scalar_count)) {
          return true;
        }
        return moi_scalar_contexts_conflict_with_physical_vcc(resolved_contexts->summaries, base,
                                                              scalar_count, arch);
      };
      for (uint32_t base = first; base + scalar_count <= scalar_limit; ++base) {
        if (unavailable(static_cast<uint16_t>(base)))
          continue;
        // A prior disconnected component may already have accumulated a VGPR
        // tuple. Commit the scalar transition only after entry scratch has
        // been proven for the complete union.
        options.moi_persistent_vgpr_assignments.clear();
        assign_scalar_persistent_state(static_cast<uint16_t>(base));
        options.moi_init_owner_epoch = true;
        options.automatic_moi_persistent_sgprs = true;
        options.moi_prologue_scratch_vgpr_assignments = std::move(*prologue_scratch);
        publish_scalar_persistent_state();
        result.moi_persistent_sgprs_automatic = true;
        result.resolved_moi_prologue_scratch_vgpr_assignments =
            options.moi_prologue_scratch_vgpr_assignments;
        result.warnings.emplace_back("ConSan MOI assigned persistent scalar owner/epoch" +
                                     std::string(needs_workgroup_key ? "/workgroup-key" : "") +
                                     " state s" + std::to_string(base) + ":s" +
                                     std::to_string(base + scalar_count - 1u) +
                                     " below a full CDNA ordinary VGPR bank for " +
                                     consan_moi_engine_name(options.moi_engine));
        return true;
      }
      scalar_persistence_failure =
          "found no persistent scalar tail outside guest and instrumentation state";
      return false;
    };

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
        if ((options.moi_engine == ConSanMoiEngine::InlineShadow ||
             options.moi_engine == ConSanMoiEngine::Sampled) &&
            !options.force_private_epoch && try_assign_scalar_persistent_state()) {
          return true;
        }
        if ((options.moi_engine == ConSanMoiEngine::InlineShadow ||
             options.moi_engine == ConSanMoiEngine::Sampled) &&
            has_dynamic_stack_owner) {
          result.outcome = ConSanTransformOutcome::Unsupported;
          result.warnings.emplace_back(
              "ConSan MOI cannot place persistent scalar state (" + scalar_persistence_failure +
              ") for a dynamic-stack CDNA owner with a full ordinary VGPR bank");
          return false;
        }
        if (options.moi_engine == ConSanMoiEngine::InlineShadow && !has_dynamic_stack_owner) {
          options.moi_persistent_vgpr_assignments.clear();
          options.moi_init_owner_epoch = true;
          options.automatic_moi_private_epoch = true;
          result.moi_private_epoch_automatic = true;
          result.warnings.emplace_back(
              "ConSan MOI selected entry-snapshotted private owner/epoch state below a full "
              "CDNA ordinary VGPR bank after scalar placement " +
              (scalar_persistence_failure.empty() ? std::string("was not requested")
                                                  : scalar_persistence_failure));
          return true;
        }
        result.outcome = ConSanTransformOutcome::Unsupported;
        if (options.moi_engine == ConSanMoiEngine::Sampled && !scalar_persistence_failure.empty()) {
          result.warnings.emplace_back(
              "ConSan MOI cannot place persistent scalar state (" + scalar_persistence_failure +
              ") or persistent VGPR state below a connected CDNA AccVGPR boundary");
        } else {
          result.warnings.emplace_back(
              "ConSan MOI cannot place persistent VGPR state below a connected CDNA AccVGPR "
              "boundary");
        }
        return false;
      }
      for (uint64_t descriptor_offset : component) {
        ConSanMoiPersistentVgprAssignment assignment;
        assignment.descriptor_file_offset = descriptor_offset;
        uint16_t next = static_cast<uint16_t>(component_base);
        if (has_explicit_owner_epoch) {
          assignment.owner_vgpr = *options.moi_owner_vgpr;
          assignment.epoch_vgpr = *options.moi_epoch_vgpr;
        } else {
          assignment.owner_vgpr = next++;
          assignment.epoch_vgpr = next++;
        }
        if (needs_workgroup_key)
          assignment.workgroup_key_vgpr = next++;
        if (needs_record_replay_workgroup_tuple) {
          assignment.record_replay_workgroup_vgprs = {
              .x = next++,
              .y = next++,
              .z = next++,
          };
        }
        options.moi_persistent_vgpr_assignments.push_back(assignment);
      }
    }
    options.moi_init_owner_epoch = true;
    options.automatic_moi_persistent_vgprs = true;
    result.resolved_moi_persistent_vgpr_assignments = options.moi_persistent_vgpr_assignments;
    result.moi_persistent_vgprs_automatic = true;
    result.warnings.emplace_back(
        "ConSan MOI assigned per-owner persistent VGPR tuples below CDNA AccVGPR boundaries");
    return true;
  }
  const bool would_grow_descriptor =
      std::ranges::any_of(result.resource_plans, [&](const ConSanCandidateResourcePlan &plan) {
        return !plan.owner_descriptor_file_offsets.empty() &&
               plan.source != ConSanRegisterAllocationSource::Unsupported &&
               persistent_end > plan.current_vgpr_count;
      });
  const bool has_dynamic_stack_owner = has_dynamic_stack_resource_owner();
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
  const bool record_replay_private_tuple_supported =
      needs_record_replay_workgroup_tuple && !record_replay_uses_automatic_banked_capture(options);
  const bool prefer_private_epoch =
      !has_dynamic_stack_owner && !prefer_compact_record_replay_barriers && !needs_workgroup_key &&
      (!needs_record_replay_workgroup_tuple || record_replay_private_tuple_supported) &&
      !sampled_requires_persistent_owner &&
      (has_dynamic_lds_owner ||
       (would_grow_descriptor && options.moi_engine != ConSanMoiEngine::InlineShadow));
  const bool rdna4_needs_scalar_epoch =
      is_rdna4_family_arch(arch) && !options.force_private_epoch &&
      (options.moi_engine == ConSanMoiEngine::RecordReplay ||
       (options.moi_engine == ConSanMoiEngine::Sampled && has_dynamic_stack_owner)) &&
      (persistent_end > kMaxVgprs || prefer_private_epoch);
  if (rdna4_needs_scalar_epoch) {
    if (needs_persistent_dispatch_vgprs) {
      result.outcome = ConSanTransformOutcome::Unsupported;
      result.warnings.emplace_back(
          "ConSan MOI cannot retain automatic Record/Replay dispatch identity without a "
          "persistent VGPR pair");
      return false;
    }
    const uint16_t scalar_count = static_cast<uint16_t>(named_persistent_count);
    MoiResourcePlanningState scalar_state(bytes, arch, options, result);
    const std::vector<uint64_t> scalar_owners = moi_scalar_owner_descriptors(result);
    const auto resolved_contexts = resolve_moi_scalar_owner_contexts(scalar_state, scalar_owners);
    if (!resolved_contexts) {
      if (has_dynamic_stack_owner) {
        result.outcome = ConSanTransformOutcome::Unsupported;
        result.warnings.emplace_back(
            "ConSan MOI could not resolve every RDNA4 dynamic-stack scalar-state owner");
        return false;
      }
      result.warnings.emplace_back("ConSan MOI could not resolve every RDNA4 scalar-state owner");
      return false;
    }
    auto prologue_scratch = plan_moi_scalar_prologue_scratch(
        scalar_state, *resolved_contexts,
        /*needs_entry_vgpr_pair=*/options.moi_engine != ConSanMoiEngine::RecordReplay);
    if (!prologue_scratch) {
      if (has_dynamic_stack_owner)
        result.outcome = ConSanTransformOutcome::Unsupported;
      result.warnings.emplace_back(
          "ConSan MOI could not place entry-local VGPR scratch for RDNA4 scalar persistent "
          "state");
      return false;
    }
    MoiOwnerRegisterReferenceSummary scalar_references =
        summarize_moi_owner_register_references(resolved_contexts->contexts);
    if (!scalar_references.control_flow_closed) {
      if (auto complete = summarize_moi_complete_text_register_references(scalar_state))
        scalar_references = std::move(*complete);
    }
    const auto overlaps_reserved_state = [&](uint16_t base) {
      return (options.moi_exec_save_sgpr &&
              range_overlaps(base, scalar_count, *options.moi_exec_save_sgpr,
                             moi_exec_save_sgpr_count(options, arch))) ||
             (options.moi_dispatch_id_sgpr &&
              range_overlaps(base, scalar_count, *options.moi_dispatch_id_sgpr,
                             moi_dispatch_state_sgpr_count(options))) ||
             (options.moi_owner_sgpr &&
              range_overlaps(base, scalar_count, *options.moi_owner_sgpr, 1u)) ||
             (has_dynamic_stack_owner &&
              (range_overlaps(base, scalar_count, kDynamicStackTopSgpr, 1u) ||
               range_overlaps(base, scalar_count, kDynamicFrameBaseSgpr, 1u))) ||
             persistent_sgpr_range_overlaps_reserved_ordinary_range(base, scalar_count, arch);
    };
    const auto referenced = [&](uint16_t base) {
      for (uint16_t index = 0; index < scalar_count; ++index) {
        if (scalar_references.references.contains(
                {RegClass::SGPR, static_cast<uint16_t>(base + index), 1u})) {
          return true;
        }
      }
      return false;
    };
    const auto admissible = [&](uint16_t base) {
      if (overlaps_reserved_state(base))
        return false;
      if (base >= resolved_contexts->tail_floor)
        return true;
      return scalar_references.control_flow_closed && !scalar_references.uses_indirect_sgprs &&
             !referenced(base);
    };
    std::optional<uint16_t> scalar_base;
    const uint16_t scalar_limit = moi_ordinary_sgpr_limit(arch);
    for (uint32_t base = resolved_contexts->tail_floor;
         !scalar_base && base + scalar_count <= scalar_limit; ++base) {
      if (admissible(static_cast<uint16_t>(base)))
        scalar_base = static_cast<uint16_t>(base);
    }
    const uint32_t hole_limit = std::min<uint32_t>(resolved_contexts->tail_floor, scalar_limit);
    for (uint32_t base = 0u;
         has_dynamic_stack_owner && !scalar_base && base + scalar_count <= hole_limit; ++base) {
      if (admissible(static_cast<uint16_t>(base)))
        scalar_base = static_cast<uint16_t>(base);
    }
    if (scalar_base) {
      const uint16_t base = *scalar_base;
      assign_scalar_persistent_state(base);
      options.moi_init_owner_epoch = true;
      options.automatic_moi_persistent_sgprs = true;
      options.moi_prologue_scratch_vgpr_assignments = std::move(*prologue_scratch);
      publish_scalar_persistent_state();
      result.moi_persistent_sgprs_automatic = true;
      result.resolved_moi_prologue_scratch_vgpr_assignments =
          options.moi_prologue_scratch_vgpr_assignments;
      result.warnings.emplace_back(
          "ConSan MOI assigned persistent scalar " +
          std::string(has_explicit_owner_epoch ? "" : "owner/epoch") +
          std::string(needs_workgroup_key ? "/workgroup-key" : "") +
          std::string(needs_record_replay_workgroup_tuple ? "/exact-workgroup-tuple" : "") +
          " state for an RDNA4 " + consan_moi_engine_name(options.moi_engine) +
          " owner without growing the VGPR allocation");
      return true;
    }
    if (has_dynamic_stack_owner) {
      result.outcome = ConSanTransformOutcome::Unsupported;
      result.warnings.emplace_back(
          std::string("ConSan MOI could not place persistent scalar owner/epoch state for an "
                      "RDNA4 dynamic-stack ") +
          consan_moi_engine_name(options.moi_engine) + " owner");
      return false;
    }
    if (needs_record_replay_workgroup_tuple) {
      result.warnings.emplace_back(
          "ConSan MOI could not place the exact Record/Replay workgroup tuple in persistent "
          "RDNA4 scalar state; retaining it in entry-captured private state");
    }
    result.warnings.emplace_back(
        std::string(
            "ConSan MOI could not place persistent scalar owner/epoch state for an RDNA4 ") +
        consan_moi_engine_name(options.moi_engine) +
        " owner; retaining the existing private-state fallback");
  }
  const bool needs_private_epoch =
      options.force_private_epoch || persistent_end > kMaxVgprs || prefer_private_epoch;
  if (needs_private_epoch && !has_dynamic_stack_owner) {
    options.moi_init_owner_epoch = true;
    options.automatic_moi_private_epoch = true;
    result.moi_private_epoch_automatic = true;
    result.warnings.emplace_back(
        std::string("ConSan MOI automatically selected derived-owner/private-"
                    "epoch persistent state") +
        (would_grow_descriptor ? " to preserve the guest VGPR allocation" : ""));
    return true;
  }
  if (needs_private_epoch) {
    result.outcome = ConSanTransformOutcome::Unsupported;
    result.warnings.emplace_back(
        "ConSan MOI cannot retain exact entry state for a dynamic-stack owner without a "
        "persistent register window");
    return false;
  }

  uint16_t next_persistent_vgpr = static_cast<uint16_t>(persistent_base);
  if (!has_explicit_owner_epoch) {
    options.moi_owner_vgpr = next_persistent_vgpr++;
    options.moi_epoch_vgpr = next_persistent_vgpr++;
  }
  if (needs_workgroup_key)
    options.moi_workgroup_key_vgpr = next_persistent_vgpr++;
  if (needs_record_replay_workgroup_tuple) {
    options.moi_record_replay_workgroup_vgprs = {
        .x = next_persistent_vgpr++,
        .y = next_persistent_vgpr++,
        .z = next_persistent_vgpr++,
    };
  }
  if (needs_persistent_dispatch_vgprs) {
    options.moi_dispatch_id_vgpr = next_persistent_vgpr;
    next_persistent_vgpr = static_cast<uint16_t>(next_persistent_vgpr + 2u);
    options.automatic_moi_dispatch_id_vgprs = true;
  }
  options.moi_init_owner_epoch = true;
  options.automatic_moi_persistent_vgprs = true;
  result.resolved_moi_owner_vgpr = options.moi_owner_vgpr;
  result.resolved_moi_epoch_vgpr = options.moi_epoch_vgpr;
  result.resolved_moi_workgroup_key_vgpr = options.moi_workgroup_key_vgpr;
  result.resolved_moi_record_replay_workgroup_vgprs = options.moi_record_replay_workgroup_vgprs;
  result.resolved_moi_dispatch_id_vgpr = options.moi_dispatch_id_vgpr;
  result.moi_persistent_vgprs_automatic = true;
  result.moi_dispatch_id_vgprs_automatic = options.automatic_moi_dispatch_id_vgprs;
  result.warnings.emplace_back(
      "ConSan MOI automatically assigned persistent " +
      std::string(has_explicit_owner_epoch ? "" : "owner/epoch") +
      std::string(needs_workgroup_key ? "/workgroup-key" : "") +
      std::string(needs_record_replay_workgroup_tuple ? "/exact-workgroup-tuple" : "") +
      std::string(needs_persistent_dispatch_vgprs ? "/dispatch-ID" : "") + " VGPRs v" +
      std::to_string(persistent_base) + ":v" + std::to_string(next_persistent_vgpr - 1u) +
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
  if (options.moi_inline_dynamic_stack_borrowed_sgpr)
    required_count = std::max<uint16_t>(
        required_count,
        static_cast<uint16_t>(*options.moi_inline_dynamic_stack_borrowed_sgpr + 2u));
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

void note_dynamic_stack_private_requirement(ConSanPatchInfo &info, const VgprSpillSequence *spill) {
  if (spill == nullptr || !spill->uses_dynamic_stack_frame)
    return;
  info.dynamic_private_segment_addend =
      std::max(info.dynamic_private_segment_addend, spill->dynamic_frame_bytes);
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
