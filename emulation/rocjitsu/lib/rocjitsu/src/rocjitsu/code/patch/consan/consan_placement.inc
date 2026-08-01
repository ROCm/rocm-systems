struct InPlaceNopSite {
  uint64_t text_offset = 0;
  uint64_t file_offset = 0;
};

struct InPlaceInstructionSite {
  uint64_t text_offset = 0;
  uint64_t file_offset = 0;
  uint32_t size = 0;
};

struct BarrierSite {
  uint64_t text_offset = 0;
  uint64_t file_offset = 0;
  uint32_t size = 0;
  std::string range_name;
  std::string mnemonic;
};

struct TextRange {
  uint64_t begin = 0;
  uint64_t end = 0;
};

struct LocalNopCave {
  uint64_t text_offset = 0;
  uint64_t file_offset = 0;
  uint32_t word_count = 0;
  uint64_t owner_entry_text_offset = 0;
};

struct LocalCaveOwnerRange {
  uint64_t entry = 0;
  uint64_t code_end = 0;
  uint64_t ownership_end = 0;
};

void add_covered_text_range(std::vector<TextRange> &ranges, uint64_t begin, uint64_t size,
                            uint64_t text_size) {
  if (size == 0 || begin >= text_size)
    return;
  const uint64_t end = std::min<uint64_t>(text_size, begin + std::min(size, text_size - begin));
  if (begin < end)
    ranges.push_back({begin, end});
}

[[nodiscard]] std::vector<TextRange> covered_text_ranges(const ConSanResult &result,
                                                         uint64_t text_size) {
  std::vector<TextRange> ranges;
  ranges.reserve(result.kernels.size() + result.functions.size());
  for (const ConSanKernelInfo &kernel : result.kernels) {
    if (kernel.has_text_range)
      add_covered_text_range(ranges, kernel.entry_text_offset, kernel.code_size, text_size);
  }
  for (const ConSanFunctionInfo &function : result.functions)
    add_covered_text_range(ranges, function.entry_text_offset, function.code_size, text_size);

  std::ranges::sort(ranges, [](const TextRange &lhs, const TextRange &rhs) {
    if (lhs.begin != rhs.begin)
      return lhs.begin < rhs.begin;
    return lhs.end < rhs.end;
  });

  std::vector<TextRange> merged;
  for (const TextRange &range : ranges) {
    if (merged.empty() || range.begin > merged.back().end) {
      merged.push_back(range);
    } else {
      merged.back().end = std::max(merged.back().end, range.end);
    }
  }
  return merged;
}

[[nodiscard]] bool text_offset_is_covered(uint64_t offset, std::span<const TextRange> ranges) {
  for (const TextRange &range : ranges) {
    if (offset < range.begin)
      return false;
    if (offset < range.end)
      return true;
  }
  return false;
}

[[nodiscard]] std::vector<LocalCaveOwnerRange> local_cave_owner_ranges(const ConSanResult &result,
                                                                       uint64_t text_size) {
  std::vector<LocalCaveOwnerRange> owners;
  owners.reserve(result.kernels.size() + result.functions.size());
  auto add_owner = [&](uint64_t entry, uint64_t code_size) {
    if (code_size == 0 || entry >= text_size)
      return;
    const uint64_t code_end = entry + std::min(code_size, text_size - entry);
    owners.push_back({entry, code_end, text_size});
  };
  for (const ConSanKernelInfo &kernel : result.kernels) {
    if (kernel.has_text_range)
      add_owner(kernel.entry_text_offset, kernel.code_size);
  }
  for (const ConSanFunctionInfo &function : result.functions)
    add_owner(function.entry_text_offset, function.code_size);

  std::ranges::sort(owners, [](const LocalCaveOwnerRange &lhs, const LocalCaveOwnerRange &rhs) {
    if (lhs.entry != rhs.entry)
      return lhs.entry < rhs.entry;
    return lhs.code_end < rhs.code_end;
  });
  std::vector<LocalCaveOwnerRange> unique_owners;
  for (const LocalCaveOwnerRange &owner : owners) {
    if (!unique_owners.empty() && unique_owners.back().entry == owner.entry) {
      unique_owners.back().code_end = std::max(unique_owners.back().code_end, owner.code_end);
      continue;
    }
    unique_owners.push_back(owner);
  }
  for (size_t i = 0; i + 1 < unique_owners.size(); ++i)
    unique_owners[i].ownership_end = unique_owners[i + 1].entry;
  return unique_owners;
}

[[nodiscard]] const LocalCaveOwnerRange *
owner_of_uncovered_text_range(uint64_t begin, uint64_t end,
                              std::span<const LocalCaveOwnerRange> owners) {
  for (const LocalCaveOwnerRange &owner : owners) {
    // Uncovered alignment/padding after a symbol belongs only to that symbol,
    // up to (but never across) the next kernel/function entry.
    if (begin >= owner.code_end && end <= owner.ownership_end)
      return &owner;
  }
  return nullptr;
}

[[nodiscard]] std::vector<LocalNopCave>
find_uncovered_nop_caves(const AmdGpuCodeObject &code_object, const ConSanResult &result,
                         rj_code_arch_t arch) {
  if (code_object.text_sections().size() != 1)
    return {};

  const Section *text = code_object.text_sections().front();
  const auto text_bytes =
      std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(text->data()), text->size());
  const std::vector<TextRange> ranges = covered_text_ranges(result, text->size());
  const std::vector<LocalCaveOwnerRange> owners = local_cave_owner_ranges(result, text->size());
  const uint32_t nop = build_s_nop(0, arch);

  std::vector<LocalNopCave> caves;
  for (uint64_t text_offset = 0; text_offset + sizeof(uint32_t) <= text_bytes.size();) {
    if (text_offset_is_covered(text_offset, ranges)) {
      text_offset += sizeof(uint32_t);
      continue;
    }

    uint32_t word = 0;
    std::memcpy(&word, text_bytes.data() + text_offset, sizeof(word));
    if (word != nop) {
      text_offset += sizeof(uint32_t);
      continue;
    }

    const uint64_t run_begin = text_offset;
    uint32_t run_words = 0;
    while (text_offset + sizeof(uint32_t) <= text_bytes.size() &&
           !text_offset_is_covered(text_offset, ranges)) {
      std::memcpy(&word, text_bytes.data() + text_offset, sizeof(word));
      if (word != nop)
        break;
      ++run_words;
      text_offset += sizeof(uint32_t);
    }
    const uint64_t run_end = run_begin + static_cast<uint64_t>(run_words) * sizeof(uint32_t);
    const LocalCaveOwnerRange *owner = owner_of_uncovered_text_range(run_begin, run_end, owners);
    if (owner != nullptr) {
      caves.push_back({run_begin, text->sectionOffset() + run_begin, run_words, owner->entry});
    }
  }

  return caves;
}

[[nodiscard]] std::optional<InPlaceNopSite> find_existing_nop_site(const AmdGpuCodeObject &obj,
                                                                   rj_code_arch_t arch) {
  if (obj.text_sections().size() != 1)
    return std::nullopt;

  std::unique_ptr<Decoder> decoder = Decoder::create(arch);
  if (!decoder)
    return std::nullopt;

  const Section *text = obj.text_sections().front();
  auto blocks = BasicBlock::build(obj, *decoder, arch);
  for (const auto &block : blocks) {
    uint64_t offset = block->start_offset();
    for (const Instruction &inst : block->instructions()) {
      if (inst.size() == static_cast<int>(sizeof(uint32_t)) && inst.mnemonic() == "s_nop") {
        InPlaceNopSite site;
        site.text_offset = offset;
        site.file_offset = text->sectionOffset() + offset;
        return site;
      }
      offset += static_cast<uint64_t>(inst.size());
    }
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<InPlaceInstructionSite>
find_preferred_in_place_instruction_site(const AmdGpuCodeObject &obj, rj_code_arch_t arch) {
  if (obj.text_sections().size() != 1)
    return std::nullopt;

  std::unique_ptr<Decoder> decoder = Decoder::create(arch);
  if (!decoder)
    return std::nullopt;

  const Section *text = obj.text_sections().front();
  auto blocks = BasicBlock::build(obj, *decoder, arch);
  for (const auto &block : blocks) {
    uint64_t offset = block->start_offset();
    for (const Instruction &inst : block->instructions()) {
      if (inst.size() == static_cast<int>(sizeof(uint32_t)) && is_preferred_proof_anchor(inst)) {
        InPlaceInstructionSite site;
        site.text_offset = offset;
        site.file_offset = text->sectionOffset() + offset;
        site.size = sizeof(uint32_t);
        return site;
      }
      offset += static_cast<uint64_t>(inst.size());
    }
  }
  return std::nullopt;
}

[[nodiscard]] bool rewrite_word_in_place(const AmdGpuCodeObject &code_object, uint64_t file_offset,
                                         uint32_t replacement, ConSanResult &result) {
  result.elf_bytes.assign(reinterpret_cast<const uint8_t *>(code_object.image_data()),
                          reinterpret_cast<const uint8_t *>(code_object.image_data()) +
                              code_object.image_size());
  if (file_offset > result.elf_bytes.size() ||
      sizeof(replacement) > result.elf_bytes.size() - file_offset) {
    result.errors.emplace_back("ConSan proof rewrite offset is outside ELF bytes");
    result.elf_bytes.clear();
    return false;
  }
  std::memcpy(result.elf_bytes.data() + file_offset, &replacement, sizeof(replacement));
  result.modified = true;
  return true;
}

[[nodiscard]] std::optional<uint64_t> find_first_relocatable_anchor(const AmdGpuCodeObject &obj,
                                                                    rj_code_arch_t arch,
                                                                    std::string *error_out) {
  if (obj.text_sections().empty()) {
    *error_out = "code object has no .text section";
    return std::nullopt;
  }
  if (obj.text_sections().size() > 1) {
    *error_out = "code object has multiple .text sections";
    return std::nullopt;
  }

  std::unique_ptr<Decoder> decoder = Decoder::create(arch);
  if (!decoder) {
    *error_out = "no decoder available for proof NOP patch";
    return std::nullopt;
  }

  const Section *text = obj.text_sections().front();
  const std::span<const uint8_t> text_bytes(reinterpret_cast<const uint8_t *>(text->data()),
                                            text->size());
  auto blocks = BasicBlock::build(obj, *decoder, arch);
  std::optional<uint64_t> fallback_anchor;
  for (const auto &block : blocks) {
    uint64_t offset = block->start_offset();
    uint32_t clause_remaining = 0;
    for (const Instruction &inst : block->instructions()) {
      const bool blocked_by_clause = clause_remaining > 0;
      if (clause_remaining > 0)
        --clause_remaining;
      if (is_s_clause(inst))
        clause_remaining = s_clause_following_instruction_count(inst);
      if (!blocked_by_clause && inst.size() == static_cast<int>(sizeof(uint32_t)) &&
          !has_unsafe_proof_trampoline_flags(inst) &&
          is_relocatable_anchor(inst, offset, text_bytes, arch)) {
        if (!fallback_anchor)
          fallback_anchor = offset;
        if (is_preferred_proof_anchor(inst))
          return offset;
      }
      offset += static_cast<uint64_t>(inst.size());
    }
  }

  if (fallback_anchor)
    return fallback_anchor;

  *error_out = "no relocatable instruction anchor found for proof NOP patch";
  return std::nullopt;
}

[[nodiscard]] std::optional<uint32_t> build_s_trap_word(uint16_t simm16, rj_code_arch_t arch) {
  return instrumentation::build_s_trap(simm16, arch);
}

[[nodiscard]] std::optional<uint32_t> build_s_cbranch_vccz_word(int16_t offset_dwords,
                                                                rj_code_arch_t arch) {
  return instrumentation::build_s_cbranch_vccz(offset_dwords, arch);
}

[[nodiscard]] std::optional<uint32_t> build_s_wait_dscnt_word(uint16_t count, rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4 ||
      arch == ROCJITSU_CODE_ARCH_RDNA3)
    return count == 0 ? instrumentation::build_s_wait_lds0(arch) : std::nullopt;
  if (!is_rdna4_family_arch(arch))
    return std::nullopt;
  const uint16_t opcode =
      arch == ROCJITSU_CODE_ARCH_GFX1250 ? gfx1250::kSWaitDscntSopp : rdna4::kSWaitDscntSopp;
  return build_sopp_encoding(arch, opcode, count);
}

[[nodiscard]] std::optional<uint32_t> delay_instruction_word_count(const ConSanOptions &options,
                                                                   std::vector<std::string> &errors,
                                                                   std::string_view context) {
  if (options.delay_nops == 0)
    return 0u;

  switch (options.delay_mode) {
  case ConSanDelayMode::Nop:
    return options.delay_nops;
  case ConSanDelayMode::Sleep:
    if (options.delay_nops > UINT16_MAX) {
      errors.emplace_back(std::string(context) +
                          " sleep delay immediate exceeds the 16-bit s_sleep field");
      return std::nullopt;
    }
    return 1u;
  case ConSanDelayMode::SleepVar:
    if (options.delay_var_ssrc > 255) {
      errors.emplace_back(std::string(context) +
                          " sleep_var source exceeds the 8-bit scalar source field");
      return std::nullopt;
    }
    return 1u;
  }

  errors.emplace_back(std::string(context) + " has unknown delay mode '" +
                      consan_delay_mode_name(options.delay_mode) + "'");
  return std::nullopt;
}

[[nodiscard]] bool append_delay_words(std::vector<uint32_t> &words, rj_code_arch_t arch,
                                      const ConSanOptions &options,
                                      std::vector<std::string> &errors, std::string_view context) {
  if (options.delay_nops == 0)
    return true;

  switch (options.delay_mode) {
  case ConSanDelayMode::Nop:
    for (uint32_t i = 0; i < options.delay_nops; ++i)
      words.push_back(build_s_nop(0, arch));
    return true;
  case ConSanDelayMode::Sleep:
    if (options.delay_nops > UINT16_MAX) {
      errors.emplace_back(std::string(context) +
                          " sleep delay immediate exceeds the 16-bit s_sleep field");
      return false;
    }
    words.push_back(build_s_sleep(static_cast<uint16_t>(options.delay_nops), arch));
    return true;
  case ConSanDelayMode::SleepVar:
    if (options.delay_var_ssrc > 255) {
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

[[nodiscard]] std::optional<uint32_t> build_v_cmp_ne_u32_e32_word(uint16_t src0, uint16_t vsrc1,
                                                                  rj_code_arch_t arch) {
  if (src0 > 255 || vsrc1 > 255)
    return std::nullopt;
  return instrumentation::build_v_cmp_ne_u32_vcc(vector_source_vgpr(src0), vsrc1, arch);
}

[[nodiscard]] std::optional<uint32_t> build_v_cmp_ne_u16_e32_word(uint16_t src0, uint16_t vsrc1,
                                                                  rj_code_arch_t arch) {
  if (src0 > 255 || vsrc1 > 255)
    return std::nullopt;
  return instrumentation::build_v_cmp_ne_u16_vcc(vector_source_vgpr(src0), vsrc1, arch);
}

[[nodiscard]] std::optional<uint32_t> build_ds_load_word0_from_vds_word0(uint32_t word0,
                                                                         uint32_t width_bits,
                                                                         bool two_address,
                                                                         rj_code_arch_t arch) {
  uint32_t base = 0;
  if (instrumentation::is_rdna_family_arch(arch)) {
    if (two_address) {
      switch (word0 & 0xFFFF0000u) {
      case 0xD8380000u:
        base = 0xD8DC0000u;
        break;
      case 0xD83C0000u:
        base = 0xD8E00000u;
        break;
      case 0xD9380000u:
        base = 0xD9DC0000u;
        break;
      case 0xD93C0000u:
        base = 0xD9E00000u;
        break;
      default:
        return std::nullopt;
      }
      constexpr uint32_t kDsOffsetMask = 0x0000FFFFu;
      return base | (word0 & kDsOffsetMask);
    }
    switch (width_bits) {
    case 8:
      base = 0xD8E80000u;
      break;
    case 16:
      base = 0xD8F00000u;
      break;
    case 32:
      base = 0xD8D80000u;
      break;
    case 64:
      base = 0xD9D80000u;
      break;
    case 96:
      base = 0xDBF80000u;
      break;
    case 128:
      base = 0xDBFC0000u;
      break;
    default:
      return std::nullopt;
    }
  } else if ((arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4) &&
             !two_address) {
    switch (width_bits) {
    case 8:
      base = 0xD8E80000u;
      break;
    case 32:
      base = 0xD86C0000u;
      break;
    case 64:
      base = 0xD8EC0000u;
      break;
    case 128:
      base = 0xD9FE0000u;
      break;
    default:
      return std::nullopt;
    }
  } else if ((arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4) &&
             two_address) {
    switch (word0 & 0xFFFF0000u) {
    case 0xD81C0000u:
      base = 0xD86E0000u;
      break;
    case 0xD81E0000u:
      base = 0xD8700000u;
      break;
    case 0xD89C0000u:
      base = 0xD8EE0000u;
      break;
    case 0xD89E0000u:
      base = 0xD8F00000u;
      break;
    default:
      return std::nullopt;
    }
  } else {
    return std::nullopt;
  }
  constexpr uint32_t kDsOffsetMask = 0x0000FFFFu;
  return base | (word0 & kDsOffsetMask);
}

[[nodiscard]] uint32_t build_ds_load_word1(uint16_t addr_vgpr, uint16_t dst_vgpr) {
  return static_cast<uint32_t>(addr_vgpr) | (static_cast<uint32_t>(dst_vgpr) << 24u);
}

[[nodiscard]] bool is_two_address_lds_store(const ConSanLdsSite &site) {
  return site.mnemonic == "ds_store_2addr_b32" || site.mnemonic == "ds_store_2addr_b64" ||
         site.mnemonic == "ds_store_2addr_stride64_b32" ||
         site.mnemonic == "ds_store_2addr_stride64_b64" || site.mnemonic == "ds_write2_b32" ||
         site.mnemonic == "ds_write2st64_b32" || site.mnemonic == "ds_write2_b64" ||
         site.mnemonic == "ds_write2st64_b64";
}

[[nodiscard]] bool is_two_address_lds_load(const ConSanLdsSite &site) {
  return site.mnemonic == "ds_load_2addr_b32" || site.mnemonic == "ds_load_2addr_b64" ||
         site.mnemonic == "ds_load_2addr_stride64_b32" ||
         site.mnemonic == "ds_load_2addr_stride64_b64" || site.mnemonic == "ds_read2_b32" ||
         site.mnemonic == "ds_read2st64_b32" || site.mnemonic == "ds_read2_b64" ||
         site.mnemonic == "ds_read2st64_b64";
}

[[nodiscard]] std::optional<uint32_t>
two_address_lds_element_width_bits(const ConSanLdsSite &site) {
  if (site.mnemonic.ends_with("b32"))
    return 32u;
  if (site.mnemonic.ends_with("b64"))
    return 64u;
  return std::nullopt;
}

[[nodiscard]] std::optional<uint32_t>
two_address_lds_offset_scale_bytes(const ConSanLdsSite &site) {
  if (site.mnemonic == "ds_load_2addr_b32" || site.mnemonic == "ds_store_2addr_b32" ||
      site.mnemonic == "ds_read2_b32" || site.mnemonic == "ds_write2_b32")
    return 4u;
  if (site.mnemonic == "ds_load_2addr_b64" || site.mnemonic == "ds_store_2addr_b64" ||
      site.mnemonic == "ds_read2_b64" || site.mnemonic == "ds_write2_b64")
    return 8u;
  if (site.mnemonic == "ds_load_2addr_stride64_b32" ||
      site.mnemonic == "ds_store_2addr_stride64_b32" || site.mnemonic == "ds_read2st64_b32" ||
      site.mnemonic == "ds_write2st64_b32")
    return 256u;
  if (site.mnemonic == "ds_load_2addr_stride64_b64" ||
      site.mnemonic == "ds_store_2addr_stride64_b64" || site.mnemonic == "ds_read2st64_b64" ||
      site.mnemonic == "ds_write2st64_b64")
    return 512u;
  return std::nullopt;
}

struct TwoAddressLdsByteOffsets {
  uint32_t first = 0;
  uint32_t second = 0;
};

[[nodiscard]] std::optional<TwoAddressLdsByteOffsets>
two_address_lds_byte_offsets(const ConSanLdsSite &site, uint32_t word0) {
  const auto scale = two_address_lds_offset_scale_bytes(site);
  if (!scale)
    return std::nullopt;
  return TwoAddressLdsByteOffsets{
      .first = (word0 & 0xffu) * *scale,
      .second = ((word0 >> 8u) & 0xffu) * *scale,
  };
}

[[nodiscard]] std::optional<uint16_t> lds_dword_count(const ConSanLdsSite &site) {
  const bool is_two_addr_load = is_two_address_lds_load(site);
  const bool is_two_addr_store = is_two_address_lds_store(site);
  if (site.mnemonic == "ds_load_i8" || site.mnemonic == "ds_load_u8" ||
      site.mnemonic == "ds_load_i16" || site.mnemonic == "ds_load_u16" ||
      site.mnemonic == "ds_load_u8_d16" || site.mnemonic == "ds_load_u8_d16_hi" ||
      site.mnemonic == "ds_load_i8_d16" || site.mnemonic == "ds_load_i8_d16_hi" ||
      site.mnemonic == "ds_load_u16_d16" || site.mnemonic == "ds_load_u16_d16_hi" ||
      site.mnemonic == "ds_read_u8_d16" || site.mnemonic == "ds_read_u8_d16_hi" ||
      site.mnemonic == "ds_read_i8_d16" || site.mnemonic == "ds_read_i8_d16_hi" ||
      site.mnemonic == "ds_read_u16_d16" || site.mnemonic == "ds_read_u16_d16_hi" ||
      site.mnemonic == "ds_read_i16" || site.mnemonic == "ds_read_u16" ||
      site.mnemonic == "ds_store_b8" || site.mnemonic == "ds_store_b8_d16_hi" ||
      site.mnemonic == "ds_store_b16" || site.mnemonic == "ds_store_b16_d16_hi")
    return 1;
  if (is_two_addr_load || is_two_addr_store) {
    const auto element_width = two_address_lds_element_width_bits(site);
    if (!element_width)
      return std::nullopt;
    return static_cast<uint16_t>(2u * *element_width / 32u);
  }
  if (site.width_bits == 32 || site.width_bits == 64 || site.width_bits == 96 ||
      site.width_bits == 128)
    return static_cast<uint16_t>(site.width_bits / 32u);
  return std::nullopt;
}

[[nodiscard]] bool vgpr_ranges_overlap(uint16_t lhs_base, uint16_t lhs_count, uint16_t rhs_base,
                                       uint16_t rhs_count) {
  const uint32_t lhs_end = static_cast<uint32_t>(lhs_base) + lhs_count;
  const uint32_t rhs_end = static_cast<uint32_t>(rhs_base) + rhs_count;
  return static_cast<uint32_t>(lhs_base) < rhs_end && static_cast<uint32_t>(rhs_base) < lhs_end;
}

/// CDNA3, CDNA4, and gfx1250 require every multi-VGPR tuple operand to start
/// at an even register. RDNA4 permits unaligned tuples. ConSan only admits
/// these four architecture families, so this captures the complete target
/// distinction used by its scratch planners.
[[nodiscard]] constexpr bool consan_arch_requires_even_vgpr_tuples(rj_code_arch_t arch) {
  return instrumentation::is_cdna_family_arch(arch) || arch == ROCJITSU_CODE_ARCH_GFX1250;
}

[[nodiscard]] bool lds_scratch_tuple_base_is_valid(const ConSanLdsSite &site, rj_code_arch_t arch,
                                                   uint16_t candidate) {
  const auto dword_count = lds_dword_count(site);
  if (!dword_count || *dword_count <= 1u || !consan_arch_requires_even_vgpr_tuples(arch))
    return true;

  // gfx1250's transposed 96-bit six-lane load is the sole admitted exception
  // to the target's even-tuple rule.
  return (arch == ROCJITSU_CODE_ARCH_GFX1250 && site.mnemonic == "ds_load_tr6_b96") ||
         candidate % 2u == 0u;
}

[[nodiscard]] bool lds_load_clobbers_address(const ConSanLdsSite &site) {
  const auto dword_count = lds_dword_count(site);
  if (site.kind != ConSanLdsAccessKind::Read || !site.addr_vgpr || !site.dst_vgpr || !dword_count)
    return false;
  return vgpr_ranges_overlap(*site.addr_vgpr, 1u, *site.dst_vgpr, *dword_count);
}

[[nodiscard]] bool is_forbidden_scratch_vgpr_run(const ConSanLdsSite &site, uint16_t candidate,
                                                 uint16_t required_vgprs) {
  if (site.addr_vgpr && vgpr_ranges_overlap(candidate, required_vgprs, *site.addr_vgpr, 1))
    return true;

  const uint16_t data_vgprs = lds_dword_count(site).value_or(1);
  if (site.dst_vgpr && vgpr_ranges_overlap(candidate, required_vgprs, *site.dst_vgpr, data_vgprs))
    return true;
  if (site.data_vgpr && vgpr_ranges_overlap(candidate, required_vgprs, *site.data_vgpr, data_vgprs))
    return true;
  const uint16_t data_vgprs_per_address =
      site.second_data_vgpr
          ? static_cast<uint16_t>(
                two_address_lds_element_width_bits(site).value_or(site.width_bits) / 32u)
          : data_vgprs;
  if (site.second_data_vgpr && vgpr_ranges_overlap(candidate, required_vgprs,
                                                   *site.second_data_vgpr, data_vgprs_per_address))
    return true;
  return false;
}

[[nodiscard]] uint16_t legacy_scratch_search_start(const ConSanLdsSite &site) {
  std::optional<uint32_t> first_after_operands;
  auto note_range = [&first_after_operands](std::optional<uint16_t> base, uint16_t count) {
    if (!base)
      return;
    const uint32_t end = static_cast<uint32_t>(*base) + count;
    if (!first_after_operands || end > *first_after_operands)
      first_after_operands = end;
  };

  const uint16_t data_vgprs = lds_dword_count(site).value_or(1);
  note_range(site.addr_vgpr, 1);
  note_range(site.dst_vgpr, data_vgprs);
  note_range(site.data_vgpr, data_vgprs);
  note_range(site.second_data_vgpr,
             site.second_data_vgpr
                 ? static_cast<uint16_t>(
                       two_address_lds_element_width_bits(site).value_or(site.width_bits) / 32u)
                 : data_vgprs);
  if (!first_after_operands)
    return 0;
  return *first_after_operands <= 255 ? static_cast<uint16_t>(*first_after_operands) : 256;
}

[[nodiscard]] uint16_t scratch_search_start(const ConSanLdsSite &site,
                                            std::optional<uint16_t> min_auto_scratch_vgpr) {
  return std::max(legacy_scratch_search_start(site), min_auto_scratch_vgpr.value_or(0));
}

[[nodiscard]] bool needs_scratch_headroom_at_descriptor_edge(const ConSanLdsSite &site) {
  return site.mnemonic == "ds_load_b64" || site.mnemonic == "ds_load_2addr_b64" ||
         site.mnemonic == "ds_load_2addr_stride64_b64" || site.mnemonic == "ds_read2_b64" ||
         site.mnemonic == "ds_read2st64_b64";
}

[[nodiscard]] std::optional<uint16_t>
required_descriptor_vgpr_allocation_for_scratch(const ConSanLdsSite &site, uint16_t scratch_vgpr,
                                                uint16_t required_vgprs) {
  uint32_t required_count = static_cast<uint32_t>(scratch_vgpr) + required_vgprs;
  if (needs_scratch_headroom_at_descriptor_edge(site))
    ++required_count;
  if (required_count > REGISTER_SET_MAX_VGPRS)
    return std::nullopt;
  return static_cast<uint16_t>(required_count);
}

[[nodiscard]] std::optional<uint16_t> find_liveness_scratch_vgpr(
    const ConSanLdsSite &site, const Instruction *inst, const LivenessAnalysis *liveness,
    std::optional<uint16_t> min_auto_scratch_vgpr, std::optional<uint16_t> max_auto_scratch_vgpr,
    uint16_t required_vgprs, rj_code_arch_t arch) {
  if (inst == nullptr || liveness == nullptr)
    return std::nullopt;

  uint16_t search_start = scratch_search_start(site, min_auto_scratch_vgpr);
  if (max_auto_scratch_vgpr && search_start >= *max_auto_scratch_vgpr)
    return std::nullopt;
  while (search_start < REGISTER_SET_MAX_VGPRS) {
    auto candidate = liveness->find_free_run(inst, required_vgprs, search_start);
    if (!candidate)
      return std::nullopt;
    const uint32_t candidate_end = static_cast<uint32_t>(*candidate) + required_vgprs;
    if (max_auto_scratch_vgpr && candidate_end > *max_auto_scratch_vgpr)
      return std::nullopt;
    if (max_auto_scratch_vgpr && candidate_end == *max_auto_scratch_vgpr &&
        needs_scratch_headroom_at_descriptor_edge(site))
      return std::nullopt;
    if (candidate_end <= 256 && lds_scratch_tuple_base_is_valid(site, arch, *candidate) &&
        !is_forbidden_scratch_vgpr_run(site, *candidate, required_vgprs))
      return candidate;
    if (*candidate == UINT16_MAX)
      return std::nullopt;
    search_start = static_cast<uint16_t>(*candidate + 1u);
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<uint16_t> choose_scratch_vgpr(
    const ConSanLdsSite &site, const ConSanOptions &options, const Instruction *inst,
    const LivenessAnalysis *liveness, std::optional<uint16_t> min_auto_scratch_vgpr,
    std::optional<uint16_t> max_auto_scratch_vgpr, uint16_t required_vgprs, rj_code_arch_t arch) {
  if (options.scratch_vgpr) {
    if (static_cast<uint32_t>(*options.scratch_vgpr) + required_vgprs <= 256 &&
        lds_scratch_tuple_base_is_valid(site, arch, *options.scratch_vgpr) &&
        !is_forbidden_scratch_vgpr_run(site, *options.scratch_vgpr, required_vgprs))
      return *options.scratch_vgpr;
    return std::nullopt;
  }

  if (auto scratch = find_liveness_scratch_vgpr(site, inst, liveness, min_auto_scratch_vgpr,
                                                max_auto_scratch_vgpr, required_vgprs, arch))
    return scratch;

  return std::nullopt;
}

[[nodiscard]] std::optional<uint16_t> choose_spill_scratch_vgpr(const ConSanLdsSite &site,
                                                                uint16_t allocation_count,
                                                                uint16_t required_vgprs,
                                                                rj_code_arch_t arch) {
  for (uint32_t candidate = 0;
       candidate + required_vgprs <= allocation_count && candidate + required_vgprs <= 256u;
       ++candidate) {
    if (lds_scratch_tuple_base_is_valid(site, arch, static_cast<uint16_t>(candidate)) &&
        !is_forbidden_scratch_vgpr_run(site, static_cast<uint16_t>(candidate), required_vgprs))
      return static_cast<uint16_t>(candidate);
  }
  return std::nullopt;
}

[[nodiscard]] std::vector<uint64_t> kernel_entry_offsets(const ConSanResult &result) {
  std::vector<uint64_t> offsets;
  offsets.reserve(result.kernels.size());
  for (const ConSanKernelInfo &kernel : result.kernels) {
    if (kernel.has_text_range)
      offsets.push_back(kernel.entry_text_offset);
  }
  std::ranges::sort(offsets);
  offsets.erase(std::ranges::unique(offsets).begin(), offsets.end());
  return offsets;
}

[[nodiscard]] std::vector<BasicBlock *>
block_ptrs_for(const std::vector<std::unique_ptr<BasicBlock>> &blocks) {
  std::vector<BasicBlock *> ptrs;
  ptrs.reserve(blocks.size());
  for (const auto &block : blocks) {
    if (block != nullptr)
      ptrs.push_back(block.get());
  }
  return ptrs;
}

[[nodiscard]] const Instruction *
find_instruction_at_text_offset(std::span<BasicBlock *const> blocks, uint64_t text_offset) {
  for (BasicBlock *block : blocks) {
    if (block == nullptr || text_offset < block->start_offset() ||
        text_offset >= block->end_offset())
      continue;
    for (const Instruction &inst : block->instructions()) {
      if (inst.src_loc() == text_offset)
        return &inst;
    }
  }
  return nullptr;
}

[[nodiscard]] bool text_offset_is_inside_s_clause(std::span<BasicBlock *const> blocks,
                                                  uint64_t text_offset) {
  for (BasicBlock *block : blocks) {
    if (block == nullptr || text_offset < block->start_offset() ||
        text_offset >= block->end_offset())
      continue;
    uint32_t clause_remaining = 0;
    for (const Instruction &inst : block->instructions()) {
      const bool blocked_by_clause = clause_remaining > 0;
      if (clause_remaining > 0)
        --clause_remaining;
      if (is_s_clause(inst))
        clause_remaining = s_clause_following_instruction_count(inst);
      if (inst.src_loc() == text_offset)
        return blocked_by_clause;
    }
  }
  return false;
}

void update_max_vgpr_ref(const RegisterSet &set, std::optional<uint16_t> &max_vgpr) {
  set.for_each([&](RegisterRef ref) {
    if (ref.cls != RegClass::VGPR)
      return;
    if (!max_vgpr || ref.index > *max_vgpr)
      max_vgpr = ref.index;
  });
}

void update_max_sgpr_ref(const RegisterSet &set, std::optional<uint16_t> &max_sgpr) {
  set.for_each([&](RegisterRef ref) {
    if (ref.cls != RegClass::SGPR)
      return;
    if (!max_sgpr || ref.index > *max_sgpr)
      max_sgpr = ref.index;
  });
}

struct KernelMaxRegisterRefs {
  std::optional<uint16_t> sgpr;
  std::optional<uint16_t> vgpr;
};

[[nodiscard]] KernelMaxRegisterRefs
max_register_refs_in_kernel(const ConSanKernelInfo &kernel, std::span<BasicBlock *const> blocks) {
  KernelMaxRegisterRefs max_refs;
  if (!kernel.has_text_range || kernel.code_size == 0)
    return max_refs;

  const uint64_t begin = kernel.entry_text_offset;
  const uint64_t end = begin + kernel.code_size;
  for (BasicBlock *block : blocks) {
    if (block == nullptr || block->end_offset() <= begin || block->start_offset() >= end)
      continue;
    for (const Instruction &inst : block->instructions()) {
      if (inst.src_loc() < begin || inst.src_loc() >= end)
        continue;
      InstDefUse du(inst);
      update_max_sgpr_ref(du.uses, max_refs.sgpr);
      update_max_sgpr_ref(du.defs, max_refs.sgpr);
      update_max_vgpr_ref(du.uses, max_refs.vgpr);
      update_max_vgpr_ref(du.defs, max_refs.vgpr);
    }
  }
  return max_refs;
}

[[nodiscard]] bool read_words_at(std::span<const uint8_t> bytes, uint64_t offset,
                                 std::span<uint32_t> words) {
  const uint64_t size = static_cast<uint64_t>(words.size() * sizeof(uint32_t));
  if (offset > bytes.size() || size > bytes.size() - offset)
    return false;
  std::memcpy(words.data(), bytes.data() + offset, static_cast<size_t>(size));
  return true;
}

void append_words(std::vector<uint8_t> &bytes, std::span<const uint32_t> words) {
  const size_t old_size = bytes.size();
  const size_t added_size = words.size() * sizeof(uint32_t);
  bytes.resize(old_size + added_size);
  std::memcpy(bytes.data() + old_size, words.data(), added_size);
}

[[nodiscard]] uint32_t count_nop_padding(std::span<const uint8_t> bytes, uint64_t offset,
                                         uint32_t max_word_count, rj_code_arch_t arch) {
  const uint32_t nop = build_s_nop(0, arch);
  uint32_t count = 0;
  for (; count < max_word_count; ++count) {
    uint32_t word = 0;
    if (!read_words_at(bytes, offset + count * sizeof(uint32_t), std::span<uint32_t>(&word, 1)) ||
        word != nop)
      break;
  }
  return count;
}

[[nodiscard]] bool is_rocclr_runtime_kernel_name(std::string_view name) {
  return name.starts_with("__amd_rocclr_");
}

[[nodiscard]] bool has_only_rocclr_runtime_kernels(const ConSanResult &result) {
  if (result.kernels.empty())
    return false;
  for (const ConSanKernelInfo &kernel : result.kernels) {
    if (!is_rocclr_runtime_kernel_name(kernel.name))
      return false;
  }
  return true;
}

[[nodiscard]] std::string fixed_hex(uint64_t value, unsigned digits) {
  constexpr char kHex[] = "0123456789abcdef";
  std::string result(digits, '0');
  for (unsigned i = 0; i < digits; ++i) {
    result[digits - i - 1] = kHex[value & 0xfu];
    value >>= 4u;
  }
  return result;
}

[[nodiscard]] std::string code_object_fingerprint(std::span<const uint8_t> bytes) {
  // FNV-1a is deliberately simple and stable across hosts. This is an
  // identity fingerprint, not a cryptographic integrity check.
  uint64_t hash = 14695981039346656037ull;
  for (uint8_t byte : bytes) {
    hash ^= byte;
    hash *= 1099511628211ull;
  }
  return "fnv1a64:" + fixed_hex(hash, 16);
}

void append_decoded_operand(std::string &summary, std::string_view name,
                            std::optional<int64_t> value) {
  if (!value)
    return;
  if (!summary.empty())
    summary += ',';
  summary += name;
  summary += '=';
  summary += std::to_string(*value);
}

[[nodiscard]] std::string atomic_semantic_role(const ConSanAtomicSite &site) {
  if (site.raw_th.value_or(0) != 0 || site.returns_old_value.value_or(false))
    return "atomic-acquire-rmw";
  return "atomic-release-rmw";
}

[[nodiscard]] std::string barrier_decoded_operands(const ConSanBarrierSite &site,
                                                   uint32_t encoding) {
  std::string result = "encoding=0x" + fixed_hex(encoding, 8);
  result += ",barrier_id=";
  result += site.barrier_id ? std::to_string(*site.barrier_id) : "-";
  result += ",operand_source=";
  result += consan_barrier_operand_source_name(site.operand_source);
  result += ",scope=";
  result += consan_barrier_scope_name(site.scope);
  result += ",raw_selector=";
  result += site.raw_operand_selector ? std::to_string(*site.raw_operand_selector) : "-";
  result += ",literal_width_bits=";
  result += site.literal_width_bits ? std::to_string(*site.literal_width_bits) : "-";
  result += ",literal_value=";
  result += site.literal_value ? std::to_string(*site.literal_value) : "-";
  result += ",raw_simm16=";
  result += site.raw_simm16 ? std::to_string(*site.raw_simm16) : "-";
  return result;
}

[[nodiscard]] std::string atomic_decoded_operands(const ConSanAtomicSite &site) {
  std::string result;
  append_decoded_operand(result, "dst_vgpr", site.dst_vgpr);
  append_decoded_operand(result, "addr_vgpr", site.addr_vgpr);
  append_decoded_operand(result, "data_vgpr", site.data_vgpr);
  append_decoded_operand(result, "saddr_sgpr", site.saddr_sgpr);
  append_decoded_operand(result, "raw_ioffset", site.raw_ioffset);
  append_decoded_operand(result, "raw_scope", site.raw_scope);
  append_decoded_operand(result, "raw_th", site.raw_th);
  append_decoded_operand(result, "raw_addr", site.raw_addr);
  append_decoded_operand(result, "raw_data0", site.raw_data0);
  append_decoded_operand(result, "raw_data1", site.raw_data1);
  append_decoded_operand(result, "raw_vdata", site.raw_vdata);
  append_decoded_operand(result, "raw_rsrc", site.raw_rsrc);
  append_decoded_operand(result, "raw_soffset", site.raw_soffset);
  if (site.returns_old_value)
    append_decoded_operand(result, "returns_old", *site.returns_old_value ? 1 : 0);
  return result.empty() ? "-" : result;
}

[[nodiscard]] std::string lds_decoded_operands(const ConSanLdsSite &site) {
  std::string result;
  append_decoded_operand(result, "dst_vgpr", site.dst_vgpr);
  append_decoded_operand(result, "dst_accvgpr", site.dst_accvgpr);
  append_decoded_operand(result, "addr_vgpr", site.addr_vgpr);
  append_decoded_operand(result, "data_vgpr", site.data_vgpr);
  append_decoded_operand(result, "second_data_vgpr", site.second_data_vgpr);
  return result.empty() ? "-" : result;
}

[[nodiscard]] std::string ordinary_memory_decoded_operands(const ConSanOrdinaryMemorySite &site) {
  std::string result;
  append_decoded_operand(result, "dst_vgpr", site.destination_vgpr);
  append_decoded_operand(result, "addr_vgpr", site.address_vgpr);
  append_decoded_operand(result, "addr_sgpr", site.address_sgpr);
  append_decoded_operand(result, "value_vgpr", site.value_vgpr);
  append_decoded_operand(result, "raw_saddr", site.raw_saddr);
  append_decoded_operand(result, "raw_nv", site.raw_nv);
  append_decoded_operand(result, "raw_sve", site.raw_sve);
  append_decoded_operand(result, "raw_vaddr", site.raw_vaddr);
  append_decoded_operand(result, "raw_vsrc", site.raw_vsrc);
  append_decoded_operand(result, "raw_vdst", site.raw_vdst);
  append_decoded_operand(result, "raw_ioffset", site.raw_ioffset);
  append_decoded_operand(result, "raw_scope", site.raw_scope);
  append_decoded_operand(result, "raw_th", site.raw_th);
  return result.empty() ? "-" : result;
}
