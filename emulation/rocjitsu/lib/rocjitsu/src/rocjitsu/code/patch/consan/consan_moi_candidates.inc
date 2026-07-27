namespace {

inline constexpr uint16_t kRdna4ExecLo = 126;
inline constexpr uint16_t kRdna4ExecHi = 127;
inline constexpr uint16_t kRdna4VccLo = 106;
inline constexpr uint16_t kRdna4VccHi = 107;
inline constexpr uint16_t kRdna4WorkitemIdX = 0;
// Architectural scalar-source encoding for the inline constant -1.
inline constexpr uint16_t kScalarInlineNegativeOneOperand = 193;
inline constexpr uint16_t kScalarOperandTtmpBase = 108;
inline constexpr uint16_t kScalarOperandSharedBase = 235;
inline constexpr uint16_t kTtmpRdna4GridYz = 7;
inline constexpr uint16_t kTtmpRdna4GridX = 9;
inline constexpr uint16_t kTtmpGfx1250ClusterWorkgroupId = 6;
inline constexpr uint32_t kMaxVgprs = 256;
inline constexpr uint32_t kMaxSgprs = 106;
inline constexpr uint16_t kGfx12HwRegHwId1 = 23;
inline constexpr uint16_t kGfx12HwIdOwnerBits = 10;
inline constexpr uint8_t kRdna4ScopeDevice = 2;
inline constexpr uint32_t kRdna4VectorFlatNoSaddr = static_cast<uint32_t>(rdna4::OPR_SREG_NULL);
inline constexpr uint32_t kGfx1250VectorFlatNoSaddr = static_cast<uint32_t>(gfx1250::OPR_SREG_NULL);

[[nodiscard]] constexpr uint32_t vector_flat_no_saddr(rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4)
    return 0;
  if (arch == ROCJITSU_CODE_ARCH_GFX1250)
    return kGfx1250VectorFlatNoSaddr;
  return kRdna4VectorFlatNoSaddr;
}

[[nodiscard]] constexpr bool is_vector_flat_no_saddr(uint32_t raw_saddr, rj_code_arch_t arch) {
  return raw_saddr == vector_flat_no_saddr(arch);
}

[[nodiscard]] std::optional<uint16_t> gfx1250_vgpr_msb_mode_at(std::span<const uint8_t> bytes,
                                                               uint64_t text_file_offset,
                                                               uint64_t container_entry_text_offset,
                                                               uint64_t site_file_offset) {
  return consan_gfx1250_vgpr_msb_mode_at(bytes, text_file_offset, container_entry_text_offset,
                                         site_file_offset);
}
[[nodiscard]] bool append_moi_flat_load_wait(std::vector<uint32_t> &words, rj_code_arch_t arch) {
  const auto wait = instrumentation::build_s_wait_flat_load0(arch);
  if (!wait)
    return false;
  words.push_back(*wait);
  return true;
}
[[nodiscard]] bool append_moi_flat_atomic_wait(std::vector<uint32_t> &words, rj_code_arch_t arch) {
  if (!append_moi_flat_load_wait(words, arch))
    return false;
  // Returning FLAT atomics have both a result dependency and a memory-side
  // effect on gfx12. LOADCNT makes the returned value consumable; STORECNT
  // makes the update visible before a later LDS shadow publication can expose
  // causally ordered work to another wave. CDNA's unified wait already drains
  // both sides.
  if (arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4)
    return true;
  const auto store_wait = instrumentation::build_s_wait_flat_store0(arch);
  if (!store_wait)
    return false;
  words.push_back(*store_wait);
  return true;
}
[[nodiscard]] bool append_moi_lds_wait(std::vector<uint32_t> &words, rj_code_arch_t arch) {
  const auto wait = instrumentation::build_s_wait_lds0(arch);
  if (!wait)
    return false;
  words.push_back(*wait);
  return true;
}
inline constexpr uint64_t kAmdhsaKernelEntryAlignment = 256;

[[nodiscard]] ConSanMoiReportBufferLayout
resolve_inline_report_layout(const ConSanOptions &options) {
  if (options.moi_report_layout) {
    return consan_moi_report_layout_from_override(
        *options.moi_report_layout, ConSanMoiEngine::InlineShadow, options.moi_report_buffer_size);
  }
  return consan_moi_inline_shadow_report_buffer_layout_for_bytes(options.moi_report_buffer_size);
}

[[nodiscard]] ConSanMoiReportBufferLayout
resolve_sampled_report_layout(const ConSanOptions &options) {
  if (options.moi_report_layout) {
    return consan_moi_report_layout_from_override(
        *options.moi_report_layout, ConSanMoiEngine::Sampled, options.moi_report_buffer_size);
  }
  return consan_moi_direct_sampled_report_buffer_layout_for_bytes(options.moi_report_buffer_size);
}

[[nodiscard]] ConSanMoiReportBufferLayout
resolve_record_replay_report_layout(const ConSanOptions &options, bool include_barriers,
                                    bool include_atomics, bool include_fences) {
  if (options.moi_report_layout) {
    return consan_moi_report_layout_from_override(
        *options.moi_report_layout, ConSanMoiEngine::RecordReplay, options.moi_report_buffer_size);
  }
  return consan_moi_report_buffer_layout_for_bytes(options.moi_report_buffer_size, include_barriers,
                                                   include_atomics, include_fences);
}

/// Automatic Record/Replay access banks promise exact dispatch/site/workgroup/
/// wave identity. Placement, scratch sizing, and emission must use this single
/// predicate so a probe can never require persistent identity that placement
/// did not reserve.
[[nodiscard]] bool record_replay_uses_automatic_banked_capture(const ConSanOptions &options,
                                                               uint32_t dispatch_token_capacity) {
  return options.moi_engine == ConSanMoiEngine::RecordReplay &&
         !options.moi_dynamic_access_records && dispatch_token_capacity != 0u;
}

[[nodiscard]] bool record_replay_uses_automatic_banked_capture(const ConSanOptions &options) {
  return options.moi_report_layout &&
         record_replay_uses_automatic_banked_capture(
             options, options.moi_report_layout->record_replay_dispatch_token_capacity);
}

[[nodiscard]] constexpr uint16_t ttmp_scalar_operand(uint16_t ttmp) {
  return static_cast<uint16_t>(kScalarOperandTtmpBase + ttmp);
}

struct ConSanMoiWorkgroupSources {
  ConSanMoiWorkgroupSource x;
  ConSanMoiWorkgroupSource y;
  ConSanMoiWorkgroupSource z;
  ConSanMoiWorkgroupSource cluster_workgroup_id;

  auto operator<=>(const ConSanMoiWorkgroupSources &) const = default;
};

struct ConSanMoiAccessRange {
  uint32_t static_byte_offset = 0;
  uint32_t byte_count = 0;
};

using KD = rocr::llvm::amdhsa::kernel_descriptor_t;
namespace kd = rocr::llvm::amdhsa;

[[nodiscard]] bool equals_any(std::string_view value,
                              std::initializer_list<std::string_view> candidates) {
  for (std::string_view candidate : candidates) {
    if (value == candidate)
      return true;
  }
  return false;
}

[[nodiscard]] std::optional<uint32_t> read_u32(std::span<const uint8_t> bytes, uint64_t offset) {
  if (offset > bytes.size() || sizeof(uint32_t) > bytes.size() - offset)
    return std::nullopt;
  uint32_t word = 0;
  std::memcpy(&word, bytes.data() + offset, sizeof(word));
  return word;
}

[[nodiscard]] std::optional<std::vector<uint32_t>>
decode_relocatable_entry_instruction(std::span<const uint8_t> text, uint64_t offset,
                                     rj_code_arch_t arch, std::vector<std::string> &errors,
                                     Decoder *reusable_decoder = nullptr) {
  std::unique_ptr<Decoder> owned_decoder;
  Decoder *decoder = reusable_decoder;
  if (decoder == nullptr) {
    owned_decoder = Decoder::create(arch);
    decoder = owned_decoder.get();
  }
  if (!decoder) {
    errors.emplace_back("ConSan MOI could not create an entry-instruction decoder");
    return std::nullopt;
  }
  const auto decode_one = [&](uint64_t instruction_offset)
      -> std::optional<std::pair<std::vector<uint32_t>, std::string>> {
    if (instruction_offset > text.size() || sizeof(uint32_t) > text.size() - instruction_offset) {
      errors.emplace_back("ConSan MOI kernel entry instruction exceeds .text");
      return std::nullopt;
    }
    std::array<uint32_t, 4> decode_words{};
    const size_t available =
        std::min<size_t>(sizeof(decode_words), text.size() - instruction_offset);
    std::memcpy(decode_words.data(), text.data() + instruction_offset, available);
    std::unique_ptr<Instruction> instruction;
    try {
      instruction.reset(decoder->decode(decode_words.data(), instruction_offset));
    } catch (const std::exception &exception) {
      errors.emplace_back("ConSan MOI could not decode the kernel entry instruction: " +
                          std::string(exception.what()));
      return std::nullopt;
    }
    if (!instruction || instruction->size() <= 0 || instruction->size() % sizeof(uint32_t) != 0 ||
        static_cast<size_t>(instruction->size()) > available) {
      errors.emplace_back("ConSan MOI kernel entry instruction has invalid decoded bounds");
      return std::nullopt;
    }
    const std::string mnemonic(instruction->mnemonic());
    if (mnemonic.starts_with("s_branch") || mnemonic.starts_with("s_cbranch") ||
        mnemonic.starts_with("s_call") || mnemonic.starts_with("s_getpc") ||
        mnemonic.starts_with("s_setpc") || mnemonic.starts_with("s_swappc")) {
      errors.emplace_back(
          "ConSan MOI dynamic-stack entry trampoline cannot relocate PC-relative control flow");
      return std::nullopt;
    }
    std::vector<uint32_t> words(static_cast<size_t>(instruction->size()) / sizeof(uint32_t));
    std::memcpy(words.data(), text.data() + instruction_offset,
                static_cast<size_t>(instruction->size()));
    return std::pair{std::move(words), mnemonic};
  };

  auto first = decode_one(offset);
  if (!first)
    return std::nullopt;
  std::vector<uint32_t> words = std::move(first->first);
  if (first->second != "s_clause")
    return words;

  // An s_clause is only valid while directly adjacent to all of the memory
  // instructions it describes. Treat that complete run as one relocation
  // unit; moving the clause alone ahead of the trampoline's return branch
  // produces an illegal shader instruction on gfx1201.
  const uint32_t following_instruction_count = (words.front() & 0xffffu) + 1u;
  uint64_t cursor = offset + words.size() * sizeof(uint32_t);
  for (uint32_t index = 0; index < following_instruction_count; ++index) {
    auto following = decode_one(cursor);
    if (!following)
      return std::nullopt;
    if (following->second == "s_clause") {
      errors.emplace_back("ConSan MOI cannot relocate a nested s_clause entry run");
      return std::nullopt;
    }
    cursor += following->first.size() * sizeof(uint32_t);
    words.insert(words.end(), following->first.begin(), following->first.end());
  }
  return words;
}

[[nodiscard]] bool is_single_range_native_lds_mnemonic(std::string_view mnemonic,
                                                       rj_code_arch_t arch);

[[nodiscard]] std::optional<uint32_t>
two_address_native_lds_offset_scale(std::string_view mnemonic);

[[nodiscard]] bool is_moi_native_lds_candidate(const ConSanLdsSite &site, rj_code_arch_t arch) {
  const bool supported_atomic =
      site.kind == ConSanLdsAccessKind::Atomic &&
      ((arch == ROCJITSU_CODE_ARCH_GFX1250 &&
        equals_any(site.mnemonic, {"ds_add_f32", "ds_add_f64", "ds_add_u32", "ds_add_u64",
                                   "ds_cmpstore_rtn_b32"})) ||
       ((arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4) &&
        equals_any(site.mnemonic,
                   {"ds_add_f32", "ds_add_f64", "ds_add_u32", "ds_add_u64", "ds_cmpst_rtn_b32"})));
  if (site.kind != ConSanLdsAccessKind::Read && site.kind != ConSanLdsAccessKind::Write &&
      !supported_atomic)
    return false;
  return is_single_range_native_lds_mnemonic(site.mnemonic, arch) ||
         two_address_native_lds_offset_scale(site.mnemonic).has_value();
}

struct SkippedMoiLdsCounts {
  uint32_t unsupported_kind = 0;
  uint32_t unsupported_mnemonic = 0;

  [[nodiscard]] bool any() const { return unsupported_kind != 0 || unsupported_mnemonic != 0; }
};

void count_skipped_moi_lds_candidate(const ConSanLdsSite &site, SkippedMoiLdsCounts &counts) {
  if (site.kind != ConSanLdsAccessKind::Read && site.kind != ConSanLdsAccessKind::Write) {
    ++counts.unsupported_kind;
    return;
  }
  ++counts.unsupported_mnemonic;
}

void warn_skipped_moi_lds_candidates(std::string_view container_name, bool in_kernel,
                                     const SkippedMoiLdsCounts &counts, ConSanResult &result) {
  if (!counts.any())
    return;
  result.warnings.emplace_back(
      "ConSan MOI skipped native LDS sites in " + std::string(in_kernel ? "kernel " : "function ") +
      std::string(container_name) +
      ": unsupported_kind=" + std::to_string(counts.unsupported_kind) +
      " unsupported_mnemonic=" + std::to_string(counts.unsupported_mnemonic));
}

[[nodiscard]] bool is_moi_flat_candidate(const ConSanFlatSite &site,
                                         ConSanFlatProvenanceMode mode) {
  if (site.kind == ConSanLdsAccessKind::Other)
    return false;
  return site.address_space_hint == ConSanFlatAddressSpaceHint::Group ||
         (mode == ConSanFlatProvenanceMode::Likely &&
          site.address_space_hint == ConSanFlatAddressSpaceHint::MaybeGroup);
}

[[nodiscard]] bool is_supported_moi_flat_access_mnemonic(std::string_view mnemonic) {
  if (consan_flat_load_subword_semantics(mnemonic) || consan_flat_store_subword_semantics(mnemonic))
    return true;
  return equals_any(mnemonic, {"flat_load_b32", "flat_load_b64", "flat_load_b128", "flat_store_b32",
                               "flat_store_b64", "flat_store_b128", "flat_load_dword",
                               "flat_load_dwordx2", "flat_load_dwordx4", "flat_store_dword",
                               "flat_store_dwordx2", "flat_store_dwordx4", "flat_load_ushort",
                               "flat_store_short", "flat_load_u16", "flat_store_b16"});
}

struct SkippedMoiFlatCounts {
  uint32_t other_kind = 0;
  uint32_t unknown = 0;
  uint32_t private_hint = 0;
  uint32_t maybe_private = 0;
  uint32_t global = 0;
  uint32_t excluded_maybe_group = 0;

  [[nodiscard]] bool any() const {
    return other_kind != 0 || unknown != 0 || private_hint != 0 || maybe_private != 0 ||
           global != 0 || excluded_maybe_group != 0;
  }
};

void count_skipped_moi_flat_candidate(const ConSanFlatSite &site, SkippedMoiFlatCounts &counts) {
  if (site.kind == ConSanLdsAccessKind::Other) {
    ++counts.other_kind;
    return;
  }

  switch (site.address_space_hint) {
  case ConSanFlatAddressSpaceHint::Unknown:
    ++counts.unknown;
    return;
  case ConSanFlatAddressSpaceHint::Private:
    ++counts.private_hint;
    return;
  case ConSanFlatAddressSpaceHint::MaybePrivate:
    ++counts.maybe_private;
    return;
  case ConSanFlatAddressSpaceHint::Global:
    ++counts.global;
    return;
  case ConSanFlatAddressSpaceHint::Group:
  case ConSanFlatAddressSpaceHint::MaybeGroup:
    return;
  }
}

void warn_skipped_moi_flat_candidates(std::string_view container_name, bool in_kernel,
                                      const SkippedMoiFlatCounts &counts, ConSanResult &result) {
  if (!counts.any())
    return;
  result.warnings.emplace_back(
      "ConSan MOI skipped flat sites in " + std::string(in_kernel ? "kernel " : "function ") +
      std::string(container_name) + " because the flat provenance policy did not admit them: " +
      "unknown=" + std::to_string(counts.unknown) +
      " private=" + std::to_string(counts.private_hint) + " maybe_private=" +
      std::to_string(counts.maybe_private) + " global=" + std::to_string(counts.global) +
      " excluded_maybe_group=" + std::to_string(counts.excluded_maybe_group) +
      " unsupported_kind=" + std::to_string(counts.other_kind));
}

[[nodiscard]] ConSanMoiCandidate make_moi_candidate(std::string container_name, bool in_kernel,
                                                    std::optional<uint64_t> descriptor_file_offset,
                                                    const ConSanLdsSite &site,
                                                    bool uses_gfx1250_cluster_workgroup_id = false,
                                                    uint64_t container_entry_text_offset = 0) {
  ConSanMoiCandidate candidate;
  candidate.container_name = std::move(container_name);
  candidate.in_kernel = in_kernel;
  candidate.container_entry_text_offset = container_entry_text_offset;
  candidate.source = ConSanMoiCandidateSource::NativeLds;
  candidate.kind = site.kind;
  candidate.text_offset = site.text_offset;
  candidate.file_offset = site.file_offset;
  candidate.size = site.size;
  candidate.width_bits = site.width_bits;
  candidate.dst_vgpr = site.dst_vgpr;
  candidate.addr_vgpr = site.addr_vgpr;
  candidate.data_vgpr = site.data_vgpr;
  candidate.second_data_vgpr = site.second_data_vgpr;
  candidate.kernel_descriptor_file_offset = descriptor_file_offset;
  candidate.uses_gfx1250_cluster_workgroup_id = uses_gfx1250_cluster_workgroup_id;
  candidate.mnemonic = site.mnemonic;
  return candidate;
}

[[nodiscard]] ConSanMoiCandidate make_moi_candidate(std::string container_name, bool in_kernel,
                                                    std::optional<uint64_t> descriptor_file_offset,
                                                    const ConSanFlatSite &site,
                                                    bool uses_gfx1250_cluster_workgroup_id = false,
                                                    uint64_t container_entry_text_offset = 0) {
  ConSanMoiCandidate candidate;
  candidate.container_name = std::move(container_name);
  candidate.in_kernel = in_kernel;
  candidate.container_entry_text_offset = container_entry_text_offset;
  candidate.source = site.address_space_hint == ConSanFlatAddressSpaceHint::Group
                         ? ConSanMoiCandidateSource::FlatGroup
                         : ConSanMoiCandidateSource::FlatMaybeGroup;
  candidate.kind = site.kind;
  candidate.flat_address_space_hint = site.address_space_hint;
  candidate.text_offset = site.text_offset;
  candidate.file_offset = site.file_offset;
  candidate.size = site.size;
  candidate.width_bits = site.width_bits;
  candidate.dst_vgpr = site.dst_vgpr;
  candidate.addr_vgpr = site.addr_vgpr;
  candidate.data_vgpr = site.data_vgpr;
  candidate.kernel_descriptor_file_offset = descriptor_file_offset;
  candidate.uses_gfx1250_cluster_workgroup_id = uses_gfx1250_cluster_workgroup_id;
  candidate.raw_op = site.raw_op;
  candidate.raw_saddr = site.raw_saddr;
  candidate.raw_vaddr = site.raw_vaddr;
  candidate.raw_vsrc = site.raw_vsrc;
  candidate.raw_vdst = site.raw_vdst;
  candidate.raw_ioffset = site.raw_ioffset;
  candidate.raw_segment = site.raw_segment;
  candidate.raw_scope = site.raw_scope;
  candidate.raw_th = site.raw_th;
  candidate.mnemonic = site.mnemonic;
  return candidate;
}

[[nodiscard]] ConSanSiteDispositionReason
classify_moi_access_candidate_support(std::span<const uint8_t> bytes,
                                      const ConSanMoiCandidate &candidate, rj_code_arch_t arch) {
  const bool supported_atomic =
      candidate.kind == ConSanLdsAccessKind::Atomic &&
      equals_any(candidate.mnemonic, {"ds_add_f32", "ds_add_f64", "ds_add_u32", "ds_add_u64",
                                      "ds_cmpstore_rtn_b32", "ds_cmpst_rtn_b32"});
  if (candidate.kind != ConSanLdsAccessKind::Read && candidate.kind != ConSanLdsAccessKind::Write &&
      !supported_atomic)
    return ConSanSiteDispositionReason::NonAccessInstruction;
  if (candidate.size == 0 || candidate.size % sizeof(uint32_t) != 0)
    return ConSanSiteDispositionReason::InvalidInstructionSize;
  if (candidate.file_offset > bytes.size() || candidate.size > bytes.size() - candidate.file_offset)
    return ConSanSiteDispositionReason::InstructionOutOfBounds;
  if (candidate.width_bits == 0 || candidate.width_bits % 8u != 0)
    return ConSanSiteDispositionReason::InvalidAccessWidth;
  if (!candidate.addr_vgpr)
    return ConSanSiteDispositionReason::MissingAddressOperand;

  if (candidate.source == ConSanMoiCandidateSource::NativeLds) {
    return is_single_range_native_lds_mnemonic(candidate.mnemonic, arch) ||
                   two_address_native_lds_offset_scale(candidate.mnemonic).has_value()
               ? ConSanSiteDispositionReason::None
               : ConSanSiteDispositionReason::UnsupportedMnemonic;
  }

  if (candidate.source != ConSanMoiCandidateSource::FlatGroup &&
      candidate.source != ConSanMoiCandidateSource::FlatMaybeGroup)
    return ConSanSiteDispositionReason::NonGroupAddressSpace;
  const bool rdna4_encoding = candidate.size == 3u * sizeof(uint32_t);
  const bool cdna4_encoding =
      candidate.size == 2u * sizeof(uint32_t) && candidate.raw_segment == 0u;
  if (!rdna4_encoding && !cdna4_encoding)
    return ConSanSiteDispositionReason::UnsupportedFlatEncoding;
  if (!candidate.raw_ioffset || (*candidate.raw_ioffset != 0 && arch != ROCJITSU_CODE_ARCH_GFX1250))
    return ConSanSiteDispositionReason::NonzeroFlatOffset;
  if (*candidate.addr_vgpr >= 255)
    return ConSanSiteDispositionReason::ReservedFlatAddressRegister;
  if (!is_supported_moi_flat_access_mnemonic(candidate.mnemonic))
    return ConSanSiteDispositionReason::UnsupportedMnemonic;
  return ConSanSiteDispositionReason::None;
}

void append_moi_access_site_disposition(std::span<const uint8_t> bytes,
                                        const ConSanMoiCandidate &candidate, rj_code_arch_t arch,
                                        ConSanResult &result) {
  const ConSanSiteDispositionReason reason =
      classify_moi_access_candidate_support(bytes, candidate, arch);
  ConSanSiteDispositionRecord record;
  record.site_kind = ConSanResourceSiteKind::Access;
  record.disposition = reason == ConSanSiteDispositionReason::None
                           ? ConSanSiteDisposition::Supported
                           : ConSanSiteDisposition::Unsupported;
  record.reason = reason;
  record.container_name = candidate.container_name;
  record.in_kernel = candidate.in_kernel;
  record.text_offset = candidate.text_offset;
  record.mnemonic = candidate.mnemonic;
  result.site_dispositions.push_back(std::move(record));
}

template <typename Container>
void append_moi_access_site_dispositions(std::span<const uint8_t> bytes, const Container &container,
                                         ConSanFlatProvenanceMode mode,
                                         const ConSanOptions &options, rj_code_arch_t arch,
                                         ConSanResult &result) {
  const bool in_kernel = std::is_same_v<Container, ConSanKernelInfo>;
  if (!options.test_kernel_name_filter.empty() &&
      container.name.find(options.test_kernel_name_filter) == std::string::npos)
    return;
  const std::optional<uint64_t> descriptor = [&]() -> std::optional<uint64_t> {
    if constexpr (std::is_same_v<Container, ConSanKernelInfo>)
      return container.descriptor_file_offset;
    return std::nullopt;
  }();

  for (const ConSanLdsSite &site : container.lds_sites) {
    if (site.kind != ConSanLdsAccessKind::Read && site.kind != ConSanLdsAccessKind::Write &&
        !((arch == ROCJITSU_CODE_ARCH_GFX1250 || arch == ROCJITSU_CODE_ARCH_CDNA3 ||
           arch == ROCJITSU_CODE_ARCH_CDNA4) &&
          site.kind == ConSanLdsAccessKind::Atomic))
      continue;
    append_moi_access_site_disposition(
        bytes, make_moi_candidate(container.name, in_kernel, descriptor, site), arch, result);
  }
  for (const ConSanFlatSite &site : container.flat_sites) {
    if (site.kind != ConSanLdsAccessKind::Read && site.kind != ConSanLdsAccessKind::Write)
      continue;
    const bool group = site.address_space_hint == ConSanFlatAddressSpaceHint::Group;
    const bool admitted_maybe_group =
        mode == ConSanFlatProvenanceMode::Likely &&
        site.address_space_hint == ConSanFlatAddressSpaceHint::MaybeGroup;
    if (!group && !admitted_maybe_group)
      continue;
    append_moi_access_site_disposition(
        bytes, make_moi_candidate(container.name, in_kernel, descriptor, site), arch, result);
  }
}

void append_moi_candidates(const ConSanKernelInfo &kernel, ConSanFlatProvenanceMode mode,
                           rj_code_arch_t arch, ConSanResult &result) {
  SkippedMoiLdsCounts skipped_lds_counts;
  for (const ConSanLdsSite &site : kernel.lds_sites) {
    if (is_moi_native_lds_candidate(site, arch)) {
      result.moi_candidates.push_back(
          make_moi_candidate(kernel.name, true, kernel.descriptor_file_offset, site,
                             kernel.uses_gfx1250_cluster_workgroup_id, kernel.entry_text_offset));
    } else {
      count_skipped_moi_lds_candidate(site, skipped_lds_counts);
    }
  }
  warn_skipped_moi_lds_candidates(kernel.name, true, skipped_lds_counts, result);
  SkippedMoiFlatCounts skipped_flat_counts;
  for (const ConSanFlatSite &site : kernel.flat_sites) {
    if (is_moi_flat_candidate(site, mode)) {
      result.moi_candidates.push_back(
          make_moi_candidate(kernel.name, true, kernel.descriptor_file_offset, site,
                             kernel.uses_gfx1250_cluster_workgroup_id, kernel.entry_text_offset));
    } else if (site.address_space_hint == ConSanFlatAddressSpaceHint::MaybeGroup &&
               mode == ConSanFlatProvenanceMode::Strict) {
      ++skipped_flat_counts.excluded_maybe_group;
    } else {
      count_skipped_moi_flat_candidate(site, skipped_flat_counts);
    }
  }
  warn_skipped_moi_flat_candidates(kernel.name, true, skipped_flat_counts, result);
}

void append_moi_candidates(const ConSanFunctionInfo &function, ConSanFlatProvenanceMode mode,
                           rj_code_arch_t arch, ConSanResult &result) {
  SkippedMoiLdsCounts skipped_lds_counts;
  for (const ConSanLdsSite &site : function.lds_sites) {
    if (is_moi_native_lds_candidate(site, arch)) {
      result.moi_candidates.push_back(make_moi_candidate(function.name, false, std::nullopt, site,
                                                         function.uses_gfx1250_cluster_workgroup_id,
                                                         function.entry_text_offset));
    } else {
      count_skipped_moi_lds_candidate(site, skipped_lds_counts);
    }
  }
  warn_skipped_moi_lds_candidates(function.name, false, skipped_lds_counts, result);
  SkippedMoiFlatCounts skipped_flat_counts;
  for (const ConSanFlatSite &site : function.flat_sites) {
    if (is_moi_flat_candidate(site, mode)) {
      result.moi_candidates.push_back(make_moi_candidate(function.name, false, std::nullopt, site,
                                                         function.uses_gfx1250_cluster_workgroup_id,
                                                         function.entry_text_offset));
    } else if (site.address_space_hint == ConSanFlatAddressSpaceHint::MaybeGroup &&
               mode == ConSanFlatProvenanceMode::Strict) {
      ++skipped_flat_counts.excluded_maybe_group;
    } else {
      count_skipped_moi_flat_candidate(site, skipped_flat_counts);
    }
  }
  warn_skipped_moi_flat_candidates(function.name, false, skipped_flat_counts, result);
}

[[nodiscard]] bool moi_candidate_semantics_equal(const ConSanMoiCandidate &lhs,
                                                 const ConSanMoiCandidate &rhs) {
  return std::tie(lhs.in_kernel, lhs.container_entry_text_offset, lhs.gfx1250_vgpr_msb_mode,
                  lhs.source, lhs.kind, lhs.flat_address_space_hint, lhs.text_offset,
                  lhs.file_offset, lhs.size, lhs.width_bits, lhs.dst_vgpr, lhs.addr_vgpr,
                  lhs.data_vgpr, lhs.second_data_vgpr, lhs.uses_gfx1250_cluster_workgroup_id,
                  lhs.raw_op, lhs.raw_saddr, lhs.raw_vaddr, lhs.raw_vsrc, lhs.raw_vdst,
                  lhs.raw_ioffset, lhs.raw_segment, lhs.raw_scope, lhs.raw_th, lhs.mnemonic) ==
         std::tie(rhs.in_kernel, rhs.container_entry_text_offset, rhs.gfx1250_vgpr_msb_mode,
                  rhs.source, rhs.kind, rhs.flat_address_space_hint, rhs.text_offset,
                  rhs.file_offset, rhs.size, rhs.width_bits, rhs.dst_vgpr, rhs.addr_vgpr,
                  rhs.data_vgpr, rhs.second_data_vgpr, rhs.uses_gfx1250_cluster_workgroup_id,
                  rhs.raw_op, rhs.raw_saddr, rhs.raw_vaddr, rhs.raw_vsrc, rhs.raw_vdst,
                  rhs.raw_ioffset, rhs.raw_segment, rhs.raw_scope, rhs.raw_th, rhs.mnemonic);
}

[[nodiscard]] bool canonicalize_moi_candidates_by_physical_site(ConSanResult &result) {
  std::unordered_map<uint64_t, size_t> canonical_index_by_file_offset;
  canonical_index_by_file_offset.reserve(result.moi_candidates.size());
  std::vector<ConSanMoiCandidate> canonical_candidates;
  canonical_candidates.reserve(result.moi_candidates.size());

  for (ConSanMoiCandidate &candidate : result.moi_candidates) {
    const auto [entry, inserted] =
        canonical_index_by_file_offset.emplace(candidate.file_offset, canonical_candidates.size());
    if (inserted) {
      canonical_candidates.push_back(std::move(candidate));
      continue;
    }

    ConSanMoiCandidate &canonical = canonical_candidates[entry->second];
    if (!moi_candidate_semantics_equal(canonical, candidate)) {
      result.errors.emplace_back(
          "ConSan MOI physical site at file offset " + std::to_string(candidate.file_offset) +
          " was decoded inconsistently through aliases '" + canonical.container_name + "' and '" +
          candidate.container_name + "'");
      return false;
    }

    // A physical instruction reached through multiple kernel descriptors is
    // emitted once, but resource planning must prove a safe register window
    // and apply descriptor growth for every dispatch that can execute it.
    // Clearing the single-owner hint makes plan_moi_resource_site derive that
    // complete owner set from the shared CFG block.
    canonical.kernel_descriptor_file_offset.reset();
  }

  result.moi_candidates = std::move(canonical_candidates);
  return true;
}

[[nodiscard]] uint32_t count_nop_padding(std::span<const uint8_t> bytes, uint64_t offset,
                                         rj_code_arch_t arch) {
  if (offset > bytes.size())
    return 0;

  const uint32_t nop = build_s_nop(0, arch);
  uint32_t count = 0;
  while (bytes.size() - offset >= sizeof(uint32_t)) {
    uint32_t word = 0;
    std::memcpy(&word, bytes.data() + offset, sizeof(word));
    if (word != nop)
      break;
    ++count;
    offset += sizeof(uint32_t);
  }
  return count;
}

[[nodiscard]] std::optional<uint32_t> byte_count_for_candidate(const ConSanMoiCandidate &site) {
  if (site.width_bits == 0 || site.width_bits % 8 != 0)
    return std::nullopt;
  return site.width_bits / 8u;
}

[[nodiscard]] bool is_unconditionally_single_range_native_lds_mnemonic(std::string_view mnemonic) {
  if (equals_any(mnemonic,
                 {"ds_load_i8", "ds_load_u8", "ds_load_i16", "ds_load_u16", "ds_load_u8_d16",
                  "ds_load_u8_d16_hi", "ds_load_i8_d16", "ds_load_i8_d16_hi", "ds_load_u16_d16",
                  "ds_load_u16_d16_hi", "ds_store_b8", "ds_store_b16", "ds_store_b8_d16_hi",
                  "ds_store_b16_d16_hi", "ds_read_u8", "ds_read_u16", "ds_write_b8",
                  "ds_write_b16"}))
    return true;

  return equals_any(mnemonic,
                    {"ds_load_b32",       "ds_load_b64",  "ds_load_b128",  "ds_load_tr8_b64",
                     "ds_load_tr16_b128", "ds_read_b32",  "ds_read_b64",   "ds_read_b64_tr_b16",
                     "ds_read_b128",      "ds_store_b32", "ds_store_b64",  "ds_store_b128",
                     "ds_write_b32",      "ds_write_b64", "ds_write_b128", "ds_add_f32",
                     "ds_add_f64",        "ds_add_u32",   "ds_add_u64",    "ds_cmpstore_rtn_b32",
                     "ds_cmpst_rtn_b32"});
}

[[nodiscard]] bool is_single_range_native_lds_mnemonic(std::string_view mnemonic,
                                                       rj_code_arch_t arch) {
  // RDNA4 B96 is admitted as a family capability: both gfx1201 and gfx1250
  // share the qualified three-dword resource plan and fixed-register lowering.
  // RDNA3/RDNA3.5 encode B96 but are intentionally outside ConSan's supported
  // target set; extend this gate only when those targets gain full support.
  // CDNA spells the operations ds_read_b96/ds_write_b96; they are absent from
  // both admitted mnemonic lists, and
  // CdnaNativeB96AccessesRemainUnsupportedAtTransformBoundary pins that
  // transform boundary.
  if (is_rdna4_family_arch(arch) && equals_any(mnemonic, {"ds_load_b96", "ds_store_b96"}))
    return true;
  return is_unconditionally_single_range_native_lds_mnemonic(mnemonic) ||
         ((arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4) &&
          equals_any(mnemonic, {"ds_read_i8", "ds_write_b8_d16_hi", "ds_write_b16_d16_hi"}));
}

[[nodiscard]] std::optional<uint32_t>
two_address_native_lds_offset_scale(std::string_view mnemonic) {
  if (equals_any(mnemonic,
                 {"ds_load_2addr_b32", "ds_store_2addr_b32", "ds_read2_b32", "ds_write2_b32"}))
    return 4u;
  if (equals_any(mnemonic,
                 {"ds_load_2addr_b64", "ds_store_2addr_b64", "ds_read2_b64", "ds_write2_b64"}))
    return 8u;
  if (equals_any(mnemonic, {"ds_load_2addr_stride64_b32", "ds_store_2addr_stride64_b32",
                            "ds_read2st64_b32", "ds_write2st64_b32"}))
    return 256u;
  if (equals_any(mnemonic, {"ds_load_2addr_stride64_b64", "ds_store_2addr_stride64_b64",
                            "ds_read2st64_b64", "ds_write2st64_b64"}))
    return 512u;
  return std::nullopt;
}

[[nodiscard]] std::optional<std::vector<ConSanMoiAccessRange>>
candidate_access_ranges(std::span<const uint8_t> bytes, const ConSanMoiCandidate &candidate,
                        rj_code_arch_t arch) {
  const std::optional<uint32_t> byte_count = byte_count_for_candidate(candidate);
  if (!byte_count)
    return std::nullopt;

  if (candidate.source == ConSanMoiCandidateSource::NativeLds) {
    const std::optional<uint32_t> word0 = read_u32(bytes, candidate.file_offset);
    if (!word0)
      return std::nullopt;
    if (is_single_range_native_lds_mnemonic(candidate.mnemonic, arch)) {
      // VDS encodes a 16-bit single-address immediate. CDNA3/4 DS atomics use
      // only the eight-bit offset0 field; offset1 is not part of their address.
      const uint32_t static_byte_offset =
          (arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4) &&
                  candidate.kind == ConSanLdsAccessKind::Atomic
              ? *word0 & 0xffu
              : *word0 & 0xffffu;
      return std::vector<ConSanMoiAccessRange>{{static_byte_offset, *byte_count}};
    }

    const std::optional<uint32_t> two_address_scale =
        two_address_native_lds_offset_scale(candidate.mnemonic);
    if (!two_address_scale)
      return std::nullopt;
    const uint32_t offset0 = *word0 & 0xffu;
    const uint32_t offset1 = (*word0 >> 8u) & 0xffu;
    return std::vector<ConSanMoiAccessRange>{
        {offset0 * *two_address_scale, *byte_count},
        {offset1 * *two_address_scale, *byte_count},
    };
  }

  if (candidate.source == ConSanMoiCandidateSource::FlatGroup ||
      candidate.source == ConSanMoiCandidateSource::FlatMaybeGroup)
    return std::vector<ConSanMoiAccessRange>{{0u, *byte_count}};

  return std::nullopt;
}

[[nodiscard]] bool is_first_light_native_lds_candidate(const ConSanMoiCandidate &candidate,
                                                       std::span<const uint8_t> bytes,
                                                       rj_code_arch_t arch) {
  if (candidate.source != ConSanMoiCandidateSource::NativeLds)
    return false;
  return candidate_access_ranges(bytes, candidate, arch).has_value();
}

[[nodiscard]] bool is_first_light_flat_candidate(const ConSanMoiCandidate &candidate,
                                                 rj_code_arch_t arch) {
  if (candidate.source != ConSanMoiCandidateSource::FlatGroup &&
      candidate.source != ConSanMoiCandidateSource::FlatMaybeGroup)
    return false;
  const bool rdna4_encoding = candidate.size == 3u * sizeof(uint32_t);
  const bool cdna4_encoding =
      candidate.size == 2u * sizeof(uint32_t) && candidate.raw_segment == 0u;
  if (!rdna4_encoding && !cdna4_encoding)
    return false;
  if (!candidate.raw_ioffset || (*candidate.raw_ioffset != 0 && arch != ROCJITSU_CODE_ARCH_GFX1250))
    return false;
  if (!candidate.addr_vgpr || *candidate.addr_vgpr >= 255)
    return false;
  if (!is_supported_moi_flat_access_mnemonic(candidate.mnemonic))
    return false;
  return true;
}

[[nodiscard]] bool is_first_light_access_record_candidate(const ConSanMoiCandidate &candidate,
                                                          std::span<const uint8_t> bytes,
                                                          rj_code_arch_t arch) {
  if (candidate.kind != ConSanLdsAccessKind::Read && candidate.kind != ConSanLdsAccessKind::Write &&
      candidate.kind != ConSanLdsAccessKind::Atomic)
    return false;
  if (candidate.size == 0 || candidate.size % sizeof(uint32_t) != 0)
    return false;
  if (candidate.file_offset > bytes.size() || candidate.size > bytes.size() - candidate.file_offset)
    return false;
  if (!byte_count_for_candidate(candidate))
    return false;
  if (candidate.source == ConSanMoiCandidateSource::NativeLds)
    return is_first_light_native_lds_candidate(candidate, bytes, arch);
  return is_first_light_flat_candidate(candidate, arch);
}

[[nodiscard]] std::vector<const ConSanMoiCandidate *>
find_first_light_access_record_candidates(const ConSanResult &result,
                                          std::span<const uint8_t> bytes, rj_code_arch_t arch) {
  std::vector<const ConSanMoiCandidate *> candidates;
  for (const ConSanMoiCandidate &candidate : result.moi_candidates) {
    if (!is_first_light_access_record_candidate(candidate, bytes, arch))
      continue;
    candidates.push_back(&candidate);
  }
  return candidates;
}

[[nodiscard]] bool append_store_u32_literal(std::vector<uint32_t> &words, uint64_t address,
                                            uint32_t value, uint16_t scratch_vgpr,
                                            rj_code_arch_t arch) {
  const uint16_t address_lo_vgpr = scratch_vgpr;
  const uint16_t address_hi_vgpr = static_cast<uint16_t>(scratch_vgpr + 1u);
  const uint16_t value_vgpr = static_cast<uint16_t>(scratch_vgpr + 2u);

  const auto mov_address_lo = instrumentation::build_v_mov_b32_literal(
      address_lo_vgpr, static_cast<uint32_t>(address), arch);
  const auto mov_address_hi = instrumentation::build_v_mov_b32_literal(
      address_hi_vgpr, static_cast<uint32_t>(address >> 32u), arch);
  const auto mov_value = instrumentation::build_v_mov_b32_literal(value_vgpr, value, arch);
  const auto store = instrumentation::build_flat_store_b32(address_lo_vgpr, value_vgpr, arch);
  InstructionSequence sequence(words);
  return sequence.emit_all(mov_address_lo, mov_address_hi, mov_value, store);
}

[[nodiscard]] bool append_store_u32_vgpr(std::vector<uint32_t> &words, uint64_t address,
                                         uint16_t value_vgpr, uint16_t scratch_vgpr,
                                         rj_code_arch_t arch) {
  const uint16_t address_lo_vgpr = scratch_vgpr;
  const uint16_t address_hi_vgpr = static_cast<uint16_t>(scratch_vgpr + 1u);

  const auto mov_address_lo = instrumentation::build_v_mov_b32_literal(
      address_lo_vgpr, static_cast<uint32_t>(address), arch);
  const auto mov_address_hi = instrumentation::build_v_mov_b32_literal(
      address_hi_vgpr, static_cast<uint32_t>(address >> 32u), arch);
  const auto store = instrumentation::build_flat_store_b32(address_lo_vgpr, value_vgpr, arch);
  InstructionSequence sequence(words);
  return sequence.emit_all(mov_address_lo, mov_address_hi, store);
}

[[nodiscard]] bool append_store_u32_sgpr(std::vector<uint32_t> &words, uint64_t address,
                                         uint16_t value_sgpr, uint16_t value_vgpr,
                                         uint16_t scratch_vgpr, rj_code_arch_t arch) {
  words.push_back(build_v_mov_b32_e32(value_vgpr, value_sgpr, arch));
  return append_store_u32_vgpr(words, address, value_vgpr, scratch_vgpr, arch);
}

[[nodiscard]] bool append_store_u32_vgpr_at_offset(std::vector<uint32_t> &words,
                                                   uint16_t address_vgpr, uint32_t byte_offset,
                                                   uint16_t value_vgpr, rj_code_arch_t arch) {
  const auto store =
      instrumentation::build_flat_store_b32(address_vgpr, value_vgpr, arch, byte_offset);
  InstructionSequence sequence(words);
  return sequence.emit(store);
}

[[nodiscard]] bool append_load_u32_vgpr_at_offset(std::vector<uint32_t> &words,
                                                  uint16_t address_vgpr, uint32_t byte_offset,
                                                  uint16_t destination_vgpr, rj_code_arch_t arch) {
  const auto load =
      instrumentation::build_flat_load_b32(address_vgpr, destination_vgpr, arch, byte_offset);
  const auto wait = instrumentation::build_s_wait_flat_load0(arch);
  InstructionSequence sequence(words);
  return sequence.emit_all(load, wait);
}

[[nodiscard]] bool append_workgroup_source_value(std::vector<uint32_t> &words,
                                                 const ConSanMoiWorkgroupSource &source,
                                                 uint16_t value_vgpr, rj_code_arch_t arch) {
  const auto operand = source.operand();
  if (!operand)
    return false;
  words.push_back(build_v_mov_b32_e32(value_vgpr, *operand, arch));
  if (source.mask_low_16) {
    const auto shift_left = instrumentation::build_v_lshlrev_b32(
        value_vgpr, scalar_positive_inline_u32(16), value_vgpr, arch);
    const auto shift_right = instrumentation::build_v_lshrrev_b32(
        value_vgpr, scalar_positive_inline_u32(16), value_vgpr, arch);
    InstructionSequence sequence(words);
    if (!sequence.emit_all(shift_left, shift_right))
      return false;
  }
  if (source.shift_right_16) {
    const auto shift = instrumentation::build_v_lshrrev_b32(
        value_vgpr, scalar_positive_inline_u32(16), value_vgpr, arch);
    InstructionSequence sequence(words);
    if (!sequence.emit(shift))
      return false;
  }
  return true;
}

/// Emits fields relative to one materialized GPU record address.
///
/// The address and temporary VGPRs are explicit because record layouts often
/// keep other live scratch values beside them. Callers retain ownership of
/// publication, synchronization, and record-specific field ordering.
class ConSanMoiRecordEmitter {
public:
  enum class MissingWorkgroupSource : uint8_t { Skip, StoreZero };

  ConSanMoiRecordEmitter(std::vector<uint32_t> &words, uint16_t address_vgpr,
                         uint16_t temporary_vgpr, rj_code_arch_t arch)
      : words_(words), address_vgpr_(address_vgpr), temporary_vgpr_(temporary_vgpr), arch_(arch) {}

  [[nodiscard]] bool materialize_address(uint64_t address) {
    const auto low = instrumentation::build_v_mov_b32_literal(
        address_vgpr_, static_cast<uint32_t>(address), arch_);
    const auto high = instrumentation::build_v_mov_b32_literal(
        static_cast<uint16_t>(address_vgpr_ + 1u), static_cast<uint32_t>(address >> 32u), arch_);
    InstructionSequence sequence(words_);
    return sequence.emit_all(low, high);
  }

  [[nodiscard]] bool load(uint32_t offset, uint16_t destination_vgpr) {
    return append_load_u32_vgpr_at_offset(words_, address_vgpr_, offset, destination_vgpr, arch_);
  }

  [[nodiscard]] bool store_vgpr(uint32_t offset, uint16_t value_vgpr) {
    return append_store_u32_vgpr_at_offset(words_, address_vgpr_, offset, value_vgpr, arch_);
  }

  [[nodiscard]] bool store_literal(uint32_t offset, uint32_t value) {
    const auto materialize =
        instrumentation::build_v_mov_b32_literal(temporary_vgpr_, value, arch_);
    InstructionSequence sequence(words_);
    return sequence.emit(materialize) && store_vgpr(offset, temporary_vgpr_);
  }

  [[nodiscard]] bool store_sgpr(uint32_t offset, uint16_t value_sgpr) {
    words_.push_back(build_v_mov_b32_e32(temporary_vgpr_, value_sgpr, arch_));
    return store_vgpr(offset, temporary_vgpr_);
  }

  [[nodiscard]] bool
  store_workgroup(uint32_t offset, const ConSanMoiWorkgroupSource &source,
                  MissingWorkgroupSource missing = MissingWorkgroupSource::Skip) {
    if (!source.is_well_formed())
      return false;
    if (!source.has_value())
      return missing == MissingWorkgroupSource::Skip ? true : store_literal(offset, 0u);
    return append_workgroup_source_value(words_, source, temporary_vgpr_, arch_) &&
           store_vgpr(offset, temporary_vgpr_);
  }

private:
  std::vector<uint32_t> &words_;
  uint16_t address_vgpr_;
  uint16_t temporary_vgpr_;
  rj_code_arch_t arch_;
};

struct ConSanMoiReportDispatchIdWordSource {
  std::optional<uint16_t> sgpr;
  uint32_t literal = 0;
};

// Code-object-wide CDNA instrumentation keeps the preloaded-SGPR contract
// until its dispatch-identity resolver and final validation accept literal
// state. Per-record and causal-metadata emitters may use the configured
// literal after owner-local resolution has established that no SGPR pair is
// available. Keep every caller explicit so a new path cannot silently choose
// the wrong side of that contract.
enum class ConSanMoiLiteralDispatchIdPolicy : uint8_t { Rdna4FamilyOnly, AnyArchitecture };

[[nodiscard]] ConSanMoiReportDispatchIdWordSource
moi_report_dispatch_id_word_source(const ConSanOptions &options, bool high_word) {
  if (options.moi_dispatch_id_sgpr) {
    return ConSanMoiReportDispatchIdWordSource{
        .sgpr = static_cast<uint16_t>(*options.moi_dispatch_id_sgpr + (high_word ? 1u : 0u))};
  }
  return ConSanMoiReportDispatchIdWordSource{
      .sgpr = std::nullopt,
      .literal = high_word ? static_cast<uint32_t>(options.moi_report_dispatch_id >> 32u)
                           : static_cast<uint32_t>(options.moi_report_dispatch_id)};
}

[[nodiscard]] bool
moi_report_dispatch_id_source_permitted(const ConSanMoiReportDispatchIdWordSource &source,
                                        ConSanMoiLiteralDispatchIdPolicy policy,
                                        rj_code_arch_t arch) {
  return source.sgpr || policy == ConSanMoiLiteralDispatchIdPolicy::AnyArchitecture ||
         is_rdna4_family_arch(arch);
}

[[nodiscard]] bool append_moi_report_dispatch_id_word(std::vector<uint32_t> &words,
                                                      const ConSanOptions &options,
                                                      uint16_t destination_vgpr, bool high_word,
                                                      rj_code_arch_t arch,
                                                      ConSanMoiLiteralDispatchIdPolicy policy) {
  const ConSanMoiReportDispatchIdWordSource source =
      moi_report_dispatch_id_word_source(options, high_word);
  if (!moi_report_dispatch_id_source_permitted(source, policy, arch))
    return false;
  if (source.sgpr) {
    words.push_back(build_v_mov_b32_e32(destination_vgpr, *source.sgpr, arch));
    return true;
  }
  const auto materialize =
      instrumentation::build_v_mov_b32_literal(destination_vgpr, source.literal, arch);
  if (!materialize)
    return false;
  words.insert(words.end(), materialize->begin(), materialize->end());
  return true;
}

[[nodiscard]] bool append_moi_report_dispatch_id_pair(std::vector<uint32_t> &words,
                                                      const ConSanOptions &options,
                                                      uint16_t low_vgpr, uint16_t high_vgpr,
                                                      rj_code_arch_t arch,
                                                      ConSanMoiLiteralDispatchIdPolicy policy) {
  return append_moi_report_dispatch_id_word(words, options, low_vgpr, /*high_word=*/false, arch,
                                            policy) &&
         append_moi_report_dispatch_id_word(words, options, high_vgpr, /*high_word=*/true, arch,
                                            policy);
}

[[nodiscard]] bool append_compare_moi_report_dispatch_id_word(
    std::vector<uint32_t> &words, const ConSanOptions &options, uint16_t value_vgpr,
    uint16_t clobberable_literal_temporary_vgpr, bool high_word, rj_code_arch_t arch,
    ConSanMoiLiteralDispatchIdPolicy policy) {
  const ConSanMoiReportDispatchIdWordSource source =
      moi_report_dispatch_id_word_source(options, high_word);
  if (!moi_report_dispatch_id_source_permitted(source, policy, arch))
    return false;
  uint16_t expected = 0u;
  if (source.sgpr) {
    expected = *source.sgpr;
  } else {
    // The literal path clobbers this temporary. Every caller must pass a dead
    // VGPR whose next use either overwrites or explicitly reloads it.
    assert(clobberable_literal_temporary_vgpr != value_vgpr);
    if (clobberable_literal_temporary_vgpr == value_vgpr ||
        !append_moi_report_dispatch_id_word(words, options, clobberable_literal_temporary_vgpr,
                                            high_word, arch, policy)) {
      return false;
    }
    expected = vector_source_vgpr(clobberable_literal_temporary_vgpr);
  }
  const auto equal = instrumentation::build_v_cmp_eq_u32_vcc(expected, value_vgpr, arch);
  if (!equal)
    return false;
  words.push_back(*equal);
  return true;
}

[[nodiscard]] bool append_store_moi_report_dispatch_id_pair(
    ConSanMoiRecordEmitter &record, const ConSanOptions &options, uint32_t low_offset,
    rj_code_arch_t arch, ConSanMoiLiteralDispatchIdPolicy policy) {
  for (const bool high_word : {false, true}) {
    const ConSanMoiReportDispatchIdWordSource source =
        moi_report_dispatch_id_word_source(options, high_word);
    if (!moi_report_dispatch_id_source_permitted(source, policy, arch))
      return false;
    const uint32_t offset = low_offset + (high_word ? sizeof(uint32_t) : 0u);
    if (source.sgpr ? !record.store_sgpr(offset, *source.sgpr)
                    : !record.store_literal(offset, source.literal)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool append_atomic_event_index_store(std::vector<uint32_t> &words,
                                                   uint64_t counter_address,
                                                   uint64_t event_index_address,
                                                   uint16_t scratch_vgpr, rj_code_arch_t arch) {
  const uint16_t address_lo_vgpr = scratch_vgpr;
  const uint16_t address_hi_vgpr = static_cast<uint16_t>(scratch_vgpr + 1u);
  const uint16_t value_vgpr = static_cast<uint16_t>(scratch_vgpr + 2u);

  const auto mov_address_lo = instrumentation::build_v_mov_b32_literal(
      address_lo_vgpr, static_cast<uint32_t>(counter_address), arch);
  const auto mov_address_hi = instrumentation::build_v_mov_b32_literal(
      address_hi_vgpr, static_cast<uint32_t>(counter_address >> 32u), arch);
  const auto mov_one = instrumentation::build_v_mov_b32_literal(value_vgpr, 1u, arch);
  const auto atomic_add = instrumentation::build_flat_atomic_add_u32(
      address_lo_vgpr, value_vgpr, value_vgpr, /*return_old_value=*/true, kRdna4ScopeDevice, arch);
  const auto wait = instrumentation::build_s_wait_flat_load0(arch);
  InstructionSequence sequence(words);
  if (!sequence.emit_all(mov_address_lo, mov_address_hi, mov_one, atomic_add, wait))
    return false;
  return append_store_u32_vgpr(words, event_index_address, value_vgpr, scratch_vgpr, arch);
}

[[nodiscard]] bool append_atomic_fetch_add_one_u32(std::vector<uint32_t> &words,
                                                   uint64_t counter_address, uint16_t result_vgpr,
                                                   uint16_t scratch_vgpr, rj_code_arch_t arch) {
  const uint16_t address_lo_vgpr = scratch_vgpr;
  const uint16_t address_hi_vgpr = static_cast<uint16_t>(scratch_vgpr + 1u);

  const auto mov_address_lo = instrumentation::build_v_mov_b32_literal(
      address_lo_vgpr, static_cast<uint32_t>(counter_address), arch);
  const auto mov_address_hi = instrumentation::build_v_mov_b32_literal(
      address_hi_vgpr, static_cast<uint32_t>(counter_address >> 32u), arch);
  const auto mov_one = instrumentation::build_v_mov_b32_literal(result_vgpr, 1u, arch);
  const auto atomic_add = instrumentation::build_flat_atomic_add_u32(
      address_lo_vgpr, result_vgpr, result_vgpr,
      /*return_old_value=*/true, kRdna4ScopeDevice, arch);
  const auto wait = instrumentation::build_s_wait_flat_load0(arch);
  InstructionSequence sequence(words);
  return sequence.emit_all(mov_address_lo, mov_address_hi, mov_one, atomic_add, wait);
}

[[nodiscard]] bool append_atomic_or_u32_literal(std::vector<uint32_t> &words, uint64_t address,
                                                uint32_t value, uint16_t scratch_vgpr,
                                                rj_code_arch_t arch) {
  const auto mov_address_lo =
      instrumentation::build_v_mov_b32_literal(scratch_vgpr, static_cast<uint32_t>(address), arch);
  const auto mov_address_hi = instrumentation::build_v_mov_b32_literal(
      static_cast<uint16_t>(scratch_vgpr + 1u), static_cast<uint32_t>(address >> 32u), arch);
  const auto mov_value = instrumentation::build_v_mov_b32_literal(
      static_cast<uint16_t>(scratch_vgpr + 2u), value, arch);
  const auto atomic_or = instrumentation::build_flat_atomic_or_u32(
      scratch_vgpr, static_cast<uint16_t>(scratch_vgpr + 2u),
      static_cast<uint16_t>(scratch_vgpr + 2u), /*return_old_value=*/true, kRdna4ScopeDevice, arch);
  const auto wait = instrumentation::build_s_wait_flat_load0(arch);
  InstructionSequence sequence(words);
  return sequence.emit_all(mov_address_lo, mov_address_hi, mov_value, atomic_or, wait);
}

[[nodiscard]] bool append_atomic_load_u32(std::vector<uint32_t> &words, uint16_t address_vgpr,
                                          uint16_t result_vgpr, rj_code_arch_t arch) {
  const auto atomic_load = instrumentation::build_flat_atomic_add_u32(
      address_vgpr, result_vgpr, result_vgpr, /*return_old_value=*/true, kRdna4ScopeDevice, arch);
  const uint32_t zero = build_v_mov_b32_e32(result_vgpr, scalar_positive_inline_u32(0), arch);
  const auto wait = instrumentation::build_s_wait_flat_load0(arch);
  InstructionSequence sequence(words);
  return sequence.emit_all(zero, atomic_load, wait);
}

// Publish only enough aggregate evidence to prove that the workgroup-local
// Inline detector executed. Exact per-access accounting is not part of this
// counter's host contract, and issuing a device-scope atomic for every exact
// LDS cell creates a severe contention cliff in wide, hot kernels. Once any
// publisher has made the counter nonzero, later publishers take the cached
// load-only path. A bounded startup race may increment it more than once; the
// value remains an honest nonzero publication count without serializing the
// workload on one global atomic.
[[nodiscard]] bool
append_publish_visible_evidence_if_zero(std::vector<uint32_t> &words, uint64_t counter_address,
                                        uint16_t result_vgpr, uint16_t scratch_vgpr,
                                        uint16_t exec_save_sgpr, rj_code_arch_t arch) {
  const auto mov_address_lo = instrumentation::build_v_mov_b32_literal(
      scratch_vgpr, static_cast<uint32_t>(counter_address), arch);
  const auto mov_address_hi =
      instrumentation::build_v_mov_b32_literal(static_cast<uint16_t>(scratch_vgpr + 1u),
                                               static_cast<uint32_t>(counter_address >> 32u), arch);
  const auto load = instrumentation::build_flat_load_b32(scratch_vgpr, result_vgpr, arch);
  const auto is_zero =
      instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), result_vgpr, arch);
  const auto select_unpublished =
      instrumentation::build_s_and_saveexec_b64(exec_save_sgpr, kRdna4VccLo, arch);
  const auto wait = instrumentation::build_s_wait_flat_load0(arch);
  InstructionSequence sequence(words);
  if (!sequence.emit_all(mov_address_lo, mov_address_hi, load, wait, is_zero, select_unpublished))
    return false;
  return append_atomic_fetch_add_one_u32(words, counter_address, result_vgpr, scratch_vgpr, arch);
}

enum class DynamicRecordStride : uint8_t {
  Bytes40 = 40,
  Bytes56 = 56,
  Bytes64 = 64,
  Bytes72 = 72,
  Bytes80 = 80,
  Bytes88 = 88,
};

struct DynamicRecordLayout {
  DynamicRecordStride stride;
  uint16_t value_vgpr_offset;
  bool address_before_value;
};

inline constexpr DynamicRecordLayout kDynamicBarrierRecordLayout = {DynamicRecordStride::Bytes40, 5,
                                                                    false};
inline constexpr DynamicRecordLayout kDynamicAccessRecordLayout = {DynamicRecordStride::Bytes80, 5,
                                                                   false};
inline constexpr DynamicRecordLayout kDynamicAtomicRecordLayout = {DynamicRecordStride::Bytes80, 4,
                                                                   true};
inline constexpr DynamicRecordLayout kDynamicFenceRecordLayout = {DynamicRecordStride::Bytes56, 5,
                                                                  false};
inline constexpr DynamicRecordLayout kDynamicDiagnosticRecordLayout = {DynamicRecordStride::Bytes88,
                                                                       4, true};
static_assert(static_cast<size_t>(kDynamicBarrierRecordLayout.stride) ==
              sizeof(ConSanMoiBarrierRecord));
static_assert(static_cast<size_t>(kDynamicAccessRecordLayout.stride) ==
              sizeof(ConSanMoiAccessRecord));
static_assert(static_cast<size_t>(kDynamicAtomicRecordLayout.stride) ==
              sizeof(ConSanMoiAtomicRecord));
static_assert(static_cast<size_t>(kDynamicFenceRecordLayout.stride) ==
              sizeof(ConSanMoiFenceRecord));
static_assert(static_cast<size_t>(kDynamicDiagnosticRecordLayout.stride) ==
              sizeof(ConSanMoiDiagnosticRecord));

[[nodiscard]] bool append_dynamic_record_address(std::vector<uint32_t> &words,
                                                 const DynamicRecordLayout &layout,
                                                 uint64_t field_address, uint16_t slot_vgpr,
                                                 uint16_t scratch_vgpr, rj_code_arch_t arch) {
  const uint16_t address_lo_vgpr = scratch_vgpr;
  const uint16_t address_hi_vgpr = static_cast<uint16_t>(scratch_vgpr + 1u);
  const bool composite_stride_uses_address_pair = layout.stride == DynamicRecordStride::Bytes80 ||
                                                  layout.stride == DynamicRecordStride::Bytes88;
  const uint16_t scaled_slot_vgpr = composite_stride_uses_address_pair
                                        ? address_hi_vgpr
                                        : static_cast<uint16_t>(scratch_vgpr + 3u);

  if (layout.stride == DynamicRecordStride::Bytes80) {
    const auto slot_times_16 = instrumentation::build_v_lshlrev_b32(
        scaled_slot_vgpr, scalar_positive_inline_u32(4), slot_vgpr, arch);
    const auto slot_times_64 = instrumentation::build_v_lshlrev_b32(
        address_lo_vgpr, scalar_positive_inline_u32(6), slot_vgpr, arch);
    const auto slot_times_80 = instrumentation::build_v_add_u32(
        scaled_slot_vgpr, vector_source_vgpr(address_lo_vgpr), scaled_slot_vgpr, arch);
    InstructionSequence sequence(words);
    if (!sequence.emit_all(slot_times_16, slot_times_64, slot_times_80))
      return false;
  } else if (layout.stride == DynamicRecordStride::Bytes88) {
    const auto slot_times_8 = instrumentation::build_v_lshlrev_b32(
        scaled_slot_vgpr, scalar_positive_inline_u32(3), slot_vgpr, arch);
    const auto slot_times_16 = instrumentation::build_v_lshlrev_b32(
        address_lo_vgpr, scalar_positive_inline_u32(4), slot_vgpr, arch);
    const auto slot_times_24 = instrumentation::build_v_add_u32(
        scaled_slot_vgpr, vector_source_vgpr(address_lo_vgpr), scaled_slot_vgpr, arch);
    const auto slot_times_64 = instrumentation::build_v_lshlrev_b32(
        address_lo_vgpr, scalar_positive_inline_u32(6), slot_vgpr, arch);
    const auto slot_times_88 = instrumentation::build_v_add_u32(
        scaled_slot_vgpr, vector_source_vgpr(address_lo_vgpr), scaled_slot_vgpr, arch);
    InstructionSequence sequence(words);
    if (!sequence.emit_all(slot_times_8, slot_times_16, slot_times_24, slot_times_64,
                           slot_times_88))
      return false;
  }

  const auto mov_address_lo = instrumentation::build_v_mov_b32_literal(
      address_lo_vgpr, static_cast<uint32_t>(field_address), arch);
  const auto mov_address_hi = instrumentation::build_v_mov_b32_literal(
      address_hi_vgpr, static_cast<uint32_t>(field_address >> 32u), arch);
  InstructionSequence sequence(words);
  if (!sequence.emit(mov_address_lo))
    return false;
  if (!composite_stride_uses_address_pair && !sequence.emit(mov_address_hi))
    return false;

  if (layout.stride == DynamicRecordStride::Bytes40 ||
      layout.stride == DynamicRecordStride::Bytes56) {
    const uint16_t tmp_vgpr = static_cast<uint16_t>(scratch_vgpr + 4u);
    const auto slot_times_32 = instrumentation::build_v_lshlrev_b32(
        scaled_slot_vgpr, scalar_positive_inline_u32(5), slot_vgpr, arch);
    const auto slot_times_8 = instrumentation::build_v_lshlrev_b32(
        tmp_vgpr, scalar_positive_inline_u32(3), slot_vgpr, arch);
    const auto slot_times_sum = instrumentation::build_v_add_u32(
        scaled_slot_vgpr, vector_source_vgpr(scaled_slot_vgpr), tmp_vgpr, arch);
    if (!sequence.emit_all(slot_times_32, slot_times_8, slot_times_sum))
      return false;
    if (layout.stride == DynamicRecordStride::Bytes56) {
      const auto slot_times_16 = instrumentation::build_v_lshlrev_b32(
          tmp_vgpr, scalar_positive_inline_u32(4), slot_vgpr, arch);
      const auto slot_times_56 = instrumentation::build_v_add_u32(
          scaled_slot_vgpr, vector_source_vgpr(scaled_slot_vgpr), tmp_vgpr, arch);
      if (!sequence.emit_all(slot_times_16, slot_times_56))
        return false;
    }
  } else if (layout.stride == DynamicRecordStride::Bytes64 ||
             layout.stride == DynamicRecordStride::Bytes72) {
    const auto slot_times_64 = instrumentation::build_v_lshlrev_b32(
        scaled_slot_vgpr, scalar_positive_inline_u32(6), slot_vgpr, arch);
    if (!sequence.emit(slot_times_64))
      return false;
    if (layout.stride == DynamicRecordStride::Bytes72) {
      const uint16_t tmp_vgpr = static_cast<uint16_t>(scratch_vgpr + 4u);
      const auto slot_times_8 = instrumentation::build_v_lshlrev_b32(
          tmp_vgpr, scalar_positive_inline_u32(3), slot_vgpr, arch);
      const auto slot_times_72 = instrumentation::build_v_add_u32(
          scaled_slot_vgpr, vector_source_vgpr(scaled_slot_vgpr), tmp_vgpr, arch);
      if (!sequence.emit_all(slot_times_8, slot_times_72))
        return false;
    }
  }

  const auto address_with_slot = instrumentation::build_v_add_u32(
      address_lo_vgpr, vector_source_vgpr(address_lo_vgpr), scaled_slot_vgpr, arch);
  if (!sequence.emit(address_with_slot))
    return false;
  if (composite_stride_uses_address_pair && !sequence.emit(mov_address_hi))
    return false;
  return true;
}

[[nodiscard]] bool append_dynamic_record_store_u32_vgpr(std::vector<uint32_t> &words,
                                                        const DynamicRecordLayout &layout,
                                                        uint64_t field_address, uint16_t value_vgpr,
                                                        uint16_t slot_vgpr, uint16_t scratch_vgpr,
                                                        rj_code_arch_t arch) {
  const auto store = instrumentation::build_flat_store_b32(scratch_vgpr, value_vgpr, arch);
  if (!store ||
      !append_dynamic_record_address(words, layout, field_address, slot_vgpr, scratch_vgpr, arch))
    return false;
  words.insert(words.end(), store->begin(), store->end());
  return true;
}

[[nodiscard]] bool append_dynamic_record_store_u32_literal(
    std::vector<uint32_t> &words, const DynamicRecordLayout &layout, uint64_t field_address,
    uint32_t value, uint16_t slot_vgpr, uint16_t scratch_vgpr, rj_code_arch_t arch) {
  const uint16_t value_vgpr = static_cast<uint16_t>(scratch_vgpr + layout.value_vgpr_offset);
  if (layout.address_before_value &&
      !append_dynamic_record_address(words, layout, field_address, slot_vgpr, scratch_vgpr, arch))
    return false;
  const auto mov_value = instrumentation::build_v_mov_b32_literal(value_vgpr, value, arch);
  if (!mov_value)
    return false;
  words.insert(words.end(), mov_value->begin(), mov_value->end());
  if (!layout.address_before_value)
    return append_dynamic_record_store_u32_vgpr(words, layout, field_address, value_vgpr, slot_vgpr,
                                                scratch_vgpr, arch);
  const auto store = instrumentation::build_flat_store_b32(scratch_vgpr, value_vgpr, arch);
  if (!store)
    return false;
  words.insert(words.end(), store->begin(), store->end());
  return true;
}

[[nodiscard]] bool append_dynamic_record_store_u32_scalar_src(
    std::vector<uint32_t> &words, const DynamicRecordLayout &layout, uint64_t field_address,
    uint16_t scalar_src, uint16_t slot_vgpr, uint16_t scratch_vgpr, rj_code_arch_t arch) {
  const uint16_t value_vgpr = static_cast<uint16_t>(scratch_vgpr + layout.value_vgpr_offset);
  if (layout.address_before_value &&
      !append_dynamic_record_address(words, layout, field_address, slot_vgpr, scratch_vgpr, arch))
    return false;
  words.push_back(build_v_mov_b32_e32(value_vgpr, scalar_src, arch));
  if (!layout.address_before_value)
    return append_dynamic_record_store_u32_vgpr(words, layout, field_address, value_vgpr, slot_vgpr,
                                                scratch_vgpr, arch);
  const auto store = instrumentation::build_flat_store_b32(scratch_vgpr, value_vgpr, arch);
  if (!store)
    return false;
  words.insert(words.end(), store->begin(), store->end());
  return true;
}

[[nodiscard]] bool append_dynamic_record_store_moi_report_dispatch_id_pair(
    std::vector<uint32_t> &words, const DynamicRecordLayout &layout, uint64_t low_field_address,
    const ConSanOptions &options, uint16_t slot_vgpr, uint16_t scratch_vgpr, rj_code_arch_t arch,
    ConSanMoiLiteralDispatchIdPolicy policy) {
  for (const bool high_word : {false, true}) {
    const ConSanMoiReportDispatchIdWordSource source =
        moi_report_dispatch_id_word_source(options, high_word);
    if (!moi_report_dispatch_id_source_permitted(source, policy, arch))
      return false;
    const uint64_t field_address = low_field_address + (high_word ? sizeof(uint32_t) : 0u);
    const bool stored =
        source.sgpr
            ? append_dynamic_record_store_u32_scalar_src(words, layout, field_address, *source.sgpr,
                                                         slot_vgpr, scratch_vgpr, arch)
            : append_dynamic_record_store_u32_literal(words, layout, field_address, source.literal,
                                                      slot_vgpr, scratch_vgpr, arch);
    if (!stored)
      return false;
  }
  return true;
}

[[nodiscard]] bool append_dynamic_record_store_workgroup_source(
    std::vector<uint32_t> &words, const DynamicRecordLayout &layout, uint64_t field_address,
    const ConSanMoiWorkgroupSource &source, uint16_t slot_vgpr, uint16_t scratch_vgpr,
    rj_code_arch_t arch) {
  if (!source.is_well_formed())
    return false;
  if (!source.has_value())
    return true;
  const uint16_t value_vgpr = static_cast<uint16_t>(scratch_vgpr + layout.value_vgpr_offset);
  const auto operand = source.operand();
  if (!operand)
    return false;
  words.push_back(build_v_mov_b32_e32(value_vgpr, *operand, arch));
  if (source.mask_low_16) {
    const auto shift_left = instrumentation::build_v_lshlrev_b32(
        value_vgpr, scalar_positive_inline_u32(16), value_vgpr, arch);
    const auto shift_right = instrumentation::build_v_lshrrev_b32(
        value_vgpr, scalar_positive_inline_u32(16), value_vgpr, arch);
    if (!shift_left || !shift_right)
      return false;
    words.push_back(*shift_left);
    words.push_back(*shift_right);
  }
  if (source.shift_right_16) {
    const auto shift = instrumentation::build_v_lshrrev_b32(
        value_vgpr, scalar_positive_inline_u32(16), value_vgpr, arch);
    if (!shift)
      return false;
    words.push_back(*shift);
  }
  return append_dynamic_record_store_u32_vgpr(words, layout, field_address, value_vgpr, slot_vgpr,
                                              scratch_vgpr, arch);
}

[[nodiscard]] bool append_dynamic_record_event_index_store(
    std::vector<uint32_t> &words, const DynamicRecordLayout &layout, uint64_t counter_address,
    uint64_t field_address, uint16_t slot_vgpr, uint16_t scratch_vgpr, rj_code_arch_t arch) {
  const uint16_t value_vgpr = static_cast<uint16_t>(scratch_vgpr + layout.value_vgpr_offset);
  return append_atomic_fetch_add_one_u32(words, counter_address, value_vgpr, scratch_vgpr, arch) &&
         append_dynamic_record_store_u32_vgpr(words, layout, field_address, value_vgpr, slot_vgpr,
                                              scratch_vgpr, arch);
}

#define RJ_DYNAMIC_RECORD_WRAPPERS(prefix, layout)                                                 \
  [[nodiscard]] bool append_dynamic_##prefix##_store_u32_vgpr(                                     \
      std::vector<uint32_t> &words, uint64_t address, uint16_t value, uint16_t slot,               \
      uint16_t scratch, rj_code_arch_t arch) {                                                     \
    return append_dynamic_record_store_u32_vgpr(words, layout, address, value, slot, scratch,      \
                                                arch);                                             \
  }                                                                                                \
  [[nodiscard]] bool append_dynamic_##prefix##_store_u32_literal(                                  \
      std::vector<uint32_t> &words, uint64_t address, uint32_t value, uint16_t slot,               \
      uint16_t scratch, rj_code_arch_t arch) {                                                     \
    return append_dynamic_record_store_u32_literal(words, layout, address, value, slot, scratch,   \
                                                   arch);                                          \
  }                                                                                                \
  [[nodiscard]] bool append_dynamic_##prefix##_store_u32_scalar_src(                               \
      std::vector<uint32_t> &words, uint64_t address, uint16_t value, uint16_t slot,               \
      uint16_t scratch, rj_code_arch_t arch) {                                                     \
    return append_dynamic_record_store_u32_scalar_src(words, layout, address, value, slot,         \
                                                      scratch, arch);                              \
  }

RJ_DYNAMIC_RECORD_WRAPPERS(barrier, kDynamicBarrierRecordLayout)
RJ_DYNAMIC_RECORD_WRAPPERS(access, kDynamicAccessRecordLayout)
RJ_DYNAMIC_RECORD_WRAPPERS(atomic, kDynamicAtomicRecordLayout)
RJ_DYNAMIC_RECORD_WRAPPERS(fence, kDynamicFenceRecordLayout)
RJ_DYNAMIC_RECORD_WRAPPERS(diagnostic, kDynamicDiagnosticRecordLayout)
#undef RJ_DYNAMIC_RECORD_WRAPPERS

[[nodiscard]] bool append_dynamic_diagnostic_record_address(std::vector<uint32_t> &words,
                                                            uint64_t address, uint16_t slot,
                                                            uint16_t scratch, rj_code_arch_t arch) {
  return append_dynamic_record_address(words, kDynamicDiagnosticRecordLayout, address, slot,
                                       scratch, arch);
}

[[nodiscard]] bool append_dynamic_barrier_store_workgroup_source(
    std::vector<uint32_t> &words, uint64_t field_address, const ConSanMoiWorkgroupSource &source,
    uint16_t slot_vgpr, uint16_t scratch_vgpr, rj_code_arch_t arch) {
  return append_dynamic_record_store_workgroup_source(
      words, kDynamicBarrierRecordLayout, field_address, source, slot_vgpr, scratch_vgpr, arch);
}

[[nodiscard]] bool append_dynamic_access_store_workgroup_source(
    std::vector<uint32_t> &words, uint64_t field_address, const ConSanMoiWorkgroupSource &source,
    uint16_t slot_vgpr, uint16_t scratch_vgpr, rj_code_arch_t arch) {
  return append_dynamic_record_store_workgroup_source(
      words, kDynamicAccessRecordLayout, field_address, source, slot_vgpr, scratch_vgpr, arch);
}

[[nodiscard]] bool append_dynamic_atomic_store_workgroup_source(
    std::vector<uint32_t> &words, uint64_t field_address, const ConSanMoiWorkgroupSource &source,
    uint16_t slot_vgpr, uint16_t scratch_vgpr, rj_code_arch_t arch) {
  return append_dynamic_record_store_workgroup_source(
      words, kDynamicAtomicRecordLayout, field_address, source, slot_vgpr, scratch_vgpr, arch);
}

[[nodiscard]] bool append_dynamic_fence_store_workgroup_source(
    std::vector<uint32_t> &words, uint64_t field_address, const ConSanMoiWorkgroupSource &source,
    uint16_t slot_vgpr, uint16_t scratch_vgpr, rj_code_arch_t arch) {
  return append_dynamic_record_store_workgroup_source(
      words, kDynamicFenceRecordLayout, field_address, source, slot_vgpr, scratch_vgpr, arch);
}

[[nodiscard]] bool
append_dynamic_barrier_event_index_store(std::vector<uint32_t> &words, uint64_t counter_address,
                                         uint64_t field_address, uint16_t slot_vgpr,
                                         uint16_t scratch_vgpr, rj_code_arch_t arch) {
  return append_dynamic_record_event_index_store(words, kDynamicBarrierRecordLayout,
                                                 counter_address, field_address, slot_vgpr,
                                                 scratch_vgpr, arch);
}

[[nodiscard]] bool
append_dynamic_access_event_index_store(std::vector<uint32_t> &words, uint64_t counter_address,
                                        uint64_t field_address, uint16_t slot_vgpr,
                                        uint16_t scratch_vgpr, rj_code_arch_t arch) {
  return append_dynamic_record_event_index_store(words, kDynamicAccessRecordLayout, counter_address,
                                                 field_address, slot_vgpr, scratch_vgpr, arch);
}

[[nodiscard]] bool
append_dynamic_atomic_event_index_store(std::vector<uint32_t> &words, uint64_t counter_address,
                                        uint64_t field_address, uint16_t slot_vgpr,
                                        uint16_t scratch_vgpr, rj_code_arch_t arch) {
  return append_dynamic_record_event_index_store(words, kDynamicAtomicRecordLayout, counter_address,
                                                 field_address, slot_vgpr, scratch_vgpr, arch);
}

[[nodiscard]] bool append_dynamic_fence_event_index_store(std::vector<uint32_t> &words,
                                                          uint64_t counter_address,
                                                          uint64_t field_address,
                                                          uint16_t slot_vgpr, uint16_t scratch_vgpr,
                                                          rj_code_arch_t arch) {
  return append_dynamic_record_event_index_store(words, kDynamicFenceRecordLayout, counter_address,
                                                 field_address, slot_vgpr, scratch_vgpr, arch);
}

[[nodiscard]] bool overlaps_scratch_range(uint16_t value, uint16_t scratch_vgpr, uint16_t count) {
  return count != 0 && value >= scratch_vgpr && value < static_cast<uint16_t>(scratch_vgpr + count);
}

[[nodiscard]] bool is_sop1_saveexec_word(uint32_t word) {
  constexpr uint32_t kSop1PrefixMask = 0xFF800000u;
  constexpr uint32_t kSop1OpMask = 0x0000FF00u;
  const uint32_t op = (word & kSop1OpMask) >> 8;
  return (word & kSop1PrefixMask) == (kSop1EncodingPrefix << 23) && op >= 0x20u && op <= 0x33u;
}

[[nodiscard]] bool has_recent_saveexec(std::span<const uint8_t> bytes,
                                       const ConSanMoiCandidate &candidate) {
  constexpr uint64_t kLookbackDwords = 3;
  for (uint64_t dword = 1; dword <= kLookbackDwords; ++dword) {
    const uint64_t byte_distance = dword * sizeof(uint32_t);
    if (candidate.file_offset < byte_distance)
      break;
    const std::optional<uint32_t> word = read_u32(bytes, candidate.file_offset - byte_distance);
    if (word && is_sop1_saveexec_word(*word))
      return true;
  }
  return false;
}

[[nodiscard]] bool range_overlaps(uint16_t lhs_base, uint16_t lhs_count, uint16_t rhs_base,
                                  uint16_t rhs_count) {
  const uint32_t lhs_end = static_cast<uint32_t>(lhs_base) + lhs_count;
  const uint32_t rhs_end = static_cast<uint32_t>(rhs_base) + rhs_count;
  return static_cast<uint32_t>(lhs_base) < rhs_end && static_cast<uint32_t>(rhs_base) < lhs_end;
}

[[nodiscard]] bool reject_optional_scratch_range_overlap(std::optional<uint16_t> value,
                                                         uint16_t scratch_vgpr,
                                                         uint16_t scratch_count,
                                                         std::string_view value_name,
                                                         std::vector<std::string> &errors) {
  if (!value || !overlaps_scratch_range(*value, scratch_vgpr, scratch_count))
    return false;
  errors.emplace_back(std::string("ConSan MOI scratch VGPRs overlap ") + std::string(value_name) +
                      " VGPR");
  return true;
}

void append_word_bytes(std::vector<uint8_t> &bytes, uint32_t word) {
  const auto *begin = reinterpret_cast<const uint8_t *>(&word);
  bytes.insert(bytes.end(), begin, begin + sizeof(word));
}

void append_words_bytes(std::vector<uint8_t> &bytes, std::span<const uint32_t> words) {
  for (uint32_t word : words)
    append_word_bytes(bytes, word);
}

[[nodiscard]] std::optional<DbiPatchPlacement>
plan_prebuilt_appended_cave(DbiPatchPlacementPlanner &planner, uint64_t anchor_offset,
                            uint32_t original_size, std::span<const uint32_t> cave_words,
                            std::vector<std::string> &errors, std::string_view probe_name) {
  if (cave_words.empty()) {
    errors.emplace_back(std::string(probe_name) + " produced an empty cave");
    return std::nullopt;
  }
  DbiPatchPlacementRequest request;
  request.anchor_offset = anchor_offset;
  request.original_size = original_size;
  request.body_size = static_cast<uint64_t>(cave_words.size() - 1u) * sizeof(uint32_t);
  request.inline_capacity = 0;
  std::string placement_error;
  const auto placement = planner.plan(request, &placement_error);
  if (!placement || placement->kind != DbiPatchPlacementKind::AppendedCave ||
      placement->return_branch_offset !=
          placement->body_offset + (cave_words.size() - 1u) * sizeof(uint32_t)) {
    errors.emplace_back(std::string(probe_name) + " placement failed: " + placement_error);
    return std::nullopt;
  }
  return placement;
}
