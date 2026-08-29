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
inline constexpr uint32_t kMaxVgprs = kConSanOrdinaryVgprLimit;
inline constexpr uint32_t kMaxSgprs = 106;
inline constexpr uint8_t kRdna4ScopeDevice = 2;
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
[[nodiscard]] bool append_moi_global_atomic_wait(std::vector<uint32_t> &words,
                                                 rj_code_arch_t arch) {
  const ConSanTargetProfile *target = consan_target_profile(arch);
  return target != nullptr && consan_detail::append_moi_global_atomic_completion(words, *target);
}
[[nodiscard]] bool append_moi_lds_wait(std::vector<uint32_t> &words, rj_code_arch_t arch) {
  const auto wait = instrumentation::build_s_wait_lds0(arch);
  if (!wait)
    return false;
  words.push_back(*wait);
  return true;
}
inline constexpr uint64_t kAmdhsaKernelEntryAlignment = 256;

[[nodiscard]] constexpr uint16_t ttmp_scalar_operand(uint16_t ttmp) {
  return static_cast<uint16_t>(kScalarOperandTtmpBase + ttmp);
}

using KD = rocr::llvm::amdhsa::kernel_descriptor_t;
namespace kd = rocr::llvm::amdhsa;

[[nodiscard]] std::optional<uint32_t> read_u32(std::span<const uint8_t> bytes, uint64_t offset) {
  if (offset > bytes.size() || sizeof(uint32_t) > bytes.size() - offset)
    return std::nullopt;
  uint32_t word = 0;
  std::memcpy(&word, bytes.data() + offset, sizeof(word));
  return word;
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
