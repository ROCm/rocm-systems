namespace consan_moi_impl {

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
