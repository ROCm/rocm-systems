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

[[nodiscard]] bool append_atomic_fetch_add_one_u32(std::vector<uint32_t> &words,
                                                   uint64_t counter_address, uint16_t result_vgpr,
                                                   uint16_t scratch_vgpr, rj_code_arch_t arch) {
  const ConSanTargetProfile *target = consan_target_profile(arch);
  return target != nullptr &&
         consan_detail::append_moi_atomic_counter_increment(words,
                                                            {
                                                                .counter_address = counter_address,
                                                                .result_vgpr = result_vgpr,
                                                                .address_vgpr = scratch_vgpr,
                                                            },
                                                            *target);
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
  InstructionSequence sequence(words);
  return sequence.emit_all(mov_address_lo, mov_address_hi, mov_value, atomic_or) &&
         append_moi_global_atomic_wait(words, arch);
}

[[nodiscard]] bool append_atomic_load_u32(std::vector<uint32_t> &words, uint16_t address_vgpr,
                                          uint16_t result_vgpr, rj_code_arch_t arch) {
  const auto atomic_load = instrumentation::build_flat_atomic_add_u32(
      address_vgpr, result_vgpr, result_vgpr, /*return_old_value=*/true, kRdna4ScopeDevice, arch);
  const uint32_t zero = build_v_mov_b32_e32(result_vgpr, scalar_positive_inline_u32(0), arch);
  InstructionSequence sequence(words);
  return sequence.emit_all(zero, atomic_load) && append_moi_global_atomic_wait(words, arch);
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
  const auto wait = instrumentation::build_s_wait_global_load0(arch);
  InstructionSequence sequence(words);
  if (!sequence.emit_all(mov_address_lo, mov_address_hi, load, wait, is_zero, select_unpublished))
    return false;
  return append_atomic_fetch_add_one_u32(words, counter_address, result_vgpr, scratch_vgpr, arch);
}

struct DynamicRecordLayout {
  uint32_t stride_bytes;
  uint16_t value_vgpr_offset;
};

inline constexpr DynamicRecordLayout kBarrierRecordLayout = {sizeof(ConSanMoiBarrierRecord), 5};
inline constexpr DynamicRecordLayout kAccessRecordLayout = {sizeof(ConSanMoiAccessRecord), 5};
inline constexpr DynamicRecordLayout kAtomicRecordLayout = {sizeof(ConSanMoiAtomicRecord), 4};
inline constexpr DynamicRecordLayout kFenceRecordLayout = {sizeof(ConSanMoiFenceRecord), 5};
inline constexpr DynamicRecordLayout kDiagnosticRecordLayout = {sizeof(ConSanMoiDiagnosticRecord),
                                                                4};

[[nodiscard]] bool append_dynamic_record_address(std::vector<uint32_t> &words,
                                                 const DynamicRecordLayout &layout,
                                                 uint64_t field_address, uint16_t slot_vgpr,
                                                 uint16_t scratch_vgpr, rj_code_arch_t arch) {
  const ConSanTargetProfile *target = consan_target_profile(arch);
  return target != nullptr &&
         consan_detail::append_dynamic_record_address(words,
                                                      {
                                                          .field_address = field_address,
                                                          .stride_bytes = layout.stride_bytes,
                                                          .address_vgpr = scratch_vgpr,
                                                          .slot_vgpr = slot_vgpr,
                                                      },
                                                      *target);
}

[[nodiscard]] bool append_dynamic_record_store_u32_vgpr(std::vector<uint32_t> &words,
                                                        const DynamicRecordLayout &layout,
                                                        uint64_t field_address, uint16_t value_vgpr,
                                                        uint16_t slot_vgpr, uint16_t scratch_vgpr,
                                                        rj_code_arch_t arch) {
  if (value_vgpr == scratch_vgpr || value_vgpr == static_cast<uint16_t>(scratch_vgpr + 1u)) {
    return false;
  }
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
  if (value_vgpr == slot_vgpr ||
      !append_dynamic_record_address(words, layout, field_address, slot_vgpr, scratch_vgpr, arch))
    return false;
  const auto mov_value = instrumentation::build_v_mov_b32_literal(value_vgpr, value, arch);
  if (!mov_value)
    return false;
  words.insert(words.end(), mov_value->begin(), mov_value->end());
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
  if (value_vgpr == slot_vgpr ||
      !append_dynamic_record_address(words, layout, field_address, slot_vgpr, scratch_vgpr, arch))
    return false;
  words.push_back(build_v_mov_b32_e32(value_vgpr, scalar_src, arch));
  const auto store = instrumentation::build_flat_store_b32(scratch_vgpr, value_vgpr, arch);
  if (!store)
    return false;
  words.insert(words.end(), store->begin(), store->end());
  return true;
}

[[nodiscard]] bool append_dynamic_record_store_moi_report_dispatch_id_pair(
    std::vector<uint32_t> &words, const DynamicRecordLayout &layout, uint64_t low_field_address,
    const std::array<ConSanMoiReportDispatchIdWordSource, 2> &sources, uint16_t slot_vgpr,
    uint16_t scratch_vgpr, rj_code_arch_t arch, ConSanMoiLiteralDispatchIdPolicy policy) {
  for (const bool high_word : {false, true}) {
    const ConSanMoiReportDispatchIdWordSource &source = sources[high_word ? 1u : 0u];
    if (!moi_report_dispatch_id_source_permitted(source, policy, arch))
      return false;
    const uint64_t field_address = low_field_address + (high_word ? sizeof(uint32_t) : 0u);
    const bool stored =
        source.sgpr ? append_dynamic_record_store_u32_scalar_src(
                          words, layout, field_address, *source.sgpr, slot_vgpr, scratch_vgpr, arch)
        : source.vgpr
            ? append_dynamic_record_store_u32_vgpr(words, layout, field_address, *source.vgpr,
                                                   slot_vgpr, scratch_vgpr, arch)
            : append_dynamic_record_store_u32_literal(words, layout, field_address, source.literal,
                                                      slot_vgpr, scratch_vgpr, arch);
    if (!stored)
      return false;
  }
  return true;
}

[[nodiscard]] bool append_dynamic_record_store_moi_report_dispatch_id_pair(
    std::vector<uint32_t> &words, const DynamicRecordLayout &layout, uint64_t low_field_address,
    const ConSanMoiOperatingPoint &point, const BoundRuntimeResources &resources,
    uint16_t slot_vgpr, uint16_t scratch_vgpr, rj_code_arch_t arch,
    ConSanMoiLiteralDispatchIdPolicy policy) {
  const std::array sources{
      moi_report_dispatch_id_word_source(point, resources, /*high_word=*/false),
      moi_report_dispatch_id_word_source(point, resources, /*high_word=*/true)};
  return append_dynamic_record_store_moi_report_dispatch_id_pair(
      words, layout, low_field_address, sources, slot_vgpr, scratch_vgpr, arch, policy);
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
  if (!consan_detail::append_workgroup_source_value(words, source, value_vgpr, arch))
    return false;
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

[[nodiscard]] bool append_dynamic_diagnostic_record_address(std::vector<uint32_t> &words,
                                                            uint64_t address, uint16_t slot,
                                                            uint16_t address_vgpr,
                                                            rj_code_arch_t arch) {
  return append_dynamic_record_address(words, kDiagnosticRecordLayout, address, slot, address_vgpr,
                                       arch);
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
