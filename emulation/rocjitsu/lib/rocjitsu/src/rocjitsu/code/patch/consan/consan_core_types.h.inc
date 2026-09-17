[[nodiscard]] inline uint32_t max_workgroup_lds_bytes(rj_code_arch_t arch) {
  const TargetProfile *profile = target_profile(arch);
  return profile ? profile->max_group_segment_bytes : 64u * 1024u;
}

[[nodiscard]] inline uint32_t max_workgroup_lds_bytes(const RuntimeCapabilities &capabilities,
                                                      rj_code_arch_t arch) {
  return capabilities.max_workgroup_lds_bytes.value_or(max_workgroup_lds_bytes(arch));
}

/// Effective-address recipe for ConSan atomic instrumentation. Synchronization
/// metadata uses exactly the address consumed by the guest instruction.
enum class AtomicAddressKind : uint8_t {
  Unsupported,
  LdsByteOffsetToken,
  FlatGuestPair,
  FlatGuestPairMaterialized,
  VglobalGuestPair,
  VglobalGuestPairMaterialized,
  VglobalMaterialized,
  BufferResourceMaterialized,
};

enum class AtomicAddressSupport : uint8_t {
  Supported,
  UnsupportedArchitecture,
  UnsupportedAddressKind,
  UnsupportedWidth,
  UnsupportedEncoding,
  MissingAddressOperands,
  UnsupportedInputWidth,
  UnsupportedOffset,
  UnsupportedScope,
  UnsupportedResourcePlan,
  UnsupportedScratchShape,
  ResultAddressAlias,
  ScratchOperandAlias,
};

struct AtomicAddressPlan {
  AtomicAddressKind kind = AtomicAddressKind::Unsupported;
  AtomicAddressSupport support = AtomicAddressSupport::UnsupportedAddressKind;
  uint16_t input_address_vgpr = 0;
  uint16_t input_address_vgpr_count = 0;
  /// Multiplicative scale applied to a scalar-base VGLOBAL's 32-bit vector
  /// offset before zero/sign extension. This is one except for CDNA5's
  /// SCALE_OFFSET form, where it is the communication access width in bytes.
  uint16_t input_address_scale = 1;
  bool sign_extend_vector_offset = false;
  std::optional<uint16_t> scalar_base_sgpr;
  std::optional<uint16_t> scalar_offset_sgpr;
  int32_t signed_byte_offset = 0;
  uint16_t result_address_vgpr = 0;
  uint16_t result_address_vgpr_count = 0;
  uint16_t scratch_vgpr = 0;
  uint16_t scratch_vgpr_count = 0;
  RegisterAllocationSource resource_source = RegisterAllocationSource::Unsupported;

  [[nodiscard]] constexpr bool supported() const {
    return support == AtomicAddressSupport::Supported;
  }
  [[nodiscard]] constexpr bool requires_materialization() const {
    return kind == AtomicAddressKind::LdsByteOffsetToken ||
           kind == AtomicAddressKind::FlatGuestPairMaterialized ||
           kind == AtomicAddressKind::VglobalGuestPairMaterialized ||
           kind == AtomicAddressKind::VglobalMaterialized ||
           kind == AtomicAddressKind::BufferResourceMaterialized;
  }
};

// ConSan state is partitioned by workgroup. Within that state, use a
// tagged 64-bit token for LDS so a byte offset cannot alias a global address.
inline constexpr uint32_t kLdsAddressTokenTag = 0xffffffffu;
[[nodiscard]] std::string_view atomic_address_support_name(AtomicAddressSupport support);

/// Build a typed effective-address plan from a classifier-owned normalized
/// target form and an already resolved scratch-resource plan. FLAT aliases its guest 64-bit
/// address unless a returned value aliases it, in which case it snapshots the
/// pair into the scratch tail. VGLOBAL likewise reserves the last two VGPRs of
/// a five-VGPR private window for an aliased pair or scalar-base address.
/// RDNA4 scalar-base VGLOBAL uses
/// `saddr + zero_extend(vaddr) + signed24(ioffset)`; CDNA5 uses
/// `saddr + zero_extend(vaddr * (scale_offset ? access_bytes : 1)) +
/// signed24(ioffset)`; CDNA3/4 uses
/// `saddr + sign_extend(vaddr) + signed13(ioffset)`. Operand overlap is only
/// valid for a spill consumer that executes the guest first and preserves its
/// effective-address inputs until materialization. The CDNA rule matches
/// `shared/addr_calc_flat.h`, the simulator's effective-address source of truth.
[[nodiscard]] AtomicAddressPlan
plan_atomic_address(const AtomicLoweringForm &form, uint16_t scratch_vgpr,
                    uint16_t scratch_vgpr_count, RegisterAllocationSource resource_source,
                    bool allow_post_guest_spill_operand_overlap = false);

/// Emit the consumer-neutral VGLOBAL materialization. The returned sequence
/// snapshots and restores both VCC and SCC; @p vcc_save_sgpr names a pair and
/// @p scc_save_sgpr names one scalar register. A supported FLAT alias produces
/// an empty sequence.
[[nodiscard]] std::optional<std::vector<uint32_t>>
build_atomic_address_materialization(const AtomicAddressPlan &plan, uint16_t vcc_save_sgpr,
                                     uint16_t scc_save_sgpr, rj_code_arch_t arch);

/// Per-buffer safety bound for automatic ConSan reports.
inline constexpr uint64_t kOrdinaryAutoReportBufferCeilingBytes = 128u * 1024u * 1024u;
/// The hook reserves automatic report allocations against this process-wide
/// ceiling under its registry lock. It bounds aggregate storage independently
/// of the per-buffer limit.
inline constexpr uint64_t kAutoReportProcessCeilingBytes = 4ull * 1024u * 1024u * 1024u;
struct AutoReportProcessBudget {
  uint64_t current_live_bytes = 0;
  uint64_t peak_live_bytes = 0;
};

/// Atomically composed by the hook under its registry lock. Kept pure here so
/// boundary, overflow, multi-object, and cleanup behavior are independently
/// testable without HSA allocation APIs.
[[nodiscard]] constexpr bool reserve_auto_report_bytes(AutoReportProcessBudget &budget,
                                                       uint64_t bytes) {
  if (budget.current_live_bytes > kAutoReportProcessCeilingBytes ||
      bytes > kAutoReportProcessCeilingBytes - budget.current_live_bytes)
    return false;
  budget.current_live_bytes += bytes;
  budget.peak_live_bytes = std::max(budget.peak_live_bytes, budget.current_live_bytes);
  return true;
}

[[nodiscard]] constexpr bool release_auto_report_bytes(AutoReportProcessBudget &budget,
                                                       uint64_t bytes) {
  if (bytes > budget.current_live_bytes)
    return false;
  budget.current_live_bytes -= bytes;
  return true;
}

[[nodiscard]] constexpr bool report_header_is_current(const ReportHeader &header) {
  return header.magic == kReportMagic && header.abi_version == kReportAbiVersion &&
         header.header_size == sizeof(ReportHeader);
}
