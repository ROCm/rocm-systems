namespace rocjitsu {

enum class ConSanMoiDiagnosticKind : uint32_t {
  None = 0,
  AccessConflict = 1,
  MetadataFull = 2,
  BarrierDivergence = 3,
};

struct ConSanMoiWorkgroupShadowLayout {
  uint32_t base = 0;
  uint32_t size = 0;
  uint32_t validity_base = 0;
  uint32_t validity_size = 0;
  uint32_t required_group_segment_size = 0;
  uint8_t workitem_id_dimensions = 1;
  uint16_t initialization_lanes = 32;
  bool lazy_initialization = false;
  bool compact = false;
  std::optional<uint16_t> visible_evidence_sgpr;
};

// A lazy, full-width layout with no validity bitmap uses the canonical
// generation field to distinguish stale LDS from the current workgroup. This
// sentinel combination is impossible for the bitmap-backed layouts below, so
// it survives patch-info serialization without another ABI field.
[[nodiscard]] constexpr bool
consan_moi_generation_tagged_workgroup_shadow(const ConSanMoiWorkgroupShadowLayout &layout) {
  return layout.lazy_initialization && !layout.compact && layout.validity_size == 0u;
}

[[nodiscard]] constexpr bool
consan_moi_prefer_generation_tagged_workgroup_shadow(uint32_t guest_lds_bytes,
                                                     bool compact_cells_available = true) {
  // Prefer the bounded generation-tagged hot path for ordinary guest-LDS
  // sizes. The compact representation saves LDS, but its two-bit first-use
  // protocol adds an atomic claim and readiness poll to every newly touched
  // cell. Reserve it for unusually large mirrors where reducing LDS footprint
  // is worth that extra protocol.
  constexpr uint32_t kGenerationTaggedGuestLdsLimit = 64u * 1024u;
  return !compact_cells_available || guest_lds_bytes <= kGenerationTaggedGuestLdsLimit;
}

[[nodiscard]] constexpr uint32_t consan_moi_max_workgroup_lds_bytes(rj_code_arch_t arch) {
  // gfx1250 follows the simulator configuration rather than duplicating a
  // provisional architectural limit in ConSan. Earlier targets retain the
  // 64 KiB limit used by the original planner.
  return arch == ROCJITSU_CODE_ARCH_GFX1250
             ? static_cast<uint32_t>(ROCJITSU_GFX1250_LDS_SIZE_KB) * 1024u
             : 64u * 1024u;
}

[[nodiscard]] constexpr uint32_t consan_moi_max_workgroup_lds_bytes(const ConSanOptions &options,
                                                                    rj_code_arch_t arch) {
  return options.moi_max_workgroup_lds_bytes.value_or(consan_moi_max_workgroup_lds_bytes(arch));
}

/// One eight-byte shadow slot covers one four-byte guest-LDS granule. Keeping
/// the mirror in the owning workgroup's LDS makes the storage bounded by the
/// kernel descriptor and prevents independent workgroups from contending on a
/// process-global slot. Dynamic-only LDS and kernels that cannot fit the
/// mirror in the target's architectural workgroup limit are not applicable.
[[nodiscard]] constexpr std::optional<ConSanMoiWorkgroupShadowLayout>
plan_consan_moi_workgroup_shadow(uint32_t original_group_segment_size,
                                 uint32_t max_workgroup_lds_bytes = 64u * 1024u) {
  constexpr uint32_t kAlignment = 8u;
  constexpr uint32_t kGuestGranuleBytes = 4u;
  if (original_group_segment_size == 0u)
    return std::nullopt;
  const uint64_t base = (static_cast<uint64_t>(original_group_segment_size) + kAlignment - 1u) &
                        ~(static_cast<uint64_t>(kAlignment) - 1u);
  const uint64_t slot_count =
      (static_cast<uint64_t>(original_group_segment_size) + kGuestGranuleBytes - 1u) /
      kGuestGranuleBytes;
  const uint64_t size = slot_count * sizeof(uint64_t);
  const uint64_t required = base + size;
  if (required > max_workgroup_lds_bytes)
    return std::nullopt;
  return ConSanMoiWorkgroupShadowLayout{.base = static_cast<uint32_t>(base),
                                        .size = static_cast<uint32_t>(size),
                                        .required_group_segment_size =
                                            static_cast<uint32_t>(required),
                                        .workitem_id_dimensions = 1u,
                                        .visible_evidence_sgpr = std::nullopt};
}

/// Retain the exact eight-byte owner/epoch slot for every guest-LDS granule,
/// but initialize it on first use. Two packed state bits per slot encode
/// unclaimed, initializing, and ready. Only this small bitmap needs an eager
/// workgroup-wide clear; a ready bit is published after the winning lane has
/// zeroed the corresponding exact slot.
[[nodiscard]] constexpr std::optional<ConSanMoiWorkgroupShadowLayout>
plan_consan_moi_lazy_workgroup_shadow(uint32_t original_group_segment_size,
                                      uint32_t max_workgroup_lds_bytes = 64u * 1024u) {
  constexpr uint32_t kStatesPerWord = 16u;
  constexpr uint32_t kInitializationAlignment = 16u;
  auto layout =
      plan_consan_moi_workgroup_shadow(original_group_segment_size, max_workgroup_lds_bytes);
  if (!layout)
    return std::nullopt;
  const uint64_t slot_count = layout->size / sizeof(uint64_t);
  const uint64_t state_word_count = (slot_count + kStatesPerWord - 1u) / kStatesPerWord;
  const uint64_t raw_validity_size = state_word_count * sizeof(uint32_t);
  const uint64_t validity_size = (raw_validity_size + kInitializationAlignment - 1u) &
                                 ~(static_cast<uint64_t>(kInitializationAlignment) - 1u);
  const uint64_t validity_base = layout->required_group_segment_size;
  const uint64_t required = validity_base + validity_size;
  if (validity_size == 0u || required > max_workgroup_lds_bytes)
    return std::nullopt;
  layout->validity_base = static_cast<uint32_t>(validity_base);
  layout->validity_size = static_cast<uint32_t>(validity_size);
  layout->required_group_segment_size = static_cast<uint32_t>(required);
  layout->lazy_initialization = true;
  return layout;
}

/// Keep one canonical eight-byte exact cell per guest-LDS granule and use its
/// existing 20-bit generation field as the workgroup identity. Unlike an
/// eagerly cleared mirror or a bitmap-backed lazy mirror, stale LDS is rejected
/// by the normal exact-shadow comparison and every hot access needs only its
/// exact exchange.
[[nodiscard]] constexpr std::optional<ConSanMoiWorkgroupShadowLayout>
plan_consan_moi_generation_tagged_workgroup_shadow(uint32_t original_group_segment_size,
                                                   uint32_t max_workgroup_lds_bytes = 64u * 1024u) {
  auto layout =
      plan_consan_moi_workgroup_shadow(original_group_segment_size, max_workgroup_lds_bytes);
  if (!layout)
    return std::nullopt;
  layout->lazy_initialization = true;
  return layout;
}

/// Use one four-byte local cell per four-byte guest-LDS granule. The compact
/// cell keeps exact kind, owner, and epoch fields; its nine remaining bits are
/// an object-unique site token resolved to the full instruction offset by the
/// report path. A packed two-bit first-use bitmap avoids clearing the full
/// compact mirror in every workgroup.
[[nodiscard]] constexpr std::optional<ConSanMoiWorkgroupShadowLayout>
plan_consan_moi_compact_workgroup_shadow(uint32_t original_group_segment_size,
                                         uint32_t max_workgroup_lds_bytes = 64u * 1024u) {
  constexpr uint32_t kAlignment = 16u;
  constexpr uint32_t kGuestGranuleBytes = 4u;
  constexpr uint32_t kStatesPerWord = 16u;
  if (original_group_segment_size == 0u)
    return std::nullopt;
  const uint64_t base = (static_cast<uint64_t>(original_group_segment_size) + kAlignment - 1u) &
                        ~(static_cast<uint64_t>(kAlignment) - 1u);
  const uint64_t slot_count =
      (static_cast<uint64_t>(original_group_segment_size) + kGuestGranuleBytes - 1u) /
      kGuestGranuleBytes;
  const uint64_t raw_size = slot_count * sizeof(uint32_t);
  const uint64_t size = (raw_size + kAlignment - 1u) & ~(static_cast<uint64_t>(kAlignment) - 1u);
  const uint64_t validity_base = base + size;
  const uint64_t state_word_count = (slot_count + kStatesPerWord - 1u) / kStatesPerWord;
  const uint64_t raw_validity_size = state_word_count * sizeof(uint32_t);
  const uint64_t validity_size =
      (raw_validity_size + kAlignment - 1u) & ~(static_cast<uint64_t>(kAlignment) - 1u);
  const uint64_t required = validity_base + validity_size;
  if (required > max_workgroup_lds_bytes)
    return std::nullopt;
  return ConSanMoiWorkgroupShadowLayout{.base = static_cast<uint32_t>(base),
                                        .size = static_cast<uint32_t>(size),
                                        .validity_base = static_cast<uint32_t>(validity_base),
                                        .validity_size = static_cast<uint32_t>(validity_size),
                                        .required_group_segment_size =
                                            static_cast<uint32_t>(required),
                                        .workitem_id_dimensions = 1u,
                                        .lazy_initialization = true,
                                        .compact = true,
                                        .visible_evidence_sgpr = std::nullopt};
}

enum class ConSanMoiInlineAtomicSupport : uint8_t {
  Supported,
  NonFlatAddress,
  NonzeroOffset,
  UnsupportedScope,
  UnsupportedWidth,
  CompareExchangeOutcomeUnavailable,
  UnsupportedEncoding,
  MissingOperands,
  MissingOrderingMetadata,
};

/// Consumer-neutral effective-address recipe for MOI atomic instrumentation.
/// Inline-shadow and record/replay must use the same recipe so that they key
/// synchronization metadata by exactly the address consumed by the guest
/// instruction.
enum class ConSanMoiAtomicAddressKind : uint8_t {
  Unsupported,
  LdsByteOffsetToken,
  FlatGuestPair,
  FlatGuestPairMaterialized,
  VglobalGuestPair,
  VglobalGuestPairMaterialized,
  VglobalMaterialized,
  BufferResourceMaterialized,
};

enum class ConSanMoiAtomicAddressSupport : uint8_t {
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

struct ConSanMoiAtomicAddressPlan {
  ConSanMoiAtomicAddressKind kind = ConSanMoiAtomicAddressKind::Unsupported;
  ConSanMoiAtomicAddressSupport support = ConSanMoiAtomicAddressSupport::UnsupportedAddressKind;
  uint16_t input_address_vgpr = 0;
  uint16_t input_address_vgpr_count = 0;
  std::optional<uint16_t> scalar_base_sgpr;
  std::optional<uint16_t> scalar_offset_sgpr;
  int32_t signed_byte_offset = 0;
  uint16_t result_address_vgpr = 0;
  uint16_t result_address_vgpr_count = 0;
  uint16_t scratch_vgpr = 0;
  uint16_t scratch_vgpr_count = 0;
  ConSanRegisterAllocationSource resource_source = ConSanRegisterAllocationSource::Unsupported;

  [[nodiscard]] constexpr bool supported() const {
    return support == ConSanMoiAtomicAddressSupport::Supported;
  }
  [[nodiscard]] constexpr bool requires_materialization() const {
    return kind == ConSanMoiAtomicAddressKind::LdsByteOffsetToken ||
           kind == ConSanMoiAtomicAddressKind::FlatGuestPairMaterialized ||
           kind == ConSanMoiAtomicAddressKind::VglobalGuestPairMaterialized ||
           kind == ConSanMoiAtomicAddressKind::VglobalMaterialized ||
           kind == ConSanMoiAtomicAddressKind::BufferResourceMaterialized;
  }
};

// Record/replay state is partitioned by workgroup. Within that state, use a
// tagged 64-bit token for LDS so a byte offset cannot alias a global address.
inline constexpr uint32_t kConSanMoiLdsAddressTokenTag = 0xffffffffu;

[[nodiscard]] std::string_view
consan_moi_atomic_address_support_name(ConSanMoiAtomicAddressSupport support);

/// Build a typed effective-address plan from decoded atomic operands and an
/// already resolved scratch-resource plan. FLAT aliases its guest 64-bit
/// address unless a returned value aliases it, in which case it snapshots the
/// pair into the scratch tail. VGLOBAL likewise reserves the last two VGPRs of
/// a five-VGPR private window for an aliased pair or for
/// `saddr + zero_extend(vaddr) + signed24(ioffset)`.
[[nodiscard]] ConSanMoiAtomicAddressPlan
plan_consan_moi_atomic_address(const ConSanAtomicSite &site, uint16_t scratch_vgpr,
                               uint16_t scratch_vgpr_count,
                               ConSanRegisterAllocationSource resource_source, rj_code_arch_t arch);

/// Emit the consumer-neutral VGLOBAL materialization. The returned sequence
/// snapshots and restores both VCC and SCC; @p vcc_save_sgpr names a pair and
/// @p scc_save_sgpr names one scalar register. A supported FLAT alias produces
/// an empty sequence.
[[nodiscard]] std::optional<std::vector<uint32_t>>
build_consan_moi_atomic_address_materialization(const ConSanMoiAtomicAddressPlan &plan,
                                                uint16_t vcc_save_sgpr, uint16_t scc_save_sgpr,
                                                rj_code_arch_t arch);

[[nodiscard]] constexpr ConSanMoiInlineAtomicSupport
classify_consan_moi_inline_atomic_support(const ConSanAtomicSite &site,
                                          ConSanMoiAtomicEventKind event_kind) {
  const bool is_flat = site.mnemonic.starts_with("flat_atomic");
  const bool is_vglobal = site.mnemonic.starts_with("global_atomic");
  if (!is_flat && !is_vglobal)
    return ConSanMoiInlineAtomicSupport::NonFlatAddress;
  const bool is_compare_exchange = site.mnemonic.find("cmpswap") != std::string::npos ||
                                   site.mnemonic.find("cmpxchg") != std::string::npos;
  (void)event_kind;
  if (site.width_bits != 32)
    return ConSanMoiInlineAtomicSupport::UnsupportedWidth;
  const bool vector_flat_encoding =
      is_flat && site.size == 3u * sizeof(uint32_t) && site.raw_saddr &&
      *site.raw_saddr == 0x7cu;
  const bool cdna4_flat_encoding =
      is_flat && site.size == 2u * sizeof(uint32_t) && site.raw_saddr && *site.raw_saddr == 0u;
  const bool vglobal_encoding = is_vglobal && site.size == 3u * sizeof(uint32_t);
  if (!vector_flat_encoding && !cdna4_flat_encoding && !vglobal_encoding)
    return ConSanMoiInlineAtomicSupport::UnsupportedEncoding;
  if (!site.raw_ioffset || (is_flat && *site.raw_ioffset != 0))
    return ConSanMoiInlineAtomicSupport::NonzeroOffset;
  if (!site.addr_vgpr || *site.addr_vgpr >= 255 || !site.data_vgpr ||
      (is_compare_exchange && *site.data_vgpr >= 255))
    return ConSanMoiInlineAtomicSupport::MissingOperands;
  // A no-return compare-exchange has no destination operand by construction.
  // Classify its missing success evidence before requiring the destination
  // that only the returning form can provide.
  if (is_compare_exchange && site.returns_old_value && !*site.returns_old_value)
    return ConSanMoiInlineAtomicSupport::CompareExchangeOutcomeUnavailable;
  if (is_compare_exchange && !site.dst_vgpr)
    return ConSanMoiInlineAtomicSupport::MissingOperands;
  if (!site.raw_scope || !site.returns_old_value ||
      ((vector_flat_encoding || vglobal_encoding) && !site.raw_th))
    return ConSanMoiInlineAtomicSupport::MissingOrderingMetadata;
  // Wave scope is not an inter-wave handoff. Workgroup, agent, and system
  // atomics are all valid causal publication/import events; the Inline keys
  // already include dispatch and workgroup identity, so scope 1 needs no
  // broader alias domain than the existing transaction.
  if (*site.raw_scope < 1u || *site.raw_scope > 3u)
    return ConSanMoiInlineAtomicSupport::UnsupportedScope;
  return ConSanMoiInlineAtomicSupport::Supported;
}

[[nodiscard]] constexpr std::string_view
consan_moi_inline_atomic_support_name(ConSanMoiInlineAtomicSupport support) {
  switch (support) {
  case ConSanMoiInlineAtomicSupport::Supported:
    return "supported";
  case ConSanMoiInlineAtomicSupport::NonFlatAddress:
    return "non-flat-address";
  case ConSanMoiInlineAtomicSupport::NonzeroOffset:
    return "nonzero-offset";
  case ConSanMoiInlineAtomicSupport::UnsupportedScope:
    return "unsupported-scope";
  case ConSanMoiInlineAtomicSupport::UnsupportedWidth:
    return "unsupported-width";
  case ConSanMoiInlineAtomicSupport::CompareExchangeOutcomeUnavailable:
    return "compare-exchange-outcome-unavailable";
  case ConSanMoiInlineAtomicSupport::UnsupportedEncoding:
    return "unsupported-encoding";
  case ConSanMoiInlineAtomicSupport::MissingOperands:
    return "missing-operands";
  case ConSanMoiInlineAtomicSupport::MissingOrderingMetadata:
    return "missing-ordering-metadata";
  }
  return "unknown";
}

inline constexpr uint64_t kConSanMoiRecordReplayDefaultReportBufferBytes = 64u * 1024u;
inline constexpr uint64_t kConSanMoiSampledDefaultReportBufferBytes = 64u * 1024u;
inline constexpr uint64_t kConSanMoiInlineShadowDefaultReportBufferBytes = 512u * 1024u;
/// Hard ordinary/expert safety bounds. G6B3 owns aggregate reservation against
/// the process ceiling; this pure planner enforces the per-buffer ceiling.
inline constexpr uint64_t kConSanMoiAutoReportBufferCeilingBytes = 128u * 1024u * 1024u;
inline constexpr uint64_t kConSanMoiAutoReportProcessCeilingBytes = 256u * 1024u * 1024u;
// Record/replay synchronization probes can execute many times per static site,
// and a code-object-scoped report remains live across repeated dispatches. Keep
// enough bounded dynamic headroom for ordinary validation and benchmark loops
// without requiring a report-size tuning knob.
inline constexpr uint64_t kConSanMoiRecordReplayDynamicEventHeadroom = 1024u;

struct ConSanMoiAutoReportProcessBudget {
  uint64_t current_live_bytes = 0;
  uint64_t peak_live_bytes = 0;
};

/// Atomically composed by the hook under its registry lock. Kept pure here so
/// boundary, overflow, multi-object, and cleanup behavior are independently
/// testable without HSA allocation APIs.
[[nodiscard]] constexpr bool
reserve_consan_moi_auto_report_bytes(ConSanMoiAutoReportProcessBudget &budget, uint64_t bytes) {
  if (budget.current_live_bytes > kConSanMoiAutoReportProcessCeilingBytes ||
      bytes > kConSanMoiAutoReportProcessCeilingBytes - budget.current_live_bytes)
    return false;
  budget.current_live_bytes += bytes;
  budget.peak_live_bytes = std::max(budget.peak_live_bytes, budget.current_live_bytes);
  return true;
}

[[nodiscard]] constexpr bool
release_consan_moi_auto_report_bytes(ConSanMoiAutoReportProcessBudget &budget, uint64_t bytes) {
  if (bytes > budget.current_live_bytes)
    return false;
  budget.current_live_bytes -= bytes;
  return true;
}

enum class ConSanMoiReportLayoutFlag : uint32_t {
  None = 0,
  VersionedInlineExactShadow = 1u << 0u,
  DispatchQualifiedInlineMetadata = 1u << 1u,
  InlineCausalSnapshots = 1u << 2u,
};

inline constexpr uint32_t kConSanMoiReportKnownLayoutFlags =
    static_cast<uint32_t>(ConSanMoiReportLayoutFlag::VersionedInlineExactShadow) |
    static_cast<uint32_t>(ConSanMoiReportLayoutFlag::DispatchQualifiedInlineMetadata) |
    static_cast<uint32_t>(ConSanMoiReportLayoutFlag::InlineCausalSnapshots);

[[nodiscard]] constexpr uint64_t
consan_moi_default_auto_report_buffer_size(ConSanMoiEngine engine) {
  switch (engine) {
  case ConSanMoiEngine::RecordReplay:
    return kConSanMoiRecordReplayDefaultReportBufferBytes;
  case ConSanMoiEngine::InlineShadow:
    return kConSanMoiInlineShadowDefaultReportBufferBytes;
  case ConSanMoiEngine::Sampled:
    return kConSanMoiSampledDefaultReportBufferBytes;
  }
  return 0;
}

[[nodiscard]] constexpr bool
consan_moi_report_header_is_current(const ConSanMoiReportHeader &header) {
  return header.magic == kConSanMoiReportMagic &&
         header.abi_version == kConSanMoiReportAbiVersion &&
         header.header_size == sizeof(ConSanMoiReportHeader) &&
         header.engine <= static_cast<uint32_t>(ConSanMoiEngine::Sampled) &&
         (header.layout_flags & ~kConSanMoiReportKnownLayoutFlags) == 0;
}

/// Static event classification stored once per instrumented PC in a compact
/// record/replay trace. Dynamic records refer to this table by index.
