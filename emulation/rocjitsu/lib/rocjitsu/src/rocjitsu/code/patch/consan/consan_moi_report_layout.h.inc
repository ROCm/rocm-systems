struct alignas(8) ConSanMoiDiagnosticRecord {
  uint32_t kind = static_cast<uint32_t>(ConSanMoiDiagnosticKind::None);
  uint32_t backend = static_cast<uint32_t>(ConSanMoiEngine::RecordReplay);
  // Report generation for ordinary records; native dispatch identity for
  // InlineShadow atomic-tracking diagnostics.
  uint64_t generation = 0;
  uint32_t epoch = 0;
  // Epoch of the prior exact-shadow access. InlineShadow atomic-tracking
  // diagnostics use this with the token's consumer segment epoch to prove
  // that an acquire preceded that specific access.
  uint32_t first_epoch = 0;
  uint32_t first_owner_id = 0;
  uint32_t second_owner_id = 0;
  // InlineShadow atomic-tracking diagnostics retain the persistent workgroup
  // key here so teardown can qualify a token publication that completed after
  // the device-side diagnostic snapshot. Zero for other engines/configurations.
  uint32_t reserved = 0;
  uint64_t first_lane_mask = 0;
  uint64_t second_lane_mask = 0;
  uint32_t first_instruction_offset = 0;
  uint32_t second_instruction_offset = 0;
  uint32_t first_lds_byte_offset = 0;
  uint32_t first_lds_byte_count = 0;
  uint32_t second_lds_byte_offset = 0;
  uint32_t second_lds_byte_count = 0;
  uint32_t first_access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Empty);
  uint32_t second_access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Empty);
};

/// Host-populated lookup record for the compact workgroup-local shadow.
/// Compact cells retain a per-kernel token instead of a full instruction
/// offset. Keeping one descriptor-qualified record per patched site in the
/// report allocation lets teardown resolve that token without an unbounded
/// process-heap side table.
struct alignas(8) ConSanMoiCompactDiagnosticTokenMapping {
  uint64_t owner_descriptor_file_offset = 0;
  uint32_t instruction_offset = 0;
  uint16_t token = 0;
  uint16_t reserved = 0;
};
static_assert(sizeof(ConSanMoiCompactDiagnosticTokenMapping) == 16);

struct ConSanMoiReportBufferLayout {
  uint32_t access_record_capacity = 0;
  uint32_t record_replay_logical_access_range_count = 0;
  uint32_t record_replay_dispatch_token_capacity = 0;
  uint32_t record_replay_access_dispatch_bank_count = 1;
  uint32_t record_replay_access_owner_bank_count = 1;
  uint32_t barrier_record_capacity = 0;
  uint32_t atomic_record_capacity = 0;
  uint32_t fence_record_capacity = 0;
  uint32_t diagnostic_capacity = 0;
  uint32_t exact_shadow_entry_capacity = 0;
  uint32_t inline_exact_dispatch_bank_count = 0;
  uint32_t inline_atomic_release_capacity = 0;
  uint32_t inline_acquired_epoch_token_capacity = 0;
  uint32_t inline_causal_snapshot_capacity = 0;
  uint32_t inline_compact_token_mapping_capacity = 0;
  uint32_t sampled_watchpoint_capacity = 0;
  uint32_t sampled_causal_window_capacity = 0;
  uint32_t sampled_sync_metadata_capacity = 0;
  uint32_t sampled_pending_acquire_capacity = 0;
  size_t record_replay_dispatch_tokens_offset = sizeof(ConSanMoiReportHeader);
  size_t access_records_offset = sizeof(ConSanMoiReportHeader);
  size_t barrier_records_offset = sizeof(ConSanMoiReportHeader);
  size_t atomic_records_offset = sizeof(ConSanMoiReportHeader);
  size_t fence_records_offset = sizeof(ConSanMoiReportHeader);
  size_t diagnostic_records_offset = sizeof(ConSanMoiReportHeader);
  size_t exact_shadow_entries_offset = sizeof(ConSanMoiReportHeader);
  size_t inline_atomic_release_slots_offset = sizeof(ConSanMoiReportHeader);
  size_t inline_acquired_epoch_token_slots_offset = sizeof(ConSanMoiReportHeader);
  size_t inline_causal_snapshots_offset = sizeof(ConSanMoiReportHeader);
  size_t inline_compact_token_mappings_offset = sizeof(ConSanMoiReportHeader);
  size_t sampled_watchpoints_offset = sizeof(ConSanMoiReportHeader);
  size_t sampled_causal_windows_offset = sizeof(ConSanMoiReportHeader);
  size_t sampled_sync_metadata_offset = sizeof(ConSanMoiReportHeader);
  size_t sampled_pending_acquires_offset = sizeof(ConSanMoiReportHeader);
  size_t required_bytes = sizeof(ConSanMoiReportHeader);
  bool valid = false;
};

/// Creates a header from the exact heterogeneous layout. The legacy
/// make_consan_moi_report_header helper intentionally keeps its historical
/// single Sampled capacity contract for explicit size-derived buffers.
[[nodiscard]] constexpr ConSanMoiReportHeader
make_consan_moi_report_header_for_layout(uint64_t generation, uint64_t dispatch_id,
                                         const ConSanMoiReportBufferLayout &layout,
                                         ConSanMoiEngine engine) {
  ConSanMoiReportHeader header;
  header.header_size = sizeof(ConSanMoiReportHeader);
  header.generation = generation;
  header.dispatch_id = dispatch_id;
  header.engine = static_cast<uint32_t>(engine);
  header.record_replay_dispatch_token_capacity =
      layout.record_replay_dispatch_token_capacity;
  header.access_record_capacity = layout.access_record_capacity;
  header.barrier_record_capacity = layout.barrier_record_capacity;
  header.atomic_record_capacity = layout.atomic_record_capacity;
  header.fence_record_capacity = layout.fence_record_capacity;
  header.diagnostic_capacity = layout.diagnostic_capacity;
  header.exact_shadow_entry_capacity = layout.exact_shadow_entry_capacity;
  header.inline_atomic_release_capacity = layout.inline_atomic_release_capacity;
  header.inline_acquired_epoch_token_capacity = layout.inline_acquired_epoch_token_capacity;
  header.inline_causal_snapshot_capacity = layout.inline_causal_snapshot_capacity;
  header.sampled_watchpoint_capacity = layout.sampled_watchpoint_capacity;
  header.sampled_causal_window_capacity = layout.sampled_causal_window_capacity;
  header.sampled_sync_metadata_capacity = layout.sampled_sync_metadata_capacity;
  header.sampled_pending_acquire_capacity = layout.sampled_pending_acquire_capacity;
  if (engine == ConSanMoiEngine::InlineShadow)
    header.layout_flags = kConSanMoiReportKnownLayoutFlags;
  return header;
}

/// Ordinary automatic Record/Replay uses a cheap report-wide dispatch
/// directory plus one report-wide access-identity table. The two access-bank
/// dimensions are sizing factors: the planner reserves that many anticipated
/// identities per static range, applies standard open-addressing headroom,
/// rounds the total table to a power of two, and the device probes by static
/// range + dispatch + workgroup + wave. Hot sites can therefore borrow unused
/// capacity from cold sites without workload-specific sizing. Large code
/// objects halve both factors under the existing per-buffer ceiling; direct
/// layouts keep their historical 1x1 contract.
inline constexpr uint32_t kConSanMoiRecordReplayMaximumDispatchBankCount = 16u;
inline constexpr uint32_t kConSanMoiRecordReplayMaximumOwnerBankCount = 16u;
inline constexpr uint32_t kConSanMoiRecordReplayHashTableHeadroom = 2u;
inline constexpr uint32_t kConSanMoiRecordReplayAnticipatedDispatchCount = 1024u;
inline constexpr uint32_t kConSanMoiRecordReplayMaximumDispatchTokenCount =
    kConSanMoiRecordReplayAnticipatedDispatchCount *
    kConSanMoiRecordReplayHashTableHeadroom;
inline constexpr uint32_t kConSanMoiRecordReplayIdentityHashMultiplier = 0x9e3779b1u;
/// Device-side open-addressing must have a practical bound independent of the
/// report allocation size. Exhaustion sets the typed saturation flag; later
/// probes observe that flag and bypass capture instead of repeatedly walking a
/// full table.
inline constexpr uint32_t kConSanMoiRecordReplayProbeLimit = 256u;
[[nodiscard]] constexpr uint32_t
consan_moi_record_replay_fold_dispatch_token(uint64_t token) {
  return static_cast<uint32_t>(token) ^ static_cast<uint32_t>(token >> 32u);
}
[[nodiscard]] constexpr uint32_t
consan_moi_record_replay_mix_identity(uint32_t hash, uint32_t coordinate) {
  return (hash * kConSanMoiRecordReplayIdentityHashMultiplier) ^ coordinate;
}
/// Advances an open-addressing probe after @p probe_count has been incremented.
/// Repeated calls therefore visit home + 1, home + 3, home + 6, ... . This
/// triangular sequence covers every slot in a power-of-two table.
/// @pre @p table_capacity is a nonzero power of two.
[[nodiscard]] constexpr uint32_t
consan_moi_record_replay_advance_probe(uint32_t bank, uint32_t probe_count,
                                      uint32_t table_capacity) {
  return (bank + probe_count) & (table_capacity - 1u);
}
static_assert((kConSanMoiRecordReplayMaximumDispatchTokenCount &
               (kConSanMoiRecordReplayMaximumDispatchTokenCount - 1u)) == 0u);
static_assert(kConSanMoiRecordReplayMaximumDispatchTokenCount ==
              kConSanMoiRecordReplayAnticipatedDispatchCount *
                  kConSanMoiRecordReplayHashTableHeadroom);
static_assert((kConSanMoiRecordReplayMaximumDispatchBankCount &
               (kConSanMoiRecordReplayMaximumDispatchBankCount - 1u)) == 0u);
static_assert((kConSanMoiRecordReplayMaximumOwnerBankCount &
               (kConSanMoiRecordReplayMaximumOwnerBankCount - 1u)) == 0u);
static_assert(kConSanMoiRecordReplayHashTableHeadroom >= 2u);
static_assert((kConSanMoiRecordReplayProbeLimit &
               (kConSanMoiRecordReplayProbeLimit - 1u)) == 0u);

/// Final-code inventory reduced to the quantities needed to size one report
/// ABI. Counts are deliberately independent: candidate extraction must not
/// assume that one sampled causal window holds only one watchpoint, or that
/// all Inline ordering tables have the same capacity.
struct ConSanMoiAutoReportInventory {
  ConSanMoiEngine engine = ConSanMoiEngine::RecordReplay;
  uint64_t access_range_count = 0;
  uint64_t barrier_event_count = 0;
  uint64_t atomic_event_count = 0;
  uint64_t fence_event_count = 0;
  uint64_t diagnostic_count = 0;

  /// Automatic static capture starts with bounded dispatch/owner diversity
  /// factors used only to size the report-wide access-identity table. The
  /// fitter reduces both factors together when a large object would exceed the
  /// ordinary report ceiling. Its separate dispatch directory retains the full
  /// lifetime bound while access storage adapts. Explicit plans remain exact.
  uint64_t record_replay_dispatch_token_capacity =
      kConSanMoiRecordReplayMaximumDispatchTokenCount;
  uint64_t record_replay_access_dispatch_bank_count =
      kConSanMoiRecordReplayMaximumDispatchBankCount;
  uint64_t record_replay_access_owner_bank_count =
      kConSanMoiRecordReplayMaximumOwnerBankCount;
  bool record_replay_bank_count_adaptive = false;

  /// Total immutable banks across admitted Sampled logical ranges. One causal
  /// window, synchronization-metadata slot, and pending-acquire slot are
  /// reserved per bank. Multi-cell ranges can require more watchpoints.
  uint64_t sampled_range_bank_count = 0;
  uint64_t sampled_watchpoint_count = 0;
  /// Automatic runtime sampling may reduce the number of independent banks,
  /// while retaining at least one bank for every logical access range, to fit
  /// the frozen per-object report ceiling. Explicit inventories remain exact.
  bool sampled_bank_count_adaptive = false;

  /// Maximum LDS span covered by an admitted Inline site. Native sites use the
  /// owning kernel's declared segment; generic-flat sites require the full
  /// architectural aperture because their runtime offset is not statically
  /// bounded by that declaration. The planner rounds this span to four-byte
  /// guest cells.
  uint64_t inline_lds_bytes = 0;
  uint64_t inline_atomic_release_count = 0;
  uint64_t inline_causal_snapshot_count = 0;
  uint64_t inline_acquired_epoch_token_count = 0;
  uint64_t inline_compact_token_mapping_count = 0;
};

enum class ConSanMoiAutoReportPlanOutcome : uint8_t {
  Complete,
  InsufficientReportCapacity,
  Overflow,
};

enum class ConSanMoiAutoReportPlanReason : uint8_t {
  None,
  PerBufferCeiling,
  AbiCapacityOverflow,
  ByteSizeOverflow,
};

struct ConSanMoiAutoReportPlan {
  ConSanMoiAutoReportPlanOutcome outcome = ConSanMoiAutoReportPlanOutcome::Overflow;
  ConSanMoiAutoReportPlanReason reason = ConSanMoiAutoReportPlanReason::ByteSizeOverflow;
  ConSanMoiEngine engine = ConSanMoiEngine::RecordReplay;
  uint64_t required_bytes = 0;
  uint64_t ceiling_bytes = kConSanMoiAutoReportBufferCeilingBytes;
  ConSanMoiReportBufferLayout layout;

  [[nodiscard]] constexpr bool complete() const {
    return outcome == ConSanMoiAutoReportPlanOutcome::Complete;
  }
};

/// Converts a complete planner result into the caller-carried exact layout.
/// An incomplete plan produces std::nullopt.
[[nodiscard]] std::optional<ConSanMoiReportLayoutOverride>
consan_moi_auto_report_layout_override(const ConSanMoiAutoReportPlan &plan);

/// Revalidates a caller-carried layout against the canonical planner and the
/// registered allocation size. A malformed, noncanonical, wrong-engine, or
/// undersized override returns an invalid layout.
[[nodiscard]] ConSanMoiReportBufferLayout
consan_moi_report_layout_from_override(const ConSanMoiReportLayoutOverride &override_layout,
                                       ConSanMoiEngine engine, uint64_t report_buffer_size);

[[nodiscard]] constexpr std::string_view
consan_moi_auto_report_plan_outcome_name(ConSanMoiAutoReportPlanOutcome outcome) {
  switch (outcome) {
  case ConSanMoiAutoReportPlanOutcome::Complete:
    return "complete";
  case ConSanMoiAutoReportPlanOutcome::InsufficientReportCapacity:
    return "insufficient_report_capacity";
  case ConSanMoiAutoReportPlanOutcome::Overflow:
    return "overflow";
  }
  return "overflow";
}

[[nodiscard]] constexpr std::string_view
consan_moi_auto_report_plan_reason_name(ConSanMoiAutoReportPlanReason reason) {
  switch (reason) {
  case ConSanMoiAutoReportPlanReason::None:
    return "none";
  case ConSanMoiAutoReportPlanReason::PerBufferCeiling:
    return "per_buffer_ceiling";
  case ConSanMoiAutoReportPlanReason::AbiCapacityOverflow:
    return "abi_capacity_overflow";
  case ConSanMoiAutoReportPlanReason::ByteSizeOverflow:
    return "byte_size_overflow";
  }
  return "byte_size_overflow";
}

/// Computes an exact ABI layout from explicit inventory without allocating,
/// reading code-object bytes, or consulting environment state. Counts that do
/// not fit the ABI produce `overflow`; a representable plan above 16 MiB
/// produces `insufficient_report_capacity` while retaining required bytes and
/// capacities for telemetry.
[[nodiscard]] ConSanMoiAutoReportPlan
plan_consan_moi_auto_report(const ConSanMoiAutoReportInventory &inventory);

/// Expands static Record/Replay synchronization counts by the largest
/// power-of-two dynamic headroom that fits the ordinary per-buffer ceiling.
/// Small objects retain the standard headroom; fat objects retain every static
/// site with a smaller, still fail-closed dynamic capacity.
[[nodiscard]] ConSanMoiAutoReportInventory
fit_consan_moi_record_replay_auto_report_inventory(ConSanMoiAutoReportInventory inventory);

/// Reduces only an explicitly adaptive Sampled inventory by powers of two
/// until its complete logical-range population fits the report ceiling or one
/// bank per range remains. All other inventory dimensions remain exact.
[[nodiscard]] ConSanMoiAutoReportInventory
fit_consan_moi_sampled_auto_report_inventory(ConSanMoiAutoReportInventory inventory);

/// Extracts the bounded ordinary report requirements from one completed
/// inventory transform. This does not allocate and deliberately does not try
/// to predict expert dynamic per-lane replay volume.
[[nodiscard]] ConSanMoiAutoReportInventory
inventory_consan_moi_auto_report(const ConSanResult &result, const ConSanOptions &options,
                                 std::span<const uint8_t> code_object_bytes);
