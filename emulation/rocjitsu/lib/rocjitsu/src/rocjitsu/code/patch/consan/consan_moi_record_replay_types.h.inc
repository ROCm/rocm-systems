enum class ConSanMoiRecordReplayEventKind : uint16_t {
  Access = 1,
  Barrier = 2,
  Atomic = 3,
  Fence = 4,
};

inline constexpr uint32_t kConSanMoiRecordReplayTraceMagic = 0x31525243u; // "CRR1"
inline constexpr uint32_t kConSanMoiRecordReplayTraceAbiVersion = 4;
inline constexpr uint32_t kConSanMoiRecordReplayTraceHeaderBytes = 72;
inline constexpr uint32_t kConSanMoiRecordReplayTraceOverflow = 1u << 0u;
inline constexpr uint32_t kConSanMoiRecordReplayTraceRejectedInput = 1u << 1u;

/// Self-describing bounded compact-trace header. Generation and dispatch are
/// window-wide rather than repeated in every event.
struct alignas(8) ConSanMoiRecordReplayTraceHeader {
  uint32_t magic = kConSanMoiRecordReplayTraceMagic;
  uint32_t abi_version = kConSanMoiRecordReplayTraceAbiVersion;
  uint32_t header_size = kConSanMoiRecordReplayTraceHeaderBytes;
  uint32_t flags = 0;
  uint64_t generation = 0;
  uint64_t dispatch_id = 0;
  uint32_t dictionary_count = 0;
  uint32_t dictionary_capacity = 0;
  uint32_t workgroup_run_count = 0;
  uint32_t workgroup_run_capacity = 0;
  uint32_t event_count = 0;
  uint32_t event_capacity = 0;
  uint32_t dropped_event_count = 0;
  uint32_t rejected_event_count = 0;
  uint32_t lane_coalesced_record_count = 0;
  uint32_t reserved = 0;
};

/// Static PC dictionary entry. Operation is an access kind for Access, an
/// atomic event kind for Atomic, fence event kind for Fence, and zero for
/// Barrier.
struct alignas(8) ConSanMoiRecordReplayPcEntry {
  uint32_t instruction_offset = 0;
  ConSanMoiRecordReplayEventKind kind = ConSanMoiRecordReplayEventKind::Access;
  uint8_t operation = 0;
  ConSanMoiAtomicOperation atomic_operation = ConSanMoiAtomicOperation::Rmw;
  uint32_t scope = 0;
  uint32_t semantics = 0;

  constexpr ConSanMoiRecordReplayPcEntry() = default;
  constexpr ConSanMoiRecordReplayPcEntry(
      uint32_t instruction_offset_, ConSanMoiRecordReplayEventKind kind_, uint16_t operation_,
      uint32_t scope_, uint32_t semantics_,
      ConSanMoiAtomicOperation atomic_operation_ = ConSanMoiAtomicOperation::Rmw)
      : instruction_offset(instruction_offset_), kind(kind_),
        operation(static_cast<uint8_t>(operation_)), atomic_operation(atomic_operation_),
        scope(scope_), semantics(semantics_) {}
};

/// A maximal contiguous run of compact events for one workgroup after global
/// event-index ordering. Repeated workgroup coordinates are stored once.
struct alignas(8) ConSanMoiRecordReplayWorkgroupRun {
  uint32_t workgroup_x = 0;
  uint32_t workgroup_y = 0;
  uint32_t workgroup_z = 0;
  uint32_t first_event = 0;
  uint32_t event_count = 0;
  uint32_t reserved = 0;
};

/// Compact dynamic event. For accesses payload packs LDS byte offset/count;
/// for atomics it is the full address; for fences it is the nonzero exact
/// communication token; barriers use zero. Lane masks from otherwise
/// identical adjacent wave records are OR-coalesced.
struct alignas(8) ConSanMoiRecordReplayCompactEvent {
  uint32_t pc_index = 0;
  uint32_t event_index = 0;
  uint16_t owner_id = 0;
  ConSanMoiAtomicOutcome atomic_outcome = ConSanMoiAtomicOutcome::NotApplicable;
  uint32_t epoch = 0;
  uint64_t lane_mask = 0;
  uint64_t payload = 0;

  constexpr ConSanMoiRecordReplayCompactEvent() = default;
  constexpr ConSanMoiRecordReplayCompactEvent(
      uint32_t pc_index_, uint32_t event_index_, uint32_t owner_id_, uint32_t epoch_,
      uint64_t lane_mask_, uint64_t payload_,
      ConSanMoiAtomicOutcome atomic_outcome_ = ConSanMoiAtomicOutcome::NotApplicable)
      : pc_index(pc_index_), event_index(event_index_), owner_id(static_cast<uint16_t>(owner_id_)),
        atomic_outcome(atomic_outcome_), epoch(epoch_), lane_mask(lane_mask_), payload(payload_) {}
};

inline constexpr uint32_t kConSanMoiRecordReplayUncapturedEvent =
    std::numeric_limits<uint32_t>::max();

/// Independent bounds for selecting complete causal workgroup/epoch windows.
struct ConSanMoiRecordReplayCaptureLimits {
  uint32_t workgroup_limit = 0;
  uint32_t epochs_per_workgroup_limit = 0;
  uint32_t event_budget = 0;
};

/// One selected barrier-delimited epoch for one workgroup. Event positions are
/// positions in the globally ordered compact-event array and may include gaps
/// occupied by other workgroups; event_window_indices provides exact membership.
struct alignas(8) ConSanMoiRecordReplayCaptureWindow {
  uint32_t workgroup_x = 0;
  uint32_t workgroup_y = 0;
  uint32_t workgroup_z = 0;
  uint32_t epoch = 0;
  uint32_t first_event_position = 0;
  uint32_t last_event_position = 0;
  uint32_t first_event_index = 0;
  uint32_t last_event_index = 0;
  uint32_t event_count = 0;
  uint32_t reserved = 0;
};

struct ConSanMoiRecordReplayCaptureResult {
  uint32_t selected_workgroup_count = 0;
  uint32_t selected_window_count = 0;
  uint32_t selected_event_count = 0;
  uint32_t omitted_event_count = 0;
  bool invalid_trace = false;
  bool workgroup_limit_exhausted = false;
  bool epoch_limit_exhausted = false;
  bool event_budget_exhausted = false;
  bool window_capacity_exhausted = false;
};

/// The packed access remains first so ABI-v4-style probes cannot accidentally
/// publish into the version field while the final even/odd transaction covers
/// all three fields.
