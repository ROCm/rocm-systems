struct ConSanMoiExactByteAccess {
  uint64_t generation = 0;
  uint32_t owner_id = 0;
  uint32_t epoch = 0;
  ConSanMoiShadowAccessKind kind = ConSanMoiShadowAccessKind::Empty;
  uint32_t lds_byte_offset = 0;
  uint32_t lds_byte_count = 0;
  uint32_t instruction_offset = 0;
  uint64_t lane_mask = 0;
  bool exact_address_group = false;
};

struct ConSanMoiExactByteAccessResult {
  bool metadata_full = false;
  bool conflict = false;
  ConSanMoiDiagnosticRecord diagnostic;
};

struct ConSanMoiRecordReplayResult {
  uint32_t processed_access_count = 0;
  uint32_t processed_barrier_count = 0;
  uint32_t processed_atomic_count = 0;
  uint32_t processed_fence_count = 0;
  uint32_t dropped_access_count = 0;
  uint32_t dropped_barrier_count = 0;
  uint32_t unsupported_access_count = 0;
  uint32_t unsupported_atomic_count = 0;
  uint32_t unsupported_fence_count = 0;
  uint32_t emitted_diagnostic_count = 0;
  bool diagnostic_capacity_exhausted = false;
  bool metadata_full = false;
  bool conflict = false;
};

/// Result of replaying only the complete RR2 windows selected from a compact
/// trace. Invalid capture metadata fails closed before any shadow or diagnostic
/// output is modified.
struct ConSanMoiRecordReplayWindowResult {
  ConSanMoiRecordReplayResult replay;
  uint32_t selected_event_count = 0;
  bool invalid_capture = false;
};

struct ConSanMoiSampledPublishResult {
  uint32_t processed_access_count = 0;
  uint32_t eligible_window_count = 0;
  uint32_t selected_window_count = 0;
  uint32_t published_window_count = 0;
  uint32_t published_entry_count = 0;
  uint32_t malformed_window_count = 0;
  bool sampled_capacity_exhausted = false;
  bool window_capacity_exhausted = false;
  bool invalid_selection = false;
};

struct ConSanMoiSampledReplayResult {
  uint32_t processed_window_count = 0;
  uint32_t processed_entry_count = 0;
  uint32_t emitted_diagnostic_count = 0;
  uint32_t empty_entry_count = 0;
  uint32_t stale_generation_entry_count = 0;
  uint32_t incomplete_publication_entry_count = 0;
  uint32_t changed_during_read_entry_count = 0;
  uint32_t malformed_entry_count = 0;
  bool diagnostic_capacity_exhausted = false;
  bool invalid_causal_metadata = false;
  bool conflict = false;
};

enum class ConSanMoiSampledCausalPublicationState : uint32_t {
  Empty = 0,
  Publishing = 1,
  Ready = 2,
  Malformed = 3,
};

struct ConSanMoiSampledCausalKey {
  uint64_t generation = 0;
  uint64_t dispatch_id = 0;
  uint32_t workgroup_x = 0;
  uint32_t workgroup_y = 0;
  uint32_t workgroup_z = 0;
  uint32_t epoch = 0;
  uint32_t cluster_workgroup_id = 0;

  [[nodiscard]] constexpr bool operator==(const ConSanMoiSampledCausalKey &) const = default;
};

enum class ConSanMoiSampledClaimOutcome : uint32_t {
  Claimed,
  Existing,
  Busy,
  CapacityExhausted,
  Invalid,
};

struct ConSanMoiSampledClaimResult {
  ConSanMoiSampledClaimOutcome outcome = ConSanMoiSampledClaimOutcome::Invalid;
  uint32_t slot = 0;
  uint32_t collision_count = 0;
  uint32_t malformed_slot_count = 0;
};

/// Synchronization metadata is parallel to, and bounded by, the selected
/// Sampled causal windows. It is not an event stream. A Ready causal window
/// protects publication of its corresponding metadata slots.
namespace consan_moi_sampled_sync_abi {
inline constexpr uint32_t version = 1;
inline constexpr uint32_t version_shift = 0;
inline constexpr uint32_t version_bits = 8;
inline constexpr uint32_t kind_shift = 8;
inline constexpr uint32_t kind_bits = 4;
inline constexpr uint32_t role_shift = 12;
inline constexpr uint32_t role_bits = 4;
inline constexpr uint32_t scope_shift = 16;
inline constexpr uint32_t scope_bits = 4;
inline constexpr uint32_t outcome_shift = 20;
inline constexpr uint32_t outcome_bits = 4;
inline constexpr uint32_t reserved_mask = 0xff000000u;
} // namespace consan_moi_sampled_sync_abi

enum class ConSanMoiSampledSyncKind : uint8_t {
  None = 0,
  Atomic = 1,
  Barrier = 2,
};

/// Bit-composable atomic role. The Rmw bit distinguishes an ordinary atomic
/// load/store from an operation whose dynamic result must be retained.
enum class ConSanMoiSampledSyncRole : uint8_t {
  None = 0,
  Release = 1,
  Acquire = 2,
  AcquireRelease = 3,
  Rmw = 4,
  RmwRelease = 5,
  RmwAcquire = 6,
  RmwAcquireRelease = 7,
};

enum class ConSanMoiSampledSyncScope : uint8_t {
  None = 0,
  Wavefront = 1,
  Workgroup = 2,
  Agent = 3,
  System = 4,
  Cluster = 5,
};

enum class ConSanMoiSampledSyncOutcome : uint8_t {
  NotApplicable = 0,
  RmwNoReturn = 1,
  RmwReturnsOld = 2,
  CasSuccess = 3,
  CasFailure = 4,
};

/// Fixed ABI v1 payload. `address` is the first byte of an atomic range and
/// `byte_count` is its nonzero size. Barriers use an empty range and an exact
/// epoch_before -> epoch_after transition. Atomics retain a stable epoch by
/// requiring epoch_before == epoch_after.
/// Descriptor commit word used while a device publisher owns the slot. The
/// payload is not consumable until this sentinel changes atomically to a valid
/// typed descriptor.
inline constexpr uint32_t kConSanMoiSampledSyncPublishingDescriptor = 0xffffffffu;

/// ABI-v7 deferred acquire publication. The acquire executes before its
/// associated sampled access can publish a causal window, so it transactionally
/// retains the complete identity and atomic payload in the statically selected
/// slot. The later access claims the even version, validates every identity
/// field, finalizes ordinary sync metadata, then clears the slot.
struct ConSanMoiSampledPendingAcquireKey {
  uint64_t generation = 0;
  uint64_t dispatch_id = 0;
  uint32_t workgroup_x = 0;
  uint32_t workgroup_y = 0;
  uint32_t workgroup_z = 0;
  uint32_t owner_id = 0;
  uint32_t source_epoch = 0;
  uint32_t selected_slot = 0;
};

enum class ConSanMoiSampledPendingAcquireState : uint8_t {
  Empty,
  Publishing,
  Ready,
  ChangedDuringRead,
  Malformed,
  IdentityMismatch,
  FutureEpoch,
};

struct ConSanMoiSampledPendingAcquireView {
  uint32_t version_before = 0;
  ConSanMoiSampledPendingAcquireSlot slot;
  uint32_t version_after = 0;
};

[[nodiscard]] ConSanMoiSampledPendingAcquireState
classify_consan_moi_sampled_pending_acquire(const ConSanMoiSampledPendingAcquireView &view,
                                            const ConSanMoiSampledPendingAcquireKey &key,
                                            uint32_t window_epoch);

struct ConSanMoiSampledSyncMetadata {
  uint32_t version = consan_moi_sampled_sync_abi::version;
  uint64_t address = 0;
  uint32_t byte_count = 0;
  ConSanMoiSampledSyncKind kind = ConSanMoiSampledSyncKind::None;
  ConSanMoiSampledSyncRole role = ConSanMoiSampledSyncRole::None;
  ConSanMoiSampledSyncScope scope = ConSanMoiSampledSyncScope::None;
  ConSanMoiSampledSyncOutcome outcome = ConSanMoiSampledSyncOutcome::NotApplicable;
  uint32_t epoch_before = 0;
  uint32_t epoch_after = 0;

  [[nodiscard]] constexpr bool operator==(const ConSanMoiSampledSyncMetadata &) const = default;
};

enum class ConSanMoiSampledSyncClassification : uint8_t {
  Valid,
  Empty,
  Publishing,
  ChangedDuringRead,
  UnsupportedVersion,
  UnsupportedKind,
  UnsupportedRole,
  UnsupportedScope,
  UnsupportedOutcome,
  InvalidRange,
  RangeOverflow,
  EpochOverflow,
  Malformed,
  UnsupportedSequence,
};

struct ConSanMoiSampledSyncDecodeResult {
  ConSanMoiSampledSyncClassification classification = ConSanMoiSampledSyncClassification::Malformed;
  ConSanMoiSampledSyncMetadata metadata;
};

struct ConSanMoiSampledSyncSnapshotWords {
  uint32_t descriptor_before = 0;
  ConSanMoiSampledSyncMetadataPacked packed;
  uint32_t descriptor_after = 0;
};

/// Classifies a descriptor/payload/descriptor host observation. A committed
/// atomic record must also name the epoch of its paired Ready causal window.
[[nodiscard]] ConSanMoiSampledSyncDecodeResult
classify_consan_moi_sampled_sync_snapshot(ConSanMoiSampledSyncSnapshotWords words,
                                          uint32_t paired_window_epoch);

struct ConSanMoiSampledSyncEncodeResult {
  ConSanMoiSampledSyncClassification classification = ConSanMoiSampledSyncClassification::Malformed;
  ConSanMoiSampledSyncMetadataPacked packed;
};

enum class ConSanMoiSampledSyncPublishOutcome : uint8_t {
  Published,
  Existing,
  Collision,
  CapacityExhausted,
  Rejected,
  MalformedSlot,
};

struct ConSanMoiSampledSyncPublishResult {
  ConSanMoiSampledSyncPublishOutcome outcome = ConSanMoiSampledSyncPublishOutcome::Rejected;
  ConSanMoiSampledSyncClassification classification = ConSanMoiSampledSyncClassification::Malformed;
  uint32_t slot = 0;
};

struct ConSanMoiAtomicReleaseRecord {
  bool valid = false;
  uint64_t generation = 0;
  uint64_t atomic_address = 0;
  uint32_t producer_owner_id = 0;
  uint32_t producer_epoch = 0;
  uint32_t release_instruction_offset = 0;
};

struct ConSanMoiAcquiredEpochToken {
  bool valid = false;
  uint64_t generation = 0;
  uint32_t consumer_owner_id = 0;
  uint32_t producer_owner_id = 0;
  uint32_t producer_epoch_plus_one = 0;
  uint32_t acquire_instruction_offset = 0;
};

struct ConSanMoiAtomicSyncResult {
  bool metadata_full = false;
  uint32_t updated_record_count = 0;
};

using ConSanMoiRecordReplayAtomicEvent = ConSanMoiAtomicRecord;
using ConSanMoiRecordReplayFenceEvent = ConSanMoiFenceRecord;

/// Converts the existing dynamic access/barrier/atomic streams into the RR1
/// bounded compact schema. Inputs are globally ordered by event_index; stable
/// input order breaks ties. Output is prefix-complete: capacity exhaustion
/// commits no part of the first event that does not fit and sets Overflow.
