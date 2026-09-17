enum class CausalPublicationState : uint32_t {
  Empty = 0,
  Publishing = 1,
  Ready = 2,
  Malformed = 3,
};

/// Synchronization metadata is parallel to, and bounded by, the selected
/// ConSan causal windows. It is not an event stream. A Ready causal window
/// protects publication of its corresponding metadata slots.
namespace sync_abi {
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
} // namespace sync_abi

enum class SyncMetadataKind : uint8_t {
  None = 0,
  Atomic = 1,
  Barrier = 2,
};

/// Bit-composable atomic role. The Rmw bit distinguishes an ordinary atomic
/// load/store from an operation whose dynamic result must be retained.
enum class SyncRole : uint8_t {
  None = 0,
  Release = 1,
  Acquire = 2,
  AcquireRelease = 3,
  Rmw = 4,
  RmwRelease = 5,
  RmwAcquire = 6,
  RmwAcquireRelease = 7,
};

enum class SyncScope : uint8_t {
  None = 0,
  Wavefront = 1,
  Workgroup = 2,
  Agent = 3,
  System = 4,
  Cluster = 5,
};

/// Translate the common synchronization scope into ConSan's stable report
/// ABI. This is the report layout mapping; emitters do not reinterpret
/// target-native scope codes.
[[nodiscard]] constexpr std::optional<SyncScope> sync_scope(MemoryScope scope) {
  switch (scope) {
  case MemoryScope::Wavefront:
    return SyncScope::Wavefront;
  case MemoryScope::Workgroup:
    return SyncScope::Workgroup;
  case MemoryScope::Agent:
    return SyncScope::Agent;
  case MemoryScope::System:
    return SyncScope::System;
  }
  return std::nullopt;
}

enum class SyncOutcome : uint8_t {
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
inline constexpr uint32_t kSyncPublishingDescriptor = 0xffffffffu;

// One HIP workgroup contains at most 1024 workitems, hence at most 32 wave32
// owners. Deferred acquires are live per wave until their associated sampled
// access publishes; banking only by workgroup lets ordinary last-arriver RMWs
// overwrite or contend with one another.
inline constexpr uint32_t kPendingAcquireOwnerBankCount = 32u;

[[nodiscard]] constexpr uint32_t pending_acquire_owner_bank_count(uint32_t pending_capacity,
                                                                  uint32_t causal_window_capacity) {
  if (causal_window_capacity == 0u || pending_capacity % causal_window_capacity != 0u)
    return 0u;
  const uint32_t bank_count = pending_capacity / causal_window_capacity;
  return bank_count != 0u && bank_count <= kPendingAcquireOwnerBankCount &&
                 std::has_single_bit(bank_count)
             ? bank_count
             : 0u;
}

/// ABI-v7 deferred acquire publication. The acquire executes before its
/// associated sampled access can publish a causal window, so it transactionally
/// retains the complete identity and atomic payload in the statically selected
/// slot. The later access claims the even version, validates every identity
/// field, finalizes ordinary sync metadata, then clears the slot.
struct PendingAcquireKey {
  uint64_t generation = 0;
  uint64_t dispatch_id = 0;
  uint32_t workgroup_x = 0;
  uint32_t workgroup_y = 0;
  uint32_t workgroup_z = 0;
  uint32_t owner_id = 0;
  uint32_t source_epoch = 0;
  uint32_t selected_slot = 0;
};

enum class PendingAcquireState : uint8_t {
  Empty,
  Publishing,
  Ready,
  ChangedDuringRead,
  Malformed,
  IdentityMismatch,
  FutureEpoch,
};

struct PendingAcquireView {
  uint32_t version_before = 0;
  PendingAcquireSlot slot;
  uint32_t version_after = 0;
};

[[nodiscard]] PendingAcquireState classify_pending_acquire(const PendingAcquireView &view,
                                                           const PendingAcquireKey &key,
                                                           uint32_t window_epoch);

struct SyncMetadata {
  uint64_t address = 0;
  uint32_t version = sync_abi::version;
  uint32_t byte_count = 0;
  SyncMetadataKind kind = SyncMetadataKind::None;
  SyncRole role = SyncRole::None;
  SyncScope scope = SyncScope::None;
  SyncOutcome outcome = SyncOutcome::NotApplicable;
  uint32_t epoch_before = 0;
  uint32_t epoch_after = 0;

  [[nodiscard]] constexpr bool operator==(const SyncMetadata &) const = default;
};

enum class SyncClassification : uint8_t {
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

struct SyncDecodeResult {
  // Reuse host metadata tail padding so decoded evidence stays compact without
  // over-aligning records. The packed device representation is separate.
  [[no_unique_address]] SyncMetadata metadata;
  SyncClassification classification = SyncClassification::Malformed;
};

struct SyncSnapshotWords {
  uint32_t descriptor_before = 0;
  SyncMetadataPacked packed;
  uint32_t descriptor_after = 0;
};

/// Classifies a descriptor/payload/descriptor host observation. A committed
/// atomic record must also name the epoch of its paired Ready causal window.
[[nodiscard]] SyncDecodeResult classify_sync_snapshot(SyncSnapshotWords words,
                                                      uint32_t paired_window_epoch);

struct SyncEncodeResult {
  SyncClassification classification = SyncClassification::Malformed;
  SyncMetadataPacked packed;
};
