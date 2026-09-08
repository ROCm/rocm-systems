enum class ConSanMoiSampledCausalPublicationState : uint32_t {
  Empty = 0,
  Publishing = 1,
  Ready = 2,
  Malformed = 3,
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

/// Translate the common synchronization scope into Sampled's stable report
/// ABI. This is the sole mode-owned mapping; emitters do not reinterpret
/// target-native scope codes.
[[nodiscard]] constexpr std::optional<ConSanMoiSampledSyncScope>
consan_moi_sampled_sync_scope(ConSanMemoryScope scope) {
  switch (scope) {
  case ConSanMemoryScope::Wavefront:
    return ConSanMoiSampledSyncScope::Wavefront;
  case ConSanMemoryScope::Workgroup:
    return ConSanMoiSampledSyncScope::Workgroup;
  case ConSanMemoryScope::Agent:
    return ConSanMoiSampledSyncScope::Agent;
  case ConSanMemoryScope::System:
    return ConSanMoiSampledSyncScope::System;
  }
  return std::nullopt;
}

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

// One HIP workgroup contains at most 1024 workitems, hence at most 32 wave32
// owners. Deferred acquires are live per wave until their associated sampled
// access publishes; banking only by workgroup lets ordinary last-arriver RMWs
// overwrite or contend with one another.
inline constexpr uint32_t kConSanMoiSampledPendingAcquireOwnerBankCount = 32u;

[[nodiscard]] constexpr uint32_t
consan_moi_sampled_pending_acquire_owner_bank_count(uint32_t pending_capacity,
                                                    uint32_t causal_window_capacity) {
  if (causal_window_capacity == 0u || pending_capacity % causal_window_capacity != 0u)
    return 0u;
  const uint32_t bank_count = pending_capacity / causal_window_capacity;
  return bank_count != 0u && bank_count <= kConSanMoiSampledPendingAcquireOwnerBankCount &&
                 std::has_single_bit(bank_count)
             ? bank_count
             : 0u;
}

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
