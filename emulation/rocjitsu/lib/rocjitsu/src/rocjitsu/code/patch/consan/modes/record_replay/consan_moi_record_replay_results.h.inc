struct ConSanMoiRecordReplayResult {
  uint32_t published_access_count = 0;
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
  uint32_t maximum_atomic_release_metadata_count = 0;
  uint32_t maximum_acquired_epoch_metadata_count = 0;
  bool diagnostic_capacity_exhausted = false;
  bool metadata_full = false;
  bool conflict = false;
};

struct ConSanMoiAtomicReleaseRecord {
  /// One sparse causal-clock component published at atomic_address. The
  /// producer can be the releasing owner or an owner ordered before it.
  bool valid = false;
  uint64_t generation = 0;
  uint64_t atomic_address = 0;
  uint32_t producer_owner_id = 0;
  uint32_t producer_epoch = 0;
  uint32_t release_instruction_offset = 0;
};

struct ConSanMoiAcquiredEpochToken {
  /// One sparse causal-clock component imported by consumer_owner_id.
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
