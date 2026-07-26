// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_abi.h
/// @brief Host/HIP-safe POD definitions for the ConSan MOI report ABI.

#pragma once

#include <cstdint>

namespace rocjitsu {

enum class ConSanMoiShadowAccessKind : uint32_t {
  Empty = 0,
  Read = 1,
  Write = 2,
  ReadWrite = 3,
  Atomic = 4,
};

enum class ConSanMoiAtomicEventKind : uint32_t {
  Release = 1,
  Acquire = 2,
  AcquireRelease = 3,
};

enum class ConSanMoiAtomicOperation : uint8_t {
  Rmw = 1,
  CompareExchange = 2,
};

enum class ConSanMoiAtomicOutcome : uint16_t {
  NotApplicable = 0,
  Success = 1,
  Failure = 2,
  Unavailable = 3,
};

enum class ConSanMoiFenceEventKind : uint8_t {
  Release = 1,
  Acquire = 2,
  AcquireRelease = 3,
};

inline constexpr uint32_t kConSanMoiReportMagic = 0x494f4d43u; // "CMOI" little-endian.
inline constexpr uint32_t kConSanMoiReportAbiVersion = 12;
inline constexpr uint32_t kConSanMoiReportFlagRecordReplayBankSaturated = 1u << 0u;
inline constexpr uint32_t kConSanMoiReportFlagRecordReplayDispatchBankSaturated = 1u << 1u;
inline constexpr uint32_t kConSanMoiReportFlagRecordReplayOwnerBankSaturated = 1u << 2u;
inline constexpr uint32_t kConSanMoiReportFlagRecordReplayPublicationIncomplete = 1u << 3u;
inline constexpr uint64_t kConSanMoiRecordReplayClaimTokenXorMask = 0x7f4a7c159e3779b9ull;

[[nodiscard]] constexpr uint64_t consan_moi_record_replay_claim_token(uint64_t dispatch_id) {
  return dispatch_id ^ kConSanMoiRecordReplayClaimTokenXorMask;
}

struct alignas(8) ConSanMoiReportHeader {
  uint32_t magic = kConSanMoiReportMagic;
  uint32_t abi_version = kConSanMoiReportAbiVersion;
  uint32_t header_size = 0;
  uint32_t flags = 0;
  uint64_t generation = 0;
  uint64_t dispatch_id = 0;
  uint32_t engine = 0;
  uint32_t layout_flags = 0;
  uint32_t access_record_capacity = 0;
  uint32_t access_record_count = 0;
  uint32_t barrier_record_capacity = 0;
  uint32_t barrier_record_count = 0;
  uint32_t atomic_record_capacity = 0;
  uint32_t atomic_record_count = 0;
  uint32_t fence_record_capacity = 0;
  uint32_t fence_record_count = 0;
  uint32_t diagnostic_capacity = 0;
  uint32_t diagnostic_count = 0;
  uint32_t exact_shadow_entry_capacity = 0;
  uint32_t inline_atomic_release_capacity = 0;
  uint32_t inline_acquired_epoch_token_capacity = 0;
  uint32_t inline_causal_snapshot_capacity = 0;
  uint32_t event_counter = 0;
  uint32_t inline_undercoverage_count = 0;
  uint32_t inline_overflow_count = 0;
  uint32_t inline_unsupported_count = 0;
  uint32_t inline_malformed_count = 0;
  uint32_t sampled_watchpoint_capacity = 0;
  uint32_t sampled_causal_window_capacity = 0;
  uint32_t sampled_causal_window_count = 0;
  uint32_t sampled_malformed_window_count = 0;
  uint32_t sampled_dropped_window_count = 0;
  uint32_t sampled_saturated_window_count = 0;
  uint32_t sampled_sync_metadata_capacity = 0;
  uint32_t sampled_sync_metadata_count = 0;
  uint32_t sampled_unsupported_sync_count = 0;
  uint32_t sampled_malformed_sync_count = 0;
  uint32_t sampled_pending_acquire_capacity = 0;
  uint32_t sampled_pending_acquire_count = 0;
  uint32_t sampled_pending_acquire_contention_count = 0;
  uint32_t sampled_pending_acquire_collision_count = 0;
  uint32_t sampled_pending_acquire_malformed_count = 0;
  /// Automatic Record/Replay uses a report-wide open-addressed table to map
  /// the complete hardware dispatch identity to one stable access-record bank.
  /// Direct caller-provided buffers retain zero capacity and their historical
  /// single-bank layout.
  uint32_t record_replay_dispatch_token_capacity = 0;
  uint32_t record_replay_dispatch_token_count = 0;
};

struct alignas(8) ConSanMoiAccessRecord {
  /// Internal publication identity. Record/Replay atomically claims a bounded
  /// identity-table slot with a compact fingerprint derived from the hardware
  /// dispatch ID, static site, workgroup, and wave. Zero remains the unpublished
  /// sentinel; any identity whose fingerprint maps to zero fails closed through
  /// the report saturation flag. access_kind is the atomic payload commit;
  /// occupied-slot reuse additionally requires exact qualification of every
  /// identity field after that commit. Host replay deliberately ignores the
  /// claim field.
  uint64_t claim_token = 0;
  uint64_t generation = 0;
  uint32_t workgroup_x = 0;
  uint32_t workgroup_y = 0;
  uint32_t workgroup_z = 0;
  uint32_t wave_id = 0;
  uint64_t lane_mask = 0;
  uint32_t instruction_offset = 0;
  uint32_t access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Empty);
  uint32_t lds_byte_offset = 0;
  uint32_t lds_byte_count = 0;
  uint32_t start_cell = 0;
  uint32_t cell_count = 0;
  uint32_t epoch = 0;
  uint32_t event_index = 0;
  /// Host-planned dense identity for one static instruction/range pair.
  /// Automatic Record/Replay includes it in the report-wide access-table key;
  /// direct layouts leave it zero.
  uint32_t site_token = 0;
  uint32_t reserved = 0;
};

struct alignas(8) ConSanMoiBarrierRecord {
  uint64_t generation = 0;
  uint32_t workgroup_x = 0;
  uint32_t workgroup_y = 0;
  uint32_t workgroup_z = 0;
  uint32_t wave_id = 0;
  uint64_t lane_mask = 0;
  uint32_t instruction_offset = 0;
  uint32_t event_index = 0;
};

struct alignas(8) ConSanMoiAtomicRecord {
  uint64_t generation = 0;
  uint32_t workgroup_x = 0;
  uint32_t workgroup_y = 0;
  uint32_t workgroup_z = 0;
  uint32_t owner_id = 0;
  uint64_t atomic_address = 0;
  uint32_t instruction_offset = 0;
  uint32_t event_index = 0;
  uint32_t epoch = 0;
  ConSanMoiAtomicEventKind kind = ConSanMoiAtomicEventKind::Release;
  uint32_t scope = 0;
  uint32_t semantics = 0;
  ConSanMoiAtomicOperation operation = ConSanMoiAtomicOperation::Rmw;
  ConSanMoiAtomicOutcome outcome = ConSanMoiAtomicOutcome::NotApplicable;
  uint64_t lane_mask = 0;
  uint64_t success_lane_mask = 0;
};

struct alignas(8) ConSanMoiFenceRecord {
  uint64_t generation = 0;
  uint32_t workgroup_x = 0;
  uint32_t workgroup_y = 0;
  uint32_t workgroup_z = 0;
  uint32_t owner_id = 0;
  uint32_t instruction_offset = 0;
  uint32_t event_index = 0;
  uint32_t epoch = 0;
  ConSanMoiFenceEventKind kind = ConSanMoiFenceEventKind::Release;
  uint32_t scope = 0;
  uint32_t semantics = 0;
  uint64_t communication_token = 0;
};

struct alignas(8) ConSanMoiSampledCausalWindow {
  uint64_t generation = 0;
  uint64_t dispatch_id = 0;
  uint32_t workgroup_x = 0;
  uint32_t workgroup_y = 0;
  uint32_t workgroup_z = 0;
  uint32_t epoch = 0;
  uint32_t first_entry = 0;
  uint32_t entry_count = 0;
  uint32_t publication_state = 0;
  // Zero for ordinary dispatches. On gfx1250 clustered kernels this carries
  // the launch-provided workgroup-within-cluster identity.
  uint32_t cluster_workgroup_id = 0;
};

struct alignas(8) ConSanMoiSampledSyncMetadataPacked {
  uint64_t address = 0;
  uint32_t byte_count = 0;
  uint32_t descriptor = 0;
  uint32_t epoch_before = 0;
  uint32_t epoch_after = 0;

  [[nodiscard]] constexpr bool
  operator==(const ConSanMoiSampledSyncMetadataPacked &) const = default;
};

struct alignas(8) ConSanMoiSampledPendingAcquireSlot {
  uint32_t version = 0;
  uint32_t selected_slot = 0;
  uint64_t generation = 0;
  uint64_t dispatch_id = 0;
  uint32_t workgroup_x = 0;
  uint32_t workgroup_y = 0;
  uint32_t workgroup_z = 0;
  uint32_t owner_id = 0;
  uint32_t source_epoch = 0;
  uint32_t reserved = 0;
  ConSanMoiSampledSyncMetadataPacked metadata;

  [[nodiscard]] constexpr bool
  operator==(const ConSanMoiSampledPendingAcquireSlot &) const = default;
};

static_assert(sizeof(ConSanMoiReportHeader) == 184);
static_assert(sizeof(ConSanMoiAccessRecord) == 80);
static_assert(sizeof(ConSanMoiBarrierRecord) == 40);
static_assert(sizeof(ConSanMoiAtomicRecord) == 80);
static_assert(sizeof(ConSanMoiFenceRecord) == 56);
static_assert(sizeof(ConSanMoiSampledCausalWindow) == 48);
static_assert(sizeof(ConSanMoiSampledSyncMetadataPacked) == 24);
static_assert(sizeof(ConSanMoiSampledPendingAcquireSlot) == 72);

} // namespace rocjitsu
