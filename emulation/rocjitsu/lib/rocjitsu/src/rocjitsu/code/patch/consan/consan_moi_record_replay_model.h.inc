// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

[[nodiscard]] constexpr uint32_t consan_moi_acquired_epoch_token_value(uint32_t epoch) {
  return epoch == std::numeric_limits<uint32_t>::max() ? epoch : epoch + 1u;
}

[[nodiscard]] bool
consan_moi_acquired_epoch_orders(std::span<const ConSanMoiAcquiredEpochToken> tokens,
                                 const ConSanMoiExactShadowEntry &current,
                                 const ConSanMoiExactShadowEntry &prior);

[[nodiscard]] ConSanMoiAtomicSyncResult
consan_moi_record_replay_atomic_release(std::span<ConSanMoiAtomicReleaseRecord> release_records,
                                        uint64_t generation, uint64_t atomic_address,
                                        uint32_t producer_owner_id, uint32_t producer_epoch,
                                        uint32_t release_instruction_offset);

/// Publishes the releasing owner's sparse causal clock at atomic_address. The
/// own epoch and every imported producer epoch commit together or not at all.
[[nodiscard]] ConSanMoiAtomicSyncResult consan_moi_record_replay_atomic_release(
    std::span<ConSanMoiAtomicReleaseRecord> release_records,
    std::span<const ConSanMoiAcquiredEpochToken> acquired_epoch_tokens, uint64_t generation,
    uint64_t atomic_address, uint32_t producer_owner_id, uint32_t producer_epoch,
    uint32_t release_instruction_offset);

[[nodiscard]] ConSanMoiAtomicSyncResult consan_moi_record_replay_atomic_acquire(
    std::span<const ConSanMoiAtomicReleaseRecord> release_records,
    std::span<ConSanMoiAcquiredEpochToken> acquired_epoch_tokens, uint64_t generation,
    uint64_t atomic_address, uint32_t consumer_owner_id, uint32_t acquire_instruction_offset);

/// Fixed per-site Record/Replay tables publish their static capacity in the
/// header. Slots which were never executed remain entirely zero, while a
/// partially published zero-kind slot must still reach replay and fail closed.
[[nodiscard]] constexpr bool
consan_moi_access_record_is_unpublished(const ConSanMoiAccessRecord &record) {
  return record.claim_token == 0 && record.generation == 0 && record.workgroup_x == 0 &&
         record.workgroup_y == 0 && record.workgroup_z == 0 && record.wave_id == 0 &&
         record.lane_mask == 0 && record.instruction_offset == 0 &&
         record.access_kind == static_cast<uint32_t>(ConSanMoiShadowAccessKind::Empty) &&
         record.lds_byte_offset == 0 && record.lds_byte_count == 0 && record.start_cell == 0 &&
         record.cell_count == 0 && record.epoch == 0 && record.event_index == 0 &&
         record.site_token == 0 && record.flags == 0;
}

[[nodiscard]] ConSanMoiRecordReplayResult consan_moi_record_replay_access_records(
    ConSanMoiReportHeader &header, std::span<const ConSanMoiAccessRecord> access_records,
    std::span<const ConSanMoiBarrierRecord> barrier_records,
    std::span<const ConSanMoiRecordReplayAtomicEvent> atomic_events,
    std::span<const ConSanMoiRecordReplayFenceEvent> fence_events,
    std::span<ConSanMoiDiagnosticRecord> diagnostic_records,
    std::span<uint64_t> exact_shadow_entries);
