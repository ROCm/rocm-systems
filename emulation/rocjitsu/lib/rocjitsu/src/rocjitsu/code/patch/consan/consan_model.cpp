// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_model.cpp
/// @brief ConSan report metadata encoding, decoding, and qualification.

#include "rocjitsu/code/patch/consan/consan_report.h"

#include <limits>

namespace rocjitsu::consan {

SyncClassification classify_sync_metadata(const SyncMetadata &metadata) {
  using Classification = SyncClassification;
  if (metadata.version != sync_abi::version)
    return Classification::UnsupportedVersion;

  const uint32_t kind = static_cast<uint32_t>(metadata.kind);
  const uint32_t role = static_cast<uint32_t>(metadata.role);
  const uint32_t scope = static_cast<uint32_t>(metadata.scope);
  const uint32_t outcome = static_cast<uint32_t>(metadata.outcome);
  if (kind != static_cast<uint32_t>(SyncMetadataKind::Atomic) &&
      kind != static_cast<uint32_t>(SyncMetadataKind::Barrier))
    return Classification::UnsupportedKind;
  if (role > static_cast<uint32_t>(SyncRole::RmwAcquireRelease) || role == 0)
    return Classification::UnsupportedRole;
  if (scope < static_cast<uint32_t>(SyncScope::Wavefront) ||
      scope > static_cast<uint32_t>(SyncScope::Cluster))
    return Classification::UnsupportedScope;
  if (outcome > static_cast<uint32_t>(SyncOutcome::CasFailure))
    return Classification::UnsupportedOutcome;

  if (metadata.kind == SyncMetadataKind::Atomic) {
    if (metadata.address == 0 || metadata.byte_count == 0)
      return Classification::InvalidRange;
    if (metadata.address >
        std::numeric_limits<uint64_t>::max() - (static_cast<uint64_t>(metadata.byte_count) - 1u))
      return Classification::RangeOverflow;
    if (metadata.epoch_before != metadata.epoch_after)
      return Classification::UnsupportedSequence;

    constexpr uint32_t rmw_bit = static_cast<uint32_t>(SyncRole::Rmw);
    const bool is_rmw = (role & rmw_bit) != 0;
    if (is_rmw) {
      if (metadata.outcome == SyncOutcome::NotApplicable)
        return Classification::UnsupportedSequence;
    } else if (metadata.outcome != SyncOutcome::NotApplicable ||
               metadata.role == SyncRole::AcquireRelease) {
      return Classification::UnsupportedSequence;
    }
    return Classification::Valid;
  }

  if (metadata.address != 0 || metadata.byte_count != 0)
    return Classification::InvalidRange;
  if (metadata.role != SyncRole::AcquireRelease ||
      (metadata.scope != SyncScope::Workgroup && metadata.scope != SyncScope::Cluster) ||
      metadata.outcome != SyncOutcome::NotApplicable)
    return Classification::UnsupportedSequence;
  if (metadata.epoch_before == std::numeric_limits<uint32_t>::max())
    return Classification::EpochOverflow;
  if (metadata.epoch_after != metadata.epoch_before + 1u)
    return Classification::UnsupportedSequence;
  return Classification::Valid;
}

SyncEncodeResult encode_sync_metadata(const SyncMetadata &metadata) {
  const SyncClassification classification = classify_sync_metadata(metadata);
  if (classification != SyncClassification::Valid)
    return {classification, {}};

  const uint32_t descriptor = (metadata.version << sync_abi::version_shift) |
                              (static_cast<uint32_t>(metadata.kind) << sync_abi::kind_shift) |
                              (static_cast<uint32_t>(metadata.role) << sync_abi::role_shift) |
                              (static_cast<uint32_t>(metadata.scope) << sync_abi::scope_shift) |
                              (static_cast<uint32_t>(metadata.outcome) << sync_abi::outcome_shift);
  return {classification,
          {metadata.address, metadata.byte_count, descriptor, metadata.epoch_before,
           metadata.epoch_after}};
}

SyncDecodeResult decode_sync_metadata(const SyncMetadataPacked &packed) {
  using Classification = SyncClassification;
  if (packed == SyncMetadataPacked{})
    return {.metadata = {}, .classification = Classification::Empty};
  if (packed.descriptor == kSyncPublishingDescriptor)
    return {.metadata = {}, .classification = Classification::Publishing};
  if ((packed.descriptor & sync_abi::reserved_mask) != 0)
    return {.metadata = {}, .classification = Classification::Malformed};

  const auto extract = [&](uint32_t shift, uint32_t bits) {
    return (packed.descriptor >> shift) & ((uint32_t{1} << bits) - 1u);
  };
  SyncMetadata metadata{
      packed.address,
      extract(sync_abi::version_shift, sync_abi::version_bits),
      packed.byte_count,
      static_cast<SyncMetadataKind>(extract(sync_abi::kind_shift, sync_abi::kind_bits)),
      static_cast<SyncRole>(extract(sync_abi::role_shift, sync_abi::role_bits)),
      static_cast<SyncScope>(extract(sync_abi::scope_shift, sync_abi::scope_bits)),
      static_cast<SyncOutcome>(extract(sync_abi::outcome_shift, sync_abi::outcome_bits)),
      packed.epoch_before,
      packed.epoch_after,
  };
  const Classification classification = classify_sync_metadata(metadata);
  return {.metadata = metadata, .classification = classification};
}

SyncDecodeResult classify_sync_snapshot(SyncSnapshotWords words, uint32_t paired_window_epoch) {
  using Classification = SyncClassification;
  if (words.descriptor_before != words.descriptor_after)
    return {.metadata = {}, .classification = Classification::ChangedDuringRead};
  // The descriptor is the commit word. Ignore the copy captured with the
  // payload so callers cannot accidentally validate anything but the stable
  // descriptor observations which bracketed it.
  words.packed.descriptor = words.descriptor_after;
  SyncDecodeResult decoded = decode_sync_metadata(words.packed);
  if (decoded.classification == Classification::Valid) {
    if (decoded.metadata.kind == SyncMetadataKind::Atomic &&
        (decoded.metadata.epoch_before != paired_window_epoch ||
         decoded.metadata.epoch_after != paired_window_epoch))
      return {.metadata = decoded.metadata, .classification = Classification::UnsupportedSequence};
    if (decoded.metadata.kind == SyncMetadataKind::Barrier &&
        decoded.metadata.epoch_before != paired_window_epoch)
      return {.metadata = decoded.metadata, .classification = Classification::UnsupportedSequence};
  }
  return decoded;
}

PendingAcquireState classify_pending_acquire(const PendingAcquireView &view,
                                             const PendingAcquireKey &key, uint32_t window_epoch) {
  using State = PendingAcquireState;
  if (view.version_before != view.version_after)
    return State::ChangedDuringRead;
  if (view.version_after == 0)
    return view.slot == PendingAcquireSlot{} ? State::Empty : State::Malformed;
  if ((view.version_after & 1u) != 0)
    return State::Publishing;
  if (view.slot.version != view.version_after)
    return State::Malformed;
  if (view.slot.selected_slot != key.selected_slot || view.slot.generation != key.generation ||
      view.slot.dispatch_id != key.dispatch_id || view.slot.workgroup_x != key.workgroup_x ||
      view.slot.workgroup_y != key.workgroup_y || view.slot.workgroup_z != key.workgroup_z ||
      view.slot.owner_id != key.owner_id || view.slot.source_epoch != key.source_epoch)
    return State::IdentityMismatch;
  if (view.slot.source_epoch > window_epoch)
    return State::FutureEpoch;
  const SyncDecodeResult decoded = decode_sync_metadata(view.slot.metadata);
  const uint32_t role = static_cast<uint32_t>(decoded.metadata.role);
  if (decoded.classification != SyncClassification::Valid ||
      decoded.metadata.kind != SyncMetadataKind::Atomic ||
      (role & static_cast<uint32_t>(SyncRole::Acquire)) == 0)
    return State::Malformed;
  return State::Ready;
}

bool qualifies_barrier_sequence(const SyncSequence &sequence, bool owner_proven) {
  const bool static_id = sequence.barrier_operand_source == BarrierSite::OperandSource::Immediate ||
                         sequence.barrier_operand_source == BarrierSite::OperandSource::Literal32;
  // A physical barrier sequence can be reached from several kernel
  // descriptors through shared helper code. Ownership is still proven when
  // every reachable descriptor is known; lowering validates that all owners
  // have compatible ABI inputs and a preceding selected causal window.
  return sequence.kind == SyncKind::Barrier && sequence.operation == SyncOperation::BarrierFull &&
         sequence.memory_role == SyncMemoryRole::AcquireRelease &&
         sync_confidence_meets(sequence.confidence, SemanticConfidence::Conservative) &&
         sync_confidence_meets(sequence.memory_role_confidence, SemanticConfidence::Conservative) &&
         sequence.basic_block_index && !sequence.inside_scalar_clause && owner_proven &&
         static_id && sequence.barrier_id &&
         (sequence.barrier_scope == BarrierSite::Scope::Workgroup ||
          sequence.barrier_scope == BarrierSite::Scope::Cluster) &&
         (sequence.member_event_ids.size() == 1u || sequence.member_event_ids.size() == 2u) &&
         sequence.begin_text_offset < sequence.end_text_offset;
}

bool atomic_attachment_matches(const CausalWindow &window, uint64_t packed_watchpoint,
                               uint32_t slot, const AtomicAttachmentKey &key) {
  if (window.publication_state != static_cast<uint32_t>(CausalPublicationState::Ready) ||
      window.generation != key.generation || window.dispatch_id != key.dispatch_id ||
      window.workgroup_x != key.workgroup_x || window.workgroup_y != key.workgroup_y ||
      window.workgroup_z != key.workgroup_z || window.epoch != key.epoch ||
      window.cluster_workgroup_id != key.cluster_workgroup_id || window.first_entry != slot ||
      window.entry_count != 1u)
    return false;
  const WatchpointEntry watchpoint = decode_watchpoint_entry(packed_watchpoint);
  return watchpoint.valid && !watchpoint.consumed &&
         (watchpoint.kind == ShadowAccessKind::Read ||
          watchpoint.kind == ShadowAccessKind::Write) &&
         watchpoint.owner_id == key.owner_id && watchpoint.epoch == key.epoch &&
         watchpoint.generation ==
             (static_cast<uint32_t>(key.generation) & watchpoint::max_generation);
}

} // namespace rocjitsu::consan
