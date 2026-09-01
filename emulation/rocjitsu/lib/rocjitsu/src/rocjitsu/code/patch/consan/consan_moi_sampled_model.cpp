// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_sampled_model.cpp
/// @brief Sampled report metadata encoding, decoding, and qualification.

#include "rocjitsu/code/patch/consan/consan_moi.h"

#include <limits>

namespace rocjitsu {

ConSanMoiSampledSyncClassification
classify_consan_moi_sampled_sync_metadata(const ConSanMoiSampledSyncMetadata &metadata) {
  using Classification = ConSanMoiSampledSyncClassification;
  if (metadata.version != consan_moi_sampled_sync_abi::version)
    return Classification::UnsupportedVersion;

  const uint32_t kind = static_cast<uint32_t>(metadata.kind);
  const uint32_t role = static_cast<uint32_t>(metadata.role);
  const uint32_t scope = static_cast<uint32_t>(metadata.scope);
  const uint32_t outcome = static_cast<uint32_t>(metadata.outcome);
  if (kind != static_cast<uint32_t>(ConSanMoiSampledSyncKind::Atomic) &&
      kind != static_cast<uint32_t>(ConSanMoiSampledSyncKind::Barrier))
    return Classification::UnsupportedKind;
  if (role > static_cast<uint32_t>(ConSanMoiSampledSyncRole::RmwAcquireRelease) || role == 0)
    return Classification::UnsupportedRole;
  if (scope < static_cast<uint32_t>(ConSanMoiSampledSyncScope::Wavefront) ||
      scope > static_cast<uint32_t>(ConSanMoiSampledSyncScope::Cluster))
    return Classification::UnsupportedScope;
  if (outcome > static_cast<uint32_t>(ConSanMoiSampledSyncOutcome::CasFailure))
    return Classification::UnsupportedOutcome;

  if (metadata.kind == ConSanMoiSampledSyncKind::Atomic) {
    if (metadata.address == 0 || metadata.byte_count == 0)
      return Classification::InvalidRange;
    if (metadata.address >
        std::numeric_limits<uint64_t>::max() - (static_cast<uint64_t>(metadata.byte_count) - 1u))
      return Classification::RangeOverflow;
    if (metadata.epoch_before != metadata.epoch_after)
      return Classification::UnsupportedSequence;

    constexpr uint32_t rmw_bit = static_cast<uint32_t>(ConSanMoiSampledSyncRole::Rmw);
    const bool is_rmw = (role & rmw_bit) != 0;
    if (is_rmw) {
      if (metadata.outcome == ConSanMoiSampledSyncOutcome::NotApplicable)
        return Classification::UnsupportedSequence;
    } else if (metadata.outcome != ConSanMoiSampledSyncOutcome::NotApplicable ||
               metadata.role == ConSanMoiSampledSyncRole::AcquireRelease) {
      return Classification::UnsupportedSequence;
    }
    return Classification::Valid;
  }

  if (metadata.address != 0 || metadata.byte_count != 0)
    return Classification::InvalidRange;
  if (metadata.role != ConSanMoiSampledSyncRole::AcquireRelease ||
      (metadata.scope != ConSanMoiSampledSyncScope::Workgroup &&
       metadata.scope != ConSanMoiSampledSyncScope::Cluster) ||
      metadata.outcome != ConSanMoiSampledSyncOutcome::NotApplicable)
    return Classification::UnsupportedSequence;
  if (metadata.epoch_before == std::numeric_limits<uint32_t>::max())
    return Classification::EpochOverflow;
  if (metadata.epoch_after != metadata.epoch_before + 1u)
    return Classification::UnsupportedSequence;
  return Classification::Valid;
}

ConSanMoiSampledSyncEncodeResult
encode_consan_moi_sampled_sync_metadata(const ConSanMoiSampledSyncMetadata &metadata) {
  const ConSanMoiSampledSyncClassification classification =
      classify_consan_moi_sampled_sync_metadata(metadata);
  if (classification != ConSanMoiSampledSyncClassification::Valid)
    return {classification, {}};

  using namespace consan_moi_sampled_sync_abi;
  const uint32_t descriptor = (metadata.version << version_shift) |
                              (static_cast<uint32_t>(metadata.kind) << kind_shift) |
                              (static_cast<uint32_t>(metadata.role) << role_shift) |
                              (static_cast<uint32_t>(metadata.scope) << scope_shift) |
                              (static_cast<uint32_t>(metadata.outcome) << outcome_shift);
  return {classification,
          {metadata.address, metadata.byte_count, descriptor, metadata.epoch_before,
           metadata.epoch_after}};
}

ConSanMoiSampledSyncDecodeResult
decode_consan_moi_sampled_sync_metadata(const ConSanMoiSampledSyncMetadataPacked &packed) {
  using Classification = ConSanMoiSampledSyncClassification;
  if (packed == ConSanMoiSampledSyncMetadataPacked{})
    return {Classification::Empty, {}};
  if (packed.descriptor == kConSanMoiSampledSyncPublishingDescriptor)
    return {Classification::Publishing, {}};
  if ((packed.descriptor & consan_moi_sampled_sync_abi::reserved_mask) != 0)
    return {Classification::Malformed, {}};

  const auto extract = [&](uint32_t shift, uint32_t bits) {
    return (packed.descriptor >> shift) & ((uint32_t{1} << bits) - 1u);
  };
  ConSanMoiSampledSyncMetadata metadata{
      extract(consan_moi_sampled_sync_abi::version_shift,
              consan_moi_sampled_sync_abi::version_bits),
      packed.address,
      packed.byte_count,
      static_cast<ConSanMoiSampledSyncKind>(
          extract(consan_moi_sampled_sync_abi::kind_shift, consan_moi_sampled_sync_abi::kind_bits)),
      static_cast<ConSanMoiSampledSyncRole>(
          extract(consan_moi_sampled_sync_abi::role_shift, consan_moi_sampled_sync_abi::role_bits)),
      static_cast<ConSanMoiSampledSyncScope>(extract(consan_moi_sampled_sync_abi::scope_shift,
                                                     consan_moi_sampled_sync_abi::scope_bits)),
      static_cast<ConSanMoiSampledSyncOutcome>(extract(consan_moi_sampled_sync_abi::outcome_shift,
                                                       consan_moi_sampled_sync_abi::outcome_bits)),
      packed.epoch_before,
      packed.epoch_after,
  };
  const Classification classification = classify_consan_moi_sampled_sync_metadata(metadata);
  return {classification, metadata};
}

ConSanMoiSampledSyncDecodeResult
classify_consan_moi_sampled_sync_snapshot(ConSanMoiSampledSyncSnapshotWords words,
                                          uint32_t paired_window_epoch) {
  using Classification = ConSanMoiSampledSyncClassification;
  if (words.descriptor_before != words.descriptor_after)
    return {Classification::ChangedDuringRead, {}};
  // The descriptor is the commit word. Ignore the copy captured with the
  // payload so callers cannot accidentally validate anything but the stable
  // descriptor observations which bracketed it.
  words.packed.descriptor = words.descriptor_after;
  ConSanMoiSampledSyncDecodeResult decoded = decode_consan_moi_sampled_sync_metadata(words.packed);
  if (decoded.classification == Classification::Valid) {
    if (decoded.metadata.kind == ConSanMoiSampledSyncKind::Atomic &&
        (decoded.metadata.epoch_before != paired_window_epoch ||
         decoded.metadata.epoch_after != paired_window_epoch))
      return {Classification::UnsupportedSequence, decoded.metadata};
    if (decoded.metadata.kind == ConSanMoiSampledSyncKind::Barrier &&
        decoded.metadata.epoch_before != paired_window_epoch)
      return {Classification::UnsupportedSequence, decoded.metadata};
  }
  return decoded;
}

ConSanMoiSampledPendingAcquireState
classify_consan_moi_sampled_pending_acquire(const ConSanMoiSampledPendingAcquireView &view,
                                            const ConSanMoiSampledPendingAcquireKey &key,
                                            uint32_t window_epoch) {
  using State = ConSanMoiSampledPendingAcquireState;
  if (view.version_before != view.version_after)
    return State::ChangedDuringRead;
  if (view.version_after == 0)
    return view.slot == ConSanMoiSampledPendingAcquireSlot{} ? State::Empty : State::Malformed;
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
  const ConSanMoiSampledSyncDecodeResult decoded =
      decode_consan_moi_sampled_sync_metadata(view.slot.metadata);
  const uint32_t role = static_cast<uint32_t>(decoded.metadata.role);
  if (decoded.classification != ConSanMoiSampledSyncClassification::Valid ||
      decoded.metadata.kind != ConSanMoiSampledSyncKind::Atomic ||
      (role & static_cast<uint32_t>(ConSanMoiSampledSyncRole::Acquire)) == 0)
    return State::Malformed;
  return State::Ready;
}

bool consan_moi_sampled_qualifies_barrier_sequence(const ConSanSyncSequence &sequence) {
  const bool static_id =
      sequence.barrier_operand_source == ConSanBarrierSite::OperandSource::Immediate ||
      sequence.barrier_operand_source == ConSanBarrierSite::OperandSource::Literal32;
  // A physical barrier sequence can be reached from several kernel
  // descriptors through shared helper code. Ownership is still proven when
  // every reachable descriptor is known; lowering validates that all owners
  // have compatible ABI inputs and a preceding selected causal window.
  const bool owner_proven = !sequence.execution_owners.empty();
  return sequence.kind == ConSanSyncSequenceKind::Barrier &&
         sequence.operation == ConSanSyncOperation::BarrierFull &&
         sequence.memory_role == ConSanSyncMemoryRole::AcquireRelease &&
         consan_sync_confidence_meets(sequence.confidence,
                                      ConSanSemanticConfidence::Conservative) &&
         consan_sync_confidence_meets(sequence.memory_role_confidence,
                                      ConSanSemanticConfidence::Conservative) &&
         sequence.basic_block_index && !sequence.inside_scalar_clause && owner_proven &&
         static_id && sequence.barrier_id &&
         (sequence.barrier_scope == ConSanBarrierSite::Scope::Workgroup ||
          sequence.barrier_scope == ConSanBarrierSite::Scope::Cluster) &&
         (sequence.member_event_identities.size() == 1u ||
          sequence.member_event_identities.size() == 2u) &&
         sequence.begin_text_offset < sequence.end_text_offset;
}

bool consan_moi_sampled_atomic_attachment_matches(const ConSanMoiSampledCausalWindow &window,
                                                  uint64_t packed_watchpoint, uint32_t slot,
                                                  const ConSanMoiSampledAtomicAttachmentKey &key) {
  if (window.publication_state !=
          static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Ready) ||
      window.generation != key.generation || window.dispatch_id != key.dispatch_id ||
      window.workgroup_x != key.workgroup_x || window.workgroup_y != key.workgroup_y ||
      window.workgroup_z != key.workgroup_z || window.epoch != key.epoch ||
      window.cluster_workgroup_id != key.cluster_workgroup_id || window.first_entry != slot ||
      window.entry_count != 1u)
    return false;
  const ConSanMoiSampledWatchpointEntry watchpoint =
      decode_consan_moi_sampled_watchpoint_entry(packed_watchpoint);
  return watchpoint.valid && !watchpoint.consumed &&
         (watchpoint.kind == ConSanMoiShadowAccessKind::Read ||
          watchpoint.kind == ConSanMoiShadowAccessKind::Write) &&
         watchpoint.owner_id == key.owner_id && watchpoint.epoch == key.epoch &&
         watchpoint.generation == (static_cast<uint32_t>(key.generation) &
                                   consan_moi_sampled_watchpoint::max_generation);
}

} // namespace rocjitsu
