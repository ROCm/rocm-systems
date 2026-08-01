// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi.h"

#include "rocjitsu/analysis/def_use_chain.h"
#include "rocjitsu/analysis/kernel_scope.h"
#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/consan/consan_resource.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"
#include "rocjitsu/code/patch/spill_manager.h"
#include "rocjitsu/code/patch/trampoline_builder.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "util/bit.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <algorithm>
#include <array>
#include <bit>
#include <compare>
#include <cstddef>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <ranges>
#include <set>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rocjitsu {

namespace {

[[nodiscard]] std::optional<ConSanMoiAtomicOutcome>
resolved_atomic_outcome(const ConSanMoiRecordReplayAtomicEvent &record) {
  if (record.operation == ConSanMoiAtomicOperation::Rmw)
    return record.outcome == ConSanMoiAtomicOutcome::NotApplicable
               ? std::optional{ConSanMoiAtomicOutcome::NotApplicable}
               : std::nullopt;
  if (record.operation != ConSanMoiAtomicOperation::CompareExchange)
    return std::nullopt;
  if (record.outcome == ConSanMoiAtomicOutcome::Success ||
      record.outcome == ConSanMoiAtomicOutcome::Failure)
    return record.outcome;
  if (record.outcome != ConSanMoiAtomicOutcome::Unavailable || record.lane_mask == 0 ||
      (record.success_lane_mask & ~record.lane_mask) != 0)
    return std::nullopt;
  if (record.success_lane_mask == 0)
    return ConSanMoiAtomicOutcome::Failure;
  if (record.success_lane_mask == record.lane_mask)
    return ConSanMoiAtomicOutcome::Success;
  // Record/replay ordering is currently wave-owned. A mixed lane outcome
  // cannot be collapsed to a wave release without ordering failed lanes.
  return std::nullopt;
}

[[nodiscard]] bool is_unpublished_atomic_record(const ConSanMoiRecordReplayAtomicEvent &record) {
  return record.generation == 0 && record.workgroup_x == 0 && record.workgroup_y == 0 &&
         record.workgroup_z == 0 && record.owner_id == 0 && record.atomic_address == 0 &&
         record.instruction_offset == 0 && record.event_index == 0 && record.epoch == 0 &&
         (static_cast<uint32_t>(record.kind) == 0 ||
          record.kind == ConSanMoiAtomicEventKind::Release) &&
         record.scope == 0 && record.semantics == 0 &&
         (static_cast<uint32_t>(record.operation) == 0 ||
          record.operation == ConSanMoiAtomicOperation::Rmw) &&
         record.outcome == ConSanMoiAtomicOutcome::NotApplicable && record.lane_mask == 0 &&
         record.success_lane_mask == 0;
}

[[nodiscard]] bool is_unpublished_fence_record(const ConSanMoiRecordReplayFenceEvent &record) {
  return record.generation == 0 && record.workgroup_x == 0 && record.workgroup_y == 0 &&
         record.workgroup_z == 0 && record.owner_id == 0 && record.instruction_offset == 0 &&
         record.event_index == 0 && record.epoch == 0 &&
         (static_cast<uint32_t>(record.kind) == 0 ||
          record.kind == ConSanMoiFenceEventKind::Release) &&
         record.scope == 0 && record.semantics == 0 && record.communication_token == 0;
}

} // namespace

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
  if (view.slot.version != view.version_after || view.slot.reserved != 0)
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

ConSanMoiSampledSyncPublishResult
consan_moi_sampled_publish_sync_metadata(std::span<ConSanMoiSampledSyncMetadataPacked> slots,
                                         uint32_t selected_slot,
                                         const ConSanMoiSampledSyncMetadata &metadata) {
  const ConSanMoiSampledSyncEncodeResult encoded =
      encode_consan_moi_sampled_sync_metadata(metadata);
  if (encoded.classification != ConSanMoiSampledSyncClassification::Valid)
    return {ConSanMoiSampledSyncPublishOutcome::Rejected, encoded.classification, selected_slot};
  if (selected_slot >= slots.size())
    return {ConSanMoiSampledSyncPublishOutcome::CapacityExhausted,
            ConSanMoiSampledSyncClassification::Valid, selected_slot};

  ConSanMoiSampledSyncMetadataPacked &slot = slots[selected_slot];
  const ConSanMoiSampledSyncDecodeResult existing = decode_consan_moi_sampled_sync_metadata(slot);
  if (existing.classification == ConSanMoiSampledSyncClassification::Empty) {
    slot = encoded.packed;
    return {ConSanMoiSampledSyncPublishOutcome::Published,
            ConSanMoiSampledSyncClassification::Valid, selected_slot};
  }
  if (existing.classification != ConSanMoiSampledSyncClassification::Valid)
    return {ConSanMoiSampledSyncPublishOutcome::MalformedSlot, existing.classification,
            selected_slot};
  if (slot == encoded.packed)
    return {ConSanMoiSampledSyncPublishOutcome::Existing, ConSanMoiSampledSyncClassification::Valid,
            selected_slot};
  return {ConSanMoiSampledSyncPublishOutcome::Collision, ConSanMoiSampledSyncClassification::Valid,
          selected_slot};
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

ConSanMoiRecordReplayTraceHeader consan_moi_compact_record_replay_trace(
    uint64_t generation, uint64_t dispatch_id,
    std::span<const ConSanMoiAccessRecord> access_records,
    std::span<const ConSanMoiBarrierRecord> barrier_records,
    std::span<const ConSanMoiRecordReplayAtomicEvent> atomic_events,
    std::span<ConSanMoiRecordReplayPcEntry> dictionary,
    std::span<ConSanMoiRecordReplayWorkgroupRun> workgroup_runs,
    std::span<ConSanMoiRecordReplayCompactEvent> events) {
  return consan_moi_compact_record_replay_trace(
      generation, dispatch_id, access_records, barrier_records, atomic_events,
      std::span<const ConSanMoiRecordReplayFenceEvent>{}, dictionary, workgroup_runs, events);
}

ConSanMoiRecordReplayTraceHeader consan_moi_compact_record_replay_trace(
    uint64_t generation, uint64_t dispatch_id,
    std::span<const ConSanMoiAccessRecord> access_records,
    std::span<const ConSanMoiBarrierRecord> barrier_records,
    std::span<const ConSanMoiRecordReplayAtomicEvent> atomic_events,
    std::span<const ConSanMoiRecordReplayFenceEvent> fence_events,
    std::span<ConSanMoiRecordReplayPcEntry> dictionary,
    std::span<ConSanMoiRecordReplayWorkgroupRun> workgroup_runs,
    std::span<ConSanMoiRecordReplayCompactEvent> events) {
  auto capacity = [](size_t size) { return consan_moi_clamp_u32_capacity(size); };
  ConSanMoiRecordReplayTraceHeader header;
  header.generation = generation;
  header.dispatch_id = dispatch_id;
  header.dictionary_capacity = capacity(dictionary.size());
  header.workgroup_run_capacity = capacity(workgroup_runs.size());
  header.event_capacity = capacity(events.size());

  struct InputEvent {
    uint64_t generation = 0;
    uint32_t workgroup_x = 0;
    uint32_t workgroup_y = 0;
    uint32_t workgroup_z = 0;
    uint32_t instruction_offset = 0;
    uint32_t event_index = 0;
    uint32_t input_order = 0;
    uint32_t owner_id = 0;
    uint32_t epoch = 0;
    uint64_t lane_mask = 0;
    uint64_t payload = 0;
    ConSanMoiRecordReplayEventKind kind = ConSanMoiRecordReplayEventKind::Access;
    uint8_t operation = 0;
    uint32_t scope = 0;
    uint32_t semantics = 0;
    ConSanMoiAtomicOperation atomic_operation = ConSanMoiAtomicOperation::Rmw;
    ConSanMoiAtomicOutcome atomic_outcome = ConSanMoiAtomicOutcome::NotApplicable;
  };

  std::vector<InputEvent> input;
  input.reserve(access_records.size() + barrier_records.size() + atomic_events.size() +
                fence_events.size());
  uint32_t input_order = 0;
  for (const ConSanMoiAccessRecord &record : access_records) {
    if (record.access_kind > std::numeric_limits<uint8_t>::max()) {
      header.flags |= kConSanMoiRecordReplayTraceRejectedInput;
      ++header.rejected_event_count;
      continue;
    }
    input.push_back({record.generation, record.workgroup_x, record.workgroup_y, record.workgroup_z,
                     record.instruction_offset, record.event_index, input_order++, record.wave_id,
                     record.epoch, record.lane_mask,
                     static_cast<uint64_t>(record.lds_byte_offset) |
                         (static_cast<uint64_t>(record.lds_byte_count) << 32u),
                     ConSanMoiRecordReplayEventKind::Access,
                     static_cast<uint8_t>(record.access_kind), 0, 0, ConSanMoiAtomicOperation::Rmw,
                     ConSanMoiAtomicOutcome::NotApplicable});
  }
  for (const ConSanMoiBarrierRecord &record : barrier_records) {
    input.push_back({record.generation, record.workgroup_x, record.workgroup_y, record.workgroup_z,
                     record.instruction_offset, record.event_index, input_order++, record.wave_id,
                     0, record.lane_mask, 0, ConSanMoiRecordReplayEventKind::Barrier, 0, 0, 0,
                     ConSanMoiAtomicOperation::Rmw, ConSanMoiAtomicOutcome::NotApplicable});
  }
  for (const ConSanMoiRecordReplayAtomicEvent &record : atomic_events) {
    if (is_unpublished_atomic_record(record))
      continue;
    const std::optional<ConSanMoiAtomicOutcome> outcome = resolved_atomic_outcome(record);
    if (static_cast<uint32_t>(record.kind) > std::numeric_limits<uint8_t>::max() ||
        record.owner_id > std::numeric_limits<uint16_t>::max() || !outcome) {
      header.flags |= kConSanMoiRecordReplayTraceRejectedInput;
      ++header.rejected_event_count;
      continue;
    }
    input.push_back({record.generation, record.workgroup_x, record.workgroup_y, record.workgroup_z,
                     record.instruction_offset, record.event_index, input_order++, record.owner_id,
                     record.epoch, 0, record.atomic_address, ConSanMoiRecordReplayEventKind::Atomic,
                     static_cast<uint8_t>(record.kind), record.scope, record.semantics,
                     record.operation, *outcome});
  }
  for (const ConSanMoiRecordReplayFenceEvent &record : fence_events) {
    if (is_unpublished_fence_record(record))
      continue;
    if ((record.kind != ConSanMoiFenceEventKind::Release &&
         record.kind != ConSanMoiFenceEventKind::Acquire &&
         record.kind != ConSanMoiFenceEventKind::AcquireRelease) ||
        record.scope == 0 || record.scope > 3 ||
        record.owner_id > std::numeric_limits<uint16_t>::max() || record.communication_token == 0) {
      header.flags |= kConSanMoiRecordReplayTraceRejectedInput;
      ++header.rejected_event_count;
      continue;
    }
    input.push_back({record.generation, record.workgroup_x, record.workgroup_y, record.workgroup_z,
                     record.instruction_offset, record.event_index, input_order++, record.owner_id,
                     record.epoch, 0, record.communication_token,
                     ConSanMoiRecordReplayEventKind::Fence, static_cast<uint8_t>(record.kind),
                     record.scope, record.semantics, ConSanMoiAtomicOperation::Rmw,
                     ConSanMoiAtomicOutcome::NotApplicable});
  }
  std::stable_sort(input.begin(), input.end(), [](const InputEvent &lhs, const InputEvent &rhs) {
    if (lhs.event_index != rhs.event_index)
      return lhs.event_index < rhs.event_index;
    return lhs.input_order < rhs.input_order;
  });

  auto same_workgroup = [](const ConSanMoiRecordReplayWorkgroupRun &run, const InputEvent &event) {
    return run.workgroup_x == event.workgroup_x && run.workgroup_y == event.workgroup_y &&
           run.workgroup_z == event.workgroup_z;
  };
  for (size_t input_index = 0; input_index < input.size(); ++input_index) {
    const InputEvent &event = input[input_index];
    if (event.generation != 0 && event.generation != generation) {
      header.flags |= kConSanMoiRecordReplayTraceRejectedInput;
      ++header.rejected_event_count;
      continue;
    }
    if (event.operation > std::numeric_limits<uint8_t>::max() ||
        event.owner_id > std::numeric_limits<uint16_t>::max()) {
      header.flags |= kConSanMoiRecordReplayTraceRejectedInput;
      ++header.rejected_event_count;
      continue;
    }

    uint32_t pc_index = header.dictionary_count;
    for (uint32_t i = 0; i < header.dictionary_count; ++i) {
      const ConSanMoiRecordReplayPcEntry &entry = dictionary[i];
      if (entry.instruction_offset == event.instruction_offset && entry.kind == event.kind &&
          entry.operation == event.operation && entry.scope == event.scope &&
          entry.semantics == event.semantics && entry.atomic_operation == event.atomic_operation) {
        pc_index = i;
        break;
      }
    }
    const bool needs_dictionary = pc_index == header.dictionary_count;
    const bool continues_run =
        header.workgroup_run_count != 0 &&
        same_workgroup(workgroup_runs[header.workgroup_run_count - 1u], event);

    if (!needs_dictionary && continues_run && header.event_count != 0 && event.lane_mask != 0) {
      ConSanMoiRecordReplayCompactEvent &prior = events[header.event_count - 1u];
      if (prior.pc_index == pc_index && prior.event_index == event.event_index &&
          prior.owner_id == event.owner_id && prior.epoch == event.epoch &&
          prior.payload == event.payload && prior.lane_mask != 0) {
        prior.lane_mask |= event.lane_mask;
        ++header.lane_coalesced_record_count;
        continue;
      }
    }

    const bool needs_run = !continues_run;
    if ((needs_dictionary && header.dictionary_count == header.dictionary_capacity) ||
        (needs_run && header.workgroup_run_count == header.workgroup_run_capacity) ||
        header.event_count == header.event_capacity) {
      header.flags |= kConSanMoiRecordReplayTraceOverflow;
      header.dropped_event_count = capacity(input.size() - input_index);
      break;
    }

    if (needs_dictionary) {
      dictionary[header.dictionary_count] = {event.instruction_offset, event.kind,
                                             event.operation,          event.scope,
                                             event.semantics,          event.atomic_operation};
      ++header.dictionary_count;
    }
    if (needs_run) {
      workgroup_runs[header.workgroup_run_count] = {
          event.workgroup_x, event.workgroup_y, event.workgroup_z, header.event_count, 0, 0};
      ++header.workgroup_run_count;
    }
    events[header.event_count] = {
        pc_index,        event.event_index, event.owner_id,      event.epoch,
        event.lane_mask, event.payload,     event.atomic_outcome};
    ++header.event_count;
    ++workgroup_runs[header.workgroup_run_count - 1u].event_count;
  }
  return header;
}

ConSanMoiRecordReplayCaptureResult consan_moi_plan_record_replay_capture(
    const ConSanMoiRecordReplayTraceHeader &header,
    std::span<const ConSanMoiRecordReplayPcEntry> dictionary,
    std::span<const ConSanMoiRecordReplayWorkgroupRun> workgroup_runs,
    std::span<const ConSanMoiRecordReplayCompactEvent> events,
    ConSanMoiRecordReplayCaptureLimits limits,
    std::span<ConSanMoiRecordReplayCaptureWindow> windows,
    std::span<uint32_t> event_window_indices) {
  ConSanMoiRecordReplayCaptureResult result;
  std::fill(event_window_indices.begin(), event_window_indices.end(),
            kConSanMoiRecordReplayUncapturedEvent);
  auto invalid = [&]() {
    result.invalid_trace = true;
    return result;
  };
  if (header.magic != kConSanMoiRecordReplayTraceMagic ||
      header.abi_version != kConSanMoiRecordReplayTraceAbiVersion ||
      header.header_size != kConSanMoiRecordReplayTraceHeaderBytes || header.flags != 0 ||
      header.dictionary_count > header.dictionary_capacity ||
      header.workgroup_run_count > header.workgroup_run_capacity ||
      header.event_count > header.event_capacity || header.dictionary_count > dictionary.size() ||
      header.workgroup_run_count > workgroup_runs.size() || header.event_count > events.size() ||
      header.event_count > event_window_indices.size())
    return invalid();

  uint32_t expected_first_event = 0;
  for (uint32_t i = 0; i < header.workgroup_run_count; ++i) {
    const ConSanMoiRecordReplayWorkgroupRun &run = workgroup_runs[i];
    if (run.event_count == 0 || run.first_event != expected_first_event ||
        run.event_count > header.event_count - expected_first_event)
      return invalid();
    expected_first_event += run.event_count;
  }
  if (expected_first_event != header.event_count)
    return invalid();
  for (uint32_t i = 0; i < header.event_count; ++i) {
    if (events[i].pc_index >= header.dictionary_count)
      return invalid();
    if (i != 0 && events[i].event_index < events[i - 1u].event_index)
      return invalid();
    switch (dictionary[events[i].pc_index].kind) {
    case ConSanMoiRecordReplayEventKind::Access:
    case ConSanMoiRecordReplayEventKind::Barrier:
    case ConSanMoiRecordReplayEventKind::Atomic:
    case ConSanMoiRecordReplayEventKind::Fence:
      break;
    default:
      return invalid();
    }
  }

  struct WorkgroupState {
    uint32_t x = 0;
    uint32_t y = 0;
    uint32_t z = 0;
    uint32_t epoch = 0;
    uint32_t selected_epoch_count = 0;
    bool in_barrier_run = false;
    uint32_t barrier_pc_index = 0;
    bool selected = false;
    bool blocked = false;
  };
  struct CandidateWindow {
    uint32_t workgroup = 0;
    uint32_t epoch = 0;
    uint32_t first_event_position = 0;
    uint32_t last_event_position = 0;
    uint32_t first_event_index = 0;
    uint32_t last_event_index = 0;
    uint32_t event_count = 0;
  };
  std::vector<WorkgroupState> workgroups;
  std::vector<CandidateWindow> candidates;
  std::vector<uint32_t> event_candidates(header.event_count, kConSanMoiRecordReplayUncapturedEvent);
  auto find_workgroup = [&](const ConSanMoiRecordReplayWorkgroupRun &run) {
    for (uint32_t i = 0; i < workgroups.size(); ++i) {
      if (workgroups[i].x == run.workgroup_x && workgroups[i].y == run.workgroup_y &&
          workgroups[i].z == run.workgroup_z)
        return i;
    }
    workgroups.push_back({run.workgroup_x, run.workgroup_y, run.workgroup_z});
    return static_cast<uint32_t>(workgroups.size() - 1u);
  };
  auto find_candidate = [&](uint32_t workgroup, uint32_t epoch, uint32_t event_position) {
    for (uint32_t i = 0; i < candidates.size(); ++i) {
      if (candidates[i].workgroup == workgroup && candidates[i].epoch == epoch)
        return i;
    }
    const uint32_t event_index = events[event_position].event_index;
    candidates.push_back(
        {workgroup, epoch, event_position, event_position, event_index, event_index, 0});
    return static_cast<uint32_t>(candidates.size() - 1u);
  };

  for (uint32_t run_index = 0; run_index < header.workgroup_run_count; ++run_index) {
    const ConSanMoiRecordReplayWorkgroupRun &run = workgroup_runs[run_index];
    const uint32_t workgroup_index = find_workgroup(run);
    WorkgroupState &state = workgroups[workgroup_index];
    for (uint32_t position = run.first_event; position < run.first_event + run.event_count;
         ++position) {
      const ConSanMoiRecordReplayCompactEvent &event = events[position];
      const bool barrier =
          dictionary[event.pc_index].kind == ConSanMoiRecordReplayEventKind::Barrier;
      if (state.in_barrier_run && (!barrier || event.pc_index != state.barrier_pc_index)) {
        if (state.epoch == std::numeric_limits<uint32_t>::max())
          return invalid();
        ++state.epoch;
        state.in_barrier_run = false;
      }
      const uint32_t candidate_index = find_candidate(workgroup_index, state.epoch, position);
      CandidateWindow &candidate = candidates[candidate_index];
      candidate.last_event_position = position;
      candidate.last_event_index = event.event_index;
      ++candidate.event_count;
      event_candidates[position] = candidate_index;
      if (barrier) {
        state.in_barrier_run = true;
        state.barrier_pc_index = event.pc_index;
      }
    }
  }

  std::vector<uint32_t> selected_window(candidates.size(), kConSanMoiRecordReplayUncapturedEvent);
  for (uint32_t candidate_index = 0; candidate_index < candidates.size(); ++candidate_index) {
    const CandidateWindow &candidate = candidates[candidate_index];
    WorkgroupState &state = workgroups[candidate.workgroup];
    if (state.blocked)
      continue;
    if (!state.selected && result.selected_workgroup_count == limits.workgroup_limit) {
      state.blocked = true;
      result.workgroup_limit_exhausted = true;
      continue;
    }
    if (state.selected_epoch_count == limits.epochs_per_workgroup_limit) {
      state.blocked = true;
      result.epoch_limit_exhausted = true;
      continue;
    }
    if (candidate.event_count > limits.event_budget - result.selected_event_count) {
      result.event_budget_exhausted = true;
      break;
    }
    if (result.selected_window_count == windows.size()) {
      result.window_capacity_exhausted = true;
      break;
    }
    if (!state.selected) {
      state.selected = true;
      ++result.selected_workgroup_count;
    }
    const uint32_t window_index = result.selected_window_count++;
    selected_window[candidate_index] = window_index;
    ++state.selected_epoch_count;
    result.selected_event_count += candidate.event_count;
    windows[window_index] = {state.x,
                             state.y,
                             state.z,
                             candidate.epoch,
                             candidate.first_event_position,
                             candidate.last_event_position,
                             candidate.first_event_index,
                             candidate.last_event_index,
                             candidate.event_count,
                             0};
  }
  for (uint32_t position = 0; position < header.event_count; ++position) {
    const uint32_t candidate_index = event_candidates[position];
    if (candidate_index < selected_window.size())
      event_window_indices[position] = selected_window[candidate_index];
  }
  result.omitted_event_count = header.event_count - result.selected_event_count;
  return result;
}

bool consan_moi_acquired_epoch_orders(std::span<const ConSanMoiAcquiredEpochToken> tokens,
                                      const ConSanMoiExactShadowEntry &current,
                                      const ConSanMoiExactShadowEntry &prior) {
  const uint32_t required_token = consan_moi_acquired_epoch_token_value(prior.epoch);
  if (required_token == 0)
    return false;
  for (const ConSanMoiAcquiredEpochToken &token : tokens) {
    if (!token.valid)
      continue;
    if (token.generation != current.generation || token.generation != prior.generation)
      continue;
    if (token.consumer_owner_id != current.owner_id || token.producer_owner_id != prior.owner_id)
      continue;
    if (token.producer_epoch_plus_one >= required_token)
      return true;
  }
  return false;
}

ConSanMoiAtomicSyncResult
consan_moi_record_replay_atomic_release(std::span<ConSanMoiAtomicReleaseRecord> release_records,
                                        uint64_t generation, uint64_t atomic_address,
                                        uint32_t producer_owner_id, uint32_t producer_epoch,
                                        uint32_t release_instruction_offset) {
  ConSanMoiAtomicSyncResult result;
  ConSanMoiAtomicReleaseRecord *empty_slot = nullptr;
  for (ConSanMoiAtomicReleaseRecord &record : release_records) {
    if (!record.valid) {
      if (empty_slot == nullptr)
        empty_slot = &record;
      continue;
    }
    if (record.generation != generation || record.atomic_address != atomic_address ||
        record.producer_owner_id != producer_owner_id)
      continue;
    if (producer_epoch > record.producer_epoch) {
      record.producer_epoch = producer_epoch;
      record.release_instruction_offset = release_instruction_offset;
      result.updated_record_count = 1;
    }
    return result;
  }

  if (empty_slot == nullptr) {
    result.metadata_full = true;
    return result;
  }

  *empty_slot = ConSanMoiAtomicReleaseRecord{
      /*valid=*/true,    generation,     atomic_address,
      producer_owner_id, producer_epoch, release_instruction_offset,
  };
  result.updated_record_count = 1;
  return result;
}

namespace {

[[nodiscard]] ConSanMoiAtomicSyncResult
record_acquired_epoch_token(std::span<ConSanMoiAcquiredEpochToken> tokens, uint64_t generation,
                            uint32_t consumer_owner_id, uint32_t producer_owner_id,
                            uint32_t producer_epoch_plus_one, uint32_t acquire_instruction_offset) {
  ConSanMoiAtomicSyncResult result;
  ConSanMoiAcquiredEpochToken *empty_slot = nullptr;
  for (ConSanMoiAcquiredEpochToken &token : tokens) {
    if (!token.valid) {
      if (empty_slot == nullptr)
        empty_slot = &token;
      continue;
    }
    if (token.generation != generation || token.consumer_owner_id != consumer_owner_id ||
        token.producer_owner_id != producer_owner_id)
      continue;
    if (producer_epoch_plus_one > token.producer_epoch_plus_one) {
      token.producer_epoch_plus_one = producer_epoch_plus_one;
      token.acquire_instruction_offset = acquire_instruction_offset;
      result.updated_record_count = 1;
    }
    return result;
  }

  if (empty_slot == nullptr) {
    result.metadata_full = true;
    return result;
  }

  *empty_slot = ConSanMoiAcquiredEpochToken{
      /*valid=*/true,          generation,
      consumer_owner_id,       producer_owner_id,
      producer_epoch_plus_one, acquire_instruction_offset,
  };
  result.updated_record_count = 1;
  return result;
}

} // namespace

ConSanMoiAtomicSyncResult consan_moi_record_replay_atomic_acquire(
    std::span<const ConSanMoiAtomicReleaseRecord> release_records,
    std::span<ConSanMoiAcquiredEpochToken> acquired_epoch_tokens, uint64_t generation,
    uint64_t atomic_address, uint32_t consumer_owner_id, uint32_t acquire_instruction_offset) {
  ConSanMoiAtomicSyncResult result;
  for (const ConSanMoiAtomicReleaseRecord &release : release_records) {
    if (!release.valid)
      continue;
    if (release.generation != generation || release.atomic_address != atomic_address ||
        release.producer_owner_id == consumer_owner_id)
      continue;
    const ConSanMoiAtomicSyncResult token_result = record_acquired_epoch_token(
        acquired_epoch_tokens, generation, consumer_owner_id, release.producer_owner_id,
        consan_moi_acquired_epoch_token_value(release.producer_epoch), acquire_instruction_offset);
    result.metadata_full |= token_result.metadata_full;
    result.updated_record_count += token_result.updated_record_count;
  }
  return result;
}

ConSanMoiSparseExactByteShadow::ConSanMoiSparseExactByteShadow(uint64_t byte_capacity,
                                                               uint32_t maximum_access_count)
    : byte_capacity_(byte_capacity), maximum_access_count_(maximum_access_count) {}

ConSanMoiExactByteAccessResult ConSanMoiSparseExactByteShadow::access(
    const ConSanMoiExactByteAccess &current,
    std::span<const ConSanMoiAcquiredEpochToken> acquired_epoch_tokens) {
  ConSanMoiExactByteAccessResult result;
  if (current.lds_byte_count == 0 || consan_moi_shadow_kind_is_empty(current.kind))
    return result;

  const uint64_t byte_end = static_cast<uint64_t>(current.lds_byte_offset) + current.lds_byte_count;
  if (byte_end > byte_capacity_) {
    result.capacity_exhausted = true;
    return result;
  }

  const ConSanMoiExactShadowEntry current_entry{
      current.kind,
      current.owner_id,
      current.epoch,
      static_cast<uint32_t>(current.generation),
      current.instruction_offset,
  };

  std::optional<uint32_t> newest_conflict_index;
  const auto consider_intervals = [&](const IntervalMap &intervals) {
    auto overlap = intervals.upper_bound(current.lds_byte_offset);
    if (overlap != intervals.begin()) {
      auto prior_interval = std::prev(overlap);
      if (prior_interval->second.end > current.lds_byte_offset)
        overlap = prior_interval;
    }
    for (; overlap != intervals.end() && overlap->first < byte_end; ++overlap) {
      if (overlap->second.end <= current.lds_byte_offset)
        continue;
      const uint32_t prior_index = overlap->second.provenance_index;
      if (prior_index == 0 || prior_index > provenance_.size()) {
        result.capacity_exhausted = true;
        return false;
      }
      const ConSanMoiExactByteAccess &prior = provenance_[prior_index - 1u];
      if (!consan_moi_exact_byte_accesses_conflict(current, prior))
        continue;
      const ConSanMoiExactShadowEntry prior_entry{
          prior.kind,
          prior.owner_id,
          prior.epoch,
          static_cast<uint32_t>(prior.generation),
          prior.instruction_offset,
      };
      if (consan_moi_acquired_epoch_orders(acquired_epoch_tokens, current_entry, prior_entry))
        continue;
      if (!newest_conflict_index || prior_index > *newest_conflict_index)
        newest_conflict_index = prior_index;
    }
    return true;
  };
  constexpr std::array access_kinds{
      ConSanMoiShadowAccessKind::Read,
      ConSanMoiShadowAccessKind::Write,
      ConSanMoiShadowAccessKind::ReadWrite,
      ConSanMoiShadowAccessKind::Atomic,
  };
  for (const ConSanMoiShadowAccessKind prior_kind : access_kinds) {
    if (!consan_moi_shadow_kind_conflicts(current.kind, prior_kind))
      continue;
    const CrossOwnerClass first_class{
        current.generation,
        current.epoch,
        prior_kind,
        0,
    };
    for (auto prior_class = cross_owner_intervals_.lower_bound(first_class);
         prior_class != cross_owner_intervals_.end() &&
         prior_class->first.generation == current.generation &&
         prior_class->first.epoch == current.epoch && prior_class->first.kind == prior_kind;
         ++prior_class) {
      if (prior_class->first.owner_id == current.owner_id)
        continue;
      if (!consider_intervals(prior_class->second))
        return result;
    }
    if (!current.exact_address_group)
      continue;
    const SameSiteClass first_site_class{
        current.generation,
        current.epoch,
        prior_kind,
        current.owner_id,
        current.instruction_offset,
        0,
    };
    for (auto prior_site = same_site_intervals_.lower_bound(first_site_class);
         prior_site != same_site_intervals_.end() &&
         prior_site->first.generation == current.generation &&
         prior_site->first.epoch == current.epoch && prior_site->first.kind == prior_kind &&
         prior_site->first.owner_id == current.owner_id &&
         prior_site->first.instruction_offset == current.instruction_offset;
         ++prior_site) {
      if (!consider_intervals(prior_site->second))
        return result;
    }
  }
  if (newest_conflict_index) {
    result.conflict = true;
    result.prior = provenance_[*newest_conflict_index - 1u];
    return result;
  }

  if (provenance_.size() >= maximum_access_count_) {
    result.capacity_exhausted = true;
    return result;
  }
  provenance_.push_back(current);
  const uint32_t provenance_index = static_cast<uint32_t>(provenance_.size());
  const auto update_intervals = [&](IntervalMap &intervals) {
    auto replace_begin = intervals.lower_bound(current.lds_byte_offset);
    if (replace_begin != intervals.begin()) {
      auto prior_interval = std::prev(replace_begin);
      if (prior_interval->second.end > current.lds_byte_offset)
        replace_begin = prior_interval;
    }
    std::optional<std::pair<uint64_t, Interval>> left_remainder;
    std::optional<std::pair<uint64_t, Interval>> right_remainder;
    auto replace_end = replace_begin;
    for (; replace_end != intervals.end() && replace_end->first < byte_end; ++replace_end) {
      if (replace_end->second.end <= current.lds_byte_offset)
        continue;
      if (replace_end->first < current.lds_byte_offset) {
        left_remainder =
            std::pair{replace_end->first,
                      Interval{current.lds_byte_offset, replace_end->second.provenance_index}};
      }
      if (replace_end->second.end > byte_end) {
        right_remainder = std::pair{
            byte_end, Interval{replace_end->second.end, replace_end->second.provenance_index}};
      }
    }
    intervals.erase(replace_begin, replace_end);
    if (left_remainder)
      intervals.insert_or_assign(left_remainder->first, left_remainder->second);
    intervals.insert_or_assign(current.lds_byte_offset, Interval{byte_end, provenance_index});
    if (right_remainder)
      intervals.insert_or_assign(right_remainder->first, right_remainder->second);
  };
  update_intervals(cross_owner_intervals_[CrossOwnerClass{
      current.generation,
      current.epoch,
      current.kind,
      current.owner_id,
  }]);
  if (current.exact_address_group) {
    update_intervals(same_site_intervals_[SameSiteClass{
        current.generation,
        current.epoch,
        current.kind,
        current.owner_id,
        current.instruction_offset,
        current.lane_mask,
    }]);
  }
  return result;
}

ConSanMoiRecordReplayResult
consan_moi_record_replay_access_records(ConSanMoiReportHeader &header,
                                        std::span<const ConSanMoiAccessRecord> access_records,
                                        std::span<ConSanMoiDiagnosticRecord> diagnostic_records,
                                        std::span<uint64_t> exact_shadow_entries) {
  return consan_moi_record_replay_access_records(header, access_records,
                                                 std::span<const ConSanMoiBarrierRecord>{},
                                                 diagnostic_records, exact_shadow_entries);
}

ConSanMoiRecordReplayResult
consan_moi_record_replay_access_records(ConSanMoiReportHeader &header,
                                        std::span<const ConSanMoiAccessRecord> access_records,
                                        std::span<const ConSanMoiBarrierRecord> barrier_records,
                                        std::span<ConSanMoiDiagnosticRecord> diagnostic_records,
                                        std::span<uint64_t> exact_shadow_entries) {
  return consan_moi_record_replay_access_records(
      header, access_records, barrier_records, std::span<const ConSanMoiRecordReplayAtomicEvent>{},
      diagnostic_records, exact_shadow_entries);
}

ConSanMoiRecordReplayResult consan_moi_record_replay_access_records(
    ConSanMoiReportHeader &header, std::span<const ConSanMoiAccessRecord> access_records,
    std::span<const ConSanMoiBarrierRecord> barrier_records,
    std::span<const ConSanMoiRecordReplayAtomicEvent> atomic_events,
    std::span<ConSanMoiDiagnosticRecord> diagnostic_records,
    std::span<uint64_t> exact_shadow_entries) {
  return consan_moi_record_replay_access_records(
      header, access_records, barrier_records, atomic_events,
      std::span<const ConSanMoiRecordReplayFenceEvent>{}, diagnostic_records, exact_shadow_entries);
}

ConSanMoiRecordReplayResult consan_moi_record_replay_access_records(
    ConSanMoiReportHeader &header, std::span<const ConSanMoiAccessRecord> access_records,
    std::span<const ConSanMoiBarrierRecord> barrier_records,
    std::span<const ConSanMoiRecordReplayAtomicEvent> atomic_events,
    std::span<const ConSanMoiRecordReplayFenceEvent> fence_events,
    std::span<ConSanMoiDiagnosticRecord> diagnostic_records,
    std::span<uint64_t> exact_shadow_entries) {
  auto span_size_u32 = [](size_t size) {
    return size > std::numeric_limits<uint32_t>::max() ? std::numeric_limits<uint32_t>::max()
                                                       : static_cast<uint32_t>(size);
  };
  auto decode_access_kind = [](uint32_t value) -> std::optional<ConSanMoiShadowAccessKind> {
    switch (static_cast<ConSanMoiShadowAccessKind>(value)) {
    case ConSanMoiShadowAccessKind::Read:
    case ConSanMoiShadowAccessKind::Write:
    case ConSanMoiShadowAccessKind::ReadWrite:
    case ConSanMoiShadowAccessKind::Atomic:
      return static_cast<ConSanMoiShadowAccessKind>(value);
    case ConSanMoiShadowAccessKind::Empty:
      return std::nullopt;
    }
    return std::nullopt;
  };
  auto is_unpublished_access = [](const ConSanMoiAccessRecord &record) {
    return record.claim_token == 0 && record.generation == 0 && record.workgroup_x == 0 &&
           record.workgroup_y == 0 && record.workgroup_z == 0 && record.wave_id == 0 &&
           record.lane_mask == 0 && record.instruction_offset == 0 &&
           record.access_kind == static_cast<uint32_t>(ConSanMoiShadowAccessKind::Empty) &&
           record.lds_byte_offset == 0 && record.lds_byte_count == 0 && record.start_cell == 0 &&
           record.cell_count == 0 && record.epoch == 0 && record.event_index == 0 &&
           record.site_token == 0 && record.flags == 0;
  };

  ConSanMoiRecordReplayResult replay;
  const uint32_t access_count = std::min({header.access_record_count, header.access_record_capacity,
                                          span_size_u32(access_records.size())});
  replay.dropped_access_count =
      header.access_record_count > access_count ? header.access_record_count - access_count : 0;
  const uint32_t barrier_count =
      std::min({header.barrier_record_count, header.barrier_record_capacity,
                span_size_u32(barrier_records.size())});
  replay.dropped_barrier_count =
      header.barrier_record_count > barrier_count ? header.barrier_record_count - barrier_count : 0;
  const uint32_t diagnostic_capacity =
      std::min(header.diagnostic_capacity, span_size_u32(diagnostic_records.size()));

  struct ReplayWorkgroupState {
    struct ReportedDiagnosticKey {
      uint32_t kind = 0;
      uint32_t first_site = 0;
      uint32_t second_site = 0;
      uint64_t overlap_begin = 0;
      uint64_t overlap_end = 0;

      auto operator<=>(const ReportedDiagnosticKey &) const = default;
    };

    ReplayWorkgroupState(uint64_t exact_byte_capacity, uint32_t access_capacity)
        : exact_byte_shadow(exact_byte_capacity, access_capacity) {}

    uint64_t generation = 0;
    uint32_t workgroup_x = 0;
    uint32_t workgroup_y = 0;
    uint32_t workgroup_z = 0;
    std::vector<uint32_t> owner_epochs;
    ConSanMoiSparseExactByteShadow exact_byte_shadow;
    std::vector<uint64_t> exported_exact_shadow_entries;
    std::vector<ConSanMoiAtomicReleaseRecord> atomic_release_records;
    std::vector<ConSanMoiAcquiredEpochToken> acquired_epoch_tokens;
    std::set<ReportedDiagnosticKey> reported_diagnostics;
    bool in_barrier_run = false;
  };
  std::vector<ReplayWorkgroupState> workgroups;
  std::optional<size_t> first_workgroup_index;
  auto find_workgroup_state = [&](uint64_t generation, uint32_t workgroup_x, uint32_t workgroup_y,
                                  uint32_t workgroup_z) -> ReplayWorkgroupState & {
    for (ReplayWorkgroupState &state : workgroups) {
      if (state.generation == generation && state.workgroup_x == workgroup_x &&
          state.workgroup_y == workgroup_y && state.workgroup_z == workgroup_z)
        return state;
    }

    ReplayWorkgroupState state(static_cast<uint64_t>(exact_shadow_entries.size()) *
                                   consan_moi_exact_shadow::granule_bytes,
                               access_count);
    state.generation = generation;
    state.workgroup_x = workgroup_x;
    state.workgroup_y = workgroup_y;
    state.workgroup_z = workgroup_z;
    state.owner_epochs.resize(consan_moi_exact_shadow::max_owner + 1u);
    if (!first_workgroup_index)
      state.exported_exact_shadow_entries.resize(exact_shadow_entries.size());
    state.atomic_release_records.resize(atomic_events.size());
    state.acquired_epoch_tokens.resize(atomic_events.size() + fence_events.size());
    workgroups.push_back(std::move(state));
    if (!first_workgroup_index)
      first_workgroup_index = workgroups.size() - 1u;
    return workgroups.back();
  };

  const auto make_access_conflict_diagnostic = [](const ConSanMoiExactByteAccess &first,
                                                  const ConSanMoiExactByteAccess &second) {
    ConSanMoiDiagnosticRecord diagnostic;
    diagnostic.kind = static_cast<uint32_t>(ConSanMoiDiagnosticKind::AccessConflict);
    diagnostic.backend = static_cast<uint32_t>(ConSanMoiEngine::RecordReplay);
    diagnostic.generation = second.generation;
    diagnostic.epoch = second.epoch;
    diagnostic.first_owner_id = first.owner_id;
    diagnostic.second_owner_id = second.owner_id;
    diagnostic.first_lane_mask = first.lane_mask;
    diagnostic.second_lane_mask = second.lane_mask;
    diagnostic.first_instruction_offset = first.instruction_offset;
    diagnostic.second_instruction_offset = second.instruction_offset;
    diagnostic.first_lds_byte_offset = first.lds_byte_offset;
    diagnostic.first_lds_byte_count = first.lds_byte_count;
    diagnostic.second_lds_byte_offset = second.lds_byte_offset;
    diagnostic.second_lds_byte_count = second.lds_byte_count;
    diagnostic.first_access_kind = static_cast<uint32_t>(first.kind);
    diagnostic.second_access_kind = static_cast<uint32_t>(second.kind);
    return diagnostic;
  };

  const auto make_metadata_full_diagnostic = [](const ConSanMoiExactByteAccess &access) {
    ConSanMoiDiagnosticRecord diagnostic;
    diagnostic.kind = static_cast<uint32_t>(ConSanMoiDiagnosticKind::MetadataFull);
    diagnostic.backend = static_cast<uint32_t>(ConSanMoiEngine::RecordReplay);
    diagnostic.generation = access.generation;
    diagnostic.epoch = access.epoch;
    diagnostic.second_owner_id = access.owner_id;
    diagnostic.second_lane_mask = access.lane_mask;
    diagnostic.second_instruction_offset = access.instruction_offset;
    diagnostic.second_lds_byte_offset = access.lds_byte_offset;
    diagnostic.second_lds_byte_count = access.lds_byte_count;
    diagnostic.second_access_kind = static_cast<uint32_t>(access.kind);
    return diagnostic;
  };

  const auto make_intra_wave_write_diagnostic =
      [&](const ConSanMoiExactByteAccess &access) -> std::optional<ConSanMoiDiagnosticRecord> {
    if (!consan_moi_shadow_kind_conflicts(access.kind, access.kind) ||
        std::popcount(access.lane_mask) < 2) {
      return std::nullopt;
    }
    ConSanMoiExactByteAccess first = access;
    ConSanMoiExactByteAccess second = access;
    first.lane_mask = access.lane_mask & (~access.lane_mask + uint64_t{1});
    second.lane_mask = access.lane_mask ^ first.lane_mask;
    return make_access_conflict_diagnostic(first, second);
  };

  const auto append_diagnostic = [&](ReplayWorkgroupState &state,
                                     const ConSanMoiDiagnosticRecord &diagnostic,
                                     uint32_t event_index) {
    const uint32_t first_site =
        std::min(diagnostic.first_instruction_offset, diagnostic.second_instruction_offset);
    const uint32_t second_site =
        std::max(diagnostic.first_instruction_offset, diagnostic.second_instruction_offset);
    const uint64_t first_end =
        static_cast<uint64_t>(diagnostic.first_lds_byte_offset) + diagnostic.first_lds_byte_count;
    const uint64_t second_end =
        static_cast<uint64_t>(diagnostic.second_lds_byte_offset) + diagnostic.second_lds_byte_count;
    const uint64_t overlap_begin =
        std::max(diagnostic.first_lds_byte_offset, diagnostic.second_lds_byte_offset);
    const uint64_t overlap_end = std::min(first_end, second_end);
    const ReplayWorkgroupState::ReportedDiagnosticKey key{diagnostic.kind, first_site, second_site,
                                                          overlap_begin, overlap_end};
    if (!state.reported_diagnostics.insert(key).second)
      return;
    if (header.diagnostic_count < diagnostic_capacity) {
      ConSanMoiDiagnosticRecord published = diagnostic;
      published.reserved = event_index;
      diagnostic_records[header.diagnostic_count] = published;
      ++header.diagnostic_count;
      ++replay.emitted_diagnostic_count;
    } else {
      replay.diagnostic_capacity_exhausted = true;
    }
  };

  struct ReplayEvent {
    enum class Kind {
      Access,
      Barrier,
      Atomic,
      Fence,
    };

    uint32_t event_index = 0;
    uint32_t input_order = 0;
    uint32_t record_index = 0;
    Kind kind = Kind::Access;
  };
  std::vector<ReplayEvent> events;
  events.reserve(static_cast<size_t>(access_count) + barrier_count + atomic_events.size() +
                 fence_events.size());
  for (uint32_t i = 0; i < access_count; ++i)
    events.push_back({access_records[i].event_index, i, i, ReplayEvent::Kind::Access});
  for (uint32_t i = 0; i < barrier_count; ++i)
    events.push_back(
        {barrier_records[i].event_index, access_count + i, i, ReplayEvent::Kind::Barrier});
  for (uint32_t i = 0; i < span_size_u32(atomic_events.size()); ++i) {
    if (is_unpublished_atomic_record(atomic_events[i]))
      continue;
    events.push_back({atomic_events[i].event_index, access_count + barrier_count + i, i,
                      ReplayEvent::Kind::Atomic});
  }
  for (uint32_t i = 0; i < span_size_u32(fence_events.size()); ++i) {
    if (is_unpublished_fence_record(fence_events[i]))
      continue;
    events.push_back({fence_events[i].event_index,
                      access_count + barrier_count + span_size_u32(atomic_events.size()) + i, i,
                      ReplayEvent::Kind::Fence});
  }
  std::stable_sort(events.begin(), events.end(),
                   [](const ReplayEvent &lhs, const ReplayEvent &rhs) {
                     if (lhs.event_index != rhs.event_index)
                       return lhs.event_index < rhs.event_index;
                     return lhs.input_order < rhs.input_order;
                   });

  struct FenceRelease {
    uint64_t generation = 0;
    uint32_t workgroup_x = 0;
    uint32_t workgroup_y = 0;
    uint32_t workgroup_z = 0;
    uint32_t owner_id = 0;
    uint32_t epoch = 0;
    uint32_t scope = 0;
    uint32_t instruction_offset = 0;
    uint64_t communication_token = 0;
  };
  std::vector<FenceRelease> fence_releases;
  fence_releases.reserve(fence_events.size());

  auto publish_fence = [&](const ConSanMoiRecordReplayFenceEvent &record, uint64_t generation,
                           uint32_t epoch) {
    for (FenceRelease &release : fence_releases) {
      if (release.generation == generation && release.workgroup_x == record.workgroup_x &&
          release.workgroup_y == record.workgroup_y && release.workgroup_z == record.workgroup_z &&
          release.owner_id == record.owner_id && release.scope == record.scope &&
          release.communication_token == record.communication_token) {
        release.epoch = std::max(release.epoch, epoch);
        release.instruction_offset = record.instruction_offset;
        return;
      }
    }
    fence_releases.push_back({generation, record.workgroup_x, record.workgroup_y,
                              record.workgroup_z, record.owner_id, epoch, record.scope,
                              record.instruction_offset, record.communication_token});
  };

  for (const ReplayEvent &event : events) {
    if (event.kind == ReplayEvent::Kind::Barrier) {
      const ConSanMoiBarrierRecord &record = barrier_records[event.record_index];
      const uint64_t generation = record.generation != 0 ? record.generation : header.generation;
      ReplayWorkgroupState &state = find_workgroup_state(generation, record.workgroup_x,
                                                         record.workgroup_y, record.workgroup_z);
      ++replay.processed_barrier_count;
      if (!state.in_barrier_run) {
        for (uint32_t &epoch : state.owner_epochs) {
          if (epoch < consan_moi_exact_shadow::max_epoch)
            ++epoch;
        }
        state.in_barrier_run = true;
      }
      continue;
    }

    if (event.kind == ReplayEvent::Kind::Fence) {
      const ConSanMoiRecordReplayFenceEvent &record = fence_events[event.record_index];
      ++replay.processed_fence_count;
      const uint64_t generation = record.generation != 0 ? record.generation : header.generation;
      ReplayWorkgroupState &state = find_workgroup_state(generation, record.workgroup_x,
                                                         record.workgroup_y, record.workgroup_z);
      state.in_barrier_run = false;
      if (record.owner_id > consan_moi_exact_shadow::max_owner || record.scope == 0 ||
          record.scope > 3 || record.communication_token == 0) {
        ++replay.unsupported_fence_count;
        continue;
      }
      const bool acquire = record.kind == ConSanMoiFenceEventKind::Acquire ||
                           record.kind == ConSanMoiFenceEventKind::AcquireRelease;
      const bool release = record.kind == ConSanMoiFenceEventKind::Release ||
                           record.kind == ConSanMoiFenceEventKind::AcquireRelease;
      if (!acquire && !release) {
        ++replay.unsupported_fence_count;
        continue;
      }
      const uint32_t epoch = record.epoch != 0 ? record.epoch : state.owner_epochs[record.owner_id];
      if (acquire) {
        for (const FenceRelease &prior : fence_releases) {
          // The current shadow tracks workgroup-local LDS. Even device/system
          // fences cannot order an LDS owner in a different workgroup.
          if (prior.generation != generation || prior.workgroup_x != record.workgroup_x ||
              prior.workgroup_y != record.workgroup_y || prior.workgroup_z != record.workgroup_z ||
              prior.owner_id == record.owner_id ||
              prior.communication_token != record.communication_token)
            continue;
          const uint32_t common_scope = std::min(prior.scope, record.scope);
          if (common_scope < 1)
            continue;
          const ConSanMoiAtomicSyncResult token_result = record_acquired_epoch_token(
              state.acquired_epoch_tokens, generation, record.owner_id, prior.owner_id,
              consan_moi_acquired_epoch_token_value(prior.epoch), record.instruction_offset);
          replay.metadata_full |= token_result.metadata_full;
        }
      }
      if (release)
        publish_fence(record, generation, epoch);
      continue;
    }

    if (event.kind == ReplayEvent::Kind::Atomic) {
      const ConSanMoiRecordReplayAtomicEvent &record = atomic_events[event.record_index];
      ++replay.processed_atomic_count;
      const uint64_t generation = record.generation != 0 ? record.generation : header.generation;
      ReplayWorkgroupState &state = find_workgroup_state(generation, record.workgroup_x,
                                                         record.workgroup_y, record.workgroup_z);
      state.in_barrier_run = false;
      if (record.owner_id > consan_moi_exact_shadow::max_owner) {
        ++replay.unsupported_atomic_count;
        replay.metadata_full = true;
        continue;
      }
      const std::optional<ConSanMoiAtomicOutcome> outcome = resolved_atomic_outcome(record);
      if (!outcome) {
        ++replay.unsupported_atomic_count;
        continue;
      }

      const uint32_t epoch = record.epoch != 0 ? record.epoch : state.owner_epochs[record.owner_id];
      ConSanMoiAtomicSyncResult atomic_result;
      switch (record.kind) {
      case ConSanMoiAtomicEventKind::Release:
        if (record.operation != ConSanMoiAtomicOperation::CompareExchange ||
            *outcome == ConSanMoiAtomicOutcome::Success)
          atomic_result = consan_moi_record_replay_atomic_release(
              state.atomic_release_records, generation, record.atomic_address, record.owner_id,
              epoch, record.instruction_offset);
        break;
      case ConSanMoiAtomicEventKind::Acquire:
        atomic_result = consan_moi_record_replay_atomic_acquire(
            state.atomic_release_records, state.acquired_epoch_tokens, generation,
            record.atomic_address, record.owner_id, record.instruction_offset);
        break;
      case ConSanMoiAtomicEventKind::AcquireRelease: {
        // RMW and CAS both consume the prior value. Failed CAS does not
        // publish the release half of an acquire-release event.
        atomic_result = consan_moi_record_replay_atomic_acquire(
            state.atomic_release_records, state.acquired_epoch_tokens, generation,
            record.atomic_address, record.owner_id, record.instruction_offset);
        if (record.operation != ConSanMoiAtomicOperation::CompareExchange ||
            *outcome == ConSanMoiAtomicOutcome::Success) {
          const ConSanMoiAtomicSyncResult release_result = consan_moi_record_replay_atomic_release(
              state.atomic_release_records, generation, record.atomic_address, record.owner_id,
              epoch, record.instruction_offset);
          atomic_result.metadata_full |= release_result.metadata_full;
        }
        break;
      }
      }
      replay.metadata_full |= atomic_result.metadata_full;
      continue;
    }

    const uint32_t i = event.record_index;
    const ConSanMoiAccessRecord &record = access_records[i];
    // Fixed per-site Record/Replay buffers publish their static capacity up
    // front. A never-executed site therefore remains an all-zero slot; it is
    // neither a malformed access nor evidence loss. Any partially initialized
    // Empty record still falls through to the fail-closed unsupported path.
    if (is_unpublished_access(record))
      continue;
    const std::optional<ConSanMoiShadowAccessKind> access_kind =
        decode_access_kind(record.access_kind);
    if (!access_kind) {
      ++replay.processed_access_count;
      ++replay.unsupported_access_count;
      continue;
    }
    if ((record.flags & ~kConSanMoiAccessRecordKnownFlags) != 0u) {
      ++replay.processed_access_count;
      ++replay.unsupported_access_count;
      replay.metadata_full = true;
      continue;
    }
    if (record.wave_id > consan_moi_exact_shadow::max_owner) {
      ++replay.processed_access_count;
      ++replay.unsupported_access_count;
      replay.metadata_full = true;
      continue;
    }

    const uint64_t lds_byte_offset_u64 =
        record.lds_byte_count != 0
            ? record.lds_byte_offset
            : static_cast<uint64_t>(record.start_cell) * consan_moi_exact_shadow::granule_bytes;
    const uint64_t lds_byte_count_u64 =
        record.lds_byte_count != 0
            ? record.lds_byte_count
            : static_cast<uint64_t>(record.cell_count) * consan_moi_exact_shadow::granule_bytes;
    if (lds_byte_offset_u64 > std::numeric_limits<uint32_t>::max() ||
        lds_byte_count_u64 > std::numeric_limits<uint32_t>::max()) {
      ++replay.processed_access_count;
      ++replay.unsupported_access_count;
      replay.metadata_full = true;
      replay.conflict = true;
      continue;
    }
    const uint32_t lds_byte_offset = static_cast<uint32_t>(lds_byte_offset_u64);
    const uint32_t lds_byte_count = static_cast<uint32_t>(lds_byte_count_u64);

    const uint64_t generation = record.generation != 0 ? record.generation : header.generation;
    ReplayWorkgroupState &state = find_workgroup_state(generation, record.workgroup_x,
                                                       record.workgroup_y, record.workgroup_z);
    state.in_barrier_run = false;
    const ConSanMoiExactByteAccess access{
        generation,
        /*owner_id=*/record.wave_id,
        record.epoch != 0 ? record.epoch : state.owner_epochs[record.wave_id],
        *access_kind,
        lds_byte_offset,
        lds_byte_count,
        record.instruction_offset,
        record.lane_mask,
        // Unflagged lane masks summarize a wave access. They do not identify
        // separable address groups that could race with one another.
        (record.flags & kConSanMoiAccessRecordFlagExactAddressGroupMask) != 0u,
    };
    const ConSanMoiExactByteAccessResult access_result =
        state.exact_byte_shadow.access(access, state.acquired_epoch_tokens);
    ++replay.processed_access_count;
    if (access_result.capacity_exhausted) {
      replay.metadata_full = true;
      replay.conflict = true;
      append_diagnostic(state, make_metadata_full_diagnostic(access), record.event_index);
      continue;
    }
    if (access_result.conflict) {
      if (!access_result.prior) {
        replay.metadata_full = true;
        replay.conflict = true;
        append_diagnostic(state, make_metadata_full_diagnostic(access), record.event_index);
        continue;
      }
      replay.conflict = true;
      append_diagnostic(state, make_access_conflict_diagnostic(*access_result.prior, access),
                        record.event_index);
      continue;
    }
    const uint64_t packed = pack_consan_moi_exact_shadow_entry(
        access.kind, access.owner_id, access.epoch, static_cast<uint32_t>(access.generation),
        access.instruction_offset);
    const uint64_t byte_end = static_cast<uint64_t>(access.lds_byte_offset) + access.lds_byte_count;
    const uint64_t start_cell = access.lds_byte_offset >> consan_moi_exact_shadow::granule_shift;
    const uint64_t end_cell = (byte_end + consan_moi_exact_shadow::granule_bytes - 1u) >>
                              consan_moi_exact_shadow::granule_shift;
    for (uint64_t cell = start_cell;
         cell < end_cell && cell < state.exported_exact_shadow_entries.size(); ++cell) {
      state.exported_exact_shadow_entries[cell] = packed;
    }
    const std::optional<ConSanMoiDiagnosticRecord> intra_wave =
        (record.flags & kConSanMoiAccessRecordFlagExactAddressGroupMask) != 0u
            ? make_intra_wave_write_diagnostic(access)
            : std::nullopt;
    if (!intra_wave)
      continue;
    replay.conflict = true;
    append_diagnostic(state, *intra_wave, record.event_index);
  }
  if (first_workgroup_index)
    std::copy(workgroups[*first_workgroup_index].exported_exact_shadow_entries.begin(),
              workgroups[*first_workgroup_index].exported_exact_shadow_entries.end(),
              exact_shadow_entries.begin());
  return replay;
}

ConSanMoiRecordReplayWindowResult consan_moi_replay_record_replay_capture(
    const ConSanMoiRecordReplayTraceHeader &header,
    std::span<const ConSanMoiRecordReplayPcEntry> dictionary,
    std::span<const ConSanMoiRecordReplayWorkgroupRun> workgroup_runs,
    std::span<const ConSanMoiRecordReplayCompactEvent> events,
    std::span<const ConSanMoiRecordReplayCaptureWindow> windows,
    std::span<const uint32_t> event_window_indices,
    std::span<ConSanMoiDiagnosticRecord> diagnostic_records,
    std::span<uint64_t> exact_shadow_entries) {
  ConSanMoiRecordReplayWindowResult result;
  auto invalid = [&]() {
    result.invalid_capture = true;
    return result;
  };
  if (header.magic != kConSanMoiRecordReplayTraceMagic ||
      header.abi_version != kConSanMoiRecordReplayTraceAbiVersion ||
      header.header_size != kConSanMoiRecordReplayTraceHeaderBytes || header.flags != 0 ||
      header.dictionary_count > header.dictionary_capacity ||
      header.workgroup_run_count > header.workgroup_run_capacity ||
      header.event_count > header.event_capacity || header.dictionary_count > dictionary.size() ||
      header.workgroup_run_count > workgroup_runs.size() || header.event_count > events.size() ||
      header.event_count > event_window_indices.size())
    return invalid();

  // Do not trust caller-supplied window boundaries to establish completeness.
  // Reconstruct every barrier-delimited RR2 epoch with unbounded selection,
  // then require the requested windows and membership map to select whole
  // canonical epochs exactly. This still permits globally interleaved
  // workgroups, whose epoch events need not occupy contiguous positions.
  std::vector<ConSanMoiRecordReplayCaptureWindow> canonical_windows(header.event_count);
  std::vector<uint32_t> canonical_membership(header.event_count);
  const ConSanMoiRecordReplayCaptureResult canonical = consan_moi_plan_record_replay_capture(
      header, dictionary, workgroup_runs, events,
      {/*workgroup_limit=*/std::numeric_limits<uint32_t>::max(),
       /*epochs_per_workgroup_limit=*/std::numeric_limits<uint32_t>::max(),
       /*event_budget=*/std::numeric_limits<uint32_t>::max()},
      canonical_windows, canonical_membership);
  if (canonical.invalid_trace || canonical.selected_event_count != header.event_count)
    return invalid();
  canonical_windows.resize(canonical.selected_window_count);
  auto same_window = [](const ConSanMoiRecordReplayCaptureWindow &lhs,
                        const ConSanMoiRecordReplayCaptureWindow &rhs) {
    return lhs.workgroup_x == rhs.workgroup_x && lhs.workgroup_y == rhs.workgroup_y &&
           lhs.workgroup_z == rhs.workgroup_z && lhs.epoch == rhs.epoch &&
           lhs.first_event_position == rhs.first_event_position &&
           lhs.last_event_position == rhs.last_event_position &&
           lhs.first_event_index == rhs.first_event_index &&
           lhs.last_event_index == rhs.last_event_index && lhs.event_count == rhs.event_count &&
           lhs.reserved == rhs.reserved;
  };
  std::vector<uint32_t> canonical_to_requested(canonical_windows.size(),
                                               kConSanMoiRecordReplayUncapturedEvent);
  for (uint32_t requested_index = 0; requested_index < windows.size(); ++requested_index) {
    uint32_t canonical_index = kConSanMoiRecordReplayUncapturedEvent;
    for (uint32_t i = 0; i < canonical_windows.size(); ++i) {
      if (same_window(windows[requested_index], canonical_windows[i])) {
        canonical_index = i;
        break;
      }
    }
    if (canonical_index == kConSanMoiRecordReplayUncapturedEvent ||
        canonical_to_requested[canonical_index] != kConSanMoiRecordReplayUncapturedEvent)
      return invalid();
    canonical_to_requested[canonical_index] = requested_index;
  }
  for (uint32_t position = 0; position < header.event_count; ++position) {
    const uint32_t canonical_index = canonical_membership[position];
    if (canonical_index >= canonical_to_requested.size() ||
        event_window_indices[position] != canonical_to_requested[canonical_index])
      return invalid();
  }

  struct WorkgroupCoordinates {
    uint32_t x = 0;
    uint32_t y = 0;
    uint32_t z = 0;
  };
  std::vector<WorkgroupCoordinates> coordinates(header.event_count);
  uint32_t expected_position = 0;
  for (uint32_t i = 0; i < header.workgroup_run_count; ++i) {
    const ConSanMoiRecordReplayWorkgroupRun &run = workgroup_runs[i];
    if (run.event_count == 0 || run.first_event != expected_position ||
        run.event_count > header.event_count - expected_position)
      return invalid();
    for (uint32_t position = run.first_event; position < run.first_event + run.event_count;
         ++position)
      coordinates[position] = {run.workgroup_x, run.workgroup_y, run.workgroup_z};
    expected_position += run.event_count;
  }
  if (expected_position != header.event_count)
    return invalid();

  struct EpochState {
    struct BarrierRun {
      uint32_t pc_index = 0;
      std::vector<uint16_t> owners;
    };
    WorkgroupCoordinates coordinates;
    uint32_t epoch = 0;
    bool in_barrier_run = false;
    uint32_t barrier_pc_index = 0;
    std::vector<uint16_t> known_owners;
    std::vector<BarrierRun> barrier_runs;
  };
  std::vector<EpochState> epoch_states;
  std::vector<uint32_t> derived_epochs(header.event_count);
  for (uint32_t position = 0; position < header.event_count; ++position) {
    const ConSanMoiRecordReplayCompactEvent &event = events[position];
    if (event.pc_index >= header.dictionary_count)
      return invalid();
    const ConSanMoiRecordReplayEventKind kind = dictionary[event.pc_index].kind;
    if (kind != ConSanMoiRecordReplayEventKind::Access &&
        kind != ConSanMoiRecordReplayEventKind::Barrier &&
        kind != ConSanMoiRecordReplayEventKind::Atomic &&
        kind != ConSanMoiRecordReplayEventKind::Fence)
      return invalid();
    EpochState *state = nullptr;
    for (EpochState &candidate : epoch_states) {
      if (candidate.coordinates.x == coordinates[position].x &&
          candidate.coordinates.y == coordinates[position].y &&
          candidate.coordinates.z == coordinates[position].z) {
        state = &candidate;
        break;
      }
    }
    if (!state) {
      EpochState initial;
      initial.coordinates = coordinates[position];
      epoch_states.push_back(std::move(initial));
      state = &epoch_states.back();
    }
    if (std::ranges::find(state->known_owners, event.owner_id) == state->known_owners.end())
      state->known_owners.push_back(event.owner_id);
    const bool barrier = kind == ConSanMoiRecordReplayEventKind::Barrier;
    if (state->in_barrier_run && (!barrier || event.pc_index != state->barrier_pc_index)) {
      if (state->epoch == std::numeric_limits<uint32_t>::max())
        return invalid();
      ++state->epoch;
      state->in_barrier_run = false;
    }
    derived_epochs[position] = state->epoch;
    if (barrier) {
      if (event.lane_mask == 0)
        return invalid();
      if (!state->in_barrier_run) {
        state->barrier_runs.push_back({event.pc_index, {}});
        state->in_barrier_run = true;
        state->barrier_pc_index = event.pc_index;
      }
      std::vector<uint16_t> &participants = state->barrier_runs.back().owners;
      if (std::ranges::find(participants, event.owner_id) != participants.end())
        return invalid();
      participants.push_back(event.owner_id);
      state->in_barrier_run = true;
    }
  }
  // A barrier advances an epoch only when every owner observed in that
  // workgroup has exactly one arrival at that static barrier. Otherwise a
  // partial capture could erase a real race by advancing absent owners.
  for (const EpochState &state : epoch_states) {
    for (const EpochState::BarrierRun &run : state.barrier_runs) {
      if (run.owners.size() != state.known_owners.size())
        return invalid();
      for (uint16_t owner : state.known_owners) {
        if (std::ranges::find(run.owners, owner) == run.owners.end())
          return invalid();
      }
    }
  }

  struct ObservedWindow {
    uint32_t count = 0;
    uint32_t first_position = 0;
    uint32_t last_position = 0;
    uint32_t first_event_index = 0;
    uint32_t last_event_index = 0;
  };
  std::vector<ObservedWindow> observed(windows.size());
  std::vector<ConSanMoiAccessRecord> accesses;
  std::vector<ConSanMoiBarrierRecord> barriers;
  std::vector<ConSanMoiRecordReplayAtomicEvent> atomics;
  std::vector<ConSanMoiRecordReplayFenceEvent> fences;
  struct FenceAssociation {
    WorkgroupCoordinates coordinates;
    uint32_t owner_id = 0;
    uint32_t scope = 0;
    uint64_t token = 0;
  };
  std::vector<FenceAssociation> fence_associations;
  uint32_t wave_scope_atomic_count = 0;
  accesses.reserve(header.event_count);
  barriers.reserve(header.event_count);
  atomics.reserve(header.event_count);
  fences.reserve(header.event_count);

  for (uint32_t position = 0; position < header.event_count; ++position) {
    const uint32_t window_index = event_window_indices[position];
    if (window_index == kConSanMoiRecordReplayUncapturedEvent)
      continue;
    if (window_index >= windows.size())
      return invalid();
    const ConSanMoiRecordReplayCaptureWindow &window = windows[window_index];
    const ConSanMoiRecordReplayCompactEvent &event = events[position];
    if (event.pc_index >= header.dictionary_count ||
        coordinates[position].x != window.workgroup_x ||
        coordinates[position].y != window.workgroup_y ||
        coordinates[position].z != window.workgroup_z || position < window.first_event_position ||
        position > window.last_event_position || event.event_index < window.first_event_index ||
        event.event_index > window.last_event_index || derived_epochs[position] != window.epoch)
      return invalid();
    ObservedWindow &seen = observed[window_index];
    if (seen.count == 0) {
      seen.first_position = position;
      seen.first_event_index = event.event_index;
    }
    seen.last_position = position;
    seen.last_event_index = event.event_index;
    ++seen.count;
    ++result.selected_event_count;

    const ConSanMoiRecordReplayPcEntry &pc = dictionary[event.pc_index];
    switch (pc.kind) {
    case ConSanMoiRecordReplayEventKind::Access: {
      const auto access_kind = static_cast<ConSanMoiShadowAccessKind>(pc.operation);
      if ((access_kind != ConSanMoiShadowAccessKind::Read &&
           access_kind != ConSanMoiShadowAccessKind::Write &&
           access_kind != ConSanMoiShadowAccessKind::ReadWrite &&
           access_kind != ConSanMoiShadowAccessKind::Atomic) ||
          pc.scope != 0 || pc.semantics != 0)
        return invalid();
      ConSanMoiAccessRecord record;
      record.generation = header.generation;
      record.workgroup_x = coordinates[position].x;
      record.workgroup_y = coordinates[position].y;
      record.workgroup_z = coordinates[position].z;
      record.wave_id = event.owner_id;
      record.lane_mask = event.lane_mask;
      record.instruction_offset = pc.instruction_offset;
      record.access_kind = pc.operation;
      record.lds_byte_offset = static_cast<uint32_t>(event.payload);
      record.lds_byte_count = static_cast<uint32_t>(event.payload >> 32u);
      record.epoch = event.epoch;
      record.event_index = event.event_index;
      accesses.push_back(record);
      break;
    }
    case ConSanMoiRecordReplayEventKind::Barrier: {
      if (pc.operation != 0 || pc.scope != 0 || pc.semantics != 0 || event.payload != 0)
        return invalid();
      ConSanMoiBarrierRecord record;
      record.generation = header.generation;
      record.workgroup_x = coordinates[position].x;
      record.workgroup_y = coordinates[position].y;
      record.workgroup_z = coordinates[position].z;
      record.wave_id = event.owner_id;
      record.lane_mask = event.lane_mask;
      record.instruction_offset = pc.instruction_offset;
      record.event_index = event.event_index;
      barriers.push_back(record);
      break;
    }
    case ConSanMoiRecordReplayEventKind::Atomic: {
      const auto operation = static_cast<ConSanMoiAtomicEventKind>(pc.operation);
      if ((operation != ConSanMoiAtomicEventKind::Release &&
           operation != ConSanMoiAtomicEventKind::Acquire &&
           operation != ConSanMoiAtomicEventKind::AcquireRelease) ||
          (pc.atomic_operation != ConSanMoiAtomicOperation::Rmw &&
           pc.atomic_operation != ConSanMoiAtomicOperation::CompareExchange) ||
          (pc.atomic_operation == ConSanMoiAtomicOperation::Rmw &&
           event.atomic_outcome != ConSanMoiAtomicOutcome::NotApplicable) ||
          pc.scope > 3u || event.lane_mask != 0)
        return invalid();
      ConSanMoiRecordReplayAtomicEvent record;
      record.generation = header.generation;
      record.workgroup_x = coordinates[position].x;
      record.workgroup_y = coordinates[position].y;
      record.workgroup_z = coordinates[position].z;
      record.owner_id = event.owner_id;
      record.atomic_address = event.payload;
      record.instruction_offset = pc.instruction_offset;
      record.event_index = event.event_index;
      record.epoch = event.epoch;
      record.kind = operation;
      record.scope = pc.scope;
      record.semantics = pc.semantics;
      record.operation = pc.atomic_operation;
      record.outcome = event.atomic_outcome;
      if (pc.scope == 0)
        ++wave_scope_atomic_count;
      else
        atomics.push_back(record);
      break;
    }
    case ConSanMoiRecordReplayEventKind::Fence: {
      const auto operation = static_cast<ConSanMoiFenceEventKind>(pc.operation);
      if ((operation != ConSanMoiFenceEventKind::Release &&
           operation != ConSanMoiFenceEventKind::Acquire &&
           operation != ConSanMoiFenceEventKind::AcquireRelease) ||
          pc.scope == 0 || pc.scope > 3u || pc.atomic_operation != ConSanMoiAtomicOperation::Rmw ||
          event.payload == 0 || event.lane_mask != 0 ||
          event.atomic_outcome != ConSanMoiAtomicOutcome::NotApplicable ||
          event.owner_id > consan_moi_exact_shadow::max_owner)
        return invalid();
      ConSanMoiRecordReplayFenceEvent record;
      record.generation = header.generation;
      record.workgroup_x = coordinates[position].x;
      record.workgroup_y = coordinates[position].y;
      record.workgroup_z = coordinates[position].z;
      record.owner_id = event.owner_id;
      record.instruction_offset = pc.instruction_offset;
      record.event_index = event.event_index;
      record.epoch = event.epoch;
      record.kind = operation;
      record.scope = pc.scope;
      record.semantics = pc.semantics;
      record.communication_token = event.payload;
      const bool acquire = operation == ConSanMoiFenceEventKind::Acquire ||
                           operation == ConSanMoiFenceEventKind::AcquireRelease;
      const bool release = operation == ConSanMoiFenceEventKind::Release ||
                           operation == ConSanMoiFenceEventKind::AcquireRelease;
      if (acquire) {
        const bool matched = std::ranges::any_of(fence_associations, [&](const auto &prior) {
          return prior.coordinates.x == coordinates[position].x &&
                 prior.coordinates.y == coordinates[position].y &&
                 prior.coordinates.z == coordinates[position].z &&
                 prior.owner_id != event.owner_id && prior.token == event.payload &&
                 std::min(prior.scope, pc.scope) >= 1u;
        });
        if (!matched)
          return invalid();
      }
      if (release) {
        fence_associations.push_back(
            {coordinates[position], event.owner_id, pc.scope, event.payload});
      }
      fences.push_back(record);
      break;
    }
    default:
      return invalid();
    }
  }

  // A selected workgroup/epoch is indivisible. Reject membership that omits
  // any event from a selected epoch even if its surrounding metadata was also
  // shortened to match the partial subset.
  for (uint32_t position = 0; position < header.event_count; ++position) {
    if (event_window_indices[position] != kConSanMoiRecordReplayUncapturedEvent)
      continue;
    for (const ConSanMoiRecordReplayCaptureWindow &window : windows) {
      if (coordinates[position].x == window.workgroup_x &&
          coordinates[position].y == window.workgroup_y &&
          coordinates[position].z == window.workgroup_z && derived_epochs[position] == window.epoch)
        return invalid();
    }
  }

  for (uint32_t i = 0; i < windows.size(); ++i) {
    const ConSanMoiRecordReplayCaptureWindow &window = windows[i];
    const ObservedWindow &seen = observed[i];
    if (window.event_count == 0 || seen.count != window.event_count ||
        seen.first_position != window.first_event_position ||
        seen.last_position != window.last_event_position ||
        seen.first_event_index != window.first_event_index ||
        seen.last_event_index != window.last_event_index)
      return invalid();
  }

  ConSanMoiReportHeader replay_header = make_consan_moi_report_header(
      header.generation, header.dispatch_id, consan_moi_clamp_u32_capacity(accesses.size()),
      consan_moi_clamp_u32_capacity(diagnostic_records.size()),
      consan_moi_clamp_u32_capacity(exact_shadow_entries.size()), 0,
      consan_moi_clamp_u32_capacity(barriers.size()),
      consan_moi_clamp_u32_capacity(atomics.size()));
  replay_header.access_record_count = replay_header.access_record_capacity;
  replay_header.barrier_record_count = replay_header.barrier_record_capacity;
  replay_header.atomic_record_count = replay_header.atomic_record_capacity;
  result.replay = consan_moi_record_replay_access_records(
      replay_header, accesses, barriers, atomics, fences, diagnostic_records, exact_shadow_entries);
  result.replay.processed_atomic_count += wave_scope_atomic_count;
  return result;
}

ConSanMoiSampledPublishResult
consan_moi_sampled_publish_access_records(const ConSanMoiReportHeader &header,
                                          std::span<const ConSanMoiAccessRecord> access_records,
                                          std::span<uint64_t> sampled_watchpoint_entries) {
  auto span_size_u32 = [](size_t size) {
    return size > std::numeric_limits<uint32_t>::max() ? std::numeric_limits<uint32_t>::max()
                                                       : static_cast<uint32_t>(size);
  };
  auto decode_access_kind = [](uint32_t value) {
    switch (static_cast<ConSanMoiShadowAccessKind>(value)) {
    case ConSanMoiShadowAccessKind::Read:
    case ConSanMoiShadowAccessKind::Write:
    case ConSanMoiShadowAccessKind::ReadWrite:
    case ConSanMoiShadowAccessKind::Atomic:
      return static_cast<ConSanMoiShadowAccessKind>(value);
    case ConSanMoiShadowAccessKind::Empty:
      return ConSanMoiShadowAccessKind::Empty;
    }
    return ConSanMoiShadowAccessKind::Empty;
  };

  ConSanMoiSampledPublishResult publish;
  const uint32_t access_count = std::min({header.access_record_count, header.access_record_capacity,
                                          span_size_u32(access_records.size())});
  const uint32_t sampled_capacity = std::min(header.sampled_watchpoint_capacity,
                                             span_size_u32(sampled_watchpoint_entries.size()));

  for (uint32_t i = 0; i < access_count; ++i) {
    ++publish.processed_access_count;
    if (publish.published_entry_count >= sampled_capacity) {
      publish.sampled_capacity_exhausted = true;
      continue;
    }

    const ConSanMoiAccessRecord &record = access_records[i];
    ConSanMoiLdsCellRange range{record.start_cell, record.cell_count};
    if (range.cell_count == 0 && record.lds_byte_count != 0)
      range = consan_moi_lds_cell_range_for_bytes(record.lds_byte_offset, record.lds_byte_count);
    if (range.cell_count == 0)
      continue;

    sampled_watchpoint_entries[publish.published_entry_count] =
        pack_consan_moi_sampled_watchpoint_entry(
            decode_access_kind(record.access_kind), record.wave_id, record.epoch,
            static_cast<uint32_t>(record.generation != 0 ? record.generation : header.generation),
            range.start_cell, range.cell_count);
    ++publish.published_entry_count;
  }
  return publish;
}

ConSanMoiSampledPublishResult
consan_moi_sampled_publish_causal_windows(const ConSanMoiReportHeader &header,
                                          std::span<const ConSanMoiAccessRecord> access_records,
                                          uint32_t selection_stride, uint32_t selection_offset,
                                          std::span<uint64_t> sampled_watchpoint_entries,
                                          std::span<ConSanMoiSampledCausalWindow> causal_windows) {
  struct WindowKey {
    uint64_t generation = 0;
    uint32_t x = 0;
    uint32_t y = 0;
    uint32_t z = 0;
    uint32_t epoch = 0;

    bool operator==(const WindowKey &) const = default;
  };
  struct WindowKeyHash {
    size_t operator()(const WindowKey &key) const {
      uint64_t hash = consan_moi_sampled_causal_mix(0x243f6a8885a308d3ull, key.generation);
      hash = consan_moi_sampled_causal_mix(hash, key.x);
      hash = consan_moi_sampled_causal_mix(hash, key.y);
      hash = consan_moi_sampled_causal_mix(hash, key.z);
      return static_cast<size_t>(consan_moi_sampled_causal_mix(hash, key.epoch));
    }
  };
  struct Item {
    ConSanMoiShadowAccessKind kind = ConSanMoiShadowAccessKind::Empty;
    uint32_t owner = 0;
    ConSanMoiLdsCellRange range;
  };
  struct Window {
    WindowKey key;
    std::vector<Item> items;
    bool malformed = false;
  };
  const auto span_size_u32 = [](size_t size) {
    return size > std::numeric_limits<uint32_t>::max() ? std::numeric_limits<uint32_t>::max()
                                                       : static_cast<uint32_t>(size);
  };
  const auto decode_kind = [](uint32_t value) {
    const auto kind = static_cast<ConSanMoiShadowAccessKind>(value);
    return kind == ConSanMoiShadowAccessKind::Read || kind == ConSanMoiShadowAccessKind::Write
               ? kind
               : ConSanMoiShadowAccessKind::Empty;
  };

  ConSanMoiSampledPublishResult publish;
  if (selection_stride == 0 || selection_stride > 1024u ||
      (selection_stride & (selection_stride - 1u)) != 0 || selection_offset >= selection_stride) {
    publish.invalid_selection = true;
    return publish;
  }

  const uint32_t access_count = std::min({header.access_record_count, header.access_record_capacity,
                                          span_size_u32(access_records.size())});
  const uint32_t entry_capacity = std::min(header.sampled_watchpoint_capacity,
                                           span_size_u32(sampled_watchpoint_entries.size()));
  const uint32_t window_capacity = span_size_u32(causal_windows.size());
  std::vector<Window> windows;
  std::unordered_map<WindowKey, size_t, WindowKeyHash> by_key;
  for (uint32_t index = 0; index < access_count; ++index) {
    ++publish.processed_access_count;
    const ConSanMoiAccessRecord &record = access_records[index];
    const uint64_t generation = record.generation != 0 ? record.generation : header.generation;
    const WindowKey key{generation, record.workgroup_x, record.workgroup_y, record.workgroup_z,
                        record.epoch};
    auto [position, inserted] = by_key.emplace(key, windows.size());
    if (inserted)
      windows.push_back(Window{key, {}, false});
    Window &window = windows[position->second];
    ConSanMoiLdsCellRange range{record.start_cell, record.cell_count};
    if (range.cell_count == 0 && record.lds_byte_count != 0)
      range = consan_moi_lds_cell_range_for_bytes(record.lds_byte_offset, record.lds_byte_count);
    const ConSanMoiShadowAccessKind kind = decode_kind(record.access_kind);
    const bool malformed =
        generation != header.generation ||
        record.wave_id > consan_moi_sampled_watchpoint::max_owner ||
        record.epoch > consan_moi_sampled_watchpoint::max_epoch ||
        kind == ConSanMoiShadowAccessKind::Empty || range.cell_count == 0 ||
        range.cell_count > consan_moi_sampled_watchpoint::max_count ||
        range.start_cell > consan_moi_sampled_watchpoint::max_start ||
        range.cell_count > consan_moi_sampled_watchpoint::max_start + 1u - range.start_cell;
    window.malformed |= malformed;
    window.items.push_back(Item{kind, record.wave_id, range});
  }

  for (const Window &window : windows) {
    ++publish.eligible_window_count;
    if (!consan_moi_sampled_causal_window_selected(
            window.key.generation, header.dispatch_id, window.key.x, window.key.y, window.key.z,
            window.key.epoch, selection_stride, selection_offset)) {
      continue;
    }
    ++publish.selected_window_count;
    if (window.malformed) {
      ++publish.malformed_window_count;
      continue;
    }
    if (publish.published_window_count >= window_capacity) {
      publish.window_capacity_exhausted = true;
      continue;
    }
    if (window.items.size() > entry_capacity - publish.published_entry_count) {
      publish.sampled_capacity_exhausted = true;
      continue;
    }
    const uint32_t first_entry = publish.published_entry_count;
    for (const Item &item : window.items) {
      sampled_watchpoint_entries[publish.published_entry_count++] =
          pack_consan_moi_sampled_watchpoint_entry(item.kind, item.owner, window.key.epoch,
                                                   static_cast<uint32_t>(window.key.generation),
                                                   item.range.start_cell, item.range.cell_count);
    }
    causal_windows[publish.published_window_count++] = ConSanMoiSampledCausalWindow{
        window.key.generation,
        header.dispatch_id,
        window.key.x,
        window.key.y,
        window.key.z,
        window.key.epoch,
        first_entry,
        static_cast<uint32_t>(window.items.size()),
        static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Ready),
        0};
  }
  return publish;
}

ConSanMoiSampledReplayResult
consan_moi_sampled_replay_entries(ConSanMoiReportHeader &header,
                                  std::span<const uint64_t> sampled_watchpoint_entries,
                                  std::span<ConSanMoiDiagnosticRecord> diagnostic_records) {
  std::vector<ConSanMoiSampledSnapshotWords> snapshots;
  snapshots.reserve(sampled_watchpoint_entries.size());
  for (uint64_t packed : sampled_watchpoint_entries) {
    const uint32_t low = static_cast<uint32_t>(packed);
    snapshots.push_back({low, static_cast<uint32_t>(packed >> 32u), low});
  }
  return consan_moi_sampled_replay_snapshots(header, snapshots, diagnostic_records);
}

ConSanMoiSampledReplayResult
consan_moi_sampled_replay_snapshots(ConSanMoiReportHeader &header,
                                    std::span<const ConSanMoiSampledSnapshotWords> snapshots,
                                    std::span<ConSanMoiDiagnosticRecord> diagnostic_records) {
  auto span_size_u32 = [](size_t size) {
    return size > std::numeric_limits<uint32_t>::max() ? std::numeric_limits<uint32_t>::max()
                                                       : static_cast<uint32_t>(size);
  };

  ConSanMoiSampledReplayResult replay;
  const uint32_t entry_count =
      std::min(header.sampled_watchpoint_capacity, span_size_u32(snapshots.size()));
  const uint32_t diagnostic_capacity =
      std::min(header.diagnostic_capacity, span_size_u32(diagnostic_records.size()));
  const uint32_t active_generation =
      static_cast<uint32_t>(header.generation) & consan_moi_sampled_watchpoint::max_generation;

  std::vector<std::optional<ConSanMoiSampledWatchpointEntry>> stable_entries(entry_count);

  for (uint32_t i = 0; i < entry_count; ++i) {
    ++replay.processed_entry_count;
    const ConSanMoiSampledSnapshot snapshot =
        classify_consan_moi_sampled_snapshot(snapshots[i], active_generation);
    switch (snapshot.state) {
    case ConSanMoiSampledSnapshotState::Empty:
      ++replay.empty_entry_count;
      continue;
    case ConSanMoiSampledSnapshotState::StaleGeneration:
      ++replay.stale_generation_entry_count;
      continue;
    case ConSanMoiSampledSnapshotState::IncompletePublication:
      ++replay.incomplete_publication_entry_count;
      continue;
    case ConSanMoiSampledSnapshotState::ChangedDuringRead:
      ++replay.changed_during_read_entry_count;
      continue;
    case ConSanMoiSampledSnapshotState::Malformed:
      ++replay.malformed_entry_count;
      continue;
    case ConSanMoiSampledSnapshotState::Stable:
      break;
    }
    stable_entries[i] = snapshot.entry;
    const ConSanMoiSampledWatchpointEntry &current = *stable_entries[i];

    for (uint32_t prior_index = 0; prior_index < i; ++prior_index) {
      if (!stable_entries[prior_index] ||
          !consan_moi_sampled_watchpoints_conflict(current, *stable_entries[prior_index]))
        continue;

      replay.conflict = true;
      ConSanMoiDiagnosticRecord diagnostic;
      diagnostic.kind = static_cast<uint32_t>(ConSanMoiDiagnosticKind::AccessConflict);
      diagnostic.backend = static_cast<uint32_t>(ConSanMoiEngine::Sampled);
      diagnostic.generation = current.generation;
      diagnostic.epoch = current.epoch;
      diagnostic.first_owner_id = stable_entries[prior_index]->owner_id;
      diagnostic.second_owner_id = current.owner_id;
      diagnostic.first_access_kind = static_cast<uint32_t>(stable_entries[prior_index]->kind);
      diagnostic.second_access_kind = static_cast<uint32_t>(current.kind);
      if (header.diagnostic_count < diagnostic_capacity) {
        diagnostic_records[header.diagnostic_count] = diagnostic;
        ++header.diagnostic_count;
        ++replay.emitted_diagnostic_count;
      } else {
        replay.diagnostic_capacity_exhausted = true;
      }
      return replay;
    }
  }
  return replay;
}

ConSanMoiSampledReplayResult consan_moi_sampled_replay_causal_windows(
    ConSanMoiReportHeader &header, std::span<const uint64_t> sampled_watchpoint_entries,
    std::span<const ConSanMoiSampledCausalWindow> causal_windows,
    std::span<ConSanMoiDiagnosticRecord> diagnostic_records) {
  ConSanMoiSampledReplayResult replay;
  const uint32_t entry_capacity =
      std::min(header.sampled_watchpoint_capacity,
               sampled_watchpoint_entries.size() > std::numeric_limits<uint32_t>::max()
                   ? std::numeric_limits<uint32_t>::max()
                   : static_cast<uint32_t>(sampled_watchpoint_entries.size()));
  uint32_t expected_first = 0;
  std::vector<ConSanMoiSampledWatchpointEntry> decoded;
  decoded.reserve(entry_capacity);
  for (const ConSanMoiSampledCausalWindow &window : causal_windows) {
    if (window.publication_state !=
            static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Ready) ||
        window.generation != header.generation || window.dispatch_id != header.dispatch_id ||
        window.epoch > consan_moi_sampled_watchpoint::max_epoch || window.entry_count == 0 ||
        window.first_entry != expected_first ||
        window.entry_count > entry_capacity - expected_first) {
      replay.invalid_causal_metadata = true;
      return replay;
    }
    expected_first += window.entry_count;
  }
  for (uint32_t index = 0; index < expected_first; ++index) {
    const ConSanMoiSampledWatchpointEntry entry =
        decode_consan_moi_sampled_watchpoint_entry(sampled_watchpoint_entries[index]);
    if (!entry.valid || entry.consumed ||
        (entry.kind != ConSanMoiShadowAccessKind::Read &&
         entry.kind != ConSanMoiShadowAccessKind::Write) ||
        entry.generation != (static_cast<uint32_t>(header.generation) &
                             consan_moi_sampled_watchpoint::max_generation)) {
      replay.invalid_causal_metadata = true;
      return replay;
    }
    decoded.push_back(entry);
  }
  for (const ConSanMoiSampledCausalWindow &window : causal_windows) {
    for (uint32_t index = window.first_entry; index < window.first_entry + window.entry_count;
         ++index) {
      if (decoded[index].epoch != window.epoch) {
        replay.invalid_causal_metadata = true;
        return replay;
      }
    }
  }

  const uint32_t diagnostic_capacity = std::min(
      header.diagnostic_capacity, diagnostic_records.size() > std::numeric_limits<uint32_t>::max()
                                      ? std::numeric_limits<uint32_t>::max()
                                      : static_cast<uint32_t>(diagnostic_records.size()));
  for (const ConSanMoiSampledCausalWindow &window : causal_windows) {
    ++replay.processed_window_count;
    for (uint32_t current_index = window.first_entry;
         current_index < window.first_entry + window.entry_count; ++current_index) {
      ++replay.processed_entry_count;
      const ConSanMoiSampledWatchpointEntry &current = decoded[current_index];
      for (uint32_t prior_index = window.first_entry; prior_index < current_index; ++prior_index) {
        const ConSanMoiSampledWatchpointEntry &prior = decoded[prior_index];
        if (!consan_moi_sampled_watchpoints_conflict(current, prior))
          continue;
        replay.conflict = true;
        ConSanMoiDiagnosticRecord diagnostic;
        diagnostic.kind = static_cast<uint32_t>(ConSanMoiDiagnosticKind::AccessConflict);
        diagnostic.backend = static_cast<uint32_t>(ConSanMoiEngine::Sampled);
        diagnostic.generation = current.generation;
        diagnostic.epoch = current.epoch;
        diagnostic.first_owner_id = prior.owner_id;
        diagnostic.second_owner_id = current.owner_id;
        diagnostic.first_access_kind = static_cast<uint32_t>(prior.kind);
        diagnostic.second_access_kind = static_cast<uint32_t>(current.kind);
        if (header.diagnostic_count < diagnostic_capacity) {
          diagnostic_records[header.diagnostic_count++] = diagnostic;
          ++replay.emitted_diagnostic_count;
        } else {
          replay.diagnostic_capacity_exhausted = true;
        }
        return replay;
      }
    }
  }
  return replay;
}

ConSanMoiSampledClaimResult
consan_moi_sampled_begin_causal_claim(std::span<ConSanMoiSampledCausalWindow> windows,
                                      const ConSanMoiSampledCausalKey &key) {
  ConSanMoiSampledClaimResult result;
  if (windows.empty() || key.epoch > consan_moi_sampled_watchpoint::max_epoch)
    return result;

  uint64_t hash = consan_moi_sampled_causal_mix(0x243f6a8885a308d3ull, key.generation);
  hash = consan_moi_sampled_causal_mix(hash, key.dispatch_id);
  hash = consan_moi_sampled_causal_mix(hash, key.workgroup_x);
  hash = consan_moi_sampled_causal_mix(hash, key.workgroup_y);
  hash = consan_moi_sampled_causal_mix(hash, key.workgroup_z);
  hash = consan_moi_sampled_causal_mix(hash, key.epoch);
  hash = consan_moi_sampled_causal_mix(hash, key.cluster_workgroup_id);
  const uint32_t start = static_cast<uint32_t>(hash % windows.size());
  for (uint32_t probe = 0; probe < windows.size(); ++probe) {
    result.slot = (start + probe) % static_cast<uint32_t>(windows.size());
    ConSanMoiSampledCausalWindow &window = windows[result.slot];
    std::atomic_ref<uint32_t> state(window.publication_state);
    uint32_t observed = state.load(std::memory_order_acquire);
    if (observed == static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Publishing)) {
      result.outcome = ConSanMoiSampledClaimOutcome::Busy;
      return result;
    }
    if (observed == static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Ready)) {
      const ConSanMoiSampledCausalKey existing{
          window.generation,  window.dispatch_id, window.workgroup_x,         window.workgroup_y,
          window.workgroup_z, window.epoch,       window.cluster_workgroup_id};
      if (existing == key) {
        result.outcome = ConSanMoiSampledClaimOutcome::Existing;
        return result;
      }
      ++result.collision_count;
      continue;
    }
    if (observed != static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Empty)) {
      ++result.malformed_slot_count;
      continue;
    }
    if (!state.compare_exchange_strong(
            observed, static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Publishing),
            std::memory_order_acq_rel, std::memory_order_acquire)) {
      result.outcome = ConSanMoiSampledClaimOutcome::Busy;
      return result;
    }
    result.outcome = ConSanMoiSampledClaimOutcome::Claimed;
    return result;
  }
  result.outcome = ConSanMoiSampledClaimOutcome::CapacityExhausted;
  return result;
}

bool consan_moi_sampled_commit_causal_claim(ConSanMoiSampledCausalWindow &window,
                                            const ConSanMoiSampledCausalKey &key,
                                            uint32_t first_entry, uint32_t entry_count) {
  std::atomic_ref<uint32_t> state(window.publication_state);
  if (state.load(std::memory_order_acquire) !=
          static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Publishing) ||
      entry_count == 0)
    return false;
  window.generation = key.generation;
  window.dispatch_id = key.dispatch_id;
  window.workgroup_x = key.workgroup_x;
  window.workgroup_y = key.workgroup_y;
  window.workgroup_z = key.workgroup_z;
  window.epoch = key.epoch;
  window.first_entry = first_entry;
  window.entry_count = entry_count;
  window.cluster_workgroup_id = key.cluster_workgroup_id;
  state.store(static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Ready),
              std::memory_order_release);
  return true;
}

bool consan_moi_sampled_abort_causal_claim(ConSanMoiSampledCausalWindow &window) {
  std::atomic_ref<uint32_t> state(window.publication_state);
  uint32_t expected = static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Publishing);
  return state.compare_exchange_strong(
      expected, static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Malformed),
      std::memory_order_release, std::memory_order_relaxed);
}

std::string_view consan_moi_atomic_address_support_name(ConSanMoiAtomicAddressSupport support) {
  switch (support) {
  case ConSanMoiAtomicAddressSupport::Supported:
    return "supported";
  case ConSanMoiAtomicAddressSupport::UnsupportedArchitecture:
    return "unsupported-architecture";
  case ConSanMoiAtomicAddressSupport::UnsupportedAddressKind:
    return "unsupported-address-kind";
  case ConSanMoiAtomicAddressSupport::UnsupportedWidth:
    return "unsupported-width";
  case ConSanMoiAtomicAddressSupport::UnsupportedEncoding:
    return "unsupported-encoding";
  case ConSanMoiAtomicAddressSupport::MissingAddressOperands:
    return "missing-address-operands";
  case ConSanMoiAtomicAddressSupport::UnsupportedInputWidth:
    return "unsupported-input-width";
  case ConSanMoiAtomicAddressSupport::UnsupportedOffset:
    return "unsupported-offset";
  case ConSanMoiAtomicAddressSupport::UnsupportedScope:
    return "unsupported-scope";
  case ConSanMoiAtomicAddressSupport::UnsupportedResourcePlan:
    return "unsupported-resource-plan";
  case ConSanMoiAtomicAddressSupport::UnsupportedScratchShape:
    return "unsupported-scratch-shape";
  case ConSanMoiAtomicAddressSupport::ResultAddressAlias:
    return "result-address-alias";
  case ConSanMoiAtomicAddressSupport::ScratchOperandAlias:
    return "scratch-operand-alias";
  }
  return "unknown";
}

ConSanMoiAtomicAddressPlan
plan_consan_moi_atomic_address(const ConSanAtomicSite &site, uint16_t scratch_vgpr,
                               uint16_t scratch_vgpr_count,
                               ConSanRegisterAllocationSource resource_source, rj_code_arch_t arch,
                               bool allow_post_guest_spill_operand_overlap) {
  ConSanMoiAtomicAddressPlan plan;
  plan.scratch_vgpr = scratch_vgpr;
  plan.scratch_vgpr_count = scratch_vgpr_count;
  plan.resource_source = resource_source;
  const auto reject = [&](ConSanMoiAtomicAddressSupport support) {
    plan.kind = ConSanMoiAtomicAddressKind::Unsupported;
    plan.support = support;
    return plan;
  };
  const auto overlaps = [](uint16_t lhs_base, uint16_t lhs_count, uint16_t rhs_base,
                           uint16_t rhs_count) {
    return static_cast<uint32_t>(lhs_base) + lhs_count > rhs_base &&
           static_cast<uint32_t>(rhs_base) + rhs_count > lhs_base;
  };
  const auto usable_resource_source = [](ConSanRegisterAllocationSource source) {
    return source == ConSanRegisterAllocationSource::Explicit ||
           source == ConSanRegisterAllocationSource::LivenessDead ||
           source == ConSanRegisterAllocationSource::DescriptorGrowth ||
           source == ConSanRegisterAllocationSource::SpillRequired;
  };
  if (!instrumentation::is_admitted_arch(arch))
    return reject(ConSanMoiAtomicAddressSupport::UnsupportedArchitecture);
  if (!usable_resource_source(resource_source))
    return reject(ConSanMoiAtomicAddressSupport::UnsupportedResourcePlan);
  if (site.mnemonic.starts_with("ds_")) {
    if (arch != ROCJITSU_CODE_ARCH_GFX1250)
      return reject(ConSanMoiAtomicAddressSupport::UnsupportedArchitecture);
    if (site.width_bits != 32u)
      return reject(ConSanMoiAtomicAddressSupport::UnsupportedWidth);
    if (site.size != 2u * sizeof(uint32_t) || !site.raw_addr || !site.raw_data0 ||
        !site.raw_ioffset)
      return reject(ConSanMoiAtomicAddressSupport::UnsupportedEncoding);
    if (!site.raw_scope || *site.raw_scope != 1u)
      return reject(ConSanMoiAtomicAddressSupport::UnsupportedScope);
    if (!site.addr_vgpr || !site.data_vgpr || *site.raw_addr != *site.addr_vgpr ||
        *site.raw_data0 != *site.data_vgpr)
      return reject(ConSanMoiAtomicAddressSupport::MissingAddressOperands);
    if (*site.raw_ioffset < 0 || *site.raw_ioffset > 0xff)
      return reject(ConSanMoiAtomicAddressSupport::UnsupportedOffset);
    if ((scratch_vgpr_count != 5u && scratch_vgpr_count < 7u) ||
        static_cast<uint32_t>(scratch_vgpr) + scratch_vgpr_count > 256u)
      return reject(ConSanMoiAtomicAddressSupport::UnsupportedScratchShape);
    const uint16_t result_address_vgpr =
        static_cast<uint16_t>(scratch_vgpr + scratch_vgpr_count - 2u);
    if (overlaps(result_address_vgpr, 2u, *site.addr_vgpr, 1u))
      return reject(ConSanMoiAtomicAddressSupport::ResultAddressAlias);
    if (!allow_post_guest_spill_operand_overlap &&
        (overlaps(scratch_vgpr, scratch_vgpr_count - 2u, *site.addr_vgpr, 1u) ||
         overlaps(scratch_vgpr, scratch_vgpr_count, *site.data_vgpr, 1u) ||
         (site.dst_vgpr && overlaps(scratch_vgpr, scratch_vgpr_count, *site.dst_vgpr, 1u))))
      return reject(ConSanMoiAtomicAddressSupport::ScratchOperandAlias);
    plan.kind = ConSanMoiAtomicAddressKind::LdsByteOffsetToken;
    plan.support = ConSanMoiAtomicAddressSupport::Supported;
    plan.input_address_vgpr = *site.addr_vgpr;
    plan.input_address_vgpr_count = 1u;
    plan.signed_byte_offset = *site.raw_ioffset;
    plan.result_address_vgpr = result_address_vgpr;
    plan.result_address_vgpr_count = 2u;
    return plan;
  }
  if (site.mnemonic.starts_with("buffer_")) {
    constexpr uint32_t kNullScalarOffset = 0x7cu;
    constexpr int32_t kSigned24Min = -(1 << 23);
    constexpr int32_t kSigned24Max = (1 << 23) - 1;
    if (arch != ROCJITSU_CODE_ARCH_GFX1250)
      return reject(ConSanMoiAtomicAddressSupport::UnsupportedArchitecture);
    if (site.width_bits == 0u || site.width_bits > 128u)
      return reject(ConSanMoiAtomicAddressSupport::UnsupportedWidth);
    if (site.size != 3u * sizeof(uint32_t) || !site.raw_rsrc || !site.raw_soffset ||
        !site.raw_vaddr || !site.raw_ioffset || !site.raw_scope || !site.raw_offen ||
        !site.raw_idxen || !*site.raw_offen || *site.raw_idxen)
      return reject(ConSanMoiAtomicAddressSupport::UnsupportedEncoding);
    if (!site.addr_vgpr || !site.saddr_sgpr || !site.data_vgpr ||
        *site.raw_vaddr != *site.addr_vgpr || *site.raw_rsrc != *site.saddr_sgpr)
      return reject(ConSanMoiAtomicAddressSupport::MissingAddressOperands);
    if ((*site.saddr_sgpr & 3u) != 0u || *site.saddr_sgpr > 124u ||
        (*site.raw_soffset != kNullScalarOffset && *site.raw_soffset > 127u) ||
        *site.addr_vgpr > 255u)
      return reject(ConSanMoiAtomicAddressSupport::UnsupportedInputWidth);
    if (*site.raw_ioffset < kSigned24Min || *site.raw_ioffset > kSigned24Max)
      return reject(ConSanMoiAtomicAddressSupport::UnsupportedOffset);
    if (*site.raw_scope < 1u || *site.raw_scope > 3u)
      return reject(ConSanMoiAtomicAddressSupport::UnsupportedScope);
    if ((scratch_vgpr_count != 5u && scratch_vgpr_count < 7u) ||
        static_cast<uint32_t>(scratch_vgpr) + scratch_vgpr_count > 256u)
      return reject(ConSanMoiAtomicAddressSupport::UnsupportedScratchShape);
    const uint16_t result_address_vgpr =
        static_cast<uint16_t>(scratch_vgpr + scratch_vgpr_count - 2u);
    if (!allow_post_guest_spill_operand_overlap &&
        (overlaps(scratch_vgpr, scratch_vgpr_count, *site.data_vgpr, 1u) ||
         (site.dst_vgpr && overlaps(scratch_vgpr, scratch_vgpr_count, *site.dst_vgpr, 1u))))
      return reject(ConSanMoiAtomicAddressSupport::ScratchOperandAlias);
    plan.kind = ConSanMoiAtomicAddressKind::BufferResourceMaterialized;
    plan.support = ConSanMoiAtomicAddressSupport::Supported;
    plan.input_address_vgpr = *site.addr_vgpr;
    plan.input_address_vgpr_count = 1u;
    plan.scalar_base_sgpr = *site.saddr_sgpr;
    if (*site.raw_soffset != kNullScalarOffset)
      plan.scalar_offset_sgpr = static_cast<uint16_t>(*site.raw_soffset);
    plan.signed_byte_offset = *site.raw_ioffset;
    plan.result_address_vgpr = result_address_vgpr;
    plan.result_address_vgpr_count = 2u;
    return plan;
  }
  // Record/Replay only needs the communication operation's effective address
  // and, for CAS, its dynamic success mask. Global/flat atomics and the
  // ordinary load/store side of a qualified fence sequence use the same
  // 64-bit address forms. Wider atomic values merely occupy more consecutive
  // guest VGPRs. Other widths still need an explicit operand-layout proof.
  if (site.width_bits != 32u && site.width_bits != 64u)
    return reject(ConSanMoiAtomicAddressSupport::UnsupportedWidth);
  const bool cdna_flat_encoding =
      arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4;
  const uint32_t expected_size = cdna_flat_encoding ? 2u * sizeof(uint32_t) : 3u * sizeof(uint32_t);
  if (site.size != expected_size || !site.raw_saddr || !site.raw_vaddr || !site.raw_ioffset)
    return reject(ConSanMoiAtomicAddressSupport::UnsupportedEncoding);
  // Scope does not affect effective-address reconstruction. Wave scope is not
  // an inter-wave synchronization event, but workgroup, agent, and system
  // atomics all have the same address operands and are valid record targets.
  if (!site.raw_scope || *site.raw_scope < 1u || *site.raw_scope > 3u)
    return reject(ConSanMoiAtomicAddressSupport::UnsupportedScope);
  if (!site.addr_vgpr || !site.data_vgpr || *site.raw_vaddr != *site.addr_vgpr)
    return reject(ConSanMoiAtomicAddressSupport::MissingAddressOperands);

  const uint32_t flat_no_saddr =
      cdna_flat_encoding
          ? (site.mnemonic.starts_with("global_atomic") ? kCdnaGlobalNoSaddrEncoding : 0u)
          : flat_no_saddr_encoding(arch);
  constexpr int32_t kSigned13Min = -(1 << 12);
  constexpr int32_t kSigned13Max = (1 << 12) - 1;
  constexpr int32_t kSigned24Min = -(1 << 23);
  constexpr int32_t kSigned24Max = (1 << 23) - 1;
  const int32_t signed_vglobal_offset_min = cdna_flat_encoding ? kSigned13Min : kSigned24Min;
  const int32_t signed_vglobal_offset_max = cdna_flat_encoding ? kSigned13Max : kSigned24Max;
  const bool is_compare_exchange = consan_atomic_is_compare_exchange(site);
  const uint16_t value_word_count = static_cast<uint16_t>(site.width_bits / 32u);
  const uint16_t data_count =
      static_cast<uint16_t>(value_word_count * (is_compare_exchange ? 2u : 1u));
  const uint16_t destination_count = site.returns_old_value.value_or(false) ? value_word_count : 0u;
  const bool returned_value_aliases_address =
      site.returns_old_value.value_or(false) && site.dst_vgpr &&
      overlaps(*site.addr_vgpr, 2u, *site.dst_vgpr, destination_count);

  if (site.mnemonic.starts_with("flat_")) {
    if (*site.raw_saddr != flat_no_saddr)
      return reject(ConSanMoiAtomicAddressSupport::UnsupportedEncoding);
    if (*site.addr_vgpr >= 255u)
      return reject(ConSanMoiAtomicAddressSupport::UnsupportedInputWidth);
    if (*site.raw_ioffset != 0)
      return reject(ConSanMoiAtomicAddressSupport::UnsupportedOffset);
    const uint16_t minimum_scratch_count = returned_value_aliases_address ? 5u : 3u;
    if ((scratch_vgpr_count != minimum_scratch_count && scratch_vgpr_count < 7u) ||
        static_cast<uint32_t>(scratch_vgpr) + scratch_vgpr_count > 256u)
      return reject(ConSanMoiAtomicAddressSupport::UnsupportedScratchShape);
    if (!allow_post_guest_spill_operand_overlap &&
        (overlaps(scratch_vgpr, scratch_vgpr_count, *site.addr_vgpr, 2u) ||
         overlaps(scratch_vgpr, scratch_vgpr_count, *site.data_vgpr, data_count) ||
         (site.dst_vgpr &&
          overlaps(scratch_vgpr, scratch_vgpr_count, *site.dst_vgpr, destination_count))))
      return reject(ConSanMoiAtomicAddressSupport::ScratchOperandAlias);
    plan.kind = returned_value_aliases_address
                    ? ConSanMoiAtomicAddressKind::FlatGuestPairMaterialized
                    : ConSanMoiAtomicAddressKind::FlatGuestPair;
    plan.support = ConSanMoiAtomicAddressSupport::Supported;
    plan.input_address_vgpr = *site.addr_vgpr;
    plan.input_address_vgpr_count = 2u;
    plan.signed_byte_offset = 0;
    plan.result_address_vgpr = returned_value_aliases_address
                                   ? static_cast<uint16_t>(scratch_vgpr + scratch_vgpr_count - 2u)
                                   : *site.addr_vgpr;
    plan.result_address_vgpr_count = 2u;
    return plan;
  }

  if (!site.mnemonic.starts_with("global_"))
    return reject(ConSanMoiAtomicAddressSupport::UnsupportedAddressKind);
  if (*site.raw_saddr == flat_no_saddr) {
    if (*site.addr_vgpr >= 255u)
      return reject(ConSanMoiAtomicAddressSupport::UnsupportedInputWidth);
    if (*site.raw_ioffset < signed_vglobal_offset_min ||
        *site.raw_ioffset > signed_vglobal_offset_max)
      return reject(ConSanMoiAtomicAddressSupport::UnsupportedOffset);
    const bool requires_materialization = *site.raw_ioffset != 0 || returned_value_aliases_address;
    const uint16_t minimum_scratch_count = requires_materialization ? 5u : 3u;
    if ((scratch_vgpr_count != minimum_scratch_count && scratch_vgpr_count < 7u) ||
        static_cast<uint32_t>(scratch_vgpr) + scratch_vgpr_count > 256u)
      return reject(ConSanMoiAtomicAddressSupport::UnsupportedScratchShape);
    if (!requires_materialization) {
      if (!allow_post_guest_spill_operand_overlap &&
          (overlaps(scratch_vgpr, scratch_vgpr_count, *site.addr_vgpr, 2u) ||
           overlaps(scratch_vgpr, scratch_vgpr_count, *site.data_vgpr, data_count) ||
           (site.dst_vgpr &&
            overlaps(scratch_vgpr, scratch_vgpr_count, *site.dst_vgpr, destination_count))))
        return reject(ConSanMoiAtomicAddressSupport::ScratchOperandAlias);
      plan.kind = ConSanMoiAtomicAddressKind::VglobalGuestPair;
      plan.support = ConSanMoiAtomicAddressSupport::Supported;
      plan.input_address_vgpr = *site.addr_vgpr;
      plan.input_address_vgpr_count = 2u;
      plan.signed_byte_offset = 0;
      plan.result_address_vgpr = *site.addr_vgpr;
      plan.result_address_vgpr_count = 2u;
      return plan;
    }
    const uint16_t result_address_vgpr =
        static_cast<uint16_t>(scratch_vgpr + scratch_vgpr_count - 2u);
    if (overlaps(result_address_vgpr, 2u, *site.addr_vgpr, 2u))
      return reject(ConSanMoiAtomicAddressSupport::ResultAddressAlias);
    if (!allow_post_guest_spill_operand_overlap &&
        (overlaps(scratch_vgpr, scratch_vgpr_count - 2u, *site.addr_vgpr, 2u) ||
         overlaps(scratch_vgpr, scratch_vgpr_count, *site.data_vgpr, data_count) ||
         (site.dst_vgpr &&
          overlaps(scratch_vgpr, scratch_vgpr_count, *site.dst_vgpr, destination_count))))
      return reject(ConSanMoiAtomicAddressSupport::ScratchOperandAlias);
    plan.kind = ConSanMoiAtomicAddressKind::VglobalGuestPairMaterialized;
    plan.support = ConSanMoiAtomicAddressSupport::Supported;
    plan.input_address_vgpr = *site.addr_vgpr;
    plan.input_address_vgpr_count = 2u;
    plan.signed_byte_offset = *site.raw_ioffset;
    plan.result_address_vgpr = result_address_vgpr;
    plan.result_address_vgpr_count = 2u;
    return plan;
  }
  if (!site.saddr_sgpr || *site.raw_saddr != *site.saddr_sgpr)
    return reject(ConSanMoiAtomicAddressSupport::UnsupportedInputWidth);
  if (*site.saddr_sgpr > 104u || (*site.saddr_sgpr & 1u) != 0u)
    return reject(ConSanMoiAtomicAddressSupport::UnsupportedEncoding);
  if (*site.raw_ioffset < signed_vglobal_offset_min ||
      *site.raw_ioffset > signed_vglobal_offset_max)
    return reject(ConSanMoiAtomicAddressSupport::UnsupportedOffset);
  if (*site.addr_vgpr > 255u)
    return reject(ConSanMoiAtomicAddressSupport::UnsupportedInputWidth);
  if ((scratch_vgpr_count != 5u && scratch_vgpr_count < 7u) ||
      static_cast<uint32_t>(scratch_vgpr) + scratch_vgpr_count > 256u)
    return reject(ConSanMoiAtomicAddressSupport::UnsupportedScratchShape);

  const uint16_t result_address_vgpr =
      static_cast<uint16_t>(scratch_vgpr + scratch_vgpr_count - 2u);
  if (overlaps(result_address_vgpr, 2u, *site.addr_vgpr, 1u))
    return reject(ConSanMoiAtomicAddressSupport::ResultAddressAlias);
  if (!allow_post_guest_spill_operand_overlap &&
      (overlaps(scratch_vgpr, scratch_vgpr_count, *site.data_vgpr, data_count) ||
       (site.dst_vgpr &&
        overlaps(scratch_vgpr, scratch_vgpr_count, *site.dst_vgpr, destination_count)) ||
       overlaps(scratch_vgpr, scratch_vgpr_count - 2u, *site.addr_vgpr, 1u)))
    return reject(ConSanMoiAtomicAddressSupport::ScratchOperandAlias);

  plan.kind = ConSanMoiAtomicAddressKind::VglobalMaterialized;
  plan.support = ConSanMoiAtomicAddressSupport::Supported;
  plan.input_address_vgpr = *site.addr_vgpr;
  plan.input_address_vgpr_count = 1u;
  plan.scalar_base_sgpr = *site.saddr_sgpr;
  plan.signed_byte_offset = *site.raw_ioffset;
  plan.result_address_vgpr = result_address_vgpr;
  plan.result_address_vgpr_count = 2u;
  return plan;
}

std::optional<std::vector<uint32_t>>
build_consan_moi_atomic_address_materialization(const ConSanMoiAtomicAddressPlan &plan,
                                                uint16_t vcc_save_sgpr, uint16_t scc_save_sgpr,
                                                rj_code_arch_t arch) {
  if (!plan.supported())
    return std::nullopt;
  if (!plan.requires_materialization())
    return std::vector<uint32_t>{};
  // Every current recipe returns its materialized address inside the declared
  // scratch allocation. Validate that shared layout before per-kind emission.
  const uint32_t scratch_end = static_cast<uint32_t>(plan.scratch_vgpr) + plan.scratch_vgpr_count;
  const uint32_t result_end =
      static_cast<uint32_t>(plan.result_address_vgpr) + plan.result_address_vgpr_count;
  if (scratch_end > 256u || plan.result_address_vgpr < plan.scratch_vgpr ||
      result_end > scratch_end)
    return std::nullopt;
  if (plan.kind == ConSanMoiAtomicAddressKind::LdsByteOffsetToken) {
    if (arch != ROCJITSU_CODE_ARCH_GFX1250 || plan.input_address_vgpr_count != 1u ||
        plan.result_address_vgpr_count != 2u || plan.result_address_vgpr >= 255u ||
        plan.signed_byte_offset < 0 || plan.signed_byte_offset > 0xff)
      return std::nullopt;
    std::vector<uint32_t> words;
    words.push_back(build_v_mov_b32_e32(plan.result_address_vgpr,
                                        vector_source_vgpr(plan.input_address_vgpr), arch));
    if (plan.signed_byte_offset != 0) {
      const auto add = instrumentation::build_v_add_u32_literal(
          plan.result_address_vgpr, static_cast<uint32_t>(plan.signed_byte_offset),
          plan.result_address_vgpr, arch);
      if (!add)
        return std::nullopt;
      words.insert(words.end(), add->begin(), add->end());
    }
    const auto tag = build_v_mov_b32_e64_literal(
        static_cast<uint16_t>(plan.result_address_vgpr + 1u), kConSanMoiLdsAddressTokenTag, arch);
    if (!tag)
      return std::nullopt;
    words.insert(words.end(), tag->begin(), tag->end());
    return words;
  }
  const bool cdna_address_materialization =
      instrumentation::is_cdna_family_arch(arch) &&
      (plan.kind == ConSanMoiAtomicAddressKind::FlatGuestPairMaterialized ||
       plan.kind == ConSanMoiAtomicAddressKind::VglobalGuestPairMaterialized ||
       plan.kind == ConSanMoiAtomicAddressKind::VglobalMaterialized);
  if (!is_rdna4_family_arch(arch) && !cdna_address_materialization)
    return std::nullopt;
  const bool buffer_resource = plan.kind == ConSanMoiAtomicAddressKind::BufferResourceMaterialized;
  const bool scalar_vector =
      plan.kind == ConSanMoiAtomicAddressKind::VglobalMaterialized || buffer_resource;
  const bool vector_pair = plan.kind == ConSanMoiAtomicAddressKind::FlatGuestPairMaterialized ||
                           plan.kind == ConSanMoiAtomicAddressKind::VglobalGuestPairMaterialized;
  if ((!scalar_vector && !vector_pair) ||
      plan.input_address_vgpr_count != (scalar_vector ? 1u : 2u) ||
      plan.result_address_vgpr_count != 2u || plan.result_address_vgpr >= 255u ||
      vcc_save_sgpr >= 105u || scc_save_sgpr >= 106u || scc_save_sgpr == vcc_save_sgpr ||
      scc_save_sgpr == vcc_save_sgpr + 1u ||
      (scalar_vector &&
       (!plan.scalar_base_sgpr ||
        (*plan.scalar_base_sgpr >= vcc_save_sgpr && *plan.scalar_base_sgpr <= vcc_save_sgpr + 1u) ||
        (*plan.scalar_base_sgpr + 1u >= vcc_save_sgpr &&
         *plan.scalar_base_sgpr + 1u <= vcc_save_sgpr + 1u) ||
        scc_save_sgpr == *plan.scalar_base_sgpr || scc_save_sgpr == *plan.scalar_base_sgpr + 1u)) ||
      (buffer_resource && plan.scalar_offset_sgpr &&
       ((*plan.scalar_offset_sgpr >= vcc_save_sgpr &&
         *plan.scalar_offset_sgpr <= vcc_save_sgpr + 1u) ||
        scc_save_sgpr == *plan.scalar_offset_sgpr)))
    return std::nullopt;

  constexpr uint16_t kVccLo = 106u;
  const auto save_scc = instrumentation::build_s_cselect_b32(
      scc_save_sgpr, scalar_positive_inline_u32(1), scalar_positive_inline_u32(0), arch);
  const auto save_vcc = instrumentation::build_s_mov_b64(vcc_save_sgpr, kVccLo, arch);
  const auto restore_vcc = instrumentation::build_s_mov_b64(kVccLo, vcc_save_sgpr, arch);
  const auto restore_scc =
      instrumentation::build_s_cmp_lg_u32(scc_save_sgpr, scalar_positive_inline_u32(0), arch);
  if (!save_scc || !save_vcc || !restore_vcc || !restore_scc)
    return std::nullopt;

  std::vector<uint32_t> words;
  words.reserve(16u);
  words.push_back(*save_scc);
  words.push_back(*save_vcc);
  if (scalar_vector) {
    uint16_t offset_vgpr = plan.input_address_vgpr;
    if (buffer_resource && plan.input_address_vgpr >= plan.result_address_vgpr &&
        plan.input_address_vgpr < plan.result_address_vgpr + 2u) {
      offset_vgpr = plan.scratch_vgpr;
      if (offset_vgpr == plan.input_address_vgpr || offset_vgpr >= plan.result_address_vgpr)
        return std::nullopt;
      words.push_back(
          build_v_mov_b32_e32(offset_vgpr, vector_source_vgpr(plan.input_address_vgpr), arch));
    }
    words.push_back(build_v_mov_b32_e32(plan.result_address_vgpr, *plan.scalar_base_sgpr, arch));
    words.push_back(build_v_mov_b32_e32(static_cast<uint16_t>(plan.result_address_vgpr + 1u),
                                        static_cast<uint16_t>(*plan.scalar_base_sgpr + 1u), arch));
    std::optional<std::vector<uint32_t>> add_vaddr;
    if (instrumentation::is_cdna_family_arch(arch) &&
        plan.kind == ConSanMoiAtomicAddressKind::VglobalMaterialized) {
      // The sign scratch needs two scratch words before the result pair. Keep
      // a defensive check for externally constructed plans.
      if (plan.result_address_vgpr <= plan.scratch_vgpr + 1u)
        return std::nullopt;
      uint16_t sign_vgpr = plan.scratch_vgpr;
      if (sign_vgpr == offset_vgpr)
        ++sign_vgpr;
      // The spacing check and shared window invariant keep this temporary
      // inside the allocation and below the result pair.
      add_vaddr = instrumentation::build_v_add_u64_signed_vgpr_offset(plan.result_address_vgpr,
                                                                      offset_vgpr, sign_vgpr, arch);
    } else {
      add_vaddr =
          instrumentation::build_v_add_u64_vgpr_offset(plan.result_address_vgpr, offset_vgpr, arch);
    }
    if (!add_vaddr)
      return std::nullopt;
    words.insert(words.end(), add_vaddr->begin(), add_vaddr->end());
    if (buffer_resource && plan.scalar_offset_sgpr) {
      const uint16_t scalar_offset_vgpr = plan.scratch_vgpr;
      if (scalar_offset_vgpr == plan.result_address_vgpr ||
          scalar_offset_vgpr == plan.result_address_vgpr + 1u)
        return std::nullopt;
      words.push_back(build_v_mov_b32_e32(scalar_offset_vgpr, *plan.scalar_offset_sgpr, arch));
      const auto add_scalar_offset = instrumentation::build_v_add_u64_vgpr_offset(
          plan.result_address_vgpr, scalar_offset_vgpr, arch);
      if (!add_scalar_offset)
        return std::nullopt;
      words.insert(words.end(), add_scalar_offset->begin(), add_scalar_offset->end());
    }
  } else {
    words.push_back(build_v_mov_b32_e32(plan.result_address_vgpr,
                                        vector_source_vgpr(plan.input_address_vgpr), arch));
    words.push_back(build_v_mov_b32_e32(
        static_cast<uint16_t>(plan.result_address_vgpr + 1u),
        vector_source_vgpr(static_cast<uint16_t>(plan.input_address_vgpr + 1u)), arch));
  }
  if (plan.signed_byte_offset != 0) {
    const auto add_displacement = instrumentation::build_v_add_u64_signed_i24(
        plan.result_address_vgpr, plan.signed_byte_offset, arch);
    if (!add_displacement)
      return std::nullopt;
    words.insert(words.end(), add_displacement->begin(), add_displacement->end());
  }
  words.push_back(*restore_vcc);
  // Keep SCC restoration last: scalar comparisons after this point would
  // overwrite the guest condition code again.
  words.push_back(*restore_scc);
  return words;
}

} // namespace rocjitsu
