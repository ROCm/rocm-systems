// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan_moi.h"

#include "rocjitsu/analysis/def_use_chain.h"
#include "rocjitsu/analysis/kernel_scope.h"
#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/consan_resource.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "util/bit.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rocjitsu {

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

ConSanMoiRecordReplayAccessResult
consan_moi_record_replay_access(std::span<uint64_t> exact_shadow_entries,
                                const ConSanMoiRecordReplayAccess &access) {
  return consan_moi_record_replay_access(exact_shadow_entries, access,
                                         std::span<const ConSanMoiAcquiredEpochToken>{});
}

ConSanMoiRecordReplayAccessResult consan_moi_record_replay_access(
    std::span<uint64_t> exact_shadow_entries, const ConSanMoiRecordReplayAccess &access,
    std::span<const ConSanMoiAcquiredEpochToken> acquired_epoch_tokens) {
  ConSanMoiRecordReplayAccessResult result;
  if (access.cell_count == 0 || consan_moi_shadow_kind_is_empty(access.kind))
    return result;

  const uint32_t end_cell = consan_moi_range_end_cell(access.start_cell, access.cell_count);
  if (end_cell < access.start_cell || end_cell > exact_shadow_entries.size()) {
    result.metadata_full = true;
    result.conflict = true;
    result.diagnostic.kind = static_cast<uint32_t>(ConSanMoiDiagnosticKind::MetadataFull);
    result.diagnostic.backend = static_cast<uint32_t>(ConSanMoiEngine::RecordReplay);
    result.diagnostic.generation = access.generation;
    result.diagnostic.epoch = access.epoch;
    result.diagnostic.second_owner_id = access.owner_id;
    result.diagnostic.second_lane_mask = access.lane_mask;
    result.diagnostic.second_instruction_offset = access.instruction_offset;
    result.diagnostic.second_lds_byte_offset = access.lds_byte_offset;
    result.diagnostic.second_lds_byte_count = access.lds_byte_count;
    result.diagnostic.second_access_kind = static_cast<uint32_t>(access.kind);
    return result;
  }

  const ConSanMoiExactShadowEntry current{
      access.kind,
      access.owner_id,
      access.epoch,
      static_cast<uint32_t>(access.generation),
      access.instruction_offset,
  };
  for (uint32_t cell = access.start_cell; cell < end_cell; ++cell) {
    const ConSanMoiExactShadowEntry prior =
        decode_consan_moi_exact_shadow_entry(exact_shadow_entries[cell]);
    if (!consan_moi_exact_shadow_entries_conflict(current, prior))
      continue;
    if (consan_moi_acquired_epoch_orders(acquired_epoch_tokens, current, prior))
      continue;
    result.conflict = true;
    result.diagnostic.kind = static_cast<uint32_t>(ConSanMoiDiagnosticKind::AccessConflict);
    result.diagnostic.backend = static_cast<uint32_t>(ConSanMoiEngine::RecordReplay);
    result.diagnostic.generation = access.generation;
    result.diagnostic.epoch = access.epoch;
    result.diagnostic.first_owner_id = prior.owner_id;
    result.diagnostic.second_owner_id = access.owner_id;
    result.diagnostic.second_lane_mask = access.lane_mask;
    result.diagnostic.first_instruction_offset = prior.instruction_offset;
    result.diagnostic.second_instruction_offset = access.instruction_offset;
    result.diagnostic.second_lds_byte_offset = access.lds_byte_offset;
    result.diagnostic.second_lds_byte_count = access.lds_byte_count;
    result.diagnostic.first_access_kind = static_cast<uint32_t>(prior.kind);
    result.diagnostic.second_access_kind = static_cast<uint32_t>(access.kind);
    return result;
  }

  const uint64_t packed = pack_consan_moi_exact_shadow_entry(
      access.kind, access.owner_id, access.epoch, static_cast<uint32_t>(access.generation),
      access.instruction_offset);
  for (uint32_t cell = access.start_cell; cell < end_cell; ++cell)
    exact_shadow_entries[cell] = packed;
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
    uint32_t workgroup_x = 0;
    uint32_t workgroup_y = 0;
    uint32_t workgroup_z = 0;
    std::vector<uint32_t> owner_epochs;
    std::vector<uint64_t> exact_shadow_entries;
    std::vector<ConSanMoiAtomicReleaseRecord> atomic_release_records;
    std::vector<ConSanMoiAcquiredEpochToken> acquired_epoch_tokens;
    bool in_barrier_run = false;
  };
  std::vector<ReplayWorkgroupState> workgroups;
  std::optional<size_t> first_workgroup_index;
  auto find_workgroup_state = [&](uint32_t workgroup_x, uint32_t workgroup_y,
                                  uint32_t workgroup_z) -> ReplayWorkgroupState & {
    for (ReplayWorkgroupState &state : workgroups) {
      if (state.workgroup_x == workgroup_x && state.workgroup_y == workgroup_y &&
          state.workgroup_z == workgroup_z)
        return state;
    }

    ReplayWorkgroupState state;
    state.workgroup_x = workgroup_x;
    state.workgroup_y = workgroup_y;
    state.workgroup_z = workgroup_z;
    state.owner_epochs.resize(consan_moi_exact_shadow::max_owner + 1u);
    state.exact_shadow_entries.resize(exact_shadow_entries.size());
    state.atomic_release_records.resize(atomic_events.size());
    state.acquired_epoch_tokens.resize(atomic_events.size());
    workgroups.push_back(std::move(state));
    if (!first_workgroup_index)
      first_workgroup_index = workgroups.size() - 1u;
    return workgroups.back();
  };

  struct ReplayEvent {
    enum class Kind {
      Access,
      Barrier,
      Atomic,
    };

    uint32_t event_index = 0;
    uint32_t input_order = 0;
    uint32_t record_index = 0;
    Kind kind = Kind::Access;
  };
  std::vector<ReplayEvent> events;
  events.reserve(static_cast<size_t>(access_count) + barrier_count + atomic_events.size());
  for (uint32_t i = 0; i < access_count; ++i)
    events.push_back({access_records[i].event_index, i, i, ReplayEvent::Kind::Access});
  for (uint32_t i = 0; i < barrier_count; ++i)
    events.push_back(
        {barrier_records[i].event_index, access_count + i, i, ReplayEvent::Kind::Barrier});
  for (uint32_t i = 0; i < span_size_u32(atomic_events.size()); ++i)
    events.push_back({atomic_events[i].event_index, access_count + barrier_count + i, i,
                      ReplayEvent::Kind::Atomic});
  std::stable_sort(events.begin(), events.end(),
                   [](const ReplayEvent &lhs, const ReplayEvent &rhs) {
                     if (lhs.event_index != rhs.event_index)
                       return lhs.event_index < rhs.event_index;
                     return lhs.input_order < rhs.input_order;
                   });

  for (const ReplayEvent &event : events) {
    if (event.kind == ReplayEvent::Kind::Barrier) {
      const ConSanMoiBarrierRecord &record = barrier_records[event.record_index];
      ReplayWorkgroupState &state =
          find_workgroup_state(record.workgroup_x, record.workgroup_y, record.workgroup_z);
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

    if (event.kind == ReplayEvent::Kind::Atomic) {
      const ConSanMoiRecordReplayAtomicEvent &record = atomic_events[event.record_index];
      ++replay.processed_atomic_count;
      ReplayWorkgroupState &state =
          find_workgroup_state(record.workgroup_x, record.workgroup_y, record.workgroup_z);
      state.in_barrier_run = false;
      if (record.owner_id > consan_moi_exact_shadow::max_owner) {
        ++replay.unsupported_atomic_count;
        replay.metadata_full = true;
        continue;
      }

      const uint64_t generation = record.generation != 0 ? record.generation : header.generation;
      const uint32_t epoch = record.epoch != 0 ? record.epoch : state.owner_epochs[record.owner_id];
      ConSanMoiAtomicSyncResult atomic_result;
      switch (record.kind) {
      case ConSanMoiAtomicEventKind::Release:
        atomic_result = consan_moi_record_replay_atomic_release(
            state.atomic_release_records, generation, record.atomic_address, record.owner_id, epoch,
            record.instruction_offset);
        break;
      case ConSanMoiAtomicEventKind::Acquire:
        atomic_result = consan_moi_record_replay_atomic_acquire(
            state.atomic_release_records, state.acquired_epoch_tokens, generation,
            record.atomic_address, record.owner_id, record.instruction_offset);
        break;
      }
      replay.metadata_full |= atomic_result.metadata_full;
      continue;
    }

    const uint32_t i = event.record_index;
    const ConSanMoiAccessRecord &record = access_records[i];
    const std::optional<ConSanMoiShadowAccessKind> access_kind =
        decode_access_kind(record.access_kind);
    if (!access_kind) {
      ++replay.processed_access_count;
      ++replay.unsupported_access_count;
      continue;
    }
    ConSanMoiLdsCellRange range{record.start_cell, record.cell_count};
    if (range.cell_count == 0 && record.lds_byte_count != 0)
      range = consan_moi_lds_cell_range_for_bytes(record.lds_byte_offset, record.lds_byte_count);

    if (record.wave_id > consan_moi_exact_shadow::max_owner) {
      ++replay.processed_access_count;
      ++replay.unsupported_access_count;
      replay.metadata_full = true;
      continue;
    }

    ReplayWorkgroupState &state =
        find_workgroup_state(record.workgroup_x, record.workgroup_y, record.workgroup_z);
    state.in_barrier_run = false;
    const ConSanMoiRecordReplayAccess access{
        record.generation != 0 ? record.generation : header.generation,
        /*owner_id=*/record.wave_id,
        record.epoch != 0 ? record.epoch : state.owner_epochs[record.wave_id],
        *access_kind,
        record.lds_byte_offset,
        record.lds_byte_count,
        range.start_cell,
        range.cell_count,
        record.instruction_offset,
        record.lane_mask,
    };
    const ConSanMoiRecordReplayAccessResult access_result = consan_moi_record_replay_access(
        state.exact_shadow_entries, access, state.acquired_epoch_tokens);
    ++replay.processed_access_count;
    if (!access_result.conflict)
      continue;

    replay.conflict = true;
    replay.metadata_full |= access_result.metadata_full;
    if (header.diagnostic_count < diagnostic_capacity) {
      diagnostic_records[header.diagnostic_count] = access_result.diagnostic;
      ++header.diagnostic_count;
      ++replay.emitted_diagnostic_count;
    } else {
      replay.diagnostic_capacity_exhausted = true;
    }
  }
  if (first_workgroup_index)
    std::copy(workgroups[*first_workgroup_index].exact_shadow_entries.begin(),
              workgroups[*first_workgroup_index].exact_shadow_entries.end(),
              exact_shadow_entries.begin());
  return replay;
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

ConSanMoiSampledReplayResult
consan_moi_sampled_replay_entries(ConSanMoiReportHeader &header,
                                  std::span<const uint64_t> sampled_watchpoint_entries,
                                  std::span<ConSanMoiDiagnosticRecord> diagnostic_records) {
  auto span_size_u32 = [](size_t size) {
    return size > std::numeric_limits<uint32_t>::max() ? std::numeric_limits<uint32_t>::max()
                                                       : static_cast<uint32_t>(size);
  };

  ConSanMoiSampledReplayResult replay;
  const uint32_t entry_count = std::min(header.sampled_watchpoint_capacity,
                                        span_size_u32(sampled_watchpoint_entries.size()));
  const uint32_t diagnostic_capacity =
      std::min(header.diagnostic_capacity, span_size_u32(diagnostic_records.size()));

  for (uint32_t i = 0; i < entry_count; ++i) {
    const ConSanMoiSampledWatchpointEntry current =
        decode_consan_moi_sampled_watchpoint_entry(sampled_watchpoint_entries[i]);
    ++replay.processed_entry_count;
    if (!current.valid)
      continue;

    for (uint32_t prior_index = 0; prior_index < i; ++prior_index) {
      const ConSanMoiSampledWatchpointEntry prior =
          decode_consan_moi_sampled_watchpoint_entry(sampled_watchpoint_entries[prior_index]);
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

namespace {

inline constexpr uint16_t kRdna4ExecLo = 126;
inline constexpr uint16_t kRdna4ExecHi = 127;
inline constexpr uint16_t kRdna4VccLo = 106;
inline constexpr uint16_t kRdna4WorkitemIdX = 0;
inline constexpr uint16_t kScalarOperandTtmpBase = 108;
inline constexpr uint16_t kTtmpRdna4GridYz = 7;
inline constexpr uint16_t kTtmpRdna4GridX = 9;
inline constexpr uint32_t kMaxVgprs = 256;
inline constexpr uint32_t kMaxSgprs = 106;
inline constexpr uint16_t kGfx12HwRegHwId1 = 23;
inline constexpr uint16_t kGfx12HwIdOwnerBits = 10;
inline constexpr uint8_t kRdna4ScopeDevice = 2;
inline constexpr uint32_t kRdna4FlatNoSaddr = 0x7C;
inline constexpr uint32_t kWaitLoadcnt0 = 0xBFC00000u;
inline constexpr uint32_t kWaitDscnt0 = 0xBFC60000u;
inline constexpr uint32_t kWaitAluDepctrSaSdst0 = 0xBF88FF9Eu;
inline constexpr uint64_t kAmdhsaKernelEntryAlignment = 256;

[[nodiscard]] constexpr uint16_t ttmp_scalar_operand(uint16_t ttmp) {
  return static_cast<uint16_t>(kScalarOperandTtmpBase + ttmp);
}

struct ConSanMoiWorkgroupSource {
  std::optional<uint16_t> scalar_src;
  bool shift_right_16 = false;
  bool mask_low_16 = false;
};

struct ConSanMoiWorkgroupSources {
  ConSanMoiWorkgroupSource x;
  ConSanMoiWorkgroupSource y;
  ConSanMoiWorkgroupSource z;
};

struct ConSanMoiAccessRange {
  uint32_t static_byte_offset = 0;
  uint32_t byte_count = 0;
};

using KD = rocr::llvm::amdhsa::kernel_descriptor_t;
namespace kd = rocr::llvm::amdhsa;

[[nodiscard]] bool equals_any(std::string_view value,
                              std::initializer_list<std::string_view> candidates) {
  for (std::string_view candidate : candidates) {
    if (value == candidate)
      return true;
  }
  return false;
}

[[nodiscard]] std::optional<uint32_t> read_u32(std::span<const uint8_t> bytes, uint64_t offset) {
  if (offset > bytes.size() || sizeof(uint32_t) > bytes.size() - offset)
    return std::nullopt;
  uint32_t word = 0;
  std::memcpy(&word, bytes.data() + offset, sizeof(word));
  return word;
}

[[nodiscard]] bool is_single_range_native_lds_mnemonic(std::string_view mnemonic);

[[nodiscard]] std::optional<uint32_t>
two_address_native_lds_offset_scale(std::string_view mnemonic);

[[nodiscard]] bool is_moi_native_lds_candidate(const ConSanLdsSite &site) {
  if (site.kind != ConSanLdsAccessKind::Read && site.kind != ConSanLdsAccessKind::Write)
    return false;
  return is_single_range_native_lds_mnemonic(site.mnemonic) ||
         two_address_native_lds_offset_scale(site.mnemonic).has_value();
}

struct SkippedMoiLdsCounts {
  uint32_t unsupported_kind = 0;
  uint32_t unsupported_mnemonic = 0;

  [[nodiscard]] bool any() const { return unsupported_kind != 0 || unsupported_mnemonic != 0; }
};

void count_skipped_moi_lds_candidate(const ConSanLdsSite &site, SkippedMoiLdsCounts &counts) {
  if (site.kind != ConSanLdsAccessKind::Read && site.kind != ConSanLdsAccessKind::Write) {
    ++counts.unsupported_kind;
    return;
  }
  ++counts.unsupported_mnemonic;
}

void warn_skipped_moi_lds_candidates(std::string_view container_name, bool in_kernel,
                                     const SkippedMoiLdsCounts &counts, ConSanResult &result) {
  if (!counts.any())
    return;
  result.warnings.emplace_back(
      "ConSan MOI skipped native LDS sites in " + std::string(in_kernel ? "kernel " : "function ") +
      std::string(container_name) +
      ": unsupported_kind=" + std::to_string(counts.unsupported_kind) +
      " unsupported_mnemonic=" + std::to_string(counts.unsupported_mnemonic));
}

[[nodiscard]] bool is_moi_flat_candidate(const ConSanFlatSite &site) {
  if (site.kind == ConSanLdsAccessKind::Other)
    return false;
  return site.address_space_hint == ConSanFlatAddressSpaceHint::Group ||
         site.address_space_hint == ConSanFlatAddressSpaceHint::MaybeGroup;
}

struct SkippedMoiFlatCounts {
  uint32_t other_kind = 0;
  uint32_t unknown = 0;
  uint32_t private_hint = 0;
  uint32_t maybe_private = 0;
  uint32_t global = 0;

  [[nodiscard]] bool any() const {
    return other_kind != 0 || unknown != 0 || private_hint != 0 || maybe_private != 0 ||
           global != 0;
  }
};

void count_skipped_moi_flat_candidate(const ConSanFlatSite &site, SkippedMoiFlatCounts &counts) {
  if (site.kind == ConSanLdsAccessKind::Other) {
    ++counts.other_kind;
    return;
  }

  switch (site.address_space_hint) {
  case ConSanFlatAddressSpaceHint::Unknown:
    ++counts.unknown;
    return;
  case ConSanFlatAddressSpaceHint::Private:
    ++counts.private_hint;
    return;
  case ConSanFlatAddressSpaceHint::MaybePrivate:
    ++counts.maybe_private;
    return;
  case ConSanFlatAddressSpaceHint::Global:
    ++counts.global;
    return;
  case ConSanFlatAddressSpaceHint::Group:
  case ConSanFlatAddressSpaceHint::MaybeGroup:
    return;
  }
}

void warn_skipped_moi_flat_candidates(std::string_view container_name, bool in_kernel,
                                      const SkippedMoiFlatCounts &counts, ConSanResult &result) {
  if (!counts.any())
    return;
  result.warnings.emplace_back(
      "ConSan MOI skipped flat sites in " + std::string(in_kernel ? "kernel " : "function ") +
      std::string(container_name) + " because they were not likely group/LDS candidates: " +
      "unknown=" + std::to_string(counts.unknown) +
      " private=" + std::to_string(counts.private_hint) + " maybe_private=" +
      std::to_string(counts.maybe_private) + " global=" + std::to_string(counts.global) +
      " unsupported_kind=" + std::to_string(counts.other_kind));
}

[[nodiscard]] ConSanMoiCandidate make_moi_candidate(std::string container_name, bool in_kernel,
                                                    std::optional<uint64_t> descriptor_file_offset,
                                                    const ConSanLdsSite &site) {
  ConSanMoiCandidate candidate;
  candidate.container_name = std::move(container_name);
  candidate.in_kernel = in_kernel;
  candidate.source = ConSanMoiCandidateSource::NativeLds;
  candidate.kind = site.kind;
  candidate.text_offset = site.text_offset;
  candidate.file_offset = site.file_offset;
  candidate.size = site.size;
  candidate.width_bits = site.width_bits;
  candidate.dst_vgpr = site.dst_vgpr;
  candidate.addr_vgpr = site.addr_vgpr;
  candidate.data_vgpr = site.data_vgpr;
  candidate.kernel_descriptor_file_offset = descriptor_file_offset;
  candidate.mnemonic = site.mnemonic;
  return candidate;
}

[[nodiscard]] ConSanMoiCandidate make_moi_candidate(std::string container_name, bool in_kernel,
                                                    std::optional<uint64_t> descriptor_file_offset,
                                                    const ConSanFlatSite &site) {
  ConSanMoiCandidate candidate;
  candidate.container_name = std::move(container_name);
  candidate.in_kernel = in_kernel;
  candidate.source = site.address_space_hint == ConSanFlatAddressSpaceHint::Group
                         ? ConSanMoiCandidateSource::FlatGroup
                         : ConSanMoiCandidateSource::FlatMaybeGroup;
  candidate.kind = site.kind;
  candidate.flat_address_space_hint = site.address_space_hint;
  candidate.text_offset = site.text_offset;
  candidate.file_offset = site.file_offset;
  candidate.size = site.size;
  candidate.width_bits = site.width_bits;
  candidate.dst_vgpr = site.dst_vgpr;
  candidate.addr_vgpr = site.addr_vgpr;
  candidate.data_vgpr = site.data_vgpr;
  candidate.kernel_descriptor_file_offset = descriptor_file_offset;
  candidate.raw_saddr = site.raw_saddr;
  candidate.raw_vaddr = site.raw_vaddr;
  candidate.raw_vsrc = site.raw_vsrc;
  candidate.raw_vdst = site.raw_vdst;
  candidate.raw_ioffset = site.raw_ioffset;
  candidate.raw_scope = site.raw_scope;
  candidate.raw_th = site.raw_th;
  candidate.mnemonic = site.mnemonic;
  return candidate;
}

void append_moi_candidates(const ConSanKernelInfo &kernel, ConSanResult &result) {
  SkippedMoiLdsCounts skipped_lds_counts;
  for (const ConSanLdsSite &site : kernel.lds_sites) {
    if (is_moi_native_lds_candidate(site)) {
      result.moi_candidates.push_back(
          make_moi_candidate(kernel.name, true, kernel.descriptor_file_offset, site));
    } else {
      count_skipped_moi_lds_candidate(site, skipped_lds_counts);
    }
  }
  warn_skipped_moi_lds_candidates(kernel.name, true, skipped_lds_counts, result);
  SkippedMoiFlatCounts skipped_flat_counts;
  for (const ConSanFlatSite &site : kernel.flat_sites) {
    if (is_moi_flat_candidate(site)) {
      result.moi_candidates.push_back(
          make_moi_candidate(kernel.name, true, kernel.descriptor_file_offset, site));
    } else {
      count_skipped_moi_flat_candidate(site, skipped_flat_counts);
    }
  }
  warn_skipped_moi_flat_candidates(kernel.name, true, skipped_flat_counts, result);
}

void append_moi_candidates(const ConSanFunctionInfo &function, ConSanResult &result) {
  SkippedMoiLdsCounts skipped_lds_counts;
  for (const ConSanLdsSite &site : function.lds_sites) {
    if (is_moi_native_lds_candidate(site)) {
      result.moi_candidates.push_back(make_moi_candidate(function.name, false, std::nullopt, site));
    } else {
      count_skipped_moi_lds_candidate(site, skipped_lds_counts);
    }
  }
  warn_skipped_moi_lds_candidates(function.name, false, skipped_lds_counts, result);
  SkippedMoiFlatCounts skipped_flat_counts;
  for (const ConSanFlatSite &site : function.flat_sites) {
    if (is_moi_flat_candidate(site)) {
      result.moi_candidates.push_back(make_moi_candidate(function.name, false, std::nullopt, site));
    } else {
      count_skipped_moi_flat_candidate(site, skipped_flat_counts);
    }
  }
  warn_skipped_moi_flat_candidates(function.name, false, skipped_flat_counts, result);
}

[[nodiscard]] uint32_t count_nop_padding(std::span<const uint8_t> bytes, uint64_t offset,
                                         rj_code_arch_t arch) {
  if (offset > bytes.size())
    return 0;

  const uint32_t nop = build_s_nop(0, arch);
  uint32_t count = 0;
  while (bytes.size() - offset >= sizeof(uint32_t)) {
    uint32_t word = 0;
    std::memcpy(&word, bytes.data() + offset, sizeof(word));
    if (word != nop)
      break;
    ++count;
    offset += sizeof(uint32_t);
  }
  return count;
}

[[nodiscard]] std::optional<uint32_t> byte_count_for_candidate(const ConSanMoiCandidate &site) {
  if (site.width_bits == 0 || site.width_bits % 8 != 0)
    return std::nullopt;
  return site.width_bits / 8u;
}

[[nodiscard]] bool is_single_range_native_lds_mnemonic(std::string_view mnemonic) {
  if (equals_any(mnemonic,
                 {"ds_load_i8", "ds_load_u8", "ds_load_i16", "ds_load_u16", "ds_load_u8_d16",
                  "ds_load_u8_d16_hi", "ds_load_i8_d16", "ds_load_i8_d16_hi", "ds_load_u16_d16",
                  "ds_load_u16_d16_hi", "ds_store_b8", "ds_store_b16", "ds_store_b8_d16_hi",
                  "ds_store_b16_d16_hi"}))
    return true;

  return equals_any(mnemonic, {"ds_load_b32", "ds_load_b64", "ds_load_b128", "ds_read_b32",
                               "ds_read_b64", "ds_read_b128", "ds_store_b32", "ds_store_b64",
                               "ds_store_b128", "ds_write_b32", "ds_write_b64", "ds_write_b128"});
}

[[nodiscard]] std::optional<uint32_t>
two_address_native_lds_offset_scale(std::string_view mnemonic) {
  if (mnemonic == "ds_load_2addr_b32" || mnemonic == "ds_store_2addr_b32")
    return 4u;
  if (mnemonic == "ds_load_2addr_b64" || mnemonic == "ds_store_2addr_b64")
    return 8u;
  if (mnemonic == "ds_load_2addr_stride64_b32" || mnemonic == "ds_store_2addr_stride64_b32")
    return 256u;
  if (mnemonic == "ds_load_2addr_stride64_b64" || mnemonic == "ds_store_2addr_stride64_b64")
    return 512u;
  return std::nullopt;
}

[[nodiscard]] std::optional<std::vector<ConSanMoiAccessRange>>
candidate_access_ranges(std::span<const uint8_t> bytes, const ConSanMoiCandidate &candidate) {
  const std::optional<uint32_t> byte_count = byte_count_for_candidate(candidate);
  if (!byte_count)
    return std::nullopt;

  if (candidate.source == ConSanMoiCandidateSource::NativeLds) {
    const std::optional<uint32_t> word0 = read_u32(bytes, candidate.file_offset);
    if (!word0)
      return std::nullopt;
    if (is_single_range_native_lds_mnemonic(candidate.mnemonic)) {
      // RDNA DS encodes the normal single-address immediate byte offset as
      // offset0 | (offset1 << 8).
      return std::vector<ConSanMoiAccessRange>{{*word0 & 0xffffu, *byte_count}};
    }

    const std::optional<uint32_t> two_address_scale =
        two_address_native_lds_offset_scale(candidate.mnemonic);
    if (!two_address_scale)
      return std::nullopt;
    const uint32_t offset0 = *word0 & 0xffu;
    const uint32_t offset1 = (*word0 >> 8u) & 0xffu;
    return std::vector<ConSanMoiAccessRange>{
        {offset0 * *two_address_scale, *byte_count},
        {offset1 * *two_address_scale, *byte_count},
    };
  }

  if (candidate.source == ConSanMoiCandidateSource::FlatGroup ||
      candidate.source == ConSanMoiCandidateSource::FlatMaybeGroup)
    return std::vector<ConSanMoiAccessRange>{{0u, *byte_count}};

  return std::nullopt;
}

[[nodiscard]] bool is_first_light_native_lds_candidate(const ConSanMoiCandidate &candidate,
                                                       std::span<const uint8_t> bytes) {
  if (candidate.source != ConSanMoiCandidateSource::NativeLds)
    return false;
  return candidate_access_ranges(bytes, candidate).has_value();
}

[[nodiscard]] bool is_first_light_flat_candidate(const ConSanMoiCandidate &candidate) {
  if (candidate.source != ConSanMoiCandidateSource::FlatGroup &&
      candidate.source != ConSanMoiCandidateSource::FlatMaybeGroup)
    return false;
  if (candidate.size != 3u * sizeof(uint32_t))
    return false;
  if (!candidate.raw_ioffset || *candidate.raw_ioffset != 0)
    return false;
  if (!candidate.addr_vgpr || *candidate.addr_vgpr >= 255)
    return false;
  if (candidate.mnemonic != "flat_load_b32" && candidate.mnemonic != "flat_load_b64" &&
      candidate.mnemonic != "flat_load_b128" && candidate.mnemonic != "flat_store_b32" &&
      candidate.mnemonic != "flat_store_b64" && candidate.mnemonic != "flat_store_b128")
    return false;
  return true;
}

[[nodiscard]] bool is_first_light_access_record_candidate(const ConSanMoiCandidate &candidate,
                                                          std::span<const uint8_t> bytes) {
  if (candidate.kind != ConSanLdsAccessKind::Read && candidate.kind != ConSanLdsAccessKind::Write)
    return false;
  if (candidate.size == 0 || candidate.size % sizeof(uint32_t) != 0)
    return false;
  if (candidate.file_offset > bytes.size() || candidate.size > bytes.size() - candidate.file_offset)
    return false;
  if (!byte_count_for_candidate(candidate))
    return false;
  if (candidate.source == ConSanMoiCandidateSource::NativeLds)
    return is_first_light_native_lds_candidate(candidate, bytes);
  return is_first_light_flat_candidate(candidate);
}

[[nodiscard]] std::vector<const ConSanMoiCandidate *>
find_first_light_access_record_candidates(const ConSanResult &result,
                                          std::span<const uint8_t> bytes) {
  std::vector<const ConSanMoiCandidate *> candidates;
  for (const ConSanMoiCandidate &candidate : result.moi_candidates) {
    if (!is_first_light_access_record_candidate(candidate, bytes))
      continue;
    candidates.push_back(&candidate);
  }
  return candidates;
}

[[nodiscard]] bool append_store_u32_literal(std::vector<uint32_t> &words, uint64_t address,
                                            uint32_t value, uint16_t scratch_vgpr,
                                            rj_code_arch_t arch) {
  const uint16_t address_lo_vgpr = scratch_vgpr;
  const uint16_t address_hi_vgpr = static_cast<uint16_t>(scratch_vgpr + 1u);
  const uint16_t value_vgpr = static_cast<uint16_t>(scratch_vgpr + 2u);

  const auto mov_address_lo =
      build_v_mov_b32_e64_literal(address_lo_vgpr, static_cast<uint32_t>(address), arch);
  const auto mov_address_hi =
      build_v_mov_b32_e64_literal(address_hi_vgpr, static_cast<uint32_t>(address >> 32u), arch);
  const auto mov_value = build_v_mov_b32_e64_literal(value_vgpr, value, arch);
  const auto store = build_flat_store_b32_vaddr_vsrc(address_lo_vgpr, value_vgpr, arch);
  if (!mov_address_lo || !mov_address_hi || !mov_value || !store)
    return false;

  words.insert(words.end(), mov_address_lo->begin(), mov_address_lo->end());
  words.insert(words.end(), mov_address_hi->begin(), mov_address_hi->end());
  words.insert(words.end(), mov_value->begin(), mov_value->end());
  words.insert(words.end(), store->begin(), store->end());
  return true;
}

[[nodiscard]] bool append_store_u32_vgpr(std::vector<uint32_t> &words, uint64_t address,
                                         uint16_t value_vgpr, uint16_t scratch_vgpr,
                                         rj_code_arch_t arch) {
  const uint16_t address_lo_vgpr = scratch_vgpr;
  const uint16_t address_hi_vgpr = static_cast<uint16_t>(scratch_vgpr + 1u);

  const auto mov_address_lo =
      build_v_mov_b32_e64_literal(address_lo_vgpr, static_cast<uint32_t>(address), arch);
  const auto mov_address_hi =
      build_v_mov_b32_e64_literal(address_hi_vgpr, static_cast<uint32_t>(address >> 32u), arch);
  const auto store = build_flat_store_b32_vaddr_vsrc(address_lo_vgpr, value_vgpr, arch);
  if (!mov_address_lo || !mov_address_hi || !store)
    return false;

  words.insert(words.end(), mov_address_lo->begin(), mov_address_lo->end());
  words.insert(words.end(), mov_address_hi->begin(), mov_address_hi->end());
  words.insert(words.end(), store->begin(), store->end());
  return true;
}

[[nodiscard]] bool append_load_u32_vgpr(std::vector<uint32_t> &words, uint64_t address,
                                        uint16_t destination_vgpr, uint16_t scratch_vgpr,
                                        rj_code_arch_t arch) {
  const uint16_t address_lo_vgpr = scratch_vgpr;
  const uint16_t address_hi_vgpr = static_cast<uint16_t>(scratch_vgpr + 1u);

  const auto mov_address_lo =
      build_v_mov_b32_e64_literal(address_lo_vgpr, static_cast<uint32_t>(address), arch);
  const auto mov_address_hi =
      build_v_mov_b32_e64_literal(address_hi_vgpr, static_cast<uint32_t>(address >> 32u), arch);
  const auto load = build_flat_load_b32_vaddr_vdst(address_lo_vgpr, destination_vgpr, arch);
  if (!mov_address_lo || !mov_address_hi || !load)
    return false;

  words.insert(words.end(), mov_address_lo->begin(), mov_address_lo->end());
  words.insert(words.end(), mov_address_hi->begin(), mov_address_hi->end());
  words.insert(words.end(), load->begin(), load->end());
  words.push_back(kWaitLoadcnt0);
  return true;
}

[[nodiscard]] bool append_store_u32_scalar_src(std::vector<uint32_t> &words, uint64_t address,
                                               uint16_t scalar_src, uint16_t scratch_vgpr,
                                               rj_code_arch_t arch) {
  const uint16_t value_vgpr = static_cast<uint16_t>(scratch_vgpr + 2u);
  words.push_back(build_v_mov_b32_e32(value_vgpr, scalar_src, arch));
  return append_store_u32_vgpr(words, address, value_vgpr, scratch_vgpr, arch);
}

[[nodiscard]] bool append_store_workgroup_source(std::vector<uint32_t> &words, uint64_t address,
                                                 const ConSanMoiWorkgroupSource &source,
                                                 uint16_t scratch_vgpr, rj_code_arch_t arch) {
  if (!source.scalar_src)
    return true;
  const uint16_t value_vgpr = static_cast<uint16_t>(scratch_vgpr + 2u);
  words.push_back(build_v_mov_b32_e32(value_vgpr, *source.scalar_src, arch));
  if (source.mask_low_16) {
    const auto shift_left =
        build_v_lshlrev_b32_e32(value_vgpr, scalar_positive_inline_u32(16), value_vgpr, arch);
    const auto shift_right =
        build_v_lshrrev_b32_e32(value_vgpr, scalar_positive_inline_u32(16), value_vgpr, arch);
    if (!shift_left || !shift_right)
      return false;
    words.push_back(*shift_left);
    words.push_back(*shift_right);
  }
  if (source.shift_right_16) {
    const auto shift =
        build_v_lshrrev_b32_e32(value_vgpr, scalar_positive_inline_u32(16), value_vgpr, arch);
    if (!shift)
      return false;
    words.push_back(*shift);
  }
  return append_store_u32_vgpr(words, address, value_vgpr, scratch_vgpr, arch);
}

[[nodiscard]] bool append_atomic_event_index_store(std::vector<uint32_t> &words,
                                                   uint64_t counter_address,
                                                   uint64_t event_index_address,
                                                   uint16_t scratch_vgpr, rj_code_arch_t arch) {
  const uint16_t address_lo_vgpr = scratch_vgpr;
  const uint16_t address_hi_vgpr = static_cast<uint16_t>(scratch_vgpr + 1u);
  const uint16_t value_vgpr = static_cast<uint16_t>(scratch_vgpr + 2u);

  const auto mov_address_lo =
      build_v_mov_b32_e64_literal(address_lo_vgpr, static_cast<uint32_t>(counter_address), arch);
  const auto mov_address_hi = build_v_mov_b32_e64_literal(
      address_hi_vgpr, static_cast<uint32_t>(counter_address >> 32u), arch);
  const auto mov_one = build_v_mov_b32_e64_literal(value_vgpr, 1u, arch);
  const auto atomic_add = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      address_lo_vgpr, value_vgpr, value_vgpr, /*return_old_value=*/true, kRdna4ScopeDevice, arch);
  if (!mov_address_lo || !mov_address_hi || !mov_one || !atomic_add)
    return false;

  words.insert(words.end(), mov_address_lo->begin(), mov_address_lo->end());
  words.insert(words.end(), mov_address_hi->begin(), mov_address_hi->end());
  words.insert(words.end(), mov_one->begin(), mov_one->end());
  words.insert(words.end(), atomic_add->begin(), atomic_add->end());
  words.push_back(kWaitLoadcnt0);
  return append_store_u32_vgpr(words, event_index_address, value_vgpr, scratch_vgpr, arch);
}

[[nodiscard]] bool append_atomic_fetch_add_one_u32(std::vector<uint32_t> &words,
                                                   uint64_t counter_address, uint16_t result_vgpr,
                                                   uint16_t scratch_vgpr, rj_code_arch_t arch) {
  const uint16_t address_lo_vgpr = scratch_vgpr;
  const uint16_t address_hi_vgpr = static_cast<uint16_t>(scratch_vgpr + 1u);

  const auto mov_address_lo =
      build_v_mov_b32_e64_literal(address_lo_vgpr, static_cast<uint32_t>(counter_address), arch);
  const auto mov_address_hi = build_v_mov_b32_e64_literal(
      address_hi_vgpr, static_cast<uint32_t>(counter_address >> 32u), arch);
  const auto mov_one = build_v_mov_b32_e64_literal(result_vgpr, 1u, arch);
  const auto atomic_add =
      build_flat_atomic_add_u32_vaddr_vsrc_vdst(address_lo_vgpr, result_vgpr, result_vgpr,
                                                /*return_old_value=*/true, kRdna4ScopeDevice, arch);
  if (!mov_address_lo || !mov_address_hi || !mov_one || !atomic_add)
    return false;

  words.insert(words.end(), mov_address_lo->begin(), mov_address_lo->end());
  words.insert(words.end(), mov_address_hi->begin(), mov_address_hi->end());
  words.insert(words.end(), mov_one->begin(), mov_one->end());
  words.insert(words.end(), atomic_add->begin(), atomic_add->end());
  words.push_back(kWaitLoadcnt0);
  return true;
}

[[nodiscard]] bool append_dynamic_barrier_record_address(std::vector<uint32_t> &words,
                                                         uint64_t field_address, uint16_t slot_vgpr,
                                                         uint16_t scratch_vgpr,
                                                         rj_code_arch_t arch) {
  const uint16_t address_lo_vgpr = scratch_vgpr;
  const uint16_t address_hi_vgpr = static_cast<uint16_t>(scratch_vgpr + 1u);
  const uint16_t scaled_slot_vgpr = static_cast<uint16_t>(scratch_vgpr + 3u);
  const uint16_t tmp_vgpr = static_cast<uint16_t>(scratch_vgpr + 4u);

  const auto mov_address_lo =
      build_v_mov_b32_e64_literal(address_lo_vgpr, static_cast<uint32_t>(field_address), arch);
  const auto mov_address_hi = build_v_mov_b32_e64_literal(
      address_hi_vgpr, static_cast<uint32_t>(field_address >> 32u), arch);
  const auto slot_times_32 =
      build_v_lshlrev_b32_e32(scaled_slot_vgpr, scalar_positive_inline_u32(5), slot_vgpr, arch);
  const auto slot_times_8 =
      build_v_lshlrev_b32_e32(tmp_vgpr, scalar_positive_inline_u32(3), slot_vgpr, arch);
  const auto slot_times_40 = build_v_add_nc_u32_e32(
      scaled_slot_vgpr, vector_source_vgpr(scaled_slot_vgpr), tmp_vgpr, arch);
  const auto address_with_slot = build_v_add_nc_u32_e32(
      address_lo_vgpr, vector_source_vgpr(address_lo_vgpr), scaled_slot_vgpr, arch);
  if (!mov_address_lo || !mov_address_hi || !slot_times_32 || !slot_times_8 || !slot_times_40 ||
      !address_with_slot)
    return false;

  words.insert(words.end(), mov_address_lo->begin(), mov_address_lo->end());
  words.insert(words.end(), mov_address_hi->begin(), mov_address_hi->end());
  words.push_back(*slot_times_32);
  words.push_back(*slot_times_8);
  words.push_back(*slot_times_40);
  words.push_back(*address_with_slot);
  return true;
}

[[nodiscard]] bool append_dynamic_barrier_store_u32_vgpr(std::vector<uint32_t> &words,
                                                         uint64_t field_address,
                                                         uint16_t value_vgpr, uint16_t slot_vgpr,
                                                         uint16_t scratch_vgpr,
                                                         rj_code_arch_t arch) {
  const uint16_t address_lo_vgpr = scratch_vgpr;
  const auto store = build_flat_store_b32_vaddr_vsrc(address_lo_vgpr, value_vgpr, arch);
  if (!store ||
      !append_dynamic_barrier_record_address(words, field_address, slot_vgpr, scratch_vgpr, arch))
    return false;
  words.insert(words.end(), store->begin(), store->end());
  return true;
}

[[nodiscard]] bool append_dynamic_barrier_store_u32_literal(std::vector<uint32_t> &words,
                                                            uint64_t field_address, uint32_t value,
                                                            uint16_t slot_vgpr,
                                                            uint16_t scratch_vgpr,
                                                            rj_code_arch_t arch) {
  const uint16_t value_vgpr = static_cast<uint16_t>(scratch_vgpr + 5u);
  const auto mov_value = build_v_mov_b32_e64_literal(value_vgpr, value, arch);
  if (!mov_value)
    return false;
  words.insert(words.end(), mov_value->begin(), mov_value->end());
  return append_dynamic_barrier_store_u32_vgpr(words, field_address, value_vgpr, slot_vgpr,
                                               scratch_vgpr, arch);
}

[[nodiscard]] bool
append_dynamic_barrier_store_u32_scalar_src(std::vector<uint32_t> &words, uint64_t field_address,
                                            uint16_t scalar_src, uint16_t slot_vgpr,
                                            uint16_t scratch_vgpr, rj_code_arch_t arch) {
  const uint16_t value_vgpr = static_cast<uint16_t>(scratch_vgpr + 5u);
  words.push_back(build_v_mov_b32_e32(value_vgpr, scalar_src, arch));
  return append_dynamic_barrier_store_u32_vgpr(words, field_address, value_vgpr, slot_vgpr,
                                               scratch_vgpr, arch);
}

[[nodiscard]] bool append_dynamic_barrier_store_workgroup_source(
    std::vector<uint32_t> &words, uint64_t field_address, const ConSanMoiWorkgroupSource &source,
    uint16_t slot_vgpr, uint16_t scratch_vgpr, rj_code_arch_t arch) {
  if (!source.scalar_src)
    return true;
  const uint16_t value_vgpr = static_cast<uint16_t>(scratch_vgpr + 5u);
  words.push_back(build_v_mov_b32_e32(value_vgpr, *source.scalar_src, arch));
  if (source.mask_low_16) {
    const auto shift_left =
        build_v_lshlrev_b32_e32(value_vgpr, scalar_positive_inline_u32(16), value_vgpr, arch);
    const auto shift_right =
        build_v_lshrrev_b32_e32(value_vgpr, scalar_positive_inline_u32(16), value_vgpr, arch);
    if (!shift_left || !shift_right)
      return false;
    words.push_back(*shift_left);
    words.push_back(*shift_right);
  }
  if (source.shift_right_16) {
    const auto shift =
        build_v_lshrrev_b32_e32(value_vgpr, scalar_positive_inline_u32(16), value_vgpr, arch);
    if (!shift)
      return false;
    words.push_back(*shift);
  }
  return append_dynamic_barrier_store_u32_vgpr(words, field_address, value_vgpr, slot_vgpr,
                                               scratch_vgpr, arch);
}

[[nodiscard]] bool
append_dynamic_barrier_event_index_store(std::vector<uint32_t> &words, uint64_t counter_address,
                                         uint64_t field_address, uint16_t slot_vgpr,
                                         uint16_t scratch_vgpr, rj_code_arch_t arch) {
  const uint16_t value_vgpr = static_cast<uint16_t>(scratch_vgpr + 5u);
  return append_atomic_fetch_add_one_u32(words, counter_address, value_vgpr, scratch_vgpr, arch) &&
         append_dynamic_barrier_store_u32_vgpr(words, field_address, value_vgpr, slot_vgpr,
                                               scratch_vgpr, arch);
}

[[nodiscard]] bool append_dynamic_access_record_address(std::vector<uint32_t> &words,
                                                        uint64_t field_address, uint16_t slot_vgpr,
                                                        uint16_t scratch_vgpr,
                                                        rj_code_arch_t arch) {
  const uint16_t address_lo_vgpr = scratch_vgpr;
  const uint16_t address_hi_vgpr = static_cast<uint16_t>(scratch_vgpr + 1u);
  const uint16_t scaled_slot_vgpr = static_cast<uint16_t>(scratch_vgpr + 3u);

  const auto mov_address_lo =
      build_v_mov_b32_e64_literal(address_lo_vgpr, static_cast<uint32_t>(field_address), arch);
  const auto mov_address_hi = build_v_mov_b32_e64_literal(
      address_hi_vgpr, static_cast<uint32_t>(field_address >> 32u), arch);
  const auto slot_times_64 =
      build_v_lshlrev_b32_e32(scaled_slot_vgpr, scalar_positive_inline_u32(6), slot_vgpr, arch);
  const auto address_with_slot = build_v_add_nc_u32_e32(
      address_lo_vgpr, vector_source_vgpr(address_lo_vgpr), scaled_slot_vgpr, arch);
  if (!mov_address_lo || !mov_address_hi || !slot_times_64 || !address_with_slot)
    return false;

  words.insert(words.end(), mov_address_lo->begin(), mov_address_lo->end());
  words.insert(words.end(), mov_address_hi->begin(), mov_address_hi->end());
  words.push_back(*slot_times_64);
  words.push_back(*address_with_slot);
  return true;
}

[[nodiscard]] bool append_dynamic_access_store_u32_vgpr(std::vector<uint32_t> &words,
                                                        uint64_t field_address, uint16_t value_vgpr,
                                                        uint16_t slot_vgpr, uint16_t scratch_vgpr,
                                                        rj_code_arch_t arch) {
  const uint16_t address_lo_vgpr = scratch_vgpr;
  const auto store = build_flat_store_b32_vaddr_vsrc(address_lo_vgpr, value_vgpr, arch);
  if (!store ||
      !append_dynamic_access_record_address(words, field_address, slot_vgpr, scratch_vgpr, arch))
    return false;
  words.insert(words.end(), store->begin(), store->end());
  return true;
}

[[nodiscard]] bool append_dynamic_access_store_u32_literal(std::vector<uint32_t> &words,
                                                           uint64_t field_address, uint32_t value,
                                                           uint16_t slot_vgpr,
                                                           uint16_t scratch_vgpr,
                                                           rj_code_arch_t arch) {
  const uint16_t value_vgpr = static_cast<uint16_t>(scratch_vgpr + 5u);
  const auto mov_value = build_v_mov_b32_e64_literal(value_vgpr, value, arch);
  if (!mov_value)
    return false;
  words.insert(words.end(), mov_value->begin(), mov_value->end());
  return append_dynamic_access_store_u32_vgpr(words, field_address, value_vgpr, slot_vgpr,
                                              scratch_vgpr, arch);
}

[[nodiscard]] bool
append_dynamic_access_store_u32_scalar_src(std::vector<uint32_t> &words, uint64_t field_address,
                                           uint16_t scalar_src, uint16_t slot_vgpr,
                                           uint16_t scratch_vgpr, rj_code_arch_t arch) {
  const uint16_t value_vgpr = static_cast<uint16_t>(scratch_vgpr + 5u);
  words.push_back(build_v_mov_b32_e32(value_vgpr, scalar_src, arch));
  return append_dynamic_access_store_u32_vgpr(words, field_address, value_vgpr, slot_vgpr,
                                              scratch_vgpr, arch);
}

[[nodiscard]] bool append_dynamic_access_store_workgroup_source(
    std::vector<uint32_t> &words, uint64_t field_address, const ConSanMoiWorkgroupSource &source,
    uint16_t slot_vgpr, uint16_t scratch_vgpr, rj_code_arch_t arch) {
  if (!source.scalar_src)
    return true;
  const uint16_t value_vgpr = static_cast<uint16_t>(scratch_vgpr + 5u);
  words.push_back(build_v_mov_b32_e32(value_vgpr, *source.scalar_src, arch));
  if (source.mask_low_16) {
    const auto shift_left =
        build_v_lshlrev_b32_e32(value_vgpr, scalar_positive_inline_u32(16), value_vgpr, arch);
    const auto shift_right =
        build_v_lshrrev_b32_e32(value_vgpr, scalar_positive_inline_u32(16), value_vgpr, arch);
    if (!shift_left || !shift_right)
      return false;
    words.push_back(*shift_left);
    words.push_back(*shift_right);
  }
  if (source.shift_right_16) {
    const auto shift =
        build_v_lshrrev_b32_e32(value_vgpr, scalar_positive_inline_u32(16), value_vgpr, arch);
    if (!shift)
      return false;
    words.push_back(*shift);
  }
  return append_dynamic_access_store_u32_vgpr(words, field_address, value_vgpr, slot_vgpr,
                                              scratch_vgpr, arch);
}

[[nodiscard]] bool
append_dynamic_access_event_index_store(std::vector<uint32_t> &words, uint64_t counter_address,
                                        uint64_t field_address, uint16_t slot_vgpr,
                                        uint16_t scratch_vgpr, rj_code_arch_t arch) {
  const uint16_t value_vgpr = static_cast<uint16_t>(scratch_vgpr + 5u);
  return append_atomic_fetch_add_one_u32(words, counter_address, value_vgpr, scratch_vgpr, arch) &&
         append_dynamic_access_store_u32_vgpr(words, field_address, value_vgpr, slot_vgpr,
                                              scratch_vgpr, arch);
}

[[nodiscard]] bool overlaps_scratch_range(uint16_t value, uint16_t scratch_vgpr, uint16_t count) {
  return count != 0 && value >= scratch_vgpr && value < static_cast<uint16_t>(scratch_vgpr + count);
}

[[nodiscard]] bool overlaps_scratch_triplet(uint16_t value, uint16_t scratch_vgpr) {
  return overlaps_scratch_range(value, scratch_vgpr, 3);
}

[[nodiscard]] bool is_sop1_saveexec_word(uint32_t word) {
  constexpr uint32_t kSop1PrefixMask = 0xFF800000u;
  constexpr uint32_t kSop1OpMask = 0x0000FF00u;
  const uint32_t op = (word & kSop1OpMask) >> 8;
  return (word & kSop1PrefixMask) == (kSop1EncodingPrefix << 23) && op >= 0x20u && op <= 0x33u;
}

[[nodiscard]] bool has_recent_saveexec(std::span<const uint8_t> bytes,
                                       const ConSanMoiCandidate &candidate) {
  constexpr uint64_t kLookbackDwords = 3;
  for (uint64_t dword = 1; dword <= kLookbackDwords; ++dword) {
    const uint64_t byte_distance = dword * sizeof(uint32_t);
    if (candidate.file_offset < byte_distance)
      break;
    const std::optional<uint32_t> word = read_u32(bytes, candidate.file_offset - byte_distance);
    if (word && is_sop1_saveexec_word(*word))
      return true;
  }
  return false;
}

[[nodiscard]] bool range_overlaps_scratch_triplet(uint16_t base, uint16_t count,
                                                  uint16_t scratch_vgpr) {
  for (uint16_t i = 0; i < count; ++i) {
    const uint32_t value = static_cast<uint32_t>(base) + i;
    if (value <= std::numeric_limits<uint16_t>::max() &&
        overlaps_scratch_triplet(static_cast<uint16_t>(value), scratch_vgpr))
      return true;
  }
  return false;
}

[[nodiscard]] bool range_overlaps(uint16_t lhs_base, uint16_t lhs_count, uint16_t rhs_base,
                                  uint16_t rhs_count) {
  const uint32_t lhs_end = static_cast<uint32_t>(lhs_base) + lhs_count;
  const uint32_t rhs_end = static_cast<uint32_t>(rhs_base) + rhs_count;
  return static_cast<uint32_t>(lhs_base) < rhs_end && static_cast<uint32_t>(rhs_base) < lhs_end;
}

[[nodiscard]] bool reject_optional_scratch_range_overlap(std::optional<uint16_t> value,
                                                         uint16_t scratch_vgpr,
                                                         uint16_t scratch_count,
                                                         std::string_view value_name,
                                                         std::vector<std::string> &errors) {
  if (!value || !overlaps_scratch_range(*value, scratch_vgpr, scratch_count))
    return false;
  errors.emplace_back(std::string("ConSan MOI scratch VGPRs overlap ") + std::string(value_name) +
                      " VGPR");
  return true;
}

void append_word_bytes(std::vector<uint8_t> &bytes, uint32_t word) {
  const auto *begin = reinterpret_cast<const uint8_t *>(&word);
  bytes.insert(bytes.end(), begin, begin + sizeof(word));
}

void append_words_bytes(std::vector<uint8_t> &bytes, std::span<const uint32_t> words) {
  for (uint32_t word : words)
    append_word_bytes(bytes, word);
}

void append_nop_padding_to_alignment(std::vector<uint8_t> &bytes, uint64_t alignment,
                                     rj_code_arch_t arch) {
  if (alignment == 0)
    return;
  while (bytes.size() % alignment != 0)
    append_word_bytes(bytes, build_s_nop(0, arch));
}

[[nodiscard]] bool write_word_bytes(std::vector<uint8_t> &bytes, uint64_t offset, uint32_t word) {
  if (offset > bytes.size() || sizeof(word) > bytes.size() - offset)
    return false;
  std::memcpy(bytes.data() + offset, &word, sizeof(word));
  return true;
}

[[nodiscard]] uint32_t moi_descriptor_wavefront_size(const KD &desc) {
  const bool wave32 = AMDHSA_BITS_GET(desc.kernel_code_properties,
                                      kd::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32);
  return wave32 ? 32 : 64;
}

[[nodiscard]] uint32_t moi_descriptor_user_sgpr_count(const KD &desc) {
  return AMDHSA_BITS_GET(desc.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT);
}

[[nodiscard]] std::optional<uint16_t> moi_descriptor_workgroup_id_sgpr(const KD &desc,
                                                                       uint32_t dimension) {
  const uint32_t rsrc2 = desc.compute_pgm_rsrc2;
  const bool enabled[3] = {
      AMDHSA_BITS_GET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X) != 0,
      AMDHSA_BITS_GET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y) != 0,
      AMDHSA_BITS_GET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z) != 0,
  };

  uint32_t sgpr = moi_descriptor_user_sgpr_count(desc);
  for (uint32_t i = 0; i < 3; ++i) {
    if (!enabled[i])
      continue;
    if (i == dimension)
      return static_cast<uint16_t>(sgpr);
    ++sgpr;
  }
  return std::nullopt;
}

[[nodiscard]] bool moi_descriptor_enables_workgroup_id(const KD &desc, uint32_t dimension) {
  switch (dimension) {
  case 0:
    return AMDHSA_BITS_GET(desc.compute_pgm_rsrc2,
                           kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X) != 0;
  case 1:
    return AMDHSA_BITS_GET(desc.compute_pgm_rsrc2,
                           kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y) != 0;
  case 2:
    return AMDHSA_BITS_GET(desc.compute_pgm_rsrc2,
                           kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z) != 0;
  default:
    return false;
  }
}

[[nodiscard]] std::optional<ConSanMoiWorkgroupSources>
moi_descriptor_workgroup_sources(std::span<const uint8_t> image, uint64_t descriptor_file_offset,
                                 rj_code_arch_t arch, std::vector<std::string> &errors) {
  if (descriptor_file_offset > image.size() || sizeof(KD) > image.size() - descriptor_file_offset) {
    errors.emplace_back("ConSan MOI workgroup-id descriptor exceeds ELF bytes");
    return std::nullopt;
  }

  KD desc{};
  std::memcpy(&desc, image.data() + descriptor_file_offset, sizeof(desc));
  if (arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_GFX1250) {
    return ConSanMoiWorkgroupSources{
        moi_descriptor_enables_workgroup_id(desc, 0)
            ? ConSanMoiWorkgroupSource{ttmp_scalar_operand(kTtmpRdna4GridX), false}
            : ConSanMoiWorkgroupSource{},
        moi_descriptor_enables_workgroup_id(desc, 1)
            ? ConSanMoiWorkgroupSource{ttmp_scalar_operand(kTtmpRdna4GridYz),
                                       /*shift_right_16=*/false, /*mask_low_16=*/true}
            : ConSanMoiWorkgroupSource{},
        moi_descriptor_enables_workgroup_id(desc, 2)
            ? ConSanMoiWorkgroupSource{ttmp_scalar_operand(kTtmpRdna4GridYz),
                                       /*shift_right_16=*/true}
            : ConSanMoiWorkgroupSource{},
    };
  }
  const auto sgpr_source = [](std::optional<uint16_t> sgpr) {
    return sgpr ? ConSanMoiWorkgroupSource{*sgpr, false} : ConSanMoiWorkgroupSource{};
  };
  return ConSanMoiWorkgroupSources{
      sgpr_source(moi_descriptor_workgroup_id_sgpr(desc, 0)),
      sgpr_source(moi_descriptor_workgroup_id_sgpr(desc, 1)),
      sgpr_source(moi_descriptor_workgroup_id_sgpr(desc, 2)),
  };
}

[[nodiscard]] uint32_t moi_descriptor_vgpr_granularity(const KD &desc) {
  return moi_descriptor_wavefront_size(desc) == 32 ? 8 : 4;
}

[[nodiscard]] uint32_t moi_descriptor_vgpr_allocation_count(const KD &desc) {
  const uint32_t granularity = moi_descriptor_vgpr_granularity(desc);
  const uint32_t granulated =
      AMDHSA_BITS_GET(desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
  const uint32_t allocated = (granulated + 1u) * granularity;
  return std::min<uint32_t>(allocated, kMaxVgprs);
}

[[nodiscard]] uint32_t moi_descriptor_sgpr_allocation_count(const KD &desc) {
  const uint32_t granulated = AMDHSA_BITS_GET(
      desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT);
  const uint32_t allocated = (granulated + 1u) * 8u;
  return std::min<uint32_t>(allocated, kMaxSgprs);
}

[[nodiscard]] std::optional<uint16_t> moi_descriptor_owner_shift(std::span<const uint8_t> image,
                                                                 uint64_t descriptor_file_offset,
                                                                 std::vector<std::string> &errors) {
  if (descriptor_file_offset > image.size() || sizeof(KD) > image.size() - descriptor_file_offset) {
    errors.emplace_back("ConSan MOI owner derivation descriptor exceeds ELF bytes");
    return std::nullopt;
  }

  KD desc{};
  std::memcpy(&desc, image.data() + descriptor_file_offset, sizeof(desc));
  return moi_descriptor_wavefront_size(desc) == 32 ? 5 : 6;
}

[[nodiscard]] std::optional<uint16_t> moi_kernel_owner_shift(std::span<const uint8_t> image,
                                                             const ConSanKernelInfo &kernel,
                                                             std::vector<std::string> &errors) {
  return moi_descriptor_owner_shift(image, kernel.descriptor_file_offset, errors);
}

[[nodiscard]] bool grow_moi_descriptor_vgpr_allocation(KD &desc, uint32_t required_count) {
  if (required_count == 0 || required_count > kMaxVgprs)
    return false;
  if (required_count <= moi_descriptor_vgpr_allocation_count(desc))
    return true;

  const uint32_t granularity = std::max<uint32_t>(moi_descriptor_vgpr_granularity(desc), 1u);
  const uint32_t rounded = ((required_count + granularity - 1u) / granularity) * granularity;
  if (rounded > kMaxVgprs)
    return false;
  const uint32_t granulated = rounded / granularity - 1u;
  AMDHSA_BITS_SET(desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                  granulated);
  return true;
}

[[nodiscard]] bool grow_moi_descriptor_sgpr_allocation(KD &desc, uint32_t required_count) {
  if (required_count == 0 || required_count > kMaxSgprs)
    return false;
  if (required_count <= moi_descriptor_sgpr_allocation_count(desc))
    return true;

  constexpr uint32_t kSgprGranularity = 8;
  const uint32_t rounded =
      ((required_count + kSgprGranularity - 1u) / kSgprGranularity) * kSgprGranularity;
  if (rounded > kMaxSgprs)
    return false;
  const uint32_t granulated = rounded / kSgprGranularity - 1u;
  AMDHSA_BITS_SET(desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  granulated);
  return true;
}

[[nodiscard]] bool grow_moi_kernel_descriptor_vgprs(CodeObjectPatcher &patcher,
                                                    std::span<const uint8_t> image,
                                                    const ConSanKernelInfo &kernel,
                                                    uint32_t required_count,
                                                    std::vector<std::string> &errors) {
  if (kernel.descriptor_file_offset > image.size() ||
      sizeof(KD) > image.size() - kernel.descriptor_file_offset) {
    errors.emplace_back("ConSan MOI descriptor VGPR growth exceeds ELF bytes");
    return false;
  }

  KD desc{};
  std::memcpy(&desc, image.data() + kernel.descriptor_file_offset, sizeof(desc));
  if (!grow_moi_descriptor_vgpr_allocation(desc, required_count)) {
    errors.emplace_back("ConSan MOI could not grow descriptor VGPR allocation");
    return false;
  }
  if (!patcher.patch_kernel_descriptor(kernel.descriptor_file_offset,
                                       {reinterpret_cast<const uint8_t *>(&desc), sizeof(desc)})) {
    errors.emplace_back("ConSan MOI could not patch descriptor VGPR allocation");
    return false;
  }
  return true;
}

[[nodiscard]] bool grow_moi_kernel_descriptor_vgprs(CodeObjectPatcher &patcher,
                                                    std::span<const uint8_t> image,
                                                    ConSanResult &result, uint32_t required_count) {
  for (const ConSanKernelInfo &kernel : result.kernels) {
    if (!grow_moi_kernel_descriptor_vgprs(patcher, image, kernel, required_count, result.errors)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool grow_moi_kernel_descriptor_registers(CodeObjectPatcher &patcher,
                                                        std::span<const uint8_t> image,
                                                        ConSanResult &result,
                                                        uint32_t required_vgpr_count,
                                                        uint32_t required_sgpr_count) {
  for (const ConSanKernelInfo &kernel : result.kernels) {
    if (kernel.descriptor_file_offset > image.size() ||
        sizeof(KD) > image.size() - kernel.descriptor_file_offset) {
      result.errors.emplace_back("ConSan MOI descriptor register growth exceeds ELF bytes");
      return false;
    }

    KD desc{};
    std::memcpy(&desc, image.data() + kernel.descriptor_file_offset, sizeof(desc));
    if (!grow_moi_descriptor_vgpr_allocation(desc, required_vgpr_count)) {
      result.errors.emplace_back("ConSan MOI could not grow descriptor VGPR allocation");
      return false;
    }
    if (required_sgpr_count != 0 &&
        !grow_moi_descriptor_sgpr_allocation(desc, required_sgpr_count)) {
      result.errors.emplace_back("ConSan MOI could not grow descriptor SGPR allocation");
      return false;
    }
    if (!patcher.patch_kernel_descriptor(
            kernel.descriptor_file_offset,
            {reinterpret_cast<const uint8_t *>(&desc), sizeof(desc)})) {
      result.errors.emplace_back("ConSan MOI could not patch descriptor register allocation");
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool grow_moi_kernel_descriptor_vgprs(std::vector<uint8_t> &image,
                                                    const ConSanResult &result,
                                                    uint32_t required_count,
                                                    std::vector<std::string> &errors) {
  for (const ConSanKernelInfo &kernel : result.kernels) {
    if (kernel.descriptor_file_offset > image.size() ||
        sizeof(KD) > image.size() - kernel.descriptor_file_offset) {
      errors.emplace_back("ConSan MOI descriptor VGPR growth exceeds ELF bytes");
      return false;
    }

    KD desc{};
    std::memcpy(&desc, image.data() + kernel.descriptor_file_offset, sizeof(desc));
    if (!grow_moi_descriptor_vgpr_allocation(desc, required_count)) {
      errors.emplace_back("ConSan MOI could not grow descriptor VGPR allocation");
      return false;
    }
    std::memcpy(image.data() + kernel.descriptor_file_offset, &desc, sizeof(desc));
  }
  return true;
}

[[nodiscard]] uint32_t first_light_required_vgpr_count(const ConSanOptions &options) {
  uint32_t required_count =
      static_cast<uint32_t>(*options.scratch_vgpr) + (options.moi_dynamic_access_records ? 8u : 3u);
  if (options.moi_owner_vgpr)
    required_count = std::max<uint32_t>(required_count, *options.moi_owner_vgpr + 1u);
  if (options.moi_epoch_vgpr)
    required_count = std::max<uint32_t>(required_count, *options.moi_epoch_vgpr + 1u);
  return required_count;
}

[[nodiscard]] uint32_t direct_sampled_required_vgpr_count(const ConSanOptions &options) {
  uint32_t required_count = static_cast<uint32_t>(*options.scratch_vgpr) + 5u;
  if (options.moi_owner_vgpr)
    required_count = std::max<uint32_t>(required_count, *options.moi_owner_vgpr + 1u);
  if (options.moi_epoch_vgpr)
    required_count = std::max<uint32_t>(required_count, *options.moi_epoch_vgpr + 1u);
  return required_count;
}

[[nodiscard]] const char *moi_delay_mode_name(ConSanDelayMode mode) {
  switch (mode) {
  case ConSanDelayMode::Nop:
    return "nop";
  case ConSanDelayMode::Sleep:
    return "sleep";
  case ConSanDelayMode::SleepVar:
    return "sleep_var";
  }
  return "unknown";
}

[[nodiscard]] bool append_moi_delay_words(std::vector<uint32_t> &words, rj_code_arch_t arch,
                                          const ConSanOptions &options,
                                          std::vector<std::string> &errors,
                                          std::string_view context) {
  if (options.delay_nops == 0)
    return true;

  switch (options.delay_mode) {
  case ConSanDelayMode::Nop:
    for (uint32_t i = 0; i < options.delay_nops; ++i)
      words.push_back(build_s_nop(0, arch));
    return true;
  case ConSanDelayMode::Sleep:
    if (options.delay_nops > std::numeric_limits<uint16_t>::max()) {
      errors.emplace_back(std::string(context) +
                          " sleep delay immediate exceeds the 16-bit s_sleep field");
      return false;
    }
    words.push_back(build_s_sleep(static_cast<uint16_t>(options.delay_nops), arch));
    return true;
  case ConSanDelayMode::SleepVar:
    if (options.delay_var_ssrc > std::numeric_limits<uint8_t>::max()) {
      errors.emplace_back(std::string(context) +
                          " sleep_var source exceeds the 8-bit scalar source field");
      return false;
    }
    words.push_back(build_s_sleep_var(options.delay_var_ssrc, arch));
    return true;
  }

  errors.emplace_back(std::string(context) + " has unknown delay mode '" +
                      moi_delay_mode_name(options.delay_mode) + "'");
  return false;
}

[[nodiscard]] uint32_t inline_shadow_required_vgpr_count(const ConSanOptions &options) {
  const uint32_t scratch_count = options.moi_exec_save_sgpr ? 9u : 7u;
  uint32_t required_count = static_cast<uint32_t>(*options.scratch_vgpr) + scratch_count;
  if (options.moi_owner_vgpr)
    required_count = std::max<uint32_t>(required_count, *options.moi_owner_vgpr + 1u);
  if (options.moi_epoch_vgpr)
    required_count = std::max<uint32_t>(required_count, *options.moi_epoch_vgpr + 1u);
  return required_count;
}

[[nodiscard]] uint16_t inline_shadow_scratch_count(const ConSanOptions &options) {
  return options.moi_exec_save_sgpr ? 9u : 7u;
}

[[nodiscard]] bool validate_inline_shadow_exec_save_sgpr(const ConSanOptions &options,
                                                         std::vector<std::string> &errors) {
  if (!options.moi_exec_save_sgpr)
    return true;
  const uint16_t max_exec_save_sgpr = options.moi_epoch_vgpr ? 98u : 100u;
  if (*options.moi_exec_save_sgpr > max_exec_save_sgpr || *options.moi_exec_save_sgpr % 2u != 0u) {
    errors.emplace_back("ConSan MOI inline-shadow diagnostics require an even "
                        "RJ_CONSAN_MOI_EXEC_SAVE_SGPR in 0.." +
                        std::to_string(max_exec_save_sgpr));
    return false;
  }
  return true;
}

[[nodiscard]] bool reject_optional_scratch_overlap(std::optional<uint16_t> value,
                                                   uint16_t scratch_vgpr,
                                                   std::string_view register_name,
                                                   std::vector<std::string> &errors) {
  if (!value || !overlaps_scratch_triplet(*value, scratch_vgpr))
    return false;
  errors.emplace_back(std::string("ConSan MOI probe scratch VGPRs overlap the ") +
                      std::string(register_name) + " VGPR");
  return true;
}

[[nodiscard]] std::optional<uint16_t>
candidate_lds_byte_offset_vgpr(const ConSanMoiCandidate &candidate,
                               std::vector<std::string> &errors) {
  if (!candidate.addr_vgpr) {
    errors.emplace_back("ConSan MOI probe requires an LDS address VGPR");
    return std::nullopt;
  }
  return *candidate.addr_vgpr;
}

[[nodiscard]] uint16_t candidate_payload_vgpr_count(const ConSanMoiCandidate &candidate) {
  if (candidate.width_bits == 0)
    return 1;
  return static_cast<uint16_t>(std::max<uint32_t>((candidate.width_bits + 31u) / 32u, 1u));
}

[[nodiscard]] bool reject_candidate_scratch_range_overlap(const ConSanMoiCandidate &candidate,
                                                          uint16_t scratch_vgpr,
                                                          uint16_t scratch_count,
                                                          std::vector<std::string> &errors) {
  auto address_vgpr = candidate_lds_byte_offset_vgpr(candidate, errors);
  if (!address_vgpr)
    return true;
  const uint16_t address_vgpr_count =
      candidate.source == ConSanMoiCandidateSource::NativeLds ? 1u : 2u;
  if (range_overlaps(*address_vgpr, address_vgpr_count, scratch_vgpr, scratch_count)) {
    errors.emplace_back("ConSan MOI probe scratch VGPRs overlap the LDS address VGPRs");
    return true;
  }
  const uint16_t payload_vgpr_count = candidate_payload_vgpr_count(candidate);
  if (candidate.dst_vgpr &&
      range_overlaps(*candidate.dst_vgpr, payload_vgpr_count, scratch_vgpr, scratch_count)) {
    errors.emplace_back("ConSan MOI probe scratch VGPRs overlap the destination VGPRs");
    return true;
  }
  if (candidate.data_vgpr &&
      range_overlaps(*candidate.data_vgpr, payload_vgpr_count, scratch_vgpr, scratch_count)) {
    errors.emplace_back("ConSan MOI probe scratch VGPRs overlap the data VGPRs");
    return true;
  }
  return false;
}

struct MoiKernelResourceContext {
  const ConSanKernelInfo *kernel = nullptr;
  KernelCfgScope scope;
  std::unique_ptr<LivenessAnalysis> liveness;
  uint16_t current_vgpr_count = 0;
  uint16_t max_referenced_vgpr_count = 0;
  uint32_t original_private_segment_size = 0;
  bool descriptor_valid = false;
};

[[nodiscard]] uint16_t max_referenced_vgpr_count(const KernelCfgScope &scope) {
  uint32_t max_count = 0;
  for (BasicBlock *block : scope.blocks) {
    if (block == nullptr)
      continue;
    for (const Instruction &inst : block->instructions()) {
      const InstDefUse du(inst);
      const RegisterSet referenced = du.defs | du.uses;
      referenced.for_each([&](RegisterRef ref) {
        if (ref.cls == RegClass::VGPR)
          max_count = std::max<uint32_t>(max_count, static_cast<uint32_t>(ref.index) + 1u);
      });
    }
  }
  return static_cast<uint16_t>(std::min<uint32_t>(max_count, kMaxVgprs));
}

[[nodiscard]] const Instruction *instruction_at(BasicBlock *block, uint64_t text_offset) {
  if (block == nullptr)
    return nullptr;
  const auto it =
      std::ranges::find_if(block->instructions(), [text_offset](const Instruction &inst) {
        return inst.src_loc() == text_offset;
      });
  return it == block->instructions().end() ? nullptr : &*it;
}

[[nodiscard]] uint16_t moi_access_scratch_vgpr_count(const ConSanOptions &options) {
  switch (options.moi_engine) {
  case ConSanMoiEngine::RecordReplay:
    return options.moi_dynamic_access_records ? 8u : 3u;
  case ConSanMoiEngine::InlineShadow:
    return inline_shadow_scratch_count(options);
  case ConSanMoiEngine::Sampled:
    return 5u;
  }
  return 0;
}

void append_moi_resource_plans(std::span<const uint8_t> bytes, const ConSanOptions &options,
                               rj_code_arch_t arch, ConSanResult &result) {
  result.resource_plans.clear();
  result.resource_plans.reserve(result.moi_candidates.size());
  if (result.moi_candidates.empty())
    return;

  auto append_unavailable = [&](size_t candidate_index, ConSanRegisterPlanReason reason) {
    const ConSanMoiCandidate &candidate = result.moi_candidates[candidate_index];
    ConSanCandidateResourcePlan plan;
    plan.candidate_index = candidate_index;
    plan.text_offset = candidate.text_offset;
    plan.reason = reason;
    result.resource_plans.push_back(std::move(plan));
  };

  AmdGpuCodeObject code_object(bytes.data(), bytes.size());
  std::unique_ptr<Decoder> decoder = Decoder::create(arch);
  if (!code_object.is_valid() || !decoder) {
    for (size_t i = 0; i < result.moi_candidates.size(); ++i)
      append_unavailable(i, ConSanRegisterPlanReason::MissingInstruction);
    return;
  }

  std::vector<uint64_t> leaders;
  leaders.reserve(result.kernels.size() + result.functions.size());
  std::vector<uint64_t> kernel_entries;
  kernel_entries.reserve(result.kernels.size());
  for (const ConSanKernelInfo &kernel : result.kernels) {
    if (!kernel.has_text_range)
      continue;
    leaders.push_back(kernel.entry_text_offset);
    kernel_entries.push_back(kernel.entry_text_offset);
  }
  for (const ConSanFunctionInfo &function : result.functions)
    leaders.push_back(function.entry_text_offset);
  std::ranges::sort(leaders);
  leaders.erase(std::ranges::unique(leaders).begin(), leaders.end());
  std::ranges::sort(kernel_entries);
  kernel_entries.erase(std::ranges::unique(kernel_entries).begin(), kernel_entries.end());

  auto blocks = BasicBlock::build(code_object, *decoder, arch, leaders);
  const BlockOffsetIndex block_index = build_block_offset_index(blocks);
  CodeObjectPatcher patcher(code_object);
  const std::span<const uint8_t> text = patcher.text_bytes();

  std::vector<MoiKernelResourceContext> contexts;
  contexts.reserve(result.kernels.size());
  for (const ConSanKernelInfo &kernel : result.kernels) {
    if (!kernel.has_text_range)
      continue;
    auto scope = build_kernel_cfg_scope(blocks, block_index,
                                        KernelScopeRequest{.entry_offset = kernel.entry_text_offset,
                                                           .additional_entry_offsets = {}},
                                        kernel_entries, text);
    if (!scope)
      continue;

    MoiKernelResourceContext new_context;
    new_context.kernel = &kernel;
    new_context.scope = std::move(*scope);
    contexts.push_back(std::move(new_context));
    MoiKernelResourceContext &context = contexts.back();
    context.liveness =
        std::make_unique<LivenessAnalysis>(KernelBlockScope(context.scope.blocks),
                                           LivenessAnalysisOptions{}, context.scope.liveness_edges);
    context.max_referenced_vgpr_count = max_referenced_vgpr_count(context.scope);

    if (kernel.descriptor_file_offset > bytes.size() ||
        sizeof(KD) > bytes.size() - kernel.descriptor_file_offset) {
      continue;
    }
    KD descriptor{};
    std::memcpy(&descriptor, bytes.data() + kernel.descriptor_file_offset, sizeof(descriptor));
    context.current_vgpr_count =
        static_cast<uint16_t>(moi_descriptor_vgpr_allocation_count(descriptor));
    context.original_private_segment_size = descriptor.private_segment_fixed_size;
    context.descriptor_valid = true;
  }

  const uint16_t scratch_count = moi_access_scratch_vgpr_count(options);
  for (size_t candidate_index = 0; candidate_index < result.moi_candidates.size();
       ++candidate_index) {
    const ConSanMoiCandidate &candidate = result.moi_candidates[candidate_index];
    ConSanCandidateResourcePlan candidate_plan;
    candidate_plan.candidate_index = candidate_index;
    candidate_plan.text_offset = candidate.text_offset;
    candidate_plan.scratch_vgpr_count = scratch_count;

    BasicBlock *anchor_block = block_for_offset(block_index, candidate.text_offset);
    const Instruction *anchor = instruction_at(anchor_block, candidate.text_offset);
    if (anchor == nullptr) {
      candidate_plan.reason = ConSanRegisterPlanReason::MissingInstruction;
      result.resource_plans.push_back(std::move(candidate_plan));
      continue;
    }

    std::vector<MoiKernelResourceContext *> owners;
    for (MoiKernelResourceContext &context : contexts) {
      if (candidate.kernel_descriptor_file_offset &&
          context.kernel->descriptor_file_offset != *candidate.kernel_descriptor_file_offset) {
        continue;
      }
      if (std::ranges::find(context.scope.blocks, anchor_block) == context.scope.blocks.end())
        continue;
      owners.push_back(&context);
      candidate_plan.owner_descriptor_file_offsets.push_back(
          context.kernel->descriptor_file_offset);
    }

    if (owners.empty()) {
      candidate_plan.reason = ConSanRegisterPlanReason::MissingOwner;
      result.resource_plans.push_back(std::move(candidate_plan));
      continue;
    }
    if (owners.size() != 1) {
      candidate_plan.reason = ConSanRegisterPlanReason::AmbiguousOwners;
      result.resource_plans.push_back(std::move(candidate_plan));
      continue;
    }

    MoiKernelResourceContext &owner = *owners.front();
    candidate_plan.current_vgpr_count = owner.current_vgpr_count;
    candidate_plan.max_referenced_vgpr_count = owner.max_referenced_vgpr_count;
    candidate_plan.required_vgpr_count = owner.current_vgpr_count;
    candidate_plan.original_private_segment_size = owner.original_private_segment_size;
    if (!owner.descriptor_valid) {
      candidate_plan.reason = ConSanRegisterPlanReason::InvalidDescriptor;
      result.resource_plans.push_back(std::move(candidate_plan));
      continue;
    }

    const InstDefUse anchor_def_use(*anchor);
    ConSanRegisterRequest request;
    request.reg_class = RegClass::VGPR;
    request.count = scratch_count;
    request.current_allocation_count = owner.current_vgpr_count;
    request.max_referenced_count = owner.max_referenced_vgpr_count;
    request.architecture_limit = kMaxVgprs;
    request.explicit_base = options.scratch_vgpr;
    request.forbidden = anchor_def_use.defs | anchor_def_use.uses;
    if (options.moi_owner_vgpr)
      request.forbidden.expand({RegClass::VGPR, *options.moi_owner_vgpr, 1});
    if (options.moi_epoch_vgpr)
      request.forbidden.expand({RegClass::VGPR, *options.moi_epoch_vgpr, 1});

    const ConSanRegisterPlan register_plan =
        plan_consan_registers(request, owner.liveness->live_before(*anchor));
    candidate_plan.source = register_plan.source;
    candidate_plan.reason = register_plan.reason;
    candidate_plan.scratch_vgpr = register_plan.base;
    candidate_plan.required_vgpr_count = register_plan.required_descriptor_count;
    result.resource_plans.push_back(std::move(candidate_plan));
  }
}

[[nodiscard]] bool append_store_start_cell_from_lds_byte_offset(std::vector<uint32_t> &words,
                                                                uint64_t address,
                                                                uint16_t lds_byte_offset_vgpr,
                                                                uint16_t scratch_vgpr,
                                                                rj_code_arch_t arch) {
  const uint16_t value_vgpr = static_cast<uint16_t>(scratch_vgpr + 2u);
  const auto shift = build_v_lshrrev_b32_e32(
      value_vgpr, scalar_positive_inline_u32(consan_moi_exact_shadow::granule_shift),
      lds_byte_offset_vgpr, arch);
  if (!shift)
    return false;

  words.push_back(*shift);
  return append_store_u32_vgpr(words, address, value_vgpr, scratch_vgpr, arch);
}

[[nodiscard]] bool append_compute_effective_lds_byte_offset(std::vector<uint32_t> &words,
                                                            uint16_t dst_vgpr, uint16_t addr_vgpr,
                                                            uint32_t static_byte_offset,
                                                            rj_code_arch_t arch) {
  if (static_byte_offset == 0)
    return true;

  const auto mov_offset = build_v_mov_b32_e64_literal(dst_vgpr, static_byte_offset, arch);
  const auto add_offset =
      build_v_add_nc_u32_e32(dst_vgpr, vector_source_vgpr(addr_vgpr), dst_vgpr, arch);
  if (!mov_offset || !add_offset)
    return false;

  words.insert(words.end(), mov_offset->begin(), mov_offset->end());
  words.push_back(*add_offset);
  return true;
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_first_light_access_record_words(
    std::span<const uint8_t> bytes, const ConSanMoiCandidate &candidate,
    const ConSanOptions &options, rj_code_arch_t arch, uint32_t record_index, uint32_t record_count,
    uint32_t access_record_capacity, std::vector<std::string> &errors) {
  if (!options.scratch_vgpr) {
    errors.emplace_back("ConSan MOI first-light probe requires RJ_CONSAN_TMP_VGPR");
    return std::nullopt;
  }
  const uint16_t scratch_count = options.moi_dynamic_access_records ? 8u : 3u;
  if (static_cast<uint32_t>(*options.scratch_vgpr) + scratch_count > 256u) {
    errors.emplace_back(options.moi_dynamic_access_records
                            ? "ConSan MOI dynamic access-record probe needs eight scratch VGPRs"
                            : "ConSan MOI first-light probe needs three scratch VGPRs");
    return std::nullopt;
  }
  if (options.moi_dynamic_access_records && !options.moi_exec_save_sgpr) {
    errors.emplace_back(
        "ConSan MOI dynamic access-record probe requires RJ_CONSAN_MOI_EXEC_SAVE_SGPR");
    return std::nullopt;
  }
  if (options.moi_dynamic_access_records &&
      (*options.moi_exec_save_sgpr > 104u || *options.moi_exec_save_sgpr % 2u != 0u)) {
    errors.emplace_back(
        "ConSan MOI dynamic access-record probe requires an even RJ_CONSAN_MOI_EXEC_SAVE_SGPR in "
        "0..104");
    return std::nullopt;
  }
  auto lds_byte_offset_vgpr = candidate_lds_byte_offset_vgpr(candidate, errors);
  if (!lds_byte_offset_vgpr)
    return std::nullopt;
  if (reject_candidate_scratch_range_overlap(candidate, *options.scratch_vgpr, scratch_count,
                                             errors))
    return std::nullopt;
  if (options.moi_dynamic_access_records && has_recent_saveexec(bytes, candidate)) {
    errors.emplace_back(
        "ConSan MOI dynamic access-record probe skipped a candidate immediately after "
        "s_*_saveexec");
    return std::nullopt;
  }
  if (reject_optional_scratch_range_overlap(options.moi_owner_vgpr, *options.scratch_vgpr,
                                            scratch_count, "MOI owner", errors) ||
      reject_optional_scratch_range_overlap(options.moi_epoch_vgpr, *options.scratch_vgpr,
                                            scratch_count, "MOI epoch", errors))
    return std::nullopt;
  auto byte_count = byte_count_for_candidate(candidate);
  if (!byte_count) {
    errors.emplace_back("ConSan MOI first-light probe could not determine LDS byte count");
    return std::nullopt;
  }
  const std::optional<std::vector<ConSanMoiAccessRange>> access_ranges =
      candidate_access_ranges(bytes, candidate);
  if (!access_ranges || access_ranges->empty()) {
    errors.emplace_back("ConSan MOI first-light probe requires a supported LDS access range");
    return std::nullopt;
  }
  std::optional<uint16_t> derived_owner_vgpr;
  std::optional<uint32_t> derived_owner_word;
  if (!options.moi_owner_vgpr && candidate.kernel_descriptor_file_offset) {
    const auto owner_shift =
        moi_descriptor_owner_shift(bytes, *candidate.kernel_descriptor_file_offset, errors);
    if (!owner_shift)
      return std::nullopt;
    const uint16_t value_vgpr = static_cast<uint16_t>(
        *options.scratch_vgpr + (options.moi_dynamic_access_records ? 4u : 2u));
    const auto owner_init = build_v_lshrrev_b32_e32(
        value_vgpr, scalar_positive_inline_u32(*owner_shift), kRdna4WorkitemIdX, arch);
    if (!owner_init) {
      errors.emplace_back("ConSan MOI first-light probe could not encode owner derivation");
      return std::nullopt;
    }
    derived_owner_vgpr = value_vgpr;
    derived_owner_word = *owner_init;
  }
  ConSanMoiWorkgroupSources workgroup_sources;
  if (candidate.kernel_descriptor_file_offset) {
    const auto descriptor_workgroup_sources = moi_descriptor_workgroup_sources(
        bytes, *candidate.kernel_descriptor_file_offset, arch, errors);
    if (!descriptor_workgroup_sources)
      return std::nullopt;
    workgroup_sources = *descriptor_workgroup_sources;
  }

  std::vector<uint32_t> words;
  words.reserve(candidate.size / sizeof(uint32_t) + 1u + 7u * 12u + 9u + 10u + 20u + 24u +
                (options.moi_owner_vgpr || derived_owner_vgpr ? 9u : 0u) +
                (options.moi_epoch_vgpr ? 9u : 0u) + (derived_owner_vgpr ? 1u : 0u));
  if (derived_owner_word)
    words.push_back(*derived_owner_word);
  for (uint64_t offset = 0; offset < candidate.size; offset += sizeof(uint32_t)) {
    uint32_t word = 0;
    std::memcpy(&word, bytes.data() + candidate.file_offset + offset, sizeof(word));
    words.push_back(word);
  }
  words.push_back(kWaitDscnt0);

  const uint64_t base = *options.moi_report_buffer_address;
  const auto kind = consan_moi_shadow_kind_from_access_kind(candidate.kind);
  auto append_effective_range_offset = [&](const ConSanMoiAccessRange &range,
                                           uint16_t value_vgpr) -> std::optional<uint16_t> {
    if (range.static_byte_offset == 0)
      return *lds_byte_offset_vgpr;
    if (!append_compute_effective_lds_byte_offset(words, value_vgpr, *lds_byte_offset_vgpr,
                                                  range.static_byte_offset, arch)) {
      return std::nullopt;
    }
    return value_vgpr;
  };

  if (options.moi_dynamic_access_records) {
    const uint16_t slot_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 2u);
    const uint16_t value_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 5u);
    const uint16_t vcc_lo_save_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 6u);
    const uint16_t vcc_hi_save_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 7u);
    const uint64_t dynamic_record_base = base + sizeof(ConSanMoiReportHeader);

    for (const ConSanMoiAccessRange &range : *access_ranges) {
      words.push_back(build_v_mov_b32_e32(vcc_lo_save_vgpr, kRdna4VccLo, arch));
      words.push_back(
          build_v_mov_b32_e32(vcc_hi_save_vgpr, static_cast<uint16_t>(kRdna4VccLo + 1u), arch));

      if (!append_atomic_fetch_add_one_u32(
              words, base + offsetof(ConSanMoiReportHeader, access_record_count), slot_vgpr,
              *options.scratch_vgpr, arch)) {
        errors.emplace_back(
            "ConSan MOI dynamic access-record probe could not reserve a record slot");
        return std::nullopt;
      }

      const auto mov_capacity =
          build_v_mov_b32_e64_literal(value_vgpr, access_record_capacity, arch);
      const auto slot_in_capacity =
          build_v_cmp_gt_u32_e32_vcc(vector_source_vgpr(value_vgpr), slot_vgpr, arch);
      const auto narrow_exec =
          build_s_and_saveexec_b64(*options.moi_exec_save_sgpr, kRdna4VccLo, arch);
      if (!mov_capacity || !slot_in_capacity || !narrow_exec) {
        errors.emplace_back(
            "ConSan MOI dynamic access-record probe could not encode capacity guard");
        return std::nullopt;
      }
      words.insert(words.end(), mov_capacity->begin(), mov_capacity->end());
      words.push_back(*slot_in_capacity);
      words.push_back(*narrow_exec);

      if ((derived_owner_vgpr &&
           !append_dynamic_access_store_u32_vgpr(
               words, dynamic_record_base + offsetof(ConSanMoiAccessRecord, wave_id),
               *derived_owner_vgpr, slot_vgpr, *options.scratch_vgpr, arch)) ||
          !append_dynamic_access_event_index_store(
              words, base + offsetof(ConSanMoiReportHeader, event_counter),
              dynamic_record_base + offsetof(ConSanMoiAccessRecord, event_index), slot_vgpr,
              *options.scratch_vgpr, arch) ||
          !append_dynamic_access_store_workgroup_source(
              words, dynamic_record_base + offsetof(ConSanMoiAccessRecord, workgroup_x),
              workgroup_sources.x, slot_vgpr, *options.scratch_vgpr, arch) ||
          !append_dynamic_access_store_workgroup_source(
              words, dynamic_record_base + offsetof(ConSanMoiAccessRecord, workgroup_y),
              workgroup_sources.y, slot_vgpr, *options.scratch_vgpr, arch) ||
          !append_dynamic_access_store_workgroup_source(
              words, dynamic_record_base + offsetof(ConSanMoiAccessRecord, workgroup_z),
              workgroup_sources.z, slot_vgpr, *options.scratch_vgpr, arch) ||
          !append_dynamic_access_store_u32_scalar_src(
              words, dynamic_record_base + offsetof(ConSanMoiAccessRecord, lane_mask),
              *options.moi_exec_save_sgpr, slot_vgpr, *options.scratch_vgpr, arch) ||
          !append_dynamic_access_store_u32_scalar_src(
              words,
              dynamic_record_base + offsetof(ConSanMoiAccessRecord, lane_mask) + sizeof(uint32_t),
              static_cast<uint16_t>(*options.moi_exec_save_sgpr + 1u), slot_vgpr,
              *options.scratch_vgpr, arch) ||
          (options.moi_owner_vgpr &&
           !append_dynamic_access_store_u32_vgpr(
               words, dynamic_record_base + offsetof(ConSanMoiAccessRecord, wave_id),
               *options.moi_owner_vgpr, slot_vgpr, *options.scratch_vgpr, arch)) ||
          (options.moi_epoch_vgpr &&
           !append_dynamic_access_store_u32_vgpr(
               words, dynamic_record_base + offsetof(ConSanMoiAccessRecord, epoch),
               *options.moi_epoch_vgpr, slot_vgpr, *options.scratch_vgpr, arch)) ||
          !append_dynamic_access_store_u32_literal(
              words, dynamic_record_base + offsetof(ConSanMoiAccessRecord, instruction_offset),
              static_cast<uint32_t>(candidate.text_offset), slot_vgpr, *options.scratch_vgpr,
              arch) ||
          !append_dynamic_access_store_u32_literal(
              words, dynamic_record_base + offsetof(ConSanMoiAccessRecord, access_kind),
              static_cast<uint32_t>(kind), slot_vgpr, *options.scratch_vgpr, arch)) {
        errors.emplace_back(
            "ConSan MOI dynamic access-record probe could not encode record stores");
        return std::nullopt;
      }

      const std::optional<uint16_t> effective_lds_byte_offset_vgpr =
          append_effective_range_offset(range, value_vgpr);
      if (!effective_lds_byte_offset_vgpr) {
        errors.emplace_back(
            "ConSan MOI dynamic access-record probe could not encode LDS byte offset");
        return std::nullopt;
      }
      if (!append_dynamic_access_store_u32_vgpr(
              words, dynamic_record_base + offsetof(ConSanMoiAccessRecord, lds_byte_offset),
              *effective_lds_byte_offset_vgpr, slot_vgpr, *options.scratch_vgpr, arch)) {
        errors.emplace_back("ConSan MOI dynamic access-record probe could not encode range fields");
        return std::nullopt;
      }
      const auto start_cell_shift = build_v_lshrrev_b32_e32(
          value_vgpr, scalar_positive_inline_u32(consan_moi_exact_shadow::granule_shift),
          *effective_lds_byte_offset_vgpr, arch);
      if (!start_cell_shift) {
        errors.emplace_back("ConSan MOI dynamic access-record probe could not encode start cell");
        return std::nullopt;
      }
      words.push_back(*start_cell_shift);
      const ConSanMoiLdsCellRange static_range =
          consan_moi_lds_cell_range_for_bytes(range.static_byte_offset, range.byte_count);
      if (!append_dynamic_access_store_u32_vgpr(
              words, dynamic_record_base + offsetof(ConSanMoiAccessRecord, start_cell), value_vgpr,
              slot_vgpr, *options.scratch_vgpr, arch) ||
          !append_dynamic_access_store_u32_literal(
              words, dynamic_record_base + offsetof(ConSanMoiAccessRecord, lds_byte_count),
              range.byte_count, slot_vgpr, *options.scratch_vgpr, arch) ||
          !append_dynamic_access_store_u32_literal(
              words, dynamic_record_base + offsetof(ConSanMoiAccessRecord, cell_count),
              static_range.cell_count, slot_vgpr, *options.scratch_vgpr, arch)) {
        errors.emplace_back("ConSan MOI dynamic access-record probe could not encode range fields");
        return std::nullopt;
      }
      const auto restore_exec = build_s_mov_b64(kRdna4ExecLo, *options.moi_exec_save_sgpr, arch);
      const auto restore_vcc_lo = build_v_readfirstlane_b32(kRdna4VccLo, vcc_lo_save_vgpr, arch);
      const auto restore_vcc_hi = build_v_readfirstlane_b32(static_cast<uint16_t>(kRdna4VccLo + 1u),
                                                            vcc_hi_save_vgpr, arch);
      if (!restore_exec || !restore_vcc_lo || !restore_vcc_hi) {
        errors.emplace_back("ConSan MOI dynamic access-record probe could not restore EXEC/VCC");
        return std::nullopt;
      }
      words.push_back(*restore_exec);
      words.push_back(*restore_vcc_lo);
      words.push_back(*restore_vcc_hi);
      words.push_back(kWaitAluDepctrSaSdst0);
    }
    return words;
  }

  for (size_t range_index = 0; range_index < access_ranges->size(); ++range_index) {
    const ConSanMoiAccessRange &range = (*access_ranges)[range_index];
    const uint64_t access_record_base =
        base + sizeof(ConSanMoiReportHeader) +
        (static_cast<uint64_t>(record_index) + range_index) * sizeof(ConSanMoiAccessRecord);
    if ((derived_owner_vgpr &&
         !append_store_u32_vgpr(words,
                                access_record_base + offsetof(ConSanMoiAccessRecord, wave_id),
                                *derived_owner_vgpr, *options.scratch_vgpr, arch)) ||
        !append_atomic_event_index_store(
            words, base + offsetof(ConSanMoiReportHeader, event_counter),
            access_record_base + offsetof(ConSanMoiAccessRecord, event_index),
            *options.scratch_vgpr, arch) ||
        !append_store_u32_literal(words,
                                  base + offsetof(ConSanMoiReportHeader, access_record_count),
                                  record_count, *options.scratch_vgpr, arch) ||
        !append_store_workgroup_source(
            words, access_record_base + offsetof(ConSanMoiAccessRecord, workgroup_x),
            workgroup_sources.x, *options.scratch_vgpr, arch) ||
        !append_store_workgroup_source(
            words, access_record_base + offsetof(ConSanMoiAccessRecord, workgroup_y),
            workgroup_sources.y, *options.scratch_vgpr, arch) ||
        !append_store_workgroup_source(
            words, access_record_base + offsetof(ConSanMoiAccessRecord, workgroup_z),
            workgroup_sources.z, *options.scratch_vgpr, arch) ||
        !append_store_u32_scalar_src(
            words, access_record_base + offsetof(ConSanMoiAccessRecord, lane_mask), kRdna4ExecLo,
            *options.scratch_vgpr, arch) ||
        !append_store_u32_scalar_src(
            words,
            access_record_base + offsetof(ConSanMoiAccessRecord, lane_mask) + sizeof(uint32_t),
            kRdna4ExecHi, *options.scratch_vgpr, arch) ||
        (options.moi_owner_vgpr &&
         !append_store_u32_vgpr(words,
                                access_record_base + offsetof(ConSanMoiAccessRecord, wave_id),
                                *options.moi_owner_vgpr, *options.scratch_vgpr, arch)) ||
        (options.moi_epoch_vgpr &&
         !append_store_u32_vgpr(words, access_record_base + offsetof(ConSanMoiAccessRecord, epoch),
                                *options.moi_epoch_vgpr, *options.scratch_vgpr, arch)) ||
        !append_store_u32_literal(
            words, access_record_base + offsetof(ConSanMoiAccessRecord, instruction_offset),
            static_cast<uint32_t>(candidate.text_offset), *options.scratch_vgpr, arch) ||
        !append_store_u32_literal(words,
                                  access_record_base + offsetof(ConSanMoiAccessRecord, access_kind),
                                  static_cast<uint32_t>(kind), *options.scratch_vgpr, arch)) {
      errors.emplace_back("ConSan MOI first-light probe could not encode record stores");
      return std::nullopt;
    }

    const std::optional<uint16_t> effective_lds_byte_offset_vgpr =
        append_effective_range_offset(range, static_cast<uint16_t>(*options.scratch_vgpr + 2u));
    if (!effective_lds_byte_offset_vgpr) {
      errors.emplace_back("ConSan MOI first-light probe could not encode LDS byte offset");
      return std::nullopt;
    }
    const ConSanMoiLdsCellRange static_range =
        consan_moi_lds_cell_range_for_bytes(range.static_byte_offset, range.byte_count);
    if (!append_store_u32_vgpr(
            words, access_record_base + offsetof(ConSanMoiAccessRecord, lds_byte_offset),
            *effective_lds_byte_offset_vgpr, *options.scratch_vgpr, arch) ||
        !append_store_start_cell_from_lds_byte_offset(
            words, access_record_base + offsetof(ConSanMoiAccessRecord, start_cell),
            *effective_lds_byte_offset_vgpr, *options.scratch_vgpr, arch) ||
        !append_store_u32_literal(
            words, access_record_base + offsetof(ConSanMoiAccessRecord, lds_byte_count),
            range.byte_count, *options.scratch_vgpr, arch) ||
        !append_store_u32_literal(words,
                                  access_record_base + offsetof(ConSanMoiAccessRecord, cell_count),
                                  static_range.cell_count, *options.scratch_vgpr, arch)) {
      errors.emplace_back("ConSan MOI first-light probe could not encode range fields");
      return std::nullopt;
    }
  }

  return words;
}

[[nodiscard]] bool append_add_shifted_vgpr_field(std::vector<uint32_t> &words,
                                                 uint16_t destination_vgpr, uint16_t field_vgpr,
                                                 uint16_t shift, uint16_t tmp_vgpr,
                                                 rj_code_arch_t arch) {
  const auto shift_word =
      build_v_lshlrev_b32_e32(tmp_vgpr, scalar_positive_inline_u32(shift), field_vgpr, arch);
  const auto add_word = build_v_add_nc_u32_e32(
      destination_vgpr, vector_source_vgpr(destination_vgpr), tmp_vgpr, arch);
  if (!shift_word || !add_word)
    return false;
  words.push_back(*shift_word);
  words.push_back(*add_word);
  return true;
}

[[nodiscard]] bool append_add_literal_field(std::vector<uint32_t> &words, uint16_t destination_vgpr,
                                            uint32_t value, uint16_t tmp_vgpr,
                                            rj_code_arch_t arch) {
  if (value == 0)
    return true;
  const auto mov_value = build_v_mov_b32_e64_literal(tmp_vgpr, value, arch);
  const auto add_word = build_v_add_nc_u32_e32(
      destination_vgpr, vector_source_vgpr(destination_vgpr), tmp_vgpr, arch);
  if (!mov_value || !add_word)
    return false;
  words.insert(words.end(), mov_value->begin(), mov_value->end());
  words.push_back(*add_word);
  return true;
}

[[nodiscard]] bool is_inline_shadow_access_candidate(const ConSanMoiCandidate &candidate,
                                                     std::span<const uint8_t> bytes) {
  if (candidate.source != ConSanMoiCandidateSource::NativeLds)
    return false;
  if (candidate.kind != ConSanLdsAccessKind::Read && candidate.kind != ConSanLdsAccessKind::Write)
    return false;
  if (candidate.size == 0 || candidate.size % sizeof(uint32_t) != 0)
    return false;
  const std::optional<std::vector<ConSanMoiAccessRange>> ranges =
      candidate_access_ranges(bytes, candidate);
  if (!ranges || ranges->empty())
    return false;
  for (const ConSanMoiAccessRange &range : *ranges) {
    const ConSanMoiLdsCellRange cell_range =
        consan_moi_lds_cell_range_for_bytes(range.static_byte_offset, range.byte_count);
    if (cell_range.cell_count == 0)
      return false;
  }
  return true;
}

[[nodiscard]] std::vector<const ConSanMoiCandidate *>
find_inline_shadow_access_candidates(const ConSanResult &result, std::span<const uint8_t> bytes) {
  std::vector<const ConSanMoiCandidate *> candidates;
  for (const ConSanMoiCandidate &candidate : result.moi_candidates) {
    if (is_inline_shadow_access_candidate(candidate, bytes))
      candidates.push_back(&candidate);
  }
  return candidates;
}

[[nodiscard]] bool append_inline_shadow_diagnostic_words(
    std::vector<uint32_t> &words, const ConSanMoiCandidate &candidate, const ConSanOptions &options,
    rj_code_arch_t arch, const ConSanMoiReportBufferLayout &layout, uint16_t old_value_vgpr,
    uint16_t old_value_hi_vgpr) {
  if (!options.moi_exec_save_sgpr || layout.diagnostic_capacity == 0)
    return true;

  const uint16_t tmp_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 4u);
  const uint16_t vcc_lo_save_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 7u);
  const uint16_t vcc_hi_save_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 8u);
  const uint64_t report_base = *options.moi_report_buffer_address;
  const uint64_t diagnostic_base = report_base + layout.diagnostic_records_offset;

  words.push_back(build_v_mov_b32_e32(vcc_lo_save_vgpr, kRdna4VccLo, arch));
  words.push_back(
      build_v_mov_b32_e32(vcc_hi_save_vgpr, static_cast<uint16_t>(kRdna4VccLo + 1u), arch));

  const auto mov_zero = build_v_mov_b32_e64_literal(tmp_vgpr, 0, arch);
  const auto prior_nonempty =
      build_v_cmp_gt_u32_e32_vcc(vector_source_vgpr(old_value_vgpr), tmp_vgpr, arch);
  const auto narrow_nonempty =
      build_s_and_saveexec_b64(*options.moi_exec_save_sgpr, kRdna4VccLo, arch);
  if (!mov_zero || !prior_nonempty || !narrow_nonempty)
    return false;
  words.insert(words.end(), mov_zero->begin(), mov_zero->end());
  words.push_back(*prior_nonempty);
  words.push_back(*narrow_nonempty);

  const auto prior_owner_shift = build_v_lshrrev_b32_e32(
      tmp_vgpr, scalar_positive_inline_u32(consan_moi_exact_shadow::owner_shift), old_value_vgpr,
      arch);
  const auto prior_owner_mask =
      build_v_and_b32_e32_literal(tmp_vgpr, consan_moi_exact_shadow::max_owner, tmp_vgpr, arch);
  const auto owner_ne =
      build_v_cmp_ne_u32_e32_vcc(vector_source_vgpr(*options.moi_owner_vgpr), tmp_vgpr, arch);
  const auto narrow_conflict = build_s_and_saveexec_b64(
      static_cast<uint16_t>(*options.moi_exec_save_sgpr + 2u), kRdna4VccLo, arch);
  if (!prior_owner_shift || !prior_owner_mask || !owner_ne || !narrow_conflict)
    return false;
  words.push_back(*prior_owner_shift);
  words.insert(words.end(), prior_owner_mask->begin(), prior_owner_mask->end());
  words.push_back(*owner_ne);
  words.push_back(*narrow_conflict);

  uint16_t next_exec_save_sgpr = static_cast<uint16_t>(*options.moi_exec_save_sgpr + 4u);
  if (options.moi_epoch_vgpr) {
    const auto prior_epoch_shift = build_v_lshrrev_b32_e32(
        tmp_vgpr, scalar_positive_inline_u32(consan_moi_exact_shadow::epoch_shift), old_value_vgpr,
        arch);
    const auto prior_epoch_mask =
        build_v_and_b32_e32_literal(tmp_vgpr, consan_moi_exact_shadow::max_epoch, tmp_vgpr, arch);
    const auto epoch_eq =
        build_v_cmp_eq_u32_e32_vcc(vector_source_vgpr(*options.moi_epoch_vgpr), tmp_vgpr, arch);
    const auto narrow_same_epoch = build_s_and_saveexec_b64(next_exec_save_sgpr, kRdna4VccLo, arch);
    if (!prior_epoch_shift || !prior_epoch_mask || !epoch_eq || !narrow_same_epoch)
      return false;
    words.push_back(*prior_epoch_shift);
    words.insert(words.end(), prior_epoch_mask->begin(), prior_epoch_mask->end());
    words.push_back(*epoch_eq);
    words.push_back(*narrow_same_epoch);
    next_exec_save_sgpr = static_cast<uint16_t>(next_exec_save_sgpr + 2u);
  }

  const auto current_kind = consan_moi_shadow_kind_from_access_kind(candidate.kind);
  if (current_kind != ConSanMoiShadowAccessKind::Write) {
    const auto prior_kind = build_v_and_b32_e32_literal(
        tmp_vgpr, static_cast<uint32_t>(consan_moi_exact_shadow::access_kind_mask), old_value_vgpr,
        arch);
    const auto kind_ne = build_v_cmp_ne_u32_e32_vcc(
        scalar_positive_inline_u32(static_cast<uint32_t>(current_kind)), tmp_vgpr, arch);
    const auto narrow_kind_conflict =
        build_s_and_saveexec_b64(next_exec_save_sgpr, kRdna4VccLo, arch);
    if (!prior_kind || !kind_ne || !narrow_kind_conflict)
      return false;
    words.insert(words.end(), prior_kind->begin(), prior_kind->end());
    words.push_back(*kind_ne);
    words.push_back(*narrow_kind_conflict);
  }

  if (!append_store_u32_literal(words,
                                report_base + offsetof(ConSanMoiReportHeader, diagnostic_count), 1u,
                                *options.scratch_vgpr, arch) ||
      !append_store_u32_literal(words, diagnostic_base + offsetof(ConSanMoiDiagnosticRecord, kind),
                                static_cast<uint32_t>(ConSanMoiDiagnosticKind::AccessConflict),
                                *options.scratch_vgpr, arch) ||
      !append_store_u32_literal(
          words, diagnostic_base + offsetof(ConSanMoiDiagnosticRecord, backend),
          static_cast<uint32_t>(ConSanMoiEngine::InlineShadow), *options.scratch_vgpr, arch) ||
      !append_store_u32_literal(words,
                                diagnostic_base + offsetof(ConSanMoiDiagnosticRecord, generation),
                                1u, *options.scratch_vgpr, arch) ||
      !append_store_u32_literal(words,
                                diagnostic_base + offsetof(ConSanMoiDiagnosticRecord, generation) +
                                    sizeof(uint32_t),
                                0u, *options.scratch_vgpr, arch) ||
      !append_store_u32_vgpr(words,
                             diagnostic_base + offsetof(ConSanMoiDiagnosticRecord, first_owner_id),
                             tmp_vgpr, *options.scratch_vgpr, arch) ||
      !append_store_u32_vgpr(words,
                             diagnostic_base + offsetof(ConSanMoiDiagnosticRecord, second_owner_id),
                             *options.moi_owner_vgpr, *options.scratch_vgpr, arch) ||
      !append_store_u32_literal(
          words, diagnostic_base + offsetof(ConSanMoiDiagnosticRecord, second_instruction_offset),
          static_cast<uint32_t>(candidate.text_offset), *options.scratch_vgpr, arch) ||
      !append_store_u32_literal(
          words, diagnostic_base + offsetof(ConSanMoiDiagnosticRecord, second_access_kind),
          static_cast<uint32_t>(current_kind), *options.scratch_vgpr, arch)) {
    return false;
  }

  const auto prior_inst_shift = build_v_lshrrev_b32_e32(
      tmp_vgpr, scalar_positive_inline_u32(consan_moi_exact_shadow::instruction_offset_shift - 32u),
      old_value_hi_vgpr, arch);
  const auto prior_kind = build_v_and_b32_e32_literal(
      tmp_vgpr, static_cast<uint32_t>(consan_moi_exact_shadow::access_kind_mask), old_value_vgpr,
      arch);
  if (!prior_inst_shift || !prior_kind)
    return false;
  words.push_back(*prior_inst_shift);
  if (!append_store_u32_vgpr(
          words, diagnostic_base + offsetof(ConSanMoiDiagnosticRecord, first_instruction_offset),
          tmp_vgpr, *options.scratch_vgpr, arch)) {
    return false;
  }
  words.insert(words.end(), prior_kind->begin(), prior_kind->end());
  if (!append_store_u32_vgpr(
          words, diagnostic_base + offsetof(ConSanMoiDiagnosticRecord, first_access_kind), tmp_vgpr,
          *options.scratch_vgpr, arch)) {
    return false;
  }
  if (options.moi_epoch_vgpr &&
      !append_store_u32_vgpr(words, diagnostic_base + offsetof(ConSanMoiDiagnosticRecord, epoch),
                             *options.moi_epoch_vgpr, *options.scratch_vgpr, arch)) {
    return false;
  }

  const auto restore_exec = build_s_mov_b64(kRdna4ExecLo, *options.moi_exec_save_sgpr, arch);
  const auto restore_vcc_lo = build_v_readfirstlane_b32(kRdna4VccLo, vcc_lo_save_vgpr, arch);
  const auto restore_vcc_hi =
      build_v_readfirstlane_b32(static_cast<uint16_t>(kRdna4VccLo + 1u), vcc_hi_save_vgpr, arch);
  if (!restore_exec || !restore_vcc_lo || !restore_vcc_hi)
    return false;
  words.push_back(*restore_exec);
  words.push_back(*restore_vcc_lo);
  words.push_back(*restore_vcc_hi);
  words.push_back(kWaitAluDepctrSaSdst0);
  return true;
}

[[nodiscard]] std::optional<std::vector<uint32_t>>
build_inline_shadow_words(std::span<const uint8_t> bytes, const ConSanMoiCandidate &candidate,
                          const ConSanOptions &options, rj_code_arch_t arch,
                          const ConSanMoiReportBufferLayout &layout,
                          std::vector<std::string> &errors) {
  if (!options.scratch_vgpr) {
    errors.emplace_back("ConSan MOI inline-shadow probe requires RJ_CONSAN_TMP_VGPR");
    return std::nullopt;
  }
  const uint16_t scratch_count = inline_shadow_scratch_count(options);
  if (static_cast<uint32_t>(*options.scratch_vgpr) + scratch_count > kMaxVgprs) {
    errors.emplace_back(options.moi_exec_save_sgpr
                            ? "ConSan MOI inline-shadow diagnostics need nine scratch VGPRs"
                            : "ConSan MOI inline-shadow probe needs seven scratch VGPRs");
    return std::nullopt;
  }
  if (!validate_inline_shadow_exec_save_sgpr(options, errors))
    return std::nullopt;
  auto lds_byte_offset_vgpr = candidate_lds_byte_offset_vgpr(candidate, errors);
  if (!lds_byte_offset_vgpr)
    return std::nullopt;
  if (reject_candidate_scratch_range_overlap(candidate, *options.scratch_vgpr, scratch_count,
                                             errors))
    return std::nullopt;
  if (!options.moi_owner_vgpr) {
    errors.emplace_back("ConSan MOI inline-shadow probe requires RJ_CONSAN_MOI_OWNER_VGPR");
    return std::nullopt;
  }
  if (reject_optional_scratch_range_overlap(options.moi_owner_vgpr, *options.scratch_vgpr,
                                            scratch_count, "MOI owner", errors) ||
      reject_optional_scratch_range_overlap(options.moi_epoch_vgpr, *options.scratch_vgpr,
                                            scratch_count, "MOI epoch", errors))
    return std::nullopt;

  const std::optional<std::vector<ConSanMoiAccessRange>> access_ranges =
      candidate_access_ranges(bytes, candidate);
  if (!access_ranges || access_ranges->empty()) {
    errors.emplace_back("ConSan MOI inline-shadow probe requires a supported LDS access range");
    return std::nullopt;
  }

  const uint16_t address_lo_vgpr = *options.scratch_vgpr;
  const uint16_t address_hi_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 1u);
  const uint16_t low_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 2u);
  const uint16_t high_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 3u);
  const uint16_t tmp_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 4u);
  const uint16_t old_value_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 5u);

  const auto kind = consan_moi_shadow_kind_from_access_kind(candidate.kind);
  const uint32_t low_literal =
      static_cast<uint32_t>(kind) | (1u << consan_moi_exact_shadow::generation_shift);
  const uint32_t high_literal = static_cast<uint32_t>(
      (candidate.text_offset & consan_moi_exact_shadow::max_instruction_offset)
      << (consan_moi_exact_shadow::instruction_offset_shift - 32u));
  const uint64_t exact_shadow_base =
      *options.moi_report_buffer_address + layout.exact_shadow_entries_offset;

  std::vector<uint32_t> words;
  words.reserve(candidate.size / sizeof(uint32_t) + 64u + (options.moi_exec_save_sgpr ? 120u : 0u) +
                (options.moi_epoch_vgpr ? 2u : 0u));
  for (uint64_t offset = 0; offset < candidate.size; offset += sizeof(uint32_t)) {
    uint32_t word = 0;
    std::memcpy(&word, bytes.data() + candidate.file_offset + offset, sizeof(word));
    words.push_back(word);
  }
  words.push_back(kWaitDscnt0);

  const auto mov_low = build_v_mov_b32_e64_literal(low_vgpr, low_literal, arch);
  const auto mov_high = build_v_mov_b32_e64_literal(high_vgpr, high_literal, arch);
  if (!mov_low || !mov_high) {
    errors.emplace_back("ConSan MOI inline-shadow probe could not encode exact-shadow literals");
    return std::nullopt;
  }
  words.insert(words.end(), mov_low->begin(), mov_low->end());
  if (!append_add_shifted_vgpr_field(words, low_vgpr, *options.moi_owner_vgpr,
                                     consan_moi_exact_shadow::owner_shift, tmp_vgpr, arch)) {
    errors.emplace_back("ConSan MOI inline-shadow probe could not encode owner field");
    return std::nullopt;
  }
  if (options.moi_epoch_vgpr &&
      !append_add_shifted_vgpr_field(words, low_vgpr, *options.moi_epoch_vgpr,
                                     consan_moi_exact_shadow::epoch_shift, tmp_vgpr, arch)) {
    errors.emplace_back("ConSan MOI inline-shadow probe could not encode epoch field");
    return std::nullopt;
  }
  words.insert(words.end(), mov_high->begin(), mov_high->end());

  for (const ConSanMoiAccessRange &range : *access_ranges) {
    const ConSanMoiLdsCellRange cell_range =
        consan_moi_lds_cell_range_for_bytes(range.static_byte_offset, range.byte_count);
    for (uint32_t cell_index = 0; cell_index < cell_range.cell_count; ++cell_index) {
      const auto mov_address_lo = build_v_mov_b32_e64_literal(
          address_lo_vgpr, static_cast<uint32_t>(exact_shadow_base), arch);
      const auto mov_address_hi = build_v_mov_b32_e64_literal(
          address_hi_vgpr, static_cast<uint32_t>(exact_shadow_base >> 32u), arch);
      if (!mov_address_lo || !mov_address_hi) {
        errors.emplace_back("ConSan MOI inline-shadow probe could not encode shadow address");
        return std::nullopt;
      }
      words.insert(words.end(), mov_address_lo->begin(), mov_address_lo->end());
      words.insert(words.end(), mov_address_hi->begin(), mov_address_hi->end());

      uint16_t effective_lds_byte_offset_vgpr = *lds_byte_offset_vgpr;
      if (range.static_byte_offset != 0) {
        if (!append_compute_effective_lds_byte_offset(words, tmp_vgpr, *lds_byte_offset_vgpr,
                                                      range.static_byte_offset, arch)) {
          errors.emplace_back("ConSan MOI inline-shadow probe could not encode LDS byte offset");
          return std::nullopt;
        }
        effective_lds_byte_offset_vgpr = tmp_vgpr;
      }

      const auto start_cell_shift = build_v_lshrrev_b32_e32(
          tmp_vgpr, scalar_positive_inline_u32(consan_moi_exact_shadow::granule_shift),
          effective_lds_byte_offset_vgpr, arch);
      const auto byte_index_shift =
          build_v_lshlrev_b32_e32(tmp_vgpr, scalar_positive_inline_u32(3), tmp_vgpr, arch);
      const auto address_with_cell = build_v_add_nc_u32_e32(
          address_lo_vgpr, vector_source_vgpr(address_lo_vgpr), tmp_vgpr, arch);
      if (!start_cell_shift || !byte_index_shift || !address_with_cell) {
        errors.emplace_back("ConSan MOI inline-shadow probe could not encode shadow publish");
        return std::nullopt;
      }
      words.push_back(*start_cell_shift);
      words.push_back(*byte_index_shift);
      words.push_back(*address_with_cell);
      if (cell_index != 0 &&
          !append_add_literal_field(words, address_lo_vgpr,
                                    static_cast<uint32_t>(cell_index * sizeof(uint64_t)), tmp_vgpr,
                                    arch)) {
        errors.emplace_back("ConSan MOI inline-shadow probe could not encode shadow cell offset");
        return std::nullopt;
      }
      const auto atomic_swap = build_flat_atomic_swap_b64_vaddr_vsrc_vdst(
          address_lo_vgpr, low_vgpr, old_value_vgpr, /*return_old_value=*/true, kRdna4ScopeDevice,
          arch);
      if (!atomic_swap) {
        errors.emplace_back("ConSan MOI inline-shadow probe could not encode shadow publish");
        return std::nullopt;
      }
      words.insert(words.end(), atomic_swap->begin(), atomic_swap->end());
      words.push_back(kWaitLoadcnt0);
      if (!append_inline_shadow_diagnostic_words(words, candidate, options, arch, layout,
                                                 old_value_vgpr,
                                                 static_cast<uint16_t>(old_value_vgpr + 1u))) {
        errors.emplace_back("ConSan MOI inline-shadow probe could not encode diagnostic stores");
        return std::nullopt;
      }
    }
  }
  return words;
}

void try_apply_inline_shadow_patch(std::span<const uint8_t> bytes, const ConSanOptions &options,
                                   rj_code_arch_t arch, ConSanResult &result) {
  if (!options.moi_report_buffer_address)
    return;
  if (arch != ROCJITSU_CODE_ARCH_RDNA4) {
    result.warnings.emplace_back("ConSan MOI inline-shadow probe currently supports only RDNA4");
    return;
  }

  const ConSanMoiReportBufferLayout layout =
      consan_moi_inline_shadow_report_buffer_layout_for_bytes(options.moi_report_buffer_size);
  if (options.max_patches == 0)
    return;
  if (layout.exact_shadow_entry_capacity < kConSanMoiInlineShadowConservativeExactShadowEntries) {
    result.warnings.emplace_back(
        "ConSan MOI inline-shadow probe requires exact-shadow capacity for the full 64 KiB "
        "LDS address range");
    return;
  }
  if (!options.scratch_vgpr) {
    result.warnings.emplace_back("ConSan MOI inline-shadow probe requires RJ_CONSAN_TMP_VGPR");
    return;
  }
  const uint16_t scratch_count = inline_shadow_scratch_count(options);
  if (static_cast<uint32_t>(*options.scratch_vgpr) + scratch_count > kMaxVgprs) {
    result.warnings.emplace_back(options.moi_exec_save_sgpr
                                     ? "ConSan MOI inline-shadow diagnostics need nine scratch "
                                       "VGPRs"
                                     : "ConSan MOI inline-shadow probe needs seven scratch VGPRs");
    return;
  }
  if (!validate_inline_shadow_exec_save_sgpr(options, result.warnings))
    return;
  if (!options.moi_owner_vgpr) {
    result.warnings.emplace_back(
        "ConSan MOI inline-shadow probe requires RJ_CONSAN_MOI_OWNER_VGPR");
    return;
  }

  std::vector<const ConSanMoiCandidate *> candidates =
      find_inline_shadow_access_candidates(result, bytes);
  if (candidates.empty()) {
    result.warnings.emplace_back(
        "ConSan MOI inline-shadow probe found no supported native LDS load/store candidate");
    return;
  }

  struct PlannedInlineShadowPatch {
    const ConSanMoiCandidate *candidate = nullptr;
    uint64_t patch_bytes = 0;
    bool use_appended_cave = false;
  };
  std::vector<PlannedInlineShadowPatch> planned_patches;
  std::vector<PlannedInlineShadowPatch> appended_cave_candidates;
  std::vector<std::pair<uint64_t, uint64_t>> patched_ranges;
  planned_patches.reserve(candidates.size());
  AmdGpuCodeObject code_object(bytes.data(), bytes.size());
  uint64_t appended_cave_text_offset = 0;
  if (code_object.text_sections().size() == 1)
    appended_cave_text_offset = code_object.text_sections().front()->size();

  for (const ConSanMoiCandidate *candidate_ptr : candidates) {
    const ConSanMoiCandidate &candidate = *candidate_ptr;
    std::vector<std::string> candidate_errors;
    auto words =
        build_inline_shadow_words(bytes, candidate, options, arch, layout, candidate_errors);
    if (!words) {
      result.warnings.insert(result.warnings.end(), candidate_errors.begin(),
                             candidate_errors.end());
      continue;
    }

    const uint64_t patch_bytes = static_cast<uint64_t>(words->size() * sizeof(uint32_t));
    const uint32_t available_padding =
        count_nop_padding(bytes, candidate.file_offset + candidate.size, arch);
    const uint64_t available_bytes =
        candidate.size + static_cast<uint64_t>(available_padding) * sizeof(uint32_t);
    if (patch_bytes > available_bytes) {
      if (appended_cave_text_offset != 0 && candidate.size >= sizeof(uint32_t)) {
        const uint64_t return_branch_pc = appended_cave_text_offset + patch_bytes;
        if (compute_sopp_branch_simm16(candidate.text_offset, appended_cave_text_offset) &&
            compute_sopp_branch_simm16(return_branch_pc, candidate.text_offset + candidate.size)) {
          appended_cave_candidates.push_back({candidate_ptr, patch_bytes, true});
          continue;
        }
      }
      result.warnings.emplace_back("ConSan MOI inline-shadow probe skipped a native LDS site "
                                   "without enough inline padding or an appended cave");
      continue;
    }
    if (candidate.file_offset > bytes.size() ||
        patch_bytes > bytes.size() - candidate.file_offset) {
      result.errors.emplace_back("ConSan MOI inline-shadow probe exceeds ELF bytes");
      return;
    }
    const uint64_t patch_begin = candidate.file_offset;
    const uint64_t patch_end = patch_begin + patch_bytes;
    const auto overlaps_existing =
        std::any_of(patched_ranges.begin(), patched_ranges.end(),
                    [&](const std::pair<uint64_t, uint64_t> range) {
                      return patch_begin < range.second && range.first < patch_end;
                    });
    if (overlaps_existing) {
      result.warnings.emplace_back(
          "ConSan MOI inline-shadow probe skipped an overlapping patch site");
      continue;
    }

    patched_ranges.emplace_back(patch_begin, patch_end);
    planned_patches.push_back({candidate_ptr, patch_bytes, false});
    if (planned_patches.size() == options.max_patches)
      break;
  }

  for (const PlannedInlineShadowPatch &candidate : appended_cave_candidates) {
    if (planned_patches.size() == options.max_patches)
      break;
    if (candidate.candidate == nullptr)
      continue;
    const uint64_t patch_begin = candidate.candidate->file_offset;
    const uint64_t patch_end = patch_begin + candidate.candidate->size;
    const auto overlaps_existing =
        std::any_of(patched_ranges.begin(), patched_ranges.end(),
                    [&](const std::pair<uint64_t, uint64_t> range) {
                      return patch_begin < range.second && range.first < patch_end;
                    });
    if (overlaps_existing) {
      result.warnings.emplace_back(
          "ConSan MOI inline-shadow probe skipped an overlapping appended-cave patch site");
      continue;
    }
    patched_ranges.emplace_back(patch_begin, patch_end);
    planned_patches.push_back(candidate);
  }

  if (planned_patches.empty())
    return;

  const bool uses_appended_cave =
      std::ranges::any_of(planned_patches, [](const PlannedInlineShadowPatch &patch) {
        return patch.use_appended_cave;
      });
  if (uses_appended_cave) {
    CodeObjectPatcher patcher(code_object);
    if (!grow_moi_kernel_descriptor_vgprs(patcher, bytes, result,
                                          inline_shadow_required_vgpr_count(options))) {
      return;
    }
    const std::span<const uint8_t> old_text = patcher.text_bytes();
    if (old_text.empty()) {
      result.errors.emplace_back("ConSan MOI inline-shadow probe found no .text section");
      return;
    }
    std::vector<uint8_t> new_text(old_text.begin(), old_text.end());
    std::vector<ConSanPatchInfo> patches;
    patches.reserve(planned_patches.size());
    for (const PlannedInlineShadowPatch &planned_patch : planned_patches) {
      const ConSanMoiCandidate &candidate = *planned_patch.candidate;
      auto words =
          build_inline_shadow_words(bytes, candidate, options, arch, layout, result.errors);
      if (!words)
        return;

      ConSanPatchInfo info;
      info.kind = planned_patch.use_appended_cave ? ConSanPatchKind::TrampolineMoiExactShadowStore
                                                  : ConSanPatchKind::InlineMoiExactShadowStore;
      info.anchor_offset = candidate.text_offset;
      info.scratch_vgpr = options.scratch_vgpr;

      if (planned_patch.use_appended_cave) {
        const uint64_t cave_text_offset = static_cast<uint64_t>(new_text.size());
        const auto fwd = compute_sopp_branch_simm16(candidate.text_offset, cave_text_offset);
        const uint64_t return_branch_pc =
            cave_text_offset + static_cast<uint64_t>(words->size()) * sizeof(uint32_t);
        const auto ret =
            compute_sopp_branch_simm16(return_branch_pc, candidate.text_offset + candidate.size);
        if (!fwd || !ret) {
          result.errors.emplace_back(
              "ConSan MOI inline-shadow probe appended cave branch is out of range");
          return;
        }

        std::vector<uint32_t> anchor_words(candidate.size / sizeof(uint32_t), build_s_nop(0, arch));
        if (anchor_words.empty()) {
          result.errors.emplace_back("ConSan MOI inline-shadow probe empty anchor");
          return;
        }
        anchor_words.front() = build_s_branch(*fwd, arch);
        if (candidate.text_offset > new_text.size() ||
            anchor_words.size() * sizeof(uint32_t) > new_text.size() - candidate.text_offset) {
          result.errors.emplace_back("ConSan MOI inline-shadow probe anchor exceeds .text");
          return;
        }
        std::memcpy(new_text.data() + candidate.text_offset, anchor_words.data(),
                    anchor_words.size() * sizeof(uint32_t));

        std::vector<uint32_t> cave_words = *words;
        cave_words.push_back(build_s_branch(*ret, arch));
        append_words_bytes(new_text, cave_words);
        info.trampoline_offset = cave_text_offset;
        info.original_size = candidate.size;
        info.trampoline_size = static_cast<uint32_t>(cave_words.size() * sizeof(uint32_t));
      } else {
        const uint64_t patch_bytes = static_cast<uint64_t>(words->size() * sizeof(uint32_t));
        if (patch_bytes != planned_patch.patch_bytes) {
          result.errors.emplace_back("ConSan MOI inline-shadow probe final patch size changed");
          return;
        }
        if (candidate.text_offset > new_text.size() ||
            patch_bytes > new_text.size() - candidate.text_offset) {
          result.errors.emplace_back("ConSan MOI inline-shadow probe exceeds .text");
          return;
        }
        std::memcpy(new_text.data() + candidate.text_offset, words->data(),
                    static_cast<size_t>(patch_bytes));
        info.trampoline_offset = candidate.text_offset + candidate.size;
        info.original_size = static_cast<uint32_t>(patch_bytes);
        info.trampoline_size = 0;
      }
      patches.push_back(info);
    }

    if (!patcher.replace_text(new_text)) {
      result.errors.emplace_back("ConSan MOI inline-shadow probe could not grow .text");
      return;
    }
    result.elf_bytes = patcher.emit();
    result.patches.insert(result.patches.end(), patches.begin(), patches.end());
    result.modified = true;
    return;
  }

  result.elf_bytes.assign(bytes.begin(), bytes.end());
  for (const PlannedInlineShadowPatch &planned_patch : planned_patches) {
    const ConSanMoiCandidate &candidate = *planned_patch.candidate;
    auto words = build_inline_shadow_words(bytes, candidate, options, arch, layout, result.errors);
    if (!words) {
      result.elf_bytes.clear();
      return;
    }
    const uint64_t patch_bytes = static_cast<uint64_t>(words->size() * sizeof(uint32_t));
    if (patch_bytes != planned_patch.patch_bytes) {
      result.errors.emplace_back("ConSan MOI inline-shadow probe final patch size changed");
      result.elf_bytes.clear();
      return;
    }
    std::memcpy(result.elf_bytes.data() + candidate.file_offset, words->data(),
                static_cast<size_t>(patch_bytes));
  }

  if (!grow_moi_kernel_descriptor_vgprs(
          result.elf_bytes, result, inline_shadow_required_vgpr_count(options), result.errors)) {
    result.elf_bytes.clear();
    return;
  }

  for (const PlannedInlineShadowPatch &planned_patch : planned_patches) {
    const ConSanMoiCandidate &candidate = *planned_patch.candidate;
    ConSanPatchInfo info;
    info.kind = ConSanPatchKind::InlineMoiExactShadowStore;
    info.anchor_offset = candidate.text_offset;
    info.trampoline_offset = candidate.text_offset + candidate.size;
    info.original_size = static_cast<uint32_t>(planned_patch.patch_bytes);
    info.trampoline_size = 0;
    info.scratch_vgpr = options.scratch_vgpr;
    result.patches.push_back(info);
  }

  result.modified = true;
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_direct_sampled_watchpoint_words(
    std::span<const uint8_t> bytes, const ConSanMoiCandidate &candidate,
    const ConSanOptions &options, rj_code_arch_t arch, uint32_t record_index,
    size_t sampled_watchpoints_offset, std::vector<std::string> &errors) {
  if (!options.scratch_vgpr) {
    errors.emplace_back("ConSan MOI sampled probe requires RJ_CONSAN_TMP_VGPR");
    return std::nullopt;
  }
  if (static_cast<uint32_t>(*options.scratch_vgpr) + 5u > kMaxVgprs) {
    errors.emplace_back("ConSan MOI sampled probe needs five scratch VGPRs");
    return std::nullopt;
  }
  auto lds_byte_offset_vgpr = candidate_lds_byte_offset_vgpr(candidate, errors);
  if (!lds_byte_offset_vgpr)
    return std::nullopt;
  if (reject_candidate_scratch_range_overlap(candidate, *options.scratch_vgpr, 5, errors))
    return std::nullopt;
  if (reject_optional_scratch_range_overlap(options.moi_owner_vgpr, *options.scratch_vgpr, 5,
                                            "MOI owner", errors) ||
      reject_optional_scratch_range_overlap(options.moi_epoch_vgpr, *options.scratch_vgpr, 5,
                                            "MOI epoch", errors))
    return std::nullopt;
  auto byte_count = byte_count_for_candidate(candidate);
  if (!byte_count) {
    errors.emplace_back("ConSan MOI sampled probe could not determine LDS byte count");
    return std::nullopt;
  }

  const ConSanMoiLdsCellRange static_range = consan_moi_lds_cell_range_for_bytes(0, *byte_count);
  std::optional<uint16_t> derived_owner_vgpr;
  std::optional<uint32_t> derived_owner_word;
  if (!options.moi_owner_vgpr && candidate.kernel_descriptor_file_offset) {
    const auto owner_shift =
        moi_descriptor_owner_shift(bytes, *candidate.kernel_descriptor_file_offset, errors);
    if (!owner_shift)
      return std::nullopt;
    const uint16_t value_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 4u);
    const auto owner_init = build_v_lshrrev_b32_e32(
        value_vgpr, scalar_positive_inline_u32(*owner_shift), kRdna4WorkitemIdX, arch);
    if (!owner_init) {
      errors.emplace_back("ConSan MOI sampled probe could not encode owner derivation");
      return std::nullopt;
    }
    derived_owner_vgpr = value_vgpr;
    derived_owner_word = *owner_init;
  }

  const uint16_t low_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 2u);
  const uint16_t high_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 3u);
  const uint16_t tmp_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 4u);
  const uint16_t owner_vgpr = options.moi_owner_vgpr.value_or(derived_owner_vgpr.value_or(0));
  const auto kind = consan_moi_shadow_kind_from_access_kind(candidate.kind);
  const uint32_t low_literal = static_cast<uint32_t>(
      consan_moi_sampled_watchpoint::valid_mask |
      (static_cast<uint64_t>(kind) << consan_moi_sampled_watchpoint::access_kind_shift));
  const uint32_t encoded_cell_count = encode_consan_moi_sampled_cell_count(static_range.cell_count)
                                      << (consan_moi_sampled_watchpoint::count_shift - 32u);
  const uint64_t sampled_entry_address = *options.moi_report_buffer_address +
                                         sampled_watchpoints_offset +
                                         static_cast<uint64_t>(record_index) * sizeof(uint64_t);

  std::vector<uint32_t> words;
  words.reserve(candidate.size / sizeof(uint32_t) + 40u + (derived_owner_word ? 1u : 0u) +
                (options.moi_epoch_vgpr ? 2u : 0u));
  if (derived_owner_word)
    words.push_back(*derived_owner_word);
  for (uint64_t offset = 0; offset < candidate.size; offset += sizeof(uint32_t)) {
    uint32_t word = 0;
    std::memcpy(&word, bytes.data() + candidate.file_offset + offset, sizeof(word));
    words.push_back(word);
  }
  words.push_back(kWaitDscnt0);
  if (!append_moi_delay_words(words, arch, options, errors, "ConSan MOI sampled probe"))
    return std::nullopt;

  const auto mov_low = build_v_mov_b32_e64_literal(low_vgpr, low_literal, arch);
  const auto start_cell_shift = build_v_lshrrev_b32_e32(
      tmp_vgpr, scalar_positive_inline_u32(consan_moi_sampled_watchpoint::granule_shift),
      *lds_byte_offset_vgpr, arch);
  const auto high_from_start = build_v_lshlrev_b32_e32(
      high_vgpr, scalar_positive_inline_u32(consan_moi_sampled_watchpoint::start_shift - 32u),
      tmp_vgpr, arch);
  if (!mov_low || !start_cell_shift || !high_from_start) {
    errors.emplace_back("ConSan MOI sampled probe could not encode base sampled entry fields");
    return std::nullopt;
  }
  words.insert(words.end(), mov_low->begin(), mov_low->end());
  if ((options.moi_owner_vgpr || derived_owner_vgpr) &&
      !append_add_shifted_vgpr_field(words, low_vgpr, owner_vgpr,
                                     consan_moi_sampled_watchpoint::owner_shift, tmp_vgpr, arch)) {
    errors.emplace_back("ConSan MOI sampled probe could not encode owner field");
    return std::nullopt;
  }
  if (options.moi_epoch_vgpr &&
      !append_add_shifted_vgpr_field(words, low_vgpr, *options.moi_epoch_vgpr,
                                     consan_moi_sampled_watchpoint::epoch_shift, tmp_vgpr, arch)) {
    errors.emplace_back("ConSan MOI sampled probe could not encode epoch field");
    return std::nullopt;
  }
  words.push_back(*start_cell_shift);
  words.push_back(*high_from_start);
  if (!append_add_literal_field(words, high_vgpr, encoded_cell_count, tmp_vgpr, arch)) {
    errors.emplace_back("ConSan MOI sampled probe could not encode cell-count field");
    return std::nullopt;
  }
  if (!append_store_u32_vgpr(words, sampled_entry_address, low_vgpr, *options.scratch_vgpr, arch) ||
      !append_store_u32_vgpr(words, sampled_entry_address + sizeof(uint32_t), high_vgpr,
                             *options.scratch_vgpr, arch)) {
    errors.emplace_back("ConSan MOI sampled probe could not encode sampled entry stores");
    return std::nullopt;
  }
  return words;
}

void try_apply_direct_sampled_watchpoint_patch(std::span<const uint8_t> bytes,
                                               const ConSanOptions &options, rj_code_arch_t arch,
                                               ConSanResult &result) {
  if (!options.moi_report_buffer_address)
    return;
  if (arch != ROCJITSU_CODE_ARCH_RDNA4) {
    result.warnings.emplace_back("ConSan MOI sampled probe currently supports only RDNA4");
    return;
  }

  const ConSanMoiReportBufferLayout layout =
      consan_moi_direct_sampled_report_buffer_layout_for_bytes(options.moi_report_buffer_size);
  const uint32_t max_candidates = std::min(options.max_patches, layout.sampled_watchpoint_capacity);
  if (max_candidates == 0) {
    result.warnings.emplace_back(
        "ConSan MOI sampled probe requires room for the report header and one sampled entry");
    return;
  }
  if (!options.scratch_vgpr) {
    result.warnings.emplace_back("ConSan MOI sampled probe requires RJ_CONSAN_TMP_VGPR");
    return;
  }
  if (static_cast<uint32_t>(*options.scratch_vgpr) + 5u > kMaxVgprs) {
    result.warnings.emplace_back("ConSan MOI sampled probe needs five scratch VGPRs");
    return;
  }
  if (options.moi_sample_stride == 0) {
    result.warnings.emplace_back("ConSan MOI sampled probe requires a nonzero sample stride");
    return;
  }
  if (options.moi_sample_offset >= options.moi_sample_stride) {
    result.warnings.emplace_back("ConSan MOI sampled probe sample offset must be less than stride");
    return;
  }
  std::vector<const ConSanMoiCandidate *> candidates =
      find_first_light_access_record_candidates(result, bytes);
  if (candidates.empty()) {
    result.warnings.emplace_back(
        "ConSan MOI sampled probe found no native LDS or likely group flat load/store candidate");
    return;
  }

  uint32_t sampled_filter_candidate_count = 0;
  for (uint32_t i = 0; i < candidates.size(); ++i) {
    if (i % options.moi_sample_stride == options.moi_sample_offset)
      ++sampled_filter_candidate_count;
  }

  struct PlannedSampledPatch {
    const ConSanMoiCandidate *candidate = nullptr;
    uint64_t patch_bytes = 0;
    bool use_appended_cave = false;
  };
  std::vector<PlannedSampledPatch> planned_patches;
  std::vector<PlannedSampledPatch> appended_cave_candidates;
  std::vector<std::pair<uint64_t, uint64_t>> patched_ranges;
  planned_patches.reserve(candidates.size());
  AmdGpuCodeObject code_object(bytes.data(), bytes.size());
  uint64_t appended_cave_text_offset = 0;
  if (code_object.text_sections().size() == 1)
    appended_cave_text_offset = code_object.text_sections().front()->size();
  uint32_t sampled_candidate_index = 0;
  for (const ConSanMoiCandidate *candidate_ptr : candidates) {
    const uint32_t current_sample_index = sampled_candidate_index++;
    if (current_sample_index % options.moi_sample_stride != options.moi_sample_offset)
      continue;

    const ConSanMoiCandidate &candidate = *candidate_ptr;
    std::vector<std::string> candidate_errors;
    auto words = build_direct_sampled_watchpoint_words(
        bytes, candidate, options, arch, 0, layout.sampled_watchpoints_offset, candidate_errors);
    if (!words) {
      result.warnings.insert(result.warnings.end(), candidate_errors.begin(),
                             candidate_errors.end());
      continue;
    }

    const uint64_t patch_bytes = static_cast<uint64_t>(words->size() * sizeof(uint32_t));
    const uint32_t available_padding =
        count_nop_padding(bytes, candidate.file_offset + candidate.size, arch);
    const uint64_t available_bytes =
        candidate.size + static_cast<uint64_t>(available_padding) * sizeof(uint32_t);
    if (patch_bytes > available_bytes) {
      if (appended_cave_text_offset != 0 && candidate.size >= sizeof(uint32_t)) {
        const uint64_t return_branch_pc = appended_cave_text_offset + patch_bytes;
        if (compute_sopp_branch_simm16(candidate.text_offset, appended_cave_text_offset) &&
            compute_sopp_branch_simm16(return_branch_pc, candidate.text_offset + candidate.size)) {
          appended_cave_candidates.push_back({candidate_ptr, patch_bytes, true});
          continue;
        }
      }
      result.warnings.emplace_back("ConSan MOI sampled probe skipped a supported load/store site "
                                   "without enough inline padding or an appended cave");
      continue;
    }
    if (candidate.file_offset > bytes.size() ||
        patch_bytes > bytes.size() - candidate.file_offset) {
      result.errors.emplace_back("ConSan MOI sampled probe exceeds ELF bytes");
      return;
    }
    const uint64_t patch_begin = candidate.file_offset;
    const uint64_t patch_end = patch_begin + patch_bytes;
    const auto overlaps_existing =
        std::any_of(patched_ranges.begin(), patched_ranges.end(),
                    [&](const std::pair<uint64_t, uint64_t> range) {
                      return patch_begin < range.second && range.first < patch_end;
                    });
    if (overlaps_existing) {
      result.warnings.emplace_back("ConSan MOI sampled probe skipped an overlapping patch site");
      continue;
    }

    patched_ranges.emplace_back(patch_begin, patch_end);
    planned_patches.push_back({candidate_ptr, patch_bytes, false});
    if (planned_patches.size() == max_candidates)
      break;
  }

  for (const PlannedSampledPatch &candidate : appended_cave_candidates) {
    if (planned_patches.size() == max_candidates)
      break;
    if (candidate.candidate == nullptr)
      continue;
    const uint64_t patch_begin = candidate.candidate->file_offset;
    const uint64_t patch_end = patch_begin + candidate.candidate->size;
    const auto overlaps_existing =
        std::any_of(patched_ranges.begin(), patched_ranges.end(),
                    [&](const std::pair<uint64_t, uint64_t> range) {
                      return patch_begin < range.second && range.first < patch_end;
                    });
    if (overlaps_existing) {
      result.warnings.emplace_back(
          "ConSan MOI sampled probe skipped an overlapping appended-cave patch site");
      continue;
    }
    patched_ranges.emplace_back(patch_begin, patch_end);
    planned_patches.push_back(candidate);
  }

  if (planned_patches.empty()) {
    if (options.moi_sample_stride != 1 || options.moi_sample_offset != 0) {
      result.warnings.emplace_back(
          "ConSan MOI sampled probe selected no candidate after applying the sample filter");
    }
    return;
  }

  if (planned_patches.size() == max_candidates &&
      sampled_filter_candidate_count > planned_patches.size()) {
    result.warnings.emplace_back(
        "ConSan MOI sampled probe limited sampled patches to " +
        std::to_string(planned_patches.size()) + " of " +
        std::to_string(sampled_filter_candidate_count) +
        " sample-filter-selected candidates by report capacity or max-patch budget");
  }

  const bool uses_appended_cave = std::ranges::any_of(
      planned_patches, [](const PlannedSampledPatch &patch) { return patch.use_appended_cave; });
  if (uses_appended_cave) {
    CodeObjectPatcher patcher(code_object);
    if (!grow_moi_kernel_descriptor_vgprs(patcher, bytes, result,
                                          direct_sampled_required_vgpr_count(options))) {
      return;
    }
    const std::span<const uint8_t> old_text = patcher.text_bytes();
    if (old_text.empty()) {
      result.errors.emplace_back("ConSan MOI sampled probe found no .text section");
      return;
    }
    std::vector<uint8_t> new_text(old_text.begin(), old_text.end());
    std::vector<ConSanPatchInfo> patches;
    patches.reserve(planned_patches.size());
    for (uint32_t i = 0; i < planned_patches.size(); ++i) {
      const PlannedSampledPatch &planned_patch = planned_patches[i];
      const ConSanMoiCandidate &candidate = *planned_patch.candidate;
      auto words = build_direct_sampled_watchpoint_words(
          bytes, candidate, options, arch, i, layout.sampled_watchpoints_offset, result.errors);
      if (!words)
        return;

      ConSanPatchInfo info;
      info.kind = planned_patch.use_appended_cave
                      ? ConSanPatchKind::TrampolineMoiSampledWatchpointStore
                      : ConSanPatchKind::InlineMoiSampledWatchpointStore;
      info.anchor_offset = candidate.text_offset;
      info.scratch_vgpr = options.scratch_vgpr;

      if (planned_patch.use_appended_cave) {
        const uint64_t cave_text_offset = static_cast<uint64_t>(new_text.size());
        const auto fwd = compute_sopp_branch_simm16(candidate.text_offset, cave_text_offset);
        const uint64_t return_branch_pc =
            cave_text_offset + static_cast<uint64_t>(words->size()) * sizeof(uint32_t);
        const auto ret =
            compute_sopp_branch_simm16(return_branch_pc, candidate.text_offset + candidate.size);
        if (!fwd || !ret) {
          result.errors.emplace_back(
              "ConSan MOI sampled probe appended cave branch is out of range");
          return;
        }

        std::vector<uint32_t> anchor_words(candidate.size / sizeof(uint32_t), build_s_nop(0, arch));
        if (anchor_words.empty()) {
          result.errors.emplace_back("ConSan MOI sampled probe empty anchor");
          return;
        }
        anchor_words.front() = build_s_branch(*fwd, arch);
        if (candidate.text_offset > new_text.size() ||
            anchor_words.size() * sizeof(uint32_t) > new_text.size() - candidate.text_offset) {
          result.errors.emplace_back("ConSan MOI sampled probe anchor exceeds .text");
          return;
        }
        std::memcpy(new_text.data() + candidate.text_offset, anchor_words.data(),
                    anchor_words.size() * sizeof(uint32_t));

        std::vector<uint32_t> cave_words = *words;
        cave_words.push_back(build_s_branch(*ret, arch));
        append_words_bytes(new_text, cave_words);
        info.trampoline_offset = cave_text_offset;
        info.original_size = candidate.size;
        info.trampoline_size = static_cast<uint32_t>(cave_words.size() * sizeof(uint32_t));
      } else {
        const uint64_t patch_bytes = static_cast<uint64_t>(words->size() * sizeof(uint32_t));
        if (patch_bytes != planned_patch.patch_bytes) {
          result.errors.emplace_back("ConSan MOI sampled probe final patch size changed");
          return;
        }
        if (candidate.text_offset > new_text.size() ||
            patch_bytes > new_text.size() - candidate.text_offset) {
          result.errors.emplace_back("ConSan MOI sampled probe exceeds .text");
          return;
        }
        std::memcpy(new_text.data() + candidate.text_offset, words->data(),
                    static_cast<size_t>(patch_bytes));
        info.trampoline_offset = candidate.text_offset + candidate.size;
        info.original_size = static_cast<uint32_t>(patch_bytes);
        info.trampoline_size = 0;
      }
      patches.push_back(info);
    }

    if (!patcher.replace_text(new_text)) {
      result.errors.emplace_back("ConSan MOI sampled probe could not grow .text");
      return;
    }
    result.elf_bytes = patcher.emit();
    result.patches.insert(result.patches.end(), patches.begin(), patches.end());
    result.modified = true;
    return;
  }

  result.elf_bytes.assign(bytes.begin(), bytes.end());
  for (uint32_t i = 0; i < planned_patches.size(); ++i) {
    const PlannedSampledPatch &planned_patch = planned_patches[i];
    const ConSanMoiCandidate &candidate = *planned_patch.candidate;
    auto words = build_direct_sampled_watchpoint_words(
        bytes, candidate, options, arch, i, layout.sampled_watchpoints_offset, result.errors);
    if (!words) {
      result.elf_bytes.clear();
      return;
    }
    const uint64_t patch_bytes = static_cast<uint64_t>(words->size() * sizeof(uint32_t));
    if (patch_bytes != planned_patch.patch_bytes) {
      result.errors.emplace_back("ConSan MOI sampled probe final patch size changed");
      result.elf_bytes.clear();
      return;
    }
    std::memcpy(result.elf_bytes.data() + candidate.file_offset, words->data(),
                static_cast<size_t>(patch_bytes));
  }

  if (!grow_moi_kernel_descriptor_vgprs(
          result.elf_bytes, result, direct_sampled_required_vgpr_count(options), result.errors)) {
    result.elf_bytes.clear();
    return;
  }

  for (const PlannedSampledPatch &planned_patch : planned_patches) {
    const ConSanMoiCandidate &candidate = *planned_patch.candidate;
    ConSanPatchInfo info;
    info.kind = ConSanPatchKind::InlineMoiSampledWatchpointStore;
    info.anchor_offset = candidate.text_offset;
    info.trampoline_offset = candidate.text_offset + candidate.size;
    info.original_size = static_cast<uint32_t>(planned_patch.patch_bytes);
    info.trampoline_size = 0;
    info.scratch_vgpr = options.scratch_vgpr;
    result.patches.push_back(info);
  }

  result.modified = true;
}

void try_apply_first_light_access_record_patch(std::span<const uint8_t> bytes,
                                               const ConSanOptions &options, rj_code_arch_t arch,
                                               ConSanResult &result) {
  if (!options.moi_report_buffer_address)
    return;
  if (arch != ROCJITSU_CODE_ARCH_RDNA4) {
    result.warnings.emplace_back("ConSan MOI first-light probe currently supports only RDNA4");
    return;
  }
  if (options.moi_report_buffer_size < consan_moi_report_buffer_min_bytes(1, 0, 0, 0)) {
    result.warnings.emplace_back(
        "ConSan MOI first-light probe requires room for the report header and one access record");
    return;
  }

  const ConSanMoiReportBufferLayout layout = consan_moi_report_buffer_layout_for_bytes(
      options.moi_report_buffer_size, options.moi_track_barriers, options.moi_track_atomics);
  if (options.max_patches == 0 || layout.access_record_capacity == 0)
    return;
  if (!options.scratch_vgpr) {
    result.warnings.emplace_back("ConSan MOI first-light probe requires RJ_CONSAN_TMP_VGPR");
    return;
  }
  const uint16_t access_scratch_count = options.moi_dynamic_access_records ? 8u : 3u;
  if (static_cast<uint32_t>(*options.scratch_vgpr) + access_scratch_count > 256u) {
    result.warnings.emplace_back(options.moi_dynamic_access_records
                                     ? "ConSan MOI dynamic access-record probe needs eight scratch "
                                       "VGPRs"
                                     : "ConSan MOI first-light probe needs three scratch VGPRs");
    return;
  }
  if (options.moi_dynamic_access_records && !options.moi_exec_save_sgpr) {
    result.warnings.emplace_back(
        "ConSan MOI dynamic access-record probe requires RJ_CONSAN_MOI_EXEC_SAVE_SGPR");
    return;
  }
  if (options.moi_dynamic_access_records &&
      (*options.moi_exec_save_sgpr > 104u || *options.moi_exec_save_sgpr % 2u != 0u)) {
    result.warnings.emplace_back(
        "ConSan MOI dynamic access-record probe requires an even RJ_CONSAN_MOI_EXEC_SAVE_SGPR in "
        "0..104");
    return;
  }
  std::vector<const ConSanMoiCandidate *> candidates =
      find_first_light_access_record_candidates(result, bytes);
  if (candidates.empty()) {
    result.warnings.emplace_back("ConSan MOI first-light probe found no native LDS or likely group "
                                 "flat load/store candidate");
    return;
  }

  struct PlannedAccessRecordPatch {
    const ConSanMoiCandidate *candidate = nullptr;
    uint64_t patch_bytes = 0;
    bool use_appended_cave = false;
    uint32_t record_index = 0;
    uint32_t record_count = 0;
  };
  std::vector<PlannedAccessRecordPatch> planned_patches;
  std::vector<PlannedAccessRecordPatch> appended_cave_candidates;
  std::vector<std::pair<uint64_t, uint64_t>> patched_ranges;
  uint32_t planned_record_count = 0;
  planned_patches.reserve(candidates.size());
  AmdGpuCodeObject code_object(bytes.data(), bytes.size());
  uint64_t appended_cave_text_offset = 0;
  if (code_object.text_sections().size() == 1)
    appended_cave_text_offset = code_object.text_sections().front()->size();
  for (const ConSanMoiCandidate *candidate_ptr : candidates) {
    const ConSanMoiCandidate &candidate = *candidate_ptr;
    const std::optional<std::vector<ConSanMoiAccessRange>> access_ranges =
        candidate_access_ranges(bytes, candidate);
    if (!access_ranges || access_ranges->empty())
      continue;
    const uint32_t candidate_record_count = static_cast<uint32_t>(access_ranges->size());
    if (!options.moi_dynamic_access_records &&
        candidate_record_count > layout.access_record_capacity - planned_record_count) {
      result.warnings.emplace_back(
          "ConSan MOI first-light probe skipped a supported load/store site without enough "
          "report-buffer access-record slots");
      continue;
    }
    std::vector<std::string> candidate_errors;
    auto words = build_first_light_access_record_words(
        bytes, candidate, options, arch, 0, 0, layout.access_record_capacity, candidate_errors);
    if (!words) {
      result.warnings.insert(result.warnings.end(), candidate_errors.begin(),
                             candidate_errors.end());
      continue;
    }

    const uint64_t patch_bytes = static_cast<uint64_t>(words->size() * sizeof(uint32_t));
    const uint32_t available_padding =
        count_nop_padding(bytes, candidate.file_offset + candidate.size, arch);
    const uint64_t available_bytes =
        candidate.size + static_cast<uint64_t>(available_padding) * sizeof(uint32_t);
    if (patch_bytes > available_bytes) {
      if (appended_cave_text_offset != 0 && candidate.size >= sizeof(uint32_t)) {
        const uint64_t return_branch_pc = appended_cave_text_offset + patch_bytes;
        if (compute_sopp_branch_simm16(candidate.text_offset, appended_cave_text_offset) &&
            compute_sopp_branch_simm16(return_branch_pc, candidate.text_offset + candidate.size)) {
          appended_cave_candidates.push_back(
              {candidate_ptr, patch_bytes, true, 0, candidate_record_count});
          continue;
        }
      }
      result.warnings.emplace_back("ConSan MOI first-light probe skipped a supported load/store "
                                   "site without enough inline padding or an appended cave");
      continue;
    }
    if (candidate.file_offset > bytes.size() ||
        patch_bytes > bytes.size() - candidate.file_offset) {
      result.errors.emplace_back("ConSan MOI first-light probe exceeds ELF bytes");
      return;
    }
    const uint64_t patch_begin = candidate.file_offset;
    const uint64_t patch_end = patch_begin + patch_bytes;
    const auto overlaps_existing =
        std::any_of(patched_ranges.begin(), patched_ranges.end(),
                    [&](const std::pair<uint64_t, uint64_t> range) {
                      return patch_begin < range.second && range.first < patch_end;
                    });
    if (overlaps_existing) {
      result.warnings.emplace_back(
          "ConSan MOI first-light probe skipped an overlapping patch site");
      continue;
    }

    patched_ranges.emplace_back(patch_begin, patch_end);
    planned_patches.push_back(
        {candidate_ptr, patch_bytes, false, planned_record_count, candidate_record_count});
    planned_record_count += candidate_record_count;
    if (planned_patches.size() == options.max_patches)
      break;
  }

  for (PlannedAccessRecordPatch candidate : appended_cave_candidates) {
    if (planned_patches.size() == options.max_patches)
      break;
    if (candidate.candidate == nullptr)
      continue;
    if (!options.moi_dynamic_access_records &&
        candidate.record_count > layout.access_record_capacity - planned_record_count) {
      result.warnings.emplace_back(
          "ConSan MOI first-light probe skipped an appended-cave site without enough "
          "report-buffer access-record slots");
      continue;
    }
    const uint64_t patch_begin = candidate.candidate->file_offset;
    const uint64_t patch_end = patch_begin + candidate.candidate->size;
    const auto overlaps_existing =
        std::any_of(patched_ranges.begin(), patched_ranges.end(),
                    [&](const std::pair<uint64_t, uint64_t> range) {
                      return patch_begin < range.second && range.first < patch_end;
                    });
    if (overlaps_existing) {
      result.warnings.emplace_back(
          "ConSan MOI first-light probe skipped an overlapping appended-cave patch site");
      continue;
    }
    patched_ranges.emplace_back(patch_begin, patch_end);
    candidate.record_index = planned_record_count;
    planned_patches.push_back(candidate);
    planned_record_count += candidate.record_count;
  }

  if (planned_patches.empty())
    return;

  const bool uses_appended_cave =
      std::ranges::any_of(planned_patches, [](const PlannedAccessRecordPatch &patch) {
        return patch.use_appended_cave;
      });
  if (uses_appended_cave) {
    CodeObjectPatcher patcher(code_object);
    if (!grow_moi_kernel_descriptor_vgprs(patcher, bytes, result,
                                          first_light_required_vgpr_count(options))) {
      return;
    }
    const std::span<const uint8_t> old_text = patcher.text_bytes();
    if (old_text.empty()) {
      result.errors.emplace_back("ConSan MOI first-light probe found no .text section");
      return;
    }
    std::vector<uint8_t> new_text(old_text.begin(), old_text.end());
    std::vector<ConSanPatchInfo> patches;
    patches.reserve(planned_patches.size());
    for (const PlannedAccessRecordPatch &planned_patch : planned_patches) {
      const ConSanMoiCandidate &candidate = *planned_patch.candidate;
      auto words = build_first_light_access_record_words(
          bytes, candidate, options, arch, planned_patch.record_index, planned_record_count,
          layout.access_record_capacity, result.errors);
      if (!words)
        return;

      ConSanPatchInfo info;
      info.kind = planned_patch.use_appended_cave ? ConSanPatchKind::TrampolineMoiAccessRecordStore
                                                  : ConSanPatchKind::InlineMoiAccessRecordStore;
      info.anchor_offset = candidate.text_offset;
      info.scratch_vgpr = options.scratch_vgpr;

      if (planned_patch.use_appended_cave) {
        const uint64_t cave_text_offset = static_cast<uint64_t>(new_text.size());
        const auto fwd = compute_sopp_branch_simm16(candidate.text_offset, cave_text_offset);
        const uint64_t return_branch_pc =
            cave_text_offset + static_cast<uint64_t>(words->size()) * sizeof(uint32_t);
        const auto ret =
            compute_sopp_branch_simm16(return_branch_pc, candidate.text_offset + candidate.size);
        if (!fwd || !ret) {
          result.errors.emplace_back(
              "ConSan MOI first-light probe appended cave branch is out of range");
          return;
        }

        std::vector<uint32_t> anchor_words(candidate.size / sizeof(uint32_t), build_s_nop(0, arch));
        if (anchor_words.empty()) {
          result.errors.emplace_back("ConSan MOI first-light probe empty anchor");
          return;
        }
        anchor_words.front() = build_s_branch(*fwd, arch);
        if (candidate.text_offset > new_text.size() ||
            anchor_words.size() * sizeof(uint32_t) > new_text.size() - candidate.text_offset) {
          result.errors.emplace_back("ConSan MOI first-light probe anchor exceeds .text");
          return;
        }
        std::memcpy(new_text.data() + candidate.text_offset, anchor_words.data(),
                    anchor_words.size() * sizeof(uint32_t));

        std::vector<uint32_t> cave_words = *words;
        cave_words.push_back(build_s_branch(*ret, arch));
        append_words_bytes(new_text, cave_words);
        info.trampoline_offset = cave_text_offset;
        info.original_size = candidate.size;
        info.trampoline_size = static_cast<uint32_t>(cave_words.size() * sizeof(uint32_t));
      } else {
        const uint64_t patch_bytes = static_cast<uint64_t>(words->size() * sizeof(uint32_t));
        if (patch_bytes != planned_patch.patch_bytes) {
          result.errors.emplace_back("ConSan MOI first-light probe final patch size changed");
          return;
        }
        if (candidate.text_offset > new_text.size() ||
            patch_bytes > new_text.size() - candidate.text_offset) {
          result.errors.emplace_back("ConSan MOI first-light probe exceeds .text");
          return;
        }
        std::memcpy(new_text.data() + candidate.text_offset, words->data(),
                    static_cast<size_t>(patch_bytes));
        info.trampoline_offset = candidate.text_offset + candidate.size;
        info.original_size = static_cast<uint32_t>(patch_bytes);
        info.trampoline_size = 0;
      }
      patches.push_back(info);
    }

    if (!patcher.replace_text(new_text)) {
      result.errors.emplace_back("ConSan MOI first-light probe could not grow .text");
      return;
    }
    result.elf_bytes = patcher.emit();
    result.patches.insert(result.patches.end(), patches.begin(), patches.end());
    result.modified = true;
    return;
  }

  result.elf_bytes.assign(bytes.begin(), bytes.end());
  for (const PlannedAccessRecordPatch &planned_patch : planned_patches) {
    const ConSanMoiCandidate &candidate = *planned_patch.candidate;
    auto words = build_first_light_access_record_words(
        bytes, candidate, options, arch, planned_patch.record_index, planned_record_count,
        layout.access_record_capacity, result.errors);
    if (!words) {
      result.elf_bytes.clear();
      return;
    }
    const uint64_t patch_bytes = static_cast<uint64_t>(words->size() * sizeof(uint32_t));
    if (patch_bytes != planned_patch.patch_bytes) {
      result.errors.emplace_back("ConSan MOI first-light probe final patch size changed");
      result.elf_bytes.clear();
      return;
    }
    std::memcpy(result.elf_bytes.data() + candidate.file_offset, words->data(),
                static_cast<size_t>(patch_bytes));
  }

  if (!grow_moi_kernel_descriptor_vgprs(result.elf_bytes, result,
                                        first_light_required_vgpr_count(options), result.errors)) {
    result.elf_bytes.clear();
    return;
  }

  for (const PlannedAccessRecordPatch &planned_patch : planned_patches) {
    const ConSanMoiCandidate &candidate = *planned_patch.candidate;
    ConSanPatchInfo info;
    info.kind = ConSanPatchKind::InlineMoiAccessRecordStore;
    info.anchor_offset = candidate.text_offset;
    info.trampoline_offset = candidate.text_offset + candidate.size;
    info.original_size = static_cast<uint32_t>(planned_patch.patch_bytes);
    info.trampoline_size = 0;
    info.scratch_vgpr = options.scratch_vgpr;
    result.patches.push_back(info);
  }

  result.modified = true;
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_owner_epoch_prologue_words(
    uint64_t prologue_text_offset, uint64_t original_entry_text_offset, uint16_t owner_vgpr,
    uint16_t epoch_vgpr, uint16_t owner_shift_bits, ConSanMoiOwnerSource owner_source,
    std::optional<uint16_t> owner_sgpr, rj_code_arch_t arch, std::vector<std::string> &errors) {
  std::vector<uint32_t> words;
  words.reserve(4);
  switch (owner_source) {
  case ConSanMoiOwnerSource::WorkitemId: {
    auto owner_init = build_v_lshrrev_b32_e32(
        owner_vgpr, scalar_positive_inline_u32(owner_shift_bits), kRdna4WorkitemIdX, arch);
    if (!owner_init) {
      errors.emplace_back("ConSan MOI owner/epoch prologue could not encode owner VGPR init");
      return std::nullopt;
    }
    words.push_back(*owner_init);
    break;
  }
  case ConSanMoiOwnerSource::HwId: {
    if (!owner_sgpr) {
      errors.emplace_back("ConSan MOI hw_id owner source requires RJ_CONSAN_MOI_OWNER_SGPR");
      return std::nullopt;
    }
    const auto hwreg = build_hwreg_imm(kGfx12HwRegHwId1, /*offset=*/0, kGfx12HwIdOwnerBits);
    if (!hwreg) {
      errors.emplace_back("ConSan MOI owner/epoch prologue could not encode HW_ID1 hwreg");
      return std::nullopt;
    }
    const auto get_hw_id = build_s_getreg_b32(*owner_sgpr, *hwreg, arch);
    if (!get_hw_id) {
      errors.emplace_back("ConSan MOI owner/epoch prologue could not encode s_getreg_b32");
      return std::nullopt;
    }
    words.push_back(*get_hw_id);
    words.push_back(build_v_mov_b32_e32(owner_vgpr, *owner_sgpr, arch));
    break;
  }
  }
  words.push_back(build_v_mov_b32_e32(epoch_vgpr, scalar_positive_inline_u32(0), arch));

  const uint64_t branch_pc =
      prologue_text_offset + static_cast<uint64_t>(words.size()) * sizeof(uint32_t);
  const auto branch = compute_sopp_branch_simm16(branch_pc, original_entry_text_offset);
  if (!branch) {
    errors.emplace_back("ConSan MOI owner/epoch prologue branch target is out of range");
    return std::nullopt;
  }
  words.push_back(build_s_branch(*branch, arch));
  return words;
}

void try_apply_owner_epoch_prologue_patch(std::span<const uint8_t> bytes,
                                          const ConSanOptions &options, rj_code_arch_t arch,
                                          ConSanResult &result) {
  if (!options.moi_init_owner_epoch)
    return;
  if (arch != ROCJITSU_CODE_ARCH_RDNA4) {
    result.warnings.emplace_back("ConSan MOI owner/epoch prologue currently supports only RDNA4");
    return;
  }
  if (!options.moi_owner_vgpr || !options.moi_epoch_vgpr) {
    result.errors.emplace_back(
        "ConSan MOI owner/epoch prologue requires RJ_CONSAN_MOI_OWNER_VGPR and "
        "RJ_CONSAN_MOI_EPOCH_VGPR");
    return;
  }
  if (*options.moi_owner_vgpr == *options.moi_epoch_vgpr) {
    result.errors.emplace_back("ConSan MOI owner and epoch VGPRs must be distinct");
    return;
  }
  if (options.moi_owner_source == ConSanMoiOwnerSource::HwId && !options.moi_owner_sgpr) {
    result.errors.emplace_back("ConSan MOI hw_id owner source requires RJ_CONSAN_MOI_OWNER_SGPR");
    return;
  }
  if (result.kernels.empty()) {
    result.warnings.emplace_back("ConSan MOI owner/epoch prologue found no kernel descriptors");
    return;
  }

  std::span<const uint8_t> active_bytes = bytes;
  if (result.modified)
    active_bytes = std::span<const uint8_t>(result.elf_bytes.data(), result.elf_bytes.size());
  AmdGpuCodeObject code_object(active_bytes.data(), active_bytes.size());
  CodeObjectPatcher patcher(code_object);
  const std::span<const uint8_t> old_text = patcher.text_bytes();
  if (old_text.empty()) {
    result.errors.emplace_back("ConSan MOI owner/epoch prologue found no .text section");
    return;
  }
  const uint32_t required_vgpr_count =
      std::max<uint32_t>(*options.moi_owner_vgpr, *options.moi_epoch_vgpr) + 1u;
  const uint32_t required_sgpr_count = options.moi_owner_source == ConSanMoiOwnerSource::HwId
                                           ? static_cast<uint32_t>(*options.moi_owner_sgpr) + 1u
                                           : 0u;
  if (!grow_moi_kernel_descriptor_registers(patcher, active_bytes, result, required_vgpr_count,
                                            required_sgpr_count))
    return;

  std::vector<uint8_t> new_text(old_text.begin(), old_text.end());
  std::vector<ConSanPatchInfo> patches;
  for (const ConSanKernelInfo &kernel : result.kernels) {
    if (!kernel.has_text_range)
      continue;

    append_nop_padding_to_alignment(new_text, kAmdhsaKernelEntryAlignment, arch);
    const uint64_t prologue_text_offset = static_cast<uint64_t>(new_text.size());
    const std::optional<uint16_t> owner_shift =
        moi_kernel_owner_shift(active_bytes, kernel, result.errors);
    if (!owner_shift)
      return;
    auto words = build_owner_epoch_prologue_words(prologue_text_offset, kernel.entry_text_offset,
                                                  *options.moi_owner_vgpr, *options.moi_epoch_vgpr,
                                                  *owner_shift, options.moi_owner_source,
                                                  options.moi_owner_sgpr, arch, result.errors);
    if (!words)
      return;
    if (!patcher.redirect_kernel_entry(kernel.descriptor_file_offset, kernel.entry_text_offset,
                                       prologue_text_offset)) {
      result.errors.emplace_back("ConSan MOI owner/epoch prologue could not redirect kernel entry");
      return;
    }
    append_words_bytes(new_text, *words);

    ConSanPatchInfo info;
    info.kind = ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue;
    info.anchor_offset = kernel.entry_text_offset;
    info.trampoline_offset = prologue_text_offset;
    info.original_size = 0;
    info.trampoline_size = static_cast<uint32_t>(words->size() * sizeof(uint32_t));
    patches.push_back(info);
  }

  if (patches.empty()) {
    result.warnings.emplace_back("ConSan MOI owner/epoch prologue found no patchable kernels");
    return;
  }
  if (!patcher.replace_text(new_text)) {
    result.errors.emplace_back("ConSan MOI owner/epoch prologue could not grow .text");
    return;
  }

  result.elf_bytes = patcher.emit();
  result.patches.insert(result.patches.end(), patches.begin(), patches.end());
  result.modified = true;
}

struct BarrierRecordCandidate {
  std::string container_name;
  ConSanBarrierSite site;
  std::optional<uint64_t> kernel_descriptor_file_offset;
};

[[nodiscard]] bool is_rocclr_runtime_kernel_name(std::string_view name) {
  return name.starts_with("__amd_rocclr_");
}

void append_barrier_epoch_candidates(const ConSanKernelInfo &kernel,
                                     std::vector<BarrierRecordCandidate> &candidates) {
  if (is_rocclr_runtime_kernel_name(kernel.name))
    return;
  for (const ConSanBarrierSite &site : kernel.barrier_sites)
    candidates.push_back({"kernel:" + kernel.name, site, kernel.descriptor_file_offset});
}

void append_barrier_epoch_candidates(const ConSanFunctionInfo &function,
                                     std::vector<BarrierRecordCandidate> &candidates) {
  for (const ConSanBarrierSite &site : function.barrier_sites)
    candidates.push_back({"function:" + function.name, site, std::nullopt});
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_barrier_record_cave_words(
    std::span<const uint8_t> bytes, const BarrierRecordCandidate &candidate,
    const ConSanOptions &options, rj_code_arch_t arch, uint32_t barrier_record_capacity,
    size_t barrier_records_offset, uint32_t original_barrier_word, uint64_t cave_text_offset,
    uint64_t return_text_offset, std::vector<std::string> &errors) {
  if (!options.scratch_vgpr) {
    errors.emplace_back("ConSan MOI barrier record patch requires RJ_CONSAN_TMP_VGPR");
    return std::nullopt;
  }
  if (static_cast<uint32_t>(*options.scratch_vgpr) + 6u > kMaxVgprs) {
    errors.emplace_back("ConSan MOI barrier record patch needs six scratch VGPRs");
    return std::nullopt;
  }
  if (!options.moi_exec_save_sgpr) {
    errors.emplace_back("ConSan MOI barrier record patch requires RJ_CONSAN_MOI_EXEC_SAVE_SGPR");
    return std::nullopt;
  }
  if (*options.moi_exec_save_sgpr > 104u || *options.moi_exec_save_sgpr % 2u != 0u) {
    errors.emplace_back(
        "ConSan MOI barrier record patch requires an even RJ_CONSAN_MOI_EXEC_SAVE_SGPR in 0..104");
    return std::nullopt;
  }
  if (reject_optional_scratch_range_overlap(options.moi_owner_vgpr, *options.scratch_vgpr, 6,
                                            "MOI owner", errors))
    return std::nullopt;

  std::optional<uint16_t> derived_owner_vgpr;
  std::optional<uint32_t> derived_owner_word;
  if (!options.moi_owner_vgpr) {
    if (!candidate.kernel_descriptor_file_offset) {
      errors.emplace_back(
          "ConSan MOI barrier record patch requires RJ_CONSAN_MOI_OWNER_VGPR for function "
          "barriers");
      return std::nullopt;
    }
    const auto owner_shift =
        moi_descriptor_owner_shift(bytes, *candidate.kernel_descriptor_file_offset, errors);
    if (!owner_shift)
      return std::nullopt;
    const uint16_t value_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 5u);
    const auto owner_init = build_v_lshrrev_b32_e32(
        value_vgpr, scalar_positive_inline_u32(*owner_shift), kRdna4WorkitemIdX, arch);
    if (!owner_init) {
      errors.emplace_back("ConSan MOI barrier record patch could not encode owner derivation");
      return std::nullopt;
    }
    derived_owner_vgpr = value_vgpr;
    derived_owner_word = *owner_init;
  }
  ConSanMoiWorkgroupSources workgroup_sources;
  if (candidate.kernel_descriptor_file_offset) {
    const auto descriptor_workgroup_sources = moi_descriptor_workgroup_sources(
        bytes, *candidate.kernel_descriptor_file_offset, arch, errors);
    if (!descriptor_workgroup_sources)
      return std::nullopt;
    workgroup_sources = *descriptor_workgroup_sources;
  }

  std::vector<uint32_t> words;
  const uint64_t base = *options.moi_report_buffer_address;
  const uint64_t barrier_record_base = base + barrier_records_offset;
  const uint16_t slot_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 2u);
  const uint16_t lane_rank_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 5u);
  constexpr uint16_t kScalarInlineMinusOne = 0xC1;

  words.reserve(160);
  const auto mbcnt_lo = build_v_mbcnt_lo_u32_b32(lane_rank_vgpr, kScalarInlineMinusOne,
                                                 scalar_positive_inline_u32(0), arch);
  const auto mbcnt_hi = build_v_mbcnt_hi_u32_b32(lane_rank_vgpr, kScalarInlineMinusOne,
                                                 vector_source_vgpr(lane_rank_vgpr), arch);
  const auto first_active_lane =
      build_v_cmp_eq_u32_e32_vcc(scalar_positive_inline_u32(0), lane_rank_vgpr, arch);
  const auto save_exec = build_s_and_saveexec_b64(*options.moi_exec_save_sgpr, kRdna4VccLo, arch);
  if (!mbcnt_lo || !mbcnt_hi || !first_active_lane || !save_exec) {
    errors.emplace_back("ConSan MOI barrier record patch could not encode EXEC narrowing");
    return std::nullopt;
  }
  words.insert(words.end(), mbcnt_lo->begin(), mbcnt_lo->end());
  words.insert(words.end(), mbcnt_hi->begin(), mbcnt_hi->end());
  words.push_back(*first_active_lane);
  words.push_back(*save_exec);

  if (!append_atomic_fetch_add_one_u32(words,
                                       base + offsetof(ConSanMoiReportHeader, barrier_record_count),
                                       slot_vgpr, *options.scratch_vgpr, arch)) {
    errors.emplace_back("ConSan MOI barrier record patch could not encode dynamic slot reserve");
    return std::nullopt;
  }

  const uint16_t value_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 5u);
  const auto mov_capacity = build_v_mov_b32_e64_literal(value_vgpr, barrier_record_capacity, arch);
  const auto slot_in_capacity =
      build_v_cmp_gt_u32_e32_vcc(vector_source_vgpr(value_vgpr), slot_vgpr, arch);
  if (!mov_capacity || !slot_in_capacity) {
    errors.emplace_back("ConSan MOI barrier record patch could not encode capacity guard");
    return std::nullopt;
  }
  words.insert(words.end(), mov_capacity->begin(), mov_capacity->end());
  words.push_back(*slot_in_capacity);

  std::vector<uint32_t> record_words;
  record_words.reserve(128);
  if (derived_owner_word)
    record_words.push_back(*derived_owner_word);
  if ((derived_owner_vgpr &&
       !append_dynamic_barrier_store_u32_vgpr(
           record_words, barrier_record_base + offsetof(ConSanMoiBarrierRecord, wave_id),
           *derived_owner_vgpr, slot_vgpr, *options.scratch_vgpr, arch)) ||
      !append_dynamic_barrier_event_index_store(
          record_words, base + offsetof(ConSanMoiReportHeader, event_counter),
          barrier_record_base + offsetof(ConSanMoiBarrierRecord, event_index), slot_vgpr,
          *options.scratch_vgpr, arch) ||
      !append_dynamic_barrier_store_workgroup_source(
          record_words, barrier_record_base + offsetof(ConSanMoiBarrierRecord, workgroup_x),
          workgroup_sources.x, slot_vgpr, *options.scratch_vgpr, arch) ||
      !append_dynamic_barrier_store_workgroup_source(
          record_words, barrier_record_base + offsetof(ConSanMoiBarrierRecord, workgroup_y),
          workgroup_sources.y, slot_vgpr, *options.scratch_vgpr, arch) ||
      !append_dynamic_barrier_store_workgroup_source(
          record_words, barrier_record_base + offsetof(ConSanMoiBarrierRecord, workgroup_z),
          workgroup_sources.z, slot_vgpr, *options.scratch_vgpr, arch) ||
      !append_dynamic_barrier_store_u32_scalar_src(
          record_words, barrier_record_base + offsetof(ConSanMoiBarrierRecord, lane_mask),
          *options.moi_exec_save_sgpr, slot_vgpr, *options.scratch_vgpr, arch) ||
      !append_dynamic_barrier_store_u32_scalar_src(
          record_words,
          barrier_record_base + offsetof(ConSanMoiBarrierRecord, lane_mask) + sizeof(uint32_t),
          static_cast<uint16_t>(*options.moi_exec_save_sgpr + 1u), slot_vgpr, *options.scratch_vgpr,
          arch) ||
      (options.moi_owner_vgpr &&
       !append_dynamic_barrier_store_u32_vgpr(
           record_words, barrier_record_base + offsetof(ConSanMoiBarrierRecord, wave_id),
           *options.moi_owner_vgpr, slot_vgpr, *options.scratch_vgpr, arch)) ||
      !append_dynamic_barrier_store_u32_literal(
          record_words, barrier_record_base + offsetof(ConSanMoiBarrierRecord, instruction_offset),
          static_cast<uint32_t>(candidate.site.text_offset), slot_vgpr, *options.scratch_vgpr,
          arch)) {
    errors.emplace_back("ConSan MOI barrier record patch could not encode record stores");
    return std::nullopt;
  }

  if (record_words.size() > static_cast<size_t>(std::numeric_limits<int16_t>::max())) {
    errors.emplace_back("ConSan MOI barrier record overflow branch is out of range");
    return std::nullopt;
  }
  const auto skip_record = build_s_cbranch_vccz(static_cast<int16_t>(record_words.size()), arch);
  const auto restore_exec = build_s_mov_b64(kRdna4ExecLo, *options.moi_exec_save_sgpr, arch);
  if (!skip_record || !restore_exec) {
    errors.emplace_back("ConSan MOI barrier record patch could not encode EXEC restore");
    return std::nullopt;
  }
  words.push_back(*skip_record);
  words.insert(words.end(), record_words.begin(), record_words.end());
  words.push_back(*restore_exec);

  words.push_back(original_barrier_word);

  const uint64_t branch_pc =
      cave_text_offset + static_cast<uint64_t>(words.size()) * sizeof(uint32_t);
  const auto ret = compute_sopp_branch_simm16(branch_pc, return_text_offset);
  if (!ret) {
    errors.emplace_back("ConSan MOI barrier record return branch is out of range");
    return std::nullopt;
  }
  words.push_back(build_s_branch(*ret, arch));
  return words;
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_inline_shadow_barrier_epoch_cave_words(
    const ConSanBarrierSite &site, const ConSanOptions &options, rj_code_arch_t arch,
    uint32_t original_barrier_word, uint64_t cave_text_offset, uint64_t return_text_offset,
    std::vector<std::string> &errors) {
  if (!options.moi_epoch_vgpr) {
    errors.emplace_back(
        "ConSan MOI inline-shadow barrier epoch patch requires RJ_CONSAN_MOI_EPOCH_VGPR");
    return std::nullopt;
  }

  const auto increment_epoch = build_v_add_nc_u32_e32(
      *options.moi_epoch_vgpr, scalar_positive_inline_u32(1), *options.moi_epoch_vgpr, arch);
  if (!increment_epoch) {
    errors.emplace_back("ConSan MOI inline-shadow barrier epoch patch could not encode epoch add");
    return std::nullopt;
  }

  std::vector<uint32_t> words;
  words.reserve(3);
  words.push_back(original_barrier_word);
  words.push_back(*increment_epoch);

  const uint64_t branch_pc =
      cave_text_offset + static_cast<uint64_t>(words.size()) * sizeof(uint32_t);
  const auto ret = compute_sopp_branch_simm16(branch_pc, return_text_offset);
  if (!ret) {
    errors.emplace_back("ConSan MOI inline-shadow barrier epoch return branch is out of range at " +
                        std::to_string(site.text_offset));
    return std::nullopt;
  }
  words.push_back(build_s_branch(*ret, arch));
  return words;
}

void try_apply_inline_shadow_barrier_epoch_patch(const ConSanOptions &options, rj_code_arch_t arch,
                                                 ConSanResult &result) {
  if (!options.moi_track_barriers)
    return;
  if (arch != ROCJITSU_CODE_ARCH_RDNA4) {
    result.warnings.emplace_back(
        "ConSan MOI inline-shadow barrier epoch patch currently supports only RDNA4");
    return;
  }
  if (!result.modified) {
    result.warnings.emplace_back(
        "ConSan MOI inline-shadow barrier epoch patch skipped because no exact-shadow access "
        "probe was emitted");
    return;
  }
  if (!options.moi_epoch_vgpr) {
    result.warnings.emplace_back(
        "ConSan MOI inline-shadow barrier epoch patch requires RJ_CONSAN_MOI_EPOCH_VGPR");
    return;
  }
  if (static_cast<uint32_t>(*options.moi_epoch_vgpr) + 1u > kMaxVgprs) {
    result.warnings.emplace_back(
        "ConSan MOI inline-shadow barrier epoch patch needs one epoch VGPR");
    return;
  }

  std::vector<BarrierRecordCandidate> candidates;
  for (const ConSanKernelInfo &kernel : result.kernels)
    append_barrier_epoch_candidates(kernel, candidates);
  for (const ConSanFunctionInfo &function : result.functions)
    append_barrier_epoch_candidates(function, candidates);
  std::sort(candidates.begin(), candidates.end(), [](const auto &lhs, const auto &rhs) {
    return lhs.site.text_offset < rhs.site.text_offset;
  });
  if (candidates.empty()) {
    result.warnings.emplace_back("ConSan MOI inline-shadow barrier epoch patch found no barriers");
    return;
  }

  if (result.elf_bytes.empty()) {
    result.errors.emplace_back("ConSan MOI inline-shadow barrier epoch patch missing patched ELF");
    return;
  }
  std::span<const uint8_t> active_bytes(result.elf_bytes.data(), result.elf_bytes.size());
  AmdGpuCodeObject code_object(active_bytes.data(), active_bytes.size());
  CodeObjectPatcher patcher(code_object);
  const std::span<const uint8_t> old_text = patcher.text_bytes();
  if (old_text.empty()) {
    result.errors.emplace_back(
        "ConSan MOI inline-shadow barrier epoch patch found no .text section");
    return;
  }
  if (!grow_moi_kernel_descriptor_vgprs(patcher, active_bytes, result,
                                        static_cast<uint32_t>(*options.moi_epoch_vgpr) + 1u)) {
    return;
  }

  std::vector<BarrierRecordCandidate> selected_candidates;
  selected_candidates.reserve(options.max_patches);
  for (const BarrierRecordCandidate &candidate : candidates) {
    if (selected_candidates.size() == options.max_patches)
      break;
    const ConSanBarrierSite &site = candidate.site;
    if (site.size != sizeof(uint32_t)) {
      result.warnings.emplace_back(
          "ConSan MOI inline-shadow barrier epoch patch skipped non-32-bit barrier " +
          site.mnemonic + " in " + candidate.container_name);
      continue;
    }
    if (site.text_offset > old_text.size() || site.size > old_text.size() - site.text_offset) {
      result.errors.emplace_back("ConSan MOI inline-shadow barrier epoch site is outside .text");
      return;
    }
    selected_candidates.push_back(candidate);
  }

  if (selected_candidates.empty()) {
    result.warnings.emplace_back(
        "ConSan MOI inline-shadow barrier epoch patch found no patchable barriers");
    return;
  }

  std::vector<uint8_t> new_text(old_text.begin(), old_text.end());
  std::vector<ConSanPatchInfo> patches;
  patches.reserve(selected_candidates.size());
  for (const BarrierRecordCandidate &candidate : selected_candidates) {
    const ConSanBarrierSite &site = candidate.site;
    uint32_t original_barrier_word = 0;
    std::memcpy(&original_barrier_word, new_text.data() + site.text_offset,
                sizeof(original_barrier_word));

    const uint64_t cave_text_offset = static_cast<uint64_t>(new_text.size());
    const uint64_t return_text_offset = site.text_offset + site.size;
    auto cave_words = build_inline_shadow_barrier_epoch_cave_words(
        site, options, arch, original_barrier_word, cave_text_offset, return_text_offset,
        result.errors);
    if (!cave_words)
      return;

    const auto fwd = compute_sopp_branch_simm16(site.text_offset, cave_text_offset);
    if (!fwd) {
      result.errors.emplace_back(
          "ConSan MOI inline-shadow barrier epoch forward branch is out of range");
      return;
    }
    if (!write_word_bytes(new_text, site.text_offset, build_s_branch(*fwd, arch))) {
      result.errors.emplace_back(
          "ConSan MOI inline-shadow barrier epoch patch could not rewrite barrier slot");
      return;
    }
    append_words_bytes(new_text, *cave_words);

    ConSanPatchInfo info;
    info.kind = ConSanPatchKind::TrampolineMoiInlineEpochBarrier;
    info.anchor_offset = site.text_offset;
    info.trampoline_offset = cave_text_offset;
    info.original_size = site.size;
    info.trampoline_size = static_cast<uint32_t>(cave_words->size() * sizeof(uint32_t));
    patches.push_back(info);
  }

  if (!patcher.replace_text(new_text)) {
    result.errors.emplace_back("ConSan MOI inline-shadow barrier epoch patch could not grow .text");
    return;
  }

  result.elf_bytes = patcher.emit();
  result.patches.insert(result.patches.end(), patches.begin(), patches.end());
  result.modified = true;
}

void try_apply_barrier_epoch_patch(std::span<const uint8_t> bytes, const ConSanOptions &options,
                                   rj_code_arch_t arch, ConSanResult &result) {
  if (!options.moi_track_barriers)
    return;
  if (options.moi_engine == ConSanMoiEngine::InlineShadow) {
    try_apply_inline_shadow_barrier_epoch_patch(options, arch, result);
    return;
  }
  if (arch != ROCJITSU_CODE_ARCH_RDNA4) {
    result.warnings.emplace_back("ConSan MOI barrier record patch currently supports only RDNA4");
    return;
  }
  if (!options.moi_report_buffer_address) {
    result.warnings.emplace_back("ConSan MOI barrier record patch requires a MOI report buffer");
    return;
  }
  if (options.moi_report_buffer_size < consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 1)) {
    result.warnings.emplace_back(
        "ConSan MOI barrier record patch requires room for access and barrier records");
    return;
  }
  const ConSanMoiReportBufferLayout layout = consan_moi_report_buffer_layout_for_bytes(
      options.moi_report_buffer_size,
      /*include_barriers=*/true, options.moi_track_atomics);
  const uint32_t max_candidates = std::min(options.max_patches, layout.barrier_record_capacity);
  if (max_candidates == 0) {
    result.warnings.emplace_back("ConSan MOI barrier record patch has no barrier record capacity");
    return;
  }
  if (!options.scratch_vgpr) {
    result.warnings.emplace_back("ConSan MOI barrier record patch requires RJ_CONSAN_TMP_VGPR");
    return;
  }
  if (static_cast<uint32_t>(*options.scratch_vgpr) + 6u > kMaxVgprs) {
    result.warnings.emplace_back("ConSan MOI barrier record patch needs six scratch VGPRs");
    return;
  }
  if (!options.moi_exec_save_sgpr) {
    result.warnings.emplace_back(
        "ConSan MOI barrier record patch requires RJ_CONSAN_MOI_EXEC_SAVE_SGPR");
    return;
  }
  if (*options.moi_exec_save_sgpr > 104u || *options.moi_exec_save_sgpr % 2u != 0u) {
    result.warnings.emplace_back(
        "ConSan MOI barrier record patch requires an even RJ_CONSAN_MOI_EXEC_SAVE_SGPR in 0..104");
    return;
  }

  std::vector<BarrierRecordCandidate> candidates;
  for (const ConSanKernelInfo &kernel : result.kernels)
    append_barrier_epoch_candidates(kernel, candidates);
  for (const ConSanFunctionInfo &function : result.functions)
    append_barrier_epoch_candidates(function, candidates);
  std::sort(candidates.begin(), candidates.end(), [](const auto &lhs, const auto &rhs) {
    return lhs.site.text_offset < rhs.site.text_offset;
  });
  if (candidates.empty()) {
    result.warnings.emplace_back("ConSan MOI barrier record patch found no barriers");
    return;
  }

  std::span<const uint8_t> active_bytes = bytes;
  if (result.modified)
    active_bytes = std::span<const uint8_t>(result.elf_bytes.data(), result.elf_bytes.size());
  AmdGpuCodeObject code_object(active_bytes.data(), active_bytes.size());
  CodeObjectPatcher patcher(code_object);
  const std::span<const uint8_t> old_text = patcher.text_bytes();
  if (old_text.empty()) {
    result.errors.emplace_back("ConSan MOI barrier record patch found no .text section");
    return;
  }
  uint32_t required_vgpr_count = static_cast<uint32_t>(*options.scratch_vgpr) + 6u;
  if (options.moi_owner_vgpr)
    required_vgpr_count = std::max<uint32_t>(required_vgpr_count,
                                             static_cast<uint32_t>(*options.moi_owner_vgpr) + 1u);
  if (!grow_moi_kernel_descriptor_vgprs(patcher, active_bytes, result, required_vgpr_count))
    return;

  std::vector<BarrierRecordCandidate> selected_candidates;
  selected_candidates.reserve(max_candidates);
  for (const BarrierRecordCandidate &candidate : candidates) {
    if (selected_candidates.size() == max_candidates)
      break;
    const ConSanBarrierSite &site = candidate.site;
    if (site.size != sizeof(uint32_t)) {
      result.warnings.emplace_back("ConSan MOI barrier record patch skipped non-32-bit barrier " +
                                   site.mnemonic + " in " + candidate.container_name);
      continue;
    }
    if (site.text_offset > old_text.size() || site.size > old_text.size() - site.text_offset) {
      result.errors.emplace_back("ConSan MOI barrier record patch site is outside .text");
      return;
    }
    if (!options.moi_owner_vgpr && !candidate.kernel_descriptor_file_offset) {
      result.warnings.emplace_back("ConSan MOI barrier record patch skipped function barrier in " +
                                   candidate.container_name +
                                   " because no owner VGPR was configured");
      continue;
    }
    selected_candidates.push_back(candidate);
  }

  if (selected_candidates.empty()) {
    result.warnings.emplace_back("ConSan MOI barrier record patch found no patchable barriers");
    return;
  }

  std::vector<uint8_t> new_text(old_text.begin(), old_text.end());
  std::vector<ConSanPatchInfo> patches;
  for (const BarrierRecordCandidate &candidate : selected_candidates) {
    const ConSanBarrierSite &site = candidate.site;
    uint32_t original_barrier_word = 0;
    std::memcpy(&original_barrier_word, new_text.data() + site.text_offset,
                sizeof(original_barrier_word));

    const uint64_t cave_text_offset = static_cast<uint64_t>(new_text.size());
    const uint64_t return_text_offset = site.text_offset + site.size;
    auto cave_words = build_barrier_record_cave_words(
        active_bytes, candidate, options, arch, layout.barrier_record_capacity,
        layout.barrier_records_offset, original_barrier_word, cave_text_offset, return_text_offset,
        result.errors);
    if (!cave_words)
      return;

    const auto fwd = compute_sopp_branch_simm16(site.text_offset, cave_text_offset);
    if (!fwd) {
      result.errors.emplace_back("ConSan MOI barrier record forward branch is out of range");
      return;
    }
    if (!write_word_bytes(new_text, site.text_offset, build_s_branch(*fwd, arch))) {
      result.errors.emplace_back("ConSan MOI barrier record patch could not rewrite barrier slot");
      return;
    }
    append_words_bytes(new_text, *cave_words);

    ConSanPatchInfo info;
    info.kind = ConSanPatchKind::TrampolineMoiBarrierRecord;
    info.anchor_offset = site.text_offset;
    info.trampoline_offset = cave_text_offset;
    info.original_size = site.size;
    info.trampoline_size = static_cast<uint32_t>(cave_words->size() * sizeof(uint32_t));
    info.scratch_vgpr = options.scratch_vgpr;
    patches.push_back(info);
  }

  if (patches.empty()) {
    result.warnings.emplace_back("ConSan MOI barrier record patch found no patchable barriers");
    return;
  }
  if (!patcher.replace_text(new_text)) {
    result.errors.emplace_back("ConSan MOI barrier record patch could not grow .text");
    return;
  }

  result.elf_bytes = patcher.emit();
  result.patches.insert(result.patches.end(), patches.begin(), patches.end());
  result.modified = true;
}

struct AtomicRecordCandidate {
  std::string container_name;
  ConSanAtomicSite site;
  std::optional<uint64_t> kernel_descriptor_file_offset;
};

[[nodiscard]] bool is_first_light_atomic_record_candidate(const ConSanAtomicSite &site) {
  if (!site.mnemonic.starts_with("flat_atomic"))
    return false;
  if (site.size != 3u * sizeof(uint32_t))
    return false;
  if (!site.raw_saddr || *site.raw_saddr != kRdna4FlatNoSaddr)
    return false;
  if (!site.raw_ioffset || *site.raw_ioffset != 0)
    return false;
  if (!site.addr_vgpr || *site.addr_vgpr >= 255)
    return false;
  if (!site.raw_scope || !site.raw_th || !site.returns_old_value)
    return false;
  return true;
}

[[nodiscard]] ConSanMoiAtomicEventKind atomic_event_kind_for_site(const ConSanAtomicSite &site) {
  return site.returns_old_value.value_or(false) ? ConSanMoiAtomicEventKind::Acquire
                                                : ConSanMoiAtomicEventKind::Release;
}

void append_atomic_record_candidates(const ConSanKernelInfo &kernel,
                                     std::vector<AtomicRecordCandidate> &candidates) {
  if (is_rocclr_runtime_kernel_name(kernel.name))
    return;
  for (const ConSanAtomicSite &site : kernel.atomic_sites) {
    if (is_first_light_atomic_record_candidate(site))
      candidates.push_back({"kernel:" + kernel.name, site, kernel.descriptor_file_offset});
  }
}

void append_atomic_record_candidates(const ConSanFunctionInfo &function,
                                     std::vector<AtomicRecordCandidate> &candidates) {
  for (const ConSanAtomicSite &site : function.atomic_sites) {
    if (is_first_light_atomic_record_candidate(site))
      candidates.push_back({"function:" + function.name, site, std::nullopt});
  }
}

[[nodiscard]] bool reject_atomic_candidate_scratch_overlap(const ConSanAtomicSite &site,
                                                           uint16_t scratch_vgpr,
                                                           std::vector<std::string> &errors) {
  if (!site.addr_vgpr) {
    errors.emplace_back("ConSan MOI atomic record patch requires a flat address VGPR pair");
    return true;
  }
  if (range_overlaps_scratch_triplet(*site.addr_vgpr, 2, scratch_vgpr)) {
    errors.emplace_back(
        "ConSan MOI atomic record patch scratch VGPRs overlap the atomic address VGPRs");
    return true;
  }
  if (site.data_vgpr && range_overlaps_scratch_triplet(*site.data_vgpr, 1, scratch_vgpr)) {
    errors.emplace_back(
        "ConSan MOI atomic record patch scratch VGPRs overlap the atomic data VGPR");
    return true;
  }
  if (site.dst_vgpr && range_overlaps_scratch_triplet(*site.dst_vgpr, 1, scratch_vgpr)) {
    errors.emplace_back(
        "ConSan MOI atomic record patch scratch VGPRs overlap the atomic destination VGPR");
    return true;
  }
  if (site.returns_old_value.value_or(false) && site.dst_vgpr &&
      *site.dst_vgpr >= *site.addr_vgpr &&
      *site.dst_vgpr <= static_cast<uint16_t>(*site.addr_vgpr + 1u)) {
    errors.emplace_back(
        "ConSan MOI atomic record patch cannot record an address clobbered by the atomic result");
    return true;
  }
  return false;
}

[[nodiscard]] bool
reject_inline_atomic_candidate_scratch_overlap(const ConSanAtomicSite &site, uint16_t scratch_vgpr,
                                               uint16_t scratch_count,
                                               std::vector<std::string> &errors) {
  if (!site.addr_vgpr) {
    errors.emplace_back("ConSan MOI inline atomic patch requires a flat address VGPR pair");
    return true;
  }
  if (range_overlaps(*site.addr_vgpr, 2, scratch_vgpr, scratch_count)) {
    errors.emplace_back(
        "ConSan MOI inline atomic patch scratch VGPRs overlap the atomic address VGPRs");
    return true;
  }
  if (site.data_vgpr && overlaps_scratch_range(*site.data_vgpr, scratch_vgpr, scratch_count)) {
    errors.emplace_back(
        "ConSan MOI inline atomic patch scratch VGPRs overlap the atomic data VGPR");
    return true;
  }
  if (site.dst_vgpr && overlaps_scratch_range(*site.dst_vgpr, scratch_vgpr, scratch_count)) {
    errors.emplace_back(
        "ConSan MOI inline atomic patch scratch VGPRs overlap the atomic destination VGPR");
    return true;
  }
  if (site.returns_old_value.value_or(false) && site.dst_vgpr &&
      *site.dst_vgpr >= *site.addr_vgpr &&
      *site.dst_vgpr <= static_cast<uint16_t>(*site.addr_vgpr + 1u)) {
    errors.emplace_back(
        "ConSan MOI inline atomic patch cannot use an address clobbered by the atomic result");
    return true;
  }
  return false;
}

[[nodiscard]] bool validate_inline_atomic_exec_save_sgpr(const ConSanOptions &options,
                                                         std::vector<std::string> &errors) {
  if (!options.moi_exec_save_sgpr) {
    errors.emplace_back(
        "ConSan MOI inline atomic acquire patch requires RJ_CONSAN_MOI_EXEC_SAVE_SGPR");
    return false;
  }
  if (*options.moi_exec_save_sgpr > 100u || *options.moi_exec_save_sgpr % 2u != 0u) {
    errors.emplace_back("ConSan MOI inline atomic acquire patch requires an even "
                        "RJ_CONSAN_MOI_EXEC_SAVE_SGPR in 0..100");
    return false;
  }
  return true;
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_inline_atomic_ordering_cave_words(
    std::span<const uint8_t> bytes, const AtomicRecordCandidate &candidate,
    const ConSanOptions &options, rj_code_arch_t arch, size_t inline_atomic_release_slots_offset,
    uint64_t cave_text_offset, uint64_t return_text_offset, std::vector<std::string> &errors) {
  constexpr uint16_t kInlineAtomicScratchCount = 5;
  if (!options.scratch_vgpr) {
    errors.emplace_back("ConSan MOI inline atomic patch requires RJ_CONSAN_TMP_VGPR");
    return std::nullopt;
  }
  if (static_cast<uint32_t>(*options.scratch_vgpr) + kInlineAtomicScratchCount > kMaxVgprs) {
    errors.emplace_back("ConSan MOI inline atomic patch needs five scratch VGPRs");
    return std::nullopt;
  }
  if (!options.moi_owner_vgpr || !options.moi_epoch_vgpr) {
    errors.emplace_back("ConSan MOI inline atomic patch requires RJ_CONSAN_MOI_OWNER_VGPR and "
                        "RJ_CONSAN_MOI_EPOCH_VGPR");
    return std::nullopt;
  }
  if (!validate_inline_atomic_exec_save_sgpr(options, errors))
    return std::nullopt;
  if (reject_inline_atomic_candidate_scratch_overlap(candidate.site, *options.scratch_vgpr,
                                                     kInlineAtomicScratchCount, errors) ||
      reject_optional_scratch_range_overlap(options.moi_owner_vgpr, *options.scratch_vgpr,
                                            kInlineAtomicScratchCount, "MOI owner", errors) ||
      reject_optional_scratch_range_overlap(options.moi_epoch_vgpr, *options.scratch_vgpr,
                                            kInlineAtomicScratchCount, "MOI epoch", errors))
    return std::nullopt;
  if (candidate.site.file_offset > bytes.size() ||
      candidate.site.size > bytes.size() - candidate.site.file_offset) {
    errors.emplace_back("ConSan MOI inline atomic patch site exceeds ELF bytes");
    return std::nullopt;
  }

  const uint64_t slot_base =
      *options.moi_report_buffer_address + inline_atomic_release_slots_offset;
  const uint16_t value_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 2u);
  const uint16_t vcc_lo_save_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 3u);
  const uint16_t vcc_hi_save_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 4u);
  const uint16_t address_vgpr = *candidate.site.addr_vgpr;

  std::vector<uint32_t> words;
  words.reserve(candidate.site.size / sizeof(uint32_t) + 96u);
  for (uint64_t offset = 0; offset < candidate.site.size; offset += sizeof(uint32_t)) {
    uint32_t word = 0;
    std::memcpy(&word, bytes.data() + candidate.site.file_offset + offset, sizeof(word));
    words.push_back(word);
  }
  words.push_back(kWaitLoadcnt0);

  if (atomic_event_kind_for_site(candidate.site) == ConSanMoiAtomicEventKind::Release) {
    if (!append_store_u32_vgpr(words,
                               slot_base + offsetof(ConSanMoiInlineAtomicReleaseSlot, owner_id),
                               *options.moi_owner_vgpr, *options.scratch_vgpr, arch) ||
        !append_store_u32_vgpr(words, slot_base + offsetof(ConSanMoiInlineAtomicReleaseSlot, epoch),
                               *options.moi_epoch_vgpr, *options.scratch_vgpr, arch) ||
        !append_store_u32_vgpr(
            words, slot_base + offsetof(ConSanMoiInlineAtomicReleaseSlot, atomic_address),
            address_vgpr, *options.scratch_vgpr, arch) ||
        !append_store_u32_vgpr(
            words,
            slot_base + offsetof(ConSanMoiInlineAtomicReleaseSlot, atomic_address) +
                sizeof(uint32_t),
            static_cast<uint16_t>(address_vgpr + 1u), *options.scratch_vgpr, arch) ||
        !append_store_u32_literal(words,
                                  slot_base + offsetof(ConSanMoiInlineAtomicReleaseSlot, valid), 1u,
                                  *options.scratch_vgpr, arch)) {
      errors.emplace_back(
          "ConSan MOI inline atomic release patch could not encode metadata stores");
      return std::nullopt;
    }
  } else {
    words.push_back(build_v_mov_b32_e32(vcc_lo_save_vgpr, kRdna4VccLo, arch));
    words.push_back(
        build_v_mov_b32_e32(vcc_hi_save_vgpr, static_cast<uint16_t>(kRdna4VccLo + 1u), arch));

    const auto narrow_if_valid =
        build_s_and_saveexec_b64(*options.moi_exec_save_sgpr, kRdna4VccLo, arch);
    const auto narrow_if_low_matches = build_s_and_saveexec_b64(
        static_cast<uint16_t>(*options.moi_exec_save_sgpr + 2u), kRdna4VccLo, arch);
    const auto narrow_if_high_matches = build_s_and_saveexec_b64(
        static_cast<uint16_t>(*options.moi_exec_save_sgpr + 4u), kRdna4VccLo, arch);
    const auto narrow_if_other_owner = build_s_and_saveexec_b64(
        static_cast<uint16_t>(*options.moi_exec_save_sgpr + 6u), kRdna4VccLo, arch);
    const auto restore_exec = build_s_mov_b64(kRdna4ExecLo, *options.moi_exec_save_sgpr, arch);
    const auto restore_vcc_lo = build_v_readfirstlane_b32(kRdna4VccLo, vcc_lo_save_vgpr, arch);
    const auto restore_vcc_hi =
        build_v_readfirstlane_b32(static_cast<uint16_t>(kRdna4VccLo + 1u), vcc_hi_save_vgpr, arch);
    if (!narrow_if_valid || !narrow_if_low_matches || !narrow_if_high_matches ||
        !narrow_if_other_owner || !restore_exec || !restore_vcc_lo || !restore_vcc_hi) {
      errors.emplace_back("ConSan MOI inline atomic acquire patch could not encode EXEC/VCC ops");
      return std::nullopt;
    }

    if (!append_load_u32_vgpr(words, slot_base + offsetof(ConSanMoiInlineAtomicReleaseSlot, valid),
                              value_vgpr, *options.scratch_vgpr, arch)) {
      errors.emplace_back("ConSan MOI inline atomic acquire patch could not load valid flag");
      return std::nullopt;
    }
    const auto valid_ne_zero =
        build_v_cmp_ne_u32_e32_vcc(scalar_positive_inline_u32(0), value_vgpr, arch);
    if (!valid_ne_zero) {
      errors.emplace_back("ConSan MOI inline atomic acquire patch could not compare valid flag");
      return std::nullopt;
    }
    words.push_back(*valid_ne_zero);
    words.push_back(*narrow_if_valid);

    if (!append_load_u32_vgpr(
            words, slot_base + offsetof(ConSanMoiInlineAtomicReleaseSlot, atomic_address),
            value_vgpr, *options.scratch_vgpr, arch)) {
      errors.emplace_back("ConSan MOI inline atomic acquire patch could not load address low");
      return std::nullopt;
    }
    const auto low_eq =
        build_v_cmp_eq_u32_e32_vcc(vector_source_vgpr(address_vgpr), value_vgpr, arch);
    if (!low_eq) {
      errors.emplace_back("ConSan MOI inline atomic acquire patch could not compare address low");
      return std::nullopt;
    }
    words.push_back(*low_eq);
    words.push_back(*narrow_if_low_matches);

    if (!append_load_u32_vgpr(words,
                              slot_base +
                                  offsetof(ConSanMoiInlineAtomicReleaseSlot, atomic_address) +
                                  sizeof(uint32_t),
                              value_vgpr, *options.scratch_vgpr, arch)) {
      errors.emplace_back("ConSan MOI inline atomic acquire patch could not load address high");
      return std::nullopt;
    }
    const auto high_eq = build_v_cmp_eq_u32_e32_vcc(
        vector_source_vgpr(static_cast<uint16_t>(address_vgpr + 1u)), value_vgpr, arch);
    if (!high_eq) {
      errors.emplace_back("ConSan MOI inline atomic acquire patch could not compare address high");
      return std::nullopt;
    }
    words.push_back(*high_eq);
    words.push_back(*narrow_if_high_matches);

    if (!append_load_u32_vgpr(words,
                              slot_base + offsetof(ConSanMoiInlineAtomicReleaseSlot, owner_id),
                              value_vgpr, *options.scratch_vgpr, arch)) {
      errors.emplace_back("ConSan MOI inline atomic acquire patch could not load owner");
      return std::nullopt;
    }
    const auto owner_ne =
        build_v_cmp_ne_u32_e32_vcc(vector_source_vgpr(*options.moi_owner_vgpr), value_vgpr, arch);
    if (!owner_ne) {
      errors.emplace_back("ConSan MOI inline atomic acquire patch could not compare owner");
      return std::nullopt;
    }
    words.push_back(*owner_ne);
    words.push_back(*narrow_if_other_owner);

    if (!append_load_u32_vgpr(words, slot_base + offsetof(ConSanMoiInlineAtomicReleaseSlot, epoch),
                              value_vgpr, *options.scratch_vgpr, arch)) {
      errors.emplace_back("ConSan MOI inline atomic acquire patch could not load epoch");
      return std::nullopt;
    }
    const auto import_epoch = build_v_add_nc_u32_e32(
        *options.moi_epoch_vgpr, scalar_positive_inline_u32(1), value_vgpr, arch);
    if (!import_epoch) {
      errors.emplace_back("ConSan MOI inline atomic acquire patch could not import epoch");
      return std::nullopt;
    }
    words.push_back(*import_epoch);
    words.push_back(*restore_exec);
    words.push_back(*restore_vcc_lo);
    words.push_back(*restore_vcc_hi);
    words.push_back(kWaitAluDepctrSaSdst0);
  }

  const uint64_t branch_pc =
      cave_text_offset + static_cast<uint64_t>(words.size()) * sizeof(uint32_t);
  const auto ret = compute_sopp_branch_simm16(branch_pc, return_text_offset);
  if (!ret) {
    errors.emplace_back("ConSan MOI inline atomic return branch is out of range");
    return std::nullopt;
  }
  words.push_back(build_s_branch(*ret, arch));
  return words;
}

void try_apply_inline_atomic_ordering_patch(std::span<const uint8_t> bytes,
                                            const ConSanOptions &options, rj_code_arch_t arch,
                                            ConSanResult &result) {
  if (!options.moi_track_atomics || options.moi_engine != ConSanMoiEngine::InlineShadow)
    return;
  if (arch != ROCJITSU_CODE_ARCH_RDNA4) {
    result.warnings.emplace_back("ConSan MOI inline atomic patch currently supports only RDNA4");
    return;
  }
  if (!options.moi_report_buffer_address) {
    result.warnings.emplace_back("ConSan MOI inline atomic patch requires a MOI report buffer");
    return;
  }
  const ConSanMoiReportBufferLayout layout =
      consan_moi_inline_shadow_report_buffer_layout_for_bytes(options.moi_report_buffer_size);
  if (layout.inline_atomic_release_slots_offset == layout.sampled_watchpoints_offset) {
    result.warnings.emplace_back("ConSan MOI inline atomic patch has no release metadata slot");
    return;
  }
  if (!options.scratch_vgpr) {
    result.warnings.emplace_back("ConSan MOI inline atomic patch requires RJ_CONSAN_TMP_VGPR");
    return;
  }
  if (!options.moi_owner_vgpr || !options.moi_epoch_vgpr) {
    result.warnings.emplace_back("ConSan MOI inline atomic patch requires "
                                 "RJ_CONSAN_MOI_OWNER_VGPR and RJ_CONSAN_MOI_EPOCH_VGPR");
    return;
  }
  if (!validate_inline_atomic_exec_save_sgpr(options, result.warnings))
    return;

  std::vector<AtomicRecordCandidate> candidates;
  for (const ConSanKernelInfo &kernel : result.kernels)
    append_atomic_record_candidates(kernel, candidates);
  for (const ConSanFunctionInfo &function : result.functions)
    append_atomic_record_candidates(function, candidates);
  std::sort(candidates.begin(), candidates.end(), [](const auto &lhs, const auto &rhs) {
    return lhs.site.text_offset < rhs.site.text_offset;
  });
  if (candidates.empty()) {
    result.warnings.emplace_back("ConSan MOI inline atomic patch found no supported flat atomics");
    return;
  }

  std::span<const uint8_t> active_bytes = bytes;
  if (result.modified)
    active_bytes = std::span<const uint8_t>(result.elf_bytes.data(), result.elf_bytes.size());
  AmdGpuCodeObject code_object(active_bytes.data(), active_bytes.size());
  CodeObjectPatcher patcher(code_object);
  const std::span<const uint8_t> old_text = patcher.text_bytes();
  if (old_text.empty()) {
    result.errors.emplace_back("ConSan MOI inline atomic patch found no .text section");
    return;
  }
  uint32_t required_vgpr_count = static_cast<uint32_t>(*options.scratch_vgpr) + 5u;
  required_vgpr_count =
      std::max<uint32_t>(required_vgpr_count, static_cast<uint32_t>(*options.moi_owner_vgpr) + 1u);
  required_vgpr_count =
      std::max<uint32_t>(required_vgpr_count, static_cast<uint32_t>(*options.moi_epoch_vgpr) + 1u);
  if (!grow_moi_kernel_descriptor_vgprs(patcher, active_bytes, result, required_vgpr_count))
    return;

  std::vector<uint8_t> new_text(old_text.begin(), old_text.end());
  std::vector<ConSanPatchInfo> patches;
  for (const AtomicRecordCandidate &candidate : candidates) {
    if (patches.size() == options.max_patches)
      break;
    const ConSanAtomicSite &site = candidate.site;
    if (site.text_offset > old_text.size() || site.size > old_text.size() - site.text_offset) {
      result.errors.emplace_back("ConSan MOI inline atomic patch site is outside .text");
      return;
    }
    const auto overlaps_existing_patch = std::any_of(
        result.patches.begin(), result.patches.end(), [&](const ConSanPatchInfo &patch) {
          const uint64_t patch_end =
              patch.anchor_offset + std::max<uint32_t>(patch.original_size, 1);
          return site.text_offset < patch_end && patch.anchor_offset < site.text_offset + site.size;
        });
    if (overlaps_existing_patch) {
      result.warnings.emplace_back(
          "ConSan MOI inline atomic patch skipped an overlapping patch in " +
          candidate.container_name);
      continue;
    }

    const uint64_t cave_text_offset = static_cast<uint64_t>(new_text.size());
    const uint64_t return_text_offset = site.text_offset + site.size;
    auto cave_words = build_inline_atomic_ordering_cave_words(
        active_bytes, candidate, options, arch, layout.inline_atomic_release_slots_offset,
        cave_text_offset, return_text_offset, result.errors);
    if (!cave_words)
      return;

    const auto fwd = compute_sopp_branch_simm16(site.text_offset, cave_text_offset);
    if (!fwd) {
      result.errors.emplace_back("ConSan MOI inline atomic forward branch is out of range");
      return;
    }
    std::vector<uint32_t> anchor_words(site.size / sizeof(uint32_t), build_s_nop(0, arch));
    if (anchor_words.empty()) {
      result.errors.emplace_back("ConSan MOI inline atomic patch empty anchor");
      return;
    }
    anchor_words.front() = build_s_branch(*fwd, arch);
    std::memcpy(new_text.data() + site.text_offset, anchor_words.data(),
                anchor_words.size() * sizeof(uint32_t));
    append_words_bytes(new_text, *cave_words);

    ConSanPatchInfo info;
    info.kind = ConSanPatchKind::TrampolineMoiInlineAtomicOrdering;
    info.anchor_offset = site.text_offset;
    info.trampoline_offset = cave_text_offset;
    info.original_size = site.size;
    info.trampoline_size = static_cast<uint32_t>(cave_words->size() * sizeof(uint32_t));
    info.scratch_vgpr = options.scratch_vgpr;
    patches.push_back(info);
  }

  if (patches.empty()) {
    result.warnings.emplace_back("ConSan MOI inline atomic patch found no patchable atomics");
    return;
  }
  if (!patcher.replace_text(new_text)) {
    result.errors.emplace_back("ConSan MOI inline atomic patch could not grow .text");
    return;
  }
  result.elf_bytes = patcher.emit();
  result.patches.insert(result.patches.end(), patches.begin(), patches.end());
  result.modified = true;
}

[[nodiscard]] std::optional<std::vector<uint32_t>>
build_atomic_record_cave_words(std::span<const uint8_t> bytes,
                               const AtomicRecordCandidate &candidate, const ConSanOptions &options,
                               rj_code_arch_t arch, uint32_t record_index, uint32_t record_count,
                               size_t atomic_records_offset, uint64_t cave_text_offset,
                               uint64_t return_text_offset, std::vector<std::string> &errors) {
  if (!options.scratch_vgpr) {
    errors.emplace_back("ConSan MOI atomic record patch requires RJ_CONSAN_TMP_VGPR");
    return std::nullopt;
  }
  if (static_cast<uint32_t>(*options.scratch_vgpr) + 3u > 256u) {
    errors.emplace_back("ConSan MOI atomic record patch needs three scratch VGPRs");
    return std::nullopt;
  }
  if (reject_atomic_candidate_scratch_overlap(candidate.site, *options.scratch_vgpr, errors) ||
      reject_optional_scratch_overlap(options.moi_owner_vgpr, *options.scratch_vgpr, "MOI owner",
                                      errors) ||
      reject_optional_scratch_overlap(options.moi_epoch_vgpr, *options.scratch_vgpr, "MOI epoch",
                                      errors))
    return std::nullopt;

  std::optional<uint16_t> derived_owner_vgpr;
  std::optional<uint32_t> derived_owner_word;
  if (!options.moi_owner_vgpr) {
    if (!candidate.kernel_descriptor_file_offset) {
      errors.emplace_back(
          "ConSan MOI atomic record patch requires RJ_CONSAN_MOI_OWNER_VGPR for function atomics");
      return std::nullopt;
    }
    const auto owner_shift =
        moi_descriptor_owner_shift(bytes, *candidate.kernel_descriptor_file_offset, errors);
    if (!owner_shift)
      return std::nullopt;
    const uint16_t value_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 2u);
    const auto owner_init = build_v_lshrrev_b32_e32(
        value_vgpr, scalar_positive_inline_u32(*owner_shift), kRdna4WorkitemIdX, arch);
    if (!owner_init) {
      errors.emplace_back("ConSan MOI atomic record patch could not encode owner derivation");
      return std::nullopt;
    }
    derived_owner_vgpr = value_vgpr;
    derived_owner_word = *owner_init;
  }

  ConSanMoiWorkgroupSources workgroup_sources;
  if (candidate.kernel_descriptor_file_offset) {
    const auto descriptor_workgroup_sources = moi_descriptor_workgroup_sources(
        bytes, *candidate.kernel_descriptor_file_offset, arch, errors);
    if (!descriptor_workgroup_sources)
      return std::nullopt;
    workgroup_sources = *descriptor_workgroup_sources;
  }

  if (candidate.site.file_offset > bytes.size() ||
      candidate.site.size > bytes.size() - candidate.site.file_offset) {
    errors.emplace_back("ConSan MOI atomic record patch site exceeds ELF bytes");
    return std::nullopt;
  }

  std::vector<uint32_t> words;
  words.reserve(candidate.site.size / sizeof(uint32_t) + 96u);
  if (derived_owner_word)
    words.push_back(*derived_owner_word);
  for (uint64_t offset = 0; offset < candidate.site.size; offset += sizeof(uint32_t)) {
    uint32_t word = 0;
    std::memcpy(&word, bytes.data() + candidate.site.file_offset + offset, sizeof(word));
    words.push_back(word);
  }
  words.push_back(kWaitLoadcnt0);

  const uint64_t base = *options.moi_report_buffer_address;
  const uint64_t atomic_record_base =
      base + atomic_records_offset +
      static_cast<uint64_t>(record_index) * sizeof(ConSanMoiAtomicRecord);
  const uint16_t address_vgpr = *candidate.site.addr_vgpr;
  if ((derived_owner_vgpr &&
       !append_store_u32_vgpr(words, atomic_record_base + offsetof(ConSanMoiAtomicRecord, owner_id),
                              *derived_owner_vgpr, *options.scratch_vgpr, arch)) ||
      !append_atomic_event_index_store(words, base + offsetof(ConSanMoiReportHeader, event_counter),
                                       atomic_record_base +
                                           offsetof(ConSanMoiAtomicRecord, event_index),
                                       *options.scratch_vgpr, arch) ||
      !append_store_u32_literal(words, base + offsetof(ConSanMoiReportHeader, atomic_record_count),
                                record_count, *options.scratch_vgpr, arch) ||
      !append_store_workgroup_source(
          words, atomic_record_base + offsetof(ConSanMoiAtomicRecord, workgroup_x),
          workgroup_sources.x, *options.scratch_vgpr, arch) ||
      !append_store_workgroup_source(
          words, atomic_record_base + offsetof(ConSanMoiAtomicRecord, workgroup_y),
          workgroup_sources.y, *options.scratch_vgpr, arch) ||
      !append_store_workgroup_source(
          words, atomic_record_base + offsetof(ConSanMoiAtomicRecord, workgroup_z),
          workgroup_sources.z, *options.scratch_vgpr, arch) ||
      (options.moi_owner_vgpr &&
       !append_store_u32_vgpr(words, atomic_record_base + offsetof(ConSanMoiAtomicRecord, owner_id),
                              *options.moi_owner_vgpr, *options.scratch_vgpr, arch)) ||
      (options.moi_epoch_vgpr &&
       !append_store_u32_vgpr(words, atomic_record_base + offsetof(ConSanMoiAtomicRecord, epoch),
                              *options.moi_epoch_vgpr, *options.scratch_vgpr, arch)) ||
      !append_store_u32_vgpr(words,
                             atomic_record_base + offsetof(ConSanMoiAtomicRecord, atomic_address),
                             address_vgpr, *options.scratch_vgpr, arch) ||
      !append_store_u32_vgpr(
          words,
          atomic_record_base + offsetof(ConSanMoiAtomicRecord, atomic_address) + sizeof(uint32_t),
          static_cast<uint16_t>(address_vgpr + 1u), *options.scratch_vgpr, arch) ||
      !append_store_u32_literal(
          words, atomic_record_base + offsetof(ConSanMoiAtomicRecord, instruction_offset),
          static_cast<uint32_t>(candidate.site.text_offset), *options.scratch_vgpr, arch) ||
      !append_store_u32_literal(words, atomic_record_base + offsetof(ConSanMoiAtomicRecord, kind),
                                static_cast<uint32_t>(atomic_event_kind_for_site(candidate.site)),
                                *options.scratch_vgpr, arch) ||
      !append_store_u32_literal(words, atomic_record_base + offsetof(ConSanMoiAtomicRecord, scope),
                                *candidate.site.raw_scope, *options.scratch_vgpr, arch) ||
      !append_store_u32_literal(words,
                                atomic_record_base + offsetof(ConSanMoiAtomicRecord, semantics),
                                *candidate.site.raw_th, *options.scratch_vgpr, arch)) {
    errors.emplace_back("ConSan MOI atomic record patch could not encode record stores");
    return std::nullopt;
  }

  const uint64_t branch_pc =
      cave_text_offset + static_cast<uint64_t>(words.size()) * sizeof(uint32_t);
  const auto ret = compute_sopp_branch_simm16(branch_pc, return_text_offset);
  if (!ret) {
    errors.emplace_back("ConSan MOI atomic record return branch is out of range");
    return std::nullopt;
  }
  words.push_back(build_s_branch(*ret, arch));
  return words;
}

void try_apply_atomic_record_patch(std::span<const uint8_t> bytes, const ConSanOptions &options,
                                   rj_code_arch_t arch, ConSanResult &result) {
  if (!options.moi_track_atomics)
    return;
  if (options.moi_engine == ConSanMoiEngine::InlineShadow)
    return;
  if (arch != ROCJITSU_CODE_ARCH_RDNA4) {
    result.warnings.emplace_back("ConSan MOI atomic record patch currently supports only RDNA4");
    return;
  }
  if (!options.moi_report_buffer_address) {
    result.warnings.emplace_back("ConSan MOI atomic record patch requires a MOI report buffer");
    return;
  }
  if (options.moi_report_buffer_size < consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 0, 1)) {
    result.warnings.emplace_back(
        "ConSan MOI atomic record patch requires room for access and atomic records");
    return;
  }
  const ConSanMoiReportBufferLayout layout = consan_moi_report_buffer_layout_for_bytes(
      options.moi_report_buffer_size, options.moi_track_barriers,
      /*include_atomics=*/true);
  const uint32_t max_candidates = std::min(options.max_patches, layout.atomic_record_capacity);
  if (max_candidates == 0) {
    result.warnings.emplace_back("ConSan MOI atomic record patch has no atomic record capacity");
    return;
  }
  if (!options.scratch_vgpr) {
    result.warnings.emplace_back("ConSan MOI atomic record patch requires RJ_CONSAN_TMP_VGPR");
    return;
  }
  if (static_cast<uint32_t>(*options.scratch_vgpr) + 3u > 256u) {
    result.warnings.emplace_back("ConSan MOI atomic record patch needs three scratch VGPRs");
    return;
  }

  std::vector<AtomicRecordCandidate> candidates;
  for (const ConSanKernelInfo &kernel : result.kernels)
    append_atomic_record_candidates(kernel, candidates);
  for (const ConSanFunctionInfo &function : result.functions)
    append_atomic_record_candidates(function, candidates);
  std::sort(candidates.begin(), candidates.end(), [](const auto &lhs, const auto &rhs) {
    return lhs.site.text_offset < rhs.site.text_offset;
  });
  if (candidates.empty()) {
    result.warnings.emplace_back("ConSan MOI atomic record patch found no supported flat atomics");
    return;
  }

  std::span<const uint8_t> active_bytes = bytes;
  if (result.modified)
    active_bytes = std::span<const uint8_t>(result.elf_bytes.data(), result.elf_bytes.size());
  AmdGpuCodeObject code_object(active_bytes.data(), active_bytes.size());
  CodeObjectPatcher patcher(code_object);
  const std::span<const uint8_t> old_text = patcher.text_bytes();
  if (old_text.empty()) {
    result.errors.emplace_back("ConSan MOI atomic record patch found no .text section");
    return;
  }
  uint32_t required_vgpr_count = static_cast<uint32_t>(*options.scratch_vgpr) + 3u;
  if (options.moi_owner_vgpr)
    required_vgpr_count = std::max<uint32_t>(required_vgpr_count,
                                             static_cast<uint32_t>(*options.moi_owner_vgpr) + 1u);
  if (options.moi_epoch_vgpr)
    required_vgpr_count = std::max<uint32_t>(required_vgpr_count,
                                             static_cast<uint32_t>(*options.moi_epoch_vgpr) + 1u);
  if (!grow_moi_kernel_descriptor_vgprs(patcher, active_bytes, result, required_vgpr_count))
    return;

  std::vector<AtomicRecordCandidate> selected_candidates;
  selected_candidates.reserve(max_candidates);
  for (const AtomicRecordCandidate &candidate : candidates) {
    if (selected_candidates.size() == max_candidates)
      break;
    const ConSanAtomicSite &site = candidate.site;
    if (site.text_offset > old_text.size() || site.size > old_text.size() - site.text_offset) {
      result.errors.emplace_back("ConSan MOI atomic record patch site is outside .text");
      return;
    }
    if (!options.moi_owner_vgpr && !candidate.kernel_descriptor_file_offset) {
      result.warnings.emplace_back("ConSan MOI atomic record patch skipped function atomic in " +
                                   candidate.container_name +
                                   " because no owner VGPR was configured");
      continue;
    }
    const auto overlaps_existing_patch = std::any_of(
        result.patches.begin(), result.patches.end(), [&](const ConSanPatchInfo &patch) {
          const uint64_t patch_end =
              patch.anchor_offset + std::max<uint32_t>(patch.original_size, 1);
          return site.text_offset < patch_end && patch.anchor_offset < site.text_offset + site.size;
        });
    if (overlaps_existing_patch) {
      result.warnings.emplace_back(
          "ConSan MOI atomic record patch skipped an overlapping patch in " +
          candidate.container_name);
      continue;
    }
    selected_candidates.push_back(candidate);
  }
  if (selected_candidates.empty()) {
    result.warnings.emplace_back("ConSan MOI atomic record patch found no patchable atomics");
    return;
  }

  std::vector<uint8_t> new_text(old_text.begin(), old_text.end());
  std::vector<ConSanPatchInfo> patches;
  const uint32_t record_count = static_cast<uint32_t>(selected_candidates.size());
  uint32_t record_index = 0;
  for (const AtomicRecordCandidate &candidate : selected_candidates) {
    const ConSanAtomicSite &site = candidate.site;
    const uint64_t cave_text_offset = static_cast<uint64_t>(new_text.size());
    const uint64_t return_text_offset = site.text_offset + site.size;
    auto cave_words = build_atomic_record_cave_words(
        active_bytes, candidate, options, arch, record_index, record_count,
        layout.atomic_records_offset, cave_text_offset, return_text_offset, result.errors);
    if (!cave_words)
      return;

    const auto fwd = compute_sopp_branch_simm16(site.text_offset, cave_text_offset);
    if (!fwd) {
      result.errors.emplace_back("ConSan MOI atomic record forward branch is out of range");
      return;
    }
    std::vector<uint32_t> anchor_words(site.size / sizeof(uint32_t), build_s_nop(0, arch));
    if (anchor_words.empty()) {
      result.errors.emplace_back("ConSan MOI atomic record patch empty anchor");
      return;
    }
    anchor_words.front() = build_s_branch(*fwd, arch);
    if (site.text_offset > new_text.size() ||
        anchor_words.size() * sizeof(uint32_t) > new_text.size() - site.text_offset) {
      result.errors.emplace_back("ConSan MOI atomic record anchor exceeds .text");
      return;
    }
    std::memcpy(new_text.data() + site.text_offset, anchor_words.data(),
                anchor_words.size() * sizeof(uint32_t));
    append_words_bytes(new_text, *cave_words);

    ConSanPatchInfo info;
    info.kind = ConSanPatchKind::TrampolineMoiAtomicRecord;
    info.anchor_offset = site.text_offset;
    info.trampoline_offset = cave_text_offset;
    info.original_size = site.size;
    info.trampoline_size = static_cast<uint32_t>(cave_words->size() * sizeof(uint32_t));
    info.scratch_vgpr = options.scratch_vgpr;
    patches.push_back(info);
    ++record_index;
  }

  if (!patcher.replace_text(new_text)) {
    result.errors.emplace_back("ConSan MOI atomic record patch could not grow .text");
    return;
  }
  result.elf_bytes = patcher.emit();
  result.patches.insert(result.patches.end(), patches.begin(), patches.end());
  result.modified = true;
}

} // namespace

bool consan_moi_supports_native_lds_record_replay_mnemonic(std::string_view mnemonic) {
  return is_single_range_native_lds_mnemonic(mnemonic) ||
         two_address_native_lds_offset_scale(mnemonic).has_value();
}

ConSanResult try_patch_consan_moi(ConSanResult result, const ConSanOptions &options,
                                  std::span<const uint8_t> code_object_bytes, rj_code_arch_t arch) {
  result.flavor = ConSanFlavor::Moi;
  result.moi_engine = options.moi_engine;
  result.modified = false;
  result.elf_bytes.clear();
  result.moi_candidates.clear();
  for (const ConSanKernelInfo &kernel : result.kernels)
    append_moi_candidates(kernel, result);
  for (const ConSanFunctionInfo &function : result.functions)
    append_moi_candidates(function, result);
  append_moi_resource_plans(code_object_bytes, options, arch, result);
  if (options.moi_report_buffer_address &&
      options.moi_report_buffer_size < sizeof(ConSanMoiReportHeader)) {
    result.warnings.emplace_back("ConSan MOI report buffer is smaller than the report ABI header");
  }
  if (result.errors.empty() && options.moi_engine == ConSanMoiEngine::Sampled)
    try_apply_direct_sampled_watchpoint_patch(code_object_bytes, options, arch, result);
  if (result.errors.empty() && options.moi_engine == ConSanMoiEngine::InlineShadow)
    try_apply_inline_shadow_patch(code_object_bytes, options, arch, result);
  if (result.errors.empty() && options.moi_engine == ConSanMoiEngine::RecordReplay)
    try_apply_first_light_access_record_patch(code_object_bytes, options, arch, result);
  if (result.errors.empty())
    try_apply_barrier_epoch_patch(code_object_bytes, options, arch, result);
  if (result.errors.empty())
    try_apply_inline_atomic_ordering_patch(code_object_bytes, options, arch, result);
  if (result.errors.empty())
    try_apply_atomic_record_patch(code_object_bytes, options, arch, result);
  if (result.errors.empty() &&
      (options.moi_engine != ConSanMoiEngine::InlineShadow || result.modified))
    try_apply_owner_epoch_prologue_patch(code_object_bytes, options, arch, result);
  if (result.modified) {
    const auto patch_count = [&result](ConSanPatchKind kind) {
      return static_cast<uint32_t>(
          std::count_if(result.patches.begin(), result.patches.end(),
                        [kind](const ConSanPatchInfo &patch) { return patch.kind == kind; }));
    };
    if (patch_count(ConSanPatchKind::InlineMoiAccessRecordStore) != 0) {
      result.warnings.emplace_back(std::string("ConSan MOI ") +
                                   consan_moi_engine_name(options.moi_engine) +
                                   " engine emitted a first-light access record probe");
    }
    if (patch_count(ConSanPatchKind::TrampolineMoiAccessRecordStore) != 0) {
      result.warnings.emplace_back(std::string("ConSan MOI ") +
                                   consan_moi_engine_name(options.moi_engine) +
                                   " engine emitted an appended-cave first-light access record "
                                   "probe");
    }
    if (patch_count(ConSanPatchKind::InlineMoiExactShadowStore) != 0) {
      result.warnings.emplace_back("ConSan MOI inline-shadow engine emitted an exact-shadow "
                                   "publish probe");
    }
    if (patch_count(ConSanPatchKind::TrampolineMoiExactShadowStore) != 0) {
      result.warnings.emplace_back("ConSan MOI inline-shadow engine emitted an appended-cave "
                                   "exact-shadow publish probe");
    }
    if (patch_count(ConSanPatchKind::InlineMoiSampledWatchpointStore) != 0) {
      result.warnings.emplace_back("ConSan MOI sampled engine emitted a direct sampled "
                                   "watchpoint probe");
    }
    if (patch_count(ConSanPatchKind::TrampolineMoiSampledWatchpointStore) != 0) {
      result.warnings.emplace_back("ConSan MOI sampled engine emitted an appended-cave direct "
                                   "sampled watchpoint probe");
    }
    if (patch_count(ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue) != 0) {
      result.warnings.emplace_back(
          "ConSan MOI initialized owner/epoch VGPRs with a kernel-entry prologue");
    }
    if (const uint32_t barrier_patches = patch_count(ConSanPatchKind::TrampolineMoiBarrierRecord);
        barrier_patches != 0) {
      result.warnings.emplace_back("ConSan MOI emitted " + std::to_string(barrier_patches) +
                                   " barrier record probe(s)");
    }
    if (const uint32_t inline_epoch_barriers =
            patch_count(ConSanPatchKind::TrampolineMoiInlineEpochBarrier);
        inline_epoch_barriers != 0) {
      result.warnings.emplace_back("ConSan MOI inline-shadow engine emitted " +
                                   std::to_string(inline_epoch_barriers) +
                                   " barrier epoch probe(s)");
    }
    if (const uint32_t inline_atomic_patches =
            patch_count(ConSanPatchKind::TrampolineMoiInlineAtomicOrdering);
        inline_atomic_patches != 0) {
      result.warnings.emplace_back("ConSan MOI inline-shadow engine emitted " +
                                   std::to_string(inline_atomic_patches) +
                                   " inline atomic ordering probe(s)");
    }
    if (const uint32_t atomic_patches = patch_count(ConSanPatchKind::TrampolineMoiAtomicRecord);
        atomic_patches != 0) {
      result.warnings.emplace_back("ConSan MOI emitted " + std::to_string(atomic_patches) +
                                   " atomic record probe(s)");
    }
  } else {
    result.warnings.emplace_back(std::string("ConSan MOI ") +
                                 consan_moi_engine_name(options.moi_engine) +
                                 " engine is an inventory-only stub");
  }
  return result;
}

} // namespace rocjitsu
