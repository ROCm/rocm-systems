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
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>
#include <unordered_map>
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
  const uint32_t active_generation =
      static_cast<uint32_t>(header.generation) & consan_moi_sampled_watchpoint::max_generation;

  for (uint32_t i = 0; i < entry_count; ++i) {
    const ConSanMoiSampledWatchpointEntry current =
        decode_consan_moi_sampled_watchpoint_entry(sampled_watchpoint_entries[i]);
    ++replay.processed_entry_count;
    if (!current.valid || current.generation != active_generation)
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

ConSanMoiDescriptorRegisterGeometry
consan_moi_descriptor_register_geometry(rj_code_arch_t arch, bool descriptor_wave32) {
  ConSanMoiDescriptorRegisterGeometry geometry;
  if (arch == ROCJITSU_CODE_ARCH_CDNA4) {
    geometry.wavefront_size = 64;
    geometry.max_sgpr_count = 102;
    geometry.vgpr_encoding_granularity = 8;
    return geometry;
  }
  geometry.wavefront_size = descriptor_wave32 ? 32 : 64;
  geometry.vgpr_encoding_granularity = descriptor_wave32 ? 8 : 4;
  return geometry;
}

std::optional<ConSanMoiSpecialRegisterGeometry>
consan_moi_special_register_geometry(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA4:
  case ROCJITSU_CODE_ARCH_RDNA4:
  case ROCJITSU_CODE_ARCH_GFX1250:
    return ConSanMoiSpecialRegisterGeometry{};
  default:
    return std::nullopt;
  }
}

uint32_t consan_moi_decode_descriptor_register_count(uint32_t granulated_count, uint32_t max_count,
                                                     uint32_t granularity) {
  if (granularity == 0)
    return 0;
  const uint64_t allocated = (static_cast<uint64_t>(granulated_count) + 1u) * granularity;
  return static_cast<uint32_t>(std::min<uint64_t>(allocated, max_count));
}

std::optional<uint32_t> consan_moi_encode_descriptor_register_count(uint32_t required_count,
                                                                    uint32_t max_count,
                                                                    uint32_t granularity) {
  if (required_count == 0 || required_count > max_count || granularity == 0)
    return std::nullopt;
  const uint64_t rounded =
      ((static_cast<uint64_t>(required_count) + granularity - 1u) / granularity) * granularity;
  return static_cast<uint32_t>(rounded / granularity - 1u);
}

namespace {

// CDNA4 and RDNA4 share these wave64 architectural operand numbers. Ordinary
// SGPR allocation remains architecture-specific and never includes them.
inline constexpr uint16_t kWave64ExecLo = 126;
inline constexpr uint16_t kWave64ExecHi = 127;
inline constexpr uint16_t kWave64VccLo = 106;
inline constexpr uint16_t kWorkitemIdX = 0;
inline constexpr uint16_t kScalarOperandTtmpBase = 108;
inline constexpr uint16_t kTtmpRdna4GridYz = 7;
inline constexpr uint16_t kTtmpRdna4GridX = 9;
inline constexpr uint32_t kMaxVgprs = 256;
inline constexpr uint32_t kMaxRdna4Sgprs = 106;
inline constexpr uint32_t kMaxCdna4Sgprs = 102;
inline constexpr uint16_t kGfx12HwRegHwId1 = 23;
inline constexpr uint16_t kGfx12HwIdOwnerBits = 10;
inline constexpr uint8_t kRdna4ScopeDevice = 2;
inline constexpr uint32_t kRdna4FlatNoSaddr = 0x7C;
inline constexpr uint64_t kAmdhsaKernelEntryAlignment = 256;

[[nodiscard]] bool require_native_feature(rj_code_arch_t arch, ConSanNativeFeature feature,
                                          std::string_view context, ConSanResult &result) {
  const ConSanNativeSupport support =
      consan_native_feature_support(consan_target_capabilities(arch), feature);
  if (support == ConSanNativeSupport::NativeEmission)
    return true;
  result.warnings.emplace_back("ConSan capability skip context=" + std::string(context) +
                               " feature=" + consan_native_feature_name(feature) +
                               " support=" + consan_native_support_name(support));
  return false;
}

[[nodiscard]] constexpr uint16_t ttmp_scalar_operand(uint16_t ttmp) {
  return static_cast<uint16_t>(kScalarOperandTtmpBase + ttmp);
}

[[nodiscard]] constexpr uint32_t moi_max_ordinary_sgprs(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_CDNA4 ? kMaxCdna4Sgprs : kMaxRdna4Sgprs;
}

struct ConSanMoiWorkgroupSource {
  std::optional<uint16_t> scalar_src;
  std::optional<uint16_t> vector_src;
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

[[nodiscard]] std::vector<ConSanLdsStaticRange>
normalized_native_lds_ranges(const ConSanLdsSite &site) {
  if (site.width_bits == 0 || site.width_bits % 8u != 0 || !site.raw_offset0 || !site.raw_offset1)
    return {};
  const uint32_t byte_count = site.width_bits / 8u;
  if (is_single_range_native_lds_mnemonic(site.mnemonic))
    return {{*site.raw_offset0 | (*site.raw_offset1 << 8u), byte_count}};
  const std::optional<uint32_t> scale = two_address_native_lds_offset_scale(site.mnemonic);
  if (!scale)
    return {};
  return {{*site.raw_offset0 * *scale, byte_count}, {*site.raw_offset1 * *scale, byte_count}};
}

[[nodiscard]] bool is_moi_native_lds_candidate(const ConSanLdsSite &site) {
  if (site.kind != ConSanLdsAccessKind::Read && site.kind != ConSanLdsAccessKind::Write)
    return false;
  if (site.raw_gds.value_or(false))
    return false;
  return is_single_range_native_lds_mnemonic(site.mnemonic) ||
         two_address_native_lds_offset_scale(site.mnemonic).has_value();
}

[[nodiscard]] ConSanMoiLdsExclusionReason moi_lds_exclusion_reason(const ConSanLdsSite &site) {
  if (site.raw_gds.value_or(false))
    return ConSanMoiLdsExclusionReason::GdsReserved;
  if (site.kind == ConSanLdsAccessKind::Atomic)
    return ConSanMoiLdsExclusionReason::AtomicReserved;
  if (site.mnemonic.find("_tr_") != std::string::npos)
    return ConSanMoiLdsExclusionReason::Transpose;
  if (equals_any(site.mnemonic, {"ds_swizzle_b32", "ds_permute_b32", "ds_bpermute_b32"}))
    return ConSanMoiLdsExclusionReason::PermuteOrSwizzle;
  if (site.kind == ConSanLdsAccessKind::Read || site.kind == ConSanLdsAccessKind::Write)
    return ConSanMoiLdsExclusionReason::UnsupportedAccessForm;
  return ConSanMoiLdsExclusionReason::OtherDs;
}

void append_moi_lds_exclusion(std::string_view container_name, bool in_kernel,
                              const ConSanLdsSite &site, ConSanResult &result) {
  result.moi_lds_exclusions.push_back({std::string(container_name), in_kernel, site.text_offset,
                                       moi_lds_exclusion_reason(site), site.mnemonic});
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

[[nodiscard]] bool is_moi_flat_candidate(const ConSanFlatSite &site,
                                         ConSanFlatProvenanceMode mode) {
  if (site.kind == ConSanLdsAccessKind::Other)
    return false;
  return site.address_space_hint == ConSanFlatAddressSpaceHint::Group ||
         (mode == ConSanFlatProvenanceMode::Likely &&
          site.address_space_hint == ConSanFlatAddressSpaceHint::MaybeGroup);
}

struct SkippedMoiFlatCounts {
  uint32_t other_kind = 0;
  uint32_t unknown = 0;
  uint32_t private_hint = 0;
  uint32_t maybe_private = 0;
  uint32_t global = 0;
  uint32_t excluded_maybe_group = 0;

  [[nodiscard]] bool any() const {
    return other_kind != 0 || unknown != 0 || private_hint != 0 || maybe_private != 0 ||
           global != 0 || excluded_maybe_group != 0;
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
      std::string(container_name) + " because the flat provenance policy did not admit them: " +
      "unknown=" + std::to_string(counts.unknown) +
      " private=" + std::to_string(counts.private_hint) + " maybe_private=" +
      std::to_string(counts.maybe_private) + " global=" + std::to_string(counts.global) +
      " excluded_maybe_group=" + std::to_string(counts.excluded_maybe_group) +
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
  candidate.secondary_data_vgpr = site.secondary_data_vgpr;
  candidate.raw_offset0 = site.raw_offset0;
  candidate.raw_offset1 = site.raw_offset1;
  candidate.raw_op = site.raw_op;
  candidate.raw_gds = site.raw_gds;
  candidate.native_static_ranges = normalized_native_lds_ranges(site);
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
  candidate.raw_op = site.raw_op;
  candidate.raw_saddr = site.raw_saddr;
  candidate.raw_vaddr = site.raw_vaddr;
  candidate.raw_vsrc = site.raw_vsrc;
  candidate.raw_vdst = site.raw_vdst;
  candidate.raw_ioffset = site.raw_ioffset;
  candidate.raw_segment = site.raw_segment;
  candidate.raw_scope = site.raw_scope;
  candidate.raw_th = site.raw_th;
  candidate.mnemonic = site.mnemonic;
  return candidate;
}

void append_moi_candidates(const ConSanKernelInfo &kernel, ConSanFlatProvenanceMode mode,
                           ConSanResult &result) {
  SkippedMoiLdsCounts skipped_lds_counts;
  for (const ConSanLdsSite &site : kernel.lds_sites) {
    if (is_moi_native_lds_candidate(site)) {
      result.moi_candidates.push_back(
          make_moi_candidate(kernel.name, true, kernel.descriptor_file_offset, site));
    } else {
      append_moi_lds_exclusion(kernel.name, true, site, result);
      count_skipped_moi_lds_candidate(site, skipped_lds_counts);
    }
  }
  warn_skipped_moi_lds_candidates(kernel.name, true, skipped_lds_counts, result);
  SkippedMoiFlatCounts skipped_flat_counts;
  for (const ConSanFlatSite &site : kernel.flat_sites) {
    if (is_moi_flat_candidate(site, mode)) {
      result.moi_candidates.push_back(
          make_moi_candidate(kernel.name, true, kernel.descriptor_file_offset, site));
    } else if (site.address_space_hint == ConSanFlatAddressSpaceHint::MaybeGroup &&
               mode == ConSanFlatProvenanceMode::Strict) {
      ++skipped_flat_counts.excluded_maybe_group;
    } else {
      count_skipped_moi_flat_candidate(site, skipped_flat_counts);
    }
  }
  warn_skipped_moi_flat_candidates(kernel.name, true, skipped_flat_counts, result);
}

void append_moi_candidates(const ConSanFunctionInfo &function, ConSanFlatProvenanceMode mode,
                           ConSanResult &result) {
  SkippedMoiLdsCounts skipped_lds_counts;
  for (const ConSanLdsSite &site : function.lds_sites) {
    if (is_moi_native_lds_candidate(site)) {
      result.moi_candidates.push_back(make_moi_candidate(function.name, false, std::nullopt, site));
    } else {
      append_moi_lds_exclusion(function.name, false, site, result);
      count_skipped_moi_lds_candidate(site, skipped_lds_counts);
    }
  }
  warn_skipped_moi_lds_candidates(function.name, false, skipped_lds_counts, result);
  SkippedMoiFlatCounts skipped_flat_counts;
  for (const ConSanFlatSite &site : function.flat_sites) {
    if (is_moi_flat_candidate(site, mode)) {
      result.moi_candidates.push_back(make_moi_candidate(function.name, false, std::nullopt, site));
    } else if (site.address_space_hint == ConSanFlatAddressSpaceHint::MaybeGroup &&
               mode == ConSanFlatProvenanceMode::Strict) {
      ++skipped_flat_counts.excluded_maybe_group;
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
  if (equals_any(mnemonic, {"ds_load_i8",         "ds_load_u8",          "ds_load_i16",
                            "ds_load_u16",        "ds_load_u8_d16",      "ds_load_u8_d16_hi",
                            "ds_load_i8_d16",     "ds_load_i8_d16_hi",   "ds_load_u16_d16",
                            "ds_load_u16_d16_hi", "ds_store_b8",         "ds_store_b16",
                            "ds_store_b8_d16_hi", "ds_store_b16_d16_hi", "ds_read_i8",
                            "ds_read_u8",         "ds_read_i16",         "ds_read_u16",
                            "ds_read_u8_d16",     "ds_read_u8_d16_hi",   "ds_read_i8_d16",
                            "ds_read_i8_d16_hi",  "ds_read_u16_d16",     "ds_read_u16_d16_hi",
                            "ds_write_b8",        "ds_write_b16",        "ds_write_b8_d16_hi",
                            "ds_write_b16_d16_hi"}))
    return true;

  return equals_any(mnemonic, {"ds_load_b32", "ds_load_b64", "ds_load_b96", "ds_load_b128",
                               "ds_read_b32", "ds_read_b64", "ds_read_b96", "ds_read_b128",
                               "ds_store_b32", "ds_store_b64", "ds_store_b96", "ds_store_b128",
                               "ds_write_b32", "ds_write_b64", "ds_write_b96", "ds_write_b128"});
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
  if (mnemonic == "ds_read2_b32" || mnemonic == "ds_write2_b32")
    return 4u;
  if (mnemonic == "ds_read2_b64" || mnemonic == "ds_write2_b64")
    return 8u;
  if (mnemonic == "ds_read2st64_b32" || mnemonic == "ds_write2st64_b32")
    return 256u;
  if (mnemonic == "ds_read2st64_b64" || mnemonic == "ds_write2st64_b64")
    return 512u;
  return std::nullopt;
}

[[nodiscard]] std::optional<std::vector<ConSanMoiAccessRange>>
candidate_access_ranges(const ConSanMoiCandidate &candidate) {
  const std::optional<uint32_t> byte_count = byte_count_for_candidate(candidate);
  if (!byte_count)
    return std::nullopt;

  if (candidate.source == ConSanMoiCandidateSource::NativeLds) {
    if (candidate.native_static_ranges.empty())
      return std::nullopt;
    std::vector<ConSanMoiAccessRange> ranges;
    ranges.reserve(candidate.native_static_ranges.size());
    for (const ConSanLdsStaticRange &range : candidate.native_static_ranges)
      ranges.push_back({range.byte_offset, range.byte_count});
    return ranges;
  }

  if (candidate.source == ConSanMoiCandidateSource::FlatGroup ||
      candidate.source == ConSanMoiCandidateSource::FlatMaybeGroup)
    return std::vector<ConSanMoiAccessRange>{{0u, *byte_count}};

  return std::nullopt;
}

[[nodiscard]] bool is_first_light_native_lds_candidate(const ConSanMoiCandidate &candidate,
                                                       std::span<const uint8_t>) {
  if (candidate.source != ConSanMoiCandidateSource::NativeLds)
    return false;
  return candidate_access_ranges(candidate).has_value();
}

[[nodiscard]] bool is_first_light_flat_candidate(const ConSanMoiCandidate &candidate) {
  if (candidate.source != ConSanMoiCandidateSource::FlatGroup &&
      candidate.source != ConSanMoiCandidateSource::FlatMaybeGroup)
    return false;
  const bool rdna4_encoding = candidate.size == 3u * sizeof(uint32_t);
  const bool cdna4_encoding =
      candidate.size == 2u * sizeof(uint32_t) && candidate.raw_segment == 0u;
  if (!rdna4_encoding && !cdna4_encoding)
    return false;
  if (!candidate.raw_ioffset || *candidate.raw_ioffset != 0)
    return false;
  if (!candidate.addr_vgpr || *candidate.addr_vgpr >= 255)
    return false;
  if (candidate.mnemonic != "flat_load_b32" && candidate.mnemonic != "flat_load_b64" &&
      candidate.mnemonic != "flat_load_b128" && candidate.mnemonic != "flat_store_b32" &&
      candidate.mnemonic != "flat_store_b64" && candidate.mnemonic != "flat_store_b128" &&
      candidate.mnemonic != "flat_load_dword" && candidate.mnemonic != "flat_load_dwordx2" &&
      candidate.mnemonic != "flat_load_dwordx4" && candidate.mnemonic != "flat_store_dword" &&
      candidate.mnemonic != "flat_store_dwordx2" && candidate.mnemonic != "flat_store_dwordx4")
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
  const auto wait = build_s_wait_flat_load0(arch);
  if (!mov_address_lo || !mov_address_hi || !load || !wait)
    return false;

  words.insert(words.end(), mov_address_lo->begin(), mov_address_lo->end());
  words.insert(words.end(), mov_address_hi->begin(), mov_address_hi->end());
  words.insert(words.end(), load->begin(), load->end());
  words.push_back(*wait);
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
  if (!source.scalar_src && !source.vector_src)
    return true;
  const uint16_t value_vgpr = static_cast<uint16_t>(scratch_vgpr + 2u);
  const uint16_t input =
      source.vector_src ? vector_source_vgpr(*source.vector_src) : *source.scalar_src;
  words.push_back(build_v_mov_b32_e32(value_vgpr, input, arch));
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
  const auto wait = build_s_wait_flat_load0(arch);
  if (!mov_address_lo || !mov_address_hi || !mov_one || !atomic_add || !wait)
    return false;

  words.insert(words.end(), mov_address_lo->begin(), mov_address_lo->end());
  words.insert(words.end(), mov_address_hi->begin(), mov_address_hi->end());
  words.insert(words.end(), mov_one->begin(), mov_one->end());
  words.insert(words.end(), atomic_add->begin(), atomic_add->end());
  words.push_back(*wait);
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
  const auto wait = build_s_wait_flat_load0(arch);
  if (!mov_address_lo || !mov_address_hi || !mov_one || !atomic_add || !wait)
    return false;

  words.insert(words.end(), mov_address_lo->begin(), mov_address_lo->end());
  words.insert(words.end(), mov_address_hi->begin(), mov_address_hi->end());
  words.insert(words.end(), mov_one->begin(), mov_one->end());
  words.insert(words.end(), atomic_add->begin(), atomic_add->end());
  words.push_back(*wait);
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
  const auto slot_times_40 = build_v_add_nc_u32_words(
      scaled_slot_vgpr, vector_source_vgpr(scaled_slot_vgpr), tmp_vgpr, arch);
  const auto address_with_slot = build_v_add_nc_u32_words(
      address_lo_vgpr, vector_source_vgpr(address_lo_vgpr), scaled_slot_vgpr, arch);
  if (!mov_address_lo || !mov_address_hi || !slot_times_32 || !slot_times_8 || !slot_times_40 ||
      !address_with_slot)
    return false;

  words.insert(words.end(), mov_address_lo->begin(), mov_address_lo->end());
  words.insert(words.end(), mov_address_hi->begin(), mov_address_hi->end());
  words.push_back(*slot_times_32);
  words.push_back(*slot_times_8);
  words.insert(words.end(), slot_times_40->begin(), slot_times_40->end());
  words.insert(words.end(), address_with_slot->begin(), address_with_slot->end());
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
  if (!source.scalar_src && !source.vector_src)
    return true;
  const uint16_t value_vgpr = static_cast<uint16_t>(scratch_vgpr + 5u);
  const uint16_t input =
      source.vector_src ? vector_source_vgpr(*source.vector_src) : *source.scalar_src;
  words.push_back(build_v_mov_b32_e32(value_vgpr, input, arch));
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
  const auto address_with_slot = build_v_add_nc_u32_words(
      address_lo_vgpr, vector_source_vgpr(address_lo_vgpr), scaled_slot_vgpr, arch);
  if (!mov_address_lo || !mov_address_hi || !slot_times_64 || !address_with_slot)
    return false;

  words.insert(words.end(), mov_address_lo->begin(), mov_address_lo->end());
  words.insert(words.end(), mov_address_hi->begin(), mov_address_hi->end());
  words.push_back(*slot_times_64);
  words.insert(words.end(), address_with_slot->begin(), address_with_slot->end());
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

[[nodiscard]] bool append_dynamic_diagnostic_record_address(std::vector<uint32_t> &words,
                                                            uint64_t field_address,
                                                            uint16_t slot_vgpr,
                                                            uint16_t scratch_vgpr,
                                                            rj_code_arch_t arch) {
  const uint16_t address_lo_vgpr = scratch_vgpr;
  const uint16_t address_hi_vgpr = static_cast<uint16_t>(scratch_vgpr + 1u);
  const uint16_t scaled_slot_vgpr = address_hi_vgpr;
  const auto slot_times_16 =
      build_v_lshlrev_b32_e32(scaled_slot_vgpr, scalar_positive_inline_u32(4), slot_vgpr, arch);
  const auto slot_times_64 =
      build_v_lshlrev_b32_e32(address_lo_vgpr, scalar_positive_inline_u32(6), slot_vgpr, arch);
  const auto slot_times_80 = build_v_add_nc_u32_words(
      scaled_slot_vgpr, vector_source_vgpr(address_lo_vgpr), scaled_slot_vgpr, arch);
  const auto mov_address_lo =
      build_v_mov_b32_e64_literal(address_lo_vgpr, static_cast<uint32_t>(field_address), arch);
  const auto mov_address_hi = build_v_mov_b32_e64_literal(
      address_hi_vgpr, static_cast<uint32_t>(field_address >> 32u), arch);
  const auto address_with_slot = build_v_add_nc_u32_words(
      address_lo_vgpr, vector_source_vgpr(address_lo_vgpr), scaled_slot_vgpr, arch);
  if (!slot_times_16 || !slot_times_64 || !slot_times_80 || !mov_address_lo || !mov_address_hi ||
      !address_with_slot)
    return false;
  words.push_back(*slot_times_16);
  words.push_back(*slot_times_64);
  words.insert(words.end(), slot_times_80->begin(), slot_times_80->end());
  words.insert(words.end(), mov_address_lo->begin(), mov_address_lo->end());
  words.insert(words.end(), address_with_slot->begin(), address_with_slot->end());
  words.insert(words.end(), mov_address_hi->begin(), mov_address_hi->end());
  return true;
}

[[nodiscard]] bool append_dynamic_diagnostic_store_u32_vgpr(std::vector<uint32_t> &words,
                                                            uint64_t field_address,
                                                            uint16_t value_vgpr, uint16_t slot_vgpr,
                                                            uint16_t scratch_vgpr,
                                                            rj_code_arch_t arch) {
  const auto store = build_flat_store_b32_vaddr_vsrc(scratch_vgpr, value_vgpr, arch);
  if (!store || !append_dynamic_diagnostic_record_address(words, field_address, slot_vgpr,
                                                          scratch_vgpr, arch))
    return false;
  words.insert(words.end(), store->begin(), store->end());
  return true;
}

[[nodiscard]] bool append_dynamic_diagnostic_store_u32_literal(std::vector<uint32_t> &words,
                                                               uint64_t field_address,
                                                               uint32_t value, uint16_t slot_vgpr,
                                                               uint16_t scratch_vgpr,
                                                               rj_code_arch_t arch) {
  const uint16_t value_vgpr = static_cast<uint16_t>(scratch_vgpr + 4u);
  if (!append_dynamic_diagnostic_record_address(words, field_address, slot_vgpr, scratch_vgpr,
                                                arch))
    return false;
  const auto mov_value = build_v_mov_b32_e64_literal(value_vgpr, value, arch);
  const auto store = build_flat_store_b32_vaddr_vsrc(scratch_vgpr, value_vgpr, arch);
  if (!mov_value || !store)
    return false;
  words.insert(words.end(), mov_value->begin(), mov_value->end());
  words.insert(words.end(), store->begin(), store->end());
  return true;
}

[[nodiscard]] bool
append_dynamic_diagnostic_store_u32_scalar_src(std::vector<uint32_t> &words, uint64_t field_address,
                                               uint16_t scalar_src, uint16_t slot_vgpr,
                                               uint16_t scratch_vgpr, rj_code_arch_t arch) {
  const uint16_t value_vgpr = static_cast<uint16_t>(scratch_vgpr + 4u);
  if (!append_dynamic_diagnostic_record_address(words, field_address, slot_vgpr, scratch_vgpr,
                                                arch))
    return false;
  words.push_back(build_v_mov_b32_e32(value_vgpr, scalar_src, arch));
  const auto store = build_flat_store_b32_vaddr_vsrc(scratch_vgpr, value_vgpr, arch);
  if (!store)
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
  if (!source.scalar_src && !source.vector_src)
    return true;
  const uint16_t value_vgpr = static_cast<uint16_t>(scratch_vgpr + 5u);
  const uint16_t input =
      source.vector_src ? vector_source_vgpr(*source.vector_src) : *source.scalar_src;
  words.push_back(build_v_mov_b32_e32(value_vgpr, input, arch));
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

[[nodiscard]] std::optional<DbiPatchPlacement>
plan_prebuilt_appended_cave(DbiPatchPlacementPlanner &planner, uint64_t anchor_offset,
                            uint32_t original_size, std::span<const uint32_t> cave_words,
                            std::vector<std::string> &errors, std::string_view probe_name) {
  if (cave_words.empty()) {
    errors.emplace_back(std::string(probe_name) + " produced an empty cave");
    return std::nullopt;
  }
  DbiPatchPlacementRequest request;
  request.anchor_offset = anchor_offset;
  request.original_size = original_size;
  request.body_size = static_cast<uint64_t>(cave_words.size() - 1u) * sizeof(uint32_t);
  request.inline_capacity = 0;
  std::string placement_error;
  const auto placement = planner.plan(request, &placement_error);
  if (!placement || placement->kind != DbiPatchPlacementKind::AppendedCave ||
      placement->return_branch_offset !=
          placement->body_offset + (cave_words.size() - 1u) * sizeof(uint32_t)) {
    errors.emplace_back(std::string(probe_name) + " placement failed: " + placement_error);
    return std::nullopt;
  }
  return placement;
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

[[nodiscard]] uint32_t moi_descriptor_wavefront_size(const KD &desc, rj_code_arch_t arch) {
  const bool wave32 = AMDHSA_BITS_GET(desc.kernel_code_properties,
                                      kd::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32);
  return consan_moi_descriptor_register_geometry(arch, wave32).wavefront_size;
}

[[nodiscard]] uint32_t moi_descriptor_user_sgpr_count(const KD &desc) {
  return AMDHSA_BITS_GET(desc.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT);
}

[[nodiscard]] bool moi_descriptor_enables_dispatch_ptr(const KD &desc) {
  return AMDHSA_BITS_GET(desc.kernel_code_properties,
                         kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR) != 0;
}

[[nodiscard]] uint16_t moi_descriptor_dispatch_ptr_sgpr(const KD &desc) {
  return AMDHSA_BITS_GET(desc.kernel_code_properties,
                         kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER)
             ? 4u
             : 0u;
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

[[nodiscard]] uint32_t moi_descriptor_abi_sgpr_count(const KD &desc, rj_code_arch_t arch) {
  uint32_t count = moi_descriptor_user_sgpr_count(desc);
  count += moi_descriptor_enables_workgroup_id(desc, 0) ? 1u : 0u;
  count += moi_descriptor_enables_workgroup_id(desc, 1) ? 1u : 0u;
  count += moi_descriptor_enables_workgroup_id(desc, 2) ? 1u : 0u;
  count += AMDHSA_BITS_GET(desc.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_INFO)
               ? 1u
               : 0u;
  count += AMDHSA_BITS_GET(desc.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT)
               ? 1u
               : 0u;
  return std::min<uint32_t>(count, moi_max_ordinary_sgprs(arch));
}

[[nodiscard]] uint32_t moi_descriptor_dispatch_insertion_width(const KD &desc) {
  return moi_descriptor_enables_dispatch_ptr(desc) ? 0u : 2u;
}

struct Cdna4IdentityAbiTransaction {
  uint16_t dispatch_sgpr = 0;
  uint32_t dispatch_insertion_width = 0;
  uint32_t workgroup_insertion_width = 0;
  uint32_t original_abi_count = 0;
  uint32_t patched_abi_count = 0;
  std::array<uint16_t, 3> workgroup_sources{};
  std::vector<std::pair<uint16_t, uint16_t>> guest_restore_moves;
};

[[nodiscard]] std::optional<Cdna4IdentityAbiTransaction>
build_cdna4_identity_abi_transaction(const KD &descriptor, std::vector<std::string> &errors) {
  Cdna4IdentityAbiTransaction transaction;
  transaction.dispatch_sgpr = moi_descriptor_dispatch_ptr_sgpr(descriptor);
  transaction.dispatch_insertion_width = moi_descriptor_dispatch_insertion_width(descriptor);
  transaction.original_abi_count =
      moi_descriptor_abi_sgpr_count(descriptor, ROCJITSU_CODE_ARCH_CDNA4);
  const uint32_t original_user_count = moi_descriptor_user_sgpr_count(descriptor);
  if (original_user_count < transaction.dispatch_sgpr) {
    errors.emplace_back("ConSan MOI CDNA4 identity transaction has an invalid user-SGPR ABI");
    return std::nullopt;
  }

  uint32_t original_workgroup_count = 0;
  for (uint32_t dimension = 0; dimension < 3; ++dimension)
    original_workgroup_count +=
        moi_descriptor_enables_workgroup_id(descriptor, dimension) ? 1u : 0u;
  transaction.workgroup_insertion_width = 3u - original_workgroup_count;
  const uint32_t patched_user_count = original_user_count + transaction.dispatch_insertion_width;
  transaction.patched_abi_count = transaction.original_abi_count +
                                  transaction.dispatch_insertion_width +
                                  transaction.workgroup_insertion_width;
  if (transaction.patched_abi_count > moi_max_ordinary_sgprs(ROCJITSU_CODE_ARCH_CDNA4)) {
    errors.emplace_back("ConSan MOI CDNA4 identity transaction exceeds SGPR bounds");
    return std::nullopt;
  }

  for (uint32_t dimension = 0; dimension < 3; ++dimension)
    transaction.workgroup_sources[dimension] =
        static_cast<uint16_t>(patched_user_count + dimension);

  for (uint32_t destination = transaction.dispatch_sgpr; destination < original_user_count;
       ++destination) {
    if (transaction.dispatch_insertion_width != 0) {
      transaction.guest_restore_moves.emplace_back(
          static_cast<uint16_t>(destination),
          static_cast<uint16_t>(destination + transaction.dispatch_insertion_width));
    }
  }

  uint32_t original_workgroup_rank = 0;
  for (uint32_t dimension = 0; dimension < 3; ++dimension) {
    if (!moi_descriptor_enables_workgroup_id(descriptor, dimension))
      continue;
    transaction.guest_restore_moves.emplace_back(
        static_cast<uint16_t>(original_user_count + original_workgroup_rank),
        transaction.workgroup_sources[dimension]);
    ++original_workgroup_rank;
  }
  uint32_t original_system_destination = original_user_count + original_workgroup_count;
  uint32_t patched_system_source = patched_user_count + 3u;
  if (AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2,
                      kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_INFO)) {
    transaction.guest_restore_moves.emplace_back(
        static_cast<uint16_t>(original_system_destination++),
        static_cast<uint16_t>(patched_system_source++));
  }
  if (AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT)) {
    transaction.guest_restore_moves.emplace_back(static_cast<uint16_t>(original_system_destination),
                                                 static_cast<uint16_t>(patched_system_source));
  }
  std::erase_if(transaction.guest_restore_moves,
                [](const auto &move) { return move.first == move.second; });
  std::ranges::sort(transaction.guest_restore_moves,
                    [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });
  return transaction;
}

void append_cdna4_identity_guest_abi_restore(std::vector<uint32_t> &words,
                                             const Cdna4IdentityAbiTransaction &transaction) {
  for (const auto &[destination, source] : transaction.guest_restore_moves)
    words.push_back(build_s_mov_b32(destination, source, ROCJITSU_CODE_ARCH_CDNA4));
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
            ? ConSanMoiWorkgroupSource{ttmp_scalar_operand(kTtmpRdna4GridX), std::nullopt, false,
                                       false}
            : ConSanMoiWorkgroupSource{},
        moi_descriptor_enables_workgroup_id(desc, 1)
            ? ConSanMoiWorkgroupSource{ttmp_scalar_operand(kTtmpRdna4GridYz), std::nullopt, false,
                                       true}
            : ConSanMoiWorkgroupSource{},
        moi_descriptor_enables_workgroup_id(desc, 2)
            ? ConSanMoiWorkgroupSource{ttmp_scalar_operand(kTtmpRdna4GridYz), std::nullopt, true,
                                       false}
            : ConSanMoiWorkgroupSource{},
    };
  }
  const auto sgpr_source = [](std::optional<uint16_t> sgpr) {
    return sgpr ? ConSanMoiWorkgroupSource{*sgpr, std::nullopt, false, false}
                : ConSanMoiWorkgroupSource{};
  };
  return ConSanMoiWorkgroupSources{
      sgpr_source(moi_descriptor_workgroup_id_sgpr(desc, 0)),
      sgpr_source(moi_descriptor_workgroup_id_sgpr(desc, 1)),
      sgpr_source(moi_descriptor_workgroup_id_sgpr(desc, 2)),
  };
}

[[nodiscard]] std::optional<ConSanMoiWorkgroupSources>
moi_probe_workgroup_sources(std::span<const uint8_t> image, uint64_t descriptor_file_offset,
                            const ConSanOptions &options, rj_code_arch_t arch,
                            std::vector<std::string> &errors) {
  if (arch == ROCJITSU_CODE_ARCH_CDNA4 && options.moi_workgroup_sgprs[0] &&
      options.moi_workgroup_sgprs[1] && options.moi_workgroup_sgprs[2]) {
    if (!options.moi_workgroup_sgprs[0] || !options.moi_workgroup_sgprs[1] ||
        !options.moi_workgroup_sgprs[2]) {
      errors.emplace_back("ConSan MOI CDNA4 stable workgroup identity has no persistent SGPRs");
      return std::nullopt;
    }
    return ConSanMoiWorkgroupSources{
        ConSanMoiWorkgroupSource{*options.moi_workgroup_sgprs[0], std::nullopt, false, false},
        ConSanMoiWorkgroupSource{*options.moi_workgroup_sgprs[1], std::nullopt, false, false},
        ConSanMoiWorkgroupSource{*options.moi_workgroup_sgprs[2], std::nullopt, false, false},
    };
  }
  if (arch == ROCJITSU_CODE_ARCH_CDNA4 && options.automatic_moi_identity_vgprs) {
    if (!options.moi_workgroup_vgprs[0] || !options.moi_workgroup_vgprs[1] ||
        !options.moi_workgroup_vgprs[2]) {
      errors.emplace_back("ConSan MOI CDNA4 stable workgroup identity has no persistent VGPRs");
      return std::nullopt;
    }
    return ConSanMoiWorkgroupSources{
        ConSanMoiWorkgroupSource{std::nullopt, *options.moi_workgroup_vgprs[0], false, false},
        ConSanMoiWorkgroupSource{std::nullopt, *options.moi_workgroup_vgprs[1], false, false},
        ConSanMoiWorkgroupSource{std::nullopt, *options.moi_workgroup_vgprs[2], false, false},
    };
  }
  return moi_descriptor_workgroup_sources(image, descriptor_file_offset, arch, errors);
}

[[nodiscard]] uint32_t moi_descriptor_vgpr_granularity(const KD &desc, rj_code_arch_t arch) {
  const bool wave32 = AMDHSA_BITS_GET(desc.kernel_code_properties,
                                      kd::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32);
  return consan_moi_descriptor_register_geometry(arch, wave32).vgpr_encoding_granularity;
}

[[nodiscard]] uint32_t moi_descriptor_vgpr_allocation_count(const KD &desc, rj_code_arch_t arch) {
  const uint32_t granularity = moi_descriptor_vgpr_granularity(desc, arch);
  const uint32_t granulated =
      AMDHSA_BITS_GET(desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
  return consan_moi_decode_descriptor_register_count(granulated, kMaxVgprs, granularity);
}

[[nodiscard]] uint32_t moi_descriptor_sgpr_allocation_count(const KD &desc, rj_code_arch_t arch) {
  const uint32_t granulated = AMDHSA_BITS_GET(
      desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT);
  return consan_moi_decode_descriptor_register_count(granulated, moi_max_ordinary_sgprs(arch), 8u);
}

enum class MoiOwnerInputKind : uint8_t {
  WorkitemId,
};

struct MoiOwnerInput {
  MoiOwnerInputKind kind = MoiOwnerInputKind::WorkitemId;
  uint16_t source = kWorkitemIdX;
  uint16_t shift_right = 0;
  uint32_t mask = consan_moi_exact_shadow::max_owner;
};

[[nodiscard]] bool same_owner_input(const MoiOwnerInput &lhs, const MoiOwnerInput &rhs) {
  return lhs.kind == rhs.kind && lhs.source == rhs.source && lhs.shift_right == rhs.shift_right &&
         lhs.mask == rhs.mask;
}

[[nodiscard]] std::optional<MoiOwnerInput>
moi_descriptor_owner_input(std::span<const uint8_t> image, uint64_t descriptor_file_offset,
                           rj_code_arch_t arch, std::vector<std::string> &errors) {
  if (descriptor_file_offset > image.size() || sizeof(KD) > image.size() - descriptor_file_offset) {
    errors.emplace_back("ConSan MOI owner derivation descriptor exceeds ELF bytes");
    return std::nullopt;
  }

  KD desc{};
  std::memcpy(&desc, image.data() + descriptor_file_offset, sizeof(desc));
  if (arch == ROCJITSU_CODE_ARCH_CDNA4) {
    // Standard gfx950 paths replace this site-local fallback with a persistent
    // entry snapshot. Keep a deterministic wave64 fallback only for
    // compatibility checks; runtime-owned TTMPs and debug HW_ID are not used.
    return MoiOwnerInput{MoiOwnerInputKind::WorkitemId, kWorkitemIdX,
                         /*shift_right=*/6u, consan_moi_exact_shadow::max_owner};
  }
  return MoiOwnerInput{
      MoiOwnerInputKind::WorkitemId, kWorkitemIdX,
      static_cast<uint16_t>(moi_descriptor_wavefront_size(desc, arch) == 32 ? 5 : 6),
      consan_moi_exact_shadow::max_owner};
}

[[nodiscard]] std::optional<MoiOwnerInput>
moi_kernel_owner_input(std::span<const uint8_t> image, const ConSanKernelInfo &kernel,
                       rj_code_arch_t arch, std::vector<std::string> &errors) {
  return moi_descriptor_owner_input(image, kernel.descriptor_file_offset, arch, errors);
}

[[nodiscard]] bool append_moi_owner_input(std::vector<uint32_t> &words, uint16_t destination_vgpr,
                                          const MoiOwnerInput &input, rj_code_arch_t arch,
                                          std::vector<std::string> &errors) {
  const uint16_t source_vgpr = input.source;
  const auto shift = build_v_lshrrev_b32_e32(
      destination_vgpr, scalar_positive_inline_u32(input.shift_right), source_vgpr, arch);
  if (!shift) {
    errors.emplace_back("ConSan MOI could not encode stable owner shift");
    return false;
  }
  words.push_back(*shift);
  if (input.mask != consan_moi_exact_shadow::max_owner) {
    const auto mask =
        build_v_and_b32_e32_literal(destination_vgpr, input.mask, destination_vgpr, arch);
    if (!mask) {
      errors.emplace_back("ConSan MOI could not encode stable owner mask");
      return false;
    }
    words.insert(words.end(), mask->begin(), mask->end());
  }
  return true;
}

[[nodiscard]] bool grow_moi_descriptor_vgpr_allocation(KD &desc, uint32_t required_count,
                                                       rj_code_arch_t arch) {
  if (required_count == 0 || required_count > kMaxVgprs)
    return false;
  if (required_count <= moi_descriptor_vgpr_allocation_count(desc, arch))
    return true;

  const uint32_t granularity = std::max<uint32_t>(moi_descriptor_vgpr_granularity(desc, arch), 1u);
  const auto granulated =
      consan_moi_encode_descriptor_register_count(required_count, kMaxVgprs, granularity);
  if (!granulated)
    return false;
  AMDHSA_BITS_SET(desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                  *granulated);
  return true;
}

[[nodiscard]] bool grow_moi_descriptor_sgpr_allocation(KD &desc, uint32_t required_count,
                                                       rj_code_arch_t arch) {
  const uint32_t max_sgprs = moi_max_ordinary_sgprs(arch);
  if (required_count == 0 || required_count > max_sgprs)
    return false;
  if (required_count <= moi_descriptor_sgpr_allocation_count(desc, arch))
    return true;

  constexpr uint32_t kSgprGranularity = 8;
  const auto granulated =
      consan_moi_encode_descriptor_register_count(required_count, max_sgprs, kSgprGranularity);
  if (!granulated)
    return false;
  AMDHSA_BITS_SET(desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  *granulated);
  return true;
}

[[nodiscard]] bool grow_moi_kernel_descriptor_vgprs(CodeObjectPatcher &patcher,
                                                    std::span<const uint8_t> image,
                                                    const ConSanKernelInfo &kernel,
                                                    uint32_t required_count, rj_code_arch_t arch,
                                                    std::vector<std::string> &errors) {
  if (kernel.descriptor_file_offset > image.size() ||
      sizeof(KD) > image.size() - kernel.descriptor_file_offset) {
    errors.emplace_back("ConSan MOI descriptor VGPR growth exceeds ELF bytes");
    return false;
  }

  KD desc{};
  std::memcpy(&desc, image.data() + kernel.descriptor_file_offset, sizeof(desc));
  if (!grow_moi_descriptor_vgpr_allocation(desc, required_count, arch)) {
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
                                                    ConSanResult &result, uint32_t required_count,
                                                    rj_code_arch_t arch) {
  for (const ConSanKernelInfo &kernel : result.kernels) {
    if (!grow_moi_kernel_descriptor_vgprs(patcher, image, kernel, required_count, arch,
                                          result.errors)) {
      return false;
    }
  }
  return true;
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

[[nodiscard]] uint16_t inline_shadow_scratch_count(const ConSanOptions &) { return 7u; }

[[nodiscard]] bool validate_inline_shadow_exec_save_sgpr(const ConSanOptions &options,
                                                         std::vector<std::string> &errors) {
  if (!options.moi_exec_save_sgpr)
    return true;
  constexpr uint16_t max_exec_save_sgpr = 94u;
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
  uint16_t ordinary_vgpr_limit = kMaxVgprs;
  uint16_t max_referenced_vgpr_count = 0;
  uint16_t current_sgpr_count = 0;
  uint16_t max_referenced_sgpr_count = 0;
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

[[nodiscard]] uint16_t max_referenced_sgpr_count(const KernelCfgScope &scope, rj_code_arch_t arch) {
  uint32_t max_count = 0;
  for (BasicBlock *block : scope.blocks) {
    if (block == nullptr)
      continue;
    for (const Instruction &inst : block->instructions()) {
      const InstDefUse du(inst);
      const RegisterSet referenced = du.defs | du.uses;
      referenced.for_each([&](RegisterRef ref) {
        if (ref.cls == RegClass::SGPR)
          max_count = std::max<uint32_t>(max_count, static_cast<uint32_t>(ref.index) + 1u);
      });
    }
  }
  return static_cast<uint16_t>(std::min<uint32_t>(max_count, moi_max_ordinary_sgprs(arch)));
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
    return options.moi_dynamic_access_records ? 6u : 3u;
  case ConSanMoiEngine::InlineShadow:
    return inline_shadow_scratch_count(options);
  case ConSanMoiEngine::Sampled:
    return options.moi_sampled_check ? 7u : 5u;
  }
  return 0;
}

struct MoiResourcePlanningState {
  std::span<const uint8_t> bytes;
  AmdGpuCodeObject code_object;
  std::unique_ptr<Decoder> decoder;
  std::vector<std::unique_ptr<BasicBlock>> blocks;
  BlockOffsetIndex block_index;
  std::vector<MoiKernelResourceContext> contexts;
  uint16_t max_sgpr_count = 0;
  bool valid = false;

  MoiResourcePlanningState(std::span<const uint8_t> input, rj_code_arch_t arch,
                           const ConSanResult &result)
      : bytes(input), code_object(input.data(), input.size()), decoder(Decoder::create(arch)),
        max_sgpr_count(static_cast<uint16_t>(moi_max_ordinary_sgprs(arch))) {
    if (!code_object.is_valid() || !decoder)
      return;

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

    std::vector<BasicBlock::CodeRange> code_ranges;
    code_ranges.reserve(code_object.functions().size());
    for (const AmdGpuFunctionInfo &function : code_object.functions()) {
      if (function.code_size != 0)
        code_ranges.push_back(
            {.start_offset = function.entry_text_offset, .size = function.code_size});
    }

    blocks = BasicBlock::build(code_object, *decoder, arch, leaders, code_ranges);
    block_index = build_block_offset_index(blocks);
    CodeObjectPatcher patcher(code_object);
    const std::span<const uint8_t> text = patcher.text_bytes();

    contexts.reserve(result.kernels.size());
    for (const ConSanKernelInfo &kernel : result.kernels) {
      if (!kernel.has_text_range)
        continue;
      auto scope =
          build_kernel_cfg_scope(blocks, block_index,
                                 KernelScopeRequest{.entry_offset = kernel.entry_text_offset,
                                                    .additional_entry_offsets = {}},
                                 kernel_entries, text);
      if (!scope)
        continue;

      MoiKernelResourceContext context;
      context.kernel = &kernel;
      context.scope = std::move(*scope);
      contexts.push_back(std::move(context));
      MoiKernelResourceContext &stored = contexts.back();
      stored.liveness = std::make_unique<LivenessAnalysis>(KernelBlockScope(stored.scope.blocks),
                                                           LivenessAnalysisOptions{},
                                                           stored.scope.liveness_edges);
      stored.max_referenced_vgpr_count = max_referenced_vgpr_count(stored.scope);
      stored.max_referenced_sgpr_count = max_referenced_sgpr_count(stored.scope, arch);

      if (kernel.descriptor_file_offset > bytes.size() ||
          sizeof(KD) > bytes.size() - kernel.descriptor_file_offset) {
        continue;
      }
      KD descriptor{};
      std::memcpy(&descriptor, bytes.data() + kernel.descriptor_file_offset, sizeof(descriptor));
      stored.max_referenced_sgpr_count = static_cast<uint16_t>(std::max<uint32_t>(
          stored.max_referenced_sgpr_count, moi_descriptor_abi_sgpr_count(descriptor, arch)));
      stored.current_vgpr_count =
          static_cast<uint16_t>(moi_descriptor_vgpr_allocation_count(descriptor, arch));
      if (arch == ROCJITSU_CODE_ARCH_CDNA4) {
        const uint32_t encoded_accum_offset = AMDHSA_BITS_GET(
            descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET);
        // The all-zero descriptor encoding is not actionable without separate
        // proof that the kernel has accumulators. A nonzero ACCUM_OFFSET,
        // however, is a hard upper bound on ordinary temporary VGPRs even when
        // the descriptor's granulated allocation extends beyond it.
        if (encoded_accum_offset != 0) {
          const uint32_t accum_vgpr_base = (encoded_accum_offset + 1u) * 4u;
          stored.ordinary_vgpr_limit =
              static_cast<uint16_t>(std::min<uint32_t>(kMaxVgprs, accum_vgpr_base));
        }
      }
      stored.current_sgpr_count =
          static_cast<uint16_t>(moi_descriptor_sgpr_allocation_count(descriptor, arch));
      stored.original_private_segment_size = descriptor.private_segment_fixed_size;
      stored.descriptor_valid = true;
    }
    valid = true;
  }
};

[[nodiscard]] ConSanCandidateResourcePlan
plan_moi_resource_site(MoiResourcePlanningState &state, const ConSanOptions &options,
                       ConSanResourceSiteKind site_kind, size_t candidate_index,
                       uint64_t text_offset, std::optional<uint64_t> kernel_descriptor_file_offset,
                       uint16_t scratch_count) {
  ConSanCandidateResourcePlan plan;
  plan.site_kind = site_kind;
  plan.candidate_index = candidate_index;
  plan.text_offset = text_offset;
  plan.scratch_vgpr_count = scratch_count;
  if (!state.valid) {
    plan.reason = ConSanRegisterPlanReason::MissingInstruction;
    return plan;
  }

  BasicBlock *anchor_block = block_for_offset(state.block_index, text_offset);
  const Instruction *anchor = instruction_at(anchor_block, text_offset);
  if (anchor == nullptr) {
    plan.reason = ConSanRegisterPlanReason::MissingInstruction;
    return plan;
  }

  std::vector<MoiKernelResourceContext *> owners;
  for (MoiKernelResourceContext &context : state.contexts) {
    if (kernel_descriptor_file_offset &&
        context.kernel->descriptor_file_offset != *kernel_descriptor_file_offset) {
      continue;
    }
    if (std::ranges::find(context.scope.blocks, anchor_block) == context.scope.blocks.end())
      continue;
    owners.push_back(&context);
    plan.owner_descriptor_file_offsets.push_back(context.kernel->descriptor_file_offset);
  }
  if (owners.empty()) {
    plan.reason = ConSanRegisterPlanReason::MissingOwner;
    return plan;
  }

  plan.current_vgpr_count = kMaxVgprs;
  plan.current_sgpr_count = state.max_sgpr_count;
  uint16_t ordinary_vgpr_limit = kMaxVgprs;
  RegisterSet live_before_all_owners;
  bool descriptors_valid = true;
  for (MoiKernelResourceContext *owner : owners) {
    plan.current_vgpr_count = std::min(plan.current_vgpr_count, owner->current_vgpr_count);
    ordinary_vgpr_limit = std::min(ordinary_vgpr_limit, owner->ordinary_vgpr_limit);
    plan.max_referenced_vgpr_count =
        std::max(plan.max_referenced_vgpr_count, owner->max_referenced_vgpr_count);
    plan.current_sgpr_count = std::min(plan.current_sgpr_count, owner->current_sgpr_count);
    plan.max_referenced_sgpr_count =
        std::max(plan.max_referenced_sgpr_count, owner->max_referenced_sgpr_count);
    plan.original_private_segment_size =
        std::max(plan.original_private_segment_size, owner->original_private_segment_size);
    live_before_all_owners |= owner->liveness->live_before(*anchor);
    descriptors_valid &= owner->descriptor_valid;
  }
  plan.required_vgpr_count = plan.current_vgpr_count;
  if (!descriptors_valid) {
    plan.reason = ConSanRegisterPlanReason::InvalidDescriptor;
    return plan;
  }

  const InstDefUse anchor_def_use(*anchor);
  ConSanRegisterRequest request;
  request.reg_class = RegClass::VGPR;
  request.count = scratch_count;
  request.current_allocation_count = std::min(plan.current_vgpr_count, ordinary_vgpr_limit);
  request.max_referenced_count = plan.max_referenced_vgpr_count;
  request.architecture_limit = ordinary_vgpr_limit;
  request.explicit_base = options.scratch_vgpr;
  request.force_spill = options.force_vgpr_spill;
  request.forbidden = anchor_def_use.defs | anchor_def_use.uses;
  if (!options.moi_owner_vgpr && options.moi_owner_source == ConSanMoiOwnerSource::WorkitemId)
    request.forbidden.expand({RegClass::VGPR, kWorkitemIdX, 1});
  if (options.moi_owner_vgpr)
    request.forbidden.expand({RegClass::VGPR, *options.moi_owner_vgpr, 1});
  if (options.moi_epoch_vgpr)
    request.forbidden.expand({RegClass::VGPR, *options.moi_epoch_vgpr, 1});
  for (const auto workgroup_vgpr : options.moi_workgroup_vgprs) {
    if (workgroup_vgpr)
      request.forbidden.expand({RegClass::VGPR, *workgroup_vgpr, 1});
  }

  const ConSanRegisterPlan register_plan = plan_consan_registers(request, live_before_all_owners);
  plan.source = register_plan.source;
  plan.reason = register_plan.reason;
  plan.scratch_vgpr = register_plan.base;
  plan.required_vgpr_count =
      std::max(plan.current_vgpr_count, register_plan.required_descriptor_count);
  return plan;
}

void append_moi_resource_plans(std::span<const uint8_t> bytes, const ConSanOptions &options,
                               rj_code_arch_t arch, ConSanResult &result) {
  result.resource_plans.clear();
  result.resource_plans.reserve(result.moi_candidates.size());
  MoiResourcePlanningState state(bytes, arch, result);
  const uint16_t scratch_count = moi_access_scratch_vgpr_count(options);
  for (size_t candidate_index = 0; candidate_index < result.moi_candidates.size();
       ++candidate_index) {
    const ConSanMoiCandidate &candidate = result.moi_candidates[candidate_index];
    result.resource_plans.push_back(plan_moi_resource_site(
        state, options, ConSanResourceSiteKind::Access, candidate_index, candidate.text_offset,
        candidate.kernel_descriptor_file_offset, scratch_count));
  }
}

struct ResolvedMoiScratchPlan {
  uint16_t base = 0;
  uint16_t count = 0;
  uint16_t required_vgpr_count = 0;
  uint16_t max_referenced_sgpr_count = 0;
  uint32_t original_private_segment_size = 0;
  std::vector<uint64_t> owner_descriptor_file_offsets;
  ConSanRegisterAllocationSource source = ConSanRegisterAllocationSource::Unsupported;
};

[[nodiscard]] const ConSanCandidateResourcePlan *
resource_plan_for_candidate(const ConSanResult &result, const ConSanMoiCandidate &candidate) {
  if (result.moi_candidates.empty())
    return nullptr;
  const auto it = std::ranges::find_if(result.moi_candidates,
                                       [&](const auto &item) { return &item == &candidate; });
  if (it == result.moi_candidates.end())
    return nullptr;
  const size_t index = static_cast<size_t>(std::distance(result.moi_candidates.begin(), it));
  if (index >= result.resource_plans.size() ||
      result.resource_plans[index].candidate_index != index) {
    return nullptr;
  }
  return &result.resource_plans[index];
}

[[nodiscard]] uint32_t moi_resource_preference(const ConSanResult &result,
                                               const ConSanMoiCandidate &candidate) {
  const ConSanCandidateResourcePlan *plan = resource_plan_for_candidate(result, candidate);
  if (plan == nullptr)
    return 4u;
  switch (plan->source) {
  case ConSanRegisterAllocationSource::Explicit:
  case ConSanRegisterAllocationSource::LivenessDead:
    return 0u;
  case ConSanRegisterAllocationSource::DescriptorGrowth:
    return 1u;
  case ConSanRegisterAllocationSource::SpillRequired:
    return 2u;
  case ConSanRegisterAllocationSource::Unsupported:
    return 3u;
  }
  return 4u;
}

void prefer_spill_free_scalar_candidates(std::vector<const ConSanMoiCandidate *> &candidates,
                                         const ConSanResult &result, const ConSanOptions &options) {
  if (!options.automatic_moi_scalar_identity)
    return;
  std::stable_sort(candidates.begin(), candidates.end(), [&](const auto *lhs, const auto *rhs) {
    return moi_resource_preference(result, *lhs) < moi_resource_preference(result, *rhs);
  });
}

[[nodiscard]] const ConSanCandidateResourcePlan *
resource_plan_for_site(const ConSanResult &result, ConSanResourceSiteKind site_kind,
                       uint64_t text_offset) {
  const auto it = std::ranges::find_if(result.resource_plans, [&](const auto &plan) {
    return plan.site_kind == site_kind && plan.text_offset == text_offset;
  });
  return it == result.resource_plans.end() ? nullptr : &*it;
}

[[nodiscard]] std::optional<ResolvedMoiScratchPlan>
resolve_moi_scratch_plan(const ConSanCandidateResourcePlan &plan, const ConSanOptions &options,
                         uint16_t expected_count) {
  if (plan.scratch_vgpr && plan.scratch_vgpr_count == expected_count &&
      (plan.source == ConSanRegisterAllocationSource::Explicit ||
       plan.source == ConSanRegisterAllocationSource::LivenessDead ||
       plan.source == ConSanRegisterAllocationSource::DescriptorGrowth ||
       plan.source == ConSanRegisterAllocationSource::SpillRequired)) {
    ResolvedMoiScratchPlan resolved;
    resolved.base = *plan.scratch_vgpr;
    resolved.count = expected_count;
    resolved.required_vgpr_count = plan.required_vgpr_count;
    resolved.max_referenced_sgpr_count = plan.max_referenced_sgpr_count;
    resolved.original_private_segment_size = plan.original_private_segment_size;
    resolved.owner_descriptor_file_offsets = plan.owner_descriptor_file_offsets;
    resolved.source = plan.source;
    if (options.moi_owner_vgpr) {
      resolved.required_vgpr_count = std::max<uint16_t>(
          resolved.required_vgpr_count, static_cast<uint16_t>(*options.moi_owner_vgpr + 1u));
    }
    if (options.moi_epoch_vgpr) {
      resolved.required_vgpr_count = std::max<uint16_t>(
          resolved.required_vgpr_count, static_cast<uint16_t>(*options.moi_epoch_vgpr + 1u));
    }
    for (const auto workgroup_vgpr : options.moi_workgroup_vgprs) {
      if (workgroup_vgpr) {
        resolved.required_vgpr_count = std::max<uint16_t>(
            resolved.required_vgpr_count, static_cast<uint16_t>(*workgroup_vgpr + 1u));
      }
    }
    return resolved;
  }

  return std::nullopt;
}

[[nodiscard]] std::optional<ResolvedMoiScratchPlan>
resolve_moi_scratch(const ConSanResult &result, const ConSanMoiCandidate &candidate,
                    const ConSanOptions &options, uint16_t expected_count) {
  const ConSanCandidateResourcePlan *plan = resource_plan_for_candidate(result, candidate);
  return plan == nullptr ? std::nullopt : resolve_moi_scratch_plan(*plan, options, expected_count);
}

void apply_test_kernel_filter(std::vector<const ConSanMoiCandidate *> &candidates,
                              const ConSanOptions &options) {
  if (options.test_kernel_name_filter.empty())
    return;
  std::erase_if(candidates, [&](const ConSanMoiCandidate *candidate) {
    return candidate == nullptr ||
           candidate->container_name.find(options.test_kernel_name_filter) == std::string::npos;
  });
}

[[nodiscard]] uint16_t moi_exec_save_sgpr_count(const ConSanOptions &options) {
  if (options.moi_engine == ConSanMoiEngine::Sampled)
    return options.moi_runtime_sample_stride > 1 || options.moi_sampled_check ? 2u : 0u;
  if (options.moi_engine == ConSanMoiEngine::InlineShadow) {
    if (!options.moi_report_buffer_address)
      return 0;
    const ConSanMoiReportBufferLayout layout =
        consan_moi_inline_shadow_report_buffer_layout_for_bytes(options.moi_report_buffer_size);
    // Four nested EXEC-save pairs, one VCC-save pair, and one SCC snapshot.
    return layout.diagnostic_capacity != 0 || options.moi_track_atomics ||
                   options.moi_track_barriers
               ? 11u
               : 0u;
  }
  // One EXEC-save pair, one VCC-save pair, and one SCC snapshot.
  return options.moi_dynamic_access_records || options.moi_track_barriers ||
                 options.moi_track_atomics
             ? 5u
             : 0u;
}

struct MoiSpecialStateSgprs {
  uint16_t vcc_save_sgpr = 0;
  uint16_t scc_save_sgpr = 0;
};

[[nodiscard]] std::optional<MoiSpecialStateSgprs>
moi_special_state_sgprs(const ConSanOptions &options) {
  if (!options.moi_exec_save_sgpr)
    return std::nullopt;
  const uint16_t vcc_offset = options.moi_engine == ConSanMoiEngine::InlineShadow ? 8u : 2u;
  return MoiSpecialStateSgprs{
      .vcc_save_sgpr = static_cast<uint16_t>(*options.moi_exec_save_sgpr + vcc_offset),
      .scc_save_sgpr = static_cast<uint16_t>(*options.moi_exec_save_sgpr + vcc_offset + 2u),
  };
}

[[nodiscard]] bool append_save_moi_special_state(std::vector<uint32_t> &words,
                                                 const ConSanOptions &options,
                                                 rj_code_arch_t arch) {
  const auto registers = moi_special_state_sgprs(options);
  if (!registers)
    return false;
  const auto save_scc = build_s_cselect_b32(registers->scc_save_sgpr, scalar_positive_inline_u32(1),
                                            scalar_positive_inline_u32(0), arch);
  const auto save_vcc = build_s_mov_b64(registers->vcc_save_sgpr, kWave64VccLo, arch);
  if (!save_scc || !save_vcc)
    return false;
  // Snapshot SCC before any instrumentation instruction can modify it.
  words.push_back(*save_scc);
  words.push_back(*save_vcc);
  return true;
}

[[nodiscard]] bool append_restore_moi_special_state(std::vector<uint32_t> &words,
                                                    const ConSanOptions &options,
                                                    rj_code_arch_t arch) {
  const auto registers = moi_special_state_sgprs(options);
  if (!registers)
    return false;
  const auto restore_vcc = build_s_mov_b64(kWave64VccLo, registers->vcc_save_sgpr, arch);
  const auto restore_scc =
      build_s_cmp_lg_u32(registers->scc_save_sgpr, scalar_positive_inline_u32(0), arch);
  if (!restore_vcc || !restore_scc)
    return false;
  words.push_back(*restore_vcc);
  // Keep this last: every scalar comparison or saveexec before it may write SCC.
  words.push_back(*restore_scc);
  return true;
}

[[nodiscard]] bool configure_automatic_moi_exec_save_sgprs(ConSanOptions &options,
                                                           std::span<const uint8_t> image,
                                                           rj_code_arch_t arch,
                                                           ConSanResult &result) {
  if (options.moi_exec_save_sgpr)
    return false;
  const uint16_t count = moi_exec_save_sgpr_count(options);
  if (count == 0)
    return false;

  uint32_t first = 0;
  bool found_owned_scope = false;
  for (const ConSanCandidateResourcePlan &plan : result.resource_plans) {
    if (plan.owner_descriptor_file_offsets.empty())
      continue;
    found_owned_scope = true;
    first = std::max<uint32_t>(first, plan.max_referenced_sgpr_count);
    if (arch == ROCJITSU_CODE_ARCH_CDNA4 &&
        options.moi_owner_source == ConSanMoiOwnerSource::WorkitemId) {
      for (uint64_t descriptor_offset : plan.owner_descriptor_file_offsets) {
        if (descriptor_offset > image.size() || sizeof(KD) > image.size() - descriptor_offset)
          continue;
        KD descriptor{};
        std::memcpy(&descriptor, image.data() + descriptor_offset, sizeof(descriptor));
        std::vector<std::string> transaction_errors;
        const auto transaction =
            build_cdna4_identity_abi_transaction(descriptor, transaction_errors);
        if (!transaction) {
          result.errors.insert(result.errors.end(), transaction_errors.begin(),
                               transaction_errors.end());
          return false;
        }
        first = std::max(first, transaction->patched_abi_count);
      }
    }
  }
  if (!found_owned_scope) {
    result.warnings.emplace_back(
        "ConSan MOI could not derive automatic EXEC-save SGPRs without an owned candidate "
        "scope");
    return false;
  }

  first = util::align_up(first, 2u);
  for (uint32_t base = first; base + count <= moi_max_ordinary_sgprs(arch); base += 2u) {
    if (options.moi_owner_sgpr &&
        range_overlaps(static_cast<uint16_t>(base), count, *options.moi_owner_sgpr, 1u)) {
      continue;
    }
    options.moi_exec_save_sgpr = static_cast<uint16_t>(base);
    options.automatic_moi_exec_save_sgprs = true;
    result.resolved_moi_exec_save_sgpr = options.moi_exec_save_sgpr;
    result.moi_exec_save_sgprs_automatic = true;
    result.warnings.emplace_back("ConSan MOI automatically assigned EXEC-save SGPRs s" +
                                 std::to_string(base) + ":s" + std::to_string(base + count - 1u));
    return true;
  }

  result.warnings.emplace_back(
      "ConSan MOI could not place a fresh automatic EXEC-save SGPR window");
  return false;
}

[[nodiscard]] bool configure_automatic_moi_owner_sgpr(ConSanOptions &options, rj_code_arch_t arch,
                                                      ConSanResult &result) {
  if (options.moi_owner_source != ConSanMoiOwnerSource::HwId || options.moi_owner_sgpr)
    return false;

  uint32_t first = 0;
  bool found_owned_scope = false;
  for (const ConSanCandidateResourcePlan &plan : result.resource_plans) {
    if (plan.owner_descriptor_file_offsets.empty())
      continue;
    found_owned_scope = true;
    first = std::max<uint32_t>(first, plan.max_referenced_sgpr_count);
  }
  if (!found_owned_scope) {
    result.warnings.emplace_back(
        "ConSan MOI could not derive an automatic hw_id owner SGPR without an owned candidate "
        "scope");
    return false;
  }

  for (uint32_t sgpr = first; sgpr < moi_max_ordinary_sgprs(arch); ++sgpr) {
    if (options.moi_exec_save_sgpr &&
        range_overlaps(static_cast<uint16_t>(sgpr), 1u, *options.moi_exec_save_sgpr,
                       moi_exec_save_sgpr_count(options))) {
      continue;
    }
    options.moi_owner_sgpr = static_cast<uint16_t>(sgpr);
    options.automatic_moi_owner_sgpr = true;
    result.resolved_moi_owner_sgpr = options.moi_owner_sgpr;
    result.moi_owner_sgpr_automatic = true;
    result.warnings.emplace_back("ConSan MOI automatically assigned hw_id owner temporary s" +
                                 std::to_string(sgpr));
    return true;
  }

  result.warnings.emplace_back("ConSan MOI could not place a fresh automatic hw_id owner SGPR");
  return false;
}

[[nodiscard]] bool configure_automatic_moi_identity_sgprs(ConSanOptions &options,
                                                          std::span<const uint8_t> image,
                                                          rj_code_arch_t arch,
                                                          ConSanResult &result) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA4 ||
      options.moi_owner_source != ConSanMoiOwnerSource::WorkitemId ||
      !options.moi_report_buffer_address || options.moi_identity_sgpr)
    return false;

  const uint16_t identity_count = options.force_private_epoch ? 5u : 2u;
  uint32_t first = 0;
  uint32_t fresh_first = 0;
  bool found_owned_scope = false;
  for (const ConSanCandidateResourcePlan &plan : result.resource_plans) {
    if (plan.owner_descriptor_file_offsets.empty())
      continue;
    found_owned_scope = true;
    first = std::max<uint32_t>(first, plan.max_referenced_sgpr_count + 2u);
    for (uint64_t descriptor_offset : plan.owner_descriptor_file_offsets) {
      if (descriptor_offset > image.size() || sizeof(KD) > image.size() - descriptor_offset)
        continue;
      KD descriptor{};
      std::memcpy(&descriptor, image.data() + descriptor_offset, sizeof(descriptor));
      std::vector<std::string> transaction_errors;
      const auto transaction = build_cdna4_identity_abi_transaction(descriptor, transaction_errors);
      if (!transaction) {
        result.errors.insert(result.errors.end(), transaction_errors.begin(),
                             transaction_errors.end());
        return false;
      }
      first = std::max(first, transaction->patched_abi_count);
      fresh_first = std::max<uint32_t>(
          fresh_first, std::max<uint32_t>(transaction->patched_abi_count,
                                          moi_descriptor_sgpr_allocation_count(descriptor, arch)));
    }
  }
  if (!found_owned_scope) {
    result.warnings.emplace_back(
        "ConSan MOI could not derive automatic CDNA4 identity SGPRs without an owned scope");
    return false;
  }

  const auto find_window = [&](uint32_t start) -> std::optional<uint32_t> {
    for (uint32_t base = start; base + identity_count <= moi_max_ordinary_sgprs(arch); ++base) {
      if (options.moi_exec_save_sgpr &&
          range_overlaps(static_cast<uint16_t>(base), identity_count, *options.moi_exec_save_sgpr,
                         moi_exec_save_sgpr_count(options)))
        continue;
      if (options.moi_owner_sgpr &&
          range_overlaps(static_cast<uint16_t>(base), identity_count, *options.moi_owner_sgpr, 1u))
        continue;
      return base;
    }
    return std::nullopt;
  };
  std::optional<uint32_t> selected = find_window(options.force_private_epoch ? fresh_first : first);
  if (selected) {
    const uint32_t base = *selected;
    options.moi_identity_sgpr = static_cast<uint16_t>(base);
    result.resolved_moi_identity_sgpr = options.moi_identity_sgpr;
    if (options.force_private_epoch) {
      for (uint32_t dimension = 0; dimension < 3; ++dimension) {
        options.moi_workgroup_sgprs[dimension] = static_cast<uint16_t>(base + 2u + dimension);
      }
    }
    result.warnings.emplace_back("ConSan MOI automatically assigned CDNA4 identity SGPRs s" +
                                 std::to_string(base) + ":s" +
                                 std::to_string(base + identity_count - 1u));
    return true;
  }

  result.errors.emplace_back(options.force_private_epoch
                                 ? "ConSan MOI could not place a fresh five-SGPR private "
                                   "workgroup identity window above the guest allocation"
                                 : "ConSan MOI could not place fresh CDNA4 identity SGPRs");
  return false;
}

[[nodiscard]] bool configure_automatic_moi_persistent_vgprs(ConSanOptions &options,
                                                            std::span<const uint8_t> image,
                                                            rj_code_arch_t arch,
                                                            ConSanResult &result) {
  if (options.moi_owner_vgpr || options.moi_epoch_vgpr)
    return false;
  const bool needs_persistent_state =
      options.moi_init_owner_epoch || options.moi_track_atomics ||
      options.moi_engine == ConSanMoiEngine::InlineShadow ||
      (arch == ROCJITSU_CODE_ARCH_CDNA4 &&
       options.moi_owner_source == ConSanMoiOwnerSource::WorkitemId);
  if (!needs_persistent_state)
    return false;
  if (!options.moi_report_buffer_address)
    return false;

  uint32_t persistent_base = 0;
  bool found_owned_scope = false;
  for (const ConSanCandidateResourcePlan &plan : result.resource_plans) {
    if (plan.owner_descriptor_file_offsets.empty())
      continue;
    found_owned_scope = true;
    persistent_base = std::max<uint32_t>(persistent_base, plan.max_referenced_vgpr_count);
    if (plan.scratch_vgpr) {
      persistent_base = std::max<uint32_t>(
          persistent_base, static_cast<uint32_t>(*plan.scratch_vgpr) + plan.scratch_vgpr_count);
    }
  }

  if (!found_owned_scope) {
    result.warnings.emplace_back(
        "ConSan MOI could not derive automatic persistent VGPRs without an owned candidate "
        "scope");
    return false;
  }
  if (options.force_private_epoch) {
    options.moi_init_owner_epoch = true;
    options.automatic_moi_private_epoch = true;
    result.moi_private_epoch_automatic = true;
    result.warnings.emplace_back("ConSan MOI automatically selected private owner/epoch state");
    return true;
  }
  bool crosses_cdna4_accum_offset = false;
  if (arch == ROCJITSU_CODE_ARCH_CDNA4) {
    for (const ConSanCandidateResourcePlan &plan : result.resource_plans) {
      for (uint64_t descriptor_offset : plan.owner_descriptor_file_offsets) {
        if (descriptor_offset > image.size() || sizeof(KD) > image.size() - descriptor_offset) {
          // An inaccessible descriptor cannot prove that five new persistent
          // VGPRs stay below ACCUM_OFFSET. Prefer private state conservatively.
          crosses_cdna4_accum_offset = true;
          continue;
        }
        KD descriptor{};
        std::memcpy(&descriptor, image.data() + descriptor_offset, sizeof(descriptor));
        const uint32_t encoded_accum_offset = AMDHSA_BITS_GET(
            descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET);
        if (encoded_accum_offset == 0)
          continue;
        const uint32_t accum_vgpr_base = (encoded_accum_offset + 1u) * 4u;
        crosses_cdna4_accum_offset |= persistent_base + 5u > accum_vgpr_base;
      }
    }
  }
  const bool needs_cdna4_scalar_fallback =
      crosses_cdna4_accum_offset ||
      (arch == ROCJITSU_CODE_ARCH_CDNA4 && persistent_base + 5u > kMaxVgprs);
  if (needs_cdna4_scalar_fallback && options.moi_owner_source != ConSanMoiOwnerSource::WorkitemId) {
    options.moi_init_owner_epoch = true;
    options.automatic_moi_private_epoch = true;
    result.moi_private_epoch_automatic = true;
    result.warnings.emplace_back(
        "ConSan MOI selected private owner/epoch state to preserve non-workitem owner semantics");
    return true;
  }
  if (needs_cdna4_scalar_fallback) {
    options.moi_init_owner_epoch = true;
    uint32_t referenced_scalar_base = 0;
    uint32_t fresh_scalar_base = 0;
    for (const ConSanCandidateResourcePlan &plan : result.resource_plans) {
      if (!plan.owner_descriptor_file_offsets.empty()) {
        referenced_scalar_base =
            std::max<uint32_t>(referenced_scalar_base, plan.max_referenced_sgpr_count + 2u);
        // Persistent state lives for the whole kernel, not merely around the
        // selected patch site. Keep it above every owner's original scalar
        // allocation instead of trusting instruction-level reference scans
        // to prove that descriptor padding is globally dead. In particular,
        // CDNA4 instructions that are only partially decoded must not make an
        // apparently unused in-allocation SGPR window look safe.
        for (uint64_t descriptor_offset : plan.owner_descriptor_file_offsets) {
          if (descriptor_offset > image.size() || sizeof(KD) > image.size() - descriptor_offset)
            continue;
          KD descriptor{};
          std::memcpy(&descriptor, image.data() + descriptor_offset, sizeof(descriptor));
          fresh_scalar_base = std::max<uint32_t>(
              fresh_scalar_base, moi_descriptor_sgpr_allocation_count(descriptor, arch));
          std::vector<std::string> transaction_errors;
          const auto transaction =
              build_cdna4_identity_abi_transaction(descriptor, transaction_errors);
          if (!transaction) {
            result.errors.insert(result.errors.end(), transaction_errors.begin(),
                                 transaction_errors.end());
            return false;
          }
          referenced_scalar_base = std::max(referenced_scalar_base, transaction->patched_abi_count);
          fresh_scalar_base = std::max(fresh_scalar_base, transaction->patched_abi_count);
        }
      }
    }
    constexpr uint16_t kScalarIdentityCount = 5u;
    const auto find_window = [&](uint32_t first) -> std::optional<uint32_t> {
      for (uint32_t base = first; base + kScalarIdentityCount <= moi_max_ordinary_sgprs(arch);
           ++base) {
        const bool overlaps_exec =
            options.moi_exec_save_sgpr &&
            range_overlaps(static_cast<uint16_t>(base), kScalarIdentityCount,
                           *options.moi_exec_save_sgpr, moi_exec_save_sgpr_count(options));
        const bool overlaps_owner =
            options.moi_owner_sgpr &&
            range_overlaps(static_cast<uint16_t>(base), kScalarIdentityCount,
                           *options.moi_owner_sgpr, 1u);
        if (!overlaps_exec && !overlaps_owner)
          return base;
      }
      return std::nullopt;
    };
    std::optional<uint32_t> scalar_base = find_window(fresh_scalar_base);
    if (!scalar_base) {
      scalar_base = find_window(referenced_scalar_base);
      if (scalar_base) {
        result.warnings.emplace_back(
            "ConSan MOI selected a globally unreferenced in-allocation five-SGPR identity window");
      }
    }
    if (!scalar_base) {
      result.errors.emplace_back(
          "ConSan MOI CDNA4 accumulator fallback has no five-SGPR identity window");
      return false;
    }
    options.moi_identity_sgpr = static_cast<uint16_t>(*scalar_base);
    result.resolved_moi_identity_sgpr = options.moi_identity_sgpr;
    options.automatic_moi_scalar_identity = true;
    result.moi_scalar_identity_automatic = true;
    options.moi_state_owner_sgpr = static_cast<uint16_t>(*scalar_base);
    options.moi_state_epoch_sgpr = static_cast<uint16_t>(*scalar_base + 1u);
    for (uint32_t dimension = 0; dimension < 3; ++dimension) {
      options.moi_workgroup_sgprs[dimension] = static_cast<uint16_t>(*scalar_base + 2u + dimension);
    }
    result.warnings.emplace_back(
        "ConSan MOI automatically selected five-SGPR CDNA4 identity state");
    return true;
  }
  const uint32_t persistent_count = arch == ROCJITSU_CODE_ARCH_CDNA4 ? 5u : 2u;
  if (persistent_base + persistent_count > kMaxVgprs) {
    if (arch == ROCJITSU_CODE_ARCH_CDNA4 &&
        options.moi_owner_source == ConSanMoiOwnerSource::WorkitemId) {
      result.errors.emplace_back(
          "ConSan MOI CDNA4 stable owner has no supported persistent representation");
      return false;
    }
    options.moi_init_owner_epoch = true;
    options.automatic_moi_private_epoch = true;
    result.moi_private_epoch_automatic = true;
    result.warnings.emplace_back("ConSan MOI automatically selected private owner/epoch state");
    return true;
  }

  options.moi_owner_vgpr = static_cast<uint16_t>(persistent_base);
  options.moi_epoch_vgpr = static_cast<uint16_t>(persistent_base + 1u);
  if (arch == ROCJITSU_CODE_ARCH_CDNA4) {
    for (uint32_t dimension = 0; dimension < 3; ++dimension) {
      options.moi_workgroup_vgprs[dimension] =
          static_cast<uint16_t>(persistent_base + 2u + dimension);
      result.resolved_moi_workgroup_vgprs[dimension] = options.moi_workgroup_vgprs[dimension];
    }
    options.automatic_moi_identity_vgprs = true;
  }
  options.moi_init_owner_epoch = true;
  options.automatic_moi_persistent_vgprs = true;
  result.resolved_moi_owner_vgpr = options.moi_owner_vgpr;
  result.resolved_moi_epoch_vgpr = options.moi_epoch_vgpr;
  result.moi_persistent_vgprs_automatic = true;
  result.warnings.emplace_back(
      "ConSan MOI automatically assigned persistent identity VGPRs v" +
      std::to_string(*options.moi_owner_vgpr) + ":v" +
      std::to_string(static_cast<uint32_t>(*options.moi_owner_vgpr) + persistent_count - 1u));
  return true;
}

[[nodiscard]] bool automatic_moi_persistent_state(const ConSanOptions &options) {
  return options.automatic_moi_persistent_vgprs || options.automatic_moi_private_epoch ||
         options.automatic_moi_scalar_identity;
}

using MoiDescriptorVgprRequirements = std::unordered_map<uint64_t, uint16_t>;
using MoiDescriptorSgprRequirements = std::unordered_map<uint64_t, uint16_t>;
using MoiDescriptorPrivateRequirements = std::unordered_map<uint64_t, uint32_t>;
using MoiSpillManagers = std::unordered_map<uint64_t, SpillManager>;

void note_descriptor_requirements(MoiDescriptorVgprRequirements &requirements,
                                  const ResolvedMoiScratchPlan &plan) {
  for (uint64_t descriptor_offset : plan.owner_descriptor_file_offsets) {
    auto [it, inserted] = requirements.emplace(descriptor_offset, plan.required_vgpr_count);
    if (!inserted)
      it->second = std::max(it->second, plan.required_vgpr_count);
  }
}

void note_moi_sgpr_requirements(MoiDescriptorSgprRequirements &requirements,
                                const ResolvedMoiScratchPlan &plan, const ConSanOptions &options) {
  uint16_t required_count = 0;
  if (options.moi_exec_save_sgpr) {
    const uint16_t count = moi_exec_save_sgpr_count(options);
    required_count = static_cast<uint16_t>(*options.moi_exec_save_sgpr + count);
  }
  if (options.moi_owner_source == ConSanMoiOwnerSource::HwId && options.moi_owner_sgpr) {
    required_count =
        std::max<uint16_t>(required_count, static_cast<uint16_t>(*options.moi_owner_sgpr + 1u));
  }
  if (options.moi_workgroup_sgprs[2]) {
    required_count = std::max<uint16_t>(
        required_count, static_cast<uint16_t>(*options.moi_workgroup_sgprs[2] + 1u));
  }
  if (required_count == 0)
    return;
  for (uint64_t descriptor_offset : plan.owner_descriptor_file_offsets) {
    auto [it, inserted] = requirements.emplace(descriptor_offset, required_count);
    if (!inserted)
      it->second = std::max(it->second, required_count);
  }
}

void note_spill_descriptor_requirements(MoiDescriptorPrivateRequirements &requirements,
                                        const ResolvedMoiScratchPlan &plan,
                                        const VgprSpillSequence &spill) {
  for (uint64_t descriptor_offset : plan.owner_descriptor_file_offsets) {
    auto [it, inserted] = requirements.emplace(descriptor_offset, spill.total_private_bytes);
    if (!inserted)
      it->second = std::max(it->second, spill.total_private_bytes);
  }
}

[[nodiscard]] const ConSanKernelInfo *kernel_for_descriptor(const ConSanResult &result,
                                                            uint64_t descriptor_offset) {
  const auto it = std::ranges::find_if(result.kernels, [descriptor_offset](const auto &kernel) {
    return kernel.descriptor_file_offset == descriptor_offset;
  });
  return it == result.kernels.end() ? nullptr : &*it;
}

struct MoiPrivateEpochLayout {
  uint32_t epoch_offset = 0;
  uint32_t owner_offset = 0;
  uint32_t ephemeral_base = 0;
};

[[nodiscard]] bool same_workgroup_source(const ConSanMoiWorkgroupSource &lhs,
                                         const ConSanMoiWorkgroupSource &rhs) {
  return lhs.scalar_src == rhs.scalar_src && lhs.vector_src == rhs.vector_src &&
         lhs.shift_right_16 == rhs.shift_right_16 && lhs.mask_low_16 == rhs.mask_low_16;
}

[[nodiscard]] bool same_workgroup_sources(const ConSanMoiWorkgroupSources &lhs,
                                          const ConSanMoiWorkgroupSources &rhs) {
  return same_workgroup_source(lhs.x, rhs.x) && same_workgroup_source(lhs.y, rhs.y) &&
         same_workgroup_source(lhs.z, rhs.z);
}

[[nodiscard]] std::optional<uint64_t> common_moi_record_owner_descriptor(
    std::span<const uint8_t> image, const ResolvedMoiScratchPlan &resources,
    const ConSanOptions &options, rj_code_arch_t arch, std::vector<std::string> &warnings) {
  if (arch == ROCJITSU_CODE_ARCH_CDNA4 &&
      (options.automatic_moi_identity_vgprs || options.automatic_moi_scalar_identity ||
       options.moi_workgroup_sgprs[2])) {
    if (resources.owner_descriptor_file_offsets.empty()) {
      warnings.emplace_back("ConSan MOI shared record owner has no owning descriptor");
      return std::nullopt;
    }
    return resources.owner_descriptor_file_offsets.front();
  }
  std::optional<MoiOwnerInput> common_owner_input;
  std::optional<ConSanMoiWorkgroupSources> common_workgroup_sources;
  std::optional<uint64_t> representative;
  for (uint64_t descriptor_offset : resources.owner_descriptor_file_offsets) {
    std::vector<std::string> errors;
    const auto owner_input = moi_descriptor_owner_input(image, descriptor_offset, arch, errors);
    const auto workgroup_sources =
        moi_descriptor_workgroup_sources(image, descriptor_offset, arch, errors);
    if (!owner_input || !workgroup_sources) {
      warnings.insert(warnings.end(), errors.begin(), errors.end());
      return std::nullopt;
    }
    if ((common_owner_input && !same_owner_input(*common_owner_input, *owner_input)) ||
        (common_workgroup_sources &&
         !same_workgroup_sources(*common_workgroup_sources, *workgroup_sources))) {
      warnings.emplace_back(
          "ConSan MOI shared record owner has incompatible descriptor ABI inputs");
      return std::nullopt;
    }
    common_owner_input = owner_input;
    common_workgroup_sources = workgroup_sources;
    if (!representative)
      representative = descriptor_offset;
  }
  if (!representative)
    warnings.emplace_back("ConSan MOI shared record owner has no owning descriptor");
  return representative;
}

[[nodiscard]] std::optional<MoiPrivateEpochLayout>
build_moi_private_epoch_layout(const ConSanResult &result, const ResolvedMoiScratchPlan &resources,
                               std::vector<std::string> &warnings) {
  if (resources.owner_descriptor_file_offsets.empty()) {
    warnings.emplace_back("ConSan MOI private epoch requires an owning kernel descriptor");
    return std::nullopt;
  }
  for (uint64_t descriptor_offset : resources.owner_descriptor_file_offsets) {
    const ConSanKernelInfo *kernel = kernel_for_descriptor(result, descriptor_offset);
    if (kernel == nullptr) {
      warnings.emplace_back("ConSan MOI private epoch references an unknown kernel descriptor");
      return std::nullopt;
    }
    if (kernel->uses_dynamic_stack.value_or(false)) {
      warnings.emplace_back(
          "ConSan MOI private epoch does not support a dynamic-stack owning kernel");
      return std::nullopt;
    }
  }
  const uint64_t epoch_offset =
      util::align_up(resources.original_private_segment_size, SpillManager::kDbiZoneAlignment);
  const uint64_t owner_offset = epoch_offset + SpillManager::kSlotBytes;
  const uint64_t ephemeral_base =
      util::align_up(owner_offset + SpillManager::kSlotBytes,
                     static_cast<uint64_t>(SpillManager::kDbiZoneAlignment));
  if (ephemeral_base > kMaxAddressFreeScratchPrivateBytes) {
    warnings.emplace_back("ConSan MOI private epoch exceeds address-free scratch capacity");
    return std::nullopt;
  }
  return MoiPrivateEpochLayout{.epoch_offset = static_cast<uint32_t>(epoch_offset),
                               .owner_offset = static_cast<uint32_t>(owner_offset),
                               .ephemeral_base = static_cast<uint32_t>(ephemeral_base)};
}

[[nodiscard]] std::optional<VgprSpillSequence>
build_moi_spill_sequence(const ConSanResult &result, const ResolvedMoiScratchPlan &resources,
                         MoiSpillManagers &managers, rj_code_arch_t arch,
                         std::vector<std::string> &warnings,
                         std::optional<uint32_t> private_layout_base = std::nullopt) {
  if (resources.source != ConSanRegisterAllocationSource::SpillRequired)
    return std::nullopt;
  if (resources.owner_descriptor_file_offsets.empty()) {
    warnings.emplace_back("ConSan MOI spill requires an owning kernel descriptor");
    return std::nullopt;
  }

  for (uint64_t descriptor_offset : resources.owner_descriptor_file_offsets) {
    const ConSanKernelInfo *kernel = kernel_for_descriptor(result, descriptor_offset);
    if (kernel == nullptr) {
      warnings.emplace_back("ConSan MOI spill references an unknown kernel descriptor");
      return std::nullopt;
    }
    if (kernel->uses_dynamic_stack.value_or(false)) {
      warnings.emplace_back("ConSan MOI spill does not support a dynamic-stack owning kernel");
      return std::nullopt;
    }
  }

  if (resources.owner_descriptor_file_offsets.size() > 1) {
    // Shared text needs one immediate layout. Starting above the maximum
    // original private size makes the same save/fill sequence legal for every
    // owner; site-local spill frames may safely reuse these offsets.
    SpillManager shared_manager(resources.original_private_segment_size,
                                kMaxAddressFreeScratchPrivateBytes);
    auto spill = build_vgpr_spill_sequence(shared_manager, resources.base, resources.count, arch);
    if (!spill)
      warnings.emplace_back("ConSan MOI could not encode the shared-owner VGPR spill window");
    return spill;
  }

  const uint64_t descriptor_offset = resources.owner_descriptor_file_offsets.front();
  auto [it, inserted] = managers.try_emplace(
      descriptor_offset, private_layout_base.value_or(resources.original_private_segment_size),
      kMaxAddressFreeScratchPrivateBytes);
  (void)inserted;
  auto spill = build_vgpr_spill_sequence(it->second, resources.base, resources.count, arch);
  if (!spill)
    warnings.emplace_back("ConSan MOI could not encode or reserve the planned VGPR spill window");
  return spill;
}

[[nodiscard]] bool
apply_descriptor_requirements(CodeObjectPatcher &patcher, const AmdGpuCodeObject &code_object,
                              std::span<const uint8_t> image, const ConSanResult &result,
                              const MoiDescriptorVgprRequirements &requirements,
                              rj_code_arch_t arch, std::vector<std::string> &errors) {
  for (const auto &[descriptor_offset, required_count] : requirements) {
    const ConSanKernelInfo *kernel = kernel_for_descriptor(result, descriptor_offset);
    if (kernel == nullptr) {
      errors.emplace_back("ConSan MOI resource plan references an unknown kernel descriptor");
      return false;
    }
    const auto active_kernel =
        std::ranges::find_if(code_object.kernels(), [&](const AmdGpuKernelInfo &candidate) {
          return candidate.name == kernel->name;
        });
    if (active_kernel == code_object.kernels().end()) {
      errors.emplace_back("ConSan MOI resource plan could not resolve an active descriptor");
      return false;
    }
    ConSanKernelInfo active = *kernel;
    active.descriptor_file_offset = active_kernel->descriptor_file_offset;
    if (!grow_moi_kernel_descriptor_vgprs(patcher, image, active, required_count, arch, errors))
      return false;
  }
  return true;
}

[[nodiscard]] bool
apply_spill_descriptor_requirements(CodeObjectPatcher &patcher, const AmdGpuCodeObject &code_object,
                                    std::span<const uint8_t> image, const ConSanResult &result,
                                    const MoiDescriptorPrivateRequirements &requirements,
                                    std::vector<std::string> &errors) {
  (void)image;
  for (const auto &[descriptor_offset, required_private_bytes] : requirements) {
    const ConSanKernelInfo *kernel = kernel_for_descriptor(result, descriptor_offset);
    const auto active_kernel =
        kernel == nullptr
            ? code_object.kernels().end()
            : std::ranges::find_if(code_object.kernels(), [&](const AmdGpuKernelInfo &candidate) {
                return candidate.name == kernel->name;
              });
    const std::span<const uint8_t> current_image = patcher.image_bytes();
    if (kernel == nullptr || active_kernel == code_object.kernels().end() ||
        active_kernel->descriptor_file_offset > current_image.size() ||
        sizeof(KD) > current_image.size() - active_kernel->descriptor_file_offset) {
      errors.emplace_back("ConSan MOI spill descriptor exceeds ELF bytes or has no kernel owner");
      return false;
    }

    std::vector<uint8_t> descriptor(sizeof(KD));
    std::memcpy(descriptor.data(), current_image.data() + active_kernel->descriptor_file_offset,
                sizeof(KD));
    const SpillDescriptorUpdate update = update_kernel_descriptor_for_spills(
        descriptor, /*descriptor_file_offset=*/0, required_private_bytes,
        kernel->uses_dynamic_stack.value_or(false));
    if (update != SpillDescriptorUpdate::Updated && update != SpillDescriptorUpdate::Unchanged) {
      errors.emplace_back("ConSan MOI could not grow the owning kernel's private spill segment");
      return false;
    }
    if (!patcher.patch_kernel_descriptor(active_kernel->descriptor_file_offset, descriptor)) {
      errors.emplace_back("ConSan MOI could not patch the owning kernel spill descriptor");
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool
apply_sgpr_descriptor_requirements(CodeObjectPatcher &patcher, const ConSanResult &result,
                                   const AmdGpuCodeObject &code_object,
                                   const MoiDescriptorSgprRequirements &requirements,
                                   rj_code_arch_t arch, std::vector<std::string> &errors) {
  for (const auto &[descriptor_offset, required_count] : requirements) {
    const ConSanKernelInfo *kernel = kernel_for_descriptor(result, descriptor_offset);
    const auto active_kernel =
        kernel == nullptr
            ? code_object.kernels().end()
            : std::ranges::find_if(code_object.kernels(), [&](const AmdGpuKernelInfo &candidate) {
                return candidate.name == kernel->name;
              });
    if (active_kernel == code_object.kernels().end()) {
      errors.emplace_back("ConSan MOI scalar plan could not resolve an active descriptor");
      return false;
    }
    const uint64_t active_descriptor_offset = active_kernel->descriptor_file_offset;
    const std::span<const uint8_t> current_image = patcher.image_bytes();
    if (active_descriptor_offset > current_image.size() ||
        sizeof(KD) > current_image.size() - active_descriptor_offset) {
      errors.emplace_back("ConSan MOI scalar-plan descriptor exceeds ELF bytes");
      return false;
    }
    KD descriptor{};
    std::memcpy(&descriptor, current_image.data() + active_descriptor_offset, sizeof(descriptor));
    if (!grow_moi_descriptor_sgpr_allocation(descriptor, required_count, arch)) {
      errors.emplace_back("ConSan MOI could not satisfy planned descriptor SGPR allocation");
      return false;
    }
    if (!patcher.patch_kernel_descriptor(
            active_descriptor_offset,
            std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(&descriptor),
                                     sizeof(descriptor)))) {
      errors.emplace_back("ConSan MOI could not patch the planned descriptor SGPR allocation");
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool
apply_spill_metadata_requirements(std::vector<uint8_t> &image, const ConSanResult &result,
                                  const MoiDescriptorPrivateRequirements &requirements,
                                  std::vector<std::string> &errors) {
  for (const auto &[descriptor_offset, required_private_bytes] : requirements) {
    const ConSanKernelInfo *kernel = kernel_for_descriptor(result, descriptor_offset);
    if (kernel == nullptr) {
      errors.emplace_back("ConSan MOI spill metadata references an unknown kernel descriptor");
      return false;
    }
    const SpillMetadataUpdate update =
        update_amdgpu_metadata_for_spills(image, kernel->name, required_private_bytes);
    switch (update) {
    case SpillMetadataUpdate::Updated:
    case SpillMetadataUpdate::Unchanged:
    case SpillMetadataUpdate::NoMetadata:
      break;
    case SpillMetadataUpdate::KernelNotFound:
      errors.emplace_back("ConSan MOI spill could not find the owning kernel in AMDGPU metadata");
      return false;
    case SpillMetadataUpdate::InvalidMetadata:
      errors.emplace_back("ConSan MOI spill found malformed AMDGPU metadata");
      return false;
    case SpillMetadataUpdate::UnencodableGrowth:
      errors.emplace_back(
          "ConSan MOI spill private growth crosses the in-place metadata integer width");
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool apply_descriptor_requirements(std::vector<uint8_t> &image,
                                                 const ConSanResult &result,
                                                 const MoiDescriptorVgprRequirements &requirements,
                                                 rj_code_arch_t arch,
                                                 std::vector<std::string> &errors) {
  for (const auto &[descriptor_offset, required_count] : requirements) {
    if (descriptor_offset > image.size() || sizeof(KD) > image.size() - descriptor_offset) {
      errors.emplace_back("ConSan MOI resource-plan descriptor exceeds ELF bytes");
      return false;
    }
    if (kernel_for_descriptor(result, descriptor_offset) == nullptr) {
      errors.emplace_back("ConSan MOI resource plan references an unknown kernel descriptor");
      return false;
    }
    KD descriptor{};
    std::memcpy(&descriptor, image.data() + descriptor_offset, sizeof(descriptor));
    if (!grow_moi_descriptor_vgpr_allocation(descriptor, required_count, arch)) {
      errors.emplace_back("ConSan MOI could not satisfy planned descriptor VGPR allocation");
      return false;
    }
    std::memcpy(image.data() + descriptor_offset, &descriptor, sizeof(descriptor));
  }
  return true;
}

[[nodiscard]] bool
apply_sgpr_descriptor_requirements(std::vector<uint8_t> &image, const ConSanResult &result,
                                   const MoiDescriptorSgprRequirements &requirements,
                                   rj_code_arch_t arch, std::vector<std::string> &errors) {
  for (const auto &[descriptor_offset, required_count] : requirements) {
    if (descriptor_offset > image.size() || sizeof(KD) > image.size() - descriptor_offset) {
      errors.emplace_back("ConSan MOI scalar-plan descriptor exceeds ELF bytes");
      return false;
    }
    if (kernel_for_descriptor(result, descriptor_offset) == nullptr) {
      errors.emplace_back("ConSan MOI scalar plan references an unknown kernel descriptor");
      return false;
    }
    KD descriptor{};
    std::memcpy(&descriptor, image.data() + descriptor_offset, sizeof(descriptor));
    if (!grow_moi_descriptor_sgpr_allocation(descriptor, required_count, arch)) {
      errors.emplace_back("ConSan MOI could not satisfy planned descriptor SGPR allocation");
      return false;
    }
    std::memcpy(image.data() + descriptor_offset, &descriptor, sizeof(descriptor));
  }
  return true;
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
      build_v_add_nc_u32_words(dst_vgpr, vector_source_vgpr(addr_vgpr), dst_vgpr, arch);
  if (!mov_offset || !add_offset)
    return false;

  words.insert(words.end(), mov_offset->begin(), mov_offset->end());
  words.insert(words.end(), add_offset->begin(), add_offset->end());
  return true;
}

[[nodiscard]] std::optional<std::vector<uint32_t>>
build_moi_private_load_b32(uint16_t vdst, uint32_t byte_offset, rj_code_arch_t arch);

[[nodiscard]] bool append_moi_private_load_wait(std::vector<uint32_t> &words, uint16_t vdst,
                                                uint32_t byte_offset, rj_code_arch_t arch) {
  const auto load = build_moi_private_load_b32(vdst, byte_offset, arch);
  const auto wait = arch == ROCJITSU_CODE_ARCH_CDNA4 ? build_cdna4_s_wait_vmcnt0(arch)
                                                     : build_s_wait_loadcnt0(arch);
  if (!load || !wait)
    return false;
  words.insert(words.end(), load->begin(), load->end());
  words.push_back(*wait);
  return true;
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_first_light_access_record_words(
    std::span<const uint8_t> bytes, const ConSanMoiCandidate &candidate,
    const ConSanOptions &options, rj_code_arch_t arch, uint32_t record_index, uint32_t record_count,
    uint32_t access_record_capacity, std::optional<uint32_t> private_owner_offset,
    std::optional<uint32_t> private_epoch_offset, std::vector<std::string> &errors) {
  if (!options.scratch_vgpr) {
    errors.emplace_back("ConSan MOI first-light probe requires RJ_CONSAN_TMP_VGPR");
    return std::nullopt;
  }
  if (options.automatic_moi_private_epoch && (!private_owner_offset || !private_epoch_offset)) {
    errors.emplace_back("ConSan MOI first-light probe has incomplete private identity layout");
    return std::nullopt;
  }
  const uint16_t scratch_count = options.moi_dynamic_access_records ? 6u : 3u;
  if (static_cast<uint32_t>(*options.scratch_vgpr) + scratch_count > 256u) {
    errors.emplace_back(options.moi_dynamic_access_records
                            ? "ConSan MOI dynamic access-record probe needs six scratch VGPRs"
                            : "ConSan MOI first-light probe needs three scratch VGPRs");
    return std::nullopt;
  }
  if (options.moi_dynamic_access_records && !options.moi_exec_save_sgpr) {
    errors.emplace_back(
        "ConSan MOI dynamic access-record probe requires RJ_CONSAN_MOI_EXEC_SAVE_SGPR");
    return std::nullopt;
  }
  if (options.moi_dynamic_access_records &&
      (*options.moi_exec_save_sgpr > 100u || *options.moi_exec_save_sgpr % 2u != 0u)) {
    errors.emplace_back(
        "ConSan MOI dynamic access-record probe requires an even RJ_CONSAN_MOI_EXEC_SAVE_SGPR in "
        "0..100");
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
      candidate_access_ranges(candidate);
  if (!access_ranges || access_ranges->empty()) {
    errors.emplace_back("ConSan MOI first-light probe requires a supported LDS access range");
    return std::nullopt;
  }
  std::optional<uint16_t> derived_owner_vgpr;
  std::vector<uint32_t> derived_owner_words;
  if (!options.moi_owner_vgpr && !options.moi_state_owner_sgpr &&
      !options.automatic_moi_private_epoch && candidate.kernel_descriptor_file_offset) {
    const auto owner_input =
        moi_descriptor_owner_input(bytes, *candidate.kernel_descriptor_file_offset, arch, errors);
    if (!owner_input)
      return std::nullopt;
    const uint16_t value_vgpr = static_cast<uint16_t>(
        *options.scratch_vgpr + (options.moi_dynamic_access_records ? 4u : 2u));
    if (!append_moi_owner_input(derived_owner_words, value_vgpr, *owner_input, arch, errors)) {
      errors.emplace_back("ConSan MOI first-light probe could not encode stable owner derivation");
      return std::nullopt;
    }
    derived_owner_vgpr = value_vgpr;
  }
  ConSanMoiWorkgroupSources workgroup_sources;
  if (options.automatic_moi_scalar_identity || candidate.kernel_descriptor_file_offset) {
    const auto descriptor_workgroup_sources = moi_probe_workgroup_sources(
        bytes, candidate.kernel_descriptor_file_offset.value_or(0), options, arch, errors);
    if (!descriptor_workgroup_sources)
      return std::nullopt;
    workgroup_sources = *descriptor_workgroup_sources;
  }

  std::vector<uint32_t> words;
  words.reserve(candidate.size / sizeof(uint32_t) + 1u + 7u * 12u + 9u + 10u + 20u + 24u +
                (options.moi_owner_vgpr || derived_owner_vgpr ? 9u : 0u) +
                (options.moi_epoch_vgpr ? 9u : 0u) + derived_owner_words.size());
  words.insert(words.end(), derived_owner_words.begin(), derived_owner_words.end());
  for (uint64_t offset = 0; offset < candidate.size; offset += sizeof(uint32_t)) {
    uint32_t word = 0;
    std::memcpy(&word, bytes.data() + candidate.file_offset + offset, sizeof(word));
    words.push_back(word);
  }
  const auto lds_wait = build_s_wait_lds0(arch);
  if (!lds_wait) {
    errors.emplace_back("ConSan MOI first-light probe could not encode LDS completion wait");
    return std::nullopt;
  }
  words.push_back(*lds_wait);

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
    const uint64_t dynamic_record_base = base + sizeof(ConSanMoiReportHeader);

    for (const ConSanMoiAccessRange &range : *access_ranges) {
      if (!append_save_moi_special_state(words, options, arch)) {
        errors.emplace_back("ConSan MOI dynamic access-record probe could not save VCC/SCC");
        return std::nullopt;
      }

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
          build_s_and_saveexec_b64(*options.moi_exec_save_sgpr, kWave64VccLo, arch);
      if (!mov_capacity || !slot_in_capacity || !narrow_exec) {
        errors.emplace_back(
            "ConSan MOI dynamic access-record probe could not encode capacity guard");
        return std::nullopt;
      }
      words.insert(words.end(), mov_capacity->begin(), mov_capacity->end());
      words.push_back(*slot_in_capacity);
      words.push_back(*narrow_exec);

      if (private_owner_offset &&
          (!append_moi_private_load_wait(words, value_vgpr, *private_owner_offset, arch) ||
           !append_dynamic_access_store_u32_vgpr(
               words, dynamic_record_base + offsetof(ConSanMoiAccessRecord, wave_id), value_vgpr,
               slot_vgpr, *options.scratch_vgpr, arch))) {
        errors.emplace_back("ConSan MOI dynamic access-record probe could not load private owner");
        return std::nullopt;
      }
      if (private_epoch_offset &&
          (!append_moi_private_load_wait(words, value_vgpr, *private_epoch_offset, arch) ||
           !append_dynamic_access_store_u32_vgpr(
               words, dynamic_record_base + offsetof(ConSanMoiAccessRecord, epoch), value_vgpr,
               slot_vgpr, *options.scratch_vgpr, arch))) {
        errors.emplace_back("ConSan MOI dynamic access-record probe could not load private epoch");
        return std::nullopt;
      }

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
          (options.moi_state_owner_sgpr &&
           !append_dynamic_access_store_u32_scalar_src(
               words, dynamic_record_base + offsetof(ConSanMoiAccessRecord, wave_id),
               *options.moi_state_owner_sgpr, slot_vgpr, *options.scratch_vgpr, arch)) ||
          (options.moi_epoch_vgpr &&
           !append_dynamic_access_store_u32_vgpr(
               words, dynamic_record_base + offsetof(ConSanMoiAccessRecord, epoch),
               *options.moi_epoch_vgpr, slot_vgpr, *options.scratch_vgpr, arch)) ||
          (options.moi_state_epoch_sgpr &&
           !append_dynamic_access_store_u32_scalar_src(
               words, dynamic_record_base + offsetof(ConSanMoiAccessRecord, epoch),
               *options.moi_state_epoch_sgpr, slot_vgpr, *options.scratch_vgpr, arch)) ||
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
      const auto restore_exec = build_s_mov_b64(kWave64ExecLo, *options.moi_exec_save_sgpr, arch);
      if (!restore_exec) {
        errors.emplace_back("ConSan MOI dynamic access-record probe could not restore EXEC");
        return std::nullopt;
      }
      words.push_back(*restore_exec);
      if (!append_restore_moi_special_state(words, options, arch)) {
        errors.emplace_back("ConSan MOI dynamic access-record probe could not restore VCC/SCC");
        return std::nullopt;
      }
    }
    return words;
  }

  for (size_t range_index = 0; range_index < access_ranges->size(); ++range_index) {
    const ConSanMoiAccessRange &range = (*access_ranges)[range_index];
    const uint64_t access_record_base =
        base + sizeof(ConSanMoiReportHeader) +
        (static_cast<uint64_t>(record_index) + range_index) * sizeof(ConSanMoiAccessRecord);
    const uint16_t private_state_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 2u);
    if (private_owner_offset &&
        (!append_moi_private_load_wait(words, private_state_vgpr, *private_owner_offset, arch) ||
         !append_store_u32_vgpr(words,
                                access_record_base + offsetof(ConSanMoiAccessRecord, wave_id),
                                private_state_vgpr, *options.scratch_vgpr, arch))) {
      errors.emplace_back("ConSan MOI first-light probe could not load private owner");
      return std::nullopt;
    }
    if (private_epoch_offset &&
        (!append_moi_private_load_wait(words, private_state_vgpr, *private_epoch_offset, arch) ||
         !append_store_u32_vgpr(words, access_record_base + offsetof(ConSanMoiAccessRecord, epoch),
                                private_state_vgpr, *options.scratch_vgpr, arch))) {
      errors.emplace_back("ConSan MOI first-light probe could not load private epoch");
      return std::nullopt;
    }
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
            words, access_record_base + offsetof(ConSanMoiAccessRecord, lane_mask), kWave64ExecLo,
            *options.scratch_vgpr, arch) ||
        !append_store_u32_scalar_src(
            words,
            access_record_base + offsetof(ConSanMoiAccessRecord, lane_mask) + sizeof(uint32_t),
            kWave64ExecHi, *options.scratch_vgpr, arch) ||
        (options.moi_owner_vgpr &&
         !append_store_u32_vgpr(words,
                                access_record_base + offsetof(ConSanMoiAccessRecord, wave_id),
                                *options.moi_owner_vgpr, *options.scratch_vgpr, arch)) ||
        (options.moi_state_owner_sgpr &&
         !append_store_u32_scalar_src(
             words, access_record_base + offsetof(ConSanMoiAccessRecord, wave_id),
             *options.moi_state_owner_sgpr, *options.scratch_vgpr, arch)) ||
        (options.moi_epoch_vgpr &&
         !append_store_u32_vgpr(words, access_record_base + offsetof(ConSanMoiAccessRecord, epoch),
                                *options.moi_epoch_vgpr, *options.scratch_vgpr, arch)) ||
        (options.moi_state_epoch_sgpr &&
         !append_store_u32_scalar_src(
             words, access_record_base + offsetof(ConSanMoiAccessRecord, epoch),
             *options.moi_state_epoch_sgpr, *options.scratch_vgpr, arch)) ||
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
                                                 uint16_t shift, uint32_t mask, uint16_t tmp_vgpr,
                                                 rj_code_arch_t arch) {
  const auto mask_words = build_v_and_b32_e32_literal(tmp_vgpr, mask, field_vgpr, arch);
  const auto shift_word =
      build_v_lshlrev_b32_e32(tmp_vgpr, scalar_positive_inline_u32(shift), tmp_vgpr, arch);
  const auto add_word = build_v_add_nc_u32_words(
      destination_vgpr, vector_source_vgpr(destination_vgpr), tmp_vgpr, arch);
  if (!mask_words || !shift_word || !add_word)
    return false;
  words.insert(words.end(), mask_words->begin(), mask_words->end());
  words.push_back(*shift_word);
  words.insert(words.end(), add_word->begin(), add_word->end());
  return true;
}

[[nodiscard]] bool append_add_literal_field(std::vector<uint32_t> &words, uint16_t destination_vgpr,
                                            uint32_t value, uint16_t tmp_vgpr,
                                            rj_code_arch_t arch) {
  if (value == 0)
    return true;
  const auto mov_value = build_v_mov_b32_e64_literal(tmp_vgpr, value, arch);
  const auto add_word = build_v_add_nc_u32_words(
      destination_vgpr, vector_source_vgpr(destination_vgpr), tmp_vgpr, arch);
  if (!mov_value || !add_word)
    return false;
  words.insert(words.end(), mov_value->begin(), mov_value->end());
  words.insert(words.end(), add_word->begin(), add_word->end());
  return true;
}

[[nodiscard]] bool append_inline_shadow_owner_field(std::vector<uint32_t> &words,
                                                    const ConSanOptions &options, uint16_t low_vgpr,
                                                    uint16_t tmp_vgpr, rj_code_arch_t arch,
                                                    std::optional<uint32_t> private_owner_offset,
                                                    std::vector<std::string> &errors) {
  if (options.moi_owner_vgpr) {
    return append_add_shifted_vgpr_field(words, low_vgpr, *options.moi_owner_vgpr,
                                         consan_moi_exact_shadow::owner_shift,
                                         consan_moi_exact_shadow::max_owner, tmp_vgpr, arch);
  }
  if (options.moi_state_owner_sgpr) {
    words.push_back(build_v_mov_b32_e32(tmp_vgpr, *options.moi_state_owner_sgpr, arch));
    return append_add_shifted_vgpr_field(words, low_vgpr, tmp_vgpr,
                                         consan_moi_exact_shadow::owner_shift,
                                         consan_moi_exact_shadow::max_owner, tmp_vgpr, arch);
  }
  if (!options.automatic_moi_private_epoch) {
    errors.emplace_back("ConSan MOI inline-shadow probe has no persistent owner representation");
    return false;
  }

  if (!private_owner_offset) {
    errors.emplace_back("ConSan MOI inline-shadow probe has no private owner offset");
    return false;
  }
  const auto load = build_moi_private_load_b32(tmp_vgpr, *private_owner_offset, arch);
  const auto wait = arch == ROCJITSU_CODE_ARCH_CDNA4 ? build_cdna4_s_wait_vmcnt0(arch)
                                                     : build_s_wait_loadcnt0(arch);
  if (!load || !wait) {
    errors.emplace_back("ConSan MOI inline-shadow probe could not load private owner state");
    return false;
  }
  words.insert(words.end(), load->begin(), load->end());
  words.push_back(*wait);

  return append_add_shifted_vgpr_field(words, low_vgpr, tmp_vgpr,
                                       consan_moi_exact_shadow::owner_shift,
                                       consan_moi_exact_shadow::max_owner, tmp_vgpr, arch);
}

[[nodiscard]] std::optional<std::vector<uint32_t>>
build_moi_private_load_b32(uint16_t vdst, uint32_t byte_offset, rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_CDNA4) {
    const auto encoded = build_cdna4_address_free_scratch_load_b32(vdst, byte_offset, arch);
    return encoded ? std::optional(std::vector<uint32_t>(encoded->begin(), encoded->end()))
                   : std::nullopt;
  }
  const auto encoded = build_address_free_scratch_load_b32(vdst, byte_offset, arch);
  return encoded ? std::optional(std::vector<uint32_t>(encoded->begin(), encoded->end()))
                 : std::nullopt;
}

[[nodiscard]] std::optional<std::vector<uint32_t>>
build_moi_private_store_b32(uint16_t vsrc, uint32_t byte_offset, rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_CDNA4) {
    const auto encoded = build_cdna4_address_free_scratch_store_b32(vsrc, byte_offset, arch);
    return encoded ? std::optional(std::vector<uint32_t>(encoded->begin(), encoded->end()))
                   : std::nullopt;
  }
  const auto encoded = build_address_free_scratch_store_b32(vsrc, byte_offset, arch);
  return encoded ? std::optional(std::vector<uint32_t>(encoded->begin(), encoded->end()))
                 : std::nullopt;
}

[[nodiscard]] bool append_inline_shadow_epoch_field(std::vector<uint32_t> &words,
                                                    const ConSanOptions &options,
                                                    std::optional<uint32_t> private_epoch_offset,
                                                    uint16_t low_vgpr, uint16_t tmp_vgpr,
                                                    rj_code_arch_t arch,
                                                    std::vector<std::string> &errors) {
  if (options.moi_epoch_vgpr) {
    return append_add_shifted_vgpr_field(words, low_vgpr, *options.moi_epoch_vgpr,
                                         consan_moi_exact_shadow::epoch_shift,
                                         consan_moi_exact_shadow::max_epoch, tmp_vgpr, arch);
  }
  if (options.moi_state_epoch_sgpr) {
    words.push_back(build_v_mov_b32_e32(tmp_vgpr, *options.moi_state_epoch_sgpr, arch));
    return append_add_shifted_vgpr_field(words, low_vgpr, tmp_vgpr,
                                         consan_moi_exact_shadow::epoch_shift,
                                         consan_moi_exact_shadow::max_epoch, tmp_vgpr, arch);
  }
  if (!options.automatic_moi_private_epoch || !private_epoch_offset) {
    errors.emplace_back("ConSan MOI inline-shadow probe has no persistent epoch representation");
    return false;
  }
  const auto load = build_moi_private_load_b32(tmp_vgpr, *private_epoch_offset, arch);
  const auto wait = arch == ROCJITSU_CODE_ARCH_CDNA4 ? build_cdna4_s_wait_vmcnt0(arch)
                                                     : build_s_wait_loadcnt0(arch);
  if (!load || !wait) {
    errors.emplace_back("ConSan MOI inline-shadow probe could not load private epoch state");
    return false;
  }
  words.insert(words.end(), load->begin(), load->end());
  words.push_back(*wait);
  return append_add_shifted_vgpr_field(words, low_vgpr, tmp_vgpr,
                                       consan_moi_exact_shadow::epoch_shift,
                                       consan_moi_exact_shadow::max_epoch, tmp_vgpr, arch);
}

[[nodiscard]] bool is_inline_shadow_access_candidate(const ConSanMoiCandidate &candidate) {
  const bool native_lds = candidate.source == ConSanMoiCandidateSource::NativeLds;
  const bool normalized_flat = (candidate.source == ConSanMoiCandidateSource::FlatGroup ||
                                candidate.source == ConSanMoiCandidateSource::FlatMaybeGroup) &&
                               is_first_light_flat_candidate(candidate);
  if (!native_lds && !normalized_flat)
    return false;
  if (candidate.kind != ConSanLdsAccessKind::Read && candidate.kind != ConSanLdsAccessKind::Write)
    return false;
  if (candidate.size == 0 || candidate.size % sizeof(uint32_t) != 0)
    return false;
  const std::optional<std::vector<ConSanMoiAccessRange>> ranges =
      candidate_access_ranges(candidate);
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
find_inline_shadow_access_candidates(const ConSanResult &result) {
  std::vector<const ConSanMoiCandidate *> candidates;
  for (const ConSanMoiCandidate &candidate : result.moi_candidates) {
    if (is_inline_shadow_access_candidate(candidate))
      candidates.push_back(&candidate);
  }
  return candidates;
}

[[nodiscard]] bool append_extract_exact_shadow_field(std::vector<uint32_t> &words,
                                                     uint16_t destination_vgpr,
                                                     uint16_t packed_vgpr, uint16_t shift,
                                                     uint32_t mask, rj_code_arch_t arch) {
  const auto shifted = build_v_lshrrev_b32_e32(destination_vgpr, scalar_positive_inline_u32(shift),
                                               packed_vgpr, arch);
  const auto masked = build_v_and_b32_e32_literal(destination_vgpr, mask, destination_vgpr, arch);
  if (!shifted || !masked)
    return false;
  words.push_back(*shifted);
  words.insert(words.end(), masked->begin(), masked->end());
  return true;
}

[[nodiscard]] bool append_inline_shadow_diagnostic_words(
    std::vector<uint32_t> &words, const ConSanMoiCandidate &candidate, const ConSanOptions &options,
    rj_code_arch_t arch, const ConSanMoiReportBufferLayout &layout, uint16_t old_value_vgpr,
    uint16_t old_value_hi_vgpr, uint16_t current_value_vgpr, uint16_t current_field_vgpr,
    uint16_t lds_byte_offset_vgpr, uint32_t static_byte_offset, uint32_t byte_count) {
  if (!options.moi_exec_save_sgpr || layout.diagnostic_capacity == 0)
    return true;

  const uint16_t tmp_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 4u);
  const uint64_t report_base = *options.moi_report_buffer_address;
  const uint64_t diagnostic_base = report_base + layout.diagnostic_records_offset;

  if (!append_save_moi_special_state(words, options, arch))
    return false;

  const auto mov_zero = build_v_mov_b32_e64_literal(tmp_vgpr, 0, arch);
  const auto prior_nonempty =
      build_v_cmp_gt_u32_e32_vcc(vector_source_vgpr(old_value_vgpr), tmp_vgpr, arch);
  const auto narrow_nonempty =
      build_s_and_saveexec_b64(*options.moi_exec_save_sgpr, kWave64VccLo, arch);
  if (!mov_zero || !prior_nonempty || !narrow_nonempty)
    return false;
  words.insert(words.end(), mov_zero->begin(), mov_zero->end());
  words.push_back(*prior_nonempty);
  words.push_back(*narrow_nonempty);

  if (!append_extract_exact_shadow_field(words, tmp_vgpr, old_value_vgpr,
                                         consan_moi_exact_shadow::owner_shift,
                                         consan_moi_exact_shadow::max_owner, arch) ||
      !append_extract_exact_shadow_field(words, current_field_vgpr, current_value_vgpr,
                                         consan_moi_exact_shadow::owner_shift,
                                         consan_moi_exact_shadow::max_owner, arch)) {
    return false;
  }
  const auto owner_ne =
      build_v_cmp_ne_u32_e32_vcc(vector_source_vgpr(current_field_vgpr), tmp_vgpr, arch);
  const auto narrow_conflict = build_s_and_saveexec_b64(
      static_cast<uint16_t>(*options.moi_exec_save_sgpr + 2u), kWave64VccLo, arch);
  if (!owner_ne || !narrow_conflict)
    return false;
  words.push_back(*owner_ne);
  words.push_back(*narrow_conflict);

  uint16_t next_exec_save_sgpr = static_cast<uint16_t>(*options.moi_exec_save_sgpr + 4u);
  const bool has_epoch =
      options.moi_epoch_vgpr || options.moi_state_epoch_sgpr || options.automatic_moi_private_epoch;
  if (has_epoch) {
    if (!append_extract_exact_shadow_field(words, tmp_vgpr, old_value_vgpr,
                                           consan_moi_exact_shadow::epoch_shift,
                                           consan_moi_exact_shadow::max_epoch, arch) ||
        !append_extract_exact_shadow_field(words, current_field_vgpr, current_value_vgpr,
                                           consan_moi_exact_shadow::epoch_shift,
                                           consan_moi_exact_shadow::max_epoch, arch)) {
      return false;
    }
    const auto epoch_eq =
        build_v_cmp_eq_u32_e32_vcc(vector_source_vgpr(current_field_vgpr), tmp_vgpr, arch);
    const auto narrow_same_epoch =
        build_s_and_saveexec_b64(next_exec_save_sgpr, kWave64VccLo, arch);
    if (!epoch_eq || !narrow_same_epoch)
      return false;
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
        build_s_and_saveexec_b64(next_exec_save_sgpr, kWave64VccLo, arch);
    if (!prior_kind || !kind_ne || !narrow_kind_conflict)
      return false;
    words.insert(words.end(), prior_kind->begin(), prior_kind->end());
    words.push_back(*kind_ne);
    words.push_back(*narrow_kind_conflict);
  }

  const uint16_t slot_vgpr = current_field_vgpr;
  constexpr uint16_t kScalarInlineMinusOne = 0xC1;
  const auto save_conflict_exec =
      build_s_mov_b64(static_cast<uint16_t>(*options.moi_exec_save_sgpr + 2u), kWave64ExecLo, arch);
  const auto mbcnt_lo = build_v_mbcnt_lo_u32_b32(tmp_vgpr, kScalarInlineMinusOne,
                                                 scalar_positive_inline_u32(0), arch);
  const auto mbcnt_hi =
      build_v_mbcnt_hi_u32_b32(tmp_vgpr, kScalarInlineMinusOne, vector_source_vgpr(tmp_vgpr), arch);
  const auto first_active_lane =
      build_v_cmp_eq_u32_e32_vcc(scalar_positive_inline_u32(0), tmp_vgpr, arch);
  const auto narrow_representative = build_s_and_saveexec_b64(
      static_cast<uint16_t>(*options.moi_exec_save_sgpr + 4u), kWave64VccLo, arch);
  if (!save_conflict_exec || !mbcnt_lo || !mbcnt_hi || !first_active_lane || !narrow_representative)
    return false;
  words.push_back(*save_conflict_exec);
  words.insert(words.end(), mbcnt_lo->begin(), mbcnt_lo->end());
  words.insert(words.end(), mbcnt_hi->begin(), mbcnt_hi->end());
  words.push_back(*first_active_lane);
  words.push_back(*narrow_representative);

  if (!append_atomic_fetch_add_one_u32(
          words, report_base + offsetof(ConSanMoiReportHeader, diagnostic_count), slot_vgpr,
          *options.scratch_vgpr, arch))
    return false;
  const auto mov_capacity = build_v_mov_b32_e64_literal(tmp_vgpr, layout.diagnostic_capacity, arch);
  const auto slot_in_capacity =
      build_v_cmp_gt_u32_e32_vcc(vector_source_vgpr(tmp_vgpr), slot_vgpr, arch);
  const auto narrow_capacity = build_s_and_saveexec_b64(
      static_cast<uint16_t>(*options.moi_exec_save_sgpr + 4u), kWave64VccLo, arch);
  if (!mov_capacity || !slot_in_capacity || !narrow_capacity)
    return false;
  words.insert(words.end(), mov_capacity->begin(), mov_capacity->end());
  words.push_back(*slot_in_capacity);
  words.push_back(*narrow_capacity);

  const auto store_literal = [&](size_t offset, uint32_t value) {
    return append_dynamic_diagnostic_store_u32_literal(words, diagnostic_base + offset, value,
                                                       slot_vgpr, *options.scratch_vgpr, arch);
  };
  const auto store_vgpr = [&](size_t offset, uint16_t value_vgpr) {
    return append_dynamic_diagnostic_store_u32_vgpr(words, diagnostic_base + offset, value_vgpr,
                                                    slot_vgpr, *options.scratch_vgpr, arch);
  };
  const auto store_scalar = [&](size_t offset, uint16_t scalar_src) {
    return append_dynamic_diagnostic_store_u32_scalar_src(
        words, diagnostic_base + offset, scalar_src, slot_vgpr, *options.scratch_vgpr, arch);
  };
  if (!store_literal(offsetof(ConSanMoiDiagnosticRecord, kind),
                     static_cast<uint32_t>(ConSanMoiDiagnosticKind::AccessConflict)) ||
      !store_literal(offsetof(ConSanMoiDiagnosticRecord, backend),
                     static_cast<uint32_t>(ConSanMoiEngine::InlineShadow)) ||
      !store_literal(offsetof(ConSanMoiDiagnosticRecord, generation), 1u) ||
      !store_literal(offsetof(ConSanMoiDiagnosticRecord, generation) + sizeof(uint32_t), 0u) ||
      !store_literal(offsetof(ConSanMoiDiagnosticRecord, second_instruction_offset),
                     static_cast<uint32_t>(candidate.text_offset)) ||
      !store_literal(offsetof(ConSanMoiDiagnosticRecord, second_access_kind),
                     static_cast<uint32_t>(current_kind)) ||
      !store_literal(offsetof(ConSanMoiDiagnosticRecord, first_lane_mask), 0u) ||
      !store_literal(offsetof(ConSanMoiDiagnosticRecord, first_lane_mask) + sizeof(uint32_t), 0u) ||
      !store_scalar(offsetof(ConSanMoiDiagnosticRecord, second_lane_mask),
                    static_cast<uint16_t>(*options.moi_exec_save_sgpr + 2u)) ||
      !store_scalar(offsetof(ConSanMoiDiagnosticRecord, second_lane_mask) + sizeof(uint32_t),
                    static_cast<uint16_t>(*options.moi_exec_save_sgpr + 3u)))
    return false;

  if (!append_extract_exact_shadow_field(words, tmp_vgpr, old_value_vgpr,
                                         consan_moi_exact_shadow::owner_shift,
                                         consan_moi_exact_shadow::max_owner, arch) ||
      !store_vgpr(offsetof(ConSanMoiDiagnosticRecord, first_owner_id), tmp_vgpr) ||
      !append_extract_exact_shadow_field(words, tmp_vgpr, current_value_vgpr,
                                         consan_moi_exact_shadow::owner_shift,
                                         consan_moi_exact_shadow::max_owner, arch) ||
      !store_vgpr(offsetof(ConSanMoiDiagnosticRecord, second_owner_id), tmp_vgpr)) {
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
  if (!store_vgpr(offsetof(ConSanMoiDiagnosticRecord, first_instruction_offset), tmp_vgpr)) {
    return false;
  }
  words.insert(words.end(), prior_kind->begin(), prior_kind->end());
  if (!store_vgpr(offsetof(ConSanMoiDiagnosticRecord, first_access_kind), tmp_vgpr)) {
    return false;
  }
  if (has_epoch) {
    if (!append_extract_exact_shadow_field(words, tmp_vgpr, current_value_vgpr,
                                           consan_moi_exact_shadow::epoch_shift,
                                           consan_moi_exact_shadow::max_epoch, arch) ||
        !store_vgpr(offsetof(ConSanMoiDiagnosticRecord, epoch), tmp_vgpr)) {
      return false;
    }
  }

  uint16_t diagnostic_offset_vgpr = lds_byte_offset_vgpr;
  if (static_byte_offset != 0) {
    if (!append_compute_effective_lds_byte_offset(words, current_value_vgpr, lds_byte_offset_vgpr,
                                                  static_byte_offset, arch))
      return false;
    diagnostic_offset_vgpr = current_value_vgpr;
  }
  if (!store_vgpr(offsetof(ConSanMoiDiagnosticRecord, first_lds_byte_offset),
                  diagnostic_offset_vgpr) ||
      !store_literal(offsetof(ConSanMoiDiagnosticRecord, first_lds_byte_count), byte_count) ||
      !store_vgpr(offsetof(ConSanMoiDiagnosticRecord, second_lds_byte_offset),
                  diagnostic_offset_vgpr) ||
      !store_literal(offsetof(ConSanMoiDiagnosticRecord, second_lds_byte_count), byte_count))
    return false;

  const auto restore_exec = build_s_mov_b64(kWave64ExecLo, *options.moi_exec_save_sgpr, arch);
  if (!restore_exec)
    return false;
  words.push_back(*restore_exec);
  return append_restore_moi_special_state(words, options, arch);
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_inline_shadow_words(
    std::span<const uint8_t> bytes, const ConSanMoiCandidate &candidate,
    const ConSanOptions &options, rj_code_arch_t arch, const ConSanMoiReportBufferLayout &layout,
    std::optional<uint32_t> private_epoch_offset, std::optional<uint32_t> private_owner_offset,
    std::vector<std::string> &errors) {
  if (!options.scratch_vgpr) {
    errors.emplace_back("ConSan MOI inline-shadow probe requires RJ_CONSAN_TMP_VGPR");
    return std::nullopt;
  }
  const uint16_t scratch_count = inline_shadow_scratch_count(options);
  if (static_cast<uint32_t>(*options.scratch_vgpr) + scratch_count > kMaxVgprs) {
    errors.emplace_back("ConSan MOI inline-shadow probe needs seven scratch VGPRs");
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
  if (!options.moi_owner_vgpr && !options.moi_state_owner_sgpr &&
      !options.automatic_moi_private_epoch) {
    errors.emplace_back("ConSan MOI inline-shadow probe requires persistent owner state");
    return std::nullopt;
  }
  if (reject_optional_scratch_range_overlap(options.moi_owner_vgpr, *options.scratch_vgpr,
                                            scratch_count, "MOI owner", errors) ||
      reject_optional_scratch_range_overlap(options.moi_epoch_vgpr, *options.scratch_vgpr,
                                            scratch_count, "MOI epoch", errors))
    return std::nullopt;

  const std::optional<std::vector<ConSanMoiAccessRange>> access_ranges =
      candidate_access_ranges(candidate);
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
  const auto wait_lds = build_s_wait_lds0(arch);
  if (!wait_lds) {
    errors.emplace_back("ConSan MOI inline-shadow probe could not encode the LDS completion wait");
    return std::nullopt;
  }
  words.push_back(*wait_lds);

  const auto mov_low = build_v_mov_b32_e64_literal(low_vgpr, low_literal, arch);
  const auto mov_high = build_v_mov_b32_e64_literal(high_vgpr, high_literal, arch);
  if (!mov_low || !mov_high) {
    errors.emplace_back("ConSan MOI inline-shadow probe could not encode exact-shadow literals");
    return std::nullopt;
  }
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
      const auto address_with_cell = build_v_add_nc_u32_words(
          address_lo_vgpr, vector_source_vgpr(address_lo_vgpr), tmp_vgpr, arch);
      if (!start_cell_shift || !byte_index_shift || !address_with_cell) {
        errors.emplace_back("ConSan MOI inline-shadow probe could not encode shadow publish");
        return std::nullopt;
      }
      words.push_back(*start_cell_shift);
      words.push_back(*byte_index_shift);
      words.insert(words.end(), address_with_cell->begin(), address_with_cell->end());
      if (cell_index != 0 &&
          !append_add_literal_field(words, address_lo_vgpr,
                                    static_cast<uint32_t>(cell_index * sizeof(uint64_t)), tmp_vgpr,
                                    arch)) {
        errors.emplace_back("ConSan MOI inline-shadow probe could not encode shadow cell offset");
        return std::nullopt;
      }
      // Diagnostic publication uses the packed-value temporaries. Rematerialize
      // the complete entry for each cell so a wide access cannot publish stale
      // metadata after diagnosing an earlier cell.
      words.insert(words.end(), mov_low->begin(), mov_low->end());
      if (!append_inline_shadow_owner_field(words, options, low_vgpr, tmp_vgpr, arch,
                                            private_owner_offset, errors)) {
        errors.emplace_back("ConSan MOI inline-shadow probe could not encode owner field");
        return std::nullopt;
      }
      if (!append_inline_shadow_epoch_field(words, options, private_epoch_offset, low_vgpr,
                                            tmp_vgpr, arch, errors)) {
        errors.emplace_back("ConSan MOI inline-shadow probe could not encode epoch field");
        return std::nullopt;
      }
      words.insert(words.end(), mov_high->begin(), mov_high->end());
      const auto atomic_swap = build_flat_atomic_swap_b64_vaddr_vsrc_vdst(
          address_lo_vgpr, low_vgpr, old_value_vgpr, /*return_old_value=*/true, kRdna4ScopeDevice,
          arch);
      if (!atomic_swap) {
        errors.emplace_back("ConSan MOI inline-shadow probe could not encode shadow publish");
        return std::nullopt;
      }
      words.insert(words.end(), atomic_swap->begin(), atomic_swap->end());
      const auto wait_flat = build_s_wait_flat_load0(arch);
      if (!wait_flat) {
        errors.emplace_back(
            "ConSan MOI inline-shadow probe could not encode the shadow atomic completion wait");
        return std::nullopt;
      }
      words.push_back(*wait_flat);
      if (!append_inline_shadow_diagnostic_words(
              words, candidate, options, arch, layout, old_value_vgpr,
              static_cast<uint16_t>(old_value_vgpr + 1u), low_vgpr, high_vgpr,
              *lds_byte_offset_vgpr, range.static_byte_offset, range.byte_count)) {
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
  if (arch != ROCJITSU_CODE_ARCH_RDNA4 && arch != ROCJITSU_CODE_ARCH_CDNA4) {
    result.warnings.emplace_back("ConSan MOI inline-shadow probe has unsupported target");
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
  const uint16_t scratch_count = inline_shadow_scratch_count(options);
  if (!validate_inline_shadow_exec_save_sgpr(options, result.warnings))
    return;
  if (!options.moi_owner_vgpr && !options.moi_state_owner_sgpr &&
      !options.automatic_moi_private_epoch) {
    result.warnings.emplace_back("ConSan MOI inline-shadow probe requires persistent owner state");
    return;
  }
  std::vector<const ConSanMoiCandidate *> candidates = find_inline_shadow_access_candidates(result);
  apply_test_kernel_filter(candidates, options);
  if (candidates.empty()) {
    result.warnings.emplace_back(
        "ConSan MOI inline-shadow probe found no supported native LDS or normalized group-flat "
        "load/store candidate");
    return;
  }

  struct PlannedInlineShadowPatch {
    const ConSanMoiCandidate *candidate = nullptr;
    DbiPatchPlacement placement;
    ResolvedMoiScratchPlan resources;
    std::optional<VgprSpillSequence> spill;
    std::optional<uint32_t> private_epoch_offset;
    std::optional<uint32_t> private_owner_offset;
    uint32_t required_private_bytes = 0;
  };
  std::vector<PlannedInlineShadowPatch> planned_patches;
  planned_patches.reserve(candidates.size());
  AmdGpuCodeObject code_object(bytes.data(), bytes.size());
  const uint64_t original_text_size =
      code_object.text_sections().size() == 1 ? code_object.text_sections().front()->size() : 0;
  DbiPatchPlacementPlanner placement_planner(arch, original_text_size);

  MoiSpillManagers spill_managers;
  for (const ConSanMoiCandidate *candidate_ptr : candidates) {
    const ConSanMoiCandidate &candidate = *candidate_ptr;
    const auto resources = resolve_moi_scratch(result, candidate, options, scratch_count);
    if (!resources)
      continue;

    std::optional<MoiPrivateEpochLayout> private_layout;
    if (options.automatic_moi_private_epoch) {
      private_layout = build_moi_private_epoch_layout(result, *resources, result.warnings);
      if (!private_layout)
        continue;
    }

    std::optional<VgprSpillSequence> spill;
    if (resources->source == ConSanRegisterAllocationSource::SpillRequired) {
      spill = build_moi_spill_sequence(
          result, *resources, spill_managers, arch, result.warnings,
          private_layout ? std::optional<uint32_t>(private_layout->ephemeral_base) : std::nullopt);
      if (!spill)
        continue;
    }

    ConSanOptions candidate_options = options;
    candidate_options.scratch_vgpr = resources->base;
    std::vector<std::string> candidate_errors;
    auto words = build_inline_shadow_words(
        bytes, candidate, candidate_options, arch, layout,
        private_layout ? std::optional<uint32_t>(private_layout->epoch_offset) : std::nullopt,
        private_layout ? std::optional<uint32_t>(private_layout->owner_offset) : std::nullopt,
        candidate_errors);
    if (!words) {
      result.warnings.insert(result.warnings.end(), candidate_errors.begin(),
                             candidate_errors.end());
      continue;
    }

    const uint64_t spill_bytes =
        spill ? static_cast<uint64_t>((spill->save_words.size() + spill->restore_words.size()) *
                                      sizeof(uint32_t))
              : 0;
    const uint64_t patch_bytes =
        static_cast<uint64_t>(words->size() * sizeof(uint32_t)) + spill_bytes;
    const uint32_t required_private_bytes =
        std::max(private_layout ? private_layout->ephemeral_base : 0u,
                 spill ? spill->total_private_bytes : 0u);
    const uint32_t available_padding =
        count_nop_padding(bytes, candidate.file_offset + candidate.size, arch);
    const uint64_t available_bytes =
        candidate.size + static_cast<uint64_t>(available_padding) * sizeof(uint32_t);
    DbiPatchPlacementRequest placement_request;
    placement_request.anchor_offset = candidate.text_offset;
    placement_request.original_size = candidate.size;
    placement_request.body_size = patch_bytes;
    placement_request.inline_capacity = spill || private_layout ? 0u : available_bytes;
    std::string placement_error;
    const auto placement = placement_planner.plan(placement_request, &placement_error);
    if (!placement) {
      result.warnings.emplace_back("ConSan MOI inline-shadow probe skipped access site: " +
                                   placement_error);
      continue;
    }

    PlannedInlineShadowPatch planned;
    planned.candidate = candidate_ptr;
    planned.placement = *placement;
    planned.resources = *resources;
    planned.spill = spill;
    if (private_layout)
      planned.private_epoch_offset = private_layout->epoch_offset;
    if (private_layout)
      planned.private_owner_offset = private_layout->owner_offset;
    planned.required_private_bytes = required_private_bytes;
    planned_patches.push_back(std::move(planned));
    if (planned_patches.size() == options.max_patches)
      break;
  }

  if (planned_patches.empty())
    return;

  MoiDescriptorVgprRequirements descriptor_requirements;
  MoiDescriptorSgprRequirements scalar_requirements;
  MoiDescriptorPrivateRequirements private_requirements;
  for (const PlannedInlineShadowPatch &planned_patch : planned_patches) {
    note_descriptor_requirements(descriptor_requirements, planned_patch.resources);
    note_moi_sgpr_requirements(scalar_requirements, planned_patch.resources, options);
    if (planned_patch.required_private_bytes != 0) {
      for (uint64_t descriptor_offset : planned_patch.resources.owner_descriptor_file_offsets) {
        auto [it, inserted] =
            private_requirements.emplace(descriptor_offset, planned_patch.required_private_bytes);
        if (!inserted)
          it->second = std::max(it->second, planned_patch.required_private_bytes);
      }
    }
  }

  const bool uses_appended_cave =
      std::ranges::any_of(planned_patches, [](const PlannedInlineShadowPatch &patch) {
        return patch.placement.kind == DbiPatchPlacementKind::AppendedCave;
      });
  if (uses_appended_cave) {
    CodeObjectPatcher patcher(code_object);
    if (!apply_descriptor_requirements(patcher, code_object, bytes, result, descriptor_requirements,
                                       arch, result.errors))
      return;
    if (!apply_spill_descriptor_requirements(patcher, code_object, bytes, result,
                                             private_requirements, result.errors))
      return;
    if (!apply_sgpr_descriptor_requirements(patcher, result, code_object, scalar_requirements, arch,
                                            result.errors))
      return;
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
      ConSanOptions candidate_options = options;
      candidate_options.scratch_vgpr = planned_patch.resources.base;
      auto words = build_inline_shadow_words(bytes, candidate, candidate_options, arch, layout,
                                             planned_patch.private_epoch_offset,
                                             planned_patch.private_owner_offset, result.errors);
      if (!words)
        return;

      ConSanPatchInfo info;
      info.kind = planned_patch.placement.kind == DbiPatchPlacementKind::AppendedCave
                      ? ConSanPatchKind::TrampolineMoiExactShadowStore
                      : ConSanPatchKind::InlineMoiExactShadowStore;
      info.anchor_offset = candidate.text_offset;
      info.scratch_vgpr = planned_patch.resources.base;
      info.owner_descriptor_file_offsets = planned_patch.resources.owner_descriptor_file_offsets;
      info.persistent_owner_private_offset = planned_patch.private_owner_offset;
      info.persistent_epoch_private_offset = planned_patch.private_epoch_offset;
      info.required_private_segment_size = planned_patch.required_private_bytes;
      if (planned_patch.spill)
        info.spilled_vgpr_count = planned_patch.spill->vgpr_count;

      if (planned_patch.placement.kind == DbiPatchPlacementKind::AppendedCave) {
        const uint64_t cave_text_offset = planned_patch.placement.body_offset;
        if (new_text.size() != cave_text_offset) {
          result.errors.emplace_back(
              "ConSan MOI inline-shadow probe emitted a stale appended-cave mapping");
          return;
        }
        const auto fwd = compute_sopp_branch_simm16(planned_patch.placement.anchor_offset,
                                                    planned_patch.placement.body_offset);
        const auto ret = compute_sopp_branch_simm16(planned_patch.placement.return_branch_offset,
                                                    planned_patch.placement.return_target);
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

        std::vector<uint32_t> cave_words;
        if (planned_patch.spill) {
          cave_words.insert(cave_words.end(), planned_patch.spill->save_words.begin(),
                            planned_patch.spill->save_words.end());
        }
        cave_words.insert(cave_words.end(), words->begin(), words->end());
        if (planned_patch.spill) {
          cave_words.insert(cave_words.end(), planned_patch.spill->restore_words.begin(),
                            planned_patch.spill->restore_words.end());
        }
        if (cave_words.size() * sizeof(uint32_t) != planned_patch.placement.body_size) {
          result.errors.emplace_back(
              "ConSan MOI inline-shadow probe body size changed after placement");
          return;
        }
        cave_words.push_back(build_s_branch(*ret, arch));
        append_words_bytes(new_text, cave_words);
        info.trampoline_offset = cave_text_offset;
        info.original_size = candidate.size;
        info.trampoline_size = static_cast<uint32_t>(cave_words.size() * sizeof(uint32_t));
      } else {
        const uint64_t patch_bytes = static_cast<uint64_t>(words->size() * sizeof(uint32_t));
        if (patch_bytes != planned_patch.placement.body_size) {
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
    if (!apply_spill_metadata_requirements(result.elf_bytes, result, private_requirements,
                                           result.errors)) {
      result.elf_bytes.clear();
      return;
    }
    result.patches.insert(result.patches.end(), patches.begin(), patches.end());
    result.modified = true;
    return;
  }

  result.elf_bytes.assign(bytes.begin(), bytes.end());
  for (const PlannedInlineShadowPatch &planned_patch : planned_patches) {
    const ConSanMoiCandidate &candidate = *planned_patch.candidate;
    ConSanOptions candidate_options = options;
    candidate_options.scratch_vgpr = planned_patch.resources.base;
    auto words = build_inline_shadow_words(bytes, candidate, candidate_options, arch, layout,
                                           planned_patch.private_epoch_offset,
                                           planned_patch.private_owner_offset, result.errors);
    if (!words) {
      result.elf_bytes.clear();
      return;
    }
    const uint64_t patch_bytes = static_cast<uint64_t>(words->size() * sizeof(uint32_t));
    if (patch_bytes != planned_patch.placement.body_size) {
      result.errors.emplace_back("ConSan MOI inline-shadow probe final patch size changed");
      result.elf_bytes.clear();
      return;
    }
    std::memcpy(result.elf_bytes.data() + candidate.file_offset, words->data(),
                static_cast<size_t>(patch_bytes));
  }

  if (!apply_descriptor_requirements(result.elf_bytes, result, descriptor_requirements, arch,
                                     result.errors)) {
    result.elf_bytes.clear();
    return;
  }
  if (!apply_sgpr_descriptor_requirements(result.elf_bytes, result, scalar_requirements, arch,
                                          result.errors)) {
    result.elf_bytes.clear();
    return;
  }

  for (const PlannedInlineShadowPatch &planned_patch : planned_patches) {
    const ConSanMoiCandidate &candidate = *planned_patch.candidate;
    ConSanPatchInfo info;
    info.kind = ConSanPatchKind::InlineMoiExactShadowStore;
    info.anchor_offset = candidate.text_offset;
    info.trampoline_offset = candidate.text_offset + candidate.size;
    info.original_size = static_cast<uint32_t>(planned_patch.placement.body_size);
    info.trampoline_size = 0;
    info.scratch_vgpr = planned_patch.resources.base;
    info.owner_descriptor_file_offsets = planned_patch.resources.owner_descriptor_file_offsets;
    info.persistent_owner_private_offset = planned_patch.private_owner_offset;
    info.persistent_epoch_private_offset = planned_patch.private_epoch_offset;
    info.required_private_segment_size = planned_patch.required_private_bytes;
    result.patches.push_back(info);
  }

  result.modified = true;
}

[[nodiscard]] bool append_direct_sampled_immediate_check(
    std::vector<uint32_t> &words, const ConSanOptions &options, rj_code_arch_t arch,
    uint32_t record_index, size_t sampled_watchpoints_offset, uint16_t current_low_vgpr,
    uint16_t current_high_vgpr, std::vector<std::string> &errors) {
  if (!options.moi_sampled_check || record_index == 0)
    return true;
  if (!options.moi_report_buffer_address || !options.scratch_vgpr || !options.moi_exec_save_sgpr) {
    errors.emplace_back("ConSan MOI sampled checker is missing report or scalar resources");
    return false;
  }

  const uint16_t address_lo_vgpr = *options.scratch_vgpr;
  const uint16_t address_hi_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 1u);
  const uint16_t tmp_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 4u);
  const uint16_t prior_low_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 5u);
  const uint16_t prior_high_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 6u);
  const uint64_t prior_address = *options.moi_report_buffer_address + sampled_watchpoints_offset +
                                 static_cast<uint64_t>(record_index - 1u) * sizeof(uint64_t);

  const auto save_vcc = build_s_mov_b64(*options.moi_exec_save_sgpr, kWave64VccLo, arch);
  if (!save_vcc ||
      !append_load_u32_vgpr(words, prior_address, prior_low_vgpr, *options.scratch_vgpr, arch) ||
      !append_load_u32_vgpr(words, prior_address + sizeof(uint32_t), prior_high_vgpr,
                            *options.scratch_vgpr, arch)) {
    errors.emplace_back("ConSan MOI sampled checker could not load the prior slot");
    return false;
  }
  words.push_back(*save_vcc);

  std::vector<size_t> skip_branch_indices;
  auto append_required_predicate = [&](std::optional<uint32_t> compare) {
    const auto skip = build_s_cbranch_vccz(0, arch);
    if (!compare || !skip)
      return false;
    words.push_back(*compare);
    skip_branch_indices.push_back(words.size());
    words.push_back(*skip);
    return true;
  };

  const auto prior_valid = build_v_and_b32_e32_literal(tmp_vgpr, 1u, prior_low_vgpr, arch);
  if (!prior_valid) {
    errors.emplace_back("ConSan MOI sampled checker could not decode validity");
    return false;
  }
  words.insert(words.end(), prior_valid->begin(), prior_valid->end());
  if (!append_required_predicate(
          build_v_cmp_ne_u32_e32_vcc(scalar_positive_inline_u32(0), tmp_vgpr, arch)) ||
      !append_extract_exact_shadow_field(words, address_lo_vgpr, current_low_vgpr,
                                         consan_moi_sampled_watchpoint::owner_shift,
                                         consan_moi_sampled_watchpoint::max_owner, arch) ||
      !append_extract_exact_shadow_field(words, address_hi_vgpr, prior_low_vgpr,
                                         consan_moi_sampled_watchpoint::owner_shift,
                                         consan_moi_sampled_watchpoint::max_owner, arch) ||
      !append_required_predicate(
          build_v_cmp_ne_u32_e32_vcc(vector_source_vgpr(address_lo_vgpr), address_hi_vgpr, arch))) {
    errors.emplace_back("ConSan MOI sampled checker could not compare owners");
    return false;
  }

  const uint32_t low_epoch_generation_mask =
      static_cast<uint32_t>(consan_moi_sampled_watchpoint::epoch_generation_mask);
  const uint32_t high_generation_mask =
      static_cast<uint32_t>(consan_moi_sampled_watchpoint::generation_mask >> 32u);
  const uint32_t high_range_mask = static_cast<uint32_t>(
      (consan_moi_sampled_watchpoint::start_mask | consan_moi_sampled_watchpoint::count_mask) >>
      32u);
  auto append_equal_masked = [&](uint16_t current_vgpr, uint16_t prior_vgpr, uint32_t mask) {
    const auto current = build_v_and_b32_e32_literal(address_lo_vgpr, mask, current_vgpr, arch);
    const auto prior = build_v_and_b32_e32_literal(address_hi_vgpr, mask, prior_vgpr, arch);
    if (!current || !prior)
      return false;
    words.insert(words.end(), current->begin(), current->end());
    words.insert(words.end(), prior->begin(), prior->end());
    return append_required_predicate(
        build_v_cmp_eq_u32_e32_vcc(vector_source_vgpr(address_lo_vgpr), address_hi_vgpr, arch));
  };
  if (!append_equal_masked(current_low_vgpr, prior_low_vgpr, low_epoch_generation_mask) ||
      !append_equal_masked(current_high_vgpr, prior_high_vgpr, high_generation_mask) ||
      !append_equal_masked(current_high_vgpr, prior_high_vgpr, high_range_mask)) {
    errors.emplace_back("ConSan MOI sampled checker could not compare ordering or ranges");
    return false;
  }

  if (!append_extract_exact_shadow_field(
          words, address_lo_vgpr, current_low_vgpr,
          consan_moi_sampled_watchpoint::access_kind_shift,
          (1u << consan_moi_sampled_watchpoint::access_kind_bits) - 1u, arch) ||
      !append_extract_exact_shadow_field(
          words, address_hi_vgpr, prior_low_vgpr, consan_moi_sampled_watchpoint::access_kind_shift,
          (1u << consan_moi_sampled_watchpoint::access_kind_bits) - 1u, arch)) {
    errors.emplace_back("ConSan MOI sampled checker could not decode access kinds");
    return false;
  }
  const auto both_read =
      build_v_and_b32_e32(tmp_vgpr, vector_source_vgpr(address_lo_vgpr), address_hi_vgpr, arch);
  if (!both_read) {
    errors.emplace_back("ConSan MOI sampled checker could not compare access kinds");
    return false;
  }
  words.push_back(*both_read);
  if (!append_required_predicate(build_v_cmp_ne_u32_e32_vcc(
          scalar_positive_inline_u32(static_cast<uint32_t>(ConSanMoiShadowAccessKind::Read)),
          tmp_vgpr, arch))) {
    errors.emplace_back("ConSan MOI sampled checker could not predicate conflicting kinds");
    return false;
  }

  const uint64_t immediate_conflict_count_address =
      *options.moi_report_buffer_address + offsetof(ConSanMoiReportHeader, event_counter);
  if (!append_atomic_fetch_add_one_u32(words, immediate_conflict_count_address, tmp_vgpr,
                                       *options.scratch_vgpr, arch)) {
    errors.emplace_back("ConSan MOI sampled checker could not increment the conflict counter");
    return false;
  }

  const size_t restore_index = words.size();
  const auto restore_vcc = build_s_mov_b64(kWave64VccLo, *options.moi_exec_save_sgpr, arch);
  if (!restore_vcc) {
    errors.emplace_back("ConSan MOI sampled checker could not restore VCC");
    return false;
  }
  for (size_t branch_index : skip_branch_indices) {
    const size_t skipped_words = restore_index - branch_index - 1u;
    if (skipped_words > static_cast<size_t>(std::numeric_limits<int16_t>::max())) {
      errors.emplace_back("ConSan MOI sampled checker skip branch is out of range");
      return false;
    }
    const auto skip = build_s_cbranch_vccz(static_cast<int16_t>(skipped_words), arch);
    if (!skip)
      return false;
    words[branch_index] = *skip;
  }
  words.push_back(*restore_vcc);
  return true;
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_direct_sampled_watchpoint_words(
    std::span<const uint8_t> bytes, const ConSanMoiCandidate &candidate,
    const ConSanOptions &options, rj_code_arch_t arch, uint32_t record_index,
    size_t sampled_watchpoints_offset, std::optional<uint32_t> private_owner_offset,
    std::optional<uint32_t> private_epoch_offset, std::vector<std::string> &errors) {
  if (!options.scratch_vgpr) {
    errors.emplace_back("ConSan MOI sampled probe requires RJ_CONSAN_TMP_VGPR");
    return std::nullopt;
  }
  if (options.automatic_moi_private_epoch && (!private_owner_offset || !private_epoch_offset)) {
    errors.emplace_back("ConSan MOI sampled probe has incomplete private identity layout");
    return std::nullopt;
  }
  const uint16_t scratch_count = options.moi_sampled_check ? 7u : 5u;
  if (static_cast<uint32_t>(*options.scratch_vgpr) + scratch_count > kMaxVgprs) {
    errors.emplace_back("ConSan MOI sampled probe exceeds the VGPR file");
    return std::nullopt;
  }
  auto lds_byte_offset_vgpr = candidate_lds_byte_offset_vgpr(candidate, errors);
  if (!lds_byte_offset_vgpr)
    return std::nullopt;
  if (reject_candidate_scratch_range_overlap(candidate, *options.scratch_vgpr, scratch_count,
                                             errors))
    return std::nullopt;
  if (reject_optional_scratch_range_overlap(options.moi_owner_vgpr, *options.scratch_vgpr,
                                            scratch_count, "MOI owner", errors) ||
      reject_optional_scratch_range_overlap(options.moi_epoch_vgpr, *options.scratch_vgpr,
                                            scratch_count, "MOI epoch", errors))
    return std::nullopt;
  auto byte_count = byte_count_for_candidate(candidate);
  if (!byte_count) {
    errors.emplace_back("ConSan MOI sampled probe could not determine LDS byte count");
    return std::nullopt;
  }

  const ConSanMoiLdsCellRange static_range = consan_moi_lds_cell_range_for_bytes(0, *byte_count);
  std::optional<uint16_t> derived_owner_vgpr;
  std::vector<uint32_t> derived_owner_words;
  if (!options.moi_owner_vgpr && !options.moi_state_owner_sgpr &&
      !options.automatic_moi_private_epoch && candidate.kernel_descriptor_file_offset) {
    const auto owner_input =
        moi_descriptor_owner_input(bytes, *candidate.kernel_descriptor_file_offset, arch, errors);
    if (!owner_input)
      return std::nullopt;
    const uint16_t value_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 4u);
    if (!append_moi_owner_input(derived_owner_words, value_vgpr, *owner_input, arch, errors)) {
      errors.emplace_back("ConSan MOI sampled probe could not encode stable owner derivation");
      return std::nullopt;
    }
    derived_owner_vgpr = value_vgpr;
  }

  const uint16_t low_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 2u);
  const uint16_t high_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 3u);
  const uint16_t tmp_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 4u);
  const uint16_t owner_vgpr = options.moi_owner_vgpr.value_or(
      (private_owner_offset || options.moi_state_owner_sgpr) ? tmp_vgpr
                                                             : derived_owner_vgpr.value_or(0));
  const bool runtime_sampled = options.moi_runtime_sample_stride > 1;
  if (runtime_sampled && !options.moi_owner_vgpr && !options.moi_state_owner_sgpr &&
      !derived_owner_vgpr && !private_owner_offset) {
    errors.emplace_back("ConSan MOI runtime sampled probe could not derive a wave owner");
    return std::nullopt;
  }
  if (runtime_sampled && (!options.moi_exec_save_sgpr || *options.moi_exec_save_sgpr > 104u ||
                          *options.moi_exec_save_sgpr % 2u != 0u)) {
    errors.emplace_back(
        "ConSan MOI runtime sampled probe needs an even VCC-save SGPR pair in 0..104");
    return std::nullopt;
  }
  const auto kind = consan_moi_shadow_kind_from_access_kind(candidate.kind);
  const uint32_t generation =
      options.moi_report_generation & consan_moi_sampled_watchpoint::max_generation;
  const uint64_t generation_field = static_cast<uint64_t>(generation)
                                    << consan_moi_sampled_watchpoint::generation_shift;
  const uint32_t low_literal = static_cast<uint32_t>(
      consan_moi_sampled_watchpoint::valid_mask |
      (static_cast<uint64_t>(kind) << consan_moi_sampled_watchpoint::access_kind_shift) |
      generation_field);
  const uint32_t encoded_cell_count = encode_consan_moi_sampled_cell_count(static_range.cell_count)
                                      << (consan_moi_sampled_watchpoint::count_shift - 32u);
  const uint32_t encoded_generation_high = static_cast<uint32_t>(generation_field >> 32u);
  const uint64_t sampled_entry_address = *options.moi_report_buffer_address +
                                         sampled_watchpoints_offset +
                                         static_cast<uint64_t>(record_index) * sizeof(uint64_t);

  std::vector<uint32_t> words;
  words.reserve(candidate.size / sizeof(uint32_t) + 40u + derived_owner_words.size() +
                (options.moi_epoch_vgpr ? 2u : 0u));
  words.insert(words.end(), derived_owner_words.begin(), derived_owner_words.end());
  for (uint64_t offset = 0; offset < candidate.size; offset += sizeof(uint32_t)) {
    uint32_t word = 0;
    std::memcpy(&word, bytes.data() + candidate.file_offset + offset, sizeof(word));
    words.push_back(word);
  }
  const auto wait_lds = build_s_wait_lds0(arch);
  if (!wait_lds) {
    errors.emplace_back("ConSan MOI sampled probe could not encode the LDS completion wait");
    return std::nullopt;
  }
  words.push_back(*wait_lds);

  if (options.moi_state_owner_sgpr)
    words.push_back(build_v_mov_b32_e32(tmp_vgpr, *options.moi_state_owner_sgpr, arch));

  std::optional<size_t> runtime_skip_branch_index;
  if (runtime_sampled) {
    if (private_owner_offset &&
        !append_moi_private_load_wait(words, tmp_vgpr, *private_owner_offset, arch)) {
      errors.emplace_back("ConSan MOI runtime sampled probe could not load private owner");
      return std::nullopt;
    }
    const auto save_vcc = build_s_mov_b64(*options.moi_exec_save_sgpr, kWave64VccLo, arch);
    const auto selected_owner = build_v_and_b32_e32_literal(
        low_vgpr, options.moi_runtime_sample_stride - 1u, owner_vgpr, arch);
    const auto selected = build_v_cmp_eq_u32_e32_vcc(
        scalar_positive_inline_u32(options.moi_runtime_sample_offset), low_vgpr, arch);
    const auto skip = build_s_cbranch_vccz(0, arch);
    if (!save_vcc || !selected_owner || !selected || !skip) {
      errors.emplace_back("ConSan MOI runtime sampled probe could not encode wave selector");
      return std::nullopt;
    }
    words.push_back(*save_vcc);
    words.insert(words.end(), selected_owner->begin(), selected_owner->end());
    words.push_back(*selected);
    runtime_skip_branch_index = words.size();
    words.push_back(*skip);
  }
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
  if (private_owner_offset &&
      !append_moi_private_load_wait(words, tmp_vgpr, *private_owner_offset, arch)) {
    errors.emplace_back("ConSan MOI sampled probe could not load private owner");
    return std::nullopt;
  }
  if ((options.moi_owner_vgpr || options.moi_state_owner_sgpr || derived_owner_vgpr ||
       private_owner_offset) &&
      !append_add_shifted_vgpr_field(words, low_vgpr, owner_vgpr,
                                     consan_moi_sampled_watchpoint::owner_shift,
                                     consan_moi_sampled_watchpoint::max_owner, tmp_vgpr, arch)) {
    errors.emplace_back("ConSan MOI sampled probe could not encode owner field");
    return std::nullopt;
  }
  if (private_epoch_offset &&
      !append_moi_private_load_wait(words, tmp_vgpr, *private_epoch_offset, arch)) {
    errors.emplace_back("ConSan MOI sampled probe could not load private epoch");
    return std::nullopt;
  }
  if (options.moi_state_epoch_sgpr)
    words.push_back(build_v_mov_b32_e32(tmp_vgpr, *options.moi_state_epoch_sgpr, arch));
  const std::optional<uint16_t> epoch_vgpr =
      options.moi_epoch_vgpr ? options.moi_epoch_vgpr
                             : ((private_epoch_offset || options.moi_state_epoch_sgpr)
                                    ? std::optional<uint16_t>(tmp_vgpr)
                                    : std::nullopt);
  if (epoch_vgpr && !append_add_shifted_vgpr_field(
                        words, low_vgpr, *epoch_vgpr, consan_moi_sampled_watchpoint::epoch_shift,
                        consan_moi_sampled_watchpoint::max_epoch, tmp_vgpr, arch)) {
    errors.emplace_back("ConSan MOI sampled probe could not encode epoch field");
    return std::nullopt;
  }
  words.push_back(*start_cell_shift);
  words.push_back(*high_from_start);
  if (!append_add_literal_field(words, high_vgpr, encoded_cell_count | encoded_generation_high,
                                tmp_vgpr, arch)) {
    errors.emplace_back("ConSan MOI sampled probe could not encode cell-count field");
    return std::nullopt;
  }
  if (!append_direct_sampled_immediate_check(words, options, arch, record_index,
                                             sampled_watchpoints_offset, low_vgpr, high_vgpr,
                                             errors))
    return std::nullopt;
  if (!append_store_u32_vgpr(words, sampled_entry_address, low_vgpr, *options.scratch_vgpr, arch) ||
      !append_store_u32_vgpr(words, sampled_entry_address + sizeof(uint32_t), high_vgpr,
                             *options.scratch_vgpr, arch)) {
    errors.emplace_back("ConSan MOI sampled probe could not encode sampled entry stores");
    return std::nullopt;
  }
  if (runtime_skip_branch_index) {
    const size_t restore_index = words.size();
    const size_t skipped_words = restore_index - *runtime_skip_branch_index - 1u;
    if (skipped_words > static_cast<size_t>(std::numeric_limits<int16_t>::max())) {
      errors.emplace_back("ConSan MOI runtime sampled probe skip branch is out of range");
      return std::nullopt;
    }
    const auto skip = build_s_cbranch_vccz(static_cast<int16_t>(skipped_words), arch);
    const auto restore_vcc = build_s_mov_b64(kWave64VccLo, *options.moi_exec_save_sgpr, arch);
    if (!skip || !restore_vcc) {
      errors.emplace_back("ConSan MOI runtime sampled probe could not restore VCC");
      return std::nullopt;
    }
    words[*runtime_skip_branch_index] = *skip;
    words.push_back(*restore_vcc);
  }
  return words;
}

void try_apply_direct_sampled_watchpoint_patch(std::span<const uint8_t> bytes,
                                               const ConSanOptions &options, rj_code_arch_t arch,
                                               ConSanResult &result) {
  if (!options.moi_report_buffer_address)
    return;
  if (!require_native_feature(arch, ConSanNativeFeature::Sampled, "moi-sampled-access", result))
    return;

  const ConSanMoiReportBufferLayout layout =
      consan_moi_direct_sampled_report_buffer_layout_for_bytes(options.moi_report_buffer_size);
  const uint32_t max_candidates = std::min(options.max_patches, layout.sampled_watchpoint_capacity);
  if (max_candidates == 0) {
    result.warnings.emplace_back(
        "ConSan MOI sampled probe requires room for the report header and one sampled entry");
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
  if (options.moi_runtime_sample_stride == 0 || options.moi_runtime_sample_stride > 1024 ||
      (options.moi_runtime_sample_stride & (options.moi_runtime_sample_stride - 1u)) != 0 ||
      options.moi_runtime_sample_offset >= options.moi_runtime_sample_stride) {
    result.warnings.emplace_back(
        "ConSan MOI runtime sample stride must be a power of two in 1..1024 and offset must "
        "be less than stride");
    return;
  }
  std::vector<const ConSanMoiCandidate *> candidates =
      find_first_light_access_record_candidates(result, bytes);
  apply_test_kernel_filter(candidates, options);
  if (candidates.empty()) {
    result.warnings.emplace_back(
        "ConSan MOI sampled probe found no native LDS or likely group flat load/store candidate");
    return;
  }

  std::vector<const ConSanMoiCandidate *> sampled_candidates;
  sampled_candidates.reserve(candidates.size());
  for (uint32_t i = 0; i < candidates.size(); ++i) {
    if (i % options.moi_sample_stride == options.moi_sample_offset)
      sampled_candidates.push_back(candidates[i]);
  }
  prefer_spill_free_scalar_candidates(sampled_candidates, result, options);

  const uint32_t sampled_filter_candidate_count = static_cast<uint32_t>(sampled_candidates.size());

  struct PlannedSampledPatch {
    const ConSanMoiCandidate *candidate = nullptr;
    DbiPatchPlacement placement;
    ResolvedMoiScratchPlan resources;
    std::optional<VgprSpillSequence> spill;
    std::optional<MoiPrivateEpochLayout> private_layout;
    uint32_t required_private_bytes = 0;
  };
  std::vector<PlannedSampledPatch> planned_patches;
  planned_patches.reserve(candidates.size());
  AmdGpuCodeObject code_object(bytes.data(), bytes.size());
  const uint64_t original_text_size =
      code_object.text_sections().size() == 1 ? code_object.text_sections().front()->size() : 0;
  DbiPatchPlacementPlanner placement_planner(arch, original_text_size);
  MoiSpillManagers spill_managers;
  for (const ConSanMoiCandidate *candidate_ptr : sampled_candidates) {
    const ConSanMoiCandidate &candidate = *candidate_ptr;
    const auto resources =
        resolve_moi_scratch(result, candidate, options, options.moi_sampled_check ? 7u : 5u);
    if (!resources) {
      continue;
    }
    std::optional<MoiPrivateEpochLayout> private_layout;
    if (options.automatic_moi_private_epoch) {
      private_layout = build_moi_private_epoch_layout(result, *resources, result.warnings);
      if (!private_layout)
        continue;
    }
    std::optional<VgprSpillSequence> spill;
    if (resources->source == ConSanRegisterAllocationSource::SpillRequired) {
      spill = build_moi_spill_sequence(
          result, *resources, spill_managers, arch, result.warnings,
          private_layout ? std::optional<uint32_t>(private_layout->ephemeral_base) : std::nullopt);
      if (!spill)
        continue;
    }
    ConSanOptions candidate_options = options;
    candidate_options.scratch_vgpr = resources->base;
    std::vector<std::string> candidate_errors;
    const uint32_t sampled_record_index = static_cast<uint32_t>(planned_patches.size());
    auto words = build_direct_sampled_watchpoint_words(
        bytes, candidate, candidate_options, arch, sampled_record_index,
        layout.sampled_watchpoints_offset,
        private_layout ? std::optional<uint32_t>(private_layout->owner_offset) : std::nullopt,
        private_layout ? std::optional<uint32_t>(private_layout->epoch_offset) : std::nullopt,
        candidate_errors);
    if (!words) {
      result.warnings.insert(result.warnings.end(), candidate_errors.begin(),
                             candidate_errors.end());
      continue;
    }

    const uint64_t spill_bytes =
        spill ? static_cast<uint64_t>((spill->save_words.size() + spill->restore_words.size()) *
                                      sizeof(uint32_t))
              : 0;
    const uint64_t patch_bytes =
        static_cast<uint64_t>(words->size() * sizeof(uint32_t)) + spill_bytes;
    const uint32_t available_padding =
        count_nop_padding(bytes, candidate.file_offset + candidate.size, arch);
    const uint64_t available_bytes =
        candidate.size + static_cast<uint64_t>(available_padding) * sizeof(uint32_t);
    DbiPatchPlacementRequest placement_request;
    placement_request.anchor_offset = candidate.text_offset;
    placement_request.original_size = candidate.size;
    placement_request.body_size = patch_bytes;
    placement_request.inline_capacity = spill ? 0u : available_bytes;
    std::string placement_error;
    const auto placement = placement_planner.plan(placement_request, &placement_error);
    if (!placement) {
      result.warnings.emplace_back("ConSan MOI sampled probe skipped load/store site: " +
                                   placement_error);
      continue;
    }

    PlannedSampledPatch planned;
    planned.candidate = candidate_ptr;
    planned.placement = *placement;
    planned.resources = *resources;
    planned.spill = spill;
    planned.private_layout = private_layout;
    planned.required_private_bytes = std::max(private_layout ? private_layout->ephemeral_base : 0u,
                                              spill ? spill->total_private_bytes : 0u);
    planned_patches.push_back(std::move(planned));
    if (planned_patches.size() == max_candidates)
      break;
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

  MoiDescriptorVgprRequirements descriptor_requirements;
  MoiDescriptorSgprRequirements scalar_requirements;
  MoiDescriptorPrivateRequirements private_requirements;
  for (const PlannedSampledPatch &planned_patch : planned_patches) {
    note_descriptor_requirements(descriptor_requirements, planned_patch.resources);
    note_moi_sgpr_requirements(scalar_requirements, planned_patch.resources, options);
    if (planned_patch.required_private_bytes != 0) {
      for (uint64_t descriptor_offset : planned_patch.resources.owner_descriptor_file_offsets) {
        auto [it, inserted] =
            private_requirements.emplace(descriptor_offset, planned_patch.required_private_bytes);
        if (!inserted)
          it->second = std::max(it->second, planned_patch.required_private_bytes);
      }
    }
  }

  const bool uses_appended_cave =
      std::ranges::any_of(planned_patches, [](const PlannedSampledPatch &patch) {
        return patch.placement.kind == DbiPatchPlacementKind::AppendedCave;
      });
  if (uses_appended_cave) {
    CodeObjectPatcher patcher(code_object);
    if (!apply_descriptor_requirements(patcher, code_object, bytes, result, descriptor_requirements,
                                       arch, result.errors)) {
      return;
    }
    if (!apply_spill_descriptor_requirements(patcher, code_object, bytes, result,
                                             private_requirements, result.errors)) {
      return;
    }
    if (!apply_sgpr_descriptor_requirements(patcher, result, code_object, scalar_requirements, arch,
                                            result.errors))
      return;
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
      ConSanOptions candidate_options = options;
      candidate_options.scratch_vgpr = planned_patch.resources.base;
      auto words = build_direct_sampled_watchpoint_words(
          bytes, candidate, candidate_options, arch, i, layout.sampled_watchpoints_offset,
          planned_patch.private_layout
              ? std::optional<uint32_t>(planned_patch.private_layout->owner_offset)
              : std::nullopt,
          planned_patch.private_layout
              ? std::optional<uint32_t>(planned_patch.private_layout->epoch_offset)
              : std::nullopt,
          result.errors);
      if (!words)
        return;
      const uint64_t spill_bytes =
          planned_patch.spill ? static_cast<uint64_t>((planned_patch.spill->save_words.size() +
                                                       planned_patch.spill->restore_words.size()) *
                                                      sizeof(uint32_t))
                              : 0;
      const uint64_t planned_probe_bytes = planned_patch.placement.body_size - spill_bytes;
      if (words->size() * sizeof(uint32_t) > planned_probe_bytes) {
        result.errors.emplace_back("ConSan MOI sampled probe final patch grew after planning");
        return;
      }
      words->resize(static_cast<size_t>(planned_probe_bytes / sizeof(uint32_t)),
                    build_s_nop(0, arch));

      ConSanPatchInfo info;
      info.kind = planned_patch.placement.kind == DbiPatchPlacementKind::AppendedCave
                      ? ConSanPatchKind::TrampolineMoiSampledWatchpointStore
                      : ConSanPatchKind::InlineMoiSampledWatchpointStore;
      info.anchor_offset = candidate.text_offset;
      info.scratch_vgpr = planned_patch.resources.base;
      info.owner_descriptor_file_offsets = planned_patch.resources.owner_descriptor_file_offsets;
      if (planned_patch.private_layout) {
        info.persistent_owner_private_offset = planned_patch.private_layout->owner_offset;
        info.persistent_epoch_private_offset = planned_patch.private_layout->epoch_offset;
      }
      info.required_private_segment_size = planned_patch.required_private_bytes;
      if (planned_patch.spill) {
        info.spilled_vgpr_count = planned_patch.spill->vgpr_count;
      }

      if (planned_patch.placement.kind == DbiPatchPlacementKind::AppendedCave) {
        const uint64_t cave_text_offset = planned_patch.placement.body_offset;
        if (new_text.size() != cave_text_offset) {
          result.errors.emplace_back(
              "ConSan MOI sampled probe emitted a stale appended-cave mapping");
          return;
        }
        const auto fwd = compute_sopp_branch_simm16(planned_patch.placement.anchor_offset,
                                                    planned_patch.placement.body_offset);
        const auto ret = compute_sopp_branch_simm16(planned_patch.placement.return_branch_offset,
                                                    planned_patch.placement.return_target);
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

        std::vector<uint32_t> cave_words;
        if (planned_patch.spill) {
          cave_words.insert(cave_words.end(), planned_patch.spill->save_words.begin(),
                            planned_patch.spill->save_words.end());
        }
        cave_words.insert(cave_words.end(), words->begin(), words->end());
        if (planned_patch.spill) {
          cave_words.insert(cave_words.end(), planned_patch.spill->restore_words.begin(),
                            planned_patch.spill->restore_words.end());
        }
        if (cave_words.size() * sizeof(uint32_t) != planned_patch.placement.body_size) {
          result.errors.emplace_back("ConSan MOI sampled probe body size changed after placement");
          return;
        }
        cave_words.push_back(build_s_branch(*ret, arch));
        append_words_bytes(new_text, cave_words);
        info.trampoline_offset = cave_text_offset;
        info.original_size = candidate.size;
        info.trampoline_size = static_cast<uint32_t>(cave_words.size() * sizeof(uint32_t));
      } else {
        const uint64_t patch_bytes = static_cast<uint64_t>(words->size() * sizeof(uint32_t));
        if (patch_bytes != planned_patch.placement.body_size) {
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
    if (!apply_spill_metadata_requirements(result.elf_bytes, result, private_requirements,
                                           result.errors)) {
      result.elf_bytes.clear();
      return;
    }
    result.patches.insert(result.patches.end(), patches.begin(), patches.end());
    result.modified = true;
    return;
  }

  result.elf_bytes.assign(bytes.begin(), bytes.end());
  for (uint32_t i = 0; i < planned_patches.size(); ++i) {
    const PlannedSampledPatch &planned_patch = planned_patches[i];
    const ConSanMoiCandidate &candidate = *planned_patch.candidate;
    ConSanOptions candidate_options = options;
    candidate_options.scratch_vgpr = planned_patch.resources.base;
    auto words = build_direct_sampled_watchpoint_words(
        bytes, candidate, candidate_options, arch, i, layout.sampled_watchpoints_offset,
        planned_patch.private_layout
            ? std::optional<uint32_t>(planned_patch.private_layout->owner_offset)
            : std::nullopt,
        planned_patch.private_layout
            ? std::optional<uint32_t>(planned_patch.private_layout->epoch_offset)
            : std::nullopt,
        result.errors);
    if (!words) {
      result.elf_bytes.clear();
      return;
    }
    const uint64_t planned_probe_bytes = planned_patch.placement.body_size;
    if (words->size() * sizeof(uint32_t) > planned_probe_bytes) {
      result.errors.emplace_back("ConSan MOI sampled probe final patch grew after planning");
      result.elf_bytes.clear();
      return;
    }
    words->resize(static_cast<size_t>(planned_probe_bytes / sizeof(uint32_t)),
                  build_s_nop(0, arch));
    const uint64_t patch_bytes = static_cast<uint64_t>(words->size() * sizeof(uint32_t));
    if (patch_bytes != planned_patch.placement.body_size) {
      result.errors.emplace_back("ConSan MOI sampled probe final patch size changed");
      result.elf_bytes.clear();
      return;
    }
    std::memcpy(result.elf_bytes.data() + candidate.file_offset, words->data(),
                static_cast<size_t>(patch_bytes));
  }

  if (!apply_descriptor_requirements(result.elf_bytes, result, descriptor_requirements, arch,
                                     result.errors)) {
    result.elf_bytes.clear();
    return;
  }
  if (!apply_sgpr_descriptor_requirements(result.elf_bytes, result, scalar_requirements, arch,
                                          result.errors)) {
    result.elf_bytes.clear();
    return;
  }

  for (const PlannedSampledPatch &planned_patch : planned_patches) {
    const ConSanMoiCandidate &candidate = *planned_patch.candidate;
    ConSanPatchInfo info;
    info.kind = ConSanPatchKind::InlineMoiSampledWatchpointStore;
    info.anchor_offset = candidate.text_offset;
    info.trampoline_offset = candidate.text_offset + candidate.size;
    info.original_size = static_cast<uint32_t>(planned_patch.placement.body_size);
    info.trampoline_size = 0;
    info.scratch_vgpr = planned_patch.resources.base;
    info.owner_descriptor_file_offsets = planned_patch.resources.owner_descriptor_file_offsets;
    if (planned_patch.private_layout) {
      info.persistent_owner_private_offset = planned_patch.private_layout->owner_offset;
      info.persistent_epoch_private_offset = planned_patch.private_layout->epoch_offset;
    }
    info.required_private_segment_size = planned_patch.required_private_bytes;
    result.patches.push_back(info);
  }

  result.modified = true;
}

void try_apply_first_light_access_record_patch(std::span<const uint8_t> bytes,
                                               const ConSanOptions &options, rj_code_arch_t arch,
                                               ConSanResult &result) {
  if (!options.moi_report_buffer_address)
    return;
  if (!require_native_feature(arch, ConSanNativeFeature::RecordReplay, "moi-record-replay-access",
                              result))
    return;
  if (options.moi_report_buffer_size < consan_moi_report_buffer_min_bytes(1, 0, 0, 0)) {
    result.warnings.emplace_back(
        "ConSan MOI first-light probe requires room for the report header and one access record");
    return;
  }

  const ConSanMoiReportBufferLayout layout = consan_moi_report_buffer_layout_for_bytes(
      options.moi_report_buffer_size, options.moi_track_barriers, options.moi_track_atomics);
  if (options.max_patches == 0 || layout.access_record_capacity == 0)
    return;
  const uint16_t access_scratch_count = options.moi_dynamic_access_records ? 6u : 3u;
  if (options.moi_dynamic_access_records && !options.moi_exec_save_sgpr) {
    result.warnings.emplace_back(
        "ConSan MOI dynamic access-record probe requires RJ_CONSAN_MOI_EXEC_SAVE_SGPR");
    return;
  }
  if (options.moi_dynamic_access_records &&
      (*options.moi_exec_save_sgpr > 100u || *options.moi_exec_save_sgpr % 2u != 0u)) {
    result.warnings.emplace_back(
        "ConSan MOI dynamic access-record probe requires an even RJ_CONSAN_MOI_EXEC_SAVE_SGPR in "
        "0..100");
    return;
  }
  std::vector<const ConSanMoiCandidate *> candidates =
      find_first_light_access_record_candidates(result, bytes);
  apply_test_kernel_filter(candidates, options);
  if (candidates.empty()) {
    result.warnings.emplace_back("ConSan MOI first-light probe found no native LDS or likely group "
                                 "flat load/store candidate");
    return;
  }
  prefer_spill_free_scalar_candidates(candidates, result, options);

  struct PlannedAccessRecordPatch {
    const ConSanMoiCandidate *candidate = nullptr;
    DbiPatchPlacement placement;
    uint32_t record_index = 0;
    uint32_t record_count = 0;
    ResolvedMoiScratchPlan resources;
    std::optional<VgprSpillSequence> spill;
    std::optional<MoiPrivateEpochLayout> private_layout;
    uint32_t required_private_bytes = 0;
  };
  std::vector<PlannedAccessRecordPatch> planned_patches;
  uint32_t planned_record_count = 0;
  planned_patches.reserve(candidates.size());
  AmdGpuCodeObject code_object(bytes.data(), bytes.size());
  const uint64_t original_text_size =
      code_object.text_sections().size() == 1 ? code_object.text_sections().front()->size() : 0;
  DbiPatchPlacementPlanner placement_planner(arch, original_text_size);
  MoiSpillManagers spill_managers;
  for (const ConSanMoiCandidate *candidate_ptr : candidates) {
    const ConSanMoiCandidate &candidate = *candidate_ptr;
    const auto resources = resolve_moi_scratch(result, candidate, options, access_scratch_count);
    if (!resources) {
      const ConSanCandidateResourcePlan *plan = resource_plan_for_candidate(result, candidate);
      if (plan != nullptr && plan->reason == ConSanRegisterPlanReason::ForbiddenOverlap &&
          options.scratch_vgpr) {
        std::vector<std::string> overlap_errors;
        const bool rejected = reject_candidate_scratch_range_overlap(
            candidate, *options.scratch_vgpr, access_scratch_count, overlap_errors);
        if (!rejected)
          overlap_errors.emplace_back("ConSan MOI probe scratch VGPRs overlap a forbidden range");
        result.warnings.insert(result.warnings.end(), overlap_errors.begin(), overlap_errors.end());
      }
      continue;
    }
    std::optional<MoiPrivateEpochLayout> private_layout;
    if (options.automatic_moi_private_epoch) {
      private_layout = build_moi_private_epoch_layout(result, *resources, result.warnings);
      if (!private_layout)
        continue;
    }
    std::optional<VgprSpillSequence> spill;
    if (resources->source == ConSanRegisterAllocationSource::SpillRequired) {
      spill = build_moi_spill_sequence(
          result, *resources, spill_managers, arch, result.warnings,
          private_layout ? std::optional<uint32_t>(private_layout->ephemeral_base) : std::nullopt);
      if (!spill)
        continue;
    }
    ConSanOptions candidate_options = options;
    candidate_options.scratch_vgpr = resources->base;
    const std::optional<std::vector<ConSanMoiAccessRange>> access_ranges =
        candidate_access_ranges(candidate);
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
        bytes, candidate, candidate_options, arch, 0, 0, layout.access_record_capacity,
        private_layout ? std::optional<uint32_t>(private_layout->owner_offset) : std::nullopt,
        private_layout ? std::optional<uint32_t>(private_layout->epoch_offset) : std::nullopt,
        candidate_errors);
    if (!words) {
      result.warnings.insert(result.warnings.end(), candidate_errors.begin(),
                             candidate_errors.end());
      continue;
    }

    const uint64_t spill_bytes =
        spill ? static_cast<uint64_t>((spill->save_words.size() + spill->restore_words.size()) *
                                      sizeof(uint32_t))
              : 0;
    const uint64_t patch_bytes =
        static_cast<uint64_t>(words->size() * sizeof(uint32_t)) + spill_bytes;
    const uint32_t available_padding =
        count_nop_padding(bytes, candidate.file_offset + candidate.size, arch);
    const uint64_t available_bytes =
        candidate.size + static_cast<uint64_t>(available_padding) * sizeof(uint32_t);
    DbiPatchPlacementRequest placement_request;
    placement_request.anchor_offset = candidate.text_offset;
    placement_request.original_size = candidate.size;
    placement_request.body_size = patch_bytes;
    placement_request.inline_capacity = spill ? 0u : available_bytes;
    std::string placement_error;
    const auto placement = placement_planner.plan(placement_request, &placement_error);
    if (!placement) {
      result.warnings.emplace_back("ConSan MOI first-light probe skipped load/store site: " +
                                   placement_error);
      continue;
    }

    PlannedAccessRecordPatch planned;
    planned.candidate = candidate_ptr;
    planned.placement = *placement;
    planned.record_index = planned_record_count;
    planned.record_count = candidate_record_count;
    planned.resources = *resources;
    planned.spill = spill;
    planned.private_layout = private_layout;
    planned.required_private_bytes = std::max(private_layout ? private_layout->ephemeral_base : 0u,
                                              spill ? spill->total_private_bytes : 0u);
    planned_patches.push_back(std::move(planned));
    planned_record_count += candidate_record_count;
    if (planned_patches.size() == options.max_patches)
      break;
  }

  if (planned_patches.empty())
    return;

  MoiDescriptorVgprRequirements descriptor_requirements;
  MoiDescriptorSgprRequirements scalar_requirements;
  MoiDescriptorPrivateRequirements private_requirements;
  for (const PlannedAccessRecordPatch &planned_patch : planned_patches) {
    note_descriptor_requirements(descriptor_requirements, planned_patch.resources);
    note_moi_sgpr_requirements(scalar_requirements, planned_patch.resources, options);
    if (planned_patch.required_private_bytes != 0) {
      for (uint64_t descriptor_offset : planned_patch.resources.owner_descriptor_file_offsets) {
        auto [it, inserted] =
            private_requirements.emplace(descriptor_offset, planned_patch.required_private_bytes);
        if (!inserted)
          it->second = std::max(it->second, planned_patch.required_private_bytes);
      }
    }
  }

  const bool uses_appended_cave =
      std::ranges::any_of(planned_patches, [](const PlannedAccessRecordPatch &patch) {
        return patch.placement.kind == DbiPatchPlacementKind::AppendedCave;
      });
  if (uses_appended_cave) {
    CodeObjectPatcher patcher(code_object);
    if (!apply_descriptor_requirements(patcher, code_object, bytes, result, descriptor_requirements,
                                       arch, result.errors)) {
      return;
    }
    if (!apply_spill_descriptor_requirements(patcher, code_object, bytes, result,
                                             private_requirements, result.errors)) {
      return;
    }
    if (!apply_sgpr_descriptor_requirements(patcher, result, code_object, scalar_requirements, arch,
                                            result.errors))
      return;
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
      ConSanOptions candidate_options = options;
      candidate_options.scratch_vgpr = planned_patch.resources.base;
      auto words = build_first_light_access_record_words(
          bytes, candidate, candidate_options, arch, planned_patch.record_index,
          planned_record_count, layout.access_record_capacity,
          planned_patch.private_layout
              ? std::optional<uint32_t>(planned_patch.private_layout->owner_offset)
              : std::nullopt,
          planned_patch.private_layout
              ? std::optional<uint32_t>(planned_patch.private_layout->epoch_offset)
              : std::nullopt,
          result.errors);
      if (!words)
        return;

      ConSanPatchInfo info;
      info.kind = planned_patch.placement.kind == DbiPatchPlacementKind::AppendedCave
                      ? ConSanPatchKind::TrampolineMoiAccessRecordStore
                      : ConSanPatchKind::InlineMoiAccessRecordStore;
      info.anchor_offset = candidate.text_offset;
      info.scratch_vgpr = planned_patch.resources.base;
      info.owner_descriptor_file_offsets = planned_patch.resources.owner_descriptor_file_offsets;
      if (planned_patch.private_layout) {
        info.persistent_owner_private_offset = planned_patch.private_layout->owner_offset;
        info.persistent_epoch_private_offset = planned_patch.private_layout->epoch_offset;
      }
      info.required_private_segment_size = planned_patch.required_private_bytes;
      if (planned_patch.spill) {
        info.spilled_vgpr_count = planned_patch.spill->vgpr_count;
      }

      if (planned_patch.placement.kind == DbiPatchPlacementKind::AppendedCave) {
        const uint64_t cave_text_offset = planned_patch.placement.body_offset;
        if (new_text.size() != cave_text_offset) {
          result.errors.emplace_back(
              "ConSan MOI first-light probe emitted a stale appended-cave mapping");
          return;
        }
        const auto fwd = compute_sopp_branch_simm16(planned_patch.placement.anchor_offset,
                                                    planned_patch.placement.body_offset);
        const auto ret = compute_sopp_branch_simm16(planned_patch.placement.return_branch_offset,
                                                    planned_patch.placement.return_target);
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

        std::vector<uint32_t> cave_words;
        if (planned_patch.spill) {
          cave_words.insert(cave_words.end(), planned_patch.spill->save_words.begin(),
                            planned_patch.spill->save_words.end());
        }
        cave_words.insert(cave_words.end(), words->begin(), words->end());
        if (planned_patch.spill) {
          cave_words.insert(cave_words.end(), planned_patch.spill->restore_words.begin(),
                            planned_patch.spill->restore_words.end());
        }
        if (cave_words.size() * sizeof(uint32_t) != planned_patch.placement.body_size) {
          result.errors.emplace_back(
              "ConSan MOI first-light probe body size changed after placement");
          return;
        }
        cave_words.push_back(build_s_branch(*ret, arch));
        append_words_bytes(new_text, cave_words);
        info.trampoline_offset = cave_text_offset;
        info.original_size = candidate.size;
        info.trampoline_size = static_cast<uint32_t>(cave_words.size() * sizeof(uint32_t));
      } else {
        const uint64_t patch_bytes = static_cast<uint64_t>(words->size() * sizeof(uint32_t));
        if (patch_bytes != planned_patch.placement.body_size) {
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
    if (!apply_spill_metadata_requirements(result.elf_bytes, result, private_requirements,
                                           result.errors)) {
      result.elf_bytes.clear();
      return;
    }
    result.patches.insert(result.patches.end(), patches.begin(), patches.end());
    result.modified = true;
    return;
  }

  result.elf_bytes.assign(bytes.begin(), bytes.end());
  for (const PlannedAccessRecordPatch &planned_patch : planned_patches) {
    const ConSanMoiCandidate &candidate = *planned_patch.candidate;
    ConSanOptions candidate_options = options;
    candidate_options.scratch_vgpr = planned_patch.resources.base;
    auto words = build_first_light_access_record_words(
        bytes, candidate, candidate_options, arch, planned_patch.record_index, planned_record_count,
        layout.access_record_capacity,
        planned_patch.private_layout
            ? std::optional<uint32_t>(planned_patch.private_layout->owner_offset)
            : std::nullopt,
        planned_patch.private_layout
            ? std::optional<uint32_t>(planned_patch.private_layout->epoch_offset)
            : std::nullopt,
        result.errors);
    if (!words) {
      result.elf_bytes.clear();
      return;
    }
    const uint64_t patch_bytes = static_cast<uint64_t>(words->size() * sizeof(uint32_t));
    if (patch_bytes != planned_patch.placement.body_size) {
      result.errors.emplace_back("ConSan MOI first-light probe final patch size changed");
      result.elf_bytes.clear();
      return;
    }
    std::memcpy(result.elf_bytes.data() + candidate.file_offset, words->data(),
                static_cast<size_t>(patch_bytes));
  }

  if (!apply_descriptor_requirements(result.elf_bytes, result, descriptor_requirements, arch,
                                     result.errors)) {
    result.elf_bytes.clear();
    return;
  }
  if (!apply_sgpr_descriptor_requirements(result.elf_bytes, result, scalar_requirements, arch,
                                          result.errors)) {
    result.elf_bytes.clear();
    return;
  }

  for (const PlannedAccessRecordPatch &planned_patch : planned_patches) {
    const ConSanMoiCandidate &candidate = *planned_patch.candidate;
    ConSanPatchInfo info;
    info.kind = ConSanPatchKind::InlineMoiAccessRecordStore;
    info.anchor_offset = candidate.text_offset;
    info.trampoline_offset = candidate.text_offset + candidate.size;
    info.original_size = static_cast<uint32_t>(planned_patch.placement.body_size);
    info.trampoline_size = 0;
    info.scratch_vgpr = planned_patch.resources.base;
    info.owner_descriptor_file_offsets = planned_patch.resources.owner_descriptor_file_offsets;
    if (planned_patch.private_layout) {
      info.persistent_owner_private_offset = planned_patch.private_layout->owner_offset;
      info.persistent_epoch_private_offset = planned_patch.private_layout->epoch_offset;
    }
    info.required_private_segment_size = planned_patch.required_private_bytes;
    result.patches.push_back(info);
  }

  result.modified = true;
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_cdna4_identity_prologue_words(
    uint64_t prologue_text_offset, uint64_t original_entry_text_offset, const KD &descriptor,
    const ConSanOptions &options, std::vector<std::string> &errors) {
  if (!options.moi_owner_vgpr || !options.moi_epoch_vgpr || !options.moi_identity_sgpr ||
      !options.moi_workgroup_vgprs[0] || !options.moi_workgroup_vgprs[1] ||
      !options.moi_workgroup_vgprs[2]) {
    errors.emplace_back("ConSan MOI CDNA4 identity prologue has incomplete register state");
    return std::nullopt;
  }

  const uint16_t owner_vgpr = *options.moi_owner_vgpr;
  const uint16_t epoch_vgpr = *options.moi_epoch_vgpr;
  const uint16_t size_x_sgpr = *options.moi_identity_sgpr;
  const uint16_t size_y_sgpr = static_cast<uint16_t>(size_x_sgpr + 1u);
  const auto transaction = build_cdna4_identity_abi_transaction(descriptor, errors);
  if (!transaction)
    return std::nullopt;

  // AQL kernel-dispatch packets store workgroup_size_x/y as adjacent uint16_t
  // fields at byte offset 4. One scalar dword load obtains both values.
  const auto load_group_xy = build_s_load_dword(size_x_sgpr, transaction->dispatch_sgpr,
                                                /*byte_offset=*/4u, ROCJITSU_CODE_ARCH_CDNA4);
  const auto wait_group_xy = build_s_wait_lds0(ROCJITSU_CODE_ARCH_CDNA4);
  if (!load_group_xy || !wait_group_xy) {
    errors.emplace_back("ConSan MOI CDNA4 identity prologue could not load dispatch dimensions");
    return std::nullopt;
  }

  std::vector<uint32_t> words;
  words.reserve(40u + transaction->guest_restore_moves.size());
  words.insert(words.end(), load_group_xy->begin(), load_group_xy->end());

  // The patched descriptor requests all three coordinates. Snapshot them
  // before restoring the exact original guest ABI.
  for (uint32_t dimension = 0; dimension < 3; ++dimension) {
    const uint16_t destination = *options.moi_workgroup_vgprs[dimension];
    words.push_back(build_v_mov_b32_e32(destination, transaction->workgroup_sources[dimension],
                                        ROCJITSU_CODE_ARCH_CDNA4));
  }
  words.push_back(*wait_group_xy);
  constexpr uint16_t kInline16 = kScalarPositiveInlineBase + 16u;
  words.push_back(build_s_lshr_b32(size_y_sgpr, size_x_sgpr, kInline16, ROCJITSU_CODE_ARCH_CDNA4));
  words.push_back(build_s_lshl_b32(size_x_sgpr, size_x_sgpr, kInline16, ROCJITSU_CODE_ARCH_CDNA4));
  words.push_back(build_s_lshr_b32(size_x_sgpr, size_x_sgpr, kInline16, ROCJITSU_CODE_ARCH_CDNA4));

  // VGPR0 is the AMDHSA packed workitem ID {00,z[9:0],y[9:0],x[9:0]}.
  // Compute x + size_x * (y + size_y * z), then divide by wave64.
  const auto z = build_v_lshrrev_b32_e32(owner_vgpr, scalar_positive_inline_u32(20), kWorkitemIdX,
                                         ROCJITSU_CODE_ARCH_CDNA4);
  const auto y = build_v_lshrrev_b32_e32(epoch_vgpr, scalar_positive_inline_u32(10), kWorkitemIdX,
                                         ROCJITSU_CODE_ARCH_CDNA4);
  const auto mask_y =
      build_v_and_b32_e32_literal(epoch_vgpr, 0x3ffu, epoch_vgpr, ROCJITSU_CODE_ARCH_CDNA4);
  const auto yz = build_v_mad_u32_u24(owner_vgpr, size_y_sgpr, owner_vgpr, epoch_vgpr,
                                      ROCJITSU_CODE_ARCH_CDNA4);
  const auto x =
      build_v_and_b32_e32_literal(epoch_vgpr, 0x3ffu, kWorkitemIdX, ROCJITSU_CODE_ARCH_CDNA4);
  const auto flat = build_v_mad_u32_u24(owner_vgpr, size_x_sgpr, owner_vgpr, epoch_vgpr,
                                        ROCJITSU_CODE_ARCH_CDNA4);
  const auto wave = build_v_lshrrev_b32_e32(owner_vgpr, scalar_positive_inline_u32(6), owner_vgpr,
                                            ROCJITSU_CODE_ARCH_CDNA4);
  if (!z || !y || !mask_y || !yz || !x || !flat || !wave) {
    errors.emplace_back("ConSan MOI CDNA4 identity prologue could not encode logical owner");
    return std::nullopt;
  }
  words.push_back(*z);
  words.push_back(*y);
  words.insert(words.end(), mask_y->begin(), mask_y->end());
  words.insert(words.end(), yz->begin(), yz->end());
  words.insert(words.end(), x->begin(), x->end());
  words.insert(words.end(), flat->begin(), flat->end());
  words.push_back(*wave);
  words.push_back(
      build_v_mov_b32_e32(epoch_vgpr, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_CDNA4));

  append_cdna4_identity_guest_abi_restore(words, *transaction);

  const uint64_t branch_pc =
      prologue_text_offset + static_cast<uint64_t>(words.size()) * sizeof(uint32_t);
  const auto branch = compute_sopp_branch_simm16(branch_pc, original_entry_text_offset);
  if (!branch) {
    errors.emplace_back("ConSan MOI CDNA4 identity prologue branch target is out of range");
    return std::nullopt;
  }
  words.push_back(build_s_branch(*branch, ROCJITSU_CODE_ARCH_CDNA4));
  return words;
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_owner_epoch_prologue_words(
    uint64_t prologue_text_offset, uint64_t original_entry_text_offset, uint16_t owner_vgpr,
    uint16_t epoch_vgpr, const MoiOwnerInput &owner_input, ConSanMoiOwnerSource owner_source,
    std::optional<uint16_t> owner_sgpr, const KD &descriptor, const ConSanOptions &options,
    rj_code_arch_t arch, std::vector<std::string> &errors) {
  if (arch == ROCJITSU_CODE_ARCH_CDNA4 && owner_source == ConSanMoiOwnerSource::WorkitemId)
    return build_cdna4_identity_prologue_words(prologue_text_offset, original_entry_text_offset,
                                               descriptor, options, errors);

  std::vector<uint32_t> words;
  words.reserve(4);
  switch (owner_source) {
  case ConSanMoiOwnerSource::WorkitemId: {
    if (!append_moi_owner_input(words, owner_vgpr, owner_input, arch, errors)) {
      errors.emplace_back("ConSan MOI owner/epoch prologue could not encode stable owner init");
      return std::nullopt;
    }
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

[[nodiscard]] std::optional<std::vector<uint32_t>> build_private_epoch_prologue_words(
    uint64_t prologue_text_offset, uint64_t original_entry_text_offset, uint16_t scratch_vgpr,
    uint32_t epoch_offset, uint32_t owner_offset, const MoiOwnerInput &owner_input,
    ConSanMoiOwnerSource owner_source, std::optional<uint16_t> owner_sgpr,
    const VgprSpillSequence &spill, rj_code_arch_t arch, std::vector<std::string> &errors) {
  const auto epoch_store = build_moi_private_store_b32(scratch_vgpr, epoch_offset, arch);
  const auto owner_store = build_moi_private_store_b32(scratch_vgpr, owner_offset, arch);
  const auto wait_store = arch == ROCJITSU_CODE_ARCH_CDNA4 ? build_cdna4_s_wait_vmcnt0(arch)
                                                           : build_s_wait_storecnt0(arch);
  if (!epoch_store || !owner_store || !wait_store) {
    errors.emplace_back("ConSan MOI private-state prologue could not encode initialization");
    return std::nullopt;
  }

  std::vector<uint32_t> words;
  words.reserve(spill.save_words.size() + spill.restore_words.size() + 10u);
  words.insert(words.end(), spill.save_words.begin(), spill.save_words.end());
  if (owner_source == ConSanMoiOwnerSource::WorkitemId) {
    if (!append_moi_owner_input(words, scratch_vgpr, owner_input, arch, errors))
      return std::nullopt;
  } else {
    if (!owner_sgpr) {
      errors.emplace_back("ConSan MOI private owner hw_id source has no temporary SGPR");
      return std::nullopt;
    }
    const auto hwreg = build_hwreg_imm(kGfx12HwRegHwId1, /*offset=*/0, kGfx12HwIdOwnerBits);
    const auto get_hw_id = hwreg ? build_s_getreg_b32(*owner_sgpr, *hwreg, arch) : std::nullopt;
    if (!get_hw_id) {
      errors.emplace_back("ConSan MOI private-state prologue could not encode HW_ID1");
      return std::nullopt;
    }
    words.push_back(*get_hw_id);
    words.push_back(build_v_mov_b32_e32(scratch_vgpr, *owner_sgpr, arch));
  }
  words.insert(words.end(), owner_store->begin(), owner_store->end());
  words.push_back(build_v_mov_b32_e32(scratch_vgpr, scalar_positive_inline_u32(0), arch));
  words.insert(words.end(), epoch_store->begin(), epoch_store->end());
  words.push_back(*wait_store);
  words.insert(words.end(), spill.restore_words.begin(), spill.restore_words.end());

  const uint64_t branch_pc =
      prologue_text_offset + static_cast<uint64_t>(words.size()) * sizeof(uint32_t);
  const auto branch = compute_sopp_branch_simm16(branch_pc, original_entry_text_offset);
  if (!branch) {
    errors.emplace_back("ConSan MOI private-epoch prologue branch target is out of range");
    return std::nullopt;
  }
  words.push_back(build_s_branch(*branch, arch));
  return words;
}

[[nodiscard]] std::optional<std::vector<uint32_t>>
build_cdna4_private_epoch_identity_prologue_words(uint64_t prologue_text_offset,
                                                  uint64_t original_entry_text_offset,
                                                  uint16_t scratch_vgpr, uint32_t epoch_offset,
                                                  uint32_t owner_offset, const KD &descriptor,
                                                  const ConSanOptions &options,
                                                  std::vector<std::string> &errors) {
  if (!options.moi_identity_sgpr || !options.moi_workgroup_sgprs[0] ||
      !options.moi_workgroup_sgprs[1] || !options.moi_workgroup_sgprs[2] ||
      scratch_vgpr == kWorkitemIdX || static_cast<uint32_t>(scratch_vgpr) + 2u > kMaxVgprs) {
    errors.emplace_back("ConSan MOI CDNA4 private identity has incomplete temporary state");
    return std::nullopt;
  }
  const uint16_t tmp_vgpr = static_cast<uint16_t>(scratch_vgpr + 1u);
  const uint16_t size_x_sgpr = *options.moi_identity_sgpr;
  const uint16_t size_y_sgpr = static_cast<uint16_t>(size_x_sgpr + 1u);
  const auto transaction = build_cdna4_identity_abi_transaction(descriptor, errors);
  if (!transaction)
    return std::nullopt;
  const auto load_group_xy = build_s_load_dword(size_x_sgpr, transaction->dispatch_sgpr,
                                                /*byte_offset=*/4u, ROCJITSU_CODE_ARCH_CDNA4);
  const auto wait_group_xy = build_s_wait_lds0(ROCJITSU_CODE_ARCH_CDNA4);
  const auto owner_store = build_cdna4_address_free_scratch_store_b32(scratch_vgpr, owner_offset,
                                                                      ROCJITSU_CODE_ARCH_CDNA4);
  const auto epoch_store = build_cdna4_address_free_scratch_store_b32(scratch_vgpr, epoch_offset,
                                                                      ROCJITSU_CODE_ARCH_CDNA4);
  const auto wait_store = build_cdna4_s_wait_vmcnt0(ROCJITSU_CODE_ARCH_CDNA4);
  if (!load_group_xy || !wait_group_xy || !owner_store || !epoch_store || !wait_store) {
    errors.emplace_back(
        "ConSan MOI CDNA4 private identity prologue could not encode initialization");
    return std::nullopt;
  }

  std::vector<uint32_t> words;
  // These temporaries execute before the guest kernel entry. CDNA4 has one
  // packed workitem-ID input in v0, which resource planning excludes from the
  // scratch window; all other VGPRs have no ABI live-in value. Saving and
  // reloading those undefined entry values is therefore both unnecessary and
  // actively unsafe: it introduces private loads into the entry transaction
  // before any guest definition. The owner/epoch stores below are the only
  // private operations the entry stub needs.
  words.reserve(40u);
  words.insert(words.end(), load_group_xy->begin(), load_group_xy->end());
  for (uint32_t dimension = 0; dimension < 3; ++dimension) {
    words.push_back(build_s_mov_b32(*options.moi_workgroup_sgprs[dimension],
                                    transaction->workgroup_sources[dimension],
                                    ROCJITSU_CODE_ARCH_CDNA4));
  }
  words.push_back(*wait_group_xy);
  constexpr uint16_t kInline16 = kScalarPositiveInlineBase + 16u;
  words.push_back(build_s_lshr_b32(size_y_sgpr, size_x_sgpr, kInline16, ROCJITSU_CODE_ARCH_CDNA4));
  words.push_back(build_s_lshl_b32(size_x_sgpr, size_x_sgpr, kInline16, ROCJITSU_CODE_ARCH_CDNA4));
  words.push_back(build_s_lshr_b32(size_x_sgpr, size_x_sgpr, kInline16, ROCJITSU_CODE_ARCH_CDNA4));

  const auto z = build_v_lshrrev_b32_e32(scratch_vgpr, scalar_positive_inline_u32(20), kWorkitemIdX,
                                         ROCJITSU_CODE_ARCH_CDNA4);
  const auto y = build_v_lshrrev_b32_e32(tmp_vgpr, scalar_positive_inline_u32(10), kWorkitemIdX,
                                         ROCJITSU_CODE_ARCH_CDNA4);
  const auto mask_y =
      build_v_and_b32_e32_literal(tmp_vgpr, 0x3ffu, tmp_vgpr, ROCJITSU_CODE_ARCH_CDNA4);
  const auto yz = build_v_mad_u32_u24(scratch_vgpr, size_y_sgpr, scratch_vgpr, tmp_vgpr,
                                      ROCJITSU_CODE_ARCH_CDNA4);
  const auto x =
      build_v_and_b32_e32_literal(tmp_vgpr, 0x3ffu, kWorkitemIdX, ROCJITSU_CODE_ARCH_CDNA4);
  const auto flat = build_v_mad_u32_u24(scratch_vgpr, size_x_sgpr, scratch_vgpr, tmp_vgpr,
                                        ROCJITSU_CODE_ARCH_CDNA4);
  const auto wave = build_v_lshrrev_b32_e32(scratch_vgpr, scalar_positive_inline_u32(6),
                                            scratch_vgpr, ROCJITSU_CODE_ARCH_CDNA4);
  if (!z || !y || !mask_y || !yz || !x || !flat || !wave) {
    errors.emplace_back("ConSan MOI CDNA4 private identity could not flatten packed workitem ID");
    return std::nullopt;
  }
  words.push_back(*z);
  words.push_back(*y);
  words.insert(words.end(), mask_y->begin(), mask_y->end());
  words.insert(words.end(), yz->begin(), yz->end());
  words.insert(words.end(), x->begin(), x->end());
  words.insert(words.end(), flat->begin(), flat->end());
  words.push_back(*wave);
  words.insert(words.end(), owner_store->begin(), owner_store->end());
  words.push_back(
      build_v_mov_b32_e32(scratch_vgpr, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_CDNA4));
  words.insert(words.end(), epoch_store->begin(), epoch_store->end());
  words.push_back(*wait_store);
  append_cdna4_identity_guest_abi_restore(words, *transaction);
  const uint64_t branch_pc =
      prologue_text_offset + static_cast<uint64_t>(words.size()) * sizeof(uint32_t);
  const auto branch = compute_sopp_branch_simm16(branch_pc, original_entry_text_offset);
  if (!branch) {
    errors.emplace_back("ConSan MOI CDNA4 private-epoch identity branch target is out of range");
    return std::nullopt;
  }
  words.push_back(build_s_branch(*branch, ROCJITSU_CODE_ARCH_CDNA4));
  return words;
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_cdna4_scalar_identity_prologue_words(
    uint64_t prologue_text_offset, uint64_t original_entry_text_offset, uint16_t scratch_vgpr,
    const KD &descriptor, const ConSanOptions &options, std::vector<std::string> &errors) {
  if (!options.moi_state_owner_sgpr || !options.moi_state_epoch_sgpr ||
      !options.moi_workgroup_sgprs[0] || !options.moi_workgroup_sgprs[1] ||
      !options.moi_workgroup_sgprs[2] || scratch_vgpr == kWorkitemIdX ||
      static_cast<uint32_t>(scratch_vgpr) + 2u > kMaxVgprs) {
    errors.emplace_back("ConSan MOI CDNA4 scalar identity has incomplete persistent state");
    return std::nullopt;
  }
  const uint16_t tmp_vgpr = static_cast<uint16_t>(scratch_vgpr + 1u);
  const uint16_t owner_sgpr = *options.moi_state_owner_sgpr;
  const uint16_t epoch_sgpr = *options.moi_state_epoch_sgpr;
  const auto transaction = build_cdna4_identity_abi_transaction(descriptor, errors);
  if (!transaction)
    return std::nullopt;
  const auto load_group_xy = build_s_load_dword(owner_sgpr, transaction->dispatch_sgpr,
                                                /*byte_offset=*/4u, ROCJITSU_CODE_ARCH_CDNA4);
  const auto wait_group_xy = build_s_wait_lds0(ROCJITSU_CODE_ARCH_CDNA4);
  if (!load_group_xy || !wait_group_xy) {
    errors.emplace_back("ConSan MOI CDNA4 scalar identity could not load dispatch dimensions");
    return std::nullopt;
  }

  std::vector<uint32_t> words;
  words.reserve(40u);
  words.insert(words.end(), load_group_xy->begin(), load_group_xy->end());
  for (uint32_t dimension = 0; dimension < 3; ++dimension) {
    words.push_back(build_s_mov_b32(*options.moi_workgroup_sgprs[dimension],
                                    transaction->workgroup_sources[dimension],
                                    ROCJITSU_CODE_ARCH_CDNA4));
  }
  words.push_back(*wait_group_xy);
  constexpr uint16_t kInline16 = kScalarPositiveInlineBase + 16u;
  words.push_back(build_s_lshr_b32(epoch_sgpr, owner_sgpr, kInline16, ROCJITSU_CODE_ARCH_CDNA4));
  words.push_back(build_s_lshl_b32(owner_sgpr, owner_sgpr, kInline16, ROCJITSU_CODE_ARCH_CDNA4));
  words.push_back(build_s_lshr_b32(owner_sgpr, owner_sgpr, kInline16, ROCJITSU_CODE_ARCH_CDNA4));

  const auto z = build_v_lshrrev_b32_e32(scratch_vgpr, scalar_positive_inline_u32(20), kWorkitemIdX,
                                         ROCJITSU_CODE_ARCH_CDNA4);
  const auto y = build_v_lshrrev_b32_e32(tmp_vgpr, scalar_positive_inline_u32(10), kWorkitemIdX,
                                         ROCJITSU_CODE_ARCH_CDNA4);
  const auto mask_y =
      build_v_and_b32_e32_literal(tmp_vgpr, 0x3ffu, tmp_vgpr, ROCJITSU_CODE_ARCH_CDNA4);
  const auto yz = build_v_mad_u32_u24(scratch_vgpr, epoch_sgpr, scratch_vgpr, tmp_vgpr,
                                      ROCJITSU_CODE_ARCH_CDNA4);
  const auto x =
      build_v_and_b32_e32_literal(tmp_vgpr, 0x3ffu, kWorkitemIdX, ROCJITSU_CODE_ARCH_CDNA4);
  const auto flat = build_v_mad_u32_u24(scratch_vgpr, owner_sgpr, scratch_vgpr, tmp_vgpr,
                                        ROCJITSU_CODE_ARCH_CDNA4);
  const auto wave = build_v_lshrrev_b32_e32(scratch_vgpr, scalar_positive_inline_u32(6),
                                            scratch_vgpr, ROCJITSU_CODE_ARCH_CDNA4);
  const auto read_owner =
      build_v_readfirstlane_b32(owner_sgpr, scratch_vgpr, ROCJITSU_CODE_ARCH_CDNA4);
  if (!z || !y || !mask_y || !yz || !x || !flat || !wave || !read_owner) {
    errors.emplace_back("ConSan MOI CDNA4 scalar identity could not encode logical owner");
    return std::nullopt;
  }
  words.push_back(*z);
  words.push_back(*y);
  words.insert(words.end(), mask_y->begin(), mask_y->end());
  words.insert(words.end(), yz->begin(), yz->end());
  words.insert(words.end(), x->begin(), x->end());
  words.insert(words.end(), flat->begin(), flat->end());
  words.push_back(*wave);
  words.push_back(*read_owner);
  words.push_back(
      build_s_mov_b32(epoch_sgpr, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_CDNA4));
  append_cdna4_identity_guest_abi_restore(words, *transaction);
  const uint64_t branch_pc =
      prologue_text_offset + static_cast<uint64_t>(words.size()) * sizeof(uint32_t);
  const auto branch = compute_sopp_branch_simm16(branch_pc, original_entry_text_offset);
  if (!branch) {
    errors.emplace_back("ConSan MOI CDNA4 scalar identity branch target is out of range");
    return std::nullopt;
  }
  words.push_back(build_s_branch(*branch, ROCJITSU_CODE_ARCH_CDNA4));
  return words;
}

[[nodiscard]] bool kernel_contains_patch_anchor(const ConSanKernelInfo &kernel,
                                                const ConSanPatchInfo &patch) {
  return kernel.has_text_range && patch.anchor_offset >= kernel.entry_text_offset &&
         patch.anchor_offset - kernel.entry_text_offset < kernel.code_size;
}

[[nodiscard]] bool kernel_owns_patch(const ConSanKernelInfo &kernel, const ConSanPatchInfo &patch) {
  if (!patch.owner_descriptor_file_offsets.empty()) {
    return std::ranges::find(patch.owner_descriptor_file_offsets, kernel.descriptor_file_offset) !=
           patch.owner_descriptor_file_offsets.end();
  }
  return kernel_contains_patch_anchor(kernel, patch);
}

[[nodiscard]] bool
grow_moi_kernel_descriptor_registers(CodeObjectPatcher &patcher, std::span<const uint8_t> image,
                                     const ConSanKernelInfo &kernel, uint32_t required_vgpr_count,
                                     uint32_t required_sgpr_count, bool enable_dispatch_ptr,
                                     rj_code_arch_t arch, std::vector<std::string> &errors) {
  if (kernel.descriptor_file_offset > image.size() ||
      sizeof(KD) > image.size() - kernel.descriptor_file_offset) {
    errors.emplace_back("ConSan MOI descriptor register growth exceeds ELF bytes");
    return false;
  }

  KD desc{};
  std::memcpy(&desc, image.data() + kernel.descriptor_file_offset, sizeof(desc));
  if (enable_dispatch_ptr && arch == ROCJITSU_CODE_ARCH_CDNA4) {
    const auto transaction = build_cdna4_identity_abi_transaction(desc, errors);
    if (!transaction)
      return false;
    required_sgpr_count = std::max(required_sgpr_count, transaction->patched_abi_count);
    AMDHSA_BITS_SET(desc.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X, 1u);
    AMDHSA_BITS_SET(desc.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y, 1u);
    AMDHSA_BITS_SET(desc.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z, 1u);
  }
  if (enable_dispatch_ptr && !moi_descriptor_enables_dispatch_ptr(desc)) {
    const uint32_t user_sgpr_count = moi_descriptor_user_sgpr_count(desc);
    if (user_sgpr_count > 29u) {
      errors.emplace_back("ConSan MOI cannot insert a dispatch pointer into the user-SGPR ABI");
      return false;
    }
    AMDHSA_BITS_SET(desc.kernel_code_properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR,
                    1u);
    AMDHSA_BITS_SET(desc.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT,
                    (user_sgpr_count + 2u));
  }
  if (!grow_moi_descriptor_vgpr_allocation(desc, required_vgpr_count, arch)) {
    errors.emplace_back("ConSan MOI could not grow descriptor VGPR allocation");
    return false;
  }
  if (required_sgpr_count != 0 &&
      !grow_moi_descriptor_sgpr_allocation(desc, required_sgpr_count, arch)) {
    errors.emplace_back("ConSan MOI could not grow descriptor SGPR allocation");
    return false;
  }
  if (!patcher.patch_kernel_descriptor(kernel.descriptor_file_offset,
                                       {reinterpret_cast<const uint8_t *>(&desc), sizeof(desc)})) {
    errors.emplace_back("ConSan MOI could not patch descriptor register allocation");
    return false;
  }
  return true;
}

void try_apply_private_epoch_prologue_patch(const ConSanOptions &options, rj_code_arch_t arch,
                                            ConSanResult &result) {
  if (!result.modified || result.elf_bytes.empty()) {
    result.warnings.emplace_back(
        "ConSan MOI private-epoch prologue skipped because no exact-shadow access probe was "
        "emitted");
    return;
  }

  std::span<const uint8_t> active_bytes(result.elf_bytes.data(), result.elf_bytes.size());
  AmdGpuCodeObject code_object(active_bytes.data(), active_bytes.size());
  CodeObjectPatcher patcher(code_object);
  const std::span<const uint8_t> old_text = patcher.text_bytes();
  if (old_text.empty()) {
    result.errors.emplace_back("ConSan MOI private-epoch prologue found no .text section");
    return;
  }

  struct PlannedPrivateEpochPrologue {
    const ConSanKernelInfo *kernel = nullptr;
    uint64_t active_descriptor_file_offset = 0;
    uint64_t active_entry_text_offset = 0;
    uint16_t scratch_vgpr = 0;
    MoiPrivateEpochLayout layout;
    std::optional<VgprSpillSequence> spill;
    KD descriptor{};
    MoiOwnerInput owner_input;
    uint32_t required_private_bytes = 0;
  };
  std::vector<PlannedPrivateEpochPrologue> planned;
  MoiDescriptorPrivateRequirements private_requirements;
  for (const ConSanKernelInfo &kernel : result.kernels) {
    const auto access_patch =
        std::ranges::find_if(result.patches, [&](const ConSanPatchInfo &patch) {
          const bool private_access =
              patch.kind == ConSanPatchKind::InlineMoiExactShadowStore ||
              patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore ||
              patch.kind == ConSanPatchKind::InlineMoiAccessRecordStore ||
              patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore ||
              patch.kind == ConSanPatchKind::InlineMoiSampledWatchpointStore ||
              patch.kind == ConSanPatchKind::TrampolineMoiSampledWatchpointStore ||
              patch.kind == ConSanPatchKind::TrampolineMoiInlineAtomicOrdering;
          return private_access && patch.scratch_vgpr && kernel_owns_patch(kernel, patch);
        });
    if (access_patch == result.patches.end())
      continue;

    const auto active_kernel =
        std::ranges::find_if(code_object.kernels(), [&](const AmdGpuKernelInfo &candidate) {
          return candidate.name == kernel.name;
        });
    if (active_kernel == code_object.kernels().end()) {
      result.warnings.emplace_back(
          "ConSan MOI private-epoch prologue could not resolve an active kernel descriptor");
      continue;
    }

    if (!access_patch->persistent_epoch_private_offset ||
        !access_patch->persistent_owner_private_offset)
      continue;
    MoiPrivateEpochLayout layout{.epoch_offset = *access_patch->persistent_epoch_private_offset,
                                 .owner_offset = *access_patch->persistent_owner_private_offset,
                                 .ephemeral_base =
                                     util::align_up(*access_patch->persistent_owner_private_offset +
                                                        SpillManager::kSlotBytes,
                                                    SpillManager::kDbiZoneAlignment)};

    std::optional<VgprSpillSequence> spill;
    const bool cdna4_packed_identity = arch == ROCJITSU_CODE_ARCH_CDNA4 &&
                                       options.moi_owner_source == ConSanMoiOwnerSource::WorkitemId;
    if (!cdna4_packed_identity) {
      SpillManager manager(layout.ephemeral_base, kMaxAddressFreeScratchPrivateBytes);
      spill = build_vgpr_spill_sequence(manager, *access_patch->scratch_vgpr,
                                        /*vgpr_count=*/1u, arch);
      if (!spill) {
        result.warnings.emplace_back(
            "ConSan MOI private-epoch prologue could not preserve its temporary VGPR");
        continue;
      }
    }
    uint32_t required_private_bytes = layout.ephemeral_base;
    if (spill)
      required_private_bytes = std::max(required_private_bytes, spill->total_private_bytes);
    for (const ConSanPatchInfo &patch : result.patches) {
      if (kernel_owns_patch(kernel, patch))
        required_private_bytes =
            std::max(required_private_bytes, patch.required_private_segment_size);
    }
    private_requirements[kernel.descriptor_file_offset] = required_private_bytes;
    KD descriptor{};
    std::memcpy(&descriptor, active_bytes.data() + active_kernel->descriptor_file_offset,
                sizeof(descriptor));
    std::vector<std::string> owner_errors;
    const auto owner_input = moi_descriptor_owner_input(
        active_bytes, active_kernel->descriptor_file_offset, arch, owner_errors);
    if (!owner_input) {
      result.warnings.insert(result.warnings.end(), owner_errors.begin(), owner_errors.end());
      continue;
    }
    planned.push_back({&kernel, active_kernel->descriptor_file_offset,
                       active_kernel->entry_text_offset, *access_patch->scratch_vgpr, layout,
                       std::move(spill), descriptor, *owner_input, required_private_bytes});
  }

  if (planned.empty()) {
    result.warnings.emplace_back("ConSan MOI private-epoch prologue found no patchable kernels");
    return;
  }
  if (arch == ROCJITSU_CODE_ARCH_CDNA4 &&
      options.moi_owner_source == ConSanMoiOwnerSource::WorkitemId) {
    if (!options.moi_identity_sgpr) {
      result.errors.emplace_back("ConSan MOI CDNA4 private owner has no identity SGPR window");
      return;
    }
    for (const PlannedPrivateEpochPrologue &item : planned) {
      ConSanKernelInfo active = *item.kernel;
      active.descriptor_file_offset = item.active_descriptor_file_offset;
      const uint32_t required_vgpr_count = static_cast<uint32_t>(item.scratch_vgpr) + 2u;
      const uint32_t required_sgpr_count =
          std::max<uint32_t>(*options.moi_workgroup_sgprs[2] + 1u,
                             moi_descriptor_abi_sgpr_count(item.descriptor, arch) +
                                 moi_descriptor_dispatch_insertion_width(item.descriptor));
      if (!grow_moi_kernel_descriptor_registers(patcher, active_bytes, active, required_vgpr_count,
                                                required_sgpr_count,
                                                /*enable_dispatch_ptr=*/true, arch, result.errors))
        return;
    }
  }
  if (!apply_spill_descriptor_requirements(patcher, code_object, active_bytes, result,
                                           private_requirements, result.errors))
    return;

  std::vector<uint8_t> new_text(old_text.begin(), old_text.end());
  std::vector<ConSanPatchInfo> patches;
  patches.reserve(planned.size());
  for (const PlannedPrivateEpochPrologue &item : planned) {
    append_nop_padding_to_alignment(new_text, kAmdhsaKernelEntryAlignment, arch);
    const uint64_t prologue_text_offset = static_cast<uint64_t>(new_text.size());
    std::optional<std::vector<uint32_t>> words;
    if (arch == ROCJITSU_CODE_ARCH_CDNA4 &&
        options.moi_owner_source == ConSanMoiOwnerSource::WorkitemId) {
      words = build_cdna4_private_epoch_identity_prologue_words(
          prologue_text_offset, item.active_entry_text_offset, item.scratch_vgpr,
          item.layout.epoch_offset, item.layout.owner_offset, item.descriptor, options,
          result.errors);
    } else {
      if (!item.spill) {
        result.errors.emplace_back("ConSan MOI private-epoch prologue has no temporary spill");
        return;
      }
      words = build_private_epoch_prologue_words(
          prologue_text_offset, item.active_entry_text_offset, item.scratch_vgpr,
          item.layout.epoch_offset, item.layout.owner_offset, item.owner_input,
          options.moi_owner_source, options.moi_owner_sgpr, *item.spill, arch, result.errors);
    }
    if (!words)
      return;
    const bool has_kernarg_preload =
        AMDHSA_BITS_GET(item.descriptor.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_LENGTH) != 0;
    std::optional<std::vector<uint32_t>> preload_words;
    if (has_kernarg_preload) {
      if (words->size() * sizeof(uint32_t) > kAmdhsaKernelEntryAlignment) {
        result.errors.emplace_back(
            "ConSan MOI private-epoch prologue exceeds the kernarg-preload entry window");
        return;
      }
      if (arch == ROCJITSU_CODE_ARCH_CDNA4 &&
          options.moi_owner_source == ConSanMoiOwnerSource::WorkitemId) {
        preload_words = build_cdna4_private_epoch_identity_prologue_words(
            prologue_text_offset + kAmdhsaKernelEntryAlignment,
            item.active_entry_text_offset + kAmdhsaKernelEntryAlignment, item.scratch_vgpr,
            item.layout.epoch_offset, item.layout.owner_offset, item.descriptor, options,
            result.errors);
      } else {
        if (!item.spill) {
          result.errors.emplace_back("ConSan MOI private-epoch preload prologue has no spill");
          return;
        }
        preload_words = build_private_epoch_prologue_words(
            prologue_text_offset + kAmdhsaKernelEntryAlignment,
            item.active_entry_text_offset + kAmdhsaKernelEntryAlignment, item.scratch_vgpr,
            item.layout.epoch_offset, item.layout.owner_offset, item.owner_input,
            options.moi_owner_source, options.moi_owner_sgpr, *item.spill, arch, result.errors);
      }
      if (!preload_words)
        return;
    }
    if (!patcher.redirect_kernel_entry(item.active_descriptor_file_offset,
                                       item.active_entry_text_offset, prologue_text_offset)) {
      result.errors.emplace_back(
          "ConSan MOI private-epoch prologue could not redirect kernel entry");
      return;
    }
    append_words_bytes(new_text, *words);
    if (preload_words) {
      while (new_text.size() < prologue_text_offset + kAmdhsaKernelEntryAlignment)
        append_word_bytes(new_text, build_s_nop(0, arch));
      append_words_bytes(new_text, *preload_words);
    }

    ConSanPatchInfo info;
    info.kind = ConSanPatchKind::KernelEntryMoiPrivateEpochPrologue;
    info.anchor_offset = item.kernel->entry_text_offset;
    info.trampoline_offset = prologue_text_offset;
    info.trampoline_size = static_cast<uint32_t>(new_text.size() - prologue_text_offset);
    info.scratch_vgpr = item.scratch_vgpr;
    info.persistent_owner_private_offset = item.layout.owner_offset;
    info.persistent_epoch_private_offset = item.layout.epoch_offset;
    info.spilled_vgpr_count = item.spill ? item.spill->vgpr_count : 0u;
    info.required_private_segment_size = item.required_private_bytes;
    patches.push_back(info);
  }

  if (!patcher.replace_text(new_text)) {
    result.errors.emplace_back("ConSan MOI private-epoch prologue could not grow .text");
    return;
  }
  result.elf_bytes = patcher.emit();
  if (!apply_spill_metadata_requirements(result.elf_bytes, result, private_requirements,
                                         result.errors)) {
    result.elf_bytes.clear();
    return;
  }
  result.patches.insert(result.patches.end(), patches.begin(), patches.end());
  result.modified = true;
}

void try_apply_scalar_identity_prologue_patch(const ConSanOptions &options, rj_code_arch_t arch,
                                              ConSanResult &result) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA4 || !options.automatic_moi_scalar_identity) {
    result.errors.emplace_back("ConSan MOI scalar identity prologue requires CDNA4 scalar state");
    return;
  }
  if (!result.modified || result.elf_bytes.empty()) {
    result.warnings.emplace_back("ConSan MOI scalar identity prologue found no emitted probes");
    return;
  }
  std::span<const uint8_t> active_bytes(result.elf_bytes.data(), result.elf_bytes.size());
  AmdGpuCodeObject code_object(active_bytes.data(), active_bytes.size());
  CodeObjectPatcher patcher(code_object);
  const std::span<const uint8_t> old_text = patcher.text_bytes();
  if (old_text.empty()) {
    result.errors.emplace_back("ConSan MOI scalar identity prologue found no .text section");
    return;
  }

  struct PlannedScalarIdentityPrologue {
    const ConSanKernelInfo *kernel = nullptr;
    uint64_t active_descriptor_file_offset = 0;
    uint64_t active_entry_text_offset = 0;
    uint16_t scratch_vgpr = 0;
    KD descriptor{};
  };
  std::vector<PlannedScalarIdentityPrologue> planned;
  for (const ConSanKernelInfo &kernel : result.kernels) {
    const auto access_patch =
        std::ranges::find_if(result.patches, [&](const ConSanPatchInfo &patch) {
          const bool identity_access =
              patch.kind == ConSanPatchKind::InlineMoiExactShadowStore ||
              patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore ||
              patch.kind == ConSanPatchKind::InlineMoiAccessRecordStore ||
              patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore ||
              patch.kind == ConSanPatchKind::InlineMoiSampledWatchpointStore ||
              patch.kind == ConSanPatchKind::TrampolineMoiSampledWatchpointStore ||
              patch.kind == ConSanPatchKind::TrampolineMoiInlineAtomicOrdering ||
              patch.kind == ConSanPatchKind::TrampolineMoiBarrierRecord ||
              patch.kind == ConSanPatchKind::TrampolineMoiAtomicRecord;
          return identity_access && patch.scratch_vgpr && kernel_owns_patch(kernel, patch);
        });
    if (access_patch == result.patches.end())
      continue;
    const auto active_kernel =
        std::ranges::find_if(code_object.kernels(), [&](const AmdGpuKernelInfo &candidate) {
          return candidate.name == kernel.name;
        });
    if (active_kernel == code_object.kernels().end()) {
      result.errors.emplace_back("ConSan MOI scalar identity could not resolve a kernel");
      return;
    }
    KD descriptor{};
    std::memcpy(&descriptor, active_bytes.data() + active_kernel->descriptor_file_offset,
                sizeof(descriptor));
    planned.push_back({&kernel, active_kernel->descriptor_file_offset,
                       active_kernel->entry_text_offset, *access_patch->scratch_vgpr, descriptor});
  }
  if (planned.empty()) {
    result.warnings.emplace_back("ConSan MOI scalar identity found no patchable kernels");
    return;
  }
  if (!options.moi_state_owner_sgpr || !options.moi_workgroup_sgprs[2]) {
    result.errors.emplace_back("ConSan MOI scalar identity has no persistent SGPR window");
    return;
  }
  for (const PlannedScalarIdentityPrologue &item : planned) {
    ConSanKernelInfo active = *item.kernel;
    active.descriptor_file_offset = item.active_descriptor_file_offset;
    const uint32_t required_vgpr_count = static_cast<uint32_t>(item.scratch_vgpr) + 2u;
    const uint32_t encoded_accum_offset = AMDHSA_BITS_GET(
        item.descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET);
    if (item.scratch_vgpr == 0 ||
        (encoded_accum_offset != 0 && required_vgpr_count > (encoded_accum_offset + 1u) * 4u)) {
      result.errors.emplace_back(
          "ConSan MOI scalar identity has no two ordinary entry temporary VGPRs");
      return;
    }
    const uint32_t required_sgpr_count =
        std::max<uint32_t>(*options.moi_workgroup_sgprs[2] + 1u,
                           moi_descriptor_abi_sgpr_count(item.descriptor, arch) +
                               moi_descriptor_dispatch_insertion_width(item.descriptor));
    if (!grow_moi_kernel_descriptor_registers(patcher, active_bytes, active, required_vgpr_count,
                                              required_sgpr_count,
                                              /*enable_dispatch_ptr=*/true, arch, result.errors))
      return;
  }

  std::vector<uint8_t> new_text(old_text.begin(), old_text.end());
  std::vector<ConSanPatchInfo> patches;
  for (const PlannedScalarIdentityPrologue &item : planned) {
    append_nop_padding_to_alignment(new_text, kAmdhsaKernelEntryAlignment, arch);
    const uint64_t prologue_text_offset = static_cast<uint64_t>(new_text.size());
    auto words = build_cdna4_scalar_identity_prologue_words(
        prologue_text_offset, item.active_entry_text_offset, item.scratch_vgpr, item.descriptor,
        options, result.errors);
    if (!words)
      return;
    const bool has_kernarg_preload =
        AMDHSA_BITS_GET(item.descriptor.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_LENGTH) != 0;
    std::optional<std::vector<uint32_t>> preload_words;
    if (has_kernarg_preload) {
      if (words->size() * sizeof(uint32_t) > kAmdhsaKernelEntryAlignment) {
        result.errors.emplace_back(
            "ConSan MOI scalar identity prologue exceeds the kernarg-preload entry window");
        return;
      }
      preload_words = build_cdna4_scalar_identity_prologue_words(
          prologue_text_offset + kAmdhsaKernelEntryAlignment,
          item.active_entry_text_offset + kAmdhsaKernelEntryAlignment, item.scratch_vgpr,
          item.descriptor, options, result.errors);
      if (!preload_words)
        return;
    }
    if (!patcher.redirect_kernel_entry(item.active_descriptor_file_offset,
                                       item.active_entry_text_offset, prologue_text_offset)) {
      result.errors.emplace_back("ConSan MOI scalar identity could not redirect kernel entry");
      return;
    }
    append_words_bytes(new_text, *words);
    if (preload_words) {
      while (new_text.size() < prologue_text_offset + kAmdhsaKernelEntryAlignment)
        append_word_bytes(new_text, build_s_nop(0, arch));
      append_words_bytes(new_text, *preload_words);
    }
    ConSanPatchInfo info;
    info.kind = ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue;
    info.anchor_offset = item.kernel->entry_text_offset;
    info.trampoline_offset = prologue_text_offset;
    info.trampoline_size = static_cast<uint32_t>(new_text.size() - prologue_text_offset);
    info.scratch_vgpr = item.scratch_vgpr;
    patches.push_back(info);
  }
  if (!patcher.replace_text(new_text)) {
    result.errors.emplace_back("ConSan MOI scalar identity could not grow .text");
    return;
  }
  result.elf_bytes = patcher.emit();
  result.patches.insert(result.patches.end(), patches.begin(), patches.end());
  result.modified = true;
}

void try_apply_owner_epoch_prologue_patch(std::span<const uint8_t> bytes,
                                          const ConSanOptions &options, rj_code_arch_t arch,
                                          ConSanResult &result) {
  if (!options.moi_init_owner_epoch)
    return;
  if (arch != ROCJITSU_CODE_ARCH_RDNA4 && arch != ROCJITSU_CODE_ARCH_CDNA4) {
    result.warnings.emplace_back(
        "ConSan MOI owner/epoch prologue currently supports only RDNA4 and CDNA4");
    return;
  }
  if (options.automatic_moi_private_epoch) {
    try_apply_private_epoch_prologue_patch(options, arch, result);
    return;
  }
  if (options.automatic_moi_scalar_identity) {
    try_apply_scalar_identity_prologue_patch(options, arch, result);
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
  uint32_t required_vgpr_count =
      std::max<uint32_t>(*options.moi_owner_vgpr, *options.moi_epoch_vgpr) + 1u;
  for (const auto workgroup_vgpr : options.moi_workgroup_vgprs) {
    if (workgroup_vgpr)
      required_vgpr_count = std::max<uint32_t>(required_vgpr_count, *workgroup_vgpr + 1u);
  }
  uint32_t required_sgpr_count = options.moi_owner_source == ConSanMoiOwnerSource::HwId
                                     ? static_cast<uint32_t>(*options.moi_owner_sgpr) + 1u
                                     : 0u;
  if (options.moi_identity_sgpr)
    required_sgpr_count = std::max<uint32_t>(required_sgpr_count, *options.moi_identity_sgpr + 2u);
  std::vector<ConSanKernelInfo> target_kernels;
  target_kernels.reserve(result.kernels.size());
  for (const ConSanKernelInfo &kernel : result.kernels) {
    if (!kernel.has_text_range)
      continue;
    if (options.automatic_moi_persistent_vgprs &&
        std::ranges::none_of(result.patches, [&](const ConSanPatchInfo &patch) {
          return kernel_owns_patch(kernel, patch);
        })) {
      continue;
    }
    const auto active_kernel =
        std::ranges::find_if(code_object.kernels(), [&](const AmdGpuKernelInfo &candidate) {
          return candidate.name == kernel.name;
        });
    if (active_kernel == code_object.kernels().end()) {
      result.warnings.emplace_back(
          "ConSan MOI owner/epoch prologue could not resolve an active kernel descriptor");
      continue;
    }
    ConSanKernelInfo active = kernel;
    active.descriptor_file_offset = active_kernel->descriptor_file_offset;
    active.entry_text_offset = active_kernel->entry_text_offset;
    active.text_file_offset = active_kernel->text_file_offset;
    target_kernels.push_back(std::move(active));
  }
  for (const ConSanKernelInfo &kernel : target_kernels) {
    if (kernel.descriptor_file_offset > active_bytes.size() ||
        sizeof(KD) > active_bytes.size() - kernel.descriptor_file_offset) {
      result.errors.emplace_back("ConSan MOI identity descriptor exceeds ELF bytes");
      return;
    }
    KD descriptor{};
    std::memcpy(&descriptor, active_bytes.data() + kernel.descriptor_file_offset,
                sizeof(descriptor));
    uint32_t kernel_required_sgprs = required_sgpr_count;
    if (arch == ROCJITSU_CODE_ARCH_CDNA4 && options.automatic_moi_identity_vgprs) {
      kernel_required_sgprs = std::max<uint32_t>(
          kernel_required_sgprs, moi_descriptor_abi_sgpr_count(descriptor, arch) +
                                     moi_descriptor_dispatch_insertion_width(descriptor));
    }
    if (!grow_moi_kernel_descriptor_registers(
            patcher, active_bytes, kernel, required_vgpr_count, kernel_required_sgprs,
            options.automatic_moi_identity_vgprs, arch, result.errors)) {
      return;
    }
  }

  std::vector<uint8_t> new_text(old_text.begin(), old_text.end());
  std::vector<ConSanPatchInfo> patches;
  for (const ConSanKernelInfo &kernel : target_kernels) {

    append_nop_padding_to_alignment(new_text, kAmdhsaKernelEntryAlignment, arch);
    const uint64_t prologue_text_offset = static_cast<uint64_t>(new_text.size());
    const auto owner_input = moi_kernel_owner_input(active_bytes, kernel, arch, result.errors);
    if (!owner_input)
      return;
    KD descriptor{};
    std::memcpy(&descriptor, active_bytes.data() + kernel.descriptor_file_offset,
                sizeof(descriptor));
    auto words = build_owner_epoch_prologue_words(
        prologue_text_offset, kernel.entry_text_offset, *options.moi_owner_vgpr,
        *options.moi_epoch_vgpr, *owner_input, options.moi_owner_source, options.moi_owner_sgpr,
        descriptor, options, arch, result.errors);
    if (!words)
      return;
    const bool has_kernarg_preload =
        AMDHSA_BITS_GET(descriptor.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_LENGTH) != 0;
    std::optional<std::vector<uint32_t>> preload_words;
    if (has_kernarg_preload) {
      const uint64_t prologue_bytes = words->size() * sizeof(uint32_t);
      if (prologue_bytes > kAmdhsaKernelEntryAlignment) {
        result.errors.emplace_back(
            "ConSan MOI owner/epoch prologue exceeds the kernarg-preload entry window");
        return;
      }
      preload_words = build_owner_epoch_prologue_words(
          prologue_text_offset + kAmdhsaKernelEntryAlignment,
          kernel.entry_text_offset + kAmdhsaKernelEntryAlignment, *options.moi_owner_vgpr,
          *options.moi_epoch_vgpr, *owner_input, options.moi_owner_source, options.moi_owner_sgpr,
          descriptor, options, arch, result.errors);
      if (!preload_words)
        return;
    }
    if (!patcher.redirect_kernel_entry(kernel.descriptor_file_offset, kernel.entry_text_offset,
                                       prologue_text_offset)) {
      result.errors.emplace_back("ConSan MOI owner/epoch prologue could not redirect kernel entry");
      return;
    }
    append_words_bytes(new_text, *words);
    if (preload_words) {
      while (new_text.size() < prologue_text_offset + kAmdhsaKernelEntryAlignment)
        append_word_bytes(new_text, build_s_nop(0, arch));
      append_words_bytes(new_text, *preload_words);
    }

    ConSanPatchInfo info;
    info.kind = ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue;
    info.anchor_offset = kernel.entry_text_offset;
    info.trampoline_offset = prologue_text_offset;
    info.original_size = 0;
    info.trampoline_size = static_cast<uint32_t>(new_text.size() - prologue_text_offset);
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
    const ConSanOptions &options, const VgprSpillSequence *spill, rj_code_arch_t arch,
    uint32_t barrier_record_capacity, size_t barrier_records_offset, uint32_t original_barrier_word,
    uint64_t cave_text_offset, uint64_t return_text_offset, std::vector<std::string> &errors) {
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
  if (*options.moi_exec_save_sgpr > 100u || *options.moi_exec_save_sgpr % 2u != 0u) {
    errors.emplace_back(
        "ConSan MOI barrier record patch requires an even RJ_CONSAN_MOI_EXEC_SAVE_SGPR in 0..100");
    return std::nullopt;
  }
  if (reject_optional_scratch_range_overlap(options.moi_owner_vgpr, *options.scratch_vgpr, 6,
                                            "MOI owner", errors))
    return std::nullopt;

  std::optional<uint16_t> derived_owner_vgpr;
  std::vector<uint32_t> derived_owner_words;
  if (!options.moi_owner_vgpr && !options.moi_state_owner_sgpr) {
    if (!candidate.kernel_descriptor_file_offset) {
      errors.emplace_back(
          "ConSan MOI barrier record patch requires RJ_CONSAN_MOI_OWNER_VGPR for function "
          "barriers");
      return std::nullopt;
    }
    const auto owner_input =
        moi_descriptor_owner_input(bytes, *candidate.kernel_descriptor_file_offset, arch, errors);
    if (!owner_input)
      return std::nullopt;
    const uint16_t value_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 5u);
    if (!append_moi_owner_input(derived_owner_words, value_vgpr, *owner_input, arch, errors)) {
      errors.emplace_back("ConSan MOI barrier record patch could not encode stable owner");
      return std::nullopt;
    }
    derived_owner_vgpr = value_vgpr;
  }
  ConSanMoiWorkgroupSources workgroup_sources;
  if (candidate.kernel_descriptor_file_offset) {
    const auto descriptor_workgroup_sources = moi_probe_workgroup_sources(
        bytes, *candidate.kernel_descriptor_file_offset, options, arch, errors);
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

  words.reserve(160 + (spill ? spill->save_words.size() + spill->restore_words.size() : 0u));
  if (spill)
    words.insert(words.end(), spill->save_words.begin(), spill->save_words.end());
  if (!append_save_moi_special_state(words, options, arch)) {
    errors.emplace_back("ConSan MOI barrier record patch could not save VCC/SCC");
    return std::nullopt;
  }
  const auto mbcnt_lo = build_v_mbcnt_lo_u32_b32(lane_rank_vgpr, kScalarInlineMinusOne,
                                                 scalar_positive_inline_u32(0), arch);
  const auto mbcnt_hi = build_v_mbcnt_hi_u32_b32(lane_rank_vgpr, kScalarInlineMinusOne,
                                                 vector_source_vgpr(lane_rank_vgpr), arch);
  const auto first_active_lane =
      build_v_cmp_eq_u32_e32_vcc(scalar_positive_inline_u32(0), lane_rank_vgpr, arch);
  const auto save_exec = build_s_and_saveexec_b64(*options.moi_exec_save_sgpr, kWave64VccLo, arch);
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
  record_words.insert(record_words.end(), derived_owner_words.begin(), derived_owner_words.end());
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
      (options.moi_state_owner_sgpr &&
       !append_dynamic_barrier_store_u32_scalar_src(
           record_words, barrier_record_base + offsetof(ConSanMoiBarrierRecord, wave_id),
           *options.moi_state_owner_sgpr, slot_vgpr, *options.scratch_vgpr, arch)) ||
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
  const auto restore_exec = build_s_mov_b64(kWave64ExecLo, *options.moi_exec_save_sgpr, arch);
  if (!skip_record || !restore_exec) {
    errors.emplace_back("ConSan MOI barrier record patch could not encode EXEC restore");
    return std::nullopt;
  }
  words.push_back(*skip_record);
  words.insert(words.end(), record_words.begin(), record_words.end());
  words.push_back(*restore_exec);
  if (!append_restore_moi_special_state(words, options, arch)) {
    errors.emplace_back("ConSan MOI barrier record patch could not restore VCC/SCC");
    return std::nullopt;
  }

  if (spill)
    words.insert(words.end(), spill->restore_words.begin(), spill->restore_words.end());
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
  if (!options.moi_epoch_vgpr && !options.moi_state_epoch_sgpr) {
    errors.emplace_back(
        "ConSan MOI inline-shadow barrier epoch patch requires RJ_CONSAN_MOI_EPOCH_VGPR");
    return std::nullopt;
  }

  std::vector<uint32_t> words;
  words.reserve(7);
  words.push_back(original_barrier_word);
  if (options.moi_epoch_vgpr) {
    const auto increment_epoch = build_v_add_nc_u32_words(
        *options.moi_epoch_vgpr, scalar_positive_inline_u32(1), *options.moi_epoch_vgpr, arch);
    if (!increment_epoch) {
      errors.emplace_back(
          "ConSan MOI inline-shadow barrier epoch patch could not encode epoch add");
      return std::nullopt;
    }
    words.insert(words.end(), increment_epoch->begin(), increment_epoch->end());
  } else {
    if (!append_save_moi_special_state(words, options, arch)) {
      errors.emplace_back(
          "ConSan MOI inline-shadow barrier epoch patch could not preserve scalar state");
      return std::nullopt;
    }
    if (arch != ROCJITSU_CODE_ARCH_CDNA4) {
      errors.emplace_back("ConSan MOI scalar epoch barrier requires CDNA4");
      return std::nullopt;
    }
    words.push_back(pack_sop2(/*s_add_u32=*/0, *options.moi_state_epoch_sgpr,
                              *options.moi_state_epoch_sgpr, scalar_positive_inline_u32(1)));
    if (!append_restore_moi_special_state(words, options, arch)) {
      errors.emplace_back(
          "ConSan MOI inline-shadow barrier epoch patch could not restore scalar state");
      return std::nullopt;
    }
  }

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

[[nodiscard]] std::optional<std::vector<uint32_t>>
build_inline_shadow_private_epoch_barrier_cave_words(
    const ConSanBarrierSite &site, uint16_t scratch_vgpr, uint32_t epoch_offset,
    const VgprSpillSequence &spill, rj_code_arch_t arch, uint32_t original_barrier_word,
    uint64_t cave_text_offset, uint64_t return_text_offset, std::vector<std::string> &errors) {
  const auto load_epoch = build_moi_private_load_b32(scratch_vgpr, epoch_offset, arch);
  const auto wait_load = arch == ROCJITSU_CODE_ARCH_CDNA4 ? build_cdna4_s_wait_vmcnt0(arch)
                                                          : build_s_wait_loadcnt0(arch);
  const auto increment_epoch =
      build_v_add_nc_u32_words(scratch_vgpr, scalar_positive_inline_u32(1), scratch_vgpr, arch);
  const auto store_epoch = build_moi_private_store_b32(scratch_vgpr, epoch_offset, arch);
  const auto wait_store = arch == ROCJITSU_CODE_ARCH_CDNA4 ? build_cdna4_s_wait_vmcnt0(arch)
                                                           : build_s_wait_storecnt0(arch);
  if (!load_epoch || !wait_load || !increment_epoch || !store_epoch || !wait_store) {
    errors.emplace_back("ConSan MOI inline-shadow barrier could not encode private epoch update");
    return std::nullopt;
  }

  std::vector<uint32_t> words;
  words.reserve(2u + spill.save_words.size() + load_epoch->size() + store_epoch->size() +
                spill.restore_words.size() + 3u);
  words.push_back(original_barrier_word);
  words.insert(words.end(), spill.save_words.begin(), spill.save_words.end());
  words.insert(words.end(), load_epoch->begin(), load_epoch->end());
  words.push_back(*wait_load);
  words.insert(words.end(), increment_epoch->begin(), increment_epoch->end());
  words.insert(words.end(), store_epoch->begin(), store_epoch->end());
  words.push_back(*wait_store);
  words.insert(words.end(), spill.restore_words.begin(), spill.restore_words.end());

  const uint64_t branch_pc =
      cave_text_offset + static_cast<uint64_t>(words.size()) * sizeof(uint32_t);
  const auto ret = compute_sopp_branch_simm16(branch_pc, return_text_offset);
  if (!ret) {
    errors.emplace_back("ConSan MOI inline-shadow private epoch return branch is out of range at " +
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
  if (arch != ROCJITSU_CODE_ARCH_RDNA4 && arch != ROCJITSU_CODE_ARCH_CDNA4) {
    result.warnings.emplace_back(
        "ConSan MOI inline-shadow barrier epoch patch has unsupported target");
    return;
  }
  if (!result.modified) {
    result.warnings.emplace_back(
        "ConSan MOI inline-shadow barrier epoch patch skipped because no exact-shadow access "
        "probe was emitted");
    return;
  }
  if (!options.moi_epoch_vgpr && !options.moi_state_epoch_sgpr &&
      !options.automatic_moi_private_epoch) {
    result.warnings.emplace_back(
        "ConSan MOI inline-shadow barrier epoch patch has no persistent epoch state");
    return;
  }
  if (options.moi_epoch_vgpr && static_cast<uint32_t>(*options.moi_epoch_vgpr) + 1u > kMaxVgprs) {
    result.warnings.emplace_back(
        "ConSan MOI inline-shadow barrier epoch patch needs one epoch VGPR");
    return;
  }

  std::vector<BarrierRecordCandidate> candidates;
  for (const ConSanKernelInfo &kernel : result.kernels)
    append_barrier_epoch_candidates(kernel, candidates);
  for (const ConSanFunctionInfo &function : result.functions)
    append_barrier_epoch_candidates(function, candidates);
  if (!options.test_kernel_name_filter.empty()) {
    std::erase_if(candidates, [&](const BarrierRecordCandidate &candidate) {
      return candidate.container_name.find(options.test_kernel_name_filter) == std::string::npos;
    });
  }
  if (automatic_moi_persistent_state(options)) {
    std::erase_if(candidates, [](const BarrierRecordCandidate &candidate) {
      return !candidate.kernel_descriptor_file_offset;
    });
  }
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

  struct PlannedInlineEpochBarrier {
    BarrierRecordCandidate candidate;
    std::optional<uint16_t> scratch_vgpr;
    std::optional<uint32_t> private_epoch_offset;
    std::optional<VgprSpillSequence> spill;
    uint32_t required_private_bytes = 0;
  };
  std::vector<PlannedInlineEpochBarrier> planned_candidates;
  MoiDescriptorPrivateRequirements private_requirements;
  if (options.automatic_moi_private_epoch) {
    for (const BarrierRecordCandidate &candidate : selected_candidates) {
      if (!candidate.kernel_descriptor_file_offset)
        continue;
      const ConSanKernelInfo *kernel =
          kernel_for_descriptor(result, *candidate.kernel_descriptor_file_offset);
      if (kernel == nullptr)
        continue;
      const auto access_patch =
          std::ranges::find_if(result.patches, [&](const ConSanPatchInfo &patch) {
            return (patch.kind == ConSanPatchKind::InlineMoiExactShadowStore ||
                    patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore) &&
                   patch.scratch_vgpr && patch.persistent_epoch_private_offset &&
                   kernel_owns_patch(*kernel, patch);
          });
      if (access_patch == result.patches.end()) {
        result.warnings.emplace_back(
            "ConSan MOI private epoch barrier has no access-probe state for kernel " +
            kernel->name);
        continue;
      }
      const uint32_t ephemeral_base =
          util::align_up(*access_patch->persistent_owner_private_offset + SpillManager::kSlotBytes,
                         SpillManager::kDbiZoneAlignment);
      SpillManager manager(ephemeral_base, kMaxAddressFreeScratchPrivateBytes);
      auto spill = build_vgpr_spill_sequence(manager, *access_patch->scratch_vgpr,
                                             /*vgpr_count=*/1, arch);
      if (!spill) {
        result.warnings.emplace_back(
            "ConSan MOI private epoch barrier could not preserve its temporary VGPR");
        continue;
      }
      const uint32_t required_private_bytes =
          std::max(access_patch->required_private_segment_size, spill->total_private_bytes);
      auto [it, inserted] = private_requirements.emplace(*candidate.kernel_descriptor_file_offset,
                                                         required_private_bytes);
      if (!inserted)
        it->second = std::max(it->second, required_private_bytes);
      planned_candidates.push_back({candidate, access_patch->scratch_vgpr,
                                    access_patch->persistent_epoch_private_offset, std::move(spill),
                                    required_private_bytes});
    }
    if (!apply_spill_descriptor_requirements(patcher, code_object, active_bytes, result,
                                             private_requirements, result.errors))
      return;
  } else if (options.automatic_moi_scalar_identity) {
    // The scalar entry prologue grows each owning descriptor after all event
    // patches have identified their owners.
  } else if (options.automatic_moi_persistent_vgprs) {
    MoiDescriptorVgprRequirements requirements;
    for (const BarrierRecordCandidate &candidate : selected_candidates) {
      if (candidate.kernel_descriptor_file_offset) {
        requirements[*candidate.kernel_descriptor_file_offset] =
            static_cast<uint16_t>(*options.moi_epoch_vgpr + 1u);
      }
    }
    if (!apply_descriptor_requirements(patcher, code_object, active_bytes, result, requirements,
                                       arch, result.errors)) {
      return;
    }
  } else if (!grow_moi_kernel_descriptor_vgprs(patcher, active_bytes, result,
                                               static_cast<uint32_t>(*options.moi_epoch_vgpr) + 1u,
                                               arch)) {
    return;
  }
  if (!options.automatic_moi_private_epoch) {
    for (const BarrierRecordCandidate &candidate : selected_candidates) {
      planned_candidates.push_back(PlannedInlineEpochBarrier{
          .candidate = candidate,
          .scratch_vgpr = std::nullopt,
          .private_epoch_offset = std::nullopt,
          .spill = std::nullopt,
          .required_private_bytes = 0,
      });
    }
  }
  if (planned_candidates.empty()) {
    result.warnings.emplace_back(
        "ConSan MOI inline-shadow barrier epoch patch found no compatible persistent state");
    return;
  }

  std::vector<uint8_t> new_text(old_text.begin(), old_text.end());
  DbiPatchPlacementPlanner placement_planner(arch, old_text.size());
  std::vector<ConSanPatchInfo> patches;
  patches.reserve(planned_candidates.size());
  for (const PlannedInlineEpochBarrier &planned : planned_candidates) {
    const BarrierRecordCandidate &candidate = planned.candidate;
    const ConSanBarrierSite &site = candidate.site;
    uint32_t original_barrier_word = 0;
    std::memcpy(&original_barrier_word, new_text.data() + site.text_offset,
                sizeof(original_barrier_word));

    const uint64_t cave_text_offset = placement_planner.appended_end();
    const uint64_t return_text_offset = site.text_offset + site.size;
    std::optional<std::vector<uint32_t>> cave_words;
    if (planned.private_epoch_offset && planned.scratch_vgpr && planned.spill) {
      cave_words = build_inline_shadow_private_epoch_barrier_cave_words(
          site, *planned.scratch_vgpr, *planned.private_epoch_offset, *planned.spill, arch,
          original_barrier_word, cave_text_offset, return_text_offset, result.errors);
    } else {
      cave_words = build_inline_shadow_barrier_epoch_cave_words(
          site, options, arch, original_barrier_word, cave_text_offset, return_text_offset,
          result.errors);
    }
    if (!cave_words)
      return;

    const auto placement =
        plan_prebuilt_appended_cave(placement_planner, site.text_offset, site.size, *cave_words,
                                    result.errors, "ConSan MOI inline-shadow barrier epoch patch");
    if (!placement || placement->body_offset != cave_text_offset ||
        new_text.size() != placement->body_offset) {
      result.errors.emplace_back(
          "ConSan MOI inline-shadow barrier epoch patch has a stale cave mapping");
      return;
    }

    const auto fwd = compute_sopp_branch_simm16(placement->anchor_offset, placement->body_offset);
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
    info.scratch_vgpr = planned.scratch_vgpr;
    if (planned.scratch_vgpr) {
      const ConSanKernelInfo *kernel =
          planned.candidate.kernel_descriptor_file_offset
              ? kernel_for_descriptor(result, *planned.candidate.kernel_descriptor_file_offset)
              : nullptr;
      if (kernel != nullptr) {
        const auto access_patch =
            std::ranges::find_if(result.patches, [&](const ConSanPatchInfo &patch) {
              return (patch.kind == ConSanPatchKind::InlineMoiExactShadowStore ||
                      patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore) &&
                     patch.persistent_owner_private_offset && kernel_owns_patch(*kernel, patch);
            });
        if (access_patch != result.patches.end())
          info.persistent_owner_private_offset = access_patch->persistent_owner_private_offset;
      }
    }
    info.persistent_epoch_private_offset = planned.private_epoch_offset;
    info.required_private_segment_size = planned.required_private_bytes;
    if (planned.spill)
      info.spilled_vgpr_count = planned.spill->vgpr_count;
    patches.push_back(info);
  }

  if (!patcher.replace_text(new_text)) {
    result.errors.emplace_back("ConSan MOI inline-shadow barrier epoch patch could not grow .text");
    return;
  }

  result.elf_bytes = patcher.emit();
  if (!apply_spill_metadata_requirements(result.elf_bytes, result, private_requirements,
                                         result.errors)) {
    result.elf_bytes.clear();
    return;
  }
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
  if (arch != ROCJITSU_CODE_ARCH_RDNA4 && arch != ROCJITSU_CODE_ARCH_CDNA4) {
    result.warnings.emplace_back("ConSan MOI barrier record patch has unsupported target");
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
  if (!options.moi_exec_save_sgpr) {
    result.warnings.emplace_back(
        "ConSan MOI barrier record patch requires RJ_CONSAN_MOI_EXEC_SAVE_SGPR");
    return;
  }
  if (*options.moi_exec_save_sgpr > 100u || *options.moi_exec_save_sgpr % 2u != 0u) {
    result.warnings.emplace_back(
        "ConSan MOI barrier record patch requires an even RJ_CONSAN_MOI_EXEC_SAVE_SGPR in 0..100");
    return;
  }

  std::vector<BarrierRecordCandidate> candidates;
  for (const ConSanKernelInfo &kernel : result.kernels)
    append_barrier_epoch_candidates(kernel, candidates);
  for (const ConSanFunctionInfo &function : result.functions)
    append_barrier_epoch_candidates(function, candidates);
  if (!options.test_kernel_name_filter.empty()) {
    std::erase_if(candidates, [&](const BarrierRecordCandidate &candidate) {
      return candidate.container_name.find(options.test_kernel_name_filter) == std::string::npos;
    });
  }
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
  struct PlannedBarrierRecord {
    BarrierRecordCandidate candidate;
    ResolvedMoiScratchPlan resources;
    std::optional<VgprSpillSequence> spill;
  };
  std::vector<PlannedBarrierRecord> selected_candidates;
  selected_candidates.reserve(max_candidates);
  MoiResourcePlanningState resource_state(bytes, arch, result);
  MoiSpillManagers spill_managers;
  MoiDescriptorVgprRequirements descriptor_requirements;
  MoiDescriptorSgprRequirements scalar_requirements;
  MoiDescriptorPrivateRequirements private_requirements;
  for (size_t candidate_index = 0; candidate_index < candidates.size(); ++candidate_index) {
    const BarrierRecordCandidate &candidate = candidates[candidate_index];
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
    const ConSanCandidateResourcePlan *existing_plan =
        resource_plan_for_site(result, ConSanResourceSiteKind::Barrier, site.text_offset);
    ConSanCandidateResourcePlan plan =
        existing_plan
            ? *existing_plan
            : plan_moi_resource_site(resource_state, options, ConSanResourceSiteKind::Barrier,
                                     candidate_index, site.text_offset,
                                     candidate.kernel_descriptor_file_offset,
                                     /*scratch_count=*/6u);
    if (!existing_plan)
      result.resource_plans.push_back(plan);
    const auto resources = resolve_moi_scratch_plan(plan, options, /*expected_count=*/6u);
    if (!resources) {
      result.warnings.emplace_back(
          "ConSan MOI barrier record patch has no usable resource plan in " +
          candidate.container_name);
      continue;
    }
    BarrierRecordCandidate planned_candidate = candidate;
    if (!options.moi_owner_vgpr && !planned_candidate.kernel_descriptor_file_offset) {
      planned_candidate.kernel_descriptor_file_offset =
          common_moi_record_owner_descriptor(bytes, *resources, options, arch, result.warnings);
      if (!planned_candidate.kernel_descriptor_file_offset)
        continue;
    }
    std::optional<VgprSpillSequence> spill;
    if (resources->source == ConSanRegisterAllocationSource::SpillRequired) {
      spill = build_moi_spill_sequence(result, *resources, spill_managers, arch, result.warnings);
      if (!spill)
        continue;
      note_spill_descriptor_requirements(private_requirements, *resources, *spill);
    }
    note_descriptor_requirements(descriptor_requirements, *resources);
    note_moi_sgpr_requirements(scalar_requirements, *resources, options);
    selected_candidates.push_back({std::move(planned_candidate), *resources, std::move(spill)});
  }

  if (selected_candidates.empty()) {
    result.warnings.emplace_back("ConSan MOI barrier record patch found no patchable barriers");
    return;
  }
  if (!apply_descriptor_requirements(patcher, code_object, active_bytes, result,
                                     descriptor_requirements, arch, result.errors) ||
      !apply_spill_descriptor_requirements(patcher, code_object, active_bytes, result,
                                           private_requirements, result.errors) ||
      !apply_sgpr_descriptor_requirements(patcher, result, code_object, scalar_requirements, arch,
                                          result.errors)) {
    return;
  }

  std::vector<uint8_t> new_text(old_text.begin(), old_text.end());
  DbiPatchPlacementPlanner placement_planner(arch, old_text.size());
  std::vector<ConSanPatchInfo> patches;
  for (const PlannedBarrierRecord &planned : selected_candidates) {
    const BarrierRecordCandidate &candidate = planned.candidate;
    const ConSanBarrierSite &site = candidate.site;
    uint32_t original_barrier_word = 0;
    std::memcpy(&original_barrier_word, new_text.data() + site.text_offset,
                sizeof(original_barrier_word));

    const uint64_t cave_text_offset = placement_planner.appended_end();
    const uint64_t return_text_offset = site.text_offset + site.size;
    ConSanOptions candidate_options = options;
    candidate_options.scratch_vgpr = planned.resources.base;
    auto cave_words = build_barrier_record_cave_words(
        active_bytes, candidate, candidate_options, planned.spill ? &*planned.spill : nullptr, arch,
        layout.barrier_record_capacity, layout.barrier_records_offset, original_barrier_word,
        cave_text_offset, return_text_offset, result.errors);
    if (!cave_words)
      return;

    const auto placement =
        plan_prebuilt_appended_cave(placement_planner, site.text_offset, site.size, *cave_words,
                                    result.errors, "ConSan MOI barrier record patch");
    if (!placement || placement->body_offset != cave_text_offset ||
        new_text.size() != placement->body_offset) {
      result.errors.emplace_back("ConSan MOI barrier record patch has a stale cave mapping");
      return;
    }

    const auto fwd = compute_sopp_branch_simm16(placement->anchor_offset, placement->body_offset);
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
    info.scratch_vgpr = planned.resources.base;
    info.owner_descriptor_file_offsets = planned.resources.owner_descriptor_file_offsets;
    if (planned.spill) {
      info.spilled_vgpr_count = planned.spill->vgpr_count;
      info.required_private_segment_size = planned.spill->total_private_bytes;
    }
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
  if (!apply_spill_metadata_requirements(result.elf_bytes, result, private_requirements,
                                         result.errors)) {
    result.elf_bytes.clear();
    return;
  }
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
  const bool rdna4_no_saddr =
      site.size == 3u * sizeof(uint32_t) && site.raw_saddr == kRdna4FlatNoSaddr;
  const bool cdna4_no_saddr =
      site.size == 2u * sizeof(uint32_t) && site.raw_segment == 0u && site.raw_saddr == 0u;
  if (!rdna4_no_saddr && !cdna4_no_saddr)
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
  if (*options.moi_exec_save_sgpr > 94u || *options.moi_exec_save_sgpr % 2u != 0u) {
    errors.emplace_back("ConSan MOI inline atomic acquire patch requires an even "
                        "RJ_CONSAN_MOI_EXEC_SAVE_SGPR in 0..94");
    return false;
  }
  return true;
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_inline_atomic_ordering_cave_words(
    std::span<const uint8_t> bytes, const AtomicRecordCandidate &candidate,
    const ConSanOptions &options, const VgprSpillSequence *spill, rj_code_arch_t arch,
    std::optional<uint32_t> private_owner_offset, std::optional<uint32_t> private_epoch_offset,
    size_t inline_atomic_release_slots_offset, uint64_t cave_text_offset,
    uint64_t return_text_offset, std::vector<std::string> &errors) {
  constexpr uint16_t kInlineAtomicScratchCount = 3;
  if (!options.scratch_vgpr) {
    errors.emplace_back("ConSan MOI inline atomic patch requires RJ_CONSAN_TMP_VGPR");
    return std::nullopt;
  }
  if (static_cast<uint32_t>(*options.scratch_vgpr) + kInlineAtomicScratchCount > kMaxVgprs) {
    errors.emplace_back("ConSan MOI inline atomic patch needs three scratch VGPRs");
    return std::nullopt;
  }
  if (options.automatic_moi_private_epoch && (!private_owner_offset || !private_epoch_offset)) {
    errors.emplace_back("ConSan MOI inline atomic patch has incomplete private identity layout");
    return std::nullopt;
  }
  if (!options.automatic_moi_private_epoch &&
      ((!options.moi_owner_vgpr && !options.moi_state_owner_sgpr) ||
       (!options.moi_epoch_vgpr && !options.moi_state_epoch_sgpr))) {
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
  const uint16_t address_vgpr = *candidate.site.addr_vgpr;

  std::vector<uint32_t> words;
  words.reserve(candidate.site.size / sizeof(uint32_t) + 96u +
                (spill ? spill->save_words.size() + spill->restore_words.size() : 0u));
  if (spill)
    words.insert(words.end(), spill->save_words.begin(), spill->save_words.end());
  for (uint64_t offset = 0; offset < candidate.site.size; offset += sizeof(uint32_t)) {
    uint32_t word = 0;
    std::memcpy(&word, bytes.data() + candidate.site.file_offset + offset, sizeof(word));
    words.push_back(word);
  }
  const auto wait_flat = build_s_wait_flat_load0(arch);
  if (!wait_flat) {
    errors.emplace_back(
        "ConSan MOI inline atomic patch could not encode the atomic completion wait");
    return std::nullopt;
  }
  words.push_back(*wait_flat);

  if (atomic_event_kind_for_site(candidate.site) == ConSanMoiAtomicEventKind::Release) {
    uint16_t owner_vgpr = options.moi_owner_vgpr.value_or(value_vgpr);
    if (private_owner_offset &&
        !append_moi_private_load_wait(words, value_vgpr, *private_owner_offset, arch)) {
      errors.emplace_back("ConSan MOI inline atomic release could not load private owner");
      return std::nullopt;
    }
    if (options.moi_state_owner_sgpr)
      words.push_back(build_v_mov_b32_e32(value_vgpr, *options.moi_state_owner_sgpr, arch));
    if (!append_store_u32_vgpr(words,
                               slot_base + offsetof(ConSanMoiInlineAtomicReleaseSlot, owner_id),
                               owner_vgpr, *options.scratch_vgpr, arch)) {
      errors.emplace_back("ConSan MOI inline atomic release could not store owner");
      return std::nullopt;
    }
    uint16_t epoch_vgpr = options.moi_epoch_vgpr.value_or(value_vgpr);
    if (private_epoch_offset &&
        !append_moi_private_load_wait(words, value_vgpr, *private_epoch_offset, arch)) {
      errors.emplace_back("ConSan MOI inline atomic release could not load private epoch");
      return std::nullopt;
    }
    if (options.moi_state_epoch_sgpr)
      words.push_back(build_v_mov_b32_e32(value_vgpr, *options.moi_state_epoch_sgpr, arch));
    if (!append_store_u32_vgpr(words, slot_base + offsetof(ConSanMoiInlineAtomicReleaseSlot, epoch),
                               epoch_vgpr, *options.scratch_vgpr, arch) ||
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
    if (!append_save_moi_special_state(words, options, arch)) {
      errors.emplace_back("ConSan MOI inline atomic acquire patch could not save VCC/SCC");
      return std::nullopt;
    }

    const auto narrow_if_valid =
        build_s_and_saveexec_b64(*options.moi_exec_save_sgpr, kWave64VccLo, arch);
    const auto narrow_if_low_matches = build_s_and_saveexec_b64(
        static_cast<uint16_t>(*options.moi_exec_save_sgpr + 2u), kWave64VccLo, arch);
    const auto narrow_if_high_matches = build_s_and_saveexec_b64(
        static_cast<uint16_t>(*options.moi_exec_save_sgpr + 4u), kWave64VccLo, arch);
    const auto narrow_if_other_owner = build_s_and_saveexec_b64(
        static_cast<uint16_t>(*options.moi_exec_save_sgpr + 6u), kWave64VccLo, arch);
    const auto restore_exec = build_s_mov_b64(kWave64ExecLo, *options.moi_exec_save_sgpr, arch);
    if (!narrow_if_valid || !narrow_if_low_matches || !narrow_if_high_matches ||
        !narrow_if_other_owner || !restore_exec) {
      errors.emplace_back("ConSan MOI inline atomic acquire patch could not encode EXEC ops");
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
    const uint16_t owner_vgpr = (private_owner_offset || options.moi_state_owner_sgpr)
                                    ? *options.scratch_vgpr
                                    : *options.moi_owner_vgpr;
    if (private_owner_offset &&
        !append_moi_private_load_wait(words, owner_vgpr, *private_owner_offset, arch)) {
      errors.emplace_back("ConSan MOI inline atomic acquire patch could not load private owner");
      return std::nullopt;
    }
    if (options.moi_state_owner_sgpr)
      words.push_back(build_v_mov_b32_e32(owner_vgpr, *options.moi_state_owner_sgpr, arch));
    const auto owner_ne =
        build_v_cmp_ne_u32_e32_vcc(vector_source_vgpr(owner_vgpr), value_vgpr, arch);
    if (!owner_ne) {
      errors.emplace_back("ConSan MOI inline atomic acquire patch could not compare owner");
      return std::nullopt;
    }
    words.push_back(*owner_ne);
    words.push_back(*narrow_if_other_owner);

    std::optional<size_t> skip_epoch_if_empty;
    if (options.moi_state_epoch_sgpr) {
      const auto skip = build_s_cbranch_execz(0, arch);
      if (!skip) {
        errors.emplace_back("ConSan MOI inline atomic acquire could not guard scalar import");
        return std::nullopt;
      }
      skip_epoch_if_empty = words.size();
      words.push_back(*skip);
    }

    if (!append_load_u32_vgpr(words, slot_base + offsetof(ConSanMoiInlineAtomicReleaseSlot, epoch),
                              value_vgpr, *options.scratch_vgpr, arch)) {
      errors.emplace_back("ConSan MOI inline atomic acquire patch could not load epoch");
      return std::nullopt;
    }
    const uint16_t imported_epoch_vgpr = (private_epoch_offset || options.moi_state_epoch_sgpr)
                                             ? value_vgpr
                                             : *options.moi_epoch_vgpr;
    const auto import_epoch = build_v_add_nc_u32_words(
        imported_epoch_vgpr, scalar_positive_inline_u32(1), value_vgpr, arch);
    if (!import_epoch) {
      errors.emplace_back("ConSan MOI inline atomic acquire patch could not import epoch");
      return std::nullopt;
    }
    words.insert(words.end(), import_epoch->begin(), import_epoch->end());
    if (options.moi_state_epoch_sgpr) {
      const auto read_epoch =
          build_v_readfirstlane_b32(*options.moi_state_epoch_sgpr, imported_epoch_vgpr, arch);
      if (!read_epoch) {
        errors.emplace_back("ConSan MOI inline atomic acquire could not import scalar epoch");
        return std::nullopt;
      }
      words.push_back(*read_epoch);
    }
    if (private_epoch_offset) {
      const auto store_epoch =
          build_moi_private_store_b32(imported_epoch_vgpr, *private_epoch_offset, arch);
      const auto wait_store = arch == ROCJITSU_CODE_ARCH_CDNA4 ? build_cdna4_s_wait_vmcnt0(arch)
                                                               : build_s_wait_storecnt0(arch);
      if (!store_epoch || !wait_store) {
        errors.emplace_back("ConSan MOI inline atomic acquire patch could not store private epoch");
        return std::nullopt;
      }
      words.insert(words.end(), store_epoch->begin(), store_epoch->end());
      words.push_back(*wait_store);
    }
    if (skip_epoch_if_empty) {
      const size_t skipped_words = words.size() - *skip_epoch_if_empty - 1u;
      if (skipped_words > static_cast<size_t>(std::numeric_limits<int16_t>::max())) {
        errors.emplace_back("ConSan MOI inline atomic scalar import branch is out of range");
        return std::nullopt;
      }
      words[*skip_epoch_if_empty] =
          *build_s_cbranch_execz(static_cast<int16_t>(skipped_words), arch);
    }
    words.push_back(*restore_exec);
    if (!append_restore_moi_special_state(words, options, arch)) {
      errors.emplace_back("ConSan MOI inline atomic acquire patch could not restore VCC/SCC");
      return std::nullopt;
    }
  }

  if (spill)
    words.insert(words.end(), spill->restore_words.begin(), spill->restore_words.end());
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
  if (arch != ROCJITSU_CODE_ARCH_RDNA4 && arch != ROCJITSU_CODE_ARCH_CDNA4) {
    result.warnings.emplace_back(
        "ConSan MOI inline atomic patch currently supports only RDNA4 and CDNA4");
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
  if (!options.automatic_moi_private_epoch &&
      ((!options.moi_owner_vgpr && !options.moi_state_owner_sgpr) ||
       (!options.moi_epoch_vgpr && !options.moi_state_epoch_sgpr))) {
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
  if (!options.test_kernel_name_filter.empty()) {
    std::erase_if(candidates, [&](const AtomicRecordCandidate &candidate) {
      return candidate.container_name.find(options.test_kernel_name_filter) == std::string::npos;
    });
  }
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
  struct PlannedInlineAtomic {
    AtomicRecordCandidate candidate;
    ResolvedMoiScratchPlan resources;
    std::optional<MoiPrivateEpochLayout> private_layout;
    std::optional<VgprSpillSequence> spill;
    uint32_t required_private_bytes = 0;
  };
  std::vector<PlannedInlineAtomic> planned_candidates;
  MoiResourcePlanningState resource_state(bytes, arch, result);
  MoiSpillManagers spill_managers;
  MoiDescriptorVgprRequirements descriptor_requirements;
  MoiDescriptorSgprRequirements scalar_requirements;
  MoiDescriptorPrivateRequirements private_requirements;
  for (size_t candidate_index = 0; candidate_index < candidates.size(); ++candidate_index) {
    if (planned_candidates.size() == options.max_patches)
      break;
    const AtomicRecordCandidate &candidate = candidates[candidate_index];
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
    const ConSanCandidateResourcePlan *existing_plan =
        resource_plan_for_site(result, ConSanResourceSiteKind::Atomic, site.text_offset);
    ConSanCandidateResourcePlan plan =
        existing_plan
            ? *existing_plan
            : plan_moi_resource_site(resource_state, options, ConSanResourceSiteKind::Atomic,
                                     candidate_index, site.text_offset,
                                     candidate.kernel_descriptor_file_offset,
                                     /*scratch_count=*/3u);
    if (!existing_plan)
      result.resource_plans.push_back(plan);
    const auto resources = resolve_moi_scratch_plan(plan, options, /*expected_count=*/3u);
    if (!resources) {
      result.warnings.emplace_back(
          "ConSan MOI inline atomic patch has no usable resource plan in " +
          candidate.container_name);
      continue;
    }
    AtomicRecordCandidate planned_candidate = candidate;
    if (!options.moi_owner_vgpr && !planned_candidate.kernel_descriptor_file_offset) {
      planned_candidate.kernel_descriptor_file_offset =
          common_moi_record_owner_descriptor(bytes, *resources, options, arch, result.warnings);
      if (!planned_candidate.kernel_descriptor_file_offset)
        continue;
    }
    std::optional<MoiPrivateEpochLayout> private_layout;
    if (options.automatic_moi_private_epoch) {
      private_layout = build_moi_private_epoch_layout(result, *resources, result.warnings);
      if (!private_layout)
        continue;
    }
    std::optional<VgprSpillSequence> spill;
    if (resources->source == ConSanRegisterAllocationSource::SpillRequired) {
      spill = build_moi_spill_sequence(
          result, *resources, spill_managers, arch, result.warnings,
          private_layout ? std::optional<uint32_t>(private_layout->ephemeral_base) : std::nullopt);
      if (!spill)
        continue;
    }
    const uint32_t required_private_bytes =
        std::max(private_layout ? private_layout->ephemeral_base : 0u,
                 spill ? spill->total_private_bytes : 0u);
    if (required_private_bytes != 0) {
      for (uint64_t descriptor_offset : resources->owner_descriptor_file_offsets) {
        auto [it, inserted] =
            private_requirements.emplace(descriptor_offset, required_private_bytes);
        if (!inserted)
          it->second = std::max(it->second, required_private_bytes);
      }
    }
    note_descriptor_requirements(descriptor_requirements, *resources);
    note_moi_sgpr_requirements(scalar_requirements, *resources, options);
    planned_candidates.push_back({candidate, *resources, std::move(private_layout),
                                  std::move(spill), required_private_bytes});
  }
  if (planned_candidates.empty()) {
    result.warnings.emplace_back("ConSan MOI inline atomic patch found no patchable atomics");
    return;
  }
  if (!apply_descriptor_requirements(patcher, code_object, active_bytes, result,
                                     descriptor_requirements, arch, result.errors) ||
      !apply_spill_descriptor_requirements(patcher, code_object, active_bytes, result,
                                           private_requirements, result.errors) ||
      !apply_sgpr_descriptor_requirements(patcher, result, code_object, scalar_requirements, arch,
                                          result.errors)) {
    return;
  }
  std::vector<uint8_t> new_text(old_text.begin(), old_text.end());
  DbiPatchPlacementPlanner placement_planner(arch, old_text.size());
  std::vector<ConSanPatchInfo> patches;
  for (const PlannedInlineAtomic &planned : planned_candidates) {
    const AtomicRecordCandidate &candidate = planned.candidate;
    const ConSanAtomicSite &site = candidate.site;

    const uint64_t cave_text_offset = placement_planner.appended_end();
    const uint64_t return_text_offset = site.text_offset + site.size;
    ConSanOptions candidate_options = options;
    candidate_options.scratch_vgpr = planned.resources.base;
    auto cave_words = build_inline_atomic_ordering_cave_words(
        active_bytes, candidate, candidate_options, planned.spill ? &*planned.spill : nullptr, arch,
        planned.private_layout ? std::optional<uint32_t>(planned.private_layout->owner_offset)
                               : std::nullopt,
        planned.private_layout ? std::optional<uint32_t>(planned.private_layout->epoch_offset)
                               : std::nullopt,
        layout.inline_atomic_release_slots_offset, cave_text_offset, return_text_offset,
        result.errors);
    if (!cave_words)
      return;

    const auto placement =
        plan_prebuilt_appended_cave(placement_planner, site.text_offset, site.size, *cave_words,
                                    result.errors, "ConSan MOI inline atomic patch");
    if (!placement || placement->body_offset != cave_text_offset ||
        new_text.size() != placement->body_offset) {
      result.errors.emplace_back("ConSan MOI inline atomic patch has a stale cave mapping");
      return;
    }

    const auto fwd = compute_sopp_branch_simm16(placement->anchor_offset, placement->body_offset);
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
    info.scratch_vgpr = planned.resources.base;
    info.owner_descriptor_file_offsets = planned.resources.owner_descriptor_file_offsets;
    if (planned.private_layout) {
      info.persistent_owner_private_offset = planned.private_layout->owner_offset;
      info.persistent_epoch_private_offset = planned.private_layout->epoch_offset;
    }
    info.required_private_segment_size = planned.required_private_bytes;
    if (planned.spill) {
      info.spilled_vgpr_count = planned.spill->vgpr_count;
    }
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
  if (!apply_spill_metadata_requirements(result.elf_bytes, result, private_requirements,
                                         result.errors)) {
    result.elf_bytes.clear();
    return;
  }
  result.patches.insert(result.patches.end(), patches.begin(), patches.end());
  result.modified = true;
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_atomic_record_cave_words(
    std::span<const uint8_t> bytes, const AtomicRecordCandidate &candidate,
    const ConSanOptions &options, const VgprSpillSequence *spill, rj_code_arch_t arch,
    uint32_t record_index, uint32_t record_count, size_t atomic_records_offset,
    uint64_t cave_text_offset, uint64_t return_text_offset, std::vector<std::string> &errors) {
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
  std::vector<uint32_t> derived_owner_words;
  if (!options.moi_owner_vgpr && !options.moi_state_owner_sgpr) {
    if (!candidate.kernel_descriptor_file_offset) {
      errors.emplace_back(
          "ConSan MOI atomic record patch requires RJ_CONSAN_MOI_OWNER_VGPR for function atomics");
      return std::nullopt;
    }
    const auto owner_input =
        moi_descriptor_owner_input(bytes, *candidate.kernel_descriptor_file_offset, arch, errors);
    if (!owner_input)
      return std::nullopt;
    const uint16_t value_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 2u);
    if (!append_moi_owner_input(derived_owner_words, value_vgpr, *owner_input, arch, errors)) {
      errors.emplace_back("ConSan MOI atomic record patch could not encode stable owner");
      return std::nullopt;
    }
    derived_owner_vgpr = value_vgpr;
  }

  ConSanMoiWorkgroupSources workgroup_sources;
  if (candidate.kernel_descriptor_file_offset) {
    const auto descriptor_workgroup_sources = moi_probe_workgroup_sources(
        bytes, *candidate.kernel_descriptor_file_offset, options, arch, errors);
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
  words.reserve(candidate.site.size / sizeof(uint32_t) + 96u +
                (spill ? spill->save_words.size() + spill->restore_words.size() : 0u));
  if (spill)
    words.insert(words.end(), spill->save_words.begin(), spill->save_words.end());
  words.insert(words.end(), derived_owner_words.begin(), derived_owner_words.end());
  for (uint64_t offset = 0; offset < candidate.site.size; offset += sizeof(uint32_t)) {
    uint32_t word = 0;
    std::memcpy(&word, bytes.data() + candidate.site.file_offset + offset, sizeof(word));
    words.push_back(word);
  }
  const auto wait_flat = build_s_wait_flat_load0(arch);
  if (!wait_flat) {
    errors.emplace_back(
        "ConSan MOI atomic record patch could not encode the atomic completion wait");
    return std::nullopt;
  }
  words.push_back(*wait_flat);

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
      (options.moi_state_owner_sgpr &&
       !append_store_u32_scalar_src(words,
                                    atomic_record_base + offsetof(ConSanMoiAtomicRecord, owner_id),
                                    *options.moi_state_owner_sgpr, *options.scratch_vgpr, arch)) ||
      (options.moi_epoch_vgpr &&
       !append_store_u32_vgpr(words, atomic_record_base + offsetof(ConSanMoiAtomicRecord, epoch),
                              *options.moi_epoch_vgpr, *options.scratch_vgpr, arch)) ||
      (options.moi_state_epoch_sgpr &&
       !append_store_u32_scalar_src(words,
                                    atomic_record_base + offsetof(ConSanMoiAtomicRecord, epoch),
                                    *options.moi_state_epoch_sgpr, *options.scratch_vgpr, arch)) ||
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

  if (spill)
    words.insert(words.end(), spill->restore_words.begin(), spill->restore_words.end());
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
  if (arch != ROCJITSU_CODE_ARCH_RDNA4 && arch != ROCJITSU_CODE_ARCH_CDNA4) {
    result.warnings.emplace_back(
        "ConSan MOI atomic record patch currently supports only RDNA4 and CDNA4");
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
  std::vector<AtomicRecordCandidate> candidates;
  for (const ConSanKernelInfo &kernel : result.kernels)
    append_atomic_record_candidates(kernel, candidates);
  for (const ConSanFunctionInfo &function : result.functions)
    append_atomic_record_candidates(function, candidates);
  if (!options.test_kernel_name_filter.empty()) {
    std::erase_if(candidates, [&](const AtomicRecordCandidate &candidate) {
      return candidate.container_name.find(options.test_kernel_name_filter) == std::string::npos;
    });
  }
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
  struct PlannedAtomicRecord {
    AtomicRecordCandidate candidate;
    ResolvedMoiScratchPlan resources;
    std::optional<VgprSpillSequence> spill;
  };
  std::vector<PlannedAtomicRecord> selected_candidates;
  selected_candidates.reserve(max_candidates);
  MoiResourcePlanningState resource_state(bytes, arch, result);
  MoiSpillManagers spill_managers;
  MoiDescriptorVgprRequirements descriptor_requirements;
  MoiDescriptorSgprRequirements scalar_requirements;
  MoiDescriptorPrivateRequirements private_requirements;
  for (size_t candidate_index = 0; candidate_index < candidates.size(); ++candidate_index) {
    const AtomicRecordCandidate &candidate = candidates[candidate_index];
    if (selected_candidates.size() == max_candidates)
      break;
    const ConSanAtomicSite &site = candidate.site;
    if (site.text_offset > old_text.size() || site.size > old_text.size() - site.text_offset) {
      result.errors.emplace_back("ConSan MOI atomic record patch site is outside .text");
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
          "ConSan MOI atomic record patch skipped an overlapping patch in " +
          candidate.container_name);
      continue;
    }
    const ConSanCandidateResourcePlan *existing_plan =
        resource_plan_for_site(result, ConSanResourceSiteKind::Atomic, site.text_offset);
    ConSanCandidateResourcePlan plan =
        existing_plan
            ? *existing_plan
            : plan_moi_resource_site(resource_state, options, ConSanResourceSiteKind::Atomic,
                                     candidate_index, site.text_offset,
                                     candidate.kernel_descriptor_file_offset,
                                     /*scratch_count=*/3u);
    if (!existing_plan)
      result.resource_plans.push_back(plan);
    const auto resources = resolve_moi_scratch_plan(plan, options, /*expected_count=*/3u);
    if (!resources) {
      result.warnings.emplace_back(
          "ConSan MOI atomic record patch has no usable resource plan in " +
          candidate.container_name);
      continue;
    }
    AtomicRecordCandidate planned_candidate = candidate;
    if (!options.moi_owner_vgpr && !planned_candidate.kernel_descriptor_file_offset) {
      planned_candidate.kernel_descriptor_file_offset =
          common_moi_record_owner_descriptor(bytes, *resources, options, arch, result.warnings);
      if (!planned_candidate.kernel_descriptor_file_offset)
        continue;
    }
    std::optional<VgprSpillSequence> spill;
    if (resources->source == ConSanRegisterAllocationSource::SpillRequired) {
      spill = build_moi_spill_sequence(result, *resources, spill_managers, arch, result.warnings);
      if (!spill)
        continue;
      note_spill_descriptor_requirements(private_requirements, *resources, *spill);
    }
    note_descriptor_requirements(descriptor_requirements, *resources);
    note_moi_sgpr_requirements(scalar_requirements, *resources, options);
    selected_candidates.push_back({std::move(planned_candidate), *resources, std::move(spill)});
  }
  if (selected_candidates.empty()) {
    result.warnings.emplace_back("ConSan MOI atomic record patch found no patchable atomics");
    return;
  }
  if (!apply_descriptor_requirements(patcher, code_object, active_bytes, result,
                                     descriptor_requirements, arch, result.errors) ||
      !apply_spill_descriptor_requirements(patcher, code_object, active_bytes, result,
                                           private_requirements, result.errors) ||
      !apply_sgpr_descriptor_requirements(patcher, result, code_object, scalar_requirements, arch,
                                          result.errors)) {
    return;
  }

  std::vector<uint8_t> new_text(old_text.begin(), old_text.end());
  DbiPatchPlacementPlanner placement_planner(arch, old_text.size());
  std::vector<ConSanPatchInfo> patches;
  const uint32_t record_count = static_cast<uint32_t>(selected_candidates.size());
  uint32_t record_index = 0;
  for (const PlannedAtomicRecord &planned : selected_candidates) {
    const AtomicRecordCandidate &candidate = planned.candidate;
    const ConSanAtomicSite &site = candidate.site;
    const uint64_t cave_text_offset = placement_planner.appended_end();
    const uint64_t return_text_offset = site.text_offset + site.size;
    ConSanOptions candidate_options = options;
    candidate_options.scratch_vgpr = planned.resources.base;
    auto cave_words = build_atomic_record_cave_words(
        active_bytes, candidate, candidate_options, planned.spill ? &*planned.spill : nullptr, arch,
        record_index, record_count, layout.atomic_records_offset, cave_text_offset,
        return_text_offset, result.errors);
    if (!cave_words)
      return;

    const auto placement =
        plan_prebuilt_appended_cave(placement_planner, site.text_offset, site.size, *cave_words,
                                    result.errors, "ConSan MOI atomic record patch");
    if (!placement || placement->body_offset != cave_text_offset ||
        new_text.size() != placement->body_offset) {
      result.errors.emplace_back("ConSan MOI atomic record patch has a stale cave mapping");
      return;
    }

    const auto fwd = compute_sopp_branch_simm16(placement->anchor_offset, placement->body_offset);
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
    info.scratch_vgpr = planned.resources.base;
    info.owner_descriptor_file_offsets = planned.resources.owner_descriptor_file_offsets;
    if (planned.spill) {
      info.spilled_vgpr_count = planned.spill->vgpr_count;
      info.required_private_segment_size = planned.spill->total_private_bytes;
    }
    patches.push_back(info);
    ++record_index;
  }

  if (!patcher.replace_text(new_text)) {
    result.errors.emplace_back("ConSan MOI atomic record patch could not grow .text");
    return;
  }
  result.elf_bytes = patcher.emit();
  if (!apply_spill_metadata_requirements(result.elf_bytes, result, private_requirements,
                                         result.errors)) {
    result.elf_bytes.clear();
    return;
  }
  result.patches.insert(result.patches.end(), patches.begin(), patches.end());
  result.modified = true;
}

void append_moi_operational_resource_plans(std::span<const uint8_t> bytes,
                                           const ConSanOptions &options, rj_code_arch_t arch,
                                           ConSanResult &result) {
  MoiResourcePlanningState state(bytes, arch, result);
  if (options.moi_track_barriers && options.moi_engine != ConSanMoiEngine::InlineShadow) {
    std::vector<BarrierRecordCandidate> candidates;
    for (const ConSanKernelInfo &kernel : result.kernels)
      append_barrier_epoch_candidates(kernel, candidates);
    for (const ConSanFunctionInfo &function : result.functions)
      append_barrier_epoch_candidates(function, candidates);
    for (size_t index = 0; index < candidates.size(); ++index) {
      const BarrierRecordCandidate &candidate = candidates[index];
      result.resource_plans.push_back(plan_moi_resource_site(
          state, options, ConSanResourceSiteKind::Barrier, index, candidate.site.text_offset,
          candidate.kernel_descriptor_file_offset, /*scratch_count=*/6u));
    }
  }
  if (options.moi_track_atomics) {
    std::vector<AtomicRecordCandidate> candidates;
    for (const ConSanKernelInfo &kernel : result.kernels)
      append_atomic_record_candidates(kernel, candidates);
    for (const ConSanFunctionInfo &function : result.functions)
      append_atomic_record_candidates(function, candidates);
    for (size_t index = 0; index < candidates.size(); ++index) {
      const AtomicRecordCandidate &candidate = candidates[index];
      result.resource_plans.push_back(plan_moi_resource_site(
          state, options, ConSanResourceSiteKind::Atomic, index, candidate.site.text_offset,
          candidate.kernel_descriptor_file_offset, /*scratch_count=*/3u));
    }
  }
}

void rebuild_moi_resource_plans(std::span<const uint8_t> bytes, const ConSanOptions &options,
                                rj_code_arch_t arch, ConSanResult &result) {
  append_moi_resource_plans(bytes, options, arch, result);
  append_moi_operational_resource_plans(bytes, options, arch, result);
}

void summarize_moi_resource_plans(ConSanResult &result) {
  result.resource_plan_summary = {};
  for (const ConSanCandidateResourcePlan &plan : result.resource_plans) {
    switch (plan.source) {
    case ConSanRegisterAllocationSource::Explicit:
      ++result.resource_plan_summary.explicit_plans;
      break;
    case ConSanRegisterAllocationSource::LivenessDead:
      ++result.resource_plan_summary.dead_plans;
      break;
    case ConSanRegisterAllocationSource::DescriptorGrowth:
      ++result.resource_plan_summary.descriptor_growth_plans;
      break;
    case ConSanRegisterAllocationSource::SpillRequired:
      ++result.resource_plan_summary.spill_plans;
      result.resource_plan_summary.planned_spill_slot_bytes +=
          static_cast<size_t>(plan.scratch_vgpr_count) * SpillManager::kSlotBytes;
      break;
    case ConSanRegisterAllocationSource::Unsupported:
      ++result.resource_plan_summary.unsupported_plans;
      break;
    }
  }
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.spilled_vgpr_count == 0)
      continue;
    ++result.resource_plan_summary.emitted_spill_patches;
    result.resource_plan_summary.emitted_spill_slot_bytes +=
        static_cast<size_t>(patch.spilled_vgpr_count) * SpillManager::kSlotBytes;
  }
}

} // namespace

bool consan_moi_supports_native_lds_record_replay_mnemonic(std::string_view mnemonic) {
  return is_single_range_native_lds_mnemonic(mnemonic) ||
         two_address_native_lds_offset_scale(mnemonic).has_value();
}

ConSanResult try_patch_consan_moi(ConSanResult result, const ConSanOptions &options,
                                  std::span<const uint8_t> code_object_bytes, rj_code_arch_t arch) {
  ConSanOptions effective_options = options;
  result.flavor = ConSanFlavor::Moi;
  result.moi_engine = effective_options.moi_engine;
  result.modified = false;
  result.elf_bytes.clear();
  result.moi_candidates.clear();
  for (const ConSanKernelInfo &kernel : result.kernels)
    append_moi_candidates(kernel, effective_options.flat_provenance_mode, result);
  for (const ConSanFunctionInfo &function : result.functions)
    append_moi_candidates(function, effective_options.flat_provenance_mode, result);
  rebuild_moi_resource_plans(code_object_bytes, effective_options, arch, result);
  if (configure_automatic_moi_owner_sgpr(effective_options, arch, result))
    rebuild_moi_resource_plans(code_object_bytes, effective_options, arch, result);
  if (configure_automatic_moi_exec_save_sgprs(effective_options, code_object_bytes, arch, result))
    rebuild_moi_resource_plans(code_object_bytes, effective_options, arch, result);
  if (configure_automatic_moi_identity_sgprs(effective_options, code_object_bytes, arch, result))
    rebuild_moi_resource_plans(code_object_bytes, effective_options, arch, result);
  if (configure_automatic_moi_persistent_vgprs(effective_options, code_object_bytes, arch, result))
    rebuild_moi_resource_plans(code_object_bytes, effective_options, arch, result);
  if (!result.resolved_moi_owner_vgpr)
    result.resolved_moi_owner_vgpr = effective_options.moi_owner_vgpr;
  if (!result.resolved_moi_epoch_vgpr)
    result.resolved_moi_epoch_vgpr = effective_options.moi_epoch_vgpr;
  for (uint32_t dimension = 0; dimension < 3; ++dimension) {
    if (!result.resolved_moi_workgroup_vgprs[dimension])
      result.resolved_moi_workgroup_vgprs[dimension] =
          effective_options.moi_workgroup_vgprs[dimension];
  }
  if (!result.resolved_moi_state_owner_sgpr)
    result.resolved_moi_state_owner_sgpr = effective_options.moi_state_owner_sgpr;
  if (!result.resolved_moi_state_epoch_sgpr)
    result.resolved_moi_state_epoch_sgpr = effective_options.moi_state_epoch_sgpr;
  for (uint32_t dimension = 0; dimension < 3; ++dimension) {
    if (!result.resolved_moi_workgroup_sgprs[dimension])
      result.resolved_moi_workgroup_sgprs[dimension] =
          effective_options.moi_workgroup_sgprs[dimension];
  }
  if (!result.resolved_moi_identity_sgpr)
    result.resolved_moi_identity_sgpr = effective_options.moi_identity_sgpr;
  if (!result.resolved_moi_exec_save_sgpr)
    result.resolved_moi_exec_save_sgpr = effective_options.moi_exec_save_sgpr;
  if (!result.resolved_moi_owner_sgpr)
    result.resolved_moi_owner_sgpr = effective_options.moi_owner_sgpr;
  if (effective_options.moi_report_buffer_address &&
      effective_options.moi_report_buffer_size < sizeof(ConSanMoiReportHeader)) {
    result.warnings.emplace_back("ConSan MOI report buffer is smaller than the report ABI header");
  }
  if (result.errors.empty() && effective_options.moi_engine == ConSanMoiEngine::Sampled)
    try_apply_direct_sampled_watchpoint_patch(code_object_bytes, effective_options, arch, result);
  if (result.errors.empty() && effective_options.moi_engine == ConSanMoiEngine::InlineShadow)
    try_apply_inline_shadow_patch(code_object_bytes, effective_options, arch, result);
  if (result.errors.empty() && effective_options.moi_engine == ConSanMoiEngine::RecordReplay)
    try_apply_first_light_access_record_patch(code_object_bytes, effective_options, arch, result);
  if (result.errors.empty())
    try_apply_barrier_epoch_patch(code_object_bytes, effective_options, arch, result);
  if (result.errors.empty())
    try_apply_inline_atomic_ordering_patch(code_object_bytes, effective_options, arch, result);
  if (result.errors.empty())
    try_apply_atomic_record_patch(code_object_bytes, effective_options, arch, result);
  if (result.errors.empty() &&
      (effective_options.moi_engine != ConSanMoiEngine::InlineShadow || result.modified))
    try_apply_owner_epoch_prologue_patch(code_object_bytes, effective_options, arch, result);
  summarize_moi_resource_plans(result);
  if (result.modified) {
    const auto patch_count = [&result](ConSanPatchKind kind) {
      return static_cast<uint32_t>(
          std::count_if(result.patches.begin(), result.patches.end(),
                        [kind](const ConSanPatchInfo &patch) { return patch.kind == kind; }));
    };
    if (patch_count(ConSanPatchKind::InlineMoiAccessRecordStore) != 0) {
      result.warnings.emplace_back(std::string("ConSan MOI ") +
                                   consan_moi_engine_name(effective_options.moi_engine) +
                                   " engine emitted a first-light access record probe");
    }
    if (patch_count(ConSanPatchKind::TrampolineMoiAccessRecordStore) != 0) {
      result.warnings.emplace_back(std::string("ConSan MOI ") +
                                   consan_moi_engine_name(effective_options.moi_engine) +
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
      result.warnings.emplace_back(result.moi_scalar_identity_automatic
                                       ? "ConSan MOI initialized scalar identity state with a "
                                         "kernel-entry prologue"
                                       : "ConSan MOI initialized owner/epoch VGPRs with a "
                                         "kernel-entry prologue");
    }
    if (patch_count(ConSanPatchKind::KernelEntryMoiPrivateEpochPrologue) != 0) {
      result.warnings.emplace_back(
          "ConSan MOI initialized private owner/epoch state with a kernel-entry prologue");
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
                                 consan_moi_engine_name(effective_options.moi_engine) +
                                 " engine is an inventory-only stub");
  }
  return result;
}

} // namespace rocjitsu
