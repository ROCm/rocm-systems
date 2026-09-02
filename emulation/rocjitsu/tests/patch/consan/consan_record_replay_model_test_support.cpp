// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_record_replay_model_test_support.h"
#include "consan_report_test_support.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <ranges>
#include <vector>

namespace rocjitsu {

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
    if (consan_moi_record_replay_atomic_event_is_unpublished(record))
      continue;
    const std::optional<ConSanMoiAtomicOutcome> outcome =
        consan_moi_record_replay_resolved_atomic_outcome(record);
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
    if (consan_moi_record_replay_fence_event_is_unpublished(record))
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

} // namespace rocjitsu
