// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_test_support.h"

namespace rocjitsu {
namespace {

TEST(ConSanMoi, RecordReplayCompactTraceCoalescesPcWorkgroupAndLaneMetadata) {
  std::array<ConSanMoiAccessRecord, 2> accesses{};
  for (ConSanMoiAccessRecord &record : accesses) {
    record.generation = 7;
    record.workgroup_x = 3;
    record.workgroup_y = 4;
    record.workgroup_z = 5;
    record.wave_id = 2;
    record.instruction_offset = 0x40;
    record.access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write);
    record.lds_byte_offset = 12;
    record.lds_byte_count = 4;
    record.epoch = 9;
    record.event_index = 20;
  }
  accesses[0].lane_mask = 0x1;
  accesses[1].lane_mask = 0x4;

  std::array<ConSanMoiBarrierRecord, 1> barriers{};
  barriers[0].generation = 7;
  barriers[0].workgroup_x = 1;
  barriers[0].wave_id = 6;
  barriers[0].lane_mask = 0xff;
  barriers[0].instruction_offset = 0x80;
  barriers[0].event_index = 10;

  std::array<ConSanMoiRecordReplayPcEntry, 4> dictionary{};
  std::array<ConSanMoiRecordReplayWorkgroupRun, 4> runs{};
  std::array<ConSanMoiRecordReplayCompactEvent, 4> events{};
  const ConSanMoiRecordReplayTraceHeader trace = consan_moi_compact_record_replay_trace(
      /*generation=*/7, /*dispatch_id=*/11, accesses, barriers,
      std::span<const ConSanMoiRecordReplayAtomicEvent>{}, dictionary, runs, events);

  EXPECT_EQ(trace.magic, kConSanMoiRecordReplayTraceMagic);
  EXPECT_EQ(trace.abi_version, kConSanMoiRecordReplayTraceAbiVersion);
  EXPECT_EQ(trace.header_size, sizeof(ConSanMoiRecordReplayTraceHeader));
  EXPECT_EQ(trace.flags, 0u);
  EXPECT_EQ(trace.generation, 7u);
  EXPECT_EQ(trace.dispatch_id, 11u);
  EXPECT_EQ(trace.dictionary_count, 2u);
  EXPECT_EQ(trace.workgroup_run_count, 2u);
  EXPECT_EQ(trace.event_count, 2u);
  EXPECT_EQ(trace.lane_coalesced_record_count, 1u);

  EXPECT_EQ(dictionary[0].kind, ConSanMoiRecordReplayEventKind::Barrier);
  EXPECT_EQ(dictionary[0].instruction_offset, 0x80u);
  EXPECT_EQ(dictionary[1].kind, ConSanMoiRecordReplayEventKind::Access);
  EXPECT_EQ(dictionary[1].instruction_offset, 0x40u);
  EXPECT_EQ(dictionary[1].operation, static_cast<uint16_t>(ConSanMoiShadowAccessKind::Write));
  EXPECT_EQ(runs[0].workgroup_x, 1u);
  EXPECT_EQ(runs[0].first_event, 0u);
  EXPECT_EQ(runs[0].event_count, 1u);
  EXPECT_EQ(runs[1].workgroup_x, 3u);
  EXPECT_EQ(runs[1].workgroup_y, 4u);
  EXPECT_EQ(runs[1].workgroup_z, 5u);
  EXPECT_EQ(runs[1].first_event, 1u);
  EXPECT_EQ(runs[1].event_count, 1u);
  EXPECT_EQ(events[1].pc_index, 1u);
  EXPECT_EQ(events[1].event_index, 20u);
  EXPECT_EQ(events[1].owner_id, 2u);
  EXPECT_EQ(events[1].epoch, 9u);
  EXPECT_EQ(events[1].lane_mask, 0x5u);
  EXPECT_EQ(events[1].payload, 12u | (uint64_t{4} << 32u));
}

TEST(ConSanMoi, RecordReplayCompactTraceRetainsAtomicStaticAndDynamicFields) {
  std::array<ConSanMoiRecordReplayAtomicEvent, 2> atomics{};
  atomics[0].generation = atomics[1].generation = 13;
  atomics[0].workgroup_x = atomics[1].workgroup_x = 2;
  atomics[0].owner_id = atomics[1].owner_id = 4;
  atomics[0].instruction_offset = atomics[1].instruction_offset = 0x90;
  atomics[0].atomic_address = 0x123456789abcdef0ull;
  atomics[1].atomic_address = 0xfedcba9876543210ull;
  atomics[0].event_index = 1;
  atomics[1].event_index = 2;
  atomics[0].kind = atomics[1].kind = ConSanMoiAtomicEventKind::AcquireRelease;
  atomics[0].scope = atomics[1].scope = 3;
  atomics[0].semantics = atomics[1].semantics = 7;
  atomics[0].operation = atomics[1].operation = ConSanMoiAtomicOperation::CompareExchange;
  atomics[0].outcome = ConSanMoiAtomicOutcome::Success;
  atomics[1].outcome = ConSanMoiAtomicOutcome::Failure;

  std::array<ConSanMoiRecordReplayPcEntry, 2> dictionary{};
  std::array<ConSanMoiRecordReplayWorkgroupRun, 2> runs{};
  std::array<ConSanMoiRecordReplayCompactEvent, 2> events{};
  const auto trace = consan_moi_compact_record_replay_trace(
      13, 17, std::span<const ConSanMoiAccessRecord>{}, std::span<const ConSanMoiBarrierRecord>{},
      atomics, dictionary, runs, events);

  EXPECT_EQ(trace.dictionary_count, 1u);
  EXPECT_EQ(trace.workgroup_run_count, 1u);
  EXPECT_EQ(trace.event_count, 2u);
  EXPECT_EQ(dictionary[0].kind, ConSanMoiRecordReplayEventKind::Atomic);
  EXPECT_EQ(dictionary[0].operation,
            static_cast<uint16_t>(ConSanMoiAtomicEventKind::AcquireRelease));
  EXPECT_EQ(dictionary[0].scope, 3u);
  EXPECT_EQ(dictionary[0].semantics, 7u);
  EXPECT_EQ(dictionary[0].atomic_operation, ConSanMoiAtomicOperation::CompareExchange);
  EXPECT_EQ(events[0].payload, 0x123456789abcdef0ull);
  EXPECT_EQ(events[1].payload, 0xfedcba9876543210ull);
  EXPECT_EQ(events[0].atomic_outcome, ConSanMoiAtomicOutcome::Success);
  EXPECT_EQ(events[1].atomic_outcome, ConSanMoiAtomicOutcome::Failure);
}

TEST(ConSanMoi, RecordReplayCompactTraceTypesFencesSeparatelyFromAtomics) {
  std::array<ConSanMoiRecordReplayFenceEvent, 2> fences{};
  fences[0].generation = fences[1].generation = 19;
  fences[0].workgroup_x = fences[1].workgroup_x = 3;
  fences[0].owner_id = 4;
  fences[1].owner_id = 5;
  fences[0].instruction_offset = fences[1].instruction_offset = 0xa0;
  fences[0].event_index = 7;
  fences[1].event_index = 8;
  fences[0].epoch = 2;
  fences[1].epoch = 3;
  fences[0].kind = fences[1].kind = ConSanMoiFenceEventKind::AcquireRelease;
  fences[0].scope = fences[1].scope = 2;
  fences[0].semantics = fences[1].semantics = 0x31;
  fences[0].communication_token = fences[1].communication_token = 0x123456789abcdef0ull;

  std::array<ConSanMoiRecordReplayPcEntry, 2> dictionary{};
  std::array<ConSanMoiRecordReplayWorkgroupRun, 2> runs{};
  std::array<ConSanMoiRecordReplayCompactEvent, 2> events{};
  const auto trace = consan_moi_compact_record_replay_trace(
      19, 23, std::span<const ConSanMoiAccessRecord>{}, std::span<const ConSanMoiBarrierRecord>{},
      std::span<const ConSanMoiRecordReplayAtomicEvent>{}, fences, dictionary, runs, events);

  ASSERT_EQ(trace.flags, 0u);
  ASSERT_EQ(trace.dictionary_count, 1u);
  ASSERT_EQ(trace.event_count, 2u);
  EXPECT_EQ(dictionary[0].kind, ConSanMoiRecordReplayEventKind::Fence);
  EXPECT_EQ(dictionary[0].operation,
            static_cast<uint16_t>(ConSanMoiFenceEventKind::AcquireRelease));
  EXPECT_EQ(dictionary[0].scope, 2u);
  EXPECT_EQ(dictionary[0].semantics, 0x31u);
  EXPECT_EQ(dictionary[0].atomic_operation, ConSanMoiAtomicOperation::Rmw);
  EXPECT_EQ(events[0].owner_id, 4u);
  EXPECT_EQ(events[0].epoch, 2u);
  EXPECT_EQ(events[0].payload, 0x123456789abcdef0ull);
  EXPECT_EQ(events[0].lane_mask, 0u);
  EXPECT_EQ(events[0].atomic_outcome, ConSanMoiAtomicOutcome::NotApplicable);
  EXPECT_EQ(events[1].owner_id, 5u);
  EXPECT_EQ(events[1].epoch, 3u);
}

TEST(ConSanMoi, RecordReplayCompactTraceRejectsFenceWithoutAssociation) {
  ConSanMoiRecordReplayFenceEvent fence;
  fence.kind = ConSanMoiFenceEventKind::Release;
  fence.scope = 1;
  fence.event_index = 1;
  std::array<ConSanMoiRecordReplayPcEntry, 1> dictionary{};
  std::array<ConSanMoiRecordReplayWorkgroupRun, 1> runs{};
  std::array<ConSanMoiRecordReplayCompactEvent, 1> events{};

  const auto trace = consan_moi_compact_record_replay_trace(
      1, 2, std::span<const ConSanMoiAccessRecord>{}, std::span<const ConSanMoiBarrierRecord>{},
      std::span<const ConSanMoiRecordReplayAtomicEvent>{},
      std::span<const ConSanMoiRecordReplayFenceEvent>(&fence, 1), dictionary, runs, events);
  EXPECT_EQ(trace.flags, kConSanMoiRecordReplayTraceRejectedInput);
  EXPECT_EQ(trace.rejected_event_count, 1u);
  EXPECT_EQ(trace.event_count, 0u);
}

TEST(ConSanMoi, RecordReplayCompactTraceCapacityFailureIsPrefixComplete) {
  std::array<ConSanMoiAccessRecord, 2> accesses{};
  accesses[0].instruction_offset = 0x10;
  accesses[1].instruction_offset = 0x20;
  accesses[0].event_index = 1;
  accesses[1].event_index = 2;
  accesses[0].lane_mask = accesses[1].lane_mask = 1;
  accesses[0].access_kind = accesses[1].access_kind =
      static_cast<uint32_t>(ConSanMoiShadowAccessKind::Read);

  std::array<ConSanMoiRecordReplayPcEntry, 1> dictionary{};
  std::array<ConSanMoiRecordReplayWorkgroupRun, 1> runs{};
  std::array<ConSanMoiRecordReplayCompactEvent, 2> events{};
  const auto trace = consan_moi_compact_record_replay_trace(
      5, 6, accesses, std::span<const ConSanMoiBarrierRecord>{},
      std::span<const ConSanMoiRecordReplayAtomicEvent>{}, dictionary, runs, events);

  EXPECT_EQ(trace.flags, kConSanMoiRecordReplayTraceOverflow);
  EXPECT_EQ(trace.dictionary_count, 1u);
  EXPECT_EQ(trace.workgroup_run_count, 1u);
  EXPECT_EQ(trace.event_count, 1u);
  EXPECT_EQ(trace.dropped_event_count, 1u);
  EXPECT_EQ(runs[0].event_count, 1u);
  EXPECT_EQ(dictionary[0].instruction_offset, 0x10u);
  EXPECT_EQ(events[0].event_index, 1u);
}

TEST(ConSanMoi, RecordReplayCompactTraceRejectsCrossGenerationInput) {
  std::array<ConSanMoiBarrierRecord, 2> barriers{};
  barriers[0].generation = 8;
  barriers[1].generation = 7;
  barriers[0].event_index = 1;
  barriers[1].event_index = 2;
  barriers[0].instruction_offset = barriers[1].instruction_offset = 0x40;
  std::array<ConSanMoiRecordReplayPcEntry, 1> dictionary{};
  std::array<ConSanMoiRecordReplayWorkgroupRun, 1> runs{};
  std::array<ConSanMoiRecordReplayCompactEvent, 1> events{};
  const auto trace = consan_moi_compact_record_replay_trace(
      7, 9, std::span<const ConSanMoiAccessRecord>{}, barriers,
      std::span<const ConSanMoiRecordReplayAtomicEvent>{}, dictionary, runs, events);

  EXPECT_EQ(trace.flags, kConSanMoiRecordReplayTraceRejectedInput);
  EXPECT_EQ(trace.rejected_event_count, 1u);
  EXPECT_EQ(trace.dropped_event_count, 0u);
  EXPECT_EQ(trace.event_count, 1u);
  EXPECT_EQ(events[0].event_index, 2u);
}

TEST(ConSanMoi, RecordReplayCapturePlansCompleteWorkgroupBarrierEpochs) {
  const ConSanMoiRecordReplayTraceHeader header = [] {
    ConSanMoiRecordReplayTraceHeader value;
    value.dictionary_count = 2;
    value.dictionary_capacity = 2;
    value.workgroup_run_count = 3;
    value.workgroup_run_capacity = 3;
    value.event_count = 4;
    value.event_capacity = 4;
    return value;
  }();
  const std::array<ConSanMoiRecordReplayPcEntry, 2> dictionary = {{
      {0x10, ConSanMoiRecordReplayEventKind::Access,
       static_cast<uint16_t>(ConSanMoiShadowAccessKind::Write), 0, 0},
      {0x20, ConSanMoiRecordReplayEventKind::Barrier, 0, 0, 0},
  }};
  const std::array<ConSanMoiRecordReplayWorkgroupRun, 3> runs = {{
      {1, 2, 3, 0, 2, 0},
      {9, 8, 7, 2, 1, 0},
      {1, 2, 3, 3, 1, 0},
  }};
  const std::array<ConSanMoiRecordReplayCompactEvent, 4> events = {{
      {0, 10, 0, 0, 1, 0},
      {1, 11, 0, 0, 1, 0},
      {0, 12, 0, 0, 1, 0},
      {0, 13, 0, 0, 1, 0},
  }};
  std::array<ConSanMoiRecordReplayCaptureWindow, 4> windows{};
  std::array<uint32_t, 4> assignments{};
  const ConSanMoiRecordReplayCaptureResult capture = consan_moi_plan_record_replay_capture(
      header, dictionary, runs, events,
      {/*workgroup_limit=*/2, /*epochs_per_workgroup_limit=*/2, /*event_budget=*/4}, windows,
      assignments);

  EXPECT_FALSE(capture.invalid_trace);
  EXPECT_EQ(capture.selected_workgroup_count, 2u);
  EXPECT_EQ(capture.selected_window_count, 3u);
  EXPECT_EQ(capture.selected_event_count, 4u);
  EXPECT_EQ(capture.omitted_event_count, 0u);
  EXPECT_EQ(assignments, (std::array<uint32_t, 4>{0, 0, 1, 2}));
  EXPECT_EQ(windows[0].workgroup_x, 1u);
  EXPECT_EQ(windows[0].epoch, 0u);
  EXPECT_EQ(windows[0].first_event_position, 0u);
  EXPECT_EQ(windows[0].last_event_position, 1u);
  EXPECT_EQ(windows[0].first_event_index, 10u);
  EXPECT_EQ(windows[0].last_event_index, 11u);
  EXPECT_EQ(windows[0].event_count, 2u);
  EXPECT_EQ(windows[2].workgroup_x, 1u);
  EXPECT_EQ(windows[2].epoch, 1u);
}

TEST(ConSanMoi, RecordReplayCaptureCoalescesConsecutiveBarrierArrivalsIntoOneEpoch) {
  ConSanMoiRecordReplayTraceHeader header;
  header.dictionary_count = header.dictionary_capacity = 2;
  header.workgroup_run_count = header.workgroup_run_capacity = 1;
  header.event_count = header.event_capacity = 4;
  const std::array<ConSanMoiRecordReplayPcEntry, 2> dictionary = {{
      {0x10, ConSanMoiRecordReplayEventKind::Access, 1, 0, 0},
      {0x20, ConSanMoiRecordReplayEventKind::Barrier, 0, 0, 0},
  }};
  const std::array<ConSanMoiRecordReplayWorkgroupRun, 1> runs = {{{0, 0, 0, 0, 4, 0}}};
  const std::array<ConSanMoiRecordReplayCompactEvent, 4> events = {{
      {0, 1, 0, 0, 1, 0},
      {1, 2, 0, 0, 1, 0},
      {1, 3, 1, 0, 2, 0},
      {0, 4, 0, 0, 1, 0},
  }};
  std::array<ConSanMoiRecordReplayCaptureWindow, 2> windows{};
  std::array<uint32_t, 4> assignments{};
  const auto capture = consan_moi_plan_record_replay_capture(header, dictionary, runs, events,
                                                             {1, 2, 4}, windows, assignments);

  EXPECT_EQ(capture.selected_window_count, 2u);
  EXPECT_EQ(windows[0].epoch, 0u);
  EXPECT_EQ(windows[0].event_count, 3u);
  EXPECT_EQ(windows[1].epoch, 1u);
  EXPECT_EQ(windows[1].event_count, 1u);
  EXPECT_EQ(assignments, (std::array<uint32_t, 4>{0, 0, 0, 1}));
}

TEST(ConSanMoi, RecordReplayCaptureDoesNotCoalesceDistinctAdjacentBarriers) {
  ConSanMoiRecordReplayTraceHeader header;
  header.dictionary_count = header.dictionary_capacity = 3;
  header.workgroup_run_count = header.workgroup_run_capacity = 1;
  header.event_count = header.event_capacity = 4;
  const std::array<ConSanMoiRecordReplayPcEntry, 3> dictionary = {{
      {0x10, ConSanMoiRecordReplayEventKind::Access, 1, 0, 0},
      {0x20, ConSanMoiRecordReplayEventKind::Barrier, 0, 0, 0},
      {0x30, ConSanMoiRecordReplayEventKind::Barrier, 0, 0, 0},
  }};
  const std::array<ConSanMoiRecordReplayWorkgroupRun, 1> runs = {{{0, 0, 0, 0, 4, 0}}};
  const std::array<ConSanMoiRecordReplayCompactEvent, 4> events = {{
      {0, 1, 0, 0, 1, 0},
      {1, 2, 0, 0, 1, 0},
      {2, 3, 0, 1, 1, 0},
      {0, 4, 0, 2, 1, 0},
  }};
  std::array<ConSanMoiRecordReplayCaptureWindow, 3> windows{};
  std::array<uint32_t, 4> assignments{};

  const auto capture = consan_moi_plan_record_replay_capture(header, dictionary, runs, events,
                                                             {1, 3, 4}, windows, assignments);
  ASSERT_FALSE(capture.invalid_trace);
  EXPECT_EQ(capture.selected_window_count, 3u);
  EXPECT_EQ(assignments, (std::array<uint32_t, 4>{0, 0, 1, 2}));
  EXPECT_EQ(windows[0].epoch, 0u);
  EXPECT_EQ(windows[1].epoch, 1u);
  EXPECT_EQ(windows[2].epoch, 2u);
}

TEST(ConSanMoi, RecordReplayCaptureNeverPartiallyCommitsAnEpochAtEventBudget) {
  ConSanMoiRecordReplayTraceHeader header;
  header.dictionary_count = header.dictionary_capacity = 1;
  header.workgroup_run_count = header.workgroup_run_capacity = 1;
  header.event_count = header.event_capacity = 3;
  const std::array<ConSanMoiRecordReplayPcEntry, 1> dictionary = {
      {{0x10, ConSanMoiRecordReplayEventKind::Access, 1, 0, 0}}};
  const std::array<ConSanMoiRecordReplayWorkgroupRun, 1> runs = {{{0, 0, 0, 0, 3, 0}}};
  const std::array<ConSanMoiRecordReplayCompactEvent, 3> events = {{
      {0, 1, 0, 0, 1, 0},
      {0, 2, 0, 0, 1, 0},
      {0, 3, 0, 0, 1, 0},
  }};
  std::array<ConSanMoiRecordReplayCaptureWindow, 1> windows{};
  std::array<uint32_t, 3> assignments{};
  const auto capture = consan_moi_plan_record_replay_capture(header, dictionary, runs, events,
                                                             {1, 1, 2}, windows, assignments);

  EXPECT_TRUE(capture.event_budget_exhausted);
  EXPECT_EQ(capture.selected_window_count, 0u);
  EXPECT_EQ(capture.selected_event_count, 0u);
  EXPECT_EQ(capture.omitted_event_count, 3u);
  EXPECT_EQ(assignments, (std::array<uint32_t, 3>{kConSanMoiRecordReplayUncapturedEvent,
                                                  kConSanMoiRecordReplayUncapturedEvent,
                                                  kConSanMoiRecordReplayUncapturedEvent}));
}

TEST(ConSanMoi, RecordReplayCaptureReportsWorkgroupAndEpochLimitOmissions) {
  ConSanMoiRecordReplayTraceHeader header;
  header.dictionary_count = header.dictionary_capacity = 2;
  header.workgroup_run_count = header.workgroup_run_capacity = 2;
  header.event_count = header.event_capacity = 4;
  const std::array<ConSanMoiRecordReplayPcEntry, 2> dictionary = {{
      {0x10, ConSanMoiRecordReplayEventKind::Access, 1, 0, 0},
      {0x20, ConSanMoiRecordReplayEventKind::Barrier, 0, 0, 0},
  }};
  const std::array<ConSanMoiRecordReplayWorkgroupRun, 2> runs = {{
      {0, 0, 0, 0, 3, 0},
      {1, 0, 0, 3, 1, 0},
  }};
  const std::array<ConSanMoiRecordReplayCompactEvent, 4> events = {{
      {0, 1, 0, 0, 1, 0},
      {1, 2, 0, 0, 1, 0},
      {0, 3, 0, 0, 1, 0},
      {0, 4, 0, 0, 1, 0},
  }};
  std::array<ConSanMoiRecordReplayCaptureWindow, 4> windows{};
  std::array<uint32_t, 4> assignments{};
  const auto capture = consan_moi_plan_record_replay_capture(header, dictionary, runs, events,
                                                             {1, 1, 10}, windows, assignments);

  EXPECT_TRUE(capture.workgroup_limit_exhausted);
  EXPECT_TRUE(capture.epoch_limit_exhausted);
  EXPECT_EQ(capture.selected_workgroup_count, 1u);
  EXPECT_EQ(capture.selected_window_count, 1u);
  EXPECT_EQ(capture.selected_event_count, 2u);
  EXPECT_EQ(capture.omitted_event_count, 2u);
  EXPECT_EQ(assignments, (std::array<uint32_t, 4>{0, 0, kConSanMoiRecordReplayUncapturedEvent,
                                                  kConSanMoiRecordReplayUncapturedEvent}));
}

TEST(ConSanMoi, RecordReplayCaptureRejectsMalformedRunCoverageWithoutSelection) {
  ConSanMoiRecordReplayTraceHeader header;
  header.dictionary_count = header.dictionary_capacity = 1;
  header.workgroup_run_count = header.workgroup_run_capacity = 1;
  header.event_count = header.event_capacity = 1;
  const std::array<ConSanMoiRecordReplayPcEntry, 1> dictionary = {
      {{0x10, ConSanMoiRecordReplayEventKind::Access, 1, 0, 0}}};
  const std::array<ConSanMoiRecordReplayWorkgroupRun, 1> runs = {{{0, 0, 0, 1, 1, 0}}};
  const std::array<ConSanMoiRecordReplayCompactEvent, 1> events = {{{0, 1, 0, 0, 1, 0}}};
  std::array<ConSanMoiRecordReplayCaptureWindow, 1> windows{};
  std::array<uint32_t, 1> assignments = {42};
  const auto capture = consan_moi_plan_record_replay_capture(header, dictionary, runs, events,
                                                             {1, 1, 1}, windows, assignments);

  EXPECT_TRUE(capture.invalid_trace);
  EXPECT_EQ(capture.selected_event_count, 0u);
  EXPECT_EQ(assignments[0], kConSanMoiRecordReplayUncapturedEvent);

  header.flags = kConSanMoiRecordReplayTraceOverflow;
  const std::array<ConSanMoiRecordReplayWorkgroupRun, 1> complete_runs = {{{0, 0, 0, 0, 1, 0}}};
  assignments[0] = 42;
  const auto truncated = consan_moi_plan_record_replay_capture(
      header, dictionary, complete_runs, events, {1, 1, 1}, windows, assignments);
  EXPECT_TRUE(truncated.invalid_trace);
  EXPECT_EQ(assignments[0], kConSanMoiRecordReplayUncapturedEvent);
}

TEST(ConSanMoi, RecordReplayCaptureReplaysAtomicOrderingAtTheExactAddress) {
  ConSanMoiRecordReplayTraceHeader header;
  header.generation = 7;
  header.dispatch_id = 9;
  header.dictionary_count = header.dictionary_capacity = 4;
  header.workgroup_run_count = header.workgroup_run_capacity = 1;
  header.event_count = header.event_capacity = 4;
  const std::array<ConSanMoiRecordReplayPcEntry, 4> dictionary = {{
      {0x10, ConSanMoiRecordReplayEventKind::Access,
       static_cast<uint16_t>(ConSanMoiShadowAccessKind::Write), 0, 0},
      {0x20, ConSanMoiRecordReplayEventKind::Atomic,
       static_cast<uint16_t>(ConSanMoiAtomicEventKind::Release), 2, 1},
      {0x30, ConSanMoiRecordReplayEventKind::Atomic,
       static_cast<uint16_t>(ConSanMoiAtomicEventKind::Acquire), 2, 2},
      {0x40, ConSanMoiRecordReplayEventKind::Access,
       static_cast<uint16_t>(ConSanMoiShadowAccessKind::Read), 0, 0},
  }};
  const std::array<ConSanMoiRecordReplayWorkgroupRun, 1> runs = {{{1, 2, 3, 0, 4, 0}}};
  constexpr uint64_t kCell = uint64_t{4} << 32u;
  constexpr uint64_t kAtomicAddress = 0x100040ull;
  const std::array<ConSanMoiRecordReplayCompactEvent, 4> events = {{
      {0, 1, 0, 0, 1, kCell},
      {1, 2, 0, 0, 0, kAtomicAddress},
      {2, 3, 1, 0, 0, kAtomicAddress},
      {3, 4, 1, 0, 2, kCell},
  }};
  const std::array<ConSanMoiRecordReplayCaptureWindow, 1> windows = {{
      {1, 2, 3, 0, 0, 3, 1, 4, 4, 0},
  }};
  const std::array<uint32_t, 4> assignments = {0, 0, 0, 0};
  std::array<ConSanMoiDiagnosticRecord, 2> diagnostics{};
  std::array<uint64_t, 4> shadow{};

  const auto ordered = consan_moi_replay_record_replay_capture(
      header, dictionary, runs, events, windows, assignments, diagnostics, shadow);

  EXPECT_FALSE(ordered.invalid_capture);
  EXPECT_EQ(ordered.selected_event_count, 4u);
  EXPECT_EQ(ordered.replay.processed_access_count, 2u);
  EXPECT_EQ(ordered.replay.processed_atomic_count, 2u);
  EXPECT_FALSE(ordered.replay.conflict);
  EXPECT_EQ(ordered.replay.emitted_diagnostic_count, 0u);

  auto wrong_address_events = events;
  wrong_address_events[2].payload += 8;
  diagnostics = {};
  shadow = {};
  const auto unordered = consan_moi_replay_record_replay_capture(
      header, dictionary, runs, wrong_address_events, windows, assignments, diagnostics, shadow);
  EXPECT_FALSE(unordered.invalid_capture);
  EXPECT_TRUE(unordered.replay.conflict);
  EXPECT_EQ(unordered.replay.emitted_diagnostic_count, 1u);

  auto workgroup_scope_dictionary = dictionary;
  workgroup_scope_dictionary[1].scope = 1;
  workgroup_scope_dictionary[2].scope = 1;
  diagnostics = {};
  shadow = {};
  const auto workgroup_scope = consan_moi_replay_record_replay_capture(
      header, workgroup_scope_dictionary, runs, events, windows, assignments, diagnostics, shadow);
  EXPECT_FALSE(workgroup_scope.invalid_capture);
  EXPECT_FALSE(workgroup_scope.replay.conflict);

  auto wave_scope_dictionary = dictionary;
  wave_scope_dictionary[1].scope = 0;
  diagnostics = {};
  shadow = {};
  const auto wave_scope = consan_moi_replay_record_replay_capture(
      header, wave_scope_dictionary, runs, events, windows, assignments, diagnostics, shadow);
  EXPECT_FALSE(wave_scope.invalid_capture);
  EXPECT_EQ(wave_scope.replay.processed_atomic_count, 2u);
  EXPECT_TRUE(wave_scope.replay.conflict);
  EXPECT_EQ(wave_scope.replay.emitted_diagnostic_count, 1u);

  auto invalid_scope_dictionary = dictionary;
  invalid_scope_dictionary[2].scope = 4;
  shadow = {0x55, 0, 0, 0};
  const auto rejected = consan_moi_replay_record_replay_capture(
      header, invalid_scope_dictionary, runs, events, windows, assignments, diagnostics, shadow);
  EXPECT_TRUE(rejected.invalid_capture);
  EXPECT_EQ(shadow[0], 0x55u);
}

TEST(ConSanMoi, RecordReplayCaptureReplaysExplicitAddressedFenceEdges) {
  ConSanMoiRecordReplayTraceHeader header;
  header.generation = 11;
  header.dictionary_count = header.dictionary_capacity = 4;
  header.workgroup_run_count = header.workgroup_run_capacity = 1;
  header.event_count = header.event_capacity = 4;
  const std::array<ConSanMoiRecordReplayPcEntry, 4> dictionary = {{
      {0x10, ConSanMoiRecordReplayEventKind::Access,
       static_cast<uint16_t>(ConSanMoiShadowAccessKind::Write), 0, 0},
      {0x20, ConSanMoiRecordReplayEventKind::Fence,
       static_cast<uint16_t>(ConSanMoiFenceEventKind::Release), 1, 0x31,
       ConSanMoiAtomicOperation::Rmw},
      {0x30, ConSanMoiRecordReplayEventKind::Fence,
       static_cast<uint16_t>(ConSanMoiFenceEventKind::Acquire), 1, 0x32,
       ConSanMoiAtomicOperation::Rmw},
      {0x40, ConSanMoiRecordReplayEventKind::Access,
       static_cast<uint16_t>(ConSanMoiShadowAccessKind::Read), 0, 0},
  }};
  const std::array<ConSanMoiRecordReplayWorkgroupRun, 1> runs = {{{0, 0, 0, 0, 4, 0}}};
  constexpr uint64_t kCell = uint64_t{4} << 32u;
  constexpr uint64_t kAddress = 0x100040;
  const std::array<ConSanMoiRecordReplayCompactEvent, 4> events = {{
      {0, 1, 0, 0, 1, kCell},
      {1, 2, 0, 0, 0, kAddress},
      {2, 3, 1, 0, 0, kAddress},
      {3, 4, 1, 0, 2, kCell},
  }};
  const std::array<ConSanMoiRecordReplayCaptureWindow, 1> windows = {{
      {0, 0, 0, 0, 0, 3, 1, 4, 4, 0},
  }};
  const std::array<uint32_t, 4> assignments = {0, 0, 0, 0};
  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
  std::array<uint64_t, 1> shadow{};

  const auto ordered = consan_moi_replay_record_replay_capture(
      header, dictionary, runs, events, windows, assignments, diagnostics, shadow);
  ASSERT_FALSE(ordered.invalid_capture);
  EXPECT_EQ(ordered.replay.processed_fence_count, 2u);
  EXPECT_EQ(ordered.replay.unsupported_fence_count, 0u);
  EXPECT_FALSE(ordered.replay.conflict);

  auto wrong_address_events = events;
  wrong_address_events[2].payload += 8;
  diagnostics = {};
  diagnostics[0].kind = 81;
  shadow = {0x4567};
  const auto unordered = consan_moi_replay_record_replay_capture(
      header, dictionary, runs, wrong_address_events, windows, assignments, diagnostics, shadow);
  EXPECT_TRUE(unordered.invalid_capture);
  EXPECT_EQ(diagnostics[0].kind, 81u);
  EXPECT_EQ(shadow[0], 0x4567u);
}

TEST(ConSanMoi, RecordReplayCaptureDoesNotInferFenceEdgesFromAtomicSemanticsBits) {
  ConSanMoiRecordReplayTraceHeader header;
  header.generation = 11;
  header.dictionary_count = header.dictionary_capacity = 4;
  header.workgroup_run_count = header.workgroup_run_capacity = 1;
  header.event_count = header.event_capacity = 4;
  const std::array<ConSanMoiRecordReplayPcEntry, 4> dictionary = {{
      {0x10, ConSanMoiRecordReplayEventKind::Access,
       static_cast<uint16_t>(ConSanMoiShadowAccessKind::Write), 0, 0},
      {0x20, ConSanMoiRecordReplayEventKind::Atomic,
       static_cast<uint16_t>(ConSanMoiAtomicEventKind::Release), 1, 0x31},
      {0x30, ConSanMoiRecordReplayEventKind::Atomic,
       static_cast<uint16_t>(ConSanMoiAtomicEventKind::Acquire), 1, 0x31},
      {0x40, ConSanMoiRecordReplayEventKind::Access,
       static_cast<uint16_t>(ConSanMoiShadowAccessKind::Read), 0, 0},
  }};
  const std::array<ConSanMoiRecordReplayWorkgroupRun, 1> runs = {{{0, 0, 0, 0, 4, 0}}};
  constexpr uint64_t kCell = uint64_t{4} << 32u;
  const std::array<ConSanMoiRecordReplayCompactEvent, 4> events = {{
      {0, 1, 0, 0, 1, kCell},
      {1, 2, 0, 0, 0, 0x1000},
      {2, 3, 1, 0, 0, 0x2000},
      {3, 4, 1, 0, 2, kCell},
  }};
  const std::array<ConSanMoiRecordReplayCaptureWindow, 1> windows = {{
      {0, 0, 0, 0, 0, 3, 1, 4, 4, 0},
  }};
  const std::array<uint32_t, 4> assignments = {0, 0, 0, 0};
  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
  std::array<uint64_t, 1> shadow{};

  const auto replay = consan_moi_replay_record_replay_capture(
      header, dictionary, runs, events, windows, assignments, diagnostics, shadow);
  ASSERT_FALSE(replay.invalid_capture);
  EXPECT_EQ(replay.replay.processed_fence_count, 0u);
  EXPECT_TRUE(replay.replay.conflict);
}

TEST(ConSanMoi, RecordReplayCaptureRejectsAddresslessFenceBeforeMutation) {
  ConSanMoiRecordReplayTraceHeader header;
  header.dictionary_count = header.dictionary_capacity = 1;
  header.workgroup_run_count = header.workgroup_run_capacity = 1;
  header.event_count = header.event_capacity = 1;
  const std::array<ConSanMoiRecordReplayPcEntry, 1> dictionary = {{
      {0x20, ConSanMoiRecordReplayEventKind::Fence,
       static_cast<uint16_t>(ConSanMoiFenceEventKind::AcquireRelease), 1, 0x31,
       ConSanMoiAtomicOperation::Rmw},
  }};
  const std::array<ConSanMoiRecordReplayWorkgroupRun, 1> runs = {{{0, 0, 0, 0, 1, 0}}};
  const std::array<ConSanMoiRecordReplayCompactEvent, 1> addressless = {{{0, 1, 0, 0, 0, 0}}};
  const std::array<ConSanMoiRecordReplayCaptureWindow, 1> windows = {{
      {0, 0, 0, 0, 0, 0, 1, 1, 1, 0},
  }};
  const std::array<uint32_t, 1> assignments = {0};
  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
  diagnostics[0].kind = 71;
  std::array<uint64_t, 1> shadow = {0x1234};

  const auto replay = consan_moi_replay_record_replay_capture(
      header, dictionary, runs, addressless, windows, assignments, diagnostics, shadow);
  EXPECT_TRUE(replay.invalid_capture);
  EXPECT_EQ(replay.selected_event_count, 1u);
  EXPECT_EQ(diagnostics[0].kind, 71u);
  EXPECT_EQ(shadow[0], 0x1234u);
}

TEST(ConSanMoi, RecordReplayCaptureReplaysBarrierEpochsAcrossACompleteWindowBoundary) {
  ConSanMoiRecordReplayTraceHeader header;
  header.generation = 5;
  header.dictionary_count = header.dictionary_capacity = 3;
  header.workgroup_run_count = header.workgroup_run_capacity = 1;
  header.event_count = header.event_capacity = 4;
  const std::array<ConSanMoiRecordReplayPcEntry, 3> dictionary = {{
      {0x10, ConSanMoiRecordReplayEventKind::Access,
       static_cast<uint16_t>(ConSanMoiShadowAccessKind::Write), 0, 0},
      {0x20, ConSanMoiRecordReplayEventKind::Barrier, 0, 0, 0},
      {0x30, ConSanMoiRecordReplayEventKind::Access,
       static_cast<uint16_t>(ConSanMoiShadowAccessKind::Read), 0, 0},
  }};
  const std::array<ConSanMoiRecordReplayWorkgroupRun, 1> runs = {{{0, 0, 0, 0, 4, 0}}};
  constexpr uint64_t kCell = uint64_t{4} << 32u;
  const std::array<ConSanMoiRecordReplayCompactEvent, 4> events = {{
      {0, 1, 0, 0, 1, kCell},
      {1, 2, 0, 0, 1, 0},
      {1, 3, 1, 0, 2, 0},
      {2, 4, 1, 0, 2, kCell},
  }};
  const std::array<ConSanMoiRecordReplayCaptureWindow, 2> windows = {{
      {0, 0, 0, 0, 0, 2, 1, 3, 3, 0},
      {0, 0, 0, 1, 3, 3, 4, 4, 1, 0},
  }};
  const std::array<uint32_t, 4> assignments = {0, 0, 0, 1};
  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
  std::array<uint64_t, 4> shadow{};

  const auto replay = consan_moi_replay_record_replay_capture(
      header, dictionary, runs, events, windows, assignments, diagnostics, shadow);

  EXPECT_FALSE(replay.invalid_capture);
  EXPECT_EQ(replay.replay.processed_barrier_count, 2u);
  EXPECT_FALSE(replay.replay.conflict);
}

TEST(ConSanMoi, RecordReplayCaptureReplaysTypedFenceEdgesNotAtomicSemanticsBits) {
  ConSanMoiRecordReplayTraceHeader header;
  header.generation = 11;
  header.dictionary_count = header.dictionary_capacity = 4;
  header.workgroup_run_count = header.workgroup_run_capacity = 1;
  header.event_count = header.event_capacity = 4;
  const std::array<ConSanMoiRecordReplayPcEntry, 4> dictionary = {{
      {0x10, ConSanMoiRecordReplayEventKind::Access,
       static_cast<uint16_t>(ConSanMoiShadowAccessKind::Write), 0, 0},
      {0x20, ConSanMoiRecordReplayEventKind::Fence,
       static_cast<uint16_t>(ConSanMoiFenceEventKind::Release), 1, 0x31},
      {0x30, ConSanMoiRecordReplayEventKind::Fence,
       static_cast<uint16_t>(ConSanMoiFenceEventKind::Acquire), 1, 0x32},
      {0x40, ConSanMoiRecordReplayEventKind::Access,
       static_cast<uint16_t>(ConSanMoiShadowAccessKind::Read), 0, 0},
  }};
  const std::array<ConSanMoiRecordReplayWorkgroupRun, 1> runs = {{{0, 0, 0, 0, 4, 0}}};
  constexpr uint64_t kCell = uint64_t{4} << 32u;
  const std::array<ConSanMoiRecordReplayCompactEvent, 4> events = {{
      {0, 1, 0, 0, 1, kCell},
      {1, 2, 0, 0, 0, 0xfeed},
      {2, 3, 1, 0, 0, 0xfeed},
      {3, 4, 1, 0, 2, kCell},
  }};
  const std::array<ConSanMoiRecordReplayCaptureWindow, 1> windows = {{
      {0, 0, 0, 0, 0, 3, 1, 4, 4, 0},
  }};
  const std::array<uint32_t, 4> assignments = {0, 0, 0, 0};
  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
  std::array<uint64_t, 1> shadow{};

  const auto fenced = consan_moi_replay_record_replay_capture(
      header, dictionary, runs, events, windows, assignments, diagnostics, shadow);
  ASSERT_FALSE(fenced.invalid_capture);
  EXPECT_EQ(fenced.replay.processed_fence_count, 2u);
  EXPECT_EQ(fenced.replay.unsupported_fence_count, 0u);
  EXPECT_FALSE(fenced.replay.conflict);

  auto unrelated_events = events;
  unrelated_events[2].payload = 0xbeef;
  diagnostics[0].kind = 83;
  shadow[0] = 0x9876;
  const auto unrelated = consan_moi_replay_record_replay_capture(
      header, dictionary, runs, unrelated_events, windows, assignments, diagnostics, shadow);
  EXPECT_TRUE(unrelated.invalid_capture);
  EXPECT_EQ(diagnostics[0].kind, 83u);
  EXPECT_EQ(shadow[0], 0x9876u);

  auto atomic_dictionary = dictionary;
  atomic_dictionary[1] = {0x20, ConSanMoiRecordReplayEventKind::Atomic,
                          static_cast<uint16_t>(ConSanMoiAtomicEventKind::Release), 1, 0x31};
  atomic_dictionary[2] = {0x30, ConSanMoiRecordReplayEventKind::Atomic,
                          static_cast<uint16_t>(ConSanMoiAtomicEventKind::Acquire), 1, 0x32};
  auto distinct_address_events = events;
  distinct_address_events[1].payload = 0x1000;
  distinct_address_events[2].payload = 0x2000;
  diagnostics = {};
  shadow = {};
  const auto raw_atomic_semantics = consan_moi_replay_record_replay_capture(
      header, atomic_dictionary, runs, distinct_address_events, windows, assignments, diagnostics,
      shadow);
  ASSERT_FALSE(raw_atomic_semantics.invalid_capture);
  EXPECT_EQ(raw_atomic_semantics.replay.processed_fence_count, 0u);
  EXPECT_TRUE(raw_atomic_semantics.replay.conflict);
}

TEST(ConSanMoi, RecordReplayCaptureRejectsMalformedFenceBeforeMutation) {
  ConSanMoiRecordReplayTraceHeader header;
  header.dictionary_count = header.dictionary_capacity = 1;
  header.workgroup_run_count = header.workgroup_run_capacity = 1;
  header.event_count = header.event_capacity = 1;
  const std::array<ConSanMoiRecordReplayPcEntry, 1> dictionary = {{
      {0x20, ConSanMoiRecordReplayEventKind::Fence,
       static_cast<uint16_t>(ConSanMoiFenceEventKind::AcquireRelease), 1, 0x31},
  }};
  const std::array<ConSanMoiRecordReplayWorkgroupRun, 1> runs = {{{0, 0, 0, 0, 1, 0}}};
  const std::array<ConSanMoiRecordReplayCompactEvent, 1> events = {{{0, 1, 0, 0, 0, 0}}};
  const std::array<ConSanMoiRecordReplayCaptureWindow, 1> windows = {{
      {0, 0, 0, 0, 0, 0, 1, 1, 1, 0},
  }};
  const std::array<uint32_t, 1> assignments = {0};
  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
  diagnostics[0].kind = 71;
  std::array<uint64_t, 1> shadow = {0x1234};

  const auto replay = consan_moi_replay_record_replay_capture(
      header, dictionary, runs, events, windows, assignments, diagnostics, shadow);
  EXPECT_TRUE(replay.invalid_capture);
  EXPECT_EQ(replay.selected_event_count, 1u);
  EXPECT_EQ(diagnostics[0].kind, 71u);
  EXPECT_EQ(shadow[0], 0x1234u);
}

TEST(ConSanMoi, RecordReplayCaptureRejectsPartialBarrierParticipantsBeforeMutation) {
  ConSanMoiRecordReplayTraceHeader header;
  header.dictionary_count = header.dictionary_capacity = 3;
  header.workgroup_run_count = header.workgroup_run_capacity = 1;
  header.event_count = header.event_capacity = 3;
  const std::array<ConSanMoiRecordReplayPcEntry, 3> dictionary = {{
      {0x10, ConSanMoiRecordReplayEventKind::Access,
       static_cast<uint16_t>(ConSanMoiShadowAccessKind::Write), 0, 0},
      {0x20, ConSanMoiRecordReplayEventKind::Barrier, 0, 0, 0},
      {0x30, ConSanMoiRecordReplayEventKind::Access,
       static_cast<uint16_t>(ConSanMoiShadowAccessKind::Read), 0, 0},
  }};
  const std::array<ConSanMoiRecordReplayWorkgroupRun, 1> runs = {{{0, 0, 0, 0, 3, 0}}};
  constexpr uint64_t kCell = uint64_t{4} << 32u;
  const std::array<ConSanMoiRecordReplayCompactEvent, 3> events = {{
      {0, 1, 0, 0, 1, kCell},
      {1, 2, 0, 0, 1, 0},
      {2, 3, 1, 1, 2, kCell},
  }};
  const std::array<ConSanMoiRecordReplayCaptureWindow, 2> windows = {{
      {0, 0, 0, 0, 0, 1, 1, 2, 2, 0},
      {0, 0, 0, 1, 2, 2, 3, 3, 1, 0},
  }};
  const std::array<uint32_t, 3> assignments = {0, 0, 1};
  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
  diagnostics[0].kind = 73;
  std::array<uint64_t, 1> shadow = {0x5678};

  const auto replay = consan_moi_replay_record_replay_capture(
      header, dictionary, runs, events, windows, assignments, diagnostics, shadow);
  EXPECT_TRUE(replay.invalid_capture);
  EXPECT_EQ(replay.selected_event_count, 0u);
  EXPECT_EQ(diagnostics[0].kind, 73u);
  EXPECT_EQ(shadow[0], 0x5678u);
}

TEST(ConSanMoi, RecordReplayCaptureRejectsDriftBeforeMutatingReplayOutputs) {
  ConSanMoiRecordReplayTraceHeader header;
  header.dictionary_count = header.dictionary_capacity = 1;
  header.workgroup_run_count = header.workgroup_run_capacity = 1;
  header.event_count = header.event_capacity = 1;
  const std::array<ConSanMoiRecordReplayPcEntry, 1> dictionary = {{
      {0x10, ConSanMoiRecordReplayEventKind::Access,
       static_cast<uint16_t>(ConSanMoiShadowAccessKind::Write), 0, 0},
  }};
  const std::array<ConSanMoiRecordReplayWorkgroupRun, 1> runs = {{{0, 0, 0, 0, 1, 0}}};
  const std::array<ConSanMoiRecordReplayCompactEvent, 1> events = {{
      {0, 1, 0, 0, 1, uint64_t{4} << 32u},
  }};
  const std::array<ConSanMoiRecordReplayCaptureWindow, 1> drifted_windows = {{
      {0, 0, 0, 0, 0, 0, 1, 1, 2, 0},
  }};
  const std::array<uint32_t, 1> assignments = {0};
  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
  diagnostics[0].kind = 99;
  std::array<uint64_t, 1> shadow = {0x1234};

  const auto replay = consan_moi_replay_record_replay_capture(
      header, dictionary, runs, events, drifted_windows, assignments, diagnostics, shadow);

  EXPECT_TRUE(replay.invalid_capture);
  EXPECT_EQ(replay.selected_event_count, 0u);
  EXPECT_EQ(shadow[0], 0x1234u);
  EXPECT_EQ(diagnostics[0].kind, 99u);
}

TEST(ConSanMoi, RecordReplayCaptureRejectsPartialBarrierEpochMembership) {
  ConSanMoiRecordReplayTraceHeader header;
  header.dictionary_count = header.dictionary_capacity = 2;
  header.workgroup_run_count = header.workgroup_run_capacity = 1;
  header.event_count = header.event_capacity = 3;
  const std::array<ConSanMoiRecordReplayPcEntry, 2> dictionary = {{
      {0x10, ConSanMoiRecordReplayEventKind::Access,
       static_cast<uint16_t>(ConSanMoiShadowAccessKind::Write), 0, 0},
      {0x20, ConSanMoiRecordReplayEventKind::Barrier, 0, 0, 0},
  }};
  const std::array<ConSanMoiRecordReplayWorkgroupRun, 1> runs = {{{0, 0, 0, 0, 3, 0}}};
  const std::array<ConSanMoiRecordReplayCompactEvent, 3> events = {{
      {0, 1, 0, 0, 1, uint64_t{4} << 32u},
      {0, 2, 0, 0, 2, uint64_t{4} << 32u},
      {1, 3, 0, 0, 3, 0},
  }};
  // This self-consistent-looking window omits the middle event from the same
  // workgroup and barrier epoch. RR3 must derive completeness rather than
  // accepting the caller's count and membership at face value.
  const std::array<ConSanMoiRecordReplayCaptureWindow, 1> partial_windows = {{
      {0, 0, 0, 0, 0, 2, 1, 3, 2, 0},
  }};
  const std::array<uint32_t, 3> partial_membership = {0, kConSanMoiRecordReplayUncapturedEvent, 0};
  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
  diagnostics[0].kind = 77;
  std::array<uint64_t, 1> shadow = {0x5678};

  const auto replay = consan_moi_replay_record_replay_capture(
      header, dictionary, runs, events, partial_windows, partial_membership, diagnostics, shadow);

  EXPECT_TRUE(replay.invalid_capture);
  EXPECT_EQ(replay.selected_event_count, 0u);
  EXPECT_EQ(shadow[0], 0x5678u);
  EXPECT_EQ(diagnostics[0].kind, 77u);
}

TEST(ConSanMoi, RecordReplayBarrierPrefersPersistentEpochOverPrivateRecords) {
  constexpr uint32_t kBarrierWait = 0xBF940000u;
  const std::array<uint32_t, 5> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32 v0, v0
      kBarrierWait,
      build_v_mov_b32_e32(/*vdst=*/62, vector_source_vgpr(62), ROCJITSU_CODE_ARCH_RDNA4),
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  constexpr uint32_t kWave64Vgpr64Granulated = 15;
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "record_replay_barrier_pressure", kWave64Vgpr64Granulated);

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_dynamic_access_records = true;
  options.moi_track_barriers = true;
  options.moi_init_owner_epoch = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(2, 0, 0, 0, 1);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.moi_persistent_vgprs_automatic);
  EXPECT_FALSE(result.moi_private_epoch_automatic);
  EXPECT_TRUE(result.resolved_moi_owner_vgpr);
  EXPECT_TRUE(result.resolved_moi_epoch_vgpr);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord,
                               &ConSanPatchInfo::kind),
            0);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiInlineEpochBarrier,
                               &ConSanPatchInfo::kind),
            1);
}

TEST(ConSanMoi, RecordReplayBarrierOnlyObjectSkipsAutomaticPersistentState) {
  constexpr uint32_t kBarrierWait = 0xBF940000u;
  const std::array<uint32_t, 2> text_words = {
      kBarrierWait,
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_track_barriers = true;
  options.moi_track_atomics = true;
  options.moi_init_owner_epoch = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 1, 1, 1, 1);

  const auto result = try_patch_consan(make_rdna4_lds_code_object(text_words), options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue,
                               &ConSanPatchInfo::kind),
            0);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord,
                               &ConSanPatchInfo::kind),
            1);
  EXPECT_TRUE(std::ranges::any_of(result.resource_plans, [](const auto &plan) {
    return plan.site_kind == ConSanResourceSiteKind::Barrier;
  }));
  EXPECT_NE(
      std::ranges::find(
          result.warnings,
          "ConSan MOI record/replay uses probe-local owner derivation without access records"),
      result.warnings.end());
}

TEST(ConSanMoi, RecordReplayAtomicOnlyObjectUsesProbeLocalOwner) {
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_track_barriers = false;
  options.moi_track_atomics = true;
  options.moi_init_owner_epoch = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 0, 1, 1);

  const ConSanResult result =
      try_patch_consan(make_rdna4_ordered_flat_atomic_code_object(), options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAtomicRecord,
                               &ConSanPatchInfo::kind),
            1);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue,
                               &ConSanPatchInfo::kind),
            0);
  EXPECT_FALSE(result.resolved_moi_owner_vgpr);
  EXPECT_FALSE(result.resolved_moi_epoch_vgpr);
  EXPECT_NE(
      std::ranges::find(
          result.warnings,
          "ConSan MOI record/replay uses probe-local owner derivation without access records"),
      result.warnings.end());
}

TEST(ConSanMoi, RecordReplayAtomicAcquireSuppressesSameEpochConflict) {
  std::array<uint64_t, 1> shadow{};
  std::array<ConSanMoiAtomicReleaseRecord, 1> releases{};
  std::array<ConSanMoiAcquiredEpochToken, 1> tokens{};

  const ConSanMoiRecordReplayAccess writer{
      /*generation=*/7,
      /*owner_id=*/0,
      /*epoch=*/0,
      ConSanMoiShadowAccessKind::Write,
      /*lds_byte_offset=*/0,
      /*lds_byte_count=*/4,
      /*start_cell=*/0,
      /*cell_count=*/1,
      /*instruction_offset=*/0x10,
      /*lane_mask=*/0x1,
  };
  const ConSanMoiRecordReplayAccess reader{
      /*generation=*/7,
      /*owner_id=*/1,
      /*epoch=*/0,
      ConSanMoiShadowAccessKind::Read,
      /*lds_byte_offset=*/0,
      /*lds_byte_count=*/4,
      /*start_cell=*/0,
      /*cell_count=*/1,
      /*instruction_offset=*/0x20,
      /*lane_mask=*/0x2,
  };

  EXPECT_FALSE(consan_moi_record_replay_access(shadow, writer).conflict);
  const ConSanMoiAtomicSyncResult release = consan_moi_record_replay_atomic_release(
      releases, /*generation=*/7, /*atomic_address=*/0x4000, /*producer_owner_id=*/0,
      /*producer_epoch=*/0, /*release_instruction_offset=*/0x100);
  EXPECT_FALSE(release.metadata_full);
  EXPECT_EQ(release.updated_record_count, 1u);

  const ConSanMoiAtomicSyncResult acquire = consan_moi_record_replay_atomic_acquire(
      releases, tokens, /*generation=*/7, /*atomic_address=*/0x4000, /*consumer_owner_id=*/1,
      /*acquire_instruction_offset=*/0x200);
  EXPECT_FALSE(acquire.metadata_full);
  EXPECT_EQ(acquire.updated_record_count, 1u);
  ASSERT_TRUE(tokens[0].valid);
  EXPECT_EQ(tokens[0].consumer_owner_id, 1u);
  EXPECT_EQ(tokens[0].producer_owner_id, 0u);
  EXPECT_EQ(tokens[0].producer_epoch_plus_one, 1u);

  const auto second = consan_moi_record_replay_access(shadow, reader, tokens);
  EXPECT_FALSE(second.conflict);
  const ConSanMoiExactShadowEntry updated = decode_consan_moi_exact_shadow_entry(shadow[0]);
  EXPECT_EQ(updated.kind, ConSanMoiShadowAccessKind::Read);
  EXPECT_EQ(updated.owner_id, 1u);
  EXPECT_EQ(updated.epoch, 0u);
}

TEST(ConSanMoi, RecordReplayAtomicOrderingRequiresMatchingAcquireAddress) {
  const auto make_writer = [] {
    return ConSanMoiRecordReplayAccess{
        /*generation=*/7,
        /*owner_id=*/0,
        /*epoch=*/0,
        ConSanMoiShadowAccessKind::Write,
        /*lds_byte_offset=*/0,
        /*lds_byte_count=*/4,
        /*start_cell=*/0,
        /*cell_count=*/1,
        /*instruction_offset=*/0x10,
        /*lane_mask=*/0x1,
    };
  };
  const auto make_reader = [] {
    return ConSanMoiRecordReplayAccess{
        /*generation=*/7,
        /*owner_id=*/1,
        /*epoch=*/0,
        ConSanMoiShadowAccessKind::Read,
        /*lds_byte_offset=*/0,
        /*lds_byte_count=*/4,
        /*start_cell=*/0,
        /*cell_count=*/1,
        /*instruction_offset=*/0x20,
        /*lane_mask=*/0x2,
    };
  };

  {
    SCOPED_TRACE("release without acquire behaves like a relaxed atomic for diagnostics");
    std::array<uint64_t, 1> shadow{};
    std::array<ConSanMoiAtomicReleaseRecord, 1> releases{};
    std::array<ConSanMoiAcquiredEpochToken, 1> tokens{};

    EXPECT_FALSE(consan_moi_record_replay_access(shadow, make_writer()).conflict);
    EXPECT_FALSE(consan_moi_record_replay_atomic_release(
                     releases, /*generation=*/7, /*atomic_address=*/0x4000,
                     /*producer_owner_id=*/0, /*producer_epoch=*/0,
                     /*release_instruction_offset=*/0x100)
                     .metadata_full);

    const auto second = consan_moi_record_replay_access(shadow, make_reader(), tokens);
    EXPECT_TRUE(second.conflict);
    EXPECT_FALSE(second.metadata_full);
  }

  {
    SCOPED_TRACE("acquire of a different atomic address does not import the producer");
    std::array<uint64_t, 1> shadow{};
    std::array<ConSanMoiAtomicReleaseRecord, 1> releases{};
    std::array<ConSanMoiAcquiredEpochToken, 1> tokens{};

    EXPECT_FALSE(consan_moi_record_replay_access(shadow, make_writer()).conflict);
    EXPECT_FALSE(consan_moi_record_replay_atomic_release(
                     releases, /*generation=*/7, /*atomic_address=*/0x4000,
                     /*producer_owner_id=*/0, /*producer_epoch=*/0,
                     /*release_instruction_offset=*/0x100)
                     .metadata_full);
    EXPECT_EQ(consan_moi_record_replay_atomic_acquire(
                  releases, tokens, /*generation=*/7, /*atomic_address=*/0x5000,
                  /*consumer_owner_id=*/1, /*acquire_instruction_offset=*/0x200)
                  .updated_record_count,
              0u);

    const auto second = consan_moi_record_replay_access(shadow, make_reader(), tokens);
    EXPECT_TRUE(second.conflict);
    EXPECT_FALSE(second.metadata_full);
  }
}

TEST(ConSanMoi, RecordReplayAtomicReleaseKeepsMaxEpochPerProducerAddress) {
  std::array<ConSanMoiAtomicReleaseRecord, 1> releases{};

  EXPECT_EQ(consan_moi_record_replay_atomic_release(releases, /*generation=*/7,
                                                    /*atomic_address=*/0x4000,
                                                    /*producer_owner_id=*/0, /*producer_epoch=*/2,
                                                    /*release_instruction_offset=*/0x100)
                .updated_record_count,
            1u);
  ASSERT_TRUE(releases[0].valid);
  EXPECT_EQ(releases[0].producer_epoch, 2u);
  EXPECT_EQ(releases[0].release_instruction_offset, 0x100u);

  EXPECT_EQ(consan_moi_record_replay_atomic_release(releases, /*generation=*/7,
                                                    /*atomic_address=*/0x4000,
                                                    /*producer_owner_id=*/0, /*producer_epoch=*/1,
                                                    /*release_instruction_offset=*/0x110)
                .updated_record_count,
            0u);
  EXPECT_EQ(releases[0].producer_epoch, 2u);
  EXPECT_EQ(releases[0].release_instruction_offset, 0x100u);

  EXPECT_EQ(consan_moi_record_replay_atomic_release(releases, /*generation=*/7,
                                                    /*atomic_address=*/0x4000,
                                                    /*producer_owner_id=*/0, /*producer_epoch=*/5,
                                                    /*release_instruction_offset=*/0x120)
                .updated_record_count,
            1u);
  EXPECT_EQ(releases[0].producer_epoch, 5u);
  EXPECT_EQ(releases[0].release_instruction_offset, 0x120u);

  const ConSanMoiAtomicSyncResult full = consan_moi_record_replay_atomic_release(
      releases, /*generation=*/7, /*atomic_address=*/0x4000, /*producer_owner_id=*/1,
      /*producer_epoch=*/0, /*release_instruction_offset=*/0x130);
  EXPECT_TRUE(full.metadata_full);
  EXPECT_EQ(full.updated_record_count, 0u);
}

TEST(ConSanMoi, RecordReplayAtomicEventsSuppressOrderedConflict) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/2,
      /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/1,
      /*sampled_watchpoint_capacity=*/0);
  header.access_record_count = 2;

  std::array<ConSanMoiAccessRecord, 2> records{};
  records[0].wave_id = 0;
  records[0].event_index = 0;
  records[0].instruction_offset = 0x10;
  records[0].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write);
  records[0].lds_byte_count = 4;
  records[0].cell_count = 1;

  records[1].wave_id = 1;
  records[1].event_index = 3;
  records[1].instruction_offset = 0x20;
  records[1].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Read);
  records[1].lds_byte_count = 4;
  records[1].cell_count = 1;

  std::array<ConSanMoiRecordReplayAtomicEvent, 2> atomics{};
  atomics[0].owner_id = 0;
  atomics[0].atomic_address = 0x4000;
  atomics[0].instruction_offset = 0x100;
  atomics[0].event_index = 1;
  atomics[0].kind = ConSanMoiAtomicEventKind::Release;

  atomics[1].owner_id = 1;
  atomics[1].atomic_address = 0x4000;
  atomics[1].instruction_offset = 0x200;
  atomics[1].event_index = 2;
  atomics[1].kind = ConSanMoiAtomicEventKind::Acquire;

  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
  std::array<uint64_t, 1> shadow{};

  const ConSanMoiRecordReplayResult replay = consan_moi_record_replay_access_records(
      header, records, std::span<const ConSanMoiBarrierRecord>{}, atomics, diagnostics, shadow);

  EXPECT_EQ(replay.processed_access_count, 2u);
  EXPECT_EQ(replay.processed_atomic_count, 2u);
  EXPECT_EQ(replay.unsupported_atomic_count, 0u);
  EXPECT_FALSE(replay.metadata_full);
  EXPECT_FALSE(replay.conflict);
  EXPECT_EQ(header.diagnostic_count, 0u);

  const ConSanMoiExactShadowEntry final = decode_consan_moi_exact_shadow_entry(shadow[0]);
  EXPECT_EQ(final.kind, ConSanMoiShadowAccessKind::Read);
  EXPECT_EQ(final.owner_id, 1u);
  EXPECT_EQ(final.epoch, 0u);
  EXPECT_EQ(final.instruction_offset, 0x20u);
}

TEST(ConSanMoi, RecordReplayAtomicEventsRequireMatchingAddress) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/2,
      /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/1,
      /*sampled_watchpoint_capacity=*/0);
  header.access_record_count = 2;

  std::array<ConSanMoiAccessRecord, 2> records{};
  records[0].wave_id = 0;
  records[0].event_index = 0;
  records[0].instruction_offset = 0x10;
  records[0].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write);
  records[0].lds_byte_count = 4;
  records[0].cell_count = 1;

  records[1].wave_id = 1;
  records[1].event_index = 3;
  records[1].instruction_offset = 0x20;
  records[1].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Read);
  records[1].lds_byte_count = 4;
  records[1].cell_count = 1;

  std::array<ConSanMoiRecordReplayAtomicEvent, 2> atomics{};
  atomics[0].owner_id = 0;
  atomics[0].atomic_address = 0x4000;
  atomics[0].instruction_offset = 0x100;
  atomics[0].event_index = 1;
  atomics[0].kind = ConSanMoiAtomicEventKind::Release;

  atomics[1].owner_id = 1;
  atomics[1].atomic_address = 0x5000;
  atomics[1].instruction_offset = 0x200;
  atomics[1].event_index = 2;
  atomics[1].kind = ConSanMoiAtomicEventKind::Acquire;

  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
  std::array<uint64_t, 1> shadow{};

  const ConSanMoiRecordReplayResult replay = consan_moi_record_replay_access_records(
      header, records, std::span<const ConSanMoiBarrierRecord>{}, atomics, diagnostics, shadow);

  EXPECT_EQ(replay.processed_access_count, 2u);
  EXPECT_EQ(replay.processed_atomic_count, 2u);
  EXPECT_TRUE(replay.conflict);
  ASSERT_EQ(header.diagnostic_count, 1u);
  EXPECT_EQ(diagnostics[0].first_instruction_offset, 0x10u);
  EXPECT_EQ(diagnostics[0].second_instruction_offset, 0x20u);
}

TEST(ConSanMoi, RecordReplayAtomicEventsAreWorkgroupLocalForLdsOrdering) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/2,
      /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/1,
      /*sampled_watchpoint_capacity=*/0);
  header.access_record_count = 2;

  std::array<ConSanMoiAccessRecord, 2> records{};
  records[0].workgroup_x = 1;
  records[0].wave_id = 0;
  records[0].event_index = 0;
  records[0].instruction_offset = 0x10;
  records[0].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write);
  records[0].lds_byte_count = 4;
  records[0].cell_count = 1;

  records[1].workgroup_x = 1;
  records[1].wave_id = 1;
  records[1].event_index = 3;
  records[1].instruction_offset = 0x20;
  records[1].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Read);
  records[1].lds_byte_count = 4;
  records[1].cell_count = 1;

  std::array<ConSanMoiRecordReplayAtomicEvent, 2> atomics{};
  atomics[0].workgroup_x = 0;
  atomics[0].owner_id = 0;
  atomics[0].atomic_address = 0x4000;
  atomics[0].instruction_offset = 0x100;
  atomics[0].event_index = 1;
  atomics[0].kind = ConSanMoiAtomicEventKind::Release;

  atomics[1].workgroup_x = 1;
  atomics[1].owner_id = 1;
  atomics[1].atomic_address = 0x4000;
  atomics[1].instruction_offset = 0x200;
  atomics[1].event_index = 2;
  atomics[1].kind = ConSanMoiAtomicEventKind::Acquire;

  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
  std::array<uint64_t, 1> shadow{};

  const ConSanMoiRecordReplayResult replay = consan_moi_record_replay_access_records(
      header, records, std::span<const ConSanMoiBarrierRecord>{}, atomics, diagnostics, shadow);

  EXPECT_EQ(replay.processed_atomic_count, 2u);
  EXPECT_TRUE(replay.conflict);
  ASSERT_EQ(header.diagnostic_count, 1u);
  EXPECT_EQ(diagnostics[0].first_owner_id, 0u);
  EXPECT_EQ(diagnostics[0].second_owner_id, 1u);
}

TEST(ConSanMoi, RecordReplayAccessRecordsEmitsConflictDiagnostic) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/2,
      /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/3,
      /*sampled_watchpoint_capacity=*/0);
  header.access_record_count = 2;

  std::array<ConSanMoiAccessRecord, 2> records{};
  records[0].wave_id = 1;
  records[0].lane_mask = 0x1;
  records[0].instruction_offset = 0x10;
  records[0].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write);
  records[0].lds_byte_offset = 8;
  records[0].lds_byte_count = 4;
  records[0].epoch = 3;

  records[1].wave_id = 2;
  records[1].lane_mask = 0x2;
  records[1].instruction_offset = 0x20;
  records[1].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Read);
  records[1].lds_byte_offset = 8;
  records[1].lds_byte_count = 4;
  records[1].epoch = 3;

  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
  std::array<uint64_t, 3> shadow{};

  const ConSanMoiRecordReplayResult replay =
      consan_moi_record_replay_access_records(header, records, diagnostics, shadow);

  EXPECT_EQ(replay.processed_access_count, 2u);
  EXPECT_EQ(replay.dropped_access_count, 0u);
  EXPECT_EQ(replay.unsupported_access_count, 0u);
  EXPECT_EQ(replay.emitted_diagnostic_count, 1u);
  EXPECT_FALSE(replay.diagnostic_capacity_exhausted);
  EXPECT_FALSE(replay.metadata_full);
  EXPECT_TRUE(replay.conflict);
  EXPECT_EQ(header.diagnostic_count, 1u);
  EXPECT_EQ(shadow[0], 0u);
  EXPECT_NE(shadow[2], 0u);

  const ConSanMoiDiagnosticRecord &diagnostic = diagnostics[0];
  EXPECT_EQ(diagnostic.kind, static_cast<uint32_t>(ConSanMoiDiagnosticKind::AccessConflict));
  EXPECT_EQ(diagnostic.backend, static_cast<uint32_t>(ConSanMoiEngine::RecordReplay));
  EXPECT_EQ(diagnostic.generation, 7u);
  EXPECT_EQ(diagnostic.epoch, 3u);
  EXPECT_EQ(diagnostic.first_owner_id, 1u);
  EXPECT_EQ(diagnostic.second_owner_id, 2u);
  EXPECT_EQ(diagnostic.second_lane_mask, 0x2u);
  EXPECT_EQ(diagnostic.first_instruction_offset, 0x10u);
  EXPECT_EQ(diagnostic.second_instruction_offset, 0x20u);
  EXPECT_EQ(diagnostic.second_lds_byte_offset, 8u);
  EXPECT_EQ(diagnostic.second_lds_byte_count, 4u);
  EXPECT_EQ(diagnostic.first_access_kind, static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write));
  EXPECT_EQ(diagnostic.second_access_kind, static_cast<uint32_t>(ConSanMoiShadowAccessKind::Read));
}

TEST(ConSanMoi, RecordReplayAccessRecordsRetainsEveryBoundedDiagnostic) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/4,
      /*diagnostic_capacity=*/2, /*exact_shadow_entry_capacity=*/2,
      /*sampled_watchpoint_capacity=*/0);
  header.access_record_count = 4;

  const auto make_record = [](uint32_t event_index, uint32_t wave_id,
                              ConSanMoiShadowAccessKind kind, uint32_t cell,
                              uint32_t instruction_offset) {
    ConSanMoiAccessRecord record{};
    record.event_index = event_index;
    record.wave_id = wave_id;
    record.lane_mask = uint64_t{1} << wave_id;
    record.instruction_offset = instruction_offset;
    record.access_kind = static_cast<uint32_t>(kind);
    record.lds_byte_offset = cell * sizeof(uint32_t);
    record.lds_byte_count = sizeof(uint32_t);
    record.start_cell = cell;
    record.cell_count = 1;
    record.epoch = 3;
    return record;
  };
  const std::array records = {
      make_record(0, 1, ConSanMoiShadowAccessKind::Write, 0, 0x10),
      make_record(1, 1, ConSanMoiShadowAccessKind::Write, 1, 0x20),
      make_record(2, 2, ConSanMoiShadowAccessKind::Read, 0, 0x30),
      make_record(3, 2, ConSanMoiShadowAccessKind::Read, 1, 0x40),
  };
  std::array<ConSanMoiDiagnosticRecord, 2> diagnostics{};
  std::array<uint64_t, 2> shadow{};

  const ConSanMoiRecordReplayResult replay =
      consan_moi_record_replay_access_records(header, records, diagnostics, shadow);

  ASSERT_EQ(replay.emitted_diagnostic_count, 2u);
  ASSERT_EQ(header.diagnostic_count, 2u);
  EXPECT_FALSE(replay.diagnostic_capacity_exhausted);
  EXPECT_EQ(diagnostics[0].first_instruction_offset, 0x10u);
  EXPECT_EQ(diagnostics[0].second_instruction_offset, 0x30u);
  EXPECT_EQ(diagnostics[1].first_instruction_offset, 0x20u);
  EXPECT_EQ(diagnostics[1].second_instruction_offset, 0x40u);
}

TEST(ConSanMoi, RecordReplayBarrierEventsAdvanceEpochsInEventOrder) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/2,
      /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/1,
      /*sampled_watchpoint_capacity=*/0, /*barrier_record_capacity=*/1);
  header.access_record_count = 2;
  header.barrier_record_count = 1;

  std::array<ConSanMoiAccessRecord, 2> records{};
  records[0].wave_id = 0;
  records[0].event_index = 0;
  records[0].instruction_offset = 0x10;
  records[0].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write);
  records[0].lds_byte_count = 4;
  records[0].cell_count = 1;

  records[1].wave_id = 1;
  records[1].event_index = 3;
  records[1].instruction_offset = 0x20;
  records[1].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Read);
  records[1].lds_byte_count = 4;
  records[1].cell_count = 1;

  std::array<ConSanMoiBarrierRecord, 1> barriers{};
  barriers[0].wave_id = 0;
  barriers[0].event_index = 1;
  barriers[0].instruction_offset = 0x18;

  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
  std::array<uint64_t, 1> shadow{};

  ConSanMoiReportHeader no_barrier_header = header;
  no_barrier_header.barrier_record_count = 0;
  const ConSanMoiRecordReplayResult no_barrier =
      consan_moi_record_replay_access_records(no_barrier_header, records, diagnostics, shadow);
  EXPECT_TRUE(no_barrier.conflict);
  EXPECT_EQ(no_barrier.processed_barrier_count, 0u);
  EXPECT_EQ(no_barrier_header.diagnostic_count, 1u);

  diagnostics = {};
  shadow = {};
  const ConSanMoiRecordReplayResult with_barrier =
      consan_moi_record_replay_access_records(header, records, barriers, diagnostics, shadow);
  EXPECT_FALSE(with_barrier.conflict);
  EXPECT_EQ(with_barrier.processed_access_count, 2u);
  EXPECT_EQ(with_barrier.processed_barrier_count, 1u);
  EXPECT_EQ(with_barrier.dropped_barrier_count, 0u);
  EXPECT_EQ(header.diagnostic_count, 0u);

  const ConSanMoiExactShadowEntry final = decode_consan_moi_exact_shadow_entry(shadow[0]);
  EXPECT_EQ(final.kind, ConSanMoiShadowAccessKind::Read);
  EXPECT_EQ(final.owner_id, 1u);
  EXPECT_EQ(final.epoch, 1u);
  EXPECT_EQ(final.instruction_offset, 0x20u);
}

TEST(ConSanMoi, RecordReplayBarrierDoesNotOrderAnotherDispatch) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/2,
      /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/1,
      /*sampled_watchpoint_capacity=*/0, /*barrier_record_capacity=*/1);
  header.access_record_count = 2;
  header.barrier_record_count = 1;

  std::array<ConSanMoiAccessRecord, 2> records{};
  records[0].generation = 101;
  records[0].wave_id = 0;
  records[0].event_index = 0;
  records[0].instruction_offset = 0x10;
  records[0].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write);
  records[0].lds_byte_count = 4;
  records[0].cell_count = 1;

  records[1].generation = 101;
  records[1].wave_id = 1;
  records[1].event_index = 2;
  records[1].instruction_offset = 0x20;
  records[1].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Read);
  records[1].lds_byte_count = 4;
  records[1].cell_count = 1;

  std::array<ConSanMoiBarrierRecord, 1> barriers{};
  barriers[0].generation = 202;
  barriers[0].wave_id = 0;
  barriers[0].event_index = 1;

  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
  std::array<uint64_t, 1> shadow{};
  const ConSanMoiRecordReplayResult replay = consan_moi_record_replay_access_records(
      header, records, barriers, std::span<const ConSanMoiRecordReplayAtomicEvent>{}, diagnostics,
      shadow);

  EXPECT_TRUE(replay.conflict);
  ASSERT_EQ(header.diagnostic_count, 1u);
  EXPECT_EQ(diagnostics[0].generation, 101u);
  EXPECT_EQ(diagnostics[0].first_instruction_offset, 0x10u);
  EXPECT_EQ(diagnostics[0].second_instruction_offset, 0x20u);
}

TEST(ConSanMoi, RecordReplayDoesNotCompareAccessesFromDifferentDispatches) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/2,
      /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/1,
      /*sampled_watchpoint_capacity=*/0);
  header.access_record_count = 2;

  std::array<ConSanMoiAccessRecord, 2> records{};
  records[0].generation = 101;
  records[0].wave_id = 0;
  records[0].event_index = 0;
  records[0].instruction_offset = 0x10;
  records[0].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write);
  records[0].lds_byte_count = 4;
  records[0].cell_count = 1;

  records[1] = records[0];
  records[1].generation = 202;
  records[1].wave_id = 1;
  records[1].event_index = 1;
  records[1].instruction_offset = 0x20;
  records[1].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Read);

  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
  std::array<uint64_t, 1> shadow{};
  const ConSanMoiRecordReplayResult replay =
      consan_moi_record_replay_access_records(header, records, diagnostics, shadow);

  EXPECT_FALSE(replay.conflict);
  EXPECT_EQ(replay.processed_access_count, 2u);
  EXPECT_EQ(header.diagnostic_count, 0u);
}

TEST(ConSanMoi, RecordReplayBarrierEventsAdvanceOnlyTheirWorkgroup) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/4,
      /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/1,
      /*sampled_watchpoint_capacity=*/0, /*barrier_record_capacity=*/1);
  header.access_record_count = 4;
  header.barrier_record_count = 1;

  std::array<ConSanMoiAccessRecord, 4> records{};
  records[0].workgroup_x = 0;
  records[0].wave_id = 0;
  records[0].event_index = 0;
  records[0].instruction_offset = 0x10;
  records[0].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write);
  records[0].lds_byte_count = 4;
  records[0].cell_count = 1;

  records[1].workgroup_x = 1;
  records[1].wave_id = 0;
  records[1].event_index = 0;
  records[1].instruction_offset = 0x30;
  records[1].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write);
  records[1].lds_byte_count = 4;
  records[1].cell_count = 1;

  records[2].workgroup_x = 0;
  records[2].wave_id = 1;
  records[2].event_index = 2;
  records[2].instruction_offset = 0x20;
  records[2].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Read);
  records[2].lds_byte_count = 4;
  records[2].cell_count = 1;

  records[3].workgroup_x = 1;
  records[3].wave_id = 1;
  records[3].event_index = 3;
  records[3].instruction_offset = 0x40;
  records[3].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Read);
  records[3].lds_byte_count = 4;
  records[3].cell_count = 1;

  std::array<ConSanMoiBarrierRecord, 1> barriers{};
  barriers[0].workgroup_x = 0;
  barriers[0].wave_id = 0;
  barriers[0].event_index = 1;
  barriers[0].instruction_offset = 0x18;

  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
  std::array<uint64_t, 1> shadow{};

  const ConSanMoiRecordReplayResult replay =
      consan_moi_record_replay_access_records(header, records, barriers, diagnostics, shadow);

  EXPECT_TRUE(replay.conflict);
  ASSERT_EQ(header.diagnostic_count, 1u);
  EXPECT_EQ(diagnostics[0].first_instruction_offset, 0x30u);
  EXPECT_EQ(diagnostics[0].second_instruction_offset, 0x40u);
}

TEST(ConSanMoi, RecordReplayCoalescesBarrierRuns) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/2,
      /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/1,
      /*sampled_watchpoint_capacity=*/0, /*barrier_record_capacity=*/2);
  header.access_record_count = 2;
  header.barrier_record_count = 2;

  std::array<ConSanMoiAccessRecord, 2> records{};
  records[0].wave_id = 0;
  records[0].event_index = 0;
  records[0].instruction_offset = 0x10;
  records[0].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write);
  records[0].lds_byte_count = 4;
  records[0].cell_count = 1;

  records[1].wave_id = 1;
  records[1].event_index = 3;
  records[1].instruction_offset = 0x20;
  records[1].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Read);
  records[1].lds_byte_count = 4;
  records[1].cell_count = 1;

  std::array<ConSanMoiBarrierRecord, 2> barriers{};
  barriers[0].wave_id = 0;
  barriers[0].event_index = 1;
  barriers[0].instruction_offset = 0x18;

  barriers[1].wave_id = 0;
  barriers[1].event_index = 2;
  barriers[1].instruction_offset = 0x18;

  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
  std::array<uint64_t, 1> shadow{};

  const ConSanMoiRecordReplayResult replay =
      consan_moi_record_replay_access_records(header, records, barriers, diagnostics, shadow);

  EXPECT_FALSE(replay.conflict);
  EXPECT_EQ(replay.processed_access_count, 2u);
  EXPECT_EQ(replay.processed_barrier_count, 2u);
  EXPECT_EQ(header.diagnostic_count, 0u);

  const ConSanMoiExactShadowEntry final = decode_consan_moi_exact_shadow_entry(shadow[0]);
  EXPECT_EQ(final.kind, ConSanMoiShadowAccessKind::Read);
  EXPECT_EQ(final.owner_id, 1u);
  EXPECT_EQ(final.epoch, 1u);
  EXPECT_EQ(final.instruction_offset, 0x20u);
}

TEST(ConSanMoi, RecordReplayReportsDiagnosticCapacityExhaustion) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/2,
      /*diagnostic_capacity=*/0, /*exact_shadow_entry_capacity=*/1,
      /*sampled_watchpoint_capacity=*/0);
  header.access_record_count = 2;

  std::array<ConSanMoiAccessRecord, 2> records{};
  records[0].wave_id = 1;
  records[0].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write);
  records[0].lds_byte_count = 4;
  records[0].cell_count = 1;

  records[1].wave_id = 2;
  records[1].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Read);
  records[1].lds_byte_count = 4;
  records[1].cell_count = 1;

  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
  std::array<uint64_t, 1> shadow{};

  const ConSanMoiRecordReplayResult replay =
      consan_moi_record_replay_access_records(header, records, diagnostics, shadow);

  EXPECT_EQ(replay.processed_access_count, 2u);
  EXPECT_EQ(replay.dropped_access_count, 0u);
  EXPECT_EQ(replay.unsupported_access_count, 0u);
  EXPECT_EQ(replay.emitted_diagnostic_count, 0u);
  EXPECT_TRUE(replay.diagnostic_capacity_exhausted);
  EXPECT_TRUE(replay.conflict);
  EXPECT_EQ(header.diagnostic_count, 0u);
}

TEST(ConSanMoi, RecordReplayReportsDroppedAndUnsupportedAccessRecords) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/3,
      /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/1,
      /*sampled_watchpoint_capacity=*/0);
  header.access_record_count = 3;

  std::array<ConSanMoiAccessRecord, 2> records{};
  records[0].wave_id = 1;
  records[0].access_kind = 0xffffffffu;
  records[0].lds_byte_count = 4;
  records[0].cell_count = 1;

  records[1].wave_id = 2;
  records[1].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write);
  records[1].lds_byte_count = 4;
  records[1].cell_count = 1;

  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
  std::array<uint64_t, 1> shadow{};

  const ConSanMoiRecordReplayResult replay =
      consan_moi_record_replay_access_records(header, records, diagnostics, shadow);

  EXPECT_EQ(replay.processed_access_count, 2u);
  EXPECT_EQ(replay.dropped_access_count, 1u);
  EXPECT_EQ(replay.unsupported_access_count, 1u);
  EXPECT_EQ(replay.emitted_diagnostic_count, 0u);
  EXPECT_FALSE(replay.conflict);
  EXPECT_EQ(header.diagnostic_count, 0u);
  EXPECT_NE(shadow[0], 0u);
}

TEST(ConSanMoi, RecordReplaySkipsOnlyCompletelyUnpublishedStaticSlots) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/3,
      /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/1,
      /*sampled_watchpoint_capacity=*/0);
  header.access_record_count = 3;

  std::array<ConSanMoiAccessRecord, 3> records{};
  records[0].wave_id = 1;
  records[0].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write);
  records[0].lds_byte_count = 4;
  records[0].cell_count = 1;

  records[1].wave_id = 2;
  records[1].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Read);
  records[1].lds_byte_count = 4;
  records[1].cell_count = 1;

  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
  std::array<uint64_t, 1> shadow{};
  const ConSanMoiRecordReplayResult replay =
      consan_moi_record_replay_access_records(header, records, diagnostics, shadow);

  EXPECT_EQ(replay.processed_access_count, 2u);
  EXPECT_EQ(replay.unsupported_access_count, 0u);
  EXPECT_TRUE(replay.conflict);

  // A claimed-but-not-committed bank is not an unpublished zero slot. Treat
  // it as malformed/incomplete evidence instead of silently dropping it.
  records[2].claim_token = 1;
  const ConSanMoiRecordReplayResult malformed =
      consan_moi_record_replay_access_records(header, records, diagnostics, shadow);
  EXPECT_EQ(malformed.processed_access_count, 3u);
  EXPECT_EQ(malformed.unsupported_access_count, 1u);
}

TEST(ConSanMoi, RecordReplayReportsDroppedBarrierRecords) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/2,
      /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/1,
      /*sampled_watchpoint_capacity=*/0, /*barrier_record_capacity=*/2);
  header.access_record_count = 2;
  header.barrier_record_count = 2;

  std::array<ConSanMoiAccessRecord, 2> records{};
  records[0].wave_id = 1;
  records[0].event_index = 0;
  records[0].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write);
  records[0].lds_byte_count = 4;
  records[0].cell_count = 1;

  records[1].wave_id = 2;
  records[1].event_index = 3;
  records[1].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Read);
  records[1].lds_byte_count = 4;
  records[1].cell_count = 1;

  std::array<ConSanMoiBarrierRecord, 1> visible_barriers{};
  visible_barriers[0].event_index = 1;

  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
  std::array<uint64_t, 1> shadow{};

  const ConSanMoiRecordReplayResult replay = consan_moi_record_replay_access_records(
      header, records, visible_barriers, diagnostics, shadow);

  EXPECT_EQ(replay.processed_access_count, 2u);
  EXPECT_EQ(replay.processed_barrier_count, 1u);
  EXPECT_EQ(replay.dropped_access_count, 0u);
  EXPECT_EQ(replay.dropped_barrier_count, 1u);
  EXPECT_FALSE(replay.conflict);
  EXPECT_EQ(header.diagnostic_count, 0u);
}

} // namespace
} // namespace rocjitsu
