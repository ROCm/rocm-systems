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

TEST(ConSanMoi, RecordReplayBarrierRecordsUsePersistentEpochState) {
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
            1);
}

TEST(ConSanMoi, RecordReplayBarrierOnlyObjectCapturesPersistentEntryState) {
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
            1);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord,
                               &ConSanPatchInfo::kind),
            1);
  EXPECT_TRUE(result.resolved_moi_owner_vgpr);
  EXPECT_TRUE(result.resolved_moi_epoch_vgpr);
  EXPECT_TRUE(result.resolved_moi_record_replay_workgroup_vgprs.complete());
  EXPECT_TRUE(std::ranges::any_of(result.resource_plans, [](const auto &plan) {
    return plan.site_kind == ConSanResourceSiteKind::Barrier;
  }));
}

TEST(ConSanMoi, RecordReplayAtomicOnlyObjectCapturesPersistentEntryState) {
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
            1);
  EXPECT_TRUE(result.resolved_moi_owner_vgpr);
  EXPECT_TRUE(result.resolved_moi_epoch_vgpr);
  EXPECT_TRUE(result.resolved_moi_record_replay_workgroup_vgprs.complete());
}

TEST(ConSanMoi, RecordReplayAtomicAcquireSuppressesSameEpochConflict) {
  std::array<ConSanMoiAtomicReleaseRecord, 1> releases{};
  std::array<ConSanMoiAcquiredEpochToken, 1> tokens{};
  ConSanMoiSparseExactByteShadow model(/*byte_capacity=*/4, /*maximum_access_count=*/2);

  const ConSanMoiExactByteAccess writer{
      /*generation=*/7,
      /*owner_id=*/0,
      /*epoch=*/0,
      ConSanMoiShadowAccessKind::Write,
      /*lds_byte_offset=*/0,
      /*lds_byte_count=*/4,
      /*instruction_offset=*/0x10,
      /*lane_mask=*/0x1,
  };
  const ConSanMoiExactByteAccess reader{
      /*generation=*/7,
      /*owner_id=*/1,
      /*epoch=*/0,
      ConSanMoiShadowAccessKind::Read,
      /*lds_byte_offset=*/0,
      /*lds_byte_count=*/4,
      /*instruction_offset=*/0x20,
      /*lane_mask=*/0x2,
  };

  EXPECT_FALSE(model.access(writer).conflict);
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

  const auto second = model.access(reader, tokens);
  EXPECT_FALSE(second.conflict);
  EXPECT_FALSE(second.capacity_exhausted);
}

TEST(ConSanMoi, RecordReplayAtomicOrderingRequiresMatchingAcquireAddress) {
  const auto make_writer = [] {
    return ConSanMoiExactByteAccess{
        /*generation=*/7,
        /*owner_id=*/0,
        /*epoch=*/0,
        ConSanMoiShadowAccessKind::Write,
        /*lds_byte_offset=*/0,
        /*lds_byte_count=*/4,
        /*instruction_offset=*/0x10,
        /*lane_mask=*/0x1,
    };
  };
  const auto make_reader = [] {
    return ConSanMoiExactByteAccess{
        /*generation=*/7,
        /*owner_id=*/1,
        /*epoch=*/0,
        ConSanMoiShadowAccessKind::Read,
        /*lds_byte_offset=*/0,
        /*lds_byte_count=*/4,
        /*instruction_offset=*/0x20,
        /*lane_mask=*/0x2,
    };
  };

  {
    SCOPED_TRACE("release without acquire behaves like a relaxed atomic for diagnostics");
    std::array<ConSanMoiAtomicReleaseRecord, 1> releases{};
    std::array<ConSanMoiAcquiredEpochToken, 1> tokens{};
    ConSanMoiSparseExactByteShadow model(/*byte_capacity=*/4, /*maximum_access_count=*/2);

    EXPECT_FALSE(model.access(make_writer()).conflict);
    EXPECT_FALSE(consan_moi_record_replay_atomic_release(
                     releases, /*generation=*/7, /*atomic_address=*/0x4000,
                     /*producer_owner_id=*/0, /*producer_epoch=*/0,
                     /*release_instruction_offset=*/0x100)
                     .metadata_full);

    const auto second = model.access(make_reader(), tokens);
    EXPECT_TRUE(second.conflict);
    EXPECT_FALSE(second.capacity_exhausted);
  }

  {
    SCOPED_TRACE("acquire of a different atomic address does not import the producer");
    std::array<ConSanMoiAtomicReleaseRecord, 1> releases{};
    std::array<ConSanMoiAcquiredEpochToken, 1> tokens{};
    ConSanMoiSparseExactByteShadow model(/*byte_capacity=*/4, /*maximum_access_count=*/2);

    EXPECT_FALSE(model.access(make_writer()).conflict);
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

    const auto second = model.access(make_reader(), tokens);
    EXPECT_TRUE(second.conflict);
    EXPECT_FALSE(second.capacity_exhausted);
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

TEST(ConSanMoi, RecordReplayAtomicReleasePublishesCausalClockTransactionally) {
  const std::array<ConSanMoiAcquiredEpochToken, 1> acquired{{
      ConSanMoiAcquiredEpochToken{
          /*valid=*/true,
          /*generation=*/7,
          /*consumer_owner_id=*/1,
          /*producer_owner_id=*/0,
          /*producer_epoch_plus_one=*/3,
          /*acquire_instruction_offset=*/0x200,
      },
  }};

  std::array<ConSanMoiAtomicReleaseRecord, 2> releases{};
  const ConSanMoiAtomicSyncResult published = consan_moi_record_replay_atomic_release(
      releases, acquired, /*generation=*/7, /*atomic_address=*/0x5000,
      /*producer_owner_id=*/1, /*producer_epoch=*/4, /*release_instruction_offset=*/0x300);
  EXPECT_FALSE(published.metadata_full);
  EXPECT_EQ(published.updated_record_count, 2u);
  const auto owner_zero =
      std::ranges::find(releases, 0u, &ConSanMoiAtomicReleaseRecord::producer_owner_id);
  const auto owner_one =
      std::ranges::find(releases, 1u, &ConSanMoiAtomicReleaseRecord::producer_owner_id);
  ASSERT_NE(owner_zero, releases.end());
  ASSERT_NE(owner_one, releases.end());
  EXPECT_EQ(owner_zero->producer_epoch, 2u);
  EXPECT_EQ(owner_one->producer_epoch, 4u);

  std::array<ConSanMoiAcquiredEpochToken, 2> imported{};
  const ConSanMoiAtomicSyncResult acquired_by_two = consan_moi_record_replay_atomic_acquire(
      releases, imported, /*generation=*/7, /*atomic_address=*/0x5000,
      /*consumer_owner_id=*/2, /*acquire_instruction_offset=*/0x400);
  EXPECT_FALSE(acquired_by_two.metadata_full);
  EXPECT_EQ(acquired_by_two.updated_record_count, 2u);
  EXPECT_NE(std::ranges::find(imported, 0u, &ConSanMoiAcquiredEpochToken::producer_owner_id),
            imported.end());
  EXPECT_NE(std::ranges::find(imported, 1u, &ConSanMoiAcquiredEpochToken::producer_owner_id),
            imported.end());

  std::array<ConSanMoiAcquiredEpochToken, 1> insufficient_import{};
  const ConSanMoiAtomicSyncResult rejected_import = consan_moi_record_replay_atomic_acquire(
      releases, insufficient_import, /*generation=*/7, /*atomic_address=*/0x5000,
      /*consumer_owner_id=*/2, /*acquire_instruction_offset=*/0x400);
  EXPECT_TRUE(rejected_import.metadata_full);
  EXPECT_EQ(rejected_import.updated_record_count, 0u);
  EXPECT_FALSE(insufficient_import[0].valid);

  std::array<ConSanMoiAtomicReleaseRecord, 1> insufficient{};
  const ConSanMoiAtomicSyncResult rejected = consan_moi_record_replay_atomic_release(
      insufficient, acquired, /*generation=*/7, /*atomic_address=*/0x5000,
      /*producer_owner_id=*/1, /*producer_epoch=*/4, /*release_instruction_offset=*/0x300);
  EXPECT_TRUE(rejected.metadata_full);
  EXPECT_EQ(rejected.updated_record_count, 0u);
  EXPECT_FALSE(insufficient[0].valid);
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

TEST(ConSanMoi, RecordReplayAtomicEventsPropagateOrderingAcrossAddresses) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/3,
      /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/1,
      /*sampled_watchpoint_capacity=*/0);
  header.access_record_count = 3;

  std::array<ConSanMoiAccessRecord, 3> records{};
  for (uint32_t owner = 0; owner < records.size(); ++owner) {
    records[owner].wave_id = owner;
    records[owner].event_index = owner * 3;
    records[owner].instruction_offset = 0x10 + owner * 0x10;
    records[owner].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write);
    records[owner].lds_byte_count = 4;
    records[owner].cell_count = 1;
  }

  std::array<ConSanMoiRecordReplayAtomicEvent, 4> atomics{};
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

  atomics[2].owner_id = 1;
  atomics[2].atomic_address = 0x5000;
  atomics[2].instruction_offset = 0x300;
  atomics[2].event_index = 4;
  atomics[2].kind = ConSanMoiAtomicEventKind::Release;

  atomics[3].owner_id = 2;
  atomics[3].atomic_address = 0x5000;
  atomics[3].instruction_offset = 0x400;
  atomics[3].event_index = 5;
  atomics[3].kind = ConSanMoiAtomicEventKind::Acquire;

  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
  std::array<uint64_t, 1> shadow{};
  const ConSanMoiRecordReplayResult replay = consan_moi_record_replay_access_records(
      header, records, std::span<const ConSanMoiBarrierRecord>{}, atomics, diagnostics, shadow);

  EXPECT_EQ(replay.processed_access_count, 3u);
  EXPECT_EQ(replay.processed_atomic_count, 4u);
  EXPECT_EQ(replay.unsupported_atomic_count, 0u);
  EXPECT_FALSE(replay.metadata_full);
  EXPECT_FALSE(replay.conflict);
  EXPECT_EQ(header.diagnostic_count, 0u);
}

TEST(ConSanMoi, RecordReplayPropagatesOrderingAcrossAtomicAndFence) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/3,
      /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/1,
      /*sampled_watchpoint_capacity=*/0);
  header.access_record_count = 3;

  std::array<ConSanMoiAccessRecord, 3> records{};
  for (uint32_t owner = 0; owner < records.size(); ++owner) {
    records[owner].wave_id = owner;
    records[owner].event_index = owner * 3;
    records[owner].instruction_offset = 0x10 + owner * 0x10;
    records[owner].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write);
    records[owner].lds_byte_count = 4;
    records[owner].cell_count = 1;
  }

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

  std::array<ConSanMoiRecordReplayFenceEvent, 2> fences{};
  fences[0].owner_id = 1;
  fences[0].instruction_offset = 0x300;
  fences[0].event_index = 4;
  fences[0].kind = ConSanMoiFenceEventKind::Release;
  fences[0].scope = 1;
  fences[0].communication_token = 0x5000;
  fences[1].owner_id = 2;
  fences[1].instruction_offset = 0x400;
  fences[1].event_index = 5;
  fences[1].kind = ConSanMoiFenceEventKind::Acquire;
  fences[1].scope = 1;
  fences[1].communication_token = 0x5000;

  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
  std::array<uint64_t, 1> shadow{};
  const ConSanMoiRecordReplayResult replay = consan_moi_record_replay_access_records(
      header, records, std::span<const ConSanMoiBarrierRecord>{}, atomics, fences, diagnostics,
      shadow);

  EXPECT_EQ(replay.processed_access_count, 3u);
  EXPECT_EQ(replay.processed_atomic_count, 2u);
  EXPECT_EQ(replay.processed_fence_count, 2u);
  EXPECT_FALSE(replay.metadata_full);
  EXPECT_FALSE(replay.conflict);
  EXPECT_EQ(header.diagnostic_count, 0u);
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
  EXPECT_EQ(diagnostic.first_lane_mask, 0x1u);
  EXPECT_EQ(diagnostic.second_lane_mask, 0x2u);
  EXPECT_EQ(diagnostic.first_instruction_offset, 0x10u);
  EXPECT_EQ(diagnostic.second_instruction_offset, 0x20u);
  EXPECT_EQ(diagnostic.first_lds_byte_offset, 8u);
  EXPECT_EQ(diagnostic.first_lds_byte_count, 4u);
  EXPECT_EQ(diagnostic.second_lds_byte_offset, 8u);
  EXPECT_EQ(diagnostic.second_lds_byte_count, 4u);
  EXPECT_EQ(diagnostic.first_access_kind, static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write));
  EXPECT_EQ(diagnostic.second_access_kind, static_cast<uint32_t>(ConSanMoiShadowAccessKind::Read));
}

TEST(ConSanMoi, RecordReplayAdjacentSubwordRangesInOneCellStayClean) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/2,
      /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/1,
      /*sampled_watchpoint_capacity=*/0);
  header.access_record_count = 2;

  std::array<ConSanMoiAccessRecord, 2> records{};
  records[0].wave_id = 1;
  records[0].lane_mask = 0x1;
  records[0].instruction_offset = 0x10;
  records[0].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write);
  records[0].lds_byte_offset = 0;
  records[0].lds_byte_count = 2;
  records[0].cell_count = 1;

  records[1].wave_id = 2;
  records[1].lane_mask = 0x2;
  records[1].instruction_offset = 0x20;
  records[1].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write);
  records[1].lds_byte_offset = 2;
  records[1].lds_byte_count = 2;
  records[1].cell_count = 1;

  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
  std::array<uint64_t, 1> shadow{};
  const ConSanMoiRecordReplayResult replay =
      consan_moi_record_replay_access_records(header, records, diagnostics, shadow);

  EXPECT_FALSE(replay.conflict);
  EXPECT_FALSE(replay.metadata_full);
  EXPECT_EQ(replay.emitted_diagnostic_count, 0u);
  EXPECT_EQ(header.diagnostic_count, 0u);
}

TEST(ConSanMoi, SparseExactByteShadowPreservesSplitProvenance) {
  ConSanMoiSparseExactByteShadow model(/*byte_capacity=*/8, /*maximum_access_count=*/4);
  ConSanMoiExactByteAccess outer{
      /*generation=*/7,
      /*owner_id=*/1,
      /*epoch=*/3,
      ConSanMoiShadowAccessKind::Write,
      /*lds_byte_offset=*/0,
      /*lds_byte_count=*/8,
      /*instruction_offset=*/0x10,
      /*lane_mask=*/0x1,
  };
  ConSanMoiExactByteAccess middle = outer;
  middle.lds_byte_offset = 2;
  middle.lds_byte_count = 2;
  middle.instruction_offset = 0x20;

  EXPECT_FALSE(model.access(outer).conflict);
  EXPECT_FALSE(model.access(middle).conflict);

  ConSanMoiExactByteAccess probe = middle;
  probe.owner_id = 2;
  probe.instruction_offset = 0x30;
  probe.lds_byte_count = 1;

  probe.lds_byte_offset = 0;
  const auto left = model.access(probe);
  ASSERT_TRUE(left.conflict);
  ASSERT_TRUE(left.prior);
  EXPECT_EQ(left.prior->instruction_offset, outer.instruction_offset);

  probe.lds_byte_offset = 2;
  const auto center = model.access(probe);
  ASSERT_TRUE(center.conflict);
  ASSERT_TRUE(center.prior);
  EXPECT_EQ(center.prior->instruction_offset, middle.instruction_offset);

  probe.lds_byte_offset = 6;
  const auto right = model.access(probe);
  ASSERT_TRUE(right.conflict);
  ASSERT_TRUE(right.prior);
  EXPECT_EQ(right.prior->instruction_offset, outer.instruction_offset);
}

TEST(ConSanMoi, SparseExactByteShadowRetainsSameSiteGroupAcrossOrderedSite) {
  ConSanMoiSparseExactByteShadow model(/*byte_capacity=*/4, /*maximum_access_count=*/3);
  ConSanMoiExactByteAccess first{
      /*generation=*/7,
      /*owner_id=*/1,
      /*epoch=*/3,
      ConSanMoiShadowAccessKind::Write,
      /*lds_byte_offset=*/0,
      /*lds_byte_count=*/4,
      /*instruction_offset=*/0x10,
      /*lane_mask=*/0x1,
      /*exact_address_group=*/true,
  };
  ConSanMoiExactByteAccess ordered = first;
  ordered.instruction_offset = 0x20;
  ConSanMoiExactByteAccess second_group = first;
  second_group.lane_mask = 0x2;

  EXPECT_FALSE(model.access(first).conflict);
  EXPECT_FALSE(model.access(ordered).conflict);
  const ConSanMoiExactByteAccessResult conflict = model.access(second_group);
  ASSERT_TRUE(conflict.conflict);
  ASSERT_TRUE(conflict.prior);
  EXPECT_EQ(conflict.prior->instruction_offset, first.instruction_offset);
  EXPECT_EQ(conflict.prior->lane_mask, first.lane_mask);
}

TEST(ConSanMoi, SparseExactByteShadowRetainsEverySameSiteLaneGroup) {
  ConSanMoiSparseExactByteShadow model(/*byte_capacity=*/4, /*maximum_access_count=*/3);
  ConSanMoiExactByteAccess first_group{
      /*generation=*/7,
      /*owner_id=*/1,
      /*epoch=*/3,
      ConSanMoiShadowAccessKind::Read,
      /*lds_byte_offset=*/0,
      /*lds_byte_count=*/4,
      /*instruction_offset=*/0x10,
      /*lane_mask=*/0x2,
      /*exact_address_group=*/true,
  };
  ConSanMoiExactByteAccess second_group = first_group;
  second_group.lane_mask = 0x1;
  ConSanMoiExactByteAccess write = second_group;
  write.kind = ConSanMoiShadowAccessKind::Write;

  EXPECT_FALSE(model.access(first_group).conflict);
  EXPECT_FALSE(model.access(second_group).conflict);
  const ConSanMoiExactByteAccessResult conflict = model.access(write);
  ASSERT_TRUE(conflict.conflict);
  ASSERT_TRUE(conflict.prior);
  EXPECT_EQ(conflict.prior->lane_mask, first_group.lane_mask);
}

TEST(ConSanMoi, SparseExactByteShadowRetainsSelectivelyOrderedOwners) {
  ConSanMoiSparseExactByteShadow model(/*byte_capacity=*/4, /*maximum_access_count=*/3);
  ConSanMoiExactByteAccess first{
      /*generation=*/7,
      /*owner_id=*/1,
      /*epoch=*/0,
      ConSanMoiShadowAccessKind::Read,
      /*lds_byte_offset=*/0,
      /*lds_byte_count=*/4,
      /*instruction_offset=*/0x10,
      /*lane_mask=*/0x1,
  };
  ConSanMoiExactByteAccess second = first;
  second.owner_id = 2;
  second.instruction_offset = 0x20;
  ConSanMoiExactByteAccess probe = first;
  probe.owner_id = 3;
  probe.kind = ConSanMoiShadowAccessKind::Write;
  probe.instruction_offset = 0x30;
  const std::array<ConSanMoiAcquiredEpochToken, 1> second_to_probe{{
      ConSanMoiAcquiredEpochToken{
          /*valid=*/true,
          /*generation=*/7,
          /*consumer_owner_id=*/3,
          /*producer_owner_id=*/2,
          /*producer_epoch_plus_one=*/1,
          /*acquire_instruction_offset=*/0x200,
      },
  }};

  EXPECT_FALSE(model.access(first).conflict);
  EXPECT_FALSE(model.access(second).conflict);
  const ConSanMoiExactByteAccessResult conflict = model.access(probe, second_to_probe);
  ASSERT_TRUE(conflict.conflict);
  ASSERT_TRUE(conflict.prior);
  EXPECT_EQ(conflict.prior->owner_id, first.owner_id);
}

TEST(ConSanMoi, SparseExactByteShadowRetiresOnlyProvenEpochsAndReclaimsCapacity) {
  ConSanMoiSparseExactByteShadow model(/*byte_capacity=*/16, /*maximum_access_count=*/3);
  ConSanMoiExactByteAccess retired{
      /*generation=*/7,
      /*owner_id=*/1,
      /*epoch=*/0,
      ConSanMoiShadowAccessKind::Write,
      /*lds_byte_offset=*/0,
      /*lds_byte_count=*/1,
      /*instruction_offset=*/0x10,
      /*lane_mask=*/0x1,
  };
  ConSanMoiExactByteAccess boundary = retired;
  boundary.epoch = 1;
  boundary.lds_byte_offset = 4;
  boundary.instruction_offset = 0x20;
  ConSanMoiExactByteAccess other_generation = retired;
  other_generation.generation = 8;
  other_generation.lds_byte_offset = 8;
  other_generation.instruction_offset = 0x30;

  EXPECT_FALSE(model.access(retired).capacity_exhausted);
  EXPECT_FALSE(model.access(boundary).capacity_exhausted);
  EXPECT_FALSE(model.access(other_generation).capacity_exhausted);

  model.retire_before_epoch(/*generation=*/7, /*first_live_epoch=*/1);

  ConSanMoiExactByteAccess reclaimed = boundary;
  reclaimed.lds_byte_offset = 12;
  reclaimed.instruction_offset = 0x40;
  EXPECT_FALSE(model.access(reclaimed).capacity_exhausted);

  ConSanMoiExactByteAccess boundary_probe = boundary;
  boundary_probe.owner_id = 2;
  boundary_probe.kind = ConSanMoiShadowAccessKind::Read;
  boundary_probe.instruction_offset = 0x50;
  const ConSanMoiExactByteAccessResult boundary_conflict = model.access(boundary_probe);
  ASSERT_TRUE(boundary_conflict.conflict);
  ASSERT_TRUE(boundary_conflict.prior);
  EXPECT_EQ(boundary_conflict.prior->instruction_offset, boundary.instruction_offset);

  ConSanMoiExactByteAccess generation_probe = other_generation;
  generation_probe.owner_id = 2;
  generation_probe.kind = ConSanMoiShadowAccessKind::Read;
  generation_probe.instruction_offset = 0x60;
  const ConSanMoiExactByteAccessResult generation_conflict = model.access(generation_probe);
  ASSERT_TRUE(generation_conflict.conflict);
  ASSERT_TRUE(generation_conflict.prior);
  EXPECT_EQ(generation_conflict.prior->instruction_offset, other_generation.instruction_offset);

  ConSanMoiExactByteAccess full = boundary;
  full.lds_byte_offset = 1;
  full.instruction_offset = 0x70;
  EXPECT_TRUE(model.access(full).capacity_exhausted);
}

TEST(ConSanMoi, SparseExactByteShadowRetirementKeepsSameSiteLaneGroup) {
  ConSanMoiSparseExactByteShadow model(/*byte_capacity=*/8, /*maximum_access_count=*/2);
  ConSanMoiExactByteAccess retired{
      /*generation=*/7,
      /*owner_id=*/1,
      /*epoch=*/0,
      ConSanMoiShadowAccessKind::Write,
      /*lds_byte_offset=*/0,
      /*lds_byte_count=*/4,
      /*instruction_offset=*/0x10,
      /*lane_mask=*/0x1,
      /*exact_address_group=*/true,
  };
  ConSanMoiExactByteAccess boundary = retired;
  boundary.epoch = 1;
  boundary.lds_byte_offset = 4;
  boundary.instruction_offset = 0x20;

  EXPECT_FALSE(model.access(retired).conflict);
  EXPECT_FALSE(model.access(boundary).conflict);
  model.retire_before_epoch(/*generation=*/7, /*first_live_epoch=*/1);

  ConSanMoiExactByteAccess second_group = boundary;
  second_group.lane_mask = 0x2;
  const ConSanMoiExactByteAccessResult conflict = model.access(second_group);
  ASSERT_TRUE(conflict.conflict);
  ASSERT_TRUE(conflict.prior);
  EXPECT_EQ(conflict.prior->instruction_offset, boundary.instruction_offset);
  EXPECT_EQ(conflict.prior->lane_mask, boundary.lane_mask);
}

TEST(ConSanMoi, SparseExactByteShadowScalesAcrossLargeOrderedSiteInventory) {
  constexpr uint32_t site_count = 10000;
  ConSanMoiSparseExactByteShadow model(/*byte_capacity=*/4,
                                       /*maximum_access_count=*/site_count + 1);
  ConSanMoiExactByteAccess access{
      /*generation=*/7,
      /*owner_id=*/1,
      /*epoch=*/3,
      ConSanMoiShadowAccessKind::Write,
      /*lds_byte_offset=*/0,
      /*lds_byte_count=*/4,
      /*instruction_offset=*/0,
      /*lane_mask=*/0x1,
      /*exact_address_group=*/true,
  };
  for (uint32_t site = 0; site < site_count; ++site) {
    access.instruction_offset = site * 4;
    const ConSanMoiExactByteAccessResult admitted = model.access(access);
    ASSERT_FALSE(admitted.conflict) << site;
    ASSERT_FALSE(admitted.capacity_exhausted) << site;
  }

  access.instruction_offset = 0;
  access.lane_mask = 0x2;
  const ConSanMoiExactByteAccessResult conflict = model.access(access);
  ASSERT_TRUE(conflict.conflict);
  ASSERT_TRUE(conflict.prior);
  EXPECT_EQ(conflict.prior->instruction_offset, 0u);
  EXPECT_EQ(conflict.prior->lane_mask, 0x1u);
}

TEST(ConSanMoi, SparseExactByteShadowFailsClosedAtStructuralAccessBound) {
  ConSanMoiSparseExactByteShadow model(/*byte_capacity=*/8, /*maximum_access_count=*/1);
  ConSanMoiExactByteAccess first{
      /*generation=*/7,
      /*owner_id=*/1,
      /*epoch=*/3,
      ConSanMoiShadowAccessKind::Write,
      /*lds_byte_offset=*/0,
      /*lds_byte_count=*/8,
      /*instruction_offset=*/0x10,
      /*lane_mask=*/0x1,
  };
  EXPECT_FALSE(model.access(first).conflict);

  ConSanMoiExactByteAccess ordered = first;
  ordered.lds_byte_offset = 2;
  ordered.lds_byte_count = 2;
  ordered.instruction_offset = 0x20;
  const auto full = model.access(ordered);
  EXPECT_TRUE(full.capacity_exhausted);
  EXPECT_FALSE(full.conflict);

  ConSanMoiExactByteAccess conflicting = first;
  conflicting.owner_id = 2;
  conflicting.kind = ConSanMoiShadowAccessKind::Read;
  conflicting.instruction_offset = 0x30;
  const auto conflict = model.access(conflicting);
  ASSERT_TRUE(conflict.conflict);
  ASSERT_TRUE(conflict.prior);
  EXPECT_EQ(conflict.prior->instruction_offset, first.instruction_offset);
}

TEST(ConSanMoi, SparseExactByteShadowPreservesRemainderAboveUint32Boundary) {
  constexpr uint64_t byte_capacity =
      static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 5;
  ConSanMoiSparseExactByteShadow model(byte_capacity, /*maximum_access_count=*/3);
  ConSanMoiExactByteAccess outer{
      /*generation=*/7,
      /*owner_id=*/1,
      /*epoch=*/3,
      ConSanMoiShadowAccessKind::Write,
      /*lds_byte_offset=*/std::numeric_limits<uint32_t>::max() - 3,
      /*lds_byte_count=*/8,
      /*instruction_offset=*/0x10,
      /*lane_mask=*/0x1,
  };
  ConSanMoiExactByteAccess middle = outer;
  middle.kind = ConSanMoiShadowAccessKind::Read;
  middle.lds_byte_offset = std::numeric_limits<uint32_t>::max() - 1;
  middle.lds_byte_count = 2;
  middle.instruction_offset = 0x20;

  EXPECT_FALSE(model.access(outer).conflict);
  EXPECT_FALSE(model.access(middle).conflict);

  ConSanMoiExactByteAccess probe = middle;
  probe.owner_id = 2;
  probe.lds_byte_offset = std::numeric_limits<uint32_t>::max();
  probe.lds_byte_count = 2;
  probe.instruction_offset = 0x30;
  const auto result = model.access(probe);
  ASSERT_TRUE(result.conflict);
  ASSERT_TRUE(result.prior);
  EXPECT_EQ(result.prior->instruction_offset, outer.instruction_offset);
}

TEST(ConSanMoi, SparseExactByteShadowMatchesFullHistoryOracle) {
  constexpr uint32_t byte_capacity = 64;
  constexpr uint32_t access_count = 40;
  for (uint64_t seed = 1; seed <= 256; ++seed) {
    SCOPED_TRACE(seed);
    uint64_t random_state = seed;
    const auto next_random = [&]() {
      random_state = random_state * 6364136223846793005ull + 1442695040888963407ull;
      return random_state;
    };

    ConSanMoiSparseExactByteShadow model(byte_capacity, access_count);
    std::vector<ConSanMoiExactByteAccess> history;
    for (uint32_t access_index = 0; access_index < access_count; ++access_index) {
      SCOPED_TRACE(access_index);
      ConSanMoiExactByteAccess current;
      current.generation = 7 + next_random() % 3;
      current.owner_id = 1 + static_cast<uint32_t>(next_random() % 4);
      current.epoch = static_cast<uint32_t>(next_random() % 3);
      current.kind =
          static_cast<ConSanMoiShadowAccessKind>(1 + static_cast<uint32_t>(next_random() % 4));
      current.lds_byte_count = 1 + static_cast<uint32_t>(next_random() % 8);
      current.lds_byte_offset =
          static_cast<uint32_t>(next_random() % (byte_capacity + 1 - current.lds_byte_count));
      current.instruction_offset = static_cast<uint32_t>(next_random() % 8) * 4;
      current.lane_mask = uint64_t{1} << (next_random() % 8);
      current.exact_address_group = (next_random() & 1u) != 0;

      std::array<ConSanMoiAcquiredEpochToken, 4> tokens{};
      for (ConSanMoiAcquiredEpochToken &token : tokens) {
        token.valid = (next_random() & 1u) != 0;
        token.generation = (next_random() & 3u) == 0 ? current.generation + 1 : current.generation;
        token.consumer_owner_id =
            (next_random() & 3u) == 0 ? current.owner_id % 4 + 1 : current.owner_id;
        token.producer_owner_id = 1 + static_cast<uint32_t>(next_random() % 4);
        token.producer_epoch_plus_one = 1 + static_cast<uint32_t>(next_random() % 4);
        token.acquire_instruction_offset = static_cast<uint32_t>(next_random() % 8) * 4;
      }

      const ConSanMoiExactShadowEntry current_entry{
          current.kind,
          current.owner_id,
          current.epoch,
          static_cast<uint32_t>(current.generation),
          current.instruction_offset,
      };

      std::optional<ConSanMoiExactByteAccess> history_prior;
      for (auto prior = history.rbegin(); prior != history.rend(); ++prior) {
        if (!consan_moi_exact_byte_accesses_conflict(current, *prior))
          continue;
        const ConSanMoiExactShadowEntry prior_entry{
            prior->kind,
            prior->owner_id,
            prior->epoch,
            static_cast<uint32_t>(prior->generation),
            prior->instruction_offset,
        };
        if (consan_moi_acquired_epoch_orders(tokens, current_entry, prior_entry))
          continue;
        history_prior = *prior;
        break;
      }

      const ConSanMoiExactByteAccessResult sparse = model.access(current, tokens);
      ASSERT_FALSE(sparse.capacity_exhausted);
      EXPECT_EQ(sparse.conflict, history_prior.has_value());
      EXPECT_EQ(sparse.prior.has_value(), history_prior.has_value());
      if (history_prior) {
        ASSERT_TRUE(sparse.prior);
        EXPECT_EQ(sparse.prior->owner_id, history_prior->owner_id);
        EXPECT_EQ(sparse.prior->epoch, history_prior->epoch);
        EXPECT_EQ(sparse.prior->kind, history_prior->kind);
        EXPECT_EQ(sparse.prior->lds_byte_offset, history_prior->lds_byte_offset);
        EXPECT_EQ(sparse.prior->lds_byte_count, history_prior->lds_byte_count);
        EXPECT_EQ(sparse.prior->instruction_offset, history_prior->instruction_offset);
        EXPECT_EQ(sparse.prior->lane_mask, history_prior->lane_mask);
        EXPECT_EQ(sparse.prior->exact_address_group, history_prior->exact_address_group);
        continue;
      }
      history.push_back(current);
    }
  }
}

TEST(ConSanMoi, RecordReplayMultiLaneWriteReportsExactSameWaveConflict) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/1,
      /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/2,
      /*sampled_watchpoint_capacity=*/0);
  header.access_record_count = 1;

  ConSanMoiAccessRecord record{};
  record.wave_id = 3;
  record.lane_mask = 0x0000000f0000000full;
  record.instruction_offset = 0x10bc;
  record.access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write);
  record.lds_byte_offset = 4;
  record.lds_byte_count = 2;
  record.start_cell = 1;
  record.cell_count = 1;
  record.flags = kConSanMoiAccessRecordFlagExactAddressGroupMask;

  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
  std::array<uint64_t, 2> shadow{};
  const ConSanMoiRecordReplayResult replay = consan_moi_record_replay_access_records(
      header, std::span<const ConSanMoiAccessRecord>(&record, 1), diagnostics, shadow);

  ASSERT_TRUE(replay.conflict);
  ASSERT_EQ(replay.emitted_diagnostic_count, 1u);
  ASSERT_EQ(header.diagnostic_count, 1u);
  const ConSanMoiDiagnosticRecord &diagnostic = diagnostics[0];
  EXPECT_EQ(diagnostic.first_owner_id, 3u);
  EXPECT_EQ(diagnostic.second_owner_id, 3u);
  EXPECT_EQ(diagnostic.first_lane_mask, 0x1u);
  EXPECT_EQ(diagnostic.second_lane_mask, 0x0000000f0000000eull);
  EXPECT_EQ(diagnostic.first_instruction_offset, 0x10bcu);
  EXPECT_EQ(diagnostic.second_instruction_offset, 0x10bcu);
  EXPECT_EQ(diagnostic.first_lds_byte_offset, 4u);
  EXPECT_EQ(diagnostic.first_lds_byte_count, 2u);
  EXPECT_EQ(diagnostic.second_lds_byte_offset, 4u);
  EXPECT_EQ(diagnostic.second_lds_byte_count, 2u);
  EXPECT_EQ(diagnostic.first_access_kind, static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write));
  EXPECT_EQ(diagnostic.second_access_kind, static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write));
}

TEST(ConSanMoi, RecordReplayDistinctSameWaveGroupsConflictOnlyOnByteOverlap) {
  const auto replay_ranges = [](uint32_t second_offset, uint64_t second_mask) {
    ConSanMoiReportHeader header = make_consan_moi_report_header(
        /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/2,
        /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/1,
        /*sampled_watchpoint_capacity=*/0);
    header.access_record_count = 2;

    std::array<ConSanMoiAccessRecord, 2> records{};
    for (ConSanMoiAccessRecord &record : records) {
      record.wave_id = 3;
      record.access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write);
      record.lds_byte_count = 2;
      record.cell_count = 1;
      record.flags = kConSanMoiAccessRecordFlagExactAddressGroupMask;
    }
    records[0].lane_mask = 0x1;
    records[0].instruction_offset = 0x40;
    records[0].lds_byte_offset = 0;
    records[1].lane_mask = second_mask;
    records[1].instruction_offset = 0x40;
    records[1].lds_byte_offset = second_offset;

    std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
    std::array<uint64_t, 1> shadow{};
    const ConSanMoiRecordReplayResult replay =
        consan_moi_record_replay_access_records(header, records, diagnostics, shadow);
    return std::pair{replay, diagnostics[0]};
  };

  const auto [overlap, diagnostic] = replay_ranges(/*second_offset=*/1, /*second_mask=*/0x4);
  EXPECT_TRUE(overlap.conflict);
  EXPECT_EQ(overlap.emitted_diagnostic_count, 1u);
  EXPECT_EQ(diagnostic.first_owner_id, 3u);
  EXPECT_EQ(diagnostic.second_owner_id, 3u);
  EXPECT_EQ(diagnostic.first_lane_mask, 0x1u);
  EXPECT_EQ(diagnostic.second_lane_mask, 0x4u);
  EXPECT_EQ(diagnostic.first_lds_byte_offset, 0u);
  EXPECT_EQ(diagnostic.first_lds_byte_count, 2u);
  EXPECT_EQ(diagnostic.second_lds_byte_offset, 1u);
  EXPECT_EQ(diagnostic.second_lds_byte_count, 2u);

  const auto [adjacent, ignored] = replay_ranges(/*second_offset=*/2, /*second_mask=*/0x4);
  (void)ignored;
  EXPECT_FALSE(adjacent.conflict);

  const auto [same_group, also_ignored] = replay_ranges(/*second_offset=*/0, /*second_mask=*/0x1);
  (void)also_ignored;
  EXPECT_FALSE(same_group.conflict);
}

TEST(ConSanMoi, RecordReplayRetainsSameSiteGroupAcrossOrderedSite) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/3,
      /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/1,
      /*sampled_watchpoint_capacity=*/0);
  header.access_record_count = 3;

  std::array<ConSanMoiAccessRecord, 3> records{};
  for (uint32_t index = 0; index < records.size(); ++index) {
    records[index].wave_id = 3;
    records[index].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write);
    records[index].lds_byte_offset = 0;
    records[index].lds_byte_count = 4;
    records[index].cell_count = 1;
    records[index].flags = kConSanMoiAccessRecordFlagExactAddressGroupMask;
    // Automatic capture increments the publication counter inside its
    // address-group loop, so concurrent groups from one execution have
    // distinct event indices.
    records[index].event_index = index + 1;
  }
  records[0].lane_mask = 0x1;
  records[0].instruction_offset = 0x10;
  records[1].lane_mask = 0x1;
  records[1].instruction_offset = 0x20;
  records[2].lane_mask = 0x2;
  records[2].instruction_offset = 0x10;

  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
  std::array<uint64_t, 1> shadow{};
  const ConSanMoiRecordReplayResult replay =
      consan_moi_record_replay_access_records(header, records, diagnostics, shadow);

  ASSERT_TRUE(replay.conflict);
  ASSERT_EQ(replay.emitted_diagnostic_count, 1u);
  EXPECT_EQ(diagnostics[0].first_owner_id, 3u);
  EXPECT_EQ(diagnostics[0].second_owner_id, 3u);
  EXPECT_EQ(diagnostics[0].first_instruction_offset, 0x10u);
  EXPECT_EQ(diagnostics[0].second_instruction_offset, 0x10u);
  EXPECT_EQ(diagnostics[0].first_lane_mask, 0x1u);
  EXPECT_EQ(diagnostics[0].second_lane_mask, 0x2u);
}

TEST(ConSanMoi, RecordReplaySameWaveDifferentSitesAreProgramOrdered) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/2,
      /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/1,
      /*sampled_watchpoint_capacity=*/0);
  header.access_record_count = 2;

  std::array<ConSanMoiAccessRecord, 2> records{};
  for (ConSanMoiAccessRecord &record : records) {
    record.wave_id = 3;
    record.access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write);
    record.lds_byte_count = 4;
    record.cell_count = 1;
    record.flags = kConSanMoiAccessRecordFlagExactAddressGroupMask;
  }
  records[0].lane_mask = 0x1;
  records[0].instruction_offset = 0x40;
  records[1].lane_mask = 0x4;
  records[1].instruction_offset = 0x48;

  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
  std::array<uint64_t, 1> shadow{};
  const ConSanMoiRecordReplayResult replay =
      consan_moi_record_replay_access_records(header, records, diagnostics, shadow);

  EXPECT_FALSE(replay.conflict);
  EXPECT_EQ(replay.emitted_diagnostic_count, 0u);
}

TEST(ConSanMoi, RecordReplayDirectWaveMaskDoesNotInventAnIntraWaveConflict) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/1,
      /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/1,
      /*sampled_watchpoint_capacity=*/0);
  header.access_record_count = 1;

  ConSanMoiAccessRecord record{};
  record.wave_id = 3;
  record.instruction_offset = 0x40;
  record.access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write);
  record.lds_byte_count = 4;
  record.cell_count = 1;
  record.lane_mask = std::numeric_limits<uint64_t>::max();

  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
  std::array<uint64_t, 1> shadow{};
  const ConSanMoiRecordReplayResult replay = consan_moi_record_replay_access_records(
      header, std::span<const ConSanMoiAccessRecord>(&record, 1), diagnostics, shadow);

  EXPECT_FALSE(replay.conflict);
  EXPECT_EQ(replay.processed_access_count, 1u);
  EXPECT_EQ(replay.emitted_diagnostic_count, 0u);
}

TEST(ConSanMoi, RecordReplayUnflaggedSameSiteLaneGroupsAreNotConcurrent) {
  constexpr uint32_t exact_group = kConSanMoiAccessRecordFlagExactAddressGroupMask;
  for (const auto &[first_flags, second_flags] :
       std::array{std::pair{0u, 0u}, std::pair{exact_group, 0u}, std::pair{0u, exact_group}}) {
    SCOPED_TRACE(testing::Message() << first_flags << ", " << second_flags);
    ConSanMoiReportHeader header = make_consan_moi_report_header(
        /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/2,
        /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/1,
        /*sampled_watchpoint_capacity=*/0);
    header.access_record_count = 2;

    std::array<ConSanMoiAccessRecord, 2> records{};
    for (ConSanMoiAccessRecord &record : records) {
      record.wave_id = 3;
      record.instruction_offset = 0x40;
      record.access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write);
      record.lds_byte_count = 4;
      record.cell_count = 1;
    }
    records[0].lane_mask = 0x1;
    records[0].flags = first_flags;
    records[1].lane_mask = 0x4;
    records[1].flags = second_flags;

    std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
    std::array<uint64_t, 1> shadow{};
    const ConSanMoiRecordReplayResult replay =
        consan_moi_record_replay_access_records(header, records, diagnostics, shadow);

    EXPECT_FALSE(replay.conflict);
    EXPECT_EQ(replay.processed_access_count, 2u);
    EXPECT_EQ(replay.emitted_diagnostic_count, 0u);
  }
}

TEST(ConSanMoi, RecordReplayExactMultiLaneReadWriteReportsIntraWaveConflict) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/1,
      /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/1,
      /*sampled_watchpoint_capacity=*/0);
  header.access_record_count = 1;

  ConSanMoiAccessRecord record{};
  record.wave_id = 3;
  record.lane_mask = 0x3;
  record.instruction_offset = 0x40;
  record.access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::ReadWrite);
  record.lds_byte_count = 4;
  record.cell_count = 1;
  record.flags = kConSanMoiAccessRecordFlagExactAddressGroupMask;

  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
  std::array<uint64_t, 1> shadow{};
  const ConSanMoiRecordReplayResult replay = consan_moi_record_replay_access_records(
      header, std::span<const ConSanMoiAccessRecord>(&record, 1), diagnostics, shadow);

  EXPECT_TRUE(replay.conflict);
  ASSERT_EQ(replay.emitted_diagnostic_count, 1u);
  EXPECT_EQ(diagnostics[0].first_access_kind,
            static_cast<uint32_t>(ConSanMoiShadowAccessKind::ReadWrite));
  EXPECT_EQ(diagnostics[0].second_access_kind,
            static_cast<uint32_t>(ConSanMoiShadowAccessKind::ReadWrite));
}

TEST(ConSanMoi, RecordReplayDeduplicatesByDiagnosticKindAndExactOverlap) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/7,
      /*diagnostic_capacity=*/4, /*exact_shadow_entry_capacity=*/5,
      /*sampled_watchpoint_capacity=*/0);
  header.access_record_count = 7;

  std::array<ConSanMoiAccessRecord, 7> records{};
  for (ConSanMoiAccessRecord &record : records) {
    record.access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write);
    record.lds_byte_count = 4;
    record.cell_count = 1;
    record.lane_mask = 1;
  }
  records[0].wave_id = 1;
  records[0].instruction_offset = 0x10;
  records[1].wave_id = 2;
  records[1].instruction_offset = 0x20;

  records[2] = records[0];
  records[2].lds_byte_offset = 16;
  records[2].start_cell = 4;
  records[3] = records[1];
  records[3].lds_byte_offset = 16;
  records[3].start_cell = 4;

  records[4] = records[0];
  records[4].instruction_offset = 0x30;
  records[4].lds_byte_offset = 4096;
  records[4].start_cell = 1024;
  records[5] = records[0];
  records[5].instruction_offset = 0;
  records[6] = records[1];
  records[6].instruction_offset = 0x30;

  std::array<ConSanMoiDiagnosticRecord, 4> diagnostics{};
  std::array<uint64_t, 5> shadow{};
  const ConSanMoiRecordReplayResult replay =
      consan_moi_record_replay_access_records(header, records, diagnostics, shadow);

  EXPECT_TRUE(replay.conflict);
  EXPECT_TRUE(replay.metadata_full);
  ASSERT_EQ(replay.emitted_diagnostic_count, 4u);
  EXPECT_EQ(diagnostics[0].kind, static_cast<uint32_t>(ConSanMoiDiagnosticKind::AccessConflict));
  EXPECT_EQ(diagnostics[0].first_lds_byte_offset, 0u);
  EXPECT_EQ(diagnostics[1].kind, static_cast<uint32_t>(ConSanMoiDiagnosticKind::AccessConflict));
  EXPECT_EQ(diagnostics[1].first_lds_byte_offset, 16u);
  EXPECT_EQ(diagnostics[2].kind, static_cast<uint32_t>(ConSanMoiDiagnosticKind::MetadataFull));
  EXPECT_EQ(diagnostics[3].kind, static_cast<uint32_t>(ConSanMoiDiagnosticKind::AccessConflict));
}

TEST(ConSanMoi, RecordReplaySparseProvenanceDoesNotScaleByWorkgroupLdsExtent) {
  constexpr uint32_t kWorkgroupCount = 1024;
  constexpr uint32_t kLdsBytes = 160u * 1024u;
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/kWorkgroupCount,
      /*diagnostic_capacity=*/1,
      /*exact_shadow_entry_capacity=*/kLdsBytes / consan_moi_exact_shadow::granule_bytes,
      /*sampled_watchpoint_capacity=*/0);
  header.access_record_count = kWorkgroupCount;

  std::vector<ConSanMoiAccessRecord> records(kWorkgroupCount);
  for (uint32_t workgroup = 0; workgroup < kWorkgroupCount; ++workgroup) {
    ConSanMoiAccessRecord &record = records[workgroup];
    record.workgroup_x = workgroup;
    record.wave_id = 1;
    record.lane_mask = 1;
    record.instruction_offset = 0x10;
    record.access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write);
    record.lds_byte_offset = kLdsBytes - sizeof(uint32_t);
    record.lds_byte_count = sizeof(uint32_t);
    record.start_cell = kLdsBytes / consan_moi_exact_shadow::granule_bytes - 1u;
    record.cell_count = 1;
  }

  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
  std::vector<uint64_t> shadow(kLdsBytes / consan_moi_exact_shadow::granule_bytes);
  const ConSanMoiRecordReplayResult replay =
      consan_moi_record_replay_access_records(header, records, diagnostics, shadow);

  EXPECT_FALSE(replay.conflict);
  EXPECT_FALSE(replay.metadata_full);
  EXPECT_EQ(replay.processed_access_count, kWorkgroupCount);
  EXPECT_NE(shadow.back(), 0u);
}

TEST(ConSanMoi, RecordReplayCollapsedWorkgroupIdentityCreatesCrossLdsConflict) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/2,
      /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/3,
      /*sampled_watchpoint_capacity=*/0);
  header.access_record_count = 2;
  std::array<ConSanMoiAccessRecord, 2> records{};
  for (uint32_t index = 0; index < records.size(); ++index) {
    records[index].wave_id = index + 1u;
    records[index].lane_mask = uint64_t{1} << index;
    records[index].instruction_offset = 0x10u + index * 8u;
    records[index].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write);
    records[index].lds_byte_offset = 8u;
    records[index].lds_byte_count = 4u;
    records[index].epoch = 3u;
  }
  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
  std::array<uint64_t, 3> shadow{};

  const ConSanMoiRecordReplayResult replay =
      consan_moi_record_replay_access_records(header, records, diagnostics, shadow);

  EXPECT_TRUE(replay.conflict);
  EXPECT_EQ(replay.emitted_diagnostic_count, 1u);
  EXPECT_EQ(header.diagnostic_count, 1u);
  EXPECT_EQ(diagnostics[0].kind, static_cast<uint32_t>(ConSanMoiDiagnosticKind::AccessConflict));
}

TEST(ConSanMoi, RecordReplayExactWorkgroupTupleSeparatesLdsInstances) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/2,
      /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/3,
      /*sampled_watchpoint_capacity=*/0);
  header.access_record_count = 2;
  std::array<ConSanMoiAccessRecord, 2> records{};
  for (uint32_t index = 0; index < records.size(); ++index) {
    records[index].workgroup_x = index;
    records[index].wave_id = index + 1u;
    records[index].lane_mask = uint64_t{1} << index;
    records[index].instruction_offset = 0x10u + index * 8u;
    records[index].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write);
    records[index].lds_byte_offset = 8u;
    records[index].lds_byte_count = 4u;
    records[index].epoch = 3u;
  }
  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
  std::array<uint64_t, 3> shadow{};

  const ConSanMoiRecordReplayResult replay =
      consan_moi_record_replay_access_records(header, records, diagnostics, shadow);

  EXPECT_FALSE(replay.conflict);
  EXPECT_EQ(replay.emitted_diagnostic_count, 0u);
  EXPECT_EQ(header.diagnostic_count, 0u);
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
  EXPECT_EQ(diagnostics[0].reserved, 2u);
  EXPECT_EQ(diagnostics[1].first_instruction_offset, 0x20u);
  EXPECT_EQ(diagnostics[1].second_instruction_offset, 0x40u);
  EXPECT_EQ(diagnostics[1].reserved, 3u);
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

TEST(ConSanMoi, CallerOwnedSingleBankReplayKeepsExactWorkgroupsDistinct) {
  // Values above 16 bits ensure the legacy packed key cannot satisfy this
  // exact-tuple regression by accident.
  const auto replay_with_second_workgroup = [](std::array<uint32_t, 3> second_workgroup) {
    ConSanMoiReportHeader header = make_consan_moi_report_header(
        /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/2,
        /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/1,
        /*sampled_watchpoint_capacity=*/0);
    header.access_record_count = 2;

    std::array<ConSanMoiAccessRecord, 2> records{};
    records[0].generation = 7;
    records[0].workgroup_x = 70'000u;
    records[0].workgroup_y = 80'000u;
    records[0].workgroup_z = 90'000u;
    records[0].wave_id = 0;
    records[0].event_index = 0;
    records[0].instruction_offset = 0x10;
    records[0].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write);
    records[0].lds_byte_count = 4;
    records[0].cell_count = 1;

    records[1] = records[0];
    records[1].workgroup_x = second_workgroup[0];
    records[1].workgroup_y = second_workgroup[1];
    records[1].workgroup_z = second_workgroup[2];
    records[1].wave_id = 1;
    records[1].event_index = 1;
    records[1].instruction_offset = 0x20;
    records[1].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Read);

    std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
    std::array<uint64_t, 1> shadow{};
    return consan_moi_record_replay_access_records(header, records, diagnostics, shadow).conflict;
  };

  EXPECT_FALSE(replay_with_second_workgroup({70'001u, 80'000u, 90'000u}));
  EXPECT_FALSE(replay_with_second_workgroup({70'000u, 80'001u, 90'000u}));
  EXPECT_FALSE(replay_with_second_workgroup({70'000u, 80'000u, 90'001u}));
  EXPECT_TRUE(replay_with_second_workgroup({70'000u, 80'000u, 90'000u}));
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

  records[2] = {};
  records[2].site_token = 1;
  const ConSanMoiRecordReplayResult partial_site =
      consan_moi_record_replay_access_records(header, records, diagnostics, shadow);
  EXPECT_EQ(partial_site.processed_access_count, 3u);
  EXPECT_EQ(partial_site.unsupported_access_count, 1u);

  records[2] = {};
  records[2].flags = 2;
  const ConSanMoiRecordReplayResult partial_flags =
      consan_moi_record_replay_access_records(header, records, diagnostics, shadow);
  EXPECT_EQ(partial_flags.processed_access_count, 3u);
  EXPECT_EQ(partial_flags.unsupported_access_count, 1u);
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
