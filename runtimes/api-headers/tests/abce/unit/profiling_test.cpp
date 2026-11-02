/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Profiled copies: the end timestamp must be recorded before the completion
// signal moves, or a profiler that reads the timestamps on completion sees an
// unwritten end (zero-duration copies).
//
// This is the bug ROCr fixed for gfx125+ in ROCm/rocm-systems#12615: the fused
// wait/signal packet signalled completion inline, ahead of the end timestamp.
// ABCE avoids it by forcing an epilogue whenever a timestamp surface is set, so
// no fused body may carry an inline signal and the epilogue writes the end
// timestamp and then the completion.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "abce_test_harness.hpp"

namespace {

using abce_test::IsaByName;
using abce_test::MakeHostToDevice;
using abce_test::Mapper;
using abce_test::MapperOptions;
using abce_test::WalkResult;

constexpr size_t kKiB = 1024;
constexpr size_t kMiB = 1024 * kKiB;

struct alignas(8) TimestampSlots {
  uint64_t start = 0;
  uint64_t end = 0;
};

abce_copy_metadata_t ProfiledMetadata(Mapper& mapper, TimestampSlots* slots) {
  abce_copy_metadata_t metadata = mapper.NewMetadata();
  metadata.timestamps.start_value = &slots->start;
  metadata.timestamps.end_value = &slots->end;
  return metadata;
}

std::vector<abce_copy_op_t> LinearBatch(uint32_t num_copies, size_t size_each) {
  std::vector<abce_copy_op_t> ops;
  for (uint32_t copy_idx = 0; copy_idx < num_copies; ++copy_idx)
    ops.push_back(MakeHostToDevice(0x20000000ull + copy_idx * 0x4000000ull,
                                   0x40000000ull + copy_idx * 0x4000000ull, size_each));
  return ops;
}

bool IsFusedCopy(const std::string& name) { return name.find("WAITSIGNAL") != std::string::npos; }

bool IsCopyPayload(const std::string& name) { return name.rfind("COPY_", 0) == 0; }

bool IsCompletionWrite(const std::string& name) {
  return name == "ATOMIC" || name == "FENCE" || name == "FENCE_64";
}

bool IsPoll(const std::string& name) { return name == "POLL_MEM_64" || name == "POLL_REGMEM"; }

// One frame's packets plus the raw words, so header bits can be inspected.
struct FrameView {
  std::vector<uint8_t> bytes;
  WalkResult walk;

  FrameView(const Mapper& mapper, const abce_plan_t& plan, uint32_t frame_idx)
      : bytes(Mapper::FrameBytes(plan, frame_idx)), walk(mapper.WalkFrame(plan, frame_idx)) {}

  uint32_t Header(size_t packet_idx) const {
    uint32_t header = 0;
    std::memcpy(&header, bytes.data() + walk.packets[packet_idx].byte_offset, sizeof(header));
    return header;
  }

  bool CarriesInlineSignal(size_t packet_idx) const { return (Header(packet_idx) >> 31) & 1u; }

  std::vector<size_t> IndicesOf(const std::string& name) const {
    std::vector<size_t> indices;
    for (size_t packet_idx = 0; packet_idx < walk.packets.size(); ++packet_idx)
      if (walk.packets[packet_idx].name == name) indices.push_back(packet_idx);
    return indices;
  }

  // Index of the last copy payload packet, or npos.
  size_t LastCopyPayload() const {
    size_t last = std::string::npos;
    for (size_t packet_idx = 0; packet_idx < walk.packets.size(); ++packet_idx)
      if (IsCopyPayload(walk.packets[packet_idx].name)) last = packet_idx;
    return last;
  }

  size_t FirstCopyPayload() const {
    for (size_t packet_idx = 0; packet_idx < walk.packets.size(); ++packet_idx)
      if (IsCopyPayload(walk.packets[packet_idx].name)) return packet_idx;
    return std::string::npos;
  }
};

class Profiling : public ::testing::TestWithParam<std::string> {};

TEST_P(Profiling, ASingleCopyRecordsItsEndTimestampBeforeSignallingCompletion) {
  Mapper mapper(IsaByName(GetParam()));
  TimestampSlots slots;
  abce_plan_t plan;
  ASSERT_EQ(mapper.Map({MakeHostToDevice(0x1000, 0x8000, 4 * kKiB)},
                       ProfiledMetadata(mapper, &slots), &plan),
            ABCE_STATUS_OK);
  ASSERT_EQ(plan.num_frames, 1u);
  EXPECT_TRUE(plan.epilogue_required) << "no epilogue means no place for the end timestamp";

  const FrameView frame(mapper, plan, 0);
  ASSERT_TRUE(frame.walk.clean);
  const std::vector<size_t> timestamps = frame.IndicesOf("TIMESTAMP_GET_GLOBAL");
  ASSERT_EQ(timestamps.size(), 2u) << "one start and one end timestamp";
  ASSERT_NE(frame.FirstCopyPayload(), std::string::npos);

  EXPECT_LT(timestamps[0], frame.FirstCopyPayload()) << "start timestamp must precede the copy";
  EXPECT_GT(timestamps[1], frame.LastCopyPayload()) << "end timestamp must follow the copy";

  const size_t last_packet = frame.walk.packets.size() - 1;
  EXPECT_TRUE(IsCompletionWrite(frame.walk.packets[last_packet].name))
      << "the frame must end by moving the completion signal, got "
      << frame.walk.packets[last_packet].name;
  EXPECT_GT(last_packet, timestamps[1]) << "completion must move after the end timestamp";

  for (size_t packet_idx = 0; packet_idx < frame.walk.packets.size(); ++packet_idx) {
    if (!IsFusedCopy(frame.walk.packets[packet_idx].name)) continue;
    EXPECT_FALSE(frame.CarriesInlineSignal(packet_idx))
        << "a fused body signalling inline would complete the copy before the end timestamp";
  }
}

TEST_P(Profiling, AFannedOutCopyWaitsForEveryBodyBeforeTheEndTimestamp) {
  MapperOptions options;
  options.num_engines = 4;
  Mapper mapper(IsaByName(GetParam()), options);
  abce_plan_t unprofiled_plan;
  ASSERT_EQ(mapper.Map(LinearBatch(4, 1 * kMiB), mapper.NewMetadata(), &unprofiled_plan),
            ABCE_STATUS_OK);
  if (unprofiled_plan.num_frames == 1)
    GTEST_SKIP() << "this part's engine policy does not fan the batch out";

  TimestampSlots slots;
  abce_plan_t plan;
  ASSERT_EQ(mapper.Map(LinearBatch(4, 1 * kMiB), ProfiledMetadata(mapper, &slots), &plan),
            ABCE_STATUS_OK);
  EXPECT_EQ(plan.num_frames, unprofiled_plan.num_frames) << "profiling must not change placement";
  EXPECT_TRUE(plan.epilogue_required);
  EXPECT_TRUE(plan.start_gate_required)
      << "without the gate, followers can start before the start timestamp is recorded";
  // The bodies count down ABCE's own word, never the caller's completion signal.
  EXPECT_EQ(plan.coordination_signal, &mapper.signal().coordination);

  uint32_t num_coordinators = 0;
  for (uint32_t frame_idx = 0; frame_idx < plan.num_frames; ++frame_idx) {
    SCOPED_TRACE("frame " + std::to_string(frame_idx));
    const FrameView frame(mapper, plan, frame_idx);
    ASSERT_TRUE(frame.walk.clean);

    const std::vector<size_t> timestamps = frame.IndicesOf("TIMESTAMP_GET_GLOBAL");
    if (timestamps.empty()) continue;
    ++num_coordinators;
    ASSERT_EQ(timestamps.size(), 2u);
    const size_t end_timestamp = timestamps[1];
    EXPECT_GT(end_timestamp, frame.LastCopyPayload());

    // The first atomic in the coordinator's prologue releases the start gate.
    const std::vector<size_t> atomics = frame.IndicesOf("ATOMIC");
    ASSERT_FALSE(atomics.empty());
    EXPECT_LT(timestamps[0], atomics[0]) << "start timestamp must precede the gate release";

    // The coordinator must observe the fan-in word reach zero between its own
    // work and the end timestamp, which is what makes the end timestamp mean
    // "every engine finished".
    bool polled_fan_in = false;
    for (size_t packet_idx = frame.LastCopyPayload(); packet_idx < end_timestamp; ++packet_idx)
      polled_fan_in = polled_fan_in || IsPoll(frame.walk.packets[packet_idx].name);
    EXPECT_TRUE(polled_fan_in) << "end timestamp recorded before the fan-in poll";

    const size_t last_packet = frame.walk.packets.size() - 1;
    EXPECT_TRUE(IsCompletionWrite(frame.walk.packets[last_packet].name));
    EXPECT_GT(last_packet, end_timestamp);
  }
  EXPECT_EQ(num_coordinators, 1u) << "exactly one frame owns the timestamps and the completion";
}

INSTANTIATE_TEST_SUITE_P(AllIsas, Profiling,
                         ::testing::Values("gfx90a", "gfx942", "gfx950", "gfx1030", "gfx1100",
                                           "gfx1201", "gfx1250"),
                         [](const ::testing::TestParamInfo<std::string>& info) { return info.param; });

// The unprofiled gfx125+ fast path signals inline; profiling must drop that
// signal block. With no dependency either, what remains is a plain 7-DWORD copy
// packet, which the decoder reports as COPY_LINEAR.
TEST(ProfilingGfx125Plus, ProfilingDropsTheSignalBlockFromTheCopyPacket) {
  Mapper mapper(IsaByName("gfx1250"));
  abce_plan_t unprofiled_plan;
  ASSERT_EQ(mapper.Map({MakeHostToDevice(0x1000, 0x8000, 4 * kKiB)}, mapper.NewMetadata(),
                       &unprofiled_plan),
            ABCE_STATUS_OK);
  const FrameView unprofiled(mapper, unprofiled_plan, 0);

  TimestampSlots slots;
  abce_plan_t profiled_plan;
  ASSERT_EQ(mapper.Map({MakeHostToDevice(0x1000, 0x8000, 4 * kKiB)},
                       ProfiledMetadata(mapper, &slots), &profiled_plan),
            ABCE_STATUS_OK);
  const FrameView profiled(mapper, profiled_plan, 0);

  const size_t unprofiled_idx = unprofiled.FirstCopyPayload();
  const size_t profiled_idx = profiled.FirstCopyPayload();
  ASSERT_NE(unprofiled_idx, std::string::npos);
  ASSERT_NE(profiled_idx, std::string::npos);

  EXPECT_TRUE(unprofiled.CarriesInlineSignal(unprofiled_idx));
  EXPECT_EQ(unprofiled.walk.packets[unprofiled_idx].dwords, 12u) << "header + 6 body + 5 signal";
  EXPECT_FALSE(profiled.CarriesInlineSignal(profiled_idx));
  EXPECT_EQ(profiled.walk.packets[profiled_idx].dwords, 7u) << "header + 6 body, signal block gone";
}

// With the HDP flush off, an unprofiled gfx125+ fan-out needs no gate and no
// epilogue (every body signals the completion directly). A timestamp is then the
// only ordering work, so it alone must bring the gate and the epilogue back.
TEST(ProfilingGfx125Plus, ATimestampAloneReintroducesTheGateAndEpilogueOnTheGateFreeFanOut) {
  MapperOptions options;
  options.num_engines = 4;
  abce_platform_caps_t caps = abce_detect_default_platform_caps(IsaByName("gfx1250").isa);
  caps.emit_hdp_flush = false;
  options.caps = caps;
  Mapper mapper(IsaByName("gfx1250"), options);

  abce_plan_t unprofiled_plan;
  ASSERT_EQ(mapper.Map(LinearBatch(4, 1 * kMiB), mapper.NewMetadata(), &unprofiled_plan),
            ABCE_STATUS_OK);
  ASSERT_EQ(unprofiled_plan.num_frames, 4u);
  ASSERT_FALSE(unprofiled_plan.start_gate_required) << "precondition: the gate-free fast path";
  ASSERT_FALSE(unprofiled_plan.epilogue_required);

  TimestampSlots slots;
  abce_plan_t profiled_plan;
  ASSERT_EQ(mapper.Map(LinearBatch(4, 1 * kMiB), ProfiledMetadata(mapper, &slots), &profiled_plan),
            ABCE_STATUS_OK);
  EXPECT_EQ(profiled_plan.num_frames, 4u);
  EXPECT_TRUE(profiled_plan.start_gate_required);
  EXPECT_TRUE(profiled_plan.epilogue_required);
  EXPECT_EQ(profiled_plan.coordination_signal, &mapper.signal().coordination)
      << "bodies must not count down the caller's completion signal";
}

// The review of #12615 asked whether the standalone completion atomic is safe
// on a gfx125+ link without 64-bit atomics. ABCE's completion write is chosen by
// the platform: atomic when the device can do them, a 64-bit fence otherwise.
// Either way it is the last packet, after the end timestamp.
TEST(ProfilingGfx125Plus, CompletionFallsBackToAFenceWhenTheLinkHasNoDeviceAtomics) {
  for (const bool device_atomics : {true, false}) {
    SCOPED_TRACE(device_atomics ? "device atomics" : "no device atomics");
    MapperOptions options;
    abce_platform_caps_t caps = abce_detect_default_platform_caps(IsaByName("gfx1250").isa);
    caps.device_atomic_support = device_atomics;
    options.caps = caps;
    Mapper mapper(IsaByName("gfx1250"), options);
    TimestampSlots slots;
    abce_plan_t plan;
    ASSERT_EQ(mapper.Map({MakeHostToDevice(0x1000, 0x8000, 4 * kKiB)},
                         ProfiledMetadata(mapper, &slots), &plan),
              ABCE_STATUS_OK);
    const FrameView frame(mapper, plan, 0);
    ASSERT_TRUE(frame.walk.clean);
    const std::vector<size_t> timestamps = frame.IndicesOf("TIMESTAMP_GET_GLOBAL");
    ASSERT_EQ(timestamps.size(), 2u);
    const size_t last_packet = frame.walk.packets.size() - 1;
    EXPECT_GT(last_packet, timestamps[1]);
    EXPECT_EQ(frame.walk.packets[last_packet].name, device_atomics ? "ATOMIC" : "FENCE_64");
    EXPECT_TRUE(frame.IndicesOf(device_atomics ? "FENCE_64" : "ATOMIC").empty())
        << "only the platform's completion write may appear";
  }
}

}  // namespace
