// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "race_test_instruction.h"
#include "rocjitsu/vm/plugins/race_detector/core/race_detector.h"
#include "rocjitsu/vm/plugins/race_detector/core/wave_race_state.h"
#include <array>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

using namespace rocjitsu::plugins::race_detector;

TEST(RaceDetector, EventRegistry_TrimmedWaveLocalEventsAreNotBarrierQueued) {
  // This test protects the boundary between two event lifetimes:
  //
  //   * Wave-local events, such as GLOBAL_TO_VGPR, only matter to the wave
  //     that issued them. After s_waitcnt says the load is complete, the race
  //     detector no longer needs to keep the event for a later workgroup
  //     barrier. EventRegistry is allowed to trim these WAVE_COMPLETE events.
  //
  //   * LDS events can matter to other waves in the same workgroup. After
  //     s_waitcnt, they must remain live until s_barrier proves every wave has
  //     reached the same synchronization point.
  //
  // The bug this test catches was that all WAVE_COMPLETE events were queued
  // for later barrier retirement, including wave-local GLOBAL_TO_VGPR events.
  // Once EventRegistry trimmed those wave-local events, the barrier queue still
  // held stale EventIds. A later barrier flush could then try to retire an
  // already-trimmed id, causing heap corruption and potentially corrupting LDS
  // race bookkeeping.
  int N = EventRegistry::kTrimAttemptInterval;
  RaceDetector detector(/*nWaves=*/
                        1, /*vgprCount=*/2, /*sgprCount=*/2, Dim3d(0), [](RaceViolation) {},
                        counterCapacitiesForArch(ROCJITSU_CODE_ARCH_CDNA4),
                        /*ldsSizeBytes=*/64 * 1024);
  auto &wave = detector.getWaveRaceState(0);

  // Allocate enough wave-local loads to trigger EventRegistry trimming. Each
  // load is immediately completed with vmcnt(0), so none should be waiting for
  // a workgroup barrier.
  for (int i = 0; i < N; ++i) {
    wave.registerEvent(testInstruction(/*pc=*/static_cast<uint64_t>(i)),
                       MemoryEventType::GLOBAL_TO_VGPR, std::vector<uint32_t>{0}, /*execMask=*/1);
    PendingWaitCount wait;
    wait.add(rocjitsu::amdgpu::WaitCounterType::VMCNT, 0);
    wave.dispatch(wait);
  }

  EXPECT_GT(detector.events().trimmedCount(), 0);
  // If wave-local events were incorrectly queued for barrier retirement, this
  // would be non-empty. Flushing it after trimming would then touch stale ids.
  EXPECT_TRUE(wave.getBarrierPendingEvents().empty());
  wave.flushBarrierPendingEvents();
}

namespace {
using namespace rocjitsu;
using amdgpu::MemoryCounterObligation;
using amdgpu::WaitCounterType;

constexpr InstructionSnapshot kInstruction{.pc = 0x1234, .words = {0xbf800007}, .word_count = 1};
constexpr std::array kObligations{
    MemoryCounterObligation{WaitCounterType::LGKMCNT, MemoryOrderClass::UNORDERED}};

TEST(InstructionSnapshotTest, EncodingRejectsEmptySnapshot) {
  EXPECT_THROW(InstructionSnapshot{}.encoding(), std::logic_error);
}

TEST(InstructionSnapshotTest, EncodingRejectsOversizedSnapshot) {
  for (const uint32_t count :
       {uint32_t{InstructionSnapshot::kMaxWords + 1}, std::numeric_limits<uint32_t>::max()}) {
    auto instruction = kInstruction;
    instruction.word_count = count;
    EXPECT_THROW(instruction.encoding(), std::logic_error);
  }
}

TEST(InstructionSnapshotTest, EncodingViewsOnlyCapturedWords) {
  auto instruction = kInstruction;
  for (uint32_t count = 1; count <= InstructionSnapshot::kMaxWords; ++count) {
    instruction.word_count = count;
    const auto encoding = instruction.encoding();
    EXPECT_EQ(encoding.data(), instruction.words.data());
    EXPECT_EQ(encoding.size(), count);
  }
}

class RaceEventRegistrationTest : public ::testing::Test {
protected:
  RaceDetector detector{1,
                        16,
                        16,
                        Dim3d(0),
                        [this](RaceViolation v) { violations.push_back(v); },
                        counterCapacitiesForArch(ROCJITSU_CODE_ARCH_CDNA4),
                        1024};
  std::vector<RaceViolation> violations;
};

TEST(EventRegistry, RetainsCapturedPcAndOwnedEncoding) {
  EventRegistry events;
  auto instruction = kInstruction;
  const auto id = events.add(WaveId{0}, instruction, MemoryEventType::GLOBAL_TO_VGPR, {2}, 1, 0xf,
                             {}, kObligations, MemoryOrderClass::UNORDERED);
  instruction.pc = 0x5678;
  instruction.words[0] = 0;
  instruction.word_count = 0;

  EXPECT_EQ(events.pc(id), kInstruction.pc);
  EXPECT_EQ(events.instruction(id).words, kInstruction.words);
  EXPECT_EQ(events.instruction(id).word_count, kInstruction.word_count);
}

TEST_F(RaceEventRegistrationTest, InvalidSizesDoNotChangeEventState) {
  EventRegistry events;
  auto &wave = detector.getWaveRaceState(0);
  IntervalSet intervals;
  intervals.append(8, 16);
  intervals.finalize();
  for (const uint32_t count :
       {0u, uint32_t{InstructionSnapshot::kMaxWords + 1}, std::numeric_limits<uint32_t>::max()}) {
    auto invalid = kInstruction;
    invalid.word_count = count;
    EXPECT_THROW(events.add(WaveId{0}, invalid, MemoryEventType::GLOBAL_TO_VGPR, {2}, 1, 0xf, {},
                            kObligations, MemoryOrderClass::UNORDERED),
                 std::logic_error);
    EXPECT_THROW(detector.allocateEventId(WaveId{0}, invalid, MemoryEventType::VGPR_TO_LDS, {2}, 1,
                                          0xf, intervals, kObligations,
                                          MemoryOrderClass::UNORDERED),
                 std::logic_error);
    EXPECT_THROW(wave.registerEvent(invalid, MemoryEventType::GLOBAL_TO_VGPR, {2}, 1, 0xf,
                                    kObligations, MemoryOrderClass::UNORDERED),
                 std::logic_error);
  }
  EXPECT_EQ(events.totalAllocated(), 0);
  EXPECT_EQ(detector.events().totalAllocated(), 0);
  EXPECT_TRUE(wave.getWaveMemoryEvents().empty());
  EXPECT_TRUE(wave.getVgprMemoryEvents(2).empty());
  EXPECT_EQ(wave.getRegEventCount(MemoryEventType::GLOBAL_TO_VGPR, 2), 0);
  EXPECT_TRUE(detector.getLdsWriteEvents().empty());
  wave.checkVgprRead(2, 0, 0xf);
  detector.validateRead(8, WaveId{0}, 0, 8);
  EXPECT_TRUE(violations.empty());
}

TEST_F(RaceEventRegistrationTest, InvalidScalarLoadDoesNotCheckPendingDestination) {
  auto &wave = detector.getWaveRaceState(0);
  const RegisterRef destination{RegClass::SGPR, 2, 1};
  wave.registerScalarLoad(kInstruction, destination, 1, kObligations, MemoryOrderClass::UNORDERED);
  ASSERT_TRUE(violations.empty());
  for (const uint32_t count : {0u, uint32_t{InstructionSnapshot::kMaxWords + 1}}) {
    auto invalid = kInstruction;
    invalid.word_count = count;
    EXPECT_THROW(
        wave.registerScalarLoad(invalid, destination, 1, kObligations, MemoryOrderClass::UNORDERED),
        std::logic_error);
  }
  EXPECT_EQ(detector.events().totalAllocated(), 1);
  EXPECT_EQ(wave.getWaveMemoryEvents().size(), 1u);
  EXPECT_TRUE(violations.empty());
  wave.registerScalarLoad(kInstruction, destination, 1, kObligations, MemoryOrderClass::UNORDERED);
  EXPECT_EQ(violations.size(), 1u);
}

TEST_F(RaceEventRegistrationTest, WaveApisRetainCapturedPcAndEncoding) {
  auto &wave = detector.getWaveRaceState(0);
  const std::array<uint32_t, 1> addresses{8};
  auto instruction = kInstruction;
  wave.registerEvent(instruction, MemoryEventType::GLOBAL_TO_VGPR, {2}, 1, 0xf, kObligations,
                     MemoryOrderClass::UNORDERED);
  ++instruction.pc;
  wave.registerScalarLoad(instruction, RegisterRef{RegClass::SGPR, 2, 1}, 1, kObligations,
                          MemoryOrderClass::UNORDERED);
  ++instruction.pc;
  wave.registerLdsEvent(instruction, MemoryEventType::LDS_TO_VGPR, {3}, 1, 1, addresses, 4, 0xf,
                        kObligations, MemoryOrderClass::UNORDERED);
  ++instruction.pc;
  wave.registerLdsEvent(instruction, MemoryEventType::LDS_TO_VGPR, {4}, 1, 1, addresses, addresses,
                        4, 0xf, kObligations, MemoryOrderClass::UNORDERED);
  ++instruction.pc;
  wave.registerDualOffsetLdsEvent(instruction, MemoryEventType::LDS_TO_VGPR, {5}, 1, 1, addresses,
                                  1, 3, kObligations, MemoryOrderClass::UNORDERED);
  instruction = {};

  ASSERT_EQ(detector.events().totalAllocated(), 5);
  for (int i = 0; i < 5; ++i) {
    const auto &stored = detector.events().instruction(EventId{i});
    EXPECT_EQ(stored.pc, kInstruction.pc + i);
    EXPECT_EQ(stored.words, kInstruction.words);
    EXPECT_EQ(stored.word_count, kInstruction.word_count);
  }
  const auto &intervals = detector.events().ldsIntervals(EventId{4});
  EXPECT_TRUE(intervals.overlapsRange(16, 24));
  EXPECT_TRUE(intervals.overlapsRange(32, 40));
  EXPECT_FALSE(intervals.overlapsRange(24, 32));
}
} // namespace
