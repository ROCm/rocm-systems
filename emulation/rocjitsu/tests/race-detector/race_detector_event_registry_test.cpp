// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "race_test_instruction.h"
#include "rocjitsu/vm/plugins/race_detector/core/race_detector.h"
#include "rocjitsu/vm/plugins/race_detector/core/wave_race_state.h"
#include <algorithm>
#include <array>
#include <gtest/gtest.h>
#include <span>
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

constexpr std::array<uint32_t, 1> kWords{0xbf800007};
constexpr InstructionSnapshot kInstruction{0x1234, kWords};
constexpr std::array kObligations{
    MemoryCounterObligation{WaitCounterType::LGKMCNT, MemoryOrderClass::UNORDERED}};

TEST(InstructionSnapshotTest, RejectsEmptyAndOversizedEncodings) {
  const std::array<uint32_t, InstructionSnapshot::kMaxWords + 1> words{};
  EXPECT_THROW(InstructionSnapshot(0x10, std::span(words).first(0)), std::logic_error);
  EXPECT_THROW(InstructionSnapshot(0x10, words), std::logic_error);
}

TEST(InstructionSnapshotTest, CopiesEveryEncodingWidth) {
  constexpr std::array<uint32_t, InstructionSnapshot::kMaxWords> expected{1, 2, 3, 4};
  for (size_t count = 1; count <= InstructionSnapshot::kMaxWords; ++count) {
    auto source = expected;
    const InstructionSnapshot instruction(0x10, std::span(source).first(count));
    source.fill(0);
    EXPECT_EQ(instruction.pc(), 0x10u);
    EXPECT_TRUE(std::ranges::equal(instruction.encoding(), std::span(expected).first(count)));
  }
}

TEST(InstructionSnapshotTest, EqualityComparesPcAndEncoding) {
  const std::array<uint32_t, 2> words{0xbf800000, 0};
  const InstructionSnapshot one_word(0x10, std::span(words).first(1));
  EXPECT_EQ(one_word, InstructionSnapshot(0x10, std::span(words).first(1)));
  EXPECT_NE(one_word, InstructionSnapshot(0x14, std::span(words).first(1)));
  EXPECT_NE(one_word, InstructionSnapshot(0x10, std::array{0xbf800007u}));
  // A trailing zero word is part of the encoding, not unused storage.
  EXPECT_NE(one_word, InstructionSnapshot(0x10, words));
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

TEST(EventRegistry, RetainsCapturedInstruction) {
  EventRegistry events;
  const auto id = events.add(WaveId{0}, kInstruction, MemoryEventType::GLOBAL_TO_VGPR, {2}, 1, 0xf,
                             {}, kObligations, MemoryOrderClass::UNORDERED);
  EXPECT_EQ(events.pc(id), kInstruction.pc());
  EXPECT_EQ(events.instruction(id), kInstruction);
}

TEST_F(RaceEventRegistrationTest, WaveApisRetainCapturedPcAndEncoding) {
  auto &wave = detector.getWaveRaceState(0);
  const std::array<uint32_t, 1> addresses{8};
  auto at = [](int offset) { return InstructionSnapshot(kInstruction.pc() + offset, kWords); };
  wave.registerEvent(at(0), MemoryEventType::GLOBAL_TO_VGPR, {2}, 1, 0xf, kObligations,
                     MemoryOrderClass::UNORDERED);
  wave.registerScalarLoad(at(1), RegisterRef{RegClass::SGPR, 2, 1}, 1, kObligations,
                          MemoryOrderClass::UNORDERED);
  wave.registerLdsEvent(at(2), MemoryEventType::LDS_TO_VGPR, {3}, 1, 1, addresses, 4, 0xf,
                        kObligations, MemoryOrderClass::UNORDERED);
  wave.registerLdsEvent(at(3), MemoryEventType::LDS_TO_VGPR, {4}, 1, 1, addresses, addresses, 4,
                        0xf, kObligations, MemoryOrderClass::UNORDERED);
  wave.registerDualOffsetLdsEvent(at(4), MemoryEventType::LDS_TO_VGPR, {5}, 1, 1, addresses, 1, 3,
                                  kObligations, MemoryOrderClass::UNORDERED);

  ASSERT_EQ(detector.events().totalAllocated(), 5);
  for (int i = 0; i < 5; ++i)
    EXPECT_EQ(detector.events().instruction(EventId{i}), at(i));
  const auto &intervals = detector.events().ldsIntervals(EventId{4});
  EXPECT_TRUE(intervals.overlapsRange(16, 24));
  EXPECT_TRUE(intervals.overlapsRange(32, 40));
  EXPECT_FALSE(intervals.overlapsRange(24, 32));
}
} // namespace
