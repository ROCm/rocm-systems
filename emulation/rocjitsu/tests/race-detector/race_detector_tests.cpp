// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Unit tests for the race detection core (RaceDetector + WaveRaceState).
//
// These tests drive the detection API directly via RaceTestBuilder — no
// assembly parser, no instruction emulator, no GPU.
//
#include "race_test_builder.h"
#include "race_test_instruction.h"
#include "rocjitsu/isa/arch/amdgpu/shared/memory_issue.h"
#include "rocjitsu/vm/plugins/race_detector/core/interval_set.h"
#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include <initializer_list>
#include <limits>
#include <span>
#include <utility>
#include <vector>

using namespace rocjitsu::plugins::race_detector;
namespace amdgpu = rocjitsu::amdgpu;

namespace {

constexpr std::array kMixedOrderGenericFlatLoadObligations{
    amdgpu::MemoryCounterObligation{amdgpu::WaitCounterType::LOADCNT, MemoryOrderClass::VMEM},
    amdgpu::MemoryCounterObligation{amdgpu::WaitCounterType::DSCNT, MemoryOrderClass::LDS}};

} // namespace

TEST(RaceDetector, MixedFlatLdsReadRequiresBothCounterWaits) {
  for (bool vmcnt_first : {false, true}) {
    SCOPED_TRACE(vmcnt_first ? "vmcnt first" : "lgkmcnt first");
    RaceTestBuilder b(/*numWaves=*/1);
    b.mixedFlatLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);

    b.waitcnt(/*wave=*/0, /*vmcnt=*/vmcnt_first ? 0 : -1, /*lgkmcnt=*/vmcnt_first ? -1 : 0);
    b.checkLdsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
    EXPECT_TRUE(b.hasLdsRace(0));

    b.clearViolations();
    b.waitcnt(/*wave=*/0, /*vmcnt=*/vmcnt_first ? -1 : 0, /*lgkmcnt=*/vmcnt_first ? 0 : -1);
    b.checkLdsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
    EXPECT_FALSE(b.hasRace());
  }
}

TEST(RaceDetector, MixedFlatLdsWriteRequiresBothCounterWaits) {
  for (bool vmcnt_first : {false, true}) {
    SCOPED_TRACE(vmcnt_first ? "vmcnt first" : "lgkmcnt first");
    RaceTestBuilder b(/*numWaves=*/1);
    b.mixedFlatLdsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);

    b.waitcnt(/*wave=*/0, /*vmcnt=*/vmcnt_first ? 0 : -1, /*lgkmcnt=*/vmcnt_first ? -1 : 0);
    b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
    EXPECT_TRUE(b.hasLdsRace(0));

    b.clearViolations();
    b.waitcnt(/*wave=*/0, /*vmcnt=*/vmcnt_first ? -1 : 0, /*lgkmcnt=*/vmcnt_first ? 0 : -1);
    b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
    EXPECT_FALSE(b.hasRace());
  }
}

TEST(RaceDetector, MixedFlatKeepsOneCounterEntryAndOnlyLdsIntervals) {
  for (uint64_t lds_mask : {0u, 1u, 2u, 5u}) {
    SCOPED_TRACE(lds_mask);
    std::vector<RaceViolation> violations;
    RaceDetector detector(
        1, Dim3d(0), [&](RaceViolation v) { violations.push_back(v); },
        counterCapacitiesForArch(ROCJITSU_CODE_ARCH_RDNA4), /*ldsSizeBytes=*/64 * 1024);
    auto &wave = detector.getWaveRaceState(0);
    const std::array<uint32_t, 32> addresses{64, 128, 256};
    for (uint32_t pc : {1u, 2u})
      wave.registerLdsEvent(testInstruction(pc), LdsAccess::Read, /*laneMask=*/3 & lds_mask,
                            /*waveSize=*/32, addresses, /*bytesPerLane=*/4,
                            kMixedOrderGenericFlatLoadObligations, MemoryOrderClass::UNORDERED);
    const auto &events = detector.events();
    ASSERT_EQ(events.totalAllocated(), 2);
    for (const EventId event : {EventId{0}, EventId{1}}) {
      EXPECT_EQ(events.ldsIntervals(event).contains(64), (lds_mask & 1) != 0);
      EXPECT_EQ(events.ldsIntervals(event).contains(128), (lds_mask & 2) != 0);
      EXPECT_FALSE(events.ldsIntervals(event).contains(256)); // Outside EXEC.
      EXPECT_EQ(events.pendingCounterIncrement(event, amdgpu::WaitCounterType::LOADCNT), 1);
      EXPECT_EQ(events.pendingCounterIncrement(event, amdgpu::WaitCounterType::DSCNT), 1);
    }
    PendingWaitCount wait;
    wait.add(amdgpu::WaitCounterType::LOADCNT, 1);
    wait.add(amdgpu::WaitCounterType::DSCNT, 1);
    wave.dispatch(wait);
    // A single-wave workgroup retires a completed event with LDS bytes at once;
    // one without LDS bytes stays wave-complete until the registry trims it.
    EXPECT_EQ(events.status(EventId{0}),
              lds_mask != 0 ? EventStatus::RETIRED : EventStatus::WAVE_COMPLETE);
    EXPECT_EQ(events.status(EventId{1}), EventStatus::ACTIVE);
    wait.updates.clear();
    wait.add(amdgpu::WaitCounterType::LOADCNT, 0);
    wait.add(amdgpu::WaitCounterType::DSCNT, 0);
    wave.dispatch(wait);
    EXPECT_NE(events.status(EventId{1}), EventStatus::ACTIVE);
    EXPECT_TRUE(violations.empty());
  }
}

TEST(RaceDetector, LdsValidationRejectsNegativeAndOverflowingRanges) {
  std::vector<RaceViolation> violations;
  RaceDetector detector(
      2, Dim3d(0), [&](RaceViolation v) { violations.push_back(v); },
      counterCapacitiesForArch(ROCJITSU_CODE_ARCH_CDNA5), /*ldsSizeBytes=*/64 * 1024);
  for (const auto type : {LdsAccess::Read, LdsAccess::Write}) {
    IntervalSet intervals;
    intervals.append(64, 68);
    detector.allocateEventId(
        WaveId{0}, testInstruction(0), type, std::move(intervals),
        CounterObligations{amdgpu::WaitCounterType::DSCNT, MemoryOrderClass::LDS},
        MemoryOrderClass::LDS);
  }
  struct Range {
    int address;
    int bytes;
  };
  for (const auto range :
       {Range{std::numeric_limits<int>::min() + 64, 4}, Range{-1, 4},
        Range{std::numeric_limits<int>::max() - 1, 4}, Range{64, 0}, Range{64, -1}}) {
    SCOPED_TRACE(testing::Message() << "address=" << range.address << " bytes=" << range.bytes);
    detector.validateRead(range.address, WaveId{1}, 0, range.bytes);
    detector.validateWrite(range.address, WaveId{1}, 0, range.bytes);
    EXPECT_TRUE(violations.empty());
  }
  detector.validateRead(64, WaveId{1}, 0, 4);
  detector.validateWrite(64, WaveId{1}, 0, 4);
  EXPECT_EQ(violations.size(), 2u);
}

TEST(RaceDetector, InvalidLdsAddressesKeepCounterParticipation) {
  for (bool dual_access : {false, true}) {
    SCOPED_TRACE(dual_access);
    std::vector<RaceViolation> violations;
    RaceDetector detector(
        1, Dim3d(0), [&](RaceViolation v) { violations.push_back(v); },
        counterCapacitiesForArch(ROCJITSU_CODE_ARCH_CDNA5), /*ldsSizeBytes=*/64 * 1024);
    auto &wave = detector.getWaveRaceState(0);
    const std::array<uint32_t, 4> addresses{0x80000040u, UINT32_MAX, 0x7ffffffeu, 64};
    const std::array<uint32_t, 4> second_addresses{0x80000080u, UINT32_MAX, 0x7fffffffu, 96};
    if (dual_access)
      wave.registerLdsEvent(testInstruction(0), LdsAccess::Read, 0xf, 4, addresses,
                            second_addresses, 4, kMixedOrderGenericFlatLoadObligations,
                            MemoryOrderClass::UNORDERED);
    else
      wave.registerLdsEvent(testInstruction(0), LdsAccess::Read, 0xf, 4, addresses, 4,
                            kMixedOrderGenericFlatLoadObligations, MemoryOrderClass::UNORDERED);
    const auto &events = detector.events();
    const EventId event{0};
    ASSERT_EQ(events.totalAllocated(), 1);
    EXPECT_EQ(events.ldsIntervals(event).getTotalBytes(), dual_access ? 8 : 4);
    EXPECT_TRUE(events.ldsIntervals(event).contains(64));
    EXPECT_EQ(events.ldsIntervals(event).contains(96), dual_access);
    EXPECT_EQ(events.pendingCounterIncrement(event, amdgpu::WaitCounterType::LOADCNT), 1);
    EXPECT_EQ(events.pendingCounterIncrement(event, amdgpu::WaitCounterType::DSCNT), 1);
    detector.validateWrite(64, WaveId{0}, 3, 4);
    ASSERT_EQ(violations.size(), 1u);
    EXPECT_EQ(violations.front().address, 64);
    PendingWaitCount wait;
    wait.add(amdgpu::WaitCounterType::LOADCNT, 0);
    wave.dispatch(wait);
    EXPECT_EQ(events.status(event), EventStatus::ACTIVE);
    wait.updates.clear();
    wait.add(amdgpu::WaitCounterType::DSCNT, 0);
    wave.dispatch(wait);
    // The event has LDS bytes, so the single-wave workgroup retires it at once.
    EXPECT_EQ(events.status(event), EventStatus::RETIRED);
    violations.clear();
    detector.validateWrite(64, WaveId{0}, 3, 4);
    EXPECT_TRUE(violations.empty());
  }
}

TEST(RaceDetector, PositiveOutOfBoundsLdsKeepsDependencies) {
  for (uint32_t capacity : {32u * 1024, 64u * 1024}) {
    std::vector<RaceViolation> violations;
    RaceDetector detector(
        2, Dim3d(0), [&](RaceViolation v) { violations.push_back(v); },
        counterCapacitiesForArch(ROCJITSU_CODE_ARCH_CDNA5), capacity);
    auto &wave = detector.getWaveRaceState(0);
    const std::array<uint32_t, 5> addresses{capacity - 4, capacity - 2, capacity, 0x100000,
                                            0x70000000};
    wave.registerLdsEvent(testInstruction(0), LdsAccess::Read, 0x1f, 5, addresses, 4,
                          kMixedOrderGenericFlatLoadObligations, MemoryOrderClass::UNORDERED);
    const EventId event{0};
    const auto &events = detector.events();
    EXPECT_EQ(events.ldsIntervals(event).getTotalBytes(), 4);
    EXPECT_TRUE(events.ldsIntervals(event).contains(capacity - 4));
    // An invalid access crossing the end must not race with the valid tail.
    for (uint32_t address : addresses)
      if (address != capacity - 4)
        detector.validateWrite(address, WaveId{1}, 0, 4);
    EXPECT_TRUE(violations.empty());
    detector.validateWrite(capacity - 4, WaveId{1}, 0, 4);
    EXPECT_EQ(violations.size(), 1u);
    PendingWaitCount wait;
    wait.add(amdgpu::WaitCounterType::LOADCNT, 0);
    wave.dispatch(wait);
    EXPECT_EQ(events.status(event), EventStatus::ACTIVE);
    wait.updates.clear();
    wait.add(amdgpu::WaitCounterType::DSCNT, 0);
    wave.dispatch(wait);
    EXPECT_EQ(events.status(event), EventStatus::WAVE_COMPLETE);
  }
}

TEST(RaceDetector, LdsVectorIntervalsRetainOnlyCompleteElementsAtBackingEnd) {
  constexpr uint32_t capacity = 64 * 1024;
  RaceDetector detector(
      1, Dim3d(0), [](RaceViolation) {}, counterCapacitiesForArch(ROCJITSU_CODE_ARCH_CDNA5),
      capacity);
  auto &wave = detector.getWaveRaceState(0);
  const std::array<uint32_t, 3> addresses{capacity - 8, capacity - 2, capacity};
  wave.registerLdsEvent(testInstruction(0), LdsAccess::Read, /*laneMask=*/7, 3, addresses,
                        /*bytesPerLane=*/16, kMixedOrderGenericFlatLoadObligations,
                        MemoryOrderClass::UNORDERED, /*elementBytes=*/4);
  const auto &events = detector.events();
  EXPECT_EQ(events.ldsIntervals(EventId{0}).getTotalBytes(), 8);
  EXPECT_TRUE(events.ldsIntervals(EventId{0}).contains(capacity - 1));
  EXPECT_FALSE(events.ldsIntervals(EventId{0}).contains(capacity));
  EXPECT_EQ(events.pendingCounterIncrement(EventId{0}, amdgpu::WaitCounterType::LOADCNT), 1);
  EXPECT_EQ(events.pendingCounterIncrement(EventId{0}, amdgpu::WaitCounterType::DSCNT), 1);
}

TEST(RaceDetector, LdsCounterAccountingIgnoresNegativeIntervals) {
  for (const auto type : {LdsAccess::Read, LdsAccess::Write}) {
    SCOPED_TRACE(static_cast<int>(type));
    std::vector<RaceViolation> violations;
    RaceDetector detector(
        2, Dim3d(0), [&](RaceViolation v) { violations.push_back(v); },
        counterCapacitiesForArch(ROCJITSU_CODE_ARCH_CDNA5), /*ldsSizeBytes=*/64 * 1024);
    IntervalSet intervals;
    intervals.append(std::numeric_limits<int>::min() + 64, std::numeric_limits<int>::min() + 68);
    intervals.append(64, 68);
    const auto event = detector.allocateEventId(
        WaveId{0}, testInstruction(0), type, std::move(intervals),
        CounterObligations{amdgpu::WaitCounterType::DSCNT, MemoryOrderClass::LDS},
        MemoryOrderClass::LDS);
    detector.validateRead(64, WaveId{1}, 0, 4);
    detector.validateWrite(64, WaveId{1}, 0, 4);
    EXPECT_EQ(violations.size(), 1u);
    detector.retireEvent(event);
    violations.clear();
    detector.validateRead(64, WaveId{1}, 0, 4);
    detector.validateWrite(64, WaveId{1}, 0, 4);
    EXPECT_TRUE(violations.empty());
  }
}

TEST(RaceDetector, FlatLdsLoadDoesNotUseSameWaveLdsOrdering) {
  RaceTestBuilder b(/*numWaves=*/1);
  b.mixedFlatLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);

  b.checkLdsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);

  EXPECT_TRUE(b.hasLdsRace(0));
}

TEST(RaceDetector, LdsReadDoesNotOrderBeforeSameWaveFlatLdsStore) {
  RaceTestBuilder b(/*numWaves=*/1);
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);

  b.mixedFlatLdsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);

  EXPECT_TRUE(b.hasLdsRace(0));
}

TEST(RaceDetector, FlatLdsStoreDoesNotUseSameWaveLdsOrdering) {
  RaceTestBuilder b(/*numWaves=*/1);
  b.mixedFlatLdsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);

  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);

  EXPECT_TRUE(b.hasLdsRace(0));
}

TEST(RaceDetector, LdsStoreDoesNotOrderBeforeSameWaveFlatLdsLoad) {
  RaceTestBuilder b(/*numWaves=*/1);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);

  b.mixedFlatLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);

  EXPECT_TRUE(b.hasLdsRace(0));
}

TEST(RaceDetector, CompletedFlatLdsStoreIsSafeForSameWaveLdsRead) {
  RaceTestBuilder b(/*numWaves=*/1);
  b.mixedFlatLdsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/0, /*lgkmcnt=*/0);

  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);

  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, ScalarAndLds_PartialWaitRetiresOnlyProvableLdsPrefix) {
  for (bool scalarFirst : {false, true})
    for (int threshold : {1, 2}) {
      SCOPED_TRACE(testing::Message() << scalarFirst << " threshold=" << threshold);
      RaceTestBuilder b(2);
      if (scalarFirst)
        b.counterIssue(0, amdgpu::WaitCounterType::LGKMCNT, MemoryOrderClass::UNORDERED);
      b.ldsRead(0, 0, 0, 4);
      if (!scalarFirst)
        b.counterIssue(0, amdgpu::WaitCounterType::LGKMCNT, MemoryOrderClass::UNORDERED);
      b.ldsRead(0, 0, 4, 4);

      b.waitcnt(0, -1, threshold);
      const EventId scalar{scalarFirst ? 0 : 1};
      const EventId firstLds{scalarFirst ? 1 : 0};
      EXPECT_EQ(b.events().status(scalar), EventStatus::ACTIVE);
      EXPECT_EQ(b.events().status(firstLds),
                threshold == 1 ? EventStatus::WAVE_COMPLETE : EventStatus::ACTIVE);
      EXPECT_EQ(b.events().status(EventId{2}), EventStatus::ACTIVE);

      b.barrier();
      b.checkLdsWrite(1, 0, 0, 4);
      EXPECT_EQ(b.hasLdsRace(0), threshold == 2);
      b.checkLdsWrite(1, 0, 4, 4);
      EXPECT_TRUE(b.hasLdsRace(4));
    }
}

TEST(RaceDetector, ScalarAndLds_PartialCombinedWaitOrdersNativeDsOperations) {
  RaceTestBuilder b(/*numWaves=*/2);
  b.counterIssue(/*wave=*/0, amdgpu::WaitCounterType::LGKMCNT, MemoryOrderClass::UNORDERED);
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/4, /*bytes=*/4);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/1);

  // Native LDS reads and writes share one FIFO completion class. Even with an
  // unordered scalar-memory event on the counter, at most one outstanding
  // event means that the older DS read has completed.
  EXPECT_EQ(b.events().status(EventId{1}), EventStatus::WAVE_COMPLETE);
  b.clearViolations();
  b.barrier();
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/4, /*bytes=*/4);
  EXPECT_TRUE(b.hasLdsRace(4));
}

TEST(RaceDetector, ScalarAndLds_PartialCombinedWaitMakesOldestDsWriteBarrierEligible) {
  RaceTestBuilder b(/*numWaves=*/2);
  b.counterIssue(/*wave=*/0, amdgpu::WaitCounterType::LGKMCNT, MemoryOrderClass::UNORDERED);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/4, /*bytes=*/4);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/1);
  b.barrier();

  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  EXPECT_FALSE(b.hasLdsRace(0));
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/4, /*bytes=*/4);
  EXPECT_TRUE(b.hasLdsRace(4));
  b.clearViolations();
  EXPECT_EQ(b.events().status(EventId{0}), EventStatus::ACTIVE);
}

TEST(RaceDetector, LdsCrossWave_ZeroLdsSize) {
  // LDS size hint is 0 (e.g. compiler metadata says no LDS) but LDS events
  // arrive anyway (e.g. via inline asm buffer_load_dword lds).
  // Byte counters must grow dynamically to accommodate.
  RaceTestBuilder b(/*numWaves=*/2);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/100, /*bytes=*/4);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0);
  // Missing barrier → cross-wave race.
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/100, /*bytes=*/4);
  EXPECT_TRUE(b.hasLdsRace(100));
}

TEST(RaceDetector, LdsCrossWave_MissingBarrier) {
  // Wave 0 writes 1 byte to LDS[1] (covering [1,2)), wave 1 reads without
  // barrier. Byte-level precision: [0,1) should NOT race, [1,2) should.
  RaceTestBuilder b(/*numWaves=*/2);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/1, /*bytes=*/1);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0);
  // Missing barrier!
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/0,
                 /*bytes=*/1); // [0,1) → safe
  EXPECT_FALSE(b.hasRace());
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/1,
                 /*bytes=*/1); // [1,2) → RACE
  EXPECT_TRUE(b.hasLdsRace(1));
}

TEST(RaceDetector, LdsCrossWave_WithBarrier) {
  // Wave 0 writes, waitcnt, barrier, then wave 1 reads → safe.
  RaceTestBuilder b(/*numWaves=*/2);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0);
  b.barrier();
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, LdsCrossWave_WarMissingBarrier) {
  // WAR: wave 0 reads LDS[0], waitcnt, then wave 1 writes to LDS[0]
  // without barrier → RACE on the write.
  RaceTestBuilder b(/*numWaves=*/2);
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0);
  // Missing barrier!
  b.checkLdsWrite(/*wave=*/1, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  EXPECT_TRUE(b.hasLdsRace(0));
}

TEST(RaceDetector, LdsCrossWave_WarWithBarrier) {
  // WAR: wave 0 reads LDS[0], waitcnt, barrier, then wave 1 writes → safe.
  RaceTestBuilder b(/*numWaves=*/2);
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0);
  b.barrier();
  b.checkLdsWrite(/*wave=*/1, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, LdsCrossWave_WarNoOverlap) {
  // WAR: wave 0 reads LDS[0..4), wave 1 writes LDS[4..8) → safe (no overlap).
  RaceTestBuilder b(/*numWaves=*/2);
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0);
  b.checkLdsWrite(/*wave=*/1, /*lane=*/0, /*addr=*/4, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, LdsSameWave_WriteWriteOk) {
  // Two writes to same address, same wave → not a race.
  // Same-wave LDS writes are ordered (same LDS unit, program order).
  // Cross-wave WAW without barrier would be a race, but same-wave is safe.
  RaceTestBuilder b(/*numWaves=*/1);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, LdsSameWave_ReadReadOk) {
  // Two reads from same address → not a race.
  RaceTestBuilder b(/*numWaves=*/1);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0);
  b.barrier();
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, LdsSameWave_WriteReadSameLaneOk) {
  RaceTestBuilder b(/*numWaves=*/1);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, LdsSameWave_WriteReadCrossLaneOk) {
  // DS instructions are wave-wide, so a later lane 1 read cannot overtake
  // lane 0's write from the earlier instruction.
  RaceTestBuilder b(/*numWaves=*/1);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.checkLdsRead(/*wave=*/0, /*lane=*/1, /*addr=*/0, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, LdsSameWave_ReadWriteSameLaneOk) {
  RaceTestBuilder b(/*numWaves=*/1);
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.checkLdsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, LdsSameWave_ReadWriteCrossLaneOk) {
  // A later lane 1 write cannot overtake lane 0's read from the earlier
  // wave-wide DS instruction.
  RaceTestBuilder b(/*numWaves=*/1);
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.checkLdsWrite(/*wave=*/0, /*lane=*/1, /*addr=*/0, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, LdsSameWave_LongHistoryRemainsVisibleToOtherWaves) {
  RaceTestBuilder b(/*numWaves=*/2);
  constexpr int iterations = 600;
  for (int i = 0; i < iterations; ++i) {
    for (int wave = 0; wave < 2; ++wave) {
      // Each wave reuses its own region, crossing a 16-byte count boundary.
      // Waits complete each wave's accesses; a barrier is still needed
      // before another wave can reuse either region.
      const int addr = wave * 64 + 12;
      b.ldsWrite(wave, /*lane=*/0, addr, /*bytes=*/8);
      b.ldsRead(wave, /*lane=*/0, addr, /*bytes=*/8);
      b.waitcnt(wave, /*vmcnt=*/-1, /*lgkmcnt=*/0);
    }
  }
  ASSERT_FALSE(b.hasRace());
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/76, /*bytes=*/4);
  b.checkLdsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/76, /*bytes=*/4);
  EXPECT_EQ(b.raceCount(), 2 * iterations);

  b.clearViolations();
  b.barrier();
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/76, /*bytes=*/4);
  b.checkLdsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/76, /*bytes=*/4);
  ASSERT_FALSE(b.hasRace());

  // After retirement, the old owner's counts must not hide new accesses
  // from the other wave at the same addresses.
  b.ldsWrite(/*wave=*/1, /*lane=*/0, /*addr=*/12, /*bytes=*/8);
  b.ldsRead(/*wave=*/1, /*lane=*/0, /*addr=*/12, /*bytes=*/8);
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/12, /*bytes=*/4);
  b.checkLdsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/12, /*bytes=*/4);
  EXPECT_EQ(b.raceCount(), 2);
}

TEST(RaceDetector, LdsSameWave_SharedChunksKeepOtherWaveHazards) {
  for (bool writes : {false, true}) {
    SCOPED_TRACE(writes ? "RAW" : "WAR");
    RaceTestBuilder b(/*numWaves=*/2);
    auto issue = [&](int wave, int addr, int bytes) {
      if (writes)
        b.ldsWrite(wave, /*lane=*/0, addr, bytes);
      else
        b.ldsRead(wave, /*lane=*/0, addr, bytes);
    };
    auto check = [&](int addr, int bytes) {
      if (writes)
        b.checkLdsRead(/*wave=*/0, /*lane=*/0, addr, bytes);
      else
        b.checkLdsWrite(/*wave=*/0, /*lane=*/0, addr, bytes);
    };
    issue(/*wave=*/0, /*addr=*/0, /*bytes=*/4);
    issue(/*wave=*/0, /*addr=*/12, /*bytes=*/8);
    issue(/*wave=*/1, /*addr=*/4, /*bytes=*/2);
    issue(/*wave=*/1, /*addr=*/16, /*bytes=*/2);
    ASSERT_FALSE(b.hasRace());
    check(/*addr=*/6, /*bytes=*/6); // Same chunk, no overlapping bytes.
    ASSERT_FALSE(b.hasRace());
    check(/*addr=*/4, /*bytes=*/4);
    EXPECT_EQ(b.raceCount(), 1);
    check(/*addr=*/12, /*bytes=*/6); // Spans two count chunks.
    EXPECT_EQ(b.raceCount(), 2);
  }
}

TEST(RaceDetector, LdsSameWave_SpanningAccessChecksLaterChunks) {
  for (bool writes : {false, true}) {
    SCOPED_TRACE(writes ? "RAW" : "WAR");
    RaceTestBuilder b(/*numWaves=*/2);
    if (writes) {
      b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/8, /*bytes=*/4);
      b.ldsWrite(/*wave=*/1, /*lane=*/0, /*addr=*/16, /*bytes=*/4);
    } else {
      b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/8, /*bytes=*/4);
      b.ldsRead(/*wave=*/1, /*lane=*/0, /*addr=*/16, /*bytes=*/4);
    }
    ASSERT_FALSE(b.hasRace());
    // The first chunk belongs only to wave 0; the query must reach the second
    // chunk to find wave 1's event.
    if (writes)
      b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/12, /*bytes=*/8);
    else
      b.checkLdsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/12, /*bytes=*/8);
    EXPECT_EQ(b.raceCount(), 1);
  }
}

TEST(RaceDetector, LdsMayConflictChecksOnlyTheAccessedChunks) {
  for (bool writes : {false, true}) {
    SCOPED_TRACE(writes ? "RAW" : "WAR");
    RaceTestBuilder builder(/*numWaves=*/2);
    // Wave 1's pending access lies in the gap between wave 0's two ranges.
    if (writes)
      builder.ldsWrite(/*wave=*/1, /*lane=*/0, /*addr=*/1024, /*bytes=*/4);
    else
      builder.ldsRead(/*wave=*/1, /*lane=*/0, /*addr=*/1024, /*bytes=*/4);
    auto intervals = [](std::initializer_list<Interval> ranges) {
      IntervalSet set;
      for (const Interval &range : ranges)
        set.append(range.start, range.end);
      set.finalize();
      return set;
    };
    const IntervalSet around = intervals({{0, 256}, {4096, 4352}});
    const IntervalSet over = intervals({{0, 256}, {1020, 1028}});
    const RaceDetector &detector = builder.detector();
    const bool checkWrite = !writes;
    const MemoryOrderClass order = MemoryOrderClass::LDS;
    EXPECT_FALSE(detector.mayConflict(around, WaveId{0}, checkWrite, order));
    EXPECT_TRUE(detector.mayConflict(over, WaveId{0}, checkWrite, order));
    // Same-wave LDS accesses are ordered, and a read never conflicts with a
    // read; WAW is not checked.
    EXPECT_FALSE(detector.mayConflict(over, WaveId{1}, checkWrite, order));
    EXPECT_FALSE(detector.mayConflict(over, WaveId{0}, !checkWrite, order));
    EXPECT_TRUE(detector.mayConflict(over, WaveId{1}, checkWrite, MemoryOrderClass::UNORDERED));
    EXPECT_FALSE(detector.mayConflict(IntervalSet{}, WaveId{0}, checkWrite, order));
  }
}

TEST(RaceDetector, LdsSameWave_MixedOrderingSurvivesPartialRetirement) {
  for (bool writes : {false, true}) {
    SCOPED_TRACE(writes ? "RAW" : "WAR");
    RaceTestBuilder b(/*numWaves=*/2);
    auto ordinary = [&](int wave) {
      if (writes)
        b.ldsWrite(wave, /*lane=*/0, /*addr=*/12, /*bytes=*/8);
      else
        b.ldsRead(wave, /*lane=*/0, /*addr=*/12, /*bytes=*/8);
    };
    auto check = [&](int wave) {
      if (writes)
        b.checkLdsRead(wave, /*lane=*/0, /*addr=*/14, /*bytes=*/4);
      else
        b.checkLdsWrite(wave, /*lane=*/0, /*addr=*/14, /*bytes=*/4);
    };

    ordinary(/*wave=*/0);
    // An unordered access from the same wave must invalidate the owner of
    // both chunks. Keep its destination distinct from the ordinary read's.
    if (writes)
      b.mixedFlatLdsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/14, /*bytes=*/4);
    else
      b.mixedFlatLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/14, /*bytes=*/4);
    ASSERT_FALSE(b.hasRace());
    check(/*wave=*/0);
    EXPECT_EQ(b.raceCount(), 1);

    // Only the ordinary event completes and retires; FLAT still needs VMCNT.
    b.clearViolations();
    b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0);
    b.barrier();
    ordinary(/*wave=*/0);
    check(/*wave=*/0);
    EXPECT_EQ(b.raceCount(), 1);

    // Completion makes the history safe for this wave, but other waves
    // must still see both events until a barrier retires them.
    b.clearViolations();
    b.waitcnt(/*wave=*/0, /*vmcnt=*/0, /*lgkmcnt=*/0);
    check(/*wave=*/0);
    ASSERT_FALSE(b.hasRace());
    check(/*wave=*/1);
    EXPECT_EQ(b.raceCount(), 2);

    b.clearViolations();
    b.barrier();
    check(/*wave=*/1);
    ASSERT_FALSE(b.hasRace());
    ordinary(/*wave=*/1);
    check(/*wave=*/0);
    EXPECT_EQ(b.raceCount(), 1);
  }
}

TEST(RaceDetector, Dtl_CrossWaveRace) {
  // Wave 0 does DTL load to LDS, waitcnt, but no barrier.
  // Wave 1 reads LDS → RACE (WAVE_COMPLETE but not RETIRED).
  RaceTestBuilder b(/*numWaves=*/2);
  b.globalToLds(/*wave=*/0, /*ldsAddrs=*/{0}, /*bytesPerLane=*/64);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/0);
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/0, /*bytes=*/4); // RACE
  EXPECT_TRUE(b.hasLdsRace(0));
}

TEST(RaceDetector, Dtl_CrossWaveSafe) {
  // Same as above but with barrier → safe.
  RaceTestBuilder b(/*numWaves=*/2);
  b.globalToLds(/*wave=*/0, /*ldsAddrs=*/{0}, /*bytesPerLane=*/64);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/0);
  b.barrier();
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, Dtl_SameWaveMissingVmcntRace) {
  // Direct-to-LDS arrives through VMEM, so unlike an ordinary DS write the
  // owning wave must wait for vmcnt before reading the LDS bytes.
  RaceTestBuilder b(/*numWaves=*/1);
  b.globalToLds(/*wave=*/0, /*ldsAddrs=*/{0}, /*bytesPerLane=*/4);
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  EXPECT_TRUE(b.hasLdsRace(0));
}

TEST(RaceDetector, Dtl_SameWaveWithVmcntSafe) {
  RaceTestBuilder b(/*numWaves=*/1);
  b.globalToLds(/*wave=*/0, /*ldsAddrs=*/{0}, /*bytesPerLane=*/4);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/0);
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, Dtl_SameWavePartialVmcntRetiresOldest) {
  RaceTestBuilder b(/*numWaves=*/1);
  b.globalToLds(/*wave=*/0, /*ldsAddrs=*/{16}, /*bytesPerLane=*/4, /*exec=*/1);
  b.globalToLds(/*wave=*/0, /*ldsAddrs=*/{64}, /*bytesPerLane=*/4, /*exec=*/1);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/1);

  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/16, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());

  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/64, /*bytes=*/4);
  EXPECT_TRUE(b.hasLdsRace(64));
}

TEST(RaceDetector, TensorDma_CrossWaveLoadRace) {
  // Wave 1 reads the destination of wave 0's tensor load with no wait or barrier.
  RaceTestBuilder b(/*numWaves=*/2);
  b.tensorLoadToLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/4, /*bytes=*/4);
  EXPECT_TRUE(b.hasLdsRace(4));
}

TEST(RaceDetector, TensorDma_CrossWaveLoadWaitWithoutBarrierRace) {
  // tensorcnt completes the load for wave 0 only; other waves need a barrier.
  RaceTestBuilder b(/*numWaves=*/2);
  b.tensorLoadToLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.waitTensorcnt(/*wave=*/0, /*tensorcnt=*/0);
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/4, /*bytes=*/4);
  EXPECT_TRUE(b.hasLdsRace(4));
}

TEST(RaceDetector, TensorDma_CrossWaveLoadSafe) {
  RaceTestBuilder b(/*numWaves=*/2);
  b.tensorLoadToLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.waitTensorcnt(/*wave=*/0, /*tensorcnt=*/0);
  b.barrier();
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/4, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, TensorDma_SameWaveMissingTensorcntRace) {
  // Tensor DMA completes independently of DS, so the issuing wave must wait
  // for tensorcnt before reading the destination.
  RaceTestBuilder b(/*numWaves=*/1);
  b.tensorLoadToLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/12, /*bytes=*/4);
  EXPECT_TRUE(b.hasLdsRace(12));
}

TEST(RaceDetector, TensorDma_SameWaveWithTensorcntSafe) {
  RaceTestBuilder b(/*numWaves=*/1);
  b.tensorLoadToLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.waitTensorcnt(/*wave=*/0, /*tensorcnt=*/0);
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/12, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, TensorDma_OnlyTransferredRangesAreTracked) {
  // A padded tile leaves the padding bytes between its rows untouched.
  RaceTestBuilder b(/*numWaves=*/2);
  b.tensorLoadToLds(/*wave=*/0, {{.address = 0, .bytes = 8}, {.address = 12, .bytes = 8}});
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/8, /*bytes=*/4);
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/20, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/12, /*bytes=*/4);
  EXPECT_TRUE(b.hasLdsRace(12));
}

// Partial waits retire by the completion class of each counter obligation; the
// event's own memory order is only compared with later accesses. Two tensor
// events with a TENSOR obligation complete in issue order whatever their memory
// order, and with an UNORDERED obligation they do not.
TEST(RaceDetector, TensorDma_PartialWaitsUseTheObligationClassNotTheEventOrder) {
  struct Case {
    MemoryOrderClass obligationClass;
    MemoryOrderClass eventOrder;
    EventStatus olderAfterWait;
  };
  // A completed event has LDS bytes, so the single-wave workgroup retires it.
  for (const Case c :
       {Case{MemoryOrderClass::TENSOR, MemoryOrderClass::UNORDERED, EventStatus::RETIRED},
        Case{MemoryOrderClass::UNORDERED, MemoryOrderClass::TENSOR, EventStatus::ACTIVE}}) {
    SCOPED_TRACE(c.olderAfterWait == EventStatus::RETIRED ? "TENSOR obligation"
                                                          : "UNORDERED obligation");
    std::vector<RaceViolation> violations;
    RaceDetector detector(
        1, Dim3d(0), [&](RaceViolation v) { violations.push_back(v); },
        counterCapacitiesForArch(ROCJITSU_CODE_ARCH_CDNA5), /*ldsSizeBytes=*/64 * 1024);
    auto &wave = detector.getWaveRaceState(0);
    const std::array obligations{
        amdgpu::MemoryCounterObligation{amdgpu::WaitCounterType::TENSORCNT, c.obligationClass}};
    for (uint32_t address : {0u, 64u}) {
      const std::array ranges{LdsByteRange{.address = address, .bytes = 16}};
      wave.registerLdsRangeEvent(testInstruction(address), LdsAccess::Write, ranges, obligations,
                                 c.eventOrder);
    }
    PendingWaitCount wait;
    wait.add(amdgpu::WaitCounterType::TENSORCNT, 1);
    wave.dispatch(wait);
    const auto &events = detector.events();
    EXPECT_EQ(events.status(EventId{0}), c.olderAfterWait);
    EXPECT_EQ(events.status(EventId{1}), EventStatus::ACTIVE);
  }
}

// A transfer without LDS bytes has nothing for another wave to conflict with,
// so once its wait completes it is trimmed without waiting for a barrier.
// Waited transfers in a loop without barriers must not accumulate.
TEST(RaceDetector, TensorDma_WaitedTransfersWithoutRangesDoNotAccumulate) {
  std::vector<RaceViolation> violations;
  RaceDetector detector(
      1, Dim3d(0), [&](RaceViolation v) { violations.push_back(v); },
      counterCapacitiesForArch(ROCJITSU_CODE_ARCH_CDNA5), /*ldsSizeBytes=*/64 * 1024);
  auto &wave = detector.getWaveRaceState(0);
  const std::array obligations{amdgpu::MemoryCounterObligation{amdgpu::WaitCounterType::TENSORCNT,
                                                               MemoryOrderClass::TENSOR}};
  constexpr int kTransfers = 4096;
  for (int i = 0; i < kTransfers; ++i) {
    wave.registerLdsRangeEvent(testInstruction(i), LdsAccess::Read, {}, obligations,
                               MemoryOrderClass::TENSOR);
    PendingWaitCount wait;
    wait.add(amdgpu::WaitCounterType::TENSORCNT, 0);
    wave.dispatch(wait);
  }
  EXPECT_EQ(detector.events().totalAllocated(), kTransfers);
  EXPECT_TRUE(wave.getBarrierPendingEvents().empty());
  // The registry trims in batches, so it holds at most two batches' worth.
  EXPECT_LT(detector.events().size(), 2000);
  EXPECT_TRUE(violations.empty());
}

// A waited transfer with LDS bytes is still visible to other waves until a
// barrier retires it.
TEST(RaceDetector, TensorDma_WaitedTransferWithRangesWaitsForABarrier) {
  RaceTestBuilder b(/*numWaves=*/2);
  b.tensorStoreFromLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.waitTensorcnt(/*wave=*/0, /*tensorcnt=*/0);
  EXPECT_FALSE(b.events().isTrimmable(EventId{0}));
  b.barrier();
  EXPECT_EQ(b.events().status(EventId{0}), EventStatus::RETIRED);
}

TEST(RaceDetector, TensorDma_SameWavePartialTensorcntRetiresOldest) {
  RaceTestBuilder b(/*numWaves=*/1);
  b.tensorLoadToLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.tensorLoadToLds(/*wave=*/0, {{.address = 64, .bytes = 16}});
  b.waitTensorcnt(/*wave=*/0, /*tensorcnt=*/1);
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/64, /*bytes=*/4);
  EXPECT_TRUE(b.hasLdsRace(64));
}

TEST(RaceDetector, TensorDma_SameWavePartialTensorcntRetiresStoreAheadOfLoad) {
  // Loads and stores share one tensorcnt FIFO.
  RaceTestBuilder b(/*numWaves=*/1);
  b.tensorStoreFromLds(/*wave=*/0, {{.address = 32, .bytes = 16}});
  b.tensorLoadToLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.waitTensorcnt(/*wave=*/0, /*tensorcnt=*/1);
  b.checkLdsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/40, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/4, /*bytes=*/4);
  EXPECT_TRUE(b.hasLdsRace(4));
}

// Tensor DMA from one wave completes in issue order, but the ISA does not
// specify the relative timing of overlapping LDS reads and writes, so the
// detector conservatively reports a tensor load and a tensor store from the same
// wave that touch the same bytes, in either order, until s_wait_tensorcnt.
TEST(RaceDetector, TensorDma_SameWaveLoadOverwritingAPendingStoreSourceRaces) {
  RaceTestBuilder b(/*numWaves=*/1);
  b.tensorStoreFromLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.tensorLoadToLds(/*wave=*/0, {{.address = 4, .bytes = 8}});
  EXPECT_TRUE(b.hasLdsRace(4));
}

TEST(RaceDetector, TensorDma_SameWaveStoreReadingAPendingLoadDestinationRaces) {
  RaceTestBuilder b(/*numWaves=*/1);
  b.tensorLoadToLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.tensorStoreFromLds(/*wave=*/0, {{.address = 4, .bytes = 8}});
  EXPECT_TRUE(b.hasLdsRace(4));
}

TEST(RaceDetector, TensorDma_SameWaveOverlappingTransfersAfterTensorcntAreSafe) {
  RaceTestBuilder b(/*numWaves=*/1);
  b.tensorStoreFromLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.waitTensorcnt(/*wave=*/0, /*tensorcnt=*/0);
  b.tensorLoadToLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.waitTensorcnt(/*wave=*/0, /*tensorcnt=*/0);
  b.tensorStoreFromLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, TensorDma_SameWaveStoresOfTheSameTileAreSafe) {
  // Both only read the tile.
  RaceTestBuilder b(/*numWaves=*/1);
  b.tensorStoreFromLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.tensorStoreFromLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, TensorDma_SameWaveLoadsOverTheSameTileAreNotReported) {
  // Two loads writing the same bytes are an LDS write/write conflict, which the
  // detector does not check; this pins that documented limitation.
  RaceTestBuilder b(/*numWaves=*/1);
  b.tensorLoadToLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.tensorLoadToLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, TensorDma_SameWaveDisjointTransfersAreSafe) {
  RaceTestBuilder b(/*numWaves=*/1);
  b.tensorStoreFromLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.tensorLoadToLds(/*wave=*/0, {{.address = 16, .bytes = 16}});
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, TensorDma_CrossWaveLoadThenStoreRace) {
  // Wave 1's store reads bytes wave 0's load may still be writing.
  RaceTestBuilder b(/*numWaves=*/2);
  b.tensorLoadToLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.tensorStoreFromLds(/*wave=*/1, {{.address = 4, .bytes = 8}});
  EXPECT_TRUE(b.hasLdsRace(4));
}

TEST(RaceDetector, TensorDma_CrossWaveStoreThenLoadRace) {
  // Wave 1's load overwrites bytes wave 0's store may still be reading.
  RaceTestBuilder b(/*numWaves=*/2);
  b.tensorStoreFromLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.tensorLoadToLds(/*wave=*/1, {{.address = 4, .bytes = 8}});
  EXPECT_TRUE(b.hasLdsRace(4));
}

TEST(RaceDetector, TensorDma_CrossWaveStoreSourceOverwriteRace) {
  RaceTestBuilder b(/*numWaves=*/2);
  b.tensorStoreFromLds(/*wave=*/0, {{.address = 32, .bytes = 16}});
  b.checkLdsWrite(/*wave=*/1, /*lane=*/0, /*addr=*/40, /*bytes=*/4);
  EXPECT_TRUE(b.hasLdsRace(40));
}

TEST(RaceDetector, TensorDma_CrossWaveStoreSourceOverwriteSafe) {
  RaceTestBuilder b(/*numWaves=*/2);
  b.tensorStoreFromLds(/*wave=*/0, {{.address = 32, .bytes = 16}});
  b.waitTensorcnt(/*wave=*/0, /*tensorcnt=*/0);
  b.barrier();
  b.checkLdsWrite(/*wave=*/1, /*lane=*/0, /*addr=*/40, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, TensorDma_StoreReadsPendingCrossWaveWriteRace) {
  RaceTestBuilder b(/*numWaves=*/2);
  b.ldsWrite(/*wave=*/1, /*lane=*/0, /*addr=*/36, /*bytes=*/4);
  b.tensorStoreFromLds(/*wave=*/0, {{.address = 32, .bytes = 16}});
  EXPECT_TRUE(b.hasLdsRace(36));
}

TEST(RaceDetector, TensorDma_LoadOverwritesPendingCrossWaveReadRace) {
  RaceTestBuilder b(/*numWaves=*/2);
  b.ldsRead(/*wave=*/1, /*lane=*/0, /*addr=*/4, /*bytes=*/4);
  b.tensorLoadToLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  EXPECT_TRUE(b.hasLdsRace(4));
}

TEST(RaceDetector, TensorDma_LoadAfterCrossWaveReadRetiredSafe) {
  RaceTestBuilder b(/*numWaves=*/2);
  b.ldsRead(/*wave=*/1, /*lane=*/0, /*addr=*/4, /*bytes=*/4);
  b.waitcnt(/*wave=*/1, /*vmcnt=*/-1, /*lgkmcnt=*/0);
  b.barrier();
  b.tensorLoadToLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, TensorDma_StoreReadsSameWavePendingDsWriteRace) {
  // DS and tensor DMA complete independently, so the store needs a DS wait.
  RaceTestBuilder b(/*numWaves=*/1);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/36, /*bytes=*/4);
  b.tensorStoreFromLds(/*wave=*/0, {{.address = 32, .bytes = 16}});
  EXPECT_TRUE(b.hasLdsRace(36));
}

TEST(RaceDetector, TensorDma_RaceIsReportedAtAByteBothAccessesTouch) {
  // The store reads bytes 0-127 but shares only bytes 120-123 with the DS
  // write, so the report names byte 120, not the start of the store.
  RaceTestBuilder b(/*numWaves=*/1);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/120, /*bytes=*/4);
  b.tensorStoreFromLds(/*wave=*/0, {{.address = 0, .bytes = 128}});
  EXPECT_TRUE(b.hasLdsRace(120));
  EXPECT_FALSE(b.hasLdsRace(0));
}

TEST(RaceDetector, TensorDma_StoreReadsSameWaveCompletedDsWriteSafe) {
  RaceTestBuilder b(/*numWaves=*/1);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/36, /*bytes=*/4);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0);
  b.tensorStoreFromLds(/*wave=*/0, {{.address = 32, .bytes = 16}});
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, TensorDma_OtherCountersDoNotRetireTransfer) {
  // Only tensorcnt tracks tensor DMA.
  RaceTestBuilder b(/*numWaves=*/1);
  b.tensorLoadToLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.waitcnt(/*wave=*/0, /*vmcnt=*/0, /*lgkmcnt=*/0);
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/12, /*bytes=*/4);
  EXPECT_TRUE(b.hasLdsRace(12));
}

TEST(RaceDetector, TensorDma_TensorcntDoesNotRetireDsWrite) {
  RaceTestBuilder b(/*numWaves=*/1);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/36, /*bytes=*/4);
  b.waitTensorcnt(/*wave=*/0, /*tensorcnt=*/0);
  b.tensorStoreFromLds(/*wave=*/0, {{.address = 32, .bytes = 16}});
  EXPECT_TRUE(b.hasLdsRace(36));
}

TEST(RaceDetector, TensorDma_CrossWaveLoadBarrierWithoutTensorcntRace) {
  // A barrier retires only transfers that their wave has already waited for.
  RaceTestBuilder b(/*numWaves=*/2);
  b.tensorLoadToLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.barrier();
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/4, /*bytes=*/4);
  EXPECT_TRUE(b.hasLdsRace(4));
}

TEST(RaceDetector, TensorDma_CrossWaveStoreWaitWithoutBarrierRace) {
  RaceTestBuilder b(/*numWaves=*/2);
  b.tensorStoreFromLds(/*wave=*/0, {{.address = 32, .bytes = 16}});
  b.waitTensorcnt(/*wave=*/0, /*tensorcnt=*/0);
  b.checkLdsWrite(/*wave=*/1, /*lane=*/0, /*addr=*/40, /*bytes=*/4);
  EXPECT_TRUE(b.hasLdsRace(40));
}

TEST(RaceDetector, TensorDma_DoubleBufferedPartialWaitAndBarrier) {
  // Pipelined kernels wait for all but the newest transfer, then synchronize.
  RaceTestBuilder b(/*numWaves=*/2);
  b.tensorLoadToLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.tensorLoadToLds(/*wave=*/0, {{.address = 64, .bytes = 16}});
  b.waitTensorcnt(/*wave=*/0, /*tensorcnt=*/1);
  b.barrier();
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/64, /*bytes=*/4);
  EXPECT_TRUE(b.hasLdsRace(64));
}

TEST(RaceDetector, TensorDma_LaterWaitNeedsAnotherBarrier) {
  // A barrier retires only transfers that were complete when it executed.
  RaceTestBuilder b(/*numWaves=*/2);
  b.tensorLoadToLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.tensorLoadToLds(/*wave=*/0, {{.address = 64, .bytes = 16}});
  b.waitTensorcnt(/*wave=*/0, /*tensorcnt=*/1);
  b.barrier();
  b.waitTensorcnt(/*wave=*/0, /*tensorcnt=*/0);
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/64, /*bytes=*/4);
  EXPECT_TRUE(b.hasLdsRace(64));
  b.clearViolations();
  b.barrier();
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/64, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, TensorDma_DtlOverwritingPendingStoreSourceRace) {
  // Direct-to-LDS completes independently of tensor DMA, so overwriting a
  // pending store's source races until tensorcnt retires the store.
  RaceTestBuilder b(/*numWaves=*/1);
  b.tensorStoreFromLds(/*wave=*/0, {{.address = 32, .bytes = 16}});
  b.globalToLds(/*wave=*/0, /*ldsAddrs=*/{36}, /*bytesPerLane=*/4, /*exec=*/1);
  EXPECT_TRUE(b.hasLdsRace(36));
}

TEST(RaceDetector, TensorDma_DtlAfterStoreRetiredSafe) {
  RaceTestBuilder b(/*numWaves=*/1);
  b.tensorStoreFromLds(/*wave=*/0, {{.address = 32, .bytes = 16}});
  b.waitTensorcnt(/*wave=*/0, /*tensorcnt=*/0);
  b.globalToLds(/*wave=*/0, /*ldsAddrs=*/{36}, /*bytesPerLane=*/4, /*exec=*/1);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, TensorDma_StoreReadsSameWavePendingDtlRace) {
  // Direct-to-LDS completes on vmcnt, independently of tensor DMA.
  RaceTestBuilder b(/*numWaves=*/1);
  b.globalToLds(/*wave=*/0, /*ldsAddrs=*/{32}, /*bytesPerLane=*/4, /*exec=*/1);
  b.tensorStoreFromLds(/*wave=*/0, {{.address = 32, .bytes = 16}});
  EXPECT_TRUE(b.hasLdsRace(32));
}

TEST(RaceDetector, TensorDma_StoreReadsSameWaveCompletedDtlSafe) {
  RaceTestBuilder b(/*numWaves=*/1);
  b.globalToLds(/*wave=*/0, /*ldsAddrs=*/{32}, /*bytesPerLane=*/4, /*exec=*/1);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/0);
  b.tensorStoreFromLds(/*wave=*/0, {{.address = 32, .bytes = 16}});
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, TensorDma_TransferWithoutRangesTakesATensorcntSlot) {
  RaceTestBuilder b(/*numWaves=*/1);
  b.tensorLoadToLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.tensorLoadToLds(/*wave=*/0, {});
  b.waitTensorcnt(/*wave=*/0, /*tensorcnt=*/1);
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/4, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, TensorDma_EachTransferWithoutRangesTakesOneTensorcntSlot) {
  RaceTestBuilder b(/*numWaves=*/1);
  b.tensorLoadToLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.tensorLoadToLds(/*wave=*/0, {});
  b.tensorLoadToLds(/*wave=*/0, {});
  b.waitTensorcnt(/*wave=*/0, /*tensorcnt=*/2);
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/4, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, TensorDma_PartialWaitRetiresAnOlderTransferWithoutRanges) {
  RaceTestBuilder b(/*numWaves=*/1);
  b.tensorLoadToLds(/*wave=*/0, {});
  b.tensorLoadToLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.waitTensorcnt(/*wave=*/0, /*tensorcnt=*/1);
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/4, /*bytes=*/4);
  EXPECT_TRUE(b.hasLdsRace(4));
}

TEST(RaceDetector, TensorDma_StoreWithoutRangesTakesATensorcntSlot) {
  RaceTestBuilder b(/*numWaves=*/1);
  b.tensorLoadToLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.tensorStoreFromLds(/*wave=*/0, {});
  b.waitTensorcnt(/*wave=*/0, /*tensorcnt=*/1);
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/4, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, TensorDma_TransferWithoutRangesConflictsWithNothing) {
  RaceTestBuilder b(/*numWaves=*/2);
  b.tensorLoadToLds(/*wave=*/0, {});
  b.tensorStoreFromLds(/*wave=*/0, {});
  for (int wave : {0, 1}) {
    b.checkLdsRead(wave, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
    b.checkLdsWrite(wave, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  }
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, TensorDma_RetiredTransferWithoutRangesNoLongerCounts) {
  RaceTestBuilder b(/*numWaves=*/1);
  b.tensorLoadToLds(/*wave=*/0, {});
  b.waitTensorcnt(/*wave=*/0, /*tensorcnt=*/0);
  // Without LDS bytes, the wait alone finishes the transfer; it needs no
  // barrier. Later loads alone could not show that, since they are newer.
  const EventId empty{0};
  ASSERT_EQ(b.events().totalAllocated(), 1);
  EXPECT_TRUE(b.events().isTrimmable(empty));
  b.barrier();
  b.tensorLoadToLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.tensorLoadToLds(/*wave=*/0, {{.address = 64, .bytes = 16}});
  b.waitTensorcnt(/*wave=*/0, /*tensorcnt=*/1);
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/64, /*bytes=*/4);
  EXPECT_TRUE(b.hasLdsRace(64));
}

TEST(RaceDetector, TensorDma_UnwaitedTransferWithoutRangesKeepsItsSlotAcrossBarrier) {
  // A barrier retires only transfers that their wave has waited for, so the
  // unwaited transfer without ranges still holds the newest slot afterwards:
  // waiting for all but one completes the load before it. Had it lost its
  // slot, the load would still be pending and the read would race.
  RaceTestBuilder b(/*numWaves=*/1);
  b.tensorLoadToLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.tensorLoadToLds(/*wave=*/0, {});
  b.barrier();
  b.waitTensorcnt(/*wave=*/0, /*tensorcnt=*/1);
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/4, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, TensorDma_RangeEdgesAreBytePrecise) {
  RaceTestBuilder b(/*numWaves=*/2);
  b.tensorLoadToLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/16, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/15, /*bytes=*/1);
  EXPECT_TRUE(b.hasLdsRace(15));
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/14, /*bytes=*/4);
  EXPECT_TRUE(b.hasLdsRace(14));
}

TEST(RaceDetector, TensorDma_RangeEndingMidElementTracksItsTailBytes) {
  // A range can end mid-element, as when an element straddles the end of the
  // wave's LDS allocation. Its trailing bytes are tracked, not rounded away.
  RaceTestBuilder b(/*numWaves=*/2);
  b.tensorLoadToLds(/*wave=*/0, {{.address = 0, .bytes = 6}});
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/4, /*bytes=*/2);
  EXPECT_TRUE(b.hasLdsRace(4));
}

TEST(RaceDetector, Exec_DirectToLdsTracksOnlyActiveLaneIntervals) {
  RaceTestBuilder b(/*numWaves=*/1);
  b.globalToLds(/*wave=*/0, /*ldsAddrs=*/{16}, /*bytesPerLane=*/4, /*exec=*/1);

  // globalToLds pads inactive lane addresses with zero. The explicit one-lane
  // exec mask must keep that padding out of the tracked LDS intervals.
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());

  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/16, /*bytes=*/4);
  EXPECT_TRUE(b.hasLdsRace(16));
}

TEST(RaceDetector, ViolationReportsAddressAndWorkgroup) {
  RaceTestBuilder b(1, 64, Dim3d(7, 3, 1));
  b.globalToLds(0, {16}, 4, 1);
  b.checkLdsRead(0, 5, 16, 4);
  ASSERT_EQ(b.raceCount(), 1);
  const auto &v = b.violations()[0];
  EXPECT_EQ(v.address, 16);
  EXPECT_EQ(v.wave, 0);
  EXPECT_EQ(v.lane, 5);
  EXPECT_FALSE(v.isWrite);
  EXPECT_EQ(v.conflictingEvent, EventId{0});
  EXPECT_EQ(v.workgroupId.x, 7);
  EXPECT_EQ(v.workgroupId.y, 3);
  EXPECT_EQ(v.workgroupId.z, 1);
}

TEST(RaceDetector, GlobalToLdsHonorsLaneMask) {
  RaceTestBuilder b(/*numWaves=*/2, /*waveSize=*/4);
  b.globalToLds(/*wave=*/0, /*ldsAddrs=*/{0, 4, 8, 12}, /*bytesPerLane=*/4,
                /*exec=*/0x1);

  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/4, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());

  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  EXPECT_TRUE(b.hasLdsRace(0));
}

TEST(RaceDetector, LdsSameWave_MultiLaneReadOk) {
  // All lanes read same LDS byte after write+waitcnt → safe.
  // Multiple lanes reading same address is not a race.
  RaceTestBuilder b(/*numWaves=*/1);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0);
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.checkLdsRead(/*wave=*/0, /*lane=*/1, /*addr=*/0, /*bytes=*/4);
  b.checkLdsRead(/*wave=*/0, /*lane=*/2, /*addr=*/0, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, LdsSameWave_ScatterBroadcastOk) {
  // Each lane writes unique addr, waitcnt, then all read addr 0 → safe.
  RaceTestBuilder b(/*numWaves=*/1);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.ldsWrite(/*wave=*/0, /*lane=*/1, /*addr=*/4, /*bytes=*/4);
  b.ldsWrite(/*wave=*/0, /*lane=*/2, /*addr=*/8, /*bytes=*/4);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0);
  // All lanes read from addr 0 (lane 0's write)
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.checkLdsRead(/*wave=*/0, /*lane=*/1, /*addr=*/0, /*bytes=*/4);
  b.checkLdsRead(/*wave=*/0, /*lane=*/2, /*addr=*/0, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, LdsSameWave_AllLanesWriteSameAddr) {
  // All lanes write same address (WAW). Currently not detected as a race.
  // TODO(newling): WAW detection is not implemented — this documents current
  // behavior.
  RaceTestBuilder b(/*numWaves=*/1);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.ldsWrite(/*wave=*/0, /*lane=*/1, /*addr=*/0, /*bytes=*/4);
  b.ldsWrite(/*wave=*/0, /*lane=*/2, /*addr=*/0, /*bytes=*/4);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0);
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  // Currently passes (WAW not flagged). If WAW detection is added, this
  // test should be updated to EXPECT_TRUE(b.hasRace()).
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, LdsCrossWave_IrregularSizesAndOffsets) {
  // Exercises byte-count tracking with non-aligned, varying-size accesses
  // that may share or straddle count granularity boundaries.
  RaceTestBuilder b(/*numWaves=*/2);

  // Wave 0: irregular writes at odd offsets and sizes.
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/3, /*bytes=*/1);   // [3,4)
  b.ldsWrite(/*wave=*/0, /*lane=*/1, /*addr=*/7, /*bytes=*/3);   // [7,10)
  b.ldsWrite(/*wave=*/0, /*lane=*/2, /*addr=*/15, /*bytes=*/6);  // [15,21)
  b.ldsWrite(/*wave=*/0, /*lane=*/3, /*addr=*/100, /*bytes=*/2); // [100,102)
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0);
  // No barrier: cross-wave reads should detect races.

  // Wave 1 reads that overlap written ranges — should race.
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/3,
                 /*bytes=*/1); // [3,4) ∩ [3,4)
  EXPECT_TRUE(b.hasLdsRace(3));
  b.clearViolations();

  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/8,
                 /*bytes=*/1); // [8,9) ∩ [7,10)
  EXPECT_TRUE(b.hasLdsRace(8));
  b.clearViolations();

  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/17,
                 /*bytes=*/2); // [17,19) ∩ [15,21)
  EXPECT_TRUE(b.hasLdsRace(17));
  b.clearViolations();

  // Reads that are adjacent but don't overlap — should be safe.
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/2,
                 /*bytes=*/1); // [2,3) ∩ [3,4) = ∅
  EXPECT_FALSE(b.hasRace());

  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/10,
                 /*bytes=*/5); // [10,15) ∩ [15,21) = ∅
  EXPECT_FALSE(b.hasRace());

  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/21,
                 /*bytes=*/4); // [21,25) ∩ [15,21) = ∅
  EXPECT_FALSE(b.hasRace());

  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/102,
                 /*bytes=*/8); // [102,110) ∩ [100,102) = ∅
  EXPECT_FALSE(b.hasRace());

  // After barrier, everything should be safe.
  b.barrier();
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/3, /*bytes=*/1); // [3,4)
  EXPECT_FALSE(b.hasRace());

  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/15, /*bytes=*/6); // [15,21)
  EXPECT_FALSE(b.hasRace());
}

//
// Two waves, each doing 4 partial-exec LDS writes to different regions,
// then cross-wave reads. The lgkmcnt values control how many writes are
// drained before each barrier.
//
// Wave 0 writes: [0,64), [64,128), [128,192), [192,256)  (4 writes, lanes 0-15
// each) Wave 1 writes: [256,320), [320,384), [384,448), [448,512) After
// barrier, each wave reads the other's region.
//
// Ported from: ExecMask_CrossWave_Correct_2_0 and variants.

// Helper: set up 2-wave cross-wave exec mask test.
// Each wave does 4 LDS writes with partial exec, then waitcnt(lgkmcnt1),
// barrier, cross-wave read, optionally waitcnt(lgkmcnt2) + barrier2, then
// second cross-wave read.
// Each wave does 4 LDS writes to distinct addresses. lgkmcnt controls how
// many are drained before the barrier. Cross-wave reads then check whether
// the barrier retired enough events.
static void crossWaveExecTest(int lgkmcnt1, int lgkmcnt2, bool barrier1, bool barrier2,
                              bool expectFirstReadRace, bool expectSecondReadRace) {
  RaceTestBuilder b(/*numWaves=*/2);

  // Wave 0: 4 writes to addrs 0, 4, 8, 12
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/4, /*bytes=*/4);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/8, /*bytes=*/4);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/12, /*bytes=*/4);

  // Wave 1: 4 writes to addrs 256, 260, 264, 268
  b.ldsWrite(/*wave=*/1, /*lane=*/0, /*addr=*/256, /*bytes=*/4);
  b.ldsWrite(/*wave=*/1, /*lane=*/0, /*addr=*/260, /*bytes=*/4);
  b.ldsWrite(/*wave=*/1, /*lane=*/0, /*addr=*/264, /*bytes=*/4);
  b.ldsWrite(/*wave=*/1, /*lane=*/0, /*addr=*/268, /*bytes=*/4);

  // Both waves: waitcnt lgkmcnt1
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/lgkmcnt1);
  b.waitcnt(/*wave=*/1, /*vmcnt=*/-1, /*lgkmcnt=*/lgkmcnt1);

  if (barrier1) {
    b.barrier();
  }

  // Cross-wave read: wave 0 reads wave 1's addr 260 (2nd write).
  // With lgkmcnt(N), the (4-N) oldest writes are drained. Addr 260 (2nd)
  // is safe only if lgkmcnt <= 2 (drain at least 2 oldest) + barrier.
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/260, /*bytes=*/4);
  EXPECT_EQ(b.hasRace(), expectFirstReadRace);
  b.clearViolations();

  if (lgkmcnt2 >= 0) {
    b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/lgkmcnt2);
    b.waitcnt(/*wave=*/1, /*vmcnt=*/-1, /*lgkmcnt=*/lgkmcnt2);
  }

  if (barrier2) {
    b.barrier();
  }

  // Cross-wave read: wave 0 reads wave 1's addr 264
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/264, /*bytes=*/4);
  EXPECT_EQ(b.hasRace(), expectSecondReadRace);
}

TEST(RaceDetector, Exec_CrossWaveCorrect) {
  // lgkmcnt(2): drain 2 oldest writes, barrier flushes them.
  // lgkmcnt(0): drain remaining, barrier flushes. All safe.
  crossWaveExecTest(/*lgkmcnt1=*/2, /*lgkmcnt2=*/0,
                    /*barrier1=*/true, /*barrier2=*/true,
                    /*expectFirstReadRace=*/false,
                    /*expectSecondReadRace=*/false);
}

TEST(RaceDetector, Exec_CrossWaveOverRetire00) {
  // lgkmcnt(0): all 4 writes drained before barrier 1. Conservative but safe.
  crossWaveExecTest(/*lgkmcnt1=*/0, /*lgkmcnt2=*/0,
                    /*barrier1=*/true, /*barrier2=*/true,
                    /*expectFirstReadRace=*/false,
                    /*expectSecondReadRace=*/false);
}

TEST(RaceDetector, Exec_CrossWaveOverRetire11) {
  // lgkmcnt(1): drain 3 of 4 writes each time. Safe.
  crossWaveExecTest(/*lgkmcnt1=*/1, /*lgkmcnt2=*/1,
                    /*barrier1=*/true, /*barrier2=*/true,
                    /*expectFirstReadRace=*/false,
                    /*expectSecondReadRace=*/false);
}

TEST(RaceDetector, Exec_CrossWaveUnderRetire30) {
  // lgkmcnt(3): only drain 1 of 4, barrier flushes only that 1.
  // First cross-wave read hits unflushed writes → RACE.
  // Second read after lgkmcnt(0)+barrier → safe.
  crossWaveExecTest(/*lgkmcnt1=*/3, /*lgkmcnt2=*/0,
                    /*barrier1=*/true, /*barrier2=*/true,
                    /*expectFirstReadRace=*/true,
                    /*expectSecondReadRace=*/false);
}

TEST(RaceDetector, Exec_CrossWaveUnderRetire22) {
  // lgkmcnt(2): drain 2 oldest (256,260), barrier flushes those 2.
  // First read at 260 → safe (was drained+flushed).
  // 2 remain (264,268). Second lgkmcnt(2) is noop (already ≤2).
  // Barrier2 flushes nothing new. Second read at 264 → RACE.
  crossWaveExecTest(/*lgkmcnt1=*/2, /*lgkmcnt2=*/2,
                    /*barrier1=*/true, /*barrier2=*/true,
                    /*expectFirstReadRace=*/false,
                    /*expectSecondReadRace=*/true);
}

TEST(RaceDetector, Exec_CrossWaveMissingBarrier1) {
  // No first barrier → cross-wave read hits WAVE_COMPLETE events → RACE.
  crossWaveExecTest(/*lgkmcnt1=*/2, /*lgkmcnt2=*/0,
                    /*barrier1=*/false, /*barrier2=*/true,
                    /*expectFirstReadRace=*/true,
                    /*expectSecondReadRace=*/false);
}

TEST(RaceDetector, Exec_CrossWaveMissingBarrier2) {
  // lgkmcnt(2) + barrier1 flushes 2 oldest (256,260).
  // First read at 260 → safe. lgkmcnt(0) drains remaining 2 (264,268)
  // but no barrier2 → WAVE_COMPLETE not RETIRED → second read RACE.
  crossWaveExecTest(/*lgkmcnt1=*/2, /*lgkmcnt2=*/0,
                    /*barrier1=*/true, /*barrier2=*/false,
                    /*expectFirstReadRace=*/false,
                    /*expectSecondReadRace=*/true);
}

TEST(RaceDetector, CounterOnlyIssueDoesNotOverRetireYoungerLdsEffect) {
  RaceTestBuilder b(1);
  b.counterIssue(0, amdgpu::WaitCounterType::VMCNT, MemoryOrderClass::VMEM);
  b.globalToLds(0, {0}, 4, 1);
  b.waitcnt(0, 1);
  EXPECT_EQ(b.events().status(EventId{0}), EventStatus::WAVE_COMPLETE);
  b.checkLdsRead(0, 0, 0, 4);
  EXPECT_TRUE(b.hasLdsRace(0));
}

TEST(RaceDetector, Dtl_MultiLane_CrossWaveRace) {
  // DTL with 4 active lanes, each writing 16 bytes to different LDS offsets.
  // Cross-wave read to lane 2's region should detect the race.
  RaceTestBuilder b(/*numWaves=*/2);
  // Wave 0: DTL load, lanes 0-3 active, each gets 16 bytes.
  // Lane 0 → LDS[0..16), lane 1 → LDS[16..32), lane 2 → LDS[32..48), lane 3 →
  // LDS[48..64)
  b.globalToLds(/*wave=*/0, /*ldsAddrs=*/{0, 16, 32, 48},
                /*bytesPerLane=*/16, /*exec=*/0xF);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/0);
  // No barrier — cross-wave read should race.
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/32,
                 /*bytes=*/4); // lane 2's region
  EXPECT_TRUE(b.hasLdsRace(32));
}

TEST(RaceDetector, Dtl_MultiLane_CrossWaveSafeWithBarrier) {
  // Same as above but with barrier → all lanes' regions safe.
  RaceTestBuilder b(/*numWaves=*/2);
  b.globalToLds(/*wave=*/0, {0, 16, 32, 48}, /*bytesPerLane=*/16, /*exec=*/0xF);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/0);
  b.barrier();
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/16, /*bytes=*/4);
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/32, /*bytes=*/4);
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/48, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, DualAddress_CrossWaveRace) {
  // Like ds_write2_b64, lane 0 writes [100, 108) and [116, 124). Another wave
  // reading either interval without a barrier races.
  std::vector<RaceViolation> violations;
  RaceDetector detector(/*nWaves=*/
                        2, Dim3d(0), [&](RaceViolation v) { violations.push_back(v); },
                        counterCapacitiesForArch(ROCJITSU_CODE_ARCH_CDNA4),
                        /*ldsSizeBytes=*/64 * 1024);
  auto &rs = detector.getWaveRaceState(0);
  std::vector<uint32_t> firstAddrs(64, 0);
  std::vector<uint32_t> secondAddrs(64, 0);
  firstAddrs[0] = 100;
  secondAddrs[0] = 116;
  const std::array obligations{
      amdgpu::MemoryCounterObligation{amdgpu::WaitCounterType::LGKMCNT, MemoryOrderClass::LDS}};
  rs.registerLdsEvent(testInstruction(/*pc=*/0), LdsAccess::Write, /*laneMask=*/1, /*waveSize=*/64,
                      firstAddrs, secondAddrs, /*bytesPerLane=*/8, obligations,
                      MemoryOrderClass::LDS);
  detector.validateRead(/*addr=*/100, WaveId{1}, /*lane=*/0, /*nBytes=*/4);
  EXPECT_EQ(violations.size(), 1u); // first interval
  detector.validateRead(/*addr=*/116, WaveId{1}, /*lane=*/0, /*nBytes=*/4);
  EXPECT_EQ(violations.size(), 2u); // second interval
  // Outside both intervals: no race
  detector.validateRead(/*addr=*/108, WaveId{1}, /*lane=*/0, /*nBytes=*/4);
  EXPECT_EQ(violations.size(), 2u); // still 2
}

TEST(RaceDetector, MixedCountersRetireIndependently) {
  for (bool vmWait : {false, true})
    for (bool lgkmWait : {false, true}) {
      SCOPED_TRACE(testing::Message() << vmWait << " " << lgkmWait);
      RaceTestBuilder b(1);
      b.counterIssue(0, amdgpu::WaitCounterType::VMCNT, MemoryOrderClass::VMEM);
      b.ldsRead(0, 0, 0, 4);
      b.counterIssue(0, amdgpu::WaitCounterType::LGKMCNT, MemoryOrderClass::UNORDERED);
      b.waitcnt(0, vmWait ? 0 : -1, lgkmWait ? 0 : -1);
      EXPECT_EQ(b.events().status(EventId{0}),
                vmWait ? EventStatus::WAVE_COMPLETE : EventStatus::ACTIVE);
      EXPECT_EQ(b.events().status(EventId{1}),
                lgkmWait ? EventStatus::RETIRED : EventStatus::ACTIVE);
      EXPECT_EQ(b.events().status(EventId{2}),
                lgkmWait ? EventStatus::WAVE_COMPLETE : EventStatus::ACTIVE);
      b.checkLdsWrite(0, 0, 0, 4, MemoryOrderClass::UNORDERED);
      EXPECT_EQ(b.hasLdsRace(0), !lgkmWait);
    }
}

TEST(RaceDetector, EventRegistry_NarrowWindow) {
  constexpr int window = 10;
  RaceTestBuilder b(1);
  for (int i = 0; i < window; ++i)
    b.counterIssue(0, amdgpu::WaitCounterType::VMCNT, MemoryOrderClass::VMEM);

  for (int i = 0; i < 1000; ++i) {
    SCOPED_TRACE(i);
    b.waitcnt(0, window - 1);
    EXPECT_EQ(b.events().status(EventId{i}), EventStatus::WAVE_COMPLETE);
    EXPECT_EQ(b.events().status(EventId{i + 1}), EventStatus::ACTIVE);
    EXPECT_EQ(b.events().status(EventId{i + window - 1}), EventStatus::ACTIVE);
    b.counterIssue(0, amdgpu::WaitCounterType::VMCNT, MemoryOrderClass::VMEM);
  }
  EXPECT_GT(b.events().trimmedCount(), 0);
}

TEST(RaceDetector, EventRegistry_TrimWithoutBarrier) {
  // Single wave, no LDS, no barrier. WAVE_COMPLETE non-LDS events are
  // trimmable. Trimming is triggered automatically inside add() every
  // kTrimAttemptInterval allocations.
  int N = EventRegistry::kTrimAttemptInterval - 1;
  RaceTestBuilder b(/*numWaves=*/1);

  // Allocate N events (one short of the trim interval) — no trim yet.
  for (int i = 0; i < N; ++i) {
    b.counterIssue(/*wave=*/0, amdgpu::WaitCounterType::VMCNT, MemoryOrderClass::VMEM);
    b.waitcnt(/*wave=*/0, /*vmcnt=*/0);
  }
  EXPECT_EQ(b.events().totalAllocated(), N);
  EXPECT_EQ(b.events().trimmedCount(), 0);

  // One more allocation triggers trimming.
  b.counterIssue(/*wave=*/0, amdgpu::WaitCounterType::VMCNT, MemoryOrderClass::VMEM);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/0);

  EXPECT_EQ(b.events().totalAllocated(), N + 1);
  EXPECT_GT(b.events().trimmedCount(), 0);
  EXPECT_LT(b.events().size(), N);

  // LDS detection still works after trimming counter-only events.
  b.globalToLds(0, {0}, 4, 1);
  b.checkLdsRead(0, 0, 0, 4);
  EXPECT_TRUE(b.hasLdsRace(0));
}

TEST(RaceDetector, EventRegistry_TrimWithBarrier) {
  // Two waves, LDS events require s_barrier to reach RETIRED.
  // Trimming is triggered inside add() after enough allocations.
  int N = EventRegistry::kTrimAttemptInterval - 1;
  RaceTestBuilder b(/*numWaves=*/2);

  // Allocate N events (one short of the trim interval) — no trim yet.
  for (int i = 0; i < N; ++i) {
    b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/(i % 16) * 4, /*bytes=*/4);
    b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0);
    b.barrier();
  }
  EXPECT_EQ(b.events().totalAllocated(), N);
  EXPECT_EQ(b.events().trimmedCount(), 0);

  // One more allocation triggers trimming.
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0);
  b.barrier();

  EXPECT_EQ(b.events().totalAllocated(), N + 1);
  EXPECT_GT(b.events().trimmedCount(), 0);
  EXPECT_LT(b.events().size(), N);

  // LDS detection still works after trimming.
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  EXPECT_TRUE(b.hasLdsRace(0));
}

//
TEST(RaceDetector, HandlerFiresPerLdsConsumerLane) {
  RaceTestBuilder b(2);
  b.ldsWrite(0, 0, 0, 4);
  for (int lane = 0; lane < 64; ++lane)
    b.checkLdsRead(1, lane, 0, 4);
  EXPECT_EQ(b.raceCount(), 64);
}

TEST(RaceDetector, MultipleDistinctLdsRaces) {
  RaceTestBuilder b(2);
  b.ldsWrite(0, 0, 0, 4);
  b.ldsWrite(0, 0, 16, 4);
  b.checkLdsRead(1, 0, 0, 4);
  b.checkLdsRead(1, 0, 16, 4);
  ASSERT_EQ(b.raceCount(), 2);
  EXPECT_TRUE(b.hasLdsRace(0));
  EXPECT_TRUE(b.hasLdsRace(16));
  EXPECT_NE(b.violations()[0].conflictingEvent, b.violations()[1].conflictingEvent);
}

TEST(RaceDetector, RepeatedLdsCheckRetainsTheConflict) {
  RaceTestBuilder b(2);
  b.ldsWrite(0, 0, 0, 4);
  for (int i = 0; i < 100; ++i)
    b.checkLdsRead(1, 0, 0, 4);
  EXPECT_EQ(b.raceCount(), 100);
}
