// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Unit tests for the race detection core (RaceDetector + WaveRaceState).
//
// These tests drive the detection API directly via RaceTestBuilder — no
// assembly parser, no instruction emulator, no GPU. Register hazards are
// covered by the core memory-wait scoreboard tests instead.
//
// Test categories:
//   Sgpr*_          — scalar-memory events and the counters they share
//   LdsCrossWave_*  — cross-wave LDS races (missing barrier)
//   LdsSameWave_*   — same-wave LDS instruction ordering
//   SameWave_*      — same-wave LDS event lifetime across waits and barriers
//   DeepStack_*     — multiple outstanding loads with partial waitcnt
//   Dtl_*           — Direct-to-LDS global loads
//   Exec_*          — exec mask effects on race detection
//   MultiWorkgroup_* — workgroup ID in violation reports
//   GlobalStore_*   — global stores and vmcnt interaction
//   Mixed_*         — simultaneous vmcnt and lgkmcnt events
//   DualOffset_*    — dual-offset LDS events

#include "race_test_builder.h"
#include <array>
#include <gtest/gtest.h>
#include <limits>

using namespace rocjitsu::plugins::race_detector;
namespace amdgpu = rocjitsu::amdgpu;

namespace {

constexpr std::array kCdna1To4GenericFlatLoadObligations{
    amdgpu::MemoryCounterObligation{amdgpu::WaitCounterType::VMCNT, MemoryOrderClass::UNORDERED},
    amdgpu::MemoryCounterObligation{amdgpu::WaitCounterType::LGKMCNT, MemoryOrderClass::UNORDERED}};

constexpr std::array kMixedOrderGenericFlatLoadObligations{
    amdgpu::MemoryCounterObligation{amdgpu::WaitCounterType::LOADCNT, MemoryOrderClass::VMEM},
    amdgpu::MemoryCounterObligation{amdgpu::WaitCounterType::DSCNT, MemoryOrderClass::LDS}};

} // namespace

TEST(RaceDetectorDefaults, CoversEveryMemoryEventType) {
  EXPECT_EQ(defaultWaitCounterType(MemoryEventType::GLOBAL_TO_VGPR),
            amdgpu::WaitCounterType::VMCNT);
  EXPECT_EQ(defaultWaitCounterType(MemoryEventType::VGPR_TO_GLOBAL),
            amdgpu::WaitCounterType::VMCNT);
  EXPECT_EQ(defaultWaitCounterType(MemoryEventType::GLOBAL_TO_LDS), amdgpu::WaitCounterType::VMCNT);
  EXPECT_EQ(defaultWaitCounterType(MemoryEventType::LDS_TO_VGPR), amdgpu::WaitCounterType::LGKMCNT);
  EXPECT_EQ(defaultWaitCounterType(MemoryEventType::VGPR_TO_LDS), amdgpu::WaitCounterType::LGKMCNT);
  EXPECT_EQ(defaultWaitCounterType(MemoryEventType::GLOBAL_TO_SGPR),
            amdgpu::WaitCounterType::LGKMCNT);
  EXPECT_EQ(defaultWaitCounterType(MemoryEventType::SCALAR_TO_GLOBAL),
            amdgpu::WaitCounterType::LGKMCNT);

  EXPECT_EQ(defaultMemoryOrder(MemoryEventType::GLOBAL_TO_VGPR), MemoryOrderClass::VMEM);
  EXPECT_EQ(defaultMemoryOrder(MemoryEventType::VGPR_TO_GLOBAL), MemoryOrderClass::VMEM);
  EXPECT_EQ(defaultMemoryOrder(MemoryEventType::GLOBAL_TO_LDS), MemoryOrderClass::VMEM);
  EXPECT_EQ(defaultMemoryOrder(MemoryEventType::LDS_TO_VGPR), MemoryOrderClass::LDS);
  EXPECT_EQ(defaultMemoryOrder(MemoryEventType::VGPR_TO_LDS), MemoryOrderClass::LDS);
  EXPECT_EQ(defaultMemoryOrder(MemoryEventType::GLOBAL_TO_SGPR), MemoryOrderClass::UNORDERED);
  EXPECT_EQ(defaultMemoryOrder(MemoryEventType::SCALAR_TO_GLOBAL), MemoryOrderClass::UNORDERED);
}

TEST(RaceDetector, FlatGlobalLoadRequiresBothCounterWaits) {
  RaceTestBuilder b(/*numWaves=*/1);
  const EventId load = b.flatGlobalLoad(/*wave=*/0, kCdna1To4GenericFlatLoadObligations);

  b.waitcnt(/*wave=*/0, /*vmcnt=*/0);
  EXPECT_TRUE(b.isPending(load));

  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0);
  EXPECT_FALSE(b.isPending(load));
}

TEST(RaceDetector, FlatLdsLoadRequiresBothCounterWaits) {
  RaceTestBuilder b(/*numWaves=*/1);
  const EventId load = b.flatLdsLoad(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);

  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0);
  EXPECT_TRUE(b.isPending(load));

  b.waitcnt(/*wave=*/0, /*vmcnt=*/0);
  EXPECT_FALSE(b.isPending(load));
}

TEST(RaceDetector, MixedFlatKeepsOneCounterEntryPerInstruction) {
  for (uint64_t lds_mask : {0u, 1u, 2u, 5u}) {
    SCOPED_TRACE(lds_mask);
    RaceDetector detector(
        1, Dim3d(0), [](RaceViolation) {}, counterCapacitiesForArch(ROCJITSU_CODE_ARCH_RDNA4),
        /*ldsSizeBytes=*/64 * 1024);
    auto &wave = detector.getWaveRaceState(0);
    const std::array<uint32_t, 32> addresses{64, 128, 256};
    for (uint64_t pc : {1u, 2u})
      wave.registerLdsEvent(pc, MemoryEventType::LDS_TO_VGPR, /*execMask=*/3,
                            /*waveSize=*/32, addresses, /*bytesPerLane=*/4,
                            kMixedOrderGenericFlatLoadObligations, MemoryOrderClass::UNORDERED,
                            lds_mask);
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
    EXPECT_EQ(events.status(EventId{0}), EventStatus::WAVE_COMPLETE);
    EXPECT_EQ(events.status(EventId{1}), EventStatus::ACTIVE);
    wait.updates.clear();
    wait.add(amdgpu::WaitCounterType::LOADCNT, 0);
    wait.add(amdgpu::WaitCounterType::DSCNT, 0);
    wave.dispatch(wait);
    EXPECT_EQ(events.status(EventId{1}), EventStatus::WAVE_COMPLETE);
  }
}

TEST(RaceDetector, LdsValidationRejectsNegativeAndOverflowingRanges) {
  std::vector<RaceViolation> violations;
  RaceDetector detector(
      2, Dim3d(0), [&](RaceViolation v) { violations.push_back(v); },
      counterCapacitiesForArch(ROCJITSU_CODE_ARCH_CDNA5), /*ldsSizeBytes=*/64 * 1024);
  for (const auto type : {MemoryEventType::LDS_TO_VGPR, MemoryEventType::VGPR_TO_LDS}) {
    IntervalSet intervals;
    intervals.append(64, 68);
    detector.allocateEventId(WaveId{0}, 0, type, std::move(intervals),
                             amdgpu::WaitCounterType::DSCNT, MemoryOrderClass::LDS);
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

TEST(RaceDetector, InvalidLdsAddressesKeepCounterDependencies) {
  for (bool dual_access : {false, true}) {
    SCOPED_TRACE(dual_access);
    RaceDetector detector(
        1, Dim3d(0), [](RaceViolation) {}, counterCapacitiesForArch(ROCJITSU_CODE_ARCH_CDNA5),
        /*ldsSizeBytes=*/64 * 1024);
    auto &wave = detector.getWaveRaceState(0);
    const std::array<uint32_t, 4> addresses{0x80000040u, UINT32_MAX, 0x7ffffffeu, 64};
    const std::array<uint32_t, 4> second_addresses{0x80000080u, UINT32_MAX, 0x7fffffffu, 96};
    if (dual_access)
      wave.registerLdsEvent(0, MemoryEventType::LDS_TO_VGPR, 0xf, 4, addresses, second_addresses, 4,
                            kMixedOrderGenericFlatLoadObligations, MemoryOrderClass::UNORDERED);
    else
      wave.registerLdsEvent(0, MemoryEventType::LDS_TO_VGPR, 0xf, 4, addresses, 4,
                            kMixedOrderGenericFlatLoadObligations, MemoryOrderClass::UNORDERED);
    const auto &events = detector.events();
    const EventId event{0};
    ASSERT_EQ(events.totalAllocated(), 1);
    EXPECT_EQ(events.ldsIntervals(event).getTotalBytes(), dual_access ? 8 : 4);
    EXPECT_TRUE(events.ldsIntervals(event).contains(64));
    EXPECT_EQ(events.ldsIntervals(event).contains(96), dual_access);
    EXPECT_EQ(events.pendingCounterIncrement(event, amdgpu::WaitCounterType::LOADCNT), 1);
    EXPECT_EQ(events.pendingCounterIncrement(event, amdgpu::WaitCounterType::DSCNT), 1);
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

TEST(RaceDetector, PositiveOutOfBoundsLdsKeepsDependencies) {
  for (uint32_t capacity : {32u * 1024, 64u * 1024}) {
    std::vector<RaceViolation> violations;
    RaceDetector detector(
        2, Dim3d(0), [&](RaceViolation v) { violations.push_back(v); },
        counterCapacitiesForArch(ROCJITSU_CODE_ARCH_CDNA5), capacity);
    auto &wave = detector.getWaveRaceState(0);
    const std::array<uint32_t, 5> addresses{capacity - 4, capacity - 2, capacity, 0x100000,
                                            0x70000000};
    wave.registerLdsEvent(0, MemoryEventType::LDS_TO_VGPR, 0x1f, 5, addresses, 4,
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
  wave.registerLdsEvent(0, MemoryEventType::LDS_TO_VGPR, 7, 3, addresses, /*bytesPerLane=*/16,
                        kMixedOrderGenericFlatLoadObligations, MemoryOrderClass::UNORDERED,
                        /*ldsLaneMask=*/7, /*elementBytes=*/4);
  const auto &events = detector.events();
  EXPECT_EQ(events.ldsIntervals(EventId{0}).getTotalBytes(), 8);
  EXPECT_TRUE(events.ldsIntervals(EventId{0}).contains(capacity - 1));
  EXPECT_FALSE(events.ldsIntervals(EventId{0}).contains(capacity));
  EXPECT_EQ(events.pendingCounterIncrement(EventId{0}, amdgpu::WaitCounterType::LOADCNT), 1);
  EXPECT_EQ(events.pendingCounterIncrement(EventId{0}, amdgpu::WaitCounterType::DSCNT), 1);
}

TEST(RaceDetector, LdsCounterAccountingIgnoresNegativeIntervals) {
  for (const auto type : {MemoryEventType::LDS_TO_VGPR, MemoryEventType::VGPR_TO_LDS}) {
    SCOPED_TRACE(static_cast<int>(type));
    std::vector<RaceViolation> violations;
    RaceDetector detector(
        2, Dim3d(0), [&](RaceViolation v) { violations.push_back(v); },
        counterCapacitiesForArch(ROCJITSU_CODE_ARCH_CDNA5), /*ldsSizeBytes=*/64 * 1024);
    IntervalSet intervals;
    intervals.append(std::numeric_limits<int>::min() + 64, std::numeric_limits<int>::min() + 68);
    intervals.append(64, 68);
    const auto event =
        detector.allocateEventId(WaveId{0}, 0, type, std::move(intervals),
                                 amdgpu::WaitCounterType::DSCNT, MemoryOrderClass::LDS);
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
  b.flatLdsLoad(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);

  b.checkLdsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);

  EXPECT_TRUE(b.hasLdsRace(0));
}

TEST(RaceDetector, LdsReadDoesNotOrderBeforeSameWaveFlatLdsStore) {
  RaceTestBuilder b(/*numWaves=*/1);
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);

  b.flatLdsStore(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);

  EXPECT_TRUE(b.hasLdsRace(0));
}

TEST(RaceDetector, FlatLdsStoreDoesNotUseSameWaveLdsOrdering) {
  RaceTestBuilder b(/*numWaves=*/1);
  b.flatLdsStore(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);

  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);

  EXPECT_TRUE(b.hasLdsRace(0));
}

TEST(RaceDetector, LdsStoreDoesNotOrderBeforeSameWaveFlatLdsLoad) {
  RaceTestBuilder b(/*numWaves=*/1);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);

  b.flatLdsLoad(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);

  EXPECT_TRUE(b.hasLdsRace(0));
}

TEST(RaceDetector, CompletedFlatLdsStoreIsSafeForSameWaveLdsRead) {
  RaceTestBuilder b(/*numWaves=*/1);
  b.flatLdsStore(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/0, /*lgkmcnt=*/0);

  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);

  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, FlatLoadCombinedZeroWaitClearsBothCounters) {
  RaceTestBuilder b(/*numWaves=*/1);
  const EventId load = b.flatGlobalLoad(/*wave=*/0, kCdna1To4GenericFlatLoadObligations);

  b.waitcnt(/*wave=*/0, /*vmcnt=*/0, /*lgkmcnt=*/0);
  EXPECT_FALSE(b.isPending(load));
}

// ---- Scalar-memory events (lgkmcnt) ----

TEST(RaceDetector, Sgpr_PartialWaitcntDoesNotSelectOutOfOrderResult) {
  RaceTestBuilder b(/*numWaves=*/1);
  const EventId first = b.scalarLoad(/*wave=*/0);
  const EventId second = b.scalarLoad(/*wave=*/0);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/1);

  // Scalar-memory loads can complete out of order, so a nonzero wait does not
  // identify either load as complete.
  EXPECT_TRUE(b.isPending(first));
  EXPECT_TRUE(b.isPending(second));

  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0);
  EXPECT_FALSE(b.isPending(first));
  EXPECT_FALSE(b.isPending(second));
}

TEST(RaceDetector, SgprAndLds_PartialCombinedWaitRetiresOnlyProvablyCompletedDs) {
  RaceTestBuilder b(/*numWaves=*/1);
  const EventId scalar = b.scalarLoad(/*wave=*/0);
  const EventId older = b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  const EventId younger = b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/4, /*bytes=*/4);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/1);

  EXPECT_FALSE(b.isPending(older));
  EXPECT_TRUE(b.isPending(younger));
  EXPECT_TRUE(b.isPending(scalar));
}

TEST(RaceDetector, SgprAndLds_PartialCombinedWaitDoesNotOverRetireDsAtBoundary) {
  // lgkmcnt(2) permits both DS reads to remain after the scalar load completes,
  // so neither ordered DS read is provably complete yet.
  RaceTestBuilder b(/*numWaves=*/1);
  const EventId scalar = b.scalarLoad(/*wave=*/0);
  const EventId older = b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  const EventId younger = b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/4, /*bytes=*/4);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/2);

  EXPECT_TRUE(b.isPending(older));
  EXPECT_TRUE(b.isPending(younger));
  EXPECT_TRUE(b.isPending(scalar));
}

TEST(RaceDetector, SgprAndLds_PartialCombinedWaitUsesDsOrderAcrossInterleaving) {
  RaceTestBuilder b(/*numWaves=*/1);
  const EventId older = b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  const EventId scalar = b.scalarLoad(/*wave=*/0);
  const EventId younger = b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/4, /*bytes=*/4);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/1);

  EXPECT_FALSE(b.isPending(older));
  EXPECT_TRUE(b.isPending(younger));
  EXPECT_TRUE(b.isPending(scalar));
}

TEST(RaceDetector, SgprAndLds_PartialCombinedWaitOrdersNativeDsOperations) {
  RaceTestBuilder b(/*numWaves=*/2);
  b.scalarLoad(/*wave=*/0);
  const EventId read = b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/4, /*bytes=*/4);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/1);

  // Native LDS reads and writes share one FIFO completion class. Even with an
  // unordered scalar-memory event on the counter, at most one outstanding
  // event means that the older DS read has completed.
  EXPECT_FALSE(b.isPending(read));
  b.barrier();
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/4, /*bytes=*/4);
  EXPECT_TRUE(b.hasLdsRace(4));
}

TEST(RaceDetector, SgprAndLds_PartialCombinedWaitMakesOldestDsWriteBarrierEligible) {
  RaceTestBuilder b(/*numWaves=*/2);
  const EventId scalar = b.scalarLoad(/*wave=*/0);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/4, /*bytes=*/4);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/1);
  b.barrier();

  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  EXPECT_FALSE(b.hasLdsRace(0));
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/4, /*bytes=*/4);
  EXPECT_TRUE(b.hasLdsRace(4));
  EXPECT_TRUE(b.isPending(scalar));
}

TEST(RaceDetector, Sgpr_KmcntRetiresOnlyScalarCounterDomain) {
  RaceTestBuilder b(/*numWaves=*/1);
  const EventId scalar =
      b.scalarLoad(/*wave=*/0, /*dwords=*/1, rocjitsu::amdgpu::WaitCounterType::KMCNT);
  const EventId lds = b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4,
                                rocjitsu::amdgpu::WaitCounterType::DSCNT);

  b.waitKmcnt(/*wave=*/0, /*kmcnt=*/0);
  EXPECT_FALSE(b.isPending(scalar));
  EXPECT_TRUE(b.isPending(lds));
}

TEST(RaceDetector, Sgpr_PartialKmcntKeepsUnorderedScalarLoad) {
  RaceTestBuilder b(/*numWaves=*/1);
  const EventId load =
      b.scalarLoad(/*wave=*/0, /*dwords=*/1, rocjitsu::amdgpu::WaitCounterType::KMCNT);
  b.scalarStore(/*wave=*/0, rocjitsu::amdgpu::WaitCounterType::KMCNT);

  b.waitKmcnt(/*wave=*/0, /*kmcnt=*/1);
  EXPECT_TRUE(b.isPending(load));

  b.waitKmcnt(/*wave=*/0, /*kmcnt=*/0);
  EXPECT_FALSE(b.isPending(load));
}

// ---- LDS cross-wave races ----

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

// ---- LDS same-wave ordering ----

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
      // Waits complete the wave's own accesses; a barrier is still needed
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
    // both chunks.
    if (writes)
      b.flatLdsStore(/*wave=*/0, /*lane=*/0, /*addr=*/14, /*bytes=*/4);
    else
      b.flatLdsLoad(/*wave=*/0, /*lane=*/0, /*addr=*/14, /*bytes=*/4);
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

// ---- Same-wave LDS event lifetime ----

TEST(RaceDetector, SameWave_PartialLgkmcntRetiresOlderLdsRead) {
  RaceTestBuilder b(/*numWaves=*/1);
  const EventId older = b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  const EventId younger = b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/64, /*bytes=*/4);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/1); // drain oldest
  EXPECT_FALSE(b.isPending(older));
  EXPECT_TRUE(b.isPending(younger));
}

TEST(RaceDetector, SameWave_BarrierNoWaitcnt) {
  // Barrier without waitcnt doesn't complete same-wave events.
  RaceTestBuilder b(/*numWaves=*/1);
  const EventId read = b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.barrier(); // no waitcnt first!
  EXPECT_TRUE(b.isPending(read));
}

TEST(RaceDetector, SameWave_WaitcntBarrierOk) {
  // Waitcnt then barrier → the completed LDS read retires.
  RaceTestBuilder b(/*numWaves=*/1);
  const EventId read = b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0);
  EXPECT_EQ(b.events().status(read), EventStatus::WAVE_COMPLETE);
  b.barrier();
  EXPECT_EQ(b.events().status(read), EventStatus::RETIRED);
}

// ---- Deep event stack ----

TEST(RaceDetector, DeepStack_PartialWaitcnt) {
  // 3 loads, waitcnt(1) drains the two oldest, the newest is still pending.
  RaceTestBuilder b(/*numWaves=*/1);
  const EventId oldest = b.globalLoad(/*wave=*/0);
  const EventId middle = b.globalLoad(/*wave=*/0);
  const EventId newest = b.globalLoad(/*wave=*/0);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/1);
  EXPECT_FALSE(b.isPending(oldest));
  EXPECT_FALSE(b.isPending(middle));
  EXPECT_TRUE(b.isPending(newest));
}

TEST(RaceDetector, DeepStack_FullWaitcnt) {
  RaceTestBuilder b(/*numWaves=*/1);
  const std::array loads{b.globalLoad(/*wave=*/0), b.globalLoad(/*wave=*/0),
                         b.globalLoad(/*wave=*/0)};
  b.waitcnt(/*wave=*/0, /*vmcnt=*/0);
  for (const EventId load : loads)
    EXPECT_FALSE(b.isPending(load));
}

TEST(RaceDetector, DeepStack_ConsecutiveWaitcnts) {
  // Two partial waitcnts in sequence. The second drain must retire the
  // correct (oldest) event — not a newer one that got reordered.
  //   load 0, load 1, load 2  (3 outstanding, oldest first)
  //   waitcnt vmcnt(2)        → drain oldest (0), keep 1 and 2
  //   load 3                  (1, 2, 3 outstanding)
  //   waitcnt vmcnt(2)        → drain oldest (1), keep 2 and 3
  RaceTestBuilder b(/*numWaves=*/1);
  const EventId load0 = b.globalLoad(/*wave=*/0);
  const EventId load1 = b.globalLoad(/*wave=*/0);
  const EventId load2 = b.globalLoad(/*wave=*/0);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/2);
  const EventId load3 = b.globalLoad(/*wave=*/0);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/2);
  EXPECT_FALSE(b.isPending(load0));
  EXPECT_FALSE(b.isPending(load1));
  EXPECT_TRUE(b.isPending(load2));
  EXPECT_TRUE(b.isPending(load3));
}

// ---- Direct-to-LDS (GlobalToLds) ----

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

// ---- Exec mask ----

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

// ---- Multi-workgroup ----

TEST(RaceDetector, MultiWorkgroup_ReportsIndex) {
  // Race violation reports correct workgroup index.
  RaceTestBuilder b(/*numWaves=*/2, /*waveSize=*/64, Dim3d(7, 3, 1));
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.checkLdsRead(/*wave=*/1, /*lane=*/5, /*addr=*/0, /*bytes=*/4);
  ASSERT_EQ(b.raceCount(), 1);
  auto &v = b.violations()[0];
  EXPECT_EQ(v.space, RaceViolation::Space::LDS);
  EXPECT_EQ(v.index, 0);
  EXPECT_EQ(v.wave, 1);
  EXPECT_EQ(v.lane, 5);
  EXPECT_EQ(v.workgroupId.x, 7);
  EXPECT_EQ(v.workgroupId.y, 3);
  EXPECT_EQ(v.workgroupId.z, 1);
}

// ---- LDS additional patterns ----

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

// ---- Cross-wave exec mask tests ----
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

// ---- Global stores ----

TEST(RaceDetector, GlobalStore_CountedByVmcnt) {
  // Global store is tracked by vmcnt. Issuing a store then a global load
  // means the load is the 2nd vmcnt event. waitcnt(1) should NOT drain it.
  // This test verifies that global stores consume vmcnt slots.
  RaceTestBuilder b(/*numWaves=*/1);
  const EventId store = b.globalStore(/*wave=*/0); // vmcnt event 0 (oldest)
  const EventId load = b.globalLoad(/*wave=*/0);   // vmcnt event 1
  b.waitcnt(/*wave=*/0, /*vmcnt=*/1);              // drain oldest (store), load stays
  EXPECT_FALSE(b.isPending(store));
  EXPECT_TRUE(b.isPending(load));
}

// ---- DTL multi-lane ----

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

// ---- Dual-offset LDS ----

TEST(RaceDetector, DualOffset_CrossWaveRace) {
  // registerDualOffsetLdsEvent: each lane writes TWO 8-byte intervals.
  // Lane 0 base=100: [100, 108) (offset0=0) and [116, 124) (offset1=2).
  // Another wave reading either interval without a barrier → RACE.
  //
  // Uses RaceDetector/WaveRaceState directly since the builder doesn't
  // expose registerDualOffsetLdsEvent.
  std::vector<RaceViolation> violations;
  RaceDetector detector(
      /*nWaves=*/
      2, Dim3d(0), [&](RaceViolation v) { violations.push_back(v); },
      counterCapacitiesForArch(ROCJITSU_CODE_ARCH_CDNA4), /*ldsSizeBytes=*/64 * 1024);
  auto &rs = detector.getWaveRaceState(0);
  std::vector<uint32_t> ldsAddrs(64, 0);
  ldsAddrs[0] = 100;
  rs.registerDualOffsetLdsEvent(/*pc=*/0, MemoryEventType::VGPR_TO_LDS,
                                /*execMask=*/1, /*waveSize=*/64, ldsAddrs,
                                /*offset0=*/0, /*offset1=*/2);
  // Lane 0 wrote: [100, 108) and [116, 124)
  detector.validateRead(/*addr=*/100, WaveId{1}, /*lane=*/0, /*nBytes=*/4);
  EXPECT_EQ(violations.size(), 1u); // first interval
  detector.validateRead(/*addr=*/116, WaveId{1}, /*lane=*/0, /*nBytes=*/4);
  EXPECT_EQ(violations.size(), 2u); // second interval
  // Outside both intervals: no race
  detector.validateRead(/*addr=*/108, WaveId{1}, /*lane=*/0, /*nBytes=*/4);
  EXPECT_EQ(violations.size(), 2u); // still 2
}

// ---- Mixed counter types ----

TEST(RaceDetector, Mixed_VmcntAndLgkmcnt) {
  // Global load (vmcnt) and LDS load (lgkmcnt) in flight simultaneously.
  // waitcnt vmcnt(0) clears global load but not LDS load.
  RaceTestBuilder b(/*numWaves=*/1);
  const EventId global = b.globalLoad(/*wave=*/0);
  const EventId lds = b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/0); // clear global load only
  EXPECT_FALSE(b.isPending(global));
  EXPECT_TRUE(b.isPending(lds));
}

TEST(RaceDetector, Mixed_LgkmcntAndVmcnt) {
  // Reverse: waitcnt lgkmcnt(0) clears LDS load but not global load.
  RaceTestBuilder b(/*numWaves=*/1);
  const EventId global = b.globalLoad(/*wave=*/0);
  const EventId lds = b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0); // clear LDS load only
  EXPECT_FALSE(b.isPending(lds));
  EXPECT_TRUE(b.isPending(global));
}

TEST(RaceDetector, Mixed_BothCleared) {
  // Both vmcnt(0) and lgkmcnt(0) → every event completes.
  RaceTestBuilder b(/*numWaves=*/1);
  const std::array events{b.globalLoad(/*wave=*/0),
                          b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4),
                          b.scalarLoad(/*wave=*/0)};
  b.waitcnt(/*wave=*/0, /*vmcnt=*/0, /*lgkmcnt=*/0);
  for (const EventId event : events)
    EXPECT_FALSE(b.isPending(event));
}

TEST(RaceDetector, EventRegistry_NarrowWindow) {
  // Circular buffer: N loads in flight. Each iteration drains the oldest,
  // checks that it completed while the others remain pending, then issues a
  // replacement load. The registry trims completed loads along the way.
  constexpr int N = 10;
  RaceTestBuilder b(/*numWaves=*/1);

  std::array<EventId, N> loads;
  for (int i = 0; i < N; ++i) {
    loads[i] = b.globalLoad(/*wave=*/0);
  }

  for (int iter = 0; iter < 1000; ++iter) {
    int drained = iter % N;
    int nextOldest = (iter + 1) % N;
    int newest = (iter + N - 1) % N;

    b.waitcnt(/*wave=*/0, /*vmcnt=*/N - 1);

    EXPECT_FALSE(b.isPending(loads[drained])) << "iter=" << iter;
    EXPECT_TRUE(b.isPending(loads[nextOldest])) << "iter=" << iter;
    EXPECT_TRUE(b.isPending(loads[newest])) << "iter=" << iter;

    loads[drained] = b.globalLoad(/*wave=*/0);
  }
  EXPECT_GT(b.events().trimmedCount(), 0);
}

// ---- EventRegistry trimming ----

TEST(RaceDetector, EventRegistry_TrimWithoutBarrier) {
  // Single wave, no LDS, no barrier. WAVE_COMPLETE non-LDS events are
  // trimmable. Trimming is triggered automatically inside add() every
  // kTrimAttemptInterval allocations.
  int N = EventRegistry::kTrimAttemptInterval - 1;
  RaceTestBuilder b(/*numWaves=*/1);

  // Allocate N events (one short of the trim interval) — no trim yet.
  for (int i = 0; i < N; ++i) {
    b.globalLoad(/*wave=*/0);
    b.waitcnt(/*wave=*/0, /*vmcnt=*/0);
  }
  EXPECT_EQ(b.events().totalAllocated(), N);
  EXPECT_EQ(b.events().trimmedCount(), 0);

  // One more allocation triggers trimming.
  b.globalLoad(/*wave=*/0);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/0);

  EXPECT_EQ(b.events().totalAllocated(), N + 1);
  EXPECT_GT(b.events().trimmedCount(), 0);
  EXPECT_LT(b.events().size(), N);

  // Counter tracking still works after trimming.
  const EventId load = b.globalLoad(/*wave=*/0);
  EXPECT_TRUE(b.isPending(load));
  b.waitcnt(/*wave=*/0, /*vmcnt=*/0);
  EXPECT_FALSE(b.isPending(load));
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

// ---- Handler invocation count ----
//
// The race handler fires for every (LDS range, lane) check that overlaps a
// conflicting event. When the plugin wraps this with deduplication (by PC), it
// must handle millions of handler calls efficiently. These tests verify the
// raw handler invocation count at the core level.

TEST(RaceDetector, HandlerFiresPerLane) {
  RaceTestBuilder b(/*numWaves=*/2);

  // One Direct-to-LDS load with all 64 lanes active, each writing its own dword.
  std::vector<uint32_t> ldsAddrs(64);
  for (int lane = 0; lane < 64; ++lane)
    ldsAddrs[lane] = lane * 4;
  b.globalToLds(/*wave=*/0, ldsAddrs, /*bytesPerLane=*/4);

  // Another wave reads every lane's dword without a barrier — each lane fires
  // the handler.
  for (int lane = 0; lane < 64; ++lane)
    b.checkLdsRead(/*wave=*/1, lane, /*addr=*/lane * 4, /*bytes=*/4);

  EXPECT_EQ(b.raceCount(), 64);
}

TEST(RaceDetector, MultipleDistinctRaces) {
  RaceTestBuilder b(/*numWaves=*/2);

  // Two independent LDS writes at different PCs.
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/16, /*bytes=*/4);

  // Another wave reads both without a barrier — two distinct races.
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/16, /*bytes=*/4);

  EXPECT_EQ(b.raceCount(), 2);
  EXPECT_TRUE(b.hasLdsRace(0));
  EXPECT_TRUE(b.hasLdsRace(16));
}

TEST(RaceDetector, RepeatedCheckSameRace) {
  RaceTestBuilder b(/*numWaves=*/2);

  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);

  // Check the same LDS range and lane 100 times. The handler fires every time
  // because the core race detector does not deduplicate — that is the
  // plugin's responsibility.
  for (int i = 0; i < 100; ++i)
    b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/0, /*bytes=*/4);

  EXPECT_EQ(b.raceCount(), 100);
}
