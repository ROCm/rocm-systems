// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Unit tests for the race detection core (RaceDetector + WaveRaceState).
//
// These tests drive the detection API directly via RaceTestBuilder — no
// assembly parser, no instruction emulator, no GPU.
//
// Test categories:
//   Vgpr_*          — VGPR races from global loads (vmcnt)
//   Sgpr_*          — SGPR races from scalar loads (lgkmcnt)
//   SgprWaw_*       — writes overlapping pending SGPR destinations
//   TtmpWaw_*       — writes overlapping pending TTMP destinations
//   LdsCrossWave_*  — cross-wave LDS races (missing barrier)
//   LdsSameWave_*   — same-wave LDS instruction ordering
//   SameWave_*      — same-wave VGPR races via LDS loads
//   DeepStack_*     — multiple outstanding loads with partial waitcnt
//   D16_*           — byte-level VGPR tracking (half-register loads)
//   Dtl_*           — Direct-to-LDS global loads
//   Exec_*          — exec mask effects on race detection
//   MultiWorkgroup_* — workgroup ID in violation reports
//   GlobalStore_*   — global stores and vmcnt interaction
//   MultiReg_*      — multi-register loads
//   Mixed_*         — simultaneous vmcnt and lgkmcnt events
//   DualAddress_*   — LDS events with two address sets per lane
//   TensorDma_*     — tensor load/store LDS races and tensorcnt waits
//   SplitBarrier_*  — ordering at each wave's barrier signal and wait

#include "race_test_builder.h"
#include "race_test_instruction.h"
#include "rocjitsu/isa/arch/amdgpu/shared/memory_issue.h"
#include "rocjitsu/vm/plugins/race_detector/core/common_register.h"
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

constexpr std::array kCdna1To4GenericFlatLoadObligations{
    amdgpu::MemoryCounterObligation{amdgpu::WaitCounterType::VMCNT, MemoryOrderClass::UNORDERED},
    amdgpu::MemoryCounterObligation{amdgpu::WaitCounterType::LGKMCNT, MemoryOrderClass::UNORDERED}};

constexpr std::array kMixedOrderGenericFlatLoadObligations{
    amdgpu::MemoryCounterObligation{amdgpu::WaitCounterType::LOADCNT, MemoryOrderClass::VMEM},
    amdgpu::MemoryCounterObligation{amdgpu::WaitCounterType::DSCNT, MemoryOrderClass::LDS}};

} // namespace

TEST(RaceDetectorDefaults, CoversEveryMemoryEventType) {
  struct TestCase {
    MemoryEventType type;
    amdgpu::WaitCounterType counter;
    MemoryOrderClass order;
  };
  constexpr TestCase cases[] = {
      {MemoryEventType::GLOBAL_TO_VGPR, amdgpu::WaitCounterType::VMCNT, MemoryOrderClass::VMEM},
      {MemoryEventType::VGPR_TO_GLOBAL, amdgpu::WaitCounterType::VMCNT, MemoryOrderClass::VMEM},
      {MemoryEventType::GLOBAL_TO_LDS, amdgpu::WaitCounterType::VMCNT, MemoryOrderClass::VMEM},
      {MemoryEventType::LDS_TO_VGPR, amdgpu::WaitCounterType::LGKMCNT, MemoryOrderClass::LDS},
      {MemoryEventType::VGPR_TO_LDS, amdgpu::WaitCounterType::LGKMCNT, MemoryOrderClass::LDS},
      {MemoryEventType::GLOBAL_TO_SGPR, amdgpu::WaitCounterType::LGKMCNT,
       MemoryOrderClass::UNORDERED},
      {MemoryEventType::GLOBAL_TO_TTMP, amdgpu::WaitCounterType::LGKMCNT,
       MemoryOrderClass::UNORDERED},
      {MemoryEventType::SCALAR_TO_GLOBAL, amdgpu::WaitCounterType::LGKMCNT,
       MemoryOrderClass::UNORDERED},
      {MemoryEventType::LDS_TO_GLOBAL, amdgpu::WaitCounterType::TENSORCNT,
       MemoryOrderClass::TENSOR},
  };
  RaceDetector detector(
      1, 8, 8, Dim3d(0), [](RaceViolation) {}, counterCapacitiesForArch(ROCJITSU_CODE_ARCH_CDNA4),
      /*ldsSizeBytes=*/1024);
  for (const TestCase &test : cases) {
    SCOPED_TRACE(static_cast<int>(test.type));
    const EventId event = detector.allocateEventId(WaveId{0}, testInstruction(0), test.type,
                                                   /*registers=*/{}, /*execMask=*/1);
    const auto &events = detector.events();
    EXPECT_TRUE(events.hasPendingWaitCounter(event, test.counter));
    EXPECT_EQ(events.pendingCompletionClass(event, test.counter), test.order);
    EXPECT_EQ(events.pendingCounterIncrement(event, test.counter), 1);
    EXPECT_EQ(events.memoryOrder(event), test.order);
    EXPECT_FALSE(events.allWaitCountersSatisfied(event));
    EXPECT_TRUE(detector.satisfyEventWaitCounter(event, test.counter));
    EXPECT_FALSE(events.hasPendingWaitCounter(event, test.counter));
  }
}

// ---- VGPR races (vmcnt) ----

TEST(RaceDetector, Vgpr_InsufficientVmcnt) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/1, /*numRegs=*/1);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/2, /*numRegs=*/1);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/1);                 // drains oldest (v1), v2 still in flight
  b.checkVgprRead(/*wave=*/0, /*reg=*/2, /*lane=*/0); // read v2 → RACE
  EXPECT_TRUE(b.hasVgprRace(2));
}

TEST(RaceDetector, Vgpr_ViolationFields) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/3, /*numRegs=*/1);
  b.checkVgprRead(/*wave=*/0, /*reg=*/3, /*lane=*/5);
  ASSERT_EQ(b.raceCount(), 1);
  auto &v = b.violations()[0];
  EXPECT_EQ(v.space, RaceViolation::Space::VGPR);
  EXPECT_EQ(v.index, 3);
  EXPECT_EQ(v.wave, 0);
  EXPECT_EQ(v.lane, 5);
}

TEST(RaceDetector, Vgpr_WaitcntClearsRace) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/1, /*numRegs=*/1);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/0);
  b.checkVgprRead(/*wave=*/0, /*reg=*/1, /*lane=*/0);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, FlatGlobalLoadRequiresBothCounterWaits) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.flatGlobalLoad(/*wave=*/0, /*vgprBase=*/1, /*numRegs=*/1, kCdna1To4GenericFlatLoadObligations);

  b.waitcnt(/*wave=*/0, /*vmcnt=*/0);
  b.checkVgprRead(/*wave=*/0, /*reg=*/1, /*lane=*/0);
  EXPECT_TRUE(b.hasVgprRace(1));

  b.clearViolations();
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0);
  b.checkVgprRead(/*wave=*/0, /*reg=*/1, /*lane=*/0);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, FlatLdsLoadRequiresBothCounterWaits) {
  for (bool vmcnt_first : {false, true}) {
    SCOPED_TRACE(vmcnt_first ? "vmcnt first" : "lgkmcnt first");
    RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
    b.flatLdsLoad(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4, /*vgprDst=*/1);

    b.waitcnt(/*wave=*/0, /*vmcnt=*/vmcnt_first ? 0 : -1, /*lgkmcnt=*/vmcnt_first ? -1 : 0);
    b.checkVgprRead(/*wave=*/0, /*reg=*/1, /*lane=*/0);
    EXPECT_TRUE(b.hasVgprRace(1));

    b.clearViolations();
    b.checkLdsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
    EXPECT_TRUE(b.hasLdsRace(0));

    b.clearViolations();
    b.waitcnt(/*wave=*/0, /*vmcnt=*/vmcnt_first ? -1 : 0, /*lgkmcnt=*/vmcnt_first ? 0 : -1);
    b.checkVgprRead(/*wave=*/0, /*reg=*/1, /*lane=*/0);
    b.checkLdsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
    EXPECT_FALSE(b.hasRace());
  }
}

TEST(RaceDetector, FlatLdsStoreRequiresBothCounterWaits) {
  for (bool vmcnt_first : {false, true}) {
    SCOPED_TRACE(vmcnt_first ? "vmcnt first" : "lgkmcnt first");
    RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
    b.flatLdsStore(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);

    b.waitcnt(/*wave=*/0, /*vmcnt=*/vmcnt_first ? 0 : -1, /*lgkmcnt=*/vmcnt_first ? -1 : 0);
    b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
    EXPECT_TRUE(b.hasLdsRace(0));

    b.clearViolations();
    b.waitcnt(/*wave=*/0, /*vmcnt=*/vmcnt_first ? -1 : 0, /*lgkmcnt=*/vmcnt_first ? 0 : -1);
    b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
    EXPECT_FALSE(b.hasRace());
  }
}

TEST(RaceDetector, MixedFlatKeepsOneCounterEntryAndAllRegisterLanes) {
  for (uint64_t lds_mask : {0u, 1u, 2u, 5u}) {
    SCOPED_TRACE(lds_mask);
    std::vector<RaceViolation> violations;
    RaceDetector detector(
        1, 8, 8, Dim3d(0), [&](RaceViolation v) { violations.push_back(v); },
        counterCapacitiesForArch(ROCJITSU_CODE_ARCH_RDNA4), /*ldsSizeBytes=*/64 * 1024);
    auto &wave = detector.getWaveRaceState(0);
    const std::array<uint32_t, 32> addresses{64, 128, 256};
    for (uint32_t reg : {1u, 2u})
      wave.registerLdsEvent(
          testInstruction(reg), MemoryEventType::LDS_TO_VGPR, {reg}, /*execMask=*/3,
          /*waveSize=*/32, addresses, /*bytesPerLane=*/4, 0xf,
          kMixedOrderGenericFlatLoadObligations, MemoryOrderClass::UNORDERED, lds_mask);
    const auto &events = detector.events();
    ASSERT_EQ(events.totalAllocated(), 2);
    for (const EventId event : {EventId{0}, EventId{1}}) {
      EXPECT_EQ(events.execMask(event), 3u);
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
    for (int lane : {0, 1}) {
      wave.checkVgprReadLanes(1, uint64_t{1} << lane, 0xf);
      EXPECT_TRUE(violations.empty());
    }
    for (int lane : {0, 1}) {
      wave.checkVgprReadLanes(2, uint64_t{1} << lane, 0xf);
      ASSERT_EQ(violations.size(), 1u);
      EXPECT_EQ(violations.front().lane, lane);
      violations.clear();
    }
    wait.updates.clear();
    wait.add(amdgpu::WaitCounterType::LOADCNT, 0);
    wait.add(amdgpu::WaitCounterType::DSCNT, 0);
    wave.dispatch(wait);
    wave.checkVgprReadLanes(2, 3, 0xf);
    EXPECT_TRUE(violations.empty());
  }
}

TEST(RaceDetector, LdsValidationRejectsNegativeAndOverflowingRanges) {
  std::vector<RaceViolation> violations;
  RaceDetector detector(
      2, 8, 8, Dim3d(0), [&](RaceViolation v) { violations.push_back(v); },
      counterCapacitiesForArch(ROCJITSU_CODE_ARCH_CDNA5), /*ldsSizeBytes=*/64 * 1024);
  for (const auto type : {MemoryEventType::LDS_TO_VGPR, MemoryEventType::VGPR_TO_LDS}) {
    IntervalSet intervals;
    intervals.append(64, 68);
    detector.allocateEventId(
        WaveId{0}, testInstruction(0), type, {}, 1, 0xf, std::move(intervals),
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

TEST(RaceDetector, InvalidLdsAddressesKeepRegisterAndCounterDependencies) {
  for (bool dual_access : {false, true}) {
    SCOPED_TRACE(dual_access);
    std::vector<RaceViolation> violations;
    RaceDetector detector(
        1, 8, 8, Dim3d(0), [&](RaceViolation v) { violations.push_back(v); },
        counterCapacitiesForArch(ROCJITSU_CODE_ARCH_CDNA5), /*ldsSizeBytes=*/64 * 1024);
    auto &wave = detector.getWaveRaceState(0);
    const std::array<uint32_t, 4> addresses{0x80000040u, UINT32_MAX, 0x7ffffffeu, 64};
    const std::array<uint32_t, 4> second_addresses{0x80000080u, UINT32_MAX, 0x7fffffffu, 96};
    if (dual_access)
      wave.registerLdsEvent(testInstruction(0), MemoryEventType::LDS_TO_VGPR, {1}, 0xf, 4,
                            addresses, second_addresses, 4, 0xf,
                            kMixedOrderGenericFlatLoadObligations, MemoryOrderClass::UNORDERED);
    else
      wave.registerLdsEvent(testInstruction(0), MemoryEventType::LDS_TO_VGPR, {1}, 0xf, 4,
                            addresses, 4, 0xf, kMixedOrderGenericFlatLoadObligations,
                            MemoryOrderClass::UNORDERED);
    const auto &events = detector.events();
    const EventId event{0};
    ASSERT_EQ(events.totalAllocated(), 1);
    EXPECT_EQ(events.execMask(event), 0xfu);
    EXPECT_EQ(events.ldsIntervals(event).getTotalBytes(), dual_access ? 8 : 4);
    EXPECT_TRUE(events.ldsIntervals(event).contains(64));
    EXPECT_EQ(events.ldsIntervals(event).contains(96), dual_access);
    EXPECT_EQ(events.pendingCounterIncrement(event, amdgpu::WaitCounterType::LOADCNT), 1);
    EXPECT_EQ(events.pendingCounterIncrement(event, amdgpu::WaitCounterType::DSCNT), 1);
    for (int lane = 0; lane < 4; ++lane)
      wave.checkVgprReadLanes(1, uint64_t{1} << lane, 0xf);
    EXPECT_EQ(violations.size(), 4u);
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
    wave.checkVgprReadLanes(1, 0xf, 0xf);
    EXPECT_TRUE(violations.empty());
  }
}

TEST(RaceDetector, PositiveOutOfBoundsLdsKeepsDependencies) {
  for (uint32_t capacity : {32u * 1024, 64u * 1024}) {
    std::vector<RaceViolation> violations;
    RaceDetector detector(
        2, 8, 8, Dim3d(0), [&](RaceViolation v) { violations.push_back(v); },
        counterCapacitiesForArch(ROCJITSU_CODE_ARCH_CDNA5), capacity);
    auto &wave = detector.getWaveRaceState(0);
    const std::array<uint32_t, 5> addresses{capacity - 4, capacity - 2, capacity, 0x100000,
                                            0x70000000};
    wave.registerLdsEvent(testInstruction(0), MemoryEventType::LDS_TO_VGPR, {1}, 0x1f, 5, addresses,
                          4, 0xf, kMixedOrderGenericFlatLoadObligations,
                          MemoryOrderClass::UNORDERED);
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
    EXPECT_EQ(events.execMask(event), 0x1fu);
    EXPECT_EQ(events.registers(event).size(), 1u);
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
      1, 8, 8, Dim3d(0), [](RaceViolation) {}, counterCapacitiesForArch(ROCJITSU_CODE_ARCH_CDNA5),
      capacity);
  auto &wave = detector.getWaveRaceState(0);
  const std::array<uint32_t, 3> addresses{capacity - 8, capacity - 2, capacity};
  wave.registerLdsEvent(testInstruction(0), MemoryEventType::LDS_TO_VGPR, {1, 2, 3, 4}, 7, 3,
                        addresses,
                        /*bytesPerLane=*/16, 0xf, kMixedOrderGenericFlatLoadObligations,
                        MemoryOrderClass::UNORDERED, /*ldsLaneMask=*/7, /*elementBytes=*/4);
  const auto &events = detector.events();
  EXPECT_EQ(events.ldsIntervals(EventId{0}).getTotalBytes(), 8);
  EXPECT_TRUE(events.ldsIntervals(EventId{0}).contains(capacity - 1));
  EXPECT_FALSE(events.ldsIntervals(EventId{0}).contains(capacity));
  EXPECT_EQ(events.registers(EventId{0}).size(), 4u);
  EXPECT_EQ(events.execMask(EventId{0}), 7u);
  EXPECT_EQ(events.pendingCounterIncrement(EventId{0}, amdgpu::WaitCounterType::LOADCNT), 1);
  EXPECT_EQ(events.pendingCounterIncrement(EventId{0}, amdgpu::WaitCounterType::DSCNT), 1);
}

TEST(RaceDetector, LdsCounterAccountingIgnoresNegativeIntervals) {
  for (const auto type : {MemoryEventType::LDS_TO_VGPR, MemoryEventType::VGPR_TO_LDS}) {
    SCOPED_TRACE(static_cast<int>(type));
    std::vector<RaceViolation> violations;
    RaceDetector detector(
        2, 8, 8, Dim3d(0), [&](RaceViolation v) { violations.push_back(v); },
        counterCapacitiesForArch(ROCJITSU_CODE_ARCH_CDNA5), /*ldsSizeBytes=*/64 * 1024);
    IntervalSet intervals;
    intervals.append(std::numeric_limits<int>::min() + 64, std::numeric_limits<int>::min() + 68);
    intervals.append(64, 68);
    const auto event = detector.allocateEventId(
        WaveId{0}, testInstruction(0), type, {}, 1, 0xf, std::move(intervals),
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
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.flatLdsLoad(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4, /*vgprDst=*/1);

  b.checkLdsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);

  EXPECT_TRUE(b.hasLdsRace(0));
}

TEST(RaceDetector, LdsReadDoesNotOrderBeforeSameWaveFlatLdsStore) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4, /*vgprDst=*/1);

  b.flatLdsStore(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);

  EXPECT_TRUE(b.hasLdsRace(0));
}

TEST(RaceDetector, FlatLdsStoreDoesNotUseSameWaveLdsOrdering) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.flatLdsStore(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);

  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);

  EXPECT_TRUE(b.hasLdsRace(0));
}

TEST(RaceDetector, LdsStoreDoesNotOrderBeforeSameWaveFlatLdsLoad) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);

  b.flatLdsLoad(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4, /*vgprDst=*/1);

  EXPECT_TRUE(b.hasLdsRace(0));
}

TEST(RaceDetector, CompletedFlatLdsStoreIsSafeForSameWaveLdsRead) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.flatLdsStore(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/0, /*lgkmcnt=*/0);

  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);

  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, FlatLoadCombinedZeroWaitClearsBothCounters) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.flatGlobalLoad(/*wave=*/0, /*vgprBase=*/1, /*numRegs=*/1, kCdna1To4GenericFlatLoadObligations);

  b.waitcnt(/*wave=*/0, /*vmcnt=*/0, /*lgkmcnt=*/0);
  b.checkVgprRead(/*wave=*/0, /*reg=*/1, /*lane=*/0);
  EXPECT_FALSE(b.hasRace());
}

// ---- SGPR races (lgkmcnt) ----

TEST(RaceDetector, Sgpr_MissingWaitcnt) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.scalarLoad(/*wave=*/0, /*sgprBase=*/4, /*numRegs=*/1);
  b.checkSgprRead(/*wave=*/0, /*reg=*/4); // no waitcnt → RACE
  EXPECT_TRUE(b.hasSgprRace(4));
}

TEST(RaceDetector, Sgpr_WithWaitcnt) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.scalarLoad(/*wave=*/0, /*sgprBase=*/4, /*numRegs=*/1);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0);
  b.checkSgprRead(/*wave=*/0, /*reg=*/4);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, Sgpr_PartialWaitcntDoesNotSelectOutOfOrderResult) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.scalarLoad(/*wave=*/0, /*sgprBase=*/4, /*numRegs=*/1);
  b.scalarLoad(/*wave=*/0, /*sgprBase=*/5, /*numRegs=*/1);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/1);

  // Scalar-memory loads can complete out of order, so a nonzero wait does not
  // identify either destination as complete.
  b.checkSgprRead(/*wave=*/0, /*reg=*/4);
  EXPECT_TRUE(b.hasSgprRace(4));
  b.clearViolations();
  b.checkSgprRead(/*wave=*/0, /*reg=*/5);
  EXPECT_TRUE(b.hasSgprRace(5));
}

TEST(RaceDetector, SgprAndLds_PartialCombinedWaitRetiresOnlyProvablyCompletedDs) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.scalarLoad(/*wave=*/0, /*sgprBase=*/4, /*numRegs=*/1);
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4, /*vgprDst=*/2);
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/4, /*bytes=*/4, /*vgprDst=*/3);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/1);

  b.checkVgprRead(/*wave=*/0, /*reg=*/2, /*lane=*/0);
  EXPECT_FALSE(b.hasVgprRace(2));
  b.checkVgprRead(/*wave=*/0, /*reg=*/3, /*lane=*/0);
  EXPECT_TRUE(b.hasVgprRace(3));
  b.clearViolations();
  b.checkSgprRead(/*wave=*/0, /*reg=*/4);
  EXPECT_TRUE(b.hasSgprRace(4));
}

TEST(RaceDetector, SgprAndLds_PartialCombinedWaitDoesNotOverRetireDsAtBoundary) {
  // lgkmcnt(2) permits both DS reads to remain after the scalar load completes,
  // so neither ordered DS destination is provably safe yet.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.scalarLoad(/*wave=*/0, /*sgprBase=*/4, /*numRegs=*/1);
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4, /*vgprDst=*/2);
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/4, /*bytes=*/4, /*vgprDst=*/3);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/2);

  b.checkVgprRead(/*wave=*/0, /*reg=*/2, /*lane=*/0);
  EXPECT_TRUE(b.hasVgprRace(2));
  b.clearViolations();
  b.checkVgprRead(/*wave=*/0, /*reg=*/3, /*lane=*/0);
  EXPECT_TRUE(b.hasVgprRace(3));
  b.clearViolations();
  b.checkSgprRead(/*wave=*/0, /*reg=*/4);
  EXPECT_TRUE(b.hasSgprRace(4));
}

TEST(RaceDetector, SgprAndLds_PartialCombinedWaitUsesDsOrderAcrossInterleaving) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4, /*vgprDst=*/2);
  b.scalarLoad(/*wave=*/0, /*sgprBase=*/4, /*numRegs=*/1);
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/4, /*bytes=*/4, /*vgprDst=*/3);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/1);

  b.checkVgprRead(/*wave=*/0, /*reg=*/2, /*lane=*/0);
  EXPECT_FALSE(b.hasVgprRace(2));
  b.checkVgprRead(/*wave=*/0, /*reg=*/3, /*lane=*/0);
  EXPECT_TRUE(b.hasVgprRace(3));
  b.clearViolations();
  b.checkSgprRead(/*wave=*/0, /*reg=*/4);
  EXPECT_TRUE(b.hasSgprRace(4));
}

TEST(RaceDetector, SgprAndLds_PartialCombinedWaitOrdersNativeDsOperations) {
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/8, /*sgprs=*/8);
  b.scalarLoad(/*wave=*/0, /*sgprBase=*/4, /*numRegs=*/1);
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4, /*vgprDst=*/2);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/4, /*bytes=*/4);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/1);

  // Native LDS reads and writes share one FIFO completion class. Even with an
  // unordered scalar-memory event on the counter, at most one outstanding
  // event means that the older DS read has completed.
  b.checkVgprRead(/*wave=*/0, /*reg=*/2, /*lane=*/0);
  EXPECT_FALSE(b.hasVgprRace(2));
  b.clearViolations();
  b.barrier();
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/4, /*bytes=*/4);
  EXPECT_TRUE(b.hasLdsRace(4));
}

TEST(RaceDetector, SgprAndLds_PartialCombinedWaitMakesOldestDsWriteBarrierEligible) {
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/8, /*sgprs=*/8);
  b.scalarLoad(/*wave=*/0, /*sgprBase=*/4, /*numRegs=*/1);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/4, /*bytes=*/4);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/1);
  b.barrier();

  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  EXPECT_FALSE(b.hasLdsRace(0));
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/4, /*bytes=*/4);
  EXPECT_TRUE(b.hasLdsRace(4));
  b.clearViolations();
  b.checkSgprRead(/*wave=*/0, /*reg=*/4);
  EXPECT_TRUE(b.hasSgprRace(4));
}

TEST(RaceDetector, Ttmp_MissingWaitcnt) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ttmpLoad(/*wave=*/0, /*ttmpBase=*/2, /*numRegs=*/2);
  b.checkTtmpRead(/*wave=*/0, /*reg=*/3);
  EXPECT_TRUE(b.hasTtmpRace(3));
}

TEST(RaceDetector, Ttmp_WithWaitcnt) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ttmpLoad(/*wave=*/0, /*ttmpBase=*/2, /*numRegs=*/2);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0);
  b.checkTtmpRead(/*wave=*/0, /*reg=*/3);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, Ttmp_KmcntRetiresOnlyScalarCounterDomain) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ttmpLoad(/*wave=*/0, /*ttmpBase=*/2, /*numRegs=*/1, rocjitsu::amdgpu::WaitCounterType::KMCNT);
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4, /*vgprDst=*/3,
            /*byteMask=*/0xF, rocjitsu::amdgpu::WaitCounterType::DSCNT);

  b.waitKmcnt(/*wave=*/0, /*kmcnt=*/0);
  b.checkTtmpRead(/*wave=*/0, /*reg=*/2);
  EXPECT_FALSE(b.hasRace());

  b.checkVgprRead(/*wave=*/0, /*reg=*/3, /*lane=*/0);
  EXPECT_TRUE(b.hasVgprRace(3));
}

TEST(RaceDetector, Ttmp_PartialKmcntKeepsUnorderedScalarLoad) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ttmpLoad(/*wave=*/0, /*ttmpBase=*/2, /*numRegs=*/1, rocjitsu::amdgpu::WaitCounterType::KMCNT);
  b.scalarStore(/*wave=*/0, rocjitsu::amdgpu::WaitCounterType::KMCNT);

  b.waitKmcnt(/*wave=*/0, /*kmcnt=*/1);
  b.checkTtmpRead(/*wave=*/0, /*reg=*/2);
  EXPECT_TRUE(b.hasTtmpRace(2));

  b.clearViolations();
  b.waitKmcnt(/*wave=*/0, /*kmcnt=*/0);
  b.checkTtmpRead(/*wave=*/0, /*reg=*/2);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, RejectsOutOfRangeScalarLoadDestination) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ttmpLoad(/*wave=*/0, /*ttmpBase=*/15, /*numRegs=*/2);
  b.scalarLoad(/*wave=*/0, /*sgprBase=*/7, /*numRegs=*/2);
  EXPECT_EQ(b.events().totalAllocated(), 0);
}

// ---- Scalar write-after-write races ----

TEST(RaceDetector, SgprWaw_ScalarLoadThenInstructionWrite) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.scalarLoad(/*wave=*/0, /*sgprBase=*/4, /*numRegs=*/1);
  b.checkSgprWrite(/*wave=*/0, /*reg=*/4);

  ASSERT_EQ(b.raceCount(), 1);
  const auto &violation = b.violations().front();
  EXPECT_EQ(violation.space, RaceViolation::Space::SGPR);
  EXPECT_EQ(violation.index, 4);
  EXPECT_EQ(violation.lane, -1);
  EXPECT_TRUE(violation.isWrite);
  EXPECT_EQ(violation.conflictingEvent, EventId{0});
}

TEST(RaceDetector, TtmpWaw_ScalarLoadThenInstructionWrite) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ttmpLoad(/*wave=*/0, /*ttmpBase=*/4, /*numRegs=*/1);
  b.checkTtmpWrite(/*wave=*/0, /*reg=*/4);

  ASSERT_EQ(b.raceCount(), 1);
  const auto &violation = b.violations().front();
  EXPECT_EQ(violation.space, RaceViolation::Space::TTMP);
  EXPECT_EQ(violation.index, 4);
  EXPECT_EQ(violation.lane, -1);
  EXPECT_TRUE(violation.isWrite);
  EXPECT_EQ(violation.conflictingEvent, EventId{0});
}

TEST(RaceDetector, SgprWaw_ScalarLoadThenScalarLoad) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.scalarLoad(/*wave=*/0, /*sgprBase=*/4, /*numRegs=*/1);
  b.scalarLoad(/*wave=*/0, /*sgprBase=*/4, /*numRegs=*/1);

  ASSERT_EQ(b.raceCount(), 1);
  EXPECT_TRUE(b.hasSgprRace(4));
  EXPECT_TRUE(b.violations().front().isWrite);
  EXPECT_EQ(b.violations().front().conflictingEvent, EventId{0});
}

TEST(RaceDetector, TtmpWaw_ScalarLoadThenScalarLoad) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ttmpLoad(/*wave=*/0, /*ttmpBase=*/4, /*numRegs=*/1);
  b.ttmpLoad(/*wave=*/0, /*ttmpBase=*/4, /*numRegs=*/1);

  ASSERT_EQ(b.raceCount(), 1);
  EXPECT_TRUE(b.hasTtmpRace(4));
  EXPECT_TRUE(b.violations().front().isWrite);
  EXPECT_EQ(b.violations().front().conflictingEvent, EventId{0});
}

TEST(RaceDetector, SgprWaw_WaitClearsPendingLoad) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.scalarLoad(/*wave=*/0, /*sgprBase=*/4, /*numRegs=*/1);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0);
  b.checkSgprWrite(/*wave=*/0, /*reg=*/4);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, TtmpWaw_WaitClearsPendingLoad) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ttmpLoad(/*wave=*/0, /*ttmpBase=*/4, /*numRegs=*/1);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0);
  b.checkTtmpWrite(/*wave=*/0, /*reg=*/4);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, SgprWaw_CleanRegisterNoRace) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.scalarLoad(/*wave=*/0, /*sgprBase=*/4, /*numRegs=*/1);
  b.checkSgprWrite(/*wave=*/0, /*reg=*/5);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, SgprWaw_WideLoadReportsOnlyOverlappingRegister) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.scalarLoad(/*wave=*/0, /*sgprBase=*/3, /*numRegs=*/2);
  b.checkSgprWrite(/*wave=*/0, /*reg=*/4);

  ASSERT_EQ(b.raceCount(), 1);
  EXPECT_EQ(b.violations().front().index, 4);
  EXPECT_EQ(b.violations().front().conflictingEvent, EventId{0});

  b.clearViolations();
  b.checkSgprWrite(/*wave=*/0, /*reg=*/5);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, Sgpr_WideReadReportsPendingEventOnce) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.scalarLoad(/*wave=*/0, /*sgprBase=*/3, /*numRegs=*/2);
  b.checkSgprRead(/*wave=*/0, /*reg=*/3, /*width=*/2);

  ASSERT_EQ(b.raceCount(), 1);
  const auto &violation = b.violations().front();
  EXPECT_EQ(violation.index, 3);
  EXPECT_FALSE(violation.isWrite);
  EXPECT_EQ(violation.conflictingEvent, EventId{0});
}

TEST(RaceDetector, SgprWaw_WideInstructionWriteReportsPendingEventOnce) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.scalarLoad(/*wave=*/0, /*sgprBase=*/3, /*numRegs=*/2);
  b.checkSgprWrite(/*wave=*/0, /*reg=*/3, /*width=*/2);

  ASSERT_EQ(b.raceCount(), 1);
  EXPECT_EQ(b.violations().front().index, 3);
  EXPECT_EQ(b.violations().front().conflictingEvent, EventId{0});
}

TEST(RaceDetector, TtmpWaw_WideScalarLoadReportsPendingEventOnce) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ttmpLoad(/*wave=*/0, /*ttmpBase=*/3, /*numRegs=*/2);
  b.ttmpLoad(/*wave=*/0, /*ttmpBase=*/3, /*numRegs=*/2);

  ASSERT_EQ(b.raceCount(), 1);
  EXPECT_EQ(b.violations().front().space, RaceViolation::Space::TTMP);
  EXPECT_EQ(b.violations().front().index, 3);
  EXPECT_EQ(b.violations().front().conflictingEvent, EventId{0});
}

TEST(RaceDetector, SgprWaw_PreservesEachPendingConflictingEvent) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.scalarLoad(/*wave=*/0, /*sgprBase=*/4, /*numRegs=*/1);
  b.scalarLoad(/*wave=*/0, /*sgprBase=*/4, /*numRegs=*/1);
  b.clearViolations();

  b.checkSgprWrite(/*wave=*/0, /*reg=*/4);

  ASSERT_EQ(b.raceCount(), 2);
  EXPECT_EQ(b.violations()[0].conflictingEvent, EventId{0});
  EXPECT_EQ(b.violations()[1].conflictingEvent, EventId{1});
}

// ---- LDS cross-wave races ----

TEST(RaceDetector, LdsCrossWave_ZeroLdsSize) {
  // LDS size hint is 0 (e.g. compiler metadata says no LDS) but LDS events
  // arrive anyway (e.g. via inline asm buffer_load_dword lds).
  // Byte counters must grow dynamically to accommodate.
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/1, /*sgprs=*/1);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/100, /*bytes=*/4);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0);
  // Missing barrier → cross-wave race.
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/100, /*bytes=*/4);
  EXPECT_TRUE(b.hasLdsRace(100));
}

TEST(RaceDetector, LdsCrossWave_MissingBarrier) {
  // Wave 0 writes 1 byte to LDS[1] (covering [1,2)), wave 1 reads without
  // barrier. Byte-level precision: [0,1) should NOT race, [1,2) should.
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/1, /*sgprs=*/1);
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
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/8, /*sgprs=*/8);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0);
  b.barrier();
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, LdsCrossWave_WarMissingBarrier) {
  // WAR: wave 0 reads LDS[0], waitcnt, then wave 1 writes to LDS[0]
  // without barrier → RACE on the write.
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/8, /*sgprs=*/8);
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4, /*vgprDst=*/0);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0);
  // Missing barrier!
  b.checkLdsWrite(/*wave=*/1, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  EXPECT_TRUE(b.hasLdsRace(0));
}

TEST(RaceDetector, LdsCrossWave_WarWithBarrier) {
  // WAR: wave 0 reads LDS[0], waitcnt, barrier, then wave 1 writes → safe.
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/8, /*sgprs=*/8);
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4, /*vgprDst=*/0);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0);
  b.barrier();
  b.checkLdsWrite(/*wave=*/1, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, LdsCrossWave_WarNoOverlap) {
  // WAR: wave 0 reads LDS[0..4), wave 1 writes LDS[4..8) → safe (no overlap).
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/8, /*sgprs=*/8);
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4, /*vgprDst=*/0);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0);
  b.checkLdsWrite(/*wave=*/1, /*lane=*/0, /*addr=*/4, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

// ---- LDS same-wave ordering ----

TEST(RaceDetector, LdsSameWave_WriteWriteOk) {
  // Two writes to same address, same wave → not a race.
  // Same-wave LDS writes are ordered (same LDS unit, program order).
  // Cross-wave WAW without barrier would be a race, but same-wave is safe.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, LdsSameWave_ReadReadOk) {
  // Two reads from same address → not a race.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0);
  b.barrier();
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, LdsSameWave_WriteReadSameLaneOk) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, LdsSameWave_WriteReadCrossLaneOk) {
  // DS instructions are wave-wide, so a later lane 1 read cannot overtake
  // lane 0's write from the earlier instruction.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.checkLdsRead(/*wave=*/0, /*lane=*/1, /*addr=*/0, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, LdsSameWave_ReadWriteSameLaneOk) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4, /*vgprDst=*/2);
  b.checkLdsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, LdsSameWave_ReadWriteCrossLaneOk) {
  // A later lane 1 write cannot overtake lane 0's read from the earlier
  // wave-wide DS instruction.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4, /*vgprDst=*/2);
  b.checkLdsWrite(/*wave=*/0, /*lane=*/1, /*addr=*/0, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, LdsSameWave_LongHistoryRemainsVisibleToOtherWaves) {
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/8, /*sgprs=*/8);
  constexpr int iterations = 600;
  for (int i = 0; i < iterations; ++i) {
    for (int wave = 0; wave < 2; ++wave) {
      // Each wave reuses its own region, crossing a 16-byte count boundary.
      // Waits release the destination register; a barrier is still needed
      // before another wave can reuse either region.
      const int addr = wave * 64 + 12;
      b.ldsWrite(wave, /*lane=*/0, addr, /*bytes=*/8);
      b.ldsRead(wave, /*lane=*/0, addr, /*bytes=*/8, /*vgprDst=*/2);
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
  b.ldsRead(/*wave=*/1, /*lane=*/0, /*addr=*/12, /*bytes=*/8, /*vgprDst=*/2);
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/12, /*bytes=*/4);
  b.checkLdsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/12, /*bytes=*/4);
  EXPECT_EQ(b.raceCount(), 2);
}

TEST(RaceDetector, LdsSameWave_SharedChunksKeepOtherWaveHazards) {
  for (bool writes : {false, true}) {
    SCOPED_TRACE(writes ? "RAW" : "WAR");
    RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/8, /*sgprs=*/8);
    auto issue = [&](int wave, int addr, int bytes) {
      if (writes)
        b.ldsWrite(wave, /*lane=*/0, addr, bytes);
      else
        b.ldsRead(wave, /*lane=*/0, addr, bytes, /*vgprDst=*/2);
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
    RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/8, /*sgprs=*/8);
    if (writes) {
      b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/8, /*bytes=*/4);
      b.ldsWrite(/*wave=*/1, /*lane=*/0, /*addr=*/16, /*bytes=*/4);
    } else {
      b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/8, /*bytes=*/4, /*vgprDst=*/2);
      b.ldsRead(/*wave=*/1, /*lane=*/0, /*addr=*/16, /*bytes=*/4, /*vgprDst=*/2);
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
    RaceTestBuilder builder(/*numWaves=*/2, /*vgprs=*/8, /*sgprs=*/8);
    // Wave 1's pending access lies in the gap between wave 0's two ranges.
    if (writes)
      builder.ldsWrite(/*wave=*/1, /*lane=*/0, /*addr=*/1024, /*bytes=*/4);
    else
      builder.ldsRead(/*wave=*/1, /*lane=*/0, /*addr=*/1024, /*bytes=*/4, /*vgprDst=*/2);
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
    RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/8, /*sgprs=*/8);
    auto ordinary = [&](int wave) {
      if (writes)
        b.ldsWrite(wave, /*lane=*/0, /*addr=*/12, /*bytes=*/8);
      else
        b.ldsRead(wave, /*lane=*/0, /*addr=*/12, /*bytes=*/8, /*vgprDst=*/2);
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
      b.flatLdsStore(/*wave=*/0, /*lane=*/0, /*addr=*/14, /*bytes=*/4);
    else
      b.flatLdsLoad(/*wave=*/0, /*lane=*/0, /*addr=*/14, /*bytes=*/4, /*vgprDst=*/4);
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

// ---- Same-wave VGPR via LDS load ----

TEST(RaceDetector, SameWave_WriteReadRace) {
  // LDS load into VGPR, read VGPR without waitcnt → RACE.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4, /*vgprDst=*/2);
  b.checkVgprRead(/*wave=*/0, /*reg=*/2, /*lane=*/0);
  EXPECT_TRUE(b.hasVgprRace(2));
}

TEST(RaceDetector, SameWave_WriteWaitcntOk) {
  // LDS load into VGPR, waitcnt, read VGPR → safe.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4, /*vgprDst=*/2);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0);
  b.checkVgprRead(/*wave=*/0, /*reg=*/2, /*lane=*/0);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, SameWave_DoubleWriteRace) {
  // Two LDS loads into same VGPR, waitcnt(1) drains oldest → newest RACE.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4,
            /*vgprDst=*/2); // oldest
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/64, /*bytes=*/4,
            /*vgprDst=*/2);                           // newest
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/1); // drain oldest
  b.checkVgprRead(/*wave=*/0, /*reg=*/2, /*lane=*/0); // still racing
  EXPECT_TRUE(b.hasVgprRace(2));
}

TEST(RaceDetector, SameWave_BarrierNoWaitcnt) {
  // Barrier without waitcnt doesn't clear same-wave events.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4, /*vgprDst=*/2);
  b.barrier(); // no waitcnt first!
  b.checkVgprRead(/*wave=*/0, /*reg=*/2, /*lane=*/0);
  EXPECT_TRUE(b.hasVgprRace(2));
}

TEST(RaceDetector, SameWave_WaitcntBarrierOk) {
  // Waitcnt then barrier → safe.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4, /*vgprDst=*/2);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0);
  b.barrier();
  b.checkVgprRead(/*wave=*/0, /*reg=*/2, /*lane=*/0);
  EXPECT_FALSE(b.hasRace());
}

// ---- VGPR write-after-write races ----

TEST(RaceDetector, VgprWaw_GlobalLoadThenInstructionWrite) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/2, /*numRegs=*/1);
  b.checkVgprWrite(/*wave=*/0, /*reg=*/2, /*lane=*/0);

  ASSERT_EQ(b.raceCount(), 1);
  const auto &violation = b.violations().front();
  EXPECT_EQ(violation.space, RaceViolation::Space::VGPR);
  EXPECT_EQ(violation.index, 2);
  EXPECT_EQ(violation.lane, 0);
  EXPECT_TRUE(violation.isWrite);
  EXPECT_EQ(violation.conflictingEvent, EventId{0});
}

TEST(RaceDetector, VgprWaw_LdsReadThenInstructionWrite) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4, /*vgprDst=*/2);
  b.checkVgprWrite(/*wave=*/0, /*reg=*/2, /*lane=*/0);
  EXPECT_TRUE(b.hasVgprRace(2));
}

TEST(RaceDetector, VgprWaw_SameCompletionClassIsOrdered) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/2, /*numRegs=*/1);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/2, /*numRegs=*/1);

  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, VgprWaw_UnorderedLoadsRace) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/2, /*numRegs=*/1, /*exec=*/0,
               /*byteMask=*/0xF, MemoryOrderClass::UNORDERED);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/2, /*numRegs=*/1, /*exec=*/0,
               /*byteMask=*/0xF, MemoryOrderClass::UNORDERED);

  EXPECT_TRUE(b.hasVgprRace(2));
}

TEST(RaceDetector, VgprWaw_DifferentCompletionClassesAreUnordered) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/2, /*numRegs=*/1);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/2, /*numRegs=*/1, /*exec=*/0,
               /*byteMask=*/0xF, MemoryOrderClass::UNORDERED);

  EXPECT_TRUE(b.hasVgprRace(2));
}

TEST(RaceDetector, VgprWaw_MixedOrderFlatLoadThenGlobalLoadRace) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.flatGlobalLoad(/*wave=*/0, /*vgprBase=*/2, /*numRegs=*/1,
                   kMixedOrderGenericFlatLoadObligations);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/2, /*numRegs=*/1);

  EXPECT_TRUE(b.hasVgprRace(2));
}

TEST(RaceDetector, VgprWaw_GlobalLoadThenMixedOrderFlatLoadRace) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/2, /*numRegs=*/1);
  b.flatGlobalLoad(/*wave=*/0, /*vgprBase=*/2, /*numRegs=*/1,
                   kMixedOrderGenericFlatLoadObligations);

  EXPECT_TRUE(b.hasVgprRace(2));
}

TEST(RaceDetector, VgprWaw_WaitcntClearsPendingLoad) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/2, /*numRegs=*/1);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/0);
  b.checkVgprWrite(/*wave=*/0, /*reg=*/2, /*lane=*/0);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, VgprWaw_CleanRegisterNoRace) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/1, /*numRegs=*/1);
  b.checkVgprWrite(/*wave=*/0, /*reg=*/2, /*lane=*/0);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, VgprWaw_WriteLaneMaskHonorsPendingLoadLanes) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8, /*waveSize=*/32);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/2, /*numRegs=*/1, /*exec=*/1ULL << 5);

  b.checkVgprWriteLanes(/*wave=*/0, /*reg=*/2, /*laneMask=*/1ULL << 3);
  EXPECT_FALSE(b.hasRace());
  b.checkVgprWriteLanes(/*wave=*/0, /*reg=*/2,
                        /*laneMask=*/(1ULL << 3) | (1ULL << 5));

  ASSERT_EQ(b.raceCount(), 1);
  EXPECT_EQ(b.violations().front().lane, 5);
}

TEST(RaceDetector, VgprWaw_SubDwordDisjointBytesNoRace) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/2, /*vgprDst=*/2,
            /*byteMask=*/0b0011);
  b.checkVgprWrite(/*wave=*/0, /*reg=*/2, /*lane=*/0, /*byteMask=*/0b1100);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, VgprWaw_SubDwordOverlappingBytesRace) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/2, /*vgprDst=*/2,
            /*byteMask=*/0b0011);
  b.checkVgprWrite(/*wave=*/0, /*reg=*/2, /*lane=*/0, /*byteMask=*/0b0011);
  EXPECT_TRUE(b.hasVgprRace(2));
}

TEST(RaceDetector, VgprWaw_PreservesExactConflictForMultiplePendingLoads) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/2, /*numRegs=*/1);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/2, /*numRegs=*/1);
  b.checkVgprWrite(/*wave=*/0, /*reg=*/2, /*lane=*/0);

  ASSERT_EQ(b.raceCount(), 2);
  EXPECT_EQ(b.violations()[0].conflictingEvent, EventId{0});
  EXPECT_EQ(b.violations()[1].conflictingEvent, EventId{1});
}

// ---- Deep event stack ----

TEST(RaceDetector, DeepStack_PartialWaitcnt) {
  // 3 loads, waitcnt(1) drains oldest, newest two still pending.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/0, /*numRegs=*/1); // oldest
  b.globalLoad(/*wave=*/0, /*vgprBase=*/1, /*numRegs=*/1);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/2, /*numRegs=*/1); // newest
  b.waitcnt(/*wave=*/0, /*vmcnt=*/1);
  b.checkVgprRead(/*wave=*/0, /*reg=*/0, /*lane=*/0); // safe (drained)
  EXPECT_FALSE(b.hasVgprRace(0));
  b.checkVgprRead(/*wave=*/0, /*reg=*/2, /*lane=*/0); // RACE (newest)
  EXPECT_TRUE(b.hasVgprRace(2));
}

TEST(RaceDetector, DeepStack_FullWaitcnt) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/0, /*numRegs=*/1);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/1, /*numRegs=*/1);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/2, /*numRegs=*/1);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/0);
  b.checkVgprRead(/*wave=*/0, /*reg=*/0, /*lane=*/0);
  b.checkVgprRead(/*wave=*/0, /*reg=*/1, /*lane=*/0);
  b.checkVgprRead(/*wave=*/0, /*reg=*/2, /*lane=*/0);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, DeepStack_ConsecutiveWaitcnts) {
  // Two partial waitcnts in sequence. The second drain must retire the
  // correct (oldest) event — not a newer one that got reordered.
  //   load v0, load v1, load v2  (3 outstanding, oldest first)
  //   waitcnt vmcnt(2)           → drain oldest (v0), keep v1 and v2
  //   load v3                    (v1, v2, v3 outstanding)
  //   waitcnt vmcnt(2)           → drain oldest (v1), keep v2 and v3
  //   read v0 → safe  (drained by first waitcnt)
  //   read v1 → safe  (drained by second waitcnt)
  //   read v2 → RACE  (still pending)
  //   read v3 → RACE  (still pending)
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/0, /*numRegs=*/1);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/1, /*numRegs=*/1);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/2, /*numRegs=*/1);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/2);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/3, /*numRegs=*/1);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/2);
  b.checkVgprRead(/*wave=*/0, /*reg=*/0, /*lane=*/0);
  EXPECT_FALSE(b.hasVgprRace(0));
  b.checkVgprRead(/*wave=*/0, /*reg=*/1, /*lane=*/0);
  EXPECT_FALSE(b.hasVgprRace(1));
  b.checkVgprRead(/*wave=*/0, /*reg=*/2, /*lane=*/0);
  EXPECT_TRUE(b.hasVgprRace(2));
  b.checkVgprRead(/*wave=*/0, /*reg=*/3, /*lane=*/0);
  EXPECT_TRUE(b.hasVgprRace(3));
}

// ---- D16 (byte-level VGPR races) ----

TEST(RaceDetector, D16_LoHiLoadDrained) {
  // Load lo half, load hi half, drain oldest (lo) → lo read safe.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4, /*vgprDst=*/2,
            /*byteMask=*/0x3); // lo
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4, /*vgprDst=*/2,
            /*byteMask=*/0xC);                        // hi
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/1); // drain oldest (lo)
  b.checkVgprRead(/*wave=*/0, /*reg=*/2, /*lane=*/0,
                  /*byteMask=*/0x3); // lo safe
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, D16_HiOutstanding_LoSafe) {
  // Hi half still outstanding, but we only read lo → safe.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4, /*vgprDst=*/2,
            /*byteMask=*/0x3); // lo
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4, /*vgprDst=*/2,
            /*byteMask=*/0xC);                        // hi
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/1); // drain lo
  b.checkVgprRead(/*wave=*/0, /*reg=*/2, /*lane=*/0,
                  /*byteMask=*/0x3); // read lo only
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, D16_FullLoadRaces) {
  // Full-register load, partial waitcnt → full read races.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4, /*vgprDst=*/2,
            /*byteMask=*/0xF);
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/64, /*bytes=*/4, /*vgprDst=*/3,
            /*byteMask=*/0xF);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/1); // drain oldest (v2)
  b.checkVgprRead(/*wave=*/0, /*reg=*/3, /*lane=*/0); // v3 still pending → RACE
  EXPECT_TRUE(b.hasVgprRace(3));
}

TEST(RaceDetector, D16_LoOutstanding_FullRaces) {
  // Lo half outstanding, full read → races (overlapping bytes).
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4, /*vgprDst=*/2,
            /*byteMask=*/0x3);
  b.checkVgprRead(/*wave=*/0, /*reg=*/2, /*lane=*/0,
                  /*byteMask=*/0xF); // full read → RACE
  EXPECT_TRUE(b.hasVgprRace(2));
}

// ---- Direct-to-LDS (GlobalToLds) ----

TEST(RaceDetector, Dtl_CrossWaveRace) {
  // Wave 0 does DTL load to LDS, waitcnt, but no barrier.
  // Wave 1 reads LDS → RACE (WAVE_COMPLETE but not RETIRED).
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/4, /*sgprs=*/4);
  b.globalToLds(/*wave=*/0, /*ldsAddrs=*/{0}, /*bytesPerLane=*/64);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/0);
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/0, /*bytes=*/4); // RACE
  EXPECT_TRUE(b.hasLdsRace(0));
}

TEST(RaceDetector, Dtl_CrossWaveSafe) {
  // Same as above but with barrier → safe.
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/4, /*sgprs=*/4);
  b.globalToLds(/*wave=*/0, /*ldsAddrs=*/{0}, /*bytesPerLane=*/64);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/0);
  b.barrier();
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, Dtl_SameWaveMissingVmcntRace) {
  // Direct-to-LDS arrives through VMEM, so unlike an ordinary DS write the
  // owning wave must wait for vmcnt before reading the LDS bytes.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/4, /*sgprs=*/4);
  b.globalToLds(/*wave=*/0, /*ldsAddrs=*/{0}, /*bytesPerLane=*/4);
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  EXPECT_TRUE(b.hasLdsRace(0));
}

TEST(RaceDetector, Dtl_SameWaveWithVmcntSafe) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/4, /*sgprs=*/4);
  b.globalToLds(/*wave=*/0, /*ldsAddrs=*/{0}, /*bytesPerLane=*/4);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/0);
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, Dtl_SameWavePartialVmcntRetiresOldest) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/4, /*sgprs=*/4);
  b.globalToLds(/*wave=*/0, /*ldsAddrs=*/{16}, /*bytesPerLane=*/4, /*exec=*/1);
  b.globalToLds(/*wave=*/0, /*ldsAddrs=*/{64}, /*bytesPerLane=*/4, /*exec=*/1);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/1);

  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/16, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());

  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/64, /*bytes=*/4);
  EXPECT_TRUE(b.hasLdsRace(64));
}

// ---- Tensor DMA (tensor_load_to_lds / tensor_store_from_lds) ----

TEST(RaceDetector, TensorDma_CrossWaveLoadRace) {
  // Wave 1 reads the destination of wave 0's tensor load with no wait or barrier.
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/4, /*sgprs=*/4);
  b.tensorLoadToLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/4, /*bytes=*/4);
  EXPECT_TRUE(b.hasLdsRace(4));
}

TEST(RaceDetector, TensorDma_CrossWaveLoadWaitWithoutBarrierRace) {
  // tensorcnt completes the load for wave 0 only; other waves need a barrier.
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/4, /*sgprs=*/4);
  b.tensorLoadToLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.waitTensorcnt(/*wave=*/0, /*tensorcnt=*/0);
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/4, /*bytes=*/4);
  EXPECT_TRUE(b.hasLdsRace(4));
}

TEST(RaceDetector, TensorDma_CrossWaveLoadSafe) {
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/4, /*sgprs=*/4);
  b.tensorLoadToLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.waitTensorcnt(/*wave=*/0, /*tensorcnt=*/0);
  b.barrier();
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/4, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, TensorDma_SameWaveMissingTensorcntRace) {
  // Tensor DMA completes independently of DS, so the issuing wave must wait
  // for tensorcnt before reading the destination.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/4, /*sgprs=*/4);
  b.tensorLoadToLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/12, /*bytes=*/4);
  EXPECT_TRUE(b.hasLdsRace(12));
}

TEST(RaceDetector, TensorDma_SameWaveWithTensorcntSafe) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/4, /*sgprs=*/4);
  b.tensorLoadToLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.waitTensorcnt(/*wave=*/0, /*tensorcnt=*/0);
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/12, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, TensorDma_OnlyTransferredRangesAreTracked) {
  // A padded tile leaves the padding bytes between its rows untouched.
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/4, /*sgprs=*/4);
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
        1, 4, 4, Dim3d(0), [&](RaceViolation v) { violations.push_back(v); },
        counterCapacitiesForArch(ROCJITSU_CODE_ARCH_CDNA5), /*ldsSizeBytes=*/64 * 1024);
    auto &wave = detector.getWaveRaceState(0);
    const std::array obligations{
        amdgpu::MemoryCounterObligation{amdgpu::WaitCounterType::TENSORCNT, c.obligationClass}};
    for (uint32_t address : {0u, 64u}) {
      const std::array ranges{LdsByteRange{.address = address, .bytes = 16}};
      wave.registerLdsRangeEvent(testInstruction(address), MemoryEventType::GLOBAL_TO_LDS, ranges,
                                 obligations, c.eventOrder);
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
      1, 4, 4, Dim3d(0), [&](RaceViolation v) { violations.push_back(v); },
      counterCapacitiesForArch(ROCJITSU_CODE_ARCH_CDNA5), /*ldsSizeBytes=*/64 * 1024);
  auto &wave = detector.getWaveRaceState(0);
  const std::array obligations{amdgpu::MemoryCounterObligation{amdgpu::WaitCounterType::TENSORCNT,
                                                               MemoryOrderClass::TENSOR}};
  constexpr int kTransfers = 4096;
  for (int i = 0; i < kTransfers; ++i) {
    wave.registerLdsRangeEvent(testInstruction(i), MemoryEventType::LDS_TO_GLOBAL, {}, obligations,
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
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/4, /*sgprs=*/4);
  b.tensorStoreFromLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.waitTensorcnt(/*wave=*/0, /*tensorcnt=*/0);
  EXPECT_FALSE(b.events().isTrimmable(EventId{0}));
  b.barrier();
  EXPECT_EQ(b.events().status(EventId{0}), EventStatus::RETIRED);
}

TEST(RaceDetector, TensorDma_SameWavePartialTensorcntRetiresOldest) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/4, /*sgprs=*/4);
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
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/4, /*sgprs=*/4);
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
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/4, /*sgprs=*/4);
  b.tensorStoreFromLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.tensorLoadToLds(/*wave=*/0, {{.address = 4, .bytes = 8}});
  EXPECT_TRUE(b.hasLdsRace(4));
}

TEST(RaceDetector, TensorDma_SameWaveStoreReadingAPendingLoadDestinationRaces) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/4, /*sgprs=*/4);
  b.tensorLoadToLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.tensorStoreFromLds(/*wave=*/0, {{.address = 4, .bytes = 8}});
  EXPECT_TRUE(b.hasLdsRace(4));
}

TEST(RaceDetector, TensorDma_SameWaveOverlappingTransfersAfterTensorcntAreSafe) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/4, /*sgprs=*/4);
  b.tensorStoreFromLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.waitTensorcnt(/*wave=*/0, /*tensorcnt=*/0);
  b.tensorLoadToLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.waitTensorcnt(/*wave=*/0, /*tensorcnt=*/0);
  b.tensorStoreFromLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, TensorDma_SameWaveStoresOfTheSameTileAreSafe) {
  // Both only read the tile.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/4, /*sgprs=*/4);
  b.tensorStoreFromLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.tensorStoreFromLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, TensorDma_SameWaveLoadsOverTheSameTileAreNotReported) {
  // Two loads writing the same bytes are an LDS write/write conflict, which the
  // detector does not check; this pins that documented limitation.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/4, /*sgprs=*/4);
  b.tensorLoadToLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.tensorLoadToLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, TensorDma_SameWaveDisjointTransfersAreSafe) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/4, /*sgprs=*/4);
  b.tensorStoreFromLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.tensorLoadToLds(/*wave=*/0, {{.address = 16, .bytes = 16}});
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, TensorDma_CrossWaveLoadThenStoreRace) {
  // Wave 1's store reads bytes wave 0's load may still be writing.
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/4, /*sgprs=*/4);
  b.tensorLoadToLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.tensorStoreFromLds(/*wave=*/1, {{.address = 4, .bytes = 8}});
  EXPECT_TRUE(b.hasLdsRace(4));
}

TEST(RaceDetector, TensorDma_CrossWaveStoreThenLoadRace) {
  // Wave 1's load overwrites bytes wave 0's store may still be reading.
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/4, /*sgprs=*/4);
  b.tensorStoreFromLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.tensorLoadToLds(/*wave=*/1, {{.address = 4, .bytes = 8}});
  EXPECT_TRUE(b.hasLdsRace(4));
}

TEST(RaceDetector, TensorDma_CrossWaveStoreSourceOverwriteRace) {
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/4, /*sgprs=*/4);
  b.tensorStoreFromLds(/*wave=*/0, {{.address = 32, .bytes = 16}});
  b.checkLdsWrite(/*wave=*/1, /*lane=*/0, /*addr=*/40, /*bytes=*/4);
  EXPECT_TRUE(b.hasLdsRace(40));
}

TEST(RaceDetector, TensorDma_CrossWaveStoreSourceOverwriteSafe) {
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/4, /*sgprs=*/4);
  b.tensorStoreFromLds(/*wave=*/0, {{.address = 32, .bytes = 16}});
  b.waitTensorcnt(/*wave=*/0, /*tensorcnt=*/0);
  b.barrier();
  b.checkLdsWrite(/*wave=*/1, /*lane=*/0, /*addr=*/40, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, TensorDma_StoreReadsPendingCrossWaveWriteRace) {
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/4, /*sgprs=*/4);
  b.ldsWrite(/*wave=*/1, /*lane=*/0, /*addr=*/36, /*bytes=*/4);
  b.tensorStoreFromLds(/*wave=*/0, {{.address = 32, .bytes = 16}});
  EXPECT_TRUE(b.hasLdsRace(36));
}

TEST(RaceDetector, TensorDma_LoadOverwritesPendingCrossWaveReadRace) {
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/4, /*sgprs=*/4);
  b.ldsRead(/*wave=*/1, /*lane=*/0, /*addr=*/4, /*bytes=*/4, /*vgprDst=*/0);
  b.tensorLoadToLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  EXPECT_TRUE(b.hasLdsRace(4));
}

TEST(RaceDetector, TensorDma_LoadAfterCrossWaveReadRetiredSafe) {
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/4, /*sgprs=*/4);
  b.ldsRead(/*wave=*/1, /*lane=*/0, /*addr=*/4, /*bytes=*/4, /*vgprDst=*/0);
  b.waitcnt(/*wave=*/1, /*vmcnt=*/-1, /*lgkmcnt=*/0);
  b.barrier();
  b.tensorLoadToLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, TensorDma_StoreReadsSameWavePendingDsWriteRace) {
  // DS and tensor DMA complete independently, so the store needs a DS wait.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/4, /*sgprs=*/4);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/36, /*bytes=*/4);
  b.tensorStoreFromLds(/*wave=*/0, {{.address = 32, .bytes = 16}});
  EXPECT_TRUE(b.hasLdsRace(36));
}

TEST(RaceDetector, TensorDma_RaceIsReportedAtAByteBothAccessesTouch) {
  // The store reads bytes 0-127 but shares only bytes 120-123 with the DS
  // write, so the report names byte 120, not the start of the store.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/4, /*sgprs=*/4);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/120, /*bytes=*/4);
  b.tensorStoreFromLds(/*wave=*/0, {{.address = 0, .bytes = 128}});
  EXPECT_TRUE(b.hasLdsRace(120));
  EXPECT_FALSE(b.hasLdsRace(0));
}

TEST(RaceDetector, TensorDma_StoreReadsSameWaveCompletedDsWriteSafe) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/4, /*sgprs=*/4);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/36, /*bytes=*/4);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0);
  b.tensorStoreFromLds(/*wave=*/0, {{.address = 32, .bytes = 16}});
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, TensorDma_OtherCountersDoNotRetireTransfer) {
  // Only tensorcnt tracks tensor DMA.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/4, /*sgprs=*/4);
  b.tensorLoadToLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.waitcnt(/*wave=*/0, /*vmcnt=*/0, /*lgkmcnt=*/0);
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/12, /*bytes=*/4);
  EXPECT_TRUE(b.hasLdsRace(12));
}

TEST(RaceDetector, TensorDma_TensorcntDoesNotRetireDsWrite) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/4, /*sgprs=*/4);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/36, /*bytes=*/4);
  b.waitTensorcnt(/*wave=*/0, /*tensorcnt=*/0);
  b.tensorStoreFromLds(/*wave=*/0, {{.address = 32, .bytes = 16}});
  EXPECT_TRUE(b.hasLdsRace(36));
}

TEST(RaceDetector, TensorDma_CrossWaveLoadBarrierWithoutTensorcntRace) {
  // A barrier retires only transfers that their wave has already waited for.
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/4, /*sgprs=*/4);
  b.tensorLoadToLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.barrier();
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/4, /*bytes=*/4);
  EXPECT_TRUE(b.hasLdsRace(4));
}

TEST(RaceDetector, TensorDma_CrossWaveStoreWaitWithoutBarrierRace) {
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/4, /*sgprs=*/4);
  b.tensorStoreFromLds(/*wave=*/0, {{.address = 32, .bytes = 16}});
  b.waitTensorcnt(/*wave=*/0, /*tensorcnt=*/0);
  b.checkLdsWrite(/*wave=*/1, /*lane=*/0, /*addr=*/40, /*bytes=*/4);
  EXPECT_TRUE(b.hasLdsRace(40));
}

TEST(RaceDetector, TensorDma_DoubleBufferedPartialWaitAndBarrier) {
  // Pipelined kernels wait for all but the newest transfer, then synchronize.
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/4, /*sgprs=*/4);
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
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/4, /*sgprs=*/4);
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
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/4, /*sgprs=*/4);
  b.tensorStoreFromLds(/*wave=*/0, {{.address = 32, .bytes = 16}});
  b.globalToLds(/*wave=*/0, /*ldsAddrs=*/{36}, /*bytesPerLane=*/4, /*exec=*/1);
  EXPECT_TRUE(b.hasLdsRace(36));
}

TEST(RaceDetector, TensorDma_DtlAfterStoreRetiredSafe) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/4, /*sgprs=*/4);
  b.tensorStoreFromLds(/*wave=*/0, {{.address = 32, .bytes = 16}});
  b.waitTensorcnt(/*wave=*/0, /*tensorcnt=*/0);
  b.globalToLds(/*wave=*/0, /*ldsAddrs=*/{36}, /*bytesPerLane=*/4, /*exec=*/1);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, TensorDma_StoreReadsSameWavePendingDtlRace) {
  // Direct-to-LDS completes on vmcnt, independently of tensor DMA.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/4, /*sgprs=*/4);
  b.globalToLds(/*wave=*/0, /*ldsAddrs=*/{32}, /*bytesPerLane=*/4, /*exec=*/1);
  b.tensorStoreFromLds(/*wave=*/0, {{.address = 32, .bytes = 16}});
  EXPECT_TRUE(b.hasLdsRace(32));
}

TEST(RaceDetector, TensorDma_StoreReadsSameWaveCompletedDtlSafe) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/4, /*sgprs=*/4);
  b.globalToLds(/*wave=*/0, /*ldsAddrs=*/{32}, /*bytesPerLane=*/4, /*exec=*/1);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/0);
  b.tensorStoreFromLds(/*wave=*/0, {{.address = 32, .bytes = 16}});
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, TensorDma_TransferWithoutRangesTakesATensorcntSlot) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/4, /*sgprs=*/4);
  b.tensorLoadToLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.tensorLoadToLds(/*wave=*/0, {});
  b.waitTensorcnt(/*wave=*/0, /*tensorcnt=*/1);
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/4, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, TensorDma_EachTransferWithoutRangesTakesOneTensorcntSlot) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/4, /*sgprs=*/4);
  b.tensorLoadToLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.tensorLoadToLds(/*wave=*/0, {});
  b.tensorLoadToLds(/*wave=*/0, {});
  b.waitTensorcnt(/*wave=*/0, /*tensorcnt=*/2);
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/4, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, TensorDma_PartialWaitRetiresAnOlderTransferWithoutRanges) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/4, /*sgprs=*/4);
  b.tensorLoadToLds(/*wave=*/0, {});
  b.tensorLoadToLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.waitTensorcnt(/*wave=*/0, /*tensorcnt=*/1);
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/4, /*bytes=*/4);
  EXPECT_TRUE(b.hasLdsRace(4));
}

TEST(RaceDetector, TensorDma_StoreWithoutRangesTakesATensorcntSlot) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/4, /*sgprs=*/4);
  b.tensorLoadToLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.tensorStoreFromLds(/*wave=*/0, {});
  b.waitTensorcnt(/*wave=*/0, /*tensorcnt=*/1);
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/4, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, TensorDma_TransferWithoutRangesConflictsWithNothing) {
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/4, /*sgprs=*/4);
  b.tensorLoadToLds(/*wave=*/0, {});
  b.tensorStoreFromLds(/*wave=*/0, {});
  for (int wave : {0, 1}) {
    b.checkLdsRead(wave, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
    b.checkLdsWrite(wave, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  }
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, TensorDma_RetiredTransferWithoutRangesNoLongerCounts) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/4, /*sgprs=*/4);
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
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/4, /*sgprs=*/4);
  b.tensorLoadToLds(/*wave=*/0, {{.address = 0, .bytes = 16}});
  b.tensorLoadToLds(/*wave=*/0, {});
  b.barrier();
  b.waitTensorcnt(/*wave=*/0, /*tensorcnt=*/1);
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/4, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, TensorDma_RangeEdgesAreBytePrecise) {
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/4, /*sgprs=*/4);
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
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/4, /*sgprs=*/4);
  b.tensorLoadToLds(/*wave=*/0, {{.address = 0, .bytes = 6}});
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/4, /*bytes=*/2);
  EXPECT_TRUE(b.hasLdsRace(4));
}

// ---- Exec mask ----

TEST(RaceDetector, Exec_DirectToLdsTracksOnlyActiveLaneIntervals) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
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
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8,
                    /*waveSize=*/64, Dim3d(7, 3, 1));
  b.globalLoad(/*wave=*/0, /*vgprBase=*/1, /*numRegs=*/1);
  b.checkVgprRead(/*wave=*/0, /*reg=*/1, /*lane=*/0);
  ASSERT_EQ(b.raceCount(), 1);
  auto &v = b.violations()[0];
  EXPECT_EQ(v.workgroupId.x, 7);
  EXPECT_EQ(v.workgroupId.y, 3);
  EXPECT_EQ(v.workgroupId.z, 1);
}

// ---- LDS additional patterns ----

TEST(RaceDetector, LdsCrossWave_GlobalLoadToLdsWriteMissingVmcnt) {
  // Global load into VGPR, then LDS write using that VGPR without vmcnt.
  // The VGPR race fires when the LDS write reads the VGPR source.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/1, /*numRegs=*/1);
  // no vmcnt before reading v1
  b.checkVgprRead(/*wave=*/0, /*reg=*/1,
                  /*lane=*/0); // simulates ds_write reading v1
  EXPECT_TRUE(b.hasVgprRace(1));
}

TEST(RaceDetector, GlobalToLdsHonorsLaneMask) {
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/8, /*sgprs=*/8, /*waveSize=*/4);
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
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0);
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.checkLdsRead(/*wave=*/0, /*lane=*/1, /*addr=*/0, /*bytes=*/4);
  b.checkLdsRead(/*wave=*/0, /*lane=*/2, /*addr=*/0, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, LdsSameWave_ScatterBroadcastOk) {
  // Each lane writes unique addr, waitcnt, then all read addr 0 → safe.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
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
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
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
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/8, /*sgprs=*/8);

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
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/8, /*sgprs=*/8);

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
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/4, /*sgprs=*/4);
  b.globalStore(/*wave=*/0);                               // vmcnt event 0 (oldest)
  b.globalLoad(/*wave=*/0, /*vgprBase=*/1, /*numRegs=*/1); // vmcnt event 1
  b.waitcnt(/*wave=*/0, /*vmcnt=*/1);                      // drain oldest (store), load stays
  b.checkVgprRead(/*wave=*/0, /*reg=*/1,
                  /*lane=*/0); // load still pending → RACE
  EXPECT_TRUE(b.hasVgprRace(1));
}

TEST(RaceDetector, GlobalStore_NoVgprRaceOnSource) {
  // Global stores read VGPRs at issue time. Writing to the source VGPR
  // after issuing the store should NOT race — the store already captured
  // the value. (Store has no destination registers.)
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/4, /*sgprs=*/4);
  b.globalStore(/*wave=*/0);
  // Overwriting the "source" VGPR is fine — store doesn't track dest regs.
  b.globalLoad(/*wave=*/0, /*vgprBase=*/0, /*numRegs=*/1);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/0);
  b.checkVgprRead(/*wave=*/0, /*reg=*/0, /*lane=*/0);
  EXPECT_FALSE(b.hasRace());
}

// ---- Multi-register loads ----

TEST(RaceDetector, MultiReg_PartialWaitcnt) {
  // Load 4 registers in one event (like global_load_dwordx4 into v[0:3]).
  // waitcnt(0) should clear all 4.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/4);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/0, /*numRegs=*/4);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/0);
  b.checkVgprRead(/*wave=*/0, /*reg=*/0, /*lane=*/0);
  b.checkVgprRead(/*wave=*/0, /*reg=*/1, /*lane=*/0);
  b.checkVgprRead(/*wave=*/0, /*reg=*/2, /*lane=*/0);
  b.checkVgprRead(/*wave=*/0, /*reg=*/3, /*lane=*/0);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, MultiReg_NoWaitcnt) {
  // Load 4 registers, no waitcnt. All 4 should race.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/4);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/0, /*numRegs=*/4);
  b.checkVgprRead(/*wave=*/0, /*reg=*/0, /*lane=*/0);
  b.checkVgprRead(/*wave=*/0, /*reg=*/3, /*lane=*/0);
  EXPECT_TRUE(b.hasVgprRace(0));
  EXPECT_TRUE(b.hasVgprRace(3));
}

TEST(RaceDetector, MultiReg_TwoLoads_PartialDrain) {
  // Two 2-register loads. waitcnt(1) drains the oldest (v[0:1]).
  // Reading v0 is safe, reading v2 races.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/4);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/0, /*numRegs=*/2); // oldest: v0,v1
  b.globalLoad(/*wave=*/0, /*vgprBase=*/2, /*numRegs=*/2); // newest: v2,v3
  b.waitcnt(/*wave=*/0, /*vmcnt=*/1);                      // drain oldest
  b.checkVgprRead(/*wave=*/0, /*reg=*/0, /*lane=*/0);      // safe
  EXPECT_FALSE(b.hasVgprRace(0));
  b.checkVgprRead(/*wave=*/0, /*reg=*/2, /*lane=*/0); // RACE
  EXPECT_TRUE(b.hasVgprRace(2));
}

// ---- DTL multi-lane ----

TEST(RaceDetector, Dtl_MultiLane_CrossWaveRace) {
  // DTL with 4 active lanes, each writing 16 bytes to different LDS offsets.
  // Cross-wave read to lane 2's region should detect the race.
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/4, /*sgprs=*/4);
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
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/4, /*sgprs=*/4);
  b.globalToLds(/*wave=*/0, {0, 16, 32, 48}, /*bytesPerLane=*/16, /*exec=*/0xF);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/0);
  b.barrier();
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/16, /*bytes=*/4);
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/32, /*bytes=*/4);
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/48, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

// ---- Two LDS address sets per lane ----

TEST(RaceDetector, DualAddress_CrossWaveRace) {
  // Like ds_write2_b64, lane 0 writes [100, 108) and [116, 124). Another wave
  // reading either interval without a barrier races.
  std::vector<RaceViolation> violations;
  RaceDetector detector(/*nWaves=*/
                        2, /*vgprCount=*/4,
                        /*sgprCount=*/4, Dim3d(0),
                        [&](RaceViolation v) { violations.push_back(v); },
                        counterCapacitiesForArch(ROCJITSU_CODE_ARCH_CDNA4),
                        /*ldsSizeBytes=*/64 * 1024);
  auto &rs = detector.getWaveRaceState(0);
  std::vector<uint32_t> firstAddrs(64, 0);
  std::vector<uint32_t> secondAddrs(64, 0);
  firstAddrs[0] = 100;
  secondAddrs[0] = 116;
  const std::array obligations{
      amdgpu::MemoryCounterObligation{defaultWaitCounterType(MemoryEventType::VGPR_TO_LDS),
                                      defaultMemoryOrder(MemoryEventType::VGPR_TO_LDS)}};
  rs.registerLdsEvent(testInstruction(/*pc=*/0), MemoryEventType::VGPR_TO_LDS,
                      /*registers=*/{}, /*execMask=*/1, /*waveSize=*/64, firstAddrs, secondAddrs,
                      /*bytesPerLane=*/8, /*byteMask=*/0xF, obligations,
                      defaultMemoryOrder(MemoryEventType::VGPR_TO_LDS));
  detector.validateRead(/*addr=*/100, WaveId{1}, /*lane=*/0, /*nBytes=*/4);
  EXPECT_EQ(violations.size(), 1u); // first interval
  detector.validateRead(/*addr=*/116, WaveId{1}, /*lane=*/0, /*nBytes=*/4);
  EXPECT_EQ(violations.size(), 2u); // second interval
  // Outside both intervals: no race
  detector.validateRead(/*addr=*/108, WaveId{1}, /*lane=*/0, /*nBytes=*/4);
  EXPECT_EQ(violations.size(), 2u); // still 2
}

// ---- Exec mask and lane-level checks ----

TEST(RaceDetector, CheckVgprReadOnlyReportsLanesInLoadExecMask) {
  RaceTestBuilder active(/*numWaves=*/1, /*vgprs=*/4, /*sgprs=*/4);
  active.globalLoad(/*wave=*/0, /*vgprBase=*/1, /*numRegs=*/1, /*exec=*/1ULL << 5);
  active.checkVgprRead(/*wave=*/0, /*reg=*/1, /*lane=*/5);
  EXPECT_TRUE(active.hasVgprRace(1));

  RaceTestBuilder inactive(/*numWaves=*/1, /*vgprs=*/4, /*sgprs=*/4);
  inactive.globalLoad(/*wave=*/0, /*vgprBase=*/1, /*numRegs=*/1, /*exec=*/1ULL << 5);
  inactive.checkVgprRead(/*wave=*/0, /*reg=*/1, /*lane=*/0);
  EXPECT_FALSE(inactive.hasRace());
}

TEST(RaceDetector, CheckVgprReadLanesReportsMaskedLane) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/4, /*sgprs=*/4);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/1, /*numRegs=*/1, /*exec=*/1ULL << 5);
  b.checkVgprReadLanes(/*wave=*/0, /*reg=*/1, /*laneMask=*/(1ULL << 0) | (1ULL << 5));

  ASSERT_EQ(b.raceCount(), 1);
  const auto &v = b.violations()[0];
  EXPECT_EQ(v.space, RaceViolation::Space::VGPR);
  EXPECT_EQ(v.index, 1);
  EXPECT_EQ(v.lane, 5);
}

// ---- Mixed counter types ----

TEST(RaceDetector, Mixed_VmcntAndLgkmcnt) {
  // Global load (vmcnt) and LDS load (lgkmcnt) in flight simultaneously.
  // waitcnt vmcnt(0) clears global load but not LDS load.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/4, /*sgprs=*/4);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/0, /*numRegs=*/1); // vmcnt
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4,
            /*vgprDst=*/1);           // lgkmcnt
  b.waitcnt(/*wave=*/0, /*vmcnt=*/0); // clear global load only
  b.checkVgprRead(/*wave=*/0, /*reg=*/0,
                  /*lane=*/0); // global load cleared → safe
  EXPECT_FALSE(b.hasVgprRace(0));
  b.checkVgprRead(/*wave=*/0, /*reg=*/1,
                  /*lane=*/0); // LDS load still pending → RACE
  EXPECT_TRUE(b.hasVgprRace(1));
}

TEST(RaceDetector, Mixed_LgkmcntAndVmcnt) {
  // Reverse: waitcnt lgkmcnt(0) clears LDS load but not global load.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/4, /*sgprs=*/4);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/0, /*numRegs=*/1); // vmcnt
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4,
            /*vgprDst=*/1);                           // lgkmcnt
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0); // clear LDS load only
  b.checkVgprRead(/*wave=*/0, /*reg=*/1, /*lane=*/0); // LDS load cleared → safe
  EXPECT_FALSE(b.hasVgprRace(1));
  b.checkVgprRead(/*wave=*/0, /*reg=*/0,
                  /*lane=*/0); // global load still pending → RACE
  EXPECT_TRUE(b.hasVgprRace(0));
}

TEST(RaceDetector, Mixed_BothCleared) {
  // Both vmcnt(0) and lgkmcnt(0) → everything safe.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/4, /*sgprs=*/4);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/0, /*numRegs=*/1);
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4, /*vgprDst=*/1);
  b.scalarLoad(/*wave=*/0, /*sgprBase=*/0, /*numRegs=*/1);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/0, /*lgkmcnt=*/0);
  b.checkVgprRead(/*wave=*/0, /*reg=*/0, /*lane=*/0);
  b.checkVgprRead(/*wave=*/0, /*reg=*/1, /*lane=*/0);
  b.checkSgprRead(/*wave=*/0, /*reg=*/0);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, EventRegistry_NarrowWindow) {
  // Circular buffer: 4 VGPRs loaded in rotation. Each iteration drains
  // the oldest, checks that it's safe, checks that others still race,
  // then reloads into the freed register.
  int N = 10;
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/N, /*sgprs=*/4);

  for (int i = 0; i < N; ++i) {
    b.globalLoad(/*wave=*/0, /*vgprBase=*/i, /*numRegs=*/1);
  }

  for (int iter = 0; iter < 1000; ++iter) {
    int drained = iter % N;
    int stillPending0 = (iter + 1) % N;
    int stillPending1 = (iter + N + 1) % N;

    b.waitcnt(/*wave=*/0, /*vmcnt=*/N - 1);

    b.clearViolations();
    b.checkVgprRead(/*wave=*/0, /*reg=*/drained, /*lane=*/0);
    EXPECT_FALSE(b.hasVgprRace(drained)) << "iter=" << iter;

    b.clearViolations();
    b.checkVgprRead(/*wave=*/0, /*reg=*/stillPending0, /*lane=*/0);
    EXPECT_TRUE(b.hasVgprRace(stillPending0)) << "iter=" << iter;

    b.clearViolations();
    b.checkVgprRead(/*wave=*/0, /*reg=*/stillPending1, /*lane=*/0);
    EXPECT_TRUE(b.hasVgprRace(stillPending1)) << "iter=" << iter;

    b.globalLoad(/*wave=*/0, /*vgprBase=*/drained, /*numRegs=*/1);
  }
}

// ---- EventRegistry trimming ----

TEST(RaceDetector, EventRegistry_TrimWithoutBarrier) {
  // Single wave, no LDS, no barrier. WAVE_COMPLETE non-LDS events are
  // trimmable. Trimming is triggered automatically inside add() every
  // kTrimAttemptInterval allocations.
  int N = EventRegistry::kTrimAttemptInterval - 1;
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/2, /*sgprs=*/2);

  // Allocate N events (one short of the trim interval) — no trim yet.
  for (int i = 0; i < N; ++i) {
    b.globalLoad(/*wave=*/0, /*vgprBase=*/0, /*numRegs=*/1);
    b.waitcnt(/*wave=*/0, /*vmcnt=*/0);
  }
  EXPECT_EQ(b.events().totalAllocated(), N);
  EXPECT_EQ(b.events().trimmedCount(), 0);

  // One more allocation triggers trimming.
  b.globalLoad(/*wave=*/0, /*vgprBase=*/0, /*numRegs=*/1);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/0);

  EXPECT_EQ(b.events().totalAllocated(), N + 1);
  EXPECT_GT(b.events().trimmedCount(), 0);
  EXPECT_LT(b.events().size(), N);

  // Race detection still works after trimming.
  b.globalLoad(/*wave=*/0, /*vgprBase=*/1, /*numRegs=*/1);
  b.checkVgprRead(/*wave=*/0, /*reg=*/1, /*lane=*/0);
  EXPECT_TRUE(b.hasVgprRace(1));
}

TEST(RaceDetector, EventRegistry_TrimWithBarrier) {
  // Two waves, LDS events require s_barrier to reach RETIRED.
  // Trimming is triggered inside add() after enough allocations.
  int N = EventRegistry::kTrimAttemptInterval - 1;
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/4, /*sgprs=*/4);

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
// The race handler fires for every (register, lane) combination where a race
// exists. When the plugin wraps this with deduplication (by PC), it must
// handle millions of handler calls efficiently. These tests verify the raw
// handler invocation count at the core level.

TEST(RaceDetector, HandlerFiresPerLane) {
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/256, /*sgprs=*/32);

  // Global load into v0 with full exec mask (all 64 lanes active).
  b.globalLoad(/*wave=*/0, /*vgprBase=*/0, /*numRegs=*/1);

  // Check v0 across all 64 lanes without waitcnt — each lane fires the handler.
  for (int lane = 0; lane < 64; ++lane)
    b.checkVgprRead(/*wave=*/0, /*reg=*/0, lane);

  EXPECT_EQ(b.raceCount(), 64);
}

TEST(RaceDetector, HandlerFiresPerRegister) {
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/256, /*sgprs=*/32);

  // Load into v0..v3 (4 registers).
  b.globalLoad(/*wave=*/0, /*vgprBase=*/0, /*numRegs=*/4);

  // Read all 4 without waitcnt.
  for (int reg = 0; reg < 4; ++reg)
    b.checkVgprRead(/*wave=*/0, reg, /*lane=*/0);

  EXPECT_EQ(b.raceCount(), 4);
}

TEST(RaceDetector, MultipleDistinctRaces) {
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/256, /*sgprs=*/32);

  // Two independent races at different PCs.
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4, /*vgprDst=*/10);
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/16, /*bytes=*/4, /*vgprDst=*/20);

  // Read both without waitcnt — two distinct races.
  b.checkVgprRead(/*wave=*/0, /*reg=*/10, /*lane=*/0);
  b.checkVgprRead(/*wave=*/0, /*reg=*/20, /*lane=*/0);

  EXPECT_EQ(b.raceCount(), 2);
  EXPECT_TRUE(b.hasVgprRace(10));
  EXPECT_TRUE(b.hasVgprRace(20));
}

TEST(RaceDetector, RepeatedCheckSameRace) {
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/256, /*sgprs=*/32);

  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4, /*vgprDst=*/0);

  // Check the same register+lane 100 times. The handler fires every time
  // because the core race detector does not deduplicate — that is the
  // plugin's responsibility.
  for (int i = 0; i < 100; ++i)
    b.checkVgprRead(/*wave=*/0, /*reg=*/0, /*lane=*/0);

  EXPECT_EQ(b.raceCount(), 100);
}

// ---- Split barriers (s_barrier_signal / s_barrier_wait) ----
//
// A barrier orders an access before it only if the issuing wave completed the
// access before its own signal, and orders an access after it only if the
// accessing wave issued it after its own wait. The signal carries no memory
// wait (CDNA5 ISA 15.3, S_BARRIER_SIGNAL), and a wave waits only for the round
// that set its completion bit (CDNA5 ISA 5.6.1).

namespace {

/// The LDS bytes the split-barrier tests race on.
constexpr int kX = 64;
constexpr int kY = 128;
constexpr int kWave32 = 32;

void dsWrite(RaceTestBuilder &b, int wave, int addr = kX) {
  b.ldsWrite(wave, /*lane=*/0, addr, /*bytes=*/4, MemoryOrderClass::LDS,
             amdgpu::WaitCounterType::DSCNT);
}

void dsRead(RaceTestBuilder &b, int wave, int vgprDst, int addr = kX) {
  b.ldsRead(wave, /*lane=*/0, addr, /*bytes=*/4, vgprDst, /*byteMask=*/0xF,
            amdgpu::WaitCounterType::DSCNT);
}

/// Exactly one LDS race, by `wave`'s access of the byte at `addr`.
void expectOneLdsRace(const RaceTestBuilder &b, int wave, int addr, bool isWrite) {
  ASSERT_EQ(b.raceCount(), 1);
  const RaceViolation &violation = b.violations().front();
  EXPECT_EQ(violation.space, RaceViolation::Space::LDS);
  EXPECT_EQ(violation.index, addr);
  EXPECT_EQ(violation.wave, wave);
  EXPECT_EQ(violation.isWrite, isWrite);
}

/// Every listed wave signals `round`, then the round completes.
void signalAndComplete(RaceTestBuilder &b, BarrierRoundId round, const std::vector<int> &members) {
  for (int wave : members)
    b.signal(wave, round);
  b.completeRound(round, members);
}

} // namespace

TEST(RaceDetector, SplitBarrier_WaitedBeforeSignalOrdersReaderAfterWait) {
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/8, /*sgprs=*/8, kWave32);
  const BarrierRoundId round = RaceTestBuilder::workgroupRound(0);
  dsWrite(b, 0);
  b.waitDscnt(0, 0);
  signalAndComplete(b, round, {0, 1});
  b.waitPassed(1, round);
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, kX, /*bytes=*/4);
  EXPECT_EQ(b.raceCount(), 0);
}

// The write completes after the round, so nothing released it.
TEST(RaceDetector, SplitBarrier_CompletedAfterRoundCompletesRaces) {
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/8, /*sgprs=*/8, kWave32);
  const BarrierRoundId round = RaceTestBuilder::workgroupRound(0);
  dsWrite(b, 0);
  signalAndComplete(b, round, {0, 1});
  b.waitDscnt(0, 0);
  b.waitPassed(1, round);
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, kX, /*bytes=*/4);
  expectOneLdsRace(b, /*wave=*/1, kX, /*isWrite=*/false);
}

// The write completes after its wave's signal but before the round completes.
// The signal released only what had completed when it executed.
TEST(RaceDetector, SplitBarrier_CompletedAfterSignalBeforeRoundCompletesRaces) {
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/8, /*sgprs=*/8, kWave32);
  const BarrierRoundId round = RaceTestBuilder::workgroupRound(0);
  dsWrite(b, 0);
  b.signal(0, round);
  b.waitDscnt(0, 0);
  b.signal(1, round);
  b.completeRound(round, {0, 1});
  b.waitPassed(1, round);
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, kX, /*bytes=*/4);
  expectOneLdsRace(b, /*wave=*/1, kX, /*isWrite=*/false);
}

// The reader's access precedes its own wait, so it acquired nothing.
TEST(RaceDetector, SplitBarrier_ReadBeforeOwnWaitRaces) {
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/8, /*sgprs=*/8, kWave32);
  const BarrierRoundId round = RaceTestBuilder::workgroupRound(0);
  dsWrite(b, 0);
  b.waitDscnt(0, 0);
  signalAndComplete(b, round, {0, 1});
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, kX, /*bytes=*/4);
  b.waitPassed(1, round);
  expectOneLdsRace(b, /*wave=*/1, kX, /*isWrite=*/false);
}

TEST(RaceDetector, SplitBarrier_ReaderAfterWaitIsSafeWhileWriterInWindow) {
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/8, /*sgprs=*/8, kWave32);
  const BarrierRoundId round = RaceTestBuilder::workgroupRound(0);
  dsWrite(b, 0);
  b.waitDscnt(0, 0);
  signalAndComplete(b, round, {0, 1});
  b.waitPassed(1, round); // Wave 0 has not passed its wait.
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, kX, /*bytes=*/4);
  EXPECT_EQ(b.raceCount(), 0);
}

// Acquisition is per wave: wave 1 has passed its wait and wave 2 has not.
TEST(RaceDetector, SplitBarrier_MemberStillInWindowRaces) {
  RaceTestBuilder b(/*numWaves=*/3, /*vgprs=*/8, /*sgprs=*/8, kWave32);
  const BarrierRoundId round = RaceTestBuilder::workgroupRound(0);
  dsWrite(b, 0);
  b.waitDscnt(0, 0);
  signalAndComplete(b, round, {0, 1, 2});
  b.waitPassed(1, round);
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, kX, /*bytes=*/4);
  EXPECT_EQ(b.raceCount(), 0);
  b.checkLdsRead(/*wave=*/2, /*lane=*/0, kX, /*bytes=*/4);
  expectOneLdsRace(b, /*wave=*/2, kX, /*isWrite=*/false);
}

// An event leaves the LDS history only once no member can still race with it.
TEST(RaceDetector, SplitBarrier_RetiresOnlyAfterLastMemberWaits) {
  RaceTestBuilder b(/*numWaves=*/3, /*vgprs=*/8, /*sgprs=*/8, kWave32);
  const BarrierRoundId round = RaceTestBuilder::workgroupRound(0);
  dsWrite(b, 0);
  b.waitDscnt(0, 0);
  signalAndComplete(b, round, {0, 1, 2});
  EXPECT_EQ(b.ldsWriteEventCount(), 1u);
  EXPECT_EQ(b.openBarrierRounds(), 1u);
  b.waitPassed(0, round);
  b.waitPassed(1, round);
  EXPECT_EQ(b.ldsWriteEventCount(), 1u);
  EXPECT_EQ(b.openBarrierRounds(), 1u);
  b.waitPassed(2, round);
  EXPECT_EQ(b.ldsWriteEventCount(), 0u);
  EXPECT_EQ(b.barrierPendingEventCount(0), 0u);
  EXPECT_EQ(b.openBarrierRounds(), 0u);
}

// A write completed in round 0's window is released by its wave's next signal.
TEST(RaceDetector, SplitBarrier_WindowCompletionReleasedByNextRound) {
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/8, /*sgprs=*/8, kWave32);
  const BarrierRoundId round0 = RaceTestBuilder::workgroupRound(0);
  const BarrierRoundId round1 = RaceTestBuilder::workgroupRound(1);
  dsWrite(b, 0);
  b.signal(0, round0);
  b.waitDscnt(0, 0);
  b.signal(1, round0);
  b.completeRound(round0, {0, 1});
  b.waitPassed(0, round0);
  b.waitPassed(1, round0);
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, kX, /*bytes=*/4);
  expectOneLdsRace(b, /*wave=*/1, kX, /*isWrite=*/false);

  b.clearViolations();
  signalAndComplete(b, round1, {0, 1});
  b.waitPassed(0, round1);
  b.waitPassed(1, round1);
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, kX, /*bytes=*/4);
  EXPECT_EQ(b.raceCount(), 0);
}

TEST(RaceDetector, SplitBarrier_WarCompletedBeforeSignalIsSafe) {
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/8, /*sgprs=*/8, kWave32);
  const BarrierRoundId round = RaceTestBuilder::workgroupRound(0);
  dsRead(b, 1, /*vgprDst=*/0);
  b.waitDscnt(1, 0);
  signalAndComplete(b, round, {1, 0});
  b.waitPassed(0, round);
  b.checkLdsWrite(/*wave=*/0, /*lane=*/0, kX, /*bytes=*/4);
  EXPECT_EQ(b.raceCount(), 0);
}

TEST(RaceDetector, SplitBarrier_WarCompletedAfterSignalRaces) {
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/8, /*sgprs=*/8, kWave32);
  const BarrierRoundId round = RaceTestBuilder::workgroupRound(0);
  dsRead(b, 1, /*vgprDst=*/0);
  b.signal(1, round);
  b.waitDscnt(1, 0);
  b.signal(0, round);
  b.completeRound(round, {1, 0});
  b.waitPassed(0, round);
  b.checkLdsWrite(/*wave=*/0, /*lane=*/0, kX, /*bytes=*/4);
  expectOneLdsRace(b, /*wave=*/0, kX, /*isWrite=*/true);
}

// A write its wave never waited for is not released by any signal.
TEST(RaceDetector, SplitBarrier_ActiveAtSignalNeverReleased) {
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/8, /*sgprs=*/8, kWave32);
  const BarrierRoundId round = RaceTestBuilder::workgroupRound(0);
  dsWrite(b, 0);
  signalAndComplete(b, round, {0, 1});
  b.waitPassed(0, round);
  b.waitPassed(1, round);
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, kX, /*bytes=*/4);
  expectOneLdsRace(b, /*wave=*/1, kX, /*isWrite=*/false);
}

// Wave 1's completion bit was set by round 0, and a flag does not count, so
// its wait acquires round 0 and not round 1 (CDNA5 ISA 5.6.1).
TEST(RaceDetector, SplitBarrier_WaitPassedForOlderRoundAcquiresOnlyIt) {
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/8, /*sgprs=*/8, kWave32);
  const BarrierRoundId round0 = RaceTestBuilder::workgroupRound(0);
  const BarrierRoundId round1 = RaceTestBuilder::workgroupRound(1);
  dsWrite(b, 0, kX);
  b.waitDscnt(0, 0);
  signalAndComplete(b, round0, {0, 1});
  b.waitPassed(0, round0);
  dsWrite(b, 0, kY);
  b.waitDscnt(0, 0);
  signalAndComplete(b, round1, {0, 1});
  b.waitPassed(1, round0);
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, kX, /*bytes=*/4);
  EXPECT_EQ(b.raceCount(), 0);
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, kY, /*bytes=*/4);
  expectOneLdsRace(b, /*wave=*/1, kY, /*isWrite=*/false);
}

enum class SplitBarrierTransfer { DsWrite, TensorLoad, TensorStore };

/// Issue `transfer` at kX from wave 0. Returns whether wave 1's conflicting
/// access is a write (a tensor store reads LDS).
bool issueTransfer(RaceTestBuilder &b, SplitBarrierTransfer transfer) {
  switch (transfer) {
  case SplitBarrierTransfer::DsWrite:
    dsWrite(b, 0);
    return false;
  case SplitBarrierTransfer::TensorLoad:
    b.tensorLoadToLds(0, {{.address = kX, .bytes = 16}});
    return false;
  case SplitBarrierTransfer::TensorStore:
    b.tensorStoreFromLds(0, {{.address = kX, .bytes = 16}});
    return true;
  }
  return false;
}

void completeTransfer(RaceTestBuilder &b, SplitBarrierTransfer transfer) {
  if (transfer == SplitBarrierTransfer::DsWrite)
    b.waitDscnt(0, 0);
  else
    b.waitTensorcnt(0, 0);
}

constexpr std::array kSplitBarrierTransfers{SplitBarrierTransfer::DsWrite,
                                            SplitBarrierTransfer::TensorLoad,
                                            SplitBarrierTransfer::TensorStore};

TEST(RaceDetector, SplitBarrier_TransfersCompletedBeforeSignalAreOrdered) {
  for (SplitBarrierTransfer transfer : kSplitBarrierTransfers) {
    SCOPED_TRACE(static_cast<int>(transfer));
    RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/8, /*sgprs=*/8, kWave32);
    const BarrierRoundId round = RaceTestBuilder::workgroupRound(0);
    const bool consumerWrites = issueTransfer(b, transfer);
    completeTransfer(b, transfer);
    signalAndComplete(b, round, {0, 1});
    b.waitPassed(1, round);
    if (consumerWrites)
      b.checkLdsWrite(/*wave=*/1, /*lane=*/0, kX, /*bytes=*/4);
    else
      b.checkLdsRead(/*wave=*/1, /*lane=*/0, kX, /*bytes=*/4);
    EXPECT_EQ(b.raceCount(), 0);
  }
}

TEST(RaceDetector, SplitBarrier_TransfersCompletedAfterSignalRace) {
  for (SplitBarrierTransfer transfer : kSplitBarrierTransfers) {
    SCOPED_TRACE(static_cast<int>(transfer));
    RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/8, /*sgprs=*/8, kWave32);
    const BarrierRoundId round = RaceTestBuilder::workgroupRound(0);
    const bool consumerWrites = issueTransfer(b, transfer);
    b.signal(0, round);
    completeTransfer(b, transfer);
    b.signal(1, round);
    b.completeRound(round, {0, 1});
    b.waitPassed(1, round);
    if (consumerWrites)
      b.checkLdsWrite(/*wave=*/1, /*lane=*/0, kX, /*bytes=*/4);
    else
      b.checkLdsRead(/*wave=*/1, /*lane=*/0, kX, /*bytes=*/4);
    expectOneLdsRace(b, /*wave=*/1, kX, consumerWrites);
  }
}

// Counter-capacity backpressure completes the oldest write before the signal,
// so the signal releases it; the newest write is still in flight.
TEST(RaceDetector, SplitBarrier_CapacityCompletionBeforeSignalIsReleased) {
  constexpr int kLgkmcntCapacity = counterCapacitiesForArch(ROCJITSU_CODE_ARCH_CDNA4).lgkmcnt;
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/8, /*sgprs=*/8, kWave32);
  const BarrierRoundId round = RaceTestBuilder::workgroupRound(0);
  // One more write than the counter holds: issuing the last one completes
  // the first.
  for (int index = 0; index <= kLgkmcntCapacity; ++index)
    b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/index * 4, /*bytes=*/4);
  signalAndComplete(b, round, {0, 1});
  b.waitPassed(1, round);
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  EXPECT_EQ(b.raceCount(), 0);
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/kLgkmcntCapacity * 4, /*bytes=*/4);
  expectOneLdsRace(b, /*wave=*/1, kLgkmcntCapacity * 4, /*isWrite=*/false);
}

// Wave 1 acquires wave 0's write on named barrier 1 and releases it to wave 2
// on named barrier 2.
TEST(RaceDetector, SplitBarrier_TransitiveThroughNamedRounds) {
  RaceTestBuilder b(/*numWaves=*/3, /*vgprs=*/8, /*sgprs=*/8, kWave32);
  const BarrierRoundId first = RaceTestBuilder::namedRound(1, 0);
  const BarrierRoundId second = RaceTestBuilder::namedRound(2, 0);
  dsWrite(b, 0);
  b.waitDscnt(0, 0);
  signalAndComplete(b, first, {0, 1});
  b.waitPassed(0, first);
  b.waitPassed(1, first);
  signalAndComplete(b, second, {1, 2});
  b.waitPassed(1, second);
  b.waitPassed(2, second);
  b.checkLdsRead(/*wave=*/2, /*lane=*/0, kX, /*bytes=*/4);
  EXPECT_EQ(b.raceCount(), 0);
}

// Wave 2 is not a member of the named round, so only a later workgroup round
// orders wave 0's write for it.
TEST(RaceDetector, SplitBarrier_NamedRoundNonMemberRaces) {
  RaceTestBuilder b(/*numWaves=*/3, /*vgprs=*/8, /*sgprs=*/8, kWave32);
  const BarrierRoundId named = RaceTestBuilder::namedRound(1, 0);
  const BarrierRoundId workgroup = RaceTestBuilder::workgroupRound(0);
  dsWrite(b, 0);
  b.waitDscnt(0, 0);
  signalAndComplete(b, named, {0, 1});
  b.waitPassed(0, named);
  b.waitPassed(1, named);
  b.checkLdsRead(/*wave=*/2, /*lane=*/0, kX, /*bytes=*/4);
  expectOneLdsRace(b, /*wave=*/2, kX, /*isWrite=*/false);

  b.clearViolations();
  signalAndComplete(b, workgroup, {0, 1, 2});
  for (int wave : {0, 1, 2})
    b.waitPassed(wave, workgroup);
  b.checkLdsRead(/*wave=*/2, /*lane=*/0, kX, /*bytes=*/4);
  EXPECT_EQ(b.raceCount(), 0);
}

// A workgroup round cannot complete until every wave has signalled or ended
// (CDNA5 ISA 5.6), so it orders an ended wave's completed write.
TEST(RaceDetector, SplitBarrier_EndedWaveReleasedToNextWorkgroupRound) {
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/8, /*sgprs=*/8, kWave32);
  const BarrierRoundId round = RaceTestBuilder::workgroupRound(0);
  dsWrite(b, 0);
  b.endWave(0);
  signalAndComplete(b, round, {1});
  b.waitPassed(1, round);
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, kX, /*bytes=*/4);
  EXPECT_EQ(b.raceCount(), 0);
}

// The ended wave's write is released into the round, which wave 1 acquires
// only at its wait.
TEST(RaceDetector, SplitBarrier_EndedWaveReleasedOnlyAtWait) {
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/8, /*sgprs=*/8, kWave32);
  const BarrierRoundId round = RaceTestBuilder::workgroupRound(0);
  dsWrite(b, 0);
  b.endWave(0);
  signalAndComplete(b, round, {1});
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, kX, /*bytes=*/4);
  b.waitPassed(1, round);
  expectOneLdsRace(b, /*wave=*/1, kX, /*isWrite=*/false);
}

// A named round can complete without the ended wave, so it orders nothing the
// ended wave did.
TEST(RaceDetector, SplitBarrier_EndedWaveNotReleasedToNamedRound) {
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/8, /*sgprs=*/8, kWave32);
  const BarrierRoundId round = RaceTestBuilder::namedRound(1, 0);
  dsWrite(b, 0);
  b.endWave(0);
  signalAndComplete(b, round, {1});
  b.waitPassed(1, round);
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, kX, /*bytes=*/4);
  expectOneLdsRace(b, /*wave=*/1, kX, /*isWrite=*/false);
}

// Wave 2 ends without signalling; the round completes for waves 0 and 1, and
// the write retires once both have passed their waits.
TEST(RaceDetector, SplitBarrier_EndedWithoutSignalDoesNotBlockRetirement) {
  RaceTestBuilder b(/*numWaves=*/3, /*vgprs=*/8, /*sgprs=*/8, kWave32);
  const BarrierRoundId round = RaceTestBuilder::workgroupRound(0);
  dsWrite(b, 0);
  b.waitDscnt(0, 0);
  b.signal(0, round);
  b.signal(1, round);
  b.endWave(2);
  b.completeRound(round, {0, 1});
  EXPECT_EQ(b.ldsWriteEventCount(), 1u);
  b.waitPassed(0, round);
  b.waitPassed(1, round);
  EXPECT_EQ(b.ldsWriteEventCount(), 0u);
  EXPECT_EQ(b.openBarrierRounds(), 0u);
}

// A loop of split rounds leaves no LDS history, no open rounds and no
// untrimmed events behind.
TEST(RaceDetector, SplitBarrier_LoopDoesNotAccumulate) {
  constexpr int kIterations = 3 * EventRegistry::kTrimAttemptInterval;
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/8, /*sgprs=*/8, kWave32);
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    const BarrierRoundId round = RaceTestBuilder::workgroupRound(iteration);
    dsWrite(b, 0);
    b.waitDscnt(0, 0);
    signalAndComplete(b, round, {0, 1});
    b.waitPassed(0, round);
    b.waitPassed(1, round);
  }
  EXPECT_EQ(b.raceCount(), 0);
  EXPECT_EQ(b.ldsWriteEventCount(), 0u);
  EXPECT_EQ(b.barrierPendingEventCount(0), 0u);
  EXPECT_EQ(b.openBarrierRounds(), 0u);
  // Each write retires before the next is issued. The last trim attempt runs
  // when the final write is allocated, while it is still in flight, so every
  // earlier event is trimmed and only that write remains.
  EXPECT_EQ(b.events().totalAllocated(), kIterations);
  EXPECT_EQ(b.events().trimmedCount(), kIterations - 1);
  EXPECT_EQ(b.events().size(), 1);
}
