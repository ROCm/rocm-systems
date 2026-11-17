// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "race_test_builder.h"

#include <gtest/gtest.h>

#include <array>
#include <ostream>

using namespace rocjitsu::plugins::race_detector;
namespace amdgpu = rocjitsu::amdgpu;

namespace {

constexpr int kWave = 0;
constexpr int kLane = 0;
constexpr int kDwordBytes = 4;

// For CDNA1-CDNA4 combined counters, the all-ones no-wait encoding is also the
// maximum number of outstanding tokens. Issuing another token forces older
// work to complete before the new operation can issue.
constexpr auto kCdnaCounterCapacities = counterCapacitiesForArch(ROCJITSU_CODE_ARCH_CDNA4);
constexpr int kLgkmcntCapacity = kCdnaCounterCapacities.lgkmcnt;
constexpr int kVmcntCapacity = kCdnaCounterCapacities.vmcnt;
constexpr int kLgkmcntOverflow = 5;
constexpr int kVmcntOverflow = 7;
constexpr int kClearlyBelowCapacity = 10;

RaceTestBuilder makeCounterTest() { return RaceTestBuilder(1); }

void issueOrderedLdsReads(RaceTestBuilder &builder, int count) {
  for (int offset = 0; offset < count; ++offset) {
    builder.ldsRead(kWave, kLane, /*addr=*/offset * kDwordBytes, /*bytes=*/kDwordBytes);
  }
}

void issueOrderedVmemEvents(RaceTestBuilder &builder, int count) {
  for (int offset = 0; offset < count; ++offset) {
    builder.counterIssue(kWave, amdgpu::WaitCounterType::VMCNT, MemoryOrderClass::VMEM);
  }
}

void issueUnorderedVmcntLoads(RaceTestBuilder &builder, int count) {
  for (int offset = 0; offset < count; ++offset) {
    builder.counterIssue(kWave, amdgpu::WaitCounterType::VMCNT, MemoryOrderClass::UNORDERED);
  }
}

void expectComplete(RaceTestBuilder &builder, int event) {
  EXPECT_NE(builder.events().status(EventId{event}), EventStatus::ACTIVE);
}

void expectPending(RaceTestBuilder &builder, int event) {
  EXPECT_EQ(builder.events().status(EventId{event}), EventStatus::ACTIVE);
}

void expectLdsReady(RaceTestBuilder &builder) {
  builder.checkLdsRead(kWave, kLane, /*addr=*/0, /*bytes=*/kDwordBytes);
  EXPECT_FALSE(builder.hasLdsRace(0));
}

void expectLdsPending(RaceTestBuilder &builder) {
  builder.checkLdsRead(kWave, kLane, /*addr=*/0, /*bytes=*/kDwordBytes);
  EXPECT_TRUE(builder.hasLdsRace(0));
}

} // namespace

// ---- Bounds above current occupancy ----

TEST(RaceDetector, CounterCapacity_LgkmcntBoundAboveOccupancyLeavesEventsPending) {
  auto builder = makeCounterTest();
  issueOrderedLdsReads(builder, kClearlyBelowCapacity);

  builder.waitcnt(kWave, /*vmcnt=*/-1, /*lgkmcnt=*/kLgkmcntCapacity);
  expectPending(builder, /*event=*/0);
}

TEST(RaceDetector, CounterCapacity_VmcntBoundAboveOccupancyLeavesEventsPending) {
  auto builder = makeCounterTest();
  issueOrderedVmemEvents(builder, kClearlyBelowCapacity);

  builder.waitcnt(kWave, /*vmcnt=*/kVmcntCapacity, /*lgkmcnt=*/-1);
  expectPending(builder, /*event=*/0);
}

// ---- Issue-time capacity backpressure ----

TEST(RaceDetector, CounterCapacity_MissingConfiguredCapacityFailsLoudly) {
  RaceDetector detector(/*nWaves=*/
                        1, Dim3d(0), [](RaceViolation) {},
                        CounterCapacities{.vmcnt = 0, .lgkmcnt = 0}, /*ldsSizeBytes=*/64 * 1024);

  EXPECT_TRUE(detector.getWaveRaceState(0)
                  .prepareForCounterIncrement(amdgpu::WaitCounterType::VMCNT)
                  .failed());
  EXPECT_TRUE(detector.getWaveRaceState(0).getWaveMemoryEvents().empty());
}

TEST(RaceDetector, CounterCapacity_LgkmcntOverflowRetiresOldestOrderedReads) {
  auto builder = makeCounterTest();
  issueOrderedLdsReads(builder, kLgkmcntCapacity + kLgkmcntOverflow);

  expectComplete(builder, /*event=*/kLgkmcntOverflow - 1);
  expectPending(builder, /*event=*/kLgkmcntOverflow);
}

// Backpressure completes the oldest DS reads like a wait does, so a
// single-wave workgroup retires their LDS events at once, while a workgroup
// with another wave keeps them for a barrier.
TEST(RaceDetector, CounterCapacity_LgkmcntOverflowRetiresLdsEventsOnlyInASingleWaveWorkgroup) {
  for (const int numWaves : {1, 2}) {
    SCOPED_TRACE(numWaves);
    RaceTestBuilder builder(numWaves);
    issueOrderedLdsReads(builder, kLgkmcntCapacity + kLgkmcntOverflow);

    const auto completed = numWaves == 1 ? EventStatus::RETIRED : EventStatus::WAVE_COMPLETE;
    EXPECT_EQ(builder.events().status(EventId{kLgkmcntOverflow - 1}), completed);
    EXPECT_EQ(builder.events().status(EventId{kLgkmcntOverflow}), EventStatus::ACTIVE);
  }
}

TEST(RaceDetector, CounterCapacity_VmcntOverflowRetiresOldestOrderedLoads) {
  auto builder = makeCounterTest();
  issueOrderedVmemEvents(builder, kVmcntCapacity + kVmcntOverflow);

  expectComplete(builder, /*event=*/kVmcntOverflow - 1);
  expectPending(builder, /*event=*/kVmcntOverflow);
}

TEST(RaceDetector, CounterCapacity_FullLgkmcntRetiresBeforeCurrentIssue) {
  auto builder = makeCounterTest();
  issueOrderedLdsReads(builder, kLgkmcntCapacity);

  builder.prepareCounterIncrement(kWave, amdgpu::WaitCounterType::LGKMCNT);
  expectComplete(builder, /*event=*/0);
}

TEST(RaceDetector, CounterCapacity_LgkmcntBelowCapacityLeavesOldestEventPending) {
  auto builder = makeCounterTest();
  issueOrderedLdsReads(builder, kLgkmcntCapacity - 1);

  builder.prepareCounterIncrement(kWave, amdgpu::WaitCounterType::LGKMCNT);
  expectPending(builder, /*event=*/0);
}

TEST(RaceDetector, CounterCapacity_FullVmcntRetiresBeforeCurrentIssue) {
  auto builder = makeCounterTest();
  issueOrderedVmemEvents(builder, kVmcntCapacity);

  builder.prepareCounterIncrement(kWave, amdgpu::WaitCounterType::VMCNT);
  expectComplete(builder, /*event=*/0);
}

TEST(RaceDetector, CounterCapacity_VmcntBelowCapacityLeavesOldestEventPending) {
  auto builder = makeCounterTest();
  issueOrderedVmemEvents(builder, kVmcntCapacity - 1);

  builder.prepareCounterIncrement(kWave, amdgpu::WaitCounterType::VMCNT);
  expectPending(builder, /*event=*/0);
}

TEST(RaceDetector, CounterCapacity_LoadcntAliasUsesVmcntCapacity) {
  auto builder = makeCounterTest();
  issueOrderedVmemEvents(builder, kVmcntCapacity);

  builder.prepareCounterIncrement(kWave, amdgpu::WaitCounterType::LOADCNT);
  expectComplete(builder, /*event=*/0);
}

TEST(RaceDetector, CounterCapacity_SplitLgkmAliasesUseCombinedCapacity) {
  for (const auto counter : {amdgpu::WaitCounterType::DSCNT, amdgpu::WaitCounterType::KMCNT}) {
    SCOPED_TRACE(static_cast<int>(counter));
    auto builder = makeCounterTest();
    issueOrderedLdsReads(builder, kLgkmcntCapacity);

    builder.prepareCounterIncrement(kWave, counter);
    expectComplete(builder, /*event=*/0);
  }
}

TEST(RaceDetector, CounterCapacity_StorecntDoesNotAdvanceVmcnt) {
  auto builder = makeCounterTest();
  issueOrderedVmemEvents(builder, kVmcntCapacity);

  builder.prepareCounterIncrement(kWave, amdgpu::WaitCounterType::STORECNT);
  expectPending(builder, /*event=*/0);
}

TEST(RaceDetector, CounterCapacity_GenericFlatAdvancesBothCounterDomains) {
  amdgpu::MemoryIssueInfo flatIssue;
  flatIssue.counter_obligations_[0] = {amdgpu::WaitCounterType::LOADCNT,
                                       amdgpu::MemoryCompletionClass::VMEM};
  flatIssue.counter_obligations_[1] = {amdgpu::WaitCounterType::DSCNT,
                                       amdgpu::MemoryCompletionClass::LDS};
  flatIssue.num_counter_obligations_ = 2;

  auto vmemBuilder = makeCounterTest();
  issueOrderedVmemEvents(vmemBuilder, kVmcntCapacity);
  vmemBuilder.prepareMemoryIssue(kWave, flatIssue);
  expectComplete(vmemBuilder, /*event=*/0);

  auto ldsBuilder = makeCounterTest();
  issueOrderedLdsReads(ldsBuilder, kLgkmcntCapacity);
  ldsBuilder.prepareMemoryIssue(kWave, flatIssue);
  expectComplete(ldsBuilder, /*event=*/0);
}

// ---- Mixed completion-order classes ----

TEST(RaceDetector, CounterCapacity_FullLgkmcntDoesNotProveScalarLoadComplete) {
  auto builder = makeCounterTest();
  builder.counterIssue(kWave, amdgpu::WaitCounterType::LGKMCNT, MemoryOrderClass::UNORDERED);
  issueOrderedLdsReads(builder, kLgkmcntCapacity);

  expectPending(builder, 0);
}

TEST(RaceDetector, CounterCapacity_ScalarIssueAdvancesFullOrderedLgkmcntClass) {
  auto builder = makeCounterTest();
  issueOrderedLdsReads(builder, kLgkmcntCapacity);

  builder.counterIssue(kWave, amdgpu::WaitCounterType::LGKMCNT, MemoryOrderClass::UNORDERED);
  expectComplete(builder, /*event=*/0);
}

TEST(RaceDetector, CounterCapacity_WideScalarIssueContributesTwoLgkmcntTokens) {
  auto builder = makeCounterTest();
  issueOrderedLdsReads(builder, kLgkmcntCapacity - 1);

  builder.counterIssue(kWave, amdgpu::WaitCounterType::LGKMCNT, MemoryOrderClass::UNORDERED, 2);
  expectComplete(builder, /*event=*/0);
}

TEST(RaceDetector, CounterCapacity_MixedLgkmcntDoesNotGuessWhichEventCompleted) {
  auto builder = makeCounterTest();
  issueOrderedLdsReads(builder, kLgkmcntCapacity - 1);
  builder.counterIssue(kWave, amdgpu::WaitCounterType::LGKMCNT, MemoryOrderClass::UNORDERED);

  // The next scalar load needs one slot, but the completed event could be the
  // first scalar load. It does not prove that the oldest LDS read completed.
  builder.counterIssue(kWave, amdgpu::WaitCounterType::LGKMCNT, MemoryOrderClass::UNORDERED);
  expectPending(builder, /*event=*/0);
}

TEST(RaceDetector, CounterCapacity_InterleavedLgkmcntSoakRetiresOnlyProvableLdsPrefix) {
  auto builder = makeCounterTest();
  // These candidates can outnumber hardware slots: no nonzero wait can
  // identify which unordered events completed.
  constexpr std::array<int, kLgkmcntCapacity> unorderedBursts = {
      3, 0, 4, 1, 2, 5, 0, 3, 1, 4, 2, 0, 5, 1, 3,
  };
  std::array<int, kLgkmcntCapacity> orderedEvents{};
  for (int ordered = 0; ordered < kLgkmcntCapacity; ++ordered) {
    for (int unordered = 0; unordered < unorderedBursts[ordered]; ++unordered)
      builder.counterIssue(kWave, amdgpu::WaitCounterType::LGKMCNT, MemoryOrderClass::UNORDERED);
    orderedEvents[ordered] = builder.events().totalAllocated();
    builder.ldsRead(kWave, kLane, ordered * kDwordBytes, kDwordBytes);
  }

  expectPending(builder, orderedEvents[0]);
  // Reserving the next token proves completion of exactly the oldest LDS event.
  builder.counterIssue(kWave, amdgpu::WaitCounterType::LGKMCNT, MemoryOrderClass::UNORDERED);
  expectComplete(builder, orderedEvents[0]);
  expectPending(builder, orderedEvents[1]);
}

TEST(RaceDetector, CounterCapacity_FullVmcntDoesNotProveUnorderedLoadComplete) {
  auto builder = makeCounterTest();
  issueUnorderedVmcntLoads(builder, /*count=*/1);
  issueOrderedVmemEvents(builder, kVmcntCapacity);

  expectPending(builder, /*event=*/0);
}

TEST(RaceDetector, CounterCapacity_UnorderedLoadsDoNotAdvanceOrderedVmcntClass) {
  auto builder = makeCounterTest();
  issueOrderedVmemEvents(builder, /*count=*/1);
  issueUnorderedVmcntLoads(builder, kVmcntCapacity);

  expectPending(builder, /*event=*/0);
}

TEST(RaceDetector, CounterCapacity_UnorderedIssueAdvancesFullOrderedVmcntClass) {
  auto builder = makeCounterTest();
  issueOrderedVmemEvents(builder, kVmcntCapacity);

  issueUnorderedVmcntLoads(builder, /*count=*/1);
  expectComplete(builder, /*event=*/0);
}

// ---- Explicit partial waits after capacity backpressure ----

TEST(RaceDetector, CounterCapacity_PartialLgkmcntKeepsUnorderedScalarLoadPending) {
  auto builder = makeCounterTest();
  issueOrderedLdsReads(builder, kLgkmcntCapacity);
  builder.counterIssue(kWave, amdgpu::WaitCounterType::LGKMCNT, MemoryOrderClass::UNORDERED);

  builder.waitcnt(kWave, /*vmcnt=*/-1, /*lgkmcnt=*/kLgkmcntCapacity - 1);
  expectComplete(builder, /*event=*/0);
  expectPending(builder, builder.events().totalAllocated() - 1);
}

TEST(RaceDetector, CounterCapacity_PartialLgkmcntPreservesAmbiguousEventsWithinLimit) {
  auto builder = makeCounterTest();
  issueOrderedLdsReads(builder, kLgkmcntCapacity - 1);
  builder.counterIssue(kWave, amdgpu::WaitCounterType::LGKMCNT, MemoryOrderClass::UNORDERED);

  builder.waitcnt(kWave, /*vmcnt=*/-1, /*lgkmcnt=*/kLgkmcntCapacity - 1);
  expectPending(builder, /*event=*/0);
  expectPending(builder, builder.events().totalAllocated() - 1);
}

TEST(RaceDetector, CounterCapacity_PartialLgkmcntRetiresNextOrderedRead) {
  auto builder = makeCounterTest();
  issueOrderedLdsReads(builder, kLgkmcntCapacity + kLgkmcntOverflow);

  builder.waitcnt(kWave, /*vmcnt=*/-1, /*lgkmcnt=*/kLgkmcntCapacity - 1);
  expectComplete(builder, /*event=*/kLgkmcntOverflow);
  expectPending(builder, /*event=*/kLgkmcntOverflow + 1);
}

TEST(RaceDetector, CounterCapacity_PartialVmcntRetiresNextOrderedLoad) {
  auto builder = makeCounterTest();
  issueOrderedVmemEvents(builder, kVmcntCapacity + kVmcntOverflow);

  builder.waitcnt(kWave, /*vmcnt=*/kVmcntCapacity - 1, /*lgkmcnt=*/-1);
  expectComplete(builder, /*event=*/kVmcntOverflow);
  expectPending(builder, /*event=*/kVmcntOverflow + 1);
}

// ---- Other VMCNT producers ----

TEST(RaceDetector, CounterCapacity_CombinedVmemStoresAdvanceVmcntCapacity) {
  RaceTestBuilder builder(/*numWaves=*/1);
  builder.counterIssue(kWave, amdgpu::WaitCounterType::VMCNT, MemoryOrderClass::VMEM);
  issueOrderedVmemEvents(builder, kVmcntCapacity);

  expectComplete(builder, /*event=*/0);
}

TEST(RaceDetector, CounterCapacity_CombinedVmemLoadRemainsPendingBeforeStoreCapacity) {
  RaceTestBuilder builder(/*numWaves=*/1);
  builder.counterIssue(kWave, amdgpu::WaitCounterType::VMCNT, MemoryOrderClass::VMEM);
  issueOrderedVmemEvents(builder, kVmcntCapacity - 1);

  expectPending(builder, /*event=*/0);
}

TEST(RaceDetector, CounterCapacity_VmcntCompletesDirectToLdsForOwningWave) {
  auto builder = makeCounterTest();
  builder.globalToLds(kWave, /*ldsAddrs=*/{0}, /*bytesPerLane=*/kDwordBytes, /*exec=*/1);
  issueOrderedVmemEvents(builder, kVmcntCapacity);

  expectLdsReady(builder);
}

TEST(RaceDetector, CounterCapacity_DirectToLdsRemainsPendingBelowVmcntCapacity) {
  auto builder = makeCounterTest();
  builder.globalToLds(kWave, /*ldsAddrs=*/{0}, /*bytesPerLane=*/kDwordBytes, /*exec=*/1);
  issueOrderedVmemEvents(builder, kVmcntCapacity - 1);

  expectLdsPending(builder);
}

// ---- gfx1250 split counters ----
//
// gfx1250 tracks memory operations on six split counters, each stalling issue
// rather than overflowing (CDNA5 ISA 5.7.1). Their capacities come from the
// wait policy that the core memory-wait checker also uses: the largest
// non-sentinel wait value plus one, so 63 for 6-bit counters and 31 for KMcnt.

namespace {

int gfx1250Capacity(amdgpu::WaitCounterType counter) {
  return counterCapacitiesForArch(ROCJITSU_CODE_ARCH_CDNA5).split(counter);
}

RaceTestBuilder makeGfx1250Test(int numWaves = 1) {
  return RaceTestBuilder(numWaves, /*waveSize=*/32, Dim3d(0),
                         counterCapacitiesForArch(ROCJITSU_CODE_ARCH_CDNA5));
}

void issueCounterOps(RaceTestBuilder &builder, int count, amdgpu::WaitCounterType counter,
                     MemoryOrderClass order) {
  for (int index = 0; index < count; ++index)
    builder.counterIssue(kWave, counter, order);
}

EventStatus statusOf(const RaceTestBuilder &builder, int event) {
  return builder.events().status(EventId{event});
}

struct SplitCounterCase {
  const char *name;
  amdgpu::WaitCounterType counter;
  MemoryOrderClass order;
};

void PrintTo(const SplitCounterCase &testCase, std::ostream *os) { *os << testCase.name; }

} // namespace

/// Every gfx1250 split counter with each ordered completion class it carries.
/// ASYNCcnt carries async loads and async stores as separate classes (CDNA5
/// ISA 10.8).
class Gfx1250SplitCounterCapacityTest : public ::testing::TestWithParam<SplitCounterCase> {
protected:
  int capacity() const { return gfx1250Capacity(GetParam().counter); }
  void issue(RaceTestBuilder &builder, int count) const {
    issueCounterOps(builder, count, GetParam().counter, GetParam().order);
  }
};

TEST_P(Gfx1250SplitCounterCapacityTest, CounterBelowCapacityKeepsOldestPending) {
  auto builder = makeGfx1250Test();
  issue(builder, capacity());

  EXPECT_EQ(statusOf(builder, 0), EventStatus::ACTIVE);
}

TEST_P(Gfx1250SplitCounterCapacityTest, OverflowRetiresOldestPrefix) {
  for (const int overflow : {1, 5}) {
    SCOPED_TRACE(overflow);
    auto builder = makeGfx1250Test();
    issue(builder, capacity() + overflow);

    for (int event = 0; event < overflow; ++event)
      EXPECT_EQ(statusOf(builder, event), EventStatus::RETIRED) << "event " << event;
    EXPECT_EQ(statusOf(builder, overflow), EventStatus::ACTIVE);
  }
}

INSTANTIATE_TEST_SUITE_P(
    RaceDetector, Gfx1250SplitCounterCapacityTest,
    ::testing::Values(
        SplitCounterCase{"Loadcnt", amdgpu::WaitCounterType::LOADCNT, MemoryOrderClass::VMEM},
        SplitCounterCase{"Storecnt", amdgpu::WaitCounterType::STORECNT, MemoryOrderClass::VMEM},
        SplitCounterCase{"Dscnt", amdgpu::WaitCounterType::DSCNT, MemoryOrderClass::LDS},
        SplitCounterCase{"AsynccntLoads", amdgpu::WaitCounterType::ASYNCCNT,
                         MemoryOrderClass::ASYNC_LOAD},
        SplitCounterCase{"AsynccntStores", amdgpu::WaitCounterType::ASYNCCNT,
                         MemoryOrderClass::ASYNC_STORE},
        SplitCounterCase{"Tensorcnt", amdgpu::WaitCounterType::TENSORCNT, MemoryOrderClass::TENSOR},
        // KMcnt's capacity is observable only through an ordered
        // class, and no gfx1250 instruction registers one, so
        // this row uses a synthetic one.
        SplitCounterCase{"KmcntSyntheticOrdered", amdgpu::WaitCounterType::KMCNT,
                         MemoryOrderClass::VMEM}),
    [](const ::testing::TestParamInfo<SplitCounterCase> &info) { return info.param.name; });

// gfx1250 has no combined counters, so metadata naming one does not match the
// target and fails loudly, as on a target whose capacity is not configured.
TEST(RaceDetector, Gfx1250CounterCapacity_CombinedCounterFailsLoudly) {
  for (const auto counter : {amdgpu::WaitCounterType::VMCNT, amdgpu::WaitCounterType::LGKMCNT}) {
    SCOPED_TRACE(static_cast<int>(counter));
    RaceDetector detector(/*nWaves=*/
                          1, Dim3d(0), [](RaceViolation) {},
                          counterCapacitiesForArch(ROCJITSU_CODE_ARCH_CDNA5),
                          /*ldsSizeBytes=*/64 * 1024);
    EXPECT_TRUE(detector.getWaveRaceState(0).prepareForCounterIncrement(counter).failed());
    EXPECT_TRUE(detector.getWaveRaceState(0).getWaveMemoryEvents().empty());
  }
}

TEST(RaceDetector, Gfx1250CounterCapacity_ArchCapacitiesConfigureSplitCountersOnly) {
  // 6-bit LOADcnt, STOREcnt, DScnt, ASYNCcnt and TENSORcnt, and 5-bit KMcnt
  // (CDNA5 ISA 5.7.1), each holding 2^n - 1 tokens.
  constexpr auto capacities = counterCapacitiesForArch(ROCJITSU_CODE_ARCH_CDNA5);
  EXPECT_EQ(capacities.loadcnt, 63);
  EXPECT_EQ(capacities.storecnt, 63);
  EXPECT_EQ(capacities.dscnt, 63);
  EXPECT_EQ(capacities.kmcnt, 31);
  EXPECT_EQ(capacities.asynccnt, 63);
  EXPECT_EQ(capacities.tensorcnt, 63);
  // gfx1250 has no combined VMCNT or LGKMCNT.
  EXPECT_EQ(capacities.vmcnt, 0);
  EXPECT_EQ(capacities.lgkmcnt, 0);
}

// split() returns each split counter's own field, and zero for the rest.
TEST(RaceDetector, CounterCapacity_SplitReturnsEachCountersOwnCapacity) {
  constexpr CounterCapacities capacities{.vmcnt = 1,
                                         .lgkmcnt = 2,
                                         .loadcnt = 3,
                                         .storecnt = 4,
                                         .dscnt = 5,
                                         .kmcnt = 6,
                                         .asynccnt = 7,
                                         .tensorcnt = 8};
  EXPECT_EQ(capacities.split(amdgpu::WaitCounterType::LOADCNT), 3);
  EXPECT_EQ(capacities.split(amdgpu::WaitCounterType::STORECNT), 4);
  EXPECT_EQ(capacities.split(amdgpu::WaitCounterType::DSCNT), 5);
  EXPECT_EQ(capacities.split(amdgpu::WaitCounterType::KMCNT), 6);
  EXPECT_EQ(capacities.split(amdgpu::WaitCounterType::ASYNCCNT), 7);
  EXPECT_EQ(capacities.split(amdgpu::WaitCounterType::TENSORCNT), 8);
  for (const auto counter : {amdgpu::WaitCounterType::VMCNT, amdgpu::WaitCounterType::LGKMCNT,
                             amdgpu::WaitCounterType::VSCNT, amdgpu::WaitCounterType::EXPCNT,
                             amdgpu::WaitCounterType::SAMPLECNT})
    EXPECT_EQ(capacities.split(counter), 0) << static_cast<int>(counter);
}

// Pre-GFX12 targets keep their combined capacities; their split counter names
// alias the combined counters, so they have no split capacities.
TEST(RaceDetector, CounterCapacity_LegacyArchCapacitiesHaveNoSplitCounters) {
  for (const auto arch :
       {ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2, ROCJITSU_CODE_ARCH_CDNA3,
        ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5}) {
    SCOPED_TRACE(static_cast<int>(arch));
    const auto capacities = counterCapacitiesForArch(arch);
    const auto properties = rocjitsu::isa_properties(arch);
    EXPECT_EQ(capacities.vmcnt, properties.vmcnt_capacity);
    EXPECT_EQ(capacities.lgkmcnt, properties.lgkmcnt_capacity);
    for (const auto counter :
         {amdgpu::WaitCounterType::LOADCNT, amdgpu::WaitCounterType::STORECNT,
          amdgpu::WaitCounterType::DSCNT, amdgpu::WaitCounterType::KMCNT,
          amdgpu::WaitCounterType::ASYNCCNT, amdgpu::WaitCounterType::TENSORCNT})
      EXPECT_EQ(capacities.split(counter), 0);
  }
}

// RDNA4 has split counters, but its capacity is not modeled: it has no combined
// counters and no split capacities.
TEST(RaceDetector, CounterCapacity_Rdna4ModelsNoCapacity) {
  constexpr auto capacities = counterCapacitiesForArch(ROCJITSU_CODE_ARCH_RDNA4);
  EXPECT_EQ(capacities.vmcnt, 0);
  EXPECT_EQ(capacities.lgkmcnt, 0);
  for (const auto counter : {amdgpu::WaitCounterType::LOADCNT, amdgpu::WaitCounterType::STORECNT,
                             amdgpu::WaitCounterType::DSCNT, amdgpu::WaitCounterType::KMCNT,
                             amdgpu::WaitCounterType::ASYNCCNT, amdgpu::WaitCounterType::TENSORCNT})
    EXPECT_EQ(capacities.split(counter), 0) << static_cast<int>(counter);
}

// A full counter of two-token events is detected even though the wave holds
// fewer events than the capacity: 15 two-token events plus a two-token issue
// overflow the 31-token KMcnt. Synthetic ordered events make it observable.
TEST(RaceDetector, Gfx1250CounterCapacity_TwoTokenEventsFillKmcnt) {
  constexpr uint8_t kTokens = amdgpu::MemoryCounterObligation::MAX_COUNTER_INCREMENT;
  const int capacity = gfx1250Capacity(amdgpu::WaitCounterType::KMCNT);
  const int events = capacity / kTokens;
  auto builder = makeGfx1250Test();
  for (int index = 0; index <= events; ++index)
    builder.counterIssue(kWave, amdgpu::WaitCounterType::KMCNT, MemoryOrderClass::VMEM, kTokens);

  EXPECT_EQ(statusOf(builder, 0), EventStatus::RETIRED);
  EXPECT_EQ(statusOf(builder, 1), EventStatus::ACTIVE);
}

// Scalar memory loads complete out of order (CDNA5 ISA 5.7.1.1), so a full
// KMcnt does not prove which one completed.
TEST(RaceDetector, Gfx1250CounterCapacity_FullKmcntDoesNotCompleteScalarLoads) {
  const int capacity = gfx1250Capacity(amdgpu::WaitCounterType::KMCNT);
  auto builder = makeGfx1250Test();
  for (int sgpr = 0; sgpr <= capacity; ++sgpr)
    builder.counterIssue(kWave, amdgpu::WaitCounterType::KMCNT, MemoryOrderClass::UNORDERED);

  EXPECT_EQ(statusOf(builder, 0), EventStatus::ACTIVE);
}

// Larger scalar loads add two KMcnt tokens each.
TEST(RaceDetector, Gfx1250CounterCapacity_FullKmcntDoesNotCompleteWideScalarLoads) {
  const int capacity = gfx1250Capacity(amdgpu::WaitCounterType::KMCNT);
  auto builder = makeGfx1250Test();
  const int wideLoads = capacity / 2 + 1;
  for (int load = 0; load < wideLoads; ++load)
    builder.counterIssue(kWave, amdgpu::WaitCounterType::KMCNT, MemoryOrderClass::UNORDERED, 2);

  EXPECT_EQ(statusOf(builder, 0), EventStatus::ACTIVE);
}

TEST(RaceDetector, Gfx1250CounterCapacity_SplitCountersDoNotShareCapacity) {
  struct Case {
    const char *name;
    amdgpu::WaitCounterType fullCounter;
    MemoryOrderClass fullOrder;
    amdgpu::WaitCounterType otherCounter;
    MemoryOrderClass otherOrder;
  };
  const Case cases[] = {
      {"StorecntAfterFullLoadcnt", amdgpu::WaitCounterType::LOADCNT, MemoryOrderClass::VMEM,
       amdgpu::WaitCounterType::STORECNT, MemoryOrderClass::VMEM},
      {"LoadcntAfterFullStorecnt", amdgpu::WaitCounterType::STORECNT, MemoryOrderClass::VMEM,
       amdgpu::WaitCounterType::LOADCNT, MemoryOrderClass::VMEM},
      {"KmcntAfterFullDscnt", amdgpu::WaitCounterType::DSCNT, MemoryOrderClass::LDS,
       amdgpu::WaitCounterType::KMCNT, MemoryOrderClass::UNORDERED},
      {"TensorcntAfterFullAsynccnt", amdgpu::WaitCounterType::ASYNCCNT,
       MemoryOrderClass::ASYNC_LOAD, amdgpu::WaitCounterType::TENSORCNT, MemoryOrderClass::TENSOR},
      {"AsynccntAfterFullTensorcnt", amdgpu::WaitCounterType::TENSORCNT, MemoryOrderClass::TENSOR,
       amdgpu::WaitCounterType::ASYNCCNT, MemoryOrderClass::ASYNC_LOAD},
  };
  for (const auto &testCase : cases) {
    SCOPED_TRACE(testCase.name);
    auto builder = makeGfx1250Test();
    issueCounterOps(builder, gfx1250Capacity(testCase.fullCounter), testCase.fullCounter,
                    testCase.fullOrder);
    builder.counterIssue(kWave, testCase.otherCounter, testCase.otherOrder);

    EXPECT_EQ(statusOf(builder, 0), EventStatus::ACTIVE);
  }
}

// Async loads and stores complete in order within their own class but out of
// order with each other (CDNA5 ISA 10.8). A full ASYNCcnt with both classes
// pending does not prove which class drained.
TEST(RaceDetector, Gfx1250CounterCapacity_MixedAsyncClassesRetireNothing) {
  const int capacity = gfx1250Capacity(amdgpu::WaitCounterType::ASYNCCNT);
  const int loads = capacity / 2 + 1;
  const int stores = capacity - loads;
  auto builder = makeGfx1250Test();
  issueCounterOps(builder, loads, amdgpu::WaitCounterType::ASYNCCNT, MemoryOrderClass::ASYNC_LOAD);
  issueCounterOps(builder, stores, amdgpu::WaitCounterType::ASYNCCNT,
                  MemoryOrderClass::ASYNC_STORE);
  builder.counterIssue(kWave, amdgpu::WaitCounterType::ASYNCCNT, MemoryOrderClass::ASYNC_LOAD);

  EXPECT_EQ(statusOf(builder, 0), EventStatus::ACTIVE);
  EXPECT_EQ(statusOf(builder, loads), EventStatus::ACTIVE);
}

TEST(RaceDetector, Gfx1250CounterCapacity_AsyncStoreIssueRetiresOldestAsyncLoad) {
  const int capacity = gfx1250Capacity(amdgpu::WaitCounterType::ASYNCCNT);
  auto builder = makeGfx1250Test();
  issueCounterOps(builder, capacity, amdgpu::WaitCounterType::ASYNCCNT,
                  MemoryOrderClass::ASYNC_LOAD);
  builder.counterIssue(kWave, amdgpu::WaitCounterType::ASYNCCNT, MemoryOrderClass::ASYNC_STORE);

  EXPECT_EQ(statusOf(builder, 0), EventStatus::RETIRED);
  EXPECT_EQ(statusOf(builder, 1), EventStatus::ACTIVE);
}

TEST(RaceDetector, Gfx1250CounterCapacity_AsyncLoadIssueRetiresOldestAsyncStore) {
  const int capacity = gfx1250Capacity(amdgpu::WaitCounterType::ASYNCCNT);
  auto builder = makeGfx1250Test();
  issueCounterOps(builder, capacity, amdgpu::WaitCounterType::ASYNCCNT,
                  MemoryOrderClass::ASYNC_STORE);
  builder.counterIssue(kWave, amdgpu::WaitCounterType::ASYNCCNT, MemoryOrderClass::ASYNC_LOAD);

  EXPECT_EQ(statusOf(builder, 0), EventStatus::RETIRED);
  EXPECT_EQ(statusOf(builder, 1), EventStatus::ACTIVE);
}

// Tensor loads and stores complete in one issue-ordered class (CDNA5 ISA
// 10.11.1). A tensor transfer is an LDS event, so in this single-wave
// workgroup the transfer that capacity completes is retired at once.
TEST(RaceDetector, Gfx1250CounterCapacity_TensorLoadsAndStoresShareOneClass) {
  const int capacity = gfx1250Capacity(amdgpu::WaitCounterType::TENSORCNT);
  auto builder = makeGfx1250Test();
  for (int index = 0; index <= capacity; ++index) {
    const auto type = index % 2 == 0 ? LdsAccess::Write : LdsAccess::Read;
    builder.counterLdsTransfer(
        kWave, amdgpu::WaitCounterType::TENSORCNT, MemoryOrderClass::TENSOR, type,
        {{.address = static_cast<uint32_t>(index * kDwordBytes), .bytes = kDwordBytes}});
  }

  EXPECT_FALSE(builder.hasRace());
  EXPECT_EQ(statusOf(builder, 0), EventStatus::RETIRED);
  EXPECT_EQ(statusOf(builder, 1), EventStatus::ACTIVE);
}

TEST(RaceDetector, Gfx1250CounterCapacity_PartialWaitAfterCapacityRetiresNextEvents) {
  const int capacity = gfx1250Capacity(amdgpu::WaitCounterType::LOADCNT);
  auto builder = makeGfx1250Test();
  issueCounterOps(builder, capacity + 1, amdgpu::WaitCounterType::LOADCNT, MemoryOrderClass::VMEM);
  // Capacity retired event 0, leaving `capacity` loads pending. Allowing two
  // fewer retires events 1 and 2.
  builder.waitCounter(kWave, amdgpu::WaitCounterType::LOADCNT, capacity - 2);

  EXPECT_EQ(statusOf(builder, 0), EventStatus::RETIRED);
  EXPECT_EQ(statusOf(builder, 1), EventStatus::RETIRED);
  EXPECT_EQ(statusOf(builder, 2), EventStatus::RETIRED);
  EXPECT_EQ(statusOf(builder, 3), EventStatus::ACTIVE);
}

// A DS write that capacity proves complete is released to other waves by the
// next barrier, as one completed by s_wait_dscnt would be.
TEST(RaceDetector, Gfx1250CounterCapacity_CapacityCompletedDsWriteIsReleasedAtBarrier) {
  const int capacity = gfx1250Capacity(amdgpu::WaitCounterType::DSCNT);
  auto builder = makeGfx1250Test(/*numWaves=*/2);
  builder.ldsWrite(/*wave=*/0, kLane, /*addr=*/0, kDwordBytes, MemoryOrderClass::LDS,
                   amdgpu::WaitCounterType::DSCNT);
  for (int index = 0; index < capacity; ++index)
    builder.ldsWrite(/*wave=*/0, kLane, /*addr=*/64, kDwordBytes, MemoryOrderClass::LDS,
                     amdgpu::WaitCounterType::DSCNT);
  builder.barrier();

  builder.checkLdsRead(/*wave=*/1, kLane, /*addr=*/0, kDwordBytes);
  EXPECT_FALSE(builder.hasLdsRace(0));
}

TEST(RaceDetector, Gfx1250CounterCapacity_DsWriteBelowCapacityIsNotReleasedAtBarrier) {
  const int capacity = gfx1250Capacity(amdgpu::WaitCounterType::DSCNT);
  auto builder = makeGfx1250Test(/*numWaves=*/2);
  builder.ldsWrite(/*wave=*/0, kLane, /*addr=*/0, kDwordBytes, MemoryOrderClass::LDS,
                   amdgpu::WaitCounterType::DSCNT);
  for (int index = 0; index < capacity - 1; ++index)
    builder.ldsWrite(/*wave=*/0, kLane, /*addr=*/64, kDwordBytes, MemoryOrderClass::LDS,
                     amdgpu::WaitCounterType::DSCNT);
  builder.barrier();

  builder.checkLdsRead(/*wave=*/1, kLane, /*addr=*/0, kDwordBytes);
  EXPECT_TRUE(builder.hasLdsRace(0));
}

// Capacity is modeled for split counters only on gfx1250. Pre-GFX12 targets
// have no ASYNCcnt or TENSORcnt capacity, so those events stay pending there.
TEST(RaceDetector, CounterCapacity_LegacyTargetsDoNotModelGfx1250Counters) {
  for (const auto counter :
       {amdgpu::WaitCounterType::ASYNCCNT, amdgpu::WaitCounterType::TENSORCNT}) {
    SCOPED_TRACE(static_cast<int>(counter));
    RaceTestBuilder builder(/*numWaves=*/1);
    issueCounterOps(builder, kVmcntCapacity + 1, counter,
                    counter == amdgpu::WaitCounterType::ASYNCCNT ? MemoryOrderClass::ASYNC_LOAD
                                                                 : MemoryOrderClass::TENSOR);

    EXPECT_EQ(statusOf(builder, 0), EventStatus::ACTIVE);
  }
}
