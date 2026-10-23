// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "race_test_builder.h"

#include <gtest/gtest.h>

#include <array>
#include <stdexcept>
#include <vector>

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

std::vector<EventId> issueOrderedLdsReads(RaceTestBuilder &builder, int count) {
  std::vector<EventId> events;
  for (int index = 0; index < count; ++index)
    events.push_back(builder.ldsRead(kWave, kLane, /*addr=*/index * kDwordBytes,
                                     /*bytes=*/kDwordBytes));
  return events;
}

std::vector<EventId> issueOrderedVmemLoads(RaceTestBuilder &builder, int count) {
  std::vector<EventId> events;
  for (int index = 0; index < count; ++index)
    events.push_back(builder.globalLoad(kWave));
  return events;
}

std::vector<EventId> issueUnorderedVmcntLoads(RaceTestBuilder &builder, int count) {
  std::vector<EventId> events;
  for (int index = 0; index < count; ++index)
    events.push_back(builder.globalLoad(kWave, MemoryOrderClass::UNORDERED));
  return events;
}

void issueOrderedVmemStores(RaceTestBuilder &builder, int count) {
  for (int index = 0; index < count; ++index)
    builder.globalStore(kWave);
}

void expectReady(const RaceTestBuilder &builder, EventId event) {
  EXPECT_FALSE(builder.isPending(event));
}

void expectPending(const RaceTestBuilder &builder, EventId event) {
  EXPECT_TRUE(builder.isPending(event));
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

// ---- All-ones no-wait operands ----

TEST(RaceDetector, CounterCapacity_LgkmcntAllOnesWaitIsNoOpBelowCapacity) {
  RaceTestBuilder builder(/*numWaves=*/1);
  const auto reads = issueOrderedLdsReads(builder, kClearlyBelowCapacity);

  builder.waitcnt(kWave, /*vmcnt=*/-1, /*lgkmcnt=*/kLgkmcntCapacity);
  expectPending(builder, reads[0]);
}

TEST(RaceDetector, CounterCapacity_VmcntAllOnesWaitIsNoOpBelowCapacity) {
  RaceTestBuilder builder(/*numWaves=*/1);
  const auto loads = issueOrderedVmemLoads(builder, kClearlyBelowCapacity);

  builder.waitcnt(kWave, /*vmcnt=*/kVmcntCapacity, /*lgkmcnt=*/-1);
  expectPending(builder, loads[0]);
}

// ---- Issue-time capacity backpressure ----

TEST(RaceDetector, CounterCapacity_MissingConfiguredCapacityFailsLoudly) {
  RaceDetector detector(
      /*nWaves=*/
      1, Dim3d(0), [](RaceViolation) {}, CounterCapacities{.vmcnt = 0, .lgkmcnt = 0},
      /*ldsSizeBytes=*/64 * 1024);

  EXPECT_THROW(
      detector.getWaveRaceState(0).prepareForCounterIncrement(amdgpu::WaitCounterType::VMCNT),
      std::logic_error);
}

TEST(RaceDetector, CounterCapacity_LgkmcntOverflowRetiresOldestOrderedReads) {
  RaceTestBuilder builder(/*numWaves=*/1);
  const auto reads = issueOrderedLdsReads(builder, kLgkmcntCapacity + kLgkmcntOverflow);

  expectReady(builder, reads[kLgkmcntOverflow - 1]);
  expectPending(builder, reads[kLgkmcntOverflow]);
}

TEST(RaceDetector, CounterCapacity_VmcntOverflowRetiresOldestOrderedLoads) {
  RaceTestBuilder builder(/*numWaves=*/1);
  const auto loads = issueOrderedVmemLoads(builder, kVmcntCapacity + kVmcntOverflow);

  expectReady(builder, loads[kVmcntOverflow - 1]);
  expectPending(builder, loads[kVmcntOverflow]);
}

TEST(RaceDetector, CounterCapacity_FullLgkmcntRetiresBeforeCurrentOperandRead) {
  RaceTestBuilder builder(/*numWaves=*/1);
  const auto reads = issueOrderedLdsReads(builder, kLgkmcntCapacity);

  builder.prepareCounterIncrement(kWave, amdgpu::WaitCounterType::LGKMCNT);
  expectReady(builder, reads[0]);
}

TEST(RaceDetector, CounterCapacity_LgkmcntBelowCapacityKeepsCurrentOperandPending) {
  RaceTestBuilder builder(/*numWaves=*/1);
  const auto reads = issueOrderedLdsReads(builder, kLgkmcntCapacity - 1);

  builder.prepareCounterIncrement(kWave, amdgpu::WaitCounterType::LGKMCNT);
  expectPending(builder, reads[0]);
}

TEST(RaceDetector, CounterCapacity_FullVmcntRetiresBeforeCurrentOperandRead) {
  RaceTestBuilder builder(/*numWaves=*/1);
  const auto loads = issueOrderedVmemLoads(builder, kVmcntCapacity);

  builder.prepareCounterIncrement(kWave, amdgpu::WaitCounterType::VMCNT);
  expectReady(builder, loads[0]);
}

TEST(RaceDetector, CounterCapacity_VmcntBelowCapacityKeepsCurrentOperandPending) {
  RaceTestBuilder builder(/*numWaves=*/1);
  const auto loads = issueOrderedVmemLoads(builder, kVmcntCapacity - 1);

  builder.prepareCounterIncrement(kWave, amdgpu::WaitCounterType::VMCNT);
  expectPending(builder, loads[0]);
}

TEST(RaceDetector, CounterCapacity_LoadcntAliasUsesVmcntCapacity) {
  RaceTestBuilder builder(/*numWaves=*/1);
  const auto loads = issueOrderedVmemLoads(builder, kVmcntCapacity);

  builder.prepareCounterIncrement(kWave, amdgpu::WaitCounterType::LOADCNT);
  expectReady(builder, loads[0]);
}

TEST(RaceDetector, CounterCapacity_SplitLgkmAliasesUseCombinedCapacity) {
  for (const auto counter : {amdgpu::WaitCounterType::DSCNT, amdgpu::WaitCounterType::KMCNT}) {
    SCOPED_TRACE(static_cast<int>(counter));
    RaceTestBuilder builder(/*numWaves=*/1);
    const auto reads = issueOrderedLdsReads(builder, kLgkmcntCapacity);

    builder.prepareCounterIncrement(kWave, counter);
    expectReady(builder, reads[0]);
  }
}

TEST(RaceDetector, CounterCapacity_StorecntDoesNotAdvanceVmcnt) {
  RaceTestBuilder builder(/*numWaves=*/1);
  const auto loads = issueOrderedVmemLoads(builder, kVmcntCapacity);

  builder.prepareCounterIncrement(kWave, amdgpu::WaitCounterType::STORECNT);
  expectPending(builder, loads[0]);
}

TEST(RaceDetector, CounterCapacity_GenericFlatAdvancesBothCounterDomains) {
  amdgpu::MemoryIssueInfo flatIssue;
  flatIssue.counter_obligations_[0] = {amdgpu::WaitCounterType::LOADCNT,
                                       amdgpu::MemoryCompletionClass::VMEM};
  flatIssue.counter_obligations_[1] = {amdgpu::WaitCounterType::DSCNT,
                                       amdgpu::MemoryCompletionClass::LDS};
  flatIssue.num_counter_obligations_ = 2;

  RaceTestBuilder vmemBuilder(/*numWaves=*/1);
  const auto loads = issueOrderedVmemLoads(vmemBuilder, kVmcntCapacity);
  vmemBuilder.prepareMemoryIssue(kWave, flatIssue);
  expectReady(vmemBuilder, loads[0]);

  RaceTestBuilder ldsBuilder(/*numWaves=*/1);
  const auto reads = issueOrderedLdsReads(ldsBuilder, kLgkmcntCapacity);
  ldsBuilder.prepareMemoryIssue(kWave, flatIssue);
  expectReady(ldsBuilder, reads[0]);
}

// ---- Mixed completion-order classes ----

TEST(RaceDetector, CounterCapacity_FullLgkmcntDoesNotProveScalarLoadComplete) {
  RaceTestBuilder builder(/*numWaves=*/1);
  const EventId scalar = builder.scalarLoad(kWave);
  issueOrderedLdsReads(builder, kLgkmcntCapacity);

  expectPending(builder, scalar);
}

TEST(RaceDetector, CounterCapacity_ScalarIssueAdvancesFullOrderedLgkmcntClass) {
  RaceTestBuilder builder(/*numWaves=*/1);
  const auto reads = issueOrderedLdsReads(builder, kLgkmcntCapacity);

  builder.scalarLoad(kWave);
  expectReady(builder, reads[0]);
}

TEST(RaceDetector, CounterCapacity_WideScalarIssueContributesTwoLgkmcntTokens) {
  RaceTestBuilder builder(/*numWaves=*/1);
  const auto reads = issueOrderedLdsReads(builder, kLgkmcntCapacity - 1);

  builder.scalarLoad(kWave, /*dwords=*/2);
  expectReady(builder, reads[0]);
}

TEST(RaceDetector, CounterCapacity_MixedLgkmcntDoesNotGuessWhichEventCompleted) {
  RaceTestBuilder builder(/*numWaves=*/1);
  const auto reads = issueOrderedLdsReads(builder, kLgkmcntCapacity - 1);
  builder.scalarLoad(kWave);

  // The next scalar load needs one slot, but the completed event could be the
  // first scalar load. It does not prove that the oldest LDS read completed.
  builder.scalarLoad(kWave);
  expectPending(builder, reads[0]);
}

TEST(RaceDetector, CounterCapacity_InterleavedLgkmcntSoakRetiresOnlyProvableLdsPrefix) {
  RaceTestBuilder builder(/*numWaves=*/1);

  // Build a deterministic, irregular stream of 49 LGKMCNT events: 15 ordered
  // LDS reads interleaved with 34 unordered scalar stores. The detector's
  // candidate set can exceed the physical capacity because it represents all
  // events that might remain pending, not one concrete hardware state. Since
  // it cannot identify which unordered events completed, all 15 LDS candidates
  // remain until one more LGKMCNT operation issues.
  constexpr std::array<int, kLgkmcntCapacity> unorderedBursts = {
      3, 0, 4, 1, 2, 5, 0, 3, 1, 4, 2, 0, 5, 1, 3,
  };
  std::vector<EventId> reads;
  for (int ordered = 0; ordered < kLgkmcntCapacity; ++ordered) {
    for (int unordered = 0; unordered < unorderedBursts[ordered]; ++unordered)
      builder.scalarStore(kWave);
    reads.push_back(builder.ldsRead(kWave, kLane, /*addr=*/ordered * kDwordBytes,
                                    /*bytes=*/kDwordBytes));
  }

  // Before the decisive issue, even the oldest LDS result may still be
  // pending. The next scalar store requires a free LGKMCNT slot. Since 15 LDS
  // events may remain, LDS FIFO ordering proves exactly the oldest one
  // complete, while the next one remains pending.
  expectPending(builder, reads[0]);

  builder.scalarStore(kWave);
  expectReady(builder, reads[0]);
  expectPending(builder, reads[1]);
}

TEST(RaceDetector, CounterCapacity_FullVmcntDoesNotProveUnorderedLoadComplete) {
  RaceTestBuilder builder(/*numWaves=*/1);
  const auto unordered = issueUnorderedVmcntLoads(builder, /*count=*/1);
  issueOrderedVmemLoads(builder, kVmcntCapacity);

  expectPending(builder, unordered[0]);
}

TEST(RaceDetector, CounterCapacity_UnorderedLoadsDoNotAdvanceOrderedVmcntClass) {
  RaceTestBuilder builder(/*numWaves=*/1);
  const auto ordered = issueOrderedVmemLoads(builder, /*count=*/1);
  issueUnorderedVmcntLoads(builder, kVmcntCapacity);

  expectPending(builder, ordered[0]);
}

TEST(RaceDetector, CounterCapacity_UnorderedIssueAdvancesFullOrderedVmcntClass) {
  RaceTestBuilder builder(/*numWaves=*/1);
  const auto ordered = issueOrderedVmemLoads(builder, kVmcntCapacity);

  issueUnorderedVmcntLoads(builder, /*count=*/1);
  expectReady(builder, ordered[0]);
}

// ---- Explicit partial waits after capacity backpressure ----

TEST(RaceDetector, CounterCapacity_PartialLgkmcntKeepsUnorderedScalarLoadPending) {
  RaceTestBuilder builder(/*numWaves=*/1);
  const auto reads = issueOrderedLdsReads(builder, kLgkmcntCapacity);
  const EventId scalar = builder.scalarLoad(kWave);

  builder.waitcnt(kWave, /*vmcnt=*/-1, /*lgkmcnt=*/kLgkmcntCapacity - 1);
  expectReady(builder, reads[0]);
  expectPending(builder, scalar);
}

TEST(RaceDetector, CounterCapacity_PartialLgkmcntPreservesAmbiguousEventsWithinLimit) {
  RaceTestBuilder builder(/*numWaves=*/1);
  const auto reads = issueOrderedLdsReads(builder, kLgkmcntCapacity - 1);
  const EventId scalar = builder.scalarLoad(kWave);

  builder.waitcnt(kWave, /*vmcnt=*/-1, /*lgkmcnt=*/kLgkmcntCapacity - 1);
  expectPending(builder, reads[0]);
  expectPending(builder, scalar);
}

TEST(RaceDetector, CounterCapacity_PartialLgkmcntRetiresNextOrderedRead) {
  RaceTestBuilder builder(/*numWaves=*/1);
  const auto reads = issueOrderedLdsReads(builder, kLgkmcntCapacity + kLgkmcntOverflow);

  builder.waitcnt(kWave, /*vmcnt=*/-1, /*lgkmcnt=*/kLgkmcntCapacity - 1);
  expectReady(builder, reads[kLgkmcntOverflow]);
  expectPending(builder, reads[kLgkmcntOverflow + 1]);
}

TEST(RaceDetector, CounterCapacity_PartialVmcntRetiresNextOrderedLoad) {
  RaceTestBuilder builder(/*numWaves=*/1);
  const auto loads = issueOrderedVmemLoads(builder, kVmcntCapacity + kVmcntOverflow);

  builder.waitcnt(kWave, /*vmcnt=*/kVmcntCapacity - 1, /*lgkmcnt=*/-1);
  expectReady(builder, loads[kVmcntOverflow]);
  expectPending(builder, loads[kVmcntOverflow + 1]);
}

// ---- Other VMCNT producers ----

TEST(RaceDetector, CounterCapacity_CombinedVmemStoresAdvanceVmcntCapacity) {
  RaceTestBuilder builder(/*numWaves=*/1);
  const EventId load = builder.globalLoad(kWave);
  issueOrderedVmemStores(builder, kVmcntCapacity);

  expectReady(builder, load);
}

TEST(RaceDetector, CounterCapacity_CombinedVmemLoadRemainsPendingBeforeStoreCapacity) {
  RaceTestBuilder builder(/*numWaves=*/1);
  const EventId load = builder.globalLoad(kWave);
  issueOrderedVmemStores(builder, kVmcntCapacity - 1);

  expectPending(builder, load);
}

TEST(RaceDetector, CounterCapacity_VmcntCompletesDirectToLdsForOwningWave) {
  RaceTestBuilder builder(/*numWaves=*/1);
  builder.globalToLds(kWave, /*ldsAddrs=*/{0}, /*bytesPerLane=*/kDwordBytes, /*exec=*/1);
  issueOrderedVmemLoads(builder, kVmcntCapacity);

  expectLdsReady(builder);
}

TEST(RaceDetector, CounterCapacity_DirectToLdsRemainsPendingBelowVmcntCapacity) {
  RaceTestBuilder builder(/*numWaves=*/1);
  builder.globalToLds(kWave, /*ldsAddrs=*/{0}, /*bytesPerLane=*/kDwordBytes, /*exec=*/1);
  issueOrderedVmemLoads(builder, kVmcntCapacity - 1);

  expectLdsPending(builder);
}
