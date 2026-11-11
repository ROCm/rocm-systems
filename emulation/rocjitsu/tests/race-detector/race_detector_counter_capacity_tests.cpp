// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "race_test_builder.h"

#include <gtest/gtest.h>

#include <array>
#include <stdexcept>

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

// ---- All-ones no-wait operands ----

TEST(RaceDetector, CounterCapacity_LgkmcntAllOnesWaitIsNoOpBelowCapacity) {
  auto builder = makeCounterTest();
  issueOrderedLdsReads(builder, kClearlyBelowCapacity);

  builder.waitcnt(kWave, /*vmcnt=*/-1, /*lgkmcnt=*/kLgkmcntCapacity);
  expectPending(builder, /*event=*/0);
}

TEST(RaceDetector, CounterCapacity_VmcntAllOnesWaitIsNoOpBelowCapacity) {
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
