// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "race_test_builder.h"

#include <gtest/gtest.h>

using namespace rocjitsu::plugins::race_detector;
namespace amdgpu = rocjitsu::amdgpu;

namespace {

constexpr int kWave = 0;
constexpr int kLane = 0;
constexpr int kScalarRegisters = 8;
constexpr int kDwordBytes = 4;

// For legacy CDNA counters, the all-ones no-wait encoding is also the maximum
// number of outstanding tokens. Issuing another token forces older work to
// complete before the new operation can issue.
constexpr int kLgkmcntCapacity = 15;
constexpr int kVmcntCapacity = 63;
constexpr int kLgkmcntOverflow = 5;
constexpr int kVmcntOverflow = 7;
constexpr int kClearlyBelowCapacity = 10;

RaceTestBuilder makeLgkmcntTest() {
  return RaceTestBuilder(/*numWaves=*/1, /*vgprs=*/kLgkmcntCapacity + kLgkmcntOverflow,
                         /*sgprs=*/kScalarRegisters);
}

RaceTestBuilder makeVmcntTest() {
  return RaceTestBuilder(/*numWaves=*/1, /*vgprs=*/kVmcntCapacity + kVmcntOverflow,
                         /*sgprs=*/kScalarRegisters);
}

RaceTestBuilder makeSyntheticLgkmcntOverflowTest() {
  return RaceTestBuilder(/*numWaves=*/1, /*vgprs=*/kLgkmcntCapacity + kLgkmcntOverflow,
                         /*sgprs=*/kScalarRegisters, /*waveSize=*/64, /*wgId=*/Dim3d(0),
                         /*modelCounterBackpressure=*/false);
}

RaceTestBuilder makeSyntheticVmcntOverflowTest() {
  return RaceTestBuilder(/*numWaves=*/1, /*vgprs=*/kVmcntCapacity + kVmcntOverflow,
                         /*sgprs=*/kScalarRegisters, /*waveSize=*/64, /*wgId=*/Dim3d(0),
                         /*modelCounterBackpressure=*/false);
}

void issueOrderedLdsReads(RaceTestBuilder &builder, int count, int firstVgpr = 0) {
  for (int offset = 0; offset < count; ++offset) {
    const int vgpr = firstVgpr + offset;
    builder.ldsRead(kWave, kLane, /*addr=*/vgpr * kDwordBytes, /*bytes=*/kDwordBytes,
                    /*vgprDst=*/vgpr);
  }
}

void issueOrderedVmemLoads(RaceTestBuilder &builder, int count, int firstVgpr = 0) {
  for (int offset = 0; offset < count; ++offset) {
    builder.globalLoad(kWave, /*vgprBase=*/firstVgpr + offset, /*numRegs=*/1);
  }
}

void issueUnorderedVmcntLoads(RaceTestBuilder &builder, int count, int firstVgpr = 0) {
  for (int offset = 0; offset < count; ++offset) {
    builder.globalLoad(kWave, /*vgprBase=*/firstVgpr + offset, /*numRegs=*/1, /*exec=*/0,
                       /*byteMask=*/0xF, MemoryOrdering{});
  }
}

void issueOrderedVmemStores(RaceTestBuilder &builder, int count) {
  for (int index = 0; index < count; ++index)
    builder.globalStore(kWave);
}

void expectVgprReady(RaceTestBuilder &builder, int vgpr) {
  builder.checkVgprRead(kWave, vgpr, kLane);
  EXPECT_FALSE(builder.hasVgprRace(vgpr));
}

void expectVgprPending(RaceTestBuilder &builder, int vgpr) {
  builder.checkVgprRead(kWave, vgpr, kLane);
  EXPECT_TRUE(builder.hasVgprRace(vgpr));
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
  auto builder = makeLgkmcntTest();
  issueOrderedLdsReads(builder, kClearlyBelowCapacity);

  builder.waitcnt(kWave, /*vmcnt=*/-1, /*lgkmcnt=*/kLgkmcntCapacity);
  expectVgprPending(builder, /*vgpr=*/0);
}

TEST(RaceDetector, CounterCapacity_LgkmcntAllOnesWaitDoesNotTrimSyntheticOverflow) {
  auto builder = makeSyntheticLgkmcntOverflowTest();
  issueOrderedLdsReads(builder, kLgkmcntCapacity + kLgkmcntOverflow);

  builder.waitcnt(kWave, /*vmcnt=*/-1, /*lgkmcnt=*/kLgkmcntCapacity);
  expectVgprPending(builder, /*vgpr=*/0);
}

TEST(RaceDetector, CounterCapacity_VmcntAllOnesWaitIsNoOpBelowCapacity) {
  auto builder = makeVmcntTest();
  issueOrderedVmemLoads(builder, kClearlyBelowCapacity);

  builder.waitcnt(kWave, /*vmcnt=*/kVmcntCapacity, /*lgkmcnt=*/-1);
  expectVgprPending(builder, /*vgpr=*/0);
}

TEST(RaceDetector, CounterCapacity_VmcntAllOnesWaitDoesNotTrimSyntheticOverflow) {
  auto builder = makeSyntheticVmcntOverflowTest();
  issueOrderedVmemLoads(builder, kVmcntCapacity + kVmcntOverflow);

  builder.waitcnt(kWave, /*vmcnt=*/kVmcntCapacity, /*lgkmcnt=*/-1);
  expectVgprPending(builder, /*vgpr=*/0);
}

// ---- Issue-time capacity backpressure ----

TEST(RaceDetector, CounterCapacity_LgkmcntOverflowRetiresOldestOrderedReads) {
  auto builder = makeLgkmcntTest();
  issueOrderedLdsReads(builder, kLgkmcntCapacity + kLgkmcntOverflow);

  expectVgprReady(builder, /*vgpr=*/kLgkmcntOverflow - 1);
  expectVgprPending(builder, /*vgpr=*/kLgkmcntOverflow);
}

TEST(RaceDetector, CounterCapacity_VmcntOverflowRetiresOldestOrderedLoads) {
  auto builder = makeVmcntTest();
  issueOrderedVmemLoads(builder, kVmcntCapacity + kVmcntOverflow);

  expectVgprReady(builder, /*vgpr=*/kVmcntOverflow - 1);
  expectVgprPending(builder, /*vgpr=*/kVmcntOverflow);
}

TEST(RaceDetector, CounterCapacity_FullLgkmcntRetiresBeforeCurrentOperandRead) {
  auto builder = makeLgkmcntTest();
  issueOrderedLdsReads(builder, kLgkmcntCapacity);

  builder.prepareCounterIncrement(kWave, amdgpu::WaitCounterType::LGKMCNT);
  expectVgprReady(builder, /*vgpr=*/0);
}

TEST(RaceDetector, CounterCapacity_LgkmcntBelowCapacityKeepsCurrentOperandPending) {
  auto builder = makeLgkmcntTest();
  issueOrderedLdsReads(builder, kLgkmcntCapacity - 1);

  builder.prepareCounterIncrement(kWave, amdgpu::WaitCounterType::LGKMCNT);
  expectVgprPending(builder, /*vgpr=*/0);
}

TEST(RaceDetector, CounterCapacity_FullVmcntRetiresBeforeCurrentOperandRead) {
  auto builder = makeVmcntTest();
  issueOrderedVmemLoads(builder, kVmcntCapacity);

  builder.prepareCounterIncrement(kWave, amdgpu::WaitCounterType::VMCNT);
  expectVgprReady(builder, /*vgpr=*/0);
}

TEST(RaceDetector, CounterCapacity_VmcntBelowCapacityKeepsCurrentOperandPending) {
  auto builder = makeVmcntTest();
  issueOrderedVmemLoads(builder, kVmcntCapacity - 1);

  builder.prepareCounterIncrement(kWave, amdgpu::WaitCounterType::VMCNT);
  expectVgprPending(builder, /*vgpr=*/0);
}

// ---- Mixed completion-order classes ----

TEST(RaceDetector, CounterCapacity_FullLgkmcntDoesNotProveScalarLoadComplete) {
  auto builder = makeLgkmcntTest();
  builder.scalarLoad(kWave, /*sgprBase=*/0, /*numRegs=*/1);
  issueOrderedLdsReads(builder, kLgkmcntCapacity);

  builder.checkSgprRead(kWave, /*reg=*/0);
  EXPECT_TRUE(builder.hasSgprRace(0));
}

TEST(RaceDetector, CounterCapacity_ScalarIssueAdvancesFullOrderedLgkmcntClass) {
  auto builder = makeLgkmcntTest();
  issueOrderedLdsReads(builder, kLgkmcntCapacity);

  builder.scalarLoad(kWave, /*sgprBase=*/0, /*numRegs=*/1);
  expectVgprReady(builder, /*vgpr=*/0);
}

TEST(RaceDetector, CounterCapacity_TwoTokenScalarIssueAdvancesNearlyFullLgkmcntClass) {
  auto builder = makeLgkmcntTest();
  issueOrderedLdsReads(builder, kLgkmcntCapacity - 1);

  builder.scalarLoad(kWave, /*sgprBase=*/0, /*numRegs=*/2);
  expectVgprReady(builder, /*vgpr=*/0);
}

TEST(RaceDetector, CounterCapacity_MixedLgkmcntDoesNotGuessWhichEventCompleted) {
  auto builder = makeLgkmcntTest();
  issueOrderedLdsReads(builder, kLgkmcntCapacity - 1);
  builder.scalarLoad(kWave, /*sgprBase=*/0, /*numRegs=*/1);

  // The next scalar load needs one slot, but the completed event could be the
  // first scalar load. It does not prove that the oldest LDS read completed.
  builder.scalarLoad(kWave, /*sgprBase=*/1, /*numRegs=*/1);
  expectVgprPending(builder, /*vgpr=*/0);
}

TEST(RaceDetector, CounterCapacity_FullVmcntDoesNotProveUnorderedLoadComplete) {
  auto builder = makeVmcntTest();
  issueUnorderedVmcntLoads(builder, /*count=*/1);
  issueOrderedVmemLoads(builder, kVmcntCapacity, /*firstVgpr=*/1);

  expectVgprPending(builder, /*vgpr=*/0);
}

TEST(RaceDetector, CounterCapacity_UnorderedLoadsDoNotAdvanceOrderedVmcntClass) {
  auto builder = makeVmcntTest();
  issueOrderedVmemLoads(builder, /*count=*/1);
  issueUnorderedVmcntLoads(builder, kVmcntCapacity, /*firstVgpr=*/1);

  expectVgprPending(builder, /*vgpr=*/0);
}

TEST(RaceDetector, CounterCapacity_UnorderedIssueAdvancesFullOrderedVmcntClass) {
  auto builder = makeVmcntTest();
  issueOrderedVmemLoads(builder, kVmcntCapacity);

  issueUnorderedVmcntLoads(builder, /*count=*/1, /*firstVgpr=*/kVmcntCapacity);
  expectVgprReady(builder, /*vgpr=*/0);
}

// ---- Explicit partial waits after capacity backpressure ----

TEST(RaceDetector, CounterCapacity_PartialLgkmcntKeepsUnorderedScalarLoadPending) {
  auto builder = makeLgkmcntTest();
  issueOrderedLdsReads(builder, kLgkmcntCapacity);
  builder.scalarLoad(kWave, /*sgprBase=*/0, /*numRegs=*/1);

  builder.waitcnt(kWave, /*vmcnt=*/-1, /*lgkmcnt=*/kLgkmcntCapacity - 1);
  expectVgprReady(builder, /*vgpr=*/0);
  builder.checkSgprRead(kWave, /*reg=*/0);
  EXPECT_TRUE(builder.hasSgprRace(0));
}

TEST(RaceDetector, CounterCapacity_PartialLgkmcntPreservesAmbiguousEventsWithinLimit) {
  auto builder = makeLgkmcntTest();
  issueOrderedLdsReads(builder, kLgkmcntCapacity - 1);
  builder.scalarLoad(kWave, /*sgprBase=*/0, /*numRegs=*/1);

  builder.waitcnt(kWave, /*vmcnt=*/-1, /*lgkmcnt=*/kLgkmcntCapacity - 1);
  expectVgprPending(builder, /*vgpr=*/0);
  builder.checkSgprRead(kWave, /*reg=*/0);
  EXPECT_TRUE(builder.hasSgprRace(0));
}

TEST(RaceDetector, CounterCapacity_PartialLgkmcntRetiresNextOrderedRead) {
  auto builder = makeLgkmcntTest();
  issueOrderedLdsReads(builder, kLgkmcntCapacity + kLgkmcntOverflow);

  builder.waitcnt(kWave, /*vmcnt=*/-1, /*lgkmcnt=*/kLgkmcntCapacity - 1);
  expectVgprReady(builder, /*vgpr=*/kLgkmcntOverflow);
  expectVgprPending(builder, /*vgpr=*/kLgkmcntOverflow + 1);
}

TEST(RaceDetector, CounterCapacity_PartialVmcntRetiresNextOrderedLoad) {
  auto builder = makeVmcntTest();
  issueOrderedVmemLoads(builder, kVmcntCapacity + kVmcntOverflow);

  builder.waitcnt(kWave, /*vmcnt=*/kVmcntCapacity - 1, /*lgkmcnt=*/-1);
  expectVgprReady(builder, /*vgpr=*/kVmcntOverflow);
  expectVgprPending(builder, /*vgpr=*/kVmcntOverflow + 1);
}

// ---- Other VMCNT producers ----

TEST(RaceDetector, CounterCapacity_LegacyVmemStoresAdvanceVmcntCapacity) {
  RaceTestBuilder builder(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/kScalarRegisters);
  builder.globalLoad(kWave, /*vgprBase=*/0, /*numRegs=*/1);
  issueOrderedVmemStores(builder, kVmcntCapacity);

  expectVgprReady(builder, /*vgpr=*/0);
}

TEST(RaceDetector, CounterCapacity_LegacyVmemLoadRemainsPendingBeforeStoreCapacity) {
  RaceTestBuilder builder(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/kScalarRegisters);
  builder.globalLoad(kWave, /*vgprBase=*/0, /*numRegs=*/1);
  issueOrderedVmemStores(builder, kVmcntCapacity - 1);

  expectVgprPending(builder, /*vgpr=*/0);
}

TEST(RaceDetector, CounterCapacity_VmcntCompletesDirectToLdsForOwningWave) {
  auto builder = makeVmcntTest();
  builder.globalToLds(kWave, /*ldsAddrs=*/{0}, /*bytesPerLane=*/kDwordBytes, /*exec=*/1);
  issueOrderedVmemLoads(builder, kVmcntCapacity);

  expectLdsReady(builder);
}

TEST(RaceDetector, CounterCapacity_DirectToLdsRemainsPendingBelowVmcntCapacity) {
  auto builder = makeVmcntTest();
  builder.globalToLds(kWave, /*ldsAddrs=*/{0}, /*bytesPerLane=*/kDwordBytes, /*exec=*/1);
  issueOrderedVmemLoads(builder, kVmcntCapacity - 1);

  expectLdsPending(builder);
}
