// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file waitcnt_counter_test.cpp
/// @brief Phase D unit tests for WaitCounters and WaitTarget.

#include "rocjitsu/isa/arch/amdgpu/shared/wait_counter_policy.h"
#include "rocjitsu/vm/amdgpu/wait_counters.h"

#include <gtest/gtest.h>

namespace {

using namespace rocjitsu::amdgpu;

// ---------------------------------------------------------------------------
// WaitCounters — increment and decrement
// ---------------------------------------------------------------------------

TEST(WaitCounterTest, IncrementVmcnt) {
  WaitCounters c;
  c.increment(WaitCounterType::VMCNT);
  EXPECT_EQ(c.vmcnt, 1);
  c.increment(WaitCounterType::VMCNT);
  EXPECT_EQ(c.vmcnt, 2);
}

TEST(WaitCounterTest, DecrementVmcnt) {
  WaitCounters c;
  c.increment(WaitCounterType::VMCNT);
  c.increment(WaitCounterType::VMCNT);
  c.decrement(WaitCounterType::VMCNT);
  EXPECT_EQ(c.vmcnt, 1);
}

TEST(WaitCounterTest, IncrementLgkmcnt) {
  WaitCounters c;
  c.increment(WaitCounterType::LGKMCNT);
  EXPECT_EQ(c.lgkmcnt, 1);
}

TEST(WaitCounterTest, IncrementExpcnt) {
  WaitCounters c;
  c.increment(WaitCounterType::EXPCNT);
  EXPECT_EQ(c.expcnt, 1);
}

TEST(WaitCounterTest, IncrementVscnt) {
  WaitCounters c;
  c.increment(WaitCounterType::VSCNT);
  EXPECT_EQ(c.vscnt, 1);
}

TEST(WaitCounterTest, IncrementStorecnt) {
  WaitCounters c;
  c.increment(WaitCounterType::STORECNT);
  EXPECT_EQ(c.vscnt, 1); // STORECNT aliases VSCNT
}

TEST(WaitCounterTest, IncrementLoadcnt) {
  WaitCounters c;
  c.increment(WaitCounterType::LOADCNT);
  EXPECT_EQ(c.vmcnt, 1); // LOADCNT aliases VMCNT
}

TEST(WaitCounterTest, IncrementDscnt) {
  WaitCounters c;
  c.increment(WaitCounterType::DSCNT);
  EXPECT_EQ(c.dscnt, 1);
  EXPECT_EQ(c.lgkmcnt, 1); // DSCNT also increments LGKMCNT
}

TEST(WaitCounterTest, IncrementKmcnt) {
  WaitCounters c;
  c.increment(WaitCounterType::KMCNT);
  EXPECT_EQ(c.kmcnt, 1);
  EXPECT_EQ(c.lgkmcnt, 1); // KMCNT also increments LGKMCNT
}

TEST(WaitCounterTest, IncrementTensorcnt) {
  WaitCounters c;
  c.increment(WaitCounterType::TENSORCNT);
  EXPECT_EQ(c.tensorcnt, 1);
}

TEST(WaitCounterTest, DecrementDscnt) {
  WaitCounters c;
  c.increment(WaitCounterType::DSCNT);
  c.decrement(WaitCounterType::DSCNT);
  EXPECT_EQ(c.dscnt, 0);
  EXPECT_EQ(c.lgkmcnt, 0);
}

TEST(WaitCounterTest, DecrementKmcnt) {
  WaitCounters c;
  c.increment(WaitCounterType::KMCNT);
  c.decrement(WaitCounterType::KMCNT);
  EXPECT_EQ(c.kmcnt, 0);
  EXPECT_EQ(c.lgkmcnt, 0);
}

TEST(WaitCounterTest, MixedDscntKmcnt) {
  WaitCounters c;
  c.increment(WaitCounterType::DSCNT);
  c.increment(WaitCounterType::KMCNT);
  EXPECT_EQ(c.lgkmcnt, 2); // Both contribute to LGKMCNT
  EXPECT_EQ(c.dscnt, 1);
  EXPECT_EQ(c.kmcnt, 1);
  c.decrement(WaitCounterType::DSCNT);
  EXPECT_EQ(c.lgkmcnt, 1);
}

TEST(WaitCounterTest, SaturationAtMax) {
  WaitCounters c;
  for (int i = 0; i < 100; ++i)
    c.increment(WaitCounterType::VMCNT);
  EXPECT_EQ(c.vmcnt, WaitCounters::VMCNT_MAX);
}

// ---------------------------------------------------------------------------
// Wait-counter families
// ---------------------------------------------------------------------------

TEST(WaitCounterCoverageTest, MonolithicCountersCoverTheirSplitFamilies) {
  EXPECT_TRUE(wait_counter_covers(WaitCounterType::VMCNT, WaitCounterType::VMCNT));
  EXPECT_TRUE(wait_counter_covers(WaitCounterType::VMCNT, WaitCounterType::LOADCNT));
  EXPECT_FALSE(wait_counter_covers(WaitCounterType::VMCNT, WaitCounterType::STORECNT));

  EXPECT_TRUE(wait_counter_covers(WaitCounterType::LGKMCNT, WaitCounterType::LGKMCNT));
  EXPECT_TRUE(wait_counter_covers(WaitCounterType::LGKMCNT, WaitCounterType::DSCNT));
  EXPECT_TRUE(wait_counter_covers(WaitCounterType::LGKMCNT, WaitCounterType::KMCNT));
  EXPECT_FALSE(wait_counter_covers(WaitCounterType::LGKMCNT, WaitCounterType::VMCNT));

  EXPECT_TRUE(wait_counter_covers(WaitCounterType::VSCNT, WaitCounterType::VSCNT));
  EXPECT_TRUE(wait_counter_covers(WaitCounterType::VSCNT, WaitCounterType::STORECNT));
  EXPECT_FALSE(wait_counter_covers(WaitCounterType::VSCNT, WaitCounterType::LOADCNT));
}

TEST(WaitCounterCoverageTest, SplitCountersCoverOnlyThemselves) {
  EXPECT_TRUE(wait_counter_covers(WaitCounterType::LOADCNT, WaitCounterType::LOADCNT));
  EXPECT_FALSE(wait_counter_covers(WaitCounterType::LOADCNT, WaitCounterType::VMCNT));
  EXPECT_TRUE(wait_counter_covers(WaitCounterType::STORECNT, WaitCounterType::STORECNT));
  EXPECT_FALSE(wait_counter_covers(WaitCounterType::STORECNT, WaitCounterType::VSCNT));
  EXPECT_TRUE(wait_counter_covers(WaitCounterType::DSCNT, WaitCounterType::DSCNT));
  EXPECT_FALSE(wait_counter_covers(WaitCounterType::DSCNT, WaitCounterType::LGKMCNT));
  EXPECT_TRUE(wait_counter_covers(WaitCounterType::KMCNT, WaitCounterType::KMCNT));
  EXPECT_FALSE(wait_counter_covers(WaitCounterType::KMCNT, WaitCounterType::LGKMCNT));
  EXPECT_TRUE(wait_counter_covers(WaitCounterType::EXPCNT, WaitCounterType::EXPCNT));
  EXPECT_TRUE(wait_counter_covers(WaitCounterType::TENSORCNT, WaitCounterType::TENSORCNT));
  EXPECT_TRUE(wait_counter_covers(WaitCounterType::ASYNCCNT, WaitCounterType::ASYNCCNT));
}

TEST(WaitCounterPolicy, SupportedRuntimeTagsRoundTripToCoveringWaits) {
  struct Case {
    WaitcntModel model;
    WaitCounterType tag;
    WaitCounterKind domain;
    WaitCounterType wait;
  };
  using enum WaitcntModel;
  using enum WaitCounterType;
  const Case cases[] = {
      {LegacyNoVscnt, VMCNT, WaitCounterKind::Load, VMCNT},
      {LegacyNoVscnt, LGKMCNT, WaitCounterKind::Ds, LGKMCNT},
      {LegacyNoVscnt, EXPCNT, WaitCounterKind::Exp, EXPCNT},
      {LegacyVscnt, VMCNT, WaitCounterKind::Load, VMCNT},
      {LegacyVscnt, LGKMCNT, WaitCounterKind::Ds, LGKMCNT},
      {LegacyVscnt, EXPCNT, WaitCounterKind::Exp, EXPCNT},
      {LegacyVscnt, VSCNT, WaitCounterKind::Store, VSCNT},
      {LegacyVscnt, LOADCNT, WaitCounterKind::Load, VMCNT},
      {LegacyVscnt, STORECNT, WaitCounterKind::Store, VSCNT},
      {LegacyVscnt, DSCNT, WaitCounterKind::Ds, LGKMCNT},
      {LegacyVscnt, KMCNT, WaitCounterKind::Ds, LGKMCNT},
      {SplitGfx12, LOADCNT, WaitCounterKind::Load, LOADCNT},
      {SplitGfx12, STORECNT, WaitCounterKind::Store, STORECNT},
      {SplitGfx12, DSCNT, WaitCounterKind::Ds, DSCNT},
      {SplitGfx12, KMCNT, WaitCounterKind::Km, KMCNT},
      {SplitGfx12, EXPCNT, WaitCounterKind::Exp, EXPCNT},
      {SplitGfx12, ASYNCCNT, WaitCounterKind::Async, ASYNCCNT},
      {SplitGfx12, TENSORCNT, WaitCounterKind::Tensor, TENSORCNT},
  };
  for (const auto &c : cases) {
    SCOPED_TRACE(static_cast<unsigned>(c.model));
    SCOPED_TRACE(static_cast<unsigned>(c.tag));
    const auto domain = canonical_wait_counter(c.tag, c.model);
    EXPECT_EQ(domain, c.domain);
    const auto wait = memory_wait_counter_type(domain, c.model);
    ASSERT_TRUE(wait);
    EXPECT_EQ(*wait, c.wait);
    EXPECT_TRUE(wait_counter_covers(*wait, c.tag));
  }
}

TEST(WaitCounterPolicy, UnmappedDomainsHaveNoMemoryPipelineWait) {
  for (auto model :
       {WaitcntModel::LegacyNoVscnt, WaitcntModel::LegacyVscnt, WaitcntModel::SplitGfx12}) {
    for (auto domain : {WaitCounterKind::Sample, WaitCounterKind::Bvh, WaitCounterKind::X,
                        WaitCounterKind::VmVsrc, WaitCounterKind::VaVdst, WaitCounterKind::Depctr,
                        WaitCounterKind::Count}) {
      SCOPED_TRACE(static_cast<unsigned>(model));
      SCOPED_TRACE(static_cast<unsigned>(domain));
      EXPECT_FALSE(memory_wait_counter_type(domain, model));
    }
  }
}

// ---------------------------------------------------------------------------
// WaitTarget — satisfaction checks
// ---------------------------------------------------------------------------

TEST(WaitTargetTest, DefaultSatisfied) {
  WaitTarget target;
  WaitCounters counters;
  EXPECT_TRUE(target.satisfied(counters));
}

TEST(WaitTargetTest, VmcntNotSatisfied) {
  WaitTarget target;
  target.vmcnt = 0;
  WaitCounters counters;
  counters.vmcnt = 1;
  EXPECT_FALSE(target.satisfied(counters));
}

TEST(WaitTargetTest, VmcntSatisfied) {
  WaitTarget target;
  target.vmcnt = 2;
  WaitCounters counters;
  counters.vmcnt = 2;
  EXPECT_TRUE(target.satisfied(counters));
}

TEST(WaitTargetTest, DscntNotSatisfied) {
  WaitTarget target;
  target.dscnt = 0;
  WaitCounters counters;
  counters.dscnt = 1;
  counters.lgkmcnt = 1;
  EXPECT_FALSE(target.satisfied(counters));
}

TEST(WaitTargetTest, TensorcntNotSatisfied) {
  WaitTarget target;
  target.tensorcnt = 0;
  WaitCounters counters;
  counters.tensorcnt = 1;
  EXPECT_FALSE(target.satisfied(counters));
}

TEST(WaitTargetTest, MultipleCounters) {
  WaitTarget target;
  target.vmcnt = 0;
  target.lgkmcnt = 0;
  WaitCounters counters;
  counters.vmcnt = 1;
  counters.lgkmcnt = 0;
  EXPECT_FALSE(target.satisfied(counters)); // vmcnt > 0
  counters.vmcnt = 0;
  EXPECT_TRUE(target.satisfied(counters));
}

} // namespace
