// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file waitcnt_counter_test.cpp
/// @brief Phase D unit tests for WaitCounters and WaitTarget.

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
