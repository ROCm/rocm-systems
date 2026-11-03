/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Pool sizing in test/common/ProcessIsolatedTestRunner.cpp.
//
// clampPoolToDeviceCount() is the one place that reconciles a detected pool
// with the device count HIP reports. Why it is needed, and why it trims by
// length rather than by value, is documented on the declaration in
// ProcessIsolatedTestRunner.hpp; these cases pin that contract.

#include <gtest/gtest.h>

#include <vector>

#include "../common/ProcessIsolatedTestRunner.hpp"

namespace
{

using RcclUnitTesting::clampPoolToDeviceCount;

std::vector<int> poolOf(int n)
{
    std::vector<int> pool(static_cast<size_t>(n));
    for(int i = 0; i < n; ++i) pool[static_cast<size_t>(i)] = i;
    return pool;
}

// The CI case: an 8-GPU host topology seen from a container where HIP can open
// four devices. Ordinals >= 4 must never reach a child.
TEST(ProcessIsolatedRunnerPool, TruncatesPoolToTheDeviceCountHipReports)
{
    EXPECT_EQ(clampPoolToDeviceCount(poolOf(8), 4), (std::vector<int>{0, 1, 2, 3}));
}

TEST(ProcessIsolatedRunnerPool, KeepsPoolWhenHipReportsAsManyDevicesAsTheSource)
{
    EXPECT_EQ(clampPoolToDeviceCount(poolOf(4), 4), poolOf(4));
}

// A pool smaller than the machine is a deliberate restriction (for example an
// explicit HIP_VISIBLE_DEVICES); the clamp must not widen it back out.
TEST(ProcessIsolatedRunnerPool, KeepsPoolWhenHipReportsMoreDevicesThanTheSource)
{
    EXPECT_EQ(clampPoolToDeviceCount(poolOf(2), 8), poolOf(2));
}

// The kept entries are the leading ones in source order, whatever ids those
// happen to be.
TEST(ProcessIsolatedRunnerPool, KeepsTheLeadingSourceIdsInSourceOrder)
{
    EXPECT_EQ(clampPoolToDeviceCount({3, 1, 7, 5}, 2), (std::vector<int>{3, 1}));
}

// An 8-GPU host restricted to HIP_VISIBLE_DEVICES=4,5,6,7: every id is >= the
// reported count, yet all four are openable. Trimming by value would empty it.
TEST(ProcessIsolatedRunnerPool, KeepsIdsAboveTheDeviceCountWhenAllOfThemAreVisible)
{
    EXPECT_EQ(clampPoolToDeviceCount({4, 5, 6, 7}, 4), (std::vector<int>{4, 5, 6, 7}));
}

// A failed probe and a genuinely GPU-less machine both leave the pool alone:
// emptying it would turn a per-test skip into a pool-detection failure.
TEST(ProcessIsolatedRunnerPool, KeepsPoolWhenTheDeviceCountIsUnknown)
{
    EXPECT_EQ(clampPoolToDeviceCount(poolOf(8), -1), poolOf(8));
}

TEST(ProcessIsolatedRunnerPool, KeepsPoolWhenHipReportsNoDevices)
{
    EXPECT_EQ(clampPoolToDeviceCount(poolOf(8), 0), poolOf(8));
}

TEST(ProcessIsolatedRunnerPool, LeavesAnUndetectedPoolEmpty)
{
    EXPECT_TRUE(clampPoolToDeviceCount({}, 8).empty());
}

} // namespace
