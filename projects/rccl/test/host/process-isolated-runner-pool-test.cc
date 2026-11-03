/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Pool sizing in test/common/ProcessIsolatedTestRunner.cpp.
//
// The runner hands each isolated child a slice of its GPU pool via
// HIP_VISIBLE_DEVICES. Neither pool source can be trusted to describe the
// devices this process can open: KFD topology sysfs is not namespaced per
// container, and an inherited HIP_VISIBLE_DEVICES is just whatever the caller
// set. clampPoolToDeviceCount() is the one place that reconciles the detected
// pool with the device count HIP reports, so it is pinned here.

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

// HIP exposes the longest valid prefix of HIP_VISIBLE_DEVICES and reports its
// length, so the kept entries are the leading ones in source order, whatever
// ids those happen to be.
TEST(ProcessIsolatedRunnerPool, KeepsTheLeadingSourceIdsInSourceOrder)
{
    EXPECT_EQ(clampPoolToDeviceCount({3, 1, 7, 5}, 2), (std::vector<int>{3, 1}));
}

// The ids are absolute device numbers, not offsets into the pool, and a count
// equal to the pool's length means every one of them is openable. Discarding
// ids >= the count instead of trimming by length would empty this pool, which
// is what an 8-GPU host restricted to HIP_VISIBLE_DEVICES=4,5,6,7 looks like.
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
