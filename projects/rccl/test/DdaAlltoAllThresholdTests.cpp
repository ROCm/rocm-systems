/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "common/DdaAlltoAllTestHelpers.hpp"
#include "common/ProcessIsolatedTestRunner.hpp"

#include "gtest/gtest.h"

namespace RcclUnitTesting
{

class DdaAlltoAllThresholdTest : public ::testing::Test
{
protected:
    DdaAlltoAllMockComm mockComm_;
};

TEST_F(DdaAlltoAllThresholdTest, Gfx942_ExactlyAt4MbThreshold_Enabled)
{
    mockComm_.reset("gfx942:sramecc+:xnack-");
    const size_t totalBytes = rcclGetArchThresholds("gfx942")->ddaVmmMax[ncclFuncAlltoAll];
    EXPECT_TRUE(rcclDdaEnabled(mockComm_.get(), totalBytes, rcclGetArchThresholds("gfx942")->ddaVmmMax[ncclFuncAlltoAll]));
    EXPECT_TRUE(testRcclDdaAlltoAllThresholdEnabled(
        mockComm_.get(), kAlltoAllFloat32CountAt4MbThreshold, ncclFloat32));
}

TEST_F(DdaAlltoAllThresholdTest, Gfx942_OneByteOverThreshold_Disabled)
{
    mockComm_.reset("gfx942:sramecc+:xnack-");
    const size_t totalBytes = rcclGetArchThresholds("gfx942")->ddaVmmMax[ncclFuncAlltoAll] + 1;
    EXPECT_FALSE(rcclDdaEnabled(mockComm_.get(), totalBytes, rcclGetArchThresholds("gfx942")->ddaVmmMax[ncclFuncAlltoAll]));
    EXPECT_FALSE(testRcclDdaAlltoAllThresholdEnabled(
        mockComm_.get(), kAlltoAllFloat32CountAt4MbThreshold + 1, ncclFloat32));
}

TEST_F(DdaAlltoAllThresholdTest, Gfx950_ExactlyAt4MbThreshold_Enabled)
{
    mockComm_.reset("gfx950:sramecc+:xnack-");
    const size_t totalBytes = rcclGetArchThresholds("gfx950")->ddaVmmMax[ncclFuncAlltoAll];
    EXPECT_TRUE(rcclDdaEnabled(mockComm_.get(), totalBytes, rcclGetArchThresholds("gfx950")->ddaVmmMax[ncclFuncAlltoAll]));
    EXPECT_TRUE(testRcclDdaAlltoAllThresholdEnabled(
        mockComm_.get(), kAlltoAllFloat32CountAt4MbThreshold, ncclFloat32));
}

TEST_F(DdaAlltoAllThresholdTest, Gfx950_AlltoAllIgnoresHighUserThreshold)
{
    mockComm_.reset("gfx950:sramecc+:xnack-");
    const size_t overCap = rcclGetArchThresholds("gfx950")->ddaVmmMax[ncclFuncAlltoAll] + 1;
    EXPECT_FALSE(rcclDdaEnabled(mockComm_.get(), overCap, rcclGetArchThresholds("gfx950")->ddaVmmMax[ncclFuncAlltoAll]));

    // Other collectives on gfx950 keep the arch table's much larger cap.
    const size_t eightMb = 8 * 1024 * 1024;
    EXPECT_TRUE(rcclDdaEnabled(mockComm_.get(), eightMb,
                               rcclDdaVmmThreshold(mockComm_.get(), ncclFuncAllReduce)));
}

TEST_F(DdaAlltoAllThresholdTest, Gfx1250_ExactlyAt4MbThreshold_Enabled)
{
    mockComm_.reset("gfx1250:sramecc+:xnack-");
    const size_t totalBytes = rcclGetArchThresholds("gfx1250")->ddaVmmMax[ncclFuncAlltoAll];
    EXPECT_TRUE(rcclDdaEnabled(
        mockComm_.get(),
        totalBytes,
        rcclGetArchThresholds("gfx1250")->ddaVmmMax[ncclFuncAlltoAll]));
    EXPECT_TRUE(testRcclDdaAlltoAllThresholdEnabled(
        mockComm_.get(), kAlltoAllFloat32CountAt4MbThreshold, ncclFloat32));
}

TEST_F(DdaAlltoAllThresholdTest, Gfx1250_OneByteOverThreshold_Disabled)
{
    mockComm_.reset("gfx1250:sramecc+:xnack-");
    const size_t totalBytes = rcclGetArchThresholds("gfx1250")->ddaVmmMax[ncclFuncAlltoAll] + 1;
    EXPECT_FALSE(rcclDdaEnabled(
        mockComm_.get(),
        totalBytes,
        rcclGetArchThresholds("gfx1250")->ddaVmmMax[ncclFuncAlltoAll]));
    EXPECT_FALSE(testRcclDdaAlltoAllThresholdEnabled(
        mockComm_.get(), kAlltoAllFloat32CountAt4MbThreshold + 1, ncclFloat32));
}

TEST_F(DdaAlltoAllThresholdTest, Gfx1250_AlltoAllIgnoresHighUserThreshold)
{
    mockComm_.reset("gfx1250:sramecc+:xnack-");
    const size_t overCap = rcclGetArchThresholds("gfx1250")->ddaVmmMax[ncclFuncAlltoAll] + 1;
    EXPECT_FALSE(rcclDdaEnabled(
        mockComm_.get(),
        overCap,
        rcclGetArchThresholds("gfx1250")->ddaVmmMax[ncclFuncAlltoAll]));
}

TEST_F(DdaAlltoAllThresholdTest, UnsupportedArch_Disabled)
{
    mockComm_.reset("gfx1100");
    EXPECT_FALSE(testRcclDdaAlltoAllThresholdEnabled(
        mockComm_.get(), kAlltoAllFloat32CountAt4MbThreshold, ncclFloat32));
}

TEST_F(DdaAlltoAllThresholdTest, FewerThanEightRanks_Disabled)
{
    mockComm_.reset("gfx950:sramecc+:xnack-");
    mockComm_.comm.nRanks = 4;
    EXPECT_FALSE(testRcclDdaAlltoAllThresholdEnabled(
        mockComm_.get(), kAlltoAllFloat32CountAt4MbThreshold, ncclFloat32));
}

TEST_F(DdaAlltoAllThresholdTest, SymmetricSupport_Disabled)
{
    mockComm_.reset("gfx950:sramecc+:xnack-");
    mockComm_.comm.symmetricSupport = 1;
    EXPECT_FALSE(testRcclDdaAlltoAllThresholdEnabled(
        mockComm_.get(), kAlltoAllFloat32CountAt4MbThreshold, ncclFloat32));
}

TEST_F(DdaAlltoAllThresholdTest, Gfx950_4KbPerRank_UsesInKernelStagingCopy)
{
    mockComm_.reset("gfx950:sramecc+:xnack-");
    EXPECT_TRUE(testRcclDdaAlltoAllThresholdEnabled(
        mockComm_.get(), kAlltoAllFloat32CountAt4KbPerRank, ncclFloat32));
    EXPECT_TRUE(testAlltoAllUsesInKernelStagingCopy(
        kAlltoAllFloat32CountAt4KbPerRank, ncclFloat32));
}

TEST_F(DdaAlltoAllThresholdTest, Gfx950_8KbPerRank_UsesPreKernelMemcpy)
{
    mockComm_.reset("gfx950:sramecc+:xnack-");
    EXPECT_TRUE(testRcclDdaAlltoAllThresholdEnabled(
        mockComm_.get(), kAlltoAllFloat32CountAt8KbPerRank, ncclFloat32));
    EXPECT_FALSE(testAlltoAllUsesInKernelStagingCopy(
        kAlltoAllFloat32CountAt8KbPerRank, ncclFloat32));
}

TEST_F(DdaAlltoAllThresholdTest, StagingBytesAtThresholdMatches4Mb)
{
    const size_t stagingBytes = testAlltoAllDdaIpcStagingBytes(
        kAlltoAllFloat32CountAt4MbThreshold,
        nccl_dda_detail::kDdaNranks,
        sizeof(float));
    EXPECT_EQ(stagingBytes, rcclGetArchThresholds("gfx942")->ddaVmmMax[ncclFuncAlltoAll]);
}

TEST(DdaAlltoAllThreshold, DdaEnableOff_Disabled)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "DdaAlltoAllThreshold_DdaEnableOff",
        []()
        {
            DdaAlltoAllMockComm mockComm;
            mockComm.reset("gfx950:sramecc+:xnack-");
            EXPECT_FALSE(testRcclDdaAlltoAllThresholdEnabled(
                mockComm.get(), kAlltoAllFloat32CountAt4MbThreshold, ncclFloat32));
        },
        {{"RCCL_DDA_ENABLE", "0"}});
}

TEST(DdaAlltoAllThreshold, Gfx1250_FallbackToUserThreshold)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "Gfx1250_FallbackToUserThreshold",
        []()
        {
            DdaAlltoAllMockComm mockComm;
            mockComm.reset("gfx1250:sramecc+:xnack-");
            const size_t eightMb = 8 * 1024 * 1024;
            const size_t twelveMb = 12 * 1024 * 1024;
            // RCCL_DDA_THRESHOLD=10MiB overrides the arch table, so 8MiB passes.
            const size_t cap = rcclDdaVmmThreshold(mockComm.get(), ncclFuncAllReduce);
            EXPECT_TRUE(rcclDdaEnabled(mockComm.get(), eightMb, cap));
            // But 12MiB should fail (over 10MiB threshold).
            EXPECT_FALSE(rcclDdaEnabled(mockComm.get(), twelveMb, cap));
        },
        {{"RCCL_DDA_THRESHOLD", "10485760"}});  // 10 MiB
}

TEST(DdaAlltoAllThreshold, Gfx950_FallbackToUserThreshold)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "Gfx950_FallbackToUserThreshold",
        []()
        {
            DdaAlltoAllMockComm mockComm;
            mockComm.reset("gfx950:sramecc+:xnack-");
            const size_t eightMb = 8 * 1024 * 1024;
            const size_t twelveMb = 12 * 1024 * 1024;
            // RCCL_DDA_THRESHOLD=10MiB overrides the arch table, so 8MiB passes.
            const size_t cap = rcclDdaVmmThreshold(mockComm.get(), ncclFuncAllReduce);
            EXPECT_TRUE(rcclDdaEnabled(mockComm.get(), eightMb, cap));
            // But 12MiB should fail (over 10MiB threshold).
            EXPECT_FALSE(rcclDdaEnabled(mockComm.get(), twelveMb, cap));
        },
        {{"RCCL_DDA_THRESHOLD", "10485760"}});  // 10 MiB
}

TEST_F(DdaAlltoAllThresholdTest, Gfx1250_FourRanks_Enabled)
{
    mockComm_.reset("gfx1250:sramecc+:xnack-");
    mockComm_.comm.nRanks = 4;
    const size_t totalBytes = rcclGetArchThresholds("gfx1250")->ddaVmmMax[ncclFuncAlltoAll];
    // gfx1250 has no nRanks<8 gate, so DDA should be enabled at nRanks=4.
    EXPECT_TRUE(rcclDdaEnabled(
        mockComm_.get(),
        totalBytes,
        rcclGetArchThresholds("gfx1250")->ddaVmmMax[ncclFuncAlltoAll]));
}

TEST_F(DdaAlltoAllThresholdTest, Gfx942_FourRanks_Disabled)
{
    mockComm_.reset("gfx942:sramecc+:xnack-");
    mockComm_.comm.nRanks = 4;
    const size_t totalBytes = rcclGetArchThresholds("gfx942")->ddaVmmMax[ncclFuncAlltoAll];
    // gfx942 requires nRanks>=8, so DDA should be disabled at nRanks=4.
    EXPECT_FALSE(rcclDdaEnabled(
        mockComm_.get(),
        totalBytes,
        rcclGetArchThresholds("gfx942")->ddaVmmMax[ncclFuncAlltoAll]));
}

} // namespace RcclUnitTesting
