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
    const size_t totalBytes = kDdaAlltoAllGfx942ThresholdBytes;
    EXPECT_TRUE(testRcclDdaEnabled(
        mockComm_.get(),
        totalBytes,
        kDdaAlltoAllGfx942ThresholdBytes,
        kDdaAlltoAllGfx950ThresholdBytes));
    EXPECT_TRUE(testRcclDdaAlltoAllThresholdEnabled(
        mockComm_.get(), kAlltoAllFloat32CountAt4MbThreshold, ncclFloat32));
}

TEST_F(DdaAlltoAllThresholdTest, Gfx942_OneByteOverThreshold_Disabled)
{
    mockComm_.reset("gfx942:sramecc+:xnack-");
    const size_t totalBytes = kDdaAlltoAllGfx942ThresholdBytes + 1;
    EXPECT_FALSE(testRcclDdaEnabled(
        mockComm_.get(),
        totalBytes,
        kDdaAlltoAllGfx942ThresholdBytes,
        kDdaAlltoAllGfx950ThresholdBytes));
    EXPECT_FALSE(testRcclDdaAlltoAllThresholdEnabled(
        mockComm_.get(), kAlltoAllFloat32CountAt4MbThreshold + 1, ncclFloat32));
}

TEST_F(DdaAlltoAllThresholdTest, Gfx950_ExactlyAt4MbThreshold_Enabled)
{
    mockComm_.reset("gfx950:sramecc+:xnack-");
    const size_t totalBytes = kDdaAlltoAllGfx950ThresholdBytes;
    EXPECT_TRUE(testRcclDdaEnabled(
        mockComm_.get(),
        totalBytes,
        kDdaAlltoAllGfx942ThresholdBytes,
        kDdaAlltoAllGfx950ThresholdBytes));
    EXPECT_TRUE(testRcclDdaAlltoAllThresholdEnabled(
        mockComm_.get(), kAlltoAllFloat32CountAt4MbThreshold, ncclFloat32));
}

TEST_F(DdaAlltoAllThresholdTest, Gfx950_AlltoAllIgnoresHighUserThreshold)
{
    mockComm_.reset("gfx950:sramecc+:xnack-");
    const size_t overCap = kDdaAlltoAllGfx950ThresholdBytes + 1;
    EXPECT_FALSE(testRcclDdaEnabled(
        mockComm_.get(),
        overCap,
        kDdaAlltoAllGfx942ThresholdBytes,
        kDdaAlltoAllGfx950ThresholdBytes));

    // Other collectives on gfx950 still honor the user threshold when gfx950Default is 0.
    const size_t eightMb = 8 * 1024 * 1024;
    EXPECT_TRUE(testRcclDdaEnabled(mockComm_.get(), eightMb, 8388608, 0));
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

TEST_F(DdaAlltoAllThresholdTest, StagingBytesAtThresholdMatches4Mb)
{
    const size_t stagingBytes = testAlltoAllDdaIpcStagingBytes(
        kAlltoAllFloat32CountAt4MbThreshold,
        nccl_dda_detail::kDdaNranks,
        sizeof(float));
    EXPECT_EQ(stagingBytes, kDdaAlltoAllGfx950ThresholdBytes);
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

} // namespace RcclUnitTesting
