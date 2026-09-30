/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "hip_rocm_version_info.h"
#include "gtest/gtest.h"

namespace RcclUnitTesting
{
    constexpr const char* kHipBase  = "HIP version  : 6.4.0";
    constexpr const char* kRocmBase = "ROCm version : 6.4.0";

    TEST(VersionInfoTests, DecodeHipVer)
    {
        VersionInfo v = decodeHipVer(60443210);
        EXPECT_TRUE(v.valid);
        EXPECT_EQ(v.major, 6u);
        EXPECT_EQ(v.minor, 4u);
        EXPECT_EQ(v.patch, 43210u);
    }

    TEST(VersionInfoTests, SameVer)
    {
        EXPECT_TRUE(sameVer({true, 6, 4, 0}, {true, 6, 4, 0}));
        EXPECT_FALSE(sameVer({true, 6, 4, 0}, {true, 6, 4, 1}));
        EXPECT_FALSE(sameVer({true, 6, 4, 0}, {true, 6, 3, 0}));
        EXPECT_TRUE(sameVer({false, 6, 4, 0}, {true, 6, 4, 0}));
    }

    TEST(VersionInfoTests, OmitMatch)
    {
        std::string s = fmtExtVer(
            kHipBase,  {true, 6, 4, 0}, {true, 6, 4, 0},
            kRocmBase, {true, 6, 4, 0}, {true, 6, 4, 0});
        EXPECT_EQ(s, "HIP version  : 6.4.0\nROCm version : 6.4.0");
    }

    TEST(VersionInfoTests, OmitInvalid)
    {
        std::string s = fmtExtVer(
            kHipBase,  {false, 9, 9, 9}, {true, 6, 4, 0},
            kRocmBase, {false, 9, 9, 9}, {true, 6, 4, 0});
        EXPECT_EQ(s, "HIP version  : 6.4.0\nROCm version : 6.4.0");
    }

    TEST(VersionInfoTests, AppendDiffer)
    {
        std::string s = fmtExtVer(
            kHipBase,  {true, 6, 5, 1}, {true, 6, 4, 0},
            kRocmBase, {true, 6, 5, 0}, {true, 6, 4, 0});
        EXPECT_EQ(s,
            "HIP version  : 6.4.0 / HIP runtime  : 6.5.1\n"
            "ROCm version : 6.4.0 / ROCm runtime : 6.5.0");
    }

    TEST(VersionInfoTests, PatchOnlyDiffer)
    {
        std::string s = fmtExtVer(
            kHipBase,  {true, 6, 4, 1}, {true, 6, 4, 0},
            kRocmBase, {true, 6, 4, 1}, {true, 6, 4, 0});
        EXPECT_EQ(s,
            "HIP version  : 6.4.0 / HIP runtime  : 6.4.1\n"
            "ROCm version : 6.4.0 / ROCm runtime : 6.4.1");
    }

    TEST(VersionInfoTests, MixedDiffer)
    {
        std::string s = fmtExtVer(
            kHipBase,  {true, 6, 5, 0}, {true, 6, 4, 0},
            kRocmBase, {true, 6, 4, 0}, {true, 6, 4, 0});
        EXPECT_EQ(s,
            "HIP version  : 6.4.0 / HIP runtime  : 6.5.0\n"
            "ROCm version : 6.4.0");
    }
}
