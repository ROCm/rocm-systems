/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include EnvVars.hpp
#include common/TestChecks.hpp
#include gtest/gtest.h

namespace RcclUnitTesting
{
    // Verify numDetectedGpus matches hipGetDeviceCount.
    // Fails on NPS4 if an improper GPU cap is applied (e.g. the old cap of 16).
    TEST(EnvVarsTests, NumDetectedGpusMatchesHip)
    {
        int hipCount = 0;
        HIP_CHECK(hipGetDeviceCount(&hipCount));
        EnvVars envVars;
        EXPECT_EQ(envVars.GetNumDetectedGpus(), hipCount);
    }
}
