/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include <gtest/gtest.h>

#include "common/ErrCode.hpp"
#include "common/ProcessIsolatedTestRunner.hpp"
#include "enqueue.h"

namespace RcclUnitTesting
{

class EnqueueCountTests : public ::testing::Test
{
};

TEST_F(EnqueueCountTests, ncclFuncSendCount_AllTests)
{
    ProcessIsolatedTestRunner::ExecutionOptions options;
    options.stopOnFirstFailure = false;
    options.verboseLogging     = true;

    RUN_ISOLATED_TESTS_WITH_OPTIONS(
        options,
        ProcessIsolatedTestRunner::TestConfig(
            "ncclFuncSendCount_AllReduce",
            []()
            {
                size_t count  = 1000;
                int    nRanks = 4;
                size_t result = ncclFuncSendCount(ncclFuncAllReduce, nRanks, count);
                EXPECT_EQ(result, count);
            }
        ),

        ProcessIsolatedTestRunner::TestConfig(
            "ncclFuncSendCount_Broadcast",
            []()
            {
                size_t count  = 1000;
                int    nRanks = 4;
                size_t result = ncclFuncSendCount(ncclFuncBroadcast, nRanks, count);
                EXPECT_EQ(result, count);
            }
        ),

        ProcessIsolatedTestRunner::TestConfig(
            "ncclFuncSendCount_Reduce",
            []()
            {
                size_t count  = 1000;
                int    nRanks = 4;
                size_t result = ncclFuncSendCount(ncclFuncReduce, nRanks, count);
                EXPECT_EQ(result, count);
            }
        ),

        ProcessIsolatedTestRunner::TestConfig(
            "ncclFuncSendCount_AllGather",
            []()
            {
                size_t count  = 1000;
                int    nRanks = 4;
                size_t result = ncclFuncSendCount(ncclFuncAllGather, nRanks, count);
                EXPECT_EQ(result, count);
            }
        ),

        ProcessIsolatedTestRunner::TestConfig(
            "ncclFuncSendCount_ReduceScatter",
            []()
            {
                size_t count  = 1000;
                int    nRanks = 4;
                size_t result = ncclFuncSendCount(ncclFuncReduceScatter, nRanks, count);
                EXPECT_EQ(result, count * nRanks);
            }
        ),

        ProcessIsolatedTestRunner::TestConfig(
            "ncclFuncSendCount_ZeroCount",
            []()
            {
                size_t result = ncclFuncSendCount(ncclFuncAllReduce, 4, 0);
                EXPECT_EQ(result, 0);
            }
        )
    );
}

TEST_F(EnqueueCountTests, ncclFuncRecvCount_AllTests)
{
    ProcessIsolatedTestRunner::ExecutionOptions options;
    options.stopOnFirstFailure = false;
    options.verboseLogging     = true;

    RUN_ISOLATED_TESTS_WITH_OPTIONS(
        options,
        ProcessIsolatedTestRunner::TestConfig(
            "ncclFuncRecvCount_AllReduce",
            []()
            {
                size_t count  = 1000;
                int    nRanks = 4;
                size_t result = ncclFuncRecvCount(ncclFuncAllReduce, nRanks, count);
                EXPECT_EQ(result, count);
            }
        ),

        ProcessIsolatedTestRunner::TestConfig(
            "ncclFuncRecvCount_Broadcast",
            []()
            {
                size_t count  = 1000;
                int    nRanks = 4;
                size_t result = ncclFuncRecvCount(ncclFuncBroadcast, nRanks, count);
                EXPECT_EQ(result, count);
            }
        ),

        ProcessIsolatedTestRunner::TestConfig(
            "ncclFuncRecvCount_Reduce",
            []()
            {
                size_t count  = 1000;
                int    nRanks = 4;
                size_t result = ncclFuncRecvCount(ncclFuncReduce, nRanks, count);
                EXPECT_EQ(result, count);
            }
        ),

        ProcessIsolatedTestRunner::TestConfig(
            "ncclFuncRecvCount_AllGather",
            []()
            {
                size_t count  = 1000;
                int    nRanks = 4;
                size_t result = ncclFuncRecvCount(ncclFuncAllGather, nRanks, count);
                EXPECT_EQ(result, count * nRanks);
            }
        ),

        ProcessIsolatedTestRunner::TestConfig(
            "ncclFuncRecvCount_ReduceScatter",
            []()
            {
                size_t count  = 1000;
                int    nRanks = 4;
                size_t result = ncclFuncRecvCount(ncclFuncReduceScatter, nRanks, count);
                EXPECT_EQ(result, count);
            }
        ),

        ProcessIsolatedTestRunner::TestConfig(
            "ncclFuncRecvCount_ZeroCount",
            []()
            {
                size_t result = ncclFuncRecvCount(ncclFuncAllReduce, 4, 0);
                EXPECT_EQ(result, 0);
            }
        )
    );
}

TEST_F(EnqueueCountTests, ncclFuncMaxSendRecvCount_AllTests)
{
    ProcessIsolatedTestRunner::ExecutionOptions options;
    options.stopOnFirstFailure = false;
    options.verboseLogging     = true;

    RUN_ISOLATED_TESTS_WITH_OPTIONS(
        options,
        ProcessIsolatedTestRunner::TestConfig(
            "ncclFuncMaxSendRecvCount_AllReduce",
            []()
            {
                size_t count  = 1000;
                int    nRanks = 4;
                size_t result = ncclFuncMaxSendRecvCount(ncclFuncAllReduce, nRanks, count);
                EXPECT_EQ(result, count);
            }
        ),

        ProcessIsolatedTestRunner::TestConfig(
            "ncclFuncMaxSendRecvCount_AllGather",
            []()
            {
                size_t count  = 1000;
                int    nRanks = 4;
                size_t result = ncclFuncMaxSendRecvCount(ncclFuncAllGather, nRanks, count);
                EXPECT_EQ(result, count * nRanks);
            }
        ),

        ProcessIsolatedTestRunner::TestConfig(
            "ncclFuncMaxSendRecvCount_ReduceScatter",
            []()
            {
                size_t count  = 1000;
                int    nRanks = 4;
                size_t result = ncclFuncMaxSendRecvCount(ncclFuncReduceScatter, nRanks, count);
                EXPECT_EQ(result, count * nRanks);
            }
        ),

        ProcessIsolatedTestRunner::TestConfig(
            "ncclFuncMaxSendRecvCount_ZeroCount",
            []()
            {
                size_t result = ncclFuncMaxSendRecvCount(ncclFuncAllReduce, 4, 0);
                EXPECT_EQ(result, 0);
            }
        )
    );
}

TEST_F(EnqueueCountTests, ncclFuncCounts_EdgeCases)
{
    ProcessIsolatedTestRunner::ExecutionOptions options;
    options.stopOnFirstFailure = false;
    options.verboseLogging     = true;

    RUN_ISOLATED_TESTS_WITH_OPTIONS(
        options,
        ProcessIsolatedTestRunner::TestConfig(
            "ncclFuncCounts_SingleRank",
            []()
            {
                size_t count  = 1000;
                int    nRanks = 1;
                EXPECT_EQ(ncclFuncSendCount(ncclFuncAllReduce, nRanks, count), count);
                EXPECT_EQ(ncclFuncRecvCount(ncclFuncAllReduce, nRanks, count), count);
                EXPECT_EQ(ncclFuncMaxSendRecvCount(ncclFuncAllReduce, nRanks, count), count);
            }
        ),

        ProcessIsolatedTestRunner::TestConfig(
            "ncclFuncCounts_LargeRankCount",
            []()
            {
                size_t count  = 1000;
                int    nRanks = 1024;
                EXPECT_EQ(ncclFuncSendCount(ncclFuncAllGather, nRanks, count), count);
                EXPECT_EQ(ncclFuncRecvCount(ncclFuncAllGather, nRanks, count), count * nRanks);
                EXPECT_EQ(
                    ncclFuncMaxSendRecvCount(ncclFuncAllGather, nRanks, count),
                    count * nRanks
                );
            }
        )
    );
}

} // namespace RcclUnitTesting
