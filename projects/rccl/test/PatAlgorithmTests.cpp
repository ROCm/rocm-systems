/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "collectives.h"
#include "gtest/gtest.h"

namespace RcclUnitTesting
{

// PatAGAlgorithm::getNextOp is __host__ as well as __device__, so the shared-conn
// data-block index can be checked here without MPI or a GPU. The skip=1 case
// (s >= nranks) is what used to feed C++ a negative dividend.
TEST(PatAGAlgorithmTests, SharedDataRankStaysInRangeOnSkippedSteps)
{
    constexpr size_t kCount = 1024;
    constexpr int    kChunk = 1024;
    const int        ranks[] = {3, 4, 5, 6, 7, 8, 9, 15, 16, 17};

    for(int nranks : ranks)
    {
        for(int rank = 0; rank < nranks; ++rank)
        {
            for(int shared = 0; shared <= 1; ++shared)
            {
                PatAGAlgorithm<char> algo(/*stepSize=*/4096,
                                          NCCL_STEPS,
                                          /*maxParallelFactor=*/16,
                                          /*offset=*/0,
                                          /*end=*/kCount,
                                          kCount,
                                          kChunk,
                                          rank,
                                          nranks,
                                          shared);
                struct ncclPatStep ps    = {};
                int                steps = 0;
                do
                {
                    algo.getNextOp(&ps);
                    // outIx is size_t: a negative recvDataRank wraps to a huge offset.
                    EXPECT_LE(ps.outIx, static_cast<size_t>(nranks - 1) * kCount)
                        << "nranks=" << nranks << " rank=" << rank << " shared=" << shared
                        << " step=" << steps;
                    ASSERT_LT(++steps, 100000) << "getNextOp did not terminate";
                } while(ps.last != 2);
            }
        }
    }
}

} // namespace RcclUnitTesting
