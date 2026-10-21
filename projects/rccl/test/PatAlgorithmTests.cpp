/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "collectives.h"
#include "gtest/gtest.h"

#include <cstring>
#include <vector>

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

// Run getNextOp to completion, zeroing the slot before every call so fields a
// skipped step leaves untouched compare deterministically.
static std::vector<ncclPatStep> RunPatAG(PatAGAlgorithm<char>& algo)
{
    std::vector<ncclPatStep> out;
    ncclPatStep              ps;
    do
    {
        memset(&ps, 0, sizeof(ps));
        algo.getNextOp(&ps);
        out.push_back(ps);
        if(out.size() >= 100000)
        {
            ADD_FAILURE() << "getNextOp did not terminate";
            break;
        }
    } while(ps.last != 2);
    return out;
}

// Splits a multi-chunk PAT AllGather at chunk boundaries and requires every chunk to
// issue the same ops as a fresh single-chunk run over that chunk's range: offset must
// advance by chunkCount, reset() must run per chunk, and only the final chunk (which
// carries the short tail) may raise last.
static void CheckChunkedAgainstReferences(size_t end, const std::vector<int>& nelems)
{
    constexpr size_t kCount = 2500;
    constexpr int    kChunk = 1000;
    const int        ranks[] = {3, 5, 8, 17};

    for(int nranks : ranks)
    {
        for(int rank = 0; rank < nranks; ++rank)
        {
            for(int shared = 0; shared <= 1; ++shared)
            {
                SCOPED_TRACE(::testing::Message()
                             << "end=" << end << " nranks=" << nranks << " rank=" << rank
                             << " shared=" << shared);
                PatAGAlgorithm<char> algo(512, NCCL_STEPS, 2, 0, end, kCount, kChunk, rank,
                                          nranks, shared);
                auto steps = RunPatAG(algo);
                ASSERT_FALSE(steps.empty());

                size_t pos = 0;
                for(size_t c = 0; c < nelems.size(); ++c)
                {
                    SCOPED_TRACE(::testing::Message() << "chunk=" << c);
                    const bool           tail = (c + 1 == nelems.size());
                    const size_t         off  = c * kChunk;
                    PatAGAlgorithm<char> ref(512, NCCL_STEPS, 2, off, off + nelems[c], kCount,
                                             kChunk, rank, nranks, shared);
                    auto expect = RunPatAG(ref);
                    ASSERT_LE(pos + expect.size(), steps.size());
                    for(size_t i = 0; i < expect.size(); ++i, ++pos)
                    {
                        const ncclPatStep& got = steps[pos];
                        const ncclPatStep& exp = expect[i];
                        SCOPED_TRACE(::testing::Message() << "step=" << pos);
                        EXPECT_EQ(got.nelem, nelems[c]);
                        EXPECT_EQ(got.nelem, exp.nelem);
                        EXPECT_EQ(got.inpIx, exp.inpIx);
                        EXPECT_EQ(got.outIx, exp.outIx);
                        EXPECT_EQ(got.recvDim, exp.recvDim);
                        EXPECT_EQ(got.sendDim, exp.sendDim);
                        EXPECT_EQ(got.recvOffset, exp.recvOffset);
                        EXPECT_EQ(got.sendOffset, exp.sendOffset);
                        EXPECT_EQ(got.stepOffset, exp.stepOffset);
                        EXPECT_EQ(got.postRecv, exp.postRecv);
                        EXPECT_EQ(got.postSend, exp.postSend);
                        EXPECT_EQ(got.flags, exp.flags);
                        // last is raised only in the final chunk; the reference marks its own end.
                        EXPECT_EQ(got.last, tail ? exp.last : 0);
                    }
                }
                EXPECT_EQ(pos, steps.size()) << "extra ops after the final chunk";
                EXPECT_EQ(steps.back().last, 2);
                for(size_t k = 0; k + 1 < steps.size(); ++k)
                    EXPECT_NE(steps[k].last, 2);
            }
        }
    }
}

TEST(PatAGAlgorithmTests, MultiChunkTailMatchesPerChunkReference)
{
    CheckChunkedAgainstReferences(2500, {1000, 1000, 500});
}

// Control: end is an exact multiple of chunkCount, so the last chunk is full.
TEST(PatAGAlgorithmTests, MultiChunkExactMultipleHasFullLastChunk)
{
    CheckChunkedAgainstReferences(3000, {1000, 1000, 1000});
}

} // namespace RcclUnitTesting
