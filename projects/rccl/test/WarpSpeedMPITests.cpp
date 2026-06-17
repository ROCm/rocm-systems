/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/*
    This test evaluates the capability of the WarpSpeed feature to correctly
    handle back-to-back enqueues of different message sizes, both below and
    above the threshold, without synchronization.
    This test is requires warpSpeed enablement and will skip if RCCL_WARP_SPEED_AUTO=0 
    or if architecture doesn't support warpSpeed.
*/

#include "DeviceBufferHelpers.hpp"
#include "MPITestBase.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"
#include "comm.h"
#include "rccl_common.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>

#ifdef ENABLE_WARP_SPEED

using namespace MPITestConstants;
using namespace RCCLTestGuards;
using namespace RCCLTestHelpers;

namespace WarpSpeedConstants
{
    constexpr int64_t kSmallBufferThresholdDivisor    = 64;
    constexpr int64_t kLargeBufferThresholdMultiplier = 2;

    inline size_t smallBufferElementCount(int64_t threshold_bytes)
    {
        const int64_t bytes = threshold_bytes / kSmallBufferThresholdDivisor;
        return static_cast<size_t>(bytes / static_cast<int64_t>(sizeof(hip_bfloat16)));
    }

    inline size_t largeBufferElementCount(int64_t threshold_bytes)
    {
        const int64_t bytes = threshold_bytes * kLargeBufferThresholdMultiplier;
        return static_cast<size_t>(bytes / static_cast<int64_t>(sizeof(hip_bfloat16)));
    }

    constexpr int kSmallOpsBefore = 10;
    constexpr int kSmallOpsAfter  = 10;

    constexpr int kTrainIterations = 30;
}

using namespace WarpSpeedConstants;

class WarpSpeedMPITest : public MPITestBase
{
protected:
    hipStream_t stream_{};
    void*       d_small_{};
    void*       d_large_{};
    int64_t     ar_threshold_bytes_{};
    size_t      small_count_{};
    size_t      large_count_{};

    void skipUnlessWarpSpeedAutoEnabled()
    {
        if(rcclParamWarpSpeedAutoMode() == 0)
        {
            GTEST_SKIP() << "RCCL_WARP_SPEED_AUTO is not 1 (WarpSpeed auto not enabled)";
        }
        struct ncclComm* comm = reinterpret_cast<struct ncclComm*>(getActiveCommunicator());
        if(!rcclCanUseWarpSpeedAuto(comm, comm->nNodes))
        {
            GTEST_SKIP() << "RCCL_WARP_SPEED_AUTO=1 but auto mode unavailable (gfx950, "
                            "single-node required)";
        }
    }

    void initBufferCounts()
    {
        ar_threshold_bytes_ = rcclParamWarpSpeedARThreshold();
        ASSERT_GE(ar_threshold_bytes_,
                  kSmallBufferThresholdDivisor * static_cast<int64_t>(sizeof(hip_bfloat16)))
            << "RCCL_WARP_SPEED_AR_THRESHOLD too small for small buffer (threshold/64)";
        small_count_ = smallBufferElementCount(ar_threshold_bytes_);
        large_count_ = largeBufferElementCount(ar_threshold_bytes_);
        TEST_INFO("RCCL_WARP_SPEED_AR_THRESHOLD=%ld bytes, small=%zu elems (%ld bytes), "
                  "large=%zu elems (%ld bytes)",
                  static_cast<long>(ar_threshold_bytes_),
                  small_count_,
                  static_cast<long>(small_count_ * sizeof(hip_bfloat16)),
                  large_count_,
                  static_cast<long>(large_count_ * sizeof(hip_bfloat16)));
    }

    void TearDown() override
    {
        if(d_small_) { hipFree(d_small_); d_small_ = nullptr; }
        if(d_large_) { hipFree(d_large_); d_large_ = nullptr; }
        if(stream_)
        {
            hipStreamSynchronize(stream_);
            hipStreamDestroy(stream_);
            stream_ = {};
        }
        MPITestBase::TearDown();
    }

    ncclResult_t allocateBuffers()
    {
        HIPCHECK(hipMalloc(&d_small_, small_count_ * sizeof(hip_bfloat16)));
        HIPCHECK(hipMalloc(&d_large_, large_count_ * sizeof(hip_bfloat16)));
        return ncclSuccess;
    }

    /** Expected value after numInPlaceSumOps consecutive in-place AllReduce sums. */
    static double expectedInPlaceAllReduceSum(int nranks, int numInPlaceSumOps)
    {
        const double base = static_cast<double>(nranks) * (nranks - 1) / 2.0;
        if(numInPlaceSumOps <= 1)
            return base;
        return base * std::pow(static_cast<double>(nranks), numInPlaceSumOps - 1);
    }

    ncclResult_t prepareIteration()
    {
        const int rank = MPIEnvironment::world_rank;
        const auto pattern = [rank](size_t) {
            return hip_bfloat16(static_cast<float>(rank));
        };
        HIPCHECK(initializeBufferWithPattern<hip_bfloat16>(d_small_, small_count_, pattern));
        HIPCHECK(initializeBufferWithPattern<hip_bfloat16>(d_large_, large_count_, pattern));
        return ncclSuccess;
    }

    bool verifyBuffers(double expected_small,
                      double expected_large) const
    {
        const auto expect_small = [expected_small](size_t) {
            return hip_bfloat16(static_cast<float>(expected_small));
        };
        const auto expect_large = [expected_large](size_t) {
            return hip_bfloat16(static_cast<float>(expected_large));
        };
        const bool small_ok = verifyBufferData<hip_bfloat16>(
            d_small_, small_count_, expect_small);
        const bool large_ok = verifyBufferData<hip_bfloat16>(
            d_large_, large_count_, expect_large);
        return small_ok && large_ok;
    }

    void runValidatedTrainingLoop()
    {
        const int    nranks         = MPIEnvironment::world_size;
        const int    small_ops      = kSmallOpsBefore + kSmallOpsAfter;
        const double expected_small = expectedInPlaceAllReduceSum(nranks, small_ops);
        const double expected_large = expectedInPlaceAllReduceSum(nranks, 1);

        int correct_count = 0;
        int wrong_count   = 0;

        for(int iter = 0; iter < kTrainIterations; iter++)
        {
            ASSERT_MPI_EQ(ncclSuccess, prepareIteration());
            ASSERT_MPI_EQ(ncclSuccess, submitIteration());

            if(verifyBuffers(expected_small, expected_large))
                correct_count++;
            else
                wrong_count++;
        }

        TEST_INFO("Expected small: %.6g (%d ops), large: %.6g (1 op), Correct: %d/%d, Wrong: %d",
                  expected_small,
                  small_ops,
                  expected_large,
                  correct_count,
                  kTrainIterations,
                  wrong_count);

        EXPECT_EQ(correct_count, kTrainIterations)
            << "AllReduce result mismatch on small and/or large buffer (possible silent "
               "corruption)";
    }

    ncclResult_t submitIteration()
    {
        ncclComm_t comm = getActiveCommunicator();

        for(int i = 0; i < kSmallOpsBefore; i++)
            RCCL_TEST_CHECK(ncclAllReduce(d_small_, d_small_, small_count_,
                                          ncclBfloat16, ncclSum, comm, stream_));

        RCCL_TEST_CHECK(ncclAllReduce(d_large_, d_large_, large_count_,
                                      ncclBfloat16, ncclSum, comm, stream_));

        for(int i = 0; i < kSmallOpsAfter; i++)
            RCCL_TEST_CHECK(ncclAllReduce(d_small_, d_small_, small_count_,
                                          ncclBfloat16, ncclSum, comm, stream_));

        HIPCHECK(hipStreamSynchronize(stream_));
        return ncclSuccess;
    }
};

TEST_F(WarpSpeedMPITest, MixedThresholdRace)
{
    int  min_processes = 8;
    int  max_processes = 8;
    int  min_nodes     = 1;
    int  max_nodes     = 1;
    ASSERT_TRUE(validateTestPrerequisites(min_processes, max_processes, kNoPowerOfTwoRequired, min_nodes, max_nodes))
        << "Requires exactly 8 ranks on a single node";

    ASSERT_EQ(ncclSuccess, createTestCommunicator());
    skipUnlessWarpSpeedAutoEnabled();
    HIP_TEST_CHECK_GTEST_FAIL(hipStreamCreate(&stream_));
    initBufferCounts();
    ASSERT_MPI_EQ(ncclSuccess, allocateBuffers());

    TEST_INFO("%d non-WS + 1 WS + %d non-WS AllReduce per iter, no intermediate sync",
              kSmallOpsBefore, kSmallOpsAfter);

    runValidatedTrainingLoop();

    TEST_INFO("All %d iterations passed", kTrainIterations);
}

#endif // ENABLE_WARP_SPEED
