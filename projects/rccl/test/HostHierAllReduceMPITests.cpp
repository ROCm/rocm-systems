/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file HostHierAllReduceMPITests.cpp
 * @brief MPI end-to-end tests for host-hierarchical AllReduce
 *
 * RCCL_HOST_HIER_ALLREDUCE is cached per process, so the suite skips unless the
 * binary is launched with RCCL_HOST_HIER_ALLREDUCE=1. Forcing it on makes every
 * eligible call take the hierarchical path on any uniform layout with at least
 * two ranks per host, including a single host. Each test checks the output data
 * and whether the communicator built its host-hierarchical child communicators,
 * so a correct fallback cannot hide a regression.
 */

#ifdef MPI_TESTS_ENABLED

#include "MPIHelpers.hpp"
#include "MPITestBase.hpp"
#include "TestChecks.hpp"
#include "comm.h"
#include "hier_host.h"
#include "rccl_common.h"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

using namespace MPITestConstants;

RCCL_PARAM_DECLARE(HostHierAllReduce);

namespace
{
struct DeviceBuffer
{
    void* ptr = nullptr;
    ~DeviceBuffer()
    {
        if(ptr)
            (void)hipFree(ptr);
    }
};

struct RunResult
{
    hipError_t   hip        = hipSuccess;
    ncclResult_t nccl       = ncclSuccess;
    size_t       mismatches = 0;
};

uint16_t toBf16(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}

float fromBf16(uint16_t value)
{
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float    out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

template<typename T>
RunResult runAllReduce(ncclComm_t                             comm,
                       hipStream_t                            stream,
                       size_t                                 count,
                       ncclDataType_t                         datatype,
                       ncclRedOp_t                            op,
                       bool                                   inPlace,
                       bool                                   grouped,
                       const std::function<T(size_t)>&        input,
                       const std::function<bool(size_t, T)>& check)
{
    RunResult      result;
    std::vector<T> host(count);
    for(size_t i = 0; i < count; i++)
        host[i] = input(i);

    const size_t bytes = count * sizeof(T);
    DeviceBuffer send, recv;
    if((result.hip = hipMalloc(&send.ptr, bytes)) != hipSuccess)
        return result;
    if(!inPlace && (result.hip = hipMalloc(&recv.ptr, bytes)) != hipSuccess)
        return result;
    void* out = inPlace ? send.ptr : recv.ptr;
    if((result.hip = hipMemcpy(send.ptr, host.data(), bytes, hipMemcpyHostToDevice)) != hipSuccess)
        return result;

    if(grouped)
        result.nccl = ncclGroupStart();
    if(result.nccl == ncclSuccess)
        result.nccl = ncclAllReduce(send.ptr, out, count, datatype, op, comm, stream);
    if(grouped)
    {
        ncclResult_t end = ncclGroupEnd();
        if(result.nccl == ncclSuccess)
            result.nccl = end;
    }
    if(result.nccl != ncclSuccess)
        return result;

    if((result.hip = hipStreamSynchronize(stream)) != hipSuccess)
        return result;
    if((result.hip = hipMemcpy(host.data(), out, bytes, hipMemcpyDeviceToHost)) != hipSuccess)
        return result;
    for(size_t i = 0; i < count; i++)
        if(!check(i, host[i]))
            result.mismatches++;
    return result;
}

#define ASSERT_RUN_OK(run)                          \
    do                                              \
    {                                               \
        ASSERT_MPI_EQ(hipSuccess, (run).hip);       \
        ASSERT_MPI_EQ(ncclSuccess, (run).nccl);     \
        ASSERT_MPI_EQ(size_t{0}, (run).mismatches); \
    }                                               \
    while(0)
} // namespace

class HostHierAllReduceMPITest : public MPITestBase
{
protected:
    int            rank_   = 0;
    int            nRanks_ = 0;
    rcclHostLayout layout_{};

    void SetUp() override
    {
        MPITestBase::SetUp();
        if(!validateTestPrerequisites(kMinProcessesForMPI))
            GTEST_SKIP() << "Need at least 2 MPI ranks";
        if(rcclParamHostHierAllReduce() != 1)
            GTEST_SKIP() << "RCCL_HOST_HIER_ALLREDUCE was not cached as 1; "
                            "launch with RCCL_HOST_HIER_ALLREDUCE=1 to force the hierarchical path";

        ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
        ASSERT_MPI_EQ(ncclSuccess, ncclCommUserRank(comm(), &rank_));
        ASSERT_MPI_EQ(ncclSuccess, ncclCommCount(comm(), &nRanks_));

        const std::string skip =
            mpiCoordinatedSkipReason(!rcclHostLayoutGet(comm(), &layout_),
                                     "Hosts are not uniform with at least two ranks each");
        if(!skip.empty())
            GTEST_SKIP() << skip;
    }

    ncclComm_t comm() { return getActiveCommunicator(); }

    size_t countFor(size_t perHostRank) const { return perHostRank * layout_.ranksPerHost; }

    bool builtChildComms() { return comm()->hostHierState == rcclHostHierReady; }

    // Sum over ranks of (rank + i) % mod; small integers stay exact in fp32 and bf16.
    float rankSumMod(size_t i, int mod) const
    {
        float sum = 0.0f;
        for(int r = 0; r < nRanks_; r++)
            sum += static_cast<float>((r + i) % mod);
        return sum;
    }

    RunResult fp32Sum(size_t count, bool inPlace, bool grouped = false)
    {
        return runAllReduce<float>(
            comm(), getActiveStream(), count, ncclFloat32, ncclSum, inPlace, grouped,
            [this](size_t i) { return static_cast<float>((rank_ + i) % 16); },
            [this](size_t i, float v) { return v == rankSumMod(i, 16); });
    }
};

TEST_F(HostHierAllReduceMPITest, Fp32SumOutOfPlace)
{
    RunResult run = fp32Sum(countFor(128 * 1024), false);
    ASSERT_RUN_OK(run);
    ASSERT_MPI_TRUE(builtChildComms());
}

TEST_F(HostHierAllReduceMPITest, Fp32SumInPlace)
{
    RunResult run = fp32Sum(countFor(128 * 1024), true);
    ASSERT_RUN_OK(run);
    ASSERT_MPI_TRUE(builtChildComms());
}

TEST_F(HostHierAllReduceMPITest, Bf16Sum)
{
    if(3 * nRanks_ > 256)
        GTEST_SKIP() << "bf16 sums are exact only up to 256";
    RunResult run = runAllReduce<uint16_t>(
        comm(), getActiveStream(), countFor(64 * 1024), ncclBfloat16, ncclSum, false, false,
        [this](size_t i) { return toBf16(static_cast<float>((rank_ + i) % 4)); },
        [this](size_t i, uint16_t v) { return fromBf16(v) == rankSumMod(i, 4); });
    ASSERT_RUN_OK(run);
    ASSERT_MPI_TRUE(builtChildComms());
}

TEST_F(HostHierAllReduceMPITest, Int32Max)
{
    auto value = [](int rank, size_t i) {
        return static_cast<int32_t>((i * 31 + static_cast<size_t>(rank) * 17) % 1009);
    };
    RunResult run = runAllReduce<int32_t>(
        comm(), getActiveStream(), countFor(64 * 1024), ncclInt32, ncclMax, false, false,
        [&](size_t i) { return value(rank_, i); },
        [&](size_t i, int32_t v) {
            int32_t expected = value(0, i);
            for(int r = 1; r < nRanks_; r++)
                expected = std::max(expected, value(r, i));
            return v == expected;
        });
    ASSERT_RUN_OK(run);
    ASSERT_MPI_TRUE(builtChildComms());
}

TEST_F(HostHierAllReduceMPITest, Fp32Avg)
{
    float rankMean = 0.0f;
    for(int r = 0; r < nRanks_; r++)
        rankMean += static_cast<float>(r % 5);
    rankMean /= static_cast<float>(nRanks_);

    RunResult run = runAllReduce<float>(
        comm(), getActiveStream(), countFor(64 * 1024), ncclFloat32, ncclAvg, false, false,
        [this](size_t i) { return static_cast<float>(rank_ % 5 + i % 3); },
        [&](size_t i, float v) {
            const float expected = rankMean + static_cast<float>(i % 3);
            return std::fabs(v - expected) <= 1e-5f * std::max(1.0f, std::fabs(expected));
        });
    ASSERT_RUN_OK(run);
    ASSERT_MPI_TRUE(builtChildComms());
}

TEST_F(HostHierAllReduceMPITest, GrowingSizesReuseChildComms)
{
    for(size_t perHostRank : {size_t{1024}, size_t{64 * 1024}, size_t{1024 * 1024}})
    {
        RunResult run = fp32Sum(countFor(perHostRank), false);
        ASSERT_RUN_OK(run);
        ASSERT_MPI_TRUE(builtChildComms());
    }
}

// With 8 ranks per host this is 208 MiB of fp32: three full 64 MiB rounds of the fused kernel and a shorter last one.
// Launch with RCCL_HOST_HIER_ALLREDUCE_FUSED=1 to run it in the fused kernel, which auto mode does not pick at this size.
// The pattern repeats every 13 elements, so data landing a power-of-two distance away does not match by chance.
TEST_F(HostHierAllReduceMPITest, ShortLastRound)
{
    RunResult run = runAllReduce<float>(
        comm(), getActiveStream(), countFor((size_t{13} << 22) / layout_.ranksPerHost), ncclFloat32, ncclSum, false,
        false, [this](size_t i) { return static_cast<float>((rank_ + i) % 13); },
        [this](size_t i, float v) { return v == rankSumMod(i, 13); });
    ASSERT_RUN_OK(run);
    ASSERT_MPI_TRUE(builtChildComms());
}

TEST_F(HostHierAllReduceMPITest, UnevenCountUsesDefaultPath)
{
    RunResult run = fp32Sum(countFor(1024) + 1, false);
    ASSERT_RUN_OK(run);
    ASSERT_MPI_FALSE(builtChildComms());
}

TEST_F(HostHierAllReduceMPITest, GroupedCallUsesDefaultPath)
{
    RunResult run = fp32Sum(countFor(1024), false, true);
    ASSERT_RUN_OK(run);
    ASSERT_MPI_FALSE(builtChildComms());
}

#endif // MPI_TESTS_ENABLED
