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
 *
 * On one node every rank has the same host hash, so the inter-host stage would
 * never run. RCCL_TEST_HOST_HIER_HOSTS=<n> splits the ranks into n emulated hosts
 * of consecutive ranks through a per-rank NCCL_HOSTID.
 */

#ifdef MPI_TESTS_ENABLED

#include "MPIHelpers.hpp"
#include "MPITestBase.hpp"
#include "TestChecks.hpp"
#include "algorithms/dda/all_reduce/dda_all_reduce.h"
#include "comm.h"
#include "hier_host.h"
#include "rccl_common.h"

#include <gtest/gtest.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

using namespace MPITestConstants;

RCCL_PARAM_DECLARE(HostHierAllReduce);
RCCL_PARAM_DECLARE(HostHierAllReduceFused);

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
                       const std::function<bool(size_t, T)>& check,
                       size_t                                 offsetBytes = 0)
{
    RunResult      result;
    std::vector<T> host(count);
    for(size_t i = 0; i < count; i++)
        host[i] = input(i);

    const size_t bytes = count * sizeof(T);
    DeviceBuffer sendAlloc, recvAlloc;
    if((result.hip = hipMalloc(&sendAlloc.ptr, bytes + offsetBytes)) != hipSuccess)
        return result;
    if(!inPlace && (result.hip = hipMalloc(&recvAlloc.ptr, bytes + offsetBytes)) != hipSuccess)
        return result;
    void* in  = static_cast<char*>(sendAlloc.ptr) + offsetBytes;
    void* out = inPlace ? in : static_cast<char*>(recvAlloc.ptr) + offsetBytes;
    if((result.hip = hipMemcpy(in, host.data(), bytes, hipMemcpyHostToDevice)) != hipSuccess)
        return result;

    if(grouped)
        result.nccl = ncclGroupStart();
    if(result.nccl == ncclSuccess)
        result.nccl = ncclAllReduce(in, out, count, datatype, op, comm, stream);
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
    int                                      rank_   = 0;
    int                                      nRanks_ = 0;
    rcclHostLayout                           layout_{};
    std::unique_ptr<MPIHelpers::MpiEnvGuard> hostGuard_;

    void SetUp() override
    {
        MPITestBase::SetUp();
        if(!validateTestPrerequisites(kMinProcessesForMPI))
            GTEST_SKIP() << "Need at least 2 MPI ranks";
        if(rcclParamHostHierAllReduce() != 1)
            GTEST_SKIP() << "RCCL_HOST_HIER_ALLREDUCE was not cached as 1; "
                            "launch with RCCL_HOST_HIER_ALLREDUCE=1 to force the hierarchical path";

        // getHostHash reads NCCL_HOSTID once per process, so this id holds for later tests too.
        const char* hostsEnv      = getenv("RCCL_TEST_HOST_HIER_HOSTS");
        const int   emulatedHosts = hostsEnv != nullptr ? atoi(hostsEnv) : 0;
        if(emulatedHosts > 1)
        {
            const int         worldSize = MPIEnvironment::world_size;
            const std::string skipHosts = mpiCoordinatedSkipReason(
                worldSize % emulatedHosts != 0 || worldSize / emulatedHosts < 2,
                "RCCL_TEST_HOST_HIER_HOSTS needs at least two ranks per emulated host, evenly split");
            if(!skipHosts.empty())
                GTEST_SKIP() << skipHosts;
            const std::string hostId =
                "hier-host-" + std::to_string(MPIEnvironment::world_rank * emulatedHosts / worldSize);
            hostGuard_ = std::make_unique<MPIHelpers::MpiEnvGuard>("NCCL_HOSTID", hostId.c_str());
        }

        ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
        ASSERT_MPI_EQ(ncclSuccess, ncclCommUserRank(comm(), &rank_));
        ASSERT_MPI_EQ(ncclSuccess, ncclCommCount(comm(), &nRanks_));

        const std::string skip =
            mpiCoordinatedSkipReason(!rcclHostLayoutGet(comm(), &layout_),
                                     "Hosts are not uniform with at least two ranks each");
        if(!skip.empty())
            GTEST_SKIP() << skip;
        if(emulatedHosts > 1)
            ASSERT_MPI_EQ(emulatedHosts, layout_.nHosts);
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

    // With RCCL_HOST_HIER_ALLREDUCE_FUSED cached as 1, a call the fused kernel takes leaves the three collectives'
    // scratch unallocated, so a silent fallback to them fails here.
    bool skippedFusedKernel(size_t count, ncclDataType_t datatype)
    {
        return rcclParamHostHierAllReduceFused() == 1 &&
               ncclAllReduceDdaFabricHierEligible(comm(), count, datatype, ncclSum, layout_.nHosts,
                                                  layout_.ranksPerHost) &&
               comm()->hostHier->scratch != nullptr;
    }
};

TEST_F(HostHierAllReduceMPITest, Fp32SumOutOfPlace)
{
    RunResult run = fp32Sum(countFor(128 * 1024), false);
    ASSERT_RUN_OK(run);
    ASSERT_MPI_TRUE(builtChildComms());
    ASSERT_MPI_FALSE(skippedFusedKernel(countFor(128 * 1024), ncclFloat32));
}

TEST_F(HostHierAllReduceMPITest, Fp32SumInPlace)
{
    RunResult run = fp32Sum(countFor(128 * 1024), true);
    ASSERT_RUN_OK(run);
    ASSERT_MPI_TRUE(builtChildComms());
    ASSERT_MPI_FALSE(skippedFusedKernel(countFor(128 * 1024), ncclFloat32));
}

// Every rank's buffers are 4 bytes off 16-byte alignment, on the fused kernel and the three collectives alike.
TEST_F(HostHierAllReduceMPITest, UnalignedBuffersOnEveryRank)
{
    for(bool inPlace : {false, true})
    {
        RunResult run = runAllReduce<float>(
            comm(), getActiveStream(), countFor(128 * 1024), ncclFloat32, ncclSum, inPlace, false,
            [this](size_t i) { return static_cast<float>((rank_ + i) % 16); },
            [this](size_t i, float v) { return v == rankSumMod(i, 16); }, sizeof(float));
        ASSERT_RUN_OK(run);
        ASSERT_MPI_TRUE(builtChildComms());
    }
}

// Rank 0's buffers are 4 bytes off 16-byte alignment while its peers' are aligned, in the fused kernel, which picks its
// user-buffer access per rank. The child collectives of the three-collective path need the rank-uniform DDA LL128
// choice of AICOMRCCL-3999 for this, so the test runs only where the fused kernel does.
TEST_F(HostHierAllReduceMPITest, UnalignedBuffersOnOneRankInTheFusedKernel)
{
    const size_t count = countFor(128 * 1024);
    if(rcclParamHostHierAllReduceFused() != 1 ||
       !ncclAllReduceDdaFabricHierEligible(comm(), count, ncclFloat32, ncclSum, layout_.nHosts, layout_.ranksPerHost))
        GTEST_SKIP() << "launch with RCCL_HOST_HIER_ALLREDUCE_FUSED=1 on a DDA fabric comm";
    for(bool inPlace : {false, true})
    {
        RunResult run = runAllReduce<float>(
            comm(), getActiveStream(), count, ncclFloat32, ncclSum, inPlace, false,
            [this](size_t i) { return static_cast<float>((rank_ + i) % 16); },
            [this](size_t i, float v) { return v == rankSumMod(i, 16); }, rank_ == 0 ? sizeof(float) : 0);
        ASSERT_RUN_OK(run);
        ASSERT_MPI_TRUE(builtChildComms());
        ASSERT_MPI_FALSE(skippedFusedKernel(count, ncclFloat32));
    }
}

// 8 MiB, inside the window where auto mode picks the fused kernel on one or two hosts of 8 ranks.
TEST_F(HostHierAllReduceMPITest, Fp16SumFusedWindow)
{
    if(3 * nRanks_ > 2048)
        GTEST_SKIP() << "fp16 sums are exact only up to 2048";
    const size_t count = countFor((size_t{4} << 20) / layout_.ranksPerHost);
    RunResult    run   = runAllReduce<__half>(
        comm(), getActiveStream(), count, ncclFloat16, ncclSum, false, false,
        [this](size_t i) { return __float2half(static_cast<float>((rank_ + i) % 4)); },
        [this](size_t i, __half v) { return __half2float(v) == rankSumMod(i, 4); });
    ASSERT_RUN_OK(run);
    ASSERT_MPI_TRUE(builtChildComms());
    ASSERT_MPI_FALSE(skippedFusedKernel(count, ncclFloat16));
}

TEST_F(HostHierAllReduceMPITest, Bf16SumFusedWindow)
{
    if(3 * nRanks_ > 256)
        GTEST_SKIP() << "bf16 sums are exact only up to 256";
    const size_t count = countFor((size_t{4} << 20) / layout_.ranksPerHost);
    RunResult    run   = runAllReduce<uint16_t>(
        comm(), getActiveStream(), count, ncclBfloat16, ncclSum, false, false,
        [this](size_t i) { return toBf16(static_cast<float>((rank_ + i) % 4)); },
        [this](size_t i, uint16_t v) { return fromBf16(v) == rankSumMod(i, 4); });
    ASSERT_RUN_OK(run);
    ASSERT_MPI_TRUE(builtChildComms());
    ASSERT_MPI_FALSE(skippedFusedKernel(count, ncclBfloat16));
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
        // One stage's chunk, twice with an inter-host stage, which runs out of place. The fused kernel stages
        // through the DDA scratch instead, so sizes it takes do not grow this one.
        const size_t  bytes     = countFor(perHostRank) * sizeof(float);
        const int64_t fusedMode = rcclParamHostHierAllReduceFused();
        if(fusedMode == 1 || (fusedMode == -1 && rcclHostHierAllReduceFusedWindow(layout_.nHosts, layout_.ranksPerHost, bytes)))
            continue;
        const size_t chunkBytes = perHostRank * sizeof(float);
        ASSERT_MPI_TRUE(comm()->hostHier->scratchBytes >= (layout_.nHosts > 1 ? 2 : 1) * chunkBytes);
    }
}

// With 8 ranks per host this is 208 MiB of fp32: three full 64 MiB rounds of the fused kernel and a shorter last one.
// Launch with RCCL_HOST_HIER_ALLREDUCE_FUSED=1 to run it in the fused kernel, which auto mode does not pick at this size.
// The pattern repeats every 13 elements, so data landing a power-of-two distance away does not match by chance.
TEST_F(HostHierAllReduceMPITest, ShortLastRound)
{
    const size_t count = countFor((size_t{13} << 22) / layout_.ranksPerHost);
    RunResult    run   = runAllReduce<float>(
        comm(), getActiveStream(), count, ncclFloat32, ncclSum, false, false,
        [this](size_t i) { return static_cast<float>((rank_ + i) % 13); },
        [this](size_t i, float v) { return v == rankSumMod(i, 13); });
    ASSERT_RUN_OK(run);
    ASSERT_MPI_TRUE(builtChildComms());
    ASSERT_MPI_FALSE(skippedFusedKernel(count, ncclFloat32));
}

// A larger call on a second stream grows the scratch while the first call, on the test stream, may still read it.
TEST_F(HostHierAllReduceMPITest, LargerCallOnAnotherStreamGrowsScratch)
{
    const size_t counts[] = {countFor(64 * 1024), countFor(1024 * 1024)};
    hipStream_t  streams[2] = {getActiveStream(), nullptr};
    ASSERT_MPI_EQ(hipSuccess, hipStreamCreateWithFlags(&streams[1], hipStreamNonBlocking));
    DeviceBuffer             send[2], recv[2];
    std::vector<float>       host[2];
    hipError_t               hip  = hipSuccess;
    ncclResult_t             nccl = ncclSuccess;
    for(int c = 0; c < 2 && hip == hipSuccess; c++)
    {
        host[c].resize(counts[c]);
        for(size_t i = 0; i < counts[c]; i++)
            host[c][i] = static_cast<float>((rank_ + i) % 16);
        const size_t bytes = counts[c] * sizeof(float);
        if((hip = hipMalloc(&send[c].ptr, bytes)) == hipSuccess && (hip = hipMalloc(&recv[c].ptr, bytes)) == hipSuccess)
            hip = hipMemcpy(send[c].ptr, host[c].data(), bytes, hipMemcpyHostToDevice);
    }
    for(int c = 0; c < 2 && hip == hipSuccess && nccl == ncclSuccess; c++)
        nccl = ncclAllReduce(send[c].ptr, recv[c].ptr, counts[c], ncclFloat32, ncclSum, comm(), streams[c]);
    for(int c = 0; c < 2 && hip == hipSuccess; c++)
        hip = hipStreamSynchronize(streams[c]);
    size_t mismatches = 0;
    for(int c = 0; c < 2 && hip == hipSuccess; c++)
    {
        hip = hipMemcpy(host[c].data(), recv[c].ptr, counts[c] * sizeof(float), hipMemcpyDeviceToHost);
        for(size_t i = 0; hip == hipSuccess && i < counts[c]; i++)
            mismatches += host[c][i] != rankSumMod(i, 16);
    }
    (void)hipStreamDestroy(streams[1]);
    ASSERT_MPI_EQ(hipSuccess, hip);
    ASSERT_MPI_EQ(ncclSuccess, nccl);
    ASSERT_MPI_EQ(size_t{0}, mismatches);
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

// Every rank but rank 0 starts a call, so the child stages wait for it until the revoke aborts them. The revoke
// syncs the comm's streams, so it only returns if the abort reaches the child comms.
TEST_F(HostHierAllReduceMPITest, RevokeAbortsChildStagesWaitingOnAPeer)
{
    const size_t count = countFor(64 * 1024);
    RunResult    run   = fp32Sum(count, false);
    ASSERT_RUN_OK(run);
    ASSERT_MPI_TRUE(builtChildComms());

    // The same size as above, so the call needs no scratch-growth vote, which would wait for rank 0 on the host.
    DeviceBuffer send, recv;
    hipError_t   hip = hipMalloc(&send.ptr, count * sizeof(float));
    if(hip == hipSuccess)
        hip = hipMalloc(&recv.ptr, count * sizeof(float));
    ASSERT_MPI_EQ(hipSuccess, hip);
    ncclResult_t nccl = ncclSuccess;
    if(rank_ != 0)
        nccl = ncclAllReduce(send.ptr, recv.ptr, count, ncclFloat32, ncclSum, comm(), getActiveStream());
    ASSERT_MPI_EQ(ncclSuccess, nccl);

    MPI_Barrier(MPI_COMM_WORLD);
    ASSERT_MPI_EQ(ncclSuccess, ncclCommRevoke(comm(), NCCL_REVOKE_DEFAULT));
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));
}

#endif // MPI_TESTS_ENABLED
