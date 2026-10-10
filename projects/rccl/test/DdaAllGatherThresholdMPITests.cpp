/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file DdaAllGatherThresholdMPITests.cpp
 * @brief DDA IPC AllGather with RCCL_DDA_THRESHOLD above the IPC scratch.
 *
 * Run (example):
 *   mpirun -np 8 -x RCCL_DDA_THRESHOLD=1073741824 ./rccl-UnitTestsMPI
 *     --gtest_filter=DdaMPI_AllGatherThreshold.*
 */

#ifdef MPI_TESTS_ENABLED

#include "DeviceBufferHelpers.hpp"
#include "MPIHelpers.hpp"
#include "MPITestBase.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"
#include "algorithms/dda/dda_init_detail.h"
#include "comm.h"
#include "rccl_common.h"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <string>

using namespace RCCLTestGuards;
using namespace RCCLTestHelpers;

namespace
{
constexpr int kDdaIpcRanks = nccl_dda_detail::kDdaNranks;

bool isDdaIpcArch()
{
    hipDeviceProp_t props{};
    if(hipGetDeviceProperties(&props, 0) != hipSuccess)
    {
        return false;
    }
    const std::string arch(props.gcnArchName);
    return arch.find("gfx942") != std::string::npos
           || arch.find("gfx950") != std::string::npos;
}

// Unique per rank and exact in fp32 (max 4095 * 8 + 7 < 2^24).
float rankPattern(int rank, size_t i)
{
    return static_cast<float>((i % 4096) * 8 + static_cast<size_t>(rank));
}
} // namespace

class DdaMPI_AllGatherThreshold : public MPITestBase
{
protected:
    int selectedAlgo(const void* sendBuf, void* recvBuf, size_t count)
    {
        int algo        = -1;
        int protocol    = -1;
        int maxChannels = -1;
        EXPECT_EQ(ncclSuccess,
                  rcclGetCollImplInfo(getActiveCommunicator(),
                                      ncclFuncAllGather, count, ncclFloat32,
                                      ncclSum, sendBuf, recvBuf,
                                      /*graphCapturing=*/0, &algo, &protocol,
                                      &maxChannels));
        return algo;
    }
};

// A total above scratch but under the threshold must fall back, not fail.
TEST_F(DdaMPI_AllGatherThreshold, AboveScratchFallsBack)
{
    const size_t threshold = 4 * static_cast<size_t>(DDA_IPC_BUFFER_SIZE);
    // RCCL caches the param on first read, so set it before comm init.
    MPIHelpers::MpiEnvGuard thresholdGuard(
        "RCCL_DDA_THRESHOLD", std::to_string(threshold).c_str());
    if(!validateTestPrerequisites(kDdaIpcRanks, kDdaIpcRanks, false, 1, 1))
    {
        GTEST_SKIP() << "DDA IPC AllGather needs 8 ranks on one node";
    }
    const bool optedOut = rcclParamDdaEnable() == 0;
    std::string skip = mpiCoordinatedSkipReason(
        !isDdaIpcArch() || optedOut,
        optedOut ? "RCCL_DDA_ENABLE=0" : "DDA IPC needs gfx942 or gfx950");
    if(!skip.empty())
    {
        GTEST_SKIP() << skip;
    }
    skip = mpiCoordinatedSkipReason(
        static_cast<size_t>(rcclParamDdaThreshold()) != threshold,
        "RCCL_DDA_THRESHOLD was read earlier; run alone or set it at launch");
    if(!skip.empty())
    {
        GTEST_SKIP() << skip;
    }

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm* comm = getActiveCommunicator();
    int rank = -1;
    int nRanks = 0;
    ASSERT_MPI_EQ(ncclSuccess, ncclCommUserRank(comm, &rank));
    ASSERT_MPI_EQ(ncclSuccess, ncclCommCount(comm, &nRanks));
    skip = mpiCoordinatedSkipReason(comm->config.launchOrderImplicit == 1,
                                    "NCCL_LAUNCH_ORDER_IMPLICIT=1");
    if(!skip.empty())
    {
        GTEST_SKIP() << skip;
    }

    // 2x the per-rank share that fills scratch: 64 MiB each, 512 MiB total.
    const size_t ranks      = static_cast<size_t>(nRanks);
    const size_t fullCount  = comm->ddaScratchBytes / (ranks * sizeof(float));
    const size_t count      = 2 * fullCount;
    const size_t totalCount = count * ranks;
    ASSERT_MPI_TRUE(count * sizeof(float) <= comm->ddaScratchBytes);
    ASSERT_MPI_TRUE(totalCount * sizeof(float) > comm->ddaScratchBytes);
    ASSERT_MPI_TRUE(totalCount * sizeof(float) <= threshold);

    void* sendBuf = nullptr;
    void* recvBuf = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&sendBuf, count * sizeof(float)));
    DeviceBufferAutoGuard sendGuard(sendBuf);
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&recvBuf, totalCount * sizeof(float)));
    DeviceBufferAutoGuard recvGuard(recvBuf);

    // Control: a total that exactly fills scratch still takes DDA IPC.
    ASSERT_MPI_EQ(RCCL_DDA_IPC, selectedAlgo(sendBuf, recvBuf, fullCount));
    EXPECT_NE(RCCL_DDA_IPC, selectedAlgo(sendBuf, recvBuf, count))
        << "Rank " << rank << ": DDA IPC picked for a total above scratch";

    ASSERT_MPI_EQ(hipSuccess,
                  initializeBufferWithPattern<float>(
                      sendBuf, count,
                      [rank](size_t i) { return rankPattern(rank, i); }));
    ASSERT_MPI_EQ(hipSuccess,
                  hipMemset(recvBuf, 0, totalCount * sizeof(float)));
    ASSERT_MPI_EQ(ncclSuccess,
                  ncclAllGather(sendBuf, recvBuf, count, ncclFloat32, comm,
                                getActiveStream()));
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));
    ASSERT_MPI_TRUE(verifyBufferData<float>(
        recvBuf, totalCount, [count](size_t j) {
            return rankPattern(static_cast<int>(j / count), j % count);
        }));
}

#endif // MPI_TESTS_ENABLED
