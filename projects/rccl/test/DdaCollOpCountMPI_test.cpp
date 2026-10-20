/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file DdaCollOpCountMPI_test.cpp
 * @brief MPI end-to-end test that DDA collectives print the per-collective COLL line and advance opCount
 *
 * DDA collectives launch through rcclAddonLaunch and return before ncclEnqueueCheck, which is where the
 * native path prints "<Coll>: opCount ..." and advances comm->opCount (ROCM-33092). Tools that trace
 * collectives from the RCCL log, such as the PyTorch FSDP force_sum tests, scrape that line.
 *
 * ReduceScatter and AllGather alternate on one stream with no host sync between calls, as in the
 * ticket's one-stream reproducer. Run (example):
 *   mpirun -np 2 ./rccl-UnitTestsMPI --gtest_filter=DdaMPI_CollOpCount.*
 */

#ifdef MPI_TESTS_ENABLED

#include "DdaFabricMPITestFixture.hpp"
#include "MPIHelpers.hpp"
#include "MPITestBase.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"
#include "comm.h"
#include "rccl_common.h"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <mpi.h>

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

using namespace MPITestConstants;
using namespace RCCLTestGuards;

namespace
{
constexpr char kRsNeedle[] = "NCCL INFO ReduceScatter: opCount ";
constexpr char kAgNeedle[] = "NCCL INFO AllGather: opCount ";

// The reproducer's shapes: both take a DDA tier at 2 to 8 ranks under the gfx1250 defaults.
constexpr size_t kRsRecvCount  = 1276;
constexpr size_t kAgSendCount  = 716;
constexpr int    kIters        = 3;
constexpr auto   kDrainTimeout = std::chrono::seconds(120);

bool isDdaAlgo(int algo)
{
    return algo == RCCL_DDA_FABRIC_LL || algo == RCCL_DDA_FABRIC_LL128 || algo == RCCL_DDA_FABRIC_VMM
           || algo == RCCL_DDA_IPC;
}
} // namespace

class DdaMPI_CollOpCount : public DdaFabricMPITest
{
};

TEST_F(DdaMPI_CollOpCount, DdaCallsLogOpCountLineAndAdvanceOpCount)
{
    if(!validateTestPrerequisites(kMinProcessesForMPI))
    {
        GTEST_SKIP() << "Need at least 2 MPI ranks";
    }

    DDA_FABRIC_SKIP_OR_FAIL(archSkipReason("DDA collectives are selected by default only on gfx1250"));

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t comm   = getActiveCommunicator();
    int        rank   = -1;
    int        nRanks = 0;
    ASSERT_MPI_EQ(ncclSuccess, ncclCommUserRank(comm, &rank));
    ASSERT_MPI_EQ(ncclSuccess, ncclCommCount(comm, &nRanks));

    void*            rsSend      = nullptr;
    const hipError_t rsSendAlloc = hipMalloc(&rsSend, kRsRecvCount * nRanks * sizeof(float));
    DeviceBufferAutoGuard rsSendGuard(rsSend);
    void*            rsRecv      = nullptr;
    const hipError_t rsRecvAlloc = hipMalloc(&rsRecv, kRsRecvCount * sizeof(float));
    DeviceBufferAutoGuard rsRecvGuard(rsRecv);
    void*            agSend      = nullptr;
    const hipError_t agSendAlloc = hipMalloc(&agSend, kAgSendCount * sizeof(float));
    DeviceBufferAutoGuard agSendGuard(agSend);
    void*            agRecv      = nullptr;
    const hipError_t agRecvAlloc = hipMalloc(&agRecv, kAgSendCount * nRanks * sizeof(float));
    DeviceBufferAutoGuard agRecvGuard(agRecv);
    ASSERT_MPI_EQ(hipSuccess, rsSendAlloc);
    ASSERT_MPI_EQ(hipSuccess, rsRecvAlloc);
    ASSERT_MPI_EQ(hipSuccess, agSendAlloc);
    ASSERT_MPI_EQ(hipSuccess, agRecvAlloc);

    // NCCL_ALGO, RCCL_DDA_ENABLE=0, lowered thresholds or a missing fabric all route these to the native path.
    int rsAlgo = -1, agAlgo = -1, protocol = -1, maxChannels = -1;
    ASSERT_MPI_EQ(ncclSuccess, rcclGetCollImplInfo(comm, ncclFuncReduceScatter, kRsRecvCount, ncclFloat32, ncclSum,
                                                   rsSend, rsRecv, /*graphCapturing=*/0, &rsAlgo, &protocol,
                                                   &maxChannels));
    ASSERT_MPI_EQ(ncclSuccess, rcclGetCollImplInfo(comm, ncclFuncAllGather, kAgSendCount, ncclFloat32, ncclSum,
                                                   agSend, agRecv, /*graphCapturing=*/0, &agAlgo, &protocol,
                                                   &maxChannels));
    const std::string notDdaSkip = mpiCoordinatedSkipReason(!isDdaAlgo(rsAlgo) || !isDdaAlgo(agAlgo),
                                                            "ReduceScatter/AllGather did not select a DDA tier");
    if(!notDdaSkip.empty())
    {
        GTEST_SKIP() << notDdaSkip;
    }

    std::vector<float> hostRsSend(kRsRecvCount * nRanks, static_cast<float>(rank + 1));
    std::vector<float> hostAgSend(kAgSendCount, static_cast<float>(rank + 1));
    ASSERT_MPI_EQ(hipSuccess, hipMemcpy(rsSend, hostRsSend.data(), hostRsSend.size() * sizeof(float),
                                        hipMemcpyHostToDevice));
    ASSERT_MPI_EQ(hipSuccess, hipMemcpy(agSend, hostAgSend.data(), hostAgSend.size() * sizeof(float),
                                        hipMemcpyHostToDevice));

    // After a failed launch this rank stops launching but keeps looping, so all ranks reach the same collectives.
    const uint64_t opCountBefore = comm->opCount;
    ncclResult_t   launchRes     = ncclSuccess;
    for(int i = 0; i < kIters && launchRes == ncclSuccess; ++i)
    {
        launchRes = ncclReduceScatter(rsSend, rsRecv, kRsRecvCount, ncclFloat32, ncclSum, comm, getActiveStream());
        if(launchRes == ncclSuccess)
        {
            launchRes = ncclAllGather(agSend, agRecv, kAgSendCount, ncclFloat32, comm, getActiveStream());
        }
    }
    DdaFabricMPI::abortJobUnlessAllRanksDrained(DdaFabricMPI::waitStreamBounded(getActiveStream(), kDrainTimeout),
                                                "DdaMPI_CollOpCount");
    ASSERT_MPI_EQ(ncclSuccess, launchRes);
    const uint64_t opCountAfter = comm->opCount;

    std::vector<float> hostRsRecv(kRsRecvCount);
    std::vector<float> hostAgRecv(kAgSendCount * nRanks);
    ASSERT_MPI_EQ(hipSuccess, hipMemcpy(hostRsRecv.data(), rsRecv, hostRsRecv.size() * sizeof(float),
                                        hipMemcpyDeviceToHost));
    ASSERT_MPI_EQ(hipSuccess, hipMemcpy(hostAgRecv.data(), agRecv, hostAgRecv.size() * sizeof(float),
                                        hipMemcpyDeviceToHost));
    const float rsExpected = static_cast<float>(nRanks * (nRanks + 1) / 2);
    size_t      rsBad      = 0;
    for(float v : hostRsRecv)
    {
        rsBad += (v != rsExpected) ? 1 : 0;
    }
    size_t agBad = 0;
    for(size_t i = 0; i < hostAgRecv.size(); ++i)
    {
        agBad += (hostAgRecv[i] != static_cast<float>(i / kAgSendCount + 1)) ? 1 : 0;
    }
    EXPECT_EQ(0u, rsBad) << "Rank " << rank << ": wrong ReduceScatter elements";
    EXPECT_EQ(0u, agBad) << "Rank " << rank << ": wrong AllGather elements";

    EXPECT_EQ(static_cast<uint64_t>(2 * kIters), opCountAfter - opCountBefore)
        << "Rank " << rank << ": comm->opCount did not advance once per DDA collective";
    EXPECT_EQ(static_cast<size_t>(kIters), countLogNeedle(kRsNeedle))
        << "Rank " << rank << ": ReduceScatter opCount lines in the COLL log";
    EXPECT_EQ(static_cast<size_t>(kIters), countLogNeedle(kAgNeedle))
        << "Rank " << rank << ": AllGather opCount lines in the COLL log";
}

#endif // MPI_TESTS_ENABLED
