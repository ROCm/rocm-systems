/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file SymmetricAllGatherTdmMPITests.cpp
 * @brief MPI end-to-end tests for the gfx1250 symmetric AllGather TDM window.
 *
 * Each case checks the reported algorithm, the kernel rank 0 logged and every rank's data
 * around the window edges, with and without graph capture.
 *
 * Run (example):
 *   mpirun -np 8 ./rccl-UnitTestsMPI --gtest_filter='SymWin_AllGatherTdm.*'
 */

#ifdef MPI_TESTS_ENABLED

#include "DdaFabricMPITestFixture.hpp"
#include "DeviceBufferHelpers.hpp"
#include "MPIHelpers.hpp"
#include "MPITestBase.hpp"
#include "ResourceGuards.hpp"
#include "SymmetricBufferHelpers.hpp"
#include "SymmetricKernelLog.hpp"
#include "TestChecks.hpp"
#include "comm.h"
#include "device/tdm/tdmCopy.h" // TDM_TOOLCHAIN_AVAILABLE
#include "rccl_common.h"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <mpi.h>

#include <chrono>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace MPITestConstants;
using namespace RCCLTestGuards;
using namespace RCCLTestHelpers;
using namespace RcclUnitTesting;

namespace
{
constexpr size_t kMiB          = 1024 * 1024;
constexpr size_t kWindowMin    = 64 * kMiB;  // symTdmMinR2[AllGather], gfx1250 SPX
constexpr size_t kEagerMax     = 128 * kMiB; // symTdmMaxR2[AllGather]
constexpr size_t kCrossTrayMax = 512 * kMiB; // symTdmCrossTrayMaxR2[AllGather]
constexpr int    kIterations   = 2;
constexpr auto   kDrainTimeout = std::chrono::seconds(120);

enum class Expect
{
    kTdm,               // the TDM symmetric kernel
    kNotSym,            // any non-symmetric path
    kTdmWithinTrayOnly, // TDM within one tray, a non-symmetric path across trays
};

// Per-rank bytes, a multiple of 16, whose total over all ranks is at least / below `total`.
size_t perRankAtLeast(size_t total)
{
    const size_t n = static_cast<size_t>(MPIEnvironment::world_size);
    return ((total + n - 1) / n + 15) & ~size_t{15};
}
size_t perRankBelow(size_t total)
{
    return ((total / static_cast<size_t>(MPIEnvironment::world_size)) & ~size_t{15}) - 16;
}

// True when the ranks sit on more than one host, i.e. the comm spans trays.
bool ranksSpanHosts()
{
    char name[MPI_MAX_PROCESSOR_NAME] = {};
    int  len                          = 0;
    MPI_Get_processor_name(name, &len);
    std::vector<char> all(static_cast<size_t>(MPIEnvironment::world_size) * MPI_MAX_PROCESSOR_NAME);
    MPI_Allgather(name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, all.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                  MPI_COMM_WORLD);
    std::set<std::string> hosts;
    for(int r = 0; r < MPIEnvironment::world_size; ++r)
        hosts.insert(std::string(&all[static_cast<size_t>(r) * MPI_MAX_PROCESSOR_NAME]));
    return hosts.size() > 1;
}
} // namespace

// DdaFabricMPITest provides the TUNING log capture and the coordinated skip-or-fail.
class SymWin_AllGatherTdm : public DdaFabricMPITest
{
protected:
    // Gathers perRankBytes per rank and checks the path and the data. Must end the test body.
    void runAllGather(size_t perRankBytes, bool inPlace, bool graph, Expect expect)
    {
        if(!validateTestPrerequisites(kMinProcessesForMPI))
        {
            GTEST_SKIP() << "Need at least 2 MPI ranks";
        }
        DDA_FABRIC_SKIP_OR_FAIL(archSkipReason("the symmetric AllGather TDM window is gfx1250 only"));
        DDA_FABRIC_SKIP_OR_FAIL(mpiCoordinatedSkipReason(!TDM_TOOLCHAIN_AVAILABLE, "built without the TDM toolchain"));
        if(expect == Expect::kTdmWithinTrayOnly)
            expect = ranksSpanHosts() ? Expect::kNotSym : Expect::kTdm;

        ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
        ncclComm_t  comm   = getActiveCommunicator();
        hipStream_t stream = getActiveStream();
        DDA_FABRIC_SKIP_OR_FAIL(mpiCoordinatedSkipReason(!comm->symmetricSupport, "no symmetric memory support"));
        DDA_FABRIC_SKIP_OR_FAIL(mpiCoordinatedSkipReason(
            !comm->MNNVL || comm->nvlDomainSize != comm->nRanks || comm->p2pCrossClique,
            "the comm is not one UALink domain"));
        int rank = -1, nRanks = 0;
        ASSERT_MPI_EQ(ncclSuccess, ncclCommUserRank(comm, &rank));
        ASSERT_MPI_EQ(ncclSuccess, ncclCommCount(comm, &nRanks));

        const size_t count      = perRankBytes / sizeof(float);
        const size_t totalCount = count * static_cast<size_t>(nRanks);
        SCOPED_TRACE("per-rank bytes " + std::to_string(perRankBytes) + (inPlace ? ", in place" : "")
                     + (graph ? ", graph" : ""));

        SymBuf recv, send;
        ASSERT_MPI_EQ(ncclSuccess, ncclSymBufAlloc(comm, totalCount * sizeof(float), recv));
        float* sendBuf = static_cast<float*>(recv.ptr) + static_cast<size_t>(rank) * count;
        if(!inPlace)
        {
            ASSERT_MPI_EQ(ncclSuccess, ncclSymBufAlloc(comm, perRankBytes, send));
            sendBuf = static_cast<float*>(send.ptr);
        }

        int algo = -1, protocol = 0, channels = 0;
        ASSERT_MPI_EQ(ncclSuccess, rcclGetCollImplInfo(comm, ncclFuncAllGather, count, ncclFloat32, ncclSum, sendBuf,
                                                       recv.ptr, graph ? 1 : 0, &algo, &protocol, &channels));
        const int  kSym    = static_cast<int>(rcclAddonAlgos_t::RCCL_SYMMETRIC);
        const int  kSymTdm = static_cast<int>(rcclAddonAlgos_t::RCCL_SYMMETRIC_TDM);
        const bool pathOk  = expect == Expect::kTdm ? algo == kSymTdm : (algo != kSym && algo != kSymTdm);
        if(!pathOk)
        {
            ADD_FAILURE() << "Rank " << rank << ": reported algo " << algo;
        }
        ASSERT_MPI_TRUE(pathOk);

        // Exact in float: every value stays far below 2^24.
        auto value = [](int r, size_t i, int iter) {
            return static_cast<float>((r + 1) * 4096 + static_cast<int>(i % 1021) + iter * 37);
        };
        for(int iter = 0; iter < kIterations; ++iter)
        {
            ASSERT_MPI_EQ(hipSuccess, hipMemset(recv.ptr, 0xff, totalCount * sizeof(float)));
            ASSERT_MPI_EQ(hipSuccess, initializeBufferWithPattern<float>(
                sendBuf, count, [&](size_t i) { return value(rank, i, iter); }));
            ncclResult_t launchRes = ncclSuccess;
            if(graph)
            {
                // End the capture before any assertion.
                ASSERT_MPI_EQ(hipSuccess, hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
                launchRes = ncclAllGather(sendBuf, recv.ptr, count, ncclFloat32, comm, stream);
                hipGraph_t       g      = nullptr;
                const hipError_t endRes = hipStreamEndCapture(stream, &g);
                auto             gGuard = makeScopeGuard([&]() { (void)hipGraphDestroy(g); });
                ASSERT_MPI_EQ(ncclSuccess, launchRes);
                ASSERT_MPI_EQ(hipSuccess, endRes);
                hipGraphExec_t exec = nullptr;
                ASSERT_MPI_EQ(hipSuccess, hipGraphInstantiate(&exec, g, nullptr, nullptr, 0));
                auto execGuard = makeScopeGuard([&]() { (void)hipGraphExecDestroy(exec); });
                launchRes      = hipGraphLaunch(exec, stream) == hipSuccess ? ncclSuccess : ncclUnhandledCudaError;
                DdaFabricMPI::abortJobUnlessAllRanksDrained(DdaFabricMPI::waitStreamBounded(stream, kDrainTimeout),
                                                            "symmetric AllGather graph");
            }
            else
            {
                launchRes = ncclAllGather(sendBuf, recv.ptr, count, ncclFloat32, comm, stream);
                DdaFabricMPI::abortJobUnlessAllRanksDrained(DdaFabricMPI::waitStreamBounded(stream, kDrainTimeout),
                                                            "symmetric AllGather");
            }
            ASSERT_MPI_EQ(ncclSuccess, launchRes);
            ASSERT_MPI_TRUE(verifyBufferData<float>(recv.ptr, totalCount, [&](size_t i) {
                return value(static_cast<int>(i / count), i % count, iter);
            }));
        }

        // Only rank 0 logs the symmetric kernel it launched; any other path logs none.
        if(rank == 0)
        {
            std::istringstream log(logCtx_->readNcclDebugLog() + logCtx_->readPerRankStderrLog());
            const std::string  kernel = LastSymmetricKernel(log, "AllGather");
            if(expect == Expect::kTdm)
                EXPECT_TRUE(IsTdmSymmetricKernel(kernel)) << "logged kernel '" << kernel << "'";
            else
                EXPECT_TRUE(kernel.empty()) << "logged symmetric kernel " << kernel;
        }
    }
};

// 2-rank comms take TDM only from about 76 MiB, so the floor is checked from 4 ranks.
TEST_F(SymWin_AllGatherTdm, WindowFloor)
{
    runAllGather(perRankAtLeast(kWindowMin), false, false,
                 MPIEnvironment::world_size >= 4 ? Expect::kTdm : Expect::kNotSym);
}

TEST_F(SymWin_AllGatherTdm, JustBelowWindow)
{
    runAllGather(perRankBelow(kWindowMin), false, false, Expect::kNotSym);
}

TEST_F(SymWin_AllGatherTdm, InPlace)
{
    runAllGather(perRankAtLeast(96 * kMiB), true, false, Expect::kTdm);
}

// Eagerly, CE-registered takes over past 128 MiB.
TEST_F(SymWin_AllGatherTdm, PastEagerCeiling)
{
    runAllGather(perRankAtLeast(kEagerMax) + 16, false, false, Expect::kNotSym);
}

TEST_F(SymWin_AllGatherTdm, GraphCrossTrayCeiling)
{
    runAllGather(perRankBelow(kCrossTrayMax) + 16, false, true, Expect::kTdm);
}

// Under capture the window passes 512 MiB in one tray and closes there across trays.
TEST_F(SymWin_AllGatherTdm, GraphPastCrossTrayCeiling)
{
    runAllGather(perRankAtLeast(kCrossTrayMax) + 16, false, true, Expect::kTdmWithinTrayOnly);
}

#endif // MPI_TESTS_ENABLED
