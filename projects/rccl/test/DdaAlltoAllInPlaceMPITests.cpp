/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file DdaAlltoAllInPlaceMPITests.cpp
 * @brief MPI end-to-end tests for gfx1250 AlltoAll on the DDA fabric tiers at buffer offsets
 *
 * Ranks choose a tier from their own buffers, so a tier that refused some alignments would let ranks
 * with different offsets run different kernels and hang; p2p also returns wrong data in place on
 * gfx1250. Every offset, down to 1 byte, in and out of place, must take the tier a 16-byte aligned
 * call takes, and every received byte must be exact, at sizes where the aligned call takes a DDA
 * tier under the active arch table.
 *
 * DdaAlltoAllPeerColumnsMPI repeats the cases with RCCL_DDA_FABRIC_MAXBLOCKS=2, so each LL128 block
 * column serves several peers.
 */

#ifdef MPI_TESTS_ENABLED

#include "DdaFabricMPITestFixture.hpp"
#include "DeviceBufferHelpers.hpp"
#include "ResourceGuards.hpp"
#include "rccl_common.h"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>

using namespace MPITestConstants;
using namespace RCCLTestGuards;
using namespace RCCLTestHelpers;

namespace
{
// A prime number of 16-byte lines per peer runs the kernels' partial-slice tails.
constexpr size_t kPrimeLineBytes = 251 * 16;
constexpr size_t kPerPeerBytes[] = {kPrimeLineBytes, 64 * 1024, 512 * 1024};
constexpr size_t kSlack          = 64;
constexpr auto   kDrainTimeout   = std::chrono::seconds(120);

struct OffsetCase
{
    ncclDataType_t datatype;
    bool           inPlace;
    size_t         sendOffset;
    size_t         recvOffset; // out of place only
};
constexpr OffsetCase kCases[] = {{ncclFloat32, true, 0, 0},       {ncclFloat32, true, 4, 0},
                                 {ncclFloat32, true, 16, 0},      {ncclBfloat16, true, 2, 0},
                                 {ncclFloat8e4m3, true, 1, 0},    {ncclBfloat16, false, 2, 0},
                                 {ncclFloat8e4m3, false, 0, 1},   {ncclFloat8e4m3, false, 1, 2}};

// Byte i of the chunk rank s sends to rank d; differs per pair and index so a misplaced chunk fails.
uint8_t chunkPattern(int src, int dst, size_t i)
{
    return static_cast<uint8_t>(src * 31 + dst * 7 + i);
}

bool isDdaFabric(int algo)
{
    return algo == RCCL_DDA_FABRIC_LL || algo == RCCL_DDA_FABRIC_LL128;
}
} // namespace

class DdaAlltoAllInPlaceMPI : public DdaFabricMPITest
{
protected:
    int rank_{};
    int nRanks_{};
    int ddaCases_{};   // cases whose aligned call took a DDA tier
    int ll128Cases_{}; // of those, on LL128
    bool ll128On_{};
    std::unique_ptr<MPIHelpers::MpiEnvGuard> maxBlocksGuard_;

    // Fabric init reads RCCL_DDA_FABRIC_MAXBLOCKS on every comm, so a fixture can set it per test.
    virtual const char* maxBlocksOverride() const
    {
        return nullptr;
    }

    void SetUp() override
    {
        DdaFabricMPITest::SetUp();
        if(!validateTestPrerequisites(kMinProcessesForMPI))
            GTEST_SKIP() << "Need at least 2 MPI ranks";
        DDA_FABRIC_SKIP_OR_FAIL(archSkipReason("DDA fabric requires gfx1250"));
        if(const char* value = maxBlocksOverride())
            maxBlocksGuard_ = std::make_unique<MPIHelpers::MpiEnvGuard>("RCCL_DDA_FABRIC_MAXBLOCKS", value);
        ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
        ASSERT_MPI_EQ(ncclSuccess, ncclCommUserRank(getActiveCommunicator(), &rank_));
        ASSERT_MPI_EQ(ncclSuccess, ncclCommCount(getActiveCommunicator(), &nRanks_));
        DDA_FABRIC_SKIP_OR_FAIL(fabricInitSkipReason());

        const ncclComm_t comm = getActiveCommunicator();
        // These switch DDA AlltoAll off for the process, so there is no DDA tier to assert.
        const bool ddaOff = rcclNcclAlgoEnvIsSet() || rcclParamDdaEnable() == 0
                            || comm->config.launchOrderImplicit == 1
                            || (rcclDdaLLThreshold(comm, ncclFuncAlltoAll) == 0
                                && rcclDdaLL128Threshold(comm, ncclFuncAlltoAll) == 0);
        const std::string ddaOffSkip = mpiCoordinatedSkipReason(
            ddaOff, "DDA AlltoAll off (NCCL_ALGO, RCCL_DDA_ENABLE, thresholds, implicit launch order)");
        if(!ddaOffSkip.empty())
            GTEST_SKIP() << ddaOffSkip;
        ll128On_ = rcclDdaLL128Threshold(comm, ncclFuncAlltoAll) > 0;
    }

    int selectedAlgo(const void* send, void* recv, size_t count, ncclDataType_t datatype, int* blocks = nullptr)
    {
        int algo = -1, protocol = -1, maxChannels = -1;
        EXPECT_EQ(ncclSuccess, rcclGetCollImplInfo(getActiveCommunicator(), ncclFuncAlltoAll, count, datatype,
                                                   ncclSum, send, recv, /*graphCapturing=*/0, &algo, &protocol,
                                                   &maxChannels));
        if(blocks != nullptr)
            *blocks = maxChannels;
        return algo;
    }

    void runCase(uint8_t* sendBase, uint8_t* recvBase, size_t perPeerBytes, const OffsetCase& c)
    {
        const size_t totalBytes = perPeerBytes * static_cast<size_t>(nRanks_);
        const size_t count      = perPeerBytes / static_cast<size_t>(ncclTypeSize(c.datatype));
        void* const  send       = sendBase + c.sendOffset;
        void* const  recv       = c.inPlace ? send : recvBase + c.recvOffset;

        int       blocks      = -1;
        const int alignedAlgo = selectedAlgo(sendBase, c.inPlace ? sendBase : recvBase, count, c.datatype, &blocks);
        if(!isDdaFabric(alignedAlgo))
            return;
        ++ddaCases_;
        ll128Cases_ += alignedAlgo == RCCL_DDA_FABRIC_LL128;
        EXPECT_EQ(selectedAlgo(send, recv, count, c.datatype), alignedAlgo) << "Rank " << rank_;
        if(alignedAlgo == RCCL_DDA_FABRIC_LL128)
            EXPECT_LE(blocks, getActiveCommunicator()->ddaFabricMaxBlocks)
                << "Rank " << rank_ << ": LL128 grid exceeds the fabric block budget";

        const int nRanks = nRanks_;
        ASSERT_MPI_EQ(hipSuccess,
                      initializeBufferWithPattern<uint8_t>(send, totalBytes, [this, perPeerBytes](size_t i) {
                          return chunkPattern(rank_, static_cast<int>(i / perPeerBytes), i % perPeerBytes);
                      }));
        ASSERT_MPI_EQ(ncclSuccess, ncclAlltoAll(send, recv, count, c.datatype, getActiveCommunicator(),
                                                getActiveStream()));
        DdaFabricMPI::abortJobUnlessAllRanksDrained(DdaFabricMPI::waitStreamBounded(getActiveStream(), kDrainTimeout),
                                                    "DDA AlltoAll");
        EXPECT_TRUE(verifyBufferData<uint8_t>(recv, totalBytes, [this, nRanks, perPeerBytes](size_t i) {
            return chunkPattern(static_cast<int>(i / perPeerBytes), rank_, i % perPeerBytes);
        })) << "Rank " << rank_ << " of " << nRanks << ": " << (c.inPlace ? "in-place" : "out-of-place")
            << " AlltoAll data mismatch";
    }

    void runAllCases()
    {
        size_t maxTotal = 0;
        for(size_t perPeer : kPerPeerBytes)
            maxTotal = std::max(maxTotal, perPeer * static_cast<size_t>(nRanks_));
        void* base = nullptr;
        ASSERT_MPI_EQ(hipSuccess, hipMalloc(&base, maxTotal + kSlack));
        DeviceBufferAutoGuard guard(base);
        void* recvBase = nullptr;
        ASSERT_MPI_EQ(hipSuccess, hipMalloc(&recvBase, maxTotal + kSlack));
        DeviceBufferAutoGuard recvGuard(recvBase);

        for(size_t perPeer : kPerPeerBytes)
        {
            for(const OffsetCase& c : kCases)
            {
                SCOPED_TRACE("perPeerBytes=" + std::to_string(perPeer) + (c.inPlace ? " in place" : " out of place")
                             + " sendOffset=" + std::to_string(c.sendOffset)
                             + " recvOffset=" + std::to_string(c.recvOffset));
                runCase(static_cast<uint8_t*>(base), static_cast<uint8_t*>(recvBase), perPeer, c);
                if(HasFatalFailure() || IsSkipped())
                    return;
            }
        }
        // Cases off DDA return before any check, so a selection change that sent them all to p2p
        // would pass with nothing tested. From 8 ranks one of the sizes sits in the LL128 band of
        // the gfx1250 tables, so LL128 has to be among them.
        EXPECT_GT(ddaCases_, 0) << "Rank " << rank_ << ": no case took a DDA tier";
        if(nRanks_ >= 8 && ll128On_)
            EXPECT_GT(ll128Cases_, 0) << "Rank " << rank_ << " of " << nRanks_ << ": no case took DDA LL128";
    }
};

TEST_F(DdaAlltoAllInPlaceMPI, OffsetsStayOnDda)
{
    runAllCases();
}

class DdaAlltoAllPeerColumnsMPI : public DdaAlltoAllInPlaceMPI
{
protected:
    static constexpr char kMaxBlocks[] = "2";

    const char* maxBlocksOverride() const override
    {
        return kMaxBlocks;
    }
};

// Two blocks for seven or more peers: each LL128 column sends to every peer it serves before it
// polls any, and a column that polled first would wait on a peer whose sender has not run.
TEST_F(DdaAlltoAllPeerColumnsMPI, ColumnsServeSeveralPeers)
{
    const std::string skip
        = mpiCoordinatedSkipReason(nRanks_ < 8 || !ll128On_, "Needs 8 or more ranks with DDA AlltoAll LL128 on");
    if(!skip.empty())
        GTEST_SKIP() << skip;
    ASSERT_MPI_EQ(std::atoi(kMaxBlocks), getActiveCommunicator()->ddaFabricMaxBlocks);
    runAllCases();
}

#endif // MPI_TESTS_ENABLED
