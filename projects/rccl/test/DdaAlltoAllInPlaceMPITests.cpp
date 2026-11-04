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
 */

#ifdef MPI_TESTS_ENABLED

#include "DdaFabricMPITestFixture.hpp"
#include "DeviceBufferHelpers.hpp"
#include "ResourceGuards.hpp"
#include "rccl_common.h"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
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

    void SetUp() override
    {
        DdaFabricMPITest::SetUp();
        if(!validateTestPrerequisites(kMinProcessesForMPI))
            GTEST_SKIP() << "Need at least 2 MPI ranks";
        DDA_FABRIC_SKIP_OR_FAIL(archSkipReason("DDA fabric requires gfx1250"));
        ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
        ASSERT_MPI_EQ(ncclSuccess, ncclCommUserRank(getActiveCommunicator(), &rank_));
        ASSERT_MPI_EQ(ncclSuccess, ncclCommCount(getActiveCommunicator(), &nRanks_));
        DDA_FABRIC_SKIP_OR_FAIL(fabricInitSkipReason());
    }

    int selectedAlgo(const void* send, void* recv, size_t count, ncclDataType_t datatype)
    {
        int algo = -1, protocol = -1, maxChannels = -1;
        EXPECT_EQ(ncclSuccess, rcclGetCollImplInfo(getActiveCommunicator(), ncclFuncAlltoAll, count, datatype,
                                                   ncclSum, send, recv, /*graphCapturing=*/0, &algo, &protocol,
                                                   &maxChannels));
        return algo;
    }

    void runCase(uint8_t* sendBase, uint8_t* recvBase, size_t perPeerBytes, const OffsetCase& c)
    {
        const size_t totalBytes = perPeerBytes * static_cast<size_t>(nRanks_);
        const size_t count      = perPeerBytes / static_cast<size_t>(ncclTypeSize(c.datatype));
        void* const  send       = sendBase + c.sendOffset;
        void* const  recv       = c.inPlace ? send : recvBase + c.recvOffset;

        const int alignedAlgo = selectedAlgo(sendBase, c.inPlace ? sendBase : recvBase, count, c.datatype);
        if(!isDdaFabric(alignedAlgo))
            return;
        EXPECT_EQ(selectedAlgo(send, recv, count, c.datatype), alignedAlgo) << "Rank " << rank_;

        const int nRanks = nRanks_;
        ASSERT_MPI_EQ(hipSuccess,
                      initializeBufferWithPattern<uint8_t>(send, totalBytes, [this, perPeerBytes](size_t i) {
                          return chunkPattern(rank_, static_cast<int>(i / perPeerBytes), i % perPeerBytes);
                      }));
        ASSERT_MPI_EQ(ncclSuccess, ncclAlltoAll(send, recv, count, c.datatype, getActiveCommunicator(),
                                                getActiveStream()));
        ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));
        EXPECT_TRUE(verifyBufferData<uint8_t>(recv, totalBytes, [this, nRanks, perPeerBytes](size_t i) {
            return chunkPattern(static_cast<int>(i / perPeerBytes), rank_, i % perPeerBytes);
        })) << "Rank " << rank_ << " of " << nRanks << ": " << (c.inPlace ? "in-place" : "out-of-place")
            << " AlltoAll data mismatch";
    }
};

TEST_F(DdaAlltoAllInPlaceMPI, OffsetsStayOnDda)
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
            SCOPED_TRACE("perPeerBytes=" + std::to_string(perPeer) + (c.inPlace ? " in place" : " out of place") +
                         " sendOffset=" + std::to_string(c.sendOffset) + " recvOffset=" + std::to_string(c.recvOffset));
            runCase(static_cast<uint8_t*>(base), static_cast<uint8_t*>(recvBase), perPeer, c);
            if(HasFatalFailure())
                return;
        }
    }
}

#endif // MPI_TESTS_ENABLED
