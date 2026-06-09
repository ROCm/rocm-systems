/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file RdmaTransferTests.cpp
 * @brief Verbs acceptance tests for the RDMA data path over GPU memory
 *        registered via DMABUF (Write and Read).
 */

#include "VerbsMPITestBase.hpp"

#ifdef MPI_TESTS_ENABLED

using namespace VerbsAcceptance;

// ===========================================================================
// RdmaWrite_DMABUF -- RC write into peer GPU memory registered via DMABUF.
// Runs once per available dmabuf version. Lower half = target.
// ===========================================================================
TEST_F(VerbsAcceptanceMPITest, RdmaWrite_DMABUF)
{
    if(!requireEvenPairs())
        GTEST_SKIP() << "Requires an even number of MPI ranks (>= 2)";
    resolveOptionalSymbols();
    if(!allRanksAgree(openIbDevice()))
        GTEST_SKIP() << "No active IB device on all ranks";
    if(!allRanksAgree(g_reg_dmabuf != nullptr && (g_dmabuf_v1 || g_dmabuf_v2)))
        GTEST_SKIP() << "DMABUF registration not available";

    const int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
    bool      ranAny = false;

    for(RegMode reg : {REG_DMABUF_V1, REG_DMABUF_V2_PCIE})
    {
        bool haveSym = (reg == REG_DMABUF_V1) ? (g_dmabuf_v1 != nullptr) : (g_dmabuf_v2 != nullptr);
        if(!allRanksAgree(haveSym))
            continue;

        ASSERT_MPI_TRUE(createQp(IBV_QPT_RC));
        QpInfo remote = exchangeQpInfo();
        ASSERT_MPI_TRUE(rcToRtrRts(remote));

        void* buf = allocDeviceBuf(kMsgSize);
        ASSERT_MPI_TRUE(buf != nullptr);
        SCOPE_EXIT(freeDeviceBuf(buf));
        RegResult rr = regBuf(buf, kMsgSize, reg, access);
        if(!allRanksAgree(rr.mr != nullptr))
        {
            deregBuf(rr);
            destroyQp();
            continue;
        }

        const uint8_t seed = static_cast<uint8_t>(0xA0 + static_cast<int>(reg));
        bool          pass = false;
        if(isLowerHalf_)
        {
            // Target: clear buffer, advertise it, wait for the data to land.
            (void)hipMemset(buf, 0, kMsgSize);
            (void)hipDeviceSynchronize();
            RemoteBuf rb{reinterpret_cast<uint64_t>(buf), rr.mr->rkey, 0};
            MPI_Send(&rb, sizeof(rb), MPI_BYTE, peerRank_, kTagRemoteBuf, MPI_COMM_WORLD);
            uint8_t go = 0;
            MPI_Recv(&go, 1, MPI_BYTE, peerRank_, kTagFlag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            pass        = verifyPatternGpu(buf, kMsgSize, seed);
            uint8_t res = pass ? 1 : 0;
            MPI_Send(&res, 1, MPI_BYTE, peerRank_, kTagResult, MPI_COMM_WORLD);
        }
        else
        {
            RemoteBuf rb{};
            MPI_Recv(&rb, sizeof(rb), MPI_BYTE, peerRank_, kTagRemoteBuf, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            fillPatternGpu(buf, kMsgSize, seed);
            bool wrote =
                postRdmaAndWait(buf, rr.mr, rb.addr, rb.rkey, kMsgSize, IBV_WR_RDMA_WRITE, 1);
            uint8_t go = 1;
            MPI_Send(&go, 1, MPI_BYTE, peerRank_, kTagFlag, MPI_COMM_WORLD);
            uint8_t res = 0;
            MPI_Recv(&res, 1, MPI_BYTE, peerRank_, kTagResult, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            pass = wrote && (res == 1);
        }
        deregBuf(rr);
        destroyQp();
        ranAny = true;
        ASSERT_MPI_TRUE(pass);
    }

    if(!allRanksAgree(ranAny))
        GTEST_SKIP() << "No DMABUF version exercised";
}

// ===========================================================================
// RdmaRead_DMABUF -- RC read from peer GPU memory via DMABUF. A separate driver
// path from Write. Runs once per available dmabuf version.
// ===========================================================================
TEST_F(VerbsAcceptanceMPITest, RdmaRead_DMABUF)
{
    if(!requireEvenPairs())
        GTEST_SKIP() << "Requires an even number of MPI ranks (>= 2)";
    resolveOptionalSymbols();
    if(!allRanksAgree(openIbDevice()))
        GTEST_SKIP() << "No active IB device on all ranks";
    if(!allRanksAgree(g_reg_dmabuf != nullptr && (g_dmabuf_v1 || g_dmabuf_v2)))
        GTEST_SKIP() << "DMABUF registration not available";

    const int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
    bool      ranAny = false;

    for(RegMode reg : {REG_DMABUF_V1, REG_DMABUF_V2_PCIE})
    {
        bool haveSym = (reg == REG_DMABUF_V1) ? (g_dmabuf_v1 != nullptr) : (g_dmabuf_v2 != nullptr);
        if(!allRanksAgree(haveSym))
            continue;

        ASSERT_MPI_TRUE(createQp(IBV_QPT_RC));
        QpInfo remote = exchangeQpInfo();
        ASSERT_MPI_TRUE(rcToRtrRts(remote));

        void* buf = allocDeviceBuf(kMsgSize);
        ASSERT_MPI_TRUE(buf != nullptr);
        SCOPE_EXIT(freeDeviceBuf(buf));
        RegResult rr = regBuf(buf, kMsgSize, reg, access);
        if(!allRanksAgree(rr.mr != nullptr))
        {
            deregBuf(rr);
            destroyQp();
            continue;
        }

        const uint8_t seed = static_cast<uint8_t>(0xB0 + static_cast<int>(reg));
        bool          pass = false;
        if(isLowerHalf_)
        {
            // Target holds the source data; initiator reads it.
            fillPatternGpu(buf, kMsgSize, seed);
            RemoteBuf rb{reinterpret_cast<uint64_t>(buf), rr.mr->rkey, 0};
            MPI_Send(&rb, sizeof(rb), MPI_BYTE, peerRank_, kTagRemoteBuf, MPI_COMM_WORLD);
            uint8_t res = 0;
            MPI_Recv(&res, 1, MPI_BYTE, peerRank_, kTagResult, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            pass = (res == 1);
        }
        else
        {
            RemoteBuf rb{};
            MPI_Recv(&rb, sizeof(rb), MPI_BYTE, peerRank_, kTagRemoteBuf, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            (void)hipMemset(buf, 0, kMsgSize);
            (void)hipDeviceSynchronize();
            bool ok =
                postRdmaAndWait(buf, rr.mr, rb.addr, rb.rkey, kMsgSize, IBV_WR_RDMA_READ, 2);
            pass        = ok && verifyPatternGpu(buf, kMsgSize, seed);
            uint8_t res = pass ? 1 : 0;
            MPI_Send(&res, 1, MPI_BYTE, peerRank_, kTagResult, MPI_COMM_WORLD);
        }
        deregBuf(rr);
        destroyQp();
        ranAny = true;
        ASSERT_MPI_TRUE(pass);
    }

    if(!allRanksAgree(ranAny))
        GTEST_SKIP() << "No DMABUF version exercised";
}

#endif // MPI_TESTS_ENABLED
