/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifdef MPI_TESTS_ENABLED

#ifdef RCCL_HAS_GIN_IB_PROXY

#include "GinMPITestBase.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

namespace RCCLGinTests
{

namespace
{

constexpr size_t kSignalSize = 64;
constexpr size_t kBlockSize  = 256;
constexpr int    kInflightN  = 16;

} // namespace

TEST_P(GinMPITest, IPutBasic)
{
    if(!SetUpFixture(/*minProcs=*/2, /*maxProcs=*/2))
    {
        return;
    }

    const size_t kSize = MessageSize();

    void* sendBuf = AllocBuf(kSize);
    void* recvBuf = AllocBuf(kSize);
    ASSERT_NE(sendBuf, nullptr);
    ASSERT_NE(recvBuf, nullptr);

    if(worldRank_ == 0)
    {
        FillBuf(sendBuf, kSize, /*seed=*/0xA0);
    }

    void *sendMh = nullptr, *sendGh = nullptr;
    void *recvMh = nullptr, *recvGh = nullptr;
    ASSERT_EQ(ncclSuccess, RegMr(sendBuf, kSize, &sendMh, &sendGh));
    ASSERT_EQ(ncclSuccess, RegMr(recvBuf, kSize, &recvMh, &recvGh));

    Barrier();

    if(worldRank_ == 0)
    {
        void* req = nullptr;
        ASSERT_EQ(ncclSuccess,
                  gin_->iput(ginCtx_, /*context=*/0,
                             /*srcOff=*/0, sendMh, kSize,
                             /*dstOff=*/0, recvMh,
                             /*peerRank=*/1, &req));
        ASSERT_TRUE(PollUntilDone(req));
    }
    Barrier();

    if(worldRank_ == 1)
    {
        EXPECT_TRUE(VerifyBuf(recvBuf, kSize, /*seed=*/0xA0));
    }
}

TEST_P(GinMPITest, IGetBasic)
{
    if(!SetUpFixture(2, 2))
    {
        return;
    }

    const size_t kSize = MessageSize();

    void* buf = AllocBuf(kSize);
    ASSERT_NE(buf, nullptr);
    if(worldRank_ == 1)
    {
        FillBuf(buf, kSize, /*seed=*/0xC3);
    }

    void *mh = nullptr, *gh = nullptr;
    ASSERT_EQ(ncclSuccess, RegMr(buf, kSize, &mh, &gh));

    Barrier();
    if(worldRank_ == 0)
    {
        void* req = nullptr;
        ASSERT_EQ(ncclSuccess,
                  gin_->iget(ginCtx_, /*context=*/0,
                             /*remoteOff=*/0, mh, kSize,
                             /*localOff=*/0,  mh,
                             /*peerRank=*/1, &req));
        ASSERT_TRUE(PollUntilDone(req));
        EXPECT_TRUE(VerifyBuf(buf, kSize, /*seed=*/0xC3));
    }
    Barrier();
}

TEST_P(GinMPITest, IPutSignalInc)
{
    if(!SetUpFixture(2, 2))
    {
        return;
    }

    const size_t kSize = MessageSize();

    void* sendBuf = AllocBuf(kSize);
    void* recvBuf = AllocBuf(kSize);
    void* sigBuf  = AllocBuf(kSignalSize);
    ASSERT_NE(sendBuf, nullptr);
    ASSERT_NE(recvBuf, nullptr);
    ASSERT_NE(sigBuf,  nullptr);

    if(worldRank_ == 0)
    {
        FillBuf(sendBuf, kSize, /*seed=*/0x55);
    }

    void *sendMh, *sendGh, *recvMh, *recvGh, *sigMh, *sigGh;
    ASSERT_EQ(ncclSuccess, RegMr(sendBuf, kSize,       &sendMh, &sendGh));
    ASSERT_EQ(ncclSuccess, RegMr(recvBuf, kSize,       &recvMh, &recvGh));
    ASSERT_EQ(ncclSuccess, RegMr(sigBuf,  kSignalSize, &sigMh,  &sigGh));

    Barrier();
    if(worldRank_ == 0)
    {
        void* req = nullptr;
        ASSERT_EQ(ncclSuccess,
                  gin_->iputSignal(ginCtx_, /*context=*/0,
                                   /*srcOff=*/0, sendMh, kSize,
                                   /*dstOff=*/0, recvMh,
                                   /*peerRank=*/1,
                                   /*signalOff=*/0, sigMh,
                                   /*signalValue=*/0, // unused for INC
                                   NCCL_NET_SIGNAL_OP_INC,
                                   &req));
        ASSERT_TRUE(PollUntilDone(req));
    }
    Barrier();

    if(worldRank_ == 1)
    {
        EXPECT_TRUE(VerifyBuf(recvBuf, kSize, /*seed=*/0x55));
        EXPECT_EQ(ReadSignal(sigBuf), 1u);
    }
}

TEST_P(GinMPITest, IPutSignalAdd)
{
    if(!SetUpFixture(2, 2))
    {
        return;
    }

    constexpr uint64_t kAddValue = 42;
    const size_t       kSize     = MessageSize();

    void* sendBuf = AllocBuf(kSize);
    void* recvBuf = AllocBuf(kSize);
    void* sigBuf  = AllocBuf(kSignalSize);
    ASSERT_NE(sendBuf, nullptr);
    ASSERT_NE(recvBuf, nullptr);
    ASSERT_NE(sigBuf,  nullptr);

    if(worldRank_ == 0)
    {
        FillBuf(sendBuf, kSize, /*seed=*/0xE7);
    }

    void *sendMh, *sendGh, *recvMh, *recvGh, *sigMh, *sigGh;
    ASSERT_EQ(ncclSuccess, RegMr(sendBuf, kSize,       &sendMh, &sendGh));
    ASSERT_EQ(ncclSuccess, RegMr(recvBuf, kSize,       &recvMh, &recvGh));
    ASSERT_EQ(ncclSuccess, RegMr(sigBuf,  kSignalSize, &sigMh,  &sigGh));

    Barrier();
    if(worldRank_ == 0)
    {
        void* req = nullptr;
        ASSERT_EQ(ncclSuccess,
                  gin_->iputSignal(ginCtx_, 0,
                                   0, sendMh, kSize,
                                   0, recvMh, 1,
                                   /*signalOff=*/0, sigMh,
                                   kAddValue,
                                   NCCL_NET_SIGNAL_OP_ADD,
                                   &req));
        ASSERT_TRUE(PollUntilDone(req));
    }
    Barrier();

    if(worldRank_ == 1)
    {
        EXPECT_TRUE(VerifyBuf(recvBuf, kSize, /*seed=*/0xE7));
        EXPECT_EQ(ReadSignal(sigBuf), kAddValue);
    }
}

TEST_P(GinMPITest, IPutSignalAtOffset)
{
    if(!SetUpFixture(2, 2))
    {
        return;
    }

    const size_t      kSize       = MessageSize();
    const size_t      kSrcOff     = kSize / 2;
    const size_t      kDstOff     = kSize / 2;
    const size_t      kSigOff     = 64;
    const size_t      kBufSize    = kSize * 2;
    const size_t      kSigBufSize = kSignalSize * 2;
    constexpr uint8_t kSentinel   = 0xCC;

    void* sendBuf = AllocBuf(kBufSize);
    void* recvBuf = AllocBuf(kBufSize);
    void* sigBuf  = AllocBuf(kSigBufSize);
    ASSERT_NE(sendBuf, nullptr);
    ASSERT_NE(recvBuf, nullptr);
    ASSERT_NE(sigBuf,  nullptr);

    if(worldRank_ == 0)
    {
        FillBuf(static_cast<uint8_t*>(sendBuf) + kSrcOff, kSize, /*seed=*/0xA5);
    }
    if(worldRank_ == 1)
    {
        FillSentinel(recvBuf, kBufSize,    kSentinel);
        FillSentinel(sigBuf,  kSigBufSize, kSentinel);
    }

    void *sendMh, *sendGh, *recvMh, *recvGh, *sigMh, *sigGh;
    ASSERT_EQ(ncclSuccess, RegMr(sendBuf, kBufSize,    &sendMh, &sendGh));
    ASSERT_EQ(ncclSuccess, RegMr(recvBuf, kBufSize,    &recvMh, &recvGh));
    ASSERT_EQ(ncclSuccess, RegMr(sigBuf,  kSigBufSize, &sigMh,  &sigGh));

    Barrier();
    if(worldRank_ == 0)
    {
        void* req = nullptr;
        ASSERT_EQ(ncclSuccess,
                  gin_->iputSignal(ginCtx_, /*context=*/0,
                                   /*srcOff=*/kSrcOff, sendMh, kSize,
                                   /*dstOff=*/kDstOff, recvMh,
                                   /*peerRank=*/1,
                                   /*signalOff=*/kSigOff, sigMh,
                                   /*signalValue=*/0,
                                   NCCL_NET_SIGNAL_OP_INC,
                                   &req));
        ASSERT_TRUE(PollUntilDone(req));
    }
    Barrier();

    if(worldRank_ == 1)
    {
        // Data window: pattern at dstOff
        EXPECT_TRUE(VerifyBuf(static_cast<uint8_t*>(recvBuf) + kDstOff,
                              kSize, /*seed=*/0xA5))
            << "data did not land at dstOff=" << kDstOff;
        // Bytes BEFORE dstOff: untouched
        EXPECT_TRUE(AllSentinel(recvBuf, kDstOff, kSentinel))
            << "recvBuf[0.." << kDstOff << ") was thrashed "
            << "(dstOff likely treated as 0)";
        // Bytes AFTER dstOff + kSize: untouched
        EXPECT_TRUE(AllSentinel(static_cast<uint8_t*>(recvBuf) + kDstOff + kSize,
                                kBufSize - (kDstOff + kSize), kSentinel))
            << "recvBuf[" << (kDstOff + kSize) << ".." << kBufSize
            << ") was thrashed (write extended past size)";

        // Signal: incremented at signalOff
        EXPECT_EQ(ReadSignal(sigBuf, kSigOff), 1u)
            << "signal at signalOff=" << kSigOff << " not incremented";
        // Bytes BEFORE signalOff: untouched
        EXPECT_TRUE(AllSentinel(sigBuf, kSigOff, kSentinel))
            << "sigBuf[0.." << kSigOff << ") was thrashed "
            << "(signalOff likely treated as 0)";
        // Bytes AFTER signalOff + 8: untouched
        EXPECT_TRUE(AllSentinel(static_cast<uint8_t*>(sigBuf) + kSigOff + sizeof(uint64_t),
                                kSigBufSize - (kSigOff + sizeof(uint64_t)), kSentinel))
            << "sigBuf[" << (kSigOff + sizeof(uint64_t)) << ".." << kSigBufSize
            << ") was thrashed (atomic wrote past 8 bytes)";
    }
}

TEST_P(GinMPITest, IFlushAfterIGet)
{
    if(!SetUpFixture(2, 2))
    {
        return;
    }

    const size_t kSize = MessageSize();

    void* buf = AllocBuf(kSize);
    ASSERT_NE(buf, nullptr);
    if(worldRank_ == 1)
    {
        FillBuf(buf, kSize, /*seed=*/0x71);
    }

    void *mh = nullptr, *gh = nullptr;
    ASSERT_EQ(ncclSuccess, RegMr(buf, kSize, &mh, &gh));

    Barrier();
    if(worldRank_ == 0)
    {
        void* getReq = nullptr;
        ASSERT_EQ(ncclSuccess,
                  gin_->iget(ginCtx_, 0, 0, mh, kSize, 0, mh, 1, &getReq));
        ASSERT_TRUE(PollUntilDone(getReq));

        // iflush is the actual unit under test here: post a flush request
        // for the remote MR and verify it completes.
        void* flushReq = nullptr;
        ASSERT_EQ(ncclSuccess,
                  gin_->iflush(ginCtx_, /*context=*/0,
                               /*mhandle=*/mh, /*peerRank=*/1, &flushReq));
        EXPECT_TRUE(PollUntilDone(flushReq));

        EXPECT_TRUE(VerifyBuf(buf, kSize, /*seed=*/0x71));
    }
    Barrier();
}

TEST_P(GinMPITest, MultipleInflightIPuts)
{
    if(!SetUpFixture(2, 2))
    {
        return;
    }

    const size_t kBlock   = MessageSize();
    const size_t kBufSize = kInflightN * kBlock;
    const int    nCtx     = NumContexts();

    void* sendBuf = AllocBuf(kBufSize);
    void* recvBuf = AllocBuf(kBufSize);
    ASSERT_NE(sendBuf, nullptr);
    ASSERT_NE(recvBuf, nullptr);

    if(worldRank_ == 0)
    {
        for(int i = 0; i < kInflightN; ++i)
        {
            FillBuf(static_cast<uint8_t*>(sendBuf) + i * kBlock, kBlock, /*seed=*/i);
        }
    }

    void *sendMh, *sendGh, *recvMh, *recvGh;
    ASSERT_EQ(ncclSuccess, RegMr(sendBuf, kBufSize, &sendMh, &sendGh));
    ASSERT_EQ(ncclSuccess, RegMr(recvBuf, kBufSize, &recvMh, &recvGh));

    Barrier();
    if(worldRank_ == 0)
    {
        std::vector<void*> reqs(kInflightN, nullptr);
        for(int i = 0; i < kInflightN; ++i)
        {
            const int ctx = i % nCtx;
            ASSERT_EQ(ncclSuccess,
                      gin_->iput(ginCtx_, /*context=*/ctx,
                                 /*srcOff=*/i * kBlock, sendMh, kBlock,
                                 /*dstOff=*/i * kBlock, recvMh,
                                 /*peerRank=*/1, &reqs[i]));
        }
        for(int i = 0; i < kInflightN; ++i)
        {
            EXPECT_TRUE(PollUntilDone(reqs[i]))
                << "request " << i << " (ctx " << (i % nCtx) << ") did not complete";
        }
    }
    Barrier();

    if(worldRank_ == 1)
    {
        for(int i = 0; i < kInflightN; ++i)
        {
            EXPECT_TRUE(VerifyBuf(static_cast<uint8_t*>(recvBuf) + i * kBlock,
                                  kBlock, /*seed=*/i))
                << "block " << i << " (ctx " << (i % nCtx) << ") mismatched";
        }
    }
}

TEST_P(GinMPITest, MultipleInflightIGets)
{
    if(!SetUpFixture(2, 2))
    {
        return;
    }

    const size_t kBlock   = MessageSize();
    const size_t kBufSize = kInflightN * kBlock;
    const int    nCtx     = NumContexts();

    void* buf = AllocBuf(kBufSize);
    ASSERT_NE(buf, nullptr);
    if(worldRank_ == 1)
    {
        for(int i = 0; i < kInflightN; ++i)
        {
            FillBuf(static_cast<uint8_t*>(buf) + i * kBlock, kBlock, /*seed=*/i);
        }
    }

    void *mh = nullptr, *gh = nullptr;
    ASSERT_EQ(ncclSuccess, RegMr(buf, kBufSize, &mh, &gh));

    Barrier();
    if(worldRank_ == 0)
    {
        std::vector<void*> reqs(kInflightN, nullptr);
        for(int i = 0; i < kInflightN; ++i)
        {
            const int ctx = i % nCtx;
            ASSERT_EQ(ncclSuccess,
                      gin_->iget(ginCtx_, /*context=*/ctx,
                                 /*remoteOff=*/i * kBlock, mh, kBlock,
                                 /*localOff=*/ i * kBlock, mh,
                                 /*peerRank=*/1, &reqs[i]));
        }
        for(int i = 0; i < kInflightN; ++i)
        {
            EXPECT_TRUE(PollUntilDone(reqs[i]))
                << "request " << i << " (ctx " << (i % nCtx) << ") did not complete";
        }
        for(int i = 0; i < kInflightN; ++i)
        {
            EXPECT_TRUE(VerifyBuf(static_cast<uint8_t*>(buf) + i * kBlock,
                                  kBlock, /*seed=*/i))
                << "block " << i << " (ctx " << (i % nCtx) << ") mismatched";
        }
    }
    Barrier();
}

TEST_P(GinMPITest, MixedIPutIGetIPutSignal)
{
    if(!SetUpFixture(2, 2))
    {
        return;
    }

    const size_t kSize = MessageSize();
    const int    nCtx  = NumContexts();

    void* putSendBuf = AllocBuf(kSize);
    void* putRecvBuf = AllocBuf(kSize);
    void* getBuf     = AllocBuf(kSize);
    void* psSendBuf  = AllocBuf(kSize);
    void* psRecvBuf  = AllocBuf(kSize);
    void* sigBuf     = AllocBuf(kSignalSize);
    ASSERT_NE(putSendBuf, nullptr);
    ASSERT_NE(putRecvBuf, nullptr);
    ASSERT_NE(getBuf,     nullptr);
    ASSERT_NE(psSendBuf,  nullptr);
    ASSERT_NE(psRecvBuf,  nullptr);
    ASSERT_NE(sigBuf,     nullptr);

    if(worldRank_ == 0)
    {
        FillBuf(putSendBuf, kSize, /*seed=*/0x10);
        FillBuf(psSendBuf,  kSize, /*seed=*/0x30);
    }
    if(worldRank_ == 1)
    {
        FillBuf(getBuf, kSize, /*seed=*/0x20);
    }

    void *putSendMh, *putSendGh, *putRecvMh, *putRecvGh;
    void *getMh, *getGh;
    void *psSendMh, *psSendGh, *psRecvMh, *psRecvGh;
    void *sigMh, *sigGh;

    ASSERT_EQ(ncclSuccess, RegMr(putSendBuf, kSize,       &putSendMh, &putSendGh));
    ASSERT_EQ(ncclSuccess, RegMr(putRecvBuf, kSize,       &putRecvMh, &putRecvGh));
    ASSERT_EQ(ncclSuccess, RegMr(getBuf,     kSize,       &getMh,     &getGh));
    ASSERT_EQ(ncclSuccess, RegMr(psSendBuf,  kSize,       &psSendMh,  &psSendGh));
    ASSERT_EQ(ncclSuccess, RegMr(psRecvBuf,  kSize,       &psRecvMh,  &psRecvGh));
    ASSERT_EQ(ncclSuccess, RegMr(sigBuf,     kSignalSize, &sigMh,     &sigGh));

    Barrier();
    if(worldRank_ == 0)
    {
        void *putReq = nullptr, *getReq = nullptr, *psReq = nullptr;

        ASSERT_EQ(ncclSuccess,
                  gin_->iput(ginCtx_, /*context=*/0 % nCtx,
                             0, putSendMh, kSize, 0, putRecvMh, 1, &putReq));
        ASSERT_EQ(ncclSuccess,
                  gin_->iget(ginCtx_, /*context=*/1 % nCtx,
                             0, getMh, kSize, 0, getMh, 1, &getReq));
        ASSERT_EQ(ncclSuccess,
                  gin_->iputSignal(ginCtx_, /*context=*/2 % nCtx,
                                   0, psSendMh, kSize,
                                   0, psRecvMh, 1,
                                   0, sigMh, /*signalValue=*/0,
                                   NCCL_NET_SIGNAL_OP_INC, &psReq));

        EXPECT_TRUE(PollUntilDone(putReq));
        EXPECT_TRUE(PollUntilDone(getReq));
        EXPECT_TRUE(PollUntilDone(psReq));

        // iget result is visible locally on rank 0 once the request completes.
        EXPECT_TRUE(VerifyBuf(getBuf, kSize, /*seed=*/0x20));
    }
    Barrier();

    if(worldRank_ == 1)
    {
        EXPECT_TRUE(VerifyBuf(putRecvBuf, kSize, /*seed=*/0x10));
        EXPECT_TRUE(VerifyBuf(psRecvBuf,  kSize, /*seed=*/0x30));
        EXPECT_EQ(ReadSignal(sigBuf), 1u);
    }
}

TEST_P(GinMPIFixedSizeTest, IPutSignalZeroSize)
{
    if(!SetUpFixture(2, 2))
    {
        return;
    }

    constexpr size_t  kPayload  = 4096;
    constexpr uint8_t kSentinel = 0xCC;

    void* sendBuf = AllocBuf(kPayload);
    void* recvBuf = AllocBuf(kPayload);
    void* sigBuf  = AllocBuf(kSignalSize);
    ASSERT_NE(sendBuf, nullptr);
    ASSERT_NE(recvBuf, nullptr);
    ASSERT_NE(sigBuf,  nullptr);

    FillSentinel(recvBuf, kPayload, kSentinel);

    void *sendMh, *sendGh, *recvMh, *recvGh, *sigMh, *sigGh;
    ASSERT_EQ(ncclSuccess, RegMr(sendBuf, kPayload,    &sendMh, &sendGh));
    ASSERT_EQ(ncclSuccess, RegMr(recvBuf, kPayload,    &recvMh, &recvGh));
    ASSERT_EQ(ncclSuccess, RegMr(sigBuf,  kSignalSize, &sigMh,  &sigGh));

    Barrier();
    if(worldRank_ == 0)
    {
        void* req = nullptr;
        ASSERT_EQ(ncclSuccess,
                  gin_->iputSignal(ginCtx_, /*context=*/0,
                                   0, sendMh, /*size=*/0,
                                   0, recvMh, /*peerRank=*/1,
                                   /*signalOff=*/0, sigMh,
                                   /*signalValue=*/0,
                                   NCCL_NET_SIGNAL_OP_INC,
                                   &req));
        ASSERT_TRUE(PollUntilDone(req));
    }
    Barrier();

    if(worldRank_ == 1)
    {
        EXPECT_TRUE(AllSentinel(recvBuf, kPayload, kSentinel))
            << "recvBuf was mutated by zero-size iputSignal";
        EXPECT_EQ(ReadSignal(sigBuf), 1u);
    }
}

TEST_P(GinMPIFixedSizeTest, IPutSignalInvalidSignalOp)
{
    if(!SetUpFixture(2, 2))
    {
        return;
    }

    constexpr size_t kPayload = 4096;

    void* sendBuf = AllocBuf(kPayload);
    void* recvBuf = AllocBuf(kPayload);
    void* sigBuf  = AllocBuf(kSignalSize);
    ASSERT_NE(sendBuf, nullptr);
    ASSERT_NE(recvBuf, nullptr);
    ASSERT_NE(sigBuf,  nullptr);

    void *sendMh, *sendGh, *recvMh, *recvGh, *sigMh, *sigGh;
    ASSERT_EQ(ncclSuccess, RegMr(sendBuf, kPayload,    &sendMh, &sendGh));
    ASSERT_EQ(ncclSuccess, RegMr(recvBuf, kPayload,    &recvMh, &recvGh));
    ASSERT_EQ(ncclSuccess, RegMr(sigBuf,  kSignalSize, &sigMh,  &sigGh));

    Barrier();
    if(worldRank_ == 0)
    {
        void*              req      = nullptr;
        constexpr uint32_t kBogusOp = 0x99;
        ncclResult_t       r        = gin_->iputSignal(ginCtx_, /*context=*/0,
                                                       0, sendMh, kPayload,
                                                       0, recvMh, 1,
                                                       0, sigMh, /*signalValue=*/1,
                                                       kBogusOp, &req);
        EXPECT_EQ(r, ncclInvalidArgument)
            << "iputSignal accepted invalid signalOp 0x99 (returned " << r << ")";
    }
    Barrier();
}

// ===========================================================================
// GinMPIStressTest — non-parameterized stress fixture
// ===========================================================================

// ---------------------------------------------------------------------------
// IPutSignalStress10k
//   10000 iterations of iputSignal(INC), pipelined with a sliding window of
//   16 inflight requests. The signal counter on the receiver acts as the
//   correctness oracle: it MUST equal kIterations after all ops drain — if
//   even one op was lost on the wire, the counter under-counts.
//
//   Stresses (vs functional tests):
//     - Sustained CQ draining concurrent with WR posting
//     - Request slot recycling (10000 / 16 = 625 reuse cycles)
//     - GFD ring management under continuous pressure
//     - Cumulative state correctness (one drop = test failure)
//
//   Payload is small (256 B) and constant across iterations to maximize
//   ops/sec — we are stressing the post/complete machinery, not bandwidth.
//   Expected runtime: well under 1 s on production hardware.
// ---------------------------------------------------------------------------
TEST_F(GinMPIStressTest, IPutSignalStress10k)
{
    if(!SetUpFixture(/*minProcs=*/2, /*maxProcs=*/2)) return;

    constexpr int    kIterations = 10000;
    constexpr int    kInflight   = 16;
    constexpr size_t kPayload    = 256;

    void* sendBuf = AllocBuf(kPayload);
    void* recvBuf = AllocBuf(kPayload);
    void* sigBuf  = AllocBuf(kSignalSize);
    ASSERT_NE(sendBuf, nullptr);
    ASSERT_NE(recvBuf, nullptr);
    ASSERT_NE(sigBuf,  nullptr);

    if(worldRank_ == 0)
    {
        // Constant pattern across iterations; we are stressing the post/complete
        // machinery, not data correctness per-iteration. The signal counter
        // proves all 10k landed; the final payload check proves the last write
        // settled correctly.
        FillBuf(sendBuf, kPayload, /*seed=*/0x33);
    }

    void *sendMh, *sendGh, *recvMh, *recvGh, *sigMh, *sigGh;
    ASSERT_EQ(ncclSuccess, RegMr(sendBuf, kPayload,    &sendMh, &sendGh));
    ASSERT_EQ(ncclSuccess, RegMr(recvBuf, kPayload,    &recvMh, &recvGh));
    ASSERT_EQ(ncclSuccess, RegMr(sigBuf,  kSignalSize, &sigMh,  &sigGh));

    Barrier();

    if(worldRank_ == 0)
    {
        // Sliding window: inflight[next] holds either nullptr (slot free) or
        // a pending request. Before posting, drain whatever currently sits in
        // the slot we are about to overwrite.
        std::vector<void*> inflight(kInflight, nullptr);
        int                next = 0;

        for(int i = 0; i < kIterations; ++i)
        {
            if(inflight[next] != nullptr)
            {
                ASSERT_TRUE(PollUntilDone(inflight[next]))
                    << "iteration " << i << " (slot " << next << ") drain failed";
                inflight[next] = nullptr;
            }

            ASSERT_EQ(ncclSuccess,
                      gin_->iputSignal(ginCtx_, /*context=*/0,
                                       /*srcOff=*/0, sendMh, kPayload,
                                       /*dstOff=*/0, recvMh,
                                       /*peerRank=*/1,
                                       /*signalOff=*/0, sigMh,
                                       /*signalValue=*/0,
                                       NCCL_NET_SIGNAL_OP_INC,
                                       &inflight[next]))
                << "iteration " << i << " post failed";
            next = (next + 1) % kInflight;
        }

        // Final drain: any slot still holding a request must complete.
        for(int s = 0; s < kInflight; ++s)
        {
            if(inflight[s] != nullptr)
            {
                EXPECT_TRUE(PollUntilDone(inflight[s]))
                    << "final drain of slot " << s << " failed";
            }
        }
    }

    Barrier();

    if(worldRank_ == 1)
    {
        // PRIMARY ORACLE: signal counter must equal exactly kIterations.
        // counter < kIterations => some ops were lost
        // counter > kIterations => test bug or duplicate delivery (shouldn't happen with reliable IB)
        EXPECT_EQ(ReadSignal(sigBuf), static_cast<uint64_t>(kIterations))
            << "Signal counter mismatch — signal != " << kIterations
            << " means some iputSignal ops did not land on receiver";

        // SECONDARY: final payload bytes match what was sent. Every iteration
        // wrote the same 256 bytes, so the final state is the seed pattern.
        EXPECT_TRUE(VerifyBuf(recvBuf, kPayload, /*seed=*/0x33))
            << "Final payload mismatch — last successful write left wrong contents";
    }
}

namespace
{

// Pretty per-instance suffix: e.g. "Ctx2_64KiB_DmaBuf"
inline std::string CtxSizeName(const ::testing::TestParamInfo<std::tuple<int, size_t, bool>>& info)
{
    const int    nCtx    = std::get<0>(info.param);
    const size_t sz      = std::get<1>(info.param);
    const bool   dmaBuf  = std::get<2>(info.param);
    std::string  sizeStr;
    if(sz % (1024 * 1024) == 0)
    {
        sizeStr = std::to_string(sz / (1024 * 1024)) + "MiB";
    }
    else if(sz % 1024 == 0)
    {
        sizeStr = std::to_string(sz / 1024) + "KiB";
    }
    else
    {
        sizeStr = std::to_string(sz) + "B";
    }
    return "Ctx" + std::to_string(nCtx) + "_" + sizeStr
           + (dmaBuf ? "_DmaBuf" : "_RegMr");
}

inline std::string CtxOnlyName(const ::testing::TestParamInfo<std::tuple<int, bool>>& info)
{
    const int  nCtx   = std::get<0>(info.param);
    const bool dmaBuf = std::get<1>(info.param);
    return "Ctx" + std::to_string(nCtx)
           + (dmaBuf ? "_DmaBuf" : "_RegMr");
}

} // namespace

INSTANTIATE_TEST_SUITE_P(
    CtxAndSize,
    GinMPITest,
    ::testing::Combine(
        ::testing::Values(1, 2),
        ::testing::Values(static_cast<size_t>(4 * 1024),
                          static_cast<size_t>(64 * 1024),
                          static_cast<size_t>(1 * 1024 * 1024)),
        ::testing::Bool()),
    CtxSizeName);

INSTANTIATE_TEST_SUITE_P(
    CtxOnly,
    GinMPIFixedSizeTest,
    ::testing::Combine(
        ::testing::Values(1, 2),
        ::testing::Bool()),
    CtxOnlyName);

} // namespace RCCLGinTests

#else // !RCCL_HAS_GIN_IB_PROXY

#include <gtest/gtest.h>

TEST(GinMPITest, BuildSkipped)
{
    GTEST_SKIP() << "IB Proxy GIN backend not built into this binary. Skipping GIN tests...";
}

#endif // RCCL_HAS_GIN_IB_PROXY

#endif // MPI_TESTS_ENABLED
