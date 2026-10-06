/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifdef MPI_TESTS_ENABLED
#ifdef RCCL_HAS_RMA_IB_PROXY

#include "RmaMultiSegmentFixture.hpp"

namespace RCCLRmaTests
{


// DeepEP Engram pattern (DeepEP/csrc/kernels/backend/symmetric.hpp and
// DeepEP/csrc/kernels/elastic/engram.hpp): one symmetric VMM window contains a
// large GPU receive segment followed by an independently-sized CPU storage
// segment. An IGet reads from a non-zero offset in the remote CPU segment into a
// different non-zero offset in the local GPU segment. This specifically guards
// independent local and remote registration-relative offset tracking.
TEST_F(RmaMultiSegmentMPITest, DeepEP_EngramMixedWindowIGet)
{
    if (!SetUpFixture(2, 2)) return;

    constexpr size_t kGpuBytes   = 4 * kMiB;
    constexpr size_t kCpuBytes   = 2 * kMiB;
    constexpr size_t kRemoteOff  = kGpuBytes + 4096;
    constexpr size_t kLocalOff   = 64 * 1024;
    constexpr size_t kPayload    = 128 * 1024;
    constexpr uint8_t kSentinel  = 0xD7;

    MultiSegmentVmmBuffer* window = AllocDeepEpElastic(kGpuBytes, kCpuBytes);
    if (SyncSkip(window == nullptr))
        GTEST_SKIP() << "DeepEP-style GPU+CPU VMM allocation unavailable on this runtime";

    if (worldRank_ == 1)
        FillBuf(static_cast<uint8_t*>(window->ptr) + kRemoteOff, kPayload, /*seed=*/0x4D);
    if (worldRank_ == 0)
        FillSentinel(window->ptr, kGpuBytes, kSentinel);

    void *mh = nullptr, *gh = nullptr;
    ASSERT_EQ(ncclSuccess, RegMr(window->ptr, window->totalSize, &mh, &gh));
    if (!MultiSegmentPathAvailable())
        GTEST_SKIP() << "DeepEP window did not take the multi-segment RMA registration path";

    Barrier();
    if (worldRank_ == 0)
    {
        void* req = nullptr;
        ASSERT_EQ(ncclSuccess,
                  rma_->iget(rmaCtx_, 0,
                             /*remoteOff=*/kRemoteOff, mh, kPayload,
                             /*localOff=*/kLocalOff, mh, /*peerRank=*/1, ncclRmaOptFlagsDefault, &req));
        ASSERT_TRUE(PollUntilDone(req));
        ExpectPayloadIsolated(window->ptr, kGpuBytes, kLocalOff, kPayload,
                              /*seed=*/0x4D, kSentinel,
                              "DeepEP CPU-to-GPU IGet");
    }
    Barrier();
}

// Multi-node stress form of the DeepEP Engram fetch pattern. The registered
// window is [GPU receive area][CPU Engram storage] at DeepEP's 2 MiB alignment.
// Each IGet reads a changing non-zero remote CPU offset into an unrelated local
// GPU offset, while sentinels ensure the GPU receive area is not over-written.
TEST_F(RmaMultiSegmentMPITest, DeepEP_MultiNodeEngramMixedWindowIGetStress)
{
    if (!SetUpFixture(2, 2)) return;
    if (MPIEnvironment::cached_multi_node_result != 1)
        GTEST_SKIP() << "requires exactly one rank on each of two nodes";

    constexpr size_t kGpuBytes  = 8 * kMiB;
    constexpr size_t kCpuBytes  = 4 * kMiB;
    constexpr int    kIterations = 32;
    constexpr uint8_t kSentinel = 0xD9;
    const std::vector<size_t> payloadSizes = {
        size_t{1}, size_t{63}, size_t{4095}, size_t{4096},
        size_t{65535}, size_t{65536}, size_t{131072}, size_t{262144}
    };

    MultiSegmentVmmBuffer* window = AllocDeepEpElastic(kGpuBytes, kCpuBytes);
    if (SyncSkip(window == nullptr))
        GTEST_SKIP() << "DeepEP-style GPU+CPU VMM allocation unavailable on this runtime";

    void *mh = nullptr, *gh = nullptr;
    ASSERT_EQ(ncclSuccess, RegMr(window->ptr, window->totalSize, &mh, &gh));
    if (!MultiSegmentPathAvailable())
        GTEST_SKIP() << "DeepEP window did not take the multi-segment RMA registration path";

    for (int i = 0; i < kIterations; ++i)
    {
        const size_t len = payloadSizes[static_cast<size_t>(i) % payloadSizes.size()];
        const size_t remoteSpan = kCpuBytes - len - 4096;
        const size_t localSpan = kGpuBytes - len - 65536;
        const size_t remoteOff = kGpuBytes + 4096 +
                                 (static_cast<size_t>(i) * 131071) % remoteSpan;
        const size_t localOff = 65536 +
                                (static_cast<size_t>(i) * 65537) % localSpan;
        const uint8_t seed = static_cast<uint8_t>(0x40 + i);

        if (worldRank_ == 1)
            FillBuf(static_cast<uint8_t*>(window->ptr) + remoteOff, len, seed);
        if (worldRank_ == 0)
            FillSentinel(window->ptr, kGpuBytes, kSentinel);

        Barrier();
        bool getOk = true;
        if (worldRank_ == 0)
        {
            void* req = nullptr;
            getOk = rma_->iget(rmaCtx_, 0, remoteOff, mh, len,
                               localOff, mh, /*peerRank=*/1, ncclRmaOptFlagsDefault, &req) == ncclSuccess;
            if (getOk) getOk = PollUntilDone(req);
        }
        ASSERT_TRUE(MPIHelpers::allRanksTrue(getOk))
            << "DeepEP Engram IGet failed at iteration " << i;
        if (worldRank_ == 0)
            ExpectPayloadIsolated(window->ptr, kGpuBytes, localOff, len,
                                  seed, kSentinel,
                                  "DeepEP Engram iteration " + std::to_string(i));
        Barrier();
    }
}

// DeepEP HybridElasticSymmetricMemory:
// [GPU][CPU local-rank 0]...[CPU local-rank 3]. Each process imports the same
// local CPU handles before registration. Fetch from the matching CPU segment
// on the other node into a non-zero local GPU offset.
TEST_F(RmaMultiSegmentMPITest, DeepEP_HybridImportedCpuSegmentIGet)
{
    if (!SetUpFixture(/*minProcesses=*/8, /*maxProcesses=*/8,
                      /*minNodes=*/2, /*maxNodes=*/2))
        GTEST_SKIP() << "requires exactly 8 ranks across 2 nodes";

    constexpr size_t kGpuBytes = 8 * kMiB;
    constexpr size_t kCpuBytes = 2 * kMiB;
    constexpr size_t kPayload = 128 * 1024;
    constexpr size_t kLocalOff = 64 * 1024;
    constexpr uint8_t kSentinel = 0xB7;

    std::string reason;
    RCCLHybridVmmTests::HybridVmmBuffer* window = nullptr;
    if (!AllocHybridForLocalRanks(
            kGpuBytes, kCpuBytes, /*expectedLocalRanks=*/4, &window, &reason))
        GTEST_SKIP() << "DeepEP hybrid allocation unavailable: " << reason;

    const size_t remoteOff = kGpuBytes + static_cast<size_t>(window->localRank) * kCpuBytes + 4096;
    const int peer = MPIHelpers::findRemotePeerForLocalRank(window->localRank);
    ASSERT_TRUE(MPIHelpers::allRanksTrue(peer >= 0)) << "no remote peer for local rank " << window->localRank;

    FillBuf(static_cast<uint8_t*>(window->ptr) + remoteOff, kPayload,
            static_cast<uint8_t>(0x30 + worldRank_));
    FillSentinel(window->ptr, kGpuBytes, kSentinel);

    void *mh = nullptr, *gh = nullptr;
    ASSERT_EQ(ncclSuccess, RegMr(window->ptr, window->totalSize, &mh, &gh));
    if (!MultiSegmentPathAvailable())
        GTEST_SKIP() << "hybrid window did not take the multi-segment RMA path";

    Barrier();
    void* req = nullptr;
    bool getOk = rma_->iget(rmaCtx_, 0, remoteOff, mh, kPayload,
                            kLocalOff, mh, peer, ncclRmaOptFlagsDefault, &req) == ncclSuccess;
    if (getOk) getOk = PollUntilDone(req);
    ASSERT_TRUE(MPIHelpers::allRanksTrue(getOk)) << "hybrid IGet failed on at least one rank";
    ExpectPayloadIsolated(window->ptr, kGpuBytes, kLocalOff, kPayload,
                          static_cast<uint8_t>(0x30 + peer), kSentinel,
                          "DeepEP hybrid imported CPU IGet");
    Barrier();
}

// Multi-node hybrid stress: alternate the remote node's imported CPU-owner
// segment while varying source/destination offsets and transfer sizes. This
// exercises segment-window selection, shared-handle lifetime, and independent
// local/remote cursors with sentinel protection.
TEST_F(RmaMultiSegmentMPITest, DeepEP_HybridMultiNodeIGetStress)
{
    if (!SetUpFixture(/*minProcesses=*/8, /*maxProcesses=*/8,
                      /*minNodes=*/2, /*maxNodes=*/2))
        GTEST_SKIP() << "requires exactly 8 ranks across 2 nodes";

    constexpr size_t kGpuBytes = 8 * kMiB;
    constexpr size_t kCpuBytes = 2 * kMiB;
    constexpr int kIterations = 32;
    constexpr uint8_t kSentinel = 0xBC;
    const std::vector<size_t> sizes = {
        size_t{1}, size_t{63}, size_t{4095}, size_t{4096},
        size_t{65535}, size_t{65536}, size_t{131072}, size_t{262144}
    };

    std::string reason;
    RCCLHybridVmmTests::HybridVmmBuffer* window = nullptr;
    if (!AllocHybridForLocalRanks(
            kGpuBytes, kCpuBytes, /*expectedLocalRanks=*/4, &window, &reason))
        GTEST_SKIP() << "DeepEP hybrid allocation unavailable: " << reason;

    void *mh = nullptr, *gh = nullptr;
    ASSERT_EQ(ncclSuccess, RegMr(window->ptr, window->totalSize, &mh, &gh));
    if (!MultiSegmentPathAvailable())
        GTEST_SKIP() << "hybrid window did not take the multi-segment RMA path";

    for (int i = 0; i < kIterations; ++i)
    {
        const size_t len = sizes[static_cast<size_t>(i) % sizes.size()];
        const size_t sourceInnerOff = 4096 +
            (static_cast<size_t>(i) * 131071) % (kCpuBytes - len - 4096);
        const size_t ownCpuOff = kGpuBytes +
            static_cast<size_t>(window->localRank) * kCpuBytes + sourceInnerOff;
        const size_t localOff = 65536 +
            (static_cast<size_t>(i) * 65537) % (kGpuBytes - len - 65536);
        const uint8_t seed = static_cast<uint8_t>(0x40 + worldRank_ + i);

        FillBuf(static_cast<uint8_t*>(window->ptr) + ownCpuOff, len, seed);
        FillSentinel(window->ptr, kGpuBytes, kSentinel);
        Barrier();

        const int sourceLocalRank = (i + window->localRank) % window->localSize;
        const int peer = MPIHelpers::findRemotePeerForLocalRank(sourceLocalRank);
        ASSERT_TRUE(MPIHelpers::allRanksTrue(peer >= 0))
            << "no remote peer for local rank " << sourceLocalRank << " at iteration " << i;
        const size_t remoteOff = kGpuBytes +
            static_cast<size_t>(sourceLocalRank) * kCpuBytes + sourceInnerOff;

        void* req = nullptr;
        bool getOk = rma_->iget(rmaCtx_, 0, remoteOff, mh, len,
                                localOff, mh, peer, ncclRmaOptFlagsDefault, &req) == ncclSuccess;
        if (getOk) getOk = PollUntilDone(req);
        ASSERT_TRUE(MPIHelpers::allRanksTrue(getOk))
            << "hybrid IGet failed on at least one rank at iteration " << i;
        ExpectPayloadIsolated(window->ptr, kGpuBytes, localOff, len,
                              static_cast<uint8_t>(0x40 + peer + i),
                              kSentinel,
                              "hybrid IGet iteration " + std::to_string(i));
        Barrier();
    }
}

// Negative hybrid range guard: an IGet that overruns the final imported CPU
// segment must be rejected before posting and leave the GPU destination intact.
TEST_F(RmaMultiSegmentMPITest, DeepEP_HybridOutOfRangeIGetRejected)
{
    if (!SetUpFixture(/*minProcesses=*/8, /*maxProcesses=*/8,
                      /*minNodes=*/2, /*maxNodes=*/2))
        GTEST_SKIP() << "requires exactly 8 ranks across 2 nodes";

    constexpr size_t kGpuBytes = 8 * kMiB;
    constexpr size_t kCpuBytes = 2 * kMiB;
    constexpr uint8_t kSentinel = 0xC7;

    std::string reason;
    RCCLHybridVmmTests::HybridVmmBuffer* window = nullptr;
    if (!AllocHybridForLocalRanks(
            kGpuBytes, kCpuBytes, /*expectedLocalRanks=*/4, &window, &reason))
        GTEST_SKIP() << "DeepEP hybrid allocation unavailable: " << reason;

    void *mh = nullptr, *gh = nullptr;
    ASSERT_EQ(ncclSuccess, RegMr(window->ptr, window->totalSize, &mh, &gh));
    FillSentinel(window->ptr, kGpuBytes, kSentinel);
    Barrier();

    const int peer = MPIHelpers::findRemotePeerForLocalRank(window->localRank);
    ASSERT_TRUE(MPIHelpers::allRanksTrue(peer >= 0)) << "no remote peer for local rank " << window->localRank;
    void* req = nullptr;
    EXPECT_EQ(ncclInvalidArgument,
              rma_->iget(rmaCtx_, 0, window->totalSize - 32, mh, 64,
                         /*localOff=*/0, mh, peer, ncclRmaOptFlagsDefault, &req));
    EXPECT_EQ(req, nullptr);
    EXPECT_TRUE(AllSentinel(window->ptr, kGpuBytes, kSentinel));
    Barrier();
}

} // namespace RCCLRmaTests

#endif // RCCL_HAS_RMA_IB_PROXY
#endif // MPI_TESTS_ENABLED
